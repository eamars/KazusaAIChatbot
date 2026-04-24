# Kazusa AI Chatbot

Kazusa AI Chatbot is a platform-agnostic character chatbot "brain" built with FastAPI, LangGraph, MongoDB, and OpenAI-compatible chat and embedding APIs. It keeps per-user memory, channel history, reply context, and character state in MongoDB, then runs a staged persona pipeline to decide whether to respond, what to retrieve, and how to answer in character.

The core service is separate from platform adapters. Discord, the browser debug UI, and NapCat QQ all forward messages into the same `/chat` API.

## Current Architecture

```
┌──────────────────┐          ┌──────────────────┐
│ Discord Adapter  │──HTTP──▶│                   │
└──────────────────┘         │  Kazusa Brain     │       ┌──────────┐
┌──────────────────┐         │  (FastAPI)        │◀─────▶│ MongoDB  │
│ Debug Web UI     │──HTTP──▶│                   │       └──────────┘
└──────────────────┘         │  POST /chat       │       ┌──────────┐
┌──────────────────┐         │  GET  /health     │◀─────▶│ LLM API  │
│ QQ / WeChat / …  │──HTTP──▶│  POST /event      │       └──────────┘
└──────────────────┘         └────────┬──────────┘
                                      │
                              ┌───────▼────────┐
                              │ LangGraph      │
                              │ Pipeline       │
                              └───────┬────────┘
                                      │
                    ┌─────────────────────────────────────────┐
                    │         Persona Supervisor v2            │
                    │                                          │
                    │  Stage 0: Message Decontextualizer  ★    │
                    │  Stage 1: Research Subgraph     ★ × N    │
                    │  Stage 2: Cognition Subgraph    ★ × 3    │
                    │  Stage 3: Dialog Agent          ★ × 1-3  │
                    │  Stage 4: Consolidation         ★ × 3+   │
                    └─────────────────────────────────────────┘
```

### Service startup

When `kazusa_ai_chatbot.service:app` starts, it:

1. Runs `db_bootstrap()` to create collections, indexes, and seed the singleton character state document.
2. Loads the active character profile from MongoDB.
3. Compiles the top-level LangGraph pipeline.
4. Starts configured MCP servers.
5. Warm-starts the in-memory RAG cache from MongoDB.
6. Loads pending scheduled events if scheduling is enabled.

### Top-level graph

```text
START
  -> listen_only? END
  -> image attachments? multimedia_descriptor_agent
  -> relevance_agent
  -> should_respond? persona_supervisor2 : END
  -> END
```

Important details:

- `listen_only` now short-circuits before relevance and supervisor work. The message is still saved to MongoDB, but the graph does not think.
- Image attachments are described first by `multimedia_descriptor_agent`, and that description is appended to the text input before relevance and downstream stages.
- `/chat` saves the incoming user message before invoking the graph and saves the bot reply in a background task afterward.

### Persona supervisor

`persona_supervisor2` is the main staged pipeline:

```text
Stage 0  Message decontextualizer
Stage 1  RAG / research subgraph
Stage 2  Cognition subgraph
Stage 3  Dialog agent
Stage 4  Consolidation subgraph
```

`no_remember` skips Stage 4. `think_only` still runs the pipeline but suppresses the visible reply from the HTTP response.

## Request Lifecycle

1. `POST /chat` resolves `global_user_id`, loads `user_profile`, loads recent channel history, hydrates reply metadata, and builds `IMProcessState`.
2. The service writes the raw user message into `conversation_history`.
3. `relevance_agent` decides:
   - `should_respond`
   - `use_reply_feature`
   - `channel_topic`
   - `indirect_speech_context`
4. If the message should be answered, `persona_supervisor2` runs:
   - Stage 0 rewrites ambiguous references into `decontexualized_input`.
   - Stage 1 gathers memory, conversation, entity, and optional web context.
   - Stage 2 produces internal reasoning plus structured action directives.
   - Stage 3 generates the final dialog with a generator/evaluator loop.
   - Stage 4 persists the turn's effects into MongoDB and invalidates relevant cache entries.
5. The service returns `messages`, `should_reply`, and a placeholder `scheduled_followups` field.

Notes:

- `POST /event` currently accepts and logs events but does not dispatch real handlers yet.
- Response `attachments` are reserved for future multimodal output and are not currently populated by the brain.
- `scheduled_followups` is currently returned as `0`; actual future-promise scheduling happens inside the consolidator.

## Stage Breakdown

### Stage 0: Message decontextualizer

`persona_supervisor2_msg_decontexualizer.py` resolves ambiguous references using recent history, `channel_topic`, and `indirect_speech_context`, but it is deliberately conservative:

- It preserves literal anchors like URLs and filenames.
- It avoids changing already-complete inputs.
- It does not inject names from thin air.

### Stage 1: RAG / research

The current RAG stack is more than a simple memory lookup. It includes:

- Query embedding for the decontextualized input.
- Semantic cache probes against:
  - `objective_user_facts`
  - `character_diary`
  - `external_knowledge`
- A SHALLOW vs DEEP depth classifier in `rag/depth_classifier.py`.
- A resolution subgraph:
  - `continuation_resolver`
  - `rag_planner`
  - `entity_grounder`
- A retrieval subgraph that may call:
  - `memory_retriever_agent`
  - tier-2 third-party/entity retrieval nodes
  - `web_search_agent2`
- A bounded evaluator that can request one repair pass for newly revealed entities only.
- Write-through caching into the RAG cache layer plus a second boundary-key cache.

`research_facts` currently carries:

- `objective_facts`
- `user_image`
- `character_image`
- `input_context_results`
- `external_rag_results`
- `knowledge_base_results`
- `third_party_profile_results`
- `channel_recent_entity_results`
- `entity_resolution_notes`

`research_metadata` carries the trace for depth, cache hits, sources used, confidence, retrieval plans, and repair-pass outcomes.

### Stage 2: Cognition

The cognition subgraph is split across three layers:

- L1 subconscious:
  - emotional appraisal
  - interaction subtext
- L2 consciousness / boundary / judgment:
  - internal monologue
  - logical stance
  - character intent
- L3 and collector:
  - contextual agent
  - style agent
  - content-anchor agent
  - preference adapter
  - visual agent
  - collector

The output is a structured `action_directives` bundle with:

- `contextual_directives`
- `linguistic_directives`
- `visual_directives`

The preference adapter now treats reply language like any other accepted soft preference. Immediate accepted preferences and promises are meant to become authoritative through `user_profile.active_commitments`, not through a post-hoc rule layer.

### Stage 3: Dialog

`agents/dialog_agent.py` runs a generator/evaluator loop:

- The generator turns cognition output into one or more chat messages.
- The evaluator checks for hard failures such as topic drift, forbidden structure, or physical-action leakage.
- Retries are capped by `MAX_DIALOG_AGENT_RETRY`.

If `expression_willingness` comes back as `silent`, the dialog stage can intentionally emit no reply.

### Stage 4: Consolidation

The consolidator runs three branches, then commits through one writer:

- `global_state_updater`
- `relationship_recorder`
- `facts_harvester` with its own evaluator loop
- `db_writer`

The persistence layer currently updates:

- `character_state` mood, global vibe, reflection summary
- `user_profiles.character_diary`
- `user_profiles.objective_facts`
- `user_profiles.active_commitments`
- `user_profiles.user_image`
- `character_state.self_image`
- `memory` append-only fact and promise entries
- `scheduled_events` for due `future_promise` items
- `rag_cache_index` invalidation and `rag_metadata_index` version bumps

There is no separate `knowledge` collection anymore. Cross-session distilled topic knowledge is stored in the RAG cache layer under `cache_type="knowledge_base"`.

## Data Model

### Collections created by `db_bootstrap()`

| Collection | Purpose |
| --- | --- |
| `conversation_history` | Stored user and assistant messages plus embeddings |
| `user_profiles` | Identity mapping, diary, objective facts, commitments, affinity, user image |
| `character_state` | Singleton global character profile plus runtime state |
| `memory` | Append-only long-term memories and promises |
| `scheduled_events` | Pending future events |
| `rag_cache_index` | Persistent write-through cache entries with TTL |
| `rag_metadata_index` | Per-user RAG version metadata |

### `user_profiles`

The current authoritative fields are:

- `platform_accounts`
- `character_diary`
- `objective_facts`
- `active_commitments`
- `user_image`
- `affinity`
- `last_relationship_insight`

Legacy flat `facts` may still appear on older documents after migration, but user profiles no longer store combined text embeddings. New code is centered on `character_diary` and `objective_facts`.

### `character_state`

`character_state` is a singleton document with `_id: "global"`. It stores both:

- personality/profile fields like `name`, `personality_brief`, `boundary_profile`, and `linguistic_texture_profile`
- runtime fields like `mood`, `global_vibe`, `reflection_summary`, and `self_image`

### `memory`

`memory` is append-only. Each entry carries:

- `memory_type`
- `source_kind`
- `confidence_note`
- `status`
- `expiry_timestamp`

Promises are stored here and also mirrored into `user_profiles.active_commitments` for the next-turn authoritative read path.

## Repository Layout

```text
src/
  adapters/
    debug_adapter.py
    discord_adapter.py
    napcat_qq_adapter.py
  kazusa_ai_chatbot/
    service.py
    config.py
    state.py
    scheduler.py
    utils.py
    mcp_client.py
    agents/
    db/
    nodes/
    rag/
  scripts/
    load_character_profile.py
    insert_memory.py
    search_conversation.py
    search_memory.py
personalities/
  kazusa.json
  asuna.json
  qingche.json
  example.json
tests/
scripts/
  inject_knowledge.py
```

## Local Development

### 1. Install the package

`requirements.txt` is no longer the canonical install source for this repo. Use the package metadata in `pyproject.toml`.

```bash
python -m venv venv
venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

### 2. Create `.env`

There is no `.env.example` in the repo right now, so create `.env` manually.

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=roleplay_bot

# Primary chat model
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=your-chat-model

# Embeddings
EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL=your-embedding-model

# Optional specialized models
SECONDARY_LLM_BASE_URL=http://localhost:1234/v1
SECONDARY_LLM_API_KEY=lm-studio
SECONDARY_LLM_MODEL=your-secondary-model
PREFERENCE_LLM_BASE_URL=http://localhost:1234/v1
PREFERENCE_LLM_API_KEY=lm-studio
PREFERENCE_LLM_MODEL=your-preference-model

# Brain service
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
BRAIN_EXECUTOR_COUNT=1
SCHEDULED_TASKS_ENABLED=true

# Optional MCP servers
MCP_SERVERS={"mcp-searxng":{"url":"http://localhost:4001/mcp"}}

# Adapter-specific
BRAIN_URL=http://localhost:8000
DISCORD_TOKEN=
NAPCAT_WS_URL=
NAPCAT_WS_TOKEN=
```

Notes:

- If you omit the secondary and preference model variables, they fall back to the primary model.
- The current web-search agent assumes an MCP server named `mcp-searxng` exposing `searxng_web_search` and `web_url_read`.

### 3. Start dependencies

You need:

- MongoDB
- an OpenAI-compatible chat completion endpoint
- an OpenAI-compatible embeddings endpoint

LM Studio works, but the code is not limited to LM Studio as long as the endpoints are OpenAI-compatible.

### 4. Load a character profile

The brain refuses to start until a character profile exists in MongoDB.

```bash
python -m scripts.load_character_profile personalities/kazusa.json
```

To overwrite an existing profile:

```bash
python -m scripts.load_character_profile personalities/kazusa.json --force
```

Use `personalities/kazusa.json` or `personalities/asuna.json` as the real schema reference. `personalities/example.json` is only a minimal skeleton and does not reflect every field the current prompts expect.

At minimum, a working profile should include:

- `name`
- `description`
- `gender`
- `age`
- `birthday`
- `backstory`
- `personality_brief`
- `boundary_profile`
- `linguistic_texture_profile`

### 5. Run the brain service

```bash
uvicorn kazusa_ai_chatbot.service:app --host 0.0.0.0 --port 8000
```

## Adapters

### Debug web UI

```bash
python -m adapters.debug_adapter --brain-url http://localhost:8000 --port 8080
```

Open `http://localhost:8080`.

The debug UI exposes per-message toggles for:

- `listen_only`
- `think_only`
- `no_remember`

### Discord

The Discord adapter reads `BRAIN_URL` and `DISCORD_TOKEN` from the environment.

```bash
python -m adapters.discord_adapter --channels 123456789012345678
```

- Listed channels are active.
- Non-listed guild channels become listen-only.
- DMs are always active.

### NapCat QQ

The NapCat adapter reads `BRAIN_URL`, `NAPCAT_WS_URL`, and `NAPCAT_WS_TOKEN` from the environment.

```bash
python -m adapters.napcat_qq_adapter --channels 987654321
```

- Listed groups are active.
- Non-listed groups become listen-only.
- Private chats are always active.

## HTTP API

### `GET /health`

Returns service health and Mongo reachability.

### `POST /chat`

Primary brain entrypoint.

Important request fields:

- `platform`
- `platform_channel_id`
- `platform_message_id`
- `platform_user_id`
- `platform_bot_id`
- `display_name`
- `channel_name`
- `content`
- `attachments`
- `reply_context`
- `debug_modes`

Current attachment behavior:

- inbound image attachments with inline base64 are supported
- image descriptions are generated before relevance
- output attachments are not wired yet

### `POST /event`

Currently a placeholder endpoint that accepts platform events and logs them.

## Testing

Useful test commands:

```bash
pytest -m "not live_db and not live_llm" -q
pytest -m live_llm -q
pytest -m live_db -q
```

The repository default in `pytest.ini` excludes `live_db`, but it does not exclude `live_llm`, so an unqualified `pytest` may still expect a reachable model backend.

## Current Notes

- The supported documented run path is local editable install plus `uvicorn`.
- `Dockerfile` and `docker-compose.yml` are present, but they still reference `requirements.txt`, which is no longer part of this repo, so they are not the canonical setup path today.
- Some older utility scripts still reflect pre-refactor DB helpers. The required provisioning script is `src/scripts/load_character_profile.py`.
