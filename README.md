# Kazusa AI Chatbot

A platform-agnostic AI chatbot that responds in-character using a configurable personality (JSON), long-term user memory, channel conversation history, and **MCP tool calling** — built with **FastAPI**, **LangGraph**, **LM Studio** (OpenAI-compatible API), and **MongoDB**.

The brain runs as a standalone service; IM adapters (Discord, debug web UI, etc.) communicate via a simple HTTP API.

## Architecture

The bot uses a **multi-stage persona simulation pipeline**. When a message arrives, a **Relevance Agent** decides whether to respond, then a **Persona Supervisor** orchestrates a 5-stage cognitive pipeline — decontextualization, research, cognition, dialog generation, and consolidation — each stage backed by its own LLM subgraph. All generative LLM calls use native JSON structures for context passing.

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

### LangGraph Graph (compiled)

The top-level graph is two nodes with a conditional edge:

```
START → relevance_agent ─┬─ should_respond=true  → persona_supervisor2 → END
                         └─ should_respond=false → END
```

The Persona Supervisor internally builds a second `StateGraph` with 5 linear stages, each of which may contain nested subgraphs.

### State Passing (explicit contract)

State is passed in two layers:

1. **Top-level graph state**: `IMProcessState` (`state.py`)
2. **Persona supervisor internal state**: `GlobalPersonaState` (`nodes/persona_supervisor2_schema.py`)

`/chat` in `service.py` builds `IMProcessState`, then:

- `relevance_agent` reads the message + context and writes routing/topic fields.
- `persona_supervisor2` copies a selected subset of fields into `GlobalPersonaState` and runs Stage 0→4.
- Stage outputs are accumulated in `GlobalPersonaState` and returned to top-level as `final_dialog` + `future_promises`.

Top-level flow:

```text
ChatRequest (+ optional debug_modes)
  -> service.py builds IMProcessState
  -> relevance_agent(IMProcessState) mutates:
       should_respond, reason_to_respond, use_reply_feature, channel_topic, user_topic
  -> if should_respond == false: END
  -> if debug_modes.listen_only: END  (record data only, skip thinking)
  -> persona_supervisor2(IMProcessState)
       -> builds GlobalPersonaState (selected fields + debug_modes)
       -> stage_0 -> stage_1 -> stage_2 -> stage_3
       -> if debug_modes.no_remember: END  (skip consolidation)
       -> stage_4
       -> returns {final_dialog, future_promises}
  -> if debug_modes.think_only: suppress final_dialog in response
  -> ChatResponse(messages=final_dialog, should_reply=use_reply_feature, scheduled_followups=future_promises)
```

`IMProcessState` keys used by top-level routing and supervisor handoff:

```text
timestamp, platform, platform_user_id, global_user_id,
user_name, user_input, user_profile,
platform_bot_id, bot_name,
character_profile, character_state,
platform_channel_id, channel_name, chat_history,
should_respond, reason_to_respond, use_reply_feature,
channel_topic, user_topic,
final_dialog, future_promises
```

`persona_supervisor2` currently passes these keys from `IMProcessState` into `GlobalPersonaState`:

```text
character_state, character_profile,
timestamp, user_input,
platform, platform_user_id, global_user_id,
user_name, user_profile,
platform_bot_id,
chat_history, user_topic, channel_topic
```

## Debug Modes

The `/chat` endpoint accepts an optional `debug_modes` object in the request body to control pipeline behavior for testing and debugging. All three flags default to `false` and can be **compounded** (e.g., `think_only + no_remember`).

```json
{
  "debug_modes": {
    "listen_only": false,
    "think_only": false,
    "no_remember": false
  }
}
```

| Flag | Behavior | Implementation |
|------|----------|----------------|
| `listen_only` | Records the user message to DB but skips all LLM processing (persona pipeline). Relevance agent still runs. | Conditional edge after `relevance_agent` → `END` |
| `think_only` | Runs the full pipeline (including consolidation) but **suppresses** the dialog in the HTTP response. The bot's reply is still saved internally. | Response-level suppression in `service.py` |
| `no_remember` | Runs the full pipeline and returns dialog but **skips Stage 4** (consolidation). No mood/fact/affinity updates are persisted. | Conditional edge after `stage_3_action` → `END` |

### Adapter CLI flags

All adapters support `--listen-only`, `--think-only`, and `--no-remember` CLI flags that set the corresponding debug modes for all messages processed by that adapter instance.

```bash
# Discord adapter in listen-only mode
python -m adapters.discord_adapter --brain-url http://localhost:8000 --listen-only

# NapCat QQ adapter with no consolidation
python -m adapters.napcat_qq_adapter --no-remember
```

The **debug web UI** (`debug_adapter.py`) provides checkboxes in the header bar to toggle each mode per-message.

## Pipeline Stages

### Relevance Agent (LLM call)
The brain service (`service.py`) loads all context from MongoDB before invoking the graph:
- **Conversation history** — last N messages for the channel
- **User profile** — facts, affinity score, last relationship insight
- **Character state** — current mood, global vibe, reflection summary

The relevance agent then analyzes the message + context to decide if the bot should engage. It outputs `should_respond`, `use_reply_feature`, `channel_topic`, and `user_topic`.

**Reads from state**:
- `user_input`, `user_multimedia_input`, `chat_history`
- `user_profile`, `character_state`, `character_profile`
- `platform_bot_id`, `bot_name`, `user_name`, `channel_name`

**Writes to state**:
- `should_respond`
- `reason_to_respond`
- `use_reply_feature`
- `channel_topic`
- `user_topic`

If `should_respond: false`, the graph short-circuits to END — no further LLM calls.

### Persona Supervisor v2 — Stage 0: Message Decontextualizer (LLM call)
Clarifies ambiguous user input by resolving pronouns, references, and implicit context from chat history. For example, "I saw him yesterday" → "I saw John yesterday". Outputs `decontexualized_input` for downstream stages.

**Stage 0 input**: `user_input`, `chat_history`, `user_name`, `channel_topic`, `user_topic`

**Stage 0 output**: `decontexualized_input`

### Persona Supervisor v2 — Stage 1: Research Subgraph (LLM calls)
Dispatches research tasks to specialist agents based on the query nature:

| Agent | Description |
|-------|-------------|
| `memory_retriever_agent` | Searches conversation history, user facts, and persistent memory via MongoDB. Uses tool-calling with `search_user_facts`, `search_conversation`, `get_conversation`, `search_persistent_memory`. |
| `web_search_agent` | Searches the internet via MCP tools (e.g., SearXNG). Multi-turn LLM loop with planning, execution, and evaluation. |

An **evaluator** determines if the retrieved information is sufficient or if another research iteration is needed (up to `MAX_RESEARCH_AGENT_RETRY`).

**Stage 1 input**: `decontexualized_input`, `user_profile`, `chat_history`, retrieval context

**Stage 1 output**:
- `research_facts`
- `research_metadata`

### Persona Supervisor v2 — Stage 2: Cognition Subgraph (7+ LLM calls)
A 3-layer cognitive simulation split into modular files:

**Layer 1 — Subconscious** (`persona_supervisor2_cognition_l1.py`)
- Emotional appraisal and interaction subtext analysis

**Layer 2 — Consciousness / Boundary / Judgment** (`persona_supervisor2_cognition_l2.py`)
- Internal monologue, logical stance, character intent
- Boundary core — relationship boundary enforcement
- Judgment core — stance refinement and intent arbitration

**Layer 3 — Contextual / Linguistic / Visual + Collector** (`persona_supervisor2_cognition_l3.py`)
- Contextual agent — social distance, emotional intensity, relational dynamics
- Linguistic agent — rhetorical strategy, style, content anchors, forbidden phrases
- Visual agent — facial expression, body language, gaze direction
- Collector — assembles all layer outputs into structured `action_directives`

The main orchestrator (`persona_supervisor2_cognition.py`) runs all layers sequentially via `call_cognition_subgraph()`.

Outputs: `internal_monologue`, `action_directives`, `emotional_appraisal`, `character_intent`, `logical_stance`, `interaction_subtext`.

**Stage 2 input**: `decontexualized_input`, `research_facts`, `research_metadata`, `character_profile`, `character_state`, `user_profile`

**Stage 2 output**:
- `interaction_subtext`
- `emotional_appraisal`
- `character_intent`
- `logical_stance`
- `internal_monologue`
- `action_directives`

### Persona Supervisor v2 — Stage 3: Dialog Agent (1–3 LLM calls)
A generator-evaluator loop that produces the final in-character reply:

- **Generator** — converts cognition outputs into natural dialog, split into 1–2 message segments
- **Evaluator** — checks for fatal errors (logic contradictions, missing facts, structure violations) and soft guideline adherence. Dynamically relaxes thresholds on retries.

Loop runs up to `MAX_DIALOG_AGENT_RETRY` times. Outputs: `final_dialog` (list of message strings).

**Stage 3 input**: `internal_monologue`, `action_directives`, `chat_history`, `user_name`, `character_profile`, `user_profile`

**Stage 3 output**: `final_dialog`

### Persona Supervisor v2 — Stage 4: Consolidation Subgraph (3+ LLM calls)
Runs inline (not deferred) after dialog generation to persist the interaction's effects:

1. **Global State Updater** — updates mood, global vibe, and reflection summary
2. **Relationship Recorder** — generates diary entry, affinity delta (with non-linear scaling breakpoints), and relationship insight
3. **Facts Harvester** — extracts new user facts and future promises (with evaluator loop up to `MAX_FACT_HARVESTER_RETRY`)
4. **DB Writer** — persists all outputs to MongoDB (character state, user facts, affinity, relationship insight, memory)

**Stage 4 input**: `final_dialog`, `interaction_subtext`, `emotional_appraisal`, `character_intent`, `logical_stance`, `user_profile`, `character_state`

**Stage 4 output**:
- `mood`, `global_vibe`, `reflection_summary`
- `diary_entry`, `affinity_delta`, `last_relationship_insight`
- `new_facts`, `future_promises`

### MCP Tool Calling

Tools are provided by external **MCP servers** (Streamable HTTP). At startup, the bot connects to all configured servers and discovers available tools. Tool names are namespaced as `{server}__{tool}` internally, but the original name is sent to the MCP server.

Configured via the `MCP_SERVERS` environment variable (JSON string):

```json
{"mcp-searxng": {"url": "http://host:4001/mcp"}, "playwright": {"url": "http://host:8931/mcp"}}
```

## Project Structure

```
src/
  kazusa_ai_chatbot/                 # Brain service package
    service.py                       # FastAPI app — /chat, /health, /event routes
    scheduler.py                     # Async event scheduler (MongoDB-backed)
    config.py                        # env vars, affinity breakpoints, retry limits
    state.py                         # IMProcessState TypedDict
    db.py                            # MongoDB helpers + document schemas (MemoryDoc, build_memory_doc, etc.)
    mcp_client.py                    # McpManager — MCP server connections + tool execution
    utils.py                         # Shared helpers (JSON parsing, affinity, history trim)
    agents/
      dialog_agent.py                # Stage 3 — generator/evaluator dialog loop
      memory_retriever_agent.py      # Research agent — MongoDB memory/history/fact search
      web_search_agent2.py           # Research agent — web search via MCP tools
    nodes/
      relevance_agent.py             # Relevance gate (context analysis + should_respond)
      persona_supervisor2.py         # Top-level 5-stage persona orchestrator
      persona_supervisor2_schema.py  # GlobalPersonaState TypedDict
      persona_supervisor2_msg_decontexualizer.py  # Stage 0
      persona_supervisor2_rag.py                   # Stage 1
      persona_supervisor2_cognition.py            # Stage 2 — orchestrator
      persona_supervisor2_cognition_l1.py         # Stage 2 — L1 subconscious
      persona_supervisor2_cognition_l2.py         # Stage 2 — L2 consciousness/boundary/judgment
      persona_supervisor2_cognition_l3.py         # Stage 2 — L3 contextual/linguistic/visual + collector
      persona_supervisor2_consolidator.py         # Stage 4
  adapters/                          # IM adapters (outside brain package)
    discord_adapter.py               # Thin Discord→HTTP adapter
    debug_adapter.py                 # Browser-based debug chat UI
  scripts/                           # Standalone utility scripts
    load_character_profile.py        # Load personality JSON into MongoDB
    insert_memory.py                 # CLI tool to insert a memory entry
    search_memory.py                 # CLI tool to search memories
    search_user_facts.py             # CLI tool to search user facts
    search_conversation.py           # CLI tool to search conversation history
personalities/
  example.json                       # Template personality JSON
  kazusa.json                        # Default character profile
tests/
Dockerfile
docker-compose.yml
requirements.txt
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in:

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=kazusa_bot_core

# LLM (OpenAI-compatible)
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=your-model-name
EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL=your-embedding-model-name

# Brain service
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000

# Discord adapter (only needed if running the Discord adapter)
DISCORD_TOKEN=your_discord_bot_token

# Optional: MCP tool servers (JSON string)
MCP_SERVERS={"mcp-searxng": {"url": "http://host:4001/mcp"}}
```

### 3. LLM Server

Run an OpenAI-compatible API server (e.g., LM Studio, vLLM, Ollama) with:
- **Chat model** — used by all LLM stages (relevance, cognition, dialog, consolidation, research agents)
- **Embedding model** — used for semantic search over conversation history, user facts, and memory

### 4. MongoDB

The brain service runs `db_bootstrap()` on startup which automatically creates all required collections and indexes:
- **`conversation_history`** — chat messages with embeddings for semantic search
- **`user_profiles`** — per-user facts, affinity score, platform accounts, relationship insight
- **`character_state`** — single global document for mood, vibe, reflection summary, and the **character profile**
- **`memory`** — persistent memories with structured metadata and embeddings (see Memory Schema below)
- **`knowledge`** — detailed knowledge entries (links, documents, reference material)
- **`scheduled_events`** — future events (follow-up messages, etc.)

#### Memory Schema (`MemoryDoc`)

Each document in the `memory` collection has:

| Field | Type | Description |
|-------|------|-------------|
| `memory_name` | `str` | Descriptive title, e.g. `[EAMARS] Lives in Auckland` |
| `content` | `str` | Full memory content |
| `source_global_user_id` | `str` | UUID4 of the user who triggered this memory |
| `timestamp` | `str` | ISO-8601 UTC creation timestamp |
| `embedding` | `list[float]` | Dense vector for similarity search |
| `memory_type` | `str` | `fact` \| `promise` \| `impression` \| `narrative` \| `defense_rule` |
| `source_kind` | `str` | `conversation_extracted` \| `relationship_inferred` \| `reflection_inferred` \| `seeded_manual` \| `external_imported` |
| `confidence_note` | `str` | How downstream should treat this memory |
| `status` | `str` | `active` \| `fulfilled` \| `expired` \| `superseded` |
| `expiry_timestamp` | `str \| None` | ISO-8601 expiry or `None` (never expires) |

**Append-only**: `save_memory()` always inserts a new document (never overwrites). Deduplication and lifecycle management are handled at query time via `status` filtering.

**Embedding format**: Embeddings are generated from structured text:
```
type:{memory_type}
source:{source_kind}
title:{memory_name}
content:{content}
```

**Search filters**: `search_memory()` and the `search_persistent_memory` tool support optional filtering by `memory_type`, `source_kind`, `status`, `expiry_before`, and `expiry_after`.

Vector search indexes are created best-effort (requires MongoDB Atlas for full vector search).

### 5. Load a character profile

The character personality profile is stored in MongoDB (in the `character_state` collection, `_id: "global"`, fields at the top level alongside runtime state). **The brain service will refuse to start if no profile is loaded.**

Create a JSON file following the schema in `personalities/example.json`, then load it:

```bash
# First time (mandatory before starting the brain service)
python -m scripts.load_character_profile personalities/kazusa.json

# Re-load / overwrite an existing profile
python -m scripts.load_character_profile personalities/kazusa.json --force
```

The personality JSON should include at minimum:
- `name`, `description`, `gender`, `age`, `birthday`
- `personality_brief` with `logic`, `tempo`, `defense`, `quirks`, `taboos`, `mbti`

Keys prefixed with `_` (e.g., `_reference`) are ignored — use these for visual descriptions or other non-prompt data. See `kazusa.json` for a full example.

## Deployment

### Local Development

```bash
# 1. Start the brain service
uvicorn kazusa_ai_chatbot.service:app --host 0.0.0.0 --port 8000

# 2a. Debug adapter — browser chat at http://localhost:8080
python -m adapters.debug_adapter --brain-url http://localhost:8000 --port 8080

# 2b. Discord adapter
python -m adapters.discord_adapter --brain-url http://localhost:8000

# 2c. Discord adapter with channel filter
python -m adapters.discord_adapter --brain-url http://localhost:8000 --channels 123456789
```

### Docker Deployment

```bash
# Start brain + MongoDB
docker-compose up -d kazusa-brain mongo

# Optionally, uncomment and start adapters in docker-compose.yml:
# docker-compose up -d discord-adapter
# docker-compose up -d debug-adapter
```

The `docker-compose.yml` defines:
- **`kazusa-brain`** — the FastAPI service on port 8000
- **`mongo`** — MongoDB 7 with persistent volume
- **`discord-adapter`** (commented) — Discord client forwarding to brain
- **`debug-adapter`** (commented) — Web chat UI on port 8080

Environment variables are read from `.env` or can be set in the compose file.

## Design Decisions

- **5-stage cognitive pipeline** — separating decontextualization, research, cognition, dialog, and consolidation gives each stage a focused LLM context and allows independent iteration. The cognition layer (subconscious → conscious → social filter) models a human-like decision process rather than a single prompt.
- **Generator-evaluator dialog loop** — the dialog agent generates candidate replies and an evaluator checks for fatal errors (logic contradictions, missing facts). This catches issues before sending while avoiding excessive retries via dynamic threshold relaxation.
- **Inline consolidation** — unlike the previous deferred memory writer, consolidation now runs as part of the graph. This ensures character state, facts, and affinity are updated before the next message arrives.
- **Append-only memory** — `save_memory()` uses `insert_one` (never `update_one`), so every memory is a permanent record. Lifecycle is managed via the `status` field (`active` → `fulfilled` / `expired` / `superseded`). This prevents accidental overwrites and preserves the full history of extracted facts and promises.
- **Structured memory metadata** — each memory carries `memory_type`, `source_kind`, `confidence_note`, `status`, and `expiry_timestamp`. This enables filtered retrieval (e.g., only active promises for a specific user) and gives downstream consumers trust signals about how to weight each memory.
- **Structured embedding text** — memory embeddings encode `type`, `source`, `title`, and `content` as a structured block rather than a flat concatenation. This improves semantic search precision by allowing the embedding model to weight metadata alongside content.
- **Non-linear affinity scaling** — affinity deltas are scaled by breakpoints: easy to gain/lose at extremes, normal in the middle, harder at high levels. This prevents runaway affinity while allowing meaningful relationship progression.
- **Affinity system** — per-user 0–1000 score (default 500) with 21 behavioral tiers from "Contemptuous" to "Unwavering". The LLM proposes a delta; non-LLM code applies non-linear scaling and clamping.
- **Research subgraph with evaluator** — the research stage can dispatch multiple specialist agents and re-evaluate whether enough information has been gathered, preventing premature or insufficient research.
- **MCP for tooling** — tools are served by external MCP servers over Streamable HTTP, making them language-agnostic and independently deployable.
- **Brain + adapter separation** — the brain service is platform-agnostic; IM adapters are thin HTTP clients. This enables multi-platform support (Discord, QQ, WeChat) without modifying the cognitive pipeline.
- **Context loading in service** — conversation history, user profile, and character state are loaded once in the `/chat` handler before graph invocation, keeping nodes stateless and testable.
- **`_`-prefixed personality keys ignored** — personality JSON can store reference data (appearance, art notes) under `_reference` without wasting prompt tokens.
- **DB bootstrap on startup** — all collections and indexes are verified/created at service start, so deployment requires zero manual setup beyond the environment variables.
- **Character profile in MongoDB** — the personality profile is stored at the top level of the `character_state` collection’s `_id: "global"` document, alongside runtime state fields (mood, global_vibe, etc.). This decouples the brain service from the filesystem and enables future consolidator-driven profile evolution. The service crashes on startup if no profile is found, enforcing a mandatory one-time load via `scripts.load_character_profile`.
