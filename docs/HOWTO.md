# Kazusa AI Chatbot HOWTO

This document keeps setup, operations, API shape, and test commands out of the
project README while preserving the practical details needed to run the brain.

## Local Setup

Install the package from `pyproject.toml`:

```bash
python -m venv venv
venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

Create a local `.env` file:

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=roleplay_bot

# Route-specific chat models
RELEVANCE_AGENT_LLM_BASE_URL=http://localhost:1234/v1
RELEVANCE_AGENT_LLM_API_KEY=lm-studio
RELEVANCE_AGENT_LLM_MODEL=your-chat-model
VISION_DESCRIPTOR_LLM_BASE_URL=http://localhost:1234/v1
VISION_DESCRIPTOR_LLM_API_KEY=lm-studio
VISION_DESCRIPTOR_LLM_MODEL=your-chat-model
MSG_DECONTEXTUALIZER_LLM_BASE_URL=http://localhost:1234/v1
MSG_DECONTEXTUALIZER_LLM_API_KEY=lm-studio
MSG_DECONTEXTUALIZER_LLM_MODEL=your-chat-model
RAG_PLANNER_LLM_BASE_URL=http://localhost:1234/v1
RAG_PLANNER_LLM_API_KEY=lm-studio
RAG_PLANNER_LLM_MODEL=your-chat-model
RAG_SUBAGENT_LLM_BASE_URL=http://localhost:1234/v1
RAG_SUBAGENT_LLM_API_KEY=lm-studio
RAG_SUBAGENT_LLM_MODEL=your-chat-model
WEB_SEARCH_LLM_BASE_URL=http://localhost:1234/v1
WEB_SEARCH_LLM_API_KEY=lm-studio
WEB_SEARCH_LLM_MODEL=your-chat-model
COGNITION_LLM_BASE_URL=http://localhost:1234/v1
COGNITION_LLM_API_KEY=lm-studio
COGNITION_LLM_MODEL=your-chat-model
DIALOG_GENERATOR_LLM_BASE_URL=http://localhost:1234/v1
DIALOG_GENERATOR_LLM_API_KEY=lm-studio
DIALOG_GENERATOR_LLM_MODEL=your-chat-model
DIALOG_EVALUATOR_LLM_BASE_URL=http://localhost:1234/v1
DIALOG_EVALUATOR_LLM_API_KEY=lm-studio
DIALOG_EVALUATOR_LLM_MODEL=your-chat-model
CONSOLIDATION_LLM_BASE_URL=http://localhost:1234/v1
CONSOLIDATION_LLM_API_KEY=lm-studio
CONSOLIDATION_LLM_MODEL=your-chat-model
JSON_REPAIR_LLM_BASE_URL=http://localhost:1234/v1
JSON_REPAIR_LLM_API_KEY=lm-studio
JSON_REPAIR_LLM_MODEL=your-chat-model

# Embeddings
EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_API_KEY=lm-studio
EMBEDDING_MODEL=your-embedding-model

# Character and service behavior
CHARACTER_GLOBAL_USER_ID=00000000-0000-4000-8000-000000000001
CONVERSATION_HISTORY_LIMIT=10
SCHEDULED_TASKS_ENABLED=true

# MCP servers and timeouts
MCP_SERVERS={"mcp-searxng":{"url":"http://localhost:4001/mcp"}}
MCP_CALL_TIMEOUT=30
MCP_CONNECT_TIMEOUT=10

# Agent retry limits
MAX_MEMORY_RETRIEVER_AGENT_RETRY=2
MAX_WEB_SEARCH_AGENT_RETRY=2
MAX_DIALOG_AGENT_RETRY=3
MAX_FACT_HARVESTER_RETRY=3

# Cache2
RAG_CACHE2_MAX_ENTRIES=5000

# Reflection cycle
REFLECTION_CYCLE_DISABLED=false
REFLECTION_CONTEXT_ENABLED=false
REFLECTION_WORKER_INTERVAL_SECONDS=900
REFLECTION_HOURLY_SLOTS_PER_TICK=3
REFLECTION_DAILY_RUN_AFTER_LOCAL_TIME=04:30
REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME=05:00

# Persistent profile-memory policy
PROFILE_MEMORY_DIARY_TTL_SECONDS=7776000
PROFILE_MEMORY_FACT_TTL_SECONDS=31536000
PROFILE_MEMORY_MILESTONE_TTL_SECONDS=94608000
PROFILE_MEMORY_COMMITMENT_TTL_SECONDS=864000
PROFILE_MEMORY_RECENT_DIARY_LIMIT=6
PROFILE_MEMORY_RECENT_FACT_LIMIT=8
PROFILE_MEMORY_RECENT_MILESTONE_LIMIT=10
PROFILE_MEMORY_DIARY_SEMANTIC_THRESHOLD=0.75
PROFILE_MEMORY_FACT_SEMANTIC_THRESHOLD=0.72
PROFILE_MEMORY_MILESTONE_SEMANTIC_THRESHOLD=0.72
PROFILE_MEMORY_BUDGET=40

# Affinity
AFFINITY_RAW_DEAD_ZONE=1

# Adapter-specific
BRAIN_URL=http://localhost:8000
BRAIN_RESPONSE_TIMEOUT=120
DISCORD_TOKEN=
NAPCAT_WS_URL=
NAPCAT_WS_TOKEN=
ADAPTER_RUNTIME_HOST=127.0.0.1
ADAPTER_RUNTIME_PUBLIC_URL=
ADAPTER_HEARTBEAT_SECONDS=30
ADAPTER_RUNTIME_SHARED_SECRET=
DISCORD_RUNTIME_PORT=8012
NAPCAT_RUNTIME_PORT=8011
```

All route-specific chat model variables are required. The generic
`LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` variables are not used after the
route migration. Missing route variables crash config loading instead of
silently falling back to another endpoint. The web-search helper expects an MCP
server named `mcp-searxng` exposing `searxng_web_search` and `web_url_read`.

## Dependencies

You need:

- MongoDB
- an OpenAI-compatible chat completion endpoint
- an OpenAI-compatible embeddings endpoint
- optional MCP servers for web search

LM Studio works for local model hosting, but any OpenAI-compatible endpoint can
be used.

## Character Profile

The brain refuses to start until a character profile exists in MongoDB.

```bash
python -m scripts.load_character_profile personalities/kazusa.json
```

To overwrite an existing profile:

```bash
python -m scripts.load_character_profile personalities/kazusa.json --force
```

Use `personalities/example.json` as a compact template, and
`personalities/kazusa.json` or `personalities/asuna.json` as practical filled
references.

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

## Run The Brain Service

```bash
uvicorn kazusa_ai_chatbot.service:app --host 0.0.0.0 --port 8000
```

On startup the service:

1. Runs `db_bootstrap()` to create current collections and indexes.
2. Drops legacy `rag_cache_index` and `rag_metadata_index` collections if they
   are still present.
3. Loads the active character profile.
4. Compiles the top-level LangGraph pipeline.
5. Hydrates persistent Cache2 initializer entries.
6. Starts configured MCP servers.
7. Loads pending scheduled events if scheduling is enabled.
8. Starts the reflection worker unless `REFLECTION_CYCLE_DISABLED=true`.

## Adapters

### Debug Web UI

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

Listed channels are active, non-listed guild channels become listen-only, and
DMs are always active.

### NapCat QQ

The NapCat adapter reads `BRAIN_URL`, `NAPCAT_WS_URL`, and `NAPCAT_WS_TOKEN`
from the environment.

```bash
python -m adapters.napcat_qq_adapter --channels 987654321
```

Listed groups are active, non-listed groups become listen-only, and private
chats are always active.

## HTTP API

### `GET /health`

Returns service health and Mongo reachability.

The response also includes sanitized Cache2 agent-level lookup stats for
display surfaces:

```json
{
  "status": "ok",
  "db": true,
  "scheduler": true,
  "cache2": {
    "agents": [
      {
        "agent_name": "user_profile_agent",
        "hit_count": 8,
        "miss_count": 2,
        "hit_rate": 0.8
      }
    ]
  }
}
```

The Cache2 block intentionally exposes only agent names and aggregate lookup
counts. It does not include cache keys, user identifiers, queries, dependency
scopes, or cached retrieval results.

### `POST /chat`

Primary brain entrypoint.

The endpoint enqueues each request into the brain's process-local input queue
and waits for the queued item's response. The queue worker processes one
surviving item at a time. When bursts grow past the queue thresholds, plain
unaddressed messages are pruned before relevance/RAG; pruned messages are still
saved to `conversation_history` and return an empty `ChatResponse`.

Adapters own platform-specific reply detection. The brain protects a queued
reply only when `reply_context.reply_to_current_bot` is `true`; a raw
`reply_to_message_id` alone is not enough.

After a surviving turn produces its user-facing reply, the worker awaits bot
message persistence, conversation-progress recording, consolidation, and the
resulting Cache2 invalidation before consuming the next queued chat item. This
keeps the next RAG pass from reading stale durable facts.

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

Useful drop-audit log line:

```text
Queued chat item dropped: sequence=... platform=... channel=... message=... user=... display_name=... tagged=... bot_reply=... content="..."
```

Current attachment behavior:

- inbound image attachments with inline base64 are supported
- image descriptions are generated before relevance
- output attachments are not wired yet

### `POST /event`

Currently a placeholder endpoint that accepts platform events and logs them.

## Runtime Data Model

Collections created or maintained by `db_bootstrap()`:

| Collection | Purpose |
| --- | --- |
| `conversation_history` | Stored user and assistant messages plus embeddings |
| `user_profiles` | Identity mapping, profile memory, affinity, commitments, user image |
| `character_state` | Singleton character profile, runtime mood/vibe, self image |
| `memory` | Append-only long-term fact and promise records |
| `scheduled_events` | Pending future events |

`db_bootstrap()` also drops the legacy RAG1 collections `rag_cache_index` and
`rag_metadata_index` if they are present.

## Legacy Collection Cleanup

Bootstrapping handles stale legacy collections automatically. There is also an
idempotent one-shot cleanup script for operations:

```bash
python scripts/drop_legacy_rag_collections.py
```

The script drops `rag_cache_index` and `rag_metadata_index` when present and is
safe to run repeatedly.

## Testing

Default test runs exclude live DB and live LLM tests through `pytest.ini`.

```bash
pytest -q
pytest -m "not live_db and not live_llm" -q
```

Live LLM tests must be run and inspected one at a time:

```bash
pytest tests/test_cognition_live_llm.py::test_live_msg_decontexualizer_returns_non_empty_output -q -s
```

Live DB tests can be run explicitly when MongoDB is available:

```bash
pytest -m live_db -q
```

Live LLM tests write inspection traces to `test_artifacts/llm_traces/`, which
is ignored by git.

## Current Notes

- The supported development run path is local editable install plus `uvicorn`.
- `Dockerfile` installs from `pyproject.toml`; `docker-compose.yml` remains a
  service-oriented deployment template that expects all required environment
  variables to be supplied.
- The required provisioning script is `src/scripts/load_character_profile.py`.
