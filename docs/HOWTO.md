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

# Character and service behavior
CHARACTER_GLOBAL_USER_ID=00000000-0000-4000-8000-000000000001
CONVERSATION_HISTORY_LIMIT=10
BRAIN_EXECUTOR_COUNT=1
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

If the secondary and preference model variables are omitted, they fall back to
the primary model. The web-search helper expects an MCP server named
`mcp-searxng` exposing `searxng_web_search` and `web_url_read`.

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
5. Starts configured MCP servers.
6. Loads pending scheduled events if scheduling is enabled.

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

- The supported documented run path is local editable install plus `uvicorn`.
- `Dockerfile` and `docker-compose.yml` are present, but they still reference
  `requirements.txt`, so they are not the canonical setup path today.
- The required provisioning script is `src/scripts/load_character_profile.py`.
