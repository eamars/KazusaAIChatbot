# Kazusa AI Chatbot HOWTO

This document keeps setup, operations, and test commands out of the project
README while preserving the practical details needed to run the brain.

This operational guide covers local setup, service startup, adapter startup,
and test commands. Brain service request/response models, adapter obligations,
delivery receipts, runtime adapter registration, and reply hydration are owned
by the [Brain Service ICD](../src/kazusa_ai_chatbot/brain_service/README.md).
The typed message envelope contract lives in the
[Message Envelope ICD](../src/kazusa_ai_chatbot/message_envelope/README.md).

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
BOUNDARY_CORE_LLM_BASE_URL=http://localhost:1234/v1
BOUNDARY_CORE_LLM_API_KEY=lm-studio
BOUNDARY_CORE_LLM_MODEL=your-chat-model
BACKGROUND_ARTIFACT_LLM_BASE_URL=http://localhost:1234/v1
BACKGROUND_ARTIFACT_LLM_API_KEY=lm-studio
BACKGROUND_ARTIFACT_LLM_MODEL=your-chat-model
BACKGROUND_WORK_LLM_BASE_URL=http://localhost:1234/v1
BACKGROUND_WORK_LLM_API_KEY=lm-studio
BACKGROUND_WORK_LLM_MODEL=your-chat-model
DIALOG_GENERATOR_LLM_BASE_URL=http://localhost:1234/v1
DIALOG_GENERATOR_LLM_API_KEY=lm-studio
DIALOG_GENERATOR_LLM_MODEL=your-chat-model
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
AUDIT_LOG_TTL_DAYS=90
DEBUG_LOG_TTL_DAYS=14
LLM_TRACE_CAPTURE_MODE=metadata
CONVERSATION_HISTORY_LIMIT=10
COGNITION_VISUAL_DIRECTIVES_ENABLED=true
COGNITION_RESOLVER_MAX_CYCLES=3
COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS=120.0
SELF_COGNITION_ENABLED=true
CHARACTER_SLEEP_LOCAL_PERIOD=02:00-12:00

# Durable calendar scheduler
CALENDAR_SCHEDULER_ENABLED=true
CALENDAR_SCHEDULER_POLL_INTERVAL_SECONDS=30
CALENDAR_SCHEDULER_CLAIM_LIMIT=10
CALENDAR_SCHEDULER_LEASE_SECONDS=300
CALENDAR_SCHEDULER_MAX_ATTEMPTS=3
CALENDAR_SCHEDULER_PER_TRIGGER_CAPACITY=5

# Background work handoff
BACKGROUND_WORK_WORKER_ENABLED=true
BACKGROUND_WORK_WORKER_INTERVAL_SECONDS=15
BACKGROUND_WORK_WORKER_CLAIM_LIMIT=2
BACKGROUND_WORK_WORKER_LEASE_SECONDS=300
BACKGROUND_WORK_WORKER_MAX_ATTEMPTS=3
BACKGROUND_WORK_INPUT_CHAR_LIMIT=12000
BACKGROUND_WORK_OUTPUT_CHAR_LIMIT=3000

# Direct web search and URL-reader behavior
SEARXNG_URL=http://your-searxng-host:8080
SEARXNG_SEARCH_TIMEOUT_SECONDS=30
SEARXNG_SEARCH_RESULT_LIMIT=10
WEB_URL_READ_TIMEOUT_SECONDS=30
WEB_URL_READ_MAX_BYTES=1048576
WEB_URL_READ_MAX_CHARS=10000
WEB_URL_READ_REDIRECT_LIMIT=5
WEB_URL_READER_USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36
WEB_URL_READER_ACCEPT_LANGUAGE=en-US,en;q=0.9

# Optional generic MCP servers and timeouts
MCP_SERVERS={}
MCP_CALL_TIMEOUT=30
MCP_CONNECT_TIMEOUT=10

# Agent retry limits
MAX_MEMORY_RETRIEVER_AGENT_RETRY=2
MAX_WEB_SEARCH_AGENT_RETRY=2
MAX_FACT_HARVESTER_RETRY=3

# Cache2
RAG_CACHE2_MAX_ENTRIES=5000

# Reflection cycle
REFLECTION_CYCLE_ENABLED=true
REFLECTION_WORKER_INTERVAL_SECONDS=900
REFLECTION_HOURLY_SLOTS_PER_TICK=3
REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS=60
REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD=3
REFLECTION_DAILY_RUN_AFTER_LOCAL_TIME=04:30
REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME=05:00
GLOBAL_CHARACTER_GROWTH_PASS_ENABLED=true
GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET=32000

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

All route-specific chat model variables are required except the background
artifact and background-work routes. The background artifact route falls back
to the cognition route when omitted; the background-work route falls back to
the background artifact route when omitted.
Route-specific variables replace the retired generic `LLM_BASE_URL`,
`LLM_API_KEY`, and `LLM_MODEL` settings. Missing required route variables stop
config loading. Chat routes also accept route-specific
`*_MAX_COMPLETION_TOKENS` and `*_THINKING_ENABLED` values, with
`DEFAULT_LLM_MAX_COMPLETION_TOKENS` as the shared completion budget default.
Thinking is a boolean route toggle and defaults to disabled. When enabled, the
LLM interface currently maps provider-specific thinking controls for Gemma 4,
Qwen3-family model names, and Qwen-compatible Qwopus 3.x model names.

`CHARACTER_GLOBAL_USER_ID` defaults to
`00000000-0000-4000-8000-000000000001`. Set it explicitly in production so the
active character keeps a stable first-class identity across service runs.

`SEARXNG_URL` is optional. When it is empty, the web agent can still read
HTTP(S) URLs directly from the Kazusa process, but web search returns a bounded
unavailable observation. When it is set, search uses the configured SearXNG
`/search?format=json` endpoint directly. URL reads do not require SearXNG or
MCP, and local/private HTTP(S) resources reachable from the Kazusa process are
allowed by default. URL reads always use browser-navigation headers,
process-memory cookies, locally supported compression encodings, and common
HTTP anti-bot challenge detection. They do not execute JavaScript, solve
CAPTCHA, or impersonate browser TLS fingerprints. `MCP_SERVERS` remains
available for unrelated generic MCP tools.

When LM Studio reports
`The model has crashed without additional information.`, chat model calls made
through `LLInterface` retry the same request once. Calls for the same
`base_url` and model wait while that retry reloads the model; calls for other
models continue normally. Other 400 responses and non-unload errors are not
retried by this recovery path.

`COGNITION_VISUAL_DIRECTIVES_ENABLED` is a brain-service level switch. Set it
to `false` to skip L3 visual-directive generation globally; adapters and
debug-client request payloads do not control this behavior.

The live persona turn always runs the cognition-preserving resolver after
decontextualization. Each resolver cycle still runs the shared L1 -> L2 -> L2d
cognition stack. L2d may request bounded evidence, HIL, approval, or private
self-resolution capabilities before final surface selection.
`COGNITION_RESOLVER_MAX_CYCLES` caps recurrence, and
`COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS` bounds one capability
observation.

`SELF_COGNITION_ENABLED` defaults to `true`. Self-cognition-created episodes
disable visual directives by default with
`origin_metadata.debug_modes.no_visual_directives=true`, so normal
self-cognition worker runs do not invoke the L3 visual-directive LLM.
`CHARACTER_SLEEP_LOCAL_PERIOD` defaults to `02:00-12:00` in
`CHARACTER_TIME_ZONE`. During that local period, active-commitment
self-cognition and reflection-attached group self-cognition do not trigger.
Scheduled future cognition, durable calendar due-run handling, reflection,
consolidation, dispatcher validation, and adapter delivery continue. Set
`CHARACTER_SLEEP_LOCAL_PERIOD` to an empty value to disable this sleep-period
suppression.

`BACKGROUND_WORK_WORKER_ENABLED` controls the generic background-work runtime.
L2d may only queue a semantic `background_work_request`; the runtime router then
chooses a worker and task without receiving adapter targets, job refs, tool
arguments, or final visible text. The current text-artifact worker remains
text-only and does not write files, run shell commands, install packages,
download resources, research the web, process attachments, create images, or
send adapter text directly. Completed jobs re-enter the brain as
`background_work_result_ready` cognition, then use the existing dialog and
delivery boundary for any visible result. `BACKGROUND_ARTIFACT_*` settings are
legacy aliases only for compatibility.

Reflection phase scheduling spreads monitor-eligible channels across the
`REFLECTION_WORKER_INTERVAL_SECONDS` period instead of running all group
review work in one burst. `REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD` defaults to
the old `REFLECTION_HOURLY_SLOTS_PER_TICK` budget, and
`REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS` rejects configurations that cannot
fit the requested slots inside the period. Each phase slot reviews at most one
group activity window, and old windows for that group are coalesced into the
reviewed-window ledger instead of being caught up visibly.

## Dependencies

You need:

- MongoDB
- an OpenAI-compatible chat completion endpoint
- an OpenAI-compatible embeddings endpoint
- optional SearXNG service for web search
- optional generic MCP servers for unrelated tools

Direct URL reads use the existing HTTP client dependency and do not require an
additional browser transport or automation dependency.

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

Normal local operation starts the control console first. The console binds to
loopback by default, serves a static buildless HTML/CSS/JS UI, authenticates
the local operator, and starts registry-declared child services with argv
subprocesses only.

For a stable local login token, generate a local operator token hash before
startup. The plaintext token is what you type into the browser login field;
only the hash belongs in the environment.

```powershell
$env:KAZUSA_CONTROL_OPERATOR_TOKEN_HASH = venv\Scripts\python -c "from getpass import getpass; from control_console.auth import hash_operator_token; print(hash_operator_token(getpass('Operator token: ')))"
```

The hash format is `pbkdf2_sha256$<iterations>$<salt>$<digest>`. Hashes are
salted, so running the command twice for the same plaintext token produces
different environment values. The login endpoint verifies the plaintext token
against this hash, then issues an HTTP-only local session cookie and a CSRF
token used by state-changing console requests.

If `KAZUSA_CONTROL_OPERATOR_TOKEN_HASH` is not set, the console generates a
random ephemeral operator token during startup, hashes it in memory, and prints
the plaintext token once in the server log:

```text
Control console access token: <random-token>
```

That fallback token is valid only for the current console process. Restarting
the console generates a new token. This is convenient for local development,
but operators should still set `KAZUSA_CONTROL_OPERATOR_TOKEN_HASH` for a
stable runbook or shared workstation setup.

```bash
kazusa-control-console --host 127.0.0.1 --port 8765
```

Useful control-console environment variables:

```env
KAZUSA_CONTROL_OPERATOR_TOKEN_HASH=<pbkdf2 hash generated from the local operator token>
KAZUSA_CONTROL_STATE_DIR=.kazusa_control
KAZUSA_CONTROL_SERVICE_REGISTRY=
KAZUSA_CONTROL_BRAIN_BASE_URL=http://127.0.0.1:8000
```

The console manages only services declared in its registry. Built-in services
are `brain`, `adapter.discord`, `adapter.napcat`, and `adapter.debug`. It does
not adopt, inspect, or stop externally started processes.

Direct service startup remains available for development fallback:

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
7. Starts the durable calendar worker when `CALENDAR_SCHEDULER_ENABLED=true`.
8. Starts the self-cognition worker when `SELF_COGNITION_ENABLED=true`.
9. Starts the reflection worker when `REFLECTION_CYCLE_ENABLED=true`.

## Adapters

Adapter ownership boundaries and runtime callback contracts are documented in
the [Adapter ICD](../src/adapters/README.md).

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

Outbound brain-originated sends follow the same public-channel list. Runtime
callback sends to non-listed guild channels are rejected before Discord
delivery, and normal `/chat` responses for listen-only guild channels are
suppressed locally with no delivery receipt. DMs remain sendable regardless of
the public channel list.

### NapCat QQ

The NapCat adapter reads `BRAIN_URL`, `NAPCAT_WS_URL`, and `NAPCAT_WS_TOKEN`
from the environment. For console-managed launches, set
`NAPCAT_ACTIVE_GROUPS` to a comma- or space-separated list of QQ group ids.
Explicit `--channels` CLI values override `NAPCAT_ACTIVE_GROUPS`.

```bash
python -m adapters.napcat_qq_adapter --channels 987654321
```

Listed groups are active, non-listed groups become listen-only, and private
chats are always active. If neither `--channels` nor `NAPCAT_ACTIVE_GROUPS` is
provided, all groups are listen-only and only private chats are active.

Outbound brain-originated sends follow the same public-group list. Runtime
callback sends to non-listed groups are rejected before NapCat `send_msg`, and
normal `/chat` responses for listen-only groups are suppressed locally with no
delivery receipt. Private chats remain sendable regardless of the public group
list.

## HTTP API

This section is a runbook-level endpoint map. Request and response schemas live
in the [Brain Service ICD](../src/kazusa_ai_chatbot/brain_service/README.md).

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

The `scheduler` boolean is a legacy health-field name kept in the public
response model. Use `/ops/runtime-status` for calendar scheduler enablement,
configuration, and worker liveness.

### `GET /ops/runtime-status`

Trusted-operator runtime status. This endpoint is separate from `/health` so
adapter readiness checks stay small and stable.

The response contains only aggregate service state:

- process last event status and timestamp,
- effective calendar, reflection, and self-cognition worker config,
- calendar poll interval, claim limit, lease duration, and retry limit,
- effective reflection phase period, minimum slot spacing, maximum slots, and
  one-group-per-slot invariant,
- process-local worker liveness flags,
- latest worker event status and timestamp,
- semantic health labels such as `worker_error_level`.

### `GET /ops/reflection/stats`

Trusted-operator reflection stats for a bounded event-log window. The response
contains counts, latest run refs, and deterministic semantic labels. It does
not expose reflection prompt text, raw reflection output, source messages, or
conversation details.

### `GET /ops/self-cognition/stats`

Trusted-operator self-cognition stats for a bounded event-log window. The
response contains the service-owned `enabled` and `task_alive` state, run
counts, dispatcher handoff counts, latest refs, and semantic liveness labels.
This avoids treating `self_cognition_liveness=inactive` as the full worker
state; inactive only means no self-cognition run events were recorded in the
window. It does not expose source packets, action candidate text, or generated
dialog.

The `/ops/*` endpoints have no authentication or authorization in this plan.
Deployments must keep them on localhost or a trusted operator network until a
separate auth plan is implemented.

In-process event logging can record startup, graceful shutdown, lifespan
failures, handled request/worker exceptions, and worker-loop exceptions. It
does not prove OS kills, interpreter aborts, host crashes, power loss, or
external supervisor restarts.

Sanitized event-log rows use `AUDIT_LOG_TTL_DAYS`. Protected LLM trace rows use
`DEBUG_LOG_TTL_DAYS`.

Routine successful chat input is not mirrored into `event_log_events`.
Successful user and assistant message writes are audited through
`conversation_history`; event logging focuses on queue drops/collapses, failed
persistence, runtime errors, worker/resource health, and model-contract issues.

Aggregate export:

```bash
python -m scripts.export_event_log --hours 24 --output test_artifacts/diagnostics/event_log_smoke.json
```

Without `--output`, the command writes
`test_artifacts/diagnostics/event_log_<UTC>.json`. The export includes the
same aggregate status/stat payloads and the deterministic snapshot write
result. It does not export raw event documents.

LLM trace export:

```bash
python -m scripts.export_llm_trace --dialog-text "14:30了"
python -m scripts.export_dialog_trace_review_input --trace-id llmtrace_<id>
```

`LLM_TRACE_CAPTURE_MODE=metadata` records stage names, route/model metadata,
prompt/output hashes, character counts, parse status, and state handoff fields.
`full` additionally stores raw prompt messages, raw response text, and parsed
output in protected trace collections. `off` skips trace row writes.

Apply or inspect logging retention for existing rows:

```bash
python -m scripts.apply_logging_retention --dry-run
python -m scripts.apply_logging_retention --apply
```

Recent terminal status:

```bash
python -m scripts.fetch_ops_status --hours 24
python -m scripts.fetch_ops_status 6 --json
```

The status command reads the same aggregate event-log builders used by the
`/ops/*` endpoints. It prints recent runtime, reflection, and self-cognition
status without writing a snapshot or exporting raw event documents. The local
CLI includes the configured self-cognition `enabled` value; use
`/ops/runtime-status` or `/ops/self-cognition/stats` on the running service when
you need process-local `task_alive` state.

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

For the exact `ChatRequest` and `ChatResponse` fields, adapter rules,
`delivery_tracking_id` semantics, and delivery receipt flow, read the
[Brain Service ICD](../src/kazusa_ai_chatbot/brain_service/README.md). For the
typed inbound envelope fields, read the
[Message Envelope ICD](../src/kazusa_ai_chatbot/message_envelope/README.md).

Useful drop-audit log line:

```text
Queued chat item dropped: sequence=... platform=... channel=... message=... user=... display_name=... tagged=... bot_reply=... content="..."
```

Current attachment behavior:

- inbound image attachments with inline base64 are supported
- image descriptions are generated before relevance
- outbound attachments are reserved for future service support

### Other Service Endpoints

The brain service also exposes delivery receipt, runtime adapter registration,
runtime adapter heartbeat, and generic event endpoints. Their contracts and
compatibility rules are maintained in the
[Brain Service ICD](../src/kazusa_ai_chatbot/brain_service/README.md).

## Runtime Data Model

`db_bootstrap()` creates current collections and indexes, and it drops legacy
RAG1 collections `rag_cache_index` and `rag_metadata_index` when present.

Collection purpose, document ownership, storage invariants, and bootstrap/index
rules are maintained in the
[Database ICD](../src/kazusa_ai_chatbot/db/README.md). Keep this HOWTO focused
on operator commands and setup.

## Legacy Collection Cleanup

Bootstrapping handles stale legacy collections automatically. There is also an
idempotent one-shot cleanup script for operations:

```bash
python scripts/drop_legacy_rag_collections.py
```

The script drops `rag_cache_index` and `rag_metadata_index` when present and is
safe to run repeatedly.

## Global Character Growth

Global character growth runs after daily global reflection promotion when the
worker is enabled and `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED=true`. It writes
only `global_character_growth_traits` and `global_character_growth_runs`.
Candidate generation receives a final rendered prompt bounded by
`GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET`, which defaults to `32000`
characters.

Dry-run:

```bash
python -m scripts.run_global_character_growth --dry-run --limit 80
```

Apply:

```bash
python -m scripts.run_global_character_growth --apply --enable-trait-writes --limit 80
```

Rollback:

```bash
GLOBAL_CHARACTER_GROWTH_PASS_ENABLED=false
```

Only active promoted traits enter L2 cognition through promoted reflection
context. Emerging and stabilizing traits remain audit-only in run records.

## Testing

Default test runs exclude live DB and live LLM tests through `pytest.ini`.

```bash
pytest -q
pytest -m "not live_db and not live_llm" -q
venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_persona_graph.py tests\test_cognition_resolver_l2d_contract.py -q
```

Live LLM tests must be run and inspected one at a time:

```bash
pytest -m live_llm tests/test_cognition_live_llm.py::test_live_msg_decontexualizer_returns_non_empty_output -q -s
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
