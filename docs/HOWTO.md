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
COGNITION_STAGE_TIMEOUT_SECONDS=120
COGNITION_LLM_CHARACTER_CARRYOVER_BASE_URL=http://localhost:1234/v1
COGNITION_LLM_CHARACTER_CARRYOVER_API_KEY=lm-studio
COGNITION_LLM_CHARACTER_CARRYOVER_MODEL=your-chat-model
COGNITION_LLM_CHARACTER_CARRYOVER_MAX_COMPLETION_TOKENS=8192
COGNITION_LLM_CHARACTER_CARRYOVER_THINKING_ENABLED=false
# Required agentic cognition chain.
COGNITION_V3_CHAIN_LLM_BASE_URL=http://localhost:1234/v1
COGNITION_V3_CHAIN_LLM_API_KEY=lm-studio
COGNITION_V3_CHAIN_LLM_MODEL=your-chat-model
COGNITION_V3_CHAIN_LLM_MAX_COMPLETION_TOKENS=8192
COGNITION_V3_CHAIN_LLM_CONTEXT_WINDOW_TOKENS=50176
COGNITION_V3_CHAIN_LLM_THINKING_ENABLED=false
# The sidecar is optional; set BASE_URL, API_KEY, and MODEL together.
COGNITION_V3_SIDECAR_LLM_BASE_URL=http://localhost:1234/v1
COGNITION_V3_SIDECAR_LLM_API_KEY=lm-studio
COGNITION_V3_SIDECAR_LLM_MODEL=your-sidecar-model
COGNITION_V3_SIDECAR_LLM_MAX_COMPLETION_TOKENS=8192
COGNITION_V3_SIDECAR_LLM_THINKING_ENABLED=false
COGNITION_V3_SUBCONSCIOUS_ENABLED=false
COGNITION_V3_TURN_DEADLINE_SECONDS=240
AGENTIC_RESOLVER_LLM_BASE_URL=http://localhost:1234/v1
AGENTIC_RESOLVER_LLM_API_KEY=lm-studio
AGENTIC_RESOLVER_LLM_MODEL=your-chat-model
AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS=50176
AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS=8192
AGENTIC_RESOLVER_LLM_THINKING_ENABLED=false
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
COGNITION_VISUAL_DIRECTIVES_ENABLED=false
COGNITION_RESOLVER_MAX_CYCLES=3
COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS=120.0
TASK_RESOLUTION_INLINE_BUDGET_SECONDS=30.0
SELF_COGNITION_ENABLED=true
CHARACTER_SLEEP_LOCAL_PERIOD=02:00-12:00
GROUP_SCENE_MAX_TURN_AGE_MINUTES=120
CONVERSATION_PROGRESS_BACKGROUND_MAX_AGE_MINUTES=120
CONVERSATION_PROGRESS_ACTIVE_SCENE_MAX_AGE_MINUTES=360
CONVERSATION_PROGRESS_DECISION_CRITICAL_MAX_AGE_MINUTES=2880
CONVERSATION_PROGRESS_NARRATIVE_MAX_AGE_MINUTES=360

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

# DSH Standard task runtime
KAZUSA_DSH_SIDECAR_URL=http://127.0.0.1:8787/rpc
KAZUSA_DSH_RPC_TOKEN=replace-with-local-rpc-secret
KAZUSA_DSH_DATA_ROOT=C:\workspace\kazusa_dsh_data
KAZUSA_DSH_BRAIN_URL=http://127.0.0.1:8000
KAZUSA_DSH_BRAIN_SHARED_SECRET=replace-with-local-brain-secret
KAZUSA_DSH_TOOL_GATEWAY_SECRET=replace-with-local-gateway-secret
KAZUSA_DSH_PYTHON_EXECUTABLE=C:\workspace\kazusa_ai_chatbot\venv\Scripts\python.exe
AGENTIC_RESOLVER_WORKSPACE_ROOT=C:\workspace\kazusa_ai_chatbot

# Direct web search and URL-reader behavior
SEARXNG_URL=http://your-searxng-host:8080
SEARXNG_SEARCH_ENGINES=yandex,360search,baidu,sogou
SEARXNG_SEARCH_TIMEOUT_SECONDS=30
SEARXNG_SEARCH_RESULT_LIMIT=10
WEB_URL_READ_TIMEOUT_SECONDS=30
WEB_URL_READ_MAX_BYTES=1048576
WEB_URL_READ_MAX_CHARS=10000
WEB_URL_READ_REDIRECT_LIMIT=5
WEB_URL_READER_USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36
WEB_URL_READER_ACCEPT_LANGUAGE=en-US,en;q=0.9
NHENTAI_TOKEN=

# Optional generic MCP servers and timeouts
MCP_SERVERS={}
MCP_CALL_TIMEOUT=30
MCP_CONNECT_TIMEOUT=10

# Agent retry limits
MAX_MEMORY_RETRIEVER_AGENT_RETRY=2
MAX_WEB_SEARCH_AGENT_RETRY=2

# Cache2
RAG_CACHE2_MAX_ENTRIES=5000

# Process-local image session cache for RAG3 media inspection
MEDIA_SESSION_CACHE_MAX_ITEMS_PER_SCOPE=8
MEDIA_SESSION_CACHE_MAX_BYTES_PER_SCOPE=16777216
MEDIA_SESSION_CACHE_MAX_ITEM_BYTES=6291456
MEDIA_SESSION_CACHE_TTL_SECONDS=900

# Reflection cycle
REFLECTION_CYCLE_ENABLED=true
REFLECTION_WORKER_INTERVAL_SECONDS=900
REFLECTION_HOURLY_SLOTS_PER_TICK=3
REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS=60
REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD=3
REFLECTION_DAILY_RUN_AFTER_LOCAL_TIME=04:30
REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME=05:00
CHARACTER_IDENTITY_GROWTH_ENABLED=true
CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_EPISODES=3
CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_LOCAL_DATES=2
CHARACTER_IDENTITY_GROWTH_MAX_INFERRED_PROMOTIONS_PER_LOCAL_DAY=1
CHARACTER_IDENTITY_GROWTH_PROMPT_CHAR_BUDGET=18000

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
the background artifact route when omitted. Code-reading PM and programmer
routes are required first-class routes. Final code-reading synthesis reuses the
PM route and has no separate route identity.
The generic `COGNITION_LLM` bundle remains required for cognition callers
outside the agentic cognition chain. The runtime requires the complete chain
bundle plus an optional complete sidecar bundle. The fixed appraisal layout is
`fixed_a1_a2`, with `COGNITION_V3_TURN_DEADLINE_SECONDS` bounded to
`30..600` (default `240`) and a declared chain context window of at least
50,000 tokens. The normal total ceiling is 50,000 tokens, with the
conditional 65,000-token tier available only when the declared serving window
supports it.
`COGNITION_LLM_CHARACTER_CARRYOVER` is the state-only background operational
carry-over route; its completion-token setting is capped at 8,192.
Route-specific variables replace the retired generic `LLM_BASE_URL`,
`LLM_API_KEY`, and `LLM_MODEL` settings. Missing required route variables stop
config loading. Chat routes also accept route-specific
`*_MAX_COMPLETION_TOKENS` and `*_THINKING_ENABLED` values, with
`DEFAULT_LLM_MAX_COMPLETION_TOKENS` retained for routes without an explicit
budget. The cognition timeout default is 120 seconds and its accepted range is
10 to 600 seconds.
Thinking is a boolean route toggle and defaults to disabled. For code reading,
the recommended local-model starting point is PM thinking enabled and
programmer thinking disabled, because PM planning and synthesis benefit more
from longer reasoning while programmer workers should stay bounded to selected
source evidence. When enabled, the LLM interface currently maps
provider-specific thinking controls for Gemma 4, Qwen3-family model names, and
Qwen-compatible Qwopus 3.x model names.

The cognition input preserves typed identity, boundaries, relationship axes,
affect values with their causes, evidence handles, conversation progress, and
current scene pressure. The agentic chain receives one canonical bounded
projection and owns semantic appraisal, response goals, action choice, affect
projection, and visible-surface planning. Context reduction preserves required
current-turn facts and causal affect/relationship context before optional
supporting rows.

The canonical model-facing packets separate five authority lanes:
`current_observation`, `direct_facts`, `participant_continuity`,
`conditional_character_context`, and `continuation_state`. A1 receives the
world-facing observation and factual lanes without conditional character
context. A2 and G may use character and relationship context only for
character judgment and motivation. Participant continuity never establishes a
new action, consent, commitment, permission, or current intent. G returns the
exact bounded `private_monologue`; P returns the bounded
`epistemic_boundary` that controls assertions, interpretations, and unknowns.

`RELEVANCE_AGENT_LLM` serves both compact frontline intake and settled
relevance. Frontline uses a 256-token completion cap, thinking disabled, and
an 8,000-character rendered-input cap; settled relevance uses a 512-token
completion cap, thinking disabled, and a 16,000-character cap. Both stages
parse JSON deterministically and never call the JSON-repair LLM. They share one
FIFO relevance executor with one in-flight call; settlement timers hold no
model slot. Settled history uses one embedding-excluded scan of at most 48
channel rows, excludes active-turn rows, collapses complete assistant response
fragments through the canonical logical-turn contract, and retains the newest
ten logical turns. Its history projection keeps the newest whole turns under an
exact 6,000-character compact-JSON sub-budget before the unchanged
16,000-character settled-input cap is applied.

## Agentic cognition operator runbook

Startup requires the complete `COGNITION_V3_CHAIN_LLM_*` bundle and uses the optional
all-or-nothing `COGNITION_V3_SIDECAR_LLM_*` bundle when configured. V3 uses the
fixed `fixed_a1_a2` appraisal-stage layout, and
`COGNITION_V3_TURN_DEADLINE_SECONDS` accepts `30..600` (default `240`). The
chain context-window declaration must be at least 50,000 tokens.
The primary caller declares a 50,000-token normal total ceiling; the
conditional 65,000-token ceiling is available only when
`COGNITION_V3_CHAIN_LLM_CONTEXT_WINDOW_TOKENS` is at least `65000`. The
context-window declaration remains caller-local and is not sent to the
provider. Runtime timing evidence is non-streaming elapsed milliseconds, with
no TTFT claim.

Collect deterministic evidence with the exact commands below:

```powershell
venv\Scripts\python -m pytest -q tests/integration/cognition_core_v3/test_chain_observability.py::test_protected_and_sanitized_records_share_exact_service_console_correlation
```

The brain endpoint and control console pair each graph with its exact chain
run; missing or mismatched `run_id`/`llm_trace_id` is shown as `not_reported`.

## DSH Standard Sidecar

DSH is the only production route for bounded multi-step task resolution.
`task_resolution_request` opens or continues one fenced sidecar session; typed
checkpoints and terminal results return through cognition, dialog, dispatcher,
and adapter delivery. RAG3 remains the ordinary local-context evidence owner.

Install, build, and type-check the pinned sidecar independently from the Python
service:

```powershell
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution install --frozen-lockfile
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution build
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution typecheck
```

Start MongoDB and the Brain first. After `/health` is ready, start the sidecar:

```powershell
node sidecars/dsh_resolution/dist/src/main.js
```

The authenticated Brain endpoint `GET /runtime/dsh/health` must report
`configured`, `durable_store`, and `cognition_judge`. Authenticated sidecar
`system.health` must agree on route, Standard, semantic-worker, web, Brain,
catalog, policy, workspace, profile, release, and store identity before task
admission is enabled.

Run the permanent non-LLM process probes with a new or empty artifact
directory for each invocation:

```powershell
venv\Scripts\python experiments\dsh_runtime_probe.py sidecar-lifecycle --artifact-dir <new-directory>
venv\Scripts\python experiments\dsh_runtime_probe.py brain-task-lifecycle --artifact-dir <new-directory>
venv\Scripts\python experiments\dsh_runtime_probe.py transport-loss --artifact-dir <new-directory>
```

The latter two require MongoDB and use a unique guarded test database that the
probe drops during cleanup. Exit codes are 0 for pass, 1 for observed failure,
and 2 for a blocked prerequisite.

The model-visible semantic catalog contains fourteen supported tools for
conversation history, memory, people, active context, calendar, attached
media, and public media. `kazusa_inspect_public_media` accepts an HTTP(S) image
URL and a visual question. Its fetch boundary rechecks DNS and redirects,
rejects private or special-use addresses, limits redirects to 3, response time
to 15 seconds, body size to 6 MiB, supported formats to PNG/JPEG/GIF/WebP, and
dimensions to `1..8192`. Raw bytes and base64 remain outside model context.

The read-only legacy drain audit is:

```powershell
venv\Scripts\python scripts/check_dsh_legacy_drain.py --legacy-coding-workspace-root <abs-root> --format json
```

It reports remaining historical task-execution state without writing to MongoDB
or legacy ledgers. Production cleanup or deployment remains a separately
authorized operation.

## Relevance Turn Settlement

Active chat intake commits one canonical `conversation_history` user receipt
with a server-generated `received_at` before queue admission, then persists
or updates that same row during semantic routing. The frontline route
receives bounded message evidence, typed target/reply labels, open-turn
descriptors, and silent same-author preludes. It returns `discard`, `start`,
or `append`.

Both relevance stages also receive bounded interaction evidence and a bounded
projection of the same process-local native character cognition state. Group
admission uses exactly two semantic bases: interaction relevance grounded by
supplied target/reply/broadcast, complete-name, continuity, open-turn, or
history evidence; or a concrete message intersection with supplied active
character state. Generic helpfulness, emotionality, and topic interest do not
establish either basis. Evidence refs are validated against the final
cap-fitted payload, and the internal recipient/basis assessment is removed from
the unchanged public decision.

Group turns use a six-second quiet window and a ten-second hard deadline.
Settled turns are ordered globally by eligibility and arrival sequence. The
settled route returns `ignore`, `proceed`, or one bounded `wait` extension;
only a matching turn version can claim the cognition lane. Private messages
retain immediate timing and adjacency-only coalescing; their frontline input
contains the full coalesced logical message. Image description runs after
frontline admission and selects the opening image plus the newest remaining
unique images, up to four across every assessment of one turn. Settled
relevance receives an overflow marker and ignores media-dependent turns when
the required retained media is undescribed.

`CHARACTER_GLOBAL_USER_ID` defaults to
`00000000-0000-4000-8000-000000000001`. Set it explicitly in production so the
active character keeps a stable first-class identity across service runs.

`web_read` is always available and can read HTTP(S) URLs directly from the
Kazusa process. `SEARXNG_URL` enables `web_search`; when it is empty, the
search source is not registered as an available web_agent3 source. When it is
set, search uses the configured SearXNG `/search?format=json` endpoint and the
comma-separated `SEARXNG_SEARCH_ENGINES` selection directly. URL reads do not
require SearXNG or MCP, and local/private HTTP(S)
resources reachable from the Kazusa process are allowed by default. URL reads
always use browser-navigation headers, process-memory cookies, locally
supported compression encodings, and common HTTP anti-bot challenge detection.
They do not execute JavaScript, solve CAPTCHA, or impersonate browser TLS
fingerprints. `NHENTAI_TOKEN` enables the nHentai metadata/search source; when
it is empty, that source is not registered. Installing the Bilibili optional
extra enables the Bilibili public read/search source:
`pip install -e .[bilibili]`. The referenced package is
`bilibili-api-python` on PyPI:
https://pypi.org/project/bilibili-api-python/. Bilibili source availability
does not require an `.env` setting. `MCP_SERVERS` remains available for
unrelated generic MCP tools.

When LM Studio reports
`The model has crashed without additional information.`, chat model calls made
through `LLInterface` retry the same request once. Calls for the same
`base_url` and model wait while that retry reloads the model; calls for other
models continue normally. Other 400 responses and non-unload errors are not
retried by this recovery path.

`COGNITION_VISUAL_DIRECTIVES_ENABLED` is a brain-service level switch. Set it
to `true` to enable terminal L3 visual-directive generation globally. It
defaults to `false`; adapters and debug-client request payloads do not control
this behavior, and visual directives have no downstream agent.

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

Text L3 runs one semantic content-planning provider call. Its typed input
includes the exact current-turn private monologue for expression only, P's
authoritative epistemic boundary, and the caller-owned addressee plan.
Deterministic projection emits an empty `visible_boundaries` list and copies
the validated addressee rows; there is no preference-model call or preference
repair path. The validated surface output copies P's exact
`epistemic_boundary` into dialog, while `private_monologue` remains absent from
the dialog payload. With no corresponding `executed` permitted-action result,
surface and dialog express only a verbal acceptance, refusal, proposal,
invitation, or intent; they do not render an external effect as completed.

The control-console cognition observation view exposes one generic selected-
section widget in Overview Latest, Debug cognition, and the dedicated latest
self-cognition Overview panel. Each view consumes the Brain-owned
`cognition_run_observation.v1` object and resolves node `section_refs` in
producer order. It renders producer labels, section status and summary,
ordered fields and records, truthful displayed/reported counts, and explicit
omission markers. Additive producer sections require no JavaScript catalog.
Long approved text, CJK, emoji, multiline values, and HTML-sensitive text are
rendered through text-safe escaping; prompts, raw model output, embeddings,
message envelopes, and operational identifiers remain excluded.

The `reasoning.context_consumption` section is the only latest turn-consumption
source used by the Character operational-posture panel. Brain creates the
already-safe section from the immutable turn snapshot and actual cognition/
surface inputs; the console only validates, transports, and renders it. When
Brain is unavailable, did not report it, or sends an invalid contract, the
panel states that availability instead of reconstructing a substitute.

The Character page additionally shows all persisted and elapsed-effective
native affect/pressure rows, source and view digests, and whether ordinary
fading changed the effective view. The User page adds a current-user causal
relationship projection, and User/Group style panels label relevance,
cognition, and surface consumers.

The brain process log prints the complete normalized visual directive only
after successful enabled validation. The protected LLM trace remains separate
for model metadata and raw-output diagnostics.
`CHARACTER_SLEEP_LOCAL_PERIOD` defaults to `02:00-12:00` in
`CHARACTER_TIME_ZONE`. During that local period, active-commitment
self-cognition and reflection-attached group self-cognition do not trigger.
Scheduled future cognition, durable calendar due-run handling, reflection,
consolidation, dispatcher validation, and adapter delivery continue. The same
sleep period also schedules daily affect settling for persistent
`character_state` mood, global vibe, and reflection summary. Set
`CHARACTER_SLEEP_LOCAL_PERIOD` to an empty value to disable both sleep-period
self-cognition suppression and affect settling.

Daily affect settling has no separate `AFFECT_SETTLING_ENABLED` rollback flag.
A non-empty `CHARACTER_SLEEP_LOCAL_PERIOD` enables the schedule; an empty value
disables both sleep-period self-cognition suppression and affect settling. The
only env-backed affect-settling knob is `AFFECT_SETTLING_WAKE_PREP_MINUTES=30`.

The remaining affect-settling policy values are named constants in
`kazusa_ai_chatbot.reflection_cycle.affect_settling`:

- `AFFECT_SETTLING_PROMPT_MAX_CHARS=12000`
- `AFFECT_SETTLING_REVIEW_PROMPT_MAX_CHARS=8000`
- `AFFECT_SETTLING_AFTER_PROMOTION_GRACE_MINUTES=15`
- `AFFECT_SETTLING_WAKE_DEFER_GRACE_MINUTES=15`

The due local time is the later of promotion time plus grace and sleep end
minus wake prep. The affect-settling module import fails if that due time is
after sleep end plus wake defer grace.

### Context fade and sleep phase

Aged conversational context is discarded deterministically before projection,
never handed to a model with an instruction to discount it. Group-scene
ambient turns older than `GROUP_SCENE_MAX_TURN_AGE_MINUTES` (default `120`)
relative to the trigger are dropped by the shared
`filter_group_scene_ambient_turns` projection used by group-scene rendering
and the persona Stage 0 decontextualizer; the filtered sequence also supplies
group scope participants. The trigger is never filtered and
`omitted_turn_count` counts only count-based truncation. Conversation-progress events older than their retention-tier
threshold are dropped on the read path immediately after packet selection:
`CONVERSATION_PROGRESS_BACKGROUND_MAX_AGE_MINUTES` (default `120`),
`CONVERSATION_PROGRESS_ACTIVE_SCENE_MAX_AGE_MINUTES` (default `360`), and
`CONVERSATION_PROGRESS_DECISION_CRITICAL_MAX_AGE_MINUTES` (default `2880`).
When no event survives, or the newest surviving event is older than
`CONVERSATION_PROGRESS_NARRATIVE_MAX_AGE_MINUTES` (default `360`), the complete
narrative field set is cleared to its canonical empty shape. Pruning is a
read-path projection concern only: it issues no database write, and the next
recorded turn persists the pruned form.

Progress evidence rows carry each originating event's own `updated_at` as
`evidence_ref.occurred_at`, normalized to the canonical UTC-Z second-truncated
format, and `scene_context.semantic_temporal_context` is derived from the
newest surviving event age using the `project_duration` vocabulary.

`scene_context.character_sleep_phase` is derived from
`CHARACTER_SLEEP_LOCAL_PERIOD`, `CHARACTER_TIME_ZONE`, and
`AFFECT_SETTLING_WAKE_PREP_MINUTES` by the deterministic projector in
`cognition_shared.state_projection`. The frozen vocabulary is `清醒时段`
outside the window, `睡眠中` inside the window, and `即将醒来` within the final
`AFFECT_SETTLING_WAKE_PREP_MINUTES` before the window ends; the two in-window
labels cover exactly the same half-open local window that gates the
self-cognition lanes. The phase reaches goal cognition only and never the
appraisal or surface prompts.

Daily affect settling runs through the `cognition_shared.morning_refresh`
`run_character_morning_refresh(...)` entrypoint, which owns the character
scope guard, the sleep-recovery reducer call, and output-state validation.
The reflection cycle keeps scheduling, idempotency, the guarded write, the
refresh callback, and the audit row.

`BACKGROUND_WORK_WORKER_ENABLED` controls the internal background-work runtime.
L2d emits `task_resolution_request` only after character cognition decides that
bounded task work is warranted. The foreground edge uses
`TASK_RESOLUTION_INLINE_BUDGET_SECONDS` (default `30.0`, range `1.0` through
`120.0`). A committed DSH checkpoint may promote the same fenced session to an
accepted task and `task_orchestrator` job; lease or process retries resume the
durable binding and operation generation.

The task worker accepts `task_orchestrator_worker_payload.v2` operations
`open_dsh_resolution` and `continue_dsh_resolution`. Direct background
admission first returns a transient, model-hidden `TaskResolutionAdmissionV1`;
only a committed checkpoint may return a deferred `TaskResolutionResultV1`
with a `DshResolutionRefV1`. Terminal and partial results return through the
accepted-task result source and normal cognition. Workers do not write final
dialog or send adapter messages.

`accepted_task_control.v1` exposes `continue`, `summarize`, and `cancel` for
the same opaque accepted-task/session binding. `accepted_task_status_check` is
read-only. Each mutating control obtains fresh DSH authority and is protected by
scope validation, revision CAS, lease fencing, and operation idempotency.
`future_speak` remains a separate deterministic scheduler handoff.

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
- optional `bilibili-api-python` package for Bilibili public read/search
- optional generic MCP servers for unrelated tools

Direct URL reads use the existing HTTP client dependency and do not require an
additional browser transport or automation dependency.

LM Studio works for local model hosting, but any OpenAI-compatible endpoint can
be used.

## Character Profile

Normal startup never creates identity data. Seed a complete canonical profile
manually before starting the brain against a clean database:

```bash
python -m scripts.load_character_profile personalities/example.json
```

The loader validates the whole profile before writing immutable revision `0`
and the operational character state. Re-running it without `--force`
preserves the existing ledger. An intentional replacement creates a new
immutable `operator_reset` revision and requires an audit identifier:

```bash
python -m scripts.load_character_profile personalities/example.json \
  --force --operator-action-id change-ticket-123
```

If the ledger has no revision, brain startup raises an error before intake and
does not derive identity from `character_state`, conversations, reflection,
residue, or any packaged fallback. Use `personalities/example.json` only as a
template; character-specific profiles can remain outside the repository.

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

Stage 3 fresh-database verification uses the configured database
`_test_kazusa_core_v2`
on the ordinary MongoDB URI, with the exact database name and child-process
guard providing isolation. The
runtime has five canonical cognition sources: `user_message`,
`internal_thought`, `self_cognition`, `scheduled_tick`, and `tool_result`.
Each episode settles exactly one `episode_trace.v2`; post-turn action work is
recorded in `post_turn_lifecycle_records`, and action-triggered internal thought
uses the durable `internal_action_latches` collection.
The native cognition input contract is `cognitive_episode.v1`.

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
2. Requires and loads the latest manually seeded immutable identity revision,
   then composes it with operational character state; missing identity stops
   startup before intake.
3. Hydrates persistent media descriptor cache entries.
4. Compiles the top-level LangGraph pipeline.
5. Starts configured MCP servers.
6. Builds the runtime adapter registry and starts the chat input worker.
7. Starts the durable calendar worker when `CALENDAR_SCHEDULER_ENABLED=true`.
8. Starts the self-cognition worker when `SELF_COGNITION_ENABLED=true`.
9. Starts the background-work runtime when
    `BACKGROUND_WORK_WORKER_ENABLED=true`.
10. Starts the reflection worker when `REFLECTION_CYCLE_ENABLED=true`.

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

For the cognition chain, both `metadata` and `full` also promote failed,
recovered, partial, and degraded invocations into one protected failure
capsule. Clean invocations create no capsule row. The capsule preserves the
raw pre-validation entrypoint input and ordered model attempts, excludes API
keys, uses `DEBUG_LOG_TTL_DAYS`, and is written asynchronously so capture
failure cannot replace or delay cognition behavior.

`scripts.export_llm_trace` keeps the original `llm_trace_steps` array and adds
`cognition_failure_capsules`, grouped by each
`cognition_invocation_id`, for self-contained replay input. Pass
`--cognition-invocation-id <id>` with `--trace-id` to select one safe-retry
invocation while retaining the compatible trace-level fields.

Trace correlation runbook for the Control Console use case:

1. Copy the value displayed as `trace <id>` in Debug Chat and record the exact
   source surface as `web_control_trace_id`. A graph `run_id`, event
   `correlation_id`, delivery tracking id, or bare opaque value is not a trace
   until exact field evidence establishes it.
2. Export the bounded manifest before requesting raw trace evidence:

   ```powershell
   venv\Scripts\python -m scripts.export_trace_correlation_manifest `
     --identifier <copied-value> `
     --source-surface web_control_trace_id `
     --output test_artifacts\diagnostics\trace_correlation_<name>.json
   ```

3. Inspect `parent_trace`, `identifiers`, `joins`, and `unresolved`. The
   resolver preserves zero and multiple candidates, reports
   `not_available_from_web` for untyped browser values, and never chooses a
   newest row. Open `export_llm_trace --trace-id <trace>` only after the parent
   status is `confirmed`.

Availability matrix:

| Identifier | Browser availability | Next exact route |
| --- | --- | --- |
| Debug Chat `trace_id` | rendered when authorized and retained | correlation manifest with `web_control_trace_id` |
| `delivery_tracking_id` | rendered separately | `export_llm_trace --delivery-tracking-id <id>` |
| `platform_message_id` | API/request-only | `export_llm_trace --platform-message-id <id>` |
| `cognition_invocation_id` | protected-only | parent trace export selector |
| `global_user_id` | protected-only | manifest with `protected_global_user_id` |
| action/background/accepted-task/calendar ids | absent from current Console projection | manifest with the matching protected source surface |
| graph `run_id` or event `correlation_id` | generic surface only | record `unknown`; do not infer a trace |
| child/future execution trace ids | protected-only | manifest child-trace joins |

The database-pull and debug-LLM skills treat this manifest as the protected
evidence handoff. Keep generated JSON under `test_artifacts/diagnostics`,
inspect each anchor one at a time, and keep raw trace export separate from the
identifier-only manifest.

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
and waits for that request's response future. Before queue admission the
service commits one durable inbound user receipt with a server-generated
`received_at`; queue pruning, shutdown drain, coalescing, and frontline
discard cannot erase it, and later intake never inserts a second copy. The
intake worker consumes or updates that same row and runs compact frontline
decisions in arrival order. It then returns to intake while admitted group
turns wait in the independent settlement scheduler. Explicit `listen_only`
input follows its debug bypass; active group messages have no queue-threshold
semantic pruning before frontline.

Adapters own platform-specific reply detection. The brain projects typed target
and reply evidence into frontline semantic labels; raw reply identifiers do not
act as queue-level protection or routing signals.

The service reuses its already loaded native character-state snapshot for both
relevance calls and performs no relevance-specific database read. Settled
history uses stable bounded participant handles instead of raw ids. A
state-salience decision may retain another participant as the actual recipient;
in that case the relevance-owned base `use_reply_feature` remains false;
deterministic delivery may still promote the final flag for a qualifying group
response when a durable same-channel user receipt arrived after the response
owner's receipt and before the response cutoff, or when the owner's durable
server-arrival age strictly exceeds 120 seconds. Author identity and later
packet disposition do not filter that evidence.

Group turns become eligible after six quiet seconds and close at ten seconds
from the opening enqueue time. Inputs enqueued before the hard boundary are
applied by frontline before the cognition claim can succeed. Waiting turns hold
neither the intake worker nor the cognition lane. Private survivors remain
immediately eligible, and adjacent queued private follow-ups keep one response
owner while appended requests return an empty response.

For the exact `ChatRequest` and `ChatResponse` fields, adapter rules,
`delivery_tracking_id` semantics, and delivery receipt flow, read the
[Brain Service ICD](../src/kazusa_ai_chatbot/brain_service/README.md). For the
typed inbound envelope fields, read the
[Message Envelope ICD](../src/kazusa_ai_chatbot/message_envelope/README.md).

Useful listen-only audit log line:

```text
Queued chat item dropped: sequence=... platform=... channel=... message=... user=... display_name=... tagged=... bot_reply=... content="..."
```

Current attachment behavior:

- inbound image attachments with inline base64 are supported
- image descriptions run after frontline admission and before settled relevance
- outbound attachments are reserved for future service support

### Other Service Endpoints

The brain service also exposes delivery receipt, runtime adapter registration,
runtime adapter heartbeat, and generic event endpoints. Their contracts and
compatibility rules are maintained in the
[Brain Service ICD](../src/kazusa_ai_chatbot/brain_service/README.md).

## Runtime Data Model

`db_bootstrap()` creates current collections and indexes. It performs no
destructive legacy collection cleanup during service startup.

Collection purpose, document ownership, storage invariants, and bootstrap/index
rules are maintained in the
[Database ICD](../src/kazusa_ai_chatbot/db/README.md). Keep this HOWTO focused
on operator commands and setup.

## Legacy Collection Cleanup

Use the explicit idempotent one-shot cleanup script when an approved operation
needs to remove legacy RAG collections:

```bash
python scripts/drop_legacy_rag_collections.py
```

The script drops `rag_cache_index` and `rag_metadata_index` when present and is
safe to run repeatedly.

## Daily Affect Settling

Manual dry-run:

```bash
python -m scripts.run_reflection_cycle affect-settle --dry-run
```

Manual apply:

```bash
python -m scripts.run_reflection_cycle affect-settle --enable-character-state-write
```

Use `--settling-local-date YYYY-MM-DD` for deterministic runs. Apply uses an
atomic compare-and-upsert against the `character_state.updated_at` value read
before the LLM call; a stale state records a skipped reflection run and does
not overwrite newer state.

## Character Identity Growth

Character identity growth runs after settled episodes selected by
consolidation and once after the daily reflection promotion when
`CHARACTER_IDENTITY_GROWTH_ENABLED=true`. Separate proposal and review LLM
stages decide whether evidence represents a character-owned, durable, global,
privacy-safe identity change. Deterministic policy owns root deduplication,
pace, reversal protection, validation, and immutable persistence.

The default inferred pace requires three distinct root episodes across two
character-local dates and permits one inferred promotion per local date.
Explicit, independently reviewed self-redefinitions may promote immediately.
Raise the minimum episode/date settings to slow change. A value of `0` for the
daily promotion cap pauses inferred promotion while retaining candidates for
review.

The operator command defaults to a read-only evaluation of the previous
character-local date:

```bash
python -m scripts.run_character_identity_growth
```

An intentional revision write requires both apply gates:

```bash
python -m scripts.run_character_identity_growth \
  --character-local-date YYYY-MM-DD \
  --apply --enable-revision-writes
```

Only the latest full revision reaches cognition, naming, text, and visual
surfaces. Older revisions and emerging candidates remain review-only. Close
relationships may shape global identity when evidence describes the
character's own durable capacity for love, trust, care, or vulnerability;
the other person's identity, private facts, promises, and intimate details
remain scoped and cannot enter the global revision.

## Testing

Default test runs exclude live DB and live LLM tests through `pytest.ini`.

```bash
pytest -q
pytest -m "not live_db and not live_llm" -q
venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_persona_graph.py tests\test_cognition_resolver_l2d_contract.py -q
```

For runtime changes, begin with a small executable probe through the actual
integration boundary. Diagnose observed failures before adding regression
checks. The DSH
[runtime completion record](../development_plans/archive/completed/bugfix/dsh_runtime_completion_plan_2026-09-05.md)
contains fresh foreground, background, internal-judgment, coding and research
evidence, followed by recovery checks and focused regressions.

Live LLM tests must be run and inspected one at a time:

```bash
pytest -m live_llm tests/test_cognition_live_llm.py::test_live_msg_decontextualizer_returns_non_empty_output -q -s
pytest -m live_llm tests/test_dsh_behavior_live_llm.py::test_live_foreground_task_resolution_is_grounded_and_character_owned -q -s
pytest -m live_llm tests/test_dsh_behavior_live_llm.py::test_live_deferred_task_result_recurs_and_delivers_once -q -s
pytest -m live_llm tests/test_dsh_behavior_live_llm.py::test_live_internal_dsh_judgment_is_character_owned -q -s
pytest -m live_llm tests/test_dsh_behavior_live_llm.py::test_live_workspace_program_runs_and_produces_verified_output -q -s
pytest -m live_llm tests/test_dsh_behavior_live_llm.py::test_live_public_document_and_conversation_evidence -q -s
```

Each DSH behavior case writes a guarded technical dossier and a pending
character-review artifact under `test_artifacts/dsh_behavior_e2e/`. Review the
actual tool evidence, character responses, durable outcomes, and cleanup before
accepting a technical pass. Live LLM cases remain
isolated from deterministic contract and browser checks.

Set `KAZUSA_RUN_LIVE_LLM=1` for the explicit DSH live commands. Each wrapper
declares its supplied documents, natural inputs, hard gates, and independent
behavior rubric. Foreground coverage asks about conflicting release notes and
then clarifies the release. Deferred coverage compares documents with a missing
fact and checks the actual result delivery. Internal coverage pairs an
answerable question with an unsupported success approval request.
Workspace coverage creates and runs a Python program against a supplied CSV,
then verifies the generated JSON and the character's reported result.
Research coverage retrieves public Python documentation and searches scoped
conversation history for a user preference that was never supplied.

Each case uses a unique guarded `_test_kazusa_dsh_behavior_<uuid>` database.
The harness uses public application setup APIs and a separate guarded
diagnostic client, then drops that exact database. Process resources are shared
through `experiments/dsh_process_support.py`; the internal case starts only its
Brain collaborators. Protected artifacts include actual Brain model calls,
provider-reported usage, and the real DSH provider traffic where applicable.

Live DB tests can be run explicitly when MongoDB is available:

```bash
pytest -m live_db -q
```

Live LLM tests write inspection traces to `test_artifacts/llm_traces/`, which
is ignored by git.

### Cognition observation and browser verification

Brain publication and console views use one strict
`cognition_run_observation.v1` contract. Run deterministic contract,
projection, service, console, and documentation tests first, then collect the
browser suites with `--collect-only` before execution. Browser verification
must inspect Overview, Debug, and Self Latest in the in-app browser when
available, or the repository Playwright harness otherwise. Check exact section
and node order, additive producer sections, status/count/omission rendering,
loading/error separation, CJK/emoji/multiline text, HTML escaping, and empty
page/console error logs. The acceptance result must contain zero page or console error logs. Live LLM cases remain one-at-a-time and are not part
of the deterministic browser contract. The browser cases live under
`tests/control_console_e2e`.

The strict v1 boundary is Brain-owned: `CognitionRunObservationV1` is
validated, frozen, extra-forbid, UTC-Z serialized, and bounded before
publication. Its live/self producer catalogs, `sequence`/`reference` edges,
`item_01`–`item_24` record keys, disclosure exclusions, truthful counts, and
131072-character payload budget are contract checks. The
`evidence.shared_memory_prewarm` section keeps the fixed worker/merge
dispositions and does not expose raw worker output. The console adds only
validation-only availability metadata and renders additive producer sections
generically; it is not a second schema owner.

## Current Notes

- The supported local run path is local editable install plus
  `kazusa-control-console`; direct `uvicorn` startup remains the development
  fallback when bypassing the console.
- `Dockerfile` installs from `pyproject.toml`; `docker-compose.yml` remains a
  service-oriented deployment template that expects all required environment
  variables to be supplied.
- The maintenance profile script is `src/scripts/load_character_profile.py`;
  the operator owns initial seeding and normal startup verifies that a revision
  already exists.
