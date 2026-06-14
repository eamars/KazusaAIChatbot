# backend_control_console_top_level_plan.md

## Summary
- Goal: Add a top-level `control_console` management service that sits beside the brain service and adapters, starts/stops/restarts the brain and adapter processes, manages future registered services, runs a debug chat console, and monitors event logs plus character, memory, image/style, calendar, background-work, and health/cache state.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `py-style`, `test-style-and-execution`, `local-llm-architecture`, `database-data-pull`
- Overall cutover strategy: Replace the embedded-brain-console design with a separate `kazusa-control-console` entrypoint and top-level `src/control_console` package; preserve the existing brain `/chat`, `/health`, `/ops/*`, adapter registration, cognition, RAG, persistence, and worker contracts while moving service lifecycle ownership to the console's deterministic process supervisor.
- Highest-risk areas: local process supervision, accidental arbitrary command execution, stopping the wrong process, leaking per-user data or secrets through logs/lookups, stale status when a child process crashes, keeping the console useful while the brain is stopped, and preserving cognition/adapter semantics.
- Acceptance criteria: Operators can start, stop, restart, inspect, and tail logs for the brain and registered adapters from the top-level console; all 11 original inspection capabilities still exist; event logs are searchable; future services can be added through the service registry; no prompts/secrets/embeddings leak; deterministic tests and the independent code-review gate pass.

## Context
KazusaAIChatbot is a self-evolving character cognition runtime with a platform-neutral FastAPI brain service, thin adapters, typed message envelopes, MongoDB persistence, Cache2, calendar scheduling, background work, reflection, self-cognition, global character growth, and sanitized operational event logging. The brain service already exposes runtime endpoints such as `/chat`, `/health`, `/ops/runtime-status`, reflection stats, self-cognition stats, delivery receipts, and adapter registration/heartbeat routes. Adapters remain transport edges that normalize platform traffic into the brain service contract.

The previous plan placed the console inside the brain service and treated enable/disable as a soft intake or adapter gate. That design cannot satisfy the new product direction because an embedded console cannot reliably stop or restart its own host process and still remain available to turn it back on. This superseding plan treats `control_console` as the always-top management process. The operator starts `kazusa-control-console`; the console starts and stops the brain and adapters through a local process supervisor.

Target architecture:

```text
Operator browser
  -> control_console FastAPI app and static UI
      -> local process supervisor
          -> brain service process
          -> adapter process: Discord
          -> adapter process: debug
          -> future registered service processes
      -> HTTP health/debug client
          -> brain /health, /ops/runtime-status, /chat, adapter-facing status when available
      -> repository/read helpers
          -> MongoDB-backed character, growth, image/style, memory, calendar, background-work, cache, and event-log data
      -> local log store
          -> process stdout/stderr tails, service lifecycle audit, console errors
```

Fixed operating inputs:
- `control_console` is a top-level package under `src/control_console`, at the same source-tree level as `src/adapters` and `src/kazusa_ai_chatbot`.
- The operator-facing entrypoint is `kazusa-control-console`; operators no longer need to run separate brain and adapter start commands during normal local operation.
- Existing direct start commands remain available for development fallback and are also the commands the console launches through its registry.
- Service lifecycle control means deterministic local process lifecycle: start, stop, restart, health probe, stdout/stderr tail, crash detection, and audit.
- The console manages only services declared in a local registry. It never accepts arbitrary shell commands from the browser or API.
- Services started by the console are tracked through a local state directory. Processes not started or adopted by the console are shown as externally running or unknown and are not killed by v1 lifecycle actions.
- v1 uses Python `asyncio` subprocess management with argv lists and no shell execution. Docker, systemd, Windows service control, Kubernetes, and remote host management are not implemented in this plan.
- The console is local/operator-scoped, binds to loopback by default, requires authentication, and is not exposed to the public internet.
- The console can remain available when the brain is stopped. Runtime-only brain data degrades to unavailable, while DB-backed historical/lookup pages and local process logs remain usable when their dependencies are available.
- The console is not a cognition layer. It does not change prompts, graph routing, RAG, memory promotion, reflection, self-cognition, calendar semantics, background-work generation, or adapter transport semantics.
- The static UI remains buildless: plain HTML, CSS, and JavaScript served by the control-console FastAPI app.

## Mandatory Skills
- `development-plan`: load before moving this file into `development_plans/active/short_term/`, changing plan status, executing the plan, reviewing the plan, or reporting completion.
- `py-style`: load before editing Python production code or Python tests. Read its positive and negative constraint references before implementation.
- `test-style-and-execution`: load before creating, changing, running, or interpreting pytest tests.
- `local-llm-architecture`: load before touching any cognition, RAG, memory, prompt, graph, dialog, evaluator, reflection, self-cognition, global-growth, or background LLM behavior. This plan does not authorize prompt or LLM pipeline changes.
- `database-data-pull`: load before exporting or inspecting live MongoDB data to validate console lookup behavior. Prefer existing scripts under `src/scripts/` and keep exports out of chat text.

## Mandatory Rules
- Keep `control_console` out of the brain service. Do not mount `/console` or `/console/api` in the brain FastAPI app.
- Keep adapters as transport edges. Do not move character judgment, RAG, cognition, memory ownership, calendar semantics, or persistence decisions into adapters.
- Keep console control deterministic. Do not ask an LLM to decide service lifecycle, enablement, permissions, health state, cache state, retry behavior, process ownership, or data redaction.
- Manage only registry-declared services. Do not allow operators to submit arbitrary commands, shell fragments, environment-variable names containing secrets, working directories, or process IDs through the UI/API.
- Start processes with argv lists and `shell=False`. Do not use `shell=True`, `os.system`, command-string concatenation, or platform shell parsing.
- Stop only console-owned or registry-adopted processes. Do not scan the OS process table and kill name-matched processes.
- Stop order is dependency-aware: stop adapters and dependent services before stopping the brain; start the brain before dependent adapters unless a service spec explicitly has no dependency.
- Every start, stop, restart, crash detection, auth failure, debug chat, lookup, and privileged log access writes a sanitized audit event with operator id, reason where applicable, target, previous state, new state, timestamp, and request id.
- Console API authentication protects every `/api/*` endpoint except the minimal unauthenticated login/challenge route and static assets required to render the login page.
- Use CSRF protection or an equivalent same-origin token for state-changing browser requests.
- Redact secrets, bearer tokens, model keys, callback secrets, prompts, raw embeddings, raw environment values, and unbounded message bodies from all API responses, logs, audit events, and UI panes.
- Keep `/ops/*` aggregate and trusted-local. Do not expand existing `/ops/*` endpoints with per-user bodies, prompts, channel IDs, secrets, raw memory content dumps, or embeddings.
- Put per-user lookup, memory lookup, image/style lookup, and episode lookup under authenticated control-console endpoints only.
- Route code must call domain helpers, repository adapters, or DB-owned helper functions. Do not import raw MongoDB clients directly in route handlers.
- Console-owned local state is written through a small state-store module with atomic writes and tests. MongoDB audit mirrors are written through named DB helper functions.
- Keep realtime summary events bounded. Do not stream full conversations, full memory bodies, full image metadata, embeddings, unbounded worker logs, or complete process logs.
- Keep the ordinary brain `/chat` response path unchanged. The console must not add extra LLM calls, broad DB scans, or lifecycle checks inside normal chat processing.
- Use deterministic tests for auth, service registry validation, process lifecycle state transitions, redaction, route contracts, event/log query limits, lookup limits, and SSE event shapes.
- Use patched LLM tests only if debug-console plumbing must prove `/chat` handoff without live model variance.
- Use real LLM tests only for existing model-facing `/chat` behavior, one at a time with inspected logs, and only when a change touches prompt/model-facing behavior. This plan does not require new real LLM tests.
- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution unless the user explicitly approves fallback execution.

## Must Do
- Create a top-level `src/control_console` package with settings, contracts, auth, CSRF/session handling, service registry, process supervisor, local state store, HTTP client, log/event monitor, repository adapters, redaction, routes, SSE streaming, static assets, and audit helpers.
- Add a `kazusa-control-console` project script and update package discovery so the new top-level package is installed alongside `kazusa_ai_chatbot` and `adapters`.
- Add a default service registry for the built-in project services: `brain`, `adapter.discord`, and `adapter.debug`. The registry must support future service specs without UI code changes.
- Add a local registry override file loaded from `KAZUSA_CONTROL_SERVICE_REGISTRY`. The override must validate against a strict `ServiceSpec` schema and reject unknown fields, shell strings, missing ids, duplicate ids, unsafe environment overrides, and unbounded values.
- Implement a process supervisor that starts services, tracks PID/generation/desired state/actual state, probes health, detects crash exits, stops gracefully, escalates after timeout, restarts services, records lifecycle events, and writes bounded stdout/stderr logs.
- Implement dependency-aware lifecycle actions so adapters are not started before the brain and the brain is not stopped before managed dependent adapters are stopped.
- Add console authentication using a configured operator token hash or an equivalent existing local secret mechanism. Reject API calls when authentication fails.
- Add state-changing API routes for service start, stop, restart, and desired-state update. Every action requires an operator identity and an explicit bounded reason.
- Add read-only service-monitoring routes for current process state, health probe state, recent lifecycle events, and bounded log tails.
- Add event-log monitoring routes that read sanitized Kazusa operational events, console audit events, console errors, and service process logs through bounded queries with filters for service id, event type, level, request id, tracking id, and time window.
- Add a debug console endpoint in `control_console` that builds a valid debug `ChatRequest` and sends it to the running brain service over HTTP using the same `/chat` contract. It must return a clear brain-not-running response when the brain service is stopped or unhealthy.
- Add read-only console endpoints for latest character status, global character growth progression, user image/style lookup with episode references, group style image lookup, memory lookup, calendar schedules/runs, background-work jobs, health/cache summaries, and audit events.
- Add an overview endpoint and one bounded SSE stream under the control-console API for compact realtime status: managed service states, brain health, adapter runtime status, Cache2 summary, latest character snapshot timestamp, pending calendar count, active background jobs, recent audit/control events, and recent log/error counters.
- Add query limits, pagination, redaction, and stable Pydantic response contracts for every lookup and log endpoint.
- Add a static HTML console served by the control-console process. The UI must use a simple light dashboard layout with a left navigation rail, top status strip, service lifecycle cards, detail tables, event-log monitor, debug chat workspace, lookup pages, health/cache panels, and audit list.
- Preserve the existing brain and adapter public contracts. The brain service must not import `control_console`, and adapters must not gain cognition or persistence responsibilities.
- Add focused tests named in this plan before production implementation starts, record their expected pre-implementation failures, and rerun them after implementation.
- Update relevant docs: root README/HOWTO, brain service README, adapters README, DB README collection list, script registry notes, and local operation docs. The docs must state that normal operator startup is `kazusa-control-console`, then service lifecycle from the console.

## Deferred
- Do not mount a console inside the brain service.
- Do not implement a brain-side self-shutdown endpoint.
- Do not expose the console to the public internet.
- Do not add role-based multi-user administration beyond the single local operator identity required by this plan.
- Do not implement Docker Compose control, systemd control, Windows service control, Kubernetes control, remote host agents, or cloud orchestration.
- Do not add arbitrary command execution, web-editable service commands, web-editable environment variables, or arbitrary PID killing.
- Do not stop externally started processes that are not owned or explicitly adopted by the control console.
- Do not redesign `/ops/*` into the console API. Keep existing `/ops/*` compatibility intact.
- Do not change Kazusa prompts, persona voice, cognition graph routing, RAG routing, memory promotion semantics, reflection semantics, self-cognition semantics, calendar trigger semantics, or background-work generation behavior.
- Do not add new LLM calls for console rendering, log summaries, lookup ranking, or control decisions.
- Do not add a frontend build pipeline, React, Vue, Vite, Webpack, Node, Tailwind build tooling, telemetry vendor, or dashboard template dependency.
- Do not add write/edit/delete actions for memories, image/style records, schedules, character growth traits, background jobs, or event-log records. This console iteration reads those domains only.
- Do not migrate historical conversation, memory, image, calendar, background-work, growth, or event-log documents.

## Cutover Policy
- Introduce `kazusa-control-console` as a new top-level command without removing existing `kazusa-brain`, `kazusa-discord-adapter`, or `kazusa-debug-adapter` commands.
- Preserve existing direct commands as developer fallback and as registry command targets. Normal operator documentation points to starting only `kazusa-control-console`.
- Keep existing brain and adapter behavior unchanged when the control console is not running.
- Do not add `KAZUSA_CONSOLE_ENABLED` brain-service route gating. The console exists only as its own process.
- Bind the control console to `127.0.0.1` by default and require an operator token before any API data or lifecycle action is available.
- Store supervisor PID/state/log files under `KAZUSA_CONTROL_STATE_DIR`, defaulting to a project-local ignored runtime directory such as `.kazusa_control/`.
- Default desired state after a fresh install is stopped for all services unless the registry marks a service as `autostart=true`. The built-in default registry sets `autostart=false` for adapters and may set `autostart=false` for the brain to avoid surprising startup during tests.
- On control-console restart, recover prior desired state and restart services whose desired state is `running` and whose previous console-owned process is no longer alive.
- Use idempotent index creation for console audit mirrors in MongoDB when MongoDB is available.
- Roll out in this order: contracts/tests, settings/auth, registry/state store, process supervisor, lifecycle routes, HTTP health/debug client, event/log monitor, DB-backed lookup endpoints, SSE stream, static UI, docs, verification, independent review.

## Target State
Operators run:

```powershell
kazusa-control-console
```

Then they open the local console URL, authenticate, and manage the whole Kazusa local runtime from the top-level console. The console shows a live service table with `brain`, `adapter.discord`, `adapter.debug`, and future registry services. Each service has desired state, actual process state, PID, uptime, health, last exit code, restart count, recent logs, recent lifecycle events, and start/stop/restart controls.

Completed behavior by requested capability:
1. Brain service enable/disable: the console starts, stops, restarts, monitors, and tails logs for the brain process. The debug chat and runtime health panes clearly show unavailable state when the brain is stopped.
2. Adapter service enable/disable: the console starts, stops, restarts, monitors, and tails logs for registered adapter processes. Dependency rules start the brain before adapters and stop adapters before the brain.
3. Integrated debug console: operators send debug messages through the control console. The console builds a valid debug `ChatRequest`, sends it to the running brain `/chat` endpoint, and renders `ChatResponse`, delivery metadata, scheduled follow-ups, tracking id, latency, and error state.
4. Latest character status: the console shows current mood, global vibe, reflection summary, updated timestamp, and source descriptors from existing character-state helpers. DB-backed status remains readable when the brain is stopped and MongoDB is available.
5. Character growth progression and status: the console shows active global-growth axes, guidance, maturity, last update, run history summaries, and drift/progression indicators from global-character-growth DB helpers.
6. User image/style lookup with conversation episode: operators search by global user id, platform id, display name, or episode id. Results include redacted user-image fields, interaction-style overlays, source episode ids, timestamps, and confidence/status fields.
7. Group style image lookup: operators search by platform/group/channel identifiers and see group style overlays, source episodes, update timestamps, and status fields.
8. Memory lookup: operators search shared memory and user memory through bounded helper calls and see ids, memory names, type, authority, status, provenance labels, privacy-review flags, timestamps, and short redacted content previews.
9. Calendar schedules: operators inspect active/paused/completed/cancelled schedules and pending/running/completed/failed/cancelled/skipped runs with trigger kind, next run, last run, source, and worker status.
10. Background works: operators inspect queued, in-progress, completed, failed, delivery-in-progress, delivered, and delivery-failed background-work jobs with task brief preview, worker, timestamps, delivery state, and failure summary.
11. Cache hit status and health page coverage: operators see `/health`, Cache2 per-agent stats, DB health, service graph readiness, worker liveness from `/ops/runtime-status` when the brain is running, reflection/self-cognition stats links, event logging status, and recent resource-health events.
12. Event logs: operators search and tail sanitized operational events, lifecycle events, process logs, console audit events, and console errors from one event-monitor workspace.
13. Future services: new services are added through the registry schema and automatically appear in the service table, lifecycle API, logs, events, health probes, and overview stream.

## Design Decisions
| Topic | Decision | Rationale |
|---|---|---|
| Console placement | Implement `control_console` as a top-level service package and CLI entrypoint outside the brain service. | The console must remain available while starting, stopping, or restarting the brain. |
| Normal startup path | Operators start `kazusa-control-console`; the console starts and stops the brain/adapters. | This makes the console the management root while retaining existing commands as implementation targets and fallback. |
| Process management | Use Python `asyncio.create_subprocess_exec` with argv lists, captured stdout/stderr, PID tracking, health probes, and graceful termination. | This matches the Python/FastAPI stack, avoids new orchestration dependencies, and supports local development. |
| Service registry | Use strict Pydantic `ServiceSpec` defaults plus a JSON override file. | Future services can be added without code changes while preventing arbitrary web commands. |
| Command safety | Registry commands are argv arrays and never shell strings. UI/API cannot edit command lines. | Prevents command injection and accidental secret exposure. |
| Process ownership | Stop/restart only console-owned or registry-adopted processes with matching generation state. | Prevents killing unrelated user processes. |
| Dependency order | Start dependencies before dependents and stop dependents before dependencies. | Adapters depend on the brain; stopping the brain first causes noisier adapter failures. |
| Brain control semantics | Start/stop/restart the brain process instead of implementing a soft `/chat` intake gate. | User intent is service management, not only traffic pausing. |
| Adapter control semantics | Start/stop/restart adapter processes instead of adding adapter soft gates in v1. | Adapters are independent processes and should be lifecycle-managed by the top-level console. |
| Debug chat path | Control console sends HTTP requests to the existing brain `/chat` endpoint. | Preserves the typed brain contract and avoids a second cognition path. |
| Runtime health | Combine supervisor process state, health URLs, brain `/health`, `/ops/runtime-status`, and adapter heartbeat data. | Process running does not equal service healthy. |
| Historical/domain lookups | Use existing domain/database helpers from the console process. | Lookups remain available when the brain is stopped and avoid adding privileged brain endpoints. |
| Event log monitor | Merge bounded views of console lifecycle/audit events, process logs, and Kazusa operational events. | Operators need one place to correlate failures across service lifecycle and cognition/runtime logs. |
| Audit storage | Write local JSONL audit first and mirror to MongoDB when available through DB-owned helper functions. | Lifecycle audit survives MongoDB outages and can still be queried centrally when DB is healthy. |
| Realtime transport | Use one SSE endpoint from the control console for compact status and recent-event deltas. | FastAPI can serve one-way status updates without WebSocket complexity. |
| UI layout | Use a simple light dashboard with left rail, top strip, cards, tables, filters, and detail drawers. | The requested content is too broad for one page and needs modern, readable workspaces. |
| Frontend dependencies | Use buildless HTML/CSS/JavaScript only. | Keeps deployment simple and avoids Node/tooling risk. |
| Security boundary | Bind locally by default, require token auth, use CSRF for state-changing browser calls, and redact sensitive data. | Lifecycle control and per-user lookup data are high-sensitivity surfaces. |
| Future services | Treat every manageable process as a `ServiceSpec` with kind, command, dependencies, health probe, log policy, and autostart flag. | The console can manage later services without redesigning the service table or lifecycle API. |

## Change Surface
### New top-level package
- `src/control_console/__init__.py`: package marker and public version.
- `src/control_console/main.py`: CLI entrypoint for `kazusa-control-console`.
- `src/control_console/app.py`: FastAPI app factory, static asset mounting, router registration, lifecycle startup/shutdown hooks.
- `src/control_console/settings.py`: environment-backed settings, state directory paths, host/port, auth, registry path, brain base URL, timeouts, limits.
- `src/control_console/contracts.py`: Pydantic request/response contracts for services, lifecycle actions, logs, events, overview, debug chat, lookups, auth, and SSE projections.
- `src/control_console/auth.py`: operator token validation, session cookie handling, CSRF helpers, local-only request validation.
- `src/control_console/redaction.py`: deterministic redaction for logs, events, env summaries, prompts, secrets, embeddings, and long text.
- `src/control_console/service_registry.py`: default service specs, JSON override loading, strict schema validation, dependency graph validation.
- `src/control_console/process_store.py`: atomic local state store for desired states, PID/generation records, lifecycle event pointers, and recovery.
- `src/control_console/supervisor.py`: process start/stop/restart, signal escalation, crash detection, health polling, dependency ordering, log capture.
- `src/control_console/audit.py`: local JSONL audit writer and Mongo mirror handoff.
- `src/control_console/log_store.py`: bounded process log writer/reader using rotating files.
- `src/control_console/event_monitor.py`: merged event/log query service for console audit, process logs, console errors, and Kazusa operational events.
- `src/control_console/kazusa_client.py`: bounded HTTP client for brain `/health`, `/ops/runtime-status`, `/chat`, and related read-only runtime endpoints.
- `src/control_console/repository.py`: read-only repository adapters for character, growth, image/style, memory, calendar, background-work, cache, and event summaries.
- `src/control_console/routes.py`: authenticated API routes and static index route.
- `src/control_console/stream.py`: compact SSE stream assembler.
- `src/control_console/static/index.html`: buildless UI shell.
- `src/control_console/static/console.css`: light dashboard styling.
- `src/control_console/static/console.js`: API client, navigation, service actions, tables, filters, debug chat, SSE updates.

### Existing project metadata
- `pyproject.toml`: add `kazusa-control-console = "control_console.main:main"` and include `control_console*` in package discovery.
- `.gitignore`: ignore `.kazusa_control/`, local process logs, local PID/state files, and local audit JSONL files.

### Existing brain service files
- `src/kazusa_ai_chatbot/service.py`: do not mount console routes; preserve `/chat`, `/health`, `/ops/*`, adapter registration, and delivery contracts. Make only narrowly required changes discovered by tests for graceful termination or health accuracy.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`: preserve existing public contracts; add no console-only payloads.
- `src/kazusa_ai_chatbot/config.py`: preserve brain settings; do not add embedded-console settings.
- `src/kazusa_ai_chatbot/brain_service/README.md`: document that lifecycle management is owned by the top-level control console.

### Existing adapter files
- Discord adapter entrypoint files: preserve transport-edge behavior; make only narrowly required changes for graceful process termination and log friendliness.
- Debug adapter entrypoint files: preserve development adapter behavior; make only narrowly required changes for graceful process termination and log friendliness.
- NapCat/QQ adapter docs or registry example: document how a locally installed QQ adapter process is represented as a registry service spec when it is available in the deployment.
- `src/adapters/README.md`: document that adapters are normally started/stopped by `kazusa-control-console` and still speak the same brain service API.

### Existing DB files
- `src/kazusa_ai_chatbot/db/__init__.py`: export named console audit/event query helpers when added.
- `src/kazusa_ai_chatbot/db/bootstrap.py`: add idempotent indexes for control-console audit mirror collections.
- `src/kazusa_ai_chatbot/db/control_console_audit.py`: new DB-owned Mongo mirror operations for sanitized console audit events.
- `src/kazusa_ai_chatbot/db/event_log_queries.py`: add or extend bounded operational-event query helpers if equivalent helpers do not already exist.
- `src/kazusa_ai_chatbot/db/README.md`: document console audit mirror collection and event-log query ownership.

### Existing domain helper surfaces
- Use existing character-state helper functions for latest character state.
- Use global-character-growth helper or DB interface functions for growth traits and run summaries.
- Use interaction-style image DB helper functions for user and group style lookup.
- Use memory and user-memory helper functions for memory lookup.
- Use calendar scheduler repository functions for schedules and runs.
- Use background-work public/db helper functions for jobs.
- Use existing health, Cache2, event logging, reflection, and self-cognition helper surfaces for health/cache/event summaries.

### Tests
- `tests/test_control_console_contracts.py`
- `tests/test_control_console_auth.py`
- `tests/test_control_console_service_registry.py`
- `tests/test_control_console_process_store.py`
- `tests/test_control_console_supervisor.py`
- `tests/test_control_console_lifecycle_routes.py`
- `tests/test_control_console_kazusa_client.py`
- `tests/test_control_console_event_monitor.py`
- `tests/test_control_console_log_store.py`
- `tests/test_control_console_redaction.py`
- `tests/test_control_console_repository.py`
- `tests/test_control_console_stream.py`
- `tests/test_console_debug_chat.py`
- `tests/test_console_lookup_limits.py`

### Docs and scripts
- `README.md`: add the control-console-first local operation path.
- `docs/HOWTO.md`: add setup, env vars, local-only warning, auth, state directory, service registry, lifecycle operations, and smoke-test steps.
- `src/scripts/README.md`: reference existing commands/scripts that remain fallback and export utilities.
- `docker-compose.yml`: document control-console environment variables in comments only; do not expose a new public port in this plan.

## Data Migration
- Add local state directory `KAZUSA_CONTROL_STATE_DIR`, default `.kazusa_control/`:
  - `services.json`: atomic desired/actual service state snapshot.
  - `audit.jsonl`: sanitized lifecycle/control/debug/lookup audit events.
  - `logs/{service_id}.log`: bounded rotating stdout/stderr logs.
  - `pids/{service_id}.json`: PID, generation id, started_at, command fingerprint, and ownership metadata.
- Add MongoDB mirror collection `control_console_audit_events` when MongoDB is available:
  - `_id`: generated event id.
  - `event_type`: `auth_failed`, `service_start_requested`, `service_started`, `service_stop_requested`, `service_stopped`, `service_restart_requested`, `service_crashed`, `service_health_changed`, `debug_chat_sent`, `lookup_executed`, `log_viewed`, `console_error`.
  - `operator_id`: bounded string or `anonymous`.
  - `service_id`: bounded string or empty string.
  - `target`: bounded object with no secrets, no command env values, and no message bodies.
  - `previous_state`: bounded object for lifecycle events.
  - `new_state`: bounded object for lifecycle events.
  - `reason`: bounded string.
  - `created_at`: UTC datetime.
  - `request_id`: bounded string.
  - `source`: `local_control_console`.
- Add indexes:
  - `control_console_audit_events`: `created_at` descending.
  - `control_console_audit_events`: `event_type, created_at`.
  - `control_console_audit_events`: `service_id, created_at`.
  - `control_console_audit_events`: `operator_id, created_at`.
- The migration is idempotent index creation only. No historical conversation, memory, image, calendar, background-work, growth, or operational-event documents are rewritten.

## Contracts And Data Shapes
### Configuration
```python
class ControlConsoleSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    require_auth: bool = True
    operator_token_hash: str
    local_only: bool = True
    session_cookie_name: str = "kazusa_control_session"
    csrf_header_name: str = "x-kazusa-control-csrf"
    state_dir: Path
    service_registry_path: Path | None = None
    brain_base_url: str
    max_lookup_limit: int = 100
    max_log_lines: int = 500
    max_event_limit: int = 200
    sse_interval_seconds: float = 2.0
    default_startup_timeout_seconds: float = 30.0
    default_shutdown_timeout_seconds: float = 15.0
```

### Service registry
```python
class ServiceSpec(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    display_name: str = Field(min_length=1, max_length=80)
    kind: Literal["brain", "adapter", "worker", "support"]
    command: list[str] = Field(min_length=1, max_length=32)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list, max_length=16)
    health_url: str | None = None
    ready_match: str | None = None
    startup_timeout_seconds: float | None = None
    shutdown_timeout_seconds: float | None = None
    autostart: bool = False
    log_name: str | None = None
```

Validation rules:
- `command` is an argv list. It cannot contain shell metacharacter-only entries, empty strings, or a single command string with spaces.
- `env` values are write-only process inputs. API responses may show only redacted env key names.
- `dependencies` must reference existing service ids and must form an acyclic graph.
- Built-in specs include `brain`, `adapter.discord`, and `adapter.debug` using existing project script commands.

### Runtime service state
```python
class ServiceRuntimeState(BaseModel):
    id: str
    display_name: str
    kind: Literal["brain", "adapter", "worker", "support"]
    desired_state: Literal["running", "stopped"]
    actual_state: Literal[
        "unknown",
        "stopped",
        "starting",
        "running",
        "stopping",
        "unhealthy",
        "crashed",
        "externally_running",
    ]
    pid: int | None = None
    generation: str | None = None
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    uptime_seconds: float | None = None
    exit_code: int | None = None
    restart_count: int = 0
    health: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    last_event_at: datetime | None = None
    last_error_preview: str | None = None
    version: int
```

### Lifecycle action contracts
```python
class ServiceActionRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=240)
    expected_version: int | None = None

class ServiceActionResponse(BaseModel):
    request_id: str
    service: ServiceRuntimeState
    action: Literal["start", "stop", "restart"]
    accepted_at: datetime
    audit_event_id: str
```

Failure conditions:
- Unknown service id returns `404`.
- Missing or invalid operator session returns `401`.
- Missing CSRF token on browser state-changing requests returns `403`.
- Version mismatch returns `409` with current state.
- Attempting to stop an externally running unowned process returns `409` with `actual_state="externally_running"`.
- Dependency violation returns `409` with the blocking services listed.
- Start command validation failure returns `500` and emits `console_error`; invalid registry must fail console startup before serving lifecycle controls.

### Auth and audit
```python
class ControlConsoleOperator(BaseModel):
    operator_id: str
    authenticated_at: datetime

class ControlAuditEvent(BaseModel):
    event_id: str
    event_type: str
    operator_id: str
    service_id: str = ""
    target: dict[str, Any]
    previous_state: dict[str, Any] | None = None
    new_state: dict[str, Any] | None = None
    reason: str = ""
    created_at: datetime
    request_id: str
    source: Literal["local_control_console"] = "local_control_console"
```

### Logs and events
```python
class ProcessLogQuery(BaseModel):
    service_id: str
    level: str | None = None
    since: datetime | None = None
    limit: int = Field(default=100, ge=1, le=500)
    cursor: str | None = None

class ProcessLogLine(BaseModel):
    service_id: str
    stream: Literal["stdout", "stderr", "supervisor"]
    line: str
    created_at: datetime
    cursor: str

class OperationalEventQuery(BaseModel):
    source: Literal["all", "kazusa", "console", "process"] = "all"
    service_id: str | None = None
    event_type: str | None = None
    level: str | None = None
    request_id: str | None = None
    tracking_id: str | None = None
    since: datetime | None = None
    limit: int = Field(default=100, ge=1, le=200)
    cursor: str | None = None
```

### Overview and health
```python
class ControlConsoleOverviewResponse(BaseModel):
    generated_at: datetime
    services: list[ServiceRuntimeState]
    brain_health: dict[str, Any]
    adapter_runtime_status: dict[str, Any]
    cache2: dict[str, Any]
    character: dict[str, Any] | None
    calendar_summary: dict[str, int]
    background_work_summary: dict[str, int]
    event_summary: dict[str, int]
    recent_audit_events: list[ControlAuditEvent]
    recent_process_errors: list[ProcessLogLine]
```

### Debug chat
```python
class ConsoleDebugChatRequest(BaseModel):
    platform: Literal["debug"] = "debug"
    channel_id: str
    user_id: str
    user_display_name: str
    message_text: str = Field(min_length=1, max_length=4000)
    debug_modes: list[Literal["listen_only", "think_only", "no_remember"]] = []
    envelope_overrides: dict[str, Any] = Field(default_factory=dict)

class ConsoleDebugChatResponse(BaseModel):
    request_id: str
    brain_available: bool
    request: dict[str, Any]
    response: dict[str, Any] | None
    tracking_id: str | None
    latency_ms: int | None
    sent_at: datetime
    error: dict[str, Any] | None = None
```

### Lookup contracts
```python
class ConsoleLookupQuery(BaseModel):
    query: str = Field(default="", max_length=240)
    platform: str | None = None
    global_user_id: str | None = None
    group_id: str | None = None
    episode_id: str | None = None
    status: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = Field(default=25, ge=1, le=100)
    cursor: str | None = None

class ConsoleLookupPage(BaseModel):
    generated_at: datetime
    items: list[dict[str, Any]]
    next_cursor: str | None
    redaction: dict[str, Any]
```

### SSE events
```text
event: control.summary
data: compact ControlConsoleOverviewResponse projection

event: control.service
data: one ServiceRuntimeState after lifecycle, health, or crash change

event: control.audit
data: latest sanitized ControlAuditEvent

event: control.log
data: latest bounded process/error counter projection, not full log lines

event: control.error
data: bounded error code and request id
```

All response models use Pydantic validation, bounded strings, pagination, stable cursors, `extra="forbid"` where practical, and explicit redaction before returning data.

## LLM Call And Context Budget
- The console adds zero new LLM calls.
- The debug console uses the existing brain `/chat` HTTP contract and obeys existing debug modes.
- Read-only lookups use deterministic database/helper calls only.
- Event-log rendering, process summaries, health labels, and cache labels use deterministic aggregation only.
- No prompt, graph, RAG, cognition, dialog, evaluator, reflection, self-cognition, or background-work prompt text changes are authorized.
- No raw console telemetry is passed into character prompts.
- No service lifecycle decision is made by a prompt, model, or semantic classifier.

## Overdesign Guardrail
- Actual problem: Operators need one top-level local management console that can start/stop/restart Kazusa services, inspect runtime/domain state, run debug chat, and monitor logs/events.
- Minimal change: Add one separate FastAPI package and CLI entrypoint with a strict service registry, local subprocess supervisor, bounded logs/events, authenticated API, static UI, and existing helper-based lookups.
- Ownership boundaries: `control_console` owns UI, auth, lifecycle, process logs, audit, event monitor, HTTP probes, and orchestration. The brain owns cognition and `/chat`. Adapters own platform transport. DB/domain packages own persistence semantics and helper functions.
- Rejected complexity: No embedded brain console, no arbitrary process control, no public internet admin, no Docker/systemd/k8s manager, no remote agents, no role matrix, no domain write/edit/delete workflows, no telemetry vendor, no new LLM summaries, no prompt rewrites, and no broad observability platform.
- Evidence threshold: Add a field, endpoint, state file, audit event, route, index, or UI panel only when it maps directly to service lifecycle, event/log monitoring, one of the 11 inspection capabilities, security/audit, or a required test contract in this plan.

## Agent Autonomy Boundaries
- The responsible agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate migration strategies, arbitrary command interfaces, compatibility shims, fallback managers, extra services, or extra features.
- The responsible agent must treat changes outside `src/control_console`, `pyproject.toml`, `.gitignore`, DB helper modules, tests, and docs as high-scrutiny changes.
- The responsible agent may not mount console routes in the brain service or add console-owned imports to the brain service.
- The responsible agent may not update prompts, cognition graph logic, RAG behavior, memory promotion behavior, global-growth semantics, calendar trigger semantics, or background-work generation behavior.
- The responsible agent must search the codebase for existing helper functions before adding repository reads. If an equivalent helper exists, reuse or extract it into the appropriate owner rather than duplicating raw DB logic.
- The responsible agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, the responsible agent must preserve the plan intent and report the discrepancy.
- If a required instruction is impossible, the responsible agent must stop and report the blocker instead of inventing a substitute.

## Implementation Order
1. Parent creates or moves this plan to `development_plans/active/short_term/backend_control_console_top_level_plan.md` with status `draft`. Execution starts only after status is changed to `approved`.
2. Parent loads mandatory skills and rereads brain service, adapter, DB, event logging, calendar, background-work, global-growth, memory, RAG/interface docs, and the current command entrypoints.
3. Parent writes focused deterministic tests first:
   - `tests/test_control_console_contracts.py::test_service_contracts_reject_extra_fields_and_unbounded_strings`
   - `tests/test_control_console_service_registry.py::test_registry_rejects_shell_strings_duplicate_ids_and_dependency_cycles`
   - `tests/test_control_console_process_store.py::test_state_store_recovers_desired_state_and_generation_atomically`
   - `tests/test_control_console_supervisor.py::test_start_stop_restart_uses_argv_no_shell_and_records_audit`
   - `tests/test_control_console_lifecycle_routes.py::test_lifecycle_routes_require_auth_csrf_reason_and_version`
   - `tests/test_control_console_event_monitor.py::test_event_monitor_merges_and_redacts_bounded_sources`
   - `tests/test_control_console_redaction.py::test_responses_exclude_secrets_prompts_embeddings_env_values_and_raw_messages`
   - `tests/test_control_console_stream.py::test_summary_stream_emits_bounded_service_and_event_payload`
4. Parent runs the new focused tests and records expected failures from missing modules.
5. Parent starts the production-code subagent with this approved plan, mandatory skills, target files, focused tests, and production-code ownership boundary.
6. Production-code subagent implements `control_console.contracts`, `settings`, `redaction`, `auth`, and static app skeleton.
7. Production-code subagent implements `service_registry`, default built-in service specs, registry override loading, and dependency validation.
8. Production-code subagent implements `process_store`, `log_store`, `audit`, and DB audit mirror helpers.
9. Production-code subagent implements `supervisor` start/stop/restart, health polling, crash detection, dependency order, log capture, and recovery.
10. Parent adds integration tests for route mounting, auth, CSRF, lifecycle actions, version mismatch, dependency ordering, crash state, debug chat unavailable state, lookup limits, and SSE event shape while the production-code subagent continues production changes.
11. Production-code subagent implements API routes for service state, lifecycle actions, log tails, event monitor, overview, debug chat, and lookup pages.
12. Production-code subagent implements `kazusa_client` for brain health, runtime status, and debug `/chat` calls.
13. Production-code subagent implements read-only repository adapters for character status, global growth, user image/style, group style image, memory, calendar, background work, health/cache, and operational events.
14. Production-code subagent implements `/api/stream` with compact SSE status updates and bounded exception events.
15. Production-code subagent adds static UI assets with service lifecycle dashboard, event monitor, debug console, lookup pages, health/cache panels, and audit list.
16. Parent updates docs, script registry notes, and local-operation HOWTO.
17. Parent runs focused tests, then relevant existing regression tests for brain service, adapters, DB bootstrap, event logging, calendar, background-work, character state, global growth, and Cache2.
18. Parent runs the independent code review gate after planned implementation verification passes.
19. Parent remediates review findings with focused tests and reruns affected verification.
20. Parent records execution evidence, updates plan checklist/status, and prepares final handoff.

## Execution Model
Execution is parent-led and uses native subagent capability.

Parent responsibilities:
- Own plan lifecycle, test contract, verification, docs, execution evidence, review feedback remediation, and final sign-off.
- Add or update focused tests before production implementation.
- Run the pre-implementation tests and record expected failures.
- Run integration and regression tests after implementation.
- Keep progress checklist and execution evidence current.

Production-code subagent responsibilities:
- Edit planned production files only.
- Preserve existing cognition, RAG, prompt, memory promotion, calendar semantics, background-work semantics, brain contracts, and adapter transport ownership.
- Report changed files, commands run, blockers, residual risks, and deviations before closing.

Independent code-review subagent responsibilities:
- Review only after planned verification passes.
- Compare the full diff against this plan, mandatory rules, security boundaries, service lifecycle safety, and tests.
- Report findings without implementing fixes.

If native subagent capability is unavailable, execution stops before production implementation and the blocker is reported.

## Progress Checklist
- [ ] Plan located at `development_plans/active/short_term/backend_control_console_top_level_plan.md` with status `approved` before execution.
- [ ] Mandatory skills loaded and this plan reread.
- [ ] Focused deterministic tests added.
- [ ] Pre-implementation expected failures recorded.
- [ ] Top-level `control_console` package skeleton implemented.
- [ ] `kazusa-control-console` script and package discovery implemented.
- [ ] Console settings implemented.
- [ ] Console auth and CSRF implemented.
- [ ] Console redaction implemented.
- [ ] Service registry and default built-in specs implemented.
- [ ] Registry override validation implemented.
- [ ] Local state store implemented.
- [ ] Local log store implemented.
- [ ] Local audit writer and Mongo audit mirror implemented.
- [ ] Process supervisor implemented.
- [ ] Dependency-aware lifecycle ordering implemented.
- [ ] Lifecycle routes implemented and tested.
- [ ] Service monitor routes implemented and tested.
- [ ] Event-log monitor implemented and tested.
- [ ] Kazusa HTTP client implemented and tested.
- [ ] Debug console endpoint implemented and tested.
- [ ] Character status endpoint implemented and tested.
- [ ] Character growth endpoint implemented and tested.
- [ ] User image/style lookup implemented and tested.
- [ ] Group style image lookup implemented and tested.
- [ ] Memory lookup implemented and tested.
- [ ] Calendar schedule/run lookup implemented and tested.
- [ ] Background-work lookup implemented and tested.
- [ ] Health/cache overview implemented and tested.
- [ ] SSE summary stream implemented and tested.
- [ ] Static console UI implemented.
- [ ] Docs updated.
- [ ] Focused verification passed.
- [ ] Relevant regression verification passed.
- [ ] Manual browser smoke checks completed.
- [ ] Independent code review completed.
- [ ] Review findings resolved and affected tests rerun.
- [ ] Execution evidence recorded.

## Verification
Run these commands from the repository root using the project virtual environment.

Focused deterministic tests:
```powershell
pytest tests/test_control_console_contracts.py -q
pytest tests/test_control_console_auth.py -q
pytest tests/test_control_console_service_registry.py -q
pytest tests/test_control_console_process_store.py -q
pytest tests/test_control_console_supervisor.py -q
pytest tests/test_control_console_lifecycle_routes.py -q
pytest tests/test_control_console_kazusa_client.py -q
pytest tests/test_control_console_event_monitor.py -q
pytest tests/test_control_console_log_store.py -q
pytest tests/test_control_console_redaction.py -q
pytest tests/test_control_console_repository.py -q
pytest tests/test_control_console_stream.py -q
pytest tests/test_console_debug_chat.py -q
pytest tests/test_console_lookup_limits.py -q
```

Relevant regression tests:
```powershell
pytest tests/test_brain_service*.py -q
pytest tests/test_service*.py -q
pytest tests/test_runtime_adapter*.py -q
pytest tests/test_event_logging*.py -q
pytest tests/test_calendar*.py -q
pytest tests/test_background_work*.py -q
pytest tests/test_character_state*.py -q
pytest tests/test_global_character_growth*.py -q
pytest tests/test_cache2*.py -q
```

Static and import checks:
```powershell
python -m compileall src/control_console src/kazusa_ai_chatbot src/adapters tests
python -c "import control_console; import kazusa_ai_chatbot; import adapters"
```

Operator smoke checks against a local development environment:
```powershell
kazusa-control-console --help
kazusa-brain --help
kazusa-discord-adapter --help
kazusa-debug-adapter --help
python -m scripts.fetch_ops_status
python -m scripts.character_state_snapshot
python -m scripts.identify_user_image --help
python -m scripts.identify_group_image --help
```

Manual browser checks:
- Start only `kazusa-control-console`; verify the console UI loads on loopback and requires authentication.
- Verify `/api/*` rejects unauthenticated requests and state-changing requests without CSRF token.
- Start the brain from the console; verify process state transitions through `starting` to `running`, PID is recorded, health becomes healthy, and audit/log events are written.
- Stop the brain from the console; verify dependent managed adapters are stopped first, brain receives graceful termination, health becomes unavailable, and audit/log events are written.
- Restart the brain from the console; verify generation id changes and prior logs remain queryable.
- Start one adapter from the console; verify dependency checks require the brain to be running and adapter status appears in service monitor.
- Kill a console-owned child process outside the console; verify crash detection marks it `crashed`, records exit code, writes lifecycle event, and does not hide the failure.
- Attempt to stop an externally started process; verify the console refuses with `409` and does not kill the process.
- Send a debug-console message while the brain is running; verify response rendering, tracking id, latency, debug mode handling, and audit event.
- Send a debug-console message while the brain is stopped; verify a clear brain-not-running response and no cognition/persistence work starts.
- Run user image/style, group style, memory, calendar, background-work, health/cache, and event-log lookups; verify pagination, redaction, no embeddings, no prompts, no secrets, and no unbounded message dumps.
- Open the event monitor; verify filters by service id, event type, level, request id, tracking id, and time window.
- Leave the overview open; verify SSE summary updates service state, health, cache/event counters, and recent audit events without streaming full logs or full lookup tables.

## Independent Code Review
Run this gate after focused and regression verification passes.

Review scope:
- Confirm the console is a top-level `src/control_console` package and is not mounted or imported by the brain service.
- Confirm `kazusa-control-console` is the new normal operator entrypoint and existing brain/adapter commands remain available.
- Confirm all lifecycle actions operate only on registry-declared services and never execute shell strings or arbitrary browser-provided commands.
- Confirm stop/restart cannot kill unowned external processes.
- Confirm start/stop/restart actions write sanitized audit events with operator id, reason, target, previous state, new state, timestamp, and request id.
- Confirm dependency order starts brain before adapters and stops adapters before brain.
- Confirm debug chat uses the existing brain `/chat` contract over HTTP and returns a safe unavailable state when the brain is stopped.
- Confirm all 11 original operator inspection capabilities are implemented under the top-level console API/UI.
- Confirm event-log monitoring covers console lifecycle/audit, process logs, console errors, and Kazusa operational events through bounded queries.
- Confirm console auth protects every `/api/*` endpoint except login/static assets and state-changing routes enforce CSRF or equivalent same-origin protection.
- Confirm redaction removes secrets, prompts, embeddings, env values, raw callback secrets, raw tokens, raw message bodies in aggregate views, and unbounded text.
- Confirm domain lookup endpoints use existing public helper boundaries or new DB-owned helper functions.
- Confirm no prompt, cognition, RAG, memory promotion, calendar semantics, background-work generation, adapter transport semantics, or global-growth semantics changed.
- Confirm tests prove registry safety, process lifecycle, dependency order, auth, redaction, lookup limits, event-log bounds, SSE event shape, and route behavior.
- Confirm the static UI has no external runtime dependency and works from the control-console route.

The parent records review findings, remediation commits, rerun commands, and final reviewer sign-off in `Execution Evidence`.

## Acceptance Criteria
- `kazusa-control-console` starts a top-level local FastAPI management console outside the brain service.
- The brain service does not mount or import console routes.
- Operators can start, stop, and restart the brain process from the console.
- Operators can start, stop, and restart registered adapter processes from the console.
- Dependency order prevents adapters from starting without the brain and prevents the brain from stopping before managed adapters stop.
- Future services can be added through a validated registry spec and appear in lifecycle controls, status, logs, events, and overview without UI code changes.
- Lifecycle actions use argv arrays with no shell execution and cannot execute browser-provided commands.
- The console refuses to kill unowned external processes.
- Every privileged lifecycle, debug, lookup, log-view, auth-failure, crash, and console-error event is audited locally and mirrored to MongoDB when available.
- The service monitor shows desired state, actual state, PID, uptime, health, exit code, restart count, dependencies, recent events, and recent logs.
- Event-log monitoring supports bounded filters across console audit, process logs, console errors, and Kazusa operational events.
- Debug console sends and receives through the existing typed brain chat contract when the brain is running and returns a clear unavailable state when stopped.
- Latest character status updates through the overview stream or DB-backed refresh.
- Character growth progression/status is readable through a bounded endpoint and rendered in the UI.
- User image/style lookup includes source episode references and redacted fields.
- Group style image lookup includes source references and redacted fields.
- Memory lookup is bounded, paginated, provenance-aware, and excludes embeddings.
- Calendar schedules/runs are visible with status and due/run metadata.
- Background-work jobs are visible with worker, lifecycle, and delivery status.
- Health/cache panels cover brain `/health`, Cache2 per-agent stats, DB readiness, service graph readiness, worker liveness from `/ops/runtime-status` when the brain is running, reflection stats, self-cognition stats, event logging status, and recent resource-health events.
- SSE summary events are compact and do not stream full lookup tables, full process logs, prompts, embeddings, secrets, or sensitive bodies.
- All planned focused tests pass.
- Relevant regression tests pass or unrelated failures are documented with evidence.
- Docs explain setup, local-only warning, auth, service registry, lifecycle controls, event monitoring, and smoke checks.

## Independent Plan Review
Before changing status to `approved`, review this plan against the repository plan contract:
- Required top matter is present.
- Mandatory sections are present in the required order.
- Mandatory skills and mandatory rules are explicit.
- No unresolved implementation decisions remain.
- Must Do and Deferred boundaries are clear.
- Contracts and data shapes are concrete.
- Execution model uses parent-led native subagent execution.
- Independent code review gate is present.
- Verification commands are specific.
- The embedded-brain-console design is fully superseded by the top-level management-console design.

## Execution Evidence
No execution evidence is recorded because implementation has not started. During execution, record:
- Pre-implementation focused test failures.
- Production-code subagent changed files and command output.
- Integration and regression test command output.
- Manual browser smoke-check results.
- Independent code review findings.
- Remediation changes and rerun evidence.
- Final status change and handoff summary.

## Risks
| Risk | Mitigation | Verification |
|---|---|---|
| Command injection through service management | Registry-only argv lists, no shell execution, no browser-editable commands | Registry validation tests and code review for `shell=False` |
| Wrong process killed | Track console-owned PID/generation, refuse unowned external processes | Supervisor tests for external process refusal |
| Console unavailable after stopping brain | Console runs as separate top-level process | Manual smoke test stops brain while console remains available |
| Adapter failures during brain stop | Dependency-aware stop order stops adapters before brain | Lifecycle route and supervisor dependency tests |
| Stale service state after crash | Supervisor watches child exits and health probes | Crash detection test and manual kill smoke test |
| Sensitive-data exposure in logs/lookups | Redaction, bounded previews, no prompts/secrets/embeddings, auth | Redaction tests and independent review |
| MongoDB outage hides lifecycle audit | Local JSONL audit is source of truth; Mongo mirror is secondary | Audit tests with Mongo mirror failure |
| Event monitor overload | Bounded filters, limits, cursors, no full table streaming | Event monitor limit tests and SSE tests |
| UI sprawl | Left navigation, overview cards, detail workspaces, filters, lazy loading | Manual browser checks |
| Brain/adapters drift from direct commands | Existing commands remain fallback and are used by registry | Smoke checks for direct command help and console lifecycle |
