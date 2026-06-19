# backend_control_console_development_plan.md

## Summary
- Goal: Add a top-level `control_console` management service that sits beside the brain service and adapters, starts/stops/restarts the local configured Kazusa application services, runs a debug chat console, and monitors event logs plus character, memory, image/style, calendar, background-work, and health/cache state.
- Plan class: high_risk_migration
- Status: in_progress
- Mandatory skills: `development-plan`, `py-style`, `test-style-and-execution`, `local-llm-architecture`, `database-data-pull`
- Overall cutover strategy: Replace the embedded-brain-console design with a separate `kazusa-control-console` entrypoint and top-level `src/control_console` package; preserve the existing brain `/chat`, `/health`, `/ops/*`, adapter registration, cognition, RAG, persistence, and worker contracts while moving local application service lifecycle ownership to the console's deterministic child-process supervisor.
- Highest-risk areas: local process supervision, accidental arbitrary command execution, stopping the wrong process, leaking per-user data or secrets through logs/lookups, stale status when a child process crashes, keeping the console useful while the brain is stopped, and preserving cognition/adapter semantics.
- Acceptance criteria: Operators can start, stop, restart, inspect, and tail logs for the brain, adapters, and future local application services from the top-level console; all 11 original inspection capabilities still exist; event logs are searchable; future local services can be added through the service registry; no prompts/secrets/embeddings leak; deterministic tests and the independent code-review gate pass.

## Context
KazusaAIChatbot is a self-evolving character cognition runtime with a platform-neutral FastAPI brain service, thin adapters, typed message envelopes, MongoDB persistence, Cache2, calendar scheduling, background work, reflection, self-cognition, global character growth, and sanitized operational event logging. The brain service already exposes runtime endpoints such as `/chat`, `/health`, `/ops/runtime-status`, reflection stats, self-cognition stats, delivery receipts, and adapter registration/heartbeat routes. Adapters remain transport edges that normalize platform traffic into the brain service contract.

The previous plan placed the console inside the brain service and treated enable/disable as a soft intake or adapter gate. That design cannot satisfy the new product direction because an embedded console cannot reliably stop or restart its own host process and still remain available to turn it back on. This superseding plan treats `control_console` as the local parent process for one configured Kazusa application instance. The operator starts `kazusa-control-console`; the console starts and stops the brain, adapters, and future local frontend/backend/worker services through a local child-process supervisor.

Target architecture:

```text
Operator browser
  -> control_console FastAPI app and static UI
      -> local process supervisor
          -> brain service process
          -> adapter process: Discord
          -> adapter process: debug
          -> future local app service processes
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
- The console loads the same configured local application environment as the command-line path, including process environment values, project `.env` loading through the existing application configuration behavior, and explicit `KAZUSA_CONTROL_*` settings.
- The console has only the operating-system permissions of the user account that launched `kazusa-control-console`.
- Service lifecycle control means deterministic local child-process lifecycle: start, stop, restart, health probe, stdout/stderr tail, crash detection, and audit.
- The console manages only services declared in the local registry for this configured application instance. It never accepts arbitrary shell commands from the browser or API.
- Services started by the console are tracked through a local state directory. Processes not started by this console are shown as unmanaged conflicts or unavailable and are never stopped, restarted, adopted, or controlled by this plan.
- v1 uses Python `asyncio` subprocess management with argv lists and no shell execution. Docker, Docker Compose, systemd, Windows service control, Kubernetes, remote host management, and external instance adoption are not implemented in this plan.
- If a configured port, callback URL, or state directory appears to belong to an unrelated running process, the console reports a conflict and refuses lifecycle actions for that service until the conflict is resolved outside the console.
- The console is local/operator-scoped, binds to loopback by default, requires authentication, and is not exposed to the public internet.
- Closing the browser tab does not affect service state. Stopping the `kazusa-control-console` process gracefully stops console-owned child services by default before the console exits.
- The console can remain available when the brain is stopped. Runtime-only brain data degrades to unavailable, while DB-backed historical/lookup pages and local process logs remain usable when their dependencies are available.
- The console is not a cognition layer. It does not change prompts, graph routing, RAG, memory promotion, reflection, self-cognition, calendar semantics, background-work generation, or adapter transport semantics.
- The static UI remains buildless: plain HTML, CSS, and JavaScript served by the control-console FastAPI app.
- No standalone HTML mockup is authoritative for this plan. The previous reference mockup drifted from the production system and must be removed; production UI review is against this plan, the `src/control_console/README.md` ICD, shadcn component-family anatomy, and the running console.

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
- Manage only registry-declared child services for the current configured local application instance. Do not allow operators to submit arbitrary commands, shell fragments, environment-variable names containing secrets, working directories, process IDs, hostnames, container names, system service names, or external instance identifiers through the UI/API.
- Start processes with argv lists and `shell=False`. Do not use `shell=True`, `os.system`, command-string concatenation, or platform shell parsing.
- Stop only console-owned child processes with matching PID, generation, and command fingerprint. Do not adopt externally started processes. Do not scan the OS process table and kill name-matched processes.
- Stop order is dependency-aware: stop adapters and dependent services before stopping the brain; start the brain before dependent adapters unless a service spec explicitly has no dependency.
- On console shutdown, gracefully stop console-owned child services by default using the same dependency-aware order. Leaving child services running after console exit is deferred unless a future approved plan adds an explicit orphaning policy.
- Every start, stop, restart, crash detection, auth failure, debug chat, lookup, and privileged log access writes a sanitized audit event with operator id, reason where applicable, target, previous state, new state, timestamp, and request id.
- Console API authentication protects every `/api/*` endpoint except the minimal unauthenticated login/challenge route and static assets required to render the login page.
- Use CSRF protection or an equivalent same-origin token for state-changing browser requests.
- Redact secrets, bearer tokens, model keys, callback secrets, prompts, raw embeddings, raw environment values, and unbounded message bodies from all API responses, logs, audit events, and UI panes.
- Keep `/ops/*` aggregate and trusted-local. Do not expand existing `/ops/*` endpoints with per-user bodies, prompts, channel IDs, secrets, raw memory content dumps, or embeddings.
- Put per-user lookup, memory lookup, image/style lookup, and episode lookup under authenticated control-console endpoints only.
- Route code must call domain helpers, repository adapters, or DB-owned helper functions. Do not import raw MongoDB clients directly in route handlers.
- Console-owned local state is written through a small state-store module with atomic writes and tests. MongoDB audit mirrors are written through named DB helper functions.
- Keep realtime summary events bounded. Do not stream full conversations, full memory bodies, full image metadata, embeddings, unbounded worker logs, or complete process logs.
- Use a snapshot plus event-stream frontend data model. The page loads once, fetches `/api/bootstrap` for initial state, opens one authenticated same-origin SSE stream, updates a browser-local state store from compact events, and patches only affected UI regions.
- Do not use full-page auto refresh, broad timer polling, WebSocket, or multiple independent live streams in v1. Use bounded detail `GET` requests only when a panel is opened, filtered, paged, or invalidated by an SSE event.
- Production UI elements MUST be taken from shadcn UI component families and follow shadcn component anatomy and behavior for standard widgets. Use shadcn component families such as Sidebar, Button, Card, Badge, Table, Input, Select, Textarea, Separator, ScrollArea, Sheet/Dialog/Drawer where needed, and Field/Form-style control grouping. The agent is not allowed to make its own widget when a shadcn component pattern exists. Custom CSS is allowed only for layout glue, theme tokens, and adapting the approved buildless static delivery model; it must not create a parallel design system.
- The production console must remain Python/FastAPI served and buildless; everything in the fundamental application stack must be based on Python. Do not introduce Node.js, npm, pnpm, yarn, React, Vue, Vite, Webpack, Tailwind build tooling, a frontend package manager workflow, or any frontend build/runtime as a fundamental software stack. Any shadcn alignment must be implemented through static HTML/CSS/JavaScript assets served by the Python console process, not through a Node-based application.
- The cognition-chain diagram is a reusable read-only cognition-run graph renderer, not a page-specific mockup and not a cognition engine. The renderer receives a bounded graph snapshot from the caller; the Overview page passes the latest cognition run, the Debug chat page passes the most recent debug cognition run, and a future historical-inspection page may pass a historical run without changing the renderer.
- The cognition-chain diagram must support directed acyclic graph layout with visible parallel branches, fork/join edges, stage lanes, status badges, and hover/focus detail disclosure. A linear-only pipeline is not acceptable because L2 and supporting evidence stages can run in parallel.
- Hover/focus detail may expose bounded and redacted reasoning fields that already exist in runtime state, such as L2 `internal_monologue`, `logical_stance`, `character_intent`, `judgment_note`, resolver route summaries, selected action reasons, and final dialog/surface summaries. It must not expose raw prompts, secrets, embeddings, full message bodies, unbounded memory bodies, or hidden environment/configuration values.
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
- Add `src/control_console/README.md` as an ICD-style module README that explains the control-console interface boundary, intended operator use cases, package ownership, public CLI/API/UI/SSE contracts, auth/CSRF/local-only security model, service registry and supervisor ownership, static UI constraints, forbidden behavior, and testing expectations.
- Add a `kazusa-control-console` project script and update package discovery so the new top-level package is installed alongside `kazusa_ai_chatbot` and `adapters`.
- Add a default service registry for the built-in local application services: `brain`, `adapter.discord`, `adapter.napcat`, and `adapter.debug`. The registry must support future local frontend, backend, adapter, worker, and support service specs for this application instance without UI code changes.
- Add a local registry override file loaded from `KAZUSA_CONTROL_SERVICE_REGISTRY`. The override must validate against a strict `ServiceSpec` schema and reject unknown fields, shell strings, missing ids, duplicate ids, unsafe environment overrides, external host/process/container/service identifiers, and unbounded values.
- Implement a process supervisor that starts services, tracks PID/generation/desired state/actual state, probes health, detects crash exits, stops gracefully, escalates after timeout, restarts services, records lifecycle events, and writes bounded stdout/stderr logs.
- Implement dependency-aware lifecycle actions so adapters are not started before the brain and the brain is not stopped before managed dependent adapters are stopped.
- Add console authentication using a configured operator token hash or an equivalent existing local secret mechanism. Reject API calls when authentication fails.
- Add state-changing API routes for service start, stop, restart, and desired-state update. Every action requires an operator identity and an explicit bounded reason.
- Add read-only service-monitoring routes for current process state, health probe state, recent lifecycle events, and bounded log tails.
- Add event-log monitoring routes that read sanitized Kazusa operational events, console audit events, console errors, and service process logs through bounded queries with filters for service id, event type, level, request id, tracking id, and time window.
- Add a debug console endpoint in `control_console` that builds a valid debug `ChatRequest` and sends it to the running brain service over HTTP using the same `/chat` contract. It must return a clear brain-not-running response when the brain service is stopped or unhealthy.
- Add read-only console endpoints for latest character status, global character growth progression, user image/style lookup with episode references, group style image lookup, memory lookup, calendar schedules/runs, background-work jobs, health/cache summaries, and audit events.
- Add `/api/bootstrap` for the initial full UI snapshot: operator session, CSRF metadata, service registry summary, current service states, overview health/cache summary, latest audit/event counters, and UI capability flags.
- Add detail REST endpoints for service state, bounded log tails, event monitor queries, debug chat, and paginated domain lookups. Commands and debug sends remain authenticated `POST` requests with CSRF and audit reasons.
- Add one bounded SSE stream under the control-console API for compact realtime status and invalidation: managed service states, brain health, adapter runtime status, Cache2 summary, latest character snapshot timestamp, pending calendar count, active background jobs, recent audit/control events, recent log/error counters, and cursors for detail refetch.
- Add query limits, pagination, redaction, and stable Pydantic response contracts for every lookup and log endpoint.
- Add a static HTML console served by the control-console process. The UI must use a shadcn-style Sidebar plus inset content, sidebar-controlled subpages instead of one long scroll page, populated overview cards, service lifecycle cards, detail tables, event-log monitor, debug chat history and composer, lookup pages, health/cache panels, audit list, and Bright/Dark theme controls.
- Implement standard UI controls from shadcn component patterns only. If a standard dashboard element is needed, map it to the nearest shadcn component family before writing markup or CSS. Do not create bespoke button, card, table, input, select, textarea, sidebar, badge, dialog/sheet/drawer, tab, or form-field widgets outside shadcn-style component anatomy; the production UI is not allowed to make its own widget for common controls.
- Add a reusable cognition-run graph gadget to the static UI. It must render any caller-provided `CognitionRunGraphSnapshot`, support Overview latest-run and Debug chat latest-debug-run placements, share the same renderer code across those placements, and leave future historical-run inspection as a data-source concern rather than a built-in widget mode.
- Add bounded read-only backend projection for cognition-run graph snapshots where existing runtime response or event data makes the information available. If a run has no available graph data, return a clear `not_reported` snapshot or `None`; do not fabricate fake nodes.
- Preserve the existing brain and adapter public contracts. The brain service must not import `control_console`, and adapters must not gain cognition or persistence responsibilities.
- Add focused tests named in this plan before production implementation starts, record their expected pre-implementation failures, and rerun them after implementation.
- Update relevant docs: root README/HOWTO, brain service README, adapters README, DB README collection list, script registry notes, and local operation docs. The docs must state that normal operator startup is `kazusa-control-console`, then service lifecycle from the console.

## Deferred
- Do not mount a console inside the brain service.
- Do not implement a brain-side self-shutdown endpoint.
- Do not expose the console to the public internet.
- Do not add role-based multi-user administration beyond the single local operator identity required by this plan.
- Do not implement Docker Compose control, Docker Engine control, systemd control, Windows service control, Kubernetes control, remote host agents, cloud orchestration, or external instance adoption.
- Do not add arbitrary command execution, web-editable service commands, web-editable environment variables, or arbitrary PID killing.
- Do not stop, restart, inspect privileged internals of, or adopt externally started processes that are not child processes of the current console run.
- Do not manage other Kazusa instances, other local projects, other users' processes, host-level infrastructure, containers, operating-system services, or remote services.
- Do not redesign `/ops/*` into the console API. Keep existing `/ops/*` compatibility intact.
- Do not change Kazusa prompts, persona voice, cognition graph routing, RAG routing, memory promotion semantics, reflection semantics, self-cognition semantics, calendar trigger semantics, or background-work generation behavior.
- Do not add new LLM calls for console rendering, log summaries, lookup ranking, or control decisions.
- Do not add a frontend build pipeline, React, Vue, Vite, Webpack, Node.js, npm, pnpm, yarn, Tailwind build tooling, telemetry vendor, dashboard template dependency, or any other non-Python fundamental application stack.
- Do not implement historical cognition-run storage, historical cognition-run query APIs, or a built-in Live/History toggle inside the graph gadget in this plan. The graph gadget must be reusable by future historical views, but historical retrieval is deferred unless an existing bounded run snapshot is already available.
- Do not change cognition graph routing, prompt text, L2/L3 execution order, RAG retrieval, memory promotion, dialog rendering, or action selection to make the diagram easier to draw.
- Do not add write/edit/delete actions for memories, image/style records, schedules, character growth traits, background jobs, or event-log records. This console iteration reads those domains only.
- Do not migrate historical conversation, memory, image, calendar, background-work, growth, or event-log documents.

## Cutover Policy
- Introduce `kazusa-control-console` as a new top-level command without removing existing `kazusa-brain`, `kazusa-discord-adapter`, or `kazusa-debug-adapter` commands.
- Preserve existing direct commands as developer fallback and as registry command targets. Normal operator documentation points to starting only `kazusa-control-console`.
- Keep existing brain and adapter behavior unchanged when the control console is not running.
- Do not add `KAZUSA_CONSOLE_ENABLED` brain-service route gating. The console exists only as its own process.
- Bind the control console to `127.0.0.1` by default and require an operator token before any API data or lifecycle action is available.
- Load configuration for managed child services from the same local environment model used by direct command startup. `KAZUSA_CONTROL_*` variables configure only the console; existing Kazusa environment variables continue to configure the brain and adapters.
- Store supervisor PID/state/log files under `KAZUSA_CONTROL_STATE_DIR`, defaulting to a project-local ignored runtime directory such as `.kazusa_control/`.
- Default desired state after a fresh install is stopped for all services unless the registry marks a service as `autostart=true`. The built-in default registry sets `autostart=false` for adapters and may set `autostart=false` for the brain to avoid surprising startup during tests.
- On control-console restart, recover prior desired state and restart services whose desired state is `running` and whose previous console-owned process is no longer alive.
- If the configured service endpoint is occupied by an unmanaged process, report `conflict` and refuse start/stop/restart for that service. The operator resolves the conflict outside the console.
- On normal console shutdown, stop console-owned child services in dependency-aware order and write audit events for each stop attempt.
- Use idempotent index creation for console audit mirrors in MongoDB when MongoDB is available.
- Roll out in this order: contracts/tests, settings/auth, registry/state store, process supervisor, lifecycle routes, HTTP health/debug client, event/log monitor, DB-backed lookup endpoints, SSE stream, static UI, docs, verification, independent review.

## Target State
Operators run:

```powershell
kazusa-control-console
```

Then they open the local console URL, authenticate, and manage the current configured Kazusa local application instance from the top-level console. The console shows a live service table with `brain`, `adapter.discord`, `adapter.napcat`, `adapter.debug`, and future local registry services for this application. Each service has desired state, actual process state, PID, uptime, health, last exit code, restart count, recent logs, recent lifecycle events, and start/stop/restart controls.

Completed behavior by requested capability:
1. Brain service enable/disable: the console starts, stops, restarts, monitors, and tails logs for the brain process. The debug chat and runtime health panes clearly show unavailable state when the brain is stopped.
2. Adapter service enable/disable: the console starts, stops, restarts, monitors, and tails logs for registered adapter processes. Dependency rules start the brain before adapters and stop adapters before the brain.
3. Integrated debug console: operators send debug messages through the control console. The console builds a valid debug `ChatRequest`, sends it to the running brain `/chat` endpoint, and renders `ChatResponse`, delivery metadata, scheduled follow-ups, tracking id, latency, and error state.
4. Latest character status: the console shows current mood, global vibe, reflection summary, updated timestamp, and source descriptors from existing character-state helpers. DB-backed status remains readable when the brain is stopped and MongoDB is available.
5. Character growth progression and status: the console shows active global-growth axes, guidance, maturity, last update, run history summaries, and drift/progression indicators from global-character-growth DB helpers.
6. User image/style lookup with conversation episode: operators search by platform name plus platform user id, display name, or episode id. The browser UI must not require operators to enter internal `global_user_id` values; the console resolves platform identity read-only before calling scoped helpers. Results include redacted user-image fields, interaction-style overlays, source episode ids, timestamps, and confidence/status fields.
7. Group style image lookup: operators search by platform/group/channel identifiers and see group style overlays, source episodes, update timestamps, and status fields.
8. Memory lookup: operators search shared memory and user memory through bounded helper calls using platform-facing identity in the UI. User-scoped memory pages resolve `platform + platform_user_id` read-only inside the console before querying global-id keyed helpers. Results show ids, memory names, type, authority, status, provenance labels, privacy-review flags, timestamps, and short redacted content previews.
9. Calendar schedules: operators inspect active/paused/completed/cancelled schedules and pending/running/completed/failed/cancelled/skipped runs with trigger kind, next run, last run, source, and worker status.
10. Background works: operators inspect queued, in-progress, completed, failed, delivery-in-progress, delivered, and delivery-failed background-work jobs with task brief preview, worker, timestamps, delivery state, and failure summary.
11. Cache hit status and health page coverage: operators see `/health`, Cache2 per-agent stats, DB health, service graph readiness, worker liveness from `/ops/runtime-status` when the brain is running, reflection/self-cognition stats links, event logging status, and recent resource-health events.
12. Event logs: operators search and tail sanitized operational events, lifecycle events, process logs, console audit events, and console errors from one event-monitor workspace.
13. Future local services: new frontend, backend, adapter, worker, or support services for this configured application are added through the registry schema and automatically appear in the service table, lifecycle API, logs, events, health probes, and overview stream.

## Design Decisions
| Topic | Decision | Rationale |
|---|---|---|
| Console placement | Implement `control_console` as a top-level service package and CLI entrypoint outside the brain service. | The console must remain available while starting, stopping, or restarting the brain. |
| Normal startup path | Operators start `kazusa-control-console`; the console starts and stops the brain/adapters/local app services. | This makes the console the local application management root while retaining existing commands as implementation targets and fallback. |
| Permission model | The console has only the operating-system permissions of the user that launched `kazusa-control-console`. | This keeps console authority understandable and avoids hidden elevation or host-level administration. |
| Instance boundary | The console manages exactly one configured local application instance using the shared local environment and `KAZUSA_CONTROL_*` settings. | The user explicitly does not want this console managing other instances or host-level infrastructure. |
| Process management | Use Python `asyncio.create_subprocess_exec` with argv lists, captured stdout/stderr, PID tracking, health probes, and graceful termination. | This matches the Python/FastAPI stack, avoids new orchestration dependencies, and supports local development. |
| Service registry | Use strict Pydantic `ServiceSpec` defaults plus a JSON override file for the configured local application. | Future local app services can be added without code changes while preventing arbitrary web commands. |
| Command safety | Registry commands are argv arrays and never shell strings. UI/API cannot edit command lines. | Prevents command injection and accidental secret exposure. |
| Process ownership | Stop/restart only console-owned child processes with matching PID, generation state, and command fingerprint. | Prevents killing unrelated user processes and enforces the no-external-instance boundary. |
| External conflicts | Report configured ports or endpoints occupied by unmanaged processes as conflicts. | The console should help diagnose conflicts without adopting or managing other processes. |
| Console shutdown | Gracefully stop console-owned child services by default in dependency-aware order. | This avoids orphaning services that were started only through the console. |
| Dependency order | Start dependencies before dependents and stop dependents before dependencies. | Adapters depend on the brain; stopping the brain first causes noisier adapter failures. |
| Brain control semantics | Start/stop/restart the brain process instead of implementing a soft `/chat` intake gate. | User intent is service management, not only traffic pausing. |
| Adapter control semantics | Start/stop/restart adapter processes instead of adding adapter soft gates in v1. | Adapters are independent processes and should be lifecycle-managed by the top-level console. |
| Debug chat path | Control console sends HTTP requests to the existing brain `/chat` endpoint. | Preserves the typed brain contract and avoids a second cognition path. |
| Runtime health | Combine supervisor process state, health URLs, brain `/health`, `/ops/runtime-status`, and adapter heartbeat data. | Process running does not equal service healthy. |
| Historical/domain lookups | Use existing domain/database helpers from the console process. | Lookups remain available when the brain is stopped and avoid adding privileged brain endpoints. |
| Event log monitor | Merge bounded views of console lifecycle/audit events, process logs, and Kazusa operational events. | Operators need one place to correlate failures across service lifecycle and cognition/runtime logs. |
| Audit storage | Write local JSONL audit first and mirror to MongoDB when available through DB-owned helper functions. | Lifecycle audit survives MongoDB outages and can still be queried centrally when DB is healthy. |
| Frontend data flow | Use REST snapshots/actions/detail queries plus one SSE stream. | The page stays loaded, initial state is explicit, commands remain auditable `POST` requests, and backend changes reach the UI without full-page refresh. |
| Initial snapshot | Serve `/api/bootstrap` as the first UI data fetch after authentication. | The browser receives a coherent baseline before applying incremental stream updates. |
| Realtime transport | Use one authenticated same-origin SSE endpoint from the control console for compact status, invalidation, and recent-event deltas. | Console updates are mostly backend-to-browser; SSE fits this better than WebSocket and is simpler to secure with same-origin session cookies. |
| Detail refresh | Stream compact cursors or invalidation events, then let active panes fetch bounded detail over REST. | Prevents full logs, lookup pages, message bodies, and large tables from becoming an unbounded stream. |
| WebSocket and polling | Defer WebSocket and broad timer polling. | V1 does not need continuous bidirectional transport, and page auto-refresh or global polling would be noisy and less precise. |
| UI layout | Use shadcn-style sidebar, inset content, sidebar-controlled subpages, card/table/form composition, populated overview, debug chat history, lookup pages, health/cache, and audit workspaces. | The requested content is too broad for one scroll page and needs modern, readable workspaces that are checked against the plan and running UI rather than a stale static mockup. |
| UI component source | Use shadcn UI component families and anatomy for every standard control and surface. Do not invent bespoke widgets for buttons, cards, tables, inputs, selects, textareas, badges, sidebar navigation, dialogs/sheets/drawers, tabs, or form fields. | Keeps the production UI aligned with the requested shadcn design language and avoids a private, inconsistent component framework. |
| Cognition graph renderer | Add one reusable static-JavaScript renderer for `CognitionRunGraphSnapshot`; Overview passes the latest cognition run, Debug chat passes the most recent debug cognition run, and future historical pages pass their own snapshot. | The user wants the same graph gadget reused across live, debug, and future historical contexts without hard-coding mode switches into the widget. |
| Cognition graph layout | Render the cognition snapshot as a DAG with stage lanes, columns, fork/join edges, and clear parallel branch positioning. | The graph must make L2 and evidence parallelism visible; a simple linear pipeline hides important execution shape. |
| Cognition graph reasoning detail | Use shadcn-style HoverCard/Sheet/Dialog anatomy around graph nodes for bounded reasoning details such as L2 internal monologue and selected route/action reasons. | Operators need to inspect why a run moved through each stage without exposing prompts, secrets, embeddings, or unbounded bodies. |
| Frontend dependencies | Use buildless HTML/CSS/JavaScript served by Python/FastAPI only. Do not add Node.js, package-manager workflows, React/Vue, Vite/Webpack, Tailwind build tooling, or any frontend runtime/build stack. | Keeps deployment simple, Python-based, and avoids introducing a foreign fundamental stack. |
| Security boundary | Bind locally by default, require token auth, use CSRF for state-changing browser calls, and redact sensitive data. | Lifecycle control and per-user lookup data are high-sensitivity surfaces. |
| Future local services | Treat every manageable local application child process as a `ServiceSpec` with kind, command, dependencies, health probe, log policy, and autostart flag. | The console can manage later frontend/backend/worker services for this app without redesigning the service table or lifecycle API. |

## Change Surface
### New top-level package
- `src/control_console/__init__.py`: package marker and public version.
- `src/control_console/README.md`: ICD-style module contract covering document control, purpose, intended use cases, interface boundary, public entrypoints, API/UI/SSE contracts, security model, service lifecycle ownership, data/redaction constraints, shadcn/Python buildless UI constraints, forbidden behavior, and testing expectations.
- `src/control_console/main.py`: CLI entrypoint for `kazusa-control-console`.
- `src/control_console/app.py`: FastAPI app factory, static asset mounting, router registration, lifecycle startup/shutdown hooks.
- `src/control_console/settings.py`: environment-backed settings, shared local app environment loading, state directory paths, host/port, auth, registry path, brain base URL, timeouts, limits.
- `src/control_console/contracts.py`: Pydantic request/response contracts for bootstrap, services, lifecycle actions, logs, events, overview, debug chat, cognition graph snapshots, lookups, auth, and SSE projections.
- `src/control_console/auth.py`: operator token validation, session cookie handling, CSRF helpers, local-only request validation.
- `src/control_console/redaction.py`: deterministic redaction for logs, events, env summaries, prompts, secrets, embeddings, and long text.
- `src/control_console/service_registry.py`: default service specs, JSON override loading, strict schema validation, dependency graph validation.
- `src/control_console/process_store.py`: atomic local state store for desired states, PID/generation records, lifecycle event pointers, and recovery.
- `src/control_console/supervisor.py`: process start/stop/restart, signal escalation, crash detection, health polling, dependency ordering, log capture.
- `src/control_console/audit.py`: local JSONL audit writer and Mongo mirror handoff.
- `src/control_console/log_store.py`: bounded process log writer/reader using rotating files.
- `src/control_console/event_monitor.py`: merged event/log query service for console audit, process logs, console errors, and Kazusa operational events.
- `src/control_console/kazusa_client.py`: bounded HTTP client for brain `/health`, `/ops/runtime-status`, `/chat`, and related read-only runtime endpoints.
- `src/control_console/repository.py`: read-only repository adapters for character, growth, image/style, memory, calendar, background-work, cache, event summaries, and any existing bounded cognition-run snapshot source.
- `src/control_console/routes.py`: authenticated bootstrap, REST API routes, cognition graph snapshot projection routes where needed, SSE route, and static index route.
- `src/control_console/stream.py`: compact SSE stream assembler.
- `src/control_console/static/index.html`: buildless UI shell.
- `src/control_console/static/console.css`: light dashboard styling.
- `src/control_console/static/console.js`: API client, browser-local state store, navigation, service actions, tables, filters, debug chat, reusable cognition-run graph renderer, SSE updates, invalidation handling, and bounded detail refetch.

### Removed obsolete reference artifact
- Delete `development_plans/reference/designs/backend_control_console_mockup.html`. It is no longer a valid design reference because it drifted from the production console and should not be used for implementation review.

### Existing project metadata
- `pyproject.toml`: add `kazusa-control-console = "control_console.main:main"`, include `control_console*` in package discovery, and package static assets under `src/control_console/static/`.
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
- `tests/test_control_console_cognition_graph.py`
- `tests/test_console_debug_chat.py`
- `tests/test_console_lookup_limits.py`

### Docs and scripts
- `README.md`: add the control-console-first local operation path.
- `docs/HOWTO.md`: add setup, env vars, local-only warning, auth, state directory, service registry, lifecycle operations, and smoke-test steps.
- `src/scripts/README.md`: reference existing commands/scripts that remain fallback and export utilities.
- `docker-compose.yml`: document control-console environment variables in comments only; do not expose a new public port or container-management flow in this plan.

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
    app_instance_id: str = "local"
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
    kind: Literal["backend", "frontend", "adapter", "worker", "support"]
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
- `env` values are write-only process inputs layered on top of the shared local application environment. API responses may show only redacted env key names.
- `ServiceSpec` cannot contain host-management identifiers such as process ids, container ids, system service names, Kubernetes object names, remote hosts, SSH targets, or Docker Compose project names.
- `cwd`, when present, must resolve inside the repository or an explicitly configured local application path. Registry entries cannot point lifecycle actions at arbitrary host directories.
- `dependencies` must reference existing service ids and must form an acyclic graph.
- Built-in specs include `brain`, `adapter.discord`, `adapter.napcat`, and `adapter.debug` using existing project script commands. The built-in `brain` service uses `kind="backend"`.

### Runtime service state
```python
class ServiceRuntimeState(BaseModel):
    id: str
    display_name: str
    kind: Literal["backend", "frontend", "adapter", "worker", "support"]
    desired_state: Literal["running", "stopped"]
    actual_state: Literal[
        "unknown",
        "stopped",
        "starting",
        "running",
        "stopping",
        "unhealthy",
        "crashed",
        "conflict",
        "unavailable",
    ]
    pid: int | None = None
    generation: str | None = None
    command_fingerprint: str | None = None
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
- Attempting to stop or restart a service whose endpoint is occupied by an unmanaged process returns `409` with `actual_state="conflict"`.
- Attempting to start a service whose configured port, callback URL, state file, or command fingerprint conflicts with an unmanaged local process returns `409` with `actual_state="conflict"`.
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
    latest_cognition_graph: "CognitionRunGraphSnapshot | None"
    calendar_summary: dict[str, int]
    background_work_summary: dict[str, int]
    event_summary: dict[str, int]
    recent_audit_events: list[ControlAuditEvent]
    recent_process_errors: list[ProcessLogLine]
```

### Cognition run graph snapshot
```python
class CognitionRunGraphNode(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.:-]{0,79}$")
    label: str = Field(min_length=1, max_length=80)
    stage: str = Field(min_length=1, max_length=40)
    lane: Literal[
        "input",
        "evidence",
        "context",
        "cognition",
        "decision",
        "surface",
        "persistence",
        "runtime",
    ]
    column: int = Field(ge=0, le=32)
    branch: str | None = Field(default=None, max_length=60)
    status: Literal[
        "queued",
        "running",
        "completed",
        "skipped",
        "failed",
        "not_reported",
    ]
    summary: str = Field(default="", max_length=240)
    reasoning_title: str | None = Field(default=None, max_length=80)
    reasoning_preview: str | None = Field(default=None, max_length=500)
    detail: dict[str, Any] = Field(default_factory=dict)

class CognitionRunGraphEdge(BaseModel):
    source: str
    target: str
    label: str = Field(default="", max_length=80)
    status: Literal[
        "running",
        "completed",
        "skipped",
        "failed",
        "not_reported",
    ] = "completed"

class CognitionRunGraphSnapshot(BaseModel):
    run_id: str | None = Field(default=None, max_length=120)
    source: Literal["overview_latest", "debug_latest", "historical"]
    generated_at: datetime
    status: Literal["running", "completed", "failed", "not_reported"]
    nodes: list[CognitionRunGraphNode] = Field(default_factory=list, max_length=64)
    edges: list[CognitionRunGraphEdge] = Field(default_factory=list, max_length=96)
    redaction: dict[str, Any] = Field(default_factory=dict)
```

Rules:
- `source` identifies who supplied the snapshot; the renderer must not contain its own live/history selector.
- `column`, `lane`, and `branch` are layout hints that let the buildless renderer show parallel branches without inferring cognition semantics from labels.
- Snapshot projection may include available L2 and downstream fields such as `internal_monologue`, `logical_stance`, `character_intent`, `judgment_note`, resolver summaries, action reasons, final dialog summaries, and persistence outcomes only after bounding and redaction.
- Snapshot projection must omit or redact raw prompts, hidden system/developer instructions, secrets, tokens, embeddings, full message bodies, raw memory documents, and unbounded event payloads.
- If a run cannot be observed through existing bounded runtime/debug data, the API returns `None` or a `status="not_reported"` snapshot; the UI shows a clear unavailable state rather than dummy graph content.

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
    cognition_graph: CognitionRunGraphSnapshot | None
    latency_ms: int | None
    sent_at: datetime
    error: dict[str, Any] | None = None
```

### Lookup contracts
```python
class ConsoleLookupQuery(BaseModel):
    query: str = Field(default="", max_length=240)
    platform: str | None = None
    platform_user_id: str | None = None
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

### Bootstrap contract
```python
class ControlConsoleBootstrapResponse(BaseModel):
    generated_at: datetime
    operator: dict[str, Any]
    csrf_header_name: str
    services: list[ServiceRuntimeState]
    overview: dict[str, Any]
    latest_cognition_graph: CognitionRunGraphSnapshot | None
    recent_audit_events: list[dict[str, Any]]
    event_counters: dict[str, int]
    ui_capabilities: dict[str, bool]
    stream_url: str = "/api/stream"
```

The browser must fetch this snapshot once after authentication and before
opening the SSE stream. If the SSE stream reports a gap or reconnects after the
server can no longer replay recent events, the browser refetches
`/api/bootstrap` and replaces its local state baseline.

### SSE events
```text
event: control.overview
data: compact ControlConsoleOverviewResponse projection

event: control.service
data: one ServiceRuntimeState after lifecycle, health, or crash change

event: control.audit
data: latest sanitized ControlAuditEvent

event: control.log
data: service id plus log cursor and bounded process/error counter projection,
      not full log lines

event: control.lookup_invalidated
data: lookup namespace, affected filters or ids, and refresh cursor

event: control.cognition_graph_invalidated
data: graph source, run id if available, and refresh cursor; not full reasoning

event: control.error
data: bounded error code and request id

event: control.gap
data: reason and latest available event id; browser must refetch bootstrap

event: control.heartbeat
data: generated_at and latest event id
```

SSE messages must include monotonically increasing event ids when practical so
the browser can resume with `Last-Event-ID`. The server may replay only a
bounded recent window. When replay is not possible, it sends `control.gap`
instead of attempting an unbounded catch-up stream.

The SSE stream is read-only. Lifecycle actions, debug chat sends, filter
changes, pagination, and detail refreshes use authenticated REST requests.
Native `EventSource` uses same-origin session cookies; state-changing REST
requests still require CSRF validation.

All response models use Pydantic validation, bounded strings, pagination, stable cursors, `extra="forbid"` where practical, and explicit redaction before returning data.

## LLM Call And Context Budget
- The console adds zero new LLM calls.
- The debug console uses the existing brain `/chat` HTTP contract and obeys existing debug modes.
- Read-only lookups use deterministic database/helper calls only.
- Event-log rendering, process summaries, health labels, and cache labels use deterministic aggregation only.
- Cognition graph snapshots use deterministic projection of already-available run/debug state only; they do not request, rerun, summarize, or reinterpret cognition through an LLM.
- No prompt, graph, RAG, cognition, dialog, evaluator, reflection, self-cognition, or background-work prompt text changes are authorized.
- No raw console telemetry is passed into character prompts.
- No service lifecycle decision is made by a prompt, model, or semantic classifier.

## Overdesign Guardrail
- Actual problem: Operators need one top-level local management console that can start/stop/restart Kazusa services, inspect runtime/domain state, run debug chat, and monitor logs/events.
- Minimal change: Add one separate FastAPI package and CLI entrypoint with a strict local-app service registry, local child-process supervisor, bounded logs/events, authenticated API, static UI, and existing helper-based lookups.
- Ownership boundaries: `control_console` owns UI, auth, lifecycle, process logs, audit, event monitor, HTTP probes, and orchestration. The brain owns cognition and `/chat`. Adapters own platform transport. DB/domain packages own persistence semantics and helper functions.
- Rejected complexity: No embedded brain console, no arbitrary process control, no public internet admin, no Docker/systemd/k8s manager, no remote agents, no external instance management, no process adoption, no role matrix, no domain write/edit/delete workflows, no telemetry vendor, no new LLM summaries, no prompt rewrites, and no broad observability platform.
- Frontend rejected complexity: No full-page auto refresh, broad timer polling, WebSocket, GraphQL subscription layer, frontend build framework, client-side persistence database, external graph framework, built-in graph history selector, historical cognition-run store, or unbounded live log table in v1.
- UI exception: A small data-driven SVG/HTML renderer is allowed for the cognition-chain graph because shadcn does not provide a graph primitive. All controls, containers, overlays, forms, navigation, tables, badges, and empty states around that graph must still follow shadcn component-family anatomy.
- Evidence threshold: Add a field, endpoint, state file, audit event, route, index, or UI panel only when it maps directly to service lifecycle, event/log monitoring, cognition-run graph inspection, one of the 11 inspection capabilities, security/audit, or a required test contract in this plan.

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
1. Parent confirms this plan is located at `development_plans/active/short_term/backend_control_console_development_plan.md` with status `approved` or `in_progress` before execution continues.
2. Parent loads mandatory skills and rereads brain service, adapter, DB, event logging, calendar, background-work, global-growth, memory, RAG/interface docs, and the current command entrypoints.
3. Parent writes focused deterministic tests first:
   - `tests/test_control_console_contracts.py::test_service_contracts_reject_extra_fields_and_unbounded_strings`
   - `tests/test_control_console_service_registry.py::test_registry_rejects_shell_strings_external_identifiers_duplicate_ids_and_dependency_cycles`
   - `tests/test_control_console_process_store.py::test_state_store_recovers_desired_state_and_generation_atomically`
   - `tests/test_control_console_supervisor.py::test_start_stop_restart_uses_argv_no_shell_and_records_audit`
   - `tests/test_control_console_lifecycle_routes.py::test_lifecycle_routes_require_auth_csrf_reason_and_version`
   - `tests/test_control_console_event_monitor.py::test_event_monitor_merges_and_redacts_bounded_sources`
   - `tests/test_control_console_redaction.py::test_responses_exclude_secrets_prompts_embeddings_env_values_and_raw_messages`
   - `tests/test_control_console_bootstrap.py::test_bootstrap_returns_initial_state_session_csrf_services_and_stream_url`
   - `tests/test_control_console_stream.py::test_summary_stream_emits_bounded_service_event_cursor_and_gap_payload`
   - `tests/test_control_console_cognition_graph.py::test_cognition_graph_snapshot_bounds_parallel_branches_and_redacts_reasoning_detail`
   - `tests/test_control_console_cognition_graph.py::test_overview_and_debug_chat_use_same_graph_snapshot_contract_without_dummy_nodes`
4. Parent runs the new focused tests and records expected failures from missing modules.
5. Parent starts the production-code subagent with this approved plan, mandatory skills, target files, focused tests, and production-code ownership boundary.
6. Production-code subagent implements `control_console.contracts`, `settings`, `redaction`, `auth`, and static app skeleton.
7. Production-code subagent implements `service_registry`, default built-in service specs, registry override loading, and dependency validation.
8. Production-code subagent implements `process_store`, `log_store`, `audit`, and DB audit mirror helpers.
9. Production-code subagent implements `supervisor` start/stop/restart, health polling, crash detection, dependency order, log capture, and recovery.
10. Parent adds integration tests while the production-code subagent continues production changes:
    - `tests/test_control_console_auth.py::test_login_sets_session_and_csrf_and_rejects_bad_tokens`
    - `tests/test_control_console_lifecycle_routes.py::test_lifecycle_routes_reject_version_mismatch_and_apply_dependency_order`
    - `tests/test_control_console_lifecycle_routes.py::test_unmanaged_port_conflict_returns_409_without_adoption`
    - `tests/test_control_console_supervisor.py::test_crash_detection_marks_service_crashed_and_preserves_logs`
    - `tests/test_console_debug_chat.py::test_debug_chat_returns_brain_unavailable_without_cognition_when_stopped`
    - `tests/test_console_lookup_limits.py::test_lookup_routes_enforce_pagination_redaction_and_no_embeddings`
    - `tests/test_control_console_stream.py::test_stream_gap_forces_bootstrap_refetch`
11. Production-code subagent implements API routes for bootstrap, service state, lifecycle actions, log tails, event monitor, overview, debug chat, and lookup pages.
12. Production-code subagent implements `kazusa_client` for brain health, runtime status, and debug `/chat` calls.
13. Production-code subagent implements read-only repository adapters for character status, global growth, user image/style, group style image, memory, calendar, background work, health/cache, and operational events.
14. Production-code subagent implements `/api/stream` with compact SSE status updates, invalidation events, event ids, bounded replay or gap handling, and bounded exception events.
15. Production-code subagent implements cognition-run graph snapshot contracts and deterministic projection for Overview latest run and Debug chat latest debug run where existing bounded runtime/debug data is available. Missing graph data returns `None` or `status="not_reported"` instead of dummy nodes.
16. Production-code subagent adds static UI assets using shadcn-style component anatomy for standard UI elements, with service lifecycle dashboard, reusable cognition-run graph renderer, event monitor, debug console with history, lookup pages, health/cache panels, audit list, one-time bootstrap loading, browser-local state updates, and EventSource reconnect handling.
17. Production-code subagent adds `src/control_console/README.md` in ICD style, aligned with existing subsystem README documents, covering interface boundary, intended use cases, public entrypoints, REST/SSE/static UI contracts, cognition graph gadget contract, security, forbidden behavior, and testing expectations.
18. Parent updates root docs, script registry notes, and local-operation HOWTO.
19. Parent runs focused tests, then relevant existing regression tests for brain service, adapters, DB bootstrap, event logging, calendar, background-work, character state, global growth, and Cache2.
20. Parent manually inspects the running UI against this plan and the `src/control_console/README.md` ICD; verify arrangement, shadcn component anatomy, density, colors, page population, debug chat history, and cognition graph behavior with parallel branches and hover/focus reasoning detail where real data is available.
21. Parent runs the independent code review gate after planned implementation verification passes and after the plan/ICD UI inspection is recorded.
22. Parent remediates review findings with focused tests and reruns affected verification.
23. Parent records execution evidence, updates plan checklist/status, removes the obsolete mockup artifact if it still exists, and prepares final handoff.

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
Each checkbox below is a checkpoint. Before ticking any item, the active agent
must record the changed files, exact verification command or manual check,
result, next unchecked checkpoint, and `<agent/date>` sign-off in `Execution
Evidence`. The checkpoint inherits its detailed target from the matching
`Implementation Order`, `Verification`, `Manual browser checks`, or
`Independent Code Review` item; if that mapping is unclear, leave the checkbox
unchecked and update the plan before proceeding.

- [x] Plan located at `development_plans/active/short_term/backend_control_console_development_plan.md` with status `approved` before execution.
- [x] Mandatory skills loaded and this plan reread.
- [x] Focused deterministic tests added.
- [x] Pre-implementation expected failures recorded.
- [x] Top-level `control_console` package skeleton implemented.
- [x] `src/control_console/README.md` ICD added and reviewed against the control-console interface boundary and intended operator use cases.
- [x] `kazusa-control-console` script and package discovery implemented.
- [x] Console settings implemented.
- [x] Console auth and CSRF implemented.
- [x] Console redaction implemented.
- [x] Service registry and default built-in specs implemented.
- [x] Registry override validation implemented.
- [x] Local state store implemented.
- [x] Local log store implemented.
- [ ] Local audit writer and Mongo audit mirror implemented.
- [x] Process supervisor implemented.
- [x] Dependency-aware lifecycle ordering implemented.
- [x] Lifecycle routes implemented and tested.
- [ ] Service monitor routes implemented and tested.
- [x] Bootstrap route implemented and tested.
- [ ] Event-log monitor implemented and tested.
- [x] Kazusa HTTP client implemented and tested.
- [x] Debug console endpoint implemented and tested.
- [x] Character status endpoint implemented and tested.
- [x] Character growth endpoint implemented and tested.
- [ ] User image/style lookup implemented and tested.
- [ ] Group style image lookup implemented and tested.
- [x] Memory lookup implemented and tested.
- [ ] Calendar schedule/run lookup implemented and tested.
- [ ] Background-work lookup implemented and tested.
- [ ] Health/cache overview implemented and tested.
- [x] SSE summary stream implemented and tested.
- [x] Static UI bootstrap, local state store, EventSource updates, and bounded detail refetch implemented.
- [x] Static console UI implemented with shadcn-style component anatomy and no bespoke standard widgets.
- [x] Running console UI previously inspected for shadcn-style layout, populated pages, Bright/Dark themes, and no horizontal overflow.
- [x] Cognition-run graph snapshot contract/projection implemented and tested.
- [x] Reusable cognition-run graph renderer implemented for Overview latest-run and Debug chat latest-debug-run placements.
- [x] Cognition-run graph manually inspected for visible parallel branch layout, hover/focus reasoning detail, and clear unavailable/not_reported state.
- [x] Obsolete standalone mockup artifact removed from `development_plans/reference/designs`.
- [x] Python/FastAPI buildless frontend constraint verified with no Node.js, npm, pnpm, yarn, React, Vue, Vite, Webpack, Tailwind build tooling, or frontend build/runtime stack added.
- [x] Docs updated.
- [x] Focused verification passed.
- [x] Relevant regression verification passed.
- [x] Manual browser smoke checks completed.
- [x] Independent code review completed.
- [ ] Review findings resolved and affected tests rerun.
- [x] Execution evidence recorded.

## Verification
Run these commands from the repository root using the project virtual environment.

Focused deterministic tests:
```powershell
venv\Scripts\python -m pytest tests/test_control_console_contracts.py -q
venv\Scripts\python -m pytest tests/test_control_console_auth.py -q
venv\Scripts\python -m pytest tests/test_control_console_service_registry.py -q
venv\Scripts\python -m pytest tests/test_control_console_process_store.py -q
venv\Scripts\python -m pytest tests/test_control_console_supervisor.py -q
venv\Scripts\python -m pytest tests/test_control_console_lifecycle_routes.py -q
venv\Scripts\python -m pytest tests/test_control_console_kazusa_client.py -q
venv\Scripts\python -m pytest tests/test_control_console_bootstrap.py -q
venv\Scripts\python -m pytest tests/test_control_console_event_monitor.py -q
venv\Scripts\python -m pytest tests/test_control_console_log_store.py -q
venv\Scripts\python -m pytest tests/test_control_console_redaction.py -q
venv\Scripts\python -m pytest tests/test_control_console_repository.py -q
venv\Scripts\python -m pytest tests/test_control_console_stream.py -q
venv\Scripts\python -m pytest tests/test_control_console_cognition_graph.py -q
venv\Scripts\python -m pytest tests/test_console_debug_chat.py -q
venv\Scripts\python -m pytest tests/test_console_lookup_limits.py -q
```

Relevant regression tests:
```powershell
venv\Scripts\python -m pytest tests/test_brain_service*.py -q
venv\Scripts\python -m pytest tests/test_service*.py -q
venv\Scripts\python -m pytest tests/test_runtime_adapter*.py -q
venv\Scripts\python -m pytest tests/test_event_logging*.py -q
venv\Scripts\python -m pytest tests/test_calendar*.py -q
venv\Scripts\python -m pytest tests/test_background_work*.py -q
venv\Scripts\python -m pytest tests/test_character_state*.py -q
venv\Scripts\python -m pytest tests/test_global_character_growth*.py -q
venv\Scripts\python -m pytest tests/test_cache2*.py -q
```

Static and import checks:
```powershell
venv\Scripts\python -m compileall src/control_console src/kazusa_ai_chatbot src/adapters tests
venv\Scripts\python -c "import control_console; import kazusa_ai_chatbot; import adapters"
```

Documentation contract checks:
```powershell
$doc = Get-Content -LiteralPath 'src/control_console/README.md' -Raw
$required = @(
  'Control Console Interface Control Document',
  'Interface boundary',
  'Intended Use Cases',
  'Public Interfaces',
  'Security Model',
  'Forbidden Behavior',
  'Testing Expectations',
  'shadcn',
  'cognition graph',
  'Python/FastAPI'
)
foreach ($term in $required) {
  if (-not $doc.Contains($term)) {
    throw "missing required control-console README section or term: $term"
  }
}
```

Operator smoke checks against a local development environment:
```powershell
venv\Scripts\kazusa-control-console.exe --help
venv\Scripts\kazusa-brain.exe --help
venv\Scripts\kazusa-discord-adapter.exe --help
venv\Scripts\kazusa-debug-adapter.exe --help
venv\Scripts\python -m scripts.fetch_ops_status
venv\Scripts\python -m scripts.character_state_snapshot
venv\Scripts\python -m scripts.identify_user_image --help
venv\Scripts\python -m scripts.identify_group_image --help
```

Manual browser checks:
- Start only `kazusa-control-console`; verify the console UI loads on loopback and requires authentication.
- Inspect the running console UI against this plan and `src/control_console/README.md`; verify sidebar page arrangement, page density, populated cards/tables/forms, debug chat history, Bright/Dark theme controls, and no empty or compressed side pages.
- Verify standard UI elements use shadcn-style component anatomy: Sidebar, Button, Card, Badge, Table, Input, Select, Textarea, Field/Form-style grouping, Separator/ScrollArea, and Sheet/Dialog/Drawer or Tabs only where those patterns are needed.
- Verify the served UI remains static and Python/FastAPI based; no Node.js process, frontend package manager, frontend dev server, or frontend build artifact is required to run the console.
- Verify `/api/*` rejects unauthenticated requests and state-changing requests without CSRF token.
- After login, verify the UI fetches `/api/bootstrap` once, opens one EventSource connection to `/api/stream`, and does not full-page refresh or run broad timer polling while idle.
- Start the brain from the console; verify process state transitions through `starting` to `running`, PID is recorded, health becomes healthy, and audit/log events are written.
- Stop the brain from the console; verify dependent managed adapters are stopped first, brain receives graceful termination, health becomes unavailable, and audit/log events are written.
- Restart the brain from the console; verify generation id changes and prior logs remain queryable.
- Start one adapter from the console; verify dependency checks require the brain to be running and adapter status appears in service monitor.
- Kill a console-owned child process outside the console; verify crash detection marks it `crashed`, records exit code, writes lifecycle event, and does not hide the failure.
- Start an unmanaged process on a configured service port; verify the console marks the service `conflict`, refuses start/stop/restart with `409`, and does not kill or adopt the process.
- Stop `kazusa-control-console`; verify console-owned child services are stopped in dependency-aware order and unmanaged conflicting processes are untouched.
- Send a debug-console message while the brain is running; verify response rendering, tracking id, latency, debug mode handling, and audit event.
- After a successful debug-console message with graph data available, verify the Debug chat page renders the most recent debug cognition run through the same graph renderer as Overview.
- Send a debug-console message while the brain is stopped; verify a clear brain-not-running response and no cognition/persistence work starts.
- On Overview, verify the latest cognition graph shows real available run state or a clear unavailable/not_reported state without dummy nodes.
- Validate the cognition graph with a fixture or live run that includes parallel branches; verify branch lanes, fork/join edges, status badges, and hover/focus reasoning detail are readable and styled with surrounding shadcn Card/Badge/HoverCard or Sheet/Dialog anatomy.
- Verify cognition graph detail redacts prompts, secrets, embeddings, full message bodies, raw memory documents, and unbounded payloads.
- Run user image/style, group style, memory, calendar, background-work, health/cache, and event-log lookups; verify pagination, redaction, no embeddings, no prompts, no secrets, and no unbounded message dumps.
- Open the event monitor; verify filters by service id, event type, level, request id, tracking id, and time window.
- Leave the overview open; verify SSE updates service state, health, cache/event counters, and recent audit events without streaming full logs or full lookup tables.
- Tail logs in an active service panel; verify SSE sends only log cursors/counters and the panel performs bounded REST detail fetches for visible log lines.
- Force an SSE reconnect after the replay window is unavailable; verify the server sends `control.gap` and the browser refetches `/api/bootstrap`.

## Independent Plan Review
Run this gate before changing status to `approved`, execution, or handoff. If
no separate reviewer is used, the parent agent must perform the review directly
from a fresh-review posture and state that no subagent was used.

Review scope:
- Required top matter is present.
- Mandatory sections are present in the required order.
- Mandatory skills and mandatory rules are explicit.
- No unresolved implementation decisions remain.
- Must Do and Deferred boundaries are clear.
- Contracts and data shapes are concrete.
- Execution model uses parent-led native subagent execution.
- Independent code review gate is present.
- Verification commands are specific and use the project virtual environment.
- The new `src/control_console/README.md` ICD requirement is explicit and checked.
- The embedded-brain-console design is fully superseded by the top-level management-console design.
- The obsolete mockup artifact is not referenced as a source of truth, and running production UI inspection is against this plan, the control-console ICD, and shadcn component-family anatomy.
- Standard UI elements must come from shadcn component families and may not become bespoke common widgets.
- The cognition graph gadget contract is explicit, reusable across Overview and Debug chat placements, supports visible parallel branches, and defers historical-run retrieval.
- The frontend stack remains Python/FastAPI served and buildless with no Node.js or frontend build/runtime stack.

Direct review result, 2026-06-17:
- Reviewer: parent agent direct review; no subagent used by user instruction.
- Inputs inspected: repository status, `README.md`, `docs/HOWTO.md`, `development_plans/README.md`, this plan, development-plan skill references, relevant subsystem README contracts, `pyproject.toml`, and current source/test path inventory.
- Findings addressed in this draft: plan class corrected to `high_risk_migration`; venv-based verification commands added; integration test targets made explicit; progress-checklist evidence/sign-off rule added; static asset packaging called out; `src/control_console/README.md` ICD requirement added; this plan-review gate moved before acceptance/sign-off.
- Remaining blockers: none known other than explicit user approval and changing status from `draft` to `approved`.
- Approval status: ready for user approval; do not execute until the status line and registry are updated to `approved`.

## Independent Code Review
Run this gate after focused and regression verification passes.

Review scope:
- Confirm the console is a top-level `src/control_console` package and is not mounted or imported by the brain service.
- Confirm `src/control_console/README.md` exists and follows the repository ICD style, including purpose, intended use cases, interface boundary, public interfaces, security model, forbidden behavior, testing expectations, and UI stack constraints.
- Confirm `kazusa-control-console` is the new normal operator entrypoint and existing brain/adapter commands remain available.
- Confirm all lifecycle actions operate only on registry-declared services and never execute shell strings or arbitrary browser-provided commands.
- Confirm stop/restart can target only console-owned child processes with matching PID, generation, and command fingerprint.
- Confirm unmanaged port/process conflicts are reported as `conflict` and are not adopted, stopped, restarted, or inspected as managed services.
- Confirm start/stop/restart actions write sanitized audit events with operator id, reason, target, previous state, new state, timestamp, and request id.
- Confirm dependency order starts brain before adapters and stops adapters before brain.
- Confirm debug chat uses the existing brain `/chat` contract over HTTP and returns a safe unavailable state when the brain is stopped.
- Confirm all 11 original operator inspection capabilities are implemented under the top-level console API/UI.
- Confirm event-log monitoring covers console lifecycle/audit, process logs, console errors, and Kazusa operational events through bounded queries.
- Confirm the frontend data path is `/api/bootstrap` initial snapshot, authenticated REST commands/detail queries, and one read-only `/api/stream` SSE connection.
- Confirm no WebSocket, full-page auto refresh, broad timer polling, or unbounded live log/detail stream was added.
- Confirm the production UI was inspected against this plan and the `src/control_console/README.md` ICD, and that any visual or arrangement mismatch was either remediated or explicitly recorded in `Execution Evidence`.
- Confirm standard UI elements follow shadcn UI component families and anatomy rather than custom bespoke widgets.
- Confirm the cognition graph gadget is reusable, data-driven, and used by both Overview latest-run and Debug chat latest-debug-run placements when graph data is available.
- Confirm the graph renderer supports parallel branches, fork/join edges, status badges, and hover/focus reasoning details without exposing prompts, secrets, embeddings, full message bodies, raw memory documents, or unbounded payloads.
- Confirm historical cognition-run retrieval, persistence, and built-in Live/History widget switching were not introduced in this plan.
- Confirm no Node.js, npm, pnpm, yarn, React, Vue, Vite, Webpack, Tailwind build tooling, frontend dev server, frontend package manager workflow, or frontend build/runtime stack was added.
- Confirm console auth protects every `/api/*` endpoint except login/static assets and state-changing routes enforce CSRF or equivalent same-origin protection.
- Confirm redaction removes secrets, prompts, embeddings, env values, raw callback secrets, raw tokens, raw message bodies in aggregate views, and unbounded text.
- Confirm domain lookup endpoints use existing public helper boundaries or new DB-owned helper functions.
- Confirm no prompt, cognition, RAG, memory promotion, calendar semantics, background-work generation, adapter transport semantics, or global-growth semantics changed.
- Confirm tests prove registry safety, process lifecycle, dependency order, auth, redaction, lookup limits, event-log bounds, SSE event shape, and route behavior.
- Confirm the static UI has no external runtime dependency and works from the control-console route.

The parent records review findings, remediation commits, rerun commands, and final reviewer sign-off in `Execution Evidence`.

## Acceptance Criteria
- `kazusa-control-console` starts a top-level local FastAPI management console outside the brain service.
- `src/control_console/README.md` documents the module in ICD style, including interface boundary, intended operator use cases, public CLI/API/UI/SSE contracts, security model, forbidden behavior, testing expectations, shadcn UI constraint, and Python/FastAPI buildless frontend constraint.
- The brain service does not mount or import console routes.
- The console manages exactly one configured local application instance and has only the OS permissions of the user that launched it.
- Operators can start, stop, and restart the brain process from the console.
- Operators can start, stop, and restart registered adapter processes from the console.
- Dependency order prevents adapters from starting without the brain and prevents the brain from stopping before managed adapters stop.
- Future local frontend, backend, adapter, worker, and support services for this application can be added through a validated registry spec and appear in lifecycle controls, status, logs, events, and overview without UI code changes.
- Lifecycle actions use argv arrays with no shell execution and cannot execute browser-provided commands.
- The console refuses to adopt, stop, restart, or kill unmanaged external processes and reports configured endpoint conflicts as `conflict`.
- Stopping `kazusa-control-console` gracefully stops console-owned child services by default and leaves unmanaged conflicting processes untouched.
- Every privileged lifecycle, debug, lookup, log-view, auth-failure, crash, and console-error event is audited locally and mirrored to MongoDB when available.
- The service monitor shows desired state, actual state, PID, uptime, health, exit code, restart count, dependencies, recent events, and recent logs.
- Event-log monitoring supports bounded filters across console audit, process logs, console errors, and Kazusa operational events.
- Production UI layout and colors align with this plan and `src/control_console/README.md`: shadcn-style sidebar/inset arrangement, sidebar-controlled subpages, populated overview, debug chat history, lookup pages, health/cache, audit, and Bright/Dark color treatments.
- Standard UI elements are implemented from shadcn component families/anatomy and do not introduce bespoke widgets for common controls or surfaces.
- The reusable cognition-run graph gadget renders caller-provided snapshots, is used by Overview for latest cognition runs and by Debug chat for the latest debug cognition run when data is available, and is not hard-coded to either page.
- The cognition-run graph makes parallel branches visible through lanes/fork-join layout and exposes bounded hover/focus reasoning detail such as L2 internal monologue where available.
- The cognition-run graph never fabricates dummy graph content; unavailable data is shown as `not_reported` or a clear empty state.
- The frontend remains static, buildless, and Python/FastAPI served, with no Node.js, npm, pnpm, yarn, React, Vue, Vite, Webpack, Tailwind build tooling, frontend dev server, frontend package manager workflow, or frontend build/runtime stack.
- The frontend loads a bootstrap snapshot once after authentication, then receives compact realtime updates through one read-only SSE stream.
- State-changing actions and detail refreshes use authenticated REST routes with CSRF for mutations.
- The console does not use full-page auto refresh, broad polling, WebSocket, multiple live streams, or unbounded streamed detail tables in v1.
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
- Docs explain setup, local-only warning, OS-user permission model, shared local environment configuration, auth, service registry, lifecycle controls, event monitoring, and smoke checks.

## Execution Evidence
- 2026-06-19 production-code subagent cross-reference: owner-oriented
  `Character`, `Users`, and `Groups` console information architecture work is
  tracked by
  `development_plans/active/short_term/control_console_entity_information_architecture_plan.md`;
  this older backend console plan no longer defines standalone top-level
  `Memory` and `Interaction style` pages as the final sidebar structure.
- 2026-06-17 parent: plan status changed from `draft` to `approved` after user approval, then to `in_progress` after focused-test baseline was recorded. Registry row updated to match. Sign-off: parent/2026-06-17. Next checkpoint: production-code subagent.
- 2026-06-17 parent: mandatory skills loaded and plan reread after context compaction. Loaded `development-plan`, `py-style`, `test-style-and-execution`, `local-llm-architecture`, `build-web-apps:shadcn`, and `build-web-apps:frontend-app-builder` guidance relevant to the static UI constraints. Sign-off: parent/2026-06-17. Next checkpoint: production-code subagent.
- 2026-06-17 parent: focused deterministic tests added:
  `tests/test_control_console_contracts.py`,
  `tests/test_control_console_service_registry.py`,
  `tests/test_control_console_process_store.py`,
  `tests/test_control_console_supervisor.py`,
  `tests/test_control_console_lifecycle_routes.py`,
  `tests/test_control_console_event_monitor.py`,
  `tests/test_control_console_redaction.py`,
  `tests/test_control_console_bootstrap.py`,
  and `tests/test_control_console_stream.py`. Sign-off: parent/2026-06-17. Next checkpoint: pre-implementation failure baseline.
- 2026-06-17 parent: pre-implementation focused-test command
  `venv\Scripts\python -m pytest tests/test_control_console_contracts.py tests/test_control_console_service_registry.py tests/test_control_console_process_store.py tests/test_control_console_supervisor.py tests/test_control_console_lifecycle_routes.py tests/test_control_console_event_monitor.py tests/test_control_console_redaction.py tests/test_control_console_bootstrap.py tests/test_control_console_stream.py -q`
  failed as expected with 9 failures, each `ModuleNotFoundError: No module named 'control_console'`. This confirms the focused contract is failing only because the planned top-level module is not implemented yet. Sign-off: parent/2026-06-17. Next checkpoint: production-code subagent.
- 2026-06-17 production-code subagent: one `gpt-5.5 high` implementation agent completed the first production pass. Changed files included the new top-level `src/control_console` package, static UI assets, focused tests, package metadata, and docs. Parent independently reran the focused control-console batch after the subagent returned; initial result before review remediation was 17 passing tests. Sign-off: parent/2026-06-17. Next checkpoint: parent verification and browser inspection.
- 2026-06-17 parent: post-subagent verification before independent review passed focused control-console tests, `compileall`, import checks, README ICD term check, no brain/adapters Python imports of `control_console`, no frontend package/build stack files, `kazusa-control-console --help`, and browser smoke through local Chrome. Browser inspection found sparse side pages; parent enriched the static UI with populated shadcn-style Card/Table/Badge surfaces, avatar favicon, theme toggle, and sidebar-controlled pages. Sign-off: parent/2026-06-17. Next checkpoint: independent code review.
- 2026-06-17 independent code review: one `gpt-5.5 xhigh` read-only reviewer reported blocking findings: lifecycle safety incomplete; shell-interpreter argv still accepted; inspection capabilities were placeholders; debug-chat request body leaked and privileged reads were under-audited; `/api/stream` was one-shot rather than persistent; tests proved scaffold behavior more than the approved contract. Sign-off: reviewer/2026-06-17. Next checkpoint: parent remediation.
- 2026-06-17 parent remediation: added focused tests for shell-interpreter rejection, dependency-aware lifecycle order, stale ownership conflict refusal, debug-chat redaction/audit, numeric SSE replay gaps, repository projection helpers, and character/growth routes. Production fixes included stricter registry command validation, supervisor dependency checks and dependent-stop ordering, PID/fingerprint ownership verification, conflict marking, crash detection, console shutdown cleanup hook, redaction of `message_text`/`body_text`/`raw_wire_text`, authenticated `/api/audit`, persistent SSE heartbeat loop, local audit/process event monitor readers, and authenticated character status/growth endpoints using DB-owned helper boundaries with safe unavailable fallback. Focused command:
  `venv\Scripts\python -m pytest tests/test_control_console_contracts.py tests/test_control_console_auth.py tests/test_control_console_service_registry.py tests/test_control_console_process_store.py tests/test_control_console_supervisor.py tests/test_control_console_lifecycle_routes.py tests/test_control_console_kazusa_client.py tests/test_control_console_bootstrap.py tests/test_control_console_event_monitor.py tests/test_control_console_log_store.py tests/test_control_console_redaction.py tests/test_control_console_repository.py tests/test_control_console_stream.py tests/test_console_debug_chat.py tests/test_console_lookup_limits.py -q`
  passed with 21 tests. Static/import checks also passed: `compileall`, `python -c "import control_console; import kazusa_ai_chatbot; import adapters"`, `git diff --check` with CRLF warnings only, and no `control_console` imports from brain/adapters. Sign-off: parent/2026-06-17. Next checkpoint: remaining domain adapters and Mongo audit mirror.
- 2026-06-17 parent browser inspection after remediation: temporary `kazusa-control-console` on `127.0.0.1:8765` with throwaway state/token loaded successfully in local Chrome. Login worked; exactly one EventSource request stayed open; Debug chat returned `brain_unavailable`; Audit page refreshed via `/api/audit` and showed `debug_chat_unavailable` without leaking the raw message; Event monitor saw local audit rows; Memory lookup used bounded `/api/lookups/memory`; theme toggle changed CSS tokens; sidebar pages remained populated. Temporary server and state directory were removed after inspection. Sign-off: parent/2026-06-17. Next checkpoint: remaining domain adapters and Mongo audit mirror.
- 2026-06-17 parent regression verification: matching regression groups passed: event logging 20 passed; calendar scheduler 59 passed; background work 24 passed, 2 deselected; Cache2 2 passed; character state 4 passed; global character growth 56 passed, 2 deselected; service tests 69 passed; runtime adapter tests 55 passed. No `test_brain_service*.py` files matched. Some groups connected to the configured MongoDB through normal project test paths; `.env` was not read by the agent. Sign-off: parent/2026-06-17. Next checkpoint: remaining domain adapters and Mongo audit mirror.
- 2026-06-17 parent regression fix after operator feedback: reproduced that the built-in registry omitted `adapter.napcat` and that `adapter.debug` activation returned HTTP 500 after `brain` was shown as running. Root cause was twofold: the registry brain command used `python -m kazusa_ai_chatbot.main`, but that module did not invoke `main()` when run with `-m`, so the child exited without serving the brain; lifecycle `ServiceLifecycleError` was not translated to an API conflict response. Fixes added `adapter.napcat`, made the brain module command executable, returned HTTP 409 with error detail for lifecycle conflicts, surfaced `last_error_preview` in service state, refreshed the UI after failed service actions, and showed service-card error text. Focused tests first failed for the missing behavior, then passed. Rendered Chrome/CDP validation used the actual Services page to log in, verify NapCat, stop brain/debug, confirm adapter Start buttons were disabled while brain was stopped, start brain, start debug adapter, toggle Bright/Dark themes, and check no horizontal overflow or alerts. Cleanup stopped the temporary console-owned brain/debug child processes and verified ports `8000`, `8080`, and `8767` were closed. Sign-off: parent/2026-06-17. Next checkpoint: remaining domain adapters and Mongo audit mirror.
- 2026-06-17 parent comprehensive web test pass: added `development_plans/active/short_term/backend_control_console_web_test_plan.md` and executed it. Deterministic coverage command passed 31 tests with 91% `control_console` statement coverage. Rendered Chrome/CDP validation on temporary `127.0.0.1:8768` exercised login, all sidebar pages, Bright/Dark themes, event source selector options, request/tracking text input presence, memory refresh, debug form output, audit refresh, and Start/Restart/Stop for `brain`, `adapter.discord`, `adapter.napcat`, and `adapter.debug` using safe registry override child services. Found and fixed `GET /api/lookups/{namespace}` HTTP 500 caused by unreachable `ControlConsoleRepository.empty_lookup()`. Cleanup verified temporary port `8768` closed and temporary state was removed. Remaining limitation: event request-id/tracking-id inputs are rendered but not wired into `refreshEvents()`, so functional filtering for those fields is not claimed. Sign-off: parent/2026-06-17. Next checkpoint: remaining domain adapters and Mongo audit mirror.
- 2026-06-17 parent live-service web verification: reran the comprehensive pass against the real built-in registry rather than a credential-free override. `.env` was not inspected by the agent; the normal runtime loaded its own configuration while starting services. Root causes found and fixed with failing-first tests: built-in registry used bare `python` and resolved outside the venv on Windows, causing live brain timezone dependency failures; control-console debug chat sent UTC ISO `local_timestamp` while the brain requires configured-local wall-clock text; the console brain client used a 5-second timeout while live local LLM turns exceed that and the existing debug adapter uses 120 seconds; successful service start retained stale `exit_code` and `last_error_preview`; Event Monitor request-id/tracking-id inputs rendered but were not wired. Fixes used `sys.executable` for built-in service commands, `build_turn_clock()["local_timestamp"]` for debug-chat payloads, a 120-second debug-chat timeout, start-time clearing of stale failure metadata, and wired event filter IDs through `refreshEvents()` and `/api/events`. Deterministic coverage command passed 33 tests with 92% `control_console` statement coverage. Final headless Chrome validation on isolated temporary `127.0.0.1:8769` logged in, opened every sidebar page, toggled Bright/Dark themes, exercised request-id/tracking-id event filters, started brain/debug/Discord/NapCat through the Services page, submitted the browser debug form after `/health`, verified history output, stopped all services, confirmed mutually exclusive service buttons, checked no horizontal overflow or card/table scrollbar anomaly, and captured no alerts/page errors/warning logs. Cleanup verified ports `8000`, `8011`, `8012`, `8080`, and `8769` were closed and no console-owned brain/adapter child process remained. Sign-off: parent/2026-06-17. Next checkpoint: remaining domain adapters and Mongo audit mirror.
- Remaining in-progress scope after review remediation:
  Mongo audit mirroring is not implemented; only local JSONL audit is complete.
  Kazusa operational event source integration in the console event monitor is not implemented; local audit and process logs are implemented.
  User image/style, group style image, calendar schedule/run, background-work, and full health/cache DB/API adapters remain incomplete beyond static UI surfaces or safe unavailable summaries.
  Unmanaged port/process conflict detection is represented through ownership metadata conflicts, but port-level external-process detection is not complete.
  Because these items remain open, this plan stays `in_progress` and must not be archived or marked complete.
- 2026-06-17 parent corrective follow-up: user review identified that several
  visible sidebar pages looked functional while still being placeholder,
  static, or safe-unavailable surfaces, and that Health/cache could report
  incorrect brain availability. Follow-up execution is tracked in
  `development_plans/active/bugfix/control_console_functional_remediation_plan.md`.
  This parent plan remains blocked from completion until that remediation
  either completes or records an explicit handoff for every remaining dummy or
  partial page. Sign-off: parent/2026-06-17.
- 2026-06-17 parent plan amendment: user rejected further mockup updates because
  the standalone HTML mockup had drifted from the production console. Removed
  `development_plans/reference/designs/backend_control_console_mockup.html` and
  replaced mockup-based review language with a production plan/ICD/shadcn
  inspection contract. Added cognition-chain diagram scope as a reusable,
  caller-supplied `CognitionRunGraphSnapshot` graph gadget for Overview latest
  runs and Debug chat latest debug runs, with visible parallel branches,
  hover/focus reasoning detail, redaction, no dummy graph data, and no historical
  retrieval implementation in this plan. Sign-off: parent/2026-06-17. Next
  checkpoint: implement and verify cognition graph contract after the active
  functional-remediation plan no longer blocks the console baseline.
- 2026-06-17 parent cognition graph implementation pass: added strict
  `CognitionRunGraphSnapshot`, node, and edge contracts; wired
  `latest_cognition_graph` into `/api/bootstrap` and `cognition_graph` into
  `/api/debug-chat`; projected bounded caller-supplied `cognition_graph` or
  `cognition_snapshot` telemetry through redaction; returned explicit
  `not_reported` snapshots with no dummy nodes when the current brain `/chat`
  contract does not report cognition internals; and added the reusable static
  graph renderer to Overview and Debug chat with visible lane/column placement,
  fork/join edge drawing, status badges, and hover/focus detail. Updated
  `src/control_console/README.md` with the ICD contract and current live-chat
  telemetry limitation. Focused failing-first command
  `venv\Scripts\python.exe -m pytest tests/test_control_console_cognition_graph.py tests/test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q`
  first failed for missing helper/API/static hooks, then passed with 4 tests.
  Sign-off: parent/2026-06-17. Next checkpoint: broad control-console
  verification and rendered UI inspection.
- 2026-06-17 parent cognition graph verification pass: broad command
  `$controlTests = @(Get-ChildItem -LiteralPath 'tests' -Filter 'test_control_console_*.py' | ForEach-Object { $_.FullName }) + @(Get-ChildItem -LiteralPath 'tests' -Filter 'test_console_*.py' | ForEach-Object { $_.FullName }); venv\Scripts\python.exe -m pytest @controlTests -q`
  passed with 54 tests. Static checks passed: `venv\Scripts\python.exe -m
  compileall -q src\control_console tests\test_control_console_cognition_graph.py`,
  import check for `control_console`, `kazusa_ai_chatbot`, and `adapters`,
  `git diff --check` with CRLF warnings only, source-tree grep showing
  `control_console` imports only inside `src/control_console`, and no
  `package.json`, Node lockfile, Vite, Webpack, or Tailwind config files.
  In-app Browser was unavailable, so rendered validation used installed
  headless Chrome through CDP on an isolated temporary auth-disabled console at
  `127.0.0.1:8770`. The real page authenticated, showed Overview
  `not_reported` and clear graph empty states, then a sample
  `CognitionRunGraphSnapshot` rendered through the production graph renderer
  with 4 nodes, 4 fork/join edges, L2 reasoning and memory on separate branch
  rows, focus-visible reasoning detail, no page horizontal overflow, and no
  graph-stage horizontal overflow. Temporary Chrome profile, console state,
  and listening ports were cleaned up. Sign-off: parent/2026-06-17. Next
  checkpoint: independent code-review gate if this scope is treated as a
  release candidate.

## Risks
| Risk | Mitigation | Verification |
|---|---|---|
| Command injection through service management | Registry-only argv lists, no shell execution, no browser-editable commands | Registry validation tests and code review for `shell=False` |
| Wrong process killed | Track console-owned PID/generation/command fingerprint, refuse unmanaged conflicts, and never adopt external processes | Supervisor tests for conflict refusal |
| Console unavailable after stopping brain | Console runs as separate top-level process | Manual smoke test stops brain while console remains available |
| Adapter failures during brain stop | Dependency-aware stop order stops adapters before brain | Lifecycle route and supervisor dependency tests |
| Stale service state after crash | Supervisor watches child exits and health probes | Crash detection test and manual kill smoke test |
| Sensitive-data exposure in logs/lookups | Redaction, bounded previews, no prompts/secrets/embeddings, auth | Redaction tests and independent review |
| MongoDB outage hides lifecycle audit | Local JSONL audit is source of truth; Mongo mirror is secondary | Audit tests with Mongo mirror failure |
| Event monitor overload | Bounded filters, limits, cursors, no full table streaming | Event monitor limit tests and SSE tests |
| UI sprawl | Left navigation, overview cards, detail workspaces, filters, lazy loading | Manual browser checks |
| Brain/adapters drift from direct commands | Existing commands remain fallback and are used by registry | Smoke checks for direct command help and console lifecycle |
