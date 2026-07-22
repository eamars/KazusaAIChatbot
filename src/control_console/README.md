# Control Console Interface Control Document

## Document Control

- ICD id: `CONTROL-CONSOLE-ICD-001`
- Owning package: `control_console`
- Interface boundary: local operator browser -> control-console FastAPI app -> local child-process supervisor, existing brain HTTP API, and read-only repository helpers
- Runtime command: `kazusa-control-console`

## Purpose

The control console is the top-level local management process for one configured Kazusa application instance. It starts, stops, restarts, monitors, and audits registry-declared local services while the brain service remains a separate platform-neutral character runtime.

The console is not mounted by the brain service and must not change `/chat`, cognition, RAG, memory promotion, calendar semantics, background-work generation, prompts, or adapter transport behavior.

## Intended Use Cases

- Start the brain and local adapters from one operator page.
- Inspect current service state, live process logs, lifecycle audit records, health summaries, and event summaries.
- Send debug-chat messages through the existing brain `/chat` contract when the brain is running, with operator-selectable visible-reply, think-only, listen-only, and no-remember debug modes.
- Browse bounded read-only Character, Users, Groups, calendar, background-work,
  health/cache, event, and audit summaries.
- Inspect cognition-debug Prompt View panels for the production prompt-facing
  windows used by calendar recall, background-work result delivery,
  conversation progress, internal-monologue carry-over, and promoted global
  growth. Supporting queue, schedule, run, and event rows remain separate
  Operational Backing panels.
- Inspect due calendar runs and sanitized background-worker telemetry as
  partial read-only workflows; schedule editing and job payload browsing are
  not implemented.

## Interface boundary

The console owns:

- static Python/FastAPI-served UI assets;
- operator token authentication and CSRF checks;
- validated service registry loading;
- argv-only child-process lifecycle operations;
- local state, process logs, SSE summary events, and local audit JSONL;
- bounded HTTP calls to the existing brain endpoints.

The brain owns cognition and persistence coordination for chat turns. Adapters own platform transport. Database/domain packages own raw MongoDB access and storage semantics.

## Public Interfaces

- CLI: `kazusa-control-console`
- Static UI: `GET /`
- Auth: `POST /api/auth/login`, `GET /api/auth/session`
- Bootstrap: `GET /api/bootstrap`
- Lifecycle: `POST /api/services/{service_id}/start|stop|restart`
- Service config:
  `GET /api/services/{service_id}/config`,
  `PUT /api/services/{service_id}/config`,
  `POST /api/services/{service_id}/config/reset`
- Brain model routes:
  `GET /api/services/brain/model-routes`,
  `PUT /api/services/brain/model-routes/{route_key}`,
  `POST /api/services/brain/model-routes/{route_key}/reset`,
  `GET /api/services/brain/model-routes/{route_key}/available-models`
- Logs and events:
  `GET /api/logs/{service_id}`,
  `GET /api/logs/stream`,
  `GET /api/events`
- Debug chat: `POST /api/debug-chat`
- Owner entity inspection:
  `GET /api/entities/character`,
  `GET /api/entities/user`,
  `GET /api/entities/group`
- Lookups: `GET /api/lookups/{namespace}`
- SSE: `GET /api/stream`

Every `/api/*` endpoint except login and the read-only session-status check
requires an authenticated local session. State-changing endpoints also require
the configured CSRF header.

`GET /api/bootstrap` also returns the active shell identity:

- `application_identity.status`: `available`, `empty`, or `unavailable`.
- `application_identity.character_name`: the `character_state._id == "global"`
  profile `name` when the database is reachable and configured; otherwise
  `not connected`.
- `csrf_token`: the current session CSRF token, returned only after the
  HTTP-only session cookie has authenticated the browser.

`GET /api/bootstrap` also returns `service_config_summaries`, keyed by
service id, for services with registered configuration descriptors. The
summary is intentionally compact: configurable state, apply behavior, and
field count only. Full field metadata is loaded on demand through the generic
service config route.

`GET /api/bootstrap` returns `latest_cognition_graph` and
`latest_self_cognition_graph`, mirroring both under `overview`. When the brain
HTTP endpoint is available, the console reads both values from the brain
`GET /ops/latest-cognition-graph` endpoint; otherwise each returns
`status: not_reported`. `POST /api/debug-chat` returns `cognition_graph` for
the most recent debug turn. These fields use the same bounded cognition-run
graph snapshot contract:

- `source`: `overview_latest`, `debug_latest`, `self_latest`, or future
  `historical`.
- `status`: `not_reported`, `running`, `completed`, `failed`, or `partial`.
- `nodes`: up to 64 stage nodes with lane, column, optional branch, status,
  and selected semantic detail. Layout metadata remains available for drawing;
  the selected inspector does not repeat it as detail rows.
- `edges`: up to 96 directed links with `sequence`, `fork`, `join`, or
  `reference` kind.
- `redaction`: an explicit policy summary for excluded prompts, embeddings,
  raw messages, message envelopes, and operational identifiers.

The brain `/chat` response may include a bounded `cognition_graph` snapshot
derived from the actual graph result and consolidation state. The console
projects that snapshot through this same redacted contract. If the brain is
unavailable or a response does not include graph telemetry, the console returns
`status: not_reported` rather than fabricating graph nodes.

### Cognition graph selected detail

Overview Latest, Debug cognition, and the latest self-cognition snapshot use
the same `renderCognitionGraph` inspector. Its selected detail order is:

`input`, `reply_context`; `decision`, `reasoning`; the four L2 reasoning fields;
retrieval answer and evidence; continuity, progress, and commitments; selected
actions, results, and continuation; the four visual-directive lists; and actual
visible `messages`.

The separate `l3.visual_directives` node carries
`facial_expression`, `body_language`, `gaze_direction`, and `visual_vibe`.
When the existing visual gate disables the stage, the node remains present with
`status: skipped` and uses the existing grey/dashed terminated rendering. An
enabled empty result remains a completed node with an explicit empty-state
message. The selected panel preserves approved semantic text and list order in
a scrollable region; generic console redaction remains in force for all other
payloads. Prompts, raw model output, embeddings, message envelopes, target
identifiers, handler metadata, and internal ids stay excluded.

The human-readable brain process log records the normalized visual directive
after enabled validation using the same complete JSON rendering convention as
visible dialog output. Protected LLM traces remain the diagnostic source for
model metadata and raw-output capture; the two surfaces have separate
disclosure purposes.

The authenticated SSE stream emits `control.cognition_graph_invalidated` when
the brain reports a different response or self-cognition latest run id. The
browser responds by refetching bootstrap data, so self-cognition completion can
update its dedicated Overview graph without the Overview page itself triggering
cognition.

`GET /api/logs/stream` is a separate authenticated SSE stream for high-volume
process-log traffic. It is intentionally not merged into the compact status
stream. Query parameters are:

- `service_id`: `all` or a registry service id.
- `streams`: comma-separated `stdout`, `stderr`, and/or `supervisor`.
- `tail`: initial retained line count, bounded by the server.
- `cursor`: optional replay cursor from a previous log event.

The stream emits:

- `log.snapshot`: retained file-backed tail rows emitted when the stream opens.
- `log.ready`: marker that the initial snapshot is complete and live events are attached.
- `log.line`: new stdout, stderr, or supervisor rows appended by the console.
- `log.gap`: explicit notification that rows were dropped or replay is unavailable.
- `log.status`: service log availability, including unmanaged endpoint conflicts.
- `log.keepalive`: idle heartbeat.

The Live logs page is the intended operator workflow for raw process output.
Event monitor remains the structured audit and application-event search page.
Service cards include a `Logs` action that opens Live logs filtered to that
service. The browser keeps only a bounded local row set and supports local
pause, clear, autoscroll, wrapping, text filtering, highlighting, and row copy.

## Page Capability Status

`GET /api/bootstrap` returns `page_capabilities` so the browser can distinguish
working pages from partial or unavailable pages. The status vocabulary is:

- `ready`: the page is backed by a current route or local state source.
- `partial`: the page has some working data but does not satisfy the full
  development-plan inspection surface yet.
- `unavailable`: the page is temporarily unavailable because a dependency is
  down or unreachable.
- `disabled`: the page must not be presented as usable because its repository
  or API adapter is not implemented.

The current first-slice status is:

| Page | Status | Source |
|---|---|---|
| Overview | `ready` | Bootstrap service and audit summary |
| Services | `ready` | Registry and supervisor state |
| Debug chat | `ready` | Existing brain `/chat` contract when brain is running |
| Event monitor | `ready` | Local audit, process logs, and sanitized Kazusa event-log telemetry |
| Character | `partial` | Owner-oriented profile, runtime state, self-image, active growth traits, promoted global-growth Prompt View, character-global carry-over Prompt View, growth-run audit rows, and background-learning summaries where safely available |
| Users | `partial` | Platform-facing user lookup for profile, relationship summary, `user_memory_units`, user-scoped interaction-style guidance, exact conversation-progress Prompt View, and current carry-over Prompt View; internal global user ids are not browser inputs |
| Groups | `partial` | Platform-facing group lookup for group-channel style, group-scene carry-over Prompt View, and participant conversation-progress Prompt View when a participant platform user id is supplied |
| Calendar | `partial` | Pending calendar recall Prompt View, schedule definition backing rows, and due `calendar_runs` inspection; schedule editing is not implemented |
| Background work | `partial` | Result-ready cognition delivery Prompt View, recent job queue backing rows, and sanitized `background_work.worker` event telemetry |
| Health/cache | `partial` | Brain `/health` and `/ops/runtime-status` when brain is running |
| Audit | `partial` | Local JSONL audit only |

## Operator Token

The preferred stable setup does not store a plaintext login token. Operators
choose a local token, hash it once, and provide only the hash through
`KAZUSA_CONTROL_OPERATOR_TOKEN_HASH`.

```powershell
$env:KAZUSA_CONTROL_OPERATOR_TOKEN_HASH = venv\Scripts\python -c "from getpass import getpass; from control_console.auth import hash_operator_token; print(hash_operator_token(getpass('Operator token: ')))"
kazusa-control-console --host 127.0.0.1 --port 8765
```

Control-console settings read `.env` first, then apply real process
environment variables as overrides. This means `KAZUSA_CONTROL_OPERATOR_TOKEN_HASH`
may be kept in `.env` for local development, while a launch script or service
manager can still override it by injecting the environment variable.

When `KAZUSA_CONTROL_OPERATOR_TOKEN_HASH` is not set, the console generates a
random ephemeral operator token during startup, hashes it in memory, and prints
the plaintext token once in the server log:

```text
Control console access token: <random-token>
```

The fallback token is valid only for the current console process. Restarting
the console generates a new token.

At login, the browser sends the plaintext token to `POST /api/auth/login`.
The server verifies it against the configured PBKDF2-SHA256 hash. On success,
the server sets an HTTP-only `kazusa_control_session` cookie and returns a CSRF
token plus header name. Browser JavaScript attaches that CSRF token to
state-changing API calls. Sessions are process-local and expire after 12 hours
or when the console process restarts.

On page load or browser refresh, the static UI first calls
`GET /api/auth/session`. If the HTTP-only session cookie is still valid, the
server returns the session CSRF token and the browser calls `GET /api/bootstrap`
to resume without showing the operator-token field. If the cookie is missing,
expired, or belongs to an earlier console process, the shell stays locked and
shows the login form.

## Security Model

The console binds to loopback by default and has only the operating-system permissions of the user that launched it. Services are controlled only through validated registry `command` argv arrays. Browser requests never submit arbitrary commands, shell strings, process ids, container ids, system service names, remote hosts, or lifecycle targets outside the registry.

Responses are redacted before they reach the browser. Secrets, tokens, prompts, embeddings, raw environment values, raw message bodies, callback secrets, and unbounded text are excluded from logs, events, audit records, and lookup pages.

Prompt View panels show only production prompt-facing projections or source
episodes returned by their owning runtime helpers. Operational Backing panels
show bounded redacted rows for queue, schedule, audit, or telemetry context and
are not prompt input. Event-log snapshot browsing remains excluded from the
control console because snapshots are aggregate debugging telemetry, not a
cognition prompt window.

## Service Registry And Supervisor

The built-in registry contains:

- `brain`
- `adapter.discord`
- `adapter.napcat`
- `adapter.debug`

Override registries are loaded from `KAZUSA_CONTROL_SERVICE_REGISTRY`, validated as strict `ServiceSpec` documents, checked for duplicate ids, unsafe command strings, unknown fields, repository-escaping working directories, unknown dependencies, and dependency cycles.

The supervisor starts services with `asyncio.create_subprocess_exec(*argv)`. It never uses `shell=True`, command concatenation, broad process scanning, external process adoption, or PID killing outside console-owned child processes.

Descriptor-backed service config may render command or environment overlays
before a service starts. Command overlays return argv parts only. Environment
overlays are descriptor-approved name/value pairs. Both are included in the
process ownership fingerprint without storing raw secret values. Browser
requests cannot submit arbitrary commands, environment dictionaries, shells,
or command strings.

If a configured dependency endpoint is already listening before the console
starts it, that dependency is marked as an unmanaged conflict. The console must
not stop, restart, or adopt that process. Dependent services may still start
against that live endpoint when the conflict is specifically
`configured endpoint is already in use by an unmanaged process`; this supports
debug-adapter and read-only inspection workflows while preserving ownership
boundaries.

## Runtime Service Config

The console exposes a generic descriptor-driven configuration API for services
that have descriptors registered inside `control_console`. Services without a
descriptor have no Configure action and no config route payload.

The config snapshot shape is descriptor-driven:

- `service_id`, `title`, `description`
- `apply_behavior: "restart"`
- `state: "default" | "override_active" | "apply_failed" | "unavailable"`
- `fields[]` with key, label, description, value type, default source,
  default value, override value, effective value, restart flag, sensitivity,
  and validation metadata

State-changing config calls are authenticated and CSRF-protected:

- `PUT /api/services/{service_id}/config` accepts `reason`,
  optional `expected_version`, and a `values` object keyed by descriptor field.
- `POST /api/services/{service_id}/config/reset` accepts `reason` and optional
  `expected_version`, clears the process-local override, and restores the
  descriptor default.
- Unknown services, services without descriptors, stale versions, and invalid
  values are rejected before restart.

Overrides are process-local. They live only in the running control-console
process, are not written to the service registry, audit JSONL, `.env`, or a
database, and disappear when the console process restarts.

Apply is restart-based. If the target service is running, the console stores
the override, audits the request, restarts only that service with reason
`config apply requires restart`, and returns the config snapshot, service
state, restart result, and audit event id. If the service is stopped, the
console stores the override without attempting a restart; the next console
start for that service uses the effective config. Reset uses the same restart
rule when the service is running.

Defaults follow the console settings precedence style: dotenv values are
loaded first and real process environment values override them. The API exposes
only descriptor-approved default sources and validated field values; it never
returns raw environment maps, secrets, tokens, or command environments.

The initial production descriptor is for the NapCat QQ adapter active-group
allowlist. The descriptor reads `NAPCAT_ACTIVE_GROUPS` as a comma- or
space-separated list of numeric group ids and renders the effective list as
the adapter's existing `--channels` argv when the list is non-empty. An empty
effective list renders no `--channels` argument.

The Brain service also has a descriptor-backed model-route workflow on the
Services tab. The Brain service card spans the full service-grid row and shows
all chat LLM routes in a route matrix with a selected-route editor. Operators
can override only the route model id, max completion token budget, and
thinking flag. The console does not expose API keys, base URLs, embedding
routes, raw dotenv values, or a general environment editor.

Brain route overrides are process-local. If the Brain service is running and
console-owned, saving a route restarts it through the existing lifecycle path.
If it is stopped, the override is rendered as descriptor-approved
child-process environment on the next start. The Brain runtime path is
unchanged: after restart it reads the existing environment variables used by
its `LLMCallConfig` constants.

The selected-route model picker fetches OpenAI-compatible `/models` data
server-side for the route's effective provider. Responses are bounded to
model ids and model-family labels; provider credentials and raw provider
errors are not returned to the browser.

The browser model editor is discovery-only. Opening a route lazily discovers
models for that route if no cached discovery result exists. Unavailable and
empty provider results render as explicit states with a retry control. A
single discovered model renders as a read-only discovered-model row; if that
model differs from the route's current effective model, applying the route
uses that discovered model. Multiple discovered models render as a select
containing only discovered model ids. The Services tab does not expose a
free-text model id field.

Audit records are written for config views, apply requests, reset requests,
restart requests, successful application, and validation or version failures.
Audit targets include service ids, field keys, config state, and restart
status, but not submitted values, secrets, raw `.env` content, or full
environment dictionaries.

## Static UI Contract

The UI is buildless static HTML, CSS, and JavaScript served by Python/FastAPI. It follows shadcn component family anatomy for common surfaces: Sidebar, Button, Card, Badge, Table, Input, Select, Textarea, Separator, ScrollArea, Field/Form grouping, and dialog/sheet-style detail surfaces where needed.

The cognition-run graph is a reusable static UI gadget, not a page-specific
mockup. Overview uses it for the latest reported run; Debug chat uses it for
the most recent debug turn; a future historical-run inspector must reuse the
same graph contract and renderer instead of adding a second diagram widget.
Nodes expose bounded reasoning detail through hover and keyboard focus.

No Node.js, npm, pnpm, yarn, React, Vue, Vite, Webpack, Tailwind build tooling, frontend dev server, frontend package manager workflow, or frontend build/runtime stack is required.

## Forbidden Behavior

- Do not mount or import the console from the brain service.
- Do not add prompt, cognition, RAG, memory promotion, reflection, calendar, background-work, global-growth, or adapter semantic changes from this package.
- Do not expose raw MongoDB clients in route handlers.
- Do not stream full logs, full conversations, full memory bodies, prompts, embeddings, secrets, or unbounded lookup tables.
- Do not use WebSocket, page auto-refresh, broad polling, arbitrary command execution, or external process adoption in v1.

## Testing Expectations

Deterministic tests cover strict contracts, auth/CSRF, registry validation, local state writes, log redaction, audit records, argv-only supervisor calls, route failure codes, event monitor redaction, repository unavailable fallbacks, debug-chat unavailable behavior, and compact SSE replay/gap behavior.
