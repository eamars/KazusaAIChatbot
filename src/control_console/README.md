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
- Discover known users and active groups before opening their bounded detail.
- Inspect canonical character cognition, relationship axes, causal emotion
  context, and user state.
- Inspect calendar schedules and recent outcomes, background jobs and
  aggregate worker activity, runtime readiness, and collapsed audit actions.

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
- Bootstrap and cross-owner aggregate: `GET /api/bootstrap`,
  `GET /api/overview`
- Health/cache owner projection: `GET /api/health`
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
  `GET /api/events`,
  `GET /api/audit`
- Debug chat: `POST /api/debug-chat`
- Owner entity inspection:
  `GET /api/entities/character`,
  `GET /api/entities/users`,
  `GET /api/entities/users/{platform}/{platform_user_id}`,
  `GET /api/entities/groups`,
  `GET /api/entities/groups/{platform}/{group_id}`
- Read-only lookups:
  `GET /api/lookups/memory`,
  `GET /api/lookups/style`,
  `GET /api/lookups/calendar`,
  `GET /api/lookups/background-work`
- SSE: `GET /api/stream`

Every `/api/*` endpoint except login and the read-only session-status check
requires an authenticated local session. State-changing endpoints also require
the configured CSRF header.

`GET /api/bootstrap` also returns the active shell identity:

- `application_identity.status`: `available`, `empty`, or `unavailable`.
- `application_identity.character_name`: the latest
  `character_identity_revisions` effective identity `name` when the database
  is reachable and configured; otherwise `not connected`.
- `csrf_token`: the current session CSRF token, returned only after the
  HTTP-only session cookie has authenticated the browser.

`GET /api/bootstrap` also returns `service_config_summaries`, keyed by
service id, for services with registered configuration descriptors. The
summary is intentionally compact: configurable state, apply behavior, and
field count only. Full field metadata is loaded on demand through the generic
service config route.

`GET /api/bootstrap` returns the canonical
`latest_cognition_observation` and `latest_self_cognition_observation` view
envelopes. `GET /api/overview` repeats those envelopes and exposes one
`cognition_observations` panel containing conversation and self-cognition
entries. `POST /api/debug-chat` returns `cognition_observation` for the direct
debug response. Each envelope has `view_kind`, `availability`, `reason_code`,
`generated_at`, and a nullable Brain-owned
`cognition_run_observation.v1` observation.

The console validates Brain's exact observation object and never infers nodes,
sections, labels, or values from messages or other response fields. `available`
requires a matching live/self run kind; `not_reported`, `unavailable`, and
`invalid` carry a bounded reason code and no observation. The envelope's time
is console-owned and never replaces Brain's observation timestamp.

### Cognition observation selected detail

Overview Latest, Debug, and Self Latest use one generic producer-driven
renderer. It resolves each selected node's ordered `section_refs`, then shows
the producer label, status, summary, ordered fields, ordered records, truthful
displayed/reported counts, and an omission marker. Additive producer sections
render without a JavaScript catalog. HTML-sensitive text is escaped and
loading/request-error banners remain separate from observation data.

The canonical `reasoning.context_consumption` and
`evidence.shared_memory_prewarm` sections are already-safe Brain sections. The
Character operational-posture panel receives the context section by its stable
identifier and performs no second semantic projection. Prewarm dispositions,
context-source parity, and group-scene status are preserved as producer-owned
fields.

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
Event monitor contains structured application events and never duplicates the
raw process stream, Audit records, or successful aggregate-owned tick/residue
access chatter. Audit remains the sole owner of collapsed operator/security
actions and summarized page views.
Service cards include a `Logs` action that opens Live logs filtered to that
service. The browser keeps only a bounded local row set and supports local
pause, clear, autoscroll, wrapping, text filtering, highlighting, and row copy.

## Information Ownership And Safe Projection Contracts

Each datum has one canonical page owner. Overview may repeat only bounded
cross-owner aggregates or exceptions; it does not repeat service, route,
worker, cache-agent, or audit-action detail.

| Page | Canonical information | Endpoint |
|---|---|---|
| Overview | Managed-service counts, one readiness aggregate, recent failures, recent changes, cognition observation entry points | `GET /api/overview` |
| Services | Authoritative lifecycle state, action/block reason, bounded process detail, one current value per model route | bootstrap and `/api/services/*` |
| Live logs | Bounded supervisor/stdout/stderr text and stream connection state | `GET /api/logs/stream` |
| Debug chat | Request controls, current response/error, cognition observation, browser-session history | `POST /api/debug-chat` |
| Event monitor | Structured events, filters, dynamic facets, duration/error/correlation detail | `GET /api/events` |
| Character | Latest profile, canonical cognition, persisted/effective operational posture, exact graph consumption, current self-image, redacted growth candidates/outcomes, immutable identity lineage and health | `GET /api/entities/character` |
| Users | Safe directory; profile, relationship/cognition, causal relationship projection, memory, consumer-labelled style, progress, carry-over | plural user entity routes |
| Groups | Activity directory; review, style, carry-over, participant progress | plural group entity routes |
| Calendar | Counts, schedules, recent run outcomes, scoped cognition visibility | `GET /api/lookups/calendar` |
| Background work | Queue counts, canonical jobs, worker aggregates, errors, delivery detail | `GET /api/lookups/background-work` |
| Health/cache | Database/scheduler readiness, worker liveness, semantic error level, Cache2 metrics | `GET /api/health` |
| Audit | Collapsed actions with explicit outcomes, view summary, dynamic facets | `GET /api/audit` |

Safe-field boundaries are fixed:

- The user directory returns only platform account labels/ids, display names,
  alias count, and update time. It excludes global ids, alias ids, cognition
  content, relationship evidence ids, and ownership ids.
- User relationship detail returns all eleven stored axes with the exact
  numeric value and canonical semantic band, plus evidence count and update
  time. It does not calculate an affinity replacement.
- Character operational posture exposes full persisted and elapsed-effective
  native affect/pressure rows through the state-projection allowlist. Its
  latest-consumption subsection uses only the graph-owned consumption field
  and distinguishes reported, partial, degraded, stale, and unavailable data.
- User causal relationship detail is current-user scoped and exposes only
  public axes, bounded causal/affect rows, and freshness labels. User and
  Group style panels label `relevance`, `cognition`, and `surface` source
  projections without style-image provenance.
- The group directory returns platform/channel label, last activity, message
  count, and participant count. Raw messages and participant ids are excluded.
- Character self-image comes from the latest identity revision. Growth shows
  bounded candidate and run projections. Carry-over shows derived health plus
  current/prior immutable revisions. These projections expose counts, coarse
  scope kinds, paths, value kinds, states, reasons, confidences, and times;
  they exclude database ids, evidence handles, correlations, before/after
  values, source text, prompts, and private facts. Group detail includes only
  the latest review window and excludes route/dispatch ledger machinery.
- User-thread continuity requires channel id and channel type. Group-thread
  style inspection passes the selected group id to the canonical style helper.
  Console carry-over reads disable residue-load telemetry so inspection does
  not write application-event rows.
- Calendar rows expose schedule/run kind, state, operator-relevant times, and
  safe result/failure summaries. Run ids, schedule ids, leases, attempts,
  idempotency machinery, and raw payloads are excluded.
- Background rows expose job/delivery state and safe aggregate worker counts.
  Task briefs, prompts, artifacts, raw messages, run ids, and leases are
  excluded.
- Health exposes readiness once, one row per runtime worker, and one row per
  cache agent with hits, misses, total, and hit rate.
- Audit groups only exact request ids, maps terminal events to explicit
  outcomes, humanizes targets from whitelisted fields, and summarizes view
  events outside the action list.
- Event monitor excludes process output and Audit-owned console records.
  Successful tick/residue access chatter stays on aggregate owner pages unless
  an exact event-type filter requests it. Error previews and correlation ids
  remain bounded row detail rather than default-page prose.

Every data-bearing panel reports `available`, `empty`, `needs_input`,
`partial`, or `unavailable`. A required child-source failure propagates to the
page as `partial` or `unavailable`; a successful sibling does not mask it.
Character profile/cognition, User profile/relationship/cognition, Group
activity/review, Calendar summary/schedules/runs, and Background
summary/jobs/worker activity own top-level availability. Optional continuity,
style, delivery, error, and scoped-visibility panels report their own status
without changing an otherwise truthful owner-page status.

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

The current implementation status is:

| Page | Status | Source |
|---|---|---|
| Overview | `ready` | Five frozen aggregate panels |
| Services | `ready` | Registry and supervisor state |
| Live logs | `ready` | Console-owned process streams |
| Debug chat | `ready` | Existing brain `/chat` contract when brain is running |
| Event monitor | `ready` | Structured Kazusa application-event telemetry |
| Character | `ready` | Canonical cognition owner projection |
| Users | `ready` | Safe directory and canonical detail |
| Groups | `ready` | Activity directory and sourced detail |
| Calendar | `ready` | Schedules, recent outcomes, scoped visibility |
| Background work | `ready` | Jobs, worker aggregates, errors, delivery |
| Health/cache | `ready` | Live readiness, workers, and Cache2 projection |
| Audit | `ready` | Collapsed local actions and view summary |

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

Responses are redacted before they reach the browser. Secrets, tokens,
prompts, embeddings, raw environment values, raw message bodies, callback
secrets, internal global ids, run/lease machinery, and unbounded text are
excluded from logs, events, audit records, and owner pages.

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

The Brain service has descriptor-backed model-route and character identity
growth configuration on the Services tab. The Brain service card spans the
full service-grid row, shows all chat LLM routes in a route matrix with a
selected-route editor, and exposes the generic Configure action. Operators can
override only the route model id, max completion token budget, thinking flag,
and these restart-applied identity-growth fields:

| Field | Default | Bounds |
|---|---:|---|
| `character_identity_growth_enabled` | `true` | boolean |
| `character_identity_growth_inferred_min_episodes` | `3` | `2..8` |
| `character_identity_growth_inferred_min_local_dates` | `2` | `1..7`, at most the episode threshold |
| `character_identity_growth_max_inferred_promotions_per_local_day` | `1` | `0..3` |
| `character_identity_growth_prompt_char_budget` | `18000` | `8000..30000` |

The console does not expose API keys, base URLs, embedding routes, raw dotenv
values, or a general environment editor.

The Brain route catalog exposes semantic route slugs rather than environment
variable names or implementation versions. The core cognition route is
`cognition`; shared supporting cognition services use `cognition_support` when
they are enabled. These slugs are stable browser capabilities and are mapped
to private server configuration internally.

Each descriptor uses the same selected-route editor and model-discovery
workflow as every existing chat route. Route implementation, provider, and
stage details are not part of the browser contract.

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

The cognition observation view is a reusable static UI surface, not a
page-specific mockup. Overview, Debug chat, and Self Latest resolve the
Brain-supplied `cognition_run_observation.v1` through the same generic section
renderer. The renderer uses producer labels, section references, field/record
order, truthful counts, and status dispositions; it does not maintain a
semantic field catalog or invent nodes. Displayed values are escaped and the
browser contract excludes prompts, raw answers, endpoint data, credentials,
and unapproved identifiers.

This is a validation-only transport and rendering boundary; Brain remains the
sole producer of the observation schema and projections.

No Node.js, npm, pnpm, yarn, React, Vue, Vite, Webpack, Tailwind build tooling, frontend dev server, frontend package manager workflow, or frontend build/runtime stack is required.

## Forbidden Behavior

- Do not mount or import the console from the brain service.
- Do not add prompt, cognition, RAG, memory promotion, reflection, calendar, background-work, global-growth, or adapter semantic changes from this package.
- Do not expose raw MongoDB clients in route handlers.
- Do not stream full logs, full conversations, full memory bodies, prompts, embeddings, secrets, or unbounded lookup tables.
- Do not use WebSocket, page auto-refresh, broad polling, arbitrary command execution, or external process adoption in v1.

## Testing Expectations

Deterministic tests cover strict contracts, auth/CSRF, registry validation, local state writes, log redaction, audit records, argv-only supervisor calls, route failure codes, event monitor redaction, repository unavailable fallbacks, debug-chat unavailable behavior, and compact SSE replay/gap behavior.

## Debug Chat Trace Synchronization

When an operator asks an agent to look up `xxx`, the value copied from the
Debug Chat metadata must be recorded as `source_surface=web_control_trace_id`.
The browser displays the brain-owned `trace_id` separately from
`delivery_tracking_id`; it never derives a trace from a graph run, event
correlation, request, action, or delivery identifier. The brain discloses the
top-level trace only when the request is `platform=debug`, the configured
`KAZUSA_CONTROL_BRAIN_SHARED_SECRET` matches, and a protected trace run was
recorded. The console sends the exact `debug-v1` marker and shared-secret
headers for that request.

| Identifier | Current browser availability | Exact next retrieval route |
| --- | --- | --- |
| `llm_trace_id` / `trace_id` | Debug Chat history shows `trace ...`; cognition references show the mapped trace row | `scripts.export_trace_correlation_manifest --source-surface web_control_trace_id --identifier <copied>` |
| `delivery_tracking_id` | Debug Chat history shows `tracking ...`, distinct from the trace | `scripts.export_llm_trace --delivery-tracking-id <id>` after the field is confirmed |
| `cognition_invocation_id` | shown in Overview Latest conversation cognition and Debug Current debug cognition `Run reference` | resolve the parent first, then `scripts.export_llm_trace --trace-id <trace> --cognition-invocation-id <id>` |
| `global_user_id` | shown in the existing Users row and selected profile | use the selected Users profile as the web source of truth |
| `background_work_job_id` | shown as `Job reference` in Jobs and Errors; delivery details retain it with parent/child trace fields | inspect the existing Background Work card details |
| `accepted_task_id`, `source_action_attempt_id`, `source_llm_trace_id` | shown in Background Work Jobs card details | inspect the existing Job reference disclosure and details |
| `calendar_schedule_id`, `source_llm_trace_id` | shown in Calendar Schedules and source details | inspect the existing Schedule reference and card details |
| `calendar_run_id`, related schedule/source trace | shown in Calendar Recent runs and existing timestamp/detail rows | inspect the existing Run reference and card details |
| event `request_id`, `correlation_id`, `tracking_id`, `run_id`, `trigger_id`, `attempt_id` | retained in the existing Event Monitor detail disclosure | open the existing event detail row |
| self-cognition child trace and calendar source | shown as `child_llm_trace_id` and `source_calendar_run_id` in the existing self-cognition `Run reference` | resolve the parent trace before opening child evidence |

The manifest is the protected evidence handoff. Inspect `parent_trace`,
`identifiers`, `joins`, and `unresolved` before opening a separate raw trace
export. Zero and multiple candidates remain explicit; no newest row is chosen.
