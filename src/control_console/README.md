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
- Inspect current service state, process logs, lifecycle audit records, health summaries, and event summaries.
- Send debug-chat messages through the existing brain `/chat` contract when the brain is running, with operator-selectable visible-reply, think-only, listen-only, and no-remember debug modes.
- Browse bounded read-only character, memory, interaction-style, calendar, background-work, health/cache, event, and audit summaries.
- Inspect due calendar runs and sanitized background-worker telemetry as partial read-only workflows; schedule editing and job payload browsing are not implemented.

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
- Logs and events: `GET /api/logs/{service_id}`, `GET /api/events`
- Debug chat: `POST /api/debug-chat`
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
| Character | `partial` | Existing character status and growth routes |
| Memory | `partial` | Scoped `user_memory_units` recent and keyword lookup by global user id; semantic vector search and all-user browsing are not exposed |
| Interaction style | `partial` | Scoped user and group interaction-style guidance lookup; image asset browsing is not implemented |
| Calendar | `partial` | Due `calendar_runs` inspection; schedule editing is not implemented |
| Background work | `partial` | Sanitized `background_work.worker` event telemetry; job payload browsing is not implemented |
| Health/cache | `partial` | Brain `/health` and `/ops/runtime-status` when brain is running |
| Audit | `partial` | Local JSONL audit only; Mongo mirroring is not implemented |

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

## Service Registry And Supervisor

The built-in registry contains:

- `brain`
- `adapter.discord`
- `adapter.napcat`
- `adapter.debug`

Override registries are loaded from `KAZUSA_CONTROL_SERVICE_REGISTRY`, validated as strict `ServiceSpec` documents, checked for duplicate ids, unsafe command strings, unknown fields, repository-escaping working directories, unknown dependencies, and dependency cycles.

The supervisor starts services with `asyncio.create_subprocess_exec(*argv)`. It never uses `shell=True`, command concatenation, broad process scanning, external process adoption, or PID killing outside console-owned child processes.

If a configured dependency endpoint is already listening before the console
starts it, that dependency is marked as an unmanaged conflict. The console must
not stop, restart, or adopt that process. Dependent services may still start
against that live endpoint when the conflict is specifically
`configured endpoint is already in use by an unmanaged process`; this supports
debug-adapter and read-only inspection workflows while preserving ownership
boundaries.

## Static UI Contract

The UI is buildless static HTML, CSS, and JavaScript served by Python/FastAPI. It follows shadcn component family anatomy for common surfaces: Sidebar, Button, Card, Badge, Table, Input, Select, Textarea, Separator, ScrollArea, Field/Form grouping, and dialog/sheet-style detail surfaces where needed.

No Node.js, npm, pnpm, yarn, React, Vue, Vite, Webpack, Tailwind build tooling, frontend dev server, frontend package manager workflow, or frontend build/runtime stack is required.

## Forbidden Behavior

- Do not mount or import the console from the brain service.
- Do not add prompt, cognition, RAG, memory promotion, reflection, calendar, background-work, global-growth, or adapter semantic changes from this package.
- Do not expose raw MongoDB clients in route handlers.
- Do not stream full logs, full conversations, full memory bodies, prompts, embeddings, secrets, or unbounded lookup tables.
- Do not use WebSocket, page auto-refresh, broad polling, arbitrary command execution, or external process adoption in v1.

## Testing Expectations

Deterministic tests cover strict contracts, auth/CSRF, registry validation, local state writes, log redaction, audit records, argv-only supervisor calls, route failure codes, event monitor redaction, repository unavailable fallbacks, debug-chat unavailable behavior, and compact SSE replay/gap behavior.
