# Backend Control Console Web Test Plan

## Goal

Validate the control-console web surface with at least 90% statement coverage
for `control_console`, plus rendered-browser checks for the operator workflows.

## Scope

This plan covers the Python/FastAPI-served web UI and every web-facing input
and output currently exposed by the console:

- Static shell: `GET /`, `/favicon.ico`, CSS and JavaScript assets.
- Auth: token login, session cookie, CSRF metadata, bad-token rejection.
- Bootstrap: authenticated service/overview/audit/capability snapshot.
- Lifecycle: Start, Stop, Restart buttons for registry services, expected
  version handling, unknown service handling, dependency conflict handling.
- Logs: bounded service log tail.
- Events: source selector output for `all`, `console`, `process`, and `kazusa`.
- Audit: sidebar-triggered bounded audit table refresh.
- Debug chat: channel, user id, display name, message text, brain-stopped
  output, brain-running HTTP failure output, and redaction.
- Lookups: memory refresh and generic lookup namespaces for unwired pages.
- Character: character status and global-growth read-only endpoints.
- SSE: authenticated stream response, replay/gap behavior, heartbeat shape.
- Browser controls: sidebar page switching, Bright/Dark theme toggle, login
  input, service buttons, debug form, event refresh, memory refresh, and audit
  navigation.

Visible but currently non-functional controls:

- Event Monitor request-id and tracking-id text fields render but are not wired
  into `refreshEvents()`. They can be checked for presence and layout only;
  functional filtering must be added before it can be counted as a tested web
  input.

Out of scope for deterministic web coverage:

- Real Discord login, real NapCat websocket connectivity, live LLM cognition,
  and platform delivery. These require external credentials/services.

## Coverage Gate

- Run deterministic tests with `coverage run --source=control_console`.
- Required final result: total `control_console` coverage >= 90%.
- Treat route-level HTTP 500s, browser alerts, horizontal overflow, missing
  service cards, missing CSRF protection, unredacted sensitive fields, and
  stale button states as failures.

## Test Matrix

| Surface | Inputs | Expected outputs |
|---|---|---|
| Static shell | `GET /` | HTML includes sidebar, login input, theme buttons, script and stylesheet |
| Favicon | `GET /favicon.ico` | 200 image response backed by `resources/avatar.png` |
| Login | valid token, invalid token | valid returns session/CSRF; invalid returns 401 and audit event |
| Bootstrap | session cookie | services include brain, Discord, NapCat, debug; capabilities and CSRF header present |
| Lifecycle start | service id, reason, expected version | 200 accepted response, stream event, updated service state |
| Lifecycle stop | service id, reason, expected version | 200 accepted response, dependents handled by supervisor |
| Lifecycle restart | service id, reason, expected version | 200 accepted response with action `restart` |
| Lifecycle errors | unknown service, stale version, dependency error | 404 or 409, never 500 |
| Logs | service id, limit | bounded redacted rows and `next_cursor` |
| Events | source selector, limit | bounded rows from console/process or empty kazusa source |
| Audit | limit | bounded sanitized local audit rows |
| Debug chat stopped | channel, user, display name, message | `brain_available=false`, safe unavailable error, redacted request |
| Debug chat running failure | same debug form fields | `brain_available=false`, HTTP error message, no 500 |
| Memory lookup | query, limit | bounded empty/redacted page, embeddings excluded |
| Generic lookups | namespace, limit | safe empty page, namespace reflected only as metadata |
| Character status | authenticated GET | available, empty, or unavailable bounded projection |
| Character growth | authenticated GET | available, empty, or unavailable bounded projection |
| SSE | session cookie, optional `Last-Event-ID` | `text/event-stream`, compact replay/heartbeat/gap event |
| Sidebar | every `data-page-link` | exactly matching page becomes active |
| Theme | Bright/Dark buttons and legacy localStorage names | `body[data-theme]` and active toggle update |
| Services UI | button clicks | Start mutually exclusive with Stop/Restart; dependencies disable adapter starts |
| Debug UI | form submit | history appends response/error row |
| Event UI | source select and refresh | table updates without alert |
| Memory UI | refresh button | table updates with redaction labels |

## Execution Commands

```powershell
$tests = @()
$tests += Get-ChildItem -LiteralPath 'tests' -Filter 'test_control_console_*.py' | ForEach-Object { $_.FullName }
$tests += Get-ChildItem -LiteralPath 'tests' -Filter 'test_console_*.py' | ForEach-Object { $_.FullName }
$tests += 'tests\test_runtime_adapter_registration.py::test_napcat_module_cli_help_exits_successfully'
venv\Scripts\python.exe -m coverage erase
venv\Scripts\python.exe -m coverage run --source=control_console -m pytest @tests -q
venv\Scripts\python.exe -m coverage report --show-missing
```

Rendered-browser validation uses a temporary loopback console, generated token,
and headless Chrome DevTools Protocol. It must clean up the temporary console
and any console-owned child services before completion.

## Execution Result: 2026-06-17

Deterministic coverage gate:

```powershell
$tests = @()
$tests += Get-ChildItem -LiteralPath 'tests' -Filter 'test_control_console_*.py' | ForEach-Object { $_.FullName }
$tests += Get-ChildItem -LiteralPath 'tests' -Filter 'test_console_*.py' | ForEach-Object { $_.FullName }
$tests += 'tests\test_runtime_adapter_registration.py::test_napcat_module_cli_help_exits_successfully'
venv\Scripts\python.exe -m coverage erase
venv\Scripts\python.exe -m coverage run --source=control_console -m pytest @tests -q
venv\Scripts\python.exe -m coverage report --show-missing
```

Result: 31 tests passed. `control_console` coverage was 91%.

Rendered-browser gate:

- Browser plugin path was unavailable as a callable browser API in this
  session, so validation used local headless Chrome through CDP.
- Temporary console ran on `127.0.0.1:8768` with a registry override whose
  four services used harmless `python -m http.server 0` commands.
- Validated login, every sidebar page, Bright/Dark theme buttons, event source
  selector options, request/tracking text inputs presence, memory refresh,
  debug form output, audit refresh, and Start/Restart/Stop for `brain`,
  `adapter.discord`, `adapter.napcat`, and `adapter.debug`.
- Browser result: no alert dialogs, no captured window errors, no runtime
  exceptions, no horizontal overflow, and final service states returned to
  stopped.
- Cleanup verified temporary port `8768` was closed and the temporary state
  directory was removed.

Issue found and fixed:

- `GET /api/lookups/{namespace}` returned HTTP 500 because
  `ControlConsoleRepository.empty_lookup()` was accidentally nested after
  `_empty_summary()` and unreachable. A failing test reproduced the route
  failure, then the method was moved onto `ControlConsoleRepository`; the test
  now passes.

Follow-up fix from live verification:

- Event Monitor request-id and tracking-id text fields now have stable IDs,
  `refreshEvents()` sends both values when present, and `/api/events` validates
  and applies both filters.

## Execution Result: 2026-06-17 Live Services

Deterministic coverage gate:

- Command: same `coverage run --source=control_console` batch listed above.
- Result: 33 tests passed. `control_console` statement coverage was 92%.
- Newly covered regressions:
  - Built-in registry commands must use the running interpreter
    (`sys.executable`) instead of bare `python`.
  - Console debug-chat payloads must use configured-local timestamps accepted
    by the brain service.
  - Console debug-chat client timeout must allow a live local LLM turn.
  - Successful service start must clear stale `exit_code` and
    `last_error_preview` from previous crashes.

Live service browser/API gate:

- Temporary console ran on `127.0.0.1:8769` with isolated state and the real
  built-in registry. The existing user console on `8765` was not touched.
- The brain, debug adapter, Discord adapter, and NapCat adapter were started
  and stopped through the actual Services page buttons.
- The debug form submitted through the browser after the brain reached
  `/health`; history appended a brain response row.
- Event Monitor request-id and tracking-id filters were filled through the
  browser; the filtered table returned the empty-state row and clearing filters
  returned console rows.
- Sidebar pages validated: Overview, Services, Debug chat, Event monitor,
  Character, Memory, Image/style, Calendar, Background work, Health/cache, and
  Audit.
- Bright and Dark theme buttons both updated `body[data-theme]`.
- Service buttons reflected state correctly: adapters were disabled while the
  brain was stopped; Start was disabled while running; Restart/Stop were
  enabled only while running.
- Browser result: no alert dialogs, no captured page errors, no warning/error
  console logs, no horizontal overflow, and no card/table element with a
  horizontal scrollbar when content width did not require it.
- Cleanup verified ports `8000`, `8011`, `8012`, `8080`, and `8769` were
  closed and no console-owned brain or adapter child processes remained.

Root causes found during live verification:

- The default registry used bare `python`; on this Windows host that resolved
  to a different interpreter without `tzdata`, producing live brain runtime
  errors. Fixed by using `sys.executable` for built-in services.
- The console debug-chat payload sent a UTC ISO `local_timestamp`; the brain
  requires configured-local wall-clock text. Fixed by using
  `build_turn_clock()["local_timestamp"]`.
- The console debug-chat HTTP timeout was 5 seconds; live local LLM turns can
  exceed that and the legacy debug adapter already uses 120 seconds. Fixed by
  using a 120 second timeout.
- Starting a service after prior failure left stale failure metadata in the UI.
  Fixed by clearing `exit_code`, `stopped_at`, and `last_error_preview` on
  successful start.
