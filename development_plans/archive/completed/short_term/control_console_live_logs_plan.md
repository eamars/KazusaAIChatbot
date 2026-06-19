# control console live logs plan

## Summary

- Goal: Add a production-grade live log viewing surface to the control console so operators can inspect console-owned service stdout, stderr, supervisor events, and recent buffered history without leaving the web UI.
- Plan class: medium
- Status: completed
- Mandatory skills: `development-plan`, `py-style`, `test-style-and-execution`, `test-driven-development`, `build-web-apps:shadcn`, `build-web-apps:frontend-testing-debugging`
- Overall cutover strategy: compatible and additive. Keep existing service lifecycle APIs, event-monitor APIs, `/api/logs/{service_id}` tails, and compact status SSE behavior. Add a dedicated authenticated log-stream contract for high-volume log traffic.
- Highest-risk areas: log flood, browser memory growth, backpressure, disconnected SSE clients, replay gaps, stale cursors after service restart, secret leakage, confusing event/log taxonomy, unmanaged process expectations, session expiry, and noisy server warnings when browsers close active streams.
- Acceptance criteria: operators can open Live logs from the sidebar, open a service-filtered log view from any service card, filter by service and stream, pause/resume, clear local view, toggle autoscroll/wrapping, see reconnect/gap states, and verify that log streaming does not starve the status stream, leak secrets, or flood the browser/server.

## Context

The control console already owns process supervision for registry-declared
child services, bounded stdout/stderr capture through the local log store,
service lifecycle audit, authenticated APIs, CSRF-protected state-changing
actions, static buildless UI, and a compact `/api/stream` for status and
invalidation events. Operators currently have bounded log access, but the UI
does not provide a focused live-tail workflow for watching startup, shutdown,
adapter connection, or debug-chat failures as they happen.

This plan treats live logs as an operator observability surface, not as an
event monitor replacement. The Event monitor remains the structured audit and
domain-event search surface. The new Live logs page is optimized for watching
raw console-owned process output and supervisor lines in near real time.

Online design references used for the proposal:

- Grafana Loki exposes dedicated APIs for log queries and tailing, keeping log
  volume separate from general dashboard state.
- Datadog Live Tail focuses on near-real-time troubleshooting with filters and
  processed log events.
- AWS CloudWatch Logs Live Tail supports filter and highlight workflows for
  live troubleshooting.
- Kubernetes and GitLab dashboards expose logs from resource detail contexts,
  so service-to-log navigation should be direct rather than hidden in a generic
  event page.

## Mandatory Skills

- `development-plan`: load before editing this plan, executing it, or changing its status.
- `py-style`: load before editing Python production files or tests.
- `test-style-and-execution`: load before adding, changing, running, or interpreting pytest tests.
- `test-driven-development`: write focused failing tests before production behavior changes.
- `build-web-apps:shadcn`: apply before editing static UI structure, component anatomy, density, and controls.
- `build-web-apps:frontend-testing-debugging`: apply before rendered-browser validation of the local UI.

## Mandatory Rules

- After automatic context compaction, reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- Do not read `.env` through tools. Production code may use the existing settings path where required.
- Do not introduce Node.js, npm, pnpm, yarn, React, Vue, Vite, Webpack, Tailwind build tooling, or a frontend package-manager workflow.
- Do not add external log vendors, database log indexes, WebSocket infrastructure, or a new frontend framework.
- Do not modify `src/adapters/**` or `src/kazusa_ai_chatbot/**` for this plan unless the user explicitly approves an expanded scope.
- Do not adopt logs from unmanaged external processes. If a service is in endpoint conflict or otherwise unmanaged, show `logs unavailable from this console run`.
- Do not stream raw prompts, secrets, bearer tokens, operator tokens, full environment values, embeddings, or unbounded memory/conversation bodies.
- Use shadcn component family anatomy for the UI: Sidebar entry, Button, Card, Badge, Select, Input, Toggle/Checkbox, Separator, Table where needed, ScrollArea-style log viewport, and Sheet/Dialog only for focused row details.
- Keep the existing compact status SSE small. High-volume log traffic must use a separate bounded log stream so status, health, service state, and cognition graph updates stay responsive.

## Must Do

- Add a `Live logs` sidebar subpage under the operational group, near `Services` and `Event monitor`.
- Add a `Logs` action on each service card that opens the same Live logs page with that service selected.
- Keep `Event monitor` as structured event/audit search. Do not merge live process logs into the event page as the primary workflow.
- Add a dedicated authenticated `GET /api/logs/stream` SSE endpoint for live log traffic.
- Keep `GET /api/logs/{service_id}` bounded tail access for compatibility and for non-live fallback.
- Add a console-owned log stream hub with bounded replay, per-subscriber queue limits, explicit drop/gap reporting, and cleanup on disconnect.
- Stream only console-owned service logs and supervisor-generated lifecycle lines.
- Emit an initial bounded tail snapshot on stream open, followed by live lines, so the UI does not have a race between separate tail fetch and stream connect.
- Support service filter, stream filter, plain-text contains filter, highlight text, pause/resume, clear local view, autoscroll toggle, wrap-lines toggle, and copy-row action.
- Show connection state in the UI: connecting, live, paused locally, reconnecting, gap, unavailable, and signed out.
- Use existing theme tokens and shadcn-like density. Avoid a fake terminal aesthetic that becomes unreadable in bright theme.
- Ensure long log lines wrap or horizontally scroll by operator choice and never force page-level overflow.
- Preserve redaction and truncation before log lines leave the backend.
- Add tests for auth, filtering, replay, gap reporting, slow subscribers, disconnect cleanup, redaction, UI wiring, and failure states.
- Update `src/control_console/README.md` with the live-log interface contract and intended operator use.

## Deferred

- No WebSocket implementation.
- No external log aggregation vendor integration.
- No durable log indexing or full-text search database.
- No log download/export feature in this plan.
- No cross-service trace correlation beyond request/tracking ids already present in log text or structured events.
- No terminal command execution, shell input, or arbitrary process attach from the log page.
- No mobile-specific tuning beyond responsive, non-desktop-hardcoded layout constraints already required by the console.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Existing log tail route | compatible | Keep `/api/logs/{service_id}` behavior and tests. |
| Compact status SSE | compatible | Keep status/invalidation stream bounded and low volume. |
| Log streaming | additive | Add a separate authenticated log SSE because live logs are high-volume and failure-prone. |
| Event monitor | compatible | Keep structured event/audit search as a separate workspace. |
| UI navigation | additive | Add `Live logs` sidebar item and service-card `Logs` actions. |

## Target State

Operators can troubleshoot from either direction:

- From `Services`, click `Logs` on `brain`, `adapter.debug`, `adapter.napcat`, or another registry service to open a filtered live tail.
- From the sidebar, open `Live logs` to watch all console-owned services, then narrow by service, stream, or text.
- When starting or restarting a service, leave the log page open and see startup output, supervisor lifecycle lines, reconnect/gap state, and redacted errors without refreshing.
- When a browser tab is refreshed, reopened, or disconnected, the log page reconnects with bounded replay and clearly reports if any lines were dropped.

The UI layout:

- Page intro with current stream status badge and concise purpose text.
- Toolbar card with shadcn-style controls: service select, stream segmented control or select, text filter input, highlight input, pause/resume button, clear button, autoscroll toggle, wrap toggle.
- Full-width log viewport card using ScrollArea-style behavior. Rows are compact, monospace only for log content, with service, stream, timestamp, sequence, and redacted line text.
- A small right-aligned status strip inside the log card for live/reconnecting/gap counts and current buffer size.
- Empty states are actionable: signed out, no service selected, service stopped with no recent logs, unmanaged conflict, stream disconnected, and filter has no matches.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Page placement | Add a dedicated `Live logs` sidebar page and service-card `Logs` actions. | Logs are a frequent troubleshooting workflow and should not be buried in Event monitor. |
| Event taxonomy | Keep Event monitor for structured audit/domain events; Live logs for process output. | Prevents a busy page that mixes different operator tasks. |
| Transport | Use SSE, not WebSocket. | The browser only receives log events; commands remain existing CSRF-protected REST calls. |
| Stream separation | Use a dedicated log SSE separate from compact status SSE. | Prevents log floods from delaying service state, health, or cognition graph updates. |
| Replay model | Stream sends bounded snapshot first, then live events. | Avoids tail-fetch plus connect race and gives immediate context. |
| Filters | Backend service/stream filter; simple client text filter/highlight. | Avoids regex abuse and keeps the server path cheap. |
| Slow clients | Per-subscriber bounded queues with `log.gap` events when dropped. | Protects the server and makes loss visible to the operator. |
| Visual style | Neutral bright/dark tokens and shadcn-style controls. | Avoids a custom terminal widget and keeps the console coherent. |

## Contracts And Data Shapes

### Stream API

```text
GET /api/logs/stream?service_id=all&streams=stdout,stderr,supervisor&tail=100&cursor=<cursor>
```

Query rules:

- `service_id` is `all` or a registry service id.
- `streams` is a comma-separated allowlist from `stdout`, `stderr`, and `supervisor`.
- `tail` is bounded by server settings.
- `cursor` is optional and used only for bounded replay from the in-memory stream buffer.
- Text filter and highlight are UI-local in v1 unless later profiling shows server-side filtering is needed.

### SSE Events

```json
{
  "event": "log.snapshot",
  "data": {
    "cursor": "brain:stdout:124",
    "service_id": "brain",
    "stream": "stdout",
    "sequence": 124,
    "created_at": "2026-06-18T00:00:00Z",
    "line": "redacted bounded text",
    "truncated": false
  }
}
```

Supported event names:

- `log.snapshot`: initial bounded tail row.
- `log.ready`: snapshot complete and live stream attached.
- `log.line`: new live row.
- `log.gap`: one or more rows were dropped or the cursor fell outside replay.
- `log.status`: service log availability changed.
- `log.keepalive`: heartbeat for idle streams.
- `log.error`: recoverable stream error shown in the UI.

## Failure Modes To Design For

| Failure mode | Expected operator behavior | Mitigation | Verification |
|---|---|---|---|
| Log flood from a noisy service | UI remains responsive and reports dropped lines instead of freezing. | Bounded backend replay buffer, bounded subscriber queues, row cap in the browser, `log.gap` event. | Unit test slow subscriber and browser row-cap static test. |
| Slow browser subscriber | One tab cannot block log capture or other subscribers. | Nonblocking publish, per-subscriber queue limit, drop-oldest with visible gap. | Unit test publish latency with a full subscriber queue. |
| Browser disconnects or refreshes | Server cleans up without noisy `socket.send()` warnings and client can reconnect. | Catch cancellation/disconnect paths, remove subscriber in `finally`, heartbeat only while connected. | Stream cancellation test and log-warning regression test where feasible. |
| Replay cursor too old | UI shows that the stream resumed with a gap. | Cursor lookup returns `log.gap` plus latest bounded snapshot. | Cursor-outside-buffer test. |
| Service restarts while log page is open | UI shows supervisor restart lines and continues with new sequence/cursor. | Include generation/sequence in payload and supervisor events in stream. | Start/restart route integration test with stream subscriber. |
| Service stopped or never started | UI clearly says no live process and shows any recent retained tail. | `log.status` event and empty-state copy tied to service state. | UI contract test for stopped service state. |
| Unmanaged endpoint conflict | UI says logs are unavailable from this console run. | Do not attach to external process; route returns availability status. | Existing conflict fixture plus log status test. |
| Partial lines or binary output | UI shows valid bounded text without breaking layout. | Decode with replacement, buffer newline chunks, cap per-line length, mark `truncated`. | Unit test invalid UTF-8 and long line truncation. |
| Secret or token appears in output | UI and APIs receive only redacted text. | Apply existing redaction before store/stream response; never expose raw env dumps. | Redaction test with bearer token/operator-token-like values. |
| Auth session expires mid-stream | Stream closes and UI shows signed-out state, not stale live status. | Same-session auth check on connect and close on invalid session where available. | Auth/session stream test. |
| Multiple tabs open live logs | Memory and tasks remain bounded. | Subscriber count cap, queue caps, explicit cleanup. | Multi-subscriber resource test. |
| Compact status stream starvation | Service state and cognition graph continue updating. | Keep log stream separate from `/api/stream`. | Test separate endpoint and no log payloads on status stream. |
| Client-side DOM growth | Long sessions do not degrade over time. | Browser row cap with visible dropped-local counter and clear button. | JS static test for max row cap and clear behavior. |
| Filter or highlight abuse | Filters do not create regex CPU spikes. | Plain substring matching only, bounded input lengths. | Static/API validation test for bounded filter fields if server-side fields are added. |
| Long CJK lines, URLs, or stack traces | Content stays readable and does not force page overflow. | Wrap toggle, overflow control, `overflow-wrap: anywhere` only where needed. | CSS contract test for log viewport overflow rules. |
| Process log store write failure | Operator sees degraded log storage/stream status. | Surface `log.status` or supervisor audit error without crashing service supervisor. | Fault-injection test for log-store append failure. |
| Console shutdown with active stream | Process exits cleanly. | Stream generator observes cancellation and supervisor shutdown order remains authoritative. | Manual or integration shutdown test. |
| Duplicate lines after reconnect | Duplicates are minimized and do not imply new service activity. | Stable cursor/sequence; client de-duplicates by cursor within current buffer. | JS/state reducer test. |
| New registry service id | UI filter updates without code changes. | Service options populated from bootstrap/registry state. | Static test with fake registry service. |
| Event monitor/log page confusion | Operators know where to go for structured events vs live process output. | Sidebar labels, intro text, and service `Logs` action. | Browser inspection checklist. |

## Change Surface

### Modify

- `src/control_console/app.py`, `routes.py`, `contracts.py`, `stream.py`, `log_store.py`, and `supervisor.py`: add the live-log stream contract, publication path, and authenticated route while preserving existing tails and status stream behavior.
- `src/control_console/static/index.html`, `console.js`, and `console.css`: add the Live logs sidebar page, service-card entry points, controls, state reducer, and shadcn-style log viewport.
- `src/control_console/README.md`: document the live-log interface and intended operator use.
- `tests/test_control_console_web_surface.py`: assert static UI wiring, row cap, and log page controls.
- `development_plans/README.md` and this plan: keep lifecycle state and evidence current.

### Create

- `tests/test_control_console_log_stream.py`: focused deterministic tests for stream hub, route contract, auth, filtering, replay, gaps, redaction, and disconnect cleanup.

### Keep

- `src/adapters/**` and `src/kazusa_ai_chatbot/**`: unchanged.
- Existing `/api/logs/{service_id}`, `/api/events`, and `/api/stream` contracts: retained.

## Overdesign Guardrail

- Actual problem: operators cannot watch console-owned service logs live from the web console while diagnosing startup, adapter, and debug-chat failures.
- Minimal change: add one dedicated authenticated log SSE stream, one Live logs page, and per-service navigation into that page using existing log storage and supervisor ownership.
- Ownership boundaries: the control console owns process-log capture, redaction, log streaming, and UI rendering; child services own their own stdout/stderr content; Event monitor continues to own structured event and audit search.
- Rejected complexity: no WebSocket, external log vendor, durable log index, terminal command input, unmanaged process attach, cross-service tracing layer, or frontend build stack.
- Evidence threshold: add any rejected complexity only after a verified production failure shows the bounded SSE plus local log-store model cannot satisfy operator troubleshooting.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when they preserve the stream, UI, and security contracts in this plan.
- The responsible agent must not introduce new architecture, alternate transports, compatibility shims, fallback process adoption, or extra features beyond this plan.
- The responsible agent must keep edits inside the listed change surface unless a blocker is reported and approved.
- The responsible agent must search for existing log, stream, redaction, auth, and UI helper behavior before adding new functions.
- The responsible agent must not perform unrelated cleanup, dependency changes, frontend framework work, or broad formatting churn.
- If implementation reveals that an existing contract prevents a Must Do item, stop and report the blocker rather than weakening the plan.

## Implementation Order

1. Contracts and failing tests: define SSE event contracts, log stream hub behavior, and UI surface assertions.
2. Backend stream hub: add bounded replay, subscriber queues, disconnect cleanup, and redacted event publication.
3. API route: add authenticated `GET /api/logs/stream` with service/stream/tail/cursor validation.
4. UI shell: add sidebar page, service-card `Logs` actions, toolbar controls, log viewport, empty states, and row cap.
5. Integration: wire supervisor log append path to the stream hub without changing adapter or brain code.
6. Documentation: update `src/control_console/README.md` with log contracts and intended use.
7. Verification: run focused unit/API/static tests, control-console test subset, browser interaction checks in Chrome, and diff hygiene.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the expected failure before production implementation starts.
- The user explicitly requested implementation in the current session.
- The parent agent executed the production changes and used one native read-only review subagent for the independent code review gate after verification.
- Independent code review remains required before final sign-off. If a native review subagent is unavailable in a future resumed session, the parent agent must perform a fresh code-review pass from review posture and record that limitation in `Execution Evidence`.

## Test Plan

- `tests/test_control_console_log_stream.py::test_log_stream_requires_authenticated_session`
- `tests/test_control_console_log_stream.py::test_log_stream_route_uses_last_event_id_header`
- `tests/test_control_console_log_stream.py::test_log_stream_route_query_cursor_overrides_last_event_id_header`
- `tests/test_control_console_log_stream.py::test_log_stream_replays_tail_then_live_line`
- `tests/test_control_console_log_stream.py::test_log_stream_does_not_duplicate_store_backed_snapshot`
- `tests/test_control_console_log_stream.py::test_log_stream_filters_by_service_and_stream`
- `tests/test_control_console_log_stream.py::test_log_stream_reports_gap_when_cursor_is_outside_buffer`
- `tests/test_control_console_log_stream.py::test_log_stream_drops_slow_subscriber_without_blocking_publish`
- `tests/test_control_console_log_stream.py::test_log_stream_redacts_and_truncates_lines`
- `tests/test_control_console_log_stream.py::test_log_stream_disconnect_cleans_up_subscriber`
- `tests/test_control_console_log_stream.py::test_log_status_reports_unmanaged_conflict_unavailable`
- `tests/test_control_console_log_store.py::test_log_store_publisher_failure_does_not_break_append`
- `tests/test_control_console_web_surface.py::test_live_logs_static_surface_and_controls`
- Browser E2E checklist: login, open Live logs, switch service filters, click service-card Logs action, pause/resume, clear, toggle autoscroll/wrap, start/stop a service while logs page is open, refresh and confirm replay/gap state, sign out and confirm stream stops.

## Verification

- Focused stream tests:
  `venv\Scripts\python -m pytest tests\test_control_console_log_stream.py -q`
- Static web-surface tests:
  `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py -q`
- Existing control-console stream/log tests:
  `venv\Scripts\python -m pytest tests\test_control_console_stream.py tests\test_control_console_log_store.py tests\test_control_console_supervisor.py -q`
- Control-console regression subset:
  `venv\Scripts\python -m pytest tests\test_control_console*.py -q`
- Rendered browser validation in Chrome for the Live logs flow when the console can be started without reading `.env`.
- Diff hygiene:
  `git diff --check`

## Independent Code Review

Run this gate after all verification commands pass and before final sign-off.
Review scope:

- Plan alignment for every `Must Do`, `Deferred`, change-surface, and failure-mode item.
- Python style, test style, auth/CSRF/session behavior, redaction, stream cleanup, backpressure, and browser memory cap.
- UI alignment with shadcn component family anatomy and the existing bright/dark console themes.
- Regression risk for existing `/api/logs/{service_id}`, `/api/events`, `/api/stream`, service lifecycle, and debug chat.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `Live logs` exists as a sidebar-controlled page with shadcn-style controls and a bounded log viewport.
- Every service card can open the Live logs page filtered to that service.
- `GET /api/logs/stream` is authenticated, bounded, redacted, filterable by service/stream, and separate from compact `/api/stream`.
- Stream replay, gap reporting, slow-subscriber handling, disconnect cleanup, stopped/unmanaged states, and browser row caps are tested.
- Existing log tail, event monitor, service lifecycle, and status stream tests continue to pass.
- The control-console README documents the live-log contract and intended use.
- Independent review is completed or its harness limitation is explicitly recorded with a parent fresh-review pass.

## Progress Checklist

- [x] User approves this plan for implementation.
- [x] Stage 1 failing tests recorded.
- [x] Backend stream hub implemented.
- [x] Log stream API implemented.
- [x] UI surface implemented.
- [x] README ICD updated.
- [x] Focused and control-console regression tests pass.
- [x] Browser interaction checks completed.
- [x] Independent review gate completed.

## Execution Evidence

- Draft created: pending verification.
- 2026-06-18: User approved starting implementation; plan promoted to `in_progress` for execution.
- 2026-06-18: Parent added focused failing tests before implementation. Initial expected failures: missing `/api/logs/stream` route and missing static `Live logs` UI surface.
- 2026-06-18: Implemented `LogStreamHub`, authenticated `/api/logs/stream`, persisted-tail snapshot plus live fanout, service/stream filters, gap/backpressure/disconnect cleanup, defensive redaction/truncation, `Last-Event-ID` reconnect support, and publisher isolation so observability fanout cannot break lifecycle log writes.
- 2026-06-18: Implemented `Live logs` sidebar page, service-card `Logs` actions, stream controls, bounded log viewport, local pause/clear/autoscroll/wrap controls, text filter/highlight repaint behavior, placeholder row count exclusion, and existing bright/dark theme alignment.
- 2026-06-18: Updated `src/control_console/README.md` with the live-log SSE interface contract and intended operator workflow.
- 2026-06-18: Focused verification passed: `venv\Scripts\python -m pytest tests\test_control_console_log_stream.py tests\test_control_console_log_store.py tests\test_control_console_web_surface.py -q` -> 23 passed.
- 2026-06-18: Rendered browser validation used a temporary explicit-settings local console on `http://127.0.0.1:8876` without reading `.env`. Verified login, Live logs open, service-card `Logs` action, `live` status, clear shows `0 rows`, text filter repaints retained rows, no duplicate retained line, and no browser console errors. Expected `net::ERR_ABORTED` entries appeared only when intentionally closing EventSource streams during page/filter changes.
- 2026-06-18: Independent review subagent `019eda1c-8dda-7581-9448-2f452e0e6a47` found four actionable issues: filter/highlight restart not reopening unchanged URL, missing `Last-Event-ID` reconnect support, placeholder rows counted as log rows, and publisher failure leaking into lifecycle paths. Parent fixed all four and also closed the residual snapshot-redaction risk.
- 2026-06-18: Final verification passed: `venv\Scripts\python -m pytest tests\test_control_console*.py -q` -> 99 passed; `venv\Scripts\python -m compileall -q src\control_console` -> passed; `git diff --check` -> no whitespace errors, only expected CRLF conversion warnings.
