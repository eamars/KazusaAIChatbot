# Full Control Console Correlation Surface Bugfix Plan

Status: completed
Owner: Codex
Created: 2026-08-09

## Objective

Close the web control-console correlation visibility gap across the existing
views so an operator who only uses the web interface can retrieve the durable
IDs needed to invoke the trace-debug skill. Preserve the current routes,
panels, cards, tables, graph layout, and visual design.

## User-approved mapping

| Existing view | ID mapping |
|---|---|
| Debug Chat history | `llm_trace_id` as trace; `delivery_tracking_id` as tracking |
| Overview → Latest conversation cognition → Run reference | `run_id`; `llm_trace_id`; `cognition_invocation_id` |
| Debug → Current debug cognition → Run reference | `run_id`; `llm_trace_id`; `cognition_invocation_id` |
| Overview → Latest self-cognition → Run reference | self-cognition `run_id`; child `llm_trace_id`; `source_calendar_run_id` |
| Background Work → Jobs | Replace created, completed, and updated with `background_work_job_id` as the Job reference. Existing card details show `accepted_task_id`, `source_action_attempt_id`, and `source_llm_trace_id`. |
| Background Work → Errors | Same `background_work_job_id` mapping in the existing error cards |
| Background Work → Delivery detail | `background_work_job_id`, `parent_llm_trace_id`, `child_llm_trace_id`, and `source_background_work_job_id` |
| Calendar → Schedules | `calendar_schedule_id` as the Schedule reference; `source_llm_trace_id` in existing card details |
| Calendar → Recent runs | `calendar_run_id` as the Run reference; related `calendar_schedule_id` and source trace in existing details |
| Event Monitor | Existing details retain `request_id`, `correlation_id`, `tracking_id`, `run_id`, `trigger_id`, and `attempt_id` |
| Users → Known users / selected profile | `global_user_id` in the existing user row/details |

`run_id` remains the cognition graph reference and `llm_trace_id` remains the
LLM trace ID. Calendar timestamps remain visible. Existing job, calendar,
event, and user details remain the owning surfaces for their additional IDs.

## Scope and ownership

1. Extend the existing read-only control-console repository projections with
   the explicitly mapped persisted identifiers, preserving bounded field
   allowlists and redaction of unrelated internals.
2. Extend the existing brain cognition-graph metadata envelope with the
   already-produced live/self-cognition identifiers needed by the graph
   reference disclosure. Keep semantic cognition decisions and delivery
   behavior unchanged.
3. Render the identifiers in the existing graph Run reference disclosure,
   record-card reference disclosure, card detail grid, event detail grid, and
   user row/profile details. Keep the existing layout and labels as the
   smallest additive presentation change.
4. Add focused deterministic contract, repository, renderer, and browser E2E
   coverage. Capture screenshots of each affected existing surface and inspect
   them for layout/design alignment.

## Explicit exclusions

- No new panel, page, widget, lookup flow, route, or redesign.
- No database migration, backfill, schema/index change, or write-path change.
- No change to live response wording, cognition decisions, scheduling,
  background execution, delivery, or calendar timing behavior.
- No identifier is inferred from a different identifier type. In particular,
  `run_id`, `llm_trace_id`, `delivery_tracking_id`, and job/calendar IDs retain
  their existing meanings.
- No raw prompt, node detail, worker payload, or unrelated internal state is
  exposed.

## Implementation targets

- `src/control_console/repository.py`
- `src/control_console/app.py`
- `src/control_console/contracts.py`
- `src/control_console/kazusa_client.py`
- `src/control_console/static/console.js`
- `src/kazusa_ai_chatbot/service.py`
- Self-cognition graph metadata boundary and background-job projection files
  only where the existing persisted fields require a bounded read projection.
- Focused control-console and cognition-graph tests, plus the relevant README
  and plan registry entries.

## Acceptance checks

- Each row in the user-approved mapping is visible from its named existing web
  surface using the existing card/table/reference anatomy.
- Jobs replace the three requested timestamp chips with the Job reference and
  retain the requested accepted-task/action-attempt/source-trace details.
- Errors and Delivery expose the requested background/job and trace fields
  without inventing values absent from persisted records.
- Graph references show the correct labeled IDs for conversation, debug, and
  self-cognition graphs while preserving the graph `run_id` meaning.
- Calendar timestamps remain visible, and schedule/run IDs plus related source
  IDs appear in their existing details/reference surfaces.
- Event details expose separately named request/correlation/tracking/run/
  trigger/attempt fields when the event contains them.
- Known-user rows and selected profiles expose `global_user_id` without adding
  a lookup flow or placing it in a URL.
- Focused tests pass, affected browser flows pass, and screenshots are
  reviewed for unchanged layout anatomy and readable ID presentation.

## Archival instruction

Before archival, record the exact tests, browser URL, screenshot paths, visual
review result, and any fields absent from source records. Then move this plan
to `development_plans/archive/completed/bugfix/` and update
`development_plans/README.md`.

## Completion record

- Backend contract and projection tests:
  `venv\Scripts\python -m pytest tests\test_control_console_cognition_debug_visibility.py tests\test_control_console_kazusa_client.py tests\test_control_console_repository.py tests\test_service_cognition_graph.py -q` — 44 passed, 1 existing Starlette warning.
- Mapped browser flows:
  `venv\Scripts\python -m pytest tests\control_console_e2e\test_page_navigation_e2e.py tests\control_console_e2e\test_cognition_graph_e2e.py tests\control_console_e2e\test_debug_chat_e2e.py -q` — 13 passed.
- Complete control-console E2E collection: 22 passed, 3 opt-in live tests skipped, 1 deselected.
- Static checks: `compileall`, `node --check src\control_console\static\console.js`, and `git diff --check` passed.
- Browser evidence was captured from `http://127.0.0.1:57101` and the mapped graph/debug flows. Reviewed screenshots:
  - `C:\Users\rba90\AppData\Local\Temp\pytest-of-rba90\pytest-897\test_overview_cognition_graph_0\artifacts\conversation_cognition_run_reference.png`
  - `C:\Users\rba90\AppData\Local\Temp\pytest-of-rba90\pytest-897\test_overview_cognition_graph_0\artifacts\self_cognition_run_reference.png`
  - `C:\Users\rba90\AppData\Local\Temp\pytest-of-rba90\pytest-897\test_debug_chat_sends_to_brain0\artifacts\debug_cognition_run_reference.png`
  - `C:\Users\rba90\AppData\Local\Temp\pytest-of-rba90\pytest-897\test_debug_chat_sends_to_brain0\artifacts\debug_chat_id_references.png`
  - `C:\Users\rba90\AppData\Local\Temp\pytest-of-rba90\pytest-898\test_semantic_owner_surfaces_e0\artifacts\calendar_id_references.png`
  - `C:\Users\rba90\AppData\Local\Temp\pytest-of-rba90\pytest-898\test_semantic_owner_surfaces_e0\artifacts\background_jobs_job_reference.png`
  - `C:\Users\rba90\AppData\Local\Temp\pytest-of-rba90\pytest-898\test_semantic_owner_surfaces_e0\artifacts\event_id_references.png`
  - `C:\Users\rba90\AppData\Local\Temp\pytest-of-rba90\pytest-898\test_owner_panels_use_panel_sp0\artifacts\user_global_id_reference.png`
- Visual review: existing console shell, card/table anatomy, graph Run reference disclosure, detail grids, calendar timestamps, and Jobs/Errors/Delivery layout remain aligned; IDs are readable in the requested existing surfaces.
- Aggregate worker tick/error records that do not carry a persisted `background_work_job_id` remain without a fabricated ID. No write-path change, migration, or backfill was added.
