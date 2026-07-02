# control console cognition debug visibility plan

## Summary

- Goal: Add read-only control-console panels that show the same scoped prompt
  windows used by the cognition chain for calendar, background-work,
  conversation progress, global growth, and internal-monologue carry-over
  debugging.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `build-web-apps:frontend-testing-debugging`,
  `browser:control-in-app-browser`
- Overall cutover strategy: additive console-only inspection surfaces with
  atomic route, repository, static UI, docs, and tests updates; no database
  migration, no cognition behavior change, and no compatibility shim layer.
- Highest-risk areas: displaying a different window than cognition receives,
  leaking raw private rows instead of prompt-facing projections, treating
  collection names as model semantics, guessing platform/channel/user scope,
  drifting away from the existing console UI system, and validating only with
  fixture data.
- Acceptance criteria: each Prompt View panel calls the production
  projection/window function used by cognition when that function exists, each
  supporting panel is bounded and redacted, UI changes reuse the existing
  console widgets and styling, real-data screenshots are captured before
  sign-off, and `event_log_snapshots` remains out of scope.

## Context

The control console currently has partial owner pages for Character, Users,
Groups, Calendar, and Background work. It exposes bounded profile, memory,
style, due calendar-run, sanitized event, and cognition-graph detail, but it
does not expose several durable sources that are useful when debugging why the
Kazusa cognition chain did or did not react.

The user-approved collections and runtime states for this feature are:

- `calendar_schedules`
- `background_work_jobs`
- `conversation_episode_state`
- `global_character_growth_runs`
- current carry-over from `internal_monologue_residue_state`

The user explicitly deferred `event_log_snapshots` for this feature because
snapshots are debugging telemetry rather than a cognition prompt window.

The main correction from collection-name thinking is that the console must not
show raw MongoDB rows as if they were cognition inputs. The operator needs the
same semantic projection and scope window that cognition receives. Supporting
panels can show bounded operational backing rows, but they must be visually and
structurally separate from Prompt View panels.

Current relevant production windows:

- Calendar recall candidates are built by
  `kazusa_ai_chatbot.rag.recall.collectors.calendar_runs.CalendarRunCollector`.
- Background result-ready cognition episodes are built from deliverable jobs by
  `kazusa_ai_chatbot.db.background_work_jobs.find_deliverable_background_work_jobs`
  and `kazusa_ai_chatbot.background_work.result_source.build_result_ready_episode_from_job`.
- Conversation progress is loaded by
  `kazusa_ai_chatbot.conversation_progress.runtime.load_progress_context`,
  which calls `conversation_progress.projection.project_prompt_doc`.
- Internal-monologue carry-over is loaded by
  `kazusa_ai_chatbot.internal_monologue_residue.loader.load_residue_context`,
  which calls `internal_monologue_residue.projection.project_residue_window`.
- Promoted global growth prompt context is built by
  `kazusa_ai_chatbot.global_character_growth.context.build_global_character_growth_context`.

## Mandatory Skills

- `development-plan`: load before approving, executing, updating, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing any prompt-facing projection,
  cognition input contract, RAG collector call path, or background cognition
  delivery projection.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `build-web-apps:frontend-testing-debugging`: load before changing or
  visually validating the static control-console frontend.
- `browser:control-in-app-browser`: load before final rendered screenshot
  validation when the Browser plugin is available.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in execution evidence.
- The plan's `Execution Model` uses parent-led native subagent execution. If
  native subagent capability is unavailable, stop before execution unless the
  user explicitly approves fallback execution.
- Do not read `.env` while executing this plan.
- Do not change `/chat`, cognition, RAG semantics, memory promotion,
  reflection, calendar scheduling behavior, background-work generation,
  adapter transport, or prompt wording.
- Prompt View panels must call the production projection/window function that
  feeds cognition when such a function exists. Do not reimplement those
  projections in control-console code.
- Supporting panels may use console-specific safe projections and newly added
  read-only listing helpers. Supporting panels must not be labeled as prompt
  input.
- Route handlers must not use raw MongoDB clients. Raw database access remains
  in existing domain/database repository modules.
- Do not expose raw prompts, embeddings, raw messages, full conversation rows,
  full memory bodies, raw reflection output, raw LLM output, tool arguments,
  idempotency keys, secrets, or unbounded lookup tables.
- Scope-sensitive Prompt View panels must return `needs_input` when platform,
  channel, channel type, or platform user id is missing. Do not infer or reuse
  a previous browser selection to fabricate an exact cognition scope.
- `event_log_snapshots` must not be added to the control console in this plan.
- UI changes must prioritize the existing control-console static framework and
  widgets: Sidebar, Button, Card, Badge, Table, Input, Select, Textarea,
  Separator, ScrollArea, Field/Form grouping, and existing dialog/sheet-style
  surfaces. Do not create a custom widget when an existing widget or renderer
  can carry the information.
- UI changes must align with the current colour scheme, spacing, typography,
  border radius, table density, badge treatment, and card anatomy. Do not add a
  new palette, decorative gradients, custom icons, or a standalone visual
  system.
- New CSS is allowed only to adapt existing components to long prompt-view
  content, wrapping, or responsive layout. Prefer adding a small state or
  modifier class to existing patterns over inventing a new component.
- Final sign-off requires rendered screenshot evidence against real configured
  database data. Fixture-only, static DOM-only, or mocked-server validation is
  not sufficient for completion.
- Real-data screenshots and browser traces must be stored in the test
  artifact/temp directory or shown in the final QA report; do not commit
  screenshots or raw real-data exports to the repository.
- Browser validation must use the in-app Browser plugin when available. If the
  Browser plugin is unavailable, record that reason and use the existing
  Playwright E2E harness for screenshots.

## Must Do

- Add a visible Prompt View versus Operational Backing distinction in the
  backend payloads and static UI.
- Calendar page: add a source-scoped Cognition Pending Runs Prompt View that
  uses `CalendarRunCollector.collect`, and add a supporting Schedule
  Definitions panel for `calendar_schedules`.
- Background work page: add a Result-Ready Cognition Deliveries Prompt View
  that uses `find_deliverable_background_work_jobs` plus
  `build_result_ready_episode_from_job`, and add a supporting Job Queue panel
  for `background_work_jobs`.
- Users page: add exact conversation-scope controls and show Conversation
  Progress Prompt View using `load_progress_context`; show current carry-over
  using `load_residue_context`.
- Groups page: show group-scene carry-over using `load_residue_context`; show
  participant conversation progress only when the operator supplies a platform
  user id for that group/channel scope.
- Character page: show Promoted Global Growth Prompt View using
  `build_global_character_growth_context`; show a Growth Runs Audit panel for
  `global_character_growth_runs`; show character-global current carry-over
  using `load_residue_context`.
- Add or update focused deterministic tests for repository projections, route
  contracts, static page wiring, redaction, and unavailable or missing-scope
  states.
- Add or update rendered UI validation so screenshots prove the new panels fit
  the existing console styling with real database-backed data.
- Update `src/control_console/README.md` so the ICD describes the new
  cognition-debug panels and the continued `event_log_snapshots` exclusion.

## Deferred

- Do not add schedule editing, job retry, job cancellation, event snapshot
  browsing, a generic DB browser, historical internal-monologue browsing, or
  full prompt rendering.
- Do not add new LLM calls.
- Do not change cognition-chain behavior to make the console easier to render.
- Do not expose `event_log_snapshots` in Event monitor or any other page.
- Do not mark all partial console pages as fully `ready` unless their broader
  existing page contracts are also complete.

## Cutover Policy

This is an additive read-only console feature. There is no database migration,
no production behavior cutover, and no rollback data step. The implementation
must update backend routes, repository helpers, static UI, tests, and ICD text
in one coherent change so the browser never depends on a legacy alias payload.
Existing due-run and worker-event panels can move under the new panel envelope
when their tests and renderers move in the same change.

## Target State

The operator can open the web control console and inspect the same scoped
semantic windows that can enter cognition:

- source-scoped pending calendar recall candidates;
- deliverable background-work result-ready cognitive episodes;
- conversation progress prompt docs for a selected platform/channel/user scope;
- current internal-monologue carry-over context for character, group, or
  user-thread scope;
- promoted global growth prompt context.

Each Prompt View panel states its projection owner and scope inputs. Each
Operational Backing panel states that it is not prompt input and shows only
bounded operational fields needed to understand the backing queue or audit
record. The rendered UI uses the existing console Card, Table, Badge, Input,
Select, and FieldGroup patterns; new visual treatment is limited to small
modifiers needed for long prompt-view content.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Prompt View contract | Reuse production window/projection functions when they exist. | The console must debug cognition, not approximate it. |
| Raw collections | Do not expose raw MongoDB documents. | Raw rows contain storage details and can misrepresent what the local LLM sees. |
| Calendar mapping | Show `calendar_runs` recall candidates as Prompt View; show `calendar_schedules` as backing definitions. | Cognition sees pending run evidence, not schedule definitions. |
| Background mapping | Show deliverable result-ready cognitive episodes as Prompt View; show jobs as backing queue. | Cognition receives a `CognitiveEpisode`, not the whole job row. |
| Conversation progress scope | Require platform, channel id, channel type, and platform user id before rendering an exact prompt view. | Runtime scope is platform/channel/global user. |
| Group page participant progress | Group-only page state shows group-scene carry-over; participant progress requires a participant platform user id. | `ConversationProgressScope` is per user within a channel. |
| Global growth mapping | Show promoted active growth context as Prompt View; show `global_character_growth_runs` as audit backing. | Runtime cognition receives promoted traits, not run records. |
| Internal monologue residue | Show only current projected carry-over context and load metadata. | The private collection is useful for state debugging, but history browsing is outside scope. |
| Event snapshots | Keep `event_log_snapshots` out of scope. | The user classified snapshots as debugging telemetry, not valuable first-slice console state. |
| UI composition | Reuse existing console widgets and styling; avoid custom widgets unless an existing Card/Table/Badge/Input/Select/ScrollArea composition cannot represent the panel. | The feature extends a working operational console and should not introduce a parallel UI system. |
| Visual sign-off | Require screenshots from real database-backed console pages before completion. | The user needs rendered evidence that the added debug surfaces work with actual cognition data, not only mocked payloads. |

## Contracts And Data Shapes

All changed entity and lookup routes must return panel envelopes with this
shape for every new panel:

```python
{
    "status": "available | empty | needs_input | unavailable",
    "generated_at": "iso-8601 utc timestamp",
    "content": "production prompt-view document, string, list, or null",
    "items": list[dict],
    "reason": "bounded operator-facing reason",
    "projection_owner": "production function or console projection name",
    "prompt_view": bool,
}
```

Prompt View panels set `prompt_view` to `True`; Operational Backing panels set
it to `False`. Prompt View panels must put exact non-row production outputs in
`content` so dict or string prompt windows are not reshaped into fake table
rows. `items` remains available for production outputs that are naturally row
lists, such as calendar recall candidates and background result-ready episodes.

### Calendar Page

Route: `GET /api/lookups/calendar`

New query parameters:

- `platform`
- `platform_channel_id`
- `platform_user_id`
- `channel_type`
- `limit`

Panels:

- `cognition_pending_runs`: Prompt View. Resolve `platform_user_id` to
  `global_user_id`, then call `CalendarRunCollector().collect` with
  `platform`, `platform_channel_id`, `global_user_id`, and current UTC
  timestamp. Display candidate `source`, `claim`, `temporal_scope`,
  `lifecycle_status`, `evidence_time`, and `authority`.
- `schedule_definitions`: Operational Backing. Add a read-only
  `calendar_scheduler.repository.list_calendar_schedules_for_inspection`
  helper that returns active and paused schedules sorted by `next_run_at`
  ascending, then `schedule_id` ascending. Display schedule id, status, trigger
  kind, next run, source platform, source channel type, and recurrence summary.
  Exclude payload internals, source-scope ids, and idempotency keys.
- `due_runs`: Operational Backing. Preserve current due-run inspection as a
  non-prompt panel using the existing due-run helper and redacted projection.

### Background Work Page

Route: `GET /api/lookups/background`

Panels:

- `result_ready_cognition_deliveries`: Prompt View. Call
  `find_deliverable_background_work_jobs(limit=limit)`, then call
  `build_result_ready_episode_from_job(job)` for each job. Display the episode
  id, trigger source, output mode, target scope summary, percept input source,
  percept content, and prompt-visible metadata keys `task_brief`,
  `failure_summary`, `result_summary`, `worker`, and scalar worker metadata.
  This panel may show `artifact_text` through the percept `content` field
  because that is model-visible result input. It must not show `source_context`,
  tool args, idempotency keys, lease fields, raw route prompts, or raw worker
  outputs.
- `job_queue`: Operational Backing. Add a read-only
  `db.background_work_jobs.list_recent_background_work_jobs` helper sorted by
  `updated_at` descending, then `job_id` ascending. Display job id, status,
  delivery state, worker, created/updated/completed timestamps, delivery
  attempt count, result summary, failure summary, artifact character count,
  source platform, source channel type, and requester display name.
- `worker_events`: Operational Backing. Keep existing sanitized
  `background_work.worker` event telemetry under this panel.

### Users Page

Route: `GET /api/entities/user`

New query parameters:

- `platform_channel_id`
- `channel_type`, constrained by UI to `private` or `group`

Panels:

- `conversation_progress_prompt`: Prompt View. Resolve `platform_user_id` to
  `global_user_id`, build `ConversationProgressScope(platform,
  platform_channel_id, global_user_id)`, then call `load_progress_context`.
  Display `conversation_progress` exactly as returned, plus source metadata
  `source`, `turn_count`, `continuity`, and `status`. Do not render raw
  `episode_state.last_user_input` or storage-only fields.
- `current_carry_over`: Prompt View. Build a residue trigger scope with the
  active character id, platform, platform channel id, selected channel type,
  and resolved global user id. Call `load_residue_context`. Display
  `internal_monologue_residue_context`, `selected_count`, `candidate_count`,
  `scope_order`, and `status`.
- Existing profile, relationship, memory, and style panels remain as bounded
  Operational Backing panels.

### Groups Page

Route: `GET /api/entities/group`

New query parameter:

- `participant_platform_user_id`

Panels:

- `group_carry_over`: Prompt View. Build a residue trigger scope with active
  character id, platform, group id as `platform_channel_id`, `channel_type`
  `group`, and empty global user id. Call `load_residue_context`. Display the
  same fields as the Users page carry-over panel.
- `participant_conversation_progress_prompt`: Prompt View when
  `participant_platform_user_id` is present. Resolve it to `global_user_id`,
  build `ConversationProgressScope(platform, group_id, global_user_id)`, then
  call `load_progress_context`. Return `needs_input` when the participant id is
  absent.
- Existing group style remains an Operational Backing panel.

### Character Page

Route: `GET /api/entities/character`

Panels:

- `promoted_global_growth_prompt`: Prompt View. Call
  `build_global_character_growth_context()` and display the returned runtime
  context exactly as the production projection returns it.
- `current_carry_over`: Prompt View. Load the character profile to get the
  character id using the same `global_user_id` then `CHARACTER_GLOBAL_USER_ID`
  fallback rule used by `service._character_id_from_profile`. Build a residue
  trigger scope with empty platform, channel, channel type, and global user id,
  then call `load_residue_context`.
- `growth_runs_audit`: Operational Backing. Add a read-only
  `db.global_character_growth.list_recent_global_character_growth_runs` helper
  sorted by `updated_at` descending, then `run_id` ascending. Display run id,
  status, started/updated/completed timestamps, processed counts, promoted
  counts, and failure summary. Exclude raw LLM output, prompt payloads, source
  text, and full trait documents.
- Existing profile, state, self-image, learning, and active traits remain
  bounded panels. Existing active traits are not a substitute for the new
  prompt projection panel.

## LLM Call And Context Budget

No new LLM calls are added. No runtime prompt text is changed. The control
console performs additional read-only database/helper calls only when an
authenticated operator refreshes the affected page. Production response-path
latency, token budget, and context composition remain unchanged.

The console can display model-visible background result content only because
that content is already bounded by the background-work job's `max_output_chars`
and is the actual percept content delivered to cognition. Display-only table
layout may wrap or scroll the content, but backend payload construction must
not truncate Prompt View fields and then claim exactness.

## Change Surface

### Modify

- `src/control_console/repository.py`: add injected helper slots, exact Prompt
  View loaders, safe supporting projections, unavailable and `needs_input`
  envelopes, and character-id derivation for residue trigger scopes.
- `src/control_console/app.py`: extend lookup/entity query parameters, audit
  target metadata, and route calls without raw Mongo access.
- `src/control_console/static/index.html`: add scope controls and panels under
  existing Character, Users, Groups, Calendar, and Background pages using
  existing Card, Table, Badge, Input, Select, and FieldGroup anatomy.
- `src/control_console/static/console.js`: fetch new query parameters, render
  Prompt View and Operational Backing panels through existing table, readable
  lookup, detail-grid, and panel-state renderers where possible, and preserve
  missing-scope states.
- `src/control_console/static/console.css`: add compact styles only when
  existing table/card classes cannot render prompt-view text cleanly; do not
  add a new palette or standalone custom widget system.
- `src/control_console/README.md`: document new page capabilities, prompt-view
  boundaries, and excluded event snapshots.
- `src/kazusa_ai_chatbot/calendar_scheduler/repository.py`: add
  `list_calendar_schedules_for_inspection`.
- `src/kazusa_ai_chatbot/db/background_work_jobs.py`: add
  `list_recent_background_work_jobs`.
- `src/kazusa_ai_chatbot/db/global_character_growth.py`: add
  `list_recent_global_character_growth_runs`.
- `tests/test_control_console_repository.py`: add focused panel extraction,
  projection reuse, redaction, and missing-scope tests.
- `tests/test_control_console_contracts.py`: assert route payload panel shape
  and prompt-view/supporting distinction.
- `tests/test_control_console_web_surface.py`: assert static IDs, page labels,
  and forbidden text exposure.
- `tests/control_console_e2e/test_page_navigation_e2e.py`: update navigation
  smoke expectations for new panels when e2e dependencies are available.
- `tests/control_console_e2e/test_live_database_owner_pages_e2e.py`: preserve
  existing live DB owner-page screenshot coverage while aligning expected
  selectors with the new panels.

### Create

- `tests/test_control_console_cognition_debug_visibility.py`: add focused
  tests that monkeypatch production projection/window functions and prove
  Prompt View panels use their outputs rather than duplicated console logic.
- `tests/control_console_e2e/test_live_database_cognition_debug_visibility_e2e.py`:
  add opt-in live DB screenshot validation for the Character, Users, Groups,
  Calendar, and Background cognition-debug panels. The test must discover real
  platform-facing samples from the configured database without mutation and
  write screenshots only to the pytest artifact directory.

### Keep

- Brain `/chat`, cognition graph building, RAG recall semantics, calendar
  scheduler claiming, background-work worker/delivery behavior, conversation
  progress recorder, internal-monologue residue recorder, global growth worker,
  adapters, and database bootstrap collection definitions remain unchanged.

## Overdesign Guardrail

- Actual problem: the operator cannot see the same scoped semantic context
  windows that cognition receives for several durable continuity systems.
- Minimal change: add read-only console panels that call existing production
  projection/window functions for Prompt View and small domain read helpers for
  supporting backing rows.
- Ownership boundaries: production domain modules own raw Mongo access and
  runtime projections; control console owns authentication, route envelopes,
  redaction, display grouping, and static rendering; cognition remains the
  owner of stance, judgment, and response behavior.
- Rejected complexity: generic DB browsing, schedule editing, job mutation,
  event snapshot browsing, historical private-residue browsing, new prompt
  renderers, new LLM calls, compatibility aliases, fallback projections, and
  broad page redesigns, custom UI widgets, new colour palettes, and committed
  screenshot artifacts.
- Evidence threshold: add broader tooling only after an approved follow-up
  plan identifies a current debugging workflow that cannot be answered by the
  exact Prompt View plus bounded backing panels in this plan.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside `control_console`, the
  named read-only domain helper functions, and the named tests as high-scrutiny
  changes.
- The responsible agent must search the codebase for existing equivalent
  projection/window behavior before adding any helper. Existing production
  prompt-facing behavior must be reused instead of duplicated.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, broad refactors, or page redesigns.
- The responsible agent must reuse existing static UI components and renderers
  before adding markup or CSS. Any new CSS selector must be justified by a
  concrete layout need that existing console classes cannot satisfy.
- The responsible agent must capture rendered screenshots with real database
  data before requesting sign-off. If real data, browser tooling, or live DB
  access is unavailable, leave the plan incomplete and record the blocker.
- If implementation discovers that a named production projection function does
  not expose enough data to build an exact Prompt View, stop and report the
  mismatch instead of creating an approximate console projection.

## Implementation Order

1. Parent adds focused failing tests in
   `tests/test_control_console_cognition_debug_visibility.py` for projection
   reuse:
   - calendar Prompt View calls `CalendarRunCollector.collect`;
   - background Prompt View calls
     `build_result_ready_episode_from_job` for deliverable jobs;
   - user and participant progress panels call `load_progress_context`;
   - carry-over panels call `load_residue_context`;
   - character global growth panel calls `build_global_character_growth_context`.
   Run:
   `venv\Scripts\python -m pytest tests\test_control_console_cognition_debug_visibility.py -q`.
   Expected before implementation: missing helper or missing panel failures.
2. Parent adds or updates route and static contract tests in
   `tests/test_control_console_contracts.py`,
   `tests/test_control_console_repository.py`, and
   `tests/test_control_console_web_surface.py`. Run the focused tests and
   record the expected failures.
3. Parent starts exactly one production-code subagent with this approved plan,
   the mandatory skills, the focused test failures, and the production
   ownership boundary: `src/control_console/**` plus the three named read-only
   domain helper functions.
4. Production-code subagent adds the read-only domain helper functions:
   `list_calendar_schedules_for_inspection`,
   `list_recent_background_work_jobs`, and
   `list_recent_global_character_growth_runs`. The helpers must be read-only,
   bounded by caller-supplied `limit`, sorted as specified in this plan, and
   covered by existing or new deterministic tests.
5. Production-code subagent implements control-console repository panel
   loaders and projections. It must call production prompt-window functions for
   Prompt View panels and return `needs_input` instead of guessing missing
   scope.
6. Production-code subagent wires FastAPI route parameters and audit metadata
   in `src/control_console/app.py`.
7. Production-code subagent updates static HTML, JS, and CSS for the new panels
   within existing pages. It must first compose existing console widgets and
   renderers, then add only minimal modifier CSS for long prompt-view content
   or responsive wrapping.
8. Production-code subagent updates `src/control_console/README.md` page
   capability text and forbidden/excluded data notes.
9. Parent reruns focused repository, route, and web-surface tests. If they
   fail because a contract in this plan is incomplete, update tests or code
   only within this plan's change surface and rerun the focused tests.
10. Parent runs the rendered UI validation with real configured database data
    and captures screenshots for Character, Users, Groups, Calendar, and
    Background. Use the in-app Browser plugin when available; otherwise use the
    existing Playwright E2E harness and record the Browser unavailability.
11. Parent runs the full verification command set in `Verification`.
12. Parent starts exactly one independent code-review subagent after planned
    verification passes. Review findings are remediated only inside this plan's
    change surface, followed by rerunning affected verification commands.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it; closes after planned
  production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused prompt-window test contract established.
  - Covers: Implementation Order steps 1 and 2.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_control_console_cognition_debug_visibility.py tests\test_control_console_repository.py tests\test_control_console_contracts.py tests\test_control_console_web_surface.py -q`
  - Evidence: record expected failures or baseline results before production
    implementation.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: record agent and date in execution evidence after verification.
- [x] Stage 2 - read-only domain helper functions complete.
  - Covers: Implementation Order step 4.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_calendar_scheduler_repository.py tests\test_background_work_jobs.py tests\test_global_character_growth_context.py -q`
  - Evidence: record changed helper files and test output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: record agent and date in execution evidence after verification.
- [x] Stage 3 - control-console backend panels and routes complete.
  - Covers: Implementation Order steps 5 and 6.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_control_console_cognition_debug_visibility.py tests\test_control_console_repository.py tests\test_control_console_contracts.py -q`
  - Evidence: record route payload shape and prompt-window reuse test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: record agent and date in execution evidence after verification.
- [x] Stage 4 - static UI and ICD updates complete.
  - Covers: Implementation Order steps 7 and 8.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py tests\control_console_e2e\test_page_navigation_e2e.py -q`
  - Evidence: record static-surface and navigation test output, including
    confirmation that existing widgets, colour scheme, spacing, and renderer
    patterns are reused.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: record agent and date in execution evidence after verification.
- [x] Stage 5 - real-data rendered screenshot sign-off complete.
  - Covers: Implementation Order step 10.
  - Verify:
    `$env:KAZUSA_RUN_CONTROL_CONSOLE_LIVE_DB_E2E='1'; venv\Scripts\python -m pytest tests\control_console_e2e\test_live_database_cognition_debug_visibility_e2e.py -q`
  - Evidence: record the generated screenshot paths, masked sample identifiers,
    page/panel coverage, viewport coverage, and console error status. If live
    DB data or browser tooling is unavailable, leave this stage unchecked and
    record the blocker.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: record agent and date in execution evidence after verification.
- [x] Stage 6 - full verification complete.
  - Covers: Implementation Order step 11.
  - Verify: run every command in `Verification`.
  - Evidence: record command output summaries. Real-data screenshot validation
    must have passed before this stage can be checked.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: record agent and date in execution evidence after verification.
- [x] Stage 7 - independent code review complete.
  - Covers: Implementation Order step 12 and `Independent Code Review`.
  - Verify: review findings resolved or explicitly recorded as no findings;
    rerun affected verification commands after any fix.
  - Evidence: record reviewer role, files reviewed, findings, fixes, reruns,
    residual risk, and approval status.
  - Handoff: plan can be signed off only after this stage is complete.
  - Sign-off: record agent and date in execution evidence after verification.

## Verification

### Static Checks

- `git diff --check`
  - Expected: no whitespace errors.
- `rg "event_log_snapshots" src/control_console`
  - Expected: no matches. For `rg`, exit code 1 is acceptable when there are no
    matches.
- `rg "source_context|tool_args|idempotency_key|raw_llm_output" src/control_console/static src/control_console/app.py`
  - Expected: no matches. Field names may appear in backend redaction tests or
    repository exclusion code only when the test asserts they are not rendered.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_control_console_cognition_debug_visibility.py -q`
- `venv\Scripts\python -m pytest tests\test_control_console_repository.py tests\test_control_console_contracts.py tests\test_control_console_web_surface.py -q`
- `venv\Scripts\python -m pytest tests\test_calendar_scheduler_repository.py tests\test_background_work_jobs.py tests\test_global_character_growth_context.py tests\test_internal_monologue_residue_loader.py tests\test_conversation_progress_runtime.py tests\test_background_work_delivery.py tests\test_rag_recall_agent.py -q`

### Console E2E

- `venv\Scripts\python -m pytest tests\control_console_e2e\test_page_navigation_e2e.py tests\control_console_e2e\test_live_database_owner_pages_e2e.py tests\control_console_e2e\test_visual_product_acceptance_e2e.py -q`
  - Expected: pass for non-live E2E tests. The live DB owner-pages test remains
    skipped unless `KAZUSA_RUN_CONTROL_CONSOLE_LIVE_DB_E2E=1` is set, and does
    not replace the required cognition-debug live DB screenshot gate below.

### Rendered UI Sign-Off With Real Data

- `$env:KAZUSA_RUN_CONTROL_CONSOLE_LIVE_DB_E2E='1'; venv\Scripts\python -m pytest tests\control_console_e2e\test_live_database_cognition_debug_visibility_e2e.py -q`
  - Expected: pass against the configured real MongoDB without mutating data.
    The test must discover real platform-facing user, group, calendar, and
    background-work samples where available, load the authenticated console,
    exercise the affected pages, assert no relevant browser console errors,
    and capture screenshots for Character, Users, Groups, Calendar, and
    Background.
  - Completion rule: this command is required for final sign-off. If the live
    database has no representative rows for an affected panel, if browser
    automation is unavailable, or if the console cannot be launched with the
    configured database, record the blocker and leave the plan incomplete.
- Manual Browser sign-off, when the in-app Browser plugin is available:
  - Open the local console, authenticate with the test/operator token for the
    validation run, and capture desktop screenshots for Character, Users,
    Groups, Calendar, and Background after loading real database-backed data.
  - Capture one narrow/mobile screenshot for the densest changed page.
  - Verify visually that controls and panels use existing widgets, current
    colours, current spacing, readable wrapping, and no overlapping text.
  - Store screenshots outside the repo or present them in the QA final report.

### Regression

- `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q`
  - Expected: pass. If runtime exceeds the session budget, record the completed
    focused tests plus the exact broader command status before stopping.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test,
  documentation, HTML, CSS, and JavaScript artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including focused and regression tests,
  execution evidence, and path-safe commands for directories containing spaces.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture or documentation corrections. If a fix would cross the approved
boundary or alter the contract, stop and update the plan or request approval
before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
execution evidence.

## Acceptance Criteria

This plan is complete when:

- Calendar Prompt View shows the same source-scoped pending calendar candidates
  produced by `CalendarRunCollector.collect`.
- Background Prompt View shows the same result-ready cognitive episodes built
  from deliverable `background_work_jobs`.
- User and group participant Conversation Progress panels show the exact
  `conversation_progress` prompt doc returned by `load_progress_context` for
  the selected runtime scope.
- Character, user, and group carry-over panels show only the current
  `internal_monologue_residue_context` returned by `load_residue_context` plus
  load metadata.
- Character Promoted Global Growth Prompt View shows the exact runtime context
  returned by `build_global_character_growth_context`.
- `calendar_schedules`, `background_work_jobs`, and
  `global_character_growth_runs` are visible only through bounded Operational
  Backing panels.
- `event_log_snapshots` remains absent from the control console.
- All new routes remain authenticated, read-only, bounded, and redacted.
- Verification commands pass or record an explicit e2e dependency blocker
  for non-sign-off E2E only. Real-data screenshot validation is required for
  completion and cannot be replaced by fixture-only tests.
- Rendered screenshots from real database-backed data show the affected
  Character, Users, Groups, Calendar, and Background pages using existing
  console widgets, current colour scheme, and readable responsive layout.
- Independent code review is complete and all blocking findings are resolved.

## Execution Evidence

- Focused feature tests passed:
  `venv\Scripts\python -m pytest tests\test_control_console_cognition_debug_visibility.py -q`
- Console regression tests passed:
  `venv\Scripts\python -m pytest tests\test_control_console_repository.py tests\test_control_console_web_surface.py tests\test_control_console_review_edges.py tests\test_console_lookup_limits.py tests\test_control_console_cognition_debug_visibility.py -q`
- Adjacent domain helper tests passed:
  `venv\Scripts\python -m pytest tests\test_calendar_scheduler_repository.py tests\test_background_work_jobs.py tests\test_global_character_growth_context.py -q`
- Syntax check passed:
  `venv\Scripts\python -m py_compile src\control_console\repository.py src\control_console\app.py src\control_console\contracts.py src\kazusa_ai_chatbot\calendar_scheduler\repository.py src\kazusa_ai_chatbot\db\background_work_jobs.py src\kazusa_ai_chatbot\db\global_character_growth.py`
- Diff whitespace check passed with line-ending warnings only:
  `git diff --check`
- Real database-backed screenshots captured with local Chrome at
  `http://127.0.0.1:8766/` using a masked QQ group scope:
  `platform=qq`, `platform_channel_id=<masked>`,
  `platform_user_id=<masked>`, `channel_type=group`.
- Screenshot artifacts were moved outside the repository working tree:
  `%TEMP%\kazusa_control_cognition_debug_screenshots_20260624\screenshots\01_character_prompt_panels.png`
  `%TEMP%\kazusa_control_cognition_debug_screenshots_20260624\screenshots\02_user_prompt_panels.png`
  `%TEMP%\kazusa_control_cognition_debug_screenshots_20260624\screenshots\03_group_prompt_panels.png`
  `%TEMP%\kazusa_control_cognition_debug_screenshots_20260624\screenshots\04_calendar_prompt_operational_panels.png`
  `%TEMP%\kazusa_control_cognition_debug_screenshots_20260624\screenshots\05_background_prompt_operational_panels.png`
- Visual review: affected Character, Users, Groups, Calendar, and Background
  pages render with existing console cards, tables, inputs, badges, and current
  colour scheme; no incoherent overlap was observed in the captured desktop
  screenshots.
- Independent review findings resolved:
  screenshot evidence sanitized and moved outside the repo; unavailable
  background event sentinel now remains unavailable; checklist and registry
  updated after review; panel contract/scope metadata is emitted and rendered;
  malformed result-ready background jobs no longer 500 the lookup; Calendar and
  Background copy now describes Prompt View plus Operational Backing surfaces.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Console displays a non-production approximation of cognition context | Prompt View tests monkeypatch production projection/window functions and assert returned panel data comes from those functions | `tests/test_control_console_cognition_debug_visibility.py` |
| Missing scope creates misleading empty panels | Scope-sensitive panels return `needs_input` until platform, channel, channel type, and user inputs required by the runtime are present | Repository and route tests for missing scope |
| Private or raw operational data leaks into the browser | Separate Prompt View from Operational Backing, restrict fields, and run static forbidden-field checks | Static checks plus web-surface tests |
| Domain read helper changes affect runtime behavior | New helpers are read-only, have no callers outside console/tests in this plan, and do not modify existing scheduler/worker paths | Domain helper tests plus regression suite |
| Static UI becomes crowded or hard to scan | Use existing page layout, cards, tables, and compact wrapping without a page redesign | Web-surface tests and e2e visual acceptance |
| Custom UI drift creates a second console design language | Require existing widgets, current colour scheme, and minimal modifier CSS only | Static web-surface tests plus real-data screenshots |
| Fixture-only validation misses real prompt-window layout pressure | Require opt-in live DB screenshot validation before sign-off | `test_live_database_cognition_debug_visibility_e2e.py` plus Browser screenshot report |
