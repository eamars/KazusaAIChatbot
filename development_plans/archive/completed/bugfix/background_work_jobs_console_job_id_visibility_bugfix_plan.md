# background work jobs console job id visibility bugfix plan

## Summary

- Goal: Make the existing background-work job ID available to the operator in
  the existing `Background Work` -> `Jobs` cards.
- Status: completed
- Scope boundary: The bounded control-console projection and renderer for the
  `panels.jobs` collection only.
- Change direction: Add the existing persisted `job_id` as one collapsed
  reference disclosure inside each existing Jobs card, using the current
  cognition `Run reference` presentation pattern.
- Acceptance state: Completed with focused deterministic tests, rendered
  browser validation, and screenshot evidence.

## Scope And Change Direction

The background-work repository already owns and persists `job_id`. The control
console currently projects the job's worker, delivery, timestamps, status, and
bounded summaries while omitting that identifier from the browser projection.
The existing Jobs renderer already owns the record-card layout.

The target state is an additive operator-facing reference in each existing
`Background Work` -> `Jobs` card:

- Keep the existing card, title, status, worker, delivery, created, completed,
  and updated values in their current presentation.
- Add one closed `Job reference` disclosure using the existing
  `graph-run-reference` / `Run reference` visual pattern.
- Place the exact non-empty source `job_id` in the disclosure's escaped `code`
  value.
- Preserve the existing card count, ordering, spacing, and page structure.
- Keep `Background Work` -> `Errors` and `Delivery detail` in their current
  visible and API projection shapes.

The ownership boundary is `control_console`: its read-only repository
projection determines the bounded browser data, and its static renderer
determines the existing Jobs-card presentation. The
`kazusa_ai_chatbot.background_work` package remains the owner of job storage,
creation, worker execution, completion, and delivery behavior.

## Mandatory Skills

- `development-plan`: lifecycle, scope, approval, and acceptance contract.
- `control-console-web-development`: control-console projection, static
  renderer, and rendered browser validation.
- `py-style`: applies when editing the Python repository projection.
- `test-style-and-execution`: applies when adding or changing deterministic or
  browser tests.

## Mandatory Rules

- Work in the development workspace `C:\workspace\kazusa_ai_chatbot`.
- Use the existing authenticated, bounded
  `GET /api/lookups/background-work` contract and its current limit behavior.
- Include `job_id` in the `panels.jobs.items` projection through an explicit
  Jobs-only projection path. The `panels.delivery_detail` projection retains
  its current fields, and worker-event `panels.errors` retains its current
  fields.
- Reuse the existing record-card and cognition reference presentation. The
  plan adds no page, panel, widget family, stylesheet system, copy action,
  filter, lookup flow, or new live stream.
- Render the value through the existing HTML escaping path. A missing or empty
  source identifier produces no synthetic ID or replacement value.
- The persisted job document, worker lifecycle, delivery path, chat path,
  cognition path, and agent skill behavior remain unchanged.

## Must Do

1. Update the control-console background-job projection so the Jobs panel
   carries the existing `job_id` field while the other background panels keep
   their current projection contracts.
2. Update the existing Jobs-card renderer to add the collapsed Job reference
   disclosure while retaining every current card field and presentation.
3. Add focused deterministic coverage for the Jobs-only API projection and
   preservation of the delivery-detail redaction boundary.
4. Add rendered control-console coverage proving that a Jobs card shows its
   source job ID after opening the reference disclosure and still shows the
   existing worker, delivery, created, completed, and updated values.
5. Validate the affected Background Work page in a fresh browser context using
   the configured control-console runtime, recording the actual URL, session
   state, interaction, screenshot or equivalent evidence, and console/page
   error state.

## Deferred

- IDs in Latest conversation cognition, Calendar, Users, Groups, Events,
  Errors, Delivery detail, or any other console view.
- Trace lookup, agent skill changes, correlation-manifest changes, new API
  routes, new identifiers, copy buttons, filters, search, or navigation.
- Changes to the existing time fields or any replacement of the current Jobs
  card content.
- Changes to background-work persistence, worker execution, delivery,
  accepted-task behavior, chat behavior, or cognition behavior.
- Database schema changes, data backfills, index changes, or migration steps;
  the plan reads the existing persisted field in place.

## Target State

For a source job containing `job_id: "job-123"`, the bounded lookup contract
has this relevant shape:

```json
{
  "panels": {
    "jobs": {
      "items": [
        {
          "job_id": "job-123",
          "worker": "...",
          "delivery_state": "...",
          "created_at": "...",
          "completed_at": "...",
          "updated_at": "..."
        }
      ]
    },
    "delivery_detail": {
      "items": [
        {
          "worker": "...",
          "delivery_state": "..."
        }
      ]
    }
  }
}
```

The Jobs card displays the existing values plus a closed reference disclosure
whose code value is `job-123`. The disclosure is present only when the source
row has a non-empty `job_id`.

## Change Surface

### Delete

None.

### Modify

- `src/control_console/repository.py`
  - Update `_project_background_job` and its Jobs-panel call path so the
    existing `job_id` is included only in `panels.jobs.items`.
  - Preserve the current allowlist, redaction, bounded summaries, and
    delivery-detail projection behavior for every other field and panel.
- `src/control_console/static/console.js`
  - Update `renderBackgroundJobs` and its Jobs-panel call site to render the
    one reference disclosure inside existing record cards.
  - Keep the Errors and Delivery detail call sites visually and semantically
    unchanged.
- `tests/test_control_console_cognition_debug_visibility.py`
  - Extend the focused background lookup contract assertions for Jobs-only
    `job_id` visibility and delivery-detail preservation.
- `tests/control_console_e2e/test_page_navigation_e2e.py`
  - Extend the existing Background Work navigation fixture and assertions for
    the visible Jobs-card reference and retained fields.

### Create

None.

### Keep

- `src/control_console/static/index.html` and the existing Background Work
  page structure.
- `src/control_console/static/console.css` and the existing reference/card
  styling.
- The current `/api/lookups/background-work` route, authentication, CSRF
  policy, limits, and panel names.
- All background-work storage and runtime ownership in
  `src/kazusa_ai_chatbot/background_work`.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Persisted background jobs | compatible | Read the existing `job_id` in place; stored documents and indexes remain unchanged. |
| Control-console API | compatible | Preserve all existing fields and panels; add `job_id` only to `panels.jobs.items`. |
| Jobs UI | compatible | Preserve the current cards and values; add one collapsed reference disclosure. |
| Runtime behavior | compatible | Keep job creation, execution, completion, delivery, chat, and cognition behavior unchanged. |

## Agent Autonomy Boundaries

The implementation agent may choose local parameter names, helper extraction,
test fixture details, and exact command order when those choices preserve the
Jobs-only contract and existing visual pattern.

The implementation agent must keep the change within the listed files and
interfaces. A need to expose the identifier in another panel, alter the route
shape outside `panels.jobs.items`, change the card layout, add a new widget,
or modify persistence/runtime behavior requires a plan amendment before work
continues.

## Verification

- Run the focused repository projection tests with the project virtual
  environment and the repository's test execution contract.
- Run the affected control-console browser test with the project-supported
  browser path.
- In the rendered Background Work page, expand one Job reference and verify
  the exact source `job_id`; verify the existing worker, delivery, created,
  completed, and updated values remain present.
- Verify Errors and Delivery detail retain their existing visible content and
  do not acquire the Jobs reference disclosure.
- Check authenticated request success, page identity, actual runtime URL,
  session assumptions, console errors, page errors, and no horizontal overflow
  in the affected page.
- Record the focused test results and browser evidence in the execution record
  before completion.

## Acceptance Criteria

- An operator using only the web console can open `Background Work` -> `Jobs`,
  expand the existing-style Job reference disclosure, and read the exact
  persisted job ID for each populated Jobs card.
- The Jobs card retains its existing title, status, worker, delivery, created,
  completed, updated, and other bounded details.
- The API exposes `job_id` only in the Jobs panel projection covered by this
  plan; Errors and Delivery detail retain their current contracts.
- No new page, panel, widget, route, lookup flow, identifier, persistence
  behavior, background-worker behavior, migration, or live response behavior
  is introduced.
- Focused deterministic tests and rendered browser validation pass with no
  new console/page errors or sensitive-field exposure.

## Progress Checklist

- [x] Added `job_id` to the bounded `panels.jobs.items` projection only.
- [x] Added one existing-style Job reference disclosure to Jobs cards while
  retaining the current card fields and layout.
- [x] Preserved the current Errors and Delivery detail projection and visual
  surfaces.
- [x] Added deterministic projection assertions and rendered browser
  assertions.
- [x] Captured and visually inspected the Background Work screenshot.

## Execution Evidence

- Baseline reproduction: the focused background lookup test passed before the
  change with `job_id` absent from the browser projection.
- Deterministic verification:
  `venv\Scripts\python -m pytest
  tests\test_control_console_cognition_debug_visibility.py
  tests\test_control_console_web_surface.py -q` passed with 23 tests passed
  and one existing Starlette/httpx deprecation warning.
- Browser verification:
  the in-app Browser connector was unavailable, so the project Playwright
  fallback launched the isolated console with system Chrome at
  `http://127.0.0.1:64608`. The authenticated E2E session opened
  `Background work`, expanded `Job reference`, verified the exact
  `job-console-001` value and all existing worker/delivery/time values, and
  verified Errors and Delivery detail had no Job reference disclosure.
- Browser test result:
  `venv\Scripts\python -m pytest
  tests\control_console_e2e\test_page_navigation_e2e.py::test_semantic_owner_surfaces_exclude_internal_projection_metadata
  --basetemp test_artifacts\control_console_jobs_e2e -q -s` passed.
- Adjacent browser regression:
  `venv\Scripts\python -m pytest
  tests\control_console_e2e\test_page_navigation_e2e.py
  --basetemp test_artifacts\control_console_page_navigation_e2e -q -s` passed
  with all 7 page-navigation tests passing.
- Screenshot evidence:
  `test_artifacts\control_console_jobs_e2e\test_semantic_owner_surfaces_e0\artifacts\background_jobs_job_reference.png`.
  Visual inspection confirmed the existing dense card layout, spacing,
  badges, chips, and panel anatomy remain intact; the job ID appears once in
  the expanded reference disclosure.
- Browser diagnostics: no `browser.console.log` artifact was emitted, and the
  E2E page-error/console-error collector remained empty.
- `git diff --check` completed with line-ending normalization warnings only.
- Persistence evidence: the change reads the existing `job_id`; stored
  documents, indexes, worker behavior, delivery behavior, and migrations are
  unchanged.
