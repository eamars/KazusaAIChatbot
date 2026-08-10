# control console web availability follow-up

## Summary

- Goal: make the committed Control Console web projections operationally
  available in the live console and make deployed static revisions revalidate
  after a console restart.
- Plan class: medium.
- Status: completed.
- Mandatory skills: `development-plan`, `control-console-web-development`,
  `py-style`, `test-style-and-execution`, and `python-venv`.
- Overall cutover strategy: apply one bounded code/test change, restart the
  console only after verification, then authenticate against the live console
  and inspect the affected API projections.
- Highest-risk areas: preserving the native V2 timestamp contract, avoiding
  invented interaction-style guidance, and changing only console-owned web
  delivery behavior.
- Acceptance criteria: character operational posture and current-user causal
  relationship projections return their native data for valid V2 state;
  root/static console responses advertise revalidation; user/group style
  panels visibly preserve relevance/cognition/surface source rows and truthful
  `missing` source status; focused and regression tests pass; live API smoke
  confirms the corrected projections after the updated console is running.

## Context

The completed short-horizon V2 plan delivered the Character operational
posture, User causal relationship, and User/Group consumer-labelled style
surfaces. The live console at `http://localhost:8764/` serves the current
static assets, but its Character API returns:

```text
operational_posture.status = unavailable
operational_posture.reason = character operational state projection is unavailable
```

The console repository passes `datetime.now(timezone.utc).isoformat()` into the
V2 projection boundary. That produces a `+00:00` suffix while the projection
parser accepts the canonical terminal `Z` form. The same boundary affects the
User causal relationship panel.

The live User and Group style endpoints already return the new role-labelled
rows, but their source documents are absent in the configured production data,
so the rows correctly report `status: missing`. This plan preserves that
truthful source state and does not fabricate learned guidance or write a style
seed.

The live static asset URLs are unversioned and currently have no explicit
revalidation policy. The console serves the correct current files, but an
already-open browser can retain an older in-memory/static response. The
console-owned root and static responses will request revalidation on subsequent
loads.

## Mandatory Skills

- `development-plan`: governs this follow-up plan, parent-led execution,
  evidence, review, and lifecycle closeout.
- `control-console-web-development`: load before changing or validating the
  Control Console API or static UI.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `python-venv`: use the project virtual environment for Python verification.

## Mandatory Rules

- The Control Console remains a read-only projection and delivery boundary for
  cognition data; it does not change cognition, RAG, style producers, or
  database semantics.
- Preserve the native `character_operational_state_view.v1`,
  `relationship_operational_context.v1`, and
  `cognition_context_consumption.v1` contracts.
- Convert only the accepted UTC `+00:00` representation at the console's
  operational-projection boundary. Invalid timestamps continue to fail closed
  through the existing typed unavailable result.
- Keep interaction-style source ownership intact. `missing` means the
  canonical style helper found no source document; it is not a reason to
  invent default guidelines.
- Root/static cache behavior may affect only browser asset delivery. It must
  not alter API response caching, authentication, service lifecycle, or brain
  behavior.
- Use `venv\Scripts\python.exe` for Python and pytest commands.
- Use `apply_patch` for manual file edits and preserve unrelated worktree
  changes.
- The parent owns tests, verification, evidence, review remediation, and
  lifecycle updates. The production-code subagent edits only the approved
  production files.
- After any context compaction, reread this entire plan before continuing.
- After each major checklist sign-off, reread this entire plan before the next
  stage.
- Before completion, run the final review gate and record its result in
  `Execution Evidence`.

## Must Do

- Normalize accepted UTC `+00:00` timestamps before character operational and
  user relationship operational V2 projection calls.
- Add deterministic tests proving both affected projections are available when
  the only difference is the console-supplied `+00:00` effective timestamp.
- Add a Control Console web-surface test proving root and static responses
  request cache revalidation.
- Preserve and verify the existing role-labelled User/Group style rendering,
  including explicit `missing` source rows.
- Run focused tests, the Control Console deterministic regression set, and a
  live authenticated API smoke against the updated console process.
- Perform one final review after verification and remediate findings inside
  this change surface before closeout.

## Deferred

- Do not seed, mutate, migrate, or delete production interaction-style image
  documents.
- Do not change style-image producers, learning cadence, overlays, or source
  status semantics.
- Do not alter the V2 projection parser, emotion formulas, relationship
  reducers, cognition graph payload, or brain service.
- Do not redesign the static UI, add a frontend build system, or introduce
  cache-busting frameworks.
- Do not change unrelated console timestamps, service lifecycle behavior,
  authentication, API cache policy, or database availability reporting.
- Do not add compatibility aliases, fallback guidance, or invented default
  style values.

## Cutover Policy

- The source tree is the implementation authority; the live console process is
  updated only after focused and regression verification passes.
- Restart the console process after verification so it serves the changed
  Python boundary and response headers. The brain and adapters remain outside
  this change unless the console lifecycle requires their existing managed
  shutdown/start behavior.
- Existing browser tabs require one reload after the console restart. The new
  revalidation headers ensure subsequent navigation checks current assets.
- A live style row with `status: missing` is an accepted, truthful data state;
  the web gate passes when the role/source row is visible and no guidance is
  fabricated.

## Target State

- `GET /api/entities/character` returns an `available` operational posture
  panel with persisted/effective views and latest graph consumption whenever
  the native character state is valid.
- `GET /api/entities/users/{platform}/{id}` returns an `available`
  `relationship_operational` panel whenever the native user state is valid and
  uses the same canonical effective-time representation.
- User and Group pages render the `relevance`, `cognition`, and `surface`
  source-labelled style rows. Missing source documents render a visible
  `missing` status and bounded empty-guidance message.
- `GET /` and `/static/*` responses include `Cache-Control: no-cache,
  must-revalidate`, allowing browser revalidation without changing API
  semantics.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| UTC projection boundary | Normalize `+00:00` to terminal `Z` in the console repository immediately before V2 operational projection calls. | Keeps the frozen V2 parser contract and fixes the web-owned representation mismatch. |
| Style source absence | Preserve `missing` source rows and render them explicitly. | Learned style data is database-owned; the console cannot invent semantic guidance. |
| Asset freshness | Add revalidation headers to the root and static asset responses. | Fixes stale browser delivery without adding a build or versioning system. |
| Database scope | No database writes or migration. | The observed style state is truthful optional-source absence, not a schema failure. |

## Change Surface

### Modify

- `src/control_console/repository.py`: normalize the effective timestamp at
  the two operational V2 projection owners.
- `src/control_console/app.py`: add root/static response revalidation headers.
- `tests/test_control_console_repository.py`: add focused timestamp-boundary
  coverage for character and user operational panels.
- `tests/test_control_console_web_surface.py`: assert the cache policy and
  existing role-labelled style surface markers.

### Create

- `development_plans/archive/completed/bugfix/control_console_web_availability_followup_plan.md`:
  this completed follow-up record.

### Keep

- `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py`: the native
  V2 parser and projection contract remain unchanged.
- `src/kazusa_ai_chatbot/db/interaction_style_images.py`: canonical style
  loading and source status remain unchanged.
- `src/control_console/static/index.html`, `console.js`, and `console.css`:
  the role-labelled UI remains the current contract; response revalidation
  makes the existing assets reliably reloadable.

## Overdesign Guardrail

- Actual problem: the live console fails to expose valid native operational
  state because its effective timestamp representation is rejected, and an
  existing browser can retain old static assets.
- Minimal change: normalize the console-owned effective timestamp, add focused
  regression tests, and set root/static revalidation headers.
- Ownership boundaries: the console owns timestamp transport and asset
  delivery; V2 owns semantic projection; the interaction-style store owns
  learned source data.
- Rejected complexity: no parser contract expansion, DB seed, fallback style,
  asset build system, frontend framework, API cache layer, or unrelated
  timestamp refactor.
- Evidence threshold: a future style-data change requires an explicit style
  producer/data plan and source evidence, not a console fallback.

## Agent Autonomy Boundaries

- The production-code subagent may choose local mechanics only inside the
  listed files and contracts.
- It must not edit tests, plan evidence, style producers, database code, brain
  code, or unrelated console routes.
- It must not introduce helpers, flags, aliases, compatibility paths, or
  default semantic values beyond the exact timestamp and cache requirements.
- The parent may remediate review findings only inside this change surface.
- A finding that requires changing the native V2 or style-data contract stops
  execution and requires a new plan or user direction.

## Implementation Order

1. Parent adds the focused repository and web-surface assertions.
2. Parent runs those tests and records the pre-fix failure for the timestamp
   assertion while confirming the existing style marker contract.
3. Parent starts exactly one DeepSeek production-code subagent with ownership
   of `src/control_console/repository.py` and `src/control_console/app.py`.
4. Parent runs the focused tests and Control Console deterministic regression
   tests while the production subagent works.
5. Parent reviews the production diff, runs focused tests again, then runs the
   live authenticated API smoke against the updated process.
6. Parent performs the final review after planned verification passes.
7. Parent remediates in-scope findings, reruns affected checks, records
   evidence, and closes this plan.

## Execution Model

- Parent agent owns orchestration, test changes, verification, live smoke,
  evidence, review remediation, lifecycle updates, and final sign-off.
- Production-code subagent: exactly one native DeepSeek subagent, started after
  the focused test contract and expected failure are established; owns only
  the two production files listed above and does not edit tests.
- Review gate: parent direct review after verification; the user explicitly
  waived a review subagent for this simple change.
- The parent remains responsible for the final live process restart and
  user-facing handoff.

## Progress Checklist

- [x] Stage 1 - plan and focused test contract established.
  - Covers: discovery, this plan, repository timestamp assertions, cache-header
    assertions, and pre-fix test output.
  - Verify: focused pytest command and static style-marker assertions.
  - Evidence: record failure/pass output and changed test paths.
  - Handoff: production subagent starts only after this stage.
  - Sign-off: parent after evidence is recorded.
- [x] Stage 2 - console production fixes implemented.
  - Covers: repository projection timestamp normalization and app cache policy.
  - Verify: focused tests pass and diff is limited to the change surface.
  - Evidence: changed-file diff and focused output.
  - Handoff: regression and live smoke verification.
  - Sign-off: parent after the production subagent closes.
- [x] Stage 3 - deterministic and live web verification complete.
  - Covers: Control Console regression tests, current static markers, API panel
    availability, and truthful style source rows.
  - Verify: all listed deterministic commands pass; live authenticated API
    smoke observes available operational panels and role-labelled styles.
  - Evidence: commands, counts, statuses, and process version are recorded.
  - Handoff: independent code review.
  - Sign-off: parent after verification.
- [x] Stage 4 - independent code review and closeout.
  - Covers: review findings, remediation, final test reruns, plan evidence,
    registry lifecycle, and final status.
  - Verify: parent direct review and affected checks rerun, per the user's
    explicit instruction to omit a review subagent for this simple change.
  - Evidence: direct review findings, fixes, residual risks, and final live
    smoke result.
  - Handoff: completed plan is historical; future style-data work needs a new
    plan.
  - Sign-off: parent after completion.

## Verification

### Focused tests

- `venv\Scripts\python.exe -m pytest tests\test_control_console_repository.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_control_console_web_surface.py -q`

### Regression tests

- `venv\Scripts\python.exe -m pytest tests\test_control_console_repository.py tests\test_control_console_web_surface.py tests\test_control_console_cognition_graph.py tests\test_control_console_redaction.py -q`

### Static checks

- `rg -n "character-operational-posture-table|user-relationship-operational-table|relevance · cognition · surface" src\control_console\static\index.html` must return the existing Character/User/Group markers.
- `git diff --check` must return zero errors.
- `git status --short` must contain only this plan and the approved source/test changes.

### Live smoke

- Authenticate to `http://localhost:8764/` with the operator token already
  supplied by the user.
- Confirm `GET /api/entities/character` returns an available
  `operational_posture` panel.
- Confirm one User detail returns an available `relationship_operational`
  panel and one Group/User style detail returns role-labelled rows.
- Confirm style source rows remain `missing` when the production source
  documents remain absent.
- Confirm `/` and `/static/console.js` return the revalidation header.
- Record that the in-app browser is unavailable if it remains unavailable;
  direct authenticated HTTP smoke is the fallback evidence for this run.

## Independent Code Review

The user explicitly waived a review subagent for this simple change. The parent
performed the final direct review after verification, checking Python style,
exception boundaries, timestamp ownership, cache scope, style-source truth,
privacy/redaction, unrelated-file drift, plan lifecycle, service restoration,
and acceptance-criterion coverage. No findings remained; the affected
regression set was rerun and passed.

## Acceptance Criteria

This plan is complete when:

- Character operational posture is available for valid production V2 state.
- User causal relationship operational context is available for valid user V2
  state.
- Root and static console responses request cache revalidation.
- User and Group style cards visibly render relevance/cognition/surface source
  rows with truthful `missing` status where the data source is absent.
- Focused and regression tests pass.
- Live authenticated HTTP smoke passes after the updated console process is
  running.
- Final direct review approves the final diff and evidence.

## Execution Evidence

- Plan/discovery: live HTTP audit identified the strict native-Z timestamp
  mismatch in both operational projection owners, missing root/static cache
  policy, and truthful absent style-source documents. The in-app browser was
  unavailable; direct authenticated HTTP was used.
- Focused baseline: the new repository and web-surface contracts both failed
  before implementation. The repository panel returned `unavailable` for a
  `+00:00` effective timestamp, and the root response had no
  `cache-control` header. Command:
  `venv\Scripts\python.exe -m pytest tests\test_control_console_repository.py::test_repository_operational_panels_accept_console_utc_offset tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q`
  Result: 2 failed, 1 warning. Static marker check passed for the character
  operational posture, user causal relationship, and both role-labelled style
  cards; `git diff --check` passed.
- Production subagent: DeepSeek worker `Dewey` (native
  `deepseek_v4_flash_0731`) changed only `src/control_console/repository.py`
  and `src/control_console/app.py`. It rewrites only a trailing accepted UTC
  `+00:00` effective timestamp to terminal `Z` at the two operational V2
  projection owners, and sets `Cache-Control: no-cache, must-revalidate` on
  the root response plus the `/static/*` mount. Its bounded checks confirmed
  valid `Z`, normalized `+00:00`, fail-closed malformed timestamps, API cache
  isolation, static markers, and zero `git diff --check` errors.
- Focused verification: the repository offset contract and static web-surface
  contract passed: 2 passed, 1 pre-existing Starlette/httpx deprecation
  warning.
- Regression verification: `venv\Scripts\python.exe -m pytest
  tests\test_control_console_repository.py tests\test_control_console_web_surface.py
  tests\test_control_console_cognition_graph.py
  tests\test_control_console_redaction.py -q` passed 40 tests with 1
  pre-existing Starlette/httpx deprecation warning. The standalone repository
  suite passed 19 tests and the standalone web-surface suite passed 12 tests.
- Static checks: the Character operational posture, User causal relationship,
  and both User/Group `relevance · cognition · surface` markers were present;
  `git diff --check` passed. The in-app browser remained unavailable, so the
  authenticated HTTP fallback was used.
- Live smoke: after restarting the verified workspace console process on
  `127.0.0.1:8764`, root and `/static/console.js` returned 200 with
  `must-revalidate, no-cache` (equivalent directive order), while the API
  response had no cache header. Character operational posture was `available`
  with 1 item; User relationship operational was `available`; User style was
  `available` with 3 role rows; Group style was `available` with 6 rows. Both
  style surfaces exposed `cognition`, `relevance`, and `surface`, and all
  source rows remained truthfully `missing`. Brain and NapCat were restored to
  `running` through the console lifecycle API (versions 124 and 182).
- Independent code review: user-directed direct parent review; no review
  subagent used; no findings.
- Final disposition: completed. No database migration or style-data mutation
  was required. Existing browser tabs should reload once after the console
  restart; Brain and NapCat are running.
