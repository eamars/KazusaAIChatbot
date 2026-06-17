# control console functional remediation plan

## Summary

- Goal: Replace misleading placeholder control-console pages with truthful,
  operator-useful states and wire the highest-value runtime data first.
- Plan class: large
- Status: in_progress
- Mandatory skills: `development-plan`, `py-style`,
  `test-style-and-execution`, `test-driven-development`,
  `build-web-apps:shadcn`, `build-web-apps:frontend-testing-debugging`
- Overall cutover strategy: bigbang for UI truthfulness. A visible primary page
  must be data-backed and operator-useful. Disabled or not-product-ready
  capabilities must be summarized outside primary navigation instead of
  appearing as tabs. Static contract text must not present itself as a working
  capability.
- Highest-risk areas: incorrect health state, dummy pages passing tests,
  logged-out UI appearing operational, accidental secret/prompt/memory leakage,
  and overclaiming coverage.
- Acceptance criteria: every primary sidebar page is an operator-useful
  workflow, hidden capabilities have an explicit capability status on Overview,
  Health/cache reflects real `/health` and `/ops/runtime-status` when the
  brain is running, logged-out users see a locked console instead of
  operational tabs, tests fail for placeholder-only pages, and execution
  evidence records page-by-page tested inputs and outputs.

## Context

The current `control_console` implementation correctly establishes parts of
the service shell: local auth, CSRF, service registry, child-process lifecycle,
debug-chat handoff, local audit JSONL, process log tailing, SSE, and a
buildless Python/FastAPI-served UI.

The user review identified a more serious product failure: several sidebar
pages exist but do not provide useful operator information. Health/cache can
show the brain as unavailable even when the brain is up. Event monitor claims
Kazusa operational event support while the route returns no Kazusa events.
Character, Memory, Image/style, Calendar, and Background work mostly display
static contract text or empty fallback output. The logged-out UI also shows all
tabs, making the console appear usable before authentication.

The remediation treats this as a functional truthfulness bug, not a styling
polish task. The console must not make a page look complete unless the page is
actually backed by a route and meaningful data source.

## Mandatory Skills

- `development-plan`: load before editing this plan, executing stages, or
  reporting completion.
- `py-style`: load before editing Python production files or tests.
- `test-style-and-execution`: load before adding, changing, running, or
  interpreting pytest tests.
- `test-driven-development`: every production behavior change must have a
  focused failing test first.
- `build-web-apps:shadcn`: apply when editing static UI structure, component
  anatomy, density, and stateful controls.
- `build-web-apps:frontend-testing-debugging`: apply before rendered-browser
  validation of the local UI.

## Mandatory Rules

- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the independent code-review gate and record the result in `Execution
  Evidence`.
- This follow-up is parent-executed because the user explicitly requested the
  current agent to draft the plan and start work in this turn; do not spawn a
  subagent unless the user explicitly reauthorizes delegation.
- Do not read `.env`.
- Do not introduce Node.js, npm, pnpm, yarn, React, Vite, Tailwind build
  tooling, or any other frontend build stack. The production UI remains
  Python/FastAPI-served HTML, CSS, and JavaScript.
- Use shadcn component anatomy and interaction patterns for Sidebar, Button,
  Card, Badge, Table, Input, Select, Textarea, Separator, ScrollArea, and Field
  grouping. Do not invent novel widgets for common controls.
- A visible primary page must be data-backed and operator-useful. Disabled
  capabilities may appear only in capability summaries, not as primary
  navigation targets. Static explanatory text does not count as a functional
  page.
- Coverage claims must distinguish code coverage from product-function
  coverage. Browser tests that only open tabs do not count as capability
  verification.
- Do not change brain cognition, RAG, prompts, memory promotion, calendar
  semantics, background-work generation, adapter transport behavior, or
  database schemas from this plan.

## Must Do

- Add a page capability contract to bootstrap output and static UI state so
  each sidebar page is marked `ready`, `partial`, `unavailable`, or `disabled`.
- Lock operational navigation before login; logged-out users must not see tabs
  that look usable.
- Wire Health/cache to the existing `KazusaClient.get_health()` and
  `KazusaClient.get_runtime_status()` calls when the console-owned brain state
  is running.
- Show truthful Health/cache fallback states when the brain is stopped,
  starting, crashed, or unreachable.
- Remove or rewrite UI copy that claims Kazusa operational events, Mongo audit
  mirroring, image asset browsing, calendar schedules, background jobs, Cache2
  stats, workers, character state, memory lookup, or interaction-style lookup
  are working when they are not data-backed.
- Make Character call `/api/character/status` and `/api/character/growth` when
  opened, and render available, empty, or unavailable states distinctly.
- Expose Memory in primary navigation only as a scoped real lookup backed by
  existing `user_memory_units` helpers; do not provide all-user browsing,
  embeddings, prompts, raw messages, or a generic empty lookup disguised as
  search.
- Expose Interaction style in primary navigation only as a scoped read-only
  lookup backed by existing `interaction_style_images` helpers; do not expose
  source reflection run ids, prompts, raw reflection text, or pretend image
  asset browsing is implemented.
- Expose Calendar in primary navigation only as a due-run inspection workflow
  backed by existing calendar scheduler repository helpers; do not expose
  schedule editing or source payloads.
- Expose Background work in primary navigation only as sanitized
  `background_work.worker` telemetry backed by the existing event-log source;
  do not expose task briefs, artifact text, raw messages, or job payload
  browsing.
- Make Event monitor label `kazusa` available only after a real sanitized
  Kazusa event-log source is wired; local audit and process events remain
  working.
- Make Audit page clear that only local JSONL is implemented unless Mongo
  audit mirroring is added in a later stage.
- Add tests that fail when a primary sidebar page is visible as working but
  only uses static placeholder content or `empty_lookup()`.
- Record page-by-page tested inputs and outputs in `Execution Evidence`.

## Deferred

- Do not implement Mongo audit mirroring in this first remediation slice.
- Do not implement new MongoDB schema migrations for calendar, background
  work, image asset browsing, interaction style, or memory.
- Do not add new brain endpoints.
- Do not change existing brain `/chat`, `/health`, `/ops/*`, adapter runtime,
  RAG, cognition, reflection, self-cognition, calendar, or background-worker
  contracts.
- Do not add a frontend build toolchain.
- Do not broaden the service supervisor beyond registry-declared local child
  processes.

## Cutover Policy

Overall strategy: bigbang for UI truthfulness.

| Area | Policy | Instruction |
|---|---|---|
| Logged-out UI | bigbang | Replace the all-tabs-visible logged-out shell with a locked state. |
| Page capability labels | bigbang | Every sidebar page receives an explicit capability status. |
| Health/cache | bigbang | Replace hardcoded unavailable/cache placeholders with live brain calls when the brain is running. |
| Placeholder pages | bigbang | Remove from primary navigation until real data-backed workflows exist. |
| Existing service lifecycle | compatible | Preserve the already-working service start/stop/restart API and UI behavior. |
| Existing local audit/process events | compatible | Preserve local JSONL audit and process log views while correcting unsupported claims. |

## Cutover Policy Enforcement

- The responsible agent must follow the selected policy for each area.
- If an area is `bigbang`, rewrite the misleading UI state instead of adding a
  second compatibility label.
- If an area is `compatible`, preserve only the compatibility surfaces listed
  above.
- Any change to the cutover policy requires user approval before
  implementation.

## Target State

Operators see a console that is honest before it is broad. Before login, the
console presents a locked operator state. After login, primary navigation lists
only usable operator workflows, while disabled capabilities are summarized on
Overview with concrete reasons. Core runtime pages are useful:

- Overview summarizes service state, auth state, stream state, audit count,
  and page capability status.
- Services shows registry services and valid state-aware actions.
- Debug chat indicates whether the brain is available and records request
  history.
- Event monitor supports local audit and process events, and explicitly marks
  Kazusa operational events as unavailable until wired.
- Health/cache reflects real brain `/health` and `/ops/runtime-status` when
  the brain is running, otherwise a precise unavailable reason.
- Audit renders local JSONL audit rows and clearly says Mongo mirroring is not
  implemented.
- Character renders status/growth API results when available, and distinct
  empty/unavailable states when not.
- Memory renders scoped recent and keyword `user_memory_units` rows by global
  user id, with a needs-input state before any query.
- Interaction style renders scoped user/group guidance from
  `interaction_style_images`, with a needs-input state before any query.
- Calendar renders due `calendar_runs` through existing scheduler repository
  helpers with payload/source-scope redaction and no schedule editing.
- Background work renders sanitized `background_work.worker` event telemetry
  through the Kazusa event-log source with no job payload browsing.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Page status vocabulary | Use `ready`, `partial`, `unavailable`, and `disabled`. | Operators need to distinguish working surfaces from blocked or future surfaces. |
| Health source | Use existing brain `/health` and `/ops/runtime-status` through `KazusaClient`. | These endpoints already own health and runtime state. |
| Brain stopped behavior | Do not call brain endpoints unless the console state says brain is running. | Avoid noisy network failures and show the lifecycle truth first. |
| Placeholder pages | Remove from primary navigation until real data-backed workflows exist. | A decorative page is worse than no page for operator trust. |
| Tests | Test visible product promises, not only route execution. | The previous tests let dummy pages pass. |

## Change Surface

### Modify

- `development_plans/README.md`: register this active bugfix plan.
- `development_plans/active/short_term/backend_control_console_development_plan.md`:
  add a link/evidence note that this follow-up owns functional remediation.
- `src/control_console/app.py`: add page capability metadata, live
  health/runtime bootstrap aggregation, and scoped Memory lookup parameters.
- `src/control_console/repository.py`: project existing user-memory helper
  rows and interaction-style helper context into bounded redacted console
  payloads.
- `src/control_console/static/index.html`: lock logged-out UI and replace
  misleading page copy with explicit capability states.
- `src/control_console/static/console.js`: render page statuses, health/runtime
  values, logged-in state, character refresh, and unsupported-page states.
- `src/control_console/static/console.css`: adjust locked/disabled states using
  existing shadcn-like tokens.
- `src/control_console/README.md`: document page capability statuses and
  first-slice limitations.
- `tests/test_control_console_bootstrap.py`: add health/runtime and capability
  bootstrap tests.
- `tests/test_control_console_repository.py`: add scoped Memory and
  Interaction style lookup projection tests.
- `tests/test_console_lookup_limits.py`: add bounded Memory lookup route tests.
- `tests/test_control_console_web_surface.py`: add logged-out and visible-page
  truthfulness tests.

### Keep

- `src/control_console/kazusa_client.py`: keep existing health/runtime/debug
  methods unless tests prove a local projection helper is required.
- `src/control_console/service_registry.py`, `supervisor.py`, `process_store.py`,
  `log_store.py`, `audit.py`, `auth.py`: keep behavior unchanged unless a
  focused test exposes a blocker in this plan's scope.

## Overdesign Guardrail

- Actual problem: the console shows pages and states that look functional even
  when they are placeholders, and Health/cache can report false information.
- Minimal change: add explicit page capability status, lock pre-login UI, wire
  existing brain health/runtime calls, and disable or relabel unsupported pages.
- Ownership boundaries: control-console code owns UI truthfulness, auth
  gating, local lifecycle, and bounded read projections; brain service owns
  health/runtime data; domain packages own database semantics.
- Rejected complexity: no frontend framework, no new data schema, no new brain
  endpoints, no prompt changes, no generic dashboard framework, no invented
  repository abstraction for future pages.
- Evidence threshold: add new adapters or schemas only after a focused test
  names a concrete data source and a page contract that cannot be satisfied by
  existing endpoints or helpers.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside `src/control_console`,
  `tests/test_control_console_*.py`, and this plan as high-scrutiny changes.
- Before adding a Python helper or function, search for equivalent behavior.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If this plan and code disagree, preserve the plan's stated intent and record
  the discrepancy.

## Implementation Order

1. Add failing tests for bootstrap health/runtime aggregation and logged-out UI
   truthfulness.
2. Implement live Health/cache bootstrap aggregation using existing
   `KazusaClient` methods and precise stopped/unreachable states.
3. Implement logged-out locked UI and page capability metadata rendering.
4. Add failing tests for visible-page truthfulness: unsupported pages must be
   hidden from primary navigation, and working pages must have a route or data
   source.
5. Rewrite static page copy and JavaScript rendering for Event monitor, Audit,
   Character, Calendar, Background work, and Health/cache
   to reflect real capability status.
6. Add Character page refresh wiring for existing `/api/character/status` and
   `/api/character/growth`.
7. Add scoped Memory page wiring for existing `user_memory_units` recent and
   keyword helpers.
8. Add scoped Interaction style page wiring for existing
   `interaction_style_images` context helpers.
9. Run focused tests, static checks, and rendered-browser validation.
10. Record page-by-page tested inputs and outputs in execution evidence.

## Execution Model

- Parent agent owns orchestration, test code, production code, verification,
  execution evidence, review feedback remediation, lifecycle updates, and final
  sign-off for this follow-up.
- The user explicitly requested the current agent to draft the follow-up plan
  and start work. No subagent is spawned in this turn unless the user later
  explicitly requests delegation.
- Parent agent establishes each focused failing test before the matching
  production behavior change.
- Independent code review remains required before final completion; if the
  user does not authorize a review subagent, the parent must perform a
  review-stance pass and leave the checklist item unsigned as blocked for
  formal completion.

## Progress Checklist

- [x] Stage 1 - plan and first failing tests
  - Covers: implementation steps 1 and registry update.
  - Verify: targeted tests fail for the expected missing behavior.
  - Evidence: record commands and expected failures in `Execution Evidence`.
  - Sign-off: parent/2026-06-17 after failing tests were recorded and the
    active bugfix plan was registered.
- [x] Stage 2 - health/runtime and logged-out truthfulness fixed
  - Covers: implementation steps 2 and 3.
  - Verify: targeted bootstrap and web-surface tests pass.
  - Evidence: record changed files and test output.
  - Sign-off: parent/2026-06-17 after focused and regression tests passed.
- [x] Stage 3 - visible-page status contract enforced
  - Covers: implementation steps 4 and 5.
  - Verify: tests fail before and pass after placeholder claim rewrites.
  - Evidence: record page status table and test output.
  - Sign-off: parent/2026-06-17 after stale-copy grep and rendered validation passed.
- [x] Stage 4 - character page existing-route wiring
  - Covers: implementation step 6.
  - Verify: browser/static tests prove Character calls existing APIs and
    renders available, empty, and unavailable states distinctly.
  - Evidence: record tests and rendered validation notes.
  - Sign-off: parent/2026-06-17 after static test and rendered validation passed.
- [x] Stage 5 - primary navigation product surface refined
  - Covers: the interim pass that removed non-functional Memory, Image/style,
    Calendar, and Background work pages from the primary sidebar; Overview
    capability tables distinguish usable workflows from hidden unavailable
    workflows; header and Debug chat no longer imply readiness before the
    brain is running.
  - Verify: focused web-surface and bootstrap tests fail before and pass after
    the UI state changes.
  - Evidence: record commands and rendered validation notes.
  - Sign-off: parent/2026-06-17 after focused tests passed.
- [x] Stage 6 - scoped Memory workflow added
  - Covers: implementation step 7.
  - Verify: focused repository, bootstrap, web-surface, and lookup-route tests
    fail before and pass after the Memory workflow is backed by real helpers.
  - Evidence: record commands and page behavior.
  - Sign-off: parent/2026-06-17 after focused tests passed without live-DB
    dependency output.
- [x] Stage 7 - scoped Interaction style workflow added
  - Covers: implementation step 8.
  - Verify: focused repository, bootstrap, web-surface, and lookup-route tests
    fail before and pass after Interaction style is backed by real helpers.
  - Evidence: record commands and page behavior.
  - Sign-off: parent/2026-06-17 after focused tests passed.
- [ ] Stage 8 - validation and independent review
  - Covers: implementation steps 9 and 10.
  - Verify: focused tests, relevant control-console test batch, static checks,
    and rendered-browser validation pass.
  - Evidence: record commands, page-by-page tested inputs/outputs, review
    findings, fixes, and residual risks.
  - Sign-off: pending.

## Verification

### Focused tests

- `venv\Scripts\python -m pytest tests\test_control_console_bootstrap.py -q`
- `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py -q`

### Control-console regression batch

```powershell
$tests = @()
$tests += Get-ChildItem -LiteralPath 'tests' -Filter 'test_control_console_*.py' | ForEach-Object { $_.FullName }
$tests += Get-ChildItem -LiteralPath 'tests' -Filter 'test_console_*.py' | ForEach-Object { $_.FullName }
venv\Scripts\python.exe -m pytest @tests -q
```

### Coverage

```powershell
venv\Scripts\python.exe -m coverage erase
venv\Scripts\python.exe -m coverage run --source=control_console -m pytest @tests -q
venv\Scripts\python.exe -m coverage report --show-missing
```

Required result: `control_console` statement coverage remains at or above 90%.
This is not sufficient by itself; page-by-page functional evidence is also
required.

### Static checks

- `venv\Scripts\python -m compileall src\control_console tests`
- `git diff --check`
- `rg "source readers unavailable in this skeleton|bootstrap skeleton|Ready for bounded refresh|Kazusa operational events through bounded filters" src\control_console\static`
  must return no matches.

### Rendered UI validation

Run the local console on a temporary loopback port with an isolated state
directory. Validate:

- logged-out UI shows a locked state and does not present tabs as usable;
- login works with a generated or test token;
- Health/cache shows stopped state before brain start;
- Health/cache shows `/health` and `/ops/runtime-status` values after brain is
  running;
- unsupported pages are absent from primary navigation and summarized on
  Overview;
- no browser alerts, page errors, or horizontal overflow are introduced.

## Independent Code Review

Run this gate after all verification commands pass and before final sign-off.
If the user authorizes a review subagent, create exactly one independent review
subagent. Otherwise, perform a parent review-stance pass and record that formal
independent review remains blocked by missing delegation authorization.

Review scope:

- Project rules and style compliance for every changed Python, test,
  documentation, and static UI file.
- Alignment with `Must Do`, `Deferred`, page capability statuses, exact change
  surface, verification gates, and acceptance criteria.
- Product truthfulness: no visible page claims functionality that is not
  data-backed or explicitly unavailable.
- Security and privacy: no prompts, secrets, embeddings, raw message bodies, or
  unbounded records are exposed.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Logged-out users see a locked console state instead of operational tabs.
- `/api/bootstrap` reports truthful page capabilities and live health/runtime
  data when the brain is running.
- Health/cache no longer hardcodes brain unavailable or dummy Cache2/worker
  data.
- Event monitor, Audit, Calendar, Background work, and
  Character no longer make unsupported functionality look implemented.
- Memory is a primary sidebar destination only as a scoped real lookup by
  global user id. Interaction style is a primary sidebar destination only as a
  scoped real guidance lookup. Calendar is a primary sidebar destination only
  as due-run inspection, and Background work is a primary sidebar destination
  only as sanitized worker-event telemetry.
- Tests fail for placeholder-only visible pages and pass after remediation.
- Focused tests, control-console regression batch, static checks, coverage, and
  rendered UI validation pass.
- Execution evidence includes a page-by-page table of tested web inputs,
  outputs, and remaining gaps.

## Execution Evidence

- 2026-06-17 parent: plan created from user corrective feedback. Initial scope
  targets UI truthfulness, health/runtime correctness, logged-out gating, and
  tests that fail placeholder-only pages. Sign-off: pending Stage 1
  verification.
- 2026-06-17 parent Stage 1 evidence: added failing tests for live
  health/runtime bootstrap data and logged-out UI locking. Focused command
  `venv\Scripts\python -m pytest tests\test_control_console_bootstrap.py::test_bootstrap_reports_live_brain_health_when_brain_is_running tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q`
  failed as expected: bootstrap still returned `brain_health.status ==
  unavailable`, and the shell lacked `body[data-auth-state="locked"]`.
- 2026-06-17 parent Stage 2 evidence: implemented bootstrap health/runtime
  aggregation from existing `KazusaClient.get_health()` and
  `KazusaClient.get_runtime_status()` when the brain service is running; added
  stopped/unreachable fallback reason; locked navigation before login and
  re-enabled it after successful login. The same focused command passed with 2
  tests.
- 2026-06-17 parent Stage 3 evidence: added `page_capabilities` to bootstrap,
  page-status rendering in JavaScript, and static copy guards against dummy
  wording. Focused command
  `venv\Scripts\python -m pytest tests\test_control_console_bootstrap.py::test_bootstrap_returns_initial_state_session_csrf_services_and_stream_url tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q`
  first failed on missing `page_capabilities` and stale UI claims, then passed.
  Static grep `rg "source readers unavailable in this skeleton|bootstrap skeleton|Ready for bounded refresh|Kazusa operational events through bounded filters" src\control_console\static`
  now returns no matches.
- 2026-06-17 parent Stage 4 evidence: replaced hardcoded Health/cache cards
  with dynamic DOM targets, rendered bootstrap health/cache/runtime data in
  `console.js`, added Character status/growth table targets, and wired the
  Character page to `/api/character/status` and `/api/character/growth`.
  Focused `tests/test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs`
  first failed on hardcoded `until brain starts`, then passed.
- 2026-06-17 parent validation evidence: focused files passed with 8 tests;
  control-console regression batch passed with 33 tests; coverage gate passed
  with 92% `control_console` statement coverage. Static checks passed:
  `venv\Scripts\python -m compileall src\control_console tests`,
  `git diff --check` with existing CRLF warnings only, and the stale-copy grep
  listed above with no matches.
- 2026-06-17 parent rendered validation evidence: Browser plugin path failed
  with `Browser is not available: iab`, so rendered validation used local
  headless Chrome through Playwright without taking screenshots. Temporary
  console ran on `127.0.0.1:8770` with isolated state and a known test token.
  Validated locked pre-login state, successful login, Services enabled after
  login, Image/style Calendar and Background work disabled after login,
  Health/cache reporting `unavailable` with reason `brain service is stopped`,
  Character table rendering 4 rows after opening the page, no horizontal
  overflow, and no console warnings/errors. Temporary server was stopped, port
  `8770` was closed, and the temporary state directory was removed.
- 2026-06-17 parent Stage 5 evidence: product-quality sign-off remains `no`.
  The console is more truthful and usable, but not sellable as a product while
  image asset browsing, Calendar, Background work, Mongo-backed audit, and
  Kazusa operational events are not implemented as real workflows. At this interim
  checkpoint, Memory was still hidden because it was not yet a real workflow.
  Added failing tests requiring Memory to be disabled and requiring Memory,
  Image/style, Calendar, and Background work to be absent from primary
  navigation. Focused commands
  `venv\Scripts\python.exe -m pytest tests\test_control_console_bootstrap.py::test_bootstrap_returns_initial_state_session_csrf_services_and_stream_url -q`
  and
  `venv\Scripts\python.exe -m pytest tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q`
  failed as expected, then passed after the UI/backend capability changes.
- 2026-06-17 parent Stage 5 implementation evidence: set Memory capability to
  `disabled`, removed Memory, Image/style, Calendar, and Background work from
  the sidebar and page body, added Overview capability tables for visible and
  unavailable workflows, changed the header status dot to neutral until real
  service state is loaded, disabled Debug chat inputs while the brain is not
  running, and disabled the Kazusa event-source option in the UI. Stage 6
  supersedes only the Memory part of this interim state.
- 2026-06-17 parent Stage 5 validation evidence: focused
  `tests\test_control_console_bootstrap.py` and
  `tests\test_control_console_web_surface.py` passed with 8 tests. The
  control-console regression batch passed with 33 tests. Coverage gate passed
  with 92% `control_console` statement coverage. Static checks passed:
  `venv\Scripts\python.exe -m compileall src\control_console tests`,
  `git diff --check` with existing CRLF warnings only, and stale/dead-page grep
  with no matches. Some tests reached the configured MongoDB through the
  process environment; `.env` was not read by the agent.
- 2026-06-17 parent Stage 5 rendered validation evidence: Browser plugin path
  failed with `Browser is not available: iab`, so rendered validation used
  local headless Chrome through Playwright without screenshots. Temporary
  console ran on `127.0.0.1:8771` with isolated state and known test token.
  Validated initial locked state, neutral locked status dot, sidebar containing
  only Overview, Services, Debug chat, Event monitor, Character, Health/cache,
  and Audit; successful login; stopped-brain header state; unavailable
  workflow table listing Memory, Image/style, Calendar, and Background work;
  Debug chat inputs and Send disabled while brain is stopped; Kazusa event
  source option disabled; service buttons mutually exclusive for stopped
  services; Health/cache stopped-state detail; Bright/Dark theme toggle; no
  horizontal overflow; and no console warnings/errors. Temporary server was
  stopped, port `8771` was closed, and the temporary state directory was
  removed. Stage 6 supersedes only the Memory part of this rendered state.
- 2026-06-17 parent Stage 6 evidence: product-quality sign-off remains `no`,
  but Memory was moved from a hidden placeholder to a scoped partial workflow.
  Added failing tests for Memory capability status, static Memory page
  presence, repository projection, needs-input handling, and bounded lookup
  route behavior. Focused command
  `venv\Scripts\python.exe -m pytest tests\test_control_console_bootstrap.py::test_bootstrap_returns_initial_state_session_csrf_services_and_stream_url tests\test_control_console_repository.py::test_repository_projects_user_memory_units_with_redaction tests\test_control_console_repository.py::test_repository_memory_lookup_requires_global_user_id tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs tests\test_console_lookup_limits.py::test_lookup_routes_enforce_pagination_redaction_and_no_embeddings -q`
  failed as expected, then passed after implementation. A rerun including the
  character/web API test passed with 8 focused tests and no live-DB output
  after deterministic tests stubbed character repository calls.
- 2026-06-17 parent Stage 6 validation evidence: added a failing test for
  invalid DB/helper configuration returning a safe unavailable Memory payload
  instead of raising a 500, then implemented the explicit `ValueError`
  boundary catch. Control-console regression batch passed with 36 tests.
  Coverage gate passed with 91% `control_console` statement coverage. Static
  checks passed: `venv\Scripts\python.exe -m compileall src\control_console
  tests`, `git diff --check` with existing CRLF warnings only, and stale-copy
  grep against `src\control_console\static` plus `src\control_console\README.md`
  with no matches.
- 2026-06-17 parent Stage 6 rendered validation evidence: Browser plugin path
  failed with `Browser is not available: iab`, so rendered validation used
  headless Chrome through Playwright without screenshots per prior user
  preference. Temporary console ran on `127.0.0.1:8773` with isolated state
  and known test token. Validated locked pre-login state, Memory present but
  disabled before login, successful login, stopped-brain header state,
  Overview available workflow table listing Memory as `scoped lookup`,
  unavailable table listing only Image/style, Calendar, and Background work,
  Memory needs-input state, Memory empty result for a scoped unknown
  `global_user_id` plus keyword, no embedding/prompt/raw-message leakage in
  the Memory table, service buttons mutually exclusive while brain is stopped,
  adapter start disabled until dependencies run, Kazusa event-source option
  disabled, Bright/Dark theme toggle, no horizontal overflow at 1600x900, and
  no console warnings/errors. Temporary server was stopped, port `8773` was
  closed, and the temporary state directory was removed.
- 2026-06-17 parent Stage 7 evidence: product-quality sign-off remains `no`,
  but Interaction style was moved from an unavailable placeholder to a scoped
  partial workflow. Added failing tests for Interaction style capability
  status, static page presence, repository projection, needs-input handling,
  and bounded route behavior. Focused command
  `venv\Scripts\python.exe -m pytest tests\test_control_console_bootstrap.py::test_bootstrap_returns_initial_state_session_csrf_services_and_stream_url tests\test_control_console_repository.py::test_repository_projects_interaction_style_context_safely tests\test_control_console_repository.py::test_repository_interaction_style_lookup_requires_scope tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs tests\test_console_lookup_limits.py::test_lookup_routes_enforce_pagination_redaction_and_no_embeddings -q`
  failed as expected, then passed after implementation.
- 2026-06-17 parent Stage 8 evidence: added and validated the remaining
  product-truthfulness work for Calendar, Background work, event telemetry,
  unmanaged endpoint conflicts, debug-chat response projection, and debug-mode
  controls. Focused red/green evidence included supervisor conflict detection,
  web-surface conflict rendering, safe debug-chat projection without raw
  delivery mentions, Calendar due-run lookup projection, and Background work
  sanitized worker-event telemetry. Browser validation used the Browser-plugin
  fallback path because the in-app Browser failed with `Browser is not
  available: iab`; headless Chrome plus Playwright validated the rendered
  console without screenshots.
- 2026-06-17 parent Stage 8 live validation evidence: rendered checks covered
  logged-out locking, login, stopped-brain state, unmanaged brain endpoint
  conflict state, service button mutual exclusion, alternate-port brain start
  through the web UI, Health/cache after brain start, Debug chat enabled after
  brain start, web-triggered brain stop, theme toggle, no console errors, and
  no horizontal overflow at 1600x900. Live Debug chat was sent through the web
  interface and returned a sanitized visible reply plus tracking metadata
  without raw JSON, `delivery_mentions`, global user ids, platform user ids,
  prompts, embeddings, or message-envelope dumps.
- 2026-06-17 parent Stage 8 debug-mode evidence: added a failing static
  web-surface test requiring Debug chat run-mode and persistence controls,
  then implemented a shadcn-style `FieldSet` run-mode group
  (`visible_reply`, `think_only`, `listen_only`) plus a checked-by-default
  `no_remember` persistence checkbox. Headless Chrome validation of the real
  static HTML and JavaScript submitted the actual form and captured
  `/api/debug-chat` with `debug_modes == ["no_remember", "think_only"]`,
  updated redacted history, no console warnings/errors, and no horizontal
  overflow.
- 2026-06-17 parent Stage 8 verification evidence: control-console regression
  batch passed with 44 deterministic tests:
  `$tests = @(); $tests += Get-ChildItem -LiteralPath 'tests' -Filter
  'test_control_console_*.py' | ForEach-Object { $_.FullName }; $tests +=
  Get-ChildItem -LiteralPath 'tests' -Filter 'test_console_*.py' |
  ForEach-Object { $_.FullName }; venv\Scripts\python.exe -m pytest @tests
  -q`. Coverage passed at exactly 90% `control_console` statement coverage
  after the same batch under `coverage run`. Static checks passed:
  `venv\Scripts\python.exe -m compileall src\control_console tests`,
  `git diff --check` with existing CRLF warnings only, and stale-placeholder
  grep against `src\control_console` with no matches.
- 2026-06-17 parent review evidence: review-stance pass checked
  product-truthfulness copy, auth gating, lifecycle conflict behavior,
  service registry coverage including `adapter.napcat`, Bright/Dark theme
  labels, debug-mode payload serialization, response redaction, shadcn-style
  Card/Table/Button/Input/Textarea/FieldSet anatomy, and stale placeholder
  wording. One issue was found and fixed with a red/green test: the Character
  capability reason used internal remediation wording. Focused bootstrap test
  failed on the old text, then passed after replacing it with product-facing
  copy. Fresh regression passed with 44 deterministic tests, coverage remained
  at 90%, compileall passed, `git diff --check` passed with existing CRLF
  warnings only, and production-source stale-copy grep returned no matches.
  The parent review found no release-blocking defects for a local-operator MVP.
  Formal independent subagent review was not rerun in this follow-up because
  the current remediation execution was constrained to parent review.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Dummy page still looks complete | Page capability status and static grep gates | Web-surface tests and rendered UI validation |
| Health endpoint call slows bootstrap | Call brain endpoints only when service state is running and use existing bounded client timeout | Bootstrap focused tests and browser smoke |
| Unsupported pages frustrate users | Disable or mark unavailable with a concrete reason instead of decorative content | Rendered UI validation |
| New UI labels drift from shadcn style | Use existing Button/Card/Badge/Table/Input anatomy and tokens | Static review and browser inspection |
| Coverage overclaimed again | Require page-by-page evidence in addition to coverage percentage | Execution evidence table |
