# control console information architecture remediation plan

## Summary

- Goal: Fix every surfaced control-console UI issue from the production review: missing coding-agent routes, stale/static status copy, misplaced character dynamic content, flattened typed records, weak table headers, and populated Users/Groups presentation defects.
- Plan class: medium
- Status: completed
- Mandatory skills: `development-plan`, `control-console-web-development`, `build-web-apps:frontend-testing-debugging`, `build-web-apps:frontend-app-builder`, `py-style`, `test-style-and-execution`
- Overall cutover strategy: bigbang for the static console UI and route catalog.
- Highest-risk areas: shared render helpers used across Character, Users, Groups, Calendar, Background work, Event monitor, Audit, and Live logs.
- Acceptance criteria: deterministic tests pass, Playwright/Chrome screenshots show old-vs-new improvement on production-backed data, and the service route matrix includes `CODING_AGENT_PM_LLM` and `CODING_AGENT_PROGRAMMER_LLM`.

## Context

The production review at `http://localhost:8765/` found the same information-architecture problem across multiple console gadgets: typed records are flattened into continuous key/value rows, dynamic prompt/debug content is visually mixed with metadata, static metric cards pretend to be live information, and several genuinely tabular surfaces lack headers. The production console is served from `C:\workspace\kazusa_cognition_core_prod`; this plan edits the dev workspace `C:\workspace\kazusa_ai_chatbot`.

Confirmed old-state evidence is held outside the repo in:

- `C:\Users\RANBAO~1\AppData\Local\Temp\kazusa_prod_console_audit_i3xvchb6`
- `C:\Users\RANBAO~1\AppData\Local\Temp\kazusa_prod_user_group_audit_umavv_gl`

The control console is buildless static HTML/CSS/JS served by FastAPI. It must keep the existing shadcn-family anatomy and must not introduce React, Vite, Tailwind, a Node build, new runtime polling, or a second visual system.

## Mandatory Skills

- `development-plan`: govern this plan lifecycle, progress evidence, and completion.
- `control-console-web-development`: govern control-console boundaries, static UI rules, Playwright/Chrome validation, and redaction.
- `build-web-apps:frontend-testing-debugging`: govern rendered desktop/mobile QA, console health, screenshots, and interaction proof.
- `build-web-apps:frontend-app-builder`: govern design quality; because this is a targeted fix inside an existing design system, do not use Image Gen and do not create a new design system.
- `py-style`: load before editing Python route catalog or Python tests.
- `test-style-and-execution`: load before adding, changing, or running pytest tests.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before continuing implementation, verification, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan before starting the next stage.
- Before production UI edits, capture and present a rendered proposed UI preview outside the repo. Production files must not be changed before that preview exists.
- Use the existing static console stack: HTML, CSS, JavaScript, FastAPI. Do not add frontend dependencies or a build step.
- Use existing component families: Card, Badge, Table, Input, Select, Field/Form grouping, ScrollArea, detail chips, timeline items, service route tiles, and dialog patterns.
- Keep tables only for truly tabular data with stable columns. Render typed records as cards with title, status badges, metadata chips, and detail grids.
- Prompt View panels must visually separate primary prompt-facing content from metadata. Metadata must not compete with the content body.
- Missing scope must remain explicit `needs_input`; do not infer previous browser selections or hidden user/channel ids.
- Keep `/chat`, cognition, RAG, memory promotion, scheduler semantics, background-work generation, prompts, adapter transport, and database persistence unchanged.
- Keep all real-data screenshots, traces, and temporary scripts outside the repo.
- Do not expose raw prompts, embeddings, secrets, raw messages, raw reflection output, full memory bodies, idempotency keys, source scopes, or unbounded documents.
- Use `venv\Scripts\python` for Python tests and local console runs.
- Use Playwright with system Chrome for rendered validation because the user explicitly said no IAB.
- Do not revert the existing dirty `AGENTS.md` worktree change.

## Must Do

- Add `CODING_AGENT_PM_LLM` and `CODING_AGENT_PROGRAMMER_LLM` to the Brain route catalog, route projection, descriptor-backed field generation, and deterministic tests.
- Reorder Character page content so dynamic Prompt View/current runtime/growth audit content appears before static profile/self-image reference content.
- Replace flattened operational backing row streams with record-card rendering for:
  - Character `Growth Runs Audit`
  - Calendar `Schedule Definitions`
  - Calendar `Due Runs`
  - Background work `Job Queue`
  - Background work `Worker Events`
- Replace Users `User Memory` flattened table rows with one card per memory unit.
- Replace Prompt View table rendering with a prompt-panel layout for Character, Users, Groups, Calendar, and Background work prompt panels.
- Make Users and Groups top summary cards dynamic after lookup instead of static `platform/user/private` and `group/channel/safe` labels.
- Replace the stale Debug `Brain stopped` contract row with capability wording that does not claim current runtime state.
- Add headers to Event monitor, Audit, Overview audit, and Live logs tables while preserving existing row rendering and copy buttons.
- Preserve explicit `needs_input`, `empty`, and `unavailable` reasons for partial panels.
- Update deterministic and Playwright e2e tests to prove the new layouts use cards where typed records exist and headers where tables remain.
- Run old-vs-new rendered comparison against the same production-backed QQ user `673225019` and QQ group `54369546`. Use group `54369546` as the Users channel id with channel type `group`, and use user `673225019` as the Groups participant user id for the post-change populated comparison because the user supplied both IDs for this review scope.

## Deferred

- Do not change repository read projections, redaction policy, data retention, or raw database access.
- Do not add schedule editing, job editing, raw payload browsing, event-log snapshot browsing, or hidden detail drawers.
- Do not redesign the full console shell, navigation, theme, brand, typography system, or color palette.
- Do not add charts, animations, icons, images, or decorative components.
- Do not introduce compatibility shims for old frontend helper names beyond keeping existing selectors stable for tests and browser callers.
- Do not change the control-console auth/session/CSRF model.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Static UI layout | bigbang | Replace the poor table-first layouts directly. Do not keep old flattened typed-record rendering as an alternate mode. |
| Route catalog | bigbang | Add the two missing first-class coding-agent routes directly to `_ROUTES`; no alias route or fallback route. |
| Frontend render helpers | bigbang | Teach shared render helpers to output card/prompt layouts for the new target containers; keep table output only where a table remains. |
| Tests | bigbang | Update existing expectations and add focused checks for the new UI contracts. |
| Data and APIs | compatible | Preserve endpoint names, selectors, payloads, redaction, and bounded limits. |

## Cutover Policy Enforcement

- Follow the selected policy for each area.
- Do not choose a more conservative strategy by default.
- For bigbang areas, rewrite the old rendering rather than preserving an operator-visible fallback.
- For compatible areas, preserve only the endpoint and payload surfaces listed in this plan.
- Any change to this cutover policy requires user approval before implementation.

## Target State

The console remains a dense operational tool, but each surface uses the right structure:

- Services shows all 14 first-class chat routes, including coding-agent PM and programmer.
- Character opens with Prompt View/current runtime panels and audit cards before profile reference panels.
- Prompt View panels show the prompt-facing content as the main body, with metadata grouped below as compact detail chips/grid.
- Operational Backing panels show one record card per run, schedule, due run, job, or worker event.
- Users shows profile and prompt views, then memory cards that make each memory unit's type, status, update time, fact, relationship signal, and appraisal separable.
- Groups shows group carry-over content as prompt content rather than as a table row, and participant progress becomes populated when a participant id is supplied.
- Event monitor, Audit, Overview audit, and Live logs retain tables because they are tabular, and each table has explicit headers.
- Static summary cards either become dynamic after data load or describe stable capability boundaries without pretending to be live status.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Visual system | Use the existing console CSS primitives and add small variants only for record lists and prompt panels. | The console already has card/detail/timeline/chip primitives and the user required no reinvention. |
| Typed record display | Use cards, not continuous key/value tables. | Runs, jobs, schedules, and memory units are typed records with relationships inside each item. |
| Prompt display | Content first, metadata second. | Prompt View's purpose is inspecting the actual prompt-facing window. |
| Real tables | Keep tables for event/audit/log rows and add headers. | Those rows have stable columns and benefit from scanning. |
| Summary cards | Update visible values from loaded panel status/counts. | Static top metrics were identified as stale or low-value. |
| Route catalog placement | Add coding-agent PM and programmer routes to the route tuple as first-class required routes. | README/HOWTO state these routes are required first-class code-reading routes. |

## Change Surface

### Modify

- `src/control_console/brain_model_routes.py`: add two missing route descriptors.
- `src/control_console/static/index.html`: reorder Character panels, convert prompt/operational/memory containers, update stale Debug contract text, and add table headers.
- `src/control_console/static/console.css`: add record-card, operational-list, prompt-panel, and responsive table-header styles using existing tokens.
- `src/control_console/static/console.js`: update shared render helpers and page refresh summary updates.
- `tests/test_control_console_brain_model_routes.py`: update route catalog and descriptor field-count assertions.
- `tests/test_control_console_cognition_debug_visibility.py`: update static surface assertions for new prompt/operational containers.
- `tests/control_console_e2e/test_page_navigation_e2e.py`: update e2e assertions to require card layouts for typed records and dynamic summaries.
- `tests/control_console_e2e/test_visual_product_acceptance_e2e.py`: preserve visual overflow and log copy invariants with headers.

### Create

- Temporary rendered UI preview and before/after screenshots under `%TEMP%`; do not commit them.

### Keep

- Existing endpoint URLs, selector ids, payload contracts, auth, CSRF, service lifecycle behavior, SSE streams, and database read helpers.

## Overdesign Guardrail

- Actual problem: The console presents operational typed records and prompt windows with generic key/value tables, causing relationships, recency, and data ownership to be hard to read.
- Minimal change: Reuse existing static UI primitives and selectors to render prompt panels, record-card lists, dynamic summaries, and table headers.
- Ownership boundaries: Backend repository helpers continue to own bounded redacted data; frontend render helpers own presentation; route catalog owns editable LLM route visibility.
- Rejected complexity: No frontend framework, no backend schema changes, no new endpoints, no drawers, no persisted UI state, no icon system, no broad visual redesign, no data exports.
- Evidence threshold: Only add heavier components or new APIs later if rendered cards cannot represent a specific approved workflow or if a bounded detail interaction is explicitly requested.

## Agent Autonomy Boundaries

- The responsible agent may choose local HTML/CSS/JS mechanics only when they preserve this plan's contracts.
- The responsible agent must not introduce new architecture, alternate route migration strategies, compatibility layers, fallback paths, or extra features.
- Changes outside the listed control-console files and tests require stopping and updating this plan before editing.
- The responsible agent must search for existing frontend helpers before adding new helpers and must reuse existing behavior where it fits.
- The responsible agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Implementation Order

1. Capture proposed rendered UI preview outside the repo.
2. Establish focused tests:
   - Update route catalog test to expect 14 routes and 42 descriptor fields.
   - Add/update static and e2e UI tests to expect prompt-panel containers, operational record cards, memory cards, dynamic summaries, and table headers.
   - Run the focused tests before implementation and record the expected failures or baseline.
3. Implement Python route catalog additions.
4. Implement static HTML container/order/header changes.
5. Implement CSS record-card and prompt-panel variants.
6. Implement JavaScript rendering and summary updates.
7. Run focused deterministic tests.
8. Run control-console e2e tests for page navigation and visual acceptance.
9. Run Playwright/Chrome rendered validation against a dev server using production-backed data on a non-production port.
10. Compare old screenshots to new screenshots and record improvements or regressions.
11. Run independent code review in local review posture unless the user explicitly authorizes native subagents.
12. Fix review findings inside this change surface and rerun affected checks.

## Execution Model

- This execution is parent-led in the current chat.
- Native subagent capability exists, but the active subagent tool policy forbids spawning unless the user explicitly asks for delegation or subagents. Because the user explicitly instructed this agent to produce the plan and address the issues, implementation proceeds locally without spawning subagents.
- The parent agent owns test changes, production changes, verification, rendered comparison, review posture, evidence updates, and final sign-off.
- The local independent code-review gate must be performed after planned verification passes and before completion. If the user explicitly authorizes subagents before that gate, use one review subagent.

## Progress Checklist

- [x] Stage 1 - rendered proposed UI preview captured
  - Covers: implementation order step 1.
  - Verify: Playwright/Chrome screenshot outside repo shows proposed prompt panels, record cards, memory cards, and table headers.
  - Evidence: record screenshot path in `Execution Evidence`.
  - Sign-off: Codex / 2026-07-05 after preview screenshots were captured and displayed.
- [x] Stage 2 - focused tests establish the contract
  - Covers: implementation order step 2.
  - Verify: focused test commands run and fail or show current baseline for missing route/card/header contracts.
  - Evidence: record commands and outcomes.
  - Sign-off: Codex / 2026-07-05 after expected failures were recorded.
- [x] Stage 3 - route catalog and static layout implemented
  - Covers: implementation order steps 3-5.
  - Verify: route catalog test passes; static surface test passes.
  - Evidence: record changed files and test output.
  - Sign-off: Codex / 2026-07-05 after route/static tests passed.
- [x] Stage 4 - render helpers and dynamic summaries implemented
  - Covers: implementation order step 6.
  - Verify: page navigation e2e passes for affected pages.
  - Evidence: record e2e output and screenshots when available.
  - Sign-off: Codex / 2026-07-05 after page navigation and visual e2e tests passed.
- [x] Stage 5 - rendered production-data comparison complete
  - Covers: implementation order steps 7-10.
  - Verify: Playwright/Chrome screenshots on desktop and mobile show no console/page errors, no horizontal overflow, populated QQ user/group flows, and old-vs-new improvement.
  - Evidence: record screenshot paths and comparison notes.
  - Sign-off: Codex / 2026-07-05 after final production-backed Chrome screenshots and diagnostics were captured.
- [x] Stage 6 - independent code review complete
  - Covers: implementation order steps 11-12.
  - Verify: local review posture or user-authorized review subagent inspects plan alignment, diff, and evidence.
  - Evidence: record findings, fixes, reruns, residual risks, and approval status.
  - Sign-off: Codex / 2026-07-05 after local review found no blocking findings.

## Verification

### Static Greps

- `rg "CODING_AGENT_PM_LLM|CODING_AGENT_PROGRAMMER_LLM" src/control_console tests/test_control_console_brain_model_routes.py README.md docs/HOWTO.md`
  - Expected: route names appear in docs, route catalog, and route tests.
- `rg "Brain stopped" src/control_console/static`
  - Expected: zero matches; `rg` exit code 1 is acceptable.

### Deterministic Tests

- `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py -q`
- `venv\Scripts\python -m pytest tests\test_control_console_cognition_debug_visibility.py -q`

### Rendered E2E Tests

- `venv\Scripts\python -m pytest tests\control_console_e2e\test_page_navigation_e2e.py -q`
- `venv\Scripts\python -m pytest tests\control_console_e2e\test_visual_product_acceptance_e2e.py -q`

### Playwright/Chrome Manual QA

- Start the dev console on a non-production port.
- Open with Playwright using system Chrome.
- Log in with the configured operator token.
- Validate these flows:
  - Services route matrix includes both coding-agent routes.
  - Character dynamic panels appear above static profile/self-image panels.
  - Users lookup uses `platform=qq`, `platform_user_id=673225019`, `platform_channel_id=54369546`, `channel_type=group`.
  - Groups lookup uses `platform=qq`, `group_id=54369546`, `participant_platform_user_id=673225019`.
  - Calendar and Background work show operational records as cards.
  - Event monitor, Audit, Overview audit, and Live logs show table headers.
- Capture desktop and mobile screenshots.
- Record console messages, page errors, HTTP failures, and horizontal overflow.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. The review must inspect:

- Project rules and style compliance for changed Python, static UI, tests, and plan evidence.
- Alignment with `Must Do`, `Deferred`, `Change Surface`, implementation order, verification gates, and acceptance criteria.
- Code quality and design weaknesses, including route catalog ownership, redaction, hidden fallback paths, brittle selectors, and avoidable blast radius.
- Regression and handoff quality, including old-vs-new screenshots, production-backed data states, desktop/mobile coverage, and residual risk.

Fix concrete findings directly only when they are inside the approved change surface. If a finding requires a different public contract or outside-boundary edit, stop and update this plan or request approval before changing code.

## Acceptance Criteria

This plan is complete when:

- All listed `Must Do` items are implemented.
- The two missing coding-agent routes render in Services and pass route catalog tests.
- Typed operational records and user memory units render as one card per item.
- Prompt View panels render prompt-facing content above metadata.
- Character dynamic content appears above static profile/self-image content.
- Users and Groups summary cards update from lookup results.
- Event monitor, Audit, Overview audit, and Live logs have headers.
- `Brain stopped` is absent from static console assets.
- Focused deterministic tests and e2e tests listed in `Verification` pass, or any blocked check is explicitly recorded with the blocker.
- Playwright/Chrome old-vs-new comparison shows improvement without new console errors, page errors, horizontal overflow, or mobile overlap.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Shared render helper change breaks existing table panels | Keep table output for `tbody` targets and card output for `div` targets. | Static tests and page navigation e2e. |
| Record cards expose raw sensitive fields | Render only already-redacted API payload fields and preserve existing repository projections. | Existing redaction tests plus rendered text checks. |
| Mobile card lists become too long | Use compact card anatomy and bounded prompt content scroll areas. | Mobile Playwright screenshots and overflow diagnostics. |
| Route descriptor count changes break service config assumptions | Update route tests and descriptor field-count expectations. | Brain route test. |

## Execution Evidence

- 2026-07-05: Plan created in `development_plans/active/bugfix/control_console_information_architecture_remediation_plan.md`.
- 2026-07-05: Stage 1 preview captured with Playwright/Chrome outside the repo:
  - `C:\Users\RANBAO~1\AppData\Local\Temp\kazusa_console_proposed_ui_zrhu3ltt\proposed_primary.png`
  - `C:\Users\RANBAO~1\AppData\Local\Temp\kazusa_console_proposed_ui_zrhu3ltt\proposed_primary_mobile.png`
- 2026-07-05: Stage 2 focused pre-implementation contract run:
  - Command: `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py::test_route_catalog_matches_configured_chat_routes tests\test_control_console_cognition_debug_visibility.py::test_static_surface_adds_existing_widget_prompt_panels tests\control_console_e2e\test_page_navigation_e2e.py::test_owner_panels_use_panel_specific_readable_layouts tests\control_console_e2e\test_page_navigation_e2e.py::test_prompt_operational_and_tabular_surfaces_use_right_structures -q`
  - Result: expected failure. Failures prove the current implementation lacks `CODING_AGENT_PM_LLM` / `CODING_AGENT_PROGRAMMER_LLM`, `prompt-panel` / `record-card` static containers, user memory cards, and prompt-body rendering.
- 2026-07-05: Stage 3 and Stage 4 implementation completed in:
  - `src/control_console/brain_model_routes.py`
  - `src/control_console/static/index.html`
  - `src/control_console/static/console.css`
  - `src/control_console/static/console.js`
  - `tests/test_control_console_brain_model_routes.py`
  - `tests/test_control_console_cognition_debug_visibility.py`
  - `tests/control_console_e2e/test_page_navigation_e2e.py`
- 2026-07-05: Focused verification passed:
  - Command: `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py tests\test_control_console_cognition_debug_visibility.py tests\control_console_e2e\test_page_navigation_e2e.py tests\control_console_e2e\test_visual_product_acceptance_e2e.py -q`
  - Result: `23 passed, 1 warning in 32.06s`. Warning is existing Starlette/httpx deprecation warning from `fastapi.testclient`.
- 2026-07-05: Static verification passed:
  - Command: `rg "CODING_AGENT_PM_LLM|CODING_AGENT_PROGRAMMER_LLM" src\control_console tests\test_control_console_brain_model_routes.py README.md docs\HOWTO.md`
  - Result: route names present in docs, route catalog, and route tests.
  - Command: `rg "Brain stopped" src\control_console\static`
  - Result: zero matches; command exited 1 as expected for no matches.
  - Command: `git diff --check`
  - Result: no whitespace errors; only Git line-ending warnings for touched files.
- 2026-07-05: Final Playwright/Chrome old-vs-new production-backed comparison captured against production data:
  - Dev console under test: `http://127.0.0.1:8876/`, started from the production working directory and current dev checkout.
  - Old production console: `http://localhost:8765/`.
  - Artifact root: `C:\Users\RANBAO~1\AppData\Local\Temp\kazusa_console_old_new_final_486_bqb9`
  - Summary JSON: `C:\Users\RANBAO~1\AppData\Local\Temp\kazusa_console_old_new_final_486_bqb9\comparison_summary.json`
  - Final desktop screenshots include `old_8765` and `new_8876` subdirectories for Overview, Services, Debug, Events, Character, Users, Groups, Calendar, Background, Health, and Audit.
  - Final mobile screenshots: `C:\Users\RANBAO~1\AppData\Local\Temp\kazusa_console_old_new_final_486_bqb9\new_mobile`
  - Old diagnostics: Services had 12 routes and both coding-agent routes absent; Character dynamic prompt panels were below static profile; Growth Runs Audit had 77 table rows; User Memory had 25 table rows; Calendar schedule definitions had 173 table rows; Background Job Queue had 119 table rows; Worker Events had 202 table rows; Event/Audit headers were absent; Debug still contained `Brain stopped`.
  - New diagnostics: Services has 14 routes with both coding-agent routes present; Character dynamic prompt panels precede profile; Growth Runs Audit has 25 record cards and zero table rows; User Memory has 25 record cards and zero table rows in a full-width card; Calendar schedule definitions have 25 record cards and zero table rows; Background Job Queue has 9 record cards and zero table rows; Worker Events has 25 record cards and zero table rows; Event/Audit headers are present; Debug has zero `Brain stopped` text matches.
- 2026-07-05: Local independent review completed after final verification:
  - Scope reviewed: route catalog descriptors, static HTML/CSS layout, JavaScript render-helper boundary, e2e/static tests, final production-backed screenshots and diagnostics, and plan compliance.
  - Findings: no blocking findings. The shared render-helper change is bounded by target element type: existing `tbody` targets keep table output, converted `div` targets receive prompt/card output.
  - Residual risk: production pages with large result sets are still long because limits are intentionally bounded at 25, but final desktop/mobile screenshots show no overlap and no horizontal overflow in the checked flows.
