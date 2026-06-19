# control_console_entity_information_architecture_plan.md

## Summary

- Goal: Reorganize the control-console inspection UI around the semantic owners `Character`, `Users`, and `Groups`, while reusing framework-aligned panel primitives for lookup, status, tables, style overlays, lineage, and unavailable states.
- Plan class: medium
- Status: completed
- Mandatory skills: `development-plan`, `py-style`, `test-style-and-execution`, `build-web-apps:shadcn`, `build-web-apps:frontend-testing-debugging`
- Overall cutover strategy: bigbang replacement of the top-level `Memory` and `Interaction style` sidebar pages with owner-oriented `Users` and `Groups` pages; no compatibility sidebar aliases, no mockup-only pages, and no Node.js frontend stack.
- Highest-risk areas: confusing entity ownership, stale UI tests that keep implementation-shaped pages alive, duplicated panel rendering, hidden custom widgets that drift away from shadcn-style controls, empty pages that look functional, and lookup contracts that leak internal identifiers.
- Acceptance criteria: The running console groups character-owned, user-owned, and group-owned data on the correct pages; top-level `Memory` and `Interaction style` pages are gone; reusable panel primitives are used where they reduce real duplication; UI controls align with the existing shadcn-style design system and scale without desktop-only hard-coding; deterministic and Playwright-backed E2E tests pass; independent code review passes.

## Context

The current control-console UI is organized by implementation data type:
`Character`, `Memory`, and `Interaction style`. That structure does not match
the operator mental model or the database ownership model. The database ICD
defines distinct semantic owners:

- `character_state`, `global_character_growth_traits`, `global_character_growth_runs`, and shared/evolving `memory` are character or world/context surfaces.
- `user_profiles` and `user_memory_units` are user-owned surfaces.
- `interaction_style_images` stores two different durable owners: user style overlays and group-channel style overlays.

The accepted product decision is to follow **Option A with reusable panel
primitives**. The console must become entity-oriented at the page level while
reusing only the highly reusable parts of an Entity Inspector design. The
implementation must not create a vague generic inspector that hides ownership.

Existing constraints remain in force:

- The console is Python/FastAPI served and buildless.
- Static UI assets live under `src/control_console/static/`.
- Standard UI elements must follow shadcn component-family anatomy using the existing static HTML/CSS/JS approach.
- Operators should not enter internal `global_user_id` values. User-facing lookup stays platform-facing.

## Mandatory Skills

- `development-plan`: load before changing this plan status, reviewing it, executing it, or reporting completion.
- `py-style`: load before editing Python production code or Python tests.
- `test-style-and-execution`: load before adding, changing, running, or interpreting tests.
- `build-web-apps:shadcn`: load before changing static UI markup, component anatomy, form controls, cards, tables, badges, dialogs, sheets, tabs, or visual styling.
- `build-web-apps:frontend-testing-debugging`: load before validating rendered UI behavior. Use the Browser plugin first when available; if unavailable, record the Browser failure and use the existing Playwright-backed E2E path.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution unless the user explicitly approves fallback execution.
- Keep the control console static, Python-served, and buildless. Do not introduce Node.js, npm, pnpm, yarn, React, Vue, Vite, Webpack, Tailwind build tooling, or a frontend package-manager workflow.
- UI elements MUST align with the existing shadcn-style component families and the current page design. Use existing widgets, gadgets, and component anatomy before creating any new markup pattern.
- Do not invent custom widgets for common controls. Buttons, cards, badges, tables, fields, selects, textareas, dialogs/sheets, tabs, empty states, scroll areas, and sidebars must map to the nearest existing shadcn-style pattern already used by the console.
- Custom CSS is allowed only for layout glue, responsive constraints, theme tokens, and static-delivery adaptation. It must not create a parallel design system.
- UI layout must support scale. Do not hard-code desktop-only widths, fixed page-only heights, or content assumptions that break when rows, cards, fields, or panel counts grow. Use wrapping field groups, bounded scroll areas, stable card/table dimensions, and reusable empty/unavailable states.
- Reusable panel primitives must be concrete and limited: panel shell, lookup scope, result table, style-guideline renderer, lineage renderer, and capability/empty/error state renderer. Do not implement a fully generic Entity Inspector framework in this plan.
- Keep semantic ownership visible in page copy and API contracts. Character memory and user memory must not be presented as the same thing.
- Keep internal identifiers internal. Browser forms must not ask operators to enter `global_user_id`.
- Route code must call existing repository/domain helpers or narrowly added repository adapters. Do not import raw MongoDB clients directly in route handlers.
- This plan does not authorize prompt, RAG, cognition, dialog, reflection, memory promotion, adapter transport, or database write-path changes.

## Must Do

- Replace top-level sidebar pages `Memory` and `Interaction style` with `Users` and `Groups`.
- Expand `Character` so it is the owner page for character state, character self-image, character/global growth, shared/world/character memory, and promoted/background learning summaries that are already safely projectable.
- Add `Users` as the owner page for platform-user lookup, user profile, affinity/relationship insight, user memory units, and user-scoped interaction style.
- Add `Groups` as the owner page for platform-group lookup, group-channel interaction style, group conversation progress or scene context when safely available, and group-related reflection-derived guidance.
- Reuse existing static UI widgets and the cognition graph gadget patterns where applicable. Reuse existing `Card`, `Badge`, `Table`, `Select`, `Input`, `Button`, `FieldGroup`, sidebar, and empty/unavailable-state patterns.
- Create a small static JS panel primitive layer only where it removes real duplication in rendering status, tables, style overlays, lineage, and lookup states.
- Update backend lookup routes or add owner-oriented lookup routes only where needed to support the new pages. Keep existing repository helper ownership semantic.
- Update tests so the old implementation-shaped pages cannot silently remain.
- Update `src/control_console/README.md` to document the new owner-oriented page capability model.
- Record execution evidence in this plan before marking it complete.

## Deferred

- Do not build a full generic Entity Inspector framework.
- Do not add a dynamic plugin architecture for arbitrary future panels.
- Do not add persistent user/group configuration.
- Do not add new database collections, migrations, or write paths.
- Do not add or change LLM prompts, cognition graphs, RAG routing, dialog generation, reflection promotion, memory evolution, or adapter behavior.
- Do not add semantic vector search for user or shared memory in this plan.
- Do not implement schedule editing, memory editing, style editing, or group moderation actions.
- Do not preserve top-level `Memory` or `Interaction style` sidebar aliases.
- Do not create standalone mockups as authoritative artifacts.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Sidebar information architecture | bigbang | Replace `Memory` and `Interaction style` with `Users` and `Groups`. Do not keep compatibility aliases. |
| Browser page IDs | bigbang | Replace old page IDs and navigation links in tests and UI. Do not keep hidden pages for old selectors. |
| Backend data ownership | compatible | Existing repository/domain helper internals may still call current DB helpers. User-facing routes and page labels must expose owner-oriented concepts. |
| UI component patterns | bigbang | New UI must use existing framework-aligned widgets/gadgets. Remove ad hoc controls when touched. |
| Tests | bigbang | Rewrite expectations to the new owner-oriented pages. Old top-level memory/style tests must fail if those pages reappear. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, delete or rewrite legacy references instead of preserving them.
- If an area is `compatible`, preserve only the compatibility surfaces explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

The sidebar contains owner-oriented inspection pages:

- `Character`
- `Users`
- `Groups`

Operational pages remain separate:

- `Overview`
- `Services`
- `Live logs`
- `Debug chat`
- `Event monitor`
- `Calendar`
- `Background work`
- `Health/cache`
- `Audit`

The completed UI answers these operator questions directly:

- "What is the character right now, and what has she learned about herself or the world?"
- "What does the system know about this user?"
- "What does the system know about this group/channel?"

Each owner page must show clear working, empty, gated, and unavailable states.
No page should look successful when required lookup input is absent or the
underlying helper is unavailable.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Page organization | Use `Character`, `Users`, and `Groups` as top-level owner pages | Operators inspect entities, not collection names. |
| Memory placement | Character/shared memory belongs on `Character`; user memory belongs on `Users` | Matches DB ICD ownership for `memory` and `user_memory_units`. |
| Style placement | User style belongs on `Users`; group-channel style belongs on `Groups` | `interaction_style_images` has separate user and group-channel scopes. |
| Reuse strategy | Reuse panel primitives, not a full generic inspector | Preserves consistency without hiding semantic differences. |
| UI controls | Use existing shadcn-style static controls | Avoids reinventing widgets and keeps visual language consistent. |
| Platform input | Use bounded platform selects | Platform is an enum-like operator choice, not free-form authored text. |
| Internal IDs | Keep `global_user_id` out of browser input | Operators should search by platform-facing identifiers. |
| Layout scale | Use wrapping fields, bounded tables, and reusable panel states | Supports growth without hard-coded desktop-only assumptions. |

## Contracts And Data Shapes

### Reusable Panel Primitives

Implement only these reusable rendering primitives in static JavaScript when
they remove repeated code:

```text
renderPanelState(target, {
  status: string,
  label: string,
  reason: string,
  generated_at: string
})

renderLookupTable(target, {
  items: list[object],
  emptyText: string,
  redaction: object
})

renderStyleOverlayRows(target, {
  items: list[object],
  scopeLabel: string
})

renderLineageRows(target, {
  revision: string | number,
  updated_at: string,
  source_reflection_run_ids: list[string],
  evidence_refs: list[object]
})
```

These are render helpers, not widgets with their own visual system. They must
emit markup that uses the current console's shadcn-style `Card`, `Badge`,
`Table`, `FieldGroup`, `Select`, `Input`, and `Button` anatomy.

### Owner Page Payloads

Owner-oriented routes or repository methods may compose existing helper results
into browser-safe envelopes:

```json
{
  "status": "available | empty | needs_input | unavailable",
  "generated_at": "ISO timestamp",
  "owner": "character | user | group",
  "identity": {},
  "panels": {
    "state": {},
    "memory": {},
    "style": {},
    "lineage": {},
    "progress": {}
  },
  "redaction": {}
}
```

Panel keys may be omitted when not applicable to the owner. Browser code must
render omitted panels as intentionally absent only when the page design says
the panel is not part of that owner. Missing data from an expected panel must
render as `unavailable`, `empty`, or `needs_input`.

### LLM Call And Context Budget

This plan adds no LLM calls and does not change any prompt, RAG, cognition,
dialog, reflection, memory promotion, or background LLM behavior. The before
and after LLM call count is unchanged.

## Change Surface

### Modify

- `src/control_console/static/index.html`
  - Replace top-level `memory` and `style` pages with `users` and `groups`.
  - Expand the `character` page with owner-owned panels.
  - Keep existing shadcn-style page, card, field, table, badge, and button anatomy.
- `src/control_console/static/console.js`
  - Replace page navigation and refresh handlers for `memory` and `style`.
  - Add narrowly scoped reusable render helpers for panel state, tables, style overlays, and lineage where they remove duplication.
  - Keep existing state-store and request patterns.
- `src/control_console/app.py`
  - Update page capability labels and owner-oriented lookup endpoints only where required.
  - Preserve auth, CSRF, redaction, and route-limit behavior.
- `src/control_console/contracts.py`
  - Add or rename browser response contracts only if needed for owner-oriented envelopes.
- `src/control_console/repository.py`
  - Add owner-oriented composition methods only where existing helper calls need grouping.
  - Keep raw MongoDB access out of route handlers.
- `src/control_console/README.md`
  - Update page capability and ICD wording to describe `Character`, `Users`, and `Groups`.
- `development_plans/archive/completed/short_term/backend_control_console_development_plan.md`
  - Add a short cross-reference to this plan if execution starts, so the older in-progress plan no longer implies the implementation-shaped `Memory`/`Interaction style` sidebar is final.
- `tests/test_control_console_web_surface.py`
  - Assert new page links, old page links absent, owner-oriented selectors present, and platform selects remain bounded.
- `tests/test_control_console_repository.py`
  - Add or update deterministic repository composition tests for owner-oriented envelopes.
- `tests/control_console_e2e/test_page_navigation_e2e.py`
  - Assert sidebar page activation, gated/empty states, and connected request paths for `Character`, `Users`, and `Groups`.
- `tests/control_console_e2e/test_clickable_inventory_e2e.py`
  - Update clickable inventory for new owner pages.
- `tests/control_console_e2e/test_visual_product_acceptance_e2e.py`
  - Update visual acceptance expectations so active navigation and panel layout remain clear.

### Keep

- Existing service lifecycle, auth/session, CSRF, debug chat, live logs, cognition graph, event monitor, calendar, background work, health/cache, and audit behavior.
- Existing database helper ownership and storage semantics.
- Existing Python/FastAPI static delivery model.

### Delete

- Top-level `Memory` sidebar/page markup and tests as a standalone product page.
- Top-level `Interaction style` sidebar/page markup and tests as a standalone product page.
- Any duplicated render code replaced by the approved small panel primitives.

## Overdesign Guardrail

- Actual problem: Operators cannot understand the console because inspection pages are grouped by data type instead of by character, user, and group ownership.
- Minimal change: Reorganize the static console around owner pages and introduce only small reusable render helpers for repeated panel behavior.
- Ownership boundaries: UI owns arrangement and presentation; `control_console.repository` owns safe read composition; DB modules own storage semantics; brain/adapters/cognition remain unchanged.
- Rejected complexity: full generic Entity Inspector framework, plugin panel registry, new frontend stack, database migrations, edit workflows, LLM summarization, compatibility sidebar aliases, and broad layout redesign.
- Evidence threshold: Add a generic inspector or plugin framework only after at least three new owner page types or external panel providers require runtime extensibility and existing panel primitives create measurable duplication or inconsistent behavior.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside `src/control_console`, its tests, and this plan's named docs as high-scrutiny changes.
- The responsible agent may extract small render helpers only when two or more pages use the same state/table/style/lineage behavior.
- The responsible agent must search the existing static HTML, CSS, JS, repository helpers, and tests before creating a new helper or markup pattern.
- If an existing widget/gadget pattern already exists, reuse or adapt it instead of creating a new one.
- The responsible agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, database writes, or broad refactors.
- If this plan and code disagree, preserve the owner-oriented intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Implementation Order

1. Parent establishes focused UI contract tests.
   - Update `tests/test_control_console_web_surface.py` so `memory` and `style` page links are forbidden, `users` and `groups` links are required, and platform selectors remain bounded.
   - Run the focused test and record the expected failure against the current UI.
2. Parent establishes focused E2E navigation tests.
   - Update `tests/control_console_e2e/test_page_navigation_e2e.py` for `Character`, `Users`, and `Groups` navigation and lookup gating.
   - Run the focused E2E test and record the expected failure.
3. Parent starts the production-code subagent.
   - Provide this approved plan, mandatory skills, focused failing tests, and the production boundary `src/control_console/**`.
   - The subagent changes production code only.
4. Production-code subagent updates static navigation and page layout.
   - Modify `src/control_console/static/index.html` and `src/control_console/static/console.js`.
   - Remove standalone memory/style page handlers.
5. Production-code subagent adds owner-oriented repository composition only where the UI needs grouped payloads.
   - Keep helper calls semantic and read-only.
6. Parent updates integration and clickable-inventory tests while production code is in progress.
   - Update E2E selectors and expected request URLs.
7. Parent runs focused tests.
   - Rerun the tests from steps 1 and 2 and record pass/fail evidence.
8. Parent runs regression verification.
   - Run all commands in `Verification`.
9. Parent starts the independent code-review subagent.
   - The review subagent reviews the full diff against this plan and reports findings only.
10. Parent remediates review findings inside the approved change surface and reruns affected verification.
11. Parent records final evidence and updates lifecycle status only after all acceptance criteria are met.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the focused test contract is established; owns production code changes only; does not edit tests unless the parent explicitly directs it; closes after planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks, and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after planned verification passes; reviews the plan, diff, and evidence; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused contract tests established
  - Covers: Implementation Order steps 1-2.
  - Verify: focused static and E2E tests fail for the current implementation-shaped page organization.
  - Evidence: record failing commands and failure reasons in `Execution Evidence`.
  - Sign-off: `parent/2026-06-19` after evidence is recorded.
- [x] Stage 2 - owner-oriented production UI implemented
  - Covers: Implementation Order steps 3-5.
  - Verify: focused static UI test passes and no old top-level memory/style page selectors remain outside negative assertions.
  - Evidence: record changed production files and focused test output.
  - Sign-off: `parent/2026-06-19` after evidence is recorded.
- [x] Stage 3 - integration and visual tests updated
  - Covers: Implementation Order steps 6-8.
  - Verify: all `Verification` commands pass or documented blockers are resolved.
  - Evidence: record command output summaries and any Browser fallback reason.
  - Sign-off: `parent/2026-06-19` after evidence is recorded.
- [x] Stage 4 - independent code review complete
  - Covers: Implementation Order steps 9-10.
  - Verify: independent review reports no unresolved blockers; affected tests rerun after fixes.
  - Evidence: record findings, fixes, commands rerun, residual risks, and approval status.
  - Sign-off: `parent/2026-06-19` after evidence is recorded.
- [x] Stage 5 - pre-live verification and review sign-off
  - Covers: Implementation Order step 11.
  - Verify: non-live acceptance criteria are checked against the running code and docs.
  - Evidence: record final status update and sign-off line.
  - Sign-off: `parent/2026-06-19` after evidence is recorded.
- [x] Stage 6 - real database and real rendered data validation
  - Covers: missing live-data verification found after the premature Stage 5 sign-off.
  - Verify: real MongoDB-backed Character, Users, and Groups paths return meaningful, browser-safe, human-readable data, or fail with concrete UX/data-format defects mapped to code.
  - Evidence: record read-only DB discovery method, tested real sample identifiers without sensitive values, endpoint panel statuses/counts, redaction scan result, rendered UI findings, fixes, and rerun evidence.
  - Sign-off: `parent/2026-06-19` after real DB rendered screenshots and verification evidence are recorded.

## Verification

### Static Greps

- `rg -n 'data-page-link="memory"|data-page="memory"|data-page-link="style"|data-page="style"' src/control_console/static tests`
  - Expected: matches only in negative assertions or historical comments explicitly documenting removed pages. Production static UI must have zero matches.
- `rg -n 'memory-global-user-id|style-global-user-id|global user id' src/control_console/static tests/control_console_e2e tests/test_control_console_web_surface.py`
  - Expected: matches only in negative assertions. Browser-facing copy must not ask for internal global user id.
- `rg -n 'npm|pnpm|yarn|vite|webpack|react|vue' src/control_console pyproject.toml package.json`
  - Expected: no new frontend-stack references introduced by this plan. Existing unrelated files outside `src/control_console` are not part of this check.

### Tests

- `venv\Scripts\python -m pytest tests\test_control_console_repository.py tests\test_console_lookup_limits.py tests\test_control_console_web_surface.py -q -s`
- `venv\Scripts\python -m pytest tests\control_console_e2e\test_page_navigation_e2e.py tests\control_console_e2e\test_clickable_inventory_e2e.py tests\control_console_e2e\test_visual_product_acceptance_e2e.py -q -s`
- `venv\Scripts\python -m pytest tests\control_console_e2e -q -s`

### Static Checks

- `venv\Scripts\python -m compileall src\control_console tests\test_control_console_repository.py tests\test_console_lookup_limits.py tests\test_control_console_web_surface.py tests\control_console_e2e`
- `git diff --check`

### Rendered UI Validation

- Use Browser plugin validation when available.
- If Browser reports unavailable, record the exact availability failure and rely on the Playwright-backed E2E commands above.
- The flow under test is: login -> sidebar navigation -> Character, Users, and Groups owner pages render populated, gated, empty, or unavailable panels correctly -> lookup controls send canonical platform-facing parameters.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for changed Python, static UI, tests, docs, and plan artifacts.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, contracts, implementation order, verification gates, and acceptance criteria.
- UI design quality: framework-aligned widgets, no bespoke common controls, scalable layout constraints, consistent empty/error/loading states, and no reintroduced busy one-page layout.
- Architecture quality: owner-oriented page grouping, no hidden generic inspector framework, no raw MongoDB route access, no prompt/RAG/cognition changes, no frontend stack changes.
- Regression quality: old page selectors forbidden, new owner pages tested, lookup URLs canonical, Browser fallback recorded if applicable.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `Character` contains character state, self-image, growth, shared/world/character memory, and promoted/background learning panels where safely available.
- `Users` contains platform-user lookup, user profile, relationship summary, user memory, and user style panels.
- `Groups` contains platform-group lookup, group-channel style, and group progress/reflection-derived context panels where safely available.
- The standalone top-level `Memory` and `Interaction style` sidebar pages are absent.
- The UI uses existing shadcn-style component anatomy and existing console widgets/gadgets before any new markup pattern.
- Reusable panel primitives exist only for repeated state, table, style, lineage, and lookup rendering.
- Layout supports scale through wrapping fields, bounded tables, stable cards, and clear empty/unavailable states.
- No browser form asks for internal `global_user_id`.
- Verification commands pass.
- Independent code review has no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Reorganization hides data that used to be visible | Owner pages must carry old working surfaces into the correct entity page | E2E navigation and static selector tests |
| Reusable helpers become an over-generic inspector | Only allowed primitives are state, table, style, lineage, and lookup render helpers | Code review and grep for new framework-like abstractions |
| UI drifts from framework design | Use existing shadcn-style component anatomy and static console widgets | Visual product acceptance E2E and code review |
| Layout works only on one desktop width | Use wrapping fields, bounded cards/tables, and no desktop-only fixed layout assumptions | Visual product acceptance E2E and CSS review |
| Backend leaks internal IDs | Keep platform-facing lookup and redact internal canonical identity from browser input | Static grep and repository tests |

## Execution Evidence

- Pre-implementation focused failures:
  - `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q -s` failed as expected: static shell still contained `data-page-link="memory"`.
  - `venv\Scripts\python -m pytest tests\control_console_e2e\test_page_navigation_e2e.py::test_each_sidebar_page_has_connected_or_explicitly_gated_state -q -s` failed as expected: rendered Character page did not request `/api/entities/character`.
  - `venv\Scripts\python -m pytest tests\test_control_console_repository.py::test_repository_composes_owner_entity_envelopes -q -s` failed as expected: `ControlConsoleRepository.character_entity` is not implemented yet.
  - `venv\Scripts\python -m pytest tests\test_console_lookup_limits.py::test_lookup_routes_enforce_pagination_redaction_and_no_embeddings -q -s` failed as expected: `/api/entities/user` is not implemented yet and returned 404.
- Production implementation summary:
  - One production-code subagent updated the approved production source files: `src/control_console/app.py`, `src/control_console/repository.py`, `src/control_console/static/index.html`, and `src/control_console/static/console.js`.
  - The same subagent also made the adjacent README/cross-reference documentation edits listed in this plan's `Change Surface`: `src/control_console/README.md` and `development_plans/archive/completed/short_term/backend_control_console_development_plan.md`. Parent retained ownership of test and plan-evidence edits. This is recorded explicitly because the execution model intended production-code-only subagent work; no unapproved code surface was changed.
  - Added owner-oriented routes `/api/entities/character`, `/api/entities/user`, and `/api/entities/group` using repository composition methods instead of raw MongoDB access in route handlers.
  - Replaced top-level `Memory` and `Interaction style` sidebar entries with `Users` and `Groups`; expanded `Character` panels for character-owned profile/state/self-image/growth/memory/learning surfaces.
  - Added static JS panel rendering helpers for repeated panel state and lookup table output while keeping existing card, badge, field, table, select, input, and button anatomy.
  - Updated `src/control_console/README.md` with the owner-oriented page capability model and cross-referenced this plan from the older backend-control-console plan.
- Static grep results:
  - `rg -n 'data-page-link="memory"|data-page="memory"|data-page-link="style"|data-page="style"' src\control_console\static tests` returned matches only in negative assertions in `tests/test_control_console_web_surface.py`; production static UI had zero matches.
  - `rg -n 'memory-global-user-id|style-global-user-id|global user id' src\control_console\static tests\control_console_e2e tests\test_control_console_web_surface.py` returned matches only in negative assertions.
  - `package.json` is absent. The frontend-stack grep found only the existing README sentence explicitly forbidding Node.js/npm/pnpm/yarn/React/Vue/Vite/Webpack/Tailwind tooling.
- Test results:
  - `venv\Scripts\python -m pytest tests\test_control_console_repository.py tests\test_console_lookup_limits.py tests\test_control_console_web_surface.py -q -s`: 22 passed.
  - `venv\Scripts\python -m pytest tests\control_console_e2e\test_page_navigation_e2e.py tests\control_console_e2e\test_clickable_inventory_e2e.py tests\control_console_e2e\test_visual_product_acceptance_e2e.py -q -s`: 4 passed.
  - `venv\Scripts\python -m pytest tests\control_console_e2e -q -s`: 13 passed, 1 skipped. The skipped test is the existing opt-in real service lifecycle test.
  - `venv\Scripts\python -m compileall src\control_console tests\test_control_console_repository.py tests\test_console_lookup_limits.py tests\test_control_console_web_surface.py tests\control_console_e2e`: passed.
  - `git diff --check`: passed with CRLF warnings only.
- Rendered UI validation:
  - Browser plugin validation attempted against an isolated console server launched without `.env` using explicit test settings. Browser runtime reported `Browser is not available: iab`, so rendered validation used the existing Playwright-backed E2E suite per plan fallback.
  - Temporary isolated browser-validation process and run-id temp artifacts were verified cleaned up after the Browser fallback.
- Independent code review:
  - One independent review subagent (`gpt-5.5`, `xhigh`) reviewed the full working-tree diff against this plan after verification.
  - Review result: no Critical findings.
  - Important finding 1: `/api/entities/user?query=<241 chars>` returned a server-side validation exception instead of bounded 422. Fix: added FastAPI `Query(default="", max_length=240)` validation for `/api/entities/user` and the sibling `/api/lookups/memory` route; added route assertions for both overlong query paths.
  - Important finding 2: unavailable owner style/guidance panels could render as generic empty rows or fake lineage. Fix: added browser-backed regression coverage and changed `console.js` to render `renderPanelState(...)` when style/guidance panels have no rows but do have status/reason; `renderLineageRows(...)` is now used only when actual lineage fields exist.
  - Minor finding: execution evidence blurred parent/subagent ownership. Fix: clarified production source, documentation, test, and plan-evidence ownership in this section.
  - Focused red checks before fixes: the route regression failed with a Pydantic validation exception, and the E2E regression rendered `No user style guidance rows are available.` instead of the unavailable reason.
  - Focused green checks after fixes:
    - `venv\Scripts\python -m pytest tests\test_console_lookup_limits.py::test_lookup_routes_enforce_pagination_redaction_and_no_embeddings -q -s`: passed.
    - `venv\Scripts\python -m pytest tests\control_console_e2e\test_page_navigation_e2e.py::test_owner_entity_unavailable_panels_render_reasons -q -s`: passed.
  - Verification rerun after review fixes:
    - `venv\Scripts\python -m pytest tests\test_control_console_repository.py tests\test_console_lookup_limits.py tests\test_control_console_web_surface.py -q -s`: 22 passed.
    - `venv\Scripts\python -m pytest tests\control_console_e2e\test_page_navigation_e2e.py tests\control_console_e2e\test_clickable_inventory_e2e.py tests\control_console_e2e\test_visual_product_acceptance_e2e.py -q -s`: 5 passed.
    - `venv\Scripts\python -m pytest tests\control_console_e2e -q -s`: 14 passed, 1 skipped. The skipped test is the existing opt-in real service lifecycle test.
    - Static greps still returned only negative assertions for old memory/style page selectors and global-user-id browser fields.
    - `venv\Scripts\python -m compileall src\control_console tests\test_control_console_repository.py tests\test_console_lookup_limits.py tests\test_control_console_web_surface.py tests\control_console_e2e`: passed.
    - `git diff --check`: passed with CRLF warnings only.
  - Residual risks: Browser plugin remained unavailable in this environment, so rendered validation continues to rely on the Playwright-backed E2E suite as permitted by this plan.
  - Approval status: no unresolved independent-review blockers remain.
- Premature final sign-off correction:
  - Previous final status `completed` was revoked on 2026-06-19 because no real database validation had been run.
  - Current status after Stage 6 completion: `completed`.
  - Required remaining gate before completion: Stage 6 real database and real rendered data validation.
  - Sign-off standard: the plan cannot be marked completed until real MongoDB-backed Character, Users, and Groups paths are verified for human-readable, correctly formatted, redacted UI output.
  - Previous non-live acceptance check:
    - `Character`, `Users`, and `Groups` are present as owner-oriented pages and top-level standalone `Memory` / `Interaction style` pages are absent.
    - Browser inputs use platform-facing identifiers and do not ask for `global_user_id`.
    - New owner routes use repository composition methods from `ControlConsoleRepository`; no raw MongoDB access was added to route handlers.
    - UI output reuses existing static card, badge, field, select, table, button, and empty-state anatomy; no frontend build stack or package manager was introduced.
    - Empty, needs-input, unavailable, and populated states are covered by deterministic and Playwright-backed E2E tests, including review-added unavailable-panel regressions.
    - Independent review has no unresolved blockers after fixes.
  - Revocation sign-off: `parent/2026-06-19`.
- Stage 6 real database and real rendered data validation:
  - Read-only DB discovery used project scripts/helpers that load configured MongoDB through the normal application path; `.env` contents were not manually opened or printed. Collections sampled for schema and candidate selection were `character_state`, `user_profiles`, `user_memory_units`, `interaction_style_images`, and recent `conversation_history`.
  - Safe live samples were selected by platform-facing lookup fields only. Evidence records masked identifiers: user platform `qq` with masked platform user id `67***19`, and group platform `qq` with masked group id `22***60`.
  - Initial live DB E2E failure: `tests\control_console_e2e\test_live_database_owner_pages_e2e.py::test_live_database_owner_pages_render_human_readable_data` failed because `#character-profile-table` rendered `[object Object]` for a nested real `personality_brief` value. Root cause: `renderLookupTable(...)` treated every value as scalar text.
  - Additional rendered-data UX gaps found by screenshot inspection:
    - Group style guidance initially used four narrow columns, which squeezed Chinese guidance text into unreadable vertical wrapping.
    - `.content-grid.two` used auto-fit, so two-column owner panels could become three columns on wide screens.
    - Generic key/value payloads rendered artificial `key` / `value` rows instead of using the key as the field label.
    - User memory rendered each document as many flattened field rows, producing a long stream of implementation fields instead of one memory unit per row.
    - Grid stretch made short cards such as `Progress` and `User Style` expand to the height of tall neighboring data cards, leaving large empty panels.
  - Fixes applied inside the approved static console surface:
    - `src/control_console/static/console.js`: added recursive value formatting for nested arrays/objects, key/value row detection, panel-specific memory-unit row rendering, and safer style-overlay rows.
    - `src/control_console/static/console.css`: capped `.content-grid.two` at two columns with a one-column responsive fallback, added table hierarchy text classes, and aligned grid items to content height.
    - `tests/control_console_e2e/test_page_navigation_e2e.py`: added regressions for nested object rendering, readable two-column style guidance, key/value panel formatting, and one-row-per-memory-unit rendering.
    - `tests/control_console_e2e/test_live_database_owner_pages_e2e.py`: records separate real DB screenshots for Character, Users, and Groups pages.
    - `tests/test_control_console_web_surface.py`: asserts scalable two-column grid and no grid-card stretch.
  - Live DB rerun evidence:
    - `KAZUSA_RUN_CONTROL_CONSOLE_LIVE_DB_E2E=1 venv\Scripts\python -m pytest tests\control_console_e2e\test_live_database_owner_pages_e2e.py::test_live_database_owner_pages_render_human_readable_data -q -s`: passed.
    - Latest summary artifact: `C:\Users\Ran Bao\AppData\Local\Temp\pytest-of-Ran Bao\pytest-1211\test_live_database_owner_pages0\artifacts\live_database_owner_pages.summary.json`.
    - Latest screenshots:
      - Character: `C:\Users\Ran Bao\AppData\Local\Temp\pytest-of-Ran Bao\pytest-1211\test_live_database_owner_pages0\artifacts\live_db_owner_character.png`.
      - Users: `C:\Users\Ran Bao\AppData\Local\Temp\pytest-of-Ran Bao\pytest-1211\test_live_database_owner_pages0\artifacts\live_db_owner_user.png`.
      - Groups: `C:\Users\Ran Bao\AppData\Local\Temp\pytest-of-Ran Bao\pytest-1211\test_live_database_owner_pages0\artifacts\live_db_owner_group.png`.
    - Panel counts from live data: Character `profile=1`, `self_image=1`, `state=4`, `learning=1`; Users `profile=1`, `relationship=1`, `memory=25`, `style=4`; Groups `style=4`, `progress=0`, `guidance=0`.
    - Redaction scan result: no visible `global_user_id`, embeddings, prompts, or raw object placeholders in the live owner-page browser output.
  - Stage 6 focused verification:
    - `venv\Scripts\python -m pytest tests\control_console_e2e\test_page_navigation_e2e.py::test_owner_tables_use_panel_specific_readable_layouts -q -s`: passed.
    - `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q -s`: passed.
    - `venv\Scripts\python -m pytest tests\control_console_e2e\test_page_navigation_e2e.py -q -s`: passed as part of focused E2E regression run.
