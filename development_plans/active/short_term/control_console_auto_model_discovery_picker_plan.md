# control console auto model discovery picker plan

## Summary

- Goal: remove manual Brain route model entry from the Control Console and make model selection depend on discovered models from the configured OpenAI-compatible route endpoint.
- Plan class: medium.
- Status: completed.
- Mandatory skills: `development-plan`, `control-console-web-development`, `py-style`, `test-style-and-execution`, `build-web-apps:frontend-testing-debugging`.
- Overall cutover strategy: bigbang.
- Highest-risk areas: route editor state transitions, provider discovery empty/single/multiple states, accidental arbitrary model submission, stale browser JavaScript during rendered validation.
- Acceptance criteria: manual model ID is absent from the Services tab, discovery is automatic for the selected route, single-model discovery renders as a read-only state instead of a pointless picker, multi-model discovery remains selectable, and focused tests plus rendered validation pass without UI warnings or errors.

## Context

Commit `26e6d2f` added Control Console Brain model route switching. The current route editor shows both an `Available model` select and a `Manual model ID` input. That creates two problems:

- When the configured provider exposes only one model, the select looks like a choice even though there is nothing useful to choose.
- The manual input undermines the intended operator workflow: the console should discover available provider models from the configured endpoint and let the operator pick from those discovered models, not type arbitrary model IDs into the route editor.

The backend discovery function already exists in `src/control_console/brain_model_routes.py`:

- `fetch_available_models(base_url, api_key, *, transport=None)`
- It calls the configured OpenAI-compatible route endpoint's `/models` path.
- It bounds the list, validates model IDs, detects model family labels, and redacts provider failures.

The public route also already exists in `src/control_console/app.py`:

- `GET /api/services/brain/model-routes/{route_key}/available-models`

This plan must refine the existing discovery and UI behavior. It must not add a parallel model discovery abstraction.

Adjacent improvement areas intentionally deferred:

- Bulk discovery for all 13 routes on page load.
- Backend enforcement that every REST `PUT /model-routes/{route_key}` model value must be present in a fresh provider discovery result.
- Generic Brain service configuration redesign.
- Persistent provider model-list caching across browser sessions or console restarts.

## Mandatory Skills

- `development-plan`: load before approving, executing, reviewing, or signing off this plan.
- `control-console-web-development`: load before changing Control Console static UI, route handlers, browser validation, or screenshot evidence.
- `py-style`: load before editing Python production code.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `build-web-apps:frontend-testing-debugging`: load before rendered frontend validation.

## Mandatory Rules

- Do not read `.env`. Use normal application configuration paths and test-provided environment fixtures.
- Preserve the Control Console boundary: local operator browser -> control-console FastAPI app -> local supervisor and bounded Brain/provider HTTP calls.
- Do not mount the console into the Brain service and do not change Brain `/chat`, cognition, RAG, memory, prompt, scheduler, adapter, or dialog behavior.
- Keep the frontend buildless: static HTML, CSS, and JavaScript served by FastAPI. Do not add React, Vue, Vite, Tailwind, npm, or a frontend build step.
- The existing `fetch_available_models()` function is the discovery owner. Do not create a duplicate provider discovery function.
- Keep secrets out of responses, audit events, browser state, screenshots, and tests. Provider API keys and raw provider error bodies must remain redacted.
- UI controls must use existing console component classes and spacing. Do not introduce a separate visual system.
- Every visible control changed by this plan must be exercised in rendered validation.
- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution unless the user explicitly approves fallback execution.

## Must Do

- Remove the `Manual model ID` field from the Brain route editor UI.
- Make the selected route automatically discover models from the configured route endpoint when the editor opens and the route has no cached discovery result.
- Keep a user-triggered retry control for discovery refresh.
- Render discovery states explicitly:
  - loading
  - unavailable
  - empty
  - single discovered model matching the current effective model
  - single discovered model different from the current effective model
  - multiple discovered models
- Render no dropdown when discovery returns exactly one model.
- Disable apply when the selected or discovered model equals the route's effective model and no other dirty value exists.
- Preserve route reset behavior.
- Preserve max completion tokens and thinking toggle editing.
- Keep API responses redacted and bounded.
- Update docs and tests for the new discovery-only UI behavior.

## Deferred

- Do not add a bulk all-routes discovery endpoint.
- Do not add server-side provider validation during route apply.
- Do not remove existing REST support for model values in `BrainModelRouteApplyRequest`.
- Do not redesign generic service configuration.
- Do not persist provider model lists outside the browser session.
- Do not change route defaults, `.env` names, LLM interface behavior, model-family detection, or Brain startup route reporting.
- Do not add compatibility UI paths for manual model entry.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Services tab route editor | bigbang | Remove the manual model text input directly. Do not preserve it hidden, disabled, or behind an advanced toggle. |
| Discovered model UI states | bigbang | Replace the current always-select rendering with explicit empty, single, and multiple states. |
| Existing discovery function | compatible | Preserve `fetch_available_models()` as the discovery owner and refine its status contract. |
| Route apply API | compatible | Preserve the existing authenticated route apply endpoint and schema. UI must submit only discovered model IDs, but the API schema is not narrowed in this plan. |
| Tests | bigbang | Update tests to expect discovery-only UI behavior and no manual model field. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, delete or rewrite legacy references instead of preserving them.
- If an area is `compatible`, preserve only the compatibility surfaces explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

The Services tab Brain card presents the selected route as a provider-discovered model editor:

- On selecting a route, the browser automatically calls `/api/services/brain/model-routes/{route_key}/available-models` if that route has no cached discovery result.
- The route editor never exposes a free-text model ID field.
- When the provider returns no valid models, the editor shows a bounded message and a retry action.
- When the provider returns one model, the editor shows the model as a read-only discovered model row.
- When the provider returns one model that differs from the current effective model, the route has a dirty `model` value equal to that discovered model and the apply button is enabled.
- When the provider returns one model equal to the current effective model, the apply button stays disabled unless another field is dirty.
- When the provider returns multiple models, the editor shows a select populated only by discovered model IDs.
- The apply payload still uses the existing route apply contract:

```json
{
  "reason": "operator console model route change",
  "expected_version": 13,
  "values": {
    "model": "discovered-model-id"
  }
}
```

The route apply payload may also include `max_completion_tokens` and `thinking_enabled` when those controls are changed.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Discovery owner | Reuse and refine `fetch_available_models()` | The codebase already has bounded, redacted provider discovery. Duplicating it would create drift. |
| Manual model entry | Remove from UI | The requested workflow is automatic discovery from the configured endpoint. Manual entry is the wrong interaction model. |
| Single model state | Render read-only state, not a select | A one-option select is misleading and provides no choice. |
| Empty provider response | Return and render explicit `empty` status | A successful provider response with zero valid model IDs is different from a failed provider call. |
| Auto discovery timing | Lazy-load the selected route only | Avoids 13 simultaneous provider calls on Services tab load while still making the editor automatic. |
| API schema | Preserve current apply schema | Narrowing the REST contract requires separate approval because existing authenticated callers may submit route model values directly. |
| Browser cache | Use existing `state.availableModelCache` | It already scopes discovery state per route and avoids new persistence complexity. |

## Contracts And Data Shapes

### Available Models Response

`GET /api/services/brain/model-routes/{route_key}/available-models` returns:

```json
{
  "route_key": "cognition_llm",
  "status": "available",
  "models": [
    {"id": "gemma-4-31b-it-qat-uncensored-heretic-nvfp4", "family": "gemma4"}
  ],
  "message": null
}
```

Allowed status values after this plan:

- `available`: provider responded successfully and at least one valid model ID was returned.
- `empty`: provider responded successfully but no valid model IDs survived sanitization.
- `unavailable`: route base URL or API key is missing, provider call failed, provider response was invalid, or provider error occurred.

`models` remains bounded by `MODEL_LIST_LIMIT`. `message` remains redacted and must not include provider response bodies, API keys, authorization headers, dotenv contents, or URLs with credentials.

### Browser Cache Shape

`state.availableModelCache[routeKey]` remains:

```js
{
  status: "not_loaded" | "loading" | "available" | "empty" | "unavailable",
  models: [{id: string, family: string}],
  message: string
}
```

### Dirty Route Values

`state.dirtyBrainRouteValues[routeKey].model` may be set only from a discovered model ID selected or accepted by the UI.

The browser must not provide a free-text source for `model`.

## LLM Call And Context Budget

No LLM calls are added, removed, or changed. No prompt, RAG, cognition, dialog, evaluator, or background LLM context is changed.

Provider model discovery remains a deterministic HTTP GET to the configured OpenAI-compatible `/models` endpoint and does not enter the Brain live response path.

## Change Surface

### Modify

- `src/control_console/brain_model_routes.py`
  - Refine `fetch_available_models()` and `_sanitize_model_payload()` handling so successful empty results return `status: "empty"` instead of `available` with an empty list.
  - Keep validation, bounding, sorting, family detection, and redaction.
- `src/control_console/app.py`
  - Preserve the existing available-models route and audit event.
  - Ensure the route passes through the refined `empty` status.
- `src/control_console/static/console.js`
  - Remove `Manual model ID` markup.
  - Add discovery-state rendering for loading, unavailable, empty, single-model, and multi-model states.
  - Add an `ensureBrainRouteModelsLoaded(routeKey)` or equivalent local UI function that triggers discovery when the selected route has `not_loaded` state.
  - Update route selection, route refresh, and route render paths to avoid repeated discovery loops.
  - Ensure apply payload model values originate only from discovered model IDs.
- `src/control_console/static/console.css`
  - Add only minimal styling for read-only discovered model state using existing card, field, badge, input, and notice visual language.
- `src/control_console/README.md`
  - Document discovery-only browser model selection and the single-model state.
- `tests/test_control_console_brain_model_routes.py`
  - Add or update focused tests for `available`, `empty`, `unavailable`, duplicate IDs, invalid IDs, and redaction.
- `tests/test_control_console_config_routes.py`
  - Add or update API tests for the `empty` discovery response and existing redaction.
- `tests/test_control_console_web_surface.py`
  - Assert `Manual model ID` no longer appears.
  - Assert the static script contains the new discovery-state rendering functions and no manual model input path.

### Create

- No new production module is required.

### Delete

- Delete the manual model ID field markup and event path from `src/control_console/static/console.js`.

### Keep

- Keep `fetch_available_models()` as the discovery function.
- Keep `GET /api/services/brain/model-routes/{route_key}/available-models`.
- Keep `PUT /api/services/brain/model-routes/{route_key}` request shape.
- Keep `Refresh models`, renamed to `Retry discovery` only where the provider state is unavailable or empty if the UI text needs clearer state-specific language.

## Overdesign Guardrail

- Actual problem: the Services tab shows manual model entry and a one-option model picker that does not represent a real choice.
- Minimal change: remove manual model input, refine discovery statuses, and render route model controls according to the discovered model count.
- Ownership boundaries: Control Console owns browser interaction, provider discovery display, route override submission, audit, and redaction; Brain owns runtime route consumption; LLM interface owns actual model call behavior.
- Rejected complexity: no bulk route discovery endpoint, no server-side provider validation on apply, no persistent model cache, no advanced/manual mode, no feature flag, no compatibility UI, no new provider abstraction.
- Evidence threshold: add bulk discovery or server-side discovered-model enforcement only after an approved requirement shows real operator need, latency data, or API misuse that cannot be handled by the current route-level discovery flow.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside the target module as high-scrutiny changes. Updating an existing module outside the target module or introducing a new code path, prompt, or variable requires strong justification in this plan before implementation.
- The responsible agent may remove code from the existing codebase with lighter justification when the removal is explicitly in scope and verified by references, greps, and tests.
- If an implementation helper is needed, the responsible agent must search the codebase first for existing equivalent behavior. Equivalent behavior must be reused or locally extracted instead of duplicated.
- The responsible agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors unless explicitly listed in `Must Do`.
- If the plan and code disagree, the responsible agent must preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, the responsible agent must stop and report the blocker instead of inventing a substitute.

## Implementation Order

1. Parent establishes the focused backend test contract.
   - File: `tests/test_control_console_brain_model_routes.py`.
   - Add or update tests proving `fetch_available_models()` returns `available` for one or more sanitized IDs, `empty` for successful responses with no valid IDs, and `unavailable` for provider failures without leaking secrets.
   - Run: `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py -q`.
   - Expected before implementation: the new `empty` expectation fails because successful empty model lists currently report `available`.
2. Parent establishes the focused web-surface test contract.
   - File: `tests/test_control_console_web_surface.py`.
   - Add assertions that `Manual model ID` is absent and that the script contains the new discovery-state rendering path.
   - Run: `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py -q`.
   - Expected before implementation: manual field absence fails.
3. Parent starts the production-code subagent with this plan, the failing tests, and the Control Console production-code boundary.
4. Production-code subagent updates backend discovery semantics.
   - Modify only `src/control_console/brain_model_routes.py` and `src/control_console/app.py` as needed.
   - Preserve existing redaction and bounded model list behavior.
5. Production-code subagent updates the Services tab route editor.
   - Modify `src/control_console/static/console.js` and minimal `src/control_console/static/console.css`.
   - Remove manual input.
   - Add automatic lazy discovery for selected route.
   - Add empty, unavailable, single-model, and multi-model render states.
6. Parent updates integration/API tests while the production-code subagent works or immediately after it closes.
   - File: `tests/test_control_console_config_routes.py`.
   - Add an API test for `status: "empty"` and preserve redaction tests.
7. Parent runs focused tests and remediates only inside the approved change surface.
8. Parent updates `src/control_console/README.md` after behavior is verified by tests.
9. Parent performs rendered validation in a fresh browser context.
   - Validate Services tab using mocked or controlled provider states where practical.
   - Validate real configured endpoint when available without reading `.env`.
10. Parent runs full planned verification.
11. Parent runs independent code review and remediates findings inside approved scope.
12. Parent records execution evidence and updates plan lifecycle only after verification and review pass.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the focused test contract is established; owns production code changes only; does not edit tests unless the parent explicitly directs it; closes after planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks, and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after planned verification passes; reviews the plan, diff, and evidence; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - backend discovery test contract established.
  - Covers: implementation order step 1.
  - Verify: `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py -q`.
  - Evidence: record expected failure or baseline in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [x] Stage 2 - web-surface test contract established.
  - Covers: implementation order step 2.
  - Verify: `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py -q`.
  - Evidence: record expected failure or baseline in `Execution Evidence`.
  - Handoff: next agent starts production implementation.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [x] Stage 3 - backend discovery semantics implemented.
  - Covers: implementation order steps 3 and 4.
  - Verify: `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py tests\test_control_console_config_routes.py -q`.
  - Evidence: record changed files and test output.
  - Handoff: next agent starts UI implementation.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [x] Stage 4 - discovery-only route editor implemented.
  - Covers: implementation order step 5.
  - Verify: `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py -q`.
  - Evidence: record changed files and test output.
  - Handoff: next agent starts docs and rendered validation.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [x] Stage 5 - docs and rendered validation complete.
  - Covers: implementation order steps 8 and 9.
  - Verify: rendered Services tab validation with no browser console warnings or errors.
  - Evidence: record URL, browser method, viewport, route state exercised, screenshot path, and console health.
  - Handoff: next agent runs full verification.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [x] Stage 6 - full verification complete.
  - Covers: implementation order step 10.
  - Verify: every command in `Verification`.
  - Evidence: record command output summary.
  - Handoff: next agent starts independent code review.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [x] Stage 7 - independent code review complete.
  - Covers: implementation order step 11.
  - Verify: independent review completed and all in-scope findings remediated.
  - Evidence: record reviewer role, findings, fixes, rerun commands, residual risks, and approval status.
  - Handoff: parent may sign off and update lifecycle.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

### Static Greps

- `rg -n "Manual model ID|provider model id" src/control_console/static tests`
  - Expected: zero matches.
  - `rg` exit code 1 is acceptable for zero matches.
- `rg -n "fetch_available_models" src/control_console tests`
  - Expected: matches remain in `src/control_console/brain_model_routes.py`, `src/control_console/app.py`, and focused tests only.
- `rg -n "API_KEY|Authorization|Bearer|dotenv|\\.env" src/control_console tests development_plans/active/short_term/control_console_auto_model_discovery_picker_plan.md`
  - Expected: matches are policy text, environment variable names, or test assertions that do not include raw secret values. Any browser payload, audit payload, screenshot, or static JS exposure of raw secrets is forbidden.

### Tests

- `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py`
- `venv\Scripts\python -m pytest tests\test_control_console_config_routes.py`
- `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py`
- `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py tests\test_control_console_config_routes.py tests\test_control_console_supervisor.py tests\test_control_console_web_surface.py`

### Rendered Browser Validation

- Start the Control Console on loopback with a test operator token and without reading `.env` directly.
- Browser path:
  - Use in-app Browser when available.
  - If unavailable with a concrete error such as `Browser is not available: iab`, record that reason and use Playwright with system Chrome.
- Validate desktop Services tab at approximately `1600x1100`.
- Validate one narrower viewport at approximately `390x844` or another mobile-sized viewport.
- Exercise these route editor states:
  - loading
  - unavailable
  - empty
  - single discovered model matching current effective model
  - single discovered model different from current effective model
  - multiple discovered models
- For each exercised state, record:
  - page URL and title
  - authenticated state
  - selected route
  - visible editor state
  - console warning/error count
  - failed request count
  - screenshot path when visual state matters

### Service Smoke

- With a configured provider endpoint available, open Services tab, select Cognition route, let discovery load, apply a different discovered model when at least two provider models exist, and verify:
  - apply response is HTTP 200
  - restart is attempted and succeeds when Brain is running
  - route effective model equals the discovered model
  - Brain health remains `ok`
- If the configured provider exposes only one model, verify the single-model state and do not force a model switch.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. The parent agent must create one independent code-review subagent through the current harness's native subagent capability. If native subagents are unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, documentation, and static UI artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden fallback paths, compatibility shims, provider secret leaks, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, exact contracts, implementation order, verification gates, and acceptance criteria.
- Regression and handoff quality, including focused tests, rendered validation, static checks, execution evidence, and path-safe commands for Windows paths.

The parent agent fixes concrete findings directly only when the fix is inside the approved change surface or this review gate explicitly allows review-only fixture or documentation corrections. If a fix would cross the approved boundary or alter the contract, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `Manual model ID` and `provider model id` no longer appear in Control Console static UI or tests.
- The selected Brain route automatically discovers provider models from the configured route endpoint.
- The UI renders explicit loading, unavailable, empty, single-model, and multi-model states.
- A one-model provider result does not render as a dropdown.
- Multi-model provider results render as a dropdown containing only discovered model IDs.
- The apply button is disabled when no model, token, or thinking value is dirty.
- The apply button can apply a discovered model when a discovered model differs from the current effective model.
- Provider failures and credentials remain redacted in API responses, browser state, logs, audit entries, tests, and screenshots.
- Focused Python and web-surface tests pass.
- Rendered desktop and mobile validation passes with no unexpected browser console warnings, errors, failed app requests, or visible UI warning/error states in correctly configured flows.
- Independent code review is complete and approved.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Automatic discovery loops on every render | Add a guarded ensure function that only fetches when cache status is `not_loaded` and no request is already loading. | Browser validation plus web-surface test coverage for route editor functions. |
| Provider returns zero valid models but call succeeds | Add explicit `empty` status and render it separately from `unavailable`. | Unit and API tests for empty payload. |
| Operator loses ability to set a model when provider lacks `/models` | This is accepted for the Services tab route editor; manual entry is removed by request. Environment configuration and existing REST schema remain outside UI scope. | Static grep proves manual UI path is gone; docs state discovery-only browser behavior. |
| Free-text model value sneaks back through UI state | Remove manual input and ensure UI model dirty values come only from discovered model state. | Static tests and rendered interaction proof. |
| Single-model result still looks selectable | Render read-only single-model state. | Rendered validation screenshot and DOM check. |

## Execution Evidence

- Pre-implementation focused test baseline: added empty-provider and
  discovery-only UI assertions. Before implementation,
  `tests\test_control_console_brain_model_routes.py -q` failed because empty
  discovery reported `available`, `tests\test_control_console_config_routes.py
  -q` failed for the same API status, and
  `tests\test_control_console_web_surface.py -q` failed because the lazy
  discovery render functions were not yet present.
- Backend discovery test results: after implementation,
  `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py
  tests\test_control_console_config_routes.py
  tests\test_control_console_web_surface.py` passed, 22 tests.
- Web-surface test results: static script assertions pass for
  `ensureBrainRouteModelsLoaded`, `renderBrainModelPicker`,
  `singleBrainModelState`, and retired manual-field strings.
- Static grep results:
  `rg -n "Manual model ID|provider model id" src/control_console/static tests`
  returned exit 1 with zero matches. `rg -n "fetch_available_models"
  src/control_console tests` matched only the expected owner, app route, and
  focused tests. Secret-policy grep matched only expected policy text,
  environment variable names, redaction code, and tests.
- Rendered browser validation: in-app Browser fallback reason was
  `Browser is not available: iab`; validation used Playwright with system
  Chrome against `http://127.0.0.1:8767/`, localized Control Console page
  title, desktop `1600x1100`, mobile `390x844`. Screenshots for the real
  Services tab and controlled picker states were emitted in the execution
  thread; temporary image files were excluded from commit during workspace
  cleanup. No browser warnings, errors, failed requests, framework overlays,
  or blank-page states were observed.
- Service smoke: rendered Services tab started Brain from stopped to running,
  selected `cognition_llm`, automatically discovered 7 provider models from
  the configured endpoint, switched effective model from
  `gemma-4-31b-it-qat-uncensored-heretic-nvfp4` to
  `gemma-4-26b-a4b-it-claude-opus-distill-v2`, observed route apply HTTP 200,
  restart attempted and succeeded, Brain PID changed, route source became
  `override`, apply button returned disabled after save, and Brain health was
  `ok`.
- Full verification: `venv\Scripts\python -m pytest
  tests\test_control_console_brain_model_routes.py
  tests\test_control_console_config_routes.py
  tests\test_control_console_supervisor.py
  tests\test_control_console_web_surface.py` passed, 42 tests. Also passed:
  `node --check src\control_console\static\console.js`,
  `venv\Scripts\python -m py_compile src\control_console\brain_model_routes.py`,
  and `git diff --check`.
- Independent code review: native review subagent `019ef99d-a8dc-7fc0-a850-67197295f808`
  reported no blocking or non-blocking findings and approved. Reviewer reran
  the retired-string grep, `fetch_available_models` grep, `git diff --check`,
  22 focused tests, Python compile, and JS syntax check. Residual risk noted:
  route dirty-state mutation during render is intentional for discovered-model
  selection and should be watched in future route-editor changes.
- Final sign-off: Codex / 2026-06-25. Plan complete.
