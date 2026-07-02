# control_console_brain_model_route_config_plan

## Summary

- Goal: Add a descriptor-backed Brain service model-route configuration workflow to the Control Console Services tab, reusing the existing Services card pattern while making the Brain card span the full row and support easy configuration of all chat LLM routes.
- Plan class: large.
- Status: completed.
- Mandatory skills: `development-plan`, `control-console-web-development`, `local-llm-architecture`, `py-style`, `test-style-and-execution`, `build-web-apps:frontend-testing-debugging`.
- Overall cutover strategy: compatible for existing Control Console APIs and service lifecycle behavior; bigbang for the Brain service Services-tab card layout.
- Highest-risk areas: secret redaction, process-environment override boundaries, route inventory drift, restart failure behavior, and UI complexity for more than 10 configurable routes.
- Acceptance criteria: Operators can view all Brain chat routes, pick an available model for a selected route, save descriptor-validated overrides, restart the managed Brain process through the existing lifecycle path, reset route overrides, and verify the full Services tab against the saved rendered reference mock.

## Context

The Control Console already has a service-configuration design used by the QQ NapCat adapter. That design exposes descriptor-backed fields from the Services tab, validates operator input server-side, stores process-local overrides, audits changes, and restarts a managed service when the setting requires restart. The current built-in descriptor covers `adapter.napcat.active_groups`.

The requested change is the same class of operator workflow, applied to the Brain service model routes. Brain has more than 10 LLM routes, so a simple generic form would be hard to scan and operate. The Services tab must reuse the existing Brain service card, but that card must span the full row and contain a route matrix plus a focused editor for the selected route.

Relevant existing ownership boundaries:

- Control Console owns operator UX, authenticated configuration APIs, service lifecycle control, audit records, and redacted snapshots.
- The supervisor owns child-process command and environment construction for managed services.
- `kazusa_ai_chatbot.llm_interface` owns backend mechanics, `LLMCallConfig`, route diagnostics, and model-family detection.
- Brain response-path code owns `/chat`, queue/intake, RAG, cognition, dialog, persistence, and scheduler behavior.
- Adapters remain thin and platform-specific. This plan does not alter QQ, Discord, debug, or future adapter contracts.

Historical design inputs carried forward:

- The Control Console runtime service config plan established descriptor-backed process-local overrides, restart-aware apply semantics, audit entries, and no raw environment or secret exposure.
- The outbound adapter channel allowlist plan established the NapCat-style active channel/group configuration workflow used as the design precedent.
- The LLM interface backend abstraction and routing migration plans established route-owned `LLMCallConfig` objects and sanitized route diagnostics.
- The dynamic backend switching design is treated here as an operator-driven restart apply. In-process hot swapping through `LLInterface.invalidate_backend` is out of scope because the route constants are imported at process startup.
- The rendered Services-tab mock is saved at `development_plans/reference/designs/assets/control_console_brain_model_route_service_tab_reference.png` and is the visual reference for the final UI.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or completing this plan.
- `control-console-web-development`: load before changing Control Console routes, static HTML/CSS/JS, service-card layout, browser validation, screenshots, or stale-cache debugging.
- `local-llm-architecture`: load before changing any LLM route contract, model-selection boundary, route diagnostics use, or runtime apply behavior.
- `py-style`: load before editing Python production or test files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `build-web-apps:frontend-testing-debugging`: load before rendered UI validation. Use the Browser plugin first when available; if unavailable, record the fallback browser method and reason.

## Mandatory Rules

- Draft status means this plan is not executable. Do not implement production code from this plan until the user approves it and its status is changed to `approved` or `in_progress`.
- Do not read `.env`. Runtime code may continue to use the existing Control Console dotenv loading path, but plan execution must not inspect local secrets directly.
- Preserve the core architecture boundary: `adapter/debug client -> brain service -> queue/intake -> RAG -> cognition -> dialog -> persistence/consolidation -> scheduler/reflection`.
- Do not modify `/chat`, cognition, RAG, dialog, persistence, consolidation, scheduler, reflection, or adapter delivery code.
- Deterministic code owns validation, restart behavior, process-environment overlays, audit records, redaction, and permissions.
- LLM stages own semantic judgment. This plan must not add prompt instructions, post-processing gates, or local semantic routing.
- Do not expose API keys, raw authorization headers, dotenv contents, or full provider error bodies in API responses, audit records, browser state, logs, screenshots, or tests.
- Apply model-route changes only through descriptor-approved fields. Do not add a raw environment editor.
- Use restart-based apply for Brain route changes. Do not add in-process hot swap, background reload threads, polling, websocket push, or fallback route remapping.
- Preserve the existing NapCat service configuration behavior and tests.
- Keep route catalog behavior deterministic and bounded. Any server-side model listing must use short timeouts, response-size limits, and redacted errors.
- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution unless the user explicitly approves fallback execution.

## Must Do

- Add Brain service route configuration to the Services tab using the existing Services card pattern, with the Brain card spanning the full service-grid row.
- Show every configurable Brain chat route in a dense route matrix.
- Provide search and filters for route group, effective source, and model-family/status so more than 10 routes remain easy to operate.
- Provide a selected-route editor with available-model selection, manual model-id entry, max completion token override, thinking-enabled override, reset, and apply actions.
- Make the final Services tab match the layout anchors in `development_plans/reference/designs/assets/control_console_brain_model_route_service_tab_reference.png`: full-row Brain card, two-panel Brain layout, dense route matrix, selected-route editor, and service cards below it. Any implementation-driven deviation must be documented in `Execution Evidence`.
- Back the UI with server-side descriptor validation, process-local overrides, CSRF protection, authentication, restart-aware apply, and audit entries.
- Add a descriptor-approved environment overlay path for managed service starts so Brain route env overrides can affect the restarted child process.
- Add a safe available-model listing API that fetches models server-side from the selected route's effective base URL and API key without exposing secrets.
- Preserve the existing generic service config APIs for current consumers.
- Add focused backend tests for route catalog projection, validation, environment overlay, available-model listing, restart behavior, audit behavior, and error handling.
- Add static surface and browser-render validation for the full-row Brain card, full Services tab, route matrix, selected-route editor, and mobile layout.
- Update Control Console documentation with the new Brain model-route workflow, safety constraints, and restart semantics.

## Deferred

- Do not add in-process Brain model hot swap.
- Do not add persistent database-backed Control Console settings.
- Do not add user-specific, character-specific, conversation-specific, or channel-specific model routing.
- Do not expose or edit API keys, base URLs, embedding routes, provider credentials, dotenv contents, or arbitrary environment variables in this UI.
- Do not create compatibility shims, alias modules, alternate route names, or fallback mappers for old call shapes.
- Do not add new LLM calls, prompt changes, RAG changes, cognition changes, or dialog behavior changes.
- Do not add websocket updates, broad polling, provider health monitoring, or long-running background discovery jobs.
- Do not redesign the entire Services tab outside the Brain full-row card and the minimum layout adjustments needed around it.
- Do not change adapter service cards beyond keeping them visually aligned below the full-row Brain card.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Existing service config API | compatible | Preserve `GET`, `PUT`, and reset behavior for existing descriptor-backed services. |
| NapCat active groups | compatible | Keep current validation, overrides, restart semantics, and tests intact. |
| Brain Services card layout | bigbang | Replace the compact Brain service card on the Services tab with a full-row Brain card containing runtime summary and model-route configuration. |
| Brain route apply semantics | compatible | Apply route overrides only when the operator saves changes and restarts the managed Brain service through the existing lifecycle path. |
| Supervisor process launch | compatible | Keep command resolution behavior and add descriptor-approved environment overlays for managed child processes. |
| Static UI | bigbang | Ship the new Services tab layout directly. Do not preserve a second legacy Brain card layout. |
| LLM runtime code | compatible | Do not change route execution code. The restarted Brain process reads the existing environment-backed `LLMCallConfig` constants. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, replace the old local UI behavior directly instead of preserving two UI paths.
- If an area is `compatible`, preserve only the compatibility surfaces explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

The Services tab shows the Brain service as a full-row card at the top of the service grid. The left side of the card shows the existing Brain runtime summary and lifecycle controls. The right side shows a model-route matrix with all supported chat LLM routes grouped by role, with compact badges for route state, effective model, model family, thinking mode, and source.

Selecting a route opens a focused editor inside the same full-row Brain card. The editor shows:

- the route label, route key, group, and required/fallback-backed status
- current effective model, default model, and override source
- a dropdown of available provider model IDs when the route provider supports model listing
- a manual model-id input for explicit operator overrides
- max completion token override
- thinking-enabled override
- reset selected route, refresh model list, apply selected route, and apply all dirty route changes actions

Saving route changes writes descriptor-validated process-local overrides for the Brain service. If the Brain process is managed and running, the Control Console restarts it through the existing service lifecycle path. If the Brain service is stopped, the next start uses the override environment. If the Brain service is unmanaged, the UI shows that changes are saved but cannot be applied to the running process by the console.

The Brain runtime does not gain new response-path logic. The restarted process reads the same `*_MODEL`, `*_MAX_COMPLETION_TOKENS`, and `*_THINKING_ENABLED` environment variables that currently define route configuration.

The final Services tab should match the saved reference mock in layout intent: Brain is a full-row card, runtime state remains in the left panel, model routes remain in the right panel, the route matrix stays dense and scannable, the selected-route editor sits below the matrix, and adapter cards remain below Brain.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Configuration ownership | Keep configuration in Control Console descriptors and process-local override store. | Matches the NapCat design and keeps validation, audit, and restart behavior centralized. |
| Brain apply mode | Use restart-based apply, not hot swap. | Route configs are imported as environment-backed constants; restart is bounded and inspectable. |
| Environment override path | Add descriptor-approved service environment overlays to the supervisor. | Brain model route changes require child-process env overrides, while NapCat command overlays must keep working. |
| Route UI shape | Use a route matrix plus selected-route editor. | A generic 39-field form is not usable for more than 10 LLM routes. |
| Visual reference | Treat `development_plans/reference/designs/assets/control_console_brain_model_route_service_tab_reference.png` as the implementation reference. | The user accepted the proposed full-page mock and asked for the final product to preserve its visible layout intent. |
| Available model picker | Fetch model IDs server-side per selected route. | Browser must not receive provider credentials; server can redact and bound provider responses. |
| Manual model entry | Keep a manual model-id input beside the picker. | Existing route config supports arbitrary provider model IDs and some providers do not expose `/models`. |
| Editable fields | Support `model`, `max_completion_tokens`, and `thinking_enabled` per chat route. | These are the route-level env values already present in the LLM config contract. |
| Non-editable fields | Do not expose API key, base URL, embedding model, or backend kind editing. | This keeps the first integration focused on model choice and avoids secret/raw environment editing. |
| Route catalog source | Keep the Control Console route catalog explicit and validated against current route diagnostics tests. | The UI needs labels and groups without importing Brain response-path behavior. |
| Audit detail | Audit route names, changed field names, redacted old/new summaries, apply result, and restart result. | Operators need accountability without leaking provider secrets. |

## Contracts And Data Shapes

### Route Catalog

Create a Control Console-owned route catalog module with a public route descriptor shape:

```python
@dataclass(frozen=True)
class BrainModelRouteDescriptor:
    route_key: str
    env_prefix: str
    label: str
    group: str
    required: bool
    fallback_backed: bool
    editable_fields: tuple[str, ...]
```

The catalog must include these chat routes:

- `RELEVANCE_AGENT_LLM`
- `VISION_DESCRIPTOR_LLM`
- `MSG_DECONTEXTUALIZER_LLM`
- `RAG_PLANNER_LLM`
- `RAG_SUBAGENT_LLM`
- `WEB_SEARCH_LLM`
- `COGNITION_LLM`
- `BOUNDARY_CORE_LLM`
- `DIALOG_GENERATOR_LLM`
- `CONSOLIDATION_LLM`
- `JSON_REPAIR_LLM`
- `BACKGROUND_ARTIFACT_LLM`
- `BACKGROUND_WORK_LLM`

Do not include embedding route editing in this plan.

### Descriptor Fields

The Brain service descriptor must expose descriptor-backed fields with lowercase keys and uppercase environment bindings:

```python
{
    "cognition_llm_model": {
        "default_env": "COGNITION_LLM_MODEL",
        "value_type": "string",
        "restart_required": True,
    },
    "cognition_llm_max_completion_tokens": {
        "default_env": "COGNITION_LLM_MAX_COMPLETION_TOKENS",
        "value_type": "integer",
        "restart_required": True,
    },
    "cognition_llm_thinking_enabled": {
        "default_env": "COGNITION_LLM_THINKING_ENABLED",
        "value_type": "boolean",
        "restart_required": True,
    },
}
```

Apply the same field pattern to every catalog route. Extend service-config field-count validation enough to support the Brain descriptor while keeping a finite maximum. The maximum must cover the 39 planned Brain fields and leave bounded room for descriptor metadata.

### Environment Overlay

Extend the service configuration registry with a public environment rendering entrypoint:

```python
def render_environment_overlay(
    self,
    service_id: str,
    base_environment: Mapping[str, str],
    overrides: Mapping[str, Any],
) -> dict[str, str]:
    ...
```

The renderer must:

- return only descriptor-approved environment variable names
- stringify values according to descriptor type
- omit unset optional values instead of writing empty strings
- never include sensitive values not owned by the descriptor
- raise validation errors before a process restart is attempted

Extend the process supervisor with an optional environment resolver used only for managed child process creation:

```python
EnvironmentResolver = Callable[[str], Mapping[str, str]]
```

The supervisor must merge the current server process environment with the resolver output for the child process. The child process environment must not be logged in full.

### Model Routes API

Add Control Console API routes under the existing authenticated and CSRF-protected app:

```text
GET  /api/services/brain/model-routes
PUT  /api/services/brain/model-routes/{route_key}
POST /api/services/brain/model-routes/{route_key}/reset
GET  /api/services/brain/model-routes/{route_key}/available-models
```

`GET /api/services/brain/model-routes` returns:

```json
{
  "service_id": "brain",
  "version": "opaque-version",
  "service_state": {
    "status": "running",
    "managed": true,
    "restart_required": false
  },
  "routes": [
    {
      "route_key": "cognition_llm",
      "env_prefix": "COGNITION_LLM",
      "label": "Cognition",
      "group": "Core response",
      "required": true,
      "fallback_backed": false,
      "effective": {
        "model": "model-id",
        "max_completion_tokens": 8192,
        "thinking_enabled": true,
        "source": "environment"
      },
      "default": {
        "model": "model-id",
        "max_completion_tokens": 8192,
        "thinking_enabled": true
      },
      "diagnostics": {
        "backend_kind": "openai_compatible_chat",
        "base_url_label": "http://localhost:11434",
        "model_family": "qwen",
        "thinking_strategy": "enabled"
      },
      "available_models": {
        "status": "not_loaded",
        "count": 0
      }
    }
  ]
}
```

`PUT /api/services/brain/model-routes/{route_key}` accepts:

```json
{
  "expected_version": "opaque-version",
  "values": {
    "model": "model-id",
    "max_completion_tokens": 8192,
    "thinking_enabled": true
  }
}
```

The server maps the route payload to descriptor field overrides, validates all values, writes the override store, audits the change, and invokes the same restart helper used by the existing service config route.

`POST /api/services/brain/model-routes/{route_key}/reset` clears only that route's descriptor-backed override fields and restarts when needed.

`GET /api/services/brain/model-routes/{route_key}/available-models` returns:

```json
{
  "route_key": "cognition_llm",
  "status": "available",
  "models": [
    {
      "id": "model-id",
      "family": "qwen"
    }
  ],
  "message": null
}
```

Error responses must be redacted:

```json
{
  "route_key": "cognition_llm",
  "status": "unavailable",
  "models": [],
  "message": "Provider model list unavailable."
}
```

### Provider Model Listing

The server-side listing implementation must:

- use the route's effective base URL and API key from the server environment/config path
- call only the provider model-list endpoint needed for OpenAI-compatible providers
- use a timeout of 5 seconds or less
- cap returned models at 200 items
- cap model ID length at 200 characters
- drop duplicate model IDs
- sort model IDs case-insensitively
- avoid logging raw provider responses
- return a redacted unavailable status for network, auth, JSON, or shape failures

### UI State

`console.js` must keep Brain route UI state local to the Services tab:

```javascript
{
  brainRoutes: [],
  selectedBrainRouteKey: "cognition_llm",
  brainRouteFilters: {
    search: "",
    group: "all",
    source: "all",
    family: "all"
  },
  dirtyBrainRouteValues: {
    cognition_llm: {
      model: "model-id",
      max_completion_tokens: 8192,
      thinking_enabled: true
    }
  },
  availableModelCache: {
    cognition_llm: {
      status: "available",
      models: []
    }
  }
}
```

State must be discarded on page reload. The server-side override store remains the source of truth.

## LLM Call And Context Budget

This plan adds no new LLM completion calls, no prompt changes, and no response-path context. The Brain response-path LLM call count remains unchanged.

Before:

- Brain response path uses the existing route-specific LLM calls according to current route configuration.
- Control Console Services tab does not call provider model-list endpoints for Brain routes.

After:

- Brain response path uses the same route-specific LLM calls after the managed Brain process restarts with updated environment values.
- Control Console may make one server-side provider model-list HTTP request when an operator explicitly opens or refreshes the available-model picker for a selected route.
- Provider model listing is not an LLM generation call and must not enter prompt, RAG, cognition, memory, dialog, or scheduler context.
- No context-window budget changes are introduced.

## Change Surface

### Delete

- No production files are planned for deletion.

### Modify

- `src/control_console/service_config.py`
  - Add Brain service descriptor construction.
  - Add descriptor-backed environment overlay rendering.
  - Increase bounded descriptor field-count validation enough to support the Brain route descriptor.
  - Keep existing command rendering and NapCat behavior intact.
- `src/control_console/supervisor.py`
  - Add optional environment resolver support for managed child process start.
  - Ensure process logging and fingerprints remain redacted and do not expose raw environment values.
- `src/control_console/app.py`
  - Register Brain route model APIs.
  - Factor shared restart/apply logic so generic config and route-specific APIs share lifecycle behavior.
  - Include Brain route summaries in service snapshots without leaking secrets.
- `src/control_console/contracts.py`
  - Add request and response models for Brain route snapshots, route apply, route reset, and available model list responses.
- `src/control_console/static/index.html`
  - Update Services tab structure so the Brain service card supports full-row runtime and route panels.
- `src/control_console/static/console.js`
  - Render the full-row Brain card, route matrix, filters, selected-route editor, model picker, reset/apply actions, dirty-state indicators, and error states.
- `src/control_console/static/console.css`
  - Add responsive layout for the full-row Brain card and dense route matrix.
  - Preserve existing design tokens and avoid card-in-card layout.
- `src/control_console/README.md`
  - Document Brain model route configuration, restart semantics, available-model listing, redaction, and scope limits.

### Create

- `src/control_console/brain_model_routes.py`
  - Public entrypoints for route catalog, route projection, descriptor field mapping, provider model listing, and redacted model-family labels.
- `tests/test_control_console_brain_model_routes.py`
  - Focused tests for catalog completeness, field mapping, projection, validation, available-model listing, redaction, and route reset behavior.
- New or expanded browser-render validation artifact during execution.
  - Store the final screenshot path in `Execution Evidence`.
- `development_plans/reference/designs/assets/control_console_brain_model_route_service_tab_reference.png`
  - Saved rendered Services-tab mock used as the implementation visual reference.

### Keep

- `src/kazusa_ai_chatbot/**`
  - Keep runtime Brain, cognition, RAG, dialog, persistence, scheduler, and LLM call execution code unchanged.
- `src/adapters/**`
  - Keep adapter delivery and QQ/Discord/debug event normalization unchanged.
- `.env`
  - Do not inspect or edit.
- Existing service cards below Brain
  - Keep their current lifecycle controls and descriptor config behavior.

## Overdesign Guardrail

- Actual problem: Operators need a usable Services-tab workflow to configure more than 10 Brain LLM routes and pick available models without editing environment files manually.
- Minimal change: Extend the existing descriptor-backed service config system with Brain route fields, a safe child-process environment overlay, a route-matrix projection API, and a full-row Brain card UI.
- Ownership boundaries: Control Console owns operator UI, validation, redaction, audit, and restart; supervisor owns child-process environment construction; `llm_interface` owns route diagnostics and backend identity; Brain response-path modules keep semantic behavior unchanged.
- Rejected complexity: no in-process hot swap, no raw environment editor, no credential UI, no persistent settings database, no per-character route policy, no adapter-specific route behavior, no prompt changes, no websocket updates, no provider health dashboard, and no visual redesign away from the saved reference mock.
- Evidence threshold: Add hot swap, persistence, per-character routing, or provider health only after a separately approved plan cites a concrete operator failure, latency requirement, or deployment workflow that restart-based apply cannot satisfy.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside `src/control_console/**`, `tests/test_control_console*.py`, and `src/control_console/README.md` as high-scrutiny changes requiring explicit plan approval.
- The responsible agent must not edit `src/kazusa_ai_chatbot/**` or `src/adapters/**` for this plan.
- The responsible agent may use existing public helpers from `kazusa_ai_chatbot.llm_interface` for model-family detection or route diagnostics if importing them does not initialize Brain runtime behavior or require secrets.
- If equivalent Control Console validation, restart, audit, or snapshot behavior already exists, reuse or factor it instead of duplicating it.
- The responsible agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, the responsible agent must preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, the responsible agent must stop and report the blocker instead of inventing a substitute.

## Implementation Order

1. Establish backend route-catalog tests.
   - File: `tests/test_control_console_brain_model_routes.py`.
   - Add tests proving the catalog contains exactly the 13 chat routes named in this plan, each route maps to `model`, `max_completion_tokens`, and `thinking_enabled` descriptor fields, and embedding routes are absent.
   - Run `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py -q`.
   - Expected before implementation: fails because `src/control_console/brain_model_routes.py` does not exist.

2. Establish descriptor and environment overlay tests.
   - Files: `tests/test_control_console_service_config.py`, `tests/test_control_console_brain_model_routes.py`.
   - Add tests proving Brain descriptor registration, bounded field-count validation, model-id validation, integer validation, boolean validation, environment overlay rendering, and NapCat regression behavior.
   - Run the focused tests and record the baseline failure.

3. Establish API integration tests.
   - File: `tests/test_control_console_config_routes.py` or a new focused API test file if the existing file becomes too broad.
   - Add tests for `GET /api/services/brain/model-routes`, route `PUT`, route reset, CSRF/auth failure, stale version failure, restart success, restart failure, stopped-service apply, unmanaged-service behavior, and audit redaction.
   - Run the focused tests and record the baseline failure.

4. Establish available-model listing tests.
   - File: `tests/test_control_console_brain_model_routes.py`.
   - Use a fake HTTP transport or patched HTTP client to verify model list success, timeout, invalid JSON, duplicate IDs, overlong IDs, response cap, auth header redaction, and provider error redaction.
   - Run the focused tests and record the baseline failure.

5. Start the production-code subagent.
   - Provide this approved plan, mandatory skills, focused test failures, and the production-code ownership boundary.
   - Production-code subagent owns only `src/control_console/**` production changes listed in `Change Surface`.
   - Parent continues test, static check, and browser-validation preparation while the production-code subagent edits production code.

6. Implement the backend route catalog and descriptor mapping.
   - File: `src/control_console/brain_model_routes.py`.
   - Implement catalog descriptors, field-key mapping, route grouping, validation helpers, route snapshot projection, and model-family labeling.
   - Do not import Brain runtime modules that execute response-path behavior.

7. Implement service-config environment overlays.
   - File: `src/control_console/service_config.py`.
   - Add environment overlay renderer and Brain descriptor registration.
   - Preserve current command renderer API and NapCat behavior.

8. Implement supervisor environment resolver.
   - File: `src/control_console/supervisor.py`.
   - Add optional environment resolver injection and child-process environment merge.
   - Keep environment values out of logs, fingerprints, and public snapshots.

9. Implement route APIs and shared restart apply.
   - Files: `src/control_console/app.py`, `src/control_console/contracts.py`.
   - Add model-route endpoints.
   - Factor the existing service config apply/restart logic only as needed so both generic config and route-specific APIs use the same lifecycle semantics.
   - Add redacted audit records.

10. Run backend focused tests.
    - Command: `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py tests\test_control_console_service_config.py tests\test_control_console_config_routes.py -q`.
    - Fix only issues inside the approved change surface.

11. Establish static web-surface tests.
    - File: `tests/test_control_console_web_surface.py`.
    - Add or update tests proving the Services tab includes the full-row Brain service container, route matrix mount points, selected-route editor controls, available-model picker controls, and no hard-coded per-route route list in static HTML.
    - Run the focused test and record the baseline or current failure.

12. Implement Services tab UI.
    - Files: `src/control_console/static/index.html`, `src/control_console/static/console.js`, `src/control_console/static/console.css`.
    - Render Brain as a full-row service card with runtime panel and model route panel.
    - Add route matrix filters, selected route editor, available model refresh, dirty-state handling, apply/reset actions, restart-state messaging, unmanaged-service messaging, loading state, and redacted error display.
    - Keep the layout visually aligned with `development_plans/reference/designs/assets/control_console_brain_model_route_service_tab_reference.png`.
    - Keep adapter service cards visually aligned below Brain.

13. Run static surface tests.
    - Command: `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py -q`.
    - Fix only issues inside the approved change surface.

14. Update Control Console documentation.
    - File: `src/control_console/README.md`.
    - Document route configuration scope, restart apply semantics, available-model fetching, redaction, unmanaged service behavior, and out-of-scope hot swap.

15. Run regression verification.
    - Run all commands listed in `Verification`.
    - Record exact command output summaries in `Execution Evidence`.

16. Run rendered UI validation.
    - Start the local Control Console test server using the repo's existing method.
    - Use Browser plugin first. If unavailable, use Playwright with the local installed browser and record the fallback reason.
    - Validate desktop and mobile Services tab screenshots.
    - Compare the desktop screenshot against `development_plans/reference/designs/assets/control_console_brain_model_route_service_tab_reference.png` and record any intentional deviations.
    - Attach or record the final screenshot artifact path in `Execution Evidence`.

17. Run independent code review.
    - Start one independent code-review subagent after verification passes.
    - Provide this plan, diff, verification evidence, and screenshot artifact.
    - Parent fixes approved-scope findings and reruns affected verification.

18. Complete lifecycle update only after approval.
    - Update progress checklist and execution evidence.
    - Change plan status to `completed` only after implementation, verification, and independent code review pass.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the focused test contract is established; owns production code changes only; does not edit tests unless the parent explicitly directs it; closes after planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks, and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after planned verification passes; reviews the plan, diff, evidence, and screenshot; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - backend test contract established.
  - Covers: implementation steps 1-4.
  - Verify: focused tests are added and fail for missing route catalog, descriptor, API, or available-model behavior before production implementation.
  - Evidence: record failing commands and failure reasons in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-06-24` after evidence is recorded.

- [x] Stage 2 - backend production implementation complete.
  - Covers: implementation steps 5-10.
  - Verify: `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py tests\test_control_console_service_config.py tests\test_control_console_config_routes.py -q`.
  - Evidence: record changed production files, test output summary, and any residual backend risks.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-06-24` after evidence is recorded.

- [x] Stage 3 - Services tab UI implementation complete.
  - Covers: implementation steps 11-13.
  - Verify: `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py -q`.
  - Evidence: record static file changes and test output summary.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-06-24` after evidence is recorded.

- [x] Stage 4 - documentation and regression verification complete.
  - Covers: implementation steps 14-15.
  - Verify: all commands in `Verification` pass or have allowed documented exceptions.
  - Evidence: record command output summaries and static grep results.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-06-24` after evidence is recorded.

- [x] Stage 5 - rendered UI validation complete.
  - Covers: implementation step 16.
  - Verify: desktop and mobile screenshots show the full-row Brain service card, all route-matrix controls, selected-route editor, and service cards below Brain without overlap; desktop screenshot matches the saved reference mock's required layout anchors.
  - Evidence: record server URL, browser method, screenshot artifact paths, reference comparison notes, console errors, and viewport sizes.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `Codex/2026-06-24` after evidence is recorded.

- [x] Stage 6 - independent code review complete.
  - Covers: implementation step 17.
  - Verify: independent code-review subagent reviews plan alignment, full diff, tests, static checks, screenshot evidence, redaction behavior, and scope boundaries.
  - Evidence: record findings, fixes, rerun commands, residual risks, and approval status.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: `Codex/2026-06-24` after evidence is recorded; native subagent review was waived by explicit user instruction to execute without subagent, and a parent-agent self-review plus browser validation was performed.

- [x] Stage 7 - lifecycle completion recorded.
  - Covers: implementation step 18.
  - Verify: status, registry row, checklist, and execution evidence are consistent.
  - Evidence: record final changed-file list and completion status.
  - Handoff: plan can be archived after project convention is followed.
  - Sign-off: `Codex/2026-06-24` after evidence is recorded.

## Verification

### Static Greps

- `rg -n "API_KEY|Authorization|Bearer|dotenv|\\.env" src/control_console tests/test_control_console*.py`
  - Expected: no new response models, screenshots, static JS state, audit strings, or test assertions expose raw secrets. Matches in existing dotenv-loading code or redaction tests are allowed only when they do not print values.
- `rg -n "COGNITION_LLM|DIALOG_GENERATOR_LLM|RAG_PLANNER_LLM|BACKGROUND_WORK_LLM" src/control_console/static`
  - Expected: zero matches. Static route inventory must come from the API, not hard-coded static HTML or JS.
- `rg -n "invalidate_backend|hot.?swap|websocket|poll" src/control_console`
  - Expected: zero matches for new hot-swap, websocket, or polling implementation. Existing unrelated text requires review before sign-off.
- `rg -n "brain_model_routes|model-routes" src/kazusa_ai_chatbot src/adapters`
  - Expected: zero matches. Brain runtime and adapters must not depend on Control Console route UI code.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py -q`
- `venv\Scripts\python -m pytest tests\test_control_console_service_config.py -q`
- `venv\Scripts\python -m pytest tests\test_control_console_config_routes.py -q`
- `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py -q`

### Regression Tests

- `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py tests\test_control_console_service_config.py tests\test_control_console_config_routes.py tests\test_control_console_web_surface.py -q`
- `venv\Scripts\python -m pytest tests\test_llm_interface_route_report.py tests\test_llm_interface_contracts.py -q`

### Static Compile

- `venv\Scripts\python -m py_compile src\control_console\brain_model_routes.py src\control_console\service_config.py src\control_console\supervisor.py src\control_console\app.py src\control_console\contracts.py`

### Browser Validation

- Start the Control Console locally through the existing documented development path.
- Open the Services tab.
- Use `development_plans/reference/designs/assets/control_console_brain_model_route_service_tab_reference.png` as the visual reference for desktop layout.
- Verify desktop viewport:
  - Brain card spans the full service-grid row.
  - Runtime panel and model-route panel are visible in one card.
  - All 13 chat routes are reachable through the matrix and filters.
  - Selecting a route updates the focused editor.
  - Available-model refresh has loading, success, and unavailable states.
  - Apply/reset controls show disabled, dirty, success, and error states without layout shift.
  - Adapter service cards remain below the Brain card.
- Verify mobile viewport:
  - Brain runtime and route panels stack without text overlap.
  - Route filters and selected-route editor remain usable.
  - Buttons and badges stay inside their containers.
- Capture final desktop and mobile screenshots, compare the desktop screenshot to the saved reference mock, and record paths plus a short mismatch ledger in `Execution Evidence`.

## Independent Plan Review

Review performed on 2026-06-24 before approval or implementation.

Findings addressed:

- Blocking: the rendered Services-tab mock existed only in a temporary path, so future implementation would not have a stable design reference. Fixed by saving `development_plans/reference/designs/assets/control_console_brain_model_route_service_tab_reference.png`.
- Blocking: the plan required screenshots but did not require fidelity to the accepted mock's layout anchors. Fixed by adding reference-fidelity requirements to `Must Do`, `Target State`, `Design Decisions`, implementation step 12, rendered validation, Stage 5, and acceptance criteria.
- Non-blocking: the plan already preserved the Control Console boundary and restart-based LLM route apply semantics. No code-scope expansion was needed.

Plan review status: no unresolved review blockers remain. User approved execution
on 2026-06-24 and explicitly requested single-agent execution without
subagents.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. The parent agent must create one independent code-review subagent through the current harness's native subagent capability. If native subagents are unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, documentation, HTML, CSS, and JavaScript file.
- Control Console ownership boundaries, including no Brain runtime, adapter, RAG, cognition, dialog, persistence, scheduler, or prompt changes.
- Service config descriptor design, validation, environment overlay redaction, restart apply behavior, unmanaged-service behavior, and audit quality.
- Available-model listing safety, including timeout, size caps, model-id validation, duplicate handling, and redacted failures.
- UI usability for more than 10 routes, full-row Brain card behavior, no card-in-card layout, responsive layout, and screenshot evidence.
- Alignment with `Must Do`, `Deferred`, `Cutover Policy`, `Contracts And Data Shapes`, `Change Surface`, implementation order, verification gates, and acceptance criteria.

The parent agent fixes concrete findings directly only when the fix is inside the approved change surface or this review gate explicitly allows review-only fixture/documentation corrections. If a fix would cross the approved boundary or alter the contract, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- The Services tab renders the Brain service card across the full service-grid row.
- The rendered desktop Services tab matches the layout anchors in `development_plans/reference/designs/assets/control_console_brain_model_route_service_tab_reference.png`, with any intentional deviations recorded in `Execution Evidence`.
- The Brain card shows a runtime panel and a model-route panel without nested cards or layout overlap.
- All 13 supported chat LLM routes are visible or reachable through route matrix search/filter controls.
- Operators can select a route, refresh available models, choose a model from the returned list, manually enter a model ID, adjust max completion tokens, adjust thinking-enabled state, apply changes, and reset the selected route.
- Route changes are validated server-side through descriptor-backed fields and stored as process-local overrides.
- Managed Brain service restarts use descriptor-approved environment overlays and do not log or expose raw environment values.
- Restart success, restart failure, stopped-service apply, and unmanaged-service behavior are visible to the operator and covered by tests.
- API keys, raw authorization headers, dotenv contents, and full provider error bodies are never exposed in API responses, audit entries, browser state, tests, or screenshots.
- Existing NapCat active-group configuration behavior remains passing.
- Control Console documentation describes the Brain model-route workflow, restart semantics, and out-of-scope hot swap.
- All verification commands pass or have explicitly allowed exceptions recorded in `Execution Evidence`.
- Desktop and mobile rendered screenshot artifacts are recorded.
- Independent code review passes with no unresolved blocking findings.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Secret leakage through model-listing or audit payloads | Keep provider calls server-side, redact failures, do not expose API keys, add static greps and redaction tests. | Static greps, available-model tests, API response assertions, code review. |
| Route catalog drift from `llm_interface` route definitions | Keep explicit catalog tests aligned with `test_llm_interface_route_report.py` and fail when route inventory changes. | Catalog completeness tests and LLM route-report regression tests. |
| Restart failure leaves operator unsure which model is active | Reuse existing apply-failed state and show saved override versus running process status. | API restart-failure test and browser validation. |
| Generic 39-field descriptor overwhelms UI | Use route projection API and selected-route editor instead of generic form rendering. | Web-surface tests and rendered screenshot validation. |
| Environment overlay changes break existing service starts | Make environment resolver optional and preserve command resolver behavior. | Existing supervisor/service config tests and NapCat regression tests. |
| Provider `/models` endpoint is unavailable or nonstandard | Return redacted unavailable status and keep manual model entry available. | Fake transport tests for error and manual-entry UI states. |
| UI becomes cramped on mobile | Use responsive full-row layout with stable dimensions and browser screenshot validation. | Mobile screenshot validation and web-surface tests. |

## Execution Evidence

Record evidence here during execution.

- Focused backend test baseline:
  - `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py -q` initially failed 3/4 because `control_console.brain_model_routes` was missing.
  - Route API tests initially failed with 404/missing module.
  - Supervisor env-overlay test initially failed because `ProcessSupervisor` did not accept `environment_resolver`.
  - Static web-surface test initially failed because Brain route UI functions/classes were missing from `console.js`/`console.css`.
- Execution mode: User explicitly requested execution without subagents on
  2026-06-24. Parent agent is executing the plan directly.
- Backend implementation verification:
  - Added `src/control_console/brain_model_routes.py`.
  - Updated `service_config.py`, `supervisor.py`, `contracts.py`, and `app.py` for Brain route descriptors, selected-route APIs, server-side model listing, and descriptor-approved process env overlays.
  - `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py tests\test_control_console_config_routes.py::test_brain_model_route_api_applies_and_resets_selected_route tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q`: 6 passed after final source-label change.
- Static UI test baseline:
  - `tests/test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs` initially failed because the static script did not contain `refreshBrainModelRoutes`, `renderBrainServiceCard`, route matrix/editor functions, or Brain card CSS classes.
- UI implementation verification:
  - Updated `src/control_console/static/console.js` and `console.css`.
  - `venv\Scripts\python -m pytest tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q`: passed.
- Static grep results:
  - `rg -n "COGNITION_LLM|DIALOG_GENERATOR_LLM|RAG_PLANNER_LLM|BACKGROUND_WORK_LLM" src\control_console\static`: zero matches.
  - `rg -n "brain_model_routes|model-routes" src\kazusa_ai_chatbot src\adapters`: zero matches.
  - `rg -n "invalidate_backend|hot.?swap|websocket|poll" src\control_console`: one README policy line only; no implementation.
  - `rg -n "API_KEY|Authorization|Bearer|dotenv|\.env" ...`: reviewed matches are existing dotenv settings, route env-name construction, server-side provider Authorization header construction, README policy text, and redaction/test fixtures; no raw secret values are exposed to browser state, audit payloads, screenshots, or static JS.
- Regression test results:
  - `venv\Scripts\python -m pytest tests\test_control_console_brain_model_routes.py tests\test_control_console_service_config.py tests\test_control_console_config_routes.py tests\test_control_console_web_surface.py -q`: 26 passed.
  - `venv\Scripts\python -m pytest tests\test_llm_interface_route_report.py tests\test_llm_interface_contracts.py -q`: 20 passed.
  - `venv\Scripts\python -m py_compile src\control_console\brain_model_routes.py src\control_console\service_config.py src\control_console\supervisor.py src\control_console\app.py src\control_console\contracts.py`: passed.
- Browser validation:
  - Browser plugin attempt failed: `Browser is not available: iab`.
  - Fallback method: Playwright with system Chrome executable `C:\Program Files\Google\Chrome\Application\chrome.exe`.
  - Temporary URL: `http://127.0.0.1:8767/`, stopped after validation.
  - Real rendered app identity: title `杏山千纱 Control Console`, Services tab loaded after token login.
  - Route count: 13 route tiles.
  - States captured: default, provider model-list loading, provider model-list success, provider model-list unavailable, dirty edit/apply-enabled, saved override, reset, and mobile layout.
  - Browser console warnings/errors: none.
  - Layout checks: zero-sized panels `[]`, overflowing controls `[]`, horizontal overflow `0`.
  - Screenshot artifacts:
    - `C:\Users\Ran Bao\AppData\Local\Temp\kazusa-control-service-tab-shots\services-default-desktop.png`
    - `C:\Users\Ran Bao\AppData\Local\Temp\kazusa-control-service-tab-shots\services-model-loading.png`
    - `C:\Users\Ran Bao\AppData\Local\Temp\kazusa-control-service-tab-shots\services-model-success.png`
    - `C:\Users\Ran Bao\AppData\Local\Temp\kazusa-control-service-tab-shots\services-model-unavailable.png`
    - `C:\Users\Ran Bao\AppData\Local\Temp\kazusa-control-service-tab-shots\services-dirty-edit.png`
    - `C:\Users\Ran Bao\AppData\Local\Temp\kazusa-control-service-tab-shots\services-applied-override.png`
    - `C:\Users\Ran Bao\AppData\Local\Temp\kazusa-control-service-tab-shots\services-reset.png`
    - `C:\Users\Ran Bao\AppData\Local\Temp\kazusa-control-service-tab-shots\services-mobile.png`
- Reference mock comparison:
  - Matches required layout anchors: full-row Brain card, left runtime panel, right route matrix/editor, adapter cards below, desktop two-panel layout, and mobile stacked layout.
  - Intentional implementation difference: compact route source badges show `default`/`override` instead of raw env names; raw per-field env source remains visible in the selected editor's source details.
- Independent code review:
  - Native subagent review was skipped because the user explicitly required execution without subagent.
  - Parent-agent self-review found and fixed one rendered layout issue: selected-route form controls could overflow at desktop width. Fixed by tightening `.brain-route-form` grid constraints and button wrapping, then reran browser validation.
- Residual risks:
  - Control Console overrides remain process-local by design and disappear on console restart.
  - Available-model listing depends on provider `/models`; unavailable provider state is handled by manual model entry.
