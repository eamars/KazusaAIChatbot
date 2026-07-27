# cognition core v2 stage llm endpoint routing plan

## Summary

- Goal: replace the four aggregate Cognition Core V2 model bindings with
  explicit stage-owned `COGNITION_LLM_<STAGE>_*` environment bundles, populate
  the current workspace `.env`, expose every new route through the existing
  Control Console model-route editor, and apply the accepted stage assignment.
- Plan class: high_risk_migration.
- Status: completed.
- Mandatory skills: development-plan, local-llm-architecture, py-style,
  cjk-safety, test-style-and-execution, and
  control-console-web-development.
- Overall cutover strategy: bigbang.
- Highest-risk areas: required environment cutover, stage-to-route mismatches,
  trace identity drift, stale route catalogs, local secret handling, and
  startup regressions.
- Acceptance criteria: the workspace imports and starts from its configured
  `.env`; every Core V2 LLM call and retry uses its exact stage route; the
  Control Console handles each route through the existing workflow; the old
  boundary route is absent; model, endpoint, and model-class choices remain
  environment-owned; deterministic, runtime, browser, and review gates pass.

## Context

The current production connector exposes four aggregate bindings through
`CognitionCoreServicesV2`:

1. `appraisal_config`
2. `goal_cognition_config`
3. `collapse_config`
4. `action_selection_config`

`appraisal_config`, `goal_cognition_config`, and `collapse_config` use the
generic `COGNITION_LLM` route. `action_selection_config` uses
`BOUNDARY_CORE_LLM`, even though that field currently serves four distinct
semantic owners: required-selection verification, action planning, action
authorization, and resolver authorization.

The completed model-assignment evaluation is archived at
[cognition_core_v2_model_assignment_quality_evaluation_plan.md](../../archive/completed/short_term/cognition_core_v2_model_assignment_quality_evaluation_plan.md).
The user accepted that experiment as limited historical evidence and rejected
its aggregate all-one-route result as production assignment authority.

The evidence that remains usable for this plan is narrower:

- 384 Core V2 runs completed with 3,308 captured configured-factor calls and
  no unrecovered provider, parse, or contract failures.
- The current `COGNITION_LLM` route passed all 24 observations for the
  user-defined allocation-sensitive goal criterion.
- The current `BOUNDARY_CORE_LLM` route passed 10 of 24 observations for the
  same goal criterion.
- Appraisal calls are the dominant parallel workload.
- Workspace collapse had zero observed calls, so it stays on the current
  `COGNITION_LLM` values.
- The wrong-target failure-mode result was `0/192` for every configuration.
  Admission rejection requires a separate owning-boundary bugfix; endpoint
  assignment cannot repair it.

The existing Core V2 DAG already launches six semantic appraisal families and
preliminary goal branches concurrently. This plan changes route ownership only.
It preserves prompts, payloads, schemas, attempt caps, task creation,
dependency edges, state reduction, and call count.

The generic `COGNITION_LLM` route also serves model calls outside the
`run_cognition(...)` intake-to-output boundary. It remains configured and
unchanged for those consumers. `BOUNDARY_CORE_LLM` has no required production
consumer after this cutover and is removed.

The Control Console already renders a descriptor-driven route matrix and
provides model discovery, process-local overrides, reset, and restart-based
application. Adding stage descriptors is sufficient for the web portal; the
API shape, static renderer, override lifecycle, and security model remain
unchanged.

Target operational surface:

- Workspace and served checkout: `C:\workspace\kazusa_ai_chatbot`
- Control Console bind: `127.0.0.1:8764`
- Control Console URL: `http://127.0.0.1:8764/`
- Environment file in scope: `C:\workspace\kazusa_ai_chatbot\.env`

## Mandatory Skills

- `development-plan`: govern execution stages, evidence, parent multi-pass
  review, lifecycle updates, and final sign-off.
- `local-llm-architecture`: preserve semantic and deterministic ownership,
  route existing model calls without adding prompts or calls, and keep local
  model context and latency bounded.
- `py-style`: load before editing or reviewing Python. Apply PEP 8, explicit
  stage ownership, top-level imports, narrow exception handling, useful
  docstrings, stable runtime language, surgical changes, and earned helpers.
- `cjk-safety`: preserve the exact UTF-8 prompt content in every edited
  CJK-bearing Python module and run syntax validation immediately after each
  edit.
- `test-style-and-execution`: establish deterministic routing contracts before
  production edits and keep live-model generation outside this execution.
- `control-console-web-development`: preserve the buildless static console,
  existing route editor, auth/CSRF, restart behavior, redaction, and rendered
  browser verification.

## Mandatory Rules

- After automatic context compaction, the parent or active execution agent
  rereads this entire plan before continuing implementation, verification,
  handoff, or reporting.
- After each major checklist stage is signed off, the parent or active
  execution agent rereads this entire plan before starting the next stage.
- Before lifecycle completion, merge, or sign-off, the parent runs the Parent
  Multi-Pass Code Review gate and records the result in Execution Evidence.
- The completed production implementation used the one required production
  subagent. By explicit user direction after the browser defect review, all
  remaining review, remediation, and lifecycle work stays with the parent;
  no further subagent call is permitted.
- Every new cognition route name starts with `COGNITION_LLM_` and ends with a
  stable semantic stage name. Production code, environment variable names,
  route keys, and UI labels use stage vocabulary exclusively.
- Endpoint URLs, credentials, model ids, token budgets, thinking flags, and
  model-class choices live in `.env` or process-local Control Console
  overrides. Python contains no endpoint, model id, endpoint role, or
  model-class assignment.
- Each stage prefix owns a complete independent environment bundle. Route
  profiles, inheritance, aliases, fallback prefixes, endpoint pools, and
  translation layers remain absent.
- `BASE_URL`, `API_KEY`, and `MODEL` are required for every stage route.
  `MAX_COMPLETION_TOKENS` and `THINKING_ENABLED` retain the same parsing and
  default semantics as existing routes, while the workspace `.env` contains
  explicit values for all five fields.
- Missing stage configuration fails at startup through the existing required
  configuration boundary. Generic `COGNITION_LLM` and
  `BOUNDARY_CORE_LLM` fallback behavior remains absent.
- The existing generic `COGNITION_LLM` bundle and its consumers outside Core
  V2 remain unchanged.
- `BOUNDARY_CORE_LLM` is removed from active source, tests, current
  documentation, Control Console descriptors, route diagnostics, and the
  workspace `.env` after its values are copied to the assigned stage bundles.
- Every initial attempt, structural replacement, provider retry, trace row,
  and diagnostic row uses the same selected stage config as its semantic
  owner.
- Prompt text, model-facing payloads, JSON contracts, semantic validators,
  attempt caps, deterministic state ownership, call count, and DAG concurrency
  remain unchanged.
- CJK prompt constants remain byte-equivalent. Edits stay outside prompt
  bodies, UTF-8 decoding is explicit in any inspection script, and each
  CJK-bearing Python file receives immediate `py_compile` validation.
- The existing Control Console model-route APIs and static route editor handle
  the new descriptors. Portal work adds route rows only; API fields, storage,
  runtime hot reload, base-URL editing, credential editing, and UI widgets
  remain unchanged.
- Control Console overrides remain process-local and restart-based. `.env`
  persistence is handled only by the workspace configuration edit in this
  plan.
- Raw `.env` file contents and credentials remain absent from command output,
  diffs, audit records, screenshots, browser payloads, and execution evidence.
  Sanitized route names, model ids, and normalized endpoint origins may appear
  where the existing diagnostics and portal contract already expose them.
- Real LLM generation is outside this implementation verification. Endpoint
  `/models` discovery, config import, service startup, health checks, and
  deterministic fake-invoker tests provide the routing evidence.
- Existing unrelated worktree changes remain preserved and outside this plan.

## Must Do

- Introduce the thirteen exact stage route prefixes in this plan.
- Replace the four aggregate `CognitionCoreServicesV2` config fields with the
  thirteen explicit stage config fields.
- Select the exact stage config at every model invocation and trace boundary.
- Populate all thirteen full bundles in the current workspace `.env` from the
  two currently configured source routes according to the assignment table.
- Preserve the old boundary variables while the old code is still active,
  then remove the obsolete `BOUNDARY_CORE_LLM` route in the production
  cutover before the first new-code startup.
- Add every stage route to startup diagnostics and the existing Control
  Console Brain model-route catalog.
- Update all direct service fixtures, historical harness bindings, subprocess
  environment allowlists, current docs, and deterministic route expectations.
- Verify a clean config import, Brain service startup, Control Console route
  rendering, provider-model discovery, and the full non-live regression suite.

## Deferred

- Leave wrong-target admission rejection to a separate owning-boundary bugfix.
- Leave new semantic quality evaluation and model comparison to a separately
  authorized evidence plan.
- Leave endpoint executor counts, queue policy, load balancing, saturation,
  latency measurement, and throughput tuning to the post-routing performance
  effort.
- Keep prompts, output schemas, semantic validators, retries, call caps,
  dependency scheduling, and action/resolver capability semantics unchanged.
- Keep generic `COGNITION_LLM` consumers outside `run_cognition(...)`
  unchanged.
- Keep dialog, text/visual surface planning, memory lifecycle, persistence,
  consolidation, adapters, scheduler, and database behavior unchanged.
- Keep the Control Console API shape, override persistence model, static
  component system, and service registry unchanged.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Core V2 service binding | bigbang | Replace the four aggregate config fields with the thirteen exact stage fields in one change. |
| Core V2 call sites | bigbang | Route every invocation and retry directly through its stage field. |
| Environment | migration | Prepopulate all new bundles while old code remains active; remove `BOUNDARY_CORE_LLM_*` at the production cutover before the first new-code startup. |
| Generic cognition route | compatible | Retain `COGNITION_LLM_*` for existing consumers outside Core V2. |
| Control Console catalog | bigbang | Replace the boundary route row with the thirteen stage rows through the existing descriptor contract. |
| Diagnostics and docs | bigbang | Show only the current route contract; remove stale boundary documentation. |
| Tests | bigbang | Rewrite fixtures and route expectations to the new service contract; compatibility aliases remain absent. |

Cutover enforcement:

- Each area follows its listed policy.
- Bigbang areas contain one canonical vocabulary and one active path.
- Compatible retention applies only to generic `COGNITION_LLM` consumers
  outside Core V2.
- A cutover-policy change requires user approval before implementation.

## Target State

`CognitionCoreServicesV2` contains one `LLMInvoker` and thirteen immutable
stage configs. Core model calls resolve as follows:

| Runtime owner | Service field | Environment prefix |
|---|---|---|
| Event and agency appraisal | `appraisal_event_agency_config` | `COGNITION_LLM_APPRAISAL_EVENT_AGENCY` |
| Relationship and social appraisal | `appraisal_relationship_social_config` | `COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL` |
| Moral and identity appraisal | `appraisal_moral_identity_config` | `COGNITION_LLM_APPRAISAL_MORAL_IDENTITY` |
| Goal, threat, and outcome appraisal | `appraisal_goal_threat_outcome_config` | `COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME` |
| Epistemic, comparison, and memory appraisal | `appraisal_epistemic_comparison_memory_config` | `COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY` |
| Existential and drive appraisal | `appraisal_existential_drive_config` | `COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE` |
| Ordinary-response goal branch | `goal_ordinary_response_config` | `COGNITION_LLM_GOAL_ORDINARY_RESPONSE` |
| Active persistent-goal branches | `goal_active_branch_config` | `COGNITION_LLM_GOAL_ACTIVE_BRANCH` |
| Required-selection verifier | `required_selection_verifier_config` | `COGNITION_LLM_REQUIRED_SELECTION_VERIFIER` |
| Workspace collapse | `workspace_collapse_config` | `COGNITION_LLM_WORKSPACE_COLLAPSE` |
| Action planning and goal resolution | `action_planning_config` | `COGNITION_LLM_ACTION_PLANNING` |
| Action authorization | `action_authorization_config` | `COGNITION_LLM_ACTION_AUTHORIZATION` |
| Resolver authorization | `resolver_authorization_config` | `COGNITION_LLM_RESOLVER_AUTHORIZATION` |

Repairs remain with their owner:

- appraisal structural replacements reuse the selected appraisal-family config;
- ordinary and active goal replacements reuse their selected goal config;
- required-selection rechecks use the verifier config, while regenerated bids
  reuse the owning goal config;
- collapse replacements reuse the collapse config;
- action-planning replacements reuse the action-planning config;
- action and resolver authorization replacements reuse their respective
  authorization configs.

The existing first parallel phase submits up to twenty model tasks:

| Concurrent work | Assigned source values | Existing task count |
|---|---|---:|
| Event/agency appraisal plus ordinary-response goal | current `COGNITION_LLM_*` | up to 2 |
| Five other appraisal families plus active-goal branches | current `BOUNDARY_CORE_LLM_*` | up to 18 |

With the currently available endpoint capacities, this exposes up to two
simultaneous executions on the first source service and five on the second
source service. Application-level executor limits and task scheduling remain
unchanged.

Later execution remains ordered:

```text
parallel preliminary appraisals and goal branches
  -> final active-goal branches after appraisal dependencies
  -> workspace collapse
  -> action planning and goal resolution
  -> action authorization or resolver authorization when requested
  -> deterministic materialization and output validation
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Route vocabulary | Use `COGNITION_LLM_<SEMANTIC_STAGE>` exclusively for Core V2 stage routes. | The route name identifies the caller stage without encoding endpoint or model type. |
| Appraisal granularity | Give all six existing appraisal families separate routes. | They are independent concurrent semantic owners, and the accepted assignment treats event/agency differently from the other five. |
| Goal granularity | Split ordinary response from active persistent-goal branches. | It expresses the accepted allocation with two routes and avoids thirteen branch-specific route bundles. |
| Repair ownership | Reuse the original stage route for every repair and retry. | A retry remains the same semantic stage and experimental assignment. |
| Required selection | Give the verifier its own route while bid regeneration stays with the owning goal route. | The verifier is a distinct high-authority judgment; regeneration remains goal cognition. |
| Action planning | Keep proposals and `goal_resolution` in one existing action-planning call and route. | Splitting them would add a prompt, call, schema, and latency path outside this routing-only change. |
| Authorization | Give action and resolver authorization separate routes. | They use distinct prompts and semantic ownership despite sharing one helper today. |
| Environment shape | Repeat all five route fields for every stage prefix. | Each configured endpoint is standalone and can point to any OpenAI-compatible service. |
| Generic cognition | Retain `COGNITION_LLM` outside Core V2. | Existing non-Core consumers remain in scope-neutral operation. |
| Boundary route | Remove `BOUNDARY_CORE_LLM` after value migration. | Its aggregate name no longer represents an active caller. |
| Portal integration | Add descriptors to the existing route catalog. | The current API and static UI already support arbitrary descriptor rows. |
| Runtime switching | Preserve restart-based route application. | This matches existing route behavior and the user's portal requirement. |

## Contracts And Data Shapes

### Environment Bundle

Every prefix in the assignment table owns:

```text
<PREFIX>_BASE_URL
<PREFIX>_API_KEY
<PREFIX>_MODEL
<PREFIX>_MAX_COMPLETION_TOKENS
<PREFIX>_THINKING_ENABLED
```

`BASE_URL`, `API_KEY`, and `MODEL` use required `os.environ[...]` loading.
`MAX_COMPLETION_TOKENS` uses the existing positive-integer parser and default
completion budget. `THINKING_ENABLED` uses the existing Boolean parser and
defaults to false. The workspace `.env` writes all five values explicitly.

### Workspace Prepopulation Assignment

| New environment prefix | Copy source values from |
|---|---|
| `COGNITION_LLM_APPRAISAL_EVENT_AGENCY` | existing `COGNITION_LLM` |
| `COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL` | existing `BOUNDARY_CORE_LLM` |
| `COGNITION_LLM_APPRAISAL_MORAL_IDENTITY` | existing `BOUNDARY_CORE_LLM` |
| `COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME` | existing `BOUNDARY_CORE_LLM` |
| `COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY` | existing `BOUNDARY_CORE_LLM` |
| `COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE` | existing `BOUNDARY_CORE_LLM` |
| `COGNITION_LLM_GOAL_ORDINARY_RESPONSE` | existing `COGNITION_LLM` |
| `COGNITION_LLM_GOAL_ACTIVE_BRANCH` | existing `BOUNDARY_CORE_LLM` |
| `COGNITION_LLM_REQUIRED_SELECTION_VERIFIER` | existing `COGNITION_LLM` |
| `COGNITION_LLM_WORKSPACE_COLLAPSE` | existing `COGNITION_LLM` |
| `COGNITION_LLM_ACTION_PLANNING` | existing `COGNITION_LLM` |
| `COGNITION_LLM_ACTION_AUTHORIZATION` | existing `COGNITION_LLM` |
| `COGNITION_LLM_RESOLVER_AUTHORIZATION` | existing `COGNITION_LLM` |

For each new prefix, copy the exact existing `BASE_URL`, `API_KEY`, and
`MODEL`. Write the effective current completion budget as
`MAX_COMPLETION_TOKENS=8192` and the effective current thinking state as
`THINKING_ENABLED=false`. Retain the original `COGNITION_LLM_*` entries.
Keep `BOUNDARY_CORE_LLM_*` only while the pre-cutover code is still active.
Remove every old entry after the production code uses the new stage contract
and before the first new-code config import or service startup.

### Service Contract

The canonical dataclass fields are exactly:

```text
llm
appraisal_event_agency_config
appraisal_relationship_social_config
appraisal_moral_identity_config
appraisal_goal_threat_outcome_config
appraisal_epistemic_comparison_memory_config
appraisal_existential_drive_config
goal_ordinary_response_config
goal_active_branch_config
required_selection_verifier_config
workspace_collapse_config
action_planning_config
action_authorization_config
resolver_authorization_config
```

The four old config fields are removed without properties, aliases, adapters,
or fallback lookup.

### Selection Contract

- `semantic_appraisal.appraise_semantic_question(...)` selects one exact
  config from the validated `question_kind`.
- `goal_cognition.run_goal_cognition(...)` selects
  `goal_ordinary_response_config` only for branch id `ordinary_response`;
  every other registered branch uses `goal_active_branch_config`.
- Required-selection verification always uses
  `required_selection_verifier_config`.
- `workspace.collapse_bids(...)` always uses
  `workspace_collapse_config`.
- Action planning always uses `action_planning_config`.
- `invoke_semantic_authorizer(...)` receives its config explicitly. Action
  callers pass `action_authorization_config`; resolver callers pass
  `resolver_authorization_config`.
- Trace helpers receive or derive the same selected config used for the call
  and report its exact `route_name` and `model`.

### Control Console Contract

Each new route is a normal required `BrainModelRouteDescriptor` in group
`Cognition Core V2`. It has the existing editable fields:

```text
model
max_completion_tokens
thinking_enabled
```

The existing endpoints remain:

```text
GET  /api/services/brain/model-routes
PUT  /api/services/brain/model-routes/{route_key}
POST /api/services/brain/model-routes/{route_key}/reset
GET  /api/services/brain/model-routes/{route_key}/available-models
```

No response or request field changes.

## LLM Call And Context Budget

Production before and after:

- total model-call count per cognition run: unchanged;
- response-path versus background classification: unchanged;
- prompt constants and rendered payloads: unchanged;
- context caps: unchanged, with existing 24,000-character stage caps and
  stricter local caps retained;
- completion budgets: explicit per-stage environment values, initially equal
  to the current effective 8,192-token budget;
- thinking: explicit per-stage environment values, initially false;
- retry and replacement caps: unchanged;
- blocking dependencies and graph edges: unchanged.

The routing cutover redistributes existing concurrent calls across configured
services. It adds zero model calls, zero judge calls, zero repair calls, and
zero new context fields. Actual throughput remains an operational measurement
after this quality-derived assignment.

## Change Surface

Target ownership boundary: Cognition Core V2 route configuration and the
existing Control Console Brain route catalog.

### Delete

- `tests/test_boundary_core_sensitivity_live_llm.py`
  - Remove the obsolete two-aggregate-route sensitivity test. The closed
    matrix evidence remains in the archived plan and ignored local artifacts.

### Modify

- `.env`
  - Populate the thirteen complete stage bundles in the current workspace and
    remove the obsolete boundary bundle without exposing secrets.
- `docker-compose.yml`
  - Replace the obsolete required boundary-route bindings with the required
    endpoint, credential, and model bindings for all thirteen stage routes so
    container startup receives the same fail-fast route contract.
- `src/kazusa_ai_chatbot/config.py`
  - Add explicit required stage constants and stage generation settings;
    remove boundary constants.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Construct thirteen stage-named `LLMCallConfig` values and inject the new
    service contract.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
  - Replace the four aggregate dataclass fields with the exact stage fields.
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`
  - Select and trace the config for each appraisal family.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
  - Select ordinary versus active-goal config and route the verifier
    independently while keeping repairs with their owner.
- `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py`
  - Use the workspace-collapse config.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
  - Use the action-planning config and preserve its combined semantic contract.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py`
  - Accept an explicit authorizer config and use the action-authorization
    config.
- `src/kazusa_ai_chatbot/cognition_core_v2/resolver_authorization.py`
  - Pass the resolver-authorization config.
- `src/kazusa_ai_chatbot/llm_interface/route_report.py`
  - Replace the boundary route with all stage routes in required startup
    diagnostics.
- `src/control_console/brain_model_routes.py`
  - Replace the boundary descriptor with thirteen ordinary stage descriptors.
- `src/control_console/service_config.py`
  - Raise the bounded descriptor field-key and field-count limits to 80 so
    the exact stage-owned route vocabulary can be represented. The current
    catalog contains 78 fields and its longest generated field key is 73
    characters; request, storage, rendering, and restart behavior remain
    unchanged.
- `src/control_console/static/console.js` and
  `src/control_console/static/console.css`
  - Keep model/source/family/thinking metadata visible on the selected route
    tile and use the console's existing balanced selected-state palette. This
    repairs the empty, heavy-bordered tile exposed by rendered validation
    without changing filtering, route data, API behavior, or editor controls.
  - Stack the compact Brain runtime panel above the route matrix at desktop
    and mobile widths so it neither stretches to the route-matrix height nor
    leaves a full-height empty side column inside the selected Brain service
    card.
- `README.md`, `README_CN.md`, `docs/HOWTO.md`,
  `src/kazusa_ai_chatbot/cognition_core_v2/README.md`,
  `src/kazusa_ai_chatbot/nodes/README.md`, and
  `src/control_console/README.md`
  - Document the exact current route contract, assignment boundary, startup,
    and unchanged portal behavior.
- `tests/cognition_core_v2_model_assignment_matrix.py` and
  `tests/test_cognition_core_v2_model_assignment_matrix.py`
  - Mechanically adapt the closed four-factor harness to the new service
    fields without rerunning or reinterpreting its real-model corpus.
- `tests/test_config.py`,
  `tests/test_cognition_core_v2_contracts.py`,
  `tests/test_cognition_core_v2_dependencies.py`,
  `tests/test_cognition_core_v2_integration.py`,
  `tests/test_cognition_current_event_grounding.py`,
  `tests/test_conversation_progress_cognition.py`,
  `tests/test_cognition_core_v2_action_authorization.py`,
  `tests/test_cognition_core_v2_action_planning_bugfix.py`,
  `tests/test_cognition_core_v2_resolver_authorization.py`,
  `tests/test_cognition_resolver_l2d_contract.py`, and
  `tests/test_persona_supervisor2_action_selection.py`
  - Rewrite fixtures and assertions to the canonical stage config fields.
- `tests/test_background_work_future_speak_live_llm.py`,
  `tests/test_coding_agent_full_workflow_integration_live_llm.py`,
  `tests/test_cognition_core_v2_action_planning_live_llm.py`,
  `tests/test_cognition_core_v2_live_llm.py`,
  `tests/test_l2d_action_selection_live_llm.py`, and
  `tests/test_l2d_unknown_context_resolver_live_llm.py`
  - Keep collection/import compatibility with the new fields; retain their
    live markers and leave them unexecuted in this plan.
- `tests/cognition_baseline_worker.py`,
  `tests/cognition_baseline_comparison.py`,
  `tests/test_stage3_fresh_database_e2e_live_llm.py`, and
  `tests/test_web_agent3.py`
  - Replace subprocess required-environment allowlists with the stage bundles.
- `tests/test_control_console_brain_model_routes.py`,
  `tests/test_control_console_config_routes.py`,
  `tests/test_control_console_web_surface.py`,
  `tests/test_llm_interface_route_report.py`, and
  `tests/test_documentation_harmonization.py`
  - Verify the exact route catalog, generic API behavior, redaction, dynamic
    UI rendering, compact desktop panel alignment, diagnostics, and
    documentation parity.
- `tests/control_console_e2e/test_page_navigation_e2e.py` and
  `tests/control_console_e2e/test_running_console_signoff_e2e.py`
  - Replace the stale selected-route-empty expectation with rendered checks
    for the model line and three metadata badges. Reject a desktop runtime
    panel stretched to the route-matrix height, unequal panel bounds, or an
    empty side column.
- `tests/test_console_debug_chat.py`,
  `tests/test_control_console_cognition_graph.py`, and
  `tests/test_decontextualizer_referents.py`
  - Reconcile stale assertions discovered by the full regression gate with
    the existing projected-audit, top-level graph, and bounded
    decontextualizer fallback contracts. Production behavior remains
    unchanged.
- `tests/fixtures/cognition_baseline_owner_matrix.json`
  - Add the missing dispatcher and message-envelope ownership rules exposed
    by the final full regression after the concurrent character-name cutover.
    The rules assign deterministic delivery and typed intake boundaries once
    without changing runtime behavior or the routing contract.
- `development_plans/README.md`
  - Track approval and later completion.

### Create

- `tests/test_cognition_core_v2_stage_model_routing.py`
  - Focused deterministic fake-invoker tests for every stage, repair route,
    trace identity, parallel first-wave assignment, and absence of aggregate
    config fields.

### Keep

- Prompt constants and model-facing JSON schemas remain byte-unchanged.
- `src/kazusa_ai_chatbot/llm_interface/` provider behavior remains unchanged
  except for the route-report catalog.
- `src/control_console/static/index.html` remains unchanged because the route
  UI is descriptor-driven.
- Database state, character profiles, local evaluation artifacts, adapters,
  dialog, persistence, scheduler, reflection, and consolidation remain
  unchanged.

## Overdesign Guardrail

- Actual problem: four aggregate route bindings prevent independent endpoint
  assignment for existing Core V2 semantic stages and mislabel several
  authority calls as one boundary route.
- Minimal change: replace aggregate config fields with thirteen explicit
  stage configs, populate `.env`, and register those configs in the existing
  diagnostics and portal catalogs.
- Ownership boundaries: stage modules select route config; LLM stages retain
  semantic judgment; deterministic code retains validation, state mutation,
  permissions, persistence, and delivery; the Control Console retains
  process-local restart-based overrides.
- Rejected complexity: route profiles, model-class flags, endpoint aliases,
  inherited bundles, dynamic routers, fallback chains, load balancers,
  application executor controls, extra LLM audits, prompt splits, new API
  fields, hot reload, and `.env` persistence from the browser.
- Evidence threshold: a future abstraction requires at least three active
  deployments whose duplicated stage bundles cause an observed maintenance
  failure, or measured load evidence that requires runtime endpoint pooling.

## Agent Autonomy Boundaries

- The responsible agent may select local implementation mechanics only when
  they preserve every exact route prefix, field, assignment, call count, and
  verification gate in this plan.
- The responsible agent searches for existing route selection, descriptor,
  config parsing, and trace helpers before introducing any helper.
- A new helper is limited to non-trivial stage-to-field table lookup or an
  established repeated structural pattern. Pass-through wrappers and generic
  routing layers remain outside scope.
- Changes outside the listed target boundary require plan-level justification
  before implementation.
- The responsible agent keeps unrelated formatting, dependency upgrades,
  prompt rewrites, schema changes, and cleanup outside the diff.
- A plan/code disagreement, missing required source value, or unsafe secret
  exposure blocks execution and is reported rather than replaced with a
  fallback.

## Implementation Order

1. Establish the focused routing contract.
   - Create `tests/test_cognition_core_v2_stage_model_routing.py`.
   - Assert all thirteen dataclass fields, question-family selection,
     ordinary/active goal selection, verifier selection, collapse, planning,
     both authorizers, retries, and trace route identity.
   - Run the focused test before production edits.
   - Expected baseline: collection or assertions fail because the stage fields
     and route constants are absent.

2. Establish configuration and portal contracts.
   - Update deterministic config, route-report, Control Console route, API,
     and static-surface tests with the exact thirteen prefixes.
   - Assert `BOUNDARY_CORE_LLM` is absent and generic `COGNITION_LLM` remains.
   - Run these tests and record their expected pre-implementation failures.

3. Prepopulate the workspace environment safely.
   - Read the existing two source bundles in memory.
   - Write all thirteen five-field bundles according to the assignment table.
   - Preserve generic `COGNITION_LLM`.
   - Retain `BOUNDARY_CORE_LLM` during this stage so the existing checkout
     remains restartable before production code changes.
   - Confirm every new route has a non-empty endpoint, credential, and model
     plus valid generation settings.
   - Record names and presence states only.

4. Start one production-code subagent.
   - Provide this approved plan, mandatory skills, focused failing tests, exact
     production files, and the production-only ownership boundary.
   - The subagent updates `src/kazusa_ai_chatbot/` and
     `src/control_console/` only.
   - The subagent reports changed files, commands, blockers, and residual risk,
     then closes.

5. Update all deterministic and collection-only test consumers.
   - Remove `BOUNDARY_CORE_LLM` from `.env` immediately after the production
     code uses the new route contract and before importing that new code.
   - Rewrite direct dataclass fixtures and `SimpleNamespace` fields.
   - Adapt the closed matrix harness mechanically.
   - Update subprocess environment allowlists.
   - Remove the obsolete boundary sensitivity live test.
   - Verify static greps find no active old field or boundary-route references.

6. Update current documentation.
   - Replace aggregate Core route examples with exact stage prefixes.
   - Document the retained generic cognition route and removed boundary route.
   - Record the existing parallel waves and unchanged Control Console behavior.
   - Keep profile-specific and raw evaluation content outside tracked docs.

7. Run focused module and integration verification.
   - Run config, Core routing, Core contract/integration, trace, route report,
     Control Console, and documentation tests.
   - Run `py_compile` for every changed Python file.
   - Inspect the zero-context production diff and confirm every CJK prompt
     body is unchanged.
   - Reconcile failures only within the approved contract and change surface.

8. Run full non-live regression and runtime import gates.
   - Run the complete suite excluding `live_llm` and `live_db`.
   - Import config and construct `CognitionCoreServicesV2` from the workspace
     `.env`.
   - Render the sanitized startup route table and verify all stage rows.
   - Start the Brain through the existing Control Console lifecycle and verify
     health without sending a chat turn.

9. Validate the existing web portal.
   - Use the in-app Browser first at `http://127.0.0.1:8764/`.
   - Authenticate through the existing operator session.
   - Open Services, filter group `Cognition Core V2`, select stage rows from
     both source assignments, and exercise model discovery.
   - Verify route count, labels, editor fields, explicit unavailable states,
     no secret exposure, no overflow, and no console/page errors.
   - Repair and retest any rendered selected-tile artifact before sign-off;
     selected tiles retain the same model and route metadata as unselected
     tiles.
   - Inspect bordered-panel dimensions and reject any sparse panel stretched
     to the route-matrix height or any empty side column left beside the route
     matrix.
   - Clear temporary filters for the final overview screenshot so all 26
     configured routes are visibly represented.
   - Capture a local screenshot and browser evidence under `test_artifacts/`.

10. Run parent multi-pass code review.
    - Pass one reviews route ownership, configuration, retries, traces,
      prompts, schemas, and call/DAG preservation.
    - Pass two reviews deployment, Control Console behavior and security,
      rendered desktop/mobile evidence, tests, and documentation.
    - Pass three re-reads the complete diff and lifecycle evidence after all
      fixes, then reruns every affected gate.

11. Complete lifecycle sign-off.
    - Record final commands and evidence.
    - Update this plan to completed and move it to completed history only after
      every acceptance criterion and review gate passes.
    - Update `development_plans/README.md`.

## Execution Model

- Parent agent owns orchestration, focused and integration tests, `.env`
  prepopulation, documentation, verification, browser evidence, review
  remediation, lifecycle updates, and final sign-off.
- Parent establishes and records the focused failing routing contract before
  production implementation begins.
- Production-code subagent: exactly one native subagent, started after the
  focused contract exists; owns only listed production Python changes; closes
  after planned production edits and before review fixes.
- Parent may update tests, docs, environment, and verification evidence while
  the production subagent edits production code.
- Parent review remains local for all remaining passes by explicit user
  direction; the previously started read-only reviewer was closed before
  producing a review result.

## Progress Checklist

- [x] Stage 1 - focused route and portal contracts established
  - Covers: implementation steps 1-2.
  - Verify: focused Core and portal tests fail for the expected missing stage
    contract, with no unrelated collection failure.
  - Evidence: record test names, failure reason, and unchanged production diff.
  - Handoff: next stage prepopulates `.env`.
  - Sign-off: `Codex/2026-07-27`.

- [x] Stage 2 - workspace `.env` prepopulated
  - Covers: implementation step 3.
  - Verify: thirteen complete bundles are present, generic cognition and the
    still-required pre-cutover boundary bundle remain, and redacted validation
    reports valid values.
  - Evidence: route names and value-presence states only.
  - Handoff: next stage starts the production-code subagent.
  - Sign-off: `Codex/2026-07-27`.

- [x] Stage 3 - production stage routing implemented
  - Covers: implementation step 4.
  - Verify: production subagent reports the exact planned files; prompt bodies,
    call count, provider behavior, static UI, and outside-Core behavior remain
    unchanged; the parent removes the old boundary bundle before importing the
    new code.
  - Evidence: changed-file inventory and subagent closeout.
  - Handoff: parent updates all test consumers.
  - Sign-off: `Codex/2026-07-27`.

- [x] Stage 4 - consumers and documentation migrated
  - Covers: implementation steps 5-6.
  - Verify: old service fields and boundary route have zero active matches;
    documentation route parity passes.
  - Evidence: static grep output and changed-file inventory.
  - Handoff: next stage runs focused and full verification.
  - Sign-off: `Codex/2026-07-27`.

- [x] Stage 5 - deterministic and runtime verification passes
  - Covers: implementation steps 7-8.
  - Verify: focused tests, `py_compile`, full non-live suite, config import,
    service construction, route report, and Brain health all pass.
  - Evidence: exact commands, counts, service URL, and sanitized route rows.
  - Handoff: next stage validates the portal.
  - Sign-off: `Codex/2026-07-27`.

- [x] Stage 6 - Control Console browser validation passes
  - Covers: implementation step 9.
  - Verify: route matrix, filtering, selection, model discovery, editor state,
    responsiveness, redaction, and browser error health pass.
  - Evidence: browser path, URL, session disposition, interactions, errors,
    and local screenshot path.
  - Handoff: next stage runs the parent multi-pass review.
  - Sign-off: `Codex/2026-07-27`.

- [x] Stage 7 - parent multi-pass code review approved
  - Covers: implementation step 10.
  - Verify: all three parent review passes find no unresolved contract,
    security, quality, test, deployment, visual, or scope issue after
    remediation.
  - Evidence: pass findings, fixes, rerun commands, residual risks, and
    approval.
  - Handoff: next stage completes lifecycle sign-off.
  - Sign-off: `Codex/2026-07-27`.

- [x] Stage 8 - lifecycle closed
  - Covers: implementation step 11.
  - Verify: all prior stages are signed, acceptance criteria pass, registry is
    current, and the archived plan contains complete execution evidence.
  - Evidence: final git status, archive path, and registry link.
  - Sign-off: `Codex/2026-07-27`.

## Verification

### Focused Routing Tests

- `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_stage_model_routing.py tests\test_cognition_core_v2_contracts.py tests\test_cognition_core_v2_dependencies.py tests\test_cognition_core_v2_action_authorization.py tests\test_cognition_core_v2_action_planning_bugfix.py tests\test_cognition_core_v2_resolver_authorization.py -q`
  - Expected: every stage and retry uses its exact config; old aggregate fields
    are absent; all tests pass.
- `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_integration.py tests\test_cognition_current_event_grounding.py tests\test_conversation_progress_cognition.py tests\test_cognition_resolver_l2d_contract.py tests\test_persona_supervisor2_action_selection.py -q`
  - Expected: Core integration and connector behavior pass with unchanged
    semantics.
- `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_model_assignment_matrix.py -q`
  - Expected: the closed harness remains mechanically valid under the new
    service shape; no real-model run occurs.

### Configuration And Portal Tests

- `venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_llm_interface_route_report.py tests\test_control_console_brain_model_routes.py tests\test_control_console_config_routes.py tests\test_control_console_web_surface.py tests\test_documentation_harmonization.py -q`
  - Expected: exact route catalog, config requirements, redaction, generic API
    behavior, dynamic static UI, and docs parity pass.

### Collection Compatibility

- `venv\Scripts\python.exe -m pytest --collect-only -q`
  - Expected: all regular and live-marked modules collect with zero import,
    missing-config, or stale-field errors.

### Full Non-Live Regression

- `venv\Scripts\python.exe -m pytest -q -m "not live_llm and not live_db"`
  - Expected: complete non-live suite passes.

### Syntax

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\cognition_core_v2\contracts.py src\kazusa_ai_chatbot\cognition_core_v2\semantic_appraisal.py src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py src\kazusa_ai_chatbot\cognition_core_v2\workspace.py src\kazusa_ai_chatbot\cognition_core_v2\action_selection.py src\kazusa_ai_chatbot\cognition_core_v2\action_authorization.py src\kazusa_ai_chatbot\cognition_core_v2\resolver_authorization.py src\kazusa_ai_chatbot\llm_interface\route_report.py src\control_console\brain_model_routes.py tests\test_cognition_core_v2_stage_model_routing.py`
  - Expected: exit code zero.
- Run the same `py_compile` command immediately after each edit to
  `semantic_appraisal.py`, `goal_cognition.py`, `workspace.py`,
  `action_selection.py`, `action_authorization.py`, or
  `resolver_authorization.py`.
  - Expected: exit code zero before any subsequent edit.

### Static Contract Gates

- `rg -n "BOUNDARY_CORE_LLM" README.md docs src tests`
  - Expected: zero matches; `rg` exit code 1 is the successful result.
- `rg -n "\b(appraisal_config|goal_cognition_config|collapse_config|action_selection_config)\b" src\kazusa_ai_chatbot\cognition_core_v2 src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py`
  - Expected: zero matches; only the thirteen canonical field names remain.
- `rg -n -i "\bdense\b|\bmoe\b" src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\cognition_core_v2 src\control_console\brain_model_routes.py`
  - Expected: zero matches.
- `git diff --name-only -- src\control_console\static`
  - Expected: exactly `console.js` and `console.css`; static HTML and the
    framework remain unchanged.
- `git diff --check`
  - Expected: no whitespace errors.
- `git diff -U0 -- src\kazusa_ai_chatbot\cognition_core_v2\semantic_appraisal.py src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py src\kazusa_ai_chatbot\cognition_core_v2\workspace.py src\kazusa_ai_chatbot\cognition_core_v2\action_selection.py src\kazusa_ai_chatbot\cognition_core_v2\action_authorization.py src\kazusa_ai_chatbot\cognition_core_v2\resolver_authorization.py`
  - Expected: changed lines are limited to config selection, invocation, and
    trace wiring; no prompt-body line changes.
- `git check-ignore .env`
  - Expected: `.env` is ignored.

### Runtime Configuration And Service Smoke

- Import `kazusa_ai_chatbot.config`, construct
  `build_cognition_core_services()`, and print only the thirteen route names,
  model ids, normalized endpoint origins, completion budgets, and thinking
  states.
  - Expected: all stage routes are present, credentials remain hidden, and
    `BOUNDARY_CORE_LLM` is absent.
- Render `render_llm_route_table()`.
  - Expected: every stage route appears exactly once, generic
    `COGNITION_LLM` remains, and the boundary route is absent.
- Start the Brain through the Control Console and request
  `http://127.0.0.1:8000/health`.
  - Expected: startup and health succeed from the workspace `.env` without a
    chat or model-generation call.

### Browser Gate

- Use the in-app Browser at `http://127.0.0.1:8764/`.
- Record page identity, URL, authenticated session state, Services tab,
  `Cognition Core V2` filter, all thirteen route labels, selected route editor,
  provider-model discovery states, console errors, page errors, horizontal
  overflow, and screenshot path.
- Expected: all route rows use the existing interaction model, sensitive
  values remain absent, and every exercised control reaches a terminal state.
- If the in-app Browser is unavailable, record its exact failure and use
  system-Chrome Playwright according to the control-console skill.

## Parent Multi-Pass Code Review

Run this gate locally after all verification commands pass and before
completion or merge. By explicit user direction, the parent performs every
remaining review iteration and fixes surfaced issues before starting the next
pass.

Review scope:

- compliance with project rules and all mandatory skills;
- exact thirteen-prefix environment and service contracts;
- exact source-value assignment and `.env` secret handling;
- absence of endpoint, model id, endpoint role, or model-class hardcoding;
- absence of aggregate-field aliases, boundary-route fallback, compatibility
  shims, and route-profile indirection;
- exact call-site, retry, trace, and diagnostic route identity;
- unchanged prompts, payloads, semantic schemas, call counts, retry caps, DAG,
  and outside-Core consumers;
- exact UTF-8 preservation of every CJK prompt body in edited modules;
- existing Control Console API, security, redaction, override, restart, and
  static-renderer behavior;
- focused, full non-live, runtime, and browser evidence accuracy;
- complete migration of all active source, tests, docs, and subprocess
  environment allowlists.

Pass one covers routing and cognition contracts. Pass two covers deployment,
tests, documentation, Control Console security, and rendered desktop/mobile
quality. Pass three re-reads the final diff and evidence after remediation.
The parent fixes findings inside this approved change surface and reruns every
affected gate. A finding requiring a new call, prompt, schema, route
abstraction, browser feature, outside-Core behavior change, or different
assignment blocks sign-off pending plan update and user approval.

Record each pass, findings, fixes, rerun commands, residual risks, and final
approval status in Execution Evidence.

## Acceptance Criteria

This plan is complete when:

- all thirteen exact stage prefixes exist as independent full bundles in the
  current workspace `.env`;
- `BOUNDARY_CORE_LLM` is absent from active source, tests, current docs,
  Control Console routes, diagnostics, and `.env`;
- generic `COGNITION_LLM` remains unchanged for outside-Core consumers;
- `CognitionCoreServicesV2` exposes only the thirteen stage config fields;
- every initial call, retry, repair, trace, and diagnostic uses the exact
  owning stage route;
- the source-value assignment table is implemented exactly;
- prompts, payloads, schemas, call counts, retry caps, and DAG scheduling are
  unchanged;
- every edited CJK-bearing module compiles and every prompt body is unchanged;
- missing required stage configuration fails startup;
- the existing Control Console route editor lists and operates every stage
  route without API, storage, or static-framework changes;
- credentials and raw `.env` file contents remain absent from tracked files,
  output, browser payloads, screenshots, and evidence; only the existing
  sanitized route diagnostics are exposed;
- focused tests, full non-live regression, collection, syntax, static,
  config-import, startup, health, and browser gates pass;
- parent multi-pass code review is approved with no unresolved finding;
- registry and archive lifecycle records are current.

## Execution Evidence

- Focused baseline failures:
  - `tests/test_cognition_core_v2_stage_model_routing.py`: 1 passed and 9
    failed before production edits. Failures were the expected missing
    thirteen-field service contract and stage attributes; collection and the
    existing aggregate-shape assertion remained healthy.
  - Configuration contract: 2 expected failures for absent required stage
    constants and missing-variable startup enforcement.
  - Route-report contract: 2 expected failures for the stale boundary row and
    absent stage service fields.
  - Control Console contract: 3 expected failures for the stale 14-route
    catalog versus the required 26-route catalog and absent stage descriptors.
  - Documentation contract: 1 expected failure for absent stage route names.
  - `git diff --name-only -- src` returned no paths after the baseline runs.
- Workspace environment prepopulation:
  - All thirteen named stage routes contain one `BASE_URL`, `API_KEY`,
    `MODEL`, `MAX_COMPLETION_TOKENS`, and `THINKING_ENABLED` entry.
  - Presence, exact assigned-source equality, explicit `8192` completion
    budget, and explicit `false` thinking state validated true for every
    route without emitting configured values.
  - Generic `COGNITION_LLM` and pre-cutover `BOUNDARY_CORE_LLM` required
    fields remained present.
  - Duplicate audit reported every field exactly once; `git check-ignore
    .env` returned `.env`; the pre-cutover config import returned
    `pre_cutover_config_import=ok`.
- Change-surface correction:
  - Added `README_CN.md` to the current-documentation list because the
    existing documentation-parity contract requires the English and Chinese
    top-level route documentation to move together. This preserves the
    approved route contract and introduces no behavior or feature change.
- Production-code subagent:
  - Native production subagent `Carver`
    (`019fa260-43f5-7c80-bc5d-6d0b0fda047d`) changed exactly the eleven
    planned production Python files and then closed.
  - Implemented all thirteen required config bundles, the exact service
    fields, owning call/repair/trace selection, startup diagnostics, and
    ordinary Control Console descriptors.
  - Per-file and aggregate `py_compile` commands exited zero. The subagent
    reported zero static-console changes and zero whitespace errors.
  - Parent static audit found zero production legacy-route or aggregate-field
    matches and exactly the planned eleven production files.
  - Parent CJK-token count and SHA-256 comparison matched the pre-edit
    baseline for all six prompt-bearing modules.
  - Parent cutover removed three legacy `.env` entries before the first
    new-code import; zero remain. All thirteen stage bundles and generic
    cognition remained complete in the redacted validation.
- Focused verification:
  - The first post-cutover config import and service construction reported
    thirteen service fields, thirteen unique routes, and no legacy config
    constant.
  - Initial focused run: 90 passed and two direct helper consumers failed
    solely because they omitted the new explicit `config=` argument.
  - Updated those two planned test consumers; rerun passed 92 tests.
- Consumer and documentation migration:
  - Direct service fixtures, action/resolver namespaces, the closed
    four-factor matrix harness, six live-marked collection consumers, and
    four subprocess environment allowlists use the canonical stage fields.
  - The obsolete aggregate sensitivity live-test module is deleted.
  - All six current documents contain all thirteen route names; top-level
    documentation parity passed.
  - Active `README.md`, `README_CN.md`, `docs`, `src`, and `tests` searches
    returned zero legacy-route and zero exact aggregate-field matches.
  - `git diff --name-only -- src/control_console/static` returned no paths.
- Verification-discovered Control Console capacity defect:
  - The first focused portal run produced 21 failures from one registry
    construction error: the exact route catalog exceeded both the existing
    64-character config-field-key limit and the 64-field descriptor limit.
  - The generated catalog contains 78 fields and has a 73-character maximum
    field key. `src/control_console/service_config.py` was added to the change
    surface so both generic bounded limits can be raised to 80 without
    changing route names, API contracts, storage, or static rendering.
- First full non-live regression:
  - Result: 3,531 passed, 10 failed, 3 skipped, and 830 deselected.
  - One deployment failure showed that `docker-compose.yml` still forwarded
    the obsolete boundary route and omitted all required stage route
    variables.
  - Six web-agent subprocess failures came from assigning the placeholder
    string `configured` to newly listed integer and Boolean generation
    variables.
  - Three failures were stale baseline assertions: bootstrap exposes reusable
    cognition graphs at the documented top level, recent audit entries are
    projected action groups with `event_types`, and bounded
    decontextualizer exhaustion preserves the input with empty referents.
- Full non-live regression:
  - Final authoritative command:
    `venv\Scripts\python.exe -m pytest -q -m "not live_llm and not live_db"
    --tb=short`.
  - Result: 3,541 passed, 3 skipped, 830 deselected, and 1 warning in
    243.18 seconds.
  - Collection compatibility completed with 3,543 of 4,374 tests collected
    and 831 deselected before the final test-name reconciliation; there were
    zero import, required-config, or stale-field collection errors.
  - Focused routing and authority verification passed 92 tests; Core
    integration passed 27 tests with 4 deselected; the mechanically migrated
    assignment matrix passed 9 tests; config and route-report verification
    passed 61 tests; the portal suite passed 24 tests; documentation parity
    passed 7 tests.
  - All changed Python files passed `py_compile`. The final full run also
    confirmed the three verification-discovered baseline assertion
    reconciliations.
- Static contract gates:
  - Active searches returned zero `BOUNDARY_CORE_LLM` matches, zero exact
    aggregate service-field matches, and zero `dense` or `moe` assignment
    terms in the production routing scope.
  - The final static Control Console diff contains exactly `console.js` and
    `console.css` for the rendered selected-tile repair; `index.html` and the
    framework remain unchanged. `git diff --check` reported no whitespace
    errors, and `git check-ignore .env` confirmed the workspace environment
    file is ignored.
  - Zero-context production diff inspection found only config selection,
    invocation, and trace wiring. SHA-256 comparisons matched the pre-edit CJK
    prompt sequences in all six prompt-bearing stage modules.
  - The Docker deployment configuration test passed. A direct
    `docker compose config --quiet` invocation was unavailable because this
    host has no Docker CLI.
- Config import and route report:
  - Workspace `.env` import constructed exactly thirteen stage config fields
    and thirteen unique stage routes. Every route had a normalized endpoint,
    model id, explicit 8,192-token budget, and `thinking=false`; credentials
    remained absent from output.
  - The sanitized startup route report contained every stage exactly once,
    retained generic `COGNITION_LLM`, omitted `BOUNDARY_CORE_LLM`, and
    contained no credentials.
- Brain startup and health:
  - The originally supplied `8764` console process was replaced at the
    user's instruction after exact PID and command-line validation. One
    controlled console then bound `http://127.0.0.1:8764/` from this
    workspace.
  - An authenticated Playwright session opened Services and used the Brain
    `Start` control. The visible lifecycle moved from `stopped` to `running`
    with notice `Brain service started.` and zero browser console or page
    errors.
  - `http://127.0.0.1:8000/health` returned status `ok`; the listener belongs
    to a workspace Python process. No chat turn or model-generation request
    was sent.
- Control Console browser validation:
  - The in-app Browser failed with `No browser is available`; its browser
    inventory was `[]`. Validation therefore used installed system Chrome
    through Playwright, as permitted by the plan and console skill.
  - Page identity was `一之濑明日奈 Control Console` at
    `http://127.0.0.1:8764/`. The process-local session authenticated as
    `local_operator`, Services was visible, and Brain remained `running`.
  - The model-route payload contained 26 total routes and exactly 13 routes in
    `Cognition Core V2`. Group filtering rendered all thirteen expected labels
    with no missing or extra row; the search interaction `authorization`
    produced exactly Action authorization and Resolver authorization, then
    clearing it restored all thirteen rows.
  - Appraisal: event and agency selected its configured first-source model;
    Appraisal: relationship and social selected its configured second-source
    model. Each editor showed 8,192 max completion tokens and thinking
    disabled.
  - Model discovery was exercised through both selected rows. Both requests
    returned HTTP 200 with terminal status `available`, nine provider models,
    and a picker whose selected value matched the effective route model.
  - The route payload exposed zero sensitive field names. The cleared login
    field was hidden, the operator token was absent, and comparison against
    the one configured API-key value found zero matches in the DOM or route
    payload.
  - Desktop width 1,440 had document and body scroll widths of 1,440. Mobile
    width 390 had document and body scroll widths of 390; the route matrix,
    editor, and controls remained bounded and readable in both inspected
    screenshots.
  - Console errors: 0. Page errors: 0.
  - Reviewed local screenshots:
    `test_artifacts/cognition_core_v2_stage_routes_desktop.png` and
    `test_artifacts/cognition_core_v2_stage_routes_mobile.png`.
  - User visual review rejected this initial sign-off: the selected route tile
    hid its model/source/family/thinking rows and retained a heavy double
    border, leaving an obvious empty-box artifact. The final screenshot also
    left the `Cognition Core V2` group filter active and therefore hid the
    other thirteen configured routes despite the API reporting 26. Stage 6
    was reopened for renderer repair and an unfiltered final sign-off.
  - A focused static-surface regression contract failed before the repair,
    then passed after `renderBrainRouteTile(...)` retained model and badge
    metadata for selected routes and selected styling switched to the
    established `nav-active` palette.
  - The affected Control Console suite passed 24 tests. At that visual
    iteration, the corrected browser loaded the updated JavaScript and CSS,
    kept the relationship/social selected tile at a compact height, displayed
    its model plus `default`, `gemma4`, and `standard` badges, and returned
    model discovery status `available`.
  - Second-pass screenshots replaced the initially rejected images with
    `Group = all` and all
    26 route tiles visible, including both Core V2 and existing non-Core
    routes. Desktop and mobile document widths remained exactly 1,440 and 390
    pixels respectively, with zero console or page errors:
    `test_artifacts/cognition_core_v2_stage_routes_desktop.png` and
    `test_artifacts/cognition_core_v2_stage_routes_mobile.png`.
  - User visual review rejected the second-pass desktop screenshot because
    the Brain runtime panel still formed a large bordered blank region beside
    the route matrix. Live DOM measurement reproduced the defect at a
    1,440-pixel viewport: `.brain-runtime-panel` and
    `.brain-routes-panel` were both 1,237 pixels high while the runtime panel
    contained only its compact lifecycle controls. Computed grid
    `align-items` was `normal`; the mobile one-column layout did not exhibit
    the stretch.
  - The same audit found stale E2E expectations in
    `test_page_navigation_e2e.py` and
    `test_running_console_signoff_e2e.py` that required the selected route
    tile to contain zero model-code elements. Stage 6 remained open while the
    rendered height defect and both stale contracts were corrected and new
    screenshots passed parent visual review.
  - Parent visual iteration 1 applied `align-items: start` and reduced the
    runtime panel from 1,237 pixels to 230 pixels, but rejected the resulting
    screenshot because the two-column Brain layout still left a full-height
    empty left column below that compact panel. The accepted rendered contract
    now requires the runtime and route panels to stack with equal horizontal
    bounds at desktop width as well as mobile width.
  - The tightened static and rendered E2E contracts failed before the final
    layout correction because the desktop runtime and route panels differed
    by about 402 pixels on the horizontal axis. Changing the owning
    `.brain-service-layout` grid to one column made both focused tests pass.
  - Parent visual iterations then inspected bright desktop, bright mobile,
    the 901/900-pixel breakpoint, and dark desktop. Width audits at 1,440,
    1,024, 901, 900, 768, and 390 pixels reported zero horizontal overflow,
    26 rendered routes, one selected model line, three selected metadata
    badges, and equal runtime/route horizontal bounds.
  - The affected Control Console suite passed 24 tests. The edited Python
    tests passed `py_compile`. The opt-in running-console information-contract
    signoff passed against `http://127.0.0.1:8764/` with 26 current route
    values, zero browser console messages, zero failed HTTP responses, zero
    failed browser requests, and no LLM call.
  - The affected relationship/social route reached terminal discovery with
    nine provider models and its effective model selected. Brain health
    remained `ok`; browser console errors and page errors remained zero.
  - Parent reviewed and accepted the final canonical screenshots:
    `test_artifacts/cognition_core_v2_stage_routes_desktop.png` and
    `test_artifacts/cognition_core_v2_stage_routes_mobile.png`. Both show the
    compact runtime panel above all 26 route cards, the populated selected
    relationship/social tile, and the full route editor without the rejected
    blank region.
  - User review reopened the visual gate again because the selected tile still
    drew a second empty rectangle inside its normal border. Parent live-DOM
    inspection found three populated children and zero empty child elements,
    then isolated the rectangle to the selected rule's additional one-pixel
    inset `box-shadow`.
  - The route-specific inset shadow was removed while the selected background
    and one-pixel border were retained. Static, deterministic browser, and
    running-console regressions now require the selected route's computed
    `box-shadow` to be `none`, alongside one non-empty model line and three
    metadata badges.
  - Parent visual iteration inspected a close selected-tile crop plus bright
    desktop, bright mobile, and dark desktop captures. Every view has one clean
    selected boundary, a visible model, three badges, and zero empty children.
    The bright desktop and mobile captures contain all 26 routes; measured page
    widths are exactly 1,440 and 390 pixels with zero overflow and zero console
    errors.
  - Final artifact paths:
    `test_artifacts/selected_route_tile_inspection.png`,
    `test_artifacts/cognition_core_v2_stage_routes_desktop.png`,
    `test_artifacts/cognition_core_v2_stage_routes_mobile.png`, and
    `test_artifacts/cognition_core_v2_stage_routes_desktop_dark_iteration5.png`.
  - Final visual-gate reruns passed 11 static-surface tests, all 7 page
    navigation E2E tests, and the opt-in running-console signoff. Its redacted
    summary reports `pass`, 26 routes, 26 populated current route values, zero
    browser console messages, zero failed HTTP responses, zero failed browser
    requests, and zero LLM calls.
- Parent multi-pass code review:
  - Pass one reviewed the thirteen-field service boundary, required
    configuration loading, production connector construction, all appraisal,
    goal, verifier, collapse, planning, action-authorization, and
    resolver-authorization calls, retries, repairs, traces, route diagnostics,
    prompt-bearing diffs, and retained outside-Core generic cognition
    consumers.
  - Pass one found that the focused suite did not directly prove the mixed
    required-selection repair path.
    `test_selection_bid_repair_uses_goal_route_and_verifier_recheck` now proves
    verifier, owning-goal repair, verifier ordering.
  - Pass-one interim rerun: 93 focused routing/contract/authority tests passed.
    Zero-context inspection showed only config selection, invocation, and
    trace wiring in all six prompt-bearing modules; no prompt body, schema,
    retry cap, call count, or DAG edge changed. Active searches returned zero
    aggregate Core fields, zero boundary routes, and zero model-role keywords
    in production routing scope.
  - Pass two reviewed Docker forwarding, descriptor capacity and validation,
    all 26 Control Console route descriptors, authenticated/CSRF-protected
    override and reset flows, model discovery redaction, JavaScript escaping,
    selected-tile rendering, responsive CSS, deterministic and live-marked
    fixture migration, historical harness compatibility, current
    documentation, and the two plan lifecycle records.
  - Pass two found one formatting regression in
    `test_cognition_core_v2_live_llm.py`: a cleanup-call closing parenthesis
    had moved to column zero, and the new route-field constant sat between
    helper functions. The indentation was restored, the constant moved into
    the module constant section, and the CJK-bearing file passed immediate
    `py_compile`.
  - Pass-two reruns passed 119 deployment, configuration, route-report,
    Control Console API/security, static-surface, documentation, historical
    harness, and reconciled baseline tests, plus all 7 rendered page-navigation
    E2E tests. Parent re-inspected the canonical bright desktop and mobile
    screenshots, the dark desktop screenshot, and the running-console Services
    screenshot at original resolution. All show populated selected tiles,
    complete route metadata, all 26 configured routes, equal panel bounds, and
    no rejected blank region.
  - Pass three re-read the complete routing diff, then reran static,
    environment, syntax, catalog-parity, prompt-preservation, focused,
    collection, and browser gates. The redacted `.env` audit found thirteen
    complete five-field bundles, exact source assignment, two distinct source
    identities, no duplicate names, and no boundary route. Service, portal,
    and diagnostic catalogs agreed on all thirteen stage routes; the portal
    retained all 26 routes, 78 editable fields, and a measured maximum
    73-character field key. All 53 changed Python files compiled, and every
    CJK-bearing prompt line remained byte-equivalent to `HEAD`.
  - Pass three caught an incorrect preliminary cleanup from pass one: the
    private generic cognition connector config is the existing outside-Core L3
    surface binding, not dead code. Full collection exposed the mistake through
    the L3 import. The exact generic binding and its five config imports were
    restored, and
    `test_l3_surface_retains_the_generic_cognition_route` now freezes that
    ownership. The final focused route file passed 12 tests and collection
    passed with 3,551 of 4,382 tests collected and 831 live-marked tests
    deselected.
  - The generic `COGNITION_LLM` route remains active for three existing
    outside-Core consumers: L3 surface construction, memory lifecycle, and
    internal-monologue residue.
  - The final complete non-live rerun initially exposed two missing baseline
    owner-matrix rules for the completed character-name cutover's dispatcher
    and message-envelope README paths. The user authorized their narrow test
    fixture remediation. The new deterministic mappings preserve the existing
    delivery and intake ownership boundaries without changing routing or
    runtime behavior; the focused owner-matrix test then passed.
  - The authoritative final command
    `venv\Scripts\python.exe -m pytest -q -m "not live_llm and not live_db"
    --tb=short` completed with exit code zero in 262.7 seconds. No live model
    or database test ran. Collection completed with 3,558 of 4,389 tests
    collected and 831 live-marked tests deselected.
  - Final static and syntax gates passed: the listed route modules compiled,
    no boundary or aggregate routes remained, no dense/MoE assignment terms
    appeared in production routing scope, static changes remained limited to
    `console.js` and `console.css`, `git diff --check` passed, and `.env`
    remained ignored.
  - Parent review approval: pass one confirmed stage selection, repairs,
    traces, call counts, and generic outside-Core retention; pass two confirmed
    deployment, descriptors, security, rendered-console contracts, tests, and
    documentation; pass three confirmed the final diff and remediation. No
    unresolved contract, security, quality, visual, or scope finding remains.
- Final lifecycle sign-off:
  - Status changed to `completed`; this record moves to
    `development_plans/archive/completed/short_term/` and the registry moves
    it from Active Short-Term Plans to Completed Short-Term Records.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Missing new variables prevent startup | Prepopulate all bundles while old config remains usable, then remove the boundary bundle at the new-code cutover. | Redacted env validation and config-import gate. |
| A call uses the wrong route while traces claim another | Pass or select one exact config per stage and use it for invocation and trace. | Focused fake-invoker and trace tests. |
| Active-goal branch output loses quality on its assigned service | Keep ordinary goal, required-selection verification, collapse, planning, and authorization on the current cognition source values. | Existing limited evidence plus exact assignment tests; future quality work remains separate. |
| Portal catalog drifts from runtime routes | Use the same exact prefix list in deterministic parity tests. | Control Console route and documentation tests. |
| Generic cognition consumers break | Retain existing generic constants and outside-Core modules unchanged. | Static diff review and full non-live regression. |
| Boundary aliases survive and hide incomplete migration | Enforce zero active matches and omit all fallback behavior. | Static grep and parent multi-pass review. |
| More route tiles impair portal usability | Reuse filtering and responsive matrix, then validate rendered interaction and overflow. | Browser gate and screenshot. |
| Secret values enter evidence | Validate and report names/presence only; keep `.env` ignored and API keys redacted. | Ignore check, output review, browser payload review, parent multi-pass review. |
| Historical harness no longer constructs services | Mechanically map its four factors onto groups of new fields without rerunning the corpus. | Focused historical-harness tests. |
| Routing is mistaken for measured throughput | State only existing task concurrency and defer performance claims. | Documentation and review gate. |
