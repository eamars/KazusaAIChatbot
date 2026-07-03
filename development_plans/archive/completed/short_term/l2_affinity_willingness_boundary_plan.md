# l2 affinity willingness boundary plan

## Summary

- Goal: Add a config-controlled L2 task-taking willingness boundary for
  requests that feel too much for the current relationship, current mood, or
  scene vibe, so L2 can refuse, deflect, tease, or offer a smaller scope before
  L2d action routing when the feature is enabled.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`, `py-style`, `cjk-safety`,
  `test-style-and-execution`
- Overall cutover strategy: compatible prompt-family rollout behind one
  fail-fast environment flag. The feature defaults on and can be disabled by
  setting the flag to `false`; no new agent, no hidden tool removal, and no
  action-spec contract migration.
- Highest-risk areas: exposing operational effort/cost too early, turning
  affinity, mood, or scene vibe into deterministic classification, letting L2d
  become the willingness decision-maker, leaking the enable flag into L2d as a
  gate, accidentally suppressing ordinary speech, over-refusing simple help,
  and breaking the route-only L2d/background-work boundary.
- Acceptance criteria: demanding task-like requests can be refused or deflected
  by L2 using existing stance/intent fields when relationship, mood, or scene
  vibe makes taking them on feel wrong; ordinary speaking remains governed by
  existing cognition and dialog behavior; L2d receives no affinity-gate,
  mood-gate, vibe-gate, or effort fields; private or background actions are
  not selected when upstream cognition has not accepted taking on the request;
  focused deterministic and one-at-a-time live LLM evidence supports the
  behavior; disabled mode preserves legacy prompt behavior.

## Context

The current live response path is:

```text
adapter/debug client
  -> brain service
  -> queue/intake
  -> RAG
  -> cognition_chain_core L1/L2/L2d
  -> dialog / action-spec materialization
  -> persistence / scheduler / background workers
```

`cognition_chain_core` owns the reusable cognition chain:

```text
L2a Conscious Core
  -> L2b Boundary Core
  -> L2c Judgment Core
  -> L2d Action Selection
```

Source inspection for this review found:

- `CognitionChainInputV1.character` already includes `mood` and
  `global_vibe`.
- graph residue already includes `vibe_check` and `visual_vibe`.
- L2a currently receives `character_mood` and `global_vibe`.
- L2b Boundary Core currently receives affinity context, emotional appraisal,
  interaction subtext, reason-to-respond, channel topic, and source payloads,
  but not direct `character_mood` or `global_vibe`.
- L2c Judgment Core currently receives affinity context and Boundary Core
  fields, but not direct `character_mood`, `global_vibe`, or `vibe_check`.
- L2d already owns route-only action selection and should follow upstream
  cognition rather than re-decide character stance.

The desired behavior is:

```text
Given how the character feels about this user, her current mood, the current
scene, and this request right now, is this something she is willing to take on?
```

This feature is not a general "does she speak at all" gate. Normal speech
continues to be influenced by mood and vibe through the existing cognition and
dialog path. The new boundary is only about whether the character takes on a
task-like request, commitment, private action, future action, or sustained help.

If the answer is no, L2 should settle the turn into refusal, deflection,
banter, or smaller-scope help using the existing stance/intent vocabulary. L2d
then routes the visible response that follows from that settled cognition and
does not schedule private, future, or background work for a request L2 has not
accepted.

This plan protects resource-heavy work after L2d action selection. It does not
avoid upstream RAG, decontextualization, relevance, L1, L2a, L2b, or L2c cost.
Avoiding pre-L2d cost is a separate design with a larger blast radius.

Disablement source mapping found during design: `config.py` owns fail-fast
boolean env settings, `docs/HOWTO.md` documents service behavior flags,
`CognitionChainInputV1.runtime_context` already carries runtime toggles,
`build_cognition_chain_input_from_global_state(...)` projects caller runtime
context, and `chain._state_from_chain_input(...)` maps runtime context into
internal stage state before L2/L2d run. L2d payload construction does not need
the new flag.

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, updating,
  or signing off this plan.
- `local-llm-architecture`: load before changing L2, L2d, prompt flow,
  graph routing, LLM call budgets, prompt payloads, or context contracts.
- `no-prepost-user-input`: load before changing logic that accepts, rejects,
  routes, persists, or acts on user requests.
- `debug-llm`: load before local/live LLM runs, prompt comparisons, trace
  inspection, or human-readable LLM quality artifacts.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files or tests containing CJK
  prompt strings or CJK test data.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or final
  reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.
- Do not execute implementation steps while `Status` is `draft`.
  Implementation requires user approval, status `approved` or `in_progress`,
  and a direct user instruction to execute.
- Do not add a new agent or LLM call for this feature.
- Add exactly one feature flag:
  `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED`, parsed with fail-fast boolean
  rules and defaulting to `true`.
- Do not add numeric affinity thresholds, mood thresholds, vibe thresholds,
  database fields, migrations, per-character settings, per-tool settings, or
  secondary feature flags for this feature.
- Do not hide, remove, or conditionally suppress action affordances from L2d
  to manufacture the gate.
- Do not expose operational concepts to L2 or L2d as decision criteria:
  `resource heavy`, `effort`, `tool cost`, `background_work`,
  `background_work_request`, `complex_task_resolution`, worker names, tool
  names, queue internals, or an `affinity threshold`.
- Do not add deterministic keyword classification over raw user input,
  `decontextualized_input`, conversation text, or action summaries to decide
  whether a request is too much.
- L2 owns semantic task-taking willingness. L2d follows the settled L2
  `logical_stance`, `character_intent`, and `judgment_note`.
- L2d must not receive a new affinity-gate, mood-gate, vibe-gate,
  patience-score, willingness-score, feature-enabled, effort, cost, or
  complexity field.
- L2b/L2c may receive existing prompt-safe semantic descriptors from current
  graph state, but those descriptors must not become new public contract
  fields and must not expose raw telemetry or numeric thresholds.
- Do not let this feature become a general speech suppression rule. Refusing
  to take on a task-like request should normally still produce a visible
  `speak` route with refusal, teasing, deflection, or a smaller offer.
- `no-response` remains owned by existing silence/no-response logic. Do not
  introduce "bad mood means no answer" behavior as part of this plan.
- Keep L2d route-only. It must not generate final dialog wording, worker task
  briefs, tool parameters, scheduler plans, or private handler parameters.
- Deterministic code continues to own validation, action materialization,
  queue persistence, adapter delivery, limits, and schema safety. It must not
  rewrite semantic willingness after the LLM stages.
- The feature flag may select legacy vs willingness-aware L2/L2d prompt
  contracts before model invocation. It must not post-process, override, drop,
  or reinterpret LLM outputs after model invocation.
- If implementation discovers that a deterministic runtime backstop is needed
  for resource safety, stop and update this plan for approval before adding it.
  This plan proves the prompt/contract behavior first.

## Must Do

- Update L2 Boundary Core prompt flow so relationship context, current mood,
  scene vibe, and recent interaction vibe contribute to a character-native
  task-taking willingness boundary:
  - whether the request feels too demanding, too familiar, too intimate, too
    controlling, or too much to take on for this person right now;
  - whether the current mood or scene makes taking on the request feel natural,
    annoying, badly timed, playful, risky, or inappropriate;
  - whether the character should accept, soften, tease, deflect, refuse, or
    offer a smaller scope;
  - without describing backend work, hidden tools, cost, queueing, or effort.
- Gate the new L2b/L2c/L2d willingness-aware prompt behavior behind
  `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED`.
- When `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED=false`, preserve the legacy
  L2b/L2c/L2d prompt behavior and do not add the new mood/vibe willingness
  descriptors to L2b/L2c payloads.
- When `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED=true`, enable the new
  willingness-aware L2b/L2c prompt behavior, allowed L2b/L2c semantic
  descriptor projection, and L2d route-following prompt guidance.
- Add `task_willingness_boundary_enabled: bool` to
  `CognitionChainInputV1.runtime_context` and internal stage state solely as a
  prompt-contract selector.
- Update L2b's internal human payload only as needed to expose existing
  prompt-safe semantic descriptors already present in graph state, such as
  `character_mood`, `global_vibe`, `vibe_check`, and `visual_vibe`.
- Keep Boundary Core output shape compatible. Reuse existing fields such as
  `boundary_summary`, `acceptance`, `stance_bias`, `pressure_policy`, and
  `trajectory` instead of adding an affinity/mood/vibe gate object.
- Update L2 Judgment Core prompt flow so the final `logical_stance`,
  `character_intent`, and `judgment_note` honor the task-taking willingness
  boundary when it is stronger than the raw helpful impulse.
- Update L2c's internal human payload only as needed to expose existing
  prompt-safe semantic descriptors or the L2b willingness summary. Do not add
  new public state or output fields.
- Preserve normal helpfulness for simple, low-pressure, low-commitment
  requests even when affinity is low or mood/scene vibe is not ideal.
- Preserve normal speaking behavior. Mood and vibe may affect tone, brevity,
  warmth, teasing, irritation, or softness, but they must not block ordinary
  speech merely because the character is reluctant to take on a task.
- Update L2d action-selection prompt flow so it treats refusal, deflection,
  teasing, and non-commitment as settled upstream cognition. In those cases,
  L2d should normally choose a visible `speak` route that matches the stance,
  not private/future/background work. `no-response` remains available only for
  existing silence/no-response cases, not as the default effect of this
  willingness boundary.
- Keep the `background_work_request` affordance visible in L2d's capability
  roster. The router should know the tool exists, but should not select it
  when L2 has not chosen to take the request on.
- Add deterministic prompt-contract tests that verify the relevant L2 prompt
  text does not introduce operational gate language, hidden tool names, numeric
  thresholds, or effort categories.
- Add focused deterministic tests around payload shape:
  - config defaults to enabled, parses true/false, and rejects invalid values;
  - connector/runtime-context mapping carries the flag into the cognition chain;
  - L2b/L2c may receive existing semantic mood/vibe descriptors;
  - L2b/L2c omit the new willingness descriptors when the flag is disabled;
  - L2d receives no affinity-gate, mood-gate, vibe-gate, feature-enabled,
    effort, cost, or complexity field.
- Add fake-LLM or parser-level regression coverage for:
  - low-affinity demanding request can resolve to refusal, deflection, or
    banter through existing L2 stance/intent fields;
  - low-affinity simple request can still resolve to ordinary help;
  - bad-mood or tense-scene simple speech still routes to visible speech when
    existing silence logic does not apply;
  - bad-mood or tense-scene demanding task-like request may resolve to refusal,
    deflection, or a smaller offer through existing L2 fields;
  - high-affinity demanding request is not automatically refused;
  - L2d route selection follows a non-accepting task-taking outcome with a
    visible `speak` route, not background/private action scheduling.
- Add one-at-a-time live LLM inspection cases, marked `live_llm`, for:
  - low-affinity simple help;
  - low-affinity demanding sustained help;
  - bad-mood or tense-scene simple speaking;
  - bad-mood or tense-scene demanding sustained help;
  - high-affinity demanding sustained help;
  - low-affinity request that would otherwise be eligible for background work.
- Record prompt-flow review notes, live LLM observations, plan-review findings,
  and review-fix evidence in this plan's `Execution Evidence` before sign-off.

## Deferred

- Do not add a new L2b2, affinity-gate, mood-gate, vibe-gate, request-effort,
  or access-control agent.
- Do not add threshold configuration, character-profile schema fields, database
  fields, migrations, or feature flags beyond
  `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED`.
- Do not add `affinity_task_gate`, `effort_score`, `complexity_score`,
  `mood_gate`, `vibe_gate`, `patience_score`, `willingness_score`, or similar
  fields to `CognitionChainInputV1`, graph state, L2 output, or L2d payloads.
- Do not integrate `complex_task_resolver` or expose
  `complex_task_resolution` to L2d as part of this plan.
- Do not redesign action-spec, background-work queueing, worker routing,
  scheduler, adapters, dialog rendering, RAG, consolidation, or persistence.
- Do not add pre-L2d resource gating, RAG pruning, decontextualizer pruning, or
  relevance-gate redesign.
- Do not add control-console toggles, per-user/per-character overrides, ops
  runtime-status fields, or live config reload for this flag in this plan.
- Do not add compatibility shims, parallel prompt paths, fallback mappers, or
  legacy aliases.
- Do not rely on fixture-specific prompt wording, keyword lists, or hardcoded
  examples to force the desired behavior.

## Cutover Policy

Overall strategy: compatible prompt-family rollout behind one explicit
environment flag. Disabled mode preserves legacy prompt behavior; enabled mode
uses the willingness-aware prompt contract.

| Area | Strategy | Policy |
|---|---|---|
| Config | compatible | Add `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED`, default `true`, using fail-fast boolean parsing. |
| Runtime context | compatible | Add `task_willingness_boundary_enabled: bool` as the only new cognition-chain public field. |
| L2 Boundary Core prompt | compatible | Legacy prompt path remains for disabled mode; enabled mode uses a coherent task-taking willingness prompt. |
| L2 Boundary Core internal payload | compatible | Add allowed mood/vibe descriptors only when the flag is enabled. |
| L2 Judgment Core prompt | compatible | Legacy prompt path remains for disabled mode; enabled mode reconciles final stance/intent with Boundary Core task-taking willingness. |
| L2 Judgment Core internal payload | compatible | Add allowed descriptors or Boundary Core summary fields only when the flag is enabled. |
| L2d action-selection prompt | compatible | Legacy prompt path remains for disabled mode; enabled mode clarifies that settled task refusal normally routes visible speech. |
| L2d action-selection payload | unchanged | Do not add gate, mood, vibe, threshold, effort, or cost metadata. |
| Public cognition contracts | compatible | Only `RuntimeContextV1.task_willingness_boundary_enabled` is added. No output or action contract changes. |
| Action-spec capabilities | unchanged | Keep existing action kinds and prompt-safe affordance projection. |
| Background work | unchanged | No worker, queue, or router behavior change in this plan. |
| Tests | compatible/additive | Add focused regression and live inspection coverage without replacing existing L2d/action tests. |

Cutover policy enforcement:

- If an area is `compatible`, preserve only the explicit old/new prompt
  behavior listed in the table.
- If an area is `unchanged`, preserve only the surface listed in the table.
- Any change to cutover policy requires user approval before implementation.

Operational rollback is config-level: set
`COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED=false` and restart the service so
legacy prompt behavior is used. Code rollback remains available by reverting
the config, runtime-context, prompt, and test changes together.

## Target State

When `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED=true`, demanding task-like
requests follow this path:

```text
User asks for something demanding, familiar, sustained, private, or future-bound
  -> L2b reads relationship, current mood, and scene vibe as character
     willingness pressure
  -> L2c settles final stance/intent
  -> L2d sees that the character is not taking it on
  -> L2d selects visible speech only
  -> dialog renders a natural refusal, tease, deflection, or smaller offer
```

For ordinary requests:

```text
User asks for simple help or normal conversation
  -> L2b does not over-treat low affinity, bad mood, or scene tension as
     rejection
  -> L2c may accept, answer, or speak normally
  -> L2d can select the appropriate existing action route
```

The completed feature should feel like the character has better judgment about
what the relationship, mood, and moment entitle the user to ask from her. It
should not feel like a tool permission system or a general refusal to talk.

When `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED=false`, the legacy L2/L2d
prompt behavior is used. The new flag should not affect normal speech,
available actions, action-spec materialization, background workers, RAG, or
dialog rendering.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Operational switch | Add `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED`, default `true`. | The feature should be active by default while keeping a direct `false` rollback for deployments that do not want it. |
| Semantic owner | L2 owns task-taking willingness. | Keeps the judgment in cognition where relationship, boundary, and character stance already live. |
| Mood/vibe handling | Use existing mood and scene-vibe descriptors as L2 modifiers. | The user wants gut-feeling reluctance, not a separate mood permission system. |
| L2d role | L2d follows settled cognition and remains route-only. | Prevents L2d from becoming a second gate or hiding tools from the character. |
| Public contracts | Add one runtime-context boolean; keep action contracts unchanged. | The reusable cognition core needs a caller-projected switch, while L2d/action-spec must not see gate metadata. |
| Deterministic backstop | Do not add one in this plan. | `no-prepost-user-input` forbids deterministic semantic overrides; prompt/schema tests and live evidence come first. |
| Normal speech | Preserve ordinary speaking behavior. | The feature targets task-taking reluctance, not whether the character speaks at all. |
| Pre-L2d cost | Defer. | This plan protects post-L2d private/background actions only. |

## Contracts And Data Shapes

- Config: add
  `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED = _bool_from_env(..., "true")`.
- `CognitionChainInputV1.runtime_context`: add
  `task_willingness_boundary_enabled: bool`.
- L2 Boundary Core output schema: unchanged.
- L2 Judgment Core output schema: unchanged.
- L2d action-selection input payload: unchanged.
- Action-spec capability registry: unchanged.
- Background-work request schema and queue documents: unchanged.

Allowed internal L2b/L2c human-payload additions use existing prompt-safe graph
state only. The implementation may add these keys to L2b and/or L2c payloads
only when the flag is enabled and the corresponding source value already
exists:

```json
{
  "character_mood": "semantic mood string",
  "global_vibe": "semantic global scene string",
  "vibe_check": "semantic current-turn vibe string",
  "visual_vibe": ["semantic visual vibe descriptor"]
}
```

Forbidden shapes:

```json
{
  "affinity_task_gate": "...",
  "mood_gate": "...",
  "vibe_gate": "...",
  "effort_score": 0,
  "complexity_score": 0,
  "patience_score": 0,
  "willingness_score": 0,
  "feature_enabled": true,
  "threshold": 0
}
```

The `task_willingness_boundary_enabled` field may exist in `runtime_context`
and internal stage state only. It selects which prompt contract code builds
before model invocation. It must not appear in model-facing human payloads,
L2d human payloads, action requests, action specs, traces sent to consolidation
as a semantic reason, or persisted memory.

Debug traces may inspect existing L2 fields and the allowed internal payload
descriptors, but no new affinity, mood, vibe, score, threshold, or willingness
gate field is introduced.

If implementation finds a real need for a new public field, stop and update
this plan before adding it.

## LLM Call And Context Budget

- No new live response-path LLM calls.
- L2b, L2c, and L2d keep their current stage count and JSON output contracts.
- Before this plan:
  - L2a already receives mood and global vibe.
  - L2b receives affinity and immediate appraisal/subtext but not direct
    character mood/global vibe fields.
  - L2c receives Boundary Core outputs and affinity but not direct mood/vibe
    fields.
  - L2d receives cognition and action affordances.
- After this plan:
  - disabled mode keeps legacy L2b/L2c/L2d prompt behavior;
  - enabled mode allows L2b and L2c to receive existing semantic mood/vibe
    descriptors from graph state when needed for the prompt contract.
  - L2d payload remains unchanged.
- Expected context growth is bounded to a few short semantic strings and one
  small descriptor list. Use a conservative budget cap of 1,000 added
  characters across L2b/L2c combined.
- No model route, completion-token cap, or timeout changes are allowed.
- Live LLM verification must run one case at a time and inspect the full L2 and
  L2d trace before marking a case pass.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/config.py`
  - add `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED` with fail-fast boolean
    parsing and default `true`.
- `docs/HOWTO.md`
  - document the flag in the character/service behavior section.
- `src/kazusa_ai_chatbot/cognition_chain_core/contracts.py`
  - add and validate `RuntimeContextV1.task_willingness_boundary_enabled`.
- `src/kazusa_ai_chatbot/cognition_chain_core/chain.py`
  - map the runtime-context flag into internal stage state.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - project the config flag into cognition-chain runtime context.
- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l2.py`
  - revise Boundary Core and Judgment Core prompt text;
  - add only allowed existing semantic descriptors to L2b/L2c internal human
    payloads when the feature is enabled;
  - preserve legacy prompt behavior when the feature is disabled;
  - preserve output schemas and graph wiring.
- `src/kazusa_ai_chatbot/cognition_chain_core/action_selection_prompt.py`
  - add enabled-mode L2d prompt guidance so action routing follows settled
    refusal or deflection for task-taking requests;
  - preserve legacy prompt behavior when the feature is disabled;
  - do not add gate metadata or hide capabilities.
- `tests/test_config.py`
  - add default true, true/false parsing, and invalid value checks.
- `tests/test_cognition_chain_core_contracts.py`
  - update runtime-context contract validation for the new boolean.
- `tests/test_cognition_chain_connector_mapping.py`
  - prove connector mapping carries the config value into runtime context.
- `tests/test_cognition_prompt_contract_text.py`
  - add prompt-text negative checks for forbidden operational/gate language.
- `tests/test_action_selection_prompt_contract.py`
  - add L2d prompt checks for route-following and no hidden gate metadata.
- `tests/test_action_selection_payload.py`
  - prove L2d payload remains free of affinity/mood/vibe/effort gate fields.
- `tests/test_cognition_chain_core_action_selection.py`
  - add focused route-following coverage if this file is the existing best
    location for fake-LLM action-selection tests.
- `tests/test_l2d_action_selection_cases.py`
  - add or extend deterministic case coverage for refusal-visible-speak
    routing.
- `tests/test_cognition_live_llm_boundary_affinity.py`
  - extend one-at-a-time live inspection coverage.

### Keep

- `src/kazusa_ai_chatbot/action_spec/*`
- `src/kazusa_ai_chatbot/background_work/*`
- `src/kazusa_ai_chatbot/cognition_resolver/*`
- `src/kazusa_ai_chatbot/rag/*`
- adapter modules
- database migrations
- public cognition contracts

## Overdesign Guardrail

- Actual problem: post-L2d private/background/future actions can be selected
  for demanding requests before the character has enough relationship or mood
  willingness to take them on.
- Minimal change: add one fail-fast boolean config flag, project it through
  runtime context, refine existing L2b/L2c/L2d prompts for enabled mode, add
  existing semantic descriptors to L2b/L2c payloads only when enabled, and add
  focused tests.
- Ownership boundaries: L2 owns semantic task-taking willingness; L2d owns
  route-only action selection after L2; L3/dialog owns final wording;
  deterministic code owns schema validation, materialization, queueing,
  scheduling, limits, and delivery.
- Rejected complexity: new agents, new LLM calls, thresholds, scores,
  permission middleware, action capability filtering, public schema fields
  beyond one runtime-context boolean, runtime deterministic semantic backstops,
  pre-L2d resource pruning, background-worker changes, and compatibility
  shims beyond the explicit legacy/enabled prompt selection.
- Evidence threshold: add rejected complexity only after focused deterministic
  and live LLM evidence shows prompt/schema refinement cannot reliably prevent
  post-L2d private/background action selection for non-accepted task-like
  requests, and after the user approves an updated plan.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve this plan's contracts and change surface.
- The responsible agent must not introduce new architecture, alternate rollout
  strategies, compatibility layers, fallback paths, or extra features.
- L2b may interpret relationship context, current mood, scene vibe, and recent
  interaction vibe as pressure, familiarity, intimacy, trust, entitlement,
  timing, patience, and task-taking willingness.
- L2c may convert that interpretation into the final character stance and
  intent.
- L2d may select actions only after reading settled final cognition.
- L2d may not decide that low affinity, bad mood, or scene vibe blocks a task.
  It only follows L2's accepted, refused, deflected, or teasing outcome.
- Dialog/L3 owns final wording and visible softness.
- Deterministic code owns schemas, parser validation, action materialization,
  queue execution, scheduling, limits, and delivery mechanics.
- If the plan and code disagree, preserve the plan's stated ownership intent
  and report the discrepancy before widening scope.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors unless explicitly
  listed in `Must Do`.

## Implementation Order

1. Re-read this plan, `development_plans/README.md`, relevant subsystem
   READMEs, and the current source/test files named in `Change Surface`.
2. Load mandatory skills required for the next step.
3. Parent writes or updates focused deterministic tests for prompt text,
   config parsing, runtime-context mapping, L2b/L2c payload shape, and L2d
   payload shape.
4. Parent runs the focused deterministic tests and records the current baseline
   or expected failure in `Execution Evidence`.
5. Fallback execution agent performs production-code edits in the main session
   because the user explicitly requested execution without subagents.
6. Fallback execution agent adds the config flag and runtime-context mapping.
7. Fallback execution agent updates L2b prompt text and allowed internal
   payload projection for enabled mode while preserving disabled-mode legacy
   behavior.
8. Fallback execution agent updates L2c prompt text and allowed internal
   payload projection for enabled mode while preserving disabled-mode legacy
   behavior.
9. Fallback execution agent updates L2d prompt text so non-accepted task-like
   requests normally route visible `speak` and not private/future/background
   actions when the feature is enabled, while disabled mode keeps legacy prompt
   behavior.
10. Parent reruns focused deterministic tests and records results.
11. Parent adds or updates fake-LLM/parser-level action-selection tests for
    refusal-visible-speak routing.
12. Parent runs the focused action-selection tests and records results.
13. Parent runs live LLM inspection cases one at a time, recording outputs and
    pass judgment.
14. Parent runs broader relevant regression tests.
15. Parent runs static greps and prompt-render checks.
16. Fallback execution agent runs an independent-review-style fresh pass after
    planned verification passes because the user explicitly requested no
    subagents.
17. Fallback execution agent remediates review findings only when fixes stay inside this plan's
    approved change surface, reruns affected verification, and records
    evidence.
18. Parent updates `Execution Evidence` and lifecycle status only after all
    verification and review gates complete.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Fallback execution mode is active for this run because the user explicitly
  requested execution without subagents on 2026-07-02.
- The fallback execution agent owns production code changes, tests,
  verification, execution evidence, review-style diff inspection, remediation,
  lifecycle updates, and final sign-off in the main session.
- The fallback execution agent must keep the same scope boundaries that would
  have applied to the production-code and review subagents.
- The review gate is satisfied by a fresh review pass in the main session,
  including plan alignment, changed-file diff inspection, style checks,
  prompt/payload leakage checks, and verification evidence review.
- Do not continue into deferred integration work without a new or updated
  approved plan.

## Progress Checklist

- [x] Stage 0 - plan approval boundary confirmed.
  - Covers: status/user authorization before implementation. Verify: plan
    `Status` is `approved` or `in_progress` and user directly instructed
    execution. Evidence/sign-off: recorded in `Execution Evidence`;
    sign-off: Codex / 2026-07-02; next Stage 1.
- [x] Stage 1 - focused test contract established.
  - Covers: steps 1-4. Verify: run config/runtime-context/prompt/payload tests
    named in `Verification` and record current baseline or expected failure.
    Evidence/sign-off: command output recorded; next Stage 2.
- [x] Stage 2 - L2 prompt and payload implementation complete.
  - Covers: steps 5-10. Verify: focused config, runtime-context, prompt, and
    payload tests pass. Evidence/sign-off: changed files and test output
    recorded; next Stage 3.
- [x] Stage 3 - action-selection behavior verified.
  - Covers: steps 10-12. Verify: action-selection prompt, payload, and
    fake-LLM/parser tests pass. Evidence/sign-off: command output recorded;
    next Stage 4.
- [x] Stage 4 - live LLM evidence inspected.
  - Covers: step 13. Verify: live LLM cases run one at a time or
    blockers/skips are recorded. Evidence/sign-off: trace ids, visible results,
    and judgment recorded; next Stage 5.
- [x] Stage 5 - regression and static verification complete.
  - Covers: steps 14-15. Verify: broader regression tests, static greps, and
    prompt-render checks pass or have documented allowed exceptions.
    Evidence/sign-off: command output recorded; next Stage 6.
- [x] Stage 6 - independent code review complete.
  - Covers: steps 16-17. Verify: independent reviewer reports no unresolved
    blockers and remediation reruns are recorded. Evidence/sign-off: findings,
    fixes, reruns, and risks recorded; next Stage 7.
- [x] Stage 7 - lifecycle sign-off complete.
  - Covers: step 18. Verify: all prior stages are checked with evidence and
    status changes only after completion criteria are met. Evidence/sign-off:
    final status update recorded; sign-off: Codex / 2026-07-02; no handoff.

## Verification

Focused deterministic commands:

```powershell
venv\Scripts\python -m pytest tests\test_config.py
venv\Scripts\python -m pytest tests\test_cognition_chain_core_contracts.py
venv\Scripts\python -m pytest tests\test_cognition_chain_connector_mapping.py
venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py
venv\Scripts\python -m pytest tests\test_action_selection_prompt_contract.py
venv\Scripts\python -m pytest tests\test_action_selection_payload.py
venv\Scripts\python -m pytest tests\test_cognition_chain_core_action_selection.py
venv\Scripts\python -m pytest tests\test_l2d_action_selection_cases.py
```

Focused live LLM commands must run one case at a time, with output inspected:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_live_llm_boundary_affinity.py -m live_llm
venv\Scripts\python -m pytest tests\test_l2d_action_selection_live_llm.py -m live_llm
```

If a new live test file is added:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_live_llm_affinity_willingness.py -m live_llm
```

Static source checks:

```powershell
rg "affinity_task_gate|effort_score|complexity_score|tool cost|affinity threshold" src\kazusa_ai_chatbot\cognition_chain_core
rg "mood_gate|vibe_gate|patience_score|willingness_score|mood threshold|vibe threshold" src\kazusa_ai_chatbot\cognition_chain_core
rg "resource heavy|complex_task_resolution" src\kazusa_ai_chatbot\cognition_chain_core\stages\l2.py
rg "task_willingness_boundary_enabled|COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED" src\kazusa_ai_chatbot\cognition_chain_core\action_selection_prompt.py
rg "\"task_willingness_boundary_enabled\"|COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED|affinity_task_gate|mood_gate|vibe_gate|feature_enabled|effort_score|complexity_score|patience_score|willingness_score" src\kazusa_ai_chatbot\cognition_chain_core\action_selection.py
```

Expected result: zero forbidden source matches. `rg` exit code 1 is acceptable
only when it means zero matches. If a pre-existing unrelated match appears,
record the file and exact reason it is unrelated before sign-off.

The prompt-module static check must return zero matches so model-facing prompt
text does not expose the flag. The action-selection static check must return
zero matches for quoted payload keys, env names, and forbidden gate metadata;
code-side branching on `state["task_willingness_boundary_enabled"]` is allowed
only to select the legacy vs willingness-aware prompt before model invocation.

Prompt-render checks:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py tests\test_action_selection_prompt_contract.py
```

Broader regression candidates:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_l2d_contract.py tests\test_persona_supervisor2_action_selection.py tests\test_l2d_l3_surface_handoff.py
```

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not draft the plan. If no separate reviewer is available, the drafting
agent must reread the plan, registry, relevant source/test context, and plan
contract from a fresh-review posture.

Review scope:

- The scope aligns with cognition-chain/action-spec ownership and gives
  concrete contracts, payloads, change surface, file paths, verification,
  checklist, and evidence instructions.
- Agent creativity is tightly bounded: no unresolved choices, broad helper
  freedom, optional fallbacks beyond the explicit config switch, deterministic
  semantic gates, or hidden tool filtering remain.
- Boundaries between task-taking refusal, ordinary speech, L2d routing,
  background-work execution, public runtime context, internal stage payloads,
  and forbidden L2d metadata are explicit.

Record blockers, non-blocking findings, required edits, open questions, and
approval status in `Execution Evidence`. Approve only when blockers are
resolved. If blockers remain, update the plan or ask the user before execution.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
For this execution run, the user explicitly requested no subagents, so the
fallback execution agent must perform an independent-review-style fresh pass in
the main session instead of creating a review subagent.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG/context leaks, persistence
  risk, brittle fixtures, ordinary-speech suppression, and avoidable blast
  radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`,
  `Change Surface`, exact contracts, implementation order, verification gates,
  and acceptance criteria.
- Regression and handoff quality, including focused and live LLM evidence,
  static-grep expectations, and next-stage handoff notes.

The fallback execution agent fixes concrete findings directly only when the fix
is inside the approved change surface or this review gate explicitly allows
review-only fixture/documentation corrections. If a fix would cross the
approved boundary or alter the contract, stop and update the plan or request
approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

- `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED` exists, defaults to `true`,
  parses true/false through fail-fast boolean config, and is documented in
  `docs/HOWTO.md`.
- When the flag is `false`, legacy L2b/L2c/L2d prompt behavior is preserved
  and the new mood/vibe willingness descriptors are not added to L2b/L2c
  model-facing payloads.
- When the flag is `true`, L2b/L2c use the willingness-aware prompt contract
  and allowed existing semantic descriptors.
- L2 can refuse, deflect, tease away, or offer a smaller scope for demanding
  task-like requests when relationship context, current mood, or scene vibe
  makes taking the request on feel too much.
- L2 can still accept, answer, or speak normally for simple low-pressure
  requests at low affinity or in a bad/tense mood when existing silence logic
  does not apply.
- The final refusal/deflection decision is represented through existing
  `logical_stance`, `character_intent`, and `judgment_note` behavior.
- L2d receives no new affinity gate, mood gate, vibe gate, effort, cost,
  threshold, feature-enabled, or complexity metadata.
- L2d does not schedule private, future, or background work when settled L2
  cognition has not accepted taking on the task-like request.
- This feature does not introduce a new reason to produce no visible response;
  task refusal normally remains visible speech.
- Action affordances remain visible; the tool is not hidden from the character.
- No new agent, schema migration, DB field, action kind, or background worker
  behavior is added.
- Focused deterministic tests pass.
- One-at-a-time live LLM inspection shows acceptable behavior for the planned
  low/high affinity, simple/demanding, and bad-mood/tense-scene cases.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Feature needs emergency rollback. | Keep `false` as a documented service-restart rollback path. | Config tests and disabled-mode prompt/payload tests. |
| Prompt drift makes the character refuse ordinary requests. | Explicit normal-speech rules and simple-request tests. | Deterministic simple-request tests and live simple-speaking case. |
| Mood/vibe becomes a general no-response rule. | `no-response` remains owned by existing silence logic. | Action-selection tests require visible `speak` for task refusal. |
| Local LLM behavior is inconsistent without thresholds. | Keep semantic question small and inspect live cases one at a time. | Live LLM cases and trace review. |
| L2d selects private action after non-accepted L2 outcome. | Strengthen L2d prompt and route-following tests. | L2d fake-LLM/parser tests and live action-selection tests. |
| Config flag leaks into semantic LLM payloads. | Use it only for prompt selection and internal state. | Payload-shape tests and static greps. |
| Upstream RAG remains expensive. | Declare pre-L2d cost out of scope. | Review verifies no RAG/relevance/decontext changes. |

## Execution Evidence

- 2026-06-30: Draft created from user-approved architectural conclusion:
  keep tools visible, keep decision-making in L2, avoid exposing effort/cost or
  thresholds, avoid a new agent, and make L2d follow settled L2 cognition.
- 2026-07-02: Plan-review pass resolved pre-implementation blockers: L2b/L2c
  may project existing prompt-safe mood/vibe descriptors while L2d payload
  remains unchanged; task refusal normally routes visible `speak`; `no-response`
  remains owned by existing silence logic; plan class is now `large`; execution
  gates, cutover policy, static checks, and review gates match the plan
  contract.
- 2026-07-02: Disablement design added after user clarified the feature may not
  be wanted in all deployments. Source mapping found `config.py` fail-fast
  boolean parsing, HOWTO config documentation, `RuntimeContextV1`, connector
  runtime-context projection, and chain internal-state mapping as the narrow
  path. The plan uses one enabled-by-default flag,
  `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED`, to select legacy vs
  willingness-aware prompt contracts before model invocation. Set the flag to
  `false` to disable the feature. The flag must not appear in L2d payloads or
  act as a post-LLM semantic override.
- 2026-07-02: User explicitly instructed: "Execute the plan without subagent."
  Lifecycle updated to `in_progress`; registry row updated; Stage 0 signed off.
  Execution will use single-agent fallback in the main session, with code-side
  prompt selection allowed before model invocation and L2d model-facing payload
  leakage still forbidden.
- 2026-07-02: Stage 1 baseline recorded. `git status --short` showed only
  plan/test edits. Focused failing tests were established before production
  edits:
  `venv\Scripts\python -m pytest tests\test_config.py::TestCognitionTaskWillingnessBoundaryConfig -q`
  failed because `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED` is not present
  and invalid values are not rejected; `venv\Scripts\python -m pytest
  tests\test_action_selection_prompt_contract.py::test_action_selection_enabled_prompt_follows_task_refusal_outcome -q`
  failed because `ACTION_ROUTER_TASK_WILLINGNESS_PROMPT` is not present.
  Sign-off: Codex / 2026-07-02; next Stage 2.
- 2026-07-02: Stage 2 implementation completed in single-agent fallback.
  Changed files: `src/kazusa_ai_chatbot/config.py`,
  `src/kazusa_ai_chatbot/cognition_chain_core/contracts.py`,
  `src/kazusa_ai_chatbot/cognition_chain_core/graph_state.py`,
  `src/kazusa_ai_chatbot/cognition_chain_core/chain.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`,
  `src/kazusa_ai_chatbot/cognition_chain_core/stages/l2.py`,
  `src/kazusa_ai_chatbot/cognition_chain_core/action_selection_prompt.py`,
  `src/kazusa_ai_chatbot/cognition_chain_core/action_selection.py`, and
  `docs/HOWTO.md`. Focused deterministic tests passed:
  `venv\Scripts\python -m pytest tests\test_config.py
  tests\test_cognition_chain_core_contracts.py
  tests\test_cognition_chain_connector_mapping.py -q` passed 81 tests;
  `venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py
  tests\test_action_selection_prompt_contract.py
  tests\test_action_selection_payload.py -q` passed 30 tests.
  Sign-off: Codex / 2026-07-02; next Stage 3.
- 2026-07-02: Stage 3 route-following coverage added and verified.
  `tests/test_cognition_chain_core_action_selection.py` now includes a
  parser-level upstream-refusal case proving the enabled prompt is selected,
  L2d human payload does not expose the feature flag, and the normalized action
  is visible `speak` without `accepted_task_request`. Focused commands passed:
  `venv\Scripts\python -m pytest tests\test_cognition_chain_core_action_selection.py
  tests\test_l2d_action_selection_cases.py -q` passed 12 tests;
  `venv\Scripts\python -m pytest tests\test_action_selection_prompt_contract.py
  tests\test_action_selection_payload.py -q` passed 7 tests.
  Sign-off: Codex / 2026-07-02; next Stage 4.
- 2026-07-02: Stage 4 live LLM evidence inspected. Added
  `tests/test_cognition_live_llm_affinity_willingness.py` and ran cases one at
  a time:
  `low-affinity-simple-help`, `low-affinity-demanding-sustained-help`,
  `bad-mood-simple-speaking`, `high-affinity-demanding-help`, and
  `test_live_l2d_task_willingness_refusal_routes_speak`; all passed schema or
  route assertions. Raw traces were written under ignored
  `test_artifacts/llm_traces/`, and the agent-authored review artifact
  `test_artifacts/llm_reviews/cognition_task_willingness_live_review_2026-07-02.md`
  summarizes the real outputs. Observed behavior: simple low-affinity help
  remained `CONFIRM` / `PROVIDE`; low-affinity sustained project tracking was
  `REFUSE` / `REJECT`; bad mood still produced ordinary `BANTAR`; high
  affinity demanding help was not auto-refused; L2d selected only `speak` for
  the settled refusal. Sign-off: Codex / 2026-07-02; next Stage 5.
- 2026-07-02: Stage 5 regression and static verification completed. Focused
  deterministic suite
  `venv\Scripts\python -m pytest tests\test_config.py
  tests\test_cognition_chain_core_contracts.py
  tests\test_cognition_chain_connector_mapping.py
  tests\test_cognition_prompt_contract_text.py
  tests\test_action_selection_prompt_contract.py
  tests\test_action_selection_payload.py
  tests\test_cognition_chain_core_action_selection.py
  tests\test_l2d_action_selection_cases.py -q` passed 123 tests. Broader
  regression command
  `venv\Scripts\python -m pytest tests\test_cognition_resolver_l2d_contract.py
  tests\test_persona_supervisor2_action_selection.py
  tests\test_l2d_l3_surface_handoff.py -q` passed 44 tests. Prompt-render
  command `venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py
  tests\test_action_selection_prompt_contract.py -q` passed 28 tests. Static
  greps for forbidden gate, threshold, effort, resource, feature-flag, and
  L2d-payload metadata terms all returned zero matches; the final
  action-selection grep was rerun after correcting PowerShell quoting.
  `git diff --check` passed with line-ending normalization warnings only.
  Sign-off: Codex / 2026-07-02; next Stage 6.
- 2026-07-02: Stage 6 independent-review-style pass completed in the main
  session per no-subagent execution. Review checked changed-file inventory,
  runtime-context call sites, prompt/payload leakage, deterministic and live
  test diffs, and plan alignment. Findings fixed: enabled prompt extensions
  now explicitly restate JSON-only output after their appended guidance, and
  the connector mapping test line break was cleaned up. Affected verification
  rerun `venv\Scripts\python -m pytest tests\test_cognition_chain_connector_mapping.py
  tests\test_cognition_prompt_contract_text.py
  tests\test_action_selection_prompt_contract.py -q` passed 34 tests; static
  greps rerun for forbidden gate/threshold/resource/flag terms returned zero
  matches. No unresolved blockers. Residual risk: live evidence is synthetic
  and the feature remains enabled-by-default with documented `false` rollback.
  Sign-off: Codex / 2026-07-02;
  next Stage 7.
- 2026-07-02: Stage 7 lifecycle sign-off completed. Final post-review live
  LLM rerun passed all one-at-a-time cases:
  `low-affinity-simple-help`, `low-affinity-demanding-sustained-help`,
  `bad-mood-simple-speaking`, `high-affinity-demanding-help`, and
  `test_live_l2d_task_willingness_refusal_routes_speak`. Final deterministic
  verification passed: focused suite passed 123 tests, broader L2d/surface
  regression passed 44 tests, `git diff --check` passed with line-ending
  warnings only, and all forbidden prompt/payload static greps returned zero
  matches. Plan status updated to `completed`; registry row updated to
  `completed`. Sign-off: Codex / 2026-07-02; no handoff.
- 2026-07-02: User clarified the feature should be enabled by default.
  Runtime config, HOWTO, config tests, and plan text were corrected so
  `COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED` defaults to `true` while
  `false` remains the documented rollback value. Live LLM coverage was expanded
  to ten one-at-a-time cases covering ordinary speech, light help, sustained
  task-taking, coercive requests, bad mood, high affinity, controlling bad
  vibe, group pressure, L2d refusal routing, and L2d accepted-work routing.
  Sign-off: Codex / 2026-07-02.
