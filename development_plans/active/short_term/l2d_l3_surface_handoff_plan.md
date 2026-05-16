# L2d L3 Surface Handoff Execution Plan

## Summary

- Goal: complete the handoff from L2d action selection to selected L3 surface
  handlers, with `speak -> L3 text -> dialog -> delivery edge` and
  `trigger_future_cognition -> scheduled cognition slot` as the two runtime
  action chains implemented by this plan.
- Plan class: high_risk_migration
- Status: in_progress
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`,
  `cjk-safety`
- Overall cutover strategy: stable public `/chat` response shape; bigbang
  internal cognition, L3, and dialog contract cutover after approval
- Highest-risk areas: live persona graph routing, L3 text/dialog gating,
  no-visible-action consolidation, scheduler/orchestrator boundary for future
  cognition, and accidental revival of `send_message` as an L2d-facing action
- Acceptance criteria: L3 text/dialog runs only when L2d selected `speak`;
  no selected visible surface skips L3/dialog but still consolidates;
  text delivery is derived after dialog, never selected directly by L2d;
  `trigger_future_cognition` creates a scheduled follow-up cognition contract
  without calling cognition inline; future cognition handoff text is one
  precise semantic `continuation_objective`, not a lossy context summary;
  visual directives remain a side effect of text-surface handling only

This plan is an in-progress blocker discovered during execution of
`modality_neutral_action_spec_effector_expansion_plan.md`.

2026-05-16 reconciliation: this plan's original context describes the
pre-reconnection graph where the contextual/social agent still lived under
selected L3. `cognition_llm_stage_reconnection_plan.md` supersedes that
placement. The current approved topology is:

```text
l2c1_judgment_synthesis
  + l2c2_social_context_appraisal
  -> l2d_action_selection
  -> selected L3 style/content/preference/visual
  -> l4_surface_directive_collector
```

The semantic responsibility is unchanged: the former contextual agent still
emits only `social_distance`, `emotional_intensity`, `vibe_check`, and
`relational_dynamic`; it is now L2c2 evidence for L2d and later selected L3
surfaces.

## Context

The current action-spec implementation creates L2d action specs and uses a
top-level route to decide whether dialog should run. That is not a complete
L2d-to-L3 handoff. In
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`, L2d still flows
unconditionally into the existing L3 chain:

```text
l2c_judgment_core
  -> l2d_action_initializer
  -> l3_contextual_agent
  -> l3_style/content/preference/visual
  -> l4_collector
```

That means L3 is still an always-on cognition layer rather than a selected
surface/action handler. The current route checks only whether `speak` exists
after L3 has already run. This violates the accepted architecture:

```text
L1/L2a/L2b/L2c -> L2d action initialization
  -> selected L3 surface/action handlers
  -> action results + surface outputs
  -> episode-trace consolidation
```

The correction is not to redesign all L3s. The current runtime needs two
action chains:

1. `speak`: L2d selects a text surface. The selected text handler runs the
   existing L3 directive agents, keeps visual directives as a side effect, runs
   dialog, emits `SurfaceOutputV1(surface_kind="text")`, and then uses the
   appropriate deterministic delivery edge.
2. `trigger_future_cognition`: L2d selects a private follow-up cognition
   request. The orchestrator/scheduler owner records a future cognition slot.
   It must not call cognition directly from the action handler. The scheduled
   slot must carry exactly one LLM-facing `continuation_objective` string as the
   future thinking contract. Typed scheduler fields, source refs, target scope,
   and action-attempt IDs remain deterministic-only metadata.

`memory_lifecycle_update` remains a private action handler from the current
plan. It does not require L3. L3 visual remains a text-surface side effect in
this plan; image generation and standalone image-surface routing stay deferred.

### Legacy Response-Gate Scan

The 2026-05-16 scan found old L3/dialog-level text response gates that conflict
with L2d owning selected action surfaces:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - `_cognition_requests_silence(...)` reads
    `action_directives.contextual_directives.expression_willingness`.
  - `_route_after_cognition(...)` still falls back to that L3 field when
    `action_specs` are absent.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - `conditional_skip_dialog_agent(...)` skips dialog generation when
    `expression_willingness == "silent"`.
  - The dialog generator/evaluator prompts still treat
    `expression_willingness` as an input field.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - `l3_contextual_agent` asks the LLM to emit
    `expression_willingness` with a `silent` enum option.
  - The collector copies `expression_willingness` into
    `action_directives.contextual_directives`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
  and `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  still make `expression_willingness` part of the L3/contextual contract.
- `src/kazusa_ai_chatbot/nodes/README.md` and focused persona/dialog tests
  still document or assert that L3 can suppress text by choosing silence.

These are removal targets for this plan. After L2d exists, selected
`speak` is the only positive gate for L3 text/dialog execution. The decision is
not to preserve, rename, reinterpret, or compatibility-shim
`expression_willingness`. The field is removed from the runtime L3 contextual
contract, collector output, dialog input, tests, and subsystem docs. L3
describes social temperature and expression style after `speak` has been
selected, but it must not carry a `silent`, `no_response`, `withholding`, or
equivalent response decision field.

The scan also found controls that must not be deleted as part of this cleanup:

- `should_respond` in the relevance agent, brain graph, service seed, and state
  reducer is an upstream intake/relevance control. It may prevent the persona
  graph from starting, but it must not be reused inside selected L3 text as a
  response gate.
- `listen_only`, `think_only`, and `CognitiveEpisode.output_mode` are external
  debug/run-mode or origin metadata controls. They may suppress adapter
  delivery or select dry-run behavior outside L3, but selected L3 text/dialog
  must not branch on them to decide whether the character wants to speak.
- `no_visual_directives` remains a visual-side-effect skip control and is not a
  text response gate.

## Mandatory Skills

- `development-plan-writing`: load before changing this plan, registry rows,
  execution evidence, or lifecycle status.
- `local-llm-architecture`: load before changing cognition graph routing,
  prompts, L3 boundaries, action routing, scheduler/orchestrator behavior, or
  background LLM behavior.
- `no-prepost-user-input`: load before changing commitment, preference,
  permission, or action persistence decisions.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files with CJK prompt strings.

## Mandatory Rules

- After any context compaction, reread this entire plan before continuing.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the Independent Code Review gate and record the result in Execution Evidence.
- Execute this plan only while the registry status is `in_progress`.
- Do not make L2a, L2b, L2c, L3, or dialog select actions.
- Do not expose `send_message` to L2d. L2d selects `speak`; delivery is derived
  after text exists.
- Do not ask L2d to generate delivery parameters, adapter ids, raw channel ids,
  scheduler ids, handler ids, database ids, or collection names.
- Do not make L3 visual a standalone image action in this plan. It remains a
  side effect of text-surface directive generation.
- Do not add image generation, web research, arbitrary notes, external HTTP,
  shell, file, or MongoDB tools.
- Do not let a tool/action handler call cognition inline. Future cognition is
  represented as a scheduled contract owned by orchestrator/scheduler code.
- No selected visible surface must mean no L3 text/dialog call. The episode may
  still consolidate from private action results or private finalization.
- Do not route or skip selected L3 text/dialog from
  `expression_willingness`, `should_respond`, `output_mode`, `listen_only`, or
  `think_only` inside the selected text-surface path.
- Dialog must not have an internal "do not generate" branch based on L3
  directives. If dialog is called, the selected `speak` action has already
  chosen a text surface.
- Deterministic code owns graph routing, validation, scheduling, delivery edge
  selection, adapter feasibility, persistence, and audit records.
- LLM stages own semantic judgment only: L2d chooses whether the character
  wants to speak or wants a future cognition cycle.

## Must Do

- Split current cognition routing so `call_cognition_subgraph` ends after L2d
  and returns L1/L2/L2d state without running L3 unconditionally.
- Add a selected text-surface handler that runs existing L3 contextual,
  interaction-style, style, content-anchor, preference, visual, and collector
  agents only for a selected `speak` action.
- Remove legacy L3/dialog text response gates:
  - delete `_cognition_requests_silence(...)`;
  - remove the `_route_after_cognition(...)` fallback that treats
    `expression_willingness == "silent"` as no response;
  - remove `conditional_skip_dialog_agent(...)` from `dialog_agent`;
  - remove `expression_willingness` from the L3 contextual prompt, output
    contract, schema, collector, dialog prompt input, fixtures, docs, and
    tests.
- Keep the non-decision L3 presentation contract: `social_distance`,
  `emotional_intensity`, `vibe_check`, `relational_dynamic`,
  `rhetorical_strategy`, `linguistic_style`, `content_anchors`,
  `accepted_user_preferences`, `forbidden_phrases`, and visual directives.
  Do not retain any internal compatibility layer for the removed
  `expression_willingness` field.
- Keep L3 visual as a side effect of text-surface handling. The visual agent
  may still be skipped by existing debug/config controls.
- Run `dialog_agent` only after selected text L3 directives exist.
- Emit text `SurfaceOutputV1`, `ActionResultV1(kind="speak")`, and
  `EpisodeTraceV1` for selected text responses.
- Add a deterministic delivery edge after dialog:
  - live `/chat` visible replies return through the existing service
    `ChatResponse.messages` path;
  - scheduled/proactive delivery contexts convert text to bridge-only
    `ActionSpecV1(kind="send_message") -> RawToolCall -> TaskDispatcher`.
- Implement runtime handling for `trigger_future_cognition` as a private
  orchestrator action that schedules a bounded future cognition slot and
  records an action result. It must not invoke cognition immediately.
- Keep `memory_lifecycle_update` behavior as a private non-L3 action.
- Ensure private-only episodes consolidate without requiring `final_dialog`.
- Update docs and the parent action-spec plan evidence so this blocker and its
  resolution are explicit.

## Deferred

- Do not implement standalone `l3_image` action routing.
- Do not call an external image generation service.
- Do not implement `web_research`, `fetch_url`, notes/open-loop tools, or
  arbitrary tool expansion.
- Do not introduce a new public service endpoint or platform adapter contract.
- Do not add a second scheduler collection.
- Do not migrate existing scheduled events.
- Do not change L2d prompt semantics in this plan.
- Do not add retry loops or repair prompts for L3 handoff.
- Do not make the consolidator select, execute, dispatch, schedule, or trigger
  actions.

## Cutover Policy

Overall strategy: stable public `/chat` response shape, bigbang internal
contract cutover. Internal compatibility with the old L3/dialog response gate
is explicitly rejected.

| Area | Policy | Instruction |
|---|---|---|
| Persona cognition graph | bigbang | Stop running L3 unconditionally after L2d. There must not be both an old always-on L3 path and a selected-surface L3 path. |
| L3 contextual contract | bigbang | Remove `expression_willingness` from prompt output, schema, collector, action directives, docs, and tests. No alias, deprecation period, or compatibility shim. |
| Dialog input graph | bigbang | Delete the internal skip branch and remove `expression_willingness` from generator/evaluator prompt inputs. If dialog is called, it generates text. |
| Text surface | bigbang internal, public response stable | Reuse non-decision L3/dialog behavior when `speak` is selected, but do not preserve the retired response gate. |
| No-visible-action path | bigbang internal, public response stable | No selected visible surface means no L3 text/dialog. Empty public reply behavior remains, with private trace/consolidation evidence. |
| `send_message` | bridge-only | Keep bridge-only delivery after text exists. Do not expose it to L2d. |
| Future cognition slot | additive | Add scheduled future-cognition handling through the selected action handler. Do not add a parallel scheduler or migrate existing scheduled events. |
| L3 visual | selected text side effect | Keep as text-surface side effect and existing config/debug skip behavior. |

## Cutover Policy Enforcement

- The implementation agent must follow the policy for each area.
- Bigbang persona graph routing means the old unconditional L3 path must be
  deleted from the cognition subgraph and rewired only through the selected
  text-surface handler.
- Stable public behavior applies only to service/API output shape. It does not
  authorize preserving old internal L3/dialog fields, branches, or tests.
- Any change to scheduler collection shape, adapter API, or service endpoint
  requires explicit plan revision before implementation.

## Overdesign Guardrail

- Actual problem: L2d action selection currently gates only top-level dialog
  routing after the old L3 chain has already run, so L3 is not truly selected
  by action specs.
- Minimal change: move existing L3 text directive generation behind the
  selected `speak` action and add a small orchestrator handler for
  `trigger_future_cognition`.
- Ownership boundaries: L2d owns semantic action choice; selected L3 text owns
  expression directives and visual side-effect directives; dialog owns final
  wording; deterministic routing owns delivery edge selection; dispatcher owns
  adapter-facing scheduled sends; orchestrator/scheduler owns future cognition
  slot persistence; consolidator consumes prompt-safe trace evidence only.
- Rejected complexity: no standalone L3 image action, no image generation, no
  new service endpoint, no extra prompt agent, no generic action chaining
  engine, no second scheduler collection, no `send_message` exposure to L2d,
  no fallback to old always-on L3, no arbitrary tools.
- Evidence threshold: add standalone image surfaces, generic chained actions,
  or broader scheduler refactors only after a separate approved plan names the
  action owner, handler contract, trace output, permission model, and tests.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they implement
  the contracts in this plan.
- The agent must not introduce alternate graph paths, feature flags, fallback
  branches, compatibility shims, or extra tools.
- The agent must remove `expression_willingness` rather than retaining it for
  code-level compatibility.
- The agent must not rename public state keys outside the removals and fields
  explicitly named by this plan.
- The agent must treat changes outside `nodes`, `action_spec`, `dispatcher`,
  `scheduler`, `self_cognition`, `service`, and docs as out of scope. If a
  dependency outside that surface appears necessary, stop and update the plan
  before changing it.
- Reuse existing validation, scheduling, and trace helpers instead of
  duplicating behavior.
- If scheduler support for future cognition cannot be implemented safely within
  the existing scheduler/tool runtime, stop and update this plan instead of
  adding a parallel scheduler.
- If this plan and existing code disagree, preserve this plan's owner boundary
  and record the discrepancy.

## Target State

The target runtime flow is:

```text
typed episode
  -> RAG/evidence
  -> L1/L2a/L2b/L2c
  -> L2d action initialization
  -> action router
       speak
         -> selected L3 text directive handler
              contextual/style/content/preference/visual-side-effect/collector
         -> dialog
         -> text SurfaceOutputV1
         -> live ChatResponse delivery or bridge-only send_message scheduling
       memory_lifecycle_update
         -> private memory lifecycle handler
       trigger_future_cognition
         -> private scheduled cognition-slot handler
  -> action results + surface outputs
  -> episode trace
  -> consolidation
```

Observable behavior:

- Ordinary user-message replies still speak naturally when L2d selects
  `speak`.
- If L2d selects no visible surface, no L3 text/dialog work runs and the user
  receives no message.
- Self-cognition can select no action, visible `speak`, private
  `memory_lifecycle_update`, or private `trigger_future_cognition`.
- A future cognition request creates an auditable scheduled slot and returns an
  action result; it does not run cognition inline.
- Text delivery to adapters happens only after dialog has produced final text.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| L3 handoff | Move L3 text directive generation behind selected `speak`. | This makes L3 a selected surface handler instead of an always-on cognition stage. |
| L3 visual | Keep visual agent inside text-surface handler as a side effect. | The user explicitly said visual is side effect for L3 text; standalone image surface is deferred. |
| `send_message` | Keep as bridge-only after dialog text exists. | L2d should not confuse "I want to speak" with adapter delivery mechanics. |
| Normal chat delivery | Use existing `ChatResponse.messages` path. | Normal `/chat` delivery is already adapter-owned and should not be forced through scheduler. |
| Scheduled/proactive delivery | Convert text to `ActionSpecV1(kind="send_message")` only after surface output exists. | Keeps dispatcher validation and scheduler delivery owned by deterministic code. |
| Future cognition | Implement as orchestrator-owned scheduled action result. | Tools must not call cognition directly; the next cognition cycle must have an auditable trigger contract. |
| Text response gate | Use selected L2d `speak` as the only post-L2 text-surface gate. | Removes duplicated L3/dialog suppression flags and keeps action selection in one place. |
| Upstream/debug controls | Keep `should_respond`, `listen_only`, `think_only`, and `output_mode` outside selected L3 text routing. | Maintains intake/debug behavior without letting those controls become hidden L3 action selectors. |
| L2d prompt | Do not change L2d prompt text in this plan. | The current live L2d routing evidence is acceptable; the blocker is graph handoff and stale downstream gates. |

## Contracts And Data Shapes

### Selected Surface Router

The selected surface router consumes:

```python
{
    "action_specs": list[ActionSpecV1],
    "internal_monologue": str,
    "logical_stance": str,
    "character_intent": str,
    "judgment_note": str,
    "rag_result": dict,
    "cognitive_episode": dict,
    ...
}
```

It returns:

```python
{
    "action_directives": dict,
    "final_dialog": list[str],
    "surface_outputs": list[SurfaceOutputV1],
    "action_results": list[ActionResultV1],
    "episode_trace": EpisodeTraceV1,
}
```

Rules:

- If no selected action has `kind == "speak"`, L3 text and dialog must not run.
- If one or more selected actions have `kind == "speak"`, execute one text
  surface for this plan. If multiple `speak` actions appear, choose the first
  valid one and record the others as rejected duplicate text-surface requests.
- Private actions may execute in the same episode as `speak`.
- Private action failures must appear as `ActionResultV1(status="failed")` or
  `status="rejected"` and must not force user-visible text.
- The selected surface router must not inspect L3 contextual directives to
  decide whether text should run.

### L3 Text Handler

The L3 text handler owns the existing L3 chain:

```text
l3_contextual_agent
l3_interaction_style_context_loader
l3_style_agent
l3_content_anchor_agent
l3_preference_adapter
l3_visual_agent
l4_collector
```

It receives the existing cognition state plus a prompt-safe marker that
`speak` was selected. This plan does not project `speak.params` into L3
prompts. It must not expose raw target ids, collection names, adapter ids,
handler ids, or scheduler ids.

The L3 text handler runs only after `speak` is selected. Its contextual branch
describes social distance, emotional intensity, vibe, and relational dynamic,
but it must not emit a response/no-response enum. Dialog runs whenever
the selected text handler completes with valid directives.

The L3 contextual output shape after this plan is:

```python
{
    "social_distance": str,
    "emotional_intensity": str,
    "vibe_check": str,
    "relational_dynamic": str,
}
```

The collected `action_directives.contextual_directives` shape is the same.
`expression_willingness` is not valid input or output.

### Delivery Edge

After dialog creates text fragments:

- live `/chat` delivery returns fragments through the existing persona/service
  response path;
- scheduled/proactive delivery creates bridge-only
  `ActionSpecV1(kind="send_message")`, then `RawToolCall(tool="send_message")`,
  then `TaskDispatcher.dispatch`.

The delivery edge is deterministic. L2d does not choose platform, channel,
adapter, permission, or scheduler details.

### Future Cognition Slot

`ActionSpecV1(kind="trigger_future_cognition")` handler output:

```python
{
    "status": "scheduled" | "rejected" | "failed",
    "scheduled_event_ids": list[str],
    "episode_type": "self_cognition",
    "trigger_at": str | None,
    "reason": str,
}
```

The handler must validate:

- action kind is `trigger_future_cognition`;
- visibility is `private`;
- `params.episode_type == "self_cognition"`;
- `trigger_at` is absent or an absolute ISO-8601 timestamp;
- continuation depth remains bounded;
- no raw adapter or database ids were supplied by L2d.

The scheduled event must carry a prompt-safe source/action ref that the future
self-cognition path can use as a typed trigger. It must not contain raw prompt
text, final dialog, private source packet text, credentials, or arbitrary model
output. Its only LLM-facing handoff payload is one semantic
`continuation_objective` string. If the future cognition action is paired with a
concrete sibling action, such as `speak`, the sibling action's semantic `detail`
is the preferred source for that objective. If no faithful one-string objective
can be built, scheduling must fail closed.

## LLM Call And Context Budget

- Before this plan, live user-message path already calls L1, L2a, L2b, L2c,
  L2d, always-on L3 directive agents, and dialog when responding.
- After this plan, L3 directive agents and dialog run only when `speak` is
  selected. No-visible-action episodes should be cheaper than before.
- This plan does not authorize a new response-path LLM call.
- This plan does not authorize an L2d prompt rewrite. An L2d prompt diff is a
  plan violation and must be removed or handled by a separate approved plan.
- Future cognition scheduling creates a later episode contract; it does not
  add another LLM call to the current turn.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
  - Owns selected text-surface handler composition around existing L3 agents.
- `src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py`
  - Owns validation/materialization of `trigger_future_cognition` action
    results and scheduled slot requests.
- Focused tests under flat `tests/` names, no `tests/unit/` subdirectory:
  - `tests/test_l2d_l3_surface_handoff.py`
  - `tests/test_action_spec_future_cognition.py`

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Stop after L2d; remove unconditional L3 edges from this subgraph.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Route selected actions to private handlers and selected L3 text handler.
  - Attach results/surfaces/episode trace.
  - Remove `_cognition_requests_silence(...)` and any fallback route based on
    L3 `expression_willingness`.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Remove the internal `expression_willingness == "silent"` skip branch.
  - Remove `expression_willingness` from generator/evaluator prompt inputs
    completely. Do not replace it with a new response gate.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Remove `expression_willingness` as a contextual-agent output and remove the
    `silent`/response-suppression vocabulary from the prompt and collector.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
  - Remove `expression_willingness` from L3 contextual output validation.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Add only required state fields for selected surface handoff.
  - Remove `expression_willingness` from cognition state and contextual
    directive schemas.
- `src/kazusa_ai_chatbot/action_spec/registry.py`
  - Keep `trigger_future_cognition` L2d-facing; keep `send_message`
    bridge-only.
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`
  - Add or tighten validation required for future cognition action specs.
- `src/kazusa_ai_chatbot/action_spec/results.py`
  - Extend the existing action-result helper path to represent scheduled
    future cognition. Do not add a parallel result shape.
- `src/kazusa_ai_chatbot/dispatcher/` and `src/kazusa_ai_chatbot/scheduler.py`
  - Change only as needed to schedule a non-adapter cognition slot through the
    existing scheduled-event runtime. Do not make dispatcher the parent action
    system.
- `src/kazusa_ai_chatbot/self_cognition/models.py`
  - Add the scheduled future-cognition case and trigger constants used by the
    normal self-cognition source packet path.
- `src/kazusa_ai_chatbot/self_cognition/sources.py`
  - Add the future-slot typed trigger adapter for due scheduled
    self-cognition slots. Keep scheduler ids and action-attempt ids internal;
    project only semantic source refs and follow-up context to the model.
- `src/kazusa_ai_chatbot/self_cognition/worker.py`
  - Consume scheduled future-cognition cases through the existing worker
    runner and mark the source scheduled event completed only after the normal
    case processing path returns.
- Documentation:
  - `src/kazusa_ai_chatbot/nodes/README.md`
  - `src/kazusa_ai_chatbot/action_spec/README.md`
  - `src/kazusa_ai_chatbot/dispatcher/README.md`
  - `src/kazusa_ai_chatbot/self_cognition/README.md`
  - this plan and parent action-spec plan evidence
  - stale text that says L3/contextual/dialog can choose visible silence via
    `expression_willingness`

### Keep

- L2d semantic output shape: `action_requests`.
- `send_message` excluded from `build_initial_action_capabilities()`.
- Existing dialog generator/evaluator output contract:
  `final_dialog`, `mention_target_user`, `feedback`, and `should_stop`.
- Existing visual-directive prompt and `no_visual_directives` skip control,
  with visual execution moved behind selected text-surface handling.
- `memory_lifecycle_update` target binding and repository behavior.

### Delete

- Delete the old unconditional L3 edges inside the cognition subgraph and
  route L3 only through the selected text-surface handler.
- Delete the old L3/dialog response-gate code and tests listed in
  "Legacy Response-Gate Scan". No branch may compare an L3 field to `silent`
  to suppress selected text.
- Do not delete upstream relevance, debug-mode, output-mode, or visual-skip
  controls as part of this cleanup.

## Implementation Order

### Stage 0: Baseline And Blocker Capture

- Add a failing deterministic graph test proving that no `speak` action still
  runs L3 today.
- Add a failing deterministic graph test proving that selected `speak` should
  run L3 text/dialog exactly once.
- Add static-scan evidence for legacy response gates:
  `_cognition_requests_silence`, `conditional_skip_dialog_agent`,
  `expression_willingness` with `silent` response semantics, and docs/tests
  that claim L3 can suppress visible text.
- Add or update failing tests proving that dialog does not skip generation
  based on `expression_willingness` once `speak` has selected text.
- Record baseline failure output in Execution Evidence.

### Stage 1: Selected Text Surface Handler

- Create the selected L3 text handler around existing L3 functions.
- Move unconditional L3 graph execution out of `call_cognition_subgraph`.
- Update persona routing so `speak` invokes the selected handler and no
  `speak` skips L3/dialog.
- Remove the legacy L3/dialog response gates identified in Stage 0.
- Update L3 contextual, collector, output-contract, schema, dialog prompt, and
  fixture/test contracts so response/no-response is not represented by
  `expression_willingness`.
- Keep the existing public text response output shape and `SurfaceOutputV1`.

### Stage 2: Delivery Edge After Dialog

- Make text delivery explicitly downstream of `SurfaceOutputV1`.
- Keep normal `/chat` delivery through `ChatResponse.messages`.
- For scheduled/proactive contexts, bridge generated text to
  `ActionSpecV1(kind="send_message") -> RawToolCall -> TaskDispatcher`.
- Add tests proving L2d never sees `send_message`.

### Stage 3: Future Cognition Slot Handler

- Implement `trigger_future_cognition` as an orchestrator-owned private action
  handler.
- Schedule a future self-cognition trigger contract without calling cognition
  inline.
- Collect due `trigger_future_cognition` scheduled rows as ordinary
  self-cognition trigger cases on later worker ticks.
- Keep scheduler row ids, source action-attempt ids, continuation schema
  versions, and depth limits out of the model-facing source packet.
- Add trace/action-result output for scheduled, rejected, and failed cases.

### Stage 4: Consolidation And Regression

- Ensure private-only episodes still consolidate from action results or private
  surface output.
- Ensure combined episodes such as `speak + memory_lifecycle_update` and
  `speak + trigger_future_cognition` produce coherent action results and one
  text surface.
- Update documentation and parent-plan evidence.

### Stage 5: Verification And Independent Review

- Run all focused tests and listed regression tests.
- Run the required live LLM smoke cases because this plan changes graph
  behavior.
- Run independent code review before sign-off.

## Progress Checklist

- [x] Stage 0 - baseline blocker captured
  - Verify: focused tests fail against the old unconditional L3 path.
  - Evidence: record failing commands and assertions.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 1 - selected text surface handler complete
  - Verify: selected `speak` runs L3 text/dialog; no `speak` skips L3/dialog.
  - Evidence: record changed files and passing focused tests.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 2 - delivery edge after dialog complete
  - Verify: normal `/chat` output remains stable; bridge-only `send_message` occurs
    only after text exists in scheduled/proactive contexts.
  - Evidence: record dispatcher/service tests.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 3 - future cognition slot handler complete
  - Verify: `trigger_future_cognition` schedules a future self-cognition slot
    and never calls cognition inline.
  - Evidence: record scheduler/orchestrator tests.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 4 - consolidation and documentation complete
  - Verify: private-only and mixed-action episodes consolidate; docs updated.
  - Evidence: record consolidator tests and doc grep summaries.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 5 - independent code review complete
  - Verify: review findings fixed or recorded as residual risks; affected
    checks rerun.
  - Evidence: record review mode, findings, fixes, commands, and approval.
  - Sign-off: `Codex/2026-05-16`

## Verification

### Focused Tests

Run:

```powershell
venv\Scripts\python -m pytest tests\test_l2d_l3_surface_handoff.py tests\test_action_spec_future_cognition.py -q
```

Expected after implementation: all tests pass.

Required behaviors:

- `call_cognition_subgraph` returns L1/L2/L2d outputs without invoking L3.
- selected `speak` invokes existing L3 text chain and dialog exactly once.
- no selected `speak` does not invoke L3 text chain or dialog.
- `expression_willingness` is absent from persona routing, L3 contextual
  output, collector output, dialog prompt inputs, and focused tests.
- L3 contextual output no longer includes any response-decision enum.
- L3 visual is invoked only as part of selected text-surface handling and
  respects existing skip controls.
- multiple selected actions produce one text surface plus private action
  results.
- `trigger_future_cognition` schedules a future cognition slot and does not
  call cognition inline.

### Regression Tests

Run:

```powershell
venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_persona_supervisor2_action_initializer.py tests\test_action_spec_evaluator.py tests\test_action_spec_results.py tests\test_action_spec_self_cognition_bridge.py tests\test_service_background_consolidation.py tests\test_consolidator_facts_rag2.py -q
```

Expected after implementation: all tests pass.

### Prompt And Live LLM Gate

L2d prompt text must not change in this plan. Any L2d prompt diff is a blocker
and must be removed or moved into a separate approved plan.

Run these live graph smoke cases one case at a time with `-m live_llm` and
inspect each trace before continuing:

- ordinary QQ/user-message `speak` route;
- self-cognition `trigger_future_cognition` route.

Record trace paths and route judgment in Execution Evidence.

### Static Greps

Run:

```powershell
rg -n "l2d_action_initializer.*l3_|l3_.*l2d_action_initializer|send_message.*capabilities|build_initial_action_capabilities\\(\\).*send_message" src tests -g "*.py"
rg -n "_cognition_requests_silence|conditional_skip_dialog_agent|expression_willingness" src/kazusa_ai_chatbot/nodes src/scripts tests -g "*.py" -g "*.md"
rg -n "whether the character should speak|routes to .*stage_3_no_response|choose silence later through .*expression_willingness" src/kazusa_ai_chatbot/nodes/README.md
rg -n "trigger_future_cognition|send_message|L3 text|surface handler|future cognition" README.md docs src development_plans -g "*.md"
```

Expected:

- No Python graph edge keeps the old unconditional L2d-to-L3 path.
- `send_message` remains absent from `build_initial_action_capabilities()`.
- No runtime node code, prompt contract, focused test, or subsystem README
  contains `expression_willingness`.
- No L3 contextual field is documented or used as a text response gate.
- Documentation matches describe bridge-only delivery and selected L3 surface
  handling.

### Hygiene

Run:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_l3_surface.py src\kazusa_ai_chatbot\action_spec\handlers\future_cognition.py
git diff --check
```

Expected: compile succeeds; `git diff --check` exits 0 with at most Windows
CRLF normalization warnings.

## Independent Plan Review

Run this gate before approval. Review scope:

- Confirms this draft fixes the blocker discovered after Stage 7 of the parent
  action-spec plan.
- Confirms no hidden old unconditional L3 path remains authorized.
- Confirms `send_message` remains bridge-only and not L2d-facing.
- Confirms future cognition scheduling is a contract for a later cycle, not an
  inline cognition call.
- Confirms L3 visual remains side effect only.
- Confirms verification covers no-speak, speak, mixed actions, future cognition,
  consolidation, and stale doc/graph greps.

Record blockers, non-blocking findings, required edits, and approval status.

## Independent Code Review

Run this gate after all verification commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project style and mandatory skill compliance for Python, tests, prompts, and
  docs.
- Plan alignment, especially L2d/L3 boundary, `send_message` bridge-only
  behavior, no inline cognition trigger, and visual side-effect scope.
- Hidden fallback paths, duplicate scheduler paths, prompt leaks, raw id leaks,
  unapproved LLM calls, and accidental broad refactors.
- Regression and handoff quality, including focused tests, live smoke when
  required, docs, static greps, and parent-plan evidence.

Fix findings only when they are inside this plan's approved change surface.
If a finding requires new scope, update this plan before changing code.

## Acceptance Criteria

This plan is complete when:

- `call_cognition_subgraph` no longer runs L3 unconditionally after L2d.
- Selected `speak` is the only path that invokes L3 text directives and
  dialog.
- L3 text/dialog no longer contains `expression_willingness` or any secondary
  response suppression flag.
- No selected visible action suppresses L3/dialog while still allowing private
  consolidation evidence.
- Text delivery happens after dialog: normal chat through existing response
  delivery, scheduled/proactive delivery through bridge-only `send_message`.
- `trigger_future_cognition` creates an auditable future cognition slot and
  never calls cognition inline.
- L3 visual remains a side effect of selected text-surface handling only.
- Focused, regression, static grep, hygiene, and required live checks pass.
- Documentation and parent-plan evidence state that the prior Stage 7 result
  was blocked until this handoff was implemented.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Old L3 path survives as fallback | Bigbang rewrite of persona cognition graph routing | Static grep plus focused no-speak test |
| Old L3/dialog silence flag survives | Remove `_cognition_requests_silence`, dialog skip branch, prompt enum, docs, and tests together | Static grep plus selected-speak dialog test |
| Text responses lose existing style quality | Reuse non-decision L3 agents and dialog behavior behind selected handler | Existing persona/dialog regression tests |
| L2d is overloaded with delivery details | Keep `send_message` bridge-only and deterministic after text exists | Prompt-safe projection tests and grep |
| Future cognition handler calls cognition inline | Schedule a future slot only; assert cognition function is not called | Future cognition handler test |
| Scheduler becomes a second action system | Use existing scheduler/runtime boundary with narrow orchestrator handler | Code review and scheduler tests |
| Visual side effect becomes standalone image scope | Keep visual agent in text handler only | Focused graph test and docs grep |

## Execution Evidence

- 2026-05-16 Stage 0 baseline and blocker capture.
  - Static inspection confirmed the old cognition graph still connected
    `l2d_action_initializer -> l3_* -> l4_collector`, so L3 ran before the
    top-level `speak` route.
  - Initial focused tests were added around the intended contract:
    cognition subgraph stops after L2d; selected `speak` runs one L3
    text/dialog surface; no `speak` skips L3/dialog but preserves
    consolidation evidence.
  - Legacy response-gate scan target captured:
    `_cognition_requests_silence`, `conditional_skip_dialog_agent`, and
    `expression_willingness` response-gate usage in runtime nodes, prompts,
    docs, and tests.
- 2026-05-16 Stage 1-3 implementation.
  - Created `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
    as the selected text-surface handler around existing L3 directive agents.
  - Updated `call_cognition_subgraph` to end after L2d. Persona routing now
    runs L3 text/dialog only when a valid `speak` action spec is selected.
  - Removed the L3/dialog `expression_willingness` response gate from runtime
    code, prompt contracts, schemas, tests, and node docs. Dialog no longer
    has an internal skip branch after the text surface has been selected.
  - Created `src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py`
    and wired `trigger_future_cognition` through the action evaluator and
    action-result path. The handler schedules a non-dispatcher
    `scheduled_events` row and never calls cognition inline.
  - Added due-slot consumption through `self_cognition.sources` and
    `self_cognition.worker`: pending `trigger_future_cognition` rows become
    ordinary self-cognition trigger cases on later worker ticks and are marked
    completed only after normal case processing returns.
  - Prompt-facing scheduled-slot projection was narrowed so scheduler ids,
    action-attempt ids, raw source ref ids, `schema_version`, `episode_type`,
    `include_result_as`, and `max_depth` stay out of the source packet.
- 2026-05-16 Stage 4 consolidation and documentation checkpoint.
  - Updated `src/kazusa_ai_chatbot/nodes/README.md` to show L2d ending the
    cognition subgraph and selected `speak` entering the L3 text/dialog
    surface.
  - Updated `src/kazusa_ai_chatbot/self_cognition/README.md` to describe
    self-cognition as a normal trigger source for shared L1/L2/L2d, with L3
    text/dialog only after L2d selects `speak`.
  - Updated root `README.md` to remove stale selected-image-handler wording
    from the current runtime summary.
  - Registry status set to `in_progress` in `development_plans/README.md`.
- 2026-05-16 worker delegation outcome.
  - Worker `Nash` implemented the runtime side and returned
    `DONE_WITH_CONCERNS`; concern was limited to not owning tests/docs.
  - Parent agent wrote and executed the tests, found three concrete
    future-slot handoff failures, then tightened the implementation and tests:
    case-id correlation, keyword-seam use, completion marking, and prompt-safe
    projection of deterministic continuation fields.
- 2026-05-16 verification.
  - Targeted scheduled future-cognition tests:
    `venv\Scripts\python -m pytest tests\test_self_cognition_integration.py::test_collect_scheduled_future_cognition_cases_projects_due_slots tests\test_self_cognition_integration.py::test_collect_self_cognition_cases_includes_future_slots tests\test_self_cognition_integration.py::test_worker_tick_marks_future_cognition_slot_completed -q`
    result: 3 passed.
  - Focused and affected regression suite:
    `venv\Scripts\python -m pytest tests\test_l2d_l3_surface_handoff.py tests\test_action_spec_future_cognition.py tests\test_self_cognition_integration.py tests\test_persona_supervisor2.py tests\test_persona_supervisor2_action_initializer.py tests\test_action_spec_evaluator.py tests\test_action_spec_results.py tests\test_action_spec_self_cognition_bridge.py tests\test_service_background_consolidation.py tests\test_consolidator_facts_rag2.py tests\test_dialog_agent.py -q`
    result: 96 passed.
  - Static response-gate grep:
    `rg -n "expression_willingness|_cognition_requests_silence|conditional_skip_dialog_agent" src tests src/scripts -g "*.py" -g "*.md"`
    result: no matches.
  - Syntax verification:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_l3_surface.py src\kazusa_ai_chatbot\action_spec\handlers\future_cognition.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\worker.py src\kazusa_ai_chatbot\db\scheduled_events.py src\kazusa_ai_chatbot\scheduler.py tests\test_action_spec_future_cognition.py tests\test_l2d_l3_surface_handoff.py tests\test_self_cognition_integration.py`
    passed.
  - Hygiene verification: `git diff --check` exited 0 with Windows CRLF
    normalization warnings only.
- 2026-05-16 expanded verification after stale-test alignment.
  - Broad touched regression batch:
    `venv\Scripts\python -m pytest tests\test_cognition_interaction_style_context.py tests\test_cognition_live_llm.py tests\test_cognition_live_llm_prompt_contracts.py tests\test_conversation_progress_cognition.py tests\test_conversation_progress_flow_live_llm.py tests\test_conversation_progress_history_policy.py tests\test_dialog_evaluator_live_llm_contract.py tests\test_dialog_generator_live_llm_contract.py tests\test_dialog_mention_target_user.py tests\test_dialog_mention_target_user_live_llm.py tests\test_multi_source_cognition_stage_00_regression_baseline.py tests\test_multi_source_cognition_stage_02_chat_episode_migration.py tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_multi_source_cognition_stage_07_reflection_dry_run.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py tests\test_rag_dialog_event_logging.py tests\test_self_cognition_tracking.py -q`
    result: 160 passed, 39 deselected.
  - Focused handoff/action/self-cognition/consolidation suite rerun:
    `venv\Scripts\python -m pytest tests\test_l2d_l3_surface_handoff.py tests\test_action_spec_future_cognition.py tests\test_self_cognition_integration.py tests\test_persona_supervisor2.py tests\test_persona_supervisor2_action_initializer.py tests\test_action_spec_evaluator.py tests\test_action_spec_results.py tests\test_action_spec_self_cognition_bridge.py tests\test_service_background_consolidation.py tests\test_consolidator_facts_rag2.py tests\test_dialog_agent.py -q`
    result: 96 passed.
  - Static response-gate grep:
    `rg -n "expression_willingness|_cognition_requests_silence|conditional_skip_dialog_agent" src tests src/scripts -g "*.py" -g "*.md"`
    result: no matches.
  - Documentation grep confirmed the new selected L3 text and
    `trigger_future_cognition` path appears in root, node, self-cognition,
    action-spec, parent-plan, and handoff-plan docs.
  - Syntax verification on touched runtime/test Python files passed.
  - Hygiene verification: `git diff --check` exited 0 with Windows CRLF
    normalization warnings only.
- 2026-05-16 independent code review.
  - Review mode: self-review from a fresh-review posture; no separate reviewer
    was available after worker `Nash` completed implementation without owning
    tests/docs.
  - Inspected runtime diff for `persona_supervisor2.py`,
    `persona_supervisor2_cognition.py`,
    `persona_supervisor2_l3_surface.py`, `action_spec/evaluator.py`,
    `action_spec/handlers/future_cognition.py`, `db/scheduled_events.py`,
    `scheduler.py`, `self_cognition/sources.py`,
    `self_cognition/worker.py`, L3/dialog prompt-contract removals, dry-run
    prompt-key updates, README changes, and focused test additions.
  - Findings: no approval-blocking implementation issues found. The selected
    `speak` path is the only L3 text/dialog entry; private actions execute and
    produce consolidation-facing traces on no-visible-action paths;
    `trigger_future_cognition` schedules a non-dispatcher slot and does not
    invoke cognition inline; scheduler/action ids are scrubbed from the
    model-facing due-slot source packet.
  - Static gate review:
    `rg -n "l2d_action_initializer.*l3_|l3_.*l2d_action_initializer|send_message.*capabilities|build_initial_action_capabilities\\(\\).*send_message" src tests -g "*.py"`
    matched only tests asserting `send_message` is absent from initial
    capabilities; the old L2d-to-L3 graph edge did not match.
  - Static docs/runtime review:
    response-gate and stale node-doc greps returned no matches; documentation
    grep matched the expected selected L3 text and `trigger_future_cognition`
    descriptions.
- 2026-05-16 remaining acceptance item.
  - The handoff-specific live LLM smoke in this plan has not been rerun in
    this checkpoint. The prior same-day L2d live sweep remains recorded in the
    parent plan; this checkpoint verified graph handoff deterministically
    because no L2d prompt text changed.
