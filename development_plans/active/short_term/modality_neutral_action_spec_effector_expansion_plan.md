# Modality Neutral Action Spec And Effector Expansion Execution Plan

## Summary

- Goal: implement the first runtime slice of the modality-neutral action spec:
  action contracts, shared validation, L2d semantic action initialization, L3
  surface-handler routing, episode-trace consolidation, `send_message` bridge
  compatibility, and character-selected lifecycle updates for
  `user_memory_units.active_commitment`.
- Plan class: high_risk_migration
- Status: in_progress
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: compatible
- Highest-risk areas: live cognition schema drift, action-ledger compatibility,
  L2d action initialization, L3 surface-handler routing, action/result trace
  propagation into consolidation, memory lifecycle writes, and accidental
  side-channel execution
- Acceptance criteria: one shared action contract exists, `send_message`
  remains compatible, expired promises can be retired only through
  character-selected lifecycle action, action/surface outcomes reach
  consolidation through an episode trace, and deferred tools have bounded
  handoff notes

This plan is approved for execution as of 2026-05-16. Execution must follow the
stage order, verification gates, and evidence requirements below.

## Context

Use
`development_plans/reference/designs/action_spec_effector_expansion_architecture.md`
as the design source for rationale, research context, decision records, and
future tool brainstorming.

This active plan is only the execution contract. Do not add design research,
new alternatives, or extra capability scope here during implementation.

## Mandatory Skills

Load these skills before execution:

- `development-plan-writing`: before changing this plan, registry rows, or
  execution evidence.
- `local-llm-architecture`: before changing cognition schema, prompts, graph
  state, action routing, evaluator behavior, or background LLM behavior.
- `no-prepost-user-input`: before changing promise, preference, commitment, or
  action persistence decisions.
- `py-style`: before editing Python files.
- `test-style-and-execution`: before adding, changing, or running tests.
- `cjk-safety`: before editing Python files containing CJK prompt text or CJK
  string literals.
  This applies in this plan only if Stage 3 edits Python prompt strings with
  CJK content; the English-only `action_spec` module still follows normal
  Python style.

## Mandatory Rules

- After any context compaction, reread this entire plan before continuing.
- After signing off a progress checklist stage, reread this entire plan before
  starting the next stage.
- Before final completion or lifecycle status changes, run the `Independent Code
  Review` gate and record the result in `Execution Evidence`.
- Use `venv\Scripts\python` for Python commands.
- Do not read `.env` unless the user explicitly asks for environment
  inspection.
- Keep one shared semantic path:

```text
typed episode -> RAG/evidence -> cognition L1/L2a/L2b/L2c
-> L2d action initialization -> action spec materialization/evaluation
-> selected L3 surfaces/action handlers -> action results + surface outputs
-> episode-trace consolidation -> persistence/scheduler/adapter bridge
```

- LLM stages own semantic judgment. Deterministic code owns schema validation,
  permissions, limits, execution eligibility, persistence, cache invalidation,
  scheduling, adapter delivery, and audit records.
- Do not remove, hide, or suppress promises based only on deterministic age, due
  date, keyword, or status.
- Do not add a new graph branch, prompt sidecar, fake user message, or separate
  cleanup channel for expired promises.
- Tool handlers must not call cognition directly and must not write final
  user-facing dialogue.
- Keep action specs optional in live cognition output until offline structured
  emission measurement is reviewed.
- Do not make L2a Consciousness or L2c Judgment emit `ActionSpecV1` records.
  L2a owns deliberative interpretation, L2b owns boundary assessment, and L2c
  owns final stance/intent adjudication. Action selection belongs to L2d.
- Add L2d as the action initializer after L2c. L2d is analogous to the RAG
  initializer: it chooses which registered action capabilities are needed, but
  does not run specialists or generate low-level handler mechanics.
- Treat L2d prompt output as semantic `action_requests` with cardinality zero,
  one, or many. Deterministic materialization converts valid requests into
  graph-visible `action_specs`.
- Treat L3 text and L3 image as registered action handlers for surface actions,
  not as an always-on layer outside the action contract. L3 handlers run only
  for selected action specs that require that surface.
- Treat self-cognition as a typed trigger source for the same path. Do not make
  self-cognition a downstream action consumer, private cleanup channel, or
  action executor.
- Treat final dialog as one `SurfaceOutputV1`, not as the sole consolidation
  input or the only reason post-turn work may run.
- Feed selected action specs, action attempts, action results, and surface
  outputs into consolidation through an episode trace. The consolidator must
  consume prompt-safe action projections, not raw handler params.
- The consolidator must not select actions, execute actions, call dispatcher,
  call scheduler, or call cognition. It may persist memory/state through its
  existing owners after consuming episode-trace evidence.
- L2d may consume prompt-safe `trigger_context`; trigger source may influence
  the semantic action request, but it never grants permission, selects handlers
  directly, generates execution parameters, or exposes adapter/database
  internals.
- Deterministic target binding must resolve action targets from trusted episode,
  trigger-source, RAG, and repository context. L2d must not receive, copy,
  compare, or emit raw target IDs, source-ref IDs, persistence IDs, owner names,
  collection names, adapter IDs, scheduler IDs, or handler IDs.
- `memory_lifecycle_update` may be materialized only when deterministic code
  has exactly one eligible active-commitment target. If no single target is
  bound, hide the lifecycle capability from the prompt when possible or reject
  the request before persistence.
- Do not execute this plan until its status is explicitly changed to
  `approved` or `in_progress`.

## Must Do

- Execute this approved plan only through the stage order, verification gates,
  and evidence requirements below.
- Create `src/kazusa_ai_chatbot/action_spec/` with public action-spec contracts,
  capability registry, evaluator, attempt ledger, and memory lifecycle handler.
- Add `ActionSpecV1` with `cognition_mode`, `continuation`, source refs, target,
  params, urgency, visibility, deadline, and reason.
- Add result/trace contracts for `ActionResultV1`, `SurfaceOutputV1`,
  `EpisodeTraceV1`, and `ConsolidationActionProjectionV1`.
- Accept only `cognition_mode="deliberative"` in this plan. Reject `reflex`.
- Add `ActionSpecEvaluator` as the shared deterministic gate for action specs.
- Add prompt-safe capability projection for cognition.
- Reuse `self_cognition_action_attempts` through a generic action-attempt
  repository. Do not create a new action ledger collection.
- Preserve existing `send_message` behavior by bridging validated
  `ActionSpecV1(kind="send_message")` to existing `RawToolCall` and
  `TaskDispatcher.dispatch`; do not include `send_message` in the L2d-facing
  `build_initial_action_capabilities()` registry.
- Add `memory_lifecycle_update` only for `user_memory_units.active_commitment`.
- Add `trigger_future_cognition` as an orchestrator-owned private request for
  a bounded future cognition episode. This capability creates a contract for a
  later cycle; it must not directly call cognition from the tool handler.
- Add L2d action initialization after L2c, before any L3 surface handler.
- Ensure L2d prompt output emits semantic `action_requests`, not
  `ActionSpecV1`. Deterministic materialization must wrap valid semantic
  requests into `action_specs: list[ActionSpecV1]` for downstream graph state.
- Do not expose deterministic fields such as schema versions, target owners,
  source refs, continuation envelopes, adapter fields, persistence IDs, or
  handler IDs in the L2d prompt contract.
- Add deterministic target binding for targetful actions. For
  `memory_lifecycle_update`, bind the target from trusted active-commitment
  context before materialization; do not recover the target from L2d output
  text, copied IDs, array indexes, or source-ref selectors.
- Support multiple action specs per cognition episode with a hard cap of 3 in
  this plan.
- Model L3 text and L3 image as action handlers. This plan may document both,
  but runtime implementation remains limited to the approved capabilities and
  must not call an external image generation service.
- Model self-cognition as a trigger-source episode that flows through L1/L2/L2d,
  action evaluation, handlers, and consolidation like user-message episodes.
- Replace the final-dialog-only post-turn/consolidation assumption with an
  episode-trace gate: `final_dialog`, surface outputs, action results, or
  private finalization can make a turn consolidatable.
- Project action specs/results/surface outputs into the consolidator as
  prompt-safe evidence. Do not expose raw handler IDs, raw adapter IDs,
  credentials, raw collection names, or arbitrary action params.
- Add prompt-safe trigger context projection for L2d.
- Add a narrow `update_user_memory_unit_lifecycle` helper in
  `src/kazusa_ai_chatbot/db/user_memory_units.py`.
- Map lifecycle statuses:
  - `fulfilled -> completed`
  - `abandoned -> cancelled`
  - `obsolete -> archived`
  - `deferred -> active`
- Reject `EvolvingMemoryDoc` lifecycle targets without persistence.
- Run offline action-spec validity and latency measurement before enabling
  broad live `/chat` action-spec emission.
- Update the measured documentation surface before final sign-off. Documentation
  updates are part of the execution contract, not a cleanup task.

## Deferred

- Do not implement `schedule_self_check` runtime execution.
- Do not implement `web_research` or `fetch_url` runtime execution.
- Do not implement `note_open_loop` or `close_open_loop` runtime execution.
- Do not implement image generation.
- Do not implement external L3 image generation service calls. L3 image may be
  documented or registered as a future surface handler only.
- Do not change platform adapters.
- Do not change proactive-contact permission policy.
- Do not mutate `EvolvingMemoryDoc`.
- Do not rename, migrate, or delete MongoDB collections.
- Do not add arbitrary file, database, shell, HTTP, or external-message tools.
- Do not add deterministic cleanup jobs that retire promises by age.
- Do not make the consolidator an action planner, action executor, dispatcher
  bridge, scheduler bridge, or cognition trigger.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Contracts reference | compatible prerequisite | Prerequisite satisfied on 2026-05-16; execute only through this approved plan's stage order and gates. |
| Action spec module | compatible additive | Add `action_spec` without removing dispatcher, self-cognition, or consolidator contracts. |
| `send_message` path | compatible bridge | Convert validated action specs to `RawToolCall` only at the bridge. Preserve `TaskDispatcher.dispatch`. |
| Action attempts ledger | compatible additive | Reuse `self_cognition_action_attempts`; add fields tolerantly. |
| Consolidation input | compatible additive | Add episode-trace/action-result projection while preserving current text consolidation until trace-aware gates pass. |
| `user_memory_units` lifecycle | compatible additive | Use existing statuses and one narrow lifecycle helper. No backfill. |
| Live cognition output | compatible gated | Keep action specs optional until offline measurement is reviewed. |
| Future tools | deferred | Do not ship runtime future tools under this plan. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- Compatibility surfaces are limited to the rows listed above.
- Any collection rename, data migration, physical deletion, or broad live-chat
  rollout requires a separate approved plan or explicit plan revision.
- If an existing helper cannot support lifecycle update safely, stop and update
  this plan instead of adding an ad hoc MongoDB write.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the contracts and owner boundaries in this plan.
- The agent must not introduce alternate migration strategies, fallback paths,
  feature flags, extra tools, or broad compatibility layers.
- The agent must not implement deferred tools.
- Changes outside the listed `Change Surface` require explicit plan revision.
- If equivalent validation, projection, or ledger behavior exists, adapt it into
  the approved module boundary instead of duplicating it.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

After execution:

- Cognition output may include optional `action_specs: list[ActionSpecV1]`
  records selected by L2d, with zero, one, or many entries.
- L2a and L2c do not emit action specs directly.
- L2d consumes final L2 state, prompt-safe trigger context, prompt-safe
  capability projection, and relevant evidence to select registered actions.
- L3 text and L3 image are represented as registered action handlers for
  surface action specs. Surface handlers run only when selected by L2d/action
  routing.
- Existing final-dialog and `action_directives` behavior remains compatible, but
  final dialog is represented as one text surface output in the episode trace.
- Self-cognition enters as a typed trigger source and then follows the same
  L1/L2/L2d/action/consolidation path as user-message episodes.
- Action specs, action attempts, action results, and surface outputs are
  available to consolidation through prompt-safe episode-trace projections.
- Consolidation is no longer gated only by text. A private action result,
  scheduled action result, image/text surface output, or private finalization can
  make the episode eligible for consolidation.
- Validated `send_message` action specs use the existing dispatcher path.
- Validated `memory_lifecycle_update` action specs can update
  `user_memory_units.active_commitment` lifecycle state.
- Action attempts are recorded through the existing
  `self_cognition_action_attempts` backing collection.
- Deferred tools are documented but not runnable.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Action spec ownership | `ActionSpecV1` is the single validated graph-visible action residue. Do not add `ActionPlanV1` or `ActionIntentV1`. | Keeps the execution contract thin and prevents parallel planning shapes. |
| Action initializer | Add L2d after L2c as the cognition stage that emits semantic `action_requests`, which deterministic code materializes into `action_specs`. | L2d is the first stage after consciousness, boundary, and final judgment are all available, but weak local LLMs should not author execution envelopes. |
| Multi-action cardinality | `action_requests` and materialized `action_specs` are lists with zero, one, or many entries, capped at 3 in this plan. | The character may legitimately choose multiple independent actions, such as memory lifecycle update plus private note plus surface reply. |
| L2d prompt boundary | L2d selects capability and semantic decision only; deterministic materializers inject schema versions, target owners, source refs, continuation defaults, persistence IDs, and executable params. | Preserves LLM-owned semantics while keeping mechanics inspectable and reducing structured-output burden on the local model. |
| Target binding boundary | Target IDs and source refs are bound by deterministic code from trusted episode/source/RAG/repository context, not from L2d-emitted text. | Prevents local LLMs from processing opaque IDs while still letting the character choose lifecycle semantics. |
| L3 surface ownership | Treat L3 text and L3 image as registered action handlers for surface actions. | This makes response, no-response, image, delayed message, and future surfaces modular instead of always-on branches. |
| Trigger context | L2d consumes prompt-safe trigger context. | Trigger source informs agency and visibility without granting permission or leaking adapter/database internals. |
| Self-cognition integration | Treat self-cognition as an upstream trigger source, not as an action consumer. | Keeps internal self-checks on the same L1/L2/L2d/action/consolidation path and avoids a private cleanup pipeline. |
| Consolidation basis | Consolidate from an episode trace containing cognition residue, selected actions, action results, and surface outputs. | Text-only consolidation loses private actions, parallel actions, scheduled follow-ups, no-reply decisions, and future image/tool surfaces. |
| Consolidator boundary | The consolidator consumes action evidence but does not plan, execute, schedule, dispatch, or trigger cognition. | Preserves LLM-owned action semantics and deterministic execution ownership while still allowing durable memory/state to reflect what happened. |
| Direct L2 emission | L2a and L2c must not emit action specs. | L2a is too early and L2c is a stance/intent adjudicator, not an action initializer. |
| Dispatcher boundary | Keep `send_message` out of the initial L2d-facing capability registry; bridge only validated delivery specs into existing `RawToolCall` and `TaskDispatcher.dispatch`. | Preserves adapter-facing delivery while preventing L2d from confusing text-surface selection with adapter delivery. |
| Future cognition trigger | Add `trigger_future_cognition` as an orchestrator-owned private action request. | Provides a first-class contract for a later cognition cycle without allowing tools to call cognition directly. |
| Memory lifecycle owner | Route `memory_lifecycle_update` to the user-memory owner, not dispatcher. | Promise retirement is private persistence, not adapter-facing delivery. |
| Promise retirement semantics | Let L2d choose `fulfilled`, `abandoned`, `obsolete`, or `deferred`; deterministic code maps only validated decisions to collection statuses. | Avoids age-based or keyword-based cleanup while keeping writes inspectable. |
| Action ledger | Back generic action attempts with `self_cognition_action_attempts`. | Reuses the existing idempotency collection and avoids a second control ledger. |
| Live rollout | Keep action specs optional and gated by offline validity and latency measurement. | Protects live chat from local-LLM structured-output drift. |
| Future tools | Reserve next-stage tools in documentation only. | Prevents this plan from shipping web research, notes, image generation, or scheduler cognition loops. |

## Contracts To Implement

Use `TypedDict`, `Literal`, dataclasses, and explicit validator functions for
internal action contracts. Do not introduce Pydantic for this internal module.

The implementation must mirror the approved
`development_plans/reference/designs/cognition_contracts_design.md` contract
names. The inline shapes below are the executable slice for this plan. If these
shapes conflict with the approved reference at implementation time, stop and
update the plan instead of silently choosing one.

### Supporting Types

```python
PolicyRefV1 = str


class EvidenceRefV1(TypedDict):
    schema_version: Literal["evidence_ref.v1"]
    evidence_kind: Literal[
        "conversation_message",
        "memory_unit",
        "reflection_run",
        "scheduled_event",
        "tool_result",
        "system_event",
        "external_document",
    ]
    evidence_id: str
    owner: str
    excerpt: str | None
    observed_at: str | None


class ActionSourceRefV1(TypedDict):
    schema_version: Literal["action_source_ref.v1"]
    ref_kind: Literal[
        "conversation_message",
        "memory_unit",
        "reflection_run",
        "scheduled_event",
        "tool_result",
        "cognitive_episode",
        "system_event",
    ]
    ref_id: str
    owner: str
    relationship: Literal["basis", "target", "lineage", "result", "audit"]
    evidence_refs: list[EvidenceRefV1]


class ActionTargetV1(TypedDict):
    schema_version: Literal["action_target.v1"]
    target_kind: Literal[
        "current_user",
        "current_channel",
        "user",
        "channel",
        "memory_unit",
        "cognitive_episode",
        "self",
        "none",
    ]
    target_id: str | None
    owner: str
    scope: dict[str, object]


class ActionContinuationV1(TypedDict):
    schema_version: Literal["action_continuation.v1"]
    mode: Literal[
        "none",
        "immediate_followup",
        "scheduled_followup",
        "background_followup",
    ]
    episode_type: str | None
    max_depth: int
    include_result_as: str | None


class CapabilitySpecV1(TypedDict):
    schema_version: Literal["capability_spec.v1"]
    capability_kind: str
    category: Literal["action"]
    owner_module: Literal[
        "dispatcher",
        "memory_lifecycle",
        "orchestrator",
        "l3_text",
        "l3_image",
    ]
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    handler_id: str
    lifecycle_hooks: list[str]
    permission_policy: PolicyRefV1
    rate_limit_policy: PolicyRefV1
    audit_policy: PolicyRefV1
    prompt_projection_policy: PolicyRefV1
```

### `TriggerContextV1`

L2d receives this prompt-safe projection from the `CognitiveEpisode`. It must
not include raw adapter IDs, database collection names, handler IDs,
credentials, raw channel IDs, or raw platform-specific payloads.

```python
class TriggerContextV1(TypedDict):
    trigger_source: Literal[
        "user_message",
        "internal_thought",
        "self_cognition",
        "scheduled_tick",
        "tool_result",
    ]
    input_sources: list[str]
    output_mode: str
    target_scope_summary: str
```

### `ActionRequestV1`

`ActionRequestV1` is the L2d prompt-output shape. It is intentionally semantic
and does not carry schema versions, target owners, source refs, persistence IDs,
handler IDs, continuation envelopes, or executable params.

```python
class ActionRequestV1(TypedDict, total=False):
    capability: str
    decision: str
    detail: str
    reason: str
```

Required runtime normalization:

- Drop non-object rows.
- Drop rows without non-empty `capability` and `reason`.
- Cap accepted rows at 3.
- Materialize supported capabilities into `ActionSpecV1` with deterministic
  target, params, source refs, and no-continuation defaults.

### `ActionSpecV1`

```python
class ActionSpecV1(TypedDict):
    schema_version: Literal["action_spec.v1"]
    kind: str
    cognition_mode: Literal["deliberative", "reflex"]
    source_refs: list[ActionSourceRefV1]
    target: ActionTargetV1
    params: dict[str, object]
    urgency: Literal["now", "background", "scheduled"]
    visibility: Literal["private", "preview", "user_visible"]
    deadline: str | None
    continuation: ActionContinuationV1
    reason: str
```

`ActionSpecV1` records materialized from L2d semantic requests are validated
action residues. The same shape is used for private actions, tool actions, and
surface actions. A surface action is routed to its registered L3 handler before
user-visible text, image prompt material, or delayed delivery payload exists.
Deterministic routing and validation decide which handler may consume a spec;
L2d only selects semantic actions from prompt-safe capability projections.

The initial action-spec list may contain multiple records. Order is priority
and audit order, not a guarantee that every handler executes sequentially.
Independent actions may run in the same cycle. Dependent chains must use the
`continuation` contract or produce a subsequent validated action spec through a
registered handler.

### `ActionEvalResult`

```python
class ActionEvalResult(TypedDict):
    ok: bool
    action_spec: ActionSpecV1 | None
    capability: CapabilitySpecV1 | None
    idempotency_key: str | None
    handler_owner: str | None
    errors: list[str]
```

### Action Results And Episode Trace

These contracts are produced by deterministic evaluators, action handlers, L3
surface handlers, and orchestration. L2d must not emit them.

```python
class ActionResultV1(TypedDict):
    schema_version: Literal["action_result.v1"]
    action_attempt_id: str
    action_kind: str
    handler_owner: str
    status: Literal[
        "rejected",
        "validated",
        "executed",
        "scheduled",
        "pending",
        "failed",
        "cancelled",
    ]
    visibility: Literal["private", "preview", "user_visible"]
    result_summary: str
    result_refs: list[EvidenceRefV1]
    continuation: ActionContinuationV1
    completed_at: str | None


class SurfaceOutputV1(TypedDict):
    schema_version: Literal["surface_output.v1"]
    surface_kind: Literal["text", "image", "audio", "motor", "tool", "private"]
    visibility: Literal["private", "preview", "user_visible"]
    action_attempt_id: str | None
    fragments: list[str]
    artifact_refs: list[EvidenceRefV1]
    delivery_intent: Literal["deliver_now", "deliver_later", "do_not_deliver"]
    created_at: str


class EpisodeTraceV1(TypedDict):
    schema_version: Literal["episode_trace.v1"]
    episode_id: str
    trigger_source: str
    cognition_refs: list[EvidenceRefV1]
    action_specs: list[ActionSpecV1]
    action_results: list[ActionResultV1]
    surface_outputs: list[SurfaceOutputV1]
    created_at: str


class ConsolidationActionProjectionV1(TypedDict):
    schema_version: Literal["consolidation_action_projection.v1"]
    action_kind: str
    status: str
    visibility: Literal["private", "preview", "user_visible"]
    semantic_decision: str
    result_summary: str
    evidence_refs: list[EvidenceRefV1]
```

The consolidator consumes `EpisodeTraceV1` or a prompt-safe projection of it.
It must not see handler IDs, credentials, raw adapter IDs, raw collection names,
or arbitrary action params.

### Initial Capability Params

The L2d-facing initial capability registry contains `speak`,
`memory_lifecycle_update`, and `trigger_future_cognition`. It must not expose
`send_message`; dispatcher delivery remains an internal bridge capability after
surface text exists.

`speak` capability params, owned by the L3 text handler:

```python
class SpeakParamsV1(TypedDict):
    delivery_mode: Literal[
        "visible_reply",
        "private_finalization",
        "delayed",
        "scheduled",
    ]
    execute_at: str | None
    surface_requirements: dict[str, object]
```

The L2d initializer may select `ActionSpecV1(kind="speak")` only when a text
surface is semantically needed. L3 text/dialog owns the wording. Dispatcher
handoff happens only after text exists and the resulting delivery action has
passed action-spec validation.

`trigger_future_cognition` capability params, owned by the orchestrator:

```python
class TriggerFutureCognitionParamsV1(TypedDict):
    episode_type: Literal["self_cognition"]
    trigger_at: str | None
    context_summary: str
```

The L2d initializer may select
`ActionSpecV1(kind="trigger_future_cognition")` only for a private bounded
request to create a future cognition episode. It must not directly call
cognition, bypass scheduler/orchestrator validation, or expose adapter/database
details. The initial target must be:

```python
{
    "schema_version": "action_target.v1",
    "target_kind": "cognitive_episode",
    "target_id": None,
    "owner": "orchestrator",
    "scope": {"episode_type": "self_cognition"},
}
```

Internal `send_message` bridge params:

```python
class SendMessageParamsV1(TypedDict):
    target_channel: Literal["same"] | str
    text: str
    execute_at: str | None
    delivery_mentions: list[dict[str, object]]
```

`send_message` is not in `build_initial_action_capabilities()`. The evaluator
must bridge validated `SendMessageParamsV1` through a separate bridge
capability set to the existing dispatcher
`RawToolCall(tool="send_message", args=...)` shape without exposing dispatcher
handler IDs to cognition prompts.

`memory_lifecycle_update` capability params:

```python
class MemoryLifecycleUpdateParamsV1(TypedDict):
    memory_kind: Literal["user_memory_unit"]
    unit_type: Literal["active_commitment"]
    unit_id: str
    lifecycle_decision: Literal[
        "fulfilled",
        "abandoned",
        "obsolete",
        "deferred",
    ]
    due_at: str | None
```

The action target for `memory_lifecycle_update` must be:

```python
{
    "schema_version": "action_target.v1",
    "target_kind": "memory_unit",
    "target_id": unit_id,
    "owner": "user_memory_units",
    "scope": {"unit_type": "active_commitment"},
}
```

Prompt-safe capability projection must include only semantic capability name,
availability, visibility, allowed lifecycle decisions, and a short parameter
summary. It must not include `handler_id`, raw MongoDB collection names,
credentials, raw channel IDs, adapter IDs, or database internals.

For targetful capabilities, prompt visibility is conditional on deterministic
binding. `memory_lifecycle_update` is prompt-visible only when materialization
can bind exactly one eligible `user_memory_units.active_commitment` target from
trusted runtime context. The prompt may receive the bound commitment's
plain-language summary, due date, due state, and status, but it must not receive
`unit_id`, source-ref IDs, owner names, repository names, collection names, or
array indexes. If the model nevertheless requests a lifecycle action when no
single target is bound, materialization must drop the action before evaluation
or persistence.

### Lifecycle Decision Vocabulary

Stage 3 L2d prompt guidance must use this exact vocabulary:

| L2 value | Meaning | Collection status |
|---|---|---|
| `fulfilled` | The character judges the commitment has been satisfied or should be recorded as completed because the promised action was carried out. | `completed` |
| `abandoned` | The character explicitly decides not to pursue the commitment anymore because continuing it would be inappropriate, unwanted, or no longer character-consistent. | `cancelled` |
| `obsolete` | Newer context makes the commitment irrelevant or superseded without implying failure or refusal. | `archived` |
| `deferred` | The character decides the commitment is still valid and should remain open for later handling. | `active` |

L2d must choose among these values semantically and provide `reason`.
Deterministic code only validates source refs, target ownership, transition
legality, and persistence mechanics. `deferred` must not suppress retrieval of
the commitment.

### `update_user_memory_unit_lifecycle`

```python
async def update_user_memory_unit_lifecycle(
    unit_id: str,
    *,
    status: Literal["active", "archived", "completed", "cancelled"],
    timestamp: str,
    reason: str,
    action_attempt_id: str,
    due_at: str | None = None,
) -> UserMemoryUnitDoc | None: ...
```

The helper updates only lifecycle fields, `updated_at`, and a `merge_history`
audit entry. It rejects missing units, non-`active_commitment` units, invalid
status transitions, and empty reasons.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/action_spec/__init__.py`
- `src/kazusa_ai_chatbot/action_spec/models.py`
- `src/kazusa_ai_chatbot/action_spec/registry.py`
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`
- `src/kazusa_ai_chatbot/action_spec/results.py`
- `src/kazusa_ai_chatbot/action_spec/attempt_ledger.py`
- `src/kazusa_ai_chatbot/action_spec/handlers/memory_lifecycle.py`
- `src/kazusa_ai_chatbot/action_spec/README.md`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
- `tests/test_action_spec_models.py`
- `tests/test_action_spec_evaluator.py`
- `tests/test_action_spec_results.py`
- `tests/test_action_spec_attempt_ledger.py`
- `tests/test_action_spec_memory_lifecycle.py`
- `tests/test_action_spec_self_cognition_bridge.py`
- `tests/test_consolidator_action_trace.py`
- `tests/test_persona_supervisor2_action_initializer.py`
- `tests/l2d_action_initializer_cases.py`
- `tests/test_l2d_action_initializer_cases.py`
- `tests/test_l2d_action_initializer_live_llm.py`
- `tests/fixtures/action_spec_cognition_cases.json`
- `src/scripts/capture_l2d_action_initializer_cases.py`
- `src/scripts/measure_action_spec_validity.py`

Runtime-only captured case sets are written under
`test_artifacts/l2d_action_initializer/` and are not committed. They may
contain private conversation text from QQ user/channel `673225019` or
self-cognition source packets.

`tests/fixtures/` already exists for JSON case data in this repo. This plan
uses it only for non-Python offline measurement fixtures. Python test modules
remain flat under `tests/`.

### Modify

- `development_plans/README.md`
- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/db/self_cognition.py`
- `src/kazusa_ai_chatbot/db/user_memory_units.py`
- `src/kazusa_ai_chatbot/self_cognition/handoff.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/db/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `src/kazusa_ai_chatbot/dispatcher/README.md`
- `src/kazusa_ai_chatbot/self_cognition/README.md`
- Conditional documentation files listed in `Documentation Change Surface
  Measurement` when implementation touches their contracts.

The `persona_supervisor2_cognition_l2.py` change is limited to feeding existing
L2a/L2b/L2c outputs into L2d. L2a and L2c must not emit action specs.

The `persona_supervisor2_cognition_l3.py` change is limited to recognizing L3
text and L3 image as action handlers for selected surface action specs while
preserving existing `action_directives` compatibility. L3 must not choose new
semantic actions, change lifecycle decisions, or choose execution owners.

The consolidator changes are limited to accepting prompt-safe episode-trace
projections. The consolidator must not execute action specs, call dispatcher,
call scheduler, or trigger cognition; it only consumes action/surface outcomes
as evidence for memory, relationship state, mood, vibe, and progress writes.

The `service.py` and `brain_service/post_turn.py` changes are limited to
replacing final-dialog-only post-turn gating with a consolidatable-output gate:
`final_dialog`, `surface_outputs`, `action_results`, or private finalization.

### Keep

- `src/kazusa_ai_chatbot/dispatcher/` remains the owner of adapter-facing
  tools. Only bridge code needed for `ActionSpecV1 -> RawToolCall` is allowed.
- `scheduled_events` schema remains unchanged.
- `EvolvingMemoryDoc` repository remains unchanged.
- Platform adapters remain unchanged.

### Delete

- No code, collection, or document is deleted in this plan.

## Documentation Change Surface Measurement

Measured on 2026-05-16 from the approved contract reference, this plan's code
change surface, and existing subsystem READMEs.

### Direct Documentation Updates

These files must be updated before the documentation stage can pass:

| Document | Why It Becomes Stale | Required Update |
|---|---|---|
| `src/kazusa_ai_chatbot/action_spec/README.md` | New public module. | Document package boundary, `ActionSpecV1`, `CapabilitySpecV1` action entries, evaluator, attempt ledger, dispatcher bridge, memory lifecycle handler, continuation policy, deferred tools, and forbidden paths. |
| `src/kazusa_ai_chatbot/nodes/README.md` | Cognition output can include optional action specs while preserving existing `action_directives`, and consolidator input is no longer text-only. | Document that L2d is the only action-spec initializer, L2a/L2c do not emit action specs, L3 text/image are registered surface handlers, L3/dialog still own wording, action specs remain optional/gated, and the consolidator consumes prompt-safe episode-trace projections. |
| `src/kazusa_ai_chatbot/dispatcher/README.md` | `send_message` can arrive through an action-spec bridge before becoming `RawToolCall`. | Document that dispatcher remains execution owner for adapter-facing tools, not the parent action system; explain `ActionSpecV1(kind="send_message") -> RawToolCall -> TaskDispatcher.dispatch`. |
| `src/kazusa_ai_chatbot/self_cognition/README.md` | Self-cognition becomes explicitly documented as a trigger source while legacy handoff and duplicate state route through the generic action-attempt contract. | Document self-cognition as a typed trigger for the shared path, action-spec validation before legacy `send_message` handoff, private `memory_lifecycle_update` for commitment retirement, and tolerant reuse of `self_cognition_action_attempts`. |
| `src/kazusa_ai_chatbot/db/README.md` | `user_memory_units.active_commitment` gains a lifecycle helper and `self_cognition_action_attempts` gains generic action-attempt metadata. | Document allowed lifecycle transitions, no age-based cleanup, action-attempt compatibility, and existing-status filtering for retired commitments. |
| `src/kazusa_ai_chatbot/brain_service/README.md` | Post-turn work and adapter delivery become surface/action-output aware. | Document that visible delivery follows surface outputs while consolidation can also run for private action results, scheduled action results, or private finalization. |
| root `README.md` | The high-level architecture diagram currently shows dialog as the only output before persistence/consolidation. | Update the architecture diagram and core boundary to show L2d action initialization, selected L3 surfaces/action handlers, action results/surface outputs, episode-trace consolidation, and adapter bridge delivery. |

### Conditional Documentation Audit

Audit these files during Stage 7. Update them only if implementation touches the
named boundary; otherwise record "no doc change required" with grep evidence:

| Document | Update Trigger |
|---|---|
| `src/kazusa_ai_chatbot/event_logging/README.md` | Any new event family, recorder, payload field, forbidden-data rule, or action-spec validation telemetry is added. |
| `src/kazusa_ai_chatbot/proactive_output/README.md` | Any proactive-contact permission, preview, outbox, or external-message boundary changes. |
| `src/kazusa_ai_chatbot/brain_service/README.md` | Any additional service startup, runtime adapter registration, scheduler callback, or delivery handoff behavior changes beyond the required action/surface-output update. |
| `src/kazusa_ai_chatbot/memory_evolution/README.md` | Any `EvolvingMemoryDoc` lifecycle mutation becomes supported. This plan should leave it unchanged. |
| `docs/HOWTO.md` | Any operator-facing command, config flag, API behavior, or runbook step is added beyond the developer-only measurement script. |
| root `README.md` | Additional public architecture table or package map entries become inaccurate after implementation. |

### Documentation Staleness Checks

Stage 7 must run static documentation greps and inspect each match:

```powershell
rg -n "only.*send_message|send_message.*only|currently send_message|Action candidates always use|dispatch_shape: \"send_message\"|future effector|one-effector" README.md docs src development_plans -g "*.md"
rg -n "ActionSpecV1|ActionResultV1|SurfaceOutputV1|EpisodeTraceV1|CapabilitySpecV1|memory_lifecycle_update|self_cognition_action_attempts|active_commitment|TaskDispatcher|RawToolCall|consolidation" src\kazusa_ai_chatbot\action_spec\README.md src\kazusa_ai_chatbot\nodes\README.md src\kazusa_ai_chatbot\dispatcher\README.md src\kazusa_ai_chatbot\self_cognition\README.md src\kazusa_ai_chatbot\db\README.md src\kazusa_ai_chatbot\brain_service\README.md README.md
```

Matches are not automatically failures. They must either describe the new
boundary accurately or be updated in the same stage.

## Overdesign Guardrail

- Actual problem: the character can notice an expired promise, but the system
  lacks a shared action contract for retiring that promise without speaking or
  deterministic age cleanup.
- Minimal change: add action contracts, shared evaluator, existing
  `send_message` bridge, L2d action initialization, L3 surface-handler routing,
  episode-trace projection into consolidation, and one private lifecycle
  capability for `user_memory_units.active_commitment`.
- Ownership boundaries: L2a interprets, L2b assesses boundaries, L2c
  adjudicates final stance/intent, L2d chooses zero-or-more semantic actions,
  L3 handlers realize selected surface actions, evaluator validates,
  dispatcher owns adapter-facing delivery, user-memory repository owns
  commitment lifecycle writes, consolidator consumes prompt-safe action/surface
  outcomes for memory/state persistence, and event logging owns sanitized
  observability.
- Rejected complexity: no arbitrary tools, no new ledger collection, no
  `EvolvingMemoryDoc` mutation, no schedule-self-check runtime tool, no web
  research runtime tool, no external image-generation runtime call, no
  `ActionPlanV1` or `ActionIntentV1`, no reflex execution, no adapter changes,
  no cleanup job, no consolidator-as-action-executor.
- Evidence threshold: add any rejected capability only through a reviewed
  follow-up plan that names owner, handler, continuation policy, latency budget,
  tests, and permission gates.

## Implementation Order

### Stage 0: Contracts Reference Reconciliation

- Verify `development_plans/reference/designs/cognition_contracts_design.md`
  is approved and linked from the registry.
- Verify the reference still resolves the prior approval blockers:
  `reflection_promoted` is not a committed trigger source, `system_event` is
  reserved rather than committed, and `ActionSourceRefV1`, `ActionTargetV1`,
  `ActionContinuationV1`, and `CapabilitySpecV1` are defined.
- Verify this plan and
  `development_plans/reference/designs/action_spec_effector_expansion_architecture.md`
  remain reconciled against contracts 1, 2, 3, 4, 6, and 7.
- Verify this plan's registry row still marks it approved before implementation
  starts.

Exit criteria:

- The contracts reference approval is verified.
- The blocker-resolution check above is recorded with grep output or explicit
  section references.
- Reconciliation notes are present in `Independent Plan Review` or
  `Execution Evidence`.

### Stage 1: Tests For Action Contracts

- Add tests for `ActionSpecV1` schema validation.
- Add tests for supporting shapes: `ActionSourceRefV1`, `ActionTargetV1`,
  `ActionContinuationV1`, and action-category `CapabilitySpecV1`.
- Add tests for `ActionResultV1`, `SurfaceOutputV1`, `EpisodeTraceV1`, and
  `ConsolidationActionProjectionV1`.
- Add tests for capability registry projection, including exact
  `memory_lifecycle_update` params schema and lifecycle vocabulary.
- Add anti-leakage assertions for `send_message` and
  `memory_lifecycle_update` prompt-safe projection: no `handler_id`, no raw
  collection name, no credentials, no adapter IDs, no raw channel IDs, and no
  database internals.
- Add tests that `reflex` is rejected for all current capabilities.
- Add tests for continuation policy validation.
- Add tests for unsupported `EvolvingMemoryDoc` lifecycle targets.

Expected pre-implementation result:

- Focused tests fail with missing module or missing symbol errors.

### Stage 2: Implement `action_spec` Module

- Implement `models.py`.
- Implement `registry.py`.
- Implement `evaluator.py`.
- Implement `attempt_ledger.py`.
- Export public symbols from `__init__.py`.
- Back the ledger with `self_cognition_action_attempts`.

Exit criteria:

- Stage 1 tests pass.
- Old `self_cognition_action_attempts` rows remain readable.

### Stage 3: L2d Action Initializer, L3 Handler Routing, And Offline Measurement

- Add optional `action_specs: list[ActionSpecV1]` state fields.
- Add prompt-safe `TriggerContextV1` projection for L2d.
- Add `persona_supervisor2_cognition_l2d.py` with one L2d action-initializer
  LLM node after L2c and before any L3 surface handler.
- Extend cognition output contract validation so only L2d may emit action specs.
  L2a and L2c must reject or ignore action-spec output.
- Update L2d guidance for deliberative action-spec decisions using the exact
  `Lifecycle Decision Vocabulary` in this plan.
- Add a dedicated frozen-upstream routing test gate before graph wiring:
  - QQ source: collect 20 user-message windows from existing QQ
    conversation history for user/channel `673225019`, run the existing
    L1/L2a/L2b/L2c stack once per case, and persist only the post-L2c
    `frozen_l2d_state` plus historical assistant-response comparison metadata.
  - Self-cognition source: collect 20 real self-cognition source cases, run
    the same upstream cognition stack once per case through the dry-run source
    path, and persist the post-L2c `frozen_l2d_state` plus route/action
    comparison metadata.
  - Store captured case sets under `test_artifacts/l2d_action_initializer/`;
    do not commit private captured conversation text.
  - Run the live L2d routing test one case at a time using
    `L2D_LIVE_CASE_FILE` and `L2D_LIVE_CASE_ID`.
  - Compare action route shape against the historical behavior: ordinary
    visible replies map to `speak`, ordinary chat must not emit
    `send_message`, self-cognition no-contact cases must not emit
    user-visible `speak`, and commitment retirement is expressed only through
    semantically justified `memory_lifecycle_update`.
  - Inspect each durable trace before running the next real LLM case.
- Update the graph so L3 text and L3 image are treated as registered
  action-spec handlers for surface actions. They must run only when a selected
  surface action requires them.
- Preserve existing `action_directives` and visible reply compatibility for the
  text surface during this plan.
- Add deterministic routing tests proving that no `speak` action means the text
  surface is not required.
- Add multi-action tests proving that L2d can emit zero, one, or multiple
  action specs up to the cap.
- Add deterministic tests for the frozen L2d routing case contract and
  comparison rules.
- Add offline fixture cases.
- Add `src/scripts/measure_action_spec_validity.py`.
- Run the offline measurement.

Exit criteria:

- Existing cognition schema tests pass.
- L2a/L2c direct action-spec emission is rejected or absent.
- The frozen-upstream L2d routing gate has 20 inspected QQ cases and 20
  inspected self-cognition cases, each run one at a time with durable traces.
- The routing gate shows no systematic misuse of `send_message` for ordinary
  dialog, no leakage of handler IDs or database internals, and acceptable
  handling of zero, one, and multiple action specs.
- Measurement records JSON validity, action-spec validity, false-positive action
  emission, invalid continuation requests, multi-action behavior, prompt size
  impact, and latency impact.
- Broad live `/chat` emission remains disabled until the measurement is reviewed.

### Stage 4: Action Results And Episode-Trace Consolidation

- Implement `ActionResultV1`, `SurfaceOutputV1`, `EpisodeTraceV1`, and
  `ConsolidationActionProjectionV1` helpers in the `action_spec` boundary.
- Ensure L3 text output is represented as `SurfaceOutputV1(surface_kind="text")`
  before any dispatcher bridge is invoked.
- Ensure private actions, rejected actions, scheduled actions, and no-reply
  decisions can produce traceable action results or private surface outputs.
- Update graph/service state plumbing so selected action specs, evaluation
  outcomes, action attempts, action results, and surface outputs can be attached
  to the episode trace.
- Update consolidator input construction to include prompt-safe episode-trace
  projections alongside existing text context.
- Update post-turn gating so consolidation can run when there is `final_dialog`,
  `surface_outputs`, `action_results`, or private finalization.
- Add tests proving the consolidator sees action outcomes for:
  private `memory_lifecycle_update`, future cognition scheduling, visible text
  surface, no-visible-reply/private finalization, and parallel independent
  actions.
- Add anti-leakage tests proving consolidator projections omit handler IDs, raw
  adapter IDs, credentials, raw collection names, raw channel IDs, and arbitrary
  action params.
- Preserve current text-only consolidation behavior for ordinary dialog while
  adding trace-aware inputs.

Exit criteria:

- Episode-trace/result tests pass.
- Consolidator action-trace tests pass.
- Text-only dialog still consolidates as before.
- Private and scheduled action outcomes can be consolidated without fabricating a
  user-visible final dialog.
- Consolidator code does not execute actions, call dispatcher, call scheduler,
  or trigger cognition.

### Stage 5: `send_message` Compatibility Bridge

- Route legacy self-cognition-triggered delivery candidates through
  `ActionSpecV1` validation as a compatibility bridge.
- Preserve `send_message` as the dispatcher-facing delivery capability. Do not
  use it as L2d's generic text-surface choice; L2d selects `speak` when a text
  surface is needed, and the text handler/dialog path produces deliverable text.
- Convert validated `send_message` action specs to existing `RawToolCall`.
- Dispatch through existing `TaskDispatcher.dispatch`.
- Preserve delivery mentions, permission checks, scheduler validation, adapter
  availability, and duplicate suppression.
- Persist attempts through the generic action-attempt repository.
- Add an automated end-to-end self-cognition-triggered delivery fixture covering:
  self-cognition trigger source case -> `ActionSpecV1(kind="send_message")`
  validation -> `RawToolCall(tool="send_message")` bridge ->
  `TaskDispatcher.dispatch`, with unchanged scheduled-event and duplicate
  suppression behavior.

Exit criteria:

- Existing dispatcher and self-cognition send-message tests pass.
- Existing action-attempt duplicate suppression remains effective.
- The self-cognition action-spec bridge fixture passes and proves no direct
  adapter call, no bypassed dispatcher validation, and no changed delivery
  mention metadata.

### Stage 6: Memory Lifecycle Capability

- Add `update_user_memory_unit_lifecycle`.
- Implement `handlers/memory_lifecycle.py`.
- Validate source refs and cognition-authored reason.
- Map semantic lifecycle statuses to collection-native statuses.
- Write lifecycle audit metadata to the action-attempt ledger and
  `merge_history`.
- Ensure active-promise retrieval excludes retired statuses through the existing
  active-status filter.
- Reject `EvolvingMemoryDoc` targets without persistence.

Exit criteria:

- A character-selected lifecycle action can mark a long-past-due active
  commitment as `completed`, `cancelled`, or `archived`.
- No deterministic age threshold hides or changes promises.
- Unsupported memory-evolution targets are rejected without writes.

### Stage 7: Documentation And Handoff

- Update every file in `Direct Documentation Updates`.
- Audit every file in `Conditional Documentation Audit`.
- Document the action-spec owner boundary.
- Document L2d as the only action initializer and L3 text/image as registered
  surface action handlers.
- Document `ActionResultV1`, `SurfaceOutputV1`, episode-trace consolidation,
  and final-dialog-as-text-surface behavior.
- Document self-cognition as a trigger source for the shared pipeline, not an
  action consumer.
- Document the consolidator boundary: consumes prompt-safe action/surface
  evidence; does not execute actions, dispatch, schedule, or trigger cognition.
- Document that dispatcher is an execution owner, not the parent action system.
- Document how `ActionSpecV1(kind="send_message")` bridges to `RawToolCall`.
- Document how `memory_lifecycle_update` changes
  `user_memory_units.active_commitment` status without age-based cleanup.
- Document compatibility expectations for old and new
  `self_cognition_action_attempts` rows.
- Add next-stage handoff notes for `schedule_self_check`, `web_research`, and
  notes/open-loop tools.
- Run the documentation staleness checks and inspect all matches.

Exit criteria:

- Direct documentation files describe the accepted action flow, owner
  boundaries, lifecycle behavior, and deferred tools.
- Conditional documentation files are either updated or explicitly recorded as
  not requiring a change.
- Static greps do not leave stale documentation claiming `send_message` is the
  only future effector without context.
- Execution evidence records changed doc paths, audited no-change doc paths,
  and grep output summaries.

## Progress Checklist

- [x] Stage 0 - contracts reference prerequisite complete
  - Verify: registry links approved `cognition_contracts_design.md`.
  - Evidence: record reference path and reconciliation notes.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 1 - action contract tests added
  - Verify: focused tests fail for missing symbols before implementation.
  - Evidence: record failing commands.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 2 - `action_spec` module implemented
  - Verify: action-spec model, evaluator, registry, and ledger tests pass.
  - Evidence: record changed files and test output.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 3 - L2d action initializer, L3 handler routing, and offline
  measurement gated
  - Verify: cognition schema/action-initializer tests pass and measurement
    output is reviewed.
  - Evidence: record validity, multi-action, false-positive, and latency
    results.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 4 - action results and episode-trace consolidation complete
  - Verify: result/trace tests and consolidator action-trace tests pass.
  - Evidence: record trace projection, consolidation gate, and anti-leakage
    results.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 5 - `send_message` bridge complete
  - Verify: dispatcher and self-cognition regression tests pass.
  - Evidence: record duplicate-suppression and old-row readability result.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 6 - memory lifecycle capability complete
  - Verify: memory lifecycle and user-memory retrieval tests pass.
  - Evidence: record status mapping and unsupported-target rejection.
  - Sign-off: `Codex/2026-05-16`
- [x] Stage 7 - documentation and handoff complete
  - Verify: direct documentation files are updated; conditional documentation
    audit is recorded; documentation static greps pass after inspection.
  - Evidence: record changed doc paths, no-change audited doc paths, and grep
    output summaries.
  - Sign-off: `Codex/2026-05-16`
- [ ] Independent code review complete
  - Verify: review findings are fixed or recorded as residual risks; affected
    tests are rerun.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, and
    approval status.
  - Sign-off: `<agent/date>`

## LLM And Latency Budget

- L2d is the only new cognition LLM call authorized by this plan.
- Do not add any additional live LLM calls beyond L2d for action initialization.
- Keep action specs optional in cognition output.
- Cap emitted action specs to 3 per cognition episode.
- Cap prompt-visible affordances to registered capabilities relevant to the
  episode source.
- Skip L3 text and L3 image handlers when no selected action spec requires that
  surface.
- L2d prompt context must include only final L2 state, prompt-safe trigger
  context, prompt-safe capability projection, and relevant evidence needed for
  current registered actions.
- Action and tool handlers return typed action results, surface outputs, and
  continuation data only. Orchestrator owns any continuation scheduling or later
  cognition episode creation.
- Episode-trace construction and consolidator projection must not add another
  live LLM call.
- Offline measurement must record prompt-size and latency impact before broad
  live `/chat` action-spec emission is enabled.
- Real LLM L2d routing verification must run one frozen case at a time. The
  operator must inspect the trace for that case before launching the next case.

## Verification

Run these checks after implementation:

```powershell
venv\Scripts\python -m pytest tests/test_action_spec_models.py
venv\Scripts\python -m pytest tests/test_action_spec_evaluator.py
venv\Scripts\python -m pytest tests/test_action_spec_results.py
venv\Scripts\python -m pytest tests/test_action_spec_attempt_ledger.py
venv\Scripts\python -m pytest tests/test_action_spec_memory_lifecycle.py
venv\Scripts\python -m pytest tests/test_action_spec_self_cognition_bridge.py
venv\Scripts\python -m pytest tests/test_consolidator_action_trace.py
venv\Scripts\python -m pytest tests/test_persona_supervisor2_action_initializer.py
venv\Scripts\python -m pytest tests/test_l2d_action_initializer_cases.py tests/test_l2d_action_initializer_live_llm.py -q
venv\Scripts\python src\scripts\capture_l2d_action_initializer_cases.py --help
venv\Scripts\python -m pytest tests/test_dispatcher.py tests/test_self_cognition_integration.py
venv\Scripts\python -m pytest tests/test_multi_source_cognition_stage_03_prompt_selection.py tests/test_persona_supervisor2_schema.py
venv\Scripts\python -m pytest tests/test_user_memory_units_rag_flow.py tests/test_db.py
venv\Scripts\python src\scripts\measure_action_spec_validity.py --fixture tests\fixtures\action_spec_cognition_cases.json
rg -n "only.*send_message|send_message.*only|currently send_message|Action candidates always use|dispatch_shape: \"send_message\"|future effector|one-effector" README.md docs src development_plans -g "*.md"
rg -n "ActionSpecV1|CapabilitySpecV1|memory_lifecycle_update|self_cognition_action_attempts|active_commitment|TaskDispatcher|RawToolCall" src\kazusa_ai_chatbot\action_spec\README.md src\kazusa_ai_chatbot\nodes\README.md src\kazusa_ai_chatbot\dispatcher\README.md src\kazusa_ai_chatbot\self_cognition\README.md src\kazusa_ai_chatbot\db\README.md
```

Targeted behavioral checks after deterministic tests pass:

- long-past-due promise, character fulfills;
- long-past-due promise, character abandons with reason;
- long-past-due promise, character defers without lifecycle action;
- unrelated live user message, no lifecycle action.
- one frozen QQ history case at a time:
  `L2D_LIVE_CASE_FILE=test_artifacts/l2d_action_initializer/qq_673225019_cases.json`
  and `L2D_LIVE_CASE_ID=qq_001`;
- one frozen self-cognition case at a time:
  `L2D_LIVE_CASE_FILE=test_artifacts/l2d_action_initializer/self_cognition_cases.json`
  and `L2D_LIVE_CASE_ID=self_001`.

## Independent Code Review

Run this gate after all verification commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan and inspect the full diff from
a fresh-review posture.

Review scope:

- Mandatory skill compliance.
- Plan alignment for `Must Do`, `Deferred`, `Change Surface`, and
  implementation stages.
- Code quality and ownership boundaries.
- Absence of hidden fallback paths, prompt leaks, direct adapter calls, direct
  cognition calls from handlers, generic MongoDB writes, or deterministic
  semantic promise cleanup.
- Absence of consolidator-owned action planning/execution, direct dispatcher
  calls, direct scheduler calls, or final-dialog-only consolidation gates.
- Verification evidence and next-stage handoff quality.

Fix findings only inside the approved change surface. If a finding requires new
scope, update this plan or request approval before changing code.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Action spec becomes a second dispatcher | Keep dispatcher as execution owner only for adapter-facing tools | Bridge tests and owner-boundary review |
| L2d becomes a bloated tool router | Keep L2d to semantic action selection from prompt-safe affordances; deterministic code owns routing and permissions | L2d prompt review and anti-leakage tests |
| L3 remains always-on despite action selection | Route L3 surface handlers only for selected surface action specs | Action-initializer routing tests |
| Action outcomes are invisible to consolidation | Build `EpisodeTraceV1` with action specs, attempts, results, and surface outputs before consolidation | Action-trace consolidator tests |
| Consolidator becomes an action executor | Limit consolidator to prompt-safe projections and persistence owners; forbid dispatcher/scheduler/cognition calls | Code review and boundary tests |
| Final-dialog-only gate drops private or scheduled actions | Use a consolidatable-output gate covering final dialog, surface outputs, action results, and private finalization | Service/post-turn tests |
| Multiple action specs execute in unsafe implicit chains | Allow same-cycle independent actions only; require continuation for dependent tool-result chains | Multi-action fixtures and continuation tests |
| Promise lifecycle becomes deterministic cleanup | Require validated character-selected action spec with source refs and reason | No-action and past-due fixtures |
| LLM processes target IDs or source refs | Resolve target binding deterministically and keep IDs out of the L2d payload/output contract | Prompt anti-leakage tests and lifecycle binding tests |
| Self-cognition commitment target is lost before L2d | Bind lifecycle targets from trusted self-cognition source refs as well as RAG active commitments | Self-cognition source-ref lifecycle materialization test |
| Ledger compatibility breaks duplicate suppression | Back action attempts with `self_cognition_action_attempts` and read old rows tolerantly | Ledger and self-cognition regression tests |
| Live cognition schema increases invalid output | Keep action specs optional and run offline measurement first | Measurement script and cognition tests |
| Memory writes bypass repository owner | Add narrow lifecycle helper and reject generic writes | Memory lifecycle tests |
| Documentation keeps stale single-effector or dispatcher-parent claims | Treat direct documentation updates and stale-string greps as Stage 7 exit criteria | Documentation change surface evidence |
| Prompt-safe affordance projection leaks runtime internals | Assert projection excludes handler IDs, raw IDs, credentials, adapter IDs, and DB internals | Capability projection anti-leakage tests |

## Acceptance Criteria

- `cognition_contracts_design.md` is approved and reconciliation is recorded.
- `ActionSpecV1` and `ActionSpecEvaluator` exist in `action_spec`.
- Supporting contracts `ActionSourceRefV1`, `ActionTargetV1`,
  `ActionContinuationV1`, and action-category `CapabilitySpecV1` are
  implemented and tested.
- Result/trace contracts `ActionResultV1`, `SurfaceOutputV1`,
  `EpisodeTraceV1`, and `ConsolidationActionProjectionV1` are implemented and
  tested.
- L2d is the only cognition stage that emits `action_specs`.
- L2d consumes prompt-safe trigger context and prompt-safe capability
  projection.
- `action_specs` supports zero, one, or multiple records with a cap of 3.
- L3 text and L3 image are documented and/or wired as registered surface action
  handlers rather than always-on action-independent branches.
- Self-cognition is documented and tested as a trigger source for the shared
  L1/L2/L2d/action/consolidation pipeline, not as a downstream action consumer.
- Consolidation consumes prompt-safe action/surface evidence through an episode
  trace and is not gated only by final dialog.
- Consolidator code does not plan actions, execute actions, call dispatcher,
  call scheduler, or trigger cognition.
- `reflex` is represented in schema but rejected for all current capabilities.
- `memory_lifecycle_update` capability params and lifecycle vocabulary match
  this plan exactly.
- `memory_lifecycle_update` target binding is deterministic: L2d never receives
  or emits target IDs, and materialization can bind the self-cognition
  active-commitment source-ref target without relying on LLM-copied IDs.
- Prompt-safe capability projection anti-leakage tests pass.
- `send_message` works through the action-spec bridge without behavior
  regression.
- Automated self-cognition-triggered delivery bridge fixture proves
  self-cognition typed episode -> action spec -> dispatcher behavior.
- `memory_lifecycle_update` can mark active commitments completed, cancelled,
  or archived only after a validated character-selected action.
- No deterministic age threshold removes or hides active promises.
- Action attempts use the existing `self_cognition_action_attempts` backing
  collection.
- `EvolvingMemoryDoc` lifecycle targets are rejected without persistence.
- Offline structured-emission measurement is reviewed before broad live `/chat`
  action-spec emission.
- Direct documentation files from `Documentation Change Surface Measurement`
  are updated.
- Conditional documentation audit is recorded with either changes or explicit
  no-change evidence.
- Static documentation grep matches are inspected and stale claims are fixed.
- Next-stage handoff notes are updated.

## Independent Plan Review

Run this gate before approval, execution, or handoff.

Review scope:

- Architecture doc and this execution plan are separated cleanly.
- This plan contains execution instructions only, not design research or
  open-ended brainstorming.
- The contracts-reference prerequisite is satisfied, and this plan status is
  approved before execution starts.
- Change surface, tests, verification, progress checklist, and acceptance
  criteria are specific enough for implementation.
- Deferred tools are not executable under this plan.

Latest review result:

- 2026-05-15: contracts design draft created after prior approval review.
  - Status change: `cognition_contracts_design.md` now exists as a draft
    reference.
  - Remaining blocker: the contracts design is not yet approved and this plan
    has not been reconciled against it.
  - Decision: this plan remains `draft` and must not be executed.
- 2026-05-15: independent plan review requested for approval.
  - Inputs reviewed: development plan registry, active execution plan, action
    spec architecture reference, and presence check for
    `development_plans/reference/designs/cognition_contracts_design.md`.
  - Blocker at review time:
    `development_plans/reference/designs/cognition_contracts_design.md` was
    absent. The execution plan's summary, mandatory rules, cutover policy,
    implementation order, and acceptance criteria required this reference before
    approval.
  - Decision: approval denied. The plan remains `draft` and must not be
    executed.
  - Current follow-up after contracts draft creation: approve
    `cognition_contracts_design.md`, then record reconciliation against this
    plan and the architecture reference.
- 2026-05-15: active-agent fresh review completed after architecture/execution
  split; no separate reviewer was available in this session.
  - Historical blocker at review time:
    `development_plans/reference/designs/cognition_contracts_design.md` did not
    exist, so this plan stayed draft.
  - Non-blocking finding: architecture research, decision justification, and
    future tool brainstorming now live in
    `development_plans/reference/designs/action_spec_effector_expansion_architecture.md`.
  - Non-blocking finding: the active plan now contains execution scope, change
    surface, stages, progress checklist, verification, review gates, and
    acceptance criteria without open-ended design alternatives.
  - Approval status: not approved for execution; ready for
    `cognition_contracts_design.md` prerequisite work.
- 2026-05-16: contracts-reference prerequisite resolved.
  - Inputs reviewed: approved
    `development_plans/reference/designs/cognition_contracts_design.md`,
    action-spec architecture reference, and this execution plan.
  - Reconciliation recorded: `ActionSpecV1` references now use
    `ActionSourceRefV1`, `ActionTargetV1`, and `ActionContinuationV1`;
    evaluator capability references now use action-category `CapabilitySpecV1`
    entries.
  - Remaining status: this plan is still `draft` and is not approved for
    execution until a final independent plan review explicitly changes its
    status.
- 2026-05-16: documentation change surface measured and added to this plan.
  - Inputs reviewed: current change surface, subsystem README list, nodes,
    dispatcher, self-cognition, DB, event logging, proactive-output, and
    brain-service documentation.
  - Direct documentation updates now required for `action_spec`, cognition
    nodes, dispatcher, self-cognition, and DB docs.
  - Conditional documentation audit now required for event logging, proactive
    output, brain service, memory evolution, HOWTO, and root README when their
    boundaries are touched.
  - Documentation stage now requires static staleness greps and recorded
    evidence.
- 2026-05-16: execution-plan review blockers addressed.
  - Inputs reviewed: approved contracts reference, this plan's
    `Contracts To Implement`, implementation stages, verification, and
    documentation gates.
  - Contracts dependency check: current contracts reference is approved,
    removes `reflection_promoted` from committed triggers, reserves
    `system_event`, and defines `ActionSourceRefV1`, `ActionTargetV1`,
    `ActionContinuationV1`, `MemoryQueryV1`, `MemoryResultV1`,
    `MemoryLifecycleUpdateV1`, and `CapabilitySpecV1`.
  - Plan update: supporting TypedDicts, capability params, lifecycle decision
    vocabulary, prompt-safe projection anti-leakage assertions, L3
    surface-handler boundary, self-cognition bridge fixture, CJK-safety
    applicability, and
    `tests/fixtures/` justification are now explicit.
  - Remaining status: this plan is still `draft` until final independent plan
    review approves execution.
- 2026-05-16: final independent plan review completed for approval.
  - Inputs reviewed: plan-writing contract, execution gates, cutover policy,
    development plan registry, approved contracts reference, action-spec
    architecture reference, and this execution plan.
  - Approval blockers: none remaining. Prior contracts-reference blockers are
    resolved in the approved reference and verified in this plan's Stage 0.
  - Non-blocking finding resolved before approval: added the required
    `Design Decisions` section so the executable plan satisfies final-plan
    contract structure.
  - Decision: approved for execution. Execute only through the stage order,
    verification commands, documentation gates, and independent code review
    gate in this plan.
- 2026-05-16: Stage 3 architecture reset after L2/L3 responsibility review.
  - Inputs reviewed: user review on L2a overload risk, requested inventory of
    L2a/L2b/L2c emitted information, L2d feasibility, L3-as-action question,
    trigger-source dependency, and missing multi-action demand.
  - Rejected design: direct `ActionSpecV1` emission from L2a, L2c, or generic
    L2 schema expansion. This overloaded upstream cognition and forced surface
    behavior outside the action contract.
  - Updated decision: add L2d after L2c as the only action initializer; keep
    `ActionSpecV1` as the only action-selection residue; support
    `action_specs: list[ActionSpecV1]` with cap 3; let L2d consume
    prompt-safe trigger context and capability projection; treat L3 text/image
    as registered surface action handlers.
  - Execution impact: Stage 3 must be restarted from this revised design before
    any cognition-module edits resume. External image generation remains
    deferred, and dependent action chains must use continuation or a subsequent
    validated action spec.
- 2026-05-16: consolidation and self-cognition boundary review completed.
  - Inputs reviewed: user clarification that self-cognition is a trigger for
    L1/L2 rather than an action consumer, and concern that action or parallel
    action outcomes must feed the consolidator.
  - Updated decision: self-cognition remains an initial trigger source for the
    same typed-episode path; L2d/action/L3/consolidation are downstream shared
    concerns, not self-cognition-owned branches.
  - Updated decision: add action result, surface output, episode trace, and
    consolidator projection contracts before dispatcher bridge and lifecycle
    work proceed.
  - Updated decision: final dialog is one text surface output. Consolidation must
    consume prompt-safe episode-trace evidence and must not remain gated only by
    final dialog.
  - Execution impact: inserted Stage 4 for action results and episode-trace
    consolidation, shifted `send_message` bridge to Stage 5, memory lifecycle to
    Stage 6, and documentation/handoff to Stage 7.

## Execution Evidence

Implementation is in progress. Execute only through the approved stage order and
record verification evidence here.

- 2026-05-16 planning revision - self-cognition and consolidation boundary
  recorded.
  - Updated
    `development_plans/reference/designs/cognition_contracts_design.md` with
    self-cognition-as-trigger, L3-as-selected-surface, action results, surface
    outputs, episode trace, continuation, and consolidation boundaries.
  - Updated this execution plan to insert Stage 4 for action results and
    episode-trace consolidation before dispatcher bridge and memory lifecycle
    work continue.
  - Updated root `README.md` architecture diagram as part of the required high
    level documentation change.
- 2026-05-16 Stage 0 - contracts reference reconciliation complete.
  - Branch: `feature/action-spec-effector-expansion`.
  - Planning baseline commit: `f8fde7a` (`Approve action spec effector
    expansion plan`).
  - Registry check: `development_plans/README.md` lists
    `modality_neutral_action_spec_effector_expansion_plan.md` as `approved`
    and `cognition_contracts_design.md` as the authoritative contract
    reference.
  - Contracts check: `cognition_contracts_design.md` is `Status: approved
    reference`; `reflection_promoted` is not a committed trigger source;
    `system_event` is reserved rather than committed; `ActionSourceRefV1`,
    `ActionTargetV1`, `ActionContinuationV1`, `MemoryQueryV1`,
    `MemoryResultV1`, `MemoryLifecycleUpdateV1`, and `CapabilitySpecV1` are
    defined.
  - Reconciliation check: action-spec architecture and execution plan both
    reference `ActionSpecV1`, action-category `CapabilitySpecV1`,
    `memory_lifecycle_update`, action/surface trace contracts, and contracts 1,
    2, 3, 4, 6, and 7 consistently.
- 2026-05-16 Stage 1 - action contract RED tests added.
  - Added focused contract tests:
    `tests/test_action_spec_models.py`,
    `tests/test_action_spec_evaluator.py`,
    `tests/test_action_spec_attempt_ledger.py`,
    `tests/test_action_spec_memory_lifecycle.py`, and
    `tests/test_action_spec_self_cognition_bridge.py`.
  - RED command:
    `venv\Scripts\python -m pytest tests/test_action_spec_models.py tests/test_action_spec_evaluator.py tests/test_action_spec_attempt_ledger.py tests/test_action_spec_memory_lifecycle.py tests/test_action_spec_self_cognition_bridge.py -q`.
  - Expected result recorded: all five focused files fail at collection with
    `ModuleNotFoundError: No module named 'kazusa_ai_chatbot.action_spec'`.
- 2026-05-16 Stage 2 - `action_spec` module implemented.
  - Added public action-spec module files:
    `src/kazusa_ai_chatbot/action_spec/models.py`,
    `src/kazusa_ai_chatbot/action_spec/registry.py`,
    `src/kazusa_ai_chatbot/action_spec/evaluator.py`,
    `src/kazusa_ai_chatbot/action_spec/attempt_ledger.py`,
    `src/kazusa_ai_chatbot/action_spec/handlers/memory_lifecycle.py`, and
    `src/kazusa_ai_chatbot/action_spec/__init__.py`.
  - Updated `SelfCognitionActionAttemptDoc` with additive generic
    action-attempt metadata fields.
  - Verification command:
    `venv\Scripts\python -m pytest tests/test_action_spec_models.py tests/test_action_spec_evaluator.py tests/test_action_spec_attempt_ledger.py tests/test_action_spec_memory_lifecycle.py tests/test_action_spec_self_cognition_bridge.py -q`.
  - Result: 20 passed in 1.48s; old `self_cognition_action_attempts` rows
    remain readable through compatibility normalization.
- 2026-05-16 Stage 3 design reset and interrupted prototype rollback.
  - User review identified that direct L2a/L2c action-spec emission was an
    architectural mismatch and that the plan lacked explicit multi-action and
    L3-as-action handling.
  - Reverted interrupted cognition-module prototype edits before updating this
    plan. Restored cognition module and schema files, removed the temporary
    Stage 3 measurement script, and removed temporary action-spec cognition
    fixtures.
  - Working-tree scope after rollback: only this execution plan remains
    modified for the revised Stage 3 design.
  - Stage 3 remains unsigned and must not continue from the discarded
    prototype.
- 2026-05-16 Stage 3 partial - isolated L2d module implemented for deterministic
  verification.
  - Added `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
    as a separate L2d action initializer module.
  - Scope deliberately excludes live graph wiring, real LLM verification,
    measurement-script execution, dispatcher bridge changes, memory lifecycle
    execution changes, and external image generation.
  - Added `speak` as an L3-text-owned surface action capability so L2d can
    select a text surface without misusing `send_message` as ordinary dialog.
  - Added prompt-safe trigger-context projection, action-spec normalization,
    invalid-row dropping, and the cap of 3 valid action specs.
  - Revised the L2d prompt to follow the established cognition-prompt
    structure used by `_COGNITION_SUBCONSCIOUS_PROMPT`: role identity, language
    policy, bounded task rules, thinking path, and exact JSON input/output.
  - Verification commands:
    `venv\Scripts\python -m pytest tests/test_persona_supervisor2_action_initializer.py tests/test_multi_source_cognition_stage_03_prompt_selection.py tests/test_action_spec_models.py tests/test_action_spec_evaluator.py tests/test_persona_supervisor2_schema.py -q`;
    `venv\Scripts\python -m pytest tests/test_action_spec_models.py tests/test_action_spec_evaluator.py tests/test_action_spec_attempt_ledger.py tests/test_action_spec_memory_lifecycle.py tests/test_action_spec_self_cognition_bridge.py -q`;
    `venv\Scripts\python -m pytest tests/test_cognition_clarification_consumers.py tests/test_cognition_interaction_style_context.py tests/test_multi_source_cognition_stage_03_prompt_selection.py tests/test_persona_supervisor2_schema.py tests/test_persona_supervisor2_action_initializer.py -q`.
  - Results: 68 passed, 22 passed, and 64 passed respectively.
  - Stage 3 remains unsigned. Next gate is one-case-at-a-time real LLM
    inspection of L2d before any live graph wiring or downstream implementation.
- 2026-05-16 Stage 3 partial - frozen-upstream L2d routing test gate added.
  - Added `tests/l2d_action_initializer_cases.py` as the local frozen-case
    contract and comparison helper for route-shape validation.
  - Added `tests/test_l2d_action_initializer_cases.py` for deterministic case
    loading and comparison checks.
  - Added `tests/test_l2d_action_initializer_live_llm.py` as a one-case
    real LLM routing check, selected by `L2D_LIVE_CASE_FILE` and
    `L2D_LIVE_CASE_ID`, with durable trace output through `tests.llm_trace`.
  - Added `src/scripts/capture_l2d_action_initializer_cases.py` to capture
    private frozen upstream case sets under `test_artifacts/l2d_action_initializer/`.
  - Updated this plan to require 20 inspected QQ history cases for
    `673225019` and 20 inspected self-cognition source cases before live graph
    wiring continues.
  - Verification commands:
    `venv\Scripts\python -m pytest tests/test_persona_supervisor2_action_initializer.py tests/test_l2d_action_initializer_cases.py tests/test_l2d_action_initializer_live_llm.py -q`;
    `venv\Scripts\python -m pytest tests/test_l2d_action_initializer_cases.py tests/test_l2d_action_initializer_live_llm.py -q`;
    `venv\Scripts\python -m py_compile tests/l2d_action_initializer_cases.py tests/test_l2d_action_initializer_cases.py tests/test_l2d_action_initializer_live_llm.py`;
    `venv\Scripts\python -m py_compile src/scripts/capture_l2d_action_initializer_cases.py`;
    `venv\Scripts\python src\scripts\capture_l2d_action_initializer_cases.py --help`.
  - Results: focused L2d deterministic tests passed with 9 passed and
    1 live-LLM test deselected by the default marker filter; routing-case-only
    command passed with 5 passed and 1 live-LLM test deselected; py_compile
    passed; capture command help printed successfully without DB or LLM access.
  - Stage 3 remains unsigned. Next step is implementing or running the
    capture command to populate `test_artifacts/l2d_action_initializer/`, then
    running one live L2d case at a time and inspecting each trace.
- 2026-05-16 Stage 3 partial - first QQ frozen-upstream live L2d case inspected.
  - Capture command:
    `venv\Scripts\python src\scripts\capture_l2d_action_initializer_cases.py --source qq_history --platform qq --platform-channel-id 673225019 --max-cases 1 --offset 0`.
  - First capture attempt reached MongoDB but failed before producing a
    fixture because the loaded personality lacked runtime `mood`; fixed the
    capture script to apply the same runtime character-profile defaults used by
    existing live cognition tests.
  - A subsequent capture attempt hit a transient local model-load error
    (`Failed to load model ... Operation canceled`); endpoint `/models`
    returned 200 and listed the configured model, so the same one-case capture
    was retried once.
  - Retry result: wrote
    `test_artifacts/l2d_action_initializer/qq_673225019_cases.json` with
    one private fixture, `case_id=qq_001`.
  - First live L2d run command:
    `L2D_LIVE_CASE_FILE=test_artifacts\l2d_action_initializer\qq_673225019_cases.json L2D_LIVE_CASE_ID=qq_001 venv\Scripts\python -m pytest tests/test_l2d_action_initializer_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -s -m live_llm`.
  - First live result: failed route validation. L2d selected
    `kind=speak`, `visibility=user_visible`, but emitted `target.owner=user`
    instead of `l3_text`.
  - Fix: tightened `_ACTION_INITIALIZER_PROMPT` with explicit target-owner
    rules for `speak`, `memory_lifecycle_update`, and `send_message`, and
    added a deterministic prompt-contract assertion for `speak` owner.
  - Post-fix deterministic command:
    `venv\Scripts\python -m pytest tests/test_persona_supervisor2_action_initializer.py tests/test_l2d_action_initializer_cases.py -q`.
  - Post-fix deterministic result: 9 passed.
  - Post-fix live result: `qq_001` passed. The inspected trace
    `test_artifacts/llm_traces/l2d_action_initializer_live_llm__qq_001__20260515T150429717132Z.json`
    contained one validated action:
    `kind=speak`, `visibility=user_visible`, `target_kind=current_channel`,
    `target_owner=l3_text`, `delivery_mode=visible_reply`, no continuation,
    no leakage errors.
  - Stage 3 remains unsigned. Only 1 of the required 20 QQ cases has been
    captured and inspected; no self-cognition frozen cases have been captured
    or inspected yet.
- 2026-05-16 Stage 3 partial - full QQ dialog frozen-upstream live L2d sweep
  completed.
  - Capture command for remaining QQ cases:
    `venv\Scripts\python src\scripts\capture_l2d_action_initializer_cases.py --source qq_history --platform qq --platform-channel-id 673225019 --max-cases 19 --offset 1`.
  - Capture result: private fixture
    `test_artifacts/l2d_action_initializer/qq_673225019_cases.json` contains
    20 QQ dialog cases from user `673225019`.
  - Live LLM execution policy: ran
    `tests/test_l2d_action_initializer_live_llm.py::test_l2d_live_case_against_frozen_upstream`
    one case at a time with `L2D_LIVE_CASE_ID=qq_001` through `qq_020`;
    inspected each emitted trace under `test_artifacts/llm_traces/`.
  - Observed route coverage: latest trace per QQ case is 20/20 passing;
    all 20 emitted exactly the dialog surface route
    `kind=speak`, `visibility=user_visible`, and no `send_message`.
  - Prompt defects found and fixed during the sweep:
    `qq_003` exposed that `speak.params.surface_requirements` must be an
    object rather than a string/list/null; `qq_010` exposed that every action
    needs a non-empty `source_refs` entry. Both were fixed in the L2d prompt
    and covered by deterministic prompt-contract tests.
- 2026-05-16 Stage 3 partial - self-cognition frozen-upstream live L2d sweep
  completed.
  - Capture command:
    `venv\Scripts\python src\scripts\capture_l2d_action_initializer_cases.py --source self_cognition --max-cases 20 --offset 0`.
  - Capture result: private fixture
    `test_artifacts/l2d_action_initializer/self_cognition_cases.json`
    contains 20 real self-cognition source cases.
  - Expectation calibration: legacy self-cognition route labels are retained
    as comparison metadata only. Hard route expectations are derived from the
    frozen L2 state because the legacy route disagreed with the post-L2c
    state in cases such as `self_013`, `self_014`, and `self_017`.
  - Live LLM execution policy: ran the same live L2d test one case at a time
    with `L2D_LIVE_CASE_ID=self_001` through `self_020`; inspected each trace.
  - Observed route coverage: latest trace per core self-cognition case is
    20/20 passing. Nineteen cases correctly emitted no user-visible action;
    `self_013` emitted one validated `speak` route and no `send_message`.
- 2026-05-16 Stage 3 partial - private memory lifecycle L2d route verified.
  - Added a dedicated private fixture `self_lifecycle_001` under
    `test_artifacts/l2d_action_initializer/self_cognition_cases.json`,
    derived from captured case `self_001`. The fixture preserves the real
    commitment `unit_id=bc1a81c563d74dcfb0fc70a4913b4c91` and
    `due_at=2026-05-06T12:00:00+00:00`, then changes only frozen L2 decision
    text to explicitly choose a private `abandoned` lifecycle action.
  - First live run selected the right action kind and decision but omitted
    required lifecycle params. Fix: tightened the L2d prompt so
    `memory_lifecycle_update.params` must include `memory_kind`, `unit_type`,
    `unit_id`, `lifecycle_decision`, and `due_at`, and the target scope must
    include `unit_type=active_commitment`.
  - Post-fix live result: `self_lifecycle_001` passed. Latest inspected trace
    emitted one private validated `memory_lifecycle_update` with
    `lifecycle_decision=abandoned`, copied `unit_id`, copied `due_at`, no
    `speak`, no `send_message`, and no prompt-safety leakage.
  - Aggregate latest-trace summary: 41/41 L2d live cases passed
    (20 QQ dialog, 20 core self-cognition, 1 lifecycle fixture). Observed
    action kinds: QQ `speak` x20; self-cognition `speak` x1,
    `memory_lifecycle_update` x1, and no-action x19.
  - Deterministic send-message routability remains covered outside L2d because
    ordinary dialog and self-cognition L2d inputs must not emit
    `send_message`. Verification command:
    `venv\Scripts\python -m pytest tests\test_action_spec_evaluator.py tests\test_action_spec_self_cognition_bridge.py tests\test_action_spec_memory_lifecycle.py tests\test_action_spec_attempt_ledger.py -q`.
    Result: 16 passed, including `send_message` action-spec to
    `RawToolCall` bridge, dispatcher evaluator acceptance, memory lifecycle
    repository payload, and attempt-ledger compatibility.
  - Additional verification commands:
    `venv\Scripts\python -m pytest tests\test_persona_supervisor2_action_initializer.py tests\test_l2d_action_initializer_cases.py -q`
    result 11 passed;
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py tests\l2d_action_initializer_cases.py tests\test_l2d_action_initializer_cases.py tests\test_l2d_action_initializer_live_llm.py`
    passed.
  - Stage 3 remains unsigned for live graph wiring, but the standalone L2d
    routing gate is now satisfied for `speak`, no-action,
    `memory_lifecycle_update`, and the deterministic `send_message` bridge.
- 2026-05-16 Stage 3 partial - final-prompt rerun completed after lifecycle
  prompt tightening.
  - Reran the 20 QQ dialog cases against the final L2d prompt, invoking the
    live pytest case separately for each `L2D_LIVE_CASE_ID=qq_001` through
    `qq_020`. Result: 20/20 passed, all latest traces route to `speak`.
  - Reran the 20 core self-cognition cases against the final L2d prompt,
    invoking the live pytest case separately for each
    `L2D_LIVE_CASE_ID=self_001` through `self_020`. Result: 20/20 passed;
    latest traces route to no-action x19 and `speak` x1.
  - The already-rerun lifecycle fixture `self_lifecycle_001` remains passing
    on the same final prompt with one private `memory_lifecycle_update`.
  - Latest trace aggregate after final rerun: 41/41 passing, no missing core
    cases, observed action kinds `speak` and `memory_lifecycle_update`, with
    no `send_message` misuse in dialog or self-cognition L2d outputs.
  - Final hygiene check: `git diff --check` reported only CRLF normalization
    warnings from Git on Windows and no whitespace errors.
- 2026-05-16 Stage 3 partial - capability registry boundary corrected.
  - User review identified that `speak` and `send_message` were too similar
    when both were exposed as initial L2d capabilities.
  - Updated implementation decision: `build_initial_action_capabilities()` now
    exposes only L2d-facing semantic capabilities: `speak`,
    `memory_lifecycle_update`, and `trigger_future_cognition`.
    `send_message` remains available only through a separate dispatcher bridge
    capability set used after final text exists.
  - Added `trigger_future_cognition` as an orchestrator-owned private action
    request with params `episode_type`, `trigger_at`, and `context_summary`.
    This is a contract for a later cognition cycle; it does not directly call
    cognition from the action handler.
  - Tightened the L2d prompt so every action must include a full
    `ActionContinuationV1` object even when no continuation is requested.
  - RED check before implementation:
    `venv\Scripts\python -m pytest tests\test_action_spec_evaluator.py tests\test_persona_supervisor2_action_initializer.py -q`
    failed at collection because
    `build_dispatcher_bridge_capabilities` did not exist.
  - Deterministic verification after implementation:
    `venv\Scripts\python -m pytest tests\test_action_spec_models.py tests\test_action_spec_evaluator.py tests\test_action_spec_attempt_ledger.py tests\test_action_spec_memory_lifecycle.py tests\test_action_spec_self_cognition_bridge.py tests\test_persona_supervisor2_action_initializer.py tests\test_l2d_action_initializer_cases.py -q`
    result 35 passed.
  - Syntax verification:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\action_spec\models.py src\kazusa_ai_chatbot\action_spec\registry.py src\kazusa_ai_chatbot\action_spec\evaluator.py src\kazusa_ai_chatbot\action_spec\__init__.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py tests\test_action_spec_evaluator.py tests\test_persona_supervisor2_action_initializer.py`
    passed.
- 2026-05-16 Stage 3 partial - L2d semantic request boundary corrected.
  - User review identified that asking L2d to emit full `ActionSpecV1`
    envelopes still crossed the semantic/mechanical boundary and overloaded
    the local LLM with deterministic fields.
  - Updated implementation decision: L2d prompt output is now
    `action_requests` with semantic fields only: `capability`, `decision`,
    `detail`, and `reason`. Deterministic materialization builds the
    graph-visible `ActionSpecV1` list with schema versions, target owners,
    source refs, params, and no-continuation defaults.
  - L2d prompt and prompt payload now exclude deterministic content such as
    schema-version fields, action-target envelopes, continuation envelopes,
    handler IDs, raw persistence IDs, and raw transport IDs. Capability
    projections expose semantic input summaries instead of parameter summaries.
  - Deterministic verification:
    `venv\Scripts\python -m pytest tests\test_persona_supervisor2_action_initializer.py tests\test_action_spec_evaluator.py -q`
    result 12 passed.
  - Adjacent regression verification:
    `venv\Scripts\python -m pytest tests\test_l2d_action_initializer_cases.py tests\test_action_spec_models.py tests\test_persona_supervisor2_schema.py tests\test_multi_source_cognition_stage_03_prompt_selection.py -q`
    result 65 passed.
  - Syntax verification:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\kazusa_ai_chatbot\action_spec\registry.py`
    passed.
  - Real LLM smoke verification after this boundary change, one case at a
    time:
    `qq_001` passed with `speak`;
    `self_lifecycle_001` passed with private `memory_lifecycle_update`;
    `self_future_001` passed with private `trigger_future_cognition`.
  - Full 20x QQ plus 20x self-cognition rerun is still required before
    approving further graph wiring.
- 2026-05-16 Stage 3 partial - code alignment cleanup after contracts update.
  - Reviewed the branch against the updated contract decisions:
    self-cognition is a trigger source, L2d emits only semantic
    `action_requests`, L3 text/image are selected action handlers, and
    `send_message` remains bridge-only.
  - Removed stale L2d prompt trigger-source vocabulary
    (`reflection_signal`, `scheduled_recall`, `system_probe`) and replaced it
    with the committed contract vocabulary: `user_message`,
    `internal_thought`, `self_cognition`, `scheduled_tick`, and `tool_result`.
    `internal_thought` remains mentioned only as a current compatibility label
    for self-cognition cases that have not yet migrated trigger-source naming.
  - Removed the frozen self-cognition capture script's deterministic
    `logical_stance` / `character_intent` expectation heuristic. Self-cognition
    routing expectations now compare against historical route artifacts and
    action-candidate artifacts instead of treating `CONFIRM` / `PROVIDE` as an
    automatic user-visible `speak` decision.
  - Kept the existing bridge and lifecycle code as non-L2d-facing contract
    slices because they remain explicitly present in this approved plan. No
    graph wiring, scheduler execution, consolidation execution, or live
    external handlers were added.
  - Verification:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\scripts\capture_l2d_action_initializer_cases.py`
    passed.
  - Verification:
    `venv\Scripts\python -m pytest tests\test_persona_supervisor2_action_initializer.py tests\test_l2d_action_initializer_cases.py tests\test_action_spec_evaluator.py tests\test_action_spec_models.py -q`
    result 25 passed.
  - Regression verification:
    `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_persona_supervisor2_schema.py tests\test_action_spec_attempt_ledger.py tests\test_action_spec_memory_lifecycle.py tests\test_action_spec_self_cognition_bridge.py -q`
    result 62 passed.
- 2026-05-16 Stage 3-6 checkpoint complete.
  - Stage 3 implementation: wired isolated L2d into the cognition graph after
    L2c and before L3; L2d remains the only stage that emits action specs.
    Persona routing now invokes L3 text only when `speak` is selected and uses
    the no-visible-output path when no text surface is selected.
  - Stage 4 implementation: added `ActionResultV1`, `SurfaceOutputV1`,
    `EpisodeTraceV1`, and prompt-safe consolidation projection helpers. Persona
    graph attaches action results, surface outputs, and episode trace to
    consolidation state. Service consolidation gating now uses consolidatable
    outputs instead of `final_dialog` alone.
  - Stage 4 consolidator boundary: facts harvester receives
    `episode_trace_projection` only; tests assert no handler IDs, raw params,
    raw DB collection names, credentials, or adapter internals are projected.
  - Stage 5 implementation: legacy self-cognition `send_message` candidates
    now build a `send_message` `ActionSpecV1` and cross the existing dispatcher
    only through `build_raw_tool_call_from_action_spec`. The old direct
    `build_raw_tool_call` helper remains for compatibility, but production
    `dispatch_action_candidate` uses the bridge.
  - Stage 6 implementation: added narrow
    `update_user_memory_unit_lifecycle` repository helper and
    `execute_user_memory_lifecycle_action`. Lifecycle execution validates
    target owner, matching memory-unit source refs, cognition-authored reason,
    and rejects `EvolvingMemoryDoc` targets.
  - First broad checkpoint run exposed two stale L2d routing-fixture tests
    missing the required memory-unit source ref. Fixed the fixture action spec
    to include `ref_kind=memory_unit`, then reran the checkpoint batch.
  - Syntax verification:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_schema.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py src\kazusa_ai_chatbot\self_cognition\handoff.py src\kazusa_ai_chatbot\self_cognition\__init__.py src\kazusa_ai_chatbot\action_spec\handlers\memory_lifecycle.py src\kazusa_ai_chatbot\db\user_memory_units.py tests\test_persona_supervisor2.py tests\test_service_background_consolidation.py tests\test_consolidator_facts_rag2.py tests\test_action_spec_self_cognition_bridge.py tests\test_action_spec_memory_lifecycle.py tests\test_user_memory_unit_lifecycle.py`
    passed.
  - Focused Stage 3-6 verification:
    `venv\Scripts\python -m pytest tests\test_action_spec_results.py tests\test_persona_supervisor2.py tests\test_service_background_consolidation.py tests\test_consolidator_facts_rag2.py tests\test_action_spec_self_cognition_bridge.py tests\test_action_spec_memory_lifecycle.py tests\test_user_memory_unit_lifecycle.py -q`
    result 51 passed.
  - Plan-level checkpoint verification:
    `venv\Scripts\python -m pytest tests\test_action_spec_models.py tests\test_action_spec_evaluator.py tests\test_action_spec_attempt_ledger.py tests\test_action_spec_memory_lifecycle.py tests\test_action_spec_self_cognition_bridge.py tests\test_action_spec_results.py tests\test_persona_supervisor2_action_initializer.py tests\test_l2d_action_initializer_cases.py tests\test_l2d_action_initializer_live_llm.py tests\test_dispatcher.py tests\test_self_cognition_integration.py tests\test_user_memory_units_rag_flow.py tests\test_user_memory_unit_lifecycle.py tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_persona_supervisor2_schema.py -q`
    result 135 passed, 5 live tests deselected by the default marker filter.
  - Hygiene verification: `git diff --check` exited 0 with Windows CRLF
    normalization warnings only.
- 2026-05-16 Stage 7 - documentation and handoff complete.
  - Direct documentation updates completed:
    `src/kazusa_ai_chatbot/action_spec/README.md`,
    `src/kazusa_ai_chatbot/nodes/README.md`,
    `src/kazusa_ai_chatbot/dispatcher/README.md`,
    `src/kazusa_ai_chatbot/self_cognition/README.md`,
    `src/kazusa_ai_chatbot/db/README.md`,
    `src/kazusa_ai_chatbot/brain_service/README.md`, and root `README.md`.
  - Additional stale-context references updated:
    `development_plans/reference/designs/cognition_core_evolution_progression.md`
    and
    `development_plans/reference/designs/action_spec_effector_expansion_architecture.md`
    now describe the prior one-effector state as historical and distinguish
    bridge-only dispatcher delivery from private action-spec capabilities.
  - Documented owner boundaries:
    `action_spec` owns action/result/trace contracts and validation;
    L2d is the only action initializer; L3 text/image are registered surface
    handlers; dispatcher remains adapter-facing execution owner;
    self-cognition is an upstream trigger source; the consolidator consumes
    prompt-safe episode-trace evidence and does not execute, dispatch,
    schedule, or trigger cognition.
  - Documented runtime behavior for `ActionSpecV1`,
    `ActionResultV1`, `SurfaceOutputV1`, `EpisodeTraceV1`,
    `memory_lifecycle_update`, bridge-only `send_message`,
    `self_cognition_action_attempts` compatibility, final-dialog-as-text
    surface, and no-visible-action consolidation.
  - Next-stage handoff notes are documented for `schedule_self_check`,
    `web_research`, notes/open-loop tools, and future image surfaces in
    `src/kazusa_ai_chatbot/action_spec/README.md`.
  - Conditional documentation audit:
    `src/kazusa_ai_chatbot/event_logging/README.md` no change required because
    no new event family, recorder, payload field, or validation telemetry was
    added; `src/kazusa_ai_chatbot/proactive_output/README.md` no change
    required because proactive permission/outbox/transport boundaries were not
    changed; `src/kazusa_ai_chatbot/memory_evolution/README.md` no change
    required because `EvolvingMemoryDoc` lifecycle mutation remains unsupported;
    `docs/HOWTO.md` no change required because no operator-facing command,
    config flag, API behavior, or runbook step was added.
  - Documentation stale-string command:
    `rg -n 'only.*send_message|send_message.*only|currently send_message|Action candidates always use|dispatch_shape: "send_message"|future effector|one-effector' README.md docs src development_plans -g '*.md'`.
    Inspected matches: live docs now describe bridge-only `send_message`,
    legacy dry-run delivery candidate artifacts, or proactive-output grep
    expectations; remaining archive matches are historical plan records.
  - Documentation coverage command:
    `rg -n "ActionSpecV1|ActionResultV1|SurfaceOutputV1|EpisodeTraceV1|CapabilitySpecV1|memory_lifecycle_update|self_cognition_action_attempts|active_commitment|TaskDispatcher|RawToolCall|consolidation" src\kazusa_ai_chatbot\action_spec\README.md src\kazusa_ai_chatbot\nodes\README.md src\kazusa_ai_chatbot\dispatcher\README.md src\kazusa_ai_chatbot\self_cognition\README.md src\kazusa_ai_chatbot\db\README.md src\kazusa_ai_chatbot\brain_service\README.md README.md`.
    Result: direct documentation files contain the required terms and describe
    the accepted action flow, lifecycle behavior, bridge path, and
    consolidation boundary.
  - Hygiene verification:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py`
    passed after removing one staged trailing-space issue in the CJK prompt
    example; `git diff --check` and `git diff --cached --check` both exited 0,
    with only Windows CRLF normalization warnings in the unstaged diff check.
- 2026-05-16 post-Stage 7 blocker - L2d to L3 handoff incomplete.
  - User review identified that the current implementation still runs the old
    L3 directive chain unconditionally after L2d inside
    `persona_supervisor2_cognition.py`; the existing route only gates final
    dialog after L3 has already executed.
  - Deterministic inspection showed `speak` gates the top-level text/dialog
    path, but `persona_supervisor2_cognition_l3.py` does not consume
    `action_specs`, and L3 visual remains an always-run directive branch unless
    skipped by existing debug/config controls.
  - Draft follow-up plan created:
    `development_plans/active/short_term/l2d_l3_surface_handoff_plan.md`.
    This plan blocks treating the action-spec expansion as architecturally
    complete until L3 text/dialog becomes a selected `speak` handler and
    `trigger_future_cognition` schedules a future cognition slot.
  - Stage 7 documentation/handoff and independent code review remain pending.
- 2026-05-16 Stage 3-6 live LLM demonstration before Stage 7.
  - User requested real LLM evidence before continuing. No Stage 7 work was
    started.
  - Live LLM execution policy: ran
    `tests\test_l2d_action_initializer_live_llm.py::test_l2d_live_case_against_frozen_upstream`
    one case at a time with `-m live_llm`, inspected each emitted trace, then
    proceeded to the next case.
  - `qq_001`: command used
    `L2D_LIVE_CASE_FILE=test_artifacts/l2d_action_initializer/qq_673225019_cases.json`
    and `L2D_LIVE_CASE_ID=qq_001`. Result: passed. Trace:
    `test_artifacts/llm_traces/l2d_action_initializer_live_llm__qq_001__20260516T010536992208Z.json`.
    Observed action: `speak`, `user_visible`, evaluator owner `l3_text`, no
    leakage errors.
  - `self_001`: command used
    `L2D_LIVE_CASE_FILE=test_artifacts/l2d_action_initializer/self_cognition_cases.json`
    and `L2D_LIVE_CASE_ID=self_001`. Result: passed. Trace:
    `test_artifacts/llm_traces/l2d_action_initializer_live_llm__self_001__20260516T010607211930Z.json`.
    Observed action count: 0; no accidental `speak` or `send_message`.
  - `self_lifecycle_001`: same self-cognition fixture with
    `L2D_LIVE_CASE_ID=self_lifecycle_001`. Result: passed. Trace:
    `test_artifacts/llm_traces/l2d_action_initializer_live_llm__self_lifecycle_001__20260516T010622457717Z.json`.
    Observed action: private `memory_lifecycle_update`, lifecycle decision
    `abandoned`, target owner `user_memory_units`, evaluator owner
    `memory_lifecycle`, no leakage errors.
  - `self_future_001`: same self-cognition fixture with
    `L2D_LIVE_CASE_ID=self_future_001`. Result: passed. Trace:
    `test_artifacts/llm_traces/l2d_action_initializer_live_llm__self_future_001__20260516T010641443297Z.json`.
    Observed action: private `trigger_future_cognition`, target/evaluator owner
    `orchestrator`, no immediate cognition trigger, no leakage errors.
  - `self_013`: same self-cognition fixture with
    `L2D_LIVE_CASE_ID=self_013`. Result: passed. Trace:
    `test_artifacts/llm_traces/l2d_action_initializer_live_llm__self_013__20260516T010701802430Z.json`.
    Observed action: `speak`, `user_visible`, evaluator owner `l3_text`, no
    leakage errors.
  - Judgment: representative real LLM behavior matches the intended routing
    contract for ordinary dialog `speak`, self-cognition no-action,
    self-cognition visible `speak`, private promise lifecycle retirement, and
    private future cognition request.
- 2026-05-16 Stage 3 full L2d live LLM sweep before Stage 7.
  - User clarified that consolidation should run even when no visible action is
    selected, and requested all L2d LLM tests before continuing.
  - Live LLM execution policy: ran a separate pytest invocation per case for
    all frozen L2d fixtures, inspected trace-derived route summaries after each
    case, and stopped neither sweep because all cases passed.
  - QQ fixture:
    `test_artifacts/l2d_action_initializer/qq_673225019_cases.json`.
    Result: 20/20 passed. Route distribution: `speak` x20,
    `user_visible` x20, evaluator owner `l3_text` x20. Leakage errors: 0.
    Latest trace examples include
    `test_artifacts/llm_traces/l2d_action_initializer_live_llm__qq_001__20260516T011225945408Z.json`
    through
    `test_artifacts/llm_traces/l2d_action_initializer_live_llm__qq_020__20260516T011347858602Z.json`.
  - Self-cognition fixture:
    `test_artifacts/l2d_action_initializer/self_cognition_cases.json`.
    Result: 22/22 passed. Route distribution: no action x17,
    `trigger_future_cognition` x3, `speak` x1, and
    `memory_lifecycle_update` x1. Leakage errors: 0.
  - Self-cognition integration observations:
    `self_013` selected user-visible `speak` through `l3_text`;
    `self_future_001`, `self_006`, and `self_019` selected private
    `trigger_future_cognition` through `orchestrator`;
    `self_lifecycle_001` selected private `memory_lifecycle_update` through
    `memory_lifecycle`; all other self-cognition cases emitted no action.
  - Deterministic materialization guard observed in the live logs: some
    self-cognition cases produced an unresolvable lifecycle request, which was
    dropped with `L2d dropped lifecycle request without target commitment`.
    The final emitted action sets still passed route comparison and leakage
    checks. This confirms deterministic code owns persistence target resolution
    and prevents generic promise cleanup when the source evidence is not
    resolvable.
  - Integration judgment: the full live L2d sweep now covers ordinary dialog
    surface selection, self-cognition no-action/private-consolidation cases,
    self-cognition visible surface selection, private future cognition request,
    and private memory lifecycle update. Stage 7 remains pending.
- 2026-05-16 target-binding correction before Stage 7.
  - Root cause: self-cognition active-commitment due checks carried the target
    as source-ref/percept lineage, while L2d lifecycle materialization only
    trusted `rag_result.user_image.user_memory_context.active_commitments`.
    Some L2d runs could therefore semantically request lifecycle work without a
    deterministic memory target.
  - Documentation update: `cognition_contracts_design.md` now defines
    deterministic target binding as runtime-only; the action-spec architecture
    and this execution plan require target IDs/source refs to be attached by
    code, never copied or selected by L2d.
  - Implementation update: self-cognition active-commitment source refs are
    projected into deterministic `active_commitments` context before L2d, L2d
    hides `memory_lifecycle_update` when no single active-commitment target is
    bound, and materialization no longer matches target IDs from LLM text.
  - Focused failing-before/fixed-after test:
    `venv\Scripts\python -m pytest tests/test_persona_supervisor2_action_initializer.py -q`
    first failed on source-ref binding, lifecycle prompt visibility, and
    ID-copied target selection; after implementation it passed with 7 passed.
  - Adjacent deterministic verification:
    `venv\Scripts\python -m pytest tests/test_persona_supervisor2_action_initializer.py tests/test_self_cognition_integration.py tests/test_l2d_action_initializer_cases.py tests/test_action_spec_evaluator.py tests/test_action_spec_memory_lifecycle.py -q`
    result 42 passed.
  - Syntax verification:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\kazusa_ai_chatbot\self_cognition\runner.py tests\test_persona_supervisor2_action_initializer.py`
    passed.
  - Real LLM spot check for a previously affected frozen case:
    `L2D_LIVE_CASE_FILE=test_artifacts\l2d_action_initializer\self_cognition_cases.json L2D_LIVE_CASE_ID=self_006 venv\Scripts\python -m pytest tests/test_l2d_action_initializer_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -s -m live_llm`
    passed. Trace:
    `test_artifacts/llm_traces/l2d_action_initializer_live_llm__self_006__20260516T014155460718Z.json`.
    Prompt capabilities were `speak,trigger_future_cognition`,
    `active_commitments` count was 0, observed action was private
    `trigger_future_cognition`, and leakage errors were 0.
  - Hygiene verification: `git diff --check` exited 0 with Windows CRLF
    normalization warnings only.
