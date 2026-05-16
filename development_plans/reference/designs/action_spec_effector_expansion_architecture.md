# Action Spec Effector Expansion Architecture

## Status

- Type: reference architecture and decision record
- Status: draft reference
- Related execution plan:
  `development_plans/active/short_term/modality_neutral_action_spec_effector_expansion_plan.md`
- Execution rule: do not execute directly from this document

This document captures research context, architectural decisions, and design
justification for the modality-neutral action spec and effector expansion work.
The active short-term plan contains only executable steps and verification
gates.

`development_plans/reference/designs/cognition_contracts_design.md` is the
approved registry-level source of truth for the seven cognition contracts. This
document is subordinate design rationale; if the two conflict, the contracts
design wins.

## Research Inputs

The architecture is based on these observed repo facts and prior design records:

- The attached log showed cognition detecting an expired promise and choosing to
  fulfill it by speaking. This proves expired promises can surface through the
  normal cognition path, but the system lacks a character-selected action for
  abandoning or retiring the promise without speaking.
- `development_plans/reference/designs/cognition_core_evolution_progression.md`
  defines trigger-source pluralism and the seven contracts. It says expansion
  plans derived from that progression must not be approved before
  `cognition_contracts_design.md` exists.
- The same progression historically recorded the single adapter-facing
  dispatcher state: scheduled delivery used `send_message`, while private
  memory or cognition actions did not yet have a shared contract.
- `development_plans/archive/completed/short_term/self_cognition_agency_loop_plan.md`
  proved self-cognition can select `send_message` and route it through the
  existing dispatcher. It also avoided forcing action only because a commitment
  is past due.
- `src/kazusa_ai_chatbot/dispatcher/README.md` separates semantic acceptance
  from operational execution: the LLM decides accepted promises, the evaluator
  checks schema and permissions, the dispatcher deduplicates and persists, the
  scheduler executes, and adapters deliver.
- `src/kazusa_ai_chatbot/proactive_output/README.md` says production proactive
  output must not bypass dispatcher or scheduler validation.
- `src/kazusa_ai_chatbot/self_cognition/README.md` documents
  `self_cognition_action_attempts` as the durable idempotency-keyed action
  ledger for self-cognition duplicate suppression.
- `src/kazusa_ai_chatbot/db/README.md` documents
  `self_cognition_action_attempts` and `scheduled_events` as existing control
  collections.
- `src/kazusa_ai_chatbot/db/schemas.py` defines `user_memory_units` statuses:
  `active`, `archived`, `completed`, and `cancelled`.
- `src/kazusa_ai_chatbot/db/user_memory_units.py` currently reads active
  commitment units by querying `unit_type=active_commitment`,
  `status=active`, and due timestamps.
- `src/kazusa_ai_chatbot/memory_evolution/README.md` documents
  `EvolvingMemoryDoc` lineage through `lineage_id`, `version`,
  `supersedes_memory_unit_ids`, `merged_from_memory_unit_ids`, and
  `evidence_refs`.
- The progression document identifies `source_reflection_run_ids` as an
  existing cross-subsystem lineage primitive on reflection-derived artifacts.

## Problem

The system can recognize that a promise is past due, and it can choose to speak
about it. It cannot yet express the equally valid character decision to retire,
abandon, or mark the promise obsolete without user-visible output.

The solution must not be a deterministic cleanup rule such as "remove promises
after X days." It must preserve this ownership split:

```text
LLM cognition: semantic decision
deterministic code: validation, permissions, persistence, scheduling, delivery,
audit, and cache invalidation
```

The action vocabulary must also support future capabilities such as scheduled
self-checks, web research, notes, and user-visible contact without creating a
new semantic channel for each tool.

## Architectural Principles

- One shared semantic path handles user input, internal thought, self-cognition,
  reflection-derived episodes, scheduled checks, and tool-result episodes.
- Action specs are cognition residue. They are not raw tool calls.
- Deterministic validators may reject unsafe or invalid action specs, but they
  must not invent or rewrite semantic intent.
- Dispatcher remains the execution owner for adapter-facing and scheduled
  user-visible tools.
- Memory lifecycle actions belong to the target memory owner, not the
  dispatcher.
- Self-cognition is a trigger source for the same shared path, not a downstream
  action consumer or private cleanup channel.
- L3 text and L3 image are selected surface handlers under action specs, not
  always-on branches outside the action contract.
- Tool and action results re-enter cognition only through typed continuation
  episodes. Tool handlers do not call cognition or write final dialogue.
- Consolidation consumes prompt-safe episode traces. Final dialog is one text
  surface output, not the only turn result.
- Consolidator consumes action/surface evidence but does not select actions,
  execute actions, dispatch, schedule, or trigger cognition.
- Local-LLM constraints require a small, explicit capability set and bounded
  prompt-safe affordance projection.
- Existing lineage and ledger primitives must be reused instead of creating
  parallel control state.

## Seven-Contract Relationship

The progression document separates seven contracts. This architecture uses only
a narrow slice of them.

| Contract | Relationship In This Architecture |
|---|---|
| 1. Trigger source | Self-cognition, scheduled ticks, tool results, and user messages enter as typed cognitive episodes. |
| 2. Inter-layer residue bus | L2d action requests, action specs, L3 surface outputs, action results, and consolidation projections are residue/trace surfaces. |
| 3. Modality-neutral action spec | This architecture defines the initial `ActionSpecV1` slice. |
| 4. Affordance registry | Referenced as prompt-safe capability projection. Full registry shape is deferred to `cognition_contracts_design.md`. |
| 5. Engine routing layer | Not defined here. |
| 6. Memory layer interface | Used through existing `user_memory_units` and memory-evolution owners. Full provider interface is deferred. |
| 7. Capability surface uniformity | Referenced through action-category `CapabilitySpecV1` entries; full extension pattern is deferred to `cognition_contracts_design.md`. |

The active execution plan must remain reconciled against the approved contracts
reference before each new implementation stage.

## Core Decision

Do not merge action spec into dispatcher.

Action spec is the graph-visible contract materialized from L2d's semantic
action request. Dispatcher is one execution owner for a subset of actions:
scheduled and adapter-facing tools.

The relationship should be:

```text
L2d semantic action request
-> ActionSpecV1 materialization
-> ActionSpecEvaluator / capability policy
-> selected execution owner or L3 surface handler
   -> L3 text/image for surface outputs
   -> dispatcher bridge for delivered text after surface output exists
   -> memory lifecycle handler for promise retirement
   -> orchestrator for future cognition continuation
-> ActionResultV1 + SurfaceOutputV1
-> EpisodeTraceV1 for consolidation
```

For visible text, `speak` selects the L3 text surface and `send_message` is only
the dispatcher bridge after deliverable text exists:

```text
ActionSpecV1(kind="speak")
-> L3 text handler
-> SurfaceOutputV1(surface_kind="text")
-> delivery validation
-> RawToolCall(tool="send_message", args=...)
-> TaskDispatcher.dispatch(...)
```

For `memory_lifecycle_update`, dispatcher is not involved because the action is
private persistence, not adapter-facing delivery.

The practical rule is: keep modules separate, unify the registration model.
Dispatcher becomes one capability owner under the action-spec system, not the
parent system.

## Decision Table

| Topic | Decision | Justification |
|---|---|---|
| Action spec vs dispatcher | Keep separate modules and bridge dispatcher-owned actions. | Prevents dispatcher from becoming a generic semantic action bus while preserving the proven `send_message` path. |
| Registration model | Use semantic action-category `CapabilitySpecV1` entries for cognition-visible capabilities; dispatcher `ToolSpec` remains execution-specific. | Avoids two competing semantic registries while preserving dispatcher permissions and adapter validation. |
| Current runtime scope | Implement action spec, evaluator, L2d semantic action initialization, L3 text surface routing, episode-trace consolidation, dispatcher bridge compatibility, future-cognition request contract, and user-memory lifecycle update first. | Solves the expired-promise case while preserving one shared action/consolidation path for text, private actions, and scheduled follow-up. |
| L2d boundary | L2d emits compact semantic `action_requests`; deterministic code materializes `ActionSpecV1`. | Keeps schema versions, handler IDs, source refs, transport targets, and persistence IDs out of the local LLM prompt. |
| Target binding | Build action targets from deterministic binding context, never from L2d-emitted IDs or copied refs. | The local LLM can decide semantics, but target IDs belong to trusted episode/source/RAG/repository context and must not become model-facing selectors. |
| L3 surface ownership | Treat L3 text/image as selected action handlers. | Enables no-response, delayed response, image surface, and future tool surfaces without always-on branches. |
| Self-cognition integration | Treat self-cognition as an upstream trigger source. | Keeps self-checks on the same L1/L2/L2d/action/consolidation path and avoids a private cleanup channel. |
| Consolidation basis | Consolidate from episode trace, not only final dialog. | Parallel actions, private memory updates, scheduled cognition requests, and future image/tool outputs need durable state updates even when no text is sent. |
| Promise retirement | Express as `memory_lifecycle_update`, not deletion or stale-age cleanup. | The character decides; deterministic code only validates and persists. |
| Promise store | Treat `user_memory_units.active_commitment` as the runtime promise/open-commitment store for this slice. | Current self-cognition due checks already read this shape. |
| Status mapping | `fulfilled -> completed`, `abandoned -> cancelled`, `obsolete -> archived`, `deferred -> active`. | Reuses existing collection statuses; no migration needed. |
| `EvolvingMemoryDoc` | Do not mutate it in the first execution plan; reject lifecycle action specs targeting it. | Prevents lineage damage until a dedicated memory-provider contract exists. |
| Action ledger | Reuse `self_cognition_action_attempts` through a generic action-attempt repository. | Avoids a second idempotency/control-state collection. |
| Source references | Project action refs onto existing `source_reflection_run_ids`, `evidence_refs`, and `source_refs`. | Avoids creating a fourth lineage primitive. |
| Cognition mode | Include `cognition_mode`, accept only `deliberative` initially, reject `reflex`. | Reserves the roadmap-required reflex slot without enabling an unreviewed reflex path. |
| Continuation | Add continuation policy to action specs; handlers return action results, orchestrator starts follow-up episodes. | Lets future tools trigger cognition without calling cognition directly. |
| Web research | Defer runtime implementation. | It requires evidence shaping, citations, timeout policy, and follow-up-cycle controls. |
| Schedule self-check | Defer runtime implementation. | It needs scheduler policy and typed trigger-source integration. |

## Target Architecture

```text
typed episode
  user message | internal thought | self-cognition | scheduled tick | tool result
        |
        v
shared evidence assembly / RAG
        |
        v
shared cognition L1/L2a/L2b/L2c
        |
        v
L2d semantic action request list
        |
        v
deterministic target binding + ActionSpecV1 materialization + evaluation
        |
        v
selected L3 surfaces / action handlers
        |
        +--> L3 text/image -> SurfaceOutputV1
        |
        +--> dispatcher bridge -> TaskDispatcher -> scheduler/adapters
        |
        +--> memory lifecycle handler -> user_memory_units repository
        |
        +--> orchestrator continuation/future cognition -> typed follow-up episode
        |
        v
ActionResultV1 + SurfaceOutputV1
        |
        v
EpisodeTraceV1 -> consolidator prompt-safe projection -> memory/state/progress
```

The action spec path is additive. Existing final dialog and dispatcher behavior
remain valid, but final dialog is treated as a text surface output rather than
the only consolidation basis.

Target binding is part of materialization, not a model-facing planning task.
For `memory_lifecycle_update`, deterministic code resolves the active
commitment from trusted episode/source references and current RAG memory
context, then attaches `target`, `source_refs`, and `unit_id` to the
materialized spec. If no single eligible commitment can be resolved, the action
is rejected before any memory write. The prompt may receive a plain-language
summary of the bound commitment and capability availability, but it must not
receive or emit memory unit IDs, source-ref IDs, collection names, owner names,
handler IDs, adapter IDs, or scheduler IDs.

## Contract Sketch

`ActionSpecV1`:

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

Action-category `CapabilitySpecV1` entry:

```python
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

Action-specific prompt affordances, allowed cognition modes, continuation
rules, and source-ref requirements are exposed through the affordance registry
and capability policies defined by `cognition_contracts_design.md`.

`ActionContinuationV1`:

```python
class ActionContinuationV1(TypedDict):
    schema_version: Literal["action_continuation.v1"]
    mode: Literal["none", "immediate_followup", "scheduled_followup", "background_followup"]
    episode_type: str | None
    max_depth: int
    include_result_as: str | None
```

The initial runtime slice accepts only:

- `speak` with owner `l3_text`;
- `memory_lifecycle_update` with owner `memory_lifecycle`;
- `trigger_future_cognition` with owner `orchestrator`;
- bridge-only `send_message` with owner `dispatcher`, after a text surface
  exists.

Future tools may use continuation, but the first execution plan validates the
contract without shipping runtime web research or external image generation.

## Existing Primitive Integration

### Action Attempts

Use a generic action-attempt repository backed by
`self_cognition_action_attempts`. Add fields tolerantly so old rows remain
readable for duplicate suppression.

Required action-attempt metadata:

- `idempotency_key`
- `trigger_source`
- `action_kind`
- `cognition_mode`
- serialized action-spec metadata
- validation status
- handler owner
- continuation status
- execution result
- recorded timestamp

### User Memory Lifecycle

Add a narrow lifecycle helper under the `user_memory_units` owner. It updates
only lifecycle fields, `updated_at`, and an audit entry. It rejects:

- missing units;
- non-`active_commitment` units;
- invalid status transitions;
- empty reasons;
- generic field updates.

### Memory Evolution

`EvolvingMemoryDoc` remains read/reference-only for this slice. If a lifecycle
action spec targets it, validation rejects the action without persistence.

Future integration must preserve lineage by using repository-owned supersede,
merge, or explicit lifecycle APIs. It must not use ad hoc MongoDB `$set`.

### Source References

Action source references are runtime projections, not a new persisted lineage
shape. Persistence maps them to existing owner fields:

- `source_reflection_run_ids` for reflection-derived artifacts;
- `evidence_refs` for memory-evolution evidence;
- `source_refs` for user memory units.

## Capability Roadmap

### Runtime Scope For First Execution Plan

1. `speak`
   - Selected text surface action.
   - Owner: L3 text handler.
   - Continuation: `none`.
   - Execution: generate `SurfaceOutputV1(surface_kind="text")`.

2. `memory_lifecycle_update`
   - Private lifecycle update for `user_memory_units.active_commitment`.
   - Owner: memory lifecycle handler plus user-memory repository.
   - Continuation: `none`.
   - Execution: validated repository write and action-attempt audit.

3. `trigger_future_cognition`
   - Private request for a future typed cognition episode.
   - Owner: orchestrator/scheduler boundary.
   - Continuation: `scheduled_followup`.
   - Execution: validated future episode request; no direct cognition call.

4. Bridge-only `send_message`
   - Existing adapter-facing outbound delivery after text exists.
   - Owner: dispatcher.
   - Continuation: `none`.
   - Execution: bridge to `RawToolCall` and `TaskDispatcher`.

### Next-Stage Tool Candidates

1. `web_research` or `fetch_url`
   - Produces evidence artifacts with citations and bounded result size.
   - Result re-enters cognition through `immediate_followup`.
   - Must not author final dialogue directly.

2. `note_open_loop` / `close_open_loop`
   - Should be wrappers over memory lifecycle or conversation-progress owner
     APIs, not a separate freeform note store.

3. Image generation or external messaging
   - Deferred until product rules, storage, adapter delivery, permission, and
     audit boundaries are specified.

## Rationale For Deferring Runtime Web, Image, And External Tools

The current need is expired-promise retirement. Shipping web research or
external image/message tools in the same execution plan would expand three
additional risk surfaces:

- external evidence acquisition and citation policy;
- adapter/storage/product policy for new output modalities;
- continuation loop depth and latency control.

The continuation field should exist now so future tools share the same action
contract, but runtime execution of those tools should wait for dedicated
capability plans.

## Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Action spec becomes a second dispatcher | Keep dispatcher as execution owner only for adapter-facing tools. |
| Dispatcher becomes the semantic action bus | Keep `ActionSpecV1` and action-category `CapabilitySpecV1` entries outside dispatcher. |
| Deterministic code retires promises by age | Require cognition-authored `memory_lifecycle_update` with source refs and reason. |
| Existing ledgers fragment | Back generic action attempts with `self_cognition_action_attempts`. |
| Lineage primitives multiply | Map action source refs to existing owner fields. |
| Reflex path ships accidentally | Reject `reflex` until a future allow-listed reflex capability plan exists. |
| Local LLM emits invalid structured actions | Keep actions optional, validate deterministically, and run offline validity/latency measurement before broad live rollout. |
| Action outcomes do not reach durable memory/state | Feed `ActionResultV1` and `SurfaceOutputV1` into `EpisodeTraceV1` and project it into consolidation. |
| Consolidator becomes an executor | Forbid dispatcher, scheduler, cognition, and action-handler calls from consolidator code. |

## Reference Status Notes

- `cognition_contracts_design.md` is approved and authoritative.
- Derived execution plans must still reconcile against that reference before
  each stage that changes contracts, prompts, graph wiring, action execution,
  consolidation, or documentation.

## Non-Goals

- This document does not approve execution.
- This document does not define the full seven-contract system.
- This document does not authorize web research, scheduled self-check runtime
  execution, arbitrary notes, image generation, adapter changes, or
  `EvolvingMemoryDoc` lifecycle mutation.
