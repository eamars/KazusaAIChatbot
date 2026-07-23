# Cognition Contracts Design

## Status

- Type: authoritative reference design
- Status: approved reference
- Scope: registry-level schemas, ownership rules, and extension patterns for the
  seven cognition contracts
- Execution rule: do not execute directly from this document

This document consolidates the forward-looking contract decisions from
`cognition_core_evolution_progression.md`, the self-cognition architecture
references, and the action-spec effector-expansion design discussion.

This document is the authoritative source for contract-level shape and
ownership. Where it conflicts with older forward-looking design notes, this
document wins. Current source code and subsystem READMEs remain authoritative
for current runtime behavior until an approved execution plan changes them.

## Purpose

Kazusa's cognition architecture must support multiple trigger sources, multiple
output surfaces, future tool use, engine swaps, and memory-provider evolution
without accumulating ad hoc per-feature shapes.

The seven contracts below are the stable boundaries that make that possible:

1. Trigger source registry
2. Inter-layer residue bus
3. Modality-neutral action spec
4. Affordance registry
5. Engine routing layer
6. Memory layer interface
7. Capability surface uniformity

This document defines the registry-level shapes for those contracts. It does
not authorize implementation by itself.

## Non-Goals

- Do not implement these contracts from this reference alone.
- Do not treat this document as permission to add runtime tools.
- Do not add a new semantic channel for self-cognition, reflection, tools, or
  promise cleanup.
- Do not replace dispatcher, scheduler, RAG, memory repositories, adapters, or
  model clients in this reference.
- Do not expose raw database schemas, raw adapter IDs, credentials, handler IDs,
  prompt text, or internal storage details to LLM prompts.
- Do not add deterministic semantic cleanup for promises, preferences,
  commitments, permissions, or user instructions.

## Global Rules

- LLM stages own semantic judgment: affect, stance, intent, action choice,
  promise retirement, and final wording.
- Deterministic code owns validation, permissions, limits, scheduling,
  persistence, cache invalidation, adapter delivery, rate limits, and audit.
- The shared semantic path is:

```text
typed episode -> evidence assembly / RAG -> cognition concerns
-> L2c2 social context appraisal -> L2d action initialization
-> ActionSpecV1 materialization/evaluation
-> selected L3 surfaces / action handlers -> ActionResultV1 + SurfaceOutputV1
-> episode-trace consolidation -> persistence / scheduler / adapter bridge
```

- User input, internal thought, self-cognition, scheduled ticks, and tool
  results enter cognition as typed episodes. They do not masquerade as one
  another.
- Self-cognition is a trigger source for the same cognition/action/consolidation
  pipeline. It is not a downstream consumer, private cleanup path, or alternate
  action executor.
- A selected self-cognition `speak` uses the same shared
  cognition/action/consolidation pipeline and the same shared cognition/dialog/persistence path
  as other selected speech, then hands the rendered text to the runtime adapter bridge
  after dialog rendering. This does not create a separate self-cognition
  action-execution owner and does not change LLM prompt, schema, or model
  ownership.
- Reflection-derived material can be projected as gated context or evidence
  after another valid trigger exists. It is not an initial trigger source.
- Raw reflection output does not enter normal cognition. Only promoted, gated,
  typed context may be used.
- Tool handlers do not call cognition directly. A tool or action result can
  request continuation only by returning a typed continuation contract;
  orchestration enqueues the next typed episode.
- LLM-to-LLM handoff information is exactly one prompt-safe semantic string.
  TypedDicts, JSON objects, source refs, IDs, schema versions, target scopes,
  lifecycle markers, and routing metadata are deterministic envelopes only.
  They may store, validate, route, or audit the handoff, but they must not be
  projected to the next LLM as structural meaning. If the required handoff
  cannot be represented faithfully as one string, the handoff fails closed.
- Final user-visible dialog is one surface output, not the whole turn result.
  Image, tool, private, scheduled, and no-reply outcomes must also be traceable.
- Consolidation consumes an episode trace: trigger, prompt-safe cognition
  residue, selected action specs, action attempts/results, and surface outputs.
  The consolidator is not an action planner or executor.
- Zero, one, or many actions may be selected for one episode. Independent
  actions may execute in parallel when handlers declare no dependency.
  Dependent chains use continuation and a later cognition episode.
- Local-model constraints are first-class. Prompt-visible registries must be
  bounded, semantic, and relevant to the current episode.
- Each contract and wire payload has a `schema_version`. Breaking changes
  require a new major version and an approved migration or compatibility plan.
- Timestamp strings, including `created_at`, `updated_at`, `observed_at`, and
  `deadline`, use ISO-8601 UTC with an explicit `Z` or `+00:00` offset unless a
  field name explicitly states another timezone.
- Policy fields are registry keys, not prose. They use lower snake-case
  namespaces such as `rate_limit.live_chat`, `privacy.user_scoped`,
  `permission.dispatcher_send_message`, `lineage.evidence_refs`,
  `audit.action_attempt`, and `prompt_projection.semantic_summary`.

### Shared Registry Types

```python
CognitionConcernV1 = Literal["L1", "L2", "L3", "orchestrator"]
PolicyRefV1 = str
```

`PolicyRefV1` strings must match:

```text
^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$
```

## L1, L2, And L3

L1, L2, and L3 are concerns, not fixed process stages.

- L1: affective interpretation, salience, and immediate appraisal.
- L2: deliberative stance, character intent, social-context appraisal,
  modality-neutral action, and reasoning residue.
- L3: output-surface specialization, including text, image, tool, motor, or
  other modality-specific rendering.

Current live text chat usually composes the final user-visible text path as:

```text
L1 -> L2 -> L3-text
```

The action-spec architecture makes that text path one selected action surface:

```text
L1/L2 residue + L2c2 social context -> L2d action request(kind="speak")
-> ActionSpecV1(kind="speak") -> L3-text handler -> SurfaceOutputV1(text)
```

`send_message` is not the semantic text action. It is the dispatcher/adaptor
bridge used after a text surface exists and delivery is allowed.

When enabled, the current cognition graph also has an L3 visual-agent path after
L2. That path can produce still-frame visual directives such as facial
expression, body language, gaze direction, and visual vibe. Those directives
are a valid precursor for an image-generation prompt and can be connected to an
image generation service through a registered L3 image surface or action
capability.

Available contract assemblies include:

```text
L1 -> L3-reflex
L1 -> L2 -> L3-text
L1 -> L2 -> L3-visual-directives -> image prompt -> image generation service
L1 -> L2 -> L3-tool
L1 -> L2 -> L3-image
L1 -> L2 -> L3-motor
```

All non-reflex L3 surfaces consume the same L2 residue and L2c2 social-context
fields. L2 must remain
output-modality-neutral. The visual/image path does not create a separate
cognition channel: image generation remains an L3 surface or action capability
with deterministic validation, permission, execution, and audit. The first
action-spec execution plan may still defer runtime image-generation execution
while this reference records the contract path as available.

When multiple L3 surfaces are selected, each surface returns its own
`SurfaceOutputV1` and action result. The episode trace joins those outputs for
delivery, audit, and consolidation instead of treating one final-dialog string
as the only observable product of cognition.

## Reflex Policy

Reflex is a first-class structural slot, not a latency shortcut.

Reflex actions use:

```text
L1 + affordance lookup -> allow-listed L3 surface
```

Rules:

- Reflex primitives must be explicitly allow-listed in the affordance registry.
- Reflex must never be used because L2 is slow or unavailable.
- A reflex that L2 policy would have forbidden is a bug.
- Current text-chat action-spec work reserves `cognition_mode="reflex"` but
  rejects it until a future approved reflex plan registers allow-listed
  primitives.

## Contract 1: Trigger Source Registry

### Purpose

Trigger sources describe why cognition is running. New sources register through
a typed registry instead of adding branches inside cognition core.

### Initial Committed Source Kinds

```text
user_message
internal_thought
self_cognition
scheduled_tick
tool_result
```

### Reserved Source Kinds

```text
system_event
```

`system_event` is reserved for future orchestration-owned events, but it is not
part of the first committed registry. A future execution plan must define its
owner, evidence policy, privacy policy, and rate limits before enabling it.

### Reflection Promotion Disposition

`reflection_promoted` is intentionally not a trigger source in the initial
registry. The reasoning basis ranks a reflection artifact as rejected for
starting self-cognition because it is produced self-observation, not a useful
agency trigger. Promoted reflection may still be included as `EvidenceRefV1` or
prompt-safe projected context when another source, such as `scheduled_tick`,
`self_cognition`, or `user_message`, legitimately triggers cognition.

Any future plan that wants reflection promotion to start cognition must amend
this reference explicitly and explain why it overrides that reasoning record.

### Self-Cognition Disposition

`self_cognition` is an initial trigger source. It represents an internally
generated reason to run the same shared cognition path, such as a private
self-check, a due commitment review, or a scheduled open-loop inspection.

It does not own action execution. Once a self-cognition episode reaches L2d,
action selection, action-spec materialization, handler execution, continuation,
and consolidation are the same as for `user_message` or `scheduled_tick`
episodes. A self-cognition result that needs later reasoning must return a
continuation contract for orchestration to enqueue as a new typed episode.

### Reference Shapes

```python
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


class PerceptV1(TypedDict):
    schema_version: Literal["percept.v1"]
    percept_kind: str
    source_kind: str
    source_id: str | None
    content: dict[str, object]
    observed_at: str


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
```

### Registry Shape

```python
class TriggerSourceSpecV1(TypedDict):
    schema_version: Literal["trigger_source_spec.v1"]
    source_kind: str
    owner: str
    entrypoint: str
    llm_visibility: Literal["prompt_visible", "metadata_only", "hidden"]
    evidence_policy: Literal["requires_evidence", "optional_evidence", "no_evidence"]
    persistence_policy: Literal["ephemeral", "audited", "durable"]
    rate_limit_policy: PolicyRefV1
    privacy_policy: PolicyRefV1
    allowed_continuation_depth: int
```

### Episode Envelope

```python
class CognitiveEpisodeV1(TypedDict):
    schema_version: Literal["cognitive_episode.v1"]
    episode_id: str
    trigger_source: str
    origin_metadata: dict[str, object]
    target_scope: dict[str, object]
    percepts: list[PerceptV1]
    evidence_refs: list[EvidenceRefV1]
    created_at: str
    privacy_scope: str
    continuation_depth: int
```

`origin_metadata` and `target_scope` remain source-specific deterministic
metadata. Source-specific schemas belong to the trigger-source owner and must
not be inferred by cognition prompts from raw adapter fields.

### Ownership

- Intake/orchestration validates source shape and continuation depth.
- RAG/evidence assembly converts source material into prompt-safe evidence.
- Cognition consumes the prompt-safe episode projection.
- Adapters do not define cognition semantics.

### Perception And Streaming Disposition

Continuous modality encoders, salience gates, TTS buffers, animation players,
and interruption handling are frontend or orchestration runtime concerns, not
an eighth contract in this reference. When a salience gate selects a discrete
event for cognition, that event enters through this trigger-source registry.
When output streaming is added, it registers as an L3 surface and capability
owner rather than a new trigger path.

## Contract 2: Inter-Layer Residue Bus

### Purpose

Residue is the typed information passed between cognition concerns. It replaces
unowned free-form dictionaries over time.

### Registry Shape

```python
class ResidueTypeSpecV1(TypedDict):
    schema_version: Literal["residue_type_spec.v1"]
    residue_kind: str
    producer_concern: CognitionConcernV1
    consumer_concerns: list[CognitionConcernV1]
    required_fields: list[str]
    optional_fields: list[str]
    prompt_visible_fields: list[str]
    persistence_policy: Literal["none", "audit_only", "durable"]
```

### Initial Residue Shapes

```python
class L1ResidueV1(TypedDict):
    schema_version: Literal["l1_residue.v1"]
    emotional_appraisal: str
    interaction_subtext: str
    salience_hints: list[str]
    risk_flags: list[str]


class L2ResidueV1(TypedDict):
    schema_version: Literal["l2_residue.v1"]
    logical_stance: str
    character_intent: str
    action_specs: list[ActionSpecV1]
    internal_monologue: str
    affordance_assumptions: list[str]


class L3ResidueV1(TypedDict):
    schema_version: Literal["l3_residue.v1"]
    output_surface: str
    rendered_output_refs: list[str]
    delivery_intent: str
    blocked_reasons: list[str]
```

### Rules

- L2 residue is modality-neutral.
- L3 surfaces consume residue; they do not reinterpret user intent.
- Prompt text is not residue. Residue is structured output or deterministic
  projection.
- Missing optional residue fields must degrade safely without adding repair LLM
  loops by default.

## Contract 3: Modality-Neutral Action Spec

### Purpose

Action specs express character-selected actions after L2d has made a semantic
action request and deterministic materialization has added mechanical envelope
fields. They are cognition residue for execution owners, not dispatcher tool
calls and not prompt text.

### Shape

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

```python
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
```

### Rules

- `kind` is drawn from the capability registry.
- `cognition_mode="reflex"` requires an allow-listed reflex affordance.
- Source refs are required for actions that mutate memory, schedule work, or
  contact a user outside the current visible reply.
- Action or tool results that need cognition use continuation. Handlers return
  typed results with continuation data; orchestration starts the next episode.
- Dispatcher-owned actions bridge to dispatcher execution shapes only after
  action-spec validation.
- Memory lifecycle actions route to memory owners, not dispatcher.
- `ActionSpecV1` may be generated from a compact LLM semantic request, but LLM
  prompts must not be asked to emit schema versions, handler IDs, raw transport
  targets, raw persistence IDs, or other deterministic envelope fields.
- L2d can select zero, one, or many action requests. Evaluation may reject,
  schedule, execute, or defer each materialized spec independently.

### LLM Handoff Text

Continuation that feeds a later cognition episode carries one LLM-facing text
contract:

```python
ContinuationObjectiveV1 = str
```

`ContinuationObjectiveV1` is not a summary. It is the future thinking contract
for the next cognition cycle. It must preserve the concrete subject, requested
outcome, unresolved commitment or action objective, and any detail required for
the next cycle to reason without guessing. Proper nouns, object names, user-
requested targets, commitments, and selected action anchors must not be
paraphrased away.

Deterministic wrappers may store `ContinuationObjectiveV1` inside typed action
params, scheduled-event args, action results, or episode metadata, but the next
LLM consumes only this one string as handoff text. If a selected future
cognition action continues another selected action, deterministic
materialization should prefer that concrete sibling action's semantic `detail`
as the continuation objective instead of asking L2d to regenerate the same
objective in a second field. Supporting reasons, timing, refs, and audit fields
remain deterministic metadata and may not replace the continuation objective.

If no single faithful string can be built, the continuation request is invalid.

### Deterministic Target Binding

Targets for persistence, delivery, scheduling, and continuation are bound by
deterministic code before or during action-spec materialization. L2d may select
the semantic capability and semantic decision, but it must not receive, copy,
rank, compare, or emit raw target identifiers, source-reference identifiers,
database owners, collection names, adapter IDs, scheduler IDs, or handler IDs.

The target binding context is runtime-only. It may be assembled from trusted
episode metadata, trigger-source references, RAG/evidence assembly output, and
existing repository rows. Prompt-safe evidence may describe the selected target
in plain language, but the model-facing payload must not contain the target ID.
If an action requires a target and deterministic code cannot resolve exactly
one eligible target from trusted context, the capability must be hidden from
the prompt or the materialized action must be rejected before persistence.

`memory_lifecycle_update` follows this rule strictly: the model decides only
whether the character semantically wants `fulfilled`, `abandoned`, `obsolete`,
or `deferred`. Deterministic materialization attaches the `memory_unit` target,
`source_refs`, `unit_id`, owner, lifecycle params, and audit fields from the
bound active-commitment context. It must not recover targets by asking L2d to
copy a memory unit ID, source ref, array index, or other opaque selector.

### Result And Surface Shapes

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


class EpisodeAttemptDiagnosticV1(TypedDict):
    stage: str
    error_code: str
    attempt_count: int
    safe_checkpoint: str
    retryable: bool
    final_status: str


class DeliveryCorrelationV1(TypedDict):
    delivery_intent: Literal["deliver_now", "deliver_later", "do_not_deliver"]
    delivery_tracking_id: str
    receipt_status: Literal[
        "not_applicable", "pending", "delivered", "failed", "unknown"
    ]
    receipt_ref: str


class EpisodeTraceV2(TypedDict):
    schema_version: Literal["episode_trace.v2"]
    episode_id: str
    trigger_source: str
    terminal_status: Literal[
        "completed_visible",
        "completed_private",
        "completed_action",
        "scheduled",
        "failed",
        "cancelled",
    ]
    cognition_refs: list[EvidenceRefV1]
    action_specs: list[ActionSpecV1]
    action_results: list[ActionResultV1]
    surface_outputs: list[SurfaceOutputV1]
    attempt_diagnostics: list[EpisodeAttemptDiagnosticV1]
    delivery_correlation: DeliveryCorrelationV1
    created_at: str
    settled_at: str


class ConsolidationActionProjectionV1(TypedDict):
    schema_version: Literal["consolidation_action_projection.v1"]
    action_kind: str
    status: str
    visibility: Literal["private", "preview", "user_visible"]
    semantic_decision: str
    result_summary: str
    evidence_refs: list[EvidenceRefV1]
```

`EpisodeTraceV2` is the immutable consolidation input contract. It settles once
after primary cognition/actions/surfaces and delivery tracking assignment. A
later adapter receipt remains separately persisted and does not mutate the
trace. The trace may include completed, scheduled, pending, rejected, or failed
actions. The consolidator receives
prompt-safe projections of the trace; it does not receive raw handler IDs,
credentials, raw adapter IDs, raw collection names, or arbitrary action params.

### Committed Runtime Action Kinds

```text
speak
memory_lifecycle_update
apply_memory_lifecycle_update
trigger_future_cognition
future_speak
accepted_task_request
accepted_coding_task_request
accepted_task_status_check
background_work_request
```

This roster is complete for the Stage 3 native runtime. The registry is the
declarative schema/permission/prompt/handler authority for all nine kinds.
`apply_memory_lifecycle_update` is internal-only. Task, coding, status,
background, and future-speech requests retain their typed lifecycle/queue/
scheduler owners; registering them does not permit the cognition model to
execute their effects directly. A tenth kind requires an approved plan and an
authoritative-reference amendment before runtime implementation.

`send_message` is not cognition-visible vocabulary. Text output remains the
semantic `speak` action, with L3 text/dialog and the live service boundary
owning wording and adapter delivery for the current turn.

The runtime roster does not expose web research, notes, image generation,
motor actions, or arbitrary external tools as direct cognition actions.
Supported accepted/background task requests may route bounded work to their
registered owners, and `trigger_future_cognition` materializes scheduler-owned
work without calling cognition directly.

## Contract 4: Affordance Registry

### Purpose

The affordance registry describes what the character can currently do. L2 uses
prompt-safe affordances as soft context. L2 must not assume hidden capability.

### Runtime Shape

```python
class AffordanceSpecV1(TypedDict):
    schema_version: Literal["affordance_spec.v1"]
    capability_kind: str
    owner: str
    surface: str
    availability: Literal["available", "unavailable", "degraded"]
    visibility: Literal["private", "preview", "user_visible"]
    latency_tier: Literal["live", "background", "scheduled"]
    cost_tier: Literal["none", "low", "medium", "high"]
    risk_tier: Literal["low", "medium", "high"]
    allowed_cognition_modes: list[Literal["deliberative", "reflex"]]
    allowed_continuation_modes: list[str]
    permission_policy: PolicyRefV1
    params_summary: dict[str, str]
    prompt_affordance: str
```

### Projection Rules

- Prompt projection includes only capability kind, availability, semantic
  affordance, user-visible consequences, and parameter summary.
- Prompt projection excludes handler IDs, adapter IDs, credentials, raw channel
  IDs, database names, cost internals, and scheduler implementation details.
- The projection must be bounded to relevant capabilities for the episode
  source and current target scope.
- Operational availability and permission checks remain deterministic even when
  a capability is prompt-visible.

### Registration And Refresh

- Registry entries are loaded from deterministic code or static configuration
  at service startup.
- Capability owners refresh availability through deterministic health,
  adapter, scheduler, and provider state before each episode projection.
- "Current affordances" means the bounded snapshot assembled for one cognitive
  episode after availability refresh and before prompt construction.
- Runtime hot-reload, remote registry mutation, or LLM-authored affordance
  registration is out of scope until a dedicated execution plan approves it.

## Contract 5: Engine Routing Layer

### Purpose

Engine routing decouples cognition nodes from a hardcoded model. It lets the
system bind L1, L2, L3, extraction, or evaluation work to engines based on
requirements.

### Shapes

```python
class EngineRequirementSpecV1(TypedDict):
    schema_version: Literal["engine_requirement_spec.v1"]
    node_id: str
    concern: Literal["L1", "L2", "L3", "extractor", "evaluator", "summarizer"]
    requires_structured_output: bool
    required_modalities: list[str]
    latency_tier: Literal["live", "interactive", "background", "batch"]
    max_context_chars: int
    minimum_capabilities: list[str]
    fallback_policy: Literal["fail_closed", "degrade_without_retry", "configured_fallback"]
```

```python
class EngineSpecV1(TypedDict):
    schema_version: Literal["engine_spec.v1"]
    engine_id: str
    provider_kind: Literal["local", "remote", "mock", "test"]
    supported_modalities: list[str]
    supports_structured_output: bool
    max_context_chars: int
    latency_tier: Literal["live", "interactive", "background", "batch"]
    cost_tier: Literal["none", "low", "medium", "high"]
    capability_labels: list[str]
    selection_priority: int
```

### Routing Algorithm

Engine binding is deterministic:

1. Filter engines by modality coverage, structured-output support, context
   capacity, and capability-label superset.
2. Filter by latency. `live` requirements accept only `live`; `interactive`
   accepts `live` or `interactive`; `background` accepts `live`,
   `interactive`, or `background`; `batch` accepts any tier.
3. Select the remaining engine with the lowest `selection_priority`.
4. Break ties by lower latency tier, then lower cost tier, then stable
   `engine_id` ordering.
5. If no engine matches, apply the requirement's `fallback_policy`; no LLM
   chooses or repairs engine routing.

### Rules

- Engine selection is deterministic configuration, not an LLM semantic choice.
- Live `/chat` call-count increases require explicit approval in an execution
  plan.
- If a node cannot bind to an engine satisfying its requirements, fail closed or
  use an explicitly configured degraded path.
- Engine routing must not leak provider internals into character prompts.

## Contract 6: Memory Layer Interface

### Purpose

Memory providers expose query and write operations through stable interfaces so
cognition does not learn collection-specific schemas.

### Query And Write Shapes

```python
class MemoryQueryV1(TypedDict):
    schema_version: Literal["memory_query.v1"]
    query_mode: Literal["semantic", "keyword", "metadata", "id", "recent"]
    semantic_query: str | None
    keyword_query: str | None
    ids: list[str]
    filters: dict[str, object]
    include_inactive: bool
    limit: int
    evidence_refs: list[EvidenceRefV1]


class MemoryResultV1(TypedDict):
    schema_version: Literal["memory_result.v1"]
    memory_kind: str
    memory_id: str
    lineage_id: str | None
    status: str
    score: float | None
    content_projection: dict[str, object]
    evidence_refs: list[EvidenceRefV1]
    source_refs: list[ActionSourceRefV1]


class MemoryLifecycleUpdateV1(TypedDict):
    schema_version: Literal["memory_lifecycle_update.v1"]
    target_status: str
    reason: str
    source_refs: list[ActionSourceRefV1]
    action_attempt_id: str | None
    updated_at: str


class MemoryDecayRequestV1(TypedDict):
    schema_version: Literal["memory_decay_request.v1"]
    decay_policy: PolicyRefV1
    candidate_scope: dict[str, object]
    reason: str
    source_refs: list[ActionSourceRefV1]
    requested_at: str
```

### Provider Shape

```python
class MemoryProviderSpecV1(TypedDict):
    schema_version: Literal["memory_provider_spec.v1"]
    memory_kind: str
    owner_module: str
    identity_fields: list[str]
    lineage_policy: PolicyRefV1
    supported_query_modes: list[str]
    supported_write_ops: list[
        Literal["insert", "supersede", "merge", "lifecycle_update", "decay"]
    ]
    lifecycle_statuses: list[str]
    evidence_ref_policy: PolicyRefV1
    prompt_projection_policy: PolicyRefV1
```

### Required Interface

```python
class MemoryProviderV1(Protocol):
    async def query(self, request: MemoryQueryV1) -> list[MemoryResultV1]: ...
    async def insert(self, document: dict[str, object]) -> dict[str, object]: ...
    async def supersede(
        self,
        source_id: str,
        replacement: dict[str, object],
    ) -> dict[str, object]: ...
    async def merge(
        self,
        source_ids: list[str],
        replacement: dict[str, object],
    ) -> dict[str, object]: ...
    async def lifecycle_update(
        self,
        source_id: str,
        update: MemoryLifecycleUpdateV1,
    ) -> dict[str, object]: ...
    async def decay(
        self,
        request: MemoryDecayRequestV1,
    ) -> list[MemoryResultV1]: ...
```

### Existing Provider Mapping

| Store | Contract Position |
|---|---|
| `user_memory_units` | User-scoped memory provider with active-commitment lifecycle states. |
| `memory` / `EvolvingMemoryDoc` | Lineage-tracked memory provider using supersede and merge. |
| `global_character_growth_traits` | Growth-trait provider using `source_reflection_run_ids` lineage. |
| `interaction_style_images` | Style-overlay provider using revision and reflection-run lineage. |
| `conversation_episode_state` | Operational progress memory, not durable identity memory. |

### Rules

- LLM prompts receive provider projections, not raw provider documents.
- Lifecycle writes must go through provider-owned APIs.
- `source_reflection_run_ids`, `evidence_refs`, and collection-native
  `source_refs` remain the existing lineage primitives. New action refs project
  onto them; they do not replace them.
- Generic database update tools are not memory-provider interfaces.

## Contract 7: Capability Surface Uniformity

### Purpose

All extensible surfaces use one mental model: a capability spec, a deterministic
handler, lifecycle hooks, prompt projection, and audit.

This does not mean one module owns all capabilities. It means every capability
declares itself in the same way while execution ownership stays local.

### Shape

```python
class CapabilitySpecV1(TypedDict):
    schema_version: Literal["capability_spec.v1"]
    capability_kind: str
    category: Literal[
        "trigger_source",
        "action",
        "retriever",
        "memory_provider",
        "engine",
        "l3_surface",
    ]
    owner_module: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    handler_id: str
    lifecycle_hooks: list[str]
    permission_policy: PolicyRefV1
    rate_limit_policy: PolicyRefV1
    audit_policy: PolicyRefV1
    prompt_projection_policy: PolicyRefV1
```

### Rules

- A capability can be prompt-visible, runtime-only, or hidden.
- Handler IDs are runtime-only and never prompt-visible.
- Permission, rate limit, idempotency, and audit are deterministic hooks.
- Dispatcher is a capability owner for adapter-facing tools. It is not the
  parent action system.
- RAG retrievers, memory providers, engines, trigger sources, action effectors,
  and L3 surfaces can all use the extension pattern without sharing one
  execution module.

## Stage 3 Registry Commitments

These entries are authoritative for the Stage 3 native execution slice once
that plan is approved.

### Trigger Sources

```text
user_message
internal_thought
self_cognition
scheduled_tick
tool_result
```

Reserved but not committed:

```text
system_event
```

### Action Capabilities

```text
speak:
  owner: l3_text
  cognition_mode: deliberative
  continuation: none

memory_lifecycle_update:
  owner: memory_lifecycle
  target_provider: user_memory_units
  cognition_mode: deliberative
  continuation: none

trigger_future_cognition:
  owner: calendar_orchestrator
  cognition_mode: deliberative
  continuation: scheduled_followup

apply_memory_lifecycle_update:
  owner: memory_lifecycle_executor
  cognition_mode: deliberative
  visibility: internal_only
  continuation: none

future_speak:
  owner: background_work
  cognition_mode: deliberative
  continuation: background_followup

accepted_task_request:
  owner: accepted_task_lifecycle
  cognition_mode: deliberative
  continuation: background_followup

accepted_coding_task_request:
  owner: accepted_task_coding
  cognition_mode: deliberative
  continuation: background_followup

accepted_task_status_check:
  owner: accepted_task_repository
  cognition_mode: deliberative
  continuation: none

background_work_request:
  owner: background_work
  cognition_mode: deliberative
  continuation: background_followup
```

All entries except internal-only `apply_memory_lifecycle_update` may be
L2d-facing when the episode's `AffordanceSpecV1` marks them available or
degraded. `send_message` is not a cognition-visible capability; delayed contact
is expressed as future cognition so the character re-decides at execution
time.

### Deferred Action Capabilities

```text
web_research
fetch_url
note_open_loop
close_open_loop
emit_image
external_message
```

Deferred entries are not runtime permissions. They are reserved names requiring
their own execution plans.

## Action Spec And Dispatcher Boundary

Action spec must not be merged into dispatcher.

The approved boundary is:

```text
L2d semantic action request
-> deterministic ActionSpecV1 materialization
-> ActionSpecEvaluator
-> execution owner / L3 surface handler
-> ActionResultV1 + SurfaceOutputV1
-> EpisodeTraceV2
```

For `speak`:

```text
ActionSpecV1(kind="speak")
-> L3-text handler
-> SurfaceOutputV1(surface_kind="text")
-> live response delivery boundary
-> ActionResultV1(kind="speak")
```

For `memory_lifecycle_update`:

```text
ActionSpecV1(kind="memory_lifecycle_update")
-> memory lifecycle handler
-> memory provider lifecycle_update(...)
-> ActionResultV1
```

For `trigger_future_cognition`:

```text
ActionSpecV1(kind="trigger_future_cognition")
-> orchestrator/scheduler handler
-> scheduled typed episode or continuation record
-> ActionResultV1
```

Scheduler/adapters own adapter-facing delivery mechanics. L3 surface
handlers own surface generation. Memory owners own memory lifecycle writes.
Orchestration owns continuation episodes and future cognition scheduling. The
consolidator consumes the resulting episode trace and does not execute any of
these actions.

The scheduled typed episode created by `trigger_future_cognition` must carry
one `ContinuationObjectiveV1` as its only LLM-facing handoff text. It may also
carry deterministic scheduling, scope, source-ref, depth, and audit metadata,
but those fields are not LLM handoff content. The previous concept of a
`context_summary` is rejected because it encourages lossy compression instead
of preserving the exact future thinking objective.

## Promise Retirement Policy

Expired-promise suppression is a character decision, not deterministic cleanup.

Allowed path:

```text
cognition selects memory_lifecycle_update
-> source refs and reason are validated
-> target memory owner validates transition
-> lifecycle write is audited
-> future retrieval excludes inactive status through provider-owned filters
```

Forbidden paths:

- deleting promises because they are old;
- hiding active promises by deterministic due-age cutoff;
- keyword-classifying promise text into abandoned state;
- directly updating MongoDB outside the memory owner;
- treating a generated internal thought as user evidence.

## Continuation Policy

Tools and action handlers may cause a later cognition cycle only through the
action continuation contract or an orchestrator-owned
`trigger_future_cognition` action result.

Allowed flow:

```text
ActionSpecV1 -> validated handler execution -> ActionResultV1
-> continuation contract or scheduled cognition request
-> orchestrator enqueues typed episode -> shared cognition path
```

Forbidden flow:

```text
tool handler -> cognition node direct call
tool handler -> final dialog write
action handler -> consolidator-only private note that bypasses episode trace
```

Continuation modes:

```text
none
immediate_followup
scheduled_followup
background_followup
```

Each capability declares allowed continuation modes in its registry entry.
Parallel actions may each return `mode="none"` or their own continuation data.
Dependent action chains must not be represented by an implicit in-handler loop;
they return a continuation so the next cognition cycle has a typed trigger,
evidence, rate-limit state, and audit trail.

## Episode Trace And Consolidation Boundary

The consolidator must not remain tied to a single `final_dialog` string. A turn
can produce no visible reply, a text reply, an image prompt, a private memory
update, a scheduled cognition request, or several independent actions. The
durable memory/state update path therefore consumes an `EpisodeTraceV2`.

Required consolidation inputs:

```text
CognitiveEpisodeV1
L1/L2/L3 prompt-safe residue
selected ActionSpecV1 records
ActionResultV1 records, including rejected and scheduled actions
SurfaceOutputV1 records, including private or non-delivered outputs
delivery/audit refs
```

Rules:

- `final_dialog` is represented as a text `SurfaceOutputV1`, not as the
  consolidation gate.
- The live service uses the settled `EpisodeTraceV2` as the
  consolidatable-output gate; graph-state fallbacks are outside the canonical
  boundary.
- The consolidator receives prompt-safe action projections, not raw action
  params, handler IDs, adapter IDs, credentials, or collection internals.
- The consolidator may persist memory, relationship state, progress, mood, and
  lifecycle updates through existing owners. It must not select new actions,
  execute selected actions, or call dispatcher/scheduler directly.
- Long-running action completion enters consolidation through a later typed
  `tool_result`, `scheduled_tick`, or source-specific trigger episode.

## Approval Rules For Derived Plans

Execution plans derived from this document must:

- state which contracts they implement or extend;
- link this reference and any lower-level architecture references;
- keep design rationale out of the execution plan unless needed for a gate;
- name exact change surfaces, verification commands, and owner boundaries;
- run independent plan review before approval;
- avoid adding runtime capability names not present or reserved in this
  reference unless this reference is updated first.

Plans may be approved once:

- this document is approved;
- the plan is reconciled against the relevant contract sections;
- blockers recorded in the plan are resolved;
- independent plan review finds no approval blocker.

## Relationship To The Stage 3 Adoption Plan

The active draft
`development_plans/archive/completed/short_term/cognition_core_v2_stage_3_system_adoption_plan.md`
and its mandatory companions define the next native runtime slice of contracts
1, 2, 3, 4, 6, and 7:

- contract 1: self-cognition and future cognition requests enter as typed
  trigger-source episodes;
- contract 2: L2d action requests, L3 surface outputs, action results, and
  episode-trace consolidation projections;
- contract 3: `ActionSpecV1`;
- contract 4: prompt-safe runtime affordance projection for the complete
  nine-kind Stage 3 action roster;
- contract 6: `user_memory_units.active_commitment` lifecycle update through
  memory owner;
- contract 7: capability spec plus handler and audit pattern.

That plan becomes executable only after approval and a separate implementation
command. Its stages keep this reference authoritative and update downstream
architecture documents when the contract surface changes.

## Independent Reference Review Resolution

The 2026-05-16 independent review identified three approval blockers:

- `reflection_promoted` conflicted with
  `self_cognition_reasoning_basis.md`;
- the trigger-source lists disagreed about `system_event`;
- several referenced registry payloads were missing.

Resolution in this revision:

- `reflection_promoted` is removed from the committed trigger-source registry
  and defined as projected context/evidence only;
- `system_event` is reserved but not committed for the first registry slice;
- `ActionSourceRefV1`, `ActionTargetV1`, `MemoryQueryV1`, `MemoryResultV1`,
  `MemoryLifecycleUpdateV1`, `EvidenceRefV1`, `PerceptV1`, and
  `MemoryDecayRequestV1` are defined;
- residue payloads now carry `schema_version`;
- policy fields use `PolicyRefV1` registry-key format;
- engine routing, affordance refresh, timestamp format, memory `decay`, and
  perception/streaming disposition are specified at registry level;
- action capability naming is reconciled with the action-spec architecture as
  action-category `CapabilitySpecV1` entries.

## Review Checklist

Before approving this reference, review:

- every contract has an owner and schema version;
- trigger source, residue, action, affordance, engine, memory, and capability
  shapes do not duplicate one another;
- reflection-derived material is context/evidence, not an initial trigger;
- dispatcher remains an execution owner, not the action-spec parent;
- promise retirement remains LLM-selected and auditable;
- continuation does not let tools call cognition directly;
- local-LLM prompt projection remains bounded and semantic;
- deferred capabilities are clearly non-executable.

## Approved State

This reference is approved as of 2026-05-16 after independent review blockers
were resolved. It satisfies the contracts-reference prerequisite for derived
execution plans, but it does not authorize implementation by itself.
