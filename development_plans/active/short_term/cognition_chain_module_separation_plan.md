# cognition chain module separation plan

## Summary

- Goal: Move the existing L1/L2/L2d/L3 cognition chain behind a reusable
  `cognition_chain_core` module boundary, leaving
  `persona_supervisor2_cognition` and the selected L3 surface wrapper as thin
  Kazusa graph connectors.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, and `test-style-and-execution`.
- Overall cutover strategy: compatible module separation. Preserve current
  runtime behavior, graph entrypoints, prompt semantics, and action-spec
  outcomes while moving ownership behind a sealed core ICD.
- Highest-risk areas: hidden `GlobalPersonaState` coupling, L2d action-spec
  materialization, resolver request threading, L3 surface directive handoff,
  prompt import churn, live LLM regression, and tests that monkeypatch current
  node module internals.
- Acceptance criteria: callers use stable core entrypoints and contracts
  instead of importing L1/L2/L2d/L3 internals from `nodes`; the persona graph
  still produces the same observable cognition/action/surface state; no
  parallel appraisal streams, conflict integrator, speech monitor, secret
  ledger, or claim ledger are implemented in this stage.

## Context

The reference architecture at
`development_plans/reference/designs/kazusa_parallel_cognition_architecture.md`
defines Phase 1 as extraction of a reusable cognition kernel before adding
parallel appraisal streams. The current implementation still keeps L1, L2,
L2d, and L3 cognition files under `src/kazusa_ai_chatbot/nodes/` and exposes
internal stage functions directly to tests and callers.

Current key files:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` builds the
  L1 -> L2a/L2b -> L2c1/L2c2 -> L2d graph and exposes
  `call_cognition_subgraph(GlobalPersonaState)`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` builds the
  selected L3 text-surface directive graph and exposes
  `call_l3_text_surface_handler(GlobalPersonaState)`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`,
  `persona_supervisor2_cognition_l2.py`,
  `persona_supervisor2_cognition_l2c2.py`,
  `persona_supervisor2_cognition_l2d.py`, and
  `persona_supervisor2_cognition_l3.py` contain the layer handlers and prompts.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` defines
  `CognitionState`, which imports Kazusa graph, DB, message-envelope,
  resolver, and action-spec shapes.

The separation stage is necessary because future parallel self-competition
requires independent cognition candidates to run without mutating Kazusa graph
state, persisting pending rows, queuing background work, or materializing
`ActionSpecV1` before a winner is selected.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing cognition graph wiring,
  prompt payloads, model calls, resolver handoff, L2d routing, L3 surface
  planning, or context budgets.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files or tests containing CJK string
  literals.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.
- Plan status is not production-code authorization. Production-code edits
  require this plan to be approved or in progress and a direct user instruction
  to execute it.
- Preserve the live response path as bounded and inspectable.
- LLM stages own semantic judgment. Deterministic code owns validation,
  persistence, limits, scheduling, adapter delivery, and audit.
- Do not expose L1, L2, L2d, or L3 internal functions as the caller-facing
  contract of the new core.
- Do not move platform transport, database mechanics, queueing, adapter
  formatting, action execution, scheduler execution, or consolidation writes
  into `cognition_chain_core`.
- Do not add deterministic keyword routing or post-processing over raw user
  input to compensate for a core contract gap.
- Do not change prompt semantics while moving files. Prompt edits are allowed
  only when required to remove project-internal module naming from the
  reusable core boundary.
- Do not increase live-response LLM call count in this separation stage.
- Live LLM tests must run one case at a time with output inspected.

## Must Do

- Create `src/kazusa_ai_chatbot/cognition_chain_core/` as the target module for
  the reusable cognition chain.
- Define an ICD-styled README for the new module.
- Define `CognitionChainInputV1`, `CognitionChainOutputV1`, and related
  semantic contracts inside the new module.
- Move the existing L1/L2/L2c2/L2d/L3 cognition logic behind core entrypoints
  without changing current prompt behavior or model route settings. Do not keep
  wrapper modules as a caller-facing compatibility layer.
- Keep `persona_supervisor2_cognition.py` as a thin Kazusa connector that maps
  `GlobalPersonaState` into `CognitionChainInputV1` and maps
  `CognitionChainOutputV1` back into graph state updates.
- Keep `persona_supervisor2_l3_surface.py` as a thin selected-surface connector
  that maps Kazusa graph state into the core's selected text-surface planning
  contract and maps the result back to `action_directives`.
- Split L2d responsibility so the core returns semantic action requests and
  resolver requests, while Kazusa connector code remains responsible for
  `ActionSpecV1` materialization.
- Preserve current resolver loop integration through
  `call_cognition_resolver_loop(...)` and its injected cognition callable.
- Update tests to use the new core public contracts where they are testing
  cognition-chain behavior, while preserving graph-connector tests for Kazusa
  integration.
- Run static import greps and focused tests listed in `Verification`.

## Deferred

- Do not implement parallel appraisal streams.
- Do not implement a self-competition runner, evaluator, candidate merger, or
  conflict integrator.
- Do not implement a speech monitor, late inhibition layer, secret commitment
  store, public claim ledger, or private truth ledger.
- Do not change RAG2 retrieval, resolver capability execution, HIL/approval
  persistence, action handlers, background-work queueing, scheduler behavior,
  adapter delivery, dialog generator/evaluator behavior, or consolidation
  write policy.
- Do not add new runtime feature flags, environment variables, model routes,
  prompt examples, repair loops, retry loops, or compatibility fallback paths.
- Do not preserve direct caller imports of moved L1/L2/L2d/L3 internals. Alias
  modules are forbidden in this stage. Migrate tests and production imports to
  the public core contracts or connector entrypoints.

## Cutover Policy

Overall strategy: compatible module separation.

| Area | Policy | Instruction |
|---|---|---|
| Persona graph entrypoints | compatible | Preserve `call_cognition_subgraph(...)` and `call_l3_text_surface_handler(...)` as Kazusa connector entrypoints. |
| Core public contract | bigbang | New callers use `cognition_chain_core` contracts and public entrypoints only. No public L1/L2/L2d/L3 imports. |
| Prompt behavior | compatible | Preserve current prompt semantics, output fields, model routes, and LLM call count. |
| L2d materialization | migration | Move route-only action selection into `cognition_chain_core.action_selection`; create `nodes/persona_supervisor2_cognition_actions.py` as the only Kazusa materializer from `SemanticActionRequestV1` to `ActionSpecV1`. |
| Tests | migration | Move cognition-core tests to the new public contracts; retain connector tests for graph state mapping. |
| Database and persistence | compatible | No schema, collection, index, write-shape, or migration changes. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- If an area is `bigbang`, delete or rewrite legacy references instead of
  preserving them.
- If an area is `migration`, follow the exact migration and cleanup gates in
  this plan.
- If an area is `compatible`, preserve only the compatibility surfaces
  explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

The completed separation stage has this shape:

```text
src/kazusa_ai_chatbot/
  cognition_chain_core/
    __init__.py
    README.md
    contracts.py
    chain.py
    surface.py
    output_contracts.py
    prompt_selection.py
    action_selection.py
    stages/
      __init__.py
      l1.py
      l2.py
      l2c2.py
      l2d.py
      l3.py
  nodes/
    persona_supervisor2_cognition.py      # Kazusa graph connector
    persona_supervisor2_l3_surface.py     # selected-surface connector
    persona_supervisor2_schema.py         # Kazusa graph state only
```

`cognition_chain_core` owns the reusable chain and hides internal stage
functions. Kazusa graph code owns graph-state projection, resolver integration,
action-spec materialization, action execution, L3 dialog invocation, and
post-turn consolidation handoff.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Input contract name | `CognitionChainInputV1` | User-selected name; explicit versioning supports future parallel candidates. |
| Output contract name | `CognitionChainOutputV1` | Pairs with input and represents one sealed candidate cognition result. |
| Core module name | `cognition_chain_core` | Describes the reusable chain without tying it to `nodes` or persona graph internals. |
| Connector module | Keep `persona_supervisor2_cognition.py` | Preserves existing graph integration while shrinking its ownership. |
| L3 connector | Keep `persona_supervisor2_l3_surface.py` | L3 text-surface selection still depends on Kazusa action specs and graph state. |
| L2d output | Core returns `SemanticActionRequestV1` rows and resolver request rows | Parallel self-competition requires candidates to bid without side effects. |
| Action router ownership | Move route-only action selection prompt, message construction, and output normalization into `cognition_chain_core.action_selection` | L2d semantic route selection is part of the reusable cognition chain. |
| ActionSpec ownership | Kazusa connector materializes `ActionSpecV1` from `SemanticActionRequestV1` | Action specs bind project-specific handlers, scopes, continuation, and validation. |
| Resolver ownership | Resolver loop stays outside core | Current resolver loop is already injected-callable friendly and owns recurrence state. |
| Prompt behavior | Preserve current prompt text and generated field semantics | The stage is module separation, not prompt redesign. |

## Contracts And Data Shapes

### Public Core Entrypoints

```python
async def run_cognition_chain(
    input_payload: CognitionChainInputV1,
    services: CognitionChainServices,
) -> CognitionChainOutputV1:
    ...

async def run_text_surface_planning(
    input_payload: CognitionTextSurfaceInputV1,
    services: CognitionChainServices,
) -> CognitionTextSurfaceOutputV1:
    ...
```

Callers must not import or invoke private stage handlers such as L1, L2, L2d,
or L3 functions. Internal stage modules may exist under
`cognition_chain_core.stages`, but they are not public integration surfaces.

### Contract Types

The implementation must define public contracts in
`cognition_chain_core.contracts` using `TypedDict`, `Literal`,
`NotRequired`, and `Protocol` where shown below. The core input is a
prompt-safe semantic envelope. It must not contain raw adapter payloads, raw
MongoDB rows, collection names, handler ids, credentials, delivery targets,
queue futures, leases, scheduler internals, `ActionSpecV1`, or graph state
types.

```python
class TextEvidenceV1(TypedDict):
    source: Literal[
        "rag",
        "conversation",
        "memory",
        "profile",
        "reflection",
        "resolver",
        "background",
        "system",
    ]
    title: str
    content: str
    relevance: NotRequired[str]
    recency: NotRequired[str]


class ModelVisiblePerceptV1(TypedDict):
    percept_id: str
    input_source: Literal[
        "chat_message",
        "self_cognition",
        "reflection",
        "scheduled_followup",
        "background_result",
    ]
    content: str
    metadata_summary: list[str]


class CognitionEpisodePromptV1(TypedDict):
    episode_id: str
    trigger_source: str
    input_sources: list[str]
    output_mode: Literal["live_response", "background_cognition", "dry_run"]
    model_visible_percepts: list[ModelVisiblePerceptV1]
    target_scope_summary: str
    origin_summary: str


class CharacterPromptV1(TypedDict):
    character_global_id: str
    name: str
    description: str
    gender: str
    age: str
    birthday: str
    backstory: str
    personality_brief: str
    boundary_profile: str
    linguistic_texture_profile: str
    mood: str
    global_vibe: str


class PromptSafeUserMemoryContextV1(TypedDict):
    durable_profile_summary: str
    relationship_summary: str
    recent_commitments_summary: str
    known_preferences_summary: str


class UserPromptV1(TypedDict):
    global_user_id: str
    display_name: str
    affinity: str
    affinity_level: str
    last_relationship_insight: str
    memory_context: PromptSafeUserMemoryContextV1


class ReferentPromptV1(TypedDict):
    label: str
    resolved_summary: str
    confidence: Literal["low", "medium", "high"]


class MediaObservationPromptV1(TypedDict):
    modality: Literal["image", "audio", "file", "link", "unknown"]
    observation: str
    source_summary: str


class CurrentEventPromptV1(TypedDict):
    user_input: str
    decontextualized_input: str
    indirect_speech_context: str
    referents: list[ReferentPromptV1]
    media_observations: list[MediaObservationPromptV1]
    reply_context_summary: str
    prompt_message_context_summary: str


class ScenePromptV1(TypedDict):
    platform: str
    channel_type: str
    channel_topic: str
    local_time_context: str
    storage_timestamp_utc: str
    interaction_history_recent: list[str]


class ConversationContextPromptV1(TypedDict):
    conversation_progress: str
    promoted_reflection_context: str
    internal_monologue_residue_context: str
    previous_action_summary: str


class EvidencePromptV1(TypedDict):
    rag_answer: str
    current_user_rag_bundle: str
    memory_evidence: list[TextEvidenceV1]
    conversation_evidence: list[TextEvidenceV1]
    external_evidence: list[TextEvidenceV1]
    recall_evidence: list[TextEvidenceV1]
    supervisor_trace: list[str]


class ResolverPromptV1(TypedDict):
    resolver_context: str
    pending_resume: str
    goal_progress: str
    recent_observations: list[str]
    max_projected_observations: int


class ActionAffordanceV1(TypedDict):
    capability: Literal[
        "speak",
        "memory_lifecycle_update",
        "trigger_future_cognition",
        "background_work_request",
    ]
    available: bool
    visibility: Literal["public", "private", "internal"]
    semantic_input_summary: str
    output_kind: Literal["semantic_action_request"]


class RuntimeContextV1(TypedDict):
    language_policy: str
    visual_directives_enabled: bool
    max_action_requests: int
    max_resolver_requests: int
```

### `CognitionChainInputV1`

```python
class CognitionChainInputV1(TypedDict):
    schema_version: Literal["cognition_chain_input.v1"]
    episode: CognitionEpisodePromptV1
    character: CharacterPromptV1
    current_user: UserPromptV1
    current_event: CurrentEventPromptV1
    scene: ScenePromptV1
    conversation_context: ConversationContextPromptV1
    evidence: EvidencePromptV1
    resolver: ResolverPromptV1
    available_actions: list[ActionAffordanceV1]
    runtime_context: RuntimeContextV1
```

The Kazusa connector maps existing `GlobalPersonaState` fields into this
contract. The core reads only this contract.

Mapping rules:

| Source state area | `CognitionChainInputV1` destination | Notes |
|---|---|---|
| cognitive episode, source case, output mode | `episode` | Convert to prompt-visible source summaries; no adapter event body. |
| character config and runtime mood | `character` | Project only text fields already visible to cognition prompts. |
| current user profile, affinity, scoped user memory | `current_user` | Convert database-backed data to summaries before core entry. |
| current user input, decontextualized input, media/reply context | `current_event` | Include prompt-safe media observations, not media cache rows. |
| platform/channel/time/recent turns | `scene` | Include summaries only; no delivery target ids. |
| conversation progress, promoted reflection, prior residue | `conversation_context` | Reflection must be promoted/gated before entry. |
| RAG, recall, resolver, and supervisor evidence | `evidence` | RAG remains evidence, not persona or final stance. |
| resolver recurrence packet | `resolver` | Core receives projected resolver context only. |
| allowed semantic actions | `available_actions` | Connector owns capability availability. |
| route caps and prompt policy | `runtime_context` | No environment variables or model names. |

Input validation raises `CognitionChainContractError` before the first LLM call
when a required field is missing, an enum value is unknown, cardinality exceeds
`runtime_context` caps, or forbidden raw project fields are present.

### `CognitionChainOutputV1`

The output is one sealed cognition result. It contains semantic decisions and
prompt-safe traces, not executable project actions.

```python
class CognitionResidueV1(TypedDict):
    emotional_appraisal: str
    interaction_subtext: str
    internal_monologue: str
    logical_stance: str
    character_intent: str
    judgment_note: str
    social_distance: str
    emotional_intensity: str
    vibe_check: str
    relational_dynamic: str


class SemanticActionRequestV1(TypedDict):
    capability: Literal[
        "speak",
        "memory_lifecycle_update",
        "trigger_future_cognition",
        "background_work_request",
    ]
    decision: Literal[
        "visible_reply",
        "private_finalization",
        "delayed_followup",
        "background_task",
        "think_only",
        "silent",
    ]
    detail: str
    reason: str


class ResolverCapabilityRequestV1(TypedDict):
    capability_kind: Literal[
        "web_search",
        "memory_search",
        "conversation_search",
        "url_read",
        "background_result_lookup",
    ]
    objective: str
    reason: str
    priority: Literal["low", "normal", "high"]


class ResolverPendingResolutionV1(TypedDict):
    pending: bool
    unresolved_question: str
    resume_hint: str


class ResolverGoalProgressV1(TypedDict):
    goal: str
    status: Literal["not_started", "in_progress", "satisfied", "blocked"]
    observation_summary: str


class CognitionChainTraceV1(TypedDict):
    stage_order: list[str]
    selected_actions_summary: str
    resolver_summary: str
    warnings: list[str]


class CognitionChainOutputV1(TypedDict):
    schema_version: Literal["cognition_chain_output.v1"]
    cognition_residue: CognitionResidueV1
    semantic_action_requests: list[SemanticActionRequestV1]
    resolver_capability_requests: list[ResolverCapabilityRequestV1]
    resolver_pending_resolution: NotRequired[ResolverPendingResolutionV1]
    resolver_goal_progress: NotRequired[ResolverGoalProgressV1]
    chain_trace: CognitionChainTraceV1
```

`semantic_action_requests` has cardinality `0..runtime_context.max_action_requests`
with a default cap of `3`. It must not contain `ActionSpecV1`, handler ids, raw
target ids, job ids, adapter ids, worker-local task parameters, or final
dialogue. `resolver_capability_requests` has cardinality
`0..runtime_context.max_resolver_requests`. Invalid optional action or resolver
rows are dropped by existing normalization behavior with a warning in
`chain_trace.warnings`; missing required cognition residue fields fail fast with
`CognitionChainContractError`.

Mapping rules:

| `CognitionChainOutputV1` field | Graph-state destination | Owner |
|---|---|---|
| `cognition_residue` | current cognition fields and next-turn residue | `persona_supervisor2_cognition.py` connector |
| `semantic_action_requests` | `ActionSpecV1` rows after materialization | `nodes/persona_supervisor2_cognition_actions.py` |
| `resolver_capability_requests` | resolver loop capability request packet | cognition resolver integration |
| `resolver_pending_resolution` | resolver pending-resolution state | cognition resolver integration |
| `resolver_goal_progress` | resolver goal-progress state | cognition resolver integration |
| `chain_trace` | debug/log trace and consolidation context | connector and event logging |

### Action Selection Boundary

Route-only L2d action selection moves into
`cognition_chain_core.action_selection` and returns
`SemanticActionRequestV1`. The old top-level `kazusa_ai_chatbot.action_router`
module must not remain a runtime dependency of the core or connector after this
stage. Its route prompt, output-normalization behavior, and tests are migrated
to core action-selection ownership; any obsolete package files are deleted once
static greps prove no production import remains.

`nodes/persona_supervisor2_cognition_actions.py` is the only connector-owned
materializer. It converts each `SemanticActionRequestV1` into project-local
`ActionSpecV1` rows using existing action-spec validation and graph context.
Allowed capabilities in this stage are exactly `speak`,
`memory_lifecycle_update`, `trigger_future_cognition`, and
`background_work_request`.

### `CognitionTextSurfaceInputV1`

Selected L3 text-surface planning is separate from the main chain because it
runs only after a selected `speak` action survives graph routing. The connector
derives this input from the selected `ActionSpecV1`, action execution results,
and graph state; the core never sees `ActionSpecV1` itself.

```python
class SelectedTextSurfaceIntentV1(TypedDict):
    decision: Literal["visible_reply"]
    original_goal: str
    goal_progress_summary: str
    observation_summary: str
    speak_intent: str
    detail: str
    tone: str
    reason: str


class PreSurfaceActionResultPromptV1(TypedDict):
    action_kind: Literal[
        "background_work_request",
        "background_artifact_request",
        "memory_lifecycle_update",
    ]
    status: Literal["completed", "queued", "failed", "skipped"]
    queue_state: str
    task_summary: str
    objective_summary: str
    acknowledgement_constraint: str


class MemoryLifecycleContextPromptV1(TypedDict):
    active_commitment_aliases: list[str]
    pending_memory_updates_summary: str
    recent_memory_resolution_summary: str


class CognitionTextSurfaceInputV1(TypedDict):
    schema_version: Literal["cognition_text_surface_input.v1"]
    chain_input: CognitionChainInputV1
    cognition_residue: CognitionResidueV1
    selected_text_surface_intent: SelectedTextSurfaceIntentV1
    pre_surface_action_results: list[PreSurfaceActionResultPromptV1]
    memory_lifecycle_context: MemoryLifecycleContextPromptV1
```

`selected_text_surface_intent.speak_intent` is prompt-safe and must not contain
handler ids, action ids, adapter ids, or final dialogue. Background-work
acknowledgements enter only through `pre_surface_action_results`, so L3 can
acknowledge queued or completed work without seeing worker internals.

### `CognitionTextSurfaceOutputV1`

```python
class ContextualDirectiveV1(TypedDict):
    response_goal: str
    conversation_anchor: str
    must_address: list[str]
    avoid: list[str]


class LinguisticDirectiveV1(TypedDict):
    tone: str
    register: str
    rhythm: str
    phrasing_constraints: list[str]


class VisualDirectiveV1(TypedDict):
    enabled: bool
    self_image_guidance: str
    composition_guidance: str
    forbidden_elements: list[str]


class ActionDirectivesV1(TypedDict):
    contextual_directives: ContextualDirectiveV1
    linguistic_directives: LinguisticDirectiveV1
    visual_directives: VisualDirectiveV1


class CognitionTextSurfaceOutputV1(TypedDict):
    schema_version: Literal["cognition_text_surface_output.v1"]
    action_directives: ActionDirectivesV1
```

`action_directives` must preserve the current dialog-facing key names while
making each nested field explicit for connector tests.

### `CognitionChainServices`

The service bundle supplies process-local dependencies without importing
Kazusa graph state into core contracts. It is implemented as
`@dataclass(frozen=True)`. LLM instances are constructed by the Kazusa connector
from existing model-route configuration and injected into the core. The core
must not read environment variables, build model clients, or choose model
names.

```python
class AsyncChatModel(Protocol):
    async def ainvoke(self, messages: Sequence[BaseMessage]) -> BaseMessage:
        ...


class JsonParser(Protocol):
    def __call__(self, content: str) -> Mapping[str, Any] | list[Any]:
        ...


class CognitionLogger(Protocol):
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        ...
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        ...
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        ...
    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        ...


@dataclass(frozen=True)
class CognitionChainServices:
    cognition_llm: AsyncChatModel
    boundary_core_llm: AsyncChatModel
    action_selection_llm: AsyncChatModel
    style_llm: AsyncChatModel
    content_plan_llm: AsyncChatModel
    preference_llm: AsyncChatModel
    visual_llm: AsyncChatModel
    parse_json: JsonParser
    logger: CognitionLogger
```

## LLM Call And Context Budget

Separation preserves route selection, prompt ownership, and call count.

| Stage | Before separation | After separation | Model route | Prompt owner | Input source | Calls per pass |
|---|---|---|---|---|---|---|
| L1 subconscious | `persona_supervisor2_cognition_l1.py` | `cognition_chain_core.stages.l1` | `COGNITION_LLM` | moved unchanged to core | `CognitionChainInputV1` projected fields | 1 |
| L2a consciousness | `persona_supervisor2_cognition_l2.py` | `cognition_chain_core.stages.l2a` | `COGNITION_LLM` | moved unchanged to core | L1 output plus chain input | 1 |
| L2b boundary core | `persona_supervisor2_cognition_l2.py` | `cognition_chain_core.stages.l2b` | `BOUNDARY_CORE_LLM` | moved unchanged to core | L2a output plus boundary context | 1 |
| L2c1 judgment core | `persona_supervisor2_cognition_l2.py` | `cognition_chain_core.stages.l2c1` | `COGNITION_LLM` | moved unchanged to core | L2a/L2b outputs plus evidence | 1 |
| L2c2 social appraisal | `persona_supervisor2_cognition_l2c2.py` | `cognition_chain_core.stages.l2c2` | `COGNITION_LLM` | moved unchanged to core | L2 outputs plus relationship context | 1 |
| L2d action selection | `persona_supervisor2_cognition_l2d.py` and `action_router` | `cognition_chain_core.action_selection` | `COGNITION_LLM` | moved unchanged to core | cognition residue plus action affordances | 1 |
| L3 interaction-style load | `persona_supervisor2_cognition_l3.py` via `persona_supervisor2_l3_surface.py` graph | `cognition_chain_core.stages.l3_interaction_style` | none | moved unchanged to core | text-surface input | 0 |
| L3 style agent | `persona_supervisor2_cognition_l3.py` via `persona_supervisor2_l3_surface.py` graph | `cognition_chain_core.surface` | `COGNITION_LLM` | moved unchanged to core | text-surface input and L3 context | 1 |
| L3 content-plan agent | `persona_supervisor2_cognition_l3.py` via `persona_supervisor2_l3_surface.py` graph | `cognition_chain_core.surface` | `COGNITION_LLM` | moved unchanged to core | selected speak intent and residue | 1 |
| L3 preference adapter | `persona_supervisor2_cognition_l3.py` via `persona_supervisor2_l3_surface.py` graph | `cognition_chain_core.surface` | `COGNITION_LLM` | moved unchanged to core | user/context preference projection | 1 |
| L3 visual agent | `persona_supervisor2_cognition_l3.py` via `persona_supervisor2_l3_surface.py` graph | `cognition_chain_core.surface` | `COGNITION_LLM` | moved unchanged to core | visual-directive projection | 1 when enabled |
| L3 collector | `persona_supervisor2_cognition_l3.py` via `persona_supervisor2_l3_surface.py` graph | `cognition_chain_core.surface` | none | moved unchanged to core | L3 sub-agent outputs | 0 |

Resolver recurrence may repeat the main cognition chain according to the
existing resolver loop. No new response-path LLM call is added. No prompt
receives raw adapter, database, action-spec, queue, or scheduler internals. The
50k-token default context cap remains unchanged. Existing truncation and
projection helpers remain the effective budget controls.

Any proposed change that increases call count, changes model route selection,
or expands prompt-visible context beyond the current projection must stop and
receive explicit user approval through a new or amended plan.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/cognition_chain_core/README.md`: module ICD.
- `src/kazusa_ai_chatbot/cognition_chain_core/__init__.py`: public exports.
- `src/kazusa_ai_chatbot/cognition_chain_core/contracts.py`: public input,
  output, surface, and service contracts.
- `src/kazusa_ai_chatbot/cognition_chain_core/chain.py`: public main-chain
  entrypoint and internal graph wiring.
- `src/kazusa_ai_chatbot/cognition_chain_core/surface.py`: public selected
  text-surface planning entrypoint.
- `src/kazusa_ai_chatbot/cognition_chain_core/output_contracts.py`: moved
  cognition output validation.
- `src/kazusa_ai_chatbot/cognition_chain_core/prompt_selection.py`: moved
  prompt variant selection.
- `src/kazusa_ai_chatbot/cognition_chain_core/action_selection.py`: moved L2d
  route-only action selection and `SemanticActionRequestV1` normalization.
- `src/kazusa_ai_chatbot/cognition_chain_core/stages/`: private stage modules.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`:
  Kazusa-only `SemanticActionRequestV1` to `ActionSpecV1` materializer.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: reduce to
  Kazusa connector and `GlobalPersonaState` mapping.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: reduce to
  selected text-surface connector and graph-state mapping.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`,
  `persona_supervisor2_cognition_l2.py`,
  `persona_supervisor2_cognition_l2c2.py`,
  `persona_supervisor2_cognition_l2d.py`, and
  `persona_supervisor2_cognition_l3.py`: move implementation into
  `cognition_chain_core.stages` or `cognition_chain_core.action_selection`;
  delete direct caller imports before completion.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
  and `persona_supervisor2_cognition_prompt_selection.py`: move ownership into
  `cognition_chain_core`.
- `src/kazusa_ai_chatbot/action_router/`: migrate route prompt,
  normalization, and tests into `cognition_chain_core.action_selection`; remove
  runtime imports of the top-level package.
- Focused tests under `tests/` that import cognition internals: update them to
  core public contracts when they test core behavior, or connector entrypoints
  when they test Kazusa graph mapping.
- `development_plans/README.md`: keep the plan registered while active and
  move it to completed history after execution sign-off.

### Keep

- `src/kazusa_ai_chatbot/cognition_resolver/`: recurrence controller remains
  outside the core in this stage.
- `src/kazusa_ai_chatbot/action_spec/`: action-spec contracts and validation
  stay project-owned.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: final text generation remains
  outside this separation plan.
- `src/kazusa_ai_chatbot/consolidation/`, `rag/`, `db/`, `dispatcher/`,
  `calendar_scheduler/`, `self_cognition/`, and adapters: no behavior changes.

## Overdesign Guardrail

- Actual problem: the cognition chain is not reusable or safely parallelizable
  because current L1/L2/L2d/L3 internals live under `nodes` and consume
  project-shaped graph state directly.
- Minimal change: introduce a sealed `cognition_chain_core` ICD, move existing
  stage implementation behind it, and leave Kazusa-specific graph and action
  materialization in connectors.
- Ownership boundaries: core owns semantic cognition; Kazusa connector owns
  graph-state mapping, action-spec materialization, resolver-loop integration,
  execution, persistence, scheduling, delivery, and consolidation handoff.
- Rejected complexity: no parallel streams, competition runner, conflict
  integrator, speech monitor, secret ledger, claim ledger, extra model calls,
  feature flags, fallback paths, compatibility shims, or prompt redesigns.
- Evidence threshold: add parallel appraisal or competition only after this
  separation passes focused core contract tests, connector integration tests,
  static import greps, and one-at-a-time live LLM smoke checks.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside `cognition_chain_core`,
  `nodes/persona_supervisor2_cognition.py`,
  `nodes/persona_supervisor2_l3_surface.py`, the moved cognition files, and
  focused tests as high-scrutiny changes requiring explicit plan justification.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- The responsible agent must search for existing equivalent behavior before
  implementing helpers. Move existing logic; do not reimplement it from memory.
- If the plan and code disagree, preserve the separation-stage intent and
  report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Parent adds focused contract tests:
   `tests/test_cognition_chain_core_contracts.py::test_cognition_chain_input_rejects_raw_graph_and_adapter_fields`,
   `::test_cognition_chain_output_rejects_action_specs`,
   `::test_text_surface_input_accepts_prompt_safe_projection`, and
   `tests/test_cognition_chain_core_action_selection.py::test_semantic_action_request_contract_matches_current_l2d_routes`.
   Expected before implementation: these fail with missing
   `kazusa_ai_chatbot.cognition_chain_core` module or missing exported contract
   symbols.
2. Parent adds connector baseline tests:
   `tests/test_cognition_chain_connector_mapping.py::test_persona_connector_maps_global_state_to_chain_input`,
   `::test_persona_connector_materializes_semantic_actions_after_core_output`,
   and
   `::test_l3_surface_connector_projects_selected_speak_without_action_spec_leak`.
   Expected before implementation: existing connector behavior baseline passes
   with patched LLM/stage calls or fails only at the missing core adapter seam
   named by the test.
3. Parent starts one production-code subagent after the focused test contract
   is established.
4. Production-code subagent creates `cognition_chain_core` contracts and README.
5. Production-code subagent moves output validation and prompt selection into
   the core and updates imports.
6. Production-code subagent moves L1/L2/L2c2/L2d/L3 implementation behind
   private core stage modules.
7. Production-code subagent implements `run_cognition_chain(...)` and
   `run_text_surface_planning(...)` using the existing graph order and LLM
   calls.
8. Production-code subagent rewrites `persona_supervisor2_cognition.py` and
   `persona_supervisor2_l3_surface.py` into connector modules.
9. Parent updates tests from direct node-stage imports to core public
   contracts or connector entrypoints. No test-only alias module is permitted;
   tests that need private stage behavior must import
   `cognition_chain_core.stages` explicitly and be named as core-stage tests.
10. Parent runs focused tests, static greps, and py_compile gates.
11. Parent runs selected deterministic integration tests and one-at-a-time
   live LLM smoke tests named in `Verification` when live LLM is available.
12. Parent starts one independent code-review subagent after planned
   verification passes.
13. Parent remediates review findings only inside the approved change surface,
   reruns affected verification, and records execution evidence.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it; closes after planned
  production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - focused contract tests established
  - Covers: implementation steps 1 and 2.
  - Verify: run the new focused tests and record expected missing-module or
    baseline results.
  - Evidence: record command output in `Execution Evidence`.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - core contracts and README created
  - Covers: implementation step 4.
  - Verify: `venv\Scripts\python -m py_compile
    src\kazusa_ai_chatbot\cognition_chain_core\contracts.py`.
  - Evidence: record changed files and compile output.
  - Sign-off: `<agent/date>` after rereading this plan.
- [ ] Stage 3 - stage implementation moved behind core
  - Covers: implementation steps 5 through 7.
  - Verify: py_compile all files under `cognition_chain_core` and run focused
    core tests.
  - Evidence: record command output and any moved-file notes.
  - Sign-off: `<agent/date>` after rereading this plan.
- [ ] Stage 4 - Kazusa connectors rewired
  - Covers: implementation step 8.
  - Verify: connector tests for `call_cognition_subgraph(...)` and
    `call_l3_text_surface_handler(...)` pass.
  - Evidence: record test output and connector changed files.
  - Sign-off: `<agent/date>` after rereading this plan.
- [ ] Stage 5 - import migration and regression checks complete
  - Covers: implementation steps 9 through 11.
  - Verify: static greps, focused deterministic tests, and available live LLM
    smoke checks pass.
  - Evidence: record every command and inspected live LLM case result.
  - Sign-off: `<agent/date>` after rereading this plan.
- [ ] Stage 6 - independent code review complete
  - Covers: implementation steps 12 and 13.
  - Verify: independent code-review subagent reports approval or all findings
    are remediated and affected checks rerun.
  - Evidence: record review findings, fixes, rerun commands, and residual
    risks.
  - Sign-off: `<agent/date>` before lifecycle completion.

## Verification

### Static Greps

- `rg "from kazusa_ai_chatbot\\.nodes\\.persona_supervisor2_cognition_(l1|l2|l2c2|l2d|l3)" src tests`
  - Expected result: no matches. `rg` exit code `1` is success for this gate.
- `rg "call_cognition_subconscious|call_cognition_consciousness|call_boundary_core_agent|call_judgment_core_agent|call_social_context_appraisal|call_action_initializer|call_content_plan_agent|call_style_agent|call_preference_adapter|call_visual_agent" src tests`
  - Expected result: direct imports or calls outside `cognition_chain_core`,
    connector entrypoint tests, and explicitly named core-stage tests are
    removed. `rg` exit code `1` is success when no forbidden direct call exists.
- `rg "ActionSpecV1" src\\kazusa_ai_chatbot\\cognition_chain_core`
  - Expected result: no matches. The core must not materialize project action
    specs. `rg` exit code `1` is success.
- `rg "GlobalPersonaState|CognitionState" src\\kazusa_ai_chatbot\\cognition_chain_core`
  - Expected result: no matches. The core must not consume Kazusa graph state
    types. `rg` exit code `1` is success.
- `rg "kazusa_ai_chatbot\\.action_router|from kazusa_ai_chatbot import action_router" src tests`
  - Expected result: no production imports remain. Any surviving test import
    must be in a migration test that asserts action-selection behavior moved to
    `cognition_chain_core.action_selection`.

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_chain_core\contracts.py`
- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_chain_core\chain.py src\kazusa_ai_chatbot\cognition_chain_core\surface.py`
- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_l3_surface.py`

### Focused Deterministic Tests

- `venv\Scripts\python -m pytest tests\test_cognition_chain_core_contracts.py -q`
  - Required tests:
    `test_cognition_chain_input_rejects_raw_graph_and_adapter_fields`,
    `test_cognition_chain_output_rejects_action_specs`, and
    `test_text_surface_input_accepts_prompt_safe_projection`.
- `venv\Scripts\python -m pytest tests\test_cognition_chain_core_action_selection.py -q`
  - Required test:
    `test_semantic_action_request_contract_matches_current_l2d_routes`.
- `venv\Scripts\python -m pytest tests\test_cognition_chain_connector_mapping.py -q`
  - Required tests:
    `test_persona_connector_maps_global_state_to_chain_input`,
    `test_persona_connector_materializes_semantic_actions_after_core_output`,
    and
    `test_l3_surface_connector_projects_selected_speak_without_action_spec_leak`.
- `venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py tests\test_cognition_resolver_l2d_contract.py tests\test_l2d_l3_surface_handoff.py -q`
- `venv\Scripts\python -m pytest tests\test_cognition_resolver_persona_graph.py -q`

### Live LLM Smoke

Run only when live LLM configuration is available. Run one case at a time and
inspect output:

- `venv\Scripts\python -m pytest -m live_llm tests/test_cognition_live_llm.py::test_live_cognition_stack_exercises_each_stage_llm -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests/test_cognition_stage_connection_live_llm.py::test_live_cognition_stage_connection_case -q -s`

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Code-quality and design-weakness review for boundary leaks, accidental
  coupling, fragile compatibility shims, hidden fallback behavior, and
  under-tested connector mappings.
- Core boundary quality: no `GlobalPersonaState`, `CognitionState`,
  `ActionSpecV1`, database rows, adapter ids, queue internals, scheduler
  internals, or action execution inside `cognition_chain_core`.
- Regression and handoff quality, including focused tests, static greps,
  live LLM evidence when available, registry updates, and next-stage notes for
  future parallel stream work.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary,
alter the contract, or add new behavior, stop and update the plan or request
approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `cognition_chain_core` exists with an ICD README and stable public
  contracts.
- `CognitionChainInputV1` is the only caller-facing input contract for the
  main chain.
- `CognitionChainOutputV1` is the only caller-facing output contract for the
  main chain.
- Selected L3 text-surface planning has an explicit core contract and a thin
  Kazusa connector.
- `persona_supervisor2_cognition.py` no longer owns the internal L1/L2/L2d
  graph implementation.
- `persona_supervisor2_l3_surface.py` no longer owns the internal L3 directive
  graph implementation.
- The core does not import `GlobalPersonaState`, `CognitionState`,
  `ActionSpecV1`, DB row types, adapter delivery types, scheduler internals,
  or queue internals.
- Current persona graph behavior, resolver integration, L2d selected action
  semantics, and L3 dialog-facing directive shape remain observably unchanged.
- Static greps, focused deterministic tests, available live LLM smoke checks,
  and independent code review pass with evidence recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Core accidentally keeps Kazusa graph-state coupling | Forbid `GlobalPersonaState` and `CognitionState` imports in core | Static greps and code review |
| L2d materialization leaks into core | Core returns semantic action requests only | `ActionSpecV1` grep and L2d contract tests |
| Prompt behavior drifts during movement | Preserve prompt text and model call order unless neutral naming is required | Prompt contract tests and live LLM smoke |
| Tests keep depending on private internals | Migrate tests to public core contracts or connector tests | Import greps and focused test review |
| Future parallel competition is blocked by hidden side effects | Keep persistence, scheduling, queueing, action execution, and delivery outside core | Code review and boundary greps |

## Execution Evidence

- Not started. This draft plan does not authorize implementation.
