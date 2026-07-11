"""Public contracts for the reusable cognition chain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast

from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMInvoker


class CognitionChainContractError(ValueError):
    """Raised when a caller crosses the cognition-chain ICD boundary."""


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
        "dialog_text",
        "image_observation",
        "audio_observation",
        "internal_monologue",
        "reflection_artifact",
        "accepted_task_result",
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
    origin_metadata: NotRequired[Mapping[str, Any]]


class CharacterPromptV1(TypedDict):
    character_global_id: str
    name: str
    description: str
    gender: str
    age: str
    birthday: str
    backstory: str
    personality_brief: Mapping[str, Any]
    boundary_profile: Mapping[str, Any]
    linguistic_texture_profile: Mapping[str, Any]
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
    affinity: int | str
    affinity_level: str
    last_relationship_insight: str
    memory_context: PromptSafeUserMemoryContextV1
    profile: NotRequired[Mapping[str, Any]]


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
    referents: list[Mapping[str, Any]]
    media_observations: list[MediaObservationPromptV1]
    reply_context_summary: str
    prompt_message_context_summary: str
    reply_context: NotRequired[Mapping[str, Any]]
    prompt_message_context: NotRequired[Mapping[str, Any]]


class ScenePromptV1(TypedDict):
    platform: str
    channel_type: str
    channel_topic: str
    local_time_context: Mapping[str, Any]
    storage_timestamp_utc: str
    interaction_history_recent: list[Mapping[str, Any]]


class ConversationContextPromptV1(TypedDict):
    conversation_progress: Mapping[str, Any] | str
    promoted_reflection_context: Mapping[str, Any] | str
    internal_monologue_residue_context: str
    past_dialog_cognition_context: NotRequired[str]
    previous_action_summary: str


class EvidencePromptV1(TypedDict):
    rag_answer: str
    current_user_rag_bundle: Mapping[str, Any] | str
    memory_evidence: list[TextEvidenceV1]
    conversation_evidence: list[TextEvidenceV1]
    external_evidence: list[TextEvidenceV1]
    recall_evidence: list[TextEvidenceV1]
    supervisor_trace: list[str]
    rag_result: NotRequired[Mapping[str, Any]]


class ResolverPromptV1(TypedDict):
    resolver_context: str
    pending_resume: Mapping[str, Any] | str
    goal_progress: Mapping[str, Any] | str
    recent_observations: list[str]
    max_projected_observations: int
    resolver_state: NotRequired[Mapping[str, Any]]


class ActionAffordanceV1(TypedDict):
    capability: Literal[
        "speak",
        "memory_lifecycle_update",
        "trigger_future_cognition",
        "future_speak",
        "accepted_task_request",
        "accepted_coding_task_request",
        "accepted_task_status_check",
    ]
    available: bool
    visibility: Literal["public", "private", "internal"]
    semantic_input_summary: str | list[str]
    output_kind: Literal["semantic_action_request"]


class RuntimeContextV1(TypedDict):
    language_policy: str
    visual_directives_enabled: bool
    task_willingness_boundary_enabled: bool
    max_action_requests: int
    max_resolver_requests: int
    background_work_output_char_limit: int


class ActionSelectionContextV1(TypedDict):
    coding_runs: list[Mapping[str, Any]]
    group_engagement_action_context: Mapping[str, Any]


class CognitionChainInputV1(TypedDict):
    schema_version: Literal["cognition_chain_input.v1"]
    llm_trace_id: NotRequired[str]
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
    action_selection_context: ActionSelectionContextV1
    coding_run_followup: NotRequired[Mapping[str, Any]]


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
        "future_speak",
        "accepted_task_request",
        "accepted_coding_task_request",
        "accepted_task_status_check",
    ]
    decision: str
    detail: str
    reason: str
    coding_run_ref: NotRequired[str]
    execution_request: NotRequired[str]


class ResolverCapabilityRequestV1(TypedDict):
    schema_version: str
    capability_kind: str
    objective: str
    reason: str
    priority: Literal["now", "background"]


class ResolverPendingResolutionV1(TypedDict):
    decision: str
    reason: str


class ResolverGoalProgressV1(TypedDict, total=False):
    original_goal: str
    current_focus: str
    deliverables: list[Mapping[str, Any]]
    missing_user_inputs: list[str]
    evidence_dependencies: list[str]
    attempted_paths: list[str]
    source_backed_facts: list[str]
    assumptions_or_inferences: list[str]
    blockers: list[str]
    final_response_requirements: list[str]


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
        "accepted_task_request",
        "accepted_coding_task_request",
        "background_work_request",
        "memory_lifecycle_update",
        "future_speak",
    ]
    status: str
    queue_state: NotRequired[str]
    task_summary: str
    objective_summary: str
    acknowledgement_constraint: str
    accepted_task_state: NotRequired[str]
    accepted_task_summary: NotRequired[str]
    wait_guidance: NotRequired[str]


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
    interaction_style_context: Mapping[str, Any]


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
    contextual_directives: Mapping[str, Any]
    linguistic_directives: Mapping[str, Any]
    visual_directives: Mapping[str, Any]


class CognitionTextSurfaceOutputV1(TypedDict):
    schema_version: Literal["cognition_text_surface_output.v1"]
    action_directives: ActionDirectivesV1


class JsonParser(Protocol):
    def __call__(self, content: str) -> Mapping[str, Any] | list[Any]:
        """Parse one JSON-like LLM response string."""


class CognitionLogger(Protocol):
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Record debug detail."""

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Record informational detail."""

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Record a recoverable contract issue."""

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Record an unrecoverable contract issue."""


@dataclass(frozen=True)
class LLMStageBinding:
    llm: LLMInvoker
    config: LLMCallConfig


@dataclass(frozen=True)
class CognitionChainServices:
    llm: LLMInvoker
    cognition_config: LLMCallConfig
    boundary_core_config: LLMCallConfig
    action_selection_config: LLMCallConfig
    style_config: LLMCallConfig
    content_plan_config: LLMCallConfig
    preference_config: LLMCallConfig
    visual_config: LLMCallConfig
    parse_json: JsonParser
    logger: CognitionLogger


def require_llm_binding(
    binding: LLMStageBinding | None,
    service_name: str,
) -> LLMStageBinding:
    """Return an injected LLM binding or fail before a stage call."""

    if binding is None:
        raise RuntimeError(f"{service_name} service was not injected")
    return_value = binding
    return return_value


_FORBIDDEN_KEYS = frozenset((
    "action_specs",
    "adapter_id",
    "collection_name",
    "credentials",
    "delivery_target",
    "handler_id",
    "job_id",
    "lease",
    "platform_bot_id",
    "platform_channel_id",
    "platform_message_id",
    "platform_user_id",
    "queue_future",
    "raw_adapter_payload",
    "scheduler",
    "target_id",
    "worker",
))

_INPUT_REQUIRED_KEYS = frozenset((
    "schema_version",
    "episode",
    "character",
    "current_user",
    "current_event",
    "scene",
    "conversation_context",
    "evidence",
    "resolver",
    "available_actions",
    "runtime_context",
    "action_selection_context",
))
_EPISODE_OUTPUT_MODES = frozenset((
    "live_response",
    "background_cognition",
    "dry_run",
))
_EPISODE_TRIGGER_SOURCES = frozenset((
    "user_message",
    "reflection_signal",
    "internal_thought",
    "scheduled_recall",
    "system_probe",
    "accepted_task_result_ready",
))
_MODEL_VISIBLE_PERCEPT_INPUT_SOURCES = frozenset((
    "dialog_text",
    "image_observation",
    "audio_observation",
    "internal_monologue",
    "reflection_artifact",
    "accepted_task_result",
))
_MEDIA_MODALITIES = frozenset((
    "image",
    "audio",
    "file",
    "link",
    "unknown",
))
_ACTION_CAPABILITIES = frozenset((
    "speak",
    "memory_lifecycle_update",
    "accepted_task_request",
    "accepted_coding_task_request",
    "accepted_task_status_check",
    "trigger_future_cognition",
    "future_speak",
))
_ACTION_VISIBILITIES = frozenset(("public", "private", "internal"))
_ACTION_OUTPUT_KINDS = frozenset(("semantic_action_request",))

_RESIDUE_REQUIRED_KEYS = frozenset((
    "emotional_appraisal",
    "interaction_subtext",
    "internal_monologue",
    "logical_stance",
    "character_intent",
    "judgment_note",
    "social_distance",
    "emotional_intensity",
    "vibe_check",
    "relational_dynamic",
))


def validate_cognition_chain_input(
    value: Mapping[str, Any],
) -> CognitionChainInputV1:
    """Validate the public main-chain input contract."""

    _require_mapping(value, "cognition chain input")
    _require_schema(value, "cognition_chain_input.v1")
    _require_keys(value, _INPUT_REQUIRED_KEYS, "cognition chain input")
    _reject_forbidden_keys(value, "cognition chain input")
    runtime_context = _require_mapping(
        value["runtime_context"],
        "runtime context",
    )
    _require_positive_int(runtime_context, "max_action_requests")
    _require_positive_int(runtime_context, "max_resolver_requests")
    _require_positive_int(runtime_context, "background_work_output_char_limit")
    _require_bool(runtime_context, "task_willingness_boundary_enabled")
    _validate_action_selection_context(value["action_selection_context"])
    _validate_episode(value["episode"])
    _validate_current_event(value["current_event"])
    _validate_available_actions(value["available_actions"])
    validated_value = cast(CognitionChainInputV1, value)
    return validated_value


def validate_cognition_chain_output(
    value: Mapping[str, Any],
) -> CognitionChainOutputV1:
    """Validate the public main-chain output contract."""

    _require_mapping(value, "cognition chain output")
    _require_schema(value, "cognition_chain_output.v1")
    _reject_forbidden_keys(value, "cognition chain output")
    _require_keys(
        value,
        frozenset((
            "schema_version",
            "cognition_residue",
            "semantic_action_requests",
            "resolver_capability_requests",
            "chain_trace",
        )),
        "cognition chain output",
    )
    residue = _require_mapping(value["cognition_residue"], "cognition residue")
    _require_keys(residue, _RESIDUE_REQUIRED_KEYS, "cognition residue")
    _require_list(value["semantic_action_requests"], "semantic action requests")
    _require_list(
        value["resolver_capability_requests"],
        "resolver capability requests",
    )
    validated_value = cast(CognitionChainOutputV1, value)
    return validated_value


def validate_text_surface_input(
    value: Mapping[str, Any],
) -> CognitionTextSurfaceInputV1:
    """Validate the selected text-surface planning input contract."""

    _require_mapping(value, "text surface input")
    _require_schema(value, "cognition_text_surface_input.v1")
    _reject_forbidden_keys(value, "text surface input")
    _require_keys(
        value,
        frozenset((
            "schema_version",
            "chain_input",
            "cognition_residue",
            "selected_text_surface_intent",
            "pre_surface_action_results",
            "memory_lifecycle_context",
            "interaction_style_context",
        )),
        "text surface input",
    )
    validate_cognition_chain_input(value["chain_input"])
    residue = _require_mapping(value["cognition_residue"], "cognition residue")
    _require_keys(residue, _RESIDUE_REQUIRED_KEYS, "cognition residue")
    _require_mapping(
        value["selected_text_surface_intent"],
        "selected text surface intent",
    )
    _require_list(value["pre_surface_action_results"], "pre surface results")
    _require_mapping(
        value["memory_lifecycle_context"],
        "memory lifecycle context",
    )
    _require_mapping(
        value["interaction_style_context"],
        "interaction style context",
    )
    validated_value = cast(CognitionTextSurfaceInputV1, value)
    return validated_value


def validate_text_surface_output(
    value: Mapping[str, Any],
) -> CognitionTextSurfaceOutputV1:
    """Validate the selected text-surface planning output contract."""

    _require_mapping(value, "text surface output")
    _require_schema(value, "cognition_text_surface_output.v1")
    _require_keys(
        value,
        frozenset(("schema_version", "action_directives")),
        "text surface output",
    )
    _require_mapping(value["action_directives"], "action directives")
    validated_value = cast(CognitionTextSurfaceOutputV1, value)
    return validated_value


def _require_schema(value: Mapping[str, Any], schema_version: str) -> None:
    """Require an exact schema version string."""

    if value.get("schema_version") != schema_version:
        raise CognitionChainContractError(f"expected {schema_version}")


def _require_keys(
    value: Mapping[str, Any],
    required_keys: frozenset[str],
    label: str,
) -> None:
    """Require all keys needed by a public contract."""

    missing_keys = sorted(required_keys.difference(value.keys()))
    if missing_keys:
        raise CognitionChainContractError(f"{label} missing keys: {missing_keys}")


def _reject_forbidden_keys(value: Any, label: str) -> None:
    """Reject project-local or executable fields at the core boundary."""

    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if str(key) in _FORBIDDEN_KEYS:
                raise CognitionChainContractError(
                    f"{label} contains forbidden field: {key}"
                )
            _reject_forbidden_keys(nested_value, label)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item, label)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Return a mapping or raise a contract error."""

    if not isinstance(value, Mapping):
        raise CognitionChainContractError(f"{label} must be an object")
    mapping_value = cast(Mapping[str, Any], value)
    return mapping_value


def _require_list(value: Any, label: str) -> list[Any]:
    """Return a list or raise a contract error."""

    if not isinstance(value, list):
        raise CognitionChainContractError(f"{label} must be a list")
    return value


def _require_positive_int(value: Mapping[str, Any], key: str) -> None:
    """Require a positive integer setting in a validated mapping."""

    raw_value = value.get(key)
    if not isinstance(raw_value, int) or raw_value < 1:
        raise CognitionChainContractError(f"{key} must be a positive integer")


def _require_bool(value: Mapping[str, Any], key: str) -> None:
    """Require an exact boolean setting in a validated mapping."""

    raw_value = value.get(key)
    if not isinstance(raw_value, bool):
        raise CognitionChainContractError(f"{key} must be a boolean")


def _validate_episode(value: object) -> None:
    """Validate nested public episode fields that affect prompt routing."""

    episode = _require_mapping(value, "episode")
    trigger_source = episode.get("trigger_source")
    if trigger_source not in _EPISODE_TRIGGER_SOURCES:
        raise CognitionChainContractError(
            "episode.trigger_source must be one of "
            f"{sorted(_EPISODE_TRIGGER_SOURCES)}"
        )
    output_mode = episode.get("output_mode")
    if output_mode not in _EPISODE_OUTPUT_MODES:
        raise CognitionChainContractError(
            f"episode.output_mode must be one of {sorted(_EPISODE_OUTPUT_MODES)}"
        )
    raw_input_sources = _require_list(
        episode.get("input_sources"),
        "input sources",
    )
    for index, input_source in enumerate(raw_input_sources):
        if input_source not in _MODEL_VISIBLE_PERCEPT_INPUT_SOURCES:
            raise CognitionChainContractError(
                f"episode.input_sources[{index}] must be one of "
                f"{sorted(_MODEL_VISIBLE_PERCEPT_INPUT_SOURCES)}"
            )
    raw_percepts = _require_list(
        episode.get("model_visible_percepts"),
        "model visible percepts",
    )
    for index, raw_percept in enumerate(raw_percepts):
        percept = _require_mapping(
            raw_percept,
            f"model visible percept {index}",
        )
        input_source = percept.get("input_source")
        if input_source not in _MODEL_VISIBLE_PERCEPT_INPUT_SOURCES:
            raise CognitionChainContractError(
                "model_visible_percepts.input_source must be one of "
                f"{sorted(_MODEL_VISIBLE_PERCEPT_INPUT_SOURCES)}"
            )


def _validate_current_event(value: object) -> None:
    """Validate current-event media rows."""

    current_event = _require_mapping(value, "current event")
    raw_media = _require_list(
        current_event.get("media_observations"),
        "media observations",
    )
    for index, raw_media_row in enumerate(raw_media):
        media_row = _require_mapping(raw_media_row, f"media observation {index}")
        modality = media_row.get("modality")
        if modality not in _MEDIA_MODALITIES:
            raise CognitionChainContractError(
                f"media_observations.modality must be one of "
                f"{sorted(_MEDIA_MODALITIES)}"
            )


def _validate_available_actions(value: object) -> None:
    """Validate caller-provided semantic action affordances."""

    raw_actions = _require_list(value, "available actions")
    for index, raw_action in enumerate(raw_actions):
        action = _require_mapping(raw_action, f"available action {index}")
        capability = action.get("capability")
        if capability not in _ACTION_CAPABILITIES:
            raise CognitionChainContractError(
                f"available_actions.capability must be one of "
                f"{sorted(_ACTION_CAPABILITIES)}"
            )
        visibility = action.get("visibility")
        if visibility not in _ACTION_VISIBILITIES:
            raise CognitionChainContractError(
                f"available_actions.visibility must be one of "
                f"{sorted(_ACTION_VISIBILITIES)}"
            )
        output_kind = action.get("output_kind")
        if output_kind not in _ACTION_OUTPUT_KINDS:
            raise CognitionChainContractError(
                f"available_actions.output_kind must be one of "
                f"{sorted(_ACTION_OUTPUT_KINDS)}"
            )


def _validate_action_selection_context(value: object) -> None:
    """Validate trusted context used by semantic action selection."""

    context = _require_mapping(value, "action selection context")
    _require_list(context.get("coding_runs"), "action selection coding runs")
    _require_mapping(
        context.get("group_engagement_action_context"),
        "group engagement action context",
    )
