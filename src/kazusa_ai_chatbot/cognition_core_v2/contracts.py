"""Internal contracts for the validation-local cognition core."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping, NotRequired, TypedDict

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeV1,
    CognitiveEpisodeValidationError,
    validate_cognitive_episode_v1,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMInvoker
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    RelationshipStateV2,
    validate_cognition_state,
    validate_relationship_state,
)


@dataclass(frozen=True)
class EmotionDefinition:
    """Define a causal emotion family and its lifecycle semantics."""

    emotion_id: str
    causal_inputs: tuple[str, ...]
    begin_guard: str
    sustain_rule: str
    fade_rule: str
    action_tendencies: tuple[str, ...]
    decay_rate_per_hour: int = 4
    causal_entity_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class BranchDefinition:
    """Describe the state conditions and dependencies for one goal branch."""

    branch_id: str
    dependencies: tuple[str, ...]
    action_tendencies: tuple[str, ...]
    required: bool = False
    goal_kind: str = "goal"
    dependency_options: tuple[tuple[str, ...], ...] = ()


class CognitionContractError(ValueError):
    """Raised when a V2 public boundary is structurally invalid."""


class CognitionExecutionError(CognitionContractError):
    """Raised when collapse or route execution cannot produce a valid result."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "internal_invariant",
        branch_id: str = "",
        stage: str = "",
        attempt_count: int = 1,
        safe_checkpoint: str = "unknown",
        retryable: bool = False,
    ) -> None:
        """Attach bounded failure metadata to one cognition execution error.

        Args:
            message: Human-readable internal error detail.
            error_code: Stable machine-readable failure class.
            branch_id: Cognition branch that failed, when applicable.
            stage: Runtime stage that detected the failure.
            attempt_count: Attempts already consumed inside the failing owner.
            safe_checkpoint: Latest checkpoint reached before the failure.
            retryable: Whether deterministic policy may repeat from that
                checkpoint.
        """

        super().__init__(message)
        self.error_code = error_code
        self.branch_id = branch_id
        self.stage = stage
        self.attempt_count = attempt_count
        self.safe_checkpoint = safe_checkpoint
        self.retryable = retryable


class CognitionContextLimitError(CognitionContractError):
    """Raised when required model context remains over its frozen cap."""


SEMANTIC_QUESTION_KINDS = (
    "event_agency",
    "relationship_social",
    "moral_identity",
    "goal_threat_outcome",
    "epistemic_comparison_memory",
    "existential_drive",
)

GoalResolutionV2 = Literal[
    "answerable_now",
    "requires_required_evidence",
    "requires_user_input",
    "blocked",
]

GOAL_RESOLUTION_VALUES = frozenset({
    "answerable_now",
    "requires_required_evidence",
    "requires_user_input",
    "blocked",
})

EVIDENCE_SOURCE_QUESTION_IDS = {
    "episode": tuple(f"q:{kind}" for kind in SEMANTIC_QUESTION_KINDS),
    "promoted_memory": tuple(f"q:{kind}" for kind in SEMANTIC_QUESTION_KINDS),
    "promoted_reflection": tuple(f"q:{kind}" for kind in SEMANTIC_QUESTION_KINDS),
    "media_observation": tuple(f"q:{kind}" for kind in SEMANTIC_QUESTION_KINDS),
    "action_result": (
        "q:event_agency",
        "q:relationship_social",
        "q:moral_identity",
        "q:goal_threat_outcome",
    ),
    "resolver_observation": (
        "q:event_agency",
        "q:relationship_social",
        "q:moral_identity",
        "q:goal_threat_outcome",
        "q:epistemic_comparison_memory",
    ),
    "tool_result": (
        "q:event_agency",
        "q:relationship_social",
        "q:moral_identity",
        "q:goal_threat_outcome",
        "q:epistemic_comparison_memory",
    ),
    "scheduler_event": ("q:goal_threat_outcome",),
}

GOAL_BRANCH_IDS = (
    "ordinary_response",
    "relationship_connection",
    "bond_protection",
    "trust_verification",
    "autonomy_boundary",
    "safety_coping",
    "obstruction_strategy",
    "loss_recovery",
    "moral_repair",
    "social_care",
    "reciprocal_response",
    "epistemic_exploration",
    "meaning_reconstruction",
    "self_improvement",
)

ENTITY_KINDS = {
    "relationship",
    "goal",
    "threat",
    "event",
    "knowledge_gap",
    "drive",
    "standard",
    "meaning",
}
ROLE_VALUES = {
    "actor",
    "experiencer",
    "target",
    "object",
    "affected_goal",
    "affected_relationship",
}
ROLE_ENTITY_KINDS = {
    "character",
    "user",
    "group",
    "third_party",
    "goal",
    "relationship",
    "standard",
    "object",
}


class EntityRefV2(TypedDict):
    """Scope-qualified reference to one persistent entity."""

    scope: Literal["user", "character"]
    kind: str
    entity_id: str


class RoleRefV2(TypedDict):
    """Semantic role assignment for a persistent entity."""

    role: str
    entity_kind: str
    entity_id: str


class EvidenceRefV2(TypedDict):
    """Complete provenance record retained by the reducer."""

    source_kind: str
    source_id: str
    occurred_at: str
    semantic_summary: str


class CognitionEvidenceV2(TypedDict):
    """Prompt-safe evidence row with an episode-local handle."""

    evidence_handle: str
    evidence_ref: EvidenceRefV2
    semantic_text: str
    visible_to: list[str]


class DirectFactV2(TypedDict):
    """Trusted typed fact accepted by the deterministic reducer."""

    fact_id: str
    producer: str
    fact_kind: str
    target_refs: list[EntityRefV2 | RoleRefV2]
    evidence_ref: EvidenceRefV2
    observed_progress: NotRequired[int]


class ActionAffordanceV2(TypedDict):
    """Semantic action capability available to route selection."""

    action_kind: str
    capability: str
    permission: str
    decision_mode: Literal["optional", "required_text", "closed"]
    allowed_decisions: list[str]
    default_decision: str
    decision_pattern: str
    context_ref: str
    target_roles: list[RoleRefV2]


class ResolverAffordanceV2(TypedDict):
    """Semantic resolver capability available to route selection."""

    capability: str
    semantic_capability: str
    availability: str


class SceneContextV2(TypedDict):
    """Prompt-safe scene context without platform identifiers."""

    channel_scope: Literal["private", "group", "internal"]
    character_role: str
    current_user_role: NotRequired[str]
    semantic_scene: str
    conversation_continuity: str
    semantic_temporal_context: str


class CharacterConstraintSnapshotV2(TypedDict):
    """Read-only character constraints supplied to user-scope appraisal."""

    drives: dict[str, dict[str, Any]]
    standards: list[dict[str, Any]]
    meaning_state: dict[str, Any]


class SemanticQuestionV2(TypedDict):
    """One bounded semantic question owned by one appraisal family."""

    question_id: str
    question_kind: str
    semantic_question: str
    evidence_handles: list[str]
    permitted_role_handles: list[str]
    permitted_delta_paths: list[str]
    dependencies: list[str]


class SemanticRoleAssignmentV2(TypedDict):
    """Model-selected semantic role mapped to a prompt-local handle."""

    role: str
    entity_handle: str


class SemanticPropositionV2(TypedDict):
    """Meaning proposition returned by one scoped appraisal."""

    proposition_kind: str
    subject_handle: str
    object_handle: NotRequired[str]
    evidence_handles: list[str]
    role_assignments: list[SemanticRoleAssignmentV2]
    semantic_value: str


class SemanticDeltaV2(TypedDict):
    """Allowlisted numeric state delta with complete evidence handles."""

    target_path: str
    delta: int
    evidence_handles: list[str]
    reason: str


class SemanticAppraisalResultV2(TypedDict):
    """Validated result from one semantic question."""

    question_id: str
    selected_evidence_handles: list[str]
    selected_role_handles: list[str]
    propositions: list[SemanticPropositionV2]
    deltas: list[SemanticDeltaV2]
    explanation: str


class ActionBidV2(TypedDict):
    """Complete branch-owned bid copied without model-authored authority."""

    branch_id: str
    goal_ref: EntityRefV2
    intention: str
    desired_outcome: str
    concrete_detail: str
    reason: str
    private_monologue: str
    target_roles: list[RoleRefV2]
    evidence_handles: list[str]
    expected_consequences: list[str]
    confidence: str


class GoalBidDraftV2(TypedDict):
    """Model-owned branch draft before deterministic handle mapping."""

    intention: str
    desired_outcome: str
    concrete_detail: str
    reason: str
    private_monologue: str
    target_role_handles: list[str]
    evidence_handles: list[str]
    expected_consequences: list[str]
    confidence: str


class SelectedIntentionV2(TypedDict):
    """Deterministic route and intention selected from a complete bid."""

    selected_branch_id: NotRequired[str]
    route: Literal["speech", "evidence", "action", "deferral", "silence"]
    intention: str
    target_roles: list[RoleRefV2]
    reason: str


class CollapsedIntentionV2(TypedDict):
    """Workspace result copied from complete internal bids."""

    primary_branch_id: str
    supporting_branch_ids: list[str]
    suppressed_branch_ids: list[str]
    primary_bid: ActionBidV2
    supporting_bids: list[ActionBidV2]
    competing_bids: list[ActionBidV2]


class WorkspaceDecisionV2(TypedDict):
    """Prompt-local workspace partition emitted by the collapse model."""

    primary_bid_handle: str
    supporting_bid_handles: list[str]
    suppressed_bid_handles: list[str]


class SemanticActionRequestV2(TypedDict):
    """Planner-selected action request; execution remains action-spec owned."""

    action_kind: str
    decision: str
    context_ref: str
    semantic_goal: str
    reason: str
    target_roles: list[RoleRefV2]
    evidence_handles: list[str]


class ResolverCapabilityRequestV2(TypedDict):
    """Planner-selected resolver request; execution remains resolver owned."""

    capability: str
    semantic_goal: str
    reason: str
    evidence_handles: list[str]


class ResolverProgressV2(TypedDict):
    """Bounded resolver recurrence status."""

    status: Literal["not_requested", "pending", "completed", "failed"]
    semantic_summary: str


class ExpressionPolicyV2(TypedDict):
    """Deterministic expression constraints passed to the text surface."""

    visibility: Literal["visible", "private", "none"]
    emotional_tone: str
    intensity: Literal["restrained", "moderate", "strong"]
    directness: Literal["indirect", "balanced", "direct"]


class SemanticAffectProjectionV2(TypedDict):
    """Semantic affect projection with no raw internal scalar."""

    emotion: str
    phase: str
    intensity: str
    trend: str
    cause_summary: str


class SemanticRelationshipProjectionV2(TypedDict):
    """Semantic relationship projection with approved qualitative bands."""

    relationship_summary: str
    axis_summaries: dict[str, str]


class StateUpdateV2(TypedDict):
    """One validated replacement state and deterministic change summary."""

    state_scope: Literal["user", "character"]
    owner_key: str
    replacement_state: dict[str, Any]
    comparison_results: list[EventComparisonResultV2]
    changed_paths: list[str]


class EventComparisonResultV2(TypedDict):
    """Cause comparison retained in evidence-source order."""

    current_event_ref: EntityRefV2
    matched_entity_ref: NotRequired[EntityRefV2]
    outcome: Literal[
        "reinforce",
        "contradict",
        "resolve",
        "replace",
        "create",
        "unrelated",
    ]
    evidence_refs: list[EvidenceRefV2]


class CognitionDiagnosticsV2(TypedDict):
    """Protected bounded execution diagnostics."""

    run_id: str
    stage_status: dict[str, Literal["completed", "failed", "skipped"]]
    selected_question_count: int
    dispatched_question_count: int
    selected_branch_count: int
    dispatched_branch_count: int
    completed_branch_count: int
    failed_branch_count: int
    overlap_ms: int
    dependency_wait_ms: int
    total_ms: int
    warnings: list[str]


class CognitionCoreInputV2(TypedDict):
    """Public V2 cognition input contract."""

    schema_version: Literal["cognition_core_input.v2"]
    episode: CognitiveEpisodeV1
    state_scope: Literal["user", "character"]
    mutable_state: dict[str, Any]
    character_constraints: CharacterConstraintSnapshotV2
    relationship_context: NotRequired[RelationshipStateV2]
    evidence: list[CognitionEvidenceV2]
    direct_facts: list[DirectFactV2]
    available_actions: list[ActionAffordanceV2]
    available_resolver_capabilities: list[ResolverAffordanceV2]
    resolver_context: str
    resolver_goal_progress: NotRequired[dict[str, Any]]
    pending_resolver_resume: NotRequired[dict[str, Any]]
    scene_context: SceneContextV2
    private_continuity_context: str


class CognitionCoreOutputV2(TypedDict):
    """Public V2 cognition output contract."""

    schema_version: Literal["cognition_core_output.v2"]
    intention: SelectedIntentionV2
    admitted_bid: NotRequired[ActionBidV2]
    supporting_bids: list[ActionBidV2]
    state_update: StateUpdateV2
    affect_projection: list[SemanticAffectProjectionV2]
    relationship_projection: NotRequired[SemanticRelationshipProjectionV2]
    action_requests: list[SemanticActionRequestV2]
    resolver_requests: list[ResolverCapabilityRequestV2]
    goal_resolution: GoalResolutionV2
    resolver_pending_resolution: dict[str, Any] | None
    resolver_goal_progress: dict[str, Any] | None
    resolver_progress: ResolverProgressV2
    selected_bid_reason: str
    private_monologue: str
    expression_policy: ExpressionPolicyV2
    diagnostics: CognitionDiagnosticsV2


class SurfaceBidProjectionV2(TypedDict):
    """Bid subset allowed to the V2 text-surface planner."""

    motive: str
    intention: str
    desired_outcome: str
    permitted_detail: str
    target_summaries: list[str]
    expected_consequences: list[str]


class SemanticActionResultV2(TypedDict):
    """Typed action result allowed into the surface planner."""

    action_kind: str
    status: Literal[
        "executed",
        "scheduled",
        "pending",
        "failed",
        "unavailable",
    ]
    semantic_result: str
    target_roles: list[RoleRefV2]


class TextSurfaceInputV2(TypedDict):
    """Public V2 text-surface input contract."""

    schema_version: Literal["text_surface_input.v2"]
    episode: CognitiveEpisodeV1
    intention: SelectedIntentionV2
    primary_bid: NotRequired[SurfaceBidProjectionV2]
    supporting_bids: list[SurfaceBidProjectionV2]
    expression_policy: ExpressionPolicyV2
    semantic_affect: list[SemanticAffectProjectionV2]
    semantic_relationship: NotRequired[SemanticRelationshipProjectionV2]
    permitted_action_results: list[SemanticActionResultV2]
    interaction_style_context: str
    character_voice_context: str


class TextSurfaceOutputV2(TypedDict):
    """Public V2 text-surface output contract."""

    schema_version: Literal["text_surface_output.v2"]
    content_plan: str
    content_requirements: list[str]
    visible_boundaries: list[str]
    addressee_plan: list[str]
    style_guidance: str
    selected_surface_intent: str
    permitted_action_results: list[SemanticActionResultV2]


class VisualSurfaceOutputV2(TypedDict):
    """Public V2 terminal visual-surface output contract."""

    schema_version: Literal["visual_surface_output.v2"]
    visual_directives: str
    selected_surface_intent: str


@dataclass(frozen=True)
class CognitionCoreServicesV2:
    """Injected V2 model bindings; services never enter model payloads."""

    llm: LLMInvoker
    appraisal_config: LLMCallConfig
    goal_cognition_config: LLMCallConfig
    collapse_config: LLMCallConfig
    action_selection_config: LLMCallConfig


@dataclass(frozen=True)
class TextSurfaceServicesV2:
    """Injected three-stage V2 text-surface bindings."""

    llm: LLMInvoker
    style_config: LLMCallConfig
    content_plan_config: LLMCallConfig
    preference_config: LLMCallConfig


@dataclass(frozen=True)
class VisualSurfaceServicesV2:
    """Injected terminal V2 visual-surface binding."""

    llm: LLMInvoker
    visual_config: LLMCallConfig


def validate_cognition_core_input(
    payload: Mapping[str, Any],
) -> CognitionCoreInputV2:
    """Validate the V2 public input before any model call or state mutation."""

    _require_exact_keys(
        payload,
        {
            "schema_version",
            "episode",
            "state_scope",
            "mutable_state",
            "character_constraints",
            "evidence",
            "direct_facts",
            "available_actions",
            "available_resolver_capabilities",
            "resolver_context",
            "scene_context",
            "private_continuity_context",
        }
        | ({"relationship_context"} if "relationship_context" in payload else set())
        | (
            {"resolver_goal_progress"}
            if "resolver_goal_progress" in payload
            else set()
        )
        | (
            {"pending_resolver_resume"}
            if "pending_resolver_resume" in payload
            else set()
        ),
        "cognition core input",
    )
    if payload["schema_version"] != "cognition_core_input.v2":
        raise CognitionContractError("unsupported cognition core input schema")
    scope = payload["state_scope"]
    if scope not in {"user", "character"}:
        raise CognitionContractError("cognition core state scope is invalid")
    state = payload["mutable_state"]
    if not isinstance(state, Mapping) or state.get("state_scope") != scope:
        raise CognitionContractError("mutable state scope does not match input")
    _validate_persistent_state(state)
    episode = _validate_canonical_episode(payload["episode"])
    if "relationship_context" in payload:
        _validate_relationship_context(
            payload["relationship_context"],
            scope=scope,
            state=state,
            episode=episode,
        )
    _validate_character_constraints(payload["character_constraints"])
    _validate_evidence_rows(payload["evidence"])
    if not isinstance(payload["direct_facts"], list):
        raise CognitionContractError("direct_facts must be a list")
    for row in payload["direct_facts"]:
        _validate_direct_fact(row)
    if not isinstance(payload["available_actions"], list):
        raise CognitionContractError("available_actions must be a list")
    for row in payload["available_actions"]:
        _validate_action_affordance(row)
    if not isinstance(payload["available_resolver_capabilities"], list):
        raise CognitionContractError(
            "available_resolver_capabilities must be a list"
        )
    for row in payload["available_resolver_capabilities"]:
        _validate_resolver_affordance(row)
    _require_bounded_text(
        payload["resolver_context"],
        "resolver context",
        maximum=8000,
    )
    if "pending_resolver_resume" in payload:
        _validate_pending_resolver_resume(payload["pending_resolver_resume"])
    if "resolver_goal_progress" in payload:
        _validate_resolver_goal_progress_input(
            payload["resolver_goal_progress"]
        )
    if not isinstance(payload["scene_context"], Mapping):
        raise CognitionContractError("scene_context must be a mapping")
    _validate_scene_context(payload["scene_context"])
    _require_bounded_text(
        payload["private_continuity_context"],
        "private continuity context",
        maximum=1000,
    )
    return dict(payload)  # type: ignore[return-value]


def validate_cognition_core_output(
    payload: Mapping[str, Any],
) -> CognitionCoreOutputV2:
    """Validate the complete V2 result before persistence or downstream work."""

    _require_exact_keys(
        payload,
        {
            "schema_version",
            "intention",
            "supporting_bids",
            "state_update",
            "affect_projection",
            "action_requests",
            "resolver_requests",
            "goal_resolution",
            "resolver_pending_resolution",
            "resolver_goal_progress",
            "resolver_progress",
            "selected_bid_reason",
            "private_monologue",
            "expression_policy",
            "diagnostics",
        } | ({"admitted_bid"} if "admitted_bid" in payload else set())
        | (
            {"relationship_projection"}
            if "relationship_projection" in payload
            else set()
        ),
        "cognition core output",
    )
    if payload["schema_version"] != "cognition_core_output.v2":
        raise CognitionContractError("unsupported cognition core output schema")
    if not isinstance(payload["intention"], Mapping):
        raise CognitionContractError("output intention must be a mapping")
    _validate_intention(payload["intention"])
    if not isinstance(payload["supporting_bids"], list):
        raise CognitionContractError("supporting_bids must be a list")
    for bid in payload["supporting_bids"]:
        _validate_action_bid(bid)
    if "admitted_bid" in payload:
        _validate_action_bid(payload["admitted_bid"])
    if not isinstance(payload["state_update"], Mapping):
        raise CognitionContractError("state_update must be a mapping")
    _validate_state_update(payload["state_update"])
    if not isinstance(payload["affect_projection"], list):
        raise CognitionContractError("affect_projection must be a list")
    for row in payload["affect_projection"]:
        _validate_affect_projection(row)
    if not isinstance(payload["action_requests"], list):
        raise CognitionContractError("action_requests must be a list")
    for row in payload["action_requests"]:
        _validate_action_request(row)
    if not isinstance(payload["resolver_requests"], list):
        raise CognitionContractError("resolver_requests must be a list")
    for row in payload["resolver_requests"]:
        _validate_resolver_request(row)
    _validate_goal_resolution(payload["goal_resolution"])
    _validate_resolver_lifecycle_output(
        payload["resolver_pending_resolution"],
        payload["resolver_goal_progress"],
    )
    _validate_resolver_progress(payload["resolver_progress"])
    _validate_expression_policy(payload["expression_policy"])
    if "relationship_projection" in payload:
        _validate_relationship_projection(payload["relationship_projection"])
    _validate_diagnostics(payload["diagnostics"])
    _require_text(
        payload["selected_bid_reason"],
        "selected bid reason",
        maximum=1000,
    )
    _require_text(payload["private_monologue"], "private monologue", maximum=1000)
    return dict(payload)  # type: ignore[return-value]


def validate_text_surface_input(
    payload: Mapping[str, Any],
) -> TextSurfaceInputV2:
    """Validate the V2 L3 input and its no-raw-state surface boundary."""

    _require_exact_keys(
        payload,
        {
            "schema_version",
            "episode",
            "intention",
            "supporting_bids",
            "expression_policy",
            "semantic_affect",
            "permitted_action_results",
            "interaction_style_context",
            "character_voice_context",
        } | ({"primary_bid"} if "primary_bid" in payload else set())
        | ({"semantic_relationship"} if "semantic_relationship" in payload else set()),
        "text surface input",
    )
    if payload["schema_version"] != "text_surface_input.v2":
        raise CognitionContractError("unsupported text surface input schema")
    _validate_intention(payload["intention"])
    _require_text(payload["interaction_style_context"], "interaction style")
    _require_text(
        payload["character_voice_context"],
        "character voice context",
        maximum=1500,
    )
    _validate_canonical_episode(payload["episode"])
    if "primary_bid" in payload:
        _validate_surface_bid(payload["primary_bid"])
    if not isinstance(payload["supporting_bids"], list):
        raise CognitionContractError("surface supporting_bids must be a list")
    for bid in payload["supporting_bids"]:
        _validate_surface_bid(bid)
    _validate_expression_policy(payload["expression_policy"])
    if not isinstance(payload["semantic_affect"], list):
        raise CognitionContractError("surface semantic_affect must be a list")
    for row in payload["semantic_affect"]:
        _validate_affect_projection(row)
    if "semantic_relationship" in payload:
        _validate_relationship_projection(payload["semantic_relationship"])
    if not isinstance(payload["permitted_action_results"], list):
        raise CognitionContractError(
            "surface permitted_action_results must be a list"
        )
    for row in payload["permitted_action_results"]:
        _validate_action_result(row)
    return dict(payload)  # type: ignore[return-value]


def validate_text_surface_output(
    payload: Mapping[str, Any],
) -> TextSurfaceOutputV2:
    """Validate the bounded V2 L3 output."""

    required = {
        "schema_version",
        "content_plan",
        "content_requirements",
        "visible_boundaries",
        "addressee_plan",
        "style_guidance",
        "selected_surface_intent",
        "permitted_action_results",
    }
    _require_exact_keys(payload, required, "text surface output")
    if payload["schema_version"] != "text_surface_output.v2":
        raise CognitionContractError("unsupported text surface output schema")
    for field_name in (
        "content_plan",
        "style_guidance",
        "selected_surface_intent",
    ):
        _require_text(payload[field_name], field_name, maximum=1000)
    requirements = payload["content_requirements"]
    if not isinstance(requirements, list) or not 1 <= len(requirements) <= 8:
        raise CognitionContractError("content_requirements must contain 1-8 items")
    if len(requirements) != len(set(requirements)):
        raise CognitionContractError("content_requirements contains duplicates")
    for index, item in enumerate(requirements):
        _require_text(item, f"content_requirements[{index}]", maximum=500)
    for field_name in ("visible_boundaries", "addressee_plan"):
        if not isinstance(payload[field_name], list):
            raise CognitionContractError(f"{field_name} must be a list")
        for index, item in enumerate(payload[field_name]):
            _require_text(
                item,
                f"{field_name}[{index}]",
                maximum=1000,
            )
    action_results = payload["permitted_action_results"]
    if not isinstance(action_results, list):
        raise CognitionContractError(
            "permitted_action_results must be a list"
        )
    for row in action_results:
        _validate_action_result(row)
    return dict(payload)  # type: ignore[return-value]


def validate_visual_surface_output(
    payload: Mapping[str, Any],
) -> VisualSurfaceOutputV2:
    """Validate the bounded terminal V2 visual output."""

    _require_exact_keys(
        payload,
        {
            "schema_version",
            "visual_directives",
            "selected_surface_intent",
        },
        "visual surface output",
    )
    if payload["schema_version"] != "visual_surface_output.v2":
        raise CognitionContractError("unsupported visual surface output schema")
    for field_name in ("visual_directives", "selected_surface_intent"):
        _require_text(payload[field_name], field_name, maximum=1000)
    return dict(payload)  # type: ignore[return-value]


def _validate_persistent_state(state: Mapping[str, Any]) -> None:
    """Delegate exact native-state validation."""

    try:
        validate_cognition_state(state)
    except ValueError as exc:
        raise CognitionContractError(str(exc)) from exc


def _validate_evidence_rows(rows: Any) -> None:
    """Validate evidence handles and complete provenance records."""

    if not isinstance(rows, list) or len(rows) > 32:
        raise CognitionContractError("evidence rows are invalid")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise CognitionContractError("evidence row must be a mapping")
        _require_exact_keys(
            row,
            {"evidence_handle", "evidence_ref", "semantic_text", "visible_to"},
            "evidence row",
        )
        handle = row["evidence_handle"]
        if (
            not isinstance(handle, str)
            or len(handle) < 2
            or handle[0] != "e"
            or not handle[1:].isdigit()
            or handle in seen
        ):
            raise CognitionContractError("evidence handle is invalid")
        seen.add(handle)
        _validate_evidence_ref(row["evidence_ref"])
        _require_text(row["semantic_text"], "semantic_text", maximum=1000)
        if (
            not isinstance(row["visible_to"], list)
            or not row["visible_to"]
            or any(
                not isinstance(audience, str) or not audience.strip()
                for audience in row["visible_to"]
            )
        ):
            raise CognitionContractError("evidence visibility must be a list")
        if len(row["visible_to"]) != len(set(row["visible_to"])):
            raise CognitionContractError("evidence visibility is duplicated")
        source_kind = row["evidence_ref"]["source_kind"]
        required_question_ids = set(EVIDENCE_SOURCE_QUESTION_IDS[source_kind])
        visibility = set(row["visible_to"])
        allowed = required_question_ids | set(GOAL_BRANCH_IDS)
        if not visibility.issubset(allowed):
            raise CognitionContractError("evidence visibility id is invalid")
        visible_question_ids = {
            value for value in visibility if value.startswith("q:")
        }
        if visible_question_ids != required_question_ids:
            raise CognitionContractError(
                "evidence visibility does not match its source kind"
            )


def _validate_intention(value: Any) -> None:
    """Validate the deterministic intention route envelope."""

    if not isinstance(value, Mapping):
        raise CognitionContractError("intention must be a mapping")
    required = {"route", "intention", "target_roles", "reason"}
    if "selected_branch_id" in value:
        required.add("selected_branch_id")
    _require_exact_keys(value, required, "intention")
    if value["route"] not in {
        "speech",
        "evidence",
        "action",
        "deferral",
        "silence",
    }:
        raise CognitionContractError("intention route is invalid")
    _require_text(value["intention"], "intention")
    _require_text(value["reason"], "intention.reason")
    _validate_roles(value["target_roles"], "intention.target_roles")
    if "selected_branch_id" in value:
        _require_text(value["selected_branch_id"], "intention.selected_branch_id")


def _validate_action_bid(value: Any) -> None:
    """Validate one complete branch-owned bid at the public boundary."""

    if not isinstance(value, Mapping):
        raise CognitionContractError("action bid must be a mapping")
    required = {
        "branch_id",
        "goal_ref",
        "intention",
        "desired_outcome",
        "concrete_detail",
        "reason",
        "private_monologue",
        "target_roles",
        "evidence_handles",
        "expected_consequences",
        "confidence",
    }
    if set(value) != required:
        raise CognitionContractError("action bid fields are not exact")
    for field_name in (
        "branch_id",
        "intention",
        "desired_outcome",
        "concrete_detail",
        "reason",
        "private_monologue",
        "confidence",
    ):
        _require_text(value[field_name], f"action bid.{field_name}")
    _validate_entity_ref(value["goal_ref"], "action bid.goal_ref")
    _validate_roles(value["target_roles"], "action bid.target_roles")
    _validate_text_list(value["evidence_handles"], "action bid.evidence_handles")
    _validate_text_list(
        value["expected_consequences"],
        "action bid.expected_consequences",
    )
def _validate_action_request(value: Any) -> None:
    """Validate one route-approved semantic action request."""

    if not isinstance(value, Mapping) or set(value) != {
        "action_kind",
        "decision",
        "context_ref",
        "semantic_goal",
        "reason",
        "target_roles",
        "evidence_handles",
    }:
        raise CognitionContractError("action request fields are not exact")
    _require_text(value["action_kind"], "action request.action_kind")
    _require_bounded_text(value["decision"], "action request.decision", maximum=200)
    _require_bounded_text(
        value["context_ref"],
        "action request.context_ref",
        maximum=200,
    )
    _require_text(value["semantic_goal"], "action request.semantic_goal")
    _require_text(value["reason"], "action request.reason")
    _validate_roles(value["target_roles"], "action request.target_roles")
    _validate_text_list(value["evidence_handles"], "action request.evidence_handles")


def _validate_resolver_request(value: Any) -> None:
    """Validate one route-approved semantic resolver request."""

    if not isinstance(value, Mapping) or set(value) != {
        "capability",
        "semantic_goal",
        "reason",
        "evidence_handles",
    }:
        raise CognitionContractError("resolver request fields are not exact")
    _require_text(value["capability"], "resolver request.capability")
    _require_text(value["semantic_goal"], "resolver request.semantic_goal")
    _require_text(value["reason"], "resolver request.reason")
    _validate_text_list(value["evidence_handles"], "resolver request.evidence_handles")


def _validate_affect_projection(value: Any) -> None:
    """Validate semantic affect without internal scalar fields."""

    required = {"emotion", "phase", "intensity", "trend", "cause_summary"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("affect projection fields are not exact")
    for field_name in required:
        _require_text(value[field_name], f"affect projection.{field_name}")


def _validate_relationship_projection(value: Any) -> None:
    """Validate semantic relationship context without raw axes."""

    required = {"relationship_summary", "axis_summaries"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("relationship projection fields are not exact")
    _require_text(value["relationship_summary"], "relationship projection.summary")
    axes = value["axis_summaries"]
    if not isinstance(axes, Mapping):
        raise CognitionContractError("relationship projection axes are invalid")
    for axis, band in axes.items():
        _require_text(axis, "relationship projection axis")
        _require_text(band, "relationship projection band")


def _validate_expression_policy(value: Any) -> None:
    """Validate deterministic visible-expression limits."""

    required = {"visibility", "emotional_tone", "intensity", "directness"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("expression policy fields are not exact")
    if value["visibility"] not in {"visible", "private", "none"}:
        raise CognitionContractError("expression policy visibility is invalid")
    if value["intensity"] not in {"restrained", "moderate", "strong"}:
        raise CognitionContractError("expression policy intensity is invalid")
    if value["directness"] not in {"indirect", "balanced", "direct"}:
        raise CognitionContractError("expression policy directness is invalid")
    _require_text(value["emotional_tone"], "expression policy.emotional_tone")


def _validate_resolver_progress(value: Any) -> None:
    """Validate bounded resolver progress."""

    required = {"status", "semantic_summary"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("resolver progress fields are not exact")
    if value["status"] not in {"not_requested", "pending", "completed", "failed"}:
        raise CognitionContractError("resolver progress status is invalid")
    _require_text(value["semantic_summary"], "resolver progress.semantic_summary")


def _validate_goal_resolution(value: Any) -> None:
    """Validate Cognition Core's user-goal answerability decision."""

    if not isinstance(value, str) or value not in GOAL_RESOLUTION_VALUES:
        raise CognitionContractError("goal resolution is invalid")


def _validate_diagnostics(value: Any) -> None:
    """Validate bounded execution metrics and stage statuses."""

    required = {
        "run_id",
        "stage_status",
        "selected_question_count",
        "dispatched_question_count",
        "selected_branch_count",
        "dispatched_branch_count",
        "completed_branch_count",
        "failed_branch_count",
        "overlap_ms",
        "dependency_wait_ms",
        "total_ms",
        "warnings",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("diagnostics fields are not exact")
    _require_text(value["run_id"], "diagnostics.run_id")
    if not isinstance(value["stage_status"], Mapping):
        raise CognitionContractError("diagnostics stage_status is invalid")
    if any(
        status not in {"completed", "failed", "skipped"}
        for status in value["stage_status"].values()
    ):
        raise CognitionContractError("diagnostics stage status is invalid")
    for field_name in required - {"run_id", "stage_status", "warnings"}:
        field_value = value[field_name]
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value < 0
        ):
            raise CognitionContractError(f"diagnostics {field_name} is invalid")
    _validate_text_list(value["warnings"], "diagnostics.warnings", allow_empty=True)


def _validate_action_affordance(value: Any) -> None:
    """Validate one semantic action affordance."""

    if not isinstance(value, Mapping) or set(value) != {
        "action_kind",
        "capability",
        "permission",
        "decision_mode",
        "allowed_decisions",
        "default_decision",
        "decision_pattern",
        "context_ref",
        "target_roles",
    }:
        raise CognitionContractError("action affordance fields are not exact")
    _require_text(value["action_kind"], "action affordance.action_kind")
    _require_text(value["capability"], "action affordance.capability")
    _require_text(value["permission"], "action affordance.permission")
    if value["decision_mode"] not in {"optional", "required_text", "closed"}:
        raise CognitionContractError("action affordance decision_mode is invalid")
    _validate_text_list(
        value["allowed_decisions"],
        "action affordance.allowed_decisions",
        allow_empty=True,
    )
    _require_bounded_text(
        value["default_decision"],
        "action affordance.default_decision",
        maximum=200,
    )
    _require_bounded_text(
        value["decision_pattern"],
        "action affordance.decision_pattern",
        maximum=200,
    )
    try:
        re.compile(value["decision_pattern"])
    except re.error as exc:
        raise CognitionContractError(
            "action affordance decision_pattern is invalid"
        ) from exc
    _require_bounded_text(
        value["context_ref"],
        "action affordance.context_ref",
        maximum=200,
    )
    if value["decision_mode"] == "closed" and not value["allowed_decisions"]:
        raise CognitionContractError(
            "closed action affordance requires allowed_decisions"
        )
    if (
        value["decision_mode"] == "closed"
        and value["default_decision"] not in value["allowed_decisions"]
    ):
        raise CognitionContractError(
            "closed action affordance default is outside allowed_decisions"
        )
    _validate_roles(value["target_roles"], "action affordance.target_roles")


def _validate_resolver_affordance(value: Any) -> None:
    """Validate one semantic resolver affordance."""

    if not isinstance(value, Mapping) or set(value) != {
        "capability",
        "semantic_capability",
        "availability",
    }:
        raise CognitionContractError("resolver affordance fields are not exact")
    _require_text(value["capability"], "resolver affordance.capability")
    _require_text(
        value["semantic_capability"],
        "resolver affordance.semantic_capability",
    )
    _require_text(value["availability"], "resolver affordance.availability")


def _validate_evidence_ref(value: Any) -> None:
    """Validate complete typed provenance."""

    required = {"source_kind", "source_id", "occurred_at", "semantic_summary"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("evidence_ref fields are not exact")
    if value["source_kind"] not in EVIDENCE_SOURCE_QUESTION_IDS:
        raise CognitionContractError("evidence_ref.source_kind is invalid")
    _require_text(value["source_id"], "evidence_ref.source_id")
    _require_utc_timestamp(value["occurred_at"], "evidence_ref.occurred_at")
    _require_text(value["semantic_summary"], "evidence_ref.semantic_summary")


def _validate_entity_ref(value: Any, label: str) -> None:
    """Validate one scope-qualified entity reference."""

    if not isinstance(value, Mapping) or set(value) != {
        "scope",
        "kind",
        "entity_id",
    }:
        raise CognitionContractError(f"{label} fields are not exact")
    if value["scope"] not in {"user", "character"}:
        raise CognitionContractError(f"{label}.scope is invalid")
    if value["kind"] not in ENTITY_KINDS:
        raise CognitionContractError(f"{label}.kind is invalid")
    _require_text(value["entity_id"], f"{label}.entity_id")


def _validate_roles(value: Any, label: str) -> None:
    """Validate structured semantic role references."""

    if not isinstance(value, list):
        raise CognitionContractError(f"{label} must be a list")
    for index, role in enumerate(value):
        if not isinstance(role, Mapping) or set(role) != {
            "role",
            "entity_kind",
            "entity_id",
        }:
            raise CognitionContractError(f"{label}[{index}] is invalid")
        if role["role"] not in ROLE_VALUES:
            raise CognitionContractError(f"{label}[{index}].role is invalid")
        if role["entity_kind"] not in ROLE_ENTITY_KINDS:
            raise CognitionContractError(
                f"{label}[{index}].entity_kind is invalid"
            )
        _require_text(role["entity_id"], f"{label}[{index}].entity_id")


def _validate_text_list(
    value: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> None:
    """Validate a list of unique non-empty strings."""

    if not isinstance(value, list) or (not allow_empty and not value):
        raise CognitionContractError(f"{label} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise CognitionContractError(f"{label} must contain text")
    if len(value) != len(set(value)):
        raise CognitionContractError(f"{label} must not contain duplicates")


def _validate_state_update(value: Mapping[str, Any]) -> None:
    """Validate the one-scope state-update envelope."""

    _require_exact_keys(
        value,
        {
            "state_scope",
            "owner_key",
            "replacement_state",
            "comparison_results",
            "changed_paths",
        },
        "state_update",
    )
    if value["state_scope"] not in {"user", "character"}:
        raise CognitionContractError("state update scope is invalid")
    _require_text(value["owner_key"], "state_update.owner_key")
    _validate_persistent_state(value["replacement_state"])
    replacement = value["replacement_state"]
    expected_owner = (
        replacement.get("owner_user_id")
        if replacement["state_scope"] == "user"
        else "global"
    )
    if (
        replacement["state_scope"] != value["state_scope"]
        or expected_owner != value["owner_key"]
    ):
        raise CognitionContractError("state update owner does not match state")
    if not isinstance(value["comparison_results"], list):
        raise CognitionContractError("comparison_results must be a list")
    for row in value["comparison_results"]:
        _validate_comparison_result(row)
    if not isinstance(value["changed_paths"], list):
        raise CognitionContractError("changed_paths must be a list")
    if any(not isinstance(path, str) or not path for path in value["changed_paths"]):
        raise CognitionContractError("changed_paths must contain text")
    if list(value["changed_paths"]) != sorted(set(value["changed_paths"])):
        raise CognitionContractError("changed_paths must be unique and sorted")


def _validate_comparison_result(value: Any) -> None:
    """Validate one deterministic causal comparison result."""

    required = {"current_event_ref", "outcome", "evidence_refs"}
    if "matched_entity_ref" in value:
        required.add("matched_entity_ref")
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("comparison result fields are not exact")
    _validate_entity_ref(
        value["current_event_ref"],
        "comparison result.current_event_ref",
    )
    if "matched_entity_ref" in value:
        _validate_entity_ref(
            value["matched_entity_ref"],
            "comparison result.matched_entity_ref",
        )
    if value["outcome"] not in {
        "create",
        "reinforce",
        "contradict",
        "resolve",
        "replace",
        "unrelated",
    }:
        raise CognitionContractError("comparison result outcome is invalid")
    if not isinstance(value["evidence_refs"], list):
        raise CognitionContractError("comparison result.evidence_refs is invalid")
    for evidence_ref in value["evidence_refs"]:
        _validate_evidence_ref(evidence_ref)


def _validate_character_constraints(value: Any) -> None:
    """Validate the read-only character constraint snapshot."""

    if not isinstance(value, Mapping) or set(value) != {
        "drives",
        "standards",
        "meaning_state",
    }:
        raise CognitionContractError("character constraints fields are not exact")
    drives = value["drives"]
    if not isinstance(drives, Mapping) or set(drives) != {
        "autonomy",
        "connection",
        "safety",
        "competence",
        "care",
        "integrity",
        "exploration",
        "meaning",
    }:
        raise CognitionContractError("character constraint drives are invalid")
    for drive_id, drive in drives.items():
        if not isinstance(drive, Mapping) or set(drive) != {
            "importance",
            "pressure",
        }:
            raise CognitionContractError(
                f"character constraint drive {drive_id} is invalid"
            )
        _require_axis(drive["importance"], f"{drive_id}.importance")
        _require_axis(drive["pressure"], f"{drive_id}.pressure")
    standards = value["standards"]
    if not isinstance(standards, list) or len(standards) > 16:
        raise CognitionContractError("character constraint standards are invalid")
    for standard in standards:
        if not isinstance(standard, Mapping) or set(standard) != {
            "standard_id",
            "description",
            "importance",
        }:
            raise CognitionContractError("character constraint standard is invalid")
        if standard["standard_id"] not in {
            "honesty",
            "avoid_harm",
            "respect_boundaries",
            "follow_through",
            "self_respect",
        }:
            raise CognitionContractError("character constraint standard id is invalid")
        _require_text(standard["description"], "standard.description")
        _require_axis(standard["importance"], "standard.importance")
    meaning = value["meaning_state"]
    allowed_meaning = {
        "purpose_coherence",
        "agency",
        "identity_continuity",
        "salience",
    }
    if isinstance(meaning, Mapping) and "low_coherence_since" in meaning:
        allowed_meaning.add("low_coherence_since")
    if not isinstance(meaning, Mapping) or set(meaning) != allowed_meaning:
        raise CognitionContractError("character constraint meaning state is invalid")
    for field_name in allowed_meaning - {"low_coherence_since"}:
        _require_axis(meaning[field_name], f"meaning_state.{field_name}")
    if "low_coherence_since" in meaning:
        _require_utc_timestamp(
            meaning["low_coherence_since"],
            "meaning_state.low_coherence_since",
        )


def _validate_direct_fact(value: Any) -> None:
    """Validate the exact trusted direct-fact envelope."""

    required = {
        "fact_id",
        "producer",
        "fact_kind",
        "target_refs",
        "evidence_ref",
    }
    if isinstance(value, Mapping) and "observed_progress" in value:
        required.add("observed_progress")
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("direct fact fields are not exact")
    _require_text(value["fact_id"], "direct fact.fact_id")
    if value["producer"] not in {
        "action_result",
        "resolver_observation",
        "tool_result",
        "scheduler_event",
        "promoted_source_metadata",
    }:
        raise CognitionContractError("direct fact producer is invalid")
    if value["fact_kind"] not in {
        "goal_progress_observed",
        "goal_completed",
        "goal_terminal_failure",
        "goal_obstruction_removed",
        "threat_resolved",
        "event_repaired",
        "knowledge_answered",
        "deadline_reached",
        "source_occurred",
    }:
        raise CognitionContractError("direct fact kind is invalid")
    target_refs = value["target_refs"]
    if not isinstance(target_refs, list) or not 1 <= len(target_refs) <= 8:
        raise CognitionContractError("direct fact target_refs are invalid")
    for target_ref in target_refs:
        if not isinstance(target_ref, Mapping):
            raise CognitionContractError("direct fact target ref is invalid")
        if set(target_ref) == {"scope", "kind", "entity_id"}:
            _validate_entity_ref(target_ref, "direct fact target ref")
        else:
            _validate_roles([target_ref], "direct fact target refs")
    _validate_evidence_ref(value["evidence_ref"])
    if "observed_progress" in value:
        _require_axis(value["observed_progress"], "direct fact.observed_progress")


def _validate_scene_context(value: Any) -> None:
    """Validate semantic scene context without platform metadata."""

    required = {
        "channel_scope",
        "character_role",
        "semantic_scene",
        "conversation_continuity",
        "semantic_temporal_context",
    }
    if isinstance(value, Mapping) and "current_user_role" in value:
        required.add("current_user_role")
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("scene context fields are not exact")
    if value["channel_scope"] not in {"private", "group", "internal"}:
        raise CognitionContractError("scene context channel_scope is invalid")
    for field_name in required - {"channel_scope", "conversation_continuity"}:
        _require_text(value[field_name], f"scene context.{field_name}")
    _require_bounded_text(
        value["conversation_continuity"],
        "scene context.conversation_continuity",
        maximum=1000,
    )


def _validate_canonical_episode(value: Any) -> CognitiveEpisodeV1:
    """Validate the frozen episode contract and translate its public error."""

    required = {
        "schema_version",
        "episode_id",
        "trigger_source",
        "origin_metadata",
        "target_scope",
        "percepts",
        "evidence_refs",
        "created_at",
        "privacy_scope",
        "continuation_depth",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("cognitive episode fields are not exact")
    try:
        validate_cognitive_episode_v1(value)
    except CognitiveEpisodeValidationError as exc:
        raise CognitionContractError(str(exc)) from exc
    return dict(value)  # type: ignore[return-value]


def _validate_relationship_context(
    value: Any,
    *,
    scope: str,
    state: Mapping[str, Any],
    episode: CognitiveEpisodeV1,
) -> None:
    """Delegate optional relationship validation to the native state owner."""

    if scope == "user":
        owner_user_id = state.get("owner_user_id")
    else:
        owner_user_id = episode["target_scope"]["current_global_user_id"]
    if not isinstance(owner_user_id, str) or not owner_user_id.strip():
        raise CognitionContractError(
            "relationship context requires an authorized target user"
        )
    try:
        validate_relationship_state(
            value,
            owner_user_id=owner_user_id,
        )
    except CognitionStateError as exc:
        raise CognitionContractError(str(exc)) from exc


def _validate_surface_bid(value: Any) -> None:
    """Validate the exact public bid projection allowed into L3."""

    required = {
        "motive",
        "intention",
        "desired_outcome",
        "permitted_detail",
        "target_summaries",
        "expected_consequences",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("surface bid fields are not exact")
    for field_name in (
        "motive",
        "intention",
        "desired_outcome",
        "permitted_detail",
    ):
        _require_text(value[field_name], f"surface bid.{field_name}", maximum=1000)
    _validate_text_list(
        value["target_summaries"],
        "surface bid.target_summaries",
        allow_empty=True,
    )
    _validate_text_list(
        value["expected_consequences"],
        "surface bid.expected_consequences",
        allow_empty=True,
    )


def _validate_action_result(value: Any) -> None:
    """Validate one permitted semantic action result for L3."""

    required = {"action_kind", "status", "semantic_result", "target_roles"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("surface action result fields are not exact")
    _require_text(value["action_kind"], "surface action result.action_kind")
    if value["status"] not in {
        "executed",
        "scheduled",
        "pending",
        "failed",
        "unavailable",
    }:
        raise CognitionContractError("surface action result.status is invalid")
    _require_text(value["semantic_result"], "surface action result.semantic_result")
    _validate_roles(value["target_roles"], "surface action result.target_roles")


def _require_axis(value: Any, label: str) -> None:
    """Require one non-boolean integer causal axis in the native range."""

    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise CognitionContractError(f"{label} is invalid")


def _require_utc_timestamp(value: Any, label: str) -> None:
    """Require an ISO-8601 UTC timestamp ending in Z."""

    if not isinstance(value, str) or not value.endswith("Z"):
        raise CognitionContractError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CognitionContractError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise CognitionContractError(f"{label} is invalid")


def _require_exact_keys(
    value: Mapping[str, Any],
    required: set[str],
    label: str,
) -> None:
    """Reject missing and extra fields at a public V2 boundary."""

    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError(f"{label} fields are not exact")


def _require_text(value: Any, label: str, maximum: int = 500) -> None:
    """Require bounded non-empty semantic text."""

    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise CognitionContractError(f"{label} is invalid")


def _validate_pending_resolver_resume(value: object) -> None:
    """Validate deterministic pending state without an import cycle."""

    from kazusa_ai_chatbot.cognition_resolver.contracts import (
        ResolverValidationError,
        validate_resolver_pending_resume,
    )

    try:
        validate_resolver_pending_resume(value)
    except ResolverValidationError as exc:
        raise CognitionContractError(
            f"pending_resolver_resume is invalid: {exc}"
        ) from exc


def _validate_resolver_goal_progress_input(value: object) -> None:
    """Validate protocol-owned goal state without an import cycle."""

    from kazusa_ai_chatbot.cognition_resolver.contracts import (
        ResolverValidationError,
        validate_resolver_goal_progress,
    )

    try:
        validate_resolver_goal_progress(value)
    except ResolverValidationError as exc:
        raise CognitionContractError(
            f"resolver_goal_progress is invalid: {exc}"
        ) from exc


def _validate_resolver_lifecycle_output(
    pending_resolution: object,
    goal_progress: object,
) -> None:
    """Validate canonical resolver lifecycle rows without an import cycle."""

    from kazusa_ai_chatbot.cognition_resolver.contracts import (
        ResolverValidationError,
        validate_resolver_goal_progress,
        validate_resolver_pending_resolution,
    )

    try:
        if pending_resolution is not None:
            validate_resolver_pending_resolution(pending_resolution)
        if goal_progress is not None:
            validate_resolver_goal_progress(goal_progress)
    except ResolverValidationError as exc:
        raise CognitionContractError(
            f"resolver lifecycle output is invalid: {exc}"
        ) from exc


def _require_bounded_text(value: Any, label: str, maximum: int) -> None:
    """Require a bounded string while allowing an empty semantic window."""

    if not isinstance(value, str) or len(value) > maximum:
        raise CognitionContractError(f"{label} is invalid")
