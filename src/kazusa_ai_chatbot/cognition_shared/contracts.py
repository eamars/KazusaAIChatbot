"""Internal contracts for the validation-local cognition core."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeV1,
    CognitiveEpisodeValidationError,
    DialogResponseOperation,
    GoalContinuationRefV1,
    validate_cognitive_episode_v1,
    validate_dialog_response_operation,
    validate_goal_continuation_ref,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    MAX_RESOLVER_EVIDENCE_EXCERPT_CHARS,
    MAX_RESOLVER_EVIDENCE_EXCERPTS,
    RESOLVER_EVIDENCE_STATE_VERSION,
    RequiredResolverEvidenceDependencyV1,
    ResolverValidationError,
    validate_resolver_evidence_state,
)
from kazusa_ai_chatbot.cognition_shared.prompt_budget import (
    CHARACTER_OPERATIONAL_CONSUMER_ROLES,
    CHARACTER_OPERATIONAL_CONTEXT_DIGEST_CHARS,
    CHARACTER_OPERATIONAL_ROOT_KINDS,
    MAX_CHARACTER_OPERATIONAL_CONTEXT_CHARS,
    MAX_CONTEXT_AFFECT_ROWS,
    MAX_CONTEXT_PRESSURE_ROWS,
    MAX_RELATIONSHIP_AFFECT_ROWS,
    MAX_RELATIONSHIP_CAUSAL_ROWS,
    MAX_RELATIONSHIP_CAUSAL_SUMMARY_CHARS,
    MAX_RELATIONSHIP_OPERATIONAL_CONTEXT_CHARS,
    MAX_SCENE_PARTICIPANT_BINDINGS,
    OPERATIONAL_CAUSE_CLASSES,
    RELATIONSHIP_AFFECT_PHASES,
    RELATIONSHIP_CAUSAL_ENTITY_KINDS,
    canonical_digest,
    serialized_character_count,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    CognitionStateError,
    validate_cognition_state,
    validate_relationship_state,
)
from kazusa_ai_chatbot.config import (
    CHARACTER_TIME_ZONE,
    L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMInvoker
from kazusa_ai_chatbot.time_boundary import (
    local_llm_datetime_to_storage_utc_iso,
    local_time_context_from_storage_utc,
    parse_storage_utc_datetime,
)

_CJK_IDEOGRAPH_RE = re.compile(r"[\u4e00-\u9fff]")
SCENE_PARTICIPANT_HANDLE_RE = re.compile(r"^p[1-9][0-9]*$")
SURFACE_ROLE_HANDLE_RE = re.compile(r"^(?:current_user|self|p[1-9][0-9]*)$")

SELF_COGNITION_RESPONSE_DECISION_VALUES = frozenset({
    "stay_silent",
    "propose_visible_reply",
})
SELF_COGNITION_RESPONSE_PARTICIPATION_VALUES = frozenset({
    "",
    "direct_address",
    "explicit_character_reference",
    "grounded_scene_intervention",
})
SELF_COGNITION_RESPONSE_CONTRACT_STATUS_VALUES = frozenset({
    "not_required",
    "valid",
    "failed",
})
SELF_COGNITION_RESPONSE_EVIDENCE_LIMIT = 4
SELF_COGNITION_RESPONSE_TEXT_LIMIT = 300


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




class CognitionContractError(ValueError):
    """Raised when a public cognition boundary is structurally invalid."""

    def __init__(
        self,
        message: str,
        *,
        field_path: str | None = None,
    ) -> None:
        """Attach an optional prompt-safe path to the contract failure."""

        super().__init__(message)
        self.field_path = (
            field_path
            if isinstance(field_path, str) and field_path
            else None
        )


class CognitionExecutionError(CognitionContractError):
    """Raised when cognition execution cannot produce a valid result."""

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


def classify_cognition_failure(exception: BaseException) -> str:
    """Return the bounded error category for one cognition-stage failure."""

    if isinstance(exception, CognitionExecutionError):
        return exception.error_code
    if isinstance(exception, CognitionContextLimitError):
        return "context_limit"
    if isinstance(exception, (TimeoutError, ConnectionError)):
        return "provider_transient"
    if isinstance(exception, ValueError):
        return "model_contract_invalid"
    return "internal_invariant"


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

PAST_DIALOG_COGNITION_CONTEXT_MAX_CHARS = 1800
GROUP_ENGAGEMENT_GUIDELINE_MAX_CHARS = 120
GROUP_ENGAGEMENT_CONFIDENCE_MAX_CHARS = 80
RELATIONAL_WILLINGNESS_MAX_TEXT_CHARS = 300
MAX_RECENT_CHARACTER_DIALOG_ROWS = 2
MAX_RECENT_CHARACTER_DIALOG_CHARS = 600
MAX_LEXICAL_AVOIDANCES = 8
MAX_LEXICAL_AVOIDANCE_CHARS = 120
RELATIONAL_PROVENANCE_ROLE_VALUES = frozenset({
    "current_episode",
    "current_user_history_only",
    "character_or_world_context_only",
    "contextual_fact_only",
})

SCHEDULED_AUTHORITY_PROPOSAL_SCHEMA_VERSION = "scheduled_authority_proposal.v1"
SCHEDULED_FUTURE_SPEECH_AUTHORITY_SCHEMA_VERSION = (
    "scheduled_future_speech_authority.v1"
)
SCHEDULED_AUTHORITY_CARRIER_SCHEMA_VERSION = "scheduled_authority_carrier.v1"

TEMPORAL_ALIGNMENT_VALUES = frozenset({
    "aligned",
    "relative_date_mismatch",
    "past_or_not_future",
    "timezone_unclear",
    "unavailable",
})
SCHEDULED_AUTHORITY_CURRENT_ROLE_VALUES = frozenset({
    "current_event",
    "public_scene",
})
SCHEDULED_AUTHORITY_DETAIL_REF_LIMIT = 8
SCHEDULED_AUTHORITY_SUMMARY_MAX_CHARS = 1000
SCHEDULED_AUTHORITY_DETAIL_SUMMARY_MAX_CHARS = 1000
SCHEDULED_AUTHORITY_OBJECTIVE_MAX_CHARS = 2000
SCHEDULED_AUTHORITY_ID_PREFIX = "sha256-"

SCHEDULED_SPEECH_GATE_CODES = frozenset((
    "scheduled_authority_missing",
    "scheduled_authority_invalid",
    "scheduled_trigger_identity_mismatch",
    "scheduled_due_not_reached",
    "scheduled_candidate_empty",
))

CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS = frozenset({
    "episode",
    "scheduler_event",
    "tool_result",
})
MEMORY_SCOPE_VALUES = frozenset({
    "current_user_continuity",
    "shared_character_or_world",
})

CognitionEvidenceAuthority = Literal[
    "current_event",
    "public_scene",
    "participant_continuity",
    "private_motive_only",
    "character_world_context",
    "conditional_character_guidance",
    "contextual_fact_only",
]
COGNITION_EVIDENCE_AUTHORITY_VALUES = frozenset({
    "current_event",
    "public_scene",
    "participant_continuity",
    "private_motive_only",
    "character_world_context",
    "conditional_character_guidance",
    "contextual_fact_only",
})

EVIDENCE_SOURCE_QUESTION_IDS = {
    "episode": tuple(f"q:{kind}" for kind in SEMANTIC_QUESTION_KINDS),
    "promoted_memory": tuple(f"q:{kind}" for kind in SEMANTIC_QUESTION_KINDS),
    "promoted_reflection": tuple(f"q:{kind}" for kind in SEMANTIC_QUESTION_KINDS),
    "media_observation": tuple(f"q:{kind}" for kind in SEMANTIC_QUESTION_KINDS),
    "conversation_evidence": tuple(
        f"q:{kind}" for kind in SEMANTIC_QUESTION_KINDS
    ),
    "recall_evidence": tuple(
        f"q:{kind}" for kind in SEMANTIC_QUESTION_KINDS
    ),
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
    authority: CognitionEvidenceAuthority
    temporal_provenance: NotRequired[dict[str, str]]
    memory_scope: NotRequired[
        Literal[
            "current_user_continuity",
            "shared_character_or_world",
        ]
    ]


class ScheduledAuthorityDetailRefV1(TypedDict):
    """One bounded authorized-detail reference in a scheduled proposal."""

    evidence_handle: str
    semantic_summary: str
    provenance_role: str


class ScheduledAuthorityProposalV1(TypedDict):
    """Closed planner-owned authority proposal for a future-speak action row.

    The proposal is a transient model-authored contract, never a durable
    record. Deterministic code validates the closed temporal alignment,
    bounded summaries, evidence-handle coverage, and actual evidence role
    before the proposal can travel toward persistence.
    """

    schema_version: Literal["scheduled_authority_proposal.v1"]
    temporal_alignment: Literal[
        "aligned",
        "relative_date_mismatch",
        "past_or_not_future",
        "timezone_unclear",
        "unavailable",
    ]
    authorized_content_summary: str
    authorized_detail_refs: list[ScheduledAuthorityDetailRefV1]


class ScheduledAuthoritySourceIdentityV1(TypedDict):
    """Immutable source identity that the scheduled authority records."""

    source_episode_id: str
    source_message_id: str
    source_action_attempt_id: str
    source_llm_trace_id: NotRequired[str]


class ScheduledAuthorityTimestampV1(TypedDict):
    """Storage UTC plus configured-local wall-clock for one authority instant."""

    utc: str
    local: str
    timezone: str


class ScheduledAuthorityTargetV1(TypedDict):
    """Target class carried by the scheduled authority without delivery ids."""

    platform: str
    channel_type: str
    audience_kind: Literal["group", "private"]


class ScheduledAuthorityAuthorizedContentV1(TypedDict):
    """Bounded semantic content that the scheduled speech may express."""

    summary: str
    detail_refs: list[ScheduledAuthorityDetailRefV1]


class ScheduledFutureSpeechAuthorityV1(TypedDict):
    """Immutable pre-persistence authority for one future-speak task.

    The authority is created before accepted-task, job, schedule, or run ids
    exist. Carrier records copy it unchanged; a changed trigger or objective
    must create a new authority through the normal deterministic path.
    """

    schema_version: Literal["scheduled_future_speech_authority.v1"]
    authority_id: str
    source: ScheduledAuthoritySourceIdentityV1
    accepted_at: ScheduledAuthorityTimestampV1
    trigger: ScheduledAuthorityTimestampV1
    target: ScheduledAuthorityTargetV1
    semantic_objective: str
    authorized_content: ScheduledAuthorityAuthorizedContentV1
    goal_continuation_ref: GoalContinuationRefV1 | None


class ScheduledAuthorityCarrierV1(TypedDict, total=False):
    """Later persistence envelope carrying the exact authority unchanged."""

    schema_version: Literal["scheduled_authority_carrier.v1"]
    authority: ScheduledFutureSpeechAuthorityV1
    accepted_task_id: str
    background_job_id: str
    calendar_schedule_id: str
    calendar_run_id: str
    child_llm_trace_id: str
    delivery_tracking_id: str


class RelationalWillingness(TypedDict):
    """Current-turn relational stance selected by cognition."""

    applicable: bool
    stance: str
    reason: str
    cause_summary: str


def project_evidence_provenance_role(
    source_kind: str,
    memory_scope: object,
) -> str:
    """Map trusted evidence metadata to a transient model-facing authority role.

    The role is derived only from validated source-kind and memory-scope
    metadata. Unknown provenance fails closed with a contract error so no
    free-text inference can assign authority.

    Args:
        source_kind: Validated ``evidence_ref.source_kind`` value.
        memory_scope: Optional validated ``memory_scope`` value carried by
            promoted-memory evidence rows.

    Returns:
        One transient provenance-role label from
        ``RELATIONAL_PROVENANCE_ROLE_VALUES``.

    Raises:
        CognitionContractError: When the metadata cannot be mapped.
    """

    if source_kind in CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS:
        return "current_episode"
    if source_kind == "promoted_memory":
        if memory_scope == "current_user_continuity":
            return "current_user_history_only"
        if memory_scope == "shared_character_or_world":
            return "character_or_world_context_only"
        raise CognitionContractError(
            "promoted memory evidence requires a trusted memory scope"
        )
    if source_kind == "promoted_reflection":
        return "character_or_world_context_only"
    if source_kind in EVIDENCE_SOURCE_QUESTION_IDS:
        return "contextual_fact_only"
    raise CognitionContractError(
        "evidence source kind is unsupported"
    )


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


class SceneParticipantBindingV1(TypedDict):
    """Prompt-safe episode-local binding for one visible scene participant.

    Bindings are derived deterministically from the already resolved scene
    roster, scoped to one cognitive episode, and contain only the episode-local
    handle and the exact visible display name. Platform, global, and database
    identifiers never enter these rows or any model payload.
    """

    handle: str
    display_name: str
    entity_kind: Literal["third_party"]


class SceneContextV2(TypedDict):
    """Prompt-safe scene context without platform identifiers."""

    channel_scope: Literal["private", "group", "internal"]
    character_role: str
    current_user_role: NotRequired[str]
    character_sleep_phase: NotRequired[str]
    semantic_scene: str
    public_group_scene: str
    conversation_continuity: str
    semantic_temporal_context: str
    participant_bindings: NotRequired[list[SceneParticipantBindingV1]]


class SelfCognitionResponseDecisionV1(TypedDict):
    """Semantic group self-cognition decision before route derivation."""

    decision: Literal["stay_silent", "propose_visible_reply"]
    evidence_handles: list[str]
    semantic_target_handle: str
    participation_basis: Literal[
        "",
        "direct_address",
        "explicit_character_reference",
        "grounded_scene_intervention",
    ]
    response_goal: str
    reason: str


class GroupEngagementActionContextV2(TypedDict):
    """Bounded advisory participation guidance for one group scene.

    ``confidence`` is a bounded semantic confidence descriptor, not a numeric
    score or a ranking signal.
    """

    engagement_guidelines: list[str]
    confidence: str


class PersonalityJudgmentV2(TypedDict):
    """Trusted semantic character descriptors used during cognition."""

    logic: str
    defense: str
    quirks: str
    taboos: str


class CharacterConstraintSnapshotV2(TypedDict):
    """Read-only character constraints supplied to user-scope appraisal."""

    drives: dict[str, dict[str, Any]]
    standards: list[dict[str, Any]]
    meaning_state: dict[str, Any]
    personality_judgment: PersonalityJudgmentV2


class CharacterOperationalContextV1(TypedDict):
    """Bounded redacted character posture selected for one V2 consumer."""

    schema_version: Literal["character_operational_context.v1"]
    source_updated_at: str
    effective_at: str
    view_digest: str
    context_digest: str
    consumer_role: Literal[
        "settled_relevance",
        "appraisal branch",
        "goal",
        "surface",
    ]
    affect: list[dict[str, str]]
    pressures: list[dict[str, str]]


class RelationshipOperationalContextV1(TypedDict):
    """Bounded current-user relationship projection for model consumption."""

    schema_version: Literal["relationship_operational_context.v1"]
    relationship_id: str
    axes: dict[str, int]
    causal_context: list[dict[str, str]]
    affect: list[dict[str, str]]
    relationship_freshness: str
    evidence_freshness: str


class SemanticQuestionV2(TypedDict):
    """One bounded semantic question owned by one appraisal family."""

    question_id: str
    question_kind: str
    semantic_question: str
    evidence_handles: list[str]
    permitted_role_handles: list[str]
    permitted_role_assignment_handles: list[str]
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


RelationshipAxisV2 = Literal[
    "positive_regard",
    "trust",
    "attachment",
    "desired_closeness",
    "perceived_closeness",
    "care",
    "boundary_safety",
    "exclusivity",
    "unresolved_injury",
]


class SemanticDeltaReceiptV2(TypedDict):
    """Authoritative result for one unique semantic delta target."""

    target_path: str
    relationship_axis: RelationshipAxisV2 | None
    requested_delta: int
    applied_delta: int
    previous_value: int
    next_value: int
    evidence_refs: list[CognitionEvidenceV2]
    duplicate_disposition: Literal["unique"]


class SemanticDeltaRejectionReceiptV2(TypedDict):
    """Deterministic rejection receipt for a duplicate semantic target."""

    target_path: str
    disposition: Literal["duplicate_target"]


class SemanticDeltaApplicationResultV2(TypedDict):
    """Native state and receipts returned by the semantic reducer."""

    updated_state: dict[str, Any]
    accepted_delta_receipts: list[SemanticDeltaReceiptV2]
    rejected_delta_receipts: list[SemanticDeltaRejectionReceiptV2]


class SemanticActionRequestV2(TypedDict):
    """Planner-selected action request; execution remains action-spec owned."""

    action_kind: str
    decision: str
    context_ref: str
    semantic_goal: str
    reason: str
    target_roles: list[RoleRefV2]
    evidence_handles: list[str]
    scheduled_authority_proposal: NotRequired[ScheduledAuthorityProposalV1]


class ResolverCapabilityRequestV2(TypedDict):
    """Planner-selected resolver request; execution remains resolver owned."""

    capability: str
    semantic_goal: str
    reason: str
    evidence_handles: list[str]
    start_in_background: NotRequired[bool]
    goal_continuation_ref: GoalContinuationRefV1 | None


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
    expected_previous_state: dict[str, Any]
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


class SurfaceResolverResultV2(TypedDict):
    """Prompt-safe capability outcome available to the surface planner."""

    capability_kind: str
    status: Literal["succeeded", "blocked", "failed"]
    semantic_result: str
    prompt_safe_observation_handle: NotRequired[str]
    evidence_state: NotRequired[
        Literal["complete", "partial", "pending", "missing", "blocked"]
    ]
    evidence_excerpts: NotRequired[list[str]]
    evidence_handles: NotRequired[list[str]]
    remaining_needs: NotRequired[list[str]]


class CharacterExpressionContextV2(TypedDict):
    """Delivery-only character context exposed to text planning."""

    tempo: str
    linguistic_texture: str


class SubjectiveExpressionContextV1(TypedDict):
    """Exact cognition-owned subjectivity available to text planning."""

    private_monologue: str
    epistemic_boundary: str


class DeliveryProfileV2(TypedDict):
    """Bounded delivery dimensions owned atomically with surface content."""

    lexical_register: str
    sentence_shape: str
    rhythm: str
    hesitation: str
    punctuation: str


class SurfaceAddresseePlanV1(TypedDict):
    """Structured prompt-safe addressee and clause-target projection.

    Rows carry the episode-local handle, the exact visible display name, the
    semantic role the participant plays in the planned wording, and the
    deterministic wording policy the wording owners must follow. The current
    user stays the transport/direct-message recipient and permits second
    person only when the current user is the intended clause target; a typed
    third-party target requires its visible name or an unambiguous
    third-person expression.
    """

    handle: str
    display_name: str
    semantic_role: Literal[
        "direct_recipient",
        "embedded_target",
        "embedded_actor",
        "observer",
    ]
    wording_policy: Literal[
        "second_person_allowed",
        "named_or_third_person_required",
    ]


class TextSurfaceOutputV2(TypedDict):
    """Public V2 text-surface output contract."""

    schema_version: Literal["text_surface_output.v2"]
    content_plan: str
    content_requirements: list[str]
    epistemic_boundary: str
    visible_boundaries: list[str]
    addressee_plan: list[SurfaceAddresseePlanV1]
    delivery_profile: DeliveryProfileV2
    selected_surface_intent: str
    permitted_action_results: list[SemanticActionResultV2]
    lexical_avoidances: NotRequired[list[str]]
    relational_willingness: NotRequired[RelationalWillingness]
    resolver_result: NotRequired[SurfaceResolverResultV2]
    runtime_capability_limits: NotRequired[list[str]]


class VisualSurfaceOutputV2(TypedDict):
    """Public V2 terminal visual-surface output contract."""

    schema_version: Literal["visual_surface_output.v2"]
    visual_directives: str
    selected_surface_intent: str


@dataclass(frozen=True)
class TextSurfaceServicesV2:
    """Injected single-stage V2 text-surface binding."""

    llm: LLMInvoker
    content_plan_config: LLMCallConfig


@dataclass(frozen=True)
class VisualSurfaceServicesV2:
    """Injected terminal V2 visual-surface binding."""

    llm: LLMInvoker
    visual_config: LLMCallConfig


def is_targetless_group_self_cognition_episode(
    episode: Mapping[str, Any],
) -> bool:
    """Identify the targetless group self-cognition route boundary."""

    target_scope = episode.get("target_scope")
    if not isinstance(target_scope, Mapping):
        return False
    if episode.get("trigger_source") != "self_cognition":
        return False
    if target_scope.get("channel_type") != "group":
        return False
    return not any(
        isinstance(target_scope.get(field_name), str)
        and target_scope[field_name].strip()
        for field_name in (
            "current_global_user_id",
            "current_platform_user_id",
        )
    )


def validate_self_cognition_response_decision(
    value: Any,
    *,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    target_handles: Sequence[str] | None = None,
) -> SelfCognitionResponseDecisionV1:
    """Validate one targetless group self-cognition semantic decision."""

    _require_exact_keys(
        value,
        {
            "decision",
            "evidence_handles",
            "semantic_target_handle",
            "participation_basis",
            "response_goal",
            "reason",
        },
        "self-cognition response",
    )
    decision = value["decision"]
    if decision not in SELF_COGNITION_RESPONSE_DECISION_VALUES:
        raise CognitionContractError(
            "self-cognition response decision is invalid"
        )
    evidence_handles = value["evidence_handles"]
    if (
        not isinstance(evidence_handles, list)
        or len(evidence_handles) > SELF_COGNITION_RESPONSE_EVIDENCE_LIMIT
        or any(
            not isinstance(handle, str) or not handle.strip()
            for handle in evidence_handles
        )
        or len(evidence_handles) != len(set(evidence_handles))
    ):
        raise CognitionContractError(
            "self-cognition response evidence handles are invalid"
        )
    available_evidence = {
        row.get("evidence_handle")
        for row in evidence or ()
        if isinstance(row, Mapping)
    }
    if evidence is not None and any(
        handle not in available_evidence for handle in evidence_handles
    ):
        raise CognitionContractError(
            "self-cognition response evidence handle is unavailable"
        )
    _require_bounded_text(
        value["semantic_target_handle"],
        "self-cognition response.semantic_target_handle",
        maximum=120,
    )
    participation_basis = value["participation_basis"]
    if participation_basis not in SELF_COGNITION_RESPONSE_PARTICIPATION_VALUES:
        raise CognitionContractError(
            "self-cognition response participation basis is invalid"
        )
    _require_bounded_text(
        value["response_goal"],
        "self-cognition response.response_goal",
        maximum=SELF_COGNITION_RESPONSE_TEXT_LIMIT,
    )
    _require_text(
        value["reason"],
        "self-cognition response.reason",
        maximum=SELF_COGNITION_RESPONSE_TEXT_LIMIT,
    )
    if decision == "stay_silent":
        if participation_basis != "" or value["response_goal"] != "":
            raise CognitionContractError(
                "silent self-cognition response must not carry participation "
                "or response-goal content"
            )
        return dict(value)  # type: ignore[return-value]
    if not evidence_handles:
        raise CognitionContractError(
            "visible self-cognition proposal requires evidence"
        )
    if evidence is not None and not any(
        row.get("evidence_handle") in evidence_handles
        and row.get("evidence_ref", {}).get("source_kind")
        in CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS
        for row in evidence
        if isinstance(row, Mapping)
        and isinstance(row.get("evidence_ref"), Mapping)
    ):
        raise CognitionContractError(
            "visible self-cognition proposal requires current-episode evidence"
        )
    semantic_target_handle = value["semantic_target_handle"]
    if not semantic_target_handle:
        raise CognitionContractError(
            "visible self-cognition proposal requires a target"
        )
    if semantic_target_handle not in {"self", "current_group_scene"} and not (
        isinstance(semantic_target_handle, str)
        and SCENE_PARTICIPANT_HANDLE_RE.fullmatch(semantic_target_handle)
    ):
        raise CognitionContractError(
            "visible self-cognition proposal target is invalid"
        )
    if target_handles is not None:
        supplied_targets = set(target_handles)
        if semantic_target_handle not in supplied_targets:
            raise CognitionContractError(
                "visible self-cognition proposal target is unavailable"
            )
    if participation_basis == "":
        raise CognitionContractError(
            "visible self-cognition proposal requires participation basis"
        )
    if not value["response_goal"].strip():
        raise CognitionContractError(
            "visible self-cognition proposal requires response goal"
        )
    return dict(value)  # type: ignore[return-value]


def validate_text_surface_output(
    payload: Mapping[str, Any],
) -> TextSurfaceOutputV2:
    """Validate the bounded V2 L3 output."""

    required = {
        "schema_version",
        "content_plan",
        "content_requirements",
        "epistemic_boundary",
        "visible_boundaries",
        "addressee_plan",
        "delivery_profile",
        "selected_surface_intent",
        "permitted_action_results",
    }
    optional = (
        (
            {"runtime_capability_limits"}
            if "runtime_capability_limits" in payload
            else set()
        )
        | (
            {"relational_willingness"}
            if "relational_willingness" in payload
            else set()
        )
        | (
            {"lexical_avoidances"}
            if "lexical_avoidances" in payload
            else set()
        )
        | ({"resolver_result"} if "resolver_result" in payload else set())
    )
    _require_exact_keys(payload, required | optional, "text surface output")
    if payload["schema_version"] != "text_surface_output.v2":
        raise CognitionContractError("unsupported text surface output schema")
    for field_name in (
        "content_plan",
        "epistemic_boundary",
        "selected_surface_intent",
    ):
        _require_text(payload[field_name], field_name, maximum=1000)
    _validate_delivery_profile(payload["delivery_profile"])
    requirements = payload["content_requirements"]
    if not isinstance(requirements, list) or not 1 <= len(requirements) <= 8:
        raise CognitionContractError("content_requirements must contain 1-8 items")
    if len(requirements) != len(set(requirements)):
        raise CognitionContractError("content_requirements contains duplicates")
    for index, item in enumerate(requirements):
        _require_text(item, f"content_requirements[{index}]", maximum=500)
    if "lexical_avoidances" in payload:
        validate_lexical_avoidances(payload["lexical_avoidances"])
    for field_name in ("visible_boundaries", "addressee_plan"):
        items = payload[field_name]
        if not isinstance(items, list) or len(items) > 8:
            raise CognitionContractError(
                f"{field_name} must contain 0-8 items"
            )
        if field_name == "visible_boundaries":
            if items:
                raise CognitionContractError(
                    "visible_boundaries must remain empty until a typed source contract exists"
                )
            if len(items) != len(set(items)):
                raise CognitionContractError(
                    f"{field_name} contains duplicates"
                )
            for index, item in enumerate(items):
                _require_text(
                    item,
                    f"{field_name}[{index}]",
                    maximum=500,
                )
        else:
            validate_surface_addressee_plan(items)
    action_results = payload["permitted_action_results"]
    if not isinstance(action_results, list):
        raise CognitionContractError(
            "permitted_action_results must be a list"
        )
    for row in action_results:
        _validate_action_result(row)
    if "relational_willingness" in payload:
        validate_relational_willingness(payload["relational_willingness"])
    if "resolver_result" in payload:
        _validate_surface_resolver_result(payload["resolver_result"])
    _validate_runtime_capability_limits(payload)
    return dict(payload)  # type: ignore[return-value]


def _validate_recent_character_dialog(value: Any) -> None:
    """Validate the bounded recent visible-character wording projection."""

    if (
        not isinstance(value, list)
        or len(value) > MAX_RECENT_CHARACTER_DIALOG_ROWS
    ):
        raise CognitionContractError(
            "recent_character_dialog must contain 0-2 items"
        )
    for index, item in enumerate(value):
        _require_text(
            item,
            f"recent_character_dialog[{index}]",
            maximum=MAX_RECENT_CHARACTER_DIALOG_CHARS,
        )


def validate_lexical_avoidances(value: Any) -> list[str]:
    """Validate current-turn expression-only wording avoidances."""

    if not isinstance(value, list) or len(value) > MAX_LEXICAL_AVOIDANCES:
        raise CognitionContractError(
            "lexical_avoidances must contain 0-8 items"
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise CognitionContractError("lexical_avoidances must contain text")
    if len(value) != len(set(value)):
        raise CognitionContractError("lexical_avoidances contains duplicates")
    for index, item in enumerate(value):
        _require_text(
            item,
            f"lexical_avoidances[{index}]",
            maximum=MAX_LEXICAL_AVOIDANCE_CHARS,
        )
    return list(value)


def _validate_delivery_profile(value: Any) -> None:
    """Validate the exact delivery-only dimensions of a text surface."""

    fields = {
        "lexical_register",
        "sentence_shape",
        "rhythm",
        "hesitation",
        "punctuation",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CognitionContractError("delivery profile fields are not exact")
    for field_name in fields:
        _require_text(
            value[field_name],
            f"delivery_profile.{field_name}",
            maximum=200,
        )


def _validate_runtime_capability_limits(
    payload: Mapping[str, Any],
) -> None:
    """Validate trusted runtime limits projected into surface prompts."""

    if "runtime_capability_limits" not in payload:
        return
    limits = payload["runtime_capability_limits"]
    if not isinstance(limits, list) or len(limits) > 8:
        raise CognitionContractError(
            "runtime_capability_limits must contain 0-8 items"
        )
    for index, item in enumerate(limits):
        _require_text(
            item,
            f"runtime_capability_limits[{index}]",
            maximum=500,
        )


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


def validate_relational_willingness(
    value: object,
) -> RelationalWillingness:
    """Validate the current four-field relational-willingness record."""

    if not isinstance(value, Mapping):
        raise CognitionContractError(
            "relational willingness must be an object"
        )
    _require_exact_keys(
        value,
        {"applicable", "stance", "reason", "cause_summary"},
        "relational willingness",
    )
    if not isinstance(value["applicable"], bool):
        raise CognitionContractError(
            "relational willingness applicable must be a boolean",
            field_path="relational_willingness.applicable",
        )
    for field_name in ("stance", "reason", "cause_summary"):
        _require_text(
            value[field_name],
            f"relational willingness.{field_name}",
            maximum=RELATIONAL_WILLINGNESS_MAX_TEXT_CHARS,
        )
    return dict(value)  # type: ignore[return-value]


def validate_scheduled_authority_proposal(
    value: object,
    *,
    evidence: Sequence[Mapping[str, Any]] | None = None,
) -> ScheduledAuthorityProposalV1:
    """Validate the closed planner-owned authority proposal for future speak.

    Args:
        value: Model-authored proposal attached to a future-speak action row.
        evidence: Optional complete current evidence set. When supplied, every
            detail handle must exist and its declared provenance role must
            match the actual evidence authority; only current-episode roles
            admitted by the parent cognition may authorize scheduled content.

    Returns:
        A shallow validated copy of the proposal.

    Raises:
        CognitionContractError: When any field, enum, bound, handle, or
            provenance rule is violated.
    """

    if not isinstance(value, Mapping):
        raise CognitionContractError(
            "scheduled authority proposal must be an object"
        )
    _require_exact_keys(
        value,
        {
            "schema_version",
            "temporal_alignment",
            "authorized_content_summary",
            "authorized_detail_refs",
        },
        "scheduled authority proposal",
    )
    if (
        value["schema_version"]
        != SCHEDULED_AUTHORITY_PROPOSAL_SCHEMA_VERSION
    ):
        raise CognitionContractError(
            "scheduled authority proposal schema is invalid"
        )
    temporal_alignment = value["temporal_alignment"]
    if (
        not isinstance(temporal_alignment, str)
        or temporal_alignment not in TEMPORAL_ALIGNMENT_VALUES
    ):
        raise CognitionContractError(
            "scheduled authority temporal alignment is invalid"
        )
    _require_bounded_text(
        value["authorized_content_summary"],
        "scheduled authority proposal.authorized_content_summary",
        maximum=SCHEDULED_AUTHORITY_SUMMARY_MAX_CHARS,
    )
    detail_refs = _validate_scheduled_authority_detail_refs(
        value["authorized_detail_refs"],
        evidence=evidence,
    )
    validated_proposal: ScheduledAuthorityProposalV1 = {
        "schema_version": SCHEDULED_AUTHORITY_PROPOSAL_SCHEMA_VERSION,
        "temporal_alignment": temporal_alignment,
        "authorized_content_summary": value[
            "authorized_content_summary"
        ].strip(),
        "authorized_detail_refs": detail_refs,
    }
    return validated_proposal


def validate_scheduled_future_speech_authority(
    value: object,
) -> ScheduledFutureSpeechAuthorityV1:
    """Validate one immutable scheduled future-speech authority.

    The validator re-derives the deterministic authority id from the canonical
    payload and rejects any mutation, including carrier-local id changes.

    Args:
        value: Candidate authority carried by a durable record.

    Returns:
        A shallow validated copy of the authority.

    Raises:
        CognitionContractError: When any field, enum, bound, time relation, or
            identity digest is violated.
    """

    if not isinstance(value, Mapping):
        raise CognitionContractError(
            "scheduled future-speech authority must be an object"
        )
    _require_exact_keys(
        value,
        {
            "schema_version",
            "authority_id",
            "source",
            "accepted_at",
            "trigger",
            "target",
            "semantic_objective",
            "authorized_content",
            "goal_continuation_ref",
        },
        "scheduled future-speech authority",
    )
    if (
        value["schema_version"]
        != SCHEDULED_FUTURE_SPEECH_AUTHORITY_SCHEMA_VERSION
    ):
        raise CognitionContractError(
            "scheduled future-speech authority schema is invalid"
        )
    source = _validate_scheduled_authority_source(value["source"])
    accepted_at = _validate_scheduled_authority_timestamp(
        value["accepted_at"],
        "accepted_at",
    )
    trigger = _validate_scheduled_authority_timestamp(
        value["trigger"],
        "trigger",
    )
    target = _validate_scheduled_authority_target(value["target"])
    _require_bounded_text(
        value["semantic_objective"],
        "scheduled future-speech authority.semantic_objective",
        maximum=SCHEDULED_AUTHORITY_OBJECTIVE_MAX_CHARS,
    )
    authorized_content = _validate_scheduled_authorized_content(
        value["authorized_content"],
    )
    goal_continuation_ref = value["goal_continuation_ref"]
    if goal_continuation_ref is not None:
        try:
            validate_goal_continuation_ref(goal_continuation_ref)
        except CognitiveEpisodeValidationError as exc:
            raise CognitionContractError(
                "scheduled future-speech authority continuation ref is "
                f"invalid: {exc}"
            ) from exc
    _require_utc_timestamp(accepted_at["utc"], "authority.accepted_at.utc")
    _require_utc_timestamp(trigger["utc"], "authority.trigger.utc")
    accepted_time = parse_storage_utc_datetime(accepted_at["utc"])
    trigger_time = parse_storage_utc_datetime(trigger["utc"])
    if trigger_time <= accepted_time:
        raise CognitionContractError(
            "scheduled trigger must be strictly later than accepted time"
        )
    authority_id = value["authority_id"]
    if (
        not isinstance(authority_id, str)
        or not authority_id.startswith(SCHEDULED_AUTHORITY_ID_PREFIX)
    ):
        raise CognitionContractError(
            "scheduled authority id is invalid"
        )
    canonical_payload = _canonical_scheduled_authority_payload(
        source=source,
        accepted_at=accepted_at,
        trigger=trigger,
        target=target,
        semantic_objective=value["semantic_objective"].strip(),
        authorized_content=authorized_content,
    )
    expected_authority_id = _scheduled_authority_id(canonical_payload)
    if authority_id != expected_authority_id:
        raise CognitionContractError(
            "scheduled authority id does not match its canonical payload"
        )
    validated_authority: ScheduledFutureSpeechAuthorityV1 = {
        "schema_version": SCHEDULED_FUTURE_SPEECH_AUTHORITY_SCHEMA_VERSION,
        "authority_id": authority_id,
        "source": source,
        "accepted_at": accepted_at,
        "trigger": trigger,
        "target": target,
        "semantic_objective": value["semantic_objective"].strip(),
        "authorized_content": authorized_content,
        "goal_continuation_ref": goal_continuation_ref,
    }
    return validated_authority


def build_scheduled_future_speech_authority(
    *,
    source_episode_id: str,
    source_message_id: str,
    source_action_attempt_id: str,
    source_llm_trace_id: str = "",
    accepted_at_utc: str,
    timezone: str,
    trigger_local: str,
    platform: str,
    channel_type: str,
    audience_kind: str,
    semantic_objective: str,
    authorized_content_summary: str,
    authorized_detail_refs: Sequence[Mapping[str, Any]],
    goal_continuation_ref: GoalContinuationRefV1 | None = None,
) -> ScheduledFutureSpeechAuthorityV1:
    """Build the immutable pre-persistence authority for one future-speak task.

    The builder validates source identity, timestamp parseability, the strict
    future relation, target class, bounded fields, and the deterministic
    authority id. It must run before accepted-task, job, schedule, or run ids
    exist.

    Args:
        source_episode_id: Episode that accepted the future speech.
        source_message_id: Source platform message identity.
        source_action_attempt_id: Deterministic action-attempt identity.
        source_llm_trace_id: Optional protected trace identity, diagnostic
            only and excluded from the deterministic authority id.
        accepted_at_utc: Storage UTC instant when the task was accepted.
        timezone: Configured IANA timezone label used by local projections.
        trigger_local: Exact configured-local ``YYYY-MM-DD HH:MM`` trigger.
        platform: Target platform key.
        channel_type: Target channel type.
        audience_kind: Closed prompt-safe audience descriptor.
        semantic_objective: Bounded current-task semantic objective.
        authorized_content_summary: Bounded authorized summary.
        authorized_detail_refs: Bounded authorized detail references.
        goal_continuation_ref: Optional deterministic continuation reference.

    Returns:
        The validated immutable authority document.

    Raises:
        CognitionContractError: When any required identity, time, target, or
            content constraint is violated.
    """

    if not isinstance(audience_kind, str) or audience_kind not in {
        "group",
        "private",
    }:
        raise CognitionContractError(
            "scheduled audience kind is invalid"
        )
    normalized_accepted_at_utc = _canonical_utc_z_iso(accepted_at_utc)
    trigger_at_utc = _canonical_utc_z_iso(
        local_llm_datetime_to_storage_utc_iso(trigger_local)
    )
    proposal_source = {
        "source_episode_id": _required_authority_text(
            source_episode_id,
            "source_episode_id",
        ),
        "source_message_id": _required_authority_text(
            source_message_id,
            "source_message_id",
        ),
        "source_action_attempt_id": _required_authority_text(
            source_action_attempt_id,
            "source_action_attempt_id",
        ),
    }
    if source_llm_trace_id.strip():
        proposal_source["source_llm_trace_id"] = source_llm_trace_id.strip()
    source = _validate_scheduled_authority_source(proposal_source)
    accepted_at = {
        "utc": normalized_accepted_at_utc,
        "local": _configured_local_text(normalized_accepted_at_utc),
        "timezone": timezone.strip(),
    }
    trigger = {
        "utc": trigger_at_utc,
        "local": trigger_local.strip(),
        "timezone": timezone.strip(),
    }
    target = {
        "platform": _required_authority_text(platform, "platform"),
        "channel_type": _required_authority_text(
            channel_type,
            "channel_type",
        ),
        "audience_kind": audience_kind,
    }
    detail_refs = _validate_scheduled_authority_detail_refs(
        authorized_detail_refs,
        evidence=None,
    )
    authorized_content = {
        "summary": _required_bounded_authority_text(
            authorized_content_summary,
            "authorized_content.summary",
            maximum=SCHEDULED_AUTHORITY_SUMMARY_MAX_CHARS,
        ),
        "detail_refs": detail_refs,
    }
    objective = _required_bounded_authority_text(
        semantic_objective,
        "semantic_objective",
        maximum=SCHEDULED_AUTHORITY_OBJECTIVE_MAX_CHARS,
    )
    canonical_payload = _canonical_scheduled_authority_payload(
        source=source,
        accepted_at=accepted_at,
        trigger=trigger,
        target=target,
        semantic_objective=objective,
        authorized_content=authorized_content,
    )
    authority_id = _scheduled_authority_id(canonical_payload)
    authority = {
        "schema_version": SCHEDULED_FUTURE_SPEECH_AUTHORITY_SCHEMA_VERSION,
        "authority_id": authority_id,
        "source": source,
        "accepted_at": accepted_at,
        "trigger": trigger,
        "target": target,
        "semantic_objective": objective,
        "authorized_content": authorized_content,
        "goal_continuation_ref": goal_continuation_ref,
    }
    validated_authority = validate_scheduled_future_speech_authority(
        authority,
    )
    return validated_authority


def validate_scheduled_authority_carrier(
    value: object,
) -> ScheduledAuthorityCarrierV1:
    """Validate one later persistence envelope around the exact authority."""

    if not isinstance(value, Mapping):
        raise CognitionContractError(
            "scheduled authority carrier must be an object"
        )
    if (
        "schema_version" in value
        and value["schema_version"] != SCHEDULED_AUTHORITY_CARRIER_SCHEMA_VERSION
    ):
        raise CognitionContractError(
            "scheduled authority carrier schema is invalid"
        )
    if "authority" not in value:
        raise CognitionContractError(
            "scheduled authority carrier requires the authority"
        )
    validate_scheduled_future_speech_authority(value["authority"])
    for field_name in (
        "accepted_task_id",
        "background_job_id",
        "calendar_schedule_id",
        "calendar_run_id",
        "child_llm_trace_id",
        "delivery_tracking_id",
    ):
        if field_name not in value:
            continue
        field_value = value[field_name]
        if not isinstance(field_value, str) or not field_value.strip():
            raise CognitionContractError(
                f"scheduled authority carrier {field_name} is invalid"
            )
    return dict(value)  # type: ignore[return-value]


def copy_scheduled_authority_immutable(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deep copy of one immutable scheduled authority.

    Carrier records must never share mutable references with each other or
    with the pre-persistence authority.
    """

    copied_authority = deepcopy(dict(authority))
    return copied_authority


def _validate_scheduled_authority_detail_refs(
    value: object,
    *,
    evidence: Sequence[Mapping[str, Any]] | None,
) -> list[ScheduledAuthorityDetailRefV1]:
    """Validate bounded authorized detail references and their roles."""

    if (
        not isinstance(value, list)
        or not value
        or len(value) > SCHEDULED_AUTHORITY_DETAIL_REF_LIMIT
    ):
        raise CognitionContractError(
            "scheduled authority detail refs must contain 1-8 items"
        )
    evidence_by_handle = (
        {
            row["evidence_handle"]: row
            for row in evidence
            if isinstance(row, Mapping)
        }
        if evidence is not None
        else {}
    )
    seen_handles: set[str] = set()
    detail_refs: list[ScheduledAuthorityDetailRefV1] = []
    for index, raw_ref in enumerate(value):
        if not isinstance(raw_ref, Mapping) or set(raw_ref) != {
            "evidence_handle",
            "semantic_summary",
            "provenance_role",
        }:
            raise CognitionContractError(
                f"scheduled authority detail ref[{index}] fields are not exact"
            )
        evidence_handle = raw_ref["evidence_handle"]
        if (
            not isinstance(evidence_handle, str)
            or not evidence_handle.strip()
        ):
            raise CognitionContractError(
                f"scheduled authority detail ref[{index}] handle is invalid"
            )
        if evidence_handle in seen_handles:
            raise CognitionContractError(
                "scheduled authority detail refs contain duplicate handles"
            )
        seen_handles.add(evidence_handle)
        _require_bounded_text(
            raw_ref["semantic_summary"],
            f"scheduled authority detail ref[{index}].semantic_summary",
            maximum=SCHEDULED_AUTHORITY_DETAIL_SUMMARY_MAX_CHARS,
        )
        provenance_role = raw_ref["provenance_role"]
        if not isinstance(provenance_role, str) or not provenance_role.strip():
            raise CognitionContractError(
                f"scheduled authority detail ref[{index}] role is invalid"
            )
        if evidence is not None:
            evidence_row = evidence_by_handle.get(evidence_handle)
            if not isinstance(evidence_row, Mapping):
                raise CognitionContractError(
                    "scheduled authority detail handle is unavailable"
                )
            actual_authority = evidence_row["authority"]
            if provenance_role != actual_authority:
                raise CognitionContractError(
                    "scheduled authority detail role does not match "
                    "evidence authority"
                )
            if actual_authority not in SCHEDULED_AUTHORITY_CURRENT_ROLE_VALUES:
                raise CognitionContractError(
                    "scheduled authority detail does not carry "
                    "current-episode authority"
                )
        detail_refs.append({
            "evidence_handle": evidence_handle.strip(),
            "semantic_summary": raw_ref["semantic_summary"].strip(),
            "provenance_role": provenance_role.strip(),
        })
    return detail_refs


def _validate_scheduled_authority_source(
    value: object,
) -> ScheduledAuthoritySourceIdentityV1:
    """Validate the required immutable source identity fields."""

    if not isinstance(value, Mapping):
        raise CognitionContractError(
            "scheduled authority source must be an object"
        )
    required = {
        "source_episode_id",
        "source_message_id",
        "source_action_attempt_id",
    }
    optional = (
        {"source_llm_trace_id"}
        if isinstance(value.get("source_llm_trace_id"), str)
        else set()
    )
    _require_exact_keys(
        value,
        required | optional,
        "scheduled authority source",
    )
    for field_name in (
        "source_episode_id",
        "source_message_id",
        "source_action_attempt_id",
    ):
        _required_authority_text(value[field_name], field_name)
    source: ScheduledAuthoritySourceIdentityV1 = {
        "source_episode_id": value["source_episode_id"].strip(),
        "source_message_id": value["source_message_id"].strip(),
        "source_action_attempt_id": value["source_action_attempt_id"].strip(),
    }
    trace_id = value.get("source_llm_trace_id")
    if isinstance(trace_id, str) and trace_id.strip():
        source["source_llm_trace_id"] = trace_id.strip()
    return source


def _validate_scheduled_authority_timestamp(
    value: object,
    label: str,
) -> ScheduledAuthorityTimestampV1:
    """Validate one storage-UTC and configured-local authority timestamp."""

    if not isinstance(value, Mapping) or set(value) != {
        "utc",
        "local",
        "timezone",
    }:
        raise CognitionContractError(
            f"scheduled authority {label} fields are not exact"
        )
    _require_utc_timestamp(value["utc"], f"authority.{label}.utc")
    _require_bounded_text(
        value["local"],
        f"authority.{label}.local",
        maximum=40,
    )
    _require_bounded_text(
        value["timezone"],
        f"authority.{label}.timezone",
        maximum=80,
    )
    timestamp: ScheduledAuthorityTimestampV1 = {
        "utc": _canonical_utc_z_iso(value["utc"]),
        "local": value["local"].strip(),
        "timezone": value["timezone"].strip(),
    }
    _require_scheduled_authority_local_consistency(timestamp, label)
    return timestamp


def _require_scheduled_authority_local_consistency(
    timestamp: ScheduledAuthorityTimestampV1,
    label: str,
) -> None:
    """Require the configured-local wall clock to match the storage UTC instant.

    The authority's local and timezone fields are bound to the configured IANA
    timezone. A mutated wall clock or zone label fails closed and also changes
    the deterministic authority identity because the local and timezone fields
    are part of the canonical identity payload.
    """

    if timestamp["timezone"] != CHARACTER_TIME_ZONE:
        raise CognitionContractError(
            f"scheduled authority {label} timezone is inconsistent"
        )
    try:
        expected_utc = _canonical_utc_z_iso(
            local_llm_datetime_to_storage_utc_iso(timestamp["local"])
        )
    except ValueError as exc:
        raise CognitionContractError(
            f"scheduled authority {label} local time is invalid"
        ) from exc
    if expected_utc != timestamp["utc"]:
        raise CognitionContractError(
            f"scheduled authority {label} local time does not match utc"
        )


def _validate_scheduled_authority_target(
    value: object,
) -> ScheduledAuthorityTargetV1:
    """Validate the closed target class without delivery identifiers."""

    if not isinstance(value, Mapping) or set(value) != {
        "platform",
        "channel_type",
        "audience_kind",
    }:
        raise CognitionContractError(
            "scheduled authority target fields are not exact"
        )
    platform = _required_authority_text(value["platform"], "platform")
    channel_type = _required_authority_text(
        value["channel_type"],
        "channel_type",
    )
    audience_kind = value["audience_kind"]
    if audience_kind not in {"group", "private"}:
        raise CognitionContractError(
            "scheduled authority audience kind is invalid"
        )
    target: ScheduledAuthorityTargetV1 = {
        "platform": platform,
        "channel_type": channel_type,
        "audience_kind": audience_kind,
    }
    return target


def _validate_scheduled_authorized_content(
    value: object,
) -> ScheduledAuthorityAuthorizedContentV1:
    """Validate the bounded authorized semantic content block."""

    if not isinstance(value, Mapping) or set(value) != {
        "summary",
        "detail_refs",
    }:
        raise CognitionContractError(
            "scheduled authority authorized content fields are not exact"
        )
    summary = _required_bounded_authority_text(
        value["summary"],
        "authorized_content.summary",
        maximum=SCHEDULED_AUTHORITY_SUMMARY_MAX_CHARS,
    )
    detail_refs = _validate_scheduled_authority_detail_refs(
        value["detail_refs"],
        evidence=None,
    )
    authorized_content: ScheduledAuthorityAuthorizedContentV1 = {
        "summary": summary,
        "detail_refs": detail_refs,
    }
    return authorized_content


def _canonical_scheduled_authority_payload(
    *,
    source: Mapping[str, Any],
    accepted_at: Mapping[str, Any],
    trigger: Mapping[str, Any],
    target: Mapping[str, Any],
    semantic_objective: str,
    authorized_content: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the identity payload that excludes trace and carrier ids.

    The configured-local wall clock and timezone label are identity truth
    together with canonical UTC: mutating any of them changes the authority id.
    """

    canonical_payload = {
        "source_episode_id": source["source_episode_id"],
        "source_message_id": source["source_message_id"],
        "source_action_attempt_id": source["source_action_attempt_id"],
        "accepted_at_utc": accepted_at["utc"],
        "accepted_at_local": accepted_at["local"],
        "accepted_at_timezone": accepted_at["timezone"],
        "trigger_utc": trigger["utc"],
        "trigger_local": trigger["local"],
        "trigger_timezone": trigger["timezone"],
        "target_platform": target["platform"],
        "target_channel_type": target["channel_type"],
        "audience_kind": target["audience_kind"],
        "semantic_objective": semantic_objective,
        "authorized_content": {
            "summary": authorized_content["summary"],
            "detail_refs": list(authorized_content["detail_refs"]),
        },
    }
    return canonical_payload


def _scheduled_authority_id(canonical_payload: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 identity for a canonical payload."""

    serialized = json.dumps(
        canonical_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    authority_id = f"{SCHEDULED_AUTHORITY_ID_PREFIX}{digest}"
    return authority_id


def _configured_local_text(storage_utc_iso: str) -> str:
    """Project one storage instant to configured-local minute text."""

    local_time_context = local_time_context_from_storage_utc(storage_utc_iso)
    local_text = local_time_context["current_local_datetime"]
    return local_text


def _required_authority_text(value: object, label: str) -> str:
    """Require one non-empty authority identity or target text field."""

    if not isinstance(value, str) or not value.strip():
        raise CognitionContractError(
            f"scheduled authority {label} is required"
        )
    text = value.strip()
    return text


def _required_bounded_authority_text(
    value: object,
    label: str,
    *,
    maximum: int,
) -> str:
    """Require one bounded non-empty authority semantic text field."""

    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
    ):
        raise CognitionContractError(f"scheduled authority {label} is invalid")
    text = value.strip()
    return text


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
        required_row_fields = {
            "evidence_handle",
            "evidence_ref",
            "semantic_text",
            "visible_to",
            "authority",
        }
        if "memory_scope" in row:
            required_row_fields.add("memory_scope")
        if "temporal_provenance" in row:
            required_row_fields.add("temporal_provenance")
        _require_exact_keys(
            row,
            required_row_fields,
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
        if row["authority"] not in COGNITION_EVIDENCE_AUTHORITY_VALUES:
            raise CognitionContractError("evidence authority is invalid")
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
        source_id = row["evidence_ref"]["source_id"]
        authority = row["authority"]
        if authority == "conditional_character_guidance" and (
            source_kind != "promoted_reflection"
            or ":self_guidance:" not in source_id
        ):
            raise CognitionContractError(
                "conditional character guidance authority is not scoped"
            )
        if source_kind == "promoted_reflection":
            if ":self_guidance:" in source_id:
                if authority != "conditional_character_guidance":
                    raise CognitionContractError(
                        "self guidance must be conditional character guidance"
                    )
            elif authority != "character_world_context":
                raise CognitionContractError(
                    "promoted reflection lore must be character-world context"
                )
        if source_kind == "conversation_evidence" and source_id.startswith(
            "conversation-progress-event:"
        ):
            if "temporal_provenance" not in row:
                raise CognitionContractError(
                    "conversation progress evidence requires temporal provenance"
                )
            _validate_temporal_provenance(row["temporal_provenance"])
        elif "temporal_provenance" in row:
            raise CognitionContractError(
                "temporal provenance is only valid for progress evidence"
            )
        if source_kind == "promoted_memory" and "memory_scope" not in row:
            raise CognitionContractError(
                "promoted memory evidence requires memory scope"
            )
        if "memory_scope" in row:
            if source_kind != "promoted_memory":
                raise CognitionContractError(
                    "memory scope is only valid for promoted memory"
                )
            if (
                not isinstance(row["memory_scope"], str)
                or row["memory_scope"] not in MEMORY_SCOPE_VALUES
            ):
                raise CognitionContractError(
                    "evidence memory scope is invalid"
                )
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
    required = {
        "route",
        "intention",
        "target_roles",
        "reason",
        "goal_continuation_ref",
    }
    if "selected_branch_id" in value:
        required.add("selected_branch_id")
    if "selected_response_operation" in value:
        required.add("selected_response_operation")
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
    _validate_goal_continuation_ref_field(
        value["goal_continuation_ref"],
        "intention.goal_continuation_ref",
    )
    if "selected_branch_id" in value:
        _require_text(value["selected_branch_id"], "intention.selected_branch_id")
    if "selected_response_operation" in value:
        _validate_response_operation(
            value["selected_response_operation"],
            "intention.selected_response_operation",
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

    if not isinstance(value, Mapping):
        raise CognitionContractError("resolver request must be a mapping")
    capability = value.get("capability")
    if capability == "task_resolution_request":
        if set(value) != {
            "capability",
            "semantic_goal",
            "reason",
            "evidence_handles",
            "start_in_background",
            "goal_continuation_ref",
        }:
            raise CognitionContractError(
                "task resolution request fields are not exact"
            )
        if not isinstance(value["start_in_background"], bool):
            raise CognitionContractError(
                "task resolution start_in_background must be a boolean"
            )
        if value["goal_continuation_ref"] is None:
            raise CognitionContractError(
                "task resolution request requires goal_continuation_ref"
            )
    elif set(value) != {
        "capability",
        "semantic_goal",
        "reason",
        "evidence_handles",
        "goal_continuation_ref",
    }:
        raise CognitionContractError("resolver request fields are not exact")
    _require_text(value["capability"], "resolver request.capability")
    _require_text(value["semantic_goal"], "resolver request.semantic_goal")
    _require_text(value["reason"], "resolver request.reason")
    _validate_text_list(value["evidence_handles"], "resolver request.evidence_handles")
    _validate_goal_continuation_ref_field(
        value["goal_continuation_ref"],
        "resolver request.goal_continuation_ref",
    )


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


def _validate_temporal_provenance(value: Any) -> None:
    """Validate source time and its bounded deterministic age descriptor."""

    required = {"occurred_at", "age_descriptor"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError(
            "temporal provenance fields are not exact"
        )
    _require_utc_timestamp(value["occurred_at"], "temporal_provenance.occurred_at")
    _require_text(
        value["age_descriptor"],
        "temporal_provenance.age_descriptor",
        maximum=40,
    )


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


def _validate_goal_continuation_ref_field(value: Any, label: str) -> None:
    """Validate one nullable canonical continuation reference field."""

    if value is None:
        return
    try:
        validate_goal_continuation_ref(value)
    except CognitiveEpisodeValidationError as exc:
        raise CognitionContractError(f"{label} is invalid: {exc}") from exc


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
            "expected_previous_state",
            "replacement_state",
            "comparison_results",
            "changed_paths",
        },
        "state_update",
    )
    if value["state_scope"] not in {"user", "character"}:
        raise CognitionContractError("state update scope is invalid")
    _require_text(value["owner_key"], "state_update.owner_key")
    _validate_persistent_state(value["expected_previous_state"])
    _validate_persistent_state(value["replacement_state"])
    expected_previous = value["expected_previous_state"]
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
    expected_previous_owner = (
        expected_previous.get("owner_user_id")
        if expected_previous["state_scope"] == "user"
        else "global"
    )
    if (
        expected_previous["state_scope"] != value["state_scope"]
        or expected_previous_owner != value["owner_key"]
    ):
        raise CognitionContractError(
            "state update expected owner does not match state"
        )
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
        "personality_judgment",
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
    personality = value["personality_judgment"]
    if not isinstance(personality, Mapping) or set(personality) != {
        "logic",
        "defense",
        "quirks",
        "taboos",
    }:
        raise CognitionContractError(
            "character constraint personality judgment is invalid"
        )
    for field_name in ("logic", "defense", "quirks", "taboos"):
        _require_text(
            personality[field_name],
            f"personality_judgment.{field_name}",
            maximum=180,
        )


def _validate_character_identity_context(value: Any) -> None:
    """Validate the exact latest-identity appraisal partitions."""

    expected_families = {
        "moral_identity",
        "existential_drive",
        "relationship_social",
        "event_agency",
        "goal_threat_outcome",
        "goal_cognition",
        "epistemic_comparison_memory",
    }
    if not isinstance(value, Mapping) or set(value) != expected_families:
        raise CognitionContractError(
            "character identity context fields are not exact"
        )
    expected_categories = {
        "moral_identity": {
            "core",
            "personality",
            "boundaries",
            "self_image",
        },
        "existential_drive": {
            "core",
            "personality",
            "self_image",
        },
        "relationship_social": {"personality", "boundaries"},
        "event_agency": {"personality", "boundaries"},
        "goal_threat_outcome": {"personality", "boundaries"},
        "goal_cognition": {
            "core",
            "personality",
            "boundaries",
            "self_image",
        },
    }
    for family, expected in expected_categories.items():
        projection = value[family]
        if not isinstance(projection, Mapping) or set(projection) != expected:
            raise CognitionContractError(
                f"character identity {family} projection is invalid"
            )
    epistemic = value["epistemic_comparison_memory"]
    if (
        not isinstance(epistemic, Mapping)
        or set(epistemic) not in (set(), {"core"})
    ):
        raise CognitionContractError(
            "character identity epistemic projection is invalid"
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
        "public_group_scene",
        "conversation_continuity",
        "semantic_temporal_context",
    }
    if isinstance(value, Mapping) and "current_user_role" in value:
        required.add("current_user_role")
    if isinstance(value, Mapping) and "character_sleep_phase" in value:
        required.add("character_sleep_phase")
    if isinstance(value, Mapping) and "participant_bindings" in value:
        required.add("participant_bindings")
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("scene context fields are not exact")
    if value["channel_scope"] not in {"private", "group", "internal"}:
        raise CognitionContractError("scene context channel_scope is invalid")
    for field_name in required - {
        "channel_scope",
        "conversation_continuity",
        "public_group_scene",
        "participant_bindings",
    }:
        _require_text(value[field_name], f"scene context.{field_name}")
    _require_bounded_text(
        value["public_group_scene"],
        "scene context.public_group_scene",
        maximum=1800,
    )
    _require_bounded_text(
        value["conversation_continuity"],
        "scene context.conversation_continuity",
        maximum=2200,
    )
    participant_bindings = value.get("participant_bindings")
    if participant_bindings is not None:
        _validate_scene_participant_bindings(participant_bindings)


def _validate_scene_participant_bindings(value: Any) -> None:
    """Validate one bounded prompt-safe scene participant roster."""

    if (
        not isinstance(value, list)
        or len(value) > MAX_SCENE_PARTICIPANT_BINDINGS
    ):
        raise CognitionContractError(
            "scene participant bindings must contain "
            f"0-{MAX_SCENE_PARTICIPANT_BINDINGS} items"
        )
    handles: list[str] = []
    for index, binding in enumerate(value):
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {
                "handle",
                "display_name",
                "entity_kind",
            }
        ):
            raise CognitionContractError(
                f"scene participant binding[{index}] fields are not exact"
            )
        handle = binding["handle"]
        if (
            not isinstance(handle, str)
            or not SCENE_PARTICIPANT_HANDLE_RE.fullmatch(handle)
        ):
            raise CognitionContractError(
                f"scene participant binding[{index}].handle is invalid"
            )
        if handle in handles:
            raise CognitionContractError(
                "scene participant bindings contain duplicate handles"
            )
        handles.append(handle)
        _require_bounded_text(
            binding["display_name"],
            f"scene participant binding[{index}].display_name",
            maximum=200,
        )
        if binding["entity_kind"] != "third_party":
            raise CognitionContractError(
                "scene participant binding entity_kind is invalid"
            )


def _validate_group_engagement_action_context(value: Any) -> None:
    """Validate advisory guidance with a descriptor-only confidence field."""

    if not isinstance(value, Mapping):
        raise CognitionContractError(
            "group engagement action context must be a mapping"
        )
    _require_exact_keys(
        value,
        {"engagement_guidelines", "confidence"},
        "group engagement action context",
    )
    guidelines = value["engagement_guidelines"]
    if (
        not isinstance(guidelines, list)
        or len(guidelines)
        > L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT
    ):
        raise CognitionContractError(
            "group engagement guidelines are invalid"
        )
    for guideline in guidelines:
        _require_text(
            guideline,
            "group engagement guideline",
            maximum=GROUP_ENGAGEMENT_GUIDELINE_MAX_CHARS,
        )
    _require_bounded_text(
        value["confidence"],
        "group engagement confidence descriptor",
        maximum=GROUP_ENGAGEMENT_CONFIDENCE_MAX_CHARS,
    )
    if not guidelines and value["confidence"]:
        raise CognitionContractError(
            "empty group engagement guidelines require empty confidence"
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


def _validate_response_operation(
    value: Any,
    label: str,
) -> DialogResponseOperation:
    """Validate one control-only response-operation carrier."""

    try:
        validated_operation = validate_dialog_response_operation(value)
    except CognitiveEpisodeValidationError as exc:
        raise CognitionContractError(
            f"{label} is invalid: {exc}"
        ) from exc
    return validated_operation


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


def _validate_character_operational_context(value: Any) -> None:
    """Validate one bounded redacted character operational selection."""

    _require_exact_keys(
        value,
        {
            "schema_version",
            "source_updated_at",
            "effective_at",
            "view_digest",
            "consumer_role",
            "affect",
            "pressures",
            "context_digest",
        },
        "character operational context",
    )
    if value["schema_version"] != "character_operational_context.v1":
        raise CognitionContractError(
            "character operational context schema is invalid"
        )
    _require_utc_timestamp(
        value["source_updated_at"],
        "character operational context.source_updated_at",
    )
    _require_utc_timestamp(
        value["effective_at"],
        "character operational context.effective_at",
    )
    for field_name in ("view_digest", "context_digest"):
        _require_text(
            value[field_name],
            f"character operational context.{field_name}",
            maximum=CHARACTER_OPERATIONAL_CONTEXT_DIGEST_CHARS,
        )
    context_body = {
        key: item
        for key, item in value.items()
        if key != "context_digest"
    }
    if value["context_digest"] != canonical_digest(context_body):
        raise CognitionContractError(
            "character operational context digest is invalid"
        )
    consumer_role = value["consumer_role"]
    if consumer_role not in CHARACTER_OPERATIONAL_CONSUMER_ROLES:
        raise CognitionContractError(
            "character operational consumer role is invalid"
        )
    affect = value["affect"]
    if not isinstance(affect, list) or len(affect) > MAX_CONTEXT_AFFECT_ROWS:
        raise CognitionContractError(
            "character operational affect selection is invalid"
        )
    for row in affect:
        _validate_character_operational_affect_row(row)
    pressures = value["pressures"]
    if (
        not isinstance(pressures, list)
        or len(pressures) > MAX_CONTEXT_PRESSURE_ROWS
        or (consumer_role == "surface" and pressures)
    ):
        raise CognitionContractError(
            "character operational pressure selection is invalid"
        )
    for row in pressures:
        _validate_character_operational_pressure_row(row)
    if serialized_character_count(value) > MAX_CHARACTER_OPERATIONAL_CONTEXT_CHARS:
        raise CognitionContextLimitError(
            "required character operational context exceeds the fixed cap"
        )


def _validate_character_operational_affect_row(value: Any) -> None:
    """Validate one source-free affect row in an operational context."""

    _require_exact_keys(
        value,
        {
            "emotion_id",
            "intensity",
            "phase",
            "trend",
            "root_kind",
            "cause_class",
            "freshness",
        },
        "character operational affect",
    )
    _require_text(value["emotion_id"], "character operational affect.emotion_id")
    _require_text(value["intensity"], "character operational affect.intensity")
    if value["phase"] not in RELATIONSHIP_AFFECT_PHASES:
        raise CognitionContractError("character operational affect phase is invalid")
    _require_text(value["trend"], "character operational affect.trend")
    if value["root_kind"] not in CHARACTER_OPERATIONAL_ROOT_KINDS:
        raise CognitionContractError(
            "character operational affect root kind is invalid"
        )
    _validate_operational_cause_class(value["cause_class"])
    _require_text(value["freshness"], "character operational affect.freshness")


def _validate_character_operational_pressure_row(value: Any) -> None:
    """Validate one source-free pressure row in an operational context."""

    _require_exact_keys(
        value,
        {"kind", "salience", "lifecycle", "cause_class", "freshness"},
        "character operational pressure",
    )
    if value["kind"] not in CHARACTER_OPERATIONAL_ROOT_KINDS:
        raise CognitionContractError(
            "character operational pressure kind is invalid"
        )
    _require_text(value["salience"], "character operational pressure.salience")
    _require_text(value["lifecycle"], "character operational pressure.lifecycle")
    _validate_operational_cause_class(value["cause_class"])
    _require_text(value["freshness"], "character operational pressure.freshness")


def _validate_operational_cause_class(value: Any) -> None:
    """Require one closed projection-only operational cause class."""

    if value not in OPERATIONAL_CAUSE_CLASSES:
        raise CognitionContractError("operational cause class is invalid")


def _validate_relationship_operational_context(value: Mapping[str, Any]) -> None:
    """Validate the isolated bounded relationship projection."""

    _require_exact_keys(
        value,
        {
            "schema_version",
            "relationship_id",
            "axes",
            "causal_context",
            "affect",
            "relationship_freshness",
            "evidence_freshness",
        },
        "relationship operational context",
    )
    if value["schema_version"] != "relationship_operational_context.v1":
        raise CognitionContractError(
            "relationship operational context schema is invalid"
        )
    _require_text(
        value["relationship_id"],
        "relationship operational context.relationship_id",
        maximum=200,
    )
    axes = value["axes"]
    if not isinstance(axes, Mapping) or set(axes) != {
        "familiarity",
        "positive_regard",
        "trust",
        "attachment",
        "desired_closeness",
        "perceived_closeness",
        "care",
        "boundary_safety",
        "exclusivity",
        "unresolved_injury",
        "salience",
    }:
        raise CognitionContractError("relationship operational axes are invalid")
    for axis_name, axis_value in axes.items():
        if axis_name in {"positive_regard", "trust", "boundary_safety"}:
            if (
                isinstance(axis_value, bool)
                or not isinstance(axis_value, int)
                or not -100 <= axis_value <= 100
            ):
                raise CognitionContractError(
                    "relationship operational signed axis is invalid"
                )
        else:
            _require_axis(axis_value, "relationship operational axis")
    causal_context = value["causal_context"]
    if (
        not isinstance(causal_context, list)
        or len(causal_context) > MAX_RELATIONSHIP_CAUSAL_ROWS
    ):
        raise CognitionContractError(
            "relationship operational causal context is invalid"
        )
    for row in causal_context:
        _require_exact_keys(
            row,
            {
                "entity_kind",
                "semantic_summary",
                "salience",
                "lifecycle",
                "freshness",
            },
            "relationship operational causal row",
        )
        if row["entity_kind"] not in RELATIONSHIP_CAUSAL_ENTITY_KINDS:
            raise CognitionContractError(
                "relationship operational causal entity kind is invalid"
            )
        _require_text(
            row["semantic_summary"],
            "relationship operational causal summary",
            maximum=MAX_RELATIONSHIP_CAUSAL_SUMMARY_CHARS,
        )
        _require_text(row["salience"], "relationship operational causal salience")
        _require_text(row["lifecycle"], "relationship operational causal lifecycle")
        _require_text(row["freshness"], "relationship operational causal freshness")
    affect = value["affect"]
    if (
        not isinstance(affect, list)
        or len(affect) > MAX_RELATIONSHIP_AFFECT_ROWS
    ):
        raise CognitionContractError("relationship operational affect is invalid")
    for row in affect:
        _require_exact_keys(
            row,
            {"emotion_id", "intensity", "phase", "trend", "freshness"},
            "relationship operational affect row",
        )
        _require_text(row["emotion_id"], "relationship operational affect id")
        _require_text(row["intensity"], "relationship operational affect intensity")
        if row["phase"] not in RELATIONSHIP_AFFECT_PHASES:
            raise CognitionContractError(
                "relationship operational affect phase is invalid"
            )
        _require_text(row["trend"], "relationship operational affect trend")
        _require_text(row["freshness"], "relationship operational affect freshness")
    _require_text(
        value["relationship_freshness"],
        "relationship operational relationship freshness",
    )
    _require_text(
        value["evidence_freshness"],
        "relationship operational evidence freshness",
    )
    if (
        serialized_character_count(value)
        > MAX_RELATIONSHIP_OPERATIONAL_CONTEXT_CHARS
    ):
        raise CognitionContextLimitError(
            "required relationship operational context exceeds the fixed cap"
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


def _validate_surface_resolver_result(value: Any) -> None:
    """Validate one source-owned resolver outcome for L3 planning."""

    base_fields = {"capability_kind", "status", "semantic_result"}
    if not isinstance(value, Mapping):
        raise CognitionContractError("surface resolver result fields are not exact")
    capability_kind = value.get("capability_kind")
    required = base_fields
    if capability_kind == "task_resolution_request":
        required = base_fields | {
            "prompt_safe_observation_handle",
            "evidence_state",
            "evidence_excerpts",
            "evidence_handles",
            "remaining_needs",
        }
    if set(value) != required:
        raise CognitionContractError("surface resolver result fields are not exact")
    _require_text(value["capability_kind"], "surface resolver capability kind")
    if value["status"] not in {"succeeded", "blocked", "failed"}:
        raise CognitionContractError("surface resolver result.status is invalid")
    _require_text(
        value["semantic_result"],
        "surface resolver semantic result",
        maximum=2000,
    )
    if capability_kind != "task_resolution_request":
        return
    _validate_surface_task_evidence_fields(value)


def _validate_surface_task_evidence_fields(value: Mapping[str, Any]) -> None:
    """Validate the prompt-safe evidence projection for a task result."""

    _require_text(
        value["prompt_safe_observation_handle"],
        "surface resolver observation handle",
        maximum=200,
    )
    evidence_state = {
        "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
        "state": value["evidence_state"],
        "remaining_needs": value["remaining_needs"],
    }
    try:
        validate_resolver_evidence_state(evidence_state)
    except ResolverValidationError as exc:
        raise CognitionContractError(
            f"surface resolver evidence state is invalid: {exc}"
        ) from exc
    evidence_excerpts = value["evidence_excerpts"]
    if (
        not isinstance(evidence_excerpts, list)
        or len(evidence_excerpts) > MAX_RESOLVER_EVIDENCE_EXCERPTS
    ):
        raise CognitionContractError(
            "surface resolver evidence excerpts are invalid"
        )
    for index, excerpt in enumerate(evidence_excerpts):
        _require_text(
            excerpt,
            f"surface resolver evidence_excerpts[{index}]",
            maximum=MAX_RESOLVER_EVIDENCE_EXCERPT_CHARS,
        )
    evidence_handles = value["evidence_handles"]
    if (
        not isinstance(evidence_handles, list)
        or len(evidence_handles) > 8
    ):
        raise CognitionContractError(
            "surface resolver evidence handles are invalid"
        )
    for index, handle in enumerate(evidence_handles):
        _require_text(
            handle,
            f"surface resolver evidence_handles[{index}]",
            maximum=240,
        )
    if len(evidence_handles) != len(set(evidence_handles)):
        raise CognitionContractError(
            "surface resolver evidence handles are duplicated"
        )
    remaining_needs = value["remaining_needs"]
    if not isinstance(remaining_needs, list) or len(remaining_needs) > 8:
        raise CognitionContractError(
            "surface resolver remaining needs are invalid"
        )
    for index, need in enumerate(remaining_needs):
        _require_text(
            need,
            f"surface resolver remaining_needs[{index}]",
            maximum=240,
        )
    state = value["evidence_state"]
    if state == "complete" and (not evidence_excerpts or remaining_needs):
        raise CognitionContractError(
            "complete surface resolver evidence needs excerpts and no needs"
        )
    if state == "partial" and (not evidence_excerpts or not remaining_needs):
        raise CognitionContractError(
            "partial surface resolver evidence needs excerpts and needs"
        )
    if state == "missing" and evidence_excerpts:
        raise CognitionContractError(
            "missing surface resolver evidence cannot contain excerpts"
        )
    if value["status"] == "succeeded" and state == "blocked":
        raise CognitionContractError(
            "succeeded surface resolver result cannot have blocked evidence"
        )
    if value["status"] != "succeeded" and state != "blocked":
        raise CognitionContractError(
            "blocked surface resolver result needs blocked evidence"
        )


def _validate_surface_resolver_result_dependency(
    result: Mapping[str, Any],
    dependency: RequiredResolverEvidenceDependencyV1,
) -> None:
    """Require the surface result to match its exact internal dependency."""

    if result["capability_kind"] != dependency["capability_kind"]:
        raise CognitionContractError(
            "surface resolver capability does not match dependency"
        )
    for field_name in (
        "prompt_safe_observation_handle",
        "evidence_state",
        "evidence_handles",
        "remaining_needs",
    ):
        dependency_field = (
            "state" if field_name == "evidence_state" else field_name
        )
        if result[field_name] != dependency[dependency_field]:
            raise CognitionContractError(
                f"surface resolver dependency field mismatch: {field_name}"
            )


def validate_surface_addressee_plan(value: Any) -> None:
    """Validate bounded, prompt-safe target and wording rows."""

    if not isinstance(value, list) or len(value) > 8:
        raise CognitionContractError(
            "surface addressee plan must contain 0-8 items"
        )
    handles: set[str] = set()
    for index, row in enumerate(value):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "handle",
                "display_name",
                "semantic_role",
                "wording_policy",
            }
        ):
            raise CognitionContractError(
                f"surface addressee plan[{index}] fields are not exact"
            )
        handle = row["handle"]
        if (
            not isinstance(handle, str)
            or not SURFACE_ROLE_HANDLE_RE.fullmatch(handle)
        ):
            raise CognitionContractError(
                f"surface addressee plan[{index}].handle is invalid"
            )
        if handle in handles:
            raise CognitionContractError(
                "surface addressee plan contains duplicate handles"
            )
        handles.add(handle)
        _require_bounded_text(
            row["display_name"],
            f"surface addressee plan[{index}].display_name",
            maximum=200,
        )
        if row["semantic_role"] not in {
            "direct_recipient",
            "embedded_target",
            "embedded_actor",
            "observer",
        }:
            raise CognitionContractError(
                f"surface addressee plan[{index}].semantic_role is invalid"
            )
        wording_policy = row["wording_policy"]
        if wording_policy not in {
            "second_person_allowed",
            "named_or_third_person_required",
        }:
            raise CognitionContractError(
                f"surface addressee plan[{index}].wording_policy is invalid"
            )
        if handle.startswith("p") and (
            wording_policy != "named_or_third_person_required"
        ):
            raise CognitionContractError(
                "third-party addressees require named or third-person wording"
            )
        if (
            handle == "current_user"
            and row["semantic_role"] in {"direct_recipient", "embedded_target"}
            and wording_policy != "second_person_allowed"
        ):
            raise CognitionContractError(
                "current-user addressees require second-person wording"
            )


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


def _canonical_utc_z_iso(value: str) -> str:
    """Canonicalize one storage UTC instant to native UTC ``Z`` text.

    The canonical authority timestamps must use the strict ``Z`` suffix both
    before identity hashing and in every validated authority output. Parsing
    through ``parse_storage_utc_datetime`` accepts both ``Z`` and ``+00:00``
    input forms and yields the same canonical instant either way.

    Args:
        value: Storage UTC timestamp string accepted by
            ``parse_storage_utc_datetime``.

    Returns:
        The canonical UTC timestamp ending in ``Z`` with no offset suffix.

    Raises:
        ValueError: When the value is not a parseable storage UTC timestamp.
    """

    storage_datetime_utc = parse_storage_utc_datetime(value)
    canonical_iso = storage_datetime_utc.isoformat()
    if canonical_iso.endswith("+00:00"):
        canonical_iso = f"{canonical_iso[:-6]}Z"
    return canonical_iso


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


def _require_simplified_chinese_reason(
    value: Any,
    label: str,
    maximum: int,
    *,
    field_path: str | None = None,
) -> None:
    """Require bounded non-empty Simplified Chinese semantic prose."""

    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
    ):
        raise CognitionContractError(
            f"{label} is invalid",
            field_path=field_path,
        )
    if not _CJK_IDEOGRAPH_RE.search(value):
        raise CognitionContractError(
            f"{label} must be Simplified Chinese",
            field_path=field_path,
        )


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


class TextSurfaceInput(TypedDict, total=False):
    """Canonical semantic input passed from cognition to surface planning."""

    schema_version: Literal["text_surface_input.v3"]
    episode: CognitiveEpisodeV1
    active_character_goal: dict[str, Any]
    response_plan: dict[str, Any]
    expression_policy: ExpressionPolicyV2
    semantic_affect: list[SemanticAffectProjectionV2]
    permitted_action_results: list[SemanticActionResultV2]
    interaction_style_context: str
    character_expression_context: CharacterExpressionContextV2
    subjective_expression_context: SubjectiveExpressionContextV1
    addressee_plan: list[SurfaceAddresseePlanV1]
    visual_character_context: str
    recent_character_dialog: NotRequired[list[str]]
    relational_willingness: NotRequired[RelationalWillingness]


def validate_text_surface_input_canonical(
    payload: Mapping[str, Any],
) -> TextSurfaceInput:
    """Validate the one canonical semantic surface input envelope."""

    required = {
        "schema_version",
        "episode",
        "active_character_goal",
        "response_plan",
        "expression_policy",
        "semantic_affect",
        "permitted_action_results",
        "interaction_style_context",
        "character_expression_context",
        "subjective_expression_context",
        "addressee_plan",
        "visual_character_context",
    }
    optional = {"recent_character_dialog", "relational_willingness"}
    _require_exact_keys(payload, required | (set(payload) & optional), "text surface input")
    if payload["schema_version"] != "text_surface_input.v3":
        raise CognitionContractError("unsupported canonical text surface input schema")
    if not isinstance(payload["episode"], Mapping):
        raise CognitionContractError("text surface episode is invalid")
    if not isinstance(payload["active_character_goal"], Mapping):
        raise CognitionContractError("text surface active goal is invalid")
    if not isinstance(payload["response_plan"], Mapping):
        raise CognitionContractError("text surface response plan is invalid")
    _validate_expression_policy(payload["expression_policy"])
    if not isinstance(payload["semantic_affect"], list):
        raise CognitionContractError("text surface affect is invalid")
    if not isinstance(payload["permitted_action_results"], list):
        raise CognitionContractError("text surface action results are invalid")
    for row in payload["permitted_action_results"]:
        _validate_action_result(row)
    _require_text(payload["interaction_style_context"], "text surface interaction style")
    if not isinstance(payload["character_expression_context"], Mapping):
        raise CognitionContractError("text surface character expression context is invalid")
    subjective_context = payload["subjective_expression_context"]
    _require_exact_keys(
        subjective_context,
        {"private_monologue", "epistemic_boundary"},
        "text surface subjective expression context",
    )
    _require_text(
        subjective_context["private_monologue"],
        "text surface private monologue",
        maximum=600,
    )
    _require_text(
        subjective_context["epistemic_boundary"],
        "text surface epistemic boundary",
        maximum=1000,
    )
    validate_surface_addressee_plan(payload["addressee_plan"])
    _require_text(payload["visual_character_context"], "text surface visual context")
    if "recent_character_dialog" in payload:
        _validate_recent_character_dialog(payload["recent_character_dialog"])
    if "relational_willingness" in payload:
        validate_relational_willingness(payload["relational_willingness"])
    return dict(payload)  # type: ignore[return-value]
