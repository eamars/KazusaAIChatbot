"""Closed domain contracts for revisioned character identity growth."""

from __future__ import annotations

from typing import Literal, TypedDict


IDENTITY_EVIDENCE_SCHEMA_VERSION = "character_identity_evidence_ref.v1"
IDENTITY_EVIDENCE_CARD_SCHEMA_VERSION = (
    "character_identity_evidence_card.v1"
)
IDENTITY_PROPOSAL_INPUT_SCHEMA_VERSION = (
    "character_identity_proposal_input.v1"
)
IDENTITY_REVIEW_INPUT_SCHEMA_VERSION = (
    "character_identity_review_input.v1"
)
IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION = (
    "character_identity_proposal_decision.v1"
)
IDENTITY_REVIEW_DECISION_SCHEMA_VERSION = (
    "character_identity_review_decision.v1"
)

TEXT_IDENTITY_PATHS = frozenset({
    "name",
    "description",
    "gender",
    "birthday",
    "backstory",
    "personality_brief.mbti",
    "personality_brief.logic",
    "personality_brief.tempo",
    "personality_brief.defense",
    "personality_brief.quirks",
    "personality_brief.taboos",
    "self_image.self_concept",
    "visual_characterization",
})
INTEGER_IDENTITY_PATHS = frozenset({"age"})
NUMERIC_IDENTITY_PATHS = frozenset({
    "boundary_profile.self_integrity",
    "boundary_profile.control_sensitivity",
    "boundary_profile.relational_override",
    "boundary_profile.control_intimacy_misread",
    "boundary_profile.authority_skepticism",
    "linguistic_texture_profile.fragmentation",
    "linguistic_texture_profile.hesitation_density",
    "linguistic_texture_profile.counter_questioning",
    "linguistic_texture_profile.softener_density",
    "linguistic_texture_profile.formalism_avoidance",
    "linguistic_texture_profile.abstraction_reframing",
    "linguistic_texture_profile.direct_assertion",
    "linguistic_texture_profile.emotional_leakage",
    "linguistic_texture_profile.rhythmic_bounce",
    "linguistic_texture_profile.self_deprecation",
})
ENUM_IDENTITY_PATHS = frozenset({
    "boundary_profile.compliance_strategy",
    "boundary_profile.boundary_recovery",
})
TEXT_LIST_IDENTITY_PATHS = frozenset({
    "self_image.current_growth_edges",
})
ALLOWED_IDENTITY_PATHS = frozenset().union(
    TEXT_IDENTITY_PATHS,
    INTEGER_IDENTITY_PATHS,
    NUMERIC_IDENTITY_PATHS,
    ENUM_IDENTITY_PATHS,
    TEXT_LIST_IDENTITY_PATHS,
)

SEMANTIC_BAND_VALUES = {
    "very_low": 0.1,
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "very_high": 0.9,
}
ENUM_VALUES_BY_PATH = {
    "boundary_profile.compliance_strategy": frozenset({
        "resist",
        "evade",
        "comply",
    }),
    "boundary_profile.boundary_recovery": frozenset({
        "rebound",
        "delayed_rebound",
        "decay",
        "detach",
    }),
}

CANDIDATE_TRANSITIONS = {
    "emerging": frozenset({
        "emerging",
        "ready",
        "rejected",
        "superseded",
    }),
    "ready": frozenset({
        "promoted",
        "rejected",
        "superseded",
    }),
    "promoted": frozenset(),
    "rejected": frozenset(),
    "superseded": frozenset(),
}

IDENTITY_GROWTH_REASON_CODES = frozenset({
    "not_routed",
    "no_eligible_evidence",
    "proposal_no_change",
    "proposal_contract_failed",
    "candidate_emerging",
    "candidate_ready",
    "review_rejected",
    "review_contract_failed",
    "privacy_blocked",
    "cadence_wait",
    "duplicate_root",
    "stale_base",
    "contradiction_blocked",
    "promotion_write_failed",
    "revision_promoted",
    "awaiting_first_consumption",
    "revision_consumed",
    "revision_consumption_mismatch",
})
IDENTITY_GROWTH_HEALTH_STATES = frozenset({
    "healthy_idle",
    "waiting_for_evidence",
    "semantic_rejection",
    "promotion_ready",
    "awaiting_consumption",
    "healthy_active",
    "pipeline_error",
    "consumption_error",
})

EVIDENCE_SOURCE_KINDS = frozenset({
    "settled_episode",
    "daily_reflection",
})
EVIDENCE_SCOPE_KINDS = frozenset({
    "private",
    "group",
    "self_cognition",
})
RUN_LIFECYCLE_STATES = frozenset({
    "in_progress",
    "committed",
    "post_commit_pending",
    "complete",
    "failed",
})
PROPOSAL_ACTIONS = frozenset({
    "no_change",
    "explicit_self_redefinition",
    "inferred_growth",
    "corroborate_candidate",
})
REVIEW_VERDICTS = frozenset({
    "accept",
    "reject",
    "no_change",
})
IDENTITY_RELEVANCE_VALUES = frozenset({
    "durable",
    "ephemeral",
    "absent",
})
GLOBAL_APPLICABILITY_VALUES = frozenset({
    "global",
    "scoped",
    "absent",
})
CHARACTER_AUTHORSHIP_VALUES = frozenset({
    "self_declared",
    "inferred",
    "absent",
})
CONFIDENCE_VALUES = frozenset({
    "low",
    "medium",
    "high",
})
PRIVATE_DETAIL_RISK_VALUES = frozenset({
    "low",
    "high",
})
REVIEW_COHERENCE_VALUES = frozenset({
    "coherent",
    "conflicting",
    "absent",
})
ACCEPTED_CHANGE_KINDS = frozenset({
    "explicit_self_redefinition",
    "inferred_growth",
})

IDENTITY_EVIDENCE_CARD_LIMIT = 12
IDENTITY_EVIDENCE_CARD_TEXT_LIMIT = 400
IDENTITY_CANDIDATE_PROMPT_LIMIT = 8
IDENTITY_PATCH_LIMIT = 5
IDENTITY_PROMPT_CHAR_BUDGET_DEFAULT = 18_000
IDENTITY_CONSUMER_KINDS = frozenset({
    "moral_identity",
    "existential_drive",
    "relationship_social",
    "event_agency",
    "goal_threat_outcome",
    "goal_cognition",
    "epistemic_comparison_memory",
    "text",
    "visual",
    "naming",
})

TOP_LEVEL_IDENTITY_KEYS = frozenset({
    "name",
    "description",
    "gender",
    "age",
    "birthday",
    "backstory",
    "personality_brief",
    "boundary_profile",
    "linguistic_texture_profile",
    "self_image",
    "visual_characterization",
})
PERSONALITY_KEYS = frozenset({
    "mbti",
    "logic",
    "tempo",
    "defense",
    "quirks",
    "taboos",
})
BOUNDARY_KEYS = frozenset({
    "self_integrity",
    "control_sensitivity",
    "compliance_strategy",
    "relational_override",
    "control_intimacy_misread",
    "boundary_recovery",
    "authority_skepticism",
})
LINGUISTIC_TEXTURE_KEYS = frozenset({
    "fragmentation",
    "hesitation_density",
    "counter_questioning",
    "softener_density",
    "formalism_avoidance",
    "abstraction_reframing",
    "direct_assertion",
    "emotional_leakage",
    "rhythmic_bounce",
    "self_deprecation",
})
SELF_IMAGE_KEYS = frozenset({
    "self_concept",
    "current_growth_edges",
})

TEXT_LIMIT_BY_PATH = {
    "name": 160,
    "description": 2400,
    "gender": 120,
    "birthday": 160,
    "backstory": 6000,
    "personality_brief.mbti": 80,
    "personality_brief.logic": 1200,
    "personality_brief.tempo": 1200,
    "personality_brief.defense": 1200,
    "personality_brief.quirks": 1600,
    "personality_brief.taboos": 1600,
    "self_image.self_concept": 2400,
    "visual_characterization": 3000,
}
GROWTH_EDGE_LIMIT = 400
GROWTH_EDGE_COUNT_LIMIT = 5

PatchValueKind = Literal[
    "text",
    "integer",
    "semantic_band",
    "closed_enum",
    "text_list",
]
EvidenceSourceKind = Literal["settled_episode", "daily_reflection"]
EvidenceScopeKind = Literal["private", "group", "self_cognition"]
CandidateStatus = Literal[
    "emerging",
    "ready",
    "promoted",
    "rejected",
    "superseded",
]
IdentityGrowthHealthState = Literal[
    "healthy_idle",
    "waiting_for_evidence",
    "semantic_rejection",
    "promotion_ready",
    "awaiting_consumption",
    "healthy_active",
    "pipeline_error",
    "consumption_error",
]
ProposalAction = Literal[
    "no_change",
    "explicit_self_redefinition",
    "inferred_growth",
    "corroborate_candidate",
]
ReviewVerdict = Literal["accept", "reject", "no_change"]
IdentityRelevance = Literal["durable", "ephemeral", "absent"]
GlobalApplicability = Literal["global", "scoped", "absent"]
CharacterAuthorship = Literal["self_declared", "inferred", "absent"]
Confidence = Literal["low", "medium", "high"]
PrivateDetailRisk = Literal["low", "high"]
ReviewCoherence = Literal["coherent", "conflicting", "absent"]
AcceptedChangeKind = Literal[
    "explicit_self_redefinition",
    "inferred_growth",
]
IdentityPolicyStatus = Literal[
    "no_change",
    "candidate_updated",
    "revision_ready",
    "rejected",
    "deferred",
]
IdentityEvaluationStatus = Literal[
    "no_change",
    "candidate_updated",
    "revision_promoted",
    "rejected",
    "failed",
    "deferred",
]
IdentityConsumptionStatus = Literal["consumed", "mismatch"]


class PersonalityBriefV1(TypedDict):
    """Prompt-visible personality fields in one effective identity."""

    mbti: str
    logic: str
    tempo: str
    defense: str
    quirks: str
    taboos: str


class BoundaryProfileV1(TypedDict):
    """Closed boundary fields in one effective identity."""

    self_integrity: float
    control_sensitivity: float
    compliance_strategy: str
    relational_override: float
    control_intimacy_misread: float
    boundary_recovery: str
    authority_skepticism: float


class LinguisticTextureProfileV1(TypedDict):
    """Closed text-expression fields in one effective identity."""

    fragmentation: float
    hesitation_density: float
    counter_questioning: float
    softener_density: float
    formalism_avoidance: float
    abstraction_reframing: float
    direct_assertion: float
    emotional_leakage: float
    rhythmic_bounce: float
    self_deprecation: float


class SelfImageV1(TypedDict):
    """Current self-concept and bounded growth edges."""

    self_concept: str
    current_growth_edges: list[str]


class CharacterEffectiveIdentityV1(TypedDict):
    """Complete immutable semantic identity snapshot."""

    name: str
    description: str
    gender: str
    age: int
    birthday: str
    backstory: str
    personality_brief: PersonalityBriefV1
    boundary_profile: BoundaryProfileV1
    linguistic_texture_profile: LinguisticTextureProfileV1
    self_image: SelfImageV1
    visual_characterization: str


class CharacterIdentityCognitionContextV1(TypedDict):
    """Closed identity categories assigned to V2 appraisal families."""

    moral_identity: dict[str, object]
    existential_drive: dict[str, object]
    relationship_social: dict[str, object]
    event_agency: dict[str, object]
    goal_threat_outcome: dict[str, object]
    goal_cognition: dict[str, object]
    epistemic_comparison_memory: dict[str, object]


class CharacterIdentitySurfaceContextV1(TypedDict):
    """Closed latest-identity contexts assigned to output surfaces."""

    text: dict[str, object]
    visual: dict[str, object]
    naming: dict[str, object]


class IdentityRevisionConsumptionV1(TypedDict):
    """Durable first-episode receipt for one promoted revision."""

    episode_id: str
    correlation_id: str
    claimed_at: str
    loaded_revision_number: int
    consumer_kinds: list[str]
    projection_digest: str
    status: IdentityConsumptionStatus


class IdentityEpisodeSnapshotV1(TypedDict):
    """Latest identity and exact projections resolved once per episode."""

    revision_number: int
    character_profile: dict[str, object]
    cognition_context: CharacterIdentityCognitionContextV1
    surface_context: CharacterIdentitySurfaceContextV1
    projection_digest: str
    consumer_kinds: list[str]


class IdentityPatchV1(TypedDict, total=False):
    """Strict tagged replacement for one canonical identity path."""

    path: str
    value_kind: PatchValueKind
    replacement_text: str
    replacement_integer: int
    replacement_band: str
    replacement_enum: str
    replacement_items: list[str]


class IdentityChangeDiffV1(TypedDict):
    """Exact before/after diff retained on an immutable revision."""

    path: str
    value_kind: PatchValueKind
    before: object
    after: object


class IdentityEvidenceRefV1(TypedDict):
    """Repository-owned root provenance for identity evidence."""

    schema_version: str
    evidence_ref_id: str
    root_episode_id: str
    correlation_id: str
    source_kind: EvidenceSourceKind
    derived_reflection_run_ids: list[str]
    character_local_date: str
    scope_kind: EvidenceScopeKind
    captured_at: str


class IdentityEvidenceCountsV1(TypedDict):
    """Deterministic cadence counts derived from root references."""

    distinct_episode_count: int
    distinct_local_dates: list[str]


class IdentityEvidenceCardV1(TypedDict):
    """Prompt-safe semantic evidence joined to one repository reference."""

    schema_version: str
    evidence_ref_id: str
    source_kind: EvidenceSourceKind
    character_local_date: str
    scope_kind: EvidenceScopeKind
    decontextualized_event: str
    character_cognition_summary: str
    visible_self_expression_summary: str


class IdentityProposalDecisionV1(TypedDict):
    """Closed output from the identity proposal semantic stage."""

    schema_version: str
    action: ProposalAction
    candidate_id: str | None
    proposed_changes: list[IdentityPatchV1]
    character_authorship: CharacterAuthorship
    identity_relevance: IdentityRelevance
    global_applicability: GlobalApplicability
    confidence: Confidence
    private_detail_risk: PrivateDetailRisk
    character_owned_abstraction: str
    evidence_ref_ids: list[str]
    contradiction_candidate_ids: list[str]
    reason_code: str


class IdentityReviewDecisionV1(TypedDict):
    """Closed independent output from the identity review semantic stage."""

    schema_version: str
    verdict: ReviewVerdict
    selected_candidate_id: str | None
    rejected_candidate_ids: list[str]
    accepted_change_kind: AcceptedChangeKind | None
    accepted_changes: list[IdentityPatchV1]
    character_authorship: CharacterAuthorship
    identity_relevance: IdentityRelevance
    coherence: ReviewCoherence
    global_applicability: GlobalApplicability
    review_confidence: Confidence
    private_detail_risk: PrivateDetailRisk
    character_owned_summary: str
    privacy_safe_evidence_summaries: list[str]
    reason_code: str


class IdentityGrowthPolicyResultV1(TypedDict):
    """Sanitized deterministic disposition after both semantic stages."""

    status: IdentityPolicyStatus
    candidate_status: CandidateStatus | None
    candidate_id: str | None
    change_kind: AcceptedChangeKind | None
    accepted_changes: list[IdentityPatchV1]
    semantic_summary: str
    privacy_safe_evidence_summaries: list[str]
    evidence_refs: list[IdentityEvidenceRefV1]
    distinct_episode_count: int
    distinct_local_dates: list[str]
    source_scope_kinds: list[str]
    claimed_root_episode_ids: list[str]
    reversal_of_paths: list[str]
    fresh_post_revision_root_count: int
    rebase_required: bool
    rejected_candidate_ids: list[str]
    proposal_reason_code: str
    review_reason_code: str
    policy_reason_code: str


class IdentityGrowthEvaluationResultV1(TypedDict):
    """Sanitized outcome returned by the single growth orchestrator."""

    status: IdentityEvaluationStatus
    run_id: str
    candidate_id: str | None
    base_revision_number: int
    promoted_revision_number: int | None
    proposal_reason_code: str
    review_reason_code: str
    policy_reason_code: str
    persistence_reason_code: str
    validation_error_codes: list[str]
    attempt_count_by_stage: dict[str, int]
    source_evidence_count: int


class CharacterIdentityGrowthHealthV1(TypedDict):
    """Redacted operator health derived from the three identity ledgers."""

    state: IdentityGrowthHealthState
    routed_count: int
    no_change_count: int
    emerging_candidate_count: int
    ready_candidate_count: int
    rejected_count: int
    failed_count: int
    promoted_count: int
    consumed_count: int
    latest_revision_number: int
    latest_consumed_revision_number: int | None
    latest_reason_code: str
    root_count: int
    local_date_count: int
