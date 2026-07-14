"""Internal contracts for the validation-local cognition core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, NotRequired, Protocol, TypedDict

from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMInvoker


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


class CognitionStateError(CognitionContractError):
    """Raised when deterministic state reduction violates an invariant."""


class CognitionExecutionError(CognitionContractError):
    """Raised when collapse or route execution cannot produce a valid result."""


class CognitionContextLimitError(CognitionContractError):
    """Raised when required model context remains over its frozen cap."""


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
    target_roles: list[RoleRefV2]
    evidence_handles: list[str]
    expected_consequences: list[str]
    confidence: str
    requested_route: Literal[
        "speech", "evidence", "action", "deferral", "silence"
    ]
    requested_action_kind: NotRequired[str]
    requested_resolver_capability: NotRequired[str]


class GoalBidDraftV2(TypedDict):
    """Model-owned branch draft before deterministic handle mapping."""

    intention: str
    desired_outcome: str
    concrete_detail: str
    reason: str
    target_role_handles: list[str]
    evidence_handles: list[str]
    expected_consequences: list[str]
    confidence: str
    requested_route: Literal[
        "speech", "evidence", "action", "deferral", "silence"
    ]
    requested_action_handle: NotRequired[str]
    requested_resolver_handle: NotRequired[str]


class SelectedIntentionV2(TypedDict):
    """Deterministic route and intention selected from a complete bid."""

    selected_branch_id: NotRequired[str]
    route: Literal["speech", "evidence", "action", "deferral", "silence"]
    intention: str
    target_roles: list[RoleRefV2]
    reason: str


class RouteDecisionV2(TypedDict):
    """Model output restricted to route and prompt-local capability handles."""

    selected_bid_handle: str
    route: Literal["speech", "evidence", "action", "deferral", "silence"]
    action_handle: NotRequired[str]
    resolver_handle: NotRequired[str]


class CollapsedIntentionV2(TypedDict):
    """Workspace result copied from complete internal bids."""

    primary_branch_id: str
    supporting_branch_ids: list[str]
    suppressed_branch_ids: list[str]
    primary_bid: ActionBidV2
    supporting_bids: list[ActionBidV2]
    competing_bids: list[ActionBidV2]
    residue: str


class WorkspaceDecisionV2(TypedDict):
    """Prompt-local workspace partition emitted by the collapse model."""

    primary_bid_handle: str
    supporting_bid_handles: list[str]
    suppressed_bid_handles: list[str]


class SemanticActionRequestV2(TypedDict):
    """Route-only action request; execution remains action-spec owned."""

    action_kind: str
    semantic_goal: str
    target_roles: list[RoleRefV2]
    evidence_handles: list[str]


class ResolverCapabilityRequestV2(TypedDict):
    """Route-only resolver request; execution remains resolver owned."""

    capability: str
    semantic_goal: str
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
    comparison_results: list[dict[str, Any]]
    changed_paths: list[str]


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
    critical_path_ms: int
    call_count: int
    total_ms: int
    warnings: list[str]


class CognitionCoreInputV2(TypedDict):
    """Public V2 cognition input contract."""

    schema_version: Literal["cognition_core_input.v2"]
    episode: Mapping[str, Any]
    state_scope: Literal["user", "character"]
    mutable_state: dict[str, Any]
    character_constraints: CharacterConstraintSnapshotV2
    relationship_context: NotRequired[dict[str, Any]]
    evidence: list[CognitionEvidenceV2]
    direct_facts: list[DirectFactV2]
    available_actions: list[ActionAffordanceV2]
    available_resolver_capabilities: list[ResolverAffordanceV2]
    scene_context: SceneContextV2


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
    resolver_progress: ResolverProgressV2
    residue: str
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
    status: Literal["completed", "failed", "unavailable"]
    semantic_result: str
    target_roles: list[RoleRefV2]


class TextSurfaceInputV2(TypedDict):
    """Public V2 text-surface input contract."""

    schema_version: Literal["text_surface_input.v2"]
    episode: Mapping[str, Any]
    intention: SelectedIntentionV2
    primary_bid: NotRequired[SurfaceBidProjectionV2]
    supporting_bids: list[SurfaceBidProjectionV2]
    expression_policy: ExpressionPolicyV2
    semantic_affect: list[SemanticAffectProjectionV2]
    semantic_relationship: NotRequired[SemanticRelationshipProjectionV2]
    permitted_action_results: list[SemanticActionResultV2]
    interaction_style_context: str


class TextSurfaceOutputV2(TypedDict):
    """Public V2 text-surface output contract."""

    schema_version: Literal["text_surface_output.v2"]
    content_plan: str
    visible_boundaries: list[str]
    addressee_plan: list[str]
    style_guidance: str
    pacing_guidance: str
    selected_surface_intent: str


class JsonParserV2(Protocol):
    """Parse one model response into a JSON-compatible object."""

    def __call__(self, content: str) -> Mapping[str, Any] | list[Any]:
        """Return parsed JSON or raise a parser error."""


class CognitionLoggerV2(Protocol):
    """Small logger boundary used by V2 services."""

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Record debug information."""

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Record informational information."""

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Record a recoverable boundary issue."""

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Record an unrecoverable boundary issue."""


@dataclass(frozen=True)
class CognitionCoreServicesV2:
    """Injected V2 model bindings; services never enter model payloads."""

    llm: LLMInvoker
    appraisal_config: LLMCallConfig
    goal_cognition_config: LLMCallConfig
    collapse_config: LLMCallConfig
    action_selection_config: LLMCallConfig
    parse_json: JsonParserV2
    logger: CognitionLoggerV2


@dataclass(frozen=True)
class TextSurfaceServicesV2:
    """Injected four-stage V2 L3 bindings."""

    llm: LLMInvoker
    style_config: LLMCallConfig
    content_plan_config: LLMCallConfig
    preference_config: LLMCallConfig
    visual_config: LLMCallConfig
    parse_json: JsonParserV2
    logger: CognitionLoggerV2


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
            "scene_context",
        } | ({"relationship_context"} if "relationship_context" in payload else set()),
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
    if not isinstance(payload["episode"], Mapping):
        raise CognitionContractError("episode must be a mapping")
    _validate_evidence_rows(payload["evidence"])
    if not isinstance(payload["direct_facts"], list):
        raise CognitionContractError("direct_facts must be a list")
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
    if not isinstance(payload["scene_context"], Mapping):
        raise CognitionContractError("scene_context must be a mapping")
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
            "resolver_progress",
            "residue",
            "expression_policy",
            "diagnostics",
        } | ({"admitted_bid"} if "admitted_bid" in payload else set())
        | ({"relationship_projection"} if "relationship_projection" in payload else set()),
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
    _validate_resolver_progress(payload["resolver_progress"])
    _validate_expression_policy(payload["expression_policy"])
    if "relationship_projection" in payload:
        _validate_relationship_projection(payload["relationship_projection"])
    _validate_diagnostics(payload["diagnostics"])
    _require_text(payload["residue"], "residue", maximum=1000)
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
        } | ({"primary_bid"} if "primary_bid" in payload else set())
        | ({"semantic_relationship"} if "semantic_relationship" in payload else set()),
        "text surface input",
    )
    if payload["schema_version"] != "text_surface_input.v2":
        raise CognitionContractError("unsupported text surface input schema")
    _validate_intention(payload["intention"])
    _require_text(payload["interaction_style_context"], "interaction style")
    if not isinstance(payload["episode"], Mapping):
        raise CognitionContractError("surface episode must be a mapping")
    return dict(payload)  # type: ignore[return-value]


def validate_text_surface_output(
    payload: Mapping[str, Any],
) -> TextSurfaceOutputV2:
    """Validate the bounded V2 L3 output."""

    required = {
        "schema_version",
        "content_plan",
        "visible_boundaries",
        "addressee_plan",
        "style_guidance",
        "pacing_guidance",
        "selected_surface_intent",
    }
    _require_exact_keys(payload, required, "text surface output")
    if payload["schema_version"] != "text_surface_output.v2":
        raise CognitionContractError("unsupported text surface output schema")
    for field_name in (
        "content_plan",
        "style_guidance",
        "pacing_guidance",
        "selected_surface_intent",
    ):
        _require_text(payload[field_name], field_name, maximum=1000)
    for field_name in ("visible_boundaries", "addressee_plan"):
        if not isinstance(payload[field_name], list):
            raise CognitionContractError(f"{field_name} must be a list")
    return dict(payload)  # type: ignore[return-value]


def _validate_persistent_state(state: Mapping[str, Any]) -> None:
    """Delegate exact native-state validation without a circular import."""

    from kazusa_ai_chatbot.cognition_core_v2.state_models import (
        validate_cognition_state,
    )

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
        if not isinstance(handle, str) or not handle or handle in seen:
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
        if not set(row["visible_to"]).issubset({
            "cognition",
            "surface",
            "dialog",
            "persistence",
        }):
            raise CognitionContractError("evidence visibility audience is invalid")


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
        "target_roles",
        "evidence_handles",
        "expected_consequences",
        "confidence",
        "requested_route",
    }
    optional = {"requested_action_kind", "requested_resolver_capability"}
    if set(value).difference(required | optional) or not required.issubset(value):
        raise CognitionContractError("action bid fields are not exact")
    for field_name in (
        "branch_id",
        "intention",
        "desired_outcome",
        "concrete_detail",
        "reason",
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
    if value["requested_route"] not in {
        "speech",
        "evidence",
        "action",
        "deferral",
        "silence",
    }:
        raise CognitionContractError("action bid route is invalid")
    for field_name in optional:
        if field_name in value:
            _require_text(value[field_name], f"action bid.{field_name}")


def _validate_action_request(value: Any) -> None:
    """Validate one route-approved semantic action request."""

    if not isinstance(value, Mapping) or set(value) != {
        "action_kind",
        "semantic_goal",
        "target_roles",
        "evidence_handles",
    }:
        raise CognitionContractError("action request fields are not exact")
    _require_text(value["action_kind"], "action request.action_kind")
    _require_text(value["semantic_goal"], "action request.semantic_goal")
    _validate_roles(value["target_roles"], "action request.target_roles")
    _validate_text_list(value["evidence_handles"], "action request.evidence_handles")


def _validate_resolver_request(value: Any) -> None:
    """Validate one route-approved semantic resolver request."""

    if not isinstance(value, Mapping) or set(value) != {
        "capability",
        "semantic_goal",
        "evidence_handles",
    }:
        raise CognitionContractError("resolver request fields are not exact")
    _require_text(value["capability"], "resolver request.capability")
    _require_text(value["semantic_goal"], "resolver request.semantic_goal")
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
        "critical_path_ms",
        "call_count",
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
        "target_roles",
    }:
        raise CognitionContractError("action affordance fields are not exact")
    _require_text(value["action_kind"], "action affordance.action_kind")
    _require_text(value["capability"], "action affordance.capability")
    _require_text(value["permission"], "action affordance.permission")
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
    _require_text(value["source_kind"], "evidence_ref.source_kind")
    _require_text(value["source_id"], "evidence_ref.source_id")
    _require_text(value["occurred_at"], "evidence_ref.occurred_at")
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
    _require_text(value["kind"], f"{label}.kind")
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
        _require_text(role["role"], f"{label}[{index}].role")
        _require_text(role["entity_kind"], f"{label}[{index}].entity_kind")
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
    comparison_keys = [
        (
            str(row["entity_kind"]),
            str(row["entity_id"]),
            str(row["outcome"]),
        )
        for row in value["comparison_results"]
    ]
    if comparison_keys != sorted(comparison_keys):
        raise CognitionContractError("comparison_results must be sorted")
    if not isinstance(value["changed_paths"], list):
        raise CognitionContractError("changed_paths must be a list")
    if any(not isinstance(path, str) or not path for path in value["changed_paths"]):
        raise CognitionContractError("changed_paths must contain text")
    if list(value["changed_paths"]) != sorted(set(value["changed_paths"])):
        raise CognitionContractError("changed_paths must be unique and sorted")


def _validate_comparison_result(value: Any) -> None:
    """Validate one deterministic causal comparison result."""

    required = {
        "entity_kind",
        "entity_id",
        "outcome",
        "evidence_source_ids",
        "reason",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CognitionContractError("comparison result fields are not exact")
    _require_text(value["entity_kind"], "comparison result.entity_kind")
    _require_text(value["entity_id"], "comparison result.entity_id")
    if value["outcome"] not in {"create", "reinforce", "contradict", "resolve", "replace"}:
        raise CognitionContractError("comparison result outcome is invalid")
    _validate_text_list(
        value["evidence_source_ids"],
        "comparison result.evidence_source_ids",
    )
    _require_text(value["reason"], "comparison result.reason")


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
