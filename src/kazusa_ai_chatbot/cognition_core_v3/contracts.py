"""Canonical contracts for the single-pass Cognition V3 boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMInvoker

CANONICAL_COGNITION_INPUT_SCHEMA = "cognition_input.v3"
CANONICAL_COGNITION_OUTPUT_SCHEMA = "cognition_output.v3"
CANONICAL_SHIFT_VALUES = frozenset({
    "slight_increase", "moderate_increase", "strong_increase",
    "slight_decrease", "moderate_decrease", "strong_decrease",
    "stable", "uncertain",
})
CANONICAL_APPRAISAL_FAMILIES = (
    "event_agency", "goal_threat_outcome", "epistemic_comparison_memory",
    "relationship_social", "moral_identity", "existential_drive",
)
CANONICAL_A1_FAMILIES = CANONICAL_APPRAISAL_FAMILIES[:3]
CANONICAL_A2_FAMILIES = CANONICAL_APPRAISAL_FAMILIES[3:]
CANONICAL_FAMILY_AXES: dict[str, tuple[str, ...]] = {
    "event_agency": ("responsibility", "intentionality"),
    "goal_threat_outcome": (
        "obstruction", "expected_success", "controllability", "recoverability",
        "urgency", "likelihood", "expected_harm", "uncertainty",
        "coping_potential", "residual_pressure", "outcome_impact",
        "expectation_mismatch",
    ),
    "epistemic_comparison_memory": (
        "comparison_gap", "vastness", "memory_warmth", "temporal_loss",
        "relevance", "uncertainty", "learnability", "novelty",
        "model_accommodation",
    ),
    "relationship_social": (
        "positive_regard", "trust", "attachment", "desired_closeness",
        "perceived_closeness", "care", "boundary_safety", "exclusivity",
        "unresolved_injury",
    ),
    "moral_identity": (
        "harm", "unfairness", "exposure", "repair_need", "reparability",
        "norm_violation", "contamination_risk", "identity_threat",
    ),
    "existential_drive": (
        "autonomy_pressure", "connection_pressure", "safety_pressure",
        "competence_pressure", "care_pressure", "integrity_pressure",
        "exploration_pressure", "meaning_pressure", "purpose_coherence",
        "agency", "identity_continuity",
    ),
}

@dataclass(frozen=True)
class CanonicalTurnWorkspace:
    observation: Mapping[str, object]
    state: Mapping[str, object]
    continuity: Mapping[str, object]
    capabilities: Mapping[str, object]
    orientation: Mapping[str, str]

@dataclass(frozen=True)
class CanonicalAppraisal:
    family: str
    applicable: bool
    semantic_summary: str
    cause_summary: str
    axis_changes: tuple[Mapping[str, object], ...]

@dataclass(frozen=True)
class CanonicalGoal:
    goal_kind: str
    intent: str
    reason: str
    cause_summary: str

@dataclass(frozen=True)
class CanonicalResponsePlan:
    goal_resolution: str
    response_goal: str
    action_requests: tuple[Mapping[str, object], ...]
    resolver_requests: tuple[Mapping[str, object], ...]
    epistemic_boundary: str
    self_cognition_response: Mapping[str, object] | None = None

@dataclass(frozen=True)
class CanonicalCognitionOutput:
    schema_version: str
    appraisals: tuple[CanonicalAppraisal, ...]
    active_character_goal: CanonicalGoal
    relational_willingness: Mapping[str, object]
    private_monologue: str
    response_plan: CanonicalResponsePlan
    affect_projection: tuple[Mapping[str, object], ...]
    relationship_projection: Mapping[str, object]
    cause_provenance: tuple[Mapping[str, object], ...]
    diagnostics: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "appraisals": [
                {
                    "family": item.family,
                    "applicable": item.applicable,
                    "semantic_summary": item.semantic_summary,
                    "cause_summary": item.cause_summary,
                    "axis_changes": [dict(row) for row in item.axis_changes],
                }
                for item in self.appraisals
            ],
            "active_character_goal": {
                "goal_kind": self.active_character_goal.goal_kind,
                "intent": self.active_character_goal.intent,
                "reason": self.active_character_goal.reason,
                "cause_summary": self.active_character_goal.cause_summary,
            },
            "relational_willingness": dict(self.relational_willingness),
            "private_monologue": self.private_monologue,
            "response_plan": {
                "goal_resolution": self.response_plan.goal_resolution,
                "response_goal": self.response_plan.response_goal,
                "action_requests": [dict(row) for row in self.response_plan.action_requests],
                "resolver_requests": [dict(row) for row in self.response_plan.resolver_requests],
                "epistemic_boundary": self.response_plan.epistemic_boundary,
            },
            "affect_projection": [dict(row) for row in self.affect_projection],
            "relationship_projection": dict(self.relationship_projection),
            "cause_provenance": [dict(row) for row in self.cause_provenance],
            "diagnostics": dict(self.diagnostics),
        }
        if self.response_plan.self_cognition_response is not None:
            result["response_plan"]["self_cognition_response"] = dict(
                self.response_plan.self_cognition_response
            )
        return result

def validate_canonical_cognition_output(
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    required = {
        "schema_version", "appraisals", "active_character_goal",
        "relational_willingness", "private_monologue", "response_plan",
        "affect_projection", "relationship_projection", "cause_provenance",
        "diagnostics",
    }
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CANONICAL_COGNITION_OUTPUT_SCHEMA:
        raise ValueError("canonical cognition output schema is invalid")
    if set(payload) != required | {"state_projection"}:
        raise ValueError("canonical cognition output fields are not exact")
    if not isinstance(payload["active_character_goal"], Mapping):
        raise ValueError("canonical goal is invalid")
    if not isinstance(payload["response_plan"], Mapping):
        raise ValueError("canonical response plan is invalid")
    validate_canonical_state_projection(payload["state_projection"])
    return payload


def validate_canonical_state_projection(value: object) -> Mapping[str, object]:
    """Validate the private compare-and-replace carrier shape."""

    if not isinstance(value, Mapping):
        raise ValueError("canonical state projection is invalid")
    required = {
        "state_scope",
        "owner_key",
        "expected_previous_state",
        "original_persisted_state",
        "replacement_state",
        "transition_contexts",
        "binding_receipts",
        "capacity_deferred",
    }
    allowed = required | {"continuation_goal_ref"}
    if set(value) - allowed or required - set(value):
        raise ValueError("canonical state projection fields are not exact")
    if value["state_scope"] not in {"user", "character"}:
        raise ValueError("canonical state projection scope is invalid")
    if not isinstance(value["owner_key"], str):
        raise ValueError("canonical state projection owner is invalid")
    for field in ("expected_previous_state", "original_persisted_state", "replacement_state"):
        if not isinstance(value[field], Mapping):
            raise ValueError(f"canonical state projection {field} is invalid")
    for field in ("transition_contexts", "binding_receipts", "capacity_deferred"):
        if not isinstance(value[field], list):
            raise ValueError(f"canonical state projection {field} is invalid")
    continuation = value.get("continuation_goal_ref")
    if continuation is not None:
        if not isinstance(continuation, Mapping) or set(continuation) != {
            "scope", "kind", "entity_id",
        }:
            raise ValueError("canonical continuation goal reference is invalid")
    return value

_MINIMUM_CHAIN_CONTEXT_WINDOW_TOKENS = 50_000
_MINIMUM_LANE_COMPLETION_TOKENS = 8_192

def _validate_lane(config: object, *, label: str, context_required: bool) -> None:
    if not isinstance(config, LLMCallConfig):
        raise TypeError(f"V3 {label} lane must be an LLMCallConfig")
    for name in ("route_name", "base_url", "api_key", "model"):
        if not isinstance(getattr(config, name), str) or not getattr(config, name).strip():
            raise ValueError(f"V3 {label} route {name} must be non-empty")
    if config.thinking.enabled:
        raise ValueError(f"V3 {label} thinking must be disabled")
    if not isinstance(config.max_completion_tokens, int) or config.max_completion_tokens < _MINIMUM_LANE_COMPLETION_TOKENS:
        raise ValueError("V3 lane completion cap is too small")
    if context_required and (
        not isinstance(config.context_window_tokens, int)
        or config.context_window_tokens < _MINIMUM_CHAIN_CONTEXT_WINDOW_TOKENS
    ):
        raise ValueError("V3 chain context window is too small")

@dataclass(frozen=True)
class CognitionChainServicesV3:
    llm: LLMInvoker
    chain_lane: LLMCallConfig
    turn_deadline_seconds: int = 240

    def __post_init__(self) -> None:
        if self.llm is None:
            raise TypeError("V3 services require an LLM invoker")
        if not isinstance(self.turn_deadline_seconds, int) or isinstance(self.turn_deadline_seconds, bool):
            raise TypeError("turn_deadline_seconds must be an integer")
        if not 30 <= self.turn_deadline_seconds <= 600:
            raise ValueError("turn deadline is outside the bounded range")
        _validate_lane(self.chain_lane, label="chain", context_required=True)

__all__ = [
    "CANONICAL_A1_FAMILIES", "CANONICAL_A2_FAMILIES",
    "CANONICAL_APPRAISAL_FAMILIES", "CANONICAL_COGNITION_INPUT_SCHEMA",
    "CANONICAL_COGNITION_OUTPUT_SCHEMA", "CANONICAL_FAMILY_AXES",
    "CANONICAL_SHIFT_VALUES", "CanonicalAppraisal", "CanonicalCognitionOutput",
    "CanonicalGoal", "CanonicalResponsePlan", "CanonicalTurnWorkspace",
    "CognitionChainServicesV3", "validate_canonical_cognition_output",
    "validate_canonical_state_projection",
]
