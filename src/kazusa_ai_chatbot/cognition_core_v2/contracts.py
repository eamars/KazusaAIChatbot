"""Internal contracts for the validation-local cognition core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class LocalStateKey:
    """Identify one isolated process-local motivational state cell."""

    character_global_id: str
    current_user_global_id: str
    trigger_source: str
    target_scope_fingerprint: str


@dataclass(frozen=True)
class TransitionProposal:
    """Describe one causally grounded state mutation for the local reducer."""

    entity_kind: str
    entity_ref: str
    expected_state_version: int
    transition_kind: str
    causal_source_refs: list[str]
    numeric_delta: dict[str, float]
    semantic_basis: str


@dataclass
class LocalMotivationalState:
    """Hold validation-only motivational entities and their shared version."""

    state_version: int = 0
    drives: dict[str, dict[str, float]] = field(default_factory=dict)
    goals: dict[str, dict[str, float]] = field(default_factory=dict)
    bonds: dict[str, dict[str, float]] = field(default_factory=dict)
    threats: dict[str, dict[str, float]] = field(default_factory=dict)
    standards: dict[str, dict[str, float]] = field(default_factory=dict)
    incidents: dict[str, dict[str, float]] = field(default_factory=dict)
    epistemic_state: dict[str, dict[str, float]] = field(default_factory=dict)
    meaning_state: dict[str, float] = field(default_factory=dict)
    emotion_activations: dict[str, "EmotionActivation"] = field(
        default_factory=dict,
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


@dataclass(frozen=True)
class EmotionActivation:
    """Represent one derived emotional activation without assigning authority."""

    emotion_id: str
    activation: float
    trend: str
    causal_source_refs: tuple[str, ...]


@dataclass(frozen=True)
class SemanticProposition:
    """Carry one structurally validated appraisal result into the reducer."""

    root_id: str
    present: bool
    causal_source_ref: str
    semantic_basis: str


@dataclass(frozen=True)
class BranchDefinition:
    """Describe the state conditions and dependencies for one goal branch."""

    branch_id: str
    activating_emotions: tuple[str, ...]
    dependencies: tuple[str, ...]
    action_tendencies: tuple[str, ...]
    required: bool = False


@dataclass(frozen=True)
class BranchResult:
    """Return a branch-owned bid without allowing direct state mutation."""

    branch_id: str
    action_bid: Mapping[str, str]
    perceived_meaning: str
    desired_outcome: str
    confidence: str


@dataclass(frozen=True)
class WorkspaceResult:
    """Capture the admitted bid selected for V1-compatible projection."""

    selected_bid_id: str | None
    public_intention: str
    internal_summary: str
    suppressed_bid_ids: tuple[str, ...]
