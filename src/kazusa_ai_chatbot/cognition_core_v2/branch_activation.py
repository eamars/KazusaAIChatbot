"""Frozen goal-branch registry and goal-owned activation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

from kazusa_ai_chatbot.cognition_core_v2.contracts import BranchDefinition


def _branch(
    branch_id: str,
    dependencies: tuple[str, ...],
    tendencies: tuple[str, ...],
    *,
    goal_kind: str,
    dependency_options: tuple[tuple[str, ...], ...] = (),
) -> BranchDefinition:
    """Construct one registry row with explicit goal ownership."""

    return BranchDefinition(
        branch_id=branch_id,
        dependencies=dependencies,
        action_tendencies=tendencies,
        required=branch_id == "ordinary_response",
        goal_kind=goal_kind,
        dependency_options=dependency_options,
    )


DEFAULT_BRANCH_DEFINITIONS: dict[str, BranchDefinition] = {
    "ordinary_response": _branch(
        "ordinary_response", (), ("respond",), goal_kind="ordinary_response"
    ),
    "relationship_connection": _branch(
        "relationship_connection",
        ("q:relationship_social",),
        ("connect", "reciprocate"),
        goal_kind="relationship_connection",
    ),
    "bond_protection": _branch(
        "bond_protection",
        ("q:relationship_social", "q:goal_threat_outcome"),
        ("protect", "verify"),
        goal_kind="bond_protection",
    ),
    "trust_verification": _branch(
        "trust_verification",
        ("q:relationship_social", "q:goal_threat_outcome"),
        ("verify", "ask"),
        goal_kind="trust_verification",
    ),
    "autonomy_boundary": _branch(
        "autonomy_boundary",
        ("q:relationship_social",),
        ("set_boundary", "refuse"),
        goal_kind="autonomy_boundary",
        dependency_options=(
            ("q:relationship_social",),
            ("q:moral_identity",),
        ),
    ),
    "safety_coping": _branch(
        "safety_coping",
        ("q:goal_threat_outcome",),
        ("protect", "cope"),
        goal_kind="safety",
    ),
    "obstruction_strategy": _branch(
        "obstruction_strategy",
        ("q:goal_threat_outcome",),
        ("confront", "repair"),
        goal_kind="obstruction_resolution",
    ),
    "loss_recovery": _branch(
        "loss_recovery",
        ("q:goal_threat_outcome",),
        ("recover", "grieve"),
        goal_kind="loss_recovery",
    ),
    "moral_repair": _branch(
        "moral_repair",
        ("q:event_agency", "q:moral_identity"),
        ("repair", "apologize"),
        goal_kind="moral_repair",
    ),
    "social_care": _branch(
        "social_care",
        ("q:event_agency", "q:moral_identity"),
        ("support", "care"),
        goal_kind="social_care",
        dependency_options=(
            ("q:event_agency", "q:moral_identity"),
            ("q:event_agency", "q:goal_threat_outcome"),
        ),
    ),
    "reciprocal_response": _branch(
        "reciprocal_response",
        ("q:event_agency", "q:goal_threat_outcome"),
        ("reciprocate", "respond"),
        goal_kind="reciprocity",
    ),
    "epistemic_exploration": _branch(
        "epistemic_exploration",
        ("q:epistemic_comparison_memory",),
        ("explore", "ask"),
        goal_kind="epistemic_exploration",
    ),
    "meaning_reconstruction": _branch(
        "meaning_reconstruction",
        ("q:existential_drive",),
        ("reconstruct_meaning", "remember"),
        goal_kind="meaning_reconstruction",
    ),
    "self_improvement": _branch(
        "self_improvement",
        ("q:epistemic_comparison_memory",),
        ("learn", "improve"),
        goal_kind="self_improvement",
    ),
}
MAX_GOAL_BRANCHES = 14


def select_preliminary_branches(
    goals: Iterable[Mapping[str, object]] | Mapping[str, object],
    definitions: Mapping[str, BranchDefinition] = DEFAULT_BRANCH_DEFINITIONS,
) -> list[BranchDefinition]:
    """Select ordinary response plus branches for active persistent goals."""

    goal_kinds = _active_goal_kinds(goals)
    selected = [definitions["ordinary_response"]]
    selected.extend(
        definition
        for definition in definitions.values()
        if definition.branch_id != "ordinary_response"
        and definition.goal_kind in goal_kinds
    )
    return sorted(selected, key=lambda definition: definition.branch_id)[:MAX_GOAL_BRANCHES]


def select_final_branches(
    preliminary: Iterable[BranchDefinition],
    goals: Iterable[Mapping[str, object]] | Mapping[str, object],
    question_ids: Iterable[str] = (),
    definitions: Mapping[str, BranchDefinition] = DEFAULT_BRANCH_DEFINITIONS,
) -> list[BranchDefinition]:
    """Add active branches whose current appraisal dependencies are complete."""

    selected = {definition.branch_id: definition for definition in preliminary}
    available_questions = set(question_ids)
    active_goal_kinds = _active_goal_kinds(goals)
    for definition in definitions.values():
        if definition.branch_id == "ordinary_response":
            continue
        if definition.goal_kind not in active_goal_kinds:
            continue
        resolved = _resolve_dependencies(definition, available_questions)
        if resolved is not None:
            selected.setdefault(definition.branch_id, resolved)
    return sorted(selected.values(), key=lambda definition: definition.branch_id)[:MAX_GOAL_BRANCHES]


def _active_goal_kinds(
    goals: Iterable[Mapping[str, object]] | Mapping[str, object],
) -> set[str]:
    """Return goal kinds whose persistent state is pursuing or blocked."""

    rows = goals.get("goals", []) if isinstance(goals, Mapping) else goals
    if not isinstance(rows, Iterable):
        return set()
    return {
        str(goal["goal_kind"])
        for goal in rows
        if isinstance(goal, Mapping)
        and goal.get("status") in {"pursuing", "blocked"}
        and isinstance(goal.get("goal_kind"), str)
    }


def _resolve_dependencies(
    definition: BranchDefinition,
    available_questions: set[str],
) -> BranchDefinition | None:
    """Choose the first complete dependency option for a branch."""

    options = definition.dependency_options or (definition.dependencies,)
    for option in options:
        if set(option).issubset(available_questions):
            return replace(definition, dependencies=option)
    return None
