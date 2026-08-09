"""Frozen goal-branch registry and goal-owned activation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

from kazusa_ai_chatbot.cognition_core_v2.contracts import BranchDefinition

DEFAULT_BRANCH_INTENT_GUIDANCE: dict[str, str] = {
    "ordinary_response": (
        '为当前事件提供中性的上下文基线；在适用时保留现有 '
        'relational_willingness 的归属，不引入其他分支的专门焦点。'
    ),
    "relationship_connection": (
        '评估是否以及如何通过自愿且符合当前情境的互惠参与来建立、维持、'
        '调整或修复人际连接。'
    ),
    "bond_protection": (
        '评估当前事件是否对重要关系纽带造成有证据支持的威胁或损害，并考虑'
        '相称的保护或修复。'
    ),
    "trust_verification": (
        '评估当前证据是否支持信任、保留信任或需要核实；不把不确定性直接解释'
        '为背叛。'
    ),
    "autonomy_boundary": (
        '评估当前事件是否对角色自身的自主权、意愿或明确边界造成有证据支持的'
        '压力或代价；在有依据时保护边界，不假定恶意。'
    ),
    "safety_coping": (
        '评估当前事件是否存在有证据支持的威胁或压力，并考虑相称的保护或应对；'
        '不凭空升级恐惧。'
    ),
    "obstruction_strategy": (
        '评估当前事件是否阻碍当前目标的进展，并考虑相称的解决、对抗或修复。'
    ),
    "loss_recovery": (
        '评估当前事件是否构成有依据的损失，并考虑恢复、适应或适当的哀悼；'
        '不强迫悲伤。'
    ),
    "moral_repair": (
        '评估当前角色是否对伤害负有有证据支持的责任；如有，考虑相称的修复或'
        '道歉。'
    ),
    "social_care": (
        '评估受当前事件影响的人是否有有依据的需要，并考虑相称的支持或照护；'
        '不强迫温柔。'
    ),
    "reciprocal_response": (
        '确定当前角色对另一方行为的有证据支持且相称的回应；互惠不等于服从，'
        '也不要求匹配情绪价性。'
    ),
    "epistemic_exploration": (
        '通过探索、提问或比较，减少当前有依据的不确定性并增进理解；区分求知'
        '与无依据的断言。'
    ),
    "meaning_reconstruction": (
        '在当前事件造成有依据的叙事或存在性中断后，评估如何重建连贯意义；'
        '不强迫乐观。'
    ),
    "self_improvement": (
        '评估当前角色是否有有证据支持的学习、纠错或能力发展机会；不预设缺陷、'
        '乐观或成功。'
    ),
}


def _branch(
    branch_id: str,
    dependencies: tuple[str, ...],
    tendencies: tuple[str, ...],
    *,
    goal_kind: str,
    dependency_options: tuple[tuple[str, ...], ...] = (),
) -> BranchDefinition:
    """Construct one registry row with explicit goal ownership."""

    branch_intent_guidance = DEFAULT_BRANCH_INTENT_GUIDANCE[branch_id]
    if not branch_intent_guidance:
        raise ValueError(
            f"default branch {branch_id} requires intent guidance"
        )
    return BranchDefinition(
        branch_id=branch_id,
        dependencies=dependencies,
        action_tendencies=tendencies,
        required=branch_id == "ordinary_response",
        goal_kind=goal_kind,
        dependency_options=dependency_options,
        branch_intent_guidance=branch_intent_guidance,
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
BRANCH_REGISTRY_ORDER = tuple(DEFAULT_BRANCH_DEFINITIONS)
_BRANCH_ORDER_INDEX = {
    branch_id: index
    for index, branch_id in enumerate(BRANCH_REGISTRY_ORDER)
}


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
    return sorted(selected, key=lambda definition: branch_order_key(
        definition.branch_id
    ))[:MAX_GOAL_BRANCHES]


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
    return sorted(
        selected.values(),
        key=lambda definition: branch_order_key(definition.branch_id),
    )[:MAX_GOAL_BRANCHES]


def branch_order_key(branch_id: str) -> tuple[int, str]:
    """Return the frozen registry position with a stable extension fallback."""

    return (_BRANCH_ORDER_INDEX.get(branch_id, len(_BRANCH_ORDER_INDEX)), branch_id)


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
