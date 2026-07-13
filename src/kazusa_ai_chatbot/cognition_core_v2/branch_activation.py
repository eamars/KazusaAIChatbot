"""State-triggered goal branch selection for validation-local cognition."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    BranchDefinition,
    EmotionActivation,
)


DEFAULT_BRANCH_DEFINITIONS: dict[str, BranchDefinition] = {
    "ordinary_conversation": BranchDefinition(
        branch_id="ordinary_conversation",
        activating_emotions=(),
        dependencies=(),
        action_tendencies=("respond",),
    ),
    "safety_coping": BranchDefinition(
        branch_id="safety_coping",
        activating_emotions=("fear",),
        dependencies=(),
        action_tendencies=("protect", "avoid"),
    ),
    "obstruction_strategy": BranchDefinition(
        branch_id="obstruction_strategy",
        activating_emotions=("anger",),
        dependencies=(),
        action_tendencies=("confront", "repair"),
    ),
    "loss_processing": BranchDefinition(
        branch_id="loss_processing",
        activating_emotions=("sadness",),
        dependencies=(),
        action_tendencies=("grieve", "withdraw"),
    ),
    "bond_protection": BranchDefinition(
        branch_id="bond_protection",
        activating_emotions=("love_attachment", "jealousy", "loneliness"),
        dependencies=(),
        action_tendencies=("connect", "protect"),
    ),
    "moral_repair": BranchDefinition(
        branch_id="moral_repair",
        activating_emotions=("guilt", "shame", "embarrassment"),
        dependencies=(),
        action_tendencies=("repair", "apologize"),
    ),
    "epistemic_exploration": BranchDefinition(
        branch_id="epistemic_exploration",
        activating_emotions=("curiosity", "awe"),
        dependencies=(),
        action_tendencies=("explore", "ask"),
    ),
    "meaning_reconstruction": BranchDefinition(
        branch_id="meaning_reconstruction",
        activating_emotions=("ennui_existential_angst", "nostalgia"),
        dependencies=(),
        action_tendencies=("reconstruct_meaning", "remember"),
    ),
    "social_care": BranchDefinition(
        branch_id="social_care",
        activating_emotions=("compassion_empathy", "gratitude"),
        dependencies=(),
        action_tendencies=("support", "reciprocate"),
    ),
}
MAX_GOAL_BRANCHES = 6


def select_preliminary_branches(
    activations: Mapping[str, EmotionActivation],
    definitions: Mapping[str, BranchDefinition] = DEFAULT_BRANCH_DEFINITIONS,
) -> list[BranchDefinition]:
    """Select branches supported by currently committed causal activations.

    Args:
        activations: Deterministic emotion projections from committed state.
        definitions: Explicit local branch registry used for validation.

    Returns:
        Ready branch definitions, or the ordinary branch when none activate.
    """

    active_emotions = {
        emotion_id
        for emotion_id, activation in activations.items()
        if activation.activation > 0.0 and activation.trend != "inactive"
    }
    selected = [
        definition
        for definition in definitions.values()
        if definition.activating_emotions
        and active_emotions.intersection(definition.activating_emotions)
    ]
    if not selected and "ordinary_conversation" in definitions:
        selected.append(definitions["ordinary_conversation"])
    ordered_definitions = sorted(selected, key=lambda definition: definition.branch_id)
    bounded_definitions = ordered_definitions[:MAX_GOAL_BRANCHES]
    return bounded_definitions


def select_final_branches(
    preliminary: Iterable[BranchDefinition],
    activations: Mapping[str, EmotionActivation],
    definitions: Mapping[str, BranchDefinition] = DEFAULT_BRANCH_DEFINITIONS,
) -> list[BranchDefinition]:
    """Add semantic-dependent branches while preserving a stable branch order."""

    selected_by_id = {definition.branch_id: definition for definition in preliminary}
    for definition in select_preliminary_branches(activations, definitions):
        selected_by_id.setdefault(definition.branch_id, definition)
    selected_definitions = sorted(
        selected_by_id.values(),
        key=lambda definition: definition.branch_id,
    )
    return selected_definitions
