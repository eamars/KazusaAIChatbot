"""Deterministic causal emotion activation and lifecycle projection."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EmotionActivation,
    LocalMotivationalState,
)
from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
    EMOTION_DEFINITIONS,
)


FADE_MULTIPLIER = 0.4


def derive_emotion_activations(
    state: LocalMotivationalState,
    root_strengths: Mapping[str, float],
) -> dict[str, EmotionActivation]:
    """Derive all concurrent activations from causal roots and prior residue.

    Args:
        state: Committed local state holding previous emotion activations.
        root_strengths: Normalized causal roots produced by deterministic state.

    Returns:
        One derived activation per approved emotion family.
    """

    activations: dict[str, EmotionActivation] = {}
    for emotion_id, definition in EMOTION_DEFINITIONS.items():
        root_id = definition.causal_inputs[0]
        root_strength = root_strengths.get(root_id, 0.0)
        previous = state.emotion_activations.get(emotion_id)
        activations[emotion_id] = _derive_activation(
            emotion_id,
            root_id,
            root_strength,
            previous,
        )
    return activations


def _derive_activation(
    emotion_id: str,
    root_id: str,
    root_strength: float,
    previous: EmotionActivation | None,
) -> EmotionActivation:
    """Calculate one causal activation trend without interpreting user text."""

    if root_strength > 0.0:
        trend = "beginning" if previous is None or previous.activation == 0.0 else "sustained"
        activation = max(0.0, min(1.0, root_strength))
        source_refs = (root_id,)
    elif previous is not None and previous.activation > 0.0:
        trend = "fading"
        activation = previous.activation * FADE_MULTIPLIER
        source_refs = previous.causal_source_refs
    else:
        trend = "inactive"
        activation = 0.0
        source_refs = ()
    derived_activation = EmotionActivation(
        emotion_id=emotion_id,
        activation=activation,
        trend=trend,
        causal_source_refs=source_refs,
    )
    return derived_activation
