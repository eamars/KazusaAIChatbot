"""Causal definitions for the approved validation emotion matrix."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v2.contracts import EmotionDefinition


def _definition(
    emotion_id: str,
    root: str,
    tendencies: tuple[str, ...],
) -> EmotionDefinition:
    """Create one uniform lifecycle definition for a named emotion family."""

    definition = EmotionDefinition(
        emotion_id=emotion_id,
        causal_inputs=(root,),
        begin_guard=f"{root} is present",
        sustain_rule=f"{root} remains unresolved",
        fade_rule=f"{root} resolves or decays",
        action_tendencies=tendencies,
    )
    return definition


EMOTION_DEFINITIONS: dict[str, EmotionDefinition] = {
    "joy": _definition("joy", "goal_reward", ("approach", "share")),
    "fear": _definition("fear", "credible_threat", ("protect", "avoid")),
    "anger": _definition("anger", "goal_obstruction", ("confront", "repair")),
    "sadness": _definition("sadness", "valued_loss", ("grieve", "withdraw")),
    "disgust": _definition(
        "disgust",
        "contamination_or_norm_rejection",
        ("distance", "reject"),
    ),
    "surprise": _definition("surprise", "prediction_error", ("orient",)),
    "love_attachment": _definition(
        "love_attachment",
        "bond_attachment",
        ("care", "connect"),
    ),
    "compassion_empathy": _definition(
        "compassion_empathy",
        "observed_other_affect",
        ("support", "care"),
    ),
    "gratitude": _definition(
        "gratitude",
        "attributed_benefit",
        ("reciprocate", "acknowledge"),
    ),
    "jealousy": _definition("jealousy", "rival_threat", ("protect", "clarify")),
    "envy": _definition("envy", "upward_comparison", ("improve",)),
    "pride": _definition(
        "pride",
        "self_caused_achievement",
        ("acknowledge", "continue"),
    ),
    "shame": _definition(
        "shame",
        "global_standard_threat",
        ("repair", "withdraw"),
    ),
    "guilt": _definition("guilt", "self_caused_harm", ("repair", "apologize")),
    "embarrassment": _definition(
        "embarrassment",
        "minor_social_error",
        ("repair", "soften"),
    ),
    "curiosity": _definition(
        "curiosity",
        "valuable_knowledge_gap",
        ("explore", "ask"),
    ),
    "awe": _definition("awe", "vastness", ("attend", "integrate")),
    "nostalgia": _definition(
        "nostalgia",
        "autobiographical_continuity",
        ("remember", "connect"),
    ),
    "loneliness": _definition("loneliness", "connection_gap", ("connect",)),
    "relief": _definition(
        "relief",
        "prior_threat_reduction",
        ("settle", "continue"),
    ),
    "ennui_existential_angst": _definition(
        "ennui_existential_angst",
        "low_purpose_coherence",
        ("reconstruct_meaning",),
    ),
}
