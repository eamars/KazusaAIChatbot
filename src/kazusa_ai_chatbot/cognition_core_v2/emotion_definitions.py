"""Causal definitions for the approved validation emotion matrix."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v2.contracts import EmotionDefinition


def _definition(
    emotion_id: str,
    root: str,
    tendencies: tuple[str, ...],
    decay_rate_per_hour: int,
    causal_entity_kinds: tuple[str, ...],
) -> EmotionDefinition:
    """Create one typed lifecycle definition for a named emotion family."""

    definition = EmotionDefinition(
        emotion_id=emotion_id,
        causal_inputs=(root,),
        begin_guard=f"{root} is present",
        sustain_rule=f"{root} remains unresolved",
        fade_rule=f"{root} resolves or decays",
        action_tendencies=tendencies,
        decay_rate_per_hour=decay_rate_per_hour,
        causal_entity_kinds=causal_entity_kinds,
    )
    return definition


EMOTION_DEFINITIONS: dict[str, EmotionDefinition] = {
    "joy": _definition(
        "joy", "goal_reward", ("approach", "share"), 4, ("event", "goal")
    ),
    "fear": _definition(
        "fear", "credible_threat", ("protect", "avoid"), 4, ("threat",)
    ),
    "anger": _definition(
        "anger",
        "goal_obstruction",
        ("confront", "repair"),
        4,
        ("event", "goal", "relationship"),
    ),
    "sadness": _definition(
        "sadness", "valued_loss", ("grieve", "withdraw"), 1, ("event", "goal")
    ),
    "disgust": _definition(
        "disgust",
        "contamination_or_norm_rejection",
        ("distance", "reject"),
        4,
        ("event",),
    ),
    "surprise": _definition(
        "surprise", "prediction_error", ("orient",), 12, ("event",)
    ),
    "love_attachment": _definition(
        "love_attachment",
        "bond_attachment",
        ("care", "connect"),
        1,
        ("relationship",),
    ),
    "compassion_empathy": _definition(
        "compassion_empathy",
        "observed_other_affect",
        ("support", "care"),
        1,
        ("event",),
    ),
    "gratitude": _definition(
        "gratitude",
        "attributed_benefit",
        ("reciprocate", "acknowledge"),
        4,
        ("event",),
    ),
    "jealousy": _definition(
        "jealousy",
        "rival_threat",
        ("protect", "clarify"),
        4,
        ("threat", "relationship"),
    ),
    "envy": _definition(
        "envy", "upward_comparison", ("improve",), 4, ("event", "goal")
    ),
    "pride": _definition(
        "pride",
        "self_caused_achievement",
        ("acknowledge", "continue"),
        4,
        ("event", "goal"),
    ),
    "shame": _definition(
        "shame",
        "global_standard_threat",
        ("repair", "withdraw"),
        1,
        ("event",),
    ),
    "guilt": _definition(
        "guilt", "self_caused_harm", ("repair", "apologize"), 1, ("event",)
    ),
    "embarrassment": _definition(
        "embarrassment",
        "minor_social_error",
        ("repair", "soften"),
        12,
        ("event",),
    ),
    "curiosity": _definition(
        "curiosity",
        "valuable_knowledge_gap",
        ("explore", "ask"),
        4,
        ("knowledge_gap",),
    ),
    "awe": _definition(
        "awe", "vastness", ("attend", "integrate"), 12, ("event", "knowledge_gap")
    ),
    "nostalgia": _definition(
        "nostalgia",
        "autobiographical_continuity",
        ("remember", "connect"),
        1,
        ("event",),
    ),
    "loneliness": _definition(
        "loneliness", "connection_gap", ("connect",), 1, ("relationship",)
    ),
    "relief": _definition(
        "relief",
        "prior_threat_reduction",
        ("settle", "continue"),
        12,
        ("event", "threat"),
    ),
    "ennui_existential_angst": _definition(
        "ennui_existential_angst",
        "low_purpose_coherence",
        ("reconstruct_meaning",),
        1,
        ("meaning",),
    ),
}
