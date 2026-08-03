"""Frozen twenty-one-emotion scope, formula, and decay contract tests."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
    EMOTION_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    EMOTION_IDS,
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_character_operational_state,
)


CHARACTER_ROOT_ELIGIBLE = {
    "joy",
    "fear",
    "anger",
    "sadness",
    "disgust",
    "surprise",
    "compassion_empathy",
    "gratitude",
    "envy",
    "pride",
    "shame",
    "guilt",
    "embarrassment",
    "curiosity",
    "awe",
    "nostalgia",
    "relief",
    "ennui_existential_angst",
}

RELATIONSHIP_REQUIRED = {"love_attachment", "jealousy", "loneliness"}

EXPECTED_DEFINITIONS = {
    "joy": ("goal_reward", 4),
    "fear": ("credible_threat", 4),
    "anger": ("goal_obstruction", 4),
    "sadness": ("valued_loss", 1),
    "disgust": ("contamination_or_norm_rejection", 4),
    "surprise": ("prediction_error", 12),
    "love_attachment": ("bond_attachment", 1),
    "compassion_empathy": ("observed_other_affect", 1),
    "gratitude": ("attributed_benefit", 4),
    "jealousy": ("rival_threat", 4),
    "envy": ("upward_comparison", 4),
    "pride": ("self_caused_achievement", 4),
    "shame": ("global_standard_threat", 1),
    "guilt": ("self_caused_harm", 1),
    "embarrassment": ("minor_social_error", 12),
    "curiosity": ("valuable_knowledge_gap", 4),
    "awe": ("vastness", 12),
    "nostalgia": ("autobiographical_continuity", 1),
    "loneliness": ("connection_gap", 1),
    "relief": ("prior_threat_reduction", 12),
    "ennui_existential_angst": ("low_purpose_coherence", 1),
}


def test_registry_has_exactly_twenty_one_ids_and_two_scope_sets() -> None:
    """Freeze the registry membership and the 18/3 ownership split."""

    assert len(EMOTION_IDS) == 21
    assert set(EMOTION_IDS) == CHARACTER_ROOT_ELIGIBLE | RELATIONSHIP_REQUIRED
    assert len(CHARACTER_ROOT_ELIGIBLE) == 18
    assert len(RELATIONSHIP_REQUIRED) == 3
    assert set(EMOTION_DEFINITIONS) == set(EMOTION_IDS)


@pytest.mark.parametrize("emotion_id", tuple(EXPECTED_DEFINITIONS))
def test_definition_root_and_decay_rate_are_unchanged(emotion_id: str) -> None:
    """Require every row to retain its existing native root and rate."""

    definition = EMOTION_DEFINITIONS[emotion_id]
    expected_root, expected_rate = EXPECTED_DEFINITIONS[emotion_id]

    assert definition.causal_inputs == (expected_root,)
    assert definition.decay_rate_per_hour == expected_rate


def test_relationship_required_rows_retain_native_causal_kinds() -> None:
    """Keep relationship ownership while preserving each frozen guard."""

    expected_causal_kinds = {
        "love_attachment": ("relationship",),
        "jealousy": ("threat", "relationship"),
        "loneliness": ("relationship",),
    }
    for emotion_id in RELATIONSHIP_REQUIRED:
        assert EMOTION_DEFINITIONS[emotion_id].causal_entity_kinds == (
            expected_causal_kinds[emotion_id]
        )


def test_character_scope_has_no_user_owner_or_relationship_copy() -> None:
    """Character and user documents remain separate mutable scopes."""

    character_state = build_character_production_state(
        updated_at="2026-08-02T00:00:00Z",
    )
    user_state = build_acquaintance_user_state(
        global_user_id="user-scope",
        updated_at="2026-08-02T00:00:00Z",
    )

    assert character_state["state_scope"] == "character"
    assert "owner_user_id" not in character_state
    assert user_state["state_scope"] == "user"
    assert user_state["owner_user_id"] == "user-scope"
    assert "relationship" not in character_state


def test_character_view_keeps_relationship_required_ids_out_of_affect() -> None:
    """The full character projection must not copy user relationship affect."""

    character_state = build_character_production_state(
        updated_at="2026-08-02T00:00:00Z",
    )
    character_state["affect_activations"] = [
        {
            "activation_id": f"emotion:{emotion_id}",
            "emotion_id": emotion_id,
            "primary_root": {
                "scope": "character",
                "kind": "event",
                "entity_id": f"event:{emotion_id}",
            },
        }
        for emotion_id in sorted(CHARACTER_ROOT_ELIGIBLE)
    ]

    view = project_character_operational_state(
        character_state,
        effective_at="2026-08-02T01:00:00Z",
    )

    projected_ids = {row["emotion_id"] for row in view["affect"]}
    assert projected_ids.isdisjoint(RELATIONSHIP_REQUIRED)
