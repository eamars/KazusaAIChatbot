"""Deterministic tests for the serialized Cognition V3 registry."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v3 import registry


def test_serial_chain_steps_match_contract() -> None:
    assert registry.SERIAL_CHAIN_STEPS == (
        "A1",
        "A2",
        "I1",
        "G1a",
        "G1b",
        "I2",
        "W1",
        "P1",
    )


def test_appraisal_stages_own_the_fixed_family_rosters() -> None:
    assert registry.APPRAISAL_STAGE_FAMILIES == (
        (
            "A1",
            (
                "event_agency",
                "goal_threat_outcome",
                "epistemic_comparison_memory",
            ),
        ),
        (
            "A2",
            (
                "relationship_social",
                "moral_identity",
                "existential_drive",
            ),
        ),
    )


def test_appraisal_family_order_is_frozen() -> None:
    assert registry.APPRAISAL_FAMILY_ORDER == (
        "event_agency",
        "goal_threat_outcome",
        "epistemic_comparison_memory",
        "relationship_social",
        "moral_identity",
        "existential_drive",
    )
    assert registry.APPRAISAL_WORLD_FAMILIES == registry.APPRAISAL_A1_FAMILIES
    assert registry.APPRAISAL_RELATION_FAMILIES == registry.APPRAISAL_A2_FAMILIES
