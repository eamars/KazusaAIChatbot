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


def test_appraisal_grouping_maps_cover_six_families_exactly() -> None:
    assert registry.VALID_APPRAISAL_GROUP_COUNTS == (1, 2, 3, 6)

    for group_count, groups in registry.APPRAISAL_GROUPING_MAPS.items():
        families: list[str] = []
        for step_id, grouped_families in groups:
            assert step_id in {"A1", "A2"}
            families.extend(grouped_families)
        assert tuple(sorted(set(families))) == tuple(
            sorted(registry.APPRAISAL_FAMILY_ORDER)
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
    assert registry.APPRAISAL_WORLD_FAMILIES == registry.APPRAISAL_FAMILY_ORDER
