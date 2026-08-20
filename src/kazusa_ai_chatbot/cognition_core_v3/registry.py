"""Immutable serialized Cognition V3 chain and appraisal-grouping registry."""

from __future__ import annotations

SERIAL_CHAIN_STEPS = (
    "A1",
    "A2",
    "I1",
    "G1a",
    "G1b",
    "I2",
    "W1",
    "P1",
)

APPRAISAL_FAMILY_ORDER = (
    "event_agency",
    "goal_threat_outcome",
    "epistemic_comparison_memory",
    "relationship_social",
    "moral_identity",
    "existential_drive",
)

APPRAISAL_WORLD_FAMILIES = APPRAISAL_FAMILY_ORDER

APPRAISAL_GROUPING_MAPS = {
    1: (("A1", APPRAISAL_FAMILY_ORDER),),
    2: (
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
            ("relationship_social", "moral_identity", "existential_drive"),
        ),
    ),
    3: (
        ("A1", ("event_agency", "goal_threat_outcome")),
        ("A1", ("epistemic_comparison_memory", "relationship_social")),
        ("A2", ("moral_identity", "existential_drive")),
    ),
    6: tuple(
        (step_id, (family_name,))
        for step_id, family_name in (
            ("A1", "event_agency"),
            ("A1", "goal_threat_outcome"),
            ("A1", "epistemic_comparison_memory"),
            ("A2", "relationship_social"),
            ("A2", "moral_identity"),
            ("A2", "existential_drive"),
        )
    ),
}

VALID_APPRAISAL_GROUP_COUNTS = (1, 2, 3, 6)


def _validate_registry() -> None:
    """Fail startup when the serial registry violates its own invariants."""

    if SERIAL_CHAIN_STEPS != (
        "A1",
        "A2",
        "I1",
        "G1a",
        "G1b",
        "I2",
        "W1",
        "P1",
    ):
        raise ValueError("serial chain step order deviates from the contract")
    if tuple(APPRAISAL_GROUPING_MAPS) != VALID_APPRAISAL_GROUP_COUNTS:
        raise ValueError("appraisal grouping map keys deviate from 1/2/3/6")
    expected_families = tuple(sorted(APPRAISAL_FAMILY_ORDER))
    for group_count, groups in APPRAISAL_GROUPING_MAPS.items():
        families: list[str] = []
        for step_id, grouped_families in groups:
            if step_id not in {"A1", "A2"}:
                raise ValueError("appraisal groups must use A1 or A2")
            families.extend(grouped_families)
        if tuple(sorted(set(families))) != expected_families:
            raise ValueError(
                f"group count {group_count} does not cover the six families exactly"
            )


_validate_registry()


__all__ = [
    "APPRAISAL_FAMILY_ORDER",
    "APPRAISAL_GROUPING_MAPS",
    "APPRAISAL_WORLD_FAMILIES",
    "SERIAL_CHAIN_STEPS",
    "VALID_APPRAISAL_GROUP_COUNTS",
]
