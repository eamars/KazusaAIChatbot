"""Immutable serialized Cognition V3 chain and appraisal-stage registry."""

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

APPRAISAL_A1_FAMILIES = (
    "event_agency",
    "goal_threat_outcome",
    "epistemic_comparison_memory",
)

APPRAISAL_A2_FAMILIES = (
    "relationship_social",
    "moral_identity",
    "existential_drive",
)

APPRAISAL_STAGE_FAMILIES = (
    ("A1", APPRAISAL_A1_FAMILIES),
    ("A2", APPRAISAL_A2_FAMILIES),
)

APPRAISAL_WORLD_FAMILIES = APPRAISAL_A1_FAMILIES
APPRAISAL_RELATION_FAMILIES = APPRAISAL_A2_FAMILIES
APPRAISAL_FAMILY_ORDER = APPRAISAL_A1_FAMILIES + APPRAISAL_A2_FAMILIES


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
    if tuple(step_id for step_id, _ in APPRAISAL_STAGE_FAMILIES) != (
        "A1",
        "A2",
    ):
        raise ValueError("appraisal stages must remain A1 then A2")
    if APPRAISAL_FAMILY_ORDER != (
        *APPRAISAL_A1_FAMILIES,
        *APPRAISAL_A2_FAMILIES,
    ):
        raise ValueError("appraisal family order is not stage-owned")
    if len(set(APPRAISAL_FAMILY_ORDER)) != len(APPRAISAL_FAMILY_ORDER):
        raise ValueError("appraisal family names must be unique")


_validate_registry()


__all__ = [
    "APPRAISAL_A1_FAMILIES",
    "APPRAISAL_A2_FAMILIES",
    "APPRAISAL_FAMILY_ORDER",
    "APPRAISAL_RELATION_FAMILIES",
    "APPRAISAL_STAGE_FAMILIES",
    "APPRAISAL_WORLD_FAMILIES",
    "SERIAL_CHAIN_STEPS",
]
