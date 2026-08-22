"""Deterministic tests for the advisory Cognition V3 L1 residue."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import subconscious


def _residue() -> dict[str, object]:
    return {
        "schema_version": "l1_residue.v1",
        "emotional_appraisal": "the current request feels pointed",
        "interaction_subtext": "the user is testing a boundary",
        "salience_hints": ["e1"],
        "risk_flags": ["boundary_pressure"],
    }


def test_l1_is_advisory_nonblocking_and_handle_bounded() -> None:
    """L1 is closed, advisory, duplicate-free, and cannot invent handles."""

    residue = _residue()
    validated = subconscious.validate_l1_residue(
        residue,
        supplied_evidence_handles=["e1", "e2"],
    )
    assert validated["schema_version"] == "l1_residue.v1"
    assert validated["salience_hints"] == ["e1"]
    assert validated["risk_flags"] == ["boundary_pressure"]

    assert subconscious.try_validate_l1_residue(
        residue,
        supplied_evidence_handles=["e1"],
    ) == validated

    with pytest.raises(subconscious.L1ContractError, match="supplied handle"):
        subconscious.validate_l1_residue(
            residue,
            supplied_evidence_handles=["e2"],
        )

    unknown_flag = _residue()
    unknown_flag["risk_flags"] = ["invented_flag"]
    assert subconscious.try_validate_l1_residue(
        unknown_flag,
        supplied_evidence_handles=["e1"],
    ) is None

    duplicate_flags = _residue()
    duplicate_flags["risk_flags"] = [
        "boundary_pressure",
        "boundary_pressure",
    ]
    assert subconscious.try_validate_l1_residue(
        duplicate_flags,
        supplied_evidence_handles=["e1"],
    ) is None
