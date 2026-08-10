"""Direct ownership tests for resolver recurrence contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot.cognition_resolver.contracts import (
    CURRENT_TURN_RELATIONAL_WILLINGNESS_VERSION,
    RELATIONAL_WILLINGNESS_SCHEMA_VERSION,
    ResolverValidationError,
    validate_current_turn_relational_willingness,
)


def _decision() -> dict[str, object]:
    """Build one complete V2 decision for the recurrence carrier."""

    return {
        "schema_version": RELATIONAL_WILLINGNESS_SCHEMA_VERSION,
        "applicability": "relationship_sensitive",
        "stance": "accept",
        "current_user_relationship_state": "established",
        "reason": '角色根据当前证据作出判断',
        "evidence_handles": ["e1"],
    }


def _carrier() -> dict[str, object]:
    """Build the complete current-turn V2 carrier."""

    return {
        "schema_version": CURRENT_TURN_RELATIONAL_WILLINGNESS_VERSION,
        "episode_id": "episode-carrier",
        "branch_id": "ordinary_response",
        "decision": _decision(),
    }


def test_current_turn_carrier_preserves_complete_v2_decision() -> None:
    """Recurrence returns the complete validated decision unchanged."""

    carrier = _carrier()
    normalized = validate_current_turn_relational_willingness(
        carrier,
        episode_id="episode-carrier",
    )

    assert normalized["schema_version"] == (
        CURRENT_TURN_RELATIONAL_WILLINGNESS_VERSION
    )
    assert normalized["decision"] == carrier["decision"]


def test_current_turn_carrier_rejects_v1_or_incomplete_decision() -> None:
    """Recurrence fails closed for old or semantically incomplete carriers."""

    old_carrier = _carrier()
    old_carrier["schema_version"] = "current_turn_relational_willingness.v1"
    with pytest.raises(ResolverValidationError):
        validate_current_turn_relational_willingness(
            old_carrier,
            episode_id="episode-carrier",
        )

    incomplete_carrier = _carrier()
    incomplete_decision = deepcopy(incomplete_carrier["decision"])
    assert isinstance(incomplete_decision, dict)
    incomplete_decision.pop("evidence_handles")
    incomplete_carrier["decision"] = incomplete_decision
    with pytest.raises(ResolverValidationError):
        validate_current_turn_relational_willingness(
            incomplete_carrier,
            episode_id="episode-carrier",
        )


def test_contracts_exposes_owned_contract() -> None:
    """Keep the current-turn validator attached to this source owner."""

    assert callable(validate_current_turn_relational_willingness)
