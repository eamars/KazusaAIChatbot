"""Direct ownership tests for cognition contract validation."""

from __future__ import annotations

import pytest

import kazusa_ai_chatbot.cognition_core_v2.contracts as contracts_module
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS,
    RELATIONAL_STANCE_VALUES,
    RELATIONAL_CURRENT_USER_RELATIONSHIP_STATE_VALUES,
    validate_action_bid,
    validate_relational_willingness,
)
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
)


def _decision(stance: str, relationship_state: str) -> dict[str, object]:
    """Build a structurally valid sensitive relational decision."""

    return {
        "schema_version": "relational_willingness.v2",
        "applicability": "relationship_sensitive",
        "stance": stance,
        "current_user_relationship_state": relationship_state,
        "reason": '角色根据当前证据作出判断',
        "evidence_handles": ["e1"],
    }


def _selected_operation() -> dict[str, object]:
    """Build one complete selected response-operation carrier."""

    return {
        "operation": "the user gives the selected reward to the character",
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }


def _bid_with_selected_operation(
    selected_operation: dict[str, object],
) -> dict[str, object]:
    """Build a complete non-relational action bid for contract tests."""

    return {
        "branch_id": "active_branch",
        "goal_ref": {
            "scope": "user",
            "kind": "goal",
            "entity_id": "goal:selected-operation",
        },
        "intention": "preserve the selected response direction",
        "desired_outcome": "carry the selected operation forward",
        "concrete_detail": "use the typed selected operation",
        "reason": "the selected operation is authoritative",
        "private_monologue": "keep role direction stable",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the dialog receives the selected operation"],
        "confidence": "high",
        "selected_response_operation": selected_operation,
    }


def test_sensitive_relational_willingness_accepts_all_real_states_and_stances() -> None:
    """Every character-owned stance remains structurally valid in each state."""

    stances = tuple(
        stance
        for stance in RELATIONAL_STANCE_VALUES
        if stance != "not_applicable"
    )
    relationship_states = tuple(
        state
        for state in RELATIONAL_CURRENT_USER_RELATIONSHIP_STATE_VALUES
        if state != "not_applicable"
    )

    for relationship_state in relationship_states:
        for stance in stances:
            decision = _decision(stance, relationship_state)
            validated = validate_relational_willingness(decision)
            assert validated["stance"] == stance
            assert (
                validated["current_user_relationship_state"]
                == relationship_state
            )


def test_contracts_exposes_owned_contract() -> None:
    """Keep the relational validator and its failure type available."""

    assert CognitionContractError is not None
    assert callable(validate_relational_willingness)


def test_selected_response_operation_has_exact_contract() -> None:
    """Action bids accept the complete selected operation shape."""

    bid = _bid_with_selected_operation(_selected_operation())

    validate_action_bid(bid)

    assert bid["selected_response_operation"]["selection_required"] is True


def test_selected_response_operation_rejects_missing_required_fields() -> None:
    """A partial selected operation cannot cross the bid boundary."""

    selected_operation = _selected_operation()
    del selected_operation["embedded_target_role"]
    bid = _bid_with_selected_operation(selected_operation)

    with pytest.raises(
        CognitionContractError,
        match="selected_response_operation",
    ):
        validate_action_bid(bid)


def test_scheduler_events_are_current_episode_evidence() -> None:
    """Scheduled self-cognition can cite its current trigger event."""

    assert "scheduler_event" in CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS


def test_goal_and_action_confidence_reject_numeric_values() -> None:
    """V2 confidence stays a bounded descriptor rather than a quality score."""

    for value in (0.5, True):
        bid = _bid_with_selected_operation(_selected_operation())
        bid["confidence"] = value

        with pytest.raises(CognitionContractError, match="confidence"):
            validate_action_bid(bid)


def test_group_confidence_rejects_numeric_values() -> None:
    """Group participation confidence rejects numeric and boolean values."""

    for value in (0.5, True):
        with pytest.raises(CognitionContractError, match="confidence"):
            contracts_module._validate_group_engagement_action_context({
                "engagement_guidelines": ["scene guidance"],
                "confidence": value,
            })
