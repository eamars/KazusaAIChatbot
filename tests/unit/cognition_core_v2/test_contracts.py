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


def _evidence_row(
    *,
    authority: str = "contextual_fact_only",
    source_id: str = "resolver-1",
) -> dict[str, object]:
    """Build one complete row for the closed authority contract."""

    return {
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "resolver_observation",
            "source_id": source_id,
            "occurred_at": "2026-07-30T00:00:00Z",
            "semantic_summary": "bounded resolver context",
        },
        "semantic_text": "bounded resolver context",
        "visible_to": list(
            contracts_module.EVIDENCE_SOURCE_QUESTION_IDS[
                "resolver_observation"
            ]
        ),
        "authority": authority,
    }


def test_cognition_evidence_requires_closed_typed_authority() -> None:
    """Missing or unknown authority cannot cross the evidence boundary."""

    missing = _evidence_row()
    missing.pop("authority")
    with pytest.raises(CognitionContractError, match="fields are not exact"):
        contracts_module._validate_evidence_rows([missing])

    invalid = _evidence_row(authority="free_text_authority")
    with pytest.raises(CognitionContractError, match="authority is invalid"):
        contracts_module._validate_evidence_rows([invalid])


def test_promoted_reflection_projects_conditional_guidance_authority() -> None:
    """Self-guidance is accepted only with its scoped promoted-reflection id."""

    row = _evidence_row(
        authority="conditional_character_guidance",
        source_id="promoted-reflection:self_guidance:1",
    )
    row["evidence_ref"] = {
        **row["evidence_ref"],
        "source_kind": "promoted_reflection",
    }
    row["visible_to"] = list(
        contracts_module.EVIDENCE_SOURCE_QUESTION_IDS[
            "promoted_reflection"
        ]
    )

    contracts_module._validate_evidence_rows([row])
