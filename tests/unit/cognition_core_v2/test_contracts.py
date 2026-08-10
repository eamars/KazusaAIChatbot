"""Direct ownership tests for cognition contract validation."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS,
    RELATIONAL_STANCE_VALUES,
    RELATIONAL_CURRENT_USER_RELATIONSHIP_STATE_VALUES,
    validate_relational_willingness,
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


def test_scheduler_events_are_current_episode_evidence() -> None:
    """Scheduled self-cognition can cite its current trigger event."""

    assert "scheduler_event" in CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS
