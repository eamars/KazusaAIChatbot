"""Direct ownership tests for resolver state preservation."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_resolver.contracts import (
    CURRENT_TURN_RELATIONAL_WILLINGNESS_VERSION,
    RELATIONAL_WILLINGNESS_SCHEMA_VERSION,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    new_resolver_state,
    validate_resolver_state,
)


def _carrier() -> dict[str, object]:
    """Build one complete current-turn relational carrier."""

    return {
        "schema_version": CURRENT_TURN_RELATIONAL_WILLINGNESS_VERSION,
        "episode_id": "state-episode",
        "branch_id": "ordinary_response",
        "decision": {
            "schema_version": RELATIONAL_WILLINGNESS_SCHEMA_VERSION,
            "applicability": "relationship_sensitive",
            "stance": "conditional_accept",
            "current_user_relationship_state": "developing_or_uncertain",
            "reason": '角色根据当前证据作出判断',
            "evidence_handles": ["e1"],
        },
    }


def test_current_turn_carrier_round_trips_without_semantic_reconstruction() -> None:
    """Resolver state preserves the complete authoritative decision."""

    state = new_resolver_state(
        decontextualized_input="preserve the current decision",
        max_cycles=3,
        episode_id="state-episode",
    )
    carrier = _carrier()
    state["current_turn_relational_willingness"] = carrier

    validated = validate_resolver_state(state)

    assert validated["current_turn_relational_willingness"] == carrier


def test_state_exposes_owned_contract() -> None:
    """Keep initial resolver state construction attached to this owner."""

    assert callable(new_resolver_state)
