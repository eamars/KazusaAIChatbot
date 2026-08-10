"""Direct ownership tests for authoritative workspace arbitration."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v2.workspace import (
    collapse_authoritative_relational_bid,
)


def _decision() -> dict[str, object]:
    """Build one selected relational stance."""

    return {
        "schema_version": "relational_willingness.v2",
        "applicability": "relationship_sensitive",
        "stance": "accept",
        "current_user_relationship_state": "established",
        "reason": "the character selected this direction",
        "evidence_handles": ["e1"],
    }


def _bid(decision: dict[str, object], branch_id: str) -> dict[str, object]:
    """Build the fields used by deterministic authoritative collapse."""

    return {
        "branch_id": branch_id,
        "relational_willingness": dict(decision),
    }


def test_ordinary_response_remains_authoritative_stance_owner() -> None:
    """Competing branches cannot replace the ordinary typed stance owner."""

    decision = _decision()
    ordinary_bid = _bid(decision, "ordinary_response")
    competing_bid = _bid(decision, "autonomy_boundary")

    collapsed = collapse_authoritative_relational_bid(
        [ordinary_bid, competing_bid],
        decision,
    )

    assert collapsed["primary_branch_id"] == "ordinary_response"
    assert collapsed["primary_bid"] == ordinary_bid
    assert collapsed["competing_bids"] == [competing_bid]


def test_workspace_exposes_owned_contract() -> None:
    """Keep deterministic workspace collapse attached to this owner."""

    assert callable(collapse_authoritative_relational_bid)
