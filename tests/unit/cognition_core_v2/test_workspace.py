"""Direct ownership tests for authoritative workspace arbitration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.workspace import (
    collapse_authoritative_relational_bid,
    collapse_bids,
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


def _workspace_bid(branch_id: str) -> dict[str, object]:
    """Build the prompt fields used by model-backed workspace collapse."""

    return {
        "branch_id": branch_id,
        "goal_ref": {
            "scope": "user",
            "kind": "goal",
            "entity_id": f"goal:{branch_id}",
        },
        "intention": f"intention for {branch_id}",
        "desired_outcome": f"outcome for {branch_id}",
        "concrete_detail": f"detail for {branch_id}",
        "reason": f"reason for {branch_id}",
        "private_monologue": f"thought for {branch_id}",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": [f"consequence for {branch_id}"],
        "confidence": "high",
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
    assert callable(collapse_bids)


@pytest.mark.asyncio
async def test_workspace_collapse_does_not_rank_by_confidence_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workspace quality comparison receives no producer confidence field."""

    import kazusa_ai_chatbot.cognition_core_v2.workspace as workspace_module

    monkeypatch.setattr(
        workspace_module,
        "_record_workspace_trace",
        lambda **_: None,
    )

    class FakeLLM:
        """Capture the model-facing workspace packet."""

        def __init__(self) -> None:
            self.messages = []

        async def ainvoke(self, messages, *, config):
            del config
            self.messages = list(messages)
            return SimpleNamespace(content=json.dumps({
                "primary_bid_handle": "b1",
                "supporting_bid_handles": [],
                "suppressed_bid_handles": ["b2"],
            }))

    fake_llm = FakeLLM()
    services = SimpleNamespace(
        llm=fake_llm,
        workspace_collapse_config=SimpleNamespace(),
    )
    bids = [
        _workspace_bid("ordinary_response"),
        _workspace_bid("active_branch"),
    ]

    await collapse_bids(
        bids,
        services,
        current_event=[{"semantic_text": "current event"}],
        goal_context_by_ref={
            "goal:active_branch": {"description": "persistent goal"},
        },
    )

    prompt_payload = json.loads(fake_llm.messages[1].content)
    for bid in prompt_payload["bids"].values():
        assert "confidence" not in bid
    assert prompt_payload["bids"]["b1"]["intention"] == (
        "intention for ordinary_response"
    )
