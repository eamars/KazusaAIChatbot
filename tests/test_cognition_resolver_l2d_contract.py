"""V2 resolver route-selection contract tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import select_route


class _ResolverRouteLLM:
    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages, config
        return SimpleNamespace(content=json.dumps({
            "selected_bid_handle": "b1",
            "route": "evidence",
            "resolver_handle": "r1",
        }))


@pytest.mark.asyncio
async def test_v2_route_returns_typed_resolver_request() -> None:
    """The route model selects an available resolver by prompt-local handle."""

    primary_bid = {
        "branch_id": "epistemic_exploration",
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        "intention": "resolve the relevant knowledge gap",
        "desired_outcome": "obtain grounded local context",
        "concrete_detail": "retrieve only prompt-safe evidence",
        "reason": "the current evidence is incomplete",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["reduce uncertainty"],
        "confidence": "high",
        "requested_route": "evidence",
        "requested_resolver_capability": "local_context_recall",
    }
    services = SimpleNamespace(
        llm=_ResolverRouteLLM(),
        action_selection_config=object(),
    )

    intention, action_requests, resolver_requests = await select_route(
        primary_bid,
        [],
        [],
        [{
            "capability": "local_context_recall",
            "semantic_capability": "recall relevant context",
            "availability": "available",
        }],
        services,
    )

    assert intention["route"] == "evidence"
    assert action_requests == []
    assert resolver_requests == [{
        "capability": "local_context_recall",
        "semantic_goal": "obtain grounded local context",
        "evidence_handles": ["e1"],
    }]
