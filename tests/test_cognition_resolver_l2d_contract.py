"""V2 semantic resolver-planning contract tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import plan_actions


class _ResolverPlannerLLM:
    def __init__(self) -> None:
        self.call_count = 0

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages, config
        self.call_count += 1
        if self.call_count == 2:
            return SimpleNamespace(content=json.dumps({
                "decisions": {"c1": True},
            }))
        return SimpleNamespace(content=json.dumps({
            "action_requests": [],
            "resolver_requests": [{
                "bid_handle": "b1",
                "resolver_handle": "r1",
                "semantic_goal": "resolve the bounded evidence task",
                "reason": "the admitted motive has an evidence gap",
                "start_in_background": False,
            }],
            "goal_resolution": "requires_required_evidence",
            "resolver_pending_resolution": None,
            "resolver_goal_progress": None,
        }))


@pytest.mark.asyncio
async def test_v2_planner_returns_typed_resolver_request() -> None:
    """The planner selects an available resolver by prompt-local handle."""

    primary_bid = {
        "branch_id": "epistemic_exploration",
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        "intention": "resolve the relevant knowledge gap",
        "desired_outcome": "resolve the bounded evidence task",
        "concrete_detail": "return only prompt-safe evidence",
        "reason": "the current evidence is incomplete",
        "private_monologue": "I need evidence before I decide.",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["reduce uncertainty"],
        "confidence": "high",
    }
    llm = _ResolverPlannerLLM()
    services = SimpleNamespace(
        llm=llm,
        action_planning_config=object(),
        action_authorization_config=object(),
        resolver_authorization_config=object(),
    )

    result = await plan_actions(
        primary_bid=primary_bid,
        supporting_bids=[],
        episode={
            "episode_id": "episode-1",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[],
        available_actions=[],
        available_resolvers=[{
            "capability": "task_resolution_request",
            "semantic_capability": "resolve a bounded task through specialists",
            "availability": "available",
        }],
        resolver_context="resolver_status=idle",
        services=services,
    )

    assert result["intention"]["route"] == "evidence"
    assert result["action_requests"] == []
    assert result["resolver_requests"] == [{
        "capability": "task_resolution_request",
        "semantic_goal": "resolve the bounded evidence task",
        "reason": "the admitted motive has an evidence gap",
        "evidence_handles": ["e1"],
        "start_in_background": False,
    }]
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_v2_planner_preserves_background_routing_boolean() -> None:
    """True survives planning, authorization, and typed materialization."""

    class _BackgroundPlannerLLM:
        def __init__(self) -> None:
            self.call_count = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            self.call_count += 1
            if self.call_count == 2:
                return SimpleNamespace(content=json.dumps({
                    "decisions": {"c1": True},
                }))
            return SimpleNamespace(content=json.dumps({
                "action_requests": [],
                "resolver_requests": [{
                    "bid_handle": "b1",
                    "resolver_handle": "r1",
                    "semantic_goal": "research the bounded public question",
                    "reason": "the work should start in the durable path",
                    "start_in_background": True,
                }],
                "goal_resolution": "requires_required_evidence",
                "resolver_pending_resolution": None,
                "resolver_goal_progress": None,
            }))

    primary_bid = {
        "branch_id": "epistemic_exploration",
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        "intention": "resolve the relevant knowledge gap",
        "desired_outcome": "resolve the bounded evidence task",
        "concrete_detail": "return only prompt-safe evidence",
        "reason": "the current evidence is incomplete",
        "private_monologue": "I need evidence before I decide.",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["reduce uncertainty"],
        "confidence": "high",
    }
    llm = _BackgroundPlannerLLM()
    services = SimpleNamespace(
        llm=llm,
        action_planning_config=object(),
        action_authorization_config=object(),
        resolver_authorization_config=object(),
    )

    result = await plan_actions(
        primary_bid=primary_bid,
        supporting_bids=[],
        episode={
            "episode_id": "episode-background",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[],
        available_actions=[],
        available_resolvers=[{
            "capability": "task_resolution_request",
            "semantic_capability": "resolve a bounded task through specialists",
            "availability": "available",
        }],
        resolver_context="resolver_status=idle",
        services=services,
    )

    assert result["resolver_requests"] == [{
        "capability": "task_resolution_request",
        "semantic_goal": "research the bounded public question",
        "reason": "the work should start in the durable path",
        "evidence_handles": ["e1"],
        "start_in_background": True,
    }]
