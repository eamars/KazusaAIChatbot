"""Persona V2 semantic action-planning and materialization tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import plan_actions
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
)


def _bid() -> dict[str, object]:
    """Build one complete admitted motive."""

    return {
        "branch_id": "ordinary_response",
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        "intention": "respond to the observed event",
        "desired_outcome": "preserve grounded continuity",
        "concrete_detail": "use only typed evidence",
        "reason": "the current evidence supports this motive",
        "private_monologue": "I should respond from what I observed.",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["maintain continuity"],
        "confidence": "high",
    }


def _decision(
    *,
    route: str,
    action_requests: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Build one exact fixed-shape planner decision."""

    return {
        "route": route,
        "action_requests": action_requests or [],
        "resolver_requests": [],
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
    }


class _ActionLLM:
    def __init__(self, decision: dict[str, object]) -> None:
        self._decision = decision

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages, config
        return SimpleNamespace(content=json.dumps(self._decision))


def _services(decision: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        llm=_ActionLLM(decision),
        action_selection_config=object(),
    )


async def _plan(
    decision: dict[str, object],
    actions: list[dict[str, object]],
) -> dict[str, object]:
    """Invoke the planner through its canonical bounded context."""

    return await plan_actions(
        primary_bid=_bid(),
        supporting_bids=[],
        episode={"episode_id": "episode-1", "trigger_source": "user_message"},
        evidence=[],
        available_actions=actions,
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=_services(decision),
    )


@pytest.mark.asyncio
async def test_action_route_copies_only_available_typed_affordance() -> None:
    """The model selects handles; code copies trusted targets and evidence."""

    result = await _plan(
        _decision(
            route="action",
            action_requests=[{
                "bid_handle": "b1",
                "action_handle": "a1",
                "decision": "",
                "semantic_goal": "complete the accepted bounded task",
                "reason": "the admitted motive requires delayed work",
            }],
        ),
        [{
            "action_kind": "background_work_request",
            "capability": "background_work_request",
            "permission": "allowed",
            "decision_mode": "optional",
            "allowed_decisions": [],
            "default_decision": "",
            "decision_pattern": "",
            "context_ref": "",
            "target_roles": [],
        }],
    )

    assert result["intention"]["route"] == "action"
    assert result["action_requests"] == [{
        "action_kind": "background_work_request",
        "decision": "",
        "semantic_goal": "complete the accepted bounded task",
        "reason": "the admitted motive requires delayed work",
        "context_ref": "",
        "target_roles": [],
        "evidence_handles": ["e1"],
    }]
    assert result["resolver_requests"] == []


@pytest.mark.asyncio
async def test_action_route_rejects_unavailable_affordance() -> None:
    """Permission and availability remain deterministic code authority."""

    with pytest.raises((CognitionExecutionError, ValueError)):
        await _plan(
            _decision(
                route="action",
                action_requests=[{
                    "bid_handle": "b1",
                    "action_handle": "a1",
                    "decision": "",
                    "semantic_goal": "complete bounded work",
                    "reason": "the admitted motive requires it",
                }],
            ),
            [],
        )


@pytest.mark.asyncio
async def test_generic_action_decision_preserves_coding_continuation() -> None:
    """The planner passes a registry-grounded continuation without rewriting."""

    result = await _plan(
        _decision(
            route="speech",
            action_requests=[{
                "bid_handle": "b1",
                "action_handle": "a1",
                "decision": "status",
                "semantic_goal": "report the active coding run status",
                "reason": "the user requested a progress update",
            }],
        ),
        [{
            "action_kind": "accepted_coding_task_request",
            "capability": "accepted_coding_task_request",
            "permission": "allowed",
            "decision_mode": "closed",
            "allowed_decisions": [
                "revise_proposal",
                "summarize",
                "status",
                "approve_and_verify",
                "respond_to_blocker",
                "cancel",
            ],
            "default_decision": "",
            "decision_pattern": "",
            "context_ref": "coding_run:run-1",
            "target_roles": [],
        }],
    )

    assert result["intention"]["route"] == "speech"
    assert result["action_requests"][0]["decision"] == "status"
    assert result["action_requests"][0]["context_ref"] == "coding_run:run-1"


@pytest.mark.asyncio
async def test_no_admitted_bid_returns_silence_without_llm() -> None:
    """A missing valid bid produces the frozen V2 silence result."""

    class _UnexpectedLLM:
        async def ainvoke(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("silence must not invoke the action planner")

    result = await plan_actions(
        primary_bid=None,
        supporting_bids=[],
        episode={"episode_id": "episode-1"},
        evidence=[],
        available_actions=[],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_UnexpectedLLM(),
            action_selection_config=object(),
        ),
    )

    assert result["intention"]["route"] == "silence"
    assert result["action_requests"] == []
    assert result["resolver_requests"] == []
