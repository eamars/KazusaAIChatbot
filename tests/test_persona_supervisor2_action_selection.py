"""Persona V2 route-selection and materialization tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import select_route
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
)


def _bid(*, route: str, action_kind: str | None = None) -> dict[str, object]:
    bid: dict[str, object] = {
        "branch_id": "ordinary_response",
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        "intention": "respond to the observed event",
        "desired_outcome": "preserve grounded continuity",
        "concrete_detail": "use only typed evidence",
        "reason": "the current evidence supports this route",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["maintain continuity"],
        "confidence": "high",
        "requested_route": route,
    }
    if action_kind is not None:
        bid["requested_action_kind"] = action_kind
    return bid


class _RouteLLM:
    def __init__(self, decision: dict[str, str]) -> None:
        self._decision = decision

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages, config
        return SimpleNamespace(content=json.dumps(self._decision))


def _services(decision: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        llm=_RouteLLM(decision),
        action_selection_config=object(),
    )


@pytest.mark.asyncio
async def test_action_route_copies_only_available_typed_affordance() -> None:
    """The model selects a handle; code copies the trusted affordance."""

    intention, action_requests, resolver_requests = await select_route(
        _bid(route="action", action_kind="accepted_task_request"),
        [],
        [{
            "action_kind": "accepted_task_request",
            "capability": "accepted_task_request",
            "permission": "allowed",
            "target_roles": [],
        }],
        [],
        _services({
            "selected_bid_handle": "b1",
            "route": "action",
            "action_handle": "a1",
        }),
    )

    assert intention["route"] == "action"
    assert action_requests == [{
        "action_kind": "accepted_task_request",
        "semantic_goal": "preserve grounded continuity",
        "target_roles": [],
        "evidence_handles": ["e1"],
    }]
    assert resolver_requests == []


@pytest.mark.asyncio
async def test_action_route_rejects_unavailable_affordance() -> None:
    """Permission and availability remain deterministic code authority."""

    with pytest.raises(CognitionExecutionError):
        await select_route(
            _bid(route="action", action_kind="accepted_task_request"),
            [],
            [],
            [],
            _services({
                "selected_bid_handle": "b1",
                "route": "action",
                "action_handle": "a1",
            }),
        )


@pytest.mark.asyncio
async def test_no_admitted_bid_returns_silence_without_llm() -> None:
    """A missing valid bid produces the frozen V2 silence result."""

    class _UnexpectedLLM:
        async def ainvoke(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("silence must not invoke the route model")

    services = SimpleNamespace(
        llm=_UnexpectedLLM(),
        action_selection_config=object(),
    )

    intention, action_requests, resolver_requests = await select_route(
        None,
        [],
        [],
        [],
        services,
    )

    assert intention["route"] == "silence"
    assert action_requests == []
    assert resolver_requests == []
