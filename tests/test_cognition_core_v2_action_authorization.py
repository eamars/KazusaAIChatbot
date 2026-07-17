"""Structural gates for V2 action authorization and route ownership."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    authorize_action_requests,
    derive_action_route,
)


def _bid() -> dict[str, object]:
    """Build one admitted visible-response motive."""

    return {
        "branch_id": "ordinary_response",
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        "intention": "respond to the current request",
        "desired_outcome": "give the user a grounded current response",
        "concrete_detail": "preserve the requested actors and effect",
        "reason": "the current evidence supports a response",
        "private_monologue": "I should respond deliberately.",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the user receives a current response"],
        "confidence": "high",
    }


def _evidence(text: str) -> dict[str, object]:
    """Build one current episode evidence row."""

    return {
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode-1",
            "occurred_at": "2026-07-17T00:00:00Z",
            "semantic_summary": text,
        },
        "semantic_text": text,
        "visible_to": ["q:event_agency"],
    }


def _action(
    bid_handle: str,
    action_handle: str,
    goal: str,
) -> dict[str, str]:
    """Build one validated planner-proposed action row."""

    return {
        "bid_handle": bid_handle,
        "action_handle": action_handle,
        "decision": "enqueue",
        "semantic_goal": goal,
        "reason": "the admitted bid proposed this durable effect",
    }


def _affordance(kind: str) -> dict[str, object]:
    """Build one registry-derived action affordance."""

    return {
        "action_kind": kind,
        "capability": (
            "Produce an explicitly accepted delayed text or repository result "
            "out of turn; this capability does not actuate a physical scene."
        ),
        "permission": "allowed",
        "decision_mode": "closed",
        "allowed_decisions": ["enqueue"],
        "default_decision": "enqueue",
        "decision_pattern": "",
        "context_ref": "",
        "target_roles": [],
    }


@pytest.mark.parametrize(
    "output_mode,has_bid,action_count,resolver_count,expected",
    [
        ("visible_reply", True, 0, 0, "speech"),
        ("visible_reply", True, 2, 0, "speech"),
        ("visible_reply", True, 0, 1, "evidence"),
        ("scheduled_action_request", True, 1, 0, "action"),
        ("think_only", True, 1, 0, "action"),
        ("preview", True, 0, 0, "silence"),
        ("silence", True, 0, 0, "silence"),
        ("visible_reply", False, 0, 0, "silence"),
    ],
)
def test_route_is_derived_from_protocol_and_validated_request_sets(
    output_mode: str,
    has_bid: bool,
    action_count: int,
    resolver_count: int,
    expected: str,
) -> None:
    """The model does not restate deterministic route shape."""

    actions = [_action("b1", "a1", "delayed work")] * action_count
    resolvers = [{"capability": "local_context_recall"}] * resolver_count

    result = derive_action_route(
        episode={"output_mode": output_mode},
        primary_bid=_bid() if has_bid else None,
        action_requests=actions,
        resolver_requests=resolvers,
    )

    assert result == expected


@pytest.mark.asyncio
async def test_action_authorization_rejects_capability_effect_mismatch() -> None:
    """A focused semantic decision blocks an ungrounded executable effect."""

    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            nonlocal calls
            calls += 1
            payload = json.loads(str(messages[-1].content))
            assert "action_handle" not in payload["candidates"]["c1"]
            assert payload["candidates"]["c1"]["current_evidence"] == [
                "Move your body into the requested position now."
            ]
            return SimpleNamespace(content=json.dumps({
                "decisions": {"c1": False},
            }))

    result = await authorize_action_requests(
        action_requests=[_action(
            "b1",
            "a1",
            "generate a physical-action description later",
        )],
        bid_handles={"b1": _bid()},
        evidence=[_evidence("Move your body into the requested position now.")],
        action_handles={
            "a1": _affordance("background_work_request"),
        },
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
    )

    assert result == []
    assert calls == 1


@pytest.mark.asyncio
async def test_action_authorization_preserves_three_grounded_actions() -> None:
    """Authorization preserves the production three-action composition floor."""

    action_requests = [
        _action("b1", f"a{index}", f"perform accepted effect {index}")
        for index in range(1, 4)
    ]
    action_handles = {
        f"a{index}": _affordance(f"capability_{index}")
        for index in range(1, 4)
    }

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            return SimpleNamespace(content=json.dumps({
                "decisions": {
                    f"c{index}": True for index in range(1, 4)
                },
            }))

    result = await authorize_action_requests(
        action_requests=action_requests,
        bid_handles={"b1": _bid()},
        evidence=[_evidence("Accept all three independent durable effects.")],
        action_handles=action_handles,
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
    )

    assert result == action_requests


@pytest.mark.asyncio
async def test_empty_action_plan_adds_no_authorization_call() -> None:
    """Ordinary speech turns retain their existing model-call latency."""

    class _LLM:
        async def ainvoke(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("authorization model must not be called")

    result = await authorize_action_requests(
        action_requests=[],
        bid_handles={"b1": _bid()},
        evidence=[_evidence("Hello")],
        action_handles={},
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
    )

    assert result == []
