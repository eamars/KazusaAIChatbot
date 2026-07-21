"""Focused semantic authorization gates for resolver proposals."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.resolver_authorization import (
    authorize_resolver_requests,
)


def _bid(*, evidence_handles: list[str] | None = None) -> dict[str, object]:
    """Build one admitted bid that may require local evidence."""

    return {
        "branch_id": "ordinary_response",
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        "intention": "answer the current user",
        "desired_outcome": "give a grounded response",
        "concrete_detail": "use relevant relationship context",
        "reason": "the current turn refers to prior interaction",
        "private_monologue": "I should ground the response.",
        "target_roles": [],
        "evidence_handles": evidence_handles or ["e1"],
        "expected_consequences": ["the response remains coherent"],
        "confidence": "high",
    }


def _evidence(
    handle: str,
    text: str,
    *,
    source_kind: str = "episode",
) -> dict[str, object]:
    """Build one prompt-safe cognition evidence row."""

    return {
        "evidence_handle": handle,
        "evidence_ref": {
            "source_kind": source_kind,
            "source_id": f"source-{handle}",
            "occurred_at": "2026-07-18T00:00:00Z",
            "semantic_summary": text,
        },
        "semantic_text": text,
        "visible_to": ["q:event_agency"],
    }


def _request(
    bid_handle: str,
    resolver_handle: str,
    goal: str,
) -> dict[str, str]:
    """Build one planner-proposed resolver request."""

    return {
        "bid_handle": bid_handle,
        "resolver_handle": resolver_handle,
        "semantic_goal": goal,
        "reason": "the admitted bid still needs this evidence",
    }


def _resolver(kind: str) -> dict[str, str]:
    """Build one registry-projected resolver affordance."""

    return {
        "capability": kind,
        "semantic_capability": f"retrieve evidence through {kind}",
        "availability": "available",
    }


@pytest.mark.asyncio
async def test_initial_missing_evidence_request_is_authorized() -> None:
    """A focused decision preserves a genuinely unresolved evidence need."""

    captured: dict[str, object] = {}

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            captured.update(json.loads(str(messages[-1].content)))
            return SimpleNamespace(content=json.dumps({
                "decisions": {"c1": True},
            }))

    request = _request("b1", "r1", "retrieve the prior agreement")
    result = await authorize_resolver_requests(
        resolver_requests=[request],
        bid_handles={"b1": _bid()},
        evidence=[_evidence("e1", "The user refers to an earlier agreement.")],
        resolver_handles={"r1": _resolver("local_context_recall")},
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
    )

    assert result == [request]
    candidate = captured["candidates"]["c1"]
    assert candidate["semantic_capability"] == (
        "retrieve evidence through local_context_recall"
    )
    assert candidate["current_evidence"][0]["handle"] == "e1"


@pytest.mark.asyncio
async def test_satisfied_rephrased_request_is_rejected() -> None:
    """Authorization rejects a renamed request after evidence satisfies it."""

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            payload = json.loads(str(messages[-1].content))
            evidence = payload["candidates"]["c1"]["current_evidence"]
            assert evidence[1]["source_kind"] == "resolver_observation"
            assert "agreement is confirmed" in evidence[1]["semantic_text"]
            return SimpleNamespace(content=json.dumps({
                "decisions": {"c1": False},
            }))

    result = await authorize_resolver_requests(
        resolver_requests=[_request(
            "b1",
            "r1",
            "confirm the same prior agreement using different wording",
        )],
        bid_handles={"b1": _bid(evidence_handles=["e1", "e2"])},
        evidence=[
            _evidence("e1", "The user refers to an earlier agreement."),
            _evidence(
                "e2",
                "The prior agreement is confirmed with its boundary.",
                source_kind="resolver_observation",
            ),
        ],
        resolver_handles={"r1": _resolver("local_context_recall")},
        resolver_context=(
            "resolver_status=active; local_context_recall succeeded with "
            "confirmed evidence"
        ),
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
    )

    assert result == []


@pytest.mark.asyncio
async def test_three_distinct_resolver_requests_are_preserved() -> None:
    """Authorization retains the registry-driven three-request capacity."""

    requests = [
        _request("b1", f"r{index}", f"retrieve distinct evidence {index}")
        for index in range(1, 4)
    ]

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

    result = await authorize_resolver_requests(
        resolver_requests=requests,
        bid_handles={"b1": _bid()},
        evidence=[_evidence("e1", "Three distinct evidence needs remain.")],
        resolver_handles={
            f"r{index}": _resolver(f"capability_{index}")
            for index in range(1, 4)
        },
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
    )

    assert result == requests


@pytest.mark.asyncio
async def test_empty_resolver_plan_adds_no_authorization_call() -> None:
    """Ordinary turns retain their existing model-call latency."""

    class _LLM:
        async def ainvoke(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("resolver authorization must not be called")

    result = await authorize_resolver_requests(
        resolver_requests=[],
        bid_handles={"b1": _bid()},
        evidence=[],
        resolver_handles={},
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
    )

    assert result == []
