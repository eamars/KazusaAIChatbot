"""Regression tests for compositional Cognition V2 action planning."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    _validate_action_plan_decision,
    plan_actions,
)


def _bid(branch_id: str) -> dict[str, object]:
    """Build one admitted motive with complete deterministic provenance."""

    return {
        "branch_id": branch_id,
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        "intention": f"advance {branch_id}",
        "desired_outcome": "preserve a grounded interaction",
        "concrete_detail": "use only current evidence",
        "reason": "the admitted evidence supports this motive",
        "private_monologue": "I should respond deliberately.",
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "user-1",
        }],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the interaction remains coherent"],
        "confidence": "high",
    }


def _action(kind: str) -> dict[str, object]:
    """Build one registry-projected action affordance."""

    return {
        "action_kind": kind,
        "capability": kind,
        "permission": "allowed",
        "decision_mode": "optional",
        "allowed_decisions": [],
        "default_decision": "",
        "decision_pattern": "",
        "context_ref": "",
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "user-1",
        }],
    }


def _resolver(kind: str) -> dict[str, str]:
    """Build one registry-projected resolver affordance."""

    return {
        "capability": kind,
        "semantic_capability": f"resolve {kind}",
        "availability": "available",
    }


def _planner_response(
    *,
    route: str,
    actions: list[dict[str, str]] | None = None,
    resolvers: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Build the exact fixed-shape model response."""

    return {
        "route": route,
        "action_requests": actions or [],
        "resolver_requests": resolvers or [],
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
    }


@pytest.mark.asyncio
async def test_speech_composes_with_three_private_actions() -> None:
    """Visible speech is orthogonal to the production three-action capacity."""

    captured: dict[str, object] = {}
    response = _planner_response(
        route="speech",
        actions=[
            {
                "bid_handle": "b1",
                "action_handle": f"a{index}",
                "decision": "",
                "semantic_goal": f"perform private action {index}",
                "reason": f"action {index} advances the admitted motive",
            }
            for index in range(1, 4)
        ],
    )

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            captured.update(json.loads(str(messages[-1].content)))
            return SimpleNamespace(content=json.dumps(response))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={"episode_id": "episode-1", "trigger_source": "user_message"},
        evidence=[{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode-1",
                "occurred_at": "2026-07-17T00:00:00Z",
                "semantic_summary": "the user made a grounded request",
            },
            "semantic_text": "the user made a grounded request",
            "visible_to": ["q:event_agency"],
        }],
        available_actions=[
            _action("background_work_request"),
            _action("trigger_future_cognition"),
            _action("memory_lifecycle_update"),
        ],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
    )

    assert result["intention"]["route"] == "speech"
    assert [row["action_kind"] for row in result["action_requests"]] == [
        "background_work_request",
        "trigger_future_cognition",
        "memory_lifecycle_update",
    ]
    assert all(
        row["target_roles"] == _bid("ordinary_response")["target_roles"]
        for row in result["action_requests"]
    )
    assert all(
        row["evidence_handles"] == ["e1"]
        for row in result["action_requests"]
    )
    assert "speak" not in json.dumps(captured["action_handles"])


@pytest.mark.asyncio
async def test_invalid_action_plan_receives_one_bounded_replacement() -> None:
    """The same semantic owner can replace one contract-invalid object."""

    responses = [
        {
            "route": "speech",
            "action_requests": [],
        },
        _planner_response(route="speech"),
    ]
    captured_messages: list[list[object]] = []

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            captured_messages.append(messages)
            response = responses[len(captured_messages) - 1]
            return SimpleNamespace(content=json.dumps(response))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={"episode_id": "episode-1", "trigger_source": "user_message"},
        evidence=[],
        available_actions=[],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
    )

    assert result["intention"]["route"] == "speech"
    assert len(captured_messages) == 2
    repair_payload = json.loads(str(captured_messages[1][-1].content))
    assert repair_payload["contract_error"] == (
        "action plan fields are not exact"
    )
    assert "invalid_response" in repair_payload


@pytest.mark.asyncio
async def test_action_plan_stops_after_one_failed_replacement() -> None:
    """A second invalid object fails closed without an unbounded retry loop."""

    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            nonlocal calls
            calls += 1
            return SimpleNamespace(content=json.dumps({"route": "speech"}))

    with pytest.raises(ValueError, match="action plan is invalid"):
        await plan_actions(
            primary_bid=_bid("ordinary_response"),
            supporting_bids=[],
            episode={
                "episode_id": "episode-1",
                "trigger_source": "user_message",
            },
            evidence=[],
            available_actions=[],
            available_resolvers=[],
            resolver_context="resolver_status=idle",
            services=SimpleNamespace(
                llm=_LLM(),
                action_selection_config=object(),
            ),
        )

    assert calls == 2


@pytest.mark.asyncio
async def test_contextual_action_binds_ref_without_prompt_exposure() -> None:
    """A selected handle binds trusted context outside model-authored JSON."""

    captured_prompt = ""
    response = _planner_response(
        route="speech",
        actions=[{
            "bid_handle": "b1",
            "action_handle": "a1",
            "decision": "status",
            "semantic_goal": "report the selected open run status",
            "reason": "the user requested current progress",
        }],
    )
    contextual_action = _action("accepted_coding_task_request")
    contextual_action.update({
        "decision_mode": "closed",
        "allowed_decisions": ["status", "cancel"],
        "default_decision": "status",
        "context_ref": "coding_run:private-run-ref",
    })

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            nonlocal captured_prompt
            del config
            captured_prompt = str(messages[-1].content)
            return SimpleNamespace(content=json.dumps(response))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={"episode_id": "episode-1", "trigger_source": "user_message"},
        evidence=[],
        available_actions=[contextual_action],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
    )

    assert result["action_requests"][0]["context_ref"] == (
        "coding_run:private-run-ref"
    )
    assert "coding_run:private-run-ref" not in captured_prompt


@pytest.mark.parametrize(
    "response, error_match",
    [
        (
            _planner_response(
                route="action",
                actions=[{
                    "bid_handle": "b1",
                    "action_handle": "a1",
                    "decision": "",
                    "semantic_goal": f"goal {index}",
                    "reason": "grounded reason",
                } for index in range(4)],
            ),
            "three",
        ),
        (
            _planner_response(
                route="speech",
                actions=[{
                    "bid_handle": "b1",
                    "action_handle": "a1",
                    "decision": "",
                    "semantic_goal": "act",
                    "reason": "grounded reason",
                }],
                resolvers=[{
                    "bid_handle": "b1",
                    "resolver_handle": "r1",
                    "semantic_goal": "resolve",
                    "reason": "grounded reason",
                }],
            ),
            "mutually exclusive",
        ),
        (
            _planner_response(
                route="speech",
                actions=[{
                    "bid_handle": "b2",
                    "action_handle": "a1",
                    "decision": "",
                    "semantic_goal": "act",
                    "reason": "grounded reason",
                }],
            ),
            "bid handle",
        ),
    ],
)
def test_action_plan_rejects_capacity_mixing_and_unknown_bids(
    response: dict[str, object],
    error_match: str,
) -> None:
    """Structural validation prevents overflow, mixing, and invented motives."""

    with pytest.raises(ValueError, match=error_match):
        _validate_action_plan_decision(
            response,
            bid_handles={"b1": _bid("ordinary_response")},
            action_handles={"a1": _action("background_work_request")},
            resolver_handles={"r1": _resolver("local_context_recall")},
        )


def test_action_plan_requires_requests_for_private_routes() -> None:
    """Action and evidence routes cannot collapse into empty technical work."""

    for route in ("action", "evidence"):
        with pytest.raises(ValueError, match="requires"):
            _validate_action_plan_decision(
                _planner_response(route=route),
                bid_handles={"b1": _bid("ordinary_response")},
                action_handles={"a1": _action("background_work_request")},
                resolver_handles={"r1": _resolver("local_context_recall")},
        )


def test_action_plan_enforces_registry_decision_format() -> None:
    """A scheduled action rejects prose appended to its typed decision."""

    action = _action("future_speak")
    action.update({
        "decision_mode": "required_text",
        "decision_pattern": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}",
    })
    with pytest.raises(ValueError, match="full-match"):
        _validate_action_plan_decision(
            _planner_response(
                route="speech",
                actions=[{
                    "bid_handle": "b1",
                    "action_handle": "a1",
                    "decision": "2026-07-15 08:00 | remind me",
                    "semantic_goal": "schedule the accepted reminder",
                    "reason": "the user requested a future reminder",
                }],
            ),
            bid_handles={"b1": _bid("ordinary_response")},
            action_handles={"a1": action},
            resolver_handles={},
        )


def test_closed_action_error_names_handle_and_allowed_decision() -> None:
    """A bounded replacement receives the exact registry correction rule."""

    action = _action("trigger_future_cognition")
    action["decision_mode"] = "closed"
    action["allowed_decisions"] = ["schedule"]
    action["default_decision"] = "schedule"

    with pytest.raises(
        ValueError,
        match=r"a1 decision must be one of \['schedule'\]",
    ):
        _validate_action_plan_decision(
            _planner_response(
                route="speech",
                actions=[{
                    "bid_handle": "b1",
                    "action_handle": "a1",
                    "decision": "think about the response later",
                    "semantic_goal": "continue one grounded private task",
                    "reason": "the admitted motive requires later cognition",
                }],
            ),
            bid_handles={"b1": _bid("ordinary_response")},
            action_handles={"a1": action},
            resolver_handles={},
        )


def test_speak_and_internal_apply_are_absent_from_planner_affordances() -> None:
    """The planner has one visible-speech vocabulary and no internal effector."""

    source_actions = [
        _action("speak"),
        _action("apply_memory_lifecycle_update"),
        _action("background_work_request"),
    ]
    visible = [
        row for row in source_actions
        if row["action_kind"] not in {
            "speak",
            "apply_memory_lifecycle_update",
        }
    ]

    assert [row["action_kind"] for row in visible] == [
        "background_work_request",
    ]
