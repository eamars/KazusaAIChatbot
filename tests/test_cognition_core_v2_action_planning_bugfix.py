"""Regression tests for compositional Cognition V2 action planning."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    invoke_semantic_authorizer,
)
from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    ACTION_PLANNING_PROMPT,
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


def _goal_progress() -> dict[str, object]:
    """Build the canonical resolver checklist owned by recurrence state."""

    return {
        "schema_version": "resolver_goal_progress.v1",
        "original_goal": "answer the user's breakfast question",
        "current_focus": "identify grounded breakfast evidence",
        "deliverables": [{
            "description": "give one grounded breakfast answer",
            "status": "pending",
            "note": "",
        }],
        "missing_user_inputs": [],
        "evidence_dependencies": ["character memory"],
        "attempted_paths": [],
        "source_backed_facts": [],
        "assumptions_or_inferences": [],
        "blockers": [],
        "final_response_requirements": ["answer the current user"],
    }


def _planner_response(
    *,
    actions: list[dict[str, str]] | None = None,
    resolvers: list[dict[str, str]] | None = None,
    goal_resolution: str | None = None,
) -> dict[str, object]:
    """Build the exact fixed-shape model response."""

    if goal_resolution is None:
        goal_resolution = (
            "requires_required_evidence"
            if resolvers
            else "answerable_now"
        )
    return {
        "action_requests": actions or [],
        "resolver_requests": resolvers or [],
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
        "goal_resolution": goal_resolution,
    }


@pytest.mark.asyncio
async def test_speech_composes_with_three_private_actions() -> None:
    """Visible speech is orthogonal to the production three-action capacity."""

    captured: dict[str, object] = {}
    response = _planner_response(
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

    authorization = {
        "decisions": {
            f"c{index}": True for index in range(1, 4)
        },
    }
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
            captured.update(json.loads(str(messages[-1].content)))
            calls += 1
            selected = response if calls == 1 else authorization
            return SimpleNamespace(content=json.dumps(selected))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-1",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
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
async def test_answerable_now_drops_optional_resolver_request() -> None:
    """A sufficient answer must not enter optional retrieval recurrence."""

    response = _planner_response(
        resolvers=[{
            "bid_handle": "b1",
            "resolver_handle": "r1",
            "semantic_goal": "retrieve an optional relationship example",
            "reason": "the model considered extra context despite a sufficient answer",
        }],
        goal_resolution="answerable_now",
    )

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            return SimpleNamespace(content=json.dumps(response))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-answerable-now",
            "trigger_source": "user_message",
        },
        evidence=[{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode-answerable-now",
                "occurred_at": "2026-07-17T00:00:00Z",
                "semantic_summary": "the user asked a general question",
            },
            "semantic_text": "the user asked a general question",
            "visible_to": ["q:relationship_social"],
        }],
        available_actions=[],
        available_resolvers=[_resolver("local_context_recall")],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
    )

    assert result["goal_resolution"] == "answerable_now"
    assert result["resolver_requests"] == []
    assert result["intention"]["route"] == "speech"


def test_action_planning_prompt_separates_missing_user_input() -> None:
    """The LLM contract distinguishes missing user input from evidence."""

    assert "resolver_context" in ACTION_PLANNING_PROMPT
    assert "requires_user_input" in ACTION_PLANNING_PROMPT
    assert "resolver_requests=[]" in ACTION_PLANNING_PROMPT
    assert "requires_required_evidence" in ACTION_PLANNING_PROMPT


@pytest.mark.asyncio
async def test_invalid_action_plan_receives_one_bounded_replacement() -> None:
    """The same semantic owner can replace one contract-invalid object."""

    responses = [
        {"action_requests": "invalid"},
        _planner_response(),
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
        episode={
            "episode_id": "episode-1",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
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

    assert result["intention"]["route"] == "speech"
    assert len(captured_messages) == 2
    repair_payload = json.loads(str(captured_messages[1][-1].content))
    assert repair_payload["contract_error"] == (
        "action requests must be an array"
    )
    assert "invalid_response" in repair_payload


@pytest.mark.asyncio
async def test_action_plan_contains_one_failed_replacement() -> None:
    """A second invalid object yields speech without an unbounded retry loop."""

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
            return SimpleNamespace(content=json.dumps({
                "action_requests": "invalid",
            }))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-1",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
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
    assert result["intention"]["route"] == "speech"
    assert result["action_requests"] == []
    assert result["resolver_requests"] == []


@pytest.mark.asyncio
async def test_contextual_action_binds_ref_without_prompt_exposure() -> None:
    """A selected handle binds trusted context outside model-authored JSON."""

    captured_prompt = ""
    response = _planner_response(
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

    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            nonlocal captured_prompt
            nonlocal calls
            del config
            captured_prompt = str(messages[-1].content)
            calls += 1
            if calls == 1:
                return SimpleNamespace(content=json.dumps(response))
            return SimpleNamespace(content=json.dumps({
                "decisions": {"c1": True},
            }))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-1",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
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


def test_action_plan_caps_rows_and_drops_unknown_bids() -> None:
    """Normalization preserves valid capacity and drops invented provenance."""

    response = _planner_response(actions=[{
        "bid_handle": "b1" if index < 4 else "b2",
        "action_handle": "a1",
        "decision": "",
        "semantic_goal": f"goal {index}",
        "reason": "grounded reason",
    } for index in range(1, 5)])

    decision = _validate_action_plan_decision(
        response,
        bid_handles={"b1": _bid("ordinary_response")},
        action_handles={"a1": _action("background_work_request")},
        resolver_handles={"r1": _resolver("local_context_recall")},
    )

    assert len(decision["action_requests"]) == 3


def test_action_plan_rejects_mixed_action_and_resolver_semantics() -> None:
    """Normalization still rejects a semantically ambiguous mixed route."""

    response = _planner_response(
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
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_action_plan_decision(
            response,
            bid_handles={"b1": _bid("ordinary_response")},
            action_handles={"a1": _action("background_work_request")},
            resolver_handles={"r1": _resolver("local_context_recall")},
        )


def test_action_plan_ignores_model_authored_route() -> None:
    """Protocol route remains derived after unknown fields are stripped."""

    response = _planner_response()
    response["route"] = "speech"
    decision = _validate_action_plan_decision(
        response,
        bid_handles={"b1": _bid("ordinary_response")},
        action_handles={"a1": _action("background_work_request")},
        resolver_handles={"r1": _resolver("local_context_recall")},
    )

    assert "route" not in decision


def test_action_plan_merges_semantic_goal_progress_delta() -> None:
    """Protocol code preserves the canonical resolver checklist shape."""

    response = _planner_response(resolvers=[{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "retrieve grounded breakfast evidence",
        "reason": "the answer depends on character memory",
    }])
    response["resolver_goal_progress"] = {
        "original_goal": "answer the user's breakfast question",
        "current_focus": "retrieve the relevant character memory",
    }

    decision = _validate_action_plan_decision(
        response,
        bid_handles={"b1": _bid("ordinary_response")},
        action_handles={},
        resolver_handles={"r1": _resolver("local_context_recall")},
        current_goal_progress=_goal_progress(),
    )

    progress = decision["resolver_goal_progress"]
    assert progress["current_focus"] == (
        "retrieve the relevant character memory"
    )
    assert progress["deliverables"] == _goal_progress()["deliverables"]
    assert progress["evidence_dependencies"] == ["character memory"]


def test_action_plan_drops_invalid_registry_decision_format() -> None:
    """A scheduled action drops prose appended to its typed decision."""

    action = _action("future_speak")
    action.update({
        "decision_mode": "required_text",
        "decision_pattern": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}",
    })
    decision = _validate_action_plan_decision(
        _planner_response(
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

    assert decision["action_requests"] == []


def test_closed_action_with_unknown_decision_is_dropped() -> None:
    """A closed action cannot escape its registry decision vocabulary."""

    action = _action("trigger_future_cognition")
    action["decision_mode"] = "closed"
    action["allowed_decisions"] = ["schedule"]
    action["default_decision"] = "schedule"

    decision = _validate_action_plan_decision(
        _planner_response(
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

    assert decision["action_requests"] == []


def test_action_plan_strips_extra_resolver_fields() -> None:
    """Harmless model metadata cannot block a grounded resolver proposal."""

    response = _planner_response(resolvers=[{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "recover the omitted local referent",
        "reason": "the current phrase is incomplete",
        "capability": "local_context_recall",
        "priority": "now",
    }])

    decision = _validate_action_plan_decision(
        response,
        bid_handles={"b1": _bid("ordinary_response")},
        action_handles={},
        resolver_handles={"r1": _resolver("local_context_recall")},
    )

    assert decision["resolver_requests"] == [{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "recover the omitted local referent",
        "reason": "the current phrase is incomplete",
    }]


@pytest.mark.asyncio
async def test_authorizer_denies_after_one_unusable_replacement() -> None:
    """Schema failure cannot authorize work or crash the visible response."""

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
            return SimpleNamespace(content=json.dumps({"invalid": True}))

    decisions = await invoke_semantic_authorizer(
        services=SimpleNamespace(
            llm=_LLM(),
            action_selection_config=object(),
        ),
        messages=[HumanMessage(content="bounded candidates")],
        candidate_handles=["c1", "c2"],
        stage_name="test_authorization",
        output_state_fields=["authorized_requests"],
    )

    assert calls == 2
    assert decisions == {"c1": False, "c2": False}


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
