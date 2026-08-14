"""Regression tests for compositional Cognition V2 action planning."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref
from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    ACTION_AUTHORIZATION_PROMPT_CAP,
    invoke_semantic_authorizer,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    GOAL_COGNITION_PROMPT,
)
from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    ACTION_PLANNING_ATTEMPT_LIMIT,
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
    """Visible speech retains three admitted private actions."""

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
            "authority": "current_event",
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
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
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
            "start_in_background": False,
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
            "authority": "current_event",
        }],
        available_actions=[],
        available_resolvers=[_resolver("task_resolution_request")],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
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
    assert "scheduled_tick" in ACTION_PLANNING_PROMPT
    assert "定时输出合同不允许 resolver_requests" in ACTION_PLANNING_PROMPT
    assert "绑定既有 coding_run_ref" in ACTION_PLANNING_PROMPT
    assert "queue-only" in ACTION_PLANNING_PROMPT


def test_action_planning_prompt_binds_goal_progress_shape() -> None:
    """The planner must not invent a partial invalid goal-progress object."""

    assert "resolver_goal_progress 为 null" in ACTION_PLANNING_PROMPT
    assert "deliverables" in ACTION_PLANNING_PROMPT
    assert "description" in ACTION_PLANNING_PROMPT
    assert "status" in ACTION_PLANNING_PROMPT
    assert "note" in ACTION_PLANNING_PROMPT
    assert "evidence_dependencies" in ACTION_PLANNING_PROMPT
    assert "assumptions_or_inferences" in ACTION_PLANNING_PROMPT
    assert "字符串数组" in ACTION_PLANNING_PROMPT
    assert "所有新生成的语义内容使用简体中文" in ACTION_PLANNING_PROMPT


def test_goal_cognition_prompt_keeps_runtime_feasibility_downstream() -> None:
    """Goal cognition preserves evidence needs without judging runtime tools."""

    assert "runtime_capability_limits" not in GOAL_COGNITION_PROMPT
    assert "不判断工具、worker、调度或运行时能力" in GOAL_COGNITION_PROMPT
    assert "取得所需证据后回应" in GOAL_COGNITION_PROMPT


def test_action_planning_repair_message_repeats_nested_contract() -> None:
    """A replacement prompt must expose the nested fields that failed."""

    from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
        _action_planning_repair_message,
    )

    payload = json.loads(
        _action_planning_repair_message(
            response_text='{"resolver_goal_progress": {}}',
            contract_error="description: expected non-empty string",
        ).content,
    )

    assert payload["contract_requirements"]["resolver_goal_progress"]
    assert payload["contract_requirements"]["task_resolution_route"]
    assert "start_in_background" in payload["contract_requirements"][
        "task_resolution_route"
    ]
    assert payload["contract_requirements"]["deliverable_fields"] == [
        "description",
        "status",
        "note",
    ]
    assert payload["contract_requirements"]["scalar_list_fields"] == [
        "missing_user_inputs",
        "evidence_dependencies",
        "attempted_paths",
        "source_backed_facts",
        "assumptions_or_inferences",
        "blockers",
        "final_response_requirements",
    ]


@pytest.mark.asyncio
async def test_action_planner_receives_runtime_owner_limits() -> None:
    """The action owner must see trusted unavailable capability boundaries."""

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
            return SimpleNamespace(content=json.dumps(_planner_response()))

    await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-runtime-limit",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[],
        available_actions=[],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        runtime_capability_limits=[
            "当前调度能力不可用，不能把未来提醒说成已经安排。",
        ],
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert captured["runtime_capability_limits"] == [
        "当前调度能力不可用，不能把未来提醒说成已经安排。",
    ]


@pytest.mark.asyncio
async def test_scheduled_planning_prompt_carries_typed_output_mode() -> None:
    """A scheduled planner receives the typed source and output mode."""

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
            return SimpleNamespace(content=json.dumps(_planner_response()))

    await plan_actions(
        primary_bid=_bid("scheduled_reminder"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-scheduled",
            "trigger_source": "scheduled_tick",
            "output_mode": "scheduled_action_request",
        },
        evidence=[],
        available_actions=[],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert captured["episode"] == {
        "trigger_source": "scheduled_tick",
        "output_mode": "scheduled_action_request",
    }


@pytest.mark.asyncio
async def test_scheduled_tick_route_remains_scheduled_action_request() -> None:
    """Scheduled planning keeps its typed output mode and speech route."""

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            return SimpleNamespace(content=json.dumps(_planner_response()))

    result = await plan_actions(
        primary_bid=_bid("scheduled_reminder"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-scheduled-route",
            "trigger_source": "scheduled_tick",
            "output_mode": "scheduled_action_request",
        },
        evidence=[],
        available_actions=[],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert result["intention"]["route"] == "speech"


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
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
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
async def test_action_plan_exhaustion_returns_empty_control_output() -> None:
    """Three invalid objects yield speech without authorizing work."""

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
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert calls == 3
    assert result["intention"]["route"] == "speech"
    assert result["action_requests"] == []
    assert result["resolver_requests"] == []


@pytest.mark.asyncio
async def test_action_plan_recovers_on_third_attempt() -> None:
    """The final bounded planner attempt can restore a valid empty plan."""

    responses = [
        {"action_requests": "invalid"},
        {"action_requests": "invalid"},
        _planner_response(),
    ]
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
            response = responses[calls]
            calls += 1
            return SimpleNamespace(content=json.dumps(response))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-third-attempt",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[],
        available_actions=[],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert calls == 3
    assert result["intention"]["route"] == "speech"
    assert result["action_requests"] == []
    assert result["resolver_requests"] == []


@pytest.mark.asyncio
async def test_denied_required_action_closes_goal_without_progress() -> None:
    """An unavailable owner cannot leave required evidence progress behind."""

    planner_response = _planner_response(
        actions=[{
            "bid_handle": "b1",
            "action_handle": "a1",
            "decision": "",
            "semantic_goal": "读取指定仓库并返回代码分析",
            "reason": "当前用户明确要求读取指定仓库",
        }],
        goal_resolution="requires_required_evidence",
    )
    responses = [
        planner_response,
        {"decisions": {"c1": False}},
    ]

    class _LLM:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            response = responses[self.calls]
            self.calls += 1
            return SimpleNamespace(content=json.dumps(response))

    llm = _LLM()
    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-denied-required-action",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode-denied-required-action",
                "occurred_at": "2026-07-17T00:00:00Z",
                "semantic_summary": "当前用户要求读取指定仓库",
            },
            "semantic_text": "当前用户要求读取指定仓库",
            "visible_to": ["q:event_agency"],
            "authority": "current_event",
        }],
        available_actions=[_action("accepted_coding_task_request")],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        runtime_capability_limits=[
            "当前后台任务能力不可用，不能把延迟任务说成已经创建、安排或完成。",
        ],
        services=SimpleNamespace(
            llm=llm,
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert llm.calls == 2
    assert result["goal_resolution"] == "blocked"
    assert result["action_requests"] == []
    assert result["resolver_requests"] == []
    assert result["resolver_goal_progress"] is None


@pytest.mark.asyncio
async def test_denied_required_resolver_closes_goal_without_progress() -> None:
    """An unavailable resolver owner cannot leave required progress behind."""

    planner_response = _planner_response(
        resolvers=[{
            "bid_handle": "b1",
            "resolver_handle": "r1",
            "semantic_goal": "分析指定仓库的源代码结构",
            "reason": "当前用户要求读取指定仓库",
            "start_in_background": False,
        }],
        goal_resolution="requires_required_evidence",
    )
    responses = [
        planner_response,
        {"decisions": {"c1": False}},
    ]

    class _LLM:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            response = responses[self.calls]
            self.calls += 1
            return SimpleNamespace(content=json.dumps(response))

    llm = _LLM()
    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-denied-required-resolver",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode-denied-required-resolver",
                "occurred_at": "2026-07-17T00:00:00Z",
                "semantic_summary": "当前用户要求读取指定仓库",
            },
            "semantic_text": "当前用户要求读取指定仓库",
            "visible_to": ["q:event_agency"],
            "authority": "current_event",
        }],
        available_actions=[],
        available_resolvers=[_resolver("task_resolution_request")],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=llm,
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert llm.calls == 2
    assert result["goal_resolution"] == "blocked"
    assert result["action_requests"] == []
    assert result["resolver_requests"] == []
    assert result["resolver_goal_progress"] is None


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
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
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
        resolver_handles={"r1": _resolver("task_resolution_request")},
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
            resolver_handles={"r1": _resolver("task_resolution_request")},
        )


def test_action_plan_ignores_model_authored_route() -> None:
    """Protocol route remains derived after unknown fields are stripped."""

    response = _planner_response()
    response["route"] = "speech"
    decision = _validate_action_plan_decision(
        response,
        bid_handles={"b1": _bid("ordinary_response")},
        action_handles={"a1": _action("background_work_request")},
        resolver_handles={"r1": _resolver("task_resolution_request")},
    )

    assert "route" not in decision


def test_action_plan_merges_semantic_goal_progress_delta() -> None:
    """Protocol code preserves the canonical resolver checklist shape."""

    response = _planner_response(resolvers=[{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "retrieve grounded breakfast evidence",
        "reason": "the answer depends on character memory",
        "start_in_background": False,
    }])
    response["resolver_goal_progress"] = {
        "current_focus": "retrieve the relevant character memory",
    }

    decision = _validate_action_plan_decision(
        response,
        bid_handles={"b1": _bid("ordinary_response")},
        action_handles={},
        resolver_handles={"r1": _resolver("task_resolution_request")},
        current_goal_progress=_goal_progress(),
    )

    progress = decision["resolver_goal_progress"]
    assert progress["current_focus"] == (
        "retrieve the relevant character memory"
    )
    assert progress["deliverables"] == _goal_progress()["deliverables"]
    assert progress["evidence_dependencies"] == ["character memory"]


def test_empty_goal_progress_shell_rejects_new_checklist() -> None:
    """An empty recurrence shell cannot receive invented planner progress."""

    empty_progress = {
        "schema_version": "resolver_goal_progress.v1",
        "original_goal": "answer the user's breakfast question",
        "current_focus": "",
        "deliverables": [],
        "missing_user_inputs": [],
        "evidence_dependencies": [],
        "attempted_paths": [],
        "source_backed_facts": [],
        "assumptions_or_inferences": [],
        "blockers": [],
        "final_response_requirements": [],
    }
    response = _planner_response(resolvers=[{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "retrieve grounded breakfast evidence",
        "reason": "the answer depends on character memory",
        "start_in_background": False,
    }])
    response["resolver_goal_progress"] = _goal_progress()

    with pytest.raises(
        ValueError,
        match="cannot update an empty shell",
    ):
        _validate_action_plan_decision(
            response,
            bid_handles={"b1": _bid("ordinary_response")},
            action_handles={},
            resolver_handles={"r1": _resolver("task_resolution_request")},
            current_goal_progress=empty_progress,
        )


def test_action_plan_rejects_invalid_registry_decision_format() -> None:
    """All-invalid action rows fail closed with one typed aggregate error."""

    action = _action("future_speak")
    action.update({
        "decision_mode": "required_text",
        "decision_pattern": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}",
    })
    with pytest.raises(
        ValueError,
        match="every proposed action request row was unusable",
    ):
        _validate_action_plan_decision(
            _planner_response(
                actions=[{
                    "bid_handle": "b1",
                    "action_handle": "a1",
                    "decision": "2026-07-15 08:00 | remind me",
                    "semantic_goal": "schedule the accepted reminder",
                    "reason": "the user requested a future reminder",
                    "scheduled_authority_proposal": {
                        "schema_version": (
                            "scheduled_authority_proposal.v1"
                        ),
                        "temporal_alignment": "aligned",
                        "authorized_content_summary": (
                            "在约定时间开始补偿考核。"
                        ),
                        "authorized_detail_refs": [
                            {
                                "evidence_handle": "e1",
                                "semantic_summary": (
                                    "当前对话明确约定在该时间开始补偿考核。"
                                ),
                                "provenance_role": "current_event",
                            }
                        ],
                    },
                }],
            ),
            bid_handles={"b1": _bid("ordinary_response")},
            action_handles={"a1": action},
            resolver_handles={},
        )


def test_closed_action_with_unknown_decision_invalidates_candidate() -> None:
    """An unusable closed-action row cannot produce an action request."""

    action = _action("trigger_future_cognition")
    action["decision_mode"] = "closed"
    action["allowed_decisions"] = ["schedule"]
    action["default_decision"] = "schedule"

    with pytest.raises(
        ValueError,
        match="every proposed action request row was unusable",
    ):
        _validate_action_plan_decision(
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


def test_invalid_resolver_row_invalidates_valid_sibling() -> None:
    """A malformed resolver sibling is dropped while the valid row survives."""

    response = _planner_response(resolvers=[
        {
            "bid_handle": "b1",
            "resolver_handle": "r1",
            "semantic_goal": "recover grounded context",
            "reason": "the answer depends on missing context",
            "start_in_background": False,
        },
        {
            "bid_handle": "b1",
            "resolver_handle": "r1",
            "semantic_goal": "recover grounded context",
        },
    ])

    decision = _validate_action_plan_decision(
        response,
        bid_handles={"b1": _bid("ordinary_response")},
        action_handles={},
        resolver_handles={"r1": _resolver("task_resolution_request")},
    )

    assert decision["resolver_requests"] == [{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "recover grounded context",
        "reason": "the answer depends on missing context",
        "start_in_background": False,
    }]


def test_action_plan_strips_extra_resolver_fields() -> None:
    """Harmless model metadata cannot block a grounded resolver proposal."""

    response = _planner_response(resolvers=[{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "recover the omitted local referent",
        "reason": "the current phrase is incomplete",
        "capability": "task_resolution_request",
        "priority": "now",
    }])

    decision = _validate_action_plan_decision(
        response,
        bid_handles={"b1": _bid("ordinary_response")},
        action_handles={},
        resolver_handles={"r1": _resolver("human_clarification")},
    )

    assert decision["resolver_requests"] == [{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "recover the omitted local referent",
        "reason": "the current phrase is incomplete",
    }]


def test_task_resolution_row_requires_exact_routing_boolean() -> None:
    """The generic task-resolution row needs exactly one JSON boolean."""

    for start_in_background in ("true", 1, 0, None, "yes"):
        response = _planner_response(resolvers=[{
            "bid_handle": "b1",
            "resolver_handle": "r1",
            "semantic_goal": "recover the omitted local referent",
            "reason": "the current phrase is incomplete",
            "start_in_background": start_in_background,
        }])
        with pytest.raises(
            ValueError,
            match="every proposed resolver request row was unusable",
        ):
            _validate_action_plan_decision(
                response,
                bid_handles={"b1": _bid("ordinary_response")},
                action_handles={},
                resolver_handles={"r1": _resolver(
                    "task_resolution_request",
                )},
            )


def test_inline_only_runtime_rejects_background_task_resolution() -> None:
    """Trusted inline-only mode rejects a background route proposal."""

    response = _planner_response(resolvers=[{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "recover the omitted local referent",
        "reason": "the current phrase is incomplete",
        "start_in_background": True,
    }])
    with pytest.raises(
        ValueError,
        match="every proposed resolver request row was unusable",
    ):
        _validate_action_plan_decision(
            response,
            bid_handles={"b1": _bid("ordinary_response")},
            action_handles={},
            resolver_handles={"r1": _resolver(
                "task_resolution_request",
            )},
            runtime_capability_limits=[
                "当前通用任务解析只有 inline 能力；task_resolution_request 必须先在本轮"
                "预算内尝试，不能写成后台已经安排。",
            ],
        )

    inline_response = _planner_response(resolvers=[{
        **response["resolver_requests"][0],
        "start_in_background": False,
    }])
    decision = _validate_action_plan_decision(
        inline_response,
        bid_handles={"b1": _bid("ordinary_response")},
        action_handles={},
        resolver_handles={"r1": _resolver("task_resolution_request")},
        runtime_capability_limits=[
            "当前通用任务解析只有 inline 能力；task_resolution_request 必须先在本轮"
            "预算内尝试，不能写成后台已经安排。",
        ],
    )
    assert decision["resolver_requests"][0]["start_in_background"] is False


def test_task_resolution_row_rejects_missing_or_extra_route_fields() -> None:
    """Missing and extra route fields fail closed for the generic row."""

    missing_boolean = _planner_response(resolvers=[{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "recover the omitted local referent",
        "reason": "the current phrase is incomplete",
    }])
    with pytest.raises(
        ValueError,
        match="every proposed resolver request row was unusable",
    ):
        _validate_action_plan_decision(
            missing_boolean,
            bid_handles={"b1": _bid("ordinary_response")},
            action_handles={},
            resolver_handles={"r1": _resolver("task_resolution_request")},
        )

    extra_route = _planner_response(resolvers=[{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "recover the omitted local referent",
        "reason": "the current phrase is incomplete",
        "start_in_background": True,
        "priority": "background",
    }])
    with pytest.raises(
        ValueError,
        match="every proposed resolver request row was unusable",
    ):
        _validate_action_plan_decision(
            extra_route,
            bid_handles={"b1": _bid("ordinary_response")},
            action_handles={},
            resolver_handles={"r1": _resolver("task_resolution_request")},
        )


def test_non_task_resolver_rejects_task_resolution_route_field() -> None:
    """The route boolean cannot be smuggled into another capability row."""

    response = _planner_response(resolvers=[{
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "recover the omitted local referent",
        "reason": "the current phrase is incomplete",
        "start_in_background": True,
    }])

    with pytest.raises(
        ValueError,
        match="every proposed resolver request row was unusable",
    ):
        _validate_action_plan_decision(
            response,
            bid_handles={"b1": _bid("ordinary_response")},
            action_handles={},
            resolver_handles={"r1": _resolver("human_clarification")},
        )


@pytest.mark.asyncio
async def test_authorizer_denies_after_bounded_exhaustion() -> None:
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
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
        config=object(),
        messages=[HumanMessage(content="bounded candidates")],
        candidate_handles=["c1", "c2"],
        stage_name="test_authorization",
        output_state_fields=["authorized_requests"],
        prompt_cap=ACTION_AUTHORIZATION_PROMPT_CAP,
    )

    assert calls == 3
    assert decisions == {"c1": False, "c2": False}


@pytest.mark.asyncio
async def test_authorizer_recovers_on_third_attempt() -> None:
    """Two invalid decisions can recover without bypassing authorization."""

    responses = [
        {"invalid": True},
        {"invalid": True},
        {"decisions": {"c1": True, "c2": False}},
    ]
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
            response = responses[calls]
            calls += 1
            return SimpleNamespace(content=json.dumps(response))

    decisions = await invoke_semantic_authorizer(
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
        config=object(),
        messages=[HumanMessage(content="bounded candidates")],
        candidate_handles=["c1", "c2"],
        stage_name="test_authorization",
        output_state_fields=["authorized_requests"],
        prompt_cap=ACTION_AUTHORIZATION_PROMPT_CAP,
    )

    assert calls == 3
    assert decisions == {"c1": True, "c2": False}


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


def _evidence(
    handle: str,
    source_kind: str,
    semantic_text: str,
    *,
    memory_scope: str | None = None,
) -> dict[str, object]:
    """Build one prompt-safe evidence row for a deterministic planner test."""

    authority_by_source_kind = {
        "episode": "current_event",
        "scheduler_event": "current_event",
        "tool_result": "current_event",
        "conversation_evidence": "participant_continuity",
        "promoted_reflection": "character_world_context",
        "recall_evidence": "contextual_fact_only",
        "resolver_observation": "contextual_fact_only",
    }
    if source_kind == "promoted_memory":
        if memory_scope == "current_user_continuity":
            authority = "participant_continuity"
        elif memory_scope == "shared_character_or_world":
            authority = "character_world_context"
        else:
            raise ValueError(
                "promoted memory fixtures require a canonical memory scope"
            )
    else:
        authority = authority_by_source_kind[source_kind]
    row: dict[str, object] = {
        "evidence_handle": handle,
        "evidence_ref": {
            "source_kind": source_kind,
            "source_id": f"raw:{handle}",
            "occurred_at": "2026-08-07T00:00:00Z",
            "semantic_summary": semantic_text,
        },
        "semantic_text": semantic_text,
        "visible_to": ["q:event_agency"],
        "authority": authority,
    }
    if memory_scope is not None:
        row["memory_scope"] = memory_scope
    return row


def test_resolver_semantic_goal_passes_through_without_rewrite() -> None:
    """Normalization preserves the model-authored objective verbatim."""

    goal = (
        "抓取 @Nagasaki-soyo-清尘 最近 10 天的聊天记录并返回给当前用户"
    )
    decision = _validate_action_plan_decision(
        _planner_response(resolvers=[{
            "bid_handle": "b1",
            "resolver_handle": "r1",
            "semantic_goal": goal,
            "reason": "当前答案需要该聊天记录作为证据",
            "start_in_background": False,
        }]),
        bid_handles={"b1": _bid("ordinary_response")},
        action_handles={},
        resolver_handles={"r1": _resolver("task_resolution_request")},
    )

    assert decision["resolver_requests"][0]["semantic_goal"] == goal


def test_explicit_audit_goal_is_not_rewritten_by_deterministic_code() -> None:
    """An explicit capability question keeps its model-owned audit meaning."""

    goal = (
        "核实当前角色是否具备抓取特定用户（@Nagasaki-soyo-清尘）最近 10 天"
        "聊天记录的技术能力及权限"
    )
    decision = _validate_action_plan_decision(
        _planner_response(resolvers=[{
            "bid_handle": "b1",
            "resolver_handle": "r1",
            "semantic_goal": goal,
            "reason": "当前用户明确询问能力与权限",
            "start_in_background": True,
        }]),
        bid_handles={"b1": _bid("ordinary_response")},
        action_handles={},
        resolver_handles={"r1": _resolver("task_resolution_request")},
    )

    assert decision["resolver_requests"][0]["semantic_goal"] == goal


@pytest.mark.asyncio
async def test_task_resolution_boolean_survives_authorization_and_materialization(
) -> None:
    """The validated routing boolean passes authorization unchanged."""

    captured_authorization_candidates: dict[str, object] = {}
    responses = [
        _planner_response(resolvers=[{
            "bid_handle": "b1",
            "resolver_handle": "r1",
            "semantic_goal": "resolve the bounded evidence task",
            "reason": "the admitted motive has an evidence gap",
            "start_in_background": True,
        }]),
        {"decisions": {"c1": True}},
    ]
    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            nonlocal calls
            del config
            if calls == 1:
                captured_authorization_candidates.update(
                    json.loads(str(messages[-1].content)),
                )
            response = responses[calls]
            calls += 1
            return SimpleNamespace(content=json.dumps(response))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-background-boolean",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode-background-boolean",
                "occurred_at": "2026-08-07T00:00:00Z",
                "semantic_summary": "the user asked for bounded research",
            },
            "semantic_text": "the user asked for bounded research",
            "visible_to": ["q:event_agency"],
            "authority": "current_event",
        }],
        available_actions=[],
        available_resolvers=[_resolver("task_resolution_request")],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    expected_continuation_ref = build_goal_continuation_ref(
        source_episode_id="episode-background-boolean",
        source_message_id="",
        branch_id="ordinary_response",
        goal_ref=_bid("ordinary_response")["goal_ref"],
    )
    assert result["resolver_requests"] == [{
        "capability": "task_resolution_request",
        "semantic_goal": "resolve the bounded evidence task",
        "reason": "the admitted motive has an evidence gap",
        "evidence_handles": ["e1"],
        "goal_continuation_ref": expected_continuation_ref,
        "start_in_background": True,
    }]
    authorization_candidates = captured_authorization_candidates["candidates"]
    assert "start_in_background" not in authorization_candidates["c1"]


@pytest.mark.asyncio
async def test_action_planning_evidence_projection_is_authority_labeled() -> None:
    """Every payload evidence row carries a deterministic provenance role."""

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
            return SimpleNamespace(content=json.dumps(_planner_response()))

    await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-provenance",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[
            _evidence("e1", "episode", "the current request"),
            _evidence("e2", "conversation_evidence", "a historical row"),
            _evidence("e3", "promoted_reflection", "a reflection row"),
            _evidence(
                "e4",
                "promoted_memory",
                "a memory row",
                memory_scope="current_user_continuity",
            ),
        ],
        available_actions=[],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    projected = captured["evidence"]
    assert [row["provenance_role"] for row in projected] == [
        "current_episode",
        "contextual_fact_only",
        "character_or_world_context_only",
        "current_user_history_only",
    ]
    serialized = json.dumps(projected, ensure_ascii=False)
    assert "source_id" not in serialized
    assert "occurred_at" not in serialized


def _future_speak_affordance() -> dict[str, object]:
    """Build the registry-projected future-speak affordance."""

    affordance = _action("future_speak")
    affordance.update({
        "decision_mode": "required_text",
        "decision_pattern": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}",
    })
    return affordance


def _future_speak_row(
    *,
    decision: str = "2026-05-16 10:00",
    temporal_alignment: str = "aligned",
) -> dict[str, object]:
    """Build one future-speak action row with a closed authority proposal."""

    return {
        "bid_handle": "b1",
        "action_handle": "a1",
        "decision": decision,
        "semantic_goal": "在约定时间开始补偿考核。",
        "reason": "用户要求在未来时间开始补偿考核。",
        "scheduled_authority_proposal": {
            "schema_version": "scheduled_authority_proposal.v1",
            "temporal_alignment": temporal_alignment,
            "authorized_content_summary": "在约定时间开始补偿考核。",
            "authorized_detail_refs": [
                {
                    "evidence_handle": "e1",
                    "semantic_summary": (
                        "当前对话明确约定在该时间开始补偿考核。"
                    ),
                    "provenance_role": "current_event",
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_future_speak_temporal_mismatch_uses_existing_planner_replacement_budget() -> None:
    """A stale relative-time candidate repairs inside the existing budget."""

    captured_repairs: list[str] = []
    responses = [
        _planner_response(
            actions=[_future_speak_row(temporal_alignment="relative_date_mismatch")]
        ),
        _planner_response(actions=[_future_speak_row()]),
        {"decisions": {"c1": True}},
    ]
    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            nonlocal calls
            del config
            if calls == 1:
                captured_repairs.append(str(messages[-1].content))
            response = responses[calls]
            calls += 1
            return SimpleNamespace(content=json.dumps(response))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-future-speak",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
            "created_at": "2026-05-15T21:00:00Z",
        },
        evidence=[{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode-future-speak",
                "occurred_at": "2026-05-15T21:00:00Z",
                "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
            },
            "semantic_text": "当前对话明确约定在该时间开始补偿考核。",
            "visible_to": ["q:event_agency"],
            "authority": "current_event",
        }],
        available_actions=[_future_speak_affordance()],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert calls == 3
    assert len(captured_repairs) == 1
    assert "scheduled_authority_contract" in captured_repairs[0]
    assert "relative_date_mismatch" in captured_repairs[0]
    assert result["action_requests"] == [{
        "action_kind": "future_speak",
        "decision": "2026-05-16 10:00",
        "context_ref": "",
        "semantic_goal": "在约定时间开始补偿考核。",
        "reason": "用户要求在未来时间开始补偿考核。",
        "target_roles": _bid("ordinary_response")["target_roles"],
        "evidence_handles": ["e1"],
        "scheduled_authority_proposal": _future_speak_row()[
            "scheduled_authority_proposal"
        ],
    }]


@pytest.mark.asyncio
async def test_future_speak_authority_exhaustion_returns_no_action() -> None:
    """Planner exhaustion produces no future-speak action."""

    responses = [
        _planner_response(
            actions=[_future_speak_row(temporal_alignment="past_or_not_future")]
        )
        for _ in range(ACTION_PLANNING_ATTEMPT_LIMIT)
    ]
    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            nonlocal calls
            del messages, config
            response = responses[calls]
            calls += 1
            return SimpleNamespace(content=json.dumps(response))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode={
            "episode_id": "episode-future-speak-exhausted",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
            "created_at": "2026-05-15T21:00:00Z",
        },
        evidence=[{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode-future-speak-exhausted",
                "occurred_at": "2026-05-15T21:00:00Z",
                "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
            },
            "semantic_text": "当前对话明确约定在该时间开始补偿考核。",
            "visible_to": ["q:event_agency"],
            "authority": "current_event",
        }],
        available_actions=[_future_speak_affordance()],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert calls == ACTION_PLANNING_ATTEMPT_LIMIT
    assert result["action_requests"] == []
    assert result["goal_resolution"] == "blocked"


def _future_speak_evidence() -> list[dict[str, object]]:
    """Build the current-episode evidence row used by planner tests."""

    return [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode-future-speak",
            "occurred_at": "2026-05-15T21:00:00Z",
            "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
        },
        "semantic_text": "当前对话明确约定在该时间开始补偿考核。",
        "visible_to": ["q:event_agency"],
        "authority": "current_event",
    }]


def _future_speak_episode(
    *,
    episode_id: str = "episode-future-speak",
    created_at: str = "2026-05-15T21:00:00Z",
    expression: str = "今晚十点提醒我。",
) -> dict[str, object]:
    """Build the user-message episode with the original relative expression."""

    return {
        "episode_id": episode_id,
        "trigger_source": "user_message",
        "output_mode": "visible_reply",
        "created_at": created_at,
        "percepts": [{
            "percept_kind": "dialog",
            "source_kind": "dialog",
            "source_id": f"{episode_id}:dialog",
            "content": {
                "semantic_text": expression,
                "text": expression,
            },
            "observed_at": created_at,
        }],
    }


@pytest.mark.asyncio
async def test_future_speak_plan_actions_preserves_authority_proposal_to_runtime_output() -> None:
    """The real plan_actions materialization keeps the validated proposal."""

    responses = [
        _planner_response(actions=[_future_speak_row()]),
        {"decisions": {"c1": True}},
    ]
    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            nonlocal calls
            del messages, config
            response = responses[calls]
            calls += 1
            return SimpleNamespace(content=json.dumps(response))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode=_future_speak_episode(),
        evidence=_future_speak_evidence(),
        available_actions=[_future_speak_affordance()],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert calls == 2
    assert len(result["action_requests"]) == 1
    action_request = result["action_requests"][0]
    assert action_request["action_kind"] == "future_speak"
    assert action_request["decision"] == "2026-05-16 10:00"
    assert action_request["scheduled_authority_proposal"] == (
        _future_speak_row()["scheduled_authority_proposal"]
    )


@pytest.mark.asyncio
async def test_future_speak_relative_time_prompt_uses_accepted_event_local_context() -> None:
    """The planner receives the original expression and accepted local context."""

    captured_payloads: list[dict[str, object]] = []
    captured_repairs: list[str] = []
    responses = [
        _planner_response(
            actions=[_future_speak_row(decision="2025-05-23 22:00")],
            goal_resolution="answerable_now",
        ),
        _planner_response(
            actions=[_future_speak_row()],
            goal_resolution="answerable_now",
        ),
        {"decisions": {"c1": True}},
    ]
    calls = 0

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            nonlocal calls
            del config
            if calls == 0:
                captured_payloads.append(
                    json.loads(str(messages[-1].content))
                )
            if calls == 1:
                captured_repairs.append(str(messages[-1].content))
            response = responses[calls]
            calls += 1
            return SimpleNamespace(content=json.dumps(response))

    result = await plan_actions(
        primary_bid=_bid("ordinary_response"),
        supporting_bids=[],
        episode=_future_speak_episode(),
        evidence=_future_speak_evidence(),
        available_actions=[_future_speak_affordance()],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert calls == 3
    assert len(captured_repairs) == 1
    scheduled_context = captured_payloads[0]["scheduled_authority_context"]
    assert captured_payloads[0]["evidence"][0]["provenance_role"] == (
        "current_event"
    )
    assert scheduled_context["original_relative_expression"] == "今晚十点提醒我。"
    assert scheduled_context["accepted_local_datetime"] == "2026-05-16 09:00"
    assert scheduled_context["accepted_timezone"] == "Pacific/Auckland"
    assert "2025-05-23 22:00" in captured_repairs[0]
    assert result["action_requests"][0]["decision"] == "2026-05-16 10:00"
    assert result["action_requests"][0]["scheduled_authority_proposal"] == (
        _future_speak_row()["scheduled_authority_proposal"]
    )
