"""Tests for cognition resolver capability execution."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    build_canonical_plan_question,
    build_canonical_turn_workspace,
)
from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref
from kazusa_ai_chatbot.cognition_resolver import capabilities as capabilities_module
from kazusa_ai_chatbot.cognition_resolver import loop as loop_module
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    PENDING_TASK_CONTINUATION_VERSION,
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    RESOLVER_GOAL_PROGRESS_VERSION,
    RESOLVER_OBSERVATION_VERSION,
    RESOLVER_PENDING_RESOLUTION_VERSION,
    RESOLVER_PENDING_RESUME_VERSION,
    ResolverValidationError,
)
from kazusa_ai_chatbot.cognition_resolver.loop import call_cognition_resolver_loop
from kazusa_ai_chatbot.cognition_resolver.pending import (
    RESOLVER_PENDING_APPROVAL_ACTION_KIND,
    RESOLVER_PENDING_HIL_ACTION_KIND,
    apply_pending_resolution,
    build_pending_resume_record,
    load_matching_pending_resume,
    load_matching_pending_resume_into_state,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    ensure_initial_resolver_inputs,
    project_resolver_context,
)
from kazusa_ai_chatbot.cognition_resolver.telemetry import (
    build_resolver_cycle_event,
    build_resolver_terminal_event,
    write_human_readable_resolver_trace,
)
from kazusa_ai_chatbot.cognition_shared.contracts import CognitionExecutionError
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_test_helpers import (
    canonical_character_identity,
    canonical_episode,
    canonical_identity_context,
)


def _resolver_request(
    *,
    capability_kind: str = "task_resolution_request",
    objective: str = "检索当前用户与这个问题有关的关系和记忆证据。",
) -> dict:
    return {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": capability_kind,
        "objective": objective,
        "reason": "当前认知循环缺少足够证据。",
        "priority": "now",
        "goal_continuation_ref": (
            _resolver_goal_ref()
            if capability_kind == "task_resolution_request"
            else None
        ),
    }


def _resolver_goal_ref() -> dict[str, object]:
    """Build the continuation identity shared by resolver loop fixtures."""

    return build_goal_continuation_ref(
        source_episode_id="resolver-capability-episode",
        source_message_id="message-123",
        branch_id="ordinary_response",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "resolver-goal-001",
        },
    )


def _resolver_scene_context() -> dict[str, object]:
    """Build one bounded scene for resolver capability tests."""

    return {
        "channel_scope": "private",
        "character_role": "Kazusa",
        "current_user_role": "Test User",
        "semantic_scene": "The resolver is handling one bounded evidence goal.",
        "public_group_scene": "",
        "conversation_continuity": "The current turn continues the evidence goal.",
        "semantic_temporal_context": "The current test turn is active.",
    }


def _task_observation_fields(request: dict) -> dict[str, object]:
    """Add the continuation field only to task-resolution observations."""

    if request["capability_kind"] != "task_resolution_request":
        return {}
    return {"goal_continuation_ref": request["goal_continuation_ref"]}


def _task_result(
    *,
    status: str = "resolved",
    specialist: str = "dsh",
    summary: str = "找到一条相关证据。",
    semantic_objective: str = "检索当前用户与这个问题有关的关系和记忆证据。",
) -> dict[str, object]:
    """Build one canonical task-resolution result for resolver tests."""

    evidence = []
    if status in {"resolved", "partial"}:
        evidence = [{
            "schema_version": "task_resolution_evidence.v1",
            "evidence_id": "evidence-1",
            "task_node_id": "node-1",
            "specialist": specialist,
            "summary": "evidence-1",
            "provenance_refs": ["source:item-1"],
            "limitations": [],
        }]
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": semantic_objective,
        "status": status,
        "scene_context": _resolver_scene_context(),
        "goal_continuation_ref": _resolver_goal_ref(),
        "evidence_state": {
            "resolved": "complete",
            "partial": "partial",
            "deferred": "pending",
            "needs_user_input": "pending",
            "approval_required": "pending",
            "unavailable": "missing",
            "failed": "blocked",
        }.get(status, "blocked"),
        "evidence_excerpts": [summary]
        if status in {"resolved", "partial"}
        else [],
        "evidence_handles": ["evidence-1"]
        if status in {"resolved", "partial"}
        else [],
        "prompt_safe_summary": summary,
        "evidence": evidence,
        "completed_subgoals": [],
        "remaining_needs": (
            ["需要用户补充缺失指代。"]
            if status == "needs_user_input"
            else ["The bounded task requires a truthful terminal disposition."]
            if status in {"unavailable", "failed"}
            else []
        ),
        "checkpoint": {},
        "coding_run_context": {},
    }


def _resolver_state() -> dict:
    turn_clock = build_turn_clock("2026-05-30 09:00:00")
    episode = canonical_episode(
        episode_id="resolver-capability-episode",
        content="Need an evidence-backed answer.",
        current_global_user_id="global-user-123",
    )
    return {
        "decontextualized_input": "Original user request about trust.",
        "referents": [],
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-123",
        },
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "channel_type": "private",
        "platform_message_id": "message-123",
        "platform_bot_id": "bot-123",
        "global_user_id": "global-user-123",
        "platform_user_id": "platform-user-123",
        "user_name": "Test User",
        "user_profile": {"relationship_state": 500},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "prompt_message_context": {
            "body_text": "Need an evidence-backed answer.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-123"],
            "broadcast": False,
        },
        "channel_topic": "debug",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "conversation_progress": {
            "current_thread": "trust question",
        },
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
        "cognition_scene_context": _resolver_scene_context(),
        "active_turn_platform_message_ids": ["message-123"],
        "active_turn_conversation_row_ids": ["row-123"],
        "cognitive_episode": episode,
        "pending_task_continuation": {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "no_task_admission",
        },
    }


def _internal_thought_resolver_state() -> dict:
    """Return resolver state for a private internal-thought cognition source."""

    state = _resolver_state()
    episode = canonical_episode(
        episode_id="resolver-internal-thought-episode",
        trigger_source="internal_thought",
        content="整理一个内部目标。",
    )
    state["cognitive_episode"] = episode
    state["channel_type"] = "group"
    return_value = state
    return return_value


def _cognition_result(
    *,
    internal_monologue: str,
    action_specs: list[dict] | None = None,
    resolver_requests: list[dict] | None = None,
    goal_resolution: str = "requires_required_evidence",
) -> dict:
    return {
        "internal_monologue": internal_monologue,
        "interaction_subtext": f"{internal_monologue} 的互动潜台词",
        "emotional_appraisal": f"{internal_monologue} 的情绪判断",
        "character_intent": f"{internal_monologue} 的角色意图",
        "logical_stance": f"{internal_monologue} 的逻辑立场",
        "judgment_note": f"{internal_monologue} 的判断备注",
        "social_distance": "close",
        "emotional_intensity": "low",
        "vibe_check": "steady",
        "relational_dynamic": "trusted",
        "action_specs": action_specs or [],
        "resolver_capability_requests": resolver_requests or [],
        "goal_resolution": goal_resolution,
    }


def _speak_action_spec(
    reason: str = "已经有足够证据，可以进入可见回复。",
    *,
    surface_role: str = "ordinary",
    goal_continuation_ref: dict[str, object] | None = None,
) -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": "speak",
        "cognition_mode": "deliberative",
        "surface_role": surface_role,
        "goal_continuation_ref": goal_continuation_ref,
        "source_refs": [],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_user",
            "target_id": None,
            "owner": "l2d",
            "scope": {},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {},
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": reason,
    }


def _pending_resume(*, capability_kind: str = "human_clarification") -> dict:
    status = "waiting_for_user"
    question = "你现在在哪个城市？"
    approval_summary = ""
    if capability_kind == "approval_preparation":
        status = "waiting_for_approval"
        question = ""
        approval_summary = "准备创建提醒，但需要用户确认。"
    pending_resume = {
        "schema_version": RESOLVER_PENDING_RESUME_VERSION,
        "resume_id": f"resolver-pending-{capability_kind}",
        "capability_kind": capability_kind,
        "status": status,
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "global_user_id": "global-user-123",
        "source_message_id": "previous-message-123",
        "prompt_safe_original_goal": "Original user request about trust.",
        "prompt_safe_question": question,
        "prompt_safe_approval_summary": approval_summary,
        "created_at_utc": "2026-05-29T21:00:00+00:00",
        "expires_at_utc": "2026-05-30T21:00:00+00:00",
    }
    if capability_kind == "human_clarification":
        pending_resume["pending_task_continuation"] = {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "no_task_admission",
        }
    return pending_resume


def _pending_resolution(
    *,
    decision: str = "answered",
    resume_id: str = "resolver-pending-human_clarification",
) -> dict:
    return {
        "schema_version": RESOLVER_PENDING_RESOLUTION_VERSION,
        "resume_id": resume_id,
        "decision": decision,
        "reason": "用户已经回答了澄清问题。",
    }


def _goal_progress(*, focus: str = "继续完成原始目标。") -> dict:
    return {
        "schema_version": RESOLVER_GOAL_PROGRESS_VERSION,
        "original_goal": "今晚安排一个两小时低预算计划。",
        "current_focus": focus,
        "deliverables": [
            {
                "description": "晚餐候选和证据边界",
                "status": "partial",
                "note": "候选方向已有，实时营业仍需 caveat。",
            },
            {
                "description": "两小时散步路线和时间切分",
                "status": "pending",
                "note": "最终回复必须覆盖。",
            },
        ],
        "missing_user_inputs": [],
        "evidence_dependencies": ["当前营业状态"],
        "attempted_paths": [],
        "source_backed_facts": ["用户在奥克兰 CBD，预算 20 NZD"],
        "assumptions_or_inferences": ["可以给出海滨散步路线骨架"],
        "blockers": ["无法确认所有店 19:30 营业"],
        "final_response_requirements": [
            "覆盖晚餐、散步、时间切分和核实清单",
        ],
    }


def test_resolver_context_projects_original_goal_and_objectives() -> None:
    """Next cognition cycles should see the goal and attempted evidence path."""

    state = ensure_initial_resolver_inputs(_resolver_state(), max_cycles=3)
    resolver_state = dict(state["resolver_state"])
    resolver_state["observations"] = [
        {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_web_1",
            "capability_kind": "task_resolution_request",
            "goal_continuation_ref": _resolver_goal_ref(),
            "request_objective": "检索奥克兰 CBD 餐厅当前营业状态。",
            "request_reason": "需要当前营业证据。",
            "status": "failed",
            "prompt_safe_summary": "搜索工具未返回已确认事实。",
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        },
    ]

    resolver_context = project_resolver_context(resolver_state)

    assert "original_goal=Original user request about trust." in resolver_context
    assert "objective=检索奥克兰 CBD 餐厅当前营业状态。" in resolver_context
    assert "summary=搜索工具未返回已确认事实。" in resolver_context
    assert "resolver_goal_progress:" in resolver_context


def test_task_resolution_context_preserves_scene_and_continuation_ref() -> None:
    """Capability context carries the same bounded scene and goal identity."""

    from kazusa_ai_chatbot.cognition_resolver.capabilities import (
        _task_resolution_execution_context_from_state,
    )

    state = _resolver_state()
    continuation_ref = _resolver_goal_ref()
    context = _task_resolution_execution_context_from_state(
        state,
        goal_continuation_ref=continuation_ref,
    )

    assert context["scene_context"] == state["cognition_scene_context"]
    assert context["goal_continuation_ref"] == continuation_ref


@pytest.mark.asyncio
async def test_task_resolution_handoff_rejects_missing_cognition_scene_before_execution(
) -> None:
    """A missing cognition carrier fails before the capability executor."""

    state = _resolver_state()
    state.pop("cognition_scene_context")
    request = _resolver_request()
    call_cognition = AsyncMock(return_value=_cognition_result(
        internal_monologue="The task requires bounded evidence.",
        resolver_requests=[request],
    ))
    execute_capability = AsyncMock()

    with pytest.raises(
        ResolverValidationError,
        match="cognition_scene_context",
    ):
        await call_cognition_resolver_loop(
            state,
            call_cognition_subgraph_func=call_cognition,
            execute_capability_func=execute_capability,
            max_cycles=3,
            capability_timeout_seconds=1.0,
        )

    call_cognition.assert_awaited_once()
    execute_capability.assert_not_awaited()


@pytest.mark.asyncio
async def test_required_dependency_references_single_task_observation() -> None:
    """A required dependency points to the observation that owns semantics."""

    request = _resolver_request(objective="Resolve one bounded evidence goal.")
    evidence_ref = {
        "schema_version": "evidence_ref.v1",
        "evidence_kind": "system_event",
        "evidence_id": "resolver-evidence-1",
        "owner": "cognition_resolver",
        "excerpt": "A bounded source-backed fact.",
        "observed_at": "2026-05-29T21:00:00+00:00",
    }

    async def call_cognition(state: dict) -> dict:
        if state.get("resolver_state", {}).get("observations"):
            return _cognition_result(
                internal_monologue="The task result is ready.",
                action_specs=[_speak_action_spec(
                    "Present the validated task result.",
                    surface_role="task_result",
                    goal_continuation_ref=_resolver_goal_ref(),
                )],
                resolver_requests=[],
                goal_resolution="answerable_now",
            )
        return _cognition_result(
            internal_monologue="The task requires bounded evidence.",
            resolver_requests=[request],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_typed_evidence",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "succeeded",
            "prompt_safe_summary": "The bounded evidence is ready.",
            "evidence_refs": [evidence_ref],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "complete",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    resolver_state = result["resolver_state"]
    assert resolver_state["observations"][0]["goal_continuation_ref"] == (
        request["goal_continuation_ref"]
    )
    assert resolver_state["required_resolver_evidence_dependency"] == {
        "schema_version": "required_resolver_evidence_dependency.v2",
        "accepted_request_handle": "resolver_request_0_1",
        "observation_id": "resolver_obs_typed_evidence",
    }


@pytest.mark.asyncio
async def test_invalid_required_dependency_fails_closed_before_l3() -> None:
    """Invalid referenced evidence becomes one retryable pre-commit error."""

    state = ensure_initial_resolver_inputs(_resolver_state(), max_cycles=3)
    resolver_state = dict(state["resolver_state"])
    resolver_state["required_resolver_evidence_dependency"] = {
        "schema_version": "required_resolver_evidence_dependency.v2",
        "accepted_request_handle": "resolver_request_0_1",
        "observation_id": "missing-observation",
    }
    state["resolver_state"] = resolver_state
    call_cognition = AsyncMock()
    execute_capability = AsyncMock()

    with pytest.raises(CognitionExecutionError) as error:
        await call_cognition_resolver_loop(
            state,
            call_cognition_subgraph_func=call_cognition,
            execute_capability_func=execute_capability,
            max_cycles=3,
            capability_timeout_seconds=1.0,
        )

    assert error.value.error_code == "resolver_state_contract"
    assert error.value.stage == "cognition_resolver"
    assert error.value.attempt_count == 1
    assert error.value.safe_checkpoint == "pre_state_commit"
    assert error.value.retryable is True
    call_cognition.assert_not_awaited()
    execute_capability.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_background_goal_reaches_acknowledgement_without_factual_surface() -> None:
    """The resolver fails closed when one goal mixes pending and factual roles."""

    request = _resolver_request(objective="Resolve one pending background goal.")
    factual_action = _speak_action_spec(
        "Answer the goal from current evidence.",
        surface_role="factual_answer",
        goal_continuation_ref=_resolver_goal_ref(),
    )
    calls = 0

    async def call_cognition(_state: dict) -> dict:
        nonlocal calls
        calls += 1
        return _cognition_result(
            internal_monologue="The model proposed a mixed lifecycle.",
            action_specs=[factual_action],
            resolver_requests=[request],
        )

    async def execute_capability(_request: dict, _state: dict) -> dict:
        raise AssertionError("mixed lifecycle must fail before task execution")

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    assert calls == 2
    assert all(
        row.get("surface_role") != "factual_answer"
        for row in result["action_specs"]
    )
    assert any(
        row.get("surface_role") == "task_status"
        and row.get("goal_continuation_ref") == request["goal_continuation_ref"]
        for row in result["action_specs"]
    )


async def _hil_pending_trace_state() -> dict:
    """Build a resolver result containing one pending HIL terminal trace."""

    request = _resolver_request(
        capability_kind="human_clarification",
        objective="请只问用户所在城市。",
    )

    async def call_cognition(state: dict) -> dict:
        resolver_context = state["resolver_context"]
        if "pending_resolver_resume" not in resolver_context:
            return _cognition_result(
                internal_monologue="第一轮：缺少用户城市",
                resolver_requests=[request],
            )
        return _cognition_result(
            internal_monologue="第二轮：提出最小澄清问题",
            action_specs=[_speak_action_spec("只询问用户所在城市。")],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_hil_city",
            "capability_kind": capability_request["capability_kind"],
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "blocked",
            "prompt_safe_summary": (
                "Human clarification required: 请只问用户所在城市。"
            ),
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    async def upsert_pending_resume(state: dict, observation: dict) -> dict:
        record = build_pending_resume_record(state, observation)
        return record["execution_result"]["pending_resume"]

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        upsert_pending_resume_func=upsert_pending_resume,
    )
    return_value = result
    return return_value


@pytest.mark.asyncio
async def test_loop_runs_cognition_capability_then_cognition_again() -> None:
    """The resolver must recur through cognition after a capability observation."""

    request = _resolver_request(
        objective="检索信任判断需要的关系证据。",
    )
    final_action = _speak_action_spec()
    cognition_inputs: list[dict] = []
    capability_inputs: list[tuple[dict, dict]] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        if len(cognition_inputs) == 1:
            return _cognition_result(
                internal_monologue="第一轮：证据不足",
                resolver_requests=[request],
            )
        return _cognition_result(
            internal_monologue="第二轮：证据足够",
            action_specs=[final_action],
        )

    async def execute_capability(
        capability_request: dict,
        state: dict,
    ) -> dict:
        capability_inputs.append((capability_request, dict(state)))
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_trust_memory",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "succeeded",
            "prompt_safe_summary": "找到一条信任相关记忆。",
            "rag_result": {
                "answer": "用户曾经稳定支持过她的判断。",
                "memory_evidence": [
                    {
                        "summary": "用户在一次困难讨论里支持过她。",
                    },
                ],
            },
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "missing",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    assert len(cognition_inputs) == 2
    assert len(capability_inputs) == 1
    assert capability_inputs[0][0] == request
    assert cognition_inputs[0]["rag_result"]["answer"] == ""
    assert cognition_inputs[0]["resolver_context"].startswith("resolver_state:")
    assert "找到一条信任相关记忆" in cognition_inputs[1]["resolver_context"]
    assert "用户曾经稳定支持过她的判断" in cognition_inputs[1]["resolver_context"]
    assert result["action_specs"] == [final_action]
    assert result["rag_result"]["answer"] == "用户曾经稳定支持过她的判断。"

    resolver_state = result["resolver_state"]
    assert resolver_state["status"] == "terminal"
    assert resolver_state["held_action_specs"] == [final_action]
    assert len(resolver_state["observations"]) == 1
    assert len(resolver_state["cycle_traces"]) == 2
    assert resolver_state["cycle_traces"][0]["selected_capability_kind"] == (
        "task_resolution_request"
    )
    assert resolver_state["cycle_traces"][0]["observation_ids"] == [
        "resolver_obs_trust_memory"
    ]
    assert resolver_state["cycle_traces"][1]["selected_capability_kind"] == ""
    assert resolver_state["cycle_traces"][1]["final_surface_decision"].startswith(
        "action_specs="
    )
    assert resolver_state["cycle_traces"][1]["terminal_reason"] == (
        "no resolver capability request"
    )


@pytest.mark.asyncio
async def test_answerable_now_terminates_without_executing_optional_resolver() -> None:
    """Goal sufficiency ends recurrence even if planning proposed extra recall."""

    request = _resolver_request(
        objective='检索一个并非回答所必需的关系例子。',
    )
    cognition_inputs: list[dict] = []
    capability_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        return _cognition_result(
            internal_monologue='当前问题已有足够依据，可以直接回答。',
            resolver_requests=[request],
            goal_resolution="answerable_now",
            action_specs=[_speak_action_spec('当前输入已经足够完成回答。')],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        capability_inputs.append(capability_request)
        raise AssertionError("optional resolver must not execute")

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    assert len(cognition_inputs) == 1
    assert capability_inputs == []
    assert result["resolver_capability_requests"] == []
    assert result["resolver_state"]["status"] == "terminal"
    assert result["resolver_state"]["terminal_reason"] == (
        "goal answerable now; optional resolver request suppressed"
    )


@pytest.mark.asyncio
async def test_answerable_now_is_independent_of_unresolved_conversation_source() -> None:
    """Source coverage false must not override an answerable goal decision."""

    state = _resolver_state()
    state["rag_result"] = {
        "conversation_evidence": [{
            "resolved": False,
            "missing_context": ["conversation_evidence"],
        }],
    }
    request = _resolver_request(
        objective="retrieve optional conversation evidence",
    )
    cognition_inputs: list[dict] = []
    capability_inputs: list[dict] = []

    async def call_cognition(current_state: dict) -> dict:
        cognition_inputs.append(dict(current_state))
        return _cognition_result(
            internal_monologue="The current bid and input are sufficient.",
            resolver_requests=[request],
            goal_resolution="answerable_now",
            action_specs=[_speak_action_spec("Answer from the current goal.")],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        capability_inputs.append(capability_request)
        raise AssertionError("unresolved optional source must not execute")

    result = await call_cognition_resolver_loop(
        state,
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    assert len(cognition_inputs) == 1
    assert capability_inputs == []
    assert result["resolver_capability_requests"] == []
    assert result["resolver_state"]["status"] == "terminal"


@pytest.mark.asyncio
async def test_loop_projects_goal_progress_across_iterations() -> None:
    """Every later cognition cycle should see L2d's goal checklist."""

    request = _resolver_request(objective="检索今晚计划需要的当前事实。")
    final_action = _speak_action_spec("给出晚餐加散步的完整计划。")
    cognition_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        if len(cognition_inputs) == 1:
            output = _cognition_result(
                internal_monologue="第一轮：需要外部证据",
                resolver_requests=[request],
            )
            output["resolver_goal_progress"] = _goal_progress(
                focus="先取得当前营业证据。",
            )
            return output
        assert "resolver_goal_progress:" in state["resolver_context"]
        assert "两小时散步路线和时间切分" in state["resolver_context"]
        output = _cognition_result(
            internal_monologue="第二轮：证据不足但可以最佳努力完成",
            action_specs=[final_action],
        )
        output["resolver_goal_progress"] = _goal_progress(
            focus="最终回答要覆盖完整计划和证据阻塞。",
        )
        return output

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_evening_plan",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "没有确认到每家店当前营业状态。",
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    assert len(cognition_inputs) == 2
    assert result["resolver_goal_progress"]["current_focus"] == (
        "最终回答要覆盖完整计划和证据阻塞。"
    )
    goal_progress = result["resolver_state"]["goal_progress"]
    assert goal_progress["deliverables"][1]["description"] == (
        "两小时散步路线和时间切分"
    )
    assert "覆盖晚餐、散步、时间切分和核实清单" in (
        result["resolver_context"]
    )


@pytest.mark.asyncio
async def test_loop_records_timeout_observation_then_returns_to_cognition() -> None:
    """Capability timeouts should become observations, not Python decisions."""

    request = _resolver_request(
        capability_kind="self_goal_resolution",
        objective="检索一个会超时的证据目标。",
    )
    cognition_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        if len(cognition_inputs) == 1:
            return _cognition_result(
                internal_monologue="第一轮：需要证据",
                resolver_requests=[request],
            )
        return _cognition_result(
            internal_monologue="第二轮：看到超时阻塞",
            action_specs=[_speak_action_spec("证据工具超时，所以说明限制。")],
        )

    async def execute_capability(_request: dict, _state: dict) -> dict:
        await AsyncMock()()
        await asyncio.sleep(1.0)
        raise AssertionError("wait_for should timeout before this returns")

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=0.01,
    )

    assert len(cognition_inputs) == 2
    assert "timed out" in cognition_inputs[1]["resolver_context"]
    observation = result["resolver_state"]["observations"][0]
    assert observation["status"] == "failed"
    assert observation["capability_kind"] == "self_goal_resolution"
    assert observation["request_objective"] == request["objective"]
    assert "timed out" in observation["prompt_safe_summary"]
    terminal_event = build_resolver_terminal_event(result, duration_ms=1)
    terminal_json = json.dumps(terminal_event, ensure_ascii=False)
    assert "failed" in terminal_json
    assert "timed out" in terminal_json
    assert "Need an evidence-backed answer" not in terminal_json
    assert "message-123" not in terminal_json


@pytest.mark.asyncio
async def test_task_resolution_uses_task_service_timeout_without_detached_resolver_task(
) -> None:
    """The loop awaits the task service's already-bounded lifecycle result."""

    request = _resolver_request(objective="检索需要转入后台的当前证据。")

    async def call_cognition(state: dict) -> dict:
        if not state["resolver_state"]["observations"]:
            return _cognition_result(
                internal_monologue="第一轮：需要当前证据。",
                resolver_requests=[request],
            )
        return _cognition_result(
            internal_monologue="第二轮：前台等待结束，后台接管。",
            action_specs=[_speak_action_spec("我会在结果完成后继续。")],
        )

    async def execute_capability(_request: dict, _state: dict) -> dict:
        await asyncio.sleep(0.02)
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_deferred_handoff",
            "capability_kind": request["capability_kind"],
            "goal_continuation_ref": request["goal_continuation_ref"],
            "request_objective": request["objective"],
            "request_reason": request["reason"],
            "status": "succeeded",
            "prompt_safe_summary": "DSH work was durably promoted.",
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "pending",
                "remaining_needs": [request["objective"]],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=0.001,
    )

    observation = result["resolver_state"]["observations"][0]
    assert observation["status"] == "succeeded"
    assert observation["prompt_safe_summary"] == (
        "DSH work was durably promoted."
    )
    assert not hasattr(loop_module, "_PENDING_TASK_RESOLUTION_EXECUTIONS")


@pytest.mark.asyncio
async def test_duplicate_task_blocker_replaces_dependency_reference_without_copying_state(
) -> None:
    """A duplicate blocker moves the V2 reference without copying semantics."""

    request = _resolver_request(
        capability_kind="task_resolution_request",
        objective="检索同一个外部证据目标。",
    )
    execute_count = 0

    async def call_cognition(state: dict) -> dict:
        resolver_context = state["resolver_context"]
        if "duplicate capability request" in resolver_context:
            return _cognition_result(
                internal_monologue="第三轮：重复请求已被阻止",
                action_specs=[_speak_action_spec("说明重复检索已阻塞。")],
            )
        return _cognition_result(
            internal_monologue="需要同一个外部证据",
            resolver_requests=[request],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        nonlocal execute_count
        execute_count += 1
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": f"resolver_obs_web_{execute_count}",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "没有找到已确认事实。",
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    observations = result["resolver_state"]["observations"]
    assert execute_count == 1
    assert result["resolver_state"]["status"] == "blocked"
    assert observations[-1]["observation_id"] == "resolver_obs_duplicate_request"
    assert observations[-1]["request_objective"] == request["objective"]
    assert result["resolver_state"]["required_resolver_evidence_dependency"] == {
        "schema_version": "required_resolver_evidence_dependency.v2",
        "accepted_request_handle": "resolver_request_0_1",
        "observation_id": "resolver_obs_duplicate_request",
    }
    assert result["action_specs"][0]["kind"] == "speak"


def test_loop_blocks_rephrased_task_request_with_same_continuation_ref() -> None:
    """One typed goal cannot be admitted twice under paraphrased objectives."""

    first_request = _resolver_request(
        objective="Resolve the bounded task objective.",
    )
    rephrased_request = _resolver_request(
        objective="Resolve the same bounded task with different wording.",
    )
    resolver_state = {
        "observations": [{
            "capability_kind": "task_resolution_request",
            "request_objective": first_request["objective"],
            "status": "succeeded",
            "goal_continuation_ref": first_request["goal_continuation_ref"],
        }],
    }

    assert loop_module._is_repeated_capability_request(
        rephrased_request,
        resolver_state,
    ) is True


def _production_resolver_state(*, original_goal: str) -> dict:
    """Build resolver state that crosses the production cognition projector."""

    state = _resolver_state()
    timestamp = "2026-05-29T21:00:00Z"
    state.pop("conversation_progress", None)
    profile = canonical_character_identity(marker="resolver-continuity")
    profile["global_user_id"] = "character-123"
    state["character_profile"] = profile
    state["character_identity_context"] = canonical_identity_context(
        marker="resolver-continuity",
    )
    state["character_cognition_state"] = build_character_production_state(
        updated_at=timestamp,
    )
    state["cognition_state"] = build_acquaintance_user_state(
        global_user_id="global-user-123",
        updated_at=timestamp,
    )
    state["decontextualized_input"] = original_goal
    return state


@pytest.mark.asyncio
async def test_loop_blocks_same_capability_retry_after_timeout() -> None:
    """Timed-out capability work should not be retried with renamed objective."""

    first_request = _resolver_request(
        capability_kind="self_goal_resolution",
        objective="检索当前外部事实。",
    )
    renamed_request = _resolver_request(
        capability_kind="self_goal_resolution",
        objective="换一种说法再次检索当前外部事实。",
    )
    execute_count = 0

    async def call_cognition(state: dict) -> dict:
        resolver_context = state["resolver_context"]
        if "duplicate capability request" in resolver_context:
            return _cognition_result(
                internal_monologue="第三轮：重复超时检索已被阻止。",
                action_specs=[_speak_action_spec("说明外部检索超时。")],
            )
        if "timed out" in resolver_context:
            return _cognition_result(
                internal_monologue="第二轮：想换个目标继续查。",
                resolver_requests=[renamed_request],
            )
        return _cognition_result(
            internal_monologue="第一轮：需要外部证据。",
            resolver_requests=[first_request],
        )

    async def execute_capability(_request: dict, _state: dict) -> dict:
        nonlocal execute_count
        execute_count += 1
        await asyncio.sleep(1.0)
        raise AssertionError("wait_for should timeout before this returns")

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=0.01,
    )

    observations = result["resolver_state"]["observations"]
    assert execute_count == 1
    assert result["resolver_state"]["status"] == "blocked"
    assert observations[-1]["observation_id"] == "resolver_obs_duplicate_request"
    assert observations[-1]["request_objective"] == renamed_request["objective"]
    assert result["action_specs"][0]["kind"] == "speak"


@pytest.mark.asyncio
async def test_loop_blocks_renamed_retry_after_failed_observation() -> None:
    """Failed capability work cannot be retried under a renamed objective."""

    first_request = _resolver_request(
        capability_kind="task_resolution_request",
        objective="Retrieve the prior agreement.",
    )
    renamed_request = _resolver_request(
        capability_kind="task_resolution_request",
        objective="Confirm the earlier agreement with different wording.",
    )
    execute_count = 0

    async def call_cognition(state: dict) -> dict:
        resolver_context = state["resolver_context"]
        if "duplicate capability request" in resolver_context:
            return _cognition_result(
                internal_monologue="The failed retry has been blocked.",
                action_specs=[_speak_action_spec("Continue without the evidence.")],
            )
        if "status=failed" in resolver_context:
            return _cognition_result(
                internal_monologue="Try the same capability with new wording.",
                resolver_requests=[renamed_request],
            )
        return _cognition_result(
            internal_monologue="The prior agreement needs evidence.",
            resolver_requests=[first_request],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        nonlocal execute_count
        execute_count += 1
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": f"resolver_obs_failed_{execute_count}",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "Local context resolution failed.",
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    observations = result["resolver_state"]["observations"]
    assert execute_count == 1
    assert result["resolver_state"]["status"] == "blocked"
    assert observations[-1]["observation_id"] == "resolver_obs_duplicate_request"
    assert observations[-1]["request_objective"] == renamed_request["objective"]
    assert result["action_specs"][0]["kind"] == "speak"


@pytest.mark.asyncio
async def test_duplicate_final_cognition_repeated_request_gets_terminal_speak() -> None:
    """Terminal duplicate handling should not leave the user with silence."""

    request = _resolver_request(
        capability_kind="task_resolution_request",
        objective="检索同一个当前外部事实目标。",
    )
    execute_count = 0

    async def call_cognition(_state: dict) -> dict:
        return _cognition_result(
            internal_monologue="仍然重复请求同一个外部证据。",
            resolver_requests=[request],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        nonlocal execute_count
        execute_count += 1
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": f"resolver_obs_web_{execute_count}",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "没有找到已确认事实。",
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    action_spec = result["action_specs"][0]
    surface_requirements = action_spec["params"]["surface_requirements"]

    assert execute_count == 1
    assert result["resolver_capability_requests"] == []
    assert result["resolver_state"]["status"] == "blocked"
    assert result["resolver_state"]["terminal_reason"] == (
        "duplicate resolver capability request converted to terminal surface"
    )
    assert action_spec["kind"] == "speak"
    assert action_spec["source_refs"][0]["ref_id"] == (
        "resolver_obs_duplicate_request"
    )
    assert action_spec["cognition_provenance"] == {
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "global-user-123",
        }],
        "evidence_handles": [],
    }
    assert "当前证据获取已经阻塞" in surface_requirements["detail"]
    assert "已由来源支持的事实" in surface_requirements["detail"]
    assert "final_response_requirements" in surface_requirements["detail"]
    assert "临时处理状态或延后承诺" in surface_requirements["detail"]
    assert "泛化说明也不能偷换成未授权的" in surface_requirements["detail"]
    assert "具体当前实体、属性、实时状态" in surface_requirements["detail"]
    assert "不能以追问结尾" in surface_requirements["detail"]


@pytest.mark.asyncio
async def test_duplicate_final_request_replaces_stale_pending_surface() -> None:
    """A blocked repeated task cannot keep a pending-work acknowledgement."""

    request = _resolver_request(
        capability_kind="task_resolution_request",
        objective="Retrieve one unavailable current fact.",
    )
    cognition_call_count = 0

    async def call_cognition(_state: dict) -> dict:
        nonlocal cognition_call_count
        cognition_call_count += 1
        action_specs = []
        if cognition_call_count == 3:
            action_specs = [_speak_action_spec("Work is still running.")]
        return _cognition_result(
            internal_monologue="The same task remains unresolved.",
            resolver_requests=[request],
            action_specs=action_specs,
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_unavailable_fact",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "The external fact is unavailable.",
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [capability_request["objective"]],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    action_spec = result["action_specs"][0]

    assert cognition_call_count == 3
    assert action_spec["source_refs"][0]["ref_id"] == (
        "resolver_obs_duplicate_request"
    )
    assert action_spec["params"]["surface_requirements"]["decision"] == (
        "explain terminal evidence blocker"
    )


@pytest.mark.asyncio
async def test_duplicate_final_cognition_changed_request_gets_terminal_speak() -> None:
    """Terminal duplicate handling should not run a rephrased tool request."""

    original_request = _resolver_request(
        capability_kind="task_resolution_request",
        objective="检索同一个当前外部事实目标。",
    )
    rephrased_request = _resolver_request(
        capability_kind="task_resolution_request",
        objective="Search for the same current external evidence with new words.",
    )
    cognition_call_count = 0
    execute_count = 0

    async def call_cognition(_state: dict) -> dict:
        nonlocal cognition_call_count
        cognition_call_count += 1
        if cognition_call_count < 3:
            return _cognition_result(
                internal_monologue="仍然重复请求同一个外部证据。",
                resolver_requests=[original_request],
            )
        return _cognition_result(
            internal_monologue="换个说法继续请求同一个外部证据。",
            resolver_requests=[rephrased_request],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        nonlocal execute_count
        execute_count += 1
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": f"resolver_obs_web_{execute_count}",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "没有找到已确认事实。",
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    action_spec = result["action_specs"][0]
    surface_requirements = action_spec["params"]["surface_requirements"]

    assert execute_count == 1
    assert result["resolver_capability_requests"] == []
    assert result["resolver_state"]["terminal_reason"] == (
        "duplicate resolver capability request converted to terminal surface"
    )
    assert action_spec["kind"] == "speak"
    assert rephrased_request["objective"] in surface_requirements["detail"]


@pytest.mark.asyncio
async def test_duplicate_final_cognition_internal_thought_stays_private() -> None:
    """Internal self-cognition must not get a fabricated visible blocker."""

    request = _resolver_request(
        capability_kind="self_goal_resolution",
        objective="整理一个内部观察目标。",
    )
    execute_count = 0

    async def call_cognition(_state: dict) -> dict:
        return _cognition_result(
            internal_monologue="仍然重复请求私有自我整理。",
            resolver_requests=[request],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        nonlocal execute_count
        execute_count += 1
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": f"resolver_obs_self_{execute_count}",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "succeeded",
            "prompt_safe_summary": "内部自我整理已经完成。",
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _internal_thought_resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    assert execute_count == 1
    assert result["resolver_capability_requests"] == []
    assert result["action_specs"] == []
    assert result["resolver_state"]["status"] == "blocked"
    assert result["resolver_state"]["terminal_reason"] == (
        "duplicate resolver capability request kept private for non-user source"
    )
    assert result["resolver_state"]["held_action_specs"] == []


@pytest.mark.asyncio
async def test_loop_runs_final_cognition_with_max_cycle_blocker() -> None:
    """When capped, the blocker still returns through cognition."""

    request = _resolver_request(objective="持续检索仍然不足的证据。")
    cognition_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        if len(cognition_inputs) == 1:
            return _cognition_result(
                internal_monologue="第一轮：还想继续查",
                resolver_requests=[request],
            )
        return _cognition_result(
            internal_monologue="封顶轮：必须收束",
            action_specs=[_speak_action_spec("循环封顶后收束。")],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_partial",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "证据仍不足。",
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=1,
        capability_timeout_seconds=1.0,
    )

    assert len(cognition_inputs) == 2
    assert "max_cycles" in cognition_inputs[1]["resolver_context"]
    assert "maximum resolver cycles" in cognition_inputs[1]["resolver_context"]
    resolver_state = result["resolver_state"]
    assert resolver_state["status"] == "max_cycles"
    assert resolver_state["terminal_reason"] == "maximum resolver cycles reached"
    assert len(resolver_state["observations"]) == 2
    assert len(resolver_state["cycle_traces"]) == 2


@pytest.mark.asyncio
async def test_loop_converts_max_cycle_request_to_visible_blocker() -> None:
    """A terminal resolver request should not silently suppress final output."""

    request = _resolver_request(
        capability_kind="task_resolution_request",
        objective="继续验证餐厅当前营业和排队情况。",
    )
    cognition_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        return _cognition_result(
            internal_monologue="仍想继续查外部事实",
            resolver_requests=[request],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_partial",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "搜索超时，但已有部分约束可说明。",
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=1,
        capability_timeout_seconds=1.0,
    )

    assert len(cognition_inputs) == 2
    assert result["resolver_capability_requests"] == []
    assert [spec["kind"] for spec in result["action_specs"]] == ["speak"]
    surface_requirements = result["action_specs"][0]["params"][
        "surface_requirements"
    ]
    assert surface_requirements["decision"] == (
        "explain terminal evidence blocker"
    )
    assert "当前证据获取已经阻塞" in surface_requirements["detail"]
    assert "已由来源支持的事实" in surface_requirements["detail"]
    assert "需要核实的最小项目" in surface_requirements["detail"]
    assert "final_response_requirements" in surface_requirements["detail"]
    assert "临时处理状态或延后承诺" in surface_requirements["detail"]
    resolver_state = result["resolver_state"]
    assert resolver_state["status"] == "max_cycles"
    assert resolver_state["held_action_specs"] == result["action_specs"]
    assert resolver_state["terminal_reason"] == (
        "maximum resolver cycles converted to terminal surface"
    )
    assert resolver_state["cycle_traces"][-1]["terminal_reason"] == (
        "maximum resolver cycles converted to terminal surface"
    )


@pytest.mark.asyncio
async def test_max_cycle_request_replaces_stale_pending_surface() -> None:
    """A capped task cannot keep a surface that promises later completion."""

    request = _resolver_request(
        capability_kind="task_resolution_request",
        objective="Retrieve one current fact before answering.",
    )
    cognition_call_count = 0

    async def call_cognition(_state: dict) -> dict:
        nonlocal cognition_call_count
        cognition_call_count += 1
        action_specs = []
        if cognition_call_count == 2:
            action_specs = [_speak_action_spec("Work is still running.")]
        return _cognition_result(
            internal_monologue="The task still needs unavailable evidence.",
            resolver_requests=[request],
            action_specs=action_specs,
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_capped_fact",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "The fact remains unavailable.",
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [capability_request["objective"]],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=1,
        capability_timeout_seconds=1.0,
    )

    action_spec = result["action_specs"][0]

    assert cognition_call_count == 2
    assert action_spec["source_refs"][0]["ref_id"] == (
        "resolver_obs_max_cycles"
    )
    assert action_spec["params"]["surface_requirements"]["decision"] == (
        "explain terminal evidence blocker"
    )


@pytest.mark.asyncio
async def test_max_cycle_internal_thought_request_stays_private() -> None:
    """Internal max-cycle terminal requests must stay private without speech."""

    request = _resolver_request(
        capability_kind="self_goal_resolution",
        objective="继续整理内部观察目标。",
    )

    async def call_cognition(_state: dict) -> dict:
        return _cognition_result(
            internal_monologue="仍然想继续内部自我整理。",
            resolver_requests=[request],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_self_partial",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "succeeded",
            "prompt_safe_summary": "内部自我整理已经完成。",
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _internal_thought_resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=1,
        capability_timeout_seconds=1.0,
    )

    assert result["resolver_capability_requests"] == []
    assert result["action_specs"] == []
    assert result["resolver_state"]["status"] == "max_cycles"
    assert result["resolver_state"]["terminal_reason"] == (
        "maximum resolver cycles kept private for non-user source"
    )
    assert result["resolver_state"]["held_action_specs"] == []


@pytest.mark.asyncio
async def test_hil_blocked_observation_persists_pending_and_reenters_cognition() -> None:
    """HIL blockers should create pending state and run one final cognition."""

    request = _resolver_request(
        capability_kind="human_clarification",
        objective="请只问用户所在城市。",
    )
    pending_rows: list[dict] = []
    cognition_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        if len(cognition_inputs) == 1:
            return _cognition_result(
                internal_monologue="第一轮：缺少用户城市",
                resolver_requests=[request],
            )
        assert "pending_resolver_resume" in state["resolver_context"]
        assert "请只问用户所在城市" in state["resolver_context"]
        return _cognition_result(
            internal_monologue="第二轮：提出最小澄清问题",
            action_specs=[_speak_action_spec("只询问用户所在城市。")],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_hil_city",
            "capability_kind": capability_request["capability_kind"],
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "blocked",
            "prompt_safe_summary": (
                "Human clarification required: 请只问用户所在城市。"
            ),
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    async def upsert_pending_resume(state: dict, observation: dict) -> dict:
        record = build_pending_resume_record(state, observation)
        pending_rows.append(record)
        return record["execution_result"]["pending_resume"]

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        upsert_pending_resume_func=upsert_pending_resume,
    )

    assert len(cognition_inputs) == 2
    assert len(pending_rows) == 1
    assert pending_rows[0]["action_kind"] == RESOLVER_PENDING_HIL_ACTION_KIND
    assert pending_rows[0]["source_kind"] == "cognitive_episode"
    assert pending_rows[0]["source_id"] == "resolver-capability-episode"
    assert pending_rows[0]["action_spec_schema_version"] == (
        RESOLVER_PENDING_RESUME_VERSION
    )
    assert pending_rows[0]["target_scope"] == {
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "global_user_id": "global-user-123",
        "source_message_id": "message-123",
    }
    pending = pending_rows[0]["execution_result"]["pending_resume"]
    assert pending["status"] == "waiting_for_user"
    assert pending["prompt_safe_original_goal"] == (
        "Original user request about trust."
    )
    assert pending["prompt_safe_question"] == "请只问用户所在城市。"
    observation = result["resolver_state"]["observations"][0]
    assert observation["pending_resume_id"] == pending["resume_id"]
    assert result["resolver_state"]["status"] == "waiting_for_user"
    assert result["action_specs"][0]["kind"] == "speak"


@pytest.mark.asyncio
async def test_hil_repeated_after_pending_surfaces_pending_question() -> None:
    """A repeated HIL request should still ask the persisted pending question."""

    request = _resolver_request(
        capability_kind="human_clarification",
        objective="请只问用户所在城市。",
    )
    pending_rows: list[dict] = []
    cognition_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        return _cognition_result(
            internal_monologue="仍然只想请求同一个澄清。",
            resolver_requests=[request],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_hil_repeat",
            "capability_kind": capability_request["capability_kind"],
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "blocked",
            "prompt_safe_summary": (
                "Human clarification required: 请只问用户所在城市。"
            ),
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    async def upsert_pending_resume(state: dict, observation: dict) -> dict:
        record = build_pending_resume_record(state, observation)
        pending_rows.append(record)
        return record["execution_result"]["pending_resume"]

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        upsert_pending_resume_func=upsert_pending_resume,
    )

    assert len(cognition_inputs) == 2
    assert len(pending_rows) == 1
    assert [spec["kind"] for spec in result["action_specs"]] == ["speak"]
    assert result["action_specs"][0]["surface_role"] == "task_status"
    pending = pending_rows[0]["execution_result"]["pending_resume"]
    expected_continuation_ref = build_goal_continuation_ref(
        source_episode_id="resolver-capability-episode",
        source_message_id=pending["source_message_id"],
        branch_id="resolver_pending_resume:human_clarification",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": pending["global_user_id"],
        },
    )
    assert result["action_specs"][0]["goal_continuation_ref"] == (
        expected_continuation_ref
    )
    action_spec = result["action_specs"][0]
    assert action_spec["cognition_provenance"] == {
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "global-user-123",
        }],
        "evidence_handles": [],
    }
    target_roles = l3_surface._surface_target_roles({
        "action_specs": [action_spec],
    })
    addressee_plan = l3_surface._surface_addressee_plan(
        target_roles,
        state=result,
    )
    assert addressee_plan == [{
        "handle": "current_user",
        "display_name": "Test User",
        "semantic_role": "direct_recipient",
        "wording_policy": "second_person_allowed",
    }]
    assert result["resolver_capability_requests"] == []
    surface_requirements = result["action_specs"][0]["params"][
        "surface_requirements"
    ]
    assert surface_requirements == {
        "decision": "ask_clarification",
        "detail": "请只问用户所在城市。",
    }
    assert result["resolver_state"]["terminal_reason"] == (
        "pending resume fallback surface after repeated capability"
    )


@pytest.mark.asyncio
async def test_hil_pending_without_action_surfaces_pending_question() -> None:
    """A created pending row must not disappear if L2d emits no action."""

    request = _resolver_request(
        capability_kind="human_clarification",
        objective="请只问用户所在城市。",
    )
    pending_rows: list[dict] = []
    cognition_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        if len(cognition_inputs) == 1:
            return _cognition_result(
                internal_monologue="第一轮：需要问用户城市。",
                resolver_requests=[request],
            )
        return _cognition_result(
            internal_monologue="第二轮：没有正确外部化 pending。",
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_hil_no_action",
            "capability_kind": capability_request["capability_kind"],
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "blocked",
            "prompt_safe_summary": (
                "Human clarification required: 请只问用户所在城市。"
            ),
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    async def upsert_pending_resume(state: dict, observation: dict) -> dict:
        record = build_pending_resume_record(state, observation)
        pending_rows.append(record)
        return record["execution_result"]["pending_resume"]

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        upsert_pending_resume_func=upsert_pending_resume,
    )

    assert len(cognition_inputs) == 2
    assert len(pending_rows) == 1
    assert result["resolver_capability_requests"] == []
    assert [spec["kind"] for spec in result["action_specs"]] == ["speak"]
    surface_requirements = result["action_specs"][0]["params"][
        "surface_requirements"
    ]
    assert surface_requirements == {
        "decision": "ask_clarification",
        "detail": "请只问用户所在城市。",
    }
    assert result["resolver_state"]["terminal_reason"] == (
        "pending resume fallback surface completed"
    )


@pytest.mark.asyncio
async def test_same_message_pending_resolution_is_ignored() -> None:
    """A newly created pending row cannot be resolved by its source message."""

    request = _resolver_request(
        capability_kind="human_clarification",
        objective="请只问用户所在城市。",
    )
    applied: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        if "pending_resolver_resume" not in state:
            return _cognition_result(
                internal_monologue="第一轮：需要问用户城市。",
                resolver_requests=[request],
            )
        output = _cognition_result(
            internal_monologue="第二轮：应该只提出问题。",
        )
        output["resolver_pending_resolution"] = _pending_resolution(
            resume_id=state["pending_resolver_resume"]["resume_id"],
        )
        return output

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_hil_city",
            "capability_kind": capability_request["capability_kind"],
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "blocked",
            "prompt_safe_summary": (
                "Human clarification required: 请只问用户所在城市。"
            ),
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    async def upsert_pending(state: dict, observation: dict) -> dict:
        record = build_pending_resume_record(state, observation)
        return record["execution_result"]["pending_resume"]

    async def apply_resolution(_state: dict, resolution: dict) -> None:
        applied.append(dict(resolution))

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        upsert_pending_resume_func=upsert_pending,
        apply_pending_resolution_func=apply_resolution,
    )

    assert applied == []
    assert "resolver_pending_resolution" not in result
    assert result["resolver_state"]["pending_resume"]["status"] == (
        "waiting_for_user"
    )


@pytest.mark.asyncio
async def test_same_message_terminal_action_closes_pending_resolution() -> None:
    """A self-corrected final answer should not leave stale pending rows."""

    request = _resolver_request(
        capability_kind="human_clarification",
        objective="确认“这是”具体指代的对象。",
    )
    pending_rows: list[dict] = []
    applied_rows: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        if "pending_resolver_resume" not in state:
            return _cognition_result(
                internal_monologue="第一轮：误以为需要澄清指代。",
                resolver_requests=[request],
            )
        output = _cognition_result(
            internal_monologue="第二轮：发现同一句里已经给出对象。",
            action_specs=[_speak_action_spec("指代已明确，直接回应。")],
        )
        output["resolver_pending_resolution"] = _pending_resolution(
            resume_id=state["pending_resolver_resume"]["resume_id"],
        )
        return output

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_hil_referent",
            "capability_kind": capability_request["capability_kind"],
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "blocked",
            "prompt_safe_summary": (
                "Human clarification required: 确认“这是”具体指代的对象。"
            ),
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    async def upsert_pending(state: dict, observation: dict) -> dict:
        record = build_pending_resume_record(state, observation)
        pending_rows.append(record)
        return record["execution_result"]["pending_resume"]

    async def list_pending_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return list(pending_rows)

    async def persist_pending_row(row: dict) -> None:
        applied_rows.append(row)

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        upsert_pending_resume_func=upsert_pending,
        apply_pending_resolution_func=(
            lambda state, resolution: apply_pending_resolution(
                state,
                resolution,
                list_action_attempts_func=list_pending_rows,
                upsert_action_attempt_func=persist_pending_row,
            )
        ),
    )

    assert result["resolver_state"]["status"] == "terminal"
    assert result["resolver_state"]["pending_resume"]["status"] == "closed"
    assert result["action_specs"][0]["kind"] == "speak"
    assert applied_rows[-1]["status"] == "closed"


@pytest.mark.asyncio
async def test_resolver_telemetry_is_sanitized_and_stage_readable() -> None:
    """Resolver telemetry should expose stage values without raw ids or text."""

    result = await _hil_pending_trace_state()
    result["debug_secret_url"] = "https://example.invalid/callback?api_key=raw"
    trace = result["resolver_state"]["cycle_traces"][0]

    cycle_event = build_resolver_cycle_event(
        result,
        trace,
        duration_ms=6000,
    )
    terminal_event = build_resolver_terminal_event(result, duration_ms=200)
    event_json = json.dumps(
        {
            "cycle": cycle_event,
            "terminal": terminal_event,
        },
        ensure_ascii=False,
    )

    assert cycle_event["component"] == "nodes.cognition_resolver"
    assert cycle_event["event_kind"] == "resolver_cycle"
    assert cycle_event["labels"]["selected_capability_kind"] == (
        "human_clarification"
    )
    assert cycle_event["labels"]["observation_status"] == "blocked"
    assert cycle_event["labels"]["duration_label"] == "slow"
    assert terminal_event["event_kind"] == "resolver_terminal"
    assert terminal_event["metrics"]["cycle_count"] == 2
    assert terminal_event["labels"]["pending_resume_status"] == "waiting_for_user"
    assert terminal_event["labels"]["duration_label"] == "fast"
    assert "第一轮：缺少用户城市" in event_json
    assert "human_clarification" in event_json
    assert "blocked" in event_json
    assert "Need an evidence-backed answer" not in event_json
    assert "message-123" not in event_json
    assert "channel-123" not in event_json
    assert "global-user-123" not in event_json
    assert "platform-user-123" not in event_json
    assert "api_key=raw" not in event_json


@pytest.mark.asyncio
async def test_resolver_human_readable_trace_is_prompt_safe(tmp_path) -> None:
    """Local resolver traces should be readable without raw platform refs."""

    result = await _hil_pending_trace_state()

    trace_path = write_human_readable_resolver_trace(
        result,
        tmp_path,
        filename_stem="B04 HIL/raw ids",
    )
    trace_text = trace_path.read_text(encoding="utf-8")

    assert trace_path.parent == tmp_path
    assert trace_path.name == "B04_HIL_raw_ids.md"
    assert "# Cognition Resolver Trace" in trace_text
    assert "## Cycle 0" in trace_text
    assert "human_clarification" in trace_text
    assert "waiting_for_user" in trace_text
    assert "Need an evidence-backed answer" not in trace_text
    assert "message-123" not in trace_text
    assert "channel-123" not in trace_text
    assert "global-user-123" not in trace_text


@pytest.mark.asyncio
async def test_approval_blocked_observation_persists_pending_without_side_effect() -> None:
    """Approval blockers should persist approval state without executing effects."""

    request = _resolver_request(
        capability_kind="approval_preparation",
        objective="说明准备创建提醒，但等待用户确认。",
    )
    pending_rows: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        if "Approval required before side effects" not in state.get(
            "resolver_context",
            "",
        ):
            return _cognition_result(
                internal_monologue="第一轮：需要审批",
                resolver_requests=[request],
            )
        assert "pending_resolver_resume" in state["resolver_context"]
        return _cognition_result(
            internal_monologue="第二轮：解释审批状态",
            action_specs=[_speak_action_spec("说明准备做什么并等待确认。")],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_approval",
            "capability_kind": capability_request["capability_kind"],
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "blocked",
            "prompt_safe_summary": (
                "Approval required before side effects: "
                "说明准备创建提醒，但等待用户确认。"
            ),
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    async def upsert_pending_resume(state: dict, observation: dict) -> dict:
        record = build_pending_resume_record(state, observation)
        pending_rows.append(record)
        return record["execution_result"]["pending_resume"]

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        upsert_pending_resume_func=upsert_pending_resume,
    )

    assert len(pending_rows) == 1
    assert pending_rows[0]["action_kind"] == RESOLVER_PENDING_APPROVAL_ACTION_KIND
    pending = pending_rows[0]["execution_result"]["pending_resume"]
    assert pending["status"] == "waiting_for_approval"
    assert pending["prompt_safe_approval_summary"] == (
        "准备审批说明，范围只限原始目标：Original user request about trust.。"
        "当前尚未执行提醒、调度、发送、文件检查、状态检查、下载监控、"
        "恢复操作或其他副作用；继续前必须等待用户明确确认；"
        "不得把审批说明扩展成原始目标以外的外部执行能力。"
    )
    assert result["resolver_state"]["status"] == "waiting_for_approval"
    assert "action_results" not in result


@pytest.mark.asyncio
async def test_pending_resolution_is_applied_only_after_l2d_decision() -> None:
    """Follow-up turns should close pending rows only from L2d decisions."""

    applied: list[dict] = []
    state = ensure_initial_resolver_inputs(_resolver_state(), max_cycles=3)
    resolver_state = dict(state["resolver_state"])
    resolver_state["pending_resume"] = _pending_resume()
    state["resolver_state"] = resolver_state
    state["pending_resolver_resume"] = _pending_resume()
    state["resolver_context"] = project_resolver_context(resolver_state)
    cognition_inputs: list[dict] = []

    async def call_cognition(cognition_state: dict) -> dict:
        cognition_inputs.append(dict(cognition_state))
        assert "pending_resolver_resume" in cognition_state["resolver_context"]
        output = _cognition_result(
            internal_monologue="用户回答了上一轮澄清。",
            action_specs=[_speak_action_spec("根据用户回答继续。")],
        )
        output["resolver_pending_resolution"] = _pending_resolution()
        return output

    async def execute_capability(
        _request: dict,
        _state: dict,
    ) -> dict:
        raise AssertionError("no capability should run on resolved pending state")

    async def apply_resolution(state_with_resolution: dict, resolution: dict) -> None:
        applied.append({
            "state": dict(state_with_resolution),
            "resolution": dict(resolution),
        })

    result = await call_cognition_resolver_loop(
        state,
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        apply_pending_resolution_func=apply_resolution,
    )

    assert len(cognition_inputs) == 1
    assert applied[0]["resolution"]["decision"] == "answered"
    assert result["resolver_state"]["pending_resume"]["status"] == "closed"
    assert result["resolver_pending_resolution"]["decision"] == "answered"
    assert result["action_specs"][0]["kind"] == "speak"


@pytest.mark.asyncio
async def test_hil_follow_up_can_continue_original_goal_after_answer() -> None:
    """A resolved HIL row should allow the original goal to continue."""

    original_goal = (
        "背景要求：奥克兰海滨散步加简餐；"
        "最终必须包含 EXACT-FINAL-MARKER: AUCKLAND-PLAN-COMPLETE。"
    )
    clarification_request = _resolver_request(
        capability_kind="human_clarification",
        objective="请只问用户所在城市。",
    )
    evidence_request = _resolver_request(
        objective="根据用户补充的城市继续生成今晚计划需要的证据。",
    )
    goal_progress = _goal_progress(
        focus="先取得城市，再完成带背景与最终标记的计划证据。",
    )
    goal_progress["original_goal"] = original_goal
    goal_progress["deliverables"][0]["description"] = "背景要求：海滨散步与简餐"
    goal_progress["final_response_requirements"] = [
        "EXACT-FINAL-MARKER: AUCKLAND-PLAN-COMPLETE",
    ]
    pending_rows: list[dict] = []
    applied_rows: list[dict] = []
    first_turn_production_inputs: list[dict] = []

    async def first_turn_cognition(state: dict) -> dict:
        first_turn_production_inputs.append(
            build_cognition_input_from_global_state(state),
        )
        if "pending_resolver_resume" not in state["resolver_context"]:
            output = _cognition_result(
                internal_monologue="第一轮：缺少城市，必须先问用户。",
                resolver_requests=[clarification_request],
            )
            output["resolver_goal_progress"] = goal_progress
            output["pending_task_continuation"] = {
                "schema_version": PENDING_TASK_CONTINUATION_VERSION,
                "on_answered_clarification": "background_task_admission",
            }
            return output
        return _cognition_result(
            internal_monologue="第一轮：已经形成最小澄清问题。",
            action_specs=[_speak_action_spec("只问城市。")],
        )

    async def first_turn_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_hil_city",
            "capability_kind": capability_request["capability_kind"],
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "blocked",
            "prompt_safe_summary": (
                "Human clarification required: 请只问用户所在城市。"
            ),
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    async def upsert_pending(state: dict, observation: dict) -> dict:
        record = build_pending_resume_record(state, observation)
        pending_rows.append(record)
        return record["execution_result"]["pending_resume"]

    first_result = await call_cognition_resolver_loop(
        _production_resolver_state(original_goal=original_goal),
        call_cognition_subgraph_func=first_turn_cognition,
        execute_capability_func=first_turn_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        upsert_pending_resume_func=upsert_pending,
    )

    assert first_result["resolver_state"]["status"] == "waiting_for_user"
    assert len(pending_rows) == 1
    assert len(first_turn_production_inputs) == 2
    assert first_turn_production_inputs[1]["resolver_goal_progress"][
        "original_goal"
    ] == original_goal
    assert first_turn_production_inputs[1]["resolver_goal_progress"][
        "final_response_requirements"
    ] == ["EXACT-FINAL-MARKER: AUCKLAND-PLAN-COMPLETE"]
    assert pending_rows[0]["resolver_pending_resume"]["prompt_safe_goal_progress"] == (
        goal_progress
    )
    assert pending_rows[0]["resolver_pending_resume"]["pending_task_continuation"] == {
        "schema_version": PENDING_TASK_CONTINUATION_VERSION,
        "on_answered_clarification": "background_task_admission",
    }

    follow_up_state = ensure_initial_resolver_inputs(
        _production_resolver_state(original_goal=original_goal),
        max_cycles=3,
    )
    follow_up_state["platform_message_id"] = "message-follow-up-123"
    follow_up_state["reply_context"] = {
        "reply_to_message_id": "message-123",
    }

    async def list_pending_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return list(pending_rows)

    async def persist_pending_row(row: dict) -> None:
        applied_rows.append(row)

    follow_up_state = await load_matching_pending_resume_into_state(
        follow_up_state,
        list_action_attempts_func=list_pending_rows,
        upsert_action_attempt_func=persist_pending_row,
    )
    assert "pending_resolver_resume" in follow_up_state
    follow_up_inputs: list[dict] = []
    follow_up_production_inputs: list[dict] = []
    executed_capability_requests: list[dict] = []

    async def follow_up_cognition(state: dict) -> dict:
        follow_up_inputs.append(dict(state))
        follow_up_production_inputs.append(
            build_cognition_input_from_global_state(state),
        )
        if len(follow_up_inputs) == 1:
            output = _cognition_result(
                internal_monologue="第二轮：用户回答了城市，继续原始目标。",
                resolver_requests=[evidence_request],
            )
            output["resolver_pending_resolution"] = _pending_resolution(
                resume_id=follow_up_state["pending_resolver_resume"][
                    "resume_id"
                ],
            )
            return output
        return _cognition_result(
            internal_monologue="第三轮：证据足够，回答原始计划问题。",
            action_specs=[_speak_action_spec("给出今晚轻松计划。")],
        )

    async def follow_up_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        executed_capability_requests.append(dict(capability_request))
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_city_plan",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "succeeded",
            "prompt_safe_summary": "已获得奥克兰轻松晚间计划证据。",
            "rag_result": {
                "answer": "奥克兰今晚可以低预算散步加简餐。",
                "memory_evidence": [],
                "recall_evidence": [],
                "conversation_evidence": [],
                "external_evidence": [],
                "third_party_profiles": [],
                "user_image": {},
                "character_image": {},
                "supervisor_trace": {
                    "loop_count": 1,
                    "unknown_slots": [],
                    "dispatched": [],
                },
            },
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "missing",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    follow_up_result = await call_cognition_resolver_loop(
        follow_up_state,
        call_cognition_subgraph_func=follow_up_cognition,
        execute_capability_func=follow_up_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        apply_pending_resolution_func=(
            lambda state, resolution: apply_pending_resolution(
                state,
                resolution,
                list_action_attempts_func=list_pending_rows,
                upsert_action_attempt_func=persist_pending_row,
            )
        ),
    )

    assert len(follow_up_inputs) == 2
    assert follow_up_production_inputs[0]["pending_resolver_continuation"][
        "original_goal"
    ] == original_goal
    assert follow_up_production_inputs[0]["response_plan_contract_variant"] == (
        "open_pending_resolution"
    )
    assert follow_up_production_inputs[0]["pending_resolver_continuation"][
        "pending_task_continuation"
    ] == {
        "schema_version": PENDING_TASK_CONTINUATION_VERSION,
        "on_answered_clarification": "background_task_admission",
    }
    assert follow_up_production_inputs[0]["resolver_goal_progress"][
        "original_goal"
    ] == original_goal
    assert follow_up_production_inputs[1]["resolver_goal_progress"][
        "original_goal"
    ] == original_goal
    assert follow_up_production_inputs[1]["resolver_goal_progress"]["deliverables"][
        0
    ]["description"] == "背景要求：海滨散步与简餐"
    assert follow_up_production_inputs[1]["resolver_goal_progress"][
        "final_response_requirements"
    ] == ["EXACT-FINAL-MARKER: AUCKLAND-PLAN-COMPLETE"]
    assert "pending_resolver_continuation" not in follow_up_production_inputs[1]
    assert follow_up_production_inputs[1]["response_plan_contract_variant"] == (
        "post_pending_resolution"
    )
    post_answer_input = follow_up_production_inputs[1]
    post_answer_workspace = build_canonical_turn_workspace(
        episode=post_answer_input["episode"],
        scene_context=post_answer_input["scene_context"],
        evidence=post_answer_input["evidence"],
        mutable_state=post_answer_input["mutable_state"],
        character_constraints=post_answer_input["character_constraints"],
        identity_context=post_answer_input["character_identity_context"],
        continuity={
            "private": post_answer_input.get("private_continuity_context", ""),
            "dialog": post_answer_input.get("past_dialog_cognition_context", ""),
        },
        available_actions=post_answer_input["available_actions"],
        available_resolvers=post_answer_input[
            "available_resolver_capabilities"
        ],
        overused_moves=post_answer_input["overused_moves"],
        direct_facts=post_answer_input.get("direct_facts", []),
        character_operational_context=post_answer_input.get(
            "character_operational_context", {}
        ),
        character_affect_context=post_answer_input.get(
            "character_affect_context", []
        ),
        relationship_context=post_answer_input.get("relationship_context", {}),
        resolver_context=post_answer_input.get("resolver_context", ""),
        resolver_progress=post_answer_input.get("resolver_goal_progress", {}),
        runtime_limits=post_answer_input.get("runtime_capability_limits", []),
        group_engagement=post_answer_input.get(
            "group_engagement_action_context", {}
        ),
        response_plan_contract_variant=post_answer_input[
            "response_plan_contract_variant"
        ],
    )
    post_answer_packet = build_canonical_plan_question(
        workspace=post_answer_workspace,
        goal={
            "goal_kind": "bounded_current_goal",
            "intent": "complete the resolved evidence task",
            "reason": "the task observation is now available",
            "cause_summary": "the resolver observation",
        },
        appraisal_summary=[],
    )
    assert "response_plan_contract_variant" not in post_answer_packet[
        "output_contract"
    ]
    assert "pending_task_continuation" not in post_answer_packet[
        "output_contract"
    ]
    assert "pending_resolution_fields" not in post_answer_packet["output_contract"]
    assert all(
        row["capability"] not in {
            "human_clarification",
            "task_resolution_request",
        }
        for row in post_answer_packet["capabilities"]["resolvers"]
    )
    assert [
        request["capability_kind"]
        for request in executed_capability_requests
    ] == ["task_resolution_request"]
    assert follow_up_result["resolver_pending_resolution"]["decision"] == (
        "answered"
    )
    assert follow_up_result["resolver_state"]["pending_resume"]["status"] == (
        "closed"
    )
    assert follow_up_result["rag_result"]["answer"] == (
        "奥克兰今晚可以低预算散步加简餐。"
    )
    assert follow_up_result["action_specs"][0]["kind"] == "speak"
    assert applied_rows[-1]["status"] == "closed"


@pytest.mark.asyncio
async def test_pending_helpers_load_and_close_matching_pending_rows() -> None:
    """Pending helper should filter by scope, expiry, and L2d resolution."""

    state = _resolver_state()
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_hil_city",
        "capability_kind": "human_clarification",
        "request_objective": "你现在在哪个城市？",
        "request_reason": "缺少用户所在地。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 你现在在哪个城市？",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    record = build_pending_resume_record(state, observation)
    expired_record = build_pending_resume_record(
        state,
        observation,
        expires_at_utc="2026-05-29T20:00:00+00:00",
    )
    expired_record["attempt_id"] = "expired"
    expired_record["execution_result"]["pending_resume"]["resume_id"] = "expired"
    expired_record["resolver_pending_resume"]["resume_id"] = "expired"
    rows = [expired_record, record]
    upserted: list[dict] = []

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return list(rows)

    async def upsert_row(row: dict) -> None:
        upserted.append(row)

    follow_up_state = dict(state)
    follow_up_state["platform_message_id"] = "follow-up-message-id"
    follow_up_state["storage_timestamp_utc"] = "2026-05-29T21:05:00+00:00"
    follow_up_state["reply_context"] = {
        "reply_to_message_id": "message-123",
    }

    loaded = await load_matching_pending_resume(
        follow_up_state,
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    assert loaded["resume_id"] == (
        record["execution_result"]["pending_resume"]["resume_id"]
    )
    assert upserted[0]["status"] == "expired"

    resume_id = record["execution_result"]["pending_resume"]["resume_id"]
    await apply_pending_resolution(
        follow_up_state,
        _pending_resolution(resume_id=resume_id),
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    assert upserted[-1]["status"] == "closed"
    assert upserted[-1]["execution_result"]["pending_resolution"]["decision"] == (
        "answered"
    )
    assert upserted[-1]["execution_result"][
        "resolver_pending_resolution"
    ]["decision"] == "answered"

    approval_observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_approval",
        "capability_kind": "approval_preparation",
        "request_objective": "准备创建提醒，但需要用户确认。",
        "request_reason": "侧效应需要确认。",
        "status": "blocked",
        "prompt_safe_summary": "Approval required: 准备创建提醒。",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    approval_record = build_pending_resume_record(state, approval_observation)
    rows = [approval_record]
    upserted.clear()
    approval_resume_id = approval_record["resolver_pending_resume"]["resume_id"]

    await apply_pending_resolution(
        follow_up_state,
        _pending_resolution(
            decision="approved",
            resume_id=approval_resume_id,
        ),
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    assert upserted[-1]["status"] == "closed"
    assert upserted[-1]["execution_result"][
        "resolver_pending_resolution"
    ]["decision"] == "approved"

    superseded_record = build_pending_resume_record(state, observation)
    rows = [superseded_record]
    upserted.clear()
    superseded_resume_id = superseded_record["resolver_pending_resume"][
        "resume_id"
    ]
    await apply_pending_resolution(
        follow_up_state,
        _pending_resolution(
            decision="superseded",
            resume_id=superseded_resume_id,
        ),
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    assert upserted[-1]["status"] == "superseded"
    assert upserted[-1]["execution_result"][
        "resolver_pending_resolution"
    ]["decision"] == "superseded"


@pytest.mark.asyncio
async def test_superseded_pending_row_leaves_the_next_turn_unbound() -> None:
    """A superseded clarification cannot constrain fresh normal cognition."""

    state = _resolver_state()
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_superseded_pending",
        "capability_kind": "human_clarification",
        "request_objective": "请补充原始任务所需的用户事实。",
        "request_reason": "缺少一个用户控制的事实。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required.",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    record = build_pending_resume_record(state, observation)
    rows = [record]

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return list(rows)

    async def upsert_row(row: dict) -> None:
        rows[:] = [row]

    resume_id = record["resolver_pending_resume"]["resume_id"]
    updated = await apply_pending_resolution(
        state,
        _pending_resolution(decision="superseded", resume_id=resume_id),
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    assert updated is not None
    assert updated["status"] == "superseded"
    fresh_state = ensure_initial_resolver_inputs(_resolver_state(), max_cycles=3)
    fresh_state["platform_message_id"] = "fresh-message-after-supersession"
    fresh_state = await load_matching_pending_resume_into_state(
        fresh_state,
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )
    assert "pending_resolver_resume" not in fresh_state
    fresh_production_state = ensure_initial_resolver_inputs(
        _production_resolver_state(
            original_goal="Handle the replacement request as a fresh task.",
        ),
        max_cycles=3,
    )
    fresh_input = build_cognition_input_from_global_state(
        fresh_production_state,
    )
    assert fresh_input["response_plan_contract_variant"] == "fresh_ordinary"


@pytest.mark.asyncio
async def test_pending_loader_ignores_future_pending_rows() -> None:
    """Replay or delayed turns must not load pending rows from the future."""

    current_state = _resolver_state()
    current_state["storage_timestamp_utc"] = "2026-05-29T21:00:00+00:00"
    future_state = dict(current_state)
    future_state["storage_timestamp_utc"] = "2026-05-30T21:00:00+00:00"
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_future_hil",
        "capability_kind": "human_clarification",
        "request_objective": "确认未来消息里的对象。",
        "request_reason": "缺少未来消息中的指代对象。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 确认未来消息里的对象。",
        "evidence_refs": [],
        "created_at_utc": "2026-05-30T21:00:00+00:00",
    }
    future_record = build_pending_resume_record(future_state, observation)

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return [future_record]

    loaded = await load_matching_pending_resume(
        current_state,
        list_action_attempts_func=list_rows,
    )

    assert loaded is None


@pytest.mark.asyncio
async def test_pending_loader_ignores_same_source_message_rows() -> None:
    """A source message should not resume the pending row it created."""

    state = _resolver_state()
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_same_message_hil",
        "capability_kind": "human_clarification",
        "request_objective": "确认当前消息里的对象。",
        "request_reason": "缺少当前消息中的指代对象。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 确认当前消息里的对象。",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    same_message_record = build_pending_resume_record(state, observation)

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return [same_message_record]

    loaded = await load_matching_pending_resume(
        state,
        list_action_attempts_func=list_rows,
    )

    assert loaded is None


@pytest.mark.asyncio
async def test_pending_loader_selects_unrelated_same_scope_candidate() -> None:
    """Cognition decides whether an ordinary same-scope turn answers pending."""

    state = _resolver_state()
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_unrelated_hil",
        "capability_kind": "human_clarification",
        "request_objective": "确认原始计划所需的用户信息。",
        "request_reason": "原始计划缺少用户输入。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 确认原始计划所需的用户信息。",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    pending_record = build_pending_resume_record(state, observation)
    unrelated_state = dict(state)
    unrelated_state["platform_message_id"] = "unrelated-message-456"
    unrelated_state["decontextualized_input"] = "我今天心情变好了，谢谢你陪我说话。"

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return [pending_record]

    loaded = await load_matching_pending_resume(
        unrelated_state,
        list_action_attempts_func=list_rows,
    )

    assert loaded is not None
    assert loaded["resume_id"] == pending_record["resolver_pending_resume"][
        "resume_id"
    ]


@pytest.mark.asyncio
async def test_pending_loader_selects_newest_candidate_without_reply_metadata() -> None:
    """Ordinary adjacency selects the newest exact-scope clarification row."""

    state = _resolver_state()
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_newest_hil",
        "capability_kind": "human_clarification",
        "request_objective": "确认当前计划缺少的用户事实。",
        "request_reason": "原始目标仍缺少用户控制的信息。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 缺少用户事实。",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    older_state = dict(state)
    older_state["storage_timestamp_utc"] = "2026-05-29T20:00:00+00:00"
    newer_state = dict(state)
    newer_state["storage_timestamp_utc"] = "2026-05-29T20:30:00+00:00"
    older_record = build_pending_resume_record(older_state, observation)
    newer_record = build_pending_resume_record(newer_state, observation)
    follow_up_state = dict(state)
    follow_up_state["storage_timestamp_utc"] = "2026-05-29T21:00:00+00:00"
    follow_up_state["platform_message_id"] = "ordinary-follow-up-456"
    follow_up_state["reply_context"] = {}

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return [older_record, newer_record]

    loaded = await load_matching_pending_resume(
        follow_up_state,
        list_action_attempts_func=list_rows,
    )

    assert loaded is not None
    assert loaded["resume_id"] == newer_record["resolver_pending_resume"][
        "resume_id"
    ]


@pytest.mark.asyncio
async def test_pending_unrelated_turn_can_continue_waiting_without_task_admission() -> None:
    """Cognition can keep an unrelated pending clarification open."""

    state = ensure_initial_resolver_inputs(_resolver_state(), max_cycles=3)
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_waiting_hil",
        "capability_kind": "human_clarification",
        "request_objective": "确认原始目标需要的用户事实。",
        "request_reason": "该事实仍未由用户提供。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 需要用户事实。",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    record = build_pending_resume_record(state, observation)
    follow_up_state = dict(state)
    follow_up_state["platform_message_id"] = "ordinary-follow-up-waiting"
    follow_up_state["decontextualized_input"] = "我今天只是想聊聊天。"
    follow_up_state["reply_context"] = {}
    pending_rows = [record]
    applied: list[dict] = []

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return list(pending_rows)

    async def apply_resolution(state_with_resolution: dict, resolution: dict) -> None:
        del state_with_resolution
        applied.append(dict(resolution))

    follow_up_state = await load_matching_pending_resume_into_state(
        follow_up_state,
        list_action_attempts_func=list_rows,
    )
    resume_id = follow_up_state["pending_resolver_resume"]["resume_id"]

    async def call_cognition(current_state: dict) -> dict:
        assert current_state["pending_resolver_resume"]["resume_id"] == resume_id
        result = _cognition_result(
            internal_monologue="当前消息没有回答原澄清，继续等待。",
            action_specs=[_speak_action_spec("先保持原澄清事项等待。")],
            goal_resolution="answerable_now",
        )
        result["resolver_pending_resolution"] = _pending_resolution(
            decision="continue_waiting",
            resume_id=resume_id,
        )
        return result

    async def execute_capability(_request: dict, _state: dict) -> dict:
        raise AssertionError("continue_waiting must not admit a task")

    result = await call_cognition_resolver_loop(
        follow_up_state,
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        apply_pending_resolution_func=apply_resolution,
    )

    assert applied[0]["decision"] == "continue_waiting"
    assert result["resolver_pending_resolution"]["resume_id"] == resume_id
    assert result["resolver_state"]["pending_resume"]["status"] == (
        "waiting_for_user"
    )
    assert result["resolver_capability_requests"] == []


@pytest.mark.asyncio
async def test_pending_resume_load_restores_original_goal_progress() -> None:
    """HIL follow-up turns should inherit the first-turn deliverable checklist."""

    state = ensure_initial_resolver_inputs(_resolver_state(), max_cycles=3)
    resolver_state = dict(state["resolver_state"])
    resolver_state["goal_progress"] = _goal_progress()
    state["resolver_state"] = resolver_state
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": "resolver_obs_hil_city",
        "capability_kind": "human_clarification",
        "request_objective": "你在奥克兰哪个区域？",
        "request_reason": "缺少用户所在地。",
        "status": "blocked",
        "prompt_safe_summary": "Human clarification required: 你在奥克兰哪个区域？",
        "evidence_refs": [],
        "created_at_utc": "2026-05-29T21:00:00+00:00",
    }
    record = build_pending_resume_record(state, observation)
    follow_up_state = _resolver_state()
    follow_up_state["platform_message_id"] = "message-456"
    follow_up_state["decontextualized_input"] = "就在奥克兰 CBD。"
    follow_up_state["reply_context"] = {}
    follow_up_state = ensure_initial_resolver_inputs(
        follow_up_state,
        max_cycles=3,
    )

    async def list_rows(*, limit: int = 1000) -> list[dict]:
        del limit
        return [record]

    async def upsert_row(row: dict) -> None:
        del row

    loaded_state = await load_matching_pending_resume_into_state(
        follow_up_state,
        list_action_attempts_func=list_rows,
        upsert_action_attempt_func=upsert_row,
    )

    loaded_progress = loaded_state["resolver_state"]["goal_progress"]
    assert loaded_progress["original_goal"] == (
        "今晚安排一个两小时低预算计划。"
    )
    assert loaded_progress["deliverables"][1]["description"] == (
        "两小时散步路线和时间切分"
    )
    assert loaded_state["resolver_state"]["original_decontextualized_input"] == (
        "今晚安排一个两小时低预算计划。"
    )
    assert "就在奥克兰 CBD" not in loaded_state["resolver_context"]


@pytest.mark.asyncio
async def test_task_resolution_uses_objective_and_preserves_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unified task resolution receives the semantic goal and trusted context."""

    captured: dict = {}

    async def resolve_task_inline(
        request: dict,
        context: dict,
        *,
        inline_budget_seconds: float,
    ) -> dict[str, object]:
        captured["request"] = request
        captured["context"] = context
        captured["inline_budget_seconds"] = inline_budget_seconds
        return _task_result(summary="存在一条信任相关记忆。")

    monkeypatch.setattr(
        capabilities_module,
        "resolve_task_inline",
        resolve_task_inline,
    )
    request = _resolver_request()

    observation = await capabilities_module.execute_resolver_capability_request(
        request,
        _resolver_state(),
    )

    assert captured["request"]["capability"] == "task_resolution_request"
    assert captured["request"]["semantic_goal"] == request["objective"]
    assert captured["request"]["reason"] == request["reason"]
    assert captured["context"]["conversation_summary"] == (
        "Original user request about trust."
    )
    assert captured["context"]["prompt_message_context"]["body_text"] == (
        "Need an evidence-backed answer."
    )
    assert captured["context"]["conversation_progress"]["current_thread"] == (
        "trust question"
    )
    assert observation["status"] == "succeeded"
    assert observation["capability_kind"] == "task_resolution_request"
    assert observation["request_objective"] == request["objective"]
    assert observation["request_reason"] == request["reason"]
    assert observation["prompt_safe_summary"] == "存在一条信任相关记忆。"
    assert observation["evidence_refs"][0]["owner"] == "dsh"
    assert observation["evidence_refs"][0]["evidence_id"] == "evidence-1"
    assert observation["evidence_refs"][0]["excerpt"] == (
        "存在一条信任相关记忆。"
    )


@pytest.mark.asyncio
async def test_task_resolution_bounds_history_to_its_context_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task resolution receives the newest eight interaction turns."""

    captured: dict = {}

    async def resolve_task_inline(
        request: dict,
        context: dict,
        *,
        inline_budget_seconds: float,
    ) -> dict[str, object]:
        captured["context"] = context
        return _task_result(summary="Bounded history accepted.")

    monkeypatch.setattr(
        capabilities_module,
        "resolve_task_inline",
        resolve_task_inline,
    )
    state = _resolver_state()
    history = [
        {"role": "user", "body_text": f"turn-{index}"}
        for index in range(10)
    ]
    state["chat_history_recent"] = history
    state["chat_history_wide"] = history

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(),
        state,
    )

    expected_history = history[-8:]
    assert captured["context"]["chat_history_recent"] == expected_history
    assert captured["context"]["chat_history_wide"] == expected_history
    assert observation["status"] == "succeeded"


@pytest.mark.asyncio
async def test_task_resolution_bounds_decontextualized_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Result-ready text is bounded before task-resolution validation."""

    captured: dict = {}

    async def resolve_task_inline(
        request: dict,
        context: dict,
        *,
        inline_budget_seconds: float,
    ) -> dict[str, object]:
        del request, inline_budget_seconds
        captured["context"] = context
        return _task_result(summary="Bounded summary accepted.")

    monkeypatch.setattr(
        capabilities_module,
        "resolve_task_inline",
        resolve_task_inline,
    )
    state = _resolver_state()
    state["decontextualized_input"] = "A" * 1300

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(),
        state,
    )

    assert captured["context"]["conversation_summary"] == "A" * 1200
    assert observation["status"] == "succeeded"


@pytest.mark.asyncio
async def test_task_resolution_user_input_result_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A semantic user-input result becomes a typed resolver blocker."""

    resolve_task_inline = AsyncMock(return_value=_task_result(
        status="needs_user_input",
        summary="需要用户补充缺失指代。",
    ))
    monkeypatch.setattr(
        capabilities_module,
        "resolve_task_inline",
        resolve_task_inline,
    )

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(),
        _resolver_state(),
    )

    assert observation["status"] == "blocked"
    assert observation["blocker_kind"] == "requires_user_input"
    assert observation["prompt_safe_summary"] == "需要用户补充缺失指代。"
    resolve_task_inline.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_input_blocker_converges_after_one_final_cognition() -> None:
    """A blocked local recall cannot cause repeated resolver cognition."""

    request = _resolver_request(
        objective="retrieve the missing referent context",
    )
    cognition_inputs: list[dict] = []
    capability_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        return _cognition_result(
            internal_monologue="The referent is still missing.",
            resolver_requests=[request],
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        capability_inputs.append(capability_request)
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_missing_referent",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "blocked",
            "blocker_kind": "requires_user_input",
            "prompt_safe_summary": (
                "Local context recall requires user input: missing referent."
            ),
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    assert len(cognition_inputs) == 2
    assert len(capability_inputs) == 1
    assert result["resolver_capability_requests"] == []
    assert result["action_specs"][0]["params"]["surface_requirements"] == {
        "decision": "ask_clarification",
        "detail": "Local context recall requires user input: missing referent.",
    }
    assert result["action_specs"][0]["cognition_provenance"] == {
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "global-user-123",
        }],
        "evidence_handles": [],
    }
    assert result["resolver_state"]["status"] == "blocked"
    assert result["resolver_state"]["terminal_reason"] == (
        "blocked user-input resolver request converted to clarification surface"
    )


@pytest.mark.asyncio
async def test_user_input_blocker_without_final_action_surfaces_clarification() -> None:
    """A silent final cognition pass cannot suppress a needed clarification."""

    request = _resolver_request(
        objective="retrieve the missing referent context",
    )
    cognition_inputs: list[dict] = []

    async def call_cognition(state: dict) -> dict:
        cognition_inputs.append(dict(state))
        if len(cognition_inputs) == 1:
            return _cognition_result(
                internal_monologue="The referent is still missing.",
                resolver_requests=[request],
            )
        return _cognition_result(
            internal_monologue="The final pass selected no visible action.",
        )

    async def execute_capability(
        capability_request: dict,
        _state: dict,
    ) -> dict:
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_missing_referent_no_action",
            "capability_kind": capability_request["capability_kind"],
            **_task_observation_fields(capability_request),
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "blocked",
            "blocker_kind": "requires_user_input",
            "prompt_safe_summary": (
                "Local context recall requires user input: missing referent."
            ),
            "evidence_refs": [],
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "blocked",
                "remaining_needs": [],
            },
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        }

    result = await call_cognition_resolver_loop(
        _resolver_state(),
        call_cognition_subgraph_func=call_cognition,
        execute_capability_func=execute_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
    )

    assert len(cognition_inputs) == 2
    assert result["resolver_capability_requests"] == []
    assert result["action_specs"][0]["params"]["surface_requirements"] == {
        "decision": "ask_clarification",
        "detail": "Local context recall requires user input: missing referent.",
    }
    assert result["resolver_state"]["terminal_reason"] == (
        "blocked user-input resolver request converted to clarification surface"
    )


@pytest.mark.asyncio
async def test_internal_thought_uses_unified_task_resolution_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Internal-thought evidence work uses the same task-resolution boundary."""

    captured: dict = {}

    async def resolve_task_inline(
        request: dict,
        context: dict,
        *,
        inline_budget_seconds: float,
    ) -> dict[str, object]:
        captured["request"] = request
        captured["context"] = context
        captured["inline_budget_seconds"] = inline_budget_seconds
        return _task_result(
            summary="前文提到同一个话题。",
            semantic_objective=request["semantic_goal"],
        )

    monkeypatch.setattr(
        capabilities_module,
        "resolve_task_inline",
        resolve_task_inline,
    )
    request = _resolver_request(
        objective="回看群聊前文，判断这个内部想法是否有足够证据。",
    )

    observation = await capabilities_module.execute_resolver_capability_request(
        request,
        _internal_thought_resolver_state(),
    )

    assert captured["request"]["semantic_goal"] == request["objective"]
    assert captured["request"]["capability"] == "task_resolution_request"
    assert captured["context"]["conversation_summary"] == (
        "Original user request about trust."
    )
    assert captured["context"]["platform"] == "debug"
    assert captured["context"]["requester_global_user_id"] == (
        "global-user-123"
    )
    assert captured["context"]["character_name"] == "Kazusa"
    assert captured["context"]["prompt_message_context"]["body_text"] == (
        "Need an evidence-backed answer."
    )
    assert observation["status"] == "succeeded"
    assert observation["capability_kind"] == "task_resolution_request"
    assert observation["prompt_safe_summary"] == "前文提到同一个话题。"


@pytest.mark.asyncio
async def test_public_evidence_projects_through_task_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public answer research should run the complex resolver boundary."""

    captured: dict = {}

    async def resolve_complex_task(request: dict, context: dict, options=None) -> dict:
        captured["request"] = request
        captured["context"] = context
        captured["options"] = options
        result = {
            "schema_version": "complex_task_resolution_packet.v1",
            "root_question": "查询当前公共网页事实。",
            "investigation_summary": "找到一条网页证据。",
            "knowledge_we_know_so_far": ["网页证据显示当前事实可用。"],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [
                "由 cognition 判断是否足够进入可见回答。",
            ],
            "evidence_boundary_notes": [],
            "graph": {
                "schema_version": "complex_task_graph.v1",
                "root_node_id": "root",
                "active_node_id": "root",
                "nodes": {
                    "root": {
                        "schema_version": "complex_task_node.v1",
                        "node_id": "root",
                        "parent_id": None,
                        "depth": 0,
                        "objective": "查询当前公共网页事实。",
                        "node_kind": "root",
                        "status": "resolved",
                        "children": [],
                        "investigation_summary": "找到一条网页证据。",
                        "knowledge_we_know_so_far": [
                            "网页证据显示当前事实可用。",
                        ],
                        "knowledge_still_lacking": [],
                        "recommended_next_iteration": [
                            "由 cognition 判断是否足够进入可见回答。",
                        ],
                        "evidence_boundary_notes": [],
                        "evidence_refs": [],
                        "source_observation_ids": [],
                        "collapsed_into": None,
                        "attempts": [],
                    },
                },
                "collapse_events": [],
                "traversal_order": ["root"],
                "max_nodes": 8,
                "max_depth": 3,
            },
            "trace_summary": {
                "iterations": 1,
                "nodes_resolved": 1,
                "nodes_blocked": 0,
                "nodes_pending": 0,
                "subagent_calls": [],
                "failure_stage": "",
            },
            "evidence_refs": [
                {
                    "schema_version": "evidence_ref.v1",
                    "evidence_kind": "tool_result",
                    "evidence_id": "complex-task-root",
                    "owner": "complex_task_resolver",
                    "excerpt": "网页证据显示当前事实可用。",
                    "observed_at": "2026-05-30T00:00:00+00:00",
                },
            ],
        }
        return result

    async def resolve_task_inline(
        request: dict,
        context: dict,
        *,
        inline_budget_seconds: float,
    ) -> dict[str, object]:
        captured["request"] = request
        captured["context"] = context
        captured["inline_budget_seconds"] = inline_budget_seconds
        return _task_result(
            specialist="dsh",
            summary="网页证据显示当前事实可用。",
            semantic_objective=request["semantic_goal"],
        )

    monkeypatch.setattr(
        capabilities_module,
        "resolve_task_inline",
        resolve_task_inline,
    )
    request = _resolver_request(
        capability_kind="task_resolution_request",
        objective="查询当前公共网页事实。",
    )

    observation = await capabilities_module.execute_resolver_capability_request(
        request,
        _resolver_state(),
    )

    assert captured["request"]["semantic_goal"] == request["objective"]
    assert captured["request"]["capability"] == "task_resolution_request"
    assert captured["context"]["conversation_summary"] == (
        "Original user request about trust."
    )
    assert observation["status"] == "succeeded"
    assert observation["capability_kind"] == "task_resolution_request"
    assert observation["prompt_safe_summary"] == (
        "网页证据显示当前事实可用。"
    )
    assert observation["knowledge_projection"] == {
        "investigation_summary": "网页证据显示当前事实可用。",
        "knowledge_we_know_so_far": ["网页证据显示当前事实可用。"],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }
    assert observation["evidence_refs"][0]["owner"] == "dsh"


@pytest.mark.asyncio
async def test_empty_resolver_objective_fails_before_task_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed resolver requests fail before unified task dispatch."""

    resolve_task_inline = AsyncMock()
    monkeypatch.setattr(
        capabilities_module,
        "resolve_task_inline",
        resolve_task_inline,
    )
    request = _resolver_request(objective=" ")

    with pytest.raises(ResolverValidationError, match="objective"):
        await capabilities_module.execute_resolver_capability_request(
            request,
            _resolver_state(),
        )

    resolve_task_inline.assert_not_awaited()


@pytest.mark.asyncio
async def test_blocked_capabilities_return_prompt_safe_observations() -> None:
    """Clarification and approval capabilities should block without side effects."""

    state = _resolver_state()

    clarification = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(
            capability_kind="human_clarification",
            objective="请只问用户所在城市。",
        ),
        state,
    )
    approval = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(
            capability_kind="approval_preparation",
            objective="说明准备创建提醒，但等待用户确认。",
        ),
        state,
    )

    assert clarification["status"] == "blocked"
    assert clarification["capability_kind"] == "human_clarification"
    assert "请只问用户所在城市" in clarification["prompt_safe_summary"]
    assert approval["status"] == "blocked"
    assert approval["capability_kind"] == "approval_preparation"
    assert "等待用户确认" in approval["prompt_safe_summary"]
    assert "approval preparation only" in approval["prompt_safe_summary"]
    assert "file inspection" in approval["prompt_safe_summary"]


@pytest.mark.asyncio
async def test_self_goal_resolution_blocks_user_message_source() -> None:
    """User-message turns must not spawn private self-goal execution."""

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(
            capability_kind="self_goal_resolution",
            objective="整理一个内部目标。",
        ),
        _resolver_state(),
    )

    assert observation["status"] == "blocked"
    assert observation["capability_kind"] == "self_goal_resolution"


@pytest.mark.asyncio
async def test_self_goal_resolution_allows_internal_thought_source() -> None:
    """Internal thought may produce a private self-resolution observation."""

    state = _resolver_state()
    episode = canonical_episode(
        episode_id="resolver-internal-thought-episode",
        trigger_source="internal_thought",
        content="整理一个内部目标。",
    )
    state["cognitive_episode"] = episode

    observation = await capabilities_module.execute_resolver_capability_request(
        _resolver_request(
            capability_kind="self_goal_resolution",
            objective="整理一个内部目标。",
        ),
        state,
    )

    assert observation["status"] == "succeeded"
    assert observation["capability_kind"] == "self_goal_resolution"
    assert "internal cognition source" in observation["prompt_safe_summary"]
    assert "rag_result" not in observation
