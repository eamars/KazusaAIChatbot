"""Tests for cognition resolver capability execution."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    build_text_chat_cognitive_episode,
    validate_cognitive_episode,
)
from kazusa_ai_chatbot.cognition_resolver import capabilities as capabilities_module
from kazusa_ai_chatbot.cognition_resolver.contracts import (
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
from kazusa_ai_chatbot.time_boundary import build_turn_clock


def _resolver_request(
    *,
    capability_kind: str = "local_context_recall",
    objective: str = "检索当前用户与这个问题有关的关系和记忆证据。",
) -> dict:
    return {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": capability_kind,
        "objective": objective,
        "reason": "当前认知循环缺少足够证据。",
        "priority": "now",
    }


def _resolver_state() -> dict:
    turn_clock = build_turn_clock("2026-05-30 09:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="resolver-capability-episode",
        percept_id="resolver-capability-percept",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="Need an evidence-backed answer.",
        platform="debug",
        platform_channel_id="channel-123",
        channel_type="private",
        platform_message_id="message-123",
        platform_user_id="platform-user-123",
        global_user_id="global-user-123",
        user_name="Test User",
        active_turn_platform_message_ids=["message-123"],
        active_turn_conversation_row_ids=["row-123"],
        debug_modes={},
        target_addressed_user_ids=["character-123"],
        target_broadcast=False,
    )
    return {
        "decontexualized_input": "Original user request about trust.",
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
        "user_name": "Test User",
        "user_profile": {"affinity": 500},
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
        "active_turn_platform_message_ids": ["message-123"],
        "active_turn_conversation_row_ids": ["row-123"],
        "cognitive_episode": episode,
    }


def _internal_thought_resolver_state() -> dict:
    """Return resolver state for a private internal-thought cognition source."""

    state = _resolver_state()
    episode = dict(state["cognitive_episode"])
    episode["trigger_source"] = "internal_thought"
    episode["input_sources"] = ["internal_monologue"]
    episode["output_mode"] = "think_only"
    episode["percepts"] = [{
        "percept_id": "resolver-internal-thought",
        "input_source": "internal_monologue",
        "content": "整理一个内部目标。",
        "visibility": "internal_only",
        "metadata": {},
    }]
    validate_cognitive_episode(episode)
    state["cognitive_episode"] = episode
    state["channel_type"] = "group"
    return_value = state
    return return_value


def _cognition_result(
    *,
    internal_monologue: str,
    action_specs: list[dict] | None = None,
    resolver_requests: list[dict] | None = None,
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
    }


def _speak_action_spec(reason: str = "已经有足够证据，可以进入可见回复。") -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": "speak",
        "cognition_mode": "deliberative",
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
    return {
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
            "capability_kind": "public_answer_research",
            "request_objective": "检索奥克兰 CBD 餐厅当前营业状态。",
            "request_reason": "需要当前营业证据。",
            "status": "failed",
            "prompt_safe_summary": "搜索工具未返回已确认事实。",
            "evidence_refs": [],
            "created_at_utc": "2026-05-29T21:00:00+00:00",
        },
    ]

    resolver_context = project_resolver_context(resolver_state)

    assert "original_goal=Original user request about trust." in resolver_context
    assert "objective=检索奥克兰 CBD 餐厅当前营业状态。" in resolver_context
    assert "summary=搜索工具未返回已确认事实。" in resolver_context
    assert "resolver_goal_progress:" in resolver_context


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
        "local_context_recall"
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
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "没有确认到每家店当前营业状态。",
            "evidence_refs": [],
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

    request = _resolver_request(objective="检索一个会超时的证据目标。")
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
    assert observation["capability_kind"] == "local_context_recall"
    assert observation["request_objective"] == request["objective"]
    assert "timed out" in observation["prompt_safe_summary"]
    terminal_event = build_resolver_terminal_event(result, duration_ms=1)
    terminal_json = json.dumps(terminal_event, ensure_ascii=False)
    assert "failed" in terminal_json
    assert "timed out" in terminal_json
    assert "Need an evidence-backed answer" not in terminal_json
    assert "message-123" not in terminal_json


@pytest.mark.asyncio
async def test_loop_blocks_duplicate_capability_objective_before_execution() -> None:
    """Exact repeated resolver objectives should not execute indefinitely."""

    request = _resolver_request(
        capability_kind="public_answer_research",
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
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "没有找到已确认事实。",
            "evidence_refs": [],
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
    assert result["action_specs"][0]["kind"] == "speak"


@pytest.mark.asyncio
async def test_loop_blocks_same_capability_retry_after_timeout() -> None:
    """Timed-out capability work should not be retried with renamed objective."""

    first_request = _resolver_request(
        capability_kind="public_answer_research",
        objective="检索当前外部事实。",
    )
    renamed_request = _resolver_request(
        capability_kind="public_answer_research",
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
async def test_duplicate_final_cognition_repeated_request_gets_terminal_speak() -> None:
    """Terminal duplicate handling should not leave the user with silence."""

    request = _resolver_request(
        capability_kind="public_answer_research",
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
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "没有找到已确认事实。",
            "evidence_refs": [],
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
    assert "当前证据获取已经阻塞" in surface_requirements["detail"]
    assert "已由来源支持的事实" in surface_requirements["detail"]
    assert "final_response_requirements" in surface_requirements["detail"]
    assert "临时处理状态或延后承诺" in surface_requirements["detail"]
    assert "泛化说明也不能偷换成未授权的" in surface_requirements["detail"]
    assert "具体当前实体、属性、实时状态" in surface_requirements["detail"]
    assert "不能以追问结尾" in surface_requirements["detail"]


@pytest.mark.asyncio
async def test_duplicate_final_cognition_changed_request_gets_terminal_speak() -> None:
    """Terminal duplicate handling should not run a rephrased tool request."""

    original_request = _resolver_request(
        capability_kind="public_answer_research",
        objective="检索同一个当前外部事实目标。",
    )
    rephrased_request = _resolver_request(
        capability_kind="public_answer_research",
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
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "没有找到已确认事实。",
            "evidence_refs": [],
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
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "证据仍不足。",
            "evidence_refs": [],
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
        capability_kind="public_answer_research",
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
            "request_objective": capability_request["objective"],
            "request_reason": capability_request["reason"],
            "status": "failed",
            "prompt_safe_summary": "搜索超时，但已有部分约束可说明。",
            "evidence_refs": [],
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

    clarification_request = _resolver_request(
        capability_kind="human_clarification",
        objective="请只问用户所在城市。",
    )
    evidence_request = _resolver_request(
        objective="根据用户补充的城市继续生成今晚计划需要的证据。",
    )
    pending_rows: list[dict] = []
    applied_rows: list[dict] = []

    async def first_turn_cognition(state: dict) -> dict:
        if "pending_resolver_resume" not in state["resolver_context"]:
            return _cognition_result(
                internal_monologue="第一轮：缺少城市，必须先问用户。",
                resolver_requests=[clarification_request],
            )
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
        _resolver_state(),
        call_cognition_subgraph_func=first_turn_cognition,
        execute_capability_func=first_turn_capability,
        max_cycles=3,
        capability_timeout_seconds=1.0,
        upsert_pending_resume_func=upsert_pending,
    )

    assert first_result["resolver_state"]["status"] == "waiting_for_user"
    assert len(pending_rows) == 1

    follow_up_state = ensure_initial_resolver_inputs(
        _resolver_state(),
        max_cycles=3,
    )
    follow_up_state["platform_message_id"] = "message-follow-up-123"

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

    async def follow_up_cognition(state: dict) -> dict:
        follow_up_inputs.append(dict(state))
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
        return {
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_city_plan",
            "capability_kind": capability_request["capability_kind"],
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
    follow_up_state["decontexualized_input"] = "就在奥克兰 CBD。"
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
    assert loaded_state["resolver_state"]["original_decontexualized_input"] == (
        "今晚安排一个两小时低预算计划。"
    )
    assert "就在奥克兰 CBD" not in loaded_state["resolver_context"]


@pytest.mark.asyncio
async def test_rag_capability_uses_objective_and_preserves_original_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG resolver execution should use objective while preserving context."""

    captured: dict = {}

    async def call_rag_supervisor(
        *,
        fresh_query: str,
        reply_context: dict,
        character_name: str,
        context: dict,
    ) -> dict[str, object]:
        captured["fresh_query"] = fresh_query
        captured["reply_context"] = reply_context
        captured["character_name"] = character_name
        captured["context"] = context
        result = {
            "answer": "找到一条关系记忆。",
            "known_facts": [
                {
                    "slot": "memory",
                    "agent": "memory_evidence_agent",
                    "resolved": True,
                    "summary": "存在一条信任相关记忆。",
                    "raw_result": {
                        "projection_payload": {"memory_rows": []},
                    },
                }
            ],
            "unknown_slots": [],
            "loop_count": 1,
        }
        return result

    monkeypatch.setattr(
        capabilities_module,
        "call_quote_aware_rag_supervisor",
        call_rag_supervisor,
    )
    monkeypatch.setattr(
        capabilities_module.event_logging,
        "record_rag_stage_event",
        AsyncMock(),
    )
    request = _resolver_request()

    observation = await capabilities_module.execute_resolver_capability_request(
        request,
        _resolver_state(),
    )

    assert captured["fresh_query"] == request["objective"]
    assert captured["context"]["original_user_request"] == (
        "Original user request about trust."
    )
    assert observation["status"] == "succeeded"
    assert observation["capability_kind"] == "local_context_recall"
    assert observation["request_objective"] == request["objective"]
    assert observation["request_reason"] == request["reason"]
    assert observation["rag_result"]["answer"] == "找到一条关系记忆。"
    assert "memory_evidence" in observation["rag_result"]
    assert "user_image" in observation["rag_result"]


@pytest.mark.asyncio
async def test_internal_thought_rag_capability_uses_existing_rag_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selected internal-thought RAG should execute through existing RAG."""

    captured: dict = {}

    async def call_rag_supervisor(
        *,
        fresh_query: str,
        reply_context: dict,
        character_name: str,
        context: dict,
    ) -> dict[str, object]:
        captured["fresh_query"] = fresh_query
        captured["reply_context"] = reply_context
        captured["character_name"] = character_name
        captured["context"] = context
        result = {
            "answer": "找到一条群聊前文证据。",
            "known_facts": [
                {
                    "slot": "Conversation-evidence: prior group context",
                    "agent": "conversation_evidence_agent",
                    "resolved": True,
                    "summary": "前文显示这是一段延续中的群聊话题。",
                    "raw_result": {
                        "projection_payload": {
                            "conversation_rows": [{
                                "content": "前文提到同一个话题。",
                            }],
                        },
                    },
                }
            ],
            "unknown_slots": [],
            "loop_count": 1,
        }
        return result

    monkeypatch.setattr(
        capabilities_module,
        "call_quote_aware_rag_supervisor",
        call_rag_supervisor,
    )
    monkeypatch.setattr(
        capabilities_module.event_logging,
        "record_rag_stage_event",
        AsyncMock(),
    )
    request = _resolver_request(
        objective="回看群聊前文，判断这个内部想法是否有足够证据。",
    )

    observation = await capabilities_module.execute_resolver_capability_request(
        request,
        _internal_thought_resolver_state(),
    )

    assert captured["fresh_query"] == request["objective"]
    assert captured["context"]["original_user_request"] == (
        "Original user request about trust."
    )
    assert captured["context"]["platform"] == "debug"
    assert captured["context"]["channel_type"] == "private"
    assert captured["context"]["global_user_id"] == "global-user-123"
    assert captured["context"]["active_turn_conversation_row_ids"] == [
        "row-123",
    ]
    assert captured["character_name"] == "Kazusa"
    assert observation["status"] == "succeeded"
    assert observation["capability_kind"] == "local_context_recall"
    assert observation["rag_result"]["answer"] == "找到一条群聊前文证据。"
    assert "conversation_evidence" in observation["rag_result"]


@pytest.mark.asyncio
async def test_public_answer_research_uses_complex_resolver(
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

    monkeypatch.setattr(
        capabilities_module,
        "resolve_complex_task",
        resolve_complex_task,
        raising=False,
    )
    request = _resolver_request(
        capability_kind="public_answer_research",
        objective="查询当前公共网页事实。",
    )

    observation = await capabilities_module.execute_resolver_capability_request(
        request,
        _resolver_state(),
    )

    assert captured["request"]["objective"] == request["objective"]
    assert captured["request"]["source"] == "l2d"
    assert captured["context"]["conversation_summary"] == (
        "Original user request about trust."
    )
    assert observation["status"] == "succeeded"
    assert observation["capability_kind"] == "public_answer_research"
    assert observation["prompt_safe_summary"] == "找到一条网页证据。"
    assert observation["knowledge_projection"] == {
        "investigation_summary": "找到一条网页证据。",
        "knowledge_we_know_so_far": ["网页证据显示当前事实可用。"],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [
            "由 cognition 判断是否足够进入可见回答。",
        ],
        "evidence_boundary_notes": [],
    }
    assert observation["evidence_refs"][0]["owner"] == "complex_task_resolver"


@pytest.mark.asyncio
async def test_empty_resolver_objective_fails_before_rag_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed resolver requests must fail before invoking RAG."""

    call_rag_supervisor = AsyncMock()
    monkeypatch.setattr(
        capabilities_module,
        "call_quote_aware_rag_supervisor",
        call_rag_supervisor,
    )
    request = _resolver_request(objective=" ")

    with pytest.raises(ResolverValidationError, match="objective"):
        await capabilities_module.execute_resolver_capability_request(
            request,
            _resolver_state(),
        )

    call_rag_supervisor.assert_not_awaited()


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
    episode = dict(state["cognitive_episode"])
    episode["trigger_source"] = "internal_thought"
    episode["input_sources"] = ["internal_monologue"]
    episode["output_mode"] = "think_only"
    episode["percepts"] = [{
        "percept_id": "resolver-internal-thought",
        "input_source": "internal_monologue",
        "content": "整理一个内部目标。",
        "visibility": "internal_only",
        "metadata": {},
    }]
    validate_cognitive_episode(episode)
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
