"""Deterministic tests for the isolated L2d action initializer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2d as l2d_module
from kazusa_ai_chatbot.self_cognition import models as self_cognition_models
from kazusa_ai_chatbot.self_cognition import runner as self_cognition_runner
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


class _FakeLLM:
    """Capture the L2d prompt call and return one configured JSON payload."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        self.messages = messages
        response = SimpleNamespace(content=self.content)
        return response


def _episode() -> dict:
    storage_timestamp_utc = "2026-05-15T21:00:00+00:00"
    turn_clock = build_turn_clock_from_storage_utc(storage_timestamp_utc)
    episode = build_text_chat_cognitive_episode(
        episode_id="user_message:debug:raw-channel-123:raw-message-456",
        percept_id="user_message:debug:raw-channel-123:raw-message-456:dialog_text:0",
        storage_timestamp_utc=storage_timestamp_utc,
        local_time_context=turn_clock["local_time_context"],
        user_input="Please handle the old spice promise naturally.",
        platform="debug",
        platform_channel_id="raw-channel-123",
        channel_type="private",
        platform_message_id="raw-message-456",
        platform_user_id="platform-user-raw",
        global_user_id="global-user-raw",
        user_name="Test User",
        active_turn_platform_message_ids=["raw-message-456"],
        active_turn_conversation_row_ids=["conversation-row-raw"],
        debug_modes={},
        output_mode="visible_reply",
        target_addressed_user_ids=[],
        target_broadcast=False,
    )
    return episode


def _state() -> dict:
    storage_timestamp_utc = "2026-05-15T21:00:00+00:00"
    turn_clock = build_turn_clock_from_storage_utc(storage_timestamp_utc)
    return {
        "storage_timestamp_utc": storage_timestamp_utc,
        "local_time_context": turn_clock["local_time_context"],
        "cognitive_episode": _episode(),
        "channel_type": "private",
        "decontexualized_input": (
            "The user asks whether the active character should deal with an "
            "old promise."
        ),
        "internal_monologue": (
            "The spice promise is overdue, but it should be handled naturally."
        ),
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "judgment_note": "The final decision is to handle the promise without forcing it.",
        "emotional_appraisal": "calm",
        "interaction_subtext": "low pressure",
        "boundary_core_assessment": {
            "boundary_issue": "none",
            "acceptance": "allow",
            "stance_bias": "confirm",
        },
        "social_distance": "friendly but not intrusive",
        "emotional_intensity": "quiet and low pressure",
        "vibe_check": "relaxed daily conversation",
        "relational_dynamic": "stable trust with room to wait",
        "rag_result": {
            "answer": "The active commitment is overdue.",
            "user_image": {
                "user_memory_context": {
                    "active_commitments": [
                        {
                            "unit_id": "promise-001",
                            "fact": "Reveal the spice answer.",
                            "due_at": "2026-05-07T00:00:00+00:00",
                            "due_state": "past_due",
                        }
                    ]
                }
            },
            "memory_evidence": [
                {
                    "summary": "The spice promise is past due.",
                    "source_id": "promise-001",
                }
            ],
        },
        "conversation_progress": {
            "current_thread": "hardware discussion",
            "next_affordances": ["wait for a natural pause"],
        },
    }


def _self_cognition_commitment_case() -> dict:
    """Build a self-cognition source case for one active commitment target."""

    case = {
        "case_name": self_cognition_models.CASE_COMMITMENT_PAST_DUE,
        "case_id": "active_commitment:promise-001:2026-05-07T00:00:00+00:00",
        "idle_timestamp_utc": "2026-05-15T21:00:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-14T21:00:00+00:00",
        "trigger_kind": (
            self_cognition_models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK
        ),
        "semantic_due_state": self_cognition_models.DUE_STATE_PAST_DUE,
        "actionability": "past_due_commitment_contact_socially_available",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "group",
            "user_id": "global-user-001",
        },
        "source_refs": [
            {
                "source_kind": "user_memory_unit",
                "source_id": "promise-001",
                "due_at": "2026-05-07T00:00:00+00:00",
                "summary": (
                    "The active character promised to reveal the spice answer."
                ),
            }
        ],
        "visible_context": [],
    }
    return case


def _speak_request(reason: str = "A visible text surface is needed.") -> dict:
    return {
        "capability": "speak",
        "decision": "visible_reply",
        "detail": "Use a natural and brief tone.",
        "reason": reason,
    }


def _memory_lifecycle_request() -> dict:
    return {
        "capability": "memory_lifecycle_update",
        "decision": "deferred",
        "detail": "The spice promise remains open until a natural pause.",
        "reason": "The promise remains open until the character can raise it naturally.",
    }


def _future_cognition_request() -> dict:
    return {
        "capability": "trigger_future_cognition",
        "decision": "future_self_check",
        "detail": "Check whether the topic has a natural pause.",
        "reason": "The character wants to revisit this later without speaking now.",
    }


def test_action_initializer_payload_is_prompt_safe() -> None:
    """L2d should receive one semantic string, not raw transport internals."""

    action_context = l2d_module.build_action_initializer_payload(_state())
    serialized = action_context.lower()

    assert action_context.startswith("当前行动上下文：")
    assert "触发来源：user_message" in action_context
    assert "输出要求：visible_reply" in action_context
    assert "场景：private 对话" in action_context
    assert "距离=friendly but not intrusive" in action_context
    assert "强度=quiet and low pressure" in action_context
    assert "氛围=relaxed daily conversation" in action_context
    assert "关系=stable trust with room to wait" in action_context
    assert "可绑定承诺：有" in action_context
    assert "Reveal the spice answer." in action_context
    assert "semantic_input_summary" not in serialized
    assert "execution_boundary" not in serialized
    assert "available_capabilities" not in serialized
    assert "send_message" not in serialized
    for forbidden in (
        "raw-channel-123",
        "raw-message-456",
        "platform-user-raw",
        "global-user-raw",
        "handler_id",
        "dispatcher.send_message",
        "l3_text",
        "user_memory_units",
        "mongodb",
        "mongo",
        "credential",
        "platform_channel_id",
        "channel_id",
        "schema_version",
        "unit_id",
        "source_id",
        "promise-001",
    ):
        assert forbidden not in serialized


def test_self_cognition_source_ref_binds_active_commitment_context() -> None:
    """Self-cognition source refs should become deterministic L2d targets."""

    state = self_cognition_runner._build_cognition_state(
        _self_cognition_commitment_case(),
        "rendered self-cognition packet",
    )

    commitments = (
        state["rag_result"]["user_image"]["user_memory_context"][
            "active_commitments"
        ]
    )

    assert commitments == [
        {
            "unit_id": "promise-001",
            "fact": "The active character promised to reveal the spice answer.",
            "summary": "The active character promised to reveal the spice answer.",
            "due_at": "2026-05-07 12:00",
            "due_state": self_cognition_models.DUE_STATE_PAST_DUE,
            "status": "active",
        }
    ]


def test_action_initializer_prompt_follows_cognition_prompt_structure() -> None:
    """The L2d prompt should follow the established cognition prompt pattern."""

    prompt = l2d_module._ACTION_INITIALIZER_PROMPT

    for required_section in (
        "# 语言政策",
        "# 可选动作",
        "# 选择流程",
        "# 未来认知判断",
        "# 输入格式",
        "# 输出格式",
    ):
        assert required_section in prompt
    assert "你是角色的语义行动选择层" in prompt
    assert "用户消息只包含本轮动态行动上下文" in prompt
    assert "行动请求只描述角色想做什么" in prompt
    assert "只返回合法 JSON 字符串" in prompt
    assert "`speak`" in prompt
    assert "`trigger_future_cognition`" in prompt
    assert "`send_message`" not in prompt
    assert "用户消息是一段中文行动上下文字符串，不是 JSON" in prompt
    assert "具体新信息" in prompt
    assert "具体问题、任务或承诺" in prompt
    assert "action_requests" in prompt
    assert "available_capabilities" not in prompt
    assert "reflection_signal" not in prompt
    assert "scheduled_recall" not in prompt
    assert "system_probe" not in prompt
    assert "schema_version" not in prompt
    assert "action_target" not in prompt
    assert "action_spec" not in prompt
    assert "continuation" not in prompt
    assert "l3_text" not in prompt
    assert "handler_id" not in prompt
    assert "final_l2" not in prompt
    assert "trigger_context" not in prompt
    assert "social_context_appraisal" not in prompt
    assert "L2c1" not in prompt
    assert "L2c2" not in prompt
    assert "小判断例" not in prompt
    assert "5090 能跑什么人工智能模型" not in prompt


def test_action_initializer_hides_lifecycle_without_single_bound_target() -> None:
    """Lifecycle should not be offered when code has no deterministic target."""

    state = _state()
    state["rag_result"]["user_image"]["user_memory_context"][
        "active_commitments"
    ] = []

    action_context = l2d_module.build_action_initializer_payload(state)

    assert "可绑定承诺：无" in action_context
    assert "memory_lifecycle_update" not in action_context


@pytest.mark.asyncio
async def test_action_initializer_ignores_lifecycle_target_ids_from_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opaque IDs in L2d text must not select one target from many."""

    state = _state()
    state["rag_result"]["user_image"]["user_memory_context"][
        "active_commitments"
    ] = [
        {
            "unit_id": "promise-001",
            "fact": "Reveal the spice answer.",
            "due_at": "2026-05-07T00:00:00+00:00",
            "due_state": "past_due",
        },
        {
            "unit_id": "promise-002",
            "fact": "Check the tea result.",
            "due_at": "2026-05-08T00:00:00+00:00",
            "due_state": "past_due",
        },
    ]
    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [
            {
                "capability": "memory_lifecycle_update",
                "decision": "abandoned",
                "detail": "Select promise-001.",
                "reason": "The model copied an opaque target identifier.",
            },
        ],
    }))
    monkeypatch.setattr(l2d_module, "_action_initializer_llm", fake_llm)

    result = await l2d_module.call_action_initializer(state)

    assert result["action_specs"] == []


@pytest.mark.asyncio
async def test_future_cognition_materialization_binds_trusted_source_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future cognition source scope is copied by code, not emitted by L2d."""

    state = _state()
    state.update(
        {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "global_user_id": "673225019",
            "platform_bot_id": "bot-001",
            "character_profile": {"name": "TestCharacter"},
        }
    )
    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [_future_cognition_request()],
    }))
    monkeypatch.setattr(l2d_module, "_action_initializer_llm", fake_llm)

    result = await l2d_module.call_action_initializer(state)

    action_spec = result["action_specs"][0]
    assert action_spec["kind"] == "trigger_future_cognition"
    assert action_spec["continuation"] == {
        "schema_version": "action_continuation.v1",
        "mode": "scheduled_followup",
        "episode_type": "self_cognition",
        "max_depth": 1,
        "include_result_as": "scheduled_event",
    }
    assert action_spec["target"]["scope"] == {
        "episode_type": "self_cognition",
        "source_platform": "qq",
        "source_channel_id": "54369546",
        "source_channel_type": "group",
        "source_user_id": "673225019",
        "source_platform_bot_id": "bot-001",
        "source_character_name": "TestCharacter",
    }
    human_payload = fake_llm.messages[1].content
    assert "54369546" not in human_payload
    assert "bot-001" not in human_payload


@pytest.mark.asyncio
async def test_future_cognition_uses_own_detail_as_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future cognition should carry its own distinct next-cycle objective."""

    speak_request = {
        "capability": "speak",
        "decision": "visible_reply",
        "detail": '回应用户关于 5090 AI 模型运行能力的询问',
        "reason": "The user asked for a visible acknowledgement first.",
    }
    future_request = {
        "capability": "trigger_future_cognition",
        "decision": "future_self_check",
        "detail": '需要查阅目前已知的泄露参数、预测数据以及社区讨论的兼容性信息',
        "reason": "The character needs a later private cognition cycle.",
    }
    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [speak_request, future_request],
    }, ensure_ascii=False))
    monkeypatch.setattr(l2d_module, "_action_initializer_llm", fake_llm)

    result = await l2d_module.call_action_initializer(_state())

    future_specs = [
        spec for spec in result["action_specs"]
        if spec["kind"] == "trigger_future_cognition"
    ]
    assert len(future_specs) == 1
    params = future_specs[0]["params"]
    assert params["continuation_objective"] == (
        '需要查阅目前已知的泄露参数、预测数据以及社区讨论的兼容性信息'
    )
    assert "context_summary" not in params


@pytest.mark.asyncio
async def test_scheduled_future_cognition_cannot_chain_another_future_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A due future-cognition cycle must not schedule itself again."""

    state = _state()
    state["cognitive_episode"]["trigger_source"] = "internal_thought"
    state["conversation_progress"] = {
        "source": "scheduled_future_cognition",
        "continuation_objective": (
            "Check whether the earlier follow-up is still useful."
        ),
    }
    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [_future_cognition_request()],
    }))
    monkeypatch.setattr(l2d_module, "_action_initializer_llm", fake_llm)

    result = await l2d_module.call_action_initializer(state)

    assert result["action_specs"] == []


@pytest.mark.asyncio
async def test_action_initializer_accepts_multiple_valid_action_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L2d can select more than one independent action in one cognition cycle."""

    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [
            _speak_request(),
            _memory_lifecycle_request(),
            _future_cognition_request(),
        ],
    }))
    monkeypatch.setattr(l2d_module, "_action_initializer_llm", fake_llm)

    result = await l2d_module.call_action_initializer(_state())

    assert [spec["kind"] for spec in result["action_specs"]] == [
        "speak",
        "memory_lifecycle_update",
        "trigger_future_cognition",
    ]
    human_context = fake_llm.messages[1].content
    assert human_context.startswith("当前行动上下文：")
    assert "trigger_context" not in human_context
    assert "available_capabilities" not in human_context


@pytest.mark.asyncio
async def test_action_initializer_drops_invalid_specs_and_caps_valid_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed rows should be skipped and valid rows capped before graph merge."""

    invalid_request = {"capability": "speak"}
    fake_llm = _FakeLLM(json.dumps({
        "action_requests": [
            invalid_request,
            _speak_request("first"),
            _speak_request("second"),
            _memory_lifecycle_request(),
            _speak_request("fourth"),
        ],
    }))
    monkeypatch.setattr(l2d_module, "_action_initializer_llm", fake_llm)

    result = await l2d_module.call_action_initializer(_state())

    assert len(result["action_specs"]) == 3
    assert [spec["reason"] for spec in result["action_specs"]] == [
        "first",
        "second",
        "The promise remains open until the character can raise it naturally.",
    ]
