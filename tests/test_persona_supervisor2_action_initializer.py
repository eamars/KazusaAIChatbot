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
from kazusa_ai_chatbot.time_context import build_character_time_context


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
    timestamp = "2026-05-16T09:00:00+12:00"
    episode = build_text_chat_cognitive_episode(
        episode_id="user_message:debug:raw-channel-123:raw-message-456",
        percept_id="user_message:debug:raw-channel-123:raw-message-456:dialog_text:0",
        timestamp=timestamp,
        time_context=build_character_time_context(timestamp),
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
    return {
        "timestamp": "2026-05-16T09:00:00+12:00",
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
        "idle_timestamp": "2026-05-16T09:00:00+12:00",
        "last_evidence_timestamp": "2026-05-15T09:00:00+12:00",
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
    """L2d should receive semantic trigger context, not raw transport internals."""

    payload = l2d_module.build_action_initializer_payload(_state())
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()

    assert payload["trigger_context"]["trigger_source"] == "user_message"
    assert "speak" in serialized
    assert "memory_lifecycle_update" in serialized
    assert "trigger_future_cognition" in serialized
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
            "due_at": "2026-05-07T00:00:00+00:00",
            "due_state": self_cognition_models.DUE_STATE_PAST_DUE,
            "status": "active",
        }
    ]


def test_action_initializer_prompt_follows_cognition_prompt_structure() -> None:
    """The L2d prompt should follow the established cognition prompt pattern."""

    prompt = l2d_module._ACTION_INITIALIZER_PROMPT

    for required_section in (
        "# 语言政策",
        "# 核心任务",
        "# 运行规则",
        "# 自我认知输出规则",
        "# 能力语义",
        "# 思考路径",
        "# 输入格式",
        "# 输出格式",
    ):
        assert required_section in prompt
    assert "你现在是角色" in prompt
    assert "请务必返回合法的 JSON 字符串" in prompt
    assert "`speak`" in prompt
    assert "`trigger_future_cognition`" in prompt
    assert "`send_message`" not in prompt
    assert "执行信封、目标对象、持久化字段" in prompt
    assert "只写语义请求" in prompt
    assert "action_requests" in prompt
    assert "self_cognition" in prompt
    assert "scheduled_tick" in prompt
    assert "tool_result" in prompt
    assert "reflection_signal" not in prompt
    assert "scheduled_recall" not in prompt
    assert "system_probe" not in prompt
    assert "schema_version" not in prompt
    assert "action_target" not in prompt
    assert "action_spec" not in prompt
    assert "continuation" not in prompt
    assert "l3_text" not in prompt
    assert "handler_id" not in prompt
    assert "不要因为 `logical_stance=CONFIRM`" in prompt
    assert "只有当 `final_l2` 明确决定要对外联系" in prompt
    assert "未来某个时刻再自检" in prompt


def test_action_initializer_hides_lifecycle_without_single_bound_target() -> None:
    """Lifecycle should not be offered when code has no deterministic target."""

    state = _state()
    state["rag_result"]["user_image"]["user_memory_context"][
        "active_commitments"
    ] = []

    payload = l2d_module.build_action_initializer_payload(state)

    capabilities = [
        capability["capability"]
        for capability in payload["capabilities"]
    ]
    assert "memory_lifecycle_update" not in capabilities


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
    human_payload = json.loads(fake_llm.messages[1].content)
    assert "trigger_context" in human_payload


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
