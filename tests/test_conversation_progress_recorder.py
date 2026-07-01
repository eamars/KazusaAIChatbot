"""Recorder boundary-profile tests for conversation progress."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.conversation_progress import recorder
from kazusa_ai_chatbot.conversation_progress import runtime
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressRecordInput,
    ConversationProgressScope,
)
from kazusa_ai_chatbot.nodes.boundary_profile import (
    get_boundary_recovery_description,
    get_relationship_priority_description,
    get_self_integrity_description,
)
from kazusa_ai_chatbot.config import DEFAULT_LLM_MAX_COMPLETION_TOKENS


_BOUNDARY_PROFILE = {
    "self_integrity": 0.82,
    "control_sensitivity": 0.3,
    "compliance_strategy": "comply",
    "relational_override": 0.24,
    "control_intimacy_misread": 0.2,
    "boundary_recovery": "rebound",
    "authority_skepticism": 0.35,
}

_VALID_RECORDER_OUTPUT = {
    "continuity": "same_episode",
    "status": "active",
    "episode_label": "clarified_reference",
    "conversation_mode": "casual_chat",
    "episode_phase": "resolving",
    "topic_momentum": "stable",
    "current_thread": "clarified referent",
    "user_goal": "",
    "current_blocker": "",
    "user_state_updates": [],
    "assistant_moves": ["clarified referent"],
    "overused_moves": [],
    "open_loops": [],
    "resolved_threads": ["referent clarified"],
    "avoid_reopening": [],
    "emotional_trajectory": "settled",
    "next_affordances": ["continue normally"],
    "progression_guidance": "continue without carrying old suspicion",
}


class _FakeResponse:
    """Small LLM response stand-in."""

    def __init__(self, payload: dict):
        self.content = json.dumps(payload)


class _CapturingLLM:
    """Capture recorder messages while returning a fixed JSON payload."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.messages = []

    async def ainvoke(self, messages, *, config):
        del config
        self.messages = messages
        response = _FakeResponse(self.payload)
        return response


def _record_input(
    boundary_profile: dict,
) -> ConversationProgressRecordInput:
    """Build a recorder input fixture.

    Args:
        boundary_profile: Boundary configuration to include.

    Returns:
        Recorder input fixture.
    """

    record_input: ConversationProgressRecordInput = {
        "scope": ConversationProgressScope("qq", "channel-1", "user-1"),
        "storage_timestamp_utc": "2026-05-01T04:00:00+00:00",
        "character_name": "TestCharacter",
        "prior_episode_state": None,
        "decontexualized_input": "I meant the other thing.",
        "chat_history_recent": [],
        "content_plan": {
            "semantic_content": "Accept the clarification.",
            "rendering": "One ordinary text message; concise.",
        },
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "final_dialog": ["Got it, then I will use that meaning."],
        "boundary_profile": boundary_profile,
    }
    return record_input


def test_render_recorder_prompt_requires_absolute_or_omit_temporal_state() -> None:
    """Recorder prompt declares producer-owned temporal grounding."""

    prompt = recorder.render_recorder_prompt("测试角色")

    assert "测试角色" in prompt
    assert "{character_name}" not in prompt
    assert "下一轮可以直接使用的短期进度" in prompt
    assert "默认删除" in prompt
    assert "不要把旧的、不确定的、相对时间的事项污染到下一轮" in prompt
    assert "用户项目名、产品名、文件名、频道名等专名可以保留" in prompt
    assert "请务必返回合法的 JSON 字符串" in prompt
    assert "当前角色" not in prompt
    assert "紧凑 assistant" not in prompt


def test_recorder_llm_uses_shared_completion_token_budget() -> None:
    """Recorder calls should inherit the shared output cap."""

    assert recorder._recorder_llm_config.max_completion_tokens == (
        DEFAULT_LLM_MAX_COMPLETION_TOKENS
    )


def test_render_recorder_prompt_shows_concrete_list_string_shapes() -> None:
    """Recorder prompt must show string-list fields with literal string examples."""

    prompt = recorder.render_recorder_prompt("测试角色")

    assert "# 生成步骤" in prompt
    assert "# 输入格式" in prompt
    assert "# 输出格式" in prompt
    assert "# 连续性判断" not in prompt
    assert "# Generation Procedure" not in prompt
    assert "# Input Format" not in prompt
    assert "# Output Format" not in prompt
    assert "枚举取值" not in prompt
    assert "活跃操作状态" not in prompt
    assert "短期工作记忆" not in prompt
    assert "open loop" not in prompt
    assert "不要照抄旧状态里的 `task_support`" in prompt
    assert '"conversation_mode": "任务协助"' in prompt
    assert '"assistant_moves": ["旧动作标签"]' in prompt
    assert "不是带 `text` 或 `first_seen_at` 的对象数组" in prompt
    assert '"user_state_updates": ["观察1", "..."]' in prompt
    assert '"assistant_moves": ["标签1", "..."]' in prompt
    assert '"overused_moves": ["标签1", "..."]' in prompt
    assert '"open_loops": ["事项1", "..."]' in prompt
    assert '"resolved_threads": ["事项1", "..."]' in prompt
    assert '"avoid_reopening": ["事项1", "..."]' in prompt
    assert '"next_affordances": ["动作1", "..."]' in prompt
    assert '"assistant_moves": ["紧凑话语动作标签"]' not in prompt


@pytest.mark.asyncio
async def test_recorder_prompt_requires_character_name_and_prompt_safe_history_projection(
    monkeypatch,
) -> None:
    """Recorder prompt renders character identity without payload duplication."""

    fake_llm = _CapturingLLM(_VALID_RECORDER_OUTPUT)
    monkeypatch.setattr(recorder, "_recorder_llm", fake_llm)
    record_input = _record_input(_BOUNDARY_PROFILE)
    record_input["character_name"] = '测试角色'
    record_input["chat_history_recent"] = [
        {
            "role": "assistant",
            "display_name": "助手",
            "body_text": '别急，我已经听到了。',
        }
    ]

    await recorder.record_with_llm(record_input)

    system_prompt = fake_llm.messages[0].content
    human_payload = json.loads(fake_llm.messages[1].content)
    projected_history = human_payload["chat_history_recent"]
    assert "character_name" not in human_payload
    assert '测试角色' in system_prompt
    assert '{character_name}' not in system_prompt
    assert '"character_name": "本轮人设名"' not in system_prompt
    assert len(projected_history) == 1
    assert isinstance(projected_history[0], str)
    assert '助手' in projected_history[0]
    assert '别急，我已经听到了。' in projected_history[0]


@pytest.mark.asyncio
async def test_record_with_llm_sends_boundary_descriptors_not_config_values(monkeypatch) -> None:
    """Recorder prompt payload contains descriptors, not boundary config values."""

    fake_llm = _CapturingLLM(_VALID_RECORDER_OUTPUT)
    monkeypatch.setattr(recorder, "_recorder_llm", fake_llm)

    result = await recorder.record_with_llm(_record_input(_BOUNDARY_PROFILE))

    human_payload = json.loads(fake_llm.messages[1].content)
    profile_payload = human_payload["character_boundary_profile"]
    assert human_payload["current_turn_timestamp"] == "2026-05-01 16:00"
    assert profile_payload == {
        "boundary_recovery_description": get_boundary_recovery_description(
            _BOUNDARY_PROFILE["boundary_recovery"],
        ),
        "self_integrity_description": get_self_integrity_description(
            _BOUNDARY_PROFILE["self_integrity"],
        ),
        "relationship_priority_description": get_relationship_priority_description(
            _BOUNDARY_PROFILE["relational_override"],
        ),
    }
    assert "boundary_recovery" not in profile_payload
    assert "self_integrity" not in profile_payload
    assert "relational_override" not in profile_payload
    serialized_profile = json.dumps(profile_payload, ensure_ascii=False)
    assert "rebound" not in serialized_profile
    assert "0.82" not in serialized_profile
    assert "0.24" not in serialized_profile
    assert result["progression_guidance"] == (
        "continue without carrying old suspicion"
    )


@pytest.mark.asyncio
async def test_runtime_record_accepts_boundary_profile_without_schema_change(monkeypatch) -> None:
    """Runtime writes a normal episode document when boundary_profile is supplied."""

    recorder_callable = AsyncMock(return_value=dict(_VALID_RECORDER_OUTPUT))
    stored_documents = []

    def _store_completed_document(*, scope, document) -> None:
        stored_documents.append(document)

    monkeypatch.setattr(
        runtime.repository,
        "upsert_episode_state_guarded",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        runtime.cache,
        "store_completed_document",
        _store_completed_document,
    )
    progress_runtime = runtime.ConversationProgressRuntime(
        recorder_callable=recorder_callable,
    )

    result = await progress_runtime.record_turn_progress(
        record_input=_record_input(_BOUNDARY_PROFILE),
    )

    recorder_callable.assert_awaited_once()
    assert recorder_callable.await_args.args[0]["boundary_profile"] == (
        _BOUNDARY_PROFILE
    )
    assert result["written"] is True
    assert stored_documents[0]["next_affordances"] == ["continue normally"]
    assert "boundary_profile" not in stored_documents[0]
