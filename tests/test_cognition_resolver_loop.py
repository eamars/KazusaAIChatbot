"""Tests for cognition resolver capability execution."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_resolver import capabilities as capabilities_module
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    ResolverValidationError,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock


def _resolver_request(
    *,
    capability_kind: str = "rag_evidence",
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
    assert observation["capability_kind"] == "rag_evidence"
    assert observation["request_objective"] == request["objective"]
    assert observation["request_reason"] == request["reason"]
    assert observation["rag_result"]["answer"] == "找到一条关系记忆。"
    assert "memory_evidence" in observation["rag_result"]
    assert "user_image" in observation["rag_result"]


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
