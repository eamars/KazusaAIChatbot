"""Tests for persona_supervisor2.py — top-level orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2, call_action_subgraph


def _base_discord_state():
    """Minimal IMProcessState with all required keys."""
    return {
        "timestamp": "2024-01-01T00:00:00Z",
        "user_name": "TestUser",
        "platform": "discord",
        "platform_message_id": "msg_123",
        "platform_user_id": "user_123",
        "global_user_id": "uuid-123",
        "user_input": "Hello",
        "message_envelope": {
            "body_text": "Hello",
            "raw_wire_text": "Hello",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
        "prompt_message_context": {
            "body_text": "Hello",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
        "user_multimedia_input": [],
        "user_profile": {"affinity": 500},
        "platform_bot_id": "bot_456",
        "character_name": "TestCharacter",
        "character_profile": {
            "name": "Character",
            "global_user_id": "character-uuid",
            "mood": "neutral",
            "global_vibe": "calm",
            "reflection_summary": "nothing notable",
        },
        "platform_channel_id": "chan_1",
        "channel_type": "group",
        "channel_name": "general",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "should_respond": True,
        "reason_to_respond": "user greeted",
        "use_reply_feature": False,
        "channel_topic": "greetings",
        "indirect_speech_context": "",
        "debug_modes": {},
    }


@pytest.mark.asyncio
async def test_call_action_subgraph_returns_final_dialog():
    """call_action_subgraph wraps dialog_agent output correctly."""
    mock_dialog_result = {
        "final_dialog": ["Hello!", "How are you?"],
        "target_addressed_user_ids": ["uuid-123"],
        "target_broadcast": False,
    }

    with patch(
        "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
        new_callable=AsyncMock,
        return_value=mock_dialog_result,
    ):
        result = await call_action_subgraph({"global_user_id": "uuid-123"})

    assert result["final_dialog"] == ["Hello!", "How are you?"]
    assert result["target_addressed_user_ids"] == ["uuid-123"]
    assert result["target_broadcast"] is False


@pytest.mark.asyncio
async def test_call_action_subgraph_empty_dialog():
    """call_action_subgraph handles empty dialog_agent output."""
    mock_dialog_result = {
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
    }

    with patch(
        "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
        new_callable=AsyncMock,
        return_value=mock_dialog_result,
    ):
        result = await call_action_subgraph({"global_user_id": "uuid-123"})

    assert result["final_dialog"] == []
    assert result["target_addressed_user_ids"] == []
    assert result["target_broadcast"] is False


@pytest.mark.asyncio
async def test_persona_supervisor2_returns_final_dialog_and_consolidation_state():
    """persona_supervisor2 should return dialog plus the consolidation snapshot."""
    state = _base_discord_state()

    # Mock graph nodes to avoid real LLM calls.
    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={"decontexualized_input": "Hello"},
        ) as m_decon,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.stage_1_research",
            new_callable=AsyncMock,
            return_value={"rag_result": {}},
        ) as m_research,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_subgraph",
            new_callable=AsyncMock,
            return_value={
                "internal_monologue": "thinking...",
                "action_directives": {},
                "interaction_subtext": "",
                "emotional_appraisal": "",
                "character_intent": "",
                "logical_stance": "",
            },
        ) as m_cognition,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["Hi there!"],
                "target_addressed_user_ids": ["uuid-123"],
                "target_broadcast": False,
            },
        ) as m_dialog,
    ):
        result = await persona_supervisor2(state)

    assert "final_dialog" in result
    assert "future_promises" in result
    assert result["final_dialog"] == ["Hi there!"]
    assert result["target_addressed_user_ids"] == ["uuid-123"]
    assert result["target_broadcast"] is False
    assert result["future_promises"] == []
    assert result["consolidation_state"]["decontexualized_input"] == "Hello"
    assert result["consolidation_state"]["final_dialog"] == ["Hi there!"]
    assert result["consolidation_state"]["reply_context"] == {}


@pytest.mark.asyncio
async def test_persona_supervisor2_scopes_group_history_before_persona_stages():
    """Persona stages should not receive another user's addressed subthread."""
    state = _base_discord_state()
    state["platform_user_id"] = "platform-user-a"
    state["global_user_id"] = "global-user-a"
    state["platform_bot_id"] = "platform-bot"
    state["chat_history_wide"] = [
        {
            "role": "user",
            "platform_user_id": "platform-user-a",
            "global_user_id": "global-user-a",
            "body_text": "current user secret",
            "addressed_to_global_user_ids": ["character-uuid"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-04-30T00:00:00+00:00",
        },
        {
            "role": "assistant",
            "platform_user_id": "platform-bot",
            "global_user_id": "character-uuid",
            "body_text": "current user reply",
            "addressed_to_global_user_ids": ["global-user-a"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-04-30T00:00:01+00:00",
        },
        {
            "role": "user",
            "platform_user_id": "platform-user-b",
            "global_user_id": "global-user-b",
            "body_text": "other user secret",
            "addressed_to_global_user_ids": ["character-uuid"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-04-30T00:00:02+00:00",
        },
        {
            "role": "assistant",
            "platform_user_id": "platform-bot",
            "global_user_id": "character-uuid",
            "body_text": "other user reply",
            "addressed_to_global_user_ids": ["global-user-b"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-04-30T00:00:03+00:00",
        },
    ]
    state["chat_history_recent"] = list(state["chat_history_wide"])

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={"decontexualized_input": "Hello"},
        ) as m_decon,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.stage_1_research",
            new_callable=AsyncMock,
            return_value={"rag_result": {}},
        ) as m_research,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_subgraph",
            new_callable=AsyncMock,
            return_value={
                "internal_monologue": "thinking...",
                "action_directives": {},
                "interaction_subtext": "",
                "emotional_appraisal": "",
                "character_intent": "",
                "logical_stance": "",
            },
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["Hi there!"],
                "target_addressed_user_ids": ["global-user-a"],
                "target_broadcast": False,
            },
        ),
    ):
        result = await persona_supervisor2(state)

    decon_state = m_decon.await_args.args[0]
    research_state = m_research.await_args.args[0]
    assert [
        row["body_text"] for row in decon_state["chat_history_recent"]
    ] == [
        "current user secret",
        "current user reply",
    ]
    assert [
        row["body_text"] for row in research_state["chat_history_wide"]
    ] == [
        "current user secret",
        "current user reply",
    ]
    assert result["consolidation_state"]["chat_history_recent"] == (
        decon_state["chat_history_recent"]
    )


@pytest.mark.asyncio
async def test_persona_supervisor2_no_remember_skips_consolidation():
    """no_remember stays a service concern; supervisor still returns the consolidation snapshot."""
    state = _base_discord_state()
    state["debug_modes"] = {"no_remember": True}

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={"decontexualized_input": "Hello"},
        ) as m_decon,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.stage_1_research",
            new_callable=AsyncMock,
            return_value={"rag_result": {}},
        ) as m_research,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_subgraph",
            new_callable=AsyncMock,
            return_value={
                "internal_monologue": "thinking...",
                "action_directives": {},
                "interaction_subtext": "",
                "emotional_appraisal": "",
                "character_intent": "",
                "logical_stance": "",
            },
        ) as m_cognition,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["Hi there!"],
                "target_addressed_user_ids": ["uuid-123"],
                "target_broadcast": False,
            },
        ) as m_dialog,
    ):
        result = await persona_supervisor2(state)

    assert result["final_dialog"] == ["Hi there!"]
    assert result["target_addressed_user_ids"] == ["uuid-123"]
    assert result["target_broadcast"] is False
    assert result["future_promises"] == []
    assert result["consolidation_state"]["debug_modes"] == {"no_remember": True}
