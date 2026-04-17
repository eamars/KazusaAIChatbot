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
        "platform_user_id": "user_123",
        "global_user_id": "uuid-123",
        "user_input": "Hello",
        "user_multimedia_input": [],
        "user_profile": {"affinity": 500},
        "platform_bot_id": "bot_456",
        "bot_name": "TestBot",
        "character_profile": {
            "name": "Kazusa",
            "mood": "neutral",
            "global_vibe": "calm",
            "reflection_summary": "nothing notable",
        },
        "platform_channel_id": "chan_1",
        "channel_name": "general",
        "chat_history": [],
        "should_respond": True,
        "reason_to_respond": "user greeted",
        "use_reply_feature": False,
        "channel_topic": "greetings",
        "user_topic": "hello",
    }


@pytest.mark.asyncio
async def test_call_action_subgraph_returns_final_dialog():
    """call_action_subgraph wraps dialog_agent output correctly."""
    mock_dialog_result = {"final_dialog": ["Hello!", "How are you?"]}

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent", new_callable=AsyncMock, return_value=mock_dialog_result):
        result = await call_action_subgraph({"any": "state"})

    assert result["final_dialog"] == ["Hello!", "How are you?"]


@pytest.mark.asyncio
async def test_call_action_subgraph_empty_dialog():
    """call_action_subgraph handles empty dialog_agent output."""
    mock_dialog_result = {}

    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent", new_callable=AsyncMock, return_value=mock_dialog_result):
        result = await call_action_subgraph({"any": "state"})

    assert result["final_dialog"] == []


@pytest.mark.asyncio
async def test_persona_supervisor2_returns_final_dialog_and_future_promises():
    """Full persona_supervisor2 call returns expected keys when all stages are mocked."""
    state = _base_discord_state()

    # Mock all 5 stage functions to avoid real LLM calls
    with patch("kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer", new_callable=AsyncMock, return_value={"decontexualized_input": "Hello"}) as m_decon, \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor2.call_rag_subgraph", new_callable=AsyncMock, return_value={"research_facts": "", "research_metadata": []}) as m_research, \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_subgraph", new_callable=AsyncMock, return_value={
             "internal_monologue": "thinking...",
             "action_directives": {},
             "interaction_subtext": "",
             "emotional_appraisal": "",
             "character_intent": "",
             "logical_stance": "",
         }) as m_cognition, \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent", new_callable=AsyncMock, return_value={"final_dialog": ["Hi there!"]}) as m_dialog, \
         patch("kazusa_ai_chatbot.nodes.persona_supervisor2.call_consolidation_subgraph", new_callable=AsyncMock, return_value={
             "mood": "happy",
             "global_vibe": "warm",
             "reflection_summary": "good chat",
             "diary_entry": [],
             "affinity_delta": 5,
             "last_relationship_insight": "friendly",
             "new_facts": [],
             "future_promises": [{"promise": "remember birthday"}],
         }) as m_consol:
        result = await persona_supervisor2(state)

    assert "final_dialog" in result
    assert "future_promises" in result
    assert result["final_dialog"] == ["Hi there!"]
