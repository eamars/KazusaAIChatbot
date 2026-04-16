"""Tests for relevance_agent.py — relevance gate logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from kazusa_ai_chatbot.nodes.relevance_agent import relevance_agent


def _base_state():
    """Minimal IMProcessState for testing relevance_agent."""
    return {
        "timestamp": "2024-01-01T00:00:00Z",
        "platform": "discord",
        "platform_user_id": "user_123",
        "global_user_id": "uuid-123",
        "user_name": "TestUser",
        "user_input": "Hello bot!",
        "user_multimedia_input": [],
        "user_profile": {"affinity": 500, "last_relationship_insight": ""},
        "platform_bot_id": "bot_456",
        "bot_name": "TestBot",
        "character_profile": {"name": "Kazusa"},
        "character_state": {
            "mood": "neutral",
            "global_vibe": "calm",
            "reflection_summary": "nothing notable",
        },
        "platform_channel_id": "chan_1",
        "channel_name": "general",
        "chat_history": [],
    }


@pytest.mark.asyncio
async def test_relevance_agent_returns_should_respond():
    """LLM says should_respond=true → agent forwards that decision."""
    llm_response = MagicMock()
    llm_response.content = '{"should_respond": true, "reason_to_respond": "user greeted", "use_reply_feature": false, "channel_topic": "greetings", "user_topic": "hello"}'

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["should_respond"] is True
    assert result["channel_topic"] == "greetings"
    assert result["user_topic"] == "hello"


@pytest.mark.asyncio
async def test_relevance_agent_should_not_respond():
    """LLM says should_respond=false → agent forwards that decision."""
    llm_response = MagicMock()
    llm_response.content = '{"should_respond": false, "reason_to_respond": "third party conversation", "use_reply_feature": false, "channel_topic": "sports", "user_topic": "football"}'

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["should_respond"] is False
    assert result["reason_to_respond"] == "third party conversation"


@pytest.mark.asyncio
async def test_relevance_agent_malformed_json_defaults_to_not_respond():
    """If LLM returns garbage JSON, parse_llm_json_output returns {} and should_respond defaults to False."""
    llm_response = MagicMock()
    llm_response.content = "this is not json at all"

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["should_respond"] is False
    assert result["channel_topic"] == ""
    assert result["user_topic"] == ""


@pytest.mark.asyncio
async def test_relevance_agent_use_reply_feature():
    """LLM says use_reply_feature=true → agent forwards it."""
    llm_response = MagicMock()
    llm_response.content = '{"should_respond": true, "reason_to_respond": "reply needed", "use_reply_feature": true, "channel_topic": "topic", "user_topic": "sub"}'

    with patch("kazusa_ai_chatbot.nodes.relevance_agent._relevance_agent_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)
        result = await relevance_agent(_base_state())

    assert result["use_reply_feature"] is True
