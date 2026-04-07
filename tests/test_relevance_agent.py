"""Tests for the relevance agent and its context-loading behavior."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json

from kazusa_ai_chatbot.nodes.relevance_agent import (
    relevance_agent
)
from kazusa_ai_chatbot.state import BotState
from kazusa_ai_chatbot.db import AFFINITY_DEFAULT


@pytest.mark.asyncio
async def test_relevance_agent_node():
    """Test the full relevance_agent node execution."""
    from unittest.mock import AsyncMock, patch
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value.content = '{"channel_topic": "test", "user_topic": "t", "should_respond": true}'
    
    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=[
            {"name": "Alice", "user_id": "user_123", "content": "prev", "role": "user"}
        ]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=["likes cats"]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value={"mood": "happy"}),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=600),
        patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_llm),
    ):
        state: BotState = {
            "message_text": "Hello",
            "user_name": "Alice",
            "user_id": "user_123",
            "channel_id": "channel_1",
            "personality": {"name": "TestBot", "description": "A bot"},
        }

        new_state = await relevance_agent(state)

    # Verify LLM was called
    assert mock_llm.ainvoke.called
    
    # Ensure correct structure was passed to the LLM (HumanMessage content)
    args, kwargs = mock_llm.ainvoke.call_args
    analysis_messages = args[0]
    
    assert len(analysis_messages) == 2
    assert isinstance(analysis_messages[0], SystemMessage)
    assert isinstance(analysis_messages[1], HumanMessage)
    
    human_json = json.loads(analysis_messages[1].content)
    assert human_json["current_message"]["name"] == "Alice"
    assert human_json["current_message"]["user_id"] == "user_123"
    assert human_json["current_message"]["content"] == "Hello"
    assert human_json["context"]["conversation_history"][0]["name"] == "Alice"
    assert human_json["context"]["conversation_history"][0]["user_id"] == "user_123"
    assert human_json["context"]["conversation_history"][0]["content"] == "prev"
    assert human_json["context"]["conversation_history"][0]["role"] == "user"
    # After fixing relationship input to pass raw affinity value
    assert human_json["context"]["relationship"] == 600

    # Verify outputs
    out = new_state["assembler_output"]
    assert out["channel_topic"] == "test"
    assert out["should_respond"] is True
    assert new_state["conversation_history"][0]["content"] == "prev"
    assert new_state["user_memory"] == ["likes cats"]
    assert new_state["character_state"]["mood"] == "happy"
    assert new_state["affinity"] == 600

    # Verify types
    assert isinstance(out["channel_topic"], str)
    assert isinstance(out["user_topic"], str)
    assert isinstance(out["should_respond"], bool)

    # Verify non-empty strings
    assert len(out["channel_topic"]) > 0
    assert len(out["user_topic"]) > 0
