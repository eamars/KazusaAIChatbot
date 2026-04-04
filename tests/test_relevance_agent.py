"""Tests for Stage 5 — Context Assembler."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json

from kazusa_ai_chatbot.nodes.relevance_agent import (
    _build_history_json,
    _parse_relevance_output,
    relevance_agent,
)
from kazusa_ai_chatbot.state import BotState


def test_parse_relevance_output_success():
    """Valid JSON returns expected parsed fields."""
    raw = '''{
        "channel_topic": "weather",
        "user_topic": "asking for rain",
        "should_respond": false
    }'''
    result = _parse_relevance_output(raw)
    assert result["channel_topic"] == "weather"
    assert result["user_topic"] == "asking for rain"
    assert result["should_respond"] is False


def test_parse_relevance_output_failure():
    """Test fallback parsing on bad JSON."""
    raw = "This is not JSON at all."
    result = _parse_relevance_output(raw)
    assert result["channel_topic"] == "Unknown"
    assert result["should_respond"] is True  # default


def test_build_history_json():
    """Test formatting of chat history into native json format."""
    hist = [
        {"name": "Alice", "content": "Hi", "role": "user"},
        {"name": "Bot", "content": "Hello", "role": "assistant"},
    ]
    
    formatted_hist = _build_history_json(hist, "Bot", "bot_456")
    assert formatted_hist == [
        {"speaker": "Alice", "speaker_id": "unknown_user_id", "message": "Hi"},
        {"speaker": "Bot", "speaker_id": "bot_456", "message": "Hello"}
    ]


@pytest.mark.asyncio
async def test_relevance_agent_node():
    """Test the full relevance_agent node execution."""
    from unittest.mock import AsyncMock, patch
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value.content = '{"channel_topic": "test", "user_topic": "t", "latest_message": "m", "should_respond": true}'
    
    with patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_llm):
        state: BotState = {
            "message_text": "Hello",
            "user_name": "Alice",
            "personality": {"name": "TestBot", "description": "A bot"},
            "rag_results": [{"text": "fact", "source": "doc"}],
            "user_memory": ["likes cats"],
            "conversation_history": [
                {"name": "Alice", "content": "prev", "role": "user"}
            ],
            "character_state": {"mood": "happy"},
            "affinity": 600,
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
    assert human_json["current_message"]["speaker"] == "Alice"
    assert human_json["current_message"]["speaker_id"] == "unknown_user_id"
    assert human_json["current_message"]["message"] == "Hello"
    assert human_json["context"]["rag"][0]["text"] == "fact"
    assert human_json["context"]["rag"][0]["source"] == "doc"
    assert human_json["context"]["user_memory"][0] == "likes cats"
    assert human_json["context"]["conversation_history"][0]["speaker"] == "Alice"
    assert human_json["context"]["conversation_history"][0]["speaker_id"] == "unknown_user_id"
    assert human_json["context"]["conversation_history"][0]["message"] == "prev"

    # Verify outputs
    out = new_state["assembler_output"]
    assert out["channel_topic"] == "test"
    assert out["should_respond"] is True

    # Verify types
    assert isinstance(out["channel_topic"], str)
    assert isinstance(out["user_topic"], str)
    assert isinstance(out["should_respond"], bool)

    # Verify non-empty strings
    assert len(out["channel_topic"]) > 0
    assert len(out["user_topic"]) > 0
