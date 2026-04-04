"""Tests for the relevance agent and its context-loading behavior."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json

from kazusa_ai_chatbot.nodes.relevance_agent import (
    _load_context,
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
    assert human_json["current_message"]["speaker"] == "Alice"
    assert human_json["current_message"]["speaker_id"] == "user_123"
    assert human_json["current_message"]["message"] == "Hello"
    assert human_json["context"]["user_memory"][0] == "likes cats"
    assert human_json["context"]["conversation_history"][0]["speaker"] == "Alice"
    assert human_json["context"]["conversation_history"][0]["speaker_id"] == "user_123"
    assert human_json["context"]["conversation_history"][0]["message"] == "prev"
    assert human_json["context"]["character_state"]["mood"] == "happy"
    assert human_json["context"]["affinity"] == 600

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


@pytest.mark.asyncio
async def test_relevance_agent_handles_context_fetch_failures():
    from unittest.mock import AsyncMock, patch

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value.content = '{"channel_topic": "Unknown", "user_topic": "Unknown", "should_respond": true}'

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, side_effect=Exception("history error")),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, side_effect=Exception("facts error")),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, side_effect=Exception("state error")),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, side_effect=Exception("affinity error")),
        patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_llm),
    ):
        state: BotState = {
            "message_text": "Hello",
            "user_name": "Alice",
            "user_id": "user_123",
            "channel_id": "channel_1",
            "personality": {"name": "TestBot"},
        }

        new_state = await relevance_agent(state)

    assert new_state["conversation_history"] == []
    assert new_state["user_memory"] == []
    assert new_state["character_state"] == {}
    assert new_state["affinity"] == 500


@pytest.mark.asyncio
async def test_load_context_returns_history_and_facts(base_state):
    from unittest.mock import AsyncMock, patch

    mock_history = [
        {"role": "user", "user_id": "user_123", "name": "TestUser", "content": "Hello"},
        {"role": "assistant", "user_id": "bot_001", "name": "bot", "content": "Hi there"},
    ]
    mock_facts = ["User likes swords"]
    mock_char_state = {"mood": "calm", "emotional_tone": "warm", "recent_events": []}

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=mock_history),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=mock_facts),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value=mock_char_state),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=700),
    ):
        history, user_memory, character_state, affinity = await _load_context(base_state)

    assert len(history) == 2
    assert user_memory == ["User likes swords"]
    assert character_state["mood"] == "calm"
    assert affinity == 700


@pytest.mark.asyncio
async def test_load_context_handles_history_failure(base_state):
    from unittest.mock import AsyncMock, patch

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, side_effect=Exception("db error")),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=500),
    ):
        history, user_memory, character_state, affinity = await _load_context(base_state)

    assert history == []
    assert user_memory == []
    assert character_state == {
        "mood": "neutral",
        "emotional_tone": "balanced",
        "recent_events": [],
        "updated_at": "",
    }
    assert affinity == 500


@pytest.mark.asyncio
async def test_load_context_handles_facts_failure(base_state):
    from unittest.mock import AsyncMock, patch

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, side_effect=Exception("db error")),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=500),
    ):
        history, user_memory, character_state, affinity = await _load_context(base_state)

    assert history == []
    assert user_memory == []
    assert character_state == {
        "mood": "neutral",
        "emotional_tone": "balanced",
        "recent_events": [],
        "updated_at": "",
    }
    assert affinity == 500


@pytest.mark.asyncio
async def test_load_context_handles_character_state_failure(base_state):
    from unittest.mock import AsyncMock, patch

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, side_effect=Exception("db error")),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=500),
    ):
        history, user_memory, character_state, affinity = await _load_context(base_state)

    assert history == []
    assert user_memory == []
    assert character_state == {}
    assert affinity == 500


@pytest.mark.asyncio
async def test_load_context_handles_affinity_failure(base_state):
    from unittest.mock import AsyncMock, patch

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, side_effect=Exception("db error")),
    ):
        history, user_memory, character_state, affinity = await _load_context(base_state)

    assert history == []
    assert user_memory == []
    assert character_state == {
        "mood": "neutral",
        "emotional_tone": "balanced",
        "recent_events": [],
        "updated_at": "",
    }
    assert affinity == 500
