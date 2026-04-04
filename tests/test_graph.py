"""Integration test — full graph execution with all external calls mocked."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.graph import build_graph
from kazusa_ai_chatbot.state import BotState, SupervisorPlan


@pytest.mark.asyncio
async def test_full_graph_question_flow(sample_personality):
    """End-to-end: a question message goes through all stages and produces a response."""
    state: BotState = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "channel_id": "chan_456",
        "guild_id": "guild_789",
        "bot_id": "999888777",
        "message_text": "What happened at the northern gate last night?",
        "timestamp": "2026-03-30T20:00:00Z",
        "should_respond": True,
        "personality": sample_personality,
    }

    # Supervisor LLM: no agents needed
    mock_supervisor_llm = MagicMock()
    mock_supervisor_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "agents": [],
            "content_directive": "Answer the lore question using context.",
            "emotion_directive": "Neutral"
        }))
    )

    mock_speech_llm = MagicMock()
    mock_speech_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="The gate held, Commander.")
    )

    mock_history = [
        {"role": "user", "user_id": "user_123", "name": "TestUser", "content": "Is it safe?"},
    ]
    mock_char_state = {"mood": "alert", "emotional_tone": "guarded", "recent_events": []}

    graph = build_graph()

    # Relevance Agent LLM: analyze topics and decide to respond
    mock_relevance_llm = MagicMock()
    mock_relevance_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "channel_topic": "General",
            "user_topic": "Question",
            "should_respond": True
        }))
    )

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=mock_history),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=["User goes by Commander"]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value=mock_char_state),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=500),
        patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_relevance_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_supervisor_llm),
        patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_speech_llm),
    ):
        result = await graph.ainvoke(state)

    assert result["response"] == "The gate held, Commander."
    assert result["should_respond"] is True


@pytest.mark.asyncio
async def test_full_graph_casual_greeting(sample_personality):
    """A casual greeting fetches context through relevance_agent."""
    state: BotState = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "channel_id": "chan_456",
        "guild_id": "guild_789",
        "bot_id": "999888777",
        "message_text": "Hey",
        "timestamp": "2026-03-30T20:00:00Z",
        "should_respond": True,
        "personality": sample_personality,
    }

    mock_supervisor_llm = MagicMock()
    mock_supervisor_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "agents": [],
            "content_directive": "Respond with a casual greeting.",
            "emotion_directive": "Casual"
        }))
    )

    mock_speech_llm = MagicMock()
    mock_speech_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="Hey there.")
    )

    graph = build_graph()

    mock_relevance_llm = MagicMock()
    mock_relevance_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "channel_topic": "General",
            "user_topic": "Greeting",
            "should_respond": True
        }))
    )

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=500),
        patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_relevance_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_supervisor_llm),
        patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_speech_llm),
    ):
        result = await graph.ainvoke(state)

    assert result["response"] == "Hey there."


@pytest.mark.asyncio
async def test_full_graph_db_lookup_flow(sample_personality):
    """End-to-end: supervisor can dispatch the db lookup agent and still produce a reply."""
    state: BotState = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "channel_id": "chan_456",
        "guild_id": "guild_789",
        "bot_id": "999888777",
        "message_text": "Do you remember what I said about the northern gate?",
        "timestamp": "2026-03-30T20:00:00Z",
        "should_respond": True,
        "personality": sample_personality,
    }

    mock_supervisor_llm = MagicMock()
    mock_supervisor_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "agents": ["db_lookup_agent"],
            "instructions": {
                "db_lookup_agent": {
                    "command": "Search recent conversation history for prior mentions of the northern gate and summarize the relevant continuity.",
                    "expected_response": "Return a short memory brief without raw transcript formatting.",
                }
            },
            "content_directive": "Answer using prior remembered conversation if found.",
            "emotion_directive": "Thoughtful",
        }))
    )

    mock_speech_llm = MagicMock()
    mock_speech_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="You mentioned the northern gate was under pressure, and I still remember that.")
    )

    mock_relevance_llm = MagicMock()
    mock_relevance_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content=json.dumps({
            "channel_topic": "General",
            "user_topic": "Memory check",
            "should_respond": True
        }))
    )

    mock_db_agent = AsyncMock()
    mock_db_agent.run = AsyncMock(return_value={
        "agent": "db_lookup_agent",
        "status": "success",
        "summary": "Earlier in this channel, the user said the northern gate was under pressure.",
        "tool_history": [],
    })

    graph = build_graph()

    with (
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.relevance_agent.get_affinity", new_callable=AsyncMock, return_value=500),
        patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_relevance_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_supervisor_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor.get_agent", return_value=mock_db_agent),
        patch("kazusa_ai_chatbot.nodes.speech_agent._get_llm", return_value=mock_speech_llm),
    ):
        result = await graph.ainvoke(state)

    assert result["response"] == "You mentioned the northern gate was under pressure, and I still remember that."
    assert result["agent_results"][0]["agent"] == "db_lookup_agent"
    assert "northern gate" in result["speech_brief"]["response_brief"]["key_points_to_cover"][-1]


def test_should_respond_after_intake():
    from kazusa_ai_chatbot.graph import _should_respond_after_intake
    from kazusa_ai_chatbot.state import BotState
    from langgraph.graph import END

    # False -> END
    state: BotState = {"should_respond": False}
    assert _should_respond_after_intake(state) == [END]

    # True -> Relevance Agent
    state = {"should_respond": True}
    assert _should_respond_after_intake(state) == ["relevance_agent"]


@pytest.mark.asyncio
async def test_full_graph_empty_message(sample_personality):
    """An empty message after intake stripping should exit early."""
    state: BotState = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "channel_id": "chan_456",
        "guild_id": "guild_789",
        "bot_id": "999888777",
        "message_text": "<@999888777>",
        "timestamp": "2026-03-30T20:00:00Z",
        "should_respond": True,
        "personality": sample_personality,
    }

    graph = build_graph()

    result = await graph.ainvoke(state)

    assert result["should_respond"] is False
    assert "response" not in result or result.get("response", "") == ""
