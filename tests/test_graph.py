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
        "message_text": "What happened at the northern gate last night?",
        "timestamp": "2026-03-30T20:00:00Z",
        "should_respond": True,
        "personality": sample_personality,
    }

    mock_embed_resp = MagicMock()
    mock_embed_resp.data = [MagicMock(embedding=[0.1] * 128)]

    mock_embed_client = AsyncMock()
    mock_embed_client.embeddings.create.return_value = mock_embed_resp

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

    mock_vector_results = [
        {"text": "Shadow wolves attacked.", "source": "lore", "score": 0.9},
    ]
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
            "latest_message": "What happened at the northern gate?",
            "should_respond": True
        }))
    )

    with (
        patch("kazusa_ai_chatbot.db.get_text_embedding", return_value=mock_embed_client),
        patch("kazusa_ai_chatbot.db.search_lore", new_callable=AsyncMock, return_value=mock_vector_results),
        patch("kazusa_ai_chatbot.nodes.memory.get_conversation_history", new_callable=AsyncMock, return_value=mock_history),
        patch("kazusa_ai_chatbot.nodes.memory.get_user_facts", new_callable=AsyncMock, return_value=["User goes by Commander"]),
        patch("kazusa_ai_chatbot.nodes.memory.get_character_state", new_callable=AsyncMock, return_value=mock_char_state),
        patch("kazusa_ai_chatbot.nodes.memory.get_affinity", new_callable=AsyncMock, return_value=500),
        patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_relevance_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_supervisor_llm),
        patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_speech_llm),
    ):
        result = await graph.ainvoke(state)

    assert result["response"] == "The gate held, Commander."
    assert result["should_respond"] is True


@pytest.mark.asyncio
async def test_full_graph_casual_greeting(sample_personality):
    """A casual greeting fetches memory and RAG."""
    state: BotState = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "channel_id": "chan_456",
        "guild_id": "guild_789",
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
        patch("kazusa_ai_chatbot.nodes.rag.get_text_embedding", new_callable=AsyncMock, return_value=[0.1] * 128),
        patch("kazusa_ai_chatbot.nodes.rag.search_lore", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.memory.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.memory.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.memory.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.memory.get_affinity", new_callable=AsyncMock, return_value=500),
        patch("kazusa_ai_chatbot.nodes.relevance_agent._get_llm", return_value=mock_relevance_llm),
        patch("kazusa_ai_chatbot.nodes.persona_supervisor._get_llm", return_value=mock_supervisor_llm),
        patch("kazusa_ai_chatbot.agents.speech_agent._get_llm", return_value=mock_speech_llm),
    ):
        result = await graph.ainvoke(state)

    assert result["response"] == "Hey there."


def test_should_respond_and_retrieve():
    from kazusa_ai_chatbot.graph import _should_respond_and_retrieve
    from kazusa_ai_chatbot.state import BotState
    from langgraph.graph import END

    # False -> END
    state: BotState = {"should_respond": False}
    assert _should_respond_and_retrieve(state) == [END]

    # True -> Both RAG and Memory
    state = {"should_respond": True}
    assert set(_should_respond_and_retrieve(state)) == {"memory_retriever", "rag_retriever"}


@pytest.mark.asyncio
async def test_full_graph_empty_message(sample_personality):
    """An empty message after intake stripping should exit early."""
    state: BotState = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "channel_id": "chan_456",
        "guild_id": "guild_789",
        "message_text": "<@12345>",
        "timestamp": "2026-03-30T20:00:00Z",
        "should_respond": True,
        "personality": sample_personality,
    }

    graph = build_graph()
    
    # We only mock intake if we want it to strip to empty,
    # or just rely on intake returning should_respond=False
    with patch("kazusa_ai_chatbot.nodes.intake.intake") as mock_intake:
        mock_intake.return_value = {**state, "should_respond": False}
        result = await graph.ainvoke(state)

    assert result["should_respond"] is False
    assert "response" not in result or result.get("response", "") == ""
