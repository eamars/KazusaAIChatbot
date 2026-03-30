"""Integration test — full graph execution with all external calls mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from bot.graph import build_graph
from bot.state import BotState


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

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
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

    with (
        patch("bot.nodes.rag._get_embed_client", return_value=mock_embed_client),
        patch("bot.nodes.rag.vector_search", new_callable=AsyncMock, return_value=mock_vector_results),
        patch("bot.nodes.memory.get_conversation_history", new_callable=AsyncMock, return_value=mock_history),
        patch("bot.nodes.memory.get_user_facts", new_callable=AsyncMock, return_value=["User goes by Commander"]),
        patch("bot.nodes.memory.get_character_state", new_callable=AsyncMock, return_value=mock_char_state),
        patch("bot.nodes.memory.get_affinity", new_callable=AsyncMock, return_value=500),
        patch("bot.nodes.persona._get_llm", return_value=mock_llm),
    ):
        result = await graph.ainvoke(state)

    assert result["response"] == "The gate held, Commander."
    assert result["should_respond"] is True
    assert result["retrieve_rag"] is True


@pytest.mark.asyncio
async def test_full_graph_casual_greeting(sample_personality):
    """A casual greeting skips RAG but still fetches memory."""
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

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Hey there."))

    graph = build_graph()

    with (
        patch("bot.nodes.memory.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("bot.nodes.memory.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("bot.nodes.memory.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("bot.nodes.memory.get_affinity", new_callable=AsyncMock, return_value=500),
        patch("bot.nodes.persona._get_llm", return_value=mock_llm),
    ):
        result = await graph.ainvoke(state)

    assert result["response"] == "Hey there."
    assert result["retrieve_rag"] is False
    assert result["retrieve_memory"] is True


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
    result = await graph.ainvoke(state)

    assert result["should_respond"] is False
    assert "response" not in result or result.get("response", "") == ""
