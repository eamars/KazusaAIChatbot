"""Tests for Stage 4 — Memory Retriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kazusa_ai_chatbot.nodes.memory import memory_retriever


@pytest.mark.asyncio
async def test_memory_skipped_when_flag_false(routed_state):
    routed_state["retrieve_memory"] = False
    result = await memory_retriever(routed_state)
    assert result["conversation_history"] == []
    assert result["user_memory"] == []
    assert result["character_state"] == {}
    assert result["affinity"] == 500


@pytest.mark.asyncio
async def test_memory_returns_history_and_facts(routed_state):
    mock_history = [
        {"role": "user", "user_id": "user_123", "name": "TestUser", "content": "Hello"},
        {"role": "assistant", "user_id": "bot_001", "name": "bot", "content": "Hi there"},
    ]
    mock_facts = ["User likes swords"]
    mock_char_state = {"mood": "calm", "emotional_tone": "warm", "recent_events": []}

    with (
        patch("kazusa_ai_chatbot.nodes.memory.get_conversation_history", new_callable=AsyncMock, return_value=mock_history),
        patch("kazusa_ai_chatbot.nodes.memory.get_user_facts", new_callable=AsyncMock, return_value=mock_facts),
        patch("kazusa_ai_chatbot.nodes.memory.get_character_state", new_callable=AsyncMock, return_value=mock_char_state),
        patch("kazusa_ai_chatbot.nodes.memory.get_affinity", new_callable=AsyncMock, return_value=700),
    ):
        result = await memory_retriever(routed_state)

    assert len(result["conversation_history"]) == 2
    assert result["user_memory"] == ["User likes swords"]
    assert result["character_state"]["mood"] == "calm"
    assert result["affinity"] == 700


@pytest.mark.asyncio
async def test_memory_handles_history_failure(routed_state):
    with (
        patch("kazusa_ai_chatbot.nodes.memory.get_conversation_history", new_callable=AsyncMock, side_effect=Exception("db error")),
        patch("kazusa_ai_chatbot.nodes.memory.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.memory.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.memory.get_affinity", new_callable=AsyncMock, return_value=500),
    ):
        result = await memory_retriever(routed_state)

    assert result["conversation_history"] == []
    assert result["user_memory"] == []


@pytest.mark.asyncio
async def test_memory_handles_facts_failure(routed_state):
    with (
        patch("kazusa_ai_chatbot.nodes.memory.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.memory.get_user_facts", new_callable=AsyncMock, side_effect=Exception("db error")),
        patch("kazusa_ai_chatbot.nodes.memory.get_character_state", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.memory.get_affinity", new_callable=AsyncMock, return_value=500),
    ):
        result = await memory_retriever(routed_state)

    assert result["conversation_history"] == []
    assert result["user_memory"] == []


@pytest.mark.asyncio
async def test_memory_handles_character_state_failure(routed_state):
    with (
        patch("kazusa_ai_chatbot.nodes.memory.get_conversation_history", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.memory.get_user_facts", new_callable=AsyncMock, return_value=[]),
        patch("kazusa_ai_chatbot.nodes.memory.get_character_state", new_callable=AsyncMock, side_effect=Exception("db error")),
        patch("kazusa_ai_chatbot.nodes.memory.get_affinity", new_callable=AsyncMock, return_value=500),
    ):
        result = await memory_retriever(routed_state)

    assert result["character_state"] == {}


@pytest.mark.asyncio
async def test_memory_returns_only_owned_fields(routed_state):
    """Parallel fan-out safety: memory should NOT return the full state."""
    routed_state["retrieve_memory"] = False
    result = await memory_retriever(routed_state)
    assert "user_id" not in result
    assert "conversation_history" in result
    assert "user_memory" in result
    assert "character_state" in result
    assert "affinity" in result
