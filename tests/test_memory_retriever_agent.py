"""Tests for memory_retriever_agent.py — memory search tools and agent orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot.agents.memory_retriever_agent import (
    search_user_facts,
    search_conversation,
    get_conversation,
    search_persistent_memory,
    _ALL_TOOLS,
    _TOOLS_BY_NAME,
)


class TestToolRegistration:
    def test_all_tools_list_has_4_tools(self):
        assert len(_ALL_TOOLS) == 4

    def test_tools_by_name_matches(self):
        assert set(_TOOLS_BY_NAME.keys()) == {t.name for t in _ALL_TOOLS}

    def test_expected_tool_names(self):
        expected = {"search_user_facts", "search_conversation", "get_conversation", "search_persistent_memory"}
        assert set(_TOOLS_BY_NAME.keys()) == expected


@pytest.mark.asyncio
async def test_search_user_facts_tool():
    """search_user_facts tool should delegate to db.get_user_facts."""
    with patch("kazusa_ai_chatbot.agents.memory_retriever_agent.get_user_facts", new_callable=AsyncMock, return_value=["fact1", "fact2"]):
        result = await search_user_facts.ainvoke({"user_id": "user_123"})

    assert result == ["fact1", "fact2"]


@pytest.mark.asyncio
async def test_search_conversation_tool():
    """search_conversation tool should delegate to db.search_conversation_history."""
    mock_results = [
        (0.9, {"content": "hello", "timestamp": "t1", "channel_id": "c1", "user_id": "u1"}),
    ]
    with patch("kazusa_ai_chatbot.agents.memory_retriever_agent.search_conversation_history", new_callable=AsyncMock, return_value=mock_results):
        result = await search_conversation.ainvoke({"search_query": "hello"})

    assert isinstance(result, list)
    assert len(result) == 1
    # Returns tuples of (score, message_dict)
    score, doc = result[0]
    assert score == 0.9
    assert doc["content"] == "hello"


@pytest.mark.asyncio
async def test_get_conversation_tool():
    """get_conversation tool should delegate to db.get_conversation_history."""
    mock_msgs = [
        {"content": "hi", "timestamp": "t1", "channel_id": "c1", "user_id": "u1", "embedding": [0.1]},
    ]
    with patch("kazusa_ai_chatbot.agents.memory_retriever_agent.get_conversation_history", new_callable=AsyncMock, return_value=mock_msgs):
        result = await get_conversation.ainvoke({})

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["content"] == "hi"


@pytest.mark.asyncio
async def test_search_persistent_memory_tool():
    """search_persistent_memory tool should delegate to db.search_memory."""
    mock_results = [
        (0.85, {"memory_name": "test_mem", "content": "some data", "timestamp": "t1"}),
    ]
    with patch("kazusa_ai_chatbot.agents.memory_retriever_agent.search_memory_db", new_callable=AsyncMock, return_value=mock_results):
        result = await search_persistent_memory.ainvoke({"search_query": "test"})

    assert isinstance(result, list)
    assert len(result) == 1
    assert "cosine_similarity" in result[0]
    assert result[0]["cosine_similarity"] == 0.85
