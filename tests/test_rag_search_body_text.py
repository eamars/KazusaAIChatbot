"""Tests that RAG conversation tools expose body_text instead of raw content."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kazusa_ai_chatbot.rag.memory_retrieval_tools import (
    get_conversation,
    search_conversation,
    search_conversation_keyword,
)


@pytest.mark.asyncio
async def test_semantic_conversation_tool_returns_body_text() -> None:
    """Semantic search results should expose clean body text to RAG agents."""

    mock_results = [
        (
            0.9,
            {
                "body_text": "clean semantic text",
                "timestamp": "t1",
                "display_name": "User",
                "role": "user",
            },
        ),
    ]

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.search_conversation_history",
        new_callable=AsyncMock,
        return_value=mock_results,
    ):
        result = await search_conversation.ainvoke({
            "search_query": "clean semantic text",
        })

    assert result[0][1]["body_text"] == "clean semantic text"
    assert "content" not in result[0][1]


@pytest.mark.asyncio
async def test_keyword_conversation_tool_returns_body_text() -> None:
    """Keyword search results should expose clean body text to RAG agents."""

    mock_results = [
        (
            -1.0,
            {
                "body_text": "keyword",
                "timestamp": "t1",
                "display_name": "User",
                "role": "user",
            },
        ),
    ]

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.search_conversation_history",
        new_callable=AsyncMock,
        return_value=mock_results,
    ):
        result = await search_conversation_keyword.ainvoke({"keyword": "keyword"})

    assert result[0]["body_text"] == "keyword"
    assert "content" not in result[0]


@pytest.mark.asyncio
async def test_get_conversation_tool_returns_body_text() -> None:
    """Structured conversation fetches should expose clean body text."""

    mock_messages = [
        {
            "body_text": "fetched",
            "timestamp": "t1",
            "display_name": "User",
            "role": "user",
        },
    ]

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.get_conversation_history",
        new_callable=AsyncMock,
        return_value=mock_messages,
    ):
        result = await get_conversation.ainvoke({"platform": "qq"})

    assert result[0]["body_text"] == "fetched"
    assert "content" not in result[0]
