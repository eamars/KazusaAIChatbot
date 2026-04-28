"""Tests for RAG memory retrieval tool wrappers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kazusa_ai_chatbot.rag.memory_retrieval_tools import (
    get_conversation,
    search_conversation,
    search_conversation_keyword,
    search_persistent_memory,
    search_persistent_memory_keyword,
)


@pytest.mark.asyncio
async def test_search_conversation_delegates_to_vector_history_search() -> None:
    """search_conversation should call conversation search in vector mode."""
    mock_results = [
        (
            0.9,
            {
                "content": "hello",
                "timestamp": "t1",
                "display_name": "User",
                "role": "user",
                "platform": "qq",
                "platform_channel_id": "channel-1",
                "platform_message_id": "message-1",
                "platform_user_id": "platform-user-1",
                "global_user_id": "global-user-1",
                "reply_context": {"content": "previous"},
                "embedding": [0.1],
            },
        ),
    ]

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.search_conversation_history",
        new_callable=AsyncMock,
        return_value=mock_results,
    ) as search_history:
        result = await search_conversation.ainvoke(
            {
                "search_query": "friendly greeting",
                "global_user_id": "global-user-1",
                "platform": "qq",
                "platform_channel_id": "channel-1",
                "from_timestamp": "2026-04-01T00:00:00Z",
                "to_timestamp": "2026-04-02T00:00:00Z",
            }
        )

    search_history.assert_awaited_once_with(
        query="friendly greeting",
        platform="qq",
        platform_channel_id="channel-1",
        global_user_id="global-user-1",
        limit=5,
        method="vector",
        from_timestamp="2026-04-01T00:00:00Z",
        to_timestamp="2026-04-02T00:00:00Z",
    )
    assert result == [
        (
            0.9,
            {
                "content": "hello",
                "timestamp": "t1",
                "display_name": "User",
                "role": "user",
                "platform": "qq",
                "platform_channel_id": "channel-1",
                "platform_message_id": "message-1",
                "platform_user_id": "platform-user-1",
                "global_user_id": "global-user-1",
                "reply_context": {"content": "previous"},
            },
        )
    ]


@pytest.mark.asyncio
async def test_search_conversation_rejects_empty_query() -> None:
    """search_conversation should fail fast when no semantic query is provided."""
    result = await search_conversation.ainvoke({"search_query": "   "})

    assert result == [
        (
            -1.0,
            {
                "error": (
                    "search_query is mandatory and must not be empty. "
                    "Please provide a natural-language semantic query."
                )
            },
        )
    ]


@pytest.mark.asyncio
async def test_search_conversation_keyword_delegates_to_keyword_history_search() -> None:
    """search_conversation_keyword should call conversation search in keyword mode."""
    mock_results = [
        (1.0, {"content": "DDR5 came up", "timestamp": "t1", "display_name": "User"}),
    ]

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.search_conversation_history",
        new_callable=AsyncMock,
        return_value=mock_results,
    ) as search_history:
        result = await search_conversation_keyword.ainvoke({"keyword": "DDR5", "top_k": 3})

    search_history.assert_awaited_once_with(
        query="DDR5",
        platform=None,
        platform_channel_id=None,
        global_user_id=None,
        limit=3,
        method="keyword",
        from_timestamp=None,
        to_timestamp=None,
    )
    assert result[0]["content"] == "DDR5 came up"
    assert result[0]["display_name"] == "User"


@pytest.mark.asyncio
async def test_get_conversation_filters_and_strips_internal_fields() -> None:
    """get_conversation should pass structured filters and hide embedding data."""
    mock_messages = [
        {
            "content": "hi",
            "timestamp": "t1",
            "display_name": "User",
            "role": "user",
            "platform": "qq",
            "platform_channel_id": "channel-1",
            "platform_message_id": "message-1",
            "platform_user_id": "platform-user-1",
            "global_user_id": "global-user-1",
            "reply_context": {},
            "embedding": [0.1],
        },
    ]

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.get_conversation_history",
        new_callable=AsyncMock,
        return_value=mock_messages,
    ) as get_history:
        result = await get_conversation.ainvoke(
            {
                "platform": "qq",
                "platform_channel_id": "channel-1",
                "limit": 2,
                "global_user_id": "global-user-1",
            }
        )

    get_history.assert_awaited_once_with(
        platform="qq",
        platform_channel_id="channel-1",
        limit=2,
        global_user_id="global-user-1",
        display_name=None,
        from_timestamp=None,
        to_timestamp=None,
    )
    assert result[0]["content"] == "hi"
    assert "embedding" not in result[0]


@pytest.mark.asyncio
async def test_search_persistent_memory_delegates_to_vector_memory_search() -> None:
    """search_persistent_memory should call vector search without type filtering."""
    mock_results = [
        (
            0.85,
            {
                "memory_name": "preference",
                "content": "likes quiet conversation",
                "timestamp": "t1",
                "source_global_user_id": "global-user-1",
                "memory_type": "preference",
                "source_kind": "conversation_extracted",
                "status": "active",
            },
        ),
    ]

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.search_memory_db",
        new_callable=AsyncMock,
        return_value=mock_results,
    ) as search_memory:
        result = await search_persistent_memory.ainvoke(
            {
                "search_query": "quiet conversation preference",
                "top_k": 4,
                "source_global_user_id": "global-user-1",
                "memory_type": "preference",
                "status": "active",
            }
        )

    search_memory.assert_awaited_once_with(
        query="quiet conversation preference",
        limit=4,
        method="vector",
        source_global_user_id="global-user-1",
        memory_type=None,
        source_kind=None,
        status="active",
        expiry_before=None,
        expiry_after=None,
    )
    assert result[0]["cosine_similarity"] == 0.85
    assert result[0]["content"] == "likes quiet conversation"


@pytest.mark.asyncio
async def test_search_persistent_memory_keyword_delegates_to_keyword_memory_search() -> None:
    """search_persistent_memory_keyword should call memory search in keyword mode."""
    mock_results = [
        (1.0, {"memory_name": "hardware", "content": "DDR5 discussion", "timestamp": "t1"}),
    ]

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.search_memory_db",
        new_callable=AsyncMock,
        return_value=mock_results,
    ) as search_memory:
        result = await search_persistent_memory_keyword.ainvoke({"keyword": "DDR5"})

    search_memory.assert_awaited_once_with(
        query="DDR5",
        limit=5,
        method="keyword",
        source_global_user_id=None,
        memory_type=None,
    )
    assert result[0]["memory_name"] == "hardware"
    assert result[0]["content"] == "DDR5 discussion"
