"""Tests for RAG memory retrieval tool wrappers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kazusa_ai_chatbot.config import (
    CONVERSATION_SEARCH_DEFAULT_TOP_K,
    CONVERSATION_SEARCH_MAX_TOP_K,
    RAG_SEARCH_DEFAULT_TOP_K,
)
from kazusa_ai_chatbot.rag.memory_retrieval_tools import (
    conversation_message_payload,
    get_conversation,
    search_conversation,
    search_conversation_keyword,
    search_persistent_memory,
    search_persistent_memory_keyword,
)


class _FakeObjectId:
    """Small ObjectId-like value for string-cast projection tests."""

    def __str__(self) -> str:
        """Return the stable string form used by Mongo row identity."""

        return_value = "row-object-id"
        return return_value


@pytest.mark.asyncio
async def test_search_conversation_delegates_to_vector_history_search() -> None:
    """search_conversation should call conversation search in vector mode."""
    mock_results = [
        (
            0.9,
            {
                "_id": _FakeObjectId(),
                "body_text": "hello",
                "timestamp": "t1",
                "display_name": "User",
                "role": "user",
                "platform": "qq",
                "platform_channel_id": "channel-1",
                "platform_message_id": "message-1",
                "platform_user_id": "platform-user-1",
                "global_user_id": "global-user-1",
                "reply_context": {
                    "reply_excerpt": "previous",
                    "reply_attachments": [
                        {
                            "media_kind": "image",
                            "description": "reply image",
                            "base64_data": "reply-bytes",
                        }
                    ],
                },
                "attachments": [
                    {
                        "media_kind": "image",
                        "description": "chart image",
                        "url": "https://example.test/chart.png",
                        "base64_data": "inline-bytes",
                    }
                ],
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
        limit=CONVERSATION_SEARCH_DEFAULT_TOP_K,
        method="vector",
        from_timestamp="2026-04-01T00:00:00Z",
        to_timestamp="2026-04-02T00:00:00Z",
    )
    assert result[0][0] == 0.9
    payload = result[0][1]
    assert payload["body_text"] == "hello\n<image>chart image</image>"
    assert payload["timestamp"] == ""
    assert payload["display_name"] == "User"
    assert payload["role"] == "user"
    assert payload["platform"] == "qq"
    assert payload["platform_channel_id"] == "channel-1"
    assert payload["platform_user_id"] == "platform-user-1"
    assert payload["global_user_id"] == "global-user-1"
    assert payload["reply_context"] == {
        "reply_excerpt": "previous\n<image>reply image</image>",
    }
    rendered = repr(payload)
    assert "conversation_row_id" not in payload
    assert "platform_message_id" not in payload
    assert "https://example.test/chart.png" not in rendered
    assert "base64_data" not in rendered


@pytest.mark.asyncio
async def test_search_conversation_clamps_too_small_direct_top_k() -> None:
    """Direct tool calls should not bypass the configured semantic top-k floor."""

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.search_conversation_history",
        new_callable=AsyncMock,
        return_value=[],
    ) as search_history:
        await search_conversation.ainvoke(
            {
                "search_query": "friendly greeting",
                "top_k": 3,
            }
        )

    assert (
        search_history.await_args.kwargs["limit"]
        == CONVERSATION_SEARCH_DEFAULT_TOP_K
    )


@pytest.mark.asyncio
async def test_search_conversation_clamps_too_large_direct_top_k() -> None:
    """Direct tool calls should not bypass the configured semantic top-k cap."""

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.search_conversation_history",
        new_callable=AsyncMock,
        return_value=[],
    ) as search_history:
        await search_conversation.ainvoke(
            {
                "search_query": "friendly greeting",
                "top_k": CONVERSATION_SEARCH_MAX_TOP_K + 1,
            }
        )

    assert search_history.await_args.kwargs["limit"] == CONVERSATION_SEARCH_MAX_TOP_K


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
        (
            1.0,
            {
                "_id": _FakeObjectId(),
                "body_text": "DDR5 came up",
                "timestamp": "t1",
                "display_name": "User",
            },
        ),
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
    assert result[0]["body_text"] == "DDR5 came up"
    assert "conversation_row_id" not in result[0]
    assert "content" not in result[0]
    assert result[0]["display_name"] == "User"


@pytest.mark.asyncio
async def test_search_conversation_keyword_uses_shared_default_top_k() -> None:
    """Keyword conversation retrieval should use the shared RAG top-k default."""

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.search_conversation_history",
        new_callable=AsyncMock,
        return_value=[],
    ) as search_history:
        await search_conversation_keyword.ainvoke({"keyword": "DDR5"})

    assert search_history.await_args.kwargs["limit"] == RAG_SEARCH_DEFAULT_TOP_K


@pytest.mark.asyncio
async def test_get_conversation_filters_and_strips_internal_fields() -> None:
    """get_conversation should pass structured filters and hide embedding data."""
    mock_messages = [
        {
            "_id": _FakeObjectId(),
            "body_text": "hi",
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
    assert result[0]["body_text"] == "hi"
    assert "conversation_row_id" not in result[0]
    assert "content" not in result[0]
    assert "embedding" not in result[0]


def test_conversation_message_payload_projects_image_blocks_from_attachments() -> None:
    payload = conversation_message_payload(
        {
            "_id": _FakeObjectId(),
            "body_text": "",
            "timestamp": "2026-05-01T12:34:56.789000+00:00",
            "attachments": [
                {
                    "media_kind": "image",
                    "media_type": "image/png",
                    "description": "chart <with> boundary",
                    "url": "https://cdn.example/chart.png",
                }
            ],
        }
    )

    assert payload["body_text"] == "<image>chart &lt;with&gt; boundary</image>"
    rendered = repr(payload)
    assert "https://cdn.example/chart.png" not in rendered
    assert "conversation_row_id" not in payload


def test_conversation_message_payload_projects_reply_image_blocks() -> None:
    payload = conversation_message_payload(
        {
            "body_text": "following up",
            "reply_context": {
                "reply_excerpt": "",
                "reply_to_display_name": "Tester",
                "reply_attachments": [
                    {
                        "media_type": "image/jpeg",
                        "description": "reply chart",
                        "url": "https://cdn.example/reply.jpg",
                    }
                ],
            },
        }
    )

    assert payload["reply_context"] == {
        "reply_to_display_name": "Tester",
        "reply_excerpt": "<image>reply chart</image>",
    }
    assert "reply_attachments" not in payload["reply_context"]


def test_conversation_message_payload_uses_local_second_precision_timestamp() -> None:
    payload = conversation_message_payload(
        {
            "body_text": "time test",
            "timestamp": "2026-05-01T12:34:56.789000+00:00",
        }
    )

    assert payload["timestamp"] == "2026-05-02 00:34:56"


def test_conversation_message_payload_drops_raw_attachment_url_and_storage_ids() -> None:
    payload = conversation_message_payload(
        {
            "_id": _FakeObjectId(),
            "body_text": "see chart",
            "platform_message_id": "message-1",
            "attachments": [
                {
                    "media_kind": "image",
                    "description": "chart image",
                    "url": "https://cdn.example/chart.png",
                    "storage_object_id": "stored-object-1",
                    "raw_wire_text": "[CQ:image,file=abc]",
                }
            ],
        }
    )

    rendered = repr(payload)
    assert "conversation_row_id" not in payload
    assert "platform_message_id" not in payload
    assert "https://cdn.example/chart.png" not in rendered
    assert "stored-object-1" not in rendered
    assert "raw_wire_text" not in rendered


def test_conversation_message_payload_strips_or_avoids_cq_wire_syntax() -> None:
    payload = conversation_message_payload(
        {
            "body_text": "[CQ:image,file=abc,url=https://cdn.example/raw.png]",
            "attachments": [
                {
                    "media_kind": "image",
                    "description": "clean image description",
                }
            ],
        }
    )

    assert payload["body_text"] == "<image>clean image description</image>"
    rendered = repr(payload)
    assert "[CQ:" not in rendered
    assert "url=" not in rendered


@pytest.mark.asyncio
async def test_search_persistent_memory_delegates_to_vector_memory_search() -> None:
    """search_persistent_memory should call vector search without type/source-kind filtering."""
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
                "source_kind": "conversation_extracted",
            }
        )

    search_memory.assert_awaited_once_with(
        query="quiet conversation preference",
        limit=4,
        method="vector",
        source_global_user_id="global-user-1",
        memory_type=None,
        source_kind=None,
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
        result = await search_persistent_memory_keyword.ainvoke(
            {"keyword": "DDR5", "top_k": 5}
        )

    search_memory.assert_awaited_once_with(
        query="DDR5",
        limit=5,
        method="keyword",
        source_global_user_id=None,
        memory_type=None,
    )
    assert result[0]["memory_name"] == "hardware"
    assert result[0]["content"] == "DDR5 discussion"


@pytest.mark.asyncio
async def test_search_persistent_memory_keyword_uses_shared_default_top_k() -> None:
    """Keyword memory retrieval should use the shared RAG top-k default."""

    with patch(
        "kazusa_ai_chatbot.rag.memory_retrieval_tools.search_memory_db",
        new_callable=AsyncMock,
        return_value=[],
    ) as search_memory:
        await search_persistent_memory_keyword.ainvoke({"keyword": "DDR5"})

    assert search_memory.await_args.kwargs["limit"] == RAG_SEARCH_DEFAULT_TOP_K
