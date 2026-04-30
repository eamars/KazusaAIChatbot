"""Deterministic tests for typed conversation-history envelope storage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.db import conversation as conversation_module
from kazusa_ai_chatbot.message_envelope import INLINE_ATTACHMENT_BYTE_LIMIT


@pytest.mark.asyncio
async def test_save_conversation_writes_typed_fields_and_embedding_source(
    monkeypatch,
) -> None:
    """Conversation writes should persist typed envelope fields and clean embeddings."""

    db = MagicMock()
    db.conversation_history.insert_one = AsyncMock()
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=0)
    get_text_embedding = AsyncMock(return_value=[0.1, 0.2])

    monkeypatch.setattr(conversation_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(conversation_module, "get_text_embedding", get_text_embedding)
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.cache2_runtime.get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )

    await conversation_module.save_conversation({
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "role": "user",
        "platform_user_id": "platform-user",
        "global_user_id": "user-1",
        "display_name": "User",
        "body_text": "clean body",
        "raw_wire_text": "[CQ:at,qq=3768713357] clean body",
        "addressed_to_global_user_ids": ["character-global"],
        "mentions": [{"platform_user_id": "3768713357", "entity_kind": "bot"}],
        "broadcast": False,
        "attachments": [
            {
                "media_type": "image/png",
                "url": "https://example.test/small.png",
                "base64_data": "small-bytes",
                "description": "small image",
                "size_bytes": INLINE_ATTACHMENT_BYTE_LIMIT,
                "storage_shape": "inline",
            },
            {
                "media_type": "image/png",
                "url": "https://example.test/large.png",
                "base64_data": "large-bytes",
                "description": "large image",
                "size_bytes": INLINE_ATTACHMENT_BYTE_LIMIT + 1,
                "storage_shape": "url_only",
            },
        ],
        "timestamp": "2026-04-30T00:00:00+00:00",
    })

    saved_doc = db.conversation_history.insert_one.await_args.args[0]
    get_text_embedding.assert_awaited_once_with(
        "clean body\nsmall image\nlarge image"
    )
    assert saved_doc["body_text"] == "clean body"
    assert "content" not in saved_doc
    assert saved_doc["raw_wire_text"].startswith("[CQ:at")
    assert saved_doc["addressed_to_global_user_ids"] == ["character-global"]
    assert saved_doc["mentions"][0]["entity_kind"] == "bot"
    assert saved_doc["attachments"][0]["base64_data"] == "small-bytes"
    assert "base64_data" not in saved_doc["attachments"][1]
    assert saved_doc["attachments"][1]["url"] == "https://example.test/large.png"


@pytest.mark.asyncio
async def test_get_conversation_history_returns_typed_rows(monkeypatch) -> None:
    """Conversation reads return typed rows directly."""

    db = MagicMock()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[{
        "_id": "row-1",
        "platform_channel_id": "chan-1",
        "timestamp": "2026-04-30T00:00:00+00:00",
        "body_text": "body text",
        "raw_wire_text": "[CQ:at,qq=3768713357] body text",
        "addressed_to_global_user_ids": ["character-global"],
    }])
    db.conversation_history.find.return_value.sort.return_value.limit.return_value = cursor

    monkeypatch.setattr(conversation_module, "get_db", AsyncMock(return_value=db))

    rows = await conversation_module.get_conversation_history(
        platform="qq",
        platform_channel_id="chan-1",
        limit=1,
    )

    assert rows[0]["body_text"] == "body text"
    assert rows[0]["raw_wire_text"].startswith("[CQ:at")
    assert rows[0]["addressed_to_global_user_ids"] == ["character-global"]


@pytest.mark.asyncio
async def test_keyword_search_uses_body_text_filter(
    monkeypatch,
) -> None:
    """Keyword search should query body_text only."""

    db = MagicMock()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[{
        "_id": "row-1",
        "platform_channel_id": "chan-1",
        "timestamp": "2026-04-30T00:00:00+00:00",
        "body_text": "keyword",
        "raw_wire_text": "[CQ:at,qq=3768713357] keyword",
    }])
    db.conversation_history.find.return_value.sort.return_value.limit.return_value = cursor

    monkeypatch.setattr(conversation_module, "get_db", AsyncMock(return_value=db))

    results = await conversation_module.search_conversation_history(
        "keyword",
        platform_channel_id="chan-1",
        method="keyword",
    )

    call_filter = db.conversation_history.find.call_args.args[0]
    assert call_filter["body_text"]["$regex"] == "keyword"
    assert "$or" not in call_filter
    assert results[0][1]["body_text"] == "keyword"
