"""Tests for Cache2 invalidation from conversation writes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.db import conversation as conversation_module


@pytest.mark.asyncio
async def test_save_conversation_invalidates_conversation_history_cache(monkeypatch) -> None:
    db = MagicMock()
    db.conversation_history.insert_one = AsyncMock()
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=1)

    monkeypatch.setattr(conversation_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(
        conversation_module,
        "get_document_text_embedding",
        AsyncMock(return_value=[0.1, 0.2]),
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.cache2_runtime.get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )

    await conversation_module.save_conversation({
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "role": "user",
        "platform_message_id": "msg-1",
        "platform_user_id": "platform-user",
        "global_user_id": "user-1",
        "display_name": "User",
        "body_text": "hello",
        "raw_wire_text": "hello",
        "content_type": "text",
        "addressed_to_global_user_ids": ["character-global"],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
        "timestamp": "2026-04-26T12:00:00+00:00",
    })

    db.conversation_history.insert_one.assert_awaited_once()
    event = runtime.invalidate.await_args.args[0]
    assert event.source == "conversation_history"
    assert event.platform == "qq"
    assert event.platform_channel_id == "chan-1"
    assert event.global_user_id == "user-1"
    assert event.storage_timestamp_utc == "2026-04-26T12:00:00+00:00"


@pytest.mark.asyncio
async def test_save_conversation_receipt_invalidates_cache_after_enrichment(
    monkeypatch,
) -> None:
    """Receipt commits first, then enriches the same row, then invalidates."""

    from bson import ObjectId

    db = MagicMock()
    insert_result = MagicMock()
    insert_result.inserted_id = ObjectId("665000000000000000000010")
    db.conversation_history.insert_one = AsyncMock(return_value=insert_result)
    db.conversation_history.update_one = AsyncMock()
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=1)

    monkeypatch.setattr(conversation_module, "get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(
        conversation_module,
        "get_document_text_embedding",
        AsyncMock(return_value=[0.1, 0.2]),
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.cache2_runtime.get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )

    row_id = await conversation_module.save_conversation_receipt({
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "role": "user",
        "platform_message_id": "msg-1",
        "platform_user_id": "platform-user",
        "global_user_id": "user-1",
        "display_name": "User",
        "body_text": "hello",
        "raw_wire_text": "hello",
        "content_type": "text",
        "addressed_to_global_user_ids": ["character-global"],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
        "timestamp": "2026-04-26T12:00:00+00:00",
        "received_at": "2026-04-26T12:00:01+00:00",
    })

    assert row_id == "665000000000000000000010"
    db.conversation_history.insert_one.assert_awaited_once()
    db.conversation_history.update_one.assert_awaited_once_with(
        {"_id": insert_result.inserted_id},
        {"$set": {"embedding": [0.1, 0.2]}},
    )
    event = runtime.invalidate.await_args.args[0]
    assert event.reason == "save_conversation_receipt"
    assert event.platform == "qq"
    assert event.storage_timestamp_utc == "2026-04-26T12:00:00+00:00"
