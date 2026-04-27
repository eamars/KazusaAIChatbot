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
    monkeypatch.setattr(conversation_module, "get_text_embedding", AsyncMock(return_value=[0.1, 0.2]))
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
        "content": "hello",
        "timestamp": "2026-04-27T00:00:00+12:00",
    })

    db.conversation_history.insert_one.assert_awaited_once()
    event = runtime.invalidate.await_args.args[0]
    assert event.source == "conversation_history"
    assert event.platform == "qq"
    assert event.platform_channel_id == "chan-1"
    assert event.global_user_id == "user-1"
    assert event.timestamp == "2026-04-27T00:00:00+12:00"
