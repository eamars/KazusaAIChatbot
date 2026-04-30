"""Deterministic tests for assistant-row typed addressing."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot import service as service_module


@pytest.mark.asyncio
async def test_save_assistant_message_persists_current_turn_addressee(monkeypatch) -> None:
    """Assistant rows should preserve the in-turn user addressee."""

    saved_docs: list[dict] = []

    async def _save_conversation(doc: dict) -> None:
        saved_docs.append(doc)

    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value="character-global-id"),
    )
    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)

    turns = [
        ("msg-a1", "user-a", "reply to A 1"),
        ("msg-b1", "user-b", "reply to B 1"),
        ("msg-a2", "user-a", "reply to A 2"),
    ]
    for message_id, global_user_id, dialog in turns:
        await service_module._save_assistant_message({
            "platform": "qq",
            "platform_channel_id": "group-1",
            "channel_type": "group",
            "platform_message_id": message_id,
            "platform_bot_id": "bot-1",
            "character_name": "Character",
            "global_user_id": global_user_id,
            "final_dialog": [dialog],
            "target_addressed_user_ids": [global_user_id],
            "target_broadcast": False,
        })

    addressed_rows = [
        doc["addressed_to_global_user_ids"]
        for doc in saved_docs
    ]
    assert addressed_rows == [["user-a"], ["user-b"], ["user-a"]]
    assert [doc["broadcast"] for doc in saved_docs] == [False, False, False]
    assert [doc["body_text"] for doc in saved_docs] == [
        "reply to A 1",
        "reply to B 1",
        "reply to A 2",
    ]


@pytest.mark.asyncio
async def test_save_assistant_message_honors_explicit_broadcast(monkeypatch) -> None:
    """Explicit broadcast assistant rows should remain unaddressed."""

    saved_docs: list[dict] = []

    async def _save_conversation(doc: dict) -> None:
        saved_docs.append(doc)

    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value="character-global-id"),
    )
    monkeypatch.setattr(service_module, "save_conversation", _save_conversation)

    await service_module._save_assistant_message({
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "character_name": "Character",
        "global_user_id": "user-a",
        "final_dialog": ["channel notice"],
        "target_addressed_user_ids": [],
        "target_broadcast": True,
    })

    assert saved_docs[0]["addressed_to_global_user_ids"] == []
    assert saved_docs[0]["broadcast"] is True
