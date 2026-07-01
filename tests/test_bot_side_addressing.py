"""Deterministic tests for assistant-row typed addressing."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot import service as service_module
from kazusa_ai_chatbot.brain_service.outbound import ConversationHistoryWriteError


@pytest.mark.asyncio
async def test_save_assistant_message_persists_current_turn_addressee(monkeypatch) -> None:
    """Assistant rows should preserve the in-turn user addressee."""

    saved_docs: list[dict] = []

    async def _save_conversation(doc: dict) -> str:
        saved_docs.append(doc)
        return_value = f"row-{len(saved_docs)}"
        return return_value

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

    async def _save_conversation(doc: dict) -> str:
        saved_docs.append(doc)
        return_value = f"row-{len(saved_docs)}"
        return return_value

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


@pytest.mark.asyncio
async def test_save_assistant_message_persists_delivery_tracking(monkeypatch) -> None:
    """Assistant rows should carry pending delivery metadata when supplied."""

    saved_docs: list[dict] = []

    async def _save_conversation(doc: dict) -> str:
        saved_docs.append(doc)
        return_value = f"row-{len(saved_docs)}"
        return return_value

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
        "final_dialog": ["tracked reply"],
        "target_addressed_user_ids": ["user-a"],
        "target_broadcast": False,
        "delivery_tracking_id": "delivery-1",
    })

    assert saved_docs[0]["delivery_tracking_id"] == "delivery-1"
    assert saved_docs[0]["logical_message_index"] == 0
    assert saved_docs[0]["delivery_status"] == "pending"


@pytest.mark.asyncio
async def test_save_assistant_message_persists_logical_dialog_rows(
    monkeypatch,
) -> None:
    """Each logical dialog message should persist as its own exact row."""

    saved_docs: list[dict] = []

    async def _save_conversation(doc: dict) -> str:
        saved_docs.append(doc)
        return_value = f"row-{len(saved_docs)}"
        return return_value

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
        "final_dialog": ["first bubble", "second bubble"],
        "target_addressed_user_ids": ["user-a"],
        "target_broadcast": False,
        "delivery_tracking_id": "delivery-1",
        "llm_trace_id": "trace-1",
    })

    assert [doc["body_text"] for doc in saved_docs] == [
        "first bubble",
        "second bubble",
    ]
    assert [doc["logical_message_index"] for doc in saved_docs] == [0, 1]
    assert [doc["delivery_tracking_id"] for doc in saved_docs] == [
        "delivery-1",
        "delivery-1",
    ]
    assert [doc["llm_trace_id"] for doc in saved_docs] == [
        "trace-1",
        "trace-1",
    ]


@pytest.mark.asyncio
async def test_save_assistant_message_persists_row_scoped_delivery_mentions(
    monkeypatch,
) -> None:
    """Assistant rows should record outbound mentions present in that row."""

    saved_docs: list[dict] = []

    async def _save_conversation(doc: dict) -> str:
        saved_docs.append(doc)
        return_value = f"row-{len(saved_docs)}"
        return return_value

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
        "final_dialog": ["@Alex first", "second @Moca"],
        "target_addressed_user_ids": ["user-a"],
        "target_broadcast": False,
        "delivery_mentions": [
            {
                "entity_kind": "user",
                "display_name": "Alex",
                "platform_user_id": "1001",
            },
            {
                "entity_kind": "user",
                "display_name": "Moca",
                "platform_user_id": "1002",
            },
        ],
    })

    assert saved_docs[0]["mentions"] == [
        {
            "entity_kind": "user",
            "display_name": "Alex",
            "platform_user_id": "1001",
            "raw_text": "@Alex",
        }
    ]
    assert saved_docs[1]["mentions"] == [
        {
            "entity_kind": "user",
            "display_name": "Moca",
            "platform_user_id": "1002",
            "raw_text": "@Moca",
        }
    ]


@pytest.mark.asyncio
async def test_save_assistant_message_fails_when_history_row_not_committed(
    monkeypatch,
) -> None:
    """Assistant output is not deliverable without a committed history row."""

    monkeypatch.setattr(
        service_module,
        "_ensure_character_global_identity",
        AsyncMock(return_value="character-global-id"),
    )
    monkeypatch.setattr(
        service_module,
        "save_conversation",
        AsyncMock(return_value=None),
    )

    with pytest.raises(ConversationHistoryWriteError):
        await service_module._save_assistant_message({
            "platform": "qq",
            "platform_channel_id": "group-1",
            "channel_type": "group",
            "platform_bot_id": "bot-1",
            "character_name": "Character",
            "global_user_id": "user-a",
            "final_dialog": ["uncommitted reply"],
            "target_addressed_user_ids": ["user-a"],
            "target_broadcast": False,
        })
