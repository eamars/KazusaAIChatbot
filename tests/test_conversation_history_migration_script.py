"""Tests for the one-off conversation-history envelope rewrite script."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.db import bootstrap as bootstrap_module
from kazusa_ai_chatbot.db import script_operations as script_operations_module
from scripts import migrate_conversation_history_envelope as migration_module


def test_runtime_bootstrap_does_not_expose_migration_path() -> None:
    """Runtime bootstrap should not own conversation-history rewrites."""

    assert not hasattr(bootstrap_module, "migrate_legacy_conversation_history_rows")
    assert not hasattr(bootstrap_module, "legacy_conversation_query")


def test_migration_selector_includes_dirty_semantic_text() -> None:
    """Maintenance selection should include typed rows with transport syntax."""

    query = script_operations_module._legacy_conversation_query(
        semantic_text_pattern=migration_module.BODY_TEXT_TRANSPORT_SYNTAX_PATTERN,
    )
    query_text = repr(query)

    assert "body_text" in query_text
    assert "reply_context.reply_excerpt" in query_text
    assert r"\[CQ:" in query_text


def test_migration_sanitizes_dirty_existing_body_text() -> None:
    """Dirty typed rows should be repaired outside the brain runtime path."""

    fields = migration_module.legacy_conversation_fields({
        "_id": "dirty-image",
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "role": "user",
        "body_text": "[CQ:image,file=sam.png]",
        "raw_wire_text": "[CQ:image,file=sam.png]",
        "attachments": [{
            "media_kind": "image",
            "description": '拓竹入驻山姆，不只是上架 3D 打印机',
        }],
        "addressed_to_global_user_ids": [CHARACTER_GLOBAL_USER_ID],
        "mentions": [],
        "broadcast": False,
        "timestamp": "2026-04-30T00:00:00+00:00",
    })

    assert fields["body_text"] == ""
    assert fields["attachments"] == [{
        "media_kind": "image",
        "description": '拓竹入驻山姆，不只是上架 3D 打印机',
    }]


def test_migration_sanitizes_dirty_reply_excerpt() -> None:
    """Dirty reply excerpts should be repaired before history projection."""

    fields = migration_module.legacy_conversation_fields({
        "_id": "dirty-reply",
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "role": "user",
        "body_text": '这个呢',
        "raw_wire_text": '这个呢',
        "reply_context": {
            "reply_to_message_id": "1733223276",
            "reply_excerpt": "look[CQ:image,file=sam.png]nice",
        },
        "attachments": [],
        "addressed_to_global_user_ids": [CHARACTER_GLOBAL_USER_ID],
        "mentions": [],
        "broadcast": False,
        "timestamp": "2026-04-30T00:00:00+00:00",
    })

    assert fields["reply_context"] == {
        "reply_to_message_id": "1733223276",
        "reply_excerpt": "look nice",
    }


@pytest.mark.asyncio
async def test_migration_dry_run_counts_rows_without_writes(monkeypatch) -> None:
    """Dry-run mode should report matching rows and leave storage untouched."""

    count_rows = AsyncMock(return_value=3)
    update_row = AsyncMock()

    monkeypatch.setattr(
        migration_module,
        "count_legacy_conversation_history_rows",
        count_rows,
    )
    monkeypatch.setattr(
        migration_module,
        "update_conversation_history_row",
        update_row,
    )

    row_count = await migration_module.migrate_legacy_conversation_history_rows(
        dry_run=True,
    )

    assert row_count == 3
    count_rows.assert_awaited_once_with(
        semantic_text_pattern=migration_module.BODY_TEXT_TRANSPORT_SYNTAX_PATTERN,
    )
    update_row.assert_not_awaited()


@pytest.mark.asyncio
async def test_migration_apply_rewrites_legacy_conversation_rows(monkeypatch) -> None:
    """Apply mode should fill typed fields and remove legacy content."""

    legacy_rows = [
        {
            "_id": "legacy-user",
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "private",
            "role": "user",
            "content": (
                "[Reply to message] [CQ:reply,id=abc] "
                "[CQ:at,qq=3768713357] hello"
            ),
            "timestamp": "2026-04-30T00:00:00+00:00",
        },
        {
            "_id": "legacy-assistant",
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "group",
            "role": "assistant",
            "content": "old assistant reply",
            "timestamp": "2026-04-30T00:00:01+00:00",
        },
    ]
    list_rows = AsyncMock(side_effect=[legacy_rows, []])
    update_row = AsyncMock()
    count_rows = AsyncMock(return_value=0)
    monkeypatch.setattr(
        migration_module,
        "list_legacy_conversation_history_rows",
        list_rows,
    )
    monkeypatch.setattr(
        migration_module,
        "update_conversation_history_row",
        update_row,
    )
    monkeypatch.setattr(
        migration_module,
        "count_legacy_conversation_history_rows",
        count_rows,
    )

    migrated_count = await migration_module.migrate_legacy_conversation_history_rows(
        dry_run=False,
    )

    assert migrated_count == 2
    assert update_row.await_count == 2

    first_update = update_row.await_args_list[0].kwargs
    assert first_update["set_fields"]["body_text"] == "hello"
    assert first_update["set_fields"]["raw_wire_text"].startswith("[Reply to message]")
    assert first_update["set_fields"]["addressed_to_global_user_ids"] == [
        CHARACTER_GLOBAL_USER_ID
    ]
    assert first_update["set_fields"]["mentions"] == []
    assert first_update["set_fields"]["broadcast"] is False
    assert first_update["set_fields"]["attachments"] == []
    assert first_update["unset_fields"] == ("content",)

    second_update = update_row.await_args_list[1].kwargs
    assert second_update["set_fields"]["body_text"] == "old assistant reply"
    assert second_update["set_fields"]["addressed_to_global_user_ids"] == []
    assert second_update["set_fields"]["broadcast"] is False
