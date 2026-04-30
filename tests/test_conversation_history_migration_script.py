"""Tests for the one-off conversation-history envelope rewrite script."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.db import bootstrap as bootstrap_module
from scripts import migrate_conversation_history_envelope as migration_module


class _Cursor:
    """Small async cursor double for migration script tests."""

    def __init__(self, rows: list[dict]) -> None:
        """Store rows returned by `to_list`.

        Args:
            rows: MongoDB-like rows to expose from the cursor.

        Returns:
            None.
        """

        self._rows = rows

    def sort(self, _field: str, _direction: int) -> "_Cursor":
        """Accept sort chaining used by the migration query.

        Args:
            _field: Ignored field name.
            _direction: Ignored sort direction.

        Returns:
            This cursor double for fluent chaining.
        """

        return self

    def limit(self, _limit: int) -> "_Cursor":
        """Accept limit chaining used by the migration query.

        Args:
            _limit: Ignored maximum row count.

        Returns:
            This cursor double for fluent chaining.
        """

        return self

    async def to_list(self, length: int) -> list[dict]:
        """Return the configured cursor rows.

        Args:
            length: Requested maximum row count.

        Returns:
            Configured MongoDB-like rows.
        """

        return self._rows


def test_runtime_bootstrap_does_not_expose_migration_path() -> None:
    """Runtime bootstrap should not own conversation-history rewrites."""

    assert not hasattr(bootstrap_module, "migrate_legacy_conversation_history_rows")
    assert not hasattr(bootstrap_module, "legacy_conversation_query")


@pytest.mark.asyncio
async def test_migration_dry_run_counts_rows_without_writes(monkeypatch) -> None:
    """Dry-run mode should report matching rows and leave storage untouched."""

    db = MagicMock()
    db.conversation_history.count_documents = AsyncMock(return_value=3)
    db.conversation_history.update_one = AsyncMock()

    monkeypatch.setattr(migration_module, "get_db", AsyncMock(return_value=db))

    row_count = await migration_module.migrate_legacy_conversation_history_rows(
        dry_run=True,
    )

    assert row_count == 3
    db.conversation_history.count_documents.assert_awaited_once_with(
        migration_module.legacy_conversation_query()
    )
    db.conversation_history.update_one.assert_not_awaited()


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
    db = MagicMock()
    db.conversation_history.find.side_effect = [
        _Cursor(legacy_rows),
        _Cursor([]),
    ]
    db.conversation_history.update_one = AsyncMock()
    db.conversation_history.count_documents = AsyncMock(return_value=0)

    monkeypatch.setattr(migration_module, "get_db", AsyncMock(return_value=db))

    migrated_count = await migration_module.migrate_legacy_conversation_history_rows(
        dry_run=False,
    )

    assert migrated_count == 2
    assert db.conversation_history.update_one.await_count == 2

    first_update = db.conversation_history.update_one.await_args_list[0].args[1]
    assert first_update["$set"]["body_text"] == "hello"
    assert first_update["$set"]["raw_wire_text"].startswith("[Reply to message]")
    assert first_update["$set"]["addressed_to_global_user_ids"] == [
        CHARACTER_GLOBAL_USER_ID
    ]
    assert first_update["$set"]["mentions"] == []
    assert first_update["$set"]["broadcast"] is False
    assert first_update["$set"]["attachments"] == []
    assert first_update["$unset"] == {"content": ""}

    second_update = db.conversation_history.update_one.await_args_list[1].args[1]
    assert second_update["$set"]["body_text"] == "old assistant reply"
    assert second_update["$set"]["addressed_to_global_user_ids"] == []
    assert second_update["$set"]["broadcast"] is False
