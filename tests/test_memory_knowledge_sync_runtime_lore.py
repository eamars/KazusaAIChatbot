"""Tests for curated shared-memory seed sync behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from scripts import manage_memory_knowledge


def _entry() -> manage_memory_knowledge.KnowledgeEntry:
    return_value = manage_memory_knowledge.KnowledgeEntry(
        data={
            "memory_name": "Seed memory",
            "content": "Seed content",
            "source_global_user_id": "",
            "memory_type": "fact",
            "source_kind": "seeded_manual",
            "confidence_note": "seed",
            "status": "active",
            "expiry_timestamp": None,
        },
        line_number=1,
    )
    return return_value


def test_default_knowledge_path_uses_personality_seed_root() -> None:
    """The maintenance CLI follows the current repository seed location."""
    assert manage_memory_knowledge.DEFAULT_KNOWLEDGE_PATH.as_posix() == (
        "personalities/knowledge/memory_seed.jsonl"
    )


@pytest.mark.asyncio
async def test_sync_entries_delegates_to_reset_without_direct_memory_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed sync uses the reset implementation that preserves runtime lore."""
    reset_memory_from_entries = AsyncMock(return_value={
        "seed_rows_inserted": 1,
        "seed_rows_updated": 0,
        "seed_rows_unchanged": 0,
        "seed_rows_deleted": 2,
        "legacy_rows_deleted": 1,
    })
    monkeypatch.setattr(
        manage_memory_knowledge,
        "reset_memory_from_entries",
        reset_memory_from_entries,
    )

    counters = await manage_memory_knowledge.sync_entries(
        [_entry()],
        apply=False,
        prune_unmanaged_global=True,
    )

    reset_memory_from_entries.assert_awaited_once_with(
        [_entry().data],
        dry_run=True,
        prune_unmanaged_global=True,
    )
    assert counters == {
        "inserted": 1,
        "updated": 0,
        "unchanged": 0,
        "duplicates_deleted": 0,
        "pruned": 3,
    }
