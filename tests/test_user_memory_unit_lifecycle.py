"""Tests for user-memory lifecycle repository updates."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.db import script_operations
from kazusa_ai_chatbot.db import user_memory_units


class _UpdateResult:
    """Minimal Motor update result."""

    matched_count = 1
    modified_count = 1


class _FakeCollection:
    """Capture update_one calls made to user_memory_units."""

    def __init__(self) -> None:
        self.filter = None
        self.update = None

    async def update_one(self, filter_doc, update_doc):
        self.filter = filter_doc
        self.update = update_doc
        return _UpdateResult()


class _FakeDb:
    """Minimal DB object with the collection under test."""

    def __init__(self) -> None:
        self.user_memory_units = _FakeCollection()


@pytest.mark.asyncio
async def test_lifecycle_update_marks_active_commitment_cancelled(monkeypatch):
    """Lifecycle updates should target active commitment rows only."""

    fake_db = _FakeDb()

    async def _fake_get_db():
        return fake_db

    monkeypatch.setattr(user_memory_units, "get_db", _fake_get_db)

    result = await user_memory_units.update_user_memory_unit_lifecycle(
        "promise-001",
        status="cancelled",
        storage_timestamp_utc="2026-05-16T00:00:00+00:00",
        reason="角色决定不再继续这个承诺。",
        action_attempt_id="action_attempt:001",
        due_at="2026-05-07T00:00:00+00:00",
    )

    assert fake_db.user_memory_units.filter == {
        "unit_id": "promise-001",
        "unit_type": "active_commitment",
        "status": "active",
    }
    assert fake_db.user_memory_units.update["$set"]["status"] == "cancelled"
    assert fake_db.user_memory_units.update["$set"]["cancelled_at"] == (
        "2026-05-16T00:00:00+00:00"
    )
    assert fake_db.user_memory_units.update["$push"]["merge_history"][
        "action_attempt_id"
    ] == "action_attempt:001"
    assert result["modified_count"] == 1


@pytest.mark.asyncio
async def test_lifecycle_update_allows_deferred_active_audit(monkeypatch):
    """A deferred lifecycle decision should audit without retiring the unit."""

    fake_db = _FakeDb()

    async def _fake_get_db():
        return fake_db

    monkeypatch.setattr(user_memory_units, "get_db", _fake_get_db)

    await user_memory_units.update_user_memory_unit_lifecycle(
        "promise-001",
        status="active",
        storage_timestamp_utc="2026-05-16T00:00:00+00:00",
        reason="角色决定继续等待自然间隙。",
        action_attempt_id="action_attempt:002",
    )

    set_doc = fake_db.user_memory_units.update["$set"]
    assert set_doc == {
        "status": "active",
        "updated_at": "2026-05-16T00:00:00+00:00",
    }
    assert fake_db.user_memory_units.update["$push"]["merge_history"][
        "status"
    ] == "active"


@pytest.mark.asyncio
async def test_semantic_identity_archive_helper_targets_any_active_unit(
    monkeypatch,
) -> None:
    """Maintenance archiving should not use the active-commitment helper."""

    fake_db = _FakeDb()

    async def _fake_get_db():
        return fake_db

    monkeypatch.setattr(script_operations, "get_db", _fake_get_db)

    result = await script_operations.archive_user_memory_unit_for_semantic_identity_repair(
        unit_id="memory-unit-001",
        reason="semantic_identity_pollution",
        storage_timestamp_utc="2026-07-03T00:00:00+00:00",
    )

    assert fake_db.user_memory_units.filter == {
        "unit_id": "memory-unit-001",
        "status": "active",
    }
    assert fake_db.user_memory_units.update["$set"]["status"] == "archived"
    assert fake_db.user_memory_units.update["$set"]["archived_at"] == (
        "2026-07-03T00:00:00+00:00"
    )
    history = fake_db.user_memory_units.update["$push"]["merge_history"]
    assert history["operation"] == "semantic_identity_repair_archive"
    assert result["matched_count"] == 1
