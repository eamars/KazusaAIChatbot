"""Tests for event-log DB helper contracts."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import kazusa_ai_chatbot.db.event_logging as db_event_logging


class _FakeCursor:
    """Small async cursor fake for event-log DB tests."""

    def __init__(self, rows: list[dict]) -> None:
        """Store rows returned by `to_list`."""

        self.rows = rows
        self.sort_calls: list[list[tuple[str, int]]] = []
        self.limit_calls: list[int] = []

    def sort(self, sort_spec: list[tuple[str, int]]):
        """Record a requested sort and return the cursor."""

        self.sort_calls.append(sort_spec)
        return self

    def limit(self, limit: int):
        """Record a requested limit and return the cursor."""

        self.limit_calls.append(limit)
        return self

    async def to_list(self, length):
        """Return stored rows."""

        _ = length
        return_value = list(self.rows)
        return return_value


class _FakeCollection:
    """Small async collection fake for event-log DB tests."""

    def __init__(self) -> None:
        """Create collection state."""

        self.indexes: list[dict] = []
        self.inserted: list[dict] = []
        self.aggregate = MagicMock(return_value=_FakeCursor([{"count": 1}]))
        self.count_documents = AsyncMock(return_value=3)
        self._find_cursor = _FakeCursor([{"event_id": "event-1"}])

    async def create_index(self, keys, **kwargs) -> None:
        """Record requested index definitions."""

        self.indexes.append({"keys": keys, "kwargs": kwargs})

    async def insert_one(self, document: dict) -> None:
        """Record inserted documents."""

        self.inserted.append(dict(document))

    def find(self, filter_doc: dict):
        """Return a cursor fake for a find request."""

        self.find_filter = filter_doc
        return self._find_cursor


class _FakeDb:
    """Small DB fake containing event-log collections."""

    def __init__(self) -> None:
        """Create DB fake state."""

        self.collections = {
            db_event_logging.EVENT_LOG_EVENTS_COLLECTION: _FakeCollection(),
            db_event_logging.EVENT_LOG_SNAPSHOTS_COLLECTION: _FakeCollection(),
        }
        self.created_collections: list[str] = []

    async def list_collection_names(self) -> list[str]:
        """Return existing collection names."""

        return_value = []
        return return_value

    async def create_collection(self, name: str) -> None:
        """Record collection creation and ensure a fake exists."""

        self.created_collections.append(name)
        if name not in self.collections:
            self.collections[name] = _FakeCollection()

    def __getitem__(self, name: str) -> _FakeCollection:
        """Return one fake collection."""

        return_value = self.collections[name]
        return return_value


@pytest.mark.asyncio
async def test_ensure_event_log_indexes_creates_collections_and_indexes(
    monkeypatch,
) -> None:
    """Event-log bootstrap should create collections and required indexes."""

    fake_db = _FakeDb()
    monkeypatch.setattr(
        db_event_logging,
        "get_db",
        AsyncMock(return_value=fake_db),
    )

    await db_event_logging.ensure_event_log_indexes()

    assert fake_db.created_collections == [
        db_event_logging.EVENT_LOG_EVENTS_COLLECTION,
        db_event_logging.EVENT_LOG_SNAPSHOTS_COLLECTION,
    ]
    event_indexes = fake_db[
        db_event_logging.EVENT_LOG_EVENTS_COLLECTION
    ].indexes
    snapshot_indexes = fake_db[
        db_event_logging.EVENT_LOG_SNAPSHOTS_COLLECTION
    ].indexes
    assert {
        "keys": "event_id",
        "kwargs": {"unique": True, "name": "event_log_event_id_unique"},
    } in event_indexes
    assert {
        "keys": "snapshot_id",
        "kwargs": {"unique": True, "name": "event_log_snapshot_id_unique"},
    } in snapshot_indexes


@pytest.mark.asyncio
async def test_insert_event_log_event_returns_event_id(monkeypatch) -> None:
    """Insert helper should return the explicit event identifier."""

    fake_db = _FakeDb()
    monkeypatch.setattr(
        db_event_logging,
        "get_db",
        AsyncMock(return_value=fake_db),
    )
    document = {"event_id": "event-123", "event_family": "worker"}

    event_id = await db_event_logging.insert_event_log_event(document)

    assert event_id == "event-123"
    inserted = fake_db[
        db_event_logging.EVENT_LOG_EVENTS_COLLECTION
    ].inserted
    assert inserted == [document]


@pytest.mark.asyncio
async def test_find_and_count_event_log_events(monkeypatch) -> None:
    """Read helpers should use bounded collection operations."""

    fake_db = _FakeDb()
    monkeypatch.setattr(
        db_event_logging,
        "get_db",
        AsyncMock(return_value=fake_db),
    )

    rows = await db_event_logging.find_event_log_events(
        {"event_family": "worker"},
        sort=[("occurred_at", -1)],
        limit=1,
    )
    count = await db_event_logging.count_event_log_events(
        {"event_family": "worker"},
    )

    collection = fake_db[db_event_logging.EVENT_LOG_EVENTS_COLLECTION]
    assert rows == [{"event_id": "event-1"}]
    assert count == 3
    assert collection.find_filter == {"event_family": "worker"}
    assert collection._find_cursor.sort_calls == [[("occurred_at", -1)]]
    assert collection._find_cursor.limit_calls == [1]
