"""Tests for direct active-commitment memory-unit reads."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.db import user_memory_units
from kazusa_ai_chatbot.db.schemas import (
    UserMemoryUnitStatus,
    UserMemoryUnitType,
)


class _AsyncCursor:
    """Small async cursor fake for user-memory read tests."""

    def __init__(self, docs: list[dict]) -> None:
        """Create a cursor over stored documents.

        Args:
            docs: Documents visible to the cursor.
        """

        self._docs = [dict(doc) for doc in docs]
        self.sort_spec: list[tuple[str, int]] | None = None
        self.limit_value: int | None = None

    def sort(self, sort_spec: list[tuple[str, int]]) -> "_AsyncCursor":
        """Record the requested sort order.

        Args:
            sort_spec: Mongo-style sort pairs requested by the repository.

        Returns:
            This cursor for chained calls.
        """

        self.sort_spec = list(sort_spec)
        return self

    def limit(self, limit: int) -> "_AsyncCursor":
        """Record the requested row limit.

        Args:
            limit: Maximum number of rows the repository requested.

        Returns:
            This cursor for chained calls.
        """

        self.limit_value = limit
        return self

    def __aiter__(self) -> "_AsyncCursor":
        self._iter_index = 0
        return self

    async def __anext__(self) -> dict:
        effective_limit = (
            self.limit_value
            if self.limit_value is not None
            else len(self._docs)
        )
        if self._iter_index >= min(effective_limit, len(self._docs)):
            raise StopAsyncIteration
        row = dict(self._docs[self._iter_index])
        self._iter_index += 1
        return row


class _FakeCollection:
    """Capture active-commitment find calls."""

    def __init__(self, docs: list[dict]) -> None:
        """Create the collection fake.

        Args:
            docs: Rows returned by the next ``find`` call.
        """

        self.cursor = _AsyncCursor(docs)
        self.filter_doc: dict | None = None
        self.projection: dict | None = None

    def find(self, filter_doc: dict, projection: dict) -> _AsyncCursor:
        """Record the query and return a cursor over stored docs.

        Args:
            filter_doc: Mongo filter supplied by the repository.
            projection: Mongo projection supplied by the repository.

        Returns:
            Cursor over the configured rows.
        """

        self.filter_doc = dict(filter_doc)
        self.projection = dict(projection)
        return self.cursor


class _FakeDb:
    """Minimal DB object exposing ``user_memory_units``."""

    def __init__(self, docs: list[dict]) -> None:
        """Create a fake database.

        Args:
            docs: Rows visible to the fake collection.
        """

        self.user_memory_units = _FakeCollection(docs)


@pytest.mark.asyncio
async def test_active_commitment_reader_includes_no_due_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct reader should not require ``due_at`` to be present."""

    docs = [
        {
            "unit_id": "unit-001",
            "global_user_id": "global-user-1",
            "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
            "status": UserMemoryUnitStatus.ACTIVE,
            "fact": "User promised dessert.",
            "due_at": None,
            "updated_at": "2026-05-29T03:09:09+00:00",
        },
        {
            "unit_id": "unit-002",
            "global_user_id": "global-user-1",
            "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
            "status": UserMemoryUnitStatus.ACTIVE,
            "fact": "User promised tea.",
            "updated_at": "2026-05-28T03:09:09+00:00",
        },
    ]
    fake_db = _FakeDb(docs)

    async def _fake_get_db() -> _FakeDb:
        return fake_db

    monkeypatch.setattr(user_memory_units, "get_db", _fake_get_db)

    result = await user_memory_units.query_active_commitment_memory_units_for_user(
        global_user_id="global-user-1",
        limit=10,
    )

    assert result["documents"] == docs
    assert result["limit"] == 10
    assert result["limit_exceeded"] is False
    assert fake_db.user_memory_units.filter_doc == {
        "global_user_id": "global-user-1",
        "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
        "status": UserMemoryUnitStatus.ACTIVE,
    }
    assert fake_db.user_memory_units.projection == {"_id": 0, "embedding": 0}
    assert fake_db.user_memory_units.cursor.sort_spec == [
        ("due_at", 1),
        ("updated_at", -1),
        ("unit_id", 1),
    ]
    assert fake_db.user_memory_units.cursor.limit_value == 11


@pytest.mark.asyncio
async def test_active_commitment_reader_reports_limit_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct reader should refuse silent partial review."""

    docs = [
        {
            "unit_id": f"unit-{index:03d}",
            "global_user_id": "global-user-1",
            "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
            "status": UserMemoryUnitStatus.ACTIVE,
            "fact": f"User promised item {index}.",
        }
        for index in range(1, 4)
    ]
    fake_db = _FakeDb(docs)

    async def _fake_get_db() -> _FakeDb:
        return fake_db

    monkeypatch.setattr(user_memory_units, "get_db", _fake_get_db)

    result = await user_memory_units.query_active_commitment_memory_units_for_user(
        global_user_id="global-user-1",
        limit=2,
    )

    assert result["documents"] == docs[:2]
    assert result["limit"] == 2
    assert result["limit_exceeded"] is True
