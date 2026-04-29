from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db import rag_cache2_persistent as cache_module
from kazusa_ai_chatbot.rag.cache2_policy import (
    INITIALIZER_AGENT_REGISTRY_VERSION,
    INITIALIZER_CACHE_NAME,
    INITIALIZER_POLICY_VERSION,
    INITIALIZER_PROMPT_VERSION,
    INITIALIZER_STRATEGY_SCHEMA_VERSION,
)
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime
from kazusa_ai_chatbot import service as service_module


class _AsyncCursor:
    """Small async cursor fake for persistent cache helper tests."""

    def __init__(self, docs: list[dict]) -> None:
        """Create the cursor over a copy of documents.

        Args:
            docs: Documents visible to the cursor.
        """
        self._docs = [dict(doc) for doc in docs]
        self._limit: int | None = None

    def sort(self, sort_spec: list[tuple[str, int]]) -> "_AsyncCursor":
        """Sort documents using Mongo-like field/direction pairs.

        Args:
            sort_spec: List of ``(field, direction)`` pairs.

        Returns:
            This cursor for chained calls.
        """
        for field, direction in reversed(sort_spec):
            self._docs.sort(
                key=lambda doc: doc.get(field, ""),
                reverse=direction < 0,
            )
        return self

    def limit(self, limit: int) -> "_AsyncCursor":
        """Apply a result limit.

        Args:
            limit: Maximum rows to return.

        Returns:
            This cursor for chained calls.
        """
        self._limit = limit
        return self

    async def to_list(self, length: int) -> list[dict]:
        """Return the visible rows.

        Args:
            length: Maximum rows requested by the caller.

        Returns:
            Cursor documents clipped by cursor and call limits.
        """
        effective_limit = self._limit if self._limit is not None else length
        return_value = [dict(doc) for doc in self._docs[:effective_limit]]
        return return_value


class _FakeCollection:
    """In-memory collection fake for deterministic persistent-cache tests."""

    def __init__(self, docs: list[dict] | None = None) -> None:
        """Create a fake collection.

        Args:
            docs: Initial documents keyed by ``_id``.
        """
        self.docs = {doc["_id"]: dict(doc) for doc in docs or []}

    def _matches(self, doc: dict, query: dict) -> bool:
        for field, expected in query.items():
            actual = doc.get(field)
            if isinstance(expected, dict) and "$ne" in expected:
                if actual == expected["$ne"]:
                    return False
            elif isinstance(expected, dict) and "$in" in expected:
                if actual not in expected["$in"]:
                    return False
            elif actual != expected:
                return False
        return True

    async def delete_many(self, query: dict) -> SimpleNamespace:
        """Delete matching documents.

        Args:
            query: Mongo-style equality filter with limited operator support.

        Returns:
            Object exposing ``deleted_count``.
        """
        ids_to_delete = [
            doc_id
            for doc_id, doc in self.docs.items()
            if self._matches(doc, query)
        ]
        for doc_id in ids_to_delete:
            del self.docs[doc_id]
        return_value = SimpleNamespace(deleted_count=len(ids_to_delete))
        return return_value

    def find(self, query: dict, projection: dict | None = None) -> _AsyncCursor:
        """Return matching documents through an async cursor.

        Args:
            query: Mongo-style equality filter with limited operator support.
            projection: Optional projection; only ``{"_id": 1}`` is needed.

        Returns:
            Cursor over matching rows.
        """
        docs = [dict(doc) for doc in self.docs.values() if self._matches(doc, query)]
        if projection == {"_id": 1}:
            docs = [{"_id": doc["_id"]} for doc in docs]
        return_value = _AsyncCursor(docs)
        return return_value

    async def update_one(
        self,
        query: dict,
        update: dict,
        *,
        upsert: bool = False,
    ) -> SimpleNamespace:
        """Apply a small subset of Mongo ``update_one`` behavior.

        Args:
            query: Must contain ``_id``.
            update: Update document with ``$set``, ``$setOnInsert``, ``$inc``.
            upsert: Whether to insert missing rows.

        Returns:
            Object exposing ``matched_count``.
        """
        doc_id = query["_id"]
        doc = self.docs.get(doc_id)
        inserted = False
        if doc is None:
            if not upsert:
                return_value = SimpleNamespace(matched_count=0)
                return return_value
            doc = {"_id": doc_id}
            doc.update(update.get("$setOnInsert", {}))
            self.docs[doc_id] = doc
            inserted = True

        doc.update(update.get("$set", {}))
        for field, amount in update.get("$inc", {}).items():
            doc[field] = int(doc.get(field, 0)) + int(amount)

        return_value = SimpleNamespace(matched_count=0 if inserted else 1)
        return return_value

    async def count_documents(self, query: dict) -> int:
        """Count matching rows.

        Args:
            query: Mongo-style equality filter with limited operator support.

        Returns:
            Count of matching rows.
        """
        return_value = sum(1 for doc in self.docs.values() if self._matches(doc, query))
        return return_value


class _FakeDb:
    """Database fake exposing the persistent cache collection by name."""

    def __init__(self, collection: _FakeCollection) -> None:
        """Create the fake database.

        Args:
            collection: Collection returned for persistent cache access.
        """
        self.collection = collection

    def __getitem__(self, name: str) -> _FakeCollection:
        """Return the fake collection for the requested name.

        Args:
            name: Collection name requested by the helper.

        Returns:
            The fake collection.
        """
        assert name == cache_module.PERSISTENT_CACHE_COLLECTION
        return self.collection


def _current_version() -> str:
    return_value = (
        f"{INITIALIZER_POLICY_VERSION}|"
        f"{INITIALIZER_PROMPT_VERSION}|"
        f"{INITIALIZER_AGENT_REGISTRY_VERSION}|"
        f"{INITIALIZER_STRATEGY_SCHEMA_VERSION}"
    )
    return return_value


def _patch_db(monkeypatch: pytest.MonkeyPatch, collection: _FakeCollection) -> None:
    db = _FakeDb(collection)
    monkeypatch.setattr(cache_module, "get_db", AsyncMock(return_value=db))


def test_build_initializer_version_key() -> None:
    """The version key should stay human-readable for MongoDB inspection."""
    version_key = cache_module.build_initializer_version_key()

    assert version_key == _current_version()


@pytest.mark.asyncio
async def test_purge_stale_initializer_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stale initializer rows are deleted without touching current or other rows."""
    collection = _FakeCollection([
        {
            "_id": "stale",
            "cache_name": INITIALIZER_CACHE_NAME,
            "version_key": "old",
        },
        {
            "_id": "current",
            "cache_name": INITIALIZER_CACHE_NAME,
            "version_key": _current_version(),
        },
        {
            "_id": "other",
            "cache_name": "rag2_other",
            "version_key": "old",
        },
    ])
    _patch_db(monkeypatch, collection)

    deleted_count = await cache_module.purge_stale_initializer_entries()

    assert deleted_count == 1
    assert set(collection.docs) == {"current", "other"}


@pytest.mark.asyncio
async def test_load_initializer_entries_orders_by_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hydration load should return current-version rows by hit count and recency."""
    collection = _FakeCollection([
        {
            "_id": "low",
            "cache_name": INITIALIZER_CACHE_NAME,
            "version_key": _current_version(),
            "hit_count": 1,
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "_id": "high_old",
            "cache_name": INITIALIZER_CACHE_NAME,
            "version_key": _current_version(),
            "hit_count": 3,
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "_id": "high_new",
            "cache_name": INITIALIZER_CACHE_NAME,
            "version_key": _current_version(),
            "hit_count": 3,
            "updated_at": "2026-01-02T00:00:00+00:00",
        },
        {
            "_id": "stale",
            "cache_name": INITIALIZER_CACHE_NAME,
            "version_key": "old",
            "hit_count": 999,
            "updated_at": "2026-01-03T00:00:00+00:00",
        },
    ])
    _patch_db(monkeypatch, collection)

    rows = await cache_module.load_initializer_entries(limit=3)

    assert [row["_id"] for row in rows] == ["high_new", "high_old", "low"]


@pytest.mark.asyncio
async def test_upsert_initializer_entry_preserves_hit_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upsert should create hit_count once and never reset existing counters."""
    collection = _FakeCollection()
    _patch_db(monkeypatch, collection)

    await cache_module.upsert_initializer_entry(
        cache_key="key-1",
        result={"unknown_slots": ["slot"], "confidence": 1.0},
        metadata={"stage": "rag_initializer"},
    )
    await cache_module.record_initializer_hit("key-1")
    created_at = collection.docs["key-1"]["created_at"]
    await cache_module.upsert_initializer_entry(
        cache_key="key-1",
        result={"unknown_slots": ["new"], "confidence": 1.0},
        metadata={"stage": "rag_initializer", "new": True},
    )

    row = collection.docs["key-1"]
    assert row["_id"] == "key-1"
    assert row["cache_name"] == INITIALIZER_CACHE_NAME
    assert row["version_key"] == _current_version()
    assert row["hit_count"] == 1
    assert row["created_at"] == created_at
    assert row["result"]["unknown_slots"] == ["new"]
    assert row["metadata"]["new"] is True


@pytest.mark.asyncio
async def test_record_initializer_hit_missing_key_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording a hit for a pruned row should not recreate it."""
    collection = _FakeCollection()
    _patch_db(monkeypatch, collection)

    await cache_module.record_initializer_hit("missing")

    assert collection.docs == {}


@pytest.mark.asyncio
async def test_prune_persistent_entries_deletes_lowest_value_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pruning keeps the highest-hit and most recent persistent rows."""
    collection = _FakeCollection([
        {"_id": "drop_old", "cache_name": INITIALIZER_CACHE_NAME, "hit_count": 0, "updated_at": "1"},
        {"_id": "drop_new", "cache_name": INITIALIZER_CACHE_NAME, "hit_count": 0, "updated_at": "2"},
        {"_id": "keep_hit", "cache_name": INITIALIZER_CACHE_NAME, "hit_count": 5, "updated_at": "1"},
        {"_id": "keep_other", "cache_name": "rag2_other", "hit_count": 0, "updated_at": "1"},
    ])
    _patch_db(monkeypatch, collection)

    deleted_count = await cache_module.prune_persistent_entries(
        cache_name=INITIALIZER_CACHE_NAME,
        max_entries=2,
    )

    assert deleted_count == 1
    assert set(collection.docs) == {"drop_new", "keep_hit", "keep_other"}


@pytest.mark.asyncio
async def test_helpers_swallow_pymongo_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """MongoDB failures should not escape persistent cache helpers."""
    monkeypatch.setattr(
        cache_module,
        "get_db",
        AsyncMock(side_effect=PyMongoError("boom")),
    )

    purged = await cache_module.purge_stale_initializer_entries()
    loaded = await cache_module.load_initializer_entries(limit=5)
    await cache_module.upsert_initializer_entry(
        cache_key="key",
        result={},
        metadata={},
    )
    await cache_module.record_initializer_hit("key")
    pruned = await cache_module.prune_persistent_entries(
        cache_name=INITIALIZER_CACHE_NAME,
        max_entries=1,
    )

    assert purged == 0
    assert loaded == []
    assert pruned == 0


@pytest.mark.asyncio
async def test_service_hydration_inserts_rows_in_reverse_mru_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup hydration should leave the highest-hit row at the LRU MRU end."""
    rows = [
        {
            "_id": "high",
            "result": {"unknown_slots": ["high"], "confidence": 1.0},
            "metadata": {"stage": "rag_initializer"},
        },
        {
            "_id": "low",
            "result": {"unknown_slots": ["low"], "confidence": 1.0},
            "metadata": {"stage": "rag_initializer"},
        },
    ]
    runtime = RAGCache2Runtime(max_entries=10)
    monkeypatch.setattr(service_module, "load_initializer_entries", AsyncMock(return_value=rows))
    monkeypatch.setattr(service_module, "get_rag_cache2_runtime", lambda: runtime)

    loaded_count = await service_module._hydrate_rag_initializer_cache()

    assert loaded_count == 2
    assert list(runtime._entries.keys()) == ["low", "high"]
