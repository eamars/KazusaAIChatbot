"""Deterministic tests for the media descriptor cache implementation."""

from __future__ import annotations

import base64
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db import rag_cache2_persistent as cache_module
from kazusa_ai_chatbot.rag.cache2_policy import (
    MEDIA_DESCRIPTOR_CACHE_NAME,
    MEDIA_DESCRIPTOR_MODEL_VERSION,
    MEDIA_DESCRIPTOR_PROMPT_VERSION,
    build_media_descriptor_cache_key,
    build_media_descriptor_version_key,
)
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AsyncCursor:
    """Small async cursor fake reused from test_rag_cache2_persistent."""

    def __init__(self, docs: list[dict]) -> None:
        self._docs = [dict(doc) for doc in docs]
        self._limit: int | None = None

    def sort(self, sort_spec: list[tuple[str, int]]) -> "_AsyncCursor":
        for field, direction in reversed(sort_spec):
            self._docs.sort(
                key=lambda doc: doc.get(field, ""),
                reverse=direction < 0,
            )
        return self

    def limit(self, limit: int) -> "_AsyncCursor":
        self._limit = limit
        return self

    async def to_list(self, length: int) -> list[dict]:
        effective_limit = self._limit if self._limit is not None else length
        return_value = [dict(doc) for doc in self._docs[:effective_limit]]
        return return_value


class _FakeCollection:
    """In-memory collection fake for deterministic persistent-cache tests."""

    def __init__(self, docs: list[dict] | None = None) -> None:
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
        ids_to_delete = [
            doc_id
            for doc_id, doc in self.docs.items()
            if self._matches(doc, query)
        ]
        for doc_id in ids_to_delete:
            del self.docs[doc_id]
        return SimpleNamespace(deleted_count=len(ids_to_delete))

    def find(self, query: dict, projection: dict | None = None) -> _AsyncCursor:
        docs = [dict(doc) for doc in self.docs.values() if self._matches(doc, query)]
        if projection == {"_id": 1}:
            docs = [{"_id": doc["_id"]} for doc in docs]
        return _AsyncCursor(docs)

    async def update_one(
        self,
        query: dict,
        update: dict,
        *,
        upsert: bool = False,
    ) -> SimpleNamespace:
        doc_id = query["_id"]
        doc = self.docs.get(doc_id)
        inserted = False
        if doc is None:
            if not upsert:
                return SimpleNamespace(matched_count=0)
            doc = {"_id": doc_id}
            doc.update(update.get("$setOnInsert", {}))
            self.docs[doc_id] = doc
            inserted = True

        doc.update(update.get("$set", {}))
        for field, amount in update.get("$inc", {}).items():
            doc[field] = int(doc.get(field, 0)) + int(amount)

        return SimpleNamespace(matched_count=0 if inserted else 1)

    async def count_documents(self, query: dict) -> int:
        return sum(1 for doc in self.docs.values() if self._matches(doc, query))


class _FakeDb:
    def __init__(self, collection: _FakeCollection) -> None:
        self.collection = collection

    def __getitem__(self, name: str) -> _FakeCollection:
        assert name == cache_module.PERSISTENT_CACHE_COLLECTION
        return self.collection


def _current_media_version() -> str:
    return f"{MEDIA_DESCRIPTOR_PROMPT_VERSION}|{MEDIA_DESCRIPTOR_MODEL_VERSION}"


def _patch_db(monkeypatch: pytest.MonkeyPatch, collection: _FakeCollection) -> None:
    db = _FakeDb(collection)
    monkeypatch.setattr(cache_module, "get_db", AsyncMock(return_value=db))


# ---------------------------------------------------------------------------
# cache2_policy tests
# ---------------------------------------------------------------------------


def test_build_media_descriptor_version_key() -> None:
    """Version key is a human-readable pipe-joined string."""
    version_key = build_media_descriptor_version_key()
    assert version_key == _current_media_version()
    assert "|" in version_key


def test_build_media_descriptor_cache_key_deterministic() -> None:
    """Same inputs produce the same key."""
    key1 = build_media_descriptor_cache_key(
        content_type="image/png",
        content_hash="abc123",
    )
    key2 = build_media_descriptor_cache_key(
        content_type="image/png",
        content_hash="abc123",
    )
    assert key1 == key2
    assert isinstance(key1, str)
    assert len(key1) == 64  # SHA-256 hex digest


def test_build_media_descriptor_cache_key_varies_by_content_type() -> None:
    """Different content types produce different keys."""
    key_png = build_media_descriptor_cache_key(
        content_type="image/png",
        content_hash="abc123",
    )
    key_jpeg = build_media_descriptor_cache_key(
        content_type="image/jpeg",
        content_hash="abc123",
    )
    assert key_png != key_jpeg


def test_build_media_descriptor_cache_key_varies_by_hash() -> None:
    """Different content hashes produce different keys."""
    key_a = build_media_descriptor_cache_key(
        content_type="image/png",
        content_hash="aaa",
    )
    key_b = build_media_descriptor_cache_key(
        content_type="image/png",
        content_hash="bbb",
    )
    assert key_a != key_b


# ---------------------------------------------------------------------------
# rag_cache2_persistent media descriptor tests
# ---------------------------------------------------------------------------


def test_persistent_build_media_descriptor_version_key() -> None:
    """Persistent module version key matches policy module."""
    version_key = cache_module.build_media_descriptor_version_key()
    assert version_key == _current_media_version()


@pytest.mark.asyncio
async def test_purge_stale_media_descriptor_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale media descriptor rows are deleted; current and other rows survive."""
    collection = _FakeCollection([
        {
            "_id": "stale",
            "cache_name": MEDIA_DESCRIPTOR_CACHE_NAME,
            "version_key": "old",
        },
        {
            "_id": "current",
            "cache_name": MEDIA_DESCRIPTOR_CACHE_NAME,
            "version_key": _current_media_version(),
        },
        {
            "_id": "other_namespace",
            "cache_name": "rag2_initializer",
            "version_key": "old",
        },
    ])
    _patch_db(monkeypatch, collection)

    deleted_count = await cache_module.purge_stale_media_descriptor_entries()

    assert deleted_count == 1
    assert set(collection.docs) == {"current", "other_namespace"}


@pytest.mark.asyncio
async def test_load_media_descriptor_entries_orders_by_updated_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydration load returns current-version rows by updated_at descending."""
    collection = _FakeCollection([
        {
            "_id": "old",
            "cache_name": MEDIA_DESCRIPTOR_CACHE_NAME,
            "version_key": _current_media_version(),
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "_id": "new",
            "cache_name": MEDIA_DESCRIPTOR_CACHE_NAME,
            "version_key": _current_media_version(),
            "updated_at": "2026-01-03T00:00:00+00:00",
        },
        {
            "_id": "mid",
            "cache_name": MEDIA_DESCRIPTOR_CACHE_NAME,
            "version_key": _current_media_version(),
            "updated_at": "2026-01-02T00:00:00+00:00",
        },
        {
            "_id": "stale",
            "cache_name": MEDIA_DESCRIPTOR_CACHE_NAME,
            "version_key": "old",
            "updated_at": "2026-01-04T00:00:00+00:00",
        },
    ])
    _patch_db(monkeypatch, collection)

    rows = await cache_module.load_media_descriptor_entries(limit=3)

    assert [row["_id"] for row in rows] == ["new", "mid", "old"]


@pytest.mark.asyncio
async def test_upsert_media_descriptor_entry_preserves_hit_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upsert creates hit_count once and never resets existing counters."""
    collection = _FakeCollection()
    _patch_db(monkeypatch, collection)

    await cache_module.upsert_media_descriptor_entry(
        cache_key="md-key-1",
        result={"description": "a cat"},
        metadata={"content_type": "image/png"},
    )
    await cache_module.record_media_descriptor_hit("md-key-1")
    created_at = collection.docs["md-key-1"]["created_at"]

    await cache_module.upsert_media_descriptor_entry(
        cache_key="md-key-1",
        result={"description": "a dog"},
        metadata={"content_type": "image/png"},
    )

    row = collection.docs["md-key-1"]
    assert row["_id"] == "md-key-1"
    assert row["cache_name"] == MEDIA_DESCRIPTOR_CACHE_NAME
    assert row["version_key"] == _current_media_version()
    assert row["hit_count"] == 1
    assert row["created_at"] == created_at
    assert row["result"]["description"] == "a dog"


@pytest.mark.asyncio
async def test_record_media_descriptor_hit_refreshes_updated_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording a hit increments hit_count and refreshes updated_at."""
    collection = _FakeCollection()
    _patch_db(monkeypatch, collection)

    await cache_module.upsert_media_descriptor_entry(
        cache_key="md-hit",
        result={"description": "x"},
        metadata={},
    )
    original_updated = collection.docs["md-hit"]["updated_at"]

    await cache_module.record_media_descriptor_hit("md-hit")

    row = collection.docs["md-hit"]
    assert row["hit_count"] == 1
    assert row["updated_at"] >= original_updated


@pytest.mark.asyncio
async def test_record_media_descriptor_hit_missing_key_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording a hit for a missing row should not recreate it."""
    collection = _FakeCollection()
    _patch_db(monkeypatch, collection)

    await cache_module.record_media_descriptor_hit("missing")

    assert collection.docs == {}


@pytest.mark.asyncio
async def test_prune_media_descriptor_entries_deletes_oldest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pruning keeps the most recently accessed media descriptor rows."""
    collection = _FakeCollection([
        {
            "_id": "oldest",
            "cache_name": MEDIA_DESCRIPTOR_CACHE_NAME,
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "_id": "middle",
            "cache_name": MEDIA_DESCRIPTOR_CACHE_NAME,
            "updated_at": "2026-01-02T00:00:00+00:00",
        },
        {
            "_id": "newest",
            "cache_name": MEDIA_DESCRIPTOR_CACHE_NAME,
            "updated_at": "2026-01-03T00:00:00+00:00",
        },
        {
            "_id": "other_ns",
            "cache_name": "rag2_other",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    ])
    _patch_db(monkeypatch, collection)

    deleted_count = await cache_module.prune_media_descriptor_entries(max_entries=2)

    assert deleted_count == 1
    assert set(collection.docs) == {"middle", "newest", "other_ns"}


@pytest.mark.asyncio
async def test_media_descriptor_helpers_swallow_pymongo_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MongoDB failures should not escape media descriptor persistent helpers."""
    monkeypatch.setattr(
        cache_module,
        "get_db",
        AsyncMock(side_effect=PyMongoError("boom")),
    )

    purged = await cache_module.purge_stale_media_descriptor_entries()
    loaded = await cache_module.load_media_descriptor_entries(limit=5)
    await cache_module.upsert_media_descriptor_entry(
        cache_key="key",
        result={},
        metadata={},
    )
    await cache_module.record_media_descriptor_hit("key")
    pruned = await cache_module.prune_media_descriptor_entries(max_entries=1)

    assert purged == 0
    assert loaded == []
    assert pruned == 0


# ---------------------------------------------------------------------------
# Cache hydration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hydrate_media_descriptor_cache_inserts_rows_in_reverse_mru_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup hydration leaves the most recently accessed row at the LRU MRU end."""
    from kazusa_ai_chatbot import service as service_module

    rows = [
        {
            "_id": "recent",
            "result": {"description": "recent image"},
            "metadata": {"content_type": "image/png"},
        },
        {
            "_id": "older",
            "result": {"description": "older image"},
            "metadata": {"content_type": "image/jpeg"},
        },
    ]
    runtime = RAGCache2Runtime(max_entries=10)
    monkeypatch.setattr(
        service_module,
        "load_media_descriptor_entries",
        AsyncMock(return_value=rows),
    )
    monkeypatch.setattr(service_module, "get_rag_cache2_runtime", lambda: runtime)

    loaded_count = await service_module._hydrate_media_descriptor_cache()

    assert loaded_count == 2
    assert list(runtime._entries.keys()) == ["older", "recent"]


# ---------------------------------------------------------------------------
# In-memory runtime integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_returns_deep_copy() -> None:
    """A cache hit should return a deep copy so mutation does not affect the cache."""
    runtime = RAGCache2Runtime(max_entries=10)
    cache_key = build_media_descriptor_cache_key(
        content_type="image/png",
        content_hash="deadbeef",
    )
    original_result = {"description": "a cat", "visible_text": ["hello"]}
    await runtime.store(
        cache_key=cache_key,
        cache_name=MEDIA_DESCRIPTOR_CACHE_NAME,
        result=original_result,
        dependencies=[],
    )

    hit = await runtime.get(
        cache_key,
        cache_name=MEDIA_DESCRIPTOR_CACHE_NAME,
        agent_name="media_descriptor",
    )

    assert hit == original_result
    hit["description"] = "mutated"

    hit2 = await runtime.get(
        cache_key,
        cache_name=MEDIA_DESCRIPTOR_CACHE_NAME,
        agent_name="media_descriptor",
    )
    assert hit2["description"] == "a cat"


@pytest.mark.asyncio
async def test_agent_stats_tracked_for_media_descriptor() -> None:
    """Cache2 runtime should track hit/miss stats under the media_descriptor agent name."""
    runtime = RAGCache2Runtime(max_entries=10)
    cache_key = build_media_descriptor_cache_key(
        content_type="image/png",
        content_hash="abc",
    )

    await runtime.get(
        cache_key,
        cache_name=MEDIA_DESCRIPTOR_CACHE_NAME,
        agent_name="media_descriptor",
    )

    await runtime.store(
        cache_key=cache_key,
        cache_name=MEDIA_DESCRIPTOR_CACHE_NAME,
        result={"description": "test"},
        dependencies=[],
    )

    await runtime.get(
        cache_key,
        cache_name=MEDIA_DESCRIPTOR_CACHE_NAME,
        agent_name="media_descriptor",
    )

    stats = runtime.get_agent_stats()
    media_stats = [s for s in stats if s["agent_name"] == "media_descriptor"]
    assert len(media_stats) == 1
    assert media_stats[0]["hit_count"] == 1
    assert media_stats[0]["miss_count"] == 1
