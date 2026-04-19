"""Unit tests for ``kazusa_ai_chatbot.rag.cache``.

MongoDB persistence is mocked via ``get_db`` — only in-memory behaviour
is exercised here.  Live persistence is covered by the module's
``test_main`` and by future integration tests.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazusa_ai_chatbot.rag.cache import (
    DEFAULT_SIMILARITY_THRESHOLD,
    RAGCache,
    _cosine_similarity,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_mock_db() -> MagicMock:
    """Mock MongoDB collection whose async methods are no-ops."""
    mock_collection = MagicMock()
    mock_collection.insert_one = AsyncMock()
    mock_collection.update_many = AsyncMock()
    mock_collection.find = MagicMock(return_value=_AsyncIter([]))

    db = MagicMock()
    db.__getitem__.return_value = mock_collection
    return db


class _AsyncIter:
    """Minimal async iterable wrapper for Motor's find() chain."""

    def __init__(self, items):
        self._items = items

    def sort(self, *_args, **_kwargs):
        return self

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _patch_db(mock_db):
    return patch(
        "kazusa_ai_chatbot.rag.cache.__import__", wraps=__import__
    )


# ── Core tests ─────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert _cosine_similarity([], [1.0, 2.0]) == 0.0
        assert _cosine_similarity([1.0], []) == 0.0

    def test_mismatched_lengths(self):
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0


class TestRAGCacheStoreAndRetrieve:
    async def test_store_and_exact_retrieve(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.db.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            emb = [0.1, 0.2, 0.3, 0.4]
            await cache.store(
                embedding=emb,
                results={"facts": ["a", "b"]},
                cache_type="user_facts",
                global_user_id="u1",
                ttl_seconds=60,
            )

            hit = await cache.retrieve_if_similar(
                embedding=emb,
                cache_type="user_facts",
                global_user_id="u1",
            )
            assert hit is not None
            assert hit["results"] == {"facts": ["a", "b"]}
            assert hit["similarity"] == pytest.approx(1.0)

    async def test_miss_below_threshold(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.db.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache(similarity_threshold=0.99)
            await cache.store(
                embedding=[1.0, 0.0, 0.0],
                results={"r": 1},
                cache_type="user_facts",
                global_user_id="u1",
                ttl_seconds=60,
            )

            miss = await cache.retrieve_if_similar(
                embedding=[0.0, 1.0, 0.0],  # orthogonal
                cache_type="user_facts",
                global_user_id="u1",
            )
            assert miss is None

    async def test_cache_type_isolation(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.db.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            emb = [0.5, 0.5, 0.5]
            await cache.store(
                embedding=emb, results={"x": 1},
                cache_type="user_facts", global_user_id="u1", ttl_seconds=60,
            )

            # Same embedding but different cache_type
            miss = await cache.retrieve_if_similar(
                embedding=emb,
                cache_type="internal_memory",
                global_user_id="u1",
            )
            assert miss is None

    async def test_user_isolation(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.db.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            emb = [0.5, 0.5, 0.5]
            await cache.store(
                embedding=emb, results={"x": 1},
                cache_type="user_facts", global_user_id="u1", ttl_seconds=60,
            )

            miss = await cache.retrieve_if_similar(
                embedding=emb,
                cache_type="user_facts",
                global_user_id="u2",
            )
            assert miss is None


class TestRAGCacheTTL:
    async def test_expired_entry_is_miss(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.db.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            await cache.store(
                embedding=[1.0, 0.0],
                results={"x": 1},
                cache_type="user_facts",
                global_user_id="u1",
                ttl_seconds=0,  # immediately expired
            )
            # Let the wall clock advance a tick
            await asyncio.sleep(0.01)
            miss = await cache.retrieve_if_similar(
                embedding=[1.0, 0.0],
                cache_type="user_facts",
                global_user_id="u1",
            )
            assert miss is None

    async def test_expired_entry_lazy_deleted(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.db.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            await cache.store(
                embedding=[1.0, 0.0],
                results={"x": 1},
                cache_type="user_facts",
                global_user_id="u1",
                ttl_seconds=0,
            )
            await asyncio.sleep(0.01)
            await cache.retrieve_if_similar(
                embedding=[1.0, 0.0],
                cache_type="user_facts",
                global_user_id="u1",
            )
            assert cache.get_stats()["size"] == 0


class TestRAGCacheInvalidation:
    async def test_invalidate_pattern_removes_matching(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.db.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            emb = [0.5, 0.5]
            await cache.store(embedding=emb, results={"x": 1},
                              cache_type="user_facts", global_user_id="u1", ttl_seconds=60)
            await cache.store(embedding=emb, results={"x": 2},
                              cache_type="internal_memory", global_user_id="u1", ttl_seconds=60)

            removed = await cache.invalidate_pattern(
                cache_type="user_facts", global_user_id="u1",
            )
            assert removed == 1

            # user_facts invalidated, internal_memory still present
            assert await cache.retrieve_if_similar(
                embedding=emb, cache_type="user_facts", global_user_id="u1",
            ) is None
            assert await cache.retrieve_if_similar(
                embedding=emb, cache_type="internal_memory", global_user_id="u1",
            ) is not None

    async def test_clear_all_user_removes_everything(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.db.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            emb = [0.5, 0.5]
            await cache.store(embedding=emb, results={"x": 1},
                              cache_type="user_facts", global_user_id="u1", ttl_seconds=60)
            await cache.store(embedding=emb, results={"x": 2},
                              cache_type="internal_memory", global_user_id="u1", ttl_seconds=60)
            await cache.store(embedding=emb, results={"x": 3},
                              cache_type="user_facts", global_user_id="u2", ttl_seconds=60)

            removed = await cache.clear_all_user("u1")
            assert removed == 2
            stats = cache.get_stats()
            assert stats["size"] == 1  # u2 still there


class TestRAGCacheStats:
    async def test_hit_miss_stats(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.db.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            emb = [1.0, 0.0]
            await cache.store(embedding=emb, results={"x": 1},
                              cache_type="user_facts", global_user_id="u1", ttl_seconds=60)

            await cache.retrieve_if_similar(
                embedding=emb, cache_type="user_facts", global_user_id="u1")
            await cache.retrieve_if_similar(
                embedding=[0.0, 1.0], cache_type="user_facts", global_user_id="u1")
            await cache.retrieve_if_similar(
                embedding=emb, cache_type="user_facts", global_user_id="u1")

            stats = cache.get_stats()
            assert stats["hits"] == 2
            assert stats["misses"] == 1
            assert stats["hit_rate"] == pytest.approx(2 / 3)


class TestRAGCacheLRU:
    async def test_eviction_at_max_size(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.db.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache(max_size=2)
            await cache.store(embedding=[1.0, 0.0], results={"x": 1},
                              cache_type="user_facts", global_user_id="u1", ttl_seconds=60)
            await cache.store(embedding=[0.0, 1.0], results={"x": 2},
                              cache_type="user_facts", global_user_id="u1", ttl_seconds=60)
            await cache.store(embedding=[0.5, 0.5], results={"x": 3},
                              cache_type="user_facts", global_user_id="u1", ttl_seconds=60)

            stats = cache.get_stats()
            assert stats["size"] == 2
            assert stats["evictions"] == 1


class TestRAGCacheDefaults:
    def test_default_threshold(self):
        cache = RAGCache()
        assert cache.get_stats()["threshold"] == DEFAULT_SIMILARITY_THRESHOLD


class TestRAGCachePersistFailureNonFatal:
    async def test_persist_failure_keeps_in_memory_entry(self):
        """Writes that fail at the DB layer must not prevent in-memory storage."""
        # Mock get_db to raise
        with patch("kazusa_ai_chatbot.db.get_db",
                   AsyncMock(side_effect=RuntimeError("db down"))):
            cache = RAGCache()
            cid = await cache.store(
                embedding=[1.0, 0.0], results={"x": 1},
                cache_type="user_facts", global_user_id="u1", ttl_seconds=60,
            )
            assert cid  # still returned
            hit = await cache.retrieve_if_similar(
                embedding=[1.0, 0.0],
                cache_type="user_facts", global_user_id="u1",
            )
            assert hit is not None
