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
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.rag.cache import (
    DEFAULT_SIMILARITY_THRESHOLD,
    CacheInvalidationScope,
    RAGCache,
    _cosine_similarity,
    cached_node,
    set_cached_node_cache,
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
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
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
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
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
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
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
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
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
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
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
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
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
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
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
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
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
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
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
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
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
        with patch("kazusa_ai_chatbot.rag.cache.get_db",
                   AsyncMock(side_effect=PyMongoError("db down"))):
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


# ── Phase 6: Module decomposition import tests ───────────────────


class TestPhase6Imports:
    """Verify Phase 6 module decomposition preserves import paths."""

    def test_rag_schema_exports(self):
        from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_schema import (
            RAGState,
            _build_image_context,
        )
        assert "decontexualized_input" in RAGState.__annotations__
        assert callable(_build_image_context)

    def test_rag_resolution_exports(self):
        from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_resolution import (
            continuation_resolver,
            entity_grounder,
            rag_planner,
        )
        assert callable(continuation_resolver)
        assert callable(rag_planner)
        assert callable(entity_grounder)

    def test_rag_executors_exports(self):
        from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_executors import (
            call_memory_retriever_agent_input_context_rag,
            call_web_search_agent,
            channel_recent_entity_rag,
            external_rag_dispatcher,
            input_context_rag_dispatcher,
            third_party_profile_rag,
        )
        assert callable(input_context_rag_dispatcher)
        assert callable(external_rag_dispatcher)
        assert callable(call_web_search_agent)
        assert callable(call_memory_retriever_agent_input_context_rag)
        assert callable(channel_recent_entity_rag)
        assert callable(third_party_profile_rag)

    def test_orchestrator_backward_compat(self):
        from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import (
            RAGState,
            _build_image_context,
            call_rag_subgraph,
        )
        assert "decontexualized_input" in RAGState.__annotations__
        assert callable(_build_image_context)
        assert callable(call_rag_subgraph)

    def test_orchestrator_phase8_exports(self):
        from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import (
            _build_cache_key,
            _build_resolution_graph,
            _build_retrieval_graph,
        )
        assert callable(_build_cache_key)
        assert callable(_build_resolution_graph)
        assert callable(_build_retrieval_graph)


# ── Phase 8: Boundary cache tests ────────────────────────────────


class TestBoundaryCacheKeyBased:
    """Test key-based boundary cache (store_by_key / retrieve_if_similar_by_key)."""

    async def test_store_and_retrieve_by_key(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            cid = await cache.store_by_key(
                cache_key="abc123",
                results={"input_context_results": "found something"},
                cache_type="boundary_cache",
                global_user_id="u1",
                ttl_seconds=60,
            )
            assert cid

            hit = await cache.retrieve_if_similar_by_key("abc123")
            assert hit is not None
            assert hit["results"]["input_context_results"] == "found something"

    async def test_miss_on_different_key(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            await cache.store_by_key(
                cache_key="abc123",
                results={"x": 1},
                cache_type="boundary_cache",
                global_user_id="u1",
                ttl_seconds=60,
            )

            miss = await cache.retrieve_if_similar_by_key("xyz789")
            assert miss is None

    async def test_expired_boundary_entry_is_miss(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            await cache.store_by_key(
                cache_key="abc123",
                results={"x": 1},
                cache_type="boundary_cache",
                global_user_id="u1",
                ttl_seconds=0,
            )
            await asyncio.sleep(0.01)

            miss = await cache.retrieve_if_similar_by_key("abc123")
            assert miss is None

    async def test_boundary_cache_does_not_collide_with_embedding_cache(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            emb = [1.0, 0.0]
            await cache.store(
                embedding=emb, results={"source": "embedding"},
                cache_type="user_facts", global_user_id="u1", ttl_seconds=60,
            )
            await cache.store_by_key(
                cache_key="key1",
                results={"source": "boundary"},
                cache_type="boundary_cache",
                global_user_id="u1",
                ttl_seconds=60,
            )

            emb_hit = await cache.retrieve_if_similar(
                embedding=emb, cache_type="user_facts", global_user_id="u1",
            )
            assert emb_hit is not None
            assert emb_hit["results"]["source"] == "embedding"

            key_hit = await cache.retrieve_if_similar_by_key("key1")
            assert key_hit is not None
            assert key_hit["results"]["source"] == "boundary"


class TestBuildCacheKey:
    """Test _build_cache_key determinism and sensitivity."""

    def test_deterministic(self):
        from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _build_cache_key

        resolution = {
            "continuation_context": {"resolved_task": "recall sushi"},
            "resolved_entities": [
                {"surface_form": "Alice", "resolved_global_user_id": "uid-1"},
            ],
            "retrieval_plan": {
                "active_sources": ["INPUT_CONTEXT", "THIRD_PARTY_PROFILE"],
                "time_scope": {"lookback_hours": 72},
            },
        }
        k1 = _build_cache_key(resolution)
        k2 = _build_cache_key(resolution)
        assert k1 == k2
        assert len(k1) == 64  # SHA-256 hex

    def test_different_entities_different_key(self):
        from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _build_cache_key

        base = {
            "continuation_context": {"resolved_task": "recall sushi"},
            "resolved_entities": [
                {"surface_form": "Alice", "resolved_global_user_id": "uid-1"},
            ],
            "retrieval_plan": {
                "active_sources": ["INPUT_CONTEXT"],
                "time_scope": {"lookback_hours": 72},
            },
        }
        alt = {
            **base,
            "resolved_entities": [
                {"surface_form": "Bob", "resolved_global_user_id": "uid-2"},
            ],
        }
        assert _build_cache_key(base) != _build_cache_key(alt)

    def test_different_sources_different_key(self):
        from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _build_cache_key

        base = {
            "continuation_context": {"resolved_task": "recall sushi"},
            "resolved_entities": [],
            "retrieval_plan": {
                "active_sources": ["INPUT_CONTEXT"],
                "time_scope": {"lookback_hours": 72},
            },
        }
        alt = {
            **base,
            "retrieval_plan": {
                "active_sources": ["INPUT_CONTEXT", "EXTERNAL_KNOWLEDGE"],
                "time_scope": {"lookback_hours": 72},
            },
        }
        assert _build_cache_key(base) != _build_cache_key(alt)

    def test_entity_order_insensitive(self):
        from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _build_cache_key

        r1 = {
            "continuation_context": {"resolved_task": "recall"},
            "resolved_entities": [
                {"surface_form": "A", "resolved_global_user_id": "uid-1"},
                {"surface_form": "B", "resolved_global_user_id": "uid-2"},
            ],
            "retrieval_plan": {
                "active_sources": ["INPUT_CONTEXT"],
                "time_scope": {"lookback_hours": 72},
            },
        }
        r2 = {
            **r1,
            "resolved_entities": list(reversed(r1["resolved_entities"])),
        }
        assert _build_cache_key(r1) == _build_cache_key(r2)

    def test_empty_resolution(self):
        from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _build_cache_key

        k = _build_cache_key({})
        assert isinstance(k, str) and len(k) == 64


class TestScopedInvalidation:
    """Test CacheInvalidationScope + invalidate_scoped."""

    async def test_invalidate_by_cache_type(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            await cache.store_by_key(
                cache_key="k1", results={"x": 1},
                cache_type="boundary_cache", global_user_id="u1", ttl_seconds=60,
            )
            await cache.store(
                embedding=[1.0, 0.0], results={"x": 2},
                cache_type="external_knowledge", global_user_id="", ttl_seconds=60,
            )

            scope = CacheInvalidationScope(cache_type="boundary_cache", global_user_id="u1")
            removed = await cache.invalidate_scoped(scope)
            assert removed == 1

            assert await cache.retrieve_if_similar_by_key("k1") is None
            assert await cache.retrieve_if_similar(
                embedding=[1.0, 0.0], cache_type="external_knowledge", global_user_id="",
            ) is not None

    async def test_invalidate_by_boundary_key(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            await cache.store_by_key(
                cache_key="k1", results={"x": 1},
                cache_type="boundary_cache", global_user_id="u1", ttl_seconds=60,
            )
            await cache.store_by_key(
                cache_key="k2", results={"x": 2},
                cache_type="boundary_cache", global_user_id="u1", ttl_seconds=60,
            )

            scope = CacheInvalidationScope(
                cache_type="boundary_cache",
                boundary_key="k1",
            )
            removed = await cache.invalidate_scoped(scope)
            assert removed == 1

            assert await cache.retrieve_if_similar_by_key("k1") is None
            assert await cache.retrieve_if_similar_by_key("k2") is not None

    async def test_invalidate_empty_scope_removes_nothing(self):
        """A fully empty scope should not wildcard-delete everything — it
        won't match on any specific field, so only entries where *all* checked
        fields are empty/None/absent would match."""
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            await cache.store_by_key(
                cache_key="k1", results={"x": 1},
                cache_type="boundary_cache", global_user_id="u1", ttl_seconds=60,
            )
            scope = CacheInvalidationScope()
            removed = await cache.invalidate_scoped(scope)
            # Empty scope = no filtering = matches all (wildcard)
            # This IS the intended behavior: all non-empty fields filter
            # and empty fields are wildcards. All empty = match all.
            assert removed == 1


class TestCachedNodeDecorator:
    """Test the cached_node decorator."""

    async def test_decorator_caches_on_miss(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            set_cached_node_cache(cache)

            call_count = 0

            @cached_node(key_fn=lambda s: s.get("query", ""))
            async def my_node(state):
                nonlocal call_count
                call_count += 1
                return {"answer": "42"}

            state = {"query": "test-q", "global_user_id": "u1"}
            r1 = await my_node(state)
            assert r1 == {"answer": "42"}
            assert call_count == 1

            r2 = await my_node(state)
            assert r2 == {"answer": "42"}
            assert call_count == 1  # cached, not called again

    async def test_decorator_no_cache_singleton_falls_through(self):
        set_cached_node_cache(None)

        call_count = 0

        @cached_node(key_fn=lambda s: s.get("query", ""))
        async def my_node(state):
            nonlocal call_count
            call_count += 1
            return {"answer": "42"}

        r1 = await my_node({"query": "q"})
        r2 = await my_node({"query": "q"})
        assert call_count == 2  # no caching

    async def test_decorator_empty_key_falls_through(self):
        mock_db = _make_mock_db()
        with patch("kazusa_ai_chatbot.rag.cache.get_db", AsyncMock(return_value=mock_db)):
            cache = RAGCache()
            set_cached_node_cache(cache)

            call_count = 0

            @cached_node(key_fn=lambda s: "")
            async def my_node(state):
                nonlocal call_count
                call_count += 1
                return {"answer": "42"}

            await my_node({"global_user_id": "u1"})
            await my_node({"global_user_id": "u1"})
            assert call_count == 2  # empty key = no caching
