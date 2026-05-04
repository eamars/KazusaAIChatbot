"""Tests for persistent-memory Cache2 dependency invalidation."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent
from kazusa_ai_chatbot.rag.cache2_policy import (
    PERSISTENT_MEMORY_SEARCH_CACHE_NAME,
    build_persistent_memory_keyword_dependencies,
    build_persistent_memory_search_dependencies,
)
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime


def test_persistent_memory_dependencies_use_memory_source() -> None:
    """Persistent-memory helper caches depend on memory writes."""
    search_deps = build_persistent_memory_search_dependencies({
        "source_global_user_id": "user-1",
    })
    keyword_deps = build_persistent_memory_keyword_dependencies({})

    assert search_deps[0].source == "memory"
    assert search_deps[0].global_user_id == "user-1"
    assert keyword_deps[0].source == "memory"


@pytest.mark.asyncio
async def test_memory_invalidation_event_removes_persistent_memory_entry() -> None:
    """A memory write invalidates matching persistent-memory cache entries."""
    runtime = RAGCache2Runtime(max_entries=5)
    await runtime.store(
        cache_key="cache-key",
        cache_name=PERSISTENT_MEMORY_SEARCH_CACHE_NAME,
        result={"rows": [1]},
        dependencies=build_persistent_memory_search_dependencies({
            "source_global_user_id": "user-1",
        }),
    )

    removed = await runtime.invalidate(
        CacheInvalidationEvent(
            source="memory",
            global_user_id="user-1",
            reason="memory_unit_inserted",
        )
    )

    assert removed == 1
    assert await runtime.get("cache-key") is None
