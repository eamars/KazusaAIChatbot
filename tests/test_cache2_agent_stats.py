"""Tests for sanitized Cache2 per-agent lookup statistics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kazusa_ai_chatbot.rag import cache2_runtime
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime


@pytest.mark.asyncio
async def test_cache2_agent_stats_count_hits_and_misses() -> None:
    """Cache2 should expose sanitized hit/miss counters grouped by agent."""
    runtime = RAGCache2Runtime(max_entries=10)

    missing = await runtime.get(
        "user-profile-key",
        cache_name="rag2_user_profile_agent",
        agent_name="user_profile_agent",
    )
    await runtime.store(
        cache_key="user-profile-key",
        cache_name="rag2_user_profile_agent",
        result={"private": "not exposed"},
        dependencies=[],
        metadata={"agent_name": "user_profile_agent"},
    )
    cached = await runtime.get(
        "user-profile-key",
        cache_name="rag2_user_profile_agent",
        agent_name="user_profile_agent",
    )

    assert missing is None
    assert cached == {"private": "not exposed"}
    assert runtime.get_agent_stats() == [
        {
            "agent_name": "user_profile_agent",
            "hit_count": 1,
            "miss_count": 1,
            "hit_rate": 0.5,
        }
    ]


@pytest.mark.asyncio
async def test_cache2_agent_stats_payload_is_sanitized() -> None:
    """Agent stats should not expose cache keys, dependencies, or result data."""
    runtime = RAGCache2Runtime(max_entries=10)

    await runtime.get(
        "conversation-key",
        cache_name="rag2_conversation_search_agent",
        agent_name="conversation_search_agent",
    )

    stats = runtime.get_agent_stats()

    assert stats == [
        {
            "agent_name": "conversation_search_agent",
            "hit_count": 0,
            "miss_count": 1,
            "hit_rate": 0.0,
        }
    ]
    assert set(stats[0]) == {"agent_name", "hit_count", "miss_count", "hit_rate"}


@pytest.mark.asyncio
async def test_cache2_ttl_expires_only_entries_with_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entries without TTL should survive while TTL-bound entries expire."""
    current_time = datetime(2026, 7, 4, tzinfo=timezone.utc)

    def now_utc() -> datetime:
        return current_time

    monkeypatch.setattr(cache2_runtime, "_now_utc", now_utc)
    runtime = RAGCache2Runtime(max_entries=10)

    await runtime.store(
        cache_key="stable-key",
        cache_name="rag3_stable",
        result={"value": "stable"},
        dependencies=[],
    )
    await runtime.store(
        cache_key="live-key",
        cache_name="rag3_live",
        result={"value": "live"},
        dependencies=[],
        ttl_seconds=60,
    )

    current_time += timedelta(seconds=61)
    stable_result = await runtime.get("stable-key", cache_name="rag3_stable")
    expired_result = await runtime.get("live-key", cache_name="rag3_live")

    assert stable_result == {"value": "stable"}
    assert expired_result is None
    assert runtime.get_stats()["expirations"] == 1
