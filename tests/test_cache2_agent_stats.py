"""Tests for sanitized Cache2 per-agent lookup statistics."""

from __future__ import annotations

import pytest

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
