from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.rag.cache2_policy import build_user_profile_cache_key
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime
from kazusa_ai_chatbot.rag.user_profile_agent import (
    UserProfileAgent,
    _extract_global_user_id_from_known_facts,
)


def test_extract_global_user_id_uses_native_known_facts_only() -> None:
    """User-profile resolution should use native facts."""

    context = {
        "known_facts": [
            {
                "raw_result": {
                    "global_user_id": "native-user-id",
                    "display_name": "Native User",
                }
            }
        ]
    }

    assert _extract_global_user_id_from_known_facts(context) == "native-user-id"


def test_extract_global_user_id_does_not_parse_stringified_known_facts() -> None:
    """Stringified native facts should not be repaired inside domain logic."""

    context = {
        "known_facts": [
            '{"raw_result":{"global_user_id":"stringified-user-id"}}',
        ]
    }

    assert _extract_global_user_id_from_known_facts(context) == ""


def test_user_profile_cache_key_varies_by_local_date() -> None:
    """User-profile cache must not reuse stale due-state projections."""

    first_key = build_user_profile_cache_key(
        "user-1",
        current_local_date="2026-05-06",
    )
    second_key = build_user_profile_cache_key(
        "user-1",
        current_local_date="2026-05-07",
    )

    assert first_key != second_key


@pytest.mark.asyncio
async def test_user_profile_agent_reads_character_profile_for_character_gid(monkeypatch) -> None:
    """The character global user ID should read character_state, not user memories."""
    get_character_profile = AsyncMock(
        return_value={
            "name": "Kazusa",
            "description": "Public character description.",
            "age": 15,
            "birthday": "August 5",
            "backstory": "Public backstory.",
            "global_vibe": "private runtime vibe",
            "boundary_profile": {"self_integrity": 0.6},
            "self_image": {
                "milestones": [{"event": "Joined the chat"}],
                "recent_window": [],
                "historical_summary": "Stable self-image summary.",
            },
        }
    )
    get_text_embedding = AsyncMock(side_effect=AssertionError("embedding should not run"))
    user_image_retriever_agent = AsyncMock(
        side_effect=AssertionError("user-image retriever should not run")
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.user_profile_agent.get_character_profile",
        get_character_profile,
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.user_profile_agent.get_text_embedding",
        get_text_embedding,
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.user_profile_agent.user_image_retriever_agent",
        user_image_retriever_agent,
    )

    agent = UserProfileAgent(cache_runtime=RAGCache2Runtime(max_entries=10))
    result = await agent.run(
        task="retrieve character profile",
        context={
            "known_facts": [
                {
                    "raw_result": {
                        "global_user_id": CHARACTER_GLOBAL_USER_ID,
                        "display_name": "Kazusa",
                    }
                }
            ]
        },
    )

    assert result["resolved"] is True
    assert result["result"]["name"] == "Kazusa"
    assert result["result"]["self_image"]["historical_summary"] == "Stable self-image summary."
    assert "global_vibe" not in result["result"]
    assert "boundary_profile" not in result["result"]
    assert result["cache"]["reason"] == "miss_stored"
    get_character_profile.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_profile_agent_keeps_character_profile_cacheable(monkeypatch) -> None:
    """A second character profile read should come from Cache 2."""
    get_character_profile = AsyncMock(return_value={"name": "Kazusa"})
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.user_profile_agent.get_character_profile",
        get_character_profile,
    )

    agent = UserProfileAgent(cache_runtime=RAGCache2Runtime(max_entries=10))
    context = {
        "known_facts": [
            {
                "raw_result": {
                    "global_user_id": CHARACTER_GLOBAL_USER_ID,
                    "display_name": "Kazusa",
                }
            }
        ]
    }

    first = await agent.run(task="retrieve character profile", context=context)
    second = await agent.run(task="retrieve character profile", context=context)

    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True
    assert second["result"]["name"] == "Kazusa"
    get_character_profile.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_profile_agent_ignores_stale_user_profile_cache_for_character_gid(
    monkeypatch,
) -> None:
    """A character read should not reuse old user-image cache entries."""
    runtime = RAGCache2Runtime(max_entries=10)
    stale_user_profile_key = build_user_profile_cache_key(CHARACTER_GLOBAL_USER_ID)
    await runtime.store(
        cache_key=stale_user_profile_key,
        cache_name="rag2_user_profile_agent",
        result={
            "user_memory_context": {
                "stable_patterns": [],
                "recent_shifts": [],
                "objective_facts": [],
                "milestones": [],
                "active_commitments": [],
            },
        },
        dependencies=[],
        metadata={},
    )
    get_character_profile = AsyncMock(return_value={"name": "Kazusa"})
    monkeypatch.setattr(
        "kazusa_ai_chatbot.rag.user_profile_agent.get_character_profile",
        get_character_profile,
    )

    agent = UserProfileAgent(cache_runtime=runtime)
    result = await agent.run(
        task="retrieve character profile",
        context={
            "known_facts": [
                {
                    "raw_result": {
                        "global_user_id": CHARACTER_GLOBAL_USER_ID,
                        "display_name": "Kazusa",
                    }
                }
            ]
        },
    )

    assert result["cache"]["hit"] is False
    assert result["result"] == {"name": "Kazusa"}
    get_character_profile.assert_awaited_once()
