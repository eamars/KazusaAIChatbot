"""Tests for db_writer Cache2 invalidation events."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "Retired profile and character-state writer assertions replaced by "
        "V2 persistence tests"
    )
)

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.consolidation.target import (
    build_consolidation_target_plan,
)
from kazusa_ai_chatbot.consolidation import persistence as persistence_module
from kazusa_ai_chatbot.consolidation.origin import (
    build_user_message_consolidation_origin,
)
from kazusa_ai_chatbot.time_boundary import local_time_context_from_storage_utc

STORAGE_TIMESTAMP_UTC = "2026-04-26T12:00:00+00:00"


def _consolidation_origin() -> dict:
    """Build valid user-message origin metadata for direct db_writer calls.

    Returns:
        Valid user-message consolidation origin metadata.
    """
    episode = build_text_chat_cognitive_episode(
        episode_id="episode-1",
        percept_id="percept-1",
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=local_time_context_from_storage_utc(
            STORAGE_TIMESTAMP_UTC,
        ),
        user_input="remember tea",
        platform="qq",
        platform_channel_id="chan-1",
        channel_type="group",
        platform_message_id="msg-1",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="User",
        active_turn_platform_message_ids=["msg-1"],
        active_turn_conversation_row_ids=["conversation-row-1"],
        debug_modes={},
    )
    origin = build_user_message_consolidation_origin(episode=episode)
    assert origin["storage_timestamp_utc"] == STORAGE_TIMESTAMP_UTC
    assert "timestamp" not in origin
    return origin


def _state() -> dict:
    state = {
        "storage_timestamp_utc": STORAGE_TIMESTAMP_UTC,
        "local_time_context": local_time_context_from_storage_utc(
            STORAGE_TIMESTAMP_UTC,
        ),
        "global_user_id": "user-1",
        "user_name": "User",
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_message_id": "msg-1",
        "character_profile": {"name": "Kazusa"},
        "metadata": {},
        "mood": "neutral",
        "vibe_check": "",
        "character_reflection": "",
        "subjective_appraisals": ["User sounded happy"],
        "interaction_subtext": "test",
        "semantic_relationship_projection": "friendly",
        "new_facts": [{
            "description": "User likes tea",
            "category": "preference",
            "dedup_key": "likes_tea",
        }],
        "future_promises": [],
        "user_profile": {"global_user_id": "user-1", "relationship_state": 500},
        "relationship_delta": 1,
        "decontexualized_input": "remember tea",
        "consolidation_origin": _consolidation_origin(),
        "enabled_consolidation_write_lanes": [
            "character_state",
            "relationship_profile",
            "user_memory_units",
        ],
    }
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)
    return state


def _patch_writers(monkeypatch, *, character_image=None) -> MagicMock:
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=1)
    monkeypatch.setattr(persistence_module, "get_rag_cache2_runtime", MagicMock(return_value=runtime))
    monkeypatch.setattr(persistence_module, "upsert_character_state", AsyncMock())
    monkeypatch.setattr(persistence_module, "update_semantic_relationship_projection", AsyncMock())
    monkeypatch.setattr(persistence_module, "update_relationship_state", AsyncMock())
    monkeypatch.setattr(
        persistence_module,
        "get_character_runtime_state",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(persistence_module, "upsert_character_self_image", AsyncMock())
    monkeypatch.setattr(persistence_module, "_update_character_image", AsyncMock(return_value=character_image))
    monkeypatch.setattr(persistence_module, "update_user_memory_units_from_state", AsyncMock(return_value=[]))
    return runtime


@pytest.mark.asyncio
async def test_db_writer_emits_user_profile_and_character_state_events(monkeypatch) -> None:
    runtime = _patch_writers(
        monkeypatch,
        character_image={"recent_window": []},
    )

    result = await persistence_module.db_writer(_state())

    sources = [call.args[0].source for call in runtime.invalidate.await_args_list]
    assert sources == ["user_profile", "character_state"]
    assert result["metadata"]["cache_invalidated"] == ["user_profile", "character_state"]
    assert result["metadata"]["cache_evicted_count"] == 2


@pytest.mark.asyncio
async def test_db_writer_never_emits_user_image_source(monkeypatch) -> None:
    runtime = _patch_writers(monkeypatch)

    await persistence_module.db_writer(_state())

    sources = [call.args[0].source for call in runtime.invalidate.await_args_list]
    assert "user_image" not in sources
