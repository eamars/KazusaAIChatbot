"""Tests for db_writer Cache2 invalidation events."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.cognition_episode import build_user_message_episode
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.consolidation.target import (
    build_consolidation_target_plan,
)
from kazusa_ai_chatbot.consolidation import persistence as persistence_module
from kazusa_ai_chatbot.consolidation.origin import (
    build_user_message_consolidation_origin,
)
from kazusa_ai_chatbot.time_boundary import local_time_context_from_storage_utc

STORAGE_TIMESTAMP_UTC = "2026-04-26T12:00:00+00:00"
COGNITION_UPDATED_AT = "2026-04-26T12:00:00Z"


def _consolidation_origin() -> dict:
    """Build valid user-message origin metadata for direct db_writer calls.

    Returns:
        Valid user-message consolidation origin metadata.
    """
    episode = build_user_message_episode(
        episode_id="episode-1",
        origin={
            "platform": "qq",
            "platform_message_id": "msg-1",
            "active_turn_platform_message_ids": ["msg-1"],
            "active_turn_conversation_row_ids": ["conversation-row-1"],
        },
        target_scope={
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "channel_type": "group",
            "current_platform_user_id": "platform-user-1",
            "current_global_user_id": "user-1",
            "current_display_name": "User",
            "target_addressed_user_ids": ["user-1"],
            "target_broadcast": False,
        },
        dialog_percept={
            "schema_version": "percept.v1",
            "percept_kind": "dialog",
            "source_kind": "dialog",
            "source_id": "msg-1",
            "content": {"semantic_text": "remember tea"},
            "observed_at": STORAGE_TIMESTAMP_UTC,
        },
        media_percepts=[],
        evidence_refs=[],
        local_time_context=local_time_context_from_storage_utc(
            STORAGE_TIMESTAMP_UTC,
        ),
        created_at=STORAGE_TIMESTAMP_UTC,
        debug_controls={},
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
        "character_profile": {
            "name": "Kazusa",
            "cognition_state": build_character_production_state(
                updated_at=COGNITION_UPDATED_AT,
            ),
        },
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
        "user_profile": {
            "global_user_id": "user-1",
            "cognition_state": build_acquaintance_user_state(
                global_user_id="user-1",
                updated_at=COGNITION_UPDATED_AT,
            ),
        },
        "relationship_delta": 1,
        "decontextualized_input": "remember tea",
        "consolidation_origin": _consolidation_origin(),
        "enabled_consolidation_write_lanes": [
            "character_state",
            "relationship_profile",
            "user_memory_units",
        ],
    }
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)
    return state


def _patch_writers(monkeypatch) -> MagicMock:
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=1)
    monkeypatch.setattr(
        persistence_module,
        "get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )
    monkeypatch.setattr(
        persistence_module,
        "update_user_memory_units_from_state",
        AsyncMock(return_value=[{"memory_unit_id": "memory-unit-1"}]),
    )
    return runtime


@pytest.mark.asyncio
async def test_db_writer_invalidates_user_profile_after_memory_write(
    monkeypatch,
) -> None:
    runtime = _patch_writers(monkeypatch)

    result = await persistence_module.db_writer(_state())

    sources = [call.args[0].source for call in runtime.invalidate.await_args_list]
    assert sources == ["user_profile"]
    assert result["metadata"]["cache_invalidated"] == ["user_profile"]
    assert result["metadata"]["cache_evicted_count"] == 1
    assert "character_state" not in sources


@pytest.mark.asyncio
async def test_db_writer_never_emits_user_image_source(monkeypatch) -> None:
    runtime = _patch_writers(monkeypatch)

    await persistence_module.db_writer(_state())

    sources = [call.args[0].source for call in runtime.invalidate.await_args_list]
    assert "user_image" not in sources
