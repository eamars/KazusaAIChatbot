"""Tests for db_writer Cache2 invalidation events."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator_persistence as persistence_module


def _state() -> dict:
    return {
        "timestamp": "2026-04-27T00:00:00+12:00",
        "global_user_id": "user-1",
        "user_name": "User",
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "platform_message_id": "msg-1",
        "character_profile": {"name": "Kazusa"},
        "metadata": {},
        "mood": "neutral",
        "global_vibe": "",
        "reflection_summary": "",
        "subjective_appraisals": ["User sounded happy"],
        "interaction_subtext": "test",
        "last_relationship_insight": "friendly",
        "new_facts": [{
            "description": "User likes tea",
            "category": "preference",
            "dedup_key": "likes_tea",
        }],
        "future_promises": [],
        "user_profile": {"affinity": 500},
        "affinity_delta": 1,
        "decontexualized_input": "remember tea",
    }


def _patch_writers(monkeypatch, *, character_image=None) -> MagicMock:
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=1)
    monkeypatch.setattr(persistence_module, "get_rag_cache2_runtime", MagicMock(return_value=runtime))
    monkeypatch.setattr(persistence_module, "upsert_character_state", AsyncMock())
    monkeypatch.setattr(persistence_module, "update_last_relationship_insight", AsyncMock())
    monkeypatch.setattr(persistence_module, "update_affinity", AsyncMock())
    monkeypatch.setattr(persistence_module, "upsert_character_self_image", AsyncMock())
    monkeypatch.setattr(persistence_module, "_update_character_image", AsyncMock(return_value=character_image))
    monkeypatch.setattr(persistence_module, "update_user_memory_units_from_state", AsyncMock(return_value=[]))
    monkeypatch.setattr(persistence_module, "_task_dispatcher", None)
    monkeypatch.setattr(persistence_module, "_task_registry", None)
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
