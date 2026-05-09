"""Tests for consolidator db_writer origin-policy enforcement."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.dispatcher import DispatchResult, RawToolCall
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_persistence as persistence_module,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_origin import (
    ConsolidationOriginMetadata,
)
from kazusa_ai_chatbot.time_context import build_character_time_context


def _origin(
    *,
    trigger_source: str = "user_message",
    input_sources: list[str] | None = None,
    output_mode: str = "visible_reply",
) -> ConsolidationOriginMetadata:
    """Build origin metadata for direct db_writer calls.

    Args:
        trigger_source: Trigger source label to project into the origin.
        input_sources: Input source labels to project into the origin.
        output_mode: Output mode to project into the origin.

    Returns:
        Consolidation origin metadata with stable identifier fields.
    """
    if input_sources is None:
        input_sources = ["dialog_text"]

    origin: ConsolidationOriginMetadata = {
        "episode_id": "episode-1",
        "trigger_source": trigger_source,
        "input_sources": input_sources,
        "output_mode": output_mode,
        "timestamp": "2026-04-27T00:00:00+12:00",
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_message_id": "msg-1",
        "active_turn_platform_message_ids": ["msg-1"],
        "active_turn_conversation_row_ids": ["conversation-row-1"],
        "current_platform_user_id": "platform-user-1",
        "current_global_user_id": "user-1",
        "current_display_name": "User",
    }
    return origin


def _state(*, origin: ConsolidationOriginMetadata | None = None) -> dict[str, Any]:
    """Build a db_writer state that would exercise every write path.

    Args:
        origin: Consolidation origin metadata for the writer policy.

    Returns:
        Direct consolidator state for db_writer tests.
    """
    if origin is None:
        origin = _origin()

    state: dict[str, Any] = {
        "timestamp": "2026-04-27T00:00:00+12:00",
        "time_context": build_character_time_context("2026-04-27T00:00:00+12:00"),
        "global_user_id": "user-1",
        "user_name": "User",
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_message_id": "msg-1",
        "character_profile": {"name": "Kazusa"},
        "metadata": {},
        "mood": "neutral",
        "global_vibe": "quiet",
        "reflection_summary": "summary",
        "subjective_appraisals": ["User sounded happy"],
        "interaction_subtext": "test",
        "last_relationship_insight": "friendly",
        "new_facts": [
            {
                "description": "User likes tea",
                "category": "preference",
                "dedup_key": "likes_tea",
            }
        ],
        "future_promises": [
            {
                "target": "user",
                "action": "send a reminder",
                "due_time": "2026-04-27 00:30",
                "commitment_type": "future_promise",
            }
        ],
        "user_profile": {"affinity": 500},
        "affinity_delta": 1,
        "decontexualized_input": "remember tea",
        "final_dialog": ["I will remind you later."],
        "action_directives": {"linguistic_directives": {"content_anchors": []}},
        "consolidation_origin": origin,
    }
    return state


def _patch_allowed_write_dependencies(
    monkeypatch,
    *,
    memory_results: list[dict[str, Any]] | None = None,
    character_image: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Patch db_writer dependencies for allowed-origin regression tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        memory_results: User-memory unit write results to return.
        character_image: Character image result to return.

    Returns:
        Named mocks and fake runtime objects for assertions.
    """
    if memory_results is None:
        memory_results = []

    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=1)
    mocks: dict[str, Any] = {
        "runtime": runtime,
        "get_runtime": MagicMock(return_value=runtime),
        "upsert_character_state": AsyncMock(),
        "update_last_relationship_insight": AsyncMock(),
        "update_user_memory_units_from_state": AsyncMock(
            return_value=memory_results
        ),
        "get_dispatcher": MagicMock(return_value=None),
        "update_affinity": AsyncMock(),
        "update_character_image": AsyncMock(return_value=character_image),
        "upsert_character_self_image": AsyncMock(),
    }
    monkeypatch.setattr(
        persistence_module,
        "get_rag_cache2_runtime",
        mocks["get_runtime"],
    )
    monkeypatch.setattr(
        persistence_module,
        "upsert_character_state",
        mocks["upsert_character_state"],
    )
    monkeypatch.setattr(
        persistence_module,
        "update_last_relationship_insight",
        mocks["update_last_relationship_insight"],
    )
    monkeypatch.setattr(
        persistence_module,
        "update_user_memory_units_from_state",
        mocks["update_user_memory_units_from_state"],
    )
    monkeypatch.setattr(
        persistence_module,
        "_get_task_dispatcher",
        mocks["get_dispatcher"],
    )
    monkeypatch.setattr(
        persistence_module,
        "update_affinity",
        mocks["update_affinity"],
    )
    monkeypatch.setattr(
        persistence_module,
        "_update_character_image",
        mocks["update_character_image"],
    )
    monkeypatch.setattr(
        persistence_module,
        "upsert_character_self_image",
        mocks["upsert_character_self_image"],
    )
    return mocks


@pytest.mark.asyncio
async def test_db_writer_denied_origin_skips_all_durable_write_effects(
    monkeypatch,
) -> None:
    """Denied origins must not reach persistence, scheduler, image, or cache."""
    failure = AssertionError("denied origin must not call durable write effects")
    denied_async_effect = AsyncMock(side_effect=failure)
    denied_sync_effect = MagicMock(side_effect=failure)
    monkeypatch.setattr(
        persistence_module,
        "upsert_character_state",
        denied_async_effect,
    )
    monkeypatch.setattr(
        persistence_module,
        "update_last_relationship_insight",
        denied_async_effect,
    )
    monkeypatch.setattr(
        persistence_module,
        "update_user_memory_units_from_state",
        denied_async_effect,
    )
    monkeypatch.setattr(
        persistence_module,
        "_get_task_dispatcher",
        denied_sync_effect,
    )
    monkeypatch.setattr(
        persistence_module,
        "_build_dispatch_context",
        denied_sync_effect,
    )
    monkeypatch.setattr(
        persistence_module,
        "_generate_raw_tool_calls",
        denied_async_effect,
    )
    monkeypatch.setattr(persistence_module, "update_affinity", denied_async_effect)
    monkeypatch.setattr(
        persistence_module,
        "_update_character_image",
        denied_sync_effect,
    )
    monkeypatch.setattr(
        persistence_module,
        "upsert_character_self_image",
        denied_async_effect,
    )
    monkeypatch.setattr(
        persistence_module,
        "get_rag_cache2_runtime",
        denied_sync_effect,
    )

    result = await persistence_module.db_writer(
        _state(
            origin=_origin(
                trigger_source="reflection_signal",
                input_sources=["reflection_artifact"],
                output_mode="think_only",
            )
        )
    )

    assert result["metadata"]["write_success"] == {
        "character_state": False,
        "relationship_insight": False,
        "user_memory_units": False,
        "affinity": False,
        "character_image": False,
    }
    assert result["metadata"]["cache_invalidated"] == []
    assert result["metadata"]["cache_evicted_count"] == 0
    assert result["metadata"]["scheduled_event_ids"] == []
    assert result["metadata"]["task_dispatch_rejected"] == []


@pytest.mark.asyncio
async def test_db_writer_user_message_origin_preserves_character_and_user_writes(
    monkeypatch,
) -> None:
    """Allowed user-message origins still run existing durable write paths."""
    mocks = _patch_allowed_write_dependencies(
        monkeypatch,
        memory_results=[{"kind": "fact", "id": "memory-1"}],
        character_image={"recent_window": []},
    )

    result = await persistence_module.db_writer(_state())

    mocks["upsert_character_state"].assert_awaited_once()
    mocks["update_last_relationship_insight"].assert_awaited_once_with(
        "user-1",
        "friendly",
    )
    mocks["update_user_memory_units_from_state"].assert_awaited_once()
    mocks["update_affinity"].assert_awaited_once()
    mocks["update_character_image"].assert_awaited_once()
    mocks["upsert_character_self_image"].assert_awaited_once_with(
        {"recent_window": []}
    )
    assert result["metadata"]["write_success"] == {
        "character_state": True,
        "relationship_insight": True,
        "user_memory_units": True,
        "affinity": True,
        "character_image": True,
    }


@pytest.mark.asyncio
async def test_db_writer_user_message_origin_preserves_scheduler_dispatch(
    monkeypatch,
) -> None:
    """Allowed user-message origins still run scheduler dispatch."""
    mocks = _patch_allowed_write_dependencies(monkeypatch)
    raw_calls = [RawToolCall(tool="send_message", args={"text": "later"})]
    generate_raw_tool_calls = AsyncMock(return_value=raw_calls)
    fake_dispatcher = MagicMock()
    fake_dispatcher.dispatch = AsyncMock(
        return_value=DispatchResult(
            scheduled=[(object(), "event-1")],
            rejected=[],
        )
    )
    get_dispatcher = MagicMock(return_value=fake_dispatcher)
    monkeypatch.setattr(
        persistence_module,
        "_generate_raw_tool_calls",
        generate_raw_tool_calls,
    )
    monkeypatch.setattr(persistence_module, "_get_task_dispatcher", get_dispatcher)

    result = await persistence_module.db_writer(_state())

    get_dispatcher.assert_called_once_with()
    generate_raw_tool_calls.assert_awaited_once()
    fake_dispatcher.dispatch.assert_awaited_once()
    assert fake_dispatcher.dispatch.await_args.args[0] == raw_calls
    assert result["metadata"]["scheduled_event_ids"] == ["event-1"]
    assert result["metadata"]["task_dispatch_rejected"] == []
    mocks["update_user_memory_units_from_state"].assert_awaited_once()


@pytest.mark.asyncio
async def test_db_writer_user_message_origin_preserves_cache_invalidation(
    monkeypatch,
) -> None:
    """Allowed user-message origins still emit current Cache2 invalidations."""
    mocks = _patch_allowed_write_dependencies(
        monkeypatch,
        memory_results=[{"kind": "fact", "id": "memory-1"}],
        character_image={"recent_window": []},
    )

    result = await persistence_module.db_writer(_state())

    sources = [call.args[0].source for call in mocks["runtime"].invalidate.await_args_list]
    assert sources == ["user_profile", "character_state"]
    assert result["metadata"]["cache_invalidated"] == [
        "user_profile",
        "character_state",
    ]
    assert result["metadata"]["cache_evicted_count"] == 2
