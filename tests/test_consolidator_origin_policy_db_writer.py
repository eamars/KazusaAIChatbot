"""Tests for consolidator db_writer origin-policy enforcement."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_images as image_module,
    persona_supervisor2_consolidator_persistence as persistence_module,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_origin import (
    ConsolidationOriginMetadata,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock


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

    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    origin: ConsolidationOriginMetadata = {
        "episode_id": "episode-1",
        "trigger_source": trigger_source,
        "input_sources": input_sources,
        "output_mode": output_mode,
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
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

    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    state: dict[str, Any] = {
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
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
        "update_affinity": AsyncMock(),
        "update_character_image": AsyncMock(return_value=character_image),
        "get_character_runtime_state": AsyncMock(return_value={}),
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
        "get_character_runtime_state",
        mocks["get_character_runtime_state"],
        raising=False,
    )
    monkeypatch.setattr(
        persistence_module,
        "upsert_character_self_image",
        mocks["upsert_character_self_image"],
    )
    return mocks


def _image_response(summary: str) -> SimpleNamespace:
    """Build a fake LangChain response with JSON content."""

    response = SimpleNamespace(
        content=f'{{"session_summary": "{summary}"}}',
    )
    return response


def _image_llm(summary: str) -> SimpleNamespace:
    """Build a fake character-image LLM object."""

    llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=_image_response(summary)),
    )
    return llm


def _patch_writer_dependencies_except_image(
    monkeypatch,
    *,
    runtime_state: dict[str, Any] | None = None,
    runtime_state_error: Exception | None = None,
) -> dict[str, Any]:
    """Patch db_writer dependencies while keeping the real image updater.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        runtime_state: Runtime character state returned by the DB reader.
        runtime_state_error: Optional exception raised by the DB reader.

    Returns:
        Named mocks and fake runtime objects for assertions.
    """

    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=1)
    if runtime_state_error is None:
        get_character_runtime_state = AsyncMock(
            return_value=runtime_state or {},
        )
    else:
        get_character_runtime_state = AsyncMock(
            side_effect=runtime_state_error,
        )

    mocks: dict[str, Any] = {
        "runtime": runtime,
        "get_runtime": MagicMock(return_value=runtime),
        "upsert_character_state": AsyncMock(),
        "update_last_relationship_insight": AsyncMock(),
        "update_user_memory_units_from_state": AsyncMock(return_value=[]),
        "update_affinity": AsyncMock(),
        "get_character_runtime_state": get_character_runtime_state,
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
        "update_affinity",
        mocks["update_affinity"],
    )
    monkeypatch.setattr(
        persistence_module,
        "get_character_runtime_state",
        mocks["get_character_runtime_state"],
        raising=False,
    )
    monkeypatch.setattr(
        persistence_module,
        "upsert_character_self_image",
        mocks["upsert_character_self_image"],
    )
    monkeypatch.setattr(
        image_module,
        "_character_image_session_summary_llm",
        _image_llm("new self-image"),
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
    assert "scheduled_event_ids" not in result["metadata"]
    assert all("dispatch" not in key for key in result["metadata"])
    denied_async_effect.assert_not_called()
    denied_sync_effect.assert_not_called()


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
async def test_db_writer_user_message_origin_does_not_emit_dispatch_metadata(
    monkeypatch,
) -> None:
    """Allowed user-message origins no longer schedule user-visible text."""
    mocks = _patch_allowed_write_dependencies(monkeypatch)

    result = await persistence_module.db_writer(_state())

    assert "scheduled_event_ids" not in result["metadata"]
    assert all("dispatch" not in key for key in result["metadata"])
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

    sources = [
        call.args[0].source
        for call in mocks["runtime"].invalidate.await_args_list
    ]
    assert sources == ["user_profile", "character_state"]
    assert result["metadata"]["cache_invalidated"] == [
        "user_profile",
        "character_state",
    ]
    assert result["metadata"]["cache_evicted_count"] == 2


@pytest.mark.asyncio
async def test_db_writer_internal_thought_uses_db_current_self_image(
    monkeypatch,
) -> None:
    """Internal-thought image writes must merge from DB-current image state."""

    existing_image = {
        "milestones": [{"summary": "stable self image"}],
        "recent_window": [
            {
                "timestamp": "2026-05-18T00:00:00+00:00",
                "summary": "previous self-image",
            }
        ],
        "historical_summary": "older self-image history",
        "meta": {"synthesis_count": 4},
    }
    mocks = _patch_writer_dependencies_except_image(
        monkeypatch,
        runtime_state={"self_image": existing_image},
    )
    state = _state(
        origin=_origin(
            trigger_source="internal_thought",
            input_sources=["internal_monologue"],
            output_mode="preview",
        )
    )

    result = await persistence_module.db_writer(state)

    expected_image = {
        "milestones": [{"summary": "stable self image"}],
        "recent_window": [
            {
                "timestamp": "2026-05-18T00:00:00+00:00",
                "summary": "previous self-image",
            },
            {
                "timestamp": state["storage_timestamp_utc"],
                "summary": "new self-image",
            },
        ],
        "historical_summary": "older self-image history",
        "meta": {
            "synthesis_count": 5,
            "last_updated": state["storage_timestamp_utc"],
        },
    }
    mocks["get_character_runtime_state"].assert_awaited_once()
    mocks["upsert_character_self_image"].assert_awaited_once_with(
        expected_image,
    )
    assert result["metadata"]["write_success"]["character_image"] is True


@pytest.mark.asyncio
async def test_db_writer_runtime_state_failure_fails_character_image_closed(
    monkeypatch,
) -> None:
    """A DB-current image read failure must not abort db_writer."""

    mocks = _patch_writer_dependencies_except_image(
        monkeypatch,
        runtime_state_error=DatabaseOperationError("runtime state unavailable"),
    )
    state = _state(
        origin=_origin(
            trigger_source="internal_thought",
            input_sources=["internal_monologue"],
            output_mode="preview",
        )
    )

    result = await persistence_module.db_writer(state)

    mocks["get_character_runtime_state"].assert_awaited_once()
    mocks["upsert_character_self_image"].assert_not_awaited()
    assert result["metadata"]["write_success"]["character_image"] is False


@pytest.mark.asyncio
async def test_db_writer_empty_reflection_skips_self_image_db_read(
    monkeypatch,
) -> None:
    """No image synthesis means no DB-current self-image read is needed."""

    mocks = _patch_writer_dependencies_except_image(
        monkeypatch,
        runtime_state_error=DatabaseOperationError("must not read self image"),
    )
    state = _state(
        origin=_origin(
            trigger_source="internal_thought",
            input_sources=["internal_monologue"],
            output_mode="preview",
        )
    )
    state["reflection_summary"] = ""

    result = await persistence_module.db_writer(state)

    mocks["get_character_runtime_state"].assert_not_awaited()
    mocks["upsert_character_self_image"].assert_not_awaited()
    assert "character_image" not in result["metadata"]["write_success"]
