"""Tests for the native V2 consolidation writer boundary."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.consolidation import persistence as persistence_module
from kazusa_ai_chatbot.consolidation.origin import (
    ConsolidationOriginMetadata,
)
from kazusa_ai_chatbot.consolidation.target import (
    build_consolidation_target_plan,
)
from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.time_boundary import build_turn_clock


def _origin(
    *,
    trigger_source: str = "user_message",
    input_sources: list[str] | None = None,
    output_mode: str = "visible_reply",
) -> ConsolidationOriginMetadata:
    """Build identifier-only origin metadata for one writer case."""

    if input_sources is None:
        input_sources = ["dialog_text"]
    turn_clock = build_turn_clock("2026-04-27 00:00:00")
    return {
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


def _state(
    *,
    origin: ConsolidationOriginMetadata | None = None,
    enabled_lanes: list[str] | None = None,
) -> dict[str, Any]:
    """Build a writer state with native embedded user cognition state."""

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
        "interaction_subtext": "test",
        "emotional_appraisal": "",
        "character_intent": "PROVIDE",
        "logical_stance": "CONFIRM",
        "internal_monologue": "",
        "final_dialog": ["I will remember that."],
        "text_surface_output_v2": {
            "schema_version": "text_surface_output.v2",
            "content_plan": "acknowledge",
            "visible_boundaries": [],
            "addressee_plan": ["current user"],
            "style_guidance": "brief",
            "pacing_guidance": "direct",
            "selected_surface_intent": "acknowledge",
        },
        "new_facts": [
            {
                "description": "User likes tea",
                "category": "preference",
                "dedup_key": "likes_tea",
            }
        ],
        "future_promises": [],
        "user_profile": {
            "global_user_id": "user-1",
            "cognition_state": build_acquaintance_user_state(
                global_user_id="user-1",
                updated_at="2026-04-27T00:00:00Z",
            ),
        },
        "decontexualized_input": "remember tea",
        "rag_result": {
            "user_image": {
                "user_memory_context": {
                    "stable_patterns": [],
                    "recent_shifts": [],
                    "objective_facts": [],
                    "milestones": [],
                    "active_commitments": [],
                }
            }
        },
        "consolidation_origin": origin,
    }
    if enabled_lanes is not None:
        state["enabled_consolidation_write_lanes"] = enabled_lanes
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)
    return state


def _patch_writer_dependencies(monkeypatch) -> dict[str, Any]:
    """Patch only native writer dependencies."""

    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=1)
    mocks = {
        "runtime": runtime,
        "get_runtime": MagicMock(return_value=runtime),
        "memory": AsyncMock(return_value=[{"kind": "fact", "id": "memory-1"}]),
        "guidance": AsyncMock(return_value={"memory_id": "guidance-1"}),
        "group_style": AsyncMock(),
    }
    monkeypatch.setattr(
        persistence_module,
        "get_rag_cache2_runtime",
        mocks["get_runtime"],
    )
    monkeypatch.setattr(
        persistence_module,
        "update_user_memory_units_from_state",
        mocks["memory"],
    )
    monkeypatch.setattr(
        persistence_module,
        "persist_character_self_guidance_from_state",
        mocks["guidance"],
    )
    monkeypatch.setattr(
        persistence_module,
        "persist_group_channel_style_image",
        mocks["group_style"],
    )
    return mocks


@pytest.mark.asyncio
async def test_db_writer_requires_attached_target_plan() -> None:
    """Persistence must receive the explicit target plan from core."""

    state = _state()
    del state["consolidation_target_plan"]
    with pytest.raises(KeyError, match="consolidation_target_plan"):
        await persistence_module.db_writer(state)


@pytest.mark.asyncio
async def test_denied_origin_has_no_durable_effects(monkeypatch) -> None:
    """Denied origins fail closed before native persistence effects."""

    mocks = _patch_writer_dependencies(monkeypatch)
    state = _state(
        origin=_origin(
            trigger_source="reflection_signal",
            input_sources=["reflection_artifact"],
            output_mode="think_only",
        ),
    )
    result = await persistence_module.db_writer(state)

    assert all(
        value is False for value in result["metadata"]["write_success"].values()
    )
    assert result["metadata"]["cache_invalidated"] == []
    mocks["memory"].assert_not_awaited()
    mocks["guidance"].assert_not_awaited()
    mocks["runtime"].invalidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_router_allowlist_has_no_durable_effects(
    monkeypatch,
) -> None:
    """Writer persistence fails closed without accepted router lanes."""

    mocks = _patch_writer_dependencies(monkeypatch)
    result = await persistence_module.db_writer(_state())

    assert all(
        value is False for value in result["metadata"]["write_success"].values()
    )
    assert result["metadata"]["cache_invalidated"] == []
    mocks["memory"].assert_not_awaited()
    mocks["guidance"].assert_not_awaited()
    mocks["group_style"].assert_not_awaited()
    mocks["runtime"].invalidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_memory_write_uses_native_lane_and_cache_invalidation(
    monkeypatch,
) -> None:
    """User facts persist through the V2 user-memory lane only."""

    mocks = _patch_writer_dependencies(monkeypatch)
    result = await persistence_module.db_writer(
        _state(enabled_lanes=["user_memory_units", "active_commitment"]),
    )

    mocks["memory"].assert_awaited_once()
    assert result["metadata"]["write_success"]["user_memory_units"] is True
    assert result["metadata"]["cache_invalidated"] == ["user_profile"]
    assert all(
        field not in result["metadata"]
        for field in (
            "relationship_state_before",
            "relationship_delta_processed",
            "relationship_insight",
            "character_state",
        )
    )


@pytest.mark.asyncio
async def test_character_guidance_is_the_only_character_consolidation_lane(
    monkeypatch,
) -> None:
    """Character-owned durable guidance is separate from mutable affect."""

    mocks = _patch_writer_dependencies(monkeypatch)
    state = _state(enabled_lanes=["character_self_guidance"])
    state["character_self_guidance"] = {
        "memory_type": "defense_rule",
        "content": "Keep accepted boundaries explicit.",
    }
    result = await persistence_module.db_writer(state)

    mocks["guidance"].assert_awaited_once()
    assert result["metadata"]["write_success"]["character_self_guidance"] is True
    assert result["metadata"]["cache_invalidated"] == ["character_self_guidance"]


@pytest.mark.asyncio
async def test_group_style_write_has_no_user_relationship_side_effect(
    monkeypatch,
) -> None:
    """Group style persistence remains independently scoped."""

    mocks = _patch_writer_dependencies(monkeypatch)
    state = _state(
        origin=_origin(
            trigger_source="internal_thought",
            input_sources=["internal_monologue"],
            output_mode="preview",
        ),
    )
    state["global_user_id"] = "self_cognition"
    state["user_profile"] = {"display_name": "group audience"}
    state["group_channel_style_image"] = {
        "overlay": {"speech_guidelines": ["Keep replies compact."]},
        "source_reflection_run_ids": ["group-review-1"],
    }
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)
    state["enabled_consolidation_write_lanes"] = ["interaction_style_image"]

    result = await persistence_module.db_writer(state)

    mocks["group_style"].assert_awaited_once()
    assert result["metadata"]["write_success"]["group_channel_style_image"] is True
    assert result["metadata"]["cache_invalidated"] == []


@pytest.mark.asyncio
async def test_native_writer_does_not_rehydrate_legacy_affect(monkeypatch) -> None:
    """Legacy prose-affect keys cannot become a writer authority."""

    mocks = _patch_writer_dependencies(monkeypatch)
    state = _state(enabled_lanes=["user_memory_units"])
    state.update(
        {
            "mood": "legacy mood",
            "vibe_check": "legacy vibe",
            "character_reflection": "legacy summary",
            "relationship_delta": 9,
            "semantic_relationship_projection": "legacy insight",
        }
    )

    result = await persistence_module.db_writer(state)

    mocks["memory"].assert_awaited_once()
    assert "legacy mood" not in str(result)
    assert "legacy insight" not in str(result)
    assert "relationship_state" not in str(result).lower()
    assert result["metadata"]["write_success"]["user_memory_units"] is True
    assert result["metadata"]["write_success"]["group_channel_style_image"] is False
