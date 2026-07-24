"""Integration tests for the canonical consolidation lane boundary."""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.consolidation import core as core_module
from kazusa_ai_chatbot.consolidation import lane_router
from kazusa_ai_chatbot.consolidation.target import (
    build_consolidation_target_plan,
    validate_write_intent,
)


def _base_state() -> dict[str, Any]:
    """Build a valid private-chat state with embedded V2 user state."""

    state: dict[str, Any] = {
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_profile": {
            "global_user_id": "global-user-1",
            "cognition_state": build_acquaintance_user_state(
                global_user_id="global-user-1",
                updated_at="2026-07-03T00:00:00Z",
            ),
        },
        "platform": "qq",
        "platform_channel_id": "private-1",
        "channel_type": "private",
        "character_profile": {"name": "Kazusa"},
        "cognitive_episode": {
            "episode_id": "episode-bigbang-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "target_scope": {
                "platform": "qq",
                "platform_channel_id": "private-1",
                "channel_type": "private",
                "current_global_user_id": "global-user-1",
                "current_display_name": "Test User",
                "target_broadcast": False,
            },
        },
        "consolidation_origin": {
            "episode_id": "episode-bigbang-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
            "platform": "qq",
            "platform_channel_id": "private-1",
            "channel_type": "private",
            "platform_message_id": "message-bigbang-1",
            "active_turn_platform_message_ids": ["message-bigbang-1"],
            "active_turn_conversation_row_ids": ["conversation-row-1"],
            "current_platform_user_id": "platform-user-1",
            "current_global_user_id": "global-user-1",
            "current_display_name": "Test User",
        },
        "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
        "local_time_context": {"current_date": "2026-07-03"},
        "decontextualized_input": "I now work in Auckland.",
        "final_dialog": ["Kazusa acknowledges the update."],
        "internal_monologue": "",
        "chat_history_recent": [],
        "metadata": {},
        "rag_result": {
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "recall_evidence": [],
            "user_image": {
                "user_memory_context": {
                    "stable_patterns": [],
                    "recent_shifts": [],
                    "objective_facts": [],
                    "milestones": [],
                    "active_commitments": [],
                }
            },
            "user_memory_unit_candidates": [],
        },
    }
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)
    return state


def _targets_by_alias(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index target-plan rows by alias."""

    return {target["target_alias"]: target for target in plan["targets"]}


def test_core_uses_lane_pipeline_and_not_the_retired_harvester() -> None:
    """The active core must route through the lane pipeline."""

    source = inspect.getsource(core_module)

    assert "facts_harvester" not in source
    assert "fact_harvester_evaluator" not in source
    assert "run_consolidation_lane_pipeline" in source


def test_target_plan_exposes_only_native_v2_write_lanes() -> None:
    """Target planning must exclude retired affect and relationship lanes."""

    targets = _targets_by_alias(build_consolidation_target_plan(_base_state()))

    assert "character_self_guidance" in targets["character"]["write_lanes"]
    assert "user_memory_units" in targets["current_user"]["write_lanes"]
    all_lanes = {
        lane
        for target in targets.values()
        for lane in target["write_lanes"]
    }
    assert not all_lanes & {
        "character_state",
        "relationship_insight",
        "relationship_state",
    }


def test_user_target_cannot_receive_character_guidance() -> None:
    """Character-owned guidance cannot be redirected into user memory."""

    plan = build_consolidation_target_plan(_base_state())

    with pytest.raises(ValueError):
        validate_write_intent(
            {
                "target_alias": "current_user",
                "write_lane": "character_self_guidance",
                "payload": {},
            },
            plan,
        )


def test_lane_roster_excludes_retired_relationship_and_affect_lanes() -> None:
    """The router must not be offered lanes removed by the V2 cutover."""

    roster = lane_router.build_lane_roster(
        build_consolidation_target_plan(_base_state())
    )
    roster_lanes = {row["lane"] for row in roster}

    assert roster_lanes == {
        "user_memory_units",
        "active_commitment",
        "character_self_guidance",
    }


def test_router_rejects_retired_lane_output() -> None:
    """A stale model response must fail closed at the router contract."""

    roster = lane_router.build_lane_roster(
        build_consolidation_target_plan(_base_state())
    )

    with pytest.raises(ValueError, match="unknown consolidation lane"):
        lane_router.validate_lane_router_output(
            {
                "lane_tasks": [
                    {
                        "lane": "relationship_profile",
                        "reason": "legacy relationship update",
                        "source_keys": ["current_turn_user_message"],
                    }
                ]
            },
            roster,
        )


@pytest.mark.asyncio
async def test_native_user_memory_route_builds_auditable_write_intent(
    monkeypatch,
) -> None:
    """A native user-memory route carries source refs into dry-run output."""

    state = _base_state()

    async def _fake_router(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "lane_tasks": [
                {
                    "lane": "user_memory_units",
                    "reason": "durable user preference",
                    "source_keys": ["current_turn_user_message"],
                }
            ]
        }

    monkeypatch.setattr(lane_router, "call_lane_router_llm", _fake_router)
    packet = await lane_router.run_consolidation_lane_pipeline(
        state,
        dry_run=True,
    )

    assert packet["accepted_lanes"] == ["user_memory_units"]
    assert packet["write_intents"][0]["target_alias"] == "current_user"
    assert packet["write_intents"][0]["write_lane"] == "user_memory_units"
    assert packet["write_intents"][0]["payload"]["source_refs"]


@pytest.mark.asyncio
async def test_native_guidance_route_runs_only_native_specialist(
    monkeypatch,
) -> None:
    """Character guidance uses the V2 specialist and writer boundary."""

    state = _base_state()
    captured: dict[str, Any] = {}

    async def _fake_router(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "lane_tasks": [
                {
                    "lane": "character_self_guidance",
                    "reason": "accepted character behavior guidance",
                    "source_keys": ["current_turn_user_message"],
                }
            ]
        }

    async def _fake_specialist(node_state: dict[str, Any]) -> dict[str, Any]:
        captured["guidance_refs"] = node_state[
            "character_self_guidance_source_refs"
        ]
        return {
            "character_self_guidance": {
                "memory_type": "defense_rule",
                "content": "Keep accepted boundaries explicit.",
            }
        }

    async def _fake_writer(node_state: dict[str, Any]) -> dict[str, Any]:
        captured["writer_state"] = node_state
        return {"metadata": {"write_success": {"character_self_guidance": True}}}

    monkeypatch.setattr(lane_router, "call_lane_router_llm", _fake_router)
    monkeypatch.setattr(
        lane_router,
        "character_self_guidance_specialist",
        _fake_specialist,
    )
    monkeypatch.setattr(lane_router, "db_writer", _fake_writer)

    packet = await lane_router.run_consolidation_lane_pipeline(state)

    assert packet["accepted_lanes"] == ["character_self_guidance"]
    assert captured["guidance_refs"]
    assert captured["writer_state"]["character_self_guidance"]["memory_type"] == (
        "defense_rule"
    )


@pytest.mark.asyncio
async def test_guidance_reviewer_rejection_disables_persistence(
    monkeypatch,
) -> None:
    """A rejected specialist candidate must leave no enabled write lane."""

    state = _base_state()

    async def _fake_router(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "lane_tasks": [{
                "lane": "character_self_guidance",
                "reason": "candidate character behavior guidance",
                "source_keys": ["current_turn_user_message"],
            }]
        }

    specialist = AsyncMock(return_value={"character_self_guidance": {}})
    writer = AsyncMock()
    monkeypatch.setattr(lane_router, "call_lane_router_llm", _fake_router)
    monkeypatch.setattr(
        lane_router,
        "character_self_guidance_specialist",
        specialist,
    )
    monkeypatch.setattr(lane_router, "db_writer", writer)

    packet = await lane_router.run_consolidation_lane_pipeline(state)

    assert packet["accepted_lanes"] == []
    assert packet["state"]["enabled_consolidation_write_lanes"] == []
    assert packet["state"]["metadata"]["review_rejected_lanes"] == [
        "character_self_guidance"
    ]
    specialist.assert_awaited_once()
    writer.assert_not_awaited()
