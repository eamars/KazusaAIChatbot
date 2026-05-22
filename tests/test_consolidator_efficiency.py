"""Tests for R2-compliant consolidator efficiency behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.consolidation import core as consolidator_module
from kazusa_ai_chatbot.consolidation.target import (
    build_consolidation_target_plan,
)
from kazusa_ai_chatbot.consolidation import persistence as persistence_module
from kazusa_ai_chatbot.consolidation.origin import (
    build_user_message_consolidation_origin,
)
from kazusa_ai_chatbot.time_boundary import local_time_context_from_storage_utc

STORAGE_TIMESTAMP_UTC = "2026-04-26T12:00:00+00:00"


def _cognitive_episode() -> dict:
    """Build a valid text-chat episode for direct consolidator calls.

    Returns:
        Valid user-message cognitive episode.
    """
    episode = build_text_chat_cognitive_episode(
        episode_id="episode-1",
        percept_id="percept-1",
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=local_time_context_from_storage_utc(
            STORAGE_TIMESTAMP_UTC,
        ),
        user_input="hello",
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
    return episode


def _consolidation_origin() -> dict:
    """Build valid user-message origin metadata for direct db_writer calls.

    Returns:
        Valid user-message consolidation origin metadata.
    """
    origin = build_user_message_consolidation_origin(
        episode=_cognitive_episode(),
    )
    return origin


def _global_state() -> dict:
    return {
        "storage_timestamp_utc": STORAGE_TIMESTAMP_UTC,
        "local_time_context": local_time_context_from_storage_utc(
            STORAGE_TIMESTAMP_UTC,
        ),
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {
            "global_user_id": "user-1",
            "affinity": 500,
        },
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_message_id": "msg-1",
        "action_directives": {"linguistic_directives": {"content_anchors": []}},
        "internal_monologue": "test",
        "final_dialog": ["ok"],
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "PROVIDE",
        "logical_stance": "CONFIRM",
        "character_profile": {"name": "Kazusa"},
        "rag_result": {
            "user_image": {
                "user_memory_context": {
                    "stable_patterns": [],
                    "recent_shifts": [],
                    "objective_facts": [
                        {
                            "fact": "User likes tea",
                            "subjective_appraisal": "Kazusa treats this as a preference.",
                            "relationship_signal": "Offer tea-related continuity.",
                        }
                    ],
                    "active_commitments": [
                        {
                            "fact": "Kazusa will reply in English",
                            "subjective_appraisal": "Kazusa treats this as active preference context.",
                            "relationship_signal": "Use English in future replies when relevant.",
                        }
                    ],
                    "milestones": [
                        {
                            "fact": "Met Kazusa",
                            "subjective_appraisal": "Kazusa treats this as relationship history.",
                            "relationship_signal": "Keep light continuity.",
                        }
                    ],
                }
            }
        },
        "decontexualized_input": "hello",
        "cognitive_episode": _cognitive_episode(),
    }


def test_build_existing_dedup_keys_ignores_memory_unit_context() -> None:
    keys = consolidator_module._build_existing_dedup_keys(_global_state())

    assert keys == set()


@pytest.mark.asyncio
async def test_empty_harvester_output_skips_evaluator(monkeypatch) -> None:
    calls = {"evaluator": 0}

    async def _global_state_updater(_state):
        return {"mood": "", "global_vibe": "", "reflection_summary": ""}

    async def _relationship_recorder(_state):
        return {"subjective_appraisals": [], "affinity_delta": 0, "last_relationship_insight": ""}

    async def _facts_harvester(state):
        assert state["existing_dedup_keys"] == set()
        return {"new_facts": [], "future_promises": []}

    async def _fact_harvester_evaluator(_state):
        calls["evaluator"] += 1
        return {"should_stop": True}

    async def _db_writer(_state):
        return {"metadata": {"write_success": {}}}

    monkeypatch.setattr(consolidator_module, "global_state_updater", _global_state_updater)
    monkeypatch.setattr(consolidator_module, "relationship_recorder", _relationship_recorder)
    monkeypatch.setattr(consolidator_module, "facts_harvester", _facts_harvester)
    monkeypatch.setattr(consolidator_module, "fact_harvester_evaluator", _fact_harvester_evaluator)
    monkeypatch.setattr(consolidator_module, "db_writer", _db_writer)

    result = await consolidator_module.call_consolidation_subgraph(_global_state())

    assert calls["evaluator"] == 0
    assert result["new_facts"] == []
    assert result["future_promises"] == []


@pytest.mark.asyncio
async def test_db_writer_runs_image_updaters_through_gather(monkeypatch) -> None:
    gather_calls = []

    async def _fake_gather(*aws, return_exceptions=False):
        gather_calls.append((aws, return_exceptions))
        results = []
        for awaitable in aws:
            results.append(await awaitable)
        return results

    monkeypatch.setattr(persistence_module.asyncio, "gather", _fake_gather)
    monkeypatch.setattr(persistence_module, "get_rag_cache2_runtime", MagicMock(return_value=MagicMock(invalidate=AsyncMock(return_value=0))))
    monkeypatch.setattr(persistence_module, "upsert_character_state", AsyncMock())
    monkeypatch.setattr(persistence_module, "update_last_relationship_insight", AsyncMock())
    monkeypatch.setattr(persistence_module, "update_affinity", AsyncMock())
    monkeypatch.setattr(
        persistence_module,
        "get_character_runtime_state",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(persistence_module, "upsert_character_self_image", AsyncMock())
    monkeypatch.setattr(persistence_module, "_update_character_image", AsyncMock(return_value=None))
    monkeypatch.setattr(persistence_module, "update_user_memory_units_from_state", AsyncMock(return_value=[]))

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
        "global_vibe": "",
        "reflection_summary": "",
        "subjective_appraisals": [],
        "interaction_subtext": "",
        "last_relationship_insight": "",
        "new_facts": [],
        "future_promises": [],
        "user_profile": {"global_user_id": "user-1", "affinity": 500},
        "affinity_delta": 0,
        "decontexualized_input": "hello",
        "consolidation_origin": _consolidation_origin(),
    }
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)

    await persistence_module.db_writer(state)

    assert len(gather_calls) == 1
    assert len(gather_calls[0][0]) == 1
    assert gather_calls[0][1] is True
