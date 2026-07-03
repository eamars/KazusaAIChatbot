"""Integration tests for the consolidator lane-router bigbang cutover."""

from __future__ import annotations

import importlib
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.consolidation import persistence as persistence_module
from kazusa_ai_chatbot.consolidation import core as core_module
from kazusa_ai_chatbot.consolidation.target import (
    build_consolidation_target_plan,
    validate_write_intent,
)


def _lane_router_module() -> Any:
    """Import the planned lane-router module with a clear failure."""

    try:
        module = importlib.import_module(
            "kazusa_ai_chatbot.consolidation.lane_router"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Missing consolidation.lane_router module required by the "
            "lane-router bigbang plan."
        )
        raise exc
    return module


def _base_state() -> dict[str, Any]:
    """Build a normal private-chat state for integration assertions."""

    state: dict[str, Any] = {
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_profile": {
            "global_user_id": "global-user-1",
            "affinity": 500,
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
    }
    return state


def _targets_by_alias(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index target-plan rows by alias."""

    targets = {
        target["target_alias"]: target
        for target in plan["targets"]
    }
    return targets


def test_core_no_longer_imports_or_wires_mono_fact_harvester() -> None:
    """Runtime consolidation must not preserve the old broad harvester path."""

    source = inspect.getsource(core_module)

    assert "facts_harvester" not in source
    assert "fact_harvester_evaluator" not in source
    assert "run_consolidation_lane_pipeline" in source


def test_character_self_guidance_is_character_write_lane() -> None:
    """Accepted Kazusa behavior guidance should target character ownership."""

    plan = build_consolidation_target_plan(_base_state())
    targets = _targets_by_alias(plan)
    character_target = targets["character"]

    assert "character_self_guidance" in character_target["write_lanes"]
    validated_intent = validate_write_intent(
        {
            "target_alias": "character",
            "write_lane": "character_self_guidance",
            "payload": {
                "memory_type": "defense_rule",
                "content": "Kazusa may join repetition chains if she accepts.",
            },
        },
        plan,
    )

    assert validated_intent["target_alias"] == "character"
    assert validated_intent["write_lane"] == "character_self_guidance"


def test_user_target_cannot_receive_character_self_guidance() -> None:
    """Global Kazusa behavior guidance must not be stored as user memory."""

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


@pytest.mark.asyncio
async def test_lane_pipeline_returns_auditable_dry_run_packet(monkeypatch) -> None:
    """The new pipeline should expose inspectable dry-run write decisions."""

    module = _lane_router_module()
    state = _base_state()
    state.update({
        "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
        "local_time_context": {"current_date": "2026-07-03"},
        "decontexualized_input": "I now work in Auckland.",
        "final_dialog": ["Kazusa acknowledges the user's update."],
        "internal_monologue": "",
        "chat_history_recent": [],
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
        "consolidation_target_plan": build_consolidation_target_plan(state),
    })

    async def _fake_router(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "lane_tasks": [
                {
                    "lane": "user_memory_units",
                    "reason": "user stated a durable personal fact",
                    "source_keys": ["current_turn_user_message"],
                }
            ]
        }

    monkeypatch.setattr(module, "call_lane_router_llm", _fake_router)
    packet = await module.run_consolidation_lane_pipeline(
        state,
        dry_run=True,
    )

    assert packet["mode"] == "dry_run"
    assert packet["router_tasks"]
    assert "write_intents" in packet
    assert "source_views" in packet


@pytest.mark.asyncio
async def test_self_guidance_pipeline_threads_refs_to_specialist_and_writer(
    monkeypatch,
) -> None:
    """Accepted character guidance should write through the character lane."""

    module = _lane_router_module()
    state = _base_state()
    state.update({
        "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
        "local_time_context": {"current_date": "2026-07-03"},
        "decontexualized_input": (
            "Kazusa, you may join harmless repetition chains if you see one."
        ),
        "final_dialog": ["Sure, I can join them when it fits."],
        "internal_monologue": "",
        "chat_history_recent": [],
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
        "metadata": {},
        "consolidation_origin": {
            "episode_id": "episode-self-guidance-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
            "platform": "qq",
            "platform_channel_id": "private-1",
            "channel_type": "private",
            "platform_message_id": "message-self-guidance-1",
            "active_turn_platform_message_ids": [
                "message-self-guidance-1",
            ],
            "active_turn_conversation_row_ids": ["conversation-row-1"],
            "current_platform_user_id": "platform-user-1",
            "current_global_user_id": "global-user-1",
            "current_display_name": "Test User",
        },
    })
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)
    captured: dict[str, Any] = {}

    async def _fake_router(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "lane_tasks": [
                {
                    "lane": "character_self_guidance",
                    "reason": "accepted character behavior guidance",
                    "source_keys": [
                        "current_turn_user_message",
                        "assistant_final_dialog",
                    ],
                }
            ]
        }

    async def _fake_specialist(node_state: dict[str, Any]) -> dict[str, Any]:
        captured["specialist_refs"] = node_state[
            "character_self_guidance_source_refs"
        ]
        return {
            "character_self_guidance": {
                "memory_name": "Harmless repetition chains",
                "content": (
                    "Kazusa may join harmless repetition chains when she "
                    "accepts that social rhythm."
                ),
                "memory_type": "defense_rule",
            }
        }

    async def _fake_db_writer(node_state: dict[str, Any]) -> dict[str, Any]:
        captured["writer_state"] = node_state
        return {
            "metadata": {
                "write_success": {"character_self_guidance": True},
            }
        }

    monkeypatch.setattr(module, "call_lane_router_llm", _fake_router)
    monkeypatch.setattr(
        module,
        "character_self_guidance_specialist",
        _fake_specialist,
    )
    monkeypatch.setattr(module, "db_writer", _fake_db_writer)

    packet = await module.run_consolidation_lane_pipeline(state)

    writer_state = captured["writer_state"]
    assert packet["accepted_lanes"] == ["character_self_guidance"]
    assert writer_state["enabled_consolidation_write_lanes"] == [
        "character_self_guidance"
    ]
    assert captured["specialist_refs"]
    assert writer_state["character_self_guidance"]["memory_type"] == (
        "defense_rule"
    )


@pytest.mark.asyncio
async def test_acceptance_lane_completes_required_source_refs(monkeypatch) -> None:
    """Accepted-rule lanes should complete request plus acceptance provenance."""

    module = _lane_router_module()
    state = _base_state()
    state.update({
        "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
        "local_time_context": {"current_date": "2026-07-03"},
        "decontexualized_input": "以后你跟我说话叫我阿然，好吗？",
        "final_dialog": ["好，以后和你说话时我叫你阿然。"],
        "internal_monologue": "",
        "chat_history_recent": [],
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
        "metadata": {},
        "consolidation_origin": {
            "episode_id": "episode-commitment-source-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
            "platform": "qq",
            "platform_channel_id": "private-1",
            "channel_type": "private",
            "platform_message_id": "message-commitment-source-1",
            "active_turn_platform_message_ids": [
                "message-commitment-source-1",
            ],
            "active_turn_conversation_row_ids": ["conversation-row-1"],
            "current_platform_user_id": "platform-user-1",
            "current_global_user_id": "global-user-1",
            "current_display_name": "Test User",
        },
    })
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)

    async def _fake_router(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "lane_tasks": [
                {
                    "lane": "active_commitment",
                    "reason": "accepted user-scoped address rule",
                    "source_keys": ["assistant_final_dialog"],
                }
            ]
        }

    monkeypatch.setattr(module, "call_lane_router_llm", _fake_router)

    packet = await module.run_consolidation_lane_pipeline(state, dry_run=True)
    lane_result = packet["lane_results"][0]
    source_keys = set(lane_result["source_keys"])
    source_refs = packet["write_intents"][0]["payload"]["source_refs"]
    source_ref_kinds = {
        source_ref["source"]
        for source_ref in source_refs
    }

    assert packet["accepted_lanes"] == ["active_commitment"]
    assert {
        "current_turn_user_message",
        "assistant_final_dialog",
    }.issubset(source_keys)
    assert {"user_message", "assistant_final_dialog"}.issubset(
        source_ref_kinds
    )


@pytest.mark.asyncio
async def test_character_and_relationship_lanes_run_reviewers_before_writer(
    monkeypatch,
) -> None:
    """Character and relationship writes must pass through lane reviewers."""

    module = _lane_router_module()
    state = _base_state()
    state.update({
        "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
        "local_time_context": {"current_date": "2026-07-03"},
        "decontexualized_input": "Your last reply felt a little dismissive.",
        "final_dialog": ["Kazusa acknowledges the user's discomfort."],
        "internal_monologue": "Kazusa notices the relationship tension.",
        "emotional_appraisal": "slightly concerned",
        "interaction_subtext": "repair after a rough reply",
        "logical_stance": "accepts the feedback",
        "character_intent": "repair the interaction",
        "chat_history_recent": [],
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
        "metadata": {},
        "consolidation_origin": {
            "episode_id": "episode-review-threading-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
            "platform": "qq",
            "platform_channel_id": "private-1",
            "channel_type": "private",
            "platform_message_id": "message-review-threading-1",
            "active_turn_platform_message_ids": [
                "message-review-threading-1",
            ],
            "active_turn_conversation_row_ids": ["conversation-row-1"],
            "current_platform_user_id": "platform-user-1",
            "current_global_user_id": "global-user-1",
            "current_display_name": "Test User",
        },
    })
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)
    captured: dict[str, Any] = {"calls": []}

    async def _fake_router(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "lane_tasks": [
                {
                    "lane": "character_state",
                    "reason": "durable character state shift",
                    "source_keys": ["assistant_final_dialog"],
                },
                {
                    "lane": "relationship_profile",
                    "reason": "relationship feedback from user",
                    "source_keys": ["current_turn_user_message"],
                },
            ]
        }

    async def _fake_character_specialist(
        node_state: dict[str, Any],
    ) -> dict[str, Any]:
        del node_state
        captured["calls"].append("character_specialist")
        return {
            "mood": "raw mood",
            "global_vibe": "raw vibe",
            "reflection_summary": "raw character summary",
        }

    async def _fake_character_reviewer(
        node_state: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        del node_state
        captured["calls"].append("character_reviewer")
        assert candidate["mood"] == "raw mood"
        return {
            "mood": "reviewed mood",
            "global_vibe": "reviewed vibe",
            "reflection_summary": "reviewed character summary",
        }

    async def _fake_relationship_specialist(
        node_state: dict[str, Any],
    ) -> dict[str, Any]:
        del node_state
        captured["calls"].append("relationship_specialist")
        return {
            "subjective_appraisals": ["raw relationship appraisal"],
            "affinity_delta": 4,
            "last_relationship_insight": "raw insight",
        }

    async def _fake_relationship_reviewer(
        node_state: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        del node_state
        captured["calls"].append("relationship_reviewer")
        assert candidate["last_relationship_insight"] == "raw insight"
        return {
            "subjective_appraisals": ["reviewed relationship appraisal"],
            "affinity_delta": 1,
            "last_relationship_insight": "reviewed insight",
        }

    async def _fake_db_writer(node_state: dict[str, Any]) -> dict[str, Any]:
        captured["writer_state"] = dict(node_state)
        return {
            "metadata": {
                "write_success": {
                    "character_state": True,
                    "relationship_profile": True,
                },
            }
        }

    monkeypatch.setattr(module, "call_lane_router_llm", _fake_router)
    monkeypatch.setattr(module, "global_state_updater", _fake_character_specialist)
    monkeypatch.setattr(module, "character_state_reviewer", _fake_character_reviewer)
    monkeypatch.setattr(module, "relationship_recorder", _fake_relationship_specialist)
    monkeypatch.setattr(
        module,
        "relationship_profile_reviewer",
        _fake_relationship_reviewer,
    )
    monkeypatch.setattr(module, "db_writer", _fake_db_writer)

    packet = await module.run_consolidation_lane_pipeline(state)

    writer_state = captured["writer_state"]
    assert captured["calls"] == [
        "character_specialist",
        "character_reviewer",
        "relationship_specialist",
        "relationship_reviewer",
    ]
    assert packet["accepted_lanes"] == [
        "character_state",
        "relationship_profile",
    ]
    assert writer_state["mood"] == "reviewed mood"
    assert writer_state["global_vibe"] == "reviewed vibe"
    assert writer_state["reflection_summary"] == "reviewed character summary"
    assert writer_state["subjective_appraisals"] == [
        "reviewed relationship appraisal",
    ]
    assert writer_state["affinity_delta"] == 1
    assert writer_state["last_relationship_insight"] == "reviewed insight"


@pytest.mark.asyncio
async def test_reviewer_rejection_disables_lane_before_writer(monkeypatch) -> None:
    """Reviewer rejection should not persist an empty accepted lane."""

    module = _lane_router_module()
    state = _base_state()
    state.update({
        "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
        "local_time_context": {"current_date": "2026-07-03"},
        "decontexualized_input": "A routine one-turn acknowledgement.",
        "final_dialog": ["Kazusa answers normally."],
        "internal_monologue": "",
        "emotional_appraisal": "",
        "interaction_subtext": "",
        "logical_stance": "neutral",
        "character_intent": "provide",
        "chat_history_recent": [],
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
        "metadata": {},
        "consolidation_origin": {
            "episode_id": "episode-review-reject-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
            "platform": "qq",
            "platform_channel_id": "private-1",
            "channel_type": "private",
            "platform_message_id": "message-review-reject-1",
            "active_turn_platform_message_ids": ["message-review-reject-1"],
            "active_turn_conversation_row_ids": ["conversation-row-1"],
            "current_platform_user_id": "platform-user-1",
            "current_global_user_id": "global-user-1",
            "current_display_name": "Test User",
        },
    })
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)

    async def _fake_router(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "lane_tasks": [
                {
                    "lane": "character_state",
                    "reason": "candidate later rejected by reviewer",
                    "source_keys": ["assistant_final_dialog"],
                }
            ]
        }

    async def _fake_character_specialist(
        node_state: dict[str, Any],
    ) -> dict[str, Any]:
        del node_state
        return {
            "mood": "raw mood",
            "global_vibe": "raw vibe",
            "reflection_summary": "raw character summary",
        }

    async def _fake_character_reviewer(
        node_state: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        del node_state, candidate
        return {
            "mood": "",
            "global_vibe": "",
            "reflection_summary": "",
        }

    db_writer = AsyncMock(return_value={"metadata": {"write_success": {}}})

    monkeypatch.setattr(module, "call_lane_router_llm", _fake_router)
    monkeypatch.setattr(module, "global_state_updater", _fake_character_specialist)
    monkeypatch.setattr(module, "character_state_reviewer", _fake_character_reviewer)
    monkeypatch.setattr(module, "db_writer", db_writer)

    packet = await module.run_consolidation_lane_pipeline(state)

    db_writer.assert_not_awaited()
    assert packet["accepted_lanes"] == []
    assert packet["state"]["enabled_consolidation_write_lanes"] == []
    assert packet["state"]["metadata"]["review_rejected_lanes"] == [
        "character_state",
    ]
    assert packet["state"]["metadata"]["write_success"] == {}


@pytest.mark.asyncio
async def test_lane_pipeline_drops_non_roster_router_output(monkeypatch) -> None:
    """Invalid router lane choices should fail closed at runtime."""

    module = _lane_router_module()
    state = _base_state()
    state["global_user_id"] = ""
    state["user_profile"] = {}
    state["cognitive_episode"]["target_scope"]["current_global_user_id"] = ""
    state.update({
        "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
        "local_time_context": {"current_date": "2026-07-03"},
        "decontexualized_input": "记住我喜欢清淡一点。",
        "final_dialog": ["千纱接受了这个说法。"],
        "internal_monologue": "",
        "chat_history_recent": [],
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
        "metadata": {},
        "consolidation_origin": {
            "episode_id": "episode-no-user-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
            "platform": "qq",
            "platform_channel_id": "private-1",
            "channel_type": "private",
            "platform_message_id": "message-no-user-1",
            "active_turn_platform_message_ids": ["message-no-user-1"],
            "active_turn_conversation_row_ids": ["conversation-row-1"],
            "current_platform_user_id": "",
            "current_global_user_id": "",
            "current_display_name": "Test User",
        },
    })
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)

    async def _fake_router(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "lane_tasks": [
                {
                    "lane": "active_commitment",
                    "reason": "non-roster user lane",
                    "source_keys": ["current_turn_user_message"],
                }
            ]
        }

    monkeypatch.setattr(module, "call_lane_router_llm", _fake_router)

    packet = await module.run_consolidation_lane_pipeline(state, dry_run=True)

    assert packet["accepted_lanes"] == []
    assert packet["write_intents"] == []
    assert "router_validation_error" in packet["state"]["metadata"][
        "lane_pipeline"
    ]


@pytest.mark.asyncio
async def test_db_writer_persists_reviewed_character_self_guidance(
    monkeypatch,
) -> None:
    """The writer should persist accepted self-guidance through memory storage."""

    state = _base_state()
    state.update({
        "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
        "local_time_context": {"current_date": "2026-07-03"},
        "platform_message_id": "message-self-guidance-1",
        "metadata": {},
        "mood": "",
        "global_vibe": "",
        "reflection_summary": "",
        "subjective_appraisals": [],
        "interaction_subtext": "",
        "last_relationship_insight": "",
        "new_facts": [],
        "future_promises": [],
        "affinity_delta": 0,
        "decontexualized_input": (
            "Kazusa, you may join harmless repetition chains if you see one."
        ),
        "final_dialog": ["Sure, I can join them when it fits."],
        "action_directives": {"linguistic_directives": {"content_plan": {}}},
        "consolidation_origin": {
            "episode_id": "episode-self-guidance-1",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
            "platform": "qq",
            "platform_channel_id": "private-1",
            "channel_type": "private",
            "platform_message_id": "message-self-guidance-1",
            "active_turn_platform_message_ids": [
                "message-self-guidance-1",
            ],
            "active_turn_conversation_row_ids": ["conversation-row-1"],
            "current_platform_user_id": "platform-user-1",
            "current_global_user_id": "global-user-1",
            "current_display_name": "Test User",
        },
        "enabled_consolidation_write_lanes": ["character_self_guidance"],
        "character_self_guidance": {
            "memory_name": "Harmless repetition chains",
            "content": (
                "Kazusa may join harmless repetition chains when she accepts "
                "that social rhythm."
            ),
            "memory_type": "defense_rule",
        },
        "character_self_guidance_source_refs": [
            {
                "source": "user_message",
                "platform_message_id": "message-self-guidance-1",
            },
            {
                "source": "assistant_final_dialog",
                "platform_message_id": "message-self-guidance-1",
            },
        ],
    })
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)
    persist_self_guidance = AsyncMock(return_value={
        "memory_unit_id": "conversation-self-guidance-1",
        "memory_type": "defense_rule",
        "memory_name": "Harmless repetition chains",
        "content": "Kazusa may join harmless repetition chains.",
    })
    runtime = MagicMock()
    runtime.invalidate = AsyncMock(return_value=0)

    monkeypatch.setattr(
        persistence_module,
        "persist_character_self_guidance_from_state",
        persist_self_guidance,
    )
    monkeypatch.setattr(
        persistence_module,
        "get_rag_cache2_runtime",
        MagicMock(return_value=runtime),
    )

    result = await persistence_module.db_writer(state)

    persist_self_guidance.assert_awaited_once()
    assert result["metadata"]["write_success"]["character_self_guidance"] is True
    assert result["metadata"]["character_self_guidance_result"][
        "memory_type"
    ] == "defense_rule"
