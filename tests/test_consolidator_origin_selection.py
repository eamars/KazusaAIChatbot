"""Tests for consolidator origin selection."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_episode import CognitiveEpisode
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator as consolidator_module,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_origin import (
    build_self_cognition_consolidation_origin,
)
from kazusa_ai_chatbot.time_context import build_character_time_context


def _self_cognition_episode() -> CognitiveEpisode:
    """Build an internal-thought episode accepted by consolidation.

    Returns:
        Valid internal-thought preview episode for origin-selection tests.
    """
    timestamp = "2026-05-10T21:00:00+12:00"
    episode: CognitiveEpisode = {
        "episode_id": "self-cognition-episode-1",
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "preview",
        "percepts": [
            {
                "percept_id": "self-cognition-percept-1",
                "input_source": "internal_monologue",
                "content": "The missed promise still feels unresolved.",
                "visibility": "model_visible",
                "metadata": {"source": "self_cognition_source_packet"},
            }
        ],
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "channel-1",
            "channel_type": "private",
            "current_platform_user_id": "platform-user-1",
            "current_global_user_id": "global-user-1",
            "current_display_name": "Test User",
            "target_addressed_user_ids": ["global-user-1"],
            "target_broadcast": False,
        },
        "origin_metadata": {
            "platform": "qq",
            "platform_message_id": "self_cognition:case-1",
            "active_turn_platform_message_ids": [],
            "active_turn_conversation_row_ids": [],
            "debug_modes": {"no_visual_directives": True},
        },
        "timestamp": timestamp,
        "time_context": build_character_time_context(timestamp),
    }
    return episode


def _self_cognition_global_state() -> dict:
    """Build a minimal self-cognition state for direct consolidator calls.

    Returns:
        Global persona state fields consumed by `call_consolidation_subgraph`.
    """
    timestamp = "2026-05-10T21:00:00+12:00"
    state = {
        "timestamp": timestamp,
        "time_context": build_character_time_context(timestamp),
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_profile": {"affinity": 500},
        "platform": "qq",
        "platform_channel_id": "channel-1",
        "channel_type": "private",
        "platform_message_id": "self_cognition:case-1",
        "action_directives": {
            "linguistic_directives": {"content_anchors": []},
        },
        "internal_monologue": "The missed promise still feels unresolved.",
        "final_dialog": ["Private finalization for consolidation only."],
        "interaction_subtext": "unresolved relationship event",
        "emotional_appraisal": "hurt and uncertain",
        "character_intent": "CLARIFY",
        "logical_stance": "TENTATIVE",
        "character_profile": {"name": "Kazusa"},
        "rag_result": {
            "user_image": {
                "user_memory_context": {
                    "stable_patterns": [],
                    "recent_shifts": [],
                    "objective_facts": [],
                    "active_commitments": [],
                    "milestones": [],
                }
            }
        },
        "decontexualized_input": "Internal thought about a missed promise.",
        "chat_history_recent": [],
        "cognitive_episode": _self_cognition_episode(),
    }
    return state


@pytest.mark.asyncio
async def test_call_consolidation_subgraph_selects_self_cognition_origin(
    monkeypatch,
) -> None:
    """Internal-thought episodes should enter the same consolidator graph."""
    state = _self_cognition_global_state()
    expected_origin = build_self_cognition_consolidation_origin(
        episode=state["cognitive_episode"],
    )
    seen_origins = {}

    async def _global_state_updater(node_state: dict) -> dict:
        """Capture origin metadata seen by the global-state updater.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Deterministic global-state updater output.
        """
        seen_origins["global_state_updater"] = node_state["consolidation_origin"]
        return {
            "mood": "hurt",
            "global_vibe": "uneasy",
            "reflection_summary": "summary",
        }

    async def _relationship_recorder(node_state: dict) -> dict:
        """Capture origin metadata seen by the relationship recorder.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Deterministic relationship-recorder output.
        """
        seen_origins["relationship_recorder"] = node_state["consolidation_origin"]
        return {
            "subjective_appraisals": ["The silence felt disappointing."],
            "affinity_delta": -1,
            "last_relationship_insight": "unreliable",
        }

    async def _facts_harvester(node_state: dict) -> dict:
        """Capture origin metadata seen by the facts harvester.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Empty facts and promises so the evaluator is skipped.
        """
        seen_origins["facts_harvester"] = node_state["consolidation_origin"]
        return {"new_facts": [], "future_promises": []}

    async def _fact_harvester_evaluator(_node_state: dict) -> dict:
        """Fail if the evaluator runs for empty harvester output.

        Args:
            _node_state: Unused consolidator state for an unexpected call.

        Returns:
            This helper never returns on the valid test path.

        Raises:
            AssertionError: Always, because evaluator should be skipped.
        """
        raise AssertionError("fact evaluator should be skipped")

    async def _db_writer(node_state: dict) -> dict:
        """Capture origin metadata seen by the persistence boundary.

        Args:
            node_state: Consolidator state passed into the patched node.

        Returns:
            Deterministic persistence metadata.
        """
        seen_origins["db_writer"] = node_state["consolidation_origin"]
        return {"metadata": {"write_success": {}}}

    monkeypatch.setattr(
        consolidator_module,
        "global_state_updater",
        _global_state_updater,
    )
    monkeypatch.setattr(
        consolidator_module,
        "relationship_recorder",
        _relationship_recorder,
    )
    monkeypatch.setattr(consolidator_module, "facts_harvester", _facts_harvester)
    monkeypatch.setattr(
        consolidator_module,
        "fact_harvester_evaluator",
        _fact_harvester_evaluator,
    )
    monkeypatch.setattr(consolidator_module, "db_writer", _db_writer)

    result = await consolidator_module.call_consolidation_subgraph(state)

    assert seen_origins == {
        "global_state_updater": expected_origin,
        "relationship_recorder": expected_origin,
        "facts_harvester": expected_origin,
        "db_writer": expected_origin,
    }
    assert result["mood"] == "hurt"
    assert result["affinity_delta"] == -1
