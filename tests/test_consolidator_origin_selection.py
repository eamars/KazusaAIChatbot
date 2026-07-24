"""Tests for consolidator origin selection."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeV1,
    build_self_cognition_episode,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.consolidation import core as consolidator_module
from kazusa_ai_chatbot.consolidation.origin import (
    build_self_cognition_consolidation_origin,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock


def _self_cognition_episode() -> CognitiveEpisodeV1:
    """Build a canonical self-cognition episode accepted by consolidation.

    Returns:
        Valid private self-cognition episode for origin-selection tests.
    """
    turn_clock = build_turn_clock("2026-05-10 21:00:00")
    return build_self_cognition_episode(
        case={
            "case_id": "self-cognition-episode-1",
            "source_case_kind": "internal_context",
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
            "privacy_scope": "private",
        },
        percepts=[{
            "schema_version": "percept.v1",
            "percept_kind": "self_cognition_context",
            "source_kind": "self_cognition",
            "source_id": "self-cognition-episode-1",
            "content": {
                "semantic_text": "The missed promise still feels unresolved.",
            },
            "observed_at": turn_clock["storage_timestamp_utc"],
        }],
        evidence_refs=[],
        local_time_context=turn_clock["local_time_context"],
        created_at=turn_clock["storage_timestamp_utc"],
    )


def _self_cognition_global_state() -> dict:
    """Build a minimal self-cognition state for direct consolidator calls.

    Returns:
        Global persona state fields consumed by `call_consolidation_subgraph`.
    """
    turn_clock = build_turn_clock("2026-05-10 21:00:00")
    state = {
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
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
        "platform_channel_id": "channel-1",
        "channel_type": "private",
        "platform_message_id": "self_cognition:case-1",
        "action_directives": {
            "linguistic_directives": {"content_plan": {}},
        },
        "internal_monologue": "The missed promise still feels unresolved.",
        "final_dialog": ["Private finalization for consolidation only."],
        "interaction_subtext": "unresolved relationship event",
        "emotional_appraisal": "hurt and uncertain",
        "character_intent": "CLARIFY",
        "logical_stance": "TENTATIVE",
        "character_profile": {"name": "Kazusa"},
        "rag_result": {
            "user_memory_unit_candidates": [],
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
        "decontextualized_input": "Internal thought about a missed promise.",
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
    seen_pipeline_state = {}

    async def _lane_pipeline(node_state: dict) -> dict:
        """Capture origin metadata seen by the lane pipeline.

        Args:
            node_state: Consolidator state passed into the patched pipeline.

        Returns:
            Deterministic lane-pipeline packet.
        """
        seen_pipeline_state.update(node_state)
        pipeline_state = {
            **node_state,
            "mood": "hurt",
            "vibe_check": "uneasy",
            "character_reflection": "summary",
            "subjective_appraisals": ["The silence felt disappointing."],
            "relationship_delta": -1,
            "semantic_relationship_projection": "unreliable",
            "new_facts": [],
            "future_promises": [],
            "metadata": {"write_success": {}},
        }
        return {
            "router_tasks": [],
            "state": pipeline_state,
        }

    monkeypatch.setattr(
        consolidator_module,
        "run_consolidation_lane_pipeline",
        _lane_pipeline,
    )

    result = await consolidator_module.call_consolidation_subgraph(state)

    assert seen_pipeline_state["consolidation_origin"] == expected_origin
    assert set(result) == {
        "new_facts",
        "future_promises",
        "consolidation_metadata",
    }
