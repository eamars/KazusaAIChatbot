"""Deterministic canonical episode fixtures for cognition core V2 tests."""

from typing import Any

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    InputSource,
    OutputMode,
    TriggerSource,
    validate_cognitive_episode,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)


NOW = "2026-07-14T00:00:00Z"


def canonical_episode(
    *,
    episode_id: str = "v2-test-episode",
    trigger_source: TriggerSource = "user_message",
    output_mode: OutputMode = "visible_reply",
    content: str = "a grounded current episode",
    current_global_user_id: str = "v2-test-user",
    metadata: dict[str, Any] | None = None,
) -> CognitiveEpisode:
    """Build one exact episode across the supported deterministic sources."""

    source_by_trigger: dict[TriggerSource, InputSource] = {
        "user_message": "dialog_text",
        "reflection_signal": "reflection_artifact",
        "internal_thought": "internal_monologue",
        "scheduled_recall": "retrieved_memory",
        "system_probe": "internal_monologue",
        "accepted_task_result_ready": "accepted_task_result",
    }
    input_source = source_by_trigger[trigger_source]
    episode: CognitiveEpisode = {
        "episode_id": episode_id,
        "trigger_source": trigger_source,
        "input_sources": [input_source],
        "output_mode": output_mode,
        "percepts": [{
            "percept_id": f"percept:{episode_id}",
            "input_source": input_source,
            "content": content,
            "visibility": "model_visible",
            "metadata": dict(metadata or {}),
        }],
        "target_scope": {
            "platform": "debug",
            "platform_channel_id": "channel-test",
            "channel_type": "private",
            "current_platform_user_id": "platform-user-test",
            "current_global_user_id": current_global_user_id,
            "current_display_name": "Test User",
            "target_addressed_user_ids": [current_global_user_id],
            "target_broadcast": False,
        },
        "origin_metadata": {
            "platform": "debug",
            "platform_message_id": "message-test",
            "active_turn_platform_message_ids": ["message-test"],
            "active_turn_conversation_row_ids": [],
            "debug_modes": {},
        },
        "storage_timestamp_utc": NOW,
        "local_time_context": {
            "current_local_datetime": "2026-07-14 12:00",
            "current_local_weekday": "Tuesday",
        },
    }
    validate_cognitive_episode(episode)
    return episode


def canonical_cognition_output(
    *,
    route: str = "speech",
    owner_user_id: str = "v2-test-user",
) -> dict[str, Any]:
    """Build one exact committed V2 cognition output for connector tests."""

    output: dict[str, Any] = {
        "schema_version": "cognition_core_output.v2",
        "intention": {
            "selected_branch_id": "ordinary_response",
            "route": route,
            "intention": "acknowledge the grounded episode",
            "target_roles": [],
            "reason": "the current episode establishes the selected route",
        },
        "supporting_bids": [],
        "state_update": {
            "state_scope": "user",
            "owner_key": owner_user_id,
            "replacement_state": build_acquaintance_user_state(
                global_user_id=owner_user_id,
                updated_at=NOW,
            ),
            "comparison_results": [],
            "changed_paths": [],
        },
        "affect_projection": [],
        "action_requests": [],
        "resolver_requests": [],
        "resolver_progress": {
            "status": "not_requested",
            "semantic_summary": "no resolver was selected",
        },
        "selected_bid_reason": "the current episode is grounded",
        "private_monologue": "I want to acknowledge this clearly.",
        "expression_policy": {
            "visibility": "visible" if route == "speech" else "none",
            "emotional_tone": "composed",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "diagnostics": {
            "run_id": "canonical-cognition-output",
            "stage_status": {},
            "selected_question_count": 0,
            "dispatched_question_count": 0,
            "selected_branch_count": 1,
            "dispatched_branch_count": 1,
            "completed_branch_count": 1,
            "failed_branch_count": 0,
            "overlap_ms": 0,
            "dependency_wait_ms": 0,
            "total_ms": 0,
            "warnings": [],
        },
    }
    if route == "speech":
        output["admitted_bid"] = {
            "branch_id": "ordinary_response",
            "goal_ref": {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary-response",
            },
            "intention": "acknowledge the grounded episode",
            "desired_outcome": "maintain continuity",
            "concrete_detail": "use only the current grounded episode",
            "reason": "the current episode establishes the selected route",
            "private_monologue": "I want to acknowledge this clearly.",
            "target_roles": [],
            "evidence_handles": ["e1"],
            "expected_consequences": ["preserve continuity"],
            "confidence": "high",
            "requested_route": "speech",
        }
    return output
