"""Deterministic canonical episode fixtures for cognition core V2 tests."""

from typing import Any

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeV1,
    MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS,
    TriggerSource,
    build_text_chat_media_description_rows,
    build_user_message_episode,
    validate_cognitive_episode_v1,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)


NOW = "2026-07-14T00:00:00Z"


def canonical_user_message_episode(
    *,
    episode_id: str,
    percept_id: str,
    storage_timestamp_utc: str,
    local_time_context: dict[str, Any],
    user_input: str,
    platform: str,
    platform_channel_id: str,
    channel_type: str,
    platform_message_id: str,
    platform_user_id: str,
    global_user_id: str,
    user_name: str,
    active_turn_platform_message_ids: list[str] | None = None,
    active_turn_conversation_row_ids: list[str] | None = None,
    debug_modes: dict[str, bool] | None = None,
    output_mode: str | None = None,
    target_addressed_user_ids: list[str] | None = None,
    target_broadcast: bool = False,
    media_description_rows: list[dict[str, Any]] | None = None,
) -> CognitiveEpisodeV1:
    """Build a canonical user-message episode for shared test fixtures."""

    del output_mode
    media_percepts: list[dict[str, Any]] = []
    for index, row in enumerate(
        build_text_chat_media_description_rows(media_description_rows or [])[
            :MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS
        ],
        start=1,
    ):
        content_type = row["content_type"]
        source_kind = (
            "image_observation"
            if content_type.startswith("image/")
            else "audio_observation"
        )
        media_percepts.append({
            "schema_version": "percept.v1",
            "percept_kind": source_kind,
            "source_kind": source_kind,
            "source_id": f"{episode_id}:media:{index}",
            "content": {
                "content_type": content_type,
                "description": row["description"],
                "observation": dict(row.get("image_observation", {})),
            },
            "observed_at": storage_timestamp_utc,
        })
    dialog_percept = {
        "schema_version": "percept.v1",
        "percept_kind": "dialog",
        "source_kind": "dialog",
        "source_id": percept_id,
        "content": {
            "semantic_text": user_input,
            "text": user_input,
        },
        "observed_at": storage_timestamp_utc,
    }
    origin = {
        "schema_version": "user_message_origin.v1",
        "owner": "tests.cognition_core_v2",
        "platform": platform,
        "platform_message_id": platform_message_id,
        "active_turn_platform_message_ids": list(
            active_turn_platform_message_ids or []
        ),
        "active_turn_conversation_row_ids": list(
            active_turn_conversation_row_ids or []
        ),
        "debug_modes": dict(debug_modes or {}),
        "privacy_scope": "private",
        "delivery_permission_ref": "",
    }
    target_scope = {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "current_platform_user_id": platform_user_id,
        "current_global_user_id": global_user_id,
        "current_display_name": user_name,
        "target_addressed_user_ids": list(target_addressed_user_ids or []),
        "target_broadcast": target_broadcast,
    }
    return build_user_message_episode(
        episode_id=episode_id,
        origin=origin,
        target_scope=target_scope,
        dialog_percept=dialog_percept,
        media_percepts=media_percepts,
        evidence_refs=[],
        local_time_context=local_time_context,
        created_at=storage_timestamp_utc,
        debug_controls=dict(debug_modes or {}),
    )


def canonical_episode(
    *,
    episode_id: str = "v2-test-episode",
    trigger_source: TriggerSource = "user_message",
    content: str = "a grounded current episode",
    current_global_user_id: str = "v2-test-user",
    metadata: dict[str, Any] | None = None,
) -> CognitiveEpisodeV1:
    """Build one exact episode across the five native deterministic sources."""

    source_kind_by_trigger: dict[TriggerSource, str] = {
        "user_message": "dialog",
        "internal_thought": "internal_thought",
        "self_cognition": "self_cognition",
        "scheduled_tick": "scheduled_tick",
        "tool_result": "tool_result",
    }
    source_kind = source_kind_by_trigger[trigger_source]
    percept_kind = "dialog" if trigger_source == "user_message" else source_kind
    percept_content: dict[str, Any] = {
        "semantic_text": content,
        "text": content,
    }
    if metadata:
        percept_content.update(metadata)
    percepts = [{
        "schema_version": "percept.v1",
        "percept_kind": percept_kind,
        "source_kind": source_kind,
        "source_id": f"percept:{episode_id}",
        "content": percept_content,
        "observed_at": NOW,
    }]
    if trigger_source == "user_message":
        percepts.append({
            "schema_version": "percept.v1",
            "percept_kind": "local_time_context",
            "source_kind": "system_event",
            "source_id": None,
            "content": {
                "local_time_context": {
                    "current_local_datetime": "2026-07-14 12:00",
                    "current_local_weekday": "Tuesday",
                },
            },
            "observed_at": NOW,
        })
    episode: CognitiveEpisodeV1 = {
        "schema_version": "cognitive_episode.v1",
        "episode_id": episode_id,
        "trigger_source": trigger_source,
        "percepts": percepts,
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
            "schema_version": f"{trigger_source}_origin.v1",
            "owner": "tests.cognition_core_v2",
            "platform": "debug",
            "platform_message_id": "message-test",
            "active_turn_platform_message_ids": ["message-test"],
            "active_turn_conversation_row_ids": [],
            "debug_modes": {},
            "privacy_scope": "private",
            "delivery_permission_ref": "",
            "created_at": NOW,
        },
        "evidence_refs": [],
        "created_at": NOW,
        "privacy_scope": "private",
        "continuation_depth": 0,
    }
    return validate_cognitive_episode_v1(episode)


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
        "goal_resolution": "answerable_now",
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
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
        }
    return output
