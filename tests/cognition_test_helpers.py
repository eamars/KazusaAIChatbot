"""Deterministic canonical episode fixtures for cognition tests."""

from typing import Any

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.projection import (
    identity_projection_digest,
    project_identity_for_cognition,
    project_identity_for_surface,
    projected_identity_consumer_kinds,
)
from kazusa_ai_chatbot.cognition_episode import (
    MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS,
    CognitiveEpisodeV1,
    TriggerSource,
    build_text_chat_media_description_rows,
    build_user_message_episode,
    validate_cognitive_episode_v1,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)

NOW = "2026-07-14T00:00:00Z"


def canonical_character_identity(
    *,
    marker: str = "current",
) -> dict[str, object]:
    """Build a complete generic effective identity for cognition tests."""

    return {
        "name": f"character-{marker}",
        "description": f"description-{marker}",
        "gender": f"gender-{marker}",
        "age": 20,
        "birthday": f"birthday-{marker}",
        "backstory": f"backstory-{marker}",
        "personality_brief": {
            "mbti": f"mbti-{marker}",
            "logic": f"logic-{marker}",
            "tempo": f"tempo-{marker}",
            "defense": f"defense-{marker}",
            "quirks": f"quirks-{marker}",
            "taboos": f"taboos-{marker}",
        },
        "boundary_profile": {
            "self_integrity": 0.7,
            "control_sensitivity": 0.7,
            "compliance_strategy": "resist",
            "relational_override": 0.3,
            "control_intimacy_misread": 0.3,
            "boundary_recovery": "rebound",
            "authority_skepticism": 0.7,
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.3,
            "hesitation_density": 0.3,
            "counter_questioning": 0.3,
            "softener_density": 0.3,
            "formalism_avoidance": 0.7,
            "abstraction_reframing": 0.7,
            "direct_assertion": 0.7,
            "emotional_leakage": 0.3,
            "rhythmic_bounce": 0.7,
            "self_deprecation": 0.3,
        },
        "self_image": {
            "self_concept": f"self-concept-{marker}",
            "current_growth_edges": [f"growth-edge-{marker}"],
        },
        "visual_characterization": f"visual-{marker}",
    }


def canonical_identity_context(
    *,
    marker: str = "current",
    include_epistemic_core: bool = False,
) -> dict[str, dict[str, object]]:
    """Build the closed cognition projection from one complete identity."""

    return project_identity_for_cognition(
        {
            "effective_identity": canonical_character_identity(
                marker=marker,
            ),
        },
        include_epistemic_core=include_epistemic_core,
    )


def maximum_character_identity() -> dict[str, object]:
    """Build an identity at every declared text and growth-edge ceiling."""

    identity = canonical_character_identity(marker="maximum")
    identity.update({
        "name": "n" * models.TEXT_LIMIT_BY_PATH["name"],
        "description": "d" * models.TEXT_LIMIT_BY_PATH["description"],
        "gender": "g" * models.TEXT_LIMIT_BY_PATH["gender"],
        "birthday": "b" * models.TEXT_LIMIT_BY_PATH["birthday"],
        "backstory": "s" * models.TEXT_LIMIT_BY_PATH["backstory"],
        "personality_brief": {
            "mbti": "m" * models.TEXT_LIMIT_BY_PATH[
                "personality_brief.mbti"
            ],
            "logic": "l" * models.TEXT_LIMIT_BY_PATH[
                "personality_brief.logic"
            ],
            "tempo": "t" * models.TEXT_LIMIT_BY_PATH[
                "personality_brief.tempo"
            ],
            "defense": "f" * models.TEXT_LIMIT_BY_PATH[
                "personality_brief.defense"
            ],
            "quirks": "q" * models.TEXT_LIMIT_BY_PATH[
                "personality_brief.quirks"
            ],
            "taboos": "o" * models.TEXT_LIMIT_BY_PATH[
                "personality_brief.taboos"
            ],
        },
        "self_image": {
            "self_concept": "c" * models.TEXT_LIMIT_BY_PATH[
                "self_image.self_concept"
            ],
            "current_growth_edges": [
                f"{index:02d}" + "e" * (models.GROWTH_EDGE_LIMIT - 2)
                for index in range(models.GROWTH_EDGE_COUNT_LIMIT)
            ],
        },
        "visual_characterization": "v" * models.TEXT_LIMIT_BY_PATH[
            "visual_characterization"
        ],
    })
    return identity


def maximum_identity_context(
    *,
    include_epistemic_core: bool = False,
) -> dict[str, dict[str, object]]:
    """Build the closed cognition projection from the maximum identity."""

    return project_identity_for_cognition(
        {"effective_identity": maximum_character_identity()},
        include_epistemic_core=include_epistemic_core,
    )


def canonical_service_character_profile(
    *,
    marker: str = "current",
    global_user_id: str = "character-global-id",
) -> dict[str, object]:
    """Build one complete latest-only profile for service boundary tests."""

    return {
        **canonical_character_identity(marker=marker),
        "global_user_id": global_user_id,
        "cognition_state": build_character_production_state(
            updated_at=NOW,
        ),
        "updated_at": NOW,
    }


def canonical_episode_identity_snapshot(
    *,
    marker: str = "current",
    global_user_id: str = "character-global-id",
    revision_number: int = 1,
    include_epistemic_core: bool = False,
) -> dict[str, object]:
    """Build exact latest-identity episode fields for service test seams."""

    identity = canonical_character_identity(marker=marker)
    revision = {
        "effective_identity": identity,
    }
    cognition_context = project_identity_for_cognition(
        revision,
        include_epistemic_core=include_epistemic_core,
    )
    surface_context = project_identity_for_surface(revision)
    return {
        "revision_number": revision_number,
        "character_profile": canonical_service_character_profile(
            marker=marker,
            global_user_id=global_user_id,
        ),
        "cognition_context": cognition_context,
        "surface_context": surface_context,
        "projection_digest": identity_projection_digest(
            revision_number=revision_number,
            cognition_context=cognition_context,
            surface_context=surface_context,
        ),
        "consumer_kinds": projected_identity_consumer_kinds(
            cognition_context,
        ),
    }


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
        "owner": "tests.cognition_test_helpers",
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
    output_mode: str | None = None,
) -> CognitiveEpisodeV1:
    """Build one exact episode across the five native deterministic sources."""

    del output_mode
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
            "owner": "tests.cognition_test_helpers",
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
    state_scope: str = "user",
) -> dict[str, Any]:
    """Build one minimal current cognition output for adjacent tests."""

    if state_scope == "user":
        state = build_acquaintance_user_state(
            global_user_id=owner_user_id,
            updated_at=NOW,
        )
        owner_key = owner_user_id
    elif state_scope == "character":
        state = build_character_production_state(updated_at=NOW)
        owner_key = "character"
    else:
        raise ValueError("canonical cognition test state scope is invalid")
    response_goal = "acknowledge the grounded episode" if route == "speech" else ""
    output: dict[str, Any] = {
        "schema_version": "cognition_output.v3",
        "appraisals": [],
        "active_character_goal": {
            "goal_kind": "ordinary_response",
            "intent": response_goal or "preserve uncertainty",
            "reason": "the current episode establishes the active goal",
            "cause_summary": "the current episode is the grounded cause",
        },
        "relational_willingness": {
            "applicable": False,
            "stance": "not_applicable",
            "reason": "当前回合证据不涉及关系立场判断",
            "cause_summary": "当前回合证据不涉及关系立场判断",
        },
        "private_monologue": (
            "I feel attentive because the current episode needs a grounded "
            "decision, and I want to preserve that judgment."
        ),
        "response_plan": {
            "goal_resolution": "answerable_now" if route == "speech" else "defer",
            "response_goal": response_goal,
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": (
                "Assert only the supplied current-episode facts; keep every "
                "unobserved cause, intent, and outcome uncertain."
            ),
        },
        "affect_projection": [],
        "relationship_projection": {"relationship_summary": "stable", "axis_summaries": {}},
        "cause_provenance": [],
        "diagnostics": {"status": "complete"},
        "state_projection": {
            "state_scope": state_scope,
            "owner_key": owner_key,
            "expected_previous_state": state,
            "original_persisted_state": state,
            "replacement_state": state,
            "transition_contexts": [],
            "binding_receipts": [],
            "capacity_deferred": [],
        },
    }
    return output
