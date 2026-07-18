"""Project cognitive episodes into the current RAG request boundary."""

from __future__ import annotations

from typing import Any, TypedDict

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeV1,
    project_text_chat_compatibility_fields,
    validate_cognitive_episode_v1,
)


class RAGEpisodeAdapterError(ValueError):
    """Raised when an episode cannot be projected into the current RAG path."""


class RAGEpisodeRequest(TypedDict):
    original_query: str
    character_name: str
    context: dict[str, Any]
    current_user_id: str
    character_user_id: str


_EXPECTED_SOURCE_PROFILES = {
    "user_message": frozenset({
        ("dialog", "system_event"),
        ("dialog", "image_observation", "system_event"),
        ("dialog", "audio_observation", "system_event"),
        (
            "dialog",
            "image_observation",
            "audio_observation",
            "system_event",
        ),
    }),
    "internal_thought": frozenset({
        ("internal_thought", "system_event"),
    }),
    "self_cognition": frozenset({
        ("self_cognition", "system_event"),
    }),
    "scheduled_tick": frozenset({
        ("scheduled_event", "system_event"),
    }),
    "tool_result": frozenset({
        ("tool_result", "system_event"),
    }),
}


def build_text_chat_rag_request(
    *,
    episode: CognitiveEpisodeV1,
    decontexualized_input: str,
    character_profile: dict[str, Any],
    user_profile: dict[str, Any],
    prompt_message_context: dict[str, Any],
    channel_topic: str,
    chat_history_recent: list[dict[str, Any]],
    chat_history_wide: list[dict[str, Any]],
    reply_context: dict[str, Any],
    indirect_speech_context: str,
    conversation_progress: dict[str, Any] | None = None,
    conversation_episode_state: dict[str, Any] | None = None,
    promoted_reflection_context: dict[str, Any] | None = None,
    llm_trace_id: str = "",
) -> RAGEpisodeRequest:
    """Build the current text-chat RAG request from a cognitive episode.

    Args:
        episode: Valid cognitive episode for the active text chat turn.
        decontexualized_input: Query string already prepared for RAG.
        character_profile: Active character identity fields.
        user_profile: Current user's profile payload.
        prompt_message_context: Safe message context already built upstream.
        channel_topic: Channel topic text supplied to RAG.
        chat_history_recent: Recent conversation history.
        chat_history_wide: Wider conversation history.
        reply_context: Current reply-thread context.
        indirect_speech_context: Rendered indirect-speech context.
        conversation_progress: Optional current conversation progress payload.
        conversation_episode_state: Optional conversation episode state payload.
        promoted_reflection_context: Optional promoted reflection context.

    Returns:
        RAG supervisor request fields and the projected user ids.

    Raises:
        RAGEpisodeAdapterError: If the episode or character identity cannot
            enter the current text-chat RAG path.
    """
    validate_cognitive_episode_v1(episode)
    _validate_source_profile(episode)

    character_user_id, character_name = _character_identity(character_profile)

    if episode["trigger_source"] == "user_message":
        projection = project_text_chat_compatibility_fields(episode)
        return _build_rag_request_from_fields(
            decontexualized_input=decontexualized_input,
            character_user_id=character_user_id,
            character_name=character_name,
            user_profile=user_profile,
            prompt_message_context=prompt_message_context,
            channel_topic=channel_topic,
            chat_history_recent=chat_history_recent,
            chat_history_wide=chat_history_wide,
            reply_context=reply_context,
            indirect_speech_context=indirect_speech_context,
            conversation_progress=conversation_progress,
            conversation_episode_state=conversation_episode_state,
            promoted_reflection_context=promoted_reflection_context,
            llm_trace_id=llm_trace_id,
            platform=projection["platform"],
            platform_channel_id=projection["platform_channel_id"],
            channel_type=projection["channel_type"],
            active_turn_platform_message_ids=projection[
                "active_turn_platform_message_ids"
            ],
            active_turn_conversation_row_ids=projection[
                "active_turn_conversation_row_ids"
            ],
            global_user_id=projection["global_user_id"],
            user_name=projection["user_name"],
            storage_timestamp_utc=projection["storage_timestamp_utc"],
            local_time_context=projection["local_time_context"],
        )

    if episode["trigger_source"] not in {
        "internal_thought",
        "self_cognition",
        "scheduled_tick",
        "tool_result",
    }:
        raise RAGEpisodeAdapterError("episode trigger_source is not supported")

    target_scope = episode["target_scope"]
    origin_metadata = episode["origin_metadata"]
    return _build_rag_request_from_fields(
        decontexualized_input=decontexualized_input,
        character_user_id=character_user_id,
        character_name=character_name,
        user_profile=user_profile,
        prompt_message_context=prompt_message_context,
        channel_topic=channel_topic,
        chat_history_recent=chat_history_recent,
        chat_history_wide=chat_history_wide,
        reply_context=reply_context,
        indirect_speech_context=indirect_speech_context,
        conversation_progress=conversation_progress,
        conversation_episode_state=conversation_episode_state,
        promoted_reflection_context=promoted_reflection_context,
        llm_trace_id=llm_trace_id,
        platform=_required_scope_string(target_scope, "platform"),
        platform_channel_id=_required_scope_string(
            target_scope,
            "platform_channel_id",
        ),
        channel_type=_required_scope_string(target_scope, "channel_type"),
        active_turn_platform_message_ids=_string_list(
            origin_metadata.get("active_turn_platform_message_ids", [])
        ),
        active_turn_conversation_row_ids=_string_list(
            origin_metadata.get("active_turn_conversation_row_ids", [])
        ),
        global_user_id=_required_scope_string(
            target_scope,
            "current_global_user_id",
        ),
        user_name=_required_scope_string(
            target_scope,
            "current_display_name",
        ),
        storage_timestamp_utc=episode["created_at"],
        local_time_context=_canonical_local_time_context(episode),
    )


def _validate_source_profile(episode: CognitiveEpisodeV1) -> None:
    """Require the source-specific percept profile at the RAG boundary."""

    trigger_source = episode["trigger_source"]
    expected_profiles = _EXPECTED_SOURCE_PROFILES[trigger_source]
    source_profile = tuple(
        str(percept["source_kind"])
        for percept in episode["percepts"]
    )
    if source_profile not in expected_profiles:
        raise RAGEpisodeAdapterError(
            f"episode source profile is invalid for {trigger_source}"
        )


def _required_scope_string(scope: dict[str, object], field_name: str) -> str:
    """Read one required canonical target-scope string."""

    value = scope.get(field_name)
    if not isinstance(value, str) or not value:
        raise RAGEpisodeAdapterError(
            f"episode target_scope.{field_name} is required"
        )
    return value


def _string_list(value: object) -> list[str]:
    """Return a validated list of string correlation identifiers."""

    if not isinstance(value, list):
        raise RAGEpisodeAdapterError(
            "episode origin metadata correlation identifiers must be lists"
        )
    if not all(isinstance(item, str) for item in value):
        raise RAGEpisodeAdapterError(
            "episode origin metadata correlation identifiers must be strings"
        )
    return list(value)


def _canonical_local_time_context(
    episode: CognitiveEpisodeV1,
) -> dict[str, Any]:
    """Extract the bounded local-time percept for downstream RAG context."""

    for percept in episode["percepts"]:
        if percept["percept_kind"] != "local_time_context":
            continue
        value = percept["content"].get("local_time_context")
        if isinstance(value, dict):
            return dict(value)
    raise RAGEpisodeAdapterError(
        "canonical episode is missing local_time_context"
    )


def _character_identity(character_profile: dict[str, Any]) -> tuple[str, str]:
    if "global_user_id" not in character_profile:
        raise RAGEpisodeAdapterError("character_profile.global_user_id is required")
    character_user_id = character_profile["global_user_id"]
    if not isinstance(character_user_id, str) or character_user_id == "":
        raise RAGEpisodeAdapterError(
            "character_profile.global_user_id must be a non-empty string"
        )

    if "name" not in character_profile:
        raise RAGEpisodeAdapterError("character_profile.name is required")
    character_name = character_profile["name"]
    if not isinstance(character_name, str) or character_name == "":
        raise RAGEpisodeAdapterError(
            "character_profile.name must be a non-empty string"
        )
    return character_user_id, character_name


def _build_rag_request_from_fields(
    *,
    decontexualized_input: str,
    character_user_id: str,
    character_name: str,
    user_profile: dict[str, Any],
    prompt_message_context: dict[str, Any],
    channel_topic: str,
    chat_history_recent: list[dict[str, Any]],
    chat_history_wide: list[dict[str, Any]],
    reply_context: dict[str, Any],
    indirect_speech_context: str,
    conversation_progress: dict[str, Any] | None,
    conversation_episode_state: dict[str, Any] | None,
    promoted_reflection_context: dict[str, Any] | None,
    llm_trace_id: str,
    platform: str,
    platform_channel_id: str,
    channel_type: str,
    active_turn_platform_message_ids: list[str],
    active_turn_conversation_row_ids: list[str],
    global_user_id: str,
    user_name: str,
    storage_timestamp_utc: str,
    local_time_context: dict[str, Any],
) -> RAGEpisodeRequest:
    context: dict[str, Any] = {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "character_profile": {
            "global_user_id": character_user_id,
            "name": character_name,
        },
        "active_turn_platform_message_ids": active_turn_platform_message_ids,
        "active_turn_conversation_row_ids": active_turn_conversation_row_ids,
        "global_user_id": global_user_id,
        "user_name": user_name,
        "user_profile": user_profile,
        "current_timestamp_utc": storage_timestamp_utc,
        "local_time_context": local_time_context,
        "prompt_message_context": prompt_message_context,
        "channel_topic": channel_topic,
        "chat_history_recent": chat_history_recent,
        "chat_history_wide": chat_history_wide,
        "reply_context": reply_context,
        "indirect_speech_context": indirect_speech_context,
        "conversation_progress": conversation_progress,
        "conversation_episode_state": conversation_episode_state,
        "promoted_reflection_context": promoted_reflection_context,
    }
    if llm_trace_id:
        context["llm_trace_id"] = llm_trace_id
    request: RAGEpisodeRequest = {
        "original_query": decontexualized_input,
        "character_name": character_name,
        "context": context,
        "current_user_id": global_user_id,
        "character_user_id": character_user_id,
    }
    return request
