"""Project cognitive episodes into the current RAG request boundary."""

from __future__ import annotations

from typing import Any, TypedDict

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    project_text_chat_compatibility_fields,
    validate_cognitive_episode,
)


class RAGEpisodeAdapterError(ValueError):
    """Raised when an episode cannot be projected into the current RAG path."""


class RAGEpisodeRequest(TypedDict):
    original_query: str
    character_name: str
    context: dict[str, Any]
    current_user_id: str
    character_user_id: str


_SUPPORTED_USER_MESSAGE_INPUT_SOURCE_PROFILES = {
    ("dialog_text",),
    ("dialog_text", "image_observation"),
    ("dialog_text", "audio_observation"),
    ("dialog_text", "image_observation", "audio_observation"),
}
_SUPPORTED_INTERNAL_THOUGHT_INPUT_SOURCE_PROFILES = {
    ("internal_monologue",),
}


def build_text_chat_rag_request(
    *,
    episode: CognitiveEpisode,
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
    validate_cognitive_episode(episode)

    character_user_id, character_name = _character_identity(character_profile)

    if episode["trigger_source"] == "user_message":
        if (
            tuple(episode["input_sources"])
            not in _SUPPORTED_USER_MESSAGE_INPUT_SOURCE_PROFILES
        ):
            raise RAGEpisodeAdapterError("episode input_sources are not supported")

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

    if episode["trigger_source"] == "internal_thought":
        if (
            tuple(episode["input_sources"])
            not in _SUPPORTED_INTERNAL_THOUGHT_INPUT_SOURCE_PROFILES
        ):
            raise RAGEpisodeAdapterError("episode input_sources are not supported")

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
            platform=target_scope["platform"],
            platform_channel_id=target_scope["platform_channel_id"],
            channel_type=target_scope["channel_type"],
            active_turn_platform_message_ids=list(
                origin_metadata["active_turn_platform_message_ids"]
            ),
            active_turn_conversation_row_ids=list(
                origin_metadata["active_turn_conversation_row_ids"]
            ),
            global_user_id=target_scope["current_global_user_id"],
            user_name=target_scope["current_display_name"],
            storage_timestamp_utc=episode["storage_timestamp_utc"],
            local_time_context=episode["local_time_context"],
        )

    raise RAGEpisodeAdapterError("episode trigger_source is not supported")


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
    request: RAGEpisodeRequest = {
        "original_query": decontexualized_input,
        "character_name": character_name,
        "context": context,
        "current_user_id": global_user_id,
        "character_user_id": character_user_id,
    }
    return request
