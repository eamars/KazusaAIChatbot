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

    if episode["trigger_source"] != "user_message":
        raise RAGEpisodeAdapterError("episode trigger_source is not supported")

    if episode["input_sources"] != ["dialog_text"]:
        raise RAGEpisodeAdapterError("episode input_sources are not supported")

    projection = project_text_chat_compatibility_fields(episode)

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

    context: dict[str, Any] = {
        "platform": projection["platform"],
        "platform_channel_id": projection["platform_channel_id"],
        "channel_type": projection["channel_type"],
        "character_profile": {
            "global_user_id": character_user_id,
            "name": character_name,
        },
        "active_turn_platform_message_ids": projection[
            "active_turn_platform_message_ids"
        ],
        "active_turn_conversation_row_ids": projection[
            "active_turn_conversation_row_ids"
        ],
        "global_user_id": projection["global_user_id"],
        "user_name": projection["user_name"],
        "user_profile": user_profile,
        "current_timestamp": projection["timestamp"],
        "time_context": projection["time_context"],
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
        "current_user_id": projection["global_user_id"],
        "character_user_id": character_user_id,
    }
    return request
