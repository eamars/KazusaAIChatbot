"""Projection contract for consolidation origin metadata."""

from __future__ import annotations

from typing import TypedDict

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    InputSource,
    OutputMode,
    TriggerSource,
    validate_cognitive_episode,
)


class ConsolidationOriginError(ValueError):
    """Raised when an episode cannot enter current consolidation."""


class ConsolidationOriginMetadata(TypedDict):
    episode_id: str
    trigger_source: TriggerSource
    input_sources: list[InputSource]
    output_mode: OutputMode
    timestamp: str
    platform: str
    platform_channel_id: str
    channel_type: str
    platform_message_id: str
    active_turn_platform_message_ids: list[str]
    active_turn_conversation_row_ids: list[str]
    current_platform_user_id: str
    current_global_user_id: str
    current_display_name: str


_SUPPORTED_USER_MESSAGE_INPUT_SOURCE_PROFILES = {
    ("dialog_text",),
    ("dialog_text", "image_observation"),
    ("dialog_text", "audio_observation"),
    ("dialog_text", "image_observation", "audio_observation"),
}


def build_user_message_consolidation_origin(
    *,
    episode: CognitiveEpisode,
) -> ConsolidationOriginMetadata:
    """Project current text-chat episode identifiers into consolidation state.

    Args:
        episode: Source-neutral cognitive episode for the current chat turn.

    Returns:
        Identifier-only origin metadata for consolidation nodes.

    Raises:
        CognitiveEpisodeValidationError: If the episode is structurally
            invalid.
        ConsolidationOriginError: If the episode is not supported by current
            text-chat consolidation.
    """
    validate_cognitive_episode(episode)

    if episode["trigger_source"] != "user_message":
        raise ConsolidationOriginError(
            "consolidation origin requires trigger_source=user_message"
        )
    input_source_profile = tuple(episode["input_sources"])
    if input_source_profile not in _SUPPORTED_USER_MESSAGE_INPUT_SOURCE_PROFILES:
        raise ConsolidationOriginError(
            "consolidation origin requires supported user-message input_sources"
        )
    if episode["output_mode"] not in {"visible_reply", "think_only", "silent"}:
        raise ConsolidationOriginError(
            "consolidation origin requires a chat-compatible output_mode"
        )

    target_scope = episode["target_scope"]
    origin_metadata = episode["origin_metadata"]
    metadata: ConsolidationOriginMetadata = {
        "episode_id": episode["episode_id"],
        "trigger_source": episode["trigger_source"],
        "input_sources": list(episode["input_sources"]),
        "output_mode": episode["output_mode"],
        "timestamp": episode["timestamp"],
        "platform": target_scope["platform"],
        "platform_channel_id": target_scope["platform_channel_id"],
        "channel_type": target_scope["channel_type"],
        "platform_message_id": origin_metadata["platform_message_id"],
        "active_turn_platform_message_ids": list(
            origin_metadata["active_turn_platform_message_ids"]
        ),
        "active_turn_conversation_row_ids": list(
            origin_metadata["active_turn_conversation_row_ids"]
        ),
        "current_platform_user_id": target_scope["current_platform_user_id"],
        "current_global_user_id": target_scope["current_global_user_id"],
        "current_display_name": target_scope["current_display_name"],
    }
    return metadata
