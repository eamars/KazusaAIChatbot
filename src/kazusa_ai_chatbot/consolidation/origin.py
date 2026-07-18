"""Projection contract for consolidation origin metadata."""

from __future__ import annotations

from typing import TypedDict

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeV1,
    TriggerSource,
)


class ConsolidationOriginError(ValueError):
    """Raised when an episode cannot enter current consolidation."""


class ConsolidationOriginMetadata(TypedDict):
    episode_id: str
    trigger_source: TriggerSource
    input_sources: list[str]
    output_mode: str
    storage_timestamp_utc: str
    platform: str
    platform_channel_id: str
    channel_type: str
    platform_message_id: str
    active_turn_platform_message_ids: list[str]
    active_turn_conversation_row_ids: list[str]
    current_platform_user_id: str
    current_global_user_id: str
    current_display_name: str


class ConsolidationOriginPromptBlock(TypedDict):
    episode_id: str
    trigger_source: TriggerSource
    input_sources: list[str]
    output_mode: str


def build_user_message_consolidation_origin(
    *,
    episode: CognitiveEpisodeV1,
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
    if episode["trigger_source"] != "user_message":
        raise ConsolidationOriginError(
            "consolidation origin requires trigger_source=user_message"
        )
    metadata = _project_consolidation_origin_metadata(episode)
    return metadata


def build_self_cognition_consolidation_origin(
    *,
    episode: CognitiveEpisodeV1,
) -> ConsolidationOriginMetadata:
    """Project internal-thought episode identifiers into consolidation state.

    Args:
        episode: Source-neutral cognitive episode for a self-cognition run.

    Returns:
        Identifier-only origin metadata for consolidation nodes.

    Raises:
        CognitiveEpisodeValidationError: If the episode is structurally
            invalid.
        ConsolidationOriginError: If the episode is not supported by current
            self-cognition consolidation.
    """
    if episode["trigger_source"] not in {
        "internal_thought",
        "self_cognition",
        "scheduled_tick",
    }:
        raise ConsolidationOriginError(
            "consolidation origin requires a self-cognition source"
        )

    metadata = _project_consolidation_origin_metadata(episode)
    return metadata


def build_tool_result_consolidation_origin(
    *,
    episode: CognitiveEpisodeV1,
) -> ConsolidationOriginMetadata:
    """Project tool-result episode identifiers into consolidation state.

    Args:
        episode: Source-neutral cognitive episode for a private self-cognition run.

    Returns:
        Identifier-only origin metadata for tool-result admission.

    Raises:
        CognitiveEpisodeValidationError: If the episode is structurally
            invalid.
        ConsolidationOriginError: If the episode is not a tool-result source.
    """
    if episode["trigger_source"] != "tool_result":
        raise ConsolidationOriginError(
            "consolidation origin requires trigger_source=tool_result"
        )

    metadata = _project_consolidation_origin_metadata(episode)
    return metadata


def project_consolidation_origin_prompt_block(
    origin: ConsolidationOriginMetadata,
) -> ConsolidationOriginPromptBlock:
    """Project origin metadata into model-facing source identity.

    Args:
        origin: Identifier-only consolidation origin metadata.

    Returns:
        Compact origin fields needed by shared consolidator prompts.
    """
    block: ConsolidationOriginPromptBlock = {
        "episode_id": origin["episode_id"],
        "trigger_source": origin["trigger_source"],
        "input_sources": list(origin["input_sources"]),
        "output_mode": origin["output_mode"],
    }
    return block


def _project_consolidation_origin_metadata(
    episode: CognitiveEpisodeV1,
) -> ConsolidationOriginMetadata:
    """Project shared identifier fields from a validated cognitive episode.

    Args:
        episode: Valid cognitive episode accepted by a consolidation origin
            builder.

    Returns:
        Identifier-only origin metadata without percept or prompt content.
    """
    target_scope = episode["target_scope"]
    origin_metadata = episode["origin_metadata"]
    percept_sources = [
        str(percept.get("source_kind", ""))
        for percept in episode["percepts"]
        if percept.get("source_kind")
    ]
    input_sources = list(dict.fromkeys(percept_sources))
    output_mode = (
        "visible_reply"
        if episode["trigger_source"] in {"user_message", "tool_result"}
        else "preview"
    )
    metadata: ConsolidationOriginMetadata = {
        "episode_id": episode["episode_id"],
        "trigger_source": episode["trigger_source"],
        "input_sources": input_sources,
        "output_mode": output_mode,
        "storage_timestamp_utc": episode["created_at"],
        "platform": str(target_scope.get("platform", "")),
        "platform_channel_id": str(target_scope.get("platform_channel_id", "")),
        "channel_type": str(target_scope.get("channel_type", "")),
        "platform_message_id": str(
            origin_metadata.get("platform_message_id", "")
        ),
        "active_turn_platform_message_ids": list(
            origin_metadata.get("active_turn_platform_message_ids", [])
        ),
        "active_turn_conversation_row_ids": list(
            origin_metadata.get("active_turn_conversation_row_ids", [])
        ),
        "current_platform_user_id": str(
            target_scope.get("current_platform_user_id", "")
        ),
        "current_global_user_id": str(
            target_scope.get("current_global_user_id", "")
        ),
        "current_display_name": str(
            target_scope.get("current_display_name", "")
        ),
    }
    return metadata
