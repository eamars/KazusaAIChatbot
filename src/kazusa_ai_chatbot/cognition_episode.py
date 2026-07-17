"""Source-neutral cognitive episode contracts and builders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NoReturn, NotRequired, TypedDict, get_args

from kazusa_ai_chatbot.time_boundary import LocalTimeContextDoc

TriggerSource = Literal[
    "user_message",
    "reflection_signal",
    "internal_thought",
    "scheduled_recall",
    "system_probe",
    "accepted_task_result_ready",
]

InputSource = Literal[
    "dialog_text",
    "image_observation",
    "audio_observation",
    "internal_monologue",
    "reflection_artifact",
    "retrieved_memory",
    "accepted_task_result",
]

Visibility = Literal[
    "model_visible",
    "internal_only",
    "audit_only",
]

OutputMode = Literal[
    "visible_reply",
    "silent",
    "think_only",
    "preview",
    "scheduled_action_request",
]


class CognitivePercept(TypedDict):
    percept_id: str
    input_source: InputSource
    content: str
    visibility: Visibility
    metadata: dict[str, Any]


class TargetScope(TypedDict):
    platform: str
    platform_channel_id: str
    channel_type: str
    current_platform_user_id: str
    current_global_user_id: str
    current_display_name: str
    target_addressed_user_ids: list[str]
    target_broadcast: bool


class OriginMetadata(TypedDict):
    platform: str
    platform_message_id: str
    active_turn_platform_message_ids: list[str]
    active_turn_conversation_row_ids: list[str]
    debug_modes: dict[str, bool]


class MediaDescriptionRow(TypedDict):
    content_type: str
    description: str
    image_observation: NotRequired["ImageObservation"]


class ImageObservation(TypedDict):
    observation_origin: str
    source_message_id: str
    media_kind: str
    summary_status: Literal["available", "unavailable"]
    summary: str
    visible_text: list[str]
    salient_visual_facts: list[str]
    spatial_or_scene_facts: list[str]
    uncertainty: list[str]


class CognitiveEpisode(TypedDict):
    episode_id: str
    trigger_source: TriggerSource
    input_sources: list[InputSource]
    output_mode: OutputMode
    percepts: list[CognitivePercept]
    target_scope: TargetScope
    origin_metadata: OriginMetadata
    storage_timestamp_utc: str
    local_time_context: LocalTimeContextDoc


class TextChatCompatibilityProjection(TypedDict):
    storage_timestamp_utc: str
    local_time_context: LocalTimeContextDoc
    user_input: str
    platform: str
    platform_channel_id: str
    channel_type: str
    platform_message_id: str
    active_turn_platform_message_ids: list[str]
    active_turn_conversation_row_ids: list[str]
    platform_user_id: str
    global_user_id: str
    user_name: str


class CognitiveEpisodeValidationError(ValueError):
    """Raised when a cognitive episode is structurally invalid."""


_TRIGGER_SOURCES = frozenset(get_args(TriggerSource))
_INPUT_SOURCES = frozenset(get_args(InputSource))
_VISIBILITIES = frozenset(get_args(Visibility))
_OUTPUT_MODES = frozenset(get_args(OutputMode))
_LOCAL_TIME_CONTEXT_FIELDS = tuple(LocalTimeContextDoc.__annotations__)
MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS = 4
MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS = 800
_IMAGE_OBSERVATION_LIST_FIELDS = (
    "visible_text",
    "salient_visual_facts",
    "spatial_or_scene_facts",
    "uncertainty",
)


def project_model_visible_percepts(
    episode: CognitiveEpisode,
) -> list[dict[str, str]]:
    """Project visible percepts with typed dialogue-role provenance."""

    validate_cognitive_episode(episode)
    user_dialog = (
        episode["trigger_source"] == "user_message"
        and "dialog_text" in episode["input_sources"]
    )
    rows: list[dict[str, str]] = []
    for percept in episode["percepts"]:
        if percept["visibility"] != "model_visible":
            continue
        row = {
            "input_source": percept["input_source"],
            "content": percept["content"],
        }
        if user_dialog and percept["input_source"] == "dialog_text":
            row.update({
                "speaker_role": "current_user",
                "addressee_role": "self",
                "first_person_role": "current_user",
                "implicit_imperative_subject_role": "self",
            })
        rows.append(row)
    return rows


def build_text_chat_media_description_rows(
    multimedia_input: list[Mapping[str, object]],
) -> list[MediaDescriptionRow]:
    """Project current media rows into prompt-safe episode description rows.

    Args:
        multimedia_input: Current-turn media rows that may include storage or
            adapter fields outside the episode contract.

    Returns:
        Rows containing only supported content types and stripped descriptions.
    """
    media_description_rows: list[MediaDescriptionRow] = []
    for item in multimedia_input:
        content_type = item.get("content_type")
        if not isinstance(content_type, str) or content_type == "":
            continue

        description = item.get("description")
        if not isinstance(description, str):
            continue

        if not (
            content_type.startswith("image/")
            or content_type.startswith("audio/")
        ):
            continue

        clean_description = description.strip()
        image_observation = _sanitize_image_observation(
            item.get("image_observation"),
            description=clean_description,
            content_type=content_type,
        )
        if clean_description == "" and image_observation is None:
            continue

        row: MediaDescriptionRow = {
            "content_type": content_type,
            "description": clean_description,
        }
        if image_observation is not None:
            row["image_observation"] = image_observation
        media_description_rows.append(row)

    return media_description_rows


def build_reply_media_description_rows(
    reply_context: Mapping[str, object] | None,
) -> list[MediaDescriptionRow]:
    """Project quoted-reply image summaries into episode media rows.

    Args:
        reply_context: Service-facing reply context that may include stored
            prompt-safe attachment summaries for the replied-to message.

    Returns:
        Image media rows that preserve quoted-image availability without
        requiring raw media bytes.
    """
    if not isinstance(reply_context, Mapping):
        return_value: list[MediaDescriptionRow] = []
        return return_value

    attachments = reply_context.get("reply_attachments")
    if not isinstance(attachments, list):
        return_value = []
        return return_value

    source_message_id = _string_field(reply_context, "reply_to_message_id")
    rows: list[MediaDescriptionRow] = []
    for attachment in attachments:
        if not isinstance(attachment, Mapping):
            continue
        media_kind = _string_field(attachment, "media_kind").casefold()
        if media_kind != "image":
            continue

        description = _string_field(attachment, "description")
        summary_status = _summary_status(attachment, description)
        observation: ImageObservation = {
            "observation_origin": "quoted_reply_attachment",
            "source_message_id": source_message_id,
            "media_kind": "image",
            "summary_status": summary_status,
            "summary": description,
            "visible_text": [],
            "salient_visual_facts": [],
            "spatial_or_scene_facts": [],
            "uncertainty": [],
        }
        row: MediaDescriptionRow = {
            "content_type": "image/quoted-reply",
            "description": description,
            "image_observation": observation,
        }
        rows.append(row)

    return_value = build_text_chat_media_description_rows(rows)
    return return_value


def _sanitize_image_observation(
    value: object,
    *,
    description: str,
    content_type: str,
) -> ImageObservation | None:
    """Build a bounded image observation from structured or legacy input.

    Args:
        value: Optional structured image observation supplied by upstream.
        description: Prompt-safe image summary retained for compatibility.
        content_type: Current media content type.

    Returns:
        Bounded image observation, or ``None`` for non-image media without a
        structured visual observation.
    """
    if not content_type.startswith("image/"):
        return_value = None
        return return_value

    observation_data: Mapping[str, object] = {}
    if isinstance(value, Mapping):
        observation_data = value

    summary = _trim_media_description(
        _string_field(observation_data, "summary") or description
    )
    summary_status = _summary_status(observation_data, summary)
    if description == "" and not observation_data:
        return_value = None
        return return_value

    list_fields = {
        field_name: _string_list_field(observation_data, field_name)
        for field_name in _IMAGE_OBSERVATION_LIST_FIELDS
    }
    observation: ImageObservation = {
        "observation_origin": (
            _string_field(observation_data, "observation_origin")
            or "current_attachment"
        ),
        "source_message_id": _string_field(
            observation_data,
            "source_message_id",
        ),
        "media_kind": "image",
        "summary_status": summary_status,
        "summary": summary,
        "visible_text": list_fields["visible_text"],
        "salient_visual_facts": list_fields["salient_visual_facts"],
        "spatial_or_scene_facts": list_fields["spatial_or_scene_facts"],
        "uncertainty": list_fields["uncertainty"],
    }
    return observation


def _string_field(data: Mapping[str, object], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = _trim_media_description(value.strip())
    return return_value


def _string_list_field(
    data: Mapping[str, object],
    field_name: str,
) -> list[str]:
    value = data.get(field_name)
    if not isinstance(value, list):
        return_value: list[str] = []
        return return_value

    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        clean_item = _trim_media_description(item.strip())
        if clean_item:
            strings.append(clean_item)
    return strings


def _summary_status(
    data: Mapping[str, object],
    summary: str,
) -> Literal["available", "unavailable"]:
    value = data.get("summary_status")
    if value == "available" or value == "unavailable":
        return_value = value
    elif summary:
        return_value = "available"
    else:
        return_value = "unavailable"
    return return_value


def validate_cognitive_episode(episode: CognitiveEpisode) -> None:
    """Validate a cognitive episode's structure without interpreting content.

    Args:
        episode: Episode payload to validate.

    Raises:
        CognitiveEpisodeValidationError: If any structural requirement is
            missing or has the wrong type.
    """
    if not isinstance(episode, dict):
        _raise_validation_error("episode must be a dict")

    episode_data: dict[str, Any] = episode

    _require_non_empty_string(episode_data, "episode_id", "episode")
    trigger_source = _require_literal(
        episode_data,
        "trigger_source",
        "episode",
        _TRIGGER_SOURCES,
    )
    input_sources = _require_literal_list(
        episode_data,
        "input_sources",
        "episode",
        _INPUT_SOURCES,
    )
    _require_literal(episode_data, "output_mode", "episode", _OUTPUT_MODES)
    percepts = _require_non_empty_list(episode_data, "percepts", "episode")
    target_scope = _require_dict(episode_data, "target_scope", "episode")
    _validate_target_scope(target_scope)
    origin_metadata = _require_dict(episode_data, "origin_metadata", "episode")
    _validate_origin_metadata(origin_metadata)
    if "timestamp" in episode_data or "time_context" in episode_data:
        _raise_validation_error(
            "episode contains legacy timestamp or time_context field"
        )
    _require_non_empty_string(
        episode_data,
        "storage_timestamp_utc",
        "episode",
    )
    local_time_context = _require_dict(
        episode_data,
        "local_time_context",
        "episode",
    )
    _validate_local_time_context(local_time_context)

    percept_input_sources = _validate_percepts(percepts)
    _validate_percept_sources_match_input_sources(
        input_sources,
        percept_input_sources,
    )
    if trigger_source == "user_message":
        _validate_user_message_has_dialog_text(percept_input_sources)


def build_text_chat_cognitive_episode(
    *,
    episode_id: str,
    percept_id: str,
    storage_timestamp_utc: str,
    local_time_context: LocalTimeContextDoc,
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
    output_mode: OutputMode = "visible_reply",
    target_addressed_user_ids: list[str] | None = None,
    target_broadcast: bool = False,
    media_description_rows: list[MediaDescriptionRow] | None = None,
) -> CognitiveEpisode:
    """Build a source-neutral episode for the current text `/chat` turn.

    Args:
        episode_id: Stable id for this cognitive episode.
        percept_id: Stable id for the single text percept.
        storage_timestamp_utc: Storage UTC timestamp for the turn.
        local_time_context: Configured-local time context for model payloads.
        user_input: Current text input seen by the chat path.
        platform: Adapter platform name.
        platform_channel_id: Platform channel or private conversation id.
        channel_type: Channel type such as private, group, or system.
        platform_message_id: Original platform message id when available.
        platform_user_id: Platform-local id for the current user.
        global_user_id: Internal global id for the current user.
        user_name: Display name for the current user.
        active_turn_platform_message_ids: Platform message ids collapsed into
            the active turn.
        active_turn_conversation_row_ids: Conversation row ids collapsed into
            the active turn.
        debug_modes: Active debug-mode flags for this turn.
        output_mode: Allowed output mode for the episode.
        target_addressed_user_ids: Current explicit addressees.
        target_broadcast: Whether the current turn targets the channel broadly.
        media_description_rows: Optional bounded image/audio descriptions for
            the same user-message turn.

    Returns:
        A validated `CognitiveEpisode` for a text chat turn.

    Raises:
        CognitiveEpisodeValidationError: If the primitive fields produce a
            structurally invalid text chat episode.
    """
    platform_message_ids = list(active_turn_platform_message_ids or [])
    conversation_row_ids = list(active_turn_conversation_row_ids or [])
    active_debug_modes = dict(debug_modes or {})
    addressed_user_ids = list(target_addressed_user_ids or [])

    percept: CognitivePercept = {
        "percept_id": percept_id,
        "input_source": "dialog_text",
        "content": user_input,
        "visibility": "model_visible",
        "metadata": {},
    }
    media_input_sources, media_percepts = _build_media_percepts(
        base_percept_id=percept_id,
        media_description_rows=media_description_rows,
    )
    target_scope: TargetScope = {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "current_platform_user_id": platform_user_id,
        "current_global_user_id": global_user_id,
        "current_display_name": user_name,
        "target_addressed_user_ids": addressed_user_ids,
        "target_broadcast": target_broadcast,
    }
    origin_metadata: OriginMetadata = {
        "platform": platform,
        "platform_message_id": platform_message_id,
        "active_turn_platform_message_ids": platform_message_ids,
        "active_turn_conversation_row_ids": conversation_row_ids,
        "debug_modes": active_debug_modes,
    }
    episode: CognitiveEpisode = {
        "episode_id": episode_id,
        "trigger_source": "user_message",
        "input_sources": ["dialog_text", *media_input_sources],
        "output_mode": output_mode,
        "percepts": [percept, *media_percepts],
        "target_scope": target_scope,
        "origin_metadata": origin_metadata,
        "storage_timestamp_utc": storage_timestamp_utc,
        "local_time_context": local_time_context,
    }

    validate_cognitive_episode(episode)
    return episode


def build_accepted_task_result_ready_cognitive_episode(
    *,
    episode_id: str,
    percept_id: str,
    storage_timestamp_utc: str,
    local_time_context: LocalTimeContextDoc,
    accepted_task_id: str,
    accepted_task_summary: str,
    artifact_text: str,
    failure_summary: str,
    result_summary: str,
    platform: str,
    platform_channel_id: str,
    channel_type: str,
    platform_message_id: str,
    requester_platform_user_id: str,
    requester_global_user_id: str,
    requester_display_name: str,
    source_platform_bot_id: str,
    source_character_name: str,
    coding_run_context: dict[str, object] | None = None,
) -> CognitiveEpisode:
    """Build a source-bound episode for a completed accepted task."""

    metadata = {
        "accepted_task_id": accepted_task_id,
        "accepted_task_summary": accepted_task_summary,
        "failure_summary": failure_summary,
        "result_summary": result_summary,
        "source_platform_bot_id": source_platform_bot_id,
        "source_character_name": source_character_name,
    }
    if coding_run_context is not None:
        metadata["coding_run_context"] = coding_run_context
    percept: CognitivePercept = {
        "percept_id": percept_id,
        "input_source": "accepted_task_result",
        "content": artifact_text or failure_summary,
        "visibility": "model_visible",
        "metadata": metadata,
    }
    target_scope: TargetScope = {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "current_platform_user_id": requester_platform_user_id,
        "current_global_user_id": requester_global_user_id,
        "current_display_name": requester_display_name,
        "target_addressed_user_ids": (
            [requester_global_user_id] if requester_global_user_id else []
        ),
        "target_broadcast": False,
    }
    origin_metadata: OriginMetadata = {
        "platform": platform,
        "platform_message_id": platform_message_id,
        "active_turn_platform_message_ids": [platform_message_id]
        if platform_message_id
        else [],
        "active_turn_conversation_row_ids": [],
        "debug_modes": {},
    }
    episode: CognitiveEpisode = {
        "episode_id": episode_id,
        "trigger_source": "accepted_task_result_ready",
        "input_sources": ["accepted_task_result"],
        "output_mode": "visible_reply",
        "percepts": [percept],
        "target_scope": target_scope,
        "origin_metadata": origin_metadata,
        "storage_timestamp_utc": storage_timestamp_utc,
        "local_time_context": local_time_context,
    }
    validate_cognitive_episode(episode)
    return episode


def replace_text_chat_media_percepts(
    *,
    episode: CognitiveEpisode,
    media_description_rows: list[MediaDescriptionRow] | None,
) -> CognitiveEpisode:
    """Return a user-message episode with refreshed media-description percepts.

    Args:
        episode: Existing text-chat episode to refresh after media descriptions
            become available.
        media_description_rows: Optional bounded image/audio descriptions for
            the same user-message turn.

    Returns:
        New validated episode preserving the text-chat fields and replacing
        only image/audio media percepts.

    Raises:
        CognitiveEpisodeValidationError: If the input episode is not a valid
            user-message episode with exactly one dialog-text percept.
    """
    validate_cognitive_episode(episode)

    if episode["trigger_source"] != "user_message":
        _raise_validation_error("episode must be a user_message episode")

    dialog_percept = _single_dialog_text_percept(episode)
    target_scope = episode["target_scope"]
    origin_metadata = episode["origin_metadata"]
    refreshed_episode = build_text_chat_cognitive_episode(
        episode_id=episode["episode_id"],
        percept_id=dialog_percept["percept_id"],
        storage_timestamp_utc=episode["storage_timestamp_utc"],
        local_time_context=episode["local_time_context"],
        user_input=dialog_percept["content"],
        platform=target_scope["platform"],
        platform_channel_id=target_scope["platform_channel_id"],
        channel_type=target_scope["channel_type"],
        platform_message_id=origin_metadata["platform_message_id"],
        platform_user_id=target_scope["current_platform_user_id"],
        global_user_id=target_scope["current_global_user_id"],
        user_name=target_scope["current_display_name"],
        active_turn_platform_message_ids=origin_metadata[
            "active_turn_platform_message_ids"
        ],
        active_turn_conversation_row_ids=origin_metadata[
            "active_turn_conversation_row_ids"
        ],
        debug_modes=origin_metadata["debug_modes"],
        output_mode=episode["output_mode"],
        target_addressed_user_ids=target_scope["target_addressed_user_ids"],
        target_broadcast=target_scope["target_broadcast"],
        media_description_rows=media_description_rows,
    )
    return refreshed_episode


def project_text_chat_compatibility_fields(
    episode: CognitiveEpisode,
) -> TextChatCompatibilityProjection:
    """Project an episode into the current text `/chat` primitive fields.

    Args:
        episode: Valid text chat episode to project.

    Returns:
        Primitive fields consumed by the current text chat path.

    Raises:
        CognitiveEpisodeValidationError: If the episode is not a valid text
            chat episode.
    """
    validate_cognitive_episode(episode)

    if episode["trigger_source"] != "user_message":
        _raise_validation_error("episode must be a user_message episode")

    dialog_content: str | None = None
    for percept in episode["percepts"]:
        if percept["input_source"] == "dialog_text":
            dialog_content = percept["content"]
            break

    if dialog_content is None:
        _raise_validation_error("episode must include dialog_text content")

    target_scope = episode["target_scope"]
    origin_metadata = episode["origin_metadata"]
    projection: TextChatCompatibilityProjection = {
        "storage_timestamp_utc": episode["storage_timestamp_utc"],
        "local_time_context": episode["local_time_context"],
        "user_input": dialog_content,
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
        "platform_user_id": target_scope["current_platform_user_id"],
        "global_user_id": target_scope["current_global_user_id"],
        "user_name": target_scope["current_display_name"],
    }
    return projection


def _build_media_percepts(
    *,
    base_percept_id: str,
    media_description_rows: list[MediaDescriptionRow] | None,
) -> tuple[list[InputSource], list[CognitivePercept]]:
    """Build bounded media percepts and deterministic input-source labels.

    Args:
        base_percept_id: Dialog percept id used as the stable media id prefix.
        media_description_rows: Optional media descriptions for the turn.

    Returns:
        Tuple containing ordered media input-source labels and media percepts.
    """
    sanitized_rows = build_text_chat_media_description_rows(
        media_description_rows or []
    )
    bounded_rows = sanitized_rows[:MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS]

    media_percepts: list[CognitivePercept] = []
    has_image_observation = False
    has_audio_observation = False
    for media_index, row in enumerate(bounded_rows, start=1):
        content_type = row["content_type"]
        if content_type.startswith("image/"):
            input_source: InputSource = "image_observation"
            has_image_observation = True
            image_observation = row["image_observation"]
            media_content = image_observation["summary"]
            media_metadata: dict[str, Any] = {
                "observation_origin": image_observation["observation_origin"],
                "source_message_id": image_observation["source_message_id"],
                "media_kind": image_observation["media_kind"],
                "summary_status": image_observation["summary_status"],
                "media_index": media_index,
                "image_observation": image_observation,
            }
        else:
            input_source = "audio_observation"
            has_audio_observation = True
            media_content = _trim_media_description(row["description"])
            media_metadata = {
                "content_type": content_type,
                "media_index": media_index,
            }

        media_percept: CognitivePercept = {
            "percept_id": f"{base_percept_id}:media:{media_index}",
            "input_source": input_source,
            "content": media_content,
            "visibility": "model_visible",
            "metadata": media_metadata,
        }
        media_percepts.append(media_percept)

    media_input_sources: list[InputSource] = []
    if has_image_observation:
        media_input_sources.append("image_observation")
    if has_audio_observation:
        media_input_sources.append("audio_observation")

    return_value = (media_input_sources, media_percepts)
    return return_value


def _trim_media_description(description: str) -> str:
    """Clamp a media description to the episode percept character budget.

    Args:
        description: Sanitized image or audio description text.

    Returns:
        Description text no longer than the configured media percept limit.
    """
    if len(description) <= MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS:
        return description

    body_limit = MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS - len("...")
    trimmed_description = description[:body_limit].rstrip()
    return_value = f"{trimmed_description}..."
    return return_value


def _single_dialog_text_percept(episode: CognitiveEpisode) -> CognitivePercept:
    """Return the unique dialog-text percept required for text-chat refresh.

    Args:
        episode: Valid user-message episode.

    Returns:
        The single dialog-text percept carried by the episode.

    Raises:
        CognitiveEpisodeValidationError: If the dialog-text percept is missing
            or duplicated.
    """
    dialog_percepts = [
        percept
        for percept in episode["percepts"]
        if percept["input_source"] == "dialog_text"
    ]
    if len(dialog_percepts) != 1:
        _raise_validation_error("episode must include exactly one dialog_text")
    return_value = dialog_percepts[0]
    return return_value


def _raise_validation_error(message: str) -> NoReturn:
    raise CognitiveEpisodeValidationError(message)


def _require_field(
    mapping: dict[str, Any],
    field_name: str,
    container_name: str,
) -> Any:
    if field_name not in mapping:
        _raise_validation_error(f"{container_name}.{field_name} is required")
    value = mapping[field_name]
    return value


def _require_non_empty_string(
    mapping: dict[str, Any],
    field_name: str,
    container_name: str,
) -> str:
    value = _require_field(mapping, field_name, container_name)
    if not isinstance(value, str) or value == "":
        _raise_validation_error(
            f"{container_name}.{field_name} must be a non-empty string"
        )
    return_value = value
    return return_value


def _require_string(
    mapping: dict[str, Any],
    field_name: str,
    container_name: str,
) -> str:
    value = _require_field(mapping, field_name, container_name)
    if not isinstance(value, str):
        _raise_validation_error(f"{container_name}.{field_name} must be a string")
    return_value = value
    return return_value


def _require_bool(
    mapping: dict[str, Any],
    field_name: str,
    container_name: str,
) -> bool:
    value = _require_field(mapping, field_name, container_name)
    if not isinstance(value, bool):
        _raise_validation_error(f"{container_name}.{field_name} must be a bool")
    return_value = value
    return return_value


def _require_dict(
    mapping: dict[str, Any],
    field_name: str,
    container_name: str,
) -> dict[str, Any]:
    value = _require_field(mapping, field_name, container_name)
    if not isinstance(value, dict):
        _raise_validation_error(f"{container_name}.{field_name} must be a dict")
    return_value = value
    return return_value


def _require_non_empty_list(
    mapping: dict[str, Any],
    field_name: str,
    container_name: str,
) -> list[Any]:
    value = _require_field(mapping, field_name, container_name)
    if not isinstance(value, list) or not value:
        _raise_validation_error(
            f"{container_name}.{field_name} must be a non-empty list"
        )
    return_value = value
    return return_value


def _require_literal(
    mapping: dict[str, Any],
    field_name: str,
    container_name: str,
    supported_values: frozenset[str],
) -> str:
    value = _require_field(mapping, field_name, container_name)
    if not isinstance(value, str) or value not in supported_values:
        _raise_validation_error(
            f"{container_name}.{field_name} must be a supported value"
        )
    return_value = value
    return return_value


def _require_literal_list(
    mapping: dict[str, Any],
    field_name: str,
    container_name: str,
    supported_values: frozenset[str],
) -> list[str]:
    value = _require_non_empty_list(mapping, field_name, container_name)
    for index, item in enumerate(value):
        if not isinstance(item, str) or item not in supported_values:
            _raise_validation_error(
                f"{container_name}.{field_name}[{index}] must be supported"
            )
    return_value = value
    return return_value


def _require_string_list(
    mapping: dict[str, Any],
    field_name: str,
    container_name: str,
) -> list[str]:
    value = _require_field(mapping, field_name, container_name)
    if not isinstance(value, list):
        _raise_validation_error(f"{container_name}.{field_name} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            _raise_validation_error(
                f"{container_name}.{field_name}[{index}] must be a string"
            )
    return_value = value
    return return_value


def _validate_target_scope(target_scope: dict[str, Any]) -> None:
    _require_string(target_scope, "platform", "target_scope")
    _require_string(target_scope, "platform_channel_id", "target_scope")
    _require_string(target_scope, "channel_type", "target_scope")
    _require_string(target_scope, "current_platform_user_id", "target_scope")
    _require_string(target_scope, "current_global_user_id", "target_scope")
    _require_string(target_scope, "current_display_name", "target_scope")
    _require_string_list(
        target_scope,
        "target_addressed_user_ids",
        "target_scope",
    )
    _require_bool(target_scope, "target_broadcast", "target_scope")


def _validate_origin_metadata(origin_metadata: dict[str, Any]) -> None:
    _require_string(origin_metadata, "platform", "origin_metadata")
    _require_string(origin_metadata, "platform_message_id", "origin_metadata")
    _require_string_list(
        origin_metadata,
        "active_turn_platform_message_ids",
        "origin_metadata",
    )
    _require_string_list(
        origin_metadata,
        "active_turn_conversation_row_ids",
        "origin_metadata",
    )
    debug_modes = _require_dict(
        origin_metadata,
        "debug_modes",
        "origin_metadata",
    )
    for key, value in debug_modes.items():
        if not isinstance(key, str) or not isinstance(value, bool):
            _raise_validation_error(
                "origin_metadata.debug_modes must map strings to bools"
            )


def _validate_local_time_context(
    local_time_context: dict[str, Any],
) -> None:
    for field_name in _LOCAL_TIME_CONTEXT_FIELDS:
        _require_non_empty_string(
            local_time_context,
            field_name,
            "local_time_context",
        )


def _validate_percepts(percepts: list[Any]) -> set[str]:
    percept_ids: set[str] = set()
    percept_input_sources: set[str] = set()
    for index, item in enumerate(percepts):
        if not isinstance(item, dict):
            _raise_validation_error(f"percepts[{index}] must be a dict")

        container_name = f"percepts[{index}]"
        percept_id = _require_non_empty_string(
            item,
            "percept_id",
            container_name,
        )
        if percept_id in percept_ids:
            _raise_validation_error("percept_id values must be unique")
        percept_ids.add(percept_id)

        input_source = _require_literal(
            item,
            "input_source",
            container_name,
            _INPUT_SOURCES,
        )
        _require_string(item, "content", container_name)
        _require_literal(item, "visibility", container_name, _VISIBILITIES)
        _require_dict(item, "metadata", container_name)
        percept_input_sources.add(input_source)

    return_value = percept_input_sources
    return return_value


def _validate_percept_sources_match_input_sources(
    input_sources: list[str],
    percept_input_sources: set[str],
) -> None:
    for input_source in input_sources:
        if input_source not in percept_input_sources:
            _raise_validation_error(
                "every input_source must be represented by a percept"
            )

    allowed_input_sources = set(input_sources)
    for input_source in percept_input_sources:
        if input_source not in allowed_input_sources:
            _raise_validation_error(
                "every percept input_source must be listed in input_sources"
            )


def _validate_user_message_has_dialog_text(
    percept_input_sources: set[str],
) -> None:
    if "dialog_text" not in percept_input_sources:
        _raise_validation_error("user_message episodes require dialog_text")
