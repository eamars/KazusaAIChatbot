"""Source-neutral cognitive episode contracts and builders."""

from __future__ import annotations

from typing import Any, Literal, NoReturn, TypedDict, get_args

from kazusa_ai_chatbot.time_context import TimeContextDoc

TriggerSource = Literal[
    "user_message",
    "reflection_signal",
    "internal_thought",
    "scheduled_recall",
    "system_probe",
]

InputSource = Literal[
    "dialog_text",
    "image_observation",
    "audio_observation",
    "internal_monologue",
    "reflection_artifact",
    "retrieved_memory",
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


class CognitiveEpisode(TypedDict):
    episode_id: str
    trigger_source: TriggerSource
    input_sources: list[InputSource]
    output_mode: OutputMode
    percepts: list[CognitivePercept]
    target_scope: TargetScope
    origin_metadata: OriginMetadata
    timestamp: str
    time_context: TimeContextDoc


class TextChatCompatibilityProjection(TypedDict):
    timestamp: str
    time_context: TimeContextDoc
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
_TIME_CONTEXT_FIELDS = tuple(TimeContextDoc.__annotations__)


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
    _require_non_empty_string(episode_data, "timestamp", "episode")
    time_context = _require_dict(episode_data, "time_context", "episode")
    _validate_time_context(time_context)

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
    timestamp: str,
    time_context: TimeContextDoc,
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
) -> CognitiveEpisode:
    """Build a source-neutral episode for the current text `/chat` turn.

    Args:
        episode_id: Stable id for this cognitive episode.
        percept_id: Stable id for the single text percept.
        timestamp: Original turn timestamp.
        time_context: Character-local time context already built by the caller.
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
        "input_sources": ["dialog_text"],
        "output_mode": output_mode,
        "percepts": [percept],
        "target_scope": target_scope,
        "origin_metadata": origin_metadata,
        "timestamp": timestamp,
        "time_context": time_context,
    }

    validate_cognitive_episode(episode)
    return episode


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
        "timestamp": episode["timestamp"],
        "time_context": episode["time_context"],
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


def _validate_time_context(time_context: dict[str, Any]) -> None:
    for field_name in _TIME_CONTEXT_FIELDS:
        _require_non_empty_string(time_context, field_name, "time_context")


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
