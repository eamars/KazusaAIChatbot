"""Internal projection from public cognition ICD to prompt-selection episode."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    CognitionChainInputV1,
)

_TRIGGER_SOURCES = frozenset((
    "user_message",
    "reflection_signal",
    "internal_thought",
    "scheduled_recall",
    "system_probe",
    "background_artifact_result_ready",
    "background_work_result_ready",
    "accepted_task_result_ready",
))
_INPUT_SOURCES = frozenset((
    "dialog_text",
    "image_observation",
    "audio_observation",
    "internal_monologue",
    "reflection_artifact",
    "retrieved_memory",
    "background_artifact_result",
    "background_work_result",
    "accepted_task_result",
))
_PUBLIC_OUTPUT_MODE_MAP = {
    "live_response": "visible_reply",
    "background_cognition": "think_only",
    "dry_run": "preview",
}
_LEGACY_OUTPUT_MODE_MAP = {
    "visible_reply": "live_response",
    "silent": "background_cognition",
    "think_only": "background_cognition",
    "preview": "dry_run",
    "scheduled_action_request": "background_cognition",
}


def build_prompt_selection_episode(
    input_payload: CognitionChainInputV1,
) -> dict[str, Any]:
    """Build the internal episode shape expected by prompt selection."""

    episode = input_payload["episode"]
    scene = input_payload["scene"]
    current_user = input_payload["current_user"]
    trigger_source = _trigger_source(episode.get("trigger_source"))
    percepts = _percepts(input_payload)
    input_sources = _input_sources(episode.get("input_sources"), percepts)
    output_mode = _output_mode(
        episode.get("output_mode"),
        trigger_source=trigger_source,
    )
    prompt_episode = {
        "episode_id": _non_empty_text(
            episode.get("episode_id"),
            "public-cognition-episode",
        ),
        "trigger_source": trigger_source,
        "input_sources": input_sources,
        "output_mode": output_mode,
        "percepts": percepts,
        "target_scope": {
            "platform": _text(scene.get("platform")),
            "platform_channel_id": "",
            "channel_type": _text(scene.get("channel_type")),
            "current_platform_user_id": "",
            "current_global_user_id": _text(current_user.get("global_user_id")),
            "current_display_name": _text(current_user.get("display_name")),
            "target_addressed_user_ids": [],
            "target_broadcast": False,
        },
        "origin_metadata": _origin_metadata(input_payload),
        "storage_timestamp_utc": _non_empty_text(
            scene.get("storage_timestamp_utc"),
            "1970-01-01T00:00:00Z",
        ),
        "local_time_context": _local_time_context(scene.get("local_time_context")),
    }
    return prompt_episode


def public_output_mode(value: object) -> str:
    """Return the public V1 output mode for old or new episode values."""

    if value in _PUBLIC_OUTPUT_MODE_MAP:
        return_value = str(value)
        return return_value
    mapped_value = _LEGACY_OUTPUT_MODE_MAP.get(str(value))
    if mapped_value is not None:
        return mapped_value
    return_value = "live_response"
    return return_value


def _trigger_source(value: object) -> str:
    """Return a prompt-selection trigger source."""

    if isinstance(value, str) and value in _TRIGGER_SOURCES:
        return_value = value
        return return_value
    return_value = str(value)
    return return_value


def _input_sources(
    raw_input_sources: object,
    percepts: list[dict[str, Any]],
) -> list[str]:
    """Return input sources represented by the generated percepts."""

    percept_sources = [str(percept["input_source"]) for percept in percepts]
    if not isinstance(raw_input_sources, list):
        return percept_sources
    sources: list[str] = []
    for raw_source in raw_input_sources:
        source = _input_source(raw_source)
        if source in percept_sources and source not in sources:
            sources.append(source)
    for source in percept_sources:
        if source not in sources:
            sources.append(source)
    return sources


def _input_source(value: object) -> str:
    """Return one prompt-selection input source."""

    if isinstance(value, str) and value in _INPUT_SOURCES:
        return_value = value
        return return_value
    return_value = str(value)
    return return_value


def _output_mode(value: object, *, trigger_source: str) -> str:
    """Return one prompt-selection output mode."""

    mapped_value = _PUBLIC_OUTPUT_MODE_MAP.get(str(value))
    if mapped_value is None:
        mapped_value = str(value)
    if trigger_source == "user_message" and mapped_value in (
        "visible_reply",
        "think_only",
        "silent",
    ):
        return mapped_value
    if trigger_source in ("reflection_signal", "internal_thought") and mapped_value in (
        "think_only",
        "preview",
        "silent",
    ):
        return mapped_value
    if trigger_source in (
        "background_artifact_result_ready",
        "background_work_result_ready",
        "accepted_task_result_ready",
    ):
        return_value = "visible_reply"
        return return_value
    if trigger_source == "user_message":
        return_value = "visible_reply"
        return return_value
    return_value = "preview"
    return return_value


def _percepts(input_payload: CognitionChainInputV1) -> list[dict[str, Any]]:
    """Build prompt-selection percept rows from public model-visible percepts."""

    episode = input_payload["episode"]
    raw_percepts = episode.get("model_visible_percepts")
    percepts: list[dict[str, Any]] = []
    if isinstance(raw_percepts, list):
        for index, raw_percept in enumerate(raw_percepts, start=1):
            if not isinstance(raw_percept, Mapping):
                continue
            input_source = _input_source(raw_percept.get("input_source"))
            content = _text(raw_percept.get("content"))
            percepts.append({
                "percept_id": _non_empty_text(
                    raw_percept.get("percept_id"),
                    f"public-percept-{index}",
                ),
                "input_source": input_source,
                "content": content,
                "visibility": "model_visible",
                "metadata": _percept_metadata(input_source, content),
            })
    existing_sources = {percept["input_source"] for percept in percepts}
    for index, media_percept in enumerate(
        _media_percepts(input_payload, existing_sources),
        start=len(percepts) + 1,
    ):
        media_percept.setdefault("percept_id", f"public-media-{index}")
        percepts.append(media_percept)
    if not percepts:
        current_event = input_payload["current_event"]
        percepts.append({
            "percept_id": "current_input",
            "input_source": "dialog_text",
            "content": _text(current_event.get("decontextualized_input")),
            "visibility": "model_visible",
            "metadata": {},
        })
    return percepts


def _media_percepts(
    input_payload: CognitionChainInputV1,
    existing_sources: set[object],
) -> list[dict[str, Any]]:
    """Build prompt-selection media percepts from public media observations."""

    current_event = input_payload["current_event"]
    raw_media = current_event.get("media_observations")
    if not isinstance(raw_media, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    percepts: list[dict[str, Any]] = []
    for index, media in enumerate(raw_media, start=1):
        if not isinstance(media, Mapping):
            continue
        input_source = _media_input_source(media.get("modality"))
        if not input_source or input_source in existing_sources:
            continue
        content = _text(media.get("observation"))
        if not content:
            continue
        percepts.append({
            "percept_id": f"public-media-{index}",
            "input_source": input_source,
            "content": content,
            "visibility": "model_visible",
            "metadata": _media_metadata(input_source, media, content),
        })
        existing_sources.add(input_source)
    return percepts


def _media_input_source(modality: object) -> str:
    """Return prompt-selection source for a public media modality."""

    if modality == "image":
        return_value = "image_observation"
        return return_value
    if modality == "audio":
        return_value = "audio_observation"
        return return_value
    return_value = ""
    return return_value


def _media_metadata(
    input_source: str,
    media: Mapping[str, Any],
    content: str,
) -> dict[str, Any]:
    """Return prompt-selection metadata for a public media observation."""

    if input_source == "image_observation":
        metadata = _percept_metadata(input_source, content)
        return metadata
    return_value = {
        "source_summary": _text(media.get("source_summary")),
    }
    return return_value


def _percept_metadata(input_source: str, content: str) -> dict[str, Any]:
    """Return minimal valid metadata for prompt-selection source payloads."""

    if input_source == "image_observation":
        image_observation = {
            "observation_origin": "public_cognition_input",
            "source_message_id": "",
            "media_kind": "image",
            "summary_status": "available" if content else "unavailable",
            "summary": content,
            "visible_text": [],
            "salient_visual_facts": [],
            "spatial_or_scene_facts": [],
            "uncertainty": [],
        }
        return_value = {
            "image_observation": image_observation,
        }
        return return_value
    return_value: dict[str, Any] = {}
    return return_value


def _origin_metadata(input_payload: CognitionChainInputV1) -> dict[str, Any]:
    """Return prompt-selection origin metadata without raw adapter ids."""

    scene = input_payload["scene"]
    episode = input_payload["episode"]
    raw_origin_metadata = episode.get("origin_metadata")
    if isinstance(raw_origin_metadata, Mapping):
        debug_modes = raw_origin_metadata.get("debug_modes")
    else:
        debug_modes = {}
    if not isinstance(debug_modes, Mapping):
        debug_modes = {}
    return_value = {
        "platform": _text(scene.get("platform")),
        "platform_message_id": "",
        "active_turn_platform_message_ids": [],
        "active_turn_conversation_row_ids": [],
        "debug_modes": {
            str(key): bool(value)
            for key, value in debug_modes.items()
            if isinstance(key, str) and isinstance(value, bool)
        },
    }
    return return_value


def _local_time_context(value: object) -> dict[str, str]:
    """Return the local time context fields required by prompt selection."""

    if isinstance(value, Mapping):
        current_local_datetime = _text(value.get("current_local_datetime"))
        current_local_weekday = _text(value.get("current_local_weekday"))
    else:
        current_local_datetime = ""
        current_local_weekday = ""
    return_value = {
        "current_local_datetime": (
            current_local_datetime or "1970-01-01 00:00"
        ),
        "current_local_weekday": current_local_weekday or "Thursday",
    }
    return return_value


def _non_empty_text(value: object, fallback: str) -> str:
    """Return stripped text or a fallback when empty."""

    text = _text(value)
    if text:
        return_value = text
        return return_value
    return fallback


def _text(value: object) -> str:
    """Return a string value or an empty string."""

    if isinstance(value, str):
        return_value = value.strip()
        return return_value
    if value is None:
        return_value = ""
        return return_value
    return_value = str(value)
    return return_value
