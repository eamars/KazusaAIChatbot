"""Prompt selection contracts for shared cognition nodes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal, TypedDict, get_args

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    InputSource,
    OutputMode,
    TriggerSource,
    validate_cognitive_episode,
)

CognitionPromptStage = Literal[
    "l1_subconscious",
    "l2a_conscious_framing",
    "l2b_boundary_appraisal",
    "l2c1_judgment_synthesis",
    "l2c2_social_context_appraisal",
    "l2d_action_selection",
    "l3_style_agent",
    "l3_content_anchor_agent",
    "l3_preference_adapter",
    "l3_visual_agent",
]

CognitionPromptVariant = Literal[
    "text_chat_user_message",
    "text_chat_user_message_image_observation",
    "text_chat_user_message_audio_observation",
    "text_chat_user_message_image_audio_observation",
    "reflection_signal_reflection_artifact",
    "internal_thought_internal_monologue",
]


class CognitionPromptSelection(TypedDict):
    stage: CognitionPromptStage
    variant: CognitionPromptVariant
    prompt_key: str
    trigger_source: TriggerSource
    input_sources: list[InputSource]
    output_mode: OutputMode


class CognitionPromptSelectionError(ValueError):
    """Raised when an episode cannot select a cognition prompt variant."""


_SUPPORTED_STAGES = frozenset(get_args(CognitionPromptStage))
_TEXT_CHAT_INPUT_SOURCES: list[InputSource] = ["dialog_text"]
_TEXT_CHAT_IMAGE_INPUT_SOURCES: list[InputSource] = [
    "dialog_text",
    "image_observation",
]
_TEXT_CHAT_AUDIO_INPUT_SOURCES: list[InputSource] = [
    "dialog_text",
    "audio_observation",
]
_TEXT_CHAT_IMAGE_AUDIO_INPUT_SOURCES: list[InputSource] = [
    "dialog_text",
    "image_observation",
    "audio_observation",
]
_TEXT_CHAT_OUTPUT_MODES = frozenset(("visible_reply", "think_only", "silent"))
_REFLECTION_INPUT_SOURCES: list[InputSource] = ["reflection_artifact"]
_REFLECTION_OUTPUT_MODES = frozenset(("think_only", "preview", "silent"))
_INTERNAL_THOUGHT_INPUT_SOURCES: list[InputSource] = ["internal_monologue"]
_INTERNAL_THOUGHT_OUTPUT_MODES = frozenset(("think_only", "preview", "silent"))
_IMAGE_OBSERVATION_PAYLOAD_FIELDS = (
    "observation_origin",
    "media_kind",
    "summary_status",
    "summary",
    "visible_text",
    "salient_visual_facts",
    "spatial_or_scene_facts",
    "uncertainty",
)


def select_cognition_prompt_variant(
    *,
    episode: CognitiveEpisode,
    stage: CognitionPromptStage,
) -> CognitionPromptSelection:
    """Select the cognition prompt variant for one stage.

    Args:
        episode: Valid cognitive episode carried by the cognition state.
        stage: L1, L2, or L3 cognition stage requesting a prompt template.

    Returns:
        Prompt-selection metadata for the current supported cognition source
        variant.

    Raises:
        CognitiveEpisodeValidationError: If the episode is structurally
            invalid.
        CognitionPromptSelectionError: If the stage or source combination is
            not enabled for runtime prompt selection.
    """
    validate_cognitive_episode(episode)

    if stage not in _SUPPORTED_STAGES:
        raise CognitionPromptSelectionError(f"stage is not supported: {stage}")

    input_sources = episode["input_sources"]
    output_mode = episode["output_mode"]
    trigger_source = episode["trigger_source"]
    if trigger_source == "user_message":
        selection = _select_text_chat_prompt(
            input_sources=input_sources,
            output_mode=output_mode,
            stage=stage,
            trigger_source=trigger_source,
        )
    elif trigger_source == "reflection_signal":
        selection = _select_reflection_prompt(
            input_sources=input_sources,
            output_mode=output_mode,
            stage=stage,
            trigger_source=trigger_source,
        )
    elif trigger_source == "internal_thought":
        selection = _select_internal_thought_prompt(
            input_sources=input_sources,
            output_mode=output_mode,
            stage=stage,
            trigger_source=trigger_source,
        )
    else:
        raise CognitionPromptSelectionError(
            f"trigger_source is not supported: {trigger_source}"
        )
    return selection


def build_cognition_prompt_source_payload(
    *,
    episode: CognitiveEpisode,
    selection: CognitionPromptSelection,
) -> dict[str, object]:
    """Build source-specific model payload fields for cognition prompts.

    Args:
        episode: Valid cognitive episode carried by the cognition state.
        selection: Prompt-selection metadata returned for this episode.

    Returns:
        Empty mapping for text-only chat; source-specific model-visible payload
        fields for current-turn media prompt variants.

    Raises:
        CognitiveEpisodeValidationError: If the episode is structurally
            invalid.
        CognitionPromptSelectionError: If the selected source payload cannot
            be projected safely.
    """
    validate_cognitive_episode(episode)

    variant = selection["variant"]
    if variant == "text_chat_user_message":
        source_payload: dict[str, object] = {}
    elif variant in (
        "text_chat_user_message_image_observation",
        "text_chat_user_message_audio_observation",
        "text_chat_user_message_image_audio_observation",
    ):
        source_payload = _media_observations_source_payload(episode)
    elif variant == "reflection_signal_reflection_artifact":
        source_payload = _reflection_source_payload(episode)
    elif variant == "internal_thought_internal_monologue":
        source_payload = _internal_thought_source_payload(episode)
    else:
        raise CognitionPromptSelectionError(
            f"variant is not supported: {variant}"
        )
    return source_payload


def _select_text_chat_prompt(
    *,
    input_sources: list[InputSource],
    output_mode: OutputMode,
    stage: CognitionPromptStage,
    trigger_source: TriggerSource,
) -> CognitionPromptSelection:
    """Select the established text-chat prompt variant.

    Args:
        input_sources: Episode input-source list.
        output_mode: Episode output mode.
        stage: Cognition stage requesting a prompt template.
        trigger_source: Episode trigger source.

    Returns:
        Prompt-selection metadata for current text chat.

    Raises:
        CognitionPromptSelectionError: If the text-chat tuple is not enabled.
    """
    if input_sources == _TEXT_CHAT_INPUT_SOURCES:
        variant: CognitionPromptVariant = "text_chat_user_message"
    elif input_sources == _TEXT_CHAT_IMAGE_INPUT_SOURCES:
        variant = "text_chat_user_message_image_observation"
    elif input_sources == _TEXT_CHAT_AUDIO_INPUT_SOURCES:
        variant = "text_chat_user_message_audio_observation"
    elif input_sources == _TEXT_CHAT_IMAGE_AUDIO_INPUT_SOURCES:
        variant = "text_chat_user_message_image_audio_observation"
    else:
        raise CognitionPromptSelectionError(
            f"input_sources are not supported: {input_sources}"
        )
    if output_mode not in _TEXT_CHAT_OUTPUT_MODES:
        raise CognitionPromptSelectionError(
            f"output_mode is not supported: {output_mode}"
        )
    selection: CognitionPromptSelection = {
        "stage": stage,
        "variant": variant,
        "prompt_key": f"{stage}.{variant}",
        "trigger_source": trigger_source,
        "input_sources": list(input_sources),
        "output_mode": output_mode,
    }
    return selection


def _media_observations_source_payload(
    episode: CognitiveEpisode,
) -> dict[str, object]:
    """Project user-message media observations into model-facing payload.

    Args:
        episode: Valid user-message cognitive episode.

    Returns:
        Mapping containing current-turn structured image observations and audio
        observation text.
    """
    image_observations: list[dict[str, object]] = []
    audio_observations: list[str] = []
    for percept in episode["percepts"]:
        if percept["input_source"] == "image_observation":
            image_observations.append(_project_image_observation(percept))
        elif percept["input_source"] == "audio_observation":
            audio_observations.append(percept["content"])

    media_payload: dict[str, object] = {
        "media_observations": {
            "image_observations": image_observations,
            "audio_observations": audio_observations,
        },
    }
    return media_payload


def _project_image_observation(percept: dict[str, object]) -> dict[str, object]:
    """Project one image percept into the cognition prompt payload.

    Args:
        percept: Valid image-observation percept from a cognitive episode.

    Returns:
        Prompt-safe structured image observation.
    """
    metadata = percept["metadata"]
    if not isinstance(metadata, Mapping):
        raise CognitionPromptSelectionError("image percept metadata must be a dict")

    observation = metadata.get("image_observation")
    if not isinstance(observation, Mapping):
        raise CognitionPromptSelectionError(
            "image percept metadata.image_observation must be a dict"
        )

    projected: dict[str, object] = {}
    for field_name in _IMAGE_OBSERVATION_PAYLOAD_FIELDS:
        projected[field_name] = observation[field_name]
    return projected


def _select_reflection_prompt(
    *,
    input_sources: list[InputSource],
    output_mode: OutputMode,
    stage: CognitionPromptStage,
    trigger_source: TriggerSource,
) -> CognitionPromptSelection:
    """Select the reflection dry-run prompt variant.

    Args:
        input_sources: Episode input-source list.
        output_mode: Episode output mode.
        stage: Cognition stage requesting a prompt template.
        trigger_source: Episode trigger source.

    Returns:
        Prompt-selection metadata for promoted reflection dry runs.

    Raises:
        CognitionPromptSelectionError: If the reflection tuple is not enabled.
    """
    if input_sources != _REFLECTION_INPUT_SOURCES:
        raise CognitionPromptSelectionError(
            f"input_sources are not supported: {input_sources}"
        )
    if output_mode not in _REFLECTION_OUTPUT_MODES:
        raise CognitionPromptSelectionError(
            f"output_mode is not supported: {output_mode}"
        )
    selection: CognitionPromptSelection = {
        "stage": stage,
        "variant": "reflection_signal_reflection_artifact",
        "prompt_key": f"{stage}.reflection_signal_reflection_artifact",
        "trigger_source": trigger_source,
        "input_sources": list(input_sources),
        "output_mode": output_mode,
    }
    return selection


def _select_internal_thought_prompt(
    *,
    input_sources: list[InputSource],
    output_mode: OutputMode,
    stage: CognitionPromptStage,
    trigger_source: TriggerSource,
) -> CognitionPromptSelection:
    """Select the internal-thought dry-run prompt variant.

    Args:
        input_sources: Episode input-source list.
        output_mode: Episode output mode.
        stage: Cognition stage requesting a prompt template.
        trigger_source: Episode trigger source.

    Returns:
        Prompt-selection metadata for internal-thought dry runs.

    Raises:
        CognitionPromptSelectionError: If the internal-thought tuple is not
            enabled.
    """
    if input_sources != _INTERNAL_THOUGHT_INPUT_SOURCES:
        raise CognitionPromptSelectionError(
            f"input_sources are not supported: {input_sources}"
        )
    if output_mode not in _INTERNAL_THOUGHT_OUTPUT_MODES:
        raise CognitionPromptSelectionError(
            f"output_mode is not supported: {output_mode}"
        )
    selection: CognitionPromptSelection = {
        "stage": stage,
        "variant": "internal_thought_internal_monologue",
        "prompt_key": f"{stage}.internal_thought_internal_monologue",
        "trigger_source": trigger_source,
        "input_sources": list(input_sources),
        "output_mode": output_mode,
    }
    return selection


def _reflection_source_payload(
    episode: CognitiveEpisode,
) -> dict[str, object]:
    """Project the single reflection artifact into model-facing payload.

    Args:
        episode: Valid reflection cognitive episode.

    Returns:
        Mapping containing exactly the model-visible reflection artifact.

    Raises:
        CognitionPromptSelectionError: If the artifact percept is missing or
            not unique.
    """
    reflection_percepts = [
        percept
        for percept in episode["percepts"]
        if percept["input_source"] == "reflection_artifact"
    ]
    if len(reflection_percepts) != 1:
        raise CognitionPromptSelectionError(
            "reflection_artifact percept must be unique"
        )

    reflection_payload: dict[str, object] = {
        "reflection_artifact": reflection_percepts[0]["content"],
    }
    return reflection_payload


def _internal_thought_source_payload(
    episode: CognitiveEpisode,
) -> dict[str, object]:
    """Project the single internal monologue into model-facing payload.

    Args:
        episode: Valid internal-thought cognitive episode.

    Returns:
        Mapping containing exactly the model-visible private residue payload.

    Raises:
        CognitionPromptSelectionError: If the internal-monologue percept is
            missing, duplicated, malformed, or structurally wrong.
    """
    internal_percepts = [
        percept
        for percept in episode["percepts"]
        if percept["input_source"] == "internal_monologue"
    ]
    if len(internal_percepts) != 1:
        raise CognitionPromptSelectionError(
            "internal_monologue percept must be unique"
        )

    percept_content = internal_percepts[0]["content"]
    try:
        content_payload = json.loads(percept_content)
    except json.JSONDecodeError as exc:
        raise CognitionPromptSelectionError(
            f"internal_monologue percept content is malformed: {exc}"
        ) from exc

    if not isinstance(content_payload, dict):
        raise CognitionPromptSelectionError(
            "internal_monologue percept content must be a dict"
        )

    residue = content_payload.get("residue")
    action_latch = content_payload.get("action_latch")
    if not isinstance(residue, dict):
        raise CognitionPromptSelectionError(
            "internal_monologue residue must be a dict"
        )
    if not isinstance(action_latch, dict):
        raise CognitionPromptSelectionError(
            "internal_monologue action_latch must be a dict"
        )
    projected_action_latch: dict[str, str] = {}
    for action_latch_key, action_latch_value in action_latch.items():
        if not isinstance(action_latch_key, str):
            raise CognitionPromptSelectionError(
                "internal_monologue action_latch keys must be strings"
            )
        if not isinstance(action_latch_value, str):
            raise CognitionPromptSelectionError(
                "internal_monologue action_latch values must be strings"
            )
        projected_action_latch[action_latch_key] = action_latch_value

    residue_id = residue.get("residue_id")
    internal_monologue = residue.get("internal_monologue")
    if not isinstance(residue_id, str) or residue_id == "":
        raise CognitionPromptSelectionError(
            "internal_monologue residue_id must be a non-empty string"
        )
    if not isinstance(internal_monologue, str) or internal_monologue == "":
        raise CognitionPromptSelectionError(
            "internal_monologue text must be a non-empty string"
        )

    internal_payload: dict[str, object] = {
        "internal_thought_residue": {
            "residue_id": residue_id,
            "internal_monologue": internal_monologue,
            "action_latch": projected_action_latch,
        },
    }
    return internal_payload
