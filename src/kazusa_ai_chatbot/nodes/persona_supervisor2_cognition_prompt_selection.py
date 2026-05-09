"""Prompt selection contracts for shared cognition nodes."""

from __future__ import annotations

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
    "l2a_consciousness",
    "l2b_boundary_core",
    "l2c_judgment_core",
    "l3_contextual_agent",
    "l3_style_agent",
    "l3_content_anchor_agent",
    "l3_preference_adapter",
    "l3_visual_agent",
]

CognitionPromptVariant = Literal[
    "text_chat_user_message",
    "reflection_signal_reflection_artifact",
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
_TEXT_CHAT_OUTPUT_MODES = frozenset(("visible_reply", "think_only", "silent"))
_REFLECTION_INPUT_SOURCES: list[InputSource] = ["reflection_artifact"]
_REFLECTION_OUTPUT_MODES = frozenset(("think_only", "preview", "silent"))


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
        Prompt-selection metadata for the current supported text chat variant.

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
        Empty mapping for current text chat; reflection artifact content for
        the reflection dry-run prompt variant.

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
    elif variant == "reflection_signal_reflection_artifact":
        source_payload = _reflection_source_payload(episode)
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
    if input_sources != _TEXT_CHAT_INPUT_SOURCES:
        raise CognitionPromptSelectionError(
            f"input_sources are not supported: {input_sources}"
        )
    if output_mode not in _TEXT_CHAT_OUTPUT_MODES:
        raise CognitionPromptSelectionError(
            f"output_mode is not supported: {output_mode}"
        )
    selection: CognitionPromptSelection = {
        "stage": stage,
        "variant": "text_chat_user_message",
        "prompt_key": f"{stage}.text_chat_user_message",
        "trigger_source": trigger_source,
        "input_sources": list(input_sources),
        "output_mode": output_mode,
    }
    return selection


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
