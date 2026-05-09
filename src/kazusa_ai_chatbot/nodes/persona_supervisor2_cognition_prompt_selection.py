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

CognitionPromptVariant = Literal["text_chat_user_message"]


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

    trigger_source = episode["trigger_source"]
    if trigger_source != "user_message":
        raise CognitionPromptSelectionError(
            f"trigger_source is not supported: {trigger_source}"
        )

    input_sources = episode["input_sources"]
    if input_sources != _TEXT_CHAT_INPUT_SOURCES:
        raise CognitionPromptSelectionError(
            f"input_sources are not supported: {input_sources}"
        )

    output_mode = episode["output_mode"]
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
