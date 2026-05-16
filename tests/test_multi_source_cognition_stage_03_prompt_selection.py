"""Stage 03 prompt-selection and cognition output-contract tests."""

from __future__ import annotations

from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    OutputMode,
    build_text_chat_cognitive_episode,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_output_contracts import (
    CognitionOutputContractError,
    validate_cognition_output_contract,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_prompt_selection import (
    CognitionPromptSelectionError,
    CognitionPromptStage,
    select_cognition_prompt_variant,
)
from kazusa_ai_chatbot.time_context import build_character_time_context


_APPROVED_STAGES: tuple[CognitionPromptStage, ...] = (
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
)

_VALID_OUTPUT_PAYLOADS: dict[CognitionPromptStage, dict[str, object]] = {
    "l1_subconscious": {
        "emotional_appraisal": "steady",
        "interaction_subtext": "routine",
    },
    "l2a_conscious_framing": {
        "internal_monologue": "Answer directly.",
        "character_intent": "PROVIDE",
        "logical_stance": "CONFIRM",
    },
    "l2b_boundary_appraisal": {
        "boundary_core_assessment": {
            "boundary_issue": "none",
            "acceptance": "allow",
        },
    },
    "l2c1_judgment_synthesis": {
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "judgment_note": "stable",
    },
    "l2c2_social_context_appraisal": {
        "social_distance": "neutral",
        "emotional_intensity": "low",
        "vibe_check": "routine",
        "relational_dynamic": "stable",
    },
    "l2d_action_selection": {
        "action_specs": [],
    },
    "l3_style_agent": {
        "rhetorical_strategy": "answer briefly",
        "linguistic_style": "plain",
        "forbidden_phrases": [],
    },
    "l3_content_anchor_agent": {
        "content_anchors": ["[DECISION] answer"],
    },
    "l3_preference_adapter": {
        "accepted_user_preferences": [],
    },
    "l3_visual_agent": {
        "facial_expression": ["neutral"],
        "body_language": ["still"],
        "gaze_direction": ["forward"],
        "visual_vibe": ["plain"],
    },
}

_WRONG_OUTPUT_VALUES: dict[CognitionPromptStage, object] = {
    "l1_subconscious": [],
    "l2a_conscious_framing": [],
    "l2b_boundary_appraisal": "not a dict",
    "l2c1_judgment_synthesis": [],
    "l2c2_social_context_appraisal": [],
    "l2d_action_selection": "not a list",
    "l3_style_agent": [],
    "l3_content_anchor_agent": "not a list",
    "l3_preference_adapter": "not a list",
    "l3_visual_agent": "not a list",
}


def _text_chat_episode(
    output_mode: OutputMode = "visible_reply",
) -> CognitiveEpisode:
    """Build a valid Stage 02 text chat episode fixture.

    Args:
        output_mode: Output mode to put on the episode.

    Returns:
        Valid `CognitiveEpisode` for selector tests.
    """
    timestamp = "2026-05-01T09:00:00+12:00"
    episode = build_text_chat_cognitive_episode(
        episode_id="user_message:debug:direct:message-1",
        percept_id="user_message:debug:direct:message-1:dialog_text:0",
        timestamp=timestamp,
        time_context=build_character_time_context(timestamp),
        user_input="Please keep it short.",
        platform="debug",
        platform_channel_id="direct",
        channel_type="private",
        platform_message_id="message-1",
        platform_user_id="platform-user-1",
        global_user_id="global-user-1",
        user_name="Test User",
        active_turn_platform_message_ids=["message-1"],
        active_turn_conversation_row_ids=["conversation-row-1"],
        debug_modes={},
        output_mode=output_mode,
        target_addressed_user_ids=[],
        target_broadcast=False,
    )
    return episode


def _reflection_episode(
    output_mode: OutputMode = "think_only",
) -> CognitiveEpisode:
    """Build a structurally valid non-chat episode fixture.

    Args:
        output_mode: Output mode to put on the reflection episode.

    Returns:
        Valid `CognitiveEpisode` whose trigger source is reflection.
    """
    episode = _text_chat_episode(output_mode=output_mode)
    episode["trigger_source"] = "reflection_signal"
    episode["input_sources"] = ["reflection_artifact"]
    episode["percepts"] = [
        {
            "percept_id": "reflection:artifact:1",
            "input_source": "reflection_artifact",
            "content": "Promoted reflection summary.",
            "visibility": "model_visible",
            "metadata": {},
        },
    ]
    return episode


def _retrieved_memory_text_episode() -> CognitiveEpisode:
    """Build a chat episode with an unsupported extra input source.

    Returns:
        Valid `CognitiveEpisode` with both `dialog_text` and a retrieved memory
        source that prompt selection still rejects for user messages.
    """
    episode = _text_chat_episode()
    episode["input_sources"] = ["dialog_text", "retrieved_memory"]
    episode["percepts"].append(
        {
            "percept_id": "user_message:debug:direct:message-1:memory:0",
            "input_source": "retrieved_memory",
            "content": "A memory summary.",
            "visibility": "model_visible",
            "metadata": {},
        }
    )
    return episode


def _first_required_key(stage: CognitionPromptStage) -> str:
    """Return the first required key for a stage output fixture.

    Args:
        stage: Cognition stage under test.

    Returns:
        First required normalized output key for that stage.
    """
    payload = _VALID_OUTPUT_PAYLOADS[stage]
    required_key = next(iter(payload))
    return required_key


def test_selector_returns_text_chat_variant_for_every_stage() -> None:
    """Current text chat episodes should select the sole active variant."""
    episode = _text_chat_episode()

    for stage in _APPROVED_STAGES:
        selection = select_cognition_prompt_variant(
            episode=episode,
            stage=stage,
        )

        assert selection == {
            "stage": stage,
            "variant": "text_chat_user_message",
            "prompt_key": f"{stage}.text_chat_user_message",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
        }


@pytest.mark.parametrize("output_mode", ("visible_reply", "think_only", "silent"))
def test_selector_accepts_stage_02_chat_output_modes(
    output_mode: OutputMode,
) -> None:
    """Stage 02 chat output modes should all keep the active prompt variant."""
    episode = _text_chat_episode(output_mode=output_mode)

    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l1_subconscious",
    )

    assert selection["variant"] == "text_chat_user_message"
    assert selection["output_mode"] == output_mode


@pytest.mark.parametrize(
    ("episode", "stage", "message"),
    (
        pytest.param(
            _reflection_episode(output_mode="scheduled_action_request"),
            "l1_subconscious",
            "output_mode",
            id="unsupported-reflection-output-mode",
        ),
        pytest.param(
            _retrieved_memory_text_episode(),
            "l1_subconscious",
            "input_sources",
            id="unsupported-input-sources",
        ),
        pytest.param(
            _text_chat_episode(output_mode="preview"),
            "l1_subconscious",
            "output_mode",
            id="unsupported-output-mode",
        ),
        pytest.param(
            _text_chat_episode(),
            "unknown_stage",
            "stage",
            id="unknown-stage",
        ),
    ),
)
def test_selector_rejects_unsupported_contracts(
    episode: CognitiveEpisode,
    stage: Any,
    message: str,
) -> None:
    """Unsupported selection inputs should fail closed."""
    with pytest.raises(CognitionPromptSelectionError, match=message):
        select_cognition_prompt_variant(episode=episode, stage=stage)


@pytest.mark.parametrize("stage", _APPROVED_STAGES)
def test_output_contract_accepts_current_normalized_shapes(
    stage: CognitionPromptStage,
) -> None:
    """Every approved stage should accept its current normalized output."""
    payload = dict(_VALID_OUTPUT_PAYLOADS[stage])

    validate_cognition_output_contract(stage=stage, payload=payload)


@pytest.mark.parametrize("stage", _APPROVED_STAGES)
def test_output_contract_rejects_missing_required_keys(
    stage: CognitionPromptStage,
) -> None:
    """Every approved stage should reject a missing required output key."""
    payload = dict(_VALID_OUTPUT_PAYLOADS[stage])
    missing_key = _first_required_key(stage)
    del payload[missing_key]

    with pytest.raises(CognitionOutputContractError, match=missing_key):
        validate_cognition_output_contract(stage=stage, payload=payload)


@pytest.mark.parametrize("stage", _APPROVED_STAGES)
def test_output_contract_rejects_wrong_required_value_types(
    stage: CognitionPromptStage,
) -> None:
    """Every approved stage should reject the wrong required value type."""
    payload = dict(_VALID_OUTPUT_PAYLOADS[stage])
    invalid_key = _first_required_key(stage)
    payload[invalid_key] = _WRONG_OUTPUT_VALUES[stage]

    with pytest.raises(CognitionOutputContractError, match=invalid_key):
        validate_cognition_output_contract(stage=stage, payload=payload)


def test_output_contract_rejects_unknown_stage() -> None:
    """Unknown stages should not share any normalized output contract."""
    with pytest.raises(CognitionOutputContractError, match="stage"):
        validate_cognition_output_contract(stage="unknown_stage", payload={})


@pytest.mark.parametrize(
    "stage",
    ("l2a_conscious_framing", "l2c1_judgment_synthesis"),
)
def test_output_contract_rejects_action_specs_outside_l2d(
    stage: CognitionPromptStage,
) -> None:
    """L2a and L2c must not become accidental action initializers."""

    payload = dict(_VALID_OUTPUT_PAYLOADS[stage])
    payload["action_specs"] = []

    with pytest.raises(CognitionOutputContractError, match="action_specs"):
        validate_cognition_output_contract(stage=stage, payload=payload)
