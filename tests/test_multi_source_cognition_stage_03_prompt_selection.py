"""Stage 03 prompt-selection and cognition output-contract tests."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    OutputMode,
    build_text_chat_cognitive_episode,
)
from kazusa_ai_chatbot.cognition_chain_core.output_contracts import (
    CognitionOutputContractError,
    validate_cognition_output_contract,
)
from kazusa_ai_chatbot.cognition_chain_core.prompt_selection import (
    CognitionPromptSelectionError,
    CognitionPromptStage,
    select_cognition_prompt_variant,
)
from kazusa_ai_chatbot.cognition_chain_core.stages import l1 as l1_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2 as l2_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2c2 as l2c2_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l3 as l3_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock


_BACKGROUND_ARTIFACT_RESULT_VARIANT = (
    "background_artifact_result_ready_background_artifact_result"
)

_APPROVED_STAGES: tuple[CognitionPromptStage, ...] = (
    "l1_subconscious",
    "l2a_conscious_framing",
    "l2b_boundary_appraisal",
    "l2c1_judgment_synthesis",
    "l2c2_social_context_appraisal",
    "l2d_action_selection",
    "l3_style_agent",
    "l3_content_plan_agent",
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
    "l3_content_plan_agent": {
        "content_plan": {
            "semantic_content": "Answer the user directly.",
            "rendering": "One concise visible reply.",
        },
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

_RESULT_READY_PROMPT_HANDLERS = (
    (l1_module, "call_cognition_subconscious"),
    (l2_module, "call_cognition_consciousness"),
    (l2_module, "call_boundary_core_agent"),
    (l2_module, "call_judgment_core_agent"),
    (l2c2_module, "call_social_context_appraisal"),
    (l3_module, "call_style_agent"),
    (l3_module, "call_content_plan_agent"),
    (l3_module, "call_preference_adapter"),
    (l3_module, "call_visual_agent"),
)

_WRONG_OUTPUT_VALUES: dict[CognitionPromptStage, object] = {
    "l1_subconscious": [],
    "l2a_conscious_framing": [],
    "l2b_boundary_appraisal": "not a dict",
    "l2c1_judgment_synthesis": [],
    "l2c2_social_context_appraisal": [],
    "l2d_action_selection": "not a list",
    "l3_style_agent": [],
    "l3_content_plan_agent": "not a dict",
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
    turn_clock = build_turn_clock("2026-05-01 09:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="user_message:debug:direct:message-1",
        percept_id="user_message:debug:direct:message-1:dialog_text:0",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
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


def _background_artifact_result_episode() -> CognitiveEpisode:
    """Build a completed-artifact source episode fixture."""

    turn_clock = build_turn_clock("2026-05-01 09:00:00")
    episode = _text_chat_episode(output_mode="visible_reply")
    episode["episode_id"] = "background_artifact_result_ready:job-001"
    episode["trigger_source"] = "background_artifact_result_ready"
    episode["input_sources"] = ["background_artifact_result"]
    episode["storage_timestamp_utc"] = turn_clock["storage_timestamp_utc"]
    episode["local_time_context"] = turn_clock["local_time_context"]
    episode["percepts"] = [
        {
            "percept_id": "background_artifact_result_ready:job-001:result:0",
            "input_source": "background_artifact_result",
            "content": "Artifact ready: Fibonacci function snippet.",
            "visibility": "model_visible",
            "metadata": {
                "job_id": "job-001",
                "work_kind": "coding_snippet",
                "objective_summary": "Generate a Fibonacci function snippet.",
                "failure_summary": "",
            },
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


def test_selector_returns_background_artifact_result_variant_for_every_stage() -> None:
    """Completed background artifacts should use their own prompt variant."""

    episode = _background_artifact_result_episode()

    for stage in _APPROVED_STAGES:
        selection = select_cognition_prompt_variant(
            episode=episode,
            stage=stage,
        )

        assert selection == {
            "stage": stage,
            "variant": (
                "background_artifact_result_ready_background_artifact_result"
            ),
            "prompt_key": (
                f"{stage}."
                "background_artifact_result_ready_background_artifact_result"
            ),
            "trigger_source": "background_artifact_result_ready",
            "input_sources": ["background_artifact_result"],
            "output_mode": "visible_reply",
        }


def test_result_ready_variant_is_registered_in_stage_prompt_maps() -> None:
    """Every cognition stage handler must accept the selected result variant."""

    for module, handler_name in _RESULT_READY_PROMPT_HANDLERS:
        handler_source = inspect.getsource(getattr(module, handler_name))

        assert _BACKGROUND_ARTIFACT_RESULT_VARIANT in handler_source, (
            f"{module.__name__}.{handler_name} does not register "
            f"{_BACKGROUND_ARTIFACT_RESULT_VARIANT}"
        )


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
    """L2a and L2c must not become accidental semantic action selections."""

    payload = dict(_VALID_OUTPUT_PAYLOADS[stage])
    payload["action_specs"] = []

    with pytest.raises(CognitionOutputContractError, match="action_specs"):
        validate_cognition_output_contract(stage=stage, payload=payload)


@pytest.mark.parametrize(
    "stage",
    ("l2a_conscious_framing", "l2c1_judgment_synthesis"),
)
def test_output_contract_rejects_resolver_fields_outside_l2d(
    stage: CognitionPromptStage,
) -> None:
    """Only L2d may emit resolver capability decisions."""

    payload = dict(_VALID_OUTPUT_PAYLOADS[stage])
    payload["resolver_capability_requests"] = []

    with pytest.raises(
        CognitionOutputContractError,
        match="resolver_capability_requests",
    ):
        validate_cognition_output_contract(stage=stage, payload=payload)
