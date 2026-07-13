"""One-case live-LLM lifecycle evidence for the validation-only V2 core."""

import json
from pathlib import Path

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import run_cognition_chain
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    reset_local_state,
    reset_validation_capture,
    validation_capture_snapshot,
    write_validation_capture,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_chain_services,
)

_FIXTURE_PATH = Path("tests/fixtures/cognition_core_v2_emotion_lifecycle_cases.json")
_CASE_MESSAGES = {
    "joy": "I finally completed the important goal I worked hard for.",
    "fear": "The building alarm is real and I do not know if we can get out safely.",
    "anger": "They deliberately blocked the work and ignored the agreed boundary.",
    "sadness": "Someone important is gone and the loss cannot be undone.",
    "disgust": "The food is contaminated and violates a basic boundary.",
    "surprise": "The result was completely unexpected and changed the situation.",
    "love_attachment": "I trust this person deeply and want to protect our closeness.",
    "compassion_empathy": "I can see they are hurting, and I want to support them.",
    "gratitude": "They made a costly effort to help me when I needed it.",
    "jealousy": "A rival is threatening an important exclusive relationship.",
    "envy": "They achieved a valued skill I want and might still be able to earn.",
    "pride": "I succeeded through my own sustained effort and met my standard.",
    "shame": "My identity and reputation are exposed as failing an important standard.",
    "guilt": "I caused harm through my own choice and need to repair it.",
    "embarrassment": "I made a small visible social mistake that was awkward but not harmful.",
    "curiosity": "There is a valuable question I can realistically learn how to answer.",
    "awe": "The scale and complexity of this phenomenon exceeds my usual model.",
    "nostalgia": "This memory connects me to a cherished past that has been lost.",
    "loneliness": "I want meaningful connection but nobody is available at the needed depth.",
    "relief": "The serious threat that was active has now materially decreased.",
    "ennui_existential_angst": "My purpose and agency feel low, and no viable goal is visible.",
}
_RESOLUTION_MESSAGE = (
    "The previously active cause is now resolved and no longer applies."
)
_NEUTRAL_MESSAGE = "Please describe the weather in one sentence."


def _cases() -> list[dict[str, object]]:
    """Load the approved lifecycle rows for individual live execution."""

    fixture_text = _FIXTURE_PATH.read_text(encoding="utf-8")
    rows = json.loads(fixture_text)
    return rows


def _chain_input() -> dict[str, object]:
    """Build a self-contained prompt-safe V1 input for a live V2 case."""

    payload = {
        "schema_version": "cognition_chain_input.v1",
        "episode": {
            "episode_id": "live-v2-case",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "live_response",
            "model_visible_percepts": [{
                "percept_id": "live-v2-percept",
                "input_source": "dialog_text",
                "content": "",
                "metadata_summary": [],
            }],
            "target_scope_summary": "live v2 validation scope",
            "origin_summary": "live causal lifecycle evidence",
        },
        "character": {
            "character_global_id": "live-v2-character",
            "name": "Validation Character",
            "description": "validation-only character",
            "gender": "unknown",
            "age": "unknown",
            "birthday": "unknown",
            "backstory": "none",
            "personality_brief": {},
            "boundary_profile": {},
            "linguistic_texture_profile": {},
            "mood": "neutral",
            "global_vibe": "calm",
        },
        "current_user": {
            "global_user_id": "live-v2-user",
            "display_name": "Validation User",
            "affinity": "neutral",
            "affinity_level": "known",
            "last_relationship_insight": "none",
            "memory_context": {
                "durable_profile_summary": "",
                "relationship_summary": "",
                "recent_commitments_summary": "",
                "known_preferences_summary": "",
            },
        },
        "current_event": {
            "user_input": "",
            "decontextualized_input": "",
            "indirect_speech_context": "",
            "referents": [],
            "media_observations": [],
            "reply_context_summary": "",
            "prompt_message_context_summary": "",
        },
        "scene": {
            "platform": "debug",
            "channel_type": "dm",
            "channel_topic": "",
            "local_time_context": {"time_of_day": "day"},
            "storage_timestamp_utc": "2026-07-13T00:00:00Z",
            "interaction_history_recent": [],
        },
        "conversation_context": {
            "conversation_progress": {},
            "promoted_reflection_context": {},
            "internal_monologue_residue_context": "",
            "previous_action_summary": "",
        },
        "evidence": {
            "rag_answer": "",
            "current_user_rag_bundle": "",
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "recall_evidence": [],
            "supervisor_trace": [],
        },
        "resolver": {
            "resolver_context": "",
            "pending_resume": "",
            "goal_progress": "",
            "recent_observations": [],
            "max_projected_observations": 3,
        },
        "available_actions": [{
            "capability": "speak",
            "available": True,
            "visibility": "public",
            "semantic_input_summary": "visible text surface",
            "output_kind": "semantic_action_request",
        }],
        "runtime_context": {
            "language_policy": "english",
            "visual_directives_enabled": False,
            "task_willingness_boundary_enabled": False,
            "max_action_requests": 1,
            "max_resolver_requests": 1,
            "background_work_output_char_limit": 1000,
        },
        "action_selection_context": {
            "coding_runs": [],
            "group_engagement_action_context": {},
        },
    }
    return payload


@pytest.mark.live_llm
@pytest.mark.asyncio
@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["case_id"])
async def test_live_v2_lifecycle_case_writes_complete_raw_capture(
    case: dict[str, object],
) -> None:
    """Run one causal scenario and preserve raw evidence for agent review."""

    case_id = case["case_id"]
    if not isinstance(case_id, str):
        raise TypeError("lifecycle case id must be text")
    payload = _chain_input()
    episode = payload["episode"]
    current_event = payload["current_event"]
    if not isinstance(episode, dict) or not isinstance(current_event, dict):
        raise TypeError("contract fixture must provide mutable episode and event")
    message = _CASE_MESSAGES[case_id]
    episode["episode_id"] = f"live-v2-{case_id}"
    episode["origin_summary"] = "live causal lifecycle evidence"
    current_event["user_input"] = message
    current_event["decontextualized_input"] = message
    await reset_local_state()
    reset_validation_capture(case_id)
    services = build_cognition_chain_services()

    output = await run_cognition_chain(payload, services)
    capture = validation_capture_snapshot()
    artifact_path = write_validation_capture()

    assert output["schema_version"] == "cognition_chain_output.v1"
    assert capture is not None
    assert capture["case_id"] == case_id
    assert artifact_path.exists()


@pytest.mark.live_llm
@pytest.mark.asyncio
@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["case_id"])
async def test_live_v2_lifecycle_sequence_writes_complete_raw_capture(
    case: dict[str, object],
) -> None:
    """Run baseline, causal lifecycle, and missing-root control in one scope."""

    case_id = case["case_id"]
    if not isinstance(case_id, str):
        raise TypeError("lifecycle case id must be text")
    await reset_local_state()
    reset_validation_capture(f"lifecycle-sequence-{case_id}")
    services = build_cognition_chain_services()
    outputs = []
    for phase, message in (
        ("baseline", _NEUTRAL_MESSAGE),
        ("begin", _CASE_MESSAGES[case_id]),
        ("sustain", _CASE_MESSAGES[case_id]),
        ("fade", _RESOLUTION_MESSAGE),
    ):
        payload = _chain_input()
        episode = payload["episode"]
        current_event = payload["current_event"]
        if not isinstance(episode, dict) or not isinstance(current_event, dict):
            raise TypeError("contract fixture must provide mutable episode and event")
        episode["episode_id"] = f"live-v2-{case_id}-{phase}"
        episode["origin_summary"] = "live causal lifecycle sequence"
        current_event["user_input"] = message
        current_event["decontextualized_input"] = message
        output = await run_cognition_chain(payload, services)
        outputs.append({"phase": phase, "output": output})
    await reset_local_state()
    negative_payload = _chain_input()
    negative_episode = negative_payload["episode"]
    negative_event = negative_payload["current_event"]
    if not isinstance(negative_episode, dict) or not isinstance(negative_event, dict):
        raise TypeError("contract fixture must provide mutable episode and event")
    negative_episode["episode_id"] = f"live-v2-{case_id}-negative-control"
    negative_episode["origin_summary"] = "live causal lifecycle sequence"
    negative_event["user_input"] = _NEUTRAL_MESSAGE
    negative_event["decontextualized_input"] = _NEUTRAL_MESSAGE
    negative_output = await run_cognition_chain(negative_payload, services)
    outputs.append({"phase": "negative_control", "output": negative_output})
    capture = validation_capture_snapshot()
    artifact_path = write_validation_capture()

    assert all(
        entry["output"]["schema_version"] == "cognition_chain_output.v1"
        for entry in outputs
    )
    assert capture is not None
    assert artifact_path.exists()
