"""One-case live-LLM lifecycle evidence for the validation-only V2 core."""

import json
from pathlib import Path

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import (
    build_acquaintance_user_state,
    build_character_production_state,
    run_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    reset_validation_capture,
    validation_capture_snapshot,
    write_validation_capture,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
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


def _chain_input(
    message: str,
    *,
    episode_id: str,
    mutable_state: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one native V2 input for a live causal lifecycle case."""

    updated_at = "2026-07-14T00:00:00Z"
    character = build_character_production_state(updated_at=updated_at)
    semantic_text = message or "no new causal event"
    state = mutable_state or build_acquaintance_user_state(
        global_user_id="live-v2-user",
        updated_at=updated_at,
    )
    return {
        "schema_version": "cognition_core_input.v2",
        "episode": {
            "episode_id": episode_id,
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
            "semantic_scene": semantic_text,
            "semantic_temporal_context": "immediate",
        },
        "state_scope": "user",
        "mutable_state": state,
        "character_constraints": {
            "drives": character["drives"],
            "standards": character["standards"],
            "meaning_state": character["meaning_state"],
        },
        "evidence": [{
            "evidence_handle": "ev1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": f"episode:{episode_id}",
                "occurred_at": updated_at,
                "semantic_summary": semantic_text,
            },
            "semantic_text": semantic_text,
            "visible_to": ["cognition", "surface"],
        }],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "scene_context": {
            "channel_scope": "private",
            "character_role": "companion",
            "semantic_scene": semantic_text,
            "semantic_temporal_context": "immediate",
        },
    }


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
    payload = _chain_input(
        "",
        episode_id=f"live-v2-{case_id}",
    )
    episode = payload["episode"]
    message = _CASE_MESSAGES[case_id]
    reset_validation_capture(case_id)
    services = build_cognition_core_services()

    payload["episode"]["semantic_scene"] = message
    payload["evidence"][0]["semantic_text"] = message
    payload["evidence"][0]["evidence_ref"]["semantic_summary"] = message
    output = await run_cognition(payload, services)
    capture = validation_capture_snapshot()
    artifact_path = write_validation_capture()

    assert output["schema_version"] == "cognition_core_output.v2"
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
    reset_validation_capture(f"lifecycle-sequence-{case_id}")
    services = build_cognition_core_services()
    outputs = []
    mutable_state = None
    for phase, message in (
        ("baseline", _NEUTRAL_MESSAGE),
        ("begin", _CASE_MESSAGES[case_id]),
        ("sustain", _CASE_MESSAGES[case_id]),
        ("fade", _RESOLUTION_MESSAGE),
    ):
        payload = _chain_input(
            message,
            episode_id=f"live-v2-{case_id}-{phase}",
            mutable_state=mutable_state,
        )
        output = await run_cognition(payload, services)
        mutable_state = output["state_update"]["replacement_state"]
        outputs.append({"phase": phase, "output": output})
    negative_payload = _chain_input(
        _NEUTRAL_MESSAGE,
        episode_id=f"live-v2-{case_id}-negative-control",
        mutable_state=mutable_state,
    )
    negative_output = await run_cognition(negative_payload, services)
    outputs.append({"phase": "negative_control", "output": negative_output})
    capture = validation_capture_snapshot()
    artifact_path = write_validation_capture()

    assert all(
        entry["output"]["schema_version"] == "cognition_core_output.v2"
        for entry in outputs
    )
    assert capture is not None
    assert artifact_path.exists()
