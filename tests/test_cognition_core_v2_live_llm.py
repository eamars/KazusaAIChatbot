"""One-case live-LLM lifecycle evidence for the validation-only V2 core."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import (
    build_acquaintance_user_state,
    build_character_production_state,
    run_cognition,
    run_text_surface_planning,
)
from kazusa_ai_chatbot.cognition_core_v2.surface import (
    run_visual_surface_planning,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    reset_validation_capture,
    validation_capture_snapshot,
    write_validation_capture,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
)
from kazusa_ai_chatbot.db.character import (
    get_character_cognition_state,
    replace_character_cognition_state,
)
from kazusa_ai_chatbot.db.users import (
    get_user_cognition_state,
    replace_user_cognition_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    _build_text_surface_services,
    _build_visual_surface_services,
)
from tests.llm_trace import write_llm_trace
from tests.live_llm_mongo import (
    live_db,
    seed_shared_documents,
    unique_owner_id,
)
from tests.cognition_core_v2_test_helpers import canonical_episode

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


class _CapturingSurfaceLLM:
    """Capture every stage-local surface request and raw model response."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, str]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> Any:
        response = await self.delegate.ainvoke(messages, config=config)
        self.calls.append({
            "system_prompt": str(getattr(messages[0], "content", "")),
            "human_payload": str(getattr(messages[-1], "content", "")),
            "raw_output": str(response.content),
        })
        return response


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
    state_scope: str = "user",
    trigger_source: str = "user_message",
    channel_scope: str = "private",
) -> dict[str, object]:
    """Build one native V2 input for a live causal lifecycle case."""

    updated_at = "2026-07-14T00:00:00Z"
    character = build_character_production_state(updated_at=updated_at)
    semantic_text = message or "no new causal event"
    state = mutable_state
    if state is None:
        state = build_acquaintance_user_state(
            global_user_id="live-v2-user",
            updated_at=updated_at,
        )
    return {
        "schema_version": "cognition_core_input.v2",
        "episode": canonical_episode(
            episode_id=episode_id,
            trigger_source=trigger_source,
            output_mode=(
                "think_only"
                if trigger_source == "internal_thought"
                else "visible_reply"
            ),
            current_global_user_id=str(
                state.get("owner_user_id", "live-v2-user")
            ),
            content=semantic_text,
        ),
        "state_scope": state_scope,
        "mutable_state": state,
        "character_constraints": {
            "drives": character["drives"],
            "standards": character["standards"],
            "meaning_state": character["meaning_state"],
        },
        "evidence": [{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": f"episode:{episode_id}",
                "occurred_at": updated_at,
                "semantic_summary": semantic_text,
            },
            "semantic_text": semantic_text,
            "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS["episode"]),
        }],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "resolver_context": "resolver_status=idle",
        "scene_context": {
            "channel_scope": channel_scope,
            "character_role": "companion",
            "semantic_scene": semantic_text,
            "conversation_continuity": "Continue only the current live case.",
            "semantic_temporal_context": "immediate",
        },
        "private_continuity_context": "I should remain grounded in this case.",
    }


async def _prepare_user_state(
    live_db: Any,
    request: pytest.FixtureRequest,
) -> tuple[str, dict[str, object]]:
    """Create one owner-isolated user row from the validated shared seed."""

    await seed_shared_documents(live_db)
    owner_id = unique_owner_id(request.node.nodeid)
    seed = await live_db.user_profiles.find_one(
        {"global_user_id": "seed-s2-acquaintance"},
    )
    if seed is None:
        raise AssertionError("the shared acquaintance seed is missing")
    state = deepcopy(seed["cognition_state"])
    state["owner_user_id"] = owner_id
    state["relationship"]["other_user_id"] = owner_id
    state["relationship"]["relationship_id"] = (
        f"relationship:user:{owner_id}"
    )
    document = {"global_user_id": owner_id, "cognition_state": state}
    await live_db.user_profiles.insert_one(document)
    loaded_state = await get_user_cognition_state(owner_id)
    return owner_id, loaded_state


async def _run_live_case(
    case_id: str,
    live_db: Any,
    request: pytest.FixtureRequest,
    *,
    message: str,
    trigger_source: str = "user_message",
    channel_scope: str = "private",
) -> None:
    """Run one guarded V2 case and persist its validated replacement state."""

    owner_id, state = await _prepare_user_state(live_db, request)
    try:
        payload = _chain_input(
            message,
            episode_id=f"live-v2-{case_id}",
            mutable_state=state,
            trigger_source=trigger_source,
            channel_scope=channel_scope,
        )
        reset_validation_capture(case_id)
        try:
            output = await run_cognition(
                payload,
                build_cognition_core_services(),
            )
        except Exception:
            write_validation_capture()
            raise
        capture = validation_capture_snapshot()
        artifact_path = write_validation_capture()
        replacement = output["state_update"]["replacement_state"]
        await replace_user_cognition_state(owner_id, replacement)
        reloaded = await get_user_cognition_state(owner_id)

        assert output["schema_version"] == "cognition_core_output.v2"
        assert output["state_update"]["state_scope"] == "user"
        assert reloaded["owner_user_id"] == owner_id
        assert capture is not None
        assert artifact_path.exists()
    finally:
        await live_db.user_profiles.delete_one(
            {"global_user_id": owner_id},
        )


async def _run_character_case(
    case_id: str,
    live_db: Any,
    *,
    message: str,
) -> None:
    """Run one character-scoped case and restore the singleton exactly."""

    await seed_shared_documents(live_db)
    snapshot = await get_character_cognition_state()
    try:
        payload = _chain_input(
            message,
            episode_id=f"live-v2-{case_id}",
            mutable_state=deepcopy(snapshot),
            state_scope="character",
            trigger_source="internal_thought",
        )
        reset_validation_capture(case_id)
        output = await run_cognition(payload, build_cognition_core_services())
        capture = validation_capture_snapshot()
        artifact_path = write_validation_capture()
        await replace_character_cognition_state(
            output["state_update"]["replacement_state"],
        )

        assert output["schema_version"] == "cognition_core_output.v2"
        assert output["state_update"]["state_scope"] == "character"
        assert capture is not None
        assert artifact_path.exists()
    finally:
        await replace_character_cognition_state(snapshot)


async def _run_lifecycle_case(
    case_id: str,
    live_db: Any,
    request: pytest.FixtureRequest,
) -> None:
    """Run baseline, natural cause, sustain, fade, and negative controls."""

    owner_id, mutable_state = await _prepare_user_state(live_db, request)
    reset_validation_capture(f"lifecycle-sequence-{case_id}")
    services = build_cognition_core_services()
    outputs: list[dict[str, object]] = []
    try:
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
            await replace_user_cognition_state(owner_id, mutable_state)
            outputs.append(output)

        negative_payload = _chain_input(
            _NEUTRAL_MESSAGE,
            episode_id=f"live-v2-{case_id}-negative-control",
            mutable_state=mutable_state,
        )
        negative_output = await run_cognition(negative_payload, services)
        outputs.append(negative_output)
        capture = validation_capture_snapshot()
        artifact_path = write_validation_capture()

        assert all(
            output["schema_version"] == "cognition_core_output.v2"
            for output in outputs
        )
        assert capture is not None
        assert artifact_path.exists()
    finally:
        await live_db.user_profiles.delete_one(
            {"global_user_id": owner_id},
        )


async def _run_cross_model_case(
    live_db: Any,
    request: pytest.FixtureRequest,
) -> None:
    """Run one ambiguous episode through two configured model routes."""

    owner_id, state = await _prepare_user_state(live_db, request)
    services = build_cognition_core_services()
    primary_model = services.appraisal_config.model
    comparison_model = services.action_selection_config.model
    if primary_model == comparison_model:
        raise AssertionError(
            "COGNITION_LLM and BOUNDARY_CORE_LLM must use distinct models"
        )
    try:
        outputs: list[dict[str, object]] = []
        for index, service_set in enumerate(
            (
                services,
                replace(
                    services,
                    appraisal_config=services.action_selection_config,
                    goal_cognition_config=services.action_selection_config,
                    collapse_config=services.action_selection_config,
                ),
            )
        ):
            payload = _chain_input(
                "The same ambiguous event could mean support or criticism;"
                " ask what the speaker intends before judging it.",
                episode_id=f"live-v2-cross-model-{index}",
                mutable_state=deepcopy(state),
            )
            reset_validation_capture(f"cross-model-{index}")
            output = await run_cognition(payload, service_set)
            outputs.append(output)
            capture = validation_capture_snapshot()
            artifact_path = write_validation_capture()
            assert capture is not None
            assert artifact_path.exists()

        assert all(
            output["schema_version"] == "cognition_core_output.v2"
            for output in outputs
        )
    finally:
        await live_db.user_profiles.delete_one(
            {"global_user_id": owner_id},
        )


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_v2_text_surface_stage_contracts_live_llm() -> None:
    """Exercise sibling text/visual L3 schemas and preserve raw outputs."""

    text_services = _build_text_surface_services()
    text_llm = _CapturingSurfaceLLM(text_services.llm)
    text_services = replace(text_services, llm=text_llm)
    visual_services = _build_visual_surface_services()
    visual_llm = _CapturingSurfaceLLM(visual_services.llm)
    visual_services = replace(visual_services, llm=visual_llm)
    payload = {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            episode_id="live-v2-surface-contracts",
            content="I finished the difficult task and want a brief reply.",
        ),
        "intention": {
            "route": "speech",
            "intention": "acknowledge the completed effort",
            "target_roles": [],
            "reason": "the completed effort is directly observed",
        },
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "joy (currently active, stable)",
            "intensity": "moderate",
            "directness": "balanced",
        },
        "semantic_affect": [{
            "emotion": "joy",
            "phase": "currently active",
            "intensity": "moderate",
            "trend": "stable",
            "cause_summary": "the difficult task was completed",
        }],
        "permitted_action_results": [],
        "interaction_style_context": "brief and natural",
        "character_voice_context": "reserved, analytical, vivid, and warm",
    }

    output = await run_text_surface_planning(payload, text_services)
    visual_output = await run_visual_surface_planning(
        payload,
        visual_services,
    )
    artifact_path = write_llm_trace(
        "cognition_core_v2_stage_2",
        "v2_text_surface_stage_contracts",
        {
            "input": payload,
            "text_calls": text_llm.calls,
            "visual_calls": visual_llm.calls,
            "output": output,
            "visual_output": visual_output,
            "judgment": "three text schemas and one terminal visual schema passed",
        },
    )

    assert len(text_llm.calls) == 3
    assert len(visual_llm.calls) == 1
    assert output["visible_boundaries"]
    assert output["addressee_plan"]
    assert output["content_requirements"]
    assert "character_voice_context" not in output
    assert visual_output["visual_directives"]
    assert artifact_path.exists()


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

    payload["episode"]["percepts"][0]["content"] = message
    payload["evidence"][0]["semantic_text"] = message
    payload["evidence"][0]["evidence_ref"]["semantic_summary"] = message
    output = await run_cognition(payload, services)
    capture = validation_capture_snapshot()
    artifact_path = write_validation_capture()

    assert output["schema_version"] == "cognition_core_output.v2"
    assert output["private_monologue"]
    assert output["private_monologue"] != output["selected_bid_reason"]
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


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_joy_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural joy lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("joy", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_fear_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural fear lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("fear", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_anger_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural anger lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("anger", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_sadness_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural sadness lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("sadness", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_disgust_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural disgust lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("disgust", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_surprise_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural surprise lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("surprise", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_love_attachment_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural attachment lifecycle through the guarded core."""

    await _run_lifecycle_case("love_attachment", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_compassion_empathy_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural compassion lifecycle through the guarded core."""

    await _run_lifecycle_case("compassion_empathy", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_gratitude_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural gratitude lifecycle through the guarded core."""

    await _run_lifecycle_case("gratitude", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_jealousy_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural jealousy lifecycle through the guarded core."""

    await _run_lifecycle_case("jealousy", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_envy_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural envy lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("envy", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_pride_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural pride lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("pride", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_shame_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural shame lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("shame", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_guilt_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural guilt lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("guilt", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_embarrassment_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural embarrassment lifecycle through the core."""

    await _run_lifecycle_case("embarrassment", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_curiosity_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural curiosity lifecycle through the guarded core."""

    await _run_lifecycle_case("curiosity", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_awe_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural awe lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("awe", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_nostalgia_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural nostalgia lifecycle through the guarded core."""

    await _run_lifecycle_case("nostalgia", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_loneliness_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural loneliness lifecycle through the guarded core."""

    await _run_lifecycle_case("loneliness", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_relief_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural relief lifecycle through the guarded V2 core."""

    await _run_lifecycle_case("relief", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_ennui_existential_angst_lifecycle_live_llm(live_db, request) -> None:
    """Exercise the natural existential-ennui lifecycle through the core."""

    await _run_lifecycle_case("ennui_existential_angst", live_db, request)


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_jealousy_scoped_appraisal_live_llm(live_db, request) -> None:
    """Capture a jealousy appraisal with only its guarded user scope."""

    await _run_live_case(
        "jealousy-scoped-appraisal",
        live_db,
        request,
        message=_CASE_MESSAGES["jealousy"],
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_guilt_scoped_appraisal_live_llm(live_db, request) -> None:
    """Capture a guilt appraisal with only its guarded user scope."""

    await _run_live_case(
        "guilt-scoped-appraisal",
        live_db,
        request,
        message=_CASE_MESSAGES["guilt"],
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_relief_scoped_appraisal_live_llm(live_db, request) -> None:
    """Capture a relief appraisal with only its guarded user scope."""

    await _run_live_case(
        "relief-scoped-appraisal",
        live_db,
        request,
        message=_CASE_MESSAGES["relief"],
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_ambiguous_user_meaning_uses_semantic_lane_live_llm(
    live_db,
    request,
) -> None:
    """Capture an ambiguous message that requires semantic clarification."""

    await _run_live_case(
        "ambiguous-user-meaning",
        live_db,
        request,
        message=(
            "The same ambiguous event could mean support or criticism; ask"
            " what the speaker intends before judging it."
        ),
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_jealousy_full_pipeline_live_llm_db(live_db, request) -> None:
    """Capture a complete jealousy pipeline with guarded persistence."""

    await _run_live_case(
        "jealousy-full-pipeline",
        live_db,
        request,
        message=_CASE_MESSAGES["jealousy"],
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_guilt_full_pipeline_live_llm_db(live_db, request) -> None:
    """Capture a complete guilt pipeline with guarded persistence."""

    await _run_live_case(
        "guilt-full-pipeline",
        live_db,
        request,
        message=_CASE_MESSAGES["guilt"],
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_relief_full_pipeline_live_llm_db(live_db, request) -> None:
    """Capture a complete relief pipeline with guarded persistence."""

    await _run_live_case(
        "relief-full-pipeline",
        live_db,
        request,
        message=_CASE_MESSAGES["relief"],
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_group_user_scope_full_pipeline_live_llm_db(live_db, request) -> None:
    """Capture group-origin cognition while retaining the user scope."""

    await _run_live_case(
        "group-user-scope",
        live_db,
        request,
        message="A group discussion created a clear request for my view.",
        trigger_source="user_message",
        channel_scope="group",
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_media_observation_full_pipeline_live_llm_db(
    live_db,
    request,
) -> None:
    """Capture a typed media observation in a guarded user scope."""

    await _run_live_case(
        "media-observation",
        live_db,
        request,
        message="The attached image shows a damaged bridge after the storm.",
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_resolver_recurrence_full_pipeline_live_llm_db(
    live_db,
    request,
) -> None:
    """Capture recurrence while retaining the originating user owner."""

    await _run_live_case(
        "resolver-recurrence",
        live_db,
        request,
        message="Continue the unresolved question using the current evidence.",
        trigger_source="user_message",
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_self_cognition_character_scope_live_llm_db(live_db) -> None:
    """Capture self-cognition against the restored character singleton."""

    await _run_character_case(
        "self-cognition-character-scope",
        live_db,
        message="Review the current purpose and choose one grounded next step.",
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_accepted_task_result_full_pipeline_live_llm_db(
    live_db,
    request,
) -> None:
    """Capture accepted-task result re-entry through the user scope."""

    await _run_live_case(
        "accepted-task-result",
        live_db,
        request,
        message="The accepted task completed and its bounded result is ready.",
        trigger_source="accepted_task_result_ready",
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
async def test_same_ambiguous_events_report_cross_model_variance_live_llm(
    live_db,
    request,
) -> None:
    """Capture the same ambiguous event through both configured model routes."""

    await _run_cross_model_case(live_db, request)
