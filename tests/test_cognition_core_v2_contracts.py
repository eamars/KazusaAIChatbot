"""Exact public V2 cognition and surface contract tests."""

import json
from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import (
    run_cognition,
    run_text_surface_planning,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    _episode_updated_at,
    _validate_output_mode,
)
from kazusa_ai_chatbot.cognition_core_v2.output_projection import (
    build_state_update,
    default_expression_policy,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    CognitionDiagnosticsV2,
    CognitionCoreServicesV2,
    CollapsedIntentionV2,
    EVIDENCE_SOURCE_QUESTION_IDS,
    EventComparisonResultV2,
    TextSurfaceServicesV2,
    VisualSurfaceServicesV2,
    validate_cognition_core_input,
    validate_cognition_core_output,
    validate_text_surface_input,
    validate_text_surface_output,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)

from llm_test_helpers import make_llm_call_config
from tests.cognition_core_v2_test_helpers import canonical_episode


NOW = "2026-07-14T00:00:00Z"


class _NoCallLLM:
    """Raise when a no-evidence episode reaches a model-owned stage."""

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages
        del config
        raise AssertionError("a grounded evidence-free episode must stay quiet")


class _FallbackSurfaceLLM:
    """Return a legacy surface fallback shape that the boundary must reject."""

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages
        del config
        return SimpleNamespace(content=json.dumps({"content": "legacy"}))


def _services() -> CognitionCoreServicesV2:
    """Build V2 service bindings for a deterministic no-call case."""

    return CognitionCoreServicesV2(
        llm=_NoCallLLM(),
        appraisal_config=make_llm_call_config("v2_appraisal"),
        goal_cognition_config=make_llm_call_config("v2_goal"),
        collapse_config=make_llm_call_config("v2_collapse"),
        action_selection_config=make_llm_call_config("v2_route"),
    )


def _input() -> dict[str, object]:
    """Build the exact V2 input without raw prompt-owned state fields."""

    character = build_character_production_state(updated_at=NOW)
    return {
        "schema_version": "cognition_core_input.v2",
        "episode": canonical_episode(
            episode_id="v2-contract-episode",
            current_global_user_id="v2-contract-user",
            content="quiet private greeting",
        ),
        "state_scope": "user",
        "mutable_state": build_acquaintance_user_state(
            global_user_id="v2-contract-user",
            updated_at=NOW,
        ),
        "character_constraints": {
            "drives": character["drives"],
            "standards": character["standards"],
            "meaning_state": character["meaning_state"],
        },
        "evidence": [],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "resolver_context": "resolver_status=idle",
        "scene_context": {
            "channel_scope": "private",
            "character_role": "companion",
            "semantic_scene": "quiet private greeting",
            "conversation_continuity": "No unresolved public commitment.",
            "semantic_temporal_context": "immediate",
        },
        "private_continuity_context": "I remain calmly attentive.",
    }


@pytest.mark.asyncio
async def test_v2_facade_returns_exact_one_scope_output() -> None:
    """The public facade returns one validated state update and no legacy shape."""

    output = await run_cognition(_input(), _services())
    validated = validate_cognition_core_output(output)

    assert validated["schema_version"] == "cognition_core_output.v2"
    assert validated["state_update"]["state_scope"] == "user"
    assert validated["intention"]["route"] == "silence"
    assert "admitted_bid" not in validated


def test_frozen_public_contract_fields_are_exact() -> None:
    """Prevent service, collapse, comparison, and diagnostic shape drift."""

    assert [field.name for field in fields(CognitionCoreServicesV2)] == [
        "llm",
        "appraisal_config",
        "goal_cognition_config",
        "collapse_config",
        "action_selection_config",
    ]
    assert [field.name for field in fields(TextSurfaceServicesV2)] == [
        "llm",
        "style_config",
        "content_plan_config",
        "preference_config",
    ]
    assert [field.name for field in fields(VisualSurfaceServicesV2)] == [
        "llm",
        "visual_config",
    ]
    assert set(CollapsedIntentionV2.__annotations__) == {
        "primary_branch_id",
        "supporting_branch_ids",
        "suppressed_branch_ids",
        "primary_bid",
        "supporting_bids",
        "competing_bids",
    }
    assert set(EventComparisonResultV2.__annotations__) == {
        "current_event_ref",
        "matched_entity_ref",
        "outcome",
        "evidence_refs",
    }
    assert set(CognitionDiagnosticsV2.__annotations__) == {
        "run_id",
        "stage_status",
        "selected_question_count",
        "dispatched_question_count",
        "selected_branch_count",
        "dispatched_branch_count",
        "completed_branch_count",
        "failed_branch_count",
        "overlap_ms",
        "dependency_wait_ms",
        "total_ms",
        "warnings",
    }


def test_expression_policy_uses_frozen_affect_and_branch_rules() -> None:
    """Derive intensity, directness, and two-emotion tone deterministically."""

    activations = [
        {
            "emotion_id": "fear",
            "score": 60,
            "phase": "active",
            "cause_status": "active",
            "trend": "rising",
        },
        {
            "emotion_id": "joy",
            "score": 60,
            "phase": "active",
            "cause_status": "active",
            "trend": "stable",
        },
    ]
    policy = default_expression_policy(
        "speech",
        [{"intensity": "high"}],
        selected_branch_id="autonomy_boundary",
        activations=activations,
    )

    assert policy["intensity"] == "strong"
    assert policy["directness"] == "direct"
    assert policy["emotional_tone"].startswith("joy (")
    assert "fear (" in policy["emotional_tone"]
    assert default_expression_policy(
        "speech",
        [{"intensity": "low"}],
        selected_branch_id="social_care",
    )["intensity"] == "restrained"
    assert default_expression_policy(
        "speech",
        [{"intensity": "moderate"}],
    )["intensity"] == "moderate"


def test_state_update_serializes_entities_and_activations_canonically() -> None:
    """Make replacement documents independent of parallel completion order."""

    previous = build_acquaintance_user_state(
        global_user_id="canonical-order-user",
        updated_at=NOW,
    )
    replacement = deepcopy(previous)
    replacement["goals"] = [
        {"entity_id": "goal:b", "created_at": "2026-07-14T00:00:01Z"},
        {"entity_id": "goal:c", "created_at": NOW},
        {"entity_id": "goal:a", "created_at": NOW},
    ]
    replacement["affect_activations"] = [
        {"emotion_id": "fear"},
        {"emotion_id": "joy"},
    ]

    update = build_state_update(previous, replacement)

    assert [
        row["entity_id"]
        for row in update["replacement_state"]["goals"]
    ] == ["goal:a", "goal:c", "goal:b"]
    assert [
        row["emotion_id"]
        for row in update["replacement_state"]["affect_activations"]
    ] == ["joy", "fear"]


def test_evidence_visibility_matches_its_source_question_ids() -> None:
    """Require exact source-owned semantic question visibility."""

    payload = _input()
    payload["evidence"] = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:visibility",
            "occurred_at": NOW,
            "semantic_summary": "a current grounded episode",
        },
        "semantic_text": "a current grounded episode",
        "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS["episode"]),
    }]

    validate_cognition_core_input(payload)
    payload["evidence"][0]["visible_to"] = ["cognition", "surface"]
    with pytest.raises(CognitionContractError):
        validate_cognition_core_input(payload)


@pytest.mark.asyncio
async def test_malformed_custom_episode_fails_before_any_model_call() -> None:
    """The public core boundary accepts only canonical CognitiveEpisode."""

    payload = _input()
    payload["episode"] = {
        "episode_id": "legacy-projection",
        "semantic_scene": "custom semantic map",
    }

    with pytest.raises(CognitionContractError):
        await run_cognition(payload, _services())


def test_relationship_context_uses_exact_native_relationship_contract() -> None:
    """Optional relationship context keeps native fields and owner exactness."""

    payload = _input()
    relationship = deepcopy(payload["mutable_state"]["relationship"])
    payload["relationship_context"] = relationship
    validate_cognition_core_input(payload)

    relationship["unexpected_axis"] = 50
    with pytest.raises(CognitionContractError):
        validate_cognition_core_input(payload)


def test_canonical_episode_timestamp_drives_native_state_time() -> None:
    """The core uses only canonical storage_timestamp_utc and emits UTC-Z."""

    episode = canonical_episode()
    episode["storage_timestamp_utc"] = "2026-07-14T00:01:02+00:00"

    assert _episode_updated_at(episode) == "2026-07-14T00:01:02Z"


def test_canonical_output_mode_restricts_visible_and_executable_routes() -> None:
    """Every frozen episode mode retains its route-authorization boundary."""

    scheduled = canonical_episode(output_mode="scheduled_action_request")
    _validate_output_mode(scheduled, "action")
    with pytest.raises(CognitionContractError):
        _validate_output_mode(scheduled, "speech")

    visible = canonical_episode(output_mode="visible_reply")
    _validate_output_mode(visible, "action")

    preview = canonical_episode(output_mode="preview")
    with pytest.raises(CognitionContractError):
        _validate_output_mode(preview, "speech")


def test_text_surface_rejects_private_state_and_branch_fields() -> None:
    """Keep L3 limited to the frozen semantic surface projection."""

    payload = {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            episode_id="v2-surface-contract",
            content="a quiet private greeting",
        ),
        "intention": {
            "route": "silence",
            "intention": "remain quiet",
            "target_roles": [],
            "reason": "there is no grounded need to speak",
        },
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "none",
            "emotional_tone": "calm",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "brief and natural",
        "character_voice_context": "reserved, analytical, and warm",
    }
    validate_text_surface_input(payload)

    raw_state_payload = deepcopy(payload)
    raw_state_payload["episode"]["mutable_state"] = {"goals": []}
    with pytest.raises(CognitionContractError):
        validate_text_surface_input(raw_state_payload)

    private_bid_payload = deepcopy(payload)
    private_bid_payload["primary_bid"] = {
        "branch_id": "ordinary_response",
        "motive": "respond naturally",
        "intention": "acknowledge the greeting",
        "desired_outcome": "maintain rapport",
        "permitted_detail": "a short greeting",
        "target_summaries": [],
        "expected_consequences": [],
    }
    with pytest.raises(CognitionContractError):
        validate_text_surface_input(private_bid_payload)


def test_text_surface_output_validates_every_list_entry() -> None:
    """Visible-boundary and addressee entries are bounded semantic strings."""

    payload = {
        "schema_version": "text_surface_output.v2",
        "content_plan": "acknowledge the current percept",
        "content_requirements": ["Preserve the current addressee."],
        "visible_boundaries": ["keep the response concise"],
        "addressee_plan": ["address the current participant"],
        "style_guidance": "natural",
        "selected_surface_intent": "acknowledge",
    }
    validate_text_surface_output(payload)

    payload["visible_boundaries"] = [""]
    with pytest.raises(CognitionContractError):
        validate_text_surface_output(payload)


@pytest.mark.asyncio
async def test_surface_stage_rejects_legacy_response_fallbacks() -> None:
    """Each surface model must emit its exact stage-local result fields."""

    input_payload = {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            episode_id="surface-exact-result",
            content="a visible greeting",
        ),
        "intention": {
            "route": "speech",
            "intention": "acknowledge the greeting",
            "target_roles": [],
            "reason": "the greeting is directly observed",
        },
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "calm",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "brief and natural",
        "character_voice_context": "reserved, analytical, and warm",
    }
    services = TextSurfaceServicesV2(
        llm=_FallbackSurfaceLLM(),
        style_config=make_llm_call_config("v2_style"),
        content_plan_config=make_llm_call_config("v2_content"),
        preference_config=make_llm_call_config("v2_preference"),
    )

    with pytest.raises(ValueError, match="fields are not exact"):
        await run_text_surface_planning(input_payload, services)


@pytest.mark.asyncio
async def test_facade_derives_relief_from_direct_threat_resolution() -> None:
    """Derive relief from the public trusted-fact lane without caller metadata."""

    payload = _input()
    state = payload["mutable_state"]
    state["threats"] = [{
        "entity_id": "threat:resolved-by-action",
        "description": "an immediate serious risk",
        "salience": 70,
        "role_refs": [],
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": "episode:threat",
            "occurred_at": NOW,
            "semantic_summary": "a serious risk was active",
        }],
        "created_at": NOW,
        "updated_at": NOW,
        "status": "active",
        "likelihood": 80,
        "expected_harm": 80,
        "uncertainty": 70,
        "controllability": 20,
        "coping_potential": 20,
        "residual_pressure": 80,
    }]
    payload["direct_facts"] = [{
        "fact_id": "fact:resolved-by-action",
        "producer": "action_result",
        "fact_kind": "threat_resolved",
        "target_refs": [{
            "scope": "user",
            "kind": "threat",
            "entity_id": "threat:resolved-by-action",
        }],
        "evidence_ref": {
            "source_kind": "action_result",
            "source_id": "action:resolved-threat",
            "occurred_at": NOW,
            "semantic_summary": "the action removed the serious risk",
        },
    }]

    output = await run_cognition(payload, _services())
    activations = output["state_update"]["replacement_state"][
        "affect_activations"
    ]
    relief = next(
        row for row in activations if row["emotion_id"] == "relief"
    )
    assert relief["primary_root"] == {
        "scope": "user",
        "kind": "threat",
        "entity_id": "threat:resolved-by-action",
    }
