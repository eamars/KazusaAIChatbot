"""V3 engine output parity against the V2 external contracts.

Each test runs the engine over the canonical connector-mapping fixture with
the scripted invoker and then validates one externally visible surface of the
result through the unchanged V2 contract entry points.
"""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.cognition_shared.contracts import (
    validate_action_bid,
    validate_cognition_core_output,
    validate_relational_willingness,
)
from kazusa_ai_chatbot.cognition_core_v3 import run_cognition
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    attach_dialog_semantic_projection,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    build_text_surface_input_from_global_state,
)
from tests.integration.cognition_core_v3.conftest import (
    ScriptedLLMInvoker,
    default_scripted_responses,
    episode_evidence_handle,
    make_v3_services,
)
from tests.test_cognition_chain_connector_mapping import _global_state

REQUIRED_OUTPUT_KEYS = frozenset(
    {
        "schema_version",
        "intention",
        "goal_continuation_ref",
        "supporting_bids",
        "state_update",
        "affect_projection",
        "action_requests",
        "resolver_requests",
        "goal_resolution",
        "resolver_pending_resolution",
        "resolver_goal_progress",
        "resolver_progress",
        "selected_bid_reason",
        "private_monologue",
        "expression_policy",
        "diagnostics",
    }
)

STATE_UPDATE_KEYS = frozenset(
    {
        "state_scope",
        "owner_key",
        "expected_previous_state",
        "replacement_state",
        "comparison_results",
        "changed_paths",
    }
)

DIAGNOSTICS_KEYS = frozenset(
    {
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
)

ENGINE_STAGE_NAMES = (
    "input_validation",
    "deterministic_preliminary",
    "semantic_appraisal",
    "final_reduction",
    "branch_cognition",
    "workspace_collapse",
    "action_planning",
)


@pytest.mark.asyncio
async def test_run_cognition_output_satisfies_v2_core_validator(
    cognition_payload, v3_services
):
    """The assembled output passes the V2 core output validator."""
    output = await run_cognition(cognition_payload, v3_services)
    validated = validate_cognition_core_output(output)

    assert validated["schema_version"] == "cognition_core_output.v2"
    assert REQUIRED_OUTPUT_KEYS.issubset(validated.keys())


@pytest.mark.asyncio
async def test_admitted_bid_passes_v2_action_bid_contract(
    cognition_payload, v3_services
):
    """The admitted bid and its relational decision pass V2 validators."""
    output = await run_cognition(cognition_payload, v3_services)
    episode_handle = next(
        row["evidence_handle"]
        for row in cognition_payload["evidence"]
        if row["evidence_ref"]["source_kind"] == "episode"
    )

    admitted_bid = output["admitted_bid"]
    validate_action_bid(admitted_bid)

    assert admitted_bid["branch_id"] == "ordinary_response"
    assert admitted_bid["target_roles"] == []
    assert admitted_bid["evidence_handles"] == [episode_handle]
    validate_relational_willingness(
        admitted_bid["relational_willingness"],
        evidence_handles={episode_handle},
    )


@pytest.mark.asyncio
async def test_state_update_and_diagnostics_carry_v2_shapes(
    cognition_payload, v3_services
):
    """State update and diagnostics expose the exact V2 field sets."""
    output = await run_cognition(cognition_payload, v3_services)

    state_update = output["state_update"]
    assert set(state_update.keys()) == STATE_UPDATE_KEYS
    assert state_update["state_scope"] == "user"

    diagnostics = output["diagnostics"]
    assert set(diagnostics.keys()) == DIAGNOSTICS_KEYS
    stage_status = diagnostics["stage_status"]
    assert tuple(stage_status[name] for name in ENGINE_STAGE_NAMES) == (
        "completed",
    ) * len(ENGINE_STAGE_NAMES)


@pytest.mark.asyncio
async def test_required_selection_nested_roles_reach_unchanged_dialog_input():
    """Cold G1a preserves its full selection draft and fixed role bindings."""

    state = _global_state()
    input_operation = {
        "operation": "The user chooses a respectful response boundary.",
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }
    state["cognitive_episode"] = attach_dialog_semantic_projection(
        state["cognitive_episode"],
        "The current user asks the current character to choose a response boundary.",
        input_operation,
    )
    payload = build_cognition_input_from_global_state(state)
    episode_handle = episode_evidence_handle(payload)

    selected_operation = {
        **input_operation,
        "operation": "The current user offers the current character a pause.",
    }
    selection_draft = {
        "selection": "Choose the grounded pause boundary.",
        "selected_response_operation": {
            "operation": selected_operation["operation"],
        },
        "reason": "The explicit episode operation requires one choice.",
        "private_monologue": "Keep the selected role boundary exact.",
        "target_role_handles": [],
        "evidence_handles": [episode_handle],
        "expected_consequences": [
            "The surface receives the validated selected operation."
        ],
        "confidence": "medium",
        "relational_willingness": {
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "current_user_relationship_state": "not_applicable",
            "reason": "当前选择不涉及关系敏感性。",
            "evidence_handles": [episode_handle],
        },
    }
    responses = default_scripted_responses(episode_handle)
    responses["G1a"] = json.dumps(selection_draft)
    output = await run_cognition(
        payload,
        make_v3_services(ScriptedLLMInvoker(defaults=responses)),
    )

    state["cognition_core_output"] = output
    surface_input = build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and direct",
    )

    assert output["intention"]["selected_response_operation"] == selected_operation
    assert surface_input["intention"]["selected_response_operation"] == (
        selected_operation
    )
    assert surface_input["selected_response_operation"] == selected_operation
    assert output["relational_willingness"]["applicability"] == (
        "not_relationship_sensitive"
    )


@pytest.mark.asyncio
async def test_targetless_group_can_silence_or_emit_grounded_reply_proposal():
    """Targetless group P1 retains either canonical semantic response route."""

    def targetless_payload():
        state = _global_state()
        state["channel_type"] = "group"
        state["global_user_id"] = ""
        state["platform_user_id"] = ""
        state["platform_message_id"] = ""
        episode = state["cognitive_episode"]
        assert isinstance(episode, dict)
        episode["trigger_source"] = "self_cognition"
        target_scope = episode["target_scope"]
        assert isinstance(target_scope, dict)
        target_scope["channel_type"] = "group"
        target_scope["current_global_user_id"] = None
        target_scope["current_platform_user_id"] = None
        character_state = state["character_cognition_state"]
        assert isinstance(character_state, dict)
        return build_cognition_input_from_global_state(
            state,
            mutable_state=character_state,
            character_state=character_state,
        )

    silent_payload = targetless_payload()
    silent_handle = episode_evidence_handle(silent_payload)
    silent_responses = default_scripted_responses(silent_handle)
    silent_responses["P1"] = json.dumps({
        "action_requests": [],
        "resolver_requests": [],
        "goal_resolution": "blocked",
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
        "self_cognition_response": {
            "decision": "stay_silent",
            "evidence_handles": [],
            "semantic_target_handle": "",
            "participation_basis": "",
            "response_goal": "",
            "reason": "The current scene has no grounded reason to enter.",
        },
    })
    silent_output = await run_cognition(
        silent_payload,
        make_v3_services(ScriptedLLMInvoker(defaults=silent_responses)),
    )

    visible_payload = targetless_payload()
    visible_handle = episode_evidence_handle(visible_payload)
    visible_responses = default_scripted_responses(visible_handle)
    visible_responses["P1"] = json.dumps({
        "action_requests": [],
        "resolver_requests": [],
        "goal_resolution": "blocked",
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
        "self_cognition_response": {
            "decision": "propose_visible_reply",
            "evidence_handles": [visible_handle],
            "semantic_target_handle": "current_group_scene",
            "participation_basis": "grounded_scene_intervention",
            "response_goal": "Offer a grounded reply to the current group scene.",
            "reason": "The current episode supplies direct group-scene evidence.",
        },
    })
    visible_output = await run_cognition(
        visible_payload,
        make_v3_services(ScriptedLLMInvoker(defaults=visible_responses)),
    )

    assert silent_output["self_cognition_response_contract_status"] == "valid"
    assert silent_output["self_cognition_response"]["decision"] == "stay_silent"
    assert silent_output["intention"]["route"] == "silence"
    assert visible_output["self_cognition_response_contract_status"] == "valid"
    assert visible_output["self_cognition_response"] == {
        "decision": "propose_visible_reply",
        "evidence_handles": [visible_handle],
        "semantic_target_handle": "current_group_scene",
        "participation_basis": "grounded_scene_intervention",
        "response_goal": "Offer a grounded reply to the current group scene.",
        "reason": "The current episode supplies direct group-scene evidence.",
    }
    assert visible_output["intention"]["route"] == "speech"
