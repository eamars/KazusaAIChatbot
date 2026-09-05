"""Focused deterministic coverage for the canonical phase-one boundary."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v3.appraisal import bind_axis_changes
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CANONICAL_A1_FAMILIES,
    CANONICAL_A2_FAMILIES,
    CANONICAL_FAMILY_AXES,
    CanonicalAppraisal,
    CognitionChainServicesV3,
    validate_response_plan_contract_variant,
)
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    CanonicalContractError,
    _validate_canonical_input,
    _validate_plan,
    bind_protected_chain_records,
    reset_protected_chain_records,
    run_cognition,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    _project_capabilities,
    build_canonical_appraisal_question,
    build_canonical_plan_question,
    build_canonical_turn_workspace,
    build_turn_workspace_stage_contracts,
)
from kazusa_ai_chatbot.cognition_episode import build_user_message_episode
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    PENDING_TASK_CONTINUATION_VERSION,
    RESOLVER_PENDING_CONTINUATION_VERSION,
    RESOLVER_PENDING_RESUME_VERSION,
    project_pending_resume_for_prompt,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContractError,
    CognitionExecutionError,
    validate_terminal_text_seed,
    validate_text_surface_output,
)
from kazusa_ai_chatbot.cognition_shared.emotion_derivation import (
    derive_persistent_emotion_activations,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMThinkingConfig
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    _build_scheduled_authority_proposal,
    _project_output_to_global_state,
)

_CODE_SPAN_PATTERN = re.compile(r"`[^`]+`")
_HAN_PATTERN = re.compile(r"[\u3400-\u9fff]")
_MULTIWORD_ENGLISH_PATTERN = re.compile(
    r"\b[A-Za-z]+(?:[ \t]+[A-Za-z]+){2,}\b"
)




def _input() -> dict[str, object]:
    timestamp = "2026-07-14T00:00:00Z"
    episode = build_user_message_episode(
        episode_id="deterministic-cognition-episode",
        origin={
            "platform": "debug",
            "platform_message_id": "deterministic-message",
            "active_turn_platform_message_ids": ["deterministic-message"],
            "active_turn_conversation_row_ids": [],
            "privacy_scope": "private",
            "delivery_permission_ref": "",
            "created_at": timestamp,
        },
        target_scope={
            "platform": "debug",
            "platform_channel_id": "deterministic-channel",
            "channel_type": "private",
            "current_platform_user_id": "platform-user",
            "current_global_user_id": "user-1",
            "current_display_name": "Test User",
            "target_addressed_user_ids": [],
            "target_broadcast": False,
        },
        dialog_percept={
            "schema_version": "percept.v1",
            "percept_kind": "dialog",
            "source_kind": "dialog",
            "source_id": "deterministic-percept",
            "content": {"semantic_text": "the current observation"},
            "observed_at": timestamp,
        },
        media_percepts=[],
        evidence_refs=[],
        local_time_context={
            "current_local_datetime": "2026-07-14 12:00",
            "current_local_weekday": "Tuesday",
        },
        created_at=timestamp,
        debug_controls={},
    )
    mutable_state = build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=timestamp,
    )
    character_state = build_character_production_state(updated_at=timestamp)
    return {
        "schema_version": "cognition_input.v3",
        "response_plan_contract_variant": "fresh_ordinary",
        "episode": episode,
        "state_scope": "user",
        "mutable_state": mutable_state,
        "character_constraints": {
            "drives": character_state["drives"],
            "meaning_state": character_state["meaning_state"],
            "standards": character_state["standards"],
        },
        "character_identity_context": {
            "name": "Test Character",
            "personality": "grounded and boundary-aware",
        },
        "character_operational_context": {},
        "overused_moves": [],
        "evidence": [{
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "deterministic-percept",
                "occurred_at": timestamp,
                "semantic_summary": "the current observation",
            },
            "semantic_text": "the current observation",
            "authority": "current_event",
        }],
        "direct_facts": [],
        "available_actions": [{
            "action_kind": "accepted_task_status_check",
            "description": "check an accepted task status",
            "context_ref": "",
            "target_roles": [],
        }],
        "available_resolver_capabilities": [
            {"capability": "human_clarification"},
            {"capability": "approval_preparation"},
            {"capability": "self_goal_resolution"},
            {"capability": "task_resolution_request"},
        ],
        "scene_context": {
            "channel_scope": "private",
            "character_role": "当前角色",
            "current_user_role": "当前用户",
            "character_sleep_phase": "清醒时段",
            "semantic_scene": "the current observation",
            "public_group_scene": "",
            "conversation_continuity": "",
            "semantic_temporal_context": "即时",
        },
    }


def _services(invoker: object) -> CognitionChainServicesV3:
    config = LLMCallConfig(
        stage_name="cognition_core_v3.chain",
        route_name="test",
        base_url="http://test",
        api_key="test",
        model="test",
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_completion_tokens=8192,
        presence_penalty=None,
        thinking=LLMThinkingConfig(enabled=False),
        context_window_tokens=50176,
    )
    return CognitionChainServicesV3(
        llm=invoker,
        chain_lane=config,
    )


def test_a1_packet_projects_state_without_authored_guidance() -> None:
    """The HumanMessage packet carries state and contracts, not prompt prose."""

    payload = _input()
    workspace = build_canonical_turn_workspace(
        episode=payload["episode"],
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
        mutable_state=payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        identity_context=payload["character_identity_context"],
        available_actions=payload["available_actions"],
        available_resolvers=payload["available_resolver_capabilities"],
        overused_moves=payload["overused_moves"],
        response_plan_contract_variant="fresh_ordinary",
    )
    packet = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A1",
    )
    assert "guidance" not in packet
    assert packet["current_observation"]


def test_canonical_stage_packets_are_handleless_and_disjoint() -> None:
    payload = _input()
    workspace = build_canonical_turn_workspace(
        episode=payload["episode"],
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
        mutable_state=payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        identity_context=payload["character_identity_context"],
        available_actions=payload["available_actions"],
        available_resolvers=payload["available_resolver_capabilities"],
        overused_moves=payload["overused_moves"],
        response_plan_contract_variant="fresh_ordinary",
    )
    packets = build_turn_workspace_stage_contracts(workspace=workspace)
    assert packets["A1"]["output_contract"]["required_fields"] == list(CANONICAL_A1_FAMILIES)
    assert packets["A2"]["output_contract"]["required_fields"] == list(CANONICAL_A2_FAMILIES)
    assert packets["G"]["output_contract"]["required_fields"] == [
        "active_character_goal", "relational_willingness", "private_monologue"
    ]
    assert packets["P"]["output_contract"]["required_fields"] == [
        "goal_resolution", "response_goal", "action_requests",
        "resolver_requests", "epistemic_boundary",
    ]
    assert set(packets["P"]) == {
        "stage", "goal", "current_observation",
        "direct_facts", "participant_continuity", "continuation_state",
        "capabilities", "output_contract",
    }
    rendered = json.dumps(packets, ensure_ascii=False)
    assert all(token not in rendered for token in ("source_id", "entity_id", "target_path"))
    assert not any(
        token in rendered
        for token in ("e1", "g1", "b1", "r1", "ce1", "ct1", "ck1")
    )
    assert sum(len(axes) for axes in CANONICAL_FAMILY_AXES.values()) == 51
    assert all(
        len(set(axes)) == len(axes)
        for axes in CANONICAL_FAMILY_AXES.values()
    )








def test_tool_result_delivery_variant_closes_recursive_admission() -> None:
    """A result-delivery P packet preserves only eligible non-task resolvers."""

    payload = _input()
    workspace = build_canonical_turn_workspace(
        episode=payload["episode"],
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
        mutable_state=payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        identity_context=payload["character_identity_context"],
        available_actions=payload["available_actions"],
        available_resolvers=[
            {"capability": "approval_preparation"},
            {"capability": "human_clarification"},
            {"capability": "self_goal_resolution"},
            {"capability": "task_resolution_request"},
        ],
        overused_moves=payload["overused_moves"],
        response_plan_contract_variant="tool_result_delivery",
    )
    assert workspace["response_plan_contract_variant"] == (
        "tool_result_delivery"
    )
    assert validate_response_plan_contract_variant("tool_result_delivery") == (
        "tool_result_delivery"
    )
    packet = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "bounded_current_goal",
            "intent": "report the resolved tool result",
            "reason": "the current episode contains bounded task evidence",
            "cause_summary": "the task result",
        },
        appraisal_summary=[],
    )

    contract = packet["output_contract"]
    assert [
        row["capability"] for row in packet["capabilities"]["resolvers"]
    ] == ["approval_preparation"]
    assert set(contract["resolver_request_item_variants"]) == {"non_task"}
    assert "pending_task_continuation" not in contract
    assert "pending_resolution_fields" not in contract
    assert "response_plan_contract_variant" not in contract
    assert "tool_result_delivery" not in json.dumps(packet)
    assert facade_module._system_prompt_for_stage(
        stage="P",
        packet=packet,
        response_plan_contract_variant="tool_result_delivery",
    ) is facade_module._P_TOOL_RESULT_SYSTEM_PROMPT

    canonical_plan = {
        "goal_resolution": "answerable_now",
        "response_goal": "report the bounded result",
        "action_requests": [],
        "resolver_requests": [],
        "epistemic_boundary": "the result is limited to returned evidence",
    }
    plan = _validate_plan(
        canonical_plan,
        self_cognition=False,
        capabilities=packet["capabilities"],
        response_plan_contract_variant="tool_result_delivery",
    )
    assert plan.resolver_requests == ()
    invalid_candidates = [
        {
            **canonical_plan,
            "pending_task_continuation": None,
        },
        {
            **canonical_plan,
            "pending_resolution": {"decision": "answered", "reason": "done"},
        },
        {
            **canonical_plan,
            "goal_resolution": "requires_user_input",
            "resolver_requests": [{
                "capability": "human_clarification",
                "goal": "ask another user question",
                "reason": "retry the result",
            }],
        },
        {
            **canonical_plan,
            "goal_resolution": "requires_required_evidence",
            "resolver_requests": [{
                "capability": "task_resolution_request",
                "goal": "admit recursive work",
                "reason": "repeat the completed task",
                "start_in_background": True,
            }],
        },
    ]
    for candidate in invalid_candidates:
        with pytest.raises(CanonicalContractError):
            _validate_plan(
                candidate,
                self_cognition=False,
                capabilities=packet["capabilities"],
                response_plan_contract_variant="tool_result_delivery",
            )


def test_pending_continuation_reaches_every_stage_without_durable_identity() -> None:
    """A pending row is semantic context across A1/A2/G/P, never a row id."""

    payload = _input()
    pending_continuation = {
        "schema_version": RESOLVER_PENDING_CONTINUATION_VERSION,
        "capability_kind": "human_clarification",
        "status": "waiting_for_user",
        "original_goal": "完成一个有界的证据任务。",
        "question": "请补充任务所需的一个用户事实。",
        "pending_task_continuation": {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "background_task_admission",
        },
    }
    resolver_goal_progress = {
        "schema_version": "resolver_goal_progress.v1",
        "original_goal": "完成一个有界的证据任务。",
        "current_focus": "等待用户补充地点后取得路线证据。",
        "deliverables": [{
            "description": "两小时路线和时间切分",
            "status": "pending",
            "note": "等待地点后继续。",
        }],
        "missing_user_inputs": ["奥克兰所在区域"],
        "evidence_dependencies": ["路线距离和营业时间证据"],
        "attempted_paths": [],
        "source_backed_facts": [],
        "assumptions_or_inferences": [],
        "blockers": [],
        "final_response_requirements": ["保留精确标记：路线已核验"],
    }
    workspace = build_canonical_turn_workspace(
        episode=payload["episode"],
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
        mutable_state=payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        identity_context=payload["character_identity_context"],
        available_actions=payload["available_actions"],
        available_resolvers=payload["available_resolver_capabilities"],
        overused_moves=payload["overused_moves"],
        resolver_progress=resolver_goal_progress,
        pending_resolver_continuation=pending_continuation,
        response_plan_contract_variant="open_pending_resolution",
    )
    packets = build_turn_workspace_stage_contracts(workspace=workspace)

    for stage in ("A1", "A2", "G", "P"):
        projected = packets[stage]["pending_resolver_continuation"]
        assert projected["original_goal"] == "完成一个有界的证据任务。"
        assert projected["question"] == "请补充任务所需的一个用户事实。"
        progress = packets[stage]["resolver_goal_progress"]
        assert progress["original_goal"] == "完成一个有界的证据任务。"
        assert progress["evidence_dependencies"] == [
            "路线距离和营业时间证据"
        ]
        assert progress["final_response_requirements"] == [
            "保留精确标记：路线已核验"
        ]
        assert "guidance" not in packets[stage]
        rendered = json.dumps(packets[stage], ensure_ascii=False)
        assert "resume_id" not in rendered
        assert "resolver_pending:" not in rendered
    assert packets["P"]["output_contract"]["required_fields"] == [
        "goal_resolution", "response_goal", "action_requests",
        "resolver_requests", "epistemic_boundary", "pending_resolution",
    ]
    assert packets["P"]["output_contract"]["pending_resolution_fields"] == [
        "decision", "reason",
    ]

    closed_workspace = build_canonical_turn_workspace(
        episode=payload["episode"],
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
        mutable_state=payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        identity_context=payload["character_identity_context"],
        available_actions=payload["available_actions"],
        available_resolvers=payload["available_resolver_capabilities"],
        overused_moves=payload["overused_moves"],
        resolver_progress=resolver_goal_progress,
        response_plan_contract_variant="post_pending_resolution",
    )
    closed_packets = build_turn_workspace_stage_contracts(
        workspace=closed_workspace,
    )

    for stage in ("A1", "A2", "G", "P"):
        assert "pending_resolver_continuation" not in closed_packets[stage]
        assert closed_packets[stage]["resolver_goal_progress"] == (
            packets[stage]["resolver_goal_progress"]
        )
    assert "pending_resolution" not in closed_packets["P"]["output_contract"][
        "required_fields"
    ]






def test_fresh_ordinary_null_pending_task_continuation_normalizes_to_absence(
) -> None:
    """A null optional carrier has no semantic continuation authority."""

    plan = facade_module._validate_plan(
        {
            "goal_resolution": "answerable_now",
            "response_goal": "Respond to the current observation.",
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": "No external evidence is required.",
            "pending_task_continuation": None,
        },
        self_cognition=False,
        capabilities={"actions": [], "resolvers": []},
        response_plan_contract_variant="fresh_ordinary",
    )

    assert plan.pending_task_continuation is None


def test_approval_pending_row_stays_outside_clarification_continuity() -> None:
    """Approval rows retain their ledger flow without P clarification binding."""

    projection = project_pending_resume_for_prompt({
        "schema_version": RESOLVER_PENDING_RESUME_VERSION,
        "resume_id": "approval-row-id",
        "capability_kind": "approval_preparation",
        "status": "waiting_for_approval",
        "platform": "debug",
        "platform_channel_id": "deterministic-channel",
        "global_user_id": "user-1",
        "source_message_id": "previous-message",
        "prompt_safe_original_goal": "安排一个提醒。",
        "prompt_safe_question": "",
        "prompt_safe_approval_summary": "等待用户确认提醒。",
        "created_at_utc": "2026-07-13T00:00:00Z",
        "expires_at_utc": "2026-07-15T00:00:00Z",
    })

    assert projection is None


def test_pending_disposition_binds_only_the_hidden_selected_resume_id() -> None:
    """Caller state supplies the row id while the model output stays handleless."""

    payload = _input()
    caller_state = dict(payload)
    caller_state.update({
        "global_user_id": payload["mutable_state"]["owner_user_id"],
        "cognitive_episode": payload["episode"],
        "pending_resolver_resume": {
            "schema_version": RESOLVER_PENDING_RESUME_VERSION,
            "resume_id": "secret-selected-row-id",
            "capability_kind": "human_clarification",
            "status": "waiting_for_user",
            "platform": "debug",
            "platform_channel_id": "deterministic-channel",
            "global_user_id": "user-1",
            "source_message_id": "previous-message",
            "prompt_safe_original_goal": "完成一个有界的证据任务。",
            "prompt_safe_question": "请补充一个用户事实。",
            "prompt_safe_approval_summary": "",
            "pending_task_continuation": {
                "schema_version": PENDING_TASK_CONTINUATION_VERSION,
                "on_answered_clarification": "background_task_admission",
            },
            "created_at_utc": "2026-07-13T00:00:00Z",
            "expires_at_utc": "2026-07-15T00:00:00Z",
        },
    })
    output = {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {
            "goal_kind": "continue",
            "intent": "继续当前目标",
            "reason": "当前消息回答了原澄清。",
            "cause_summary": "当前观察",
        },
        "private_monologue": "我会沿着已回答的目标继续。",
        "response_plan": {
            "goal_resolution": "answerable_now",
            "response_goal": "继续当前目标",
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": "后续事实仍需来源确认。",
            "pending_resolution": {
                "decision": "answered",
                "reason": "当前消息回答了原澄清。",
            },
        },
        "state_projection": {"replacement_state": payload["mutable_state"]},
        "affect_projection": [],
        "relationship_projection": {},
        "relational_willingness": {},
        "cause_provenance": [],
    }

    projected = _project_output_to_global_state(
        output,
        caller_state,
        available_actions=payload["available_actions"],
        available_resolver_capabilities=(
            payload["available_resolver_capabilities"]
        ),
    )

    assert projected["resolver_pending_resolution"] == {
        "schema_version": "resolver_pending_resolution.v1",
        "resume_id": "secret-selected-row-id",
        "decision": "answered",
        "reason": "当前消息回答了原澄清。",
    }
    assert "resume_id" not in output["response_plan"]["pending_resolution"]










def test_terminal_text_seed_requires_authored_text_after_http_url_removal() -> None:
    """The shared terminal invariant excludes URL-only semantic seeds."""

    with pytest.raises(CognitionContractError):
        validate_terminal_text_seed(
            "https://unusable.example/only",
            "semantic seed",
        )
    assert validate_terminal_text_seed(
        "authored text https://allowed.example/reference",
        "semantic seed",
    ) == "authored text https://allowed.example/reference"


def test_selected_surface_intent_uses_the_shared_terminal_text_invariant() -> None:
    """The text-surface boundary rejects a URL-only selected intent."""

    surface_output = {
        "schema_version": "text_surface_output.v2",
        "content_plan": "a bounded content plan",
        "content_requirements": ["retain the current operation"],
        "epistemic_boundary": "Keep unsupported details uncertain.",
        "visible_boundaries": [],
        "addressee_plan": [],
        "delivery_profile": {
            "lexical_register": "warm",
            "sentence_shape": "concise",
            "rhythm": "steady",
            "hesitation": "light",
            "punctuation": "restrained",
        },
        "selected_surface_intent": "https://unusable.example/only",
        "permitted_action_results": [],
    }

    with pytest.raises(CognitionContractError):
        validate_text_surface_output(surface_output)


def test_canonical_input_requires_bounded_overused_moves_without_exposing_handles() -> None:
    """Require one exact bounded semantic continuity field at the boundary."""

    payload = _input()
    assert _validate_canonical_input(payload)["overused_moves"] == []

    missing = dict(payload)
    missing.pop("overused_moves")
    with pytest.raises(CanonicalContractError):
        _validate_canonical_input(missing)

    oversized = dict(payload)
    oversized["overused_moves"] = ["x" * 121]
    with pytest.raises(CanonicalContractError):
        _validate_canonical_input(oversized)


def test_canonical_input_requires_response_plan_contract_variant() -> None:
    """The P lifecycle selector is required internal turn data."""

    payload = _input()
    missing = dict(payload)
    missing.pop("response_plan_contract_variant")

    with pytest.raises(
        CanonicalContractError,
        match="canonical cognition input missing.*response_plan_contract_variant",
    ):
        _validate_canonical_input(missing)


@pytest.mark.parametrize("value", [None, [], {}, "unknown"])
def test_response_plan_contract_variant_rejects_invalid_values_as_value_error(
    value: object,
) -> None:
    """Malformed selector values retain the typed contract error boundary."""

    with pytest.raises(ValueError, match="response plan contract variant is invalid"):
        validate_response_plan_contract_variant(value)


def test_model_capabilities_are_semantic_and_reserve_speak_capacity() -> None:
    capabilities = _project_capabilities(
        [
            {
                "action_kind": "private_action",
                "description": "perform the approved private action",
                "decision_mode": "closed",
                "allowed_decisions": ["approve"],
                "default_decision": "approve",
                "decision_pattern": "",
                "context_ref": "private-run-123",
                "target_roles": [{"entity_id": "private-entity"}],
            },
        ],
        [{
            "capability": "lookup_current_public_information",
            "semantic_capability": "retrieve a current public fact",
            "context_ref": "resolver-private",
        }],
    )
    action = capabilities["actions"][0]
    resolver = capabilities["resolvers"][0]
    assert action == {
        "action_kind": "private_action",
        "description": "perform the approved private action",
        "decision_mode": "closed",
        "allowed_decisions": ["approve"],
        "default_decision": "approve",
        "decision_pattern": "",
    }
    assert resolver == {
        "capability": "lookup_current_public_information",
        "description": "retrieve a current public fact",
    }
    rendered = json.dumps(capabilities, ensure_ascii=False)
    assert all(secret not in rendered for secret in (
        "context_ref", "target_roles", "entity_id", "private-run-123",
        "resolver-private", "private-entity",
    ))
    action_rows = [
        {
            "action_kind": f"private_action_{index}",
            "description": "an allowed action",
            "decision_mode": "closed",
            "allowed_decisions": ["approve"],
            "default_decision": "approve",
            "decision_pattern": "",
        }
        for index in range(3)
    ]
    three_action_capabilities = {
        "actions": action_rows,
        "resolvers": [],
    }
    with pytest.raises(CanonicalContractError):
        _validate_plan(
            {
                "goal_resolution": "answerable_now",
                "response_goal": "acknowledge the request",
                "action_requests": [
                    {
                        "action_kind": row["action_kind"],
                        "decision": "approve",
                        "detail": "do the action",
                        "reason": "the request supports it",
                    }
                    for row in action_rows
                ],
                "resolver_requests": [],
                "epistemic_boundary": (
                    "Assert only the supplied current observation."
                ),
            },
            self_cognition=False,
            capabilities=three_action_capabilities,
            response_plan_contract_variant="fresh_ordinary",
        )


def test_future_speak_authority_uses_trusted_episode_not_model_detail() -> None:
    payload = _input()
    proposal = _build_scheduled_authority_proposal(
        {
            "cognitive_episode": payload["episode"],
            "storage_timestamp_utc": "2026-07-14T00:00:00Z",
        },
        {
            "decision": "2026-07-14 21:30",
            "detail": "a divergent model-selected authority claim",
        },
    )

    assert proposal["authorized_content_summary"] == "the current observation"
    assert proposal["authorized_detail_refs"][0]["semantic_summary"] == (
        "the current observation"
    )


def test_canonical_binder_covers_every_axis_with_receipts_and_valid_state() -> None:
    payload = _input()
    appraisals = tuple(
        CanonicalAppraisal(
            family=family,
            applicable=True,
            semantic_summary="the accepted semantic meaning",
            cause_summary="the accepted concrete cause",
            axis_changes=tuple(
                {"axis": axis, "shift": "slight_increase", "reason": "the cause changes this axis"}
                for axis in axes
            ),
        )
        for family, axes in CANONICAL_FAMILY_AXES.items()
    )
    evidence = [{"evidence_ref": payload["evidence"][0]["evidence_ref"]}]
    disposition_values = {
        "applied", "no_numeric_change", "scope_inapplicable", "turn_local",
    }
    results = []
    for state in (
        payload["mutable_state"],
        build_character_production_state(updated_at=payload["mutable_state"]["updated_at"]),
    ):
        updated, _transitions, receipts, roots = bind_axis_changes(
            {"mutable_state": state, "state_scope": state["state_scope"], "evidence": evidence},
            appraisals,
            goal={"intent": "retain a grounded active-character goal"},
            goal_resolution="requires_user_input",
        )
        validate_cognition_state(updated)
        assert len(receipts) == sum(len(axes) for axes in CANONICAL_FAMILY_AXES.values())
        assert {row["disposition"] for row in receipts} <= disposition_values
        assert all(row["disposition"] in disposition_values for row in receipts)
        assert len(roots) == 6
        results.append((state, updated, receipts))
    user_state, user_updated, user_receipts = results[0]
    assert user_updated["relationship"]["trust"] != user_state["relationship"]["trust"]
    assert user_updated["active_events"]
    assert user_updated["goals"]
    assert user_updated["threats"]
    assert user_updated["knowledge_gaps"]
    control = [row for row in user_receipts if row["axis"] == "controllability"]
    assert any(len(row.get("applied_targets", [])) == 2 for row in control)
    _character_state, character_updated, _character_receipts = results[1]
    assert character_updated["active_events"]
    assert character_updated["goals"]
    assert character_updated["threats"]
    assert character_updated["knowledge_gaps"]
    assert updated["drives"]["autonomy"]["pressure"] > 20
    assert updated["meaning_state"]["agency"] > 70

    bounded_state = deepcopy(payload["mutable_state"])
    bounded_state["relationship"]["evidence_refs"] = [
        {
            **payload["evidence"][0]["evidence_ref"],
            "source_id": f"historical-{index}",
        }
        for index in range(8)
    ]
    current_evidence = {
        **payload["evidence"][0]["evidence_ref"],
        "source_id": "current-observation",
        "semantic_summary": "the distinct current observation",
    }
    bounded_updated, _transitions, _receipts, _roots = bind_axis_changes(
        {
            "mutable_state": bounded_state,
            "state_scope": "user",
            "evidence": [{"evidence_ref": current_evidence}],
        },
            (CanonicalAppraisal(
                family="relationship_social",
                applicable=True,
                semantic_summary="relationship meaning",
                cause_summary="relationship cause",
                axis_changes=({
                    "axis": "trust",
                    "shift": "slight_increase",
                    "reason": "the current observation changes trust",
                },),
            ),),
        goal={"intent": "retain the relationship context"},
    )
    validate_cognition_state(bounded_updated)
    retained_refs = bounded_updated["relationship"]["evidence_refs"]
    assert len(retained_refs) == 8
    assert retained_refs[-1]["source_id"] == "current-observation"


def test_epistemic_gap_cause_is_retained_without_relief_transition() -> None:
    payload = _input()
    appraisal = CanonicalAppraisal(
        family="epistemic_comparison_memory",
        applicable=True,
        semantic_summary="the observation leaves a knowledge gap",
        cause_summary="the current observation is not yet answerable",
        axis_changes=(
            {"axis": "uncertainty", "shift": "strong_increase", "reason": "the answer is unknown"},
        ),
    )
    updated, transitions, _receipts, provenance = bind_axis_changes(
        {
            "mutable_state": payload["mutable_state"],
            "state_scope": "user",
            "evidence": [{"evidence_ref": payload["evidence"][0]["evidence_ref"]}],
        },
        (appraisal,),
        goal={"intent": "preserve uncertainty until evidence is available"},
    )
    validate_cognition_state(updated)
    assert updated["knowledge_gaps"]
    epistemic = next(row for row in provenance if row["family"] == appraisal.family)
    assert epistemic["primary_root"]["kind"] == "knowledge_gap"
    assert all(row["root_ref"]["kind"] in {"event", "threat"} for row in transitions)


def test_evidence_free_cognition_synthesizes_a_valid_episode_evidence_root() -> None:
    payload = _input()
    appraisals = (
        CanonicalAppraisal(
            family="event_agency",
            applicable=True,
            semantic_summary="the current observation has a grounded meaning",
            cause_summary="the current observation is the concrete cause",
            axis_changes=(),
        ),
        CanonicalAppraisal(
            family="goal_threat_outcome",
            applicable=True,
            semantic_summary="the evidence-free input does not ground a threat",
            cause_summary="no evidence is available for a threat judgment",
            axis_changes=({
                "axis": "expected_harm",
                "shift": "strong_increase",
                "reason": "this nonzero proposal must remain turn-local",
            },),
        ),
        CanonicalAppraisal(
            family="epistemic_comparison_memory",
            applicable=True,
            semantic_summary="the evidence-free input does not ground a gap",
            cause_summary="no evidence is available for a knowledge-gap judgment",
            axis_changes=({
                "axis": "novelty",
                "shift": "strong_increase",
                "reason": "this nonzero proposal must remain turn-local",
            },),
        ),
    )
    updated, _transitions, _receipts, _roots = bind_axis_changes(
        {
            "episode": payload["episode"],
            "mutable_state": payload["mutable_state"],
            "state_scope": payload["state_scope"],
            "evidence": [],
        },
        appraisals,
        goal={"intent": "retain the current observation"},
    )
    validate_cognition_state(updated)
    assert updated["active_events"] == []
    assert updated["goals"] == []
    assert updated["threats"] == []
    assert updated["knowledge_gaps"] == []


def test_strong_causal_shifts_retain_magnitude_and_derive_concrete_affect() -> None:
    payload = _input()
    appraisals = (
        CanonicalAppraisal(
            family="event_agency",
            applicable=True,
            semantic_summary="a deliberate harmful event",
            cause_summary="the accepted concrete abuse cause",
            axis_changes=({
                "axis": "intentionality",
                "shift": "strong_increase",
                "reason": "the harm was deliberate",
            },),
        ),
        CanonicalAppraisal(
            family="moral_identity",
            applicable=True,
            semantic_summary="the event violates a boundary",
            cause_summary="the accepted concrete abuse cause",
            axis_changes=tuple({
                "axis": axis,
                "shift": "strong_increase",
                "reason": "the concrete cause supports this judgment",
            } for axis in ("harm", "unfairness")),
        ),
    )
    updated, _transitions, _receipts, provenance = bind_axis_changes(
        {
            "mutable_state": payload["mutable_state"],
            "state_scope": "user",
            "evidence": [{"evidence_ref": payload["evidence"][0]["evidence_ref"]}],
        },
        appraisals,
        goal={"intent": "protect the boundary", "cause_summary": "the accepted concrete abuse cause"},
    )
    assert updated["active_events"][0]["harm"] == 40
    assert updated["active_events"][0]["unfairness"] == 40
    assert updated["active_events"][0]["salience"] >= 40
    activations = derive_persistent_emotion_activations(
        updated,
        updated_at=updated["updated_at"],
    )
    assert any(row["emotion_id"] == "anger" for row in activations)
    assert any(
        row["cause_summary"] == "the accepted concrete abuse cause"
        for row in provenance
    )


def test_controllability_only_updates_goal_without_fabricating_threat() -> None:
    payload = _input()
    appraisal = CanonicalAppraisal(
        family="goal_threat_outcome",
        applicable=True,
        semantic_summary="the goal remains controllable",
        cause_summary="the accepted goal cause",
        axis_changes=({
            "axis": "controllability",
            "shift": "moderate_increase",
            "reason": "the goal is controllable",
        },),
    )
    updated, _transitions, receipts, _provenance = bind_axis_changes(
        {
            "mutable_state": payload["mutable_state"],
            "state_scope": "user",
            "evidence": [{"evidence_ref": payload["evidence"][0]["evidence_ref"]}],
        },
        (appraisal,),
        goal={"intent": "answer the request", "cause_summary": "the accepted goal cause"},
    )
    assert updated["threats"] == []
    assert receipts[0]["disposition"] == "turn_local"
    assert receipts[0]["target_paths"] == []


def test_caller_materializes_typed_speak_and_resolver_envelopes() -> None:
    payload = _input()
    replacement, _transitions, _receipts, _provenance = bind_axis_changes(
        payload,
        (),
        goal={"intent": "answer the request", "cause_summary": "the request"},
    )
    caller_state = dict(payload)
    caller_state.update({
        "global_user_id": payload["mutable_state"]["owner_user_id"],
        "cognitive_episode": payload["episode"],
        "cognition_scene_context": payload["scene_context"],
    })
    output = {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {
            "goal_kind": "clarify",
            "intent": "answer the request",
            "reason": "the request needs a grounded clarification",
            "cause_summary": "the request",
        },
        "private_monologue": (
            "I want to answer, but I need the missing detail first."
        ),
        "response_plan": {
            "goal_resolution": "requires_user_input",
            "response_goal": "ask for the missing detail",
            "action_requests": [],
            "resolver_requests": [{
                "capability": "human_clarification",
                "goal": "clarify the missing detail",
                "reason": "the current evidence is insufficient",
            }],
            "pending_task_continuation": {
                "schema_version": PENDING_TASK_CONTINUATION_VERSION,
                "on_answered_clarification": "no_task_admission",
            },
            "epistemic_boundary": (
                "The missing detail remains unknown and must be asked."
            ),
        },
        "state_projection": {"replacement_state": replacement},
        "affect_projection": [],
        "relationship_projection": {},
        "relational_willingness": {},
        "cause_provenance": [],
    }
    projected = _project_output_to_global_state(
        output,
        caller_state,
        available_actions=payload["available_actions"],
        available_resolver_capabilities=payload[
            "available_resolver_capabilities"
        ],
    )
    assert [row["kind"] for row in projected["action_specs"]] == ["speak"]
    assert projected["pending_task_continuation"] == {
        "schema_version": PENDING_TASK_CONTINUATION_VERSION,
        "on_answered_clarification": "no_task_admission",
    }
    resolver = projected["resolver_capability_requests"]
    assert resolver[0]["schema_version"] == "resolver_capability_request.v1"
    assert resolver[0]["capability_kind"] == "human_clarification"
    assert resolver[0]["goal_continuation_ref"] is None
    with pytest.raises(CognitionExecutionError, match="not unique"):
        _project_output_to_global_state(
            output,
            caller_state,
            available_actions=payload["available_actions"] * 2,
            available_resolver_capabilities=payload[
                "available_resolver_capabilities"
            ],
        )




class _FourStageInvoker:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.configs: list[LLMCallConfig] = []
        self.packets: list[dict[str, object]] = []

    async def ainvoke(self, messages: object, *, config: LLMCallConfig) -> object:
        self.calls.append(config.stage_name.rsplit(".", 1)[-1])
        self.configs.append(config)
        self.packets.append(json.loads(messages[1].content))
        stage = self.calls[-1]
        if stage == "A1":
            value = {family: {
                "applicable": True, "semantic_summary": "meaning",
                "cause_summary": "current observation", "axis_changes": [],
            } for family in CANONICAL_A1_FAMILIES}
            value["event_agency"]["axis_changes"] = [{
                "axis": "responsibility",
                "shift": "moderate_increase",
                "reason": "the current observation assigns responsibility",
            }]
        elif stage == "A2":
            value = {family: {
                "applicable": True, "semantic_summary": "meaning",
                "cause_summary": "current observation", "axis_changes": [],
            } for family in CANONICAL_A2_FAMILIES}
            value["relationship_social"]["axis_changes"] = [{
                "axis": "trust",
                "shift": "moderate_increase",
                "reason": "the current observation changes trust",
            }]
        elif stage == "G":
            value = {
                "active_character_goal": {
                    "goal_kind": "clarify", "intent": "clarify the request",
                    "reason": "the request is underspecified",
                    "cause_summary": "current observation",
                },
                "relational_willingness": {
                    "applicable": False, "stance": "not sensitive",
                    "reason": "no relationship judgment is needed",
                    "cause_summary": "current observation",
                },
                "private_monologue": (
                    "I want to understand what they mean before I answer."
                ),
            }
        else:
            value = {
                "goal_resolution": "requires_user_input",
                "response_goal": "ask for clarification",
                "action_requests": [], "resolver_requests": [],
                "epistemic_boundary": (
                    "The intended referent remains unknown; ask rather than assert."
                ),
            }
        return SimpleNamespace(content=json.dumps(value, ensure_ascii=False))




