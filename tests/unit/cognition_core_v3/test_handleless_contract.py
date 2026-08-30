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
    RESOLVER_CAPABILITY_SEMANTICS,
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
from kazusa_ai_chatbot.cognition_shared.state_reducers import create_guarded_goal
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


def _assert_native_chinese_instruction(value: str) -> None:
    assert _HAN_PATTERN.search(value)
    prose = _CODE_SPAN_PATTERN.sub("", value)
    assert _MULTIWORD_ENGLISH_PATTERN.search(prose) is None


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


def test_cognition_chain_system_prompts_use_native_chinese() -> None:
    prompts = (
        facade_module._A1_SYSTEM_PROMPT,
        facade_module._A2_SYSTEM_PROMPT,
        facade_module._G_SYSTEM_PROMPT,
        facade_module._P_ORDINARY_SYSTEM_PROMPT,
        facade_module._P_PENDING_CLARIFICATION_SYSTEM_PROMPT,
        facade_module._P_DSH_INTERACTION_SYSTEM_PROMPT,
        facade_module._P_PENDING_AND_DSH_SYSTEM_PROMPT,
        facade_module._P_SELF_COGNITION_SYSTEM_PROMPT,
    )
    for prompt in prompts:
        _assert_native_chinese_instruction(prompt)
        assert "current_observation" in prompt
        assert "resolver_goal_progress" in prompt
        assert "contract_repair" in prompt


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
    assert "current_observation" in facade_module._A1_SYSTEM_PROMPT


def test_complete_prompt_variants_are_selected_without_prompt_composition() -> None:
    """Each runtime call selects one literal prompt instead of joining guidance."""

    for stage, packet, expected in (
        ("A1", {"output_contract": {}}, facade_module._A1_SYSTEM_PROMPT),
        ("A2", {"output_contract": {}}, facade_module._A2_SYSTEM_PROMPT),
        ("G", {"output_contract": {}}, facade_module._G_SYSTEM_PROMPT),
        ("P", {"output_contract": {}}, facade_module._P_ORDINARY_SYSTEM_PROMPT),
        (
            "P",
            {"output_contract": {}, "pending_resolver_continuation": {}},
            facade_module._P_PENDING_CLARIFICATION_SYSTEM_PROMPT,
        ),
        (
            "P",
            {"output_contract": {}, "pending_dsh_interaction": {}},
            facade_module._P_DSH_INTERACTION_SYSTEM_PROMPT,
        ),
        (
            "P",
            {
                "output_contract": {},
                "pending_resolver_continuation": {},
                "pending_dsh_interaction": {},
            },
            facade_module._P_PENDING_AND_DSH_SYSTEM_PROMPT,
        ),
        (
            "P",
            {"output_contract": {"required_fields": ["self_cognition_response"]}},
            facade_module._P_SELF_COGNITION_SYSTEM_PROMPT,
        ),
    ):
        assert facade_module._system_prompt_for_stage(
            stage=stage,
            packet=packet,
        ) == expected


def test_a1_system_prompt_and_packet_share_current_authority_contract() -> None:
    """A1 keeps authority in its complete prompt and state in its packet."""

    payload = _input()
    workspace = build_canonical_turn_workspace(
        episode=payload["episode"], scene_context=payload["scene_context"],
        evidence=payload["evidence"], mutable_state=payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        identity_context=payload["character_identity_context"],
        available_actions=payload["available_actions"],
        available_resolvers=payload["available_resolver_capabilities"],
        overused_moves=payload["overused_moves"],
        response_plan_contract_variant="fresh_ordinary",
    )
    packet = build_canonical_appraisal_question(workspace=workspace, stage_name="A1")
    assert "current_observation" in facade_module._A1_SYSTEM_PROMPT
    assert "guidance" not in packet and packet["current_observation"]


def test_all_stage_system_prompts_share_request_agency_authority_contract() -> None:
    for prompt in (
        facade_module._A1_SYSTEM_PROMPT, facade_module._A2_SYSTEM_PROMPT,
        facade_module._G_SYSTEM_PROMPT, facade_module._P_ORDINARY_SYSTEM_PROMPT,
    ):
        assert "current_observation` 是用户当下行动、意图、接受、许可和回应对象的唯一当前依据" in prompt


def test_goal_and_plan_system_prompts_share_background_goal_authority_contract() -> None:
    assert "只有当前观察把它们带入当前请求" in facade_module._G_SYSTEM_PROMPT
    assert "此前回应模式" in facade_module._P_ORDINARY_SYSTEM_PROMPT


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


def test_plan_packet_separates_user_prerequisite_from_task_admission() -> None:
    """Expose the producer contract for prerequisite and task admission."""

    payload = _input()
    resolver_affordances = [
        {
            "capability": capability,
            "semantic_capability": description,
            "availability": "available",
        }
        for capability, description in RESOLVER_CAPABILITY_SEMANTICS.items()
    ]
    workspace = build_canonical_turn_workspace(
        episode=payload["episode"],
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
        mutable_state=payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        identity_context=payload["character_identity_context"],
        available_actions=payload["available_actions"],
        available_resolvers=resolver_affordances,
        overused_moves=payload["overused_moves"],
        response_plan_contract_variant="fresh_ordinary",
    )
    packet = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "open_goal",
            "intent": "understand the current request",
            "reason": "the current request needs a grounded response",
            "cause_summary": "the current request",
        },
        appraisal_summary=[],
    )
    descriptions = {
        row["capability"]: row["description"]
        for row in packet["capabilities"]["resolvers"]
    }
    clarification = descriptions["human_clarification"]
    task_resolution = descriptions["task_resolution_request"]

    assert clarification == RESOLVER_CAPABILITY_SEMANTICS[
        "human_clarification"
    ]
    assert task_resolution == RESOLVER_CAPABILITY_SEMANTICS[
        "task_resolution_request"
    ]
    assert "missing user-controlled fact" in clarification
    assert "task objective itself is not yet bounded" in clarification
    assert "known prerequisite" in clarification
    assert "task admission closed" in clarification
    assert "bounded executable semantic task" in task_resolution
    assert "current request supplies a bounded executable" in task_resolution
    assert "required evidence" in task_resolution
    assert "missing known user-controlled fact" in task_resolution
    assert "admit the task first" in task_resolution
    assert clarification != task_resolution
    assert packet["output_contract"]["resolver_request_item_variants"] == {
        "non_task": {
            "required_fields": ["capability", "goal", "reason"],
            "additionalProperties": False,
        },
        "task_resolution_request": {
            "capability": "task_resolution_request",
            "required_fields": [
                "capability", "goal", "reason", "start_in_background",
            ],
            "additionalProperties": False,
            "field_rules": {
                "start_in_background": {
                    "type": "boolean",
                    "semantic_values": [
                        {
                            "value": False,
                            "delivery_timing": (
                                "required_evidence_before_current_visible_answer"
                            ),
                        },
                        {
                            "value": True,
                            "delivery_timing": (
                                "current_visible_acknowledgement_then_later_delivery"
                            ),
                            "selection_sources": [
                                {
                                    "source": "current_observation",
                                    "selection_condition": (
                                        "explicit_later_delivery"
                                    ),
                                },
                                {
                                    "source": "pending_resolver_continuation",
                                    "selection_condition": (
                                        "explicit_background_or_later_delivery"
                                    ),
                                    "required_pending_disposition": "answered",
                                    "current_observation_condition": (
                                        "answers_clarification_without_override_or_rejection"
                                    ),
                                },
                            ],
                        },
                    ],
                },
            },
        },
    }
    assert "task_resolution_start_in_background" not in packet["output_contract"]
    assert "goal_resolution_meanings" not in packet["output_contract"]
    assert "guidance" not in packet
    assert {
        row["capability"] for row in packet["capabilities"]["resolvers"]
    } >= {"human_clarification", "task_resolution_request"}
    assert packet["output_contract"]["pending_task_continuation"] == {
        "required_when": {
            "resolver_request_capability": "human_clarification",
            "exact_count": 1,
        },
        "forbidden_when": {
            "resolver_request_capability": "human_clarification",
            "exact_count": 0,
        },
        "fields": ["schema_version", "on_answered_clarification"],
        "schema_version": PENDING_TASK_CONTINUATION_VERSION,
        "on_answered_clarification_values": [
            "no_task_admission",
            "foreground_task_admission",
            "background_task_admission",
        ],
        "on_answered_clarification_semantics": {
            "no_task_admission": "answered_clarification_does_not_admit_task",
            "foreground_task_admission": "evidence_before_current_visible_answer",
            "background_task_admission": (
                "current_visible_acknowledgement_then_later_delivery"
            ),
        },
    }

    pending_workspace = build_canonical_turn_workspace(
        episode=payload["episode"],
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
        mutable_state=payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        identity_context=payload["character_identity_context"],
        available_actions=payload["available_actions"],
        available_resolvers=resolver_affordances,
        overused_moves=payload["overused_moves"],
        pending_resolver_continuation={
            "schema_version": RESOLVER_PENDING_CONTINUATION_VERSION,
            "capability_kind": "human_clarification",
            "status": "waiting_for_user",
            "original_goal": (
                "Please handle this task in the background after I choose "
                "the target file."
            ),
            "question": "Which file should I summarize?",
            "pending_task_continuation": {
                "schema_version": PENDING_TASK_CONTINUATION_VERSION,
                "on_answered_clarification": "background_task_admission",
            },
        },
        response_plan_contract_variant="open_pending_resolution",
    )
    pending_packet = build_canonical_plan_question(
        workspace=pending_workspace,
        goal={
            "goal_kind": "open_goal",
            "intent": "await the selected target file",
            "reason": "the user must choose a file before task admission",
            "cause_summary": "the pending clarification",
        },
        appraisal_summary=[],
    )
    pending_background_source = pending_packet["output_contract"][
        "resolver_request_item_variants"
    ]["task_resolution_request"]["field_rules"]["start_in_background"][
        "semantic_values"
    ][1]["selection_sources"][1]

    assert pending_packet["pending_resolver_continuation"]["original_goal"] == (
        "Please handle this task in the background after I choose the target file."
    )
    assert "pending_task_continuation" not in pending_packet["output_contract"]
    assert all(
        row["capability"] != "human_clarification"
        for row in pending_packet["capabilities"]["resolvers"]
    )
    assert any(
        row["capability"] == "task_resolution_request"
        for row in pending_packet["capabilities"]["resolvers"]
    )
    assert set(
        pending_packet["output_contract"]["resolver_request_item_variants"]
    ) == {"non_task", "task_resolution_request"}
    assert pending_background_source == {
        "source": "pending_resolver_continuation",
        "selection_condition": "explicit_background_or_later_delivery",
        "required_pending_disposition": "answered",
        "current_observation_condition": (
            "answers_clarification_without_override_or_rejection"
        ),
    }
    assert pending_packet["output_contract"]["pending_resolution_values"] == [
        "answered", "continue_waiting", "rejected", "superseded",
    ]


def test_plan_packet_variants_follow_the_visible_resolver_roster() -> None:
    """P advertises only item shapes available in its own resolver roster."""

    payload = _input()

    def packet_for(
        resolver_capabilities: list[dict[str, str]],
    ) -> dict[str, object]:
        workspace = build_canonical_turn_workspace(
            episode=payload["episode"],
            scene_context=payload["scene_context"],
            evidence=payload["evidence"],
            mutable_state=payload["mutable_state"],
            character_constraints=payload["character_constraints"],
            identity_context=payload["character_identity_context"],
            available_actions=payload["available_actions"],
            available_resolvers=resolver_capabilities,
            overused_moves=payload["overused_moves"],
            response_plan_contract_variant="fresh_ordinary",
        )
        return build_canonical_plan_question(
            workspace=workspace,
            goal={
                "goal_kind": "bounded_current_goal",
                "intent": "respond to the current observation",
                "reason": "the current request is present",
                "cause_summary": "the current observation",
            },
            appraisal_summary=[],
        )

    task_only = packet_for([{"capability": "task_resolution_request"}])
    non_task_only = packet_for([{"capability": "approval_preparation"}])
    no_resolvers = packet_for([])

    assert set(task_only["output_contract"]["resolver_request_item_variants"]) == {
        "task_resolution_request",
    }
    assert set(
        non_task_only["output_contract"]["resolver_request_item_variants"]
    ) == {"non_task"}
    assert no_resolvers["output_contract"]["resolver_request_item_variants"] == {}


def test_post_pending_resolution_variant_forbids_carrier_and_admission() -> None:
    """A closed pending resolution keeps ordinary P processing carrier-free."""

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
        response_plan_contract_variant="post_pending_resolution",
    )
    packet = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "bounded_current_goal",
            "intent": "continue after the resolved clarification",
            "reason": "the required evidence is now available",
            "cause_summary": "the resolver observation",
        },
        appraisal_summary=[],
    )

    contract = packet["output_contract"]
    assert "response_plan_contract_variant" not in contract
    assert "pending_resolver_continuation" not in packet
    assert "pending_task_continuation" not in contract
    assert "pending_resolution_fields" not in contract
    assert "post_pending_resolution" not in contract
    assert "response_plan_contract_variant" not in json.dumps(packet)
    assert all(
        row["capability"] not in {
            "human_clarification",
            "task_resolution_request",
        }
        for row in packet["capabilities"]["resolvers"]
    )
    assert set(contract["resolver_request_item_variants"]) == {"non_task"}

    ordinary_plan = {
        "goal_resolution": "answerable_now",
        "response_goal": "give the grounded result",
        "action_requests": [],
        "resolver_requests": [],
        "epistemic_boundary": "retain source limits",
    }
    plan = facade_module._validate_plan(
        ordinary_plan,
        self_cognition=False,
        capabilities={"actions": [], "resolvers": []},
        response_plan_contract_variant="post_pending_resolution",
    )
    assert plan.pending_resolution is None
    with pytest.raises(
        CanonicalContractError,
        match=r"unexpected fields \['pending_task_continuation'\]",
    ):
        facade_module._validate_plan(
            {
                **ordinary_plan,
                "pending_task_continuation": {
                    "schema_version": PENDING_TASK_CONTINUATION_VERSION,
                    "on_answered_clarification": "background_task_admission",
                },
            },
            self_cognition=False,
            capabilities={"actions": [], "resolvers": []},
            response_plan_contract_variant="post_pending_resolution",
        )
    with pytest.raises(
        CanonicalContractError,
        match="pending continuation cannot create human clarification",
    ):
        facade_module._validate_plan(
            {
                **ordinary_plan,
                "goal_resolution": "requires_user_input",
                "resolver_requests": [{
                    "capability": "human_clarification",
                    "goal": "ask for the missing fact",
                    "reason": "the task remains unbounded",
                }],
            },
            self_cognition=False,
            capabilities={
                "actions": [],
                "resolvers": [{"capability": "human_clarification"}],
            },
            response_plan_contract_variant="post_pending_resolution",
        )
    with pytest.raises(
        CanonicalContractError,
        match="post pending continuation cannot create task resolution",
    ):
        facade_module._validate_plan(
            {
                **ordinary_plan,
                "goal_resolution": "requires_required_evidence",
                "resolver_requests": [{
                    "capability": "task_resolution_request",
                    "goal": "repeat the accepted task",
                    "reason": "the stale admission was echoed",
                    "start_in_background": True,
                }],
            },
            self_cognition=False,
            capabilities={
                "actions": [],
                "resolvers": [{"capability": "task_resolution_request"}],
            },
            response_plan_contract_variant="post_pending_resolution",
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
    assert "当前观察" in facade_module._P_PENDING_CLARIFICATION_SYSTEM_PROMPT
    assert "确实已经回答" in facade_module._P_PENDING_CLARIFICATION_SYSTEM_PROMPT

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


def test_pending_resolution_contract_is_exact_and_task_compatible() -> None:
    """Pending dispositions are semantic and gate retained task admission."""

    pending_continuation = {
        "schema_version": RESOLVER_PENDING_CONTINUATION_VERSION,
        "capability_kind": "human_clarification",
        "status": "waiting_for_user",
        "original_goal": "完成一个有界的证据任务。",
        "question": "请补充一个用户事实。",
        "pending_task_continuation": {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "foreground_task_admission",
        },
    }
    common = {
        "goal_resolution": "requires_required_evidence",
        "response_goal": "继续处理当前有界目标",
        "action_requests": [],
        "resolver_requests": [{
            "capability": "task_resolution_request",
            "goal": "取得目标所需的证据",
            "reason": "当前目标需要证据",
            "start_in_background": False,
        }],
        "epistemic_boundary": "证据仍需由来源确认。",
    }
    plan = facade_module._validate_plan(
        {
            **common,
            "pending_resolution": {
                "decision": "answered",
                "reason": "当前消息回答了原澄清。",
            },
        },
        self_cognition=False,
        capabilities={
            "actions": [],
            "resolvers": [{"capability": "task_resolution_request"}],
        },
        pending_resolver_continuation=pending_continuation,
        response_plan_contract_variant="open_pending_resolution",
    )
    assert plan.pending_resolution == {
        "decision": "answered",
        "reason": "当前消息回答了原澄清。",
    }

    with pytest.raises(CanonicalContractError, match="fields are not exact"):
        facade_module._validate_plan(
            {
                **common,
                "pending_resolution": {
                    "decision": "answered",
                    "reason": "当前消息回答了原澄清。",
                    "resume_id": "hidden-row-id",
                },
            },
            self_cognition=False,
            capabilities={
                "actions": [],
                "resolvers": [{"capability": "task_resolution_request"}],
            },
            pending_resolver_continuation=pending_continuation,
            response_plan_contract_variant="open_pending_resolution",
        )
    for decision in ("continue_waiting", "rejected", "superseded"):
        with pytest.raises(
            CanonicalContractError,
            match="pending disposition must be answered",
        ):
            facade_module._validate_plan(
                {
                    **common,
                    "pending_resolution": {
                        "decision": decision,
                        "reason": "当前消息没有回答原澄清。",
                    },
                },
                self_cognition=False,
                capabilities={
                    "actions": [],
                    "resolvers": [{"capability": "task_resolution_request"}],
                },
                pending_resolver_continuation=pending_continuation,
                response_plan_contract_variant="open_pending_resolution",
            )
    with pytest.raises(
        CanonicalContractError,
        match="response plan: missing fields \\['pending_resolution'\\]",
    ):
        facade_module._validate_plan(
            common,
            self_cognition=False,
            capabilities={
                "actions": [],
                "resolvers": [{"capability": "task_resolution_request"}],
            },
                pending_resolver_continuation=pending_continuation,
                response_plan_contract_variant="open_pending_resolution",
        )


def test_pending_task_continuation_gates_answered_task_admission() -> None:
    """An answer-conditioned carrier fixes admission timing without text inference."""

    common = {
        "goal_resolution": "requires_required_evidence",
        "response_goal": "继续处理当前有界目标",
        "action_requests": [],
        "resolver_requests": [{
            "capability": "task_resolution_request",
            "goal": "取得目标所需的证据",
            "reason": "当前目标需要证据",
            "start_in_background": True,
        }],
        "epistemic_boundary": "证据仍需由来源确认。",
        "pending_resolution": {
            "decision": "answered",
            "reason": "当前消息回答了原澄清。",
        },
    }
    background_continuation = {
        "schema_version": RESOLVER_PENDING_CONTINUATION_VERSION,
        "capability_kind": "human_clarification",
        "status": "waiting_for_user",
        "original_goal": "后台完成有界的证据任务。",
        "question": "请补充一个用户事实。",
        "pending_task_continuation": {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "background_task_admission",
        },
    }
    capabilities = {
        "actions": [],
        "resolvers": [{"capability": "task_resolution_request"}],
    }

    plan = facade_module._validate_plan(
        common,
        self_cognition=False,
        capabilities=capabilities,
        pending_resolver_continuation=background_continuation,
        response_plan_contract_variant="open_pending_resolution",
    )

    assert plan.resolver_requests[0]["start_in_background"] is True
    assert plan.pending_task_continuation is None
    mismatched_background = {
        **common,
        "resolver_requests": [{
            **common["resolver_requests"][0],
            "start_in_background": False,
        }],
    }
    with pytest.raises(
        CanonicalContractError,
        match="mismatches pending continuation",
    ):
        facade_module._validate_plan(
            mismatched_background,
            self_cognition=False,
            capabilities=capabilities,
            pending_resolver_continuation=background_continuation,
            response_plan_contract_variant="open_pending_resolution",
        )

    foreground_continuation = {
        **background_continuation,
        "pending_task_continuation": {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "foreground_task_admission",
        },
    }
    with pytest.raises(
        CanonicalContractError,
        match="mismatches pending continuation",
    ):
        facade_module._validate_plan(
            common,
            self_cognition=False,
            capabilities=capabilities,
            pending_resolver_continuation=foreground_continuation,
            response_plan_contract_variant="open_pending_resolution",
        )

    no_task_continuation = {
        **background_continuation,
        "pending_task_continuation": {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "no_task_admission",
        },
    }
    with pytest.raises(CanonicalContractError, match="forbids task admission"):
        facade_module._validate_plan(
            mismatched_background,
            self_cognition=False,
            capabilities=capabilities,
            pending_resolver_continuation=no_task_continuation,
            response_plan_contract_variant="open_pending_resolution",
        )

    clarification_plan = {
        "goal_resolution": "requires_user_input",
        "response_goal": "询问一个缺失事实。",
        "action_requests": [],
        "resolver_requests": [{
            "capability": "human_clarification",
            "goal": "询问一个缺失事实。",
            "reason": "当前任务尚未有界。",
        }],
        "epistemic_boundary": "缺失事实仍未知。",
    }
    clarification_capabilities = {
        "actions": [],
        "resolvers": [{"capability": "human_clarification"}],
    }
    with pytest.raises(CanonicalContractError, match="continuation is required"):
        facade_module._validate_plan(
            clarification_plan,
            self_cognition=False,
            capabilities=clarification_capabilities,
            response_plan_contract_variant="fresh_ordinary",
        )
    validated_clarification = facade_module._validate_plan(
        {
            **clarification_plan,
            "pending_task_continuation": {
                "schema_version": PENDING_TASK_CONTINUATION_VERSION,
                "on_answered_clarification": "background_task_admission",
            },
        },
        self_cognition=False,
        capabilities=clarification_capabilities,
        response_plan_contract_variant="fresh_ordinary",
    )
    assert validated_clarification.pending_task_continuation == {
        "schema_version": PENDING_TASK_CONTINUATION_VERSION,
        "on_answered_clarification": "background_task_admission",
    }

    with pytest.raises(
        CanonicalContractError,
        match=r"unexpected fields \['pending_task_continuation'\]",
    ):
        facade_module._validate_plan(
            {
                **common,
                "pending_task_continuation": {
                    "schema_version": PENDING_TASK_CONTINUATION_VERSION,
                    "on_answered_clarification": "background_task_admission",
                },
            },
            self_cognition=False,
            capabilities=capabilities,
            pending_resolver_continuation=background_continuation,
            response_plan_contract_variant="open_pending_resolution",
        )

    with pytest.raises(
        CanonicalContractError,
        match="pending continuation cannot create human clarification",
    ):
        facade_module._validate_plan(
            {
                **clarification_plan,
                "pending_resolution": {
                    "decision": "continue_waiting",
                    "reason": "The current observation remains incomplete.",
                },
            },
            self_cognition=False,
            capabilities=clarification_capabilities,
            pending_resolver_continuation=background_continuation,
            response_plan_contract_variant="open_pending_resolution",
        )

    with pytest.raises(
        CanonicalContractError,
        match="human clarification cannot co-occur with task admission",
    ):
            facade_module._validate_plan(
                {
                    **clarification_plan,
                    "goal_resolution": "requires_required_evidence",
                    "pending_task_continuation": {
                        "schema_version": PENDING_TASK_CONTINUATION_VERSION,
                        "on_answered_clarification": "background_task_admission",
                    },
                    "resolver_requests": [
                    *clarification_plan["resolver_requests"],
                    common["resolver_requests"][0],
                ],
            },
            self_cognition=False,
            capabilities={
                "actions": [],
                "resolvers": [
                    {"capability": "human_clarification"},
                    {"capability": "task_resolution_request"},
                ],
            },
            response_plan_contract_variant="fresh_ordinary",
        )

    with pytest.raises(
        CanonicalContractError,
        match="pending_task_continuation is limited to human clarification",
    ):
        facade_module._validate_plan(
            {
                "goal_resolution": "answerable_now",
                "response_goal": "Respond to the current observation.",
                "action_requests": [],
                "resolver_requests": [],
                "epistemic_boundary": "No external evidence is required.",
                "pending_task_continuation": {
                    "schema_version": PENDING_TASK_CONTINUATION_VERSION,
                    "on_answered_clarification": "no_task_admission",
                },
            },
            self_cognition=False,
            capabilities={"actions": [], "resolvers": []},
            response_plan_contract_variant="fresh_ordinary",
        )


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


def test_task_resolution_request_requires_required_evidence_resolution() -> None:
    """The P-stage keeps background intent in the typed task request."""

    plan = facade_module._validate_plan(
        {
            "goal_resolution": "requires_required_evidence",
            "response_goal": "obtain the required evidence",
            "action_requests": [],
            "resolver_requests": [{
                "capability": "task_resolution_request",
                "goal": "complete the bounded task",
                 "reason": "the bounded task requires evidence",
                "start_in_background": True,
            }],
            "epistemic_boundary": "The task parameter remains unknown.",
        },
        self_cognition=False,
        capabilities={
            "actions": [],
            "resolvers": [{"capability": "task_resolution_request"}],
        },
        response_plan_contract_variant="fresh_ordinary",
    )

    assert plan.resolver_requests == ({
        "capability": "task_resolution_request",
        "goal": "complete the bounded task",
        "reason": "the bounded task requires evidence",
        "start_in_background": True,
    },)


def test_task_resolution_background_flag_requires_nested_canonical_shape() -> None:
    """P rejects the observed top-level task flag without moving it into a request."""

    invalid_plan = {
        "goal_resolution": "requires_required_evidence",
        "response_goal": "obtain the required evidence",
        "action_requests": [],
        "resolver_requests": [{
            "capability": "task_resolution_request",
            "goal": "complete the bounded task",
            "reason": "the bounded task requires evidence",
        }],
        "epistemic_boundary": "The task parameter remains unknown.",
        "task_resolution_start_in_background": False,
    }
    capabilities = {
        "actions": [],
        "resolvers": [{"capability": "task_resolution_request"}],
    }

    with pytest.raises(
        CanonicalContractError,
        match=(
            "response plan: unexpected fields "
            "\\['task_resolution_start_in_background'\\]"
        ),
    ):
        facade_module._validate_plan(
            invalid_plan,
            self_cognition=False,
            capabilities=capabilities,
            response_plan_contract_variant="fresh_ordinary",
        )

    invalid_nested_plan = dict(invalid_plan)
    invalid_nested_plan.pop("task_resolution_start_in_background")
    with pytest.raises(
        CanonicalContractError,
        match="resolver_requests\\[0\\]: missing fields \\['start_in_background'\\]",
    ):
        facade_module._validate_plan(
            invalid_nested_plan,
            self_cognition=False,
            capabilities=capabilities,
            response_plan_contract_variant="fresh_ordinary",
        )


@pytest.mark.parametrize(
    "goal_resolution",
    ("answerable_now", "requires_user_input", "blocked"),
)
def test_task_resolution_request_rejects_incompatible_goal_resolution(
    goal_resolution: str,
) -> None:
    """The P contract rejects task admission outside required evidence."""

    with pytest.raises(CanonicalContractError, match="requires goal_resolution"):
        facade_module._validate_plan(
            {
                "goal_resolution": goal_resolution,
                "response_goal": "describe the bounded task",
                "action_requests": [],
                "resolver_requests": [{
                    "capability": "task_resolution_request",
                    "goal": "complete the bounded task",
                    "reason": "the task needs required evidence",
                    "start_in_background": False,
                }],
                "epistemic_boundary": "The required evidence is not complete.",
            },
            self_cognition=False,
            capabilities={
                "actions": [],
                "resolvers": [{"capability": "task_resolution_request"}],
            },
            response_plan_contract_variant="fresh_ordinary",
        )


def test_task_resolution_projection_maps_background_choice_to_priority() -> None:
    """The caller projects the typed P-stage choice into the V1 request."""

    payload = _input()
    caller_state = {
        **payload,
        "global_user_id": payload["mutable_state"]["owner_user_id"],
        "cognitive_episode": payload["episode"],
    }
    output = {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {
            "goal_kind": "ordinary_response",
            "intent": "admit the bounded task",
            "reason": "the current observation supplies the task objective",
            "cause_summary": "the current observation",
        },
        "private_monologue": "I can admit this bounded task and preserve its question.",
        "response_plan": {
            "goal_resolution": "requires_required_evidence",
            "response_goal": "obtain the required evidence",
            "action_requests": [],
            "resolver_requests": [{
                "capability": "task_resolution_request",
                "goal": "complete the bounded task",
                "reason": "the bounded task requires evidence",
                "start_in_background": True,
            }],
            "epistemic_boundary": "The task parameter remains unknown.",
        },
        "state_projection": {
            "replacement_state": payload["mutable_state"],
            "continuation_goal_ref": {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary_response:user:current",
            },
        },
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

    request = projected["resolver_capability_requests"][0]
    assert request["capability_kind"] == "task_resolution_request"
    assert request["priority"] == "background"


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


def test_goal_capacity_deferral_preserves_response_and_unrelated_resolver() -> None:
    payload = _input()
    state = deepcopy(payload["mutable_state"])
    evidence = payload["evidence"][0]["evidence_ref"]
    for index in range(16):
        row_evidence = {
            **evidence,
            "source_id": f"goal-capacity-{index}",
        }
        state["goals"].append(create_guarded_goal(
            state,
            goal_kind="safety",
            description=f"protected continuing goal {index}",
            role_refs=[],
            evidence_refs=[row_evidence],
            axes={},
            updated_at=state["updated_at"],
        ))
    validate_cognition_state(state)
    replacement, _transitions, receipts, _provenance = bind_axis_changes(
        {
            "episode": payload["episode"],
            "mutable_state": state,
            "state_scope": "user",
            "evidence": payload["evidence"],
        },
        (),
        goal={
            "intent": "preserve the answer while capacity is full",
            "cause_summary": "the current request",
        },
        goal_resolution="requires_user_input",
        resolver_requests=[{"capability": "task_resolution_request"}],
    )
    assert any(
        row.get("kind") == "goal"
        and row.get("disposition") == "capacity_deferred"
        for row in receipts
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
            "intent": "preserve the answer while capacity is full",
            "reason": "the current request still has a visible response",
            "cause_summary": "the current request",
        },
        "private_monologue": (
            "I still want to give a useful answer despite the blocked task."
        ),
        "response_plan": {
            "goal_resolution": "requires_user_input",
            "response_goal": "explain the missing detail",
            "action_requests": [],
            "resolver_requests": [
                {
                    "capability": "task_resolution_request",
                    "goal": "continue the task",
                    "reason": "durable task lineage is required",
                    "start_in_background": False,
                },
                {
                    "capability": "human_clarification",
                    "goal": "clarify the missing detail",
                    "reason": "the current evidence is insufficient",
                },
            ],
            "epistemic_boundary": (
                "State the current limitation without claiming a task result."
            ),
        },
        "state_projection": {
            "replacement_state": replacement,
            "capacity_deferred": [
                row for row in receipts
                if row.get("disposition") == "capacity_deferred"
            ],
        },
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
    assert projected["active_character_goal"]["intent"] == (
        "preserve the answer while capacity is full"
    )
    assert [row["kind"] for row in projected["action_specs"]] == ["speak"]
    assert [
        row["capability_kind"]
        for row in projected["resolver_capability_requests"]
    ] == ["human_clarification"]


class _FourStageInvoker:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.configs: list[LLMCallConfig] = []
        self.packets: list[dict[str, object]] = []
        self.system_prompts: list[str] = []

    async def ainvoke(self, messages: object, *, config: LLMCallConfig) -> object:
        self.calls.append(config.stage_name.rsplit(".", 1)[-1])
        self.configs.append(config)
        self.system_prompts.append(messages[0].content)
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


@pytest.mark.asyncio
async def test_canonical_cognition_calls_a1_a2_g_p_once_with_subjective_outputs(
    monkeypatch,
) -> None:
    invoker = _FourStageInvoker()
    trace_rows: list[dict[str, object]] = []

    async def record_trace_step(**kwargs: object) -> dict[str, object]:
        trace_rows.append(kwargs)
        return {
            "accepted": True,
            "trace_id": "deterministic-trace",
            "status": "recorded",
            "reason": "",
        }

    monkeypatch.setattr(
        facade_module.llm_tracing,
        "record_llm_trace_step",
        record_trace_step,
    )
    trace_token = facade_module.llm_tracing.bind_trace_id("deterministic-trace")
    token = bind_protected_chain_records(run_id="deterministic-test")
    try:
        output = await run_cognition(_input(), _services(invoker))
        records = snapshot_protected_chain_records()
    finally:
        reset_protected_chain_records(token)
        facade_module.llm_tracing.reset_trace_id(trace_token)
    assert invoker.calls == ["A1", "A2", "G", "P"]
    assert [row["stage_name"] for row in trace_rows] == [
        "cognition_core_v3.A1",
        "cognition_core_v3.A2",
        "cognition_core_v3.G",
        "cognition_core_v3.P",
    ]
    assert [row["trace_id"] for row in trace_rows] == [
        "deterministic-trace",
    ] * 4
    assert [row["status"] for row in trace_rows] == ["succeeded"] * 4
    assert [row["attempt_index"] for row in trace_rows] == [1] * 4
    assert all(row["parsed_output"] for row in trace_rows)
    assert [config.output_mode for config in invoker.configs] == [
        "json_object"
    ] * 4
    assert set(invoker.packets[0]) >= {
        "stage", "current_observation", "direct_facts",
        "continuation_state", "output_contract",
    }
    assert "conditional_character_context" not in invoker.packets[0]
    assert set(invoker.packets[1]) >= {
        "accepted_a1_meaning", "conditional_character_context",
        "participant_continuity",
    }
    assert "capabilities" not in invoker.packets[1]
    assert all(
        "axis_changes" not in row
        for row in invoker.packets[1]["accepted_a1_meaning"]
    )
    assert set(invoker.packets[2]) >= {
        "appraisal_summary", "conditional_character_context",
    }
    assert "output_contract" in invoker.packets[3]
    assert "appraisal_summary" not in invoker.packets[3]
    assert "缺少证据" in invoker.system_prompts[3]
    assert "只能作为不确定的解释" in invoker.system_prompts[3]
    assert "当前观察已经给出有界、可执行的语义目标" in (
        invoker.system_prompts[3]
    )
    assert all("guidance" not in packet for packet in invoker.packets)
    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                assert not str(key).endswith((
                    "_id", "_ids", "_handle", "_handles", "_ref", "_refs", "_path", "_paths",
                ))
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            assert not re.search(r"\b(?:e|ce|ct|ck|ev|g|r|b|a|p)\d+\b", value)
    for packet in invoker.packets:
        walk(packet)
    assert [record["stage"] for record in records] == ["A1", "A2", "G", "P"]
    assert all(record["status"] == "parsed" for record in records)
    assert output["schema_version"] == "cognition_output.v3"
    assert output["active_character_goal"]["goal_kind"] == "clarify"
    assert output["private_monologue"] == (
        "I want to understand what they mean before I answer."
    )
    assert output["response_plan"]["epistemic_boundary"] == (
        "The intended referent remains unknown; ask rather than assert."
    )
    assert len(output["appraisals"]) == 6
    assert output["diagnostics"] == {"status": "complete"}
    assert output["state_projection"]["replacement_state"]["relationship"]["trust"] > (
        output["state_projection"]["expected_previous_state"]["relationship"]["trust"]
    )
    assert all(output["cause_provenance"])
    assert all(
        {"primary_root", "root_refs", "cause_status", "cause_summary"}
        <= set(row)
        for row in output["cause_provenance"]
    )
    assert output["state_projection"]["binding_receipts"]
    assert len(output["state_projection"]["binding_receipts"]) == 2
    assert all(
        set(row["primary_root"]) == {"scope", "kind", "entity_id"}
        for row in output["cause_provenance"]
    )
    state = output["state_projection"]["replacement_state"]
    for row in output["cause_provenance"]:
        root = row["primary_root"]
        if root["kind"] == "relationship":
            assert state["relationship"]["relationship_id"] == root["entity_id"]
        elif root["kind"] == "event":
            assert any(item["entity_id"] == root["entity_id"] for item in state["active_events"])
        elif root["kind"] == "goal":
            assert any(item["entity_id"] == root["entity_id"] for item in state["goals"])
    assert set(output["relationship_projection"]["axes"]) >= {
        "familiarity", "positive_regard", "boundary_safety", "salience"
    }


@pytest.mark.asyncio
async def test_canonical_cognition_completes_without_input_evidence() -> None:
    invoker = _FourStageInvoker()
    payload = _input()
    payload["evidence"] = []
    output = await run_cognition(payload, _services(invoker))
    assert invoker.calls == ["A1", "A2", "G", "P"]
    assert output["state_projection"]["replacement_state"]["active_events"] == []
