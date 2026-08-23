"""Focused deterministic coverage for the canonical phase-one boundary."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v3.appraisal import bind_axis_changes
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CANONICAL_A1_FAMILIES,
    CANONICAL_A2_FAMILIES,
    CANONICAL_FAMILY_AXES,
    CanonicalAppraisal,
    CognitionChainServicesV3,
)
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    CanonicalContractError,
    _validate_plan,
    bind_protected_chain_records,
    reset_protected_chain_records,
    run_cognition,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    _project_capabilities,
    build_canonical_turn_workspace,
    build_turn_workspace_stage_contracts,
)
from kazusa_ai_chatbot.cognition_episode import build_user_message_episode
from kazusa_ai_chatbot.cognition_shared.contracts import CognitionExecutionError
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
    _coding_run_action_affordances,
    _project_output_to_global_state,
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
    )
    packets = build_turn_workspace_stage_contracts(workspace=workspace)
    assert packets["A1"]["output_contract"]["required_fields"] == list(CANONICAL_A1_FAMILIES)
    assert packets["A2"]["output_contract"]["required_fields"] == list(CANONICAL_A2_FAMILIES)
    assert packets["G"]["output_contract"]["required_fields"] == [
        "active_character_goal", "relational_willingness"
    ]
    assert packets["P"]["output_contract"]["required_fields"] == [
        "goal_resolution", "response_goal", "action_requests", "resolver_requests"
    ]
    assert set(packets["P"]) == {
        "stage", "guidance", "goal", "capabilities", "output_contract",
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
            },
            self_cognition=False,
            capabilities=three_action_capabilities,
        )


def test_ambiguous_coding_context_does_not_advertise_a_selector() -> None:
    base_affordance = {
        "action_kind": "accepted_coding_task_request",
        "capability": "continue one accepted coding task",
        "permission": "allowed",
        "decision_mode": "closed",
        "allowed_decisions": ["status"],
        "default_decision": "status",
        "decision_pattern": "",
        "context_ref": "",
        "target_roles": [],
    }
    state = {
        "action_selection_context": {
            "coding_runs": [
                {"coding_run_ref": "run-a", "allowed_next_actions": ["status"]},
                {"coding_run_ref": "run-b", "allowed_next_actions": ["status"]},
            ],
        },
        "action_availability_runtime": {},
    }
    assert _coding_run_action_affordances(
        state,
        base_affordance=base_affordance,
    ) == []


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
        "response_plan": {
            "goal_resolution": "requires_user_input",
            "response_goal": "ask for the missing detail",
            "action_requests": [],
            "resolver_requests": [{
                "capability": "human_clarification",
                "goal": "clarify the missing detail",
                "reason": "the current evidence is insufficient",
            }],
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
            goal_kind="ordinary_response",
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
        "response_plan": {
            "goal_resolution": "requires_user_input",
            "response_goal": "explain the missing detail",
            "action_requests": [],
            "resolver_requests": [
                {
                    "capability": "task_resolution_request",
                    "goal": "continue the task",
                    "reason": "durable task lineage is required",
                },
                {
                    "capability": "human_clarification",
                    "goal": "clarify the missing detail",
                    "reason": "the current evidence is insufficient",
                },
            ],
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
            }
        else:
            value = {
                "goal_resolution": "requires_user_input",
                "response_goal": "ask for clarification",
                "action_requests": [], "resolver_requests": [],
            }
        return SimpleNamespace(content=json.dumps(value, ensure_ascii=False))


@pytest.mark.asyncio
async def test_canonical_cognition_calls_a1_a2_g_p_once_with_one_goal() -> None:
    invoker = _FourStageInvoker()
    token = bind_protected_chain_records(run_id="deterministic-test")
    try:
        output = await run_cognition(_input(), _services(invoker))
        records = snapshot_protected_chain_records()
    finally:
        reset_protected_chain_records(token)
    assert invoker.calls == ["A1", "A2", "G", "P"]
    assert [config.output_mode for config in invoker.configs] == ["text"] * 4
    assert set(invoker.packets[0]) >= {"stage", "observation", "output_contract"}
    assert "character_context" not in invoker.packets[0]
    assert set(invoker.packets[1]["context"]) >= {"accepted_a1_meaning", "character_context"}
    assert "capabilities" not in invoker.packets[1]["context"]
    assert all("axis_changes" not in row for row in invoker.packets[1]["context"]["accepted_a1_meaning"])
    assert set(invoker.packets[2]) >= {"appraisal_summary", "character_context"}
    assert "output_contract" in invoker.packets[3]
    assert "appraisal_summary" not in invoker.packets[3]
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
