"""Focused deterministic proofs for typed stage context and evidence priority."""

from __future__ import annotations

import json
import re

from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    build_canonical_appraisal_question,
    build_canonical_goal_question,
    build_canonical_plan_question,
    build_canonical_turn_workspace,
    build_turn_workspace_stage_contracts,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    PENDING_TASK_CONTINUATION_VERSION,
    RESOLVER_PENDING_CONTINUATION_VERSION,
)
from kazusa_ai_chatbot.cognition_shared.state_models import validate_cognition_state
from kazusa_ai_chatbot.cognition_shared.state_reducers import materialize_causal_root
from tests.unit.cognition_core_v3.test_handleless_contract import _input

_CODE_SPAN_PATTERN = re.compile(r"`[^`]+`")
_HAN_PATTERN = re.compile(r"[\u3400-\u9fff]")
_MULTIWORD_ENGLISH_PATTERN = re.compile(
    r"\b[A-Za-z]+(?:[ \t]+[A-Za-z]+){2,}\b"
)




def _workspace(
    payload: dict[str, object],
    evidence: list[dict[str, object]],
    *,
    character_affect_context: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return build_canonical_turn_workspace(
        episode=payload["episode"],
        scene_context=payload["scene_context"],
        evidence=evidence,
        mutable_state=payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        identity_context=payload["character_identity_context"],
        available_actions=payload["available_actions"],
        available_resolvers=payload["available_resolver_capabilities"],
        character_operational_context=payload.get(
            "character_operational_context", {}
        ),
        relationship_context=payload["mutable_state"].get("relationship", {}),
        character_affect_context=character_affect_context,
        overused_moves=payload.get("overused_moves", []),
        response_plan_contract_variant=payload[
            "response_plan_contract_variant"
        ],
    )




def test_ordinary_plan_declares_response_goal_text_contract() -> None:
    """Keep the model-facing P schema aligned with deterministic validation."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    packet = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "clarify",
            "intent": "answer the current observation",
            "reason": "the current observation needs an answer",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )

    assert packet["output_contract"]["response_goal"] == {
        "type": "string",
        "minimum_characters": 1,
        "maximum_characters": 2000,
    }


def test_stage_context_preserves_identity_relationship_and_emotion_cause() -> None:
    payload = _input()
    payload["character_operational_context"] = {
        "schema_version": "operational.v1",
        "consumer_role": "cognition",
        "source_updated_at": payload["mutable_state"]["updated_at"],
        "effective_at": payload["mutable_state"]["updated_at"],
        "affect": [{"summary": "a concrete current feeling"}],
        "pressures": [{"summary": "a bounded current pressure"}],
    }
    affect_state = payload["mutable_state"]
    evidence = payload["evidence"][0]["evidence_ref"]
    affect_state, event_id, _created = materialize_causal_root(
        affect_state,
        kind="event",
        primary_evidence=evidence,
        description="the current observation recalls a concrete loss",
    )
    affect_state["active_events"][0]["responsibility"] = 40
    affect_state["active_events"][0]["salience"] = 40
    affect_state["affect_activations"] = [{
        "activation_id": "emotion:sadness",
        "emotion_id": "sadness",
        "primary_root": {
            "scope": "user", "kind": "event", "entity_id": event_id,
        },
        "root_refs": [{
            "scope": "user", "kind": "event", "entity_id": event_id,
        }],
        "phase": "active",
        "score": 65,
        "peak_score": 65,
        "trend": "rising",
        "cause_status": "active",
        "started_at": affect_state["updated_at"],
        "updated_at": affect_state["updated_at"],
        "last_reinforced_at": affect_state["updated_at"],
    }]
    validate_cognition_state(affect_state)
    payload["mutable_state"] = affect_state
    workspace = _workspace(payload, payload["evidence"])
    a2 = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A2",
        accepted_appraisal_summary=[{
            "family": "event_agency",
            "applicable": True,
            "semantic_summary": "the event has meaning",
            "cause_summary": "the observed action caused the change",
        }],
    )
    context = a2["conditional_character_context"]
    assert context["identity"]["name"] == "Test Character"
    assert context["constraints"]["standards"]
    assert context["relationship"]["axes"]["trust"] == "信任尚未建立"
    assert context["affect"][0]["emotion"] == "sadness"
    assert context["affect"][0]["cause_summary"] == (
        "the current observation recalls a concrete loss"
    )
    operational = context["operational"]
    assert set(operational) == {"affect", "pressures"}
    assert context["constraints"]["standards"][0][
        "importance"
    ] == "高"


def test_actual_identity_partitions_are_selected_by_stage() -> None:
    payload = _input()
    payload["character_identity_context"] = {
        family: {
            "core": {"name": "Test Character", "description": "grounded"},
            "personality": {"logic": "careful"},
            "boundaries": {"avoid_harm": "high"},
        }
        for family in (
            "event_agency", "goal_threat_outcome",
            "epistemic_comparison_memory", "relationship_social",
            "moral_identity", "existential_drive", "goal_cognition",
        )
    }
    workspace = _workspace(payload, payload["evidence"])
    a1 = build_canonical_appraisal_question(workspace=workspace, stage_name="A1")
    a2 = build_canonical_appraisal_question(workspace=workspace, stage_name="A2")
    goal = build_canonical_goal_question(
        workspace=workspace,
        appraisal_summary=[],
    )
    assert "conditional_character_context" not in a1
    assert set(a2["conditional_character_context"]["identity"]) == {
        "relationship_social", "moral_identity", "existential_drive",
    }
    assert set(goal["conditional_character_context"]["identity"]) == {
        "goal_cognition",
    }
    assert "constraints" in a2["conditional_character_context"]


def test_scheduler_visibility_reaches_a1_goal_threat_but_not_a2() -> None:
    payload = _input()
    scheduler = {
        "evidence_ref": {
            "source_kind": "scheduler_event",
            "source_id": "scheduler:due",
            "occurred_at": payload["mutable_state"]["updated_at"],
            "semantic_summary": "a scheduled task is due",
        },
        "semantic_text": "a scheduled task is due",
        "visible_to": ["q:goal_threat_outcome"],
        "authority": "current_event",
    }
    workspace = _workspace(payload, [*payload["evidence"], scheduler])
    a1 = build_canonical_appraisal_question(workspace=workspace, stage_name="A1")
    a2 = build_canonical_appraisal_question(workspace=workspace, stage_name="A2")
    assert any(
        row["semantic_text"] == "a scheduled task is due"
        for row in a1["current_observation"]["evidence"]
    )
    assert all(
        row["semantic_text"] != "a scheduled task is due"
        for row in a2["current_observation"]["evidence"]
    )


def test_current_resolver_and_action_evidence_precedes_supporting_rag() -> None:
    payload = _input()
    current_resolver = {
        "evidence_ref": {
            "source_kind": "resolver_observation",
            "source_id": "resolver:current",
            "occurred_at": payload["mutable_state"]["updated_at"],
            "semantic_summary": "current resolver result",
        },
        "semantic_text": "current resolver result",
        "authority": "supporting",
    }
    current_action = {
        "evidence_ref": {
            "source_kind": "action_result",
            "source_id": "action:current",
            "occurred_at": payload["mutable_state"]["updated_at"],
            "semantic_summary": "current action result",
        },
        "semantic_text": "current action result",
        "authority": "supporting",
    }
    rag_rows = [
        {
                "evidence_ref": {
                "source_kind": "promoted_memory",
                "source_id": f"rag:{index}",
                "occurred_at": payload["mutable_state"]["updated_at"],
                "semantic_summary": f"supporting memory {index}",
            },
            "semantic_text": f"supporting memory {index}",
            "authority": "supporting",
            "memory_scope": "current_user_continuity",
        }
        for index in range(64)
    ]
    workspace = _workspace(payload, [*rag_rows, current_resolver, current_action])
    a1 = build_canonical_appraisal_question(workspace=workspace, stage_name="A1")
    goal = build_canonical_goal_question(workspace=workspace, appraisal_summary=[])
    for packet in (a1, goal):
        evidence = packet["direct_facts"]["evidence"]
        assert len(evidence) == 32
        assert evidence[0]["semantic_text"] == "current resolver result"
        assert evidence[1]["semantic_text"] == "current action result"
        assert all(
            row["semantic_text"] != "supporting memory 0"
            for row in evidence[:2]
        )


def test_character_affect_projection_reaches_a2_and_goal_context() -> None:
    payload = _input()
    character_affect = [{
        "emotion": "sadness",
        "phase": "active",
        "intensity": "high",
        "trend": "rising",
        "cause_status": "active",
        "cause_summary": "the character remembers a concrete loss",
    }]
    workspace = _workspace(
        payload,
        payload["evidence"],
        character_affect_context=character_affect,
    )
    a2 = build_canonical_appraisal_question(workspace=workspace, stage_name="A2")
    goal = build_canonical_goal_question(workspace=workspace, appraisal_summary=[])
    for packet in (a2, goal):
        affect_context = packet["conditional_character_context"]["affect"]
        assert any(
            row.get("cause_summary") == "the character remembers a concrete loss"
            for row in affect_context
        )


def test_stage_authority_lanes_partition_fact_continuity_and_character_context() -> None:
    """Every semantic source reaches only its declared authority lane."""

    payload = _input()
    timestamp = payload["mutable_state"]["updated_at"]
    evidence = [
        *payload["evidence"],
        {
            "evidence_ref": {
                "source_kind": "resolver_observation",
                "source_id": "resolver:fact",
                "occurred_at": timestamp,
                "semantic_summary": "the resolver confirmed one fact",
            },
            "semantic_text": "the resolver confirmed one fact",
            "authority": "contextual_fact_only",
        },
        {
            "evidence_ref": {
                "source_kind": "conversation_evidence",
                "source_id": "conversation:prior",
                "occurred_at": timestamp,
                "semantic_summary": "the user discussed a prior action",
            },
            "semantic_text": "the user discussed a prior action",
            "authority": "participant_continuity",
        },
        {
            "evidence_ref": {
                "source_kind": "promoted_reflection",
                "source_id": "reflection:tendency",
                "occurred_at": timestamp,
                "semantic_summary": "the character tends to test reciprocity",
            },
            "semantic_text": "the character tends to test reciprocity",
            "authority": "conditional_character_guidance",
        },
    ]
    workspace = _workspace(payload, evidence)
    a1 = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A1",
    )
    a2 = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A2",
        accepted_appraisal_summary=[],
    )
    goal = build_canonical_goal_question(
        workspace=workspace,
        appraisal_summary=[],
    )
    plan = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "clarify",
            "intent": "clarify the observation",
            "reason": "the referent is unknown",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )

    assert [
        row["semantic_text"]
        for row in a1["current_observation"]["evidence"]
    ] == ["the current observation"]
    assert [
        row["semantic_text"] for row in a1["direct_facts"]["evidence"]
    ] == ["the resolver confirmed one fact"]
    assert "participant_continuity" not in a1
    assert "conditional_character_context" not in a1
    assert [
        row["semantic_text"] for row in a2["participant_continuity"]
    ] == ["the user discussed a prior action"]
    assert [
        row["semantic_text"]
        for row in a2["conditional_character_context"]["evidence"]
    ] == ["the character tends to test reciprocity"]
    assert goal["participant_continuity"] == a2["participant_continuity"]
    assert plan["participant_continuity"] == a2["participant_continuity"]
    assert "conditional_character_context" not in plan


def test_overused_moves_reach_participant_continuity_after_a1_only() -> None:
    """Expose observed response moves to A2/G/P while keeping A1 current-only."""

    payload = _input()
    payload["overused_moves"] = [
        "the character has already used one visible relationship maneuver",
        "the character has already used a second visible maneuver",
    ]
    workspace = _workspace(payload, payload["evidence"])

    a1 = build_canonical_appraisal_question(workspace=workspace, stage_name="A1")
    a2 = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A2",
        accepted_appraisal_summary=[],
    )
    goal = build_canonical_goal_question(workspace=workspace, appraisal_summary=[])
    plan = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "clarify",
            "intent": "answer the current observation",
            "reason": "the current observation needs an answer",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )

    assert "participant_continuity" not in a1
    assert all(move not in str(a1) for move in payload["overused_moves"])
    for packet in (a2, goal, plan):
        assert [
            row["semantic_text"]
            for row in packet["participant_continuity"][-2:]
        ] == payload["overused_moves"]


def test_semantic_progression_context_preserves_all_existing_multi_affect_rows_and_causes() -> None:
    """Adding bounded moves cannot alter any continuation or affect row."""

    payload = _input()
    state = payload["mutable_state"]
    timestamp = state["updated_at"]
    emotions = [
        ("sadness", "a concrete loss remains unresolved"),
        ("anger", "a boundary was crossed in the current event"),
        ("gratitude", "a specific act of care was received"),
        ("embarrassment", "a private mistake became visible"),
        ("nostalgia", "a remembered shared moment was recalled"),
    ]
    for index, (emotion, cause) in enumerate(emotions):
        state, root_id, _created = materialize_causal_root(
            state,
            kind="event",
            primary_evidence={
                "source_kind": "episode",
                "source_id": f"episode:multi-affect-{index}",
                "occurred_at": timestamp,
                "semantic_summary": cause,
            },
            description=cause,
        )
        state["active_events"][-1]["salience"] = 70 - index
        state["affect_activations"].append({
            "activation_id": f"emotion:{emotion}",
            "emotion_id": emotion,
            "primary_root": {
                "scope": "user",
                "kind": "event",
                "entity_id": root_id,
            },
            "root_refs": [{
                "scope": "user",
                "kind": "event",
                "entity_id": root_id,
            }],
            "phase": "active",
            "score": 70 - index,
            "peak_score": 70 - index,
            "trend": "rising",
            "cause_status": "active",
            "started_at": timestamp,
            "updated_at": timestamp,
            "last_reinforced_at": timestamp,
        })
    validate_cognition_state(state)
    payload["mutable_state"] = state
    baseline = _workspace(payload, payload["evidence"])
    payload["overused_moves"] = [
        "a" * 120,
        "b" * 120,
        "c" * 120,
        "d" * 120,
    ]
    candidate = _workspace(payload, payload["evidence"])
    baseline_packets = build_turn_workspace_stage_contracts(workspace=baseline)
    candidate_packets = build_turn_workspace_stage_contracts(workspace=candidate)

    for stage in ("A1", "A2", "G", "P"):
        assert candidate_packets[stage]["continuation_state"] == baseline_packets[stage]["continuation_state"]
        continuation = candidate_packets[stage]["continuation_state"]
        assert [
            (row["description"], row["status"])
            for row in continuation["active_events"]
        ] == [(cause, "active") for _emotion, cause in emotions]
        expected_emotions = {emotion for emotion, _cause in emotions}
        assert [
            (row["emotion"], row["cause_summary"])
            for row in continuation["affect_activations"]
            if row["emotion"] in expected_emotions
        ] == emotions
    for stage in ("A2", "G"):
        assert candidate_packets[stage]["conditional_character_context"]["affect"] == (
            baseline_packets[stage]["conditional_character_context"]["affect"]
        )
        assert [
            (row["emotion"], row["cause_summary"])
            for row in candidate_packets[stage][
                "conditional_character_context"
            ]["affect"]
        ] == emotions


def test_a1_excludes_conditional_character_context() -> None:
    """A1 remains a world-facing appraisal over facts and causal pressure."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    packet = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A1",
    )

    rendered = str(packet)
    assert "conditional_character_context" not in packet
    assert "character_context" not in packet
    assert "personality" not in rendered


def test_a1_packet_separates_current_observation_from_stable_direct_facts() -> None:
    """A1 keeps stable facts outside the current-observation authority lane."""

    payload = _input()
    timestamp = payload["mutable_state"]["updated_at"]
    stable_fact = {
        "evidence_ref": {
            "source_kind": "promoted_memory",
            "source_id": "promoted-memory:fact:stable-1",
            "occurred_at": timestamp,
            "semantic_summary": "stable background fact",
        },
        "semantic_text": "stable background fact",
        "visible_to": ["q:event_agency"],
        "authority": "character_world_context",
        "memory_scope": "shared_character_or_world",
    }
    workspace = _workspace(payload, [payload["evidence"][0], stable_fact])

    packet = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A1",
    )

    current_text = [
        row["semantic_text"]
        for row in packet["current_observation"]["evidence"]
    ]
    direct_text = [
        row["semantic_text"]
        for row in packet["direct_facts"]["evidence"]
    ]
    assert "the current observation" in current_text
    assert "stable background fact" in direct_text
    assert "stable background fact" not in current_text


def test_packets_project_only_state_and_typed_output_contracts() -> None:
    payload = _input()
    packets = build_turn_workspace_stage_contracts(
        workspace=_workspace(payload, payload["evidence"]),
    )

    for packet in packets.values():
        assert "guidance" not in packet
        assert "current_observation" in packet
        assert "output_contract" in packet
    rendered = json.dumps(packets, ensure_ascii=False)
    assert "repair_instruction" not in rendered
    assert "source_id" not in rendered


def test_plan_variant_packets_match_projected_pending_lanes() -> None:
    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    ordinary = build_canonical_plan_question(
        workspace=workspace,
        goal={"goal_kind": "open_goal"},
        appraisal_summary=[],
    )
    assert "pending_resolver_continuation" not in ordinary

    workspace["pending_resolver_continuation"] = {
        "schema_version": RESOLVER_PENDING_CONTINUATION_VERSION,
        "capability_kind": "human_clarification",
        "status": "waiting_for_user",
        "original_goal": "取得带背景要求的证据。",
        "question": "请补充地点。",
        "pending_task_continuation": {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "background_task_admission",
        },
    }
    workspace["response_plan_contract_variant"] = "open_pending_resolution"
    pending = build_canonical_plan_question(
        workspace=workspace,
        goal={"goal_kind": "open_goal"},
        appraisal_summary=[],
    )
    assert "pending_resolver_continuation" in pending
    assert "response_plan_contract_variant" not in json.dumps(pending)

    post_workspace = _workspace(payload, payload["evidence"])
    post_workspace["response_plan_contract_variant"] = "post_pending_resolution"
    post_workspace["capabilities"] = {
        **post_workspace["capabilities"],
        "resolvers": [
            row
            for row in post_workspace["capabilities"]["resolvers"]
            if row["capability"] != "human_clarification"
        ],
    }
    post_pending = build_canonical_plan_question(
        workspace=post_workspace,
        goal={"goal_kind": "open_goal"},
        appraisal_summary=[],
    )
    assert "response_plan_contract_variant" not in post_pending["output_contract"]
    assert "pending_resolver_continuation" not in post_pending
    assert "pending_task_continuation" not in post_pending["output_contract"]
    assert "pending_resolution_fields" not in post_pending["output_contract"]
    assert "post_pending_resolution" not in json.dumps(post_pending)
    assert all(
        row["capability"] != "human_clarification"
        for row in post_pending["capabilities"]["resolvers"]
    )


def test_resolver_goal_progress_survives_without_pending_clarification() -> None:
    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    workspace["resolver_goal_progress"] = {
        "schema_version": "resolver_goal_progress.v1",
        "original_goal": "完成带背景要求的证据任务。",
        "current_focus": "取得证据。",
        "deliverables": [],
        "missing_user_inputs": [],
        "evidence_dependencies": ["来源证据"],
        "attempted_paths": [],
        "source_backed_facts": [],
        "assumptions_or_inferences": [],
        "blockers": [],
        "final_response_requirements": ["精确标记：已完成"],
    }
    packets = build_turn_workspace_stage_contracts(workspace=workspace)

    for packet in packets.values():
        assert packet["resolver_goal_progress"]["original_goal"] == (
            "完成带背景要求的证据任务。"
        )
        assert "pending_resolver_continuation" not in packet
    assert "pending_resolution" not in packets["P"]["output_contract"][
        "required_fields"
    ]
