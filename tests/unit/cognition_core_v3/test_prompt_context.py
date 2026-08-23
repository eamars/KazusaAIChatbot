"""Focused deterministic proofs for typed stage context and evidence priority."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    build_canonical_appraisal_question,
    build_canonical_goal_question,
    build_canonical_turn_workspace,
)
from kazusa_ai_chatbot.cognition_shared.state_models import validate_cognition_state
from kazusa_ai_chatbot.cognition_shared.state_reducers import materialize_causal_root
from tests.unit.cognition_core_v3.test_handleless_contract import _input


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
    )


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
    context = a2["context"]
    assert context["character_context"]["identity"]["name"] == "Test Character"
    assert context["character_context"]["constraints"]["standards"]
    assert context["relationship_context"]["axes"]["trust"] == "信任尚未建立"
    assert context["affect_context"][0]["emotion"] == "sadness"
    assert context["affect_context"][0]["cause_summary"] == (
        "the current observation recalls a concrete loss"
    )
    operational = context["character_context"]["operational"]
    assert set(operational) == {"affect", "pressures"}
    assert context["character_context"]["constraints"]["standards"][0][
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
    assert set(a1["context"]["character_context"]["identity"]) == {
        "event_agency", "goal_threat_outcome", "epistemic_comparison_memory",
    }
    assert set(a2["context"]["character_context"]["identity"]) == {
        "relationship_social", "moral_identity", "existential_drive",
    }
    assert "constraints" not in a1["context"]["character_context"]
    assert "constraints" in a2["context"]["character_context"]


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
        for row in a1["observation"]["evidence"]
    )
    assert all(
        row["semantic_text"] != "a scheduled task is due"
        for row in a2["observation"]["evidence"]
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
        evidence = packet["observation"]["evidence"]
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
        affect_context = packet.get("affect_context")
        if affect_context is None:
            affect_context = packet["context"]["affect_context"]
        assert any(
            row.get("cause_summary") == "the character remembers a concrete loss"
            for row in affect_context
        )
