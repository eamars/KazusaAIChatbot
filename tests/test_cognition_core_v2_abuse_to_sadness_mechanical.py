"""Deterministic proof of the accepted abuse-to-sadness state path."""

from __future__ import annotations

import json
from pathlib import Path

from kazusa_ai_chatbot.cognition_core_v2.emotion_derivation import (
    derive_persistent_emotion_activations,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_semantic_appraisals,
    apply_state_update,
)


_FIXTURE_PATH = Path(
    "tests/fixtures/cognition_core_v2_abuse_to_sadness_e2e_cases.json"
)
_EVENT_AXES = (
    "outcome_impact",
    "responsibility",
    "intentionality",
    "harm",
    "unfairness",
    "exposure",
    "repair_need",
    "reparability",
    "expectation_mismatch",
    "norm_violation",
    "contamination_risk",
    "identity_threat",
    "comparison_gap",
    "vastness",
    "memory_warmth",
    "temporal_loss",
)


def _load_fixture() -> dict[str, object]:
    """Load the same Chinese abuse precondition used by the live probe."""

    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("abuse-to-sadness fixture root is invalid")
    return payload


def _build_event(
    event_spec: dict[str, object],
    *,
    user_id: str,
    occurred_at: str,
) -> dict[str, object]:
    """Build one complete typed abuse event with neutral outcome."""

    entity_id = event_spec["entity_id"]
    description = event_spec["description"]
    salience = event_spec["salience"]
    if not all(isinstance(value, str) for value in (entity_id, description)):
        raise ValueError("mechanical event identity is invalid")
    if not isinstance(salience, int) or isinstance(salience, bool):
        raise ValueError("mechanical event salience is invalid")
    axes: dict[str, int] = {}
    for axis in _EVENT_AXES:
        default = 80 if axis == "reparability" else 0
        value = event_spec.get(axis, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"mechanical event axis is invalid: {axis}")
        axes[axis] = value
    return {
        "entity_id": entity_id,
        "description": description,
        "salience": salience,
        "role_refs": [{
            "role": "actor",
            "entity_kind": "user",
            "entity_id": user_id,
        }],
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": "mechanical-abuse-event",
            "occurred_at": occurred_at,
            "semantic_summary": description,
        }],
        "created_at": occurred_at,
        "updated_at": occurred_at,
        "status": str(event_spec.get("status", "active")),
        **axes,
    }


def _character_constraints() -> dict[str, object]:
    """Build the character-side constraints used by deterministic derivation."""

    character_state = build_character_production_state(
        updated_at="2026-07-21T00:00:00Z",
    )
    return {
        "drives": character_state["drives"],
        "standards": character_state["standards"],
        "meaning_state": character_state["meaning_state"],
    }


def test_negative_abuse_outcome_reaches_sadness_mechanically() -> None:
    """Verify semantic delta, bounded reducer, and sadness derivation in order."""

    payload = _load_fixture()
    relationship_spec = payload["relationship_seed"]
    event_spec = payload["abuse_event"]
    proof_spec = payload["mechanical_proof"]
    if not isinstance(relationship_spec, dict):
        raise ValueError("mechanical relationship precondition is invalid")
    if not isinstance(event_spec, dict):
        raise ValueError("mechanical abuse event is invalid")
    if not isinstance(proof_spec, dict):
        raise ValueError("mechanical proof specification is invalid")
    outcome_delta = proof_spec.get("outcome_impact_delta")
    reason = proof_spec.get("reason")
    if not isinstance(outcome_delta, int) or outcome_delta >= 0:
        raise ValueError("mechanical proof must use a negative outcome delta")
    if not isinstance(reason, str) or not reason:
        raise ValueError("mechanical proof reason is invalid")

    user_id = "mechanical-abuse-user"
    now = "2026-07-21T00:00:00Z"
    state = build_acquaintance_user_state(
        global_user_id=user_id,
        updated_at=now,
    )
    relationship = state["relationship"]
    if not isinstance(relationship, dict):
        raise ValueError("mechanical relationship state is invalid")
    for field_name, value in relationship_spec.items():
        if field_name == "description":
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"mechanical relationship axis is invalid: {field_name}")
        relationship[field_name] = value
    event = _build_event(
        event_spec,
        user_id=user_id,
        occurred_at=now,
    )
    state["active_events"] = [event]
    constraints = _character_constraints()
    state["affect_activations"] = derive_persistent_emotion_activations(
        state,
        updated_at=now,
        character_constraints=constraints,
    )
    validate_cognition_state(state)
    before_emotions = {
        str(row["emotion_id"])
        for row in state["affect_activations"]
        if isinstance(row, dict) and row.get("emotion_id")
    }
    assert event["outcome_impact"] == 0
    assert "anger" in before_emotions
    assert "sadness" not in before_emotions

    evidence = [{
        "evidence_handle": "e1",
        "semantic_text": str(event_spec["description"]),
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "mechanical-abuse-event",
            "occurred_at": now,
            "semantic_summary": str(event_spec["description"]),
        },
        "visible_to": ["q:goal_threat_outcome"],
    }]
    handle_to_ref = {
        "e1": {
            "scope": "user",
            "kind": "event",
            "entity_id": str(event["entity_id"]),
        }
    }
    appraisal = [{
        "question_id": "q:goal_threat_outcome",
        "selected_evidence_handles": ["e1"],
        "selected_role_handles": ["e1"],
        "propositions": [],
        "deltas": [{
            "target_path": "active_events.e1.outcome_impact",
            "delta": outcome_delta,
            "evidence_handles": ["e1"],
            "reason": reason,
        }],
        "explanation": reason,
    }]
    reduced = apply_semantic_appraisals(
        state,
        appraisal,
        evidence,
        handle_to_ref,
    )
    reduced = apply_state_update(
        reduced,
        updated_at=now,
        character_constraints=constraints,
    )
    validate_cognition_state(reduced)

    reduced_event = reduced["active_events"][0]
    after_emotions = {
        str(row["emotion_id"]): row
        for row in reduced["affect_activations"]
        if isinstance(row, dict) and row.get("emotion_id")
    }
    assert reduced_event["outcome_impact"] == outcome_delta
    assert "sadness" in after_emotions
    assert after_emotions["sadness"]["primary_root"]["entity_id"] == (
        event["entity_id"]
    )
    assert after_emotions["sadness"]["score"] >= 40
    assert "anger" in after_emotions
