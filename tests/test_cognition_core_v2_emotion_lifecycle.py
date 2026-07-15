"""Checkpoint C typed V2 emotion formula and lifecycle tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
    EMOTION_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.emotion_derivation import (
    derive_emotion_activation_v2,
    derive_persistent_emotion_activations,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
    validate_cognition_state,
)


NOW = "2026-07-14T00:00:00Z"


def _evidence(source_kind: str, source_id: str) -> dict[str, str]:
    """Build one complete V2 evidence record."""

    return {
        "source_kind": source_kind,
        "source_id": source_id,
        "occurred_at": NOW,
        "semantic_summary": f"typed evidence for {source_id}",
    }


def _role(role: str, entity_kind: str, entity_id: str) -> dict[str, str]:
    """Build one structured role reference."""

    return {
        "role": role,
        "entity_kind": entity_kind,
        "entity_id": entity_id,
    }


def _event(
    event_id: str,
    *,
    roles: list[dict[str, str]] | None = None,
    evidence: list[dict[str, str]] | None = None,
    **axes: int,
) -> dict[str, object]:
    """Build one complete typed event with zeroed non-causal axes."""

    fields = {
        "outcome_impact": 0,
        "responsibility": 0,
        "intentionality": 0,
        "harm": 0,
        "unfairness": 0,
        "exposure": 0,
        "repair_need": 0,
        "reparability": 80,
        "expectation_mismatch": 0,
        "norm_violation": 0,
        "contamination_risk": 0,
        "identity_threat": 0,
        "comparison_gap": 0,
        "vastness": 0,
        "memory_warmth": 0,
        "temporal_loss": 0,
    }
    fields.update(axes)
    return {
        "entity_id": event_id,
        "description": f"typed event {event_id}",
        "salience": 70,
        "role_refs": roles or [],
        "evidence_refs": evidence or [_evidence("episode", event_id)],
        "created_at": NOW,
        "updated_at": NOW,
        "status": "active",
        **fields,
    }


def _goal(
    goal_id: str,
    *,
    status: str = "pursuing",
    role_refs: list[dict[str, str]] | None = None,
    progress: int = 0,
    recoverability: int = 60,
    obstruction: int = 0,
) -> dict[str, object]:
    """Build one complete typed goal."""

    return {
        "entity_id": goal_id,
        "description": f"typed goal {goal_id}",
        "salience": 70,
        "role_refs": role_refs or [],
        "evidence_refs": [_evidence("episode", goal_id)],
        "created_at": NOW,
        "updated_at": NOW,
        "status": status,
        "goal_kind": "ordinary_response",
        "importance": 70,
        "progress": progress,
        "obstruction": obstruction,
        "expected_success": 60,
        "controllability": 60,
        "recoverability": recoverability,
        "urgency": 70,
    }


def _threat(threat_id: str) -> dict[str, object]:
    """Build one complete typed threat with a third-party target."""

    return {
        "entity_id": threat_id,
        "description": "a third party threatens an attached bond",
        "salience": 70,
        "role_refs": [
            _role("target", "third_party", "third-party-1"),
            _role("object", "relationship", "relationship:user:user-cause"),
        ],
        "evidence_refs": [_evidence("episode", threat_id)],
        "created_at": NOW,
        "updated_at": NOW,
        "status": "active",
        "likelihood": 80,
        "expected_harm": 70,
        "uncertainty": 60,
        "controllability": 40,
        "coping_potential": 30,
        "residual_pressure": 70,
    }


def _gap(gap_id: str) -> dict[str, object]:
    """Build one complete typed knowledge gap."""

    return {
        "entity_id": gap_id,
        "description": "an important unresolved question",
        "salience": 70,
        "role_refs": [],
        "evidence_refs": [_evidence("episode", gap_id)],
        "created_at": NOW,
        "updated_at": NOW,
        "status": "open",
        "relevance": 80,
        "uncertainty": 70,
        "learnability": 80,
        "novelty": 70,
        "model_accommodation": 70,
    }


def _user_case_state() -> dict[str, object]:
    """Build a user state containing typed roots for the formula matrix."""

    state = build_acquaintance_user_state(
        global_user_id="user-cause",
        updated_at=NOW,
    )
    state["relationship"].update({
        "relationship_id": "relationship:user:user-cause",
        "attachment": 70,
        "positive_regard": 70,
        "trust": 70,
        "care": 70,
        "desired_closeness": 80,
        "perceived_closeness": 30,
        "exclusivity": 70,
        "salience": 70,
    })
    state["goals"] = [
        _goal(
            "goal:reward",
            status="satisfied",
            progress=100,
            role_refs=[_role("actor", "character", "character:global")],
        ),
        _goal("goal:block", status="blocked", obstruction=70),
        _goal("goal:loss", status="failed", recoverability=10),
    ]
    state["threats"] = [_threat("threat:rival")]
    state["knowledge_gaps"] = [_gap("gap:curiosity")]
    state["active_events"] = [
        _event(
            "event:positive",
            roles=[_role("actor", "third_party", "helper-1")],
            outcome_impact=70,
            responsibility=70,
        ),
        _event(
            "event:anger",
            roles=[_role("actor", "third_party", "offender-1")],
            harm=70,
            unfairness=70,
            intentionality=70,
        ),
        _event(
            "event:loss",
            outcome_impact=-70,
            harm=70,
            temporal_loss=70,
            repair_need=70,
        ),
        _event(
            "event:disgust",
            roles=[_role("target", "object", "contaminated-object")],
            contamination_risk=80,
            norm_violation=70,
        ),
        _event(
            "event:surprise",
            expectation_mismatch=80,
        ),
        _event(
            "event:compassion",
            roles=[_role("experiencer", "third_party", "hurt-person")],
            harm=80,
        ),
        _event(
            "event:envy",
            roles=[
                _role("actor", "third_party", "successful-peer"),
                _role("object", "object", "owned-object"),
            ],
            comparison_gap=80,
        ),
        _event(
            "event:pride",
            roles=[_role("actor", "character", "character:global")],
            outcome_impact=80,
            responsibility=80,
        ),
        _event(
            "event:shame",
            roles=[_role("actor", "character", "character:global")],
            responsibility=80,
            norm_violation=80,
            identity_threat=80,
        ),
        _event(
            "event:guilt",
            roles=[_role("actor", "character", "character:global")],
            responsibility=80,
            harm=80,
            repair_need=80,
        ),
        _event(
            "event:embarrassment",
            roles=[_role("actor", "character", "character:global")],
            responsibility=70,
            exposure=70,
            expectation_mismatch=70,
        ),
        _event(
            "event:awe",
            roles=[_role("affected_goal", "goal", "gap:curiosity")],
            vastness=80,
        ),
        _event(
            "event:nostalgia",
            evidence=[
                _evidence("promoted_memory", "memory:old"),
                _evidence("episode", "cue:present"),
            ],
            memory_warmth=80,
            temporal_loss=70,
        ),
    ]
    validate_cognition_state(state)
    return state


def _character_case_state() -> dict[str, object]:
    """Build character-only state for meaning and sleep-owned constraints."""

    state = build_character_production_state(updated_at=NOW)
    state["meaning_state"].update({
        "purpose_coherence": 20,
        "agency": 20,
        "salience": 70,
        "low_coherence_since": "2026-07-12T00:00:00Z",
    })
    state["drives"]["meaning"]["pressure"] = 80
    validate_cognition_state(state)
    return state


def _character_constraints() -> dict[str, object]:
    """Build the cross-scope character axes used by emotion formulas."""

    return {
        "drives": {
            "care": {"importance": 80, "pressure": 80},
            "competence": {"importance": 80, "pressure": 80},
            "connection": {"importance": 80, "pressure": 80},
        },
        "meaning_state": {"identity_continuity": 80},
    }


EMOTION_CASES = tuple(EMOTION_DEFINITIONS)


@pytest.mark.parametrize("emotion_id", EMOTION_CASES)
def test_each_emotion_derives_from_typed_state_causes(emotion_id: str) -> None:
    """Prove every registry emotion has a typed V2 cause and activation."""

    state = (
        _character_case_state()
        if emotion_id == "ennui_existential_angst"
        else _user_case_state()
    )
    constraints = _character_constraints()
    transitions = {
        "root_ref": {
            "scope": "user",
            "kind": "threat",
            "entity_id": "threat:rival",
        },
        "prior": {"status": "active", "residual_pressure": 80},
        "current": {"status": "resolved", "residual_pressure": 10},
        "evidence_ref": _evidence("action_result", "relief:transition"),
        "salience": 70,
    }
    activations = derive_persistent_emotion_activations(
        state,
        updated_at=NOW,
        character_constraints=constraints,
        transition_contexts=[transitions] if emotion_id == "relief" else [],
    )

    by_id = {activation["emotion_id"]: activation for activation in activations}
    assert emotion_id in by_id
    assert by_id[emotion_id]["score"] >= 40
    assert by_id[emotion_id]["primary_root"]["kind"] in {
        "relationship",
        "goal",
        "threat",
        "event",
        "knowledge_gap",
        "meaning",
    }


def test_typed_formula_negative_controls_require_roles_and_evidence() -> None:
    """Prevent emotion onset from axis values without the required cause shape."""

    state = _user_case_state()
    disgust_event = next(
        event for event in state["active_events"]
        if event["entity_id"] == "event:disgust"
    )
    disgust_event["role_refs"] = []
    gratitude_event = next(
        event for event in state["active_events"]
        if event["entity_id"] == "event:positive"
    )
    gratitude_event["role_refs"] = [_role("actor", "user", "user-cause")]
    nostalgia_event = next(
        event for event in state["active_events"]
        if event["entity_id"] == "event:nostalgia"
    )
    nostalgia_event["evidence_refs"] = [_evidence("episode", "cue:present")]

    activations = derive_persistent_emotion_activations(
        state,
        updated_at=NOW,
        character_constraints=_character_constraints(),
    )
    active_ids = {activation["emotion_id"] for activation in activations}

    assert "disgust" not in active_ids
    assert "gratitude" in active_ids
    assert "nostalgia" not in active_ids


def test_owner_user_is_other_and_only_character_actor_is_self() -> None:
    """Keep user-document ownership separate from character self identity."""

    state = build_acquaintance_user_state(
        global_user_id="owner-user",
        updated_at=NOW,
    )
    state["active_events"] = [_event(
        "event:owner-helped",
        roles=[_role("actor", "user", "owner-user")],
        outcome_impact=80,
        responsibility=80,
    )]
    constraints = _character_constraints()

    activations = derive_persistent_emotion_activations(
        state,
        updated_at=NOW,
        character_constraints=constraints,
    )
    active_ids = {row["emotion_id"] for row in activations}
    assert "gratitude" in active_ids
    assert "pride" not in active_ids

    state["active_events"] = [_event(
        "event:self-helped",
        roles=[_role("actor", "character", "character:global")],
        outcome_impact=80,
        responsibility=80,
    )]
    activations = derive_persistent_emotion_activations(
        state,
        updated_at=NOW,
        character_constraints=constraints,
    )
    active_ids = {row["emotion_id"] for row in activations}
    assert "pride" in active_ids
    assert "gratitude" not in active_ids


def test_compassion_threat_uses_character_care_across_user_scope() -> None:
    """Read care from character constraints for another user's pressure."""

    state = build_acquaintance_user_state(
        global_user_id="other-experiencer",
        updated_at=NOW,
    )
    threat = _threat("threat:user-under-pressure")
    threat["role_refs"] = [
        _role("experiencer", "user", "other-experiencer"),
    ]
    state["threats"] = [threat]

    activations = derive_persistent_emotion_activations(
        state,
        updated_at=NOW,
        character_constraints=_character_constraints(),
    )
    active_ids = {row["emotion_id"] for row in activations}
    assert "compassion_empathy" in active_ids


def test_anger_and_sadness_cannot_borrow_unrelated_goals() -> None:
    """Require event-to-goal affected_goal links before using goal axes."""

    state = build_acquaintance_user_state(
        global_user_id="goal-link-user",
        updated_at=NOW,
    )
    state["goals"] = [
        _goal("goal:unrelated-block", status="blocked", obstruction=90),
        _goal(
            "goal:unrelated-loss",
            status="failed",
            recoverability=0,
        ),
    ]
    state["active_events"] = [
        _event(
            "event:weak-anger",
            unfairness=30,
            intentionality=30,
        ),
        _event("event:weak-loss", outcome_impact=-30),
    ]

    activations = derive_persistent_emotion_activations(
        state,
        updated_at=NOW,
        character_constraints=_character_constraints(),
    )
    active_ids = {row["emotion_id"] for row in activations}
    assert "anger" not in active_ids
    sadness = next(row for row in activations if row["emotion_id"] == "sadness")
    assert sadness["primary_root"] == {
        "scope": "user",
        "kind": "goal",
        "entity_id": "goal:unrelated-loss",
    }


def test_activation_lifecycle_preserves_resolved_score_until_elapsed_decay() -> None:
    """Keep recomputation idempotent and decay resolved causes exactly once."""

    root_ref = {
        "scope": "user",
        "kind": "event",
        "entity_id": "event:resolved",
    }
    previous = {
        "activation_id": "emotion:joy",
        "emotion_id": "joy",
        "primary_root": root_ref,
        "root_refs": [root_ref],
        "phase": "fading",
        "score": 60,
        "peak_score": 60,
        "trend": "stable",
        "cause_status": "resolved",
        "started_at": NOW,
        "updated_at": NOW,
        "last_reinforced_at": NOW,
    }
    recomputed = derive_emotion_activation_v2(
        "joy",
        candidates=[],
        previous=previous,
        updated_at=NOW,
    )

    assert recomputed is not None
    assert recomputed["score"] == 60
    assert recomputed["phase"] == "fading"

    beginning = derive_emotion_activation_v2(
        "joy",
        candidates=[{
            "root_ref": root_ref,
            "score": 60,
            "cause_status": "active",
            "salience": 60,
        }],
        previous=None,
        updated_at=NOW,
    )
    sustained = derive_emotion_activation_v2(
        "joy",
        candidates=[{
            "root_ref": root_ref,
            "score": 60,
            "cause_status": "active",
            "salience": 60,
        }],
        previous=beginning,
        updated_at="2026-07-14T01:00:00Z",
    )
    assert beginning["phase"] == "active"
    assert sustained["phase"] == "active"
    assert sustained["last_reinforced_at"] == NOW


def test_fading_activation_requires_begin_threshold_to_reactivate() -> None:
    """Require supported reinforcement at 40 before a fading row reactivates."""

    root_ref = {
        "scope": "user",
        "kind": "event",
        "entity_id": "event:fading",
    }
    previous = {
        "activation_id": "emotion:joy",
        "emotion_id": "joy",
        "primary_root": root_ref,
        "root_refs": [root_ref],
        "phase": "fading",
        "score": 30,
        "peak_score": 60,
        "trend": "falling",
        "cause_status": "active",
        "started_at": NOW,
        "updated_at": NOW,
        "last_reinforced_at": NOW,
    }
    fading = derive_emotion_activation_v2(
        "joy",
        candidates=[{
            "root_ref": root_ref,
            "score": 30,
            "cause_status": "active",
            "salience": 60,
        }],
        previous=previous,
        updated_at=NOW,
    )
    active = derive_emotion_activation_v2(
        "joy",
        candidates=[{
            "root_ref": root_ref,
            "score": 40,
            "cause_status": "active",
            "salience": 60,
        }],
        previous=previous,
        updated_at="2026-07-14T01:00:00Z",
        reinforced=True,
    )

    assert fading is not None
    assert fading["phase"] == "fading"
    assert active is not None
    assert active["phase"] == "active"
    assert active["last_reinforced_at"] == "2026-07-14T01:00:00Z"


def test_score_rise_without_reinforce_outcome_does_not_refresh_timestamp() -> None:
    """Keep last reinforcement tied to the reducer's matched outcome."""

    root_ref = {
        "scope": "user",
        "kind": "event",
        "entity_id": "event:rise",
    }
    previous = {
        "activation_id": "emotion:joy",
        "emotion_id": "joy",
        "primary_root": root_ref,
        "root_refs": [root_ref],
        "phase": "active",
        "score": 30,
        "peak_score": 30,
        "trend": "stable",
        "cause_status": "active",
        "started_at": NOW,
        "updated_at": NOW,
        "last_reinforced_at": NOW,
    }

    updated = derive_emotion_activation_v2(
        "joy",
        candidates=[{
            "root_ref": root_ref,
            "score": 45,
            "cause_status": "active",
            "salience": 60,
        }],
        previous=previous,
        updated_at="2026-07-14T01:00:00Z",
    )

    assert updated is not None
    assert updated["last_reinforced_at"] == NOW
