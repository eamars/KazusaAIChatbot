"""Checkpoint C deterministic failure and transition contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_elapsed_decay,
    apply_sleep_recovery,
    apply_state_update,
    canonical_event_entity_id,
    reduce_causal_event,
)
from kazusa_ai_chatbot.cognition_core_v2.transition_guards import (
    apply_direct_fact,
    apply_semantic_deltas,
    compare_event,
    transition_event,
    transition_goal,
    transition_knowledge_gap,
    transition_threat,
)


def _evidence(source_kind: str = "action_result") -> dict[str, str]:
    """Build one complete typed evidence record."""

    return {
        "source_kind": source_kind,
        "source_id": "action-c",
        "occurred_at": "2026-07-14T00:00:00Z",
        "semantic_summary": "typed result evidence",
    }


def _goal(
    *,
    status: str = "pursuing",
    obstruction: int = 0,
    recoverability: int = 50,
) -> dict[str, object]:
    """Build one complete bounded goal fixture."""

    return {
        "entity_id": "goal:ordinary_response:user:user-c:root-1",
        "description": "complete the bounded fixture goal",
        "status": status,
        "goal_kind": "ordinary_response",
        "importance": 70,
        "progress": 0,
        "obstruction": obstruction,
        "expected_success": 50,
        "controllability": 50,
        "recoverability": recoverability,
        "urgency": 40,
        "salience": 70,
        "role_refs": [],
        "evidence_refs": [_evidence()],
        "created_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
    }


def _state_with_goal(**goal_kwargs: object) -> dict[str, object]:
    """Build one user state with a single mutable goal."""

    state = build_acquaintance_user_state(
        global_user_id="user-c",
        updated_at="2026-07-14T00:00:00Z",
    )
    state["goals"] = [_goal(**goal_kwargs)]
    return state


def _fact(
    fact_kind: str,
    target_kind: str,
    entity_id: str,
    *,
    observed_progress: int | None = None,
    source_kind: str = "action_result",
) -> dict[str, object]:
    """Build one canonical typed direct fact."""

    fact: dict[str, object] = {
        "fact_id": f"fact:{fact_kind}:{entity_id}",
        "fact_kind": fact_kind,
        "target_refs": [{
            "scope": "user",
            "kind": target_kind,
            "entity_id": entity_id,
        }],
        "evidence_ref": _evidence(source_kind),
    }
    if observed_progress is not None:
        fact["observed_progress"] = observed_progress
    return fact


def test_invalid_direct_fact_leaves_state_unchanged() -> None:
    """Reject an untrusted producer before any state mutation."""

    state = _state_with_goal()
    before = deepcopy(state)

    with pytest.raises(CognitionStateError):
        apply_direct_fact(
            state,
            _fact("goal_completed", "goal", state["goals"][0]["entity_id"]),
            producer="dialog_text",
        )

    assert state == before


def test_direct_fact_rejects_extra_fields_and_terminal_mutation() -> None:
    """Keep the direct-fact lane closed to invented fields and terminal rows."""

    state = _state_with_goal()
    extra = _fact("goal_completed", "goal", state["goals"][0]["entity_id"])
    extra["observed_progress"] = 100
    with pytest.raises(CognitionStateError):
        apply_direct_fact(state, extra, producer="action_result")

    terminal_state = _state_with_goal(status="satisfied")
    with pytest.raises(CognitionStateError):
        apply_direct_fact(
            terminal_state,
            _fact("goal_completed", "goal", terminal_state["goals"][0]["entity_id"]),
            producer="action_result",
        )


def test_progress_observation_at_full_progress_completes_goal() -> None:
    """Treat trusted observed progress at 100 as deterministic completion."""

    state = _state_with_goal()
    updated = apply_direct_fact(
        state,
        _fact(
            "goal_progress_observed",
            "goal",
            state["goals"][0]["entity_id"],
            observed_progress=100,
            source_kind="resolver_observation",
        ),
        producer="resolver_observation",
    )

    assert updated["goals"][0]["progress"] == 100
    assert updated["goals"][0]["status"] == "satisfied"
    assert updated["goals"][0]["evidence_refs"][1]["source_kind"] == (
        "resolver_observation"
    )


def test_direct_fact_rejects_evidence_producer_mismatch() -> None:
    """Keep evidence provenance aligned with the trusted producer."""

    state = _state_with_goal()
    fact = _fact(
        "goal_progress_observed",
        "goal",
        state["goals"][0]["entity_id"],
        observed_progress=20,
        source_kind="action_result",
    )
    with pytest.raises(CognitionStateError):
        apply_direct_fact(state, fact, producer="resolver_observation")


def test_direct_fact_source_occurrence_preserves_full_evidence() -> None:
    """Allow scheduler occurrence facts without reducing provenance to a string."""

    state = _state_with_goal()
    goal_id = state["goals"][0]["entity_id"]
    fact = _fact(
        "source_occurred",
        "goal",
        goal_id,
        source_kind="scheduler_event",
    )

    scheduled = apply_direct_fact(state, fact, producer="scheduler_event")

    assert scheduled["goals"][0]["evidence_refs"][-1] == fact["evidence_ref"]


def test_source_occurrence_can_address_one_structured_role_target() -> None:
    """Append occurrence evidence through a unique role-owned causal entity."""

    state = _state_with_goal()
    goal_id = state["goals"][0]["entity_id"]
    state["active_events"] = [{
        "entity_id": "event:role-target",
        "description": "an event carrying a target role",
        "status": "active",
        "outcome_impact": 0,
        "responsibility": 0,
        "intentionality": 0,
        "harm": 0,
        "unfairness": 0,
        "exposure": 0,
        "repair_need": 0,
        "reparability": 100,
        "expectation_mismatch": 0,
        "norm_violation": 0,
        "contamination_risk": 0,
        "identity_threat": 0,
        "comparison_gap": 0,
        "vastness": 0,
        "memory_warmth": 0,
        "temporal_loss": 0,
        "salience": 40,
        "role_refs": [{
            "role": "affected_goal",
            "entity_kind": "goal",
            "entity_id": goal_id,
        }],
        "evidence_refs": [_evidence("episode")],
        "created_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
    }]
    fact = _fact(
        "source_occurred",
        "goal",
        goal_id,
        source_kind="scheduler_event",
    )
    fact["target_refs"] = [{
        "role": "affected_goal",
        "entity_kind": "goal",
        "entity_id": goal_id,
    }]

    updated = apply_direct_fact(state, fact, producer="scheduler_event")

    assert updated["active_events"][0]["evidence_refs"][-1] == fact["evidence_ref"]


@pytest.mark.parametrize(
    "axis",
    ["importance", "progress", "salience"],
)
def test_semantic_deltas_reject_reducer_owned_goal_axes(axis: str) -> None:
    """Keep reducer-owned goal axes outside the semantic delta lane."""

    state = _state_with_goal()
    goal_id = state["goals"][0]["entity_id"]
    with pytest.raises(CognitionStateError):
        apply_semantic_deltas(
            state,
            [{
                "target_path": f"goals.{goal_id}.{axis}",
                "delta": 10,
                "evidence_handles": ["action-c"],
                "reason": "forbidden reducer-owned update",
            }],
        )


def test_causal_event_identity_and_salience_are_reducer_owned() -> None:
    """Use the frozen evidence digest and accepted-axis magnitude on create."""

    state = build_acquaintance_user_state(
        global_user_id="user-c",
        updated_at="2026-07-14T00:00:00Z",
    )
    primary_evidence = _evidence("episode")
    event = {
        "description": "a reducer-owned causal event",
        "role_refs": [{
            "role": "actor",
            "entity_kind": "user",
            "entity_id": "user-c",
        }],
        "outcome_impact": 0,
        "responsibility": 0,
        "intentionality": 0,
        "harm": 40,
        "unfairness": 0,
        "exposure": 0,
        "repair_need": 0,
        "reparability": 100,
        "expectation_mismatch": 0,
        "norm_violation": 0,
        "contamination_risk": 0,
        "identity_threat": 0,
        "comparison_gap": 0,
        "vastness": 0,
        "memory_warmth": 0,
        "temporal_loss": 0,
        "salience": 99,
    }

    updated, outcome = reduce_causal_event(
        state,
        event,
        accepted_deltas={"harm": 30},
        primary_evidence=primary_evidence,
    )

    assert outcome == "create"
    stored = updated["active_events"][0]
    assert stored["entity_id"] == canonical_event_entity_id(state, primary_evidence)
    assert stored["salience"] == 30


def test_duplicate_semantic_targets_do_not_block_unique_targets() -> None:
    """Invalidate duplicate targets while applying unaffected unique paths."""

    state = _state_with_goal()
    target_id = state["goals"][0]["entity_id"]
    before = deepcopy(state)
    updated = apply_semantic_deltas(
        state,
        [
            {
                "target_path": f"goals.{target_id}.obstruction",
                "delta": 10,
                "evidence_handles": ["action-c"],
                "reason": "duplicate candidate one",
            },
            {
                "target_path": f"goals.{target_id}.obstruction",
                "delta": -10,
                "evidence_handles": ["action-c"],
                "reason": "duplicate candidate two",
            },
            {
                "target_path": f"goals.{target_id}.urgency",
                "delta": 10,
                "evidence_handles": ["action-c"],
                "reason": "independent urgency update",
            },
        ],
    )

    assert state == before
    assert updated["goals"][0]["obstruction"] == 0
    assert updated["goals"][0]["urgency"] == 50


def test_event_comparison_uses_refs_and_reports_unrelated_text() -> None:
    """Keep event comparison structural rather than description-driven."""

    current = {
        "entity_id": "event-current",
        "role_refs": [
            {
                "role": "actor",
                "entity_kind": "character",
                "entity_id": "character:global",
            },
            {"role": "target", "entity_kind": "user", "entity_id": "user-1"},
        ],
        "axis_deltas": {"harm": 10},
        "salience": 50,
    }
    stored = {
        "entity_id": "event-1",
        "status": "active",
        "role_refs": current["role_refs"],
        "harm": 30,
    }

    assert compare_event(current, stored, {"harm": 5}) == "reinforce"
    assert compare_event(
        {**current, "description": "same words, different refs"},
        {**stored, "role_refs": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "user-2",
        }]},
        {"harm": 5},
    ) == "unrelated"


def test_goal_threat_event_and_gap_fsms_require_frozen_guards() -> None:
    """Allow transitions only with threshold and typed evidence guards."""

    blocked = transition_goal(_goal(obstruction=40), transition="blocked")
    assert blocked["status"] == "blocked"
    pursuing = transition_goal(
        _goal(status="blocked", obstruction=24),
        transition="pursuing",
    )
    assert pursuing["status"] == "pursuing"
    with pytest.raises(CognitionStateError):
        transition_goal(_goal(status="satisfied"), transition="pursuing")
    with pytest.raises(CognitionStateError):
        transition_goal(_goal(), transition="abandoned")

    threat = transition_threat(
        {"status": "active", "residual_pressure": 20},
        transition="resolved",
        evidence={"outcome_kind": "resolve"},
    )
    assert threat["status"] == "resolved"
    event = transition_event(
        {"status": "active", "repair_need": 0},
        transition="resolved",
        evidence={"outcome_kind": "repair"},
    )
    assert event["status"] == "resolved"
    gap = transition_knowledge_gap(
        {"status": "open", "uncertainty": 50},
        transition="reduced",
        previous_uncertainty=80,
    )
    assert gap["status"] == "reduced"
    resolved_gap = transition_knowledge_gap(
        {"status": "reduced", "uncertainty": 0},
        transition="resolved",
        previous_uncertainty=50,
        evidence={"outcome_kind": "answer"},
    )
    assert resolved_gap["status"] == "resolved"


def test_event_repair_axis_alone_does_not_auto_resolve() -> None:
    """Keep repair_need zero insufficient without typed completion evidence."""

    state = _state_with_goal()
    event = {
        "status": "active",
        "repair_need": 0,
    }
    with pytest.raises(CognitionStateError):
        transition_event(event, transition="resolved")
    assert state["goals"][0]["status"] == "pursuing"


def test_state_update_uses_fixed_order_and_derives_cache() -> None:
    """Run elapsed, facts, deltas, guarded lifecycle, cache, and retention."""

    state = _state_with_goal()
    goal_id = state["goals"][0]["entity_id"]
    state["active_events"] = [{
        "entity_id": "event:blocked-goal",
        "description": "the blocked goal was obstructed by an unfair action",
        "status": "active",
        "outcome_impact": 0,
        "responsibility": 0,
        "intentionality": 70,
        "harm": 0,
        "unfairness": 70,
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
        "salience": 70,
        "role_refs": [{
            "role": "affected_goal",
            "entity_kind": "goal",
            "entity_id": goal_id,
        }],
        "evidence_refs": [_evidence("episode")],
        "created_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
    }]
    updated = apply_state_update(
        state,
        direct_facts=[(
            "action_result",
            _fact(
                "goal_progress_observed",
                "goal",
                goal_id,
                observed_progress=20,
            ),
        )],
        semantic_deltas=[{
            "target_path": f"goals.{goal_id}.obstruction",
            "delta": 40,
            "evidence_handles": ["action-c"],
            "reason": "the result confirms obstruction",
        }],
        elapsed_seconds=3600,
    )

    assert updated["goals"][0]["progress"] == 20
    assert updated["goals"][0]["obstruction"] == 40
    assert updated["goals"][0]["status"] == "blocked"
    assert [row["emotion_id"] for row in updated["affect_activations"]] == [
        "anger",
    ]


def test_elapsed_decay_and_sleep_recovery_are_scope_specific() -> None:
    """Apply user elapsed evolution and character sleep recovery exactly once."""

    user_state = _state_with_goal()
    user_state["affect_activations"] = [{
        "activation_id": "emotion:joy",
        "emotion_id": "joy",
        "primary_root": {
            "scope": "user",
            "kind": "goal",
            "entity_id": user_state["goals"][0]["entity_id"],
        },
        "root_refs": [{
            "scope": "user",
            "kind": "goal",
            "entity_id": user_state["goals"][0]["entity_id"],
        }],
        "phase": "active",
        "score": 60,
        "peak_score": 60,
        "trend": "stable",
        "cause_status": "active",
        "started_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
        "last_reinforced_at": "2026-07-14T00:00:00Z",
    }]
    decayed = apply_elapsed_decay(
        user_state,
        elapsed_seconds=3600,
        rate_per_hour=4,
    )
    assert decayed["goals"][0]["salience"] == 66
    assert decayed["affect_activations"][0]["score"] == 56

    character_state = build_character_production_state(
        updated_at="2026-07-14T00:00:00Z",
    )
    character_state["goals"] = [deepcopy(user_state["goals"][0])]
    character_state["goals"][0]["role_refs"] = [{
        "role": "actor",
        "entity_kind": "character",
        "entity_id": "character:global",
    }]
    character_state["affect_activations"] = [{
        **user_state["affect_activations"][0],
        "primary_root": {
            "scope": "character",
            "kind": "goal",
            "entity_id": user_state["goals"][0]["entity_id"],
        },
        "root_refs": [{
            "scope": "character",
            "kind": "goal",
            "entity_id": user_state["goals"][0]["entity_id"],
        }],
    }]
    recovered = apply_sleep_recovery(
        character_state,
        elapsed_sleep_seconds=7200,
    )

    assert recovered["goals"][0]["salience"] == 42
    assert recovered["affect_activations"][0]["score"] == 32
