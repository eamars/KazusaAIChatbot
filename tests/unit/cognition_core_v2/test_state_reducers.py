"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import state_reducers
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    build_acquaintance_user_state,
    validate_cognition_state,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.state_reducers"
EXPECTED_SYMBOLS = ["apply_relationship_maintenance", "apply_state_update"]
_TIMESTAMP = "2026-08-18T00:00:00Z"


def test_state_reducers_exposes_owned_contract() -> None:
    """Keep the module's named owner contract discoverable."""

    module = import_module(MODULE_PATH)
    missing_symbols = [
        symbol
        for symbol in EXPECTED_SYMBOLS
        if not hasattr(module, symbol)
    ]

    assert not missing_symbols, (
        f"{MODULE_PATH} is missing owner symbols: {missing_symbols}"
    )


def _state() -> dict[str, object]:
    """Build one validated user state for maintenance tests."""

    return validate_cognition_state(
        build_acquaintance_user_state(
            global_user_id="reducer-maintenance-user",
            updated_at=_TIMESTAMP,
        )
    )


def _receipt(
    *,
    target_path: str = "relationship.trust",
    requested_delta: int = 4,
    applied_delta: int = 4,
) -> dict[str, object]:
    """Build one authoritative accepted relationship receipt."""

    return {
        "target_path": target_path,
        "relationship_axis": target_path.split(".", maxsplit=1)[1],
        "requested_delta": requested_delta,
        "applied_delta": applied_delta,
        "previous_value": 0,
        "next_value": applied_delta,
        "evidence_refs": [],
        "duplicate_disposition": "unique",
    }


def test_relationship_familiarity_reinforces_once_per_utc_interaction_date() -> None:
    """Advance familiarity once when a new interaction date is accepted."""

    updated = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )

    assert updated["relationship"]["familiarity"] == 11


def test_relationship_familiarity_applies_same_day_bonus_once() -> None:
    """Allow one same-day evidence bonus in addition to date reinforcement."""

    state = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="second",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[_receipt()],
    )
    replay = state_reducers.apply_relationship_maintenance(
        updated,
        source_episode_id="third",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[_receipt()],
    )

    assert updated["relationship"]["familiarity"] == 12
    assert replay["relationship"]["familiarity"] == 12


def test_relationship_familiarity_accepts_clamped_zero_delta_bonus() -> None:
    """Count an accepted boundary-clamped receipt as relationship evidence."""

    state = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="clamped",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[
            _receipt(requested_delta=4, applied_delta=0),
        ],
    )

    assert updated["relationship"]["familiarity"] == 12


def test_relationship_familiarity_crosses_utc_date_boundary() -> None:
    """Advance familiarity again after the canonical UTC date changes."""

    state = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="next-day",
        interaction_date_utc="2026-08-19",
        elapsed_seconds=0,
    )

    assert updated["relationship"]["familiarity"] == 12
    assert updated["relationship"]["relationship_maintenance"][
        "last_interaction_date_utc"
    ] == "2026-08-19"


def test_relationship_familiarity_is_idempotent_for_replayed_source() -> None:
    """Replaying the same source leaves maintenance unchanged."""

    state = _state()
    first = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="same-source",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    replay = state_reducers.apply_relationship_maintenance(
        first,
        source_episode_id="same-source",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )

    assert replay == first


def test_relationship_duplicate_source_after_intervening_episode_is_ignored() -> None:
    """Keep an active-date ledger so an old duplicate cannot reinforce."""

    state = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    state = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="second",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )
    replay = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="first",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
    )

    assert replay == state


def test_relationship_maintenance_ignores_older_source_after_newer_episode() -> None:
    """Ignore stale source dates after a newer interaction is recorded."""

    state = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="newer",
        interaction_date_utc="2026-08-19",
        elapsed_seconds=0,
    )
    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="older",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[_receipt()],
    )

    assert updated == state


def test_relationship_maintenance_rejects_source_ledger_overflow() -> None:
    """Fail closed instead of evicting active-date source identities."""

    state = _state()
    maintenance = state["relationship"]["relationship_maintenance"]
    maintenance["last_interaction_date_utc"] = "2026-08-18"
    maintenance["processed_source_ids"] = [
        f"episode:source-{index}" for index in range(256)
    ]

    with pytest.raises(CognitionStateError, match="source ledger"):
        state_reducers.apply_relationship_maintenance(
            state,
            source_episode_id="overflow",
            interaction_date_utc="2026-08-18",
            elapsed_seconds=0,
        )


def test_relationship_salience_decays_before_downstream_derivation() -> None:
    """Apply elapsed salience decay before accepted delta reinforcement."""

    state = _state()
    state["relationship"]["salience"] = 50

    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="salience",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=3600,
    )

    assert updated["relationship"]["salience"] == 46


def test_relationship_salience_uses_four_points_per_hour() -> None:
    """Keep the policy's four-point hourly salience decay exact."""

    state = _state()
    state["relationship"]["salience"] = 100

    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="two-hours",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=2 * 3600,
    )

    assert updated["relationship"]["salience"] == 92


def test_final_goal_reconciliation_removes_stale_connection_goal() -> None:
    """Remove an active connection goal after final salience crosses below 40."""

    state = _state()
    state["relationship"].update({
        "attachment": 60,
        "desired_closeness": 80,
        "perceived_closeness": 0,
        "salience": 40,
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": "episode:goal-reconciliation",
            "occurred_at": _TIMESTAMP,
            "semantic_summary": "The episode carries relationship evidence.",
        }],
    })
    with_goal = state_reducers.create_deterministic_goals(
        state,
        evidence=[],
        updated_at=_TIMESTAMP,
    )
    assert any(
        goal["goal_kind"] == "relationship_connection"
        for goal in with_goal["goals"]
    )

    with_goal["relationship"]["salience"] = 39
    reconciled = state_reducers.create_deterministic_goals(
        with_goal,
        evidence=[],
        updated_at=_TIMESTAMP,
        reconcile_salience_gated_goals=True,
    )

    assert not any(
        goal["goal_kind"] == "relationship_connection"
        and goal["status"] in {"pursuing", "blocked"}
        for goal in reconciled["goals"]
    )


def test_relationship_salience_reinforces_from_strongest_unique_delta() -> None:
    """Use only the strongest unique accepted relationship delta."""

    state = _state()
    state["relationship"]["salience"] = 10

    updated = state_reducers.apply_relationship_maintenance(
        state,
        source_episode_id="strongest",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[
            _receipt(target_path="relationship.trust", applied_delta=3),
            _receipt(
                target_path="relationship.attachment",
                requested_delta=-8,
                applied_delta=-8,
            ),
        ],
    )

    assert updated["relationship"]["salience"] == 18


def test_cumulative_trial_reductions_do_not_multiply_relationship_maintenance() -> None:
    """A final accepted receipt set applies maintenance exactly once."""

    updated = state_reducers.apply_relationship_maintenance(
        _state(),
        source_episode_id="cumulative",
        interaction_date_utc="2026-08-18",
        elapsed_seconds=0,
        accepted_relationship_deltas=[
            _receipt(target_path="relationship.trust", applied_delta=4),
            _receipt(target_path="relationship.attachment", applied_delta=9),
        ],
    )

    assert updated["relationship"]["familiarity"] == 12
    assert updated["relationship"]["salience"] == 9


def _causal_event(
    *,
    entity_id: str,
    source_id: str,
    status: str = "active",
) -> dict[str, Any]:
    """Build one complete event row for source-identity reducer tests."""

    return {
        "entity_id": entity_id,
        "description": "A source-bound event.",
        "salience": 50,
        "role_refs": [],
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": source_id,
            "occurred_at": _TIMESTAMP,
            "semantic_summary": "A source-bound event.",
        }],
        "created_at": _TIMESTAMP,
        "updated_at": _TIMESTAMP,
        "status": status,
        "outcome_impact": 20,
        "responsibility": 20,
        "intentionality": 20,
        "harm": 20,
        "unfairness": 20,
        "exposure": 20,
        "repair_need": 20,
        "reparability": 20,
        "expectation_mismatch": 20,
        "norm_violation": 20,
        "contamination_risk": 20,
        "identity_threat": 20,
        "comparison_gap": 20,
        "vastness": 20,
        "memory_warmth": 20,
        "temporal_loss": 20,
    }


def _causal_threat(source_id: str) -> dict[str, Any]:
    """Build one active threat with a custom persistent identity."""

    return {
        "entity_id": "threat:custom-source",
        "description": "A source-bound threat.",
        "salience": 50,
        "role_refs": [],
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": source_id,
            "occurred_at": _TIMESTAMP,
            "semantic_summary": "A source-bound threat.",
        }],
        "created_at": _TIMESTAMP,
        "updated_at": _TIMESTAMP,
        "status": "active",
        "likelihood": 50,
        "expected_harm": 50,
        "uncertainty": 50,
        "controllability": 50,
        "coping_potential": 20,
        "residual_pressure": 50,
    }


def _causal_gap(source_id: str) -> dict[str, Any]:
    """Build one open knowledge gap with a custom persistent identity."""

    return {
        "entity_id": "knowledge_gap:custom-source",
        "description": "A source-bound knowledge gap.",
        "salience": 50,
        "role_refs": [],
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": source_id,
            "occurred_at": _TIMESTAMP,
            "semantic_summary": "A source-bound knowledge gap.",
        }],
        "created_at": _TIMESTAMP,
        "updated_at": _TIMESTAMP,
        "status": "open",
        "relevance": 50,
        "uncertainty": 50,
        "learnability": 50,
        "novelty": 50,
        "model_accommodation": 50,
    }


def _candidate_event_result() -> dict[str, Any]:
    """Build an accepted candidate event product with one source handle."""

    return {
        "question_id": "q:event_agency",
        "selected_evidence_handles": ["e1"],
        "selected_role_handles": [],
        "propositions": [{
            "proposition_kind": "responsibility",
            "subject_handle": "ce1",
            "evidence_handles": ["e1"],
            "role_assignments": [],
            "semantic_value": "A source-bound event carries responsibility.",
        }],
        "deltas": [{
            "target_path": "active_events.ce1.responsibility",
            "delta": 4,
            "evidence_handles": ["e1"],
            "reason": "The source-bound event is reinforced.",
        }, {
            "target_path": "active_events.ce1.outcome_impact",
            "delta": 30,
            "evidence_handles": ["e1"],
            "reason": "The source-bound event has a grounded outcome.",
        }],
        "explanation": "The source-bound event is grounded.",
    }


def test_same_source_candidate_reinforces_custom_event_without_roles() -> None:
    """Match a candidate to a custom-ID event by exact provenance identity."""

    state = _state()
    stored_event = _causal_event(
        entity_id="event:custom-source",
        source_id="episode-source-e1",
    )
    state["active_events"] = [stored_event]
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": stored_event["evidence_refs"][0],
    }]
    updated = state_reducers.apply_semantic_appraisals(
        state,
        [_candidate_event_result()],
        evidence,
        {
            "ce1": {
                "scope": "user",
                "kind": "event",
                "entity_id": "candidate:event:e1",
            },
        },
    )["updated_state"]

    assert len(updated["active_events"]) == 1
    assert updated["active_events"][0]["entity_id"] == "event:custom-source"
    assert updated["active_events"][0]["responsibility"] > 20


def test_same_source_threat_candidate_terminalizes_custom_threat() -> None:
    """Resolve a custom threat through its exact source-bound candidate."""

    state = _state()
    stored_threat = _causal_threat("episode-threat-e1")
    state["threats"] = [stored_threat]
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": stored_threat["evidence_refs"][0],
    }]
    result = {
        "question_id": "q:goal_threat_outcome",
        "selected_evidence_handles": ["e1"],
        "selected_role_handles": [],
        "propositions": [{
            "proposition_kind": "threat_resolved",
            "subject_handle": "ct1",
            "evidence_handles": ["e1"],
            "role_assignments": [],
            "semantic_value": "The source-bound threat is resolved.",
        }],
        "deltas": [],
        "explanation": "The source-bound threat is resolved.",
    }

    updated = state_reducers.apply_semantic_appraisals(
        state,
        [result],
        evidence,
        {
            "ct1": {
                "scope": "user",
                "kind": "threat",
                "entity_id": "candidate:threat:e1",
            },
        },
    )["updated_state"]

    assert len(updated["threats"]) == 1
    assert updated["threats"][0]["entity_id"] == "threat:custom-source"
    assert updated["threats"][0]["status"] == "resolved"


def test_same_source_gap_candidate_answers_custom_gap() -> None:
    """Answer a custom knowledge gap through its exact source identity."""

    state = _state()
    stored_gap = _causal_gap("episode-gap-e1")
    state["knowledge_gaps"] = [stored_gap]
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": stored_gap["evidence_refs"][0],
    }]
    result = {
        "question_id": "q:goal_threat_outcome",
        "selected_evidence_handles": ["e1"],
        "selected_role_handles": [],
        "propositions": [{
            "proposition_kind": "knowledge_answered",
            "subject_handle": "ck1",
            "evidence_handles": ["e1"],
            "role_assignments": [],
            "semantic_value": "The source-bound gap is answered.",
        }],
        "deltas": [],
        "explanation": "The source-bound gap is answered.",
    }

    updated = state_reducers.apply_semantic_appraisals(
        state,
        [result],
        evidence,
        {
            "ck1": {
                "scope": "user",
                "kind": "knowledge_gap",
                "entity_id": "candidate:knowledge_gap:e1",
            },
        },
    )["updated_state"]

    assert len(updated["knowledge_gaps"]) == 1
    assert (
        updated["knowledge_gaps"][0]["entity_id"]
        == "knowledge_gap:custom-source"
    )
    assert updated["knowledge_gaps"][0]["status"] == "resolved"


def test_distinct_source_candidate_creates_separate_event() -> None:
    """Keep distinct source identities on separate causal event rows."""

    state = _state()
    state["active_events"] = [
        _causal_event(
            entity_id="event:stored-abuse",
            source_id="episode-abuse",
        )
    ]
    transit_ref = {
        "source_kind": "episode",
        "source_id": "episode-transit",
        "occurred_at": _TIMESTAMP,
        "semantic_summary": "A transit event.",
    }
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": transit_ref,
    }]
    updated = state_reducers.apply_semantic_appraisals(
        state,
        [_candidate_event_result()],
        evidence,
        {
            "ce1": {
                "scope": "user",
                "kind": "event",
                "entity_id": "candidate:event:e1",
            },
        },
    )["updated_state"]

    assert len(updated["active_events"]) == 2
    assert {
        tuple(row["source_id"] for row in event["evidence_refs"])
        for event in updated["active_events"]
    } == {("episode-abuse",), ("episode-transit",)}


def test_ambiguous_same_source_candidates_fail_closed() -> None:
    """Reject two eligible native rows sharing one exact source identity."""

    state = _state()
    state["active_events"] = [
        _causal_event(
            entity_id="event:source-one",
            source_id="episode-duplicate",
        ),
        _causal_event(
            entity_id="event:source-two",
            source_id="episode-duplicate",
        ),
    ]
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": state["active_events"][0]["evidence_refs"][0],
    }]

    with pytest.raises(CognitionStateError, match="ambiguous same-source"):
        state_reducers.apply_semantic_appraisals(
            state,
            [_candidate_event_result()],
            evidence,
            {
                "ce1": {
                    "scope": "user",
                    "kind": "event",
                    "entity_id": "candidate:event:e1",
                },
            },
        )


def test_distinct_source_events_keep_distinct_same_kind_goals() -> None:
    """Create one source-bound goal per genuinely distinct causal event."""

    state = _state()
    first_event = _causal_event(
        entity_id="event:first-source",
        source_id="episode:first-source",
    )
    second_event = _causal_event(
        entity_id="event:second-source",
        source_id="episode:second-source",
    )
    first_event["identity_threat"] = 60
    second_event["identity_threat"] = 60
    state["active_events"] = [first_event, second_event]

    updated = state_reducers.create_deterministic_goals(
        state,
        evidence=[],
        updated_at=_TIMESTAMP,
    )
    boundary_goals = [
        goal
        for goal in updated["goals"]
        if goal["goal_kind"] == "autonomy_boundary"
    ]

    assert len(boundary_goals) == 2
    assert {
        goal["entity_id"]
        for goal in boundary_goals
    } == {
        "goal:autonomy_boundary:user:event:first-source",
        "goal:autonomy_boundary:user:event:second-source",
    }
