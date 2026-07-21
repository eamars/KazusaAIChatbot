"""Deterministic native-state tests for cognition core V2."""

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    build_acquaintance_user_state,
    build_character_production_state,
    prune_terminal_entities,
    resolve_state_scope,
    validate_cognition_state,
)
def test_acquaintance_user_state_has_the_contract_defaults() -> None:
    """Build a user-scoped state with the explicit acquaintance baseline."""

    state = build_acquaintance_user_state(
        global_user_id="user-b",
        updated_at="2026-07-14T00:00:00Z",
    )

    assert state["schema_version"] == "cognition_state.v2"
    assert state["state_scope"] == "user"
    assert state["owner_user_id"] == "user-b"
    assert state["relationship"]["familiarity"] == 10
    assert state["relationship"]["desired_closeness"] == 10
    assert state["goals"] == []
    assert state["affect_activations"] == []
    validate_cognition_state(state)


def test_character_state_has_the_contract_defaults() -> None:
    """Build the singleton character state with all required defaults."""

    state = build_character_production_state(
        updated_at="2026-07-14T00:00:00Z",
    )

    assert state["schema_version"] == "cognition_state.v2"
    assert state["state_scope"] == "character"
    assert set(state["drives"]) == {
        "autonomy",
        "connection",
        "safety",
        "competence",
        "care",
        "integrity",
        "exploration",
        "meaning",
    }
    assert [standard["standard_id"] for standard in state["standards"]] == [
        "honesty",
        "avoid_harm",
        "respect_boundaries",
        "follow_through",
        "self_respect",
    ]
    validate_cognition_state(state)


@pytest.mark.parametrize(
    ("caller", "target_user_id", "expected_scope"),
    [
        ("persona_user_message", "user-a", ("user", "user-a")),
        ("tool_result", "user-a", ("user", "user-a")),
        ("background_result", "user-a", ("user", "user-a")),
        ("group_sender", "user-a", ("user", "user-a")),
        ("self_cognition", "user-a", ("user", "user-a")),
        ("self_cognition", None, ("character", "global")),
        ("reflection", None, ("character", "global")),
        ("scheduled_tick", "user-a", ("user", "user-a")),
        ("scheduled_tick", None, ("character", "global")),
    ],
)
def test_state_scope_matrix(
    caller: str,
    target_user_id: str | None,
    expected_scope: tuple[str, str],
) -> None:
    """Resolve every direct caller class to its persisted state owner."""

    assert resolve_state_scope(caller, target_user_id) == expected_scope


def test_recurrence_inherits_the_origin_scope() -> None:
    """Keep resolver recurrence attached to its originating owner."""

    assert resolve_state_scope(
        "resolver_recurrence",
        origin_scope=("user", "user-a"),
    ) == ("user", "user-a")


def test_state_scope_matrix_rejects_missing_required_owner() -> None:
    """Require an explicit user owner for user-bound callers."""

    for caller in (
        "persona_user_message",
        "tool_result",
        "background_result",
        "group_sender",
    ):
        with pytest.raises(CognitionStateError):
            resolve_state_scope(caller)


def test_state_validation_rejects_wrong_scope_and_uncapped_lists() -> None:
    """Reject malformed state before it can cross the persistence boundary."""

    state = build_acquaintance_user_state(
        global_user_id="user-c",
        updated_at="2026-07-14T00:00:00Z",
    )
    state["state_scope"] = "character"

    with pytest.raises(CognitionStateError):
        validate_cognition_state(state)

    state = build_acquaintance_user_state(
        global_user_id="user-c",
        updated_at="2026-07-14T00:00:00Z",
    )
    state["goals"] = [
        {
            "entity_id": f"goal-{index}",
            "description": "bounded test goal",
            "status": "pursuing",
            "goal_kind": "ordinary_response",
            "importance": 50,
            "progress": 0,
            "obstruction": 0,
            "expected_success": 50,
            "controllability": 50,
            "recoverability": 50,
            "urgency": 50,
            "salience": 50,
            "role_refs": [],
            "evidence_refs": [],
            "created_at": "2026-07-14T00:00:00Z",
            "updated_at": "2026-07-14T00:00:00Z",
        }
        for index in range(17)
    ]

    with pytest.raises(CognitionStateError):
        validate_cognition_state(state)


def test_pruning_removes_old_terminal_entities_but_protects_active_roots() -> None:
    """Prune only unreferenced terminal entities when a list exceeds its cap."""

    state = build_acquaintance_user_state(
        global_user_id="user-d",
        updated_at="2026-07-14T00:00:00Z",
    )
    state["goals"] = [
        {
            "entity_id": f"goal-{index}",
            "description": "terminal test goal",
            "status": "satisfied",
            "goal_kind": "ordinary_response",
            "importance": 50,
            "progress": 100,
            "obstruction": 0,
            "expected_success": 50,
            "controllability": 50,
            "recoverability": 50,
            "urgency": 0,
            "salience": 50,
            "role_refs": [],
            "evidence_refs": [],
            "created_at": f"2026-07-14T00:00:{index:02d}Z",
            "updated_at": f"2026-07-14T00:00:{index:02d}Z",
        }
        for index in range(17)
    ]
    state["affect_activations"] = [
        {
            "activation_id": "emotion:joy",
            "emotion_id": "joy",
            "primary_root": {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal-0",
            },
            "root_refs": [{
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal-0",
            }],
            "phase": "active",
            "score": 30,
            "peak_score": 30,
            "trend": "stable",
            "cause_status": "active",
            "started_at": "2026-07-14T00:00:00Z",
            "updated_at": "2026-07-14T00:00:00Z",
            "last_reinforced_at": "2026-07-14T00:00:00Z",
        }
    ]

    pruned = prune_terminal_entities(state)

    assert len(pruned["goals"]) == 16
    assert {goal["entity_id"] for goal in pruned["goals"]} == {
        "goal-0",
        *[f"goal-{index}" for index in range(2, 17)],
    }


def test_entity_records_require_structured_refs_and_all_typed_axes() -> None:
    """Reject string provenance, incomplete events, and plural root kinds."""

    state = build_acquaintance_user_state(
        global_user_id="user-e",
        updated_at="2026-07-14T00:00:00Z",
    )
    event = {
        "entity_id": "event:episode-1",
        "description": "a typed event",
        "salience": 60,
        "role_refs": [{
            "role": "actor",
            "entity_kind": "user",
            "entity_id": "user-e",
        }],
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": "episode-1",
            "occurred_at": "2026-07-14T00:00:00Z",
            "semantic_summary": "the event was observed",
        }],
        "created_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
        "status": "active",
        "outcome_impact": 20,
        "responsibility": 40,
        "intentionality": 40,
        "harm": 10,
        "unfairness": 10,
        "exposure": 10,
        "repair_need": 20,
        "reparability": 80,
        "expectation_mismatch": 10,
        "norm_violation": 0,
        "contamination_risk": 0,
        "identity_threat": 0,
        "comparison_gap": 0,
        "vastness": 0,
        "memory_warmth": 0,
        "temporal_loss": 0,
    }
    state["active_events"] = [event]
    validate_cognition_state(state)

    state["active_events"][0]["role_refs"] = ["actor:user-e"]
    with pytest.raises(CognitionStateError):
        validate_cognition_state(state)

    state["active_events"][0]["role_refs"] = [{
        "role": "actor",
        "entity_kind": "user",
        "entity_id": "user-e",
    }]
    del state["active_events"][0]["vastness"]
    with pytest.raises(CognitionStateError):
        validate_cognition_state(state)

    state["active_events"][0]["vastness"] = 0
    state["affect_activations"] = [{
        "activation_id": "emotion:joy",
        "emotion_id": "joy",
        "primary_root": {
            "scope": "user",
            "kind": "events",
            "entity_id": "event:episode-1",
        },
        "root_refs": [{
            "scope": "user",
            "kind": "events",
            "entity_id": "event:episode-1",
        }],
        "phase": "active",
        "score": 50,
        "peak_score": 50,
        "trend": "rising",
        "cause_status": "active",
        "started_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
        "last_reinforced_at": "2026-07-14T00:00:00Z",
    }]
    with pytest.raises(CognitionStateError):
        validate_cognition_state(state)


def test_duplicate_activation_rows_are_rejected() -> None:
    """Keep activation identity one-row-per-emotion and deterministic."""

    state = build_acquaintance_user_state(
        global_user_id="user-f",
        updated_at="2026-07-14T00:00:00Z",
    )
    event = {
        "entity_id": "event:episode-2",
        "description": "an event",
        "salience": 50,
        "role_refs": [],
        "evidence_refs": [],
        "created_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
        "status": "active",
        "outcome_impact": 10,
        "responsibility": 0,
        "intentionality": 0,
        "harm": 0,
        "unfairness": 0,
        "exposure": 0,
        "repair_need": 10,
        "reparability": 90,
        "expectation_mismatch": 0,
        "norm_violation": 0,
        "contamination_risk": 0,
        "identity_threat": 0,
        "comparison_gap": 0,
        "vastness": 0,
        "memory_warmth": 0,
        "temporal_loss": 0,
    }
    state["active_events"] = [event]
    root = {
        "scope": "user",
        "kind": "event",
        "entity_id": event["entity_id"],
    }
    activation = {
        "activation_id": "emotion:joy",
        "emotion_id": "joy",
        "primary_root": root,
        "root_refs": [root],
        "phase": "active",
        "score": 30,
        "peak_score": 30,
        "trend": "stable",
        "cause_status": "active",
        "started_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
        "last_reinforced_at": "2026-07-14T00:00:00Z",
    }
    state["affect_activations"] = [activation, dict(activation)]

    with pytest.raises(CognitionStateError):
        validate_cognition_state(state)
