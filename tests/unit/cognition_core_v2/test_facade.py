"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/facade.py."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import facade
from kazusa_ai_chatbot.cognition_core_v2.contracts import BranchDefinition
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
    validate_cognition_state,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.facade"
EXPECTED_SYMBOLS = ["run_cognition"]
_TIMESTAMP = "2026-07-27T00:00:00Z"


def test_facade_exposes_owned_contract() -> None:
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


@pytest.mark.asyncio
async def test_branch_handler_carries_input_episode_into_recurrence_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Carry the validated input episode into a recurrence goal branch."""

    captured_context: dict[str, Any] = {}

    async def capture_goal_context(
        definition: Any,
        goal_ref: Any,
        semantic_context: dict[str, Any],
        evidence: Any,
        services: Any,
        current_turn_relational_willingness: Any = None,
    ) -> dict[str, Any]:
        """Capture the branch context handed to the goal owner."""

        del (
            definition,
            goal_ref,
            evidence,
            services,
            current_turn_relational_willingness,
        )
        captured_context.update(semantic_context)
        return {}

    monkeypatch.setattr(facade, "run_goal_cognition", capture_goal_context)
    state = validate_cognition_state(
        build_acquaintance_user_state(
            global_user_id="facade-recurrence-user",
            updated_at=_TIMESTAMP,
        )
    )
    episode_id = "facade-recurrence-episode"
    relational_carrier = {
        "schema_version": "current_turn_relational_willingness.v2",
        "episode_id": episode_id,
        "branch_id": "ordinary_response",
        "decision": {
            "schema_version": "relational_willingness.v2",
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "current_user_relationship_state": "not_applicable",
            "reason": "the current episode is not relationship sensitive",
            "evidence_handles": ["e1"],
        },
    }
    handler = facade._branch_handler(
        {},
        state,
        {
            "episode": {"episode_id": episode_id},
            "evidence": [_evidence()],
            "resolver_cycle_index": 1,
            "current_turn_relational_willingness": relational_carrier,
        },
        None,
    )

    await handler(
        BranchDefinition(
            branch_id="ordinary_response",
            dependencies=(),
            action_tendencies=("respond",),
        )
    )

    assert captured_context["_episode_id"] == episode_id


def _character_constraints() -> dict[str, Any]:
    """Build the complete character constraint projection for finalization."""

    state = build_character_production_state(updated_at=_TIMESTAMP)
    return {
        "drives": state["drives"],
        "standards": state["standards"],
        "meaning_state": state["meaning_state"],
        "personality_judgment": {
            "logic": "evidence-led",
            "defense": "reserved under pressure",
            "quirks": "brief hesitation",
            "taboos": "preserve character agency",
        },
    }


def _evidence(handle: str = "e1") -> dict[str, Any]:
    """Build one current-episode evidence row for reducer tests."""

    return {
        "evidence_handle": handle,
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": f"episode-facade-{handle}",
            "occurred_at": _TIMESTAMP,
            "semantic_summary": "A bounded reducer observation is available.",
        },
        "semantic_text": "A bounded reducer observation is available.",
        "visible_to": ["q:event_agency", "q:goal_threat_outcome"],
        "authority": "current_event",
    }


def _event_row(
    index: int,
    *,
    status: str = "active",
    outcome_impact: int = 0,
    salience: int = 0,
) -> dict[str, Any]:
    """Build one complete event row without affect protection."""

    evidence_ref = _evidence(f"existing-{index}")["evidence_ref"]
    return {
        "entity_id": f"event:fixture-{index}",
        "description": f"Fixture event {index}.",
        "salience": salience,
        "role_refs": [],
        "evidence_refs": [evidence_ref],
        "created_at": _TIMESTAMP,
        "updated_at": _TIMESTAMP,
        "status": status,
        "outcome_impact": outcome_impact,
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
    }


def _goal_row(index: int) -> dict[str, Any]:
    """Build one pursuing goal that does not itself derive another goal."""

    evidence_ref = _evidence(f"goal-{index}")["evidence_ref"]
    return {
        "entity_id": f"goal:fixture-{index}",
        "description": f"Fixture goal {index}.",
        "salience": 70,
        "role_refs": [],
        "evidence_refs": [evidence_ref],
        "created_at": _TIMESTAMP,
        "updated_at": _TIMESTAMP,
        "status": "pursuing",
        "goal_kind": "ordinary_response",
        "importance": 70,
        "progress": 0,
        "obstruction": 0,
        "expected_success": 50,
        "controllability": 50,
        "recoverability": 50,
        "urgency": 40,
    }


def _state_with_events(
    count: int,
    *,
    terminal_index: int | None = None,
    protected_terminal: bool = False,
) -> dict[str, Any]:
    """Build a valid user state with a controlled event-cap boundary."""

    state = build_acquaintance_user_state(
        global_user_id="facade-reducer-user",
        updated_at=_TIMESTAMP,
    )
    rows = []
    for index in range(count):
        is_terminal = index == terminal_index
        rows.append(
            _event_row(
                index,
                status="resolved" if is_terminal else "active",
                outcome_impact=80 if is_terminal and protected_terminal else 0,
                salience=80 if is_terminal and protected_terminal else 0,
            )
        )
    state["active_events"] = rows
    return validate_cognition_state(state)


def _state_with_goal_capacity() -> dict[str, Any]:
    """Build a valid state whose pursuing goals fill the native cap."""

    state = build_acquaintance_user_state(
        global_user_id="facade-goal-cap-user",
        updated_at=_TIMESTAMP,
    )
    state["goals"] = [_goal_row(index) for index in range(16)]
    return validate_cognition_state(state)


def _candidate_event_result(
    question_id: str,
    *,
    evidence_handle: str = "e1",
    event_handle: str = "ce1",
    outcome_impact: int = 30,
) -> dict[str, Any]:
    """Build one evidence-backed event candidate with a retained delta."""

    return {
        "question_id": question_id,
        "selected_evidence_handles": [evidence_handle],
        "selected_role_handles": [],
        "propositions": [{
            "proposition_kind": "intentionality",
            "subject_handle": event_handle,
            "evidence_handles": [evidence_handle],
            "role_assignments": [],
            "semantic_value": "A candidate event is grounded in the episode.",
        }],
        "deltas": [{
            "target_path": (
                f"active_events.{event_handle}.outcome_impact"
            ),
            "delta": outcome_impact,
            "evidence_handles": [evidence_handle],
            "reason": "The episode supports a bounded event impact.",
        }],
        "explanation": "The bounded evidence supports this candidate.",
    }


def _handle_map(*handles: str) -> dict[str, dict[str, str]]:
    """Build candidate event prompt bindings for direct reducer tests."""

    return {
        handle: {
            "scope": "user",
            "kind": "event",
            "entity_id": f"candidate:event:{handle[1:]}",
        }
        for handle in handles
    }


def _reduce(
    state: dict[str, Any],
    results: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    handle_to_ref: dict[str, dict[str, str]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
]:
    """Call the private facade owner with the canonical finalization inputs."""

    return facade._reduce_appraisals_with_isolation(
        state,
        results,
        evidence,
        handle_to_ref,
        updated_at=_TIMESTAMP,
        character_constraints=_character_constraints(),
        relationship_context=None,
    )


def test_reduction_rejects_over_cap_candidate_during_full_finalization() -> None:
    """Reject a 33rd active event without deleting existing active causes."""

    state = _state_with_events(32)
    result = _candidate_event_result("q:over-cap")

    updated, accepted, failures, comparisons = _reduce(
        state,
        [result],
        [_evidence()],
        _handle_map("ce1"),
    )

    assert len(updated["active_events"]) == 32
    assert updated == state
    assert accepted == []
    assert failures == {
        "q:over-cap": "semantic_appraisal_reduction_rejected",
    }
    assert comparisons == []


def test_reduction_rejects_affect_protected_terminal_during_finalization() -> None:
    """Reject admission when final affect derivation protects a terminal row."""

    state = _state_with_events(
        32,
        terminal_index=0,
        protected_terminal=True,
    )
    result = _candidate_event_result("q:protected-terminal")

    updated, accepted, failures, comparisons = _reduce(
        state,
        [result],
        [_evidence()],
        _handle_map("ce1"),
    )

    assert len(updated["active_events"]) == 32
    assert updated["active_events"][0]["status"] == "resolved"
    assert any(
        root["entity_id"] == "event:fixture-0"
        for activation in updated["affect_activations"]
        for root in activation["root_refs"]
    )
    assert accepted == []
    assert failures == {
        "q:protected-terminal": "semantic_appraisal_reduction_rejected",
    }
    assert comparisons == []


def test_reduction_accepts_candidate_when_terminal_row_is_removable() -> None:
    """Admit a candidate when canonical pruning can remove one terminal row."""

    state = _state_with_events(32, terminal_index=0)
    result = _candidate_event_result("q:removable-terminal")

    updated, accepted, failures, comparisons = _reduce(
        state,
        [result],
        [_evidence()],
        _handle_map("ce1"),
    )

    assert len(updated["active_events"]) == 32
    assert not any(
        event["entity_id"] == "event:fixture-0"
        for event in updated["active_events"]
    )
    assert any(
        event["description"] == "A candidate event is grounded in the episode."
        for event in updated["active_events"]
    )
    assert accepted == [result]
    assert failures == {}
    assert [row["outcome"] for row in comparisons] == ["create"]


def test_reduction_rejects_goal_capacity_during_finalization() -> None:
    """Reject a candidate whose derived goal would exceed the goal cap."""

    state = _state_with_goal_capacity()
    result = _candidate_event_result(
        "q:goal-cap",
        outcome_impact=-80,
    )

    updated, accepted, failures, comparisons = _reduce(
        state,
        [result],
        [_evidence()],
        _handle_map("ce1"),
    )

    assert len(updated["goals"]) == 16
    assert updated["active_events"] == []
    assert accepted == []
    assert failures == {
        "q:goal-cap": "semantic_appraisal_reduction_rejected",
    }
    assert comparisons == []


def test_reduction_preserves_accepted_prefix_and_comparison_rows() -> None:
    """Keep the admitted prefix and discard only the rejected comparison row."""

    state = _state_with_events(32, terminal_index=0)
    first = _candidate_event_result(
        "q:accepted-prefix",
        evidence_handle="e1",
        event_handle="ce1",
        outcome_impact=30,
    )
    second = _candidate_event_result(
        "q:rejected-suffix",
        evidence_handle="e2",
        event_handle="ce2",
        outcome_impact=40,
    )

    updated, accepted, failures, comparisons = _reduce(
        state,
        [first, second],
        [_evidence("e1"), _evidence("e2")],
        _handle_map("ce1", "ce2"),
    )

    assert accepted == [first]
    assert failures == {
        "q:rejected-suffix": "semantic_appraisal_reduction_rejected",
    }
    assert [row["outcome"] for row in comparisons] == ["create"]
    assert all(
        evidence_ref["source_id"] == "episode-facade-e1"
        for row in comparisons
        for evidence_ref in row["evidence_refs"]
    )
    assert len(updated["active_events"]) == 32
    assert any(
        event["description"] == "A candidate event is grounded in the episode."
        for event in updated["active_events"]
    )


def test_reduction_records_bounded_rejection_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record the finalization step and bounded error text for rejection."""

    state = _state_with_events(32)
    result = _candidate_event_result("q:bounded-evidence")
    failure_records: list[dict[str, Any]] = []
    validation_events: list[dict[str, Any]] = []
    original_apply_state_update = facade.apply_state_update
    call_count = 0

    def fail_on_candidate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise facade.CognitionStateError("x" * 1000)
        return original_apply_state_update(*args, **kwargs)

    def record_failure(**kwargs: Any) -> None:
        failure_records.append(kwargs)

    def record_validation_event(
        event_id: str,
        payload: dict[str, Any],
    ) -> None:
        validation_events.append({"event_id": event_id, **payload})

    monkeypatch.setattr(facade, "apply_state_update", fail_on_candidate)
    monkeypatch.setattr(
        facade.failure_capsule,
        "mark_current_failure",
        record_failure,
    )
    monkeypatch.setattr(
        facade,
        "capture_validation_event",
        record_validation_event,
    )

    _, accepted, failures, comparisons = _reduce(
        state,
        [result],
        [_evidence()],
        _handle_map("ce1"),
    )

    assert accepted == []
    assert failures == {
        "q:bounded-evidence": "semantic_appraisal_reduction_rejected",
    }
    assert comparisons == []
    assert len(failure_records) == 1
    details = failure_records[0]["details"]
    assert details["question_id"] == "q:bounded-evidence"
    assert details["failure_code"] == (
        "semantic_appraisal_reduction_rejected"
    )
    assert details["finalization_step"] == "apply_state_update"
    assert len(details["exception_text"]) == 500
    assert len(validation_events) == 1
    assert validation_events[0]["error"] == details["exception_text"]


def test_reduction_runs_finalization_once_per_admitted_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run every finalization stage exactly once for each admitted prefix."""

    state = build_acquaintance_user_state(
        global_user_id="facade-once-user",
        updated_at=_TIMESTAMP,
    )
    result = {
        "question_id": "q:once",
        "selected_evidence_handles": [],
        "selected_role_handles": [],
        "propositions": [],
        "deltas": [],
        "explanation": "No-op semantic appraisal for orchestration testing.",
    }
    function_names = (
        "apply_semantic_appraisals",
        "_semantic_relief_transitions",
        "apply_state_update",
        "create_deterministic_goals",
        "validate_cognition_state",
    )
    counts = {name: 0 for name in function_names}
    for name in function_names:
        original = getattr(facade, name)

        def counted(
            *args: Any,
            _name: str = name,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(facade, name, counted)

    updated, accepted, failures, comparisons = _reduce(
        state,
        [result],
        [],
        {},
    )

    assert updated == state
    assert accepted == [result]
    assert failures == {}
    assert comparisons == []
    assert counts == {name: 2 for name in function_names}
