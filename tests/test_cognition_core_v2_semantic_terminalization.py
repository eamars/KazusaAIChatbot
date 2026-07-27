"""Terminal semantic-appraisal contracts and graceful reduction isolation."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import facade
from kazusa_ai_chatbot.cognition_core_v2 import semantic_source_planner
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    appraise_semantic_question,
    validate_semantic_appraisal_result,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    _apply_proposition_transition,
    apply_semantic_appraisals,
)
from tests.test_cognition_core_v2_stage_model_routing import (
    _CapturingInvoker,
    _services,
)


_TIMESTAMP = "2026-07-27T00:00:00Z"
_OUTCOME_KINDS = (
    "goal_release",
    "goal_supersession",
    "goal_completed",
    "event_completed",
    "threat_resolved",
    "event_repaired",
    "knowledge_answered",
    "outcome_pending",
)


def _evidence() -> dict[str, Any]:
    """Build one current-episode evidence row for the terminal family."""

    return {
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode-terminal-1",
            "occurred_at": _TIMESTAMP,
            "semantic_summary": "The observed outcome is available.",
        },
        "semantic_text": "The observed outcome is available.",
        "visible_to": ["q:goal_threat_outcome"],
    }


def _goal(
    entity_id: str,
    *,
    status: str = "pursuing",
    obstruction: int = 0,
) -> dict[str, Any]:
    """Build one complete goal fixture."""

    evidence_ref = _evidence()["evidence_ref"]
    return {
        "entity_id": entity_id,
        "description": "Complete the observed interaction outcome.",
        "salience": 70,
        "role_refs": [],
        "evidence_refs": [evidence_ref],
        "created_at": _TIMESTAMP,
        "updated_at": _TIMESTAMP,
        "status": status,
        "goal_kind": "ordinary_response",
        "importance": 70,
        "progress": 0,
        "obstruction": obstruction,
        "expected_success": 50,
        "controllability": 50,
        "recoverability": 50,
        "urgency": 40,
    }


def _state_with_goals() -> dict[str, Any]:
    """Build one valid state with pursuing and blocked goals."""

    state = build_acquaintance_user_state(
        global_user_id="terminal-user",
        updated_at=_TIMESTAMP,
    )
    state["goals"] = [
        _goal("goal:primary"),
        _goal("goal:blocked", status="blocked", obstruction=60),
    ]
    return state


def _character_constraints() -> dict[str, Any]:
    """Build the complete read-only character projection input."""

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


def _terminal_result(
    proposition: dict[str, Any],
    *,
    question_id: str = "q:goal_threat_outcome",
) -> dict[str, Any]:
    """Build one structurally complete semantic-appraisal result."""

    return {
        "question_id": question_id,
        "selected_evidence_handles": ["e1"],
        "selected_role_handles": [],
        "propositions": [proposition],
        "deltas": [],
        "explanation": "The bounded evidence supports this semantic result.",
    }


def _proposition(
    proposition_kind: str,
    subject_handle: str,
    *,
    object_handle: str | None = None,
) -> dict[str, Any]:
    """Build one terminal proposition over prompt-local handles."""

    result = {
        "proposition_kind": proposition_kind,
        "subject_handle": subject_handle,
        "evidence_handles": ["e1"],
        "role_assignments": [],
        "semantic_value": "The named entity has reached the asserted outcome.",
    }
    if object_handle is not None:
        result["object_handle"] = object_handle
    return result


def test_outcome_vocabulary_is_explicit_and_entity_specific() -> None:
    """Expose exact terminal assertions plus an explicit pending outcome."""

    assert semantic_source_planner.question_proposition_kinds(
        "goal_threat_outcome"
    ) == _OUTCOME_KINDS
    semantics_builder = getattr(
        semantic_source_planner,
        "question_proposition_kind_semantics",
    )
    semantics = semantics_builder("goal_threat_outcome")
    assert tuple(semantics) == _OUTCOME_KINDS
    assert all(isinstance(value, str) and value for value in semantics.values())


def test_goal_completed_atomically_establishes_transition_invariants() -> None:
    """A valid goal-completion assertion sets progress before the FSM guard."""

    state = _state_with_goals()
    result = _terminal_result(_proposition("goal_completed", "g1"))

    updated = apply_semantic_appraisals(
        state,
        [result],
        [_evidence()],
        {
            "g1": {"kind": "goal", "entity_id": "goal:primary"},
            "g2": {"kind": "goal", "entity_id": "goal:blocked"},
        },
    )

    assert updated["goals"][0]["progress"] == 100
    assert updated["goals"][0]["status"] == "satisfied"


def test_goal_terminal_postcondition_survives_same_batch_delta() -> None:
    """Goal completion remains authoritative after an accepted goal delta."""

    state = _state_with_goals()
    result = _terminal_result(_proposition("goal_completed", "g1"))
    result["deltas"] = [{
        "target_path": "goals.g1.obstruction",
        "delta": 40,
        "evidence_handles": ["e1"],
        "reason": "The bounded evidence supports this obstruction observation.",
    }]

    updated = apply_semantic_appraisals(
        state,
        [result],
        [_evidence()],
        {
            "g1": {"kind": "goal", "entity_id": "goal:primary"},
            "g2": {"kind": "goal", "entity_id": "goal:blocked"},
        },
    )

    assert updated["goals"][0]["progress"] == 100
    assert updated["goals"][0]["status"] == "satisfied"
    assert updated["goals"][0]["obstruction"] == 40


def test_outcome_pending_has_no_state_authority() -> None:
    """A nonterminal observation cannot mutate or materialize state."""

    state = _state_with_goals()
    state["goals"][0]["evidence_refs"] = []
    result = _terminal_result(_proposition("outcome_pending", "g1"))

    updated = apply_semantic_appraisals(
        state,
        [result],
        [_evidence()],
        {
            "g1": {"kind": "goal", "entity_id": "goal:primary"},
            "g2": {"kind": "goal", "entity_id": "goal:blocked"},
        },
    )

    assert updated == state


@pytest.mark.parametrize(
    ("kind", "entity_kind", "entity", "expected"),
    [
        (
            "event_completed",
            "event",
            {"status": "active", "repair_need": 70, "reparability": 10},
            {"status": "resolved", "repair_need": 0},
        ),
        (
            "threat_resolved",
            "threat",
            {"status": "active", "residual_pressure": 90},
            {"status": "resolved", "residual_pressure": 0},
        ),
        (
            "event_repaired",
            "event",
            {"status": "active", "repair_need": 70, "reparability": 10},
            {"status": "resolved", "repair_need": 0, "reparability": 100},
        ),
        (
            "knowledge_answered",
            "knowledge_gap",
            {"status": "open", "uncertainty": 90},
            {"status": "resolved", "uncertainty": 0},
        ),
    ],
)
def test_terminal_assertions_atomically_satisfy_each_fsm(
    kind: str,
    entity_kind: str,
    entity: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Every accepted terminal assertion establishes its guarded axes."""

    outcome = _apply_proposition_transition(
        entity,
        entity_kind,
        kind,
        _evidence()["evidence_ref"],
    )

    assert outcome == "resolve"
    assert entity | expected == entity


@pytest.mark.parametrize(
    ("kind", "entity_kind", "field_name", "expected"),
    [
        (
            "event_completed",
            "event",
            "active_events",
            {"status": "resolved", "repair_need": 0},
        ),
        (
            "event_repaired",
            "event",
            "active_events",
            {
                "status": "resolved",
                "repair_need": 0,
                "reparability": 100,
            },
        ),
        (
            "threat_resolved",
            "threat",
            "threats",
            {"status": "resolved", "residual_pressure": 0},
        ),
        (
            "knowledge_answered",
            "knowledge_gap",
            "knowledge_gaps",
            {"status": "resolved", "uncertainty": 0},
        ),
    ],
)
def test_candidate_terminal_assertions_use_the_same_atomic_fsm(
    kind: str,
    entity_kind: str,
    field_name: str,
    expected: dict[str, Any],
) -> None:
    """A single candidate proposition must terminalize on its create pass."""

    state = build_acquaintance_user_state(
        global_user_id="candidate-terminal-user",
        updated_at=_TIMESTAMP,
    )
    result = _terminal_result(_proposition(kind, "candidate1"))
    comparisons: list[dict[str, Any]] = []

    updated = apply_semantic_appraisals(
        state,
        [result],
        [_evidence()],
        {
            "candidate1": {
                "scope": "user",
                "kind": entity_kind,
                "entity_id": f"candidate:{entity_kind}:e1",
            }
        },
        comparisons,
    )

    assert comparisons[0]["outcome"] == "resolve"
    assert len(updated[field_name]) == 1
    assert updated[field_name][0] | expected == updated[field_name][0]


@pytest.mark.parametrize(
    ("kind", "entity_kind", "field_name", "axis_deltas", "expected"),
    [
        (
            "event_completed",
            "event",
            "active_events",
            {"repair_need": 40},
            {"status": "resolved", "repair_need": 0},
        ),
        (
            "event_repaired",
            "event",
            "active_events",
            {"repair_need": 40, "reparability": -40},
            {
                "status": "resolved",
                "repair_need": 0,
                "reparability": 100,
            },
        ),
        (
            "threat_resolved",
            "threat",
            "threats",
            {"residual_pressure": 40},
            {"status": "resolved", "residual_pressure": 0},
        ),
        (
            "knowledge_answered",
            "knowledge_gap",
            "knowledge_gaps",
            {"uncertainty": 40},
            {"status": "resolved", "uncertainty": 0},
        ),
    ],
)
def test_terminal_postconditions_survive_same_batch_deltas(
    kind: str,
    entity_kind: str,
    field_name: str,
    axis_deltas: dict[str, int],
    expected: dict[str, Any],
) -> None:
    """Terminal assertions remain authoritative after accepted batch deltas."""

    state = build_acquaintance_user_state(
        global_user_id="terminal-delta-user",
        updated_at=_TIMESTAMP,
    )
    result = _terminal_result(_proposition(kind, "candidate1"))
    result["deltas"] = [
        {
            "target_path": f"{field_name}.candidate1.{axis}",
            "delta": delta,
            "evidence_handles": ["e1"],
            "reason": "The bounded evidence supports this axis observation.",
        }
        for axis, delta in axis_deltas.items()
    ]

    updated = apply_semantic_appraisals(
        state,
        [result],
        [_evidence()],
        {
            "candidate1": {
                "scope": "user",
                "kind": entity_kind,
                "entity_id": f"candidate:{entity_kind}:e1",
            }
        },
    )

    assert len(updated[field_name]) == 1
    assert updated[field_name][0] | expected == updated[field_name][0]


def test_terminal_proposition_rejects_the_wrong_subject_kind() -> None:
    """A goal handle cannot receive a threat-resolution assertion."""

    question = {
        "question_id": "q:goal_threat_outcome",
        "question_kind": "goal_threat_outcome",
        "semantic_question": "Assess exact terminal outcomes.",
        "evidence_handles": ["e1"],
        "permitted_role_handles": ["g1"],
        "permitted_delta_paths": [],
        "dependencies": [],
    }

    with pytest.raises(
        ValueError,
        match="semantic proposition kind requires subject kind",
    ):
        validate_semantic_appraisal_result(
            _terminal_result(_proposition("threat_resolved", "g1")),
            question,
            {"e1"},
            {
                "g1": {
                    "scope": "user",
                    "kind": "goal",
                    "entity_id": "goal:primary",
                }
            },
        )


def test_invalid_role_handle_reports_its_structured_domain() -> None:
    """Role-handle repair receives an actionable structured field domain."""

    question = {
        "question_id": "q:goal_threat_outcome",
        "question_kind": "goal_threat_outcome",
        "semantic_question": "Assess exact terminal outcomes.",
        "evidence_handles": ["e1"],
        "permitted_role_handles": ["ce1", "self"],
        "permitted_delta_paths": [],
        "dependencies": [],
    }
    result = _terminal_result(_proposition("outcome_pending", "ce1"))
    result["propositions"][0]["role_assignments"] = [{
        "role": "experiencer",
        "entity_handle": "当前角色",
    }]

    with pytest.raises(
        ValueError,
        match=(
            r"role_assignments\[\*\]\.entity_handle must be one of "
            r'\["ce1", "self"\]'
        ),
    ):
        validate_semantic_appraisal_result(
            result,
            question,
            {"e1"},
            {
                "ce1": {
                    "scope": "user",
                    "kind": "event",
                    "entity_id": "candidate:event:e1",
                },
                "self": {
                    "scope": "character",
                    "kind": "character",
                    "entity_id": "character:global",
                },
            },
        )


@pytest.mark.asyncio
async def test_appraisal_retries_a_reducer_incompatible_candidate() -> None:
    """Native trial reduction keeps transition repair with the producer."""

    state = _state_with_goals()
    evidence = [_evidence()]
    projection = project_state_for_prompt(
        state,
        character_constraints=_character_constraints(),
        evidence=evidence,
    )
    question = semantic_source_planner.plan_semantic_questions(
        evidence,
        state,
        projection.handle_to_ref,
    )[0]
    invalid = _terminal_result(
        _proposition(
            "goal_supersession",
            "g1",
            object_handle="g2",
        )
    )
    replacement = {
        **_terminal_result(_proposition("goal_release", "g1")),
        "selected_evidence_handles": [],
        "propositions": [],
    }
    llm = _CapturingInvoker([invalid, replacement])

    result = await appraise_semantic_question(
        question,
        evidence,
        projection,
        _services(llm),
        validation_state=state,
    )

    assert result["propositions"] == []
    assert len(llm.configs) == 2
    first_payload = json.loads(str(llm.messages[0][1].content))
    handle_domains = first_payload["question"]["handle_field_domains"]
    assert handle_domains == {
        "subject_handle": question["permitted_role_handles"],
        "object_handle": question["permitted_role_handles"],
        "entity_handle": question["permitted_role_handles"],
        "evidence_handles": question["evidence_handles"],
    }
    assert first_payload["question"]["role_handle_semantics"] == {
        "self": {
            "structured_handle": "self",
            "semantic_text_reference": "当前角色",
        },
        "current_user": {
            "structured_handle": "current_user",
            "semantic_text_reference": "当前用户",
        },
    }
    assert state == _state_with_goals()


def test_final_reduction_isolates_one_residual_invalid_appraisal() -> None:
    """One rejected result cannot discard the last valid cognition state."""

    reducer = getattr(facade, "_reduce_appraisals_with_isolation")
    state = _state_with_goals()
    invalid = _terminal_result(
        _proposition(
            "goal_supersession",
            "g1",
            object_handle="g2",
        )
    )
    accepted = {
        **_terminal_result(
            _proposition("goal_release", "g1"),
            question_id="q:accepted",
        ),
        "selected_evidence_handles": [],
        "propositions": [],
    }

    (
        updated,
        accepted_results,
        failures,
        comparisons,
    ) = reducer(
        state,
        [invalid, accepted],
        [_evidence()],
        {
            "g1": {"kind": "goal", "entity_id": "goal:primary"},
            "g2": {"kind": "goal", "entity_id": "goal:blocked"},
        },
    )

    assert updated == state
    assert accepted_results == [accepted]
    assert failures == {
        "q:goal_threat_outcome": "semantic_appraisal_reduction_rejected"
    }
    assert comparisons == []
    assert state == _state_with_goals()


def test_final_reduction_preserves_cross_appraisal_composition() -> None:
    """Candidate creation and a later delta retain native batch semantics."""

    reducer = getattr(facade, "_reduce_appraisals_with_isolation")
    state = build_acquaintance_user_state(
        global_user_id="composition-user",
        updated_at=_TIMESTAMP,
    )
    materializer = _terminal_result(
        _proposition("intentionality", "ce1"),
        question_id="q:event_materializer",
    )
    delta_result = {
        "question_id": "q:event_delta",
        "selected_evidence_handles": ["e1"],
        "selected_role_handles": [],
        "propositions": [],
        "deltas": [{
            "target_path": "active_events.ce1.outcome_impact",
            "delta": 30,
            "evidence_handles": ["e1"],
            "reason": "The event has a material observed impact.",
        }],
        "explanation": "The bounded evidence supports an event impact.",
    }
    handle_to_ref = {
        "ce1": {
            "scope": "user",
            "kind": "event",
            "entity_id": "candidate:event:e1",
        }
    }

    (
        updated,
        accepted_results,
        failures,
        comparisons,
    ) = reducer(
        state,
        [materializer, delta_result],
        [_evidence()],
        handle_to_ref,
    )

    assert accepted_results == [materializer, delta_result]
    assert failures == {}
    assert [row["outcome"] for row in comparisons] == ["create"]
    assert len(updated["active_events"]) == 1
    assert updated["active_events"][0]["outcome_impact"] == 30
    assert updated["active_events"][0]["salience"] == 30
