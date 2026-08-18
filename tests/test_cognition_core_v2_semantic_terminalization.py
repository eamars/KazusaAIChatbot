"""Terminal semantic-appraisal contracts and graceful reduction isolation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import facade
from kazusa_ai_chatbot.cognition_core_v2 import semantic_source_planner
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    appraise_semantic_question,
    validate_semantic_appraisal_result,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    _apply_proposition_transition,
    _causal_candidate_id,
    _matching_event,
    _retain_current_batch_evidence,
    apply_semantic_appraisals as _apply_semantic_appraisals,
)
from tests.test_cognition_core_v2_stage_model_routing import (
    _CapturingInvoker,
    _services,
)
from tests.cognition_core_v2_test_helpers import canonical_identity_context


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
_FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "cognition_v2_group_ownership_terminalization.json"
)


def _reduced_state(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Extract the native state from the reducer's receipt envelope."""

    return _apply_semantic_appraisals(*args, **kwargs)["updated_state"]


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
        "authority": "current_event",
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

    updated = _reduced_state(
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

    updated = _reduced_state(
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

    updated = _reduced_state(
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

    updated = _reduced_state(
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


def test_role_matching_does_not_reuse_a_resolved_historical_event() -> None:
    """A distinct candidate can share roles with a resolved prior event."""

    role_refs = [{
        "role": "experiencer",
        "entity_kind": "relationship",
        "entity_id": "relationship:user:fixture",
    }]
    resolved = {
        "entity_id": "event:resolved",
        "status": "resolved",
        "role_refs": role_refs,
    }
    incoming = {
        "entity_id": "event:new",
        "status": "active",
        "role_refs": role_refs,
    }

    assert _matching_event({"active_events": [resolved]}, incoming) is None
    assert _matching_event(
        {"active_events": [resolved]},
        {**incoming, "entity_id": "event:resolved"},
    ) is resolved


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

    updated = _reduced_state(
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
        "permitted_role_assignment_handles": ["g1"],
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
        "permitted_role_assignment_handles": ["self"],
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
            r'\["self"\]'
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
async def test_appraisal_state_incompatibility_terminates_family_without_retry(
) -> None:
    """Reducer-owned state conflicts terminate with an empty family."""

    state = _state_with_goals()
    original_state = deepcopy(state)
    evidence = [_evidence()]
    projection = project_state_for_prompt(
        state,
        character_constraints=_character_constraints(),
        character_identity_context=canonical_identity_context(),
        evidence=evidence,
    )
    question = semantic_source_planner.plan_semantic_questions(
        evidence,
        state,
        projection.handle_to_ref,
    )[0]
    invalid = {
        "question_id": question["question_id"],
        "proposition": _proposition(
            "goal_supersession",
            "g1",
            object_handle="g2",
        ),
        "delta": None,
    }
    llm = _CapturingInvoker([invalid])

    result = await appraise_semantic_question(
        question,
        evidence,
        projection,
        _services(llm),
        validation_state=state,
    )

    assert result == {
        "question_id": question["question_id"],
        "selected_evidence_handles": [],
        "selected_role_handles": [],
        "propositions": [],
        "deltas": [],
        "explanation": "No additional supported semantic item.",
    }
    assert len(llm.configs) == 1
    first_payload = json.loads(str(llm.messages[0][1].content))
    handle_domains = first_payload["question"]["handle_field_domains"]
    assert handle_domains == {
        "subject_handle": question["permitted_role_handles"],
        "object_handle": question["permitted_role_handles"],
        "entity_handle": question["permitted_role_assignment_handles"],
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
    assert first_payload["question"]["permitted_target_paths"] == (
        question["permitted_delta_paths"]
    )
    assert state == original_state


def test_terminalized_low_salience_candidate_survives_same_batch_pruning(
) -> None:
    """A same-batch terminal proposition outranks weak-candidate pruning."""

    document = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    state = document["state"]
    evidence = document["evidence"]
    handle_to_ref = document["handle_to_ref"]

    updated = _reduced_state(
        state,
        document["appraisals"],
        evidence,
        handle_to_ref,
    )

    assert len(updated["active_events"]) == 1
    surviving = updated["active_events"][0]
    assert surviving["status"] == "resolved"
    assert surviving["repair_need"] == 0
    assert surviving["salience"] == 20
    expected_id = _causal_candidate_id(
        state,
        "event",
        evidence[0]["evidence_ref"],
    )
    assert surviving["entity_id"] == expected_id

    weak_updated = _reduced_state(
        state,
        [deepcopy(document["appraisals"][0])],
        evidence,
        handle_to_ref,
    )
    assert weak_updated["active_events"] == []


@pytest.mark.asyncio
async def test_appraisal_preserves_accepted_prefix_before_later_state_conflict(
) -> None:
    """A later reducer conflict keeps the accepted micro-item prefix only."""

    state = _state_with_goals()
    evidence = [_evidence()]
    projection = project_state_for_prompt(
        state,
        character_constraints=_character_constraints(),
        character_identity_context=canonical_identity_context(),
        evidence=evidence,
    )
    question = semantic_source_planner.plan_semantic_questions(
        evidence,
        state,
        projection.handle_to_ref,
    )[0]
    accepted = {
        "question_id": question["question_id"],
        "proposition": _proposition("outcome_pending", "g1"),
        "delta": None,
    }
    later_conflict = {
        "question_id": question["question_id"],
        "proposition": _proposition(
            "goal_supersession",
            "g1",
            object_handle="g2",
        ),
        "delta": None,
    }
    llm = _CapturingInvoker([accepted, later_conflict])

    result = await appraise_semantic_question(
        question,
        evidence,
        projection,
        _services(llm),
        validation_state=state,
    )

    assert len(llm.configs) == 2
    assert [
        proposition["proposition_kind"]
        for proposition in result["propositions"]
    ] == ["outcome_pending"]
    assert result["deltas"] == []
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
        _receipts,
    ) = reducer(
        state,
        [invalid, accepted],
        [_evidence()],
        {
            "g1": {"kind": "goal", "entity_id": "goal:primary"},
            "g2": {"kind": "goal", "entity_id": "goal:blocked"},
        },
        updated_at=_TIMESTAMP,
        character_constraints=_character_constraints(),
        relationship_context=None,
    )

    assert updated["goals"][0]["entity_id"] == "goal:primary"
    assert any(
        goal["entity_id"] == "goal:obstruction_resolution:user:goal:blocked"
        for goal in updated["goals"]
    )
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
        _receipts,
    ) = reducer(
        state,
        [materializer, delta_result],
        [_evidence()],
        handle_to_ref,
        updated_at=_TIMESTAMP,
        character_constraints=_character_constraints(),
        relationship_context=None,
    )

    assert accepted_results == [materializer, delta_result]
    assert failures == {}
    assert [row["outcome"] for row in comparisons] == ["create"]
    assert len(updated["active_events"]) == 1
    assert updated["active_events"][0]["outcome_impact"] == 30
    assert updated["active_events"][0]["salience"] == 30


def test_relationship_reduction_pins_all_current_batch_evidence() -> None:
    """Current delta provenance survives before historical retention fills."""

    state = build_acquaintance_user_state(
        global_user_id="retention-user",
        updated_at=_TIMESTAMP,
    )
    historical_ids = [
        "old-1",
        "old-2",
        "shared-current",
        "old-4",
        "old-5",
        "old-6",
        "old-7",
        "old-8",
    ]
    state["relationship"]["evidence_refs"] = [
        {
            "source_kind": "episode",
            "source_id": source_id,
            "occurred_at": _TIMESTAMP,
            "semantic_summary": f"Historical evidence {source_id}.",
        }
        for source_id in historical_ids
    ]
    source_ids = {
        "e1": "new-1",
        "e2": "new-2",
        "e3": "new-3",
        "e8": "shared-current",
    }
    evidence = [
        {
            "evidence_handle": handle,
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": source_id,
                "occurred_at": _TIMESTAMP,
                "semantic_summary": f"Current evidence {source_id}.",
            },
            "semantic_text": f"Current evidence {source_id}.",
            "visible_to": ["q:relationship_social"],
            "authority": "current_event",
        }
        for handle, source_id in source_ids.items()
    ]
    result = {
        "question_id": "q:relationship_social",
        "selected_evidence_handles": list(source_ids),
        "selected_role_handles": ["r1"],
        "propositions": [],
        "deltas": [
            {
                "target_path": "relationship.r1.trust",
                "delta": 5,
                "evidence_handles": ["e2", "e3"],
                "reason": "Current privacy evidence supports trust.",
            },
            {
                "target_path": "relationship.r1.desired_closeness",
                "delta": 5,
                "evidence_handles": ["e1", "e8"],
                "reason": "Current commitment evidence supports closeness.",
            },
        ],
        "explanation": "The current relationship evidence supports both deltas.",
    }

    updated = _reduced_state(
        state,
        [result],
        evidence,
        {
            "r1": {
                "scope": "user",
                "kind": "relationship",
                "entity_id": state["relationship"]["relationship_id"],
            }
        },
    )

    retained_ids = {
        row["source_id"]
        for row in updated["relationship"]["evidence_refs"]
    }
    assert len(updated["relationship"]["evidence_refs"]) == 8
    assert set(source_ids.values()) <= retained_ids
    assert updated["relationship"]["trust"] == 5
    assert updated["relationship"]["desired_closeness"] == 15


def test_causal_retention_preserves_primary_and_current_evidence() -> None:
    """A causal root remains first while current provenance is pinned."""

    historical = [
        {
            "source_kind": "episode",
            "source_id": f"historical-{index}",
            "occurred_at": _TIMESTAMP,
            "semantic_summary": f"Historical evidence {index}.",
        }
        for index in range(1, 9)
    ]
    current = [
        {
            "source_kind": "episode",
            "source_id": f"current-{index}",
            "occurred_at": _TIMESTAMP,
            "semantic_summary": f"Current evidence {index}.",
        }
        for index in range(1, 3)
    ]
    target = {"entity_id": "goal:retention", "evidence_refs": historical}

    _retain_current_batch_evidence(target, current)

    retained_ids = [row["source_id"] for row in target["evidence_refs"]]
    assert len(retained_ids) == 8
    assert retained_ids[0] == "historical-1"
    assert {"current-1", "current-2"} <= set(retained_ids)


def test_causal_retention_rejects_current_union_that_displaces_primary() -> None:
    """A full current union fails closed when its causal root cannot fit."""

    primary = {
        "source_kind": "episode",
        "source_id": "causal-root",
        "occurred_at": _TIMESTAMP,
        "semantic_summary": "The causal root evidence.",
    }
    current = [
        {
            "source_kind": "episode",
            "source_id": f"current-{index}",
            "occurred_at": _TIMESTAMP,
            "semantic_summary": f"Current evidence {index}.",
        }
        for index in range(1, 9)
    ]
    target = {"entity_id": "goal:retention", "evidence_refs": [primary]}

    with pytest.raises(
        CognitionStateError,
        match="current evidence exceeds retention capacity",
    ):
        _retain_current_batch_evidence(target, current)
