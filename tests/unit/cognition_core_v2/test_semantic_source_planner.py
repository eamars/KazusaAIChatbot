"""Direct ownership tests for semantic source planning."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    plan_semantic_questions,
    question_proposition_kinds,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from tests.cognition_core_v2_test_helpers import canonical_identity_context


NOW = "2026-07-14T00:00:00Z"


def _constraints() -> dict[str, Any]:
    """Build the bounded character constraint projection used by the planner."""

    state = build_character_production_state(updated_at=NOW)
    constraints = {
        "drives": state["drives"],
        "standards": state["standards"],
        "meaning_state": state["meaning_state"],
        "personality_judgment": {
            "logic": "evidence-led",
            "defense": "reserved",
            "quirks": "brief hesitation",
            "taboos": "preserve character agency",
        },
    }
    return constraints


def _projection() -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    """Build one episode evidence row and its prompt-safe state projection."""

    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode-unit",
            "occurred_at": NOW,
            "semantic_summary": "bounded episode evidence",
        },
        "semantic_text": "A bounded semantic observation is available.",
        "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS["episode"]),
        "authority": "current_event",
    }]
    state = build_acquaintance_user_state(
        global_user_id="user-unit",
        updated_at=NOW,
    )
    projection = project_state_for_prompt(
        state,
        character_constraints=_constraints(),
        character_identity_context=canonical_identity_context(),
        evidence=evidence,
    )
    return evidence, state, projection


def _entity_evidence(kind: str, index: int) -> dict[str, str]:
    """Build one complete provenance row for a planner state entity."""

    return {
        "source_kind": "episode",
        "source_id": f"planner-{kind}-{index}",
        "occurred_at": NOW,
        "semantic_summary": f"Planner evidence for {kind} {index}.",
    }


def _goal(index: int, status: str) -> dict[str, Any]:
    """Build one complete goal row with a selected lifecycle status."""

    return {
        "entity_id": f"goal:planner-{index}",
        "description": f"Planner goal {index}.",
        "salience": 50,
        "role_refs": [],
        "evidence_refs": [_entity_evidence("goal", index)],
        "created_at": NOW,
        "updated_at": NOW,
        "status": status,
        "goal_kind": "ordinary_response",
        "importance": 50,
        "progress": 0,
        "obstruction": 0,
        "expected_success": 50,
        "controllability": 50,
        "recoverability": 50,
        "urgency": 50,
    }


def _threat(index: int, status: str) -> dict[str, Any]:
    """Build one complete threat row with a selected lifecycle status."""

    return {
        "entity_id": f"threat:planner-{index}",
        "description": f"Planner threat {index}.",
        "salience": 50,
        "role_refs": [],
        "evidence_refs": [_entity_evidence("threat", index)],
        "created_at": NOW,
        "updated_at": NOW,
        "status": status,
        "likelihood": 50,
        "expected_harm": 50,
        "uncertainty": 50,
        "controllability": 50,
        "coping_potential": 50,
        "residual_pressure": 50,
    }


def _event(index: int, status: str) -> dict[str, Any]:
    """Build one complete event row with a selected lifecycle status."""

    return {
        "entity_id": f"event:planner-{index}",
        "description": f"Planner event {index}.",
        "salience": 50,
        "role_refs": [],
        "evidence_refs": [_entity_evidence("event", index)],
        "created_at": NOW,
        "updated_at": NOW,
        "status": status,
        "outcome_impact": 50,
        "responsibility": 50,
        "intentionality": 50,
        "harm": 50,
        "unfairness": 50,
        "exposure": 50,
        "repair_need": 50,
        "reparability": 50,
        "expectation_mismatch": 50,
        "norm_violation": 50,
        "contamination_risk": 50,
        "identity_threat": 50,
        "comparison_gap": 50,
        "vastness": 50,
        "memory_warmth": 50,
        "temporal_loss": 50,
    }


def _knowledge_gap(index: int, status: str) -> dict[str, Any]:
    """Build one complete knowledge-gap row with a selected lifecycle."""

    return {
        "entity_id": f"knowledge_gap:planner-{index}",
        "description": f"Planner knowledge gap {index}.",
        "salience": 50,
        "role_refs": [],
        "evidence_refs": [_entity_evidence("knowledge-gap", index)],
        "created_at": NOW,
        "updated_at": NOW,
        "status": status,
        "relevance": 50,
        "uncertainty": 50,
        "learnability": 50,
        "novelty": 50,
        "model_accommodation": 50,
    }


def _lifecycle_state() -> dict[str, Any]:
    """Build eligible and terminal rows for every q:goal_threat_outcome kind."""

    state = build_acquaintance_user_state(
        global_user_id="planner-lifecycle-user",
        updated_at=NOW,
    )
    state["goals"] = [
        _goal(1, "pursuing"),
        _goal(2, "blocked"),
        _goal(3, "satisfied"),
    ]
    state["threats"] = [
        _threat(1, "active"),
        _threat(2, "resolved"),
    ]
    state["active_events"] = [
        _event(1, "active"),
        _event(2, "replaced"),
    ]
    state["knowledge_gaps"] = [
        _knowledge_gap(1, "open"),
        _knowledge_gap(2, "reduced"),
        _knowledge_gap(3, "resolved"),
    ]
    return state


def _goal_threat_evidence() -> list[dict[str, Any]]:
    """Build evidence that authorizes only the terminal-outcome family."""

    return [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode-planner-terminal",
            "occurred_at": NOW,
            "semantic_summary": "A current outcome observation is available.",
        },
        "semantic_text": "A current outcome observation is available.",
        "visible_to": ["q:goal_threat_outcome"],
        "authority": "current_event",
    }]


def _goal_outcome_question() -> dict[str, Any]:
    """Build the terminal-outcome question from mixed lifecycle state."""

    evidence = _goal_threat_evidence()
    state = _lifecycle_state()
    projection = project_state_for_prompt(
        state,
        character_constraints=_constraints(),
        character_identity_context=canonical_identity_context(),
        evidence=evidence,
    )

    question = next(
        question
        for question in plan_semantic_questions(
            evidence,
            state,
            projection.handle_to_ref,
        )
        if question["question_kind"] == "goal_threat_outcome"
    )
    return question


def test_goal_outcome_filters_terminal_handles_and_delta_paths() -> None:
    """Exclude terminal native rows and every path owned by those rows."""

    question = _goal_outcome_question()
    handles = set(question["permitted_role_handles"])
    paths = set(question["permitted_delta_paths"])

    assert {"g3", "t2", "ev2", "k3"}.isdisjoint(handles)
    assert not any(path.startswith("goals.g3.") for path in paths)
    assert not any(path.startswith("threats.t2.") for path in paths)
    assert not any(path.startswith("active_events.ev2.") for path in paths)
    assert not any(path.startswith("knowledge_gaps.") for path in paths)


def test_goal_outcome_keeps_eligible_handles_and_candidates() -> None:
    """Keep eligible native rows, candidates, and proposition-only answers."""

    question = _goal_outcome_question()
    handles = set(question["permitted_role_handles"])
    paths = set(question["permitted_delta_paths"])

    assert {"g1", "g2", "t1", "ev1", "k1", "k2"} <= handles
    assert {"ce1", "ct1", "ck1"} <= handles
    assert any(path.startswith("goals.g1.") for path in paths)
    assert any(path.startswith("goals.g2.") for path in paths)
    assert any(path.startswith("threats.t1.") for path in paths)
    assert any(path.startswith("active_events.ev1.") for path in paths)
    assert any(path.startswith("active_events.ce1.") for path in paths)
    assert any(path.startswith("threats.ct1.") for path in paths)
    assert not any(path.endswith(".ck1.uncertainty") for path in paths)
    assert "knowledge_answered" in question_proposition_kinds(
        question["question_kind"]
    )


def test_moral_identity_questions_exclude_standard_handles() -> None:
    """Keep repository standards out of model-facing role-handle domains."""

    evidence, state, projection = _projection()
    questions = plan_semantic_questions(
        evidence,
        state,
        projection.handle_to_ref,
    )
    moral_question = next(
        question
        for question in questions
        if question["question_kind"] == "moral_identity"
    )

    assert moral_question["permitted_role_handles"]
    assert all(
        not (handle.startswith("s") and handle[1:].isdigit())
        for handle in moral_question["permitted_role_handles"]
    )
    assert "s1" not in moral_question["permitted_role_handles"]


def test_semantic_source_planner_exposes_owned_contract() -> None:
    """Keep the public planner entrypoint attached to this source owner."""

    assert callable(plan_semantic_questions)
