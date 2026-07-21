"""Checkpoint D anti-drift alignment and two-phase boundary tests."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    EVIDENCE_SOURCE_QUESTION_IDS,
    validate_cognition_core_input,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    QUESTION_KINDS,
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    _project_question_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from tests.cognition_core_v2_test_helpers import canonical_episode
NOW = "2026-07-14T00:00:00Z"


def _constraints() -> dict[str, object]:
    """Return the character state as a separate read-only constraint object."""

    state = build_character_production_state(updated_at=NOW)
    return {
        "drives": state["drives"],
        "standards": state["standards"],
        "meaning_state": state["meaning_state"],
    }


def _evidence(source_kind: str = "episode") -> list[dict[str, object]]:
    """Build one typed semantic evidence row."""

    return [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": source_kind,
            "source_id": "episode-d",
            "occurred_at": NOW,
            "semantic_summary": "bounded episode evidence",
        },
        "semantic_text": "A bounded semantic observation is available.",
        "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS[source_kind]),
    }]


def test_episode_evidence_selects_each_family_once_with_unique_path_owners() -> None:
    """Planner coverage is provenance-driven and has exactly six families."""

    state = build_acquaintance_user_state(
        global_user_id="user-d",
        updated_at=NOW,
    )
    evidence = _evidence()
    constraints = _constraints()
    projection = project_state_for_prompt(
        state,
        character_constraints=constraints,
        evidence=evidence,
    )
    questions = plan_semantic_questions(
        evidence,
        state,
        projection.handle_to_ref,
    )

    assert [question["question_kind"] for question in questions] == list(
        QUESTION_KINDS
    )
    unique_paths = {
        path
        for question in questions
        for path in question["permitted_delta_paths"]
    }
    assert len(unique_paths) == sum(
        len(question["permitted_delta_paths"]) for question in questions
    )
    all_paths = [
        path
        for question in questions
        for path in question["permitted_delta_paths"]
    ]
    assert all(
        not path.endswith((".salience", ".importance", ".progress"))
        for path in all_paths
    )


def test_scheduler_evidence_selects_only_goal_threat_outcome() -> None:
    """Source provenance, not user-language keywords, selects question families."""

    state = build_acquaintance_user_state(
        global_user_id="user-d",
        updated_at=NOW,
    )
    evidence = _evidence("scheduler_event")
    constraints = _constraints()
    projection = project_state_for_prompt(
        state,
        character_constraints=constraints,
        evidence=evidence,
    )
    questions = plan_semantic_questions(
        evidence,
        state,
        projection.handle_to_ref,
    )

    assert [question["question_kind"] for question in questions] == [
        "goal_threat_outcome",
    ]


def test_each_question_receives_only_family_local_handles_and_state() -> None:
    """Keep every appraisal partition local to its evidence and entity family."""

    state = build_acquaintance_user_state(
        global_user_id="user-family-local",
        updated_at=NOW,
    )
    evidence = _evidence()
    constraints = _constraints()
    projection = project_state_for_prompt(
        state,
        character_constraints=constraints,
        evidence=evidence,
    )
    questions = plan_semantic_questions(
        evidence,
        state,
        projection.handle_to_ref,
    )
    by_kind = {question["question_kind"]: question for question in questions}

    assert "ct1" not in by_kind["event_agency"]["permitted_role_handles"]
    assert "ce1" not in by_kind["relationship_social"][
        "permitted_role_handles"
    ]
    assert "ck1" not in by_kind["moral_identity"]["permitted_role_handles"]
    assert "ct1" not in by_kind["epistemic_comparison_memory"][
        "permitted_role_handles"
    ]

    event_state = _project_question_state(
        projection,
        by_kind["event_agency"],
    )

    assert [row["handle"] for row in event_state["causal_candidates"]] == [
        "ce1"
    ]
    assert "relationship" not in event_state
    assert "character_constraints" not in event_state


def test_candidate_handles_share_the_projection_authority() -> None:
    """Keep sparse evidence ids resolvable through one prompt-handle owner."""

    state = build_acquaintance_user_state(
        global_user_id="user-candidate-binding",
        updated_at=NOW,
    )
    evidence = _evidence()
    evidence[0]["evidence_handle"] = "e7"
    constraints = _constraints()
    projection = project_state_for_prompt(
        state,
        character_constraints=constraints,
        evidence=evidence,
    )
    questions = plan_semantic_questions(
        evidence,
        state,
        projection.handle_to_ref,
    )
    event_question = next(
        question
        for question in questions
        if question["question_kind"] == "event_agency"
    )

    assert "ce1" in projection.handle_to_ref
    assert projection.handle_to_ref["ce1"]["entity_id"] == (
        "candidate:event:e7"
    )
    assert set(event_question["permitted_role_handles"]) <= set(
        projection.handle_to_ref
    )
    assert "ce1" in event_question["permitted_role_handles"]
    assert "active_events.ce1.intentionality" in event_question[
        "permitted_delta_paths"
    ]


def test_v2_input_rejects_scope_mismatch_before_any_model_call() -> None:
    """Invalid scope ownership fails at the public boundary."""

    state = build_acquaintance_user_state(
        global_user_id="user-d",
        updated_at=NOW,
    )
    payload = {
        "schema_version": "cognition_core_input.v2",
        "episode": canonical_episode(
            episode_id="alignment-direct-fact",
            trigger_source="internal_thought",
            current_global_user_id="user-d",
        ),
        "state_scope": "character",
        "mutable_state": state,
        "character_constraints": _constraints(),
        "evidence": _evidence(),
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "resolver_context": "resolver_status=idle",
        "scene_context": {
            "channel_scope": "internal",
            "character_role": "character",
            "semantic_scene": "test",
            "conversation_continuity": "No unresolved public commitment.",
            "semantic_temporal_context": "now",
        },
        "private_continuity_context": "I remain attentive.",
    }

    with pytest.raises(CognitionContractError, match="scope"):
        validate_cognition_core_input(payload)
