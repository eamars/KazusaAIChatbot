"""Direct ownership tests for semantic source planning."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    plan_semantic_questions,
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
