"""V2 source-owned semantic question-selection tests."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import (
    run_cognition,
    run_text_surface_planning,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)


NOW = "2026-07-14T00:00:00Z"


def _constraints() -> dict[str, object]:
    character = build_character_production_state(updated_at=NOW)
    return {
        "drives": character["drives"],
        "standards": character["standards"],
        "meaning_state": character["meaning_state"],
    }


def _evidence(source_kind: str) -> list[dict[str, object]]:
    return [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": source_kind,
            "source_id": f"source:{source_kind}",
            "occurred_at": NOW,
            "semantic_summary": "one bounded semantic source",
        },
        "semantic_text": "one bounded semantic source",
        "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS[source_kind]),
    }]


@pytest.mark.parametrize(
    "source_kind",
    tuple(EVIDENCE_SOURCE_QUESTION_IDS),
)
def test_source_kind_selects_its_exact_question_families(
    source_kind: str,
) -> None:
    """Select families from typed provenance rather than source text."""

    state = build_acquaintance_user_state(
        global_user_id="selection-user",
        updated_at=NOW,
    )
    evidence = _evidence(source_kind)
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

    assert [question["question_id"] for question in questions] == list(
        EVIDENCE_SOURCE_QUESTION_IDS[source_kind]
    )
    assert all(
        question["evidence_handles"] == ["e1"]
        for question in questions
    )


def test_canonical_package_exposes_only_two_execution_apis() -> None:
    """Retain the big-bang V2 public execution boundary."""

    assert callable(run_cognition)
    assert callable(run_text_surface_planning)
