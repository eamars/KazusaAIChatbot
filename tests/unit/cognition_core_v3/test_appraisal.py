"""Deterministic tests for grouped appraisal serial semantics."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import appraisal


def test_grouped_appraisal_validator_uses_exact_micro_item_contract() -> None:
    valid = {
        "event_agency": [
            {
                "question_id": "event_agency",
                "proposition": None,
                "delta": None,
            }
        ]
    }
    assert appraisal.validate_grouped_appraisal_output(
        valid,
        planned_families=["event_agency"],
    ) == valid

    with pytest.raises(ValueError, match="exactly"):
        appraisal.validate_grouped_appraisal_output(
            valid,
            planned_families=["event_agency", "relationship_social"],
        )


def test_grouped_appraisal_reducer_maps_to_v2_validated_row() -> None:
    raw = {
        "event_agency": [
            {
                "question_id": "event_agency",
                "proposition": None,
                "delta": None,
            }
        ]
    }
    question = {
        "question_id": "q:event_agency",
        "question_kind": "event_agency",
        "semantic_question": "判断责任和意图",
        "evidence_handles": [],
        "permitted_role_handles": [],
        "permitted_role_assignment_handles": [],
        "permitted_delta_paths": [],
        "dependencies": [],
    }

    reduced = appraisal.reduce_grouped_appraisal_output(
        raw,
        planned_families=["event_agency"],
        questions_by_family={"event_agency": question},
        evidence_handles=[],
        handle_to_ref={},
    )

    assert len(reduced) == 1
    assert reduced[0]["question_id"] == "q:event_agency"
    assert reduced[0]["selected_evidence_handles"] == []
    assert reduced[0]["selected_role_handles"] == []
    assert reduced[0]["propositions"] == []
    assert reduced[0]["deltas"] == []
