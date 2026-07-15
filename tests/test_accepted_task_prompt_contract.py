"""V2 accepted-task cognition ownership tests."""

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
)


def test_accepted_task_result_reenters_as_typed_evidence() -> None:
    """Accepted-task completion is evidence, not a private executor prompt."""

    question_ids = EVIDENCE_SOURCE_QUESTION_IDS["accepted_task_result"]

    assert "q:event_agency" in question_ids
    assert "q:goal_threat_outcome" in question_ids
    assert len(question_ids) == len(set(question_ids))
