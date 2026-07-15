"""V2 promoted-reflection evidence ownership tests."""

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
)


def test_promoted_reflection_is_typed_evidence_for_all_appraisals() -> None:
    """Global growth reaches cognition only through promoted evidence."""

    question_ids = EVIDENCE_SOURCE_QUESTION_IDS["promoted_reflection"]

    assert len(question_ids) == 6
    assert all(question_id.startswith("q:") for question_id in question_ids)
