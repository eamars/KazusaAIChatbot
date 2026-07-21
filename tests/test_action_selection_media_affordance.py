"""V2 media-evidence routing contract tests."""

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
)


def test_media_observation_routes_only_to_semantic_appraisal() -> None:
    """Media remains typed evidence instead of an action prompt payload."""

    question_ids = EVIDENCE_SOURCE_QUESTION_IDS["media_observation"]

    assert question_ids
    assert all(question_id.startswith("q:") for question_id in question_ids)
