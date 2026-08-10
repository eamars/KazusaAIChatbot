"""V2 accepted-task cognition ownership tests."""

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS,
    EVIDENCE_SOURCE_QUESTION_IDS,
    project_evidence_provenance_role,
)


def test_tool_result_reenters_as_typed_evidence() -> None:
    """Accepted-task completion is evidence, not a private executor prompt."""

    question_ids = EVIDENCE_SOURCE_QUESTION_IDS["tool_result"]

    assert "q:event_agency" in question_ids
    assert "q:goal_threat_outcome" in question_ids
    assert len(question_ids) == len(set(question_ids))


def test_tool_result_receives_current_episode_authority() -> None:
    """Completed task results stay typed and project as current evidence."""

    assert "episode" in CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS
    assert "tool_result" in CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS
    assert project_evidence_provenance_role("tool_result", None) == (
        "current_episode"
    )
