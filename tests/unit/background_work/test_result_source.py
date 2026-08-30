"""Executable tests for DSH task-result cognition re-entry."""

from __future__ import annotations

from typing import Any

import pytest

from tests.task_resolution_test_helpers import _goal_continuation_ref


def _task_result() -> dict[str, object]:
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve one bounded goal.",
        "status": "resolved",
        "scene_context": {
            "channel_scope": "private",
            "character_role": "Test Character",
            "current_user_role": "Test User",
            "semantic_scene": "A result-source test.",
            "public_group_scene": "",
            "conversation_continuity": "The result continues the goal.",
            "semantic_temporal_context": "The test turn is current.",
        },
        "goal_continuation_ref": _goal_continuation_ref(),
        "evidence_state": "complete",
        "evidence_excerpts": ["The source supplied the bounded fact."],
        "evidence_handles": ["semantic-ref-1"],
        "prompt_safe_summary": "The bounded fact is available.",
        "evidence": [{
            "schema_version": "task_resolution_evidence.v1",
            "evidence_id": "evidence-1",
            "task_node_id": "node-1",
            "specialist": "dsh",
            "summary": "semantic-ref-1",
            "provenance_refs": ["receipt-1", "sha256:content"],
            "limitations": [],
        }],
        "completed_subgoals": ["source lookup"],
        "remaining_needs": [],
        "checkpoint": {},
        "coding_run_context": {},
    }


def _job() -> dict[str, Any]:
    return {
        "job_id": "job-1",
        "accepted_task_id": "task-1",
        "created_at": "2026-08-30T22:00:00Z",
        "completed_at": "2026-08-30T22:01:00Z",
        "source_platform": "debug",
        "source_channel_id": "channel-1",
        "source_channel_type": "private",
        "requester_platform_user_id": "debug-user-1",
        "requester_global_user_id": "user-1",
        "requester_display_name": "Test User",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Test Character",
        "source_message_id": "message-1",
        "source_llm_trace_id": "trace-1",
        "task_resolution_result": _task_result(),
        "worker_metadata": {},
    }


def test_dsh_result_reenters_cognition_with_exact_goal_and_evidence_provenance() -> None:
    """The real result-source projection preserves typed DSH provenance."""

    try:
        from kazusa_ai_chatbot.background_work import result_source
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned result-source owner is unavailable: {exc}")

    source = result_source.validate_tool_result_cognition_source({
        "source_kind": "tool_result",
        "source_id": "task-1",
        "occurred_at": "2026-08-30T22:01:00Z",
        "semantic_summary": "The bounded fact is available.",
        "semantic_objective": "Resolve one bounded goal.",
        "task_status": "resolved",
        "evidence_state": "complete",
        "evidence_excerpts": ["The source supplied the bounded fact."],
        "evidence_handles": ["semantic-ref-1"],
        "remaining_needs": [],
        "goal_continuation_ref": _goal_continuation_ref(),
    })
    assert source["goal_continuation_ref"] == _goal_continuation_ref()
    assert source["evidence_handles"] == ["semantic-ref-1"]

    episode = result_source.build_result_ready_episode_from_job(_job())
    percept = episode["percepts"][0]["content"]
    cognition_source = percept["cognition_source"]
    assert percept["semantic_summary"] == "The bounded fact is available."
    assert cognition_source["semantic_summary"] == "The bounded fact is available."
    assert cognition_source["semantic_objective"] == "Resolve one bounded goal."
    assert cognition_source["evidence_handles"] == ["semantic-ref-1"]
    assert cognition_source["goal_continuation_ref"] == _goal_continuation_ref()
    assert "task_resolution_context" in percept
    assert "coding_run_context" not in percept


def test_result_source_preserves_semantic_result_separate_from_receipts() -> None:
    """A resolved result retains semantic meaning and receipt provenance separately."""

    from kazusa_ai_chatbot.background_work import result_source

    job = _job()
    task_result = job["task_resolution_result"]
    assert isinstance(task_result, dict)
    marker = "PLAN3_E2E_BETA_SELECTED"
    task_result["prompt_safe_summary"] = f"The selected marker is {marker}."
    task_result["evidence_excerpts"] = [
        "The selected file was read successfully.",
        "The user-selected path was retained.",
    ]

    episode = result_source.build_result_ready_episode_from_job(job)
    cognition_source = episode["percepts"][0]["content"]["cognition_source"]

    assert cognition_source["evidence_excerpts"] == [
        "The selected file was read successfully.",
        "The user-selected path was retained.",
    ]
    assert cognition_source["semantic_summary"] == (
        f"The selected marker is {marker}."
    )
    assert episode["percepts"][0]["content"]["semantic_summary"] == (
        f"The selected marker is {marker}."
    )
    assert episode["evidence_refs"] == [{
        "schema_version": "evidence_ref.v1",
        "evidence_kind": "tool_result",
        "evidence_id": "semantic-ref-1",
        "owner": "dsh",
        "excerpt": "The selected file was read successfully.",
        "observed_at": "2026-08-30T22:01:00Z",
    }]


def test_result_source_retains_partial_semantics_and_limitations() -> None:
    """A partial result carries its semantic outcome and declared limitation."""

    from kazusa_ai_chatbot.background_work import result_source

    job = _job()
    task_result = job["task_resolution_result"]
    assert isinstance(task_result, dict)
    task_result["status"] = "partial"
    task_result["evidence_state"] = "partial"
    task_result["prompt_safe_summary"] = "The selected marker is incomplete."
    task_result["evidence_excerpts"] = ["The selected file was read."]
    task_result["remaining_needs"] = ["One bounded source remains unavailable."]

    episode = result_source.build_result_ready_episode_from_job(job)
    cognition_source = episode["percepts"][0]["content"]["cognition_source"]

    expected_summary = (
        "The selected marker is incomplete.; Remaining limitations: "
        "One bounded source remains unavailable."
    )
    assert cognition_source["semantic_summary"] == expected_summary
    assert cognition_source["evidence_excerpts"] == [
        "The selected file was read."
    ]
    assert cognition_source["remaining_needs"] == [
        "One bounded source remains unavailable."
    ]


def test_result_source_does_not_invent_refs_for_extra_artifact_handles() -> None:
    """Artifact handles remain source data instead of becoming evidence rows."""

    from kazusa_ai_chatbot.background_work import result_source

    job = _job()
    task_result = job["task_resolution_result"]
    assert isinstance(task_result, dict)
    task_result["evidence_handles"] = ["semantic-ref-1", "artifact-ref-2"]

    episode = result_source.build_result_ready_episode_from_job(job)
    cognition_source = episode["percepts"][0]["content"]["cognition_source"]

    assert cognition_source["evidence_handles"] == [
        "semantic-ref-1",
        "artifact-ref-2",
    ]
    assert [
        evidence_ref["evidence_id"] for evidence_ref in episode["evidence_refs"]
    ] == ["semantic-ref-1"]


def test_result_source_rejects_evidence_without_a_semantic_handle() -> None:
    """Each evidence receipt must retain its declared semantic handle."""

    from kazusa_ai_chatbot.background_work import result_source

    job = _job()
    task_result = job["task_resolution_result"]
    assert isinstance(task_result, dict)
    evidence = task_result["evidence"]
    assert isinstance(evidence, list)
    evidence[0]["summary"] = "missing-semantic-ref"

    with pytest.raises(
        ValueError,
        match="task result evidence handle is missing its semantic reference",
    ):
        result_source.build_result_ready_episode_from_job(job)
