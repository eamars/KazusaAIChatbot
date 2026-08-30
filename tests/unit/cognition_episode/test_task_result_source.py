"""Executable tests for typed DSH task-result cognition episodes."""

from __future__ import annotations

from tests.task_resolution_test_helpers import _goal_continuation_ref


def test_tool_result_episode_projects_dsh_task_context_and_provenance() -> None:
    """The real episode builder preserves DSH lineage without coding context."""

    from kazusa_ai_chatbot import cognition_episode

    task_context = {
        "schema_version": "task_resolution_context.v1",
        "resolution_ref": {
            "schema_version": "dsh_resolution_ref.v1",
            "resolution_thread_id": "thread-1",
            "segment_id": "segment-1",
            "dsh_session_id": "session-1",
            "activation_id": "activation-1",
            "lease_epoch": 1,
            "document_revision": 2,
            "last_committed_seq": 3,
        },
        "operation_generation": 0,
    }
    result = {
        "schema_version": "tool_result_ready.v1",
        "task_id": "task-1",
        "task_kind": "task_resolution",
        "semantic_summary": "The DSH task returned bounded evidence.",
        "artifact_text": "",
        "failure_text": "",
        "completed_at": "2026-08-30T00:00:00Z",
        "target_scope": {
            "platform": "debug",
            "platform_channel_id": "channel-1",
            "global_user_id": "user-1",
        },
        "evidence_refs": [],
        "result_ref": "result-1",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Test Character",
        "source_message_id": "message-1",
        "goal_continuation_ref": _goal_continuation_ref(),
        "task_resolution_context": task_context,
    }
    evidence_refs = [{
        "schema_version": "evidence_ref.v1",
        "evidence_kind": "dsh",
        "evidence_id": "receipt-1",
        "owner": "dsh",
        "excerpt": "The DSH result is bounded.",
        "observed_at": "2026-08-30T00:00:00Z",
    }]

    episode = cognition_episode.build_tool_result_episode(
        result=result,
        evidence_refs=evidence_refs,
        local_time_context={
            "current_local_datetime": "2026-08-30T12:00:00",
            "current_local_weekday": "Sunday",
        },
        created_at="2026-08-30T00:00:00Z",
    )

    assert episode["trigger_source"] == "tool_result"
    assert episode["origin_metadata"]["goal_continuation_ref"] == result[
        "goal_continuation_ref"
    ]
    assert episode["origin_metadata"]["task_resolution_context"] == task_context
    assert episode["evidence_refs"] == evidence_refs
    assert episode["percepts"][0]["content"]["goal_continuation_ref"] == result[
        "goal_continuation_ref"
    ]
    assert "coding_run_context" not in episode["origin_metadata"]
