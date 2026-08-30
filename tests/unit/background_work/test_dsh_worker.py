"""Executable tests for the generation-fenced DSH worker."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.task_resolution_test_helpers import _goal_continuation_ref


def _job(*, state: str = "running", generation: int = 0) -> dict[str, Any]:
    return {
        "schema_version": "background_work_job.v2",
        "job_id": "job-1",
        "idempotency_key": "background_work:task-1",
        "source_action_attempt_id": "attempt-1",
        "source_llm_trace_id": "trace-1",
        "correlation_write_status": "written",
        "correlation_conflict_source_llm_trace_id": "",
        "accepted_task_id": "task-1",
        "task_identity_key": "identity-1",
        "semantic_objective": "Resolve one goal.",
        "goal_continuation_ref": _goal_continuation_ref(),
        "status": state,
        "delivery_state": "not_ready",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "source_platform": "debug",
        "source_channel_id": "channel-1",
        "source_channel_type": "private",
        "source_message_id": "message-1",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Test Character",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "requester_display_name": "Test User",
        "created_at": "2026-08-30T22:00:00Z",
        "updated_at": "2026-08-30T22:00:00Z",
        "lease_owner": "worker-1",
        "lease_expires_at": "2026-08-30T22:01:00Z",
        "attempt_count": 1,
        "max_attempts": 4,
        "requested_worker": "task_orchestrator",
        "worker_payload": {
            "schema_version": "task_orchestrator_worker_payload.v2",
            "operation": "open_dsh_resolution",
            "task_session_id": "session-1",
            "operation_generation": generation,
            "control": None,
        },
        "task_execution_context": {},
        "task_resolution_result": None,
        "artifact_text": "",
        "failure_summary": "",
        "result_summary": "",
        "completed_at": "",
        "delivery_attempt_count": 0,
        "delivery_failure_summary": "",
        "delivery_tracking_id": "",
        "delivered_conversation_message_id": "",
        "delivered_at": "",
    }


def _result(status: str = "resolved") -> dict[str, object]:
    terminal = status in {"resolved", "partial"}
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve one goal.",
        "status": status,
        "scene_context": {
            "channel_scope": "private",
            "character_role": "Test Character",
            "current_user_role": "Test User",
            "semantic_scene": "A worker test.",
            "public_group_scene": "",
            "conversation_continuity": "The same goal.",
            "semantic_temporal_context": "Now.",
        },
        "goal_continuation_ref": _goal_continuation_ref(),
        "evidence_state": "complete" if status == "resolved" else "pending",
        "evidence_excerpts": [],
        "evidence_handles": [],
        "prompt_safe_summary": "The task is complete.",
        "evidence": [{
            "schema_version": "task_resolution_evidence.v1",
            "evidence_id": "evidence-1",
            "task_node_id": "node-1",
            "specialist": "dsh",
            "summary": "A bounded worker result.",
            "provenance_refs": ["receipt-1"],
            "limitations": [],
        }] if terminal else [],
        "completed_subgoals": ["bounded worker result"] if terminal else [],
        "remaining_needs": [] if terminal else ["one typed next step"],
        "checkpoint": {},
        "coding_run_context": {},
    }


def _worker() -> Any:
    from kazusa_ai_chatbot.background_work import worker

    return worker


@pytest.mark.asyncio
async def test_worker_checkpoints_waits_and_terminalizes_current_generation_through_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claimed generation reaches the task sink before job completion."""

    worker = _worker()
    execute = AsyncMock(return_value=_result())
    mark_running = AsyncMock(return_value={"state": "running"})
    mark_ready = AsyncMock(return_value={"state": "result_ready"})
    complete = AsyncMock(return_value={"status": "completed"})
    monkeypatch.setattr(worker, "execute_task_orchestrator_job", execute)
    monkeypatch.setattr(worker, "mark_accepted_task_running", mark_running)
    monkeypatch.setattr(worker, "mark_tool_result_ready", mark_ready)
    monkeypatch.setattr(worker, "complete_background_work_job", complete)

    outcome = await worker._run_claimed_job(_job(), lease_owner="worker-1")

    assert outcome == "completed"
    execute.assert_awaited_once()
    mark_ready.assert_awaited_once()
    complete.assert_awaited_once()
    assert complete.await_args.kwargs["task_resolution_result"]["status"] == "resolved"


@pytest.mark.asyncio
async def test_internal_dsh_interaction_never_requeues_for_user_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cooperative checkpoint is requeued only as queued continuation."""

    worker = _worker()
    requeue = AsyncMock(return_value={"status": "queued"})
    monkeypatch.setattr(worker, "requeue_background_work_job", requeue)
    result = _result("partial")
    result["checkpoint"] = {
        "schema_version": "dsh_resolution_ref.v1",
        "dsh_session_id": "session-1",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "activation_id": "activation-1",
        "lease_epoch": 1,
        "document_revision": 1,
        "last_committed_seq": 1,
    }
    await worker._requeue_task_orchestrator_job(
        _job(),
        lease_owner="worker-1",
        result=result,
    )
    requeue.assert_awaited_once()
    assert requeue.await_args.kwargs["status"] == "queued"


@pytest.mark.asyncio
async def test_worker_retry_reuses_idempotent_task_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry claims the same job/session without opening a replacement."""

    worker = _worker()
    claimed = [_job(), _job()]
    claim = AsyncMock(side_effect=[claimed[0], claimed[1], None])
    running = AsyncMock(return_value={"state": "running"})
    process = AsyncMock(return_value="completed")
    monkeypatch.setattr(worker, "claim_background_work_job", claim)
    monkeypatch.setattr(worker, "mark_accepted_task_running", running)
    monkeypatch.setattr(worker, "_run_claimed_job", process)

    result = await worker.run_background_work_worker_tick(
        claim_limit=2,
        lease_seconds=30,
        max_attempts=4,
        worker_id="worker-1",
    )

    assert result["processed_count"] == 2
    assert [call.args[0]["job_id"] for call in process.await_args_list] == [
        "job-1",
        "job-1",
    ]
    assert all(
        call.args[0]["worker_payload"]["task_session_id"] == "session-1"
        for call in process.await_args_list
    )


@pytest.mark.asyncio
async def test_worker_rejects_stale_or_canceled_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale/canceled rows fail closed without invoking DSH execution."""

    worker = _worker()
    claim = AsyncMock(return_value=_job(state="canceled", generation=9))
    running = AsyncMock(return_value={"state": "running"})
    execute = AsyncMock()
    failed = AsyncMock(return_value={"status": "failed"})
    task_failed = AsyncMock(return_value={"state": "failure_ready"})
    monkeypatch.setattr(worker, "claim_background_work_job", claim)
    monkeypatch.setattr(worker, "mark_accepted_task_running", running)
    monkeypatch.setattr(worker, "execute_task_orchestrator_job", execute)
    monkeypatch.setattr(worker, "fail_background_work_job", failed)
    monkeypatch.setattr(worker, "mark_accepted_task_failure_ready", task_failed)

    result = await worker.run_background_work_worker_tick(
        claim_limit=1,
        worker_id="worker-1",
    )

    assert result["processed_count"] == 1
    execute.assert_not_awaited()
    failed.assert_awaited_once()


@pytest.mark.asyncio
async def test_task_orchestrator_dispatches_only_generation_bound_dsh_payload_v2_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker adapter forwards only the DSH payload V2 operation."""

    worker = _worker()
    execute = AsyncMock(return_value=_result())
    monkeypatch.setattr(worker, "execute_task_orchestrator_job", execute)
    monkeypatch.setattr(
        worker,
        "mark_tool_result_ready",
        AsyncMock(return_value={"state": "result_ready"}),
    )
    monkeypatch.setattr(
        worker,
        "complete_background_work_job",
        AsyncMock(return_value={"status": "completed"}),
    )
    await worker._run_claimed_job(_job(), lease_owner="worker-1")
    payload = execute.await_args.args[0]["worker_payload"]
    assert set(payload) == {
        "schema_version",
        "operation",
        "task_session_id",
        "operation_generation",
        "control",
    }
    assert payload["operation"] in {
        "open_dsh_resolution",
        "continue_dsh_resolution",
    }
