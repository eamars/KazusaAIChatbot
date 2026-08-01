"""Tests for durable task-resolution resume and lease recovery."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock

import pytest

from tests.test_background_work_jobs import _resume_queue_request
from tests.test_task_resolution_orchestrator import _context
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.task_resolution import TaskResolutionContractError


def _job(
    checkpoint: dict[str, object],
    *,
    task_result: dict[str, object] | None = None,
) -> dict[str, object]:
    request = _resume_queue_request()
    worker_payload = dict(request["worker_payload"])
    worker_payload["checkpoint"] = checkpoint
    return {
        "schema_version": "background_work_job.v2",
        "job_id": "job-resume-001",
        "worker_payload": worker_payload,
        "task_execution_context": _context(),
        "task_resolution_result": task_result or {},
    }


def _recorded_checkpoint(
    *,
    status: str,
    initial_checkpoint: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")
    initial = initial_checkpoint or (
        _resume_queue_request()["worker_payload"]["checkpoint"]
    )
    specialist = (
        "public_research" if status == "resolved" else "text_computation"
    )
    result: dict[str, object] = {
        "schema_version": "task_specialist_result.v1",
        "specialist": specialist,
        "status": status,
        "evidence": [],
        "completed_subgoals": [],
        "remaining_needs": ["Find one public source."],
        "reason": "The first route was incompatible.",
        "retryable": False,
    }
    if status == "resolved":
        result["evidence"] = [{
            "schema_version": "task_resolution_evidence.v1",
            "evidence_id": "public-evidence-1",
            "task_node_id": initial["active_node_id"],
            "specialist": "public_research",
            "summary": "A public source resolved the requested fact.",
            "provenance_refs": ["https://example.com/source"],
            "limitations": [],
        }]
        result["completed_subgoals"] = ["Find one public source."]
        result["remaining_needs"] = []
        result["reason"] = "Public evidence resolved the task."
    updated = state.record_specialist_result(initial, result)
    if status == "resolved":
        snapshot = state.result_from_checkpoint(
            updated,
            status="resolved",
            prompt_safe_summary="The task resolved with validated evidence.",
            completed_subgoals=list(result["completed_subgoals"]),
            coding_run_context={},
        )
    else:
        snapshot = state.result_from_checkpoint(
            updated,
            status="deferred",
            prompt_safe_summary="The task needs durable continuation.",
            completed_subgoals=[],
            coding_run_context={},
        )
    return updated, snapshot


@pytest.mark.asyncio
async def test_terminal_snapshot_recovers_without_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reclaimed terminal checkpoint returns its atomically paired result."""

    worker = importlib.import_module(
        "kazusa_ai_chatbot.background_work.subagent.task_orchestrator"
    )
    checkpoint, snapshot = _recorded_checkpoint(status="resolved")
    run = AsyncMock(side_effect=AssertionError("terminal work must not rerun"))
    monkeypatch.setattr(worker, "run_task_orchestrator", run)

    result = await worker.execute_task_orchestrator_job(
        _job(checkpoint, task_result=snapshot),
        lease_owner="worker-a",
    )

    assert result == snapshot
    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_snapshot_persists_under_active_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each completed dispatch atomically stores checkpoint and result state."""

    worker = importlib.import_module(
        "kazusa_ai_chatbot.background_work.subagent.task_orchestrator"
    )
    checkpoint, snapshot = _recorded_checkpoint(status="resolved")
    persist = AsyncMock(return_value={"job_id": "job-resume-001"})

    async def run_task_orchestrator(
        initial_checkpoint: dict[str, object],
        execution_context: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        assert initial_checkpoint["dispatch_count"] == 0
        assert execution_context == _context()
        callback = kwargs["checkpoint_persist_func"]
        await callback(checkpoint, snapshot)
        return snapshot

    monkeypatch.setattr(worker, "checkpoint_background_work_job", persist)
    monkeypatch.setattr(worker, "run_task_orchestrator", run_task_orchestrator)

    result = await worker.execute_task_orchestrator_job(
        _job(_resume_queue_request()["worker_payload"]["checkpoint"]),
        lease_owner="worker-a",
    )

    assert result == snapshot
    persisted = persist.await_args.kwargs
    assert persisted["job_id"] == "job-resume-001"
    assert persisted["lease_owner"] == "worker-a"
    assert persisted["checkpoint"]["dispatch_count"] == 1
    assert persisted["task_resolution_result"] == snapshot
    assert "release_for_resume" not in persisted


@pytest.mark.asyncio
async def test_checkpoint_write_fails_closed_after_lease_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker stops as soon as its checkpoint lease is no longer current."""

    worker = importlib.import_module(
        "kazusa_ai_chatbot.background_work.subagent.task_orchestrator"
    )
    checkpoint, snapshot = _recorded_checkpoint(status="resolved")

    async def run_task_orchestrator(
        initial_checkpoint: dict[str, object],
        execution_context: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del initial_checkpoint, execution_context
        callback = kwargs["checkpoint_persist_func"]
        await callback(checkpoint, snapshot)
        raise AssertionError("lost lease must stop before another dispatch")

    monkeypatch.setattr(
        worker,
        "checkpoint_background_work_job",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(worker, "run_task_orchestrator", run_task_orchestrator)

    with pytest.raises(TaskResolutionContractError, match="lease"):
        await worker.execute_task_orchestrator_job(
            _job(_resume_queue_request()["worker_payload"]["checkpoint"]),
            lease_owner="stale-worker",
        )


@pytest.mark.asyncio
async def test_queue_retry_preserves_semantic_counters_and_prior_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lease retry resumes the durable dispatch ledger instead of resetting."""

    worker = importlib.import_module(
        "kazusa_ai_chatbot.background_work.subagent.task_orchestrator"
    )
    checkpoint, prior_snapshot = _recorded_checkpoint(status="incompatible")
    terminal_checkpoint, terminal_snapshot = _recorded_checkpoint(
        status="resolved",
        initial_checkpoint=checkpoint,
    )

    async def run_task_orchestrator(
        resumed_checkpoint: dict[str, object],
        execution_context: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del execution_context
        assert resumed_checkpoint["dispatch_count"] == 1
        assert resumed_checkpoint["route_correction_count"] == 1
        assert kwargs["prior_result"] == prior_snapshot
        callback = kwargs["checkpoint_persist_func"]
        await callback(terminal_checkpoint, terminal_snapshot)
        return terminal_snapshot

    monkeypatch.setattr(
        worker,
        "checkpoint_background_work_job",
        AsyncMock(return_value={"job_id": "job-resume-001"}),
    )
    monkeypatch.setattr(worker, "run_task_orchestrator", run_task_orchestrator)

    result = await worker.execute_task_orchestrator_job(
        _job(checkpoint, task_result=prior_snapshot),
        lease_owner="worker-b",
    )

    assert result == terminal_snapshot


@pytest.mark.asyncio
async def test_worker_boundary_records_unexpected_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected worker exception releases the job into failed delivery."""

    worker = importlib.import_module("kazusa_ai_chatbot.background_work.worker")
    request = _resume_queue_request()
    job = {
        **request,
        "schema_version": "background_work_job.v2",
        "job_id": "job-worker-failure",
    }
    claim = AsyncMock(side_effect=[job, None])
    fail = AsyncMock(return_value={"status": "failed"})
    monkeypatch.setattr(worker, "claim_background_work_job", claim)
    monkeypatch.setattr(
        worker,
        "mark_accepted_task_running",
        AsyncMock(return_value={"state": "running"}),
    )
    monkeypatch.setattr(worker, "_run_claimed_job", AsyncMock(
        side_effect=RuntimeError("provider connection closed"),
    ))
    monkeypatch.setattr(worker, "fail_background_work_job", fail)
    monkeypatch.setattr(worker, "mark_accepted_task_failure_ready", AsyncMock())

    result = await worker.run_background_work_worker_tick(
        claim_limit=2,
        lease_seconds=60,
        max_attempts=4,
        worker_id="worker-failure-test",
    )

    assert result == {
        "processed_count": 1,
        "succeeded_count": 0,
        "failed_count": 1,
    }
    fail.assert_awaited_once()
    assert fail.await_args.kwargs["job_id"] == "job-worker-failure"
    assert fail.await_args.kwargs["lease_owner"] == "worker-failure-test"


@pytest.mark.asyncio
async def test_worker_rejects_claim_when_accepted_task_is_not_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claimed job cannot execute without a valid running transition."""

    worker = importlib.import_module("kazusa_ai_chatbot.background_work.worker")
    request = _resume_queue_request()
    job = {
        **request,
        "schema_version": "background_work_job.v2",
        "job_id": "job-invalid-task-state",
    }
    execute = AsyncMock()
    fail = AsyncMock(return_value={"status": "failed"})
    monkeypatch.setattr(
        worker,
        "claim_background_work_job",
        AsyncMock(side_effect=[job, None]),
    )
    monkeypatch.setattr(
        worker,
        "mark_accepted_task_running",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(worker, "_run_claimed_job", execute)
    monkeypatch.setattr(worker, "fail_background_work_job", fail)

    result = await worker.run_background_work_worker_tick(
        claim_limit=2,
        lease_seconds=60,
        max_attempts=4,
        worker_id="worker-state-guard",
    )

    assert result == {
        "processed_count": 1,
        "succeeded_count": 0,
        "failed_count": 1,
    }
    execute.assert_not_awaited()
    fail.assert_awaited_once()
    assert fail.await_args.kwargs["skip_result_delivery"] is True


@pytest.mark.asyncio
async def test_accepted_result_is_ready_before_job_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash cannot strand a completed job behind a running accepted task."""

    worker = importlib.import_module("kazusa_ai_chatbot.background_work.worker")
    checkpoint, snapshot = _recorded_checkpoint(status="resolved")
    request = _resume_queue_request()
    job = {
        **request,
        "schema_version": "background_work_job.v2",
        "job_id": "job-completion-order",
        "max_output_chars": 3000,
        "worker_payload": {
            **request["worker_payload"],
            "checkpoint": checkpoint,
        },
    }
    events: list[str] = []

    async def mark_result(**kwargs: object) -> dict[str, object]:
        del kwargs
        events.append("accepted_result_ready")
        return {"state": "result_ready"}

    async def complete_job(**kwargs: object) -> dict[str, object]:
        del kwargs
        events.append("job_completed")
        return {"status": "completed"}

    monkeypatch.setattr(worker, "mark_tool_result_ready", mark_result)
    monkeypatch.setattr(worker, "complete_background_work_job", complete_job)

    await worker._complete_task_orchestrator_job(
        job,
        lease_owner="worker-order-test",
        result=snapshot,
    )

    assert events == ["accepted_result_ready", "job_completed"]


@pytest.mark.asyncio
async def test_missing_accepted_task_blocks_job_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker keeps its job reclaimable if accepted-task state is missing."""

    worker = importlib.import_module("kazusa_ai_chatbot.background_work.worker")
    checkpoint, snapshot = _recorded_checkpoint(status="resolved")
    request = _resume_queue_request()
    job = {
        **request,
        "schema_version": "background_work_job.v2",
        "job_id": "job-missing-task",
        "max_output_chars": 3000,
        "worker_payload": {
            **request["worker_payload"],
            "checkpoint": checkpoint,
        },
    }
    complete = AsyncMock(return_value={"status": "completed"})
    monkeypatch.setattr(worker, "mark_tool_result_ready", AsyncMock(
        return_value=None,
    ))
    monkeypatch.setattr(worker, "complete_background_work_job", complete)

    with pytest.raises(DatabaseOperationError, match="accepted task"):
        await worker._complete_task_orchestrator_job(
            job,
            lease_owner="worker-missing-task",
            result=snapshot,
        )

    complete.assert_not_awaited()
