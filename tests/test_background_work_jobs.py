"""Tests for v2 task-orchestrator background-work persistence."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.test_task_resolution_inline_promotion import _request
from tests.test_task_resolution_orchestrator import (
    _context,
    _goal_continuation_ref,
)
from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.task_resolution.contracts import (
    TaskResolutionContractError,
)


def _resume_queue_request() -> dict[str, object]:
    state = importlib.import_module("kazusa_ai_chatbot.task_resolution.state")
    checkpoint = state.create_task_resolution_checkpoint(_request(), _context())
    return {
        "job_id": "job-001",
        "source_action_attempt_id": "action_attempt:task-resolution-001",
        "source_llm_trace_id": "llmtrace_source-1",
        "idempotency_key": "background_work:task-resolution-001",
        "accepted_task_id": "task-001",
        "task_identity_key": "accepted_task:v2:abc",
        "semantic_objective": "Resolve one bounded public question.",
        "goal_continuation_ref": _goal_continuation_ref(),
        "requested_worker": "task_orchestrator",
        "worker_payload": {
            "schema_version": "task_orchestrator_worker_payload.v1",
            "operation": "resume_task_resolution",
            "checkpoint": checkpoint,
            "coding_request": None,
        },
        "task_execution_context": _context(),
        "source_platform": "debug",
        "source_channel_id": "debug:user:test-user",
        "source_channel_type": "private",
        "source_message_id": "message-001",
        "source_platform_bot_id": "debug-bot-001",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-001",
        "requester_platform_user_id": "debug-user-001",
        "requester_display_name": "Test User",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "storage_timestamp_utc": "2026-06-06T00:00:00+00:00",
    }


def test_background_work_public_entrypoints_are_v2_only() -> None:
    """The package exports the canonical task worker and queue contracts."""

    module = importlib.import_module("kazusa_ai_chatbot.background_work")

    for name in (
        "BackgroundWorkQueueRequest",
        "BackgroundWorkQueueResult",
        "TaskOrchestratorWorkerPayloadV1",
        "TASK_ORCHESTRATOR_WORKER",
        "FUTURE_SPEAK_WORKER",
        "BackgroundWorkRuntimeHandle",
        "enqueue_background_work_request",
        "run_background_work_runtime_tick",
        "start_background_work_runtime",
        "stop_background_work_runtime",
    ):
        assert hasattr(module, name), name


def test_db_background_work_job_module_exports_state_helpers() -> None:
    """The DB owner exposes named helpers for every v2 job transition."""

    module = importlib.import_module("kazusa_ai_chatbot.db.background_work_jobs")

    for name in (
        "ensure_background_work_job_indexes",
        "insert_background_work_job",
        "claim_background_work_job",
        "complete_background_work_job",
        "fail_background_work_job",
        "find_deliverable_background_work_jobs",
        "mark_background_work_delivery_in_progress",
        "mark_background_work_delivered",
        "mark_background_work_delivery_failed",
    ):
        assert hasattr(module, name), name


@pytest.mark.asyncio
async def test_enqueue_persists_one_v2_task_orchestrator_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validated checkpoint and context persist without route rationale."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    stored_jobs: list[dict[str, object]] = []

    async def insert_job(job: dict[str, object]) -> dict[str, object]:
        stored_jobs.append(job)
        return job

    monkeypatch.setattr(jobs, "insert_background_work_job", insert_job)

    result = await jobs.enqueue_background_work_request(_resume_queue_request())

    assert result["status"] == "pending"
    assert len(stored_jobs) == 1
    job = stored_jobs[0]
    assert job["schema_version"] == "background_work_job.v2"
    assert job["requested_worker"] == "task_orchestrator"
    assert job["source_llm_trace_id"] == "llmtrace_source-1"
    assert job["semantic_objective"] == (
        "Resolve one bounded public question."
    )
    assert job["worker_payload"]["checkpoint"]["dispatch_count"] == 0
    assert "task_brief" not in job
    assert "source_context" not in job


@pytest.mark.asyncio
async def test_enqueue_rejects_removed_direct_worker() -> None:
    """Direct generic coding and text workers are outside the v2 roster."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    request = _resume_queue_request()
    request["requested_worker"] = "coding_agent"

    with pytest.raises(ValueError, match="requested_worker"):
        await jobs.enqueue_background_work_request(request)


def test_resume_payload_rejects_v1_checkpoint() -> None:
    """The big-bang queue boundary rejects every v1 checkpoint shape."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    payload = dict(_resume_queue_request()["worker_payload"])
    checkpoint = dict(payload["checkpoint"])
    checkpoint["schema_version"] = "task_resolution_checkpoint.v0"
    payload["checkpoint"] = checkpoint

    with pytest.raises(TaskResolutionContractError, match="schema_version"):
        jobs.validate_task_orchestrator_worker_payload(payload)


def test_task_orchestrator_payload_requires_exact_operation_union() -> None:
    """Resume and bound coding payload branches remain mutually exclusive."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    payload = dict(_resume_queue_request()["worker_payload"])
    payload["coding_request"] = {
        "workspace_root": "workspace",
        "run_id": "run-1",
        "action": "status",
    }

    with pytest.raises(ValueError, match="coding_request"):
        jobs.validate_task_orchestrator_worker_payload(payload)


@pytest.mark.parametrize(
    "coding_request,match",
    [
        ({"workspace_root": "workspace", "action": "status"}, "run_id"),
        ({"run_id": "run-1", "action": "status"}, "workspace_root"),
        (
            {
                "workspace_root": "workspace",
                "run_id": "run-1",
                "action": "shell_command",
            },
            "action",
        ),
        (
            {
                "workspace_root": "workspace",
                "run_id": "run-1",
                "action": "approve_and_verify",
            },
            "approval",
        ),
    ],
)
def test_bound_coding_payload_validates_frozen_closed_request(
    coding_request: dict[str, object],
    match: str,
) -> None:
    """Bound continuation admits only complete reviewed public operations."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    payload = {
        "schema_version": "task_orchestrator_worker_payload.v1",
        "operation": "continue_bound_coding_run",
        "checkpoint": None,
        "coding_request": coding_request,
    }

    with pytest.raises(ValueError, match=match):
        jobs.validate_task_orchestrator_worker_payload(payload)


def test_future_speak_payload_remains_deterministic() -> None:
    """The retained scheduler worker accepts only its exact closed payload."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    request = _resume_queue_request()
    request["requested_worker"] = "future_speak"
    request["worker_payload"] = {
        "trigger_at": "2026-08-02T03:00:00+00:00",
        "continuation_objective": "Deliver the scheduled reminder.",
    }
    request.pop("task_execution_context")
    request["goal_continuation_ref"] = None
    jobs._validate_queue_request(request)

    invalid = dict(request)
    invalid["worker_payload"] = {
        **request["worker_payload"],
        "requested_worker": "task_orchestrator",
    }
    with pytest.raises(ValueError, match="fields"):
        jobs._validate_queue_request(invalid)


def test_job_document_keeps_accepted_task_audit_identity() -> None:
    """Internal queue rows retain v2 task linkage and idempotency material."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    request = _resume_queue_request()
    job = jobs._build_job_document(
        request,
        job_id="job-accepted-task-001",
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
    )

    assert job["accepted_task_id"] == "task-001"
    assert job["task_identity_key"] == "accepted_task:v2:abc"
    assert job["idempotency_key"] == "background_work:task-resolution-001"
    assert job["source_llm_trace_id"] == "llmtrace_source-1"
    assert job["attempt_count"] == 0


def test_job_document_keeps_goal_continuation_ref() -> None:
    """The queued task keeps the exact continuation identity."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    request = _resume_queue_request()
    job = jobs._build_job_document(
        request,
        job_id="job-continuation-001",
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
    )

    assert job["goal_continuation_ref"] == request["goal_continuation_ref"]


def test_background_action_handler_copies_continuation_ref_to_both_requests() -> None:
    """Action-owned task and queue requests retain the same typed ref."""

    handler = importlib.import_module(
        "kazusa_ai_chatbot.action_spec.handlers.background_work"
    )
    continuation_ref = _goal_continuation_ref()
    scope = {
        "source_trigger_source": "user_message",
        "source_platform": "debug",
        "source_channel_id": "debug:user:test-user",
        "source_channel_type": "private",
        "source_message_id": "message-001",
        "source_platform_bot_id": "debug-bot-001",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-001",
        "requester_platform_user_id": "debug-user-001",
        "requester_display_name": "Test User",
    }
    validated = {
        "params": {
            "max_output_chars": 3000,
            "task_execution_context": _context(),
        },
        "target": {"scope": scope},
        "goal_continuation_ref": continuation_ref,
    }
    accepted_request = handler._accepted_task_create_request(
        validated,
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
        task_kind="coding_continuation",
        semantic_objective="Resolve one bounded public question.",
        accepted_task_summary="Resolve one bounded public question.",
    )
    queue_request = handler._queue_request_from_accepted_task(
        validated,
        {
            "accepted_task_id": "task-001",
            "task_identity_key": "accepted_task:v2:abc",
        },
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
        action_attempt_id="action_attempt:task-resolution-001",
        source_llm_trace_id="llmtrace_source-1",
        semantic_objective="Resolve one bounded public question.",
        requested_worker="task_orchestrator",
        worker_payload=_resume_queue_request()["worker_payload"],
    )

    assert accepted_request["goal_continuation_ref"] == continuation_ref
    assert queue_request["goal_continuation_ref"] == continuation_ref


@pytest.mark.asyncio
async def test_enqueue_rejects_task_job_without_goal_continuation_ref() -> None:
    """A task-resolution queue row cannot omit its goal continuation ref."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    request = _resume_queue_request()
    request["goal_continuation_ref"] = None

    with pytest.raises(ValueError, match="goal_continuation_ref"):
        await jobs.enqueue_background_work_request(request)


@pytest.mark.asyncio
async def test_insert_background_work_job_records_source_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idempotent duplicate preserves the original source trace."""

    db_module = importlib.import_module(
        "kazusa_ai_chatbot.db.background_work_jobs"
    )
    collection = MagicMock()
    collection.insert_one = AsyncMock(
        side_effect=db_module.DuplicateKeyError("duplicate")
    )
    collection.find_one = AsyncMock(return_value={
        "schema_version": "background_work_job.v2",
        "idempotency_key": "background_work:conflict",
        "job_id": "job-original",
        "source_llm_trace_id": "llmtrace-original",
        "goal_continuation_ref": _goal_continuation_ref(),
    })
    collection.update_one = AsyncMock()
    db = MagicMock()
    db.__getitem__.return_value = collection
    monkeypatch.setattr(db_module, "get_db", AsyncMock(return_value=db))

    jobs_module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.jobs"
    )
    job = jobs_module._build_job_document(
        _resume_queue_request(),
        job_id="job-incoming",
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
    )
    job["idempotency_key"] = "background_work:conflict"
    job["source_llm_trace_id"] = "llmtrace-incoming"

    result = await db_module.insert_background_work_job(job)

    assert result["source_llm_trace_id"] == "llmtrace-original"
    assert result["correlation_write_status"] == "conflict"
    assert result["correlation_conflict_source_llm_trace_id"] == (
        "llmtrace-incoming"
    )
    collection.update_one.assert_awaited_once_with(
        {
            "schema_version": db_module.BACKGROUND_WORK_JOB_SCHEMA_VERSION,
            "idempotency_key": "background_work:conflict",
            "source_llm_trace_id": "llmtrace-original",
        },
        {
            "$set": {
                "correlation_write_status": "conflict",
                "correlation_conflict_source_llm_trace_id": (
                    "llmtrace-incoming"
                ),
            },
        },
    )


@pytest.mark.asyncio
async def test_insert_background_work_job_rejects_continuation_ref_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idempotency collision cannot cross continuation lineages."""

    db_module = importlib.import_module(
        "kazusa_ai_chatbot.db.background_work_jobs"
    )
    collection = MagicMock()
    collection.insert_one = AsyncMock(
        side_effect=db_module.DuplicateKeyError("duplicate")
    )
    stored_ref = build_goal_continuation_ref(
        source_episode_id="stored-job-episode",
        source_message_id="stored-job-message",
        branch_id="ordinary_response",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "stored-job-goal",
        },
    )
    collection.find_one = AsyncMock(return_value={
        "schema_version": "background_work_job.v2",
        "idempotency_key": "background_work:ref-mismatch",
        "job_id": "job-original",
        "source_llm_trace_id": "llmtrace-source-1",
        "goal_continuation_ref": stored_ref,
    })
    collection.update_one = AsyncMock()
    db = MagicMock()
    db.__getitem__.return_value = collection
    monkeypatch.setattr(db_module, "get_db", AsyncMock(return_value=db))

    jobs_module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.jobs"
    )
    job = jobs_module._build_job_document(
        _resume_queue_request(),
        job_id="job-incoming",
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
    )
    job["idempotency_key"] = "background_work:ref-mismatch"

    with pytest.raises(DatabaseOperationError, match="goal_continuation_ref"):
        await db_module.insert_background_work_job(job)

    collection.update_one.assert_not_awaited()
