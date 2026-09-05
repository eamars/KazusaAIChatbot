"""Retained deterministic coverage for the V2 background-work boundary."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref
from kazusa_ai_chatbot.cognition_shared.contracts import (
    build_scheduled_future_speech_authority,
)
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from tests.task_resolution_test_helpers import (
    _goal_continuation_ref,
    resume_queue_request,
)


def _future_speak_queue_request() -> dict[str, object]:
    """Build one retained future-speak request with its typed authority."""

    authority = build_scheduled_future_speech_authority(
        source_episode_id="episode-future-speak-001",
        source_message_id="message-1",
        source_action_attempt_id="action_attempt:future-speak-001",
        source_llm_trace_id="llmtrace_source-1",
        accepted_at_utc="2026-06-06T00:00:00+00:00",
        timezone="Pacific/Auckland",
        trigger_local="2026-06-06 13:00",
        platform="debug",
        channel_type="private",
        audience_kind="private",
        semantic_objective="Deliver the scheduled reminder.",
        authorized_content_summary="The scheduled reminder is due.",
        authorized_detail_refs=[{
            "evidence_handle": "e1",
            "semantic_summary": "The current conversation requested a reminder.",
            "provenance_role": "current_event",
        }],
    )
    request = resume_queue_request()
    request["requested_worker"] = "future_speak"
    request["worker_payload"] = {
        "trigger_at": "2026-06-06 13:00",
        "continuation_objective": "Deliver the scheduled reminder.",
    }
    request.pop("task_execution_context")
    request["goal_continuation_ref"] = None
    request["source_action_attempt_id"] = "action_attempt:future-speak-001"
    request["scheduled_future_speech_authority"] = dict(authority)
    return request


def test_background_work_public_entrypoints_are_v2_only() -> None:
    """The package exposes the canonical task worker and queue contracts."""

    module = importlib.import_module("kazusa_ai_chatbot.background_work")
    for name in (
        "BackgroundWorkQueueRequest",
        "BackgroundWorkQueueResult",
        "TaskOrchestratorWorkerPayloadV2",
        "TASK_ORCHESTRATOR_WORKER",
        "FUTURE_SPEAK_WORKER",
        "BackgroundWorkRuntimeHandle",
        "enqueue_background_work_request",
        "run_background_work_runtime_tick",
        "start_background_work_runtime",
        "stop_background_work_runtime",
    ):
        assert hasattr(module, name), name


def test_db_background_work_job_module_exports_transitions() -> None:
    """The DB owner exposes named helpers for each durable job transition."""

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
async def test_enqueue_persists_one_generation_zero_dsh_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid start request persists exact V2 payload and trusted context."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    stored: list[dict[str, object]] = []

    async def insert(job: dict[str, object]) -> dict[str, object]:
        stored.append(job)
        return job

    monkeypatch.setattr(jobs, "insert_background_work_job", insert)
    result = await jobs.enqueue_background_work_request(resume_queue_request())

    assert result["status"] == "pending"
    assert len(stored) == 1
    job = stored[0]
    assert job["schema_version"] == "background_work_job.v2"
    assert job["requested_worker"] == "task_orchestrator"
    assert job["worker_payload"]["operation"] == "open_dsh_resolution"
    assert job["worker_payload"]["operation_generation"] == 0
    assert "authority" not in job["worker_payload"]


@pytest.mark.asyncio
async def test_enqueue_rejects_unsupported_worker() -> None:
    """Only DSH task orchestration and future-speak remain queue workers."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    request = resume_queue_request()
    request["requested_worker"] = "unsupported_worker"
    with pytest.raises(ValueError, match="requested_worker"):
        await jobs.enqueue_background_work_request(request)


def test_worker_payload_rejects_unknown_operations() -> None:
    """The payload validator closes the operation vocabulary."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    payload = dict(resume_queue_request()["worker_payload"])
    payload["operation"] = "unsupported"
    with pytest.raises(ValueError, match="operation"):
        jobs.validate_task_orchestrator_worker_payload(payload)


def test_future_speak_payload_remains_deterministic() -> None:
    """The retained scheduler worker accepts its exact closed payload."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    request = _future_speak_queue_request()
    jobs._validate_queue_request(request)
    invalid = dict(request)
    invalid["worker_payload"] = {
        "trigger_at": "2026-06-06 13:00",
        "unexpected": True,
    }
    with pytest.raises(ValueError, match="fields"):
        jobs._validate_queue_request(invalid)


def test_future_speak_job_preserves_authority_identity() -> None:
    """Serialization keeps the immutable future-speak authority unchanged."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    request = _future_speak_queue_request()
    job = jobs._build_job_document(
        request,
        job_id="job-future-speak-001",
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
    )
    assert job["scheduled_future_speech_authority"] == (
        request["scheduled_future_speech_authority"]
    )
    assert job["source_message_id"] == "message-1"


def test_job_document_keeps_task_lineage() -> None:
    """Durable queue rows retain task identity, trace, and continuation ref."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    request = resume_queue_request()
    job = jobs._build_job_document(
        request,
        job_id="job-task-001",
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
    )
    assert job["accepted_task_id"] == "task-001"
    assert job["task_identity_key"] == "accepted_task:v2:abc"
    assert job["goal_continuation_ref"] == request["goal_continuation_ref"]
    assert job["source_llm_trace_id"] == "llmtrace_source-1"


@pytest.mark.asyncio
async def test_enqueue_rejects_task_without_goal_continuation_ref() -> None:
    """A DSH task cannot be queued without its exact goal lineage."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    request = resume_queue_request()
    request["goal_continuation_ref"] = None
    with pytest.raises(ValueError, match="goal_continuation_ref"):
        await jobs.enqueue_background_work_request(request)


@pytest.mark.asyncio
async def test_insert_job_records_source_trace_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idempotent duplicate preserves the original source trace."""

    db_module = importlib.import_module("kazusa_ai_chatbot.db.background_work_jobs")
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
    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    job = jobs._build_job_document(
        resume_queue_request(),
        job_id="job-incoming",
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
    )
    job["idempotency_key"] = "background_work:conflict"
    job["source_llm_trace_id"] = "llmtrace-incoming"

    result = await db_module.insert_background_work_job(job)
    assert result["source_llm_trace_id"] == "llmtrace-original"
    assert result["correlation_write_status"] == "conflict"
    collection.update_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_insert_job_rejects_continuation_ref_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idempotency collision cannot cross goal-continuation lineages."""

    db_module = importlib.import_module("kazusa_ai_chatbot.db.background_work_jobs")
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
    collection = MagicMock()
    collection.insert_one = AsyncMock(
        side_effect=db_module.DuplicateKeyError("duplicate")
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
    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    job = jobs._build_job_document(
        resume_queue_request(),
        job_id="job-incoming",
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
    )
    job["idempotency_key"] = "background_work:ref-mismatch"
    with pytest.raises(DatabaseOperationError, match="goal_continuation_ref"):
        await db_module.insert_background_work_job(job)
    collection.update_one.assert_not_awaited()
