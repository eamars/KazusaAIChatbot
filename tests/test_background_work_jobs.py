"""Tests for generic background-work job persistence contracts."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, patch

import pytest


def test_background_work_public_entrypoints_exist() -> None:
    """The generic background-work package should expose runtime entrypoints."""

    module = importlib.import_module("kazusa_ai_chatbot.background_work")

    for name in (
        "BackgroundWorkQueueRequest",
        "BackgroundWorkQueueResult",
        "BackgroundWorkRuntimeHandle",
        "enqueue_background_work_request",
        "run_background_work_runtime_tick",
        "start_background_work_runtime",
        "stop_background_work_runtime",
    ):
        assert hasattr(module, name)


def test_db_background_work_job_module_exports_state_helpers() -> None:
    """The DB owner should expose named helpers for every job transition."""

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
        assert hasattr(module, name)


@pytest.mark.asyncio
async def test_enqueue_background_work_rejects_worker_local_fields() -> None:
    """Live-turn enqueue should persist task briefs, not worker internals."""

    module = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")

    with pytest.raises(ValueError, match="worker-local"):
        await module.enqueue_background_work_request({
            "action_attempt_id": "action_attempt:background-work-001",
            "idempotency_key": "background_work:test-001",
            "task_brief": "Generate a Fibonacci function snippet.",
            "source_platform": "debug",
            "source_channel_id": "debug:user:test-user",
            "source_channel_type": "private",
            "source_message_id": "message-001",
            "source_platform_bot_id": "debug-bot-001",
            "source_character_name": "Test Character",
            "requester_global_user_id": (
                "00000000-0000-4000-8000-000000000002"
            ),
            "requester_platform_user_id": "debug-user-001",
            "requester_display_name": "Test User",
            "requested_delivery": "send_result_when_done",
            "max_output_chars": 3000,
            "storage_timestamp_utc": "2026-06-06T00:00:00+00:00",
            "worker": "text_artifact",
            "work_kind": "coding_snippet",
        })


@pytest.mark.asyncio
async def test_worker_tick_records_failure_on_unhandled_exception() -> None:
    """If routing or dispatch raises, the job should be marked failed immediately."""

    worker_module = importlib.import_module(
        "kazusa_ai_chatbot.background_work.worker"
    )
    fake_job = {
        "job_id": "job-exc-001",
        "task_brief": "Generate something.",
        "max_output_chars": 3000,
        "source_context": "",
    }
    claim_mock = AsyncMock(side_effect=[fake_job, None])
    fail_mock = AsyncMock(return_value=None)
    route_mock = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    with (
        patch.object(worker_module, "claim_background_work_job", claim_mock),
        patch.object(worker_module, "fail_background_work_job", fail_mock),
        patch.object(worker_module, "route_background_work", route_mock),
    ):
        result = await worker_module.run_background_work_worker_tick(
            claim_limit=2,
            lease_seconds=60,
            max_attempts=4,
            worker_id="test-worker",
        )

    assert result["processed_count"] == 1
    assert result["failed_count"] == 1
    assert result["succeeded_count"] == 0
    fail_mock.assert_awaited_once()
    fail_kwargs = fail_mock.await_args.kwargs
    assert fail_kwargs["job_id"] == "job-exc-001"
    assert fail_kwargs["lease_owner"] == "test-worker"
    assert "RuntimeError" in fail_kwargs["failure_summary"]
    assert "LLM timeout" in fail_kwargs["failure_summary"]


def test_source_context_field_exists_in_queue_request_schema() -> None:
    """BackgroundWorkQueueRequest should accept an optional source_context."""

    models = importlib.import_module("kazusa_ai_chatbot.background_work.models")
    annotations = models.BackgroundWorkQueueRequest.__annotations__
    assert "source_context" in annotations


def test_source_context_persisted_in_job_document() -> None:
    """Job builder should persist source_context from the queue request."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    build = getattr(jobs, "_build_job_document")
    request = {
        "action_attempt_id": "action_attempt:ctx-001",
        "idempotency_key": "background_work:ctx-001",
        "task_brief": "Generate a Fibonacci function.",
        "source_context": "User asked for a code example in a math discussion.",
        "source_platform": "debug",
        "source_channel_id": "debug:user:test",
        "source_channel_type": "private",
        "source_message_id": "message-001",
        "source_platform_bot_id": "bot-001",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-001",
        "requester_platform_user_id": "debug-user-001",
        "requester_display_name": "Test User",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "storage_timestamp_utc": "2026-06-06T00:00:00+00:00",
    }
    job = build(request, job_id="job-ctx-001", storage_timestamp_utc="2026-06-06T00:00:00+00:00")
    assert job["source_context"] == "User asked for a code example in a math discussion."


def test_source_context_defaults_to_empty_when_absent() -> None:
    """Job builder should default source_context to empty when not provided."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_work.jobs")
    build = getattr(jobs, "_build_job_document")
    request = {
        "action_attempt_id": "action_attempt:ctx-002",
        "idempotency_key": "background_work:ctx-002",
        "task_brief": "Summarize this.",
        "source_platform": "debug",
        "source_channel_id": "debug:user:test",
        "source_channel_type": "private",
        "source_message_id": "message-002",
        "source_platform_bot_id": "bot-001",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-001",
        "requester_platform_user_id": "debug-user-001",
        "requester_display_name": "Test User",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "storage_timestamp_utc": "2026-06-06T00:00:00+00:00",
    }
    job = build(request, job_id="job-ctx-002", storage_timestamp_utc="2026-06-06T00:00:00+00:00")
    assert job["source_context"] == ""
