"""Tests for generic background-work job persistence contracts."""

from __future__ import annotations

import importlib

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
