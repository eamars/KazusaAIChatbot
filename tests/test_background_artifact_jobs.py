"""Tests for background artifact job persistence contracts."""

from __future__ import annotations

import importlib

from unittest.mock import AsyncMock

import pytest


def test_db_background_artifact_job_module_exports_state_helpers() -> None:
    """The DB owner should expose named helpers for every job transition."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.db.background_artifact_jobs"
    )

    for name in (
        "ensure_background_artifact_job_indexes",
        "insert_background_artifact_job",
        "claim_background_artifact_job",
        "complete_background_artifact_job",
        "fail_background_artifact_job",
        "find_deliverable_background_artifact_jobs",
        "mark_background_artifact_delivery_in_progress",
        "mark_background_artifact_delivered",
        "mark_background_artifact_delivery_failed",
    ):
        assert hasattr(module, name)


def test_delivery_failure_summary_field_exists_in_job_schema() -> None:
    """Legacy job docs should separate delivery failures from worker failures."""

    models = importlib.import_module("kazusa_ai_chatbot.background_artifact.models")
    annotations = models.BackgroundArtifactJobDoc.__annotations__

    assert "delivery_failure_summary" in annotations


def test_delivery_failure_summary_initialized_empty() -> None:
    """New legacy jobs should start with empty delivery failure text."""

    jobs = importlib.import_module("kazusa_ai_chatbot.background_artifact.jobs")
    build = getattr(jobs, "_build_job_document")

    job = build(
        {
            "action_attempt_id": "attempt-001",
            "idempotency_key": "idem-001",
            "work_kind": "coding_snippet",
            "objective": "Generate Fibonacci code.",
            "input_summary": "The user asked for Fibonacci code.",
            "requested_delivery": "send_result_when_done",
            "max_output_chars": 200,
            "source_platform": "debug",
            "source_channel_id": "debug-private-1",
            "source_channel_type": "private",
            "source_message_id": "message-1",
            "source_platform_bot_id": "bot-1",
            "source_character_name": "Test Character",
            "requester_global_user_id": "global-user-1",
            "requester_platform_user_id": "platform-user-1",
            "requester_display_name": "Test User",
            "storage_timestamp_utc": "2026-06-06T00:00:00+00:00",
        },
        job_id="job-001",
        storage_timestamp_utc="2026-06-06T00:00:00+00:00",
    )

    assert job["delivery_failure_summary"] == ""


@pytest.mark.asyncio
async def test_delivery_in_progress_claim_requires_ready_delivery_state(
    monkeypatch,
) -> None:
    """Delivery claims must not reopen already delivered or active rows."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.db.background_artifact_jobs"
    )

    class _Collection:
        def __init__(self) -> None:
            self.filter = None

        async def find_one_and_update(self, filter, update, **kwargs):
            del update, kwargs
            self.filter = filter
            return None

    collection = _Collection()

    class _Db:
        def __getitem__(self, name):
            assert name == "background_artifact_jobs"
            return collection

    monkeypatch.setattr(module, "get_db", AsyncMock(return_value=_Db()))

    await module.mark_background_artifact_delivery_in_progress(
        job_id="job-001",
        delivery_tracking_id="tracking-001",
        started_at="2026-06-06T00:00:00+00:00",
    )

    assert collection.filter == {
        "job_id": "job-001",
        "status": {"$in": ["completed", "failed", "delivery_failed"]},
        "delivery_state": {"$in": ["ready", "failed"]},
    }


@pytest.mark.asyncio
async def test_deliverable_query_uses_delivery_attempt_cap(monkeypatch) -> None:
    """Deliverable legacy jobs should stop reappearing after retry exhaustion."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.db.background_artifact_jobs"
    )

    class _Cursor:
        def sort(self, *args, **kwargs):
            del args, kwargs
            return self

        def limit(self, value):
            del value
            return self

        async def to_list(self, *, length):
            del length
            return []

    class _Collection:
        def __init__(self) -> None:
            self.filter = None

        def find(self, filter, *args, **kwargs):
            del args, kwargs
            self.filter = filter
            return _Cursor()

    collection = _Collection()

    class _Db:
        def __getitem__(self, name):
            assert name == "background_artifact_jobs"
            return collection

    monkeypatch.setattr(module, "get_db", AsyncMock(return_value=_Db()))

    await module.find_deliverable_background_artifact_jobs(
        limit=3,
        max_delivery_attempts=5,
    )

    assert collection.filter == {
        "status": {"$in": ["completed", "failed", "delivery_failed"]},
        "delivery_state": {"$in": ["ready", "failed"]},
        "$or": [
            {"delivery_attempt_count": {"$exists": False}},
            {"delivery_attempt_count": {"$lt": 5}},
        ],
    }


@pytest.mark.asyncio
async def test_delivery_failure_does_not_overwrite_worker_failure(
    monkeypatch,
) -> None:
    """Delivery failure text should not replace the artifact failure summary."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.db.background_artifact_jobs"
    )
    update_calls = []

    async def _fake_update_job(*, job_id, update):
        update_calls.append({"job_id": job_id, "update": update})
        return None

    monkeypatch.setattr(module, "_update_job", _fake_update_job)

    await module.mark_background_artifact_delivery_failed(
        job_id="job-001",
        failure_summary="Adapter callback failed.",
        failed_at="2026-06-06T00:00:00+00:00",
    )

    update = update_calls[0]["update"]
    set_doc = update["$set"]
    assert set_doc["delivery_failure_summary"] == "Adapter callback failed."
    assert "failure_summary" not in set_doc


@pytest.mark.asyncio
async def test_worker_completion_requires_current_lease_owner(
    monkeypatch,
) -> None:
    """Completion must not update stale or stolen worker leases."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.db.background_artifact_jobs"
    )

    class _Collection:
        def __init__(self) -> None:
            self.filter = None

        async def find_one_and_update(self, filter, update, **kwargs):
            del update, kwargs
            self.filter = filter
            return None

    collection = _Collection()

    class _Db:
        def __getitem__(self, name):
            assert name == "background_artifact_jobs"
            return collection

    monkeypatch.setattr(module, "get_db", AsyncMock(return_value=_Db()))

    await module.complete_background_artifact_job(
        job_id="job-001",
        lease_owner="worker-001",
        artifact_text="def fibonacci(n): ...",
        completed_at="2026-06-06T00:00:00+00:00",
    )

    assert collection.filter == {
        "job_id": "job-001",
        "status": "in_progress",
        "lease_owner": "worker-001",
    }


@pytest.mark.asyncio
async def test_worker_failure_requires_current_lease_owner(
    monkeypatch,
) -> None:
    """Failure must not update stale or stolen worker leases."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.db.background_artifact_jobs"
    )

    class _Collection:
        def __init__(self) -> None:
            self.filter = None

        async def find_one_and_update(self, filter, update, **kwargs):
            del update, kwargs
            self.filter = filter
            return None

    collection = _Collection()

    class _Db:
        def __getitem__(self, name):
            assert name == "background_artifact_jobs"
            return collection

    monkeypatch.setattr(module, "get_db", AsyncMock(return_value=_Db()))

    await module.fail_background_artifact_job(
        job_id="job-001",
        lease_owner="worker-001",
        failure_summary="Worker model returned malformed output.",
        failed_at="2026-06-06T00:00:00+00:00",
    )

    assert collection.filter == {
        "job_id": "job-001",
        "status": "in_progress",
        "lease_owner": "worker-001",
    }
