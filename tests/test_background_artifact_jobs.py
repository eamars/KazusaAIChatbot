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
