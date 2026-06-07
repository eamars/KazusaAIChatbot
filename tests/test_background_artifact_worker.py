"""Tests for bounded background artifact worker prompts and payloads."""

from __future__ import annotations

import importlib
import json

from unittest.mock import ANY, AsyncMock

import pytest


def test_worker_payload_excludes_execution_and_delivery_internals() -> None:
    """Worker LLM input should contain artifact semantics, not mechanics."""

    prompts = importlib.import_module(
        "kazusa_ai_chatbot.background_artifact.prompts"
    )
    build_payload = getattr(prompts, "build_background_artifact_worker_payload")

    payload = build_payload(
        work_kind="coding_snippet",
        objective="Generate a Fibonacci function snippet.",
        input_summary="The user asked for a simple Fibonacci generator.",
        max_output_chars=3000,
    )
    serialized = json.dumps(payload, ensure_ascii=False).lower()

    assert payload["work_kind"] == "coding_snippet"
    assert payload["max_output_chars"] == 3000
    assert "fibonacci" in payload["objective"].lower()
    for forbidden in (
        "adapter",
        "source_channel_id",
        "platform_channel_id",
        "credentials",
        "mongodb",
        "lease",
        "retry",
        "shell",
        "filesystem",
        "repository path",
        ):
            assert forbidden not in serialized


@pytest.mark.asyncio
async def test_worker_threads_lease_owner_when_completing_job(
    monkeypatch,
) -> None:
    """A worker may complete only the job lease it claimed."""

    worker = importlib.import_module("kazusa_ai_chatbot.background_artifact.worker")
    claimed_job = {
        "job_id": "job-001",
        "lease_owner": "worker-001",
    }
    monkeypatch.setattr(
        worker,
        "storage_utc_now_iso",
        lambda: "2026-06-06T00:00:00+00:00",
    )
    monkeypatch.setattr(
        worker,
        "claim_background_artifact_job",
        AsyncMock(side_effect=[claimed_job, None]),
    )
    monkeypatch.setattr(
        worker,
        "_run_job",
        AsyncMock(
            return_value={
                "status": "succeeded",
                "artifact_text": "def fibonacci(n): ...",
                "failure_summary": "",
            },
        ),
    )
    complete_job = AsyncMock()
    monkeypatch.setattr(
        worker,
        "complete_background_artifact_job",
        complete_job,
    )

    result = await worker.run_background_artifact_worker_tick(
        claim_limit=1,
        worker_id="worker-001",
    )

    assert result["succeeded_count"] == 1
    complete_job.assert_awaited_once_with(
        job_id="job-001",
        lease_owner="worker-001",
        artifact_text="def fibonacci(n): ...",
        completed_at=ANY,
    )


@pytest.mark.asyncio
async def test_worker_threads_lease_owner_when_failing_job(
    monkeypatch,
) -> None:
    """A worker may fail only the job lease it claimed."""

    worker = importlib.import_module("kazusa_ai_chatbot.background_artifact.worker")
    claimed_job = {
        "job_id": "job-001",
        "lease_owner": "worker-001",
    }
    monkeypatch.setattr(
        worker,
        "storage_utc_now_iso",
        lambda: "2026-06-06T00:00:00+00:00",
    )
    monkeypatch.setattr(
        worker,
        "claim_background_artifact_job",
        AsyncMock(side_effect=[claimed_job, None]),
    )
    monkeypatch.setattr(
        worker,
        "_run_job",
        AsyncMock(
            return_value={
                "status": "failed",
                "artifact_text": "",
                "failure_summary": "Worker model returned malformed output.",
            },
        ),
    )
    fail_job = AsyncMock()
    monkeypatch.setattr(
        worker,
        "fail_background_artifact_job",
        fail_job,
    )

    result = await worker.run_background_artifact_worker_tick(
        claim_limit=1,
        worker_id="worker-001",
    )

    assert result["failed_count"] == 1
    fail_job.assert_awaited_once_with(
        job_id="job-001",
        lease_owner="worker-001",
        failure_summary="Worker model returned malformed output.",
        failed_at=ANY,
    )
