"""Contract tests for calendar worker claim and handler orchestration."""

from __future__ import annotations

from typing import Any

import pytest


NOW_UTC = "2026-06-04T00:15:00+00:00"


class _RepositoryDouble:
    def __init__(self, runs: list[dict[str, Any]]) -> None:
        self.runs = list(runs)
        self.succeeded: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []

    async def claim_due_calendar_runs(
        self,
        *,
        current_timestamp_utc: str,
        lease_owner: str,
        lease_duration_seconds: int,
        limit: int,
        max_attempts: int,
        trigger_kinds: list[str],
    ) -> list[dict[str, Any]]:
        assert current_timestamp_utc == NOW_UTC
        assert lease_owner == "worker-a"
        assert lease_duration_seconds == 300
        assert limit == 2
        assert max_attempts == 3
        assert trigger_kinds
        return self.runs

    async def mark_calendar_run_completed(
        self,
        run_id: str,
        *,
        lease_owner: str,
        storage_timestamp_utc: str,
        result: dict[str, Any],
    ) -> bool:
        self.succeeded.append({
            "run_id": run_id,
            "lease_owner": lease_owner,
            "storage_timestamp_utc": storage_timestamp_utc,
            "result": result,
        })
        return True

    async def mark_calendar_run_failed(
        self,
        run_id: str,
        *,
        lease_owner: str,
        storage_timestamp_utc: str,
        error: str,
        retryable: bool,
    ) -> bool:
        self.failed.append({
            "run_id": run_id,
            "lease_owner": lease_owner,
            "storage_timestamp_utc": storage_timestamp_utc,
            "error": error,
            "retryable": retryable,
        })
        return True

    async def mark_calendar_run_skipped(
        self,
        run_id: str,
        *,
        storage_timestamp_utc: str,
        reason: str,
    ) -> bool:
        self.skipped.append({
            "run_id": run_id,
            "storage_timestamp_utc": storage_timestamp_utc,
            "reason": reason,
        })
        return True


@pytest.mark.asyncio
async def test_worker_tick_dispatches_claimed_runs_to_typed_handler() -> None:
    """Claimed due runs should be completed only after their handler returns."""

    from kazusa_ai_chatbot.calendar_scheduler import models, worker

    repository = _RepositoryDouble(
        [
            {
                "run_id": "run-1",
                "trigger_kind": models.TRIGGER_FUTURE_COGNITION,
                "payload": {"episode_type": "self_cognition"},
            }
        ]
    )
    handled_runs: list[dict[str, Any]] = []

    async def handle_future_cognition(run: dict[str, Any]) -> dict[str, Any]:
        handled_runs.append(run)
        return {"status": "case_created", "case_id": "case-1"}

    registry = worker.CalendarRunHandlerRegistry()
    registry.register(models.TRIGGER_FUTURE_COGNITION, handle_future_cognition)

    result = await worker.run_calendar_worker_tick(
        current_timestamp_utc=NOW_UTC,
        repository=repository,
        handler_registry=registry,
        lease_owner="worker-a",
        lease_duration_seconds=300,
        claim_limit=2,
        max_attempts=3,
    )

    assert result == {
        "claimed_count": 1,
        "completed_count": 1,
        "failed_count": 0,
        "skipped_count": 0,
    }
    assert handled_runs == repository.runs
    assert repository.succeeded == [
        {
            "run_id": "run-1",
            "lease_owner": "worker-a",
            "storage_timestamp_utc": NOW_UTC,
            "result": {"status": "case_created", "case_id": "case-1"},
        }
    ]
    assert repository.failed == []
    assert repository.skipped == []


@pytest.mark.asyncio
async def test_worker_tick_fails_unknown_trigger_without_running_code() -> None:
    """Unknown triggers should fail closed instead of invoking ad hoc jobs."""

    from kazusa_ai_chatbot.calendar_scheduler import worker

    repository = _RepositoryDouble(
        [
            {
                "run_id": "run-unknown",
                "trigger_kind": "send_message",
                "payload": {"text": "visible delayed text"},
            }
        ]
    )
    registry = worker.CalendarRunHandlerRegistry()

    result = await worker.run_calendar_worker_tick(
        current_timestamp_utc=NOW_UTC,
        repository=repository,
        handler_registry=registry,
        lease_owner="worker-a",
        lease_duration_seconds=300,
        claim_limit=2,
        max_attempts=3,
    )

    assert result == {
        "claimed_count": 1,
        "completed_count": 0,
        "failed_count": 1,
        "skipped_count": 0,
    }
    assert repository.succeeded == []
    assert repository.failed == [
        {
            "run_id": "run-unknown",
            "lease_owner": "worker-a",
            "storage_timestamp_utc": NOW_UTC,
            "error": "unsupported calendar trigger kind: send_message",
            "retryable": False,
        }
    ]
