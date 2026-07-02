"""Contract tests for calendar worker claim and handler orchestration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest


NOW_UTC = "2026-06-04T00:15:00+00:00"


class _RepositoryDouble:
    def __init__(self, runs: list[dict[str, Any]]) -> None:
        self.runs = list(runs)
        self.succeeded: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []
        self.deferred: list[dict[str, Any]] = []

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
        lease_owner: str,
        storage_timestamp_utc: str,
        reason: str,
    ) -> bool:
        self.skipped.append({
            "run_id": run_id,
            "lease_owner": lease_owner,
            "storage_timestamp_utc": storage_timestamp_utc,
            "reason": reason,
        })
        return True

    async def mark_calendar_run_deferred(
        self,
        run_id: str,
        *,
        lease_owner: str,
        storage_timestamp_utc: str,
        reason: str,
    ) -> bool:
        self.deferred.append({
            "run_id": run_id,
            "lease_owner": lease_owner,
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
        "deferred_count": 0,
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
        "deferred_count": 0,
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


@pytest.mark.asyncio
async def test_worker_tick_marks_skipped_handler_result_skipped() -> None:
    """Skipped handler results should not complete the run."""

    from kazusa_ai_chatbot.calendar_scheduler import models, worker

    repository = _RepositoryDouble(
        [
            {
                "run_id": "run-skipped",
                "trigger_kind": models.TRIGGER_COMMITMENT_DUE_COGNITION,
                "payload": {"unit_id": "commitment-1"},
            }
        ]
    )

    async def handle_commitment_due(run: dict[str, Any]) -> dict[str, Any]:
        assert run["run_id"] == "run-skipped"
        return {
            "status": "skipped",
            "reason": "stale_active_commitment_due_at",
        }

    registry = worker.CalendarRunHandlerRegistry()
    registry.register(
        models.TRIGGER_COMMITMENT_DUE_COGNITION,
        handle_commitment_due,
    )

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
        "failed_count": 0,
        "skipped_count": 1,
        "deferred_count": 0,
    }
    assert repository.succeeded == []
    assert repository.skipped == [
        {
            "run_id": "run-skipped",
            "lease_owner": "worker-a",
            "storage_timestamp_utc": NOW_UTC,
            "reason": "stale_active_commitment_due_at",
        }
    ]


@pytest.mark.asyncio
async def test_worker_tick_requeues_deferred_handler_result() -> None:
    """Deferred handler results should requeue instead of completing runs."""

    from kazusa_ai_chatbot.calendar_scheduler import models, worker

    repository = _RepositoryDouble(
        [
            {
                "run_id": "run-deferred",
                "trigger_kind": models.TRIGGER_REFLECTION_PHASE_SLOT,
                "payload": {"reflection_phase_intent": {}},
            }
        ]
    )

    async def handle_reflection_phase(run: dict[str, Any]) -> dict[str, Any]:
        assert run["run_id"] == "run-deferred"
        return {
            "status": "deferred",
            "defer_reason": "same_scope_foreground_active",
        }

    registry = worker.CalendarRunHandlerRegistry()
    registry.register(
        models.TRIGGER_REFLECTION_PHASE_SLOT,
        handle_reflection_phase,
    )

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
        "failed_count": 0,
        "skipped_count": 0,
        "deferred_count": 1,
    }
    assert repository.succeeded == []
    assert repository.failed == []
    assert repository.skipped == []
    assert repository.deferred == [
        {
            "run_id": "run-deferred",
            "lease_owner": "worker-a",
            "storage_timestamp_utc": NOW_UTC,
            "reason": "same_scope_foreground_active",
        }
    ]


@pytest.mark.asyncio
async def test_worker_tick_marks_handler_exception_failed() -> None:
    """Handler crashes should leave terminal, non-retryable failed runs."""

    from kazusa_ai_chatbot.calendar_scheduler import models, worker

    repository = _RepositoryDouble(
        [
            {
                "run_id": "run-crashed",
                "trigger_kind": models.TRIGGER_REFLECTION_PHASE_SLOT,
                "attempt_count": 1,
                "max_attempts": 3,
                "payload": {"reflection_phase_intent": {}},
            }
        ]
    )

    async def handle_reflection_phase(run: dict[str, Any]) -> dict[str, Any]:
        assert run["run_id"] == "run-crashed"
        raise RuntimeError("phase handler crashed")

    registry = worker.CalendarRunHandlerRegistry()
    registry.register(
        models.TRIGGER_REFLECTION_PHASE_SLOT,
        handle_reflection_phase,
    )

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
        "deferred_count": 0,
    }
    assert repository.succeeded == []
    assert repository.skipped == []
    assert len(repository.failed) == 1
    failure = repository.failed[0]
    assert failure["run_id"] == "run-crashed"
    assert failure["lease_owner"] == "worker-a"
    assert failure["storage_timestamp_utc"] == NOW_UTC
    assert failure["retryable"] is False
    assert "phase handler crashed" in failure["error"]


@pytest.mark.asyncio
async def test_calendar_worker_loop_materializes_reflection_period_before_claims() -> None:
    """The service-owned worker loop should materialize phase runs before claims."""

    from kazusa_ai_chatbot.calendar_scheduler import models, worker

    now = datetime(2026, 6, 4, 0, 15, tzinfo=timezone.utc)
    repository = _RepositoryDouble([])
    calls: list[tuple[str, str]] = []

    async def materialize_reflection_phase_period(
        *,
        period_start_utc: datetime,
        storage_timestamp_utc: str,
        repository: object,
    ) -> dict[str, object]:
        assert repository is repository_double
        calls.append(("materialize", period_start_utc.isoformat()))
        return {"materialized_count": 0, "run_ids": []}

    async def run_worker_tick(**kwargs: Any) -> dict[str, int]:
        assert kwargs["repository"] is repository_double
        assert kwargs["current_timestamp_utc"] == NOW_UTC
        assert kwargs["lease_owner"] == "calendar-worker-test"
        assert kwargs["lease_duration_seconds"] == 300
        assert kwargs["claim_limit"] == 2
        assert kwargs["max_attempts"] == 3
        handler_registry = kwargs["handler_registry"]
        assert handler_registry.trigger_kinds() == [
            models.TRIGGER_REFLECTION_PHASE_SLOT,
        ]
        calls.append(("tick", kwargs["current_timestamp_utc"]))
        return {
            "claimed_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "deferred_count": 0,
        }

    repository_double = repository
    registry = worker.CalendarRunHandlerRegistry()

    async def handle_reflection_phase(run: dict[str, Any]) -> dict[str, Any]:
        assert run["trigger_kind"] == models.TRIGGER_REFLECTION_PHASE_SLOT
        result = {"status": "completed"}
        return result

    registry.register(
        models.TRIGGER_REFLECTION_PHASE_SLOT,
        handle_reflection_phase,
    )

    handle = worker.start_calendar_scheduler_worker(
        repository=repository_double,
        handler_registry=registry,
        poll_interval_seconds=3600,
        lease_owner="calendar-worker-test",
        lease_duration_seconds=300,
        claim_limit=2,
        max_attempts=3,
        now_func=lambda: now,
        materialize_reflection_phase_period_func=(
            materialize_reflection_phase_period
        ),
        run_worker_tick_func=run_worker_tick,
    )
    try:
        for _ in range(20):
            if len(calls) >= 2:
                break
            await asyncio.sleep(0.01)
    finally:
        await worker.stop_calendar_scheduler_worker(handle)

    assert calls == [
        ("materialize", NOW_UTC),
        ("tick", NOW_UTC),
    ]
