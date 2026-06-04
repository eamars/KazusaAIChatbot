"""Contract tests for calendar scheduler MongoDB repository helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo import ReturnDocument


NOW_UTC = "2026-06-04T00:15:00+00:00"
LEASE_EXPIRES_AT = "2026-06-04T00:20:00+00:00"


def _db() -> MagicMock:
    calendar_runs = MagicMock()
    calendar_runs.find_one_and_update = AsyncMock(
        return_value={"run_id": "run-1", "trigger_kind": "future_cognition"}
    )
    calendar_runs.update_one = AsyncMock(
        return_value=MagicMock(
            matched_count=1,
            modified_count=1,
            upserted_id="x",
        )
    )
    calendar_schedules = MagicMock()
    calendar_schedules.update_one = AsyncMock(
        return_value=MagicMock(matched_count=0, modified_count=0, upserted_id="x")
    )
    db = MagicMock()
    db.calendar_runs = calendar_runs
    db.calendar_schedules = calendar_schedules
    return db


class _AsyncCursor:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        self.sort_args = None
        self.limit_value = None

    def sort(self, args):
        self.sort_args = args
        return self

    def limit(self, value: int):
        self.limit_value = value
        return self

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for doc in self.docs:
            yield doc


@pytest.mark.asyncio
async def test_refresh_calendar_schedule_state_updates_mutable_schedule() -> None:
    """Mutable schedules should keep calendar_schedules aligned with source state."""

    from kazusa_ai_chatbot.calendar_scheduler import repository

    db = _db()
    schedule = {
        "schedule_id": "calendar_schedule_commitment_1",
        "idempotency_key": "commitment_due:commitment-1",
        "trigger_kind": "commitment_due_cognition",
        "status": "active",
        "next_run_at": NOW_UTC,
        "payload": {
            "unit_id": "commitment-1",
            "global_user_id": "user-1",
            "due_at": NOW_UTC,
        },
        "created_at": NOW_UTC,
        "updated_at": NOW_UTC,
    }

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))
        result = await repository.refresh_calendar_schedule_state(schedule)

    assert result.upserted_id == "x"
    update_call = db.calendar_schedules.update_one.await_args
    assert update_call.args[0] == {
        "idempotency_key": schedule["idempotency_key"]
    }
    expected_set = dict(schedule)
    expected_set.pop("created_at")
    assert update_call.args[1] == {
        "$set": expected_set,
        "$setOnInsert": {"created_at": schedule["created_at"]},
    }
    assert update_call.kwargs == {"upsert": True}


@pytest.mark.asyncio
async def test_upsert_calendar_run_uses_idempotency_key() -> None:
    """Run materialization should be idempotent across repeated passes."""

    from kazusa_ai_chatbot.calendar_scheduler import repository

    db = _db()
    run = {
        "run_id": "run-1",
        "idempotency_key": "future_cognition:attempt-1:2026-06-04T00:15:00Z",
        "trigger_kind": "future_cognition",
        "status": "pending",
        "due_at": NOW_UTC,
    }

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))
        result = await repository.upsert_calendar_run(run)

    assert result.upserted_id == "x"
    update_call = db.calendar_runs.update_one.await_args
    assert update_call.args[0] == {"idempotency_key": run["idempotency_key"]}
    assert update_call.args[1] == {"$setOnInsert": run}
    assert update_call.kwargs == {"upsert": True}


@pytest.mark.asyncio
async def test_list_due_calendar_runs_reads_eligible_runs_without_claiming() -> None:
    """Source collectors may inspect due runs before worker ownership claim."""

    from kazusa_ai_chatbot.calendar_scheduler import models, repository

    db = _db()
    cursor = _AsyncCursor([
        {"run_id": "run-1", "trigger_kind": models.TRIGGER_FUTURE_COGNITION}
    ])
    db.calendar_runs.find = MagicMock(return_value=cursor)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))
        runs = await repository.list_due_calendar_runs(
            current_timestamp_utc=NOW_UTC,
            trigger_kinds=[models.TRIGGER_FUTURE_COGNITION],
            max_attempts=3,
            limit=2,
        )

    assert runs == [
        {"run_id": "run-1", "trigger_kind": models.TRIGGER_FUTURE_COGNITION}
    ]
    assert db.calendar_runs.find.call_args.args[0] == {
        "due_at": {"$lte": NOW_UTC},
        "trigger_kind": {"$in": [models.TRIGGER_FUTURE_COGNITION]},
        "attempt_count": {"$lt": 3},
        "$or": [
            {"status": models.RUN_STATUS_PENDING},
            {
                "status": models.RUN_STATUS_RUNNING,
                "lease_expires_at": {"$lte": NOW_UTC},
            },
        ],
    }
    assert cursor.sort_args == [("due_at", 1), ("run_id", 1)]
    assert cursor.limit_value == 2


@pytest.mark.asyncio
async def test_list_pending_calendar_runs_for_source_scopes_future_evidence() -> None:
    """Recall evidence should read only scoped pending future calendar runs."""

    from kazusa_ai_chatbot.calendar_scheduler import models, repository

    db = _db()
    cursor = _AsyncCursor([
        {"run_id": "run-1", "trigger_kind": models.TRIGGER_FUTURE_COGNITION}
    ])
    db.calendar_runs.find = MagicMock(return_value=cursor)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))
        runs = await repository.list_pending_calendar_runs_for_source(
            platform="qq",
            platform_channel_id="chan-1",
            global_user_id="user-1",
            current_timestamp_utc=NOW_UTC,
            limit=3,
        )

    assert runs == [
        {"run_id": "run-1", "trigger_kind": models.TRIGGER_FUTURE_COGNITION}
    ]
    assert db.calendar_runs.find.call_args.args[0] == {
        "status": models.RUN_STATUS_PENDING,
        "due_at": {"$gte": NOW_UTC},
        "trigger_kind": {
            "$in": [
                models.TRIGGER_FUTURE_COGNITION,
                models.TRIGGER_COMMITMENT_DUE_COGNITION,
            ]
        },
        "$or": [
            {
                "source_scope.source_platform": "qq",
                "source_scope.source_channel_id": "chan-1",
                "source_scope.source_user_id": "user-1",
            },
            {"payload.global_user_id": "user-1"},
        ],
    }
    assert cursor.sort_args == [("due_at", 1), ("run_id", 1)]
    assert cursor.limit_value == 3


@pytest.mark.asyncio
async def test_claim_calendar_run_claims_one_named_due_run() -> None:
    """Self-cognition should claim the same run projected into its case."""

    from kazusa_ai_chatbot.calendar_scheduler import models, repository

    db = _db()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))
        claimed = await repository.claim_calendar_run(
            "run-1",
            trigger_kind=models.TRIGGER_FUTURE_COGNITION,
            current_timestamp_utc=NOW_UTC,
            lease_owner="self_cognition_worker",
            lease_expires_at=LEASE_EXPIRES_AT,
            max_attempts=3,
        )

    assert claimed == {"run_id": "run-1", "trigger_kind": "future_cognition"}
    claim_call = db.calendar_runs.find_one_and_update.await_args
    assert claim_call.args[0] == {
        "run_id": "run-1",
        "due_at": {"$lte": NOW_UTC},
        "trigger_kind": models.TRIGGER_FUTURE_COGNITION,
        "attempt_count": {"$lt": 3},
        "$or": [
            {"status": models.RUN_STATUS_PENDING},
            {
                "status": models.RUN_STATUS_RUNNING,
                "lease_expires_at": {"$lte": NOW_UTC},
            },
        ],
    }
    assert claim_call.args[1] == {
        "$set": {
            "status": models.RUN_STATUS_RUNNING,
            "claimed_at": NOW_UTC,
            "lease_owner": "self_cognition_worker",
            "lease_expires_at": LEASE_EXPIRES_AT,
            "updated_at": NOW_UTC,
        },
        "$inc": {"attempt_count": 1},
    }
    assert claim_call.kwargs["return_document"] == ReturnDocument.AFTER


@pytest.mark.asyncio
async def test_claim_due_calendar_runs_are_atomic_and_lease_bound() -> None:
    """Claims should use one pending-or-expired update as the ownership gate."""

    from kazusa_ai_chatbot.calendar_scheduler import models, repository

    db = _db()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))
        claimed = await repository.claim_due_calendar_runs(
            current_timestamp_utc=NOW_UTC,
            lease_owner="worker-a",
            lease_expires_at=LEASE_EXPIRES_AT,
            trigger_kinds=[models.TRIGGER_FUTURE_COGNITION],
            max_attempts=3,
            limit=1,
        )

    assert claimed == [
        {
            "run_id": "run-1",
            "trigger_kind": "future_cognition",
        }
    ]
    claim_call = db.calendar_runs.find_one_and_update.await_args
    query = claim_call.args[0]
    update = claim_call.args[1]
    assert query == {
        "due_at": {"$lte": NOW_UTC},
        "trigger_kind": {"$in": [models.TRIGGER_FUTURE_COGNITION]},
        "attempt_count": {"$lt": 3},
        "$or": [
            {"status": models.RUN_STATUS_PENDING},
            {
                "status": models.RUN_STATUS_RUNNING,
                "lease_expires_at": {"$lte": NOW_UTC},
            },
        ],
    }
    assert update == {
        "$set": {
            "status": models.RUN_STATUS_RUNNING,
            "claimed_at": NOW_UTC,
            "lease_owner": "worker-a",
            "lease_expires_at": LEASE_EXPIRES_AT,
            "updated_at": NOW_UTC,
        },
        "$inc": {"attempt_count": 1},
    }
    assert claim_call.kwargs["sort"] == [("due_at", 1), ("run_id", 1)]
    assert claim_call.kwargs["return_document"] == ReturnDocument.AFTER


@pytest.mark.asyncio
async def test_claim_due_calendar_runs_stops_at_claim_limit() -> None:
    """Repository claiming should not exceed the configured per-tick limit."""

    from kazusa_ai_chatbot.calendar_scheduler import models, repository

    db = _db()
    db.calendar_runs.find_one_and_update = AsyncMock(
        side_effect=[
            {"run_id": "run-1", "trigger_kind": models.TRIGGER_FUTURE_COGNITION},
            {"run_id": "run-2", "trigger_kind": models.TRIGGER_FUTURE_COGNITION},
            {"run_id": "run-3", "trigger_kind": models.TRIGGER_FUTURE_COGNITION},
        ]
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))
        claimed = await repository.claim_due_calendar_runs(
            current_timestamp_utc=NOW_UTC,
            lease_owner="worker-a",
            lease_expires_at=LEASE_EXPIRES_AT,
            trigger_kinds=[models.TRIGGER_FUTURE_COGNITION],
            max_attempts=3,
            limit=2,
        )

    assert [run["run_id"] for run in claimed] == ["run-1", "run-2"]
    assert db.calendar_runs.find_one_and_update.await_count == 2


@pytest.mark.asyncio
async def test_mark_run_terminal_clears_lease_fields() -> None:
    """Terminal transitions must release the durable run lease."""

    from kazusa_ai_chatbot.calendar_scheduler import models, repository

    db = _db()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))
        ok = await repository.mark_calendar_run_completed(
            "run-1",
            lease_owner="worker-a",
            storage_timestamp_utc=NOW_UTC,
            result={"case_id": "case-1"},
        )

    assert ok is True
    update_call = db.calendar_runs.update_one.await_args
    assert update_call.args[0] == {
        "run_id": "run-1",
        "status": models.RUN_STATUS_RUNNING,
        "lease_owner": "worker-a",
    }
    assert update_call.args[1] == {
        "$set": {
            "status": models.RUN_STATUS_COMPLETED,
            "completed_at": NOW_UTC,
            "updated_at": NOW_UTC,
            "result_summary": {"case_id": "case-1"},
            "lease_owner": None,
            "lease_expires_at": None,
        }
    }


@pytest.mark.asyncio
async def test_mark_run_failed_writes_failure_summary_only() -> None:
    """Failure transitions should not store raw top-level error fields."""

    from kazusa_ai_chatbot.calendar_scheduler import models, repository

    db = _db()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))
        ok = await repository.mark_calendar_run_failed(
            "run-1",
            lease_owner="worker-a",
            storage_timestamp_utc=NOW_UTC,
            error="unsupported calendar trigger kind: send_message",
            retryable=False,
        )

    assert ok is True
    update_call = db.calendar_runs.update_one.await_args
    assert update_call.args[0] == {
        "run_id": "run-1",
        "status": models.RUN_STATUS_RUNNING,
        "lease_owner": "worker-a",
    }
    assert update_call.args[1] == {
        "$set": {
            "status": models.RUN_STATUS_FAILED,
            "failed_at": NOW_UTC,
            "updated_at": NOW_UTC,
            "failure_summary": {
                "error": "unsupported calendar trigger kind: send_message",
                "retryable": False,
            },
            "lease_owner": None,
            "lease_expires_at": None,
        }
    }


@pytest.mark.asyncio
async def test_mark_run_skipped_is_lease_bound() -> None:
    """Skipped transitions should use the same lease gate as terminal writes."""

    from kazusa_ai_chatbot.calendar_scheduler import models, repository

    db = _db()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(repository, "get_db", AsyncMock(return_value=db))
        ok = await repository.mark_calendar_run_skipped(
            "run-1",
            lease_owner="worker-a",
            storage_timestamp_utc=NOW_UTC,
            reason="stale_active_commitment_due_at",
        )

    assert ok is True
    update_call = db.calendar_runs.update_one.await_args
    assert update_call.args[0] == {
        "run_id": "run-1",
        "status": models.RUN_STATUS_RUNNING,
        "lease_owner": "worker-a",
    }
    assert update_call.args[1] == {
        "$set": {
            "status": models.RUN_STATUS_SKIPPED,
            "skipped_at": NOW_UTC,
            "updated_at": NOW_UTC,
            "skip_reason": "stale_active_commitment_due_at",
            "failure_summary": {
                "reason": "stale_active_commitment_due_at",
            },
            "lease_owner": None,
            "lease_expires_at": None,
        }
    }
