"""MongoDB repository helpers for calendar schedules and runs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import ReturnDocument

from kazusa_ai_chatbot.calendar_scheduler import models
from kazusa_ai_chatbot.db._client import get_db


CALENDAR_SCHEDULES_COLLECTION = "calendar_schedules"
CALENDAR_RUNS_COLLECTION = "calendar_runs"


async def upsert_calendar_schedule(schedule: dict[str, Any]) -> object:
    """Insert a schedule once, keyed by idempotency key."""

    db = await get_db()
    result = await db.calendar_schedules.update_one(
        {"idempotency_key": schedule["idempotency_key"]},
        {"$setOnInsert": schedule},
        upsert=True,
    )
    return result


async def refresh_calendar_schedule_state(
    schedule: dict[str, Any],
) -> object:
    """Upsert mutable schedule state while preserving original creation time."""

    set_doc = dict(schedule)
    created_at = set_doc.pop("created_at")
    db = await get_db()
    result = await db.calendar_schedules.update_one(
        {"idempotency_key": schedule["idempotency_key"]},
        {
            "$set": set_doc,
            "$setOnInsert": {"created_at": created_at},
        },
        upsert=True,
    )
    return result


async def upsert_calendar_run(run: dict[str, Any]) -> object:
    """Insert a run once, keyed by idempotency key."""

    db = await get_db()
    result = await db.calendar_runs.update_one(
        {"idempotency_key": run["idempotency_key"]},
        {"$setOnInsert": run},
        upsert=True,
    )
    return result


async def list_due_calendar_runs(
    *,
    current_timestamp_utc: str,
    trigger_kinds: list[str],
    max_attempts: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Return due runs eligible for later worker ownership claim."""

    db = await get_db()
    cursor = (
        db.calendar_runs.find(
            _due_run_claim_filter(
                current_timestamp_utc=current_timestamp_utc,
                trigger_kinds=trigger_kinds,
                max_attempts=max_attempts,
            )
        )
        .sort([("due_at", 1), ("run_id", 1)])
        .limit(limit)
    )
    runs: list[dict[str, Any]] = []
    async for run in cursor:
        runs.append(run)
    return runs


async def claim_calendar_run(
    run_id: str,
    *,
    trigger_kind: str,
    current_timestamp_utc: str,
    lease_owner: str,
    max_attempts: int,
    lease_expires_at: str | None = None,
    lease_duration_seconds: int | None = None,
) -> dict[str, Any] | None:
    """Atomically claim one due run by id for source-case workers."""

    if lease_expires_at is None:
        if lease_duration_seconds is None:
            raise ValueError("lease_expires_at or lease_duration_seconds required")
        lease_expires_at = _lease_expiry(
            current_timestamp_utc=current_timestamp_utc,
            lease_duration_seconds=lease_duration_seconds,
        )

    db = await get_db()
    claim_filter = _due_run_claim_filter(
        current_timestamp_utc=current_timestamp_utc,
        trigger_kinds=[trigger_kind],
        max_attempts=max_attempts,
    )
    claim_filter["run_id"] = run_id
    claim_filter["trigger_kind"] = trigger_kind
    run = await db.calendar_runs.find_one_and_update(
        claim_filter,
        _claim_update(
            current_timestamp_utc=current_timestamp_utc,
            lease_owner=lease_owner,
            lease_expires_at=lease_expires_at,
        ),
        return_document=ReturnDocument.AFTER,
    )
    return run


async def claim_due_calendar_runs(
    *,
    current_timestamp_utc: str,
    lease_owner: str,
    trigger_kinds: list[str],
    max_attempts: int,
    limit: int,
    lease_expires_at: str | None = None,
    lease_duration_seconds: int | None = None,
) -> list[dict[str, Any]]:
    """Atomically claim due pending or lease-expired runs.

    Args:
        current_timestamp_utc: Claim timestamp and due upper bound.
        lease_owner: Worker identity written to the lease fields.
        trigger_kinds: Closed trigger kinds this worker may execute.
        max_attempts: Runs at or above this count are not claimed.
        limit: Maximum number of claims for this call.
        lease_expires_at: Absolute lease expiry timestamp.
        lease_duration_seconds: Relative lease duration, used when absolute
            expiry is not supplied.

    Returns:
        Claimed run documents after update.
    """

    if lease_expires_at is None:
        if lease_duration_seconds is None:
            raise ValueError("lease_expires_at or lease_duration_seconds required")
        lease_expires_at = _lease_expiry(
            current_timestamp_utc=current_timestamp_utc,
            lease_duration_seconds=lease_duration_seconds,
        )

    db = await get_db()
    claimed: list[dict[str, Any]] = []
    for _ in range(limit):
        run = await db.calendar_runs.find_one_and_update(
            _due_run_claim_filter(
                current_timestamp_utc=current_timestamp_utc,
                trigger_kinds=trigger_kinds,
                max_attempts=max_attempts,
            ),
            _claim_update(
                current_timestamp_utc=current_timestamp_utc,
                lease_owner=lease_owner,
                lease_expires_at=lease_expires_at,
            ),
            sort=[("due_at", 1), ("run_id", 1)],
            return_document=ReturnDocument.AFTER,
        )
        if run is None:
            break
        claimed.append(run)

    return claimed


async def mark_calendar_run_completed(
    run_id: str,
    *,
    lease_owner: str,
    storage_timestamp_utc: str,
    result: dict[str, Any],
) -> bool:
    """Mark a running leased run completed and release its lease."""

    db = await get_db()
    update_result = await db.calendar_runs.update_one(
        {
            "run_id": run_id,
            "status": models.RUN_STATUS_RUNNING,
            "lease_owner": lease_owner,
        },
        {
            "$set": {
                "status": models.RUN_STATUS_COMPLETED,
                "completed_at": storage_timestamp_utc,
                "updated_at": storage_timestamp_utc,
                "result_summary": result,
                "lease_owner": None,
                "lease_expires_at": None,
            }
        },
    )
    matched = update_result.matched_count == 1
    return matched


async def mark_calendar_run_failed(
    run_id: str,
    *,
    lease_owner: str,
    storage_timestamp_utc: str,
    error: str,
    retryable: bool,
) -> bool:
    """Mark a running leased run failed and release its lease."""

    db = await get_db()
    update_result = await db.calendar_runs.update_one(
        {
            "run_id": run_id,
            "status": models.RUN_STATUS_RUNNING,
            "lease_owner": lease_owner,
        },
        {
            "$set": {
                "status": models.RUN_STATUS_FAILED,
                "failed_at": storage_timestamp_utc,
                "updated_at": storage_timestamp_utc,
                "failure_summary": {
                    "error": error,
                    "retryable": retryable,
                },
                "lease_owner": None,
                "lease_expires_at": None,
            }
        },
    )
    matched = update_result.matched_count == 1
    return matched


async def mark_calendar_run_skipped(
    run_id: str,
    *,
    lease_owner: str,
    storage_timestamp_utc: str,
    reason: str,
) -> bool:
    """Mark a running leased run skipped and release its lease."""

    db = await get_db()
    update_result = await db.calendar_runs.update_one(
        {
            "run_id": run_id,
            "status": models.RUN_STATUS_RUNNING,
            "lease_owner": lease_owner,
        },
        {
            "$set": {
                "status": models.RUN_STATUS_SKIPPED,
                "skipped_at": storage_timestamp_utc,
                "updated_at": storage_timestamp_utc,
                "skip_reason": reason,
                "failure_summary": {"reason": reason},
                "lease_owner": None,
                "lease_expires_at": None,
            }
        },
    )
    matched = update_result.matched_count == 1
    return matched


async def cancel_calendar_schedule_by_idempotency_key(
    idempotency_key: str,
    *,
    storage_timestamp_utc: str,
    reason: str,
) -> bool:
    """Cancel an active schedule by its stable idempotency key."""

    db = await get_db()
    update_result = await db.calendar_schedules.update_one(
        {
            "idempotency_key": idempotency_key,
            "status": models.SCHEDULE_STATUS_ACTIVE,
        },
        {
            "$set": {
                "status": models.SCHEDULE_STATUS_CANCELLED,
                "cancelled_at": storage_timestamp_utc,
                "updated_at": storage_timestamp_utc,
                "cancel_reason": reason,
            }
        },
    )
    matched = update_result.matched_count == 1
    return matched


def _lease_expiry(
    *,
    current_timestamp_utc: str,
    lease_duration_seconds: int,
) -> str:
    """Compute an absolute lease expiry timestamp from a duration."""

    current = datetime.fromisoformat(current_timestamp_utc)
    if current.tzinfo is None:
        raise ValueError("current_timestamp_utc must be timezone-aware")
    expires_at = current.astimezone(timezone.utc) + timedelta(
        seconds=lease_duration_seconds,
    )
    rendered = expires_at.isoformat()
    return rendered


def _due_run_claim_filter(
    *,
    current_timestamp_utc: str,
    trigger_kinds: list[str],
    max_attempts: int,
) -> dict[str, Any]:
    """Build the shared due-run eligibility filter for reads and claims."""

    claim_filter = {
        "due_at": {"$lte": current_timestamp_utc},
        "trigger_kind": {"$in": trigger_kinds},
        "attempt_count": {"$lt": max_attempts},
        "$or": [
            {"status": models.RUN_STATUS_PENDING},
            {
                "status": models.RUN_STATUS_RUNNING,
                "lease_expires_at": {"$lte": current_timestamp_utc},
            },
        ],
    }
    return claim_filter


def _claim_update(
    *,
    current_timestamp_utc: str,
    lease_owner: str,
    lease_expires_at: str,
) -> dict[str, Any]:
    """Build the lease update applied by every claim operation."""

    update = {
        "$set": {
            "status": models.RUN_STATUS_RUNNING,
            "claimed_at": current_timestamp_utc,
            "lease_owner": lease_owner,
            "lease_expires_at": lease_expires_at,
            "updated_at": current_timestamp_utc,
        },
        "$inc": {"attempt_count": 1},
    }
    return update
