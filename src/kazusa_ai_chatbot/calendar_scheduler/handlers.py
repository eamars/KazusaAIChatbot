"""Typed calendar trigger helpers owned by scheduler-adjacent subsystems."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.calendar_scheduler import models


ACTIVE_COMMITMENT_TYPE = "active_commitment"
ACTIVE_STATUS = "active"


async def reconcile_active_commitment_calendar_schedule(
    unit: dict[str, Any],
    *,
    repository: Any,
    storage_timestamp_utc: str,
) -> dict[str, str]:
    """Upsert or cancel the due schedule for one active-commitment unit."""

    unit_id = unit["unit_id"]
    idempotency_key = f"commitment_due:{unit_id}"
    if not _is_schedulable_active_commitment(unit):
        await repository.cancel_calendar_schedule_by_idempotency_key(
            idempotency_key,
            storage_timestamp_utc=storage_timestamp_utc,
            reason="active_commitment_not_schedulable",
        )
        result = {"status": "cancelled", "unit_id": unit_id}
        return result

    due_at = unit["due_at"]
    payload = {
        "unit_id": unit_id,
        "global_user_id": unit["global_user_id"],
        "due_at": due_at,
    }
    schedule = models.build_one_time_calendar_schedule(
        trigger_kind=models.TRIGGER_COMMITMENT_DUE_COGNITION,
        due_at=due_at,
        payload=payload,
        source_scope={},
        idempotency_key=idempotency_key,
        storage_timestamp_utc=storage_timestamp_utc,
    )
    await repository.upsert_calendar_schedule(schedule)
    result = {"status": "scheduled", "unit_id": unit_id}
    return result


async def handle_commitment_due_cognition_run(
    run: dict[str, Any],
    *,
    memory_unit_reader: Callable[[str], Awaitable[dict[str, Any] | None]],
    active_commitment_case_builder: Callable[
        [dict[str, Any]],
        Awaitable[dict[str, Any]],
    ],
) -> dict[str, Any]:
    """Re-read and validate an active commitment before creating a due case."""

    payload = run["payload"]
    unit_id = payload["unit_id"]
    stored_unit = await memory_unit_reader(unit_id)
    if stored_unit is None:
        result = _skip(unit_id, "active_commitment_not_found")
        return result
    if stored_unit["status"] != ACTIVE_STATUS:
        result = _skip(unit_id, "active_commitment_not_active")
        return result
    if stored_unit["unit_type"] != ACTIVE_COMMITMENT_TYPE:
        result = _skip(unit_id, "active_commitment_wrong_type")
        return result
    if stored_unit["global_user_id"] != payload["global_user_id"]:
        result = _skip(unit_id, "active_commitment_wrong_user")
        return result
    if stored_unit["due_at"] != payload["due_at"]:
        result = _skip(unit_id, "stale_active_commitment_due_at")
        return result

    case = await active_commitment_case_builder(stored_unit)
    result = {"status": "case_created", **case}
    return result


def _is_schedulable_active_commitment(unit: dict[str, Any]) -> bool:
    """Return whether a memory unit can own an absolute due schedule."""

    if unit["unit_type"] != ACTIVE_COMMITMENT_TYPE:
        return False
    if unit["status"] != ACTIVE_STATUS:
        return False

    due_at = unit["due_at"]
    if not isinstance(due_at, str) or not due_at:
        return False

    try:
        parsed_due_at = datetime.fromisoformat(due_at)
    except ValueError:
        return False

    schedulable = parsed_due_at.tzinfo is not None
    return schedulable


def _skip(unit_id: str, reason: str) -> dict[str, Any]:
    """Build the sanitized skip result used by due commitment handlers."""

    result = {
        "status": "skipped",
        "reason": reason,
        "unit_id": unit_id,
    }
    return result
