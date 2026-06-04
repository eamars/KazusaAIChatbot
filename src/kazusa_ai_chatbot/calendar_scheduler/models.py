"""Document builders for the durable calendar scheduler."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any


CALENDAR_SCHEDULE_SCHEMA_VERSION = "calendar_schedule.v1"
CALENDAR_RUN_SCHEMA_VERSION = "calendar_run.v1"
CALENDAR_OWNER = "calendar_scheduler"
DEFAULT_RUN_MAX_ATTEMPTS = 3

TRIGGER_FUTURE_COGNITION = "future_cognition"
TRIGGER_COMMITMENT_DUE_COGNITION = "commitment_due_cognition"
TRIGGER_REFLECTION_PHASE_SLOT = "reflection_phase_slot"
TRIGGER_RECURRING_SELF_CHECK = "recurring_self_check"
CALENDAR_TRIGGER_KINDS = {
    TRIGGER_FUTURE_COGNITION,
    TRIGGER_COMMITMENT_DUE_COGNITION,
    TRIGGER_REFLECTION_PHASE_SLOT,
    TRIGGER_RECURRING_SELF_CHECK,
}

SCHEDULE_STATUS_ACTIVE = "active"
SCHEDULE_STATUS_PAUSED = "paused"
SCHEDULE_STATUS_COMPLETED = "completed"
SCHEDULE_STATUS_CANCELLED = "cancelled"

RUN_STATUS_PENDING = "pending"
RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_CANCELLED = "cancelled"
RUN_STATUS_SKIPPED = "skipped"


def build_one_time_calendar_schedule(
    *,
    trigger_kind: str,
    due_at: str,
    payload: dict[str, Any],
    source_scope: dict[str, Any],
    idempotency_key: str,
    storage_timestamp_utc: str,
    timezone: str = "UTC",
    legacy_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one active one-time schedule for a closed trigger kind.

    Args:
        trigger_kind: One value from the calendar trigger-kind roster.
        due_at: Absolute UTC storage timestamp for the first and only run.
        payload: Trigger-owned scheduler metadata.
        source_scope: Structural source identity for the owning subsystem.
        idempotency_key: Stable duplicate-suppression key for this schedule.
        storage_timestamp_utc: Storage timestamp used for created/updated.
        timezone: IANA timezone label used by recurring schedule types.
        legacy_source: Optional migration provenance for one-time legacy rows.

    Returns:
        A deterministic schedule document ready for upsert.
    """

    _validate_trigger_kind(trigger_kind)
    schedule_id = _stable_id("calendar_schedule", idempotency_key)
    schedule = {
        "schema_version": CALENDAR_SCHEDULE_SCHEMA_VERSION,
        "owner": CALENDAR_OWNER,
        "schedule_id": schedule_id,
        "trigger_kind": trigger_kind,
        "status": SCHEDULE_STATUS_ACTIVE,
        "start_at": due_at,
        "next_run_at": due_at,
        "recurrence": {"kind": "once"},
        "payload": deepcopy(payload),
        "source_scope": deepcopy(source_scope),
        "idempotency_key": idempotency_key,
        "timezone": timezone,
        "legacy_source": deepcopy(legacy_source),
        "created_at": storage_timestamp_utc,
        "updated_at": storage_timestamp_utc,
    }
    return schedule


def build_calendar_run_from_schedule(
    schedule: dict[str, Any],
    *,
    due_at: str,
    storage_timestamp_utc: str,
) -> dict[str, Any]:
    """Build a due run from a schedule without adapter-send semantics.

    Args:
        schedule: Calendar schedule document.
        due_at: Absolute UTC timestamp this run should execute at.
        storage_timestamp_utc: Storage timestamp used for created/updated.

    Returns:
        A deterministic pending run document.
    """

    trigger_kind = schedule["trigger_kind"]
    _validate_trigger_kind(trigger_kind)
    idempotency_key = f"{schedule['idempotency_key']}:{due_at}"
    run_id = _stable_id("calendar_run", idempotency_key)
    run = {
        "schema_version": CALENDAR_RUN_SCHEMA_VERSION,
        "owner": CALENDAR_OWNER,
        "run_id": run_id,
        "schedule_id": schedule["schedule_id"],
        "trigger_kind": trigger_kind,
        "status": RUN_STATUS_PENDING,
        "due_at": due_at,
        "payload": deepcopy(schedule["payload"]),
        "source_scope": deepcopy(schedule["source_scope"]),
        "idempotency_key": idempotency_key,
        "attempt_count": 0,
        "max_attempts": schedule.get("max_attempts", DEFAULT_RUN_MAX_ATTEMPTS),
        "claimed_at": None,
        "completed_at": None,
        "failed_at": None,
        "skipped_at": None,
        "lease_owner": None,
        "lease_expires_at": None,
        "result_summary": None,
        "failure_summary": None,
        "legacy_source": schedule.get("legacy_source"),
        "period_start_utc": schedule.get("period_start_utc"),
        "slot_index": schedule.get("slot_index"),
        "offset_seconds": schedule.get("offset_seconds"),
        "created_at": storage_timestamp_utc,
        "updated_at": storage_timestamp_utc,
    }
    return run


def _validate_trigger_kind(trigger_kind: str) -> None:
    """Fail closed for trigger kinds outside the calendar roster."""

    if trigger_kind not in CALENDAR_TRIGGER_KINDS:
        raise ValueError(f"unsupported calendar trigger kind: {trigger_kind}")


def _stable_id(prefix: str, value: str) -> str:
    """Build a stable readable id from one idempotency value."""

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    stable_id = f"{prefix}_{digest[:32]}"
    return stable_id


def stable_json_hash(value: dict[str, Any]) -> str:
    """Return the deterministic SHA-256 hex digest for a JSON document."""

    serialized = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest
