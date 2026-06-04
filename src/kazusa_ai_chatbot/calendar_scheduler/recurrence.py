"""Deterministic recurrence calculation for calendar schedules."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def compute_next_run_at(
    schedule: dict,
    *,
    after_utc: str,
) -> str:
    """Compute the next run timestamp strictly after a UTC instant.

    Args:
        schedule: Schedule document containing ``start_at`` and recurrence.
        after_utc: Absolute UTC timestamp used as the lower bound.

    Returns:
        The next due timestamp as an ISO-8601 UTC string.

    Raises:
        ValueError: If the recurrence shape is unsupported or invalid.
    """

    recurrence = schedule["recurrence"]
    kind = recurrence["kind"]
    if kind == "once":
        next_run_at = schedule["next_run_at"]
        return next_run_at
    if kind == "fixed_interval_seconds":
        next_run_at = _fixed_interval_next_run_at(
            start_at=schedule["start_at"],
            interval_seconds=recurrence["interval_seconds"],
            after_utc=after_utc,
        )
        return next_run_at
    if kind == "daily_local_time":
        next_run_at = _daily_local_time_next_run_at(
            timezone_name=schedule["timezone"],
            local_time_text=recurrence["local_time"],
            after_utc=after_utc,
        )
        return next_run_at

    raise ValueError(f"unsupported recurrence kind: {kind}")


def compute_phase_period_offsets(config: dict[str, int]) -> list[int]:
    """Return valid reflection phase offsets inside one period.

    Args:
        config: Timing values for period length, minimum spacing, and slot cap.

    Returns:
        Slot offsets in seconds.

    Raises:
        ValueError: If the requested slot budget cannot fit in the period.
    """

    period_seconds = config["period_seconds"]
    min_slot_spacing_seconds = config["min_slot_spacing_seconds"]
    max_slots_per_period = config["max_slots_per_period"]
    if period_seconds < 1:
        raise ValueError("phase_period period_seconds must be >= 1")
    if min_slot_spacing_seconds < 1:
        raise ValueError("phase_period min_slot_spacing_seconds must be >= 1")
    if max_slots_per_period < 1:
        raise ValueError("phase_period max_slots_per_period must be >= 1")

    allowed_slot_count = (
        (period_seconds - 1) // min_slot_spacing_seconds
    ) + 1
    if max_slots_per_period > allowed_slot_count:
        raise ValueError("phase_period slot budget cannot fit in period")

    offsets = [
        slot_index * min_slot_spacing_seconds
        for slot_index in range(max_slots_per_period)
    ]
    return offsets


def _fixed_interval_next_run_at(
    *,
    start_at: str,
    interval_seconds: int,
    after_utc: str,
) -> str:
    """Compute fixed-interval recurrence from the anchor, avoiding drift."""

    if interval_seconds < 1:
        raise ValueError("recurrence interval_seconds must be >= 1")

    start = _parse_utc(start_at)
    after = _parse_utc(after_utc)
    if after < start:
        next_run = start
        return _iso_utc(next_run)

    elapsed_seconds = int((after - start).total_seconds())
    completed_intervals = (elapsed_seconds // interval_seconds) + 1
    next_run = start + timedelta(
        seconds=completed_intervals * interval_seconds,
    )
    next_run_at = _iso_utc(next_run)
    return next_run_at


def _daily_local_time_next_run_at(
    *,
    timezone_name: str,
    local_time_text: str,
    after_utc: str,
) -> str:
    """Compute the next daily wall-clock slot in the character timezone."""

    local_zone = ZoneInfo(timezone_name)
    after = _parse_utc(after_utc)
    after_local = after.astimezone(local_zone)
    local_slot_time = _parse_local_time(local_time_text)
    candidate_local = datetime.combine(
        after_local.date(),
        local_slot_time,
        tzinfo=local_zone,
    )
    if candidate_local.astimezone(timezone.utc) <= after:
        candidate_local = datetime.combine(
            after_local.date() + timedelta(days=1),
            local_slot_time,
            tzinfo=local_zone,
        )

    next_run_at = _iso_utc(candidate_local)
    return next_run_at


def _parse_local_time(value: str) -> time:
    """Parse an exact HH:MM local wall-clock time."""

    if len(value) != 5 or value[2] != ":":
        raise ValueError("recurrence local_time must use HH:MM")
    hour_text = value[:2]
    minute_text = value[3:]
    if not hour_text.isdecimal() or not minute_text.isdecimal():
        raise ValueError("recurrence local_time must use HH:MM")

    hour = int(hour_text)
    minute = int(minute_text)
    if hour > 23 or minute > 59:
        raise ValueError("recurrence local_time must use HH:MM")

    parsed_time = time(hour=hour, minute=minute)
    return parsed_time


def _parse_utc(value: str) -> datetime:
    """Parse a storage timestamp and normalize it to UTC."""

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("UTC timestamp must be timezone-aware")

    normalized = parsed.astimezone(timezone.utc)
    return normalized


def _iso_utc(value: datetime) -> str:
    """Render an aware datetime as a UTC ISO storage timestamp."""

    rendered = value.astimezone(timezone.utc).isoformat()
    return rendered
