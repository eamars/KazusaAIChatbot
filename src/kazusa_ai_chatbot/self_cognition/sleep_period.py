"""Self-cognition trigger sleep-period policy.

The predicate in this module gates self-cognition trigger collection only. It
does not pause reflection, consolidation, scheduler execution, dispatcher
delivery, or service worker loops.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from kazusa_ai_chatbot.config import (
    CHARACTER_SLEEP_LOCAL_PERIOD,
    CHARACTER_TIME_ZONE,
)


def is_self_cognition_sleep_period(
    now: datetime,
    *,
    sleep_local_period: str = CHARACTER_SLEEP_LOCAL_PERIOD,
    character_time_zone: str = CHARACTER_TIME_ZONE,
) -> bool:
    """Return whether self-cognition triggers should sleep for this instant.

    Args:
        now: Timezone-aware instant to project into character-local time.
        sleep_local_period: Exact ``HH:MM-HH:MM`` local clock period. Empty
            text disables sleep suppression.
        character_time_zone: IANA timezone used for character-local projection.

    Returns:
        True when selected self-cognition sources should be suppressed.

    Raises:
        ValueError: If ``now`` is timezone-naive or the period is invalid.
    """

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    clean_period = sleep_local_period.strip()
    if not clean_period:
        return False

    start_minutes, end_minutes = _local_period_bounds(clean_period)
    local_timezone = ZoneInfo(character_time_zone)
    local_now = now.astimezone(local_timezone)
    current_minutes = (local_now.hour * 60) + local_now.minute

    if start_minutes < end_minutes:
        return start_minutes <= current_minutes < end_minutes

    return (
        current_minutes >= start_minutes
        or current_minutes < end_minutes
    )


def _local_period_bounds(value: str) -> tuple[int, int]:
    """Parse exact ``HH:MM-HH:MM`` text into local-minute bounds."""

    parts = value.split("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("sleep period must use HH:MM-HH:MM")

    start_minutes = _local_time_minutes(parts[0])
    end_minutes = _local_time_minutes(parts[1])
    if start_minutes == end_minutes:
        raise ValueError("sleep period start and end must differ")

    return start_minutes, end_minutes


def _local_time_minutes(value: str) -> int:
    """Parse exact ``HH:MM`` text into minutes after local midnight."""

    if len(value) != 5 or value[2] != ":":
        raise ValueError("sleep period must use HH:MM-HH:MM")
    hour_text = value[:2]
    minute_text = value[3:]
    if not hour_text.isdecimal() or not minute_text.isdecimal():
        raise ValueError("sleep period must use HH:MM-HH:MM")

    hour = int(hour_text)
    minute = int(minute_text)
    if hour > 23 or minute > 59:
        raise ValueError("sleep period must use HH:MM-HH:MM")

    minutes = (hour * 60) + minute
    return minutes
