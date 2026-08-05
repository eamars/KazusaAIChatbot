"""Self-cognition trigger sleep-period policy.

The predicate in this module gates self-cognition trigger collection only. It
does not pause reflection, consolidation, scheduler execution, dispatcher
delivery, or service worker loops.
"""

from __future__ import annotations

from datetime import datetime

from kazusa_ai_chatbot.config import (
    CHARACTER_SLEEP_LOCAL_PERIOD,
    CHARACTER_TIME_ZONE,
)
from kazusa_ai_chatbot.time_boundary import (
    local_minutes_in_zone,
    local_period_bounds,
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

    start_minutes, end_minutes = local_period_bounds(clean_period)
    current_minutes = local_minutes_in_zone(now, time_zone=character_time_zone)

    if start_minutes < end_minutes:
        return start_minutes <= current_minutes < end_minutes

    return (
        current_minutes >= start_minutes
        or current_minutes < end_minutes
    )
