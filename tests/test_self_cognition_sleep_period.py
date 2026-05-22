"""Tests for self-cognition sleep-period matching."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kazusa_ai_chatbot.self_cognition.sleep_period import (
    is_self_cognition_sleep_period,
)


def test_empty_sleep_period_is_disabled() -> None:
    """Empty sleep period should preserve normal self-cognition triggering."""

    now = datetime(2026, 5, 13, 3, 0, tzinfo=timezone.utc)

    is_sleeping = is_self_cognition_sleep_period(
        now,
        sleep_local_period="",
        character_time_zone="UTC",
    )

    assert is_sleeping is False


def test_same_day_sleep_period_includes_start_and_excludes_end() -> None:
    """Same-day sleep windows include start minute and exclude end minute."""

    before_start = datetime(2026, 5, 13, 1, 59, tzinfo=timezone.utc)
    at_start = datetime(2026, 5, 13, 2, 0, tzinfo=timezone.utc)
    before_end = datetime(2026, 5, 13, 11, 59, tzinfo=timezone.utc)
    at_end = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)

    assert is_self_cognition_sleep_period(
        before_start,
        sleep_local_period="02:00-12:00",
        character_time_zone="UTC",
    ) is False
    assert is_self_cognition_sleep_period(
        at_start,
        sleep_local_period="02:00-12:00",
        character_time_zone="UTC",
    ) is True
    assert is_self_cognition_sleep_period(
        before_end,
        sleep_local_period="02:00-12:00",
        character_time_zone="UTC",
    ) is True
    assert is_self_cognition_sleep_period(
        at_end,
        sleep_local_period="02:00-12:00",
        character_time_zone="UTC",
    ) is False


def test_overnight_sleep_period_wraps_across_midnight() -> None:
    """Overnight sleep windows should match both sides of midnight."""

    at_start = datetime(2026, 5, 13, 23, 30, tzinfo=timezone.utc)
    after_midnight = datetime(2026, 5, 14, 6, 0, tzinfo=timezone.utc)
    at_end = datetime(2026, 5, 14, 7, 30, tzinfo=timezone.utc)
    daytime = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)

    assert is_self_cognition_sleep_period(
        at_start,
        sleep_local_period="23:30-07:30",
        character_time_zone="UTC",
    ) is True
    assert is_self_cognition_sleep_period(
        after_midnight,
        sleep_local_period="23:30-07:30",
        character_time_zone="UTC",
    ) is True
    assert is_self_cognition_sleep_period(
        at_end,
        sleep_local_period="23:30-07:30",
        character_time_zone="UTC",
    ) is False
    assert is_self_cognition_sleep_period(
        daytime,
        sleep_local_period="23:30-07:30",
        character_time_zone="UTC",
    ) is False


def test_sleep_period_uses_character_time_zone_projection() -> None:
    """The predicate should compare against configured-local wall time."""

    utc_instant = datetime(2026, 5, 12, 14, 0, tzinfo=timezone.utc)

    is_sleeping = is_self_cognition_sleep_period(
        utc_instant,
        sleep_local_period="02:00-12:00",
        character_time_zone="Pacific/Auckland",
    )

    assert is_sleeping is True


def test_sleep_period_rejects_naive_datetime() -> None:
    """The predicate requires an aware instant for timezone projection."""

    now = datetime(2026, 5, 13, 3, 0)

    with pytest.raises(ValueError):
        is_self_cognition_sleep_period(
            now,
            sleep_local_period="02:00-12:00",
            character_time_zone="UTC",
        )
