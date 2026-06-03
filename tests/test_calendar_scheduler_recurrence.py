"""Contract tests for deterministic calendar recurrence calculation."""

from __future__ import annotations

import pytest


def test_fixed_interval_recurrence_uses_anchor_not_last_tick_time() -> None:
    """Recurring schedules should not drift when the worker wakes late."""

    from kazusa_ai_chatbot.calendar_scheduler import recurrence

    schedule = {
        "schedule_id": "schedule-interval",
        "start_at": "2026-06-04T00:00:00+00:00",
        "recurrence": {
            "kind": "fixed_interval_seconds",
            "interval_seconds": 900,
        },
    }

    next_run_at = recurrence.compute_next_run_at(
        schedule,
        after_utc="2026-06-04T00:16:31+00:00",
    )

    assert next_run_at == "2026-06-04T00:30:00+00:00"


def test_daily_local_time_recurrence_uses_configured_timezone() -> None:
    """Daily recurrences should be calculated in character-local time."""

    from kazusa_ai_chatbot.calendar_scheduler import recurrence

    schedule = {
        "schedule_id": "schedule-daily",
        "start_at": "2026-06-03T00:00:00+00:00",
        "timezone": "Pacific/Auckland",
        "recurrence": {
            "kind": "daily_local_time",
            "local_time": "04:30",
        },
    }

    before_today_slot = recurrence.compute_next_run_at(
        schedule,
        after_utc="2026-06-03T12:00:00+00:00",
    )
    after_today_slot = recurrence.compute_next_run_at(
        schedule,
        after_utc="2026-06-03T17:00:00+00:00",
    )

    assert before_today_slot == "2026-06-03T16:30:00+00:00"
    assert after_today_slot == "2026-06-04T16:30:00+00:00"


def test_daily_local_time_preserves_wall_clock_across_dst_boundary() -> None:
    """DST transitions may change UTC deltas but not the local slot time."""

    from kazusa_ai_chatbot.calendar_scheduler import recurrence

    schedule = {
        "schedule_id": "schedule-dst",
        "start_at": "2026-04-03T00:00:00+00:00",
        "timezone": "Pacific/Auckland",
        "recurrence": {
            "kind": "daily_local_time",
            "local_time": "04:30",
        },
    }

    after_april_four_slot = recurrence.compute_next_run_at(
        schedule,
        after_utc="2026-04-03T16:00:00+00:00",
    )
    after_april_five_slot = recurrence.compute_next_run_at(
        schedule,
        after_utc="2026-04-04T17:00:00+00:00",
    )

    assert after_april_four_slot == "2026-04-04T16:30:00+00:00"
    assert after_april_five_slot == "2026-04-05T16:30:00+00:00"


def test_phase_period_offsets_reject_unfit_slot_budget() -> None:
    """Reflection phase fan-out must fit inside the named period values."""

    from kazusa_ai_chatbot.calendar_scheduler import recurrence

    with pytest.raises(ValueError, match="phase_period"):
        recurrence.compute_phase_period_offsets(
            {
                "period_seconds": 900,
                "min_slot_spacing_seconds": 60,
                "max_slots_per_period": 16,
            }
        )


def test_recurrence_rejects_unbounded_or_unknown_shapes() -> None:
    """Unknown recurrence payloads must fail closed before persistence."""

    from kazusa_ai_chatbot.calendar_scheduler import recurrence

    schedule = {
        "schedule_id": "schedule-unknown",
        "start_at": "2026-06-04T00:00:00+00:00",
        "recurrence": {"kind": "cron_expression", "value": "* * * * *"},
    }

    with pytest.raises(ValueError, match="recurrence"):
        recurrence.compute_next_run_at(
            schedule,
            after_utc="2026-06-04T00:00:00+00:00",
        )
