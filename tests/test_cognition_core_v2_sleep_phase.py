"""Sleep-phase projector, shared clock helpers, and window consistency."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    CHARACTER_SLEEP_PHASE_IN_WINDOW,
    CHARACTER_SLEEP_PHASE_OUTSIDE,
    CHARACTER_SLEEP_PHASE_WAKE_PREP,
    project_character_sleep_phase,
)
from kazusa_ai_chatbot.cognition_core_v2 import (
    run_character_morning_refresh,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
)
from kazusa_ai_chatbot.self_cognition.sleep_period import (
    is_self_cognition_sleep_period,
)
from kazusa_ai_chatbot.time_boundary import (
    local_minutes_in_zone,
    local_period_bounds,
)


def _instant(hour: int, minute: int = 0) -> datetime:
    """Build one aware UTC instant on a fixed test day."""

    return datetime(2026, 5, 13, hour, minute, tzinfo=timezone.utc)


def test_sleep_phase_same_day_window_boundaries() -> None:
    """In-window labels cover the half-open start-to-end window."""

    assert project_character_sleep_phase(
        _instant(1, 59),
        sleep_local_period="02:00-12:00",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_OUTSIDE
    assert project_character_sleep_phase(
        _instant(2, 0),
        sleep_local_period="02:00-12:00",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_IN_WINDOW
    assert project_character_sleep_phase(
        _instant(11, 29),
        sleep_local_period="02:00-12:00",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_IN_WINDOW
    assert project_character_sleep_phase(
        _instant(11, 30),
        sleep_local_period="02:00-12:00",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_WAKE_PREP
    assert project_character_sleep_phase(
        _instant(11, 59),
        sleep_local_period="02:00-12:00",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_WAKE_PREP
    assert project_character_sleep_phase(
        _instant(12, 0),
        sleep_local_period="02:00-12:00",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_OUTSIDE


def test_sleep_phase_overnight_window_wraps_midnight() -> None:
    """Overnight windows keep wake prep adjacent to the exclusive end."""

    assert project_character_sleep_phase(
        _instant(23, 30),
        sleep_local_period="23:30-07:30",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_IN_WINDOW
    assert project_character_sleep_phase(
        _instant(6, 59),
        sleep_local_period="23:30-07:30",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_IN_WINDOW
    assert project_character_sleep_phase(
        _instant(7, 0),
        sleep_local_period="23:30-07:30",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_WAKE_PREP
    assert project_character_sleep_phase(
        _instant(7, 29),
        sleep_local_period="23:30-07:30",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_WAKE_PREP
    assert project_character_sleep_phase(
        _instant(7, 30),
        sleep_local_period="23:30-07:30",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_OUTSIDE
    assert project_character_sleep_phase(
        _instant(12, 0),
        sleep_local_period="23:30-07:30",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    ) == CHARACTER_SLEEP_PHASE_OUTSIDE


def test_sleep_phase_empty_period_is_always_outside() -> None:
    """An empty sleep period disables the window entirely."""

    phase = project_character_sleep_phase(
        _instant(6, 0),
        sleep_local_period="",
        character_time_zone="UTC",
        wake_prep_minutes=30,
    )

    assert phase == CHARACTER_SLEEP_PHASE_OUTSIDE


def test_sleep_phase_rejects_invalid_domain_input() -> None:
    """Out-of-domain projector input raises ValueError."""

    with pytest.raises(ValueError):
        project_character_sleep_phase(
            datetime(2026, 5, 13, 6, 0),
            sleep_local_period="02:00-12:00",
            character_time_zone="UTC",
            wake_prep_minutes=30,
        )
    with pytest.raises(ValueError):
        project_character_sleep_phase(
            _instant(6, 0),
            sleep_local_period="02:00",
            character_time_zone="UTC",
            wake_prep_minutes=30,
        )
    for invalid_wake_prep in (0, -5, 1.5, True):
        with pytest.raises(ValueError):
            project_character_sleep_phase(
                _instant(6, 0),
                sleep_local_period="02:00-12:00",
                character_time_zone="UTC",
                wake_prep_minutes=invalid_wake_prep,
            )
    with pytest.raises(ValueError):
        project_character_sleep_phase(
            _instant(6, 0),
            sleep_local_period="02:00-12:00",
            character_time_zone="Not/AZone",
            wake_prep_minutes=30,
        )


@pytest.mark.parametrize(
    ("sleep_local_period", "wake_prep_minutes"),
    [
        ("02:00-12:00", 30),
        ("23:30-07:30", 30),
        ("00:00-23:59", 15),
    ],
)
def test_sleep_phase_union_matches_self_cognition_predicate(
    sleep_local_period: str,
    wake_prep_minutes: int,
) -> None:
    """The two in-window labels cover exactly the self-cognition window."""

    day_start = datetime(2026, 5, 13, 0, 0, tzinfo=timezone.utc)
    for minute_offset in range(24 * 60):
        instant = day_start + timedelta(minutes=minute_offset)
        phase = project_character_sleep_phase(
            instant,
            sleep_local_period=sleep_local_period,
            character_time_zone="UTC",
            wake_prep_minutes=wake_prep_minutes,
        )
        sleeping = is_self_cognition_sleep_period(
            instant,
            sleep_local_period=sleep_local_period,
            character_time_zone="UTC",
        )
        assert (phase in {
            CHARACTER_SLEEP_PHASE_IN_WINDOW,
            CHARACTER_SLEEP_PHASE_WAKE_PREP,
        }) is sleeping


def test_local_period_bounds_parses_exact_hh_mm_text() -> None:
    """Shared period parsing returns local minutes after midnight."""

    assert local_period_bounds("02:00-12:00") == (120, 720)
    assert local_period_bounds("23:30-07:30") == (1410, 450)


def test_local_period_bounds_rejects_invalid_text() -> None:
    """Invalid or equal bounds are rejected by the shared parser."""

    for invalid_period in (
        "02:00",
        "02:00-02:00",
        "24:00-12:00",
        "02-00-12:00",
        "02:0-12:00",
    ):
        with pytest.raises(ValueError):
            local_period_bounds(invalid_period)


def test_local_minutes_in_zone_projects_aware_instants() -> None:
    """Local minutes projection uses the requested IANA timezone."""

    assert local_minutes_in_zone(
        _instant(0, 30),
        time_zone="UTC",
    ) == 30
    assert local_minutes_in_zone(
        _instant(0, 30),
        time_zone="Pacific/Auckland",
    ) == 750

    with pytest.raises(ValueError):
        local_minutes_in_zone(
            datetime(2026, 5, 13, 0, 30),
            time_zone="UTC",
        )


def test_morning_refresh_requires_integer_elapsed_seconds() -> None:
    """The public refresh result keeps its declared integer contract."""

    state = build_character_production_state(
        updated_at='2026-05-13T00:00:00Z',
    )
    for invalid_elapsed in (1.5, True, -1):
        with pytest.raises(ValueError):
            run_character_morning_refresh(
                state,
                elapsed_sleep_seconds=invalid_elapsed,
                updated_at='2026-05-13T08:00:00Z',
            )
