"""Deterministic tests for the canonical time boundary module."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kazusa_ai_chatbot.time_boundary import (
    LocalTimeContextDoc,
    build_turn_clock,
    build_turn_clock_from_storage_utc,
    format_storage_utc_fields_for_llm,
    format_storage_utc_for_llm,
    format_storage_utc_history_for_llm,
    local_date_bounds_to_storage_utc_iso,
    local_datetime_to_storage_utc_iso,
    local_llm_datetime_to_storage_utc_iso,
    local_time_context_from_storage_utc,
    normalize_storage_utc_iso,
    one_second_before_storage_utc_iso,
    parse_configured_local_datetime,
    parse_storage_utc_datetime,
    storage_utc_now,
    storage_utc_now_iso,
)


def test_storage_utc_now_returns_aware_utc_datetime() -> None:
    """Current storage clock values must be timezone-aware UTC instants."""

    now_utc = storage_utc_now()

    assert now_utc.tzinfo is timezone.utc


def test_storage_utc_now_iso_returns_utc_offset_string() -> None:
    """Current storage ISO strings must carry the explicit UTC offset."""

    now_utc_iso = storage_utc_now_iso()

    assert now_utc_iso.endswith("+00:00")
    assert parse_storage_utc_datetime(now_utc_iso).tzinfo is timezone.utc


def test_build_turn_clock_preserves_configured_local_input() -> None:
    """Configured local input must not be converted as though it were UTC."""

    turn_clock = build_turn_clock("2026-05-17 16:55:28.395")

    assert turn_clock["storage_timestamp_utc"] == (
        "2026-05-17T04:55:28.395000+00:00"
    )
    assert turn_clock["local_timestamp"] == "2026-05-17 16:55:28.395000"
    assert turn_clock["local_time_context"] == {
        "current_local_datetime": "2026-05-17 16:55",
        "current_local_weekday": "Sunday",
    }


def test_build_turn_clock_from_storage_utc_converts_to_configured_local() -> None:
    """Storage UTC input must be projected to the configured local wall-clock."""

    turn_clock = build_turn_clock_from_storage_utc(
        "2026-05-17T04:55:28.395000+00:00"
    )

    assert turn_clock["storage_timestamp_utc"] == (
        "2026-05-17T04:55:28.395000+00:00"
    )
    assert turn_clock["local_timestamp"] == "2026-05-17 16:55:28.395000"
    assert turn_clock["local_time_context"] == {
        "current_local_datetime": "2026-05-17 16:55",
        "current_local_weekday": "Sunday",
    }


def test_local_time_context_from_storage_utc_has_only_local_fields() -> None:
    """Prompt time context must stay local and small."""

    local_time_context = local_time_context_from_storage_utc(
        "2026-05-03T00:00:03.036410+00:00"
    )

    expected_keys = {"current_local_datetime", "current_local_weekday"}
    assert isinstance(local_time_context, dict)
    assert set(local_time_context.keys()) == expected_keys
    assert local_time_context["current_local_datetime"] == "2026-05-03 12:00"
    assert local_time_context["current_local_weekday"] == "Sunday"


def test_local_time_context_type_contract() -> None:
    """The public context type name must match the plan contract."""

    local_time_context: LocalTimeContextDoc = {
        "current_local_datetime": "2026-05-17 16:55",
        "current_local_weekday": "Sunday",
    }

    assert local_time_context["current_local_datetime"] == "2026-05-17 16:55"


def test_parse_storage_utc_datetime_accepts_utc_iso_and_z_suffix() -> None:
    """Storage parsing accepts UTC instants and normalizes them to UTC."""

    plus_zero = parse_storage_utc_datetime("2026-05-03T00:00:03.036410+00:00")
    z_suffix = parse_storage_utc_datetime("2026-05-03T00:00:03Z")

    assert plus_zero == datetime(
        2026,
        5,
        3,
        0,
        0,
        3,
        36410,
        tzinfo=timezone.utc,
    )
    assert z_suffix == datetime(2026, 5, 3, 0, 0, 3, tzinfo=timezone.utc)


def test_parse_storage_utc_datetime_rejects_non_utc_offsets() -> None:
    """A storage timestamp with a non-UTC offset is not storage UTC."""

    with pytest.raises(ValueError):
        parse_storage_utc_datetime("2026-05-03T12:00:00+12:00")


def test_parse_storage_utc_datetime_rejects_naive_values() -> None:
    """Naive datetimes are not valid storage instants."""

    with pytest.raises(ValueError):
        parse_storage_utc_datetime("2026-05-03T00:00:03")


def test_normalize_storage_utc_iso_returns_canonical_offset() -> None:
    """Z-suffixed UTC should normalize to the explicit +00:00 suffix."""

    result = normalize_storage_utc_iso("2026-05-03T00:00:03Z")

    assert result == "2026-05-03T00:00:03+00:00"


def test_parse_configured_local_datetime_accepts_plan_formats() -> None:
    """Configured local parsing accepts only local wall-clock forms."""

    minute = parse_configured_local_datetime("2026-05-17 16:55")
    second = parse_configured_local_datetime("2026-05-17 16:55:28")
    fractional_dot = parse_configured_local_datetime("2026-05-17 16:55:28.395")
    fractional_comma = parse_configured_local_datetime("2026-05-17 16:55:28,395")

    assert minute == datetime(2026, 5, 17, 16, 55)
    assert second == datetime(2026, 5, 17, 16, 55, 28)
    assert fractional_dot == datetime(2026, 5, 17, 16, 55, 28, 395000)
    assert fractional_comma == datetime(2026, 5, 17, 16, 55, 28, 395000)


def test_local_input_rejects_offset_utc_and_timezone_markers() -> None:
    """Configured local input must not accept timezone-aware forms."""

    invalid_values = [
        "2026-05-17T16:55:28",
        "2026-05-17 16:55Z",
        "2026-05-17 16:55 UTC",
        "2026-05-17 16:55+12:00",
        "2026-05-17 16:55-04:00",
        "2026-05-17 16:55 Pacific/Auckland",
        "2026-05-17",
        "tomorrow afternoon",
    ]

    for value in invalid_values:
        with pytest.raises(ValueError):
            parse_configured_local_datetime(value)


def test_local_datetime_to_storage_utc_iso_converts_configured_local() -> None:
    """Configured local wall-clock values convert to storage UTC."""

    result = local_datetime_to_storage_utc_iso("2026-05-17 16:55:28.395")

    assert result == "2026-05-17T04:55:28.395000+00:00"


def test_llm_local_datetime_to_storage_utc_accepts_exact_minute() -> None:
    """LLM-produced datetimes use the exact local minute contract."""

    result = local_llm_datetime_to_storage_utc_iso("2026-05-03 14:00")

    assert result == "2026-05-03T02:00:00+00:00"


def test_llm_local_datetime_to_storage_utc_rejects_offset_iso() -> None:
    """LLM output must not contain offset-bearing ISO timestamps."""

    invalid_values = [
        "2026-05-03T14:00:00+12:00",
        "2026-05-03T02:00:00+00:00",
        "2026-05-03 14:00+12:00",
        "2026-05-03 14:00Z",
        "2026-05-03 14:00 UTC",
        "2026-05-03 14:00:00",
    ]

    for value in invalid_values:
        with pytest.raises(ValueError):
            local_llm_datetime_to_storage_utc_iso(value)


def test_format_storage_utc_for_llm_converts_storage_utc_to_local() -> None:
    """Storage UTC values must become configured local model-facing strings."""

    result = format_storage_utc_for_llm("2026-05-03T00:00:03.036410+00:00")

    assert result == "2026-05-03 12:00"


def test_format_storage_utc_for_llm_allows_already_local_values() -> None:
    """Already local prompt strings may pass through unchanged."""

    minute = format_storage_utc_for_llm("2026-05-03 12:00")
    local_date = format_storage_utc_for_llm("2026-05-03")

    assert minute == "2026-05-03 12:00"
    assert local_date == "2026-05-03"


def test_format_storage_utc_for_llm_rejects_non_utc_offset() -> None:
    """Non-UTC offset strings must not pass through to prompt payloads."""

    result = format_storage_utc_for_llm("2026-05-03T12:00:00+12:00")

    assert result == ""


def test_format_storage_utc_history_for_llm_projects_timestamp_copy() -> None:
    """Conversation rows are copied with local prompt timestamp labels."""

    rows = [
        {
            "role": "user",
            "body_text": "hello",
            "timestamp": "2026-05-03T00:00:03.036410+00:00",
        },
        {
            "role": "assistant",
            "body_text": "hi",
            "timestamp": "2026-05-03T00:01:25.283613+00:00",
        },
    ]

    formatted = format_storage_utc_history_for_llm(rows)

    assert formatted[0]["timestamp"] == "2026-05-03 12:00"
    assert formatted[1]["timestamp"] == "2026-05-03 12:01"
    assert rows[0]["timestamp"] == "2026-05-03T00:00:03.036410+00:00"


def test_format_storage_utc_fields_for_llm_projects_explicit_fields() -> None:
    """Explicit timestamp fields are projected without recursive guessing."""

    row = {
        "timestamp": "2026-05-03T00:00:03+00:00",
        "created_at": "2026-05-03T01:30:00+00:00",
        "nested": {"timestamp": "2026-05-03T02:00:00+00:00"},
    }

    formatted = format_storage_utc_fields_for_llm(row, ("timestamp", "created_at"))

    assert formatted["timestamp"] == "2026-05-03 12:00"
    assert formatted["created_at"] == "2026-05-03 13:30"
    assert formatted["nested"]["timestamp"] == "2026-05-03T02:00:00+00:00"
    assert row["timestamp"] == "2026-05-03T00:00:03+00:00"


def test_local_date_bounds_to_storage_utc_iso_uses_configured_local_day() -> None:
    """A configured local date maps to UTC start and exclusive end bounds."""

    start_utc, end_utc = local_date_bounds_to_storage_utc_iso("2026-05-03")

    assert start_utc == "2026-05-02T12:00:00+00:00"
    assert end_utc == "2026-05-03T12:00:00+00:00"


def test_local_date_bounds_to_storage_utc_iso_rejects_datetime() -> None:
    """Date bounds accept exact local dates, not local datetimes."""

    with pytest.raises(ValueError):
        local_date_bounds_to_storage_utc_iso("2026-05-03 14:00")


def test_one_second_before_storage_utc_iso_subtracts_from_utc_instant() -> None:
    """Exclusive upper bounds can be converted to inclusive display bounds."""

    result = one_second_before_storage_utc_iso("2026-05-03T00:00:00+00:00")

    assert result == "2026-05-02T23:59:59+00:00"
