"""Deterministic tests for the character-local time context module."""

from __future__ import annotations

import logging

import pytest

from kazusa_ai_chatbot.time_context import (
    TimeContextDoc,
    build_character_time_context,
    format_history_for_llm,
    format_time_fields_for_llm,
    format_timestamp_for_llm,
    local_date_bounds_to_utc_iso,
    local_llm_time_to_utc_iso,
    structured_llm_time_to_utc_iso,
)

# ---------------------------------------------------------------------------
# build_character_time_context
# ---------------------------------------------------------------------------


def test_build_character_time_context_basic() -> None:
    """UTC midnight 2026-05-03 should be 12:00 local in Pacific/Auckland (NZST, UTC+12)."""
    ctx = build_character_time_context("2026-05-03T00:00:03.036410+00:00")

    assert isinstance(ctx, dict)
    assert ctx["current_local_datetime"] == "2026-05-03 12:00"
    assert ctx["current_local_weekday"] == "Sunday"


def test_build_character_time_context_z_suffix() -> None:
    """Z-suffixed UTC should be handled identically."""
    ctx = build_character_time_context("2026-05-03T00:00:03Z")

    assert ctx["current_local_datetime"] == "2026-05-03 12:00"
    assert ctx["current_local_weekday"] == "Sunday"


def test_build_character_time_context_offset_input() -> None:
    """A non-UTC offset-aware input should still be interpreted correctly."""
    # 2026-05-02T20:00:00-04:00 == 2026-05-03T00:00:00Z == 2026-05-03 12:00 NZST
    ctx = build_character_time_context("2026-05-02T20:00:00-04:00")

    assert ctx["current_local_datetime"] == "2026-05-03 12:00"


def test_build_character_time_context_none_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    """None input should fall back to current UTC and log a warning."""
    with caplog.at_level(logging.WARNING):
        ctx = build_character_time_context(None)

    assert "current_local_datetime" in ctx
    assert "current_local_weekday" in ctx
    # Should have logged a warning about invalid input
    assert any("fallback" in r.message.lower() or "invalid" in r.message.lower() for r in caplog.records)


def test_build_character_time_context_empty_string_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    """Empty string should fall back to current UTC and log a warning."""
    with caplog.at_level(logging.WARNING):
        ctx = build_character_time_context("")

    assert "current_local_datetime" in ctx
    assert "current_local_weekday" in ctx


def test_build_character_time_context_only_two_keys() -> None:
    """TimeContextDoc must contain exactly current_local_datetime and current_local_weekday."""
    ctx = build_character_time_context("2026-05-03T00:00:00+00:00")

    assert set(ctx.keys()) == {"current_local_datetime", "current_local_weekday"}


# ---------------------------------------------------------------------------
# format_timestamp_for_llm
# ---------------------------------------------------------------------------


def test_format_timestamp_none_returns_empty() -> None:
    assert format_timestamp_for_llm(None) == ""


def test_format_timestamp_empty_returns_empty() -> None:
    assert format_timestamp_for_llm("") == ""


def test_format_timestamp_utc_iso_converts_to_local() -> None:
    """Full UTC ISO with offset should become local naive YYYY-MM-DD HH:MM."""
    result = format_timestamp_for_llm("2026-05-03T00:00:03.036410+00:00")

    assert result == "2026-05-03 12:00"


def test_format_timestamp_z_suffix_converts() -> None:
    result = format_timestamp_for_llm("2026-05-03T00:00:03Z")

    assert result == "2026-05-03 12:00"


def test_format_timestamp_already_local_datetime_passthrough() -> None:
    """Already-formatted YYYY-MM-DD HH:MM should pass through unchanged."""
    result = format_timestamp_for_llm("2026-05-03 12:00")

    assert result == "2026-05-03 12:00"


def test_format_timestamp_already_local_date_passthrough() -> None:
    """Already-formatted YYYY-MM-DD should pass through unchanged."""
    result = format_timestamp_for_llm("2026-05-03")

    assert result == "2026-05-03"


def test_format_timestamp_naive_iso_with_t_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    """Naive ISO with T but no offset is ambiguous and should be rejected."""
    with caplog.at_level(logging.WARNING):
        result = format_timestamp_for_llm("2026-05-03T12:00:00")

    assert result == ""


def test_format_timestamp_natural_language_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    """Natural language timestamps are not handled by this helper."""
    with caplog.at_level(logging.WARNING):
        result = format_timestamp_for_llm("明天下午两点")

    assert result == ""


def test_format_timestamp_slash_date_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    """Slash-separated date is rejected."""
    with caplog.at_level(logging.WARNING):
        result = format_timestamp_for_llm("2026/05/03")

    assert result == ""


# ---------------------------------------------------------------------------
# format_history_for_llm
# ---------------------------------------------------------------------------


def test_format_history_converts_timestamps() -> None:
    rows = [
        {
            "role": "user",
            "display_name": "蚝爹油",
            "body_text": "你好",
            "timestamp": "2026-05-03T00:00:03.036410+00:00",
        },
        {
            "role": "assistant",
            "display_name": "千纱",
            "body_text": "你好啊",
            "timestamp": "2026-05-03T00:01:25.283613+00:00",
        },
    ]

    formatted = format_history_for_llm(rows)

    assert formatted[0]["timestamp"] == "2026-05-03 12:00"
    assert formatted[1]["timestamp"] == "2026-05-03 12:01"


def test_format_history_preserves_non_time_fields() -> None:
    rows = [
        {
            "role": "user",
            "display_name": "蚝爹油",
            "body_text": "你也记得一会儿多穿点别感冒了。",
            "timestamp": "2026-05-03T00:00:03+00:00",
            "reply_context": {"reply_to_display_name": "千纱"},
        },
    ]

    formatted = format_history_for_llm(rows)

    assert formatted[0]["body_text"] == rows[0]["body_text"]
    assert formatted[0]["display_name"] == rows[0]["display_name"]
    assert formatted[0]["role"] == rows[0]["role"]
    assert formatted[0]["reply_context"] == rows[0]["reply_context"]


def test_format_history_does_not_mutate_input() -> None:
    original_ts = "2026-05-03T00:00:03+00:00"
    rows = [{"timestamp": original_ts, "body_text": "test"}]

    format_history_for_llm(rows)

    assert rows[0]["timestamp"] == original_ts


def test_format_history_missing_timestamp_preserved() -> None:
    """Rows without timestamp should remain without timestamp."""
    rows = [{"body_text": "test", "role": "user"}]

    formatted = format_history_for_llm(rows)

    assert "timestamp" not in formatted[0]


# ---------------------------------------------------------------------------
# format_time_fields_for_llm
# ---------------------------------------------------------------------------


def test_format_time_fields_converts_explicit_fields_only() -> None:
    row = {
        "timestamp": "2026-05-03T00:00:03+00:00",
        "body_text": "hello",
        "created_at": "2026-05-03T01:30:00+00:00",
        "random_time_field": "2026-05-03T02:00:00+00:00",
    }

    result = format_time_fields_for_llm(row, ("timestamp", "created_at"))

    assert result["timestamp"] == "2026-05-03 12:00"
    assert result["created_at"] == "2026-05-03 13:30"
    assert result["random_time_field"] == "2026-05-03T02:00:00+00:00"
    assert result["body_text"] == "hello"


def test_format_time_fields_does_not_recurse() -> None:
    row = {
        "evidence": {
            "timestamp": "2026-05-03T00:00:03+00:00",
        },
    }

    result = format_time_fields_for_llm(row, ("timestamp",))

    assert result["evidence"]["timestamp"] == "2026-05-03T00:00:03+00:00"


def test_format_time_fields_does_not_mutate_input() -> None:
    original = {"timestamp": "2026-05-03T00:00:03+00:00", "fact": "test"}

    format_time_fields_for_llm(original, ("timestamp",))

    assert original["timestamp"] == "2026-05-03T00:00:03+00:00"


# ---------------------------------------------------------------------------
# local_llm_time_to_utc_iso
# ---------------------------------------------------------------------------


def test_local_to_utc_basic() -> None:
    """Local 2026-05-03 14:00 NZST => 2026-05-03T02:00:00+00:00."""
    result = local_llm_time_to_utc_iso("2026-05-03 14:00")

    assert result == "2026-05-03T02:00:00+00:00"


def test_local_to_utc_midnight() -> None:
    """Local midnight should map to previous UTC day."""
    result = local_llm_time_to_utc_iso("2026-05-03 00:00")

    assert result == "2026-05-02T12:00:00+00:00"


def test_local_to_utc_rejects_date_only() -> None:
    with pytest.raises(ValueError):
        local_llm_time_to_utc_iso("2026-05-03")


def test_local_to_utc_rejects_time_only() -> None:
    with pytest.raises(ValueError):
        local_llm_time_to_utc_iso("14:00")


def test_local_to_utc_rejects_seconds() -> None:
    with pytest.raises(ValueError):
        local_llm_time_to_utc_iso("2026-05-03 14:00:00")


def test_local_to_utc_rejects_t_separator() -> None:
    with pytest.raises(ValueError):
        local_llm_time_to_utc_iso("2026-05-03T14:00")


def test_local_to_utc_rejects_offset() -> None:
    with pytest.raises(ValueError):
        local_llm_time_to_utc_iso("2026-05-03 14:00+12:00")


def test_local_to_utc_rejects_z() -> None:
    with pytest.raises(ValueError):
        local_llm_time_to_utc_iso("2026-05-03 14:00Z")


def test_local_to_utc_rejects_utc_label() -> None:
    with pytest.raises(ValueError):
        local_llm_time_to_utc_iso("2026-05-03 14:00 UTC")


def test_local_to_utc_rejects_natural_language() -> None:
    with pytest.raises(ValueError):
        local_llm_time_to_utc_iso("明天下午两点")


# ---------------------------------------------------------------------------
# structured_llm_time_to_utc_iso
# ---------------------------------------------------------------------------


def test_structured_llm_time_accepts_local_contract() -> None:
    result = structured_llm_time_to_utc_iso("2026-05-03 14:00")

    assert result == "2026-05-03T02:00:00+00:00"


def test_structured_llm_time_accepts_legacy_offset_iso() -> None:
    result = structured_llm_time_to_utc_iso("2026-05-03T14:00:00+12:00")

    assert result == "2026-05-03T02:00:00+00:00"


def test_structured_llm_time_rejects_naive_iso() -> None:
    with pytest.raises(ValueError):
        structured_llm_time_to_utc_iso("2026-05-03T14:00:00")


def test_structured_llm_time_rejects_natural_language() -> None:
    with pytest.raises(ValueError):
        structured_llm_time_to_utc_iso("明天下午两点")


# ---------------------------------------------------------------------------
# local_date_bounds_to_utc_iso
# ---------------------------------------------------------------------------


def test_date_bounds_basic() -> None:
    """Local 2026-05-03 in Pacific/Auckland (UTC+12) should span UTC 2026-05-02T12:00 to 2026-05-03T12:00."""
    start, end = local_date_bounds_to_utc_iso("2026-05-03")

    assert start == "2026-05-02T12:00:00+00:00"
    assert end == "2026-05-03T12:00:00+00:00"


def test_date_bounds_rejects_datetime() -> None:
    with pytest.raises(ValueError):
        local_date_bounds_to_utc_iso("2026-05-03 14:00")


def test_date_bounds_rejects_iso_with_t() -> None:
    with pytest.raises(ValueError):
        local_date_bounds_to_utc_iso("2026-05-03T00:00")
