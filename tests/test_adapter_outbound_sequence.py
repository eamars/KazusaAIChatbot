"""Tests for adapter-owned outbound message sequencing helpers."""

from adapters.outbound_sequence import followup_delay_seconds


def test_followup_delay_seconds_uses_minimum_for_short_text() -> None:
    """Short follow-ups should still have a small readable pause."""

    delay = followup_delay_seconds("ok")

    assert delay == 1.0


def test_followup_delay_seconds_uses_visible_stripped_length() -> None:
    """Surrounding whitespace should not make follow-up delay longer."""

    delay = followup_delay_seconds("  " + ("x" * 24) + "  ")

    assert delay == 2.0


def test_followup_delay_seconds_clamps_long_text() -> None:
    """Long follow-ups should not hold adapter delivery open indefinitely."""

    delay = followup_delay_seconds("x" * 200)

    assert delay == 5.0


def test_followup_delay_seconds_clamps_empty_text() -> None:
    """Empty follow-up strings should use the minimum delay."""

    delay = followup_delay_seconds("   ")

    assert delay == 1.0
