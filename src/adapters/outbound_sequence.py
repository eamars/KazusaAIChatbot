"""Shared helpers for adapter-owned outbound message sequences."""

FOLLOWUP_MIN_DELAY_SECONDS = 1.0
FOLLOWUP_MAX_DELAY_SECONDS = 5.0
FOLLOWUP_VISIBLE_CHARS_PER_SECOND = 12.0


def followup_delay_seconds(text: str) -> float:
    """Return the deterministic pause before sending one follow-up message."""

    visible_length = len(text.strip())
    raw_delay = visible_length / FOLLOWUP_VISIBLE_CHARS_PER_SECOND
    clamped_delay = min(
        FOLLOWUP_MAX_DELAY_SECONDS,
        max(FOLLOWUP_MIN_DELAY_SECONDS, raw_delay),
    )
    return clamped_delay
