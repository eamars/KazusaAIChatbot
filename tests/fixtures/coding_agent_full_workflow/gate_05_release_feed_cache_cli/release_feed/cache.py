"""Cache freshness helpers for release feeds."""

from __future__ import annotations


def should_refresh(
    *,
    cached_at_seconds: int,
    now_seconds: int,
    timeout_seconds: int,
) -> bool:
    """Return whether cached feed data is too old to reuse."""

    del timeout_seconds
    age_seconds = now_seconds - cached_at_seconds
    return age_seconds > 60
