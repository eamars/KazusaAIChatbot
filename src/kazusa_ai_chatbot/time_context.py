"""Character-local time context for LLM prompt payloads.

Converts UTC timestamps to character-local, timezone-unaware strings
for model-facing payloads, and converts LLM-produced local times back
to UTC for storage and scheduling.

All callers must use the public functions here for prompt-time formatting.
Do not import private timezone internals or call ``zoneinfo.ZoneInfo``
directly for prompt-facing time conversion.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import TypedDict
from zoneinfo import ZoneInfo

from kazusa_ai_chatbot.config import CHARACTER_TIME_ZONE

logger = logging.getLogger(__name__)

_LOCAL_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
_LOCAL_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class TimeContextDoc(TypedDict):
    """Character-local time context for LLM prompt payloads.

    Contains only two fields: the current local datetime and weekday.
    The LLM can derive other date/time information from these values.
    """

    current_local_datetime: str
    current_local_weekday: str


def _get_tz() -> ZoneInfo:
    """Return the configured character timezone."""
    tz = ZoneInfo(CHARACTER_TIME_ZONE)
    return tz


def _parse_utc_instant(timestamp: str) -> datetime:
    """Parse an offset-aware ISO 8601 string into a UTC datetime.

    Args:
        timestamp: An offset-aware ISO 8601 string, including Z-suffixed.

    Returns:
        A timezone-aware datetime normalized to UTC.

    Raises:
        ValueError: If the string cannot be parsed as offset-aware ISO 8601.
    """
    normalized = timestamp.strip()
    if normalized.endswith("Z") or normalized.endswith("z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        raise ValueError(f"Timestamp is not offset-aware: {timestamp!r}")
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt


def build_character_time_context(timestamp: str | None) -> TimeContextDoc:
    """Build a character-local time context from a UTC turn timestamp.

    Args:
        timestamp: UTC ISO 8601 string from the request/turn. ``None`` or
            empty string triggers a fallback to current UTC with a warning.

    Returns:
        A ``TimeContextDoc`` with character-local datetime and weekday.
    """
    if not timestamp:
        logger.warning(
            f"Invalid or missing turn timestamp (got {timestamp!r}), "
            f"using current UTC as fallback"
        )
        utc_now = datetime.now(timezone.utc)
    else:
        try:
            utc_now = _parse_utc_instant(timestamp)
        except (ValueError, TypeError) as exc:
            logger.warning(
                f"Invalid turn timestamp {timestamp!r}: {exc}; "
                f"using current UTC as fallback"
            )
            utc_now = datetime.now(timezone.utc)

    local_dt = utc_now.astimezone(_get_tz())

    time_context: TimeContextDoc = {
        "current_local_datetime": local_dt.strftime("%Y-%m-%d %H:%M"),
        "current_local_weekday": local_dt.strftime("%A"),
    }
    return time_context


def format_timestamp_for_llm(timestamp: str | None) -> str:
    """Convert a single timestamp to a character-local, timezone-unaware string.

    Args:
        timestamp: A raw timestamp value from state or database rows.

    Returns:
        A local string in ``YYYY-MM-DD HH:MM`` or ``YYYY-MM-DD``
        format. Returns empty string for ``None``, empty, or invalid input.
    """
    if not timestamp or not isinstance(timestamp, str):
        return ""

    stripped = timestamp.strip()
    if not stripped:
        return ""

    # Already-formatted canonical local datetime
    if _LOCAL_DATETIME_RE.match(stripped):
        return stripped

    # Already-formatted canonical local date
    if _LOCAL_DATE_RE.match(stripped):
        return stripped

    # Attempt to parse as offset-aware ISO 8601
    try:
        utc_dt = _parse_utc_instant(stripped)
    except (ValueError, TypeError) as exc:
        logger.warning(
            f"Cannot format timestamp for LLM: {stripped!r} ({exc})"
        )
        return ""

    local_dt = utc_dt.astimezone(_get_tz())
    formatted_timestamp = local_dt.strftime("%Y-%m-%d %H:%M")
    return formatted_timestamp


def format_history_for_llm(rows: list[dict]) -> list[dict]:
    """Project chat history rows with local ``timestamp`` fields.

    Args:
        rows: Chat history rows from state or database. Each row is a dict
            with optional ``timestamp`` key.

    Returns:
        A new list with formatted ``timestamp`` values. Non-time fields are
        preserved exactly. Missing ``timestamp`` keys remain missing.
    """
    formatted_rows: list[dict] = []
    for row in rows:
        new_row = dict(row)
        if "timestamp" in new_row:
            new_row["timestamp"] = format_timestamp_for_llm(new_row["timestamp"])
        formatted_rows.append(new_row)
    return formatted_rows


def format_time_fields_for_llm(
    row: dict,
    time_fields: tuple[str, ...],
) -> dict:
    """Project one known row shape with explicitly named local time fields.

    This helper intentionally does not recurse and does not infer which fields
    are time fields. Callers must pass the timestamp keys owned by their source
    schema when constructing an LLM-facing payload.

    Args:
        row: One known-source row to project.
        time_fields: Exact field names in ``row`` that carry timestamps.

    Returns:
        A shallow copy of ``row`` with only the requested fields formatted.
    """
    formatted_row = dict(row)
    for field in time_fields:
        value = formatted_row.get(field)
        if isinstance(value, str):
            formatted_row[field] = format_timestamp_for_llm(value)

    return_value = formatted_row
    return return_value


def local_llm_time_to_utc_iso(value: str) -> str:
    """Convert a character-local ``YYYY-MM-DD HH:MM`` string to UTC ISO 8601.

    Args:
        value: Exactly ``YYYY-MM-DD HH:MM``. No other format is accepted.

    Returns:
        A timezone-aware UTC ISO string with ``+00:00`` suffix.

    Raises:
        ValueError: If the input does not match the required format exactly,
            or contains date-only, time-only, seconds, ``T``, offsets,
            timezone names, ``UTC``, ``Z``, or natural language.
    """
    stripped = value.strip() if isinstance(value, str) else ""

    if not _LOCAL_DATETIME_RE.match(stripped):
        raise ValueError(
            f"Expected exactly YYYY-MM-DD HH:MM, got: {value!r}"
        )

    # Additional rejection for content that sneaks past the regex
    if "T" in stripped or "Z" in stripped or "UTC" in stripped or "+" in stripped:
        raise ValueError(
            f"Input contains forbidden characters (T/Z/UTC/+): {value!r}"
        )

    naive_dt = datetime.strptime(stripped, "%Y-%m-%d %H:%M")
    local_dt = naive_dt.replace(tzinfo=_get_tz())
    utc_dt = local_dt.astimezone(timezone.utc)
    utc_iso = utc_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return utc_iso


def structured_llm_time_to_utc_iso(value: str) -> str:
    """Convert structured LLM time output to UTC ISO 8601.

    Accepts the new prompt contract, exact local ``YYYY-MM-DD HH:MM``. Also
    accepts legacy offset-aware ISO values so old tests or cached prompts do
    not force a behavior break. Naive ISO, date-only, time-only, and natural
    language values are rejected.

    Args:
        value: Structured time emitted by an LLM or legacy internal caller.

    Returns:
        A timezone-aware UTC ISO string with ``+00:00`` suffix.

    Raises:
        ValueError: If the value is not exact local time or offset-aware ISO.
    """
    stripped = value.strip() if isinstance(value, str) else ""
    try:
        utc_iso = local_llm_time_to_utc_iso(stripped)
        return utc_iso
    except ValueError:
        pass

    parsed = _parse_utc_instant(stripped)
    return_value = parsed.isoformat()
    return return_value


def local_date_bounds_to_utc_iso(local_date: str) -> tuple[str, str]:
    """Convert a local date to UTC start-of-day (inclusive) and end-of-day (exclusive) bounds.

    Args:
        local_date: Exactly ``YYYY-MM-DD``.

    Returns:
        A tuple of ``(start_utc_iso, end_utc_iso)`` for the local day
        converted to UTC.

    Raises:
        ValueError: If the input does not match ``YYYY-MM-DD`` exactly.
    """
    stripped = local_date.strip() if isinstance(local_date, str) else ""

    if not _LOCAL_DATE_RE.match(stripped):
        raise ValueError(
            f"Expected exactly YYYY-MM-DD, got: {local_date!r}"
        )

    # Reject anything with time-like or offset-like content
    if " " in stripped or "T" in stripped:
        raise ValueError(
            f"Input contains time component: {local_date!r}"
        )

    tz = _get_tz()
    naive_start = datetime.strptime(stripped, "%Y-%m-%d")
    local_start = naive_start.replace(tzinfo=tz)
    local_end = local_start + timedelta(days=1)

    utc_start = local_start.astimezone(timezone.utc)
    utc_end = local_end.astimezone(timezone.utc)

    start_iso = utc_start.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    end_iso = utc_end.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return start_iso, end_iso
