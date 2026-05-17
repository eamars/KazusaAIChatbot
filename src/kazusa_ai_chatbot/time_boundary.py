"""Canonical boundary for storage UTC and configured local time.

Storage timestamps are aware UTC datetimes or ISO strings with ``+00:00``.
Configured local timestamps are timezone-unaware wall-clock strings produced
from ``CHARACTER_TIME_ZONE`` and are safe to place in model-facing payloads.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import TypedDict
from zoneinfo import ZoneInfo

from kazusa_ai_chatbot.config import CHARACTER_TIME_ZONE

_LOCAL_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LOCAL_MINUTE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
_LOCAL_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} "
    r"\d{2}:\d{2}"
    r"(?::\d{2}(?:[\.,]\d{1,6})?)?$"
)


class LocalTimeContextDoc(TypedDict):
    """Configured-local time context for model-facing payloads."""

    current_local_datetime: str
    current_local_weekday: str


class TurnClock(TypedDict):
    """Single-turn storage and configured-local timestamp bundle."""

    storage_timestamp_utc: str
    local_timestamp: str
    local_time_context: LocalTimeContextDoc


def storage_utc_now() -> datetime:
    """Return the current storage instant as an aware UTC datetime."""

    now_utc = datetime.now(timezone.utc)
    return now_utc


def storage_utc_now_iso() -> str:
    """Return the current storage instant as canonical UTC ISO text."""

    now_utc = storage_utc_now()
    now_utc_iso = now_utc.isoformat()
    return now_utc_iso


def parse_storage_utc_datetime(value: str) -> datetime:
    """Parse a storage UTC timestamp with only ``+00:00`` or ``Z`` allowed.

    Args:
        value: Offset-aware storage timestamp string.

    Returns:
        A timezone-aware datetime normalized to ``timezone.utc``.

    Raises:
        ValueError: If the value is not parseable storage UTC.
    """

    stripped = _stripped_string(value, "storage UTC timestamp")
    if stripped.endswith(("Z", "z")):
        normalized = f"{stripped[:-1]}+00:00"
    elif stripped.endswith("+00:00"):
        normalized = stripped
    else:
        raise ValueError(
            f"Expected storage UTC timestamp with +00:00 or Z: {value!r}"
        )

    parsed_datetime = datetime.fromisoformat(normalized)
    if parsed_datetime.tzinfo is None:
        raise ValueError(f"Storage timestamp is not timezone-aware: {value!r}")

    parsed_offset = parsed_datetime.utcoffset()
    if parsed_offset != timedelta(0):
        raise ValueError(f"Storage timestamp is not UTC: {value!r}")

    storage_datetime_utc = parsed_datetime.astimezone(timezone.utc)
    return storage_datetime_utc


def normalize_storage_utc_iso(value: str) -> str:
    """Normalize storage UTC text to ``datetime.isoformat()`` with ``+00:00``.

    Args:
        value: Storage UTC timestamp string accepted by
            ``parse_storage_utc_datetime``.

    Returns:
        Canonical UTC ISO timestamp with the explicit ``+00:00`` suffix.
    """

    storage_datetime_utc = parse_storage_utc_datetime(value)
    normalized_iso = storage_datetime_utc.isoformat()
    return normalized_iso


def parse_configured_local_datetime(value: str) -> datetime:
    """Parse configured local wall-clock text without timezone information.

    Args:
        value: Local wall-clock text in one of the supported date-time forms.

    Returns:
        A naive datetime representing configured local wall-clock time.

    Raises:
        ValueError: If the value contains timezone markers, offsets, ISO ``T``,
            date-only text, or natural language.
    """

    stripped = _stripped_string(value, "configured local datetime")
    if not _LOCAL_DATETIME_RE.match(stripped):
        raise ValueError(
            f"Expected configured local datetime, got: {value!r}"
        )

    normalized = stripped.replace(",", ".")
    parsed_local_datetime = datetime.fromisoformat(normalized)
    if parsed_local_datetime.tzinfo is not None:
        raise ValueError(
            f"Configured local datetime must not include timezone: {value!r}"
        )

    return parsed_local_datetime


def local_datetime_to_storage_utc_iso(value: str) -> str:
    """Convert configured local wall-clock text to canonical storage UTC ISO.

    Args:
        value: Configured local datetime string accepted by
            ``parse_configured_local_datetime``.

    Returns:
        Canonical storage UTC timestamp string.
    """

    local_datetime = parse_configured_local_datetime(value)
    storage_datetime_utc = _local_naive_to_storage_utc(local_datetime)
    storage_datetime_utc_iso = storage_datetime_utc.isoformat()
    return storage_datetime_utc_iso


def build_turn_clock(local_timestamp: str | None = None) -> TurnClock:
    """Build one turn's storage UTC and configured-local time fields.

    Args:
        local_timestamp: Optional configured local timestamp supplied by a
            caller. When omitted, one storage UTC clock read supplies both UTC
            and local values.

    Returns:
        A ``TurnClock`` containing storage UTC, local timestamp, and local time
        context derived from the same instant.
    """

    if local_timestamp:
        local_datetime = parse_configured_local_datetime(local_timestamp)
        storage_datetime_utc = _local_naive_to_storage_utc(local_datetime)
        turn_clock = _turn_clock_from_datetimes(
            storage_datetime_utc,
            local_datetime,
        )
        return turn_clock

    storage_datetime_utc = storage_utc_now()
    turn_clock = _turn_clock_from_storage_datetime(storage_datetime_utc)
    return turn_clock


def build_turn_clock_from_storage_utc(
    storage_timestamp_utc: str,
) -> TurnClock:
    """Build turn clock fields from an existing storage UTC timestamp.

    Args:
        storage_timestamp_utc: Storage UTC timestamp string.

    Returns:
        A ``TurnClock`` with normalized storage UTC and configured-local
        projections.
    """

    storage_datetime_utc = parse_storage_utc_datetime(storage_timestamp_utc)
    turn_clock = _turn_clock_from_storage_datetime(storage_datetime_utc)
    return turn_clock


def local_time_context_from_storage_utc(
    storage_timestamp_utc: str,
) -> LocalTimeContextDoc:
    """Project a storage UTC timestamp to configured-local prompt context.

    Args:
        storage_timestamp_utc: Storage UTC timestamp string.

    Returns:
        Configured-local context with minute precision and weekday.
    """

    storage_datetime_utc = parse_storage_utc_datetime(storage_timestamp_utc)
    local_datetime = _storage_utc_to_local_naive(storage_datetime_utc)
    local_time_context = _local_time_context_from_local_datetime(
        local_datetime,
    )
    return local_time_context


def format_storage_utc_for_llm(value: str | None) -> str:
    """Project storage UTC text to configured-local text for LLM payloads.

    Args:
        value: Optional storage UTC timestamp, local ``YYYY-MM-DD HH:MM``, or
            local ``YYYY-MM-DD`` text.

    Returns:
        Configured-local ``YYYY-MM-DD HH:MM`` text, pass-through local date or
        minute text, or an empty string for invalid and ambiguous input.
    """

    if value is None:
        formatted_value = ""
        return formatted_value

    if not isinstance(value, str):
        formatted_value = ""
        return formatted_value

    stripped = value.strip()
    if not stripped:
        formatted_value = ""
        return formatted_value

    if _LOCAL_MINUTE_RE.match(stripped) or _LOCAL_DATE_RE.match(stripped):
        formatted_value = stripped
        return formatted_value

    try:
        storage_datetime_utc = parse_storage_utc_datetime(stripped)
    except ValueError:
        formatted_value = ""
        return formatted_value

    local_datetime = _storage_utc_to_local_naive(storage_datetime_utc)
    formatted_value = local_datetime.strftime("%Y-%m-%d %H:%M")
    return formatted_value


def format_storage_utc_history_for_llm(rows: list[dict]) -> list[dict]:
    """Shallow-copy history rows and project top-level ``timestamp`` fields.

    Args:
        rows: History row dictionaries. Only a top-level ``timestamp`` key is
            interpreted as a time field.

    Returns:
        A new list of shallow-copied rows.
    """

    formatted_rows: list[dict] = []
    for row in rows:
        formatted_row = dict(row)
        if "timestamp" in formatted_row:
            raw_timestamp = formatted_row["timestamp"]
            if isinstance(raw_timestamp, str):
                timestamp_value = raw_timestamp
            else:
                timestamp_value = None
            formatted_row["timestamp"] = format_storage_utc_for_llm(
                timestamp_value,
            )
        formatted_rows.append(formatted_row)

    return formatted_rows


def format_storage_utc_fields_for_llm(
    row: dict,
    time_fields: tuple[str, ...],
) -> dict:
    """Shallow-copy a row and project explicitly named top-level time fields.

    Args:
        row: Source row to copy.
        time_fields: Exact top-level field names to format.

    Returns:
        A shallow copy with only the named fields projected for LLM payloads.
    """

    formatted_row = dict(row)
    for field_name in time_fields:
        if field_name in formatted_row:
            raw_value = formatted_row[field_name]
            if isinstance(raw_value, str):
                timestamp_value = raw_value
            else:
                timestamp_value = None
            formatted_row[field_name] = format_storage_utc_for_llm(
                timestamp_value,
            )

    return formatted_row


def local_llm_datetime_to_storage_utc_iso(value: str) -> str:
    """Convert exact LLM local minute text to canonical storage UTC ISO.

    Args:
        value: Exact configured local ``YYYY-MM-DD HH:MM`` text.

    Returns:
        Canonical storage UTC timestamp string.

    Raises:
        ValueError: If the value contains seconds, ISO separators, offsets,
            timezone markers, date-only text, or natural language.
    """

    stripped = _stripped_string(value, "LLM local datetime")
    if not _LOCAL_MINUTE_RE.match(stripped):
        raise ValueError(f"Expected exact YYYY-MM-DD HH:MM, got: {value!r}")

    local_datetime = datetime.strptime(stripped, "%Y-%m-%d %H:%M")
    storage_datetime_utc = _local_naive_to_storage_utc(local_datetime)
    storage_datetime_utc_iso = storage_datetime_utc.isoformat()
    return storage_datetime_utc_iso


def local_date_bounds_to_storage_utc_iso(
    local_date: str,
) -> tuple[str, str]:
    """Convert a configured local date to UTC start and exclusive end bounds.

    Args:
        local_date: Exact configured local ``YYYY-MM-DD`` date text.

    Returns:
        ``(start_utc_iso, exclusive_end_utc_iso)`` for the local day.

    Raises:
        ValueError: If the value is not an exact local date.
    """

    stripped = _stripped_string(local_date, "configured local date")
    if not _LOCAL_DATE_RE.match(stripped):
        raise ValueError(f"Expected exact YYYY-MM-DD, got: {local_date!r}")

    parsed_date = date.fromisoformat(stripped)
    local_start = datetime(
        parsed_date.year,
        parsed_date.month,
        parsed_date.day,
    )
    local_end = local_start + timedelta(days=1)
    start_utc = _local_naive_to_storage_utc(local_start)
    exclusive_end_utc = _local_naive_to_storage_utc(local_end)
    bounds = (start_utc.isoformat(), exclusive_end_utc.isoformat())
    return bounds


def one_second_before_storage_utc_iso(timestamp_utc: str) -> str:
    """Subtract one second from a storage UTC instant.

    Args:
        timestamp_utc: Storage UTC timestamp string.

    Returns:
        Canonical storage UTC timestamp one second earlier.
    """

    storage_datetime_utc = parse_storage_utc_datetime(timestamp_utc)
    previous_datetime_utc = storage_datetime_utc - timedelta(seconds=1)
    previous_timestamp_utc = previous_datetime_utc.isoformat()
    return previous_timestamp_utc


def _stripped_string(value: str, label: str) -> str:
    """Validate and strip public string input."""

    if not isinstance(value, str):
        raise ValueError(f"Expected {label} string, got: {value!r}")

    stripped = value.strip()
    if not stripped:
        raise ValueError(f"Expected non-empty {label}, got: {value!r}")

    return stripped


def _local_naive_to_storage_utc(local_datetime: datetime) -> datetime:
    """Attach configured timezone to a local datetime and convert to UTC."""

    local_timezone = ZoneInfo(CHARACTER_TIME_ZONE)
    aware_local_datetime = local_datetime.replace(tzinfo=local_timezone)
    storage_datetime_utc = aware_local_datetime.astimezone(timezone.utc)
    return storage_datetime_utc


def _storage_utc_to_local_naive(storage_datetime_utc: datetime) -> datetime:
    """Project an aware storage UTC datetime to configured local wall-clock."""

    local_timezone = ZoneInfo(CHARACTER_TIME_ZONE)
    aware_local_datetime = storage_datetime_utc.astimezone(local_timezone)
    local_datetime = aware_local_datetime.replace(tzinfo=None)
    return local_datetime


def _format_local_timestamp(local_datetime: datetime) -> str:
    """Format configured local turn timestamps with optional microseconds."""

    local_timestamp = local_datetime.strftime("%Y-%m-%d %H:%M:%S")
    if local_datetime.microsecond:
        local_timestamp = (
            f"{local_timestamp}.{local_datetime.microsecond:06d}"
        )

    return local_timestamp


def _local_time_context_from_local_datetime(
    local_datetime: datetime,
) -> LocalTimeContextDoc:
    """Build model-facing context from configured local wall-clock time."""

    local_time_context: LocalTimeContextDoc = {
        "current_local_datetime": local_datetime.strftime("%Y-%m-%d %H:%M"),
        "current_local_weekday": local_datetime.strftime("%A"),
    }
    return local_time_context


def _turn_clock_from_datetimes(
    storage_datetime_utc: datetime,
    local_datetime: datetime,
) -> TurnClock:
    """Build a turn clock from already paired UTC and local datetimes."""

    local_timestamp = _format_local_timestamp(local_datetime)
    local_time_context = _local_time_context_from_local_datetime(
        local_datetime,
    )
    turn_clock: TurnClock = {
        "storage_timestamp_utc": storage_datetime_utc.isoformat(),
        "local_timestamp": local_timestamp,
        "local_time_context": local_time_context,
    }
    return turn_clock


def _turn_clock_from_storage_datetime(
    storage_datetime_utc: datetime,
) -> TurnClock:
    """Build a turn clock by projecting storage UTC to configured local time."""

    local_datetime = _storage_utc_to_local_naive(storage_datetime_utc)
    turn_clock = _turn_clock_from_datetimes(storage_datetime_utc, local_datetime)
    return turn_clock
