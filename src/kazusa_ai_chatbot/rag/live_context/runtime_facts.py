"""Runtime date and time fact helpers for live context."""

from __future__ import annotations

import re


_RUNTIME_DATETIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")

_RUNTIME_WEEKDAY_PATTERN = re.compile(r"^[A-Za-z]+$")

_RUNTIME_FACT_TYPES = {
    "current_time",
    "current_date",
    "current_weekday",
}

def _validated_runtime_time_context(
    raw_time_context: object,
) -> dict[str, str] | None:
    """Validate the sanitized runtime time context consumed by live facts.

    Args:
        raw_time_context: Runtime state payload from the current turn.

    Returns:
        Sanitized datetime and weekday fields, or ``None`` when malformed.
    """

    if not isinstance(raw_time_context, dict):
        return None

    raw_local_datetime = raw_time_context.get("current_local_datetime")
    raw_local_weekday = raw_time_context.get("current_local_weekday")
    if not isinstance(raw_local_datetime, str):
        return None
    if not isinstance(raw_local_weekday, str):
        return None

    local_datetime = raw_local_datetime.strip()
    local_weekday = raw_local_weekday.strip()
    if not local_datetime or not local_weekday:
        return None
    if _RUNTIME_DATETIME_PATTERN.fullmatch(local_datetime) is None:
        return None
    if _RUNTIME_WEEKDAY_PATTERN.fullmatch(local_weekday) is None:
        return None

    time_context = {
        "current_local_datetime": local_datetime,
        "current_local_weekday": local_weekday,
    }
    return time_context

def _runtime_selected_summary(
    fact_type: str,
    time_context: dict[str, str],
) -> str:
    """Build the direct evidence sentence for a runtime-backed live fact.

    Args:
        fact_type: Runtime-backed live fact type.
        time_context: Validated sanitized time context.

    Returns:
        A compact direct-evidence sentence for the evaluator and projection.
    """

    local_datetime = time_context["current_local_datetime"]
    local_weekday = time_context["current_local_weekday"]
    local_date, _, _ = local_datetime.partition(" ")

    if fact_type == "current_time":
        return_value = f"当前本地时间是 {local_datetime}，{local_weekday}。"
        return return_value
    if fact_type == "current_date":
        return_value = f"当前本地日期是 {local_date}，{local_weekday}。"
        return return_value
    return_value = f"当前本地星期是 {local_weekday}，日期是 {local_date}。"
    return return_value
