"""Read-only selection of monitored conversation channels."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from kazusa_ai_chatbot.db.conversation_reflection import (
    explain_monitored_channel_query,
    list_recent_character_message_channels,
    list_reflection_scope_messages,
)
from kazusa_ai_chatbot.reflection_cycle.models import (
    READONLY_REFLECTION_FALLBACK_LOOKBACK_HOURS,
    READONLY_REFLECTION_MAX_MESSAGES_PER_SCOPE,
    READONLY_REFLECTION_MAX_SCOPES,
    READONLY_REFLECTION_MONITOR_ELIGIBILITY_HOURS,
    ReflectionInputSet,
    ReflectionScopeInput,
)


def normalize_utc_datetime(value: datetime | None) -> datetime:
    """Return a timezone-aware UTC timestamp for deterministic callers."""

    if value is None:
        return_value = datetime.now(timezone.utc)
        return return_value
    if value.tzinfo is None:
        return_value = value.replace(tzinfo=timezone.utc)
        return return_value
    return_value = value.astimezone(timezone.utc)
    return return_value


def isoformat_utc(value: datetime) -> str:
    """Render a UTC datetime in the repository's ISO timestamp shape."""

    normalized = normalize_utc_datetime(value)
    return_value = normalized.isoformat()
    return return_value


def build_scope_ref(platform: str, platform_channel_id: str, channel_type: str) -> str:
    """Build a stable non-identifying reference for a conversation scope."""

    raw = f"{platform}:{platform_channel_id}:{channel_type}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return_value = f"scope_{digest}"
    return return_value


def select_monitored_channel_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort monitored channel rows by latest character message timestamp."""

    selected_rows: list[dict[str, Any]] = []
    for row in rows:
        last_character_message = str(row["last_character_message_timestamp"])
        if not last_character_message:
            continue
        selected_rows.append(row)
    selected_rows.sort(
        key=lambda item: str(item["last_character_message_timestamp"]),
        reverse=True,
    )
    return_value = selected_rows[:READONLY_REFLECTION_MAX_SCOPES]
    return return_value


async def collect_reflection_inputs(
    *,
    lookback_hours: int = 24,
    now: datetime | None = None,
) -> ReflectionInputSet:
    """Collect read-only reflection inputs from recent conversation history.

    Args:
        lookback_hours: Requested message evaluation window.
        now: Optional deterministic clock value.

    Returns:
        A read-only input set. If no monitored channel exists inside the
        current monitor window, a bounded recent fallback search is attempted.
    """

    effective_now = normalize_utc_datetime(now)
    requested_start_dt = effective_now - timedelta(hours=lookback_hours)
    monitor_start_dt = effective_now - timedelta(
        hours=READONLY_REFLECTION_MONITOR_ELIGIBILITY_HOURS
    )
    requested_start = isoformat_utc(requested_start_dt)
    monitor_start = isoformat_utc(monitor_start_dt)
    requested_end = isoformat_utc(effective_now)

    query_result = await list_recent_character_message_channels(
        start_timestamp=monitor_start,
        end_timestamp=requested_end,
        limit=READONLY_REFLECTION_MAX_SCOPES,
    )

    fallback_used = False
    fallback_reason = ""
    effective_start = requested_start
    rows = query_result["rows"]
    channel_rows = select_monitored_channel_rows(rows)
    query_diagnostics = {
        f"requested_{key}": value
        for key, value in query_result["diagnostics"].items()
    }

    if not channel_rows:
        fallback_used = True
        fallback_start_dt = effective_now - timedelta(
            hours=READONLY_REFLECTION_FALLBACK_LOOKBACK_HOURS
        )
        effective_start = isoformat_utc(fallback_start_dt)
        fallback_reason = (
            "No monitored channel had a character message inside the monitor "
            "window; selected nearest recent character-message channel from "
            "the bounded fallback window."
        )
        fallback_query_result = await list_recent_character_message_channels(
            start_timestamp=effective_start,
            end_timestamp=requested_end,
            limit=READONLY_REFLECTION_MAX_SCOPES,
        )
        fallback_rows = fallback_query_result["rows"]
        channel_rows = select_monitored_channel_rows(fallback_rows)
        for key, value in fallback_query_result["diagnostics"].items():
            query_diagnostics[f"fallback_{key}"] = value

    query_diagnostics["explain_summary"] = await explain_monitored_channel_query(
        start_timestamp=monitor_start if not fallback_used else effective_start,
        end_timestamp=requested_end,
        limit=READONLY_REFLECTION_MAX_SCOPES,
    )
    selected_scopes = await _fetch_scope_messages(
        channel_rows=channel_rows,
        start_timestamp=effective_start,
        end_timestamp=requested_end,
    )
    input_set = ReflectionInputSet(
        lookback_hours=lookback_hours,
        requested_start=requested_start,
        requested_end=requested_end,
        effective_start=effective_start,
        effective_end=requested_end,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        selected_scopes=selected_scopes,
        query_diagnostics=query_diagnostics,
    )
    return input_set


async def _fetch_scope_messages(
    *,
    channel_rows: list[dict[str, Any]],
    start_timestamp: str,
    end_timestamp: str,
) -> list[ReflectionScopeInput]:
    """Fetch bounded transcript slices for selected monitored channels."""

    selected_scopes: list[ReflectionScopeInput] = []
    for row in channel_rows:
        row_id = row["_id"]
        platform = str(row_id["platform"])
        platform_channel_id = str(row_id["platform_channel_id"])
        channel_type = str(row_id["channel_type"])
        messages = await list_reflection_scope_messages(
            platform=platform,
            platform_channel_id=platform_channel_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            limit=READONLY_REFLECTION_MAX_MESSAGES_PER_SCOPE,
        )
        assistant_count = sum(
            1
            for message in messages
            if message.get("role") == "assistant"
        )
        user_count = sum(
            1
            for message in messages
            if message.get("role") == "user"
        )
        first_timestamp = ""
        last_timestamp = ""
        if messages:
            first_timestamp = str(messages[0]["timestamp"])
            last_timestamp = str(messages[-1]["timestamp"])
        selected_scope = ReflectionScopeInput(
            scope_ref=build_scope_ref(platform, platform_channel_id, channel_type),
            platform=platform,
            platform_channel_id=platform_channel_id,
            channel_type=channel_type,
            assistant_message_count=assistant_count,
            user_message_count=user_count,
            total_message_count=len(messages),
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            messages=messages,
        )
        selected_scopes.append(selected_scope)

    return_value = selected_scopes
    return return_value
