"""Deterministic permission checks for proactive preview delivery."""

from __future__ import annotations

from datetime import datetime

from kazusa_ai_chatbot.cognition_episode import OutputMode
from kazusa_ai_chatbot.proactive_output.contracts import (
    ProactivePermissionRecord,
    ProactivePolicyDecision,
    ProactivePreviewRecord,
    QuietHoursPolicy,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime

__all__ = [
    "PROACTIVE_ALLOWED_OUTPUT_MODE",
    "evaluate_proactive_permission",
    "is_local_time_in_quiet_hours",
]

PROACTIVE_ALLOWED_OUTPUT_MODE: OutputMode = "preview"


def _parse_storage_utc(value: str) -> datetime:
    """Parse a persisted storage UTC timestamp.

    Args:
        value: Storage UTC timestamp string to parse.

    Returns:
        Parsed storage UTC datetime.

    Raises:
        ValueError: If the timestamp is not valid storage UTC text.
    """

    return_value = parse_storage_utc_datetime(value)
    return return_value


def _parse_local_time_minutes(value: str) -> int:
    """Parse ``HH:MM`` local time text into minutes since midnight.

    Args:
        value: Local clock time formatted as ``HH:MM``.

    Returns:
        Number of minutes since local midnight.

    Raises:
        ValueError: If the time is not valid ``HH:MM`` text.
    """

    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(value)

    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(value)

    return_value = (hour * 60) + minute
    return return_value


def is_local_time_in_quiet_hours(
    *,
    current_local_time: str,
    quiet_hours: QuietHoursPolicy,
) -> bool:
    """Return whether a local clock time falls inside quiet hours.

    Args:
        current_local_time: Local clock time formatted as ``HH:MM``.
        quiet_hours: Quiet-hour policy carried by the permission record.

    Returns:
        True when proactive delivery should be blocked by quiet hours.

    Raises:
        ValueError: If any enabled quiet-hour time is invalid.
    """

    if not quiet_hours["enabled"]:
        return False

    current_minutes = _parse_local_time_minutes(current_local_time)
    start_minutes = _parse_local_time_minutes(quiet_hours["start_local_time"])
    end_minutes = _parse_local_time_minutes(quiet_hours["end_local_time"])

    if start_minutes == end_minutes:
        return True
    if start_minutes < end_minutes:
        return_value = start_minutes <= current_minutes < end_minutes
        return return_value

    return_value = current_minutes >= start_minutes or current_minutes < end_minutes
    return return_value


def evaluate_proactive_permission(
    *,
    preview: ProactivePreviewRecord,
    permission: ProactivePermissionRecord | None,
    existing_idempotency_keys: set[str],
    adapter_platforms: set[str],
    current_timestamp_utc: str,
    current_local_time: str,
) -> ProactivePolicyDecision:
    """Evaluate whether one proactive preview is allowed to leave dry run.

    Args:
        preview: Approved candidate preview record.
        permission: Explicit permission record for the target, if one exists.
        existing_idempotency_keys: Already accepted or sent outbox keys.
        adapter_platforms: Platform keys with an available messaging adapter.
        current_timestamp_utc: Current storage UTC timestamp for expiry checks.
        current_local_time: Current local clock time formatted as ``HH:MM``.

    Returns:
        Deterministic allow or deny decision with a stable reason string.

    Raises:
        ValueError: If timestamp or quiet-hour fields are invalid.
    """

    if permission is None:
        decision: ProactivePolicyDecision = {
            "allowed": False,
            "reason": "missing_permission",
        }
        return decision

    if permission["enabled"] is False:
        decision = {
            "allowed": False,
            "reason": "permission_disabled",
        }
        return decision

    expires_at_utc = _parse_storage_utc(permission["expires_at"])
    current_utc = _parse_storage_utc(current_timestamp_utc)
    if expires_at_utc <= current_utc:
        decision = {
            "allowed": False,
            "reason": "permission_expired",
        }
        return decision

    if preview["trigger_source"] == "user_message":
        decision = {
            "allowed": False,
            "reason": "user_message_not_proactive",
        }
        return decision

    if preview["trigger_source"] not in permission["allowed_trigger_sources"]:
        decision = {
            "allowed": False,
            "reason": "trigger_source_not_allowed",
        }
        return decision

    output_mode = preview["output_mode"]
    output_allowed = (
        output_mode in permission["allowed_output_modes"]
        and output_mode == PROACTIVE_ALLOWED_OUTPUT_MODE
    )
    if not output_allowed:
        decision = {
            "allowed": False,
            "reason": "unsafe_output_mode",
        }
        return decision

    if preview["visibility"] != "model_visible":
        decision = {
            "allowed": False,
            "reason": "content_not_public",
        }
        return decision

    target_matches = (
        preview["platform"] == permission["platform"]
        and preview["platform_channel_id"] == permission["platform_channel_id"]
        and preview["channel_type"] == permission["channel_type"]
        and preview["target_global_user_id"] == permission["target_global_user_id"]
        and preview["target_platform_user_id"] == permission["target_platform_user_id"]
    )
    if not target_matches:
        decision = {
            "allowed": False,
            "reason": "target_mismatch",
        }
        return decision

    in_quiet_hours = is_local_time_in_quiet_hours(
        current_local_time=current_local_time,
        quiet_hours=permission["quiet_hours"],
    )
    if in_quiet_hours:
        decision = {
            "allowed": False,
            "reason": "quiet_hours",
        }
        return decision

    if preview["platform"] not in adapter_platforms:
        decision = {
            "allowed": False,
            "reason": "adapter_unavailable",
        }
        return decision

    if preview["idempotency_key"] in existing_idempotency_keys:
        decision = {
            "allowed": False,
            "reason": "duplicate_idempotency_key",
        }
        return decision

    if not preview["preview_text"].strip():
        decision = {
            "allowed": False,
            "reason": "empty_preview_text",
        }
        return decision

    decision = {
        "allowed": True,
        "reason": "allowed",
    }
    return decision
