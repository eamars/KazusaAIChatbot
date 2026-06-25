"""Shared retention helpers for audit and debug logging collections."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kazusa_ai_chatbot.time_boundary import storage_utc_now


def _normalize_utc_datetime(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime for MongoDB TTL fields."""

    if value.tzinfo is None:
        return_value = value.replace(tzinfo=timezone.utc)
        return return_value
    return_value = value.astimezone(timezone.utc)
    return return_value


def expiry_from_datetime(value: datetime, *, ttl_days: int) -> datetime:
    """Return the TTL expiry datetime derived from a source datetime."""

    source = _normalize_utc_datetime(value)
    expires_at = source + timedelta(days=ttl_days)
    return expires_at


def expiry_from_now(*, ttl_days: int) -> datetime:
    """Return the TTL expiry datetime derived from the current UTC time."""

    expires_at = expiry_from_datetime(storage_utc_now(), ttl_days=ttl_days)
    return expires_at


def expiry_from_storage_iso(value: str, *, ttl_days: int) -> datetime:
    """Return a TTL expiry datetime from a stored ISO timestamp.

    Invalid or empty timestamps receive a conservative future expiry based on
    the current time, so malformed legacy rows do not expire immediately.
    """

    try:
        source = datetime.fromisoformat(value)
    except ValueError:
        expires_at = expiry_from_now(ttl_days=ttl_days)
        return expires_at
    expires_at = expiry_from_datetime(source, ttl_days=ttl_days)
    return expires_at
