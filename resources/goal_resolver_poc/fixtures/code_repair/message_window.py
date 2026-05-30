"""Message window utilities for the code repair fixture."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into a UTC-aware datetime."""

    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    utc_value = parsed.astimezone(timezone.utc)
    return utc_value


def collapse_followups(
    messages: list[dict[str, Any]],
    *,
    window_seconds: int,
) -> list[dict[str, Any]]:
    """Collapse adjacent messages from the same author inside a time window.

    Args:
        messages: Chat message rows with author_id, body, and created_at fields.
        window_seconds: Maximum gap between two messages in the same batch.

    Returns:
        Batches preserving chronological order and author boundaries.
    """

    ordered_messages = sorted(messages, key=lambda item: item["created_at"])
    batches: list[dict[str, Any]] = []
    current_bodies: list[str] = []
    current_author_id: str | None = None
    last_created_at: datetime | None = None

    for message in ordered_messages:
        author_id = message["author_id"]
        created_at = _parse_timestamp(message["created_at"])
        if last_created_at is None:
            gap_seconds = 0.0
        else:
            gap_seconds = (last_created_at - created_at).total_seconds()

        same_author = author_id == current_author_id
        inside_window = gap_seconds <= window_seconds
        starts_new_batch = bool(current_bodies) and not (
            same_author and inside_window
        )

        if starts_new_batch:
            batch = {
                "author_id": current_author_id,
                "messages": current_bodies,
            }
            batches.append(batch)
            current_bodies = []

        current_bodies.append(message["body"])
        current_author_id = author_id
        last_created_at = created_at

    if current_bodies:
        batch = {
            "author_id": current_author_id,
            "messages": current_bodies,
        }
        batches.append(batch)

    return batches
