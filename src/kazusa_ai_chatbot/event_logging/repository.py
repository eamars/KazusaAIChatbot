"""Private adapter from event logging APIs to database helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.db.event_logging import (
    aggregate_event_log_events,
    count_event_log_events,
    find_event_log_events,
    insert_event_log_event,
    insert_event_log_snapshot,
)


async def write_event(document: Mapping[str, Any]) -> str:
    """Persist one sanitized event document through the DB package."""

    event_id = await insert_event_log_event(document)
    return event_id


async def write_snapshot(document: Mapping[str, Any]) -> str:
    """Persist one sanitized aggregate snapshot through the DB package."""

    snapshot_id = await insert_event_log_snapshot(document)
    return snapshot_id


async def aggregate_events(pipeline: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Run an event-log aggregation through the DB package."""

    rows = await aggregate_event_log_events(pipeline)
    return rows


async def find_events(
    filter_doc: Mapping[str, Any],
    *,
    sort: list[tuple[str, int]] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Find bounded event-log rows through the DB package."""

    rows = await find_event_log_events(filter_doc, sort=sort, limit=limit)
    return rows


async def count_events(filter_doc: Mapping[str, Any]) -> int:
    """Count event-log rows through the DB package."""

    count = await count_event_log_events(filter_doc)
    return count
