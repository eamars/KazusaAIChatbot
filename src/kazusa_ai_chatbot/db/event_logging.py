"""MongoDB persistence helpers for event logging internals."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot.db._client import get_db

EVENT_LOG_EVENTS_COLLECTION = "event_log_events"
EVENT_LOG_SNAPSHOTS_COLLECTION = "event_log_snapshots"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO string."""

    timestamp = datetime.now(timezone.utc).isoformat()
    return timestamp


async def ensure_event_log_indexes() -> None:
    """Create event-log collections and indexes idempotently."""

    db = await get_db()
    existing = set(await db.list_collection_names())
    for collection_name in (
        EVENT_LOG_EVENTS_COLLECTION,
        EVENT_LOG_SNAPSHOTS_COLLECTION,
    ):
        if collection_name not in existing:
            await db.create_collection(collection_name)

    events = db[EVENT_LOG_EVENTS_COLLECTION]
    snapshots = db[EVENT_LOG_SNAPSHOTS_COLLECTION]
    await events.create_index("event_id", unique=True, name="event_log_event_id_unique")
    await events.create_index(
        [("event_family", 1), ("component", 1), ("occurred_at", -1)],
        name="event_log_family_component_time",
    )
    await events.create_index(
        [("component", 1), ("event_type", 1), ("status", 1), ("occurred_at", -1)],
        name="event_log_component_type_status_time",
    )
    await events.create_index(
        [("correlation_id", 1), ("occurred_at", -1)],
        name="event_log_correlation_time",
    )
    await events.create_index(
        [("run_id", 1), ("occurred_at", -1)],
        name="event_log_run_time",
    )
    await events.create_index(
        [("trigger_id", 1), ("occurred_at", -1)],
        name="event_log_trigger_time",
    )
    await events.create_index(
        [("attempt_id", 1), ("occurred_at", -1)],
        name="event_log_attempt_time",
    )
    await snapshots.create_index(
        "snapshot_id",
        unique=True,
        name="event_log_snapshot_id_unique",
    )
    await snapshots.create_index(
        [("snapshot_kind", 1), ("generated_at", -1)],
        name="event_log_snapshot_kind_time",
    )


async def insert_event_log_event(document: Mapping[str, Any]) -> str:
    """Insert one canonical event document.

    Args:
        document: Fully sanitized event document built by event logging.

    Returns:
        Event identifier from the inserted document.
    """

    db = await get_db()
    await db[EVENT_LOG_EVENTS_COLLECTION].insert_one(dict(document))
    event_id = str(document["event_id"])
    return event_id


async def insert_event_log_snapshot(document: Mapping[str, Any]) -> str:
    """Insert one aggregate event-log snapshot."""

    db = await get_db()
    await db[EVENT_LOG_SNAPSHOTS_COLLECTION].insert_one(dict(document))
    snapshot_id = str(document["snapshot_id"])
    return snapshot_id


async def aggregate_event_log_events(
    pipeline: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run an event-log aggregation and return all rows."""

    db = await get_db()
    cursor = db[EVENT_LOG_EVENTS_COLLECTION].aggregate(list(pipeline))
    rows = await cursor.to_list(length=None)
    return rows


async def find_event_log_events(
    filter_doc: Mapping[str, Any],
    *,
    sort: list[tuple[str, int]] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Find bounded event-log rows for status builders or exports."""

    db = await get_db()
    cursor = db[EVENT_LOG_EVENTS_COLLECTION].find(dict(filter_doc))
    if sort is not None:
        cursor = cursor.sort(sort)
    if limit > 0:
        cursor = cursor.limit(limit)
    rows = await cursor.to_list(length=limit if limit > 0 else None)
    return rows


async def count_event_log_events(filter_doc: Mapping[str, Any]) -> int:
    """Count event-log rows for a bounded query."""

    db = await get_db()
    count = await db[EVENT_LOG_EVENTS_COLLECTION].count_documents(dict(filter_doc))
    return count


def window_start_iso(*, window_hours: int) -> str:
    """Return a UTC lower-bound timestamp for an event-log query window."""

    now = datetime.now(timezone.utc)
    seconds = max(0, int(window_hours)) * 3600
    window_start = datetime.fromtimestamp(now.timestamp() - seconds, timezone.utc)
    return_value = window_start.isoformat()
    return return_value
