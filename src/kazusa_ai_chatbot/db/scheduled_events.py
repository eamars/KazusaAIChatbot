"""Persistence helpers for scheduled future events."""

from __future__ import annotations

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.db.schemas import ScheduledEventDoc


async def query_pending_scheduled_events(
    *,
    platform: str,
    platform_channel_id: str,
    global_user_id: str,
    current_timestamp: str,
    limit: int = 10,
) -> list[ScheduledEventDoc]:
    """Read pending future events scoped to one source user and channel.

    Args:
        platform: Source platform of the conversation that created the event.
        platform_channel_id: Source channel or private thread identifier.
        global_user_id: Internal UUID for the source user.
        current_timestamp: Lower bound for pending ``execute_at`` timestamps.
        limit: Maximum number of pending events to return.

    Returns:
        Pending scheduled event documents sorted by execution time.
    """

    query = {
        "status": "pending",
        "source_platform": platform,
        "source_channel_id": platform_channel_id,
        "source_user_id": global_user_id,
        "execute_at": {"$gte": current_timestamp},
    }

    db = await get_db()
    cursor = (
        db.scheduled_events
        .find(query, {"_id": 0})
        .sort("execute_at", 1)
        .limit(limit)
    )
    return_value = [doc async for doc in cursor]
    return return_value


async def insert_scheduled_event(event: ScheduledEventDoc) -> None:
    """Insert one pending scheduled event document."""

    try:
        db = await get_db()
        await db.scheduled_events.insert_one(event)
    except PyMongoError as exc:
        raise DatabaseOperationError(
            f"failed to insert scheduled event: {exc}"
        ) from exc


async def list_pending_scheduler_events() -> list[ScheduledEventDoc]:
    """Return all pending scheduler documents in execution order."""

    db = await get_db()
    cursor = db.scheduled_events.find({"status": "pending"}).sort("execute_at", 1)
    return_value = [doc async for doc in cursor]
    return return_value


async def mark_scheduled_event_running(event_id: str) -> bool:
    """Mark one scheduled event as running."""

    db = await get_db()
    result = await db.scheduled_events.update_one(
        {"event_id": event_id},
        {"$set": {"status": "running"}},
    )
    return_value = result.modified_count > 0
    return return_value


async def mark_scheduled_event_completed(event_id: str) -> bool:
    """Mark one scheduled event as completed."""

    db = await get_db()
    result = await db.scheduled_events.update_one(
        {"event_id": event_id},
        {"$set": {"status": "completed"}},
    )
    return_value = result.modified_count > 0
    return return_value


async def mark_scheduled_event_failed(event_id: str) -> bool:
    """Mark one scheduled event as failed."""

    db = await get_db()
    result = await db.scheduled_events.update_one(
        {"event_id": event_id},
        {"$set": {"status": "failed"}},
    )
    return_value = result.modified_count > 0
    return return_value


async def cancel_pending_scheduled_event(
    event_id: str,
    *,
    cancelled_at: str,
) -> bool:
    """Mark one pending scheduled event as cancelled."""

    db = await get_db()
    result = await db.scheduled_events.update_one(
        {"event_id": event_id, "status": "pending"},
        {
            "$set": {
                "status": "cancelled",
                "cancelled_at": cancelled_at,
            }
        },
    )
    return_value = result.modified_count > 0
    return return_value
