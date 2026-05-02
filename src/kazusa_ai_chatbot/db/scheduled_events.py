"""Read-only helpers for scheduled future events."""

from __future__ import annotations

from kazusa_ai_chatbot.db._client import get_db
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
