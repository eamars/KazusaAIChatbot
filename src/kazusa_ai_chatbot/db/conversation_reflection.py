"""Read-only conversation-history queries for reflection evaluation."""

from __future__ import annotations

import time
from typing import Any

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db._client import get_db


_REFLECTION_MESSAGE_PROJECTION = {
    "_id": 0,
    "platform": 1,
    "platform_channel_id": 1,
    "channel_type": 1,
    "role": 1,
    "platform_user_id": 1,
    "global_user_id": 1,
    "display_name": 1,
    "body_text": 1,
    "timestamp": 1,
    "attachments.description": 1,
}


async def list_recent_character_message_channels(
    *,
    start_timestamp: str,
    end_timestamp: str,
    limit: int,
) -> dict[str, Any]:
    """Return channels where the character spoke inside the monitor window.

    Args:
        start_timestamp: Inclusive ISO lower bound.
        end_timestamp: Inclusive ISO upper bound.
        limit: Maximum channels to return.

    Returns:
        Rows plus read-only query diagnostics.
    """

    db = await get_db()
    pipeline = _recent_character_channels_pipeline(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        limit=limit,
    )
    started_at = time.perf_counter()
    raw_rows = await db.conversation_history.aggregate(pipeline).to_list(
        length=limit,
    )
    rows = [
        _channel_row_from_aggregate(row)
        for row in raw_rows
    ]
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    result = {
        "rows": rows,
        "diagnostics": {
            "channel_query_elapsed_ms": elapsed_ms,
            "channel_row_count": len(raw_rows),
            "pipeline_summary": {
                "collection": "conversation_history",
                "window_start": start_timestamp,
                "window_end": end_timestamp,
                "max_channels": limit,
                "monitor_rule": "latest assistant message inside monitor window",
            },
        },
    }
    return result


async def list_reflection_scope_messages(
    *,
    platform: str,
    platform_channel_id: str,
    start_timestamp: str,
    end_timestamp: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Return bounded read-only messages for one reflection scope.

    Args:
        platform: Platform namespace for the selected scope.
        platform_channel_id: Channel or private-thread id for the scope.
        start_timestamp: Inclusive ISO lower bound.
        end_timestamp: Inclusive ISO upper bound.
        limit: Maximum messages to return.

    Returns:
        Chronological conversation rows containing only fields needed for
        prompt projection and bounded attachment descriptions.
    """

    db = await get_db()
    query = {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "timestamp": {"$gte": start_timestamp, "$lte": end_timestamp},
        "role": {"$in": ["assistant", "user"]},
    }
    cursor = (
        db.conversation_history
        .find(query, _REFLECTION_MESSAGE_PROJECTION)
        .sort("timestamp", -1)
        .limit(limit)
    )
    messages = await cursor.to_list(length=limit)
    messages.reverse()
    return_value = messages
    return return_value


async def resolve_single_private_scope_user_id(
    *,
    platform: str,
    platform_channel_id: str,
    start_timestamp: str,
    end_timestamp: str,
    character_global_user_id: str,
) -> str:
    """Resolve the only non-character user in a private reflection window.

    Args:
        platform: Platform namespace for the private channel.
        platform_channel_id: Private channel id.
        start_timestamp: Inclusive ISO lower bound.
        end_timestamp: Inclusive ISO upper bound.
        character_global_user_id: Internal UUID for the character identity.

    Returns:
        The sole non-character ``global_user_id``, or an empty string when the
        scope cannot be resolved to exactly one user.
    """

    db = await get_db()
    cursor = db.conversation_history.find(
        {
            "platform": platform,
            "platform_channel_id": platform_channel_id,
            "channel_type": "private",
            "timestamp": {"$gte": start_timestamp, "$lte": end_timestamp},
            "role": "user",
        },
        {"_id": 0, "global_user_id": 1},
    )
    rows = await cursor.to_list(length=None)
    user_ids: set[str] = set()
    for row in rows:
        global_user_id = str(row.get("global_user_id", "") or "").strip()
        if not global_user_id:
            continue
        if global_user_id == character_global_user_id:
            continue
        user_ids.add(global_user_id)
    if len(user_ids) != 1:
        return_value = ""
        return return_value
    return_value = next(iter(user_ids))
    return return_value


async def explain_monitored_channel_query(
    *,
    start_timestamp: str,
    end_timestamp: str,
    limit: int,
) -> dict[str, Any]:
    """Return a best-effort read-only explain summary for channel monitoring.

    Args:
        start_timestamp: Inclusive ISO lower bound.
        end_timestamp: Inclusive ISO upper bound.
        limit: Maximum channels in the explained aggregate pipeline.

    Returns:
        Explain diagnostics when MongoDB supports the command, otherwise a
        structured unavailable reason.
    """

    db = await get_db()
    pipeline = _recent_character_channels_pipeline(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        limit=limit,
    )
    command = {
        "explain": {
            "aggregate": "conversation_history",
            "pipeline": pipeline,
            "cursor": {},
        },
        "verbosity": "executionStats",
    }
    try:
        explain = await db.command(command)
    except (AttributeError, TypeError, PyMongoError) as exc:
        return_value = {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
        return return_value

    execution_stats = explain.get("executionStats", {})
    result = {
        "available": True,
        "execution_time_millis": execution_stats.get("executionTimeMillis"),
        "total_docs_examined": execution_stats.get("totalDocsExamined"),
        "total_keys_examined": execution_stats.get("totalKeysExamined"),
    }
    return result


def _recent_character_channels_pipeline(
    *,
    start_timestamp: str,
    end_timestamp: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Build the MongoDB pipeline for monitored channel selection."""

    pipeline: list[dict[str, Any]] = [
        {
            "$match": {
                "timestamp": {"$gte": start_timestamp, "$lte": end_timestamp},
                "role": "assistant",
            }
        },
        {
            "$group": {
                "_id": {
                    "platform": "$platform",
                    "platform_channel_id": "$platform_channel_id",
                    "channel_type": "$channel_type",
                },
                "character_message_count": {"$sum": 1},
                "first_character_message_timestamp": {"$min": "$timestamp"},
                "last_character_message_timestamp": {"$max": "$timestamp"},
            }
        },
        {"$sort": {"last_character_message_timestamp": -1}},
        {"$limit": limit},
    ]
    return pipeline


def _channel_row_from_aggregate(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one MongoDB aggregate row for reflection selectors.

    Args:
        row: Raw aggregate row from ``conversation_history``.

    Returns:
        Channel row with required keys populated for internal selector use.
    """

    raw_id = row.get("_id")
    if not isinstance(raw_id, dict):
        raw_id = {}
    normalized_id = {
        "platform": str(raw_id.get("platform", "") or ""),
        "platform_channel_id": str(raw_id.get("platform_channel_id", "") or ""),
        "channel_type": str(raw_id.get("channel_type", "") or "unknown"),
    }
    normalized_row = {
        "_id": normalized_id,
        "character_message_count": int(row.get("character_message_count", 0)),
        "first_character_message_timestamp": str(
            row.get("first_character_message_timestamp", "")
        ),
        "last_character_message_timestamp": str(
            row.get("last_character_message_timestamp", "")
        ),
    }
    return normalized_row
