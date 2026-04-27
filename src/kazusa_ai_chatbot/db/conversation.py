"""Operations against the ``conversation_history`` collection."""

from __future__ import annotations

import logging
import re
from typing import Any

from kazusa_ai_chatbot.config import CONVERSATION_HISTORY_LIMIT
from kazusa_ai_chatbot.db._client import get_db, get_text_embedding
from kazusa_ai_chatbot.db.schemas import ConversationMessageDoc

logger = logging.getLogger(__name__)


async def get_conversation_history(
    platform: str | None = None,
    platform_channel_id: str | None = None,
    limit: int = CONVERSATION_HISTORY_LIMIT,
    global_user_id: str | None = None,
    display_name: str | None = None,
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
) -> list[ConversationMessageDoc]:
    """Fetch the last ``limit`` messages for a channel (or all channels), oldest first."""
    db = await get_db()

    query: dict[str, Any] = {}
    if platform:
        query["platform"] = platform
    if platform_channel_id:
        query["platform_channel_id"] = platform_channel_id

    if global_user_id:
        query["global_user_id"] = global_user_id
    elif display_name:
        query["display_name"] = display_name

    if from_timestamp or to_timestamp:
        query["timestamp"] = {}
        if from_timestamp:
            query["timestamp"]["$gte"] = from_timestamp
        if to_timestamp:
            query["timestamp"]["$lte"] = to_timestamp

    cursor = (
        db.conversation_history
        .find(query)
        .sort("timestamp", -1)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    docs.reverse()  # oldest first
    return docs


async def search_conversation_history(
    query: str,
    platform: str | None = None,
    platform_channel_id: str | None = None,
    global_user_id: str | None = None,
    limit: int = 5,
    method: str = "vector",  # "keyword", "vector"
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
) -> list[tuple[float, ConversationMessageDoc]]:
    """Search conversation history using keyword or vector relevance.

    Args:
        query: The search query string.
        platform: Optional platform filter ("discord", "qq", etc.).
        platform_channel_id: Optional channel filter.
        global_user_id: Optional user filter (internal UUID).
        limit: Maximum number of results.
        method: "keyword" for regex text search, "vector" for semantic search.

    Returns:
        A list of ``(similarity_score, message_doc)`` tuples. Keyword results
        always have score ``-1.0``.
    """
    db = await get_db()
    collection = db.conversation_history

    if method == "keyword":
        base_filter: dict[str, Any] = {"content": {"$regex": query, "$options": "i"}}
        if platform:
            base_filter["platform"] = platform
        if platform_channel_id:
            base_filter["platform_channel_id"] = platform_channel_id
        if global_user_id:
            base_filter["global_user_id"] = global_user_id
        if from_timestamp or to_timestamp:
            base_filter["timestamp"] = {}
            if from_timestamp:
                base_filter["timestamp"]["$gte"] = from_timestamp
            if to_timestamp:
                base_filter["timestamp"]["$lte"] = to_timestamp

        cursor = collection.find(base_filter).sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [(-1.0, doc) for doc in docs]

    # method == "vector"
    query_embedding = await get_text_embedding(query)
    index_name = "conversation_history_vector_index"

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": limit * 10,
                "limit": limit,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
    ]

    match_filter: dict[str, Any] = {}
    if platform:
        match_filter["platform"] = platform
    if platform_channel_id:
        match_filter["platform_channel_id"] = platform_channel_id
    if global_user_id:
        match_filter["global_user_id"] = global_user_id
    if from_timestamp or to_timestamp:
        match_filter["timestamp"] = {}
        if from_timestamp:
            match_filter["timestamp"]["$gte"] = from_timestamp
        if to_timestamp:
            match_filter["timestamp"]["$lte"] = to_timestamp
    if match_filter:
        pipeline.append({"$match": match_filter})

    pipeline.append({"$limit": limit})
    pipeline.append({"$unset": "embedding"})

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=limit)
    return [(doc.pop("score", 0.0), doc) for doc in docs]


async def aggregate_conversation_by_user(
    *,
    platform: str | None = None,
    platform_channel_id: str | None = None,
    global_user_id: str | None = None,
    keyword: str | None = None,
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
    limit: int = 10,
) -> dict:
    """Compute factual message counts grouped by user.

    Args:
        platform: Optional platform filter.
        platform_channel_id: Optional channel filter.
        global_user_id: Optional user UUID filter.
        keyword: Optional literal content keyword to count only matching messages.
        from_timestamp: Optional inclusive start timestamp.
        to_timestamp: Optional inclusive end timestamp.
        limit: Maximum number of grouped rows to return.

    Returns:
        Dict containing total matched messages and ranked per-user counts.
    """
    effective_limit = max(1, min(limit, 50))
    db = await get_db()

    match_filter: dict[str, Any] = {"role": "user"}
    if platform:
        match_filter["platform"] = platform
    if platform_channel_id:
        match_filter["platform_channel_id"] = platform_channel_id
    if global_user_id:
        match_filter["global_user_id"] = global_user_id
    if keyword:
        match_filter["content"] = {"$regex": re.escape(keyword), "$options": "i"}
    if from_timestamp or to_timestamp:
        match_filter["timestamp"] = {}
        if from_timestamp:
            match_filter["timestamp"]["$gte"] = from_timestamp
        if to_timestamp:
            match_filter["timestamp"]["$lte"] = to_timestamp

    pipeline: list[dict[str, Any]] = [
        {"$match": match_filter},
        {
            "$group": {
                "_id": {
                    "global_user_id": "$global_user_id",
                    "platform_user_id": "$platform_user_id",
                    "platform": "$platform",
                },
                "display_names": {"$addToSet": "$display_name"},
                "message_count": {"$sum": 1},
                "first_timestamp": {"$min": "$timestamp"},
                "last_timestamp": {"$max": "$timestamp"},
            }
        },
        {"$sort": {"message_count": -1, "last_timestamp": -1}},
        {"$limit": effective_limit},
    ]
    rows = await db.conversation_history.aggregate(pipeline).to_list(length=effective_limit)
    total_count = await db.conversation_history.count_documents(match_filter)

    return {
        "total_count": total_count,
        "rows": [
            {
                "global_user_id": str(row["_id"].get("global_user_id", "")),
                "platform_user_id": str(row["_id"].get("platform_user_id", "")),
                "platform": str(row["_id"].get("platform", "")),
                "display_names": [
                    str(name)
                    for name in row.get("display_names", [])
                    if str(name).strip()
                ],
                "message_count": int(row.get("message_count", 0)),
                "first_timestamp": str(row.get("first_timestamp", "")),
                "last_timestamp": str(row.get("last_timestamp", "")),
            }
            for row in rows
        ],
        "query": {
            "platform": platform,
            "platform_channel_id": platform_channel_id,
            "global_user_id": global_user_id,
            "keyword": keyword,
            "from_timestamp": from_timestamp,
            "to_timestamp": to_timestamp,
            "limit": effective_limit,
        },
    }


async def save_conversation(doc: ConversationMessageDoc) -> None:
    """Persist one conversation message and invalidate matching cache entries.

    Args:
        doc: Conversation-history row to store.

    Returns:
        None. The row is written and matching Cache2 entries are invalidated.
    """
    from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent
    from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime

    db = await get_db()

    doc.setdefault("content_type", "text")
    doc.setdefault("mentioned_bot", False)
    doc.setdefault("attachments", [])
    doc.setdefault("channel_type", "group")

    if "embedding" not in doc or not doc.get("embedding"):
        doc["embedding"] = await get_text_embedding(doc.get("content", ""))

    await db.conversation_history.insert_one(doc)
    evicted_count = await get_rag_cache2_runtime().invalidate(CacheInvalidationEvent(
        source="conversation_history",
        platform=doc.get("platform", ""),
        platform_channel_id=doc.get("platform_channel_id", ""),
        global_user_id=doc.get("global_user_id", ""),
        timestamp=doc.get("timestamp", ""),
        reason="save_conversation",
    ))
    if evicted_count:
        logger.debug(
            "Cache2 invalidation source=conversation_history platform=%s channel=%s global_user=%s evicted=%d",
            doc.get("platform", ""),
            doc.get("platform_channel_id", ""),
            doc.get("global_user_id", ""),
            evicted_count,
        )
