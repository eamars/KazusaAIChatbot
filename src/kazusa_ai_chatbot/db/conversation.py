"""Operations against the ``conversation_history`` collection."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.config import CONVERSATION_HISTORY_LIMIT
from kazusa_ai_chatbot.db._client import get_db, get_text_embedding
from kazusa_ai_chatbot.db.schemas import ConversationMessageDoc


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
    if match_filter:
        pipeline.append({"$match": match_filter})

    pipeline.append({"$limit": limit})
    pipeline.append({"$unset": "embedding"})

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=limit)
    return [(doc.pop("score", 0.0), doc) for doc in docs]


async def save_conversation(doc: ConversationMessageDoc) -> None:
    """Persist a single message to conversation history, generating its embedding."""
    db = await get_db()

    doc.setdefault("content_type", "text")
    doc.setdefault("attachments", [])
    doc.setdefault("channel_type", "group")

    if "embedding" not in doc or not doc.get("embedding"):
        doc["embedding"] = await get_text_embedding(doc.get("content", ""))

    await db.conversation_history.insert_one(doc)
