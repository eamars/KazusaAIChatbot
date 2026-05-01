"""Operations against the ``conversation_history`` collection."""

from __future__ import annotations

import logging
import re
from typing import Any

from kazusa_ai_chatbot.config import CONVERSATION_HISTORY_LIMIT
from kazusa_ai_chatbot.db._client import get_db, get_text_embedding
from kazusa_ai_chatbot.db.schemas import ConversationMessageDoc
from kazusa_ai_chatbot.message_envelope import INLINE_ATTACHMENT_BYTE_LIMIT
from kazusa_ai_chatbot.rag import cache2_runtime
from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent

logger = logging.getLogger(__name__)


def _safe_attachment_docs(attachments: object) -> list[dict[str, Any]]:
    """Apply the shared attachment storage policy before persistence.

    Args:
        attachments: Raw attachment list from a conversation document.

    Returns:
        Attachment docs with descriptions preserved and binary payloads stored
        only when the normalized storage shape permits inline data.
    """

    if not isinstance(attachments, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    safe_docs: list[dict[str, Any]] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue

        storage_shape = attachment.get("storage_shape", "")
        size_bytes = attachment.get("size_bytes")
        has_large_payload = (
            isinstance(size_bytes, int)
            and size_bytes > INLINE_ATTACHMENT_BYTE_LIMIT
        )
        if storage_shape == "drop":
            continue

        attachment_doc: dict[str, Any] = {}
        for field in ["media_type", "url", "description", "storage_shape"]:
            value = attachment.get(field)
            if value not in ("", None):
                attachment_doc[field] = value
        if isinstance(size_bytes, int):
            attachment_doc["size_bytes"] = size_bytes

        base64_data = attachment.get("base64_data")
        should_store_inline = (
            isinstance(base64_data, str)
            and bool(base64_data)
            and storage_shape != "url_only"
            and not has_large_payload
        )
        if should_store_inline:
            attachment_doc["base64_data"] = base64_data

        if attachment_doc:
            safe_docs.append(attachment_doc)

    return safe_docs


def _attachment_description_text(attachments: list[dict[str, Any]]) -> str:
    """Compose attachment descriptions for embedding-source text."""

    descriptions = [
        str(attachment["description"])
        for attachment in attachments
        if attachment.get("description")
    ]
    return_value = "\n".join(descriptions)
    return return_value


def _embedding_source_text(doc: ConversationMessageDoc) -> str:
    """Build semantic source text for conversation embeddings."""

    parts = [doc["body_text"]]
    attachment_text = _attachment_description_text(doc["attachments"])
    if attachment_text:
        parts.append(attachment_text)
    source_text = "\n".join(part for part in parts if part)
    return source_text


def _merge_attachment_descriptions(
    *,
    attachments: object,
    descriptions: list[str],
) -> list[dict[str, Any]]:
    """Apply generated descriptions to stored attachment documents by order.

    Args:
        attachments: Stored attachment list from one conversation row.
        descriptions: Current-turn generated descriptions aligned by order.

    Returns:
        Attachment dictionaries with non-empty descriptions applied.
    """

    if not isinstance(attachments, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    updated_attachments: list[dict[str, Any]] = []
    for index, attachment in enumerate(attachments):
        if not isinstance(attachment, dict):
            continue
        updated_attachment = dict(attachment)
        if index < len(descriptions):
            description = descriptions[index]
            if description.strip():
                updated_attachment["description"] = description.strip()
        updated_attachments.append(updated_attachment)

    return updated_attachments


def _keyword_text_filter(pattern: str) -> dict[str, Any]:
    """Build a case-insensitive filter for typed conversation body text."""

    regex_filter = {"$regex": pattern, "$options": "i"}
    filter_doc: dict[str, Any] = {"body_text": regex_filter}
    return filter_doc


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
    return_value = docs
    return return_value


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
        base_filter: dict[str, Any] = _keyword_text_filter(query)
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
        return_value = [(-1.0, doc) for doc in docs]
        return return_value

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
    return_value = [(doc.pop("score", 0.0), doc) for doc in docs]
    return return_value


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
        keyword: Optional literal body-text keyword to count only matching messages.
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
        match_filter.update(_keyword_text_filter(re.escape(keyword)))
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

    return_value = {
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
    return return_value


async def save_conversation(doc: ConversationMessageDoc) -> None:
    """Persist one conversation message and invalidate matching cache entries.

    Args:
        doc: Conversation-history row to store.

    Returns:
        None. The row is written and matching Cache2 entries are invalidated.
    """
    db = await get_db()

    if "content" in doc:
        raise KeyError("content")

    for required_key in (
        "body_text",
        "raw_wire_text",
        "addressed_to_global_user_ids",
        "mentions",
        "broadcast",
        "attachments",
    ):
        if required_key not in doc:
            raise KeyError(required_key)
    doc["attachments"] = _safe_attachment_docs(doc["attachments"])

    if "embedding" not in doc or not doc.get("embedding"):
        doc["embedding"] = await get_text_embedding(_embedding_source_text(doc))

    await db.conversation_history.insert_one(doc)
    evicted_count = await cache2_runtime.get_rag_cache2_runtime().invalidate(CacheInvalidationEvent(
        source="conversation_history",
        platform=doc.get("platform", ""),
        platform_channel_id=doc.get("platform_channel_id", ""),
        global_user_id=doc.get("global_user_id", ""),
        timestamp=doc.get("timestamp", ""),
        reason="save_conversation",
    ))
    if evicted_count:
        platform = doc.get("platform", "")
        platform_channel_id = doc.get("platform_channel_id", "")
        global_user_id = doc.get("global_user_id", "")
        logger.debug(f'Cache2 invalidation source=conversation_history platform={platform} channel={platform_channel_id} global_user={global_user_id} evicted={evicted_count}')


async def update_conversation_attachment_descriptions(
    *,
    platform: str,
    platform_channel_id: str,
    platform_message_id: str,
    descriptions: list[str],
) -> bool:
    """Persist generated attachment descriptions for one current message row.

    Args:
        platform: Runtime platform of the current message.
        platform_channel_id: Channel/group/DM id of the current message.
        platform_message_id: Platform message id of the current message.
        descriptions: Generated media descriptions aligned to attachment order.

    Returns:
        True when the current row was found and updated, otherwise false.
    """

    clean_descriptions = [
        description.strip() if isinstance(description, str) else ""
        for description in descriptions
    ]
    has_description = any(clean_descriptions)
    if not has_description or not platform_message_id:
        return False

    db = await get_db()
    query = {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "platform_message_id": platform_message_id,
    }
    row = await db.conversation_history.find_one(query)
    if row is None:
        return False

    attachments = _merge_attachment_descriptions(
        attachments=row.get("attachments", []),
        descriptions=clean_descriptions,
    )
    updated_row: ConversationMessageDoc = dict(row)
    updated_row["attachments"] = attachments
    embedding = await get_text_embedding(_embedding_source_text(updated_row))
    result = await db.conversation_history.update_one(
        query,
        {"$set": {"attachments": attachments, "embedding": embedding}},
    )
    return_value = bool(result.modified_count)
    return return_value
