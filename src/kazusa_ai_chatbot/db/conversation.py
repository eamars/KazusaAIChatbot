"""Operations against the ``conversation_history`` collection."""

from __future__ import annotations

import logging
import re
from typing import Any

from kazusa_ai_chatbot.config import (
    CONVERSATION_HISTORY_LIMIT,
    RAG_SEARCH_DEFAULT_TOP_K,
    RAG_VECTOR_CANDIDATE_MULTIPLIER,
    RAG_VECTOR_MAX_CANDIDATES,
    RAG_VECTOR_MIN_CANDIDATES,
    SAVE_ATTACHMENT_BASE64_TO_DB,
)
from kazusa_ai_chatbot.db._client import (
    get_document_text_embedding,
    get_db,
    get_query_text_embedding,
    get_search_index_definition,
    vector_index_has_filter_paths,
)
from kazusa_ai_chatbot.db.schemas import ConversationMessageDoc
from kazusa_ai_chatbot.message_envelope import INLINE_ATTACHMENT_BYTE_LIMIT
from kazusa_ai_chatbot.rag import cache2_runtime
from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent

logger = logging.getLogger(__name__)

CONVERSATION_VECTOR_INDEX_NAME = "conversation_history_vector_index"
CONVERSATION_VECTOR_FILTER_FIELDS = (
    "platform",
    "platform_channel_id",
    "global_user_id",
    "role",
    "timestamp",
)
_conversation_vector_prefilter_support_cache: bool | None = None


def reset_conversation_vector_prefilter_support_cache() -> None:
    """Clear cached conversation vector-index capability inspection.

    Tests and maintenance scripts use this after recreating the Atlas search
    index so the next retrieval observes the current definition.
    """

    global _conversation_vector_prefilter_support_cache
    _conversation_vector_prefilter_support_cache = None


def _vector_num_candidates(limit: int) -> int:
    """Return a bounded Atlas candidate count for conversation vector search."""

    candidate_count = max(
        RAG_VECTOR_MIN_CANDIDATES,
        limit * RAG_VECTOR_CANDIDATE_MULTIPLIER,
    )
    candidate_count = min(candidate_count, RAG_VECTOR_MAX_CANDIDATES)
    return candidate_count


def _conversation_search_filter(
    *,
    platform: str | None,
    platform_channel_id: str | None,
    global_user_id: str | None,
    from_timestamp: str | None,
    to_timestamp: str | None,
) -> dict[str, Any]:
    """Build the shared filter document for conversation retrieval."""

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
    return match_filter


async def _conversation_vector_prefilter_supported() -> bool:
    """Return whether the live vector index supports required prefilters."""

    global _conversation_vector_prefilter_support_cache
    if _conversation_vector_prefilter_support_cache is True:
        return _conversation_vector_prefilter_support_cache

    try:
        index_document = await get_search_index_definition(
            "conversation_history",
            CONVERSATION_VECTOR_INDEX_NAME,
        )
    except Exception as exc:
        logger.warning(
            "Could not inspect conversation vector index filters; "
            f"falling back to post-filtering: {exc}"
        )
        return_value = False
        return return_value

    if index_document is None:
        logger.warning(
            "Conversation vector index is missing; "
            "falling back to post-filtering."
        )
        return_value = False
        return return_value

    supported = vector_index_has_filter_paths(
        index_document,
        CONVERSATION_VECTOR_FILTER_FIELDS,
    )
    if supported:
        _conversation_vector_prefilter_support_cache = True
    return supported


def _safe_attachment_docs(attachments: object) -> list[dict[str, Any]]:
    """Apply the shared attachment storage policy before persistence.

    Args:
        attachments: Raw attachment list from a conversation document.

    Returns:
        Attachment docs with descriptions preserved and binary payloads stored
        only when configured and the normalized storage shape permits inline
        data.
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
            SAVE_ATTACHMENT_BASE64_TO_DB
            and isinstance(base64_data, str)
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
    """Build a case-insensitive filter for searchable conversation text."""

    regex_filter = {"$regex": pattern, "$options": "i"}
    filter_doc: dict[str, Any] = {
        "$or": [
            {"body_text": regex_filter},
            {"attachments.description": regex_filter},
        ]
    }
    return filter_doc


async def get_conversation_history(
    platform: str | None = None,
    platform_channel_id: str | None = None,
    limit: int = CONVERSATION_HISTORY_LIMIT,
    global_user_id: str | None = None,
    display_name: str | None = None,
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
    sort_direction: int = -1,
) -> list[ConversationMessageDoc]:
    """Fetch bounded messages for a channel, returned oldest first.

    Args:
        platform: Optional platform filter.
        platform_channel_id: Optional channel filter.
        limit: Maximum number of rows to fetch before chronological projection.
        global_user_id: Optional speaker UUID filter.
        display_name: Optional speaker display-name filter when no UUID is set.
        from_timestamp: Optional inclusive lower timestamp bound.
        to_timestamp: Optional inclusive upper timestamp bound.
        sort_direction: ``-1`` for newest rows in the window, ``1`` for oldest.

    Returns:
        Conversation rows sorted chronologically.
    """
    db = await get_db()
    effective_sort_direction = 1 if sort_direction == 1 else -1

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
        .sort("timestamp", effective_sort_direction)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    if effective_sort_direction == -1:
        docs.reverse()  # oldest first
    return_value = docs
    return return_value


async def search_conversation_history(
    query: str,
    platform: str | None = None,
    platform_channel_id: str | None = None,
    global_user_id: str | None = None,
    limit: int = RAG_SEARCH_DEFAULT_TOP_K,
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
        base_filter = _keyword_text_filter(query)
        base_filter.update(
            _conversation_search_filter(
                platform=platform,
                platform_channel_id=platform_channel_id,
                global_user_id=global_user_id,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
            )
        )

        cursor = collection.find(base_filter).sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return_value = [(-1.0, doc) for doc in docs]
        return return_value

    # method == "vector"
    query_embedding = await get_query_text_embedding(query)
    index_name = CONVERSATION_VECTOR_INDEX_NAME
    match_filter = _conversation_search_filter(
        platform=platform,
        platform_channel_id=platform_channel_id,
        global_user_id=global_user_id,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )
    vector_stage: dict[str, Any] = {
        "index": index_name,
        "path": "embedding",
        "queryVector": query_embedding,
        "numCandidates": _vector_num_candidates(limit),
        "limit": limit,
    }
    use_prefilter = (
        bool(match_filter)
        and await _conversation_vector_prefilter_supported()
    )
    if use_prefilter:
        vector_stage["filter"] = match_filter

    pipeline: list[dict[str, Any]] = [
        {"$vectorSearch": vector_stage},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
    ]

    if match_filter and not use_prefilter:
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


async def save_conversation(doc: ConversationMessageDoc) -> str:
    """Persist one conversation message and invalidate matching cache entries.

    Args:
        doc: Conversation-history row to store.

    Returns:
        MongoDB row ID string for the committed conversation-history row.
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
        doc["embedding"] = await get_document_text_embedding(
            _embedding_source_text(doc)
        )

    insert_result = await db.conversation_history.insert_one(doc)
    inserted_id_str = str(insert_result.inserted_id)
    invalidation_event = CacheInvalidationEvent(
        source="conversation_history",
        platform=doc.get("platform", ""),
        platform_channel_id=doc.get("platform_channel_id", ""),
        global_user_id=doc.get("global_user_id", ""),
        storage_timestamp_utc=doc.get("timestamp", ""),
        reason="save_conversation",
    )
    cache_runtime = cache2_runtime.get_rag_cache2_runtime()
    try:
        evicted_count = await cache_runtime.invalidate(invalidation_event)
    except Exception as exc:
        logger.warning(
            f"Cache2 invalidation after save_conversation failed: {exc}"
        )
        return inserted_id_str

    if evicted_count:
        platform = doc.get("platform", "")
        platform_channel_id = doc.get("platform_channel_id", "")
        global_user_id = doc.get("global_user_id", "")
        logger.debug(f'Cache2 invalidation source=conversation_history platform={platform} channel={platform_channel_id} global_user={global_user_id} evicted={evicted_count}')

    return inserted_id_str


async def apply_assistant_delivery_receipt(
    *,
    platform: str,
    platform_channel_id: str,
    delivery_tracking_id: str,
    platform_message_id: str,
    delivered_at: str,
    adapter: str,
) -> bool:
    """Record delivered platform metadata for one assistant conversation row.

    Args:
        platform: Runtime platform that delivered the assistant row.
        platform_channel_id: Channel/group/DM id for the assistant row.
        delivery_tracking_id: Brain-generated local row tracking id.
        platform_message_id: Message id returned by the platform adapter.
        delivered_at: ISO timestamp for delivery receipt creation.
        adapter: Adapter name reporting the receipt.

    Returns:
        True when a matching assistant row was found, otherwise false.
    """

    db = await get_db()
    query = {
        "platform": platform,
        "role": "assistant",
        "delivery_tracking_id": delivery_tracking_id,
    }
    if platform_channel_id:
        query["platform_channel_id"] = platform_channel_id

    update = {
        "$set": {
            "platform_message_id": platform_message_id,
            "delivery_status": "delivered",
            "delivered_at": delivered_at,
            "delivery_adapter": adapter,
        }
    }
    result = await db.conversation_history.update_one(query, update)
    return_value = bool(result.matched_count)
    return return_value


async def get_conversation_by_platform_message_id(
    *,
    platform: str,
    platform_channel_id: str,
    platform_message_id: str,
) -> ConversationMessageDoc | None:
    """Fetch one conversation row by exact platform message identity.

    Args:
        platform: Runtime platform of the reply target.
        platform_channel_id: Channel/group/DM id of the reply target.
        platform_message_id: Platform-native message id of the reply target.

    Returns:
        Conversation row when found, otherwise ``None``.
    """

    if not platform_message_id:
        return None

    db = await get_db()
    query = {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "platform_message_id": platform_message_id,
    }
    row = await db.conversation_history.find_one(query)
    return_value = row
    return return_value


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
    embedding = await get_document_text_embedding(
        _embedding_source_text(updated_row)
    )
    result = await db.conversation_history.update_one(
        query,
        {"$set": {"attachments": attachments, "embedding": embedding}},
    )
    return_value = bool(result.modified_count)
    return return_value
