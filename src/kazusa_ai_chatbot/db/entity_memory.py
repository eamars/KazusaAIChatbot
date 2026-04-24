"""Operations against the ``entity_memory`` collection.

Durable entity/topic memory substrate (Phase 3).  Stores recurring
third-party people, groups, topics, and events with bounded recent-mention
windows and compressed historical summaries.

Schema follows the unified Option B design: one storage substrate,
``subject_kind`` tag discriminates at retrieval time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot.db._client import enable_vector_index, get_db, get_text_embedding
from kazusa_ai_chatbot.db.schemas import EntityMemoryDoc

logger = logging.getLogger(__name__)

_VALID_SUBJECT_KINDS = frozenset({"topic", "person", "group", "event"})
_MAX_RECENT_MENTIONS = 20


async def enable_entity_memory_vector_index() -> None:
    """Create the vector search index on the ``entity_memory`` collection."""
    await enable_vector_index("entity_memory", "entity_memory_vector_index")


async def upsert_entity_memory(
    subject_key: str,
    subject_kind: str,
    *,
    display_names: list[str] | None = None,
    resolved_global_user_id: str = "",
    mention_summary: str = "",
    mention_platform: str = "",
    mention_channel_id: str = "",
    historical_summary: str | None = None,
    memory_scope: str = "global",
) -> None:
    """Create or update an entity memory entry.

    On insert, creates a new document.  On update, appends the mention to
    ``recent_mentions`` (bounded to ``_MAX_RECENT_MENTIONS``) and optionally
    refreshes ``historical_summary`` and ``display_names``.

    Args:
        subject_key: Normalised identifier (lowercase, stripped).
        subject_kind: One of ``topic``, ``person``, ``group``, ``event``.
        display_names: Known surface forms / aliases for the entity.
        resolved_global_user_id: Linked platform user ID if resolved.
        mention_summary: Short factual summary of the current mention.
        mention_platform: Platform where the mention occurred.
        mention_channel_id: Channel where the mention occurred.
        historical_summary: Compressed long-term summary (set or replace).
        memory_scope: ``"global"`` | ``"platform"`` | ``"channel"``.

    Raises:
        ValueError: If ``subject_kind`` is not in the allowed enum.
    """
    if subject_kind not in _VALID_SUBJECT_KINDS:
        raise ValueError(
            f"subject_kind must be one of {_VALID_SUBJECT_KINDS}, got {subject_kind!r}"
        )

    db = await get_db()
    collection = db.entity_memory
    now = datetime.now(timezone.utc).isoformat()

    existing = await collection.find_one({"subject_key": subject_key})

    if existing is None:
        combined_text = f"subject:{subject_key}\nkind:{subject_kind}\nsummary:{mention_summary}"
        embedding = await get_text_embedding(combined_text)

        recent_mentions = []
        if mention_summary:
            recent_mentions.append({
                "timestamp": now,
                "platform": mention_platform,
                "platform_channel_id": mention_channel_id,
                "summary": mention_summary,
            })

        doc = {
            "subject_key": subject_key,
            "subject_kind": subject_kind,
            "display_names": display_names or [subject_key],
            "resolved_global_user_id": resolved_global_user_id,
            "memory_scope": memory_scope,
            "recent_mentions": recent_mentions,
            "historical_summary": historical_summary or "",
            "embedding": embedding,
            "created_at": now,
            "updated_at": now,
        }
        await collection.insert_one(doc)
        logger.debug("Entity memory created: subject_key=%s kind=%s", subject_key, subject_kind)
        return

    # ── Update path ──
    update_ops: dict[str, Any] = {"$set": {"updated_at": now}}

    if display_names:
        merged = list(dict.fromkeys(
            (existing.get("display_names") or []) + display_names
        ))
        update_ops["$set"]["display_names"] = merged

    if resolved_global_user_id:
        update_ops["$set"]["resolved_global_user_id"] = resolved_global_user_id

    if historical_summary is not None:
        update_ops["$set"]["historical_summary"] = historical_summary

    if mention_summary:
        mention_entry = {
            "timestamp": now,
            "platform": mention_platform,
            "platform_channel_id": mention_channel_id,
            "summary": mention_summary,
        }
        # Append and trim to bounded window
        recent = list(existing.get("recent_mentions") or [])
        recent.append(mention_entry)
        update_ops["$set"]["recent_mentions"] = recent[-_MAX_RECENT_MENTIONS:]

    # Refresh embedding with latest summary
    combined_text = f"subject:{subject_key}\nkind:{subject_kind}\nsummary:{mention_summary or existing.get('historical_summary', '')}"
    embedding = await get_text_embedding(combined_text)
    update_ops["$set"]["embedding"] = embedding

    await collection.update_one({"subject_key": subject_key}, update_ops)
    logger.debug("Entity memory updated: subject_key=%s kind=%s", subject_key, subject_kind)


async def search_entity_memory(
    query: str,
    *,
    limit: int = 5,
    method: str = "vector",
    subject_kind: str | None = None,
    memory_scope: str | None = None,
) -> list[tuple[float, dict]]:
    """Search entity memory by keyword or vector similarity.

    Args:
        query: Search query string.
        limit: Maximum results to return.
        method: ``"keyword"`` for regex, ``"vector"`` for semantic.
        subject_kind: Optional filter on entity type.
        memory_scope: Optional filter on scope.

    Returns:
        List of ``(score, doc)`` tuples.  Keyword results have score ``-1.0``.
    """
    db = await get_db()
    collection = db.entity_memory

    extra_filter: dict[str, Any] = {}
    if subject_kind:
        extra_filter["subject_kind"] = subject_kind
    if memory_scope:
        extra_filter["memory_scope"] = memory_scope

    if method == "keyword":
        base_filter: dict[str, Any] = {
            "$or": [
                {"subject_key": {"$regex": query, "$options": "i"}},
                {"display_names": {"$regex": query, "$options": "i"}},
                {"historical_summary": {"$regex": query, "$options": "i"}},
                {"recent_mentions.summary": {"$regex": query, "$options": "i"}},
            ]
        }
        base_filter.update(extra_filter)
        cursor = collection.find(base_filter).limit(limit)
        docs = await cursor.to_list(length=limit)
        for doc in docs:
            doc.pop("embedding", None)
        return [(-1.0, doc) for doc in docs]

    # method == "vector"
    query_embedding = await get_text_embedding(query)
    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": "entity_memory_vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": limit * 10,
                "limit": limit,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
    ]
    if extra_filter:
        pipeline.append({"$match": extra_filter})
    pipeline.append({"$unset": "embedding"})

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=limit)
    return [(doc.pop("score", 0.0), doc) for doc in docs]


async def get_entity_by_key(subject_key: str) -> dict | None:
    """Fetch a single entity memory document by its normalised key.

    Args:
        subject_key: The normalised identifier.

    Returns:
        The document dict (without embedding) or ``None``.
    """
    db = await get_db()
    doc = await db.entity_memory.find_one({"subject_key": subject_key})
    if doc:
        doc.pop("embedding", None)
    return doc


async def get_entity_by_resolved_id(resolved_global_user_id: str) -> dict | None:
    """Fetch entity memory by resolved platform user ID.

    Args:
        resolved_global_user_id: The linked platform user ID.

    Returns:
        The document dict (without embedding) or ``None``.
    """
    db = await get_db()
    doc = await db.entity_memory.find_one(
        {"resolved_global_user_id": resolved_global_user_id}
    )
    if doc:
        doc.pop("embedding", None)
    return doc
