"""Operations against the ``memory`` collection."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.db._client import enable_vector_index, get_db, get_text_embedding
from kazusa_ai_chatbot.db.schemas import MemoryDoc


async def enable_memory_vector_index() -> None:
    """Create the vector search index on the ``memory`` collection."""
    await enable_vector_index("memory", "memory_vector_index")


async def save_memory(doc: MemoryDoc, timestamp: str) -> None:
    """Insert a memory entry (append-only).

    Each call creates a new document. Memories are append-only; deduplication
    and superseding are handled at query time via the ``status`` field.

    Args:
        doc: A dict produced by :func:`build_memory_doc`.
        timestamp: ISO-8601 UTC timestamp for when the memory was created.
    """
    db = await get_db()

    memory_name = doc["memory_name"]
    content = doc["content"]

    combined_text = (
        f"type:{doc.get('memory_type', '')}\n"
        f"source:{doc.get('source_kind', '')}\n"
        f"title:{memory_name}\n"
        f"content:{content}"
    )
    embedding = await get_text_embedding(combined_text)

    payload = {
        **doc,
        "timestamp": timestamp,
        "embedding": embedding,
    }

    await db.memory.insert_one(payload)


async def get_active_promises(
    source_global_user_id: str,
    now_timestamp: str,
    limit: int = 10,
) -> list[MemoryDoc]:
    """Return active, non-expired promise memories for a single user.

    Args:
        source_global_user_id: Internal UUID of the user whose promises should be
            loaded.
        now_timestamp: ISO-8601 timestamp used as the freshness cutoff.
        limit: Maximum number of promises to return, newest first.

    Returns:
        A list of promise memory documents without embeddings.
    """
    db = await get_db()
    cursor = db.memory.find(
        {
            "source_global_user_id": source_global_user_id,
            "memory_type": "promise",
            "status": "active",
            "$or": [
                {"expiry_timestamp": None},
                {"expiry_timestamp": {"$exists": False}},
                {"expiry_timestamp": {"$gt": now_timestamp}},
            ],
        }
    ).sort("timestamp", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for doc in docs:
        doc.pop("embedding", None)
    return docs


async def search_memory(
    query: str,
    limit: int = 5,
    method: str = "vector",  # "keyword", "vector"
    source_global_user_id: str | None = None,
    memory_type: str | None = None,
    source_kind: str | None = None,
    status: str | None = None,
    expiry_before: str | None = None,
    expiry_after: str | None = None,
) -> list[tuple[float, MemoryDoc]]:
    """Search the ``memory`` collection using keyword or vector relevance.

    Args:
        query: The search query string.
        limit: Maximum number of results.
        method: ``"keyword"`` for regex text search, ``"vector"`` for semantic search.
        source_global_user_id: Optional filter on origin user.
        memory_type: Optional filter (e.g. ``"fact"``, ``"promise"``).
        source_kind: Optional filter (e.g. ``"conversation_extracted"``).
        status: Optional filter (e.g. ``"active"``).
        expiry_before: Optional ISO-8601 upper bound on ``expiry_timestamp``.
        expiry_after: Optional ISO-8601 lower bound on ``expiry_timestamp``.

    Returns:
        A list of ``(score, memory_doc)`` tuples. Keyword results have score ``-1.0``.
    """
    db = await get_db()
    collection = db.memory

    extra_filter: dict[str, Any] = {}
    if source_global_user_id:
        extra_filter["source_global_user_id"] = source_global_user_id
    if memory_type:
        extra_filter["memory_type"] = memory_type
    if source_kind:
        extra_filter["source_kind"] = source_kind
    if status:
        extra_filter["status"] = status
    if expiry_before or expiry_after:
        expiry_cond: dict[str, str] = {}
        if expiry_before:
            expiry_cond["$lt"] = expiry_before
        if expiry_after:
            expiry_cond["$gt"] = expiry_after
        extra_filter["expiry_timestamp"] = expiry_cond

    if method == "keyword":
        base_filter: dict[str, Any] = {
            "$or": [
                {"memory_name": {"$regex": query, "$options": "i"}},
                {"content": {"$regex": query, "$options": "i"}},
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
    index_name = "memory_vector_index"

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

    if extra_filter:
        pipeline.append({"$match": extra_filter})

    pipeline.append({"$unset": "embedding"})

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=limit)
    return [(doc.pop("score", 0.0), doc) for doc in docs]
