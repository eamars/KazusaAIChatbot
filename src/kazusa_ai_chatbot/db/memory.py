"""Operations against the ``memory`` collection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot.db._client import enable_vector_index, get_db, get_text_embedding
from kazusa_ai_chatbot.db.memory_evolution import build_active_memory_filter
from kazusa_ai_chatbot.db.schemas import MemoryDoc
from kazusa_ai_chatbot.memory_evolution.repository import insert_memory_unit
from kazusa_ai_chatbot.memory_evolution.models import (
    MemoryAuthority,
    MemorySourceKind,
    MemoryStatus,
)
from kazusa_ai_chatbot.memory_evolution.identity import (
    deterministic_memory_unit_id,
    memory_embedding_source_text as evolving_memory_embedding_source_text,
)


def _now_iso() -> str:
    current_time = datetime.now(timezone.utc).isoformat()
    return current_time


async def enable_memory_vector_index() -> None:
    """Create the vector search index on the ``memory`` collection."""
    await enable_vector_index(
        "memory",
        "memory_vector_index",
        filter_paths=[
            "status",
            "memory_type",
            "source_kind",
            "source_global_user_id",
            "authority",
            "lineage_id",
        ],
    )


def memory_embedding_source_text(doc: MemoryDoc | dict) -> str:
    """Build the semantic source text used for memory embeddings.

    Args:
        doc: Stored memory document or memory payload.

    Returns:
        Combined text matching the vectorization contract for ``memory`` rows.
    """
    source_text = evolving_memory_embedding_source_text(doc)
    return source_text


def _legacy_memory_unit_id(doc: MemoryDoc) -> str:
    """Build a stable id for callers still using ``save_memory``.

    Args:
        doc: Legacy memory document without evolving ids.

    Returns:
        Stable id derived from the legacy semantic payload.
    """
    memory_unit_id = str(doc.get("memory_unit_id", "")).strip()
    if memory_unit_id:
        return memory_unit_id

    return_value = deterministic_memory_unit_id(
        "manual",
        [
            str(doc.get("memory_name", "")).strip(),
            str(doc.get("source_global_user_id", "")).strip(),
            str(doc.get("source_kind", "")).strip(),
            str(doc.get("memory_type", "")).strip(),
            str(doc.get("content", "")).strip(),
        ],
    )
    return return_value


def _legacy_memory_authority(doc: MemoryDoc) -> str:
    """Infer authority for callers still using ``save_memory``.

    Args:
        doc: Legacy memory document without an explicit evolving authority.

    Returns:
        Existing authority when provided, otherwise the schema authority lane
        that best matches the legacy source kind.
    """
    authority = str(doc.get("authority", "")).strip()
    if authority:
        return authority

    source_kind = str(doc.get("source_kind", "")).strip()
    if source_kind in {
        MemorySourceKind.CONVERSATION_EXTRACTED,
        MemorySourceKind.REFLECTION_INFERRED,
        MemorySourceKind.RELATIONSHIP_INFERRED,
    }:
        return MemoryAuthority.REFLECTION_PROMOTED

    return_value = MemoryAuthority.MANUAL
    return return_value


async def save_memory(doc: MemoryDoc, timestamp: str) -> None:
    """Insert a memory entry through the evolving memory-unit API.

    Legacy callers that do not yet provide evolving ids receive a deterministic
    id derived from the semantic payload so retries do not duplicate rows.

    Args:
        doc: A dict produced by :func:`build_memory_doc`.
        timestamp: ISO-8601 UTC timestamp for when the memory was created.
    """
    if "embedding" in doc:
        raise ValueError("save_memory callers must not provide embedding")

    memory_unit_id = _legacy_memory_unit_id(doc)
    lineage_id = str(doc.get("lineage_id", "")).strip() or memory_unit_id
    payload = {
        **doc,
        "memory_unit_id": memory_unit_id,
        "lineage_id": lineage_id,
        "version": doc.get("version", 1),
        "authority": _legacy_memory_authority(doc),
        "timestamp": timestamp,
    }

    await insert_memory_unit(document=payload)


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
    status: str | None = MemoryStatus.ACTIVE,
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
        status: Optional filter. Defaults to ``"active"``. Pass ``None`` only
            for audit/debug reads that need every lifecycle state.
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
    if status == MemoryStatus.ACTIVE:
        extra_filter.update(build_active_memory_filter(_now_iso()))
    elif status:
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
        return_value = [(-1.0, doc) for doc in docs]
        return return_value

    # method == "vector"
    query_embedding = await get_text_embedding(query)
    index_name = "memory_vector_index"
    vector_filter: dict[str, str] = {}
    if source_global_user_id:
        vector_filter["source_global_user_id"] = source_global_user_id
    if memory_type:
        vector_filter["memory_type"] = memory_type
    if source_kind:
        vector_filter["source_kind"] = source_kind
    if status == MemoryStatus.ACTIVE:
        vector_filter["status"] = MemoryStatus.ACTIVE
    elif status:
        vector_filter["status"] = status

    vector_search = {
        "index": index_name,
        "path": "embedding",
        "queryVector": query_embedding,
        "numCandidates": max(100, limit * 10),
        "limit": max(100, limit * 10),
    }
    if vector_filter:
        vector_search["filter"] = vector_filter

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": vector_search
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
    ]

    if extra_filter:
        pipeline.append({"$match": extra_filter})

    pipeline.append({"$unset": "embedding"})
    pipeline.append({"$limit": limit})

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=limit)
    return_value = [(doc.pop("score", 0.0), doc) for doc in docs]
    return return_value
