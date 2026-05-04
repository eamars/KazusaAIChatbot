"""MongoDB interface for evolving shared memory units."""

from __future__ import annotations

from typing import Any

from pymongo.errors import DuplicateKeyError

from kazusa_ai_chatbot.db._client import get_db, get_text_embedding
from kazusa_ai_chatbot.memory_evolution.models import (
    EvolvingMemoryDoc,
    MemorySourceKind,
    MemoryStatus,
    MemoryUnitQuery,
    MemoryUnitSearchResult,
    SEED_MANAGED_SOURCE_KINDS,
)

MEMORY_WRITE_LOCK_ID = "__memory_evolution_write_lock__"
_VECTOR_FILTER_FIELDS = {
    "status",
    "memory_type",
    "source_kind",
    "source_global_user_id",
    "authority",
    "lineage_id",
}
_LIFECYCLE_UPDATE_FIELDS = {
    "status",
    "updated_at",
    "expiry_timestamp",
}


async def compute_memory_embedding(text: str) -> list[float]:
    """Compute an embedding through the configured embedding adapter.

    Args:
        text: Semantic memory text.

    Returns:
        Embedding vector.
    """
    embedding = await get_text_embedding(text)
    return embedding


def build_active_memory_filter(now_timestamp: str) -> dict[str, Any]:
    """Build the MongoDB filter for active, non-expired memory rows.

    Args:
        now_timestamp: ISO timestamp used as the expiry cutoff.

    Returns:
        MongoDB filter fragment.
    """
    return_value = {
        "$and": [
            {
                "$or": [
                    {"expiry_timestamp": None},
                    {"expiry_timestamp": {"$exists": False}},
                    {"expiry_timestamp": {"$gt": now_timestamp}},
                ]
            },
            {"status": MemoryStatus.ACTIVE},
        ]
    }
    return return_value


def _active_query_filter(
    query: MemoryUnitQuery,
    *,
    now_timestamp: str,
) -> dict[str, Any]:
    filter_doc = build_active_memory_filter(now_timestamp)
    for field in (
        "memory_name",
        "source_global_user_id",
        "memory_type",
        "source_kind",
        "authority",
        "lineage_id",
    ):
        value = query.get(field)
        if value:
            filter_doc[field] = value

    memory_name_contains = query.get("memory_name_contains")
    if memory_name_contains:
        filter_doc["memory_name"] = {
            "$regex": memory_name_contains,
            "$options": "i",
        }

    exclude_ids = query.get("exclude_memory_unit_ids")
    if exclude_ids:
        filter_doc["memory_unit_id"] = {"$nin": exclude_ids}
    return filter_doc


def _vector_prefilter(query: MemoryUnitQuery) -> dict[str, Any]:
    """Build the vector-index prefilter for indexed scalar fields.

    Args:
        query: Constrained query shape.

    Returns:
        Filter document safe for ``$vectorSearch.filter``.
    """
    filter_doc: dict[str, Any] = {"status": MemoryStatus.ACTIVE}
    for field in _VECTOR_FILTER_FIELDS - {"status"}:
        value = query.get(field)
        if value:
            filter_doc[field] = value
    return filter_doc


def _validate_lifecycle_update_fields(fields: dict[str, Any]) -> None:
    """Require DB lifecycle updates to stay within their narrow contract."""
    unsupported_fields = set(fields) - _LIFECYCLE_UPDATE_FIELDS
    if unsupported_fields:
        raise ValueError(
            f"unsupported memory lifecycle update fields: {sorted(unsupported_fields)}"
        )


async def find_memory_unit_by_id(memory_unit_id: str) -> EvolvingMemoryDoc | None:
    """Find one memory unit by stable id.

    Args:
        memory_unit_id: Stable memory-unit id.

    Returns:
        Matching document or ``None``.
    """
    db = await get_db()
    document = await db.memory.find_one({"memory_unit_id": memory_unit_id})
    return_value = dict(document) if document is not None else None
    return return_value


async def insert_memory_unit_document(document: EvolvingMemoryDoc) -> None:
    """Insert one memory-unit document.

    Args:
        document: Fully prepared document including embedding.
    """
    db = await get_db()
    await db.memory.insert_one(document)


async def replace_memory_unit_document(document: EvolvingMemoryDoc) -> None:
    """Replace or insert a memory-unit document by id.

    Args:
        document: Fully prepared document including embedding.
    """
    db = await get_db()
    await db.memory.replace_one(
        {"memory_unit_id": document["memory_unit_id"]},
        document,
        upsert=True,
    )


async def update_memory_unit_fields(
    memory_unit_id: str,
    fields: dict[str, Any],
) -> None:
    """Update lifecycle fields for one memory unit.

    Args:
        memory_unit_id: Stable memory-unit id.
        fields: Lifecycle fields to set.
    """
    _validate_lifecycle_update_fields(fields)
    db = await get_db()
    await db.memory.update_one(
        {"memory_unit_id": memory_unit_id},
        {"$set": fields},
    )


async def update_many_memory_unit_fields(
    memory_unit_ids: list[str],
    fields: dict[str, Any],
) -> None:
    """Update lifecycle fields for many memory units.

    Args:
        memory_unit_ids: Stable ids to update.
        fields: Lifecycle fields to set.
    """
    _validate_lifecycle_update_fields(fields)
    db = await get_db()
    await db.memory.update_many(
        {"memory_unit_id": {"$in": memory_unit_ids}},
        {"$set": fields},
    )


async def find_active_memory_documents(
    *,
    query: MemoryUnitQuery,
    limit: int,
    now_timestamp: str,
    query_embedding: list[float] | None,
) -> list[MemoryUnitSearchResult]:
    """Find active memory documents with retrieval scores through MongoDB.

    Args:
        query: Constrained query shape.
        limit: Maximum rows to return.
        now_timestamp: ISO timestamp used as the expiry cutoff.
        query_embedding: Optional semantic query embedding.

    Returns:
        ``(score, memory document)`` pairs without embeddings. Metadata-only
        reads use score ``-1.0``.
    """
    db = await get_db()
    filter_doc = _active_query_filter(query, now_timestamp=now_timestamp)
    if query_embedding is not None:
        vector_filter = _vector_prefilter(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "memory_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": max(100, limit * 10),
                    "limit": max(100, limit * 10),
                    "filter": vector_filter,
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$match": filter_doc},
            {"$unset": "embedding"},
            {"$limit": limit},
        ]
        cursor = db.memory.aggregate(pipeline)
        docs = await cursor.to_list(length=limit)
        return_value = []
        for doc in docs:
            document = dict(doc)
            score = float(document.pop("score", 0.0))
            document.pop("embedding", None)
            return_value.append((score, document))
        return return_value

    cursor = (
        db.memory
        .find(filter_doc, {"embedding": 0})
        .sort([("updated_at", -1), ("timestamp", -1)])
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    return_value = []
    for doc in docs:
        document = dict(doc)
        document.pop("embedding", None)
        return_value.append((-1.0, document))
    return return_value


async def count_legacy_seed_managed_memory() -> int:
    """Count seed-managed legacy rows without evolving ids."""
    db = await get_db()
    count = await db.memory.count_documents({
        "$and": [
            {
                "source_global_user_id": "",
                "source_kind": {"$in": sorted(SEED_MANAGED_SOURCE_KINDS)},
            },
            {"memory_unit_id": {"$exists": False}},
        ]
    })
    return count


async def count_unmanaged_seed_memory(seed_ids: list[str]) -> int:
    """Count seed-managed rows absent from the current seed file.

    Args:
        seed_ids: Current deterministic seed memory-unit ids.

    Returns:
        Number of stale seed-managed rows.
    """
    db = await get_db()
    count = await db.memory.count_documents({
        "$and": [
            {
                "source_global_user_id": "",
                "source_kind": {"$in": sorted(SEED_MANAGED_SOURCE_KINDS)},
            },
            {"memory_unit_id": {"$exists": True}},
            {"memory_unit_id": {"$nin": seed_ids}},
        ]
    })
    return count


async def count_runtime_reflection_memory() -> int:
    """Count future runtime reflection-inferred memory rows."""
    db = await get_db()
    count = await db.memory.count_documents({
        "source_kind": MemorySourceKind.REFLECTION_INFERRED,
    })
    return count


async def delete_reset_seed_managed_memory(seed_ids: list[str]) -> int:
    """Delete reset-managed seed-lane rows not represented by current seeds.

    Args:
        seed_ids: Current deterministic seed memory-unit ids.

    Returns:
        Deleted row count.
    """
    db = await get_db()
    result = await db.memory.delete_many({
        "$and": [
            {
                "source_global_user_id": "",
                "source_kind": {"$in": sorted(SEED_MANAGED_SOURCE_KINDS)},
            },
            {
                "$or": [
                    {"memory_unit_id": {"$exists": False}},
                    {"memory_unit_id": {"$nin": seed_ids}},
                ]
            },
        ]
    })
    return_value = int(result.deleted_count)
    return return_value


async def acquire_memory_write_lock(owner: str, write_time: str) -> bool:
    """Try to acquire the shared memory write guard document.

    Args:
        owner: Operation name requesting the lock.
        write_time: Creation timestamp.

    Returns:
        True when acquired, False when another memory write holds the lock.
    """
    db = await get_db()
    try:
        await db.memory.insert_one({
            "_id": MEMORY_WRITE_LOCK_ID,
            "lock_type": "memory_evolution_write",
            "owner": owner,
            "created_at": write_time,
        })
    except DuplicateKeyError:
        return False
    return True


async def release_memory_write_lock() -> None:
    """Release the shared memory write guard document."""
    db = await get_db()
    await db.memory.delete_one({"_id": MEMORY_WRITE_LOCK_ID})
