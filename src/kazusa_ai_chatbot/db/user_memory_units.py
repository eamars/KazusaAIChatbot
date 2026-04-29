"""Persistence helpers for fact-anchored user memory units."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db._client import get_db, get_text_embedding
from kazusa_ai_chatbot.db.schemas import (
    UserMemoryUnitDoc,
    UserMemoryUnitStatus,
    UserMemoryUnitType,
)
from kazusa_ai_chatbot.utils import text_or_empty

logger = logging.getLogger(__name__)


VALID_USER_MEMORY_UNIT_TYPES = {
    UserMemoryUnitType.STABLE_PATTERN,
    UserMemoryUnitType.RECENT_SHIFT,
    UserMemoryUnitType.OBJECTIVE_FACT,
    UserMemoryUnitType.MILESTONE,
    UserMemoryUnitType.ACTIVE_COMMITMENT,
}

VALID_USER_MEMORY_UNIT_STATUSES = {
    UserMemoryUnitStatus.ACTIVE,
    UserMemoryUnitStatus.ARCHIVED,
    UserMemoryUnitStatus.COMPLETED,
    UserMemoryUnitStatus.CANCELLED,
}


def _now_iso() -> str:
    current_time = datetime.now(timezone.utc).isoformat()
    return current_time


def _semantic_text(unit: dict) -> str:
    semantic_text = "\n".join(
        part
        for part in (
            text_or_empty(unit.get("fact")),
            text_or_empty(unit.get("subjective_appraisal")),
            text_or_empty(unit.get("relationship_signal")),
        )
        if part
    )
    return semantic_text


def validate_user_memory_unit_semantics(unit: dict) -> None:
    """Validate the semantic triple and routing fields for one memory unit.

    Args:
        unit: Candidate or stored memory-unit dictionary.

    Raises:
        ValueError: If required fields are absent or structurally invalid.
    """

    unit_type = text_or_empty(unit.get("unit_type"))
    if unit_type not in VALID_USER_MEMORY_UNIT_TYPES:
        raise ValueError(f"invalid user memory unit type: {unit_type!r}")

    status = text_or_empty(unit.get("status")) or UserMemoryUnitStatus.ACTIVE
    if status not in VALID_USER_MEMORY_UNIT_STATUSES:
        raise ValueError(f"invalid user memory unit status: {status!r}")

    for field in ("fact", "subjective_appraisal", "relationship_signal"):
        if not text_or_empty(unit.get(field)):
            raise ValueError(f"missing user memory unit field: {field}")


def build_user_memory_unit_doc(
    global_user_id: str,
    unit: dict,
    *,
    timestamp: str | None = None,
    unit_id: str | None = None,
) -> UserMemoryUnitDoc:
    """Build a structurally valid memory-unit document.

    Args:
        global_user_id: Internal UUID for the memory owner.
        unit: LLM-authored semantic unit with ``unit_type`` and triple fields.
        timestamp: Write timestamp. Defaults to current UTC time.
        unit_id: Optional stable id, primarily for tests or migrations.

    Returns:
        A document ready for persistence in ``user_memory_units``.
    """

    validate_user_memory_unit_semantics(unit)
    write_time = timestamp or _now_iso()
    status = text_or_empty(unit.get("status")) or UserMemoryUnitStatus.ACTIVE
    count = unit.get("count")
    source_refs = unit.get("source_refs")
    merge_history = unit.get("merge_history")

    return {
        "unit_id": unit_id or text_or_empty(unit.get("unit_id")) or uuid4().hex,
        "global_user_id": global_user_id,
        "unit_type": text_or_empty(unit["unit_type"]),
        "fact": text_or_empty(unit["fact"]),
        "subjective_appraisal": text_or_empty(unit["subjective_appraisal"]),
        "relationship_signal": text_or_empty(unit["relationship_signal"]),
        "status": status,
        "count": count if isinstance(count, int) and count > 0 else 1,
        "first_seen_at": text_or_empty(unit.get("first_seen_at")) or write_time,
        "last_seen_at": text_or_empty(unit.get("last_seen_at")) or write_time,
        "updated_at": write_time,
        "source_refs": source_refs if isinstance(source_refs, list) else [],
        "merge_history": merge_history if isinstance(merge_history, list) else [],
        "due_at": unit.get("due_at"),
        "completed_at": unit.get("completed_at"),
        "cancelled_at": unit.get("cancelled_at"),
    }


async def insert_user_memory_units(
    global_user_id: str,
    units: list[dict],
    *,
    timestamp: str | None = None,
    include_embeddings: bool = True,
) -> list[UserMemoryUnitDoc]:
    """Insert new memory units for one user.

    Args:
        global_user_id: Internal UUID for the memory owner.
        units: LLM-authored memory unit dictionaries.
        timestamp: Optional write timestamp.
        include_embeddings: Whether to compute vector embeddings before insert.

    Returns:
        The persisted documents.
    """

    docs = [
        build_user_memory_unit_doc(global_user_id, unit, timestamp=timestamp)
        for unit in units
    ]
    if not docs:
        return []

    if include_embeddings:
        for doc in docs:
            doc["embedding"] = await get_text_embedding(_semantic_text(doc))

    db = await get_db()
    await db.user_memory_units.insert_many(docs)
    return docs


async def query_user_memory_units(
    global_user_id: str,
    *,
    unit_types: list[str] | None = None,
    statuses: list[str] | None = None,
    limit: int = 100,
) -> list[UserMemoryUnitDoc]:
    """Read user memory units for projection or merge-candidate retrieval.

    Args:
        global_user_id: Internal UUID for the memory owner.
        unit_types: Optional unit-type filter.
        statuses: Optional status filter. Defaults to active units.
        limit: Maximum documents to return.

    Returns:
        Matching memory-unit documents sorted by recency.
    """

    query: dict = {"global_user_id": global_user_id}
    if unit_types:
        query["unit_type"] = {"$in": unit_types}
    query["status"] = {"$in": statuses or [UserMemoryUnitStatus.ACTIVE]}

    db = await get_db()
    cursor = (
        db.user_memory_units
        .find(query, {"_id": 0, "embedding": 0})
        .sort([("last_seen_at", -1), ("updated_at", -1)])
        .limit(limit)
    )
    return [doc async for doc in cursor]


async def search_user_memory_units_by_vector(
    global_user_id: str,
    embedding: list[float],
    *,
    unit_types: list[str] | None = None,
    statuses: list[str] | None = None,
    limit: int = 25,
) -> list[UserMemoryUnitDoc]:
    """Run semantic retrieval over user memory units.

    Args:
        global_user_id: Internal UUID for the memory owner.
        embedding: Query embedding generated by RAG for the current task.
        unit_types: Optional unit-type filter.
        statuses: Optional status filter. Defaults to active units.
        limit: Maximum semantic hits to return.

    Returns:
        Matching memory-unit documents with Mongo internals and embeddings
        removed. If vector search is unavailable, returns an empty list so the
        caller can still use the recency path.
    """

    if not global_user_id or not embedding:
        return []

    vector_filter: dict = {
        "global_user_id": global_user_id,
        "status": {"$in": statuses or [UserMemoryUnitStatus.ACTIVE]},
    }
    if unit_types:
        vector_filter["unit_type"] = {"$in": unit_types}

    db = await get_db()
    pipeline = [
        {
            "$vectorSearch": {
                "index": "user_memory_units_vector",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": max(100, limit * 10),
                "limit": limit,
                "filter": vector_filter,
            }
        },
        {"$unset": ["_id", "embedding"]},
    ]
    try:
        cursor = db.user_memory_units.aggregate(pipeline)
        return [doc async for doc in cursor]
    except PyMongoError:
        logger.warning("user_memory_units vector search unavailable; falling back to recency")
        return []


async def update_user_memory_unit_semantics(
    unit_id: str,
    updated_unit: dict,
    *,
    timestamp: str | None = None,
    merge_history_entry: dict | None = None,
    increment_count: bool = True,
) -> None:
    """Update semantic fields and lifecycle metadata for an existing unit.

    Args:
        unit_id: Stable memory-unit id.
        updated_unit: LLM-authored replacement semantic triple.
        timestamp: Optional write timestamp.
        merge_history_entry: Optional merge/evolve audit row.
        increment_count: Whether to increment reinforcement count.
    """

    for field in ("fact", "subjective_appraisal", "relationship_signal"):
        if not text_or_empty(updated_unit.get(field)):
            raise ValueError(f"missing rewritten user memory unit field: {field}")

    write_time = timestamp or _now_iso()
    set_doc = {
        "fact": text_or_empty(updated_unit["fact"]),
        "subjective_appraisal": text_or_empty(updated_unit["subjective_appraisal"]),
        "relationship_signal": text_or_empty(updated_unit["relationship_signal"]),
        "last_seen_at": write_time,
        "updated_at": write_time,
        "embedding": await get_text_embedding(_semantic_text(updated_unit)),
    }
    update_doc: dict = {"$set": set_doc}
    if increment_count:
        update_doc["$inc"] = {"count": 1}
    if merge_history_entry:
        update_doc["$push"] = {"merge_history": merge_history_entry}

    db = await get_db()
    await db.user_memory_units.update_one({"unit_id": unit_id}, update_doc)


async def update_user_memory_unit_window(
    unit_id: str,
    *,
    window: str,
    timestamp: str | None = None,
) -> None:
    """Apply an LLM-selected recent/stable window to one pattern unit.

    Args:
        unit_id: Stable memory-unit id.
        window: LLM-selected ``recent`` or ``stable`` value.
        timestamp: Optional write timestamp.
    """

    if window == "stable":
        unit_type = UserMemoryUnitType.STABLE_PATTERN
    elif window == "recent":
        unit_type = UserMemoryUnitType.RECENT_SHIFT
    else:
        raise ValueError(f"invalid memory unit window: {window!r}")

    db = await get_db()
    await db.user_memory_units.update_one(
        {"unit_id": unit_id},
        {"$set": {"unit_type": unit_type, "updated_at": timestamp or _now_iso()}},
    )
