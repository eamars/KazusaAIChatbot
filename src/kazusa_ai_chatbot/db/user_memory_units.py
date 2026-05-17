"""Persistence helpers for fact-anchored user memory units."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from uuid import uuid4

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import (
    RAG_VECTOR_CANDIDATE_MULTIPLIER,
    RAG_VECTOR_MAX_CANDIDATES,
    RAG_VECTOR_MIN_CANDIDATES,
)
from kazusa_ai_chatbot.db._client import get_db, get_document_text_embedding
from kazusa_ai_chatbot.db.schemas import (
    UserMemoryUnitDoc,
    UserMemoryUnitStatus,
    UserMemoryUnitType,
)
from kazusa_ai_chatbot.time_boundary import (
    parse_storage_utc_datetime,
    storage_utc_now_iso,
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

ACTIVE_COMMITMENT_DUE_BUCKET_READY = 0
ACTIVE_COMMITMENT_DUE_BUCKET_FUTURE = 1


def _now_iso() -> str:
    current_time = storage_utc_now_iso()
    return current_time


def _parse_datetime_for_query(value: str) -> datetime:
    """Parse stored ISO-like timestamps into aware UTC datetimes."""

    return_value = parse_storage_utc_datetime(value)
    return return_value


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
    storage_timestamp_utc: str | None = None,
    unit_id: str | None = None,
) -> UserMemoryUnitDoc:
    """Build a structurally valid memory-unit document.

    Args:
        global_user_id: Internal UUID for the memory owner.
        unit: LLM-authored semantic unit with ``unit_type`` and triple fields.
        storage_timestamp_utc: Write storage UTC timestamp. Defaults to the
            current storage UTC time.
        unit_id: Optional stable id, primarily for tests or migrations.

    Returns:
        A document ready for persistence in ``user_memory_units``.
    """

    validate_user_memory_unit_semantics(unit)
    write_time = storage_timestamp_utc or _now_iso()
    status = text_or_empty(unit.get("status")) or UserMemoryUnitStatus.ACTIVE
    count = unit.get("count")
    source_refs = unit.get("source_refs")
    merge_history = unit.get("merge_history")

    return_value = {
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
        "archived_at": unit.get("archived_at"),
    }
    return return_value


async def insert_user_memory_units(
    global_user_id: str,
    units: list[dict],
    *,
    storage_timestamp_utc: str | None = None,
    include_embeddings: bool = True,
) -> list[UserMemoryUnitDoc]:
    """Insert new memory units for one user.

    Args:
        global_user_id: Internal UUID for the memory owner.
        units: LLM-authored memory unit dictionaries.
        storage_timestamp_utc: Optional write storage UTC timestamp.
        include_embeddings: Whether to compute vector embeddings before insert.

    Returns:
        The persisted documents.
    """

    docs = [
        build_user_memory_unit_doc(
            global_user_id,
            unit,
            storage_timestamp_utc=storage_timestamp_utc,
        )
        for unit in units
    ]
    if not docs:
        return_value = []
        return return_value

    if include_embeddings:
        for doc in docs:
            doc["embedding"] = await get_document_text_embedding(
                _semantic_text(doc)
            )

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
    return_value = [doc async for doc in cursor]
    return return_value


async def query_active_commitment_memory_units(
    *,
    current_timestamp_utc: str,
    limit: int = 100,
) -> list[UserMemoryUnitDoc]:
    """Read active commitment units across users for idle due checks.

    Args:
        current_timestamp_utc: Worker tick storage UTC timestamp used to
            prioritize due or past-due commitments before future commitments.
        limit: Maximum documents to return.

    Returns:
        Active commitment memory units without embeddings.
    """

    current_time = _parse_datetime_for_query(current_timestamp_utc)
    db = await get_db()
    pipeline = [
        {
            "$match": {
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "due_at": {"$type": "string", "$ne": ""},
            }
        },
        {
            "$addFields": {
                "_self_cognition_due_at": {
                    "$dateFromString": {
                        "dateString": {
                            "$replaceOne": {
                                "input": "$due_at",
                                "find": " ",
                                "replacement": "T",
                            }
                        },
                        "onError": None,
                        "onNull": None,
                    }
                },
            }
        },
        {"$match": {"_self_cognition_due_at": {"$ne": None}}},
        {
            "$addFields": {
                "_self_cognition_due_bucket": {
                    "$cond": [
                        {"$lte": ["$_self_cognition_due_at", current_time]},
                        ACTIVE_COMMITMENT_DUE_BUCKET_READY,
                        ACTIVE_COMMITMENT_DUE_BUCKET_FUTURE,
                    ]
                }
            }
        },
        {
            "$sort": {
                "_self_cognition_due_bucket": 1,
                "_self_cognition_due_at": 1,
                "last_seen_at": -1,
                "updated_at": -1,
            }
        },
        {"$limit": limit},
        {
            "$project": {
                "_id": 0,
                "embedding": 0,
                "_self_cognition_due_at": 0,
                "_self_cognition_due_bucket": 0,
            }
        },
    ]
    cursor = db.user_memory_units.aggregate(pipeline)
    return_value = [doc async for doc in cursor]
    return return_value


async def search_user_memory_units_by_keyword(
    global_user_id: str,
    keyword: str,
    *,
    unit_types: list[str] | None = None,
    statuses: list[str] | None = None,
    limit: int = 25,
) -> list[UserMemoryUnitDoc]:
    """Run scoped lexical retrieval over durable user memory units.

    Args:
        global_user_id: Internal UUID for the memory owner.
        keyword: Literal term or phrase to match against memory semantics.
        unit_types: Optional unit-type filter.
        statuses: Optional status filter. Defaults to active units.
        limit: Maximum keyword hits to return.

    Returns:
        Matching memory-unit documents sorted by recency, with Mongo internals
        and embeddings removed.
    """

    stripped_keyword = text_or_empty(keyword).strip()
    if not global_user_id or not stripped_keyword:
        return_value = []
        return return_value

    escaped_keyword = re.escape(stripped_keyword)
    regex_filter = {"$regex": escaped_keyword, "$options": "i"}
    query: dict = {
        "global_user_id": global_user_id,
        "status": {"$in": statuses or [UserMemoryUnitStatus.ACTIVE]},
        "$or": [
            {"fact": regex_filter},
            {"subjective_appraisal": regex_filter},
            {"relationship_signal": regex_filter},
        ],
    }
    if unit_types:
        query["unit_type"] = {"$in": unit_types}

    db = await get_db()
    cursor = (
        db.user_memory_units
        .find(query, {"_id": 0, "embedding": 0})
        .sort([("last_seen_at", -1), ("updated_at", -1)])
        .limit(limit)
    )
    return_value = [doc async for doc in cursor]
    return return_value


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
        return_value = []
        return return_value

    vector_filter: dict = {
        "global_user_id": global_user_id,
        "status": {"$in": statuses or [UserMemoryUnitStatus.ACTIVE]},
    }
    if unit_types:
        vector_filter["unit_type"] = {"$in": unit_types}

    db = await get_db()
    candidate_count = max(
        RAG_VECTOR_MIN_CANDIDATES,
        limit * RAG_VECTOR_CANDIDATE_MULTIPLIER,
    )
    candidate_count = min(candidate_count, RAG_VECTOR_MAX_CANDIDATES)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "user_memory_units_vector",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": candidate_count,
                "limit": limit,
                "filter": vector_filter,
            }
        },
        {"$unset": ["_id", "embedding"]},
    ]
    try:
        cursor = db.user_memory_units.aggregate(pipeline)
        return_value = [doc async for doc in cursor]
        return return_value
    except PyMongoError as exc:
        logger.warning(
            f"user_memory_units vector search unavailable; "
            f"falling back to recency: {exc}"
        )
        return_value = []
        return return_value


async def update_user_memory_unit_semantics(
    unit_id: str,
    updated_unit: dict,
    *,
    storage_timestamp_utc: str | None = None,
    lifecycle_fields: dict | None = None,
    merge_history_entry: dict | None = None,
    increment_count: bool = True,
) -> None:
    """Update semantic fields and lifecycle metadata for an existing unit.

    Args:
        unit_id: Stable memory-unit id.
        updated_unit: LLM-authored replacement semantic triple.
        storage_timestamp_utc: Optional write storage UTC timestamp.
        lifecycle_fields: Optional structural lifecycle fields to preserve
            from the extractor, such as due_at.
        merge_history_entry: Optional merge/evolve audit row.
        increment_count: Whether to increment reinforcement count.
    """

    for field in ("fact", "subjective_appraisal", "relationship_signal"):
        if not text_or_empty(updated_unit.get(field)):
            raise ValueError(f"missing rewritten user memory unit field: {field}")

    write_time = storage_timestamp_utc or _now_iso()
    set_doc = {
        "fact": text_or_empty(updated_unit["fact"]),
        "subjective_appraisal": text_or_empty(updated_unit["subjective_appraisal"]),
        "relationship_signal": text_or_empty(updated_unit["relationship_signal"]),
        "last_seen_at": write_time,
        "updated_at": write_time,
        "embedding": await get_document_text_embedding(
            _semantic_text(updated_unit)
        ),
    }
    if lifecycle_fields:
        for field in ("due_at", "completed_at", "cancelled_at"):
            if field in lifecycle_fields:
                set_doc[field] = lifecycle_fields[field]

    update_doc: dict = {"$set": set_doc}
    if increment_count:
        update_doc["$inc"] = {"count": 1}
    if merge_history_entry:
        update_doc["$push"] = {"merge_history": merge_history_entry}

    db = await get_db()
    await db.user_memory_units.update_one({"unit_id": unit_id}, update_doc)


async def update_user_memory_unit_lifecycle(
    unit_id: str,
    *,
    status: str,
    storage_timestamp_utc: str,
    reason: str,
    action_attempt_id: str,
    due_at: str | None = None,
) -> dict[str, object]:
    """Apply a private lifecycle action to one active commitment unit.

    Args:
        unit_id: Stable ``user_memory_units.unit_id`` selected by cognition.
        status: Collection-native lifecycle status to write.
        storage_timestamp_utc: Storage UTC timestamp for the update and audit
            row.
        reason: Cognition-authored semantic reason for the lifecycle change.
        action_attempt_id: Action-attempt identifier used for audit lineage.
        due_at: Optional due timestamp copied from the action evidence.

    Returns:
        Update counts and the merge-history audit row.

    Raises:
        ValueError: If the lifecycle request is structurally invalid.
    """

    if not text_or_empty(unit_id):
        raise ValueError("unit_id is required")
    if status not in VALID_USER_MEMORY_UNIT_STATUSES:
        raise ValueError(f"invalid user memory unit status: {status!r}")
    if not text_or_empty(storage_timestamp_utc):
        raise ValueError("storage_timestamp_utc is required")
    if not text_or_empty(reason):
        raise ValueError("reason is required")
    if not text_or_empty(action_attempt_id):
        raise ValueError("action_attempt_id is required")

    merge_history_entry = {
        "operation": "lifecycle_update",
        "status": status,
        "reason": reason,
        "action_attempt_id": action_attempt_id,
        "timestamp": storage_timestamp_utc,
    }
    set_doc: dict[str, object] = {
        "status": status,
        "updated_at": storage_timestamp_utc,
    }
    if due_at is not None:
        set_doc["due_at"] = due_at
        merge_history_entry["due_at"] = due_at
    if status == UserMemoryUnitStatus.COMPLETED:
        set_doc["completed_at"] = storage_timestamp_utc
    elif status == UserMemoryUnitStatus.CANCELLED:
        set_doc["cancelled_at"] = storage_timestamp_utc
    elif status == UserMemoryUnitStatus.ARCHIVED:
        set_doc["archived_at"] = storage_timestamp_utc

    db = await get_db()
    result = await db.user_memory_units.update_one(
        {
            "unit_id": unit_id,
            "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
            "status": UserMemoryUnitStatus.ACTIVE,
        },
        {
            "$set": set_doc,
            "$push": {"merge_history": merge_history_entry},
        },
    )
    return_value = {
        "unit_id": unit_id,
        "status": status,
        "matched_count": result.matched_count,
        "modified_count": result.modified_count,
        "merge_history_entry": merge_history_entry,
    }
    return return_value


async def update_user_memory_unit_window(
    unit_id: str,
    *,
    window: str,
    storage_timestamp_utc: str | None = None,
) -> None:
    """Apply an LLM-selected recent/stable window to one pattern unit.

    Args:
        unit_id: Stable memory-unit id.
        window: LLM-selected ``recent`` or ``stable`` value.
        storage_timestamp_utc: Optional write storage UTC timestamp.
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
        {
            "$set": {
                "unit_type": unit_type,
                "updated_at": storage_timestamp_utc or _now_iso(),
            }
        },
    )
