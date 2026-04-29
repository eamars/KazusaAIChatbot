"""Persistent backing storage for selected RAG Cache2 entries."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import RAG_CACHE2_MAX_ENTRIES
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.schemas import RAGCache2PersistentEntryDoc
from kazusa_ai_chatbot.rag.cache2_policy import (
    INITIALIZER_AGENT_REGISTRY_VERSION,
    INITIALIZER_CACHE_NAME,
    INITIALIZER_POLICY_VERSION,
    INITIALIZER_PROMPT_VERSION,
    INITIALIZER_STRATEGY_SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

PERSISTENT_CACHE_COLLECTION = "rag_cache2_persistent_entries"
PERSISTENT_CACHE_LOOKUP_INDEX = "cache2_persistent_lookup_idx"
PERSISTENT_CACHE_LOOKUP_KEYS = [
    ("cache_name", ASCENDING),
    ("version_key", ASCENDING),
    ("hit_count", DESCENDING),
    ("updated_at", DESCENDING),
]


def _now_iso() -> str:
    current_time = datetime.now(timezone.utc).isoformat()
    return current_time


async def _get_collection():
    db = await get_db()
    collection = db[PERSISTENT_CACHE_COLLECTION]
    return collection


def build_initializer_version_key() -> str:
    """Build the durable version fingerprint for initializer strategies.

    Returns:
        A human-readable pipe-joined version key made from the initializer
        policy, prompt, agent-registry, and strategy-schema versions.
    """

    return_value = (
        f"{INITIALIZER_POLICY_VERSION}|"
        f"{INITIALIZER_PROMPT_VERSION}|"
        f"{INITIALIZER_AGENT_REGISTRY_VERSION}|"
        f"{INITIALIZER_STRATEGY_SCHEMA_VERSION}"
    )
    return return_value


async def purge_stale_initializer_entries() -> int:
    """Delete persisted initializer rows that do not match current versions.

    Returns:
        Number of rows deleted. Returns ``0`` if MongoDB is unavailable.
    """

    try:
        collection = await _get_collection()
    except PyMongoError as exc:
        logger.exception(f"Could not get persistent Cache2 collection for purge: {exc}")
        return 0

    delete_filter = {
        "cache_name": INITIALIZER_CACHE_NAME,
        "version_key": {"$ne": build_initializer_version_key()},
    }
    try:
        result = await collection.delete_many(delete_filter)
    except PyMongoError as exc:
        logger.exception(f"Could not purge stale initializer cache entries: {exc}")
        return 0

    deleted_count = int(result.deleted_count)
    if deleted_count:
        logger.info(f"Purged {deleted_count} stale persistent initializer cache entries")
    return deleted_count


async def load_initializer_entries(
    *,
    limit: int = RAG_CACHE2_MAX_ENTRIES,
) -> list[RAGCache2PersistentEntryDoc]:
    """Load current-version persistent initializer entries for hydration.

    Args:
        limit: Maximum number of rows to load, normally the in-memory LRU cap.

    Returns:
        Current-version rows ordered by ``hit_count`` descending, then
        ``updated_at`` descending. Returns an empty list on MongoDB failure.
    """

    try:
        collection = await _get_collection()
    except PyMongoError as exc:
        logger.exception(f"Could not get persistent Cache2 collection for hydration: {exc}")
        return []

    query = {
        "cache_name": INITIALIZER_CACHE_NAME,
        "version_key": build_initializer_version_key(),
    }
    try:
        cursor = (
            collection
            .find(query)
            .sort([("hit_count", DESCENDING), ("updated_at", DESCENDING)])
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
    except PyMongoError as exc:
        logger.exception(f"Could not load persistent initializer cache entries: {exc}")
        return []

    return_value = [dict(doc) for doc in docs]
    return return_value


async def upsert_initializer_entry(
    *,
    cache_key: str,
    result: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Persist one cacheable initializer strategy entry.

    Args:
        cache_key: Stable Cache2 key for the initializer strategy.
        result: Cache payload, normally ``unknown_slots`` and ``confidence``.
        metadata: Operational metadata already used by the in-memory store.

    Returns:
        None. MongoDB failures are logged and swallowed.
    """

    try:
        collection = await _get_collection()
    except PyMongoError as exc:
        logger.exception(f"Could not get persistent Cache2 collection for key={cache_key}: {exc}")
        return

    write_time = _now_iso()
    update = {
        "$set": {
            "cache_name": INITIALIZER_CACHE_NAME,
            "version_key": build_initializer_version_key(),
            "result": dict(result),
            "metadata": dict(metadata),
            "updated_at": write_time,
        },
        "$setOnInsert": {
            "created_at": write_time,
        },
        "$inc": {
            "hit_count": 0,
        },
    }
    try:
        await collection.update_one({"_id": cache_key}, update, upsert=True)
    except PyMongoError as exc:
        logger.exception(f"Could not upsert persistent initializer key={cache_key}: {exc}")


async def record_initializer_hit(cache_key: str) -> None:
    """Record a persistent hit counter update for one initializer cache key.

    Args:
        cache_key: Stable Cache2 key that was served from memory.

    Returns:
        None. Missing rows and MongoDB failures do not affect the caller.
    """

    try:
        collection = await _get_collection()
    except PyMongoError as exc:
        logger.exception(f"Could not get persistent Cache2 collection for key={cache_key}: {exc}")
        return

    update = {
        "$inc": {"hit_count": 1},
        "$set": {"updated_at": _now_iso()},
    }
    try:
        await collection.update_one({"_id": cache_key}, update, upsert=False)
    except PyMongoError as exc:
        logger.exception(f"Could not record persistent initializer hit key={cache_key}: {exc}")


async def prune_persistent_entries(
    *,
    cache_name: str,
    max_entries: int,
) -> int:
    """Prune low-value persistent rows for one cache namespace.

    Args:
        cache_name: Cache namespace to prune.
        max_entries: Maximum rows to retain for that namespace.

    Returns:
        Number of rows deleted. Returns ``0`` on MongoDB failure.
    """

    try:
        collection = await _get_collection()
    except PyMongoError as exc:
        logger.exception(f"Could not get persistent Cache2 collection for cache_name={cache_name}: {exc}")
        return 0

    query = {"cache_name": cache_name}
    try:
        row_count = await collection.count_documents(query)
    except PyMongoError as exc:
        logger.exception(f"Could not count persistent Cache2 rows for cache_name={cache_name}: {exc}")
        return 0

    excess_count = row_count - max_entries
    if excess_count <= 0:
        return 0

    try:
        cursor = (
            collection
            .find(query, {"_id": 1})
            .sort([("hit_count", ASCENDING), ("updated_at", ASCENDING)])
            .limit(excess_count)
        )
        docs_to_delete = await cursor.to_list(length=excess_count)
    except PyMongoError as exc:
        logger.exception(f"Could not select persistent Cache2 prune rows for cache_name={cache_name}: {exc}")
        return 0

    ids_to_delete = [doc["_id"] for doc in docs_to_delete if "_id" in doc]
    if not ids_to_delete:
        return 0

    delete_filter = {
        "cache_name": cache_name,
        "_id": {"$in": ids_to_delete},
    }
    try:
        delete_result = await collection.delete_many(delete_filter)
    except PyMongoError as exc:
        logger.exception(f"Could not prune persistent Cache2 rows for cache_name={cache_name}: {exc}")
        return 0

    deleted_count = int(delete_result.deleted_count)
    if deleted_count:
        logger.info(f"Pruned {deleted_count} persistent Cache2 rows for cache_name={cache_name}")
    return deleted_count
