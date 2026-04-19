"""Operations against the ``rag_cache_index`` and ``rag_metadata_index`` collections.

These two collections back the in-memory ``RAGCache``:

* ``rag_cache_index`` — write-through persistence so the cache can warm-start
  after a crash. TTL-indexed on ``ttl_expires_at`` for automatic cleanup.
* ``rag_metadata_index`` — one document per ``global_user_id`` carrying
  ``rag_version`` (a monotonically increasing cache-bust signal).

The ``RAGCache`` class talks to ``rag_cache_index`` directly today; once it's
ready, those calls should be replaced with the helpers below so all DB shape
knowledge lives in one place.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from kazusa_ai_chatbot.db._client import get_db

logger = logging.getLogger(__name__)


_CACHE_COLLECTION = "rag_cache_index"
_METADATA_COLLECTION = "rag_metadata_index"


def _now_utc() -> datetime:
    """Current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


# ── rag_cache_index ────────────────────────────────────────────────


async def insert_cache_entry(
    *,
    cache_type: str,
    global_user_id: str,
    embedding: list[float],
    results: dict,
    ttl_seconds: int,
    metadata: dict | None = None,
) -> str:
    """Insert a new cache entry and return its ``cache_id``.

    Args:
        cache_type: Cache namespace (see ``rag.cache.DEFAULT_TTL_SECONDS``).
        global_user_id: Owner UUID, or empty string ``""`` for global entries.
        embedding: Query vector that produced ``results``.
        results: RAG output payload to cache.
        ttl_seconds: Lifetime in seconds; the TTL index drops the doc after this.
        metadata: Optional auxiliary data.

    Returns:
        The newly generated UUID4 ``cache_id``.
    """
    db = await get_db()
    cache_id = str(uuid.uuid4())
    now = _now_utc()
    await db[_CACHE_COLLECTION].insert_one({
        "cache_id": cache_id,
        "cache_type": cache_type,
        "global_user_id": global_user_id,
        "embedding": embedding,
        "results": results,
        "ttl_expires_at": now + timedelta(seconds=ttl_seconds),
        "created_at": now.isoformat(),
        "deleted": False,
        "metadata": metadata or {},
    })
    return cache_id


async def find_cache_entries(
    cache_type: str,
    global_user_id: str | None = None,
) -> list[dict]:
    """Return non-expired, non-deleted cache documents matching the filter.

    Args:
        cache_type: Cache namespace to look up.
        global_user_id: When provided, restricts to that owner. Pass ``""`` for
            global entries; pass ``None`` to match any owner.

    Returns:
        A list of cache documents (with ``embedding`` and ``results`` intact).
    """
    db = await get_db()
    query: dict[str, Any] = {
        "cache_type": cache_type,
        "deleted": {"$ne": True},
        "ttl_expires_at": {"$gt": _now_utc()},
    }
    if global_user_id is not None:
        query["global_user_id"] = global_user_id
    cursor = db[_CACHE_COLLECTION].find(query)
    return await cursor.to_list(length=None)


async def soft_delete_cache_entries(
    cache_type: str,
    global_user_id: str,
) -> int:
    """Mark all entries matching ``(cache_type, global_user_id)`` as deleted.

    Returns:
        Number of documents updated.
    """
    db = await get_db()
    result = await db[_CACHE_COLLECTION].update_many(
        {"cache_type": cache_type, "global_user_id": global_user_id, "deleted": {"$ne": True}},
        {"$set": {"deleted": True}},
    )
    return result.modified_count


async def clear_all_cache_for_user(global_user_id: str) -> int:
    """Soft-delete every cache entry owned by ``global_user_id`` regardless of cache_type.

    Returns:
        Number of documents updated.
    """
    db = await get_db()
    result = await db[_CACHE_COLLECTION].update_many(
        {"global_user_id": global_user_id, "deleted": {"$ne": True}},
        {"$set": {"deleted": True}},
    )
    return result.modified_count


# ── rag_metadata_index ─────────────────────────────────────────────


async def get_rag_version(global_user_id: str) -> int:
    """Return the current ``rag_version`` for the user (0 if not present)."""
    db = await get_db()
    doc = await db[_METADATA_COLLECTION].find_one({"global_user_id": global_user_id})
    if doc is None:
        return 0
    return int(doc.get("rag_version", 0))


async def increment_rag_version(global_user_id: str) -> int:
    """Atomically bump ``rag_version`` for the user and return the new value.

    Upserts the document if it doesn't exist yet. Also stamps ``last_rag_run``.
    """
    db = await get_db()
    result = await db[_METADATA_COLLECTION].find_one_and_update(
        {"global_user_id": global_user_id},
        {
            "$inc": {"rag_version": 1},
            "$set": {"last_rag_run": _now_utc().isoformat()},
            "$setOnInsert": {"global_user_id": global_user_id},
        },
        upsert=True,
        return_document=True,
    )
    return int(result["rag_version"])
