"""Semantic RAG cache with crash-resilient MongoDB persistence.

The cache keeps its hot path in-memory (LRU of (embedding, results) pairs)
and asynchronously writes every store to the ``rag_cache_index`` MongoDB
collection.  On startup, non-expired entries are reloaded back into memory
so the cache warm-starts after a crash/restart.

Until Stage 2's db restructure, this module talks to MongoDB directly.
Once ``db/rag_cache.py`` exists, the in-module DB helpers should be
replaced with imports from that module — the public API of ``RAGCache``
does not need to change.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db import get_db

logger = logging.getLogger(__name__)


# ── Collection name ────────────────────────────────────────────────
_CACHE_COLLECTION = "rag_cache_index"


# ── Defaults ───────────────────────────────────────────────────────
DEFAULT_SIMILARITY_THRESHOLD = 0.82
DEFAULT_MAX_SIZE = 10_000
DEFAULT_TTL_SECONDS = {
    "user_facts": 1800,        # 30 minutes
    "internal_memory": 900,    # 15 minutes
}


# ── Helpers ────────────────────────────────────────────────────────


def _cosine_similarity(
    a: list[float],
    b: list[float],
    *,
    a_norm: float | None = None,
    b_norm: float | None = None,
) -> float:
    """Cosine similarity between two equal-length vectors.

    Args:
        a: First vector.
        b: Second vector. Must be the same length as ``a``.
        a_norm: Pre-computed L2 norm of ``a``. Computed internally if omitted.
        b_norm: Pre-computed L2 norm of ``b``. Computed internally if omitted.

    Returns:
        Similarity in [0, 1], or 0.0 if either vector is empty, mismatched, or zero.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    for x, y in zip(a, b):
        dot += x * y
    if a_norm is None:
        a_norm = math.sqrt(sum(x * x for x in a))
    if b_norm is None:
        b_norm = math.sqrt(sum(y * y for y in b))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return dot / (a_norm * b_norm)


def _vec_norm(v: list[float]) -> float:
    """L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _now_utc() -> datetime:
    """Current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


# ── Cache entry ────────────────────────────────────────────────────


class _CacheEntry:
    """Single in-memory cache record, including its pre-computed embedding norm."""

    __slots__ = (
        "cache_id",
        "cache_type",
        "global_user_id",
        "embedding",
        "embedding_norm",
        "results",
        "ttl_expires_at",
        "created_at",
        "metadata",
    )

    def __init__(
        self,
        *,
        cache_id: str,
        cache_type: str,
        global_user_id: str,
        embedding: list[float],
        results: dict,
        ttl_expires_at: datetime,
        created_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Args:
            cache_id: Unique UUID4 identifier for this entry.
            cache_type: Namespace key, e.g. ``"user_facts"`` or ``"internal_memory"``.
            global_user_id: Internal UUID of the user who owns this entry.
            embedding: Query vector that produced ``results``.
            results: Cached RAG output payload.
            ttl_expires_at: Absolute expiry time; entry is ignored after this.
            created_at: When the entry was created. Defaults to now if omitted.
            metadata: Optional auxiliary data attached to the entry.
        """
        self.cache_id = cache_id
        self.cache_type = cache_type
        self.global_user_id = global_user_id
        self.embedding = embedding
        self.embedding_norm = _vec_norm(embedding)
        self.results = results
        self.ttl_expires_at = ttl_expires_at
        self.created_at = created_at or _now_utc()
        self.metadata = metadata or {}

    def is_expired(self, now: datetime | None = None) -> bool:
        """Return True if the entry's TTL has passed.

        Args:
            now: Reference time. Defaults to current UTC time if omitted.

        Returns:
            True when the entry should no longer be served.
        """
        now = now or _now_utc()
        return now >= self.ttl_expires_at


# ── RAGCache ───────────────────────────────────────────────────────


class RAGCache:
    """In-memory LRU cache with MongoDB write-through for crash resilience."""

    def __init__(
        self,
        *,
        max_size: int = DEFAULT_MAX_SIZE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        default_ttl_seconds: dict[str, int] | None = None,
    ) -> None:
        """
        Args:
            max_size: Maximum number of entries kept in memory. LRU entries are
                evicted when the store exceeds this limit.
            similarity_threshold: Minimum cosine similarity for a retrieval hit.
                Callers can override per-query via ``retrieve_if_similar``.
            default_ttl_seconds: Mapping of cache_type → TTL in seconds.
                Entries not in this map fall back to 600 seconds. Defaults to
                ``DEFAULT_TTL_SECONDS`` if omitted.
        """
        self._max_size = max_size
        self._threshold = similarity_threshold
        self._ttl = dict(default_ttl_seconds or DEFAULT_TTL_SECONDS)

        # LRU store: key = cache_id, value = _CacheEntry
        self._store: "OrderedDict[str, _CacheEntry]" = OrderedDict()

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        self._started = False

    # ── lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Warm-start from MongoDB.  Safe to call multiple times."""
        if self._started:
            return
        try:
            loaded = await self._load_from_db()
        except PyMongoError:
            logger.exception("RAGCache warm-start failed — continuing with empty in-memory store")
            self._started = True
            return
        logger.info("RAGCache warm-started with %d entries from MongoDB", loaded)
        self._started = True

    async def shutdown(self) -> None:
        """Log final stats and mark the cache as stopped.

        Every ``store()`` call is write-through, so there is nothing to flush.
        This method exists for lifecycle symmetry and future-proofing.
        """
        logger.info(
            "RAGCache shutdown — hits=%d, misses=%d, evictions=%d, size=%d",
            self._hits, self._misses, self._evictions, len(self._store),
        )
        self._started = False

    # ── public API ─────────────────────────────────────────────

    async def retrieve_if_similar(
        self,
        *,
        embedding: list[float],
        cache_type: str,
        global_user_id: str | None = None,
        threshold: float | None = None,
    ) -> dict | None:
        """Return the cached result whose stored embedding best matches the query.

        Scans in-memory entries of the given ``cache_type``. If ``global_user_id``
        is provided, only entries owned by that user are considered. Expired entries
        are lazily removed during the scan.

        Args:
            embedding: Query vector to compare against stored entries.
            cache_type: Namespace key, e.g. ``"user_facts"`` or ``"internal_memory"``.
            global_user_id: When given, restricts the search to a single user's entries.
            threshold: Minimum cosine similarity to count as a hit. Defaults to the
                value set at construction time.

        Returns:
            A dict with keys ``cache_id``, ``similarity``, ``results``, and
            ``metadata`` if a match is found above threshold; ``None`` on a miss.
        """
        threshold = threshold if threshold is not None else self._threshold
        if not embedding:
            self._misses += 1
            return None

        q_norm = _vec_norm(embedding)
        if q_norm == 0.0:
            self._misses += 1
            return None

        best: tuple[float, _CacheEntry] | None = None
        expired: list[str] = []
        now = _now_utc()

        for cid, entry in self._store.items():
            if entry.cache_type != cache_type:
                continue
            if global_user_id is not None and entry.global_user_id != global_user_id:
                continue
            if entry.is_expired(now):
                expired.append(cid)
                continue
            sim = _cosine_similarity(
                embedding, entry.embedding,
                a_norm=q_norm, b_norm=entry.embedding_norm,
            )
            if best is None or sim > best[0]:
                best = (sim, entry)

        for cid in expired:
            self._store.pop(cid, None)

        if best is not None and best[0] >= threshold:
            self._store.move_to_end(best[1].cache_id)
            self._hits += 1
            return {
                "cache_id": best[1].cache_id,
                "similarity": best[0],
                "results": best[1].results,
                "metadata": best[1].metadata,
            }

        self._misses += 1
        return None

    async def store(
        self,
        *,
        embedding: list[float],
        results: dict,
        cache_type: str,
        global_user_id: str,
        ttl_seconds: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store a new cache entry and persist it asynchronously to MongoDB.

        The entry is added to the in-memory store immediately. A MongoDB write
        is attempted afterwards; if that write fails, the in-memory entry is
        still valid and the error is logged without propagating.

        Args:
            embedding: Query vector that produced ``results``.
            results: RAG output payload to cache.
            cache_type: Namespace key, e.g. ``"user_facts"`` or ``"internal_memory"``.
            global_user_id: Internal UUID of the user who owns this entry.
            ttl_seconds: How long the entry is valid. Defaults to the per-type TTL
                configured at construction time, or 600 seconds if the type is unknown.
            metadata: Optional auxiliary data to attach to the entry.

        Returns:
            The newly assigned ``cache_id`` (UUID4 string).
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl.get(cache_type, 600)
        expires_at = _now_utc() + timedelta(seconds=ttl)
        entry = _CacheEntry(
            cache_id=str(uuid.uuid4()),
            cache_type=cache_type,
            global_user_id=global_user_id,
            embedding=embedding,
            results=results,
            ttl_expires_at=expires_at,
            metadata=metadata or {},
        )
        self._store[entry.cache_id] = entry
        self._store.move_to_end(entry.cache_id)
        self._evict_if_needed()

        try:
            await self._persist(entry)
        except PyMongoError:
            logger.exception("Failed to persist cache entry %s — in-memory still valid", entry.cache_id)

        return entry.cache_id

    async def invalidate_pattern(
        self,
        *,
        cache_type: str,
        global_user_id: str,
    ) -> int:
        """Remove all in-memory entries matching (cache_type, global_user_id) and soft-delete in MongoDB.

        Args:
            cache_type: Namespace key identifying which category of entries to remove.
            global_user_id: Internal UUID whose entries should be invalidated.

        Returns:
            Number of in-memory entries removed. MongoDB soft-delete failures are
            logged but do not affect this count.
        """
        removed: list[str] = []
        for cid, entry in list(self._store.items()):
            if entry.cache_type == cache_type and entry.global_user_id == global_user_id:
                removed.append(cid)
        for cid in removed:
            self._store.pop(cid, None)

        try:
            await self._soft_delete(cache_type=cache_type, global_user_id=global_user_id)
        except PyMongoError:
            logger.exception("Failed to soft-delete cache for %s/%s", cache_type, global_user_id)

        return len(removed)

    async def clear_all_user(self, global_user_id: str) -> int:
        """Remove every in-memory entry for the user and soft-delete all in MongoDB.

        Args:
            global_user_id: Internal UUID whose entire cache should be cleared,
                regardless of cache_type.

        Returns:
            Number of in-memory entries removed.
        """
        removed: list[str] = []
        for cid, entry in list(self._store.items()):
            if entry.global_user_id == global_user_id:
                removed.append(cid)
        for cid in removed:
            self._store.pop(cid, None)

        try:
            await self._soft_delete(cache_type=None, global_user_id=global_user_id)
        except PyMongoError:
            logger.exception("Failed to soft-delete all cache for user %s", global_user_id)

        return len(removed)

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of cache performance counters.

        Returns:
            Dict with keys: ``hits``, ``misses``, ``hit_rate``, ``evictions``,
            ``size``, ``max_size``, ``threshold``.
        """
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "evictions": self._evictions,
            "size": len(self._store),
            "max_size": self._max_size,
            "threshold": self._threshold,
        }

    # ── internal: LRU eviction ─────────────────────────────────

    def _evict_if_needed(self) -> None:
        """Drop the least-recently-used entry until the store is within max_size."""
        while len(self._store) > self._max_size:
            cid, _ = self._store.popitem(last=False)
            self._evictions += 1
            logger.debug("RAGCache evicted %s", cid)

    # ── internal: MongoDB persistence ──────────────────────────

    async def _persist(self, entry: _CacheEntry) -> None:
        """Insert a single cache entry document into MongoDB.

        Args:
            entry: The in-memory entry to persist. Caller handles PyMongoError.
        """
        db = await get_db()
        await db[_CACHE_COLLECTION].insert_one({
            "cache_id": entry.cache_id,
            "cache_type": entry.cache_type,
            "global_user_id": entry.global_user_id,
            "embedding": entry.embedding,
            "results": entry.results,
            "ttl_expires_at": entry.ttl_expires_at,
            "created_at": entry.created_at.isoformat(),
            "deleted": False,
            "metadata": entry.metadata,
        })

    async def _soft_delete(self, *, cache_type: str | None, global_user_id: str) -> None:
        """Mark matching cache documents as deleted in MongoDB without removing them.

        Args:
            cache_type: Which category to target. When ``None``, all types for
                ``global_user_id`` are deleted.
            global_user_id: Owner whose entries should be soft-deleted.
        """
        db = await get_db()
        query: dict[str, Any] = {"global_user_id": global_user_id, "deleted": False}
        if cache_type is not None:
            query["cache_type"] = cache_type
        await db[_CACHE_COLLECTION].update_many(
            query,
            {"$set": {"deleted": True}},
        )

    async def _load_from_db(self) -> int:
        """Reload non-expired, non-deleted entries from MongoDB into memory.

        Called once during ``start()``. Stops early if the loaded count reaches
        ``max_size`` to avoid blowing the memory budget on a large persisted store.

        Returns:
            Number of entries loaded into the in-memory store.
        """
        db = await get_db()
        now = _now_utc()
        cursor = db[_CACHE_COLLECTION].find({
            "deleted": {"$ne": True},
            "ttl_expires_at": {"$gt": now},
        }).sort("created_at", 1)
        loaded = 0
        async for doc in cursor:
            ttl_expires_at = doc.get("ttl_expires_at")
            if isinstance(ttl_expires_at, str):
                try:
                    ttl_expires_at = datetime.fromisoformat(ttl_expires_at)
                except ValueError:
                    continue
            created_at_raw = doc.get("created_at")
            created_at = None
            if isinstance(created_at_raw, str):
                try:
                    created_at = datetime.fromisoformat(created_at_raw)
                except ValueError:
                    created_at = None
            elif isinstance(created_at_raw, datetime):
                created_at = created_at_raw

            entry = _CacheEntry(
                cache_id=doc["cache_id"],
                cache_type=doc["cache_type"],
                global_user_id=doc["global_user_id"],
                embedding=doc["embedding"],
                results=doc["results"],
                ttl_expires_at=ttl_expires_at,
                created_at=created_at,
                metadata=doc.get("metadata", {}),
            )
            self._store[entry.cache_id] = entry
            loaded += 1
            if loaded >= self._max_size:
                break
        return loaded


# ── Standalone test harness ────────────────────────────────────────


async def test_main() -> None:
    """Round-trip smoke test: store → similar-retrieve → invalidate.

    Requires MongoDB to be running for the write-through persistence path.
    The in-memory round-trip works without MongoDB.
    """
    import json

    logging.basicConfig(level=logging.INFO)

    cache = RAGCache(max_size=100, similarity_threshold=0.80)

    emb_a = [0.10, 0.20, 0.30, 0.40, 0.50]
    emb_a_similar = [0.11, 0.21, 0.29, 0.41, 0.49]
    emb_unrelated = [0.90, -0.40, 0.10, 0.05, -0.20]

    cid = await cache.store(
        embedding=emb_a,
        results={"facts": ["user likes sushi", "user lives in Auckland"]},
        cache_type="user_facts",
        global_user_id="user-test-001",
        ttl_seconds=60,
        metadata={"origin": "test_main"},
    )
    print(f"[store] cache_id = {cid}")

    hit = await cache.retrieve_if_similar(
        embedding=emb_a_similar,
        cache_type="user_facts",
        global_user_id="user-test-001",
    )
    print(f"[retrieve_if_similar] similar={hit is not None}")
    print(json.dumps(hit, indent=2, default=str))

    miss = await cache.retrieve_if_similar(
        embedding=emb_unrelated,
        cache_type="user_facts",
        global_user_id="user-test-001",
    )
    print(f"[retrieve_if_similar] unrelated hit={miss is not None}")

    removed = await cache.invalidate_pattern(
        cache_type="user_facts",
        global_user_id="user-test-001",
    )
    print(f"[invalidate_pattern] removed={removed}")

    miss_after_invalidate = await cache.retrieve_if_similar(
        embedding=emb_a_similar,
        cache_type="user_facts",
        global_user_id="user-test-001",
    )
    print(f"[retrieve_if_similar after invalidate] hit={miss_after_invalidate is not None}")

    print(f"[get_stats] {cache.get_stats()}")


if __name__ == "__main__":
    asyncio.run(test_main())
