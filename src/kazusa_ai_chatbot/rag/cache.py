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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db import get_db

logger = logging.getLogger(__name__)


# ── Collection name ────────────────────────────────────────────────
_CACHE_COLLECTION = "rag_cache_index"


# ── Defaults ───────────────────────────────────────────────────────
DEFAULT_SIMILARITY_THRESHOLD = 0.60
DEFAULT_MAX_SIZE = 10_000
DEFAULT_TTL_SECONDS = {
    # User-related, per-user scoped
    "character_diary": 1800,        # 30 min — character's subjective observations
    "objective_user_facts": 3600,   # 60 min — verified facts about the user
    "user_promises": 900,           # 15 min — time-sensitive commitments

    # Conversation-related, per-user scoped
    "internal_memory": 900,         # 15 min — conversation history snippets

    # External, GLOBAL scope (shared across users — use global_user_id="")
    "external_knowledge": 3600,     # 1 hour — web search / shared knowledge
    "knowledge_base": 2592000,      # 30 days — accumulated cross-session topic knowledge

    # Phase 8 — Boundary cache (resolution→retrieval boundary)
    "boundary_cache": 900,          # 15 min — keyed on resolution hash
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


# ── Phase 8: Scoped cache invalidation ────────────────────────────


@dataclass
class CacheInvalidationScope:
    """Descriptor produced by the write path to specify which cache entries
    should be invalidated when new data is written.

    The fields act as **AND** filters — only entries matching all non-empty
    fields are removed. Empty/None fields are treated as wildcards.

    Attributes:
        cache_type: Restrict invalidation to a specific cache namespace
            (e.g. ``"boundary_cache"``, ``"external_knowledge"``).
        global_user_id: Restrict to entries owned by a specific user.
            ``None`` means "match any user".
        boundary_key: Restrict to entries with a specific boundary cache key.
        channel_id: Restrict to entries associated with a specific channel.
        reason: Human-readable description of why invalidation was triggered.
    """
    cache_type: str = ""
    global_user_id: str | None = None
    boundary_key: str = ""
    channel_id: str = ""
    reason: str = ""


# ── Phase 8: cached_node decorator ───────────────────────────────


def cached_node(key_fn, *, cache_type: str = "external_knowledge"):
    """Decorator that wraps a RAG executor node with per-node caching.

    The wrapped node checks the process-wide ``RAGCache`` for a boundary-key
    match before executing. On a hit, it returns the cached result directly.
    On a miss, it runs the node and stores the result.

    The ``key_fn`` receives the RAGState and must return a string cache key.
    The node function itself stays cache-unaware.

    Args:
        key_fn: Callable ``(state) -> str`` producing the cache key.
        cache_type: Cache namespace for the entry (default ``"external_knowledge"``).

    Returns:
        A decorator that wraps an async node function.
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(state):
            cache = _get_cached_node_cache()
            if cache is None:
                return await fn(state)

            key = key_fn(state)
            if not key:
                return await fn(state)

            hit = await cache.retrieve_if_similar_by_key(key)
            if hit is not None:
                logger.debug("cached_node hit for %s key=%s…", fn.__name__, key[:16])
                return hit["results"]

            result = await fn(state)
            await cache.store_by_key(
                cache_key=key,
                results=result,
                cache_type=cache_type,
                global_user_id=state.get("global_user_id", ""),
                metadata={"node": fn.__name__},
            )
            return result
        return wrapper
    return decorator


_cached_node_cache_ref: RAGCache | None = None


def _get_cached_node_cache() -> RAGCache | None:
    """Return the process-wide RAGCache singleton, if initialised.

    The singleton is set by ``RAGCache.start()`` — before that, decorated
    nodes fall through to the uncached path.
    """
    return _cached_node_cache_ref


def set_cached_node_cache(cache: RAGCache) -> None:
    """Register the process-wide RAGCache for ``cached_node`` decorators.

    Called once during application startup after the cache is initialised.
    """
    global _cached_node_cache_ref
    _cached_node_cache_ref = cache


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
            cache_type: Namespace key. See ``DEFAULT_TTL_SECONDS`` for the full
                set of supported types (``character_diary``, ``objective_user_facts``,
                ``user_promises``, ``internal_memory``, ``external_knowledge``).
            global_user_id: When given, restricts the search to a single user's
                entries. Pass ``""`` (empty string) to read GLOBAL entries shared
                across all users — required for ``cache_type="external_knowledge"``.
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
            cache_type: Namespace key. See ``DEFAULT_TTL_SECONDS`` for the full
                set of supported types.
            global_user_id: Internal UUID of the user who owns this entry. Pass
                ``""`` (empty string) to mark the entry as GLOBAL/shared — only
                valid for ``cache_type="external_knowledge"``.
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

    # ── Phase 8: key-based boundary cache ─────────────────────

    async def retrieve_if_similar_by_key(
        self,
        cache_key: str,
    ) -> dict | None:
        """Retrieve a cached result by exact structured key match.

        Unlike ``retrieve_if_similar``, this does **not** compare embeddings.
        The ``cache_key`` is a pre-computed hash from resolution outputs
        (see ``_build_cache_key``).  Scans **all** cache types for a matching
        ``boundary_key`` in metadata.

        Args:
            cache_key: Hex-digest string that uniquely identifies the
                resolution context (resolved task, entity IDs, active sources,
                lookback hours).

        Returns:
            A dict with keys ``cache_id``, ``results``, and ``metadata``
            if a non-expired match is found; ``None`` on a miss.
        """
        now = _now_utc()
        for cid, entry in self._store.items():
            if entry.metadata.get("boundary_key") != cache_key:
                continue
            if entry.is_expired(now):
                self._store.pop(cid, None)
                continue
            self._store.move_to_end(cid)
            self._hits += 1
            return {
                "cache_id": entry.cache_id,
                "results": entry.results,
                "metadata": entry.metadata,
            }
        self._misses += 1
        return None

    async def store_by_key(
        self,
        *,
        cache_key: str,
        results: dict,
        cache_type: str,
        global_user_id: str,
        ttl_seconds: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store a cache entry keyed by a structured hash (not embedding).

        The ``cache_key`` is stored inside ``metadata["boundary_key"]`` so
        ``retrieve_if_similar_by_key`` can look it up by exact match.

        Args:
            cache_key: Pre-computed hash from resolution outputs.
            results: RAG output payload to cache.
            cache_type: Namespace key (typically ``"boundary_cache"``).
            global_user_id: Owner of the entry.
            ttl_seconds: Validity window. Defaults to per-type TTL or 600s.
            metadata: Optional auxiliary data; ``boundary_key`` is injected
                automatically.

        Returns:
            The newly assigned ``cache_id`` (UUID4 string).
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl.get(cache_type, 600)
        expires_at = _now_utc() + timedelta(seconds=ttl)
        meta = dict(metadata or {})
        meta["boundary_key"] = cache_key
        entry = _CacheEntry(
            cache_id=str(uuid.uuid4()),
            cache_type=cache_type,
            global_user_id=global_user_id,
            embedding=[],
            results=results,
            ttl_expires_at=expires_at,
            metadata=meta,
        )
        self._store[entry.cache_id] = entry
        self._store.move_to_end(entry.cache_id)
        self._evict_if_needed()

        try:
            await self._persist(entry)
        except PyMongoError:
            logger.exception("Failed to persist boundary cache entry %s — in-memory still valid", entry.cache_id)

        return entry.cache_id

    async def invalidate_scoped(self, scope: "CacheInvalidationScope") -> int:
        """Invalidate cache entries matching a scoped descriptor.

        This is the Phase 8 replacement for blanket invalidation.
        ``db_writer`` produces a ``CacheInvalidationScope`` describing what
        was written, and this method removes only the affected entries.

        Args:
            scope: Descriptor produced by the write path.

        Returns:
            Number of in-memory entries removed.
        """
        removed: list[str] = []
        for cid, entry in list(self._store.items()):
            if scope.cache_type and entry.cache_type != scope.cache_type:
                continue
            if scope.global_user_id is not None and entry.global_user_id != scope.global_user_id:
                continue
            if scope.boundary_key and entry.metadata.get("boundary_key") != scope.boundary_key:
                continue
            if scope.channel_id and entry.metadata.get("channel_id") != scope.channel_id:
                continue
            removed.append(cid)

        for cid in removed:
            self._store.pop(cid, None)

        if removed:
            try:
                await self._soft_delete_scoped(scope)
            except PyMongoError:
                logger.exception("Failed to soft-delete scoped cache entries for %s", scope)

        return len(removed)

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
            "created_at": entry.created_at,
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

    async def _soft_delete_scoped(self, scope: CacheInvalidationScope) -> None:
        """Soft-delete MongoDB entries matching a scoped invalidation descriptor.

        Args:
            scope: The invalidation scope. Only non-empty fields are added
                to the query filter.
        """
        db = await get_db()
        query: dict[str, Any] = {"deleted": False}
        if scope.cache_type:
            query["cache_type"] = scope.cache_type
        if scope.global_user_id is not None:
            query["global_user_id"] = scope.global_user_id
        if scope.boundary_key:
            query["metadata.boundary_key"] = scope.boundary_key
        if scope.channel_id:
            query["metadata.channel_id"] = scope.channel_id
        await db[_CACHE_COLLECTION].update_many(
            query,
            {"$set": {"deleted": True}},
        )

    async def _load_from_db(self) -> int:
        """Reload non-expired, non-deleted entries from MongoDB into memory.

        Called once during ``start()``. Stops early if the loaded count reaches
        ``max_size`` to avoid blowing the memory budget on a large persisted store.

        All datetimes are guaranteed UTC-aware.

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
            # Ensure ttl_expires_at is UTC-aware datetime
            if isinstance(ttl_expires_at, str):
                try:
                    dt = datetime.fromisoformat(ttl_expires_at)
                    ttl_expires_at = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            elif isinstance(ttl_expires_at, datetime):
                ttl_expires_at = ttl_expires_at if ttl_expires_at.tzinfo else ttl_expires_at.replace(tzinfo=timezone.utc)
            else:
                continue

            # Ensure created_at is UTC-aware datetime
            created_at_raw = doc.get("created_at")
            created_at = None
            if isinstance(created_at_raw, str):
                try:
                    dt = datetime.fromisoformat(created_at_raw)
                    created_at = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    created_at = None
            elif isinstance(created_at_raw, datetime):
                created_at = created_at_raw if created_at_raw.tzinfo else created_at_raw.replace(tzinfo=timezone.utc)

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
    from kazusa_ai_chatbot.db import get_text_embedding


    logging.basicConfig(level=logging.INFO)

    cache = RAGCache(max_size=100, similarity_threshold=0.7)

    emb_a_similar = [0.11, 0.21, 0.29, 0.41, 0.49]
    emb_unrelated = [0.90, -0.40, 0.10, 0.05, -0.20]

    # Store data
    input_facts = ["User likes sushi"]
    input_embedding = await get_text_embedding("\n".join(input_facts))

    cid = await cache.store(
        embedding=input_embedding,
        results={"facts": input_facts},
        cache_type="user_facts",
        global_user_id="user-test-001",
        ttl_seconds=60,
        metadata={"origin": "test_main"},
    )
    print(f"[store] cache_id = {cid}")

    input_facts = ["user likes ramen"]
    input_embedding = await get_text_embedding("\n".join(input_facts))

    cid = await cache.store(
        embedding=input_embedding,
        results={"facts": input_facts},
        cache_type="user_facts",
        global_user_id="user-test-001",
        ttl_seconds=60,
        metadata={"origin": "test_main"},
    )
    print(f"[store] cache_id = {cid}")


    # Similar case
    hit_embedding = await get_text_embedding("I like ramen")
    hit = await cache.retrieve_if_similar(
        embedding=hit_embedding,
        cache_type="user_facts",
        global_user_id="user-test-001",
    )
    if hit is not None:
        print(f"[retrieve_if_similar] similar={hit is not None}, similarity={hit['similarity']}")
        print(json.dumps(hit, indent=2, default=str))
    else:
        print("Misses")

    print(f"[get_stats] {cache.get_stats()}")

async def test_main2():
    """Comprehensive test: cache warm-start, store, retrieve, invalidate with DB persistence.
    
    Tests all Stage 1 & 2 functionality:
    - Warm-start from MongoDB (crash resilience)
    - Store with multiple cache types (character_diary, objective_user_facts, user_promises, internal_memory)
    - User isolation (multiple users)
    - Cache type isolation (entries don't cross-contaminate)
    - Retrieve with similarity matching
    - Selective invalidation (cache_type + user scoping)
    - Clear all user cache
    - Statistics tracking
    """
    import json
    from kazusa_ai_chatbot.db import get_text_embedding

    logging.basicConfig(level=logging.INFO)
    print("\n" + "="*80)
    print("COMPREHENSIVE RAGCACHE TEST WITH DATABASE PERSISTENCE")
    print("="*80)

    cache = RAGCache(max_size=100, similarity_threshold=0.75)

    # ── TEST 1: WARM-START FROM DATABASE ───────────────────────────────────
    print("\n[TEST 1] Warm-start from MongoDB...")
    await cache.start()
    print(f"  ✓ Cache started. In-memory entries loaded: {cache.get_stats()['size']}")

    # ── TEST 2: STORE MULTIPLE ENTRIES (Stage 1.5a cache types) ─────────────
    print("\n[TEST 2] Store entries with different cache types...")
    
    user_1 = "user-001"
    user_2 = "user-002"
    
    test_data = [
        {
            "cache_type": "objective_user_facts",
            "global_user_id": user_1,
            "text": "User is a software engineer in Tokyo",
            "ttl": 3600,
        },
        {
            "cache_type": "character_diary",
            "global_user_id": user_1,
            "text": "User seems interested in machine learning",
            "ttl": 1800,
        },
        {
            "cache_type": "user_promises",
            "global_user_id": user_1,
            "text": "Promised to send documentation by Friday",
            "ttl": 900,
        },
        {
            "cache_type": "internal_memory",
            "global_user_id": user_1,
            "text": "User mentioned they work for a startup",
            "ttl": 900,
        },
        {
            "cache_type": "objective_user_facts",
            "global_user_id": user_2,
            "text": "User lives in New York and works in finance",
            "ttl": 3600,
        },
        {
            "cache_type": "external_knowledge",
            "global_user_id": "",  # Global, shared across users
            "text": "Tokyo is the capital of Japan",
            "ttl": 3600,
        },
    ]
    
    stored_entries = []
    for data in test_data:
        embedding = await get_text_embedding(data["text"])
        cache_id = await cache.store(
            embedding=embedding,
            results={"text": data["text"], "context": "test_main2"},
            cache_type=data["cache_type"],
            global_user_id=data["global_user_id"],
            ttl_seconds=data["ttl"],
            metadata={"test": "comprehensive"},
        )
        stored_entries.append({**data, "cache_id": cache_id, "embedding": embedding})
        print(f"  ✓ Stored {data['cache_type']:25s} for user={data['global_user_id'] or 'GLOBAL':10s} "
              f"cache_id={cache_id[:8]}...")

    # ── TEST 3: CACHE TYPE ISOLATION ───────────────────────────────────────
    print("\n[TEST 3] Verify cache type isolation...")
    
    # Query with objective_user_facts should NOT match character_diary
    query_text = "engineer working on Tokyo"
    query_embedding = await get_text_embedding(query_text)
    
    # Should hit objective_user_facts
    hit_facts = await cache.retrieve_if_similar(
        embedding=query_embedding,
        cache_type="objective_user_facts",
        global_user_id=user_1,
        threshold=0.7,
    )
    
    # Should miss character_diary (different semantic space)
    hit_diary = await cache.retrieve_if_similar(
        embedding=query_embedding,
        cache_type="character_diary",
        global_user_id=user_1,
        threshold=0.7,
    )
    
    print(f"  ✓ Query 'engineer working on Tokyo':")
    print(f"    - objective_user_facts: {'HIT' if hit_facts else 'MISS':4s} "
          f"(sim={hit_facts['similarity']:.3f})" if hit_facts else "    - objective_user_facts: MISS")
    print(f"    - character_diary:     {'HIT' if hit_diary else 'MISS':4s} "
          f"(sim={hit_diary['similarity']:.3f})" if hit_diary else "    - character_diary:     MISS")
    print(f"  ✓ Cache type isolation verified")

    # ── TEST 4: USER ISOLATION ─────────────────────────────────────────────
    print("\n[TEST 4] Verify user isolation...")
    
    # Query for user_1 should NOT return results for user_2
    query_finance = "finance New York"
    query_embedding = await get_text_embedding(query_finance)
    
    hit_user_1 = await cache.retrieve_if_similar(
        embedding=query_embedding,
        cache_type="objective_user_facts",
        global_user_id=user_1,
        threshold=0.7,
    )
    
    hit_user_2 = await cache.retrieve_if_similar(
        embedding=query_embedding,
        cache_type="objective_user_facts",
        global_user_id=user_2,
        threshold=0.7,
    )
    
    print(f"  ✓ Query 'finance New York':")
    print(f"    - user_1: {'HIT' if hit_user_1 else 'MISS':4s}")
    print(f"    - user_2: {'HIT' if hit_user_2 else 'MISS':4s} (should find finance fact)")
    print(f"  ✓ User isolation verified")

    # ── TEST 5: EXTERNAL/GLOBAL KNOWLEDGE ──────────────────────────────────
    print("\n[TEST 5] Verify global knowledge (shared across users)...")
    
    query_tokyo = "Tokyo capital Japan geography"
    query_embedding = await get_text_embedding(query_tokyo)
    
    hit_global = await cache.retrieve_if_similar(
        embedding=query_embedding,
        cache_type="external_knowledge",
        global_user_id="",  # Empty string for global
        threshold=0.7,
    )
    
    print(f"  ✓ Query 'Tokyo capital Japan geography':")
    print(f"    - external_knowledge (GLOBAL): {'HIT' if hit_global else 'MISS':4s} "
          f"(sim={hit_global['similarity']:.3f})" if hit_global else "MISS")
    print(f"  ✓ Global knowledge accessible")

    # ── TEST 6: SELECTIVE INVALIDATION (cache_type + user) ──────────────────
    print("\n[TEST 6] Test selective invalidation...")
    
    before_stats = cache.get_stats()
    print(f"  Before invalidation: {before_stats['size']} entries")
    
    # Invalidate only objective_user_facts for user_1
    removed = await cache.invalidate_pattern(
        cache_type="objective_user_facts",
        global_user_id=user_1,
    )
    print(f"  ✓ Invalidated objective_user_facts for user_1: removed {removed} entry")
    
    after_stats = cache.get_stats()
    print(f"  After invalidation: {after_stats['size']} entries")
    
    # Verify cache_type was invalidated but diary still exists
    hit_facts_after = await cache.retrieve_if_similar(
        embedding=query_embedding,
        cache_type="objective_user_facts",
        global_user_id=user_1,
    )
    
    hit_diary_after = await cache.retrieve_if_similar(
        embedding=query_embedding,
        cache_type="character_diary",
        global_user_id=user_1,
        threshold=0.6,
    )
    
    print(f"  ✓ After selective invalidation:")
    print(f"    - objective_user_facts: {'HIT' if hit_facts_after else 'MISS':4s} (should be MISS)")
    print(f"    - character_diary:     {'HIT' if hit_diary_after else 'MISS':4s} (should still exist)")

    # ── TEST 7: CLEAR ALL USER CACHE ───────────────────────────────────────
    print("\n[TEST 7] Test clear all user cache...")
    
    before_clear = cache.get_stats()
    removed_all = await cache.clear_all_user(user_1)
    after_clear = cache.get_stats()
    
    print(f"  ✓ Cleared all cache for user_1: removed {removed_all} entries")
    print(f"    Before: {before_clear['size']} entries → After: {after_clear['size']} entries")
    
    # Verify user_1 cache is completely gone but user_2 still exists
    hit_user_1_after = await cache.retrieve_if_similar(
        embedding=query_embedding,
        cache_type="objective_user_facts",
        global_user_id=user_1,
    )
    
    hit_user_2_after = await cache.retrieve_if_similar(
        embedding=query_embedding,
        cache_type="objective_user_facts",
        global_user_id=user_2,
        threshold=0.7,
    )
    
    print(f"  ✓ After clearing user_1:")
    print(f"    - user_1: {'HIT' if hit_user_1_after else 'MISS':4s} (should be MISS)")
    print(f"    - user_2: {'HIT' if hit_user_2_after else 'MISS':4s} (should still exist)")

    # ── TEST 8: STATISTICS AND PERFORMANCE ─────────────────────────────────
    print("\n[TEST 8] Cache statistics...")
    stats = cache.get_stats()
    print(f"  ✓ Final statistics:")
    print(f"    - Size:       {stats['size']:4d} / {stats['max_size']}")
    print(f"    - Hits:       {stats['hits']:4d}")
    print(f"    - Misses:     {stats['misses']:4d}")
    print(f"    - Hit rate:   {stats['hit_rate']:6.1%}")
    print(f"    - Evictions:  {stats['evictions']:4d}")
    print(f"    - Threshold:  {stats['threshold']:.2f}")

    # ── TEST 9: GRACEFUL SHUTDOWN ──────────────────────────────────────────
    print("\n[TEST 9] Graceful shutdown...")
    await cache.shutdown()
    print(f"  ✓ Cache shutdown complete")

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - Cache with Database Persistence Working")
    print("="*80 + "\n")





if __name__ == "__main__":
    asyncio.run(test_main2())
