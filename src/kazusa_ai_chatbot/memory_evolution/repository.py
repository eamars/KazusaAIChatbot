"""Repository APIs for evolving shared memory units."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot.db import memory_evolution as memory_store
from kazusa_ai_chatbot.memory_evolution.identity import (
    deterministic_memory_unit_id,
    memory_embedding_source_text,
    seed_memory_unit_id,
)
from kazusa_ai_chatbot.memory_evolution.models import (
    EvolvingMemoryDoc,
    MemoryAuthority,
    MemorySourceKind,
    MemoryStatus,
    MemoryUnitQuery,
    MemoryUnitSearchResult,
    VALID_MEMORY_AUTHORITIES,
    VALID_MEMORY_SOURCE_KINDS,
    VALID_MEMORY_STATUSES,
)
from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime

_IGNORED_IDEMPOTENCY_FIELDS = {"_id", "embedding", "updated_at"}
_QUERY_FIELDS = {
    "semantic_query",
    "memory_name",
    "memory_name_contains",
    "source_global_user_id",
    "memory_type",
    "source_kind",
    "authority",
    "lineage_id",
    "exclude_memory_unit_ids",
}


def now_iso() -> str:
    """Return the current UTC timestamp string."""
    current_time = datetime.now(timezone.utc).isoformat()
    return current_time


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _expiry_is_past(expiry_timestamp: object, now_timestamp: str) -> bool:
    if expiry_timestamp is None:
        return False
    if not isinstance(expiry_timestamp, str) or not expiry_timestamp.strip():
        return False
    expiry_time = _parse_timestamp(expiry_timestamp.strip())
    now_time = _parse_timestamp(now_timestamp)
    return_value = expiry_time <= now_time
    return return_value


def normalize_memory_document(
    document: EvolvingMemoryDoc,
    *,
    updated_at: str,
) -> EvolvingMemoryDoc:
    """Validate and normalize a caller-supplied memory unit.

    Args:
        document: Candidate memory-unit document without caller-supplied embedding.
        updated_at: Write timestamp to place on the normalized document.

    Returns:
        Normalized memory-unit document ready for embedding and persistence.
    """
    if "embedding" in document:
        raise ValueError("memory embeddings are repository-owned")

    memory_unit_id = str(document.get("memory_unit_id", "")).strip()
    lineage_id = str(document.get("lineage_id", "")).strip()
    if not memory_unit_id:
        raise ValueError("memory_unit_id is required")
    if not lineage_id:
        raise ValueError("lineage_id is required")

    memory_name = str(document.get("memory_name", "")).strip()
    content = str(document.get("content", "")).strip()
    memory_type = str(document.get("memory_type", "")).strip()
    source_kind = str(
        document.get("source_kind", MemorySourceKind.SEEDED_MANUAL)
    ).strip()
    authority = str(document.get("authority", MemoryAuthority.MANUAL)).strip()
    status = str(document.get("status", MemoryStatus.ACTIVE)).strip()
    if not memory_name:
        raise ValueError("memory_name is required")
    if not content:
        raise ValueError("content is required")
    if not memory_type:
        raise ValueError("memory_type is required")
    if source_kind not in VALID_MEMORY_SOURCE_KINDS:
        raise ValueError(f"invalid memory source_kind: {source_kind!r}")
    if authority not in VALID_MEMORY_AUTHORITIES:
        raise ValueError(f"invalid memory authority: {authority!r}")
    if status not in VALID_MEMORY_STATUSES:
        raise ValueError(f"invalid memory status: {status!r}")

    expiry_timestamp = document.get("expiry_timestamp")
    if status == MemoryStatus.ACTIVE and _expiry_is_past(
        expiry_timestamp,
        updated_at,
    ):
        status = MemoryStatus.EXPIRED

    version = document.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("version must be a positive integer")

    normalized: EvolvingMemoryDoc = {
        "memory_unit_id": memory_unit_id,
        "lineage_id": lineage_id,
        "version": version,
        "memory_name": memory_name,
        "content": content,
        "source_global_user_id": str(
            document.get("source_global_user_id", "")
        ).strip(),
        "memory_type": memory_type,
        "source_kind": source_kind,
        "authority": authority,
        "status": status,
        "supersedes_memory_unit_ids": list(
            document.get("supersedes_memory_unit_ids", [])
        ),
        "merged_from_memory_unit_ids": list(
            document.get("merged_from_memory_unit_ids", [])
        ),
        "evidence_refs": list(document.get("evidence_refs", [])),
        "privacy_review": dict(document.get("privacy_review", {})),
        "confidence_note": str(document.get("confidence_note", "")).strip(),
        "timestamp": str(document.get("timestamp", updated_at)).strip(),
        "updated_at": updated_at,
        "expiry_timestamp": expiry_timestamp,
    }
    return normalized


def _without_ignored_fields(document: dict[str, Any]) -> dict[str, Any]:
    comparable = {
        key: value
        for key, value in document.items()
        if key not in _IGNORED_IDEMPOTENCY_FIELDS
    }
    return comparable


def memory_documents_equivalent(
    existing: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    """Compare persisted and candidate memory documents for idempotent writes."""
    existing_view = _without_ignored_fields(existing)
    candidate_view = _without_ignored_fields(candidate)
    return_value = existing_view == candidate_view
    return return_value


async def document_with_embedding(document: EvolvingMemoryDoc) -> EvolvingMemoryDoc:
    """Attach a repository-owned embedding to a normalized memory unit.

    Args:
        document: Normalized memory-unit document.

    Returns:
        Copy of the document with an embedding vector.
    """
    embedding = await memory_store.compute_memory_embedding(
        memory_embedding_source_text(document)
    )
    return_value: EvolvingMemoryDoc = {
        **document,
        "embedding": embedding,
    }
    return return_value


async def invalidate_memory_cache(
    *,
    document: dict[str, Any],
    reason: str,
) -> None:
    """Invalidate memory-derived RAG cache entries after memory changes."""
    runtime = get_rag_cache2_runtime()
    await runtime.invalidate(
        CacheInvalidationEvent(
            source="memory",
            global_user_id=str(document.get("source_global_user_id", "")).strip(),
            timestamp=str(document.get("updated_at", "")).strip(),
            reason=reason,
        )
    )


async def _acquire_memory_write_guard(owner: str, write_time: str) -> None:
    """Fail fast when another shared-memory mutation is running.

    Args:
        owner: Current operation requesting the shared write guard.
        write_time: Timestamp recorded on the guard document.
    """
    lock_acquired = await memory_store.acquire_memory_write_lock(
        owner,
        write_time,
    )
    if not lock_acquired:
        raise RuntimeError("memory write or reset is already running")


def _active_source_or_raise(
    document: dict[str, Any],
    field_name: str,
    *,
    now_timestamp: str,
) -> None:
    if document.get("status") != MemoryStatus.ACTIVE:
        raise ValueError(f"{field_name} must be active")
    expiry_timestamp = document.get("expiry_timestamp")
    if _expiry_is_past(expiry_timestamp, now_timestamp):
        raise ValueError(f"{field_name} must be active and non-expired")


async def insert_memory_unit(*, document: EvolvingMemoryDoc) -> EvolvingMemoryDoc:
    """Insert an idempotent evolving memory unit.

    Args:
        document: Caller-supplied memory unit with stable ``memory_unit_id`` and
            ``lineage_id``. The caller must not provide an embedding.

    Returns:
        The existing equivalent or newly persisted document.
    """
    write_time = now_iso()
    normalized = normalize_memory_document(document, updated_at=write_time)
    await _acquire_memory_write_guard("insert_memory_unit", write_time)
    try:
        existing = await memory_store.find_memory_unit_by_id(
            normalized["memory_unit_id"],
        )
        if existing is not None:
            if memory_documents_equivalent(existing, normalized):
                return_value: EvolvingMemoryDoc = dict(existing)
                return return_value
            raise ValueError(
                "memory_unit_id already exists with different content"
            )

        payload = await document_with_embedding(normalized)
        await memory_store.insert_memory_unit_document(payload)
        await invalidate_memory_cache(
            document=payload,
            reason="memory_unit_inserted",
        )
        return payload
    finally:
        await memory_store.release_memory_write_lock()


async def supersede_memory_unit(
    *,
    active_unit_id: str,
    replacement: EvolvingMemoryDoc,
) -> EvolvingMemoryDoc:
    """Replace one active memory unit with the next lineage version.

    Args:
        active_unit_id: ``memory_unit_id`` of the active row being superseded.
        replacement: New memory unit using a stable id and the same lineage id.

    Returns:
        The newly inserted replacement document.
    """
    write_time = now_iso()
    await _acquire_memory_write_guard("supersede_memory_unit", write_time)
    try:
        target = await memory_store.find_memory_unit_by_id(active_unit_id)
        if target is None:
            raise ValueError(f"memory unit not found: {active_unit_id!r}")
        _active_source_or_raise(
            target,
            "supersede target",
            now_timestamp=write_time,
        )

        expected_lineage = str(target["lineage_id"])
        replacement_lineage = str(replacement.get("lineage_id", "")).strip()
        if replacement_lineage != expected_lineage:
            raise ValueError("replacement lineage_id must match the target lineage")
        expected_version = int(target.get("version", 1)) + 1

        replacement_doc: EvolvingMemoryDoc = {
            **replacement,
            "lineage_id": expected_lineage,
            "version": expected_version,
            "supersedes_memory_unit_ids": [active_unit_id],
        }
        normalized = normalize_memory_document(
            replacement_doc,
            updated_at=write_time,
        )
        existing = await memory_store.find_memory_unit_by_id(
            normalized["memory_unit_id"],
        )
        if existing is not None:
            raise ValueError("replacement memory_unit_id already exists")

        payload = await document_with_embedding(normalized)
        await memory_store.insert_memory_unit_document(payload)
        await memory_store.update_memory_unit_fields(
            active_unit_id,
            {
                "status": MemoryStatus.SUPERSEDED,
                "updated_at": write_time,
            },
        )
        await invalidate_memory_cache(
            document=payload,
            reason="memory_unit_superseded",
        )
        return payload
    finally:
        await memory_store.release_memory_write_lock()


async def merge_memory_units(
    *,
    source_unit_ids: list[str],
    replacement: EvolvingMemoryDoc,
) -> EvolvingMemoryDoc:
    """Merge active memory units into one replacement unit.

    Args:
        source_unit_ids: Active source ``memory_unit_id`` values.
        replacement: New memory unit with caller-supplied stable ids.

    Returns:
        The newly inserted replacement document.
    """
    if not source_unit_ids:
        raise ValueError("source_unit_ids is required")
    unique_source_ids = list(dict.fromkeys(source_unit_ids))
    if len(unique_source_ids) != len(source_unit_ids):
        raise ValueError("source_unit_ids must be unique")

    write_time = now_iso()
    await _acquire_memory_write_guard("merge_memory_units", write_time)
    try:
        sources: list[dict[str, Any]] = []
        for source_unit_id in unique_source_ids:
            source = await memory_store.find_memory_unit_by_id(source_unit_id)
            if source is None:
                raise ValueError(f"memory unit not found: {source_unit_id!r}")
            _active_source_or_raise(
                source,
                "merge source",
                now_timestamp=write_time,
            )
            sources.append(source)

        source_lineages = {str(source["lineage_id"]) for source in sources}
        replacement_lineage = str(replacement.get("lineage_id", "")).strip()
        if len(source_lineages) == 1:
            expected_lineage = next(iter(source_lineages))
            if replacement_lineage != expected_lineage:
                raise ValueError("replacement lineage_id must match source lineage")
            expected_version = (
                max(int(source.get("version", 1)) for source in sources) + 1
            )
        else:
            if not replacement_lineage or replacement_lineage in source_lineages:
                raise ValueError("replacement lineage_id must be new for merged lineages")
            expected_lineage = replacement_lineage
            expected_version = 1

        replacement_doc: EvolvingMemoryDoc = {
            **replacement,
            "lineage_id": expected_lineage,
            "version": expected_version,
            "merged_from_memory_unit_ids": unique_source_ids,
        }
        normalized = normalize_memory_document(
            replacement_doc,
            updated_at=write_time,
        )
        existing = await memory_store.find_memory_unit_by_id(
            normalized["memory_unit_id"],
        )
        if existing is not None:
            raise ValueError("replacement memory_unit_id already exists")

        payload = await document_with_embedding(normalized)
        await memory_store.insert_memory_unit_document(payload)
        await memory_store.update_many_memory_unit_fields(
            unique_source_ids,
            {
                "status": MemoryStatus.SUPERSEDED,
                "updated_at": write_time,
            },
        )
        await invalidate_memory_cache(
            document=payload,
            reason="memory_units_merged",
        )
        return payload
    finally:
        await memory_store.release_memory_write_lock()


def _validate_query_shape(query: MemoryUnitQuery) -> None:
    unsupported = set(query) - _QUERY_FIELDS
    if unsupported:
        raise ValueError(f"unsupported memory query fields: {sorted(unsupported)}")


async def find_active_memory_units(
    *,
    query: MemoryUnitQuery,
    limit: int,
) -> list[MemoryUnitSearchResult]:
    """Find active, non-expired memory units with retrieval scores.

    Args:
        query: Constrained metadata and optional semantic query shape.
        limit: Maximum rows to return.

    Returns:
        ``(score, memory document)`` pairs without embeddings. Semantic queries
        use vector-search scores; metadata-only queries use score ``-1.0``.
    """
    if limit <= 0:
        return_value: list[MemoryUnitSearchResult] = []
        return return_value

    _validate_query_shape(query)
    semantic_query = query.get("semantic_query")
    query_embedding = None
    if semantic_query:
        query_embedding = await memory_store.compute_memory_embedding(
            semantic_query,
        )
    matches = await memory_store.find_active_memory_documents(
        query=query,
        limit=limit,
        now_timestamp=now_iso(),
        query_embedding=query_embedding,
    )
    return_value = [(score, dict(doc)) for score, doc in matches]
    return return_value
