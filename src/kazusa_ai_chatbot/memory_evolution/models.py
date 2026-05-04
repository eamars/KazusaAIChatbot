"""Typed contracts and constants for evolving shared memory units."""

from __future__ import annotations

from typing import Literal, TypedDict


class MemoryStatus:
    """Lifecycle constants for shared memory units."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FULFILLED = "fulfilled"


class MemoryAuthority:
    """Authority lane constants for shared memory units."""

    SEED = "seed"
    REFLECTION_PROMOTED = "reflection_promoted"
    MANUAL = "manual"


class MemorySourceKind:
    """Source-kind constants for shared memory units."""

    SEEDED_MANUAL = "seeded_manual"
    EXTERNAL_IMPORTED = "external_imported"
    REFLECTION_INFERRED = "reflection_inferred"
    CONVERSATION_EXTRACTED = "conversation_extracted"
    RELATIONSHIP_INFERRED = "relationship_inferred"


VALID_MEMORY_STATUSES = {
    MemoryStatus.ACTIVE,
    MemoryStatus.SUPERSEDED,
    MemoryStatus.REJECTED,
    MemoryStatus.EXPIRED,
    MemoryStatus.FULFILLED,
}
VALID_MEMORY_AUTHORITIES = {
    MemoryAuthority.SEED,
    MemoryAuthority.REFLECTION_PROMOTED,
    MemoryAuthority.MANUAL,
}
VALID_MEMORY_SOURCE_KINDS = {
    MemorySourceKind.SEEDED_MANUAL,
    MemorySourceKind.EXTERNAL_IMPORTED,
    MemorySourceKind.REFLECTION_INFERRED,
    MemorySourceKind.CONVERSATION_EXTRACTED,
    MemorySourceKind.RELATIONSHIP_INFERRED,
}
SEED_MANAGED_SOURCE_KINDS = {
    MemorySourceKind.SEEDED_MANUAL,
    MemorySourceKind.EXTERNAL_IMPORTED,
}


class MemoryEvidenceMessageRef(TypedDict, total=False):
    """Optional source-message reference attached to a memory evidence record."""

    conversation_history_id: str
    platform: str
    platform_channel_id: str
    channel_type: str
    timestamp: str
    role: str


class MemoryEvidenceRef(TypedDict, total=False):
    """Evidence metadata describing where a memory unit came from."""

    reflection_run_id: str
    scope_ref: str
    message_refs: list[MemoryEvidenceMessageRef]
    captured_at: str
    source: str


class MemoryPrivacyReview(TypedDict, total=False):
    """Privacy review summary attached to a shared memory unit."""

    private_detail_risk: Literal["low", "medium", "high"]
    user_details_removed: bool
    boundary_assessment: str
    reviewer: Literal["automated_llm", "human", "seed_tool"]


class EvolvingMemoryDoc(TypedDict, total=False):
    """MongoDB document shape for the evolving shared ``memory`` collection."""

    memory_unit_id: str
    lineage_id: str
    version: int
    memory_name: str
    content: str
    source_global_user_id: str
    memory_type: str
    source_kind: str
    authority: str
    status: str
    supersedes_memory_unit_ids: list[str]
    merged_from_memory_unit_ids: list[str]
    evidence_refs: list[MemoryEvidenceRef]
    privacy_review: MemoryPrivacyReview
    confidence_note: str
    timestamp: str
    updated_at: str
    expiry_timestamp: str | None
    embedding: list[float]


MemoryUnitSearchResult = tuple[float, EvolvingMemoryDoc]


class MemoryUnitQuery(TypedDict, total=False):
    """Constrained active-memory query shape used by repository readers."""

    semantic_query: str
    memory_name: str
    memory_name_contains: str
    source_global_user_id: str
    memory_type: str
    source_kind: str
    authority: str
    lineage_id: str
    exclude_memory_unit_ids: list[str]


class MemoryResetResult(TypedDict):
    """Summary counters for a seed reset or dry-run."""

    dry_run: bool
    seed_rows_loaded: int
    seed_rows_inserted: int
    seed_rows_updated: int
    seed_rows_unchanged: int
    seed_rows_deleted: int
    legacy_rows_deleted: int
    runtime_rows_preserved: int
    embeddings_computed: int
    cache_invalidated: bool
    warnings: list[str]
