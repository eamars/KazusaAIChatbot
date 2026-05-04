# Memory Evolution Interface Control Document

## Document Control

- ICD id: `ME-ICD-001`
- Owning package: `kazusa_ai_chatbot.memory_evolution`
- Interface boundary: evolving shared-memory unit APIs -> DB interface ->
  `memory` collection
- Runtime consumers: legacy `save_memory` facade, seed maintenance CLI,
  persistent-memory retrieval, future reflection integration
- Upstream data owners: seed JSONL source, future reflection pipeline, manual
  admin writers
- Downstream owners: RAG persistent-memory retrieval and future memory
  promotion flows

This document defines the contract for the evolving shared-memory package. It
is the source of truth for what the package may read, what it may write, how it
may reach MongoDB, and how other modules may call it.

## Purpose

`memory_evolution` owns stable lifecycle mechanics for the shared `memory`
collection. It provides:

- stable memory-unit ids and lineage ids,
- idempotent inserts,
- supersede and merge operations,
- active-only retrieval for normal runtime use,
- reset/reseed of curated global knowledge,
- embedding ownership,
- synchronous Cache2 invalidation.

The package does not decide what new lore or memory should exist. Callers must
provide already-approved memory documents with provenance and privacy review
metadata when the source is not seed-managed.

## Scope

This ICD covers:

- public module entry points,
- DB-interface boundary rules,
- evolving memory document shape,
- authority, source-kind, and lifecycle semantics,
- embedding and cache invalidation ownership,
- reset/reseed behavior,
- persistent-memory retrieval constraints,
- dependency and import rules,
- future reflection integration requirements.

This ICD does not cover:

- reflection prompt design,
- lore candidate extraction,
- user-memory consolidation,
- conversation-history selection,
- active chat cognition,
- platform adapter parsing,
- scheduler orchestration,
- manual approval UX.

Those features must integrate through explicit package entry points rather than
by importing repository internals or operating on MongoDB directly.

## Boundary Summary

```text
callers
  -> memory_evolution public repository/reset APIs
       validate evolving memory contract
       own embedding generation request
       own lineage/status/cache semantics
  -> db.memory_evolution
       owns MongoDB filters, writes, vector search, and write guard document
  -> memory collection
```

The package boundary is deliberate: `memory_evolution` may not use Motor,
`get_db`, raw collection methods, Mongo aggregation syntax, or raw embedding
adapters directly. Database and embedding operations belong in
`kazusa_ai_chatbot.db.memory_evolution`.

## Public Entry Points

Consumers should use package exports from `kazusa_ai_chatbot.memory_evolution`:

```python
from kazusa_ai_chatbot.memory_evolution import (
    find_active_memory_units,
    insert_memory_unit,
    merge_memory_units,
    reset_memory_from_seed,
    supersede_memory_unit,
)
```

### `insert_memory_unit`

```python
async def insert_memory_unit(
    *,
    document: EvolvingMemoryDoc,
) -> EvolvingMemoryDoc
```

Inserts one memory unit idempotently by `memory_unit_id`.

The caller must provide:

- `memory_unit_id`
- `lineage_id`
- `memory_name`
- `content`
- `memory_type`
- `source_kind`
- `authority`
- `status`
- `timestamp` when preserving source capture time matters
- `evidence_refs` and `privacy_review` for non-seed promoted rows

The caller must not provide `embedding`. Embeddings are repository-owned. If a
row with the same `memory_unit_id` already exists, the repository returns it
only when the non-volatile fields are equivalent. Different content under the
same id is rejected.

### `supersede_memory_unit`

```python
async def supersede_memory_unit(
    *,
    active_unit_id: str,
    replacement: EvolvingMemoryDoc,
) -> EvolvingMemoryDoc
```

Replaces one active, non-expired memory unit with the next version in the same
lineage. The replacement must carry the same `lineage_id` as the active source.
The repository sets:

- `version` to source version plus one,
- `supersedes_memory_unit_ids` to `[active_unit_id]`,
- source row `status` to `superseded`.

Inactive, expired, missing, or lineage-mismatched sources are rejected.

### `merge_memory_units`

```python
async def merge_memory_units(
    *,
    source_unit_ids: list[str],
    replacement: EvolvingMemoryDoc,
) -> EvolvingMemoryDoc
```

Merges active, non-expired memory units into one replacement. Source ids must
be unique.

For one source lineage, the replacement must keep that lineage and receives the
next version. For multiple source lineages, the replacement must provide a new
lineage id and starts at version one. The repository sets:

- `merged_from_memory_unit_ids` to the unique source ids,
- all source rows `status` to `superseded`.

### `find_active_memory_units`

```python
async def find_active_memory_units(
    *,
    query: MemoryUnitQuery,
    limit: int,
) -> list[tuple[float, EvolvingMemoryDoc]]
```

Returns active, non-expired memory units with retrieval scores through a
constrained query shape. It does not expose raw MongoDB filters.

Allowed query fields:

```python
{
    "semantic_query": str,
    "memory_name": str,
    "memory_name_contains": str,
    "source_global_user_id": str,
    "memory_type": str,
    "source_kind": str,
    "authority": str,
    "lineage_id": str,
    "exclude_memory_unit_ids": list[str],
}
```

`exclude_memory_unit_ids` is for duplicate-detection and promotion flows that
must find similar active rows while excluding a known source row.

Semantic queries return Atlas `vectorSearchScore` values. Metadata-only queries
return `-1.0`, matching the keyword-search convention used by legacy memory
search. Promotion callers must use the returned score when deciding whether a
candidate is similar enough to supersede or merge instead of inserting a new
lineage.

### `reset_memory_from_seed`

```python
async def reset_memory_from_seed(
    *,
    dry_run: bool,
) -> MemoryResetResult
```

Resets seed-managed global memory rows from
`personalities/knowledge/memory_seed.jsonl`.

Dry-run reports planned changes without writes, embeddings, deletion, or cache
invalidation. Apply mode acquires the shared memory write guard, deletes
obsolete seed-managed global rows, upserts changed seed rows with embeddings,
and invalidates memory Cache2 entries.

## Package-Internal Shared Helpers

`reset.py` may use these public repository helpers because reset and runtime
writes must share normalization and equivalence semantics:

```python
from kazusa_ai_chatbot.memory_evolution.repository import (
    document_with_embedding,
    invalidate_memory_cache,
    memory_documents_equivalent,
    normalize_memory_document,
)
```

External packages should not call these helpers unless this ICD is updated to
promote them as supported public entry points.

## DB Interface

All MongoDB and embedding access for this package is isolated in:

```python
kazusa_ai_chatbot.db.memory_evolution
```

The repository may import it only as the DB interface:

```python
from kazusa_ai_chatbot.db import memory_evolution as memory_store
```

Approved DB-interface functions:

```python
async def compute_memory_embedding(text: str) -> list[float]
def build_active_memory_filter(now_timestamp: str) -> dict[str, Any]
async def find_memory_unit_by_id(memory_unit_id: str) -> EvolvingMemoryDoc | None
async def insert_memory_unit_document(document: EvolvingMemoryDoc) -> None
async def replace_memory_unit_document(document: EvolvingMemoryDoc) -> None
async def update_memory_unit_fields(memory_unit_id: str, fields: dict[str, Any]) -> None
async def update_many_memory_unit_fields(memory_unit_ids: list[str], fields: dict[str, Any]) -> None
async def find_active_memory_documents(...) -> list[tuple[float, EvolvingMemoryDoc]]
async def count_legacy_seed_managed_memory() -> int
async def count_unmanaged_seed_memory(seed_ids: list[str]) -> int
async def count_runtime_reflection_memory() -> int
async def delete_reset_seed_managed_memory(seed_ids: list[str]) -> int
async def acquire_memory_write_lock(owner: str, write_time: str) -> bool
async def release_memory_write_lock() -> None
```

`update_memory_unit_fields` and `update_many_memory_unit_fields` are lifecycle
updates only. They may set only:

```python
{"status", "updated_at", "expiry_timestamp"}
```

No future module may use these helpers as a generic `$set` escape hatch.

## Data Contracts

### `EvolvingMemoryDoc`

```python
{
    "memory_unit_id": str,
    "lineage_id": str,
    "version": int,
    "memory_name": str,
    "content": str,
    "source_global_user_id": str,
    "memory_type": str,
    "source_kind": str,
    "authority": str,
    "status": str,
    "supersedes_memory_unit_ids": list[str],
    "merged_from_memory_unit_ids": list[str],
    "evidence_refs": list[MemoryEvidenceRef],
    "privacy_review": MemoryPrivacyReview,
    "confidence_note": str,
    "timestamp": str,
    "updated_at": str,
    "expiry_timestamp": str | None,
    "embedding": list[float],
}
```

`embedding` exists only on persisted rows and repository return values after
write operations. Retrieval APIs must remove or avoid returning embeddings
unless a future audit API explicitly documents otherwise.

### Status Values

```python
active
superseded
rejected
expired
fulfilled
```

Normal runtime retrieval returns only `active` rows whose `expiry_timestamp` is
missing, null, or later than the query time.

### Authority Values

```python
seed
reflection_promoted
manual
```

`seed` is for repository-managed JSONL rows. `manual` is for direct human or
admin insertion. `reflection_promoted` is for extracted or inferred memory that
has passed the caller's promotion and privacy gate.

### Source-Kind Values

```python
seeded_manual
external_imported
reflection_inferred
conversation_extracted
relationship_inferred
```

Reset manages only seed-lane global rows:

```python
source_global_user_id == ""
source_kind in {"seeded_manual", "external_imported"}
```

Runtime reflection rows such as `source_kind="reflection_inferred"` must be
preserved by seed reset.

### Evidence and Privacy

Future promotion callers must provide evidence and privacy review metadata
beside the memory document. These fields are not reconstructed by the
repository and must not be inferred from prompt text after identifying fields
have been stripped.

```python
{
    "evidence_refs": [
        {
            "reflection_run_id": str,
            "scope_ref": str,
            "message_refs": [
                {
                    "conversation_history_id": str,
                    "platform": str,
                    "platform_channel_id": str,
                    "channel_type": str,
                    "timestamp": str,
                    "role": str,
                }
            ],
            "captured_at": str,
            "source": str,
        }
    ],
    "privacy_review": {
        "private_detail_risk": "low|medium|high",
        "user_details_removed": bool,
        "boundary_assessment": str,
        "reviewer": "automated_llm|human|seed_tool",
    },
}
```

## Identity and Lineage

Callers own stable ids. The repository validates presence and lifecycle
semantics, but it does not invent runtime promotion ids.

Seed reset uses deterministic ids from:

```python
seed_memory_unit_id(
    memory_name=...,
    source_global_user_id=...,
    source_kind=...,
)
```

Legacy `save_memory` callers without ids receive deterministic compatibility
ids derived from their semantic payload. New runtime promotion code should
construct ids explicitly so retries and merge decisions are transparent.

## Embedding Ownership

Embeddings are owned by the repository and DB interface:

- callers must not provide `embedding`,
- `insert_memory_unit`, `supersede_memory_unit`, and `merge_memory_units`
  compute embeddings for inserted replacements,
- reset apply computes embeddings only for changed seed rows,
- reset dry-run computes zero embeddings,
- semantic retrieval computes query embeddings through the DB interface.

The source text for embeddings comes from
`memory_evolution.identity.memory_embedding_source_text`.

## Write Guard and Cache Invalidation

Memory writes and reset share a guard document in the `memory` collection:

```text
_id = "__memory_evolution_write_lock__"
```

The guard prevents reset and runtime writes from mutating the shared collection
concurrently. Write APIs release the guard in `finally` blocks.

Every successful memory write or reset apply must invalidate Cache2 before
returning success:

```python
CacheInvalidationEvent(source="memory", ...)
```

Persistent-memory cache dependencies must use `source="memory"`.

## Seed Reset Contract

The canonical seed file is:

```text
personalities/knowledge/memory_seed.jsonl
```

Reset behavior:

- validates all seed rows before mutation,
- reports destructive work in dry-run,
- deletes legacy seed-managed global rows without evolving ids,
- deletes seed-managed global rows absent from the current seed file when
  pruning is enabled,
- preserves future runtime reflection rows,
- preserves existing seed timestamps when updating rows,
- computes embeddings only for changed rows in apply mode,
- invalidates memory cache after apply.

Seed reset must not read reflection outputs, conversation history, user memory
units, user profiles, character state, or external data.

## Retrieval Contract

Normal persistent-memory retrieval is active-only and non-expired.

LLM-facing tools must not expose `status`, `expiry_before`, or `expiry_after`
arguments. Audit/debug code that needs inactive rows must use an explicit
lower-level interface and document that it is outside normal RAG retrieval.

Semantic retrieval uses vector prefilters only for indexed scalar fields and
post-filters with the full active query. Keyword retrieval uses regex filters
and does not construct vector-only state.

## Compatibility Interfaces

### Legacy `save_memory`

`kazusa_ai_chatbot.db.memory.save_memory` is a compatibility facade. It wraps
`insert_memory_unit` so old callers stop writing raw memory rows directly.

Authority inference for legacy calls:

- explicit `authority` is preserved,
- `conversation_extracted`, `reflection_inferred`, and
  `relationship_inferred` default to `reflection_promoted`,
- other legacy calls default to `manual`.

New code should prefer `memory_evolution` public entry points directly.

### Seed Maintenance CLI

`scripts.manage_memory_knowledge` owns local JSONL editing and delegates sync
to `reset_memory_from_entries`.

### Reset CLI

```powershell
python -m scripts.reset_memory_lore --dry-run
python -m scripts.reset_memory_lore --apply
```

`--dry-run` is the required inspection path before applying destructive seed
changes.

## Dependency Rules

Allowed imports from `memory_evolution`:

- typed contracts from `memory_evolution.models`,
- id helpers from `memory_evolution.identity`,
- public entry points from `memory_evolution.__init__`,
- internal modules within `memory_evolution`,
- `kazusa_ai_chatbot.db.memory_evolution` as the DB interface,
- Cache2 invalidation event/runtime helpers for memory cache invalidation.

Forbidden imports and calls:

- `memory_evolution` importing `get_db`, `get_text_embedding`, Motor, PyMongo,
  or raw MongoDB clients directly.
- `memory_evolution` calling `.find(...)`, `.aggregate(...)`,
  `.insert_one(...)`, `.update_one(...)`, `.update_many(...)`,
  `.delete_one(...)`, `.delete_many(...)`, `.replace_one(...)`, or
  `.count_documents(...)`.
- `memory_evolution` importing `reflection_cycle`, cognition nodes, dialog
  nodes, platform adapters, service handlers, scheduler dispatch, or
  conversation-history DB modules.
- DB-interface lifecycle update helpers accepting arbitrary fields.
- Prompt or LLM code writing memory rows without calling public
  `memory_evolution` APIs.
- Reset reading conversation history, reflection artifacts, user state, or
  external data.
- Retrieval tools exposing inactive-row controls to LLM callers.

The only DB module that may use raw MongoDB operations for evolving memory is
`kazusa_ai_chatbot.db.memory_evolution`.

## Integration Points

### Current Runtime Memory Writers

Legacy global/shared memory writers must route through `save_memory` or direct
`memory_evolution` public APIs. They must not write `db.memory` directly.

### Current Persistent-Memory Retrieval

RAG persistent-memory tools call DB search helpers that default to active,
non-expired memory. Tool arguments intentionally hide lifecycle override
controls.

### Future Reflection Integration

Reflection may consume its own approved outputs and then call
`insert_memory_unit`, `supersede_memory_unit`, `merge_memory_units`, and
`find_active_memory_units`.

Reflection must provide:

- stable ids,
- source kind and authority,
- evidence refs from source metadata,
- privacy review,
- confidence note,
- candidate content that has already passed promotion gates.

Reflection duplicate detection must call `find_active_memory_units` with a
semantic query and use the returned score tuples. It must not re-fetch
embeddings for Python-side cosine similarity, call lower-level DB helpers, or
silently default to inserting every approved candidate as a new lineage when
similar active memory exists.

Reflection output alone is evidence, not authorization to write memory.
Promotion policy and privacy stripping belong outside this package. This
package enforces the persistence contract after a caller has made the promotion
decision.

### Future Scheduler Integration

A scheduler may invoke reset or future promotion workers, but it must call the
public facade for that worker. Scheduler code must not call DB-interface
functions or repository internals directly.

## Verification Checklist

Before merging memory-evolution changes, verify:

- Static boundary tests prove `memory_evolution` does not touch MongoDB
  directly.
- Repository tests cover idempotent insert, caller-supplied embedding
  rejection, supersede, merge, inactive source rejection, evidence refs, and
  cache invalidation.
- DB-interface tests cover active filters, vector prefiltering,
  `exclude_memory_unit_ids`, and lifecycle update allowlisting.
- Reset tests cover dry-run, apply, runtime-row preservation, and write guard
  failure.
- Writer migration tests prove legacy `save_memory` delegates to the evolving
  API and preserves truthful authority.
- Seed maintenance validation passes for
  `personalities/knowledge/memory_seed.jsonl`.
- Reset dry-run is inspected before reset apply.

Recommended commands:

```powershell
pytest tests\test_memory_evolution_module_boundary.py -q
pytest tests\test_memory_evolution_repository.py -q
pytest tests\test_memory_evolution_reset.py -q
pytest tests\test_memory_evolution_retrieval.py -q
pytest tests\test_memory_evolution_idempotency.py -q
pytest tests\test_memory_evolution_writer_migration.py -q
python -m scripts.manage_memory_knowledge validate
python -m scripts.reset_memory_lore --dry-run
```

## Change Control

Changes to this ICD are required when:

- a public entry point is added, removed, or changes semantics,
- `EvolvingMemoryDoc`, `MemoryUnitQuery`, `MemoryEvidenceRef`, or
  `MemoryPrivacyReview` changes,
- a new source kind, authority, status, or memory type is introduced,
- a DB-interface function gains a new operation shape,
- reset manages a new source lane,
- retrieval exposes inactive or expired rows,
- another package consumes memory-evolution outputs,
- reflection integration begins writing memory,
- scheduler integration is added,
- cache invalidation behavior changes.

Changing the DB boundary is a breaking architectural decision. It must be
reviewed as an interface change, not as an implementation cleanup.
