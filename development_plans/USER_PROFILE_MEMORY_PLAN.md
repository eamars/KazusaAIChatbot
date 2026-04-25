# User Profile Memory: Addressable & Vector-Searchable Recall

**Status:** Draft — 2026-04-25
**Scope:** RAG layer + consolidation layer + schema migration

---

## Problem Statement

The current `UserProfileDoc` stores three growing arrays inside the profile document:

| Field                | Type                        | Write Pattern                           |
| -------------------- | --------------------------- | --------------------------------------- |
| `character_diary`    | `list[CharacterDiaryEntry]` | append-only                             |
| `objective_facts`    | `list[ObjectiveFactEntry]`  | upsert by text (case-insensitive dedup) |
| `active_commitments` | `list[ActiveCommitmentDoc]` | upsert by action text                   |

All three are loaded deterministically — either in full or as a tail window — with no semantic
relevance filtering. As each list grows:

1. **Forgetting**: early entries scroll out of whatever window is loaded for a turn. A promise
   made 200 sessions ago is silently absent even though it is still `active`.
2. **No topic affinity**: the loaded set is the same regardless of whether the user is asking
   about their job, a hobby, or a past commitment. Irrelevant entries compete for context budget.
3. **No addressability**: there is no way to ask "what do we know about this user's relationship
   with their sister?" without loading every fact.
4. **Unbounded document growth**: MongoDB document size caps at 16 MB; embedded arrays with
   no eviction path will eventually breach this.

---

## Proposed Architecture

### Core Idea

Extract the three growing arrays out of `UserProfileDoc` into a dedicated collection
`user_profile_memories`. Each document in that collection is a single atomic memory unit
with its own text embedding. The profile document becomes a lightweight header; the memory
collection is the authoritative source for diary entries, facts, and commitments.

Loading switches from *deterministic tail* to *recent baseline + topic-targeted semantic recall*,
combining both into the existing image-builder step.

```
Write Path (Consolidator)
─────────────────────────
  facts_harvester / relationship_recorder
          │
          │  raw diary entries, facts, commitments
          ▼
    [embed content]
          │
          ▼
  user_profile_memories  (new collection, one doc per memory unit)
          │
          ▼
    bump rag_version  →  invalidate user's RAG cache

Read Path (RAG dispatcher)
──────────────────────────
  current topic/query
          │
          ├─── (A) Recent memories   (sorted by created_at DESC, limit per type)
          │
          ├─── (B) Semantic recall   (vector search by topic embedding, cosine ≥ threshold)
          │         DEEP only; threshold is configurable per memory_type
          │
          ├─── (C) Merge + dedup     (by memory_id, cap total budget)
          │
          ▼
    build_user_image(merged_memories)  →  existing image format unchanged
```

No change to the image format or how it is fed into cognition nodes.

---

## New Collection: `user_profile_memories`

### Document Schema

```python
class UserProfileMemoryDoc(TypedDict, total=False):
    memory_id: str           # UUID4, primary key
    global_user_id: str      # Owner; indexed
    memory_type: str         # Discriminator — see MemoryType enum below
    content: str             # Human-readable text; this is what gets embedded
    embedding: list[float]   # Vector of `content`; used for vector search
    created_at: str          # ISO-8601 UTC; immutable after insert
    updated_at: str          # ISO-8601 UTC; mutable (for commitment status)

    # ── Type-specific fields (sparse) ────────────────────────────────────────
    category: str            # ObjectiveFactEntry.category when memory_type=fact
    source: str              # ObjectiveFactEntry.source when memory_type=fact
    confidence: float        # 0.0–1.0; diary and fact entries
    context: str             # CharacterDiaryEntry.context when memory_type=diary

    # Commitment-specific
    commitment_id: str       # Stable ID across status updates (not memory_id)
    commitment_type: str     # ActiveCommitmentDoc.commitment_type
    target: str              # Who/what the commitment is about
    status: str              # active | fulfilled | expired | superseded
    due_time: str            # ISO-8601 UTC, optional

    # Milestone-specific
    event_category: str      # preference | relationship_state | permission | revelation
    scope: str               # For supersedence resolution
    superseded_by: str       # memory_id of the superseding memory, if any

    # Search support
    deleted: bool            # Soft delete; default false
```

### `MemoryType` Values

```
diary_entry     — CharacterDiaryEntry (subjective, character's voice)
objective_fact  — ObjectiveFactEntry  (verified user fact)
milestone       — user_image.milestones entry (notable life event)
commitment      — ActiveCommitmentDoc  (promise / obligation)
```

Commitments remain status-mutable. All other types are append-only (never edited, only
soft-deleted when superseded).

### MongoDB Indexes

```python
# Primary lookup by owner, sorted by recency
{"global_user_id": 1, "memory_type": 1, "created_at": -1}

# Vector search (Atlas Search / $vectorSearch)
# Index name: "user_profile_memories_vector"
# Field: "embedding", type: "vector", dimensions: <model_dim>, similarity: "cosine"
# Pre-filter support: global_user_id, memory_type, deleted, status

# Soft-delete filter (compound)
{"global_user_id": 1, "deleted": 1}

# Commitment status lookups
{"global_user_id": 1, "commitment_id": 1}
{"global_user_id": 1, "status": 1, "memory_type": 1}

# Commitment expiry sweep
{"memory_type": 1, "status": 1, "due_time": 1}
```

---

## Updated `UserProfileDoc`

Fields **removed** (data moves to `user_profile_memories`):

```python
character_diary: list[CharacterDiaryEntry]      # REMOVED
diary_updated_at: str                            # REMOVED
objective_facts: list[ObjectiveFactEntry]        # REMOVED
facts_updated_at: str                            # REMOVED
active_commitments: list[ActiveCommitmentDoc]    # REMOVED
active_commitments_updated_at: str               # REMOVED
```

Fields **kept** (lightweight metadata and generated artifacts):

```python
global_user_id: str
platform_accounts: list[PlatformAccountDoc]
suspected_aliases: list[str]
affinity: int
last_relationship_insight: str
user_image: dict                 # Generated; rebuilt on each consolidation run
facts: list[str]                 # DEPRECATED; kept for backward compat; not written
```

`user_image` remains in the profile document because it is a *generated artifact* — a
compressed, ready-to-inject summary — not source data. Source data now lives in
`user_profile_memories`.

---

## RAG Layer Changes

### New Loading Sequence (`persona_supervisor2_rag.py`)

The RAG dispatcher gains a new internal function `_load_user_memories()` that replaces
the direct reads of `user_profile.character_diary`, `user_profile.objective_facts`, and
`user_profile.active_commitments`.

```
_load_user_memories(global_user_id, topic_context, db, depth):
    Step A — Recent baseline
        For each memory_type in [diary_entry, objective_fact, milestone, commitment]:
            recent[type] = db.user_profile_memories.find(
                {global_user_id, deleted: false, status≠expired/fulfilled (for commitments)},
                sort=[created_at DESC],
                limit=RECENT_LIMITS[type]
            )

    Step B — Semantic recall  (DEEP only)
        IF depth == DEEP:
            query_embedding = embed(topic_context)   # decontextualized_input verbatim
            For each memory_type in [diary_entry, objective_fact, milestone]:
                # commitments are fully covered by Step A; skip here
                hits = db.user_profile_memories.aggregate([
                    $vectorSearch(
                        index="user_profile_memories_vector",
                        queryVector=query_embedding,
                        numCandidates=100,
                        limit=25,
                        filter={global_user_id: X, deleted: false, memory_type: type}
                    )
                ])
                semantic_hits += [h for h in hits if h.score ≥ SEMANTIC_THRESHOLDS[type]]

    Step C — Merge + dedup
        seen_ids = set()
        merged = []
        # Priority: recent first (higher trust), then semantic
        for m in flatten(recent.values()) + semantic_hits:
            if m.memory_id not in seen_ids and len(merged) < MEMORY_BUDGET:
                seen_ids.add(m.memory_id)
                merged.append(m)
        return merged
```

**Budget constants** (initial values; override via `settings.json`):

```python
RECENT_LIMITS = {
    "diary_entry":    6,
    "objective_fact": 8,
    "milestone":      10,   # milestones are sparse; load more
    "commitment":     10,   # all active, non-expired commitments
}

# Per-type cosine similarity floor; configurable in settings.json
# Commitments omitted — they bypass vector search entirely (Step A only)
SEMANTIC_THRESHOLDS = {
    "diary_entry":    0.75,  # higher bar; subjective prose embeds noisily
    "objective_fact": 0.72,
    "milestone":      0.72,
}

MEMORY_BUDGET = 40          # Total cap across all types after merge
```

Active commitments are always loaded in full via Step A regardless of topic. They bypass
vector search because a commitment's relevance is not topic-dependent — an outstanding
promise must be visible in every turn. Commitments past their `due_time` are treated as
`expired` and excluded from Step A results.

### RAG Cache Integration

Add a new cache namespace:

```python
"user_profile_memories": 900,   # 15 min TTL (same as user_promises today)
```

Cache key: `SHA256(global_user_id + topic_embedding_hex[:16])`

The existing `rag_version` bump in `db_writer` already invalidates this namespace — no
change needed there. The cache stores the merged memory list (post-dedup) keyed by the
topic query so repeated turns on the same topic do not re-run the vector search.

### Depth Classifier Impact

No change to the SHALLOW/DEEP decision. The new memory loader runs for both depths:

- **SHALLOW**: runs Steps A + C only — deterministic recent baseline, no embedding call
- **DEEP**: runs Steps A + B + C — full semantic recall

The classifier's existing affinity and cache-hit logic is unaffected.

---

## Consolidation Layer Changes

### Write Path in `db_writer`

Current calls that are **replaced**:

```python
# OLD — remove these
await upsert_character_diary(global_user_id, diary_entries)
await upsert_objective_facts(global_user_id, new_facts)
await upsert_active_commitments(global_user_id, future_promises)
```

New function: `insert_profile_memories(global_user_id, memories, db)`

```
insert_profile_memories(global_user_id, memories, db):
    For each memory in memories:
        1. Generate embedding: memory.embedding = embed(memory.content)
        2. Resolve due_time (commitments only):
               if memory.due_time is None:
                   memory.due_time = memory.created_at + 10 days
        3. Insert into user_profile_memories
           - Diary entries and facts: always insert (append-only)
           - Commitments: check for existing active commitment with same
             commitment_id (legacy) or matching (target, action) pair;
             if found, update status in-place rather than inserting a duplicate
           - Milestones: check scope field; if existing milestone with same scope
             exists, set superseded_by on old entry, insert new one
```

Dedup rules by type:

| Type             | Dedup Key                           | On Collision                                         |
| ---------------- | ----------------------------------- | ---------------------------------------------------- |
| `diary_entry`    | None (always append)                | —                                                    |
| `objective_fact` | `content` text, case-insensitive    | Skip insert if near-duplicate exists (cosine ≥ 0.95) |
| `commitment`     | `(target, action)` pair, normalised | Update `status` / `updated_at` on existing doc       |
| `milestone`      | `scope` value                       | Set `superseded_by` on old; insert new               |

### `facts_harvester` Node Changes

Currently emits `new_facts: list[str]` and `future_promises: list[dict]`. These flow through
`_build_objective_fact_entries()` and `_build_active_commitment_entries()` in persistence.py.

**Change**: persistence builder functions are moved to `_build_memory_docs()` which returns a
unified `list[UserProfileMemoryDoc]` covering all types. The caller (`db_writer`) calls
`insert_profile_memories()` with this list.

No change to the LLM prompt in `facts_harvester` — it still emits the same structured output.
The transformation layer absorbs the type unification.

### `relationship_recorder` Node Changes

Currently emits `diary_entry: list[str]`. These become `diary_entry` typed `UserProfileMemoryDoc`
instances. No change to the LLM prompt.

### `consolidator_images.py` Changes

The image builder currently reads `user_profile.user_image` for milestone lifecycle and
`new_facts` for the current turn's additions.

**Change**: `_apply_milestone_lifecycle()` reads from `user_profile_memories` (filtered to
`memory_type=milestone`) rather than `user_image.milestones`. The built `user_image` dict
is still written back to `UserProfileDoc.user_image` as today — it remains a generated
summary artifact, not moved to the memories collection.

---

## Migration Strategy

### Phase M1 — Dual-write (non-breaking)

Enable `insert_profile_memories()` alongside the existing upsert calls. Both paths write
simultaneously. The new collection accumulates data from new conversations. No read path
change yet.

### Phase M2 — Backfill script

One-shot migration script reads `character_diary`, `objective_facts`, `active_commitments`
from every `UserProfileDoc`, converts them to `UserProfileMemoryDoc` instances, generates
embeddings, and inserts into `user_profile_memories`. The script is idempotent (skip if
`memory_id` already exists from dual-write).

```
scripts/migrate_user_profile_memories.py
    --batch-size 100
    --dry-run
    --user-ids ...  (optional subset)
```

### Phase M3 — Switch reads

Update `_load_user_memories()` to read from `user_profile_memories` instead of profile
arrays. Run in production with the profile arrays still present as fallback.

### Phase M4 — Stop old writes, remove arrays from profile

Remove the upsert calls from db_writer. Drop `character_diary`, `objective_facts`,
`active_commitments` from `UserProfileDoc`. The migration is complete.

Phases M1 → M4 are deployed as separate releases, each behind a feature flag in
`settings.json`:

```json
"features": {
    "user_profile_memories_dual_write": false,
    "user_profile_memories_read": false,
    "user_profile_memories_only": false
}
```

---

## Task Breakdown

**No need to consider backward compatibility**

### Schema & DB

- [x] **T1** — Add `UserProfileMemoryDoc` TypedDict to `db/schemas.py`
- [x] **T2** — Add `MemoryType` string enum constants to `db/schemas.py`
- [x] **T3** — Add MongoDB index declarations to `db/indexes.py` (or equivalent init script)
- [x] **T4** — Add Atlas Search vector index definition for `user_profile_memories`
- [x] **T5** — Remove array fields from `UserProfileDoc` (Phase M4 only)

### Persistence Layer

- [x] **T6** — Add `insert_profile_memories(global_user_id, memories, db)` to `db/users.py`; resolve `due_time` default (created_at + 10 days) for commitments with no explicit due date
- [x] **T7** — Add `query_profile_memories_recent(global_user_id, limits, db)` to `db/users.py`; exclude commitments where `due_time` < now
- [x] **T8** — Add `query_profile_memories_vector(global_user_id, embedding, limits, thresholds, db)` to `db/users.py`; accept per-type threshold dict from `settings.json`
- [x] **T9** — Add `update_commitment_status(global_user_id, commitment_id, new_status, db)` to `db/users.py`
- [x] **T10** — Add background sweep `expire_overdue_commitments(db)` that sets `status=expired` on commitment docs where `due_time` < now and `status=active`; run on scheduler heartbeat

### Consolidation

- [x] **T11** — Refactor `_build_objective_fact_entries()` and `_build_active_commitment_entries()` in `persona_supervisor2_consolidator_persistence.py` into `_build_memory_docs()` returning `list[UserProfileMemoryDoc]`
- [x] **T12** — Update `db_writer` node to call `insert_profile_memories()` (dual-write in M1, replace in M4)
- [x] **T13** — Update `consolidator_images.py` `_apply_milestone_lifecycle()` to read from `user_profile_memories`

### RAG Layer

- [x] **T14** — Add `_load_user_memories(global_user_id, topic_context, db, depth)` to `persona_supervisor2_rag.py`
- [x] **T15** — Add `"user_profile_memories"` cache namespace with TTL=900 to `rag/cache.py`
- [x] **T16** — Wire `_load_user_memories()` into the RAG dispatcher (replace direct profile array reads)
- [x] **T17** — Update depth classifier: SHALLOW path calls Steps A+C; DEEP path calls A+B+C
- [x] **T18** — Expose `SEMANTIC_THRESHOLDS` and `RECENT_LIMITS` in `settings.json` under `rag.user_profile_memories` — implemented as env-backed constants in `config.py` per final plan.

### Migration

- [x] **T19** — Write `scripts/migrate_user_profile_memories.py` with `--dry-run` + `--batch-size`; backfill `due_time` for existing commitments using `created_at + 10 days` where null
- [x] **T20** — Add feature flags to `settings.json` — intentionally replaced by big-bang implementation with no feature flags and env-backed config.

### Tests

- [x] **T21** — Unit tests for `insert_profile_memories()`: dedup rules per type, `due_time` defaulting
- [x] **T22** — Unit tests for `_load_user_memories()`: budget cap, dedup, SHALLOW (A+C) vs DEEP (A+B+C) paths, expired commitment exclusion
- [x] **T23** — Unit tests for `expire_overdue_commitments()`: transitions active→expired at due_time boundary
- [x] **T24** — Integration test for full write→read round-trip (consolidator → memories → RAG → image)
- [x] **T25** — Migration script dry-run test against fixture data

---

## Design Decisions & Rationale

### One collection vs. four

The user's proposal listed `user_facts`, `user_milestone`, `user_character_diary`,
`user_character_commitments` as candidate separate collections. This plan uses a single
`user_profile_memories` collection with a `memory_type` discriminator instead, because:

- A single vector index covers all types; cross-type recall (e.g., find both a diary entry
  and a related fact about the same topic) is a single query with no joins.
- Type-specific queries are handled by adding `memory_type` to the pre-filter — no
  performance difference.
- Index management and cache invalidation stay simple.
- The existing `memory` collection precedent in this codebase already uses this pattern.

### Why not reuse the existing `memory` collection

The `memory` collection uses 15-minute TTL and is keyed as a short-lived RAG cache
namespace (`"internal_memory": 900`). User profile memories are persistent with no
planned TTL. Sharing the collection would conflate two different data lifecycles.

### Commitment mutability

Commitments are the only type that require status mutation after insert. Rather than
deleting and re-inserting, `update_commitment_status()` patches `status` and `updated_at`
in-place. The `commitment_id` field (stable across status changes) provides the update key,
matching the existing `ActiveCommitmentDoc.commitment_id` field.

### Embedding at write time vs. read time

Embedding is computed in the consolidator's `db_writer` node, synchronously before
inserting into MongoDB. This keeps the read path fast (vector search only, no embedding
step). The embedding model used is the same one already wired into the RAG layer.

### Commitment auto-expiry

Every commitment must have a `due_time`. If the LLM does not extract one from conversation,
`insert_profile_memories()` defaults it to `created_at + 10 days`. A background sweep
(`expire_overdue_commitments`) marks commitments with `due_time < now` as `expired`; this
runs on the existing scheduler heartbeat. Step A excludes expired commitments at query time
as a double-check, so no commitment ever surfaces as active past its due date even if the
sweep is delayed.

### Semantic threshold tuning

Per-type cosine similarity thresholds are configurable in `settings.json` under
`rag.user_profile_memories.semantic_thresholds`. The initial values (`diary_entry: 0.75`,
`objective_fact: 0.72`, `milestone: 0.72`) are starting estimates. Adjust based on observed
false-positive / false-negative rates in real conversation logs. Commitments have no threshold
— they bypass vector search entirely.

### Topic context for semantic recall

`topic_context` passed to `_load_user_memories()` is the `decontextualized_input` produced
by the decontextualiser node — already available in RAG state at the point memories are
loaded. This is the cleanest single-string representation of "what this turn is about" without
adding an extra LLM call or coupling to a new multi-stage retrieval pipeline.
