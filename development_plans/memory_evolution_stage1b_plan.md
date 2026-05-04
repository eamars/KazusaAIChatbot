# memory evolution stage 1b plan

## Summary

- Goal: Upgrade the existing global persistent `memory` collection, search path, Cache2 dependencies, and seed tooling into an evolving active/superseded memory-unit model with explicit reset, lineage, evidence, privacy, embedding, and cache-invalidation contracts for Stage 1c.
- Plan class: large
- Status: completed
- Mandatory skills: `memory-knowledge-maintenance`, `development-plan-writing`, `py-style`, `test-style-and-execution`, `database-data-pull`, `local-llm-architecture`
- Overall cutover strategy: bigbang within the `memory` subsystem only. The existing global `memory` rows may be erased and re-seeded from current codebase seed files into the new evolving-unit schema.
- Highest-risk areas: destructive memory reset, accidental deletion of future reflection-inferred lore, duplicate writes on retry, lineage corruption, embedding cost during reseed, persistent-memory retrieval regressions, vector index filter behavior, stale Cache2 persistent-memory results, and unmanaged legacy memory writers.
- Acceptance criteria: memory reset/reseed works from current repository seed data only and preserves runtime `reflection_inferred` rows after cutover; persistent-memory search reads active non-expired rows by default; memory rows support validated insert/supersede/merge lineage; IDs are idempotent; embeddings are computed inside the memory API; Cache2 memory invalidation is synchronous from the caller perspective; no reflection, conversation-history, generation LLM, or external signal is accepted.

## Context

Stage 1b is database/search/seeding work only. It prepares global memory to support future lore evolution, but it does not decide what new lore should exist.

The current codebase has:

- `memory` collection used by persistent-memory RAG helpers.
- `db/memory.py` with append-only `save_memory` and `search_memory`.
- `personalities/knowledge/memory_seed.jsonl` as curated seed lane when present.
- `scripts.manage_memory_knowledge` as local seed maintenance tooling.
- Persistent-memory helper caches whose dependencies currently need memory-specific invalidation.

Current direct writers into the `memory` collection before implementation are:

- `src/kazusa_ai_chatbot/db/memory.py::save_memory`
- `src/scripts/manage_memory_knowledge.py::sync_entries`
- `src/scripts/insert_memory.py`
- `src/scripts/inject_knowledge.py`
- test-only/live-test helpers that call `save_memory`

Stage 1b must migrate these writers to the new `memory_evolution` API or make
them thin wrappers around it. No direct `db.memory.insert_one`,
`db.memory.update_one`, or `db.memory.delete_many` writer may remain outside the
seed/reset implementation after cutover.

Stage 1b must be independent from Stage 1a and Stage 1c.

## Mandatory Skills

- `memory-knowledge-maintenance`: load before changing `memory`, seed sync, or memory maintenance scripts.
- `development-plan-writing`: load before modifying this plan.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `database-data-pull`: load before exporting current memory rows for diagnostics.
- `local-llm-architecture`: load before changing RAG helper prompts, tool schemas, retrieval planning, or LLM-facing memory-search behavior.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major checklist stage, the active agent must reread this entire plan before starting the next stage.
- Stage 1b must not import or call `reflection_cycle`.
- Stage 1b must not read `conversation_history`, `conversation_episode_state`, `user_memory_units`, or `character_state` for memory content.
- Stage 1b must not call any generation LLM. Embedding calls through the existing embedding endpoint are allowed only inside memory write/search APIs and only when `dry_run=False`.
- Stage 1b must not use web search or external data.
- Stage 1b may use only current codebase seed files and existing `memory` DB rows for dry-run reporting.
- Stage 1b cutover is bigbang for legacy global `memory`: pre-cutover DB rows do not need preservation unless re-created from current codebase seed data.
- Stage 1b must not add lore promotion, reflection candidates, or daily reflection.
- `reset_memory_from_seed` must reset only seed-managed global rows with `source_kind in {"seeded_manual", "external_imported"}` after cutover. It must preserve `source_kind="reflection_inferred"` runtime rows.
- `reset_memory_from_seed` is offline/admin-only. It must not run concurrently with runtime memory writes; implementation must use a lock, guard document, or fail-fast mutual exclusion.
- Public insert/supersede/merge APIs must compute embeddings internally and must not require callers such as Stage 1c to supply embeddings.
- Public insert/supersede/merge APIs must validate caller-supplied `memory_unit_id` and `lineage_id`; they must not silently auto-generate retry-unsafe IDs for runtime writes.
- Public write APIs must emit Cache2 `source="memory"` invalidation synchronously before returning success.
- Persistent-memory retrieval must default to `status="active"` and must exclude rows whose `expiry_timestamp` is in the past.
- Superseded rows must not appear in normal RAG persistent-memory results.
- Seed sync must not prune future `source_kind="reflection_inferred"` runtime lore rows.

## Must Do

- Replace the flat global memory contract with evolving memory units:
  - `memory_unit_id`
  - `lineage_id`
  - `version`
  - `status`
  - `supersedes_memory_unit_ids`
  - `merged_from_memory_unit_ids`
  - `evidence_refs`
  - `authority`
  - `privacy_review`
  - `updated_at`
- Define `MemoryResetResult`, `MemoryEvidenceRef`, `MemoryPrivacyReview`, `MemoryUnitQuery`, and authority/status/source-kind constants in `memory_evolution.models`.
- Add `memory_evolution` package with public database-only APIs:

```python
async def reset_memory_from_seed(*, dry_run: bool) -> MemoryResetResult: ...
async def insert_memory_unit(*, document: EvolvingMemoryDoc) -> EvolvingMemoryDoc: ...
async def supersede_memory_unit(*, active_unit_id: str, replacement: EvolvingMemoryDoc) -> EvolvingMemoryDoc: ...
async def merge_memory_units(*, source_unit_ids: list[str], replacement: EvolvingMemoryDoc) -> EvolvingMemoryDoc: ...
async def find_active_memory_units(*, query: MemoryUnitQuery, limit: int) -> list[tuple[float, EvolvingMemoryDoc]]: ...
```

- Make `find_active_memory_units` accept only the constrained `MemoryUnitQuery` shape defined in this plan; it must not accept raw MongoDB filters.
- Return `(score, EvolvingMemoryDoc)` pairs from `find_active_memory_units`; semantic queries must return Atlas vector similarity scores, while metadata-only queries use score `-1.0`.
- Add reset/reseed CLI:

```powershell
python src\scripts\reset_memory_lore.py --dry-run
python src\scripts\reset_memory_lore.py --apply
```

- Update `db/bootstrap.py` memory indexes.
- Update `db/memory.py` search/save helpers for evolving active rows.
- Update `rag/memory_retrieval_tools.py` so persistent-memory search defaults to active rows.
- Remove LLM-controllable status/expiry filters from persistent-memory RAG tools; audit/debug status access must use lower-level DB or `memory_evolution` APIs.
- Update Cache2 policy so persistent-memory helpers depend on `source="memory"` instead of `source="user_profile"`.
- Emit synchronous Cache2 `source="memory"` invalidation from `insert_memory_unit`, `supersede_memory_unit`, `merge_memory_units`, and `reset_memory_from_seed`.
- Update memory seed tooling so current seed data can populate evolving rows and future reflection-inferred rows are not managed by seed pruning.
- Migrate current memory writers (`save_memory`, `manage_memory_knowledge`, `insert_memory.py`, `inject_knowledge.py`) to use or wrap `memory_evolution`.
- Add deterministic tests for reset/reseed, ID idempotency, active-only retrieval, supersede/merge edge cases, seed tooling, writer migration, and Cache2 invalidation.

## Deferred

- Reflection-cycle code.
- Daily lore promotion.
- Reading conversation data.
- Generation LLM calls. Embedding calls remain allowed only under the rules in this plan.
- Prompt changes unless required only to preserve existing search-path wording.
- Service worker integration.
- Autonomous messages.
- User memory changes.

## Cutover Policy

| Area | Policy | Notes |
|---|---|---|
| `memory` collection | bigbang | Current rows may be deleted and re-seeded from current repo seed files. |
| Persistent-memory search path | bigbang | Search contract becomes active evolving rows only. |
| Reset/reseed | bigbang | Initial cutover may delete legacy rows; post-cutover reset preserves runtime `reflection_inferred` rows. |
| Seed tooling | bigbang | Seed lane writes evolving rows; future runtime lore is not seed-managed or pruned. |
| Cache2 dependencies | compatible | Persistent-memory caches move to memory invalidation source. |
| Legacy memory writers | bigbang | Existing direct writers must migrate or wrap the new API. |
| Reflection | compatible | No reflection dependency or integration in Stage 1b. |

## Agent Autonomy Boundaries

- The agent may choose private helper names only inside `memory_evolution`.
- The agent must not add any reflection-specific fields beyond generic `source_kind` and evidence fields.
- The agent must not preserve old memory rows unless they are re-created from current seed data.
- The agent must not add a compatibility retrieval path that returns superseded rows in normal RAG.
- The agent must not expose raw MongoDB query filters through `find_active_memory_units`.
- The agent must not make Stage 1c or any caller supply embeddings.
- The agent must not allow reset/reseed to delete `reflection_inferred` rows after cutover.
- If the seed file is missing, start from an empty memory collection and record it in execution evidence.

## Target State

```text
current codebase seed data
  -> reset_memory_lore.py
  -> memory collection in evolving-unit shape
  -> persistent-memory retrieval filters active rows
  -> Cache2 invalidates on memory writes
```

No reflection output, conversation transcript, or external information enters Stage 1b.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Plan class | large | DB/search/seeding changes are broad but isolated. |
| Cutover | Bigbang memory-only | User accepts erasing current lore and re-seeding. |
| External signal | Forbidden | Stage 1b only prepares storage/search mechanics. |
| Reset semantics | Reset seed-managed global rows only after cutover | Admin reseed must not delete future reflection-promoted lore. |
| Initial cutover | Legacy global rows may be wiped | User accepted bigbang reseed; this applies before 1c runtime rows exist. |
| Retrieval | Active non-expired rows only by default | Prevents old, superseded, or expired lore conflict. |
| Runtime lore protection | Exclude from reset and seed pruning | Needed for Stage 1c future reflection-inferred rows. |
| ID ownership | Caller supplies stable IDs for runtime writes | Retries must not duplicate promoted lore. |
| Embedding ownership | Memory API computes embeddings | 1c should not know embedding contracts or vector index mechanics. |
| Cache invalidation | Synchronous `source="memory"` event before write API returns | The next chat turn must see memory changes after promotion. |
| Tool status filters | Removed from RAG tool surface | Normal LLM-facing retrieval should not ask for superseded or expired rows. |

## Data Contracts

```python
class MemoryStatus:
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FULFILLED = "fulfilled"


class MemoryAuthority:
    SEED = "seed"
    REFLECTION_PROMOTED = "reflection_promoted"
    MANUAL = "manual"


class MemorySourceKind:
    SEEDED_MANUAL = "seeded_manual"
    EXTERNAL_IMPORTED = "external_imported"
    REFLECTION_INFERRED = "reflection_inferred"
    CONVERSATION_EXTRACTED = "conversation_extracted"
    RELATIONSHIP_INFERRED = "relationship_inferred"


class MemoryEvidenceMessageRef(TypedDict, total=False):
    conversation_history_id: str
    platform: str
    platform_channel_id: str
    channel_type: str
    timestamp: str
    role: str


class MemoryEvidenceRef(TypedDict, total=False):
    reflection_run_id: str
    scope_ref: str
    message_refs: list[MemoryEvidenceMessageRef]
    captured_at: str
    source: str


class MemoryPrivacyReview(TypedDict, total=False):
    private_detail_risk: Literal["low", "medium", "high"]
    user_details_removed: bool
    boundary_assessment: str
    reviewer: Literal["automated_llm", "human", "seed_tool"]


class EvolvingMemoryDoc(TypedDict, total=False):
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
```

```python
class MemoryUnitQuery(TypedDict, total=False):
    semantic_query: str
    memory_name: str
    memory_name_contains: str
    source_global_user_id: str
    memory_type: str
    source_kind: str
    authority: str
    lineage_id: str
    exclude_memory_unit_ids: list[str]
```

`find_active_memory_units` returns `list[tuple[float, EvolvingMemoryDoc]]`.
The score is required for future promotion logic to decide whether a candidate
is similar enough to supersede or merge instead of inserting a new lineage.
For metadata-only reads, the score is `-1.0`.

```python
class MemoryResetResult(TypedDict):
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
```

Required statuses:

```text
active
superseded
rejected
expired
fulfilled
```

Status and expiry rules:

- `status` is the lifecycle source of truth.
- Normal retrieval returns only `status="active"` rows whose
  `expiry_timestamp` is missing, null, or greater than the current timestamp.
- Write APIs must reject or normalize active rows with past `expiry_timestamp`
  to `status="expired"` before persistence.
- `status="expired"` rows must not appear in normal RAG retrieval even when
  their `expiry_timestamp` is missing.
- Audit/debug callers may pass an explicit status to lower-level DB APIs, but
  LLM-facing persistent-memory tools must not expose status or expiry controls.

Authority values:

```text
seed
reflection_promoted
manual
```

Lineage and idempotency rules:

- Public runtime writes require stable caller-supplied `memory_unit_id` and
  `lineage_id`; missing IDs are validation errors.
- Seed/reset writes generate deterministic `memory_unit_id` values from the
  seed key `(memory_name, source_global_user_id, source_kind)` and set
  `lineage_id=memory_unit_id` for version 1 seed rows.
- `insert_memory_unit` is idempotent by `memory_unit_id`: if the existing row is
  byte-equivalent excluding `updated_at` and embedding, return it; if the same
  id points to different content, fail.
- `supersede_memory_unit` requires the target row to be active. Already
  superseded, expired, rejected, or fulfilled targets fail; the API does not
  trace forward or silently overwrite.
- A supersede replacement must use the target's `lineage_id`, set
  `version=target.version + 1`, and include the target id in
  `supersedes_memory_unit_ids`.
- `merge_memory_units` requires all source rows to be active. When all sources
  share one lineage, the replacement inherits that lineage and uses
  `version=max(source.version)+1`. When sources have different lineages, the
  replacement must use a new `lineage_id`, `version=1`, and include every
  source id in `merged_from_memory_unit_ids`. Source rows become superseded.

Embedding rules:

- `embedding` is stored in `EvolvingMemoryDoc` but is repository-owned.
- Public callers must not supply embeddings; write APIs compute embeddings from
  the same semantic text contract used by `db.memory.memory_embedding_source_text`.
- Dry-run reset performs no embedding calls. Apply reset computes embeddings
  only for inserted or changed seed rows.
- Stage 1b does not drop or rebuild the vector index during reset. It keeps the
  existing vector index and writes fresh embeddings with changed documents.
- `find_active_memory_units` supports semantic duplicate detection by accepting
  `semantic_query`; the repository computes the query embedding internally.

Search API rules:

- `db.memory.search_memory` defaults to `status="active"` and applies the
  active non-expired filter by default.
- Audit/debug callers may pass `status="superseded"`, `status="expired"`, or
  `status=None` to the lower-level DB helper. `status=None` means no status
  filter and is not allowed through LLM-facing tools.
- `rag.memory_retrieval_tools.search_persistent_memory` and
  `search_persistent_memory_keyword` must always use active non-expired search.
  They must not expose status or expiry arguments in the tool schema.

Minimum indexes:

```text
memory_unit_id_unique:
  [("memory_unit_id", 1)], unique

memory_lineage_version:
  [("lineage_id", 1), ("version", -1)]

memory_active_lookup:
  [("status", 1), ("memory_type", 1), ("source_kind", 1), ("updated_at", -1)]

memory_seed_sync_lookup:
  [("memory_name", 1), ("source_global_user_id", 1), ("source_kind", 1)]
```

## Change Surface

### Create

- `src/kazusa_ai_chatbot/memory_evolution/__init__.py`
- `src/kazusa_ai_chatbot/memory_evolution/models.py`
- `src/kazusa_ai_chatbot/memory_evolution/repository.py`
- `src/kazusa_ai_chatbot/memory_evolution/reset.py`
- `src/kazusa_ai_chatbot/memory_evolution/README.md`
- `src/kazusa_ai_chatbot/memory_evolution/identity.py`
- `src/kazusa_ai_chatbot/db/memory_evolution.py`
- `src/scripts/reset_memory_lore.py`
- `tests/test_memory_evolution_module_boundary.py`
- `tests/test_memory_evolution_repository.py`
- `tests/test_memory_evolution_reset.py`
- `tests/test_memory_evolution_retrieval.py`
- `tests/test_memory_evolution_idempotency.py`
- `tests/test_memory_evolution_writer_migration.py`
- `tests/test_memory_knowledge_sync_runtime_lore.py`
- `tests/test_persistent_memory_cache_invalidation.py`

### Modify

- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/db/bootstrap.py`
- `src/kazusa_ai_chatbot/db/memory.py`
- `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py`
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`
- `src/scripts/manage_memory_knowledge.py`
- `src/scripts/export_memory.py`

### Keep

- `reflection_cycle`
- `conversation_history` search
- cognition prompts
- service worker behavior
- consolidator user-memory writers

## Implementation Order

1. Add deterministic tests for reset/reseed, evolving repository, active-only retrieval, seed pruning boundary, and Cache2 invalidation.
2. Add idempotency, lineage edge-case, writer-migration, and expiry/status tests.
3. Add evolving memory schemas and repository helpers.
4. Add reset/reseed implementation and CLI with offline guard semantics.
5. Update bootstrap indexes.
6. Update memory search helpers and RAG retrieval tools.
7. Update Cache2 dependency policy and synchronous invalidation events.
8. Migrate seed/manual memory writers to the new API.
9. Run focused tests.
10. Run reset dry-run and record current-row deletion/seed-row insertion summary.
11. Run broader tests.

## Progress Checklist

- [x] Tests for memory evolution contract added.
- [x] Tests for idempotency, lineage edge cases, writer migration, and expiry added.
- [x] Evolving memory schemas and repository helpers implemented.
- [x] Reset/reseed CLI implemented.
- [x] Bootstrap indexes updated.
- [x] Active-only persistent-memory retrieval implemented.
- [x] Memory Cache2 dependency implemented.
- [x] Seed tooling updated.
- [x] Legacy memory writers migrated or wrapped.
- [x] Focused tests pass.
- [x] Reset dry-run evidence recorded.

## Verification

```powershell
pytest tests\test_memory_evolution_repository.py -q
pytest tests\test_memory_evolution_reset.py -q
pytest tests\test_memory_evolution_retrieval.py -q
pytest tests\test_memory_evolution_idempotency.py -q
pytest tests\test_memory_evolution_writer_migration.py -q
pytest tests\test_memory_knowledge_sync_runtime_lore.py -q
pytest tests\test_persistent_memory_cache_invalidation.py -q
```

```powershell
python src\scripts\reset_memory_lore.py --dry-run
venv\Scripts\python.exe -m scripts.manage_memory_knowledge validate
venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync
```

## Acceptance Criteria

- Reset/reseed dry-run reports destructive memory changes before apply.
- Apply path can reset/reseed memory from current codebase seed data.
- Post-cutover reset/reseed preserves `source_kind="reflection_inferred"` rows.
- `MemoryResetResult` reports reset counters, preserved runtime rows, embedding calls, warnings, and cache invalidation.
- Embeddings are computed internally for changed writes and are not caller-supplied by Stage 1c.
- Normal persistent-memory search returns active non-expired rows only.
- LLM-facing persistent-memory tools do not expose status or expiry filters.
- Supersede and merge preserve validated post-cutover lineage and fail on inactive targets.
- Insert is idempotent by `memory_unit_id` and retries do not create duplicate rows.
- `find_active_memory_units` supports constrained metadata filters plus semantic vector duplicate detection.
- Seed sync does not manage or prune future `reflection_inferred` rows.
- Cache2 persistent-memory helper entries can be invalidated by `source="memory"`.
- Memory write APIs emit memory invalidation before returning success.
- Current direct memory writers are migrated to or wrapped around `memory_evolution`.
- No reflection, generation LLM, conversation-history, or external data is used.

## Validation Against Current Implementation

This stage has been implemented in the current workspace. Evidence is recorded
below; the memory evolution package depends on `src/kazusa_ai_chatbot/db`
interfaces for MongoDB access and does not operate Motor collections directly.

| Area | Expected by plan | Current implementation | Status |
|---|---|---|---|
| `memory_evolution` package | Required | `src/kazusa_ai_chatbot/memory_evolution` exists with models, identity helpers, repository APIs, reset API, and README | Implemented |
| Reset/reseed CLI | Required | `src/scripts/reset_memory_lore.py` exists and dry-run evidence is recorded | Implemented |
| Evolving memory schema | Required | `memory_evolution.models` defines evolving documents, reset result, query, evidence, privacy, authority, status, and source-kind contracts | Implemented |
| Active-only persistent-memory retrieval | Required | `db.memory.search_memory` defaults to active non-expired rows and LLM-facing tools expose no status/expiry filters | Implemented |
| Cache2 memory invalidation | Required | Persistent-memory Cache2 dependencies use `source="memory"` and write APIs emit `CacheInvalidationEvent(source="memory")` | Implemented |
| Memory writer migration | Required | `save_memory` wraps `memory_evolution`; seed sync delegates to reset; manual scripts use `save_memory` | Implemented |
| Reset semantics | Required | Reset deletes seed-managed rows only, preserves `reflection_inferred` rows by source-kind, and shares a DB-interface write guard with runtime memory writes | Implemented |
| ID and lineage validation | Required | Repository tests cover idempotency, active-only supersede, inactive rejection, and merge lineage behavior | Implemented |
| Embedding ownership | Required | Public writes reject caller-supplied embeddings and compute embeddings through the DB interface | Implemented |
| Reflection independence | Required | Current Stage 1a code does not import `memory_evolution`; Stage 1b still must not import `reflection_cycle` | Boundary currently preserved |

Stage 1b implementation and database-apply evidence is recorded below. Stage
1c may consume the memory evolution APIs after reviewing this evidence.

## Plan Sign-Off

- Approved for implementation: 2026-05-05
- Sign-off scope: Stage 1b plan only; no code or database changes are signed off here.
- Sign-off basis: the plan now defines reset semantics, ID/idempotency, lineage, evidence refs, privacy review, authority values, embedding ownership, active-only retrieval, Cache2 invalidation, writer migration, and 1c handoff contracts.
- Implementation completed: 2026-05-05
- Completion basis: focused tests, static boundary check, compile check, memory export, reset apply, post-apply dry-run, seed validation, and seed sync dry-run recorded below.

## Execution Evidence

- Focused test results:
  - `venv\Scripts\python.exe -m pytest tests\test_memory_evolution_module_boundary.py tests\test_memory_evolution_repository.py tests\test_memory_evolution_reset.py tests\test_memory_evolution_retrieval.py tests\test_memory_evolution_idempotency.py tests\test_memory_evolution_writer_migration.py tests\test_memory_knowledge_sync_runtime_lore.py tests\test_persistent_memory_cache_invalidation.py tests\test_memory_retrieval_tools.py tests\test_rag_helper_arg_boundaries.py tests\test_db.py::test_save_memory_wraps_evolving_insert_api tests\test_db.py::test_search_memory_keyword tests\test_db.py::test_search_memory_vector -q` passed: 56 passed after Stage 1c handoff scoring fixes.
  - `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\memory_evolution\__init__.py src\kazusa_ai_chatbot\memory_evolution\identity.py src\kazusa_ai_chatbot\memory_evolution\models.py src\kazusa_ai_chatbot\memory_evolution\repository.py src\kazusa_ai_chatbot\memory_evolution\reset.py src\kazusa_ai_chatbot\db\memory_evolution.py src\kazusa_ai_chatbot\db\memory.py src\scripts\reset_memory_lore.py src\scripts\manage_memory_knowledge.py src\scripts\export_memory.py` passed.
- Reset/reseed dry-run summary:
  - `venv\Scripts\python.exe src\scripts\reset_memory_lore.py --dry-run` returned `seed_rows_loaded=104`, `seed_rows_inserted=104`, `seed_rows_updated=0`, `seed_rows_unchanged=0`, `seed_rows_deleted=0`, `legacy_rows_deleted=104`, `runtime_rows_preserved=0`, `embeddings_computed=0`, `cache_invalidated=false`, `warnings=[]`.
- Database apply summary:
  - Pre-apply export: `venv\Scripts\python.exe -m scripts.export_memory --limit 1000 --output test_artifacts\memory_before_memory_evolution_stage1b_apply.json` wrote 104 memory rows.
  - Apply reset: `venv\Scripts\python.exe src\scripts\reset_memory_lore.py --apply` returned `seed_rows_loaded=104`, `seed_rows_inserted=104`, `seed_rows_updated=0`, `seed_rows_unchanged=0`, `seed_rows_deleted=0`, `legacy_rows_deleted=104`, `runtime_rows_preserved=0`, `embeddings_computed=104`, `cache_invalidated=true`, `warnings=[]`.
  - Post-apply dry-run: `venv\Scripts\python.exe src\scripts\reset_memory_lore.py --dry-run` returned `seed_rows_loaded=104`, `seed_rows_inserted=0`, `seed_rows_updated=0`, `seed_rows_unchanged=104`, `seed_rows_deleted=0`, `legacy_rows_deleted=0`, `runtime_rows_preserved=0`, `embeddings_computed=0`, `cache_invalidated=false`, `warnings=[]`.
- Seed validation/sync summary:
  - `venv\Scripts\python.exe -m scripts.manage_memory_knowledge validate` returned `valid: 104 entries`.
  - Pre-apply `venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync` returned `{"duplicates_deleted": 0, "inserted": 104, "pruned": 0, "unchanged": 0, "updated": 0}`.
  - Post-apply `venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync` returned `{"duplicates_deleted": 0, "inserted": 0, "pruned": 0, "unchanged": 104, "updated": 0}`.
- Cache2 invalidation evidence:
  - `tests\test_persistent_memory_cache_invalidation.py` passed, proving persistent-memory dependencies use `source="memory"` and a memory invalidation event evicts matching entries.
  - `src\kazusa_ai_chatbot\memory_evolution\repository.py` emits `CacheInvalidationEvent(source="memory", ...)` from write APIs.
- Writer migration grep evidence:
  - `rg --glob "*.py" "get_db|db\.memory|\.find\(|aggregate\(|insert_one\(|update_one\(|update_many\(|delete_many\(|delete_one\(|replace_one\(|count_documents\(|get_text_embedding|pymongo|motor" src\kazusa_ai_chatbot\memory_evolution -n` returned no matches.
  - Direct `db.memory.*` writes in `src` are confined to `src\kazusa_ai_chatbot\db\memory_evolution.py`, the DB interface layer.
- Self-audit remediation evidence:
  - Public write APIs and reset now share `src\kazusa_ai_chatbot\db\memory_evolution.py`'s write guard, so reset and runtime memory writes cannot run concurrently.
  - `src\kazusa_ai_chatbot\db\memory.py::save_memory` imports `insert_memory_unit` at module scope from `memory_evolution.repository`, removing the runtime inline import while preserving the DB-interface boundary.
  - Vector persistent-memory searches avoid `$vectorSearch.filter` for deployed-index compatibility and post-filter lifecycle fields before returning results.
  - The memory-maintenance skill source path was corrected to `personalities\knowledge\memory_seed.jsonl`.
  - `git diff --check` passed after remediation.
  - `venv\Scripts\python.exe -m scripts.manage_memory_knowledge validate` returned `valid: 104 entries`.
  - Post-remediation `venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync` returned `{"duplicates_deleted": 0, "inserted": 0, "pruned": 0, "unchanged": 104, "updated": 0}`.
  - Post-remediation `venv\Scripts\python.exe src\scripts\reset_memory_lore.py --dry-run` returned `seed_rows_loaded=104`, `seed_rows_inserted=0`, `seed_rows_updated=0`, `seed_rows_unchanged=104`, `seed_rows_deleted=0`, `legacy_rows_deleted=0`, `runtime_rows_preserved=0`, `embeddings_computed=0`, `cache_invalidated=false`, `warnings=[]`.
- Quality feedback remediation evidence:
  - `save_memory` now infers `authority="reflection_promoted"` for extracted/inferred legacy source kinds unless the caller explicitly provides authority.
  - Supersede and merge active-source checks now reuse the operation `write_time`, removing the extra clock read between validation and write.
  - DB lifecycle update helpers allow only `status`, `updated_at`, and `expiry_timestamp`.
  - Reset embedding generation for changed rows now runs concurrently before sequential DB replacement.
  - Reset uses public repository helpers for normalization, equivalence, and cache invalidation.
  - Keyword memory search no longer builds vector-only filter state.
  - Added tests for `exclude_memory_unit_ids`, evidence-ref round-trip through `insert_memory_unit`, legacy extracted-source authority, and DB lifecycle update allowlisting.
  - `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\db\memory.py src\kazusa_ai_chatbot\db\memory_evolution.py src\kazusa_ai_chatbot\memory_evolution\repository.py src\kazusa_ai_chatbot\memory_evolution\reset.py tests\test_memory_evolution_repository.py tests\test_memory_evolution_retrieval.py tests\test_memory_evolution_writer_migration.py` passed.
  - `git diff --check` passed.
  - Boundary grep for MongoDB, Motor, direct collection methods, and raw embedding calls under Python files in `src\kazusa_ai_chatbot\memory_evolution` returned no matches.
  - `venv\Scripts\python.exe -m scripts.manage_memory_knowledge validate` returned `valid: 104 entries`.
  - `venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync` returned `{"duplicates_deleted": 0, "inserted": 0, "pruned": 0, "unchanged": 104, "updated": 0}`.
  - `venv\Scripts\python.exe -m scripts.reset_memory_lore --dry-run` returned `seed_rows_loaded=104`, `seed_rows_inserted=0`, `seed_rows_updated=0`, `seed_rows_unchanged=104`, `seed_rows_deleted=0`, `legacy_rows_deleted=0`, `runtime_rows_preserved=0`, `embeddings_computed=0`, `cache_invalidated=false`, `warnings=[]`.
- Stage 1c handoff remediation evidence:
  - `find_active_memory_units` now returns `(score, EvolvingMemoryDoc)` pairs so promotion logic can make merge/supersede decisions from vector similarity scores.
  - `db.memory_evolution.find_active_memory_documents` now projects `vectorSearchScore` into the result tuple and still removes embeddings from returned documents.
  - `tests\test_memory_evolution_retrieval.py` covers vector score propagation, metadata placeholder scores, and public repository score tuple preservation.
  - `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\db\memory_evolution.py src\kazusa_ai_chatbot\memory_evolution\__init__.py src\kazusa_ai_chatbot\memory_evolution\models.py src\kazusa_ai_chatbot\memory_evolution\repository.py tests\test_memory_evolution_retrieval.py` passed.
  - `git diff --check` passed.
- ID/lineage edge-case evidence:
  - `tests\test_memory_evolution_repository.py` and `tests\test_memory_evolution_idempotency.py` passed, covering insert idempotency, supersede active-only behavior, inactive-target rejection, and merge lineage behavior.
- Embedding call evidence:
  - Dry-run reset computed `embeddings_computed=0`.
  - Repository tests verify caller-supplied embeddings are rejected and embeddings are computed through the DB interface for changed writes.
- Any deviations:
  - Seed source path is `personalities/knowledge/memory_seed.jsonl` per user correction, replacing the older `knowledge/memory_seed.jsonl` references.
  - Added `src\kazusa_ai_chatbot\db\memory_evolution.py` as the DB interface boundary and `tests\test_memory_evolution_module_boundary.py` to prevent `memory_evolution` from directly using MongoDB handles.
