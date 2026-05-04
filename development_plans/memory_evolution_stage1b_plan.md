# memory evolution stage 1b plan

## Summary

- Goal: Upgrade the existing global persistent `memory` collection, search path, Cache2 dependencies, and seed tooling into an evolving active/superseded memory-unit model without using reflection output, conversation data, LLM calls, or external signals.
- Plan class: large
- Status: draft
- Mandatory skills: `memory-knowledge-maintenance`, `development-plan-writing`, `py-style`, `test-style-and-execution`, `database-data-pull`
- Overall cutover strategy: bigbang within the `memory` subsystem only. The existing global `memory` rows may be erased and re-seeded from current codebase seed files into the new evolving-unit schema.
- Highest-risk areas: destructive memory reset, persistent-memory retrieval regressions, seed sync deleting runtime lore later, vector index filter behavior, and stale Cache2 persistent-memory results.
- Acceptance criteria: memory reset/reseed works from current repository seed data only; persistent-memory search reads active rows by default; memory rows support insert/supersede/merge lineage; Cache2 has memory invalidation; no reflection, LLM, conversation-history, or external signal is accepted.

## Context

Stage 1b is database/search/seeding work only. It prepares global memory to support future lore evolution, but it does not decide what new lore should exist.

The current codebase has:

- `memory` collection used by persistent-memory RAG helpers.
- `db/memory.py` with append-only `save_memory` and `search_memory`.
- `knowledge/memory_seed.jsonl` as curated seed lane when present.
- `scripts.manage_memory_knowledge` as local seed maintenance tooling.
- Persistent-memory helper caches whose dependencies currently need memory-specific invalidation.

Stage 1b must be independent from Stage 1a and Stage 1c.

## Mandatory Skills

- `memory-knowledge-maintenance`: load before changing `memory`, seed sync, or memory maintenance scripts.
- `development-plan-writing`: load before modifying this plan.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `database-data-pull`: load before exporting current memory rows for diagnostics.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major checklist stage, the active agent must reread this entire plan before starting the next stage.
- Stage 1b must not import or call `reflection_cycle`.
- Stage 1b must not read `conversation_history`, `conversation_episode_state`, `user_memory_units`, or `character_state` for memory content.
- Stage 1b must not call any LLM.
- Stage 1b must not use web search or external data.
- Stage 1b may use only current codebase seed files and existing `memory` DB rows for dry-run reporting.
- Stage 1b cutover is bigbang for global `memory`: pre-cutover DB rows do not need preservation.
- Stage 1b must not add lore promotion, reflection candidates, or daily reflection.
- Persistent-memory retrieval must default to `status="active"`.
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
- Add `memory_evolution` package with public database-only APIs:

```python
async def reset_memory_from_seed(*, dry_run: bool) -> MemoryResetResult: ...
async def insert_memory_unit(*, document: EvolvingMemoryDoc) -> EvolvingMemoryDoc: ...
async def supersede_memory_unit(*, active_unit_id: str, replacement: EvolvingMemoryDoc) -> EvolvingMemoryDoc: ...
async def merge_memory_units(*, source_unit_ids: list[str], replacement: EvolvingMemoryDoc) -> EvolvingMemoryDoc: ...
async def find_active_memory_units(*, query: dict, limit: int) -> list[EvolvingMemoryDoc]: ...
```

- Add reset/reseed CLI:

```powershell
python src\scripts\reset_memory_lore.py --dry-run
python src\scripts\reset_memory_lore.py --apply
```

- Update `db/bootstrap.py` memory indexes.
- Update `db/memory.py` search/save helpers for evolving active rows.
- Update `rag/memory_retrieval_tools.py` so persistent-memory search defaults to active rows.
- Update Cache2 policy so persistent-memory helpers depend on `source="memory"`.
- Update memory seed tooling so current seed data can populate evolving rows and future reflection-inferred rows are not managed by seed pruning.
- Add deterministic tests for reset/reseed, active-only retrieval, supersede/merge, seed tooling, and Cache2 invalidation.

## Deferred

- Reflection-cycle code.
- Daily lore promotion.
- Reading conversation data.
- LLM calls.
- Prompt changes unless required only to preserve existing search-path wording.
- Service worker integration.
- Autonomous messages.
- User memory changes.

## Cutover Policy

| Area | Policy | Notes |
|---|---|---|
| `memory` collection | bigbang | Current rows may be deleted and re-seeded from current repo seed files. |
| Persistent-memory search path | bigbang | Search contract becomes active evolving rows only. |
| Seed tooling | bigbang | Seed lane writes evolving rows; future runtime lore is not seed-managed. |
| Cache2 dependencies | compatible | Adds memory invalidation source for helper caches. |
| Reflection | compatible | No reflection dependency or integration in Stage 1b. |

## Agent Autonomy Boundaries

- The agent may choose private helper names only inside `memory_evolution`.
- The agent must not add any reflection-specific fields beyond generic `source_kind` and evidence fields.
- The agent must not preserve old memory rows unless they are re-created from current seed data.
- The agent must not add a compatibility retrieval path that returns superseded rows in normal RAG.
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
| Retrieval | Active rows only | Prevents old and new lore conflict. |
| Runtime lore protection | Exclude from seed pruning | Needed for Stage 1c future reflection-inferred rows. |

## Data Contracts

```python
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
    evidence_refs: list[dict]
    privacy_review: dict
    confidence_note: str
    timestamp: str
    updated_at: str
    expiry_timestamp: str | None
    embedding: list[float]
```

Required statuses:

```text
active
superseded
rejected
expired
fulfilled
```

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
- `src/scripts/reset_memory_lore.py`
- `tests/test_memory_evolution_repository.py`
- `tests/test_memory_evolution_reset.py`
- `tests/test_memory_evolution_retrieval.py`
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
2. Add evolving memory schemas and repository helpers.
3. Add reset/reseed implementation and CLI.
4. Update bootstrap indexes.
5. Update memory search helpers and RAG retrieval tools.
6. Update seed tooling.
7. Run focused tests.
8. Run reset dry-run and record current-row deletion/seed-row insertion summary.
9. Run broader tests.

## Progress Checklist

- [ ] Tests for memory evolution contract added.
- [ ] Evolving memory schemas and repository helpers implemented.
- [ ] Reset/reseed CLI implemented.
- [ ] Bootstrap indexes updated.
- [ ] Active-only persistent-memory retrieval implemented.
- [ ] Memory Cache2 dependency implemented.
- [ ] Seed tooling updated.
- [ ] Focused tests pass.
- [ ] Reset dry-run evidence recorded.

## Verification

```powershell
pytest tests\test_memory_evolution_repository.py -q
pytest tests\test_memory_evolution_reset.py -q
pytest tests\test_memory_evolution_retrieval.py -q
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
- Normal persistent-memory search returns active rows only.
- Supersede and merge preserve post-cutover lineage.
- Seed sync does not manage or prune future `reflection_inferred` rows.
- Cache2 persistent-memory helper entries can be invalidated by `source="memory"`.
- No reflection, LLM, conversation-history, or external data is used.

## Execution Evidence

- Focused test results:
- Reset/reseed dry-run summary:
- Seed validation/sync summary:
- Cache2 invalidation evidence:
- Any deviations:

