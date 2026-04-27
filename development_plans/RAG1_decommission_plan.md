# RAG1 / Cache1 Big-Bang Decommission Plan

> Final destination on execution: `development_plans/rag1_cache1_decommission_plan.md`

## Context

The codebase carries two parallel RAG stacks:

- **RAG2** (source of truth, currently test-only): `nodes/persona_supervisor2_rag_supervisor2.py` exposes `call_rag_supervisor(...)`; `rag/cache2_runtime.py` + `rag/cache2_policy.py` + `rag/cache2_events.py` provide a session-LRU cache with dependency-event invalidation; eleven helper agents subclass `BaseRAGHelperAgent` in `rag/helper_agent.py`.
- **RAG1 / Cache1** (legacy, in production): `nodes/persona_supervisor2_rag.py` (1153 LOC) drives an embedding-similarity dispatcher; `rag/cache.py` (1099 LOC) implements an in-memory LRU with MongoDB write-through; collections `rag_cache_index` and `rag_metadata_index` persist entries and a per-user `rag_version` counter.

The two stacks have incompatible state shapes. RAG1 returns a structured `research_facts` dict with ~12 sub-keys plus a `research_metadata` block carrying `depth`, `cache_hit`, `depth_confidence`. RAG2 returns `{answer, known_facts[], unknown_slots[], loop_count}`. Cognition L2/L3, the facts-harvester, and the knowledge-distiller all read RAG1's shape directly.

The user has mandated:
1. RAG2 is the source of truth; consumers must be adjusted to it.
2. **No shims, no adapters, no compatibility layer.** Every legacy reference is rewritten or deleted.
3. RAG1 and Cache1 are deleted in full — code, tests, schemas, MongoDB collections.
4. The consolidator is reworked subject to project scope; recommendations welcome.
5. Cache invalidation follows `development_plans/rag_cache2_design.md` §"Consolidator → Cache 2 Integration Path".
6. Database deprecated collections are removed.
7. All checklist items mandatory; no deferred work.

Outcome: a single RAG path (RAG2), a single cache (Cache2 — session LRU), event-driven invalidation from `db_writer`, and a smaller, simpler codebase (~3,400 LOC removed; ~3 obsolete LangGraph files; 2 MongoDB collections dropped).

---

## Recommendations Made (rationale before plan)

The Plan agent surfaced six decisions that shape the refactor. The recommendations baked into this plan:

| Topic | Recommendation | Rationale |
|---|---|---|
| Knowledge-base distillation (`consolidator_knowledge.py`) | **DELETE** | Cache1 stored it with 30-day TTL; Cache2 is session-LRU only. No V2 consumer reads `knowledge_base`. Re-implementing as a persistent collection is net-new feature work outside the cutover scope. |
| Depth classification (`SHALLOW`/`DEEP` gating) | **DELETE** | V2 supervisor has no depth concept; loop-count cap (`_MAX_LOOP_COUNT`) already bounds work. Remove `rag/depth_classifier.py` + every reference. Consolidator no longer reads `metadata["depth"]`. |
| Affinity-gated external web search (`EXTERNAL_AFFINITY_SKIP_PERCENT = 40`) | **DELETE** | V1-only behaviour. V2 dispatcher selects `web_search_agent2` only when an unknown slot maps to it; no separate gate needed. |
| User-profile hydration in stage_1 | **DELETE** from supervisor; **RE-USE** in `user_profile_agent` | V2's `user_profile_agent` already returns the hydrated bundle. Consumers that previously read `state["user_profile"]` post-RAG must instead read it from `known_facts` produced by `user_profile_agent`. |
| `rag_version` counter / `increment_rag_version` | **DELETE** | Cache2 uses event-driven invalidation; the counter is dead weight. |
| RAG2 output → cognition/consolidator | **REWRITE consumers** to natively consume `{known_facts}` | Per "no shim" mandate. Each consumer extracts the slices it needs from `known_facts` filtered by `agent` field. |

---

## Target Architecture (post-refactor)

```
service.py
  └─ persona_supervisor2 (StateGraph)
       ├─ stage_0_msg_decontexualizer
       ├─ stage_1_research  →  call_rag_supervisor (RAG2)
       │                         returns {answer, known_facts, unknown_slots, loop_count}
       │                         → emitted as state["rag_result"]
       ├─ stage_2_cognition (L1/L2/L3 — read state["rag_result"])
       └─ stage_3_action    (dialog_agent — reads stage_2 outputs)

  └─ Background: call_consolidation_subgraph
       ├─ reflection / facts_harvester (read state["rag_result"])
       └─ db_writer
            ├─ MongoDB writes (steps 1, 2a, 2b, 3a, 3b, 4, 6, 7)
            └─ Emit CacheInvalidationEvent(s) to get_rag_cache2_runtime()
            (NO knowledge-base step; NO rag_version increment)
```

State key changes:
- `research_facts` (dict)  → **REMOVED**
- `research_metadata` (list[dict]) → **REMOVED**
- `rag_result` (dict) → **NEW**: `{answer: str, known_facts: list[KnownFact], unknown_slots: list[str], loop_count: int}` where `KnownFact = {slot, agent, resolved, summary, raw_result, attempts, cache: {hit, reason, key}}`

---

## Files: Delete / Modify / Create

### Delete (entire file)

| Path | Reason |
|---|---|
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag.py` | RAG1 entry + helpers |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor.py` | RAG1 evaluator |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_executors.py` | RAG1 executor agents |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_resolution.py` | RAG1 resolution helpers |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_schema.py` | RAG1 state schema |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_knowledge.py` | KB distillation deleted |
| `src/kazusa_ai_chatbot/rag/cache.py` | Cache1 implementation |
| `src/kazusa_ai_chatbot/rag/depth_classifier.py` | V1-only construct |
| `src/kazusa_ai_chatbot/db/rag_cache.py` | Cache1 DB ops |
| `tests/test_rag_cache.py` | Cache1 unit tests |
| `tests/test_rag_live_llm.py` | RAG1 integration |
| `tests/test_persona_supervisor2_rag_and_l2.py` | RAG1+L2 path |
| `tests/test_persona_supervisor2_l3_and_consolidator.py` | Reads RAG1 shape; rewrite as new file |

### Modify

| Path | Change |
|---|---|
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | Replace stage_1 import; build `rag_result` directly from `call_rag_supervisor`; remove `research_facts`/`research_metadata` from state plumbing |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` (`GlobalPersonaState`) | Remove `research_facts`, `research_metadata`; add `rag_result` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | Read `rag_result` instead of `research_facts` when forwarding to L2/L3 |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` | Rework prompt context block to consume `known_facts` (filtered/grouped by agent) |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` | Lines ~389-397, 533-540 — replace `research_facts.get(...)` with helpers that pull slices from `known_facts` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py` | Stop forwarding `research_facts`/`research_metadata`; forward `rag_result` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py` | Replace `research_facts`/`metadata` fields in `ConsolidatorState` with `rag_result` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py` | Lines ~142, 194, 216, 240, 257 — rework facts-harvester prompts to consume `known_facts` summaries grouped by agent (drop `user_image`, `input_context_results`, `external_rag_results` references; replace with structured `known_facts` block) |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py` | Remove `_get_rag_cache`, `increment_rag_version` imports; remove `_update_knowledge_base` import + Step 8 call; replace lines 624–650 with `CacheInvalidationEvent` emissions per design doc; drop `try/except PyMongoError` (Cache2 is in-memory) |
| `src/kazusa_ai_chatbot/service.py` | Remove cache1 warm-start (~line 436) and cache1 shutdown (~line 469); no Cache2 startup needed (singleton lazy-inits) |
| `src/kazusa_ai_chatbot/db/__init__.py` | Remove imports + `__all__` entries for `RagCacheIndexDoc`, `RagMetadataIndexDoc`, `clear_all_cache_for_user`, `find_cache_entries`, `get_rag_version`, `increment_rag_version`, `insert_cache_entry`, `soft_delete_cache_entries`; update docstring submodule map |
| `src/kazusa_ai_chatbot/db/schemas.py` | Lines 298–326 — delete `RagCacheIndexDoc`, `RagMetadataIndexDoc` |
| `src/kazusa_ai_chatbot/db/bootstrap.py` | Drop `rag_cache_index`/`rag_metadata_index` from `required_collections`; remove their index-creation blocks (~lines 122–141) and the vector-index entry (~line 147); add idempotent `drop_collection` for both legacy names; remove `enable_vector_index` call for `rag_cache_index` |
| `src/kazusa_ai_chatbot/rag/user_profile_agent.py` | Remove `from rag.depth_classifier import DEEP` (line 18); replace with literal value or own constant |
| `tests/test_user_profile_memories.py` | Audit; remove any `_get_rag_cache` / `increment_rag_version` references |
| `tests/test_e2e_live_llm.py` | Audit; replace cache1 assertions with `get_rag_cache2_runtime().get_stats()` |

### Create

| Path | Purpose |
|---|---|
| `scripts/drop_legacy_rag_collections.py` | One-shot production cleanup: drops `rag_cache_index` + `rag_metadata_index`. Idempotent — silent if already gone. |
| `tests/test_db_writer_cache2_invalidation.py` | Unit tests covering each `write_log` outcome → expected `CacheInvalidationEvent` emission and matching eviction |
| `tests/test_persona_supervisor2_rag2_integration.py` | Replaces `test_persona_supervisor2_rag_and_l2.py` — exercises stage_1_research with V2 supervisor end-to-end (mocked LLM) |
| `tests/test_consolidator_facts_rag2.py` | Replaces `test_persona_supervisor2_l3_and_consolidator.py` — exercises facts-harvester against new `rag_result` shape |
| `development_plans/rag1_cache1_decommission_plan.md` | Final destination of this plan after approval |

### Keep (no change)

- `src/kazusa_ai_chatbot/rag/cache2_runtime.py`, `cache2_policy.py`, `cache2_events.py`, `helper_agent.py`
- All eleven RAG2 helper agents in `src/kazusa_ai_chatbot/rag/`
- `src/kazusa_ai_chatbot/agents/user_image_retriever_agent.py` (used by `user_profile_agent`)
- `tests/test_persona_supervisor2_rag_supervisor2_live.py` — primary RAG2 integration test
- `tests/test_rag_initializer_cache2.py`, all `tests/test_<helper_agent>_*.py`

---

## Cache Invalidation Event Emission (db_writer)

Replace `persona_supervisor2_consolidator_persistence.py` lines 624–650. Emit events ONLY when the corresponding `write_log` flag is `True`. Reference: `development_plans/rag_cache2_design.md` §Domain event table (lines 172–179).

```python
from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime

# ── Step 5: Cache2 invalidation events (after persistence) ──────
runtime = get_rag_cache2_runtime()
events: list[CacheInvalidationEvent] = []

if global_user_id and (
    write_log.get('user_profile_memories')
    or write_log.get('affinity')
    or write_log.get('relationship_insight')
):
    events.append(CacheInvalidationEvent(
        source='user_profile',
        platform=state['platform'],
        platform_channel_id=state['platform_channel_id'],
        global_user_id=global_user_id,
        timestamp=timestamp,
        reason='consolidator: user_profile',
    ))

if write_log.get('character_state'):
    events.append(CacheInvalidationEvent(
        source='character_state',
        reason='consolidator: character_state',
    ))

if global_user_id and write_log.get('user_image'):
    events.append(CacheInvalidationEvent(
        source='user_image',
        platform=state['platform'],
        platform_channel_id=state['platform_channel_id'],
        global_user_id=global_user_id,
        timestamp=timestamp,
        reason='consolidator: user_image',
    ))

if write_log.get('character_image'):
    events.append(CacheInvalidationEvent(
        source='user_image',
        reason='consolidator: character_self_image',
    ))

evicted_total = 0
for event in events:
    evicted_total += await runtime.invalidate(event)

cache_invalidated = [event.source for event in events]
metadata['cache_evicted_count'] = evicted_total
```

Notes:
- No `try/except` — Cache2 invalidation is in-memory; per `py-style` rule 6, try/except is reserved for external failures.
- Drop the `AFFINITY_CACHE_NUKE_THRESHOLD` constant and its conditional `clear_all_user` branch — the wildcard scope on the `user_profile` event already invalidates every cached entry for that user.
- Drop `await increment_rag_version(...)` and `metadata['cache_invalidation_scope']` (replace with `cache_invalidated` populated above).

---

## RAG2 Wiring in `persona_supervisor2.py`

The new `stage_1_research` node:

```python
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor2 import call_rag_supervisor

async def stage_1_research(state: GlobalPersonaState) -> dict:
    '''Run RAG2 progressive supervisor and emit rag_result.'''
    context = {
        'platform': state['platform'],
        'platform_channel_id': state['platform_channel_id'],
        'channel_type': state.get('channel_type', 'group'),
        'global_user_id': state['global_user_id'],
        'user_name': state['user_name'],
        'user_profile': state.get('user_profile', {}),
        'current_timestamp': state['timestamp'],
        'channel_topic': state.get('channel_topic', ''),
        'chat_history_recent': state.get('chat_history_recent', []),
        'chat_history_wide': state.get('chat_history_wide', []),
        'reply_context': state.get('reply_context', {}),
        'indirect_speech_context': state.get('indirect_speech_context', ''),
    }
    result = await call_rag_supervisor(
        original_query=state['decontexualized_input'],
        character_name=state['character_profile'].get('name', ''),
        context=context,
    )
    return {'rag_result': result}
```

Audit `call_rag_supervisor`'s actual signature/context contract during implementation; the `context` dict above must match what RAG2 helper agents read from `state['context']` (see `rag/user_profile_agent.py`, `rag/conversation_*_agent.py` for canonical reads).

---

## Database Migration

### `db/bootstrap.py` — new section after `existing = ...`

```python
# Drop legacy RAG1 collections (idempotent — safe across deploys)
for legacy in ('rag_cache_index', 'rag_metadata_index'):
    if legacy in existing:
        await db.drop_collection(legacy)
        logger.info("Dropped legacy collection '%s'", legacy)
        existing.discard(legacy)
```

Then remove the two collections from `required_collections`, their index-creation calls (lines ~122–141), and the `enable_vector_index('rag_cache_index', ...)` call (~line 147).

### `scripts/drop_legacy_rag_collections.py`

Standalone idempotent CLI (for ops to run once on production before redeploy):

```python
'''One-shot cleanup: drop legacy RAG1 MongoDB collections.

Run before deploying the RAG2-only build to ensure no stale collections persist.
Safe to run repeatedly.
'''

import asyncio
import logging

from kazusa_ai_chatbot.db._client import close_db, get_db

logger = logging.getLogger(__name__)


async def main() -> None:
    '''Drop rag_cache_index and rag_metadata_index if present.'''
    db = await get_db()
    existing = set(await db.list_collection_names())
    for name in ('rag_cache_index', 'rag_metadata_index'):
        if name in existing:
            await db.drop_collection(name)
            logger.info("Dropped collection '%s'", name)
        else:
            logger.info("Collection '%s' not present; skipping", name)
    await close_db()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

---

## Verification

### Static greps (must all return zero matches inside `src/`)

- `_get_rag_cache`
- `RAGCache\b`
- `from kazusa_ai_chatbot.rag.cache import`
- `from kazusa_ai_chatbot.nodes.persona_supervisor2_rag\b` (note: `_supervisor2` is allowed)
- `rag_cache_index`, `rag_metadata_index`
- `RagCacheIndexDoc`, `RagMetadataIndexDoc`
- `increment_rag_version`, `get_rag_version`, `insert_cache_entry`, `find_cache_entries`, `soft_delete_cache_entries`, `clear_all_cache_for_user`
- `InputDepthClassifier`, `from kazusa_ai_chatbot.rag.depth_classifier import`
- `research_facts`, `research_metadata` (state-key names)
- `_update_knowledge_base`, `knowledge_base_results`
- `EXTERNAL_AFFINITY_SKIP_PERCENT`, `AFFINITY_CACHE_NUKE_THRESHOLD`

### Smoke tests

- `pytest tests/test_persona_supervisor2_rag_supervisor2_live.py` — must pass.
- `pytest tests/test_rag_initializer_cache2.py` — must pass.
- `pytest tests/test_db_writer_cache2_invalidation.py` (new) — must pass.
- `pytest tests/test_persona_supervisor2_rag2_integration.py` (new) — must pass.
- `pytest tests/test_consolidator_facts_rag2.py` (new) — must pass.
- Boot the service: `uvicorn kazusa_ai_chatbot.service:app --port 8000` — must start clean (no errors, no warnings about missing collections).
- Issue one `/chat` request — must produce non-empty response and emit Cache2 stats in logs.

### DB verification

After clean boot:
```
> db.getCollectionNames()
# Must NOT include 'rag_cache_index' or 'rag_metadata_index'
```

---

## Implementation Order (avoid mid-refactor breakage)

1. State-shape rewrite first (cognition + consolidator consumers updated to read `rag_result`).
2. Wire RAG2 entry into `persona_supervisor2.py` — V1 stack still on disk but unused.
3. Replace `db_writer` cache1 invalidation with Cache2 events.
4. Delete `_update_knowledge_base` call + its module.
5. Remove cache1 warm-start in `service.py`.
6. Delete RAG1 modules (5 files).
7. Delete `rag/cache.py`, `rag/depth_classifier.py`, `db/rag_cache.py`.
8. Update `db/__init__.py`, `db/schemas.py`, `db/bootstrap.py`; add `scripts/drop_legacy_rag_collections.py`.
9. Test migration: delete obsolete tests, write new ones.
10. Run all greps + smoke tests.

Each step leaves the code in a runnable state for incremental commits. Step 1 is the largest and highest-risk; steps 6–8 are mechanical deletions.

---

## Implementation Checklist (ALL MANDATORY)

### Phase A — Consumer rewrite to RAG2 shape

- [ ] Update `nodes/persona_supervisor2_schema.py` — `GlobalPersonaState`: remove `research_facts`, `research_metadata`; add `rag_result: dict`
- [ ] Update `nodes/persona_supervisor2_consolidator_schema.py` — `ConsolidatorState`: remove `research_facts`, `metadata.depth`/`cache_hit`; add `rag_result`
- [ ] Rewrite `nodes/persona_supervisor2_cognition.py` — pass `rag_result` to L2/L3
- [ ] Rewrite `nodes/persona_supervisor2_cognition_l2.py` — consume `known_facts` (apply `py-style`, `cjk-safety` if file has CJK)
- [ ] Rewrite `nodes/persona_supervisor2_cognition_l3.py` — replace `research_facts.get(...)` reads with `known_facts` projection helpers (lines ~389-397, 533-540)
- [ ] Rewrite `nodes/persona_supervisor2_consolidator.py` — forward `rag_result`, drop `research_facts`/`research_metadata`
- [ ] Rewrite `nodes/persona_supervisor2_consolidator_facts.py` — facts-harvester prompts consume structured `known_facts` (lines ~142, 194, 216, 240, 257)

### Phase B — Wire RAG2

- [ ] Edit `nodes/persona_supervisor2.py` — import `call_rag_supervisor`; replace `stage_1_research` with new wrapper that builds context and emits `rag_result`
- [ ] Verify `call_rag_supervisor` context-dict contract by reading helper agents in `rag/`

### Phase C — db_writer rework

- [ ] Edit `nodes/persona_supervisor2_consolidator_persistence.py`:
  - [ ] Remove imports: `_get_rag_cache` (line 49), `increment_rag_version` (line 24 group), `_update_knowledge_base` (line 37)
  - [ ] Replace lines 624–650 with `CacheInvalidationEvent` emission block (see plan §"Cache Invalidation Event Emission")
  - [ ] Remove `AFFINITY_CACHE_NUKE_THRESHOLD` constant (line 54) and its branch
  - [ ] Remove Step 8 (`_update_knowledge_base` call, ~lines 676–683); drop `kb_count`, `metadata['knowledge_base_entries_written']`
  - [ ] Update `metadata` dict: replace `cache_invalidation_scope` with `cache_invalidated`; add `cache_evicted_count`
  - [ ] Remove `try/except PyMongoError` around the new event emission (in-memory call)

### Phase D — Service startup/shutdown

- [ ] Edit `service.py` — remove cache1 warm-start (~line 436) and shutdown (~line 469); confirm no other Cache1 references in service module

### Phase E — Delete RAG1 modules

- [ ] Delete `nodes/persona_supervisor2_rag.py`
- [ ] Delete `nodes/persona_supervisor2_rag_supervisor.py`
- [ ] Delete `nodes/persona_supervisor2_rag_executors.py`
- [ ] Delete `nodes/persona_supervisor2_rag_resolution.py`
- [ ] Delete `nodes/persona_supervisor2_rag_schema.py`
- [ ] Delete `nodes/persona_supervisor2_consolidator_knowledge.py`
- [ ] Delete `rag/cache.py`
- [ ] Delete `rag/depth_classifier.py` (after editing `rag/user_profile_agent.py` line 18 to drop the `DEEP` import)
- [ ] Edit `rag/user_profile_agent.py` — remove `from kazusa_ai_chatbot.rag.depth_classifier import DEEP`; substitute literal or local constant

### Phase F — Database

- [ ] Edit `db/__init__.py` — remove imports + `__all__` for `RagCacheIndexDoc`, `RagMetadataIndexDoc`, `clear_all_cache_for_user`, `find_cache_entries`, `get_rag_version`, `increment_rag_version`, `insert_cache_entry`, `soft_delete_cache_entries`; update docstring submodule map (line 16)
- [ ] Edit `db/schemas.py` — delete `RagCacheIndexDoc`, `RagMetadataIndexDoc` (lines 298–326)
- [ ] Edit `db/bootstrap.py`:
  - [ ] Add idempotent drop block for `rag_cache_index`, `rag_metadata_index`
  - [ ] Remove the two collections from `required_collections`
  - [ ] Remove their index-creation calls (~lines 122–141)
  - [ ] Remove `enable_vector_index('rag_cache_index', ...)` (~line 147)
- [ ] Delete `db/rag_cache.py`
- [ ] Create `scripts/drop_legacy_rag_collections.py` per template above

### Phase G — Tests

- [ ] Delete `tests/test_rag_cache.py`
- [ ] Delete `tests/test_rag_live_llm.py`
- [ ] Delete `tests/test_persona_supervisor2_rag_and_l2.py`
- [ ] Delete `tests/test_persona_supervisor2_l3_and_consolidator.py`
- [ ] Audit `tests/test_user_profile_memories.py`; remove cache1 references
- [ ] Audit `tests/test_e2e_live_llm.py`; replace cache1 assertions with Cache2 stats
- [ ] Create `tests/test_db_writer_cache2_invalidation.py` — one test per row of the event-emission table
- [ ] Create `tests/test_persona_supervisor2_rag2_integration.py` — stage_1_research end-to-end with mocked LLM
- [ ] Create `tests/test_consolidator_facts_rag2.py` — facts-harvester against new `rag_result` shape

### Phase H — Verification

- [ ] Run all static greps from §"Verification — Static greps"; confirm zero matches in `src/`
- [ ] `pytest tests/test_persona_supervisor2_rag_supervisor2_live.py` — passes
- [ ] `pytest tests/test_rag_initializer_cache2.py` — passes
- [ ] `pytest tests/test_db_writer_cache2_invalidation.py` — passes
- [ ] `pytest tests/test_persona_supervisor2_rag2_integration.py` — passes
- [ ] `pytest tests/test_consolidator_facts_rag2.py` — passes
- [ ] Service boot: `uvicorn kazusa_ai_chatbot.service:app` — clean startup
- [ ] Live `/chat` smoke: one request returns non-empty response, logs show Cache2 hit/miss stats
- [ ] Mongo: `db.getCollectionNames()` does NOT include `rag_cache_index` or `rag_metadata_index`
- [ ] Move this plan to `development_plans/rag1_cache1_decommission_plan.md`

### Phase I — Documentation

- [ ] Update README.md if it references RAG1, Cache1, `rag_cache_index`, depth classifier, or knowledge_base
- [ ] Update `development_plans/rag_cache2_design.md` if any decision in this refactor diverges from the design (e.g., affinity-event always invalidates `user_profile`)

---

## Critical files referenced

- [src/kazusa_ai_chatbot/nodes/persona_supervisor2.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2.py)
- [src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py)
- [src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py)
- [src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py)
- [src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py)
- [src/kazusa_ai_chatbot/rag/cache2_runtime.py](src/kazusa_ai_chatbot/rag/cache2_runtime.py)
- [src/kazusa_ai_chatbot/rag/cache2_events.py](src/kazusa_ai_chatbot/rag/cache2_events.py)
- [src/kazusa_ai_chatbot/db/__init__.py](src/kazusa_ai_chatbot/db/__init__.py)
- [src/kazusa_ai_chatbot/db/schemas.py](src/kazusa_ai_chatbot/db/schemas.py)
- [src/kazusa_ai_chatbot/db/bootstrap.py](src/kazusa_ai_chatbot/db/bootstrap.py)
- [src/kazusa_ai_chatbot/service.py](src/kazusa_ai_chatbot/service.py)
- [development_plans/rag_cache2_design.md](development_plans/rag_cache2_design.md)
