# Implementation Plan: 5-Phase Personality System

**Status**: READY FOR EXECUTION
**Execution Model**: Single sprint (Claude does all changes)
**Scope**: Unified RAG-Consolidator system with intelligent memory management
**Target**: Deploy and validate all changes, then run comprehensive E2E tests
**Breaking Changes**: Allowed (refactored architecture)

---

## Executive Summary

This plan details **what files to modify, in what order**, to implement the 5-phase personality system with:
- Semantic cache layer (60-70% hit rate)
- Depth-aware intelligent routing (40% fewer dispatcher calls)
- Consolidator evaluator loop (95%+ contradiction detection)
- Atomic database writes with strategic cache invalidation
- Scheduled task infrastructure

**Implementation approach**:
1. Modify files in dependency order
2. Deploy all changes in one atomic merge
3. Run comprehensive E2E tests
4. Measure before/after metrics

---

## Part 1: Implementation Sequence (Dependency Order)

### Stage 1: Foundation Modules (No dependencies, can implement in parallel) ✅ COMPLETE

**Stage 1 Goal**: Implement and self-test all foundation modules in isolation. No module is plugged into the main workflow yet. Each file must include an `async def test_main()` function and `if __name__ == "__main__": asyncio.run(test_main())` block at the bottom — following the exact pattern used in `persona_supervisor2_rag.py` and other existing modules — so each file can be run standalone to inspect output values before integration.

New RAG-related files live under `src/kazusa_ai_chatbot/rag/` (new subfolder).

**Stage 1 Status (2026-04-19)**:
- ✅ `src/kazusa_ai_chatbot/rag/__init__.py` created
- ✅ `src/kazusa_ai_chatbot/rag/cache.py` — RAGCache with in-memory LRU + MongoDB write-through
- ✅ `src/kazusa_ai_chatbot/rag/depth_classifier.py` — SHALLOW/DEEP via embedding + LLM fallback
- ✅ `src/kazusa_ai_chatbot/scheduler.py` extended — `future_promise` handler registered
- ✅ `src/kazusa_ai_chatbot/db.py` — `ScheduledEventDoc` updated (adds `future_promise`,
  `cancelled`, `cancelled_at`)
- ✅ Unit tests added under `tests/` — 34 tests, all passing:
    - `tests/test_rag_cache.py` (16 tests)
    - `tests/test_depth_classifier.py` (10 tests)
    - `tests/test_scheduler_future_promise.py` (8 tests)
- ✅ Full test suite green: 138 passed, 18 deselected

---

#### 1a. Create `kazusa_ai_chatbot/rag/cache.py` (NEW FILE) ✅ IMPLEMENTED
**Purpose**: RAGCache class for semantic caching with crash resilience
**Changes** in this file:

**Core Methods**:
- [x] `__init__()` - Initialize in-memory LRU cache (no external dependencies)
  - In-memory store: Typed dict with LRU eviction (built-in Python)
  - No Redis/external service required
  - Load persisted entries from MongoDB on startup
- [x] `async retrieve_if_similar(embedding, cache_type, threshold)` - Vector similarity lookup
  - Check in-memory cache first (fast path)
  - Fallback to MongoDB if not in memory (recovery path)
- [x] `async store(embedding, results, cache_type, ttl_seconds, metadata)` - Cache storage
  - Store in-memory immediately
  - Persist to MongoDB asynchronously (for recovery)
- [x] `async invalidate_pattern(cache_type, global_user_id, trigger)` - Selective invalidation
  - Remove from in-memory cache
  - Mark as deleted in MongoDB (soft delete) for audit trail
- [x] `async clear_all_user(global_user_id)` - Full user cache clear
  - Clear from in-memory
  - Mark deleted in MongoDB
- [x] `get_stats()` - Cache stats (hits, misses, size, memory)
  - Track in-memory performance metrics
  - Return hit rate, eviction count, current size

**Embedding Similarity** (Cosine Distance):
- [x] Compute cosine similarity between embeddings
- [x] Threshold-based matching (default 0.82)
- [x] Pre-computed norms for performance

**Crash Resilience** (CRITICAL - Addresses failure mode):
- [x] On startup: Load persisted cache entries from `rag_cache_index` MongoDB collection
  - Query entries with `ttl_expires_at > now` (skip expired)
  - Reconstruct in-memory LRU from persisted entries
  - Result: Cache warm-starts after crash, no cold-start latency penalty
- [x] During operation: Async write to MongoDB after each store
  - If write fails, log error but continue (in-memory cache still works)
  - Next restart will replay from MongoDB (eventual consistency)
- [x] Graceful shutdown: Flush all in-memory entries to MongoDB
  - Stop accepting new cache requests
  - Complete pending writes before shutdown
  - Next startup: Full cache populated and ready
  - Note: every `store()` is write-through, so `shutdown()` only logs stats —
    the DB is already up-to-date.

**TTL Management**:
- [x] In-memory: Check expiry on lookup (lazy deletion)
- [x] MongoDB: TTL index handles automatic cleanup (expireAfterSeconds=0) — index will be
  created in Stage 2b (`db/bootstrap.py`). For Stage 1, documents expire only on
  in-memory lookup; MongoDB-side TTL starts working once the index is added.
- [ ] Periodic: Optional background cleanup task (future)

**Dependencies**: None (uses built-in Python LRU, MongoDB already in project - NO NEW EXTERNAL SERVICES)
**Lines of code**: ~350-450
**test_main()**: Instantiate `RAGCache`, store a dummy embedding + result, retrieve by similar embedding, print hit/miss result and stats. Run standalone to verify cache round-trip works.

---

#### 1b. Create `kazusa_ai_chatbot/rag/depth_classifier.py` (NEW FILE) ✅ IMPLEMENTED
**Purpose**: Classify query into two depths (SHALLOW/DEEP) via embedding similarity, with LLM fallback
**Rationale**: Two layers map naturally to the two storage layers — SHALLOW serves from cache + user_rag only; DEEP triggers full database search across all dispatchers. Bot is multilingual (Chinese/English minimum), so keyword sets are enumerated in both languages and matched via text embedding + cosine similarity. LLM fallback handles ambiguous inputs that don't match either keyword centroid confidently.

**Changes** in this file:
- [x] Two enumerated keyword sets (Chinese + English):
  - `SHALLOW_KEYWORDS`: greetings, simple preferences, yes/no facts
    - EN: `["what", "who", "do you like", "your name", "favorite", ...]`
    - ZH: `["喜欢", "叫什么", "你是", "好不好", "什么颜色", ...]`
  - `DEEP_KEYWORDS`: emotional history, temporal references, contradictions, complex reasoning
    - EN: `["always", "why do you", "remember when", "last time", "you said", ...]`
    - ZH: `["你为什么总是", "以前", "你说过", "上次", "为什么", "记得吗", ...]`
  - Keyword embeddings pre-computed at module load time (one-time cost, both sets merged per depth)
- [x] `InputDepthClassifier` class:
  - `async classify(input, user_topic, affinity)` - Main classification method
  - **Fast path**: Compute embedding of input, cosine similarity against SHALLOW and DEEP centroids
    - If `sim(SHALLOW) > 0.75` and `sim(SHALLOW) > sim(DEEP)` → return `SHALLOW`
    - If `sim(DEEP) > 0.75` and `sim(DEEP) > sim(SHALLOW)` → return `DEEP`
  - **Fallback path**: If neither centroid scores above 0.75, call LLM:
    ```python
    _depth_classifier_llm = get_llm(temperature=0.0, top_p=1.0)
    response = await _depth_classifier_llm.ainvoke([system_prompt, user_prompt])
    result = parse_llm_json_output(response.content)
    ```
  - **Affinity override**: if `affinity < 400`, always return `DEEP` regardless of classification
  - **Final fallback**: if LLM output is unparseable, default to `DEEP` (safer — better to over-retrieve than miss context)
- [x] LLM system prompt (fallback path) must specify:
  - **Input format**:
    ```
    Input JSON fields:
    - user_input (string): the user's message, may be in Chinese or English
    - user_topic (string): topic category derived from the conversation
    - affinity (integer): relationship score 0–1000 between user and bot
    ```
  - **Output format** (strict JSON, no extra keys):
    ```json
    {
      "depth": "SHALLOW or DEEP",
      "reasoning": "one sentence explaining why"
    }
    ```
  - **Classification rules** explained to LLM:
    - `SHALLOW`: input is a simple factual question, greeting, or preference check that requires no deep memory retrieval — cache or basic user profile is sufficient
    - `DEEP`: input references past events, emotional context, asks "why" about behaviour, or involves temporal reasoning — requires full memory search
    - If `affinity` in the input is below 400, always output `DEEP`
    - Input language may be Chinese or English — classify based on meaning, not language
- [x] Output structure: `{depth, trigger_dispatchers, confidence}`
  - `SHALLOW` → `trigger_dispatchers: ["user_rag"]`
  - `DEEP` → `trigger_dispatchers: ["user_rag", "internal_rag", "external_rag"]`

**Dependencies**: `utils.get_llm`, `utils.parse_llm_json_output`, `config.LLM_*` (all already in project)
**Lines of code**: ~200-250
**test_main()**: Run `classify()` against 4 sample inputs — one clear SHALLOW (EN), one clear SHALLOW (ZH), one clear DEEP (EN), one ambiguous (triggers LLM fallback). Print depth + trigger_dispatchers for each.

---

#### 1c. Update `kazusa_ai_chatbot/scheduler.py` (EXISTING FILE - EXTEND) ✅ IMPLEMENTED
**Purpose**: Extend the existing scheduler to support `future_promise` event type from the consolidator
**Current state**: `scheduler.py` already exists and is well-structured. `ScheduledEventDoc` TypedDict is defined in `db.py`. The `scheduled_events` MongoDB collection already exists with `schedule_event`, `cancel_event`, `load_pending_events`, `shutdown` functions. Only `followup_message` event type is currently implemented.

**What already works (do not touch)**:
- `schedule_event(event)` — persists + registers asyncio task
- `cancel_event(event_id)` — cancels and marks as "cancelled" in DB
- `load_pending_events()` — crash recovery on startup
- `shutdown()` — graceful cleanup
- Handler registry (`register_handler`, `_handlers`)

**Changes needed**:

**1. Fix `ScheduledEventDoc` in `db.py`** — add missing `"cancelled"` to status comment (code already sets it, TypedDict comment omits it):
```python
class ScheduledEventDoc(TypedDict, total=False):
    event_id: str               # UUID4
    event_type: str             # "followup_message" | "future_promise" | ...
    target_platform: str        # Platform to deliver on
    target_channel_id: str      # Channel/group to deliver to
    target_global_user_id: str  # User the event relates to
    payload: dict               # Event-specific data — schema varies by event_type (see below)
    scheduled_at: str           # ISO-8601 UTC when to fire
    created_at: str             # ISO-8601 UTC when the event was created
    status: str                 # "pending" | "running" | "completed" | "failed" | "cancelled"
    cancelled_at: str           # ISO-8601 UTC — set when status becomes "cancelled"
```

**2. Define `payload` sub-schema per event_type**:

`followup_message` payload (existing):
```python
{
    "message": str,           # Text to send
    "platform": str,          # Target platform
    "channel_id": str,        # Target channel
}
```

`future_promise` payload (NEW — from consolidator future_promises):
```python
{
    "promise_text": str,       # What was promised, verbatim from extraction
    "memory_id": str,          # ID of the MemoryDoc saved for this promise (memory_type="promise")
    "original_input": str,     # The user message that triggered the promise
    "context_summary": str,    # Brief context so bot can recall why the promise was made
}
```

**3. Register `future_promise` handler in `scheduler.py`**:
- [x] Add `async def _handle_future_promise(event: ScheduledEventDoc)` stub
  - Logs the firing (full implementation deferred to Stage 4 consolidator refactor)
  - Marks promise `MemoryDoc` status as `"fulfilled"` in DB
- [x] Call `register_handler("future_promise", _handle_future_promise)`
- [x] `cancel_event()` now also stamps `cancelled_at` (ISO-8601) alongside setting
  status to `cancelled`.

**Collection**: Reuses existing `scheduled_events` — no new collection needed.

**Indices** (verify exist, create if missing):
- `{ "status": 1, "scheduled_at": 1 }` — for `load_pending_events` query
- `{ "target_global_user_id": 1 }` — for per-user event lookup
- `{ "event_id": 1 }` — unique, for update/cancel operations

**Dependencies**: `db.ScheduledEventDoc`, `db.get_db` (already imported)
**Lines of code**: ~30-50 lines added to existing file
**test_main()**: Schedule one `followup_message` and one `future_promise` event (future-dated), call `load_pending_events()`, print both docs from DB, cancel one, verify status updated to `"cancelled"` in DB.

---

### Stage 2: Database Layer (After Stage 1 foundation)

#### 2a. Restructure `kazusa_ai_chatbot/db.py` → `kazusa_ai_chatbot/db/` (SPLIT INTO SUBMODULES)
**Purpose**: `db.py` is too large. Split by responsibility into a `db/` package. Backward compatibility is preserved — all existing `from kazusa_ai_chatbot.db import ...` imports continue to work via `__init__.py` re-exports.

**New folder structure**:
```
src/kazusa_ai_chatbot/db/
    __init__.py         ← re-exports everything (backward compat)
    _client.py          ← connection + embedding (shared by all submodules)
    schemas.py          ← all TypedDict document schemas
    bootstrap.py        ← db_bootstrap() startup logic
    conversation.py     ← conversation_history collection
    users.py            ← user_profiles collection (identity + profile + facts + affinity)
    character.py        ← character_state collection
    memory.py           ← memory collection
    rag_cache.py        ← NEW: rag_cache_index + rag_metadata_index collections
```

**Submodule breakdown** (what moves where from current `db.py`):

`db/_client.py` — Connection + shared utilities (imported by every other submodule):
- `_get_embed_client()`, `get_text_embedding()`
- `get_db()`, `close_db()`
- `enable_vector_index()`

`db/schemas.py` — All TypedDict document schemas (no logic):
- `AttachmentDoc`
- `ConversationMessageDoc`, `PlatformAccountDoc`
- `UserProfileDoc`
- `CharacterProfileDoc`
- `MemoryDoc`, `build_memory_doc()`
- `ScheduledEventDoc`

`db/bootstrap.py` — Startup and index creation:
- `db_bootstrap()` — creates collections, seeds character_state, creates all indices

`db/conversation.py` — `conversation_history` collection:
- `get_conversation_history()`
- `search_conversation_history()`
- `save_conversation()`

`db/users.py` — `user_profiles` collection (identity resolution, profile, facts, affinity):
- `resolve_global_user_id()`, `link_platform_account()`, `add_suspected_alias()`
- `get_user_profile()`, `create_user_profile()`
- `get_user_facts()`, `upsert_user_facts()`, `overwrite_user_facts()`
- `get_affinity()`, `update_affinity()`, `update_last_relationship_insight()`
- `enable_user_facts_vector_index()`, `search_users_by_facts()`

`db/character.py` — `character_state` collection:
- `get_character_profile()`, `save_character_profile()`
- `get_character_state()`, `upsert_character_state()`

`db/memory.py` — `memory` collection:
- `enable_memory_vector_index()`
- `save_memory()`, `search_memory()`

`db/rag_cache.py` — NEW: `rag_cache_index` + `rag_metadata_index` collections:

*Collections and schemas*:
```python
# rag_cache_index document
{
    "cache_id": str,             # UUID4
    "cache_type": str,           # "user_facts" | "internal_memory"
    "global_user_id": str,       # Owner (for scoped invalidation)
    "embedding": list[float],    # Query embedding that produced these results
    "results": dict,             # Cached RAG results payload
    "ttl_expires_at": datetime,  # TTL — MongoDB TTL index auto-deletes after this
    "created_at": str,           # ISO-8601 UTC
    "deleted": bool,             # Soft-delete flag (set by invalidate before TTL fires)
}

# rag_metadata_index document (one doc per global_user_id)
{
    "global_user_id": str,       # UUID4 — unique key
    "rag_version": int,          # Incremented on every successful DB write (cache bust signal)
    "last_rag_run": str,         # ISO-8601 UTC of last RAG execution
}
```

*Indices on `rag_cache_index`*:
- `{ "ttl_expires_at": 1 }` with `expireAfterSeconds=0` — auto-deletion
- `{ "cache_type": 1, "global_user_id": 1, "deleted": 1 }` — scoped invalidation queries
- Vector search index on `embedding` (cosine, same dim as other collections)

*Indices on `rag_metadata_index`*:
- `{ "global_user_id": 1 }` unique

*New functions in `db/rag_cache.py`*:
- [ ] `async insert_cache_entry(cache_type, global_user_id, embedding, results, ttl_seconds)` → `str` (cache_id)
- [ ] `async find_cache_entries(cache_type, global_user_id)` → `list[dict]` (non-expired, non-deleted, with embeddings)
- [ ] `async soft_delete_cache_entries(cache_type, global_user_id)` — sets `deleted=True`
- [ ] `async clear_all_cache_for_user(global_user_id)` — soft-deletes all cache_types for user
- [ ] `async get_rag_version(global_user_id)` → `int` (0 if not found)
- [ ] `async increment_rag_version(global_user_id)` — upserts, increments by 1

`db/__init__.py` — Re-export all public symbols for backward compatibility:
```python
from kazusa_ai_chatbot.db._client import get_db, close_db, get_text_embedding, enable_vector_index
from kazusa_ai_chatbot.db.schemas import (
    AttachmentDoc, ConversationMessageDoc, PlatformAccountDoc,
    UserProfileDoc, CharacterProfileDoc, MemoryDoc, ScheduledEventDoc, build_memory_doc
)
from kazusa_ai_chatbot.db.bootstrap import db_bootstrap
from kazusa_ai_chatbot.db.conversation import get_conversation_history, search_conversation_history, save_conversation
from kazusa_ai_chatbot.db.users import (
    resolve_global_user_id, link_platform_account, add_suspected_alias,
    get_user_profile, create_user_profile, get_user_facts, upsert_user_facts,
    overwrite_user_facts, get_affinity, update_affinity,
    update_last_relationship_insight, enable_user_facts_vector_index, search_users_by_facts
)
from kazusa_ai_chatbot.db.character import (
    get_character_profile, save_character_profile, get_character_state, upsert_character_state
)
from kazusa_ai_chatbot.db.memory import enable_memory_vector_index, save_memory, search_memory
from kazusa_ai_chatbot.db.rag_cache import (
    insert_cache_entry, find_cache_entries, soft_delete_cache_entries,
    clear_all_cache_for_user, get_rag_version, increment_rag_version
)
```

**Dependencies**: None (all internal)
**Lines of code**: ~100 lines reorganised + ~150 new lines in `rag_cache.py`
**No test_main()** (covered by integration tests and used by cache.py test_main)

---

#### 2b. Update `db/bootstrap.py` — add new collections to startup
- [ ] Add `rag_cache_index` and `rag_metadata_index` to `db_bootstrap()` required collections list
- [ ] Add TTL + vector indices for `rag_cache_index` to bootstrap
- [ ] Add unique index for `rag_metadata_index.global_user_id` to bootstrap

---

### Stage 3: RAG Layer Updates (After Stage 2)

#### 3a. Update `kazusa_ai_chatbot/nodes/persona_supervisor2_rag.py` (EXISTING FILE - MAJOR REFACTOR)
**Purpose**: Add cache, depth classification, early-exit logic
**Current state**: 3 parallel dispatchers, no caching
**Changes**:
- [ ] **Phase 0: Input Analysis** (NEW)
  - Compute embedding (768-dim) from `decontexualized_input`
  - Add embedding computation before dispatchers
  - Initialize metadata bundle: `{embedding, entities, temporal_markers, depth_hint}`

- [ ] **Phase 1: Cache Check** (NEW)
  - Import RAGCache, integrate before dispatchers
  - Check similarity against cached embeddings
  - If hit: Return cached results + `cache_hit=True`
  - If miss: Proceed to Phase 2a

- [ ] **Depth Classification** (NEW)
  - Import InputDepthClassifier
  - Add classification before dispatcher conditional edges
  - Determine which dispatchers to trigger based on depth

- [ ] **Conditional Dispatch (Early Exit)** (MODIFY)
  - Update dispatcher nodes to return confidence scores
  - Implement early-exit logic:
    - SHALLOW + user_rag confidence > 0.90 → Skip internal_rag
    - MEDIUM + internal_rag confidence > 0.85 → Skip external_rag
  - Update graph edges to support conditional routing

- [ ] **Cache Storage** (NEW)
  - After RAG completes, store results in cache
  - Store before returning (if not cache hit)
  - TTL: user_facts=30min, internal_memory=15min

- [ ] **Metadata Propagation** (NEW)
  - Thread metadata through all phases
  - Accumulate: cache_hit, rag_sources_used, confidence_scores, response_confidence
  - Return with final state

**Breaking Changes**:
- [ ] RAGState schema includes new fields (embedding, metadata, confidence scores)
- [ ] Return signature changes (now includes cache_hit, metadata)
- [ ] Consolidator receives updated state shape

**Dependencies**: Stage 1 (cache, depth_classifier), Stage 2 (DB functions)
**Lines of code**: ~600-800 (refactor existing + add new phases)
**Test files created during testing**

---

### Stage 4: Consolidator Layer (After Stage 3)

#### 4a. Update `kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py` (EXISTING FILE - MAJOR REFACTOR)
**Purpose**: Add evaluator loop, metadata propagation, consistency validation
**Current state**: 3 parallel jobs, facts_harvester writes directly
**Changes**:
- [ ] **facts_harvester Enhancement** (MODIFY)
  - Include `research_facts` in LLM context (NEW)
  - Include `metadata` (cache_hit, depth, confidence) in message (NEW)
  - Instruct LLM to validate facts don't contradict research_facts
  - Same extraction logic, but with additional context

- [ ] **fact_harvester_evaluator Enhancement** (MODIFY - currently basic, expand)
  - Add contradiction detection against research_facts (NEW)
  - Implement consistency validation matrix:
    - New fact vs research_facts (semantic similarity)
    - New fact vs existing memory (deduplication)
    - Promise format validation (due_time ISO 8601, action grammar)
    - Identity anchoring (whose fact is this?)
  - Clear feedback messages for each failure type
  - Enforce max 3 retries (already exists, keep)

- [ ] **Metadata Propagation** (NEW)
  - Initialize metadata in initial_state from RAG
  - Accumulate through each node:
    - After global_state_updater: Add none (state update)
    - After relationship_recorder: Add none (relationship update)
    - After facts_harvester: Add extraction attempt count
    - After evaluator: Add evaluator_passes, contradiction_flags
    - After db_writer: Add write_success, cache_invalidation_scope
  - Return with final metadata

- [ ] **db_writer: Atomic Writes + Cache Invalidation** (ENHANCE)
  - Wrap all writes in MongoDB transaction (NEW)
  - Write sequence:
    1. upsert_character_state(mood, vibe, reflection)
    2. upsert_user_facts(diary_entry)
    3. update_last_relationship_insight(insight)
    4. Loop: save_memory() for each new_fact
    5. Loop: save_memory() for each future_promise
    6. update_affinity(affinity_delta)
    7. COMMIT transaction

  - AFTER commit succeeds, invalidate cache (NEW):
    - If new_facts written: `await rag_cache.invalidate_pattern("user_facts", user_id)`
    - If |affinity_delta| > 50: `await rag_cache.clear_all_user(user_id)`
    - Update RAG version: `await increment_rag_version(user_id)`

  - Handle transaction failure (rollback automat, log error)

- [ ] **Promise Scheduling Integration** (NEW)
  - Extract future_promises with due_time
  - For each promise: `await task_scheduler.schedule_task("promise_fulfillment_check", user_id, due_time, promise_metadata)`
  - Task execution will happen asynchronously

**Breaking Changes**:
- [ ] ConsolidatorState schema changes (includes metadata propagation)
- [ ] Return signature changes (includes metadata)
- [ ] Promise fulfillment now asynchronous via scheduler (not immediate)

**Dependencies**: Stage 1 (cache, scheduler), Stage 2 (DB functions), Stage 3 (RAG metadata)
**Lines of code**: ~400-600 (refactor existing + add enhancements)
**Test files created during testing**

---

### Stage 5: Integration & Configuration (After Stage 4)

#### 5a. Update `kazusa_ai_chatbot/config.py` (EXISTING FILE - ADD CONFIG)
**Purpose**: Add configuration for new features
**Changes** (ADD, don't remove existing):
- [ ] Cache configuration:
  - `RAG_CACHE_BACKEND = "redis"  # or "memory"`
  - `RAG_CACHE_USER_FACTS_TTL = 1800  # 30 minutes`
  - `RAG_CACHE_INTERNAL_MEMORY_TTL = 900  # 15 minutes`
  - `RAG_CACHE_SIMILARITY_THRESHOLD = 0.82`
  - `RAG_CACHE_MAX_SIZE = 100_000`

- [ ] Depth classifier configuration:
  - `DEPTH_CLASSIFIER_USE_LIGHT_LLM = False  # Only heuristics for speed`
  - `DEPTH_CLASSIFIER_THRESHOLDS = {...}`

- [ ] Consolidator configuration:
  - `FACT_HARVESTER_MAX_RETRIES = 3`
  - `EVALUATOR_CONSISTENCY_CHECK = True`

- [ ] Scheduler configuration:
  - `SCHEDULED_TASKS_ENABLED = True`
  - `OFFLINE_CONSOLIDATION_CRON = "0 2 * * *"  # Future`
  - `PERSONALITY_ANALYTICS_CRON = "0 8 * * 0"  # Future`

**Dependencies**: None (constants only)
**Lines of code**: ~50-100

---

#### 5b. Update `kazusa_ai_chatbot/main.py` or app initialization (EXISTING FILE - STARTUP)
**Purpose**: Initialize cache and scheduler on app startup
**Changes**:
- [ ] Import RAGCache, TaskScheduler
- [ ] In startup hook:
  ```python
  app.state.rag_cache = RAGCache(backend=config.RAG_CACHE_BACKEND)
  app.state.task_scheduler = TaskScheduler(config=config.SCHEDULED_TASKS)
  await app.state.task_scheduler.start()
  ```
- [ ] In shutdown hook:
  ```python
  await app.state.task_scheduler.stop()
  ```

**Dependencies**: Stage 1 (cache, scheduler), Stage 5a (config)
**Lines of code**: ~20-50

---

## Part 2: File Modification Summary

### Files to CREATE (NEW)
1. `kazusa_ai_chatbot/cache.py` (300-400 lines)
2. `kazusa_ai_chatbot/depth_classifier.py` (200-300 lines)
3. `migrations/001_add_cache_metadata_collections.py` (100-150 lines)

### Files to MODIFY (EXISTING)
1. `kazusa_ai_chatbot/scheduler.py` (200-300 lines added/refactored)
2. `kazusa_ai_chatbot/db.py` (400-500 lines added)
3. `kazusa_ai_chatbot/nodes/persona_supervisor2_rag.py` (600-800 lines refactored)
4. `kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py` (400-600 lines refactored)
5. `kazusa_ai_chatbot/config.py` (50-100 lines added)
6. `kazusa_ai_chatbot/main.py` (20-50 lines added)

### Total Changes
- ~2,300-2,700 lines of new code
- 3 new files
- 6 existing files modified
- Database schema changes (TTL indices, new collections)

---

## Part 3: Breaking Changes Impact

**What breaks** (intentional, handled in tests):
- RAGState schema changes (new fields: embedding, metadata, confidence_scores)
- ConsolidatorState schema changes (metadata propagation)
- Return signatures change (now include cache_hit, metadata)
- Promise fulfillment becomes asynchronous (via scheduler)

**What does NOT break**:
- Existing RAG/Consolidator interface (still async functions)
- Database queries (backward compatible, old queries still work)
- Configuration loading (additive changes only)
- User-facing APIs (internal refactor only)

**Compatibility approach**:
- Don't need to maintain old interface
- After deployment, old code paths no longer exist
- Tests validate "current system works end-to-end after refactor"

---

## Part 4: Execution Steps (Order Matters)

### Step 1: Create Foundation Modules
```
1. Create cache.py (no dependencies)
2. Create depth_classifier.py (no dependencies)
3. Refactor scheduler.py (extract from consolidator)
```
**Validation**: Module imports work, unit tests pass

### Step 2: Database Updates
```
4. Add functions to db.py (import cache/scheduler)
5. Run migration script (create collections)
```
**Validation**: Collections exist, indices created, no data loss

### Step 3: RAG Refactor
```
6. Update persona_supervisor2_rag.py (use cache, depth_classifier, metadata)
```
**Validation**: RAG still produces results, cache gets populated

### Step 4: Consolidator Refactor
```
7. Update persona_supervisor2_consolidator.py (evaluator loop, metadata, cache invalidation)
```
**Validation**: Facts still extracted, evaluator catches contradictions, cache invalidated appropriately

### Step 5: Integration
```
8. Update config.py (add configuration)
9. Update main.py (initialize cache and scheduler)
```
**Validation**: App starts, cache and scheduler initialized

### Step 6: Deploy & Test
```
10. Merge all changes to main branch
11. Run comprehensive E2E test suite (see TEST_PLAN.md)
12. Measure before/after metrics
13. Validate no data corruption
```

---

## Part 5: Database Migration Details

### Automatic Schema Creation (if not exists)
```python
# In db.py initialization, check and create if needed:

# 1. rag_cache_index collection
if "rag_cache_index" not in db.list_collection_names():
    await db.create_collection("rag_cache_index")
    await db.rag_cache_index.create_index([("embedding", "2dsphere")])
    await db.rag_cache_index.create_index([("ttl_expires_at", 1)], expireAfterSeconds=0)
    await db.rag_cache_index.create_index([("cache_type", 1)])

# 2. rag_metadata_index collection
if "rag_metadata_index" not in db.list_collection_names():
    await db.create_collection("rag_metadata_index")
    await db.rag_metadata_index.create_index([("global_user_id", 1)])
    await db.rag_metadata_index.create_index([("rag_version", 1)])

# 3. Verify vector indices on existing collections
# Run MongoDB commands to verify/create
```

### Rollback Plan
- Collections can be dropped (temporary cache, non-critical)
- rag_cache_index: Safe to drop (only cache, repopulated)
- rag_metadata_index: Safe to drop (only metadata)
- No data loss risk

---

## Part 6: Configuration Points

### Feature Flags (in config.py)
```python
# Disable features if needed:
USE_RAG_CACHE = True  # Set False to bypass cache
USE_DEPTH_CLASSIFIER = True  # Set False to use depth=MEDIUM always
EVALUATOR_CONSISTENCY_CHECK = True  # Set False to skip evaluator
USE_ATOMIC_WRITES = True  # Set False for old behavior
```

### Performance Tuning (in config.py)
```python
# Adjustable thresholds:
RAG_CACHE_SIMILARITY_THRESHOLD = 0.82  # Higher = fewer hits
RAG_CACHE_USER_FACTS_TTL = 1800  # Seconds, adjust for memory usage
EVALUATOR_RETRY_MAX = 3  # If too strict, increase
```

---

## Part 7: Validation Checklist

### Pre-Merge Checklist
- [ ] All 3 new files compile without errors
- [ ] All 6 modified files compile without errors
- [ ] Database functions callable (no import errors)
- [ ] Config values have defaults (no KeyError on startup)
- [ ] App starts without crashing (main.py initialization works)

### Post-Merge Checklist (See TEST_PLAN.md)
- [ ] Unit tests: 30+ tests passing
- [ ] Integration tests: 12 tests passing
- [ ] E2E tests: Full conversation flow works
- [ ] Performance tests: Benchmarks met
- [ ] Load tests: 10 concurrent conversations stable
- [ ] Before/after metrics: Show improvement

---

## Part 8: Rollback Strategy

If something breaks post-deployment:

1. **Quick disable**: Set feature flags to False in config
2. **Git revert**: `git revert <commit>` back to previous stable state
3. **Database rollback**: Drop rag_cache_index and rag_metadata_index (non-destructive)
4. **Scheduled tasks**: Cancel pending tasks from scheduled_tasks collection

---

## Part 9: Unified Metadata Thread (Flows Through All 5 Phases)

The system uses a single metadata bundle that accumulates information as it flows through all phases. This enables complete visibility into decision-making and future analytics.

### Metadata Structure
```python
{
  "embedding": [...],              # 768-dim embedding of input
  "depth": "SHALLOW|MEDIUM|DEEP",  # Query complexity classification
  "cache_hit": bool,               # Was result from cache?
  "confidence_scores": {           # Per-dispatcher confidence
    "user_rag": float,
    "internal_rag": float,
    "external_rag": float
  },
  "rag_sources_used": [...],       # Which dispatchers actually ran
  "extraction_results": {          # From consolidator phase
    "attempt": int,                # Harvester attempt count
    "evaluator_passes": int,       # Evaluator feedback loops
    "contradiction_flags": [...]   # What was rejected
  },
  "write_success": bool,           # Did atomic write commit?
  "cache_invalidation": {          # What was invalidated
    "patterns_cleared": [...],
    "scope": "pattern|user|all"
  }
}
```

### Flow Through Phases
1. **Phase 0 (Input Analysis)**: Create metadata with embedding
2. **Phase 1 (Cache Check)**: Add cache_hit flag
3. **Phase 2 (RAG)**: Add depth, dispatcher scores, sources_used
4. **Phase 3 (Response)**: Pass metadata through (no changes)
5. **Phase 4 (Consolidation)**: Add extraction_results, write_success, cache_invalidation
6. **Return**: Final metadata available for logging/analytics

This unified approach ensures no information is lost between phases and enables future adaptation.

---

## Part 10: Before/After Metrics (See TEST_PLAN.md for Details)

### What to Measure BEFORE Implementation
Collect baseline metrics from current system (before any code changes):
- **RAG Latency**: Average & p95 dispatcher round-time
- **Dispatcher Calls**: Average per conversation
- **Database Load**: Query count, write count, transaction duration
- **Error Rate**: Extraction failures, contradictions, retries
- **Memory**: Cache misses (baseline = 0% hit rate)
- **Consistency**: Fact contradiction rate in existing memory

### What to Measure AFTER Deployment
Collected same metrics from new system (after all 5 stages deployed):
- **RAG Latency**: Should be 25% faster (1200ms → 900ms)
- **Dispatcher Calls**: Should drop 50% (3.0 → 1.5)
- **Cache Hit Rate**: Should reach 60-70% (from 0%)
- **Cache Lookup**: Should be <15ms (new capability)
- **Error Rate**: Should drop to <0.5% (from 1-2%)
- **Consolidator Time**: Should stay <4000ms (acceptable +33%)
- **Load**: 10 concurrent conversations at p95 <3000ms

### Comparison Approach
Run TEST_PLAN.md comprehensive suite covering:
- 40+ unit tests (per-component validation)
- 12+ integration tests (interaction validation)
- 3 E2E scenarios (full conversation flow)
- Performance benchmarks (cache, depth routing, RAG)
- Load tests (10 concurrent users)
- Generate before/after visual comparison

---

## Summary: What Gets Built

```
BEFORE:
Input → RAG (3 parallel) → Response → Consolidator → DB → Done

AFTER:
Input
  ↓ (embedding, depth)
Cache Check (fast path)
  ↓ (if hit)
RAG (depth-aware, selective dispatch, early-exit)
  ↓ (results + metadata)
Response
  ↓
Consolidator (with evaluator loop, validation)
  ↓
Atomic DB Write + Cache Invalidation
  ↓
Scheduled Tasks (promise fulfillment async)
  ↓
Done
```

**Key improvements**:
- ✅ Cache: 60-70% hit rate, <15ms lookup
- ✅ Depth routing: 40% fewer dispatcher calls
- ✅ Evaluator: 95%+ contradiction detection
- ✅ Atomicity: No partial writes
- ✅ Cache invalidation: Strategic, not blunt

---

**Next Step**: See TEST_PLAN.md for comprehensive validation after implementation
