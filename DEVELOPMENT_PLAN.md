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

### Stage 1: Foundation Modules (No dependencies, can implement in parallel)

#### 1a. Create `kazusa_ai_chatbot/cache.py` (NEW FILE)
**Purpose**: RAGCache class for semantic caching
**Changes** in this file:
- [ ] RAGCache class with methods:
  - `__init__(backend_type)` - Support "redis" or "memory"
  - `async retrieve_if_similar(embedding, cache_type, threshold)` - Vector similarity lookup
  - `async store(embedding, results, cache_type, ttl_seconds, metadata)` - Store results
  - `async invalidate_pattern(cache_type, global_user_id, trigger)` - Selective invalidation
  - `async clear_all_user(global_user_id)` - Full user cache clear
  - `get_stats()` - Cache stats (hits, misses, size)
- [ ] Backend implementations:
  - Redis backend (if chosen) with connection pooling
  - In-memory LRU backend with thread safety
- [ ] Embedding similarity computation (cosine, threshold-based)
- [ ] TTL management (expiry checking)

**Dependencies**: None (standalone module)
**Lines of code**: ~300-400
**Test file**: `tests/unit/test_rag_cache.py` (will be created during testing)

---

#### 1b. Create `kazusa_ai_chatbot/depth_classifier.py` (NEW FILE)
**Purpose**: Classify query depth (SHALLOW/MEDIUM/DEEP)
**Changes** in this file:
- [ ] InputDepthClassifier class:
  - `async classify(input, user_topic, affinity)` - Main classification method
  - Heuristic engine (temporal keywords, pronouns, emotional words)
  - Optional light LLM for ambiguous cases (disabled by default)
  - Fallback to MEDIUM for edge cases
- [ ] Output structure: `{depth, trigger_dispatchers, confidence_threshold}`
- [ ] Affinity-aware routing (low affinity → DEEP for careful retrieval)
- [ ] Contradiction detection heuristics

**Dependencies**: None (standalone module)
**Lines of code**: ~200-300
**Test file**: `tests/unit/test_depth_classifier.py`

---

#### 1c. Update `kazusa_ai_chatbot/scheduler.py` (EXISTING FILE - REFACTOR)
**Purpose**: Extract and generalize scheduler for all async tasks
**Current state**: Promise scheduling embedded in consolidator
**Changes**:
- [ ] Extract scheduler logic to standalone module
- [ ] Create `ScheduledTask` dataclass (task_id, task_type, global_user_id, due_time, metadata, status)
- [ ] Create `TaskScheduler` class:
  - `async schedule_task(task_type, global_user_id, due_time, metadata)` - Schedule
  - `async cancel_task(task_id)` - Cancel
  - `async list_pending()` - List pending tasks
  - Task execution handlers (extensible)
- [ ] Database storage for task state (scheduled_tasks collection)
- [ ] Task recovery on restart (query pending from DB)
- [ ] Integration with FastAPI background tasks or APScheduler

**Dependencies**: None (may import DB utility)
**Lines of code**: ~200-300
**Test file**: `tests/unit/test_scheduler.py`

---

### Stage 2: Database Layer (After Stage 1 foundation)

#### 2a. Update `kazusa_ai_chatbot/db.py` (EXISTING FILE - ADD FUNCTIONS)
**Purpose**: New database operations for cache and metadata
**Current state**: Existing queries for memory, user_profiles, etc.
**Changes** (ADD, don't remove):
- [ ] New collection operations:
  - `async insert_cache_entry(embedding, cache_type, ttl_seconds, metadata)` - Cache storage
  - `async find_similar_embeddings(embedding, cache_type, similarity_threshold)` - Vector search
  - `async invalidate_cache_pattern(cache_type, global_user_id, trigger)` - Invalidation
  - `async clear_cache_for_user(global_user_id)` - Full user clear
  - `async increment_rag_version(global_user_id)` - Version tracking
  - `async get_rag_version(global_user_id)` - Version retrieval
  - `async create_scheduled_task(task_doc)` - Schedule task storage
  - `async update_task_status(task_id, status)` - Task state update

- [ ] Verify/create indices:
  - On `rag_cache_index`: `(embedding, cache_type, ttl_expires_at)`
  - On `rag_metadata_index`: `(global_user_id, rag_version)`
  - Ensure `memory.embedding` has vector index (768-dim, cosine)
  - Ensure `user_profiles.embedding` has vector index

- [ ] Update existing functions (non-breaking):
  - `save_memory()` - Add optional metadata parameter
  - `update_affinity()` - Return delta applied
  - `upsert_user_facts()` - Return count

**Dependencies**: Stage 1 (may need constants from cache/scheduler modules)
**Lines of code**: ~400-500
**No test file created** (tested in integration)

---

#### 2b. Create migration script `migrations/001_add_cache_metadata_collections.py` (NEW FILE)
**Purpose**: MongoDB schema changes
**Changes** (this is a one-time migration):
- [ ] Create `rag_cache_index` collection with schema
- [ ] Create `rag_metadata_index` collection with schema
- [ ] Create TTL index on `rag_cache_index.ttl_expires_at`
- [ ] Verify vector indices on `memory.embedding` and `user_profiles.embedding`
- [ ] Rollback function (drop collections, restore indices)

**Dependencies**: Database module
**Lines of code**: ~100-150

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
