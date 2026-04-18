# Personality Simulation Architecture: Medium-Term Development Plan

**Status**: PLANNING
**Last Updated**: 2026-04-18
**Target Completion**: 2026-06-30 (Estimated 10 weeks)
**Scope**: Unified RAG-Consolidator System with Intelligent Memory Management

---

## Executive Summary

This plan describes the evolution from current **loosely-coordinated RAG + Consolidator** to a **tightly-integrated 5-phase personality system** that mirrors human memory and learning patterns.

**Key improvements**:
- 60-70% cache hit rate on repetitive queries
- 40% reduction in unnecessary dispatcher calls
- Faster learning through immediate consolidation
- Better personality consistency via evaluator loop
- Foundation for offline consolidation (future)

**Total Effort**: ~10 weeks (4 devs, parallel tracks)
**Risk Level**: MEDIUM (requires database schema changes, testing critical)

---

## Part 1: High-Level Architecture (10 Steps Ahead)

### Current State (2026-04-18)
```
User Input
  ↓
Decontextualize + Sentiment
  ↓
RAG (3 parallel dispatchers, no depth awareness)
  ├─ External search agent
  ├─ Internal memory retrieval
  └─ User facts retrieval
  ↓
Response Generation
  ↓
Consolidation (3 parallel, independent)
  ├─ Global state updater
  ├─ Relationship recorder
  └─ Facts harvester (no evaluator loop)
  ↓
Database Write
  ↓
Output
```

**Problems**:
- No caching → Full RAG on every similar query
- No depth awareness → Wastes LLM on shallow queries
- Facts harvester has no feedback loop → Potential inconsistencies
- No early exit → All dispatchers run regardless

---

### Target State (2026-06-30)

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPREHENSIVE SYSTEM FLOW                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 0: INPUT ANALYSIS                                        │
│  ├─ Decontextualization                                         │
│  ├─ Compute embedding (768-dim)                                │
│  ├─ Detect entities, temporal markers                           │
│  ├─ Build metadata bundle                                       │
│  └─ Output: {embedding, depth_hint, entities, metadata}        │
│                                                                 │
│  PHASE 1: SEMANTIC CACHE CHECK (NEW)                           │
│  ├─ Compute similarity against cached embeddings               │
│  ├─ If similarity > 0.82 & within TTL                          │
│  │  └─ Return cached facts + metadata.cache_hit=T             │
│  ├─ Else proceed to Phase 2a                                   │
│  └─ Cache implemented in RedisCache + in-memory LRU            │
│                                                                 │
│  PHASE 2a: DEPTH-AWARE RAG DISPATCH (UPDATED)                  │
│  ├─ Classify input depth: SHALLOW / MEDIUM / DEEP              │
│  ├─ Conditional dispatcher execution:                          │
│  │  ├─ SHALLOW: user_rag only                                  │
│  │  ├─ MEDIUM: user_rag + conditional internal_rag            │
│  │  └─ DEEP: user_rag + internal_rag + external_rag (seq)     │
│  ├─ Sequential execution with early-exit scoring               │
│  ├─ Merge results with consistency validation                  │
│  └─ Output: research_facts + confidence_scores                 │
│                                                                 │
│  PHASE 2b: FALLBACK (Cache Hit Path)                           │
│  └─ Skip RAG, return research_facts from cache                 │
│                                                                 │
│  PHASE 3: RESPONSE GENERATION                                  │
│  ├─ Generate personality-aligned response                      │
│  ├─ Score response confidence                                  │
│  ├─ Quality check: does response depth match query depth?      │
│  └─ Output: final_dialog + internal_monologue + subtext        │
│                                                                 │
│  PHASE 4: CONSOLIDATION WITH CONSISTENCY VALIDATION (UPDATED)  │
│  ├─ Parallel stream A: global_state_updater                    │
│  │                    → mood, vibe, reflection                 │
│  ├─ Parallel stream B: relationship_recorder                   │
│  │                    → diary, affinity_delta, insight         │
│  ├─ Parallel stream C: facts_harvester (UPDATED)               │
│  │  ├─ Extract new_facts, future_promises                      │
│  │  ├─ Include research_facts in context                       │
│  │  └─ Loop to evaluator                                       │
│  │                                                              │
│  ├─ fact_harvester_evaluator (NEW)                             │
│  │  ├─ Validate against research_facts                         │
│  │  ├─ Check taxonomy, identity, temporal consistency          │
│  │  ├─ If PASS: should_stop=T → proceed to Phase 5             │
│  │  ├─ If FAIL: feedback loop back to facts_harvester          │
│  │  └─ Max retries: 3                                          │
│  │                                                              │
│  └─ All 4 jobs join at Phase 5                                 │
│                                                                 │
│  PHASE 5: ATOMIC WRITE + CACHE INVALIDATION (UPDATED)          │
│  ├─ 1. Write character state                                   │
│  ├─ 2. Write user relationship (diary, insight)                │
│  ├─ 3. Write facts & promises                                  │
│  ├─ 4. Write affinity update                                   │
│  ├─ 5. COMMIT transaction                                      │
│  ├─ 6. Invalidate cache strategically:                         │
│  │  ├─ If new_facts: RAGCache.invalidate(user_facts, user_id)  │
│  │  ├─ If |affinity_delta| > 50: RAGCache.clear_all(user_id)   │
│  │  └─ Update RAG metadata version                             │
│  ├─ 7. Trigger scheduled event creation (if due_time exists)   │
│  └─ Output: write_success + metadata                           │
│                                                                 │
│  PHASE 6: ASYNC SCHEDULED TASKS (NEW)                          │
│  ├─ Scheduled promise fulfillment checks                       │
│  ├─ Offline consolidation (future)                             │
│  └─ Personality analytics aggregation                          │
│                                                                 │
│  NEXT INPUT → Sees updated memory + warm cache                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Requirements & Rationale

### Requirement 1: Semantic Cache Layer
**Why**: Reduce repetitive RAG calls, improve latency
- Current: user asks "What's my favorite food?" → Full 3-dispatcher RAG
- Future: Similar question within 30min → Cache hit in 5-10ms
- Impact: 60-70% of conversational queries are repetitive

**Components**:
- Redis or in-memory cache backend
- Embedding similarity computation (cosine, threshold 0.82)
- TTL-based expiry (user_facts: 30min, internal: 15min)
- Atomic invalidation on db_writer commit

---

### Requirement 2: Depth-Aware Intelligent Routing
**Why**: Allocate retrieval effort proportionally to query complexity
- Current: "Hello" and "Why do you think I'm important?" both run 3 dispatchers
- Future: Classify depth, skip unnecessary retrievals
- Impact: 40% fewer dispatcher calls, local LLM gets easier queries

**Components**:
- InputDepthClassifier (heuristics + optional light LLM)
- Conditional dispatch graph (skip non-essential retrievals)
- Early-exit scoring (if confidence high enough, skip next tier)
- Metadata attachment (depth classification flows through system)

---

### Requirement 3: Consistency Evaluator Loop
**Why**: Catch extraction errors before writing, prevent memory contradictions
- Current: facts_harvester writes directly without validation
- Future: Evaluator validates against research_facts, rejects inconsistencies
- Impact: Personality stays coherent, learning is accurate

**Components**:
- fact_harvester_evaluator LLM node
- Contradiction detection (new_fact vs research_facts)
- Feedback loop with max 3 retries
- Clear/rejected fact handling

---

### Requirement 4: Unified Metadata Thread
**Why**: Enable system visibility and adaptation across 5 phases
- Current: Each phase independent, no context sharing
- Future: Single metadata bundle threads through all phases
- Impact: Can correlate cache hits → response quality → extraction accuracy

**Components**:
- Metadata schema (embedding, depth, cache_hit, confidence scores, etc.)
- Propagation through all 5 phases
- Optional analytics/logging endpoint

---

### Requirement 5: Database Schema Updates for Efficiency
**Why**: Support new queries (cache lookups, metadata), improve performance
- Current: Basic collections (memory, user_profiles, etc.), no cache indices
- Future: Add rag_cache_index, improve vector search, add metadata
- Impact: Faster cache lookups, efficient invalidation patterns

**Components**:
- rag_cache_index collection (embeddings, TTL, cache_type)
- rag_metadata_index collection (version tracking, analytics)
- Vector indices on memory.embedding for fast similarity
- TTL indices for automatic expiry

---

### Requirement 6: Scheduled Task Infrastructure
**Why**: Foundation for offline consolidation, promise fulfillment, analytics
- Current: FastAPI handles future_promises scheduling
- Future: Extend to general async tasks (offline consolidation, analytics)
- Impact: Personality improves overnight, autonomous maintenance

**Components**:
- Unified scheduler abstraction (currently specific to promises)
- Task registry for offline consolidation
- Cron-based triggers (nightly consolidation, weekly analytics)
- Task state tracking (pending, running, completed, failed)

---

## Part 3: Phased Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Set up database & infrastructure for new requirements

#### 1a. Database Schema Updates
- [ ] Create `rag_cache_index` collection
  - Schema: `{_id, embedding: [768], cache_type, ttl_expires_at, metadata, created_at}`
  - Indices: `{embedding: 1, ttl_expires_at: 1, cache_type: 1}`
  - TTL index: `db.rag_cache_index.createIndex({ttl_expires_at: 1}, {expireAfterSeconds: 0})`
- [ ] Create `rag_metadata_index` collection
  - Schema: `{_id, global_user_id, rag_version, last_rag_run, metadata}`
  - Indices: `{global_user_id: 1, rag_version: 1}`
- [ ] Add vector indices to `memory` collection (if not exists)
  - Index name: `memory_vector_index`
  - Verify 768-dim cosine similarity configured
- [ ] Add version field to `user_profiles` collection
  - Schema: `{..., profile_version: int, last_updated: timestamp}`

**Deliverable**: Migration script + verified schema

---

#### 1b. Cache Backend Setup
- [ ] Evaluate Redis vs in-memory LRU (decision point)
  - Redis: Shareable across processes, persistent options
  - In-memory: Simpler, faster, but process-local
- [ ] Install dependencies (redis-py if Redis chosen, or use built-in collections)
- [ ] Create RAGCache class with methods:
  - `async retrieve_if_similar(embedding, cache_type, threshold)`
  - `async store(embedding, results, cache_type, ttl_seconds, metadata)`
  - `async invalidate_pattern(cache_type, global_user_id, trigger)`
  - `async clear_all_user(global_user_id)`
  - `async get_stats()` (hits, misses, size)

**Deliverable**: RAGCache class + tests

---

#### 1c. Scheduler Infrastructure
- [ ] Extract scheduler logic from future_promises into standalone module
  - New file: `kazusa_ai_chatbot/scheduler.py`
  - Classes: `ScheduledTask`, `TaskScheduler`
  - Methods: `schedule_task()`, `cancel_task()`, `list_pending()`
- [ ] Integrate with FastAPI background tasks or APScheduler
- [ ] Add task state tracking to database (ScheduledTaskDoc)

**Deliverable**: Scheduler module + integration test

---

### Phase 2: RAG Layer Updates (Weeks 3-4)
**Goal**: Add caching and depth classification to RAG

#### 2a. Input Depth Classifier
- [ ] Create `InputDepthClassifier` class in `persona_supervisor2_rag.py`
  - Methods:
    - `async classify(input, user_topic, affinity)` → {depth, trigger_dispatchers, confidence_threshold}
  - Heuristics:
    - Temporal markers? ("when", "before", "after") → MEDIUM+
    - Question about past? ("remember", "did") → MEDIUM+
    - Emotional keywords? (sentiment score) → MEDIUM
    - Affinity < 400? (mistrusting) → DEEP
    - Contradiction signals? → DEEP
  - Light LLM (only for ambiguous cases)
  - Fallback: Default to MEDIUM (safe)

- [ ] Add depth classification to RAG dispatch logic
  - Update `call_rag_subgraph()` to include depth step
  - Pass depth to conditional edges

**Deliverable**: InputDepthClassifier + integration test

---

#### 2b. Semantic Cache Integration
- [ ] Add Phase 1 (Cache Check) before Phase 2a (RAG Dispatch)
  - In `call_rag_subgraph()`:
    ```python
    # Phase 1
    cache_results = await rag_cache.retrieve_if_similar(
        embedding=input_embedding,
        cache_type="user_facts",
        threshold=0.82
    )
    if cache_results:
        return {research_facts: cache_results, cache_hit: True}

    # Phase 2a (existing)
    ...
    ```

- [ ] Update results to include cache metadata
  - Add `research_facts` → `{user_rag_finalized, internal_rag_results, external_rag_results, cache_hit, metadata}`

**Deliverable**: Cache integration + test

---

#### 2c. Conditional Dispatch (Early Exit)
- [ ] Update dispatcher nodes to return confidence scores
  - `call_web_search_agent()` → `{response, confidence_score}`
  - `call_memory_retriever_agent_*()` → `{response, confidence_score}`

- [ ] Update conditional edges to support early exit
  - SHALLOW + user_rag confidence > 0.90 → Skip internal_rag
  - MEDIUM + internal_rag confidence > 0.85 → Skip external_rag
  - Example:
    ```python
    def conditional_internal_rag_needed(state: RAGState) -> str:
        depth = state["metadata"]["depth"]
        user_confidence = state.get("user_confidence", 0)
        if depth == "SHALLOW" and user_confidence > 0.90:
            return "skip"
        return "continue"
    ```

- [ ] Verify with tests on all depth levels

**Deliverable**: Conditional dispatch with early-exit + tests

---

#### 2d. Cache Storage
- [ ] After RAG completes, store results in cache
  - In `call_rag_subgraph()` return, before exiting:
    ```python
    if not state.get("cache_hit"):
        # Store RAG results for future queries
        await rag_cache.store(
            embedding=state["input_embedding"],
            results=research_facts,
            cache_type="user_facts",
            ttl_seconds=1800,  # 30 min
            metadata=metadata
        )
    ```

**Deliverable**: Cache storage + verification

---

### Phase 3: Consolidator Updates (Weeks 5-6)
**Goal**: Add evaluator loop and consistency validation

#### 3a. facts_harvester Enhancement
- [ ] Update `facts_harvester()` to receive `research_facts` in context
  - Pass to LLM system prompt
  - Instruct: "Validate new facts don't contradict research_facts"

- [ ] Update message construction:
  ```python
  msg = {
      "decontexualized_input": state["decontexualized_input"],
      "research_facts": state["research_facts"],  # NEW
      "content_anchors": state["action_directives"][...],
      "metadata": {  # NEW
          "cache_hit": state.get("cache_hit"),
          "depth": state.get("metadata", {}).get("depth"),
      }
  }
  ```

**Deliverable**: Enhanced facts_harvester + test

---

#### 3b. Implement Evaluator Loop
- [ ] Create `fact_harvester_evaluator()` (already in code, enhance it)
  - Current: Validates format, entity anchoring, taxonomy
  - NEW: Add consistency check against research_facts
    ```python
    # Check for contradictions
    for new_fact in new_facts:
        for research_fact in research_facts:
            if semantic_contradiction(new_fact, research_fact):
                return {"should_stop": False, "feedback": "Fact contradicts RAG result..."}
    ```

- [ ] Extend feedback mechanism
  - `fact_harvester_feedback_message` already used
  - Ensure facts_harvester processes feedback correctly
  - Max retries: 3 (currently configurable)

- [ ] Test evaluator loop with contradiction scenarios
  - Create test cases with contradictory facts
  - Verify feedback loop fires
  - Verify max retries enforced

**Deliverable**: Evaluator enhancement + test scenarios

---

#### 3c. Metadata Propagation
- [ ] Create unified metadata schema (see Part 1)
  ```python
  metadata = {
      "input_embedding": [...],
      "depth": "MEDIUM",
      "entities": ["user_name", "project"],
      "cache_hit": False,
      "rag_sources_used": ["user_rag", "internal_rag"],
      "confidence_scores": {"user_rag": 0.92, "internal_rag": 0.78},
      "evaluator_passes": 2,
      ...
  }
  ```

- [ ] Thread metadata through all 5 phases
  - Phase 0: Initialize
  - Phase 1: Add cache_hit
  - Phase 2a: Add rag_sources_used, confidence_scores
  - Phase 3: Add response_confidence
  - Phase 4: Add extraction_confidence, evaluator_passes
  - Phase 5: Add write_success, cache_invalidation_scope

**Deliverable**: Metadata schema + propagation

---

### Phase 4: Database Integration (Weeks 7-8)
**Goal**: Implement cache invalidation and write coordination

#### 4a. Atomic Write Transaction
- [ ] Update `db_writer()` to wrap all writes in a transaction
  ```python
  async def db_writer(state: ConsolidatorState):
      async with mongo_client.start_session() as session:
          async with session.start_transaction():
              # Step 1: Character state
              await upsert_character_state(...)
              # Step 2: User relationship
              await upsert_user_facts(...)
              # Step 3: Memory (facts/promises)
              for new_fact in new_facts:
                  await save_memory(...)
              # Step 4: Affinity
              await update_affinity(...)
              # Commit happens here automatically
  ```

- [ ] Test transaction rollback scenarios
  - Simulate write failure in middle of transaction
  - Verify rollback works

**Deliverable**: Atomic transaction + test

---

#### 4b. Strategic Cache Invalidation
- [ ] Implement invalidation logic in `db_writer()`
  ```python
  # After successful commit:

  global_user_id = state["global_user_id"]
  new_facts = state.get("new_facts", [])
  affinity_delta = state.get("affinity_delta", 0)

  if new_facts:
      await rag_cache.invalidate_pattern(
          cache_type="user_facts",
          global_user_id=global_user_id,
          trigger="new_facts_written"
      )

  if abs(affinity_delta) > 50:
      await rag_cache.clear_all_user(global_user_id)

  # Update RAG version
  await update_rag_version(global_user_id)
  ```

- [ ] Test invalidation patterns
  - Verify user_facts cache cleared on new_facts
  - Verify all-user cache cleared on major affinity change
  - Verify version incremented

**Deliverable**: Invalidation logic + test

---

#### 4c. Future Promises → Scheduled Tasks
- [ ] Extract promise scheduling into `db_writer()`
  ```python
  future_promises = state.get("future_promises", [])
  for promise in future_promises:
      if promise.get("due_time"):
          await task_scheduler.schedule_task(
              task_type="promise_fulfillment_check",
              global_user_id=global_user_id,
              promise_id=promise.get("_id"),
              due_time=promise["due_time"]
          )
  ```

- [ ] Implement promise fulfillment check handler
  - Handler: Generates reminder message to bot
  - Stored in scheduled_events collection

**Deliverable**: Promise scheduling integration + test

---

#### 4d. Version Tracking
- [ ] Create `rag_metadata_index` entries on write
  ```python
  await db.rag_metadata_index.update_one(
      {"global_user_id": global_user_id},
      {
          "$inc": {"rag_version": 1},
          "$set": {"last_rag_run": datetime.now()},
          "$set": {"metadata": {...}}
      },
      upsert=True
  )
  ```

- [ ] Next RAG run: Check version before using cache
  - If metadata.rag_version < db.rag_version → Cache invalid

**Deliverable**: Version tracking + test

---

### Phase 5: Scheduled Tasks & Monitoring (Weeks 9-10)
**Goal**: Enable autonomous tasks and observability

#### 5a. Scheduled Task Framework
- [ ] Define task types in config
  ```python
  SCHEDULED_TASKS = {
      "promise_fulfillment_check": {
          "handler": check_promise_due,
          "cron": None,  # One-shot, use due_time
      },
      "offline_consolidation": {
          "handler": consolidate_overnight,
          "cron": "0 2 * * *",  # 2 AM daily
      },
      "personality_analytics": {
          "handler": compute_analytics,
          "cron": "0 8 * * 0",  # 8 AM Sunday
      }
  }
  ```

- [ ] Implement task scheduler in main app
  ```python
  # In main.py / startup
  task_scheduler = TaskScheduler(config=SCHEDULED_TASKS)
  await task_scheduler.start()
  ```

**Deliverable**: Task framework + config

---

#### 5b. Offline Consolidation (Future)
- [ ] Implement `consolidate_overnight()` handler
  - Runs nightly
  - Samples recent facts from memory collection
  - Re-evaluates them against character_profile
  - Strengthens personality-aligned facts
  - Flags contradictions for next day review

- [ ] (Detailed implementation in future phase)

**Deliverable**: Placeholder + doc

---

#### 5c. Observability & Monitoring
- [ ] Add logging at each phase
  - Phase 0: Input classification
  - Phase 1: Cache hit/miss
  - Phase 2a: Dispatcher selection
  - Phase 3: Response confidence
  - Phase 4: Evaluator passes/fails
  - Phase 5: Write success + invalidation scope

- [ ] Create metrics endpoint (`/metrics`)
  - Cache hit rate
  - Average dispatcher time
  - Evaluator retry rate
  - Write latency

- [ ] Add metadata logging
  - Persist metadata from each conversation to analytics collection
  - Enable post-hoc analysis of personality formation

**Deliverable**: Logging + metrics endpoint

---

#### 5d. Testing & Documentation
- [ ] Integration tests for entire 5-phase flow
  - Test SHALLOW query → DEEP query → Same query (cache hit)
  - Test consolidation loop with contradiction
  - Test cache invalidation
  - Test scheduled task execution

- [ ] Performance benchmarks
  - Cache hit latency: target < 15ms
  - Full RAG latency: < 2000ms (current baseline)
  - Consolidator latency: < 5000ms

- [ ] Documentation
  - Update architecture docs
  - API documentation for new endpoints
  - Troubleshooting guide

**Deliverable**: Tests + benchmarks + docs

---

## Part 4: Component-by-Component Checklist

### RAG Module (`persona_supervisor2_rag.py`)

**Initialization & Setup**
- [ ] Import RAGCache, InputDepthClassifier
- [ ] Initialize rag_cache in module startup
- [ ] Initialize depth_classifier

**Input Processing (Phase 0)**
- [ ] Add embedding computation (use existing or add new)
- [ ] Store embedding in metadata

**Cache Check (Phase 1)**
- [ ] Implement cache lookup before dispatcher
- [ ] Check similarity threshold
- [ ] Return early if cache hit with metadata

**Depth Classification**
- [ ] Integrate InputDepthClassifier
- [ ] Pass depth to conditional edges
- [ ] Handle edge cases (ambiguous inputs, fallback to MEDIUM)

**Conditional Dispatch (Phase 2a)**
- [ ] Add confidence scoring to all dispatcher nodes
- [ ] Implement early-exit logic based on depth + confidence
- [ ] Merge results with consistency checks

**Cache Storage**
- [ ] Store RAG results in cache after dispatch
- [ ] Handle metadata attachment

**Testing**
- [ ] Unit test: depth classification on known inputs
- [ ] Unit test: cache hit/miss scenarios
- [ ] Integration test: full Phase 0-2 flow
- [ ] Performance test: cache latency < 15ms

---

### Consolidator Module (`persona_supervisor2_consolidator.py`)

**facts_harvester Enhancement**
- [ ] Add research_facts to LLM context
- [ ] Add metadata to message construction
- [ ] Test with consistent facts (should pass)
- [ ] Test with contradictory facts (should fail evaluator)

**fact_harvester_evaluator Enhancement**
- [ ] Add research_facts comparison logic
- [ ] Implement contradiction detection
- [ ] Ensure feedback message specific
- [ ] Test max retries (should stop at 3)

**Metadata Propagation**
- [ ] Initialize metadata in initial_state
- [ ] Thread through each node
- [ ] Accumulate fields at each phase
- [ ] Return with final state

**db_writer Atomicity**
- [ ] Wrap writes in transaction
- [ ] Implement rollback test
- [ ] Verify all-or-nothing semantics

**Cache Invalidation**
- [ ] Call rag_cache.invalidate_pattern() on new_facts
- [ ] Call rag_cache.clear_all_user() on major affinity change
- [ ] Update rag_metadata_index version
- [ ] Test invalidation timing (after commit, not before)

**Testing**
- [ ] Unit test: evaluator with contradictions
- [ ] Integration test: facts_harvester → evaluator loop
- [ ] Integration test: db_writer with transaction rollback
- [ ] End-to-end test: Full consolidation with cache invalidation

---

### Database Module (`kazusa_ai_chatbot/db.py`)

**New Collections**
- [ ] Create `rag_cache_index`
  - Schema defined
  - Indices created (embedding, ttl_expires_at, cache_type)
  - TTL index configured
- [ ] Create `rag_metadata_index`
  - Schema defined
  - Indices created (global_user_id, rag_version)

**New Queries**
- [ ] `insert_cache_entry(embedding, cache_type, ttl_seconds, metadata)`
- [ ] `find_similar_embeddings(embedding, cache_type, similarity_threshold)`
- [ ] `invalidate_cache_pattern(cache_type, global_user_id, trigger)`
- [ ] `clear_cache_for_user(global_user_id)`
- [ ] `increment_rag_version(global_user_id)`
- [ ] `get_rag_version(global_user_id)`

**Existing Modifications**
- [ ] `save_memory()` - add metadata parameter (optional)
- [ ] `update_affinity()` - return delta applied (for scheduling)
- [ ] `upsert_user_facts()` - return count of updates (for logging)

**Vector Index Verification**
- [ ] Verify `memory_vector_index` exists (768-dim, cosine)
- [ ] Verify `user_profiles` has embedding index
- [ ] Add migration if indices missing

**Testing**
- [ ] Test cache entry insertion/retrieval
- [ ] Test similarity search (various thresholds)
- [ ] Test TTL expiry (mock or wait)
- [ ] Test invalidation patterns
- [ ] Performance test: cache lookup < 5ms

---

### Scheduler Module (`kazusa_ai_chatbot/scheduler.py`) - NEW

**Core Classes**
- [ ] `ScheduledTask` (dataclass)
  - Fields: task_id, task_type, global_user_id, due_time, metadata, status
- [ ] `TaskScheduler` (main)
  - Methods: schedule_task(), cancel_task(), list_pending(), run_tasks()

**Integration with FastAPI**
- [ ] Use APScheduler or native asyncio scheduling
- [ ] Startup hook in app initialization
- [ ] Graceful shutdown hook

**Task Handlers**
- [ ] `promise_fulfillment_check()` - Check promise due date
- [ ] `offline_consolidation()` - TBD (future)
- [ ] `personality_analytics()` - TBD (future)

**Persistence**
- [ ] Store task state in `scheduled_tasks` collection
- [ ] Query on startup to recover pending tasks
- [ ] Update status on completion

**Testing**
- [ ] Unit test: schedule_task() creates DB entry
- [ ] Integration test: task executes at due_time
- [ ] Test task recovery on restart

---

### Cache Module (`kazusa_ai_chatbot/cache.py`) - NEW

**RAGCache Class**
- [ ] `__init__()` - Initialize cache backend
- [ ] `async retrieve_if_similar(embedding, cache_type, threshold)` - Lookup
- [ ] `async store(embedding, results, cache_type, ttl_seconds, metadata)` - Store
- [ ] `async invalidate_pattern(cache_type, global_user_id, trigger)` - Invalidate
- [ ] `async clear_all_user(global_user_id)` - Full user clear
- [ ] `async get_stats()` - Hits, misses, size

**Backend Options**
- [ ] Redis backend (if chosen)
  - [ ] Connection pooling
  - [ ] Error handling + retry
- [ ] In-memory backend
  - [ ] LRU eviction
  - [ ] Thread-safe (if needed)

**Testing**
- [ ] Unit test: store + retrieve cycle
- [ ] Unit test: TTL expiry
- [ ] Unit test: similarity matching
- [ ] Performance test: < 15ms latency

---

### Depth Classifier Module (`kazusa_ai_chatbot/depth_classifier.py`) - NEW

**InputDepthClassifier Class**
- [ ] `async classify(input, user_topic, affinity)` - Main method
- [ ] Heuristic engine (keyword matching)
- [ ] Light LLM for ambiguous cases
- [ ] Fallback to MEDIUM
- [ ] Output schema: {depth, trigger_dispatchers, confidence_threshold}

**Testing**
- [ ] Test SHALLOW classification ("What's your name?")
- [ ] Test MEDIUM classification ("Do you remember...?")
- [ ] Test DEEP classification ("Why do you think...?")
- [ ] Test edge cases (empty input, very long input)
- [ ] Performance test: < 100ms (should be heuristic-only most of the time)

---

## Part 5: Testing Strategy

### Unit Tests (Per Component)
- [ ] RAGCache: 10 tests (store, retrieve, TTL, invalidate)
- [ ] InputDepthClassifier: 8 tests (classifications, edge cases)
- [ ] facts_harvester_evaluator: 6 tests (consistency checks, retries)
- [ ] TaskScheduler: 5 tests (schedule, cancel, recovery)

### Integration Tests (Multi-Component)
- [ ] RAG Module: Full Phase 0-2 flow
- [ ] Consolidator Module: Full Phase 4-5 flow
- [ ] RAG + Consolidator: End-to-end personality loop
- [ ] Database: Transaction rollback scenarios
- [ ] Scheduler: Task execution timing

### End-to-End Tests (Full System)
- [ ] Conversation flow: Input → DepthClassify → CacheCheck → RAG → Response → Consolidate → DB → InvalidateCache
- [ ] Learning scenario: User tells bot something new → Bot consolidates → Next similar query uses cache and learned fact
- [ ] Contradiction scenario: Facts_harvester extracts contradictory fact → Evaluator rejects → Feedback loop fixes
- [ ] Promise scenario: User makes commitment with due_time → Task scheduled → Promise fulfillment check triggered

### Performance Tests
- [ ] Cache lookup: < 15ms
- [ ] Full RAG dispatch: < 2000ms (baseline)
- [ ] Consolidation: < 5000ms
- [ ] Metadata overhead: < 5% latency inflation

### Load Tests
- [ ] 10 concurrent conversations
- [ ] Cache hit rate under load
- [ ] Database write throughput

---

## Part 6: Rollout Plan

### Rollout Strategy: Feature Flags + Gradual Deployment

```
Week 9-10: Staging
  - Deploy to staging environment
  - Run full integration tests
  - Performance benchmarks
  - Stakeholder review

Week 10: Canary (5% Traffic)
  - Deploy to production
  - Monitor: errors, latency, cache hit rate
  - Rollback plan ready

Week 10-11: Ramp (25% → 50% → 100%)
  - Gradually increase traffic
  - Monitor metrics
  - Rollback if needed

Week 11: Monitoring & Tuning
  - Cache TTL tuning
  - Depth classification improvements
  - Off-hours performance optimization
```

### Rollback Plan
- [ ] Disable cache via feature flag: `USE_RAG_CACHE=false`
- [ ] Disable depth classification: `USE_DEPTH_CLASSIFIER=false`
- [ ] Revert database schema changes (backup + restore scripts)
- [ ] Test rollback procedure before going live

---

## Part 7: Success Metrics

### Baseline (Current System - 2026-04-18)
- RAG latency: 500-2000ms per query
- Dispatcher calls: 3 per query
- Consolidator errors: ~2-3% (fact inconsistencies)
- Memory database size: Growing linearly

### Target (2026-06-30)
- RAG latency: **200-800ms avg, <15ms with cache hit** (target: 60-70% cache hit rate)
- Dispatcher calls: **1.5 avg** (40% reduction via early-exit)
- Consolidator errors: **<0.5%** (evaluator loop catches inconsistencies)
- Learning velocity: **Immediate** (no 30min wait for cache)
- Personality consistency: **High** (evaluator + metadata validation)

### Key Performance Indicators (KPIs)
1. **Cache Hit Rate**: Target ≥ 60%
2. **RAG Latency (P95)**: Target ≤ 800ms
3. **Evaluator Retry Rate**: Target ≤ 5%
4. **Affinity Accuracy**: Track user satisfaction over time
5. **System Error Rate**: Target ≤ 0.1%

---

## Part 8: Risk Assessment & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| Cache invalidation bug | Lost learning | MEDIUM | Extensive testing, feature flag |
| Evaluator too strict | Dropped facts | MEDIUM | Tuning, manual review |
| Database transaction overhead | Latency increase | LOW | Connection pooling, optimization |
| Depth classifier misclassification | Wrong dispatcher selection | MEDIUM | Fallback to MEDIUM, logging |
| Scheduled task failures | Missed promises | LOW | Task state recovery, alerts |
| Database schema migration issues | Data corruption | LOW | Backup + dry-run, rollback ready |

---

## Part 9: Dependencies & Constraints

### External Dependencies
- MongoDB 5.0+ (transactions support)
- FastAPI (for APIScheduler integration)
- numpy + scipy (for embedding similarity)
- LLM API availability (for depth classifier, evaluators)

### Internal Dependencies
- Existing RAG infrastructure (web_search_agent, memory_retriever_agent)
- Existing consolidator (global_state_updater, relationship_recorder)
- Existing database module (save_memory, update_affinity)

### Constraints
- Cannot break existing conversation flow (backward compatible)
- Local LLM limitations (cannot run 3 heavy dispatchers in parallel)
- Database write throughput (MongoDB write optimization needed)
- Memory efficiency (cache size grows with active users)

---

## Part 10: Future Work (Beyond This Plan)

### Phase 6: Offline Consolidation
- Nightly batch job to re-evaluate facts
- Personality-aware memory strengthening
- Implicit learning insights generation

### Phase 7: Analytics & Insights
- Personality traits evolution tracking
- Relationship dynamics visualization
- Recommendation engine for bot improvements

### Phase 8: Multi-User Personality
- Shared memory (facts about common knowledge)
- Multiple relationship personas (same bot, different users)

### Phase 9: Explainability
- Trace why bot made personality decision
- Audit memory sources (fact provenance)
- Personality consistency scores

---

## Appendix A: Configuration Reference

```python
# config.py additions

# RAG Cache Configuration
RAG_CACHE_BACKEND = "redis"  # or "memory"
RAG_CACHE_USER_FACTS_TTL = 1800  # 30 minutes
RAG_CACHE_INTERNAL_MEMORY_TTL = 900  # 15 minutes
RAG_CACHE_SIMILARITY_THRESHOLD = 0.82
RAG_CACHE_MAX_SIZE = 100_000  # items

# Depth Classification
DEPTH_CLASSIFIER_THRESHOLD_SHALLOW = 0.7
DEPTH_CLASSIFIER_THRESHOLD_MEDIUM = 0.5
DEPTH_CLASSIFIER_USE_LIGHT_LLM = True

# Consolidator
FACT_HARVESTER_MAX_RETRIES = 3
EVALUATOR_CONSISTENCY_CHECK = True

# Scheduled Tasks
SCHEDULED_TASKS_ENABLED = True
OFFLINE_CONSOLIDATION_CRON = "0 2 * * *"  # 2 AM daily
PERSONALITY_ANALYTICS_CRON = "0 8 * * 0"  # Sunday 8 AM

# Performance
RAG_CACHE_LATENCY_TARGET_MS = 15
RAG_DISPATCH_LATENCY_TARGET_MS = 2000
CONSOLIDATOR_LATENCY_TARGET_MS = 5000
```

---

## Appendix B: Database Migration Script Template

```python
# migrations/001_add_cache_metadata.py

async def up():
    """Add rag_cache_index and rag_metadata_index collections"""

    db = mongo_client["kazusa_bot_core"]

    # Create rag_cache_index
    await db.create_collection("rag_cache_index")
    await db.rag_cache_index.create_index([("embedding", "2dsphere")])
    await db.rag_cache_index.create_index([("ttl_expires_at", 1)], expireAfterSeconds=0)
    await db.rag_cache_index.create_index([("cache_type", 1)])

    # Create rag_metadata_index
    await db.create_collection("rag_metadata_index")
    await db.rag_metadata_index.create_index([("global_user_id", 1)])
    await db.rag_metadata_index.create_index([("rag_version", 1)])

    print("Migration 001 completed successfully")

async def down():
    """Rollback"""
    db = mongo_client["kazusa_bot_core"]
    await db.drop_collection("rag_cache_index")
    await db.drop_collection("rag_metadata_index")
    print("Migration 001 rolled back")
```

---

## Document History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-04-18 | 1.0 | System Design | Initial draft |
| TBD | 1.1 | Team | Add time estimates per task |
| TBD | 2.0 | Team | Post-implementation updates |

---

## Sign-Off

- [ ] Technical Lead: Approved for development
- [ ] Database Admin: Schema changes reviewed
- [ ] DevOps: Infrastructure ready
- [ ] Product: Requirements understood

---

**Document Owner**: System Architect
**Last Review**: 2026-04-18
**Next Review**: 2026-05-18
