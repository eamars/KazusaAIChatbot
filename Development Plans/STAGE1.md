
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
- ✅ `src/kazusa_ai_chatbot/rag/cache.py` — `DEFAULT_TTL_SECONDS` extended to 6 cache types (Stage 1.5a)

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

#### 1.5a. Update `src/kazusa_ai_chatbot/rag/cache.py` — Add New Cache Types ✅ IMPLEMENTED

**Purpose**: Update cache.py with new cache type definitions BEFORE Stage 3a tries to use them. This prevents forward dependencies.

**Why critical**: Stage 3a needs to store cache entries with `cache_type="character_diary"`, but Stage 1a only defines `"user_facts"` and `"internal_memory"`. Without this step, Stage 3a will get `KeyError: "character_diary not in DEFAULT_TTL_SECONDS"`.
**Purpose**: Define new cache type keys so downstream stages can use them
**Current state**: Only `"user_facts"` (1800s) and `"internal_memory"` (900s) defined
**Changes needed**:

Replace `DEFAULT_TTL_SECONDS` with:
```python
DEFAULT_TTL_SECONDS = {
    # User-related (per-user scoped)
    "character_diary": 1800,           # 30 min - character's subjective observations
    "objective_user_facts": 3600,      # 60 min - verified facts about user
    "user_promises": 900,              # 15 min - time-sensitive commitments
    
    # Conversation-related (per-user scoped)
    "internal_memory": 900,            # 15 min - conversation history
    
    # External (global scope)
    "external_knowledge": 3600,        # 1 hour - shared knowledge (web search)
    
    # Legacy (temporary, for backward compat during transition)
    "user_facts": 1800,                # DEPRECATED: see character_diary + objective_user_facts
}
```

**Why keep "user_facts"**: Stage 1a is already deployed with cache entries stored as `cache_type="user_facts"`. Removing it would make those entries unfindable. Keep it through Stage 4a (consolidator deployment), then remove it.

**Impact on RAGCache**: No changes to RAGCache code. The cache doesn't validate cache_type keys — it just uses them as dict keys. This is intentional (cache is generic).

**Dependencies**: Modifies Stage 1a file only (cache.py)
**Lines of code**: 8 lines changed in DEFAULT_TTL_SECONDS
**test_main()**: No new tests needed; Stage 1a tests already cover this

**Backward Compatibility**: 
- ✓ Old code using `cache_type="user_facts"` still works
- ✓ New code can use `cache_type="character_diary"` without errors
- ✓ No database changes needed
- ✓ Can deploy independently after Stage 1a

---

#### 1d. Design External Knowledge Caching Strategy (PLANNING)
**Purpose**: Define caching policy for `external_rag` results (web search) — currently not cached, but the cache architecture must support it.

**Current state**: RAGCache only defines two cache_types (`user_facts`, `internal_memory`). External web search results are fetched fresh on every query with no deduplication across users, even when queries are identical.

**Problem**: Three knowledge sources have conflicting cache requirements. External knowledge (web search results) is fundamentally different from user-scoped knowledge:

| Source | Scope | Stability | Volume | Sharing |
|--------|-------|-----------|--------|---------|
| user_facts | Per-user | High (days) | 10-100/user | No sharing |
| internal_memory | Per-user | Medium (hours) | 50-500/user | No sharing |
| external_knowledge | **Global** | Low (hours→min) | Unbounded | **Shared** |

**Design Decision: Global Cache (Shared Across All Users)** ✅ LOCKED

**Rationale**: When the AI learns knowledge from the internet during conversation (weather facts, documentation, news, scientific information), this knowledge is universally applicable. Like a human learning something useful and sharing it with others, the system applies the same learned knowledge to all users. This maximizes efficiency and aligns with shared human knowledge principles.

**Changes needed**:

**1. Add `external_knowledge` cache type to `cache.py`**:
```python
DEFAULT_TTL_SECONDS = {
    "user_facts": 1800,           # Per-user, 30 min (character's diary)
    "internal_memory": 900,       # Per-user, 15 min (conversation history)
    "external_knowledge": 3600,   # GLOBAL, 1 hour (shared by all users)
}
```

**2. Usage pattern — Global scope with empty `global_user_id`**:
```python
# Storage (external_rag writes):
await cache.store(
    embedding=query_embedding,
    results=external_rag_results,
    cache_type="external_knowledge",
    global_user_id="",  # ← EMPTY: Global, not per-user
    ttl_seconds=3600
)

# Retrieval (cache check phase):
# Alice: "What is the capital of France?"
#   → Cache stores with global_user_id=""
# Bob: "What's the capital of France?"
#   → Cache HIT: Same entry returned to both users
```

**Collection**: Reuses existing `rag_cache_index` — no new collection needed.

**Indices**: Existing indices support global scope (already handles `global_user_id=""`).

**Dependencies**: None (design-only stage)
**Lines of code**: 0 (design-only; implementation in Stage 3a)
**Impact on RAGCache**: Minimal — already supports `global_user_id=""`, just needs config addition

**Implementation Checklist**:

*Phase 1: Cache Type Definition*
- [x] Add `"external_knowledge": 3600` to `DEFAULT_TTL_SECONDS` in `src/kazusa_ai_chatbot/rag/cache.py`
- [x] Verify TTL is 3600 seconds (1 hour)
- [x] Add comment: `# GLOBAL, 1 hour (shared by all users)`
- [x] Run `tests/test_rag_cache.py` to verify config is recognized

*Phase 2: RAGCache Documentation*
- [x] Update `RAGCache.retrieve_if_similar()` docstring:
  - Add: `global_user_id=""` means global/shared scope (all users can access)
  - Add: `global_user_id=user_id` means per-user scope (only that user)
  - Add example: `# Global: await cache.retrieve_if_similar(..., global_user_id="")`
  - Add example: `# Per-user: await cache.retrieve_if_similar(..., global_user_id="alice_123")`
- [x] Update `RAGCache.store()` docstring:
  - Add warning: Setting `global_user_id=""` shares results across all users

*Phase 3: Testing (Stage 1d)*
- [ ] Add test case to `tests/test_rag_cache.py`:
  ```python
  async def test_global_external_knowledge_shared_across_users():
      """Verify external knowledge cache is shared (not user-scoped)."""
      cache = RAGCache()
      await cache.start()
      
      # User A stores external knowledge
      await cache.store(
          embedding=[0.1, 0.2, ...],
          results={"answer": "Paris is the capital of France"},
          cache_type="external_knowledge",
          global_user_id="",  # GLOBAL
          ttl_seconds=3600
      )
      
      # User B retrieves with identical embedding
      hit = await cache.retrieve_if_similar(
          embedding=[0.1, 0.2, ...],
          cache_type="external_knowledge",
          global_user_id=""  # GLOBAL - same scope
      )
      
      assert hit is not None, "User B should share User A's external knowledge"
      assert hit["results"]["answer"] == "Paris is the capital of France"
  ```
- [ ] Run: `pytest tests/test_rag_cache.py::test_global_external_knowledge_shared_across_users -v`

*Phase 4: Database Indices (Stage 2b)*
- [x] Add to `db/bootstrap.py` for `rag_cache_index`:
  - [x] Index: `{cache_type: 1, global_user_id: 1}` for global queries
  - [x] Verify TTL index: `{ttl_expires_at: 1}` with `expireAfterSeconds=0`

*Phase 5: Integration (Stage 3a)*
- [ ] `persona_supervisor2_rag.py` — external_rag phase:
  - [ ] Store with `cache_type="external_knowledge"`, `global_user_id=""`
- [ ] `persona_supervisor2_rag.py` — cache check phase:
  - [ ] Retrieve with `cache_type="external_knowledge"`, `global_user_id=""`

*Phase 6: Code Review Validation*
- [ ] `global_user_id=""` used ONLY for `external_knowledge`
- [ ] `global_user_id=""` NEVER used for `user_facts` or `internal_memory`
- [ ] Cache invalidation does NOT apply to external_knowledge (expires naturally)
- [ ] No user-specific data stored with `global_user_id=""`

---

#### 1e. Separate Cache Types for Distinct Knowledge Categories (PLANNING)
**Purpose**: Clarify semantic distinction between three user-related knowledge types that are currently conflated in a single `"user_facts"` cache_type.

**Current state**: The consolidator produces three distinct output types stored separately, but the cache treats them identically:
- `diary_entry` → `user_profiles.facts` (character's subjective observations)
- `new_facts` → `memory` collection (objective, verified facts)
- `future_promises` → `memory` collection (time-bound commitments)

All three are cached as `cache_type="user_facts"` (ambiguous).

**Problem**: Single cache type prevents precise invalidation:
- When character learns new diary entry, should NOT invalidate old facts
- When user shares a verifiable fact, should NOT invalidate diary entries
- When promise is fulfilled, should NOT invalidate either facts or diary

Current ambiguity means one change invalidates all three, causing unnecessary cache flushes.

**Design Decision: Three Semantic Cache Types** ✅ RECOMMENDED

**Rationale**: Cache effectiveness depends on granular invalidation. By distinguishing subjective observations (diary), objective facts (facts), and commitments (promises) at the cache layer, we enable selective invalidation and reduce false cache misses.

**Changes needed**:

**1. Define three cache types with distinct semantics and TTLs**:
```python
DEFAULT_TTL_SECONDS = {
    # User-related (per-user, scoped to specific user)
    "character_diary": 1800,           # 30 min - character's subjective observations
    "objective_user_facts": 3600,      # 60 min - verified facts about user
    "user_promises": 900,              # 15 min - time-sensitive commitments
    
    # Conversation-related (per-user, existing)
    "internal_memory": 900,            # 15 min - conversation history
    
    # External (global, see Section 1d)
    "external_knowledge": 3600,        # 1 hour - shared knowledge
}
```

| Type | Content | TTL | Invalidation Trigger |
|------|---------|-----|----------------------|
| `character_diary` | "User seems happy", "User is interesting" | 30 min | New diary_entry written |
| `objective_user_facts` | "User is engineer", "User lives in Tokyo" | 60 min | New facts extracted + validated |
| `user_promises` | "I promised to help", "I'll send recipe" | 15 min | Promise fulfilled or due_time expires |

**2. Selective invalidation based on cache type**:
```python
# When character writes new diary:
await cache.invalidate_pattern("character_diary", global_user_id)
# ✓ Clears: old diary observations
# ✗ Keeps: objective facts, promises

# When new facts extracted:
await cache.invalidate_pattern("objective_user_facts", global_user_id)
# ✓ Clears: old facts
# ✗ Keeps: diary, promises

# When promise fulfilled:
await cache.invalidate_pattern("user_promises", global_user_id)
# ✓ Clears: old promises
# ✗ Keeps: diary, facts
```

**Collection**: Reuses existing `rag_cache_index` — no new collection needed.

**Indices**: Existing indices support all types (already per-user scoped).

**Dependencies**: Section 1d (external_knowledge design); Section 2a-ii (database schema migration)
**Lines of code**: 0 (design-only; implementation in Stage 3a + Stage 4a)
**Impact on RAGCache**: None — already supports multiple cache_types generically

**Implementation Checklist**:

*Phase 1: Cache Type Configuration (Stage 1)*
- [x] Add three new cache_types to `DEFAULT_TTL_SECONDS` in `src/kazusa_ai_chatbot/rag/cache.py`:
  - [x] `"character_diary": 1800`
  - [x] `"objective_user_facts": 3600`
  - [x] `"user_promises": 900`
- [x] Add comments explaining each type's purpose
- [x] Run `tests/test_rag_cache.py` to verify all types are recognized

*Phase 2: RAGCache Documentation (Stage 1)*
- [x] Update docstrings to document the three types:
  - [x] Purpose and usage for each
  - [x] Invalidation semantics (selective, not all-or-nothing)
  - [x] TTL rationale (diary < facts < promises)

*Phase 3: Database Schema (Stage 2a-ii)*
- [x] Migrate `user_profiles.facts` → separate `character_diary` and `objective_facts` fields
- [x] Add metadata per entry: `timestamp`, `confidence`, `category`, `source`
- [x] Add vector indices: `diary_embedding`, `facts_embedding`

*Phase 4: Consolidator Integration (Stage 4a)*
- [ ] Update `relationship_recorder()`:
  - [ ] Call `upsert_character_diary()` instead of `upsert_user_facts()`
  - [ ] Include `timestamp`, `confidence`, `context` metadata
- [ ] Update `facts_harvester()`:
  - [ ] Call `upsert_objective_facts()` for new facts
  - [ ] Include `category`, `source`, `confidence` metadata
  - [ ] Add deduplication logic
- [ ] Update cache invalidation:
  - [ ] `await cache.invalidate_pattern("character_diary", global_user_id)` after diary write
  - [ ] `await cache.invalidate_pattern("objective_user_facts", global_user_id)` after facts write
  - [ ] `await cache.invalidate_pattern("user_promises", global_user_id)` after promise fulfillment

*Phase 5: RAG Integration (Stage 3a)*
- [ ] Update `persona_supervisor2_rag.py` cache check to retrieve all three types:
  ```python
  diary_hit = await cache.retrieve_if_similar(..., cache_type="character_diary", ...)
  facts_hit = await cache.retrieve_if_similar(..., cache_type="objective_user_facts", ...)
  promises_hit = await cache.retrieve_if_similar(..., cache_type="user_promises", ...)
  ```
- [ ] Update RAG store to save results with proper type:
  - [ ] Store opinions as `character_diary`
  - [ ] Store facts as `objective_user_facts`

*Phase 6: Testing (Stage 3a + 4a)*
- [ ] Add test: `test_character_diary_invalidation_does_not_affect_facts()`
- [ ] Add test: `test_objective_facts_invalidation_does_not_affect_diary()`
- [ ] Add test: `test_promise_invalidation_independent_from_facts()`
- [ ] Run: `pytest tests/test_rag_cache.py -v -k "invalidation"`

*Phase 7: Code Review Validation*
- [ ] Each cache store call specifies correct type
- [ ] Each cache retrieve call specifies correct type
- [ ] Invalidation calls are selective (not `clear_all_user()`)
- [ ] Consolidator writes use new semantic functions
- [ ] No cross-type confusion in code logic
