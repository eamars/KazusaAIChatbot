# Personality Simulation Architecture: Test Plan

**Status**: DRAFT
**Last Updated**: 2026-04-18
**Test Lead**: TBD
**Target Test Start**: Week 3 (Phase 2 parallel start)
**Test Completion**: Week 10 (before Canary rollout)

---

## Executive Summary

This test plan validates that the 5-phase personality system achieves performance targets while maintaining system reliability and user experience quality.

**Key Objectives**:
1. Verify cache layer delivers 60-70% hit rate with <15ms latency
2. Validate depth classifier correctly routes queries (SHALLOW/MEDIUM/DEEP)
3. Confirm evaluator loop catches 95%+ of fact contradictions
4. Ensure database schema changes don't increase p95 latency
5. Load test: System handles 10 concurrent conversations without degradation
6. Personality consistency: Learned facts appear in next conversation correctly

---

## Part 1: Test Strategy

### Test Pyramid

```
         ▲
         │  E2E Tests (5%)
         │  - Full conversation flows
         │  - Personality continuity
         ├───────────────────
         │  Integration Tests (20%)
         │  - Phase interactions
         │  - Cache + RAG + DB
         ├───────────────────
         │  Unit Tests (75%)
         │  - Component behavior
         │  - Edge cases
         ▼
```

### Testing Timeline

```
Week 3:    Unit tests written + Phase 2 implementation
Week 4:    Unit tests passing, integration tests written
Week 5:    Integration tests passing, E2E tests written
Week 6:    E2E tests passing, performance baseline established
Week 7-8:  Performance & load tests
Week 9:    Regression tests, UAT preparation
Week 10:   Sign-off, canary deployment ready
```

### Test Environments

```
LOCAL:       Developer machines (unit tests)
STAGING:     Dedicated MongoDB + API server
             - Full schema, realistic data volume (50K users, 1M conversations)
             - Load testing tools installed
             - Monitoring enabled

PRODUCTION:  Canary traffic (5% → 25% → 50% → 100%)
             - Real user data
             - Live monitoring
             - Rollback ready
```

---

## Part 2: Unit Tests (Per Component)

### 2.1 RAGCache Unit Tests

**File**: `tests/unit/test_rag_cache.py`

```python
class TestRAGCacheStore:
    """Test storing and retrieving cached results"""

    async def test_store_and_retrieve_exact_match():
        """Same embedding → exact retrieval"""
        cache = RAGCache()
        embedding = [0.1, 0.2, ..., 0.9]  # 768-dim
        results = {"user_facts": "likes sushi"}

        await cache.store(embedding, results, "user_facts", 1800, {})
        retrieved = await cache.retrieve_if_similar(embedding, "user_facts", 0.82)

        assert retrieved is not None
        assert retrieved["user_facts"] == "likes sushi"

    async def test_similarity_check_threshold():
        """Embedding similarity > threshold → hit, else miss"""
        cache = RAGCache()
        emb1 = [0.1, 0.2, ..., 0.9]
        emb2 = [0.10001, 0.20001, ..., 0.90001]  # 99.9% cosine similarity

        await cache.store(emb1, {"result": "A"}, "user_facts", 1800, {})

        # High similarity → hit
        retrieved = await cache.retrieve_if_similar(emb2, "user_facts", threshold=0.98)
        assert retrieved is not None

        # Low similarity → miss
        emb3 = [0.9, 0.8, ..., 0.1]  # Opposite embedding
        retrieved = await cache.retrieve_if_similar(emb3, "user_facts", threshold=0.98)
        assert retrieved is None

    async def test_ttl_expiry():
        """Cached entry expires after TTL"""
        cache = RAGCache()
        embedding = [0.1, 0.2, ..., 0.9]

        await cache.store(embedding, {"result": "A"}, "user_facts", 1)  # 1 sec TTL

        # Immediately: Should exist
        retrieved = await cache.retrieve_if_similar(embedding, "user_facts", 0.82)
        assert retrieved is not None

        # After TTL: Should not exist
        await asyncio.sleep(1.5)
        retrieved = await cache.retrieve_if_similar(embedding, "user_facts", 0.82)
        assert retrieved is None

    async def test_invalidate_pattern():
        """Invalidate by global_user_id + cache_type"""
        cache = RAGCache()
        user_id = "user123"
        emb1 = [0.1, 0.2, ..., 0.9]

        # Store 3 cache entries for user123
        await cache.store(emb1, {"A": 1}, "user_facts", 1800, {"global_user_id": user_id})
        await cache.store(emb1, {"B": 2}, "internal_memory", 900, {"global_user_id": user_id})
        await cache.store(emb1, {"C": 3}, "external_search", 14400, {"global_user_id": user_id})

        # Invalidate only user_facts
        await cache.invalidate_pattern("user_facts", user_id, "test_trigger")

        # user_facts gone, others remain
        assert await cache.retrieve_if_similar(emb1, "user_facts", 0.82) is None
        assert await cache.retrieve_if_similar(emb1, "internal_memory", 0.82) is not None
        assert await cache.retrieve_if_similar(emb1, "external_search", 0.82) is not None

    async def test_clear_all_user():
        """Clear all cache for a user"""
        cache = RAGCache()
        user_id = "user123"
        emb1 = [0.1, 0.2, ..., 0.9]

        # Store for user123
        await cache.store(emb1, {"A": 1}, "user_facts", 1800, {"global_user_id": user_id})
        await cache.store(emb1, {"B": 2}, "internal_memory", 900, {"global_user_id": user_id})

        # Clear all for user123
        await cache.clear_all_user(user_id)

        # All gone
        assert await cache.retrieve_if_similar(emb1, "user_facts", 0.82) is None
        assert await cache.retrieve_if_similar(emb1, "internal_memory", 0.82) is None

    async def test_cache_stats():
        """Track hits, misses, size"""
        cache = RAGCache()
        emb1 = [0.1, 0.2, ..., 0.9]

        await cache.store(emb1, {"result": "A"}, "user_facts", 1800, {})

        # Hit
        await cache.retrieve_if_similar(emb1, "user_facts", 0.82)

        # Miss (wrong cache_type)
        await cache.retrieve_if_similar(emb1, "internal_memory", 0.82)

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
```

**Coverage**: 5 tests, all critical paths

---

### 2.2 InputDepthClassifier Unit Tests

**File**: `tests/unit/test_depth_classifier.py`

```python
class TestInputDepthClassifier:
    """Test query depth classification"""

    async def test_shallow_query():
        """Simple fact questions → SHALLOW"""
        classifier = InputDepthClassifier()

        test_cases = [
            "What's your name?",
            "How old are you?",
            "What's your favorite color?",
            "Where do you live?",
        ]

        for input_str in test_cases:
            result = await classifier.classify(input_str, "greeting", affinity=500)
            assert result["depth"] == "SHALLOW"
            assert "user_rag" in result["trigger_dispatchers"]
            assert "internal_rag" not in result["trigger_dispatchers"]

    async def test_medium_query():
        """Contextual recall questions → MEDIUM"""
        classifier = InputDepthClassifier()

        test_cases = [
            "Do you remember when we talked about X?",
            "What did I tell you about my job?",
            "Remember the promise you made?",
            "What have we discussed before?",
        ]

        for input_str in test_cases:
            result = await classifier.classify(input_str, "conversation", affinity=500)
            assert result["depth"] == "MEDIUM"
            assert "user_rag" in result["trigger_dispatchers"]
            assert "internal_rag" in result["trigger_dispatchers"]
            assert "external_rag" not in result["trigger_dispatchers"]

    async def test_deep_query():
        """Complex reasoning → DEEP"""
        classifier = InputDepthClassifier()

        test_cases = [
            "Based on our relationship, why do you think I'm important?",
            "How has your feeling about me changed over time?",
            "What's your honest assessment of my personality?",
        ]

        for input_str in test_cases:
            result = await classifier.classify(input_str, "relationship", affinity=500)
            assert result["depth"] == "DEEP"
            assert "user_rag" in result["trigger_dispatchers"]
            assert "internal_rag" in result["trigger_dispatchers"]
            assert "external_rag" in result["trigger_dispatchers"]

    async def test_affinity_influences_depth():
        """Low affinity → more careful retrieval (deeper)"""
        classifier = InputDepthClassifier()
        input_str = "Tell me something about yourself"

        # High affinity: SHALLOW
        result_high = await classifier.classify(input_str, "chat", affinity=900)

        # Low affinity: DEEP (more careful)
        result_low = await classifier.classify(input_str, "chat", affinity=200)

        assert result_high["depth"] in ["SHALLOW", "MEDIUM"]
        assert result_low["depth"] in ["DEEP"]

    async def test_temporal_markers_increase_depth():
        """Temporal words → MEDIUM+ """
        classifier = InputDepthClassifier()

        # Without temporal: SHALLOW default
        r1 = await classifier.classify("What's your name?", "chat", affinity=500)

        # With temporal: MEDIUM+
        r2 = await classifier.classify("What did you do yesterday?", "chat", affinity=500)

        # Depth should be >= original
        depth_order = {"SHALLOW": 1, "MEDIUM": 2, "DEEP": 3}
        assert depth_order[r2["depth"]] >= depth_order[r1["depth"]]

    async def test_fallback_to_medium():
        """Ambiguous input → default to MEDIUM (safe)"""
        classifier = InputDepthClassifier()

        ambiguous_input = "xyzabc 123 !@#"
        result = await classifier.classify(ambiguous_input, "chat", affinity=500)

        assert result["depth"] == "MEDIUM"  # Safe default

    async def test_performance_under_100ms():
        """Classifier should be fast (no LLM calls most of the time)"""
        classifier = InputDepthClassifier()

        import time
        start = time.time()
        for _ in range(100):
            await classifier.classify("What's your name?", "chat", affinity=500)
        elapsed_ms = (time.time() - start) * 1000 / 100

        assert elapsed_ms < 100  # Average per query
```

**Coverage**: 7 tests, all heuristic paths

---

### 2.3 fact_harvester_evaluator Unit Tests

**File**: `tests/unit/test_evaluator.py`

```python
class TestFactHarvesterEvaluator:
    """Test consistency validation"""

    async def test_pass_consistent_facts():
        """New fact aligns with research_facts → PASS"""
        evaluator = FactHarvesterEvaluator()

        new_facts = [
            {
                "entity": "user123",
                "category": "food_preference",
                "description": "user123 likes sushi"
            }
        ]

        research_facts = {
            "user_rag_finalized": ["user123 eats Japanese food"],
            "internal_rag_results": []
        }

        result = await evaluator.evaluate(new_facts, [], research_facts, "user123")

        assert result["should_stop"] is True
        assert "pass" in result["feedback"].lower()

    async def test_fail_contradictory_facts():
        """New fact contradicts research_facts → FAIL"""
        evaluator = FactHarvesterEvaluator()

        new_facts = [
            {
                "entity": "user123",
                "category": "occupation",
                "description": "user123 is a teacher"
            }
        ]

        research_facts = {
            "user_rag_finalized": ["user123 is a doctor and practices medicine"],
            "internal_rag_results": []
        }

        result = await evaluator.evaluate(new_facts, [], research_facts, "user123")

        assert result["should_stop"] is False
        assert "contradict" in result["feedback"].lower()

    async def test_fail_format_violation():
        """New fact violates format (e.g., no category) → FAIL"""
        evaluator = FactHarvesterEvaluator()

        new_facts = [
            {
                "entity": "user123",
                # Missing "category"
                "description": "user123 likes sushi"
            }
        ]

        result = await evaluator.evaluate(new_facts, [], {}, "user123")

        assert result["should_stop"] is False
        assert "category" in result["feedback"].lower()

    async def test_pass_empty_facts_ok():
        """No new facts is valid (not an error)"""
        evaluator = FactHarvesterEvaluator()

        result = await evaluator.evaluate([], [], {}, "user123")

        assert result["should_stop"] is True
        assert "pass" in result["feedback"].lower()

    async def test_fail_identity_mismatch():
        """Fact about wrong user → FAIL"""
        evaluator = FactHarvesterEvaluator()

        new_facts = [
            {
                "entity": "other_user",  # Not the target user
                "category": "preference",
                "description": "other_user likes pizza"
            }
        ]

        result = await evaluator.evaluate(new_facts, [], {}, "user123")

        assert result["should_stop"] is False
        assert "identity" in result["feedback"].lower()

    async def test_promise_without_timestamp_ok():
        """Promise without due_time is allowed (null is valid)"""
        evaluator = FactHarvesterEvaluator()

        futures = [
            {
                "target": "user123",
                "action": "will help user123 with project",
                "due_time": None  # OK
            }
        ]

        result = await evaluator.evaluate([], futures, {}, "user123")

        assert result["should_stop"] is True

    async def test_max_retries_exhausted():
        """After 3 retries, force should_stop=True"""
        evaluator = FactHarvesterEvaluator()

        # This would be called with retry_count in real code
        result = await evaluator.evaluate(
            new_facts=[{"invalid": "format"}],
            futures=[],
            research_facts={},
            user_name="user123",
            retry_count=3  # Max retries
        )

        assert result["should_stop"] is True  # Force stop
```

**Coverage**: 8 tests, critical validation paths

---

### 2.4 Database Operations Unit Tests

**File**: `tests/unit/test_db_operations.py`

```python
class TestDatabaseOperations:
    """Test cache and metadata DB operations"""

    async def test_insert_cache_entry():
        """Insert new cache entry"""
        db = MongoDBTestClient()

        result = await db.insert_cache_entry(
            embedding=[0.1, 0.2, ..., 0.9],
            cache_type="user_facts",
            ttl_seconds=1800,
            metadata={"user_id": "user123"}
        )

        assert result.inserted_id is not None

    async def test_find_similar_embeddings():
        """Vector similarity search"""
        db = MongoDBTestClient()

        # Insert test embeddings
        emb1 = [0.1, 0.2, ..., 0.9]
        emb2 = [0.10001, 0.20001, ..., 0.90001]  # 99.9% similar
        emb3 = [0.9, 0.8, ..., 0.1]  # Opposite

        await db.insert_cache_entry(emb1, "user_facts", 1800, {})
        await db.insert_cache_entry(emb3, "user_facts", 1800, {})

        # Search with emb2 (should match emb1)
        results = await db.find_similar_embeddings(
            embedding=emb2,
            cache_type="user_facts",
            similarity_threshold=0.98
        )

        assert len(results) >= 1
        assert results[0] contains emb1 data

    async def test_increment_rag_version():
        """Version increment on writes"""
        db = MongoDBTestClient()
        user_id = "user123"

        v1 = await db.get_rag_version(user_id)
        assert v1 == 0  # First time

        await db.increment_rag_version(user_id)
        v2 = await db.get_rag_version(user_id)
        assert v2 == 1

        await db.increment_rag_version(user_id)
        v3 = await db.get_rag_version(user_id)
        assert v3 == 2

    async def test_vector_index_performance():
        """Vector index lookup is fast"""
        db = MongoDBTestClient()

        # Insert 1000 embeddings
        for i in range(1000):
            await db.insert_cache_entry(
                embedding=generate_random_embedding(),
                cache_type="user_facts",
                ttl_seconds=1800,
                metadata={"i": i}
            )

        # Search should be fast
        import time
        start = time.time()
        await db.find_similar_embeddings(
            embedding=generate_random_embedding(),
            cache_type="user_facts",
            similarity_threshold=0.82
        )
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 50  # Vector index should be fast
```

**Coverage**: 4 tests, database operations

---

## Part 3: Integration Tests

### 3.1 RAG + Cache Integration

**File**: `tests/integration/test_rag_cache_integration.py`

```python
class TestRAGCacheIntegration:
    """Test RAG with cache layer"""

    async def test_cache_miss_runs_rag():
        """No cache → full RAG dispatch"""
        # Setup: Empty cache
        rag_system = RAGSystem(cache=RAGCache())

        state = {
            "decontexualized_input": "What's my favorite food?",
            "user_topic": "greeting",
            "input_embedding": generate_embedding(...),
            ...
        }

        result = await rag_system.call_rag_subgraph(state)

        # Should have run RAG
        assert result["cache_hit"] is False
        assert result["research_facts"]["user_rag_finalized"] is not None

    async def test_cache_hit_skips_rag():
        """Cache hit → skip RAG dispatchers"""
        cache = RAGCache()
        embedding = generate_embedding("What's my favorite food?")
        cached_facts = {"user_facts": "likes sushi"}

        # Pre-populate cache
        await cache.store(embedding, cached_facts, "user_facts", 1800, {})

        rag_system = RAGSystem(cache=cache)

        state = {
            "decontexualized_input": "What's my favorite food?",
            "input_embedding": embedding,
            ...
        }

        result = await rag_system.call_rag_subgraph(state)

        # Should have hit cache
        assert result["cache_hit"] is True
        assert result["research_facts"] == cached_facts
        # Verify dispatchers were not called (check logs or metrics)

    async def test_cache_stores_rag_results():
        """After RAG, results stored in cache"""
        cache = RAGCache()
        rag_system = RAGSystem(cache=cache)

        embedding = generate_embedding("What's my favorite food?")

        state = {
            "decontexualized_input": "What's my favorite food?",
            "input_embedding": embedding,
            ...
        }

        result = await rag_system.call_rag_subgraph(state)

        # Immediately repeat query with same embedding
        result2 = await rag_system.call_rag_subgraph(state)

        # Second should hit cache
        assert result2["cache_hit"] is True

    async def test_cache_invalidation_on_new_facts():
        """New facts written → cache invalidated"""
        cache = RAGCache()
        user_id = "user123"
        embedding = generate_embedding("food question")

        # Pre-populate cache
        await cache.store(embedding, {"old": "data"}, "user_facts", 1800,
                         {"global_user_id": user_id})

        # Simulate: consolidator writes new facts
        await consolidator.call_consolidation_subgraph({...})  # Writes new facts

        # Cache should be invalidated
        cached = await cache.retrieve_if_similar(embedding, "user_facts", 0.82)
        assert cached is None  # Invalidated
```

**Coverage**: 4 integration scenarios

---

### 3.2 Depth Classifier + Dispatcher Conditional Routing

**File**: `tests/integration/test_depth_routing.py`

```python
class TestDepthRouting:
    """Test depth-aware selective dispatch"""

    async def test_shallow_skips_internal_rag():
        """SHALLOW input → user_rag only, skip internal_rag"""
        graph = build_rag_graph()

        state = {
            "decontexualized_input": "What's your favorite color?",
            "metadata": {"depth": "SHALLOW"},
            ...
        }

        result = await graph.ainvoke(state)

        # Verify internal_rag was not called
        assert result["internal_rag_next_action"] == "end"  # Not run
        assert result["user_rag_results"] is not None

    async def test_medium_conditionally_runs_internal():
        """MEDIUM input → user_rag + conditional internal_rag"""
        graph = build_rag_graph()

        state = {
            "decontexualized_input": "Do you remember the promise about X?",
            "metadata": {"depth": "MEDIUM"},
            ...
        }

        result = await graph.ainvoke(state)

        # Both should run
        assert result["user_rag_results"] is not None
        # internal_rag may run depending on user_rag confidence

    async def test_deep_runs_all():
        """DEEP input → user_rag + internal_rag + external_rag"""
        graph = build_rag_graph()

        state = {
            "decontexualized_input": "Based on our relationship, why...?",
            "metadata": {"depth": "DEEP"},
            ...
        }

        result = await graph.ainvoke(state)

        # All should run
        assert result["user_rag_results"] is not None
        assert result["internal_rag_next_action"] != "end"
        assert result["external_rag_next_action"] != "end"

    async def test_early_exit_on_high_confidence():
        """User RAG high confidence + MEDIUM depth → skip internal_rag"""
        graph = build_rag_graph()

        state = {
            "decontexualized_input": "What's your favorite food?",
            "metadata": {"depth": "MEDIUM"},
            # Simulate high confidence user_rag result
            "user_confidence_score": 0.95,
            ...
        }

        result = await graph.ainvoke(state)

        # internal_rag should be skipped
        assert result["internal_rag_next_action"] == "end"
```

**Coverage**: 4 depth-aware routing scenarios

---

### 3.3 Consolidator Evaluator Loop

**File**: `tests/integration/test_evaluator_loop.py`

```python
class TestEvaluatorLoop:
    """Test facts_harvester → evaluator feedback loop"""

    async def test_valid_facts_pass_first_try():
        """Good facts → Pass on first evaluation"""
        graph = build_consolidator_graph()

        state = {
            "decontexualized_input": "I live in Auckland",
            "research_facts": {"user_rag_finalized": ["..."]},
            "action_directives": {...},
            ...
        }

        result = await graph.ainvoke(state)

        # Should pass evaluator on first try
        assert result["fact_harvester_retry"] == 1
        assert result["new_facts"] is not None

    async def test_contradictory_facts_loop_and_fix():
        """Bad facts → Evaluator rejects → facts_harvester retries"""
        graph = build_consolidator_graph()

        # Mock facts_harvester to initially return contradiction
        # Then fix on retry

        state = {
            "decontexualized_input": "I'm a teacher but also I'm a doctor",
            "research_facts": {"user_rag_finalized": ["User is a doctor"]},
            ...
        }

        result = await graph.ainvoke(state)

        # Should retry
        assert result["fact_harvester_retry"] >= 2
        # Final result should be fixed
        assert result["new_facts"] doesn't contain "teacher"

    async def test_max_retries_stops_loop():
        """After 3 retries → Force stop evaluator"""
        graph = build_consolidator_graph()

        # Mock facts_harvester to always return bad facts
        with patch("facts_harvester", side_effect=lambda s: {
            "new_facts": [{"invalid": "format"}],
            "future_promises": []
        }):
            state = {...}
            result = await graph.ainvoke(state)

        # Should stop at max retries
        assert result["fact_harvester_retry"] == MAX_RETRIES
        assert result["should_stop"] is True

    async def test_metadata_propagates_through_loop():
        """Metadata accumulates through evaluator retries"""
        graph = build_consolidator_graph()

        state = {
            "metadata": {"depth": "MEDIUM", "cache_hit": False},
            ...
        }

        result = await graph.ainvoke(state)

        # Metadata should have additional fields
        assert result["metadata"]["extraction_confidence"] is not None
        assert result["metadata"]["evaluator_passes"] >= 1
```

**Coverage**: 4 evaluator loop scenarios

---

## Part 4: Performance Tests

### 4.1 Cache Performance Benchmarks

**File**: `tests/performance/test_cache_performance.py`

**Metrics**:
- Cache lookup latency (target: <15ms avg, <50ms p95)
- Cache insertion latency (target: <20ms)
- Hit rate under repetition (target: >60%)
- Memory usage (track growth pattern)

```python
class TestCachePerformance:
    """Benchmark cache operations"""

    async def test_cache_lookup_latency():
        """Measure: Cache lookup time"""
        cache = RAGCache()
        embedding = generate_embedding(...)

        await cache.store(embedding, {"data": "..."},  "user_facts", 1800, {})

        # Warm up
        await cache.retrieve_if_similar(embedding, "user_facts", 0.82)

        # Benchmark: 1000 lookups
        import time
        start = time.time()
        for _ in range(1000):
            await cache.retrieve_if_similar(embedding, "user_facts", 0.82)
        elapsed_ms = (time.time() - start) * 1000 / 1000

        # Publish metrics
        assert elapsed_ms < 15, f"Cache lookup too slow: {elapsed_ms}ms"
        print(f"Cache lookup: {elapsed_ms:.2f}ms avg")

    async def test_cache_insertion_latency():
        """Measure: Cache insertion time"""
        cache = RAGCache()

        import time
        start = time.time()
        for i in range(1000):
            await cache.store(
                generate_embedding(...),
                {"data": f"result_{i}"},
                "user_facts",
                1800,
                {}
            )
        elapsed_ms = (time.time() - start) * 1000 / 1000

        assert elapsed_ms < 20, f"Cache insertion too slow: {elapsed_ms}ms"
        print(f"Cache insertion: {elapsed_ms:.2f}ms avg")

    async def test_hit_rate_under_repetition():
        """Measure: Cache hit rate on repetitive queries"""
        cache = RAGCache()

        # Simulate 100 conversations with 10 queries each
        # 70% of queries are repetitions within same conversation

        hit_count = 0
        total_count = 0

        for conv in range(100):
            for query in range(10):
                embedding = generate_embedding(f"conversation_{conv}_query_{query % 7}")  # 70% repeat

                if conv > 0 and query < 7:
                    # Should hit from previous conversation
                    result = await cache.retrieve_if_similar(embedding, "user_facts", 0.82)
                    total_count += 1
                    if result:
                        hit_count += 1

                await cache.store(embedding, {"result": "..."}, "user_facts", 1800, {})

        hit_rate = hit_count / total_count if total_count > 0 else 0
        assert hit_rate > 0.60, f"Cache hit rate too low: {hit_rate:.1%}"
        print(f"Cache hit rate: {hit_rate:.1%}")

    async def test_cache_memory_growth():
        """Measure: Memory usage as cache grows"""
        cache = RAGCache()

        # Insert 10K cache entries
        import psutil
        process = psutil.Process()

        initial_mb = process.memory_info().rss / 1024 / 1024

        for i in range(10000):
            await cache.store(
                generate_embedding(...),
                {"data": f"result_{i}" * 10},  # ~100 bytes per entry
                "user_facts",
                1800,
                {}
            )

        final_mb = process.memory_info().rss / 1024 / 1024
        growth_mb = final_mb - initial_mb

        # 10K entries of ~100 bytes = ~1MB expected
        assert growth_mb < 50, f"Cache memory growth too high: {growth_mb}MB"
        print(f"Memory growth for 10K entries: {growth_mb}MB")
```

---

### 4.2 RAG Latency Benchmarks

**File**: `tests/performance/test_rag_latency.py`

**Metrics**:
- Full RAG dispatch latency (target: <2000ms, maintain baseline)
- Cache-hit latency (target: <100ms total, <15ms cache lookup + <85ms other)
- Depth classification latency (target: <100ms)
- Early-exit latency (target: <1500ms for successful early-exit)

```python
class TestRAGLatency:
    """Benchmark RAG pipeline"""

    async def test_full_rag_dispatch_latency():
        """Measure: Full Phase 0-2 latency without cache"""
        rag_system = RAGSystem(cache_disabled=True)

        test_inputs = [
            "What's your name?",
            "Do you remember...?",
            "Based on our history...",
        ]

        latencies = []
        import time
        for input_str in test_inputs * 10:  # 30 queries
            state = {"decontexualized_input": input_str, ...}

            start = time.time()
            result = await rag_system.call_rag_subgraph(state)
            elapsed_ms = (time.time() - start) * 1000

            latencies.append(elapsed_ms)

        avg_ms = sum(latencies) / len(latencies)
        p95_ms = sorted(latencies)[int(len(latencies) * 0.95)]

        assert avg_ms < 2000, f"RAG too slow: avg {avg_ms}ms"
        assert p95_ms < 2500, f"RAG p95 too slow: {p95_ms}ms"
        print(f"Full RAG latency: avg {avg_ms:.0f}ms, p95 {p95_ms:.0f}ms")

    async def test_cache_hit_latency():
        """Measure: Phase 0-1 latency with cache hit"""
        cache = RAGCache()
        rag_system = RAGSystem(cache=cache)

        # Populate cache
        embedding = generate_embedding("What's your name?")
        await cache.store(embedding, {"user_facts": "..."}, "user_facts", 1800, {})

        latencies = []
        import time
        for _ in range(100):
            state = {"decontexualized_input": "What's your name?", "input_embedding": embedding, ...}

            start = time.time()
            result = await rag_system.call_rag_subgraph(state)
            elapsed_ms = (time.time() - start) * 1000

            assert result["cache_hit"] is True
            latencies.append(elapsed_ms)

        avg_ms = sum(latencies) / len(latencies)
        p95_ms = sorted(latencies)[95]

        assert avg_ms < 100, f"Cache-hit latency too slow: avg {avg_ms}ms"
        assert p95_ms < 150, f"Cache-hit p95 too slow: {p95_ms}ms"
        print(f"Cache-hit latency: avg {avg_ms:.0f}ms, p95 {p95_ms:.0f}ms")

    async def test_early_exit_latency():
        """Measure: SHALLOW query with early-exit (fastest path)"""
        rag_system = RAGSystem()

        latencies = []
        import time
        for _ in range(50):
            state = {
                "decontexualized_input": "What's your name?",
                "metadata": {"depth": "SHALLOW"},
                ...
            }

            start = time.time()
            result = await rag_system.call_rag_subgraph(state)
            elapsed_ms = (time.time() - start) * 1000

            latencies.append(elapsed_ms)

        avg_ms = sum(latencies) / len(latencies)

        # SHALLOW should be significantly faster than DEEP
        # If cache miss, should still be faster due to skipped dispatchers
        assert avg_ms < 1500, f"Early-exit too slow: {avg_ms}ms"
        print(f"Early-exit latency: {avg_ms:.0f}ms")
```

---

### 4.3 Consolidator Latency Benchmarks

**File**: `tests/performance/test_consolidator_latency.py`

**Metrics**:
- Consolidation total latency (target: <5000ms)
- Evaluator loop cost (each retry ~500ms additional)
- Database write latency (target: <1000ms)

```python
class TestConsolidatorLatency:
    """Benchmark consolidation pipeline"""

    async def test_consolidation_total_latency():
        """Measure: Full Phase 4-5 latency"""
        consolidator = ConsolidatorSystem()

        latencies = []
        import time
        for _ in range(20):
            state = {
                "internal_monologue": "...",
                "interaction_subtext": "...",
                "decontexualized_input": "...",
                ...
            }

            start = time.time()
            result = await consolidator.call_consolidation_subgraph(state)
            elapsed_ms = (time.time() - start) * 1000

            latencies.append(elapsed_ms)

        avg_ms = sum(latencies) / len(latencies)
        p95_ms = sorted(latencies)[int(len(latencies) * 0.95)]

        assert avg_ms < 5000, f"Consolidation too slow: avg {avg_ms}ms"
        assert p95_ms < 6000, f"Consolidation p95 too slow: {p95_ms}ms"
        print(f"Consolidation latency: avg {avg_ms:.0f}ms, p95 {p95_ms:.0f}ms")

    async def test_evaluator_retry_cost():
        """Measure: Cost of evaluator retries"""
        consolidator = ConsolidatorSystem()

        # Scenario 1: No retries (good facts)
        state_good = {
            "decontexualized_input": "I live in Auckland",
            "research_facts": {"user_rag_finalized": ["..."]},
            ...
        }

        import time
        start = time.time()
        result = await consolidator.call_consolidation_subgraph(state_good)
        no_retry_ms = (time.time() - start) * 1000

        # Scenario 2: With 1 retry (bad facts initially)
        # (Would need to mock facts_harvester to intentionally fail once)

        print(f"No retry: {no_retry_ms:.0f}ms")
        # Each retry adds ~500ms expected

    async def test_db_write_latency():
        """Measure: Database write latency"""
        db = DatabaseClient()

        latencies = []
        import time
        for i in range(50):
            doc = build_memory_doc(
                memory_name="test",
                content="...",
                source_global_user_id="user123",
                memory_type="fact",
                ...
            )

            start = time.time()
            await db.save_memory(doc, timestamp)
            elapsed_ms = (time.time() - start) * 1000

            latencies.append(elapsed_ms)

        avg_ms = sum(latencies) / len(latencies)

        assert avg_ms < 1000, f"DB write too slow: {avg_ms}ms"
        print(f"DB write latency: {avg_ms:.0f}ms")
```

---

## Part 5: Load Tests

### 5.1 Concurrent Conversation Load Test

**File**: `tests/load/test_concurrent_conversations.py`

**Scenario**: 10 concurrent conversations, each with 20 turns

```python
class TestConcurrentLoad:
    """Load test: Multiple conversations in parallel"""

    async def test_10_concurrent_conversations():
        """10 users, 20 turns each, all parallel"""
        system = FullPersonalitySystem()

        async def conversation_thread(user_id):
            """One user's conversation"""
            latencies = []

            for turn in range(20):
                input_str = generate_input(user_id, turn)

                import time
                start = time.time()
                response = await system.process_conversation(
                    user_id=user_id,
                    input=input_str,
                    ...
                )
                elapsed_ms = (time.time() - start) * 1000
                latencies.append(elapsed_ms)

            return latencies

        # Run 10 conversations concurrently
        import asyncio
        all_latencies = await asyncio.gather(*[
            conversation_thread(f"user_{i}") for i in range(10)
        ])

        # Flatten and analyze
        flat_latencies = [lat for latencies in all_latencies for lat in latencies]

        avg_ms = sum(flat_latencies) / len(flat_latencies)
        p95_ms = sorted(flat_latencies)[int(len(flat_latencies) * 0.95)]
        p99_ms = sorted(flat_latencies)[int(len(flat_latencies) * 0.99)]

        # Check degradation under load
        # (Compare to single-user baseline)

        print(f"10 concurrent: avg {avg_ms:.0f}ms, p95 {p95_ms:.0f}ms, p99 {p99_ms:.0f}ms")
        assert p95_ms < 3000, f"Latency degradation under load: p95 {p95_ms}ms"

    async def test_cache_hit_rate_under_load():
        """Cache performance under concurrent load"""
        system = FullPersonalitySystem()

        # All 10 users ask same question (cache should benefit)
        cache_hits = 0
        total_queries = 0

        async def query_task(user_id):
            nonlocal cache_hits, total_queries

            # User asks about favorite food (20 times)
            for turn in range(20):
                result = await system.process_conversation(
                    user_id=user_id,
                    input="What's your favorite food?",
                    ...
                )

                total_queries += 1
                if result.get("cache_hit"):
                    cache_hits += 1

        await asyncio.gather(*[query_task(f"user_{i}") for i in range(10)])

        hit_rate = cache_hits / total_queries if total_queries > 0 else 0

        # Under load, hit rate should still be good
        assert hit_rate > 0.50, f"Cache hit rate too low under load: {hit_rate:.1%}"
        print(f"Cache hit rate under load: {hit_rate:.1%}")

    async def test_database_write_throughput():
        """DB write throughput under load"""
        db = DatabaseClient()

        # All 10 users write facts concurrently
        import time
        start = time.time()

        async def write_task(user_id):
            for i in range(10):
                doc = build_memory_doc(...)
                await db.save_memory(doc, timestamp)

        await asyncio.gather(*[write_task(f"user_{i}") for i in range(10)])

        elapsed_sec = time.time() - start
        throughput_per_sec = (10 * 10) / elapsed_sec  # 100 writes total

        # Should handle at least 50 writes/sec
       assert throughput_per_sec > 50, f"Write throughput too low: {throughput_per_sec:.0f}/sec"
        print(f"DB write throughput: {throughput_per_sec:.0f} writes/sec")
```

---

### 5.2 Stress Test

**File**: `tests/load/test_stress.py`

**Scenario**: Scale until system breaks or degrades significantly

```python
class TestStress:
    """Stress test: Push system to limits"""

    async def test_concurrent_scaling():
        """Gradually increase concurrency until latency degrades"""
        system = FullPersonalitySystem()

        results = []
        for concurrent in [5, 10, 20, 50, 100]:
            latencies = []

            async def task():
                result = await system.process_conversation(
                    user_id=f"user_{concurrent}_{i}",
                    input="Hi, how are you?",
                    ...
                )
                latencies.append(result["latency_ms"])

            await asyncio.gather(*[task() for i in range(concurrent)])

            avg_ms = sum(latencies) / len(latencies)
            results.append((concurrent, avg_ms))
            print(f"{concurrent} concurrent: avg {avg_ms:.0f}ms")

            # Stop if latency becomes unacceptable
            if avg_ms > 5000:
                print("System degraded, stopping scaling test")
                break
```

---

## Part 6: Regression Tests

### 6.1 Backward Compatibility

**File**: `tests/regression/test_backward_compat.py`

```python
class TestBackwardCompatibility:
    """Ensure no regressions in existing functionality"""

    async def test_existing_rag_still_works():
        """RAG with cache disabled should behave identically to before"""
        system_old = RAGSystem(cache_enabled=False)
        system_new = RAGSystem(cache_enabled=False)

        input_str = "Do you remember the promise?"

        result_old = await system_old.call_rag_subgraph({"input": input_str, ...})
        result_new = await system_new.call_rag_subgraph({"input": input_str, ...})

        # Results should be semantically equivalent
        assert content_similarity(result_old["research_facts"], result_new["research_facts"]) > 0.95

    async def test_existing_consolidator_still_works():
        """Consolidator with evaluator should not break facts"""
        consolidator = ConsolidatorSystem()

        # Simulate a conversation that extracts facts
        state = {...}
        result = await consolidator.call_consolidation_subgraph(state)

        # Facts should be written successfully
        assert result["new_facts"] is not None
        assert result["write_success"] is True

    async def test_database_backward_compat():
        """New DB schema doesn't break old queries"""
        db = DatabaseClient()

        # Old-style query should still work
        old_facts = await db.find(
            collection="memory",
            filter={"source_global_user_id": "user123"}
        )

        assert old_facts is not None
```

---

## Part 7: Acceptance Criteria

### Phase 1 Acceptance (End of Week 2)

- [ ] All database collections created and indices verified
- [ ] Cache backend operational and tested (10+ tests passing)
- [ ] Scheduler module extracted and integrated
- [ ] RAG version tracking implemented
- [ ] Zero data corruption in migration

### Phase 2 Acceptance (End of Week 4)

- [ ] Depth classifier achieves >95% accuracy on test set
- [ ] Cache integration passes all tests
- [ ] Cache hit rate: >50% in unit tests
- [ ] Cache lookup latency: <15ms avg
- [ ] No regressions in RAG functionality

### Phase 3 Acceptance (End of Week 6)

- [ ] Evaluator loop catches >95% of contradictions
- [ ] Max retry enforcement tested (stops at 3)
- [ ] Metadata propagates through all 5 phases
- [ ] Integration test suite: 20+ tests passing
- [ ] Zero data inconsistencies in schema

### Phase 4 Acceptance (End of Week 8)

- [ ] Atomic transaction implementation verified
- [ ] Cache invalidation timing correct (after commit, not before)
- [ ] Promise scheduling integrated
- [ ] Database write performance: <1000ms avg
- [ ] Version tracking working end-to-end

### Phase 5 Acceptance (End of Week 10)

- [ ] Full end-to-end scenario passing (input → output → cache)
- [ ] Performance benchmarks meet targets
- [ ] Load test: 10 concurrent conversations stable
- [ ] Regression tests: 100% passing
- [ ] Documentation complete
- [ ] Rollback procedure tested

---

## Part 8: Performance Baseline & Targets

### Current System (Baseline - 2026-04-18)

| Metric | Current | Target | Improvement |
|--------|---------|--------|------------|
| **RAG Latency (avg)** | 800ms | 600ms | 25% faster |
| **RAG Latency (p95)** | 1500ms | 800ms | 47% faster |
| **Dispatcher calls/query** | 3 | 1.5 | 50% fewer |
| **Cache hit rate** | N/A | >60% | New feature |
| **Cache lookup latency** | N/A | <15ms | New feature |
| **Consolidator latency** | 3000ms | 3500ms | +17% (more validation, acceptable) |
| **Evaluator retry rate** | 0% | <5% | Near-perfect validation |
| **Affinity accuracy** | Baseline | +10% | Better personality tracking |
| **Error rate** | ~1.5% | <0.5% | 67% reduction |

### Measurement Protocol

1. **Baseline Run** (Pre-implementation)
   - Run 1000 conversation samples
   - Record all latencies, error rates, cache behavior
   - Store in `baseline_metrics.json`

2. **Implementation Run** (Post-implementation)
   - Run same 1000 conversation samples
   - Record all metrics
   - Compare: (`new - baseline`) / `baseline`

3. **Statistical Significance**
   - Latency improvements >10% statistically significant (p < 0.05)
   - Error rate improvements >50% statistically significant

---

## Part 9: Test Execution Timeline

```
Week 3:    Unit tests (RAGCache, InputDepthClassifier, Evaluator, DB) - 30 tests
Week 4:    Integration tests (RAG+Cache, Depth+Routing, Evaluator loop) - 12 tests
Week 5:    E2E tests (Full personality flow) - 4 tests
Week 6:    Performance benchmarks (establish baselines)
Week 7-8:  Load tests (concurrent conversations)
Week 9:    Stress tests + Regression tests
Week 10:   Sign-off + UAT

Total Test Count: ~100 tests
Automation Level: 100% (all tests CI/CD)
Expected Pass Rate: >95% before canary
```

---

## Part 10: Test Tools & Infrastructure

### Required Tools

- **Testing Framework**: pytest + pytest-asyncio
- **Mocking**: unittest.mock, pytest-mock
- **Performance**: timeit, psutil, locust (load testing)
- **Monitoring**: prometheus, grafana (real-time metrics)
- **CI/CD**: GitHub Actions (run tests on every PR)
- **Coverage**: pytest-cov (target: >85% coverage)

### Test Environment Setup

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock pytest-cov locust prometheus-client

# Run all tests
pytest tests/ -v --cov=src/kazusa_ai_chatbot --cov-report=html

# Run only unit tests
pytest tests/unit/ -v

# Run performance tests
pytest tests/performance/ -v --benchmark

# Run load tests
locust -f tests/load/locustfile.py
```

---

## Part 11: Metrics Dashboard

### Prometheus Metrics to Collect

```
# Cache metrics
rag_cache_hits_total
rag_cache_misses_total
rag_cache_lookup_latency_ms
rag_cache_size_bytes

# Depth classifier metrics
depth_classifier_latency_ms
depth_classification_distribution {depth: SHALLOW|MEDIUM|DEEP}

# RAG latency
rag_dispatch_latency_ms {depth: SHALLOW|MEDIUM|DEEP}
dispatcher_calls_per_query

# Consolidator metrics
consolidation_latency_ms
evaluator_retry_count
fact_extraction_errors

# Database metrics
db_write_latency_ms
db_write_throughput_per_sec
cache_invalidation_latency_ms

# System health
error_rate
cache_hit_rate
personality_consistency_score
```

### Grafana Dashboards

- **Performance Dashboard**: Latencies, throughput, error rates
- **Cache Dashboard**: Hit rate, lookup times, invalidation patterns
- **Personality Dashboard**: Fact extraction accuracy, consistency score
- **System Health**: Error rates, uptime, resource usage

---

## Part 12: Sign-Off Checklist

**Pre-Canary Deployment**

- [ ] All 100 tests passing (>95% pass rate)
- [ ] Performance benchmarks meet targets (see Part 8)
- [ ] Load test: 10 concurrent conversations stable
- [ ] No data corruption observed
- [ ] Backward compatibility: Zero regressions
- [ ] Coverage: >85%
- [ ] Documentation: Complete with examples
- [ ] Rollback procedure: Tested and verified
- [ ] Monitoring: Alerts configured (latency, errors, cache)
- [ ] Team sign-off: Technical lead, DB admin, DevOps

**Canary Phase** (5% traffic)

- [ ] Monitor metrics for 24 hours
- [ ] Cache hit rate: Observe >50% (should see improvement)
- [ ] Latency: p95 <1000ms (improvement vs baseline)
- [ ] Error rate: <0.5%
- [ ] No unexpected behaviors

**Ramp Phase** (25%, 50%, 100%)

- [ ] Each increment monitored for 4 hours
- [ ] Metrics stable or improving
- [ ] Zero emergency rollbacks

---

## Document History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-04-18 | 1.0 | Test Lead | Initial draft |
| TBD | 1.1 | Team | Add test data generators |
| TBD | 2.0 | Team | Post-implementation updates |

---

## Sign-Off

- [ ] Test Lead: TEST_PLAN approved
- [ ] Technical Lead: Requirements understood
- [ ] QA Lead: Ready to execute tests
- [ ] DevOps: Infrastructure prepared

---

**Test Plan Owner**: QA Lead
**Last Review**: 2026-04-18
**Next Review**: 2026-05-18 (after Phase 1 completion)
