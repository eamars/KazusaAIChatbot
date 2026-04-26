# RAG Cache 2 Design

## Goals

- Cache expensive reusable retrieval work at the individual RAG helper-agent level.
- Cache initializer search strategy separately so the system can learn better retrieval paths over time.
- Keep result cache session-scoped and implementation-simple: cache entries may be discarded on Kazusa service reboot.
- Remove cache ownership from the legacy RAG path. New cache runtime must not depend on `persona_supervisor2_rag._get_rag_cache`.

## What To Cache

- Individual RAG helper-agent results, when the underlying data source is stable enough for reuse.
- RAG initializer path / search strategy against similar normalized queries.
- Query embeddings as a supporting utility cache, keyed by normalized text and embedding model.

The initializer cache should not become deterministic hard routing too early. It should provide a preferred plan plus confidence, and fall back to live initialization when context changes, confidence is low, or the cached strategy has poor historical success. This keeps the system adaptive instead of fossilized.

## What Not To Cache

- `web_search_agent2`: web search and URL reads depend on external real-time data. Do not cache for now.
- `rag_dispatcher`: cheap enough and tightly coupled to current loop state.
- `rag_evaluator`: summary/verdict is supervisor-local and should follow the live agent result.
- `rag_finalizer`: final answers should not be cached; regenerate from current known facts.
- Recent / open-ended / last-N conversation search results: new messages arrive frequently enough that these entries become stale too easily.

## Storage Location

- Store actual RAG result cache in Python in-memory session LRU.
- It is acceptable for all result-cache entries to disappear on Kazusa service reboot.
- Do not use MongoDB or a dedicated vector database for helper-agent result cache initially.
- Optional long-term initializer strategy memory may be persisted later, but it should be treated as learned strategy evidence, not as trusted cached output.

## TTL / Eviction Policy

- No time-based TTL for session cache by default.
- Evict by max size pressure, preferably LRU rather than pure oldest.
- Clear all session cache on service restart.
- Correctness during a session is handled by invalidation events, not expiry timers.

## Invalidation Model

Use a global cache handler with per-agent cache policies.

- The global handler owns storage, LRU, lookup/store APIs, invalidation dispatch, and metrics.
- Per-agent policies define whether to cache, how to build keys, and which data dependencies an entry has.
- Write paths emit domain invalidation events to the global handler. They should not know about individual agents.

Primary invalidation sources:

- `save_conversation`: invalidates overlapping conversation-history cache entries.
- Consolidator writes: invalidate persistent-memory and profile-memory cache entries for affected users/types.
- Prompt/schema/agent-roster changes: invalidate initializer strategy cache by version mismatch.
- Embedding model/index changes: invalidate semantic search result cache and embedding utility cache.

Implementation may combine eager deletion with lazy invalidation. Lazy invalidation means an entry can be treated as a miss when its recorded dependency scope/version no longer matches current state.

## Per Agent Policy

| Agent / Stage                     | Cache Key                                                                                                                                                                   | Invalidator                                                                                                                                                                                                                                                                                                 | Storage Location                                                                   |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `rag_initializer`                 | Normalized/decontextualized query embedding + context signature + `initializer_prompt_version` + `agent_registry_version` + `strategy_schema_version`                       | Prompt/schema/agent-roster version mismatch. Not invalidated by normal DB writes.                                                                                                                                                                                                                           | Python in-memory session LRU; optional persistent strategy-memory collection later |
| `rag_dispatcher`                  | N/A                                                                                                                                                                         | N/A                                                                                                                                                                                                                                                                                                         | N/A                                                                                |
| `rag_evaluator`                   | N/A                                                                                                                                                                         | N/A                                                                                                                                                                                                                                                                                                         | N/A                                                                                |
| `rag_finalizer`                   | N/A                                                                                                                                                                         | N/A                                                                                                                                                                                                                                                                                                         | N/A                                                                                |
| `user_lookup_agent`               | Normalized display name + platform/channel scope                                                                                                                            | User profile write/rename for the **resolved** `global_user_id`. Dependency is keyed by resolved UUID (not query alias) so misspelled or aliased queries are invalidated correctly when the real profile changes. Both exact-profile and vector-search-fallback results are cached at the `run()` boundary. | Python in-memory session LRU                                                       |
| `user_list_agent`                 | Normalized `display_name_value` + `operator` + `source` + platform + channel + `limit`                                                                                      | Depends on `source`: `"user_profiles"` → consolidator profile write (`"user_profile"` event); `"conversation_participants"` → `save_conversation` (`"conversation_history"` event); `"both"` → either event.                                                                                                | Python in-memory session LRU                                                       |
| `user_profile_agent`              | `global_user_id` + requested profile bundle/projection + profile-memory version                                                                                             | Consolidator writes profile memories, objective facts, user image/profile fields for that user.                                                                                                                                                                                                             | Python in-memory session LRU                                                       |
| `conversation_filter_agent`       | Only for closed historical ranges: normalized filters + absolute `from_timestamp` + absolute `to_timestamp` + limit/sort                                                    | `save_conversation` invalidates overlapping channel/user/time range. Recent/open-ended/last-N queries are not cached.                                                                                                                                                                                       | Python in-memory session LRU                                                       |
| `conversation_keyword_agent`      | Only for closed historical ranges: normalized keyword + platform/channel/user filters + absolute time range + `top_k`                                                       | `save_conversation` invalidates overlapping channel/user/time range. Recent/open-ended queries are not cached.                                                                                                                                                                                              | Python in-memory session LRU                                                       |
| `conversation_search_agent`       | Only for closed historical ranges: normalized semantic query embedding/hash + platform/channel/user filters + absolute time range + `top_k` + embedding model/index version | `save_conversation` invalidates overlapping channel/user/time range. Also clear on embedding model/index change. Recent/open-ended queries are not cached.                                                                                                                                                  | Python in-memory session LRU                                                       |
| `persistent_memory_keyword_agent` | Normalized keyword + source user/global scope + memory type/status/source kind filters + `top_k`                                                                            | Consolidator writes/updates/deletes matching persistent memories.                                                                                                                                                                                                                                           | Python in-memory session LRU                                                       |
| `persistent_memory_search_agent`  | Normalized semantic query embedding/hash + memory filters + `top_k` + embedding model/index version                                                                         | Consolidator writes/updates/deletes matching persistent memories. Also clear on embedding model/index change.                                                                                                                                                                                               | Python in-memory session LRU                                                       |
| `web_search_agent2`               | N/A                                                                                                                                                                         | N/A                                                                                                                                                                                                                                                                                                         | N/A                                                                                |

## Conversation History Policy

Remove auto exclusion of recent chat history (`input_context_to_timestamp`). The old design injected a `to_timestamp` bound to avoid re-fetching recent messages, but that makes RAG secretly blind to highly relevant data.

Preferred replacement:

- Allow conversation retrieval to search all explicitly allowed data.
- Dedupe before cognition by `platform_message_id` or equivalent stable message identity.
- Do not cache recent/open-ended conversation retrieval.
- Cache only closed historical windows where both `from_timestamp` and `to_timestamp` are absolute.

## Proposed Module Ownership

New cache code should be path-neutral:

- `kazusa_ai_chatbot/rag/cache2_runtime.py`: global session cache handler.
- `kazusa_ai_chatbot/rag/cache2_policy.py`: per-agent cache policy definitions.
- `kazusa_ai_chatbot/rag/cache2_events.py`: invalidation event types and dependency scopes.

The existing `rag.cache.RAGCache` and legacy boundary cache can be referenced for lessons learned, but Cache 2 should not be owned by the old RAG graph because that path is planned for removal.

---

## Consolidator → Cache 2 Integration Path

### Context

RAG1 (`rag.cache.RAGCache`, `persona_supervisor2_rag._get_rag_cache`) is planned for full removal. Cache 2 is its replacement. This section documents the concrete integration path and the decisions made before implementing it, so the work lands consistently.

### The Lifecycle Gap

Cache 2 has three phases:

| Phase                                       | Owner                                                       | Code location           |
| ------------------------------------------- | ----------------------------------------------------------- | ----------------------- |
| Write (miss → compute → store)              | `BaseRAGHelperAgent.run()` via `read_cache` / `write_cache` | `rag/helper_agent.py`   |
| Serve (hit → return cached payload)         | `RAGCache2Runtime.get()`                                    | `rag/cache2_runtime.py` |
| Invalidate (DB write → evict stale entries) | `db_writer` in consolidator                                 | **not yet wired**       |

Until RAG1 is removed, Step 5 of `db_writer` calls `_get_rag_cache()`. Once RAG1 is removed, that block disappears and nothing signals Cache 2 that a domain write occurred. This section defines what replaces it.

### Design Decision: db_writer emits domain events to Cache 2 runtime directly

The end-to-end chain is:

```text
RAG helper agents
  -> research_facts / research_metadata
Cognition
  -> internal_monologue / directives / stance / subtext
Dialog
  -> final_dialog
Background consolidator
  -> accepted facts, promises, relationship updates, image updates
db_writer
  -> durable MongoDB writes
  -> Cache 2 invalidation events
```

Invalidation is driven by durable state changes, not by whether cognition used
or accepted a particular RAG result. Cognition sits between RAG and the
consolidator, but it does not need to participate in cache invalidation because
it does not own durable write scope. The consolidator, specifically
`db_writer`, has the authoritative write outcome and enough structured scope
(`global_user_id`, platform/channel, timestamp, write type, and generated
memory docs) to announce what changed.

`db_writer` must not try to remember or reverse-map the RAG cache key used
earlier in the turn. Cache entries declare their dependencies at write time;
`db_writer` emits domain events after successful persistence; the runtime
matches dependency/event overlap and evicts stale entries.

```text
cache entry dependency:
  source="user_profile", global_user_id="abc"

successful consolidator write:
  source="user_profile", global_user_id="abc"

runtime:
  dependency/event overlap -> evict
```

After each successful MongoDB write, `db_writer` calls:

```python
get_rag_cache2_runtime().invalidate(CacheInvalidationEvent(
    source="<domain>",
    platform=state["platform"],
    platform_channel_id=state["platform_channel_id"],
    global_user_id=global_user_id,
    reason="consolidator write",
))
```

The `dependency_matches_event` matching rule treats every empty field as a wildcard. An event with `display_name=""` correctly evicts all cached entries for that domain+user, regardless of which display-name variant was cached. This is intentional: if a profile changed, every cached lookup for that user must be re-fetched.

**Domain event table** — maps each `db_writer` write step to the Cache 2 `source`:

| `db_writer` write                                               | Event `source=`     | Triggers invalidation for                                               |
| --------------------------------------------------------------- | ------------------- | ----------------------------------------------------------------------- |
| `insert_profile_memories` (diary, objective facts, commitments) | `"user_profile"`    | `user_lookup_agent`, `user_profile_agent`, `persistent_memory_*` agents |
| `upsert_character_state`                                        | `"character_state"` | future character-state agents                                           |
| `upsert_user_image` / `upsert_character_self_image`             | `"user_image"`      | future user/character image agents                                      |

Emit only for write steps that actually succeeded (check `write_log` before emitting).

### Decisions and Rationale

**Cache correctness is an invalidation problem, not a cognition-acceptance problem.**
Do not decide invalidation based on whether a RAG result was consumed by
cognition or accepted by the consolidator. A cached retrieval is safe only when
its backing source can be described by `CacheDependency` and every write path
that can change that source emits a matching `CacheInvalidationEvent`.

**Positive retrieval results may be cached when their source is invalidatable.**
For retrieval agents such as `persistent_memory_search_agent`, "search returned
usable records" and "the downstream answer accepted those records" are separate
concepts. The cache should memoize expensive retrieval output when the
dependency surface is covered; the supervisor/cognition/synthesizer can decide
whether the evidence answers the user.

**Negative or unresolved retrieval results require extra care.**
Empty or unresolved results can become stale as soon as a later write adds
matching data. Cache them only after the relevant source has a reliable
invalidation emitter. Until then, prefer not caching negative/unresolved
retrievals.

**Do not import specific agent classes in `db_writer`.**
The consolidator is a domain writer. It knows *what changed* (source + scope), not *who is caching it*. Importing agent classes would invert the dependency and require `db_writer` to grow as new agents are added.

**Do not add a lifecycle registry or observer/event-bus pattern.**
The `CacheInvalidationEvent` + `dependency_matches_event` scan already acts as the fan-out. A separate observer registry would replicate that mechanism without adding capability. Add one only if per-agent lifecycle hooks (e.g. "warm cache after invalidation") become necessary.

**Do not put invalidation calls inside DB-layer functions** (`insert_profile_memories`, etc.).
The DB layer should not know about cache topology. Invalidation is a consolidator concern triggered by the outcome of a write sequence, not a side-effect of an individual DB helper.

**No backward-compatibility shim layer for agents.**
When an agent is migrated to `BaseRAGHelperAgent`, the old standalone function is removed. The supervisor registry is updated to reference the new class. There is no intermediate shim module.

### BaseRAGHelperAgent Contract

Every Cache 2 agent subclasses `BaseRAGHelperAgent` and must:

1. Declare `cache_name` in `__init__` — the logical namespace used in `store()`.
2. Build a stable `cache_key` from the policy module and call `self.read_cache(cache_key)` before computing.
3. Call `self.write_cache(cache_key, result, dependencies, metadata)` with explicit `CacheDependency` entries after a successful compute.
4. Declare the correct `source=` in each `CacheDependency` so the domain event from `db_writer` will match.

The base class provides `read_cache`, `write_cache`, and `invalidate_cache`. Agents must not call `get_rag_cache2_runtime()` directly.

# 
