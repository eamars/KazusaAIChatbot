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

| Agent / Stage | Cache Key | Invalidator | Storage Location |
| --- | --- | --- | --- |
| `rag_initializer` | Normalized/decontextualized query embedding + context signature + `initializer_prompt_version` + `agent_registry_version` + `strategy_schema_version` | Prompt/schema/agent-roster version mismatch. Not invalidated by normal DB writes. | Python in-memory session LRU; optional persistent strategy-memory collection later |
| `rag_dispatcher` | N/A | N/A | N/A |
| `rag_evaluator` | N/A | N/A | N/A |
| `rag_finalizer` | N/A | N/A | N/A |
| `user_lookup_agent` | Normalized display name + platform/channel scope + lookup mode | User profile rename/write. If fallback used conversation search, do not cache that fallback result unless closed historical scope exists. | Python in-memory session LRU |
| `user_profile_agent` | `global_user_id` + requested profile bundle/projection + profile-memory version | Consolidator writes profile memories, objective facts, user image/profile fields for that user. | Python in-memory session LRU |
| `conversation_filter_agent` | Only for closed historical ranges: normalized filters + absolute `from_timestamp` + absolute `to_timestamp` + limit/sort | `save_conversation` invalidates overlapping channel/user/time range. Recent/open-ended/last-N queries are not cached. | Python in-memory session LRU |
| `conversation_keyword_agent` | Only for closed historical ranges: normalized keyword + platform/channel/user filters + absolute time range + `top_k` | `save_conversation` invalidates overlapping channel/user/time range. Recent/open-ended queries are not cached. | Python in-memory session LRU |
| `conversation_search_agent` | Only for closed historical ranges: normalized semantic query embedding/hash + platform/channel/user filters + absolute time range + `top_k` + embedding model/index version | `save_conversation` invalidates overlapping channel/user/time range. Also clear on embedding model/index change. Recent/open-ended queries are not cached. | Python in-memory session LRU |
| `persistent_memory_keyword_agent` | Normalized keyword + source user/global scope + memory type/status/source kind filters + `top_k` | Consolidator writes/updates/deletes matching persistent memories. | Python in-memory session LRU |
| `persistent_memory_search_agent` | Normalized semantic query embedding/hash + memory filters + `top_k` + embedding model/index version | Consolidator writes/updates/deletes matching persistent memories. Also clear on embedding model/index change. | Python in-memory session LRU |
| `web_search_agent2` | N/A | N/A | N/A |

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
