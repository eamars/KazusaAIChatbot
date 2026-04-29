# rag cache2 persistent initializer plan

## Summary

- Goal: Persist the RAG2 initializer strategy cache in MongoDB so cached paths survive brain-service restarts.
- Plan class: large
- Status: completed
- Overall cutover strategy: compatible
- Highest-risk areas: stale initializer strategies after prompt/schema/agent changes; adding database persistence without turning normal cache reads into database reads; avoiding accidental persistence of helper-agent result caches; hit-count fire-and-forget writes leaking into the response-path latency budget.
- Acceptance criteria: current-version `rag2_initializer` entries hydrate into the Python LRU at startup ordered by `hit_count desc, updated_at desc`, new cacheable initializer paths write through to MongoDB immediately, in-memory cache hits trigger a fire-and-forget hit-counter upsert that does not block the response, stale-version entries are cleared before hydration, and helper-agent caches remain process-local.

## Context

`rag_initializer` currently stores strategy payloads only in `RAGCache2Runtime`, a process-local LRU in `src/kazusa_ai_chatbot/rag/cache2_runtime.py`. This means a repeated query can skip the initializer LLM during one brain-service process, but the learned path disappears after restart.

The initializer cache key is built by `build_initializer_cache_key(...)` in `src/kazusa_ai_chatbot/rag/cache2_policy.py`. The key already includes `INITIALIZER_POLICY_VERSION`, `INITIALIZER_PROMPT_VERSION`, `INITIALIZER_AGENT_REGISTRY_VERSION`, and `INITIALIZER_STRATEGY_SCHEMA_VERSION`. Those constants define whether a persisted initializer strategy is still valid.

This plan intentionally persists only initializer strategy cache entries. It does not restore the deleted RAG1 MongoDB write-through cache, and it does not persist helper-agent result caches.

## Mandatory Rules

- Keep `RAGCache2Runtime` as the hot serving cache. Normal request-path cache **reads** must check Python memory only, not MongoDB.
- Hit-counter persistence and write-through upserts may write to MongoDB, but must run as fire-and-forget background tasks scheduled via `asyncio.create_task(...)` so the response-path coroutine never awaits a MongoDB round-trip on a cache hit.
- Use MongoDB as durable backing storage for startup hydration and write-through persistence.
- Persist new cacheable initializer paths immediately after generation. Do not wait for brain-service shutdown to flush entries.
- Treat MongoDB writes as best-effort for chat latency. Catch `pymongo.errors.PyMongoError` (the common base class for motor/pymongo failures) around every persistent-cache write, log the failure with the cache_key, and continue. Do not catch bare `Exception`.
- Do not persist raw `original_query`, raw runtime context, prompt text, user message text, display names, user IDs, channel IDs, or retrieved evidence in persistent cache rows. The cached `result` payload (`unknown_slots`, `confidence`) and version metadata are the only persisted content.
- Do not persist helper-agent result caches, dispatcher output, evaluator output, finalizer output, web search results, or any Cache2 entries other than explicitly allowed `rag2_initializer` entries.
- Do not resurrect legacy RAG1 collections. `rag_cache_index` and `rag_metadata_index` must continue to be dropped by bootstrap.
- Do not couple `RAGCache2Runtime` LRU eviction to MongoDB row deletion. A memory eviction must leave the persistent row intact so its accumulated `hit_count` survives for the next restart's hydration ranking.
- Follow project Python style: typed helper signatures, complete docstrings for public helpers, specific exception handling, imports at top, and focused changes.
- Tests that mock LLM output may verify deterministic cache plumbing only. They must not be used as evidence that a prompt semantically works.

## Must Do

- Create one generic persistent Cache2 collection named `rag_cache2_persistent_entries`.
- Initially allow only `cache_name == "rag2_initializer"` to write to and hydrate from this collection.
- Build the initializer version key as the pipe-joined string `f"{INITIALIZER_POLICY_VERSION}|{INITIALIZER_PROMPT_VERSION}|{INITIALIZER_AGENT_REGISTRY_VERSION}|{INITIALIZER_STRATEGY_SCHEMA_VERSION}"`. Pipe-joined plain text is mandatory for human readability in MongoDB inspection; do not substitute a digest.
- Create exactly one compound index `cache2_persistent_lookup_idx` on `(cache_name asc, version_key asc, hit_count desc, updated_at desc)`. This single index covers the stale-purge filter, the hydration filter, and the hydration sort. Do not create additional indexes in this plan.
- During startup bootstrap (`db_bootstrap()`), delete persisted rows where `cache_name == "rag2_initializer"` and `version_key != build_initializer_version_key()` before any hydration runs.
- During startup bootstrap, after the stale-version purge, run `prune_persistent_entries(cache_name="rag2_initializer", max_entries=5 * RAG_CACHE2_MAX_ENTRIES)` to bound long-term collection growth. The 5× factor preserves a working-set tail beyond what fits in memory while preventing unbounded growth without a background sweeper.
- During service startup (in `service.lifespan()`, after `db_bootstrap()` and before `_build_graph()`), load current-version initializer rows into `RAGCache2Runtime`. Use `limit=RAG_CACHE2_MAX_ENTRIES`, sort by `hit_count desc, updated_at desc`, and insert into the LRU in **reversed** order so the highest-hit row is the last `store(...)` call and lands at the most-recently-used end of the LRU.
- On an initializer cache miss that produces a cacheable strategy, store in Python LRU first, then schedule a fire-and-forget upsert to MongoDB via `asyncio.create_task(...)`. The response path must not await the upsert.
- On an initializer cache hit, schedule a fire-and-forget `record_initializer_hit(cache_key)` via `asyncio.create_task(...)` that increments `hit_count` and refreshes `updated_at` in MongoDB. The response path must return the cached slots before the task runs.
- Wrap every persistent-cache MongoDB call in a `try/except pymongo.errors.PyMongoError` that logs the cache key when one exists, otherwise the helper name and cache name, then returns silently. Hydration failure must not prevent the service from starting.
- Add deterministic unit tests for: stale-version purge, prune cap, empty-DB hydration, mixed stale-and-current hydration ordering, write-through after miss, hit-counter increment after hit, no-DB-read on the synchronous request path, and `PyMongoError` swallowing in each write helper.
- Update RAG/Cache2 docs to describe persistent initializer strategy cache, the hit-count-ordered hydration policy, and the LRU/database relationship.

## Deferred

- Do not persist any helper-agent result cache in this plan.
- Do not add shutdown-time cache flushing.
- Do not add a background cache writer, write batching, or retry queue.
- Do not add TTL expiration for persistent entries.
- Do not add admin APIs, UI pages, or health payloads for inspecting persistent cache contents.
- Do not change initializer prompt behavior, dispatcher behavior, helper-agent routing, evaluator behavior, finalizer behavior, or cognition/consolidation prompts.
- Do not introduce new LLM calls or model retries.

## Cutover Policy

Overall strategy: compatible

| Area | Policy | Instruction |
|---|---|---|
| Runtime cache reads | compatible | Keep request-path reads on `RAGCache2Runtime.get(...)`; do not add per-request MongoDB fallback. |
| Cache writes (miss) | compatible | After a cacheable initializer miss, write to memory first, then schedule a fire-and-forget MongoDB upsert via `asyncio.create_task(...)`. |
| Cache writes (hit) | compatible | After every initializer cache hit, schedule a fire-and-forget `record_initializer_hit(cache_key)` upsert that bumps `hit_count` and `updated_at`. The response coroutine must not await the task. |
| Startup behavior | compatible | Hydrate current-version persistent initializer entries into memory before serving traffic, ordered by `hit_count desc, updated_at desc`, inserted in reverse so the hottest row is MRU. |
| Version invalidation | compatible | Delete stale initializer rows at bootstrap before hydration. Existing in-memory invalidation behavior is unchanged. |
| Long-term capacity | compatible | At end of bootstrap, prune the collection to `5 * RAG_CACHE2_MAX_ENTRIES` rows per `cache_name` to bound growth without a background sweeper. |
| Collection strategy | compatible | Use one generic persistent Cache2 collection with `cache_name` separation so future cache types can reuse the storage layer. |
| MongoDB failure handling | compatible | Catch `pymongo.errors.PyMongoError` in every persistence helper, log with cache key when available or helper/cache name otherwise, and continue. Hydration failure must not block startup. |

## Agent Autonomy Boundaries

- The implementation agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The agent must not introduce alternate storage collections, compatibility layers, fallback database reads on cache miss, shutdown flush behavior, or extra persistent cache types.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If a required instruction is impossible because of current code shape, the agent must stop and report the blocker instead of inventing a substitute.
- If implementation discovers existing tests or docs that conflict with this plan, preserve this plan's intent and update only the directly affected tests/docs.

## Target State

The target runtime lifecycle is:

```text
brain service startup
  -> db_bootstrap()
       -> ensure rag_cache2_persistent_entries exists
       -> create cache2_persistent_lookup_idx if missing
       -> purge_stale_initializer_entries() (deletes rows with old version_key)
       -> prune_persistent_entries(cache_name="rag2_initializer",
                                    max_entries=5 * RAG_CACHE2_MAX_ENTRIES)
  -> load_initializer_entries(limit=RAG_CACHE2_MAX_ENTRIES)
       sorted (hit_count desc, updated_at desc)
  -> for row in reversed(rows):
       RAGCache2Runtime.store(...)
       (last store wins MRU; highest-hit row ends up at MRU position)
  -> _build_graph()
  -> serve chat traffic

normal initializer request
  -> RAGCache2Runtime.get(cache_key)  [memory only, no DB]
  -> hit:
       return cached unknown_slots
       asyncio.create_task(record_initializer_hit(cache_key))   [fire-and-forget]
  -> miss:
       call initializer LLM
       if cacheable:
         _write_initializer_cache(...)                          [in-memory store]
         asyncio.create_task(upsert_initializer_entry(...))     [fire-and-forget]
       return slots
```

The Python LRU remains the fast serving layer. MongoDB makes selected initializer strategy entries survive restarts, supplies startup hydration, and accumulates per-key hit counts that rank the next restart's hydration. All MongoDB writes triggered by request handling are scheduled fire-and-forget via `asyncio.create_task` so the response coroutine never awaits a database round-trip.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Collection layout | Use `rag_cache2_persistent_entries` for persistent Cache2 entries. | One generic collection supports future cache types without new collection plumbing for each type. |
| Initial persistence scope | Persist only `rag2_initializer`. | Initializer strategies are stable across ordinary DB writes and are the user-requested target. |
| Read path | No MongoDB lookup during normal requests. | Keeps cache hits low-latency and prevents DB latency from entering the response path. |
| Write timing (miss) | Write-through immediately after each new cacheable initializer path is generated. | Shutdown flushing is unreliable across crashes, deploy kills, and process termination. |
| Hit-counter persistence | Fire-and-forget upsert per cache hit that increments `hit_count` and refreshes `updated_at`. | Hit count is the strongest predictor of which entries are worth rehydrating after restart. The cache key includes `(query, character, user_id, user_name)`, so a high `hit_count` identifies a single user's repeated question — exactly the entries that should be hot post-restart. |
| Hydration ranking | Sort by `hit_count desc, updated_at desc`, insert into LRU in reverse so highest-hit ends up MRU. | Maximizes per-session hit rate after restart. Loading lowest-priority first means high-hit rows survive longest under post-startup eviction pressure. |
| Hydration limit | `RAG_CACHE2_MAX_ENTRIES`. | Loading more than the LRU cap silently evicts during hydration, wasting reads and discarding the lowest-hit entries first (still acceptable, but pointless). |
| Long-term capacity bound | Prune to `5 * RAG_CACHE2_MAX_ENTRIES` per cache_name once at end of bootstrap. | Keeps a working-set tail beyond the LRU cap (so a hit on a cold-but-popular entry is still possible after promotion) without unbounded growth. Avoids a background sweeper, which is in `Deferred`. |
| Hit-counter overflow | Stored as MongoDB int64 via `$inc`. No periodic reset. | int64 cannot realistically overflow at chat throughput; a periodic reset adds complexity without benefit. |
| Version clearing | Delete stale-version rows during bootstrap before hydration. | Prevents old prompt/schema/agent strategies from entering memory. |
| Version key format | Pipe-joined plain text of the four version constants. | Human-readable in MongoDB inspection; deterministic; trivially diffable in logs. A SHA digest would be opaque without runtime benefit. |
| LRU eviction | Evict from memory only; never delete the persistent row on memory eviction. | An entry's accumulated `hit_count` is exactly the signal needed for the next restart's hydration ranking; deleting it on memory eviction destroys that signal. |
| Index strategy | One compound index `(cache_name, version_key, hit_count desc, updated_at desc)`. | Covers stale-purge filter, hydration filter, and hydration sort with a single index. Additional indexes are not justified at this scope. |
| Failure exception scope | Catch `pymongo.errors.PyMongoError` only. | Project py-style rule 3 forbids broad `except Exception`. `PyMongoError` is the documented base class for motor/pymongo runtime failures. |

## Persistent Cache Contract

Create `src/kazusa_ai_chatbot/db/rag_cache2_persistent.py` as the public database module for persistent Cache2 storage. Existing code must import its public helpers only.

Collection: `rag_cache2_persistent_entries`

Document shape:

```python
{
    "_id": str,              # stable cache_key (single source of truth — no separate cache_key field)
    "cache_name": str,       # initially only "rag2_initializer"
    "version_key": str,      # pipe-joined version string, see build_initializer_version_key()
    "result": dict,          # cached payload (e.g. {"unknown_slots": [...], "confidence": 1.0})
    "metadata": dict,        # forwarded verbatim from RAGCache2Runtime.store(...) metadata
    "created_at": str,       # ISO 8601 UTC, set on insert only
    "updated_at": str,       # ISO 8601 UTC, refreshed on every upsert and on every hit
    "hit_count": int,        # incremented on every cache hit via $inc
}
```

Forbidden fields in this plan: `cache_key` (redundant with `_id`), `dependencies` (always `[]` for `rag2_initializer`; can be added later when a cache type with dependencies is allowlisted), `last_hit_at` (redundant with `updated_at` once hits bump `updated_at`).

Index (exactly one):

```python
[
    ("cache_name", ASCENDING),
    ("version_key", ASCENDING),
    ("hit_count", DESCENDING),
    ("updated_at", DESCENDING),
]
# name="cache2_persistent_lookup_idx"
```

Public helpers:

```python
INITIALIZER_CACHE_NAME = "rag2_initializer"  # re-exported from cache2_policy

def build_initializer_version_key() -> str: ...

async def purge_stale_initializer_entries() -> int: ...

async def load_initializer_entries(
    *,
    limit: int = RAG_CACHE2_MAX_ENTRIES,
) -> list[dict]: ...

async def upsert_initializer_entry(
    *,
    cache_key: str,
    result: dict,
    metadata: dict,
) -> None: ...

async def record_initializer_hit(cache_key: str) -> None: ...

async def prune_persistent_entries(
    *,
    cache_name: str,
    max_entries: int,
) -> int: ...
```

Helper behavior (every helper that touches MongoDB must wrap its call in `try/except pymongo.errors.PyMongoError`, log the cache key when available or the helper/cache name otherwise, and return silently — including counts as `0` where applicable):

- `build_initializer_version_key()` returns `f"{INITIALIZER_POLICY_VERSION}|{INITIALIZER_PROMPT_VERSION}|{INITIALIZER_AGENT_REGISTRY_VERSION}|{INITIALIZER_STRATEGY_SCHEMA_VERSION}"`. No hashing, no normalization beyond the constants themselves.
- `purge_stale_initializer_entries()` deletes only rows where `cache_name == INITIALIZER_CACHE_NAME` and `version_key != build_initializer_version_key()`. Returns the deleted count for logging.
- `load_initializer_entries(*, limit=RAG_CACHE2_MAX_ENTRIES)` queries `{cache_name: INITIALIZER_CACHE_NAME, version_key: build_initializer_version_key()}`, sorts `[(hit_count, -1), (updated_at, -1)]`, applies `limit`, and returns the raw documents. Does not mutate the in-memory LRU; that is the caller's job.
- `upsert_initializer_entry(*, cache_key, result, metadata)` performs `update_one({_id: cache_key}, {"$set": {cache_name, version_key, result, metadata, updated_at}, "$setOnInsert": {created_at}, "$inc": {hit_count: 0}}, upsert=True)`. The `$inc` of zero ensures `hit_count` exists on insert without overwriting an existing counter. `cache_name` is forced to `INITIALIZER_CACHE_NAME` and `version_key` to the current value regardless of the metadata dict.
- `record_initializer_hit(cache_key)` performs `update_one({_id: cache_key}, {"$inc": {"hit_count": 1}, "$set": {"updated_at": now_iso()}})` with no upsert. A missing `_id` (e.g., the row was pruned between hydration and the hit) is a no-op and must not log as an error.
- `prune_persistent_entries(*, cache_name, max_entries)` deletes the oldest rows beyond `max_entries` for the given cache_name, ranked by `(hit_count asc, updated_at asc)` so the lowest-value rows go first. Called once at end of `db_bootstrap()` and never on the request path. Returns the deleted count for logging.

Call-site contract for fire-and-forget writes:

```python
# at the cache hit branch in rag_initializer:
asyncio.create_task(record_initializer_hit(cache_key))

# at the cache miss + cacheable result branch in rag_initializer:
await _write_initializer_cache(cache_key=cache_key, unknown_slots=unknown_slots)
asyncio.create_task(
    upsert_initializer_entry(
        cache_key=cache_key,
        result={"unknown_slots": list(unknown_slots), "confidence": 1.0},
        metadata={
            "stage": "rag_initializer",
            "initializer_prompt_version": INITIALIZER_PROMPT_VERSION,
            "agent_registry_version": INITIALIZER_AGENT_REGISTRY_VERSION,
            "strategy_schema_version": INITIALIZER_STRATEGY_SCHEMA_VERSION,
        },
    )
)
```

Both `create_task` call sites must occur after the in-memory cache mutation and before the function returns. The response coroutine must not await either task. Each helper's internal `try/except PyMongoError` is the only required error containment; the caller does not need to wrap `create_task(...)` in additional handlers.

## Change Surface

### Create

| Path | Purpose |
|---|---|
| `src/kazusa_ai_chatbot/db/rag_cache2_persistent.py` | MongoDB persistence helpers for generic persistent Cache2 entries. |
| `tests/test_rag_cache2_persistent.py` | Deterministic DB-helper and hydration/write-through tests. |

### Modify

| Path | Purpose |
|---|---|
| `src/kazusa_ai_chatbot/db/schemas.py` | Add `RAGCache2PersistentEntryDoc` TypedDict. |
| `src/kazusa_ai_chatbot/db/bootstrap.py` | Create the `rag_cache2_persistent_entries` collection, build the single compound index, drop stale initializer rows by `version_key`, run `prune_persistent_entries` to the `5×` cap, and keep dropping legacy RAG1 collections. |
| `src/kazusa_ai_chatbot/db/__init__.py` | Re-export `build_initializer_version_key`, `purge_stale_initializer_entries`, `load_initializer_entries`, `upsert_initializer_entry`, `record_initializer_hit`, `prune_persistent_entries`, and `RAGCache2PersistentEntryDoc` to match the existing DB-helper export style. |
| `src/kazusa_ai_chatbot/service.py` | After `db_bootstrap()` and before `_build_graph()`, call `load_initializer_entries(...)` and store each row into `RAGCache2Runtime` in reverse-iterated order so highest-hit lands MRU. Wrap the hydration in `try/except PyMongoError` and log row count on success or failure cause on error; do not abort startup. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` | At the cache-hit branch of `rag_initializer`, schedule `asyncio.create_task(record_initializer_hit(cache_key))` after `_read_cached_initializer_slots` returns slots and before returning. At the cache-miss branch, after `_write_initializer_cache(...)`, schedule `asyncio.create_task(upsert_initializer_entry(...))` with the cached `result` and the same metadata dict already used for the in-memory store. |
| `tests/test_rag_initializer_cache2.py` | Extend deterministic initializer cache coverage to assert that hits schedule a hit-recording task and misses schedule an upsert task, that neither is awaited on the response path, and that `PyMongoError` from either helper does not affect the returned slots. |
| `src/kazusa_ai_chatbot/rag/README.md` and `development_plans/rag_cache2_design.md` | Document that initializer strategy cache is durable, that hydration is ranked by `hit_count desc, updated_at desc`, that helper-agent result caches remain process-local, and that LRU eviction never deletes a persistent row. |

### Keep

| Path | Instruction |
|---|---|
| `src/kazusa_ai_chatbot/rag/cache2_runtime.py` | Keep MongoDB-free. Add a small hydration helper only if needed, but do not import DB code here. |
| `src/kazusa_ai_chatbot/rag/cache2_policy.py` | Keep existing initializer cache key material; do not remove version constants. |

## Implementation Order

1. Add `RAGCache2PersistentEntryDoc` to `db/schemas.py` and create `db/rag_cache2_persistent.py` with all six public helpers, the constant `INITIALIZER_CACHE_NAME` re-export (or import from `cache2_policy`), the single compound index name, and per-helper `try/except PyMongoError` wrappers. Re-export from `db/__init__.py`.
2. Extend `db/bootstrap.py` to create the collection if missing, create `cache2_persistent_lookup_idx`, call `purge_stale_initializer_entries()`, and then call `prune_persistent_entries(cache_name="rag2_initializer", max_entries=5 * RAG_CACHE2_MAX_ENTRIES)`. Order matters: purge stale rows before prune so the prune count is computed against current-version rows only.
3. Wire startup hydration in `service.lifespan()` immediately after `db_bootstrap()` and before `_build_graph()`. Iterate `load_initializer_entries(limit=RAG_CACHE2_MAX_ENTRIES)` in reverse so highest-hit is the last `RAGCache2Runtime.store(...)` call. Log the loaded row count.
4. Wire write-through and hit-recording in `nodes/persona_supervisor2_rag_supervisor2.py`. Schedule `asyncio.create_task(...)` for both call sites; do not await. Confirm by reading the response path that no MongoDB call is awaited.
5. Add tests in `tests/test_rag_cache2_persistent.py` for the DB helper module (purge, prune, load ordering, upsert idempotence, hit-counter increment, missing-key no-op, `PyMongoError` swallow). Add tests in `tests/test_rag_initializer_cache2.py` for fire-and-forget scheduling, response-path-not-awaiting behavior, and unchanged in-memory hit behavior.
6. Update `src/kazusa_ai_chatbot/rag/README.md` and `development_plans/rag_cache2_design.md` to reflect the final architecture, including the hit-count-ordered hydration policy and the LRU/persistent-row decoupling.
7. Run all verification commands listed below and record results in `Execution Evidence`.

## LLM Call And Context Budget

- Before this plan:
  - Initializer cache hit: 0 LLM calls, 0 MongoDB operations on the response path.
  - Initializer cache miss: 1 response-path initializer LLM call, 0 MongoDB operations.
  - Startup: 0 LLM calls, 0 MongoDB cache operations.
- After this plan:
  - Initializer cache hit: 0 LLM calls, 0 awaited MongoDB operations on the response path. One fire-and-forget `record_initializer_hit` upsert is scheduled via `asyncio.create_task` after the cached slots are validated and immediately before returning them.
  - Initializer cache miss: 1 response-path initializer LLM call, 0 awaited MongoDB operations on the response path. One fire-and-forget `upsert_initializer_entry` is scheduled via `asyncio.create_task` after the in-memory `store(...)` succeeds and immediately before returning slots.
  - Startup: 0 LLM calls. MongoDB cache operations: one purge, one prune, one bounded `find` (≤ `RAG_CACHE2_MAX_ENTRIES` rows), and one `RAGCache2Runtime.store(...)` per loaded row. All run before `_build_graph()` so they cannot affect chat latency.
- Response-path latency budget: unchanged. The fire-and-forget tasks are scheduled but not awaited; they execute concurrently with subsequent request work and do not extend the cache-hit or cache-miss latency. Verify with a deterministic test that asserts the response coroutine resolves before the scheduled task completes.
- Throughput cost: at ~50 cache hits/min, ~50 single-document upserts/min hit MongoDB. This is well below MongoDB write capacity and below the existing `conversation_history` and `user_profiles` write rates. No additional connection pool sizing is required.
- Context budget: unchanged. No prompts, prompt inputs, model routes, or prompt-facing payloads change.

## Verification

### Static Greps

- `rg "rag_cache_index|rag_metadata_index" src tests` shows only allowed legacy-drop references and docs/tests that assert the old collections stay removed.
- `rg "rag_cache2_persistent_entries" src tests development_plans` shows the new collection is used only by bootstrap, DB helper tests, persistence helpers, startup hydration, and docs.
- `rg "rag_cache2_persistent|get_db" src/kazusa_ai_chatbot/rag/cache2_runtime.py` returns no matches.
- `rg "except Exception" src/kazusa_ai_chatbot/db/rag_cache2_persistent.py` returns no matches (project rule: specific exception types only).
- `rg "await record_initializer_hit|await upsert_initializer_entry" src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` returns no matches (these must be `asyncio.create_task(...)` only).
- `rg "last_hit_at|cache_key.*=.*cache_key" src/kazusa_ai_chatbot/db/rag_cache2_persistent.py` returns no matches against the doc shape (forbidden fields are not introduced).

### Tests

- `pytest tests/test_rag_cache2_persistent.py -q` covering:
  - `build_initializer_version_key()` returns the pipe-joined string of the four constants.
  - `purge_stale_initializer_entries()` deletes only stale-version `rag2_initializer` rows.
  - `purge_stale_initializer_entries()` does not delete current-version rows or rows with other `cache_name` values.
  - `load_initializer_entries(limit=N)` returns at most N rows, sorted `(hit_count desc, updated_at desc)`, filtered by current `version_key`.
  - `load_initializer_entries(...)` against an empty collection returns `[]` without error.
  - `upsert_initializer_entry(...)` inserts a new row with `hit_count=0`, `created_at` set, `updated_at` set, current `version_key`, and `cache_name == INITIALIZER_CACHE_NAME`.
  - `upsert_initializer_entry(...)` on an existing `_id` updates `result`, `metadata`, `updated_at` but preserves `created_at` and `hit_count`.
  - `record_initializer_hit(cache_key)` increments `hit_count` by 1 and refreshes `updated_at`.
  - `record_initializer_hit(missing_key)` is a no-op and does not insert a row.
  - `prune_persistent_entries(cache_name, max_entries=N)` deletes oldest-by-`(hit_count asc, updated_at asc)` rows beyond the cap.
  - Each helper swallows a simulated `pymongo.errors.PyMongoError` (e.g. via a patched collection) and logs without raising.
- `pytest tests/test_rag_initializer_cache2.py -q` covering:
  - On a cache hit, `rag_initializer` returns slots before any awaited MongoDB call. Use a slow patched `record_initializer_hit` to assert the response resolves first.
  - On a cache miss with a cacheable result, `rag_initializer` calls `_write_initializer_cache` (in-memory) and schedules `upsert_initializer_entry`; `upsert` is not awaited.
  - When the patched `record_initializer_hit` raises `PyMongoError`, the cache hit still returns the correct slots.
  - When the patched `upsert_initializer_entry` raises `PyMongoError`, the cache miss still returns the LLM-derived slots.
  - In-memory cache hit makes zero `db.rag_cache2_persistent_entries.find(...)` calls (the request path stays read-free).
- Bootstrap/startup tests (`tests/test_db_bootstrap.py` or equivalent if extant; otherwise add focused cases):
  - With an empty `rag_cache2_persistent_entries`, hydration completes and the LRU stays empty without error.
  - With a mix of stale and current rows, `purge_stale_initializer_entries` runs before `prune_persistent_entries`, and only current-version rows reach `RAGCache2Runtime`.
  - With more current rows than `RAG_CACHE2_MAX_ENTRIES`, hydration loads the top-N by `(hit_count, updated_at)` and inserts them in reversed order so the highest-hit row is MRU. Verify by inspecting `RAGCache2Runtime._entries` ordering after hydration.
  - With a patched `load_initializer_entries` raising `PyMongoError`, `service.lifespan()` still completes startup and logs the failure.

### Compile

- `python -m py_compile src/kazusa_ai_chatbot/db/rag_cache2_persistent.py src/kazusa_ai_chatbot/db/schemas.py src/kazusa_ai_chatbot/db/bootstrap.py src/kazusa_ai_chatbot/db/__init__.py src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`

### Manual Smoke

- Start the brain service with an empty test MongoDB.
- Trigger one initializer miss with a cacheable result. Confirm one `rag_cache2_persistent_entries` row exists with `cache_name == "rag2_initializer"`, current `version_key`, `hit_count == 0`, and `_id == cache_key`.
- Trigger the same initializer input twice while the service is still running. Confirm the row's `hit_count` is now `2` and `updated_at` is more recent than `created_at`.
- Restart the service. Confirm startup log shows the loaded row count and that hydration completes before chat traffic.
- Trigger the same initializer input post-restart. Confirm it is served from the hydrated LRU without an initializer LLM call and that `hit_count` continues to increment (now `3`).
- Bump `INITIALIZER_PROMPT_VERSION` in code, restart. Confirm the prior row is deleted by `purge_stale_initializer_entries` and the LRU starts empty.

## Acceptance Criteria

This plan is complete when:

- `rag_cache2_persistent_entries` exists with the single compound index `cache2_persistent_lookup_idx` on `(cache_name asc, version_key asc, hit_count desc, updated_at desc)`.
- `rag2_initializer` current-version rows load from MongoDB into `RAGCache2Runtime` before chat traffic is served, ranked by `hit_count desc, updated_at desc`, and inserted in reversed order so the highest-hit row is at the LRU's MRU position.
- The hydration limit is exactly `RAG_CACHE2_MAX_ENTRIES`.
- New cacheable initializer paths are stored in the LRU first and then scheduled for fire-and-forget MongoDB upsert via `asyncio.create_task(...)`. The response coroutine never awaits the upsert.
- Every initializer cache hit schedules a fire-and-forget `record_initializer_hit(cache_key)` task that increments `hit_count` and refreshes `updated_at`. The response coroutine never awaits this task.
- Changing any of the four initializer version constants causes stale persisted initializer rows to be deleted before hydration on next startup.
- The persistent collection is bounded by a single bootstrap-time prune to `5 * RAG_CACHE2_MAX_ENTRIES` per cache_name, ranked by `(hit_count asc, updated_at asc)`.
- Normal request-path cache hits perform zero MongoDB reads.
- Every persistent-cache helper catches `pymongo.errors.PyMongoError` (and only that), logs cache key when available or helper/cache name otherwise, and returns silently. No `except Exception:` appears in the new module.
- LRU eviction never deletes a persistent row.
- Helper-agent result caches remain process-local and are not persisted.
- Legacy RAG1 cache collections (`rag_cache_index`, `rag_metadata_index`) remain dropped by `db_bootstrap()`.
- The persistent doc shape matches the contract exactly: `_id`, `cache_name`, `version_key`, `result`, `metadata`, `created_at`, `updated_at`, `hit_count` — no `cache_key`, no `dependencies`, no `last_hit_at`.
- Verification commands pass or any blocker is recorded with exact failure output in `Execution Evidence`.

## Rollback / Recovery

- Code rollback path: revert the new persistence module, startup hydration call, write-through call, schema/export additions, bootstrap collection/index setup, and docs.
- Data rollback path: drop `rag_cache2_persistent_entries`; this only removes cache data, not user memory or conversation data.
- Irreversible operations: none. The collection stores reconstructable cache entries only.
- Required backup: no production data backup is required for cache deletion, but normal deployment backup policy still applies.
- Recovery verification: after rollback or collection drop, the service should still run with process-local Cache2 behavior and `tests/test_rag_initializer_cache2.py` should pass.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Stale initializer paths survive prompt/schema/agent/schema-version changes | Pipe-joined version key includes all four initializer version constants; stale rows are deleted before hydration in `db_bootstrap()`. | DB helper stale-purge test and manual version-key smoke (bump `INITIALIZER_PROMPT_VERSION`, restart, confirm row deleted). |
| Request latency increases from DB reads | MongoDB is used only at startup, write-through after miss, and fire-and-forget hit recording — never on the awaited request path. | Test that an LRU hit does not call any `db.find(...)`/`db.find_one(...)`. Static grep that there is no `await record_initializer_hit` or `await upsert_initializer_entry` in the node. |
| Hit-recording fire-and-forget tasks block the response coroutine | All hit/miss persistence is scheduled via `asyncio.create_task(...)`; never `await`ed in the response path. | Test asserts response resolves while patched helper is still sleeping. |
| Lost persistent entry after process crash | Write-through `create_task` is scheduled immediately after the in-memory store; no shutdown flush dependency. | Write-through unit test plus manual restart smoke. |
| MongoDB outage breaks chat | Every helper wraps its MongoDB call in `try/except pymongo.errors.PyMongoError`, logs, and returns silently. Hydration failure does not abort startup. | `PyMongoError` injection test for each helper; service-startup test with patched `load_initializer_entries` raising. |
| LRU eviction destroys persistent hit history | LRU eviction runs in `RAGCache2Runtime` only; persistent rows are never deleted on memory eviction. `cache2_runtime.py` has no DB import. | Static grep `rg "rag_cache2_persistent\|get_db" src/kazusa_ai_chatbot/rag/cache2_runtime.py` returns no matches. |
| Persistent collection grows unbounded | `prune_persistent_entries(cache_name="rag2_initializer", max_entries=5 * RAG_CACHE2_MAX_ENTRIES)` runs once at end of bootstrap. | Prune test with seeded > cap rows asserts deletion to exactly the cap. |
| Hot row evicted from LRU before MRU promotion | Hydration inserts in reversed sort order so the highest-hit row is the last `store(...)` call and lands at the MRU end. | Hydration ordering test inspects `RAGCache2Runtime._entries` order after seeded mixed-`hit_count` rows. |
| Hit-counter race between cache miss-then-hit | First miss does `upsert_initializer_entry` with `$inc hit_count: 0` (creates row with `hit_count=0`); subsequent hits do `$inc hit_count: 1`. Concurrent ordering across the two ops is safe because Mongo's `$inc` is atomic and no operation overwrites `hit_count` to a literal value. | Concurrency test schedules upsert + multiple hits and asserts final `hit_count` matches the number of hits. |
| Collection becomes generic too early | Only `rag2_initializer` is allowlisted for this plan; other `cache_name` values cannot be written by the supplied helpers. | Tests assert that other `cache_name` values are not hydrated and that no helper accepts an arbitrary `cache_name` for writes. |

## Progress Checklist

- [x] Stage 1 - persistence contract added
  - Covers: `RAGCache2PersistentEntryDoc` in `db/schemas.py`; new `db/rag_cache2_persistent.py` with all six public helpers; the single compound index name; per-helper `try/except PyMongoError`; `db/__init__.py` re-exports.
  - Verify: `python -m py_compile src/kazusa_ai_chatbot/db/rag_cache2_persistent.py src/kazusa_ai_chatbot/db/schemas.py src/kazusa_ai_chatbot/db/__init__.py`; focused helper tests in `tests/test_rag_cache2_persistent.py` pass (purge, prune, load ordering, upsert idempotence, hit-counter increment, missing-key no-op, `PyMongoError` swallow).
  - Evidence: record changed files and test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-04-29` after verification and evidence are recorded.
- [x] Stage 2 - bootstrap, prune, and startup hydration wired
  - Covers: `db/bootstrap.py` (collection creation, `cache2_persistent_lookup_idx`, `purge_stale_initializer_entries`, `prune_persistent_entries` to `5×` cap); `service.lifespan()` hydration with reversed-iteration insertion.
  - Verify: bootstrap/startup tests pass for empty DB, mixed stale+current rows, and `> RAG_CACHE2_MAX_ENTRIES` rows; static grep `rg "rag_cache2_persistent|get_db" src/kazusa_ai_chatbot/rag/cache2_runtime.py` returns no matches; static grep `rg "except Exception" src/kazusa_ai_chatbot/db/rag_cache2_persistent.py` returns no matches.
  - Evidence: record test output and grep output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-04-29` after verification and evidence are recorded.
- [x] Stage 3 - write-through and hit-recording wired
  - Covers: both `asyncio.create_task(...)` call sites in `rag_initializer` (hit branch schedules `record_initializer_hit`; miss branch schedules `upsert_initializer_entry` after the in-memory store).
  - Verify: `tests/test_rag_initializer_cache2.py` passes including the response-resolves-before-task assertion and the `PyMongoError` injection assertions; static grep `rg "await record_initializer_hit|await upsert_initializer_entry" src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` returns no matches.
  - Evidence: record test output and grep output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-04-29` after verification and evidence are recorded.
- [x] Stage 4 - docs and final verification complete
  - Covers: `src/kazusa_ai_chatbot/rag/README.md` and `development_plans/rag_cache2_design.md` updates describing durable initializer cache, hit-count-ordered hydration, and LRU/persistent-row decoupling.
  - Verify: every command in `Verification` (greps, tests, compile, manual smoke) passes or blockers are recorded with exact failure output.
  - Evidence: record command output summaries in `Execution Evidence`.
  - Handoff: plan can move to implementation completion review.
  - Sign-off: `Codex/2026-04-29` after verification and evidence are recorded.

## Execution Evidence

- Static grep results:
- `rg "rag_cache_index|rag_metadata_index" src tests` returned only `src/kazusa_ai_chatbot/db/bootstrap.py` legacy drop references.
- `rg "rag_cache2_persistent_entries" src tests development_plans` returned the new persistence module and docs/plan references.
- `rg "rag_cache2_persistent|get_db" src/kazusa_ai_chatbot/rag/cache2_runtime.py` returned no matches.
- `rg "except Exception" src/kazusa_ai_chatbot/db/rag_cache2_persistent.py` returned no matches.
- `rg "await record_initializer_hit|await upsert_initializer_entry" src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` returned no matches.
- `rg "last_hit_at|cache_key.*=.*cache_key" src/kazusa_ai_chatbot/db/rag_cache2_persistent.py` returned no matches after logger wording was changed to `key=...`.
- `git diff --check` passed; it reported only line-ending conversion warnings from Git.
- Test results: `pytest tests/test_rag_cache2_persistent.py tests/test_rag_initializer_cache2.py -q` passed, 17 tests.
- Compile results: `python -m py_compile src/kazusa_ai_chatbot/db/rag_cache2_persistent.py src/kazusa_ai_chatbot/db/schemas.py src/kazusa_ai_chatbot/db/bootstrap.py src/kazusa_ai_chatbot/db/__init__.py src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` passed.
- Manual smoke: run on 2026-04-29 against live MongoDB at `mongodb://192.168.2.10:27027/?directConnection=true`. Service health returned `{"status":"ok","db":true}`; `rag_cache2_persistent_entries` existed with `cache2_persistent_lookup_idx`; a temporary current-version `_id == "live-smoke-rag-initializer-20260429"` row was inserted with `hit_count == 1`; after service restart the log showed `Hydrated 1 persistent RAG initializer cache entries`; the temporary row was deleted afterward (`remaining == 0`).
- Changed files: `src/kazusa_ai_chatbot/db/rag_cache2_persistent.py`, `src/kazusa_ai_chatbot/db/schemas.py`, `src/kazusa_ai_chatbot/db/bootstrap.py`, `src/kazusa_ai_chatbot/db/__init__.py`, `src/kazusa_ai_chatbot/service.py`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`, `tests/test_rag_cache2_persistent.py`, `tests/test_rag_initializer_cache2.py`, `src/kazusa_ai_chatbot/rag/README.md`, `development_plans/rag_cache2_design.md`, `development_plans/rag_cache2_persistent_initializer_plan.md`.
