# media descriptor cache plan

## Summary

- Goal: Cache the vision LLM output of `multimedia_descriptor_agent` so that identical images bypass the vision LLM call and serve a cached description + structured observation.
- Plan class: medium
- Status: completed
- Overall cutover strategy: compatible
- Highest-risk areas: cache key stability across platforms (same image may arrive with slightly different base64 from different re-encodings — accepted as separate entries); persistent collection growth without a background sweeper; stale cached descriptions surviving a vision prompt or model change.
- Acceptance criteria: identical base64 payloads produce cache hits that skip the vision LLM call; cache entries survive brain-service restarts; changing the vision descriptor prompt version or model invalidates all stale entries on next startup; the persistent collection is bounded by a configurable time-ordered oldest-first prune at bootstrap; the response path never awaits a MongoDB read or write for media descriptor cache operations.

## Context

`multimedia_descriptor_agent` in `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py` calls a vision LLM (`VISION_DESCRIPTOR_LLM_MODEL`) for every image that has `base64_data`. The LLM output is a structured JSON with `description`, `visible_text`, `salient_visual_facts`, `spatial_or_scene_facts`, and `uncertainty`. This output is deterministic for a given image + prompt + model combination.

The existing `rag_cache2_persistent_entries` collection and `RAGCache2Runtime` in-memory LRU already provide a proven two-layer cache with fire-and-forget persistence, version-gated invalidation, and startup hydration. This plan reuses that infrastructure for the media descriptor cache.

Images arrive as base64 from adapter code (Discord downloads from CDN, QQ adapter downloads similarly). If the base64 bytes differ — even for visually similar content — they are treated as distinct images. No perceptual deduplication is attempted.

## Mandatory Rules

- Keep `RAGCache2Runtime` as the hot serving cache. Normal request-path cache reads must check Python memory only, not MongoDB.
- Hit-counter persistence and write-through upserts must run as fire-and-forget background tasks via `asyncio.create_task(...)` so the response-path coroutine never awaits a MongoDB round-trip.
- Use the existing `rag_cache2_persistent_entries` collection with `cache_name == "media_descriptor"` for persistent storage. Do not create a new collection.
- The cached `result` payload must contain the full structured LLM output dict (`description`, `visible_text`, `salient_visual_facts`, `spatial_or_scene_facts`, `uncertainty`). Do not cache `base64_data` in the persistent row.
- Do not cache raw `base64_data`, user IDs, display names, channel IDs, or platform identifiers in persistent cache rows.
- Treat MongoDB writes as best-effort for chat latency. Catch `pymongo.errors.PyMongoError` around every persistent-cache write, log the failure with the cache key, and continue.
- Follow project Python style: typed helper signatures, complete docstrings for public helpers, specific exception handling, imports at top, and focused changes.
- Tests that mock LLM output may verify deterministic cache plumbing only.

## Must Do

### Cache Key

- Compute `content_hash = hashlib.sha256(base64.b64decode(base64_data)).hexdigest()`, then call `build_media_descriptor_cache_key(content_type=content_type, content_hash=content_hash)` from `cache2_policy.py`.
- Internally, `build_media_descriptor_cache_key` calls `stable_cache_key(namespace="media_descriptor", payload={"content_type": content_type, "content_hash": content_hash, "version_key": build_media_descriptor_version_key()})`.
- The `content_hash` is SHA-256 of the **decoded raw bytes**, not the base64 string, to ensure encoding-agnostic identity.
- `content_type` is included so the same raw bytes with different MIME types (unlikely but possible) produce different keys.
- `version_key` is included so a prompt or model change automatically produces different keys, making old entries unreachable without explicit purge.

### Version Key

- Add constants in `cache2_policy.py`:
  - `MEDIA_DESCRIPTOR_CACHE_NAME = "media_descriptor"`
  - `MEDIA_DESCRIPTOR_PROMPT_VERSION = "vision_descriptor:v1"` — bump when `_VISION_DESCRIPTOR_PROMPT` changes.
  - `MEDIA_DESCRIPTOR_MODEL_VERSION = "vision_model:v1"` — bump when `VISION_DESCRIPTOR_LLM_MODEL` changes.
- `build_media_descriptor_version_key()` returns `f"{MEDIA_DESCRIPTOR_PROMPT_VERSION}|{MEDIA_DESCRIPTOR_MODEL_VERSION}"`. Pipe-joined plain text for human readability in MongoDB inspection.

### Configuration

- Add `MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES` to `config.py` as a separate configurable with a default of `500`. Read from `os.environ.get("MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES", "500")`.
- Add `MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES` to `config.py` with a default of `100`. Read from `os.environ.get("MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES", "100")`. This controls how many persistent rows are loaded into the shared `RAGCache2Runtime` LRU at startup. Media descriptor entries share the process-wide LRU with RAG initializer entries and all other Cache2 namespaces; this is not a separate per-namespace cap.

### Persistent Storage

- Reuse `rag_cache2_persistent_entries` with `cache_name == "media_descriptor"`.
- Document shape follows the existing `RAGCache2PersistentEntryDoc` exactly: `_id` (cache key), `cache_name`, `version_key`, `result`, `metadata`, `created_at`, `updated_at`, `hit_count`.
- The existing compound index `cache2_persistent_lookup_idx` on `(cache_name, version_key, hit_count desc, updated_at desc)` covers the stale-purge filter and prune filter. However, media descriptor hydration sorts by `(updated_at desc)` without `hit_count`, which means the index cannot provide a sorted scan for hydration — MongoDB will filter by `(cache_name, version_key)` using the index prefix, then perform an in-memory sort on `updated_at`. At the expected collection size (≤ 500 rows for `media_descriptor`), this is acceptable and does not justify a second index.

### Persistent Helpers

Add the following to `rag/cache2_policy.py` (alongside the existing initializer cache key builder):

```python
def build_media_descriptor_cache_key(
    *,
    content_type: str,
    content_hash: str,
) -> str: ...
```

- `build_media_descriptor_cache_key()` calls `stable_cache_key(namespace="media_descriptor", payload={"content_type": content_type, "content_hash": content_hash, "version_key": build_media_descriptor_version_key()})`. The version key is included in the payload so that a prompt or model change automatically produces different keys.

Add the following public helpers to `db/rag_cache2_persistent.py`:

```python
MEDIA_DESCRIPTOR_CACHE_NAME = "media_descriptor"  # re-exported from cache2_policy

def build_media_descriptor_version_key() -> str: ...

async def purge_stale_media_descriptor_entries() -> int: ...

async def load_media_descriptor_entries(
    *,
    limit: int = MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES,
) -> list[RAGCache2PersistentEntryDoc]: ...

async def upsert_media_descriptor_entry(
    *,
    cache_key: str,
    result: dict,
    metadata: dict,
) -> None: ...

async def record_media_descriptor_hit(cache_key: str) -> None: ...

async def prune_media_descriptor_entries(
    *,
    max_entries: int,
) -> int: ...
```

Helper behavior:

- `build_media_descriptor_version_key()` returns `f"{MEDIA_DESCRIPTOR_PROMPT_VERSION}|{MEDIA_DESCRIPTOR_MODEL_VERSION}"`.
- `purge_stale_media_descriptor_entries()` deletes only rows where `cache_name == MEDIA_DESCRIPTOR_CACHE_NAME` and `version_key != build_media_descriptor_version_key()`. Returns deleted count.
- `load_media_descriptor_entries(*, limit)` queries `{cache_name: MEDIA_DESCRIPTOR_CACHE_NAME, version_key: build_media_descriptor_version_key()}`, sorts `[(updated_at, -1)]`, applies `limit`. Sorted by `updated_at desc` because the discard policy is time-based (oldest first), not hit-count-based.
- `upsert_media_descriptor_entry(*, cache_key, result, metadata)` performs `update_one({_id: cache_key}, {"$set": {cache_name, version_key, result, metadata, updated_at}, "$setOnInsert": {created_at}, "$inc": {hit_count: 0}}, upsert=True)`.
- `record_media_descriptor_hit(cache_key)` performs `update_one({_id: cache_key}, {"$inc": {"hit_count": 1}, "$set": {"updated_at": now_iso()}})` with no upsert.
- `prune_media_descriptor_entries(*, max_entries)` counts rows where `cache_name == MEDIA_DESCRIPTOR_CACHE_NAME`, calculates excess, finds the excess rows sorted by `(updated_at asc)`, and deletes them. Returns the deleted count.
- Every helper wraps MongoDB calls in `try/except PyMongoError`, logs cache key when available, and returns silently.

### Discard Policy

- When running out of space, discard entries with the **oldest `updated_at`** first. Because `record_media_descriptor_hit` refreshes `updated_at` on every cache hit, this effectively implements a "least recently accessed" eviction policy: entries that are still being hit stay young and survive pruning, while entries nobody re-sends age out. This is the intended behavior — the user requirement "discard the oldest" maps naturally to "discard least recently used" when hits refresh the timestamp.
- `prune_persistent_entries` already exists but sorts by `(hit_count asc, updated_at asc)`. For media descriptor pruning, add a new helper `prune_media_descriptor_entries(*, max_entries)` that sorts by `(updated_at asc)` so the least recently accessed entries are discarded first. This is called at bootstrap after stale-version purge.
- The prune cap is `MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES`.
- No multiplier factor like the initializer cache (no `5×`). The persistent cap is the hard cap.

### Startup Hydration

- During `service.lifespan()`, after the existing RAG initializer hydration (step 2), add a new step 2b to hydrate media descriptor cache:
  1. Call `purge_stale_media_descriptor_entries()` to delete rows with old `version_key`.
  2. Call `prune_media_descriptor_entries(max_entries=MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES)` to bound growth.
  3. Call `load_media_descriptor_entries(limit=MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES)` to load current-version rows sorted by `updated_at desc`.
  4. Iterate in reverse and call `RAGCache2Runtime.store(...)` for each row so the most recently updated entry lands at MRU.
- Log the loaded row count.
- Wrap in `try/except` that logs and continues — hydration failure must not block startup.

### Runtime Flow in `multimedia_descriptor_agent`

For each image piece with `base64_data`:

1. Compute `content_hash = hashlib.sha256(base64.b64decode(piece["base64_data"])).hexdigest()`.
2. Build `cache_key = build_media_descriptor_cache_key(content_type=piece["content_type"], content_hash=content_hash)`.
3. Probe `get_rag_cache2_runtime().get(cache_key, cache_name=MEDIA_DESCRIPTOR_CACHE_NAME, agent_name="media_descriptor")`. Note: `multimedia_descriptor_agent` is a standalone graph node function, not a `BaseRAGHelperAgent` subclass, so it must call `get_rag_cache2_runtime()` directly.
4. On **hit**: use cached `result` dict to build `description` and `image_observation`. Schedule `asyncio.create_task(record_media_descriptor_hit(cache_key))`. Skip the vision LLM call.
5. On **miss**: call the vision LLM as today. If the call succeeds and produces a valid result dict:
   - Store in memory: `get_rag_cache2_runtime().store(cache_key=..., cache_name=MEDIA_DESCRIPTOR_CACHE_NAME, result=result_dict, dependencies=[], metadata={...})`.
   - Schedule fire-and-forget persistence: `asyncio.create_task(upsert_media_descriptor_entry(cache_key=cache_key, result=result_dict, metadata={...}))`.
6. No `CacheDependency` entries — media descriptions do not depend on conversation/user state.

### Cache Hit Reconstruction

On cache hit, the cached `result` dict contains the raw structured LLM output. The hit path must:

- Extract `description` from `result.get("description", "")`.
- Call `_build_current_image_observation(result=result, description=description, source_message_id=state["platform_message_id"])` identically to the miss path.
- Produce the same `output_multimedia_input` entry shape as the miss path.

This ensures downstream consumers (`update_conversation_attachment_descriptions`, `build_text_chat_media_description_rows`, `replace_text_chat_media_percepts`) receive identical data whether the result was cached or freshly computed.

## Deferred

- Do not cache audio descriptions in this plan.
- Do not add perceptual image deduplication or similarity matching.
- Do not add admin APIs or health payloads for inspecting media descriptor cache contents.
- Do not add TTL expiration. Entries persist until version invalidation or oldest-first prune.
- Do not add a background cache sweeper.
- Do not change vision descriptor prompt behavior, cognition prompts, or consolidation prompts.
- Do not introduce new LLM calls or model retries.

## Cutover Policy

Overall strategy: compatible

| Area | Policy | Instruction |
|---|---|---|
| Runtime cache reads | compatible | Request-path reads check `RAGCache2Runtime.get(...)` only; no MongoDB fallback. |
| Cache writes (miss) | compatible | After a valid vision LLM result, store in memory first, then schedule fire-and-forget MongoDB upsert. |
| Cache writes (hit) | compatible | After a memory hit, schedule fire-and-forget `record_media_descriptor_hit`. The response coroutine never awaits the task. |
| Startup behavior | compatible | Purge stale entries, prune to cap, hydrate current-version rows into memory before serving traffic. |
| Version invalidation | compatible | Delete rows with old `version_key` at bootstrap before hydration. |
| Persistent capacity | compatible | Prune to `MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES` oldest-first at bootstrap. |
| Collection strategy | compatible | Reuse `rag_cache2_persistent_entries` with `cache_name == "media_descriptor"`. No new collection. |
| MongoDB failure handling | compatible | Catch `PyMongoError` in every helper, log with cache key, continue silently. |

## Agent Autonomy Boundaries

- The implementation agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The agent must not introduce new collections, fallback database reads on cache miss, shutdown flush behavior, or extra persistent cache types beyond `media_descriptor`.
- The agent must not modify vision descriptor prompt content, cognition prompts, or consolidation behavior.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, or broad refactors.
- If a required instruction is impossible because of current code shape, the agent must stop and report the blocker.

## Target State

```text
brain service startup
  -> db_bootstrap()
       -> (existing) purge stale initializer entries, prune, index
       -> purge_stale_media_descriptor_entries()
       -> prune_media_descriptor_entries(max_entries=MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES)
  -> (existing) hydrate RAG initializer cache
  -> hydrate media descriptor cache
       -> load_media_descriptor_entries(limit=MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES)
            sorted (updated_at desc)
       -> for row in reversed(rows):
            RAGCache2Runtime.store(...)
  -> _build_graph()
  -> serve chat traffic

multimedia_descriptor_agent image processing
  -> compute content_hash = SHA256(decoded base64 bytes)
  -> build cache_key from (namespace, content_type, content_hash, version_key)
  -> RAGCache2Runtime.get(cache_key)   [memory only, no DB]
  -> hit:
       reconstruct description + image_observation from cached result
       asyncio.create_task(record_media_descriptor_hit(cache_key))
  -> miss:
       call vision LLM
       if valid result:
         RAGCache2Runtime.store(...)
         asyncio.create_task(upsert_media_descriptor_entry(...))
       build description + image_observation from result
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Cache key material | SHA-256 of decoded bytes + content_type + version_key | Content-addressable; different encodings or MIME types produce different keys; version changes produce different keys. |
| Collection reuse | Use existing `rag_cache2_persistent_entries` | The collection was designed as a generic persistent Cache2 store with `cache_name` separation. Adding a new namespace follows the original design. |
| Discard policy | Oldest first by `updated_at` | User requirement. Simpler than hit-count-based for content-addressable media. Most recently seen images are most likely to be seen again. |
| Hydration and persistent caps | Separate configurables | Hydration cap controls how many entries enter the shared LRU at startup; persistent cap controls MongoDB growth. Media descriptor entries share the process-wide `RAGCache2Runtime` LRU with all other Cache2 namespaces. |
| No dependencies | `dependencies=[]` | Media descriptions are pure functions of (image bytes, prompt, model). They do not depend on conversation state, user profiles, or any other mutable data source. |
| Hydration sort | `updated_at desc` | Consistent with discard policy (oldest-first). Most recently updated entries are hydrated first. |
| Version key in cache key payload | Included | Ensures stale entries are naturally unreachable even before explicit purge runs. Double safety with bootstrap purge. |
| No perceptual dedup | Different base64 = different image | User requirement. Simple, correct, no image processing dependency. |

## Change Surface

### Modify

| Path | Purpose |
|---|---|
| `src/kazusa_ai_chatbot/config.py` | Add `MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES` and `MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES` config variables. |
| `src/kazusa_ai_chatbot/rag/cache2_policy.py` | Add `MEDIA_DESCRIPTOR_CACHE_NAME`, `MEDIA_DESCRIPTOR_PROMPT_VERSION`, `MEDIA_DESCRIPTOR_MODEL_VERSION`, and `build_media_descriptor_cache_key()`. The cache key builder calls `stable_cache_key` with the version key embedded in the payload. |
| `src/kazusa_ai_chatbot/db/rag_cache2_persistent.py` | Add `build_media_descriptor_version_key()`, `purge_stale_media_descriptor_entries()`, `load_media_descriptor_entries()`, `upsert_media_descriptor_entry()`, `record_media_descriptor_hit()`, and `prune_media_descriptor_entries()`. |
| `src/kazusa_ai_chatbot/db/__init__.py` | Re-export the new media descriptor persistent helpers. |
| `src/kazusa_ai_chatbot/db/bootstrap.py` | Call `purge_stale_media_descriptor_entries()` and `prune_media_descriptor_entries()` during bootstrap. |
| `src/kazusa_ai_chatbot/brain_service/cache_startup.py` | Add a `hydrate_media_descriptor_cache()` helper following the same pattern as the initializer hydration. The existing hydration helper catches `DatabaseOperationError` (not `PyMongoError`); the new helper must follow the same pattern for consistency. |
| `src/kazusa_ai_chatbot/service.py` | Call media descriptor hydration during startup (step 2b), after RAG initializer hydration. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py` | Add cache lookup before the vision LLM call and cache write after a successful call. Schedule fire-and-forget persistence. |
| `development_plans/reference/designs/rag_cache2_design.md` | Add media descriptor to the per-agent policy table and storage location. |

### Create

| Path | Purpose |
|---|---|
| `tests/test_media_descriptor_cache.py` | Deterministic tests for cache key generation, persistent helper behavior, hydration, hit/miss flow, and fire-and-forget scheduling. |

### Keep

| Path | Instruction |
|---|---|
| `src/kazusa_ai_chatbot/rag/cache2_runtime.py` | Keep MongoDB-free. No changes needed — the existing `get`/`store` API is sufficient. |
| `src/kazusa_ai_chatbot/rag/cache2_events.py` | No changes needed — media descriptor entries have no dependencies. |
| `src/kazusa_ai_chatbot/db/schemas.py` | No changes needed — `RAGCache2PersistentEntryDoc` already covers the document shape. |

## Implementation Order

1. Add version constants and cache key builder to `cache2_policy.py`. Add config variables to `config.py`.
2. Add persistent helpers to `db/rag_cache2_persistent.py`. Re-export from `db/__init__.py`.
3. Add bootstrap calls to `db/bootstrap.py` for stale purge and prune.
4. Add hydration helper to `brain_service/cache_startup.py`. Wire hydration in `service.py` lifespan.
5. Wire cache lookup and write-through in `persona_supervisor2_msg_decontexualizer.py`.
6. Add deterministic tests in `tests/test_media_descriptor_cache.py`.
7. Update `development_plans/reference/designs/rag_cache2_design.md`.
8. Run all verification commands and record results in `Execution Evidence`.

## LLM Call And Context Budget

- Before this plan:
  - Every image with `base64_data`: 1 vision LLM call, 0 MongoDB operations on the response path.
  - Startup: 0 LLM calls, 0 media-descriptor MongoDB operations.
- After this plan:
  - Cache hit (identical image seen before): 0 LLM calls, 0 awaited MongoDB operations. One fire-and-forget `record_media_descriptor_hit` scheduled.
  - Cache miss (new image): 1 vision LLM call, 0 awaited MongoDB operations. One fire-and-forget `upsert_media_descriptor_entry` scheduled.
  - Startup: 0 LLM calls. MongoDB operations: one purge, one prune, one bounded `find` (≤ `MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES` rows), and one `RAGCache2Runtime.store(...)` per loaded row.
- Response-path latency budget: unchanged for cache miss; eliminated vision LLM latency on cache hit. Fire-and-forget tasks do not extend response latency.
- Context budget: unchanged. No prompts or prompt-facing payloads change.

## Verification

### Static Greps

- `rg "MEDIA_DESCRIPTOR_CACHE_NAME" src tests` shows usage only in `cache2_policy.py`, `rag_cache2_persistent.py`, `persona_supervisor2_msg_decontexualizer.py`, `cache_startup.py`, `bootstrap.py`, `__init__.py`, test files, and docs.
- `rg "rag_cache2_persistent|get_db" src/kazusa_ai_chatbot/rag/cache2_runtime.py` returns no matches (unchanged from current).
- `rg "except Exception" src/kazusa_ai_chatbot/db/rag_cache2_persistent.py` returns no matches.
- `rg "await record_media_descriptor_hit|await upsert_media_descriptor_entry" src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py` returns no matches (must be `asyncio.create_task` only).

### Tests

- `pytest tests/test_media_descriptor_cache.py -q` covering:
  - `build_media_descriptor_version_key()` returns the pipe-joined version string.
  - `build_media_descriptor_cache_key()` produces stable keys: same input = same key, different content = different key, different content_type = different key, different version = different key.
  - `purge_stale_media_descriptor_entries()` deletes only stale-version `media_descriptor` rows and leaves current-version rows and other cache names intact.
  - `load_media_descriptor_entries(limit=N)` returns at most N rows sorted `updated_at desc`, filtered by current `version_key`.
  - `load_media_descriptor_entries(...)` against empty collection returns `[]`.
  - `upsert_media_descriptor_entry(...)` inserts new row with `hit_count=0`, current `version_key`, `cache_name == MEDIA_DESCRIPTOR_CACHE_NAME`.
  - `upsert_media_descriptor_entry(...)` on existing `_id` updates `result`, `metadata`, `updated_at` but preserves `created_at` and `hit_count`.
  - `record_media_descriptor_hit(cache_key)` increments `hit_count` by 1 and refreshes `updated_at`.
  - `record_media_descriptor_hit(missing_key)` is a no-op.
  - `prune_media_descriptor_entries(max_entries=N)` deletes oldest-by-`updated_at` rows beyond the cap.
  - Each helper swallows `PyMongoError` and logs without raising.
  - Cache hit in `multimedia_descriptor_agent` returns result without calling vision LLM.
  - Cache miss calls vision LLM and schedules fire-and-forget upsert.
  - `PyMongoError` from persistence helper does not affect the returned description.
  - Cache hit schedules `record_media_descriptor_hit` via `create_task`, not `await`.

### Compile

- `python -m py_compile src/kazusa_ai_chatbot/config.py src/kazusa_ai_chatbot/rag/cache2_policy.py src/kazusa_ai_chatbot/db/rag_cache2_persistent.py src/kazusa_ai_chatbot/db/__init__.py src/kazusa_ai_chatbot/db/bootstrap.py src/kazusa_ai_chatbot/brain_service/cache_startup.py src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`

### Manual Smoke

- Start the brain service with an empty test MongoDB.
- Send an image through the debug adapter. Confirm vision LLM is called and one `rag_cache2_persistent_entries` row exists with `cache_name == "media_descriptor"`, current `version_key`, `hit_count == 0`.
- Send the same image again. Confirm the vision LLM is **not** called, the cached description is used, and `hit_count` increments to `1`.
- Restart the service. Confirm startup log shows media descriptor cache hydration count.
- Send the same image post-restart. Confirm it is served from the hydrated LRU without a vision LLM call.
- Bump `MEDIA_DESCRIPTOR_PROMPT_VERSION`, restart. Confirm stale row is purged and the LRU starts empty for media descriptors.

## Acceptance Criteria

This plan is complete when:

- Identical base64 image payloads produce in-memory cache hits that skip the vision LLM call.
- Cached results produce identical `description`, `image_observation`, and downstream state as fresh LLM calls.
- Media descriptor cache entries persist in `rag_cache2_persistent_entries` with `cache_name == "media_descriptor"` and survive brain-service restarts via startup hydration.
- Changing `MEDIA_DESCRIPTOR_PROMPT_VERSION` or `MEDIA_DESCRIPTOR_MODEL_VERSION` causes all stale entries to be purged on next startup.
- Persistent collection is bounded by `MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES` with oldest-first (by `updated_at`) discard at bootstrap.
- Startup hydration loads at most `MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES` rows into the shared `RAGCache2Runtime` LRU.
- Normal request-path cache hits perform zero MongoDB reads.
- Every persistent helper catches `PyMongoError` only and returns silently.
- LRU eviction never deletes a persistent row.
- Fire-and-forget writes are scheduled via `asyncio.create_task(...)` and never awaited on the response path.
- All verification commands pass or blockers are recorded in `Execution Evidence`.

## Rollback / Recovery

- Code rollback: revert changes to `config.py`, `cache2_policy.py`, `rag_cache2_persistent.py`, `__init__.py`, `bootstrap.py`, `cache_startup.py`, `service.py`, `persona_supervisor2_msg_decontexualizer.py`, and docs.
- Data rollback: delete media descriptor rows with `db.rag_cache2_persistent_entries.deleteMany({cache_name: "media_descriptor"})`. This only removes reconstructable cache entries.
- Irreversible operations: none.
- Recovery verification: after rollback, the service should run with uncached vision LLM calls for every image.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Stale descriptions survive prompt/model change | Version key includes prompt and model version constants; stale rows purged at bootstrap; version key is also part of the cache key payload so stale entries are naturally unreachable. | Stale-purge test; manual smoke with version bump. |
| Response latency from DB reads on cache hit | MongoDB used only at startup and fire-and-forget writes; never on awaited request path. | Static grep confirms no `await record_media_descriptor_hit` or `await upsert_media_descriptor_entry` in the node. |
| Persistent collection grows unbounded | `prune_media_descriptor_entries()` runs at bootstrap with oldest-first discard to `MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES`. | Prune test with seeded rows beyond cap. |
| Base64 decoding overhead on cache key computation | SHA-256 of decoded bytes adds a `base64.b64decode` + hash per image. The LLM call uses the raw base64 string in the data URI, so the decode is new work. For a typical 1–5 MB image, decode + SHA-256 takes ~5–15 ms — negligible compared to the ~1–3 s vision LLM call it may skip. | No explicit test needed; computational overhead is negligible relative to saved LLM latency. |
| MongoDB outage breaks chat | Every helper catches `PyMongoError`, logs, continues. Hydration failure does not block startup. Vision LLM fallback always available on cache miss. | `PyMongoError` injection test for each helper. |
| Cache hit reconstruction produces different output than LLM path | Hit path calls `_build_current_image_observation(result=cached_result, ...)` identically to the miss path. | Test asserts identical `output_multimedia_input` for hit vs miss. |

## Progress Checklist

- [x] Stage 1 - cache policy and config
  - Covers: `MEDIA_DESCRIPTOR_CACHE_NAME`, version constants, `build_media_descriptor_cache_key()` in `cache2_policy.py`; `MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES` and `MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES` in `config.py`.
  - Verify: `python -m py_compile src/kazusa_ai_chatbot/config.py src/kazusa_ai_chatbot/rag/cache2_policy.py`
  - Handoff: next stage starts at Stage 2.
- [x] Stage 2 - persistent helpers
  - Covers: all seven media descriptor helpers in `db/rag_cache2_persistent.py` (`build_media_descriptor_version_key`, `purge_stale_media_descriptor_entries`, `load_media_descriptor_entries`, `upsert_media_descriptor_entry`, `record_media_descriptor_hit`, `prune_media_descriptor_entries`, plus the `MEDIA_DESCRIPTOR_CACHE_NAME` constant); `db/__init__.py` re-exports.
  - Verify: focused helper tests pass.
  - Handoff: next stage starts at Stage 3.
- [x] Stage 3 - bootstrap and hydration
  - Covers: `db/bootstrap.py` (stale purge, prune); `brain_service/cache_startup.py` hydration helper; `service.py` lifespan wiring.
  - Verify: startup/hydration tests pass.
  - Handoff: next stage starts at Stage 4.
- [x] Stage 4 - runtime integration
  - Covers: cache lookup and write-through in `persona_supervisor2_msg_decontexualizer.py`.
  - Verify: hit/miss flow tests pass; static greps confirm no awaited persistence calls.
  - Handoff: next stage starts at Stage 5.
- [x] Stage 5 - tests and code review
  - Covers: 14 deterministic tests; code review against py-style positive/negative constraints.
  - Verify: all commands in `Verification` pass; review issues addressed.
  - Handoff: plan moves to completion review.
- [x] Stage 6 - docs update and manual smoke waived during archival
  - Covers: `rag_cache2_design.md` update; manual smoke tests.
  - Verify: not run; earlier production implementation and deterministic
    verification were already complete, and this stage was optional cleanup
    scope.
  - Handoff: plan complete.

## Execution Evidence

### Implementation (2026-06-06)

Branch: `feature/media-descriptor-cache`

**Stages 1–4 committed** (`bcddbb9`): all production code changes.

**Stage 5 committed** (`6367b6f`): 15 deterministic tests, all passing.

**Code review committed** (`3546cdb`): addressed 5 issues found by independent review against py-style constraints:

| Issue | Constraint | Fix |
|---|---|---|
| Duplicate `build_media_descriptor_version_key` in `cache2_policy.py` and `rag_cache2_persistent.py` | N-011 | Removed from `cache2_policy.py`; inlined version key in `build_media_descriptor_cache_key`. Authoritative copy in `rag_cache2_persistent.py`. |
| `hydrate_media_descriptor_cache` near-identical to `hydrate_rag_initializer_cache` in `cache_startup.py` | N-011, N-012 | Merged into single `hydrate_persistent_cache` with `label` parameter. Removed duplicate `LoadMediaDescriptorEntries` type alias. |
| Import ordering: `rag.*` imports between `nodes.*` imports | PEP 8 | Moved `nodes.referent_resolution` import above `rag.*` imports. |
| Unguarded `base64.b64decode` on adapter data | N-003 | Added `except binascii.Error` guard; skips caching on corrupt data, falls through to LLM call. |
| Unused `patch` import and `base64`/`hashlib` in test file | PEP 8 | Removed unused imports. |

### Static Verification

- `MEDIA_DESCRIPTOR_CACHE_NAME` usage: `cache2_policy.py`, `rag_cache2_persistent.py`, `persona_supervisor2_msg_decontexualizer.py`, `service.py`, `test_media_descriptor_cache.py` — **PASS**
- `rg "rag_cache2_persistent|get_db" cache2_runtime.py` — no matches — **PASS**
- `rg "except Exception" rag_cache2_persistent.py` — no matches — **PASS**
- `rg "await record_media_descriptor_hit|await upsert_media_descriptor_entry" persona_supervisor2_msg_decontexualizer.py` — no matches — **PASS**

### Test Results

- `pytest tests/test_media_descriptor_cache.py` — 14 passed
- `pytest tests/test_rag_cache2_persistent.py` — 8 passed (no regressions)
- `pytest tests/test_cache2_agent_stats.py` — 2 passed (no regressions)
- All 24 cache-related tests pass.

### Compile Verification

All 8 production files compile clean with `py_compile`.

### Remaining

- Manual smoke tests were not run before archival; production implementation and deterministic verification were already complete, and docs/smoke were optional cleanup scope.

- Archived on 2026-06-10 during active-plan cleanup.
