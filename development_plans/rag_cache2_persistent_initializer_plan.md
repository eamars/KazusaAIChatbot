# rag cache2 persistent initializer plan

## Summary

- Goal: Persist the RAG2 initializer strategy cache in MongoDB so cached paths survive brain-service restarts.
- Plan class: medium
- Status: draft
- Overall cutover strategy: compatible
- Highest-risk areas: stale initializer strategies after prompt/schema/agent changes; adding database persistence without turning normal cache reads into database reads; avoiding accidental persistence of helper-agent result caches.
- Acceptance criteria: current-version `rag2_initializer` entries hydrate into the Python LRU at startup, new cacheable initializer paths write through to MongoDB immediately, stale-version entries are cleared before hydration, and helper-agent caches remain process-local.

## Context

`rag_initializer` currently stores strategy payloads only in `RAGCache2Runtime`, a process-local LRU in `src/kazusa_ai_chatbot/rag/cache2_runtime.py`. This means a repeated query can skip the initializer LLM during one brain-service process, but the learned path disappears after restart.

The initializer cache key is built by `build_initializer_cache_key(...)` in `src/kazusa_ai_chatbot/rag/cache2_policy.py`. The key already includes `INITIALIZER_POLICY_VERSION`, `INITIALIZER_PROMPT_VERSION`, `INITIALIZER_AGENT_REGISTRY_VERSION`, and `INITIALIZER_STRATEGY_SCHEMA_VERSION`. Those constants define whether a persisted initializer strategy is still valid.

This plan intentionally persists only initializer strategy cache entries. It does not restore the deleted RAG1 MongoDB write-through cache, and it does not persist helper-agent result caches.

## Mandatory Rules

- Keep `RAGCache2Runtime` as the hot serving cache. Normal request-path cache reads must check Python memory only, not MongoDB.
- Use MongoDB as durable backing storage for startup hydration and write-through persistence.
- Persist new cacheable initializer paths immediately after generation. Do not wait for brain-service shutdown to flush entries.
- Treat MongoDB writes as best-effort for chat latency: log persistence failures and continue after the in-memory LRU store succeeds.
- Do not persist raw `original_query`, raw runtime context, prompt text, user message text, display names, user IDs, channel IDs, or retrieved evidence in persistent cache rows.
- Do not persist helper-agent result caches, dispatcher output, evaluator output, finalizer output, web search results, or any Cache2 entries other than explicitly allowed `rag2_initializer` entries.
- Do not resurrect legacy RAG1 collections. `rag_cache_index` and `rag_metadata_index` must continue to be dropped by bootstrap.
- Follow project Python style: typed helper signatures, complete docstrings for public helpers, specific exception handling, imports at top, and focused changes.
- Tests that mock LLM output may verify deterministic cache plumbing only. They must not be used as evidence that a prompt semantically works.

## Must Do

- Create one generic persistent Cache2 collection named `rag_cache2_persistent_entries`.
- Initially allow only `cache_name == "rag2_initializer"` to write to and hydrate from this collection.
- Add a deterministic version key for initializer entries derived from the current initializer policy, prompt, agent-registry, and strategy-schema versions.
- During startup bootstrap, delete persisted `rag2_initializer` rows whose `version_key` differs from the current initializer version key.
- During service startup, load current-version initializer rows from MongoDB into `RAGCache2Runtime` before chat traffic is served.
- On an initializer cache miss that produces a cacheable strategy, store in Python LRU first and then immediately upsert the same strategy into MongoDB.
- Add deterministic unit tests for version clearing, startup hydration, write-through, and no-DB-read request-path behavior.
- Update RAG/Cache2 docs to describe persistent initializer strategy cache and the LRU/database relationship.

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
| Cache writes | compatible | After a cacheable initializer miss, write to memory first, then best-effort upsert to MongoDB. |
| Startup behavior | compatible | Hydrate current-version persistent initializer entries into memory before serving traffic. |
| Version invalidation | compatible | Delete stale initializer rows at bootstrap before hydration. Existing in-memory invalidation behavior is unchanged. |
| Collection strategy | compatible | Use one generic persistent Cache2 collection with `cache_name` separation so future cache types can reuse the storage layer. |

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
  -> delete stale rag2_initializer rows by version_key
  -> load current-version rag2_initializer rows
  -> store loaded rows into Python RAGCache2Runtime LRU
  -> serve chat traffic

normal initializer request
  -> check Python LRU only
  -> hit: return cached unknown_slots, no DB read, no LLM call
  -> miss: call initializer LLM
  -> validate/cacheable payload
  -> store in Python LRU
  -> best-effort upsert to MongoDB immediately
```

The Python LRU remains the fast serving layer. MongoDB makes selected initializer strategy entries survive restarts and supplies startup hydration only.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Collection layout | Use `rag_cache2_persistent_entries` for persistent Cache2 entries. | One generic collection supports future cache types without new collection plumbing for each type. |
| Initial persistence scope | Persist only `rag2_initializer`. | Initializer strategies are stable across ordinary DB writes and are the user-requested target. |
| Read path | No MongoDB lookup during normal requests. | Keeps cache hits low-latency and prevents DB latency from entering the response path. |
| Write timing | Write-through immediately after each new cacheable initializer path is generated. | Shutdown flushing is unreliable across crashes, deploy kills, and process termination. |
| Write failure behavior | Log and continue when MongoDB upsert fails. | The in-memory cache still benefits the current process; persistence is an optimization. |
| Version clearing | Delete stale-version rows during bootstrap before hydration. | Prevents old prompt/schema/agent strategies from entering memory. |
| LRU eviction | Evict from memory only. | A memory eviction should not erase durable restart benefit unless explicit pruning later deletes old rows. |

## Persistent Cache Contract

Create `src/kazusa_ai_chatbot/db/rag_cache2_persistent.py` as the public database module for persistent Cache2 storage. Existing code must import its public helpers only.

Collection: `rag_cache2_persistent_entries`

Document shape:

```python
{
    "_id": str,              # same as cache_key
    "cache_key": str,
    "cache_name": str,       # initially only "rag2_initializer"
    "version_key": str,
    "result": dict,
    "dependencies": list[dict],
    "metadata": dict,
    "created_at": str,
    "updated_at": str,
    "last_hit_at": str | None,
    "hit_count": int,
}
```

Public helpers:

```python
def build_initializer_version_key() -> str: ...

async def purge_stale_initializer_entries() -> int: ...

async def load_initializer_entries(*, limit: int) -> list[dict]: ...

async def upsert_initializer_entry(
    *,
    cache_key: str,
    result: dict,
    metadata: dict,
) -> None: ...

async def prune_persistent_entries(
    *,
    cache_name: str,
    max_entries: int,
) -> int: ...
```

Helper behavior:

- `build_initializer_version_key()` must derive a stable string or digest from `INITIALIZER_POLICY_VERSION`, `INITIALIZER_PROMPT_VERSION`, `INITIALIZER_AGENT_REGISTRY_VERSION`, and `INITIALIZER_STRATEGY_SCHEMA_VERSION`.
- `purge_stale_initializer_entries()` must delete only rows where `cache_name == "rag2_initializer"` and `version_key != build_initializer_version_key()`.
- `load_initializer_entries(...)` must return only current-version `rag2_initializer` rows, sorted by most useful restart candidates. Use `last_hit_at` descending where present, then `updated_at` descending.
- `upsert_initializer_entry(...)` must set `_id` and `cache_key` to the stable cache key, set `cache_name` to `INITIALIZER_CACHE_NAME`, set the current `version_key`, preserve `created_at` on update, update `updated_at`, and initialize `hit_count` to `0` when inserting.
- `prune_persistent_entries(...)` is used only for startup or write-through cleanup. It must remove oldest rows over the cap for the supplied cache name and must not run on every normal cache read.

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
| `src/kazusa_ai_chatbot/db/bootstrap.py` | Create collection/indexes, drop stale initializer rows, keep dropping legacy RAG1 collections. |
| `src/kazusa_ai_chatbot/db/__init__.py` | Re-export new public persistence helpers only if local style expects DB helpers to be exported. |
| `src/kazusa_ai_chatbot/service.py` | Hydrate current-version initializer rows into `RAGCache2Runtime` after bootstrap and before graph traffic. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` | Add best-effort MongoDB write-through after `_write_initializer_cache(...)` stores in memory. |
| `tests/test_rag_initializer_cache2.py` | Extend deterministic initializer cache coverage for hydration/write-through behavior where appropriate. |
| `src/kazusa_ai_chatbot/rag/README.md` and `development_plans/rag_cache2_design.md` | Document that initializer strategy cache is durable while helper-agent result caches remain process-local. |

### Keep

| Path | Instruction |
|---|---|
| `src/kazusa_ai_chatbot/rag/cache2_runtime.py` | Keep MongoDB-free. Add a small hydration helper only if needed, but do not import DB code here. |
| `src/kazusa_ai_chatbot/rag/cache2_policy.py` | Keep existing initializer cache key material; do not remove version constants. |

## Implementation Order

1. Add `RAGCache2PersistentEntryDoc` and the new `db/rag_cache2_persistent.py` helper module.
2. Add bootstrap collection/index creation and stale initializer deletion.
3. Add startup hydration in `service.lifespan()` immediately after `db_bootstrap()` and before `_build_graph()`.
4. Add initializer write-through by calling `upsert_initializer_entry(...)` after the in-memory `store(...)` succeeds.
5. Add tests for the DB helper module, startup hydration behavior, write-through behavior, and unchanged no-DB-read hit behavior.
6. Update docs to reflect the final architecture.
7. Run verification commands and record results in `Execution Evidence`.

## LLM Call And Context Budget

- Before this plan:
  - Initializer cache hit: 0 LLM calls.
  - Initializer cache miss: 1 response-path initializer LLM call.
  - Startup: 0 LLM calls.
- After this plan:
  - Initializer cache hit: 0 LLM calls.
  - Initializer cache miss: 1 response-path initializer LLM call, followed by best-effort MongoDB upsert.
  - Startup: 0 LLM calls, plus MongoDB cache hydration.
- Context budget is unchanged. No prompts, prompt inputs, model routes, or prompt-facing payloads should change.

## Verification

### Static Greps

- `rg "rag_cache_index|rag_metadata_index" src tests` shows only allowed legacy-drop references and docs/tests that assert the old collections stay removed.
- `rg "rag_cache2_persistent_entries" src tests development_plans` shows the new collection is used only by bootstrap, DB helper tests, persistence helpers, startup hydration, and docs.
- `rg "rag_cache2_persistent|get_db" src/kazusa_ai_chatbot/rag/cache2_runtime.py` returns no matches.

### Tests

- `pytest tests/test_rag_cache2_persistent.py -q`
- `pytest tests/test_rag_initializer_cache2.py -q`
- Run any changed bootstrap/service-health tests if startup wiring touches their fixtures.

### Compile

- `python -m py_compile src/kazusa_ai_chatbot/db/rag_cache2_persistent.py src/kazusa_ai_chatbot/db/bootstrap.py src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`

### Manual Smoke

- Start the brain service with an empty or test MongoDB.
- Trigger one initializer miss with a cacheable result.
- Confirm one `rag_cache2_persistent_entries` row exists with `cache_name == "rag2_initializer"` and current `version_key`.
- Restart the service.
- Trigger the same initializer input and confirm it is served from the Python LRU hydrated at startup without an initializer LLM call.

## Acceptance Criteria

This plan is complete when:

- `rag_cache2_persistent_entries` exists and has indexes supporting `cache_name`, `version_key`, and hydration order.
- `rag2_initializer` current-version rows load from MongoDB into `RAGCache2Runtime` before chat traffic is served.
- New cacheable initializer paths are stored in memory and immediately upserted to MongoDB.
- Changing any initializer version constant causes stale persisted initializer rows to be deleted before hydration.
- Normal request-path cache hits do not read from MongoDB.
- Helper-agent result caches remain process-local and are not persisted.
- Legacy RAG1 cache collections remain dropped.
- Verification commands pass or any blocker is recorded with exact failure output.

## Rollback / Recovery

- Code rollback path: revert the new persistence module, startup hydration call, write-through call, schema/export additions, bootstrap collection/index setup, and docs.
- Data rollback path: drop `rag_cache2_persistent_entries`; this only removes cache data, not user memory or conversation data.
- Irreversible operations: none. The collection stores reconstructable cache entries only.
- Required backup: no production data backup is required for cache deletion, but normal deployment backup policy still applies.
- Recovery verification: after rollback or collection drop, the service should still run with process-local Cache2 behavior and `tests/test_rag_initializer_cache2.py` should pass.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Stale initializer paths survive prompt/schema changes | Version key includes all initializer version constants and stale rows are deleted before hydration. | DB helper stale-purge test and manual version-key smoke. |
| Request latency increases from DB reads | MongoDB is used only at startup and write-through after miss; cache hits remain memory-only. | Test that LRU hit does not call DB helper. |
| Lost persistent entry after process crash | Write-through occurs immediately after a new cacheable path is generated, not at shutdown. | Write-through unit test. |
| MongoDB outage breaks chat | Write-through errors are logged and swallowed after memory store succeeds. | Test persistence failure path still returns initializer result. |
| Collection becomes generic too early | Only `rag2_initializer` is allowlisted for this plan. | Tests assert other cache names are not hydrated/written by new initializer path. |

## Progress Checklist

- [ ] Stage 1 - persistence contract added
  - Covers: schema and `db/rag_cache2_persistent.py`.
  - Verify: `python -m py_compile src/kazusa_ai_chatbot/db/rag_cache2_persistent.py`; focused helper tests pass.
  - Evidence: record changed files and test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - bootstrap and startup hydration wired
  - Covers: `db/bootstrap.py`, `service.py`, and collection/index setup.
  - Verify: bootstrap/startup tests pass; static grep confirms `cache2_runtime.py` has no DB import.
  - Evidence: record test output and grep output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - initializer write-through wired
  - Covers: `_write_initializer_cache(...)` integration and failure handling.
  - Verify: `tests/test_rag_initializer_cache2.py` and write-through failure-path test pass.
  - Evidence: record test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - docs and final verification complete
  - Covers: RAG docs/design updates and full verification section.
  - Verify: all commands in `Verification` pass or blockers are recorded.
  - Evidence: record command output summaries in `Execution Evidence`.
  - Handoff: plan can move to implementation completion review.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Execution Evidence

- Static grep results:
- Test results:
- Compile results:
- Manual smoke:
- Changed files:
