# cache2 agent stats health plan

## Summary

- Goal: Expose compact Cache2 per-agent hit/miss statistics through the existing `/health` data path.
- Plan class: small
- Status: completed
- Overall cutover strategy: compatible
- Highest-risk areas: preserving the existing `/health` contract; counting misses by agent without exposing cache keys, queries, user IDs, or cached results.
- Acceptance criteria: `/health` still returns existing health fields and also returns `cache2.agents[]` rows with `agent_name`, `hit_count`, `miss_count`, and `hit_rate`.

## Context

The brain service currently exposes `GET /health` from `src/kazusa_ai_chatbot/service.py`. Its response model contains only `status`, `db`, and `scheduler`.

Cache2 already has global aggregate counters in `src/kazusa_ai_chatbot/rag/cache2_runtime.py`, but the desired display is narrower and grouped by retrieval agent:

```json
{
  "cache2": {
    "agents": [
      {
        "agent_name": "user_profile_agent",
        "hit_count": 8,
        "miss_count": 2,
        "hit_rate": 0.8
      }
    ]
  }
}
```

This is agent-level stats, not individual cache-entry inspection. Cache misses occur when no entry exists for a key, so miss counting belongs to the runtime lookup path grouped by cache namespace / agent, not to live cache entries.

## Mandatory Rules

- Do not expose raw cached results, retrieval task text, search queries, cache keys, dependency scopes, user IDs, display names, message content, or metadata values through `/health`.
- Preserve the existing `/health` fields: `status`, `db`, and `scheduler`.
- Keep the new payload read-only and side-effect free.
- Keep Cache2 session-scoped. Do not add MongoDB persistence, migrations, TTLs, background jobs, or metrics exporters.
- Keep implementation local to service response modeling and Cache2 runtime statistics.
- Follow project Python style: imports at top, narrow changes, complete docstrings for new public helpers, specific exception handling where needed.
- Do not modify RAG2 routing, helper-agent prompts, cache invalidation behavior, or persistence paths.

## Must Do

- Add per-agent Cache2 counters for cache hits and cache misses.
- Expose per-agent stats through `GET /health` under `cache2.agents`.
- Each stats row must include only:
  - `agent_name`
  - `hit_count`
  - `miss_count`
  - `hit_rate`
- Use agent names suitable for display, for example `user_profile_agent` rather than the internal cache namespace when possible.
- Add focused tests for the Cache2 runtime counter behavior and the `/health` response shape.
- Update `docs/HOWTO.md` to document the enriched `/health` response.

## Deferred

- Do not build a separate HTML stats page in this plan.
- Do not add authentication or admin routing in this plan.
- Do not add per-cache-key, per-entry, per-user, per-channel, or per-query stats.
- Do not expose invalidation or eviction details in `cache2.agents`.
- Do not redesign the debug adapter. Its existing `/api/health` proxy may continue forwarding the brain service response unchanged.
- Do not add Prometheus, OpenTelemetry, or any external metrics dependency.

## Cutover Policy

Overall strategy: compatible

| Area | Policy | Instruction |
|---|---|---|
| `/health` API shape | compatible | Preserve existing fields and add `cache2`. Do not remove or rename existing fields. |
| Cache2 runtime | compatible | Add counters without changing cache lookup, store, invalidation, or eviction semantics. |
| Debug adapter | compatible | Rely on the existing `/api/health` proxy. Do not require adapter changes unless tests show serialization breaks. |
| Tests | compatible | Add tests for new behavior while keeping existing health/cache tests valid. |

## Cutover Policy Enforcement

- The implementation agent must preserve the compatible API behavior.
- The implementation agent must not introduce fallback routes or alternate stats endpoints unless this plan is revised.
- Any change from agent-level stats to cache-key or entry-level stats requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose small local naming details only if the final API contract remains exactly `cache2.agents[].agent_name`, `hit_count`, `miss_count`, and `hit_rate`.
- The agent must not introduce new architecture, storage, metrics frameworks, UI pages, authentication layers, or broad refactors.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or RAG behavior changes.
- If the current Cache2 runtime cannot map a lookup to an agent name cleanly, the agent must add the smallest explicit mapping in Cache2 policy/runtime code and report the choice.

## Target State

`GET /health` returns existing service health plus a sanitized Cache2 stats block:

```json
{
  "status": "ok",
  "db": true,
  "scheduler": true,
  "cache2": {
    "agents": [
      {
        "agent_name": "user_profile_agent",
        "hit_count": 8,
        "miss_count": 2,
        "hit_rate": 0.8
      }
    ]
  }
}
```

When an agent has no lookups, it may be absent from `agents`. When an agent has lookups and zero total count cannot occur, `hit_rate` is `0.0` for zero hits.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| API location | Extend `/health` | The user wants to reuse the health interface for display. |
| Granularity | Group by agent | Misses are meaningful at the agent lookup boundary, not at the live cache-entry boundary. |
| Payload size | Minimal rows only | The requested display needs only agent name, hit count, miss count, and hit rate. |
| Privacy | Sanitized stats only | Cache results and dependencies can contain user-derived data. |
| Persistence | Process-local counters | Cache2 is already session-scoped; stats should match cache lifetime. |

## Change Surface

### Modify

| Path | Change |
|---|---|
| `src/kazusa_ai_chatbot/rag/cache2_runtime.py` | Track per-agent hit/miss counters and expose a sanitized stats method. |
| `src/kazusa_ai_chatbot/rag/cache2_policy.py` | Add or expose a small cache-name to agent-name mapping if needed. |
| `src/kazusa_ai_chatbot/service.py` | Extend `HealthResponse` models and include Cache2 agent stats in `/health`. |
| `docs/HOWTO.md` | Document the enriched `/health` response. |

### Create Or Modify Tests

| Path | Change |
|---|---|
| `tests/test_rag_initializer_cache2.py` or new `tests/test_cache2_agent_stats.py` | Verify Cache2 per-agent hit/miss counting. |
| Existing service/API test file or new focused test | Verify `/health` includes `cache2.agents` while preserving existing fields. |

### Keep

| Path | Instruction |
|---|---|
| `src/adapters/debug_adapter.py` | Keep unchanged unless a test proves the proxy cannot forward the enriched response. |
| RAG helper agents | Keep behavior unchanged. They should not need code changes for this plan. |

## Implementation Order

1. Add per-agent stats data structures to `RAGCache2Runtime`.
2. Update cache lookup accounting so each `get(cache_key)` records hits and misses for the correct agent.
3. Add a public sanitized method such as `get_agent_stats()` returning a list of rows with `agent_name`, `hit_count`, `miss_count`, and `hit_rate`.
4. Update `HealthResponse` in `service.py` to include a `cache2` model with `agents`.
5. Update the `/health` endpoint to call `get_rag_cache2_runtime().get_agent_stats()`.
6. Add Cache2 runtime tests for hit, miss, and hit-rate calculation.
7. Add or update health endpoint tests for response shape and backward-compatible fields.
8. Update `docs/HOWTO.md`.
9. Run verification commands.

## Implementation Notes

- Current Cache2 lookups use only `cache_key`, so the runtime may need a way to associate a miss with a cache namespace / agent.
- Preferred approach: maintain an internal key-to-cache-name index when `store()` writes an entry, and add an optional `cache_name` or `agent_name` parameter to `get()` only if needed.
- If `get()` signature changes, update all Cache2 call sites deliberately:
  - `BaseRAGHelperAgent.read_cache`
  - initializer cache lookup in `persona_supervisor2_rag_supervisor2.py`
- The initializer may be represented as `rag_initializer` or `rag2_initializer`; use one stable display name and cover it in tests if counted.

## Verification

### Static Checks

- `rg "cache2.*agents|agent_name|hit_count|miss_count|hit_rate" src tests docs` shows the new contract in implementation, tests, and docs.
- `rg "raw_result|global_user_id|display_name|search_query|cache_key" src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/rag/cache2_runtime.py` must not show these fields being added to the health payload.

### Tests

- `pytest tests/test_rag_initializer_cache2.py`
- Run the new or updated Cache2 stats test file.
- Run the new or updated health endpoint test file.

### Smoke

- Start the brain service.
- Call `GET /health`.
- Confirm the response contains `status`, `db`, `scheduler`, and `cache2.agents`.
- Confirm each `cache2.agents[]` row contains only `agent_name`, `hit_count`, `miss_count`, and `hit_rate`.

## Acceptance Criteria

This plan is complete when:

- `/health` remains backward compatible for existing health consumers.
- `/health` includes `cache2.agents`.
- Cache2 agent rows expose only the four approved fields.
- Hit and miss counts are grouped by agent name.
- Hit rate is calculated as `hit_count / (hit_count + miss_count)` with safe zero handling.
- Tests cover both Cache2 stats counting and health response serialization.
- Documentation describes the enriched `/health` response.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Misses cannot be attributed to an agent | Add the smallest explicit namespace/agent parameter or mapping at the Cache2 runtime boundary. | Cache2 stats tests include a miss before any store when applicable. |
| Health endpoint leaks user-derived data | Return only four approved scalar fields per agent. | Static grep and response-shape tests. |
| Existing health clients break | Preserve existing fields and add optional nested data. | Health response test asserts existing fields remain. |
| Stats drift from cache behavior | Count only inside the Cache2 runtime lookup path. | Runtime tests exercise `get()` miss, `store()`, and `get()` hit. |

## Rollback / Recovery

- Code rollback path: revert the Cache2 stats additions and the `/health` response-model extension.
- Data rollback path: none. This plan adds only process-local counters and no persistent data.
- Irreversible operations: none.
- Recovery verification: `GET /health` returns the original fields and Cache2 behavior tests still pass.

## Execution Evidence

- Static grep results:
  - `rg "cache2.*agents|agent_name|hit_count|miss_count|hit_rate" src tests docs\HOWTO.md` found the implemented contract in service/runtime/tests/docs.
  - `rg "raw_result|global_user_id|display_name|search_query|cache_key" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\rag\cache2_runtime.py` found existing internal service/runtime fields and no added health payload fields beyond the approved four agent stats.
- Compile check:
  - `python -m py_compile src\kazusa_ai_chatbot\rag\cache2_runtime.py src\kazusa_ai_chatbot\rag\helper_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py src\kazusa_ai_chatbot\service.py tests\test_cache2_agent_stats.py tests\test_service_health.py` passed.
- Test results:
  - `pytest tests\test_cache2_agent_stats.py -q` passed.
  - `pytest tests\test_service_health.py -q` passed.
  - `pytest tests\test_rag_initializer_cache2.py -q` passed.
  - `pytest tests\test_user_profile_agent.py -q` passed.
- Health smoke response:
  - Covered by `tests\test_service_health.py::test_health_includes_cache2_agent_stats`, which verifies existing fields plus `cache2.agents`.
