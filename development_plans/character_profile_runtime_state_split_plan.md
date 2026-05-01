# character profile runtime state split plan

## Summary

- Goal: Split static character profile data from dynamic runtime character state in code, while preserving the existing MongoDB `character_state` document shape.
- Plan class: medium
- Status: draft
- Overall cutover strategy: compatible code refactor with no database migration and no prompt/schema output changes.
- Highest-risk areas: stale runtime mood/vibe in the service cache; accidentally dropping profile fields consumed by cognition/RAG/dialog; adding response-path latency; allowing separate runtime fields to drift.
- Acceptance criteria: normal chat turns receive a composed `character_profile` containing static profile fields plus fresh runtime fields, `mood`, `global_vibe`, and `reflection_summary` are synchronized as one bundle, the current ad hoc service cache mutation is replaced or constrained by the new runtime-state contract, and existing tests plus focused state-sync tests pass.

## Context

The current service loads the singleton MongoDB `character_state` document into `_personality` at startup. That document contains both static profile data, such as `personality_brief`, `boundary_profile`, and `linguistic_texture_profile`, and mutable runtime state, such as `mood`, `global_vibe`, `reflection_summary`, and `self_image`.

The consolidator already writes `mood`, `global_vibe`, and `reflection_summary` together through `upsert_character_state(...)`. A narrow hotfix updated `_personality` after a successful background consolidation write so the next queued turn can see the new values. That fixes service-owned writes, but it keeps mutable runtime state hidden inside a broadly named static profile cache.

The target is a code-only separation:

```text
MongoDB character_state document stays unchanged
  -> service splits the loaded document into:
       static character profile
       runtime character state
  -> each chat turn receives a composed character_profile
```

This preserves the existing graph and prompt contracts while making the runtime-state synchronization boundary explicit.

## Mandatory Rules

- Do not change the MongoDB collection name, document `_id`, field names, indexes, or persistence layout.
- Do not add a data migration.
- Do not change cognition, dialog, relevance, RAG, or consolidator LLM output schemas.
- Do not add any response-path LLM calls.
- Treat `mood`, `global_vibe`, and `reflection_summary` as one runtime-state bundle. Do not synchronize one without the others.
- Keep dynamic runtime fields as semantic descriptors before they enter LLM prompts. Do not expose low-level DB freshness metadata to LLM prompts unless a prompt already uses that metadata.
- Python edits must follow project style: imports at top, narrow `try` blocks, no broad defensive exception handling for internal bugs, docstrings for non-trivial functions, no scattered internal defaults, and no hidden runtime imports.
- The implementation agent must preserve user edits already present in the worktree and must not revert unrelated files.

## Must Do

- Add an explicit code contract for static profile vs runtime state.
- Keep `get_character_profile()` available for existing DB callers that need the full singleton document.
- Add a DB helper or projection path that retrieves only runtime character-state fields for service synchronization.
- Replace service-level use of `_personality` as a mixed static/dynamic cache with separate process-local state:
  - `_static_character_profile`
  - `_runtime_character_state`
- Compose a fresh per-turn `character_profile` from static and runtime state before building `IMProcessState`.
- Ensure composed `character_profile` still includes `global_user_id` for runtime identity.
- Ensure `mood`, `global_vibe`, and `reflection_summary` refresh before each queued chat item or after every successful consolidation write through the same runtime-state contract.
- Decide in code that an empty runtime field from consolidation means "leave cached value unchanged", matching `upsert_character_state(...)`.
- Add deterministic tests covering per-turn composition, refresh after consolidation, and no DB schema changes.
- Remove or refactor the current ad hoc field loop in `_run_consolidation_background` once the explicit runtime-state contract exists.

## Deferred

- Do not redesign `character_state` persistence.
- Do not split MongoDB collections.
- Do not migrate existing documents.
- Do not redesign `self_image` synthesis or character-image storage.
- Do not change prompt wording or LLM behavior for mood interpretation.
- Do not make RAG agents depend on the service-local cache.
- Do not refactor unrelated service queue, scheduler, adapter, or conversation-progress code.

## Cutover Policy

| Area | Policy | Notes |
|---|---|---|
| MongoDB `character_state` document | compatible | Existing document shape remains authoritative. |
| Service state build | compatible | `IMProcessState.character_profile` remains the graph-facing payload. |
| Cognition/dialog/relevance prompts | compatible | Same fields are available under `character_profile`; no prompt contract changes. |
| Consolidator persistence | compatible | `upsert_character_state(...)` continues to write the same fields. |
| Process-local cache | bigbang | Replace the mixed `_personality` cache with explicit static/runtime caches in one service refactor. |
| Tests | compatible | Existing tests should continue to pass, with focused assertions added for the new contract. |

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, the agent must preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, the agent must stop and report the blocker instead of inventing a substitute.

## Target State

The service owns two process-local objects:

```python
_static_character_profile: dict = {}
_runtime_character_state: dict = {}
```

Static profile fields are loaded at startup from the current full `character_state` document and exclude runtime-state keys. Runtime state is loaded from the same MongoDB document and includes:

```python
{
    "mood": str,
    "global_vibe": str,
    "reflection_summary": str,
    "self_image": dict | absent,
    "updated_at": str | absent,
}
```

Each chat turn builds:

```python
character_profile = {
    **_static_character_profile,
    **_runtime_character_state,
    "global_user_id": character_global_user_id,
}
```

The graph-facing state shape does not change. Existing nodes continue reading `state["character_profile"]["mood"]`, `state["character_profile"]["global_vibe"]`, and `state["character_profile"]["reflection_summary"]`.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Database schema | Keep the existing singleton document | The user explicitly prefers no DB design change at this stage. |
| Runtime-state owner | Service layer owns composition for response-path graph state | The service already builds `IMProcessState` and can keep prompts unchanged. |
| Freshness source | MongoDB remains authoritative; service cache is a performance/convenience copy | External scripts and consolidator writes should converge through the same runtime state contract. |
| Runtime bundle | Refresh `mood`, `global_vibe`, and `reflection_summary` together | These fields are generated and consumed together as character psychological background. |
| `self_image` | Treat as runtime state in code but do not redesign its writer | It is mutable state in the same document; keeping it in the runtime bundle avoids pretending it is static. |
| LLM budget | No new LLM calls and no prompt changes | This is deterministic orchestration, not a cognition redesign. |
| Response-path DB cost | Allow one small runtime-state DB read before a queued turn if needed | Correctness is more important than preserving a stale startup cache; projection keeps the read bounded. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/db/character.py`
  - Add a runtime-state projection helper, for example `get_character_runtime_state()`.
  - Add or expose constants for runtime-state field names if no better local module owns them.
- `src/kazusa_ai_chatbot/service.py`
  - Replace `_personality` with explicit static and runtime service caches.
  - Compose `character_profile` per queued chat item.
  - Refresh runtime state through the new helper at startup and at the chosen synchronization boundary.
  - Replace the ad hoc post-consolidation `_personality[field] = value` loop with the runtime-state update helper.
- `tests/test_service_background_consolidation.py`
  - Update the current cache-refresh test to assert the new runtime cache behavior.
- `tests/test_service_input_queue.py` or a new focused service-state test file
  - Add coverage that the graph receives a composed profile with static and runtime fields.
- `tests/test_db.py`
  - Add deterministic coverage for the runtime-state projection helper.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`

These consumers should keep reading `character_profile` in the same shape they already receive.

## Implementation Order

1. Add runtime/static split constants and helper functions.
   - Define the runtime-state field set in one place.
   - Add a helper that splits a full profile document into static and runtime dictionaries.
   - Add a helper that composes a graph-facing profile from static profile, runtime state, and `global_user_id`.
2. Add the DB runtime-state projection.
   - Implement a focused `get_character_runtime_state()` helper.
   - Use MongoDB projection so the response-path read does not hydrate the full profile unnecessarily.
3. Refactor service startup.
   - Load the full singleton document once.
   - Split it into `_static_character_profile` and `_runtime_character_state`.
   - Preserve startup validation that a character profile exists.
4. Refactor per-turn state build.
   - Refresh `_runtime_character_state` before building `IMProcessState` for a queued chat item.
   - Compose `character_profile` for that item.
   - Keep `character_name` sourced from the static profile name.
5. Refactor background consolidation sync.
   - After successful `character_state` write, update `_runtime_character_state` through the same helper logic or reload runtime state.
   - Do not mutate static profile fields from consolidator output.
6. Update deterministic tests.
   - Cover DB projection, split/compose behavior, per-turn graph payload, and consolidation refresh.
7. Run verification gates and record execution evidence.

## Progress Checklist

- [ ] Stage 1 - Runtime/static contract established
  - Covers: constants/helpers for runtime field set, split, and compose behavior.
  - Verify: `python -m py_compile src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/db/character.py`.
  - Evidence: record helper names and compile result in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - DB runtime projection added
  - Covers: `get_character_runtime_state()` or equivalent focused helper.
  - Verify: `pytest tests/test_db.py -q`.
  - Evidence: record test result and any allowed unrelated failures.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - Service composition integrated
  - Covers: startup split, per-turn runtime refresh, graph-facing profile composition.
  - Verify: `pytest tests/test_service_input_queue.py tests/test_service_background_consolidation.py -q`.
  - Evidence: record changed service symbols and test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - Consolidation refresh uses runtime-state contract
  - Covers: replacement of ad hoc mixed-cache mutation after background consolidation.
  - Verify: `pytest tests/test_service_background_consolidation.py -q`.
  - Evidence: record whether consolidation refresh reloads DB or updates `_runtime_character_state` directly.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - Final regression and static greps complete
  - Covers: source greps, targeted tests, and compile checks.
  - Verify: all commands in `Verification`.
  - Evidence: record command outputs in `Execution Evidence`.
  - Handoff: plan can move to completed after acceptance criteria are satisfied.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

### Static Greps

- `rg "_personality" src/kazusa_ai_chatbot/service.py tests`
  - Expected: no service runtime use remains, or only an intentional compatibility alias documented in code and tests.
- `rg "mood|global_vibe|reflection_summary" src/kazusa_ai_chatbot/service.py`
  - Expected: these fields appear through the runtime-state contract, not scattered independent cache mutation.

### Compile

- `python -m py_compile src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/db/character.py`

### Tests

- `pytest tests/test_db.py -q`
- `pytest tests/test_service_background_consolidation.py -q`
- `pytest tests/test_service_input_queue.py -q`
- `pytest tests/test_relevance_agent.py tests/test_persona_supervisor2.py -q`

### Manual Smoke

- Start the service with an existing character profile.
- Send one normal chat turn that produces a reply.
- Confirm background consolidation writes `character_state`.
- Send a second turn and confirm logs or captured state show the composed `character_profile.mood` equals the runtime state expected after consolidation.

## Acceptance Criteria

This plan is complete when:

- The MongoDB `character_state` document shape is unchanged.
- The service no longer treats one `_personality` object as both static profile and mutable runtime state.
- Each response-path graph invocation receives a composed `character_profile` with all existing static fields plus fresh runtime fields.
- `mood`, `global_vibe`, and `reflection_summary` refresh together.
- `self_image`, if present, is carried as runtime state without redesigning its writer.
- The ad hoc post-consolidation mutation of `_personality` is removed or replaced by the explicit runtime-state contract.
- Focused DB and service tests pass.
- No cognition, dialog, relevance, RAG, or consolidator prompt output schemas are changed.

## Rollback / Recovery

- Code rollback path: revert the service/db helper refactor and restore the previous startup-loaded profile behavior.
- Data rollback path: none required because this plan does not change stored data shape or migrate documents.
- Irreversible operations: none.
- Required backup: no special backup beyond normal MongoDB backup policy.
- Recovery verification: service starts, `get_character_profile()` returns the full singleton document, and one chat request reaches the graph with a complete `character_profile`.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Per-turn DB read adds latency | Use a projection helper limited to runtime fields; do not load full profile each turn | Service smoke and log timing inspection |
| Static field accidentally omitted from graph profile | Compose from full static profile plus runtime state; keep existing consumer tests | `test_persona_supervisor2.py`, relevance tests |
| Runtime fields drift because one is refreshed alone | Treat runtime fields as a named bundle and test all three together | service background consolidation test |
| External DB edits still do not appear | Refresh runtime state before each queued turn or document why post-write-only refresh is chosen | manual smoke with DB-edited runtime state |
| `self_image` stale in long-running service | Include `self_image` in runtime state projection | DB projection and composition tests |

## LLM Call And Context Budget

- Response-path LLM calls before: unchanged existing relevance/cognition/dialog calls.
- Response-path LLM calls after: unchanged.
- Background LLM calls before: unchanged existing consolidation calls.
- Background LLM calls after: unchanged.
- Prompt payload shape: unchanged graph-facing `character_profile` keys remain available.
- Context budget impact: no added prompt sections, no larger prompt contracts. Runtime values may be fresher, but not larger by design.
- Latency impact: at most one bounded MongoDB projection read before a queued response-path turn, unless implementation proves post-write refresh alone is sufficient and records that choice.

## Execution Evidence

- Static grep results:
- Compile results:
- Test results:
- Manual smoke:
- Notes:
