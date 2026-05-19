# character self image rolling state bugfix plan

## Summary

- Goal: Stop `character_state.self_image` from being overwritten from an empty
  or prompt-projected character profile, so `recent_window`,
  `historical_summary`, and `meta.synthesis_count` accumulate correctly.
- Plan class: large
- Status: completed
- Mandatory skills: `py-style`, `cjk-safety`, `test-style-and-execution`,
  `local-llm-architecture`, `systematic-debugging`.
- Overall cutover strategy: bigbang for write behavior; no compatibility path.
- Highest-risk areas: production `character_state.self_image` overwrite
  semantics, self-cognition internal-thought consolidation, Cache2
  invalidation behavior, tests that patch private image update helpers.
- Acceptance criteria: DB-current self-image is the merge base for every
  character-image write; stale or prompt-projected profiles cannot reset the
  rolling image; focused deterministic tests prove internal-thought
  consolidation preserves existing recent and historical state.

## Context

Production evidence from `kazusa_bot_core`:

- `character_state` has one global document.
- `character_state.self_image.historical_summary` is `""`.
- `character_state.self_image.recent_window` length is `1`.
- `character_state.self_image.meta.synthesis_count` is `1`.
- `conversation_history` has `50,492` rows.
- `event_log_events` has `47` `group_chat_trigger_review` records with
  `consolidation_outcome.write_success.character_image=true`.
- All inspected `group_chat_trigger_review` character-image writes came from
  `origin_trigger_source="internal_thought"`.
- Three successful internal-thought `character_image=true` event records had
  `event_log_events.occurred_at` later than the current
  `character_state.self_image.meta.last_updated` timestamp.

Current code shape:

- `db_writer` writes `character_state.self_image` through
  `upsert_character_self_image(...)`.
- `_update_character_image(...)` builds the next full image document from
  `state["character_profile"].get("self_image")`.
- Normal chat builds `character_profile` from service runtime state, which is
  refreshed from DB before processing a chat request.
- Self-cognition cases receive a prompt-safe projected character profile from
  `self_cognition.sources._project_character_profile(...)`.
- That projection intentionally excludes `self_image`.
- Self-cognition still uses the shared consolidator and is currently allowed
  to write `character_image`.

Failure mode:

```text
self-cognition source collector
  -> projects character_profile without self_image
  -> runner builds internal_thought consolidation state
  -> db_writer allows character_image write
  -> _update_character_image sees no state.character_profile.self_image
  -> existing_image = {}
  -> writes full self_image with one recent_window item and synthesis_count=1
  -> previous recent_window / historical_summary are replaced
```

This is a contract bug at the persistence boundary. Prompt-facing or worker
case profiles are not authoritative durable state. The character-image writer
must merge against the current DB self-image immediately before writing.

## Mandatory Skills

- `py-style`: load before editing Python files. Follow project fail-fast,
  explicit data-shape, default, docstring, and exception-handling rules.
- `cjk-safety`: load before editing Python files that contain CJK prompt text,
  especially `persona_supervisor2_consolidator_images.py`.
- `test-style-and-execution`: load before adding, changing, or running tests.
  Use deterministic tests for this fix; do not run live LLM tests unless the
  user explicitly asks.
- `local-llm-architecture`: load before changing any LLM-adjacent pipeline
  contract. This plan must not add LLM calls or expand prompt payloads.
- `systematic-debugging`: load before implementing, because this is a
  production failure mode. Preserve the RCA and verify the exact failure.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan
  before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.
- Use `venv\Scripts\python` for tests and Python commands.
- Use `rg` for searches.
- Use `apply_patch` for manual file edits.
- Do not read `.env`.
- Do not mutate production MongoDB as part of implementation verification.
- Do not add a new LLM call, prompt field, retry path, self-cognition route,
  feature flag, compatibility shim, or fallback writer.
- Do not add `self_image` to self-cognition prompt-facing case profiles. That
  expands background prompt context and does not fix the persistence ownership
  bug.
- Do not disable self-cognition as the code fix. Disabling self-cognition is an
  operator mitigation only.
- Do not backfill or invent `historical_summary` from sanitized event logs.
  Event logs do not contain the lost self-image summaries.
- The DB-current `self_image` read for character-image synthesis must be inside
  the same failure-isolated character-image update boundary as
  `_update_character_image(...)`. A runtime-state read failure must set
  `write_success.character_image=false`, skip `upsert_character_self_image(...)`,
  and not abort `db_writer`.

## Must Do

- Make the current MongoDB `character_state.self_image` the base for every
  character-image rolling merge.
- Remove `_update_character_image(...)`'s dependency on
  `state["character_profile"].self_image` as durable rolling state.
- Preserve existing behavior of the image session-summary LLM and compression
  LLM. The prompt contracts must not change.
- Preserve origin policy: internal-thought consolidations may still write
  `character_image` when the existing origin policy allows it.
- Add deterministic tests that reproduce the production failure shape:
  internal-thought consolidation state has a `character_profile` with no
  `self_image`, but DB-current self-image already has rolling state.
- Add deterministic tests for rollover into `historical_summary`.
- Add deterministic `db_writer` coverage for runtime-state read failure:
  `character_image=false`, no self-image upsert, and no `db_writer` abort.
- Add deterministic or static coverage that self-cognition source projection
  still excludes `self_image`.
- Keep Cache2 invalidation behavior unchanged: a successful character-image
  write still invalidates `character_state`.

## Deferred

- Do not perform production data repair in this plan.
- Do not redesign self-cognition case projection.
- Do not add `self_image` to self-cognition source payloads.
- Do not add compare-and-swap, distributed locks, transactions, or optimistic
  concurrency unless verification proves concurrent self-image writers remain
  after the DB-current merge fix.
- Do not redesign character image memory, `recent_window` limits, compression
  thresholds, or historical summary semantics.
- Do not modify RAG, cognition, dialog, reflection promotion, global character
  growth, or user memory units.
- Do not add new event-log payload fields unless a test proves existing
  sanitized metadata is insufficient for this bugfix.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Character-image merge base | bigbang | Replace profile-derived merge base with DB-current merge base for all character-image writes. |
| Self-cognition case profile | bigbang | Keep current prompt-safe projection; do not add `self_image` as a compatibility surface. |
| Existing production data | migration | No automatic migration. Existing damaged `historical_summary` remains until future valid writes accumulate or an explicit operator restore plan is approved. |
| Tests | bigbang | Update tests to assert the new source-of-truth contract. Do not preserve tests that expect profile-only image updates. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- Bigbang areas must be replaced directly; do not keep alternate old writer
  behavior.
- Any change to cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the contracts in this plan.
- The agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, or extra features.
- Changes outside the listed `Change Surface` require stopping and updating the
  plan or asking for approval.
- If an existing helper already provides the needed DB read, use it instead of
  creating a new public DB API.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.
- Tests may use local fake LLM response objects or monkeypatch existing LLM
  instances; they must not call live LLM endpoints.

## Target State

Every character-image write follows this ownership boundary:

```text
completed consolidation state
  -> db_writer character_image block
  -> read latest character_state.self_image from MongoDB
  -> pass DB-current self_image into image rolling merge
  -> _update_character_image appends one session_summary
  -> upsert_character_self_image writes the merged full self_image
  -> Cache2 invalidates character_state when the write succeeds
```

`state["character_profile"]` remains a prompt/runtime profile. It may contain
`name`, static persona fields, mood, vibe, reflection summary, or no
`self_image`. It must not be treated as the durable self-image source.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Merge base owner | `db_writer` reads DB-current self-image inside the failure-isolated character-image update path immediately before character-image synthesis | Persistence owns durable state. Prompt-facing profiles can be clipped or projected. |
| Failure isolation | Runtime-state read failure is handled as a character-image update failure, not a whole-writer failure | The existing `return_exceptions=True` behavior must not be weakened by moving the DB read outside it. |
| Prompt contract | No prompt changes | The failure is not LLM judgment; it is deterministic state ownership. |
| Self-cognition profile | Keep `self_image` excluded from self-cognition case projection | Adding it would expand prompt context and still leave profile-derived persistence fragile. |
| DB helper | Prefer existing `get_character_runtime_state()` unless implementation proves a narrow helper is necessary | It already projects `self_image` from MongoDB and avoids expanding the DB facade. |
| Runtime cache | Do not rely on service cache refresh for correctness | Self-cognition calls the consolidator directly and may process multiple cases from one stale/profile-projected snapshot. |
| Concurrency | Do not add locks or transactions in this fix | Current self-cognition tick processes cases sequentially; the observed failure is missing merge base, not concurrent writes. |

## Contracts And Data Shapes

`_update_character_image(...)` must use an explicit base image argument:

```python
async def _update_character_image(
    state: ConsolidatorState,
    *,
    storage_timestamp_utc: str,
    existing_image: dict,
) -> dict | None:
    ...
```

Allowed `existing_image` shape:

```python
{
    "milestones": list,
    "recent_window": list[dict],
    "historical_summary": str,
    "meta": {
        "synthesis_count": int,
        "last_updated": str,
    },
}
```

Missing fields are normalized as today:

- `milestones` -> `[]`
- `recent_window` -> `[]`
- `historical_summary` -> `""`
- `meta.synthesis_count` -> `0`

Forbidden contract:

```python
existing_image = state["character_profile"].get("self_image") or {}
```

That pattern must not remain in the image writer.

## LLM Call And Context Budget

No LLM calls are added or removed.

Before:

- Character image session summary: one background `CONSOLIDATION_LLM` call only
  when `reflection_summary` is non-empty.
- Character image compression: optional background `CONSOLIDATION_LLM` call
  only when historical summary exceeds the configured char limit.

After:

- Same call count, same prompts, same payload shape, same response-path impact.
- The only input change is deterministic: the old rolling image is read from
  MongoDB instead of `state["character_profile"]`.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
  - Change `_update_character_image(...)` to accept explicit `existing_image`.
  - Remove durable rolling-state reads from `character_profile.self_image`.
  - Keep `character_profile["name"]` for prompt rendering only.
  - Keep session summary and compression prompt constants unchanged.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
  - Import and call the existing DB runtime-state reader before image update.
  - Extract `runtime_state.get("self_image")` when it is a dict, otherwise `{}`.
  - Pass that value to `_update_character_image(...)`.
  - Keep existing `asyncio.gather(..., return_exceptions=True)` error behavior
    for image update failures. The runtime-state read must occur inside the
    same gathered coroutine or an equivalent local try/except that converts the
    read failure into `write_success.character_image=false`.
  - Do not call the runtime-state reader outside the character-image failure
    boundary in a way that can abort `db_writer`.
  - Keep `upsert_character_self_image(...)` as the only write helper.

- `tests/test_consolidator_origin_policy_db_writer.py`
  - Add regression coverage for an `internal_thought` origin whose
    `character_profile` lacks `self_image`, while DB-current runtime state has
    existing self-image.
  - The new merge regression must exercise the real
    `_update_character_image(...)`; patch only the DB runtime-state reader and
    image LLM responses needed to make the test deterministic.
  - Assert `upsert_character_self_image(...)` receives the merged image.
  - Add runtime-state read failure coverage: reader raises, self-image upsert is
    not called, `write_success.character_image=false`, and `db_writer` returns
    normally.

- `tests/test_consolidator_efficiency.py` or a new focused test file under
  `tests/`
  - Add direct image-rolling tests for `_update_character_image(...)` using a
    fake session-summary LLM.
  - Cover accumulation and rollover into `historical_summary`.

- `tests/test_service_background_consolidation.py`
  - Update only if implementation changes service runtime-cache behavior.
  - Do not add cache-refresh requirements unless needed by the chosen code.

### Keep

- `src/kazusa_ai_chatbot/self_cognition/sources.py`
  - Keep `_CHARACTER_PROFILE_FIELDS` excluding `self_image`.

- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Keep direct shared-consolidator use. Do not special-case
    character-image writes in the runner.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
  - Keep existing origin policy.

- `src/kazusa_ai_chatbot/db/character.py`
  - Keep `upsert_character_self_image(...)` full-document write behavior unless
    focused tests prove a narrower DB helper is required.

### Delete

- No production code deletion is planned.

## Overdesign Guardrail

- Actual problem: internal-thought self-cognition consolidation can overwrite
  `character_state.self_image` from a profile that intentionally lacks
  `self_image`, resetting `recent_window`, `historical_summary`, and
  `synthesis_count`.
- Minimal change: read DB-current `character_state.self_image` at the
  persistence boundary and pass it explicitly into the existing rolling-image
  merge helper.
- Ownership boundaries: deterministic persistence code owns DB-current state,
  merge base, write validation, and Cache2 invalidation; LLMs only generate the
  new session summary and optional compressed historical text; self-cognition
  source projection owns prompt-safe case fields, not durable write state.
- Rejected complexity: no new prompts, no new LLM retries, no self-cognition
  route changes, no extra feature flag, no profile compatibility shim, no
  automatic data repair, no transaction or lock layer.
- Evidence threshold: add optimistic concurrency or a narrower DB update helper
  only if deterministic tests or production logs show concurrent character
  image writers still lose updates after the DB-current merge fix.

## Implementation Order

1. Load mandatory skills and reread this plan.
2. Add direct image-rolling tests for `_update_character_image(...)`.
   - Test file: `tests/test_consolidator_character_image.py` or existing
     nearest test file if it already contains image writer tests.
   - Test 1: stale/profile-missing state plus DB-current existing image yields
     appended `recent_window` and incremented `synthesis_count`.
   - Test 2: six existing recent entries plus one new summary rolls the oldest
     entry into `historical_summary` and keeps recent length at six.
   - Run the new focused tests before implementation and record the expected
     failure.
3. Add `db_writer` regression coverage for internal-thought origin.
   - Use `tests/test_consolidator_origin_policy_db_writer.py`.
   - Patch DB runtime-state reader to return existing `self_image`.
   - Leave `_update_character_image(...)` unpatched in the new merge regression.
   - Patch image LLM responses to return deterministic `session_summary`.
   - Assert `upsert_character_self_image(...)` receives the merged document.
   - Add a second `db_writer` regression where the DB runtime-state reader
     raises; assert no self-image upsert, `write_success.character_image=false`,
     and no exception escapes `db_writer`.
   - Run the new test before implementation and record the expected failure.
4. Implement `_update_character_image(...)` explicit `existing_image` contract.
5. Implement `db_writer` DB-current self-image read and pass-through.
   - The read must run inside the same failure-isolated character-image update
     path as `_update_character_image(...)`.
   - If the read fails, `db_writer` must preserve current image failure behavior:
     log the failure, set `write_success.character_image=false`, skip
     `upsert_character_self_image(...)`, and continue to metadata/cache handling.
6. Run focused image and db-writer tests.
7. Run adjacent consolidation tests and service cache tests listed in
   `Verification`.
8. Run static greps to prove forbidden profile-derived merge behavior is gone.
9. Run independent code review.
10. Record execution evidence and leave the plan status unchanged unless the
    user approves execution completion and lifecycle update.

## Progress Checklist

- [x] Stage 1 - failure tests added
  - Covers: Implementation Order steps 1-3.
  - Verify: focused new tests fail before implementation for the expected
    reason.
  - Evidence: record commands and failure summaries in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-19` after verification and evidence are recorded.

- [x] Stage 2 - DB-current merge implemented
  - Covers: Implementation Order steps 4-5.
  - Verify: focused new tests pass.
  - Evidence: record changed files and focused test output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-19` after verification and evidence are recorded.

- [x] Stage 3 - regression verification complete
  - Covers: Implementation Order steps 6-8.
  - Verify: all commands in `Verification` pass or have approved documented
    blockers.
  - Evidence: record test and grep outputs.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-19` after verification and evidence are recorded.

- [x] Stage 4 - independent code review complete
  - Covers: Implementation Order step 9.
  - Verify: review findings are closed or explicitly approved as residual
    risk; affected tests are rerun after fixes.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, and
    approval status.
  - Handoff: plan can be marked completed only after user-approved execution
    and evidence recording.
  - Sign-off: `Codex/2026-05-19` after review evidence is recorded.

## Verification

### Static Greps

- `rg -n "character_profile\\.get\\([\"']self_image[\"']\\)|state\\[[\"']character_profile[\"']\\].*self_image" src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
  - Expected: no match that uses `state["character_profile"]` or
    `character_profile.get("self_image")` as the merge base.

- `rg -n "_update_character_image\\(" src tests`
  - Expected: all call sites pass the explicit `existing_image=` argument.

- `rg -n "\"self_image\"" src/kazusa_ai_chatbot/self_cognition/sources.py`
  - Expected: no match. Self-cognition source projection must not add
    `self_image` to prompt-facing worker cases.

### Focused Tests

- `venv\Scripts\python -m pytest tests/test_consolidator_character_image.py -q`
  - Expected: pass.
  - If this plan reuses an existing test file instead, run the exact focused
    test names added by this plan.

- `venv\Scripts\python -m pytest tests/test_consolidator_origin_policy_db_writer.py -q`
  - Expected: pass, including the real `_update_character_image(...)` merge
    regression and runtime-state read failure regression.

### Adjacent Regression Tests

- `venv\Scripts\python -m pytest tests/test_consolidator_efficiency.py tests/test_db_writer_cache2_invalidation.py -q`
  - Expected: pass.

- `venv\Scripts\python -m pytest tests/test_service_background_consolidation.py::test_background_consolidation_refreshes_cached_character_state -q`
  - Expected: pass. If implementation does not touch service cache, this test
    should remain unchanged and passing.

### Production Read-Only Diagnostic

Use only if the user explicitly asks for post-fix production inspection:

- Read `character_state` singleton and report only structural fields:
  `self_image.recent_window` length, `self_image.historical_summary` length,
  `self_image.meta.synthesis_count`, and `self_image.meta.last_updated`.
- Do not mutate MongoDB.
- Do not dump raw summary text into chat unless the user asks.

## Data Migration

No automatic data migration is included.

The current production `historical_summary` appears already overwritten. This
plan prevents further loss. It does not reconstruct lost history because the
available event logs are sanitized and do not contain previous self-image
summary text. Any restore from backup or manual reconstruction requires a
separate operator-approved repair plan.

## Operational Steps

- Before deploying the code fix, operators may temporarily disable
  self-cognition if they need to stop further overwrites immediately.
- After deploying the code fix, re-enable self-cognition only after focused
  regression tests pass.
- After at least two self-cognition or chat consolidations, perform the
  read-only structural diagnostic if the user requests it. Expected structural
  trend: `synthesis_count` increases instead of staying at `1`, and
  `recent_window` grows instead of being replaced with one item.

## Independent Plan Review

Review date: 2026-05-19.

Approval status: approved for execution.

Findings resolved before approval:

- DB runtime-state read failure is now explicitly inside the character-image
  failure boundary and has required regression coverage.
- The `db_writer` merge regression must exercise the real
  `_update_character_image(...)`; it may patch only the DB runtime-state reader
  and deterministic image LLM responses.
- Verification now proves self-cognition source projection still excludes
  `self_image`.
- The plan class is `large`, matching the current line count and risk profile.
- RCA timestamp evidence now names the compared fields:
  `event_log_events.occurred_at` and
  `character_state.self_image.meta.last_updated`.

No open plan-review blockers remain.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including focused tests, static checks,
  execution evidence, and path-safe commands.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only fixture/documentation
corrections. If a fix would cross the approved boundary or alter the contract,
stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `_update_character_image(...)` no longer reads durable rolling state from
  `state["character_profile"].self_image`.
- `db_writer` uses DB-current `character_state.self_image` as the merge base
  for character-image writes.
- Internal-thought consolidation with no `character_profile.self_image`
  preserves existing DB-current `recent_window`, `historical_summary`, and
  `synthesis_count`.
- Rollover from six recent entries into `historical_summary` is covered by a
  deterministic test.
- Existing Cache2 invalidation behavior for `character_state` remains covered
  and passing.
- No self-cognition source/profile projection is expanded with `self_image`.
- All `Verification` gates pass or have documented, user-approved blockers.
- Independent Code Review is complete and approved.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Existing tests patch `_update_character_image` and miss the real merge behavior | Add direct image writer tests plus db_writer integration test without patching `_update_character_image` | Focused tests in `Verification` |
| DB read failure escapes `db_writer` | Read DB-current self-image inside the character-image failure boundary; assert reader failure returns `character_image=false` and skips self-image upsert | db_writer read-failure regression |
| Prompt payload expands accidentally | Keep prompts unchanged and forbid `self_image` in self-cognition source projection | Static diff review and independent code review |
| Lost production history is mistaken for fixed behavior | State that migration/backfill is out of scope; verify future accumulation structurally | Data Migration and Operational Steps |

## Execution Evidence

- RCA evidence:
  - `kazusa_bot_core.character_state`: `historical_summary_len=0`,
    `recent_window_len=1`, `synthesis_count=1`,
    `self_image_last_updated=2026-05-18T12:58:10.447906+00:00`.
  - `kazusa_bot_core.conversation_history`: `50,492` rows.
  - `kazusa_bot_core.event_log_events`: `47` internal-thought
     `group_chat_trigger_review` records with `character_image=true`.
  - Three internal-thought `character_image=true` event records had
    `event_log_events.occurred_at` later than
    `character_state.self_image.meta.last_updated`.
- Independent plan review result:
  - 2026-05-19: review findings resolved in-plan: DB runtime-state read failure
    isolation, real-helper db_writer merge regression, self-cognition projection
    verification, plan class alignment, and timestamp-field precision.
  - Approval status: approved for execution.
- Pre-implementation focused test failures:
  - `venv\Scripts\python -m pytest tests/test_consolidator_character_image.py -q`
    failed as expected: 2 tests failed because `_update_character_image(...)`
    does not accept `existing_image=`.
  - `venv\Scripts\python -m pytest tests/test_consolidator_origin_policy_db_writer.py -q`
    failed as expected: 4 passed and 2 new regressions failed because
    `db_writer` never awaited `get_character_runtime_state`.
- Post-implementation focused test results:
  - Changed files for implementation: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`,
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`,
    `tests/test_consolidator_character_image.py`, and
    `tests/test_consolidator_origin_policy_db_writer.py`.
  - `venv\Scripts\python -m py_compile src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
    passed.
  - `venv\Scripts\python -m pytest tests/test_consolidator_character_image.py -q`
    passed: 2 passed.
  - `venv\Scripts\python -m pytest tests/test_consolidator_origin_policy_db_writer.py -q`
    passed: 6 passed.
- Static grep results:
  - `rg -n 'character_profile.*self_image|state.*character_profile.*self_image' src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
    returned no matches.
  - `rg -n -C 4 "_update_character_image\\(" src tests` found the function
    definition plus three call sites; every call site passes `existing_image=`.
  - `rg -n '"self_image"' src/kazusa_ai_chatbot/self_cognition/sources.py`
    returned no matches.
  - `git diff --check` passed.
- Adjacent regression test results:
  - `venv\Scripts\python -m pytest tests/test_consolidator_efficiency.py tests/test_db_writer_cache2_invalidation.py -q`
    passed: 5 passed.
  - `venv\Scripts\python -m pytest tests/test_service_background_consolidation.py::test_background_consolidation_refreshes_cached_character_state -q`
    passed: 1 passed.
  - `venv\Scripts\python -m py_compile src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py tests/test_consolidator_character_image.py tests/test_consolidator_origin_policy_db_writer.py tests/test_consolidator_efficiency.py tests/test_db_writer_cache2_invalidation.py`
    passed.
- Independent code review result:
  - Reviewer mode: active agent self-review; no separate reviewer was available.
  - Findings: adjacent isolation tests initially allowed real
    `get_character_runtime_state()` DB reads, and two newly added monkeypatch
    lines plus an import order needed style cleanup.
  - Fixes: patched `get_character_runtime_state` in
    `tests/test_consolidator_efficiency.py` and
    `tests/test_db_writer_cache2_invalidation.py`; reordered
    `DatabaseOperationError` import and wrapped new monkeypatch lines.
  - Rerun commands:
    - `venv\Scripts\python -m py_compile tests/test_consolidator_origin_policy_db_writer.py tests/test_consolidator_efficiency.py tests/test_db_writer_cache2_invalidation.py`
      passed.
    - `venv\Scripts\python -m pytest tests/test_consolidator_origin_policy_db_writer.py tests/test_consolidator_efficiency.py tests/test_db_writer_cache2_invalidation.py -q`
      passed: 11 passed.
    - `git diff --check` passed.
  - Residual risk: existing production `historical_summary` is not repaired by
    this bugfix; future writes will accumulate from the current DB state.
  - Approval status: approved.

## Plan Self-Review

- Coverage: every `Must Do` item maps to implementation steps and verification
  gates.
- Minimality: the plan changes only the durable merge-base ownership and tests.
- Placeholder scan: no unresolved design choice is left for implementation.
- Contract consistency: `_update_character_image(...)` has one explicit
  `existing_image` contract and all callers must pass it.
- Verification: focused tests cover the production failure mode, runtime-state
  read failure isolation, and rollover.
- Closure: completed and archived on 2026-05-19 during registry cleanup after
  all checklist stages and review evidence were already recorded.
