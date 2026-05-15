# reflection global promotion replay bugfix plan

## Summary

- Goal: stop daily global reflection promotion from replaying successful memory
  writes on every worker tick, and make duplicate replacement-id collisions
  degrade to an inspectable skipped result instead of crashing the reflection
  worker.
- Plan class: medium
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, `superpowers:systematic-debugging`,
  `superpowers:test-driven-development`, `superpowers:verification-before-completion`
- Overall cutover strategy: bigbang for worker/global-promotion retry semantics;
  compatible for persisted `character_reflection_runs` document shape.
- Highest-risk areas: treating an LLM-only persisted run as a completed
  memory-write run, hiding real memory lineage bugs as harmless duplicates,
  preventing legitimate retry after skipped or failed promotion work, and
  expanding the memory-evolution API beyond the reflection boundary.
- Acceptance criteria: a successful daily global promotion for one
  `character_local_date` and prompt version runs at most once; skipped,
  deferred, dry-run, and failed promotion runs remain retryable when appropriate;
  known duplicate replacement-id write collisions no longer crash the worker;
  no prompt, adapter, cognition, RAG, or memory-evolution contract is broadened.

## Context

Production logged this worker failure on 2026-05-15:

```text
ValueError: replacement memory_unit_id already exists
```

The stack reached:

```text
reflection_cycle.worker._run_worker_tick
-> reflection_cycle.promotion._run_global_reflection_promotion
-> _write_validated_promotion_decisions
-> _resolve_similarity_and_write
-> _write_memory_doc
-> memory_evolution.repository.supersede_memory_unit
```

A prior commit, `4a8d8c7 Fix reflection promotion replay supersede`, added a
narrow guard in `src/kazusa_ai_chatbot/reflection_cycle/promotion.py` that skips
one replay case: the similarity search returns the already-active replacement as
the exact supersede source. That fix is valid but incomplete.

Deep-dive verification found a broader structural replay path:

- `src/kazusa_ai_chatbot/reflection_cycle/worker.py` calls global promotion on
  every worker tick after `REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME`.
- Hourly and daily-channel reflection compute candidate run ids and call
  `repository.existing_run_ids(...)` before doing work.
- Daily global promotion has a deterministic run id via
  `repository.daily_global_promotion_run_id(...)`, but neither the worker nor
  `_run_global_reflection_promotion(...)` checks an existing successful global
  run before the LLM/write path.
- `_memory_document_for_decision(...)` builds a deterministic replacement
  `memory_unit_id` from lane, date, memory name, content, and evidence refs.
  Replaying the same decision rebuilds the same replacement id even when
  similarity selects a different active supersede source.
- `memory_evolution.repository.supersede_memory_unit(...)` correctly rejects a
  replacement id that already exists, and the current promotion write path lets
  that `ValueError` escape the worker tick.

This plan fixes the scheduler/promotion ownership issue first. Memory evolution
continues to own lineage validation and must not silently accept duplicate
replacement ids.

## Mandatory Skills

Load these skills before execution:

- `development-plan-writing`: before changing this plan, registry rows, status,
  progress checklist, or execution evidence.
- `local-llm-architecture`: before changing reflection promotion, memory
  promotion boundaries, background LLM call behavior, or prompt-facing context.
- `py-style`: before editing Python files.
- `test-style-and-execution`: before adding, changing, or running tests.
- `superpowers:systematic-debugging`: before changing behavior if new evidence
  contradicts the root cause in this plan.
- `superpowers:test-driven-development`: before implementation. Add focused
  failing deterministic tests first.
- `superpowers:verification-before-completion`: before claiming the bugfix is
  complete or passing.

## Mandatory Rules

- Do not execute this plan while `Status` is `draft`. Execution requires a
  status change to `approved` or `in_progress`.
- Use `venv\Scripts\python` for Python commands.
- Do not read `.env` unless the user explicitly asks for environment inspection.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run this plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- Reflection promotion stays outside the live response path. Do not add live
  chat blocking, adapter behavior, cognition prompt changes, RAG retrieval
  changes, or dialog behavior changes.
- LLM stages own semantic promotion decisions. Deterministic code owns
  scheduling idempotency, retry gates, memory-write validation, persistence,
  and worker failure handling.
- Do not change `memory_evolution.repository.supersede_memory_unit` or
  `merge_memory_units` to silently accept duplicate replacement ids.
- Do not add a `force`, `rerun`, compatibility mode, new config flag, retry
  loop, extra LLM call, or alternate promotion path under this plan.
- Do not treat skipped, deferred, dry-run, or failed global promotion runs as
  successful completed work.
- If an existing persisted `succeeded` daily global promotion row predates this
  bugfix, treat it as already processed for idempotency. Do not add a data
  migration or automatic backfill in this plan.

## Must Do

- Add deterministic tests that prove successful daily global promotion is
  idempotent by `character_local_date` plus
  `GLOBAL_PROMOTION_PROMPT_VERSION`.
- Add deterministic tests that prove skipped, failed, and dry-run global
  promotion rows do not block a later apply run.
- Add a deterministic test that reproduces the remaining duplicate replacement
  collision when the replacement id already exists but the selected supersede
  source is a different active memory unit.
- Compute the daily global promotion run id before LLM execution and memory
  writes.
- Add an early existing-run gate in
  `src/kazusa_ai_chatbot/reflection_cycle/promotion.py` so an existing
  `status="succeeded"` global promotion run returns a skipped
  `ReflectionPromotionResult` without calling the LLM, similarity search, or
  memory mutation APIs.
- Preserve retry for existing `status="skipped"`, `status="failed"`, and
  `status="dry_run"` global promotion rows.
- Rework global promotion persistence so `status="succeeded"` is persisted only
  after the write phase has completed without an unhandled write exception, or
  after the run completed normally with no valid memory mutations to apply.
- Persist `status="skipped"` for `enable_memory_writes=False`, deferred busy
  probes, malformed score rows, memory write-lock contention, and known
  duplicate replacement-id skips.
- Persist `status="failed"` and return a failed result for unexpected write
  exceptions instead of letting the worker tick crash.
- Convert the known duplicate replacement-id `ValueError` from
  `_write_memory_doc(...)` into a skipped promotion warning in the promotion
  layer. The repository must still raise; promotion decides that this known
  replay collision is non-fatal.
- Keep all changes inside reflection-cycle promotion/worker tests unless a
  focused schema or documentation update is required by the changed contract.
- Update reflection-cycle documentation only if the once-per-day success gate
  or retry semantics are not already stated clearly.
- Update `development_plans/README.md` registry status only as part of normal
  lifecycle maintenance.

## Deferred

- Do not migrate or rewrite existing `character_reflection_runs` rows.
- Do not add a manual force-rerun feature.
- Do not change reflection prompt text, LLM route selection, prompt budgets, or
  promotion decision schema.
- Do not redesign similarity thresholds, merge-vs-supersede selection, memory
  lineage identity, or deterministic memory id generation.
- Do not change `memory_evolution` public APIs, DB indexes, embedding behavior,
  or Cache2 invalidation.
- Do not add a new scheduler, global lock, worker queue, or cross-process
  election mechanism.
- Do not change global character growth behavior except through existing
  promotion result counts naturally produced by this bugfix.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Global promotion success replay | bigbang | Replace repeat-on-every-tick behavior with an early skip when the deterministic daily global run id already has `status="succeeded"`. |
| Retryable prior runs | compatible | Preserve retry for existing `skipped`, `failed`, and `dry_run` rows. |
| Reflection run document shape | compatible | Use existing fields unless a narrow optional field is required. Do not require migration. |
| Memory-evolution duplicate validation | bigbang | Keep repository duplicate rejection. Convert the known replay collision to a skipped promotion result in reflection promotion only. |
| Manual CLI promotion | bigbang | The public facade follows the same idempotency contract as the worker. No force-rerun flag is added. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- If an area is `bigbang`, replace the old behavior directly without preserving
  a fallback path.
- If an area is `compatible`, preserve only the compatibility surface listed in
  the table above.
- Any force-rerun option, migration, compatibility shim, or broader memory API
  change requires a separate approved plan or explicit plan revision.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the contracts in this plan.
- The agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, feature flags, or extra
  promotion modes.
- Changes outside `reflection_cycle.promotion`, focused reflection-cycle tests,
  and directly relevant documentation require explicit justification in
  `Execution Evidence` before implementation.
- If equivalent existing behavior already exists, reuse or adapt it instead of
  duplicating it.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

After execution:

- A daily global promotion with the same `character_local_date` and prompt
  version can be requested repeatedly, but only the first successful run can
  perform LLM and memory-write work.
- Later worker ticks return a skipped promotion result that includes the
  existing run id and a clear defer reason such as
  `daily global promotion already succeeded`.
- Failed, skipped, and dry-run runs remain retryable.
- Known duplicate replacement-id replay collisions are recorded as skipped
  promotion warnings and do not crash the worker.
- Unexpected memory write failures are recorded as failed promotion results and
  failed run documents.
- Memory-evolution repository invariants remain strict.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Idempotency owner | Put the success-replay gate in `reflection_cycle.promotion`, not only in the worker. | The public CLI and worker facade share the same daily global promotion contract. |
| Skip condition | Skip only existing `status="succeeded"` rows for the deterministic global run id. | Skipped, failed, and dry-run rows must remain retryable. |
| Write status timing | Persist final `succeeded` only after write handling is complete. | A prompt-only persisted row must not masquerade as a completed memory-write run during new execution. |
| Duplicate replacement collision | Convert only the known duplicate replacement-id `ValueError` into a skipped promotion warning. | The repository should keep rejecting invalid lineage writes; promotion owns replay tolerance. |
| Force rerun | Do not add it. | It adds operational modes not needed to fix the current production failure. |
| Existing historical rows | Treat old `succeeded` rows as processed. | No migration evidence justifies rewriting historical reflection audit records. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
  - Add deterministic existing-success gate for daily global promotion.
  - Reorder or refactor global run persistence so write outcomes determine the
    final persisted status.
  - Catch known duplicate replacement-id write collisions and convert them to a
    skipped warning.
  - Keep prompt construction, LLM invocation, validation schema, similarity
    thresholds, and memory document identity unchanged.
- `tests/test_reflection_cycle_stage1c_promotion.py`
  - Add focused deterministic tests for the idempotency gate, retryable prior
    statuses, write-phase status persistence, and duplicate replacement-id
    collision handling.
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
  - Update only if the final code changes the documented worker or memory
    boundary contract.

### Keep

- `src/kazusa_ai_chatbot/memory_evolution/repository.py`
  - Keep duplicate replacement-id rejection unchanged.
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`
  - Keep worker tick ordering unchanged unless a narrow test reveals the
    promotion-level gate cannot satisfy the worker contract.
- `src/scripts/run_reflection_cycle.py`
  - Keep CLI arguments unchanged. The public facade now owns idempotency.

### Create

- No new production module is planned.

### Delete

- No deletion is planned.

## Overdesign Guardrail

- Actual problem: daily global reflection promotion can replay successful
  memory writes on later worker ticks and can crash on duplicate replacement
  ids.
- Minimal change: add a deterministic successful-run gate in global promotion,
  persist final status after write handling, and turn the known duplicate
  replacement-id replay collision into a skipped promotion result.
- Ownership boundaries: LLM promotion decides semantic candidates; reflection
  promotion owns daily idempotency and write-result classification; memory
  evolution owns lineage validation, embeddings, insertion, supersession, merge,
  and Cache2 invalidation; the worker owns tick order only.
- Rejected complexity: force-rerun flags, compatibility shims, new DB
  collections, migrations, new locks, retry loops, prompt changes, broader
  memory APIs, similarity threshold tuning, and adapter or live-chat changes.
- Evidence threshold: add rejected complexity only after a separate observed
  production need, such as an operator requirement to force-rerun a succeeded
  global promotion date or a verified cross-process race that cannot be handled
  by existing deterministic run ids and memory write guards.

## LLM Call And Context Budget

- Before: after the promotion time, every worker tick can spend one background
  `CONSOLIDATION_LLM_*` call for the same `character_local_date` and prompt
  version, then can repeat similarity and memory-write work.
- After: once a daily global promotion run has `status="succeeded"`, later
  calls spend zero LLM calls, zero similarity searches, and zero memory
  mutation calls for that run id.
- Response path impact: none. Reflection promotion remains background work.
- Prompt and context changes: none.
- Estimated context budget: unchanged for the first eligible run; repeated
  successful replays are eliminated.

## Data Migration

No data migration is authorized.

Existing `character_reflection_runs` rows with `run_kind="daily_global_promotion"`
and `status="succeeded"` are treated as already processed for the same
`character_local_date` and prompt version. If an operator later discovers a
specific historical row was a partial write, that remediation requires a
separate manual operation or follow-up plan.

## Implementation Order

1. Add focused failing tests in
   `tests/test_reflection_cycle_stage1c_promotion.py`.
   - `test_global_promotion_skips_existing_succeeded_run_without_llm_or_writes`
   - `test_global_promotion_retries_existing_skipped_failed_and_dry_run_rows`
   - `test_global_promotion_persists_skipped_when_memory_writes_disabled`
   - `test_global_promotion_records_failed_write_phase_without_worker_crash`
   - `test_promotion_skips_duplicate_replacement_id_from_different_source`
   - Expected before implementation: at least the existing-succeeded and
     duplicate-from-different-source cases fail on current code.
2. Implement the existing-success gate in
   `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`.
   - Compute the run id using `repository.daily_global_promotion_run_id(...)`.
   - Load the existing run with `repository.reflection_run_by_id(...)`.
   - If the existing run has `status="succeeded"`, return a skipped
     `ReflectionPromotionResult` with `processed_count=1`, `skipped_count=1`,
     the existing run id, and no LLM or memory calls.
3. Refactor global promotion persistence in `promotion.py`.
   - Do not persist `status="succeeded"` before the write phase.
   - Persist `dry_run`, `skipped`, `failed`, or `succeeded` according to the
     actual terminal outcome.
   - Preserve existing prompt output and promotion decision storage.
4. Add write-phase failure classification.
   - Known duplicate replacement-id `ValueError` becomes skipped with a warning.
   - Memory write-lock contention remains deferred.
   - Unexpected write exceptions persist a failed run and return a failed
     result.
5. Run focused tests and adjust only within the approved change surface.
6. Inspect `src/kazusa_ai_chatbot/reflection_cycle/README.md`.
   - Update it only if the current text conflicts with the implemented
     once-per-success idempotency and retry semantics.
7. Run regression tests listed in `Verification`.
8. Run the independent code review gate and remediate approved findings.

## Progress Checklist

- [x] Stage 1 - failing promotion replay tests added
  - Covers: implementation order step 1.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_skips_existing_succeeded_run_without_llm_or_writes tests\test_reflection_cycle_stage1c_promotion.py::test_promotion_skips_duplicate_replacement_id_from_different_source -q`
  - Evidence: record expected pre-implementation failures in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-15`.
- [x] Stage 2 - idempotency gate implemented
  - Covers: implementation order step 2.
  - Verify: focused existing-succeeded and retryable-status tests pass.
  - Evidence: record changed files and focused test output.
  - Sign-off: `Codex/2026-05-15`.
- [x] Stage 3 - write-phase persistence and collision handling implemented
  - Covers: implementation order steps 3 and 4.
  - Verify: focused write-disabled, write-failed, and duplicate-collision tests
    pass.
  - Evidence: record changed files and focused test output.
  - Sign-off: `Codex/2026-05-15`.
- [x] Stage 4 - documentation inspection and regression verification complete
  - Covers: implementation order steps 5 through 7.
  - Verify: all commands in `Verification` pass or documented allowed
    exceptions are recorded.
  - Evidence: record README decision and command output summaries.
  - Sign-off: `Codex/2026-05-15`.
- [x] Stage 5 - independent code review complete
  - Covers: implementation order step 8 and `Independent Code Review`.
  - Verify: review findings are recorded; affected tests are rerun after fixes.
  - Evidence: record reviewer mode, findings, fixes, reruns, and residual risks.
  - Sign-off: `Codex/2026-05-15`.

## Verification

### Focused Deterministic Tests

- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_skips_existing_succeeded_run_without_llm_or_writes -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_retries_existing_skipped_failed_and_dry_run_rows -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_persists_skipped_when_memory_writes_disabled -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_records_failed_write_phase_without_worker_crash -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_promotion.py::test_promotion_skips_duplicate_replacement_id_from_different_source -q`

### Regression Tests

- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_promotion.py -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_worker.py -q`
- `venv\Scripts\python -m pytest tests\test_global_character_growth_worker.py -q`

### Static Greps

- `rg -n "replacement memory_unit_id already exists" src tests`
  - Expected: matches may remain in
    `src/kazusa_ai_chatbot/memory_evolution/repository.py` and focused
    promotion tests. Any production match outside memory-evolution repository
    and reflection promotion must be justified in `Execution Evidence`.
- `rg -n "daily_global_promotion_run_id|existing_run_ids|reflection_run_by_id" src/kazusa_ai_chatbot/reflection_cycle tests/test_reflection_cycle_stage1c_promotion.py tests/test_reflection_cycle_stage1c_worker.py`
  - Expected: global promotion has an explicit existing-run status check; worker
    hourly/daily existing-run checks remain intact.

### Syntax

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\promotion.py`

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, and
  documentation artifact.
- Plan alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`,
  `Change Surface`, implementation order, verification gates, and acceptance
  criteria.
- Scheduler and persistence correctness: successful runs skip, retryable runs
  retry, write-disabled runs do not become successful, and unexpected write
  failures do not crash the worker.
- Memory boundary correctness: repository duplicate validation remains strict,
  and promotion handles only the known replay collision as skipped.
- Regression quality: tests cover the verified production failure path and do
  not rely on live LLMs or a live database.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only documentation or test
corrections. If a fix would cross the approved boundary or alter the contract,
stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Existing successful daily global promotion rows skip future LLM, similarity,
  and memory mutation work for the same date and prompt version.
- Existing skipped, failed, and dry-run rows do not block retry.
- `enable_memory_writes=False` no longer persists an apply-mode success row.
- Known duplicate replacement-id replay collisions return skipped warnings
  instead of crashing the worker.
- Unexpected memory write exceptions produce failed promotion results and failed
  run documents.
- The listed focused and regression tests pass.
- The independent code review gate is complete and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| A historical `succeeded` row was actually a partial memory write. | No migration is attempted; historical succeeded rows are treated as processed and exceptional repair requires a separate operation. | Documented in `Data Migration`; no automatic rewrite code added. |
| The idempotency gate blocks legitimate retries. | Skip only `status="succeeded"`; keep skipped, failed, and dry-run retryable. | Retryable-status test. |
| Duplicate replacement handling hides a real memory bug. | Convert only the exact known duplicate replacement-id `ValueError`; leave other repository errors failed. | Duplicate-collision and unexpected-write-failure tests. |
| Persistence refactor changes prompt output storage. | Preserve `output` and `promotion_decisions` fields for every terminal run. | Focused tests inspect persisted documents. |
| Global character growth stops after a skipped replay. | This is intended after a prior successful promotion; growth should only run when the current promotion created memory mutations. | `tests/test_global_character_growth_worker.py`. |

## Execution Evidence

- Draft created: 2026-05-15.
- Implementation: started 2026-05-15 after user requested execution.
- Stage 1 RED verification:
  `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_skips_existing_succeeded_run_without_llm_or_writes tests\test_reflection_cycle_stage1c_promotion.py::test_promotion_skips_duplicate_replacement_id_from_different_source -q`
  exited 1 before implementation. The existing-success case replayed promotion
  work instead of returning `daily global promotion already succeeded`; the
  duplicate-from-different-source case raised
  `ValueError: replacement memory_unit_id already exists`.
- Implementation files changed:
  `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`,
  `tests/test_reflection_cycle_stage1c_promotion.py`, and
  `src/kazusa_ai_chatbot/reflection_cycle/README.md`.
- Syntax verification:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\promotion.py`
  passed.
- Focused verification:
  `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_skips_existing_succeeded_run_without_llm_or_writes tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_retries_existing_skipped_failed_and_dry_run_rows tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_persists_skipped_when_memory_writes_disabled tests\test_reflection_cycle_stage1c_promotion.py::test_global_promotion_records_failed_write_phase_without_worker_crash tests\test_reflection_cycle_stage1c_promotion.py::test_promotion_skips_duplicate_replacement_id_from_different_source -q`
  passed with 7 tests including parametrized retry statuses.
- Regression verification:
  `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_promotion.py tests\test_reflection_cycle_stage1c_worker.py tests\test_global_character_growth_worker.py -q`
  passed with 27 tests.
- Static grep verification:
  `rg -n "replacement memory_unit_id already exists" src tests` showed the
  strict repository raises remain in
  `src/kazusa_ai_chatbot/memory_evolution/repository.py`, the promotion-layer
  constant, and focused tests only.
- Static grep verification:
  `rg -n "daily_global_promotion_run_id|existing_run_ids|reflection_run_by_id" src/kazusa_ai_chatbot/reflection_cycle tests/test_reflection_cycle_stage1c_promotion.py tests/test_reflection_cycle_stage1c_worker.py`
  showed the new global promotion existing-run gate and the pre-existing
  hourly/daily worker idempotency checks.
- Documentation: `src/kazusa_ai_chatbot/reflection_cycle/README.md` was updated
  to state that deterministic daily global promotion run ids are the
  idempotency key and that skipped, failed, and dry-run rows remain retryable.
- Independent code review: separate reviewer `Hilbert` reviewed the
  uncommitted diff after verification. No Critical issues were found. One
  Important commit-hygiene issue was reported: unrelated
  `development_plans/README.md` registry rows for pre-existing active/reference
  plan files must not be included in this bugfix commit. Fix: stage only the
  completed reflection bugfix registry row from `development_plans/README.md`
  and leave unrelated development-plan edits uncommitted. Residual risk remains
  the documented historical-row policy: pre-existing `succeeded` rows are
  treated as processed.

## Plan Self-Review

- Coverage: every `Must Do` item maps to implementation order, checklist, or
  verification.
- Minimality: the plan fixes replay idempotency and known write collision
  handling without prompt changes, memory API changes, migration, force rerun,
  or new scheduler architecture.
- Placeholder scan: no placeholder decision points remain.
- Contract consistency: target files, tests, statuses, and run-id ownership are
  named consistently.
- Verification: focused deterministic tests cover the verified production
  failure path and retry boundaries.
