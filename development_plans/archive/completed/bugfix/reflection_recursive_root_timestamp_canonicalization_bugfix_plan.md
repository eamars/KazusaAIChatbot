# reflection recursive-root timestamp canonicalization bugfix plan

## Summary

- Status: completed.
- Parent plan: [`self_cognition_trigger_state_contract_recovery_bugfix_plan.md`](self_cognition_trigger_state_contract_recovery_bugfix_plan.md).
- Goal: unblock the reflection-selected group self-cognition caller when one
  settled episode legitimately spans multiple stored conversation messages.
- Scope: deterministic recursive reflection-root normalization, its ICD, focused
  regression coverage, and a write-capable guarded test-database observation.

## Evidence And Root Cause

The read-only production export
`test_artifacts/diagnostics/character_reflection_runs_blocker_review.json`
showed repeated `source_episode_refs` with the same
`root_episode_id`, `correlation_id`, `character_local_date`, and `scope_kind`
but different `captured_at` values. The service attaches one settled episode
root to every message in that episode, so message timestamps are expected to
vary. `_normalize_source_episode_refs` compared the complete row and raised
`recursive reflection root metadata is inconsistent` for this valid case,
preventing the reflection worker from reaching group self-cognition.

GPT-5.6 Sol (`high`, normal service speed) reviewed the parent proposal and
confirmed that the repository normalization boundary owns this correction.
The earliest timestamp is required because downstream freshness and reversal
policies use `captured_at`; retaining the latest timestamp could make an old
root appear newly eligible.

## Target Contract

For each recursive source row:

1. Validate the exact shape, required text, supported scope, local date, and
   canonical UTC timestamp.
2. Deduplicate by `root_episode_id`.
3. Require identical `correlation_id`, `character_local_date`, and `scope_kind`
   for repeated roots; fail closed on any conflict.
4. Retain the earliest normalized `captured_at` for timestamp-only duplicates.
5. Return roots sorted by canonical `captured_at`, then root ID.

Distinct root IDs remain distinct rows. Historical reflection documents remain
untouched; only newly built derivative documents use the corrected projection.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/reflection_cycle/repository.py`
  - Replace complete-row equality with stable-metadata validation plus earliest
    timestamp canonicalization.
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
  - Document recursive root provenance and fail-closed conflict handling.
- `tests/test_reflection_cycle_stage1c_repository.py`
  - Cover order-independent earliest-timestamp deduplication, distinct roots,
    stable-metadata conflict rejection, and daily/global derivative propagation.

### Preserve

- Reflection cadence, LLM prompts, promotion policy, identity-growth policy,
  self-cognition source contracts, action authorization, and delivery routing.
- Historical rows and production data. The guarded run may write only to the
  exact `_test_kazusa_core_v2` database selected by the Stage 3 harness.

## Test Impact And Traceability

| Source or governed artifact | Exact deterministic pytest node(s) | Purpose |
| --- | --- | --- |
| `repository.py` | `tests/test_reflection_cycle_stage1c_repository.py::test_recursive_episode_root_union_uses_earliest_captured_at` | Timestamp-only duplicates are order-independent and retain the earliest instant. |
| `repository.py` | `tests/test_reflection_cycle_stage1c_repository.py::test_recursive_episode_root_union_rejects_conflicting_identity_metadata` | Correlation, local-date, and scope conflicts remain fail-closed. |
| `repository.py` | `tests/test_reflection_cycle_stage1c_repository.py::test_recursive_episode_root_union_preserves_distinct_root_ids` | Different settled roots are not collapsed. |
| `repository.py` | `tests/test_reflection_cycle_stage1c_repository.py::test_daily_and_global_runs_recursively_union_episode_roots` | Daily and global derivatives carry one canonical root after repeated hourly evidence. |

Downstream regression nodes:

- `tests/test_character_identity_growth_causal_lineage.py::test_same_episode_and_derived_reflection_count_once`
- `tests/test_character_identity_growth_causal_lineage.py::test_conflicting_repository_metadata_for_one_root_fails_closed`
- `tests/test_character_identity_growth_policy.py::test_inferred_reversal_requires_fresh_post_revision_threshold`

Guarded live node:

- `tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_group_review_worker_records_reviewed_ledger`

## Guarded Acceptance

Run the retained promoted-reflection group caller directly under the Stage 3
database guard. The generic `tests.stage3_fresh_database run-case` harness is
reserved for frozen user-message cases and does not invoke this self-cognition
branch.

```powershell
$env:MONGODB_DB_NAME='_test_kazusa_core_v2'
$env:STAGE3_DATABASE_GUARD='1'
venv\Scripts\python -m pytest `
  tests\test_stage3_fresh_database_e2e_live_llm.py::test_live_group_review_worker_records_reviewed_ledger `
  -q -s -o addopts=
```

The acceptance evidence must show:

- the exact guarded database is `_test_kazusa_core_v2`;
- hourly reflection succeeds;
- daily-channel reflection succeeds without the recursive-root error;
- the derivative contains one canonical earliest root;
- the selected group case reaches one fresh `reviewed` terminal ledger row
  (older windows may also be recorded as `coalesced_skipped`);
- any action, dispatcher, adapter, or model-quality disposition is reported
  separately from state/ledger progression.

The command is write-capable only against the harness-reserved test database.
It does not authorize production writes or historical repair. If the guarded
environment is unavailable, retain the deterministic fix and leave the parent
plan active with the exact environment disposition.

## Execution Checklist

- [x] Root cause confirmed by read-only production export.
- [x] Parent proposal formed before architectural guidance.
- [x] GPT-5.6 Sol architectural review completed.
- [x] Repository, ICD, and focused regression changes implemented.
- [x] Exact owner and downstream deterministic nodes pass.
- [x] Guarded reflection-to-self-cognition group case produces fresh evidence.
- [x] Parent plan consumes the evidence and closes its combined acceptance gate.

## Execution Evidence

Final deterministic evidence:

- The focused repository, reflection worker, promotion, identity-lineage, and
  identity-policy suites passed: `108 passed`.
- The parent self-cognition owner list passed: `23 passed in 1.07s`.
- Static compilation and `git diff --check` passed.
- The guarded node
  `tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_group_review_worker_records_reviewed_ledger`
  passed in `71.69s` against `_test_kazusa_core_v2`.

Guarded database evidence is recorded in
`test_artifacts/cognition_core_v2/stage_3/focused_live/focused_group_review_worker_ledger.json`:
two hourly runs shared one settled root, the daily derivative retained one
earliest `captured_at`, the self-cognition worker processed one case with zero
failures, and the ledger contained a fresh `reviewed` row plus the expected
older `coalesced_skipped` row. Independent exports are recorded in
`test_artifacts/diagnostics/reflection_group_worker_postfix_guarded.json` and
`test_artifacts/diagnostics/self_cognition_group_review_postfix_guarded_reviewed.json`.

No production database mutation or historical row rewrite is part of this
plan. The plan is complete and is ready for archival with its parent plan.
