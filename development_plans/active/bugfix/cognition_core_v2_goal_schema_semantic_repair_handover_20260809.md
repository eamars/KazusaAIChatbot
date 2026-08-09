# Cognition Core V2 goal-schema / semantic-repair handover

Date: 2026-08-09

Status: execution completed after the user's explicit handover. This document
records the continuation contract and final execution outcome.

## Completed plan

[`cognition_core_v2_goal_schema_and_semantic_repair_compaction_bugfix_plan.md`](../../archive/completed/bugfix/cognition_core_v2_goal_schema_and_semantic_repair_compaction_bugfix_plan.md)

The plan was amended after the requested read-only `gpt-5.6-sol` high-reasoning
review. `development_plans/README.md` now registers it as `completed`.

## Reviewer conclusion

The reviewer confirmed both production directions address the demonstrated root
causes:

- Workstream A isolates the ordinary-only `relational_willingness` schema from
  non-ordinary generic goal prompts while retaining strict validation.
- Workstream B removes only the duplicated validator-owned `; permitted paths:`
  suffix from model-facing semantic repair feedback while retaining the failed
  rule, offending path, `allowed_values`, full protected error, and the existing
  24,000-character cap.

The amended plan also requires exact parsed key facts, no `invalid_draft`
vocabulary in exact-field model feedback, separate real-LLM test functions,
and independent near-cap coverage for `a1a573`, `caad1a`, and `df6eb4`.

Review record:

[`cognition_core_v2_goal_schema_semantic_repair_plan_review_20260809.md`](../../../test_artifacts/diagnostics/cognition_core_v2_goal_schema_semantic_repair_plan_review_20260809.md)

## Retrieved failure evidence

- 87 post-draft Cognition trace runs.
- 43 traces with failed goal attempts.
- 70 failed goal-model attempts.
- Five terminal episodes and six terminal branches.
- All six terminal branches are the recurring non-ordinary exact-field
  contamination family.
- The `caad1a...` and `df6eb4...` terminal traces also contain semantic
  context-limit recurrences.
- The other 45 failures are recoverable ordinary relational/evidence contract
  controls, not additional terminal root causes.

Primary evidence:

- `test_artifacts/diagnostics/cognition_goal_bid_postdraft_trace_runs.json`
- `test_artifacts/diagnostics/cognition_goal_bid_postdraft_failed_attempts_all.json`
- `test_artifacts/diagnostics/cognition_goal_bid_postdraft_failure_capsules_all_goal.json`
- `test_artifacts/diagnostics/cognition_goal_bid_postdraft_failure_review_20260809.md`

## Baseline

- Source commit: `a1fe299f3ef8ae2056f589a0a131448c08d73bcc`.
- Baseline deterministic result before production edits: `96 passed, 2
  skipped`.
- Baseline live collection before the latest additions: 20 tests collected.
- Baseline production hashes are recorded in
  `test_artifacts/diagnostics/cognition_core_v2_goal_schema_semantic_repair_signoff_manifest_20260808.json`.
- No `.env` file was inspected.

The worktree was already dirty before production implementation. Current
status at handover:

```text
 M development_plans/README.md
 M src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py
 M src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py
 M tests/test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm.py
 M tests/test_cognition_core_v2_trace_failure_modes_live_llm.py
?? development_plans/active/bugfix/cognition_core_v2_goal_schema_and_semantic_repair_compaction_bugfix_plan.md
?? development_plans/active/bugfix/cognition_core_v2_goal_schema_semantic_repair_handover_20260809.md
?? tests/test_cognition_core_v2_self_improvement_schema_live_llm.py
```

## Changes made before pausing

### Plan and evidence

- Updated the active plan to `in_progress`.
- Updated the lifecycle registry to `in progress`.
- Added the `gpt-5.6-sol` review record and baseline sign-off manifest under
  `test_artifacts/diagnostics/`.
- Added post-draft failure counts, terminal trace identities, and the three
  semantic envelope requirements to the plan.

### Goal production code — incomplete, unverified

`src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` now contains:

- a separate `NON_ORDINARY_GOAL_COGNITION_PROMPT` without the relational schema;
- a separate `NON_ORDINARY_GENERIC_GOAL_REPAIR_INSTRUCTIONS` tuple;
- branch-specific generic prompt selection;
- parsed observed/missing/unexpected key feedback for the exact-field error;
- omission of the complete `invalid_draft` value and its vocabulary for that
  exact-field feedback path.

Required follow-up:

- compile and run focused tests;
- inspect every existing assertion for the old repair payload;
- verify ordinary generic, required-selection, and active-branch contracts;
- verify the exact captured shape has `missing_top_level_fields == []` and
  `unexpected_top_level_fields == ["relational_willingness"]`;
- verify the non-ordinary prompt contains no relational schema vocabulary.

### Semantic production code — incomplete and currently references a missing helper

`src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py` now passes a
projected error to `_appraisal_repair_messages` and preserves `str(exc)` in the
existing trace/capture calls. However, the helper
`_compact_semantic_contract_error` has not yet been defined. The next session
must define and test it before running pytest.

Required helper behavior:

- compact only the validator-owned suffix beginning exactly
  `; permitted paths:` for the unowned semantic-delta-path error;
- preserve the failed rule and exact offending path;
- return other contract-error strings unchanged;
- leave protected `validation_error` and failure-capsule evidence as the full
  original `str(exc)`;
- keep the existing cap and retry behavior unchanged.

### Tests — partially updated, unverified after the latest edits

- `tests/test_cognition_core_v2_self_improvement_schema_live_llm.py` contains
  separate test functions for the two plan traces and one autonomy-boundary
  post-draft trace.
- `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py` contains the
  historical unowned-path/live-repair hybrid and checks compact repair facts.
- `tests/test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm.py` was
  extended to separate `a1a573`, `caad1a`, and `df6eb4` near-cap cases.
- Existing deterministic tests have not yet been reconciled with the new
  production contract.
- No test or compile command was run after the latest production/test edits.

The live tests intentionally use preserved historical invalid candidates for
the first call and a real model for the repair call. Their artifacts must label
that hybrid clearly and each case must be run individually.

## Delegation state

The requested `gpt-5.6-sol` reviewer completed successfully and its result was
integrated. The project `kazusa_plan_worker` Workstream A handoff completed
with disjoint changes to goal repair behavior and its focused tests. The parent
completed Workstream B, integrated the worker result, ran all required gates,
and obtained a final read-only `kazusa_plan_reviewer` PASS.

## Completion record

- The copied ordinary trace is preserved at
  `test_artifacts/diagnostics/turn_173098348_llm_trace.json`; source and target
  SHA256 are identical.
- Focused deterministic verification: 84 passed, 19 deselected.
- Broader Cognition V2 deterministic verification: 527 passed, 2 skipped,
  237 deselected, with the unrelated pre-existing preference prompt cap
  assertion recorded as a waiver.
- Three non-ordinary goal live hybrids, ordinary captured relational replay,
  unowned semantic-path replay, and three near-cap semantic live cases were
  run one at a time and inspected.
- Parent review:
  `test_artifacts/diagnostics/cognition_core_v2_goal_schema_semantic_repair_review_20260808.md`.
- Final independent review: PASS with no blocking findings.

## Original fresh-session execution order

1. Read this handover, the active plan, `development_plans/README.md`, and the
   required project skills before acting.
2. Inspect the current diff and define `_compact_semantic_contract_error` in
   `semantic_appraisal.py`.
3. Reconcile deterministic goal, semantic-budget, dependency, prompt-guidance,
   failure-matrix, ledger, and trace-capture tests.
4. Update `src/kazusa_ai_chatbot/cognition_core_v2/README.md` as required by
   the plan.
5. Run `py_compile`, `git diff --check`, and focused deterministic tests.
6. Run each live case separately, inspecting its raw artifact before starting
   the next case:
   - two plan self-improvement schema cases;
   - one post-draft autonomy-boundary schema case;
   - ordinary relational recovery controls;
   - unowned semantic-path repair;
   - `a1a573`, `caad1a`, and `df6eb4` near-cap repair cases.
7. Verify all three semantic repair payloads fit below 24,000 characters and
   construct a second invocation.
8. Write the parent-authored debug-LLM quality review, perform an independent
   code review, update the plan checklist, and only then close the lifecycle.

Do not revert the pre-existing dirty test changes. Do not increase caps, relax
validators, change retry budgets, or modify files outside the amended plan
change surface.
