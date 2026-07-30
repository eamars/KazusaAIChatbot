# Required-Selection Contract Separation Bugfix Plan

## Summary

- Goal: eliminate partially failed active-goal bids caused by the overloaded
  required-selection output contract.
- Plan class: `medium`.
- Status: `completed`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `debug-llm`,
  `test-style-and-execution`, `py-style`, and `cjk-safety`.
- Cutover: one big-bang producer-contract update; no compatibility vocabulary.
- Acceptance: reproduce the production-shaped failure, probe adjacent cases,
  replace the contract, and pass the same one-at-a-time real-LLM gates.

## Context

Production trace `llmtrace_efa8a6440f5742cfb20c3fe14717fb4f` showed the
`autonomy_boundary` bid exhausting its required-selection structure attempts
while the ordinary bid succeeded. The current worker had loaded the intended
regeneration prompt. A source-faithful live test reproduced the first-attempt
malformed JSON on the active 26B route.

The producing model currently owns three responsibilities in one object:

1. make the character-owned semantic selection;
2. cite required operation and conversation-progress evidence; and
3. emit an exact per-progress relation matrix.

Deterministic validation requires asymmetric handle membership across
`evidence_handles` and `conversation_evidence_relations`, while the relation
matrix is discarded when the selection draft becomes an `ActionBidV2`. This
raises structural load without supplying a downstream contract.

## Target State

Required-selection turns use one producing LLM call under the existing bounded
three-attempt policy:

1. Deterministic code partitions required operations, conversation-progress
   constraints, and optional supporting evidence by provenance.
2. The LLM owns only the character's selection and supporting semantic fields.
3. Deterministic validation owns exact fields, bounds, allowed handles, and
   mandatory citation coverage.
4. Conversation-progress constraints remain model-visible and mandatory to
   cite, but no discarded relation matrix is produced.
5. Required-selection production uses the existing dense ordinary-goal route
   for every branch, including active persistent-goal branches.
6. Generic active-goal cognition without a typed required selection continues
   to use the active-branch route.
7. No new response-path LLM call, evaluator, parser, fallback mapper, database
   shape, or public API is introduced.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
- `README.md`
- `docs/HOWTO.md`
- Focused deterministic tests that directly assert the producer contract,
  prompt partition, prompt cap, and model routing.
- `tests/test_cognition_core_v2_required_selection_live_llm.py`
- `test_artifacts/reviews/required_selection_contract_live_llm_review.md`
- This plan and `development_plans/README.md`.

### Keep

- Existing action-bid shape and downstream consumers.
- Existing deterministic-only JSON cleanup for required selection.
- Existing total attempt cap and fail-closed typed execution error.
- Existing active-goal route for non-selection active branches.
- Conversation Progress V2 persistence, projection, and provenance.

### Delete

- Required-selection `conversation_evidence_relations` output contract,
  validation, regeneration feedback, tests, and documentation.

## Design Decisions

| Topic | Decision | Reason |
|---|---|---|
| Semantic owner | One required-selection producer | The character choice remains LLM-owned. |
| Deterministic owner | Input partition, handle domains, citation coverage | These are provenance and contract checks. |
| Continuity | Model-visible progress constraints plus mandatory citations | Preserves semantic continuity without a discarded matrix. |
| Model route | Existing dense ordinary-goal route for all required selections | The reproduced failure is specific to the weaker active route under contract pressure. |
| Retries | Existing producer-only regeneration | Preserves call count and ownership. |
| Cutover | Big-bang exact-field update | Avoids aliases and parallel contract vocabularies. |

## Agent Autonomy Boundaries

- The parent owns test authoring, real-LLM execution and inspection, evidence,
  documentation, plan lifecycle, and review remediation.
- Exactly one production-code subagent implements the bounded production and
  deterministic-test change after pre-fix evidence is recorded.
- Exactly one independent review subagent reviews the final diff and evidence
  without implementing.
- Preserve unrelated worktree changes and the active Conversation Progress V2
  sign-off plan.
- Stop if the fix requires another response-path LLM call, a database change,
  or a route outside the listed cognition goal owners.

## Implementation Order

1. Record the exact production-scene reproduction and current route evidence.
2. Add adjacent real-LLM probes for branch transfer and combined mandatory
   evidence pressure.
3. Run each adjacent case alone and inspect raw outputs before implementation.
4. Delegate the bounded production/test implementation to one subagent.
5. Review the patch, remediate focused issues, and compile every changed Python
   file immediately after edits.
6. Run focused deterministic tests.
7. Run every selected real-LLM case one at a time and inspect each artifact.
8. Run adjacent non-live and full non-live regression.
9. Delegate one independent code review, remediate blockers, and rerun affected
   gates.
10. Record execution evidence and archive the completed plan.

## Verification

### Pre-fix real-LLM evidence

Run alone:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_required_selection_live_llm.py::test_live_autonomy_selection_with_exact_production_scene -q -s
```

Then run each newly added adjacent case alone. Inspect route/model, all raw
attempts, parser disposition, failure, and final bid before continuing.

### Focused deterministic gates

```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_dependencies.py tests\test_cognition_core_v2_stage_model_routing.py tests\test_conversation_progress_stage12_architecture.py tests\test_conversation_progress_v2_regression.py tests\test_cognition_core_v2_prompt_budget_continuity.py -q
```

### Post-fix real-LLM gates

Run the exact production scene and every adjacent case one at a time. Expected:

- active required-selection calls report the dense ordinary-goal route/model;
- the prompt exposes disjoint required operations, progress constraints, and
  supporting evidence;
- outputs contain the exact reduced selection fields;
- every mandatory handle is cited;
- no relation matrix appears;
- the branch returns a complete bid without structure exhaustion.

### Regression

```powershell
venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q
```

## Progress Checklist

- [x] Stage 1 — source-faithful production failure reproduced.
- [x] Stage 2 — adjacent pre-fix real-LLM probes recorded and inspected.
- [x] Stage 3 — separated producer contract and dense route implemented.
- [x] Stage 4 — focused deterministic and one-at-a-time real-LLM gates pass.
- [x] Stage 5 — runnable regression and independent review pass.
- [x] Stage 6 — evidence recorded and plan archived.

## Independent Code Review

The reviewer checks:

1. the relation matrix and all aliases are removed in one cutover;
2. deterministic provenance partitioning cannot hide mandatory facts;
3. active required selections use the dense route while other active branches
   retain their current route;
4. prompt fitting protects required operations and progress constraints;
5. parser, retry, trace, and failure semantics remain bounded;
6. live evidence supports the claimed structural and adjacent improvements;
7. no hidden keyword semantic ownership, CJK quoting hazard, or unrelated
   change is present.

## Acceptance Criteria

- The exact production-shaped failure is preserved as pre-fix evidence.
- At least two adjacent real-LLM cases are run and inspected before the fix.
- Required operations, progress constraints, and supporting evidence are
  disjoint in the model payload.
- Required-selection output has one exact reduced schema and no relation
  matrix.
- Mandatory citations and bounded structural regeneration remain enforced.
- Active required-selection branches use the configured dense goal model.
- The exact and adjacent real-LLM cases pass individually after the fix.
- Focused and all runnable non-live regression pass; unrelated missing local
  fixture/module exclusions are documented.
- Independent review has no unresolved blocker.

## Execution Evidence

- Pre-fix exact reproduction: recorded in
  `test_artifacts/llm_traces/cognition_core_v2_required_selection_live_llm__autonomy_exact_production_scene.json`.
- Adjacent pre-fix evidence:
  - `relationship_connection` reproduced the asymmetric-domain error on the
    active 26B route. Attempt one incorrectly placed required-operation handle
    `e1` in the progress-only relation array; attempt two recovered.
  - `autonomy_boundary` with two required operations, four progress events,
    and three supporting rows passed on attempt one. This bounds cardinality
    as pressure rather than a sufficient root cause.
  - Raw artifacts:
    `cognition_core_v2_required_selection_live_llm__relationship_production_state.json`
    and
    `cognition_core_v2_required_selection_live_llm__autonomy_compound_mandatory_pressure.json`.
- Implementation:
  - Required-selection routing now selects
    `goal_ordinary_response_config` before branch-route fallback.
  - Prompt input is partitioned into `required_selection_operations`,
    `conversation_progress_constraints`, and `supporting_evidence`.
  - The retired relation matrix is absent from producer output, validation,
    regeneration feedback, tests, and ICD documentation.
  - Mandatory citations, deterministic-only parsing, three attempts, tracing,
    and fail-closed errors remain intact.
  - Focused deterministic verification after review remediation:
    `82 passed`.
- Post-fix real-LLM evidence:
  - Five selected cases passed one at a time.
  - Every case used one
    `COGNITION_LLM_GOAL_ORDINARY_RESPONSE` call on
    `gemma-4-31b-fable-5-agent-distill`.
  - Exact production scene: 1 operation, 1 progress constraint, 7 supporting
    rows; cited `e1,e2`.
  - Relationship branch transfer: 1/1/7; cited `e1,e2`.
  - Compound pressure: 2 operations, 4 progress constraints, 3 supporting
    rows; cited `e1` through `e6`.
  - Multi-progress production state: 1/4/4; cited `e1` through `e5`.
  - Alias collision: only provenance-owned `e2` entered the one-row progress
    lane; seven semantic lookalikes remained supporting evidence.
  - Ten-mandatory-citation boundary: 2 operations, 8 progress constraints,
    and 0 supporting rows; attempt one missed four citations, bounded
    regeneration cited exactly `e1` through `e10`, and the bid passed.
  - Final exact-production rerun after review remediation passed on one dense
    model call with `e1` and `e2` cited.
- Regression:
  - Focused deterministic batch: `82 passed`.
  - Changed-file compile and `git diff --check`: passed.
  - Standard full non-live collection is blocked by unrelated absent local
    modules and fixtures.
  - All runnable non-live tests passed with exit code `0` after excluding the
    seven files whose failures are exclusively missing `asuna.json`, replay
    manifests, real-history exports, a replay experiment module, or a
    historical audit artifact.
- Independent review:
  - Initial verdict: BLOCKED on the ten-citation boundary, required-selection
    regeneration aggregate budgeting, and stale test vocabulary.
  - Remediation: mandatory citations may expand the selection-only handle
    limit; each required-selection replacement refits against its complete
    current system prompt; stale descriptions were updated.
  - Re-review: APPROVED. The remaining non-blocking wording note was
    remediated and the changed test file compiled cleanly.
  - Residual risks: stochastic output can still consume bounded regeneration;
    live coverage proves ten mandatory handles; the standard full-suite
    collector still depends on absent local fixtures and modules.
