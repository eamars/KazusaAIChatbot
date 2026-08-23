# Cognition V3 State Transaction, Capacity, And Context Integrity Bugfix Plan

## Summary

- **Goal:** Prevent mechanically recoverable cognition-state pressure from
  aborting cognition, restore the deterministic lifecycle lost during the
  handleless cutover, and ensure each cognition stage receives the semantic
  state features it owns.
- **Status:** `completed` and archived on 2026-08-23.
- **Scope boundary:** Canonical Cognition V3, shared cognition state owners,
  the immediate cognition connector and final cognition-state commit, plus
  directly governed V3 tests and documentation.
- **Change direction:** Keep the one-pass handleless A1/A2/G/P semantic flow;
  replace direct per-turn persistence with one cap-aware deterministic state
  transaction and typed stage-specific context projection.
- **Cutover:** Big-bang correction of the current V3-only contract. No V2
  runtime, compatibility layer, alias, fallback schema, or migrated V2 test.
- **Acceptance state:** Passed all twelve scoped acceptance criteria. The live
  cognition sample and its human review are recorded below.

## Evidence

- Production trace: `llmtrace_1051202cbb394e078d3dde351dc81657`.
- Production failure: `CognitionStateError: active_events exceeds its state
  cap`, no dialog, `graph_failure:internal_invariant`.
- Production user state: 32 events comprising 26 terminal rows and 6 active
  rows.
- Deterministic reproduction: the 33rd direct event fails before pruning.
- Clean-state reproduction: the 17th ordinary turn fails on the goal cap.
- Character-state reproduction: the replacement timestamp does not advance,
  violating the character compare-and-replace contract.
- Two-turn reproduction: continuation lookup selects the oldest ordinary goal
  instead of the current goal.
- Context projection reproduction: character constraints, identity,
  operational state, relationship context, and affect cause project to empty
  or causally incomplete prompt objects.
- Full review artifact:
  `test_artifacts/diagnostics/llmtrace_1051202_cognition_state_rca_2026-08-23.md`.

## Confirmed Decisions

1. The event overflow is an immediate trigger, while the root defect is the
   handleless binder bypassing the canonical lifecycle transaction and
   promoting transient turn products into durable state.
2. A valid state at its declared cap is accepted input. A new insertion first
   reclaims the oldest unprotected terminal capacity and then validates.
3. Active or fading affect roots and meaningful active causal entities are
   never evicted merely because they are old.
4. When no semantically safe persistent capacity exists, cognition retains
   the prior valid state for that mutation, records a private typed
   `capacity_deferred` receipt, and continues the already-produced semantic
   cognition output. Deterministic code does not invent a replacement cause.
5. The current observation remains turn-local unless at least one applicable
   non-zero state change requires a durable event, threat, or knowledge-gap
   root.
6. The current response goal remains turn-local for an answerable ordinary
   response. It becomes a durable `ordinary_response` goal only when
   `goal_resolution` is not `answerable_now`, a task resolver needs
   continuation, or an accepted coding task needs continuation.
7. A durable continuation carries its exact caller-private goal reference in
   `state_projection`. Callers never scan persistent goals by kind or order.
8. Stable and uncertain zero-shift rows do not create durable causal entities.
9. Trusted direct facts, elapsed evolution, relationship maintenance, guarded
   lifecycle transitions, affect derivation, monotonic timestamps, retention,
   and final state validation remain deterministic cognition features.
10. Legacy deterministic multi-goal synthesis, model-emitted handles, bid
    rosters, sibling salvage, semantic validators, semantic repair loops, and
    prompt tuning are excluded.
11. The immutable persisted base state is carried across resolver recurrence.
    The final replacement commits against that base rather than the prior
    uncommitted cycle.
12. The configured cognition turn deadline bounds the complete A1/A2/G/P
    operation. Provider or structurally unusable model output remains a typed
    fail-closed operational failure because deterministic code cannot invent
    character judgment.
13. Emotion identity and concrete cause remain preserved in persistence,
    output projection, and subsequent prompt input.
14. Prompt context uses explicit typed stage projections. Generic recursive
    allowlisting no longer silently strips required context.
15. Evidence admission is priority-based: current episode and current
    resolver/action results, current media, then bounded continuity and RAG
    evidence. Existing stage visibility metadata is enforced.
16. The web control console remains implementation-agnostic. This plan changes
    no console topology, prompt, stage label, or private state detail.

## Scope And Change Direction

### In scope

- Cap-aware native insertion for events, threats, gaps, and continuation
  goals.
- One deterministic state transaction around the handleless semantic products.
- Transient-versus-durable admission based on applied state change and
  continuation need.
- Monotonic state timestamps and final complete-state validation.
- Original-base commit lineage across resolver recurrence.
- Exact current continuation-goal binding.
- Typed stage context and priority evidence projection.
- Enforcement of the existing full-turn deadline.
- Deletion of the obsolete V2-stage connection test cluster rather than
  migration.
- One synthetic full-cap live-LLM cognition case, run and inspected alone.

### Excluded

- Dialog wording, surface quality, adapters, delivery, consolidation,
  reflection, scheduler behavior, and control-console internals.
- Prompt tuning, retry prompts, semantic scoring, output repair, or a second
  semantic authority.
- V2 restoration, compatibility, aliases, shims, or test migration.
- Changes to declared state caps.
- Arbitrary eviction of meaningful active state.
- Broad test-suite rewrites or documentation tests.

## Target State

```text
validated persisted base
  -> monotonic episode mutation time
  -> trusted facts + elapsed lifecycle + bounded retention
  -> typed stage-specific prompt workspace
  -> A1 once -> A2 once -> G once -> P once
  -> admit only required durable roots
  -> bind guarded deltas and exact continuation reference
  -> relationship maintenance + affect derivation with concrete causes
  -> prune safe terminal/inert rows
  -> validate one complete replacement
  -> commit final replacement against immutable persisted base
```

### Capacity contract

For each capped causal collection:

1. Reuse an existing eligible row with the same source identity.
2. Admit a new row only when the semantic product requires durable state.
3. Remove the oldest unprotected terminal rows needed to meet the cap.
4. Reclaim an inert V3 turn row only when its lifecycle fields prove it has no
   unresolved pressure and no affect reference.
5. If capacity remains protected, retain the previous valid collection and
   emit a private `capacity_deferred` binding receipt.
6. Validate only the post-retention state, never the transient cap-plus-one
   intermediate.

### Stage context contract

| Stage | Required semantic context |
| --- | --- |
| A1 | Current observation, current resolver/action evidence, event/threat/gap state needed for world appraisal, and bounded supporting evidence |
| A2 | Accepted A1 meaning, relationship axes and causal context, affect identity and concrete causes, standards, boundaries, drives, meaning state, and the A2 identity partition |
| G | Compact A1/A2 meaning, continuing durable goals, active pressures, affect with causes, relationship, identity, boundaries, and continuity |
| P | Exactly one accepted active-character goal, response ownership, resolver progress needed for planning, and only currently available semantic capabilities |

No stage receives persistent IDs, model handles, target paths, adapter syntax,
raw traces, or another stage's output schema.

## Must Do

1. Restore direct facts and elapsed lifecycle before model-facing projection.
2. Derive one monotonic mutation timestamp strictly later than the persisted
   state version when a state mutation is committed.
3. Replace unconditional `need_event = True` and unconditional ordinary-goal
   persistence with the admission contract above.
4. Route every durable capped insertion through one existing or canonical
   cap-aware reducer boundary.
5. Preserve exact active and affect-protected entities during pruning.
6. Complete relationship maintenance and affect derivation before final
   validation.
7. Validate the replacement after `affect_activations` is assigned.
8. Carry the original persisted user base through every resolver cycle.
9. Read character scope from canonical `state_projection` and supply the
   original character version to commit.
10. Carry the exact current continuation goal reference privately.
11. Replace generic context filtering with explicit typed projection.
12. Prioritize current resolver/action observations ahead of supporting RAG
    evidence under the existing bounded evidence count.
13. Enforce `turn_deadline_seconds` around the complete canonical cognition
    operation.
14. Update cognition and node READMEs to describe the resulting transaction
    and transient/durable boundary.
15. Delete the obsolete V2-stage connection test files rather than adapt them
    to V3.
16. Update the source-to-test impact manifest only for changed production
    owners.

## Deferred

- State-cap size tuning.
- New background compaction services.
- New telemetry dashboards or console panels.
- Semantic coalescing of unrelated active causes.
- Dialog or surface-quality changes.
- Cleanup of cognition-shared code that is unreachable but unrelated to this
  failure transaction.

## Cutover Policy

Overall strategy: `bigbang`.

| Area | Policy | Instruction |
| --- | --- | --- |
| Cognition state binding | bigbang | Replace the direct append/validate path with the one transaction; keep no alternate binder |
| Prompt projection | bigbang | Replace the generic context allowlist with typed stage projections |
| Commit lineage | bigbang | Use one canonical state-projection/base contract across live and self cognition |
| Tests | bigbang | Delete V2-stage connection tests; add only direct V3 contract checks |
| Persisted state | compatible data shape | Keep the current cognition-state schema and rows; no migration or dual read/write is required |

## Execution Roles

### Cognition implementation and verification owner

- **Responsibility:** Implement the approved cognition-only correction and
  produce focused deterministic plus one-case live evidence.
- **Owned surface:** The production and test paths listed in Change Surface.
- **Authority:** Edit only the approved cognition boundary, its immediate
  commit connector, directly governed tests, READMEs, and impact manifest. No
  DB writes outside an explicitly selected test operation.
- **Applicable skills:** `development-plan`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, `debug-llm`, and
  `llm-trace-debug` when production evidence is revisited.
- **Capability floor:** System-level cognition architecture, deterministic
  state reducers, async commit lineage, typed prompt projection, and ability
  to run and inspect one real-LLM case.
- **Independence requirement:** None. The owner explicitly required the same
  Luna agent for implementation and verification; root performs final diff
  and evidence review.
- **Acceptance output:** Scoped production diff, exact deterministic test
  results, one individually inspected live-LLM artifact, and updated plan
  evidence.
- **Gate:** Starts after owner approval; exits after every acceptance criterion
  is evidenced.
- **Plan-scoped fixed execution constraint:** Reuse the existing
  `gate7_luna` agent (`gpt-5.6-luna`, maximum reasoning, normal speed) for both
  production implementation and test execution, as directed by the owner.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py`
  - implement transient/durable root admission and return the exact private
    continuation reference and capacity receipts.
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
  - restore the one state transaction, enforce the full-turn deadline, and
    validate the final state after affect derivation.
- `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`
  - replace generic filtering with typed stage projections and priority
    evidence selection.
- `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py`
  - define and validate the private state-projection fields required by the
    transaction without exposing them to the model or console.
- `src/kazusa_ai_chatbot/cognition_shared/state_reducers.py`
  - make causal-root admission prune-before-validate and expose existing
    lifecycle owners to the canonical transaction.
- `src/kazusa_ai_chatbot/cognition_shared/state_models.py`
  - change only if the approved safe inert-row retention rule requires an
    explicit canonical predicate; retain existing caps and protected-root
    semantics.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - carry immutable base state and exact continuation reference through
    recurrence and commit.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - use canonical `state_projection` for character commit dispatch.
- `src/kazusa_ai_chatbot/cognition_core_v3/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
  - document stable runtime ownership and failure behavior.
- `tests/ownership/source_test_impact_manifest.json`
  - replace stale mappings only for production paths changed by this plan.
- `development_plans/README.md`
  - maintain lifecycle registration.

### Create

- `tests/unit/cognition_core_v3/test_state_transaction.py`
- `tests/unit/cognition_core_v3/test_prompt_context.py`
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py`
- `tests/test_cognition_v3_capacity_live_llm.py`
- one human-readable live-LLM review artifact under
  `test_artifacts/diagnostics/`.

### Delete

- `tests/test_cognition_stage_connection_live_llm.py`
- `tests/test_cognition_stage_connection.py`
- `tests/cognition_stage_connection_cases.py`

These tests encode the deleted L1/L2a/L2b/L2c/L2d/V2 stage topology and are
deleted rather than migrated.

### Keep

- Existing state schema version and cap values.
- Existing A1/A2/G/P model-facing semantic shapes unless an exact private
  state carrier requires a non-model contract addition.
- Existing control-console public semantic projection.
- Existing terminal and affect-root protection rules.

## Test Impact And Traceability

| Source or governed artifact | Changed contract | Semantic owner | Exact deterministic pytest nodes | Supplemental node | Mode | Regression prevented |
| --- | --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/cognition_shared/state_reducers.py` | cap-aware causal admission | state reducer owner | `tests/unit/cognition_core_v3/test_state_transaction.py::test_state_transaction_reclaims_terminal_capacity_and_preserves_active_causes` | none | deterministic unit | legal full state crashes before safe prune |
| `src/kazusa_ai_chatbot/cognition_shared/state_models.py` | protected and inert retention predicate, if changed | state model owner | `tests/unit/cognition_core_v3/test_state_transaction.py::test_state_transaction_reclaims_terminal_capacity_and_preserves_active_causes` | none | deterministic unit | meaningful active or affect-root entity is evicted |
| `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py` | transient/durable admission and exact goal reference | appraisal binding owner | `tests/unit/cognition_core_v3/test_state_transaction.py::test_repeated_answerable_turns_do_not_accumulate_transient_events_or_goals` `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_current_continuation_uses_exact_private_goal_ref` | none | deterministic unit | 17-turn goal exhaustion and stale goal continuation |
| `src/kazusa_ai_chatbot/cognition_core_v3/facade.py` | lifecycle transaction, final validation, and deadline | cognition facade owner | `tests/unit/cognition_core_v3/test_state_transaction.py::test_character_state_transaction_advances_timestamp_and_validates_final_affect` `tests/unit/cognition_core_v3/test_state_transaction.py::test_cognition_turn_deadline_bounds_full_chain` | `tests/test_cognition_v3_capacity_live_llm.py::test_live_captured_full_state_greeting_completes_first_pass` | deterministic unit + live LLM | character commit rejection, invalid final affect, unbounded provider stall |
| `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py` | private state-projection shape | V3 contract owner | `tests/unit/cognition_core_v3/test_state_transaction.py::test_character_state_transaction_advances_timestamp_and_validates_final_affect` | none | deterministic unit | missing base/current-goal/capacity disposition carrier |
| `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` | typed stage context and priority evidence | prompt projection owner | `tests/unit/cognition_core_v3/test_prompt_context.py::test_stage_context_preserves_identity_relationship_and_emotion_cause` `tests/unit/cognition_core_v3/test_prompt_context.py::test_current_resolver_and_action_evidence_precedes_supporting_rag` | `tests/test_cognition_v3_capacity_live_llm.py::test_live_captured_full_state_greeting_completes_first_pass` | deterministic unit + live LLM | empty character/relationship context, lost emotion cause, resolver evidence starvation |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | immutable recurrence base and exact continuation | cognition connector owner | `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_resolver_recurrence_commits_against_original_user_base` `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_current_continuation_uses_exact_private_goal_ref` | none | deterministic unit | multi-cycle user CAS conflict and wrong goal lineage |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | canonical character commit dispatch | persona graph owner | `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_persona_character_commit_reads_canonical_state_projection` | none | deterministic unit | missing character base version at final commit |
| `tests/ownership/source_test_impact_manifest.json` | current exact source-to-test mapping | verification owner | `tests/test_test_impact_manifest.py::test_source_test_impact_manifest_is_complete_and_collectable` | none | deterministic unit | changed cognition source accepted without an exact owner node |

The implementation owner may combine setup inside these named tests, but the
exact nodes and observable contracts remain fixed. No assertion targets prompt
wording, exact model prose, or internal stage order beyond the canonical
one-call A1/A2/G/P contract.

## Verification

1. Re-run the effect-free production-state capacity reproduction before the
   fix and record the exact failure.
2. Run each exact deterministic node in the traceability table.
3. Run the manifest-backed changed-source impact command.
4. Run one live-LLM case only:
   `tests/test_cognition_v3_capacity_live_llm.py::test_live_captured_full_state_greeting_completes_first_pass`.
5. Inspect that live artifact manually for:
   - four first-pass usable model products;
   - non-empty character-owned goal and plan;
   - no role reversal, material self-conflict, or supplied boundary/safety
     conflict;
   - complete valid replacement state at or below every cap;
   - active/protected cause preservation;
   - no `capacity_deferred` in the captured terminal-cap case because safe
     terminal capacity exists;
   - preserved emotion identity and concrete cause in A2/G context when affect
     is present.
6. Run `git diff --check` and review the final scoped diff.

## Acceptance Criteria

1. The exact production state with 32 events completes state binding, retains
   all 6 active rows and all affect-protected roots, admits the new required
   durable root when applicable, and remains at or below 32.
2. Forty sequential answerable ordinary turns do not accumulate durable
   ordinary-response goals or neutral observation events and never raise a
   state-cap error.
3. A collection containing only protected meaningful active rows leaves the
   prior valid state intact, emits `capacity_deferred`, and returns semantic
   cognition output without a state exception.
4. Stable or uncertain zero-shift appraisals create no threat, event, or gap.
5. Character replacement time strictly advances and the complete replacement
   validates after affect derivation.
6. A state-changing two-cycle resolver run commits against the original user
   base and uses the exact current continuation goal.
7. Character-scope persona and self-cognition commits receive the original
   character version through the canonical projection.
8. A2 and G receive non-empty required character, relationship, continuing
   state, emotion identity, and concrete emotion-cause context; P remains
   limited to the selected goal and available capabilities.
9. Current resolver/action evidence cannot be displaced by supporting RAG rows
   at the evidence cap.
10. The configured turn deadline is enforced.
11. The one live-LLM captured greeting case completes on first pass and is
    manually acceptable under the owner's relaxed semantic rubric.
12. No V2 runtime, compatibility path, migrated V2 test, semantic validator,
    repair loop, or prompt-specific output tuning is introduced.

## Agent Autonomy Boundaries

The implementation owner may choose local function decomposition and command
order within the listed files. The owner may reuse existing canonical
reducers and projection functions when they satisfy the fixed contracts.

A plan amendment or owner decision is required before changing state caps,
persisted schema shape, A1/A2/G/P model-facing fields, the semantic pass
criteria, control-console contracts, or any subsystem outside cognition and
its immediate commit boundary.

## Execution Evidence And Closure

- The reused GPT-5.6 Luna implementation and verification owner completed the
  production-code slice, the focused regression slice, and the live-LLM run.
- Final focused cognition verification reported `26 passed`; the final exact
  continuation/capacity regression reported `1 passed`. Earlier scoped commit
  and public-contract batches reported `3 passed` and `3 passed`, respectively.
- The final architecture audit removed residual orientation and resolver-progress
  context from P. Its packet now contains only `stage`, instruction guidance,
  the selected goal, advertised capabilities, and its output contract; the
  exact packet-boundary node passed (`1 passed`).
- Python compilation and Ruff `F`, `E4`, `E7`, `E9`, and `I` checks passed for
  the changed Python surface. `git diff --check` passed.
- The strict repository-wide source-impact validator still reports 37 inherited
  manifest errors outside this change. Every changed cognition source has a
  valid exact owner-node entry. The inherited manifest debt is classified as
  separate quality work and does not keep this plan open.
- The one required effect-free live command passed:
  `venv\Scripts\python -m pytest tests\test_cognition_v3_capacity_live_llm.py::test_live_captured_full_state_greeting_completes_first_pass -q -s -o addopts=`
  (`1 passed in 21.83s`).
- Protected raw artifact:
  `test_artifacts/diagnostics/cognition_v3_capacity_live_llm/capacity_transaction_1787440952077052900.json`,
  SHA-256
  `82488254A155FC95EEFD8AC63852F9DFA5C7D27C1E922F3B8C24A22CF3BDE28F`.
- Human review artifact:
  `test_artifacts/diagnostics/cognition_v3_capacity_live_llm_review_2026-08-23.md`,
  SHA-256
  `156B275AD917832B92F5556E9FDAEE1316930FD283FBB306D7916ED48FB06FB9`.
- The live run issued exactly one parsed A1, A2, G, and P call. It retained all
  six pre-existing active events, reclaimed one safe terminal event, admitted
  the new evidence-grounded event, finished at the 32-event cap, emitted no
  `capacity_deferred`, and contained no role reversal, material self-conflict,
  or boundary/safety conflict.
- Focused deterministic evidence covers fully protected-capacity deferral,
  transient answerable goals, evidence-free no-op admission, relationship
  ledger rotation, final affect derivation, monotonic character timestamps,
  original-base recurrence commit, exact continuation lineage, full-chain
  timeout, priority evidence, and the typed stage context projections.
- Emotion remains a typed value with concrete cause provenance. The live
  low-pressure greeting produced no affect activation; the deterministic
  persisted-affect projection test verifies that both emotion identity and its
  cause reach A2 and G when affect is present.
- The production `run_cognition` facade was exercised with a real model and the
  captured full-cap state without persistence effects. Immediate connector and
  compare-and-replace lineage were verified deterministically; no production
  database mutation was part of this cognition-scoped plan.
- Control-console contracts and files remained unchanged. The console consumes
  the existing public semantic projection and stays agnostic to the internal
  transaction implementation.
- Stale V2 cognition-stage tests were deleted rather than migrated. The change
  adds no V2 runtime, compatibility shim, handle protocol, semantic validator,
  semantic repair loop, or prompt-specific output tuning.

### Acceptance Disposition

1. **Passed:** full 32-event input safely reclaimed one terminal row and
   preserved all active rows.
2. **Passed:** repeated answerable turns remain transient and cap-safe.
3. **Passed:** fully protected capacity rolls back the mutation, records
   `capacity_deferred`, and preserves semantic output.
4. **Passed:** zero-shift or evidence-free proposals create no durable causal
   entity.
5. **Passed:** character replacement time advances and final state validates
   after affect derivation.
6. **Passed:** recurrence uses the original persisted base and exact current
   continuation goal.
7. **Passed:** character-scope commits receive the canonical original version.
8. **Passed:** A2/G receive owned identity, relationship, continuing-state,
   emotion, and cause context; P remains goal/capability scoped.
9. **Passed:** current resolver/action evidence wins admission priority.
10. **Passed:** one deadline bounds the complete cognition chain.
11. **Passed:** the required first-pass live case passed manual semantic review.
12. **Passed:** the cutover remains V3-only and validator/repair-free.

## Progress Checklist

- [x] Production trace and correlation evidence retrieved.
- [x] Exact cap failure reproduced without effects.
- [x] Similar cap, timestamp, recurrence, continuation, and context failures
  scanned and reproduced where deterministic.
- [x] Root-cause review artifact authored.
- [x] Owner approves this plan.
- [x] Luna implementation handoff completed.
- [x] Exact deterministic verification completed.
- [x] One live-LLM case run and inspected.
- [x] Root diff/evidence review completed.
- [x] Plan closed and archived as separate quality work.
