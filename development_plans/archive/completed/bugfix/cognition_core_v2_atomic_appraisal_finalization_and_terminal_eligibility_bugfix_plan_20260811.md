# Cognition Core V2 atomic appraisal finalization and terminal eligibility bugfix plan 20260811

## Summary

- Goal: prevent the fatal active-event capacity failure reproduced by
  llmtrace_8d0d42952b76450c9e1dc32574f9fd44 and
  llmtrace_9164e957298e4cffb68db7911bcd28b1, while preserving the
  LLM-owned semantic appraisal decision and the existing fail-closed state
  invariants.
- Plan class: high-risk, evidence-gated Cognition V2 bugfix with no database
  migration and no public response-shape change.
- Status: completed.
- Acceptance state: this completed plan records the RCA agreement, the
  amendments from the previous maximum-reasoning gpt-5.6-sol review, and the
  integrated independent high-reasoning gpt-5.6-sol plan review. Production
  execution is authorized by the explicit user command recorded below.
- Fixed change direction:
  1. P0: admit cumulative appraisal prefixes only after the complete
     deterministic finalization transaction succeeds.
  2. P1: restrict terminal-outcome handles and delta paths to lifecycle-
     eligible native entities, while retaining evidence-bound current-turn
     candidates and strict reducer guards.
  3. P2: preserve a bounded validation-error descriptor in protected
     decontextualizer attempt evidence without changing its fallback policy.
- No alternate fix is open for selection. Raising caps, deleting active or
  affect-protected causes, weakening terminal guards, adding retries, and
  moving semantic terminalization into deterministic code are rejected.

### Evidence baseline

- Requested protected trace exports and RCA artifacts:
  - test_artifacts/diagnostics/llm_trace_llmtrace_8d0d42952b76450c9e1dc32574f9fd44_20260811T010259Z.json
  - test_artifacts/diagnostics/llm_trace_llmtrace_8d0d42952b76450c9e1dc32574f9fd44_20260811T010259Z_rca.md
  - test_artifacts/diagnostics/llm_trace_llmtrace_9164e957298e4cffb68db7911bcd28b1_20260811T011003Z.json
  - test_artifacts/diagnostics/llm_trace_llmtrace_9164e957298e4cffb68db7911bcd28b1_20260811T011003Z_rca.md
- Consolidated RCA and proposal:
  test_artifacts/diagnostics/cognition_failure_mode_consolidated_20260811_rca_and_fix_proposal.md
- Previous independent review:
  test_artifacts/diagnostics/cognition_failure_mode_consolidated_20260811_gpt56_sol_review.md
- Adjacent-failure corpus:
  test_artifacts/diagnostics/cognition_failure_capsules_today_20260811_full.json
- The corpus contains 80 capsules and 101 structured failure events. It is
  supporting evidence for recurrence and saturation, not a fatal-exception
  incidence denominator: it predates the two requested traces and contains no
  terminal active_events-capacity capsule.

## Scope And Change Direction

### Consolidated failure mode

Both requested runs enter semantic appraisal with 32 of 32 active-event slots
occupied and all entry events active. A valid current-turn event-agency
appraisal can materialize one net-new causal event after compatible-role
matching, producing a 32-to-33 candidate state. The current
_reduce_appraisals_with_isolation function validates the semantic reduction
without running the later finalization sequence. The caller then runs
apply_state_update, affect derivation protects the relevant causes, and
prune_terminal_entities raises:

    CognitionStateError: active_events capacity is protected by active causes

The exception escapes the bounded appraisal degradation boundary, so both runs
finish with zero dialog output. Trace 1 has two accepted event-agency
assertions but only one net-new root, so its resulting count is 33, not 34.

The same traces also show a secondary upstream contract family. The
goal/outcome question can target an already-resolved knowledge gap, emitting
knowledge_answered together with an unowned knowledge_gaps.k1.uncertainty
delta. The bounded repair removes the unowned numeric delta but leaves the
terminal proposition, and the strict reducer rejects the second terminal
transition. The adjacent corpus contains 33 distinct traces with the resolved
knowledge-gap transition cause and 69 of 80 entry states at the active-event
cap.

Trace 2 also contains three decontextualizer contract-error attempts before
the documented normalized-original-input fallback. Trace 1 reaches the same
active-event capacity failure without that upstream exhaustion, establishing
that decontextualizer exhaustion is independent of the fatal Cognition V2
cause.

### Root-cause decisions

1. Primary root cause: missing capacity admission at the semantic appraisal
   boundary. Optional appraisal output is accepted before the canonical
   affect-derivation, retention, deterministic-goal, and final-state
   validation sequence proves that the resulting state is retainable.
2. Secondary root cause: terminal-outcome source planning offers stale
   terminal native entities too broadly. Handle filtering alone is
   insufficient; the deterministic delta allowlist must use the same
   lifecycle eligibility.
3. Independent observability defect: decontextualizer contract failures do not
   pass a bounded validation-error descriptor into the protected attempt
   evidence, making the already-bounded fallback harder to diagnose.

### P0: atomic cumulative appraisal finalization

Owner: src/kazusa_ai_chatbot/cognition_core_v2/facade.py,
_reduce_appraisals_with_isolation and its local finalization path.

For every cumulative accepted-prefix plus current appraisal, trial from the
original validated preliminary state in this exact order:

1. apply_semantic_appraisals;
2. _semantic_relief_transitions using the original preliminary state, the
   candidate reduced state, the candidate appraisal prefix, evidence, and
   handle bindings;
3. apply_state_update, including affect derivation and canonical terminal
   retention;
4. create_deterministic_goals;
5. validate_cognition_state.

The transaction must receive the same updated_at, character constraints, and
relationship context as the current finalization path. A candidate appraisal
is admitted only when the entire transaction succeeds. The returned state is
already finalized; the caller must not finalize it a second time.

The reduction boundary has these fixed semantics:

- Start with the finalized zero-appraisal prefix. If no appraisal is
  available, this is the single canonical finalization path.
- Recompute each non-empty cumulative prefix from the original preliminary
  state so accepted appraisals retain cross-question composition.
- On a CognitionStateError from semantic reduction, retention, affect
  derivation, deterministic goal creation, or final validation, reject the
  complete current appraisal-family result with the existing
  semantic_appraisal_reduction_rejected disposition.
- Preserve the last successfully finalized accepted prefix.
- Commit no comparison rows from a rejected current result. Preserve
  comparison rows from accepted results as audit evidence even if canonical
  retention later removes a terminal entity.
- Record the exact deterministic exception text and the finalization step in
  the bounded failure evidence.
- Return the last finalized state and do not run the old post-reduction
  apply_state_update, create_deterministic_goals, and validate_cognition_state
  sequence again.

This is question-atomic degradation. It does not introduce an ephemeral
semantic-appraisal carrier, persistence exception, additional retry, new LLM
judge, cap increase, event deletion, or replacement event.

### P1: lifecycle-eligible terminal-outcome planning

Owner: src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py.

For q:goal_threat_outcome, filter both existing native handles and existing
native delta paths by the authoritative lifecycle status:

- goals: pursuing or blocked;
- threats: active;
- active events: active;
- knowledge gaps: open or reduced.

Keep current-turn candidate handles ce, ct, and ck when their evidence binding
is valid. Keep knowledge_answered as the terminal-outcome proposition, but keep
numeric knowledge-gap deltas owned by q:epistemic_comparison_memory. Do not add
knowledge_gaps.*.uncertainty to q:goal_threat_outcome. Preserve the existing
unowned-path validator and strict terminal-transition guards because the
generic handle contract cannot express that a blocked goal may be a valid
subject but an invalid goal_supersession object.

This is deterministic source eligibility validation. The LLM remains the
semantic owner of whether current evidence supports a terminal proposition for
an eligible entity.

### P2: bounded decontextualizer validation evidence

Owner: src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py.

When a decontextualizer attempt is rejected by structural or semantic contract
validation, pass the existing bounded exception text through the existing
llm_tracing.record_llm_trace_step validation_error field. The destination is
the existing protected failure-capsule field
cognition_failure_capsule.attempts[].validation_error, bounded to 500
characters. The Stage 0 decontextualizer opens and closes its own bounded
failure-capsule session around the attempt sequence when the turn carries a
trace id, so the existing recorder has an active protected destination before
the first contract-error attempt. The persisted llm_trace_steps document
remains unchanged and does not gain a validation_error field. Keep the raw
prompt/output capture policy, attempt cap, repair prompt,
normalized-original-input fallback, and public output unchanged. Provider
failures retain their current provider-error disposition; this work adds
diagnostic detail to contract-error attempts only.

## Mandatory Skills

- development-plan: lifecycle status, execution gates, roles, and exact
  source-to-test traceability.
- llm-trace-debug: protected trace evidence, deterministic replay, and
  correlation artifact handling.
- debug-llm: human-readable evidence artifacts and human inspection of any
  live model result.
- local-llm-architecture: LLM semantic ownership, deterministic state-safety
  boundaries, bounded latency, and rejection of speculative retries or
  compatibility layers.
- py-style: every Python source or test edit during approved execution.
- test-style-and-execution: deterministic reducer/planner tests, patched
  orchestration checks, and one-at-a-time live LLM execution.

## Mandatory Rules

- This plan is planning authority for the scoped implementation. Production
  implementation requires approved or in_progress status and an explicit user
  command to implement the fix.
- Capture git status, the relevant README/HOWTO guidance, source files, direct
  tests, and the ownership manifest before implementation. Preserve unrelated
  user changes.
- Use venv\Scripts\python.exe for Python and pytest commands. Use apply_patch
  for manual edits. Keep .env outside inspection.
- Keep the existing LLM attempt caps and JSON parsing entry point. Every raw
  LLM response continues through
  kazusa_ai_chatbot.utils.parse_llm_json_output before semantic evaluation.
- LLM stages retain semantic judgment. Deterministic code owns state
  validation, lifecycle eligibility, retention, caps, permissions, persistence,
  trace bounds, and fallback execution.
- Keep the live response path bounded and inspectable. The P0 transaction is
  deterministic and adds no model call or retry.
- Do not add compatibility shims, alias vocabularies, fallback mappers,
  keyword routing, or parallel contract layers.
- Do not raise active-event or any other list cap. Do not delete active or
  affect-protected causes to force admission. Do not weaken terminal
  immutability or transition guards.
- Do not move RAG evidence into persona or stance ownership. Do not alter
  dialog wording, dialog policy, adapter delivery, persistence, scheduler,
  consolidation, or reflection behavior.
- Do not introduce a new semantic-appraisal persistence carrier or an
  evaluator/agent/retry layer.
- Keep P2 bounded and diagnostic-only; never expose raw prompts, model output,
  credentials, platform identifiers, or trace identifiers through the public
  response.

## Must Do

1. Record the current worktree and changed-path baseline, the explicitly owned
   file set, current source-test manifest, current cap and lifecycle contracts,
   and the two protected trace artifact paths. Preserve pre-existing changes.
2. Add deterministic red tests for the P0 admission contract before changing
   production behavior. Cover:
   - a 32-to-33 active-event candidate rejected inside full finalization;
   - a terminal entity that becomes affect-protected during finalization;
   - a genuinely removable terminal entity retained by canonical pruning;
   - a candidate-induced deterministic-goal capacity failure;
   - active/protected rows preserved without deletion;
   - accepted-prefix and comparison-row preservation;
   - exactly-once finalization and no second post-reduction finalization.
3. Implement P0 with one canonical atomic transaction path in facade.py and
   update the caller to consume its finalized state directly.
4. Add deterministic planner tests proving lifecycle filtering applies to
   both q:goal_threat_outcome handles and delta paths, while eligible native
   rows, current-turn candidates, and knowledge_answered remain available.
5. Implement P1 without changing the generic handle schema or reducer guard.
6. Add the bounded P2 validation_error handoff and deterministic trace-capture
   coverage. Preserve all existing fallback and repair behavior.
7. Update tests/ownership/source_test_impact_manifest.json with every new exact
   required unit node and supplemental node. Run the impact verifier before
   broad tests.
8. Replay both protected traces through the patched deterministic path, one
   case at a time, and author a human-readable review artifact from the raw
   replay evidence. Require active_events <= 32, final cognition output,
   bounded appraisal rejection, and no terminal capacity capsule.
9. Run any live confirmation one case at a time. Inspect the emitted artifact
   and character-quality behavior before starting the next case. Treat live
   results as supplemental behavior evidence, not as a replacement for
   deterministic state-invariant tests.

## Deferred

- Any increase or redesign of STATE_LIST_CAPS, historical-state migration, or
  retention policy.
- Any ephemeral semantic-appraisal carrier or persistence of rejected
  semantic candidates.
- Any new LLM retry, judge, agent, model route, or prompt rewrite.
- Any redesign of the generic handle contract to encode subject/object
  lifecycle asymmetry.
- Any deterministic semantic terminalization, TTL deletion, or heuristic
  keyword routing.
- Any decontextualizer fallback-policy, prompt, role-projection, or public
  response change.
- Any dialog, adapter, queue, persistence, scheduler, consolidation, or
  reflection change.

## Target State

After the approved implementation:

- Every accepted appraisal prefix has passed semantic reduction, relief
  transition derivation, state update with affect derivation and retention,
  deterministic goal creation, and final state validation.
- The final state returned by appraisal reduction is already finalized and is
  not finalized again by the caller.
- An unretainable optional appraisal is rejected with
  semantic_appraisal_reduction_rejected, its current comparison rows are
  discarded, and its deterministic rejection evidence contains question_id,
  failure_code, finalization_step, and exception text bounded to the
  diagnostic limit. The accepted prefix remains usable.
- All final and intermediate accepted states satisfy every list cap, including
  active_events <= 32, without deleting active or protected causes.
- Accepted comparison rows remain available as diagnostic evidence even when
  canonical retention removes a terminal entity.
- q:goal_threat_outcome exposes only lifecycle-eligible existing handles and
  paths; valid evidence-bound current-turn candidates remain available.
- q:goal_threat_outcome has no numeric knowledge-gap delta path.
  knowledge_answered remains the proposition-only terminal operation.
- A stale terminal proposition cannot escape as a full-run fatal exception; it
  remains bounded by the producing contract or isolated reducer disposition.
- Decontextualizer contract-error attempts carry a bounded validation_error in
  protected evidence at cognition_failure_capsule.attempts[].validation_error,
  while the persisted llm_trace_steps schema and three-attempt
  repair/fallback behavior remain unchanged.
- Normal under-cap candidate materialization, cross-appraisal composition,
  terminal pruning, strict transition guards, final branches, and dialog
  ownership remain unchanged.
- No additional LLM call, retry, schema field, adapter surface, or unrelated
  database/persistence behavior is introduced. P2 may use the existing
  protected failure-capsule writer for Stage 0 contract-error attempts; it
  adds no new persistence schema or storage path.

## Execution Roles

### plan_owner

- Responsibility: Maintain this plan, evidence links, lifecycle status, scope,
  and review integration.
- Owned surface: This plan, the development-plan registry row, and
  human-readable diagnostic/review artifacts only.
- Authority: May edit planning and diagnostic artifacts. May not edit
  production code under this draft or approve implementation.
- Applicable skills: development-plan, llm-trace-debug, debug-llm, and
  local-llm-architecture.
- Capability floor: Repository-level system reasoning, protected-trace
  evidence handling, plan-contract literacy, and ability to inspect exact
  pytest ownership.
- Independence requirement: Must remain separate from the independent plan
  reviewer and independent code reviewer.
- Acceptance output: Closed plan, integrated review artifact, status update,
  and explicit execution handoff package.
- Gate: Entry requires the RCA evidence and repository contracts to be read.
  Exit requires the independent plan review to be recorded and all amendments
  integrated before the plan transitions from draft to in_progress.

### cognition_reducer_implementer

- Responsibility: Implement P0 atomic appraisal admission and its deterministic
  owner tests.
- Owned surface: src/kazusa_ai_chatbot/cognition_core_v2/facade.py,
  tests/unit/cognition_core_v2/test_facade.py, the named semantic
  terminalization/trace replay tests, and their manifest rows.
- Authority: May change facade admission orchestration and tests after the
  plan is approved or in_progress and the user explicitly commands
  implementation. May not change semantic appraisal meaning, caps, state
  deletion, dialog behavior, or unowned files.
- Applicable skills: development-plan, py-style, test-style-and-execution,
  llm-trace-debug, debug-llm, and local-llm-architecture.
- Capability floor: Python/Pytest competence, Cognition V2 reducer and
  lifecycle knowledge, ability to preserve bounded exception ownership, and
  ability to run exact-node collection and deterministic replay.
- Independence requirement: Must be separate from the final code reviewer and
  may not self-approve remediation.
- Acceptance output: Scoped diff, exact unit/integration tests, two replay
  artifacts, cap/finalization invariants, and bounded rejection evidence.
- Gate: Entry requires approved/in_progress status, explicit user
  implementation authorization, baseline ownership snapshot, and applicable
  skills. Exit requires all P0 nodes and replay gates passing with no
  unrelated changes.

### semantic_source_planner_implementer

- Responsibility: Implement P1 lifecycle filtering for terminal-outcome
  handles and delta paths.
- Owned surface: src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py,
  tests/unit/cognition_core_v2/test_semantic_source_planner.py, the named
  planner/transition tests, and their manifest rows.
- Authority: May change deterministic source eligibility and its tests after
  the same approval and user-authorization gate. May not change the generic
  handle schema, LLM semantic ownership, or strict reducer guards.
- Applicable skills: development-plan, py-style, test-style-and-execution,
  debug-llm, and local-llm-architecture.
- Capability floor: Python/Pytest competence, knowledge of native lifecycle
  statuses and evidence-bound candidate handles, and ability to prove both
  handle and path projections.
- Independence requirement: Must be separate from the final code reviewer.
  Coordination with the P0 implementer is required for the shared manifest
  and adjacent test collection.
- Acceptance output: Scoped diff, exact planner tests, eligible/terminal
  projection evidence, and preserved candidate coverage.
- Gate: Entry requires the approved/in_progress plan, explicit user
  authorization, baseline ownership snapshot, and applicable skills. Exit
  requires exact collection and all P1 regression nodes passing.

### trace_observability_implementer

- Responsibility: Implement P2 bounded validation-error capture while
  preserving the decontextualizer fallback contract.
- Owned surface: src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py,
  tests/test_msg_decontextualizer.py, and the related manifest row.
- Authority: May change the protected attempt-recording call and deterministic
  tests after the same approval and user-authorization gate. May not change
  raw capture policy, repair prompts, attempt caps, fallback behavior, or
  public output.
- Applicable skills: development-plan, py-style, test-style-and-execution,
  debug-llm, and local-llm-architecture.
- Capability floor: Python/Pytest competence, knowledge of the existing
  llm_tracing validation_error contract, and ability to assert bounded
  protected evidence without exposing raw model data.
- Independence requirement: Must be separate from the final code reviewer.
- Acceptance output: Scoped diff, a 500-character-bound test, unchanged
  fallback evidence, and the protected failure-capsule destination proof.
- Gate: Entry requires the approved/in_progress plan, explicit user
  authorization, baseline ownership snapshot, and applicable skills. Exit
  requires exact decontextualizer nodes passing and no llm_trace_steps schema
  change.

### independent_plan_reviewer

- Responsibility: Review the completed draft against the RCA, previous review,
  source contracts, test matrix, and execution gates.
- Owned surface: Read-only access to the plan, registry row, evidence
  artifacts, source, tests, and referenced skills.
- Authority: May issue PASS, PASS WITH AMENDMENTS, or FAIL and require plan
  remediation. May not edit files, change plan status, implement code, or
  approve its own amendments.
- Applicable skills: development-plan, llm-trace-debug, debug-llm, and
  local-llm-architecture.
- Capability floor: Independent system-level review, exact pytest/test-impact
  contract inspection, and ability to evaluate semantic/deterministic
  ownership boundaries.
- Independence requirement: Must be a separate executor from plan_owner and
  every implementation role.
- Acceptance output: Written verdict with evidence, exact amendments, root
  ownership decision, and draft-status recommendation.
- Gate: Entry requires the completed draft and prior review artifact. This
  plan-scoped fixed execution constraint is model gpt-5.6-sol, reasoning high,
  normal/default service speed. Only the user or an explicitly approving plan
  authority may change that fixed constraint. Exit requires the verdict to be
  recorded and any required amendments integrated by plan_owner.

### independent_code_reviewer

- Responsibility: Review the implemented diff, exact test collection,
  deterministic replays, and any live evidence independently of implementers.
- Owned surface: Read-only access to the final changed paths, artifacts, and
  acceptance evidence.
- Authority: May accept or return the implementation for remediation. May not
  edit the implementation or self-approve its own remediation.
- Applicable skills: development-plan, py-style, test-style-and-execution,
  llm-trace-debug, debug-llm, and local-llm-architecture.
- Capability floor: Python/reducer review, source-impact verification,
  protected-trace interpretation, and residual-risk assessment.
- Independence requirement: Must be separate from all implementation roles;
  remediation requires a new independent review.
- Acceptance output: Review verdict tied to every acceptance criterion and a
  residual-risk statement.
- Gate: Entry requires implementation evidence, complete exact-node
  collection, and replay artifacts. Exit requires no unresolved high-severity
  finding and a recorded acceptance or remediation handoff.

Agent selection for implementation and code review remains runtime-owned. The
fixed gpt-5.6-sol constraint applies to this plan review only.

## Test Impact And Traceability

The following matrix is the closed source-to-test contract for the planned
change. New test names are exact required node IDs to be created during
implementation; existing nodes remain required regressions.

| Repo-relative path | Changed symbol, contract, and semantic owner | Exact deterministic pytest node IDs | Supplemental/live node IDs or none | Test mode | Observable regression prevented |
| --- | --- | --- | --- | --- | --- |
| src/kazusa_ai_chatbot/cognition_core_v2/facade.py | Modify _reduce_appraisals_with_isolation and the final-reduction caller; semantic owner: cognition_core_v2 facade | tests/unit/cognition_core_v2/test_facade.py::test_facade_exposes_owned_contract; tests/unit/cognition_core_v2/test_facade.py::test_reduction_rejects_over_cap_candidate_during_full_finalization; tests/unit/cognition_core_v2/test_facade.py::test_reduction_rejects_affect_protected_terminal_during_finalization; tests/unit/cognition_core_v2/test_facade.py::test_reduction_accepts_candidate_when_terminal_row_is_removable; tests/unit/cognition_core_v2/test_facade.py::test_reduction_rejects_goal_capacity_during_finalization; tests/unit/cognition_core_v2/test_facade.py::test_reduction_preserves_accepted_prefix_and_comparison_rows; tests/unit/cognition_core_v2/test_facade.py::test_reduction_records_bounded_rejection_evidence; tests/unit/cognition_core_v2/test_facade.py::test_reduction_runs_finalization_once_per_admitted_prefix | tests/test_cognition_core_v2_semantic_terminalization.py::test_appraisal_retries_a_reducer_incompatible_candidate; tests/test_cognition_core_v2_semantic_terminalization.py::test_final_reduction_isolates_one_residual_invalid_appraisal; tests/test_cognition_core_v2_semantic_terminalization.py::test_final_reduction_preserves_cross_appraisal_composition; tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_captured_trace_8d0d4295_stays_within_active_event_cap; tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_captured_trace_9164e957_stays_within_active_event_cap; live: tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_captured_trace_8d0d4295_capacity_path_live_llm; tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_captured_trace_9164e957_capacity_path_live_llm | Deterministic unit, deterministic integration/replay, and one-at-a-time real LLM supplemental | Prevents optional appraisal state from escaping admission with 33 active events, a later goal-capacity failure, or missing rejection evidence. |
| src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py | Modify q:goal_threat_outcome lifecycle filtering for handles and delta paths; semantic owner: semantic source planner | tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_semantic_source_planner_exposes_owned_contract; tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_moral_identity_questions_exclude_standard_handles; tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_goal_outcome_filters_terminal_handles_and_delta_paths; tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_goal_outcome_keeps_eligible_handles_and_candidates | tests/test_cognition_core_v2_alignment_gates.py::test_scheduler_evidence_selects_only_goal_threat_outcome; tests/test_cognition_core_v2_alignment_gates.py::test_each_question_receives_only_family_local_handles_and_state; tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_resolved_knowledge_gap_transition_is_rejected; tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_terminal_event_transition_is_rejected; live: none | Deterministic unit and deterministic integration | Prevents stale terminal subjects and unowned numeric paths from repeatedly entering the goal/outcome appraisal contract. |
| src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py | Modify protected attempt recording to pass bounded validation_error for contract errors; semantic owner: message decontextualizer trace boundary | tests/test_msg_decontextualizer.py::test_decontextualizer_prompt_explains_reply_ellipsis_decision_owner; tests/test_msg_decontextualizer.py::test_decontextualizer_trace_records_bounded_validation_error; tests/test_msg_decontextualizer.py::test_decontextualizer_fallback_on_malformed_json; tests/test_msg_decontextualizer.py::test_decontextualizer_recovers_on_third_contract_attempt | tests/test_decontextualizer_live_llm.py::test_live_decontextualizer_reward_offer_preserves_user_actor_character_target | Deterministic unit/contract and one-at-a-time real LLM supplemental | Prevents metadata-only contract exhaustion from losing the bounded reason needed for RCA, without changing fallback or role ownership. |
| tests/ownership/source_test_impact_manifest.json | Modify required and supplemental node inventory; semantic owner: test-ownership governance | tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary; tests/test_test_impact_manifest.py::test_required_node_collection_failure_is_reported | none | Deterministic manifest/collection verifier | Prevents changed source from bypassing its exact deterministic owner tests or leaving stale node IDs. |

Keep unchanged and regression-tested rather than modify:
src/kazusa_ai_chatbot/cognition_core_v2/state_models.py and
src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py. Their caps,
canonical retention, strict terminal guards, and final validation remain the
deterministic authority. Required existing nodes include
tests/test_cognition_core_v2_state.py::test_pruning_removes_old_terminal_entities_but_protects_active_roots,
tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_semantic_delta_path_not_owned_is_rejected,
and tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_resolved_knowledge_gap_transition_is_rejected.

## Change Surface

### Modify after approval

- src/kazusa_ai_chatbot/cognition_core_v2/facade.py
- src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py
- src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py
- tests/unit/cognition_core_v2/test_facade.py
- tests/unit/cognition_core_v2/test_semantic_source_planner.py
- tests/test_cognition_core_v2_semantic_terminalization.py
- tests/test_cognition_core_v2_trace_failure_mode_matrix.py
- tests/test_msg_decontextualizer.py
- tests/test_cognition_core_v2_trace_failure_modes_live_llm.py
- tests/ownership/source_test_impact_manifest.json

### Create as diagnostic artifacts

- A deterministic replay evidence JSON or JSONL artifact for each protected
  trace, containing input identity, captured appraisal candidates, accepted
  prefix, rejection cause, state counts, final output status, and terminal
  capsule status.
- A human-readable replay review Markdown artifact authored from those raw
  results. Scripts may emit raw/structured evidence only; they must not author
  the human judgment report.
- A live one-case-at-a-time trace artifact only when the live gate is run.

### Keep unchanged

- state_models.py caps and prune semantics.
- state_reducers.py transition, affect, and goal semantics.
- LLM prompts, model routing, attempt counts, JSON parser ownership, dialog,
  adapter, queue, persistence, scheduler, consolidation, and reflection
  surfaces.

## Agent Autonomy Boundaries

- The plan reviewer may inspect all named evidence, source, tests, and plan
  content and may recommend amendments. It may not edit files, run production
  changes, change the plan status, or approve its own remediation.
- The implementation roles may edit only their explicitly owned files after
  both execution gates are satisfied: plan status approved/in_progress and an
  explicit user implementation command.
- The P0 implementer owns admission orchestration, not semantic appraisal
  meaning, cap policy, event deletion, or dialog behavior.
- The P1 implementer owns deterministic source eligibility, not LLM
  terminalization judgment or a new generic handle contract.
- The P2 implementer owns bounded trace evidence, not raw capture policy,
  fallback policy, prompt semantics, or public output.
- The code reviewer must inspect the final diff, exact test collection, replay
  artifacts, and any live result independently. A remediation author cannot
  be the sole final approver.

## Verification

### Pre-implementation gate

- Confirm git status and preserve unrelated work.
- The pre-implementation gate required the plan to remain draft until the
  requested independent review was integrated and the user explicitly
  authorized implementation; that gate is recorded as complete above.
- Confirm the two protected traces, RCA, prior review, and adjacent corpus are
  readable without inspecting .env.
- Confirm the current ownership manifest validates before changing it.

### Implementation sequence

1. Add the deterministic red tests and capture the expected pre-fix failure.
2. Implement P0 and run its exact unit nodes plus the existing semantic
   terminalization and state regression nodes.
3. Implement P1 and run its exact planner, alignment-gate, and strict
   transition nodes.
4. Implement P2 and run its exact decontextualizer contract/fallback nodes.
5. Update and validate the ownership manifest.
6. Run the impacted collection through:
   venv\Scripts\python.exe -m scripts.validate_test_impact --base-ref HEAD --run
7. Run the adjacent deterministic suite:
   venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_semantic_terminalization.py tests/test_cognition_core_v2_trace_failure_mode_matrix.py tests/test_cognition_core_v2_alignment_gates.py tests/test_cognition_core_v2_state.py tests/test_msg_decontextualizer.py -q
8. Replay trace 8d0d4295 and inspect its artifact. Then replay trace 9164e957
   and inspect its artifact. The order is fixed and the cases are separate.
9. If live confirmation is enabled, run each exact live node separately with
   -q -s, inspect the durable trace and quality notes, and only then run the
   next node.

### Verification evidence required for handoff

- Before/after test status with exact node IDs.
- Source-impact verifier output showing complete collection.
- Both deterministic replay artifacts and a human-readable review.
- State counts proving active_events <= 32 and no deletion of active or
  affect-protected causes.
- Failure capsules showing bounded appraisal rejection rather than terminal
  capacity failure.
- Rejection evidence proving question_id, failure_code,
  finalization_step, and bounded exception text for every P0 admission
  failure.
- P1 planner output showing terminal rows absent, eligible rows present, and
  candidate handles preserved.
- P2 trace evidence showing validation_error bounded to the existing diagnostic
  limit and fallback behavior unchanged.
- Live artifacts and human inspection notes, or an explicit environment
  blocker recorded without treating deterministic success as live quality
  proof.

## Acceptance Criteria

The plan implementation is acceptable only when all conditions hold:

1. Both protected trace replays reach final Cognition V2 output with no
   terminal active_events-capacity exception and no active_events count above
   32.
2. A candidate that cannot be retained is rejected atomically at appraisal
   admission, with its family unavailable to later branches, accepted-prefix
   state preserved, and no rejected comparison rows committed.
3. Every bounded P0 rejection records question_id, failure_code,
   finalization_step, and exception text capped at the diagnostic limit.
4. Full finalization runs exactly once for the returned reduction result; no
   second caller-side finalization recreates the original failure.
5. Affect-protected terminal rows and active causal roots are preserved.
   Genuinely removable terminal rows are pruned only by existing canonical
   retention.
6. Deterministic goal-capacity failures are contained by the same appraisal
   transaction and do not escape downstream.
7. q:goal_threat_outcome no longer offers resolved/failed/abandoned/replaced/
   terminal native rows or their delta paths, while pursuing/blocked goals,
   active threats/events, open/reduced gaps, and valid current-turn candidates
   remain available.
8. Numeric knowledge-gap deltas remain epistemic-owned and
   knowledge_answered remains the only goal/outcome terminal proposition.
9. Existing strict terminal-transition and unowned-path tests pass.
10. Decontextualizer contract-error attempt evidence carries a bounded
   validation_error, and the documented repair/fallback behavior and public
   output remain unchanged.
11. No additional model call, retry, cap, schema, adapter, dialog, scheduler,
    consolidation, or reflection behavior is introduced, and no unrelated
    database/persistence behavior is added. P2 may write the existing
    protected failure-capsule document for bounded Stage 0 contract-error
    attempts; it adds no new field or storage path.
12. The source-test impact manifest validates and all newly named exact nodes
    collect and pass.
13. Independent code review accepts the implementation and evidence with no
    unresolved high-severity residual risk.

## Independent Plan Review

- Requested executor: gpt-5.6-sol.
- Requested settings: reasoning high; normal/default service speed; read-only
  review.
- Review question: does this concrete plan preserve the previous
  gpt-5.6-sol amendments, fit the current Cognition V2 ownership boundaries,
  name executable source/test surfaces, and address the fatal root rather than
  merely moving the exception?
- Review status: completed; verdict PASS WITH AMENDMENTS. The reviewer
  confirmed that P0 addresses the fatal root at the facade admission boundary
  and that P1 remains correctly deterministic. The required amendments below
  are integrated into this draft.
- Integrated amendments:
  1. Execution roles now declare responsibility, owned surface, authority,
     applicable skills, capability floor, independence, acceptance output, and
     entry/exit gates. Only the user or explicitly approving plan authority may
     change the fixed review executor constraint.
  2. The impact matrix now contains a test-mode column and direct facade-unit
     nodes for affect-protected retention, successful removable-terminal
     retention, goal-capacity rejection, and bounded rejection evidence.
  3. P2 now names cognition_failure_capsule.attempts[].validation_error as the
     destination, caps it at 500 characters, and explicitly excludes any
     llm_trace_steps schema change.
  4. P0 acceptance now requires question_id, failure_code,
     finalization_step, and bounded exception text.
  5. Baseline wording now requires the current worktree/changed-path snapshot
     and explicitly owned file set, preserving pre-existing changes.
- Review artifact:
  test_artifacts/diagnostics/cognition_failure_mode_plan_20260811_gpt56_sol_review.md
- Plan status is now in_progress after explicit user implementation
  authorization. Completion still requires every acceptance criterion and
  independent code review.

## Execution Evidence

### 2026-08-11 execution start

- Lifecycle transition: draft to in_progress.
- Authorization: explicit user command to execute this reviewed plan.
- Pre-edit worktree baseline:
  - modified: development_plans/README.md
  - untracked: this plan file
  - no production source or test changes present.
- Explicit initial owned file set:
  - src/kazusa_ai_chatbot/cognition_core_v2/facade.py
  - src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py
  - src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py
  - tests/unit/cognition_core_v2/test_facade.py
  - tests/unit/cognition_core_v2/test_semantic_source_planner.py
  - tests/test_cognition_core_v2_semantic_terminalization.py
  - tests/test_cognition_core_v2_trace_failure_mode_matrix.py
  - tests/test_msg_decontextualizer.py
  - tests/test_cognition_core_v2_trace_failure_modes_live_llm.py
  - tests/ownership/source_test_impact_manifest.json
- Explicitly preserved outside the implementation diff:
  - development_plans/README.md and this plan/registry work;
  - test_artifacts/diagnostics evidence and review artifacts.
- Current verification: development-plan contract checks passed 4/4 before
  execution. Production source and mapped implementation tests are not yet
  changed or accepted.
- Next checkpoint: complete the P0 facade slice and its direct deterministic
  owner tests, then compare the worktree against this baseline.

### 2026-08-11 implementation and deterministic verification checkpoint

- Production slice was implemented by the delegated gpt-5.6-luna agent within
  the explicitly owned three-file production scope. The parent retained
  ownership of all fixes, tests, evidence, reviews, and close-off.
- P0 now finalizes every cumulative appraisal prefix atomically from the
  original preliminary state, preserves the last accepted finalized prefix,
  records bounded rejection evidence, and returns an already-finalized state
  to the caller without a second finalization sequence.
- P1 now applies lifecycle eligibility to both native goal/outcome handles and
  native delta paths while retaining valid current-turn candidates and
  knowledge_answered proposition ownership.
- P2 now forwards bounded contract validation detail into the existing
  protected attempt evidence and establishes the Stage 0 capsule session
  before recording attempts, without changing the trace-step schema, attempt
  cap, repair policy, fallback, or public output.
- Parent-owned exact P0, P1, and P2 tests pass. The source-impact verifier
  collected and passed all 16 required exact nodes.
- Adjacent deterministic suite result: 109 passed, 2 skipped. The two skips are
  the day-wide inventory-dependent supplemental matrix nodes because that
  inventory is not present in the workspace.
- The source-impact verifier collected and passed all 17 required exact
  nodes after the Stage 0 capsule-boundary correction.
- Protected deterministic replays were run separately in the required order:
  `test_captured_trace_8d0d4295_stays_within_active_event_cap` and
  `test_captured_trace_9164e957_stays_within_active_event_cap`; both passed.
  Each replay holds active events at `32 -> 32`, rejects the candidate with
  `semantic_appraisal_reduction_rejected` at `apply_state_update`, records
  bounded evidence, returns validated `cognition_core_output.v2` through the
  public facade, and produces a real `partial_failure` capsule with no
  terminal failure. Raw and human-readable evidence are recorded in
  `test_artifacts/diagnostics/cognition_failure_replay_8d0d4295.json`,
  `test_artifacts/diagnostics/cognition_failure_replay_9164e957.json`, and
  `test_artifacts/diagnostics/cognition_failure_replay_review_20260811.md`.
- Supplemental live capacity-replay nodes remain explicitly gated and skipped
  pending an enabled live endpoint/model run. This is recorded as a live
  quality-evidence limitation, not as deterministic state-safety evidence.
- Independent code review and final gpt-5.6-sol sign-off remain required
  before lifecycle completion.

### 2026-08-11 native code review amendment

- The project-native gpt-5.6-terra reviewer returned `PASS WITH AMENDMENTS`.
  P0, P1, P2, scope, and deterministic ownership all passed. The reviewer
  identified one high-severity evidence gap: the initial replay harness called
  only the private reducer and used literal artifact status values instead of
  proving the public final output and actual failure-capsule disposition.
- The parent corrected the replay harness in
  `tests/test_cognition_core_v2_trace_failure_mode_matrix.py`. Each protected
  candidate now enters the real `facade.run_cognition` wrapper; only branch
  execution and action planning are deterministic no-op/silence stubs. The
  test captures the actual protected capsule document, asserts validated
  `cognition_core_output.v2`, asserts `partial_failure` rather than terminal
  failure, preserves active-event IDs, and extracts the real bounded
  rejection event.
- Both corrected replay nodes pass. The regenerated artifacts now use
  `cognition_failure_replay.v2`, report
  `replay_mode=captured_candidate_through_run_cognition`,
  `final_output_status=cognition_core_output.v2`,
  `failure_capsule_outcome=partial_failure`, and
  `terminal_capsule_status=none` for both traces.
- The amendment is documented in
  `test_artifacts/diagnostics/cognition_failure_replay_review_20260811.md`.
  A fresh independent gpt-5.6-sol final sign-off remains required.

### 2026-08-11 final-signoff P2 correction

- The first final gpt-5.6-sol review returned `FAIL` after identifying that
  Stage 0 ran before any outer Cognition V2 capsule session, so the existing
  recorder discarded the forwarded validation error. The review also required
  a real-capsule owner test rather than a recorder mock assertion.
- The parent corrected the existing decontextualizer owner by opening a
  protected `message_decontextualizer` capsule session before its attempt
  loop and finishing it after fallback/success. Clean runs discard the
  session; contract-error attempts persist the existing bounded
  `attempts[].validation_error` field. No trace-step schema field was added.
- The parent added
  `test_decontextualizer_actual_capsule_captures_validation_error`, which
  exercises the actual recorder and asserts three captured contract attempts,
  non-empty errors capped at 500 characters, and a non-terminal partial
  capsule outcome. The manifest now names this exact node.
- A fresh independent gpt-5.6-sol review is required after this correction.

### 2026-08-11 final gpt-5.6-sol sign-off

- Fresh reviewer: gpt-5.6-sol, high reasoning, normal/default service speed,
  independent read-only review.
- Verdict: `PASS WITH AMENDMENTS`; no blocking or high-severity issue remains.
- The reviewer accepted all 13 criteria, including the real Stage 0 capsule
  persistence test, the corrected full-core replays, P0 root-cause
  containment, P1 lifecycle ownership, and the explicitly gated live limit.
- The parent applied the required record-only amendments: corrected the plan
  and replay review counts to 17 exact nodes and 109 passed with two
  inventory skips, updated the manifest contract description, and clarified
  acceptance criterion 11's allowance for the existing protected capsule
  write.
- Final sign-off artifact:
  `test_artifacts/diagnostics/cognition_failure_mode_final_gpt56_sol_signoff_20260811.md`.
- The plan is ready for parent-owned lifecycle close-off.

### 2026-08-11 lifecycle closeout

- Parent-owned implementation and review remediation are complete. The final
  gpt-5.6-sol sign-off is `PASS WITH AMENDMENTS` with no blocking or
  high-severity finding; all record-only amendments were applied.
- Verification evidence is current: 17/17 required impact nodes passed,
  109 adjacent tests passed with two day-wide trace-inventory skips, the
  manifest suite passed 10/10, changed Python files compiled, and
  `git diff --check` passed.
- The two protected replay artifacts exercise captured candidates through the
  public cognition wrapper and preserve actual bounded partial-failure
  evidence. Supplemental live nodes remain gated because no live endpoint/model
  was enabled during this run.
- The default repository suite also reported 4,417 passed, 16 skipped, 1,175
  deselected, and 10 failures. The failures were reviewed as unrelated
  baseline/live/contract-harness issues outside the changed ownership surface;
  the mapped and adjacent acceptance suites remained green.
- Lifecycle status is `completed`; this record is closed historical scope and
  has no remaining implementation checkpoint.
