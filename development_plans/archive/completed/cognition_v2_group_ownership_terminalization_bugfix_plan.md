# cognition v2 group ownership terminalization bugfix

## Summary

- Goal: Repair the Cognition V2 group-conversation ownership failure in which a low-salience causal candidate is pruned before a later terminal proposition in the same appraisal batch can resolve it, and reject causal candidate handles when the model assigns participant/entity roles.
- Status: completed
- Scope boundary: cognition_core_v2 semantic-question contracts, semantic-appraisal validation, causal-candidate terminalization, the exact captured group trace replay, and the inherited Gate D verification references. Database exports are diagnostic evidence only.
- Change direction: Keep the current six-family appraisal and bounded retry structure, make candidate materialization and terminalization an atomic reducer contract, and split causal subject/object handles from role-assignment handles in one canonical contract update.
- Acceptance state: The reviewed plan was explicitly authorized, implemented within the declared scope, and verified through the exact deterministic impact map and the full-family real-LLM regression. The plan is closed with the residual baseline findings recorded below.

## Evidence And Root Clause

The evidence set is preserved at these paths:

- test_artifacts/diagnostics/trace_correlation_llmtrace_cb507d084dc64436b4a5cdc3232013b9.json
- test_artifacts/diagnostics/llm_trace_llmtrace_cb507d084dc64436b4a5cdc3232013b9.json
- test_artifacts/diagnostics/group_ownership_user_profile_0c16b202.json
- test_artifacts/diagnostics/group_ownership_user_memories_0c16b202.json
- test_artifacts/diagnostics/group_ownership_character_state_global.json
- test_artifacts/diagnostics/group_ownership_group_history_20260813.json
- test_artifacts/llm_debug/group_ownership_error/group_ownership_error_1786624015306248100.json
- test_artifacts/llm_debug/group_ownership_error/group_ownership_error_terminal_target__agent_review.md
- test_artifacts/diagnostics/group_ownership_terminalization_prefix_reconstruction.json
- test_artifacts/diagnostics/group_ownership_terminalization_deepseek_pro_review.md

The trace is one QQ group run in channel 480386272, with four addressed users,
current response ownership assigned to the current character, the embedded
actor assigned to the current user, and the embedded target assigned to other
participants. The protected failure is at semantic_appraisal_reduction for
q:goal_threat_outcome with semantic_appraisal_reduction_rejected and the
exception terminal proposition postcondition target is unknown.

The formal real-LLM reproduction passed one case at a time with the exact
captured input and preserved target candidate. It executed all six planned
question families, made 16 model calls including the preserved candidate and
the real repair/follow-up calls, and observed the same typed target reduction
failure. A one-question replay did not fail; the full family is required.

### Root clause

The group episode exposes a causal event candidate (ce1) and participant
ownership metadata in the same semantic appraisal surface; earlier appraisal
results create ce1 with a final salience of 20, deterministic retention prunes
that new candidate below the 25-point threshold, and the later
event_completed(ce1) proposition follows the stale candidate binding into
terminal postcondition validation, while the shared role-handle domain also
allows the model to misidentify ce1 as a participant target.

The causal sequence is:

1. The projected evidence creates ce1 -> candidate:event:e1 and participant
   handles separately in the canonical projection.
2. The accepted q:event_agency appraisal creates the candidate event and
   supplies a maximum accepted candidate delta of 20.
3. The cumulative batch reaches _recompute_new_causal_salience; the new event
   is removed because its salience is below 25. The local candidate binding is
   still mapped to the removed native event id.
4. The accepted q:goal_threat_outcome result contains the preserved
   event_completed proposition with subject_handle=ce1. Its terminal
   postcondition lookup cannot find the pruned native event and raises the
   captured error.
5. The same preserved proposition assigns target=ce1. ce1 is a causal event
   handle, not a participant/entity role handle; the current validator accepts
   it because permitted_role_handles mixes both domains. The reducer silently
   drops the event-typed role reference, so this is a semantic ownership defect
   and an independent contract hole even though the stale binding produces the
   terminal exception.

The target state must preserve an event that is terminalized by any accepted
proposition in the same batch, keep every referenced handle resolvable until
all postconditions finish, and reject ceN, ctN, and ckN as
role_assignments[*].entity_handle. Weak, non-terminal candidates must remain
eligible for the existing below-threshold pruning behavior.

## Confirmed Decisions

- The current trace, exact profile, memory, character state, and group-history
  exports are the reproduction evidence; no database write or profile rewrite
  is part of this plan.
- The completed
  development_plans/archive/completed/group_topic_continuity_authority_fix_plan.md
  is the predecessor reference. Its archived scope and Gate D record remain
  immutable; this plan owns the newly isolated terminalization/role-domain
  defect.
- The existing six appraisal families, model routes, attempt limits, prompt
  caps, public facade, and fail-closed action path remain unchanged.
- Deterministic validation owns handle-domain enforcement. The LLM stage may
  regenerate within its existing bounded contract-attempt limit; deterministic
  code does not infer a replacement participant or invent a semantic role.
- A terminal proposition is stronger than the weak-candidate salience filter
  within the same reduction transaction. A non-terminal candidate below the
  current threshold keeps the existing pruning behavior.
- No compatibility aliases, fallback mappers, parallel question vocabularies,
  new arbitration calls, keyword gates, forced-silence rules, or dialog-layer
  ownership logic are introduced.

## Scope And Change Direction

### Included

1. Add one canonical question field for role-assignment handles, generated from
   the existing projection references. Subject/object and delta-path handles
   continue to carry the causal lifecycle handles needed by each appraisal
   family; role assignments use only role-bearing handles and the explicit
   self/current_user handles.
2. Update semantic prompt construction, repair allowed-values, selection
   metadata, and deterministic appraisal validation to expose and enforce the
   separated domains.
3. Make same-batch candidate terminalization atomic with candidate retention:
   a candidate that receives a terminal proposition is retained through salience
   recomputation and terminal postcondition reassertion, with its local handle
   resolved to the surviving native entity id.
4. Convert the captured live reproduction into a regression expectation that
   exercises the exact input, all six appraisal families, preserved invalid role
   assignment, real repair call, and final public boundary. The repaired run
   must produce no stale-target reduction failure and must preserve group
   ownership semantics.
5. Add a sanitized deterministic fixture for the low-salience candidate followed
   by same-batch terminalization. The fixture contains typed state/evidence and
   handles only; it does not contain raw user profile, conversation, or model
   output text.
6. Update every direct SemanticQuestionV2 producer and consumer, including
   contract fixtures, prompt-budget assertions, stage-routing fixtures, failure
   matrices, the exported-appraisal context helper, and the ownership manifest.

### Deferred

- Any change to group scene selection, conversation-progress persistence,
  memory promotion, character profile data, adapter/intake ownership, action
  planning, dialog wording, delivery, or database schemas.
- Any redesign of candidate salience thresholds outside the terminalization
  invariant.
- Any change to model route, context budget, retry count, concurrency, or
  required-selection policy.
- Any production data correction or replay write against the captured QQ run.
- Any broad cleanup of pre-existing appraisal vocabulary not required by the
  two corrected contracts.

## Mandatory Skills And Rules

- development-plan: lifecycle, exact source-to-test traceability, review,
  approval, and execution boundaries.
- debug-llm: protected trace evidence, real-model replay, raw artifact
  inspection, and human-readable quality judgment.
- local-llm-architecture: semantic ownership, bounded appraisal contracts,
  handle authority, and no deterministic replacement of LLM judgment.
- test-style-and-execution: deterministic unit tests, exported replay tests,
  exact live collection, and one-at-a-time real-LLM execution.
- py-style: every Python source or test change.
- cjk-safety: any Python prompt or test fixture change containing CJK text.

The implementation owner preserves the repository rule that production code
changes require explicit user authorization after this draft is reviewed and
approved.

## Must Do

### Candidate terminalization invariant

In apply_semantic_appraisals and its owned helpers:

- Track candidate roots that are terminalized during the current batch.
- Remove a terminalized root from the weak-new-candidate pruning set, or apply
  an equivalent deterministic invariant that guarantees the same result.
- Reassert terminal postconditions only against native entities that remain in
  the candidate state. The candidate handle used by the proposition must resolve
  to that surviving native id for the entire transaction.
- Preserve existing pruning for a newly created candidate that has no terminal
  proposition and whose final accepted salience remains below 25.
- Preserve accepted-prefix isolation: one unrelated invalid appraisal remains
  rejectable without discarding the valid prefix.

### Role-assignment domain invariant

- Extend SemanticQuestionV2 with required
  permitted_role_assignment_handles.
- Keep the existing question-local causal handle domain for subject/object and
  target-path ownership. Use the new field only for
  role_assignments[*].entity_handle and the prompt's entity_handle field
  domain.
- Generate assignment handles from canonical projection references whose kind
  is a valid role-bearing entity, plus self and current_user; exclude all
  causal candidate handles (ceN, ctN, ckN) and lifecycle event/threat/gap
  handles from role assignments.
- Keep role token validation closed to the existing six role values. Do not
  map an invalid candidate to a participant, current user, or self.
- Validate selected-role metadata against the union of handles referenced by
  valid subject/object/assignment/path fields, while preserving candidate-origin
  evidence checks for subject, object, and delta handles.
- Give contract repair the exact separated domains. The invalid preserved
  target=ce1 candidate must be classified as a bounded semantic contract error
  and kept out of reduction, persistence, action planning, and delivery.
- Derive assignment handles per question family from that family's existing
  permitted subject/object handles whose canonical reference kind is in
  ROLE_ENTITY_KINDS, then add scene participant handles with kind
  third_party when the projected group scene contains them, plus the explicit
  self and current_user overrides. This admits family-owned relationship and
  goal handles where their role semantics are valid, while excluding event,
  threat, knowledge-gap, drive, meaning, and every ceN/ctN/ckN candidate.
- Keep participant handles out of subject/object and delta-path domains unless
  the existing family contract explicitly owns them. Participant handles are
  assignment-only in this fix.

### Regression evidence

- Keep the protected original trace and exact diagnostic exports immutable.
- Update the live replay assertions only after the deterministic contract and
  reducer tests establish the target state. The live case must still inject the
  preserved first candidate and delegate later calls to the real local model.
- Record raw calls, parsed results, validation disposition, reduction outcome,
  public observability, and final surface evidence in a new artifact; inspect
  the artifact before accepting the live node.

## Target State And Contracts

### Question contract

The canonical SemanticQuestionV2 shape contains:

    question_id
    question_kind
    semantic_question
    evidence_handles
    permitted_role_handles
    permitted_role_assignment_handles
    permitted_delta_paths
    dependencies

permitted_role_handles remains the family-local subject/object handle set,
including evidence-grounded causal candidates where that family owns them.
Delta target handles remain authorized by permitted_delta_paths.
permitted_role_assignment_handles is the only valid set for
role_assignments[*].entity_handle; it contains role-bearing native or scene
handles and self/current_user, and contains no ceN, ctN, ckN, evN, tN, or kN
handles.

For each question family, assignment handles are the intersection of that
family's existing permitted_role_handles with canonical references whose kind
is in ROLE_ENTITY_KINDS, plus projected scene handles whose kind is
third_party, plus self/current_user. The explicit family result is:

- event_agency: self, current_user, and scene third-party handles.
- relationship_social: r1 where authorized, self, current_user, and scene
  third-party handles.
- moral_identity: self, current_user, and scene third-party handles.
- goal_threat_outcome: eligible goal handles, self, current_user, and scene
  third-party handles.
- epistemic_comparison_memory: self, current_user, and scene third-party
  handles.
- existential_drive: self, current_user, and scene third-party handles.

No event, threat, knowledge-gap, drive, meaning, or causal candidate handle is
admitted solely because it is present in the projection.

The prompt's handle_field_domains is therefore:

    subject_handle: permitted_role_handles
    object_handle: permitted_role_handles
    entity_handle: permitted_role_assignment_handles
    evidence_handles: evidence_handles

The prompt's role_handle_semantics also contains every surviving scene
participant handle as a bounded structured reference with the semantic text
reference "群聊中的其他参与者". The participant handle is the only permitted
way to assign a third-party target; the model must omit the assignment when
the evidence does not identify a participant.

The human-readable prompt must describe the same domain split. The repair
payload must carry both allowed lists without substituting one for the other.

During context fitting, subject/object handles and assignment handles are
pruned independently. The fitted payload updates
permitted_role_handles, permitted_role_assignment_handles, and their
corresponding handle_field_domains entries as separate sets; the surviving
handle set used for delta-path fitting is independent from the assignment
survivor set. A participant assignment handle is never retained only because a
causal subject or delta path survived, and vice versa.

### Reducer contract

For one apply_semantic_appraisals transaction, these conditions hold:

    terminalized_candidate_ids ∩ weak_new_candidate_ids = empty
    every terminal proposition subject binding resolves to a state entity
    every terminal postcondition is applied before the transaction returns
    non-terminal weak candidates below the threshold remain prunable

The implementation may choose local data-structure mechanics, but it must
preserve these observable invariants and must not turn a stale binding into an
ordinary silence or an untyped exception.

## Execution Roles

### root_cause_and_plan_owner

- Responsibility: Maintain this plan, incorporate independent review findings,
  preserve the exact reproduction evidence, and keep lifecycle status accurate.
- Owned surface: This plan, the registry row, diagnostic/review artifacts, and
  plan evidence; no production implementation before authorization.
- Authority: May amend the draft within the confirmed scope. May not silently
  authorize production changes, widen Gate D, or edit the archived predecessor.
- Applicable skills: development-plan, debug-llm,
  local-llm-architecture, test-style-and-execution.
- Capability floor: Able to reason across prompt contracts, deterministic
  reducers, live LLM evidence, protected Mongo exports, and exact pytest node
  traceability.
- Independence requirement: Must not self-approve the independent plan review.
- Acceptance output: Reviewed draft with findings resolved or explicitly
  recorded, exact matrix, Gate D reference, and no unresolved implementation
  decisions.
- Gate: Reproduction evidence is present, worktree baseline is captured, and
  the independent review is complete before approval is requested.

### deepseek_pro_plan_reviewer

- Responsibility: Independently review the draft's root cause, ownership
  boundaries, contract changes, reducer invariant, test matrix, and Gate D
  acceptance without modifying repository files.
- Owned surface: Read-only review of this plan, the referenced source/tests,
  and the reproduction artifacts.
- Authority: May report findings and pass/fail plan readiness. May not edit
  source, tests, plan files, database state, or artifacts.
- Applicable skills: development-plan, debug-llm,
  local-llm-architecture, test-style-and-execution.
- Capability floor: Strong cross-boundary code and contract review with access
  to the full root-cause evidence and exact test-node mapping.
- Independence requirement: Separate runtime agent from the plan owner; the
  user fixed the review executor to DeepSeek Pro for this handoff.
- Acceptance output: Review feedback with findings grouped as blocking,
  required amendment, or advisory, and an explicit readiness verdict.
- Gate: The draft exists, its baseline/owned files are recorded, and the
  reviewer receives the complete plan plus reproduction evidence.

### runtime_implementation_owner

- Responsibility: Implement the approved contract and reducer change within
  this plan and produce the mapped verification evidence.
- Owned surface: The cognition_core_v2 paths in the change surface and their
  mapped tests/fixture only.
- Authority: May choose local mechanics that preserve the fixed target state.
  Requires explicit user implementation authorization and an approved or
  in-progress plan. May not add compatibility aliases, change route/caps, or
  modify deferred subsystems.
- Applicable skills: py-style, cjk-safety, debug-llm,
  local-llm-architecture, test-style-and-execution, development-plan.
- Capability floor: Strong Python state-reducer reasoning, typed LLM contract
  design, exact pytest mapping, and live artifact inspection.
- Independence requirement: Separate from the reviewer for final review.
- Acceptance output: Source/test diff, exact collection and execution results,
  deterministic and live artifacts, and a changed-path handoff.
- Gate: Approved plan, explicit implementation authorization, clean baseline
  comparison, and all required skills loaded.

## Test Impact And Traceability

| Repository path | Owned symbol or contract | Semantic owner | Exact deterministic nodes | Supplemental integration/live nodes | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| src/kazusa_ai_chatbot/cognition_core_v2/contracts.py | SemanticQuestionV2.permitted_role_assignment_handles | semantic question contract owner | tests/unit/cognition_core_v2/test_contracts.py::test_semantic_question_v2_requires_role_assignment_handles | tests/test_cognition_core_v2_appraisal_contract_live_llm.py::test_live_appraisal_projects_exact_paths_without_crossing_domains | deterministic unit plus live contract evidence | Prevents an incomplete question object from crossing the typed contract boundary. |
| src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py | family-local role-assignment handle projection | semantic question planner owner | tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_goal_outcome_keeps_eligible_handles_and_candidates; tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_group_questions_separate_causal_and_role_assignment_handles; tests/test_cognition_core_v2_alignment_gates.py::test_candidate_handles_share_the_projection_authority | tests/test_qq_group_public_scene_live_llm.py::test_live_participant_branch_isolation | deterministic unit plus inherited live shield | Prevents a group causal candidate from crossing into a participant branch. |
| src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py | prompt field domains, repair values, selected-role validation, _validate_proposition, fitted-domain survival | semantic appraisal contract owner | tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_causal_candidate_is_rejected_as_role_assignment_handle; tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_candidate_role_assignment_handle_is_rejected; tests/test_cognition_core_v2_semantic_terminalization.py::test_invalid_role_handle_reports_its_structured_domain; tests/test_cognition_core_v2_prompt_contract_guidance.py::test_prompt_payloads_preserve_contract_order; tests/test_cognition_core_v2_prompt_budget_continuity.py::test_appraisal_question_keeps_candidate_origin_contract; tests/test_cognition_core_v2_prompt_budget_continuity.py::test_appraisal_repair_uses_residual_budget_for_second_call | tests/test_cognition_core_v2_appraisal_contract_live_llm.py::test_live_appraisal_projects_exact_paths_without_crossing_domains | deterministic contract plus one-at-a-time live contract evidence | Keeps invalid ownership candidates out of semantic reduction and bounded repair. |
| src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py | _materialize_proposition_root, _recompute_new_causal_salience, _reassert_terminal_postconditions | causal-state reducer owner | tests/test_cognition_core_v2_semantic_terminalization.py::test_terminalized_low_salience_candidate_survives_same_batch_pruning; tests/test_cognition_core_v2_semantic_terminalization.py::test_terminal_postconditions_survive_same_batch_deltas; tests/test_cognition_core_v2_semantic_terminalization.py::test_final_reduction_preserves_cross_appraisal_composition | tests/test_cognition_core_v2_group_ownership_error_live_llm.py::test_live_group_ownership_error_repairs_terminal_target | deterministic reducer plus live replay | Prevents a same-batch terminal proposition from resolving a pruned or stale candidate. |
| tests/ownership/source_test_impact_manifest.json | required_unit_tests rows for every changed cognition_core_v2 owner | test-impact ownership | tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary; tests/test_development_plan_test_impact_contract.py::test_skill_requires_exact_test_impact_matrix | none | deterministic manifest validation | Prevents a green local suite from bypassing exact changed-path ownership. |
| tests/fixtures/cognition_v2_group_ownership_terminalization.json | sanitized state/evidence/appraisal fixture | regression-fixture owner | tests/test_cognition_core_v2_semantic_terminalization.py::test_terminalized_low_salience_candidate_survives_same_batch_pruning | none | deterministic fixture replay | Reproduces the exact low-salience-before-terminal ordering without private data. |
| tests/test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm.py | _build_appraisal_context scene_context propagation and split-domain fitting | replay-context owner | tests/test_cognition_core_v2_prompt_contract_guidance.py::test_prompt_payloads_preserve_contract_order | tests/test_cognition_core_v2_group_ownership_error_live_llm.py::test_live_group_ownership_error_repairs_terminal_target | deterministic helper contract plus one-at-a-time live replay | Ensures the formal replay sees the same pN participant handles as production. |
| tests/test_cognition_core_v2_group_ownership_error_live_llm.py | exact trace replay and repaired public-boundary acceptance | live regression owner | tests/test_cognition_core_v2_semantic_terminalization.py::test_final_reduction_preserves_cross_appraisal_composition | tests/test_cognition_core_v2_group_ownership_error_live_llm.py::test_live_group_ownership_error_repairs_terminal_target | one-at-a-time real LLM with raw artifact review | Prevents recurrence of the fetched group failure across the complete six-family flow. |

### SemanticQuestionV2 ripple inventory

The required new field is a big-bang contract update. These direct producers and
consumers are in scope and must be updated together:

- tests/test_cognition_core_v2_failures.py: direct question dictionaries in
  test_unsupported_appraisal_can_select_no_evidence,
  test_appraisal_handle_errors_name_rejected_values_and_allowlist,
  test_candidate_proposition_rejects_mismatched_evidence,
  test_candidate_delta_rejects_mismatched_evidence, and
  test_candidate_binding_uses_canonical_projection_reference.
- tests/test_cognition_core_v2_appraisal_contract_live_llm.py: _build_case
  fixtures and exact handle_field_domains assertions.
- tests/test_cognition_core_v2_stage_model_routing.py: the direct question
  fixture used by test_each_appraisal_family_reuses_its_route_for_repair_and_trace.
- tests/test_cognition_core_v2_prompt_contract_guidance.py: exact payload key
  order, field-domain assertions, repair allowed-values, and independent
  fitted-domain assertions.
- tests/test_cognition_core_v2_prompt_budget_continuity.py: residual-budget
  appraisal payload, candidate-origin repair values, and independent fitting
  of assignment versus subject/object domains.
- tests/test_cognition_core_v2_semantic_terminalization.py: terminal and
  state-incompatibility question fixtures and domain assertions.
- tests/test_cognition_core_v2_trace_failure_mode_matrix.py: _semantic_question
  and the new candidate-role rejection node.
- tests/test_cognition_core_v2_alignment_gates.py: existing planner-alignment
  assertions and group candidate authority checks must include the separate
  assignment-domain invariant.
- tests/unit/cognition_core_v2/test_semantic_appraisal.py and
  tests/unit/cognition_core_v2/test_semantic_source_planner.py: canonical
  unit fixtures and new domain assertions.
- tests/test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm.py:
  _build_appraisal_context must pass scene_context=payload.get("scene_context")
  to project_state_for_prompt; the helper must preserve p1-p5 in the
  assignment domain for the captured group input.
- tests/test_cognition_core_v2_group_ownership_error_live_llm.py: rename the
  post-fix node to
  test_live_group_ownership_error_repairs_terminal_target, assert pN
  participant assignment handles, and expect bounded contract repair followed
  by a valid reduction.

Every changed production path must retain a collected exact deterministic node
before live evidence is accepted. Future node names in this draft are fixed
acceptance identifiers for the implementation; they are not permission to
skip the corresponding tests.

## Change Surface

### Delete

- None.

### Modify

- src/kazusa_ai_chatbot/cognition_core_v2/contracts.py: add the required
  role-assignment domain to the canonical question contract.
- src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py: derive
  the separate assignment-handle set from canonical projected references.
- src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py: expose the
  separated prompt/repair domains and reject candidate handles in role
  assignments before reducer trial or commit.
- src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py: preserve
  same-batch terminalized candidates through weak-candidate pruning and keep
  handle resolution valid until postcondition reassertion completes.
- tests/unit/cognition_core_v2/test_semantic_source_planner.py and
  tests/unit/cognition_core_v2/test_semantic_appraisal.py: enforce the new
  contract domains.
- tests/unit/cognition_core_v2/test_contracts.py: add the direct required-field
  contract node for SemanticQuestionV2.
- tests/test_cognition_core_v2_semantic_terminalization.py and
  tests/test_cognition_core_v2_trace_failure_mode_matrix.py: add reducer and
  contract regression nodes.
- tests/test_cognition_core_v2_failures.py,
  tests/test_cognition_core_v2_appraisal_contract_live_llm.py,
  tests/test_cognition_core_v2_prompt_contract_guidance.py, and
  tests/test_cognition_core_v2_stage_model_routing.py: update every direct
  question fixture and exact prompt/repair assertion in the contract ripple.
- tests/test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm.py:
  pass scene_context through the deterministic replay helper and fit the two
  handle domains independently.
- tests/test_cognition_core_v2_group_ownership_error_live_llm.py: retain the
  exact trace replay while renaming the post-fix node to
  test_live_group_ownership_error_repairs_terminal_target and changing the
  expected disposition from stale-target reduction failure to bounded contract
  repair plus successful reduction.
- tests/ownership/source_test_impact_manifest.json: add exact required unit
  nodes for every changed cognition_core_v2 owner.
- development_plans/README.md: register this draft under active bugfix plans.

### Create

- tests/fixtures/cognition_v2_group_ownership_terminalization.json: sanitized
  deterministic reproduction fixture.

### Keep

- The archived predecessor plan and all protected trace/database exports.
- Existing candidate salience threshold and pruning for non-terminal weak
  candidates.
- Existing appraisal routes, attempt caps, prompt caps, accepted-prefix
  isolation, action selection, dialog generation, and delivery ownership.
- tests/ownership/source_test_impact_manifest.json remains the authoritative
  source-to-test map and is updated in the same contract change.

## Gate D Reference From The Predecessor Plan

The following is the inherited reference set from
group_topic_continuity_authority_fix_plan.md. It is not an instruction to
append scope to that completed plan. The new group failure regression is added
to the same preflight and review discipline.

The predecessor's four residue-recorder live cases and its Gate E latency
evidence are excluded from this plan because this bugfix defers residue
persistence and runtime-latency ownership. They remain historical acceptance
evidence of the completed predecessor and are not silently re-adopted here.
The predecessor's S1/S2/S4 nodes remain characterization/non-expansion gates:
this plan does not claim to fix those baseline failures and must not convert
their outcomes into a repair claim.

### Gate D execution protocol

Before each live node, verify service health, exact character identity, exact
local model route and effective 50,000 context setting, MongoDB connectivity
and explicit isolated test database, artifact writability, protected fixture
availability, and debug permission. Collect and run one exact node at a time
with live markers explicitly enabled:

    $env:PYTHONPATH="src"
    venv\Scripts\python.exe -m pytest -o addopts= -m live_llm --collect-only -q tests/<live_file>.py::<node>
    venv\Scripts\python.exe -m pytest -o addopts= -m live_llm tests/<live_file>.py::<node> -q -s

Skipped, deselected, xfailed, setup-failed, wrong-route, missing-artifact, or
unreviewed-output cases are gate failures.

### Mandatory captured all-lane references

- tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s1_private_surface_characterization
- tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s2_public_boundary_characterization
- tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s4_reality_correction_characterization
- tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s6_final_surface_keeps_triage_counteraction_return_order
- tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s7_all_lanes_keep_crisis_foreground
- tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s8_all_lanes_keep_injury_foreground_repeat_1
- tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s8_all_lanes_keep_injury_foreground_repeat_2
- tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s8_all_lanes_keep_injury_foreground_repeat_3

### Focused group and changed-branch references

- tests/test_qq_group_public_scene_live_llm.py::test_live_group_crisis_anchor_beats_other_user_noise
- tests/test_qq_group_public_scene_live_llm.py::test_live_group_reward_control_remains_playful
- tests/test_qq_group_public_scene_live_llm.py::test_live_group_same_user_continuity_survives_unrelated_noise
- tests/test_cognition_core_v2_live_character_judgment.py::test_live_goal_progresses_high_affinity_guarded_continuity
- tests/test_cognition_core_v2_live_character_judgment.py::test_live_goal_releases_stale_residue_for_changed_group_scene
- tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_separates_progress_and_optional_rows
- tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_accepts_one_progress_event
- tests/test_cognition_core_v2_captured_run_failures_live_llm.py::test_captured_run_goal_relational_willingness_repair_live_llm
- tests/test_cognition_core_v2_goal_capability_live_llm.py::test_live_captured_online_search_goal_preserves_required_evidence

### Mandatory quality-shield references

- tests/test_qq_group_public_scene_live_llm.py::test_live_public_target_distinct
- tests/test_qq_group_public_scene_live_llm.py::test_live_parallel_addresses
- tests/test_qq_group_public_scene_live_llm.py::test_live_public_topic_pivot
- tests/test_qq_group_public_scene_live_llm.py::test_live_participant_branch_isolation
- tests/test_qq_group_public_scene_live_llm.py::test_live_noise_only_silence
- tests/test_conversation_progress_v2_live_llm.py::test_live_original_failure_progress_semantic_handoff
- tests/test_conversation_progress_v2_live_llm.py::test_live_asuna_houjing_long_thread_regression
- tests/test_conversation_progress_v2_live_llm.py::test_live_interleaved_group_multifragment_continuation
- tests/test_conversation_progress_v2_live_llm.py::test_live_group_stale_ambient_is_absent_from_stage_zero_prompt
- tests/test_conversation_progress_v2_live_llm.py::test_live_private_stale_progress_is_pruned_before_cognition
- tests/test_cognition_core_v2_p0_context_reconnection_live_llm.py::test_live_reply_residual_reaches_goal_only
- tests/test_cognition_core_v2_p0_context_reconnection_live_llm.py::test_live_group_self_cognition_uses_one_advisory_projection
- tests/test_e2e_live_llm.py::test_live_chat_multi_user_photo_thread_keeps_user_intents_separated
- tests/test_e2e_live_llm.py::test_live_chat_multi_user_quantization_thread_keeps_xuezhang_bound_to_haodieyou
- tests/test_e2e_live_llm.py::test_live_chat_multi_user_understanding_thread_keeps_joke_and_self_definition_separate
- tests/test_e2e_live_llm.py::test_live_chat_multi_user_preferences_remain_isolated_across_suffix_english_and_switch
- tests/test_short_horizon_state_composition_e2e_live_llm.py::test_private_event_changes_next_group_turn
- tests/test_short_horizon_state_composition_e2e_live_llm.py::test_group_event_changes_next_private_turn

The new focused regression is additionally:

- tests/test_cognition_core_v2_group_ownership_error_live_llm.py::test_live_group_ownership_error_repairs_terminal_target

Gate D acceptance requires no exception, non-empty valid output, valid trace,
correct route, parseable output, durable raw/parsed evidence, no private
leakage or wrong participant/addressee/semantic owner, no conflicting
objective or unauthorized action, unchanged prompt/call/retry shape, and a
human Markdown review of the raw output. The quality judgment must confirm
that the response owner remains the current character and participant targets
remain participant handles rather than causal candidates.

## Verification And Acceptance Criteria

### Plan-stage evidence already complete

- Exact trace id llmtrace_cb507d084dc64436b4a5cdc3232013b9 was located and
  exported with one run, eight steps, one failure capsule, three event-log
  records, and six conversation-history rows.
- Exact profile, memories, character state, and group-history exports were
  produced read-only.
- The new formal live replay collected one exact live node and passed while
  reproducing the expected target reduction failure. Its raw artifact and
  human-readable review are preserved.
- The offline reconstruction narrowed the failure to the fourth cumulative
  result: after the first three result families, the event list still lacks
  the low-salience candidate while the candidate comparison records create;
  adding the terminal target result raises the captured error. The exact
  prefix evidence is preserved at
  test_artifacts/diagnostics/group_ownership_terminalization_prefix_reconstruction.json.

### Implementation acceptance

1. Exact deterministic nodes in the traceability matrix collect and pass with
   no skip, deselection, xfail, or setup failure.
2. The low-salience same-batch terminalization fixture produces one surviving
   resolved event with the terminal postcondition applied; its candidate handle
   resolves to the surviving native id. A weak candidate without terminal
   ownership remains pruned.
3. A candidate causal handle in a role assignment fails deterministic contract
   validation, triggers only the existing bounded repair path, and never enters
   reducer, persistence, action, dialog, or delivery state.
4. Existing valid role handles (self, current_user, scene participants,
   relationship, and valid goal/role-bearing handles) remain accepted where
   their question family authorizes them.
5. The renamed exact live replay completes the full six-family public boundary without
   terminal proposition postcondition target is unknown, without a
   semantic_appraisal_reduction_failure for q:goal_threat_outcome, and with a
   valid final cognition/action/surface result. The raw artifact records the
   first preserved contract rejection and the real follow-up disposition.
6. The predecessor Gate D references remain preserved as historical acceptance
   evidence, and the new focused group regression passes its applicable Gate D
   hard gate with individually reviewed raw evidence. The excluded predecessor
   residue/latency cases and S1/S2/S4 characterization cases remain outside
   this plan; unrelated baseline failures are explicitly labeled and do not
   become a claim of repair by this plan.
7. scripts.validate_test_impact --base-ref HEAD --run and exact collection
   prove every changed production source path maps to a collected deterministic
   node.
8. No production implementation begins until this plan is independently
   reviewed, amended, approved, and explicitly authorized by the user.

## Agent Autonomy Boundaries

The implementation owner may choose helper names, local decomposition, and
command order only when the target contracts and invariants remain unchanged.
The owner must request a plan amendment before changing the question schema,
candidate threshold semantics, terminalization precedence, model route,
retry/call shape, state persistence, or any deferred subsystem. The owner may
not add aliases or translate old question shapes for compatibility. The plan
owner remains responsible for recording every amendment and routing corrected
work through an independent review.

## Independent Plan Review

DeepSeek Pro receives this draft as an independent, read-only review under the
user-fixed reviewer role. The review must assess:

- whether the root clause is supported by the exact trace and offline prefix
  reconstruction;
- whether candidate terminalization and handle-domain separation preserve the
  architecture's semantic ownership boundaries;
- whether the target contract is canonical and free of compatibility shims;
- whether the source-to-test matrix contains exact deterministic nodes and
  sufficient live evidence;
- whether the inherited Gate D references are correctly identified without
  expanding the completed predecessor plan; and
- whether acceptance criteria prove both failure removal and preservation of
  weak-candidate pruning/valid group behavior.

The review output was incorporated before implementation authorization. The
reviewer did not remediate its own findings or approve the corrected plan.

### DeepSeek Pro review disposition

DeepSeek Pro completed the requested read-only review after the full handoff
deadline allowance and returned verdict needs amendments. The reviewer
confirmed the trace counts, formal replay, primary prune-then-terminalize root
clause, and secondary candidate-role contract hole. The parent addressed every
blocking and required item in this revision:

- the complete SemanticQuestionV2 producer/consumer ripple is listed above;
- the ownership manifest and direct contracts node are mapped;
- scene_context and p1-p5 assignment preservation are explicit;
- per-family role-assignment domains, participant semantics, and independent
  budget fitting are fixed;
- the post-fix live node name is coherent;
- the cumulative-prefix artifact is cited;
- the predecessor Gate D boundary, excluded residue/latency evidence, and
  S1/S2/S4 characterization caveat are explicit.

The durable review summary is
test_artifacts/diagnostics/group_ownership_terminalization_deepseek_pro_review.md.
All blocking and required findings were resolved before the user authorized
implementation.

## Execution Evidence

### Runtime handoffs

- The initial DeepSeek Pro implementation handoff was stopped at the user's
  explicit model correction before any owned implementation file changed.
- DeepSeek Flash 0731 was then resolved as the runtime implementation owner
  under the existing `runtime_implementation_owner` role. Its model was
  `deepseek-v4-flash` with the fixed high reasoning setting and repository
  verification access. The owned production surface was contracts.py,
  semantic_source_planner.py, semantic_appraisal.py, and state_reducers.py,
  together with the mapped tests, fixture, replay context, and ownership
  manifest listed in the change surface.
- The baseline and owned-scope comparison preserved all pre-existing worktree
  changes. No archived plan or protected diagnostic export was modified.

### Implementation and verification

- `venv\Scripts\python.exe -m scripts.validate_test_impact --base-ref HEAD --run`
  collected and passed all 27 exact mapped impact nodes.
- The broader non-live collection produced 177 passes and 2 inventory-dependent
  skips. Four unrelated baseline failures remained unchanged: the goal prompt
  maximum-evidence budget case, two goal-route reuse cases, and the relational
  willingness exact-field case.
- `venv\Scripts\python.exe -m py_compile` passed for all 17 changed Python
  files. Ruff's repository findings remained at the pristine-HEAD baseline;
  the implementation introduced no new finding codes according to the
  comparison recorded by the implementation owner.
- The exact formal regression command
  `venv\Scripts\python.exe -m pytest -o addopts= -m live_llm tests/test_cognition_core_v2_group_ownership_error_live_llm.py::test_live_group_ownership_error_repairs_terminal_target -q -s`
  passed one case in 62.94 seconds. It completed all six appraisal families,
  rejected the preserved `ce1` role assignment at contract validation,
  accepted one bounded live repair, and produced no stale-target reduction
  failure. The latest raw artifact is
  `test_artifacts/llm_debug/group_ownership_error/group_ownership_error_1786628869738918000.json`.
  The human-readable review is
  `test_artifacts/llm_debug/group_ownership_error/group_ownership_error_terminal_target_repaired__agent_review.md`.

### Residual risk and sign-off

- The live model repaired the event target as `self` rather than a `pN`
  participant. The deterministic boundary correctly preserved the explicit
  participant domain and did not invent a participant; this is a model-level
  semantic-quality observation, not an unresolved contract or reducer defect.
- The inherited predecessor Gate D references, excluded residue/latency
  evidence, and S1/S2/S4 characterization caveat remain historical scope
  boundaries. The new focused Gate D regression and its raw/review artifacts
  are the applicable live sign-off for this bugfix.
- The plan owner accepts the implementation and verification evidence above;
  the reviewed findings, execution checkpoint, residual baseline results, and
  lifecycle registry update are complete.

## Progress Checklist

- [x] Fetch protected trace correlation and LLM evidence.
- [x] Export exact user profile, user memory, character state, and group history.
- [x] Identify the completed predecessor plan and Gate D reference nodes.
- [x] Reproduce the failure through the formal full-family real-LLM test.
- [x] Confirm the root clause with an offline cumulative-prefix reconstruction.
- [x] Draft this bugfix plan.
- [x] Obtain independent DeepSeek Pro plan review.
- [x] Incorporate review findings and finalize the reviewed draft for user approval.
- [x] Receive explicit implementation authorization.
- [x] Execute the approved plan and complete deterministic/live Gate D evidence.
