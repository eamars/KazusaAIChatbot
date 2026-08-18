# Cognition V2 semantic-appraisal boundary recovery and typed failure disposition

## Summary

- **Status:** completed
- **Independent proposal review:** GPT-5.6 SOL, high reasoning, normal speed tier, reviewed the RCA and fix proposal on the reused existing SOL subagent thread. The proposal review returned `REJECT` because the initial proposal did not fully specify the typed disposition matrix, the completed-plan boundary amendment, the evidence-preservation invariant, or the exhaustion metadata contract. Those findings are incorporated below.
- **Owner:** parent agent, as Cognition V2 contract owner
- **Trigger:** protected trace `llmtrace_79651aa48cfd41d0a50c06343dbaa8db` failed its first semantic-appraisal item with `cognition_boundary_rejected`, leaving `final_dialog_count=0`.
- **Goal:** restore narrowly bounded producer-contract recovery for correctable semantic-appraisal candidates while preserving deterministic ownership, provenance, permission, state, effect, and delivery safety.
- **Execution boundary:** the user explicitly authorized implementation by commanding execution of this bugfix plan. The parent owns production-code changes, test changes, verification, remediation, and closure evidence.
- **Change boundary:** semantic-appraisal failure classification, bounded producer replacement, evidence-preservation validation, protected failure metadata, facade omission handling, tests, source-to-test ownership, and the current Cognition V2 contract documentation.
- **No database or migration work:** this bugfix changes runtime admission and diagnostics only. It has no schema, persisted-state, adapter, model-route, scheduler, or deployment migration.

The bugfix is deliberately narrow. It restores recovery only for producer-contract errors whose correction is an explicit replacement decision owned by the producing LLM stage. It does not restore retries for semantic ownership, proposition meaning, target policy, permissions, state/FSM transitions, or effect-boundary decisions.

## Incident evidence and root cause

### Evidence set

The RCA used the protected trace export, a pre-regression successful trace, the current implementation, and the existing deterministic failure matrix:

| Evidence | Location | Finding |
|---|---|---|
| Current protected trace export | `test_artifacts/diagnostics/llm_trace_llmtrace_79651aa48cfd41d0a50c06343dbaa8db_20260818T004324Z.json` | One failed run, no final dialog, first failure at `semantic_appraisal.q:goal_threat_outcome.item_1`. |
| Current human-readable RCA | `test_artifacts/diagnostics/llm_trace_79651aa48cfd41d0a50c06343dbaa8db_rca.md` | Raw candidates, failure capsule, adjacent failure, historical comparison, and proposal. |
| Historical pre-regression trace | `test_artifacts/diagnostics/llm_trace_llmtrace_56ad102f8f07411bafa9b74cca38fdf8_20260818T004834Z.json` | The same provenance family reached a bounded repair call and produced four dialogs. |
| Current implementation | `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py` | Broad boundary-marker substring matching turns validator `ValueError` instances into terminal `cognition_boundary_rejected` errors. |
| Completed-plan contract | `development_plans/archive/completed/cognition_v2_stale_axes_and_validator_policy_plan.md` | Parseable semantic content was intentionally excluded from semantic retries; the new plan must amend that boundary narrowly. |
| Adjacent failure matrix | `tests/test_cognition_core_v2_trace_failure_mode_matrix.py` | The same broad branch covers origin, handle-domain, role, proposition-kind, delta, and state-policy failures that need separate dispositions. |

### First failure

The first failed candidate was from `COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME`:

- candidate subject `ck1` was mapped by the question to originating evidence `e1`;
- the candidate cited resolution evidence `e3` only;
- the validator correctly rejected the missing origin citation as `ck1->e1`;
- the current `_SemanticBoundaryValidationError` path classified that rejection as terminal and raised `CognitionExecutionError` with `error_code="cognition_boundary_rejected"` and `retryable=False`;
- no replacement candidate was requested, and the run completed with zero visible dialogs.

The second independent failure used `ev1` for selected roles even though that question permitted only `current_user` and `self`. It was also classified by the same broad terminal path. This proves the problem is a family-level disposition regression rather than one isolated `ck1` prompt error.

### Direct and contributing causes

1. The direct cause is the new terminal branch in `_appraise_semantic_item`, which intercepts boundary-like `ValueError` messages before the existing bounded contract-repair loop.
2. The classification is unsafe because `_is_boundary_validation_error` uses broad message substrings, and `_validate_semantic_boundary_candidate` wraps every validator `ValueError` into one undifferentiated exception type.
3. The earlier validator rule requiring candidate-origin evidence is correct and remains authoritative. The regression is that a producer-owned correction was treated as an unrecoverable semantic boundary decision.
4. The same broad branch can terminalize adjacent producer handle-domain errors, while an indiscriminate retry would be unsafe for proposition-kind mismatch, unowned delta paths, duplicate/conflicting entries, permissions, and state/FSM failures.
5. Existing failure metadata does not consistently expose typed failure kind, repair status, attempt count, retryability, and final disposition when raw output is unavailable.

## Contract amendment to the completed plan

The completed stale-axes and validator-policy plan remains immutable historical scope. This bugfix plan is a narrow superseding contract for semantic-appraisal boundary disposition; it is not an append to that completed plan.

The completed plan’s rule of zero semantic retries for parseable semantic content remains in force for:

- semantic ownership and proposition-kind judgment;
- subject/object meaning and target-policy mismatches;
- role semantics and delta ownership;
- duplicate/conflicting propositions or deltas;
- permissions, state/FSM transitions, and effect-boundary decisions.

This plan amends only the producer-contract subset where the validator can name a declared field domain or a canonical evidence-origin binding and the producing stage can issue a bounded replacement. The runtime still performs no semantic correction, handle substitution, evidence union, role rebinding, or policy decision.

## Scope and fixed change direction

### In scope

1. Replace substring-based boundary classification with typed, field-specific failure classification at the validator boundary.
2. Route only `candidate_origin_missing` and explicitly declared producer handle-domain failures through the existing bounded producer replacement budget.
3. Preserve valid citations from the failed candidate during replacement validation at both result-level selected evidence and nested proposition/delta evidence fields.
4. Keep replacement ownership with the producing LLM stage. The runtime validates the replacement and rejects it if it drops preserved evidence or still violates its declared contract.
5. Record protected metadata for every typed failure and final disposition without requiring raw model output.
6. Keep the existing `semantic_appraisal_contract_exhausted` facade omission route for exhausted recoverable candidates.
7. Extend deterministic, public-boundary, protected-trace, adjacent-failure, and source-ownership coverage.
8. Update the current Cognition V2 README and registry entry to document the amended contract.

### Explicitly out of scope

- A general semantic evaluator, critic, or repair agent.
- Retrying semantic ownership, proposition-kind, subject-kind, target, delta-ownership, permission, state/FSM, effect, or delivery failures.
- Deterministic rewriting, unioning, sorting into a replacement, deleting invalid handles, or rebinding roles.
- Replacing a missing origin citation with the origin while dropping a valid resolution citation.
- Adding a compatibility shim, alternate schema, parallel vocabulary, or legacy fallback.
- Changing the canonical JSON parser or `JSON_REPAIR_LLM` behavior.
- Changing appraisal prompt content, model routing, attempt caps, dialogue wording, persistence schemas, or adapter delivery.

## Mandatory skills and execution roles

### Mandatory skills

- `.agents/skills/development-plan`: plan lifecycle, exact change contract, traceability, and closure evidence.
- `.agents/skills/llm-trace-debug`: protected trace export and evidence handling.
- `.agents/skills/debug-llm`: human-readable raw/structured quality evidence and replay inspection.
- `.agents/skills/local-llm-architecture`: preserve LLM semantic ownership and deterministic boundary ownership.
- `.agents/skills/py-style`: all Python implementation and review.
- `.agents/skills/test-style-and-execution`: all test changes and execution; live LLM cases run individually with inspected output.
- `.agents/skills/cjk-safety`: apply if implementation or tests add CJK string literals to Python files.

### Execution roles

| Work | Owner | Boundary |
|---|---|---|
| Production implementation | Parent | Owns all source edits and contract decisions. |
| Test implementation and execution | Parent | Owns all test edits, deterministic batches, individual live-LLM runs, and evidence. |
| Implementation review | Reused GPT-5.6 SOL subagent thread | Read-only review after parent implementation; high reasoning, normal speed tier; no new reviewer spawn unless the user explicitly directs one. |
| Final sign-off | Reused GPT-5.6 SOL subagent thread | Reviews the implemented target and verification evidence, not the plan text alone. |
| Plan registry and closure | Parent | Updates status only after implementation, verification, review, and user-directed closure conditions are complete. |

## Must do

### 1. Define typed failure classes

Implement a stable internal failure classification with the following wire-safe `failure_kind` values:

| `failure_kind` | Meaning | Repair route |
|---|---|---|
| `candidate_origin_missing` | A candidate-bearing proposition or delta cites valid question evidence but omits the canonical originating evidence for its candidate handle. | Bounded producer replacement. |
| `producer_handle_domain_invalid` | A generated handle in a field with an explicit question-local producer allowlist is outside that allowlist, with the exact field path identified. | Bounded producer replacement. |
| `semantic_boundary_terminal` | A known deterministic semantic boundary violation that is not producer-contract repairable. | Terminal rejection; no repair call. |
| `unknown_validation_failure` | An untyped, unmapped, or unexpected validator failure. | Fail closed; no repair call. |

The implementation must classify from validator-owned field and contract facts, not exception-message substrings. `_is_boundary_validation_error` is removed or made unreachable; no broad marker list may decide retryability. Validator sites must provide the field path, expected domain/relationship, and failure kind directly.

The existing structural, provider, prompt-cap, and state-incompatibility dispositions remain separate. A `CognitionStateError`, permission/effect rejection, or terminal FSM failure cannot be converted into a producer repair merely because it is represented as a `ValueError` downstream.

### 2. Enforce the fixed disposition matrix

Every failure family in the current matrix receives exactly one route:

| Failure family | Fixed disposition | Producer repair |
|---|---|---:|
| Provider/transport failure | Existing provider attempt cap and terminal provider disposition | Existing provider path only |
| Prompt-cap exhaustion | Existing zero-call prompt-cap disposition | No |
| Malformed JSON, wrapper, object shape, required structural field, or wrong structural type | Existing canonical parser/structural recovery and owner cap | Existing structural path |
| Candidate-origin evidence missing in a proposition | `candidate_origin_missing`; replacement must preserve valid citations and add the origin | Yes |
| Candidate-origin evidence missing in a delta | `candidate_origin_missing`; same preservation rule | Yes |
| Subject, object, role-assignment, selected-evidence, or selected-role value outside a declared question-local producer handle domain | `producer_handle_domain_invalid` with exact field path and allowlist | Yes |
| Selected role `ev1` when only `current_user` and `self` are permitted | `producer_handle_domain_invalid`; producer reselects from the explicit allowlist | Yes |
| Unknown/noncanonical handle with no declared producer-domain repair contract | `unknown_validation_failure` | No |
| Duplicate handles or duplicate/conflicting propositions or deltas | `semantic_boundary_terminal` | No |
| Invalid role enum or role-value shape | `semantic_boundary_terminal` | No |
| Proposition-kind/subject-kind mismatch, target ownership mismatch, or object not permitted by semantic contract | `semantic_boundary_terminal` | No |
| Unowned, duplicate, or unsupported delta path/type/reason | `semantic_boundary_terminal` | No |
| Goal/relationship evidence or role ownership outside the owning question contract | `semantic_boundary_terminal` unless it is the exact declared producer handle-domain case above | No |
| Permission, consent, authorization, effect-boundary, or delivery rejection | `semantic_boundary_terminal` | No |
| Terminal event/goal/FSM transition or state incompatibility | Existing terminal state disposition | No |
| Unmapped validator exception | `unknown_validation_failure` | No |
| A repairable class whose replacement attempts all fail | Existing `semantic_appraisal_contract_exhausted`; facade records `question_omitted` | Cap applies |

There is no broad “handle/domain” retry category. A handle repair is eligible only when the validator has an explicit field-level allowlist and the failure is a generated producer value. Duplicate, conflicting, noncanonical, or semantically disallowed values remain terminal unless they match that exact contract.

### 3. Preserve origin and resolution evidence during repair

The replacement contract is fixed as follows:

1. Validate the failed candidate’s evidence handles against the question evidence set.
2. Retain every valid citation already present in the failed candidate.
3. Require the replacement candidate to include the canonical origin citation in addition to those retained citations.
4. Apply the invariant at both levels: `selected_evidence_handles` and each candidate-bearing proposition/delta `evidence_handles` field.
5. Rerun the complete validator after the replacement; do not deterministically union, replace, sort away, or rewrite candidate content.
6. Reject a replacement that drops a valid citation, even when it adds the missing origin.

For the captured `ck1` case, the accepted replacement must preserve `e3` and add `e1`, so the relevant nested and selected evidence sets contain `{e1, e3}`. The producing model owns the replacement ordering and content; deterministic code only verifies preservation and contract compliance.

The same preservation check applies to all candidate-bearing proposition and delta paths. It must not be implemented only for `goal_threat_outcome` or only for the top-level selected-evidence field.

### 4. Keep handle-domain recovery producer-owned

For a declared field-domain error:

- the repair prompt names the exact invalid field, expected allowlist, and typed `failure_kind`;
- the producing stage chooses a replacement value from the explicit allowlist;
- deterministic code never substitutes `self`, `current_user`, the origin handle, or any other value;
- the replacement is fully revalidated;
- after the attempt cap, the question is omitted through the existing contract-exhaustion route rather than represented by a normal empty `SemanticAppraisalResultV2`.

The `ev1` selected-role failure is the required regression test for this rule. `ev1` must never be admitted into the selected-role result, state, action, persistence, scheduling, dialog, or delivery paths.

### 5. Preserve bounded attempts and public failure semantics

- Reuse `SEMANTIC_APPRAISAL_ATTEMPT_LIMIT`; do not add a second semantic retry budget.
- Reuse the existing `semantic_appraisal_contract_exhausted` error code for exhausted producer-contract repairs; no new facade error code is required.
- Keep terminal `cognition_boundary_rejected` failures terminal and non-retryable.
- Keep provider, structural, prompt-cap, and state-incompatibility paths unchanged except for shared metadata normalization where required.
- Update `_collect_appraisals` to expose generic typed failure metadata for contract exhaustion and contained failures without requiring raw output.

The protected metadata record must include, at minimum:

```text
question_id
question_kind
failure_code
failure_kind
field_path (when applicable)
repair_attempted
attempt_count
retryable
disposition
```

The final disposition is one of the existing boundary outcomes plus `question_omitted` for exhausted degradable appraisal repair. Metadata must preserve the original typed cause and the final disposition; it must not report a successful empty appraisal as a substitute.

### 6. Keep downstream effects fail-closed

For terminal rejection or exhausted repair:

- no invalid candidate enters accepted appraisal state;
- no state reducer, action planner, authorization stage, scheduler, persistence, consolidation, surface, dialog, or adapter delivery consumes the invalid candidate;
- accepted-prefix behavior remains limited to already validated items;
- the failure capsule records the typed failure even when raw capture is disabled.

### 7. Update documentation and ownership in the same change

- Document the narrow producer-contract recovery boundary in `src/kazusa_ai_chatbot/cognition_core_v2/README.md`.
- Update `tests/ownership/source_test_impact_manifest.json` with exact source-to-test rows.
- Keep the completed plan immutable; this active bugfix plan is the superseding record for this disposition change.
- Keep protected trace exports and the human-readable RCA as diagnostic evidence, not runtime fixtures or production data.

## Deferred

- Broad population-wide retry-frequency measurement beyond the protected and adjacent replay cases.
- Prompt/model quality tuning unrelated to the typed contract boundary.
- New evaluator or critic stages.
- Changes to semantic question planning, model routing, relationship maintenance, dialog wording, adapters, persistence schema, scheduler behavior, or database migrations.
- Automatic remediation of other historical trace families not represented in the fixed disposition matrix.

## Target runtime behavior

The implementation target is:

```text
provider result
  -> canonical JSON/structural handling
  -> typed field-specific failure classification
       -> producer repair only for candidate_origin_missing or
          declared producer_handle_domain_invalid
       -> terminal fail-closed for all other deterministic boundaries
  -> complete deterministic validation
  -> accepted appraisal or existing contract exhaustion/question omission
```

The repaired `ck1` candidate follows:

```text
failed: candidate evidence [e3]
required origin: e1
replacement: [e1, e3]
selected evidence: contains e1 and e3
proposition/delta evidence: contains e1 and e3
full validation: required before admission
```

The `ev1` selected-role candidate follows:

```text
failed: selected role [ev1]
declared domain: [current_user, self]
replacement: model-selected value from the declared domain
runtime substitution: prohibited
exhaustion: semantic_appraisal_contract_exhausted + question_omitted
```

## Change surface and ownership

| Path | Owned symbols or records | Planned change |
|---|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py` | `_SemanticBoundaryValidationError`, `_appraise_semantic_item`, `_validate_semantic_boundary_candidate`, `_is_boundary_validation_error`, evidence-binding validators, `validate_semantic_appraisal_result` | Introduce typed field-specific failure classification, narrow repair eligibility, evidence-preservation checks, and typed attempt/disposition metadata. Remove broad substring routing. |
| `src/kazusa_ai_chatbot/cognition_core_v2/facade.py` | `_collect_appraisals` and failure-detail construction | Preserve existing error-code routing and add typed `failure_kind`, repair status, attempt count, retryability, and disposition for contained appraisal failures. |
| `tests/unit/cognition_core_v2/test_semantic_appraisal.py` | Existing appraisal attempt and boundary tests plus new typed-disposition tests | Prove bounded repair, evidence preservation, handle-domain producer ownership, terminal no-retry, exhaustion, and raw-off metadata. |
| `tests/unit/cognition_core_v2/test_facade.py` | Public appraisal collection contract | Prove question omission metadata and no normal empty-result substitution. |
| `tests/test_cognition_core_v2_failures.py` | Candidate proposition/delta binding and appraisal failure-cause tests | Preserve origin-binding rejection tests and assert typed failure metadata/cause chains. |
| `tests/test_cognition_core_v2_prompt_budget_continuity.py` | Appraisal origin prompt, accepted-prefix, and repair-budget tests | Prove origin context remains available and the existing cap is reused. |
| `tests/cognition_core_v2_appraisal_replay_harness.py` | `replay_appraisal_through_public_boundary` | Extend replay assertions for replacement preservation and downstream exclusion. |
| `tests/test_cognition_v2_protected_qq_replay.py` | Captured QQ replay tests | Change the current no-retry expectation to the narrow producer repair expectation while retaining no semantic-policy retry assertions. |
| `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py` | Existing adjacent live-LLM failure cases | Mark candidate-origin and declared producer-domain cases as requiring bounded repair; keep terminal families explicitly no-repair. Run individually and inspect artifacts. |
| `tests/test_cognition_core_v2_trace_failure_mode_matrix.py` | Deterministic failure-family matrix | Add the exact typed disposition for every listed adjacent family and assert terminal families do not issue repair calls. |
| `tests/ownership/source_test_impact_manifest.json` | `semantic_appraisal.py` and `facade.py` ownership rows | Add/update exact pytest node IDs for every changed semantic owner. |
| `src/kazusa_ai_chatbot/cognition_core_v2/README.md` | Appraisal admission and failure policy documentation | Record the narrow typed producer-contract recovery and terminal semantic-policy boundary. |
| `development_plans/README.md` | Active bugfix registry | Register this draft plan. |
| `development_plans/active/bugfix/cognition_v2_semantic_appraisal_boundary_recovery_bugfix_plan.md` | This change contract | Record scope, review findings, exact matrix, verification, and closure evidence. |

## Test impact and source-to-test traceability

The following node IDs are mandatory. New node IDs are fixed by this plan and must be created with these names; existing IDs are retained or explicitly renamed only where the expectation changes from broad no-retry to narrow producer repair.

| Changed source owner | Deterministic tests | Supplemental/public or live tests |
|---|---|---|
| `semantic_appraisal.py::_appraise_semantic_item` | `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_candidate_origin_repair_preserves_resolution_evidence_at_both_levels`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_candidate_origin_repair_is_bounded_and_exhaustion_is_question_omission`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_selected_role_handle_domain_repair_never_admits_disallowed_handle`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_terminal_boundary_classes_do_not_issue_repair_call`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_boundary_failure_metadata_is_typed_without_raw_output`; existing `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_recoverable_structure_uses_one_replacement_then_completes`; existing `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_unrecoverable_structure_exhausts_replacement_budget` | `tests/test_cognition_v2_protected_qq_replay.py::test_protected_qq_replay_repairs_candidate_origin_without_semantic_policy_retry`; `tests/test_cognition_v2_protected_qq_replay.py::test_protected_qq_replay_preserves_origin_and_resolution_evidence`; individual cases in `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_candidate_origin_evidence_missing_live_llm` and `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_selected_roles_unknown_handle_live_llm` |
| `semantic_appraisal.py::validate_semantic_appraisal_result` and evidence-binding validators | Existing `tests/test_cognition_core_v2_failures.py::test_candidate_proposition_rejects_mismatched_evidence`; existing `tests/test_cognition_core_v2_failures.py::test_candidate_delta_rejects_mismatched_evidence`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_candidate_origin_repair_preserves_resolution_evidence_at_both_levels`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_terminal_boundary_classes_do_not_issue_repair_call` | `tests/cognition_core_v2_appraisal_replay_harness.py::replay_appraisal_through_public_boundary` exercised by the protected replay nodes above |
| `facade.py::_collect_appraisals` | `tests/unit/cognition_core_v2/test_facade.py::test_appraisal_collection_records_typed_contract_exhaustion_metadata`; existing `tests/test_cognition_core_v2_failures.py::test_appraisal_collection_records_original_failure_cause` | `tests/test_cognition_core_v2_prompt_budget_continuity.py::test_appraisal_exhaustion_returns_the_accepted_prefix`; `tests/test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm.py::test_goal_threat_outcome_contract_exhaustion_live_llm` |
| terminal adjacent validator families | `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_selected_evidence_unknown_handle_is_rejected`; `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_selected_roles_unknown_handle_is_rejected`; `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_semantic_delta_path_not_owned_is_rejected`; `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_semantic_role_value_invalid_is_rejected`; `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_semantic_proposition_subject_kind_mismatch_is_rejected`; `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_semantic_proposition_object_handle_not_permitted_is_rejected`; `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_delta_reason_invalid_is_rejected`; `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_semantic_delta_type_invalid_is_rejected`; `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_terminal_event_transition_is_rejected`; `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_goal_evidence_handles_not_permitted_is_rejected`; `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_goal_role_handles_not_permitted_is_rejected` | `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_semantic_delta_path_not_owned_live_llm`; `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_terminal_event_transition_rejected_live_llm`; `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_semantic_role_value_invalid_live_llm`; `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_resolved_knowledge_gap_transition_rejected_live_llm`; `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_semantic_proposition_subject_kind_mismatch_live_llm`; `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_semantic_proposition_object_handle_not_permitted_live_llm`; `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_delta_reason_invalid_live_llm`; `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_semantic_delta_type_invalid_live_llm`; `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py::test_semantic_micro_appraisal_fields_not_exact_live_llm` retain `require_repair_call=False` unless they meet the exact declared producer-domain rule. |
| documentation and ownership manifest | `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_trace_inventory_contains_all_observed_contract_families`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_boundary_validation_preserves_existing_rejection_behavior` | Protected trace artifact and human-readable RCA inspection; source-to-test manifest validation in the repository ownership checks. |

Every node in the matrix is fully file-qualified. Implementation must preserve exact collection and ensure each node is discoverable by pytest.

## Verification procedure

The parent executes and records the following in order after implementation approval:

1. Capture `git status --short`, the approved plan status, and the exact changed-file set.
2. Run the focused deterministic appraisal and facade nodes:

   ```powershell
   venv\Scripts\python.exe -m pytest -q tests/unit/cognition_core_v2/test_semantic_appraisal.py tests/unit/cognition_core_v2/test_facade.py tests/test_cognition_core_v2_failures.py tests/test_cognition_core_v2_prompt_budget_continuity.py
   ```

3. Run the deterministic adjacent-failure matrix and protected replay tests:

   ```powershell
   venv\Scripts\python.exe -m pytest -q tests/test_cognition_core_v2_trace_failure_mode_matrix.py tests/test_cognition_v2_protected_qq_replay.py
   ```

4. Run the relevant failure-capsule/integration nodes, including the public boundary and accepted-prefix assertions:

   ```powershell
   venv\Scripts\python.exe -m pytest -q tests/test_cognition_core_v2_integration.py::test_appraisal_retry_or_omission_preserves_cognition tests/test_cognition_core_v2_integration.py::test_appraisal_internal_invariant_propagates tests/test_cognition_core_v2_prompt_budget_continuity.py::test_appraisal_exhaustion_returns_the_accepted_prefix
   ```

5. Run each live LLM failure-family case one at a time. Inspect the generated human-readable artifact after each case. Candidate-origin and declared producer-domain cases must show a bounded repair attempt or typed exhaustion; terminal families must show zero repair calls and no downstream action.
6. Replay the captured current trace through the public boundary. Verify that `ck1` retains `e3`, adds `e1`, and passes complete validation before admission. Verify that the invalid `ev1` handle is never admitted.
7. Verify protected metadata with raw capture disabled. Assert `failure_kind`, `repair_attempted`, `attempt_count`, `retryable`, and `disposition` are present for both repaired/exhausted and terminal cases.
8. Run the project’s Python style/lint and compile checks, then inspect `git diff --check` and the source-to-test impact manifest.
9. Confirm no `.env` was read or changed, no database migration was added, and no unrelated source or test files changed.

## Acceptance criteria

The bugfix is accepted only when all conditions hold:

1. The current protected failure no longer terminalizes the first `candidate_origin_missing` candidate solely because the validator raised a `ValueError`.
2. The `ck1` replacement preserves resolution evidence `e3` and adds origin evidence `e1` at both selected and nested candidate evidence levels.
3. A replacement that drops a valid prior citation is rejected and cannot enter state or downstream work.
4. The `ev1` selected-role candidate is handled only through the explicit producer-domain repair contract; runtime code never substitutes or admits it.
5. Exhausted recoverable repair produces the existing `semantic_appraisal_contract_exhausted` path with `question_omitted`, typed metadata, and no normal empty appraisal result.
6. Role enum, proposition-kind/subject-kind, unowned/duplicate delta, duplicate/conflict, permission/effect, and state/FSM failures remain terminal and issue no repair call.
7. Unknown or unmapped validator failures fail closed and do not enter any effectful or visible downstream path.
8. The facade and failure capsule expose typed metadata without raw-output capture.
9. All mandatory deterministic nodes pass; live LLM cases are run individually with inspected artifacts; the trace failure matrix and ownership manifest are complete.
10. The reused GPT-5.6 SOL reviewer performs a read-only implementation review at high reasoning and normal speed tier and returns a final disposition on the implemented target.
11. The plan was `draft` until the user approved execution and is now
`in_progress`. Closure requires implementation, verification, SOL implementation
review/final sign-off, committed changes if requested, and an updated archived
execution record.

## Execution evidence (2026-08-18)

Implementation is complete and remains parent-owned. The plan is in
`in_progress` pending the reused SOL implementation review and final
sign-off.

### Implemented surface

- `semantic_appraisal.py` now uses validator-owned typed failures. Only
  `candidate_origin_missing` and declared
  `producer_handle_domain_invalid` failures enter the existing bounded
  producer replacement loop.
- Replacement validation preserves valid selected and nested proposition/delta
  citations and rejects a replacement that drops them. Runtime code performs
  no evidence union, handle substitution, role rebinding, or semantic repair.
- Terminal semantic, unknown, structural, provider, prompt-cap, and
  state/FSM dispositions retain separate fail-closed routes.
- `facade.py` records raw-output-independent failure kind, field path, repair
  status, attempt count, retryability, and final disposition.
- The protected QQ replay, deterministic matrix, individual live cases,
  ownership manifest, and Cognition V2 README were updated in the same scope.

### Deterministic verification

The following parent-run commands passed:

```text
venv\Scripts\python.exe -m pytest -q tests/unit/cognition_core_v2/test_semantic_appraisal.py tests/unit/cognition_core_v2/test_facade.py tests/test_cognition_core_v2_failures.py::test_appraisal_collection_records_original_failure_cause tests/test_cognition_core_v2_failures.py::test_candidate_proposition_rejects_mismatched_evidence tests/test_cognition_core_v2_failures.py::test_candidate_delta_rejects_mismatched_evidence tests/test_cognition_core_v2_prompt_budget_continuity.py::test_appraisal_question_keeps_candidate_origin_contract tests/test_cognition_core_v2_prompt_budget_continuity.py::test_appraisal_exhaustion_returns_the_accepted_prefix tests/test_cognition_core_v2_prompt_budget_continuity.py::test_appraisal_repair_uses_residual_budget_for_second_call
# 37 passed

venv\Scripts\python.exe -m scripts.validate_test_impact --base-ref HEAD --run
# 30 exact impact-test nodes passed

venv\Scripts\python.exe -m pytest -q tests/test_cognition_core_v2_trace_failure_mode_matrix.py tests/test_cognition_v2_protected_qq_replay.py
# 31 passed; 1 unrelated pre-existing failure:
# test_relational_willingness_fields_not_exact_is_rejected

venv\Scripts\python.exe -m pytest -q tests/test_cognition_core_v2_integration.py::test_appraisal_retry_or_omission_preserves_cognition tests/test_cognition_core_v2_integration.py::test_appraisal_internal_invariant_propagates tests/test_cognition_core_v2_prompt_budget_continuity.py::test_appraisal_exhaustion_returns_the_accepted_prefix
# 2 relevant appraisal tests passed; 2 unrelated pre-existing failures in
# goal_cognition.py relational-willingness schema validation

venv\Scripts\python.exe -m ruff check --select I <changed Python files>
# passed

venv\Scripts\python.exe -m compileall -q <changed Python files>
# passed

git diff --check
# passed; Git emitted only LF-to-CRLF normalization warnings
```

The full focused command required by the original procedure also ran. Its
semantic-appraisal, facade, failure-cause, and prompt-appraisal nodes passed;
the remaining failures are existing surface/reducer/action/relational tests
whose source owners are outside this change. No production fix was added for
those unrelated failures.

### Current protected failure verification

The current trace `llmtrace_79651aa48cfd41d0a50c06343dbaa8db` was replayed
individually through the public boundary. The first `ck1 -> e1` failure is now
typed `candidate_origin_missing` with `producer_repair`, receives one bounded
replacement attempt, carries the valid `e3` citation in the preservation
contract, and reaches the existing
`semantic_appraisal_contract_exhausted`/`question_omitted` route when the local
model repeats `[e3]`. A deterministic protected replay supplies `[e1, e3]`
and proves that both selected and nested evidence sets are admitted only after
complete validation. No invalid candidate or downstream candidate-sensitive
effect is admitted.

### Generalized failure-family evidence

Each listed case was run as an individual `live_llm` node and its artifact was
inspected. The complete narrative and result table are preserved in
`test_artifacts/diagnostics/cognition_v2_semantic_appraisal_boundary_recovery_verification.md`.

| Case | Inspected artifact | Disposition |
|---|---|---|
| candidate origin missing | `test_artifacts/cognition_core_v2_appraisal_boundary/candidate_origin_evidence_missing_1787016967681157000.json` | one producer repair; exhaustion omits question |
| selected evidence unknown handle | `test_artifacts/cognition_core_v2_appraisal_boundary/selected_evidence_unknown_handle_1787017835640710700.json` | one producer repair; repaired result completes |
| unowned semantic delta path | `test_artifacts/cognition_core_v2_appraisal_boundary/semantic_delta_path_not_owned_1787017510531379400.json` | terminal; zero repair |
| terminal event transition | `test_artifacts/cognition_core_v2_appraisal_boundary/terminal_event_transition_rejected_1787017945555405300.json` | state incompatibility; zero semantic repair |
| invalid role enum/value | `test_artifacts/cognition_core_v2_appraisal_boundary/semantic_role_value_invalid_1787017960469102000.json` | terminal; zero repair |
| current-run invalid role value | `test_artifacts/cognition_core_v2_appraisal_boundary/current_run_event_agency_role_value_invalid_1787017976620887100.json` | terminal; zero repair |
| unowned knowledge-gap path | `test_artifacts/cognition_core_v2_appraisal_boundary/a1a573_goal_threat_unowned_knowledge_gap_path_1787018029761857300.json` | terminal; zero repair |
| resolved knowledge-gap transition | `test_artifacts/cognition_core_v2_appraisal_boundary/resolved_knowledge_gap_transition_rejected_1787018166504396000.json` | state incompatibility; zero semantic repair |
| selected-role producer handle domain | `test_artifacts/cognition_core_v2_appraisal_boundary/selected_roles_unknown_handle_1787018305459526500.json` | one producer repair; invalid handle not admitted |
| proposition subject-kind mismatch | `test_artifacts/cognition_core_v2_appraisal_boundary/semantic_proposition_subject_kind_mismatch_1787018321123562500.json` | terminal; zero repair |
| proposition object handle domain | `test_artifacts/cognition_core_v2_appraisal_boundary/semantic_proposition_object_handle_not_permitted_1787018350031264200.json` | one producer repair; invalid candidate excluded |
| invalid delta reason shape | `test_artifacts/cognition_core_v2_appraisal_boundary/delta_reason_invalid_1787018530228619400.json` | structural recovery; no semantic-policy retry |
| invalid delta type shape | `test_artifacts/cognition_core_v2_appraisal_boundary/semantic_delta_type_invalid_1787018603230263300.json` | structural recovery; no semantic-policy retry |
| micro-appraisal wrapper shape | `test_artifacts/cognition_core_v2_appraisal_boundary/semantic_micro_appraisal_fields_not_exact_1787018624800049800.json` | structural recovery; no semantic-policy retry |

The selected-role live fixture contains a second simultaneous invalid subject
field; validator order surfaces that typed producer-domain failure first. The
exact `selected_role_handles` domain is covered by the deterministic matrix.
The selected-evidence live fixture uses a canonical parsed candidate with the
captured unknown handle represented as `e999` because the original wrapper is
structurally malformed; the structural and semantic boundaries remain
separately covered.

### Review and closure status

- Parent implementation, test changes, verification, and evidence capture are
  complete.
- The initial reused GPT-5.6 SOL implementation review returned
  `REQUEST_CHANGES` for all-domain evidence preservation and explicit unknown
  validator classification. The parent implemented both findings and added
  deterministic coverage plus the four affected live reruns documented in the
  verification artifact.
- Reused GPT-5.6 SOL final implementation sign-off returned `APPROVE` with no
  remaining implementation blockers. The reviewer confirmed the unrelated
  relational-willingness failure is outside the changed source.
- No database migration, `.env` read, adapter change, scheduler change, or
  unrelated production source change was made.

## Final closure record (2026-08-18)

- SOL reviewer: reused GPT-5.6 SOL thread, high reasoning, normal speed tier;
  final disposition `APPROVE` after remediation.
- Remediation: all producer-handle-domain repairs now preserve valid evidence
  at stable selected/nested item-field positions; dropped citations are
  terminally rejected; unmapped validator failures are typed
  `unknown_validation_failure` and receive no repair.
- Deterministic verification: 22 semantic-appraisal unit tests passed, 33
  exact source-impact nodes passed, and the protected replay plus adjacent
  matrix passed all 31 in-scope nodes. The single matrix failure is the
  unchanged `goal_cognition.py` relational-willingness schema-message fixture.
- Live verification: the current protected trace and all generalized failure
  families were run individually with inspected artifacts. The four affected
  producer-domain reruns are recorded in the verification artifact under
  `Remediation reruns (2026-08-18)`.
- Static verification: compileall, Ruff import-order checks, JSON validation,
  and `git diff --check` passed. Git emitted only line-ending normalization
  warnings.
- Commit: recorded in the implementation commit that closes this plan.
- The unrelated untracked
  `development_plans/active/short_term/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md`
  remains untouched.

## Closure record requirements

At closure, the plan must append or record:

- the implementation commit hash;
- exact changed files and source-to-test manifest updates;
- deterministic test commands and outcomes;
- each live LLM case and inspected artifact path;
- current-trace replay outcome;
- SOL’s final implementation review disposition and remediation evidence, using the same subagent thread;
- final `git status --short` and `git diff --check` results;
- the move from `active/bugfix/` to `archive/completed/` only after all acceptance criteria pass.
