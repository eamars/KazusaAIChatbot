# required-selection prompt contract and evaluator relaxation bugfix

## Summary

- Goal: correct the required-selection model-facing contract and apply the
  RCA-authorized evaluator relaxation so valid required selections survive
  bounded regeneration.
- Status: completed
- Scope boundary: required-selection prompt construction, required-selection
  repair feedback, selected-operation binding in `goal_cognition.py`, its
  direct owner tests, prompt/live harness assertions, Cognition V2 contract
  documentation, ownership manifest, and one human-readable before/after
  review artifact.
- Change direction: expose only the operation and currently unresolved
  embedded endpoints as writable model fields; preserve code-owned carrier
  values; accept equal redundant known endpoint fields; reject conflicting or
  unknown fields with precise errors; accept usable operation wording equal to
  the authoritative input wording.
- Acceptance state: implementation authorized by the user and executable by
  the plan-scoped fixed Luna/max/default-speed executor; completion requires
  exact mapped deterministic evidence, source-impact validation, CJK syntax
  evidence, one inspected post-change live case when the LLM is available or
  a recorded blocker, a human-readable before/after review, and lifecycle
  archival.

## Confirmed Decisions

1. The production failure is addressed as a prompt-contract construction defect
   with secondary local-model contract-adherence weakness. The existing
   deterministic semantic ownership boundary remains authoritative.
2. `operation` is always writable for a selected response operation. Each
   embedded endpoint is writable only when its authoritative input value is
   `无`. Known `embedded_actor_role` and `embedded_target_role` values remain
   code-owned and are omitted from the writable model-field inventory.
3. A redundant known embedded endpoint is accepted only when its value exactly
   equals the authoritative input value; the returned operation is always
   canonicalized from the authoritative input carrier.
4. Conflicting known embedded endpoints receive field-specific expected/actual
   errors. Unknown fields, malformed values, fixed carrier fields, role values,
   evidence, bounds, parser behavior, retry caps, routes, and fail-closed
   behavior remain strict.
5. Selected operation wording may equal authoritative input wording when it is
   otherwise valid. The validator does not compare or rewrite semantic prose.
6. The prompt and repair contract use one canonical selected-operation contract
   projection. Repair feedback repeats the exact writable field list and
   code-owned values as concise structured facts and does not invoke another
   LLM.
7. The cutover is big-bang. The legacy broad endpoint inventory, exact-string
   rejection, and generic known-field error are replaced directly without
   aliases, compatibility paths, or parallel vocabularies.
8. The sole modification executor is Luna subagent `01a0006b-d28f-7852-a7ae-
   bcb0adf01fc` (nickname `Nash`) using the user-fixed model constraint
   `gpt-5.6-luna`, reasoning effort `max`, speed `normal/default`. The parent
   performs read-only review and verification after implementation; no other
   executor authors any file change.

## Root-Cause Evidence

- Source trace: `test_artifacts/diagnostics/llmtrace_680df1e7b3f94ac1aea297b8c560c8eb.json`.
- RCA: `test_artifacts/diagnostics/llmtrace_680df1e7b3f94ac1aea297b8c560c8eb_rca.md`.
- The authoritative trace operation had a known actor and unresolved target,
  while the dynamic contract advertised both endpoint fields as optional.
- The model repeated the wrapper operation and then oscillated between the
  exact-repeat and known-endpoint errors across both ordinary-response and
  self-improvement attempts.
- The RCA found no validator false positive; the fix therefore narrows the
  model-visible contract and removes only the redundant endpoint and lexical
  difference barriers explicitly authorized by the user.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| Required-selection prompt contract | bigbang | Replace the broad endpoint inventory with the exact per-input writable projection and concise positive instructions. |
| Repair feedback | bigbang | Replace low-salience generic facts with the canonical writable/code-owned projection. |
| Selected-operation evaluator | bigbang | Accept equal redundant known endpoints, reject conflicts precisely, and remove exact-string rejection. |
| Tests and documentation | bigbang | Replace assertions and prose that encode the removed contract. |

## Mandatory Skills

- `development-plan` with `plan_contract.md`, `cutover_policy.md`, and
  `execution_gates.md` for lifecycle, scope, traceability, cutover, and
  evidence.
- `local-llm-architecture` for short positive prompts, local-model context
  shaping, semantic ownership, and bounded recovery.
- `llm-trace-debug` for protected trace evidence and retry inspection.
- `debug-llm` for the Luna-authored before/after review artifact and live
  quality judgment.
- `py-style` plus `references/positive_constraints.md` and
  `references/negative_constraints.md` for all Python edits.
- `cjk-safety` for every CJK-bearing Python edit and immediate syntax checks.
- `test-style-and-execution` for deterministic owner tests, live one-at-a-time
  execution, durable artifacts, and inspection criteria.

## Mandatory Rules

- Modify only the explicitly owned files in this plan and preserve the
  pre-existing untracked `src/scripts/clear_internal_monologue_residue_state.py`
  byte-for-byte.
- Keep `response_owner_role`, `selection_owner_role`, and
  `selection_required` code-owned. The only newly accepted redundant model
  fields are matching known embedded endpoints.
- Keep LLM semantic ownership in the goal stage. Deterministic code validates
  shape, ownership, roles, evidence, bounds, and canonical carrier binding;
  it does not extract verbs, infer endpoints from prose, rewrite operation
  wording, or assign semantic roles.
- Use canonical `parse_llm_json_output(...)` and the existing deterministic-only
  required-selection parser path.
- Keep prompt constants triple-single-quoted, concise, positive, and adjacent
  to their existing handler boundary. Render dynamic contracts as structured
  JSON facts already owned by the goal producer.
- Run `venv\Scripts\python` AST or `py_compile` syntax validation immediately
  after every CJK-bearing Python edit.
- Keep production and test changes surgical; no schema, route, retry-budget,
  persistence, dialog, resolver, episode, or adapter changes are in scope.

## Must Do

1. Project the exact selected-operation writable fields from the authoritative
   required operation into the dynamic goal contract: `operation` always;
   unresolved actor and/or target endpoints only when their input value is
   `无`.
2. Project the code-owned selected-operation values and exact writable fields
   into repair feedback so every regeneration receives the same concise facts.
3. Shorten and align required-selection initial, recurrence, active-branch, and
   repair instructions around one concrete selected action and the exact
   dynamic field contract. Allow equal operation wording when valid.
4. Relax `_bind_selected_response_operation` only as specified: accept equal
   redundant known endpoint values, canonicalize from input, reject conflicts
   with field-specific expected/actual errors, reject unknown/fixed fields and
   malformed values, and preserve all other validation behavior.
5. Add direct owner tests for exact writable-field projection, matching
   redundant endpoint canonicalization, conflicting endpoint rejection, equal
   operation wording acceptance, and unknown-field rejection.
6. Update prompt and README consistency coverage plus the production-shaped
   live required-selection harness to inspect the exact dynamic contract and
   permit equal usable wording.
7. Create the human-readable before/after review artifact before post-change
   live verification is complete, then update it with the real post-change
   evidence and judgment.
8. Update the ownership manifest with every new exact owner node required for
   the changed production source.
9. Record execution evidence, archive the completed plan, and update the plan
   registry.

## Deferred

- Changes to `cognition_episode.py`, dialog, schemas, routes, retry budgets,
  persistence, resolver, action planning, surface planning, or delivery.
- Any deterministic semantic parsing, keyword extraction, operation rewriting,
  endpoint inference, or evaluator-authored semantic repair.
- New aliases, compatibility shims, alternate vocabularies, fallback paths,
  additional LLM calls, model routing changes, or attempt-cap changes.
- Broad prompt shortening outside required-selection contracts.
- Changes to the unrelated untracked maintenance script.

## Target State

### Model-visible selected-operation contract

For the current authoritative input operation, the dynamic contract contains
one canonical `selected_response_operation` description with:

- `writable_fields`: `operation` plus only endpoint fields whose input value is
  `无`;
- `required_fields`: `operation`;
- `optional_fields`: the unresolved endpoint subset;
- field types and allowed role values for writable endpoint fields;
- `code_owned_fields`: `response_owner_role`, `selection_owner_role`,
  `selection_required`, and every known embedded endpoint with its exact
  authoritative value; and
- a short positive rule to state one concrete selected action while preserving
  those code-owned values.

Repair feedback carries this same selected-operation contract projection as
structured facts alongside the original validation error and exact top-level
goal contract. It keeps the current writable keys and authoritative values
salient on every retry.

### Deterministic evaluator

`_bind_selected_response_operation` accepts `operation` and the exact unresolved
endpoint fields. It also accepts a known endpoint repeated with the exact
authoritative value, then binds every carrier field from the authoritative
operation. A conflicting known endpoint error names that endpoint and its
expected/actual values. Unknown fields and fixed carrier fields remain errors.
The selected operation text is validated for type and bounds but is not
compared lexically with the input operation.

## Execution Roles

### implementation_owner

- Responsibility: Luna subagent `01a0006b-d28f-7852-a7ae-bcb0adf01fc`
  (nickname `Nash`) authors and verifies the complete scoped prompt-contract
  and evaluator bugfix.
- Owned surface: all files listed under `Change Surface`; no other path may be
  edited.
- Authority: may implement the fixed target contract, update owned tests,
  documentation, manifest, plan lifecycle, and review artifact; may run
  read-only inspection and verification commands.
- Applicable skills: every skill in `Mandatory Skills`.
- Capability floor: production Python prompt/evaluator editing, local-model
  contract design, CJK-safe editing, exact pytest ownership mapping, trace
  evidence review, and live artifact inspection.
- Independence requirement: the fixed Luna subagent performs all modifications;
  the parent review is read-only and separate from implementation acceptance.
- Acceptance output: scoped diff, exact deterministic owner evidence, source
  impact evidence, syntax evidence, one inspected live case or blocker, updated
  human-readable review, and archived execution record.
- Gate: baseline state and hashes captured; this plan is `in_progress`; no
  unresolved scope or contract decision remains.
- Plan-scoped fixed execution constraint: Luna subagent
  `01a0006b-d28f-7852-a7ae-bcb0adf01fc` (nickname `Nash`) on
  `gpt-5.6-luna`, reasoning effort `max`, speed `normal/default`, explicitly
  fixed by the user. Only the user may change this constraint.

### parent_read_only_review

- Responsibility: inspect the final diff, tests, artifacts, and lifecycle
  evidence and report acceptance or residual risk.
- Owned surface: read-only inspection of owned files and generated evidence.
- Authority: may run verification and identify findings; may not author any
  implementation, test, documentation, manifest, plan, or review-artifact
  change.
- Applicable skills: `development-plan`, `py-style`, `cjk-safety`,
  `test-style-and-execution`, and `debug-llm`.
- Capability floor: source-to-test traceability, contract review, prompt-quality
  judgment, and evidence inspection.
- Independence requirement: separate from the implementation edits.
- Acceptance output: read-only review verdict and residual-risk statement.
- Gate: implementation owner has completed all required evidence and the
  worktree remains within the owned-file boundary.

## Test Impact And Traceability

| Repository path | Changed symbol or contract | Semantic owner | Exact deterministic pytest node IDs | Supplemental live node IDs | Test mode | Regression prevented |
| --- | --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` | required-selection dynamic contract projection, repair feedback, and selected-operation binder | required-selection goal cognition and deterministic contract boundary | `tests/unit/cognition_core_v2/test_goal_cognition.py::test_required_selection_prompt_projects_exact_writable_endpoint_fields`; `tests/unit/cognition_core_v2/test_goal_cognition.py::test_required_selection_accepts_matching_known_endpoint_and_canonicalizes`; `tests/unit/cognition_core_v2/test_goal_cognition.py::test_required_selection_rejects_conflicting_known_endpoint_with_field_error`; `tests/unit/cognition_core_v2/test_goal_cognition.py::test_required_selection_accepts_authoritative_operation_text`; `tests/unit/cognition_core_v2/test_goal_cognition.py::test_required_selection_rejects_unknown_operation_field` | `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_projects_exact_writable_fields` | deterministic owner plus one-at-a-time live LLM | Prevents broad model-visible endpoint inventory, retry oscillation, false rejection of matching redundant endpoints, and rejection of usable equal operation wording while preserving ownership and unknown-field rejection. |
| `src/kazusa_ai_chatbot/cognition_core_v2/README.md` | required-selection writable endpoint and canonicalization documentation | Cognition V2 ICD | `tests/test_cognition_core_v2_prompt_contract_guidance.py::test_selected_response_operation_contract_is_documented` | none | deterministic documentation contract | Prevents documentation and prompt guidance from drifting from runtime ownership. |
| `tests/test_cognition_core_v2_prompt_contract_guidance.py` | prompt/repair contract consistency checks | prompt-contract test owner | `tests/test_cognition_core_v2_prompt_contract_guidance.py::test_required_selection_contract_projects_exact_fields_and_retry_facts` | none | deterministic prompt contract | Prevents the model-facing contract from re-advertising forbidden fields or hiding code-owned values. |
| `tests/test_cognition_core_v2_required_selection_live_llm.py` | production-shaped dynamic contract inspection and equal-wording live gate | required-selection live evidence owner | `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_harness_contract_inspection_is_exact` | `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_projects_exact_writable_fields` | deterministic harness plus one-at-a-time live LLM | Keeps live evidence aligned with the exact per-input contract and makes the post-change quality judgment inspectable. |
| `tests/ownership/source_test_impact_manifest.json` | exact source-to-test ownership registration | verification contract owner | `tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary` | none | deterministic manifest validation | Prevents a changed production source from bypassing its exact owner nodes. |

## Change Surface

### Delete

None.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: implement the
  exact dynamic contract, concise retry facts, matching-endpoint relaxation,
  field-specific conflict errors, and equal-wording acceptance.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`: document exact writable
  endpoints, redundant matching canonicalization, conflict rejection, equal
  wording acceptance, and continued LLM semantic ownership.
- `tests/unit/cognition_core_v2/test_goal_cognition.py`: add direct owner tests
  for every new deterministic behavior.
- `tests/test_cognition_core_v2_prompt_contract_guidance.py`: assert prompt,
  contract, repair, and README consistency.
- `tests/test_cognition_core_v2_required_selection_live_llm.py`: update dynamic
  contract inspection and add one production-shaped live case.
- `tests/ownership/source_test_impact_manifest.json`: register the exact owner
  nodes.
- `development_plans/README.md`: register the active plan and later its
  completed archive record.
- `development_plans/active/bugfix/required_selection_prompt_contract_and_evaluator_relaxation_plan.md`:
  record execution checkpoints and evidence before archival.

### Create

- `test_artifacts/llm_reviews/llmtrace_680df1e7b3f94ac1aea297b8c560c8eb_required_selection_prompt_contract_review.md`:
  Luna-authored before/after review with production trace evidence and the
  inspected post-change live artifact or explicit availability blocker.
- `development_plans/archive/completed/required_selection_prompt_contract_and_evaluator_relaxation_plan.md`:
  completed immutable execution record after acceptance.

### Keep

- `src/scripts/clear_internal_monologue_residue_state.py` exactly unchanged.
- `src/kazusa_ai_chatbot/cognition_episode.py`, schemas, routes, retry
  budgets, persistence, dialog, resolver, action planning, and delivery.
- Canonical parser, role/evidence validation, bounds, provenance, route
  selection, attempt ledger, and fail-closed behavior outside this contract.

## Agent Autonomy Boundaries

The implementation owner may choose local code arrangement, prompt wording,
test assertion decomposition, artifact formatting, and command order within
the fixed target state and owned file list. The implementation owner must
preserve semantic LLM ownership, deterministic validation boundaries, exact
field projection, fixed code-owned values, and all exclusions. A need to edit
another path, change a schema or route, add semantic parsing, alter attempt
budgets, or introduce compatibility behavior requires a plan amendment and
user decision before editing; no such decision is open in this plan.

## Verification

1. Baseline recorded before implementation: status `??
   src/scripts/clear_internal_monologue_residue_state.py`, HEAD
   `9cfe8f3d3aa575f5d215dcc00e82c6c89a0e3e8b`, and owned-file SHA-256 hashes
   recorded in `Execution Evidence`.
2. Read the active plan back and review scope, ownership, target contract,
   exact nodes, and no-open-decision status before production edits.
3. After every CJK-bearing Python edit, run the immediate `venv\Scripts\python`
   AST/`py_compile` syntax check.
4. Collect and run the exact new deterministic owner nodes plus the mapped
   existing `test_goal_cognition.py` and prompt-contract nodes.
5. Run `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run`
   and require exact mapped production nodes to collect and pass.
6. Create the human-readable review artifact before live verification is
   complete. Inspect the production trace and deterministic post-change facts
   in the artifact.
7. Run the new live case individually with `-m live_llm -q -s` when its
   configured LLM is available. Inspect its durable raw artifact before any
   later live case. If unavailable, record the exact environment blocker and
   retain deterministic evidence.
8. Review the final diff for owned scope, untracked-file preservation, prompt
   adjacency, CJK safety, manifest coverage, contract/documentation parity,
   and absence of forbidden semantic rewriting.
9. Update this plan with commands, results, artifact paths, residual risks,
   and final status; copy it to the completed archive and update the registry.

## Acceptance Criteria

- The exact writable endpoint projection is covered by a collected and passing
  deterministic owner node.
- Matching redundant known endpoints are accepted and canonicalized; conflicting
  known endpoints are rejected with field-specific expected/actual errors.
- Unknown and fixed code-owned fields remain rejected, and malformed values
  remain rejected by the existing typed validators.
- Authoritative operation wording may be accepted when valid; the removed exact
  lexical-difference rejection is absent from source, tests, and documentation.
- Repair feedback contains the exact writable field list and code-owned values
  for the current input without another LLM call.
- Prompt and README consistency tests pass.
- `validate_test_impact --base-ref HEAD --run` passes with every changed
  production source mapped to collected, passing exact deterministic nodes.
- One post-change production-shaped live case is run individually and its raw
  artifact is inspected, or the exact unavailable-LLM blocker is recorded.
- The human-readable review compares the production failure against
  post-change deterministic/live evidence.
- The diff contains only owned files; the unrelated untracked script is
  preserved exactly.
- The active plan is completed and archived, and `development_plans/README.md`
  points to the completed record.

## Progress Checklist

- [x] Mandatory repository docs, source, tests, historical plan, skills, and
  RCA artifacts read.
- [x] Baseline status, HEAD, owned-file set, and hashes captured.
- [x] User implementation authorization and fixed Luna/max/default-speed
  constraint recorded.
- [x] Closed executable plan reviewed with no open decisions.
- [x] Prompt contract and repair feedback implemented.
- [x] Evaluator relaxation implemented.
- [x] Direct owner, prompt-consistency, harness, and manifest tests updated.
- [x] Immediate CJK syntax checks completed after Python edits.
- [x] Human-readable before/after review artifact completed.
- [x] Exact mapped deterministic tests and source-impact validation passed.
- [x] Individual post-change live case inspected or blocker recorded.
- [x] Final scope review completed; plan archived and registry updated.

## Execution Evidence

- Baseline HEAD: `9cfe8f3d3aa575f5d215dcc00e82c6c89a0e3e8b`.
- Baseline worktree: `?? src/scripts/clear_internal_monologue_residue_state.py`;
  this file is outside the plan and remains byte-for-byte preserved.
- Baseline owned-file hashes: `development_plans/README.md`
  `CB3F96FE76EF76F25AEB39C00A71CB6A1987C1CF6A405E64C7F38B3E0EE3256E`;
  `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
  `FC3452BB4EE4436D398371AD86F2D26C0E1769DF34DFE03167D017DF19FE945E`;
  `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
  `AFA18A01764FFB74497F96A305AA0340CE5D76C81EA79080C05485AFA426B1B8`;
  `tests/unit/cognition_core_v2/test_goal_cognition.py`
  `5861837EEBEE2C5BE3E635DF906E97AA84602440749238F7255077E3A978DB3A`;
  `tests/test_cognition_core_v2_prompt_contract_guidance.py`
  `D567712E95A7FA1B5473BA4242406DCCF9AE44F64BC772327EBBA61EB8EF0DB4`;
  `tests/test_cognition_core_v2_required_selection_live_llm.py`
  `B76A3FA9422728294C62C762EC3F5F4BCFEF225192F3820AF9B94499C2007510`;
  `tests/ownership/source_test_impact_manifest.json`
  `CB9F977DE78DFDE5FFDE3232773689A4217A39B83F9C5818589EABE81B8DE792`.
- Missing at baseline and created by this plan: both lifecycle plan paths and
  the human-readable review artifact.
- Execution owner: Luna subagent `01a0006b-d28f-7852-a7ae-bcb0adf01fc`
  (nickname `Nash`), fixed `gpt-5.6-luna`, max reasoning, normal/default speed;
  the parent is read-only and authors no file modifications.
- Source RCA review: `test_artifacts/diagnostics/llmtrace_680df1e7b3f94ac1aea297b8c560c8eb_rca.md`.
- Implemented source contract: `goal_output_contract.selected_response_operation`
  now projects exact per-input writable fields and code-owned values; repair
  feedback carries the same projection as structured facts.
- Implemented evaluator relaxation: matching known endpoints are accepted and
  canonicalized, conflicting endpoints produce field-specific expected/actual
  errors, unknown/fixed fields remain rejected, and usable equal operation
  wording is accepted.
- Deterministic owner evidence: the mapped batch passed 33 tests, including
  `tests/unit/cognition_core_v2/test_goal_cognition.py` (17),
  `tests/test_cognition_core_v2_prompt_contract_guidance.py` (15), and
  `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_harness_contract_inspection_is_exact` (1).
- Syntax evidence: immediate AST checks after each CJK Python edit and final
  `venv\Scripts\python -m py_compile` for all four changed Python files passed.
- Initial live evidence: the individual command
  `venv\Scripts\python -m pytest -o addopts='' -m live_llm tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_projects_exact_writable_fields -q -s`
  passed in 9.17 seconds with one accepted model call. The raw artifact is
  `test_artifacts/llm_traces/cognition_core_v2_required_selection_live_llm__known_actor_unresolved_target_contract.json`.
- Initial live inspection: contract diagnostics and selected-operation diagnostics
  passed; the model returned the authoritative operation wording unchanged,
  supplied only the unresolved target field, and the attempt ledger accepted
  local attempt 1 of 3. The focused prompt was 3,963 characters and its
  dynamic payload was 4,420 characters.
- Source-impact evidence: the first validator invocation identified the
  baseline pre-existing untracked maintenance script as outside the manifest.
  The script was then rerun with only that explicit file temporarily held
  outside the repository, restored byte-for-byte with SHA-256
  `019D5DCD55FD25D4933B6FBFDB11252B6EF08E7DA1A7433B51CEFA2114EB08E7`, and
  `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run`
  passed 14 exact impacted nodes.
- Review artifact: `test_artifacts/llm_reviews/llmtrace_680df1e7b3f94ac1aea297b8c560c8eb_required_selection_prompt_contract_review.md`.
- Initial final scope review: only the owned tracked paths changed; the unrelated
  untracked maintenance script remains unchanged and the parent remains
  read-only.

## Corrective Pass Evidence

This completed plan record was amended by the sole modification executor, Luna
subagent Nash (`01a0006b-d28f-7852-a7ae-bcb0adf01fc`), under the fixed
`gpt-5.6-luna` / max-reasoning / normal-default-speed constraint. The parent
session remained read-only.

- Corrected the remaining structured-contract contradiction: top-level
  `field_types.selected_response_operation` is now
  `per_input_writable_selected_response_operation`, pointing to the exact
  nested per-input writable contract. Unit, prompt-contract, and live-harness
  assertions reject the old full-shape descriptor.
- Revised the production-shaped live fixture so the authoritative operation is
  the concrete action `当前角色与当前用户一起去安静的地方散步并聊最近的心情`.
  This makes the observed equal wording semantically honest while preserving
  LLM semantic ownership and avoiding deterministic semantic parsing.
- Corrected the Cognition README sentence to begin `Matching known endpoint
  values`.
- Corrective deterministic command:
  `venv\Scripts\python -m pytest tests/unit/cognition_core_v2/test_goal_cognition.py tests/test_cognition_core_v2_prompt_contract_guidance.py tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_harness_contract_inspection_is_exact -q`
  — 33 passed.
- Corrective live command:
  `venv\Scripts\python -m pytest -o addopts='' -m live_llm tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_projects_exact_writable_fields -q -s`
  — 1 passed in 8.72 seconds.
- Corrective raw artifact inspected:
  `test_artifacts/llm_traces/cognition_core_v2_required_selection_live_llm__known_actor_unresolved_target_contract__20260814T135743565613Z.json`.
  It records one accepted call, the corrected top-level descriptor, exact
  writable fields `operation` and `embedded_target_role`, code-owned actor
  `当前角色`, concrete authoritative wording returned unchanged, target
  `当前用户`, empty parse/validation errors, and local attempt 1 accepted of
  3. Prompt and dynamic payload lengths were 3,963 and 4,459 characters.
- Corrective source-impact command
  `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run`
  collected and passed 14 exact mapped owner nodes. The pre-existing
  untracked maintenance script was held outside the repository only for this
  validator invocation and restored byte-for-byte with SHA-256
  `019D5DCD55FD25D4933B6FBFDB11252B6EF08E7DA1A7433B51CEFA2114EB08E7`.
- The strict manifest boundary test passed:
  `venv\Scripts\python -m pytest tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary -q`.
- Final corrective scope review: only the existing plan-owned paths changed;
  the unrelated untracked maintenance script remains byte-for-byte preserved,
  the completed plan remains archived, and the parent remains read-only.
- The human-readable review artifact was updated with this corrective evidence
  and the archived plan remains completed with no open decisions.
