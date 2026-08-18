# Cognition V2 semantic-admission original-contract restoration

## Summary

- Goal: restore the semantic-appraisal producer boundary to the approved original contract after the validator-policy and boundary-recovery regressions.
- Status: completed
- Scope boundary: semantic-appraisal admission, its direct deterministic tests and protected replay, the exact source-test ownership row, and current Cognition V2 documentation.
- Change direction: one surgical big-bang replacement of the regressed runtime admission path; no compatibility path.
- Acceptance state: the user explicitly commanded implementation and fixed GPT-5.6 Luna at max reasoning and normal speed as the code-change executor.

## Scope And Change Direction

The original completed plan requires structurally usable semantic content to consume zero semantic-validation retries. Runtime producer admission therefore consists of canonical JSON parsing, singular-item canonicalization, structural normalization, deterministic boundary/state trial, and either acceptance or safe family termination. Semantic ownership and semantic-coherence policy do not participate in runtime admission.

The current implementation violates that direction in two ways: it routes every normalized candidate through the combined semantic validator, and its follow-up repair path requires preservation of citations that candidate-origin validation may require replacing. The fix removes those runtime behaviors while preserving structural replacement, provider retry, prompt-cap, deterministic provenance, target, state/FSM, permission, persistence, action, and delivery ownership.

## Mandatory Skills

- development-plan governs lifecycle, execution evidence, traceability, and closure.
- local-llm-architecture governs the local-model failure and graceful-degradation boundary.
- py-style governs every Python edit and review.
- test-style-and-execution governs deterministic and replay tests.
- debug-llm governs the final human-readable before/after trace review artifact.

## Mandatory Rules

- Preserve the pre-existing untracked `development_plans/active/short_term/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md`.
- Keep prompts, model routing, retry limits, reducers, facade, state schemas, persistence, action authorization, surfaces, and delivery unchanged.
- Use the canonical parser and current structural normalizer.
- Keep malformed producer structure inside the current bounded replacement policy.
- Keep provider and prompt-cap dispositions unchanged.
- Do not add a semantic evaluator, semantic retry, citation union, evidence substitution, compatibility shim, feature flag, or new model call.

## Must Do

1. Remove the combined semantic validator from `_appraise_semantic_item` runtime admission.
2. Split deterministic boundary-carrier validation from semantic ownership checks so the strict standalone validator can remain available without governing producer admission.
3. Keep handle domains, evidence provenance, role enums, target paths, bounded deltas, duplicate conflicts, and state/FSM trial checks fail-closed before accepted state reaches downstream work.
4. Let structurally valid proposition labels and proposition-kind/subject semantic combinations pass runtime admission as opaque producer semantics; state/FSM trial remains authoritative for effect compatibility.
5. Remove candidate-origin and handle-domain producer repair eligibility and remove candidate-evidence preservation capture/validation.
6. Terminate the current appraisal family safely on a deterministic boundary rejection: retain an already accepted prefix or return the canonical empty family result. Spend exactly one model call for the rejected item.
7. Keep structural replacement and provider retry behavior unchanged.
8. Update current documentation and exact source-test ownership to the restored contract.
9. Isolate structural-output replacement behind a dedicated structural error so unexpected boundary or reducer defects propagate without consuming a model retry.
10. Replace stale deterministic, replay-harness, and live-suite declarations that still require semantic repair or semantic contract exhaustion.

## Deferred

- Prompt changes or model tuning.
- Changes outside semantic-appraisal admission.
- Relationship-maintenance, migration, persistence, goal, action, surface, or delivery changes.
- Population-level quality or latency measurement.
- Live database operations.

## Target State

```text
provider response
  -> canonical JSON parser
  -> singular-item canonicalization
  -> structural normalization
  -> deterministic boundary carriers and state/FSM trial
       accepted              -> merge item
       boundary/state reject -> retain accepted prefix or empty family
  -> no semantic validator and no semantic producer repair
```

The strict `validate_semantic_appraisal_result(...)` helper may continue to serve direct deterministic contract tests. `_appraise_semantic_item(...)` must not invoke it, directly or through an alias.

Candidate-origin mismatch, unknown evidence or role handles, unauthorized target paths, invalid role enums, duplicate boundary carriers, and state/FSM incompatibility all consume zero repair calls. They authorize no mutation and end only the affected appraisal family. Structurally malformed envelopes retain the existing complete-replacement attempt.

## Execution Roles

### Parent coordinator and verification owner

- Responsibility: own this plan, baseline, Luna instruction, diff review, deterministic verification, replay review artifact, lifecycle evidence, and closure.
- Owned surface: development plan and registry, verification commands, diagnostic Markdown, review and closure records.
- Authority: direct Luna, reject or remediate its patch, and close the plan after evidence passes.
- Applicable skills: development-plan, local-llm-architecture, py-style, test-style-and-execution, debug-llm.
- Capability floor: full repository inspection, exact-node verification, trace comparison, and production diff review.
- Independence requirement: the parent may review and integrate but does not replace the fixed Luna implementation assignment.
- Acceptance output: reviewed diff, exact test evidence, before/after trace review, and completed lifecycle record.
- Gate: starts from commit `7505e78914655d028fdf248d9048acb2b5b2402f`; exits only when every acceptance criterion is evidenced.

### GPT-5.6 Luna implementation owner

- Fixed execution constraint: GPT-5.6 Luna, max reasoning, normal speed; only the user may change this assignment.
- Responsibility: implement the scoped production, test, manifest, and current-documentation changes test-first.
- Owned surface: `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`, `src/kazusa_ai_chatbot/cognition_core_v2/README.md`, `tests/unit/cognition_core_v2/test_semantic_appraisal.py`, `tests/unit/cognition_core_v2/test_facade.py`, `tests/test_cognition_v2_protected_qq_replay.py`, `tests/test_cognition_core_v2_semantic_terminalization.py`, `tests/test_cognition_core_v2_prompt_budget_continuity.py`, `tests/test_cognition_core_v2_integration.py`, `tests/test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm.py`, `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py`, `tests/cognition_core_v2_appraisal_replay_harness.py`, and `tests/ownership/source_test_impact_manifest.json`.
- Authority: edit only the owned files and run deterministic tests needed for the implementation checkpoint.
- Applicable skills: local-llm-architecture, py-style, test-style-and-execution.
- Capability floor: Python production implementation, async deterministic tests, state-boundary reasoning, and exact pytest-node execution.
- Independence requirement: none for implementation; final acceptance remains with the parent.
- Acceptance output: a surgical diff, expected-failure evidence before production edit, passing owned tests, and an exact changed-file list.
- Gate: receives this fixed contract and clean owned-file baseline; exits with no edits outside the owned surface.

## Test Impact And Traceability

| Path | Changed symbol or contract | Semantic owner | Exact deterministic pytest nodes | Supplemental nodes | Mode | Regression prevented |
| --- | --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py` | `_appraise_semantic_item`, `_validate_semantic_boundary_candidate`, boundary-only proposition admission, family termination | semantic-appraisal producer and deterministic boundary owner | `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_structurally_usable_semantic_content_passes_runtime_without_retry`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_candidate_origin_mismatch_omits_family_without_retry`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_handle_domain_violation_omits_family_without_retry`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_unowned_delta_is_blocked_without_retry`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_runtime_admission_does_not_call_strict_semantic_validator`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_recoverable_structure_uses_one_replacement_then_completes`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_provider_failure_exhausts_the_stage_attempt_budget`; `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_prompt_cap_stops_before_the_first_provider_call` | `tests/test_cognition_v2_protected_qq_replay.py::test_current_candidate_origin_trace_terminates_family_without_retry`; `tests/test_cognition_v2_protected_qq_replay.py::test_captured_unowned_path_terminates_family_without_retry`; `tests/test_cognition_core_v2_semantic_terminalization.py::test_appraisal_state_incompatibility_terminates_family_without_retry` | deterministic unit and captured replay | Semantic-policy rejection, contradictory citation preservation, producer repair, or whole-turn failure returns after structural admission. |
| `src/kazusa_ai_chatbot/cognition_core_v2/README.md` | Failure Behavior and Semantic Admission And Relationship Maintenance | Cognition V2 documentation owner | `tests/test_cognition_v2_parent_guardrail_docs.py::test_core_readme_documents_semantic_admission_and_boundary_contract` | none | deterministic static documentation | Current documentation continues to describe the regressed repair policy. |
| `tests/ownership/source_test_impact_manifest.json` | semantic-appraisal exact-node mapping | verification owner | `tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary`; `tests/test_test_impact_manifest.py::test_manifest_rejects_empty_unit_mapping` | `scripts.validate_test_impact --base-ref 7505e78914655d028fdf248d9048acb2b5b2402f --run` | deterministic manifest validation | A changed production owner lacks collected exact regression nodes. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`: restore structural-first runtime admission and safe family termination.
- `tests/unit/cognition_core_v2/test_semantic_appraisal.py`: replace repair/preservation assertions with runtime zero-retry and graceful-termination assertions.
- `tests/test_cognition_v2_protected_qq_replay.py`: replay the captured failure shapes as one-call safe family terminations.
- `tests/test_cognition_core_v2_semantic_terminalization.py`: align the state-incompatibility regression with one-call family termination.
- `tests/unit/cognition_core_v2/test_facade.py`: remove the obsolete repairable-boundary fixture while preserving structural-exhaustion metadata coverage.
- `tests/test_cognition_core_v2_prompt_budget_continuity.py`: replace semantic-boundary repair expectations with one-call family termination while retaining structural-repair budget coverage.
- `tests/test_cognition_core_v2_integration.py`: prove a boundary-terminated appraisal family does not abort the surrounding cognition turn.
- `tests/test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm.py`: align live-suite declarations with zero semantic-repair calls.
- `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py`: remove candidate-origin repair requirements from the live failure matrix.
- `tests/cognition_core_v2_appraisal_replay_harness.py`: report bounded family termination rather than semantic-repair reachability for boundary candidates.
- `tests/ownership/source_test_impact_manifest.json`: map the new exact deterministic nodes.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`: describe the restored contract.

### Keep

- Every other production source, prompt, model route, retry constant, schema, reducer, facade, persistence, action, surface, and delivery path.
- Archived plans and raw trace evidence.

## Agent Autonomy Boundaries

Luna may choose local function decomposition that cleanly separates boundary carriers from semantic ownership. Luna must preserve the stated runtime sequence, zero semantic repair calls, safe family termination, exact owned surface, and all exclusions. Any required change outside the owned surface returns to the parent as a blocker rather than expanding scope.

## Verification

1. Luna adds or rewrites the exact tests first and records their expected failures against the current production source.
2. Luna implements the production and documentation patch and runs every owned exact node.
3. The parent reviews the full diff against this contract and the original completed plan.
4. The parent collects and runs every exact mapped node, then runs `scripts.validate_test_impact --base-ref 7505e78914655d028fdf248d9048acb2b5b2402f --run` and `git diff --check`.
5. The parent replays each captured case one at a time, inspects its evidence, and authors a human-readable before/after review under `test_artifacts/diagnostics/`.

## Acceptance Criteria

1. A structurally usable unfamiliar proposition label reaches runtime admission with one model call and no semantic-validator invocation.
2. The `llmtrace_79651...` candidate-origin shape consumes one call, authorizes no mutation, ends only its appraisal family, and permits the surrounding cognition turn to continue.
3. The `llmtrace_bc5e...` replacement shape has no citation-preservation rejection path because semantic producer repair and preservation are absent.
4. Unknown evidence/role handles and unauthorized target paths consume one call, authorize no mutation, and safely end the appraisal family.
5. Malformed producer structure retains the existing bounded replacement behavior.
6. Provider failures and prompt-cap behavior remain unchanged.
7. The runtime admission path does not call `validate_semantic_appraisal_result(...)`.
8. No prompt, model route, retry cap, reducer, facade, state, persistence, action, surface, or delivery behavior changes.
9. Every exact node is collected and passes; impact validation and `git diff --check` pass.
10. The implementation diff contains only the declared owned files plus this plan, registry lifecycle entry, and diagnostic review artifact.

## Progress Checklist

- [x] RCA and pre-change baseline captured at `7505e78914655d028fdf248d9048acb2b5b2402f`.
- [x] Original completed-plan direction reconstructed.
- [x] Superseding bugfix contract fixed and promoted to `in_progress` by explicit user command.
- [x] Luna test-first implementation completed.
- [x] Parent diff review completed.
- [x] Exact deterministic and impact verification completed.
- [x] Captured replay comparison artifact completed.
- [x] Closure evidence recorded and plan archived.

## Execution Evidence

- Baseline worktree: one pre-existing untracked user file, `development_plans/active/short_term/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md`.
- Baseline commit: `7505e78914655d028fdf248d9048acb2b5b2402f`.
- Fixed implementation executor: GPT-5.6 Luna, max reasoning, normal speed.
- Test-first reviewer-remediation evidence: the injected unmapped boundary
  defect initially entered a second provider call; after structural-error
  isolation it propagates after one call.
- Independent code review identified broad exception classification, stale
  repair-era tests, and incomplete impact traceability. All findings were
  resolved before parent closure.
- Parent final scoped suite: 71 passed and 3 skipped because the referenced
  protected near-cap captures are unavailable in this workspace.
- Source-impact validation: 21 exact nodes collected and passed.
- Live LLM suites: 21 nodes collected with no live model or database execution.
- Protected replays were run one case at a time for the `79651...` candidate,
  the `bc5e...` canonical `e1` correction, and the captured unowned path; all
  passed.
- Full cognition continuation passed with one rejected appraisal call and a
  valid `cognition_core_output.v2` result.
- Python compilation, CJK source safety, manifest JSON validation, and
  `git diff --check` passed.
- Human-readable review artifact:
  `test_artifacts/diagnostics/cognition_v2_semantic_admission_original_contract_restoration_review.md`.
- Parent closure sign-off: APPROVE. Runtime admission now follows the original
  structural-first contract; deterministic carrier/state failures terminate
  only their family, structural failures retain bounded replacement, and no
  semantic repair or citation-preservation path remains.
