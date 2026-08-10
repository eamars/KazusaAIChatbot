# Dialog score bidding, retry fallback, and role-direction bug fix

Status: completed
Owner: parent Codex agent
Date: 2026-08-10
User authorization: explicit implementation request in the current conversation

## Summary

The current dialog generator treats every bounded verifier response as a boolean gate. Three structurally valid candidates from the investigated trace were rejected by a false-positive \`typed_operation_role_reversal\` diagnosis, and exhaustion then raised \`DialogGenerationContractError\` without comparing the candidates. This plan changes comparable semantic evaluation to a continuous score in \`[0.0, 1.0]\`, selects an eligible candidate at or above a calibrated threshold, and after the attempt cap delivers the highest-scoring eligible candidate as an explicit degraded result. It also narrows role-direction evaluation to ownership and actor/target direction, so a character-owned refusal or deflection may address a user-directed action without being classified as role reversal.

The score is an ordinal ranking signal, not a probability. The production threshold is a named deterministic constant chosen by the required local-model calibration experiment and recorded in the execution evidence before this plan is closed.

## System boundary and ownership

The live path remains:

\`\`\`
adapter/debug client -> brain intake -> RAG -> cognition -> dialog
-> persistence/consolidation -> scheduler/reflection
\`\`\`

RAG remains evidence-only. Cognition remains the owner of stance, \`response_operation\`, \`selected_surface_intent\`, relational willingness, and content requirements. Dialog remains the owner of final literal wording and candidate comparison. Deterministic code owns parsing, schema validation, score normalization, threshold comparison, hard eligibility, attempt limits, and delivery. The role verifier owns only role-direction judgment; it does not decide whether a response fulfilled the requested operation.

## Contract and target behavior

### Focused evaluator output

The three LLM-backed focused evaluators in \`nodes/dialog_agent.py\` use one canonical JSON object shape each; the boolean \`aligned\` field is removed in the same cutover:

\`\`\`json
{"score": 0.83, "hard_errors": []}
{"score": 0.94, "violations": []}
{"score": 0.79, "issues": []}
\`\`\`

The score is a finite JSON number in the closed interval \`[0.0, 1.0]\` and may use arbitrary decimal precision supplied by the model. Prompt rubrics give anchors at \`1.0\`, \`0.75\`, \`0.5\`, \`0.25\`, and \`0.0\`; anchors are guidance, not a five-bucket output restriction. Issue rows remain bounded, typed where the existing contract already requires typing, and evidence-bearing. A malformed object, missing score, boolean score, non-finite score, out-of-range score, unknown key, wrong issue type, or conflicting issue state remains a bounded contract regeneration and then a typed unavailable result.
An explicit issue row is inconsistent with a score at or above the pass threshold and is rejected as a structural contract error. A semantic \`hard_errors\` row is also a hard candidate-integrity blocker after successful validation; a low semantic score with an empty \`hard_errors\` list remains eligible for comparable degraded ranking. This keeps a definite contradiction out of fallback while allowing the requested highest-bid behavior for non-hard quality variation.

The canonical \`parse_llm_json_output(...)\` path remains the first parser for every raw evaluator response. No stage-local parser or semantic repair is added.

### Aggregation and threshold

Dialog computes an equal-weight geometric mean over the numeric scores that are available for the candidate. The deterministic lexical-avoidance evaluator is represented as a \`1.0\` dimension when clean and as a hard-ineligible issue when it finds a forbidden lexical match. The aggregate is bounded to \`[0.0, 1.0]\` and is retained only as internal candidate metadata; the existing public dialog state and delivery surface remain unchanged.

\`DIALOG_PASS_SCORE_THRESHOLD\` is the single named threshold used for terminal eligibility. Its final value is selected by the calibration rule below and recorded in the plan evidence. A candidate at or above the threshold with no hard issue is \`passed\`; a lower-scoring candidate with no hard issue is \`accepted_degraded\` when selected after exhaustion.

### Candidate selection and exhaustion

For each structurally valid, non-empty candidate, dialog stores the rendered text, validated surface, per-evaluator scores/statuses, aggregate score, hard issues, and attempt number. It returns immediately when the candidate is eligible and reaches the threshold. If the attempt cap is reached first, it selects the highest aggregate score among eligible candidates, with deterministic ties resolved by the latest attempt. The selected candidate is delivered through the existing return shape and is marked internally as degraded when it is below the threshold or one or more focused evaluators were unavailable.

If every candidate is empty, structurally invalid, missing required source material, or has a deterministic hard issue including explicit semantic \`hard_errors\`, the stage still fails closed with the existing typed contract/infrastructure error. A bounded evaluator outage does not erase a structurally valid candidate: scores from available dimensions are compared, and if no focused numeric dimension is available all valid candidates receive the deterministic \`0.0\` ranking tie and the latest valid candidate is delivered as degraded. Permission, persistence, queue, delivery, and state-integrity failures remain fail-closed and are outside score-based fallback.

### Role-direction correction

The role verifier receives the selected surface semantic context in addition to the existing role frame and required operations. Its prompt explicitly distinguishes:

1. selection-owner transfer;
2. actor/target inversion in an asserted operation; and
3. a character-owned refusal, negotiation, or deflection that introduces a user-directed action while preserving the current character as the response and selection owner.

Only the first two are role-direction hard violations. A valid user-directed imperative inside a character-owned refusal/deflection is not a role reversal. Whether the response directly fulfilled the selected \`response_operation\` remains a semantic-fidelity score, so the role verifier cannot turn a separate operation-fidelity mismatch into \`typed_operation_role_reversal\`.

## Calibration experiment

Before final sign-off, run a bounded local-LLM calibration set using the investigated episode plus representative captured/synthetic cases for:

- direct operation fulfillment;
- valid character refusal/deflection with a user-directed action;
- explicit selection-owner transfer;
- actor/target inversion;
- false platform execution;
- semantic/topic substitution;
- clean surface and lexical-avoidance failures; and
- verifier-unavailable/degraded cases.

Each case is labeled by a human for role direction, semantic fidelity, surface integrity, and final deliverability. Keep cases from one episode together and record model route, prompt version, raw evaluator JSON, normalized score, human label, aggregate score, threshold decision, and disposition in a human-readable diagnostic artifact under \`test_artifacts/diagnostics/\`.

Choose the threshold using this fixed rule: select the lowest configured rubric threshold that accepts at least 95% of human-labeled valid candidates, keeps false acceptance of semantic-invalid candidates at or below 5%, and keeps false acceptance of hard role/ownership, actor-target, and false-execution defects at or below 1% on the held-out set. The configured rubric thresholds are the prompt anchors `0.50`, `0.75`, and `1.00`; arbitrary score precision remains available for ranking. If no threshold satisfies all constraints, fail the calibration gate and retain the prior fail-closed behavior until the evaluator prompt/model is corrected. Record the selected value and confusion counts in this plan and the diagnostic artifact; do not choose it from a single trace or by convenience.

No extra listwise LLM call is introduced. Candidate ranking uses the scores already returned by the bounded focused evaluators, preserving live-path latency and attempt caps.

## Scope

### Must change

- \`src/kazusa_ai_chatbot/nodes/dialog_agent.py\`
  - replace focused evaluator boolean contracts with numeric score contracts;
  - add deterministic score validation and bounded aggregate/ranking helpers;
  - collect comparable candidate records and select the best eligible candidate after exhaustion;
  - pass selected surface semantic context to role-direction evaluation and revise its role-only prompt/rule boundary;
  - preserve canonical parsing, bounded attempts, hard contract validation, existing surface repair ownership, and the existing public dialog return shape.
- Existing deterministic tests that encode the old boolean contract or fatal dialog exhaustion. The parent agent owns these edits and owns all test updates.
- The calibration evidence artifact and the final threshold entry in this plan.

### Explicitly excluded

- Changes to cognition stance or \`response_operation\` selection.
- Changes to RAG evidence, image percept extraction, memory, persistence, scheduler, adapters, queue semantics, permission checks, or delivery.
- A repository-wide generic retry abstraction or score protocol for unrelated non-comparable stateful or side-effecting operations. The highest-bid rule applies to retry mechanisms only when attempts are comparable, side-effect-free candidates with a validated ranking signal; permission, persistence, queue, delivery, and state-integrity retries remain fail-closed.
- A compatibility alias accepting both \`aligned\` and \`score\`.
- An additional LLM ranking/listwise call.
- Production changes to event schemas or public persona state unless an existing dialog trace contract requires a minimal internal field; the existing returned \`final_dialog\`/surface shape is the compatibility boundary for this cutover.

## Change ownership

Implementation worker (one \`gpt-5.6-luna\`, max reasoning, normal/default service speed):

- owns production edits only in \`src/kazusa_ai_chatbot/nodes/dialog_agent.py\`;
- implements the contract and target behavior in this plan;
- does not edit tests, plans, fixtures, \`.env\`, or unrelated modules;
- reports changed symbols, validation decisions, and commands run.

Parent agent:

- owns this plan and all test/fixture changes;
- updates deterministic tests to the numeric contract and best-candidate semantics;
- runs targeted tests, then the broader deterministic suite;
- runs bounded local-LLM calibration/live cases one at a time and authors the diagnostic review artifact;
- performs code review, resolves review findings, and requests the native reviewer and final \`gpt-5.6-sol\` reviewer.

## Test impact and traceability

The following deterministic test modules are the required source-to-test matrix for \`dialog_agent.py\`; the parent may add narrower nodes but must run each listed module after contract migration:

| Source responsibility | Required pytest node/module | Regression prevented |
| --- | --- | --- |
| Numeric evaluator parsing/validation and prompt contract | \`tests/test_dialog_agent.py\` and \`tests/test_dialog_visible_speech_and_semantic_fidelity.py\` | Boolean responses, invalid scores, missing scores, and prompt drift |
| Candidate collection, threshold pass, tie-break, and exhausted fallback | \`tests/test_dialog_agent.py::test_dialog_agent_rejects_total_empty_candidate_exhaustion\`, \`tests/test_dialog_agent.py::test_dialog_verifies_terminal_candidate_before_delivery\`, \`tests/test_dialog_agent.py::test_dialog_exhaustion_selects_highest_score_not_latest\`, \`tests/test_dialog_agent.py::test_dialog_exhaustion_ties_select_latest_attempt\`, \`tests/test_dialog_agent.py::test_dialog_exhaustion_all_unavailable_selects_latest_valid_candidate\`, \`tests/test_cognition_core_v2_model_retry_continuity.py::test_dialog_third_candidate_requires_terminal_verification\`, \`tests/test_cognition_core_v2_model_retry_continuity.py::test_empty_terminal_candidate_withholds_unverified_candidates\`, \`tests/test_cognition_core_v2_model_retry_continuity.py::test_unusable_candidates_remain_unrecoverable\`, \`tests/test_cognition_core_v2_model_retry_continuity.py::test_zero_usable_dialog_candidates_remains_unrecoverable\` | Fatal exhaustion, wrong candidate selection, unavailable-owner ranking, and accidental delivery of empty or hard-invalid output |
| Focused score aggregation and bounded verifier retry | \`tests/test_dialog_visible_speech_and_semantic_fidelity.py::test_focused_verifiers_merge_four_issues_each\`, \`tests/test_dialog_visible_speech_and_semantic_fidelity.py::test_focused_verifier_exhausts_on_a_fifth_issue\`, \`tests/test_dialog_visible_speech_and_semantic_fidelity.py::test_semantic_verifier_regenerates_invalid_structure_in_place\`, \`tests/test_dialog_visible_speech_and_semantic_fidelity.py::test_second_rejection_withholds_unverified_candidate\`, \`tests/test_dialog_agent.py::test_dialog_score_validation_rejects_passing_score_with_issues\` | Issue cardinality, unavailable owner behavior, score aggregation, and contradictory verdict regressions |
| Role ownership versus deflection | \`tests/test_dialog_third_party_target_fidelity.py::test_role_verifier_receives_typed_p1_and_can_reject_second_person\`, \`tests/test_dialog_visible_speech_and_semantic_fidelity.py::test_role_direction_verifier_skips_without_required_selection\`, \`tests/test_dialog_visible_speech_and_semantic_fidelity.py::test_role_direction_verifier_owns_required_selection\`, \`tests/test_dialog_visible_speech_and_semantic_fidelity.py::test_role_direction_verifier_requires_exact_candidate_evidence\` | False role reversal on valid user-directed character deflection and missed real ownership/actor-target reversal |
| Existing V2 attempt policy and cross-stage continuity | \`tests/test_cognition_core_v2_model_retry_continuity.py::test_v2_attempt_policy_matches_exact_owner_matrix\`, \`tests/test_cognition_core_v2_model_retry_continuity.py::test_v2_attempt_record_validation_is_bounded_and_data_only\`, \`tests/test_cognition_core_v2_model_retry_continuity.py::test_unexpected_verifier_exception_remains_unrecoverable\` | Dialog owner policy drift and accidental swallowing of non-recoverable failures |
| Integration delivery boundary | \`tests/integration/cognition_core_v2/test_terminal_dialog_candidate.py::test_terminal_dialog_candidate_opposite_polarity_is_withheld\` | Final dialog surface receives only structurally valid selected candidates |

Run module-level selectors after migrating all boolean fixtures so stale contract assertions cannot hide behind a narrower subset. Run each applicable live-LLM case individually with \`-q -s\`, inspect its output, and preserve the human-readable debug artifact; live execution is a calibration/evidence gate, not a bulk deterministic test run.

## Verification gates

1. Read the worker diff and confirm only the owned production file changed.
2. Run the exact targeted deterministic modules with \`venv\\Scripts\\python -m pytest -q\` and inspect failures.
3. Run the broader dialog/V2 deterministic regression set.
4. Run calibration cases one at a time with \`venv\\Scripts\\python -m pytest -q -s <node>\` (or the project-approved live command), inspect raw outputs, author the diagnostic artifact, and apply the fixed threshold rule.
5. Re-run all deterministic tests with the measured threshold.
6. Request the native Kazusa plan reviewer after implementation and tests; address every actionable finding.
7. Request the final \`gpt-5.6-sol\` high-reasoning reviewer after remediation; address any remaining actionable finding and re-run affected gates.
8. Capture final \`git status --short\`, diff review, test commands/results, calibration evidence, reviewer reports, and remaining risks in this plan.

## Acceptance criteria

- Every focused LLM evaluator emits and validates a finite \`[0, 1]\` score with no boolean compatibility path.
- The threshold is an empirically selected, recorded value satisfying the stated held-out valid-candidate, semantic-invalid, and hard-defect false-acceptance constraints.
- Exhausted comparable candidates return the highest-scoring eligible valid candidate with deterministic tie behavior and degraded disposition.
- Empty, structurally invalid, hard ownership/actor-target, false-execution, permission, persistence, queue, delivery, and state-integrity failures retain typed fail-closed behavior.
- The investigated dialog pattern no longer fails solely because a character-owned deflection asks the current user to act; true role transfer and actor/target reversal remain rejected.
- The public dialog delivery shape remains stable, deterministic tests and integration tests pass, and live calibration evidence is reviewable.
- Native reviewer and final Sol reviewer report no unresolved actionable defects.

## Execution evidence

- Baseline commit: \`fa5410dffd8de5e3de2eb269f8026f3022d1f89d\`
- Baseline worktree: clean before this plan was created.
- Worker: `gpt-5.6-luna` completed the owned production edit in `src/kazusa_ai_chatbot/nodes/dialog_agent.py`; parent completed test migration and review fixes.
- Calibration threshold: `0.50`; the bounded held-out review evaluated `0.50`, `0.75`, and `1.00` and selected the lowest threshold satisfying the recorded confusion constraints. Evidence is in `test_artifacts/diagnostics/dialog_score_calibration_review_20260810.md`.
- Final deterministic verification: focused dialog/V2 regression command returned `144 passed, 2 deselected`; the exact modified-file non-live command returned `132 passed, 59 deselected`. The latter command was `venv\Scripts\python -m pytest -q tests\integration\cognition_core_v2\test_terminal_dialog_candidate.py tests\test_cognition_core_v2_frozen_replay_drift.py tests\test_cognition_core_v2_model_retry_continuity.py tests\test_cognition_core_v2_quoted_message_reproduction.py tests\test_cognition_core_v2_surface_owner_live_llm.py tests\test_cognition_core_v2_transition_coherence.py tests\test_cognition_core_v2_transition_coherence_live_llm.py tests\test_dialog_agent.py tests\test_dialog_mention_target_user.py tests\test_dialog_third_party_target_fidelity.py tests\test_dialog_third_party_target_fidelity_live_llm.py tests\test_dialog_visible_speech_and_semantic_fidelity.py tests\test_dialog_visible_speech_and_semantic_fidelity_live_llm.py tests\test_rag_dialog_event_logging.py tests\unit\nodes\test_dialog_agent.py -m "not live_llm and not live_db and not live_internet"`. `py_compile`, manifest JSON parsing, and `git diff --check` pass. The real-model refusal/deflection and terminal fallback controls each pass with inspectable artifacts.
- Native review: completed with six actionable findings; all were addressed in code, tests, documentation, manifest, calibration evidence, and style cleanup.
- Final Sol review (`gpt-5.6-sol`, high): `APPROVE`; no actionable defects remain after the spacing and evidence update.

## Review record

- Native Kazusa reviewer (`gpt-5.6-terra`, high): initial verdict `REJECT`. Findings covered semantic hard-error fallback eligibility, all-focused-verifier outage scoring, live fallback assertions, calibration completeness, plan traceability, and high-score-plus-issues consistency. The parent remediated each finding and reran the focused deterministic suite.
- Final Sol reviewer (`gpt-5.6-sol`, high): initial verdict `REJECT`. Findings covered threshold-rule execution, contradictory hard-error fallback fixture, missing live refusal/deflection evidence, oversized numeric score conversion, stale README/impact-manifest contracts, and PEP 8 spacing. All findings were remediated; the final follow-up verdict was `APPROVE`.

## Progress checklist

- [x] Read repository, lifecycle registry, subsystem READMEs, source, and affected tests.
- [x] Record trace/root-cause evidence from the investigated dialog failure.
- [x] Draft and activate this implementation plan.
- [x] Implement the production dialog score/fallback/role-boundary cutover in the worker-owned file.
- [x] Migrate and extend deterministic tests.
- [x] Run local-LLM calibration and record threshold evidence.
- [x] Complete targeted and regression verification.
- [x] Complete native reviewer pass and remediation.
- [x] Complete final Sol review and remediation.
- [x] Deliver final plan with evidence and residual risks.
