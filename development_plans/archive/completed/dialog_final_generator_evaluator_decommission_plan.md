# Final dialog generator evaluator decommission

## Summary

- Goal: Remove every evaluator/verifier LLM stage attached to final dialog text, including scheduled and background-worker dialog candidates.
- Status: completed
- Scope boundary: Final dialog generation and scheduled/background speech admission only; unrelated action, RAG, relevance, cognition, and intake evaluators remain.
- Change direction: Big-bang removal of evaluator calls, evaluator-owned contracts, evaluator retry/scoring paths, and evaluator-only tests.
- Acceptance state: Verified by the implementation worker and independent parent evaluation.

## Confirmed Decisions

- The final dialog generator owns visible wording from the already-authoritative cognition/L3 surface.
- Deterministic parser, schema, authority, due-time, permission, and delivery checks remain.
- No replacement evaluator, compatibility path, feature flag, or new evaluator test is added.
- Evaluator-specific tests are deleted or have their evaluator cases removed without replacement.
- The existing worktree edit in src/kazusa_ai_chatbot/nodes/dialog_agent.py is preserved.

## Scope And Change Direction

The normal dialog path becomes generator-only after the validated surface is available. Candidate scoring, focused semantic/role/surface verifiers, verifier aggregation, evaluator-driven L3 repair, and evaluator retry decisions are removed. Structural generator parsing and bounded provider/JSON failure handling remain within the existing production contract.

Scheduled and background-worker dialog candidates use deterministic authority, trigger identity, due-time, candidate-presence, and delivery checks. The scheduled semantic content evaluator, its JSON verdict contract, evaluator attempt metadata, and evaluator-specific suppression codes are removed.

## Mandatory Skills

- development-plan: governs this decommission and cutover.
- local-llm-architecture: preserves cognition/L3 semantic ownership and keeps final wording generation bounded.
- py-style: applies to every changed Python source file.
- test-style-and-execution: applies to deletion/refactoring and verification of pytest surfaces.

## Mandatory Rules

- Preserve unrelated user changes and keep the production diff scoped to this contract.
- Use the canonical JSON parser for retained generator output.
- Do not add compatibility aliases, fallback evaluators, new semantic gates, or replacement tests.
- Do not modify unrelated evaluator stages.

## Must Do

- Remove runtime final-dialog focused verifier prompts, models, handlers, aggregation, scoring, and evaluator-driven repair flow.
- Make the normal dialog subgraph direct generator-to-end.
- Remove scheduled/background dialog content evaluation and its evaluator-owned state, trace, and contract fields.
- Remove active tests that assert, patch, or live-run the removed dialog evaluators.
- Preserve generator output validation, required-source-URL deterministic validation, cognition/L3 authority, and background due/authority/delivery safeguards.
- Run deterministic compile and pytest checks selected for the changed retained boundaries.

## Deferred

- Action-spec evaluation, RAG evaluation, web-agent evaluation, frontline/settled relevance evaluation, cognition authorization, and any other evaluator not consuming final dialog text.
- Historical archived plans, historical test artifacts, and historical trace records.
- New quality tests or replacement evaluator coverage.

## Target State

    cognition/L3 surface -> dialog generator -> deterministic parse/source checks -> delivery
    scheduled authority/due guard -> dialog generator candidate -> deterministic admission -> dispatch

No LLM call after a final dialog candidate is produced may judge its semantic quality, role direction, surface integrity, scheduled objective, or source alignment. The scheduled worker must not require a semantic verdict to admit a candidate.

## Execution Roles

### deepseek_flash_0731_implementation_worker

- Responsibility: Implement the decommission within the owned source and test surfaces.
- Owned surface: The exact paths under Change Surface; intentional overlap is limited to the pre-existing dialog_agent.py edit, which must be preserved.
- Authority: Edit and delete only the listed paths; run read-only scans and deterministic checks; no unrelated cleanup or architecture changes.
- Applicable skills: local-llm-architecture, py-style, test-style-and-execution.
- Capability floor: Trace multi-stage Python/LangGraph ownership, update typed contracts and callers, and run project pytest/compile checks.
- Independence requirement: Parent performs independent diff review and acceptance.
- Acceptance output: Changed-path list, evaluator-remnant scan, compile/test results, and concise residual-risk report.
- Gate: Start from the recorded baseline and stop on any contract conflict outside the owned surface.
- Fixed execution constraint: The user explicitly selected one DeepSeek Flash 0731 worker for implementation.

### parent_evaluation_owner

- Responsibility: Review the worker diff, verify ownership boundaries, run the final checks, and decide acceptance.
- Owned surface: Full repository read-only review and verification; production edits only for worker defects after review.
- Authority: Accept, reject, or request bounded remediation; no evaluator replacement.
- Independence requirement: Separate from the implementation worker.
- Acceptance output: Evidence-backed final status and residual-risk report.

## Test Impact And Traceability

| Source or contract | Semantic owner | Deterministic pytest nodes | Supplemental nodes | Mode | Regression prevented |
| --- | --- | --- | --- | --- | --- |
| src/kazusa_ai_chatbot/nodes/dialog_agent.py dialog_generator and final graph | Dialog rendering | tests/test_dialog_agent.py::test_dialog_agent_returns_final_dialog_and_target | tests/integration/cognition_core_v2/test_terminal_dialog_candidate.py::test_terminal_candidate_flows_through_dialog_generator | deterministic + live | Final dialog remains structurally renderable without any evaluator call. |
| src/kazusa_ai_chatbot/self_cognition/worker.py scheduled admission | Deterministic scheduled authority/due admission | tests/test_scheduled_future_speech_contract.py::test_scheduled_content_gate_accepts_current_authority | none | deterministic | Background dispatch no longer depends on an evaluator verdict while retaining authority and due guards. |
| src/kazusa_ai_chatbot/cognition_core_v2/contracts.py scheduled authority contract | Scheduled authority schema | tests/test_scheduled_future_speech_contract.py::test_scheduled_authority_contract_round_trip | none | deterministic | Removing the verdict does not weaken retained authority validation. |
| src/kazusa_ai_chatbot/self_cognition/tracking.py scheduled trace projection | Scheduled trace persistence | tests/test_self_cognition_tracking.py::test_build_scheduled_gate_trace_preserves_authority_and_dispatch | none | deterministic | Trace rows remain coherent after evaluator metadata removal. |
| src/kazusa_ai_chatbot/cognition_core_v2/model_attempt_policy.py dialog policy | Attempt-policy registry | tests/unit/cognition_core_v2/test_model_attempt_policy.py::test_dialog_generator_policy_is_bounded | none | deterministic | Removed verifier policy entries cannot remain routable. |
| Evaluator-only test files and cases | Test ownership | none for deleted behavior, by explicit user instruction | none | deletion | No active test preserves removed evaluator behavior or creates a replacement evaluator. |

The worker must confirm each named retained node exists and is collected before using it as acceptance evidence. If a named node is absent in the current branch, report the exact mismatch and use the closest existing owner test only after recording the deviation.

## Change Surface

### Modify

- src/kazusa_ai_chatbot/nodes/dialog_agent.py: remove final-dialog focused evaluators and evaluator-driven selection/repair while preserving the existing user edit and generator parser/trace path.
- src/kazusa_ai_chatbot/cognition_core_v2/surface.py: remove the now-unreachable evaluator-driven text-surface repair facade and implementation.
- src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py: remove the evaluator-owned dialog-compliance repair prompt, stage handler, and validator while retaining independent surface stages.
- src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py: remove the dead dialog-verifier repair adapter.
- src/kazusa_ai_chatbot/self_cognition/worker.py: remove scheduled content evaluator invocation and verdict-based admission; retain deterministic safeguards.
- src/kazusa_ai_chatbot/self_cognition/models.py: remove evaluator verdict/attempt fields from scheduled gate state.
- src/kazusa_ai_chatbot/self_cognition/tracking.py: remove evaluator-only trace fields and normalization.
- src/kazusa_ai_chatbot/self_cognition/runner.py: remove evaluator-only blocked-artifact fields.
- src/kazusa_ai_chatbot/cognition_core_v2/contracts.py: remove scheduled semantic-verdict types, validator, and evaluator-only codes/constants.
- src/kazusa_ai_chatbot/cognition_core_v2/model_attempt_policy.py: remove final-dialog verifier policy entries; retain generator policy.
- Active mixed test files named by the evaluator-remnant scan: remove evaluator-only cases, imports, patches, and assertions without adding replacements.
- tests/ownership/source_test_impact_manifest.json: remove mappings to deleted evaluator tests and obsolete dialog-verifier owner nodes.
- Active cognition/surface documentation and fixtures: remove current evaluator-owned contracts while preserving historical research records.

### Delete

- tests/test_dialog_visible_speech_and_semantic_fidelity.py: evaluator-owned deterministic suite.
- tests/test_dialog_visible_speech_and_semantic_fidelity_live_llm.py: evaluator-owned live suite.
- tests/test_scheduled_future_speech_content_gate.py: scheduled evaluator contract suite.
- tests/test_scheduled_future_speech_content_gate_live_llm.py: scheduled evaluator live suite.

### Create

- None.

### Keep

- README.md, docs/HOWTO.md, and subsystem READMEs unless the active scan finds a current evaluator contract reference; historical mentions remain historical.
- Unrelated evaluator stages and their tests.
- Existing user modifications outside this task.

## Agent Autonomy Boundaries

The worker may choose local deletion order, helper removal order, and exact deterministic admission implementation while preserving the target state. It must not retain aliases, invent a replacement semantic gate, alter unrelated evaluator stages, or add tests. Any need to change a path outside this surface requires parent review before proceeding.

## Verification

- Scan active src, tests, README.md, docs, and subsystem documentation for final-dialog evaluator symbols and scheduled evaluator symbols.
- Compile every changed Python source file with venv/Scripts/python.exe -m py_compile.
- Collect and run the retained deterministic owner nodes one at a time or as a regular deterministic batch.
- Review deletion status to ensure evaluator tests are gone and no replacement tests were created.
- Inspect the final diff against the pre-handoff baseline and verify the pre-existing dialog_agent.py edit is intact.

## Acceptance Criteria

- No runtime final-dialog evaluator/verifier LLM call, prompt, model, aggregation, score gate, or evaluator-driven retry remains.
- No scheduled/background dialog candidate invokes or requires a semantic evaluator.
- No active evaluator-only test file or evaluator-only test case remains; no replacement evaluator tests are added.
- Retained deterministic parser, authority, due-time, source-token, permission, and delivery checks remain operational.
- Unrelated evaluator stages remain unchanged.
- Parent evaluation confirms the worker diff, scans, compile checks, and collected deterministic tests satisfy this contract.

## Execution Evidence

### Baseline and handoff

- The parent captured `git status --short`, the pre-handoff changed-path set, and the explicit owned-path set before implementation handoff.
- The fixed executor was `deepseek_v4_flash_0731` (`deepseek-v4-flash`, agent `01a006dc-04c0-7800-86f8-d41a6748f11c`) under the plan's fixed execution constraint.
- The worker completed the bounded slice within the 600-second deadline. Existing user edits, the pre-existing `dialog_agent.py` edit, and the four evaluator-only test deletions were preserved.

### Attributable implementation and parent remediation

- Worker changes: `cognition_core_v2/model_attempt_policy.py`, `cognition_core_v2/README.md`, `tests/fixtures/cognition_core_v2_retry_exhaustion_cases.json`, and `tests/ownership/source_test_impact_manifest.json`.
- Parent remediation: removed non-semantic trailing whitespace in `nodes/dialog_agent.py` and removed one stale generator-graph step from `nodes/README.md` that still described an L3 replacement and second render after hard rejection.
- The manifest now maps the dialog owner to the retained generator tests and contains no deleted evaluator-test or obsolete dialog-verifier node mappings.

### Verification

- Exact active-source, test, README, and docs scans found no final-dialog verifier/evaluator symbols, scheduled semantic-verdict symbols, evaluator-only trace fields, or replacement evaluator tests. Historical research references remain under `docs/research/` as permitted.
- `venv\\Scripts\\python.exe -m py_compile` passed for all ten changed production Python paths in the owned implementation surface.
- The retained deterministic owner collection contained 26 exact nodes; all 26 passed in the final parent run. This included the retained dialog generator path, scheduled authority/due/trace paths, surface and L3 ownership tests, the attempt-policy owner test, and the retry-fixture ownership check.
- The five plan-named nodes absent from this branch were checked individually and recorded as deviations: the deleted terminal-dialog integration file, the absent scheduled-current-authority node, the absent scheduled-authority round-trip node, the absent scheduled-trace node, and the absent dialog-policy node. Closest existing owner tests were collected and passed as listed in the parent handoff evidence.
- JSON parsing passed for the updated ownership manifest and retry fixture. Owned-path `git diff --check` passed.

### Residual risk and scope boundaries

- The repository-wide manifest command stopped at the pre-existing changed `src/kazusa_ai_chatbot/cognition_core_v2/__init__.py`, which has no manifest entry and is outside this handoff's owned path set. The parent preserved that existing change rather than expanding scope.
- The worker reported six pre-existing failures in the already-modified `tests/test_cognition_core_v2_model_retry_continuity.py` when exercising its broader file; the retained attempt-policy fixture node passed, and no unrelated test file was changed for this plan.
- No live LLM test was run. The active evaluator-only live suites were deleted by the pre-existing plan work, and the retained acceptance surface is deterministic structural, authority, due-time, trace, and delivery validation.
