# coding agent phase2.5 security boundary plan

## Summary

- Goal: finish the coding-agent agent-space security boundary cleanup so
  Phase 2 proposal artifacts stay inspectable data and cannot be validated by
  executing generated code, generated tests, generated commands, or generated
  scripts.
- Plan class: high_risk_migration.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  and `test-style-and-execution`.
- Pre-execution codebase state as of 2026-07-07: Phase 0, Phase 1, Phase 2,
  and Phase 3 were implemented and archived; `code_writing` production flow
  called `materialize_patch_artifacts_for_review(...)`; legacy
  `validate_patch_artifacts(...)` still existed and could execute patched
  Python tests through pytest if a caller reached it.
- Completed state as of 2026-07-07: Phase 2 exposes
  `materialize_patch_artifacts_for_review(...)` as the single review boundary;
  `validate_patch_artifacts(...)`, `_python_test_execution_error(...)`,
  `_sandbox_test_env(...)`, and generated-test pytest execution wiring are
  removed from `patch_validation.py`.
- Overall cutover strategy: bigbang removal of execution-capable generated
  artifact validation from the coding-agent Phase 2 code-writing boundary.
- Highest-risk areas: dormant validation helpers becoming callable again,
  hidden pytest/subprocess execution paths, ambiguous "validation" wording, and
  future code-modifying/code-executing phases reusing Phase 2 helpers without a
  reviewed capability contract.
- Acceptance criteria: Phase 2.5 is complete when the architecture, ICD, code,
  and tests prove Phase 2 generated artifacts are review packages only, and any
  future execution-capable validation requires a separately approved capability
  with isolation, permission, and audit.

## Context

The coding-agent architecture separates agent-space artifact generation from
real-world execution. Phase 2 now produces source-free new-artifact patch
proposals and materialized review packages. Phase 3 background-worker
integration is already completed and routes coding tasks through the accepted
background-work path. Existing-source semantic modification, patch apply, and
execution remain future capabilities.

The current `code_writing` ICD already states that generated code and
generated tests are not executed. The current `code_writing` supervisor uses
`materialize_patch_artifacts_for_review(...)`, which parses and applies
proposed diffs into managed review storage without running generated tests.

The security cleanup target was narrower than the original draft:

- Before execution,
  `src/kazusa_ai_chatbot/coding_agent/code_writing/patch_validation.py`
  defined `validate_patch_artifacts(...)`.
- That legacy validation helper called `_python_test_execution_error(...)`.
- `_python_test_execution_error(...)` ran `sys.executable -m pytest` against
  patched test files inside the validation sandbox.
- Static search showed no production caller of `validate_patch_artifacts`
  outside the helper module, but keeping the helper in the Phase 2 package
  conflicted with the current inert-artifact boundary and created a future
  footgun.

Phase 2.5 closed that gap. It did not authorize execution, patch apply,
existing-source semantic modification, package installation, service startup,
database writes, network-backed execution, or Docker/sandbox execution.

## Confirmed Change Boundary

This planning cleanup changes only development-plan lifecycle files:

- Refresh this active Phase 2.5 plan.
- Close the stale Gate 02 role-boundary contract by moving it out of
  `development_plans/active/short_term/`.
- Update `development_plans/README.md` so the active registry matches the
  filesystem.

This planning cleanup does not modify production code or tests. Production-code
changes require a later explicit implementation command from the user.

The future Phase 2.5 implementation may modify only the files listed in
`Change Surface` unless the user explicitly expands scope.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing coding-agent role boundaries,
  tool capability contracts, or prompt-facing tool descriptions.
- `py-style`: load before editing Python production files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After automatic context compaction, the active agent must reread this entire
  plan before continuing implementation, verification, handoff, lifecycle
  updates, or final reporting.
- After signing off any major checklist stage, the active agent must reread
  this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the code-review gate defined by this plan and record the result in
  `Execution Evidence`.
- Implementation execution uses parent-led native subagents for production-code
  changes. If native subagents are unavailable, pause at the execution gate and
  get an explicit user instruction for the fallback execution model.
- Agent stages may generate only structured tool-call intents, proposed
  artifacts, traces, and review records.
- Tool calls must target approved agent-space capabilities and must not accept
  raw executable payloads.
- Generated code, tests, commands, and scripts must remain inert artifacts.
- Real-world execution, package installation, network access, database writes,
  service startup, workspace mutation, patch application, arbitrary shell
  access, and Docker-backed execution require a separate approved capability
  with isolation, permission, and audit.
- Do not treat `git apply` review materialization as generated-code execution.
  It may remain only for patch-shape review and only when it does not run
  generated code, tests, commands, or scripts.

## Must Do

- Audit current coding-agent writing, patching, validation, review
  materialization, E2E, and contract-test paths for generated-artifact
  execution.
- Delete `validate_patch_artifacts(...)` from the Phase 2 code-writing module
  so Phase 2 no longer exposes an execution-capable generated-artifact
  validation helper.
- Delete `_python_test_execution_error(...)`, `_sandbox_test_env(...)`, and the
  pytest execution wiring from the Phase 2 validation module. Delete any
  execution-only helper chain after caller audit confirms it is not shared by
  non-executing review materialization.
- Keep non-executing checks for patch parseability, path safety, secret/binary
  exclusions, markdown env-assignment inspection, Python syntax parsing via
  `ast.parse`, import/reference static inspection, and review-package
  materialization where those checks do not execute generated artifacts.
- Update coding-agent architecture and ICD wording so current implemented
  Phase 2 and Phase 3 state is accurate and Phase 2 "validation" consistently
  means non-executing review materialization and structural inspection.
- Add deterministic tests that prove the Phase 2 supervisor calls only
  `materialize_patch_artifacts_for_review(...)` and that no Phase 2 helper runs
  generated Python tests, generated scripts, shell commands, package commands,
  or target project tests.
- Keep Phase 2 E2E pass/fail focused on artifact quality and workflow
  correctness while this plan owns the security boundary cleanup.

## Deferred

- Do not implement `code_executing`.
- Do not implement patch application to a real checkout.
- Do not implement existing-source semantic modification.
- Do not add Docker execution, package installation, service startup, database
  writes, or network-backed execution.
- Do not broaden Phase 2 new-artifact scope.
- Do not add compatibility shims for old validation shapes.
- Do not add validation feedback loops or generated-artifact repair loops.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Validation boundary | bigbang | Replace the old execution-capable validation helper with one non-executing Phase 2 review boundary. |
| Tool-call contract | bigbang | Keep one agent-space tool-call guideline with no raw executable payloads. |
| Tests | bigbang | Replace security-sensitive execution assumptions with non-executing checks and review evidence. |
| Documentation | bigbang | Update active architecture/ICD wording to current implemented Phase 2 and Phase 3 state. |

## Cutover Policy Enforcement

- Use one canonical Phase 2 review boundary:
  `materialize_patch_artifacts_for_review(...)`.
- Remove the old execution-capable validation boundary in the same
  implementation pass as the tests and documentation updates.
- Leave no compatibility wrapper, alias, fallback mapper, or deprecated
  forwarding function for `validate_patch_artifacts(...)`.
- Treat any future execution-capable validation as a new capability with a new
  approved plan, interface, permission model, isolation boundary, and audit
  trail.

## Target State

Coding-agent Phase 2 outputs are proposed artifacts and reviewable metadata.
Review may inspect artifact shape, paths, caps, diff parseability,
public-safe metadata, workflow handoff records, Python syntax with `ast.parse`,
static import/reference coherence, and materialized files in managed storage.
Review does not run generated code or generated tests.

`patch_validation.py` has no Phase 2 callable path that executes generated
Python tests, generated scripts, generated commands, target project tests,
package commands, shell verification, or service startup.

Later execution-capable phases must introduce a dedicated capability with an
explicit interface, isolation boundary, permission model, and audit trail.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Agent-space boundary | Treat generated artifacts as inert data | Keeps LLM output separate from real-world effects. |
| Phase ownership | Phase 2.5 owns security remediation | Phase 2 remains focused on new-artifact proposal quality. |
| Current production path | Preserve `materialize_patch_artifacts_for_review(...)` as the Phase 2 boundary | It matches the current ICD and avoids executing generated tests. |
| Legacy helper | Delete `validate_patch_artifacts(...)` from Phase 2 | A dormant pytest-running helper conflicts with the inert-artifact boundary. |
| Validation style | Prefer inspection, static checks, and AI review over execution | Preserves real LLM gate intent without crossing the execution boundary. |
| Future execution | Require a separate approved capability | Prevents accidental shell or test execution through validation helpers. |

## Contracts And Data Shapes

- Public Phase 2 review boundary:
  `materialize_patch_artifacts_for_review(...) -> PatchValidationSummary`.
- Removed Phase 2 boundary: `validate_patch_artifacts(...)`.
- Removed Phase 2 execution helper: `_python_test_execution_error(...)`.
- `PatchValidationSummary` remains the review-materialization result shape and
  must not gain fields that imply generated-code execution.
- Phase 2 contract tests should assert the supervisor calls
  `materialize_patch_artifacts_for_review(...)` and that the code-writing
  package exposes no callable path for generated-test execution.

## LLM Call And Context Budget

- No new LLM call is introduced by this plan.
- No prompt template, model route, token budget, response schema, or live
  chatbot response path changes are in scope.
- Live LLM E2E remains a one-case gate used for artifact-quality evidence after
  deterministic security-boundary evidence passes.

## Change Surface

### Modify

- `development_plans/reference/designs/coding_agent_architecture.md`: update
  implemented-state wording so Phase 3 is completed, Phase 2 review is
  non-executing, and execution-capable validation remains future work.
- `src/kazusa_ai_chatbot/coding_agent/README.md`: keep the implemented
  coding-agent ICD aligned with inert proposal artifacts and completed
  background-worker integration.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/README.md`: keep the
  code-writing ICD aligned with review-package materialization, non-executing
  validation terminology, and no generated-test execution.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/patch_validation.py`:
  delete the execution-capable validation helper, pytest execution path, and
  execution-only helper chain from Phase 2 while preserving shared
  non-executing review helpers.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/supervisor.py`: only if
  needed to preserve the current materialization-only call boundary or to make
  the boundary easier to test.
- `tests/test_coding_agent_phase2_new_artifact_contracts.py`: add
  non-executing boundary checks for Phase 2 review materialization and absence
  of generated-test execution.
- `tests/test_coding_agent_phase2_new_artifact_e2e_live_llm.py`: only if
  needed to keep E2E evidence collection aligned with inert artifact review.

### Keep

- Phase 2 PM lifecycle, generated-artifact readback, File Agent path
  reservation, patch artifact materialization, and review-package
  materialization except where they expose execution-capable semantics.
- Phase 2 hard-gate challenge definitions as reference evidence.
- Background-work `coding_agent` worker routing and result mapping.
- `code_fetching` and `code_reading` subprocess use for git/rg/source
  inspection, provided it remains read-only and does not execute generated
  artifacts.

## Overdesign Guardrail

- Actual problem: a legacy Phase 2 validation helper can execute generated
  tests and conflicts with the current inert-artifact contract.
- Minimal change: delete the legacy execution-capable validation helper, keep
  materialization-only review, align docs, and add focused tests.
- Ownership boundaries: LLM stages own semantic artifact generation;
  deterministic code owns structure, limits, path safety, metadata validation,
  and audit; execution-capable work belongs to a later approved capability.
- Rejected complexity: no sandbox runner, Docker runner, package installer,
  patch applier, service launcher, command executor, validation repair loop, or
  compatibility wrapper in this plan.
- Evidence threshold: execution support requires a separate approved plan with
  isolation, permission, audit, and dedicated tests.

## Agent Autonomy Boundaries

- The active agent may change only the files listed in `Change Surface` unless
  the user explicitly expands scope.
- The active agent must not add execution, mutation, package, service, network,
  database, or patch-apply capabilities.
- The active agent must not weaken Phase 2 anti-cheat or real LLM review rules.
- If artifact quality cannot be proven without execution, record the
  limitation and keep the generated artifact inert.
- If a subprocess match is read-only source inspection, git patch-shape review,
  or ordinary test harness code rather than generated-artifact execution,
  record the classification in `Execution Evidence` instead of deleting it
  blindly.

## Implementation Order

1. Parent rereads this plan, the architecture reference, the coding-agent ICD,
   the code-writing ICD, the Phase 2 completed plan, and required skills.
2. Parent records baseline grep evidence for `validate_patch_artifacts(...)`,
   `materialize_patch_artifacts_for_review(...)`,
   `_python_test_execution_error`, `subprocess.run`, `pytest`, and
   `sys.executable -m pytest` under `src/kazusa_ai_chatbot/coding_agent` and
   relevant coding-agent tests.
3. Parent adds or updates deterministic contract tests that fail while Phase 2
   exposes `validate_patch_artifacts(...)` or a generated-test execution path.
   Record the expected failing or baseline result before production-code edits.
4. Production-code subagent removes the execution-capable validation boundary
   from `patch_validation.py`, preserving only helpers required by
   `materialize_patch_artifacts_for_review(...)` and non-executing static
   inspection.
5. Parent updates architecture and ICD wording to current implemented state:
   Phase 2 review materialization is non-executing, Phase 3 worker integration
   is complete, and execution-capable validation is future work.
6. Parent runs deterministic verification, classifies remaining static-search
   matches in `Execution Evidence`, and fixes any unresolved boundary failures.
7. Parent runs one live LLM E2E gate only after deterministic evidence passes,
   then inspects generated artifact paths without executing generated code.
8. Independent code-review subagent reviews the final diff and evidence before
   lifecycle sign-off.

## Execution Model

- Execute this plan with parent-led native subagents for production-code changes
  and independent code review.
- The parent agent owns orchestration, tests, documentation, verification,
  execution evidence, review remediation, lifecycle updates, and final sign-off.
- The production-code subagent owns only the approved production-code files in
  `Change Surface`, primarily `patch_validation.py` and `supervisor.py` if the
  supervisor boundary requires a testability edit.
- If native subagents are unavailable, pause before production-code execution
  and get an explicit user instruction for a fallback model.
- Live LLM tests must run one case at a time with output inspected.
- Deterministic tests may verify security boundaries, structural validation,
  path safety, review materialization, and absence of generated-artifact
  execution.

## Progress Checklist

- [x] Stage 1 - audit current boundary and contracts.
  Covers: architecture reference, coding-agent ICD, code-writing ICD, completed
  Phase 2 evidence, and current grep baseline.
  Verify: baseline grep commands complete.
  Evidence: record matches and classifications in `Execution Evidence`.
  Handoff: parent only.
  Sign-off: parent confirms implementation scope remains unchanged.
- [x] Stage 2 - test contract for non-executing Phase 2 review.
  Covers: contract tests for supervisor materialization call boundary and
  absence of Phase 2 generated-test execution helpers.
  Verify: focused test run records failing or baseline result before
  production-code edit.
  Evidence: record command and result in `Execution Evidence`.
  Handoff: parent to production-code subagent after tests are committed to the
  working tree.
  Sign-off: parent confirms tests encode the security boundary.
- [x] Stage 3 - validation boundary cleanup.
  Covers: `patch_validation.py` execution-capable helper deletion and any
  necessary supervisor testability edit.
  Verify: focused grep shows no Phase 2 callable generated-test execution path.
  Evidence: production-code subagent handoff plus parent grep result.
  Handoff: production-code subagent to parent.
  Sign-off: parent confirms no out-of-scope capabilities were added.
- [x] Stage 4 - architecture and ICD alignment.
  Covers: architecture reference, coding-agent README, and code-writing README.
  Verify: docs use one non-executing Phase 2 review vocabulary.
  Evidence: doc diff summary in `Execution Evidence`.
  Handoff: parent only.
  Sign-off: parent confirms no completed-plan scope was reopened.
- [x] Stage 5 - verification.
  Covers: static search, deterministic pytest, compileall, and one gated live
  LLM E2E if deterministic evidence passes.
  Verify: all listed verification commands pass or have explicit documented
  residual risk.
  Evidence: command outputs summarized in `Execution Evidence`.
  Handoff: parent to independent code-review subagent.
  Sign-off: parent confirms verification is ready for review.
- [x] Stage 6 - independent code review.
  Covers: architecture boundary, security cleanup, tests, docs, and evidence.
  Verify: independent reviewer reports no unresolved blockers.
  Evidence: review findings, fixes, reruns, and final review status in
  `Execution Evidence`.
  Handoff: reviewer to parent.
  Sign-off: parent updates lifecycle only after review passes.

## Verification

- `rg -n "validate_patch_artifacts\\(" src/kazusa_ai_chatbot/coding_agent tests/test_coding_agent_phase2_new_artifact_contracts.py tests/test_coding_agent_phase2_new_artifact_e2e_live_llm.py`
  - Expected: no matches after Stage 3. Exit code 1 is acceptable when it means
    the removed boundary is absent.
- `rg -n "_python_test_execution_error|sys\\.executable.*pytest|\"pytest\"" src/kazusa_ai_chatbot/coding_agent/code_writing/patch_validation.py`
  - Expected: no generated-test execution wiring remains in Phase 2 validation.
- `rg -n "materialize_patch_artifacts_for_review\\(|subprocess\\.run|pytest" src/kazusa_ai_chatbot/coding_agent tests/test_coding_agent_phase2_new_artifact_contracts.py tests/test_coding_agent_phase2_new_artifact_e2e_live_llm.py`
  - Expected: no Phase 2 generated-artifact execution path remains. Approved
    matches such as git patch-shape review, source fetching, `rg` source
    inspection, pytest assertion text, and ordinary test runner code must be
    classified in `Execution Evidence`.
- `venv\Scripts\python -m pytest -q tests/test_coding_agent_phase2_new_artifact_contracts.py`
- `venv\Scripts\python -m compileall -q src/kazusa_ai_chatbot/coding_agent tests/test_coding_agent_phase2_new_artifact_contracts.py tests/test_coding_agent_phase2_new_artifact_e2e_live_llm.py`
- Run one Phase 2 E2E live LLM gate only after deterministic Stage 5 evidence
  is complete and inspect generated artifact paths without executing generated
  code.

## Independent Plan Review

Plan review scope:

- Compliance with the Kazusa development-plan lifecycle contract.
- Compliance with parent-led subagent execution gates for implementation.
- Clear change boundary between this plan-review cleanup and later
  production-code implementation.
- Single bigbang cutover with no compatibility wrapper for the removed
  validation boundary.
- Test-first implementation order with evidence checkpoints.
- Verification commands that distinguish allowed read-only subprocess use from
  forbidden generated-artifact execution.

Record plan-review findings, plan edits, residual risks, and readiness status
in `Execution Evidence`.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Use a native independent code-review subagent. If native subagents are
unavailable, pause at this gate and get an explicit user instruction for a
fallback review model.

Review scope:

- Alignment with the architecture security boundary.
- Absence of generated-artifact execution in Phase 2 validation and review
  materialization paths.
- Clear separation between agent-space artifacts and real-world effects.
- No new execution, mutation, install, network, database, service, or patch
  application capability.
- Current Phase 3 background-worker integration remains accurately documented.
- Test and evidence quality.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- The architecture, ICD, code, and tests share one agent-space security
  boundary.
- Phase 2 review/validation does not execute generated code, generated tests,
  generated commands, or generated scripts.
- No Phase 2 callable helper exposes generated-test execution.
- Coding-agent tests prove the non-executing boundary.
- Phase 2 E2E review artifacts identify generated artifact paths for
  inspection without generated-code execution.
- Independent code review finds no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Dormant helper becomes reachable again | Delete execution-capable helper and test current call boundary | Static grep, contract tests, code review |
| Git materialization is mistaken for generated-code execution | Document allowed non-executing patch-shape review separately from forbidden generated-test execution | ICD review and evidence classification |
| Artifact quality becomes harder to judge | Use static inspection, direct artifact review, and AI-authored review | Contract tests and live LLM review artifacts |
| Boundary wording drifts across docs | Align architecture, coding-agent ICD, code-writing ICD, and plan wording | Plan review and grep |

## Execution Evidence

- 2026-07-07: Relevance refresh against current codebase. Active Phase 2.5
  remains needed because `patch_validation.py` still contains
  `validate_patch_artifacts(...)` and `_python_test_execution_error(...)`,
  which can execute patched Python tests with `sys.executable -m pytest`.
  Current `code_writing` production flow calls
  `materialize_patch_artifacts_for_review(...)`, so the immediate task is to
  delete the stale execution-capable helper and prove the current
  materialization-only boundary.
- 2026-07-07: Independent plan-review pass found readiness gaps in the draft:
  single-agent execution conflicted with the execution-gate contract, the
  checklist lacked evidence/handoff/sign-off detail, the legacy-helper decision
  allowed alternate de-exposure paths, and verification did not separate the
  removed boundary check from allowed read-only subprocess matches. The plan
  now requires parent-led native subagents for implementation and independent
  code review, fixed deletion of the execution-capable Phase 2 helper, explicit
  cutover enforcement, contracts/data-shape boundaries, LLM budget scope, a
  staged checklist, and sharper verification commands.
- 2026-07-07: User explicitly requested fallback execution without subagents.
  Plan status moved to `in_progress`; execution proceeds single-agent under the
  same change surface and security boundary.
- 2026-07-07: Stage 1 audit baseline:
  `rg -n "validate_patch_artifacts\\(|materialize_patch_artifacts_for_review\\(|_python_test_execution_error|sys\\.executable|pytest|subprocess\\.run" src/kazusa_ai_chatbot/coding_agent tests/test_coding_agent_phase2_new_artifact_contracts.py tests/test_coding_agent_phase2_new_artifact_e2e_live_llm.py`
  found `validate_patch_artifacts(...)`,
  `_python_test_execution_error(...)`, `sys.executable`, and a `"pytest"`
  command argument in `patch_validation.py`; it also confirmed
  `supervisor.py` calls `materialize_patch_artifacts_for_review(...)`.
- 2026-07-07: Stage 2 test contract added
  `test_patch_validation_exposes_only_review_materialization_boundary`,
  `test_patch_validation_does_not_run_generated_python_tests`, and
  `test_writing_supervisor_uses_review_materialization_boundary_only` in
  `tests/test_coding_agent_phase2_new_artifact_contracts.py`.
  Pre-implementation focused run:
  `venv\Scripts\python -m pytest -q tests/test_coding_agent_phase2_new_artifact_contracts.py::test_patch_validation_exposes_only_review_materialization_boundary tests/test_coding_agent_phase2_new_artifact_contracts.py::test_patch_validation_does_not_run_generated_python_tests tests/test_coding_agent_phase2_new_artifact_contracts.py::test_writing_supervisor_uses_review_materialization_boundary_only`
  returned 2 failed and 1 passed as expected because the legacy callable and
  helper still existed.
- 2026-07-07: Stage 3 implementation removed
  `validate_patch_artifacts(...)`, `_python_test_execution_error(...)`,
  `_sandbox_test_env(...)`, generated-test pytest execution, and the
  execution-only output-compaction helper chain from
  `src/kazusa_ai_chatbot/coding_agent/code_writing/patch_validation.py`.
  `materialize_patch_artifacts_for_review(...)` remains the public review
  boundary and now performs non-executing static inspection after sandbox patch
  materialization. A live-gate retry exposed that `sys` is still needed for
  static import classification through `sys.builtin_module_names`; the import
  was restored without reintroducing `sys.executable`.
- 2026-07-07: Stage 4 documentation alignment updated
  `development_plans/reference/designs/coding_agent_architecture.md` and
  `src/kazusa_ai_chatbot/coding_agent/README.md` so Phase 3 is recorded as
  completed and Phase 2 validation language points to non-executing review
  materialization. `src/kazusa_ai_chatbot/coding_agent/code_writing/README.md`
  already stated that generated code and generated tests are not executed.
- 2026-07-07: Stage 5 deterministic verification:
  focused boundary tests passed 3/3; full
  `venv\Scripts\python -m pytest -q tests/test_coding_agent_phase2_new_artifact_contracts.py`
  passed 19/19; `venv\Scripts\python -m compileall -q src/kazusa_ai_chatbot/coding_agent tests/test_coding_agent_phase2_new_artifact_contracts.py tests/test_coding_agent_phase2_new_artifact_e2e_live_llm.py`
  passed; `rg -n "validate_patch_artifacts\\(" ...` and
  `rg -n "_python_test_execution_error|sys\\.executable.*pytest|\"pytest\"" src/kazusa_ai_chatbot/coding_agent/code_writing/patch_validation.py`
  returned exit 1 with no matches, which is the expected zero-match result.
  Broader static search still reports allowed matches for read-only git/source
  subprocess use, static `pytest.raises` parsing, and pytest test-harness
  annotations.
- 2026-07-07: Stage 5 live LLM verification:
  default pytest marker filtering deselected the case until rerun with
  `-m live_llm`; the first marked run failed with `400 Model unloaded`; a
  second run exposed the static-import `sys` regression noted above; after the
  fix, one run timed out at 304 seconds; a longer single-case run passed:
  `venv\Scripts\python -m pytest -m live_llm -q -s tests/test_coding_agent_phase2_new_artifact_e2e_live_llm.py::test_live_gate_01_single_file_log_counter`
  passed 1/1 in 409.75 seconds. The live run wrote
  `test_artifacts\llm_traces\coding_agent_phase2_e2e\gate_01_response.json`
  and
  `test_artifacts\llm_traces\coding_agent_phase2_e2e\gate_01_materialized_artifacts.json`.
  Manual inspection identified materialized review paths
  `test_artifacts\coding_agent_phase2_e2e_workspace\gate_01\writing_validation\1446af214dee418984a0a3075ebe6e26\src\impl_log_counter.py`
  and
  `test_artifacts\coding_agent_phase2_e2e_workspace\gate_01\writing_validation\1446af214dee418984a0a3075ebe6e26\tests\test_log_counter.py`.
  The artifacts were inspected as inert files only; generated code and tests
  were not executed.
- 2026-07-07: Live artifact inspection found a non-blocking quality limitation:
  the generated source counts prefixes like `DEBUG ` while the generated test
  inputs use `DEBUG:` style lines. This is not a Phase 2.5 security-boundary
  blocker because the plan explicitly avoids executing generated tests; it is
  recorded as artifact-quality evidence for later writing-quality work.
- 2026-07-07: Stage 6 single-agent fallback code review inspected the full
  diff, plan alignment, style constraints, static searches, deterministic test
  results, compile result, and live LLM evidence. Findings: no unresolved
  security-boundary blockers; no production path exposes generated-test
  execution; no patch apply, package install, service startup, database,
  network, or code-execution capability was added; residual risk is limited to
  live artifact quality judging without execution.
