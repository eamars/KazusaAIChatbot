# coding agent full workflow integration test plan

## Summary

- Goal: define five committed real-LLM full-workflow integration gates for the
  coding agent after Phases 5-9.
- Plan class: large.
- Status: completed.
- Mandatory skills: `development-plan`, `test-style-and-execution`,
  `debug-llm`, `local-llm-architecture`, and `py-style`.
- Overall cutover strategy: test-contract-first; no production behavior changes
  are authorized by this test plan.
- Highest-risk areas: true L2d handoff fidelity, durable follow-up binding,
  approval semantics, managed execution, repair loops, and anti-cheat
  enforcement.
- Acceptance criteria: the five tests are implemented under `tests/`, use the
  real L2d/action-spec/background-worker entry path, run one at a time, emit
  durable raw trace evidence locally, and have agent-authored review evidence
  recorded in the hardening closure record.

## Context

The current coding-agent direct APIs can read code, propose patches, apply
approved patches into managed copies, execute allowlisted checks, verify and
repair, and persist durable coding-run state. The current background-work
coding worker still enters through `handle_background_coding_task(...)`, which
supports read-only answers and review-only proposal artifacts. It does not
currently expose the Phase 9 durable run lifecycle through the L2d accepted-task
and background-worker entrypoint.

The integration tests in this plan intentionally target the desired user-facing
workflow, not the easier direct API workflow. The input boundary is the handoff
that L2d produces as part of accepted background work. The tests must not call a
loose coding-agent subagent interface as the primary entrypoint.

## Mandatory Skills

- `development-plan`: load before editing this plan or any hardening plan.
- `test-style-and-execution`: load before creating, changing, or running any
  deterministic, live DB, or real LLM test.
- `debug-llm`: load before running or reviewing the real LLM gates, and author
  readable review records from raw evidence.
- `local-llm-architecture`: load before changing prompts, LLM routing, worker
  payloads, action-spec contracts, or background-worker semantics.
- `py-style`: load before writing Python test or harness code.

## Mandatory Rules

- After automatic context compaction, the parent or active execution agent must
  reread this entire plan before continuing.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Live LLM gates must run one test case at a time and must be inspected one
  test case at a time.
- Passing pytest is necessary but not sufficient. Gate closure requires
  durable raw evidence plus an agent-authored review record.
- Deterministic handoff assertions belong in deterministic or live DB tests.
  Real LLM tests assert only structural, safety, and contract gates in pytest
  and use review records for quality judgment.
- Live DB usage must be explicit. The pre-integration hardening gates may use
  deterministic in-memory accepted-task/background-job persistence seams when
  they preserve the same public handoff contract. A later live DB E2E pass may
  reuse the same gate cases against MongoDB.
- The tests must use `venv\Scripts\python` for execution commands.

## Anti-Cheat Rules

- The primary test entrypoint must be L2d handoff materialized through
  action-spec execution into accepted-task/background-work state. A test must
  not pass by calling `answer_code_question(...)`, `propose_code_change(...)`,
  `start_coding_run(...)`, `continue_coding_run(...)`, or
  `verify_and_repair_code_change(...)` as the primary workflow entrypoint.
- Real LLM gates must not monkeypatch L2d, the background-work router, the
  coding-agent supervisor, coding PMs, coding programmers, or repair LLM calls.
  Deterministic storage isolation and adapter test harnessing are allowed only
  when they preserve the same public handoff contract.
- Test fixtures must be defined before implementation. Expected behavior may be
  tightened after review, but it must not be relaxed after seeing a failure
  unless the user explicitly accepts that gate exclusion.
- Tests must not use `skip`, `xfail`, broad markers, or environment guards to
  hide a required gate. A missing runtime dependency must fail with a clear
  setup error or be documented as a blocked run.
- Tests must not assert only that a trace file exists. They must verify the
  required public states, redaction rules, managed workspace boundaries, and
  durable run lifecycle fields.
- Tests must not edit fixture source, generated patch artifacts, execution
  specs, or review criteria during a gate run to make the current output pass.
- Tests must not broaden pytest selectors, remove failing assertions, or accept
  test-file edits as a repair success for source-repair gates.
- Tests must not inspect or depend on `.env`, absolute workspace roots, raw
  local paths, hidden prompts, raw full command output, `.git` internals, or
  secret-like file contents.

## Must Do

- Create `tests/test_coding_agent_full_workflow_integration_live_llm.py`.
- Create durable raw evidence under `test_artifacts/llm_traces/` with a
  stable coding-agent full-workflow test-name prefix.
- Create one agent-authored review record per real LLM gate.
- Mark each gate with `live_llm`; add `live_db` only for variants that use
  MongoDB persistence.
- Exercise the L2d handoff, accepted-task lifecycle, background-work queue,
  background worker tick, coding worker, coding run state, and result-ready
  handoff where the case requires them.
- Include at least three multi-turn cases with two or more follow-up requests.

## Deferred

- Do not implement hardening changes from this file alone.
- Do not add a UI, adapter delivery test, Git push behavior, package
  installation, arbitrary shell execution, or Phase 10 repository-scale reading.
- Do not test broad repository intelligence as a substitute for full workflow
  state and continuation behavior.

## Cutover Policy

Overall strategy: test-contract-first.

| Area | Policy | Instruction |
|---|---|---|
| Production behavior | bigbang | This plan adds no production behavior. Production hardening requires a separate approved plan. |
| Test entrypoint | bigbang | Use L2d/action-spec/background-worker entry only. Do not preserve direct-API shortcuts for full workflow tests. |
| Evidence artifacts | bigbang | Generate raw trace evidence under `test_artifacts/llm_traces/`; commit executable tests and closure review records, not ignored raw traces. |
| Existing Phase 9 tests | compatible | Keep direct Phase 9 tests as module-level coverage; do not treat them as full integration proof. |

## Target State

The repository contains five executable real LLM integration gates that mimic
regular Codex-style coding tasks from simple to hard. The gates prove whether
Kazusa can carry coding work from L2d accepted-task handoff through background
execution and follow-up behavior, including managed apply, execution, repair,
durable run state, and public-safe result projection.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Test boundary | Start from L2d handoff/action-spec execution, not direct coding-agent APIs. | The user requested the real integration point. |
| Persistence | Use deterministic storage seams for pre-integration LLM gates; reserve `live_db` for the later DB E2E pass. | The gates must prove the handoff path while keeping LLM regressions isolated from external DB availability. |
| LLM behavior | Use real LLM calls for L2d and coding-agent prompts. | Mocked LLM output cannot prove routing or coding quality. |
| Assertions | Use structural hard gates plus review rubrics. | Local LLM output varies and should not be pinned to exact prose. |
| Follow-up model | Multi-turn gates must reuse durable run state. | Codex-like behavior depends on continuity across requests. |

## Contracts And Data Shapes

Each gate must capture this evidence shape in raw JSON:

```json
{
  "case_id": "string",
  "turns": [],
  "l2d_action_specs": [],
  "accepted_task_states": [],
  "background_jobs": [],
  "worker_results": [],
  "coding_run_public_projection": {},
  "final_dialog_or_result_ready_surface": [],
  "hard_gate_results": [],
  "forbidden_failure_modes": [],
  "model_routes": []
}
```

The review record must contain:

- run context and command;
- fixture identity;
- raw user turns;
- interpreted action/result flow;
- selected worker and coding operation;
- coding run status and attempts when present;
- public safety and redaction judgment;
- behavior quality judgment;
- raw evidence path.

## LLM Call And Context Budget

These tests intentionally use the production L2d and coding-agent model routes.
They do not add production LLM calls. Each gate may invoke:

- L2d action selection for each user turn;
- background-work router when the hardening plan has not yet bound coding work
  deterministically;
- coding-agent background or run-supervisor prompts;
- code-reading, code-writing, code-modifying, and repair prompts depending on
  the case;
- L3/dialog only when verifying result-ready handoff wording.

The tests must record route names and model metadata when available. They must
not increase prompt budgets to make a case pass unless a separate approved
hardening plan changes the production budget.

## Change Surface

### Create

- `tests/test_coding_agent_full_workflow_integration_live_llm.py`: five
  real-LLM integration gates.
- `test_artifacts/llm_traces/`: generated local raw evidence for closure,
  using a stable coding-agent full-workflow test-name prefix.

### Modify

- `development_plans/README.md`: record this active test plan while it is open.

### Keep

- Existing Phase 5-9 direct API tests remain module-level coverage.

## Overdesign Guardrail

- Actual problem: direct Phase 9 tests do not prove the requested full
  L2d-to-background-worker coding workflow.
- Minimal change: create five integration gates that enter through the real
  accepted-task/background-work boundary and capture evidence.
- Ownership boundaries: L2d selects user-facing delayed work; deterministic
  action-spec and background-work code own lifecycle and persistence; coding
  agents own coding semantics; deterministic apply/execution code owns side
  effects; L3/dialog owns visible wording.
- Rejected complexity: no UI automation, no adapter delivery sends, no package
  installation, no arbitrary shell, no Phase 10 broad reading, and no direct API
  shortcuts.
- Evidence threshold: add broader tests only after one of the five gates exposes
  a missing full-workflow risk that this plan does not cover.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper layout only when the helper
  preserves the entrypoint and anti-cheat rules in this plan.
- The responsible agent must not add production behavior while implementing
  this test plan.
- The responsible agent must not relax gate criteria after observing failures.
- The responsible agent must not introduce alternate entrypoints, compatibility
  shims, fallback paths, or hidden fixture mutation.
- If a gate cannot be implemented without production hardening, leave the test
  failing or blocked and record the exact hardening requirement.

## Implementation Order

1. Add the real LLM test file with five collected tests and no production code
   changes.
2. Run collection only and record missing marker or import issues.
3. Run Gate 01 before hardening and record the current behavior.
4. Run Gates 02-05 before hardening only far enough to record the first
   architectural blocker for each gate.
5. Hand the blockers to
   `coding_agent_pre_integration_hardening_plan.md`.
6. After the hardening plan is implemented, rerun all five gates one at a time.
7. Author one review record per gate from raw evidence.
8. Run independent code review against tests, artifacts, and
   anti-cheat compliance.

## Integration Gates

### Gate 01 - read-only codebase question

- User turn: "In this fixture repo, explain where the CLI discovers commands."
- Fixture: a small CLI package with `src/tooling/cli.py`,
  `src/tooling/commands.py`, and `tests/test_cli.py`.
- Entry: L2d handoff creates accepted background work; background worker
  executes the coding worker.
- Hard gates:
  - accepted task reaches `result_ready` or an equivalent completed result
    state;
  - worker is `coding_agent`;
  - output is read-only and contains evidence references;
  - no patch artifacts, apply attempts, execution attempts, or repair attempts
    are recorded;
  - public output does not expose local roots.
- Behavior rubric: answer identifies the command discovery owner and uses
  evidence from the fixture source.

### Gate 02 - source-free new artifact with follow-ups

- Turns:
  1. "Create a small Python CSV normalizer CLI with tests."
  2. "Add a dry-run mode and make the output deterministic."
  3. "Summarize the files I should review."
- Fixture: source-free task; the workspace must create only review artifacts
  until approval.
- Entry: every turn enters through L2d handoff and background work.
- Hard gates:
  - the first turn creates a durable coding run or equivalent review state;
  - follow-ups bind to the same coding task or run;
  - patch/proposal metadata lists created files without raw file dumps;
  - no apply or execution occurs before structured approval;
  - final summary references reviewable files.
- Behavior rubric: follow-up changes are incorporated without losing the
  original CSV-normalizer intent.

### Gate 03 - existing-source proposal with follow-ups

- Turns:
  1. "In this fixture repo, add JSON output to the counter CLI."
  2. "Keep tests unchanged; propose only runtime source changes."
  3. "What exact files changed and why?"
- Fixture: `counter_cli` source with a CLI module and existing tests.
- Entry: L2d handoff and background work.
- Hard gates:
  - the source-backed request resolves the fixture repo;
  - the proposal is for existing source and remains review-only;
  - test files are not listed as changed files;
  - evidence references are repo-relative;
  - the follow-up file summary is grounded in proposal metadata.
- Behavior rubric: the model respects the "do not edit tests" follow-up and
  explains runtime file ownership.

### Gate 04 - approval, verify, and repair

- Turns:
  1. "Fix the slug normalization bug in this fixture repo."
  2. "Approve the patch and run the focused slug pytest."
  3. "If it fails, repair the source without editing tests."
- Fixture: slug package with one failing behavior seeded in source and a
  focused pytest selector.
- Entry: L2d handoff and background work for all turns.
- Hard gates:
  - proposal pauses before approval;
  - approval is represented as structured trusted data before managed apply;
  - apply happens only in a managed copy;
  - execution uses allowlisted focused pytest;
  - repair feedback is redacted and bounded;
  - final public projection contains attempt history and no absolute paths.
- Behavior rubric: the final source fix targets the slug bug and does not solve
  the gate by editing tests.

### Gate 05 - hard multi-file workflow with cancellation/status

- Turns:
  1. "Fix the release feed cache timeout and CLI flag behavior in this repo."
  2. "Proceed with approval; run the focused feed and CLI tests."
  3. "After completion, show the attempt history and final changed source files."
  4. "Cancel any remaining work if protected tests would need edits."
  5. "Give me the final status."
- Fixture: multi-file package with cache, feed, CLI, and protected tests.
- Entry: L2d handoff and background work.
- Hard gates:
  - one run identity survives all follow-ups;
  - protected tests are never modified;
  - apply/execution/repair attempts are recorded durably;
  - cancellation is accepted only when the run is non-terminal;
  - final status is consistent with the durable run state;
  - raw command output and local roots remain hidden.
- Behavior rubric: the agent behaves like a bounded local Codex workflow: it
  keeps state, handles approval, validates, repairs when allowed, and reports
  status without inventing unavailable actions.

## Execution Model

- Parent agent owns the test contract, fixture creation, real LLM trace
  evidence, review records, and final sign-off.
- No production-code subagent is required for this test-plan drafting step.
- If this plan is later approved for test implementation, the parent must add
  fixtures and tests before production hardening work starts.
- If native subagent capability is unavailable during later execution, stop
  before implementation unless the user explicitly approves fallback execution.

## Progress Checklist

- [x] Stage 1 - fixture repositories specified and created
  - Covers: all five gate fixtures.
  - Verify: fixture files exist and contain no secrets or `.env` references.
  - Evidence: fixture intent recorded in the test file; each local source gate
    creates a GitHub-backed temp checkout.
  - Sign-off: completed 2026-07-09.
- [x] Stage 2 - real LLM test harness created
  - Covers: `tests/test_coding_agent_full_workflow_integration_live_llm.py`.
  - Verify: collection succeeds for the five named tests.
  - Evidence: collection command output recorded in execution evidence.
  - Sign-off: completed 2026-07-09.
- [x] Stage 3 - gates run one at a time
  - Covers: Gate 01 through Gate 05.
  - Verify: each gate command runs individually with `-q -s`.
  - Evidence: raw JSON generated locally under `test_artifacts/llm_traces/`;
    review outcomes recorded in the hardening plan closure evidence.
  - Sign-off: completed 2026-07-09.
- [x] Stage 4 - independent code review
  - Covers: test files, fixtures, artifact allowlist, and evidence.
  - Verify: review findings are recorded and remediated or explicitly blocked.
  - Evidence: review result recorded before completion in the hardening plan.
  - Sign-off: completed 2026-07-09.

## Verification

Run these commands only after the tests and fixtures are implemented:

- `venv\Scripts\python -m pytest tests\test_coding_agent_full_workflow_integration_live_llm.py --collect-only -q`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_01_read_only_question_from_l2d_to_worker -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_02_source_free_proposal_from_l2d_to_worker -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_03_existing_source_proposal_then_status_followup -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_04_approval_verify_followup_from_l2d_to_worker -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_05_cancel_followup_from_l2d_to_worker -q -s`

## Independent Plan Review

Review performed during drafting against the current coding-agent,
background-work, accepted-task, and Phase 9 direct-run contracts.

### Surfaced Issues

| Issue | Severity | Required Resolution |
|---|---|---|
| Gates 02-05 require durable coding-run continuation through background work, which current code does not expose. | blocker | Resolve through `coding_agent_pre_integration_hardening_plan.md`. |
| Gates 04-05 require structured approval and allowed execution specs from follow-up text. | blocker | Add deterministic approval validation and worker-local execution-spec planning before closure. |
| Real full workflow tests need live DB state because accepted tasks and background jobs are persistence-backed. | non-blocking | Mark gates `live_db` and use isolated test identities. |
| Direct Phase 9 API calls could make integration tests pass without proving the requested entrypoint. | blocker | Enforce anti-cheat greps and test call-path review. |
| Result-ready dialog could hide missing worker metadata. | non-blocking | Assert accepted-task/job metadata and coding-run public projection, not final prose only. |

### Review Result

The test plan is draft-ready. It intentionally exposes current architectural
blockers instead of weakening the gates to match current code.

## Independent Code Review

Run this gate after all test implementation verification passes and before
marking this test plan complete. The reviewer must inspect:

- anti-cheat compliance;
- real LLM test style;
- live DB setup and cleanup;
- fixture integrity;
- raw evidence completeness;
- review records;
- whether the tests prove the full L2d/background-worker entrypoint instead of
  a direct coding-agent shortcut.

## Acceptance Criteria

This test plan is complete when:

- five real LLM gates are committed under `tests/`;
- fixture repositories are committed under `tests/fixtures/`;
- every gate has durable raw evidence and an agent-authored review record;
- every gate enters through the L2d/action-spec/background-work boundary;
- at least three gates include two or more follow-up requests;
- anti-cheat rules are enforced by code review and test assertions;
- all accepted exclusions are explicitly documented by the user.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Tests accidentally call direct APIs | Anti-cheat review and entrypoint assertions | Inspect test call graph and raw evidence |
| Local LLM variance creates brittle failures | Structural hard gates plus review rubrics | One-at-a-time live LLM run review |
| Live DB state leaks between gates | Use isolated test identities and cleanup | Query accepted-task/job state per case |
| Current code cannot pass full workflow | Keep failing tests as valid blockers | Map failure to hardening plan |

## Execution Evidence

- Draft created: committed during Phase 9 closeout.
- Plan review: current blockers surfaced and linked to the hardening plan.
- Test implementation evidence: completed by
  `tests/test_coding_agent_full_workflow_integration_live_llm.py`.
- Collection evidence:
  `venv\Scripts\python -m pytest tests\test_coding_agent_full_workflow_integration_live_llm.py -m live_llm --collect-only -q`
  collected five gates.
- Real LLM execution evidence: all five gates passed one at a time on
  2026-07-09. Raw traces were generated under `test_artifacts/llm_traces/` and
  are intentionally ignored by repository policy; gate review outcomes and
  remediations are recorded in
  `coding_agent_pre_integration_hardening_plan.md`.
