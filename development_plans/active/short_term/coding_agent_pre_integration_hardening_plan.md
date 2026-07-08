# coding agent pre-integration hardening plan

## Summary

- Goal: make the Phase 5-9 coding-agent workflow reachable from the real
  L2d accepted-task/background-worker entrypoint and able to pass the full
  workflow integration gates.
- Plan class: large.
- Status: draft.
- Mandatory skills: `development-plan`, `test-style-and-execution`,
  `debug-llm`, `local-llm-architecture`, and `py-style`.
- Overall cutover strategy: compatible extension with no fallback shims.
- Highest-risk areas: accepted-task lifecycle semantics, structured approval,
  execution-spec planning, durable run references, background-worker side
  effects, and real LLM routing reliability.
- Acceptance criteria: the five full workflow integration gates pass one at a
  time with raw evidence and review artifacts, and current Phase 5-9 direct API
  regressions still pass.

## Context

Phase 9 added durable direct coding-run APIs. The direct run supervisor can
start read-only, proposal, and verify/repair runs; continue awaiting-approval
runs; cancel runs; and reload public run projections. The integration point the
user requires is broader: L2d must hand off accepted coding work into
background work, and follow-up requests must continue the same coding task
without calling a loose subagent interface.

Current code does not yet satisfy that full entrypoint. The background coding
worker calls `handle_background_coding_task(...)`, which can answer code
questions or produce review-only proposals. The direct `coding_run` API is not
bound to accepted tasks, background jobs, result-ready cognition, approval
follow-ups, or status checks.

This plan hardens the integration boundary before the full integration tests
are run as closure gates.

## Mandatory Skills

- `development-plan`: load before editing this plan, implementing it, reviewing
  it, or changing lifecycle status.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running real LLM gates and before authoring review
  artifacts from raw evidence.
- `local-llm-architecture`: load before changing L2d/action-spec prompts,
  background routing, coding worker prompts, or coding-run continuation
  semantics.
- `py-style`: load before editing Python files.

## Mandatory Rules

- After automatic context compaction, the parent or active execution agent must
  reread this entire plan before continuing.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle updates, merge, or sign-off, run the
  independent code review gate and record findings in `Execution Evidence`.
- Execution uses parent-led native subagents by default. If native subagents are
  unavailable, stop unless the user explicitly approves fallback execution.
- The full workflow tests must be written before production implementation.
- Real LLM gates must run one case at a time and be inspected one case at a
  time.
- Deterministic code owns permissions, lifecycle state, duplicate identity,
  structured approval validation, allowed execution tools, path containment,
  and public sanitization.
- LLM stages own semantic task interpretation, source-owner reasoning, coding
  plan quality, and bounded repair reasoning only.
- Background coding execution must never mutate original source checkouts,
  install packages, run arbitrary shell commands, send adapter text directly,
  or infer approval from casual prose.

## Anti-Cheat Rules

- Full integration closure must use the tests in
  `coding_agent_full_workflow_integration_test_plan.md`. Direct Phase 9 API
  tests are useful regression coverage but cannot close this plan.
- Do not patch or fake L2d, background-work router behavior, coding-agent PMs,
  coding programmers, or repair LLM calls in real LLM gates.
- Do not introduce a hidden direct-API shortcut from the tests to
  `start_coding_run(...)`, `continue_coding_run(...)`, or
  `verify_and_repair_code_change(...)`.
- Do not pass approval from free-form chat text into apply/execute boundaries
  without a deterministic structured approval object.
- Do not generate arbitrary command lines. Execution specs must remain closed
  to `python_compileall` and focused `pytest` selectors.
- Do not mark a gate passing from harness success alone. Raw evidence and
  agent-authored review must support the behavior judgment.
- Do not relax failed gate criteria, broaden selectors, skip repair assertions,
  remove follow-up turns, or change fixture bugs after observing a failure.
- Do not allow the coding worker to expose local roots, workspace roots,
  `.env`, `.git`, raw full command output, raw diffs in metadata, or full source
  dumps.

## Must Do

- Implement the five full workflow integration tests before production
  hardening starts.
- Add a reviewed coding-specific accepted-task/background-work contract that
  can bind a coding task to the `coding_agent` worker deterministically.
- Preserve the existing review-only background coding path for generic coding
  jobs while adding a distinct durable coding-run path.
- Store and surface a prompt-safe coding run reference so follow-up requests
  can continue the same run.
- Add structured follow-up actions for approval, cancellation, status, and
  continuation through accepted-task/background-work state.
- Add worker-local planning for allowed verification specs when the user asks
  to run focused tests or compile checks, with deterministic validation before
  execution.
- Route approved verification through the existing `coding_run` and
  `verify_and_repair_code_change(...)` boundaries.
- Keep apply, execution, and repair inside managed coding workspaces and leave
  original source checkouts unchanged.
- Update ICDs, architecture references, and HOWTO text to match implemented
  contracts.
- Run Phase 5-9 deterministic regressions and the five real LLM integration
  gates before closure.

## Deferred

- Do not add arbitrary shell execution, package installation, network access,
  repository push, deployment, or adapter direct-send behavior.
- Do not implement Phase 10 repository-scale reading.
- Do not add UI controls for coding runs.
- Do not add compatibility aliases or fallback mappers from stale direct API
  request shapes.
- Do not move generic L2d routing, delivery, permission, or persistence
  decisions into a coding prompt.

## Cutover Policy

Overall strategy: compatible extension with no fallback shims.

| Area | Policy | Instruction |
|---|---|---|
| Existing background coding worker | compatible | Preserve read-only and review-only proposal behavior for generic jobs. |
| Durable coding-run background path | bigbang | Add one explicit contract. Do not route through hidden direct shortcuts or fallback mappers. |
| Requested worker validation | bigbang | Permit `coding_agent` only for the new validated coding payload. Reject unvalidated worker-local fields. |
| Approval and execution | bigbang | Require structured approval and closed execution specs before managed apply/execution. |
| Tests | bigbang | Close with the full workflow integration gates, not direct API substitutes. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- Bigbang areas must be implemented as one canonical contract, not as shims.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

Supported coding work can start from L2d accepted-task handoff, run in the
background worker, create or continue a durable coding run, pause for approval,
apply approved patches into managed copies, execute allowlisted verification,
repair within caps, expose public-safe attempt history, and accept follow-up
status/cancel requests through the same lifecycle.

The system remains bounded for local LLMs. L2d selects a semantic coding-task
handoff. The coding worker and coding-agent prompts interpret coding intent.
Deterministic owners validate approval, persistence, path containment, and
execution.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Entry contract | Add a coding-specific accepted-task/background-worker contract. | Generic background routing cannot safely carry approval and run continuation. |
| Worker binding | Use deterministic `requested_worker="coding_agent"` only after action-spec validation. | Reduces local LLM route variance and prevents worker-local leakage. |
| Run reference | Persist a prompt-safe `coding_run_ref` in accepted-task/job metadata. | Follow-ups need stable binding without relying on chat memory. |
| Approval | Convert only explicit approval actions into structured approval objects. | Apply/execute boundaries must not infer approval from prose. |
| Execution specs | Add worker-local allowed-spec planning plus deterministic validation. | Users naturally request "run pytest"; direct APIs require structured specs. |
| Legacy behavior | Keep existing review-only generic coding jobs. | Existing accepted coding tasks still need read/proposal behavior. |

## Contracts And Data Shapes

New validated worker payload:

```json
{
  "schema_version": "coding_agent_background_run.v1",
  "operation": "start_run | continue_run | get_run",
  "task_brief": "string",
  "coding_run_ref": "coding_run:<run_id>",
  "follow_up_text": "string",
  "requested_action": "approve_and_verify | cancel | status | none",
  "source_scope": {},
  "approval": {},
  "execution_request_text": "string"
}
```

The final implementation may split this shape across typed request classes, but
it must preserve these semantics:

- `schema_version` is required for all durable coding background payloads.
- `operation` is closed and validated before worker dispatch.
- `coding_run_ref` is required for continuation and status operations.
- `approval` is created only by deterministic code after an explicit user
  approval action.
- `execution_request_text` may enter only the coding worker's verification-spec
  planner; deterministic validation must reject unsupported tools, paths, and
  selectors.

Prompt-safe worker metadata must include:

```json
{
  "schema_version": "coding_agent_worker_metadata.v2",
  "coding_operation": "code_reading | code_writing | code_modifying | coding_run",
  "coding_run_ref": "coding_run:<run_id>",
  "run_status": "string",
  "allowed_next_actions": [],
  "changed_files": [],
  "attempt_summaries": [],
  "evidence_refs": []
}
```

Worker metadata must not include absolute roots, raw diffs, full source,
unbounded command output, cache keys, `.env`, `.git`, or secret-like content.

## LLM Call And Context Budget

Before hardening:

- L2d may select accepted-task actions.
- Background-work router may select `coding_agent`.
- Coding-agent background router selects reading, writing, modifying, or
  unsupported.
- Direct coding-run calls are outside background work.

After hardening:

- L2d selects a coding accepted-task/follow-up action when the user asks for
  coding work or coding-run continuation.
- Deterministic action execution binds the coding worker.
- Coding worker may call a bounded coding-run operation classifier and a
  bounded verification-spec planner when continuation requires execution.
- Existing coding PM/programmer calls remain unchanged.

All new coding-worker LLM calls are background calls. They must use the coding
PM route or an existing coding route, stay within current model budgets, and
receive concise semantic text plus public run summaries, not raw source trees or
raw command output.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/action_spec/registry.py`: expose coding accepted-task
  capability or follow-up capability to L2d.
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`: validate the new action
  kind.
- `src/kazusa_ai_chatbot/action_spec/execution.py`: execute the new action via
  accepted-task/background-work materialization.
- `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py`: build the
  validated coding worker payload and accepted-task identity.
- `src/kazusa_ai_chatbot/background_work/jobs.py`: allow
  `requested_worker="coding_agent"` only for the validated coding payload.
- `src/kazusa_ai_chatbot/background_work/worker.py`: preserve worker payload
  and metadata needed for coding-run lifecycle and result-ready handling.
- `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`: dispatch
  durable coding-run operations in addition to legacy read/proposal behavior.
- `src/kazusa_ai_chatbot/coding_agent/`: add only the minimal worker-local
  helper or module needed to classify durable background coding run operations
  and build allowed execution specs.
- `src/kazusa_ai_chatbot/accepted_task/`: add prompt-safe run-reference status
  projection only if current state/result fields cannot carry it safely.
- `README.md`, `docs/HOWTO.md`,
  `src/kazusa_ai_chatbot/background_work/README.md`,
  `src/kazusa_ai_chatbot/coding_agent/README.md`, and
  `development_plans/reference/designs/coding_agent_architecture.md`: update
  docs after implementation.

### Create

- `tests/test_coding_agent_full_workflow_integration_live_llm.py`.
- `tests/test_coding_agent_background_run_contracts.py`.
- `tests/fixtures/coding_agent_full_workflow/`.
- `test_artifacts/llm_traces/coding_agent_full_workflow/`.

### Keep

- Existing direct Phase 5-9 APIs and tests.
- Existing review-only generic background coding behavior.
- Existing apply/execution/verify safety boundaries.

## Overdesign Guardrail

- Actual problem: the durable Phase 9 coding run is not reachable through the
  required L2d accepted-task/background-worker integration point.
- Minimal change: add one validated coding-run background contract with stable
  run references, structured approval, allowed execution spec planning, and
  result/status projection.
- Ownership boundaries: L2d owns semantic request/follow-up selection;
  action-spec owns validation and accepted-task materialization; background
  work owns queue and worker dispatch; coding worker owns coding-run
  interpretation; deterministic coding APIs own apply, execution, repair, and
  sanitization.
- Rejected complexity: no UI, no arbitrary shell, no package install, no Phase
  10 reading, no adapter send path, no generic tool-permission system, no
  compatibility aliases, and no hidden direct API shortcuts.
- Evidence threshold: add more worker modes or prompt fields only if one of the
  five integration gates fails because the minimal contract cannot represent a
  required supported workflow.

## Agent Autonomy Boundaries

- The responsible agent may choose internal function names only when the public
  contracts in this plan remain unchanged.
- The responsible agent must not add side-effect behavior outside the coding
  worker, managed apply, managed execution, or verify/repair owners named here.
- The responsible agent must search for existing helpers before adding new
  helpers.
- The responsible agent must not add feature flags, compatibility bridges,
  alternate payload shapes, or fallback paths unless this plan is updated and
  re-approved.
- The responsible agent must stop if accepted-task lifecycle changes require a
  data migration not described in this plan.

## Implementation Order

1. Add the five real LLM integration tests and fixtures from the integration
   test plan.
   - Expected before implementation: Gate 01 may pass or partially pass; Gates
     02-05 fail or block because durable coding-run continuation is not wired
     through background work.
2. Add deterministic contract tests for validated coding worker payloads and
   `requested_worker="coding_agent"` queue validation.
3. Add the action-spec contract for coding accepted-task/follow-up requests.
4. Bind validated coding action execution to accepted-task creation and
   background queue rows.
5. Add coding worker dispatch for durable `start_run`, `continue_run`, and
   `get_run` operations.
6. Add run-reference projection into accepted-task status/result metadata.
7. Add worker-local verification-spec planning and deterministic validation for
   focused `pytest` and `python_compileall`.
8. Run deterministic contract tests and Phase 5-9 regressions.
9. Run the five real LLM integration gates one at a time and author review
   artifacts.
10. Update documentation and architecture references.
11. Run independent code review, remediate findings, and rerun affected
    verification.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records expected
  failures before production implementation starts.
- Production-code subagent: exactly one native subagent after the focused test
  contract is established; it owns production code changes only.
- Independent code-review subagent: exactly one native subagent after planned
  verification passes; it reviews the plan, diff, and evidence.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - integration tests and fixtures exist
  - Covers: implementation order step 1.
  - Verify: collect the five live LLM tests and record current fail/block
    behavior.
  - Evidence: collection output and baseline notes.
  - Sign-off: pending.
- [ ] Stage 2 - deterministic contract tests exist
  - Covers: implementation order step 2.
  - Verify: focused contract tests fail for missing payload/queue/action
    behavior before implementation.
  - Evidence: failing output recorded.
  - Sign-off: pending.
- [ ] Stage 3 - action-spec and queue hardening complete
  - Covers: implementation order steps 3-4.
  - Verify: focused action-spec and queue tests pass.
  - Evidence: test output recorded.
  - Sign-off: pending.
- [ ] Stage 4 - coding worker durable run dispatch complete
  - Covers: implementation order steps 5-7.
  - Verify: worker contract tests and Phase 9 direct run tests pass.
  - Evidence: test output recorded.
  - Sign-off: pending.
- [ ] Stage 5 - full regression and real LLM gates complete
  - Covers: implementation order steps 8-10.
  - Verify: deterministic regressions plus five one-at-a-time live LLM gates.
  - Evidence: raw JSON and review markdown per gate.
  - Sign-off: pending.
- [ ] Stage 6 - independent code review complete
  - Covers: implementation order step 11.
  - Verify: review findings recorded, fixes applied, affected tests rerun.
  - Evidence: review outcome and rerun output.
  - Sign-off: pending.

## Verification

### Static Greps

- `rg "coding_agent_background_run.v1" src tests development_plans`
  - Expected after implementation: matches in the validated contract, tests,
    and docs only.
- `rg "start_coding_run\\(|continue_coding_run\\(|verify_and_repair_code_change\\(" tests\\test_coding_agent_full_workflow_integration_live_llm.py`
  - Expected: no matches. A nonzero `rg` exit code is acceptable and means the
    anti-cheat direct API shortcut is absent.

### Deterministic Tests

- `venv\Scripts\python -m pytest tests\test_coding_agent_background_run_contracts.py -q`
- `venv\Scripts\python -m pytest tests\test_background_work_jobs.py tests\test_background_work_coding_agent.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_phase5_patch_apply_contracts.py tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase8_verify_repair_contracts.py tests\test_coding_agent_phase9_run_supervisor_contracts.py tests\test_coding_agent_phase9_e2e_workflows.py -q`

### Real LLM Gates

Run these one at a time and inspect each artifact before starting the next:

- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_gate_01_read_only_codebase_question -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_gate_02_source_free_new_artifact_followups -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_gate_03_existing_source_proposal_followups -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_gate_04_approval_verify_repair_followups -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_gate_05_hard_multi_file_status_and_cancel -q -s`

## Independent Plan Review

Review performed during drafting against the full integration gates.

### Surfaced Issues

| Issue | Severity | Required Resolution |
|---|---|---|
| Background coding worker cannot continue durable coding runs today. | blocker | Add durable coding-run worker payload and dispatch. |
| Generic background router selection is too variable for required coding tests. | blocker | Bind coding worker deterministically after validated action-spec handoff. |
| Follow-up approval cannot target a run today. | blocker | Persist and project prompt-safe `coding_run_ref`. |
| Approval could be misread from prose if not structured. | blocker | Require deterministic structured approval before apply/execute. |
| Focused pytest requests lack a background execution-spec planner. | blocker | Add worker-local allowed-spec planning plus deterministic validation. |
| Accepted-task lifecycle may mark proposal work delivered before approval. | blocker | Keep non-terminal coding runs discoverable for follow-up/status. |
| Direct Phase 9 tests can mask missing integration. | blocker | Enforce anti-cheat entrypoint rules and full workflow gates. |
| Raw paths or command output may leak through new metadata. | blocker | Add v2 metadata sanitizer tests and review gates. |

### Review Result

The plan is draft-ready after converting all surfaced issues into `Must Do`,
`Contracts And Data Shapes`, `Change Surface`, `Verification`, and
`Anti-Cheat Rules`. It is not approved for execution until the user explicitly
approves production-code hardening.

## Independent Code Review

Run this gate after all verification commands pass and before final sign-off.
The reviewer must inspect:

- plan alignment and anti-cheat compliance;
- action-spec and accepted-task ownership;
- worker payload validation;
- coding-run metadata sanitization;
- structured approval and execution-spec validation;
- absence of direct API shortcuts in integration tests;
- real LLM trace and review evidence.

The parent agent may fix review findings only within this plan's change
surface. Findings that require new public contracts, data migrations, or
broader behavior changes must stop execution until the plan is updated and
approved.

## Acceptance Criteria

This plan is complete when:

- the full workflow integration tests exist and were established before
  production hardening;
- the coding worker can start, continue, get, approve, verify, repair, cancel,
  and report durable coding runs from accepted-task/background-work handoff;
- follow-up requests bind to the same run through prompt-safe metadata;
- apply/execute/repair still run only in managed copies and never mutate
  original source;
- Phase 5-9 deterministic regressions pass;
- all five real LLM integration gates pass one at a time with raw evidence and
  agent-authored review;
- independent code review finds no unresolved blockers.

## Execution Evidence

- Draft created: pending commit.
- Plan review: blockers surfaced and converted into required hardening scope.
- Implementation evidence: pending future execution.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Side-effect boundary expands too broadly | Keep apply/execute inside existing managed APIs | Phase 5-9 regressions and metadata review |
| Local LLM misroutes follow-up actions | Deterministic run binding plus closed action values | Full workflow follow-up gates |
| Approval inferred unsafely | Structured approval object required | Contract tests and Gate 04 |
| Execution spec planner emits unsafe commands | Closed validation to compileall/pytest only | Contract tests and static review |
| Non-terminal accepted task state is ambiguous | Persist prompt-safe run status and allowed next actions | Status/follow-up gates |
