# coding_agent_assessment_gap_phase_b_plan

## Summary

- Goal: Add the mechanical execution feedback loop that lets proposals learn
  from managed apply and focused verification before approval and during repair.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `py-style`,
  `test-style-and-execution`, `local-llm-architecture`
- Overall cutover strategy: Add a coding-agent-owned execution-planning and
  preflight path behind a config flag, then move verification repair evidence to
  a structured failure bundle.
- Highest-risk areas: operation routing rejecting preflight as unsupported,
  running generated code before approval, incorrectly broad verification scope,
  repair loops hiding environment blockers, and worker adapter ownership drift.
- Acceptance criteria: Gates 07, 08, and 09 pass as normal live LLM tests;
  deterministic execution-planning, failure-bundle, and blocker tests pass.

## Context

The assessment correctly identifies the highest-leverage robustness gap:
proposals are delivered after static validation but before any managed apply or
execution evidence. Approval verification specs are selected in the background
worker from approval prose, defaulting to `python_compileall .`, instead of
being derived by the coding agent from changed files and repo tests. Repair sees
bounded redacted excerpts rather than structured failure evidence, and missing
dependencies are treated like repairable source failures.

This phase addresses the testable Loop B gaps:

- `test_live_gate_07_proposal_has_preapproval_preflight_evidence`
- `test_live_gate_08_vague_approval_runs_changed_file_tests`
- `test_live_gate_09_missing_dependency_becomes_typed_blocker`

The first real run of gate 07 did not reach proposal generation. The
background coding operation router rejected the task because the user requested
managed-copy preflight and focused tests, and the current router limits describe
all patch application and execution as unsupported. This phase therefore owns
the operation-router capability wording and tests, not only the later preflight
implementation.

## Mandatory Skills

- `development-plan`: load before changing this plan or executing it.
- `py-style`: load before editing Python production code.
- `test-style-and-execution`: load before changing or running tests.
- `local-llm-architecture`: load before prompt, LLM contract, or agent workflow
  changes.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan
  before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the Independent Code Review gate and record the result in Execution Evidence.
- Use parent-led native subagent execution unless the user explicitly approves
  fallback execution.
- Keep execution argv-only, allowlisted to `python_compileall` and `pytest`,
  bounded by existing timeouts and output caps.
- Preflight execution must use managed copies only and must never mutate the
  original checkout.
- Human approval must still be required for approval-mode verification and any
  future target-side effect.
- Anti-cheat rule: execution specs must be derived from changed files, test
  files, and deterministic repository evidence. Do not satisfy gate 08 by
  keywording the test name, hardcoding fixture paths, or letting L2d/worker
  prose choose the primary verification scope.
- Anti-cheat rule: do not satisfy gate 07 by special-casing the words
  "preflight", "slug", or the fixture name. The route must accept any
  source-backed patch proposal that asks for managed-copy preflight when
  `CODING_AGENT_PREFLIGHT_EXECUTION` is enabled.

## Must Do

- Add `CODING_AGENT_PREFLIGHT_EXECUTION` with default disabled until this phase
  is explicitly enabled for live gates.
- Update the background coding operation router prompt, operation-limit
  payload, and focused tests so managed-copy preflight requests are treated as
  supported `code_modifying` proposal work when the preflight flag is enabled.
- Add focused routing tests proving
  `decide_background_coding_operation(...)` does not return `unsupported` for
  the gate-07 preflight shape when the flag is enabled, and still rejects
  arbitrary live execution, deployment, package installation, and original-tree
  mutation requests.
- Add a preflight apply path that is separate from human approval and valid only
  for managed-copy execution inside the coding-agent workspace.
- Derive default execution specs inside the coding agent from changed files,
  repository evidence, and nearby tests.
- Remove the worker adapter's LLM execution-spec planner from the primary
  verification path; keep only deterministic parsing of explicit user-requested
  extra selectors if needed.
- Run preflight apply and derived focused checks after static validation for
  `propose_patch` objectives when the config flag is enabled.
- Reuse the same derived execution plan for approval verification unless the
  user adds safe extra selectors.
- Replace repair feedback prose with structured failure evidence bundles.
- Add deterministic no-progress and regression stop rules.
- Classify missing external dependencies and missing interpreters as typed
  environment blockers, and do not spend repair attempts on them.
- Run the environment blocker classifier before repair generation. A failure
  classified as `environment_dependency_missing` must set the durable run to
  `blocked`, append a typed blocker, and leave `repair_attempts` empty.
- Convert gates 07, 08, and 09 from known-gap `xfail(strict=True)` to normal
  live gates only after implementation passes them.

## Deferred

- Do not add package installation, arbitrary shell, network checks, private
  source authentication, or non-Python executor plugins.
- Do not add the generic JSON-action loop.
- Do not implement `respond_to_blocker`, L2d affordance binding, source locking,
  approval evidence hardening, or the benchmark harness in this phase.
- Do not broaden preflight beyond managed copies and the existing two execution
  tools.

## Cutover Policy

Introduce the new execution-planning module and route proposal, approval, and
repair callers through it in one change. Remove the adapter-owned LLM planner
from the primary path in the same change so there is one canonical owner for
verification scope.

## Target State

When enabled, a patch proposal reaches `awaiting_approval` only after the
coding agent has applied it in a managed copy, run focused checks derived from
the changed files, and repaired repairable failures within budget. Approval
then reuses that coding-agent-owned execution plan. Environment failures surface
as typed blockers rather than source repairs.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Preflight safety | Use a separate managed-copy preflight authorization | Avoid confusing preflight with human approval. |
| Preflight routing | Treat managed-copy preflight as supported proposal work under the flag | Gate 07 currently fails before proposal generation. |
| Spec ownership | Coding agent derives specs from changed files | Verification scope belongs with the proposal being verified. |
| Worker role | Worker passes explicit user extra checks only | The worker is an adapter, not a verification planner. |
| Failure evidence | Structured bundle plus bounded excerpts | Small models need file, line, kind, and signature, not flattened prose. |
| Environment failures | Typed blocker with zero repair attempts | The agent cannot install dependencies today. |

## Contracts And Data Shapes

Preflight result:

```python
{
    "preflight_enabled": bool,
    "apply_attempts": list[dict[str, object]],
    "execution_attempts": list[dict[str, object]],
    "repair_attempts": list[dict[str, object]],
    "blockers": list[dict[str, object]],
}
```

Derived execution plan:

```python
{
    "origin": "changed_files",
    "specs": [
        {"tool": "python_compileall", "paths": ["package_or_file"]},
        {"tool": "pytest", "pytest_selectors": ["tests/test_x.py"]},
    ],
    "source_paths": list[str],
    "test_paths": list[str],
    "reason": str,
}
```

Failure evidence bundle:

```python
{
    "failure_id": str,
    "tool": "pytest | python_compileall",
    "selector": str,
    "failure_kind": (
        "assertion | exception | import_error | compile_error | timeout | "
        "environment"
    ),
    "exception_type": str,
    "exception_message": str,
    "trace_frames": [
        {"path": str, "line": int, "function": str, "code_line": str},
    ],
    "failure_signature": str,
    "stdout_excerpt": str,
    "stderr_excerpt": str,
}
```

Environment blocker:

```python
{
    "code": "environment_dependency_missing",
    "blocker_kind": "environment",
    "message": str,
    "details": {
        "missing_module": str,
        "tool": str,
        "selector": str,
    },
}
```

## LLM Call And Context Budget

Before this plan, execution repair may call `propose_code_change` up to the
bounded repair limit after approval. After this plan, preflight may perform the
same repair loop before approval when enabled. The repair prompt receives the
failure bundle, relevant bounded source context, protected paths, and previous
patch summary. It must stay under 50k tokens by truncating excerpts, limiting
trace frames, and prioritizing changed files and failing frames.

This plan removes the worker adapter's LLM execution-spec planner from the
primary path, reducing model variance at approval time.

## Change Surface

### Delete

- Remove or retire primary-path use of
  `EXECUTION_SPEC_PLANNER_PROMPT` in
  `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`.

### Modify

- `src/kazusa_ai_chatbot/config.py`: add preflight flag.
- `src/kazusa_ai_chatbot/coding_agent/supervisor.py`: update operation-router
  prompt and operation-limit payload so enabled managed-copy preflight remains
  source-backed proposal work instead of unsupported execution.
- `src/kazusa_ai_chatbot/coding_agent/code_patching/apply.py`: add managed
  preflight apply authorization without weakening human approval apply.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/models.py`: add execution
  plan, failure bundle, and typed blocker shapes.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/supervisor.py`: run
  derived specs, classify failures, enforce budget and stop rules.
- `src/kazusa_ai_chatbot/coding_agent/supervisor.py`: connect proposal
  preflight for `propose_patch` objectives.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/supervisor.py`: persist
  preflight evidence and reuse execution plans at approval.
- `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`: stop owning
  primary execution planning; pass only safe user extra request text and
  explicit safe selectors.
- `tests/test_coding_agent_full_workflow_integration_live_llm.py`: remove
  `xfail` from gates 07, 08, and 09 after implementation.

### Create

- `src/kazusa_ai_chatbot/coding_agent/code_verifying/execution_planning.py`:
  deterministic execution spec derivation and merge logic for explicit safe
  extra selectors.
- Focused tests for spec derivation, failure-bundle parsing, environment
  blocker classification, preflight routing, and no-progress/regression stop
  rules.

### Keep

- Existing executor allowlist and managed apply workspace containment.

## Overdesign Guardrail

- Actual problem: The coding agent delivers unexecuted proposals and then
  verifies with approval-prose-planned checks.
- Minimal change: Add managed-copy preflight execution and coding-agent-owned
  deterministic execution planning, then feed structured failures to repair.
- Ownership boundaries: Coding agent owns proposal verification; worker adapter
  owns queue/runtime handoff; deterministic code owns execution safety and
  failure classification; LLM repair owns source-level semantic edits.
- Rejected complexity: arbitrary shell, installs, network, private credentials,
  generic action loop, and language-plugin executor framework.
- Evidence threshold: Add rejected complexity only after gates 07-09 pass and a
  benchmark shows remaining failures caused by that missing breadth.

## Agent Autonomy Boundaries

- The responsible agent may choose local parser heuristics for failure bundles
  only when deterministic tests cover the supported patterns.
- Changes outside coding-agent, background-worker adapter, config, and tests
  require explicit justification in Execution Evidence.
- The responsible agent must not loosen path safety, output redaction, timeout,
  execution-tool, or protected-path rules.
- If a proposed fix requires package installation or shell access, stop and
  report the blocker.

## Implementation Order

1. Add deterministic tests for preflight routing through
   `decide_background_coding_operation(...)`:
   - Enabled managed-copy preflight plus local source returns `code_modifying`.
   - Arbitrary live execution, deployment, package installation, or original
     checkout mutation still returns `unsupported`.
   - Expected baseline before implementation: the managed-copy preflight shape
     can return `unsupported`.
2. Add deterministic tests for execution-plan derivation from changed files.
3. Add failure-bundle and environment-blocker classifier tests, including the
   missing-module shape from gate 09.
4. Add managed preflight apply authorization and tests.
5. Implement execution-planning module and route approval verification through
   it.
6. Add preflight path behind config flag and persist evidence.
7. Replace repair feedback with structured bundles and add stop rules.
8. Run environment blocker classification before repair generation.
9. Remove primary-path worker LLM spec planning.
10. Convert gates 07, 08, and 09 from xfail and run them one at a time.
11. Run independent code review and address in-scope findings.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes focused tests and expected baselines before
  production implementation starts.
- Production-code subagent: exactly one native subagent after focused tests are
  established; owns production code changes only.
- Independent code-review subagent: exactly one native subagent after planned
  verification passes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Preflight routing tests added and baseline recorded.
- [ ] Execution-plan derivation tests added and baseline recorded.
- [ ] Failure-bundle and environment-blocker tests added.
- [ ] Preflight authorization implemented and tested.
- [ ] Coding-agent-owned execution planning wired.
- [ ] Structured repair evidence and stop rules implemented.
- [ ] Worker primary-path LLM spec planning retired.
- [ ] Gates 07, 08, and 09 converted from xfail and passing.
- [ ] Independent code review completed and findings addressed.
- [ ] Execution Evidence updated.

## Verification

Run deterministic checks first:

```powershell
venv\Scripts\python -m pytest tests/test_coding_agent_phase9_run_supervisor_contracts.py -q
venv\Scripts\python -m pytest tests/test_coding_agent_background_run_contracts.py -q
venv\Scripts\python -m pytest --collect-only tests/test_coding_agent_full_workflow_integration_live_llm.py -q
```

Run live gates one at a time and inspect traces:

```powershell
venv\Scripts\python -m pytest tests/test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_07_proposal_has_preapproval_preflight_evidence -q -m live_llm
venv\Scripts\python -m pytest tests/test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_08_vague_approval_runs_changed_file_tests -q -m live_llm
venv\Scripts\python -m pytest tests/test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_09_missing_dependency_becomes_typed_blocker -q -m live_llm
```

## Independent Code Review

Run this gate after all Verification commands pass and before final sign-off.
The reviewer must inspect preflight safety, approval separation, execution-plan
ownership, failure classification, repair-loop budgets, and public trace
sanitization. The parent agent may fix findings only inside this plan's Change
Surface, then rerun affected verification commands and record the result.

## Acceptance Criteria

This plan is complete when:

- Proposal preflight evidence is present when the config flag is enabled.
- Vague approval runs deterministic focused checks derived from changed files.
- Missing dependency failures produce typed environment blockers and spend zero
  repair attempts.
- Gates 07, 08, and 09 pass without `xfail`.
- Deterministic regression tests and independent code review pass.

## Execution Evidence

- Pending execution.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Preapproval execution runs unsafe commands | Keep argv-only two-tool allowlist and managed copies | Gate 07 plus executor regression tests |
| Vague approval misses tests | Derive specs from changed source and test mapping | Gate 08 |
| Missing dependency is misclassified | Deterministic classifier tests | Gate 09 |
| Repair loops regress passing checks | Regression stop rule | Focused repair-loop tests |
