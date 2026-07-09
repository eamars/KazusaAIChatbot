# coding agent assessment gap gate 06-10 failure modes

## Run Context

| Field | Value |
| --- | --- |
| Run date | 2026-07-09 |
| Test file | `tests/test_coding_agent_full_workflow_integration_live_llm.py` |
| Entry point | L2d action selection -> accepted coding task -> background worker -> coding agent |
| Cognition LLM route | `http://localhost:1234/v1` |
| Cognition model | `gemma-4-31b-fable-5-agent-distill` |
| Coding-agent PM LLM route | `http://localhost:1234/v1` |
| Coding-agent PM model | `gemma-4-31b-fable-5-agent-distill` |
| Test status convention | All five gates are known-gap `xfail(strict=True)` tests. |

## Evaluation Goal

Run gates 06 through 10 one at a time and record the observed failure modes
from real LLM traces. These gates are assessment-derived gaps and are expected
to fail until the corresponding development plans are implemented.

## Commands

```powershell
venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_06_mixed_create_and_existing_edit_workflow -q -s --tb=short -rxX
venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_07_proposal_has_preapproval_preflight_evidence -q -s --tb=short -rxX
venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_08_vague_approval_runs_changed_file_tests -q -s --tb=short -rxX
venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_09_missing_dependency_becomes_typed_blocker -q -s --tb=short -rxX
venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_10_source_free_proposal_records_alignment_gate -q -s --tb=short -rxX
```

## Results Summary

| Gate | Pytest result | Recorded at UTC | Trace |
| --- | --- | --- | --- |
| 06 | `XFAIL` | `2026-07-09T06:58:10.508113+00:00` | `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_06_mixed_create_existing_edit.json` |
| 07 | `XFAIL` | `2026-07-09T06:59:14.419402+00:00` | `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_07_preapproval_preflight.json` |
| 08 | `XFAIL` | `2026-07-09T07:03:53.927781+00:00` | `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_08_changed_file_execution_derivation.json` |
| 09 | `XFAIL` | `2026-07-09T07:10:05.783791+00:00` | `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_09_dependency_blocker.json` |
| 10 | `XFAIL` | `2026-07-09T07:21:51.002166+00:00` | `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_10_source_free_alignment.json` |

## Failure Modes

| Gate | Intended gap | Observed failure mode | Architectural meaning |
| --- | --- | --- | --- |
| 06 | Mixed create plus existing-source edit | The request was routed as source-free writing. Writing rejected it with `the request requires changing existing source files (counter_cli/cli.py), which is not permitted`. No patch artifacts or changed files were produced. | The mixed-change gap is broader than only missing `create_file` in `code_modifying`: routing can choose the source-free path when a request mentions adding a new module plus wiring existing code. |
| 07 | Preapproval preflight evidence | The supervisor rejected the task as unsupported before proposal generation. Worker metadata used v1 shape with `coding_operation=unsupported`; route reason said preflighting requires execution and patch application, which are prohibited by current operation limits. | The current capability model treats preflight as forbidden execution, so Phase B must update routing and capability wording as well as adding preflight mechanics. |
| 08 | Deterministic execution specs from changed files | Start produced a valid proposal changing `slug_tools/slug.py`. Approval completed successfully, but verification ran only `python_compileall` on `.`. It did not run `pytest` or `tests/test_slug.py`. | The worker/adaptor default path remains too weak for vague approval. Verification scope is not derived from changed source and test evidence. |
| 09 | Missing dependency becomes typed environment blocker | Approval ran explicit `pytest tests/test_yaml_dependency.py` and failed. The system did not return a typed `environment_dependency_missing` blocker and did not set run status to `blocked`. It entered repair handling and failed with `Synthesis included ungrounded code terms`, plus `Modifying PM blocked the request`. | Execution failure classification is missing. A non-repairable environment/dependency failure is routed into source repair, wasting the repair path and returning an opaque failure. |
| 10 | Source-free artifact alignment gate | Source-free generation produced four patch artifacts and an awaiting-approval proposal, but trace summary had `writing_acceptance:status=pass` and no `writing_alignment:status=pass` row. No apply, execution, or repair attempts existed. | The implemented alignment judge remains unwired. Source-free proposals can be delivered with acceptance criteria extracted but no semantic alignment verdict. |

## Per-Gate Evidence Notes

### Gate 06

- Turn 1 worker operation: `start`.
- Run status: `failed`.
- Limitations: `the request requires changing existing source files (counter_cli/cli.py), which is not permitted`.
- Trace rows include `writing_pm:mode=create_new_project` and
  `writing_acceptance:status=fail criteria=0`.
- Turn 2 summary kept the same failed state and no changed files.

### Gate 07

- L2d produced `accepted_coding_task_request` with `decision=start`.
- Background job status: `failed`.
- Worker metadata: `schema_version=coding_agent_worker_metadata.v1`,
  `coding_operation=unsupported`.
- Route reason explicitly classified managed-copy preflight and tests as
  prohibited operation limits.

### Gate 08

- Start turn proposed a change to `slug_tools/slug.py` and reached
  `awaiting_approval`.
- Approval turn applied the patch successfully in a managed copy.
- Execution attempts: one `python_compileall` run over `.`.
- Missing expected behavior: no `pytest` execution and no
  `tests/test_slug.py` executed path.

### Gate 09

- Start turn proposed a change to `dep_tool/loader.py`.
- Approval turn ran `pytest` against `tests/test_yaml_dependency.py`.
- Execution status: `failed`.
- Run status after approval: `failed`, not `blocked`.
- Blockers: empty.
- Repair attempts: empty in public metadata because the repair proposal was
  rejected before a new apply/execute attempt.
- Limitations: `Synthesis included ungrounded code terms`,
  `Structured repair feedback triggered deterministic bounded source evidence
  fallback for the repair proposal`, and `Modifying PM blocked the request`.

### Gate 10

- Source-free start turn reached `awaiting_approval`.
- Changed files: `src/toml_linter_core.py`, `src/toml_linter_cli.py`,
  `tests/test_toml_linter_core.py`, and `tests/test_toml_linter_cli.py`.
- Trace rows include `writing_acceptance:status=pass criteria=5`.
- Missing expected behavior: no `writing_alignment:status=pass` trace row and
  no projected alignment metadata.
- Limitation: `Generated artifacts were not executed in the target project.`

## Quality Assessment

- Gates 06, 08, 09, and 10 directly confirmed the assessment gaps they were
  designed to expose.
- Gate 07 exposed an earlier failure than the test assertion: the system cannot
  even accept preflight as a supported coding-run requirement today. The Phase B
  plan should explicitly update operation routing and capability descriptions,
  not only add preflight execution after proposal generation.
- Gate 09 is the most operationally risky failure. It converts an environment
  condition into a repair-generation failure, which makes the final failure
  harder for the user to act on.

## Recommended Plan Adjustments

- Phase A should include routing tests for mixed create/edit requests so the
  task reaches existing-source modification instead of source-free writing.
- Phase B should include a routing/capability test proving preflight requests
  are accepted when `CODING_AGENT_PREFLIGHT_EXECUTION` is enabled.
- Phase B should prioritize environment failure classification before expanding
  repair budgets; otherwise larger budgets will spend more time on
  non-repairable missing dependencies.
- Phase B should make changed-file execution derivation the default approval
  path before retaining any worker prose-based fallback.
- Phase A should wire alignment projection into worker metadata and trace
  summary before trying to judge alignment quality.

## Raw Evidence

- `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_06_mixed_create_existing_edit.json`
- `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_07_preapproval_preflight.json`
- `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_08_changed_file_execution_derivation.json`
- `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_09_dependency_blocker.json`
- `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_10_source_free_alignment.json`
