# coding agent final integration Ornith comparison

## Run Configuration

- Baseline archive:
  `development_plans/archive/completed/short_term/coding_agent_final_integration_gemma4_baseline_results.md`.
- Ornith support commit:
  `8ef8154 Add Ornith Qwen thinking support`.
- Target model: `ornith-1.0-35b-nvfp4-mtp`.
- Routes overridden for this run:
  `COGNITION_LLM`, `CODING_AGENT_PM_LLM`, and
  `CODING_AGENT_PROGRAMMER_LLM`.
- Thinking settings:
  `COGNITION_LLM_THINKING_ENABLED=false`,
  `CODING_AGENT_PM_LLM_THINKING_ENABLED=true`,
  and `CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED=false`.
- Verified route descriptors before running gates:
  `COGNITION_LLM` -> `model_family=qwen`, `thinking_strategy=qwen3_disabled`;
  `CODING_AGENT_PM_LLM` -> `model_family=qwen`,
  `thinking_strategy=qwen3_enabled`;
  `CODING_AGENT_PROGRAMMER_LLM` -> `model_family=qwen`,
  `thinking_strategy=qwen3_disabled`.
- Endpoint model availability: `/models` returned HTTP 200 and listed
  `ornith-1.0-35b-nvfp4-mtp` for all three routes.

Each gate is run one at a time with real LLM calls through the
L2d/action-spec/background-worker entrypoint. After each gate, raw traces are
inspected and this file records the evaluation before the next gate runs.

## Gate 01 - Read-Only Question

- Result: passed.
- Runtime: 66.90 seconds.
- Trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_01_read_only_question__20260709T034930065308Z.json`.
- Workflow evidence: one evidence item, one patch proposal record, one managed
  apply record, one execution record, and zero repair attempts.
- Artifact review: the response explained command discovery through the fixture
  registry and argparse wiring. No visible `<think>`/`</think>` style text was
  present, and the inspected artifact did not expose forbidden private runtime
  fields such as workspace roots, source roots, stdout/stderr excerpts, `.env`,
  or `.git` paths.
- Architectural review: the L2d-to-worker entrypoint stayed intact for a
  read-only task, public artifact sanitation held, and the Qwen thinking path
  did not leak reasoning into the final artifact.
- Gemma4 comparison: both models passed. Ornith was slower than the Gemma4
  baseline for this gate, 66.90 seconds versus 55.92 seconds, with equivalent
  capability coverage.

## Gate 02 - Source-Free Proposal With Revision Follow-Ups

- Result: failed after deterministic hardening and rerun.
- Initial observed deterministic failures:
  - source-free package review rejected `import pytest` as an unresolved local
    import even though pytest is available in the project test environment;
  - patcher materialization diagnostics such as `Compiled patch operations
    exceed the diff limit` stayed terminal because rejected review-package
    failures were not fed into the bounded validation-feedback pass;
  - the writing PM could mark a package complete without generating any
    artifacts, and the supervisor treated that invalid completion as terminal.
- Fixes applied during this gate:
  - patch validation now treats importable top-level Python modules as known
    imports while preserving generated local-module validation;
  - rejected review materialization errors now enter the existing bounded PM
    feedback pass;
  - empty PM completion now receives one explicit PM feedback retry before
    terminal failure.
- Deterministic verification:
  `tests/test_coding_agent_phase2_new_artifact_contracts.py` and
  `tests/test_coding_agent_phase4_code_patching_contracts.py` passed after the
  fixes, including new regressions for pytest imports, rejected patcher
  feedback, and empty PM completion.
- Latest live start-turn trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_02_source_free_revision_turn_1__20260709T042553745897Z.json`.
- Latest start-turn result: completed with
  `coding_run:020f9e856a6d4cfb867c76feb4045518`, run status
  `awaiting_approval`, four changed files, and four review artifacts.
- Remaining failure: a later Gate 02 follow-up failed before a turn trace was
  written because live L2d produced no materialized
  `accepted_coding_task_request`; pytest reported
  `Live L2d did not produce accepted_coding_task_request`.
- Architectural review: the deterministic writer/review-package recovery gaps
  were real Kazusa issues and were fixed. The remaining failure is a model
  routing/continuation weakness under Ornith on a follow-up turn, not the same
  deterministic source-free proposal failure.
- Gemma4 comparison: Gemma4 also failed Gate 02, but for a different
  source-free revision quality issue: it generated an invalid duplicate
  artifact path during revise-proposal. Ornith reached a valid awaiting-approval
  start proposal after hardening, then failed on follow-up action selection.

## Gate 03 - Existing Source Proposal With Runtime-Only Follow-Ups

- Result: passed.
- Runtime: 231.03 seconds.
- Trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_03_existing_source_runtime_only__20260709T043144660214Z.json`.
- Durable run: all three turns used
  `coding_run:c241c9bd4b5649b1890d9d6e3d47cd10`.
- Workflow operations:
  - turn 1 completed `start` and left the run `awaiting_approval` with three
    changed-file records;
  - turn 2 completed `revise_proposal` and kept the same awaiting-approval run;
  - turn 3 completed `summarize` and kept the same awaiting-approval run.
- Artifact review: no inspected public artifact exposed private workspace,
  source-root, stdout/stderr excerpt, `.env`, `.git`, or Qwen thinking tokens.
- Architectural review: existing-source proposal, runtime-only follow-up
  handling, durable run continuity, and public artifact sanitation all held for
  Ornith.
- Gemma4 comparison: both models passed. Ornith was faster on this gate,
  231.03 seconds versus the Gemma4 baseline of 358.06 seconds.

## Gate 04 - Approval, Verify, And Repair Follow-Ups

- Result: passed.
- Runtime: 219.26 seconds.
- Trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_04_approval_verify_repair__20260709T043554912133Z.json`.
- Durable run: all three turns used
  `coding_run:491a27351e4d4913b4a230c7cafdcc47`.
- Workflow operations:
  - turn 1 completed `start` and left the run `awaiting_approval`;
  - turn 2 completed `approve_and_verify` and moved the run to `completed`;
  - turn 3 completed `summarize` against the completed run.
- Verification evidence: the approval path accepted the proposal, matched
  source identity, created a managed copy, applied the patch with git, and ran
  pytest successfully. No repair attempt was required because verification
  passed on the approved patch.
- Artifact review: no inspected public artifact exposed private workspace,
  source-root, stdout/stderr excerpt, `.env`, `.git`, or Qwen thinking tokens.
- Architectural review: managed apply, source identity validation, protected
  execution, durable completion state, and post-completion summarize behavior
  held for Ornith.
- Gemma4 comparison: both models passed. Ornith was slower on this gate,
  219.26 seconds versus the Gemma4 baseline of 164.01 seconds.

## Gate 05 - Hard Multi-File Approval, History, Cancel, Status

- Result: failed.
- Runtime: 103.56 seconds.
- Latest traces:
  - `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_1__20260709T043639946058Z.json`;
  - `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_2__20260709T043710821343Z.json`.
- Turn 1 result: the worker completed the start action but left
  `coding_run:01ebc0249b6148dfb0b0a84f6abd3c3f` in `blocked` status with zero
  changed files. The artifact asked the user to narrow or clarify because the
  agent was "read-only" and claimed a write-capable tool was needed.
- Turn 2 result: approval was rejected because the run was already terminal.
  This is correct state protection after the turn 1 block.
- Later failure: the next follow-up did not produce a materialized
  `accepted_coding_task_request`, so pytest stopped before writing a complete
  five-turn gate trace.
- Architectural review: terminal-run protection behaved correctly, but Ornith
  failed the core planning requirement for an explicit multi-file fix request
  against a local checkout. The failure is a model routing/planning weakness:
  it selected a read-only posture instead of driving the code-modifying
  workflow.
- Gemma4 comparison: Gemma4 passed this gate in 337.38 seconds, with known
  qualitative weakness around README scope drift and occasional Chinese
  language drift. Ornith was faster to fail and did not reach approval,
  execution, history, cancel, or final status coverage.

## Overall Comparison

- Gemma4 baseline: 4/5 gates passed. Gate 02 failed on source-free
  revise-proposal quality with a duplicate generated artifact path.
- Ornith after Qwen-path support and deterministic hardening: 3/5 gates passed.
  Gates 01, 03, and 04 passed; Gates 02 and 05 failed.
- Deterministic Kazusa fixes discovered by Ornith:
  - importable third-party/test dependencies such as `pytest` must not be
    reported as missing local generated modules;
  - rejected review-package materialization errors need to enter the bounded
    source-free feedback loop;
  - source-free PM completion with zero generated artifacts needs one explicit
    corrective PM feedback pass.
- Ornith-specific residual weaknesses:
  - follow-up L2d continuation can miss `accepted_coding_task_request` even
    when the user provides a valid `coding_run` reference;
  - hard existing-source fix requests can be misplanned as read-only
    unsupported requests instead of code-modifying work.
- Capability conclusion: Ornith can execute the simpler and medium integrated
  coding workflows once deterministic recovery gaps are closed, but it is less
  reliable than Gemma4 on follow-up routing and hard multi-file modification
  planning.
