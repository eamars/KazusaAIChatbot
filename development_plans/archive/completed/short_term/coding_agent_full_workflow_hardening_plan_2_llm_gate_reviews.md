# coding agent full workflow hardening plan 2 LLM gate reviews

## Gate 01 - read-only source question

- Run command: `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_01_read_only_question_from_l2d_to_worker -q -s --tb=short`
- Pytest result: passed on 2026-07-09.
- Raw aggregate trace: `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_01_read_only_question.json`.
- Raw turn trace: `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_01_read_only_question_turn_1.json`.
- Review: L2d selected `accepted_coding_task_request` with decision `start`
  and paired it with a visible acknowledgement. The background worker routed
  the job to `coding_agent`, created one durable run
  `coding_run:746148610298468ea432960e4e3f6990`, and completed it.
- Contract checks: worker operation was `start`; run status was `completed`;
  evidence refs were populated; patch, apply, execution, and repair attempt
  lists were empty; public artifact text explained command discovery from the
  fixture source. The gate's private-leak assertion passed.
- Judgment: pass. This gate proves the read-only full workflow entrypoint
  from L2d to queue to durable coding worker without hidden direct API use.

## Gate 02 - source-free proposal revision and summary

- Run command: `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_02_source_free_proposal_with_revision_followups -q -s --tb=short`
- Pytest result: passed on 2026-07-09 after the Gate 02 code-writing
  validation-feedback remediation.
- Raw aggregate trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_02_source_free_revision__20260709T013928050744Z.json`.
- Raw turn traces:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_02_source_free_revision_turn_1__20260709T012941003669Z.json`,
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_02_source_free_revision_turn_2__20260709T013914783834Z.json`,
  and
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_02_source_free_revision_turn_3__20260709T013928049704Z.json`.
- Review: L2d selected `start`, then `revise_proposal`, then `summarize` for
  one durable coding run. The same run reference,
  `coding_run:876829b6929342e9b0c72fbf6247d961`, was preserved across all
  three turns.
- Contract checks: turn 1 reached `awaiting_approval` with source-free
  proposal artifacts; turn 2 revised the proposal and stayed
  `awaiting_approval`; turn 3 summarized changed files. Apply, execution, and
  repair attempt lists stayed empty on every turn, so the source-free work
  remained review-only before approval. The gate's private-leak assertion
  passed.
- Judgment: pass. This gate proves source-free start, same-run proposal
  revision, and deterministic summary through the L2d/background-worker
  entrypoint, with no hidden direct API shortcut.

## Gate 03 - existing-source runtime-only revision

- Run command: `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_03_existing_source_proposal_with_runtime_only_followups -q -s --tb=short`
- Pytest result: passed on 2026-07-09.
- Raw aggregate trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_03_existing_source_runtime_only.json`.
- Raw turn traces:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_03_existing_source_runtime_only_turn_1.json`,
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_03_existing_source_runtime_only_turn_2.json`,
  and
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_03_existing_source_runtime_only_turn_3.json`.
- Review: L2d selected start, revision, and summary for one durable run,
  `coding_run:2a78d6e89569485e8a41b71caa472bb5`. The initial proposal touched
  runtime, tests, and README. The follow-up revision preserved the same run id
  and narrowed the changed files to `counter_cli/cli.py`.
- Contract checks: all three worker turns succeeded and stayed
  `awaiting_approval`; apply, execution, and repair attempt lists stayed empty.
  The runtime-only follow-up respected the protected-test constraint by not
  changing `tests/` in the revised proposal. The gate's private-leak assertion
  passed.
- Judgment: pass. This gate proves existing-source proposal revision can
  narrow scope within one durable run without applying or executing anything.

## Gate 04 - approval, verify, and protected tests

- Run command: `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_04_approval_verify_and_repair_followups -q -s --tb=short`
- Pytest result: passed on 2026-07-09 after approval-time protected-path
  filtering was added.
- Raw aggregate trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_04_approval_verify_repair__20260709T015913674897Z.json`.
- Raw turn traces:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_04_approval_verify_repair_turn_1__20260709T015844816846Z.json`,
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_04_approval_verify_repair_turn_2__20260709T015901719409Z.json`,
  and
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_04_approval_verify_repair_turn_3__20260709T015913673629Z.json`.
- Review: L2d selected start, approve-and-verify, and summarize for one
  durable run, `coding_run:04a5ee6a8b504117af79ee61aa4f412d`. The initial
  proposal included `tests/test_slug.py`, but the approval-time verifier
  omitted that protected pytest selector before managed apply.
- Contract checks: approval used one managed apply attempt and one focused
  pytest execution attempt; execution succeeded; no repair attempt was needed.
  Final changed files were `slug_tools/slug.py` and `README.md`, with no
  `tests/` path. The trace recorded
  `verify_repair:protected_initial_artifacts_omitted count=1`, and the gate's
  private-leak assertion passed.
- Judgment: pass. This gate proves structured approval, managed apply,
  focused execution, protected verification paths, and durable summary through
  the full background-worker entrypoint.

## Gate 05 - hard multi-file approval history, cancel, and status

- Run command: `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_05_hard_multifile_approval_history_cancel_status -q -s --tb=short`
- Pytest result: passed on 2026-07-09.
- Raw aggregate trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status.json`.
- Raw turn traces:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_1.json`,
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_2.json`,
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_3.json`,
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_4.json`,
  and
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_5.json`.
- Review: L2d selected start, approve-and-verify, summarize, cancel, and
  status for one durable run,
  `coding_run:6437ae63ef4b43cdb2cc5f9125af46db`. The approval turn completed
  the run with one managed apply attempt and one focused pytest execution
  attempt.
- Contract checks: start was review-only; approval applied only
  `release_feed/cache.py` and `release_feed/cli.py`; protected verification
  tests `tests/test_cache.py` and `tests/test_cli.py` were omitted before
  managed apply. Summary and final status preserved the same attempt history.
  The later cancel request finished as a terminal-state rejection, leaving the
  completed run intact. The gate's private-leak assertion passed.
- Judgment: pass. This gate proves the hard multi-turn workflow from the real
  L2d/background-worker entrypoint, including durable run identity,
  approval-time verification, attempt-history projection, protected tests,
  terminal cancellation behavior, and final status.
