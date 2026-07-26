# coding agent final integration gate reviews

Execution started after commit
`568fa71 Complete coding agent full workflow hardening`.

Each gate is run one at a time with real LLM calls through the
L2d/action-spec/background-worker entrypoint. After each gate, the raw trace is
inspected and this file records the behavioral evaluation and architecture
review before the next gate is run.

## Gate 01 - read-only codebase question

- Command:
  `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_01_read_only_question_from_l2d_to_worker -q -s --tb=short`
- Result: passed in 55.92 seconds.
- Raw aggregate trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_01_read_only_question__20260709T024229160410Z.json`.
- Flow: L2d selected `accepted_coding_task_request` with decision `start`,
  paired with a visible acknowledgement. The worker created
  `coding_run:637c5dc60adc4bdc869e32c73b8b6d5f` and completed it.
- Contract evidence: operation `start`; status `completed`; 4 evidence refs;
  0 patch artifacts; 0 apply attempts; 0 execution attempts; 0 repair
  attempts.
- Behavioral evaluation: the answer correctly identified command discovery in
  `src/tooling/commands.py` through `COMMANDS` and `discover_commands()`, then
  tied parser setup and dispatch to `src/tooling/cli.py`. This satisfies the
  read-only source-grounding contract.
- Architectural review: this gate exercised the intended L2d/action-spec/
  background-worker entrypoint and preserved the read-only boundary. Public
  worker output did not expose local roots, raw command output, `.env`, or
  `.git` internals. No deterministic issue surfaced.

## Gate 02 - source-free proposal revision and summary

- Command:
  `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_02_source_free_proposal_with_revision_followups -q -s --tb=short`
- Result: failed in 996.03 seconds.
- Raw aggregate trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_02_source_free_revision__20260709T025945149048Z.json`.
- Raw failing turn trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_02_source_free_revision_turn_2__20260709T025935243923Z.json`.
- Flow: turn 1 selected `start` and created
  `coding_run:a35e0e14831e4b779b8bcab45265d964` in `awaiting_approval`.
  Turn 2 selected `revise_proposal` with the same run ref, but the coding
  worker returned status `failed`. Turn 3 selected `summarize` against the
  same run and reported the failed terminal state.
- Contract evidence: no apply, execution, or repair attempts occurred on any
  turn. The failure stayed inside review-only source-free proposal generation.
- Failure detail: revision generated artifacts for
  `src/impl_csv_normalizer_logic.py`, `src/impl_csv_normalizer_cli.py`, and
  `tests/test_csv_normalizer_logic.py`, but deterministic validation reported
  `Generated artifact path 'src/impl_csv_normalizer_cli.py' is duplicated`
  and `Generated artifact path 'src/impl_csv_normalizer_cli.py' is duplicated
  in the manifest`.
- Behavioral evaluation: L2d understood the follow-up and preserved the
  run-reference contract. The coding LLM failed to produce a coherent revised
  source-free artifact package.
- Architectural review: deterministic validation behaved correctly by
  rejecting the duplicate artifact package and keeping the proposal
  review-only. This is classified as a local-LLM generation weakness rather
  than a deterministic code defect. The known weak point is source-free
  artifact-manifest consistency during proposal revision; adding more retry
  depth would be an architectural tradeoff, not a required deterministic fix
  from this evidence.

## Gate 03 - existing-source runtime-only revision

- Command:
  `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_03_existing_source_proposal_with_runtime_only_followups -q -s --tb=short`
- Result: passed in 358.06 seconds.
- Raw aggregate trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_03_existing_source_runtime_only__20260709T030644099430Z.json`.
- Flow: turn 1 selected `start`, turn 2 selected `revise_proposal`, and turn
  3 selected `summarize`, all against
  `coding_run:9984f5b9de674e84a0c28ff63e718d67`.
- Contract evidence: all turns stayed `awaiting_approval`; no apply,
  execution, or repair attempts occurred; changed files were limited to
  `counter_cli/cli.py` across all turns.
- Behavioral evaluation: the model satisfied the existing-source proposal
  workflow and the follow-up instruction to keep tests unchanged. The summary
  projected the changed file and allowed next actions.
- Architectural review: the gate exercised durable same-run revision and
  summary from the real background entrypoint while preserving review-only
  behavior. One L2d warning appeared because `resolver_goal_progress` omitted
  or malformed `missing_user_inputs`; action selection remained valid. This is
  a non-blocking model-schema weakness rather than a deterministic coding-run
  issue.

## Gate 04 - approval, verify, and protected tests

- Command:
  `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_04_approval_verify_and_repair_followups -q -s --tb=short`
- Result: passed in 164.01 seconds.
- Raw aggregate trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_04_approval_verify_repair__20260709T031005913001Z.json`.
- Flow: turn 1 selected `start`, turn 2 selected `approve_and_verify`, and
  turn 3 selected `summarize`, all against
  `coding_run:371162efae7649cb80c53dfafa91c8e8`.
- Contract evidence: approval used 1 managed apply attempt and 1 focused
  pytest execution attempt; pytest succeeded; repair was not needed. Final
  changed files contained `slug_tools/slug.py` only.
- Behavioral evaluation: the initial proposal included `tests/test_slug.py`,
  but approval-time filtering omitted that protected verification path before
  managed apply. The runtime source fix passed focused pytest and summary
  preserved attempt history.
- Architectural review: this gate confirms the deterministic approval boundary
  works even when the coding LLM proposes protected test edits. The apply and
  execution path stayed managed and allowlisted. The L2d goal-progress warning
  recurred but did not affect action selection or worker safety. No
  deterministic issue surfaced.

## Gate 05 - hard multi-file approval history, cancel, and status

- Command:
  `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_05_hard_multifile_approval_history_cancel_status -q -s --tb=short`
- Result: passed in 337.38 seconds.
- Raw aggregate trace:
  `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status__20260709T031614381447Z.json`.
- Flow: turns selected `start`, `approve_and_verify`, `summarize`, `cancel`,
  and `status`, all against
  `coding_run:cb7cb5a0d7a64f7d954a2d76087dc69a`.
- Contract evidence: approval used 1 managed apply attempt and 1 focused
  pytest execution attempt; pytest succeeded; no repair was needed. The cancel
  request was rejected because the run was already terminal, and the final
  status request returned `completed`.
- Behavioral evaluation: the workflow preserved durable run identity and
  attempt history across five turns. Final changed files were
  `release_feed/cache.py`, `release_feed/cli.py`, and `README.md`.
- Architectural review: the durable run lifecycle, terminal cancellation
  handling, allowed execution, and status projection behaved correctly. The
  current hard gate allows README edits, but the user follow-up asked for
  "final changed source files"; including `README.md` is a model-scope
  weakness worth tracking if future gates require stricter runtime-source-only
  summaries. L2d also drifted some acknowledgement details into Chinese while
  preserving valid structured actions. No deterministic issue surfaced.

## Final Run Summary

| Gate | Result | Architectural judgment |
|---|---|---|
| 01 | Pass | Read-only L2d/background-worker path is sound. |
| 02 | Fail | Deterministic validation correctly rejected a duplicated source-free artifact path; classify as local-LLM artifact-generation weakness. |
| 03 | Pass | Same-run existing-source revision and summary are sound. |
| 04 | Pass | Approval, managed apply, focused pytest, and protected verification filtering are sound. |
| 05 | Pass | Multi-turn durable run lifecycle is sound; note README scope drift and occasional L2d language drift. |

Overall result: 4 of 5 final real LLM gates passed in this run. The failed
gate did not expose a deterministic code defect; it exposed remaining local
LLM brittleness in source-free revised artifact generation.
