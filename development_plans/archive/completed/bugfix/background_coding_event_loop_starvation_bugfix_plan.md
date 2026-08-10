# background coding event-loop starvation bugfix

## Summary

- Goal: Keep the brain service and Control Console responsive while durable background coding work performs blocking source, reading, patch, and verification operations.
- Status: completed
- Scope boundary: Existing async callers in the coding-agent source-fetching, reading, verification, and coding-run preflight paths, plus deterministic regression tests and this plan record.
- Change direction: Offload existing synchronous operations with `asyncio.to_thread` at their async ownership boundaries while preserving every public API, job contract, ledger shape, and result projection.
- Acceptance state: Accepted after bounded DeepSeek implementation, parent review, and deterministic verification.

## Scope And Change Direction

The confirmed failure is event-loop starvation: the background coding run executes synchronous LLM, filesystem, subprocess, and network work from a coroutine on the single brain event loop. The fix keeps the current coding-agent architecture and places bounded thread offload at the async-to-sync boundaries:

```text
brain event loop
  -> async coding-agent operation
  -> await asyncio.to_thread(existing synchronous stage)
  -> resume event loop with the unchanged result
```

The implementation covers:

- synchronous local checkout, managed clone, raw download, and inline source materialization called by `code_fetching.run(...)`;
- synchronous `code_reading.run(...)` calls used by direct reading, generated-artifact readback, and existing-repository proposal preparation;
- synchronous managed patch application and bounded code execution called by the async verification path;
- synchronous managed candidate and preflight execution called by the async durable coding-run supervisor.

Already-async LLM calls, database checkpoints, worker leasing, result delivery, prompt contracts, and deterministic public projections remain on their current ownership boundaries.

## Confirmed Decisions

- Use `asyncio.to_thread` at the existing async callers; this bugfix does not move background work into a new process.
- Keep direct synchronous APIs such as `code_reading.run(...)`, `apply_approved_patch(...)`, and `execute_code_check(...)` synchronous for existing direct callers. Only their async runtime callers receive thread offload.
- Preserve exception propagation, trace content, ledger updates, lease behavior, result schemas, and public cancellation/result contracts except for allowing the event loop to schedule while the synchronous operation is running in its worker thread. A synchronous operation already running in a worker thread remains non-interruptible by task cancellation; new termination semantics remain deferred.
- Treat the Control Console as the affected acceptance surface. No Control Console route, static asset, timeout, SSE, or browser contract changes are part of this fix.

## Mandatory Skills

- `local-llm-architecture`: preserve the bounded brain/worker ownership boundary and avoid adding a new agent, prompt, route, or process architecture for this targeted latency failure.
- `py-style`: applies to every modified Python production and test file.
- `test-style-and-execution`: applies to deterministic async-boundary tests and all test execution.
- `control-console-web-development`: applies to reviewing the console impact; no frontend or console API edit is authorized by this plan.
- `development-plan`: governs this plan, the DeepSeek handoff, evidence, parent review, and closeout.

## Mandatory Rules

- Use `venv\Scripts\python` for Python checks and tests.
- Keep `.env` unread and keep secrets, local roots, and raw model output out of committed artifacts.
- Keep the change surgical. Do not add compatibility shims, fallback paths, new configuration, worker discovery, new public fields, or unrelated cleanup.
- Do not replace async LLM calls with synchronous calls. Offload only operations that are synchronously implemented today.
- Do not claim responsiveness from a passing unit test alone; the regression must demonstrate that a controlled blocking sync operation does not prevent an independent event-loop heartbeat from running.
- Do not run live LLM or live database tests for this fix unless a separate user-approved verification request expands the plan.

## Must Do

1. Add thread offload at every listed async-to-sync boundary in the coding-agent runtime path.
2. Add deterministic regression coverage for event-loop progress while representative blocking reading, source-resolution, and verification operations are held in a controlled test double.
3. Run the new regression and the directly affected coding-agent, background-work, and coding-run contract suites.
4. Review the final diff for scope, public-contract preservation, exception behavior, thread-boundary correctness, and absence of direct blocking calls in the covered async paths.
5. Record implementation, verification, review findings, residual risk, and final workspace status in this plan before lifecycle closeout.

## Deferred

- Moving background coding into a separate process or service.
- Changing Control Console request timeouts, bootstrap concurrency, SSE behavior, browser rendering, or static assets.
- CPU/memory limits, executor sizing, model latency tuning, job-lease redesign, or new cancellation/termination semantics.
- Refactoring the legacy `code_action_loop` path that is outside the current accepted-task coding-run entrypoint.
- Live LLM, live MongoDB, production smoke, and browser-session verification.

## Target State

- A blocking synchronous coding stage runs in a worker thread and does not monopolize the brain event-loop thread.
- `answer_code_question(...)`, `propose_code_change(...)`, `verify_and_repair_code_change(...)`, and durable coding-run proposal/preflight callers retain their existing public result contracts.
- Background job checkpoints, coding-run ledgers, patch artifacts, execution results, and error handling remain unchanged apart from scheduling.
- Direct synchronous specialist APIs retain their current signatures and behavior for non-async callers.

## Change Surface

### Delete

- None.

### Modify

- `src/kazusa_ai_chatbot/coding_agent/supervisor.py`: offload synchronous reading calls used by direct reading and write-proposal readback/preparation.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/agent.py`: offload synchronous local resolution and managed source materialization calls from the async fetch entrypoint.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/supervisor.py`: offload synchronous managed apply and execution calls from async verification, including source-free verification.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/supervisor.py`: offload synchronous proposal preflight candidate materialization and execution from async durable-run transitions.
- `development_plans/README.md`: register this active bugfix plan.

### Create

- `tests/test_coding_agent_async_boundaries.py`: deterministic event-loop responsiveness regressions for the covered async-to-sync boundaries.
- `development_plans/active/bugfix/background_coding_event_loop_starvation_bugfix_plan.md`: this execution contract and evidence record.

### Keep

- `src/control_console/**`: unchanged; it remains the observable client surface.
- `src/kazusa_ai_chatbot/background_work/**`: unchanged worker leasing, checkpointing, delivery, and runtime scheduling contracts.
- Public coding-agent and coding-run schemas, direct synchronous specialist APIs, prompt/model routes, persistence, and adapter delivery.

## Agent Autonomy Boundaries

The implementation agent may choose the local placement of `asyncio.to_thread`, whether a private helper remains synchronous behind the offload, and the deterministic test-double mechanics, provided the target state and change surface remain intact. It may adjust focused test fixtures needed to preserve existing contracts.

The implementation agent must keep the listed public APIs synchronous or asynchronous as currently defined, must not modify the console or background job contracts, and must not introduce a process worker, executor configuration, compatibility layer, retry, timeout, or unrelated refactor. A required change outside this surface pauses execution for a plan amendment.

## Verification

- Run `venv\Scripts\python -m pytest tests\test_coding_agent_async_boundaries.py -q` first and inspect the event-loop heartbeat assertions.
- Run the affected deterministic suites, including `tests\test_coding_agent_reading.py`, `tests\test_coding_agent_reading_acceptance.py`, `tests\test_coding_agent_fetching.py`, `tests\test_coding_agent_phase5_patch_apply_contracts.py`, `tests\test_coding_agent_phase6_code_executing_contracts.py`, `tests\test_coding_agent_phase8_verify_repair_contracts.py`, `tests\test_coding_agent_phase9_run_supervisor_contracts.py`, `tests\test_coding_agent_phase_d_coding_run_integration.py`, and `tests\test_background_work_jobs.py`.
- Run `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\coding_agent` as a static syntax check.
- Run `git diff --check` and review `git diff --stat` plus the complete diff.
- Confirm `git status --short` contains only the plan, registry, implementation, and test paths listed here.
- Do not run live LLM or live database cases as part of this plan.

## Acceptance Criteria

1. The new deterministic tests demonstrate that an independent event-loop heartbeat runs while each representative synchronous coding stage is blocked in a test double.
2. The covered async coding-agent and coding-run paths contain no direct calls to the blocking synchronous stages listed in this plan; each is awaited through a thread boundary.
3. Existing focused deterministic tests pass without public result, ledger, job, or direct-specialist contract changes.
4. Python compilation and `git diff --check` pass.
5. No Control Console source or static asset changes are present.
6. Parent review records no unresolved scope, contract, exception, or thread-safety finding, or the finding is resolved within this plan boundary.
7. The plan records exact test results, review evidence, residual risk, and a clean final workspace status before it moves to `completed`.

## Progress Checklist

- [x] RCA confirmed and async-to-sync ownership boundary identified.
- [x] Plan registered under `active/bugfix/` with `in_progress` status.
- [x] DeepSeek implementation handoff completed.
- [x] Parent diff and contract review completed.
- [x] Focused and affected deterministic verification completed.
- [x] Execution evidence and residual risk recorded.
- [x] Plan lifecycle closed after acceptance.

## Execution Evidence

### 2026-08-05 DeepSeek implementation handoff

Branch: `cognition_core_v2`. Baseline matched the handoff: only
`development_plans/README.md` modified and this plan file untracked.

Implemented (all listed async-to-sync boundaries, `asyncio.to_thread` only):

- `src/kazusa_ai_chatbot/coding_agent/supervisor.py`: offloaded
  `code_reading.run` for direct reading in `answer_code_question`, generated-
  artifact readback in `_propose_new_project_change`, and initial read for
  existing-repository proposal preparation in `_propose_existing_repo_change`.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/agent.py`: offloaded
  synchronous local checkout resolution (`_resolve_local_path`,
  `_resolve_local_root`) and managed inline, clone, and raw-download
  materialization inside async `run`.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/supervisor.py`: offloaded
  managed candidate/apply (`apply_approved_patch`, source-free
  `_verify_source_free_candidate`) and bounded execution
  (`_run_execution_specs`) inside async verification.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/supervisor.py`: offloaded
  proposal preflight binding (`_bind_proposal_and_preflight`, which owns
  managed candidate materialization and bounded preflight execution) from the
  async proposal-start and revision transitions.
- `tests/test_coding_agent_async_boundaries.py`: created deterministic
  heartbeat regressions (10 cases) that hold each representative blocking
  stage in a controlled test double and require an independent event-loop
  heartbeat to run before the stage is released.

Private synchronous helpers remain synchronous behind the offload; direct
synchronous specialist APIs, public signatures, result schemas, job/lease/
checkpoint behavior, ledger shapes, trace/error behavior, prompts, routes, and
console code are unchanged.

Verification:

- `venv\Scripts\python -m pytest tests\test_coding_agent_async_boundaries.py -q`
  -> 10 passed.
- `venv\Scripts\python -m pytest tests\test_coding_agent_reading.py
  tests\test_coding_agent_reading_acceptance.py tests\test_coding_agent_fetching.py
  tests\test_coding_agent_phase5_patch_apply_contracts.py
  tests\test_coding_agent_phase6_code_executing_contracts.py
  tests\test_coding_agent_phase8_verify_repair_contracts.py
  tests\test_coding_agent_phase9_run_supervisor_contracts.py
  tests\test_coding_agent_phase_d_coding_run_integration.py
  tests\test_background_work_jobs.py -q` -> 153 passed.
- `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\coding_agent`
  -> passed (exit 0).
- `git diff --check` -> passed (line-ending warnings only).
- Final `git status --short` contains only the parent-owned registry/plan
  paths plus the four production files and the new test file.

Deviations: none. The plan-authorized choice to keep private helpers
synchronous behind the offload was used.

Residual risk for parent review:

- Deterministic bounded fallback reading (`_fallback_reading_result_for_write`,
  up to ten capped file reads) and the single-path scope existence stat in
  `code_fetching.agent._source_scope_validation_error` remain synchronous in
  async paths because they are not listed stages of this plan.
- Durable ledger/event file writes in `coding_run` and bounded repair-feedback
  file reads in `code_verifying` remain on their existing ownership boundaries
  as planned.
- No live LLM, live database, or browser-session verification was run; no
  Control Console source or static asset was modified.

## Independent Code Review

Completed by the parent agent on 2026-08-05.

Review findings:

- The complete diff stays within the approved surface: four coding-agent production files, one deterministic regression file, this plan, and the plan registry.
- Every named dominant synchronous boundary is awaited through `asyncio.to_thread`: direct/generated/existing-repository reading, local/inline/clone/raw source materialization, managed verification apply/execute, and durable coding-run preflight.
- Direct synchronous specialist APIs retain their signatures. No console, background-work, schema, ledger, lease, route, prompt, process-worker, retry, timeout, or executor changes were introduced.
- The heartbeat tests hold representative synchronous stages and prove independent event-loop progress before release. The implementation does not alter returned values, exception ownership, or persistence projections.
- The remaining synchronous operations are bounded fallback file reads, one source-scope existence stat, and existing ledger/event or repair-feedback file I/O. They are outside the dominant blocking stages in this plan and are recorded as residual risk rather than silently expanded into new scope.
- Cancellation review: cancellation can release the awaiting coroutine while an already-started worker-thread operation finishes in the background. This is the expected limitation of `asyncio.to_thread`; no new termination semantics or ledger behavior were introduced, and the risk is explicitly deferred by this plan.

Independent verification:

- `venv\Scripts\python -m pytest tests\test_coding_agent_async_boundaries.py -q` -> 10 passed in 6.08s.
- The nine affected deterministic suites listed in `Verification` -> 153 passed in 36.03s.
- `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\coding_agent` -> exit 0.
- `git diff --check` -> passed; Git emitted only existing LF-to-CRLF conversion warnings.
- Final status contains only the registered plan/registry, four owned production files, and the new regression test; no Control Console or background-work source changed.

Conclusion: the plan acceptance criteria are satisfied. The event-loop starvation fix is ready for normal repository review/commit; live LLM, database, browser, and production smoke verification remain deferred items documented above.
