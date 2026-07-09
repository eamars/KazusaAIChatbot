# coding_agent_assessment_gap_phase_c_plan

## Summary

- Goal: Close the outer coding-run loop by making blockers, allowed actions,
  approval evidence, source serialization, and benchmarking first-class seams.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `py-style`,
  `test-style-and-execution`, `local-llm-architecture`
- Overall cutover strategy: Extend durable run contracts and action affordances
  in one canonical update, then add the benchmark seam for future model and
  architecture comparisons.
- Highest-risk areas: L2d affordance drift, ambiguous blocker semantics,
  approval evidence leakage, and concurrency races.
- Acceptance criteria: Typed blockers can be answered on the same
  `coding_run_ref`; L2d sees run-specific allowed actions; approval events carry
  auditable evidence; same-source runs serialize; benchmark harness records
  repeatable results.

## Context

After Phase A and B, the coding agent has stronger proposal and execution
loops, but the outer cognition seam still treats many coding results as
narration input. The assessment correctly flags that `allowed_next_actions` is
informational, blockers are not a typed user-resolution channel,
`approve_and_verify` approval is fabricated from L2d's semantic decision
without durable quote/message evidence, and concurrent jobs against the same
source can race longer-running verification loops.

This phase makes the accepted-task/coding-run interface self-contained for
follow-up behavior without exposing coding internals to the persona layer.

This plan depends on Phase B's typed blocker contract. Phase C must not
reclassify execution failures or invent blockers; it must make already-typed
blockers answerable through the same accepted-task/coding-run loop.

## Mandatory Skills

- `development-plan`: load before changing this plan or executing it.
- `py-style`: load before editing Python production code.
- `test-style-and-execution`: load before changing or running tests.
- `local-llm-architecture`: load before prompt, L2d, action-spec, worker, or
  coding-run contract changes.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan
  before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the Independent Code Review gate and record the result in Execution Evidence.
- Use parent-led native subagent execution unless the user explicitly approves
  fallback execution.
- Keep L2d inputs semantic and prompt-safe. Worker-local fields, execution
  specs, paths, shell commands, and approval internals must not become model
  selectable action params.
- Deterministic code validates run refs, allowed actions, approval evidence,
  source locks, and benchmark result schemas.
- Anti-cheat rule: tests must prove the integrated L2d-to-background-worker path
  honors run-specific affordances. Do not bypass L2d by directly calling
  `continue_coding_run` for the integration assertions.
- Anti-cheat rule: blocker-response tests must start from a real or
  deterministic typed blocker in a durable coding run. Do not fabricate a
  successful continuation by directly editing the ledger or bypassing worker
  payload validation.

## Must Do

- Add canonical typed blocker fields to `CodingRunBlocker` and public
  projections.
- Add `respond_to_blocker` as a coding-run continuation action and an
  `accepted_coding_task_request` decision.
- Route `respond_to_blocker` to the correct blocked run and feed the user's
  answer into the same proposal or verification lifecycle.
- Preserve Phase B environment blockers as blockers until the user supplies an
  actionable answer through `respond_to_blocker`; the response path must not
  rerun repair or execution before the blocker answer has been validated.
- Make run-specific `allowed_next_actions` available to L2d as the
  authoritative action affordance for the next user turn.
- Add accepted-task integration tests proving L2d can choose
  `respond_to_blocker` only when the run projection exposes that allowed action,
  and cannot approve, revise, or cancel when those actions are absent.
- Carry approval evidence from the triggering user message into the worker
  payload, approval object, ledger event, and public-safe projection:
  bounded quote, source message id, source trigger, requester id, and timestamp.
- Add per-run and per-source-identity locking for coding-run worker execution.
- Add a benchmark harness that drives tasks through
  `start_coding_run`/`continue_coding_run` or the accepted coding task entry
  point, records model/config, traces, pass/fail, elapsed time, and LLM-call
  counts.
- Add deterministic and live integration tests for blocker response,
  approval-evidence recording, allowed-action affordance binding, and locking.

## Deferred

- Do not add a live interactive CLI, web console surface, arbitrary shell,
  dependency installation, git branch/PR publishing, or JSON-action loop.
- Do not expose worker-local execution specs or raw ledger internals to L2d.
- Do not make benchmark results a production runtime dependency.
- Do not broaden Phase B's environment classifier or add installation behavior
  in this phase.

## Cutover Policy

Update coding-run models, action-spec contracts, L2d normalization, materialized
actions, worker payloads, and tests in one canonical contract move. Existing
`revise_proposal` remains for ordinary revision; `respond_to_blocker` is the
only canonical action for answering a typed blocker.

## Target State

A coding run can ask a typed question or report an environment/scope blocker,
and the next user answer can continue the same durable run through
`respond_to_blocker`. L2d receives the run's allowed actions and cannot approve,
cancel, or revise outside the run's current state. Approval records show the
message and quote that caused the approval. Concurrent jobs for the same run or
source identity serialize. Benchmark runs are repeatable and archived for model
comparison.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Blocker action | Add `respond_to_blocker` | It separates missing-information answers from proposal revision. |
| Blocker source | Consume typed blockers from coding-run state | Classification belongs to Phase B verification, not L2d. |
| L2d affordance source | Use run metadata, not generic capability prose | The run state owns what actions are legal. |
| Approval evidence | Store bounded quote plus source ids | Keeps semantic approval auditable without adding UI approval. |
| Lock owner | Coding-run ledger/workspace | The ledger already owns run state and source identity. |
| Benchmark seam | Drive public coding-run or accepted-task entrypoints | Measures the same behavior users exercise. |

## Contracts And Data Shapes

Typed blocker:

```python
{
    "code": str,
    "blocker_kind": "needs_user_input | environment | scope | safety",
    "message": str,
    "question": str,
    "options": list[str],
    "details": dict[str, object],
    "created_at": str,
}
```

`respond_to_blocker` action payload:

```python
{
    "operation": "respond_to_blocker",
    "coding_run_ref": "coding_run:<run_id>",
    "task_brief": "user's answer",
}
```

Approval evidence:

```python
{
    "approved": True,
    "approved_by": str,
    "approved_at": str,
    "approval_reason": str,
    "approval_evidence": {
        "source_message_id": str,
        "source_trigger_source": "user_message",
        "requester_global_user_id": str,
        "quote": str,
    },
}
```

Benchmark result:

```python
{
    "case_id": str,
    "model": str,
    "status": "passed | failed | blocked",
    "entrypoint": "accepted_task | coding_run",
    "elapsed_ms": int,
    "llm_call_count": int,
    "final_run_status": str,
    "trace_path": str,
    "notes": list[str],
}
```

## LLM Call And Context Budget

This phase changes L2d prompt inputs and accepted-task result projections. It
must not add a new response-path LLM call. L2d receives bounded run affordance
metadata and typed blocker text only; raw worker metadata, diffs, execution
output, private paths, and ledger internals remain excluded.

The benchmark harness may run live LLM calls, but only under explicit test or
benchmark invocation. It must record model and config metadata for comparison.

## Change Surface

### Delete

- None.

### Modify

- `src/kazusa_ai_chatbot/coding_agent/coding_run/models.py`,
  `coding_run/supervisor.py`, and `coding_run/ledger.py`: typed blockers,
  `respond_to_blocker`, approval evidence, and locks.
- `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`: payload and
  metadata projection for blocker responses and approval evidence.
- `src/kazusa_ai_chatbot/action_spec/registry.py`,
  `action_spec/handlers/background_work.py`,
  `cognition_chain_core/action_selection.py`, and `stages/l2d.py`: action
  contract and L2d normalization.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`:
  materialize the new decision and preserve approval evidence fields.
- Relevant README files for coding-run and action-spec contracts.
- Tests covering action-spec, background worker, L2d, coding-run, and full
  workflow integration.

### Create

- Benchmark harness under `tests/` or `scripts/` with archived output under a
  test-results or development-plan evidence path.
- Focused fixtures for blocker response and source-lock contention if existing
  fixtures do not cover them.

### Keep

- Existing accepted-task lifecycle and background-worker queue ownership.

## Overdesign Guardrail

- Actual problem: The outer loop cannot reliably answer typed blockers, bind
  legal next actions to L2d, audit approval, or serialize same-source work.
- Minimal change: Extend existing durable run/action-spec contracts and add a
  benchmark seam; keep the async accepted-task model.
- Ownership boundaries: Coding run owns state, blockers, locks, and public
  projection; action-spec owns deterministic validation; L2d owns semantic
  action selection from prompt-safe affordances; worker owns queue handoff.
- Rejected complexity: interactive CLI, web UI, generic action loop, raw worker
  metadata in prompts, production benchmark scheduler, and new approval UI.
- Evidence threshold: Add rejected surfaces only after benchmark evidence shows
  the current async seam is the limiting factor.

## Agent Autonomy Boundaries

- The responsible agent may add helper functions only within the listed
  ownership modules.
- The responsible agent must not expose raw execution output, raw diffs,
  private roots, or worker-local execution specs to L2d.
- Locking must fail closed with a retryable public status rather than allowing
  concurrent writes to the same run or source identity.
- If approval evidence cannot be sourced from a user message trigger, the
  action must be rejected.

## Implementation Order

1. Add deterministic tests for typed blocker projection and
   `respond_to_blocker` action validation.
2. Add accepted-task integration tests for run-specific allowed-action
   affordance binding:
   - A blocked run exposes `respond_to_blocker`.
   - L2d-to-action materialization rejects unavailable approve/revise/cancel
     choices for that run state.
   - Expected baseline before implementation: allowed actions are only
     informational and are not enforced as the action contract.
3. Add approval-evidence tests through action materialization and worker
   payloads.
4. Add lock tests for same run and same source identity.
5. Implement typed blockers and `respond_to_blocker` in coding-run.
6. Wire action-spec, L2d normalization, materialization, and worker payloads.
7. Add approval evidence recording and public-safe projection.
8. Add per-run/source locking.
9. Add benchmark harness and one smoke benchmark case.
10. Run live integration checks and independent code review.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes focused tests and baselines before production
  implementation starts.
- Production-code subagent: exactly one native subagent after focused tests are
  established; owns production code changes only.
- Independent code-review subagent: exactly one native subagent after planned
  verification passes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Blocker/action-spec tests added and baseline recorded.
- [ ] Accepted-task allowed-action integration tests added and baseline
  recorded.
- [ ] Approval-evidence tests added.
- [ ] L2d affordance-binding tests added.
- [ ] Locking tests added.
- [ ] Typed blockers and `respond_to_blocker` implemented.
- [ ] Approval evidence recorded and projected.
- [ ] Per-run/source locking implemented.
- [ ] Benchmark harness added with smoke case.
- [ ] Independent code review completed and findings addressed.
- [ ] Execution Evidence updated.

## Verification

Run deterministic suites:

```powershell
venv\Scripts\python -m pytest tests/test_coding_agent_background_run_contracts.py -q
venv\Scripts\python -m pytest tests/test_action_spec_evaluator.py -q
venv\Scripts\python -m pytest tests/test_action_spec_results.py -q
venv\Scripts\python -m pytest tests/test_coding_agent_phase9_run_supervisor_contracts.py -q
```

Run full-workflow live gates relevant to blockers, approval, and follow-ups one
at a time after deterministic tests pass.

## Independent Code Review

Run this gate after all Verification commands pass and before final sign-off.
The reviewer must inspect action-spec safety, L2d prompt-safe boundaries,
approval evidence provenance, lock correctness, benchmark reproducibility, and
metadata redaction. The parent agent may fix findings only inside this plan's
Change Surface, then rerun affected verification commands and record the
result.

## Acceptance Criteria

This plan is complete when:

- Typed blockers are durable, public-safe, and answerable through
  `respond_to_blocker`.
- L2d sees and follows run-specific allowed next actions.
- Approval events include bounded quote and source message evidence.
- Same-run and same-source coding jobs serialize.
- Benchmark smoke output is archived and reproducible.
- Deterministic regression tests, relevant live integration tests, and
  independent code review pass.

## Execution Evidence

- Pending execution.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| L2d receives private worker internals | Prompt-safe projection only | Action prompt/result tests |
| Approval evidence leaks sensitive text | Bound quote and sanitize projections | Approval evidence tests |
| Locks deadlock background work | Time-bounded lock acquisition and public retry status | Lock contention tests |
| Benchmark becomes flaky | Pin fixtures and archive traces | Benchmark smoke test |
