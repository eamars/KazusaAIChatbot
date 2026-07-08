# coding agent phase9 run supervisor plan

## Summary

- Goal: implement the durable coding-run supervisor for the coding-agent
  architecture, adding public run start, continuation, and inspection APIs
  backed by workspace-local JSON ledgers.
- Plan class: high_risk_migration.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, and `debug-llm`.
- Overall cutover strategy: compatible public API addition. Existing direct
  primitives remain callable; `coding_run` composes them into durable run
  state.
- Highest-risk areas: state-machine drift, public ledger sanitization,
  approval boundary leakage, repeated apply/execute side effects, stale source
  identity, run resume after process boundary, and live-gate dependence on
  weak local LLM planning.
- Acceptance criteria: deterministic run-state and public E2E workflow tests
  pass, existing Phase 5 to Phase 8 regressions pass, five prepared real LLM
  coding-run gates pass one at a time with committed raw/review evidence, docs
  describe the run boundary, Phase 9 supports comprehensive post-phase E2E
  testing for all supported coding-agent workflows without depending on Phase
  10 or later plans, and independent code review accepts the implementation.

## Context

Phase 8 completed the direct trusted verify-and-repair primitive. The coding
agent can now resolve source, gather read-only evidence, propose source-free or
existing-source patches, apply approved patches into managed copies, execute
bounded Python verification, and repair failed verification inside hard caps.

The current implementation still exposes those capabilities as independent
direct calls. It does not yet persist a coding task as one inspectable local
session with a run id, lifecycle state, proposal history, approval status,
apply attempts, execution attempts, repair attempts, cancellation, blockers,
and a public-safe event stream. That missing durable run layer is the Phase 9
scope.

Phase 9 is the first point where Kazusa forms a self-contained coding-agent
loop for supported coding work:

```text
start_coding_run(...)
-> durable ledger
-> source/evidence/proposal or verify path
-> await structured continuation when needed
-> approve, apply, execute, repair, cancel, or inspect
-> completed, blocked, rejected, failed, or cancelled
```

The run supervisor is deterministic. It coordinates existing specialists and
records state; it does not become a monolithic shell agent.

After Phase 9, the user will conduct integration tests through E2E coding-agent
workflows. Phase 9 closure must therefore leave the supported coding-agent
surface self-contained for E2E verification. The implementation must not rely
on Phase 10 or any later development plan for read-only runs, patch proposals,
proposal inspection, approval continuation, managed apply, bounded execution,
repair, cancellation, get-by-run-id inspection, ledger reload, public
sanitization, and original-source immutability. Phase 10 remains only the
repository-scale reading expansion for broad architecture, impact, ownership,
and migration questions.

## Mandatory Skills

- `development-plan`: load before reading, executing, reviewing, or closing
  this plan.
- `local-llm-architecture`: load before changing coding-agent prompt context,
  run orchestration, role boundaries, or live-gate expectations.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before changing prompts, running live LLM gates, or
  reviewing live-gate trace quality.

## Mandatory Rules

- Keep source fetching, code reading, writing, modifying, patching, applying,
  execution, and verification in their current specialist boundaries.
- `coding_run` owns only ledger persistence, deterministic state transitions,
  continuation validation, public projection, event appending, and specialist
  invocation order.
- Deterministic code owns source identity validation, approval validation,
  allowed continuation actions, ledger writes, path containment, side-effect
  caps, and public sanitization.
- LLM stages remain inside existing specialists. The run supervisor must not
  add a broad new global planning prompt.
- Require explicit `objective_type` in run-start requests:
  `read_only`, `propose_patch`, or `verify_repair`.
- Require explicit `action` in continuation requests:
  `approve_and_verify` or `cancel`.
- Apply and execute only through the Phase 5, Phase 6, and Phase 8 trusted
  primitives.
- Preserve original source checkouts. All apply and execution effects remain
  inside managed apply workspaces.
- Store ledgers only under
  `<workspace_root>/coding_runs/<run_id>/` in this phase.
- Public responses and events must omit absolute source roots, workspace roots,
  cache keys, `.env`, `.git`, raw command lines, full stdout/stderr, full
  source dumps, secrets, and binary content.
- Existing background-worker accepted coding tasks remain review-only. This
  phase adds a direct trusted run API; it does not add background auto-apply,
  auto-execute, or adapter delivery.
- Do not close Phase 9 while any supported run-supervisor E2E workflow still
  depends on Phase 10, control-console work, background-worker auto-execution,
  MongoDB run persistence, repository-scale reading fan-out, or another future
  plan.
- Live LLM gates must run one case at a time. Passing pytest is necessary but
  not sufficient; each gate requires raw trace inspection and a human-authored
  Markdown review artifact.
- After automatic context compaction, the parent or active execution agent must
  reread this entire plan before continuing implementation, verification,
  handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Create the `coding_run` package with models, ledger persistence, supervisor,
  and README.
- Export `start_coding_run(...)`, `continue_coding_run(...)`, and
  `get_coding_run(...)` from `kazusa_ai_chatbot.coding_agent`.
- Add deterministic state-machine validation for all legal and illegal
  transitions in this plan.
- Persist one canonical JSON ledger plus append-only event JSONL for every run.
- Implement `read_only`, `propose_patch`, and `verify_repair` start flows.
- Implement `approve_and_verify` and `cancel` continuation flows.
- Route apply, execution, and repair only through
  `verify_and_repair_code_change(...)` when a run has patch artifacts and
  structured execution specs.
- Preserve existing direct primitive APIs without compatibility shims or alias
  request shapes.
- Add deterministic interface, ledger, transition, persistence, sanitization,
  approval, cancellation, and regression tests.
- Add deterministic public E2E workflow tests covering `read_only`,
  `propose_patch`, `propose_patch -> get -> approve_and_verify`,
  `propose_patch -> get -> cancel`, seeded `verify_repair`, process-boundary
  reload, source immutability, and public projection sanitization through the
  Phase 9 APIs.
- Keep the five prepared live LLM gates in
  `tests/test_coding_agent_phase9_run_supervisor_live_llm.py`.
- During execution closure, run the five live gates one at a time and author a
  Markdown review artifact from each raw JSON trace.
- Update coding-agent ICDs, HOWTO, architecture references, and the plan
  registry.

## Deferred

- Repository-scale reading fan-out remains Phase 10 scope and must not be
  required for post-Phase-9 E2E tests of supported run-supervisor workflows.
- MongoDB run persistence is outside this phase.
- Control-console UI for run display is outside this phase.
- Background-worker auto-apply, auto-execute, auto-repair, and adapter delivery
  are outside this phase.
- Arbitrary shell command planning, package installation, dependency solving,
  Docker, deployment, database mutation, repository push, and network tools are
  outside this phase.
- Direct mutation of original source checkouts is outside this phase.
- Run scope-revision continuation is outside this phase. Blocked runs preserve
  enough public detail for a caller to cancel or start a narrower new run.
- Compatibility wrappers, fallback entrypoints, feature flags, alternate
  ledgers, and alias request shapes are outside this phase.

## Cutover Policy

Overall strategy: compatible public API addition.

| Area | Policy | Instruction |
|---|---|---|
| Run APIs | compatible | Add `start_coding_run(...)`, `continue_coding_run(...)`, and `get_coding_run(...)` beside existing direct primitives. |
| Existing direct APIs | compatible | Preserve `answer_code_question(...)`, `propose_code_change(...)`, `apply_approved_patch(...)`, `execute_code_check(...)`, and `verify_and_repair_code_change(...)`. |
| Run objective routing | bigbang inside new API | Require closed `objective_type`; do not infer run objective through a new broad LLM router. |
| Continuation routing | bigbang inside new API | Require closed `action`; do not infer approval, cancellation, or execution from prose. |
| Ledger persistence | bigbang inside new package | Use one workspace-local JSON ledger and one JSONL event file. Do not add MongoDB or alternate stores. |
| Background worker | compatible | Keep accepted coding tasks review-only. Do not wire background work to trusted run continuation. |
| Tests and docs | bigbang | Update tests and docs to describe the implemented durable run boundary. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- Any compatible surface preserves only the listed API. It does not authorize
  fallback mappers, alias modules, or alternate request shapes.
- Any bigbang area must use the canonical new behavior directly.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

The coding agent exposes:

```python
from kazusa_ai_chatbot.coding_agent import continue_coding_run
from kazusa_ai_chatbot.coding_agent import get_coding_run
from kazusa_ai_chatbot.coding_agent import start_coding_run
```

The APIs operate on one workspace-local ledger:

```text
<workspace_root>/coding_runs/<run_id>/run.json
<workspace_root>/coding_runs/<run_id>/events.jsonl
```

Supported start objectives:

- `read_only`: resolve source, gather evidence, synthesize answer, persist
  completed run.
- `propose_patch`: resolve source or source-free request, create review-only
  patch proposal, persist `awaiting_approval`.
- `verify_repair`: accept trusted approval, execution specs, and either
  supplied initial artifacts or generated proposal artifacts, then compose
  Phase 8 verify-and-repair into a completed or terminal run.

Supported continuation actions:

- `approve_and_verify`: from `awaiting_approval`, validate approval and
  execution specs, call Phase 8 verify-and-repair with the run proposal
  artifacts, then persist attempts and terminal state.
- `cancel`: from any non-terminal state, persist `cancelled` without applying
  or executing anything.

Post-Phase-9 E2E readiness means a caller can exercise the following workflows
only through Phase 9 public APIs and committed fixtures:

- start a read-only run, inspect its persisted result, and reload it by run id;
- start a patch proposal run, inspect proposal artifacts, and verify no side
  effects occurred before approval;
- continue an awaiting-approval proposal into managed apply, bounded
  execution, repair when needed, and terminal status;
- cancel a non-terminal run and verify cancellation persists without apply or
  execution side effects;
- start a seeded verify-repair run and inspect apply, execution, repair, and
  final attempt history;
- prove every workflow preserves the original source tree and public
  projection sanitization.

Any broad repository-scale question that needs whole-repo subsystem fan-out,
impact analysis across many owners, or repository evidence graph synthesis is
outside Phase 9 E2E readiness and belongs to Phase 10.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| New module | Add `coding_run` | Durable run lifecycle is a new owner and should not bloat direct primitive modules. |
| Objective routing | Require `objective_type` | Weak local LLMs should not infer global run mode; direct callers and future UI can supply the closed objective. |
| Continuation routing | Require `action` | Approval, execution, and cancellation are deterministic control inputs, not semantic prompt decisions. |
| Ledger storage | Workspace-local JSON and JSONL | Matches current coding-agent managed storage and avoids database blast radius. |
| Apply/execute/repair | Reuse Phase 8 verifier | Avoids duplicating approval, managed-copy, execution, repair, and redaction logic. |
| Supplied artifacts | Allow only for `verify_repair` | Enables trusted callers to resume or test existing reviewed proposals without bypassing validation. |
| Post-phase E2E readiness | Phase 9 owns E2E closure for supported workflows | The user plans comprehensive E2E tests immediately after Phase 9; run lifecycle, continuation, verification, repair, cancellation, reload, sanitization, and source immutability must be complete before Phase 10. |
| Blocked runs | Public blocker records only | Keeps scope narrow; new narrowed runs are explicit caller decisions in this phase. |
| Background worker | Remains review-only | Trusted side-effect continuation is not authorized from accepted chat prose. |

## Change Surface

### Create

- `src/kazusa_ai_chatbot/coding_agent/coding_run/__init__.py`: submodule
  exports for the run supervisor.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/models.py`: run request,
  response, ledger, event, blocker, attempt, and status contracts.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/ledger.py`: JSON ledger and
  JSONL event persistence, atomic write, load, append, and public projection.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/supervisor.py`: start,
  continue, get orchestration and transition validation.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/README.md`: module ICD.
- `tests/test_coding_agent_phase9_interface.py`: public export and model
  interface checks.
- `tests/test_coding_agent_phase9_run_supervisor_contracts.py`: deterministic
  transition, ledger, sanitization, approval, cancellation, and persistence
  tests.
- `tests/test_coding_agent_phase9_e2e_workflows.py`: deterministic public E2E
  workflow tests for supported Phase 9 run paths and source immutability.
- `tests/test_coding_agent_phase9_run_supervisor_live_llm.py`: prepared real
  LLM gates already authored ahead of implementation.

### Modify

- `src/kazusa_ai_chatbot/coding_agent/__init__.py`: export public run APIs and
  models.
- `src/kazusa_ai_chatbot/coding_agent/models.py`: re-export or reference
  public run model aliases only when needed for top-level imports.
- `src/kazusa_ai_chatbot/coding_agent/README.md`: document run APIs and update
  architecture diagram.
- `docs/HOWTO.md`: add direct trusted run API runbook notes.
- `development_plans/README.md`: register this active approved plan.
- `development_plans/reference/designs/coding_agent_architecture.md`: point
  Phase 8 to the completed archive and Phase 9 to this active plan.
- `.gitignore` and `test_artifacts/.gitignore`: during execution closure,
  allow committed Phase 9 run-supervisor raw/review evidence only under
  `test_artifacts/llm_traces/coding_agent_run_supervisor/`.

### Keep

- Preserve all existing direct primitive APIs.
- Preserve current `code_fetching`, `code_reading`, `code_writing`,
  `code_modifying`, `code_patching`, `code_executing`, and `code_verifying`
  contracts.
- Preserve background-worker review-only behavior.

### Delete

- Do not delete production files in this phase.

## Overdesign Guardrail

- Actual problem: direct coding-agent primitives cannot yet hold one durable,
  inspectable coding session across proposal, approval, apply, verification,
  repair, cancellation, and status inspection.
- Minimal change: add one deterministic `coding_run` package that persists run
  state and composes existing primitives through closed start objectives and
  continuation actions.
- Ownership boundaries: `coding_run` owns lifecycle and ledger state;
  specialists own source, reading, writing, modifying, patching, apply,
  execution, and repair internals; deterministic code owns side-effect gates;
  LLM stages stay role-specific.
- Rejected complexity: global run-planning prompt, arbitrary tools, background
  auto-continuation, MongoDB persistence, UI, scope-revision continuation,
  compatibility request aliases, feature flags, fallback ledgers, and
  repository-scale reading.
- Evidence threshold: add deferred complexity only after a committed live gate
  or approved next-phase integration shows the closed objective/action run
  supervisor cannot support a required real coding workflow. A post-Phase-9
  E2E test failure inside `read_only`, `propose_patch`, approval,
  verify/repair, cancellation, reload, sanitization, or source immutability is
  a Phase 9 defect, not Phase 10 scope.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve this plan's contracts.
- The responsible agent must keep changes inside the listed change surface.
- The responsible agent must search for existing helper behavior before adding
  helpers. Reuse `work_ledger.py` ideas only when the durable run ledger needs
  equivalent bounded projection behavior; do not turn the writing work ledger
  into the run ledger.
- The responsible agent must not add new architecture, alternate persistence,
  compatibility layers, fallback paths, worker behavior, prompt routes, or
  extra features.
- The responsible agent must not perform unrelated cleanup, dependency
  upgrades, formatting churn, broad refactors, or prompt rewrites.
- If a required instruction is impossible under the current source contracts,
  stop and report the blocker instead of inventing a substitute.

## Contracts And Data Shapes

`CodingRunStartRequest`:

```python
{
    "question": str,
    "objective_type": "read_only | propose_patch | verify_repair",
    "source_url": str,
    "repo_url": str,
    "repo_hint": str,
    "local_root_hint": str,
    "local_path_hint": str,
    "requested_ref": str,
    "source_scope_hint": "repository | directory | file",
    "inline_sources": list[dict[str, object]],
    "workspace_root": str,
    "preferred_language": str,
    "max_answer_chars": int,
    "max_artifact_chars": int,
    "session_id": str,
    "approval": dict[str, object],
    "execution_specs": list[dict[str, object]],
    "repair_attempt_limit": int,
    "initial_patch_artifacts": list[dict[str, object]],
    "expected_source_identity": dict[str, object],
}
```

`CodingRunContinueRequest`:

```python
{
    "workspace_root": str,
    "run_id": str,
    "action": "approve_and_verify | cancel",
    "approval": dict[str, object],
    "execution_specs": list[dict[str, object]],
    "repair_attempt_limit": int,
    "reason": str,
}
```

`CodingRunGetRequest`:

```python
{
    "workspace_root": str,
    "run_id": str,
}
```

`CodingRunResponse`:

```python
{
    "status": (
        "created | source_resolved | evidence_collected | proposal_ready | "
        "awaiting_approval | applying | verifying | repairing | completed | "
        "blocked | rejected | failed | cancelled"
    ),
    "run_id": str,
    "goal": str,
    "objective_type": str,
    "answer_text": str,
    "repository": dict[str, object] | None,
    "source_scope": dict[str, object] | None,
    "evidence": list[dict[str, object]],
    "patch_artifacts": list[dict[str, object]],
    "changed_files": list[dict[str, str]],
    "apply_attempts": list[dict[str, object]],
    "execution_attempts": list[dict[str, object]],
    "repair_attempts": list[dict[str, object]],
    "attempts": list[dict[str, object]],
    "blockers": list[dict[str, object]],
    "events": list[dict[str, object]],
    "limitations": list[str],
    "trace_summary": list[str],
}
```

`CodingRunLedger`:

```python
{
    "schema_version": 1,
    "run_id": str,
    "status": str,
    "goal": str,
    "objective_type": str,
    "created_at": str,
    "updated_at": str,
    "source_request": dict[str, object],
    "repository": dict[str, object] | None,
    "source_scope": dict[str, object] | None,
    "answer_text": str,
    "evidence": list[dict[str, object]],
    "patch_artifacts": list[dict[str, object]],
    "changed_files": list[dict[str, str]],
    "approvals": list[dict[str, object]],
    "apply_attempts": list[dict[str, object]],
    "execution_attempts": list[dict[str, object]],
    "repair_attempts": list[dict[str, object]],
    "attempts": list[dict[str, object]],
    "blockers": list[dict[str, object]],
    "limitations": list[str],
    "trace_summary": list[str],
}
```

Event shape:

```python
{
    "event_id": str,
    "run_id": str,
    "sequence": int,
    "event_type": str,
    "status": str,
    "summary": str,
    "public_payload": dict[str, object],
}
```

Required event types:

- `run_created`
- `source_resolved`
- `evidence_collected`
- `proposal_ready`
- `awaiting_approval`
- `approval_received`
- `apply_attempt_recorded`
- `execution_attempt_recorded`
- `repair_attempt_recorded`
- `completed`
- `blocked`
- `rejected`
- `failed`
- `cancelled`

Illegal continuations return `rejected` without mutating the previous ledger
state except for a rejected-continuation event.

## LLM Call And Context Budget

Default context cap: 50k tokens per coding-agent model call unless route config
sets a lower model limit.

| Path | Before | After | Context inputs | Cap and truncation |
|---|---|---|---|---|
| `read_only` start | Caller invokes `answer_code_question(...)` directly | Run supervisor invokes the same direct primitive and stores public result | Existing code-reading source projection only | Existing code-reading caps remain authoritative |
| `propose_patch` start | Caller invokes `propose_code_change(...)` directly | Run supervisor invokes the same direct primitive and stores proposal/public evidence | Existing writing/modifying prompts only | Existing writing/modifying caps remain authoritative |
| `verify_repair` start | Caller invokes `verify_and_repair_code_change(...)` directly | Run supervisor invokes the same verifier and stores attempt summaries | Existing verifier repair feedback only | Existing Phase 8 repair-feedback caps remain authoritative |
| `approve_and_verify` continuation | Caller manually invokes apply/execute/verify primitives | Run supervisor invokes Phase 8 verifier using stored proposal artifacts and supplied approval/specs | Stored proposal artifacts plus structured continuation input | No raw ledger, command output, or full source enters prompts |

The run supervisor itself must not add a new LLM call. It may only call
existing specialist APIs whose prompt budgets are already bounded.

## Implementation Order

1. Create deterministic interface tests in
   `tests/test_coding_agent_phase9_interface.py`.
   - Expected before implementation: fail because run APIs and models are not
     exported.
2. Create deterministic run-supervisor tests in
   `tests/test_coding_agent_phase9_run_supervisor_contracts.py`.
   - Cover missing workspace, missing objective, illegal objective, missing
     run, illegal continuation, approval validation, cancellation, JSON
     persistence, event sequence, public sanitization, source immutability, and
     process-boundary reload.
   - Expected before implementation: fail for missing `coding_run`.
3. Create deterministic public E2E workflow tests in
   `tests/test_coding_agent_phase9_e2e_workflows.py`.
   - Cover `read_only`, `propose_patch`, proposal get, approval continuation,
     cancellation, seeded `verify_repair`, ledger reload, source immutability,
     and public sanitization through the Phase 9 public APIs.
   - Expected before implementation: fail because run APIs and `coding_run`
     behavior are missing.
4. Implement `coding_run.models`.
   - Define closed status, objective, action, request, response, ledger, event,
     and blocker contracts.
5. Implement `coding_run.ledger`.
   - Create safe run id generation, run directory containment, atomic JSON
     write, JSON load, event append, event sequence, and public projection.
6. Implement `coding_run.supervisor` start/get for `read_only`.
   - Compose `answer_code_question(...)`, persist completed or terminal runs,
     and satisfy read-only tests.
7. Implement `coding_run.supervisor` start/get for `propose_patch`.
   - Compose `propose_code_change(...)`, persist proposal and
     `awaiting_approval`, and satisfy proposal tests.
8. Implement continuation `cancel`.
   - Validate terminal-state rejection and non-terminal cancellation.
9. Implement start `verify_repair` and continuation `approve_and_verify`.
   - Compose `verify_and_repair_code_change(...)` with stored proposal
     artifacts and structured continuation input.
10. Wire top-level exports.
   - Update `coding_agent.__init__` and top-level model exports.
11. Update docs and architecture references.
    - Update coding-agent ICD, HOWTO, plan registry, and architecture doc.
12. Run focused deterministic tests, public E2E workflow tests, and
    regressions.
13. Run five live LLM gates one at a time, inspect raw evidence, and author
    review artifacts.
14. Run full non-live regression and independent code review.
15. Remediate review findings inside the approved change surface and rerun
    affected verification.
16. Move the plan to completed archive only after all acceptance criteria are
    met and evidence is recorded.

## Execution Model

- Parent agent owns orchestration, test code, static checks, verification,
  live-gate trace review, execution evidence, review remediation, lifecycle
  updates, and final sign-off.
- Parent agent establishes deterministic tests and confirms the prepared live
  gates collect before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns planned production code changes
  only; does not edit tests unless the parent explicitly directs it.
- Parent agent may continue docs, static checks, live-gate preparation, and
  regression validation while the production-code subagent edits production
  code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, evidence, and live-gate
  artifacts; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - deterministic test and E2E contract established.
  - Covers: implementation steps 1-3.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_coding_agent_phase9_interface.py tests\test_coding_agent_phase9_run_supervisor_contracts.py tests\test_coding_agent_phase9_e2e_workflows.py -q`.
  - Evidence: record expected pre-implementation failures or baseline results.
  - Handoff: production-code subagent starts Stage 2.
  - Sign-off: `Codex/2026-07-09`.
- [x] Stage 2 - `coding_run` models and ledger implemented.
  - Covers: implementation steps 4-5.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_coding_agent_phase9_run_supervisor_contracts.py -q`.
  - Evidence: record JSON ledger, event sequence, and sanitization results.
  - Handoff: continue with Stage 3.
  - Sign-off: `Codex/2026-07-09`.
- [x] Stage 3 - read-only and proposal start flows implemented.
  - Covers: implementation steps 6-7.
  - Verify: focused interface and run-supervisor tests.
  - Evidence: record public response and ledger projection contract results.
  - Handoff: continue with Stage 4.
  - Sign-off: `Codex/2026-07-09`.
- [x] Stage 4 - continuation actions implemented.
  - Covers: implementation steps 8-9.
  - Verify: approval, verification, repair, and cancellation tests.
  - Evidence: record managed apply/execute/repair attempt ledger results.
  - Handoff: continue with Stage 5.
  - Sign-off: `Codex/2026-07-09`.
- [x] Stage 5 - exports and documentation complete.
  - Covers: implementation steps 10-11.
  - Verify: static greps and documentation diff review.
  - Evidence: record changed docs and grep results.
  - Handoff: continue with Stage 6.
  - Sign-off: `Codex/2026-07-09`.
- [x] Stage 6 - focused deterministic regression complete.
  - Covers: implementation step 12.
  - Verify: Phase 5, Phase 6, Phase 7, Phase 8, Phase 9 focused commands,
    and Phase 9 public E2E workflow command.
  - Evidence: record command output.
  - Handoff: continue with Stage 7.
  - Sign-off: `Codex/2026-07-09`.
- [x] Stage 7 - five live LLM gates complete.
  - Covers: implementation step 13.
  - Verify: run every live gate command one at a time with raw evidence
    inspection before the next gate.
  - Evidence: commit raw JSON and human-authored Markdown review artifacts.
  - Handoff: continue with Stage 8.
  - Sign-off: `Codex/2026-07-09`.
- [x] Stage 8 - regression, independent code review, and lifecycle closeout
  complete.
  - Covers: implementation steps 14-16.
  - Verify: full non-live regression, independent code review, and affected
    reruns after remediation.
  - Evidence: record review findings, fixes, residual risks, and final
    completion decision.
  - Handoff: plan may move to completed archive after acceptance criteria pass.
  - Sign-off: `Codex/2026-07-09`.

## Verification

Static greps:

```powershell
rg -n "start_coding_run|continue_coding_run|get_coding_run" src\kazusa_ai_chatbot\coding_agent tests docs development_plans
rg -n "coding_agent_phase8_verify_repair_loop_plan.md" development_plans\reference\designs\coding_agent_architecture.md
rg -n "CODING_AGENT_RUN_LLM|RUN_SUPERVISOR_LLM" src\kazusa_ai_chatbot\coding_agent tests
```

Expected results:

- The first grep shows only public exports, `coding_run` implementation,
  tests, docs, and this plan.
- The second grep points only to the completed archive path for Phase 8.
- The third grep returns no matches. A nonzero `rg` exit code is expected;
  Phase 9 must not add a new broad run-supervisor LLM route.

Prepared live-gate collection:

```powershell
venv\Scripts\python -m py_compile tests\test_coding_agent_phase9_run_supervisor_live_llm.py
venv\Scripts\python -m pytest tests\test_coding_agent_phase9_run_supervisor_live_llm.py --collect-only -q -m live_llm
```

Focused deterministic commands:

```powershell
venv\Scripts\python -m pytest tests\test_coding_agent_phase9_interface.py -q
venv\Scripts\python -m pytest tests\test_coding_agent_phase9_run_supervisor_contracts.py -q
venv\Scripts\python -m pytest tests\test_coding_agent_phase9_e2e_workflows.py -q
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_interface.py tests\test_coding_agent_phase8_verify_repair_contracts.py tests\test_coding_agent_existing_source_planning_contracts.py tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase5_patch_apply_contracts.py -q
```

Post-Phase-9 E2E readiness command:

```powershell
venv\Scripts\python -m pytest tests\test_coding_agent_phase9_e2e_workflows.py tests\test_coding_agent_phase9_run_supervisor_contracts.py tests\test_coding_agent_phase9_interface.py -q
```

Expected result: every supported Phase 9 direct workflow passes through
`start_coding_run(...)`, `continue_coding_run(...)`, and `get_coding_run(...)`
without Phase 10, control-console UI, MongoDB run persistence, background
auto-execution, repository-scale reading fan-out, arbitrary shell planning, or
adapter delivery.

Live LLM gates, one command at a time:

```powershell
venv\Scripts\python -m pytest tests\test_coding_agent_phase9_run_supervisor_live_llm.py::test_coding_run_live_gate_01_read_only_state_persistence -q -s -m live_llm
venv\Scripts\python -m pytest tests\test_coding_agent_phase9_run_supervisor_live_llm.py::test_coding_run_live_gate_02_patch_proposal_awaits_approval -q -s -m live_llm
venv\Scripts\python -m pytest tests\test_coding_agent_phase9_run_supervisor_live_llm.py::test_coding_run_live_gate_03_approve_and_verify_success -q -s -m live_llm
venv\Scripts\python -m pytest tests\test_coding_agent_phase9_run_supervisor_live_llm.py::test_coding_run_live_gate_04_cancel_after_proposal -q -s -m live_llm
venv\Scripts\python -m pytest tests\test_coding_agent_phase9_run_supervisor_live_llm.py::test_coding_run_live_gate_05_seeded_repair_attempt_ledger -q -s -m live_llm
```

After each live command, inspect the raw JSON trace and author a matching
Markdown review artifact:

```text
test_artifacts/llm_traces/coding_agent_run_supervisor/<gate_id>_review.md
```

Final non-live regression:

```powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
```

## Prepared Real LLM Gates

The five real LLM gates were prepared before implementation in
`tests/test_coding_agent_phase9_run_supervisor_live_llm.py`.

| Gate | Test | Contract focus |
|---|---|---|
| Gate 01 | `test_coding_run_live_gate_01_read_only_state_persistence` | Read-only source/evidence answer, durable get, public sanitization. |
| Gate 02 | `test_coding_run_live_gate_02_patch_proposal_awaits_approval` | Live proposal generation pauses before approval, no apply or execution. |
| Gate 03 | `test_coding_run_live_gate_03_approve_and_verify_success` | Approval continuation applies and verifies through managed copy and Phase 8 repair when needed. |
| Gate 04 | `test_coding_run_live_gate_04_cancel_after_proposal` | Proposal history survives deterministic cancellation, no side effects. |
| Gate 05 | `test_coding_run_live_gate_05_seeded_repair_attempt_ledger` | Hard mixed seeded repair records failed execution, repair, final success, and persisted attempt history. |

Gate coverage against the architecture closure policy:

- simple focused case: Gate 01;
- small multi-file case: Gate 02;
- parser or edge-case case: Gate 03;
- cross-layer behavior case: Gate 04;
- hard mixed source/test/docs interaction: Gate 05.

## Confidence Review

Initial first-time live-gate confidence from the directional architecture was
78%.

| Gap | Root cause | Plan update |
|---|---|---|
| Global run objective was inferred from prose. | A weak local LLM could route read/propose/verify incorrectly before specialists see bounded context. | Add required closed `objective_type`; no broad run-router prompt. |
| Continuation intent was underspecified. | Approval, cancellation, and execution are side-effect decisions and should not be inferred from answer text. | Add required closed `action`; continuation validation is deterministic. |
| The ledger schema was directional and too open. | Tests need stable public fields for events, attempts, blockers, and reload behavior. | Define exact `CodingRunLedger`, `CodingRunResponse`, event shape, and event types. |
| Post-Phase-9 E2E readiness was implicit. | The user will run comprehensive E2E workflows immediately after Phase 9, so missing run lifecycle behavior cannot be deferred to Phase 10. | Add public E2E workflow tests, explicit E2E readiness target state, and acceptance criteria that treat supported workflow failures as Phase 9 defects. |
| Hard repair gate depended on first proposal quality. | A live LLM might produce a correct initial patch and fail to exercise repair attempts. | Allow supplied reviewed initial artifacts only for `verify_repair`; all artifacts still pass Phase 8 validation. |
| Closure evidence path was not committed or reviewable. | Raw pytest success does not prove live LLM quality. | Require raw JSON traces and human-authored Markdown review artifacts per gate. |

Post-update first-time pass confidence is 91% for Phase 9 closure plus the
post-Phase-9 E2E readiness contract.

The confidence increase comes from removing global routing ambiguity and
reusing the already verified Phase 8 apply/execute/repair primitive instead of
duplicating side-effect logic. The plan does not loosen the gates or tune
prompts to the fixture text. Remaining risk is live LLM proposal quality in
Gates 02 and 03; Gate 03 is protected by managed verify/repair, and Gate 02 is
only required to produce a reviewable source-backed proposal that pauses before
side effects. The added E2E readiness risk is process-boundary state fidelity
across multiple public API calls; the plan addresses it with ledger reload,
event sequence, public projection, source immutability, and multi-step E2E
tests before live-gate closure.

## Independent Plan Review

Review date: 2026-07-09.

| Finding | Severity | Resolution |
|---|---|---|
| The directional reference had no executable plan, status, progress checklist, or verification gates. | blocker | Created this approved active short-term plan with required sections and exact commands. |
| The directional reference did not define prepared real LLM gates ahead of execution. | blocker | Added five prepared live LLM gates under `tests/` and listed exact one-at-a-time commands. |
| Run objective and continuation routing were left open to interpretation. | blocker | Added closed `objective_type` and `action` enums and banned a broad run-router LLM. |
| Public ledger fields were not stable enough for tests, review, or future UI display. | blocker | Defined `CodingRunResponse`, ledger shape, event shape, and required event types. |
| Phase 9 could be misread as depending on Phase 10 for comprehensive E2E integration. | blocker | Added post-Phase-9 E2E readiness as a target state, deterministic E2E workflow tests, verification command, confidence gap, and acceptance criterion. |
| The architecture reference still pointed Phase 8 at the old active plan path. | non-blocking | Update the architecture reference to the completed Phase 8 archive and this active Phase 9 plan. |
| Scope-revision continuation risked expanding the first executable run supervisor. | non-blocking | Deferred scope-revision continuation and required blocked runs to preserve public restart details. |

Approval status: no unresolved plan-review blockers remain after these
resolutions.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt payload leaks, persistence risk,
  brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`,
  `Change Surface`, exact contracts, implementation order, verification gates,
  and acceptance criteria.
- Regression and handoff quality, including prior-stage artifacts, focused and
  regression tests, execution evidence, next-stage handoff notes, and
  path-safe commands.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `start_coding_run(...)`, `continue_coding_run(...)`, and `get_coding_run(...)`
  are exported from `kazusa_ai_chatbot.coding_agent`.
- The run supervisor persists JSON ledgers and JSONL events under
  `<workspace_root>/coding_runs/<run_id>/`.
- The run supervisor rejects missing workspace, missing objective, illegal
  objective, missing run, illegal continuation, missing approval, and
  unsupported execution specs without creating managed apply workspaces.
- `read_only`, `propose_patch`, and `verify_repair` start flows work through
  existing specialists and preserve their independent public contracts.
- `approve_and_verify` and `cancel` continuation flows are deterministic and
  auditable.
- Public responses and events are sanitized and omit absolute roots, cache
  keys, raw command output, full source dumps, `.env`, `.git`, and secrets.
- Original source trees remain unchanged during all run flows.
- Focused deterministic Phase 9 tests and Phase 5 to Phase 8 regressions pass.
- Public Phase 9 E2E workflow tests pass for read-only, proposal, get,
  approval continuation, managed apply, bounded execution, repair,
  cancellation, ledger reload, public sanitization, and source immutability.
- Comprehensive post-Phase-9 E2E tests for supported coding-agent workflows do
  not require Phase 10, control-console UI, MongoDB run persistence,
  background-worker auto-execution, repository-scale reading fan-out,
  arbitrary shell planning, or adapter delivery.
- Five live LLM gates pass one at a time with raw evidence and human-authored
  Markdown review artifacts committed for future regression review.
- Full non-live regression either passes or has failures classified outside the
  Phase 9 change surface with focused Phase 9 reruns passing.
- Documentation and architecture references describe the implemented durable
  run boundary.
- Independent code review finds no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Run ledger leaks local paths | Central public projection and sanitizer in `coding_run.ledger` | Sanitization deterministic tests and every live gate raw trace. |
| State machine accepts unsafe continuation | Closed transition table in deterministic code | Illegal-transition tests and cancellation tests. |
| Approval applies stale source | Reuse Phase 8 expected source identity and managed apply validation | Approval/verify tests and source hash checks. |
| New supervisor duplicates apply/execute/repair logic | Compose `verify_and_repair_code_change(...)` only | Static review and focused Phase 8 regression. |
| Live gates fail from route ambiguity | Closed objective/action inputs | Prepared gate contracts and confidence review. |

## Execution Evidence

Status: completed.

Pre-execution evidence:

- 2026-07-09: Prepared five live LLM gates in
  `tests/test_coding_agent_phase9_run_supervisor_live_llm.py`.
- 2026-07-09: `venv\Scripts\python -m py_compile
  tests\test_coding_agent_phase9_run_supervisor_live_llm.py` passed.
- 2026-07-09: `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase9_run_supervisor_live_llm.py --collect-only -q
  -m live_llm` collected 5 tests.
- 2026-07-09: Plan confidence review raised first-time pass confidence from
  78% to 91% by adding closed objective/action contracts, stable ledger
  shapes, and explicit post-Phase-9 E2E readiness gates.
- 2026-07-09: Independent plan review blockers were resolved before approval.

Execution evidence:

- 2026-07-09: User explicitly approved execution without subagents.
- 2026-07-09: Phase 9 deterministic TDD baseline failed as expected before
  implementation: `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase9_interface.py
  tests\test_coding_agent_phase9_run_supervisor_contracts.py
  tests\test_coding_agent_phase9_e2e_workflows.py -q` reported 17 failures
  for missing top-level exports and missing `coding_run`.
- 2026-07-09: Implemented `coding_run` models, ledger, supervisor, public
  exports, and deterministic contract/E2E tests.
- 2026-07-09: `venv\Scripts\python -m py_compile
  src\kazusa_ai_chatbot\coding_agent\coding_run\models.py
  src\kazusa_ai_chatbot\coding_agent\coding_run\ledger.py
  src\kazusa_ai_chatbot\coding_agent\coding_run\supervisor.py
  src\kazusa_ai_chatbot\coding_agent\coding_run\__init__.py
  src\kazusa_ai_chatbot\coding_agent\__init__.py
  tests\test_coding_agent_phase9_interface.py
  tests\test_coding_agent_phase9_run_supervisor_contracts.py
  tests\test_coding_agent_phase9_e2e_workflows.py` passed.
- 2026-07-09: `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase9_interface.py
  tests\test_coding_agent_phase9_run_supervisor_contracts.py
  tests\test_coding_agent_phase9_e2e_workflows.py -q` passed 17 tests.
- 2026-07-09: Implemented Phase 9 durable run supervision with
  `coding_run` models, workspace-local JSON ledger, JSONL event stream,
  deterministic start/continue/get state transitions, cancellation,
  approval-gated verification, and public-safe projection.
- 2026-07-09: Added deterministic tests:
  `tests/test_coding_agent_phase9_interface.py`,
  `tests/test_coding_agent_phase9_run_supervisor_contracts.py`, and
  `tests/test_coding_agent_phase9_e2e_workflows.py`.
- 2026-07-09: Added five prepared live LLM gates in
  `tests/test_coding_agent_phase9_run_supervisor_live_llm.py`.
- 2026-07-09: Updated docs and architecture references in
  `src/kazusa_ai_chatbot/coding_agent/README.md`, `docs/HOWTO.md`,
  `development_plans/reference/designs/coding_agent_architecture.md`, and
  `development_plans/reference/designs/coding_agent_phase9_run_supervisor_architecture.md`.
- 2026-07-09: Focused deterministic command passed 68 tests:
  `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase9_interface.py
  tests\test_coding_agent_phase9_run_supervisor_contracts.py
  tests\test_coding_agent_phase9_e2e_workflows.py
  tests\test_coding_agent_phase8_interface.py
  tests\test_coding_agent_phase8_verify_repair_contracts.py
  tests\test_coding_agent_existing_source_planning_contracts.py
  tests\test_coding_agent_phase6_code_executing_contracts.py
  tests\test_coding_agent_phase5_patch_apply_contracts.py -q`.
- 2026-07-09: Static grep
  `rg -n "start_coding_run|continue_coding_run|get_coding_run" src\kazusa_ai_chatbot\coding_agent tests docs development_plans`
  showed expected public exports, `coding_run` implementation, tests, docs,
  and plan references.
- 2026-07-09: Static grep
  `rg -n "coding_agent_phase8_verify_repair_loop_plan.md" development_plans\reference\designs\coding_agent_architecture.md`
  returned only the completed archive reference at line 19.
- 2026-07-09: Static grep
  `rg -n "CODING_AGENT_RUN_LLM|RUN_SUPERVISOR_LLM" src\kazusa_ai_chatbot\coding_agent tests`
  returned no matches, as expected.
- 2026-07-09: Live Gate 01
  `test_coding_run_live_gate_01_read_only_state_persistence` passed. Review
  artifact:
  `test_artifacts/llm_traces/coding_agent_run_supervisor/run_gate_01_read_only_persistence_review.md`.
- 2026-07-09: Live Gate 02
  `test_coding_run_live_gate_02_patch_proposal_awaits_approval` passed. Review
  artifact:
  `test_artifacts/llm_traces/coding_agent_run_supervisor/run_gate_02_proposal_awaits_approval_review.md`.
- 2026-07-09: Live Gate 03
  `test_coding_run_live_gate_03_approve_and_verify_success` passed after
  remediation. Root cause: the modifying PM/validator still treated focused
  verification tests as required writable companion targets even when the task
  made provided tests read-only. Fix: PM prompt and handoff validation now
  honor `read_only_paths`. Review artifact:
  `test_artifacts/llm_traces/coding_agent_run_supervisor/run_gate_03_approve_verify_success_review.md`.
- 2026-07-09: Live Gate 04
  `test_coding_run_live_gate_04_cancel_after_proposal` passed. Review
  artifact:
  `test_artifacts/llm_traces/coding_agent_run_supervisor/run_gate_04_cancel_after_proposal_review.md`.
- 2026-07-09: Live Gate 05
  `test_coding_run_live_gate_05_seeded_repair_attempt_ledger` passed after
  remediation. Root cause: repair proposal reading had partial evidence, while
  deterministic fallback only activated on zero evidence. Fix: structured
  execution repair feedback now shapes the read-only survey and bounded
  fallback ranking so source owners, caller wrappers, and protected
  verification tests are available together. Review artifact:
  `test_artifacts/llm_traces/coding_agent_run_supervisor/run_gate_05_seeded_repair_attempt_ledger_review.md`.
- 2026-07-09: Gate 05 inspected raw evidence showed final status `completed`,
  reloaded status `completed`, original source tree unchanged, final repaired
  artifacts for `releasefeed/cli.py` and `releasefeed/fetch.py`, protected
  verification paths `tests/test_fetch.py` and `tests/test_cli.py`, and final
  pytest status `succeeded`.
- 2026-07-09: Full non-live regression command
  `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q`
  completed with 5 failures, 2966 passes, 2 skips, and 559 deselections. Two
  coding-agent image acceptance failures passed on direct rerun. The remaining
  failures were classified outside the Phase 9 changed file set:
  `test_brain_model_route_api_applies_and_resets_selected_route` expects 13
  model routes while the current service returns 14, and two cognition prompt
  fingerprint tests expect `_COGNITION_SUBCONSCIOUS_PROMPT` byte length 3395
  while the current prompt is 3335 bytes.
- 2026-07-09: Independent code review was performed inline because the user
  explicitly required execution without subagents. Findings: clean up one test
  fixture indentation issue and sanitize event summaries before writing JSONL
  events. Fixes were applied.
- 2026-07-09: Affected post-review verification passed:
  `venv\Scripts\python -m py_compile
  src\kazusa_ai_chatbot\coding_agent\coding_run\ledger.py
  tests\test_coding_agent_existing_source_planning_contracts.py` and
  `venv\Scripts\python -m pytest
  tests\test_coding_agent_existing_source_planning_contracts.py
  tests\test_coding_agent_phase9_run_supervisor_contracts.py
  tests\test_coding_agent_phase9_e2e_workflows.py -q` passed 29 tests.
- 2026-07-09: Final focused Phase 5-9 command passed 68 tests after review
  remediation. No unresolved Phase 9 code-review blockers remain.
