# coding agent phase8 controlled verify repair loop plan

## Summary

- Goal: implement Phase 8 of the coding-agent architecture by adding a direct
  trusted verify-and-repair workflow that applies a patch proposal into a
  managed copy, runs bounded verification, and performs capped repair attempts
  through the Phase 7 modifying workflow.
- Plan class: high_risk_migration.
- Status: draft.
- Execution mode: parent-led implementation after explicit approval.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, and `debug-llm`.
- Cutover strategy: compatible public API addition. Existing proposal, apply,
  and execution APIs remain callable as independent primitives. The new Phase
  8 API composes them through a stricter direct trusted workflow.
- Highest-risk areas: approval scope, original-source immutability, execution
  output sanitization, repair feedback shape, repeated failed repairs, stale
  source identity, and keeping raw command output away from LLM prompts.
- Acceptance criteria: deterministic verify/repair contract tests pass,
  execution feedback is bounded and redacted, original source trees remain
  unchanged, repair attempts use fresh managed apply workspaces, Phase 5 and
  Phase 6 regressions remain passing, six committed live LLM repair gates
  pass one at a time with trace review, docs describe the direct trusted
  boundary, and independent code review accepts the implementation.

## Context

Phase 5 added explicit patch application into a managed apply workspace.
Phase 6 added bounded execution against that managed apply workspace. Phase 7
added a real existing-source modifying PM and File Agent planning boundary.

The current system can already:

- produce review-only patch proposals with `propose_code_change(...)`;
- apply approved patch artifacts into a managed copy with
  `apply_approved_patch(...)`;
- run allowlisted Python verification commands with `execute_code_check(...)`.

The system currently stops after execution. Failed verification results are not
fed into a repair attempt. This phase adds a small orchestration boundary that
can use failed verification evidence as structured repair feedback while
preserving the existing safety model.

## Mandatory Skills

- `development-plan`: load before reading, approving, executing, reviewing, or
  closing this plan.
- `local-llm-architecture`: load before changing role routing, repair-loop
  behavior, prompt input contracts, or supervisor orchestration.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before changing prompts, adding live LLM tests, running
  live LLM gates, or reviewing LLM trace quality.

## Mandatory Rules

- Require explicit structured approval before creating any managed apply
  workspace.
- Apply patches only into Phase 5 managed apply workspaces.
- Execute commands only through the Phase 6 structured execution boundary.
- Preserve the original source checkout byte-for-byte during verify and repair.
- Use fresh managed apply workspaces for each repair attempt.
- Feed only structured, bounded, redacted execution summaries into LLM repair.
- Reject raw stdout/stderr, absolute paths, environment values, cache keys,
  shell text, full source dumps, `.env`, `.git`, and secret-like content in
  repair prompts or public metadata.
- Cap repair attempts deterministically. The default cap is 1 and the hard cap
  is 2.
- Preserve the current allowlist of execution tools:
  `python_compileall` and `pytest`.
- Preserve direct primitive APIs:
  `propose_code_change(...)`, `apply_approved_patch(...)`, and
  `execute_code_check(...)`.
- Keep background-worker accepted tasks review-only. Background tasks gain no
  auto-apply, auto-execute, or auto-repair behavior in this phase.
- After context compaction, reread this entire plan before continuing
  execution, verification, lifecycle updates, or final reporting.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.

## Current Architecture Map

Current direct primitive flow:

```text
propose_code_change(...)
-> CodingPatchProposalResponse

apply_approved_patch(...)
-> CodingPatchApplyResponse

execute_code_check(...)
-> CodeExecutionResponse
```

Target Phase 8 direct trusted flow:

```text
CodingVerifyRepairRequest
-> source fetch and initial proposal when needed
-> structured approval validation
-> apply_approved_patch(...) into managed copy
-> execute_code_check(...) for each execution spec
-> if all checks pass: return succeeded
-> if checks fail and repair attempts remain:
     build bounded ExecutionRepairFeedback
     regenerate proposal through Phase 7 modifying workflow
     apply repaired proposal into a fresh managed copy
     rerun execution specs
-> CodingVerifyRepairResponse
```

## Scope

Phase 8 includes:

- adding a new direct trusted verify/repair module under
  `src/kazusa_ai_chatbot/coding_agent/code_verifying/`;
- adding public models for `CodingVerifyRepairRequest`,
  `CodingVerifyRepairResponse`, `VerifyRepairAttempt`, and
  `ExecutionRepairFeedback`;
- exporting `verify_and_repair_code_change(...)` from
  `kazusa_ai_chatbot.coding_agent`;
- composing existing proposal, apply, and execution primitives with stricter
  orchestration rules;
- extending `code_modifying` repair feedback to accept structured execution
  feedback from Phase 8 while continuing to reject raw command output;
- adding deterministic tests for approval, containment, caps, redaction,
  original-source immutability, fresh apply workspaces, and repair behavior;
- adding live LLM repair gates over existing source fixtures;
- updating coding-agent README, submodule READMEs, HOWTO, and architecture
  reference.

Phase 8 excludes:

- broad command planning;
- new execution tools beyond `python_compileall` and `pytest`;
- package installation, dependency repair, virtualenv creation, Docker image
  building, deployment, database access, adapter sends, or repository pushes;
- direct mutation of the original source checkout;
- background-worker auto-execution;
- persistent run ledgers or session state machines;
- repository-scale reading fan-out.

## Target State

The coding agent exposes one new direct trusted orchestration API:

```python
from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change
```

The API receives a source-backed coding request, structured approval scope, and
one or more structured execution specs. It returns a compact attempt ledger:

```text
attempt 1:
  proposal -> apply -> execute
attempt 2 when needed:
  repair proposal -> apply fresh copy -> execute
terminal:
  succeeded | failed | rejected | timed_out
```

The workflow never exposes the absolute managed apply path in public metadata.
The workflow never mutates the original source checkout. The workflow can
repair only by creating a new patch proposal and applying it into a fresh
managed copy.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| New module | Add `code_verifying` for orchestration | Keeps apply and execution primitives focused while giving verify/repair a clear owner. |
| Public API | Add `verify_and_repair_code_change(...)` | Avoids overloading proposal, apply, or execute APIs with orchestration state. |
| Approval | Require structured approval in the verify request | Preserves the Phase 5 trust boundary. |
| Apply target | Fresh managed apply workspace per attempt | Prevents repair attempts from building on a partially failed workspace. |
| Execution feedback | Structured redacted summary only | Makes repair local-LLM friendly and public safe. |
| Repair cap | Default 1, hard cap 2 | Provides useful recovery without unbounded loops. |
| Prompt owner | Phase 7 modifying PM owns repair interpretation | Keeps semantic repair with the existing-source planning role. |
| Background worker | Remains review-only | Prevents accepted chat tasks from silently running code. |

## Contracts And Data Shapes

Create `src/kazusa_ai_chatbot/coding_agent/code_verifying/models.py`.

`CodingVerifyRepairRequest`:

```python
{
    "question": str,
    "source_url": str,
    "repo_url": str,
    "repo_hint": str,
    "local_root_hint": str,
    "local_path_hint": str,
    "requested_ref": str,
    "source_scope_hint": str,
    "workspace_root": str,
    "preferred_language": str,
    "max_answer_chars": int,
    "max_artifact_chars": int,
    "approval": dict[str, object],
    "execution_specs": list[dict[str, object]],
    "repair_attempt_limit": int,
    "max_repair_feedback_chars": int,
    "initial_patch_artifacts": list[dict[str, object]],
    "expected_source_identity": dict[str, object],
}
```

`initial_patch_artifacts` is optional. Trusted direct callers may supply an
already reviewed proposal so verify/repair can apply, execute, and repair an
existing patch package instead of always generating the first proposal inside
the verifier. Live LLM closure gates use this field to seed a known-bad but
review-valid proposal, guaranteeing the repair loop is exercised. When present,
`expected_source_identity` is required and must match the resolved source
identity before any managed apply workspace is created.

`ExecutionRepairFeedback`:

```python
{
    "feedback_source": "execution_verification",
    "attempt_index": int,
    "overall_status": "failed | timed_out",
    "failed_tools": list[str],
    "failed_paths": list[str],
    "exit_codes": list[dict[str, object]],
    "failure_summaries": list[str],
    "stdout_excerpt": str,
    "stderr_excerpt": str,
    "output_truncated": bool,
    "instruction": str,
}
```

`VerifyRepairAttempt`:

```python
{
    "attempt_index": int,
    "proposal_status": str,
    "apply_status": str,
    "execution_statuses": list[str],
    "patch_artifact_count": int,
    "changed_files": list[dict[str, str]],
    "apply_package_id": str | None,
    "limitations": list[str],
    "trace_summary": list[str],
}
```

`CodingVerifyRepairResponse`:

```python
{
    "status": "succeeded | failed | rejected | timed_out",
    "answer_text": str,
    "repository": dict[str, object] | None,
    "source_scope": dict[str, object] | None,
    "attempts": list[dict[str, object]],
    "final_patch_artifacts": list[dict[str, object]],
    "final_changed_files": list[dict[str, str]],
    "final_apply": dict[str, object] | None,
    "final_execution": list[dict[str, object]],
    "limitations": list[str],
    "trace_summary": list[str],
}
```

Extend `CodeModificationRequest` with optional structured repair input:

```python
{
    "repair_feedback": {
        "feedback_source": "execution_verification",
        "attempt_index": int,
        "execution_feedback": dict[str, object],
        "previous_modification_artifacts": list[dict[str, object]],
        "instruction": str,
    }
}
```

Keep `code_modifying.models.ALLOWED_REPAIR_FEEDBACK_SOURCES` closed. Add
`execution_verification` only after deterministic tests prove raw output is
never accepted by the normalizer or PM prompt builder.

## Orchestration Rules

- Validate `workspace_root` before proposal generation.
- Require explicit source-backed request fields. Phase 8 targets existing
  source repair, not source-free project generation.
- Call `propose_code_change(...)` for the initial proposal unless the trusted
  direct request supplies `initial_patch_artifacts` with matching
  `expected_source_identity`.
- Validate proposal status before apply.
- Call `apply_approved_patch(...)` with structured approval and expected
  source identity.
- Call `execute_code_check(...)` once per structured execution spec.
- Treat any `failed` or `timed_out` execution response as verification
  failure.
- Treat any `rejected` execution response as terminal unless rejection is due
  to a repaired file path that no longer exists.
- Build repair feedback only from sanitized `CodeExecutionResponse` fields.
- Truncate combined repair feedback to `max_repair_feedback_chars`.
- Rerun source reading before a repair proposal when the current source
  evidence is absent or stale.
- Apply repaired proposals into a new managed apply workspace.
- Return success only when every execution spec succeeds in the same attempt.
- Return terminal failure when repair attempts are exhausted.

## Repair Feedback Redaction

Before any execution feedback reaches `code_modifying`, the verifier must:

- remove absolute paths;
- remove workspace roots and managed package ids from excerpts;
- remove environment variable names and values that look secret-like;
- remove command argv details beyond the allowlisted tool name;
- cap stdout and stderr excerpts independently;
- cap total feedback characters;
- replace multiline failure output with compact bullet summaries;
- preserve enough context for semantic repair, including failing test names,
  exception type names, file-relative paths, and assertion summaries.

## Implementation Checklist

### Stage 0 - Baseline

- Read this plan, `development_plans/README.md`, coding-agent README files, and
  relevant source/test files.
- Record `git status --short`.
- Run focused baseline tests:

```powershell
venv\Scripts\python -m pytest tests\test_coding_agent_phase5_patch_apply_contracts.py tests\test_coding_agent_phase6_code_executing_contracts.py -q
```

### Stage 1 - Contract Tests First

- Add failing deterministic tests for:
  - missing approval rejection;
  - source-free request rejection;
  - unsupported execution spec rejection;
  - original-source immutability;
  - fresh managed apply workspace per attempt;
  - failed execution triggers one repair attempt;
  - repair attempt cap enforcement;
  - timed-out execution feedback handling;
  - redaction of absolute paths and raw command output;
  - terminal success after repaired attempt;
  - terminal failure after exhausted repairs.

### Stage 2 - Models And Module Boundary

- Create `code_verifying/README.md`.
- Create `code_verifying/models.py`.
- Create `code_verifying/supervisor.py`.
- Export `verify_and_repair_code_change(...)` from
  `code_verifying/__init__.py` and top-level `coding_agent/__init__.py`.
- Keep response shapes public-safe and compatible with existing coding-agent
  model style.

### Stage 3 - Apply And Execute Composition

- Implement initial proposal generation through `propose_code_change(...)`.
- Implement apply through `apply_approved_patch(...)`.
- Implement execution through `execute_code_check(...)`.
- Preserve apply/execution response sanitization.
- Aggregate attempt summaries without exposing managed source roots.

### Stage 4 - Structured Repair Feedback

- Add `execution_verification` as an allowed repair feedback source.
- Build `ExecutionRepairFeedback` from sanitized execution responses.
- Add deterministic tests proving raw stdout/stderr and absolute paths are
  rejected or redacted before PM input.
- Update Phase 7 modifying PM prompt builder to consume structured execution
  repair feedback.

### Stage 5 - Repair Attempt Loop

- Reinvoke existing-source proposal generation with repair feedback.
- Apply repaired artifacts into a fresh managed apply workspace.
- Rerun the same execution specs against the repaired apply workspace.
- Preserve an attempt ledger for traceability.
- Stop immediately when all specs succeed.
- Stop with terminal failure when repair attempts are exhausted.

### Stage 6 - Documentation

- Update:
  - `src/kazusa_ai_chatbot/coding_agent/README.md`
  - `src/kazusa_ai_chatbot/coding_agent/code_modifying/README.md`
  - `src/kazusa_ai_chatbot/coding_agent/code_patching/README.md`
  - `src/kazusa_ai_chatbot/coding_agent/code_executing/README.md`
  - `docs/HOWTO.md`
  - `development_plans/reference/designs/coding_agent_architecture.md`
- Document Phase 8 as a direct trusted API that composes proposal, apply,
  execute, and capped repair.

### Stage 7 - Live LLM Gates

- Load `debug-llm` before running live cases.
- Run live LLM gates one command at a time.
- Use the committed source fixtures in
  `tests/test_coding_agent_phase8_verify_repair_live_llm.py`.
- Each gate supplies a known-bad but review-valid `initial_patch_artifacts`
  package so the first managed apply/execution attempt fails and the real LLM
  repair path must run.
- The six gates, from simple to hard, are:
  - Gate 01 `verify_repair_gate_01_median_boundary`: single-file median
    boundary repair.
  - Gate 02 `verify_repair_gate_02_cli_flag_handoff`: small multi-file counter
    and CLI flag handoff repair.
  - Gate 03 `verify_repair_gate_03_duplicate_anchor_parser`: parser edge-case
    duplicate-anchor repair.
  - Gate 04 `verify_repair_gate_04_soft_delete_cross_layer`: cross-layer model,
    store, and API soft-delete repair.
  - Gate 05 `verify_repair_gate_05_fetch_cache_cli`: hard mixed fetch, retry,
    cache, CLI, tests, and README context repair.
  - Gate 06 `verify_repair_gate_06_release_feed_cache_cli`: retained hard
    mocked-I/O repair gate based on the Phase 7 Gate 05 failure mode, using a
    different release-feed cache/offline/CLI fixture.
- Require trace review that confirms:
  - failed execution output was summarized and redacted;
  - PM repair targeted the source owner path;
  - repaired proposal applied into a fresh managed copy;
  - final execution succeeded in the repaired attempt.
  - the original source tree remained byte-for-byte unchanged.

### Stage 7A - Hard-Gate Root-Cause Readiness

Before running Gate 05 or Gate 06, add and pass deterministic readiness checks
that address the root cause surfaced by the Phase 7 retained diagnostic gate:
the local model selected the right fetch/cache task but omitted required
companion artifacts after bounded repair. These checks must not weaken the live
gate assertions.

- Add a repair-feedback fixture where the first proposal updates only a fetch
  source file while pytest failures identify missing CLI flag wiring and mocked
  I/O behavior.
- Prove the verifier converts that execution failure into structured repair
  feedback that contains failing test names, relative source/test paths, and
  concise assertion summaries, while redacting raw full output and absolute
  paths.
- Prove the modifying request produced for repair includes explicit required
  source-owner paths and companion context paths from the failed execution, so
  the PM/programmer cannot treat absent exact helper code or absent newly
  authored tests as a blocker.
- Add deterministic contract validation that rejects a repaired proposal when
  it omits any required changed source owner path or when it changes protected
  verification tests for a gate.
- Run a dry-run repair trace review for Gate 05 and Gate 06 before live LLM
  execution, and record whether the PM handoff contains both runtime owner
  paths and the focused test evidence paths.

### Stage 8 - Regression And Review

- Run focused deterministic tests.
- Run Phase 5 and Phase 6 regressions.
- Run Phase 7 tests when Phase 7 is implemented.
- Run full non-live pytest.
- Run the `Independent Code Review` gate.
- Record all outcomes in `Execution Evidence`.

## Verification

Focused deterministic commands:

```powershell
venv\Scripts\python -m pytest tests\test_coding_agent_phase5_patch_apply_contracts.py -q
venv\Scripts\python -m pytest tests\test_coding_agent_phase6_code_executing_contracts.py -q
venv\Scripts\python -m pytest tests\test_coding_agent_phase7_existing_source_planning_contracts.py -q
```

New deterministic tests added by this plan:

```powershell
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_verify_repair_contracts.py -q
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_interface.py -q
```

Live LLM gates, one command at a time:

```powershell
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_verify_repair_live_llm.py::test_verify_repair_live_gate_01_median_boundary -q -s -m live_llm
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_verify_repair_live_llm.py::test_verify_repair_live_gate_02_cli_flag_handoff -q -s -m live_llm
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_verify_repair_live_llm.py::test_verify_repair_live_gate_03_duplicate_anchor_parser -q -s -m live_llm
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_verify_repair_live_llm.py::test_verify_repair_live_gate_04_soft_delete_cross_layer -q -s -m live_llm
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_verify_repair_live_llm.py::test_verify_repair_live_gate_05_fetch_cache_cli -q -s -m live_llm
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_verify_repair_live_llm.py::test_verify_repair_live_gate_06_release_feed_cache_cli -q -s -m live_llm
```

Final non-live regression:

```powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
```

## Independent Code Review

Before this plan can move to completed:

- run an independent code-review pass over changed production code, tests,
  prompts, and docs;
- lead with findings by severity and file/line reference;
- remediate accepted findings;
- rerun affected tests;
- record review outcome in `Execution Evidence`.

## Execution Evidence

Status: not started.

Record implementation commands, test results, live LLM gate notes, trace
review decisions, independent review findings, remediations, and final
closeout evidence here during approved execution.
