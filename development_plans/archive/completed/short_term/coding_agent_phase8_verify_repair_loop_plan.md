# coding agent phase8 controlled verify repair loop plan

## Summary

- Goal: implement Phase 8 of the coding-agent architecture by adding a direct
  trusted verify-and-repair workflow that applies a patch proposal into a
  managed copy, runs bounded verification, and performs capped repair attempts
  through the Phase 7 modifying workflow.
- Plan class: high_risk_migration.
- Status: completed.
- Execution mode: user-approved fallback execution without subagents.
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
  boundary, independent code review accepts the implementation, and any
  full-suite failures outside the Phase 8 change surface are recorded as
  residual external risks.

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
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.

## Must Do

- Add the direct trusted `verify_and_repair_code_change(...)` API.
- Add a `code_verifying` module that owns verify/repair orchestration.
- Compose the existing proposal, patch-apply, and execution primitives without
  changing their independent public contracts.
- Require structured approval and matching source identity before each managed
  apply workspace is created.
- Apply every attempt into a fresh Phase 5 managed apply workspace.
- Run only Phase 6 structured execution specs through `execute_code_check(...)`.
- Build repair feedback only from bounded, redacted execution response fields.
- Extend `code_modifying` to accept the structured `execution_verification`
  repair-feedback source.
- Add deterministic tests for approval, containment, caps, redaction, original
  source immutability, fresh apply workspaces, terminal success/failure, and
  Stage 7A hard-gate readiness.
- Keep the six real LLM gates in
  `tests/test_coding_agent_phase8_verify_repair_live_llm.py` and run them one
  at a time during execution closure.
- Produce a human-authored Markdown review artifact for each live LLM gate.
- Update coding-agent docs, HOWTO, and the architecture reference.
- Run every verification command listed in this plan and record evidence.

## Deferred

- Broad command planning is deferred.
- New execution tools beyond `python_compileall` and `pytest` are deferred.
- Package installation, dependency repair, virtualenv creation, Docker image
  building, deployment, database access, adapter sends, and repository pushes
  are deferred.
- Direct mutation of the original source checkout is outside this phase.
- Background-worker auto-apply, auto-execute, and auto-repair are outside this
  phase.
- Persistent run ledgers, session state machines, and repository-scale reading
  fan-out are deferred to later phases.
- Compatibility shims, alternate verify/repair entrypoints, feature flags,
  and fallback execution paths are outside this phase.

## Cutover Policy

Overall strategy: compatible public API addition.

| Area | Policy | Instruction |
|---|---|---|
| Direct verify/repair API | compatible | Add `verify_and_repair_code_change(...)` beside existing direct primitives. |
| Proposal/apply/execute primitives | compatible | Preserve `propose_code_change(...)`, `apply_approved_patch(...)`, and `execute_code_check(...)` as independent callable surfaces. |
| Repair feedback source | bigbang inside new feedback source | Add one canonical `execution_verification` source. Do not add aliases or legacy names. |
| Apply workspace handling | bigbang inside verifier | Every attempt uses a fresh managed apply workspace. Do not continue in a failed workspace. |
| Background worker | compatible | Keep accepted coding tasks review-only in this phase. |
| Tests and docs | bigbang | Update tests and docs to describe the implemented direct trusted boundary. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- Any compatible surface preserves only the listed API. It does not authorize
  fallback mappers, alias modules, or alternate request shapes.
- Any bigbang area must use the canonical new behavior directly.
- Any change to this cutover policy requires user approval before
  implementation.

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

## Change Surface

### Create

- `src/kazusa_ai_chatbot/coding_agent/code_verifying/__init__.py`: public
  submodule export for the verifier.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/models.py`: request,
  response, attempt, and repair-feedback shapes.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/supervisor.py`: direct
  orchestration of proposal, apply, execute, and bounded repair attempts.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/README.md`: module ICD.
- `tests/test_coding_agent_phase8_verify_repair_contracts.py`: deterministic
  verifier contract and Stage 7A readiness tests.
- `tests/test_coding_agent_phase8_interface.py`: top-level export and public
  response interface tests.

### Modify

- `src/kazusa_ai_chatbot/coding_agent/__init__.py`: export
  `verify_and_repair_code_change(...)`.
- `src/kazusa_ai_chatbot/coding_agent/code_modifying/models.py`: add the
  `execution_verification` repair-feedback source and validation.
- `src/kazusa_ai_chatbot/coding_agent/code_modifying/supervisor.py`: include
  structured execution repair feedback in PM/programmer handoff payloads.
- `src/kazusa_ai_chatbot/coding_agent/code_modifying/programmer.py`: explain
  the durable structured repair-feedback contract when needed by the existing
  prompt block.
- `src/kazusa_ai_chatbot/coding_agent/README.md`,
  `src/kazusa_ai_chatbot/coding_agent/code_modifying/README.md`,
  `src/kazusa_ai_chatbot/coding_agent/code_patching/README.md`,
  `src/kazusa_ai_chatbot/coding_agent/code_executing/README.md`,
  `docs/HOWTO.md`, and
  `development_plans/reference/designs/coding_agent_architecture.md`: document
  the implemented direct trusted verify/repair boundary.

### Keep

- Keep `propose_code_change(...)`, `apply_approved_patch(...)`, and
  `execute_code_check(...)` callable as independent primitives.
- Keep accepted background coding tasks review-only.
- Keep execution tool support limited to `python_compileall` and `pytest`.

### Delete

- No production files are deleted by this phase.

## Overdesign Guardrail

- Actual problem: the coding agent can apply and execute an approved patch, but
  failed verification is not converted into a bounded repair attempt.
- Minimal change: add one direct verifier module that composes existing
  proposal, apply, and execution primitives and adds a capped repair loop with
  structured feedback.
- Ownership boundaries: `code_verifying` owns orchestration, deterministic
  caps, source identity, public response safety, and repair attempt ledger;
  `code_patching` owns patch validation and managed apply workspaces;
  `code_executing` owns command validation and sanitized execution output;
  `code_modifying` owns semantic repair interpretation.
- Rejected complexity: broad command planning, package installation, arbitrary
  shell execution, background-worker auto-repair, persistent run ledgers,
  compatibility request aliases, fallback verifier paths, and repository-scale
  reading fan-out.
- Evidence threshold: add later complexity only after a committed real LLM gate
  or approved near-term integration shows the current bounded verifier cannot
  complete a supported coding repair workflow.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must keep changes inside the listed change surface.
- Changes outside the target coding-agent modules require strong justification
  in this plan before implementation.
- Existing helpers must be searched before adding new helpers.
- New helpers are allowed only when they remove meaningful repeated validation
  logic, isolate non-obvious domain behavior, or match an established local
  pattern.
- The responsible agent must preserve the direct primitive APIs and the
  background-worker review-only boundary.
- The responsible agent must stop and report a blocker if a required
  instruction is impossible under the current source contracts.

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

Supplied `initial_patch_artifacts` skip only the initial LLM proposal step.
They still go through the same patch review validation, approval validation,
source-identity validation, managed-copy apply, execution, repair, and public
sanitization path as generated proposal artifacts. Reject supplied initial
artifacts when the expected source identity is missing, mismatched, or when
patch review validation fails.

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

## LLM Call And Context Budget

Default context cap: 50k tokens per coding-agent model call unless route config
sets a lower model limit.

| Path | Before | After | Context inputs | Cap and truncation |
|---|---|---|---|---|
| Initial proposal | One existing Phase 7 modifying PM/programmer path when `propose_code_change(...)` is called | Same path, skipped only when a trusted request supplies reviewed `initial_patch_artifacts` | User question, source evidence, File Agent context, bounded companion paths | Existing `max_answer_chars`, `max_artifact_chars`, File Agent caps |
| Successful verification | No repair call | No additional LLM call after apply/execution succeeds | Sanitized attempt metadata only in response | Public response caps |
| Failed verification repair | Terminal failure after execution | Up to `repair_attempt_limit` additional Phase 7 modifying proposal attempts, hard capped at 2 | Structured `ExecutionRepairFeedback`, previous artifact summaries, source-owner paths, companion context paths | `max_repair_feedback_chars`, stdout/stderr caps, file/context caps |
| Live gate review | Raw test trace only | Agent-authored Markdown review after each real LLM gate | Raw JSON evidence emitted by the test | Human-readable review artifact, raw trace linked |

The verifier must not add a new model route. Repair uses existing
`CODING_AGENT_PM_LLM` and `CODING_AGENT_PROGRAMMER_LLM` through the Phase 7
modifying workflow. No direct chat response path gains a new model call.

## Orchestration Rules

- Validate `workspace_root` before proposal generation.
- Require explicit source-backed request fields. Phase 8 targets existing
  source repair, not source-free project generation.
- Call `propose_code_change(...)` for the initial proposal unless the trusted
  direct request supplies `initial_patch_artifacts` with matching
  `expected_source_identity`.
- Treat supplied `initial_patch_artifacts` as trusted only for caller identity,
  not for patch safety. Run the same review/apply validation used for generated
  proposals before creating a managed apply workspace.
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

## Implementation Order

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
- After each live gate run, inspect the raw JSON evidence and author a
  human-readable Markdown review artifact under
  `test_artifacts/llm_traces/coding_agent_verify_repair/`.
  Tests and scripts may emit raw JSON evidence only; the readable quality
  decision must be written by the agent from the raw trace.

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
- Add these exact deterministic readiness cases to
  `tests/test_coding_agent_phase8_verify_repair_contracts.py`:
  - `test_verify_repair_feedback_preserves_failure_summary_for_repair`
  - `test_verify_repair_feedback_redacts_absolute_paths_and_raw_output`
  - `test_verify_repair_request_includes_required_owner_and_context_paths`
  - `test_verify_repair_rejects_repair_omitting_required_owner_paths`
  - `test_verify_repair_rejects_repair_modifying_protected_verification_tests`
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

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused deterministic test contract before
  production implementation starts.
- Production-code subagent: exactly one native subagent, started after focused
  tests are written and baseline behavior is recorded; owns production code
  changes only; does not edit tests unless the parent explicitly directs it.
- Parent agent may continue integration tests, live-gate preparation, static
  checks, and validation while the production-code subagent edits production
  code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 0 - baseline recorded.
  - Covers: reading plan/docs/source and running Phase 5/6 baseline tests.
  - Verify: focused baseline command in Stage 0.
  - Evidence: record git status and baseline test output in `Execution
    Evidence`.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 1 - focused deterministic contract tests added.
  - Covers: missing approval, source-free rejection, unsupported execution
    spec rejection, immutability, fresh workspace, repair cap, timeout
    feedback, redaction, terminal success, and terminal failure tests.
  - Verify: targeted new test command fails for missing Phase 8 API or behavior
    before production implementation.
  - Evidence: record expected failure in `Execution Evidence`.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 2 - verifier module boundary implemented.
  - Covers: `code_verifying` models, supervisor, README, and public exports.
  - Verify: `tests/test_coding_agent_phase8_interface.py -q`.
  - Evidence: record changed files and command output.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 3 - apply and execute composition implemented.
  - Covers: proposal/apply/execute orchestration, initial artifact validation,
    public attempt ledger, and sanitized response aggregation.
  - Verify: relevant tests in
    `tests/test_coding_agent_phase8_verify_repair_contracts.py -q`.
  - Evidence: record command output and any fixed contract gaps.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 4 - structured repair feedback implemented.
  - Covers: `execution_verification` source, redaction, PM/programmer handoff,
    and raw-output rejection.
  - Verify: Stage 7A readiness command plus full Phase 8 deterministic tests.
  - Evidence: record command output and sanitizer notes.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 5 - repair attempt loop implemented.
  - Covers: capped retry, fresh apply workspace per attempt, terminal success,
    and terminal failure.
  - Verify: full Phase 8 deterministic tests.
  - Evidence: record attempt-ledger behavior from tests.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 6 - documentation updated.
  - Covers: coding-agent ICDs, HOWTO, and architecture reference.
  - Verify: documentation diff review plus static greps in Verification.
  - Evidence: record changed docs and grep results.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 7 - six live LLM gates passed one at a time.
  - Covers: gates 01 through 06 and human-authored review artifacts.
  - Verify: each live command in the Verification section run individually
    with trace inspection before the next command.
  - Evidence: record raw evidence paths, review artifact paths, and judgment.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 8 - regression and independent code review complete.
  - Covers: focused regression, full non-live pytest, independent code review,
    remediation, rerun checks, and lifecycle closeout.
  - Verify: final non-live regression plus affected focused reruns.
  - Evidence: record review findings, fixes, rerun commands, residual risks,
    and completion decision.
  - Sign-off: `Codex/2026-07-08`.

## Verification

Static greps:

```powershell
rg -n "execution_verification" src\kazusa_ai_chatbot\coding_agent tests
rg -n "verify_and_repair_code_change" src\kazusa_ai_chatbot\coding_agent tests docs development_plans\reference\designs\coding_agent_architecture.md
$stalePhase7Name = 'test_coding_agent_' + 'phase7_existing_source_planning_contracts'
rg -n $stalePhase7Name development_plans src tests
```

Expected results:

- The first grep must show only the canonical repair-feedback source and its
  tests/docs.
- The second grep must show the public export, verifier implementation, tests,
  and docs.
- The third grep must return no matches. A nonzero `rg` exit code is expected
  when no stale filename remains.

Focused deterministic commands:

```powershell
venv\Scripts\python -m pytest tests\test_coding_agent_phase5_patch_apply_contracts.py -q
venv\Scripts\python -m pytest tests\test_coding_agent_phase6_code_executing_contracts.py -q
venv\Scripts\python -m pytest tests\test_coding_agent_existing_source_planning_contracts.py -q
```

New deterministic tests added by this plan:

```powershell
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_verify_repair_contracts.py -q
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_interface.py -q
```

Stage 7A readiness command:

```powershell
venv\Scripts\python -m pytest tests\test_coding_agent_phase8_verify_repair_contracts.py::test_verify_repair_feedback_preserves_failure_summary_for_repair tests\test_coding_agent_phase8_verify_repair_contracts.py::test_verify_repair_feedback_redacts_absolute_paths_and_raw_output tests\test_coding_agent_phase8_verify_repair_contracts.py::test_verify_repair_request_includes_required_owner_and_context_paths tests\test_coding_agent_phase8_verify_repair_contracts.py::test_verify_repair_rejects_repair_omitting_required_owner_paths tests\test_coding_agent_phase8_verify_repair_contracts.py::test_verify_repair_rejects_repair_modifying_protected_verification_tests -q
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

After each live command, author a matching review artifact from the raw trace
before accepting the gate:

```text
test_artifacts/llm_traces/coding_agent_verify_repair/<gate_id>_review.md
```

The review artifact must include run context, evaluation goal, input summary,
output summary, repair behavior, validation results, quality assessment, and
the raw evidence path.

Final non-live regression:

```powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
```

## Independent Plan Review

Review date: 2026-07-08.

| Finding | Resolution |
|---|---|
| The draft lacked final-plan contract sections for `Must Do`, `Deferred`, cutover, change surface, overdesign guardrail, autonomy boundaries, execution model, progress checklist, LLM budget, static greps, and acceptance criteria. | Added the missing sections and mapped each required behavior to implementation and verification gates. |
| The Phase 7 regression command referenced a stale test filename that is not present in the current repository. | Updated the command to `tests\test_coding_agent_existing_source_planning_contracts.py`. |
| Stage 7A required deterministic readiness checks but did not name the exact test cases or focused command. | Added the five required readiness test names and a focused Stage 7A command. |
| The optional seeded `initial_patch_artifacts` path did not explicitly state that supplied artifacts still require normal review/apply validation. | Added validation rules that reject supplied artifacts with missing or mismatched source identity or failed patch review validation. |
| Live gate trace review required human judgment, but the plan did not name the durable human-readable review artifact. | Added per-gate Markdown review artifact requirements under `test_artifacts/llm_traces/coding_agent_verify_repair/`. |

Approval status: no blockers remain from this review. The user explicitly
approved fallback execution without subagents on 2026-07-08.

## Independent Code Review

Before this plan can move to completed:

- run an independent code-review pass over changed production code, tests,
  prompts, and docs;
- lead with findings by severity and file/line reference;
- remediate accepted findings;
- rerun affected tests;
- record review outcome in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `verify_and_repair_code_change(...)` is exported from
  `kazusa_ai_chatbot.coding_agent`.
- The verifier rejects missing approval, missing source identity, unsupported
  execution specs, unsafe supplied initial artifacts, and protected test edits.
- The verifier applies every attempt into a fresh managed apply workspace and
  never mutates the original source checkout.
- Execution repair feedback is structured, bounded, redacted, and accepted by
  `code_modifying` only through the canonical `execution_verification` source.
- Deterministic Phase 8 tests, Phase 5/6/7 regressions, and static greps pass.
  Full non-live regression either passes or has failures classified outside the
  Phase 8 change surface with focused Phase 8 reruns passing.
- Six live LLM gates pass one at a time with raw evidence and human-authored
  Markdown review artifacts committed for future regression review.
- Documentation and architecture references describe the implemented direct
  trusted verify/repair boundary.
- Independent code review finds no unresolved blockers.

## Execution Evidence

Status: completed.

Record implementation commands, test results, live LLM gate notes, trace
review decisions, independent review findings, remediations, and final
closeout evidence here during approved execution.

- 2026-07-08: User approved fallback execution without subagents. Plan status
  changed to `in_progress`.
- 2026-07-08: Baseline `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase5_patch_apply_contracts.py
  tests\test_coding_agent_phase6_code_executing_contracts.py -q` passed:
  19 passed.
- 2026-07-08: Added deterministic Phase 8 interface and verify/repair
  contract tests. Expected pre-implementation run failed for missing
  `verify_and_repair_code_change(...)` and missing `code_verifying`.
- 2026-07-08: Implemented `code_verifying` models, supervisor, README, and
  public exports; wired structured `execution_verification` feedback into
  `code_modifying` and top-level modifying requests.
- 2026-07-08: `venv\Scripts\python -m py_compile` passed for modified coding
  agent modules and new Phase 8 tests.
- 2026-07-08: `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase8_verify_repair_contracts.py -q` passed:
  11 passed.
- 2026-07-08: `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase8_interface.py -q` passed: 2 passed.
- 2026-07-08: Stage 7A readiness command passed: 5 passed.
- 2026-07-08: Focused regressions passed individually:
  Phase 5 patch apply 7 passed, Phase 6 code execution 12 passed, Phase 7
  existing-source planning 11 passed.
- 2026-07-08: Static greps passed. `execution_verification` appears only in
  the canonical production/docs/tests locations; `verify_and_repair_code_change`
  appears in public export, implementation, tests, and docs; the stale Phase 7
  existing-source planning test filename grep returned no matches as expected.
- 2026-07-08: Updated coding-agent ICDs, HOWTO, and architecture reference for
  the direct trusted verify/repair boundary.
- 2026-07-08: Final non-live regression command
  `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q` timed
  out at both 120 seconds and 600 seconds. Collection succeeded with
  2948/3502 selected tests, 554 deselected.
- 2026-07-08: Live LLM Gate 01 command timed out after 600 seconds before raw
  evidence was created. The timed pytest and fake-service child processes from
  this execution were identified by command line and stopped explicitly.
- 2026-07-08: After the user reported the slow issue fixed, live LLM Gate 01
  first failed on repair handoff ownership: the PM/programmer targeted the
  protected verification test instead of `stats_tools.py`. Added deterministic
  execution-repair handoff validation so required source-owner paths are
  writable and protected verification paths are read-only evidence. Rerun
  passed: `verify_repair_gate_01_median_boundary`, raw evidence
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_01_median_boundary_raw_evidence.json`,
  review artifact
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_01_median_boundary_review.md`.
- 2026-07-08: Live LLM Gate 02 first failed because execution repair was too
  strict and rejected the valid caller source target `wordcount/cli.py`.
  Allowed caller/source collaborator targets during execution repair while
  still requiring source-owner paths and protecting verification tests. Rerun
  passed: `verify_repair_gate_02_cli_flag_handoff`, raw evidence
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_02_cli_flag_handoff_raw_evidence.json`,
  review artifact
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_02_cli_flag_handoff_review.md`.
- 2026-07-08: Live LLM Gate 03 initially failed before LLM execution because
  the seeded patch was identical to the fixture source. Corrected the fixture
  so the seed patch fixes punctuation normalization while leaving duplicate
  suffixing for repair. Rerun passed:
  `verify_repair_gate_03_duplicate_anchor_parser`, raw evidence
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_03_duplicate_anchor_parser_raw_evidence.json`,
  review artifact
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_03_duplicate_anchor_parser_review.md`.
- 2026-07-08: Live LLM Gate 04 passed:
  `verify_repair_gate_04_soft_delete_cross_layer`, raw evidence
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_04_soft_delete_cross_layer_raw_evidence.json`,
  review artifact
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_04_soft_delete_cross_layer_review.md`.
- 2026-07-08: Live LLM Gate 05 surfaced two additional handoff gaps:
  invalid README/doc target recovery lacked an explicit allowed source target
  list, and execution repair incorrectly required a non-protected companion
  doc target. Added `allowed_source_target_paths` feedback and removed the
  companion target requirement for execution repair. Rerun passed after two
  repairs: `verify_repair_gate_05_fetch_cache_cli`, raw evidence
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_05_fetch_cache_cli_raw_evidence.json`,
  review artifact
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_05_fetch_cache_cli_review.md`.
- 2026-07-08: Live LLM Gate 06 passed:
  `verify_repair_gate_06_release_feed_cache_cli`, raw evidence
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_06_release_feed_cache_cli_raw_evidence.json`,
  review artifact
  `test_artifacts/llm_traces/coding_agent_verify_repair/verify_repair_gate_06_release_feed_cache_cli_review.md`.
- 2026-07-08: Focused closure regression command passed:
  `venv\Scripts\python -m pytest tests\test_coding_agent_phase8_interface.py
  tests\test_coding_agent_phase8_verify_repair_contracts.py
  tests\test_coding_agent_existing_source_planning_contracts.py
  tests\test_coding_agent_phase6_code_executing_contracts.py
  tests\test_coding_agent_phase5_patch_apply_contracts.py -q`: 46 passed.
- 2026-07-08: Independent code review found two Phase 8 issues. First,
  accepted repair proposals did not recompute the required source-owner path
  list for later repair attempts, so a later repair could theoretically drop a
  source path introduced by the previous repair. Second, `python_compileall`
  execution paths were incorrectly treated as protected verification paths even
  though they are executable source targets, not tests. Fixed both issues in
  `code_verifying.supervisor` and added deterministic tests:
  `test_verify_repair_updates_required_paths_after_each_repair` and
  `test_verify_repair_compileall_paths_are_not_protected_targets`.
- 2026-07-08: Post-review Phase 8 contract command passed:
  `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase8_verify_repair_contracts.py -q`: 16 passed.
- 2026-07-08: Post-review focused closure regression command passed:
  `venv\Scripts\python -m pytest tests\test_coding_agent_phase8_interface.py
  tests\test_coding_agent_phase8_verify_repair_contracts.py
  tests\test_coding_agent_existing_source_planning_contracts.py
  tests\test_coding_agent_phase6_code_executing_contracts.py
  tests\test_coding_agent_phase5_patch_apply_contracts.py -q`: 48 passed.
- 2026-07-08: Post-review Gate 05 live LLM rerun passed on current code:
  `verify_repair_gate_05_fetch_cache_cli`, 3 attempts, final changed files
  `inventory_sync/fetch.py` and `inventory_sync/cli.py`, final focused pytest
  3 passed, original source tree unchanged.
- 2026-07-08: Final non-live regression completed:
  `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q`.
  Result: 2945 passed, 2 skipped, 554 deselected, 4 failed in 415.33 seconds.
  Focused rerun of the four failures showed the coding-agent image-reading
  failure passed on retry; the remaining failures are outside this Phase 8
  change surface:
  `tests/test_control_console_config_routes.py::test_brain_model_route_api_applies_and_resets_selected_route`
  expects 13 model routes but the current runtime exposes 14, and
  `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py::test_text_chat_prompt_fingerprints_remain_stable`
  plus
  `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py::test_existing_l1_l2_l3_prompt_bytes_are_unchanged`
  fail on `_COGNITION_SUBCONSCIOUS_PROMPT` byte fingerprint drift. These files
  are not part of the Phase 8 coding-agent change surface and were not
  modified by this plan.
- 2026-07-08: Static greps rerun after review fixes. `execution_verification`
  and `verify_and_repair_code_change(...)` appear in expected code, docs, plan,
  and tests; stale Phase 7 filename grep returned no matches.
