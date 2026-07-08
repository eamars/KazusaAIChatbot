# coding agent phase6 code executing plan

## Summary

- Goal: implement Phase 6 of the coding-agent architecture: bounded
  verification execution for approved managed apply workspaces through a
  deterministic `code_executing` boundary.
- Plan class: high_risk_migration.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, and `test-style-and-execution`.
- Overall cutover strategy: bigbang for the new execution boundary; existing
  reading, writing, modifying, patching, and apply flows remain non-executing
  unless a trusted caller invokes the new structured execution API.
- Highest-risk areas: command allowlisting, workspace containment, timeout and
  output caps, environment isolation, execution-result sanitization, and
  preventing package installation or arbitrary shell access.
- Acceptance criteria: deterministic executor contract tests pass, the five
  precommitted live LLM execution gates pass one at a time with review
  evidence, integration tests prove execution only runs inside a Phase 5
  managed apply workspace, public metadata remains sanitized, docs reflect the
  boundary, and independent code review accepts the implementation.

## Context

Phase 5 introduces a managed apply workspace for approved patch artifacts.
Phase 6 uses that managed apply workspace as the only execution target. The
coding agent still must not run commands against original source checkouts,
managed clones, raw download stores, inline bundles, caller workspaces, or the
Kazusa source tree.

The reference architecture defines `code_executing` as the subagent that can
run bounded sandbox execution or delegate to Docker when local sandboxing is
unavailable. This plan implements the first controlled execution boundary for
verification commands. It does not add dependency installation, arbitrary shell
access, network-enabled execution, or LLM-driven command generation.

## Mandatory Skills

- `development-plan`: load before reading, approving, executing, reviewing, or
  closing this plan.
- `local-llm-architecture`: load before changing coding-agent supervisor
  routing, prompt surfaces, or background LLM behavior.
- `no-prepost-user-input`: load before changing accepted command handling.
  Execution requests must use structured execution specs, not deterministic
  keyword extraction from user prose.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not infer commands from raw user text, accepted-task prose, route reason,
  `answer_text`, or LLM-generated explanation.
- Do not accept free-form shell strings. Execution requests must use a closed
  structured command spec that deterministic code maps to argv lists.
- Do not use `shell=True`, `cmd /c`, PowerShell script text, shell pipes,
  shell redirection, glob expansion, or command chaining.
- Do not run commands outside a Phase 5 managed apply workspace.
- Do not run package managers, dependency installers, network tools, deploy
  commands, database commands, adapter sends, or mutation commands outside the
  managed apply workspace.
- Do not pass raw stdout, stderr, absolute paths, environment values, cache
  keys, tokens, `.env` contents, or full source dumps into public metadata.
- Do not add an LLM command generator in this phase. The only allowed command
  choice is deterministic mapping from structured execution specs.
- Do not add retry loops around failed target-project commands.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- Execution uses parent-led native subagent execution unless the user
  explicitly approves fallback execution.

## Must Do

- Create `src/kazusa_ai_chatbot/coding_agent/code_executing/` with README,
  models, deterministic validator, and runner boundary.
- Add a public direct execution API for trusted structured execution requests.
- Require a Phase 5 managed apply workspace identity before execution.
- Support a small allowlist of verification tools:
  `python_compileall` and `pytest`.
- Build argv lists deterministically from structured specs.
- Enforce cwd containment, path containment, timeout caps, stdout/stderr caps,
  file-count caps, and environment sanitization.
- Return structured execution results with bounded output summaries.
- Update coding-agent README, HOWTO, architecture reference, and worker docs to
  describe Phase 6 boundaries.
- Add deterministic tests for allowlist rejection, cwd containment, path
  containment, timeout handling, output capping, nonzero exit handling,
  successful compile, successful pytest, and public metadata sanitization.
- Preserve and satisfy the precommitted live LLM execution gates in
  `tests/test_coding_agent_phase6_code_executing_live_llm.py`; do not weaken
  their gate metadata, hard assertions, acceptable execution statuses, or trace
  requirements during implementation.

## Deferred

- Do not support arbitrary commands.
- Do not support package installation, dependency resolution, virtualenv
  creation, Docker image building, network access, deployment, database access,
  adapter sends, or repository push operations.
- Do not execute against original source checkouts or managed clone roots.
- Do not add LLM-generated command planning.
- Do not add automatic repair loops from execution failures back into
  `code_modifying` or `code_writing`.
- Do not add background-worker auto-execution from accepted-task prose.
- Do not persist execution results to MongoDB in this phase.
- Do not add cross-language execution beyond the listed Python verification
  tools.

## Cutover Policy

Overall strategy: bigbang for the new execution boundary.

| Area | Policy | Instruction |
|---|---|---|
| Execution API | bigbang | Add one trusted structured API; no free-form command API. |
| Command selection | bigbang | Use closed deterministic specs only. |
| Workspace target | bigbang | Require Phase 5 managed apply workspace identity. |
| Existing proposal/apply flows | compatible | Preserve existing non-executing behavior unless the new execution API is explicitly invoked. |
| Background worker | compatible | Keep generic coding background tasks non-executing. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a broader command strategy by default.
- If an area is `bigbang`, reject legacy or free-form shapes instead of adding
  compatibility shims.
- If an area is `compatible`, preserve only the compatibility surfaces listed
  in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

The coding agent exposes a controlled execution path:

```text
Phase 5 managed apply workspace
-> trusted structured execution request
-> code_executing.run(...)
-> deterministic command spec validation
-> bounded subprocess execution in managed apply workspace
-> sanitized CodingExecutionResponse
```

The execution result can be inspected by direct callers and later used as
evidence for a future repair phase. It is not automatically fed back into an
LLM repair loop in this plan.

The executor resolves the managed source directory deterministically from:

```text
<workspace_root>/patch_apply/<apply_package_id>/source
```

It must not accept an absolute apply-directory path from public request data.
The trusted public handoff is the Phase 5 `apply_workspace_ref` plus
`apply_package_id`.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Target workspace | Execute only inside Phase 5 managed apply workspaces | Prevents accidental mutation or command execution against original sources. |
| Command shape | Closed structured specs | Removes shell parsing and keyword command inference. |
| Initial tools | `python_compileall` and `pytest` only | Covers useful verification while keeping the trust boundary small. |
| Runner | Local subprocess with argv list and hard caps | Matches current Python test environment and remains deterministic. |
| Docker | Defer Docker-backed runner | Requires separate host runtime contract and image policy. |
| Repair | Defer automatic repair loops | Execution evidence must first be bounded and inspectable. |

## Contracts And Data Shapes

Create:

```python
from kazusa_ai_chatbot.coding_agent.code_executing import run
from kazusa_ai_chatbot.coding_agent import execute_code_check
```

`CodeExecutionRequest`:

```python
{
    "workspace_root": str,
    "apply_package_id": str,
    "apply_workspace_ref": {
        "kind": "managed_apply_workspace",
        "source_identity": dict,
        "applied_files": list[str],
    },
    "execution": {
        "tool": "python_compileall | pytest",
        "paths": list[str],
        "pytest_selectors": list[str],
        "timeout_seconds": int,
    },
    "max_stdout_chars": int,
    "max_stderr_chars": int,
}
```

`CodeExecutionResponse`:

```python
{
    "status": "succeeded | failed | rejected | timed_out",
    "tool": str,
    "exit_code": int | None,
    "timed_out": bool,
    "duration_ms": int,
    "stdout_excerpt": str,
    "stderr_excerpt": str,
    "output_truncated": bool,
    "executed_paths": list[str],
    "limitations": list[str],
    "trace_summary": list[str],
}
```

Command mapping:

- `python_compileall` maps to
  `[sys.executable, "-m", "compileall", *safe_paths]`.
- `pytest` maps to
  `[sys.executable, "-m", "pytest", *safe_selectors, "-q"]`.

Executor success is separate from target-project success. For `pytest`, a
nonzero target test exit is a valid structured `failed` execution response,
not an unhandled executor error. The executor must not report `succeeded` when
the underlying allowed command exits nonzero.

Rejection conditions:

- missing or invalid managed apply workspace identity;
- any absolute path, `..`, `.env`, `.git`, secret-like path, binary-only path,
  or path outside the apply workspace;
- unsupported tool name;
- empty execution target;
- timeout above the plan cap;
- output cap above the plan cap;
- any request requiring package installation, network access, shell syntax, or
  real repository mutation.

## LLM Call And Context Budget

This plan adds no new LLM calls and changes no prompt contract. Execution
requests are trusted structured data. No LLM receives raw stdout/stderr in this
phase. Future repair plans may feed bounded execution summaries into a PM, but
that is explicitly outside this plan.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/coding_agent/code_executing/README.md`: execution ICD.
- `src/kazusa_ai_chatbot/coding_agent/code_executing/__init__.py`: public
  subagent export.
- `src/kazusa_ai_chatbot/coding_agent/code_executing/models.py`: typed request
  and response shapes.
- `src/kazusa_ai_chatbot/coding_agent/code_executing/supervisor.py`: validator
  and runner orchestration.
- `src/kazusa_ai_chatbot/coding_agent/code_executing/runner.py`: argv-based
  bounded subprocess runner.
- `tests/test_coding_agent_phase6_code_executing_contracts.py`: focused
  executor contract tests.
- `tests/test_coding_agent_phase6_interface.py`: direct API and integration
  tests against a managed apply workspace fixture.
- `tests/test_coding_agent_phase6_code_executing_live_llm.py`: five
  precommitted real LLM execution gates with objective, existing code base,
  modification instruction, execution spec, expected state, pass criteria,
  behavior rubric, forbidden failure modes, and durable raw evidence trace
  requirements.

### Modify

- `src/kazusa_ai_chatbot/coding_agent/__init__.py`: export
  `execute_code_check` and execution models.
- `src/kazusa_ai_chatbot/coding_agent/models.py`: add public response/request
  aliases only when top-level export requires them.
- `src/kazusa_ai_chatbot/coding_agent/README.md`: document Phase 6 direct API
  and non-background execution boundary.
- `docs/HOWTO.md`: document allowed execution tools and safety limits.
- `development_plans/reference/designs/coding_agent_architecture.md`: update
  implemented Phase 6 boundary after implementation.

### Keep

- `handle_background_coding_task(...)` continues to reject ordinary tasks that
  require live execution from prose.
- `code_patching` and Phase 5 apply do not run tests or commands.
- `code_writing` and `code_modifying` do not consume execution results.

## Overdesign Guardrail

- Actual problem: the coding agent cannot run bounded verification against an
  approved applied workspace.
- Minimal change: add one deterministic `code_executing` boundary that maps
  trusted structured specs to allowlisted Python verification argv lists.
- Ownership boundaries: deterministic code owns command validation, execution,
  timeouts, output caps, and sanitization; LLM stages own no command selection
  or execution permission.
- Rejected complexity: arbitrary shell, package installation, Docker runner,
  network isolation, repair loops, background auto-execution, multi-language
  execution, persistent execution storage, and direct original-checkout
  execution.
- Evidence threshold: Docker, repair loops, or broader command families may be
  planned only after this narrow executor passes deterministic safety tests and
  a separate isolation contract is approved.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper names only when they preserve
  this plan's contracts.
- The responsible agent must not introduce alternate command APIs,
  compatibility shims, fallback paths, or extra features.
- The responsible agent must treat changes outside `coding_agent`, docs, and
  tests as out of scope.
- The responsible agent must search for existing path and subprocess helpers
  before adding new helpers.
- The responsible agent must not perform unrelated cleanup, dependency
  upgrades, broad refactors, or formatting churn.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Confirm Phase 5 is completed and the managed apply workspace contract is
   present. If Phase 5 is still draft, stop before implementation.
2. Preserve the precommitted live LLM gate file
   `tests/test_coding_agent_phase6_code_executing_live_llm.py`. These gates are
   the signoff contract and must not be relaxed during implementation.
3. Add focused failing tests in
   `tests/test_coding_agent_phase6_code_executing_contracts.py` for unsupported
   tools, free-form command rejection, path traversal rejection, missing apply
   workspace rejection, timeout handling, output capping, nonzero exit result,
   successful compileall, and successful pytest.
4. Add focused failing tests in `tests/test_coding_agent_phase6_interface.py`
   for direct top-level export and execution against a Phase 5 managed apply
   workspace fixture.
5. Implement `code_executing.models`, validator, and runner.
6. Export the direct public API.
7. Update README, HOWTO, and architecture reference.
8. Run focused executor tests, Phase 5 apply tests, Phase 4 patching tests,
   static greps, compile checks, and the five live LLM execution gates one
   case at a time.
9. Run independent code review and remediate findings inside this change
   surface.
10. Move the plan to completed archive only after verification and review
   evidence are recorded.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it.
- Parent agent may continue integration tests, regression tests, static
  checks, and validation while the production-code subagent edits production
  code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 0 - Phase 5 dependency confirmed.
  - Covers: implementation step 1.
  - Verify: Phase 5 plan is completed and `apply_approved_patch(...)` contract
    exists.
  - Evidence: record Phase 5 commit or archive path.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 1 - focused test contract established.
  - Covers: implementation steps 2, 3, and 4.
  - Verify: focused Phase 6 tests fail for missing API or missing behavior.
  - Evidence: record test commands and expected failures.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 2 - executor module implemented.
  - Covers: implementation steps 5 and 6.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_coding_agent_phase6_code_executing_contracts.py -q`.
  - Evidence: record changed production files and passing focused test output.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 3 - docs and integration updated.
  - Covers: implementation step 7.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_coding_agent_phase6_interface.py -q`.
  - Evidence: record integration output and documentation files changed.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 4 - regression verification complete.
  - Covers: implementation step 8.
  - Verify: all commands in `Verification`.
  - Evidence: record command outputs and accepted static grep results.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 5 - independent code review complete.
  - Covers: implementation step 9.
  - Verify: review findings are recorded, fixes are applied, and affected
    verification commands pass again.
  - Evidence: record review result, remediation, residual risk, and approval.
  - Sign-off: `Codex/2026-07-08`.
- [x] Stage 6 - lifecycle closure complete.
  - Covers: implementation step 10.
  - Verify: plan status and registry are updated, active plan moved to
    completed archive, and `git status --short` is reviewed.
  - Evidence: record final commit or handoff state.
  - Sign-off: `Codex/2026-07-08`.

## Live LLM Gating Tests

The five live gates are committed ahead of implementation in
`tests/test_coding_agent_phase6_code_executing_live_llm.py`. Each gate produces
a real LLM patch proposal through `propose_code_change(...)`, applies approved
patch artifacts through `apply_approved_patch(...)`, executes a bounded
structured verification request through `execute_code_check(...)`, writes
durable raw evidence to
`test_artifacts/llm_traces/coding_agent_phase6_code_executing/`, and requires
one-at-a-time human or AI review of the raw evidence.

### Gate 01 - CLI JSON compile

- Objective: prove a live LLM patch for a small CLI change can be applied and
  verified through the bounded `python_compileall` tool inside the managed
  apply workspace.
- Existing code base: `gate_01_log_counter`, a standard-library
  `log_counter.py` CLI with text output and focused tests.
- Modification instruction: add `--json`, emit valid JSON with severity and
  skipped-line counts, preserve default text output, and update focused tests.
- Execution spec: `python_compileall` on `log_counter.py`, timeout 15 seconds.
- Expected state: proposal and apply succeed, compileall uses argv inside the
  managed apply workspace, execution status is `succeeded`, and public output
  is bounded.
- Pass criteria: execution reports `tool = python_compileall`, status
  `succeeded`, relative executed path `log_counter.py`, no absolute paths, and
  unchanged original source hashes.

### Gate 02 - JSONL focused pytest

- Objective: prove bounded pytest execution for a live LLM multi-file utility
  patch, including structured reporting when target tests pass or fail.
- Existing code base: `gate_02_contacts_jsonl_to_csv`, a package with
  converter, CLI, README, converter tests, and CLI tests.
- Modification instruction: make `--fields` exact-order, write blanks for
  missing fields, report malformed JSON with 1-based line numbers, continue by
  default, fail fast with `--strict`, and update tests/docs.
- Execution spec: `pytest` selectors `tests/test_converter.py` and
  `tests/test_cli.py`, timeout 30 seconds.
- Expected state: executor runs pytest only inside the managed apply workspace
  and returns `succeeded` or `failed` according to the target test exit.
- Pass criteria: proposal/apply succeed, execution status is `succeeded` or
  `failed`, exit code is represented, selectors are relative, and output is
  bounded and sanitized.

### Gate 03 - Markdown parser focused pytest

- Objective: prove bounded pytest execution for a parser-owned live LLM patch
  where scanner and anchor tests exercise edge cases.
- Existing code base: `gate_03_markdown_link_checker`, a package with
  `anchors.py`, `scanner.py`, CLI, README, and anchor/scanner tests.
- Modification instruction: ignore links inside fenced code and HTML comments,
  support duplicate heading suffixes `base`, `base-1`, `base-2`, update
  parser/scanner tests, and preserve normal links.
- Execution spec: `pytest` selectors `tests/test_anchors.py` and
  `tests/test_scanner.py`, timeout 30 seconds.
- Expected state: executor reports parser test success or failure
  structurally, with unchanged original source and sanitized output.
- Pass criteria: proposal/apply succeed, execution status is `succeeded` or
  `failed`, executed paths include both selectors, no shell or package manager
  is used, and original source hashes match.

### Gate 04 - Issue tracker focused pytest

- Objective: prove bounded pytest execution for a cross-layer live LLM patch
  spanning model, store, API, tests, and docs.
- Existing code base: `gate_04_issue_tracker_soft_delete`, a package with
  model, store, API, README, and store/API tests.
- Modification instruction: mark deleted issues archived, hide archived issues
  from normal lookup/list, add `include_archived`, update tests/docs, and avoid
  wrappers around old hard-delete semantics.
- Execution spec: `pytest` selectors `tests/test_store.py` and
  `tests/test_api.py`, timeout 30 seconds.
- Expected state: executor runs only against the managed applied issue tracker
  copy and reports target test result without repair loops.
- Pass criteria: proposal/apply succeed, execution status is `succeeded` or
  `failed`, exit code is represented, selectors are relative, and public
  metadata omits absolute paths.

### Gate 05 - Inventory package compile

- Objective: prove bounded compile execution for a hard live LLM patch that may
  introduce a helper file while preserving the existing package layout.
- Existing code base: `gate_05_inventory_sync_fetch_cache`, a package with CSV
  reading, urllib fetch, HTML extraction, report writing, CLI, README, and
  mocked HTTP tests.
- Modification instruction: add timeout/retry fetch behavior, file-backed
  cache, CLI flags `--cache-dir`, `--refresh-cache`, and `--timeout`, mocked
  HTTP tests, and README workflow documentation.
- Execution spec: `python_compileall` on `inventory_sync`, timeout 20 seconds.
- Expected state: proposal and apply succeed, compileall runs against the
  managed applied package, execution status is `succeeded`, and output is
  sanitized.
- Pass criteria: execution reports `tool = python_compileall`, status
  `succeeded`, relative executed path `inventory_sync`, no package manager or
  network command appears, and original source hashes match.

## Verification

### Static Checks

- `venv\Scripts\python -m compileall src\kazusa_ai_chatbot\coding_agent tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase6_interface.py tests\test_coding_agent_phase6_code_executing_live_llm.py`
  must pass.
- `git diff --check` must report no whitespace errors.
- `rg "shell=True|cmd /c|powershell|Start-Process|pip install|npm install|curl|Invoke-WebRequest" src\kazusa_ai_chatbot\coding_agent\code_executing -n`
  must return no matches.
- `rg "subprocess\\.(run|Popen)" src\kazusa_ai_chatbot\coding_agent -n`
  may match only the approved deterministic runner and existing bounded
  coding-agent tool facades: `code_executing/runner.py`, `tools/git.py`,
  `code_patching/patch_validation.py`, and `code_reading/evidence.py`.
- `rg "code_executing" src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\action_spec src\kazusa_ai_chatbot\background_work -n`
  must return no matches unless this plan is explicitly updated and approved
  for background execution integration.

### Tests

- `venv\Scripts\python -m pytest tests\test_coding_agent_phase6_code_executing_contracts.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_phase6_interface.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_phase5_patch_apply_contracts.py tests\test_coding_agent_phase4_code_patching_contracts.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_phase2_new_artifact_contracts.py tests\test_coding_agent_phase4_interface.py -q`

### Live LLM Tests

Run one case at a time with `-s`, inspect the emitted raw evidence, and record
the quality judgment before running the next case:

- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_phase6_code_executing_live_llm.py::test_phase6_live_gate_01_cli_json_compile -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_phase6_code_executing_live_llm.py::test_phase6_live_gate_02_jsonl_pytest -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_phase6_code_executing_live_llm.py::test_phase6_live_gate_03_markdown_pytest -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_phase6_code_executing_live_llm.py::test_phase6_live_gate_04_issue_tracker_pytest -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_phase6_code_executing_live_llm.py::test_phase6_live_gate_05_inventory_compile -q -s`

### Manual Review

- Inspect one successful execution response and confirm stdout/stderr are
  bounded and contain no absolute workspace path.
- Confirm the applied workspace remains inside the Phase 5 managed apply root.
- For every live LLM gate, inspect the raw evidence JSON and judge the
  behavior rubric plus forbidden failure modes before signoff.

## Confidence Assessment

Confidence to pass the five live LLM execution gates after Phase 5 is completed
and this plan is implemented as written: 90%.

Rationale:

- The live gates test the execution boundary with realistic LLM-generated patch
  outputs, not synthetic no-op files.
- The hard pass criteria are aligned with Phase 6 ownership: workspace
  containment, allowlisted command mapping, structured status, nonzero exit
  handling, output caps, sanitization, and original source immutability.
- Two gates require `python_compileall` to succeed, which proves useful
  successful-command behavior without requiring network or dependency setup.
- Three gates run focused pytest selectors and accept either `succeeded` or
  `failed` execution status because target-test success is patch-quality
  evidence, while Phase 6 owns truthful bounded execution reporting.
- The plan explicitly rejects repair loops and LLM command generation, keeping
  local-model and deterministic ownership boundaries inspectable.

Residual risk:

- If the intended product signoff is redefined to require every live
  LLM-generated patch to pass its target pytest suite, confidence is below 90%
  because this plan intentionally excludes repair loops. That stronger
  objective belongs to a later repair-orchestration plan.
- Phase 6 depends on the Phase 5 managed apply workspace contract. The plan
  remains blocked until Phase 5 is completed and `apply_workspace_ref` exists.

## Independent Plan Review

Fresh review inputs:

- `development_plans/reference/designs/coding_agent_architecture.md`
- `development_plans/active/short_term/coding_agent_phase5_patch_apply_plan.md`
- `src/kazusa_ai_chatbot/coding_agent/README.md`
- `src/kazusa_ai_chatbot/coding_agent/code_patching/README.md`
- `tests/test_coding_agent_phase6_code_executing_live_llm.py`

Review findings and remediation:

- Blocker: the original Phase 6 draft had no precommitted real LLM signoff
  tests under `tests/`. Resolved by adding
  `tests/test_coding_agent_phase6_code_executing_live_llm.py` with five
  populated execution gates and durable trace requirements.
- Blocker: the original Phase 6 contract depended on a Phase 5 workspace but
  Phase 5 did not expose a handoff reference. Resolved by updating the Phase 5
  plan to require `apply_workspace_ref` and by specifying deterministic
  executor resolution from `workspace_root`, `apply_package_id`, and that ref.
- Blocker: execution pass/fail semantics could be confused with generated
  patch quality. Resolved by explicitly defining nonzero pytest exit as a
  structured `failed` execution response and by making compile gates the
  successful-command hard checks.
- Blocker: live gate verification was absent from the command list. Resolved by
  adding exact one-at-a-time live LLM commands and manual evidence review.
- Non-blocking finding: Phase 6 cannot be implemented before Phase 5 is
  complete. Accepted and retained as Stage 0 dependency.

Approval status: draft strengthened for user review. It is not executable until
the user approves or commands implementation after Phase 5 completion.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- No command is inferred from prose or LLM text.
- No free-form shell string enters subprocess execution.
- Every execution target is contained inside a Phase 5 managed apply workspace.
- Unsupported tools, package managers, network commands, and path traversal
  fail closed.
- Timeouts and output caps are enforced.
- Public metadata is sanitized.
- Tests cover success, failure, rejection, timeout, output cap, and regression
  paths.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `code_executing.run(...)` and `execute_code_check(...)` exist.
- Execution accepts only structured specs for `python_compileall` and `pytest`.
- Execution runs only inside Phase 5 managed apply workspaces.
- Unsupported commands, free-form command strings, unsafe paths, over-cap
  requests, and missing apply workspace identity fail closed.
- Timeout, nonzero exit, and output truncation are represented in structured
  execution results.
- Public execution responses are sanitized.
- Coding-agent docs and architecture reference describe Phase 6 accurately.
- All verification commands pass.
- Independent code review accepts the implementation.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Arbitrary shell execution | Closed structured specs and argv lists only | Static grep and rejection tests |
| Workspace escape | Resolve every path inside managed apply root | Path traversal tests |
| Long-running commands | Hard timeout cap | Timeout test |
| Large or sensitive output | Output caps and path scrubbing | Output cap and sanitization tests |
| Scope creep into repair loops | Keep execution result handoff direct only | Static grep and review |

## Execution Evidence

- Phase 5 dependency confirmation: Phase 5 is completed in
  `development_plans/archive/completed/short_term/coding_agent_phase5_patch_apply_plan.md`;
  `apply_approved_patch(...)` and `apply_workspace_ref` are present in
  `src/kazusa_ai_chatbot/coding_agent/code_patching/apply.py`.
- Focused test baseline:
  `venv\Scripts\python -m pytest tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase6_interface.py -q`
  failed before implementation with missing `execute_code_check`,
  `CodeExecutionRequest`, and `CodeExecutionResponse` imports.
- Implementation files changed: added
  `src/kazusa_ai_chatbot/coding_agent/code_executing/models.py`,
  `src/kazusa_ai_chatbot/coding_agent/code_executing/runner.py`,
  `src/kazusa_ai_chatbot/coding_agent/code_executing/supervisor.py`,
  `src/kazusa_ai_chatbot/coding_agent/code_executing/__init__.py`, and
  exported `execute_code_check` from
  `src/kazusa_ai_chatbot/coding_agent/__init__.py`.
- Focused test pass:
  `venv\Scripts\python -m pytest tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase6_interface.py -q`
  passed with 13 passed.
- Integration and docs update: added
  `src/kazusa_ai_chatbot/coding_agent/code_executing/README.md` and updated
  `src/kazusa_ai_chatbot/coding_agent/README.md`, `docs/HOWTO.md`, and
  `development_plans/reference/designs/coding_agent_architecture.md`.
  `venv\Scripts\python -m pytest tests\test_coding_agent_phase6_interface.py -q`
  passed with 3 passed.
- Regression test pass:
  `venv\Scripts\python -m pytest tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase6_interface.py -q`
  passed with 15 passed after review remediation;
  `venv\Scripts\python -m pytest tests\test_coding_agent_phase5_patch_apply_contracts.py tests\test_coding_agent_phase4_code_patching_contracts.py -q`
  passed with 12 passed;
  `venv\Scripts\python -m pytest tests\test_coding_agent_phase2_new_artifact_contracts.py tests\test_coding_agent_phase4_interface.py -q`
  passed with 20 passed.
- Static check results:
  `venv\Scripts\python -m compileall src\kazusa_ai_chatbot\coding_agent tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase6_interface.py tests\test_coding_agent_phase6_code_executing_live_llm.py`
  passed; `git diff --check` passed with line-ending warnings only;
  forbidden execution grep under `code_executing` returned no matches; the
  `subprocess` grep matched only existing bounded coding-agent tool facades and
  the new `code_executing/runner.py`; background/action/dialog grep returned
  no `code_executing` matches.
- Manual review: all five live LLM gates were run one at a time with `-s` and
  raw evidence inspected under
  `test_artifacts/llm_traces/coding_agent_phase6_code_executing/`.
  Gate 01 succeeded through `python_compileall` on `log_counter.py`; Gate 02
  succeeded through focused `pytest` on `tests/test_converter.py` and
  `tests/test_cli.py`; Gate 03 succeeded through focused `pytest` on
  `tests/test_anchors.py` and `tests/test_scanner.py`; Gate 04 truthfully
  returned `failed` with exit code 1 for issue tracker tests; Gate 05 succeeded
  through `python_compileall` on `inventory_sync`. Every inspected public
  response omitted managed source/workspace roots and every original source
  tree hash comparison remained unchanged. A transient Gate 02 rerun failed
  before execution because the live proposal produced invalid test syntax; raw
  evidence showed execution correctly rejected the missing apply package, and
  the gate passed on rerun.
- Independent code review: user requested no-subagent execution, so Codex ran
  the review gate as the fallback execution owner. Findings: directory
  execution targets needed pre-run tree scanning for file-count cap and
  symlink escape containment; child process environment values needed public
  output redaction if target code printed them; the static `subprocess` grep
  expectation was stale because existing bounded `code_reading` and
  `code_patching` facades already use subprocess calls. Remediation: added
  directory tree scan and file-count cap in `code_executing/supervisor.py`,
  minimized and redacted execution environment values in
  `code_executing/runner.py`, added focused tests for environment redaction and
  directory file-count rejection, and updated the static grep expectation.
  Rerun evidence: Phase 6 focused/interface tests passed with 15 passed,
  compileall passed, forbidden `code_executing` grep returned no matches,
  background/action/dialog grep returned no matches, and prior Phase 2/4/5
  regressions remained green. Residual risk: live proposal quality still varies
  before execution, which is outside the Phase 6 executor boundary.
- Lifecycle closure: plan marked completed for archive move; registry update
  records Phase 6 completion; final commit will reference this completed
  implementation.
