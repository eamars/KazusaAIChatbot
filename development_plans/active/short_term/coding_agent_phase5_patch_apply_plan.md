# coding agent phase5 patch apply plan

## Summary

- Goal: implement Phase 5 of the coding-agent architecture: explicit approved
  patch application into a controlled managed apply workspace, without running
  commands, installing packages, or mutating the original source checkout.
- Plan class: high_risk_migration.
- Status: draft.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, and `test-style-and-execution`.
- Overall cutover strategy: bigbang for the new apply boundary; existing
  proposal paths remain review-only unless a trusted caller invokes the new
  apply API with structured approval data.
- Highest-risk areas: approval boundary, source identity checks, filesystem
  containment, public metadata sanitization, and preventing original checkout
  mutation.
- Acceptance criteria: deterministic apply contract tests pass, the five
  precommitted live LLM apply gates pass one at a time with review evidence,
  direct apply integration proves original source immutability, coding-agent
  docs reflect the new boundary, and independent code review accepts the
  implementation.

## Context

Phase 4 completed review-only patch proposals for source-free writing and
existing-source modification. Current `code_patching` can compile patch
operations and materialize review packages, but it deliberately stops before
applying the patch to a workspace. The coding-agent architecture lists Phase 5
as the point where approved patches can be applied in a controlled sandbox or
approved workspace.

The next narrow capability is not command execution. It is deterministic
application of already-produced patch artifacts into a managed apply copy so
the user or a later execution stage can inspect the actual patched file tree.
The original fetched checkout or caller workspace must remain unchanged.

One Phase 4 carryover defect must be fixed in this plan: the background worker
metadata currently maps `code_modifying` responses to `unsupported`. The
coding-agent supervisor already returns `code_modifying`; the background
adapter must preserve that operation in sanitized metadata.

## Mandatory Skills

- `development-plan`: load before reading, approving, executing, reviewing, or
  closing this plan.
- `local-llm-architecture`: load before changing coding-agent supervisor
  routing, prompt surfaces, or background LLM behavior.
- `no-prepost-user-input`: load before changing approval or accepted-command
  handling. Approval for patch application must be a trusted structured
  runtime input, not inferred from chat text by keyword logic.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not infer approval from raw user text, accepted-task prose, route reason,
  `answer_text`, or LLM-generated explanation. Patch application requires a
  structured approval object supplied by a trusted caller.
- Do not add keyword-based routing, approval detection, or semantic filtering
  over user text. LLM stages may decide whether a task is a coding task, but
  deterministic apply code accepts only explicit structured approval data.
- Do not mutate the original source checkout, caller workspace, Kazusa source
  tree, managed clone, or managed raw/inline source. Apply into a new managed
  apply workspace under the configured coding-agent workspace root.
- Do not run generated tests, target project tests, shell commands, package
  managers, formatters, or build tools in this phase.
- Do not add patch execution to L2d, dialog, generic background-work router, or
  adapter delivery. The coding-agent direct apply API owns this capability.
- Do not create compatibility shims for old patching paths. `code_patching`
  remains the canonical patch boundary.
- Public responses must not expose absolute source roots, workspace roots,
  cache keys, raw command output, raw traces, `.env` contents, secret-like
  paths, `.git` internals, or full source dumps.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- Execution uses parent-led native subagent execution unless the user
  explicitly approves fallback execution.

## Must Do

- Add a deterministic patch apply boundary under
  `src/kazusa_ai_chatbot/coding_agent/code_patching/`.
- Add a public direct coding-agent API for trusted patch apply requests.
- Require structured approval data before any apply workspace is created.
- Verify source identity before applying patch artifacts.
- Apply only into a managed apply workspace under the caller-provided
  `workspace_root`.
- Preserve the original source tree byte-for-byte during apply.
- Preserve `code_modifying` in background worker metadata.
- Add deterministic tests for approval rejection, path containment, source
  identity mismatch, apply conflict, sanitized output, original source
  immutability, and background `code_modifying` metadata mapping.
- Preserve and satisfy the precommitted live LLM apply gates in
  `tests/test_coding_agent_phase5_patch_apply_live_llm.py`; do not weaken
  their gate metadata, hard assertions, or trace requirements during
  implementation.
- Update coding-agent README, code-patching README, HOWTO, and the coding
  architecture reference.

## Deferred

- Do not apply patches directly to the original checkout.
- Do not support arbitrary user-selected target directories outside the
  configured coding-agent workspace.
- Do not run commands, tests, package installation, build tools, or formatters.
- Do not add `code_executing`; that belongs to Phase 6.
- Do not add Docker, network isolation, command allowlists, or execution
  result repair loops.
- Do not persist approval records to MongoDB in this phase.
- Do not add new L2d or dialog affordances for patch application.
- Do not make background jobs apply patches from accepted-task prose.

## Cutover Policy

Overall strategy: bigbang for the new apply boundary.

| Area | Policy | Instruction |
|---|---|---|
| Apply API | bigbang | Add one direct trusted API; no alternate helper path. |
| Proposal path | compatible | Existing `propose_code_change(...)` remains review-only. |
| Workspace mutation | bigbang | Apply only to a managed copy; never mutate original source. |
| Background metadata | bigbang | Preserve `code_modifying` instead of mapping it to `unsupported`. |
| Execution | bigbang | Keep all command/test execution unavailable in Phase 5. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative or broader strategy by default.
- If an area is `bigbang`, delete or rewrite stale references instead of
  preserving parallel behavior.
- If an area is `compatible`, preserve only the compatibility surfaces listed
  in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

The coding agent exposes an explicit trusted apply flow:

```text
previous patch proposal response
-> trusted structured approval object
-> apply_approved_patch(...)
-> source identity check
-> managed apply workspace copy
-> git apply check and apply inside managed copy
-> CodingPatchApplyResponse
```

The original source root remains unchanged. The result reports only public-safe
relative paths, source identity labels, apply status, changed file summaries,
managed apply package id, an opaque managed apply workspace reference, and
limitations. The response does not expose the managed apply directory path.

The deterministic internal layout is:

```text
<workspace_root>/patch_apply/<apply_package_id>/source
```

The public response may include `apply_package_id` and
`apply_workspace_ref.kind = "managed_apply_workspace"`, but it must not include
the absolute resolved path.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Apply location | Apply into a managed copy under `workspace_root/patch_apply/` | This proves apply mechanics while preserving source immutability. |
| Approval | Require structured `PatchApplyApproval` from trusted caller | Prevents keyword-based approval inference from chat text. |
| Public API | Add `apply_approved_patch(...)` to `coding_agent` | Mirrors existing direct interfaces and keeps background routing separate. |
| Source identity | Require expected source identity and compare before copy | Prevents stale patch application against a different checkout. |
| Patch artifacts | Consume existing `PatchArtifact` shape | Avoids a second patch vocabulary. |
| Execution | No execution in apply flow | Phase 6 owns command/test execution. |

## Contracts And Data Shapes

Create `src/kazusa_ai_chatbot/coding_agent/code_patching/apply.py` with:

```python
def apply_approved_patch(request: CodingPatchApplyRequest) -> CodingPatchApplyResponse:
    ...
```

Expose the public function from `kazusa_ai_chatbot.coding_agent` as:

```python
from kazusa_ai_chatbot.coding_agent import apply_approved_patch
```

`CodingPatchApplyRequest`:

```python
{
    "workspace_root": str,
    "source_root": str,
    "source_identity": {
        "provider": str,
        "owner": str | None,
        "repo": str | None,
        "current_commit": str,
        "dirty_state": "clean",
    },
    "expected_source_identity": {
        "provider": str,
        "owner": str | None,
        "repo": str | None,
        "current_commit": str,
        "dirty_state": "clean",
    },
    "patch_artifacts": list[PatchArtifact],
    "approval": {
        "approved": True,
        "approved_by": str,
        "approved_at": str,
        "approval_reason": str,
    },
    "max_files": int,
    "max_diff_chars": int,
}
```

`CodingPatchApplyResponse`:

```python
{
    "status": "succeeded | failed | rejected",
    "apply_package_id": str,
    "source_identity": dict,
    "apply_workspace_ref": {
        "kind": "managed_apply_workspace",
        "apply_package_id": str,
        "source_identity": dict,
        "applied_files": list[str],
    },
    "applied_files": list[str],
    "changed_files": list[dict[str, str]],
    "validation": {
        "status": "succeeded | failed | rejected",
        "errors": list[str],
        "warnings": list[str],
    },
    "limitations": list[str],
    "trace_summary": list[str],
}
```

Failure and refusal conditions:

- Missing approval or `approved is not True` returns `rejected`.
- Dirty or mismatched source identity returns `rejected`.
- Unsafe paths, over-cap diffs, missing files, malformed patches, or patch
  conflicts return `failed` or `rejected` according to the existing patching
  validation style.
- Any attempt to apply outside the managed apply workspace returns `rejected`.

## LLM Call And Context Budget

This plan adds no new LLM calls and changes no prompt contract. The only
LLM-adjacent change is background worker metadata preserving `code_modifying`
as a deterministic enum value. The normal background LLM router call count
remains unchanged.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/coding_agent/code_patching/apply.py`: deterministic
  managed apply boundary.
- `tests/test_coding_agent_phase5_patch_apply_contracts.py`: focused apply
  contract and safety tests.
- `tests/test_coding_agent_phase5_interface.py`: direct public API and
  background metadata integration tests.
- `tests/test_coding_agent_phase5_patch_apply_live_llm.py`: five
  precommitted real LLM apply gates with objective, existing code base,
  modification instruction, expected state, pass criteria, behavior rubric,
  forbidden failure modes, and durable raw evidence trace requirements.

### Modify

- `src/kazusa_ai_chatbot/coding_agent/__init__.py`: export
  `apply_approved_patch` and apply response/request models.
- `src/kazusa_ai_chatbot/coding_agent/models.py`: add public typed shapes when
  they belong at top-level; keep patch-internal shapes in `code_patching`.
- `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`: preserve
  `code_modifying` in worker metadata.
- `src/kazusa_ai_chatbot/coding_agent/code_patching/README.md`: document apply
  boundary and no-execution rule.
- `src/kazusa_ai_chatbot/coding_agent/README.md`: document Phase 5 public API.
- `docs/HOWTO.md`: document operator-level apply boundary.
- `development_plans/reference/designs/coding_agent_architecture.md`: update
  Phase 5 status and boundary wording after implementation.

### Keep

- `propose_code_change(...)` remains review-only.
- `handle_background_coding_task(...)` does not apply patches.
- `code_writing`, `code_modifying`, and `code_reading` do not gain execution
  authority.

## Overdesign Guardrail

- Actual problem: Phase 4 produces reviewable patch artifacts but cannot
  produce a patched managed file tree after explicit approval.
- Minimal change: add one deterministic apply API that applies existing patch
  artifacts into a managed copy only.
- Ownership boundaries: deterministic code owns approval shape validation,
  identity checks, copying, patch application, limits, and public metadata;
  LLM stages own no approval or apply decision.
- Rejected complexity: direct original-checkout mutation, arbitrary target
  directories, persistent approval storage, background auto-apply, execution,
  Docker, patch repair loops, adapter delivery, and prompt-based approval
  extraction.
- Evidence threshold: direct original-checkout mutation, persistent approval
  storage, or auto-apply may be planned only after managed-copy apply passes
  deterministic safety tests and a separate approval/permission design exists.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper names only when they preserve
  this plan's contracts.
- The responsible agent must not introduce alternate migration strategies,
  compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside `coding_agent`,
  `background_work/subagent/coding_agent.py`, docs, and tests as out of scope.
- The responsible agent must search for existing path and patch helpers before
  adding new helpers.
- The responsible agent must not perform unrelated cleanup, dependency
  upgrades, prompt rewrites, broad refactors, or formatting churn.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Preserve the precommitted live LLM gate file
   `tests/test_coding_agent_phase5_patch_apply_live_llm.py`. These gates are
   the signoff contract and must not be relaxed during implementation.
2. Add focused failing tests in
   `tests/test_coding_agent_phase5_patch_apply_contracts.py`:
   approval required, source identity mismatch rejection, managed-copy apply
   success, original source immutability, patch conflict failure, unsafe path
   rejection, and public metadata sanitization.
3. Add focused failing tests in `tests/test_coding_agent_phase5_interface.py`
   for `apply_approved_patch` export and background `code_modifying` metadata
   preservation.
4. Implement `code_patching.apply` with path-safe managed copy creation, source
   identity validation, patch cap validation, `git apply --check`, and `git
   apply` inside the managed copy only.
5. Export the direct public API and typed models.
6. Fix background worker metadata normalization to include `code_modifying`.
7. Update README, HOWTO, and reference architecture.
8. Run focused tests, regression tests, static greps, compile checks, and the
   five live LLM gates one case at a time.
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

- [ ] Stage 1 - focused test contract established.
  - Covers: implementation steps 1, 2, and 3.
  - Verify: focused Phase 5 tests fail for missing API or missing behavior.
  - Evidence: record test commands and expected failures.
  - Sign-off: `<agent/date>`.
- [ ] Stage 2 - patch apply module implemented.
  - Covers: implementation steps 4 and 5.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_coding_agent_phase5_patch_apply_contracts.py -q`.
  - Evidence: record changed production files and passing focused test output.
  - Sign-off: `<agent/date>`.
- [ ] Stage 3 - background metadata and docs updated.
  - Covers: implementation steps 6 and 7.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_coding_agent_phase5_interface.py tests\test_background_work_coding_agent.py -q`.
  - Evidence: record test output and documentation files changed.
  - Sign-off: `<agent/date>`.
- [ ] Stage 4 - regression verification complete.
  - Covers: implementation step 8.
  - Verify: all commands in `Verification`.
  - Evidence: record command outputs and accepted static grep results.
  - Sign-off: `<agent/date>`.
- [ ] Stage 5 - independent code review complete.
  - Covers: implementation step 9.
  - Verify: review findings are recorded, fixes are applied, and affected
    verification commands pass again.
  - Evidence: record review result, remediation, residual risk, and approval.
  - Sign-off: `<agent/date>`.
- [ ] Stage 6 - lifecycle closure complete.
  - Covers: implementation step 10.
  - Verify: plan status and registry are updated, active plan moved to
    completed archive, and `git status --short` is reviewed.
  - Evidence: record final commit or handoff state.
  - Sign-off: `<agent/date>`.

## Live LLM Gating Tests

The five live gates are committed ahead of implementation in
`tests/test_coding_agent_phase5_patch_apply_live_llm.py`. Each gate produces a
real LLM patch proposal through `propose_code_change(...)`, applies approved
patch artifacts through `apply_approved_patch(...)`, writes durable raw
evidence to `test_artifacts/llm_traces/coding_agent_phase5_patch_apply/`, and
requires one-at-a-time human or AI review of the raw evidence.

### Gate 01 - CLI JSON apply

- Objective: prove an explicitly approved live LLM patch proposal for a small
  single-file CLI change applies into a managed copy while the original
  checkout remains unchanged.
- Existing code base: `gate_01_log_counter`, a standard-library
  `log_counter.py` CLI with text output and focused tests.
- Modification instruction: add `--json`, emit valid JSON with severity and
  skipped-line counts, preserve default text output, and update focused tests.
- Expected state: proposal succeeds, apply succeeds under
  `patch_apply/<package>/source`, original hashes are unchanged, and applied
  files include source plus focused tests.
- Pass criteria: proposal and apply statuses are `succeeded`, managed applied
  files exist, public metadata omits absolute paths, and no execution output is
  present.

### Gate 02 - JSONL utility apply

- Objective: prove managed apply for a parser plus CLI change where source,
  tests, and README move together.
- Existing code base: `gate_02_contacts_jsonl_to_csv`, a package with
  converter, CLI, README, converter tests, and CLI tests.
- Modification instruction: make `--fields` exact-order, write blanks for
  missing fields, report malformed JSON with 1-based line numbers, continue by
  default, fail fast with `--strict`, and update tests/docs.
- Expected state: converter, CLI, tests, and README apply into the managed
  workspace while the original checkout remains unchanged.
- Pass criteria: proposal evidence cites converter and CLI ownership, apply
  succeeds, applied paths include source/tests/docs, and apply metadata has no
  stdout, stderr, command, or exit code.

### Gate 03 - Markdown parser apply

- Objective: prove managed apply for a parser-owned Markdown link checker
  change with scanner and anchor behavior kept coherent.
- Existing code base: `gate_03_markdown_link_checker`, a package with
  `anchors.py`, `scanner.py`, CLI, README, and anchor/scanner tests.
- Modification instruction: ignore links inside fenced code and HTML comments,
  support duplicate heading suffixes `base`, `base-1`, `base-2`, update
  parser/scanner tests, and preserve normal links.
- Expected state: scanner, anchor, and focused test changes apply atomically
  into one managed package.
- Pass criteria: proposal evidence cites scanner/anchors/tests, apply
  succeeds, managed files exist, public response omits absolute paths, and the
  original checkout is unchanged.

### Gate 04 - Issue tracker apply

- Objective: prove managed apply for a cross-layer soft-delete change without
  hard-delete compatibility shims.
- Existing code base: `gate_04_issue_tracker_soft_delete`, a package with
  model, store, API, README, and store/API tests.
- Modification instruction: mark deleted issues archived, hide archived issues
  from normal lookup/list, add `include_archived`, update tests/docs, and avoid
  wrappers around old hard-delete semantics.
- Expected state: model, store, API, tests, and README apply into a managed
  workspace while the original checkout remains unchanged.
- Pass criteria: proposal evidence cites model/store/API, apply succeeds,
  applied source/test/doc paths exist, public metadata is sanitized, and no
  old hard-delete compatibility path is accepted by the review rubric.

### Gate 05 - Inventory cache apply

- Objective: prove managed apply for a hard mixed patch that may include a
  justified helper while preserving existing package responsibilities.
- Existing code base: `gate_05_inventory_sync_fetch_cache`, a package with CSV
  reading, urllib fetch, HTML extraction, report writing, CLI, README, and
  mocked HTTP tests.
- Modification instruction: add timeout/retry fetch behavior, file-backed
  cache, CLI flags `--cache-dir`, `--refresh-cache`, and `--timeout`, mocked
  HTTP tests, and README workflow documentation.
- Expected state: fetch, CLI, tests, README, and any justified helper apply
  under one managed package while the original checkout remains unchanged.
- Pass criteria: proposal evidence cites fetch/CLI/tests/README, apply
  succeeds, managed files exist, public metadata has no absolute paths or
  execution output, and tests remain mocked by review.

## Verification

### Static Checks

- `venv\Scripts\python -m compileall src\kazusa_ai_chatbot\coding_agent src\kazusa_ai_chatbot\background_work tests\test_coding_agent_phase5_patch_apply_contracts.py tests\test_coding_agent_phase5_interface.py tests\test_coding_agent_phase5_patch_apply_live_llm.py`
  must pass.
- `git diff --check` must report no whitespace errors.
- `rg "code_modifying\"\\)|code_modifying\", \"unsupported\"|not in \\(\"code_reading\", \"code_writing\", \"unsupported\"\\)" src\kazusa_ai_chatbot\background_work src\kazusa_ai_chatbot\coding_agent -n`
  must show no stale metadata enum gate that excludes `code_modifying`.
- `rg "subprocess|pytest|pip|npm|shell=True|Start-Process" src\kazusa_ai_chatbot\coding_agent\code_patching -n`
  may match only existing `git apply` implementation in the apply/review
  boundary; any command execution outside patch apply is forbidden.

### Tests

- `venv\Scripts\python -m pytest tests\test_coding_agent_phase5_patch_apply_contracts.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_phase5_interface.py tests\test_background_work_coding_agent.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_phase4_code_patching_contracts.py tests\test_coding_agent_phase4_interface.py tests\test_coding_agent_phase2_new_artifact_contracts.py -q`

### Live LLM Tests

Run one case at a time with `-s`, inspect the emitted raw evidence, and record
the quality judgment before running the next case:

- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_phase5_patch_apply_live_llm.py::test_phase5_live_gate_01_cli_json_apply -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_phase5_patch_apply_live_llm.py::test_phase5_live_gate_02_jsonl_errors_apply -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_phase5_patch_apply_live_llm.py::test_phase5_live_gate_03_markdown_parser_apply -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_phase5_patch_apply_live_llm.py::test_phase5_live_gate_04_issue_tracker_apply -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_phase5_patch_apply_live_llm.py::test_phase5_live_gate_05_inventory_cache_apply -q -s`

### Manual Review

- Inspect one successful managed apply workspace locally and confirm the
  original fixture source hashes are unchanged.
- Confirm public response representation omits absolute source and workspace
  paths.
- For every live LLM gate, inspect the raw evidence JSON and judge the
  behavior rubric plus forbidden failure modes before signoff.

## Confidence Assessment

Confidence to pass the five live LLM apply gates after this plan is implemented
as written: 92%.

Rationale:

- The live gates reuse the completed Phase 4 real-world fixture matrix, so the
  LLM-facing proposal workload is already aligned with existing code-modifying
  evidence.
- The new Phase 5 behavior under test is deterministic after patch artifacts
  exist: approval validation, source identity validation, managed copy
  creation, `git apply --check`, `git apply`, hash preservation, and metadata
  sanitization.
- The apply response contract now includes `apply_workspace_ref`, removing the
  previous handoff gap between Phase 5 and Phase 6.
- The gates assert original checkout immutability and public metadata
  sanitation directly.

Residual risk:

- A live LLM proposal regression from Phase 4 can still fail a Phase 5 gate
  before apply starts. The Phase 5 plan mitigates this by retaining Phase 4
  regression tests and by requiring raw evidence review that separates proposal
  quality from apply-boundary defects.
- The gates prove managed apply and inspection readiness; they do not prove
  runtime behavior of applied code. Runtime verification belongs to Phase 6.

## Independent Plan Review

Fresh review inputs:

- `development_plans/reference/designs/coding_agent_architecture.md`
- `development_plans/archive/completed/short_term/coding_agent_phase4_code_modifying_and_patching_plan.md`
- `src/kazusa_ai_chatbot/coding_agent/README.md`
- `src/kazusa_ai_chatbot/coding_agent/code_patching/README.md`
- `tests/test_coding_agent_existing_source_e2e_live_llm.py`
- `tests/test_coding_agent_phase5_patch_apply_live_llm.py`

Review findings and remediation:

- Blocker: the original Phase 5 draft had no precommitted real LLM signoff
  tests under `tests/`. Resolved by adding
  `tests/test_coding_agent_phase5_patch_apply_live_llm.py` with five populated
  gates and durable trace requirements.
- Blocker: the original apply response contract did not provide a Phase 6
  handoff reference. Resolved by adding `apply_workspace_ref` and the internal
  `patch_apply/<apply_package_id>/source` layout while preserving the rule that
  absolute paths stay private.
- Blocker: live gate verification was absent from the command list. Resolved by
  adding exact one-at-a-time live LLM commands and manual evidence review.
- Non-blocking finding: Phase 5 live gates depend on Phase 4 proposal quality.
  Accepted because Phase 5 intentionally consumes patch artifacts and the plan
  keeps Phase 4 regression tests in scope.

Approval status: draft strengthened for user review. It is not executable until
the user approves or commands implementation.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Approval contract cannot be inferred from raw user text or LLM prose.
- Original checkout mutation is impossible through the implemented path.
- Managed apply workspace path safety and cleanup behavior are bounded.
- Source identity mismatch and dirty checkout checks fail closed.
- Public metadata is sanitized.
- Background metadata preserves `code_modifying`.
- Tests cover success, rejection, conflict, sanitization, and regression paths.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `apply_approved_patch(...)` exists and applies approved patch artifacts only
  into a managed apply workspace.
- Missing approval, mismatched source identity, unsafe paths, and patch
  conflicts fail closed.
- Original source trees remain unchanged in focused tests.
- Background worker metadata preserves `code_modifying`.
- Coding-agent docs and architecture reference describe Phase 5 accurately.
- All verification commands pass.
- Independent code review accepts the implementation.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Accidental original checkout mutation | Apply only to managed copy | Original hash comparison test |
| Approval inferred from prose | Require structured approval object | Approval rejection tests |
| Stale patch applied to wrong source | Compare source identity before copy | Mismatch rejection test |
| Private path leakage | Public response sanitization | Metadata assertions |
| Scope creep into execution | Keep command/test execution deferred | Static grep and review |

## Execution Evidence

- Focused test baseline:
- Implementation files changed:
- Focused test pass:
- Regression test pass:
- Static check results:
- Manual review:
- Independent code review:
- Lifecycle closure:
