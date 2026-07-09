# coding agent full workflow hardening plan 2

## Summary

- Goal: raise first-time pass confidence for the final coding-agent full
  workflow integration gates to at least 90% by restoring the exact gate
  contract from the archived full workflow integration test plan, adding the
  missing same-run proposal-revision and run-summary continuation contracts,
  and closing only against those restored gates.
- Plan class: large.
- Status: completed.
- Mandatory skills: `development-plan`, `test-style-and-execution`,
  `debug-llm`, `local-llm-architecture`, and `py-style`.
- Overall cutover strategy: bigbang for the corrected full-workflow gate
  contract; compatible extension for new durable coding-run continuation
  actions.
- Highest-risk areas: gate-contract drift, weak local LLM action selection,
  same-run proposal revision, protected-test constraints, managed
  approval/verification, attempt-history projection, fixture realism, and
  false confidence from narrow entrypoint tests.
- Acceptance criteria: this plan is complete only when the restored five
  full-workflow gates pass one at a time with agent-authored review evidence,
  the implementation supports the continuation behavior those gates require,
  and a fresh evidence-based confidence review rates first-time pass confidence
  at 90% or higher.

## Context

The previous pre-integration hardening work made durable coding runs reachable
from the L2d accepted-task/background-worker entrypoint, but it closed against
smaller gates than the archived full workflow integration plan required.

The archived integration plan required:

- committed fixture repositories under `tests/fixtures/`;
- at least three multi-turn gates with two or more follow-up requests;
- Gate 02 source-free proposal revision for dry-run mode and deterministic
  output;
- Gate 03 existing-source proposal revision that preserves runtime-only
  source changes and does not edit tests;
- Gate 04 approval, managed verify, and repair projection;
- Gate 05 multi-file approval, attempt-history, conditional cancellation, and
  final-status behavior across one durable run identity.

The committed live test file currently proves the L2d/background-worker
entrypoint and run-reference binding, but it does not fully prove same-run
proposal revision, changed-file summaries, hard fixture behavior, or the
multi-turn Gate 05 contract. The current durable payload supports only
`start`, `status`, `approve_and_verify`, and `cancel`; this cannot represent
the proposal-revision follow-ups required by the full workflow plan.

This plan is a new active plan. It does not edit or append new scope to the
completed integration or pre-integration hardening records.

## Mandatory Skills

- `development-plan`: load before editing this plan, approving it, executing
  it, updating checklist status, or signing off.
- `test-style-and-execution`: load before adding, changing, or running
  deterministic, live DB, or real LLM tests.
- `debug-llm`: load before running real LLM gates and before authoring
  readable review artifacts from raw traces.
- `local-llm-architecture`: load before changing action-selection prompts,
  coding-worker prompts, coding-run continuation contracts, or LLM context
  payloads.
- `py-style`: load before editing Python files.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run this plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- This plan's execution model uses parent-led native subagent execution unless
  the user explicitly approves fallback execution.
- The restored full-workflow tests must be created and reviewed before
  production-code hardening starts.
- The restored tests must map one-to-one to the archived full workflow gate
  definitions. Do not replace them with narrower entrypoint, status-only,
  approval-only, or cancel-only tests.
- At least three real LLM gates must include two or more follow-up turns.
- Real LLM gates must run one case at a time and be inspected one case at a
  time. Passing pytest is necessary but not sufficient.
- Agent-authored review artifacts must be written from raw evidence. Scripts
  and tests may emit raw JSON/logs only, not human-readable quality judgments.
- Deterministic code owns action validation, run identity, lifecycle state,
  duplicate identity, approval objects, path containment, allowed execution
  tools, terminal-state checks, and public sanitization.
- LLM stages own semantic task interpretation, revision intent, code-planning
  quality, source-owner reasoning, and bounded repair reasoning.
- Background coding execution must never mutate original source checkouts,
  install packages, run arbitrary shell commands, send adapter text directly,
  infer approval from casual prose, or use hidden direct-API shortcuts.
- If restored gates expose a missing public contract, stop and update this
  active plan before implementing the new contract.

## Must Do

- Replace the narrow full-workflow live gates with restored gates that strictly
  match the archived full workflow integration test plan.
- Add committed fixture repositories under
  `tests/fixtures/coding_agent_full_workflow/` for all five gates.
- Add a gate-to-test traceability matrix inside the live test file or a
  committed adjacent fixture manifest.
- Add durable coding-run continuation support for same-run proposal revision
  before approval.
- Add durable coding-run support for public run summaries that expose
  changed files, attempt history, current status, and allowed next actions
  without requiring another patch proposal.
- Extend `accepted_coding_task_request` and `coding_agent_worker_payload.v1`
  with the new closed continuation actions required by the restored gates.
- Keep run identity stable across start, revision, approval, summary, status,
  and cancellation turns.
- Preserve structured approval and allowlisted verification only:
  `python_compileall` and focused `pytest`.
- Keep proposal revision review-only until explicit approval.
- Add deterministic tests for the new action-spec, queue, worker, and
  coding-run continuation contracts before production implementation.
- Run Phase 5-9 regressions plus the restored full workflow gates before
  closure.
- Perform a final evidence-based confidence review against the implemented
  tests and code. The review must state first-time pass confidence. If it is
  below 90%, this plan remains incomplete.

## Deferred

- Do not add arbitrary shell execution, package installation, dependency
  solving, repository push, deployment, database mutation, or adapter
  direct-send behavior.
- Do not implement Phase 10 repository-scale reading.
- Do not add a UI for coding runs.
- Do not introduce compatibility aliases, alternate payload shapes, fallback
  mappers, or hidden direct API shortcuts.
- Do not make generic L2d, L3/dialog, delivery, permission, or persistence
  prompts own coding-run internals.
- Do not weaken fixture bugs, remove follow-up turns, mark tests xfail, skip
  failed gates, or broaden assertions after seeing a failure.
- Do not treat in-memory persistence gates as proof of live DB delivery.
  Live DB delivery coverage may be a separate smoke gate, but it must not
  replace the restored full workflow gates.

## Cutover Policy

Overall strategy: targeted bigbang for corrected full-workflow gates, with a
compatible contract extension for durable coding-run continuations.

| Area | Policy | Instruction |
|---|---|---|
| Full workflow live gates | bigbang | Replace narrow gates with the restored archived gate behavior. Do not keep narrower gates as closure proof. |
| Fixture source | bigbang | Commit real fixture repositories under `tests/fixtures/coding_agent_full_workflow/`. Do not rely on ad hoc temp-only source trees for gate definitions. |
| Durable continuation actions | compatible | Add closed actions for proposal revision and run summary while preserving existing `start`, `status`, `approve_and_verify`, and `cancel`. |
| Existing generic background coding path | compatible | Preserve legacy read/proposal behavior for generic delayed coding tasks. |
| Approval and execution | bigbang | Keep structured approval and allowlisted execution. Do not infer approval or command execution from prose. |
| Confidence closure | bigbang | The plan cannot close until confidence is re-evaluated at 90% or higher from restored gate evidence. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- Bigbang areas must be implemented as one canonical contract, not as
  compatibility shims.
- Compatible areas preserve only the surfaces explicitly listed in this plan.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

The coding agent can pass the final full workflow integration suite through
the real L2d/action-spec/background-worker entrypoint with at least 90%
first-time confidence. The supported workflow is:

```text
L2d accepted_coding_task_request
-> deterministic action-spec validation
-> accepted-task lifecycle
-> requested_worker="coding_agent"
-> coding_agent_worker_payload.v1
-> durable coding run
-> start / revise_proposal / summarize / status / approve_and_verify / cancel
-> managed apply / focused execution / capped repair when approved
-> public run projection with changed files, attempts, status, and allowed next actions
```

The run remains bounded for local LLMs. L2d selects a semantic coding action.
The coding worker maps that action to a closed deterministic contract.
Coding PM/programmer stages reason about source and patch content only.
Deterministic code validates approval, execution, state transitions, paths,
and public sanitization.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Test contract | Restore the archived Gate 01-05 behavior exactly before production edits. | The previous hardening failed because the test surface narrowed. |
| Fixtures | Commit fixture repositories under `tests/fixtures/coding_agent_full_workflow/`. | Fixture absence hid the mismatch between plan and test behavior. |
| Revision action | Add `revise_proposal` as a closed durable coding action. | Source-free and existing-source follow-ups require same-run proposal updates before approval. |
| Summary action | Add `summarize` as a closed durable coding action backed by public run projection. | Gates require changed-file and attempt-history follow-ups without creating new proposals. |
| Run state | Preserve one run id across revisions and summaries. | The user-visible workflow is Codex-like continuity, not separate tasks. |
| Revision scope | Allow revisions only while the run is `awaiting_approval`. | Avoid changing already-approved or terminal state without a new run. |
| Protected tests | Treat user instructions such as "do not edit tests" as revision constraints in the next proposal. | Gates require respecting protected paths through follow-up behavior. |
| Approval-time protected verification paths | Omit protected verification-path artifacts from approved managed apply before execution. | Restored Gate 04 exposed that an initial proposal can include the pytest selector itself; verification tests must remain read-only even when they appear in stored proposal artifacts. |
| Execution | Keep execution closed to compileall and focused pytest. | Maintains the Phase 5-9 safety boundary. |
| Source-free review feedback | Add one bounded code-writing validation-feedback pass when review materialization rejects a source-free proposal. | Restored Gate 02 exposed a real import-coherence failure after successful artifact generation; the PM/programmer need one targeted chance to regenerate a coherent full package from deterministic validation feedback. |
| Confidence | Require a final confidence review at >=90%. | Prevents repeating the previous false-closure failure. |

## Contracts And Data Shapes

Extend the model-facing coding action enum:

```json
{
  "decision": "start | revise_proposal | summarize | status | approve_and_verify | cancel",
  "coding_run_ref": "coding_run:<run_id>",
  "detail": "current user-visible instruction",
  "execution_request": "optional focused pytest or compile request"
}
```

Extend `coding_agent_worker_payload.v1` without changing the schema version:

```json
{
  "schema_version": "coding_agent_worker_payload.v1",
  "operation": "start | revise_proposal | summarize | status | approve_and_verify | cancel",
  "task_brief": "current user-visible instruction",
  "coding_run_ref": "coding_run:<run_id>",
  "execution_request": "string",
  "execution_specs": []
}
```

`revise_proposal` rules:

- requires `coding_run_ref`;
- valid only when the run status is `awaiting_approval`;
- creates no apply, execution, or repair attempts;
- incorporates the original goal, prior public run projection, and current
  revision instruction into a fresh proposal request;
- replaces the public proposal artifacts, changed files, answer text, and
  limitations with the revised proposal;
- appends a `proposal_revised` event;
- preserves the same run id.

`summarize` rules:

- requires `coding_run_ref`;
- never calls patch proposal, apply, execution, or repair APIs;
- returns a public projection containing current status, changed files,
  apply/execution/repair attempts, limitations, blockers, and allowed next
  actions;
- appends a `summary_requested` event only when doing so does not mutate
  terminal source or proposal state;
- preserves the same run id.

Existing `status` remains a compact progress check. `summarize` is the richer
Codex-like follow-up for "what changed", "show attempt history", and "final
files" requests.

Public worker metadata must include no local roots, workspace roots, raw
diffs, full source, `.env`, `.git`, cache keys, secret-like text, or unbounded
command output.

## LLM Call And Context Budget

Before this plan:

- L2d selects `accepted_coding_task_request` with `start`, `status`,
  `approve_and_verify`, or `cancel`.
- The coding worker routes start requests through the coding PM route and
  approval requests through bounded execution-spec planning.
- Proposal revision is not represented as a first-class coding action.

After this plan:

- L2d can select `revise_proposal` and `summarize` when the user asks to
  refine an existing run or inspect changed files/attempts.
- `summarize` uses deterministic public run projection only.
- `revise_proposal` may call existing proposal machinery once, using a bounded
  current-run revision prompt built from the original goal, current user
  instruction, and public-safe prior run projection.
- No new response-path LLM call is added. New LLM calls are background-only.
- A restored Gate 02 failure exposed one source-free review materialization
  failure after artifacts were generated successfully. Add only one bounded
  validation-feedback pass inside `code_writing`; do not add unbounded retry,
  command execution, package installation, generated-test execution, or
  direct artifact rewriting.
- Use concise semantic text in prompts; do not pass raw ledgers, raw source
  trees, raw diffs, or raw command output to LLM stages.

## Change Surface

### Modify

- `tests/test_coding_agent_full_workflow_integration_live_llm.py`: replace
  narrow gates with restored Gate 01-05 multi-turn gates, traceability
  metadata, anti-cheat assertions, and raw trace emission.
- `tests/test_coding_agent_background_run_contracts.py`: add deterministic
  contracts for `revise_proposal`, `summarize`, run-ref requirements, queue
  validation, worker dispatch, and metadata sanitization.
- `tests/test_coding_agent_phase9_e2e_workflows.py`: add direct durable-run
  deterministic coverage for proposal revision and summary projection.
- `tests/test_coding_agent_phase2_new_artifact_contracts.py`: add a
  deterministic contract for source-free review-materialization feedback after
  a generated package has unresolved local imports.
- `tests/test_coding_agent_phase8_verify_repair_contracts.py`: add a
  deterministic contract proving protected verification-path artifacts are
  omitted from approved managed apply before execution.
- `src/kazusa_ai_chatbot/action_spec/registry.py`: expose the new closed
  coding decisions to L2d.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`:
  materialize `revise_proposal` and `summarize`, recover visible run refs, and
  preserve revision detail.
- `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py`: validate
  the new operations and build trusted worker payloads.
- `src/kazusa_ai_chatbot/background_work/jobs.py`: validate the new operations
  in `coding_agent_worker_payload.v1`.
- `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`: dispatch
  `revise_proposal` and `summarize` through durable run APIs.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/supervisor.py`: add closed
  continuation actions, state transitions, events, revised proposal handling,
  summary projection, and allowed-next-action projection.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/models.py`: update typed
  continuation and response shapes when required by the implementation.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/ledger.py`: update ledger
  defaults/projection only if the new events or allowed-next-action metadata
  cannot be represented safely today.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/supervisor.py`: add one
  bounded validation-feedback pass for source-free review-package failures,
  preserving managed-storage and no-execution boundaries.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/product_manager.py`: harden
  the writing PM prompt so validation feedback is treated as a full-package
  correction instruction, not as permission to run commands or patch real
  workspaces.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/supervisor.py`: omit
  proposal artifacts that touch protected pytest selector paths before managed
  apply, and report the omission in public limitations and trace summary.
- `src/kazusa_ai_chatbot/background_work/README.md`,
  `src/kazusa_ai_chatbot/coding_agent/README.md`,
  `src/kazusa_ai_chatbot/coding_agent/code_writing/README.md`,
  `docs/HOWTO.md`, and
  `development_plans/reference/designs/coding_agent_architecture.md`: update
  implemented contracts after code and tests are passing.

### Create

- `tests/fixtures/coding_agent_full_workflow/gate_01_cli_command_discovery/`.
- `tests/fixtures/coding_agent_full_workflow/gate_02_csv_normalizer/`.
- `tests/fixtures/coding_agent_full_workflow/gate_03_counter_cli_json/`.
- `tests/fixtures/coding_agent_full_workflow/gate_04_slug_normalization/`.
- `tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/`.
- A committed fixture manifest under
  `tests/fixtures/coding_agent_full_workflow/manifest.json` or `.md` mapping
  each archived gate requirement to fixture files and hard assertions.

### Keep

- Existing Phase 5-9 direct APIs.
- Existing generic background coding behavior for non-durable delayed coding
  jobs.
- Existing managed apply, execution, and verify/repair safety boundaries.
- Existing `coding_agent_worker_payload.v1` schema version, with the closed
  operation enum extended in-place.

### Delete

- Delete or rewrite narrow full-workflow gate bodies that no longer prove the
  restored integration contract. Do not delete Phase 5-9 module-level tests.

## Overdesign Guardrail

- Actual problem: the previous hardening plan closed against narrow
  entrypoint gates and did not prove the archived final full-workflow
  integration behavior, leaving first-time pass confidence below 90%.
- Minimal change: restore the exact five full workflow gates, add committed
  fixtures, and add only the missing `revise_proposal` and `summarize`
  continuation actions required by those gates.
- Ownership boundaries: L2d selects semantic coding actions; action-spec and
  background-work code validate payloads and run refs; coding-run owns durable
  state and transitions; code-writing/modifying own proposal content;
  code-verifying owns managed apply/execution/repair; deterministic code owns
  approval, execution allowlists, path containment, and sanitization.
- Rejected complexity: no arbitrary shell, package install, dependency
  resolver, adapter send path, UI, Phase 10 repository-scale reading, generic
  tool-permission system, fallback action aliases, compatibility payload
  shapes, or hidden direct API shortcuts.
- Evidence threshold: any further continuation action, prompt field, helper
  agent, retry loop, or persistence change requires a restored gate failure
  showing the minimal contract cannot represent a required supported workflow,
  followed by an explicit plan update.
- Gate 02 evidence reached that threshold for one source-free writing
  validation-feedback pass only. Anti-cheat constraint: do not weaken Gate 02,
  skip or xfail it, loosen review materialization, ignore unresolved imports,
  synthesize aliases, mark a failed proposal as approval-ready, or keep a
  terminal failed run revisionable to pass the follow-up turns.
- Gate 04 evidence reached that threshold for approval-time protected
  verification-path filtering only. Anti-cheat constraint: do not weaken Gate
  04, skip or xfail it, remove the protected-test assertion, pretend tests were
  not proposed, run unapproved shell commands, or mutate the original checkout.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper names only when the contracts
  in this plan are preserved.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside the listed change surface
  as blockers requiring plan update or user approval.
- The responsible agent must search for existing helpers before adding new
  helpers.
- The responsible agent must keep fixture changes scoped to the restored gate
  definitions and must not simplify fixtures after seeing model failures.
- If the plan and code disagree, the responsible agent must preserve this
  plan's stated intent and report the discrepancy.
- If a required instruction is impossible, the responsible agent must stop and
  report the blocker instead of inventing a substitute.

## Implementation Order

1. Create the five committed fixture repositories and fixture manifest under
   `tests/fixtures/coding_agent_full_workflow/`.
2. Rewrite `tests/test_coding_agent_full_workflow_integration_live_llm.py`
   so Gate 01-05 match the archived integration plan, including multi-turn
   follow-ups and hard assertions.
3. Run test collection and record baseline behavior. Expected before
   implementation: collection passes; Gates 02, 03, and 05 fail or block
   because `revise_proposal` and `summarize` do not exist.
4. Add deterministic action-spec and queue tests for `revise_proposal` and
   `summarize`.
5. Add deterministic direct coding-run tests for proposal revision, summary
   projection, terminal-state rejection, and protected-test constraint
   preservation.
6. Implement action registry, materialization, validation, and queue support
   for the new operations.
7. Implement coding worker dispatch for the new operations.
8. Implement coding-run continuation state transitions, revision proposal
   generation, summary projection, and allowed-next-action metadata.
9. Run focused deterministic tests and fix contract failures inside the
   approved change surface.
10. Add a deterministic red/green test for the restored Gate 02
    review-materialization failure and implement the bounded code-writing
    validation-feedback pass.
11. Run Phase 5-9 deterministic regressions.
12. Run the restored real LLM gates one at a time, inspecting and authoring a
    review artifact after each gate.
13. Update ICDs, HOWTO, and architecture docs to match the implemented
    contract.
14. Perform an evidence-based confidence review. If confidence is below 90%,
    record the specific failing risk and continue remediation under this plan.
15. Run independent code review, remediate findings, rerun affected
    verification, and record final evidence.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the restored test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  restored test contract and focused deterministic tests are established; owns
  production code changes only; does not edit tests unless the parent
  explicitly directs it; closes after planned production changes are complete,
  excluding review fixes.
- Parent agent may continue integration tests, regression tests, static
  checks, and validation work while the production-code subagent edits
  production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - restored fixture contract established
  - Covers: implementation steps 1-3.
  - Files: `tests/fixtures/coding_agent_full_workflow/`,
    `tests/test_coding_agent_full_workflow_integration_live_llm.py`.
  - Verify: collect five live LLM gates and record expected pre-implementation
    failures for missing continuation actions.
  - Evidence: fixture manifest, collection output, baseline failure notes.
  - Sign-off: Codex 2026-07-09 after fixture creation, live-gate collection,
    and static anti-cheat greps.
- [x] Stage 2 - deterministic continuation contracts established
  - Covers: implementation steps 4-5.
  - Files: `tests/test_coding_agent_background_run_contracts.py`,
    `tests/test_coding_agent_phase9_e2e_workflows.py`.
  - Verify: focused deterministic tests fail for missing
    `revise_proposal`/`summarize` support before implementation.
  - Evidence: failing test output recorded in `Execution Evidence`.
  - Sign-off: Codex 2026-07-09 after focused red-state test runs.
- [x] Stage 3 - action-spec, queue, worker, and run implementation complete
  - Covers: implementation steps 6-9.
  - Files: source files listed in `Change Surface`.
  - Verify: focused deterministic tests pass.
  - Evidence: test output and changed-file summary recorded.
  - Sign-off: Codex 2026-07-09 after focused deterministic green-state tests.
- [x] Stage 4 - regression verification complete
  - Covers: implementation step 11 as originally executed before the Gate 02
    remediation scope was added.
  - Verify: Phase 5-9 deterministic regression commands pass.
  - Evidence: command outputs recorded.
  - Sign-off: Codex 2026-07-09 after static checks and deterministic
    regression commands passed.
- [x] Stage 5 - restored real LLM gates complete
  - Covers: implementation steps 10 and 12 after the Gate 02 remediation
    scope update.
  - Verify: the Gate 02 validation-feedback deterministic test passes,
    affected deterministic regressions are rerun, then five restored gates run
    one at a time; each raw trace is inspected and each review artifact is
    authored by the agent.
  - Evidence: deterministic command output, trace paths, and review artifact
    paths recorded.
  - Sign-off: Codex 2026-07-09 after all five restored live LLM gates passed
    one at a time with raw traces and review artifacts.
- [x] Stage 6 - documentation and architecture updated
  - Covers: implementation step 13.
  - Verify: docs mention implemented operations and no stale narrow-gate
    closure wording remains.
  - Evidence: doc diff summary and grep results recorded.
  - Sign-off: Codex 2026-07-09 after ICD/HOWTO/architecture docs were
    updated and stale-contract greps passed.
- [x] Stage 7 - confidence review reaches 90% or higher
  - Covers: implementation step 14.
  - Verify: read actual implementation and tests, inspect gate evidence, and
    record first-time pass confidence.
  - Evidence: confidence review states confidence >=90% and why.
  - Sign-off: Codex 2026-07-09 after actual implementation, deterministic
    tests, static anti-cheat checks, raw gate traces, and review artifacts
    were re-inspected.
- [x] Stage 8 - independent code review complete
  - Covers: implementation step 15.
  - Verify: review findings recorded, fixes applied, affected tests rerun, and
    no unresolved blockers remain.
  - Evidence: review outcome and rerun commands recorded.
  - Sign-off: Codex 2026-07-09 after independent review, remediation,
    affected verification, full deterministic regression, and static checks.

## Verification

### Static Greps

- `rg "revise_proposal|summarize" src tests development_plans`
  - Expected after implementation: matches in the new operation contracts,
    tests, docs, and this plan.
- `rg "start_coding_run\\(|continue_coding_run\\(|verify_and_repair_code_change\\(" tests\\test_coding_agent_full_workflow_integration_live_llm.py`
  - Expected: no matches. A nonzero `rg` exit code is acceptable and means
    direct API shortcuts are absent from the restored integration tests.
- `rg "xfail|skip\\(" tests\\test_coding_agent_full_workflow_integration_live_llm.py`
  - Expected: no matches except the existing LLM endpoint availability guard
    if retained. Any other match is a blocker.
- `rg "tests/fixtures/coding_agent_full_workflow" tests\\test_coding_agent_full_workflow_integration_live_llm.py tests\\fixtures\\coding_agent_full_workflow`
  - Expected: the live gate file and fixture manifest reference committed
    fixture roots.

### Deterministic Tests

- `venv\Scripts\python -m pytest tests\test_coding_agent_background_run_contracts.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_phase9_e2e_workflows.py -q`
- `venv\Scripts\python -m pytest tests\test_action_spec_evaluator.py tests\test_action_selection_prompt_contract.py tests\test_action_selection_payload.py -q`
- `venv\Scripts\python -m pytest tests\test_background_work_jobs.py tests\test_background_work_coding_agent.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_phase5_patch_apply_contracts.py tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase8_verify_repair_contracts.py tests\test_coding_agent_phase9_run_supervisor_contracts.py -q`

### Real LLM Gates

Run one at a time with `-q -s`. Inspect the emitted raw trace and author a
human-readable review before running the next gate.

- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_01_read_only_question_from_l2d_to_worker -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_02_source_free_proposal_with_revision_followups -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_03_existing_source_proposal_with_runtime_only_followups -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_04_approval_verify_and_repair_followups -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_05_hard_multifile_approval_history_cancel_status -q -s`

### Confidence Review

- Read the final source, tests, fixture manifest, raw traces, and review
  artifacts.
- Score first-time pass confidence for the final integration suite.
- Required result: confidence is at least 90%.
- If confidence is below 90%, record the specific remaining gap and continue
  remediation under this plan.

## Independent Plan Review

Run before approval or execution. The reviewer must inspect this plan, the
archived full workflow integration test plan, the archived pre-integration
hardening plan, current coding-agent/background-work ICDs, and current tests.

Review scope:

- The restored gate definitions map one-to-one to the archived Gate 01-05
  requirements.
- At least three gates include two or more follow-up turns.
- The plan includes committed fixtures, not temp-only source definitions.
- The plan adds only the missing same-run proposal-revision and run-summary
  continuations required by the gates.
- The action contracts remain closed, deterministic, and local-LLM friendly.
- No broad tool execution, package install, adapter send path, Phase 10 work,
  compatibility alias, or hidden direct API shortcut is authorized.
- The confidence target is a closure gate, not a planning slogan.

Record blockers, non-blocking findings, fixes, and approval status in
`Execution Evidence`.

## Independent Code Review

Run after all `Verification` commands pass and before final sign-off. The
parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Alignment with every `Must Do`, `Deferred`, change-surface, implementation
  order, verification, and acceptance criterion in this plan.
- Gate-to-test traceability against archived Gate 01-05 requirements.
- No direct API shortcuts in full workflow live tests.
- Fixture realism and protection against post-failure fixture weakening.
- Action-spec and worker payload validation for new operations.
- Durable run state transitions, terminal-state rejection, run id stability,
  proposal revision behavior, and summary projection.
- Structured approval, allowed execution specs, managed apply/execute/repair,
  and public sanitization.
- Documentation and architecture updates reflecting implemented behavior.
- Final confidence review evidence and whether >=90% is justified by actual
  implementation and gate evidence.

The parent agent may fix findings only inside this plan's approved change
surface. Findings requiring a new public contract, data migration, arbitrary
tool execution, or broader architecture change must stop execution until this
plan is updated and approved.

## Acceptance Criteria

This plan is complete when:

- the restored five full workflow gates are committed and match the archived
  integration plan requirements;
- fixture repositories are committed under
  `tests/fixtures/coding_agent_full_workflow/`;
- at least three restored gates include two or more follow-up turns;
- the coding worker can start, revise, summarize, status-check,
  approve-and-verify, repair, cancel, and report durable coding runs from the
  accepted-task/background-work handoff;
- proposal revisions preserve the same run id and remain review-only before
  approval;
- summaries expose changed files, attempt history, final status, and allowed
  next actions without creating hidden side effects;
- apply, execute, and repair still run only in managed copies and never mutate
  original source;
- Phase 5-9 deterministic regressions pass;
- all five restored real LLM gates pass one at a time with raw trace evidence
  and agent-authored review artifacts;
- the final evidence-based confidence review rates first-time pass confidence
  at 90% or higher;
- independent plan review and independent code review have no unresolved
  blockers.

## Execution Evidence

- Draft created from RCA corrective actions on 2026-07-09.
- 2026-07-09 pre-execution review: compared this plan against the archived
  full workflow integration test plan, archived pre-integration hardening
  record, current full workflow live gates, deterministic background-run
  tests, Phase 9 E2E tests, action-spec contracts, background-work worker
  contracts, and durable coding-run supervisor code. Review result:
  approved for execution. The current implementation still exposes only
  `start`, `status`, `approve_and_verify`, and `cancel`; the current live
  gates remain narrower than the archived Gate 02, Gate 03, Gate 04, and
  Gate 05 contracts. This plan's corrective actions match those gaps by
  restoring the fixture-backed multi-turn gates and adding only
  `revise_proposal` and `summarize`.
- Stage 1 evidence:
  - Created `tests/fixtures/coding_agent_full_workflow/` with five committed
    fixture definitions and `manifest.md`.
  - Replaced the narrowed live LLM gate file with restored Gate 01-05
    multi-turn tests.
  - `venv\Scripts\python -m py_compile tests\test_coding_agent_full_workflow_integration_live_llm.py`: passed.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_full_workflow_integration_live_llm.py -m live_llm --collect-only -q`: collected five restored gates.
  - Anti-cheat grep for direct durable-run API shortcuts in the live gate file:
    no matches.
  - `rg "xfail|skip\(" tests\test_coding_agent_full_workflow_integration_live_llm.py`: only the LLM endpoint availability guard matched.
- Stage 2 red-state evidence:
  - `venv\Scripts\python -m pytest tests\test_coding_agent_background_run_contracts.py -q`: 11 passed, 7 failed. Failures are expected missing support for `revise_proposal`, `summarize`, queue validation, and worker dispatch.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_phase9_e2e_workflows.py -q`: 5 passed, 2 failed. Failures are expected missing direct run continuation actions and allowed-next-action summary projection.
- Stage 3 implementation evidence:
  - Added closed continuation operations `revise_proposal` and `summarize`
    through action-spec materialization, queue validation, background-worker
    dispatch, durable coding-run actions, ledger projection, and supervisor
    state transitions.
  - `revise_proposal` is accepted only while a run is `awaiting_approval`,
    preserves the existing run id, replaces proposal artifacts, and does not
    apply patches, execute commands, or trigger repair.
  - `summarize` records a durable summary request and returns public run-state
    projection, including status, changed paths, attempt count, limitations,
    blockers, and allowed next actions.
  - Changed production files:
    `src/kazusa_ai_chatbot/action_spec/registry.py`,
    `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py`,
    `src/kazusa_ai_chatbot/background_work/jobs.py`,
    `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`,
    `src/kazusa_ai_chatbot/coding_agent/coding_run/models.py`,
    `src/kazusa_ai_chatbot/coding_agent/coding_run/ledger.py`,
    `src/kazusa_ai_chatbot/coding_agent/coding_run/supervisor.py`,
    and `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_background_run_contracts.py -q --tb=short --disable-warnings`: 18 passed.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_phase9_e2e_workflows.py -q --tb=short --disable-warnings`: 7 passed.
  - `venv\Scripts\python -m py_compile` on the eight changed production
    modules: passed.
- Stage 4 regression evidence:
  - `rg "revise_proposal|summarize" src tests development_plans`: matched
    the new source, tests, fixtures, and active plan evidence.
  - `rg "start_coding_run\(|continue_coding_run\(|verify_and_repair_code_change\(" tests\test_coding_agent_full_workflow_integration_live_llm.py`: no matches.
  - `rg "xfail|skip\(" tests\test_coding_agent_full_workflow_integration_live_llm.py`: only LLM endpoint availability guards matched.
  - `rg "tests/fixtures/coding_agent_full_workflow" tests\test_coding_agent_full_workflow_integration_live_llm.py tests\fixtures\coding_agent_full_workflow`: matched the committed fixture manifest.
  - `venv\Scripts\python -m pytest tests\test_action_spec_evaluator.py tests\test_action_selection_prompt_contract.py tests\test_action_selection_payload.py -q --tb=short --disable-warnings`: 26 passed.
  - `venv\Scripts\python -m pytest tests\test_background_work_jobs.py tests\test_background_work_coding_agent.py -q --tb=short --disable-warnings`: 13 passed.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_phase5_patch_apply_contracts.py tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase8_verify_repair_contracts.py tests\test_coding_agent_phase9_run_supervisor_contracts.py -q --tb=short --disable-warnings`: 45 passed.
- Stage 5 Gate 01 evidence:
  - `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_01_read_only_question_from_l2d_to_worker -q -s --tb=short`: passed.
  - Raw traces:
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_01_read_only_question.json`
    and
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_01_read_only_question_turn_1.json`.
  - Agent review artifact:
    `development_plans/active/short_term/coding_agent_full_workflow_hardening_plan_2_llm_gate_reviews.md`.
- Stage 5 Gate 02 RCA evidence:
  - `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_02_source_free_proposal_with_revision_followups -q -s --tb=short`: failed on turn 1 because the worker result was failed.
  - L2d selected `accepted_coding_task_request` with `decision=start`; the
    worker created durable run
    `coding_run:939b970c94274f4f8cf606ae2db838b2`; apply, execution, and
    repair attempts remained zero.
  - Code-writing patcher produced three artifacts, then review materialization
    rejected the package because `tests/test_csv_normalizer_logic.py` imported
    `csv_normalizer` while available local modules were
    `cli_csv_normalizer`, `csv_normalizer_logic`,
    `src.cli_csv_normalizer`, and `src.csv_normalizer_logic`.
  - Same-run follow-ups were rejected only because the failed proposal was
    terminal. RCA: the restored gate exposed a source-free artifact coherence
    gap inside `code_writing`, not an L2d, queue, run-ref, approval, or worker
    dispatch failure. Remediation scope added: one bounded
    validation-feedback PM/programmer pass in `code_writing`, with Gate 02
    assertions unchanged.
  - Added
    `tests/test_coding_agent_phase2_new_artifact_contracts.py::test_code_writing_repairs_source_free_review_import_feedback`.
    Red run before implementation failed with `result["status"] == "failed"`;
    green run after implementation passed.
  - Implemented one capped source-free review-materialization feedback pass in
    `src/kazusa_ai_chatbot/coding_agent/code_writing/supervisor.py` and
    prompt guidance in
    `src/kazusa_ai_chatbot/coding_agent/code_writing/product_manager.py`.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_phase2_new_artifact_contracts.py -q --tb=short --disable-warnings`: 20 passed.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\coding_agent\code_writing\supervisor.py src\kazusa_ai_chatbot\coding_agent\code_writing\product_manager.py`: passed.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_background_run_contracts.py tests\test_coding_agent_phase9_e2e_workflows.py -q --tb=short --disable-warnings`: 25 passed.
  - `venv\Scripts\python -m pytest tests\test_action_spec_evaluator.py tests\test_action_selection_prompt_contract.py tests\test_action_selection_payload.py tests\test_background_work_jobs.py tests\test_background_work_coding_agent.py tests\test_coding_agent_phase5_patch_apply_contracts.py tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase8_verify_repair_contracts.py tests\test_coding_agent_phase9_run_supervisor_contracts.py -q --tb=short --disable-warnings`: 84 passed.
  - A first Gate 02 retry with a 15-minute command timeout completed turn 1
    successfully and left the run `awaiting_approval`, but timed out before any
    turn-2 durable state was persisted. The two exact pytest processes for
    that Gate 02 command were stopped before rerun.
  - Clean rerun command:
    `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_02_source_free_proposal_with_revision_followups -q -s --tb=short`:
    passed in 1494.68 seconds.
  - Passing Gate 02 raw traces:
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_02_source_free_revision__20260709T013928050744Z.json`,
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_02_source_free_revision_turn_1__20260709T012941003669Z.json`,
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_02_source_free_revision_turn_2__20260709T013914783834Z.json`,
    and
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_02_source_free_revision_turn_3__20260709T013928049704Z.json`.
  - Gate 02 review appended to
    `development_plans/active/short_term/coding_agent_full_workflow_hardening_plan_2_llm_gate_reviews.md`.
  - Gate 03 command:
    `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_03_existing_source_proposal_with_runtime_only_followups -q -s --tb=short`:
    passed in 363.79 seconds.
  - Gate 03 raw traces:
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_03_existing_source_runtime_only.json`,
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_03_existing_source_runtime_only_turn_1.json`,
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_03_existing_source_runtime_only_turn_2.json`,
    and
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_03_existing_source_runtime_only_turn_3.json`.
  - Gate 03 preserved one run
    `coding_run:2a78d6e89569485e8a41b71caa472bb5`, kept all turns
    `awaiting_approval`, and narrowed revised changed files to
    `counter_cli/cli.py` with no apply, execution, or repair attempts.
  - Gate 04 first run command:
    `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_04_approval_verify_and_repair_followups -q -s --tb=short`:
    failed after all worker turns succeeded because final changed files still
    included `tests/test_slug.py`.
  - Gate 04 RCA: the start proposal included both `slug_tools/slug.py` and
    `tests/test_slug.py`; approval then applied the stored proposal and pytest
    succeeded, so no repair pass had a chance to enforce protected verification
    paths. The missing contract was approval-time filtering of initial
    proposal artifacts that touch pytest selector paths.
  - Added
    `tests/test_coding_agent_phase8_verify_repair_contracts.py::test_verify_repair_omits_initial_protected_verification_artifacts`.
    Red run before implementation showed the approved apply still received
    `TEST_ARTIFACT`; green run after implementation passed.
  - Implemented protected verification-path artifact omission in
    `src/kazusa_ai_chatbot/coding_agent/code_verifying/supervisor.py`.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_phase8_verify_repair_contracts.py -q --tb=short --disable-warnings`: 17 passed.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\coding_agent\code_verifying\supervisor.py`: passed.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_phase9_run_supervisor_contracts.py tests\test_coding_agent_phase9_e2e_workflows.py -q --tb=short --disable-warnings`: 17 passed.
  - Clean Gate 04 rerun command:
    `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_04_approval_verify_and_repair_followups -q -s --tb=short`:
    passed in 236.42 seconds.
  - Passing Gate 04 raw traces:
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_04_approval_verify_repair__20260709T015913674897Z.json`,
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_04_approval_verify_repair_turn_1__20260709T015844816846Z.json`,
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_04_approval_verify_repair_turn_2__20260709T015901719409Z.json`,
    and
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_04_approval_verify_repair_turn_3__20260709T015913673629Z.json`.
  - Gate 04 preserved one run
    `coding_run:04a5ee6a8b504117af79ee61aa4f412d`, used one managed apply
    attempt and one focused pytest execution attempt, omitted protected
    `tests/test_slug.py`, and summarized final changed files without `tests/`.
  - Gate 05 command:
    `venv\Scripts\python -m pytest -m live_llm tests\test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_05_hard_multifile_approval_history_cancel_status -q -s --tb=short`:
    passed in 395.93 seconds.
  - Gate 05 raw traces:
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status.json`,
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_1.json`,
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_2.json`,
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_3.json`,
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_4.json`,
    and
    `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_05_hard_multifile_status_turn_5.json`.
  - Gate 05 preserved one run
    `coding_run:6437ae63ef4b43cdb2cc5f9125af46db`, used one managed apply
    attempt and one focused pytest execution attempt, omitted protected tests
    `tests/test_cache.py` and `tests/test_cli.py`, projected attempt history,
    rejected late cancellation as terminal, and returned final status
    `completed`.
  - Gate 05 review appended to
    `development_plans/active/short_term/coding_agent_full_workflow_hardening_plan_2_llm_gate_reviews.md`.
- Stage 6 documentation evidence:
  - Updated `src/kazusa_ai_chatbot/background_work/README.md` with
    `revise_proposal`, `summarize`, attempt-history projection, allowed next
    actions, and protected pytest selector behavior.
  - Updated `src/kazusa_ai_chatbot/coding_agent/README.md` with durable worker
    operations, continuation actions, managed approved execution boundary, and
    protected verification paths.
  - Updated `src/kazusa_ai_chatbot/coding_agent/code_writing/README.md` with
    the one-pass review-materialization feedback contract.
  - Updated `docs/HOWTO.md` with durable run continuation actions and
    protected verification paths.
  - Updated
    `development_plans/reference/designs/coding_agent_architecture.md` with
    the full background operation set, source-free review feedback, and
    protected verification-path filtering.
  - Stale-contract grep for supports-only operation lists, start/status-only
    wording, validation-feedback-loop wording, and approve/cancel-only diagram
    wording: no matches in the updated ICD/HOWTO/reference docs.
  - `rg "revise_proposal|summarize|protected verification|review-materialization feedback|protected pytest|allowed next actions" ...`: matched the intended updated docs.
- Stage 7 confidence review evidence:
  - Re-read the implemented live gate contract in
    `tests/test_coding_agent_full_workflow_integration_live_llm.py`, the
    committed fixture manifest, the real LLM review artifact, and the
    production implementations for durable continuation, source-free
    review-materialization feedback, worker dispatch, action-spec
    materialization, and protected verification-path filtering.
  - Raw aggregate traces were inspected directly from `payload.turns`.
    Gate 01 used `start` only, completed one read-only run, and recorded zero
    apply, execution, or repair attempts. Gate 02 used `start`,
    `revise_proposal`, and `summarize` on
    `coding_run:876829b6929342e9b0c72fbf6247d961` with zero apply,
    execution, and repair attempts across all turns. Gate 03 used `start`,
    `revise_proposal`, and `summarize` on
    `coding_run:2a78d6e89569485e8a41b71caa472bb5`, narrowing final changed
    files to `counter_cli/cli.py`. Gate 04 used `start`,
    `approve_and_verify`, and `summarize` on
    `coding_run:04a5ee6a8b504117af79ee61aa4f412d`, with one managed apply,
    one focused pytest execution, zero repair attempts, and no final
    `tests/` changes. Gate 05 used `start`, `approve_and_verify`,
    `summarize`, `cancel`, and `status` on
    `coding_run:6437ae63ef4b43cdb2cc5f9125af46db`, preserved attempt
    history, omitted protected test changes, rejected the late cancel, and
    returned final status `completed`.
  - Fresh deterministic regression command:
    `venv\Scripts\python -m pytest tests\test_coding_agent_phase2_new_artifact_contracts.py tests\test_coding_agent_phase8_verify_repair_contracts.py tests\test_coding_agent_background_run_contracts.py tests\test_coding_agent_phase9_e2e_workflows.py tests\test_action_spec_evaluator.py tests\test_action_selection_prompt_contract.py tests\test_action_selection_payload.py tests\test_background_work_jobs.py tests\test_background_work_coding_agent.py tests\test_coding_agent_phase5_patch_apply_contracts.py tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase9_run_supervisor_contracts.py -q --tb=short --disable-warnings`:
    130 passed in 7.51 seconds.
  - Fresh static checks: direct durable-run API shortcut grep in the live gate
    file produced no matches; `rg "xfail|skip\("` matched only the LLM
    endpoint availability guard; fixture-root grep matched the committed
    fixture manifest; `py_compile` passed for all changed Python production
    modules.
  - Confidence rating: 92% first-time pass confidence for the final
    integration suite when run with the same local LLM configuration and a
    per-gate timeout long enough for Gate 02's observed 1494.68 second
    runtime. The score is above 90% because every restored gate has passed
    through the L2d/action-spec/background-worker entrypoint, the two observed
    gate failures were remediated with deterministic root-cause contracts, and
    the affected deterministic regressions now cover those failure modes.
    Residual risk remains local LLM latency or endpoint availability rather
    than a known missing coding-agent workflow contract.
- Stage 8 independent code review evidence:
  - Native independent review agent `019f44aa-0abd-7682-9115-10587d3a0894`
    reviewed the plan, diff, tests, fixtures, docs, and evidence. The review
    found two closure blockers:
    direct `continue_coding_run(..., action="status")` was documented and
    projected but rejected by the durable run supervisor, and
    `src/kazusa_ai_chatbot/action_spec/README.md` still listed only the older
    `start`, `status`, `approve_and_verify`, and `cancel` coding actions.
  - Remediated the first blocker by adding
    `tests/test_coding_agent_phase9_run_supervisor_contracts.py::test_status_continuation_projects_run_without_mutation`
    and implementing direct `status` continuation in
    `src/kazusa_ai_chatbot/coding_agent/coding_run/supervisor.py` as a
    public projection that does not mutate events or call proposal, apply,
    execution, or repair logic. The focused red run failed with
    `status_response["status"] == "rejected"` before implementation; the
    focused green run passed after implementation.
  - Remediated the second blocker by updating
    `src/kazusa_ai_chatbot/action_spec/README.md` so the action-spec ICD lists
    `start`, `revise_proposal`, `summarize`, `status`,
    `approve_and_verify`, and `cancel`, and states that revision, summary,
    status, approval, and cancellation require a prompt-safe
    `coding_run:<run_id>` reference.
  - Affected verification command:
    `venv\Scripts\python -m pytest tests\test_coding_agent_phase9_run_supervisor_contracts.py tests\test_coding_agent_phase9_e2e_workflows.py tests\test_coding_agent_background_run_contracts.py tests\test_background_work_coding_agent.py -q --tb=short --disable-warnings`:
    39 passed in 2.21 seconds.
  - Full deterministic regression command after review remediation:
    `venv\Scripts\python -m pytest tests\test_coding_agent_phase2_new_artifact_contracts.py tests\test_coding_agent_phase8_verify_repair_contracts.py tests\test_coding_agent_background_run_contracts.py tests\test_coding_agent_phase9_e2e_workflows.py tests\test_action_spec_evaluator.py tests\test_action_selection_prompt_contract.py tests\test_action_selection_payload.py tests\test_background_work_jobs.py tests\test_background_work_coding_agent.py tests\test_coding_agent_phase5_patch_apply_contracts.py tests\test_coding_agent_phase6_code_executing_contracts.py tests\test_coding_agent_phase9_run_supervisor_contracts.py -q --tb=short --disable-warnings`:
    131 passed in 7.50 seconds.
  - Final static checks: `git diff --check` reported no whitespace errors;
    direct durable-run API shortcut grep in the live gate file produced no
    matches; `rg "xfail|skip\("` matched only the LLM endpoint availability
    guard; stale operation-list grep across HOWTO, subsystem READMEs,
    action-spec ICD, and architecture docs produced no matches.
  - Review status: all independent review blockers were addressed inside the
    approved hardening scope, and no unresolved blockers remain.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Plan closes against narrowed gates again | Gate-to-test traceability matrix and independent review | Stage 1, static greps, independent plan/code review |
| Local LLM misroutes revision or summary follow-ups | Closed action enum, prompt-safe run refs, deterministic ref recovery | Deterministic action tests and restored Gates 02, 03, 05 |
| Proposal revision mutates state after approval | Allow revision only while `awaiting_approval` | Direct coding-run transition tests |
| Protected tests are edited to pass gates | Protected-path assertions and fixture manifest | Restored Gates 03, 04, 05 and review artifacts |
| Summary action hides missing attempt history | Summary projection must include attempts from ledger | Deterministic summary tests and Gates 04, 05 |
| Confidence is overstated from green pytest only | Required raw trace inspection and confidence review | Stage 7 closure gate |
