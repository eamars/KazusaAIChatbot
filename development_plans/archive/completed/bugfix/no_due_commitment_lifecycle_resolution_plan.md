# no-due active commitment lifecycle resolution bugfix plan

## Summary

- Goal: make active commitments without `due_at` resolvable after visible
  dialog proves fulfillment, without depending on RAG projection.
- Plan class: large.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `cjk-safety`,
  `test-style-and-execution`, and `debug-llm` for live LLM checks.
- Overall cutover strategy: additive post-turn direct lifecycle review. Existing
  pre-dialog lifecycle routing remains unchanged.
- Highest-risk areas: LLM latency, accidental consolidation action execution,
  silent partial review when many active commitments exist, and deterministic
  keyword cleanup over user or dialog text.
- Acceptance criteria: the direct post-turn review reaches the current user's
  no-due active commitments, chunks all reviewed rows into prompt-safe aliases,
  uses final visible dialog as evidence, iterates after successful lifecycle
  writes, closes the POC-proven tiramisu failure shape, leaves ambiguous dessert
  chatter open, removes POC artifacts before code review, and keeps
  consolidation as a trace consumer only.

## Context

The production diagnosis found stale active commitments around `提拉米苏`.
Fresh POC evidence under
`test_artifacts/no_due_commitment_lifecycle_poc/` showed:

- `23` active commitments for user
  `256e8a10-c406-47e9-ac8f-efd270d18160`.
- `23` of `23` active commitments had `due_at = null`.
- `8` active commitments contained `提拉米苏`.
- A single direct post-dialog review materialized `3` validated
  `apply_memory_lifecycle_update` specs.
- The same review over ambiguous future dessert wording materialized `0`
  specs.
- A bounded iterative dry run, with successful writes simulated in memory,
  closed all `8` active tiramisu rows over `4` productive passes and stopped
  with no remaining tiramisu rows on pass `5`.

The current code cannot produce that result:

- `query_active_commitment_memory_units(...)` in
  `src/kazusa_ai_chatbot/db/user_memory_units.py` requires string `due_at`;
  no-due commitments are invisible to self-cognition due checks.
- The live-chat memory lifecycle specialist in
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py` runs
  before final dialog exists.
- RAG-selected memory evidence and prompt-context active commitments are capped;
  relying on RAG projection cannot guarantee reachability for every active row.
- Consolidation sees final dialog, but the Action Spec ICD forbids it from
  selecting or executing lifecycle actions.

## Mandatory Skills

- `development-plan`: load before executing or revising this plan.
- `local-llm-architecture`: load before changing lifecycle prompts, graph
  timing, model calls, or post-turn orchestration.
- `no-prepost-user-input`: load before changing commitment interpretation
  logic; lifecycle decisions must remain LLM-owned.
- `py-style`: load before editing Python source.
- `cjk-safety`: load before editing Python files that contain CJK strings.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running or reviewing live LLM tests.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, and plan edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in
  `Execution Evidence`.
- Use parent-led native subagent execution. If native subagents are
  unavailable, stop before execution unless the user explicitly approves
  fallback execution.
- Do not add deterministic semantic filters over user input, final dialog, or
  commitment facts.
- Do not close commitments by age, missing `due_at`, keyword match, stale
  count, or topic string.
- Do not make consolidation select actions, execute actions, call the
  dispatcher, call the scheduler, or mutate lifecycle state directly.
- Do not expose raw `unit_id`, collection names, MongoDB query shape, handler
  ids, action attempts, or lifecycle write internals to the LLM.
- Do not block visible response delivery. The new review runs only after
  `_chat_input_queue.complete(...)`.
- Do not let `think_only` hidden dialog close commitments as visible dialog.
- Respect `no_remember`: skip post-turn lifecycle writes when `no_remember` is
  active.
- If an executed `apply_memory_lifecycle_update` already exists in the turn's
  `action_results`, skip the post-turn review for that turn.
- Remove `test_artifacts/no_due_commitment_lifecycle_poc/` after implementation
  verification and before independent code review.

## Must Do

- Add a direct DB reader for the current user's active commitment rows without
  requiring `due_at`.
- Feed the post-turn lifecycle specialist from that direct reader, not from RAG
  memory projection.
- Chunk direct-reader rows into prompt-safe aliases of at most
  `ACTIVE_COMMITMENT_ALIAS_LIMIT` rows per LLM call.
- Include `final_dialog` and user-visible text surface fragments in the
  post-turn specialist payload.
- Execute returned `apply_memory_lifecycle_update` specs through
  `execute_action_specs_for_trace(...)`.
- Re-query active commitments after each productive pass and repeat until no
  new lifecycle actions execute or `POST_SURFACE_LIFECYCLE_MAX_PASSES` is
  reached.
- Rebuild `action_specs`, `action_results`, and `episode_trace` in the state
  passed to conversation progress, consolidation, and residue recording.
- Add deterministic and live LLM tests that prove no-due tiramisu final-dialog
  closure and ambiguous dessert non-closure.
- Remove POC artifacts before independent code review.

## Deferred

- Do not repair production data in this bugfix.
- Do not add a broad stale-commitment sweeper.
- Do not route no-due commitments through self-cognition due checks.
- Do not increase RAG retrieval budgets to solve lifecycle reachability.
- Do not rewrite the memory-unit extractor to forbid no-due active
  commitments.
- Do not add a feature flag, fallback path, compatibility shim, or alternate
  lifecycle execution path.

## Cutover Policy

Overall strategy: additive bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Post-turn lifecycle reachability | bigbang | Use direct active-commitment DB reads for post-turn lifecycle review. Do not use RAG projection as the reachability source. |
| Pre-dialog lifecycle route | compatible | Keep existing `memory_lifecycle_update` route behavior unchanged. |
| Consolidation | bigbang | Keep consolidation as trace consumer only. Do not give it action selection or execution behavior. |
| POC artifacts | bigbang | Delete `test_artifacts/no_due_commitment_lifecycle_poc/` before independent code review. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- Any change to a cutover policy requires user approval before implementation.

## Target State

After a visible response is completed, the service performs a bounded
post-turn lifecycle finalization:

```text
persona graph returns completed state
  -> assistant visible response is persisted
  -> _chat_input_queue.complete(...) releases the response
  -> post-turn lifecycle helper directly reads active commitments for state["global_user_id"]
  -> helper chunks all returned rows into alias batches
  -> memory lifecycle specialist judges each batch using final_dialog evidence
  -> deterministic code resolves trusted aliases into apply_memory_lifecycle_update
  -> execute_action_specs_for_trace(...) performs audited lifecycle writes
  -> helper re-queries active commitments for the next pass
  -> updated state flows into conversation progress, consolidation, and residue
```

This gives the original failure mode a reliable path because reachability no
longer depends on `due_at`, self-cognition due checks, or RAG-selected memory
evidence.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Reachability source | Query active commitments directly by `global_user_id`, `unit_type`, and `status`. | POC proved RAG-independent reachability is required. |
| Semantic owner | Keep lifecycle status decisions inside the memory lifecycle specialist LLM. | Avoid deterministic keyword closure over user or dialog text. |
| DB owner | Keep lifecycle writes inside `apply_memory_lifecycle_update` execution. | Preserves Action Spec audit and repository validation. |
| Iteration | Re-query after productive passes, capped at `POST_SURFACE_LIFECYCLE_MAX_PASSES = 5`. | POC needed 4 productive passes and a 5th stop pass to clear all 8 tiramisu rows. |
| Query saturation | Fetch `POST_SURFACE_ACTIVE_COMMITMENT_REVIEW_LIMIT + 1`; if exceeded, skip review and record a warning context. | Prevents silent partial review while keeping a bounded background workload. |
| POC artifacts | Remove POC directory before code review. | POC is proof evidence, not deliverable source or permanent test fixture. |

## Contracts And Data Shapes

### Direct Reader

Create in `src/kazusa_ai_chatbot/db/user_memory_units.py`:

```python
async def query_active_commitment_memory_units_for_user(
    *,
    global_user_id: str,
    limit: int,
) -> dict[str, object]:
    ...
```

Return shape:

```python
{
    "documents": list[UserMemoryUnitDoc],
    "limit": int,
    "limit_exceeded": bool,
}
```

Rules:

- Query only `global_user_id`, `unit_type == "active_commitment"`, and
  `status == "active"`.
- Exclude `_id` and `embedding`.
- Sort by `due_at` ascending, then `updated_at` descending, then `unit_id`
  ascending.
- Fetch `limit + 1` rows and set `limit_exceeded` when more rows exist than
  the fixed limit.

### Post-Surface Specialist Entrypoint

Create in `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py`:

```python
def prepare_post_surface_memory_lifecycle_review(
    state: dict,
    active_commitment_units: list[dict[str, object]],
) -> dict[str, object]:
    ...

async def call_post_surface_memory_lifecycle_review(
    state: dict,
    active_commitment_units: list[dict[str, object]],
) -> dict[str, object]:
    ...
```

Rules:

- Build aliases only from `active_commitment_units`.
- Never read RAG memory candidates for post-surface review.
- Set prompt-row `due_state` to `"no_due_date"` when `due_at` is missing.
- Include `final_dialog` and projected user-visible text surfaces.
- Return only executable `apply_memory_lifecycle_update` specs.
- Return `{}` when no alias bindings or no visible final dialog exist.

### Post-Turn Helper

Create in `src/kazusa_ai_chatbot/brain_service/post_turn.py`:

```python
POST_SURFACE_ACTIVE_COMMITMENT_REVIEW_LIMIT = 500
POST_SURFACE_LIFECYCLE_MAX_PASSES = 5

async def run_post_turn_memory_lifecycle_background(
    state: dict,
    *,
    active_commitment_reader: ActiveCommitmentReader,
    review_func: PostSurfaceMemoryLifecycleReview,
    execute_action_specs_func: ExecuteActionSpecsForTrace,
    logger: logging.Logger,
    no_remember: bool,
    visible_response_sent: bool,
    think_only_suppressed: bool,
) -> dict:
    ...
```

Rules:

- Return the original state unchanged for skip conditions.
- Return a shallow updated copy when appending post-turn lifecycle evidence.
- Execute only `apply_memory_lifecycle_update` specs.
- Append new action specs and action results to existing state lists.
- Rebuild `episode_trace` with `build_episode_trace(...)`.
- Stop after a pass that produces no executed lifecycle results.
- Stop after `POST_SURFACE_LIFECYCLE_MAX_PASSES`.

## LLM Call And Context Budget

| Call | Before | After | Blocking behavior | Cap |
|---|---|---|---|---|
| Pre-dialog lifecycle specialist | 0-1 response-path calls when L2d selects `memory_lifecycle_update` | unchanged | response path | existing alias cap |
| Post-turn lifecycle specialist | none | 0 to `ceil(active_commitments / 12) * 5` background calls after visible response completion | does not block response delivery; runs before progress/consolidation evidence consumers | 500 active commitments, 12 aliases per call, 5 passes |

Each post-turn prompt contains at most 12 commitment prompt rows plus bounded
final-dialog and visible-surface fragments. If the direct reader reports
`limit_exceeded`, the helper records a warning and returns the original state
unchanged. No model call receives raw ids, collection names, or DB query shape.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/db/user_memory_units.py`
  - Add the direct current-user active-commitment reader.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py`
  - Add post-surface preparation and review entrypoints using supplied DB rows.
  - Factor the shared specialist invocation without changing pre-dialog public
    behavior.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`
  - Add the bounded iterative post-turn helper.
- `src/kazusa_ai_chatbot/service.py`
  - Wire the helper after `_chat_input_queue.complete(...)` and before
    progress, consolidation, and residue recording.
- `src/kazusa_ai_chatbot/nodes/README.md`
  - Document the post-surface lifecycle boundary.
- `tests/test_memory_lifecycle_specialist.py`
  - Add deterministic post-surface review tests.
- `tests/test_service_background_consolidation.py`
  - Add post-turn ordering, skip, and iteration tests.
- `tests/test_memory_lifecycle_specialist_live_llm.py`
  - Add the live no-due final-dialog tiramisu regression.

### Create

- `tests/test_user_memory_units_active_commitment_reader.py`
  - Add deterministic DB-reader tests with a fake collection/cursor.

### Delete During Cleanup

- `test_artifacts/no_due_commitment_lifecycle_poc/`
  - Delete after implementation verification and before independent code
    review.

### Keep

- `src/kazusa_ai_chatbot/consolidation/**`
  - No consolidation behavior changes.
- `src/kazusa_ai_chatbot/self_cognition/**`
  - No due-check behavior changes for no-due commitments.
- RAG retrieval budgets and memory-unit extraction prompts.

## Overdesign Guardrail

- Actual problem: no-due active commitments can remain active after final
  visible dialog clearly fulfills them because the reliable lifecycle paths
  cannot see both the rows and the final dialog.
- Minimal change: add one post-turn direct-reader lifecycle review that reuses
  the existing specialist and executable action path.
- Ownership boundaries: the LLM judges lifecycle semantics; deterministic code
  owns DB reads, alias binding, validation, execution, iteration caps, and
  trace rebuilding; consolidation consumes the final trace.
- Rejected complexity: no stale sweeper, no keyword matcher, no RAG budget
  increase, no feature flag, no compatibility shim, no consolidation executor,
  no scheduler/self-cognition reroute, and no data-repair migration.
- Evidence threshold: add future complexity only after a separate POC or
  production trace proves this direct-reader path cannot handle a named
  lifecycle class within its fixed caps.

## Agent Autonomy Boundaries

- The responsible agent must implement the exact public functions, constants,
  file paths, tests, and verification commands named in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, feature flags, or
  extra lifecycle tools.
- The responsible agent must not change files outside `Change Surface`.
- If the plan and source disagree, preserve the plan's stated ownership
  boundary and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Parent adds the focused reader, specialist, and post-turn helper tests and
   records expected failures.
2. Parent starts exactly one production-code subagent with this plan and the
   focused test contract.
3. Production-code subagent implements production files in this order:
   `db/user_memory_units.py`, `nodes/persona_supervisor2_memory_lifecycle.py`,
   `brain_service/post_turn.py`, `service.py`, `nodes/README.md`.
4. Parent runs focused deterministic tests and fixes only in-scope failures.
5. Parent adds or finalizes the live LLM regression and inspects the trace.
6. Parent runs focused verification.
7. Parent removes POC artifacts.
8. Parent starts exactly one independent code-review subagent.
9. Parent remediates review findings only inside the approved change surface
   and reruns affected verification.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only;
  closes after planned production changes are complete.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes and POC artifacts are removed; reviews the plan,
  diff, and evidence; reports findings to the parent; does not implement
  fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly approves fallback execution.

## Progress Checklist

- [x] Stage 1 - focused test contract established
  - Covers: reader, post-surface specialist, post-turn helper, service
    ordering, and live LLM regression tests.
  - Verify: run each new focused test and record the expected failure or
    baseline.
  - Evidence: record commands and failures in `Execution Evidence`.
  - Handoff: production-code subagent starts at Stage 2.
  - Sign-off: `Codex / 2026-05-30`.
- [x] Stage 2 - production implementation complete
  - Covers: direct reader, post-surface specialist entrypoint, iterative
    post-turn helper, service wiring, and nodes README update.
  - Verify: `py_compile` command in `Verification` succeeds.
  - Evidence: record changed production files and compile output.
  - Handoff: parent runs Stage 3.
  - Sign-off: `Codex / 2026-05-30`.
- [x] Stage 3 - deterministic verification complete
  - Covers: focused deterministic tests and regression tests.
  - Verify: deterministic pytest command in `Verification` passes.
  - Evidence: record test output.
  - Handoff: parent runs Stage 4.
  - Sign-off: `Codex / 2026-05-30`.
- [x] Stage 4 - live LLM verification complete
  - Covers: no-due final-dialog tiramisu live regression.
  - Verify: live LLM pytest command in `Verification` passes and the trace is
    inspected.
  - Evidence: record command, trace path, selected alias summary, and prompt-id
    leak check.
  - Handoff: parent runs Stage 5.
  - Sign-off: `Codex / 2026-05-30`.
- [x] Stage 5 - POC artifacts removed
  - Covers: deletion of `test_artifacts/no_due_commitment_lifecycle_poc/`.
  - Verify: cleanup command and `Test-Path` check in `Verification`.
  - Evidence: record cleanup command and `git status --short`.
  - Handoff: independent code-review subagent starts at Stage 6.
  - Sign-off: `Codex / 2026-05-30`.
- [x] Stage 6 - independent code review complete
  - Covers: plan alignment, full diff, tests, live trace, and cleanup evidence.
  - Verify: review subagent reports no unresolved blockers.
  - Evidence: record findings, fixes, rerun commands, and approval status.
  - Handoff: parent completes final sign-off.
  - Sign-off: `Codex / 2026-05-30`.

## Verification

### Static Compile

```powershell
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/db/user_memory_units.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py src/kazusa_ai_chatbot/brain_service/post_turn.py src/kazusa_ai_chatbot/service.py
```

Expected: exit code 0 and no output.

### Deterministic Tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_user_memory_units_active_commitment_reader.py tests\test_memory_lifecycle_specialist.py tests\test_service_background_consolidation.py tests\test_service_input_queue.py tests\test_action_spec_memory_lifecycle.py tests\test_action_spec_results.py -q
```

Expected: pass.

### Live LLM Regression

```powershell
venv\Scripts\python.exe -m pytest tests\test_memory_lifecycle_specialist_live_llm.py::test_live_tiramisu_no_due_final_dialog_fulfilled -q -s -m live_llm
```

Expected: pass. If the configured LLM endpoint is unavailable, record
verification as blocked and leave the plan incomplete. Inspect the trace for
correct alias selection and no raw `unit_id` leak in the prompt.

### POC Cleanup

```powershell
Remove-Item -LiteralPath 'test_artifacts/no_due_commitment_lifecycle_poc' -Recurse -Force
if (Test-Path -LiteralPath 'test_artifacts/no_due_commitment_lifecycle_poc') { throw 'POC artifact directory still exists' }
git status --short
```

Expected: the POC directory is absent. `git status --short` must not show POC
artifacts.

### Read-Only DB Inspection

```powershell
venv\Scripts\python.exe -m scripts.export_collection user_memory_units --filter '{"global_user_id":"256e8a10-c406-47e9-ac8f-efd270d18160","unit_type":"active_commitment","status":"active"}' --sort '{"updated_at":-1}' --limit 500 --output test_artifacts/no_due_commitment_lifecycle_post_deploy_user_memory_units.json
```

Expected: read-only export succeeds. Inspect the JSON and record active
no-due commitment counts. Do not perform data repair in this task.

## Independent Plan Review

Review date: 2026-05-30.

Review scope:

- Current plan draft after POC update.
- `development_plans/README.md`, plan contract, cutover policy, and execution
  gates.
- `src/kazusa_ai_chatbot/nodes/README.md` and
  `src/kazusa_ai_chatbot/action_spec/README.md`.
- Source contracts in `db/user_memory_units.py`,
  `nodes/persona_supervisor2_memory_lifecycle.py`,
  `brain_service/post_turn.py`, `service.py`, and
  `action_spec/execution.py`.
- POC summary from `poc_trace.json` and `poc_iterative_trace.json`.

Findings fixed in this revision:

- Blocker: prior plan still depended on state/RAG-surfaced active commitments.
  Fix: post-turn reachability now comes from a direct DB reader scoped to the
  current `global_user_id`.
- Blocker: prior plan did not handle the existing lifecycle materialization
  cap; POC showed a single pass can leave selected rows unresolved. Fix:
  post-turn helper now requires bounded iterative passes with DB re-query.
- Blocker: prior plan did not prevent silent partial review if the direct read
  exceeded the background cap. Fix: reader returns `limit_exceeded`, and the
  helper records a warning and skips instead of partial-closing.
- Blocker: prior plan did not require POC cleanup. Fix: Stage 5 and
  `Verification` require deleting `test_artifacts/no_due_commitment_lifecycle_poc/`
  before independent code review.
- Blocker: prior plan allowed implementation creativity around helper names,
  files, and execution seams. Fix: exact function names, constants, files, and
  injected callable seams are now specified.

Approval status: approved by user on 2026-05-30. No remaining plan-review
blockers.

## Independent Code Review

Run this gate after all `Verification` commands pass and after POC artifacts
are removed. The parent agent must create one independent code-review subagent
through the current harness's native subagent capability. If native subagents
are unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Full implementation diff against this plan.
- Mandatory skill compliance for Python, tests, CJK strings, LLM prompts, and
  live LLM evidence.
- Ownership boundaries: no RAG reachability dependency, no consolidation
  execution, no deterministic keyword closure, no raw id leak to prompts.
- Iteration correctness: direct reader, chunking, re-query, pass cap, skip
  conditions, action execution, trace rebuilding, and state ordering.
- Verification evidence, live trace inspection, read-only DB inspection, and
  POC cleanup evidence.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface. If a finding requires a new contract, fallback,
module, migration, or behavior outside this plan, stop and request approval
before changing code.

## Acceptance Criteria

This plan is complete when:

- Post-turn lifecycle review directly reads active commitments for the current
  user without requiring `due_at`.
- The post-turn specialist prompt receives all reviewed commitments through
  prompt-safe aliases, final visible dialog, and user-visible text surfaces.
- No raw `unit_id`, collection name, action attempt, or DB query shape appears
  in LLM prompt payloads.
- The tiramisu no-due final-dialog case materializes and executes
  `apply_memory_lifecycle_update` actions through the existing action-spec
  path.
- Ambiguous dessert chatter produces no lifecycle actions.
- Conversation progress, consolidation, and residue receive the updated
  post-turn state after lifecycle execution.
- POC artifacts are removed before independent code review.
- Focused deterministic tests, live LLM regression, compile check, read-only DB
  inspection, and independent code review all pass with evidence recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Added background LLM calls increase post-turn time | Run after response completion, use 12-row chunks, 500-row cap, 5-pass cap | service ordering test and live trace |
| Too many active commitments cause silent partial review | Fetch `limit + 1` and skip on `limit_exceeded` | reader test and post-turn skip test |
| Model over-closes from playful dialog | Keep semantic judgment in specialist; add ambiguous dessert negative tests | deterministic and live LLM tests |
| Consolidation gains action authority by accident | Keep execution in post-turn action path only | code review and consolidation tests |

## Execution Evidence

Record execution evidence here only after the plan is approved and run:

```text
git status --short:
  M development_plans/README.md
  M tests/test_memory_lifecycle_specialist.py
  M tests/test_memory_lifecycle_specialist_live_llm.py
  M tests/test_service_background_consolidation.py
  ?? development_plans/active/bugfix/no_due_commitment_lifecycle_resolution_plan.md
  ?? tests/test_user_memory_units_active_commitment_reader.py

focused expected failures:
  venv\Scripts\python.exe -m pytest tests\test_user_memory_units_active_commitment_reader.py -q
    FAILED 2: missing query_active_commitment_memory_units_for_user.

  venv\Scripts\python.exe -m pytest tests\test_memory_lifecycle_specialist.py::test_post_surface_review_uses_final_dialog_and_direct_rows tests\test_memory_lifecycle_specialist.py::test_post_surface_review_leaves_ambiguous_dessert_open -q
    FAILED 2: missing call_post_surface_memory_lifecycle_review.

  venv\Scripts\python.exe -m pytest tests\test_service_background_consolidation.py::test_post_turn_lifecycle_iterates_after_productive_passes tests\test_service_background_consolidation.py::test_post_turn_lifecycle_skips_structural_blockers tests\test_service_background_consolidation.py::test_chat_runs_post_turn_lifecycle_before_progress_and_consolidation -q
    FAILED 3: missing run_post_turn_memory_lifecycle_background and service hook.

  venv\Scripts\python.exe -m pytest tests\test_memory_lifecycle_specialist_live_llm.py::test_live_tiramisu_no_due_final_dialog_fulfilled -q -s -m live_llm
    ERROR during collection: missing call_post_surface_memory_lifecycle_review import, before LLM call.

production implementation summary:
  Production worker 019e771e-a4b8-7d20-ac8d-81454b44c64b completed and was
  closed. Changed files:
    src/kazusa_ai_chatbot/db/user_memory_units.py
    src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py
    src/kazusa_ai_chatbot/brain_service/post_turn.py
    src/kazusa_ai_chatbot/service.py
    src/kazusa_ai_chatbot/nodes/README.md

  Summary:
    Added direct current-user active-commitment reader without due_at filtering.
    Added post-surface lifecycle review from direct rows plus final visible
    dialog and user-visible surface text.
    Added bounded post-turn lifecycle loop with direct re-query and trace
    rebuild.
    Wired service execution after response completion and before progress,
    consolidation, and residue.

static compile:
  venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\db\user_memory_units.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_memory_lifecycle.py src\kazusa_ai_chatbot\brain_service\post_turn.py src\kazusa_ai_chatbot\service.py
    PASSED: exit code 0, no output.

  After independent-review remediation:
  venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\db\__init__.py src\kazusa_ai_chatbot\db\user_memory_units.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_memory_lifecycle.py src\kazusa_ai_chatbot\brain_service\post_turn.py src\kazusa_ai_chatbot\service.py
    PASSED: exit code 0, no output.

focused deterministic tests:
  venv\Scripts\python.exe -m pytest tests\test_user_memory_units_active_commitment_reader.py tests\test_memory_lifecycle_specialist.py tests\test_service_background_consolidation.py tests\test_service_input_queue.py tests\test_action_spec_memory_lifecycle.py tests\test_action_spec_results.py -q
    PASSED: 82 passed in 2.84s.

  After independent-review remediation:
  venv\Scripts\python.exe -m pytest tests\test_user_memory_units_active_commitment_reader.py tests\test_memory_lifecycle_specialist.py tests\test_service_background_consolidation.py tests\test_service_input_queue.py tests\test_action_spec_memory_lifecycle.py tests\test_action_spec_results.py -q
    PASSED: 82 passed in 2.87s.

  Remediation spot checks:
  venv\Scripts\python.exe -m pytest tests\test_service_background_consolidation.py::test_chat_queues_background_consolidation_for_mapping_state tests\test_service_background_consolidation.py::test_chat_runs_post_turn_lifecycle_before_progress_and_consolidation tests\test_service_input_queue.py::test_worker_saves_dropped_messages_before_next_graph -q
    PASSED: 3 passed in 2.09s.

  git diff --check
    PASSED: no whitespace errors; line-ending warnings only.

live LLM test and trace inspection:
  venv\Scripts\python.exe -m pytest tests\test_memory_lifecycle_specialist_live_llm.py::test_live_tiramisu_no_due_final_dialog_fulfilled -q -s -m live_llm
    PASSED: 1 passed in 4.43s.
    Trace: test_artifacts/llm_traces/memory_lifecycle_specialist_live_llm__tiramisu_no_due_final_dialog_fulfilled.json
    Inspection: lifecycle_decisions[0] selected target_alias=commitment_1,
    decision=fulfilled; apply_unit_ids=["unit-tiramisu"].

  Prompt leak check:
    Reconstructed prepare_post_surface_memory_lifecycle_review prompt payload
    for the same no-due tiramisu case.
    Result: leaked=[], aliases=["commitment_1"], due_state="no_due_date".

  After independent-review remediation:
  venv\Scripts\python.exe -m pytest tests\test_memory_lifecycle_specialist_live_llm.py::test_live_tiramisu_no_due_final_dialog_fulfilled -q -s -m live_llm
    PASSED: 1 passed in 4.39s.
    Trace: test_artifacts/llm_traces/memory_lifecycle_specialist_live_llm__tiramisu_no_due_final_dialog_fulfilled__20260530T044744485848Z.json
    Inspection: lifecycle_decisions[0] selected target_alias=commitment_1,
    decision=fulfilled; apply_unit_ids=["unit-tiramisu"].
    Prompt leak check: leaked=[], aliases=["commitment_1"],
    due_state="no_due_date".

POC cleanup:
  Remove-Item -LiteralPath 'test_artifacts/no_due_commitment_lifecycle_poc' -Recurse -Force
  if (Test-Path -LiteralPath 'test_artifacts/no_due_commitment_lifecycle_poc') { throw 'POC artifact directory still exists' }
  git status --short
    PASSED: POC directory absent from git status.
    Status showed only plan, production, and test files; no POC artifacts.

read-only DB inspection:
  venv\Scripts\python.exe -m scripts.export_collection user_memory_units --filter '{"global_user_id":"256e8a10-c406-47e9-ac8f-efd270d18160","unit_type":"active_commitment","status":"active"}' --sort '{"updated_at":-1}' --limit 500 --output test_artifacts/no_due_commitment_lifecycle_post_deploy_user_memory_units.json
    PASSED: wrote 23 document(s) to test_artifacts/no_due_commitment_lifecycle_post_deploy_user_memory_units.json.
    Summary: exported_record_count=23, no_due_active=23, tiramisu_active=8.
    Interpretation: current DB still has unresolved no-due active commitments;
    this task adds the reliable future post-turn closure path and does not run
    data repair.

independent code review:
  First independent review subagent:
    019e772c-756f-7983-abda-6ec7ed75ea90
    Findings: two blockers.
      1. Deterministic service tests could hit real post-turn DB path.
      2. service.py imported the new DB reader from the DB submodule instead
         of the public kazusa_ai_chatbot.db facade.

  Remediation:
    Re-exported query_active_commitment_memory_units_for_user from
    src/kazusa_ai_chatbot/db/__init__.py and changed service.py to import via
    kazusa_ai_chatbot.db.
    Stubbed _run_post_turn_memory_lifecycle_background in deterministic
    service test helpers by default, while the explicit ordering test opts out.

  Second independent review subagent:
    019e7736-3f49-7362-9503-9605655b6585
    Result: No blockers found.
    Prior blockers confirmed fixed. Mandatory checks confirmed direct no-due
    reader, prompt-safe post-surface alias payload, existing executor-only
    lifecycle execution, and ordering after response completion before
    progress/consolidation/residue.
```
