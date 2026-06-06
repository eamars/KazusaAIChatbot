# l2d router split and background acknowledgement hardening plan

## Summary

- Goal: Reduce L2d overload by splitting route selection from specialist
  payload shaping, and fix the observed background artifact acknowledgement
  invariant so private background jobs cannot proceed without a valid visible
  handoff path.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`, `py-style`, `cjk-safety`,
  `test-style-and-execution`
- Overall cutover strategy: compatible internal refactor. Existing public
  service APIs, adapter contracts, background artifact job rows, and result
  delivery contracts remain compatible.
- Highest-risk areas: L2d route regression, resolver/action boundary drift,
  accidental prompt overfitting to Stage 0 cases, silent background enqueue
  without acknowledgement, goal-progress continuity regression, and extra
  live-response latency.
- Acceptance criteria: L2d is reduced to a top-level semantic router,
  background artifact intake owns `work_kind` and objective shaping, resolver
  goal progress is not forced for ordinary social chat, and focused tests plus
  one-case live LLM evidence show no silent background artifact enqueue.

## Context

The completed background artifact handoff plan proved the first production
path, but the Stage 0 input quality review exposed a broader L2d contract
problem:

```text
test_artifacts/background_artifact_handoff/20260606T_stage0_input_quality/quality_review.md
```

The important observed behavior is not a single missing field. L2d is currently
asked to choose resolver requests, choose action requests, maintain full
resolver goal progress, handle pending continuation, route private actions,
route background artifact work, and preserve visible speech behavior in one
local-LLM call. The local model can usually make the right coarse route, but
raw output shows contract pressure: malformed `speak` actions, invalid
goal-progress fields, resolver/action confusion, and implementation vocabulary
inside user-facing clarification objectives.

The user explicitly rejected prompt tuning that targets this fixture-specific
failure mode. This plan therefore addresses only the prompt logic and system
ownership weakness: L2d must become a top-level semantic router, and detailed
domain extraction must move to the owning specialist.

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, updating,
  or signing off this plan.
- `local-llm-architecture`: load before changing L2d, resolver,
  background-artifact intake, graph routing, or any prompt input/output
  contract.
- `no-prepost-user-input`: load before changing logic that accepts, rejects,
  persists, or acts on user requests.
- `debug-llm`: load before live/local LLM runs, prompt comparison, or quality
  review artifacts.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files that contain CJK strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- The `Execution Model` must use parent-led native subagent execution unless
  the user explicitly approves fallback execution.
- Plan status is not production-code authorization. Production edits require
  explicit user approval after this draft is reviewed.
- Do not tune prompt wording to memorize the nine Stage 0 cases.
- Do not add deterministic keyword routing or post-processing over raw user
  text to rescue LLM route decisions.
- LLM stages own semantic judgment only. Deterministic code owns validation,
  persistence, idempotency, adapter delivery feasibility, delivery status,
  permissions, retries, and invariant enforcement.
- L2d must not receive a detailed worker registry, DB schema, adapter fields,
  job leases, retry state, or delivery mechanics.
- Background artifact intake may classify only bounded text artifact work.
  It must reject filesystem writes, shell, package install, repository
  mutation, test execution, downloads, web research, images, attachments, and
  chunked delivery.
- Live/local LLM checks must run one case at a time and be inspected one case
  at a time.

## Must Do

- Split L2d into a top-level semantic route contract that chooses only route
  families and immediate visible-surface need.
- Move background artifact `work_kind`, objective shaping, and input-readiness
  judgment to a background artifact intake specialist.
- Stop requiring full `resolver_goal_progress` for ordinary social turns that
  do not have a concrete user goal, pending resolver row, resolver observation,
  or background artifact job.
- Preserve existing resolver goal progress for real multi-turn goals and
  pending resolver flows.
- Add deterministic invariant enforcement: a background artifact job must not
  be durably enqueued unless the current turn also has a valid visible handoff
  route or an explicitly approved no-visible-handoff state. This plan's first
  scope uses the visible handoff route only.
- Keep all user-visible wording in L3/dialog. L2d and intake produce semantic
  requirements, not final text.
- Preserve the existing `background_artifact_request` capability and durable
  job store.
- Preserve first-scope artifact work kinds:
  `coding_snippet`, `text_rewrite`, and `summary`.

## Deferred

- Do not implement multi-agent coding orchestration.
- Do not add filesystem-editing coding work, shell execution, package install,
  test execution, downloads, web research, images, attachments, or chunked
  delivery.
- Do not redesign the whole cognition resolver loop.
- Do not remove existing resolver pending HIL or approval behavior.
- Do not add retry prompts, repair prompts, fallback LLM calls, compatibility
  shims, or keyword fallback routers.
- Do not change adapter protocols, dispatcher delivery contracts, calendar
  scheduler, proactive output, reflection, RAG retrieval ownership, or
  consolidator write routing.
- Do not solve `delivery_in_progress` crash recovery in this plan; it requires
  a separate delivery-lease recovery plan if prioritized.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| L2d output contract | migration | Introduce the route-only contract behind focused tests, then remove old detailed responsibility once callers are wired. |
| Background artifact queue | compatible | Preserve existing job row shape and queue facade. Intake changes only the semantic request materialization before enqueue. |
| Resolver goal progress | compatible | Preserve full progress for resolver goals; skip or minimize it only for non-goal social turns. |
| Visible text surface | compatible | Keep L3/dialog as the only user-visible wording owner. |
| Existing live APIs | compatible | Do not change `/chat`, adapter, dispatcher, or result-ready delivery API contracts. |

## Target State

The target response-path ownership is:

```text
L1/L2 cognition
  -> L2d top-level router
       route_family:
         speak_now | resolver_needed | private_action_needed |
         background_text_artifact_candidate | no_action
       visible_handoff_required: bool
  -> route owner
       resolver handles evidence/HIL/approval/self-resolution
       background_artifact_intake handles work_kind/objective/input readiness
       memory lifecycle specialist handles commitment lifecycle
  -> deterministic validators and graph wiring
  -> L3/dialog owns visible wording
```

For background artifacts:

```text
L2d:
  declares accepted bounded background text artifact candidate
  declares visible handoff is required

background_artifact_intake:
  classifies work_kind
  extracts objective and input summary from model-visible context
  rejects missing input or unsupported scope

deterministic graph/action layer:
  enforces paired visible handoff before durable enqueue
  validates target scope and queue params
  persists the job

L3:
  acknowledges only prompt-safe pending enqueue result
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| L2d role | L2d becomes a top-level semantic router. | Reduces one-call cognitive load and keeps capability selection close to character judgment. |
| Background details | Move `work_kind` and objective shaping to `background_artifact_intake`. | The background subsystem owns text artifact scope and can reject out-of-domain work without bloating L2d. |
| Goal progress | Keep full `resolver_goal_progress` only for real user goals or resolver continuation. | Ordinary chat should not be forced into deliverable tracking. |
| Acknowledgement invariant | Deterministic graph/action code blocks background enqueue when no valid visible handoff route exists. | Prevents silent operational commitments caused by malformed LLM output. |
| Prompt fix style | Rewrite contracts around ownership, not fixture-specific failures. | Avoids overfitting to missing `speak.reason` or Stage 0 nouns. |
| Worker boundary | Keep the existing monolithic worker unchanged unless intake contract requires a prompt-safe payload update. | This plan targets routing/intake overload, not artifact generation quality. |

## Contracts And Data Shapes

### L2d Router Output

L2d returns route intent, not detailed executable action params:

```python
{
    "resolver_capability_requests": list[ResolverCapabilityRequestV1],
    "route_requests": [
        {
            "route_family": (
                "speak_now"
                " | private_action"
                " | background_text_artifact_candidate"
                " | no_action"
            ),
            "decision": str,
            "detail": str,
            "reason": str,
            "visible_handoff_required": bool,
        }
    ],
    "resolver_pending_resolution": ResolverPendingResolutionV1 | None,
    "resolver_goal_progress": ResolverGoalProgressV1 | None,
}
```

`resolver_goal_progress` is present only when the turn has a concrete user
goal, pending resolver state, resolver observations, or an in-progress
deliverable. Ordinary social chat may omit it.

### Background Artifact Intake Output

The intake specialist returns a prompt-safe semantic request:

```python
{
    "status": "accepted | rejected | needs_user_input",
    "work_kind": "coding_snippet | text_rewrite | summary | none",
    "objective": str,
    "input_summary": str,
    "visible_handoff_detail": str,
    "reason": str,
}
```

The intake specialist must not output target ids, platform ids, job ids,
adapter names, delivery status, lease state, retries, DB fields, filesystem
paths, or final user-visible wording.

### Deterministic Background Handoff Invariant

Before enqueue, deterministic code verifies:

```text
background artifact accepted
AND valid visible handoff route exists in the same cognition result
AND target scope is complete
AND action params pass bounded background artifact validation
```

If the invariant fails, no durable job is created. The graph records a
prompt-safe rejection result for L3 or consolidation.

## LLM Call And Context Budget

Default cap: 50k tokens, estimated as characters divided by four.

| Call | Before | After | Response path | Context policy |
|---|---|---|---|---|
| L2d action initializer | 1 call; route, action params, resolver, goal progress | 1 call; route family, resolver request, pending resolution, limited goal progress | yes | Shorter output contract; stable system prompt; current dynamic action context only. |
| Background artifact intake | none | 0-1 call only when L2d selects `background_text_artifact_candidate` | yes, but only on accepted artifact candidate path | Receives prompt-safe current context, upstream route detail, and user-visible source text summary. No runtime ids or worker mechanics. |
| Background artifact worker | unchanged | unchanged unless intake output field names change | background | Existing worker input cap and output cap remain. |

No new live-response call is allowed for ordinary chat, direct answers,
resolver-needed turns, or non-artifact private actions. The new intake call is
allowed only for the already-accepted background artifact candidate path.

## Change Surface

### Delete

- No files are planned for deletion.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`: rewrite
  the L2d prompt, parser normalization, and materialization boundary so L2d
  emits route requests instead of detailed background artifact params.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: wire L2d route output
  to background artifact intake and enforce the no-silent-background-enqueue
  invariant before `stage_2a_background_artifact_enqueue`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: add the minimal
  graph state keys for route requests and intake results.
- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`: adjust validation
  helpers only if full goal progress becomes optional for non-goal turns.
- `src/kazusa_ai_chatbot/background_artifact/models.py`: add the typed intake
  decision contract.
- `src/kazusa_ai_chatbot/background_artifact/README.md`: update the ICD for
  background artifact intake ownership and forbidden prompt inputs.
- `src/kazusa_ai_chatbot/background_artifact/__init__.py`: export the public
  intake entrypoint if a new intake module is created.
- `src/kazusa_ai_chatbot/action_spec/registry.py`: update prompt-safe
  capability projection only if L2d no longer needs `work_kind` in the root
  action schema.
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`: keep deterministic
  validation aligned with the intake-produced params.
- `src/kazusa_ai_chatbot/action_spec/execution.py`: preserve queue execution;
  adjust only if the accepted intake request is represented separately before
  action-spec materialization.
- `src/kazusa_ai_chatbot/action_spec/results.py`: expose prompt-safe rejection
  results for blocked silent-background-enqueue cases if not already covered.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: consume the
  prompt-safe rejection or pending result without exposing internals.
- `src/kazusa_ai_chatbot/nodes/README.md`: document L2d router, background
  artifact intake, and acknowledgement invariant ownership.
- `src/kazusa_ai_chatbot/action_spec/README.md`: document the changed
  ownership split between router, intake, deterministic validation, and L3.
- `tests/l2d_action_selection_cases.py`: update frozen-case comparison support
  for route-only L2d output.
- `tests/test_l2d_action_selection_cases.py`: add deterministic route
  normalization and materialization tests.
- `tests/test_l2d_action_selection_live_llm.py`: update one-case live LLM
  evidence to inspect route quality rather than old detailed work-kind output.
- `tests/test_cognition_prompt_contract_text.py`: cover L2d router prompt
  contract and forbidden implementation vocabulary.
- `tests/test_persona_supervisor2.py`: cover graph wiring from L2d route to
  intake to pre-surface enqueue.
- `tests/test_l2d_l3_surface_handoff.py`: cover no visible handoff means no
  background enqueue and no later delivery promise.
- `tests/test_action_spec_evaluator.py`: keep acceptance/rejection tests for
  the final queued action params.
- `tests/test_background_artifact_runtime.py`: cover public intake entrypoint
  and facade boundary if a new module is created.
- `tests/test_background_artifact_worker_live_llm.py`: rerun only if worker
  input shape changes.

### Create

- `src/kazusa_ai_chatbot/background_artifact/intake.py`: background artifact
  intake specialist, prompt, LLM call, parser, and validation.
- `tests/test_background_artifact_intake.py`: deterministic tests for intake
  parser/validator and patched LLM handoff behavior.
- `tests/test_background_artifact_intake_live_llm.py`: one-case-at-a-time live
  LLM tests for accepted coding, accepted rewrite, accepted summary, missing
  input, and unsupported shell/filesystem/web cases.
- `test_artifacts/llm_reviews/l2d_router_split_review_<date>.md`: human
  review artifact for L2d route and intake quality evidence.

### Keep

- Keep `src/kazusa_ai_chatbot/background_artifact/worker.py` as the monolithic
  snippet/text/summary worker unless intake output field names force a small
  adapter update.
- Keep `src/kazusa_ai_chatbot/db/background_artifact_jobs.py` job row shape.
- Keep `src/kazusa_ai_chatbot/background_artifact/delivery.py` and
  `result_source.py` delivery contracts.
- Keep adapter, dispatcher, calendar, proactive output, reflection, and
  consolidator ownership unchanged except for tests proving they are not used
  as background job owners.

## Overdesign Guardrail

- Actual problem: L2d is overloaded and can create correct private background
  route intent while losing the visible social handoff or corrupting unrelated
  resolver/goal fields.
- Minimal change: split L2d into route-only output plus a background artifact
  intake specialist, and add deterministic no-silent-enqueue enforcement.
- Ownership boundaries: L2d owns coarse semantic route; intake owns artifact
  kind/objective readiness; deterministic code owns validation and enqueue;
  L3/dialog owns visible wording; resolver owns evidence/HIL/approval
  recurrence and goal progress.
- Rejected complexity: no multi-agent coding system, no new worker registry,
  no retry prompts, no keyword fallback routing, no background web research,
  no adapter delivery changes, no DB migration, and no delivery crash-recovery
  mechanism in this plan.
- Evidence threshold: add more specialists only after repeated live LLM
  evidence shows intake itself is overloaded across approved artifact kinds.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper names only when they preserve
  the contracts in this plan.
- The responsible agent must not introduce new architecture, compatibility
  layers, fallback paths, feature flags, or extra agents.
- The responsible agent must not tune prompts to the Stage 0 fixture nouns.
- The responsible agent must keep changes outside the listed files out of
  scope unless the plan is updated and re-reviewed.
- If the L2d split reveals a required public contract not listed here, stop
  and update the plan before production edits.
- If live LLM output quality is mixed but the semantic architecture is correct,
  record the quality evidence and avoid overfitting.

## Implementation Order

1. Parent records the focused test contract before production edits:
   - silent background enqueue is rejected when no valid visible handoff route
     exists;
   - L2d route-only normalization accepts valid route output;
   - background artifact intake accepts bounded text work and rejects unsafe
     scope.
2. Parent runs the focused tests and records the expected failures or baseline.
3. Parent starts exactly one production-code subagent for production edits.
4. Production-code subagent implements L2d route output, intake, and invariant
   enforcement inside the approved change surface.
5. Parent adds or updates integration and live LLM tests while the production
   subagent works.
6. Parent reruns focused deterministic tests, integration tests, static checks,
   and one-case live LLM checks.
7. Parent starts the independent code-review subagent.
8. Parent remediates review findings only inside the approved change surface,
   reruns affected checks, and records evidence.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it.
- Parent agent may continue integration tests, live LLM evidence, static
  checks, and validation work while the production-code subagent edits
  production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 0 - plan review and approval
  - Covers: this draft, registry row, and user approval.
  - Verify: independent plan review records no blockers.
  - Evidence: record review result in `Execution Evidence`.
  - Sign-off: `<agent/date>` after user approval.
- [ ] Stage 1 - focused test contract established
  - Covers: silent enqueue invariant, L2d route parser, intake contract.
  - Verify:
    `venv\Scripts\python -m pytest tests/test_l2d_action_selection_cases.py tests/test_background_artifact_intake.py tests/test_l2d_l3_surface_handoff.py -q`
  - Evidence: record expected failures or baseline.
  - Sign-off: `<agent/date>`.
- [ ] Stage 2 - production split implemented
  - Covers: L2d route-only contract, intake specialist, graph wiring, no silent
    enqueue invariant.
  - Verify: rerun Stage 1 focused command.
  - Evidence: changed production files and focused result.
  - Sign-off: `<agent/date>`.
- [ ] Stage 3 - integration and prompt-quality verification complete
  - Covers: persona graph handoff, action-spec validation, L3 prompt-safe
    acknowledgement, one-case live LLM checks.
  - Verify: all commands in `Verification`.
  - Evidence: trace paths and human-readable LLM review artifact.
  - Sign-off: `<agent/date>`.
- [ ] Stage 4 - independent code review complete
  - Covers: review of full diff, plan alignment, prompt boundaries, tests, and
    evidence.
  - Verify: review findings resolved and affected checks rerun.
  - Evidence: review subagent identity, findings, fixes, residual risks.
  - Sign-off: `<agent/date>`.

## Verification

### Deterministic Tests

```powershell
venv\Scripts\python -m pytest tests/test_l2d_action_selection_cases.py tests/test_background_artifact_intake.py tests/test_l2d_l3_surface_handoff.py tests/test_persona_supervisor2.py tests/test_action_spec_evaluator.py tests/test_background_artifact_runtime.py -q
```

Expected result after implementation: all selected deterministic tests pass.

### Live LLM Tests

Run one case at a time and inspect output after each run:

```powershell
$env:L2D_LIVE_CASE_FILE='tests\fixtures\l2d_background_artifact_cases.json'; $env:L2D_LIVE_CASE_ID='coding_snippet_accept_fibonacci'; venv\Scripts\python -m pytest tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -m live_llm -s
venv\Scripts\python -m pytest tests/test_background_artifact_intake_live_llm.py::test_background_artifact_intake_live_case -q -m live_llm -s
```

The parent must write a human-readable review artifact under
`test_artifacts/llm_reviews/` before claiming prompt quality.

### Static Checks

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\kazusa_ai_chatbot\background_artifact\intake.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py
git diff --check
rg "AdapterRegistry|dispatcher|send_message|RemoteHttpAdapter" src\kazusa_ai_chatbot\background_artifact
rg "首轮 artifact|artifact 阶段|resolver_capability_requests.*background_artifact_request" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\kazusa_ai_chatbot\background_artifact\intake.py
```

Expected static results: compile succeeds; `git diff --check` has no
whitespace errors; forbidden delivery imports do not appear in background
artifact worker/intake files; runtime prompts do not contain user-facing
implementation vocabulary such as `首轮 artifact`.

## Independent Plan Review

Run this gate before approval or execution. Prefer a reviewer that did not
draft the plan. If no separate reviewer is available, the drafting agent must
reread the completed background artifact plan, this draft, relevant source, and
the Stage 0 quality review from a fresh-review posture.

Review scope:

- L2d is truly reduced to top-level routing.
- Background artifact intake has a bounded public contract and forbidden paths.
- The no-silent-enqueue code bug is covered by deterministic tests.
- Prompt changes address general contract logic, not fixture-specific wording.
- File manifest is complete and does not authorize unrelated graph or adapter
  refactors.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, prompt,
  documentation, and test file.
- L2d prompt ownership, input audit, output schema, and local LLM context
  budget.
- Background artifact intake public boundary and forbidden runtime fields.
- Deterministic no-silent-enqueue invariant.
- Regression coverage and live LLM evidence quality.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- L2d no longer owns background artifact `work_kind` or detailed objective
  shaping.
- Background artifact intake owns bounded artifact classification and
  input-readiness judgment.
- Ordinary social chat is not forced into full resolver goal progress.
- No durable background artifact job is enqueued when the current turn lacks a
  valid visible handoff route.
- Deterministic tests, static checks, and one-case-at-a-time live LLM reviews
  pass or have accepted documented residual risks.
- Independent code review reports no blocking findings.

## Execution Evidence

No execution evidence yet. This plan is draft and not approved for production
edits.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| L2d route regression | Route-only focused tests and live LLM review | `test_l2d_action_selection_live_llm.py` |
| Extra response-path latency | Intake call only on accepted artifact candidate | LLM budget and graph tests |
| Prompt overfitting | No Stage 0 nouns or fixture-specific repair rules | Prompt grep and review |
| Silent background job | Deterministic invariant blocks enqueue | `test_l2d_l3_surface_handoff.py` |
| Goal continuity loss | Keep full progress for resolver goals only | Resolver and L3 handoff tests |
| Specialist scope creep | Intake forbidden paths and validator checks | Intake tests and action evaluator tests |
