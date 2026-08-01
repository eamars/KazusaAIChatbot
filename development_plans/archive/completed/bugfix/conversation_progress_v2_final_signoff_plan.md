# Conversation Progress V2 Final Sign-Off Plan

## Summary

- Goal: close Conversation Progress V2 through one readable semantic proof of
  the original failure and one explicit maximum-turn projection.
- Plan class: `medium`.
- Status: `completed`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `debug-llm`,
  `test-style-and-execution`, `py-style`, and `cjk-safety`.
- Cutover: replace the prior long plan as the sole active closeout contract;
  preserve V2 architecture and correct only a proven sign-off blocker.
- Highest risks: incomplete semantics hidden by schema success, an overstated
  turn limit, and forcing the user to inspect raw evidence.
- Acceptance: complete original-failure semantics reach the actual cognition
  prompt; the turn projection is proven and qualified; one semantic report is
  sufficient for user approval.
- Independent plan review: omitted by user instruction.

## Context

The implemented V2 system and detailed execution history remain in
`development_plans/archive/superseded/conversation_progress_v2_long_thread_continuation_bigbang_plan.md`.
This shorter successor alone governs remaining verification, bounded
remediation, evidence, and closure.

The original failure progressed through a selected neck/shoulder massage, its
completion and acceptance, a later ear/earlobe massage, completion of that
later action, and a request for the next selection. The earlier completed
location was then selected again. This plan signs off at the Conversation
Progress handoff: established facts must be complete in cognition input.
Cognition's later choice and dialog wording are separate concerns.

Capacity has two meanings. The immediate participant lane contains 10 complete
logical turns. Older continuity is semantic rather than a verbatim transcript.
The tested semantic frontier is 1,048 facts: 1,024 archived snapshots plus 24
active events. In the sign-off workload, cycle one creates the original
decision-critical fact plus one ordinary fact, and each later cycle adds one
ordinary fact. The result is 1,047 accepted settled interaction cycles; the
next cycle would add fact 1,049 and must fail before changing retained state.

## Mandatory Skills

- `development-plan`: execution, evidence, review, lifecycle, and sign-off.
- `local-llm-architecture`: semantic/deterministic ownership and prompt limits.
- `debug-llm`: agent-authored readable evidence rather than raw-first review.
- `test-style-and-execution`: deterministic batches and one-at-a-time live LLM.
- `py-style`: load before reviewing or changing Python.
- `cjk-safety`: load before changing CJK Python fixtures or tests.

## Mandatory Rules

1. This is the sole active Conversation Progress V2 closeout plan.
2. Sign-off ends at serialized cognition input; response choice and dialog are
   non-blocking.
3. Run one real recorder case alone and inspect it before later live work.
4. Deterministic tests prove transport and capacity, not model semantic quality.
5. The parent authors the readable report after inspecting real evidence.
6. The user-facing report contains no raw JSON, rows, IDs, timestamps, prompt
   dumps, trace dumps, or unprocessed model output.
7. A settled interaction cycle is one eligible progress update after a user
   turn and accepted response or eligible silence. A logical speaker turn is
   one user row or one grouped assistant response.
8. Present 1,047 cycles as a workload-bound projection, not a universal
   guarantee.
9. Add no case-specific production rule, new LLM call, compatibility path, or
   parallel memory lane.
10. Start any Python remediation with the focused failing test and limit it to
    the proven owner in `Change Surface`.
11. Run no live database read, migration, restore, or deployment operation.
12. After context compaction or a major stage sign-off, reread this plan.
13. Before completion, run `Independent Code Review` and record the result.
14. Change status to `completed` only after explicit user approval.

## Must Do

1. Run the source-faithful failure through the real scene observer and event
   reconciler.
2. Prove the completed event contains actor, action, object, beneficiary,
   outcome, completed state, decision-critical retention, and source lineage.
3. Prove the same facts survive active state, evidence projection, and actual
   serialized goal-prompt input.
4. Show that the later action is complete and the user requests the next
   selection.
5. Prove the 1,048-fact capacity and fail-closed next cycle through production
   merge, repository, compaction, and graph validation.
6. Author
   `test_artifacts/reviews/conversation_progress_v2_final_signoff.md` as the
   only required user approval surface.
7. Run focused, static, adjacent, and full non-live gates.
8. Record review, remediation, reruns, residual risks, and user decision.

## Deferred

- Do not require cognition to choose a different next action.
- Do not tune final dialog, prompts, retrieval, or architecture beyond a
  focused red sign-off gate.
- Do not add a benchmark framework, generated report, dashboard, or UI.
- Do not expose raw evidence as required user reading.
- Do not execute database migration or production deployment.
- Do not copy prior stage history into this plan.

## Cutover Policy

Overall strategy: lifecycle big-bang with verify-and-close implementation.

| Area | Policy | Instruction |
|---|---|---|
| Plan lifecycle | bigbang | Use this plan only; keep the old plan as superseded history. |
| V2 runtime | retained | Preserve the current single V2 path. |
| Sign-off evidence | bigbang | Use one semantic Markdown report instead of raw-first approval. |
| Correction | bounded | Modify only an owner proven red by a focused gate. |
| Database/deployment | unchanged | Keep both outside this plan. |

Follow each area policy, preserve only listed retained surfaces, and obtain user
approval before changing the cutover.

## Target State

The user reads one report containing:

1. a pass or blocked verdict;
2. a plain-language timeline of the original failure;
3. a semantic handoff table showing what was produced, stored, projected, and
   serialized for cognition;
4. a maximum-turn table with assumptions and fail-closed behavior;
5. test results and residual scope; and
6. a user approval line.

The handoff table must show:

| Dimension | Required evidence |
|---|---|
| Actor | The current user performed the action. |
| Action/object | The requested neck/shoulder massage was completed. |
| Beneficiary/outcome | The character received, accepted, and evaluated it. |
| Lifecycle/retention | The event is completed and decision-critical. |
| Lineage | It is grounded in the user's action and accepted response. |
| Current transition | The ear/earlobe action is complete and the user asks what is next. |
| Cognition handoff | The same facts appear intact in citeable evidence and serialized goal input. |

The capacity table must show:

| Item | Required result |
|---|---|
| Semantic capacity | 1,048 facts: 1,024 archived plus 24 active. |
| Accepted frontier | 1,047 settled cycles under the specified workload. |
| Speaker-turn projection | About 2,094 logical turns for one user plus one grouped assistant turn per visible cycle. |
| Recent context | 10 complete participant logical turns; older content is semantic, not verbatim. |
| Next cycle | Fact 1,049 is rejected before packet or graph mutation. |
| Assumptions | One new fact per later cycle, bounded text, continuous 48-hour sliding expiry, and no extra facts per cycle. |
| Variability | Multiple or longer facts lower the frontier; no-new-fact cycles can exceed it. |

## Design Decisions

| Topic | Decision | Reason |
|---|---|---|
| Boundary | Serialized cognition input | Conversation Progress owns evidence, not the final choice. |
| Semantic proof | One real recorder replay | Model production needs real-model evidence. |
| Transport proof | Deterministic source-faithful replay | Preservation must be exact and stable. |
| Approval surface | Agent-authored semantic Markdown | The user should not need raw data. |
| Capacity unit | Settled interaction cycle | Packet turn count advances per eligible update. |
| Projection | 1,047 cycles / about 2,094 alternating speaker turns | Cycle one contributes two of the 1,048 facts. |

## Change Surface

### Create

- `test_artifacts/reviews/conversation_progress_v2_final_signoff.md`: readable
  semantic approval artifact.

### Modify

- `tests/test_conversation_progress_v2_live_llm.py`: add one live
  progress-to-cognition-boundary case.
- `tests/test_conversation_progress_v2_regression.py`: change only when its
  existing handoff or frontier assertions cannot provide required evidence.
- On a focused red gate only:
  `src/kazusa_ai_chatbot/conversation_progress/recorder.py`,
  `src/kazusa_ai_chatbot/conversation_progress/delta_merge.py`,
  `src/kazusa_ai_chatbot/conversation_progress/projection.py`,
  `src/kazusa_ai_chatbot/conversation_progress/compaction.py`,
  `src/kazusa_ai_chatbot/conversation_progress/repository.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`, or
  `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`.

### Delete

- No production or test artifact.

### Keep

- Service, adapters, dialog, database migration, durable memory, reflection,
  retrieval routing, and public V2 facade contracts remain unchanged.

## Overdesign Guardrail

- Actual problem: approval lacks one concise semantic proof and qualified turn
  projection.
- Minimal change: one focused live case, existing deterministic proofs, one
  bounded correction when red, and one readable report.
- Ownership: recorder LLMs observe semantics; code validates, stores, compacts,
  and projects; cognition consumes; dialog remains downstream.
- Rejected complexity: new memory, prompts, evaluators, retries, reports,
  dashboards, migrations, or response tuning.
- Expansion threshold: a focused gate must prove the present V2 owner cannot
  satisfy a sign-off condition.

## Agent Autonomy Boundaries

- Reuse existing fixtures and helpers.
- Keep the live case explicit and unparameterized.
- Do not turn report wording into production keyword logic.
- Do not claim an unconditional 1,047-turn guarantee.
- Do not change a production owner whose focused gate is green.
- Stop when a fix needs a new public contract, LLM call, DB shape, or file
  outside `Change Surface`.
- Preserve unrelated user work in the dirty worktree.

## Implementation Order

1. Rebaseline the worktree and read this plan, relevant source, tests, and ICD.
2. Add and run `test_live_original_failure_progress_semantic_handoff` alone.
3. Run the deterministic source-faithful handoff test.
4. Run balanced-compaction and exact-frontier tests; calculate the report table.
5. For a red gate, record RED, use one production subagent for its exact owner,
   apply the minimal fix, and rerun GREEN.
6. Run adjacent, static, and full non-live gates.
7. Inspect raw evidence internally and author the semantic report.
8. Run independent code review, remediate in-scope findings, present the
   report, and request user approval.

## Execution Model

- The parent owns tests, live inspection, deterministic verification, capacity
  calculation, report, evidence, remediation, lifecycle, and sign-off.
- No production subagent starts while focused gates are green.
- One red gate starts exactly one production-code subagent after its failing
  contract is recorded; it edits production only within the proven owner.
- After verification, one independent code-review subagent reviews the
  relevant diff, evidence, capacity math, and report without implementing.
- If required native subagents are unavailable, stop unless the user authorizes
  fallback execution.

## Progress Checklist

- [x] Stage 1 — original-failure producer and handoff.
  - Covers: steps 1-3.
  - Verify: targeted live and deterministic handoff cases pass.
  - Evidence: readable timeline and field-by-field semantic handoff.
  - Handoff: Stage 2.
  - Sign-off: `Codex/2026-07-30`; isolated live producer/handoff and
    deterministic source-faithful transport are GREEN.
- [x] Stage 2 — maximum-turn projection.
  - Covers: steps 4-5.
  - Verify: balanced and exact-frontier cases pass.
  - Evidence: facts, cycle definition, assumptions, frontier, and rejection.
  - Handoff: Stage 3.
  - Sign-off: `Codex/2026-07-30`; 1,048 facts, 1,047 accepted
    workload-bound cycles, and fail-closed fact 1,049 are GREEN.
- [x] Stage 3 — regression and semantic report.
  - Covers: steps 6-7.
  - Verify: static, adjacent, and full non-live gates pass.
  - Evidence: complete report with no raw required reading.
  - Handoff: Stage 4.
  - Sign-off: `Codex/2026-07-30`; static, adjacent, full non-live, and
    human-readable semantic-report gates are GREEN.
- [x] Stage 4 — code review and user approval.
  - Covers: step 8 and `Independent Code Review`.
  - Verify: no unresolved review blocker; affected gates rerun after fixes.
  - Evidence: findings, remediation, residual risks, and user decision.
  - Handoff: complete and archive only after approval.
  - Sign-off: `Codex/2026-08-01`; user explicitly approved completion and
    lifecycle archive without requesting additional review work.

## Verification

Run the live case alone and inspect it:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests\test_conversation_progress_v2_live_llm.py::test_live_original_failure_progress_semantic_handoff -q -s
```

Expected: complete original-failure semantics reach serialized cognition input;
no downstream cognition choice is invoked or judged.

Run deterministic handoff and capacity:

```powershell
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_v2_regression.py::test_source_faithful_regression_projects_key_details_to_cognition -q
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_v2_regression.py::test_balanced_compaction_survives_old_depth_failure_frontier tests\test_conversation_progress_v2_regression.py::test_maximum_history_capacity_reaches_node_cap_and_fails_closed -q
```

Expected: all required fields remain intact; 1,047 cycles retain 1,048 facts;
the next fact fails before mutation.

Run adjacent budget checks and full non-live regression:

```powershell
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_cognition_evidence.py tests\test_cognition_core_v2_prompt_budget_continuity.py -q
venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q
```

Expected: exit code 0; record exact pass/skip counts.

Run the production vocabulary scan:

```powershell
rg -n "后颈|耳根|摸摸|按摩" src\kazusa_ai_chatbot
```

Expected: zero matches; `rg` exit code 1 is the accepted no-match result.

The parent then authors the report defined in `Target State`. Raw paths may be
listed as optional audit references, while raw content remains outside the
approval surface.

## Independent Code Review

No independent plan review is required. After verification, one review
subagent checks:

1. original semantics against recorder output, active event, cognition
   evidence, and serialized goal input;
2. report accuracy and absence of raw-data burden;
3. the 1,047-cycle calculation, assumptions, and fail-closed next cycle;
4. relevant style, CJK safety, prompt safety, hidden lexical rules, and scope.

Record findings, fixes, reruns, residual risks, and approval. A finding that
requires new architecture, public contract, DB change, or LLM call blocks
closure and requires a separate plan.

## Acceptance Criteria

This plan is complete only when:

- the real recorder produces every required original-failure semantic dimension;
- the later completion and next-selection request remain present;
- active state, cognition evidence, and serialized goal input preserve those
  facts intact;
- deterministic verification proves 1,048 facts, 1,047 accepted cycles, and
  fail-closed rejection of the next fact;
- the report qualifies the approximately 2,094 alternating-speaker-turn
  projection and distinguishes it from the 10-turn recent window;
- focused, static, adjacent, and full non-live gates pass;
- the report supports approval without reading raw data;
- independent code review has no unresolved blocker; and
- the user explicitly approves final sign-off.

## Execution Evidence

- Live semantic case and judgment: GREEN on 2026-07-30 after fail-closed
  structural and semantic RED iterations. Human inspection confirmed the
  current user, completed neck/shoulder massage, character beneficiary and
  evaluation, decision retention, source lineage, earlobe completion request,
  and unchanged serialized goal input. Final raw audit:
  `test_artifacts/llm_traces/test_live_original_failure_progress_semantic_handoff__original_failure_progress_semantic_handoff.json`.
- Deterministic handoff: GREEN on 2026-07-30; the source-faithful
  recorder-to-goal-prompt case passed with actor, action, object, beneficiary,
  outcome, state, retention, and exact semantic text preserved.
- Capacity frontier and projection: GREEN on 2026-07-30; balanced compaction
  and exact frontier both passed (`2 passed`), retaining 1,048 facts through
  1,047 settled cycles and rejecting fact 1,049 before mutation.
- Adjacent/full non-live: adjacent evidence and prompt-budget gates passed
  (`35 passed`); the production vocabulary scan returned zero matches; the full
  non-live suite passed (`3,815 passed`, `3 skipped`).
- Semantic report:
  `test_artifacts/reviews/conversation_progress_v2_final_signoff.md` contains
  the readable verdict, handoff evidence, capacity projection, verification,
  residual risks, and approval boundary without raw-data reading.
- Review and remediation: independent review APPROVED with no blocker. The
  typed-input boundary was clarified, full block-graph immutability was added
  to the fact-1,049 gate, and prompt-cap assertions now use production policy.
  The affected frontier and deterministic handoff tests passed (`2 passed`).
- Residual risks: local-model variance; upstream decontextualization is outside
  scope; capacity varies with fact volume, text size, expiry, and no-fact
  cycles; full-suite counts have no separate retained terminal transcript.
- User decision: explicit completion and lifecycle-archive approval recorded on
  2026-08-01.

## Lifecycle Closure

- User explicitly approved completion and lifecycle archive on 2026-08-01.
- The recorded implementation, verification, semantic report, and prior review
  evidence are retained as historical execution evidence; no additional code
  review or verification was performed for this closure request.
- Archived from `active/bugfix/` to `archive/completed/bugfix/` on 2026-08-01.

## Risks

| Risk | Control | Gate |
|---|---|---|
| Schema pass hides missing meaning | Review real semantic fields | Live original-failure case |
| Report distorts evidence | Reviewer cross-check | Code review |
| Turn limit is overclaimed | Explicit workload and variability | Capacity report |
| More/larger events lower capacity | Label projection, not guarantee | Frontier evidence |
| Scope expands into response tuning | Stop at cognition input | Review |
