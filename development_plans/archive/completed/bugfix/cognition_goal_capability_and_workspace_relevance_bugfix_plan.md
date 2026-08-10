# Cognition Goal Capability And Workspace Relevance Bugfix Plan

Status: completed

Plan class: medium

Cutover: bigbang

## Summary

Correct the dual Cognition V2 contract failure that caused Asuna to declare
online search unavailable before action planning could select
`task_resolution_request`.

The cutover moves operational capability and evidence-feasibility ownership
out of goal cognition and into the existing action-planning boundary. It also
grounds the existing workspace-collapse call in the current typed event and
the persistent goal behind each non-ordinary bid, allowing the model to
suppress stale, unrelated branches.

The implementation keeps the current LLM call graph, `ActionBidV2`, workspace
partition schema, resolver roster, inline task-resolution budget, and
background handoff unchanged.

Acceptance requires the captured online-search case to produce a healthy
ordinary goal, suppress the unrelated autonomy bid, and select
`task_resolution_request`; matching autonomy and unavailable-future-effect
controls must retain truthful character judgment.

## Context

Production cognition run `f30538559bd245dd85e9a96996d4f5d4` received a typed
request for the current character to search current memory and GPU prices.
`task_resolution_request` was available and runtime capability limits were
empty. Both goal branches invented an unavailable-search limitation, workspace
collapse selected the autonomy refusal, and action planning correctly treated
that refusal as answerable speech.

The autonomy bid came from an active persistent goal about body contact and
cleaning from days earlier. Branch activation admits every pursuing or blocked
goal kind and checks dependency identifiers, not current semantic relevance.
Workspace collapse currently sees only generated bid text, so it cannot
compare the bid's persistent goal with the current event.

Real-model probes recorded in
`test_artifacts/llm_traces/online_search_dual_root_cause_validation_review.md`
established:

- exact ordinary and autonomy prompt replays reproduce the refusal;
- an explicit empty capability-limit list and an exhaustive-list clarification
  still reproduce it;
- a capability-neutral goal contract repairs the ordinary branch;
- the same goal contract cannot overcome an unrelated refusal-oriented
  autonomy branch;
- a current-event-grounded workspace-collapse prompt suppresses that stale
  branch;
- the same collapse prompt admits autonomy when the current event matches the
  persistent goal;
- the unchanged action planner selects `task_resolution_request` from a
  healthy current-fact bid.

The adjacent active
`required_selection_partial_recovery_bugfix_plan.md` also targets
`goal_cognition.py` and `facade.py`. Its reproduction gate stopped before
production edits. Execution of this plan must recheck worktree overlap and
sequence any later required-selection work after this contract cutover.

## Mandatory Skills

Execution must read and apply these skills in full before the corresponding
work:

- `.agents/skills/development-plan/SKILL.md`
- `.agents/skills/local-llm-architecture/SKILL.md`
- `.agents/skills/debug-llm/SKILL.md`
- `.agents/skills/no-prepost-user-input/SKILL.md`
- `.agents/skills/test-style-and-execution/SKILL.md`
- `.agents/skills/py-style/SKILL.md`
- `.agents/skills/cjk-safety/SKILL.md`

After context compaction, reread this entire plan and every mandatory skill
before continuing. Reread the plan before final verification.

## Mandatory Rules

- Use `venv\Scripts\python.exe` for Python and pytest.
- Run live LLM tests one case at a time and inspect each durable artifact
  before continuing.
- Pass raw model output through
  `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` before validation.
- Keep semantic current-relevance judgment in the existing workspace-collapse
  LLM.
- Keep evidence sufficiency, resolver choice, and operational feasibility in
  the existing action-planning LLM and deterministic authorization boundary.
- Keep goal cognition responsible for character motive, stance, target
  outcome, and evidence-grounded character refusal.
- Preserve `ActionBidV2`, `WorkspaceDecisionV2`, action/resolver affordance
  schemas, retry caps, model routes, and endpoint configuration.
- Preserve current task-resolution, inline budget, checkpoint, queue, and
  background-handoff behavior.
- Keep one canonical contract and perform a big-bang caller/callee/test/doc
  update.
- Use bounded context projections with explicit caps; full state dumps remain
  outside workspace collapse.
- Preserve unrelated user changes in the worktree.

## Must Do

- Add production-shaped deterministic and real-model regression tests before
  production edits.
- Replace goal cognition's operational feasibility instruction with a
  capability-neutral semantic-goal contract.
- Remove runtime capability limits from the goal-cognition dynamic payload.
- Preserve execution-neutral evidence needs such as "obtain the required
  current facts, then respond."
- Feed workspace collapse a bounded current-event projection and the exact
  persistent-goal projection associated with each non-ordinary bid.
- Require collapse to suppress persistent-goal bids concerning a different
  concrete matter.
- Preserve ordinary response as the current-turn baseline candidate.
- Preserve matching autonomy, safety, relationship, and other persistent-goal
  branches when current evidence directly advances, obstructs, threatens, or
  requests handling of the same matter.
- Verify that the captured online-search flow reaches
  `task_resolution_request` before any task worker is invoked.
- Verify unavailable future-reminder and unavailable coding-owner cases remain
  truthful at action planning and visible surface boundaries.
- Update the cognition-core README ownership description.
- Produce a post-fix, agent-authored real-LLM comparison artifact.
- Complete an independent code review after verification.

## Deferred

- Task-orchestrator specialist selection and public-research implementation.
- Inline resolver timing, the 30-second default, checkpointing, leasing,
  background queues, and worker execution.
- Action-selector enum, resolver affordance, and authorization redesign.
- Persistent-goal lifecycle cleanup or historical state migration.
- Reflection and memory content changes, including the captured technical
  boundary reflection.
- Dialog style or character profile changes.
- New model routes, verifier calls, classifiers, retries, feature flags,
  compatibility aliases, or keyword routing.
- General branch-activation redesign beyond current relevance at collapse.

## Cutover Policy

Use one big-bang contract cutover:

- goal prompt, goal payload projection, collapse signature, collapse prompt,
  facade call, tests, and cognition-core documentation move together;
- workspace collapse continues to emit the existing three-way partition;
- action planning continues to consume only primary and supporting bids;
- the previous goal-owned feasibility wording and context projection are
  removed in the same change;
- rollback is one source revert of the complete change.

## Target State

```text
typed current event + evidence + persistent goals
  -> goal cognition branches
       ordinary: character motive + evidence-neutral desired outcome
       active goal: character motive for its persistent goal
  -> workspace collapse
       current event + bounded per-bid persistent-goal provenance
       unrelated active-goal bid -> suppressed
       matching active-goal bid -> primary/supporting as character judgment
  -> action planning
       evidence sufficient -> answerable_now
       required current facts missing + resolver available
         -> requires_required_evidence + task_resolution_request
       required effect unavailable -> blocked/truthful boundary
  -> authorization -> inline task resolution
  -> background handoff only after the existing inline budget is exhausted
```

## Design Decisions

### Root-cause ownership

The validated failure has two independent contributors:

1. Goal cognition converts knowledge gaps and generic model priors into
   runtime capability claims.
2. Workspace collapse lacks enough provenance to reject a bid generated from
   an unrelated persistent goal.

Action planning is downstream and behaves correctly when the admitted goal
requires current external evidence.

### Goal-cognition contract

Goal cognition decides what the character wants to accomplish in the current
scene and why. It may express a genuine value, relationship, consent, safety,
or identity refusal supported by current evidence. It expresses missing facts
as a desired evidence-backed outcome without deciding which tool, resolver,
worker, or scheduler can provide them.

The prompt must distinguish:

- character refusal: a semantic stance owned by cognition and grounded in the
  current event;
- epistemic uncertainty: current facts are missing, so the goal preserves the
  need for evidence;
- runtime infeasibility: owned by action planning and deterministic
  authorization using the supplied runtime limits and affordances.

### Workspace current relevance

Workspace collapse remains the sole partition owner. Its prompt receives:

- a compact typed current-event projection from authoritative episode
  evidence;
- existing bid fields used for collapse;
- branch identity;
- for non-ordinary bids, a compact projection of the persistent goal resolved
  from the bid's `goal_ref`: handle, description, lifecycle, salience,
  urgency, progress, and obstruction when present.

The model suppresses a non-ordinary bid when the persistent goal and current
event concern different concrete matters. Active lifecycle, shared user,
generic relationship appraisal, general drive importance, and action
tendencies are context rather than proof of current relevance.

### Rejected alternatives

- Explicit empty capability limits alone: reproduced the refusal.
- Stronger exhaustive-list wording alone: reproduced the refusal.
- Goal-output `irrelevant_this_turn` union: semantic abstention worked, while
  the local model produced avoidable shape errors on the matching control.
- Deterministic keyword or embedding relevance filters: they would move
  semantic user-input judgment out of the LLM boundary.
- New branch-relevance LLM: the existing collapse call already owns
  partitioning and passed both live controls.

## Contracts And Data Shapes

`ActionBidV2` and `WorkspaceDecisionV2` remain unchanged.

Extend the private `collapse_bids(...)` input with bounded semantic context:

```python
async def collapse_bids(
    bids: Sequence[ActionBidV2],
    services: CognitionCoreServicesV2,
    *,
    current_event: Mapping[str, object],
    goal_context_by_ref: Mapping[str, Mapping[str, object]],
) -> CollapsedIntentionV2:
    ...
```

The exact private argument names may follow existing project style, while the
semantic content and ownership remain fixed by this plan.

The prompt payload keeps stable handles:

```text
current_event:
  response_operation / role-explicit semantic event summary
bids.bN:
  branch_id
  persistent_goal: null or bounded goal projection
  intention
  desired_outcome
  reason
  confidence
```

Deterministic code resolves `goal_ref` to an existing state goal and copies
bounded fields. It performs provenance mapping and size enforcement only; the
model decides relevance and partition.

If a non-ordinary bid references a missing required internal goal, execution
uses the project's fail-fast internal-data policy rather than inventing a
goal. Ordinary response uses `persistent_goal: null`.

## LLM Call And Context Budget

| Surface | Before | After |
|---|---:|---:|
| Goal calls per selected branch | 1 within current bounded attempts | unchanged |
| Workspace-collapse calls | 1 when multiple bids exist | unchanged |
| Action-planning calls | 1 within current bounded attempts | unchanged |
| Added evaluator/verifier calls | 0 | 0 |
| Goal aggregate prompt cap | 24,000 characters | unchanged |
| Workspace aggregate prompt cap | 24,000 characters | unchanged |

Goal context becomes smaller by removing runtime capability limits. Workspace
context grows only by bounded current-event and goal-provenance projections.
The implementation must retain deterministic fitting within the existing
24,000-character cap and preserve every bid handle in the partition contract.

## Change Surface

### Production

- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
  - replace operational-feasibility wording with semantic-goal ownership.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`
  - stop projecting runtime capability limits into goal cognition;
  - build and pass bounded collapse grounding.
- `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py`
  - accept grounded context;
  - update collapse prompt and prompt fitting while preserving output schema.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
  - document goal, collapse, and action-planning ownership.

### Tests

- `tests/test_cognition_core_v2_goal_capability_live_llm.py` (new)
- `tests/test_cognition_core_v2_workspace_live_llm.py` (new or existing focused
  workspace live file if one exists at execution)
- `tests/test_cognition_core_v2_action_planning_live_llm.py`
- `tests/test_cognition_core_v2_action_planning_bugfix.py`
- `tests/test_cognition_core_v2_integration.py`
- `tests/test_cognition_core_v2_prompt_budget_continuity.py`
- `tests/test_cognition_core_v2_stage_model_routing.py`
- focused existing future-reminder and coding-owner boundary tests discovered
  during execution.

### Evidence and lifecycle

- `test_artifacts/llm_traces/` for per-case real-model traces.
- `test_artifacts/reviews/cognition_goal_capability_workspace_relevance_review.md`
  for pre-fix/post-fix comparison.
- this plan and `development_plans/README.md`.

### Preserved production surfaces

- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/resolver_authorization.py`
- `src/kazusa_ai_chatbot/task_resolution/`
- `src/kazusa_ai_chatbot/background_work/`
- persistence, adapters, and dialog generation.

## Overdesign Guardrail

- Two prompt ownership corrections in existing stages.
- One bounded private context projection for workspace collapse.
- Existing bid and partition schemas.
- Existing model routes and call count.
- Existing resolver and worker flow.
- Scope remains limited to the two reproduced contract defects and their
  immediate regression controls.

## Agent Autonomy Boundaries

The executing agent may:

- add the named deterministic and live regression tests;
- adjust private helper signatures and bounded prompt fitting within the
  listed files;
- update the cognition-core README and evidence artifact;
- make mechanical call-site and test-fixture updates required by the big-bang
  private contract cutover.

The executing agent must stop and request a plan amendment when:

- the exact captured case fails after both contract corrections;
- the fix requires a new LLM call, model route, retry increase, public schema,
  persistence change, or worker change;
- unavailable future effects become false promises at the surface;
- matching autonomy is consistently suppressed;
- resolving bid `goal_ref` requires invented or ambiguous goal data;
- unrelated worktree changes overlap a target and cannot be preserved.

## Implementation Order

1. Re-read this plan, mandatory skills, repository instructions,
   `README.md`, `docs/HOWTO.md`, cognition-core README, archived unified
   task-resolution plan, adjacent active required-selection plan, current git
   status, and every target source/test file.
2. Add deterministic red tests for goal payload ownership, bounded collapse
   grounding, unrelated-goal suppression, matching-goal admission, prompt cap,
   and unchanged partition validation.
3. Add production-shaped live tests using the captured event and goal context.
4. Run each pre-fix live case separately and preserve/inspect raw artifacts.
5. Update goal cognition prompt and dynamic payload ownership.
6. Run immediate Python compile checks because the prompt contains CJK text.
7. Extend workspace collapse with bounded current-event and persistent-goal
   grounding while preserving its output schema.
8. Update the facade caller in the same cutover.
9. Run focused deterministic tests and repair only in-scope regressions.
10. Run the exact goal, collapse, and goal-to-action live cases one at a time.
11. Run matching autonomy and unavailable-effect controls one at a time.
12. Update cognition-core documentation and the human-readable LLM review.
13. Run broader non-live cognition regressions, static checks, and diff
    inspection.
14. Run one fresh independent code-review subagent, remediate valid findings,
    and repeat affected verification.
15. Record commands, artifacts, model routes, observed latencies, test results,
    review findings, and final signoff in this plan.

## Execution Model

Execution uses parent-led native subagents:

- the parent owns live-test execution and artifact inspection, integration,
  final verification, and plan evidence;
- one focused implementation subagent may modify the production and
  deterministic test surface after red tests exist;
- one fresh independent review subagent inspects the final diff and evidence;
- one production-code subagent works at a time.

Subagents receive this complete plan, exact file ownership, acceptance gates,
and instruction to preserve unrelated changes. If native subagents are
unavailable at an implementation or review gate, execution pauses for explicit
user direction.

## Progress Checklist

- [x] Production run and full LLM trace retrieved.
- [x] Exact action-planner failure reproduced.
- [x] Healthy-bid action-planner control selects task resolution.
- [x] Exact goal-cognition failures reproduced.
- [x] Explicit-empty-limits hypothesis falsified.
- [x] Capability-neutral ordinary-goal candidate validated.
- [x] Stale autonomy-goal contamination isolated.
- [x] Grounded workspace-collapse failure and positive controls validated.
- [x] Human-readable pre-fix validation review written.
- [x] Deterministic red tests added.
- [x] Production contract cutover implemented.
- [x] Focused deterministic tests pass.
- [x] Exact post-fix live path selects task resolution.
- [x] Matching autonomy control passes.
- [x] Unavailable-effect controls pass through action planning and surface.
- [x] Broader regression and static checks pass.
- [x] Independent review complete and findings resolved.
- [x] Execution evidence and acceptance criteria signed off.

## Verification

### Deterministic focused verification

```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_action_planning_bugfix.py tests\test_cognition_core_v2_integration.py tests\test_cognition_core_v2_prompt_budget_continuity.py tests\test_cognition_core_v2_stage_model_routing.py -m "not live_llm and not live_db" -q
```

Required assertions:

- goal prompt assigns runtime feasibility to action planning;
- goal payload excludes runtime capability limits;
- collapse payload includes the typed current event and exact goal associated
  with each non-ordinary bid;
- ordinary bid uses `persistent_goal: null`;
- missing internal goal provenance fails fast;
- unrelated persistent goal is suppressed by a fixed model response fixture;
- matching persistent goal remains primary or supporting;
- partition exact-key and exact-cover validation remains unchanged;
- prompt fitting remains under 24,000 characters and retains all bid handles;
- action planner still receives runtime limits and resolver affordances;
- call counts, model routes, and retry caps remain unchanged.

### Live LLM verification

Run each node separately and inspect its newly written artifact:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_goal_capability_live_llm.py::test_live_captured_online_search_goal_preserves_required_evidence -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_workspace_live_llm.py::test_live_captured_online_search_suppresses_unrelated_autonomy_goal -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_workspace_live_llm.py::test_live_matching_autonomy_goal_remains_admitted -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_action_planning_live_llm.py::test_live_captured_online_search_goal_to_action_selects_task_resolution -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_action_planning_live_llm.py::test_unavailable_reminder_does_not_change_capability_owner -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_action_planning_live_llm.py::test_live_unavailable_coding_owner_does_not_use_task_resolution -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_surface_owner_live_llm.py::test_live_unavailable_reminder_surface_is_truthful -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_surface_owner_live_llm.py::test_live_unavailable_coding_owner_surface_is_truthful -q -s
```

The exact end-to-end cognition case may use a dedicated focused node if the
existing test harness can stop immediately after authorized resolver selection.
It must avoid invoking task resolution or a background worker.

Per-case artifacts must include rendered messages, route/model identity, raw
output, parsed output, validator disposition, partition or action decision,
latency, and agent-authored quality judgment.

### Broader regression and static checks

```powershell
$cognitionV2Tests = Get-ChildItem -LiteralPath 'tests' -Filter 'test_cognition_core_v2*.py' | Where-Object { $_.Name -ne 'test_cognition_core_v2_replay_clock.py' } | ForEach-Object { $_.FullName }
venv\Scripts\python.exe -m pytest $cognitionV2Tests -m "not live_llm and not live_db" -q
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py src\kazusa_ai_chatbot\cognition_core_v2\workspace.py src\kazusa_ai_chatbot\cognition_core_v2\facade.py tests\test_cognition_core_v2_goal_capability_live_llm.py tests\test_cognition_core_v2_workspace_live_llm.py
git diff --check
git status --short
```

Run the existing focused task-resolution public-current-fact live case
separately after cognition routing passes:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests\test_task_resolution_live_llm.py::test_live_public_current_fact_research -q -s
```

## Independent Code Review

The independent reviewer receives this plan, final diff, production trace
artifacts, pre-fix/post-fix LLM review, and all verification results. The
review must check:

- goal cognition expresses character semantics without operational capability
  invention;
- workspace collapse receives sufficient bounded provenance and remains the
  model owner of current relevance;
- deterministic code only maps provenance, validates structure, and enforces
  limits;
- genuine character refusal and autonomy remain available;
- action planning retains all runtime limits and affordances;
- the exact online-search request reaches task resolution;
- no new LLM call, schema alias, compatibility path, keyword classifier,
  worker change, or retry expansion appears;
- prompt caps, CJK source safety, tracing, and failure behavior remain sound;
- the adjacent required-selection draft has no unresolved implementation
  conflict.

All valid findings are remediated and affected verification is rerun before
signoff.

## Risks

- Capability-neutral goal wording could create an execution promise. The goal
  contract requires execution-neutral outcomes, and unavailable future/coding
  controls verify action-planning and surface truthfulness.
- Collapse could over-suppress durable motives. Matching autonomy and another
  matching persistent-goal control verify admission when the same matter is
  active.
- Added collapse context could exceed its prompt cap. Bounded projection and
  prompt-budget tests enforce the existing cap and handle coverage.
- Persistent-goal provenance may be missing because of state drift. Required
  internal data follows fail-fast handling and receives a deterministic test.
- Prompt stochasticity may vary. Exact captured cases, positive controls,
  bounded model attempts, and inspected artifacts provide the signoff gate.

## Execution Evidence

- Pre-edit git state: the worktree already contained the adjacent active
  required-selection plan and user-owned changes. No overlapping production
  implementation from that plan was present; all unrelated changes were
  preserved.
- Deterministic red gates reproduced the two contracts: goal cognition could
  invent a capability refusal, and ungrounded workspace collapse could select
  a bid from an unrelated persistent goal. Production-shaped live probes
  reproduced both before the cutover.
- Production files changed:
  `cognition_core_v2/goal_cognition.py`, `workspace.py`, `facade.py`, and the
  subsystem `README.md`. Action selection, authorization, task resolution,
  background work, persistence, adapters, and public schemas were unchanged.
- Focused non-live command: the nine changed Cognition V2 test modules with
  `-m "not live_llm and not live_db"`; result: 113 passed, 46 live tests
  deselected.
- Full runnable non-live Cognition V2 command: every
  `tests/test_cognition_core_v2*.py` except the independently broken
  `test_cognition_core_v2_replay_clock.py`; result: 350 passed, 180 live tests
  deselected.
- Adjacent conversation-projection command:
  `test_conversation_progress_cognition.py` plus
  `test_conversation_progress_cognition_evidence.py`; result: 10 passed.
- Exact goal artifact:
  `test_artifacts/llm_traces/cognition_core_v2_goal_capability_live_llm__captured_online_search_ordinary_goal.json`;
  one goal call at 6996.625 ms plus one quality call at 2285.598 ms; quality
  passed with no invented no-search limit.
- Stale and matching workspace artifacts:
  `cognition_core_v2_workspace_live_llm__captured_online_search_stale_autonomy.json`
  at 2098.456 ms and
  `cognition_core_v2_workspace_live_llm__matching_autonomy_goal.json` at
  1928.026 ms. The stale branch was suppressed and the matching autonomy
  branch was selected.
- Exact goal-to-action artifact:
  `cognition_core_v2_action_planning_live_llm__captured_online_search_goal_to_action.json`;
  action planning took 4728.023 ms, resolver authorization took 1140.556 ms,
  and the result was `requires_required_evidence` with one authorized
  `task_resolution_request` and no action or worker call.
- Unavailable-owner action controls ran individually. Reminder action planning
  returned `answerable_now` with no action/resolver substitution; coding-owner
  planning returned `blocked` with no action/resolver substitution.
- Fresh visible-surface artifacts:
  `cognition_core_v2_surface_owner_live_llm__unavailable_reminder_owner_surface.json`
  (content 36359.840 ms, judge 8592.544 ms) and
  `cognition_core_v2_surface_owner_live_llm__unavailable_coding_owner_surface.json`
  (content 7053.020 ms, judge 2552.199 ms). Both quality judgments passed.
- All real-model calls resolved to
  `gemma-4-31b-fable-5-agent-distill`; artifact route metadata records the
  stage-specific goal, workspace, action, authorization, and surface routes.
- Static checks: all changed Python files compiled; CJK prompt files compiled
  immediately after edits; the 88-character scan is clean; `git diff --check`
  is clean. `ruff` is not installed in the project virtual environment, so its
  command could not run and no package was installed.
- Independent review pass one found no critical or high-severity code defect.
  Its medium findings were stale lifecycle commands/evidence and missing fresh
  visible-surface owner controls. Commands and counts were corrected; both
  surface controls were added, run one at a time, and inspected. Final reviewer
  recheck approved with no unresolved findings.
- Known unrelated blocker: `test_cognition_core_v2_replay_clock.py` imports the
  absent `experiments.cognition_core_v2_real_conversation_replay` module. The
  runnable suite excludes only that collector.

Final signoff: all acceptance criteria are satisfied. The regression is fixed
before task-worker execution, unavailable-owner controls remain truthful at
action and visible-surface boundaries, and the independent reviewer approved
the remediated implementation and evidence with no unresolved findings.

Pre-plan validation artifacts are listed in
`test_artifacts/llm_traces/online_search_dual_root_cause_validation_review.md`.

## Acceptance Criteria

- The captured ordinary goal preserves the need for current external evidence
  without asserting that online search is unavailable.
- The captured stale autonomy bid is suppressed because its persistent goal
  concerns a different matter.
- A matching autonomy case remains primary or supporting according to
  character judgment.
- The captured goal-to-action case emits
  `requires_required_evidence` and an authorized
  `task_resolution_request`.
- The test stops before task worker execution while proving the inline
  resolver path would start.
- Unavailable future and coding effects remain truthful and blocked by their
  existing owners.
- Goal, bid, workspace partition, action, resolver, persistence, and worker
  public schemas remain unchanged.
- Healthy-path and worst-case LLM call counts and retry caps remain unchanged.
- Goal and workspace prompts stay within existing caps.
- Focused and full non-live regressions pass.
- Every live case runs separately with an inspected artifact and human-readable
  review.
- Independent review has no unresolved findings.
