# task resolution character background handoff plan

## Summary

- Goal: let the character choose one boolean route for generic
  `task_resolution_request` work, then expose validated partial progress before
  durable continuation when inline-first work does not finish.
- Status: completed
- Scope boundary: cognition V2 action selection, resolver recurrence,
  task-resolution checkpoint promotion, and the existing background-worker
  continuation path.
- Change direction: add `start_in_background: bool` to the semantic
  task-resolution request; `true` enters the existing durable handoff path
  immediately, while `false` keeps the existing inline-first path.
- Acceptance state: implementation completed under the user's explicit
  implementation instruction. Deterministic verification, compile checks,
  live LLM evidence, parent review, and independent final review are recorded
  below; the guarded full live-DB workflow remains fixture-blocked as noted in
  the live review artifact.

## Confirmed Decisions

- The character selects exactly one boolean for a task-resolution request:
  `start_in_background`.
- `start_in_background=true` creates the task-resolution checkpoint and enters
  the existing accepted-task/background-work queue path immediately.
- `start_in_background=false` runs inline using
  `TASK_RESOLUTION_INLINE_BUDGET_SECONDS` as an approximate foreground budget.
  A small overrun is acceptable; this plan introduces no strict timer protocol.
- A deferred inline result uses the same checkpoint for background continuation.
  The worker resumes the existing task limits and counters.
- When committed evidence exists at handoff, the final visible response presents
  that evidence first and the continuation notice second in the normal dialog
  response. The delivery path remains one visible response.
- When committed evidence is empty, the response contains the continuation
  notice without inventing a partial result.
- The existing `ResolverCapabilityRequestV1.priority` field remains the
  deterministic runtime projection: task-resolution `true` maps to `background`
  and `false` maps to `now`. Other resolver capabilities retain their current
  behavior.
- Existing accepted-task, background-job, result-source, and worker failure
  states remain authoritative. The character receives an acceptance statement
  only after durable queue success.

## Scope And Change Direction

The current generic task-resolution path is inline-first and automatically
promotes a validated deferred result. This change adds the character's narrow
initial routing decision and makes already-committed evidence visible in the
handoff observation.

The target flow is:

~~~text
action selection
  -> task_resolution_request(start_in_background=true)
  -> checkpoint -> accepted task -> background queue -> acknowledgement

action selection
  -> task_resolution_request(start_in_background=false)
  -> inline resolver using approximate budget
       -> resolved: normal result
       -> deferred: committed evidence + remaining needs -> same queue path
                    -> visible partial result followed by continuation notice

background worker
  -> resume same checkpoint -> existing result-ready/tool_result delivery
~~~

The boolean is the only model-selected routing value. Worker identity,
specialist choice, queue parameters, checkpoint contents, persistence, and
failure transitions remain deterministic runtime responsibilities.

## Mandatory Skills

- `development-plan`: governs this plan's lifecycle, scope, approval, and
  closeout.
- `local-llm-architecture`: preserves the narrow character decision and the
  deterministic queue/checkpoint boundary.
- `py-style`: applies to all Python implementation and test changes.
- `test-style-and-execution`: applies to deterministic and live test changes
  and execution.
- `debug-llm`: applies to the action-selection prompt change and inspected
  partial-handoff behavior.
- `python-venv`: applies to every Python and pytest command.
- `cjk-safety`: applies when editing the existing Python prompts containing CJK
  text.

## Mandatory Rules

- Use `venv\Scripts\python.exe` for Python checks and tests.
- Apply strict structural validation to `start_in_background`; accept JSON
  booleans only. Treat strings, numbers, missing values, and extra route fields
  as contract errors handled by the existing bounded regeneration/fail-closed
  path.
- Preserve the existing LLM error-state contract and keep malformed model
  output out of action, queue, persistence, and delivery paths.
- Preserve `TASK_RESOLUTION_INLINE_BUDGET_SECONDS` as an approximate
  cooperative budget. The existing outer resolver capability timeout remains
  the failure ceiling.
- Promote only a validated deferred result with a validated checkpoint.
- Surface only evidence already committed to that checkpoint. A cancelled or
  failed in-flight specialist call contributes no partial evidence.
- Reuse the existing accepted-task idempotency key and queue lifecycle. A retry
  must return or reuse the existing active continuation instead of creating a
  second task.
- Keep final visible wording owned by cognition/dialog. Deterministic code
  supplies typed evidence and continuation state.
- Keep the production database read-only during verification and preserve all
  unrelated worktree changes.

## Contracts And Data Shapes

For a `task_resolution_request`, the action-planning semantic row becomes:

~~~json
{
  "bid_handle": "...",
  "resolver_handle": "...",
  "semantic_goal": "...",
  "reason": "...",
  "start_in_background": false
}
~~~

The field is required only for the generic task-resolution resolver request.
Other resolver request rows retain their current shape and execution behavior.
The deterministic materializer preserves the boolean and projects it to the
existing V1 `priority` field.

The deferred handoff observation keeps the existing resolver observation
contract. When the deferred result contains evidence, it also carries the
existing `evidence_refs` and `knowledge_projection` fields used by resolved and
partial task results:

- `knowledge_we_know_so_far`: summaries of committed evidence;
- `knowledge_still_lacking`: checkpoint remaining needs;
- `evidence_boundary_notes`: the incomplete scope and continuation boundary.

The internal task result remains `status="deferred"` so the existing promotion
contract remains valid. The visible observation communicates both accepted
continuation and available knowledge through existing typed projections.

## Failure Modes And Recovery

| Failure | Required outcome |
|---|---|
| Missing, non-boolean, or malformed `start_in_background` | Bounded action-planning regeneration; after exhaustion, fail closed with no task or queue claim. |
| Task-resolution capability unavailable or unauthorized | Existing blocked/failed observation; no background acceptance statement. |
| Invalid execution context or checkpoint creation failure | Failed resolver observation; no accepted task and no queue entry. |
| Accepted-task creation failure | Existing failure observation; no acceptance statement. |
| Pending-state or queue insertion failure | Reuse existing enqueue-failure transition and report that continuation was not made durable. |
| Duplicate planning, retry, or delivery after successful enqueue | Existing accepted-task identity and `background_work:<accepted_task_id>` idempotency return one continuation. |
| Inline work resolves before handoff | Return the validated result and create no background task. |
| Inline work returns deferred with committed evidence | Promote the same checkpoint and expose evidence before the continuation notice. |
| Inline work returns deferred with no evidence | Promote the same checkpoint and expose only the continuation notice. |
| Specialist failure, unavailability, or outer capability timeout before a valid deferred result exists | Use the existing failed/unavailable observation; preserve truthful failure state and create no unmaterialized continuation. |
| Background worker is delayed or never claimed | Accepted task remains pending and existing status-check behavior remains authoritative. |
| Background worker reaches failed, unavailable, needs-user-input, or approval-required state | Existing typed result delivery reports that state; no final success claim is generated. |
| Current-turn acknowledgement fails after queue success | The durable accepted task remains the source of truth; retry and later result delivery use the existing idempotent path. |

## Must Do

1. Update the action-selection prompt and validator so a generic
   `task_resolution_request` returns the exact `start_in_background` boolean.
2. Preserve the boolean through resolver authorization and semantic request
   materialization, while keeping authorization focused on evidence need and
   capability match.
3. Project the boolean to the existing V1 resolver priority and add the
   deterministic background branch to the resolver recurrence.
4. Add a task-resolution service entry point for immediate background start
   that creates the initial checkpoint and reuses deferred-result promotion,
   accepted-task lifecycle, queue idempotency, and existing worker payloads.
5. Update deferred task-resolution observation projection so committed partial
   evidence and remaining needs reach the final cognition/dialog pass.
6. Preserve the current approximate inline budget and the existing worker
   resume behavior; keep the direct-background path free of inline specialist
   execution.
7. Add deterministic coverage for success, partial handoff, empty-evidence
   handoff, malformed boolean, unavailable capability, checkpoint failure,
   enqueue failure, duplicate enqueue, worker failure, and delayed pending
   state.
8. Add one-at-a-time live LLM evidence for the character selecting `true` and
   `false`, plus an inspected partial-handoff response proving that committed
   evidence precedes the continuation notice.
9. Update the cognition, resolver, and task-resolution READMEs to describe the
   boolean and the failure truthfulness boundary.

## Deferred

- Strict wall-clock enforcement or a new timer/cancellation protocol.
- New worker types, specialist handles, resolver statuses, action-spec kinds,
  or background payload shapes.
- Two separate visible chat messages for partial evidence and handoff.
- Changes to coding-agent behavior, accepted-task status/check operations,
  result-source delivery, adapter behavior, or worker scheduling intervals.
- Resetting task-resolution counters or extending the semantic dispatch/call
  limits during background continuation.
- Compatibility aliases, legacy parallel request shapes, fallback keyword
  routing, deterministic response text, or unrelated cleanup.
- Production requeue, database migration, or deployment changes.

## Target State

The character's action selector chooses one narrow semantic boolean. Runtime
code validates and routes it, then the existing task-resolution and
background-work ownership boundaries remain intact.

Inline-first handoff produces a resolver context equivalent to:

~~~text
accepted for continued work
known so far: <validated committed evidence>
still lacking: <checkpoint remaining needs>
boundary: <bounded partial scope>
~~~

The final dialog uses that context to present the partial result first and the
continuation notice second. A direct background selection produces an accepted
continuation acknowledgement without claiming that research has completed.

## Cutover Policy

Overall strategy: bigbang for the action-planning contract and runtime route.
Existing persisted accepted-task, checkpoint, background-job, and result-source
shapes remain in place; no data migration is required.

| Area | Policy | Instruction |
|---|---|---|
| Action-planning output | bigbang | Update the exact resolver request contract and all callers, validators, fixtures, and tests together. |
| Resolver runtime | bigbang | Replace the single inline-only generic route with the boolean-controlled route. |
| Persisted task/background state | compatible | Reuse the existing v1 checkpoint, accepted-task, job, and worker payload contracts without adding a second shape. |
| Visible wording | compatible | Preserve normal cognition/dialog ownership and existing result delivery. |

## Change Surface

### Delete

- None.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`: prompt,
  request validation, normalization, and trace projection for the boolean.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`: typed and structural
  validation of the task-resolution boolean.
- `src/kazusa_ai_chatbot/cognition_core_v2/resolver_authorization.py`: preserve
  the accepted boolean while keeping authorization responsible for evidence
  need and capability match.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: project the
  boolean to the existing resolver priority field.
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py`: process one authorized
  background task-resolution request through the deterministic handoff branch.
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`: dispatch direct
  background start and project deferred committed evidence into the existing
  knowledge observation.
- `src/kazusa_ai_chatbot/task_resolution/service.py`: create an initial
  checkpoint and reuse deferred promotion for direct background start.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`: document the narrow
  boolean ownership and runtime routing boundary.
- `src/kazusa_ai_chatbot/cognition_resolver/README.md`: document immediate
  background start and partial deferred observation.
- `src/kazusa_ai_chatbot/task_resolution/README.md`: document checkpoint reuse,
  approximate inline budget, and failure states.
- `tests/test_action_selection_prompt_contract.py`: assert the boolean prompt
  and exact route field contract.
- `tests/test_action_selection_payload.py`: cover task-resolution route payload
  shape.
- `tests/test_cognition_core_v2_action_planning_bugfix.py`: cover boolean
  validation and propagation through action planning.
- `tests/test_cognition_core_v2_contracts.py`: cover the V2 request contract.
- `tests/test_cognition_chain_connector_mapping.py`: cover projection into the
  resolver request consumed by the persona graph.
- `tests/test_cognition_resolver_l2d_contract.py`: cover immediate background
  and deferred partial observation behavior.
- `tests/test_task_resolution_inline_promotion.py`: cover partial evidence and
  failure-aware promotion behavior.
- `tests/test_task_resolution_background_resume.py`: cover same-checkpoint
  resume, idempotency, and worker terminal failures.
- `development_plans/README.md`: register this active short-term plan.

### Create

- `tests/test_task_resolution_character_background_handoff.py`: focused
  deterministic route and failure-mode coverage not belonging to an existing
  contract suite.
- `test_artifacts/diagnostics/task_resolution_character_background_handoff_live_review.md`:
  parent-authored live LLM quality review.
- `test_artifacts/diagnostics/task_resolution_character_background_handoff_live_*.json`:
  raw or protected live evidence following the repository diagnostic-artifact
  convention.

### Keep

- `src/kazusa_ai_chatbot/task_resolution/orchestrator.py`: existing
  approximate inline checks and bounded semantic task limits.
- `src/kazusa_ai_chatbot/background_work/subagent/task_orchestrator.py`:
  existing same-checkpoint background resume.
- `src/kazusa_ai_chatbot/accepted_task/`: existing lifecycle and idempotency
  ownership.
- `src/kazusa_ai_chatbot/background_work/`: existing queue, worker, retry, and
  result delivery ownership.
- `src/kazusa_ai_chatbot/action_spec/`: generic task-resolution routing remains
  a resolver capability rather than an action-spec capability.

## Agent Autonomy Boundaries

The implementation agent may choose local helper names, function decomposition,
test fixture construction, and Markdown wording while preserving the fixed
boolean name, route semantics, existing priority projection, failure outcomes,
and change surface.

The implementation agent must keep malformed booleans fail-closed, preserve
checkpoint and queue idempotency, keep partial evidence checkpoint-backed, and
retain final wording ownership in cognition/dialog. It must not add a second
route field, new worker protocol, strict timer mechanism, compatibility alias,
deterministic fallback sentence, or unrelated migration.

If the current contract cannot support the stated boolean without changing a
path outside this change surface, the implementation pauses for a plan
amendment before editing that path.

## Verification

1. Capture the worktree baseline and run the existing focused action-planning,
   resolver, inline-promotion, and background-resume tests.
2. Run deterministic tests with the project virtual environment covering the
   modified test files and the new focused handoff suite.
3. Verify the `true` path reaches accepted-task pending state and one durable
   queue job without invoking an inline specialist.
4. Verify the `false` path resolves inline without queueing when complete, and
   promotes one same-checkpoint job when deferred.
5. Verify deferred observations with and without evidence, including the
   projected knowledge order and the absence of invented partial text.
6. Verify every listed failure mode through deterministic tests, including
   enqueue failure, duplicate retry, delayed pending state, and terminal worker
   failure.
7. Run one live LLM case for each boolean route individually, inspect raw model
   output, parsed contract, selected route, and trace status, and save the
   quality review artifact.
8. Run one individually inspected partial-handoff case and verify that the
   visible response uses committed evidence before the continuation notice.
9. Run compile checks for changed Python packages, `git diff --check`, and a
   complete diff/scope review.

## Acceptance Criteria

1. A valid task-resolution request contains exactly one validated
   `start_in_background` boolean and the boolean survives authorization and
   runtime projection.
2. `true` enters the durable checkpoint, accepted-task, and background queue
   path without inline specialist execution.
3. `false` preserves inline-first behavior with an approximate budget; complete
   work remains inline and deferred work resumes from the same checkpoint.
4. A deferred result with evidence exposes only committed evidence and
   remaining needs, with the partial content preceding the continuation notice.
5. A deferred result without evidence produces no fabricated partial result.
6. Malformed model output, unavailable capability, checkpoint failure, enqueue
   failure, duplicate retry, delayed worker, and terminal worker failure all
   produce truthful existing failure or pending states without a false
   completion claim.
7. Queue success remains durable even when the current-turn acknowledgement
   fails, and retries remain idempotent.
8. Existing worker resume, accepted-task, result-source, adapter, and status
   check behavior remains intact.
9. Focused deterministic tests, inspected live LLM evidence, compile checks,
   whitespace checks, documentation updates, and final scope review pass.

## Progress Checklist

- [x] User explicitly commands implementation and the approval boundary is
      satisfied.
- [x] Baseline and focused pre-change verification are recorded.
- [x] Boolean action-planning contract and prompt are updated.
- [x] Direct background handoff and inline deferred projection are implemented.
- [x] Failure-mode deterministic tests pass.
- [x] Individually inspected live LLM route and partial-handoff evidence are
      recorded.
- [x] Documentation and registry entries are updated.
- [x] Parent diff review, final verification, and residual-risk review are
      complete.
- [x] Plan lifecycle is closed with acceptance evidence.

## Execution Evidence

- Focused deterministic implementation suites: 170 passed.
- Independent adjacent review suites: 216 passed in total across the
  reviewer's focused and adjacent commands.
- Changed-source compile checks and `git diff --check`: passed.
- Live planner and dialog evidence: [task resolution character background
  handoff live review](../../../../test_artifacts/diagnostics/task_resolution_character_background_handoff_live_review.md).
- Independent final DeepSeek review: PASS; no acceptance findings.
- Full guarded live-DB workflow: fixture-blocked before task execution because
  the test database lacks the `character-global` identity revision; no
  production database was used.
