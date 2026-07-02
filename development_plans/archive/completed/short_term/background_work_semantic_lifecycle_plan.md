# background work semantic lifecycle plan

## Summary

- Goal: Make delayed/background work transparent to the character by exposing
  only a semantic accepted-task lifecycle, while keeping the existing
  background-work worker queue as an internal executor.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`, `py-style`, `cjk-safety`,
  `test-style-and-execution`
- Overall cutover strategy: migration. New delayed work uses an
  `accepted_task` lifecycle boundary. `background_work_jobs` remains an
  internal worker queue and must stop being the model-facing or
  character-facing contract.
- Highest-risk areas: duplicate scheduling from repeated cognition cycles,
  prompt leakage of job/worker/background vocabulary, result-ready delivery
  failing silently, live-turn latency from the post-schedule cognition loop,
  overbroad duplicate identity, and source-case drift in self-cognition.
- Acceptance criteria: the character only sees ordinary accepted-task states,
  a picked delayed task is durably scheduled before acknowledgement, the
  acknowledgement is produced after a bounded cognition read of the scheduled
  state, completed work actively re-enters cognition and sends the result, and
  repeated cognition/self-cognition/progress checks cannot enqueue duplicate
  work for the same active accepted task.

## Context

The user requested a design change for delayed/background work:

- Background work should be transparent to the character.
- The character should not know whether work is background work or normal work.
- If delayed work is picked, the system should immediately return to a
  cognition cycle with a state like "the task is scheduled and the user can be
  told to wait."
- When the background schedule completes, it should trigger a source-bound
  proactive cognition delivery path, so the character actively tells the user
  the task is done with the artifact.
- The lifecycle is complete after delivery.
- Failure mode: background work must not be scheduled repeatedly. A progress
  checking and repeat rejection system is required because self-cognition may
  otherwise schedule duplicate work.

The current code already has several useful pieces:

- `src/kazusa_ai_chatbot/background_work/` owns durable internal queueing,
  worker routing, worker dispatch, and result-ready delivery.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` queues background work
  in `stage_2a_background_work_enqueue` before L3 builds the acknowledgement.
  This preserves the invariant that an acknowledgement should only happen
  after durable queue persistence.
- `src/kazusa_ai_chatbot/background_work/delivery.py` converts completed jobs
  into source-bound cognition episodes instead of letting workers send adapter
  messages directly.
- `src/kazusa_ai_chatbot/cognition_chain_core/episode_projection.py` forces
  `background_work_result_ready` to `visible_reply`, which is close to the
  desired active follow-up behavior.
- `src/kazusa_ai_chatbot/self_cognition/` already has useful patterns for
  source-case projection, suppression/audit ledgers, and dispatcher handoff,
  but those ledgers suppress visible self-cognition sends, not accepted user
  task scheduling.

The current code does not yet meet the proposed design because the public
cognition/action boundary is still named and shaped around background jobs.

## Current Codebase Gap Map

| Requirement | Current implementation | Gap | Risk |
|---|---|---|---|
| Character should not know background vs normal work. | `action_spec.registry._background_work_projection()` exposes `background_work_request`, "background text work", and "background-work queue" wording to runtime affordances. `action_spec.execution` projects `queue_state`, `operational_owner`, and `job_ref` into action results. | The model sees implementation vocabulary and job mechanics. | The character can talk like an operator instead of treating the work as an ordinary accepted task. |
| Background work should be transparent to the character. | `cognition_episode.build_background_work_result_ready_cognitive_episode()` uses trigger/input sources named `background_work_result_ready` and `background_work_result`, and metadata includes `worker` and `worker_metadata`. | Result-ready cognition is source-bound, but still background/worker flavored. | Prompt leakage and final wording can reveal executor internals. |
| Picked delayed work should return to cognition after scheduling. | `persona_supervisor2.stage_2a_background_work_enqueue()` queues before L3 and stores pre-surface action results. | The current handoff is an action-result surface input, not a bounded post-schedule cognition observation. | The acknowledgement may be based on action execution fields rather than character cognition over a semantic scheduled-task state. |
| Work must be scheduled once only. | `action_spec.handlers.background_work.enqueue_background_work_action()` builds `idempotency_key` from `action_attempt_id`. `background_work_jobs` has a unique index on that key. | Idempotency protects retry of the same action attempt only. A second cognition cycle can emit a new action attempt and enqueue a duplicate. | Self-cognition, progress checks, or user repeats can create duplicate jobs for the same active task. |
| Progress checking must reject repeats. | There is no first-class accepted-task progress lookup or progress-check action. | A user asking "how is that going?" can be interpreted as a new delayed-work request. | Duplicate scheduling and confusing acknowledgements. |
| Self-cognition must not schedule duplicates. | Self-cognition has send suppression and attempt ledgers, but background-work requests are handled by the generic action execution path. | There is no source-aware policy that prevents self-cognition/result-ready sources from creating a new accepted delayed task for the same active work. | Autonomous source cases can reopen or duplicate accepted user work. |
| Completed work should actively notify the user. | Background-work delivery re-enters cognition and dispatcher delivery. | Delivery still depends on the persona path producing final dialog, and `delivery_in_progress` has no lease-style recovery if the process crashes mid-delivery. | Result can stall without a terminal lifecycle state or visible artifact delivery. |
| Lifecycle should be complete after delivery. | `background_work_jobs` has delivered/failed states, but no higher-level accepted-task state visible to cognition. | The user-facing lifecycle is coupled to worker job states. | Progress reporting and completion semantics cannot be stated without job vocabulary. |
| Contract/docs should match implementation. | `development_plans/archive/completed/short_term/l2d_action_router_prompt_separation_plan.md` implies an action-router module boundary, but `src/kazusa_ai_chatbot/action_router/` currently contains only `__pycache__`. | The action-selection ownership boundary should be reverified before editing. | A plan may target a stale module name instead of the actual `cognition_chain_core.action_selection` and `persona_supervisor2_cognition_actions` owners. |
| Tests should cover the new lifecycle. | Existing tests cover route-only background-worker separation, worker-local fields, and result-ready delivery. | There are no tests for accepted-task identity, duplicate rejection across turns, progress checks, self-cognition duplicate rejection, post-schedule cognition observation, or prompt leakage of background/job/worker vocabulary. | Regressions would pass the current test suite. |

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, updating,
  or signing off this plan.
- `local-llm-architecture`: load before changing cognition graph flow, L2d
  affordances, prompt inputs, result-ready source contracts, or LLM budgets.
- `no-prepost-user-input`: load before changing acceptance, duplicate,
  progress, or persistence logic for user-directed tasks.
- `debug-llm`: load before live/local LLM runs, prompt comparison, or quality
  review artifacts.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files that contain CJK strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute implementation steps while `Status` is `draft`.
  Implementation requires user approval and status `approved` or
  `in_progress`.
- Plan status is not production-code authorization. Production-code edits still
  require an explicit implementation instruction from the user.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or final
  reporting.
- After signing off any major checklist stage, reread this entire plan before
  starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- Use parent-led native subagent execution. If native subagent capability is
  unavailable, stop before production implementation unless the user explicitly
  approves fallback execution.
- Use `venv\Scripts\python.exe` for Python commands.
- Never read `.env` during implementation or verification.
- Deterministic code owns accepted-task identity, atomic duplicate rejection,
  persistence, state transitions, source permissions, retry limits, delivery
  status, and adapter delivery audit.
- The action materializer must thread the cognitive episode trigger source into
  trusted action target scope as `source_trigger_source`. Lifecycle creation
  policy must read that trusted field, not raw user text.
- LLM stages own semantic judgment: whether the user request should be
  accepted, whether a reply/progress answer is needed, and how the final
  acknowledgement/result should be worded.
- Do not add deterministic keyword matching over raw user text to decide
  whether the user asked for progress or asked to create new work. The
  deterministic layer may only validate structured LLM/action outputs and
  trusted source metadata.
- Do not expose job ids, queue ids, worker names, leases, retry counters, raw
  adapter ids, or DB field names in model-visible character/cognition surfaces.
- Do not route completed jobs directly from a worker to an adapter. Result
  delivery must continue through source-bound cognition and dispatcher
  validation.
- Do not turn the calendar scheduler into the generic background-work queue.

## Must Do

- Add a first-class accepted-task lifecycle boundary above `background_work`.
  The lifecycle must be the only model-facing contract for delayed user work.
- Add a small `accepted_task` module and README/ICD that defines ownership,
  states, source rules, duplicate policy, prompt projection policy, and public
  functions.
- Add a durable accepted-task repository in
  `src/kazusa_ai_chatbot/db/accepted_tasks.py` with atomic active-task
  duplicate rejection. It must be backed by a separate `accepted_tasks`
  collection so user-facing task state is not coupled to worker queue state.
- Add an accepted-task enqueue lock for every new delayed work request before
  inserting the internal `background_work_jobs` row.
- Use this exact creation order for new work: create or claim an
  `enqueueing` accepted-task record through the active unique key; insert the
  internal background-work job with idempotency key
  `background_work:{accepted_task_id}`; mark the accepted task `pending` with
  the internal executor ref; only then allow user acknowledgement.
- If internal job insertion fails after the accepted-task lock is created, mark
  the accepted task `enqueue_failed`, release the active uniqueness claim, and
  return a prompt-visible failure state. Do not acknowledge a promise for a
  task that has no durable worker job.
- Add recovery for stale `enqueueing` accepted tasks so a process crash between
  task creation and job insertion cannot permanently block a retry.
- Add `accepted_task_id` and `task_identity_key` threading into internal
  background-work queue rows for audit and state synchronization.
- Build task identity from trusted conversation/requester scope plus the
  structured semantic action output already accepted by cognition. Do not
  classify raw user text in code. Do not include `source_message_id` in the
  duplicate identity because repeat turns and progress checks have different
  source messages.
- Replace prompt-facing background-work wording with ordinary accepted-task
  wording. The model may understand that a task has been accepted and will take
  time; it must not see queue/job/worker/background mechanics.
- Change the background-work action execution result projection to return
  semantic accepted-task states such as `scheduled`, `already_active`,
  `running`, `result_ready`, `delivered`, or `failed`, without `job_ref` or
  `operational_owner` in prompt-visible fields.
- Add a bounded post-schedule cognition observation after durable task
  creation or duplicate rejection. This observation must let cognition/L3
  produce the acknowledgement or progress answer after reading a semantic task
  state, and it must not allow scheduling the same task again in that cycle.
- Add deterministic source rules so self-cognition, result-ready, and
  post-schedule observation sources cannot create a new accepted delayed task
  unless a user-message turn explicitly starts a new task through the normal
  route.
- Add an explicit private action capability,
  `accepted_task_status_check`, owned by `accepted_task`. L2d may emit this
  capability when it semantically decides the user is asking about active work.
  The handler must bind to active accepted-task state and must never enqueue a
  worker job.
- Migrate completed background-work result delivery to emit an
  `accepted_task_result_ready` cognition source for new accepted tasks. Keep
  old `background_work_result_ready` handling only for already persisted legacy
  rows that do not have an accepted-task id.
- Remove `worker` and `worker_metadata` from model-visible result-ready
  metadata. Keep worker details audit-only.
- Add result delivery state synchronization from internal worker completion to
  accepted-task `result_ready`/`failed`, and from dispatcher success/failure to
  accepted-task `delivered`/`delivery_failed`.
- Add recovery for stuck accepted-task delivery attempts so
  `delivery_in_progress` cannot remain the final observable lifecycle state
  forever after a process crash.
- Update README/HOWTO/subsystem READMEs so the documented design matches the
  accepted-task lifecycle and the remaining internal role of background work.
- Add deterministic and live LLM tests listed under `Verification`.
- Add or replace the L2d/accepted-task live LLM delayed-work selection case so
  the model sees the correct action affordances and emits the expected
  accepted delayed-work action path. The plan cannot be signed off while the
  current live failure in
  `test_artifacts\llm_traces\l2d_action_selection_live_llm__coding_snippet_accept_fibonacci.json`
  remains unresolved.

## Deferred

- Do not implement new worker types, coding-agent integration, shell execution,
  package installs, downloads, web browsing workers, image generation workers,
  or chunked artifact delivery in this plan.
- Do not decommission the internal `background_work` worker queue. It remains
  the executor behind accepted tasks.
- Do not remove legacy `background_artifact` compatibility handling for old
  rows.
- Do not rewrite unrelated RAG, reflection, memory lifecycle, calendar,
  adapters, dispatcher, or consolidation contracts.
- Do not add a semantic duplicate-detection LLM. Duplicate rejection must be
  deterministic over structured task identity and trusted scope.
- Do not invent numeric wait times unless a deterministic estimate exists.
  The character may say the task will take a little while, but not fabricate an
  ETA.
- Do not create a compatibility alias layer with parallel model-facing
  vocabularies. New model-facing delayed-work language is accepted-task
  language.

## Cutover Policy

Overall strategy: migration.

| Area | Policy | Instruction |
|---|---|---|
| Model-facing delayed-work contract | migration | Replace background-work wording and result fields with accepted-task lifecycle wording. |
| Internal executor | compatible | Keep `background_work` as the internal queue/router/worker runtime, but add `accepted_task_id` threading for new jobs. |
| New delayed-work requests | migration | New requests create or reuse an accepted-task record before any internal job is queued. |
| Duplicate handling | bigbang | Duplicate active accepted tasks must be rejected or converted to progress state at the lifecycle boundary before job insertion. |
| Accepted-task enqueue atomicity | bigbang | New jobs must pass through `enqueueing -> pending` and must not be acknowledged until both the accepted-task row and internal worker job are durable. |
| Post-schedule cognition | migration | Add one bounded no-reschedule cognition observation after accepted-task scheduling or duplicate detection. |
| Result-ready source | migration | New accepted tasks use `accepted_task_result_ready`; old background-work result-ready handling remains only for persisted legacy rows without accepted-task ids. |
| Self-cognition | migration | Self-cognition may report/progress accepted tasks, but must not create duplicate accepted tasks for existing active work. |
| Tests | additive plus migration | Add new accepted-task lifecycle tests and update old background-work tests where prompt-facing contracts intentionally change. |
| Docs | migration | Update README/HOWTO/background-work/action-spec/self-cognition/brain-service docs to use the new ownership model. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- If an area is `migration`, update caller, callee, tests, and docs in the
  same implementation scope.
- If an area is `compatible`, preserve only the explicitly listed legacy
  behavior.
- Legacy `background_work_result_ready` handling must be marked as legacy-row
  compatibility and must not be used for new accepted-task rows after cutover.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

The live delayed-work path becomes:

```text
user message
  -> cognition/action selection decides an ordinary accepted delayed task
  -> deterministic accepted_task lifecycle resolver
       create enqueueing accepted_task, or return existing active task
  -> internal background_work job enqueue only for newly created task
  -> accepted_task marked pending with internal executor ref
  -> bounded post-schedule cognition observation with scheduling disabled
  -> L3 acknowledgement/progress wording
  -> user waits
  -> background_work worker completes internal job
  -> accepted_task moves to result_ready or failed
  -> accepted_task_result_ready cognitive episode
  -> normal cognition/dialog/dispatcher sends artifact or failure note
  -> accepted_task moves to delivered or delivery_failed
```

The character-facing model sees:

- the user's request;
- whether Kazusa accepted a task;
- whether the task is pending, running, complete, failed, or already active;
- the completed artifact or failure summary when ready.

The character-facing model does not see:

- `background_work_request`;
- `background_work_result_ready`;
- `job_ref`;
- `queue_state`;
- `operational_owner`;
- worker names;
- worker-local task types;
- leases;
- retry counters;
- DB ids or adapter internals.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Public lifecycle owner | Add an `accepted_task` lifecycle boundary above `background_work`. | The user-facing lifecycle is not the same thing as an executor queue row. |
| Internal executor | Keep `background_work` for queue, router, worker, and artifact production. | The current worker architecture already has the right bounded execution and result-ready separation. |
| Duplicate key | Use a deterministic active-task identity key built from trusted requester/channel scope plus the accepted task seed/detail. Exclude `source_message_id`; store message ids only as provenance. | This rejects repeated scheduling across turns without keyword parsing raw user text. |
| Enqueue consistency | Use an `enqueueing` accepted-task lock before internal job insertion, then move to `pending` only after job insertion succeeds. | This preserves the existing "acknowledge only after durable work exists" invariant while closing the duplicate race. |
| Progress capability | Add `accepted_task_status_check` as a private action capability. | Progress questions need a structured route that resolves lifecycle state without creating work. |
| Unique scope | Enforce uniqueness only for active terminal-incomplete states. | The same kind of task can be requested again after a prior task is delivered or cancelled. |
| Progress checks | Add structured progress handling against accepted-task state. | Progress questions should answer from lifecycle state, not create a new worker job. |
| Self-cognition policy | Self-cognition may surface progress or completion but cannot create new accepted work for an existing active task. | This closes the duplicate failure mode without blocking result delivery. |
| Post-schedule loop | Add exactly one bounded post-schedule cognition observation with delayed-task scheduling disabled. | This satisfies the user's "return to cognition cycle" requirement while preventing recursive enqueue. |
| Result-ready source | Use `accepted_task_result_ready` for new accepted tasks. | The completion event is user-work lifecycle state, not worker implementation state. |
| Wait wording | Do not invent numeric ETA. | The system has no deterministic duration estimate. |
| Audit data | Keep worker/job data audit-only and DB-visible. | Operators still need diagnostics, but the character should not see internals. |

## Contracts And Data Shapes

### Accepted Task Record

Add this durable record shape:

```text
schema_version: accepted_task.v1
accepted_task_id: task-...
task_identity_key: sha256(...)
task_identity_material:
  source_platform
  source_channel_id
  source_channel_type
  requester_global_user_id
  requester_platform_user_id
  accepted_task_seed
  accepted_task_detail
first_source_message_id: provenance only, not duplicate identity
related_source_message_ids: bounded provenance list, not duplicate identity
source_trigger_source: user_message | self_cognition | accepted_task_result_ready |
                       background_work_result_ready | other supported source
state: enqueueing | pending | running | result_ready | failure_ready |
       delivery_in_progress | delivery_retryable | delivered |
       enqueue_failed | delivery_exhausted | cancelled | superseded
result_kind: none | artifact | failure
executor_kind: background_work
executor_ref: internal job id, audit-only
accepted_task_summary: prompt-safe task summary
source_context: prompt-safe reason/context
requested_delivery: send_result_when_done
max_output_chars: int
created_at / updated_at
started_at / completed_at / delivered_at
result_summary
failure_summary
delivery_failure_summary
last_progress_reported_at
```

The repository must create indexes for:

- `accepted_task_id` unique;
- `task_identity_key` unique through a partial index for active states only:
  `enqueueing`, `pending`, `running`, `result_ready`, `failure_ready`,
  `delivery_in_progress`, and `delivery_retryable`;
- `(state, updated_at)` for progress/result delivery scans;
- requester/channel lookup for progress checks.

### Prompt-Visible Action Result

New prompt-visible action result fields for delayed work should be shaped like:

```text
accepted_task_state: scheduled | already_active | running |
                     result_ready | delivered | failed |
                     enqueue_failed | delivery_failed
accepted_task_summary: string
acknowledgement_constraint: promise_allowed |
                            progress_report_allowed |
                            promise_forbidden_explain_failure
wait_guidance: non_numeric_wait | no_wait | unavailable
```

No prompt-visible action result should include job ids, job refs, queue state,
operational owner, worker, lease, retry, or DB field names.

### Cognitive Episode Sources

Add or migrate model-facing sources to semantic accepted-task vocabulary:

- post-schedule/progress observation: current user-message episode plus an
  `accepted_task_status` percept that does not expose internal executor state;
- completed result: `trigger_source = accepted_task_result_ready`;
- completed result input: `input_source = accepted_task_result`;
- legacy persisted background-work rows without `accepted_task_id` may still
  use `background_work_result_ready` until drained.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/accepted_task/README.md`: public ICD for accepted-task
  lifecycle ownership, states, action capabilities, source rules, and prompt
  projection policy.
- `src/kazusa_ai_chatbot/accepted_task/models.py`: typed record, state, queue
  request, status-check request, and prompt projection contracts.
- `src/kazusa_ai_chatbot/accepted_task/lifecycle.py`: public lifecycle
  functions for create-or-return-active, mark-pending, status-check,
  worker-complete, delivery-start, delivery-success, delivery-failure, and
  stale-enqueueing recovery.
- `src/kazusa_ai_chatbot/db/accepted_tasks.py`: MongoDB repository and index
  owner for the `accepted_tasks` collection.
- `tests/test_accepted_task_lifecycle.py`: focused lifecycle repository and
  duplicate/race contract tests.
- `tests/test_accepted_task_prompt_contract.py`: prompt-visible projection
  tests for no background/job/worker vocabulary in accepted-task surfaces.

### Modify

- DB startup/index wiring: call `ensure_accepted_task_indexes()` beside other
  service-owned collection index setup.
- `src/kazusa_ai_chatbot/background_work/models.py`: add
  `accepted_task_id` and `task_identity_key` as internal audit fields for new
  jobs.
- `src/kazusa_ai_chatbot/background_work/jobs.py`: accept
  `accepted_task_id`, use `background_work:{accepted_task_id}` idempotency for
  new jobs, and keep old action-attempt idempotency only for legacy callers
  still explicitly listed in tests.
- `src/kazusa_ai_chatbot/background_work/worker.py`: mark accepted tasks
  `running`, `result_ready`, or `failure_ready` when the internal worker state
  changes.
- `src/kazusa_ai_chatbot/background_work/delivery.py` and
  `src/kazusa_ai_chatbot/background_work/result_source.py`: emit
  `accepted_task_result_ready` for jobs with `accepted_task_id`, keep legacy
  `background_work_result_ready` only for rows without `accepted_task_id`, and
  synchronize delivery states.
- `src/kazusa_ai_chatbot/action_spec/registry.py`: add
  `accepted_task_status_check`, change delayed-work affordance wording to
  accepted-task language, and remove prompt-visible queue/job fields from the
  background-work output schema.
- `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py`: call
  accepted-task lifecycle creation before internal enqueue, enforce
  `source_trigger_source`, and return semantic accepted-task results.
- Add `src/kazusa_ai_chatbot/action_spec/handlers/accepted_task.py` if the
  status-check handler does not fit cleanly in the existing background-work
  handler.
- `src/kazusa_ai_chatbot/action_spec/execution.py`: project only accepted-task
  prompt fields and execute `accepted_task_status_check`.
- `src/kazusa_ai_chatbot/cognition_chain_core/action_selection.py` and related
  prompt contract files: normalize `accepted_task_status_check` without worker
  parameters and keep accepted delayed-work creation separate from status
  checks.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: add the one-pass
  post-schedule cognition observation with delayed-task creation disabled.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: include
  accepted-task status percepts and remove background-work wording from
  model-facing affordances.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`:
  materialize `source_trigger_source`, build accepted-task creation and
  status-check action specs, and enforce no delayed-task creation from
  post-schedule/result-ready/self-cognition sources.
- `src/kazusa_ai_chatbot/cognition_episode.py`: add
  `accepted_task_result_ready`, `accepted_task_result`, and
  `accepted_task_status` source/input support.
- `src/kazusa_ai_chatbot/cognition_chain_core/episode_projection.py`: project
  accepted-task sources without worker metadata and force result-ready accepted
  tasks to visible reply.
- `src/kazusa_ai_chatbot/self_cognition/sources.py`,
  `self_cognition/projection.py`, and `self_cognition/runner.py`: surface
  accepted-task progress context and rely on lifecycle duplicate rejection for
  any attempted delayed-task creation.
- `src/kazusa_ai_chatbot/service.py`: synchronize accepted-task delivery
  terminal states after dispatcher success/failure.
- Tests covering the modified paths:
  `tests/test_background_work_jobs.py`,
  `tests/test_background_work_delivery.py`,
  `tests/test_action_spec_evaluator.py`,
  `tests/test_action_spec_results.py`,
  `tests/test_persona_supervisor2_action_selection.py`,
  `tests/test_action_selection_prompt_contract.py`,
  `tests/test_self_cognition_integration.py`, and
  `tests/test_self_cognition_tracking.py`.
- Docs:
  `README.md`, `docs/HOWTO.md`,
  `src/kazusa_ai_chatbot/background_work/README.md`,
  `src/kazusa_ai_chatbot/action_spec/README.md`,
  `src/kazusa_ai_chatbot/self_cognition/README.md`, and
  `src/kazusa_ai_chatbot/brain_service/README.md`.

### Keep

- Keep `background_work` as the internal executor queue/router/worker package.
- Keep worker-local `text_artifact` classification and generation inside the
  worker; L2d must not receive worker names, worker-local task types, tool
  args, or final artifact text.
- Keep adapter and dispatcher public delivery contracts unchanged.
- Keep calendar scheduler behavior unchanged.
- Keep legacy `background_artifact` processing for already persisted old rows.

## Data Migration

- No destructive migration is allowed in this plan.
- Create the new `accepted_tasks` collection and indexes idempotently at
  startup.
- Do not backfill existing `background_work_jobs` rows. Rows without
  `accepted_task_id` are legacy rows and continue through the existing
  `background_work_result_ready` compatibility path until drained.
- New rows after cutover must include `accepted_task_id` and
  `task_identity_key`.
- Add a read-only operator diagnostic query or test helper only if existing
  inspection surfaces cannot verify accepted-task states. Do not add a new
  public user API in this plan.

## Overdesign Guardrail

- Actual problem: accepted delayed work is currently exposed to cognition as
  background-worker/job mechanics and can be scheduled repeatedly across
  cognition cycles.
- Minimal change: add one accepted-task lifecycle boundary and one status-check
  action above the existing internal background-work executor.
- Ownership boundaries: LLM stages decide whether a user-facing task is
  accepted or whether a status answer is needed; deterministic lifecycle code
  owns identity, active duplicate rejection, persistence, source permissions,
  and state transitions; dispatcher/service code owns delivery status.
- Rejected complexity: no general task management product, scheduler
  replacement, arbitrary worker marketplace, multi-worker expansion, semantic
  duplicate classifier, fuzzy history search, new public API, repair prompt, or
  adapter protocol change.
- Evidence threshold: add rejected complexity only after an approved follow-up
  plan names a real worker expansion, cross-thread task search requirement,
  public API consumer, or observed failure not covered by exact lifecycle state.

## Agent Autonomy Boundaries

- This draft does not authorize production-code edits.
- Before implementation, answer any user questions about this design and obtain
  explicit approval to move the plan to `approved` or `in_progress`.
- Do not run live DB tests or inspect environment secrets unless the user
  explicitly requests it.
- If implementation discovers the action-router ownership is materially
  different from this plan, stop and update the plan before editing production
  code.
- If active duplicate identity would reject a legitimate repeated task in a
  common scenario, stop and tighten the identity material instead of weakening
  duplicate protection globally.

## Implementation Order

1. Reread this plan, the plan contract, and the required skills.
2. Parent adds failing lifecycle tests in
   `tests/test_accepted_task_lifecycle.py` for active unique identity,
   `source_message_id` exclusion, `enqueueing -> pending`, enqueue failure,
   stale `enqueueing` recovery, duplicate return, and status-check lookup.
3. Parent runs the lifecycle test file and records expected failures in
   `Execution Evidence`.
4. Parent starts the production-code subagent after the focused lifecycle test
   contract is recorded.
5. Production-code subagent adds `accepted_task` models, repository, indexes,
   README/ICD, and lifecycle functions.
6. Parent reruns lifecycle tests and records pass/fail evidence.
7. Parent adds or updates action-spec tests for accepted-task creation,
   `accepted_task_status_check`, source-trigger policy, prompt-visible result
   projection, and no job/queue/worker fields.
8. Production-code subagent integrates accepted-task lifecycle resolution into
   background-work action execution before internal job insertion.
9. Production-code subagent threads `accepted_task_id` through internal
   background-work job rows and worker completion/failure state updates.
10. Parent reruns action-spec and background-work focused tests and records
    evidence.
11. Parent adds graph and prompt-contract tests for post-schedule cognition,
    no-reschedule guard, accepted-task status percepts, and prompt leakage.
12. Production-code subagent updates persona graph/action-selection prompt
    projection and adds the bounded post-schedule cognition observation.
13. Parent reruns graph and prompt-contract tests and records evidence.
14. Parent adds result-ready delivery tests for accepted-task result source,
    audit-only worker metadata, delivery success/failure, and stale delivery
    recovery.
15. Production-code subagent migrates result-ready delivery for new accepted
    tasks and preserves legacy-row compatibility for rows without
    `accepted_task_id`.
16. Parent reruns delivery and self-cognition tests and records evidence.
17. Parent updates docs and READMEs after behavior tests pass.
18. Parent runs deterministic verification commands.
19. Parent runs live LLM checks one case at a time and records inspected
    evidence.
20. Parent starts the independent code-review subagent and records findings.
21. Parent remediates review findings that are inside this plan, reruns
    affected verification, and records evidence.
22. Move the plan to completed archive only after verification and user
    signoff.

## Execution Model

- Parent agent owns orchestration, test code, test validation, static checks,
  execution evidence, review feedback remediation, lifecycle updates, and final
  sign-off.
- Parent agent establishes the focused lifecycle test contract first and
  records the expected failures before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it; closes after planned
  production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 0 - plan approved for implementation.
  - Covers: status change only.
  - Verify: user approval is recorded before production-code edits.
  - Evidence: add approval note to `Execution Evidence`.
  - Handoff: next agent starts at Stage 1.
  - Sign-off: `Codex/2026-07-02` after evidence is recorded.
- [x] Stage 1 - lifecycle contract tests established.
  - Covers: implementation steps 1-3.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_accepted_task_lifecycle.py -q`.
  - Expected before implementation: failing tests for missing module or
    missing lifecycle behavior.
  - Evidence: record failures in `Execution Evidence`.
  - Handoff: production-code subagent starts at Stage 2.
  - Sign-off: `Codex/2026-07-02` after evidence is recorded.
- [x] Stage 2 - accepted-task module and repository complete.
  - Covers: implementation steps 4-6.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_accepted_task_lifecycle.py -q`.
  - Evidence: record changed files, index contract, and test result.
  - Handoff: parent starts Stage 3 tests.
  - Sign-off: `Codex/2026-07-02` after verification and evidence are recorded.
- [x] Stage 3 - action-spec and internal enqueue integration complete.
  - Covers: implementation steps 7-10.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_action_spec_evaluator.py tests\test_action_spec_results.py tests\test_background_work_jobs.py -q`.
  - Evidence: record action-result projection, duplicate rejection, and
    idempotency test output.
  - Handoff: parent starts Stage 4 tests.
  - Sign-off: `Codex/2026-07-02` after verification and evidence are recorded.
- [x] Stage 4 - post-schedule cognition and prompt projection complete.
  - Covers: implementation steps 11-13.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_action_selection.py tests\test_action_selection_prompt_contract.py tests\test_accepted_task_prompt_contract.py -q`.
  - Evidence: record no-reschedule guard and prompt leakage test output.
  - Handoff: parent starts Stage 5 tests.
  - Sign-off: `Codex/2026-07-02` after verification and evidence are recorded.
- [x] Stage 5 - result-ready delivery and self-cognition safeguards complete.
  - Covers: implementation steps 14-16.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_background_work_delivery.py tests\test_self_cognition_integration.py tests\test_self_cognition_tracking.py -q`.
  - Evidence: record delivery state, legacy-row compatibility, and
    self-cognition duplicate-rejection test output.
  - Handoff: parent starts Stage 6 docs.
  - Sign-off: `Codex/2026-07-02` after verification and evidence are recorded.
- [x] Stage 6 - docs updated.
  - Covers: implementation step 17.
  - Verify: documentation grep confirms new accepted-task wording in listed
    READMEs and no stale claim that new user work is exposed as
    `background_work_result_ready`.
  - Evidence: record changed docs and grep output.
  - Handoff: parent starts Stage 7 verification.
  - Sign-off: `Codex/2026-07-02` after verification and evidence are recorded.
- [x] Stage 7 - deterministic verification complete.
  - Covers: implementation step 18.
  - Verify: run every deterministic command in `Verification`.
  - Evidence: record command output summaries.
  - Handoff: parent starts Stage 8 live checks.
  - Sign-off: `Codex/2026-07-02` after verification and evidence are recorded.
- [x] Stage 8 - live LLM checks complete.
  - Covers: implementation step 19.
  - Verify: run each live LLM case one at a time and inspect output.
  - Evidence: record trace ids or artifact paths and manual assessment.
  - Handoff: parent starts Stage 9 review.
  - Sign-off: `Codex/2026-07-02` after verification and evidence are recorded.
- [x] Stage 9 - independent code review complete.
  - Covers: implementation steps 20-21.
  - Verify: independent code-review subagent reports no unresolved blockers;
    rerun affected tests after any fixes.
  - Evidence: record review findings, fixes, rerun commands, and residual
    risks.
  - Handoff: parent requests user signoff.
  - Sign-off: `Codex/2026-07-02` after review evidence is recorded.
- [x] Stage 10 - full future_speak E2E live proof complete.
  - Covers: user-message input, accepted scheduling, internal worker,
    calendar due execution, self-cognition handover, and adapter delivery.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_background_work_future_speak_live_llm.py::test_live_user_message_future_speak_e2e_feedback_handover -q -s -o addopts="" -m live_llm`.
  - Evidence: record trace path, raw user input, initial visible
    acknowledgement, duplicate/job counts, worker result, due calendar run
    completion, final dispatched text, and cleanup result.
  - Handoff: parent starts Stage 11 signoff.
  - Sign-off: `Codex/2026-07-02` after verification and evidence are recorded.
- [x] Stage 11 - user signoff and archive complete.
  - Covers: implementation step 22.
  - Verify: user signoff recorded and completed plan moved to archive.
  - Evidence: final status and archive path.
  - Handoff: none.
  - Sign-off: `Codex/2026-07-02` after archive is complete.

## Verification

Deterministic verification must include focused tests before broader suites:

```powershell
venv\Scripts\python.exe -m pytest tests\test_accepted_task_lifecycle.py
venv\Scripts\python.exe -m pytest tests\test_accepted_task_prompt_contract.py
venv\Scripts\python.exe -m pytest tests\test_background_work_jobs.py tests\test_background_work_delivery.py
venv\Scripts\python.exe -m pytest tests\test_action_spec_evaluator.py tests\test_action_spec_results.py
venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_action_selection.py tests\test_action_selection_prompt_contract.py
venv\Scripts\python.exe -m pytest tests\test_self_cognition_integration.py tests\test_self_cognition_tracking.py
```

Expected result: all deterministic commands exit 0 after implementation.
Before implementation, newly added contract tests are expected to fail for
missing accepted-task module or missing behavior; record those expected
failures in `Execution Evidence`.

Prompt leakage verification must inspect model-facing accepted-task prompt
payloads and fail if they contain implementation vocabulary for new accepted
tasks:

```powershell
rg -n "background_work|background work|job_ref|queue_state|operational_owner|worker_metadata|worker" tests src\kazusa_ai_chatbot
```

The `rg` output must be reviewed manually because internal audit files,
legacy compatibility tests, and the internal executor package may still contain
those terms. The pass condition is:

- allowed: internal executor code, DB audit fields, ops/diagnostic views,
  legacy-row compatibility tests, and historical plan text;
- forbidden: rendered accepted-task prompts, accepted-task status percepts,
  accepted-task result-ready model-visible metadata, prompt-visible action
  results, and L3-visible handoff fields.

Live LLM verification must run one case at a time and record inspected output:

- user asks for a bounded delayed text artifact: task is accepted, scheduled,
  and acknowledged without background/job/worker vocabulary;
- user repeats the same request before completion: no second job is queued and
  the response reports the active task state;
- user asks for progress: lifecycle state is reported without scheduling new
  work;
- completed internal job triggers active result delivery with artifact;
- self-cognition sees the open task: it may report/progress if appropriate,
  but it does not schedule duplicate work.

Minimum live LLM commands after implementation:

```powershell
$env:L2D_LIVE_CASE_FILE='tests\fixtures\l2d_background_artifact_cases.json'
$env:L2D_LIVE_CASE_ID='coding_snippet_accept_fibonacci'
venv\Scripts\python.exe -m pytest tests\test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -s -o addopts="" -m live_llm
venv\Scripts\python.exe -m pytest tests\test_background_work_router_live_llm.py::test_background_work_router_live_case -q -s -o addopts="" -m live_llm
venv\Scripts\python.exe -m pytest tests\test_background_work_text_artifact_live_llm.py::test_background_work_text_artifact_live_case -q -s -o addopts="" -m live_llm
venv\Scripts\python.exe -m pytest tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_15_background_work_ack -q -s -o addopts="" -m live_llm
venv\Scripts\python.exe -m pytest tests\test_background_work_future_speak_live_llm.py::test_live_user_message_future_speak_e2e_feedback_handover -q -s -o addopts="" -m live_llm
```

If the accepted-task implementation replaces these exact live cases, the
replacement cases must cover the same four model responsibilities: action
selection, internal routing, worker artifact generation, and visible
acknowledgement/result wording.

## LLM Call And Context Budget

- Before this plan, live delayed-work acceptance uses the existing cognition
  and action-selection path, then queues background work before L3
  acknowledgement. Completed jobs use existing result-ready cognition/dialog.
- After this plan, queue insertion, duplicate rejection, lifecycle status
  lookup, state transitions, and delivery-state updates add zero LLM calls.
- After this plan, `accepted_task_status_check` adds zero LLM calls beyond the
  normal user-message cognition/action-selection path that selected the
  capability.
- The post-schedule cognition observation may add one bounded response-path
  cognition pass only when delayed work is newly scheduled, duplicate-active
  work is detected, or progress state is returned.
- The post-schedule pass must disable delayed-work creation and status-check
  affordances, must not run background-worker routing, must not generate the
  artifact, and must not perform web/tool loops.
- The post-schedule pass must receive only the original prompt-safe user
  context plus one semantic `accepted_task_status` percept. It must not receive
  worker/job ids, queue fields, retries, leases, or raw DB documents.
- The post-schedule pass must stay within the existing 50k-token default cap.
  If exact tokenization is unavailable, estimate by character count and record
  the estimate in `Execution Evidence`.
- Result-ready delivery may use the existing cognition/dialog path, but workers
  must not trigger extra repair prompts or direct adapter sends.
- No LLM is added for semantic duplicate detection.

## Independent Plan Review

Required before changing `Status` to `approved`:

- Verify that the accepted-task layer is necessary and not a cosmetic rename of
  `background_work_jobs`.
- Verify that duplicate rejection is deterministic and atomic.
- Verify that no deterministic raw-user-text keyword classifier is introduced.
- Verify that post-schedule cognition cannot recursively enqueue the same task.
- Verify that result-ready delivery remains source-bound and dispatcher
  validated.
- Verify that the change surface is scoped to delayed accepted work.

Review result: complete for this draft revision. The plan remains `draft` and
is not approved for implementation until the user explicitly approves it.

Findings addressed in this revision:

- Blocker resolved: duplicate identity previously included `source_message_id`,
  which would fail to reject repeated turns. The identity now excludes
  `source_message_id` and stores message ids as provenance only.
- Blocker resolved: accepted-task creation and internal job insertion lacked a
  crash-safe handoff. The plan now requires `enqueueing -> pending`,
  `enqueue_failed`, and stale `enqueueing` recovery.
- Blocker resolved: progress checking lacked an explicit contract. The plan
  now adds the private `accepted_task_status_check` action capability.
- Blocker resolved: source rules were not implementable because trigger source
  was not threaded. The plan now requires `source_trigger_source` in trusted
  action target scope.
- Blocker resolved: broad owner wording left implementation choices open. The
  plan now names exact owner files and accepted-task status percept behavior.
- Non-blocking finding resolved: execution model and progress checklist were
  too loose for a high-risk migration. They now follow parent-led test-first
  execution with one production-code subagent and one independent review
  subagent.
- Non-blocking finding resolved: prompt leakage verification did not define
  allowed and forbidden matches. It now does.

## Independent Code Review

Required before final completion:

- Run this gate through exactly one independent code-review subagent after all
  planned verification passes and before final sign-off.
- Review prompt-facing surfaces for leaked background/job/worker vocabulary.
- Review accepted-task active-state uniqueness and race behavior.
- Review self-cognition and progress-check duplicate failure modes.
- Review lifecycle state transitions for completion, failure, delivery failure,
  retry, and stuck in-progress recovery.
- Review tests for both the positive lifecycle and duplicate rejection.

Review result: complete by manual no-subagent review.

Stage 9 execution note: native independent subagent review was not used because
the user explicitly requested execution without subagents. Codex performed the
gate manually against the plan, diff, live evidence, and deterministic tests.
No unresolved blockers remain after the fixes recorded in `Execution Evidence`.

## Execution Evidence

Execution started:

- 2026-07-02: User explicitly approved executing the plan without subagents
  with: "Ok, now execute the plan without subagent." This records both
  production-code authorization and the development-plan fallback execution
  path required when native subagent execution is not used.
- Stage 1 lifecycle contract tests:
  `venv\Scripts\python.exe -m pytest tests\test_accepted_task_lifecycle.py -q`
  failed as expected before implementation. All seven tests failed with
  `ModuleNotFoundError: No module named 'kazusa_ai_chatbot.accepted_task'`,
  proving the new lifecycle module contract was not yet implemented.
- Stage 2 accepted-task module and repository:
  added `src/kazusa_ai_chatbot/accepted_task/`,
  `src/kazusa_ai_chatbot/db/accepted_tasks.py`, and startup index wiring in
  `src/kazusa_ai_chatbot/db/bootstrap.py`. The repository owns
  `accepted_task_id_unique`, `accepted_task_active_identity_unique`,
  `accepted_task_state_updated`, and `accepted_task_scope_active_lookup`.
  Also made `background_work.__init__` lazy like `background_artifact` to keep
  DB imports acyclic.
- Stage 2 verification:
  `venv\Scripts\python.exe -m pytest tests\test_accepted_task_lifecycle.py -q`
  passed, 7 tests.
- Stage 3 action-spec and internal enqueue integration:
  `background_work_request` and `future_speak` now create or reuse an
  accepted-task lifecycle row before inserting the internal worker job. Active
  duplicates return `already_active` without enqueueing a second job. New
  internal jobs use `background_work:{accepted_task_id}` idempotency and carry
  `accepted_task_id` plus `task_identity_key` for audit/state sync. Prompt
  action results expose accepted-task fields only and omit queue/job/worker
  internals. The worker tick moves accepted tasks to running, result-ready, or
  failure-ready when real background execution happens.
- Stage 3 verification:
  `venv\Scripts\python.exe -m pytest tests\test_action_spec_evaluator.py tests\test_action_spec_results.py tests\test_background_work_jobs.py -q`
  passed, 42 tests.
- Stage 4 prompt projection and no-reschedule guard:
  L2d now sees model-facing `accepted_task_request` and
  `accepted_task_status_check` affordances. The internal executable action
  remains `background_work_request` after deterministic materialization.
  `source_trigger_source` is threaded into accepted-task action scope, and
  non-user sources such as result-ready/post-schedule/autonomous cases are
  rejected before accepted-task creation. L3 acknowledgement projection now
  uses `accepted_task_state`, `accepted_task_summary`, and `wait_guidance`
  without queue/job/worker fields.
- Stage 4 verification:
  `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_action_selection.py tests\test_action_selection_prompt_contract.py tests\test_accepted_task_prompt_contract.py -q`
  passed, 32 tests.
- Stage 5 result-ready delivery and self-cognition safeguards:
  new accepted-task-backed background-work rows now emit
  `accepted_task_result_ready` / `accepted_task_result` cognition episodes with
  accepted-task metadata only. Legacy rows without `accepted_task_id` still use
  the old `background_work_result_ready` path. Delivery ticks synchronize
  accepted tasks through delivery in progress, delivered, and delivery failed
  states. Self-cognition/result-ready/non-user sources are covered by the
  source-trigger no-new-task guard from Stage 4.
- Stage 5 verification:
  `venv\Scripts\python.exe -m pytest tests\test_background_work_delivery.py tests\test_self_cognition_integration.py tests\test_self_cognition_tracking.py -q`
  passed, 94 tests.
- Stage 6 docs update:
  updated `README.md`, `docs/HOWTO.md`,
  `src/kazusa_ai_chatbot/accepted_task/README.md`,
  `src/kazusa_ai_chatbot/action_spec/README.md`,
  `src/kazusa_ai_chatbot/background_work/README.md`,
  `src/kazusa_ai_chatbot/background_artifact/README.md`,
  `src/kazusa_ai_chatbot/brain_service/README.md`,
  `src/kazusa_ai_chatbot/nodes/README.md`, and
  `src/kazusa_ai_chatbot/self_cognition/README.md` to document
  accepted-task as the model-facing delayed-work lifecycle and
  `background_work` as the internal executor.
- Stage 6 verification:
  `rg -n -g 'README.md' "background_work_request|background_work_result_ready|job_ref|queue_state|operational_owner|worker_metadata" README.md docs src/kazusa_ai_chatbot`
  returns only internal-executor and legacy-compatibility references. No stale
  README claim remains that new user work is exposed to cognition as
  `background_work_result_ready`.
- Stage 7 deterministic fixes:
  broader verification found stale model-facing `background_work_request`
  expectations in action-selection/L3 handoff tests and a real contract gap in
  `cognition_chain_core.contracts._ACTION_CAPABILITIES`, which still allowed
  the old model-facing delayed-work capability set. Updated the validator and
  tests to use `accepted_task_request` at the model-facing boundary while
  keeping materialized internal `background_work_request` specs valid. Also
  added the missing stuck-delivery recovery required by this plan:
  `recover_stale_delivery_in_progress_tasks`,
  `recover_stale_background_work_delivery_in_progress`, and delivery-tick
  recovery before deliverable job scans.
- Stage 7 verification:
  `venv\Scripts\python.exe -m pytest tests\test_accepted_task_lifecycle.py tests\test_accepted_task_prompt_contract.py tests\test_background_work_jobs.py tests\test_background_work_delivery.py tests\test_action_spec_evaluator.py tests\test_action_spec_results.py tests\test_persona_supervisor2_action_selection.py tests\test_action_selection_prompt_contract.py tests\test_self_cognition_integration.py tests\test_self_cognition_tracking.py tests\test_action_selection_payload.py tests\test_cognition_chain_core_action_selection.py tests\test_cognition_chain_core_contracts.py tests\test_cognitive_episode_contract.py tests\test_background_work_future_speak.py tests\test_l2d_action_selection_cases.py tests\test_l2d_l3_surface_handoff.py`
  passed, 240 tests.
- Stage 7 prompt leakage review:
  ran the required broad grep over `tests` and `src\kazusa_ai_chatbot` and
  reviewed the hits manually. Remaining hits are internal executor code,
  legacy artifact/background-work compatibility, DB audit/result schemas,
  operator/runtime code, or tests that assert forbidden vocabulary is absent.
  Narrow review of action-selection, prompt-selection, episode projection, L3
  surface projection, and action-result projection confirmed new accepted-task
  prompts use `accepted_task_request`, accepted-task states, summaries, and
  wait guidance, and hide queue/job/worker fields.
- Stage 8 live LLM fixture update:
  updated `tests\fixtures\l2d_background_artifact_cases.json` so the frozen
  `coding_snippet_accept_fibonacci` state includes the normal raw action
  affordances. A local payload check confirmed the prompt-facing affordance is
  `accepted_task_request`, while runtime materialization still produces the
  internal `background_work_request`.
- Stage 8 live LLM action selection:
  `venv\Scripts\python.exe -m pytest tests\test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -s -o addopts="" -m live_llm`
  passed with `L2D_LIVE_CASE_FILE=tests\fixtures\l2d_background_artifact_cases.json`
  and `L2D_LIVE_CASE_ID=coding_snippet_accept_fibonacci`. Trace:
  `test_artifacts\llm_traces\l2d_action_selection_live_llm__coding_snippet_accept_fibonacci__20260701T145238832121Z.json`.
  The model selected one delayed accepted-task route and one visible speak
  route; materialized action specs were `background_work_request` plus
  `speak`, with no forbidden action-spec leakage.
- Stage 8 live LLM internal worker routing:
  `venv\Scripts\python.exe -m pytest tests\test_background_work_router_live_llm.py::test_background_work_router_live_case -q -s -o addopts="" -m live_llm`
  passed. Trace:
  `test_artifacts\llm_traces\background_work_router_live_llm__fibonacci_text_artifact_route__20260701T145249378369Z.json`.
  The internal router selected `text_artifact` for bounded code-snippet work.
- Stage 8 live LLM worker artifact generation:
  `venv\Scripts\python.exe -m pytest tests\test_background_work_text_artifact_live_llm.py::test_background_work_text_artifact_live_case -q -s -o addopts="" -m live_llm`
  passed. Trace:
  `test_artifacts\llm_traces\background_work_text_artifact_live_llm__fibonacci_code_snippet__20260701T145300264453Z.json`.
  The worker produced a real Fibonacci code artifact and summary.
- Stage 8 live LLM L3 acknowledgement wording:
  the first green run exposed stale test-fixture wording that encouraged the
  model to say `排队`; updated
  `tests\test_dialog_l3_surface_contract_live_llm.py` so the live contract
  requires accepted-task wording and forbids queue/backend vocabulary. Rerun:
  `venv\Scripts\python.exe -m pytest tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_15_background_work_ack -q -s -o addopts="" -m live_llm`
  passed. Trace:
  `test_artifacts\llm_traces\dialog_l3_surface_contract_live_llm__manual__background_work_ack__20260701T145419636490Z.json`.
  Output used accepted-task wording: `摘要的任务我接下了……等结果回来之后，我们再接着看重点吧。`
- Stage 8 real `future_speak` background-worker proof:
  live DB index creation initially failed because four stale debug test rows
  already duplicated the same active accepted-task identity for
  `debug:user:test-user` / `global-user-001` / `future_speak`. Inspected the
  rows, confirmed they were test artifacts with no remaining background job,
  and deleted only that exact test scope (`deleted_count=4`). Rerun of
  `venv\Scripts\python.exe -m pytest tests\test_background_work_future_speak_live_llm.py::test_live_l2d_future_speak_runs_real_background_worker -q -s -o addopts="" -m live_llm`
  passed after updating the test to assert accepted-task fields rather than
  legacy `job_ref`/`queue_state`.
- Stage 8 live progress/repeat proof:
  extended the same `future_speak` live test to replay the LLM-picked
  `future_speak` action before completion and to execute
  `accepted_task_status_check` against the live accepted-task row. This
  surfaced a real bug where `execute_action_specs_for_trace` passed unused
  execution kwargs into the status-check handler. Fixed the call site in
  `src\kazusa_ai_chatbot\action_spec\execution.py` and added deterministic
  regression coverage in `tests\test_action_spec_results.py`.
  `venv\Scripts\python.exe -m pytest tests\test_action_spec_results.py -q`
  passed, 10 tests.
- Stage 8 final `future_speak` live proof:
  `venv\Scripts\python.exe -m pytest tests\test_background_work_future_speak_live_llm.py::test_live_l2d_future_speak_runs_real_background_worker -q -s -o addopts="" -m live_llm`
  passed. Trace:
  `test_artifacts\llm_traces\background_work_future_speak_live_llm__l2d_to_background_worker_to_calendar__20260701T150536389448Z.json`.
  Trace evidence: observed action kinds were `speak` and `future_speak`;
  initial execution returned `accepted_task_state=scheduled`; replay returned
  `accepted_task_state=already_active`; status check returned
  `accepted_task_state=scheduled`; job count stayed `1` after replay and
  status check; worker tick processed and succeeded one real job; the worker
  created a real pending calendar run with `trigger_kind=future_cognition`;
  the accepted-task row moved to terminal `delivered` and released
  `active_identity_key` after the calendar slot was durably created.
  Cleanup verification confirmed zero live-test accepted-task, background-job,
  calendar-schedule, or calendar-run rows remained for the
  `debug:user:future-speak-live-*` scope.
- Stage 9 manual no-subagent review:
  reviewed accepted-task lifecycle/repository code, background-work enqueue,
  worker, delivery, result-source projection, L2d action-selection projection,
  L3 handoff projection, service result delivery, and the new `future_speak`
  worker path. Review used the independent-code-review checklist manually
  because the user requested no subagents.
- Stage 9 review finding fixed: progress/status lookup fail-closed behavior.
  `find_active_accepted_task_for_scope()` previously built a partial query if
  a future caller bypassed the action-spec validator with missing requester or
  channel fields. Fixed `src\kazusa_ai_chatbot\db\accepted_tasks.py` so
  incomplete trusted scope returns no match, and added
  `test_status_check_with_incomplete_scope_does_not_match_global_task`.
  `venv\Scripts\python.exe -m pytest tests\test_accepted_task_lifecycle.py -q`
  passed, 9 tests.
- Stage 9 review finding fixed: skip-result worker lifecycle completion.
  `future_speak` sets `skip_result_delivery=True`, which makes the internal
  background job terminal `delivered` after the calendar slot is created.
  The accepted-task row previously moved to `result_ready`, which could leave
  active duplicate suppression stuck without a delivery tick. Fixed
  `src\kazusa_ai_chatbot\background_work\worker.py` so skip-result workers
  mark accepted tasks `delivered` and release `active_identity_key`. Added
  deterministic assertions in `tests\test_background_work_future_speak.py`
  and live assertions in
  `tests\test_background_work_future_speak_live_llm.py`.
- Stage 9 review finding fixed: deterministic `future_speak` test touched the
  live MongoDB accepted-task collection. Isolated
  `test_future_speak_execution_enqueues_requested_worker` by mocking the
  accepted-task lifecycle boundary, and deleted one exact stale debug test row
  from live DB scope `debug:user:test-user` /
  `global-user-001` / `future_speak`.
- Stage 9 review cleanup:
  fixed a payload-builder indentation defect in
  `src\kazusa_ai_chatbot\cognition_chain_core\action_selection.py`.
- Stage 9 prompt-leakage review:
  reran the targeted accepted-task prompt-surface grep for
  `background_work`, `background work`, `job_ref`, `queue_state`,
  `operational_owner`, `worker_metadata`, `worker`, `排队`, `队列`, and
  `后台` across action-selection, prompt-selection, episode projection,
  L3 surface projection, action results, live dialog fixture, and the L2d
  fixture. Remaining hits are internal executor/legacy compatibility,
  sanitizer constants, fixture expected internal materialization, or explicit
  forbidden-word assertions. No accepted-task prompt payload, status percept,
  result-ready metadata, prompt-visible action result, or L3-visible handoff
  leaks queue/job/worker internals.
- Stage 9 verification:
  `venv\Scripts\python.exe -m pytest tests\test_background_work_future_speak.py -q`
  passed, 5 tests. Rerun live proof:
  `venv\Scripts\python.exe -m pytest tests\test_background_work_future_speak_live_llm.py::test_live_l2d_future_speak_runs_real_background_worker -q -s -o addopts="" -m live_llm`
  passed. Final deterministic sweep:
  `venv\Scripts\python.exe -m pytest tests\test_accepted_task_lifecycle.py tests\test_accepted_task_prompt_contract.py tests\test_background_work_jobs.py tests\test_background_work_delivery.py tests\test_action_spec_evaluator.py tests\test_action_spec_results.py tests\test_persona_supervisor2_action_selection.py tests\test_action_selection_prompt_contract.py tests\test_self_cognition_integration.py tests\test_self_cognition_tracking.py tests\test_action_selection_payload.py tests\test_cognition_chain_core_action_selection.py tests\test_cognition_chain_core_contracts.py tests\test_cognitive_episode_contract.py tests\test_background_work_future_speak.py tests\test_l2d_action_selection_cases.py tests\test_l2d_l3_surface_handoff.py`
  passed, 242 tests. Live cleanup verification confirmed zero
  `debug:user:future-speak-live-*` accepted-task, background-job,
  calendar-schedule, or calendar-run rows.
- Stage 9 residual risk:
  `future_speak` currently treats the accepted background task as complete
  once the future-cognition calendar slot is durably created. If the product
  requirement is to suppress duplicate future reminders until the later
  message is actually sent, a follow-up plan should link calendar-run
  completion back to accepted-task lifecycle state.
- Stage 10 full `future_speak` E2E proof:
  the first live E2E proof run passed transport but exposed a real
  relative-date quality risk: for user text containing `tomorrow`, the model
  scheduled `2025-07-03 09:00` instead of a future 2026 date. Updated
  `tests\test_background_work_future_speak_live_llm.py` so the full E2E case
  uses explicit user text
  `Can you remind me on July 3, 2030 at 09:00 to drink water?` and asserts the
  scheduled due year is 2030, preventing a false-positive pass on an
  accidentally past schedule.
- Stage 10 syntax verification:
  `venv\Scripts\python.exe -m py_compile tests\test_background_work_future_speak_live_llm.py`
  passed.
- Stage 10 corrected live E2E verification:
  `venv\Scripts\python.exe -m pytest tests\test_background_work_future_speak_live_llm.py::test_live_user_message_future_speak_e2e_feedback_handover -q -s -o addopts="" -m live_llm`
  passed, 1 test in 87.78 seconds.
- Stage 10 trace:
  `test_artifacts\llm_traces\background_work_future_speak_e2e_live_llm__user_message_to_due_feedback_handover__20260702T000809480043Z.json`.
- Stage 10 human review artifact:
  `test_artifacts\llm_reviews\background_work_future_speak_e2e_live_llm__20260702T000809480043Z.md`.
- Stage 10 observed live outputs:
  initial visible acknowledgement was `Four years... Seriously? Setting a
  water reminder four years in advance is just absurd. But fine, I've got it.
  July 3rd, 2030, at 9 AM—I'll make sure you drink your water.` The accepted
  task row was `state=pending`, `executor_kind=background_work`,
  `action_kind=future_speak`, with summary `提醒用户喝水`. Exactly one
  background job existed after chat; the worker tick processed 1, succeeded 1,
  and failed 0. The job completed as `worker=future_speak`, with
  `delivery_state=delivered` and `skip_result_delivery=true`. Calendar handoff
  created `calendar_run_7c3a25956869eb705a5d31cf561006fe` with
  `trigger_kind=future_cognition`, due `2030-07-02T21:00:00+00:00`, then due
  self-cognition processed 1 case and the calendar run became `completed`.
  The final action attempt was `status=sent`, `dispatch_status=sent`, and the
  fake debug adapter captured one private send:
  `九点啦，喝口水吧……我想看你照顾好自己的样子。`
- Stage 10 cleanup verification:
  exact-scope post-test counts were all zero:
  `accepted_task_rows=0`, `background_job_rows=0`,
  `calendar_schedule_rows=0`, `calendar_run_rows=0`,
  `conversation_rows=0`, `self_cognition_action_attempt_rows=0`, and
  `user_profile_rows=0`.
- Interface documentation hardening after Stage 10:
  updated `src\kazusa_ai_chatbot\background_work\README.md`,
  `src\kazusa_ai_chatbot\action_spec\README.md`, and `docs\HOWTO.md` to make
  the background-work extension interface explicit. The documented boundary is:
  future workers enter through accepted-task/action-spec semantics, use an
  internal `background_work_request`, expose only prompt-safe semantic worker
  descriptions, own their worker-local parameter extraction and validation,
  return bounded `BackgroundWorkResult` rows, and hand visible output back via
  `accepted_task_result_ready` or a durable scheduled follow-up. The docs also
  state that coding-agent or complex-resolver workers are possible future
  integrations only after a separate reviewed capability, permission,
  side-effect, duplicate-identity, output, failure, and verification contract.
  This documentation change does not implement those workers or authorize
  shell execution, repository edits, web access, package installs, direct
  adapter sends, or direct cognition calls.
- Stage 11 closure:
  user approved closing the plan and committing the changes on 2026-07-02.
  Plan status was set to `completed`, Stage 11 was signed off, and the plan
  was moved to
  `development_plans\archive\completed\short_term\background_work_semantic_lifecycle_plan.md`.

Plan review update:

- Performed a plan-contract, local-LLM architecture, and no-pre/post-user-input
  review.
- Addressed duplicate identity, enqueue atomicity, progress-check contract,
  source-trigger threading, open-ended wording, execution-model granularity,
  and prompt-leakage verification issues in the draft plan.
- No production code was changed.

Baseline proof for current background-work implementation:

- 2026-07-02:
  `venv\Scripts\python.exe -m pytest tests\test_background_work_jobs.py tests\test_background_work_router.py tests\test_background_work_providers.py tests\test_background_work_text_artifact.py tests\test_background_work_delivery.py -q`
  passed, 24 tests.
- 2026-07-02:
  `venv\Scripts\python.exe -m pytest tests\test_action_spec_evaluator.py tests\test_action_selection_prompt_contract.py tests\test_action_selection_payload.py tests\test_action_spec_results.py tests\test_persona_supervisor2_action_selection.py -k "background_work or background_artifact" -q`
  passed, 11 selected tests and 46 deselected.
- 2026-07-02:
  `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2.py::test_background_artifact_executes_before_l3_acknowledgement tests\test_l2d_l3_surface_handoff.py::test_background_work_acknowledgement_requires_pending_queue_result tests\test_l2d_l3_surface_handoff.py::test_background_work_failed_enqueue_blocks_later_delivery_promise tests\test_l2d_l3_surface_handoff.py::test_background_work_no_handoff_result_uses_task_brief tests\test_l2d_action_selection_cases.py::test_compare_accepts_background_work_route_only_request tests\test_cognitive_episode_contract.py::test_background_work_result_ready_builder_creates_valid_episode -q`
  passed, 6 tests.
- This baseline proves the current implemented background-work mechanics:
  public entrypoints, route-only worker separation, provider dispatch,
  text-artifact worker stage separation, bounded failure handling,
  source-bound result-ready delivery, current action-spec validation, and
  current L2d/L3 acknowledgement handoff.
- This baseline does not prove the proposed accepted-task semantic lifecycle,
  duplicate rejection across cognition/self-cognition/progress turns, or
  background vocabulary hiding for the new design. Those require the
  contract-first tests and implementation stages in this plan.

Real LLM baseline for current background-work implementation:

- 2026-07-02:
  `venv\Scripts\python.exe -m pytest tests\test_background_work_router_live_llm.py::test_background_work_router_live_case -q -s -o addopts="" -m live_llm`
  passed. Trace:
  `test_artifacts\llm_traces\background_work_router_live_llm__fibonacci_text_artifact_route.json`.
- 2026-07-02:
  `venv\Scripts\python.exe -m pytest tests\test_background_work_text_artifact_live_llm.py::test_background_work_text_artifact_live_case -q -s -o addopts="" -m live_llm`
  passed. Trace:
  `test_artifacts\llm_traces\background_work_text_artifact_live_llm__fibonacci_code_snippet.json`.
- 2026-07-02:
  `venv\Scripts\python.exe -m pytest tests\test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -s -o addopts="" -m live_llm`
  with `L2D_LIVE_CASE_FILE=tests\fixtures\l2d_background_artifact_cases.json`
  and `L2D_LIVE_CASE_ID=coding_snippet_accept_fibonacci` failed. Trace:
  `test_artifacts\llm_traces\l2d_action_selection_live_llm__coding_snippet_accept_fibonacci.json`.
  The model emitted no actions because the live payload had
  `capabilities.action_affordances = []`. This is a real current-path blocker:
  current L2d background-work selection is not proven by live LLM evidence.
- 2026-07-02:
  `venv\Scripts\python.exe -m pytest tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_15_background_work_ack -q -s -o addopts="" -m live_llm`
  passed. Trace:
  `test_artifacts\llm_traces\dialog_l3_surface_contract_live_llm__manual__background_work_ack.json`.
- Human-readable review artifact:
  `test_artifacts\llm_traces\background_work_live_llm_review_20260702.md`.

Initial mapping inspected:

- `README.md`
- `docs/HOWTO.md`
- `development_plans/README.md`
- `src/kazusa_ai_chatbot/background_work/README.md`
- `src/kazusa_ai_chatbot/action_spec/README.md`
- `src/kazusa_ai_chatbot/self_cognition/README.md`
- `src/kazusa_ai_chatbot/brain_service/README.md`
- `src/kazusa_ai_chatbot/background_work/*.py`
- `src/kazusa_ai_chatbot/db/background_work_jobs.py`
- `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py`
- `src/kazusa_ai_chatbot/action_spec/execution.py`
- `src/kazusa_ai_chatbot/action_spec/registry.py`
- `src/kazusa_ai_chatbot/cognition_episode.py`
- `src/kazusa_ai_chatbot/cognition_chain_core/episode_projection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`
- focused background-work, action-spec, action-selection, self-cognition, and
  persona supervisor tests

## Risks

- Duplicate identity may be too broad and reject legitimate new work. Mitigate
  with source scope, accepted task seed/detail, and active-state-only
  uniqueness.
- Duplicate identity may be too narrow and allow duplicates. Mitigate with
  tests for repeated user turns, self-cognition, and progress checks.
- A process can crash after creating an accepted-task lock and before inserting
  the internal worker job. Mitigate with `enqueueing`, `enqueue_failed`, and
  stale `enqueueing` recovery tests.
- The post-schedule cognition pass may add live-turn latency. Mitigate with a
  bounded no-resolver/no-worker pass and live timing evidence.
- Hiding background vocabulary may make operator debugging harder. Mitigate by
  keeping audit-only job and worker metadata in DB/trace surfaces.
- Existing queued rows may still use legacy `background_work_result_ready`.
  Mitigate with explicit legacy-row compatibility and a drain/removal decision
  in a later plan.
- Result delivery can still fail if adapters are unavailable. Mitigate with
  terminal `delivery_failed` state, retry caps, and operator-visible audit.
- Source policy may block future legitimate autonomous task creation. Mitigate
  by scoping this plan to accepted user work and requiring a later explicit
  permission plan for autonomous task creation.

## Acceptance Criteria

- New delayed work creates an accepted-task record and then one internal
  background-work job through the required
  `enqueueing -> job inserted -> pending` order.
- No user-facing acknowledgement is allowed until the accepted-task row is
  `pending` and the internal worker job exists durably.
- If internal job insertion fails, the accepted task becomes `enqueue_failed`,
  no promise is acknowledged, and retry is not blocked forever.
- Repeating the same active accepted task from another cognition cycle,
  self-cognition source case, or progress-check turn does not create another
  worker job.
- The user receives an acknowledgement only after durable accepted-task
  creation or duplicate/progress resolution.
- The acknowledgement is produced after cognition reads a semantic accepted
  task state, not after reading job/queue internals.
- Completed work triggers active source-bound cognition and dispatcher delivery
  with the artifact or failure summary.
- After dispatcher success, the accepted-task lifecycle is terminal
  `delivered`; after delivery exhaustion, it is terminal or operator-visible
  `delivery_failed`.
- Model-facing prompt payloads and percept metadata for new accepted tasks do
  not include background/job/queue/worker implementation vocabulary.
- Existing route-only worker separation remains intact: L2d does not choose
  worker names, worker-local task types, tool args, or final artifact text.
- Focused deterministic tests and inspected live LLM checks pass.
