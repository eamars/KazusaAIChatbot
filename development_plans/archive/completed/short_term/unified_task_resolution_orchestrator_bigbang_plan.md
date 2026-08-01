# unified task resolution orchestrator bigbang plan

## Summary

- Goal: Replace overlapping cognition-visible local-context, public-research,
  and generic accepted-task routing with one `task_resolution_request` and one
  resumable task orchestrator that can invoke the existing local-context,
  public complex-resolver, coding-agent, and text/computation specialists.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`, `py-style`, `cjk-safety`,
  `test-style-and-execution`
- Coding boundary: freeze and consume only the current public exports from
  `kazusa_ai_chatbot.coding_agent`: `CodingRunStartRequest`, `CodingRunContinueRequest`,
  `CodingRunGetRequest`, `CodingRunResponse`, `start_coding_run(...)`,
  `continue_coding_run(...)`, and `get_coding_run(...)`. Coding-agent plan status and internals are irrelevant to this execution.
- Coding verification boundary: do not execute the coding-agent path in any
  deterministic, live-LLM, persona, database, or cutover test for this plan.
  Verify only frozen public imports/signatures, validated durable handover
  payloads, approval-field preservation, and an orchestrator stub proving that
  coding selection never invokes a coding handler during the test. The known
  coding branch remains outside this plan's quality conclusion.
- Overall cutover strategy: bigbang. Replace the old model-facing capability
  roster, generic worker router, direct generic coding/text worker routes, and
  v1 background-task records in one offline cutover. Preserve no compatibility
  aliases, fallback routes, dual reads, or v1 task rows.
- Highest-risk areas: changing the live cognition resolver graph, promoting an
  inline resolver session into durable work without duplicate execution,
  preserving typed evidence and partial results across a lease restart,
  calling the coding agent without bypassing its approval lifecycle, and
  deleting all existing background-work and accepted-task history.
- Acceptance criteria: cognition selects one task-resolution capability;
  the orchestrator can recover from a wrong specialist choice; inline work is
  bounded by a configurable 30-second default; deferred work resumes the same
  checkpoint and counters; evidence-bearing partial results are deliverable;
  coding mutations retain approval; the two approved history collections are
  empty at cutover; and the two 2026-07-30 production failure cases route to
  public research and return resolved or partial evidence instead of reaching
  coding or text-artifact rejection.

## Context

Two production tasks created on 2026-07-30 exposed the same upstream semantic
failure through different downstream workers:

- `job-7a5222270623434286e4d8a1d5d6d896` routed a public documentation URL to
  `coding_agent`, which rejected the non-code task.
- `job-a00a90a2afb04b72991f0ed7bbe99624` routed the same public URL to
  `text_artifact`, which rejected the required external web access.

Both cognition traces parsed and validated. The action planner selected
`accepted_task_request`, emitted no resolver request, and returned
`goal_resolution="answerable_now"` despite lacking the article contents. The
first request's route rationale then entered `source_context` and biased the
background router toward coding. The second request avoided coding but exposed
that no generic background worker could acquire public evidence.

The current model-facing roster asks one local model to choose among:

- `local_context_recall` for private/local evidence;
- `public_answer_research` for public/current evidence;
- `accepted_task_request` for generic delayed work, including unbound coding;
- retained stateful actions such as future contact, memory lifecycle, and
  bound coding-run continuation.

Those capabilities overlap along two axes: evidence domain and execution
horizon. The exclusive action-versus-resolver output contract turns one wrong
choice into a terminal path. The later generic background router then makes a
second exclusive worker choice from an incomplete roster. Worker rejection
currently fails the job rather than returning a semantic observation to a
coordinator.

The target architecture makes task resolution one semantic path:

```text
cognition
  -> evidence sufficient: continue to normal surface/dialog
  -> task_resolution_request
       -> bounded inline task-resolution session
       -> task orchestrator selects one existing specialist at a time
       -> specialist result is validated and merged
       -> resolved/partial inside inline budget: return observation to cognition
       -> inline budget exhausted: persist checkpoint, accepted task, and job
       -> task_orchestrator background worker resumes the same checkpoint
       -> resolved/partial result re-enters cognition for visible delivery
```

The action planner continues to decide semantic acceptance and evidence
sufficiency. Deterministic code owns the 30-second foreground budget,
checkpoint persistence, dispatch counters, lease recovery, idempotency,
permissions, and delivery. The orchestrator LLM chooses only the next semantic
specialist and subgoal. Each specialist owns its low-level parameters and
refusal contract.

The user explicitly waived every coding-agent lifecycle dependency on
2026-08-01. This plan treats the current public coding-run exports as a frozen
  entry/exit boundary. It must not inspect or change coding-agent plan status, internal controllers, indexes, prompts, candidate state, or cutover work.

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, updating,
  or signing off this plan.
- `local-llm-architecture`: load before changing the cognition capability
  roster, task-orchestrator prompt, specialist descriptions, graph edges,
  budgets, checkpoint projection, or LLM call count.
- `no-prepost-user-input`: load before changing how an accepted semantic task
  becomes inline work or durable accepted-task state.
- `debug-llm`: load before changing LLM prompts, running local/live routing
  cases, comparing old/new behavior, or authoring quality evidence.
- `py-style`: load before editing any Python production or test file.
- `cjk-safety`: load before editing Python files or tests containing CJK
  strings.
- `test-style-and-execution`: load before adding, changing, deleting, or
  running tests. Live LLM tests must run and be inspected one case at a time.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the Independent Code Review gate and record the result
  in Execution Evidence.
- Use parent-led native subagent execution. Fallback execution requires
  explicit user approval.
- Production-code implementation requires a separate explicit user command.
  Approval of this draft or its architecture does not authorize source edits,
  database deletion, deployment, or cutover.
- Before implementation, contract-test the frozen public coding exports named
  in Summary. An absent or incompatible export stops this plan and is reported
  without editing coding-agent internals.
- Keep the action planner's semantic question small: determine whether current
  evidence is sufficient or one task-resolution request is needed. It must not
  choose a specialist, worker, timeout, retry, persistence mode, lease,
  delivery target, tool argument, filesystem path, or database operation.
- Preserve LLM ownership of user-task acceptance. Deterministic code must
  faithfully materialize a validated `task_resolution_request`; it must not
  keyword-match, reclassify, suppress, or rewrite user intent.
- Deterministic code owns inline timing, counters, attempt ledgers,
  checkpointing, persistence, duplicate rejection, permissions, side-effect
  limits, leases, queue recovery, and adapter delivery audit.
- The task orchestrator may choose one next specialist and one semantic
  subgoal per dispatch. It must not generate low-level repository, RAG, web,
  database, or tool parameters.
- Each specialist must return a typed result and refuse incompatible work.
  Raw exceptions, stack traces, credentials, provider internals, collection
  names, filesystem paths, and adapter identifiers must stay out of
  orchestrator prompts.
- Track attempted `(task_node_id, specialist)` pairs. The orchestrator must not
  repeat an incompatible pair or exceed the same-specialist-per-node cap.
- Apply these fixed session limits:
  - four total specialist dispatches;
  - four total orchestrator LLM calls, including structurally invalid calls
    and their bounded replacements;
  - two route corrections;
  - two invocations of the same specialist for one task node;
  - zero blind semantic retries.
- A second invocation of the same specialist is permitted only after a typed
  retryable operational failure or material new evidence changes its input.
- `partial` is a successful terminal outcome only when at least one validated
  evidence item exists. Completed subgoals alone never authorize `partial`;
  zero-evidence output must become
  `unavailable`, `needs_user_input`, `approval_required`, or `failed`.
- Queue-level process and lease retries must resume the persisted checkpoint
  and counters. They must never reset the four-dispatch budget.
- Keep local/private and public evidence as evidence. Cognition owns stance;
  L3/dialog owns final visible wording.
- Coding reading and proposal work may run through the coding specialist.
  Code mutation, command execution, approval, cancellation, blocker response,
  and bound-run continuation must remain governed by the existing coding-run
  lifecycle and permission contracts.
- Background workers must not send adapter text directly or call shared
  cognition directly. Completed work must use the accepted-task result source,
  cognition, dispatcher validation, and normal delivery.
- Use `parse_llm_json_output(...)` for canonical JSON parsing. Structural
  replacement remains owned by the producing LLM stage, consumes the persisted
  four-call orchestrator budget, and never resets across queue retries. Do not add
  orchestrator-side semantic repair, keyword routing, or unbounded retry.
- Keep stable capability descriptions and output contracts in static system
  prompts. Keep per-run task state and accumulated evidence in bounded human
  payloads.
- Define Python prompt constants with triple-single-quoted strings. Use
  `.format(...)` only for process-stable values. Put runtime values in the
  human message.
- Use `venv\Scripts\python.exe` for Python commands.
- Never read `.env` during planning, implementation, verification, or cutover.
- Before deleting production rows, stop all brain-service, background-worker,
  and delivery processes that can write either target collection; record
  pre-delete counts; execute the approved maintenance command; verify both
  counts are zero; then start the cutover runtime.
- Delete documents through the reviewed maintenance path. Do not drop the
  database, use collection-name globs, or delete any collection beyond
  `background_work_jobs` and `accepted_tasks`.
- Preserve `coding_runs`, calendar/scheduler rows, conversations, memories,
  traces, and every collection outside the two explicitly approved targets.
- Preserve unrelated user changes and all coding-agent work.

## Must Do

- Add one canonical L2d-visible resolver capability: `task_resolution_request`.
- Remove `local_context_recall` and `public_answer_research` from the
  cognition-visible resolver roster.
- Remove model-facing `accepted_task_request` and its generic action-spec
  projection.
- Retain `human_clarification`, `approval_preparation`,
  `self_goal_resolution`, `accepted_task_status_check`, `future_speak`,
  memory-lifecycle actions, and bound coding-run lifecycle semantics.
- Retain the existing `goal_resolution` enum. Define `answerable_now` as
  current-evidence sufficiency, not a scheduling or latency prediction.
- Add a `kazusa_ai_chatbot.task_resolution` package with one public inline
  entrypoint, one resume entrypoint, typed contracts, deterministic checkpoint
  management, a bounded orchestrator LLM stage, and a README.
- Register exactly four initial specialists: `local_context`, `public_research`,
  `coding`, and `text_computation`.
- Integrate the existing local-context resolver, public complex resolver,
  coding-run public API, and text/computation implementation through explicit
  specialist handlers.
- Add typed `resolved`, `partial`, `incompatible`,
  `temporarily_unavailable`, `needs_user_input`, `approval_required`, and
  `failed` specialist outcomes.
- Add a typed task-resolution checkpoint with nodes, validated evidence,
  remaining needs, attempted specialist pairs, dispatch counters, route
  corrections, specialist invocation counts, and prompt-safe trace summaries.
- Add `TASK_RESOLUTION_INLINE_BUDGET_SECONDS` to `config.py` with default `30.0`,
  minimum `1.0`, and maximum `120.0`.
- Execute the foreground session inline until it reaches a terminal outcome or
  the configured wall-clock budget.
- On budget exhaustion, persist the same checkpoint through accepted-task and
  background-work materialization. Queue
  `requested_worker="task_orchestrator"` and preserve the semantic objective
  independently from persona rationale.
- Resume deferred work from the persisted checkpoint after process or lease
  retries. Persist checkpoint progress after every completed specialist
  dispatch.
- Persist a selected dispatch before durable execution and persist its
  `started` phase before invoking the specialist. A resumed `started` dispatch
  is never launched again; it becomes an at-most-once unavailable result.
- Defer every newly selected coding dispatch before invoking it inline. The
  background worker may consume the persisted handover through the frozen
  coding public boundary, while this plan's tests substitute a closed stub and
  never execute the coding-agent path.
- When a validated `resolved` or evidence-bearing `partial` specialist result
  retains semantic `remaining_needs`, materialize those needs as bounded child
  nodes that depend on the completed current node. Activate the next eligible
  node and continue while node, dispatch, orchestrator-call, and timing budgets
  remain. Terminalize evidence-bearing `partial` only when no eligible
  continuation remains or a fixed budget is exhausted.
- Add one generic `task_orchestrator` worker and remove the generic background
  router and its LLM route choice.
- Remove direct generic `coding_agent` and `text_artifact` background-worker
  registrations and worker wrappers.
- Route bound coding-run worker payloads through `task_orchestrator` to the
  coding specialist while retaining deterministic operation validation.
- Retain `future_speak` as a deterministic non-generic scheduling worker.
- Extend resolver observations, background worker results, accepted-task
  result projection, and result-ready cognition to preserve a prompt-safe
  `partial` outcome and explicit remaining limitations.
- Change accepted-task identity to use the validated semantic resolution
  objective and trusted requester/conversation scope. Do not use route reason
  or raw worker selection language as executable task identity.
- Big-bang the accepted-task and background-job document schemas to v2 and
  reject v1 rows. The approved cutover deletes every v1 row before the v2
  runtime starts.
- Add a maintenance CLI that counts and deletes all documents from exactly
  `background_work_jobs` and `accepted_tasks`, requires an explicit execution
  flag and exact confirmation phrase, reports deleted counts, and verifies
  zero remaining rows.
- Delete all existing rows in both approved collections during offline cutover,
  including delivered history, while preserving coding-run and calendar data.
- Delete obsolete tests that enforce the old capability names, generic router,
  direct coding-worker route, direct text-artifact-worker route, or v1 task
  documents. Replace them with target-state tests.
- Update subsystem READMEs, the root README architecture map, HOWTO settings,
  subagent-interface documentation, and the cognition contract reference.
- Produce raw deterministic/live evidence and agent-authored readable quality
  reviews for the two production failures and the complete target routing
  matrix.

## Deferred

- Do not add new specialist families beyond the four approved initial
  specialists.
- Do not add image generation, arbitrary MCP tools, package installation,
  publishing, email, calendar mutation, adapter messaging, arbitrary shell,
  arbitrary filesystem access, or arbitrary database access to the
  orchestrator.
- Do not generalize the complex-task resolver into a universal resolver. It
  remains the public-research specialist behind its declared public IO.
- Do not merge local-context resolver internals into task resolution. It
  remains the local/private specialist behind its declared public IO.
- Do not redesign coding-agent internals, coding-run persistence,
  repository indexing, approval rules, patch application, or verification.
- Do not change future-speak scheduling semantics.
- Do not change memory-lifecycle, self-cognition, reflection, consolidation,
  dialog generation, adapter delivery, or calendar contracts beyond required
  task-result projection wiring.
- Do not add compatibility aliases for `local_context_recall`,
  `public_answer_research`, or model-facing `accepted_task_request`.
- Do not retain the generic background router as a fallback.
- Do not preserve, migrate, archive, or export the two deleted task-history
  collections as part of cutover.
- Do not delete coding runs, calendar rows, conversations, memories, protected
  traces, or unrelated audit data.
- Do not make session limits dynamically model-controlled. Only the inline
  wall-clock budget is configurable in this plan.
- Do not add a universal subagent base class or bridge the existing
  family-local registries.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Cognition capability roster | bigbang | Replace local/public resolver handles and generic accepted-task action with `task_resolution_request`. |
| Goal resolution | retained | Keep the existing enum and define `answerable_now` as evidence sufficiency only. |
| Foreground execution | bigbang | Execute one task-resolution session inline for the configured 30-second default budget. |
| Background execution | bigbang | Resume the same checkpoint through `task_orchestrator`; remove generic worker routing. |
| Specialist roster | bigbang | Register exactly local context, public research, coding, and text/computation handlers. |
| Coding lifecycle | retained | Preserve the public coding-run and approval lifecycle; access it through the coding specialist. |
| Future speak | retained | Preserve the deterministic scheduling worker outside generic task routing. |
| Accepted-task documents | bigbang | Move to v2 and delete every existing document before runtime start. |
| Background-job documents | bigbang | Move to v2 and delete every existing document before runtime start. |
| Coding-run and calendar data | retained | Preserve existing documents unchanged. |
| Tests and docs | bigbang | Delete old-route assertions and document only the canonical target state. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- For every big-bang area, delete or rewrite legacy references instead of
  preserving them.
- Do not add compatibility shims, adapters, fallback routes, dual reads,
  dual writes, aliases, or preservation of old task state.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

The action planner sees only `task_resolution_request`. Evidence-sufficient
turns keep `answerable_now` and proceed to normal dialog; required-evidence
turns enter resolver authorization and the inline task-resolution service.
Specialist names, timing, counters, queue ids, leases, and persistence remain
outside the planner.

Inline execution creates `task_resolution_checkpoint.v1`, dispatches one
specialist/subgoal at a time, validates and merges evidence, and either returns
resolved/evidence-bearing partial evidence to the next cognition cycle or
atomically persists `accepted_task.v2` plus `background_work_job.v2` when the
wall-clock budget expires.

The `task_orchestrator` worker resumes the same counters and checkpoint,
persists after every dispatch, and projects terminal result-ready state through
source-bound cognition, dialog, dispatcher validation, and adapter delivery.
An incompatible specialist records its `(node, specialist)` pair and remaining
need; a different specialist may then resolve it within the fixed correction
cap. User-input, approval, unavailable, and structural failures remain distinct.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Public capability | Use `task_resolution_request` | One semantic path removes domain/timing competition from cognition. |
| Answer timing | Inline first, then checkpoint | The selector judges evidence sufficiency while deterministic runtime owns latency. |
| Inline budget | Configurable float, default 30 seconds | Fast work can finish in-turn while long work remains bounded. |
| Background executor | One generic `task_orchestrator` worker | Worker selection is removed from the generic queue. |
| Specialist selection | One next specialist per orchestrator dispatch | A wrong choice can return typed incompatibility and be corrected. |
| Specialist roster | Existing local, public, coding, and text/computation agents | Reuse approved public IO and avoid speculative tools. |
| Session dispatch cap | Four | Covers wrong-route recovery and one evidence-dependent revisit without unbounded loops. |
| Route-correction cap | Two | Prevents specialist ping-pong. |
| Same specialist/node cap | Two | Allows one justified revisit after transient failure or new evidence. |
| Partial result | Successful when evidence-bearing | Local/public resolvers often provide useful incomplete evidence. |
| Process retry | Resume checkpoint and counters | Queue retries must not multiply semantic work. |
| Dispatch crash safety | Persist selected/started phase and never relaunch started work | Preserves at-most-once specialist execution when a result cannot be recovered. |
| Dependency continuation | Materialize validated remaining needs as bounded child nodes | Enables multi-specialist continuation without deterministic semantic routing. |
| Orchestrator structural errors | Count every raw call against the four-call session cap | Bounded replacement survives promotion and queue retry. |
| Coding mutation | Existing approval lifecycle | Task orchestration does not expand side-effect authority. |
| Coding verification | Frozen imports and closed handover stubs only | User directed this plan to avoid the known-broken coding execution branch. |
| Task identity | Semantic objective plus trusted scope | Prevents route rationale from biasing execution or duplicate identity. |
| Task-history cutover | Delete both approved collections | User approved a clean big-bang history boundary. |
| Coding/calendar history | Preserve | These are independent retained lifecycles. |
| Compatibility | None | User selected proper big-bang replacement. |

## Data Migration

### Approved destructive scope

Delete every document from:

- `background_work_jobs`;
- `accepted_tasks`.

Preserve every other collection, including:

- coding-run ledger collections;
- calendar and scheduler collections;
- conversations and delivery receipts;
- user and shared memory;
- character state and profiles;
- protected LLM traces and event logs.

### Maintenance CLI contract

Add:

```text
scripts/clear_background_task_history.py
```

The CLI must:

- use the project database/config boundary without reading `.env` directly;
- target only the two exact collection constants;
- default to count-only dry-run behavior;
- require `--execute`;
- require the exact confirmation phrase
  `DELETE_BACKGROUND_WORK_JOBS_AND_ACCEPTED_TASKS`;
- report before, deleted, and remaining counts for both collections;
- fail if either remaining count is nonzero;
- emit no document contents;
- return nonzero on database, validation, or verification failure;
- remain idempotent when both collections are already empty.

### Offline cutover order

1. Complete code and test verification without deleting production data.
2. Stop every brain-service, worker, and delivery process that can write either
   target collection.
3. Run the maintenance CLI in dry-run mode and record both counts.
4. Run the explicit destructive command with the exact confirmation phrase.
5. Run dry-run/count verification again and record zero/zero.
6. Start the v2 runtime.
7. Verify v2 indexes and create one new target-state accepted task.
8. Verify that the task completes or becomes partial through the new
   orchestrator and that no v1 document appears.

The deletion step is irreversible. The user explicitly selected deletion
without archive or export to keep history clean.

## Contracts And Data Shapes

### Resolver capability

`ResolverCapabilityRequestV2.capability` must accept
`task_resolution_request` and reject the removed local/public names.

The model-authored fields remain:

```python
{
    "capability": "task_resolution_request",
    "semantic_goal": str,
    "reason": str,
    "evidence_handles": list[str],
}
```

Runtime budget exhaustion, not the model, determines durable continuation.

### Public task-resolution IO

```python
async def resolve_task_inline(
    request: ResolverCapabilityRequestV2,
    execution_context: TaskResolutionExecutionContextV1,
    *,
    inline_budget_seconds: float,
) -> TaskResolutionResultV1: ...

async def resume_task_resolution(
    checkpoint: TaskResolutionCheckpointV1,
    execution_context: TaskResolutionExecutionContextV1,
) -> TaskResolutionResultV1: ...
```

`TaskResolutionExecutionContextV1` is persisted with the checkpoint and has
exact fields: `schema_version`, character/platform/channel/requester/source
message identifiers, prompt-safe local time and message context, bounded recent
and wide history, conversation progress, bounded persona/conversation
summaries, current timestamp, active-turn ids, session media refs, coding
workspace root, and `max_output_chars`. The handler mappers may consume only
these fields. Raw messages, credentials, adapter objects, database handles,
and coding-agent internal state are forbidden.

### Task-resolution checkpoint

```python
{
    "schema_version": "task_resolution_checkpoint.v1",
    "session_id": str,
    "semantic_objective": str,
    "source_scope": {
        "trigger_source": str,
        "platform": str,
        "channel_id": str,
        "channel_type": str,
        "source_message_id": str,
        "requester_global_user_id": str,
        "requester_platform_user_id": str,
    },
    "nodes": list[TaskResolutionNodeV1],
    "active_node_id": str,
    "evidence": list[TaskResolutionEvidenceV1],
    "remaining_needs": list[str],
    "attempted_specialists": list[TaskSpecialistAttemptV1],
    "dispatch_count": int,
    "orchestrator_call_count": int,
    "route_correction_count": int,
    "specialist_invocation_counts": list[TaskSpecialistInvocationCountV1],
    "pending_dispatch": TaskPendingDispatchV1 | None,
    "terminal_status": str,
    "trace_summary": list[TaskResolutionTraceEntryV1],
}
```

`TaskResolutionNodeV1` contains exact `schema_version`, `node_id`, `objective`,
`status`, and `depends_on` fields. `TaskResolutionEvidenceV1` contains exact
`schema_version`, `evidence_id`, `task_node_id`, `specialist`, `summary`,
`provenance_refs`, and `limitations` fields. Attempt and invocation-count rows
use exact `(task_node_id, specialist)` fields; trace rows contain only dispatch
index, node id, specialist, result status, and bounded reason. Every list is
bounded by the four-dispatch/eight-node session limits. Raw specialist payloads,
prompts, stack traces, credentials, worker metadata, and adapter ids are
forbidden.

`TaskPendingDispatchV1` contains exact `schema_version`, `task_node_id`,
`specialist`, `subgoal`, `coding_objective_mode`, and `phase` fields.
`coding_objective_mode` is `none`, `read_only`, or `propose_patch`; it must be
`none` for every non-coding specialist. `phase` is `selected` or `started`.
The orchestrator decision has exact `specialist`, `subgoal`, and
`coding_objective_mode` fields. Every raw orchestrator LLM call increments
`orchestrator_call_count` before parsing; structural replacement is allowed
only while the fixed four-call budget remains.

The deterministic state owner assigns child node ids. For each validated
remaining need returned by a successful specialist, it creates at most one
bounded child node with `depends_on=[completed_node_id]`, deduplicates exact
normalized objectives, respects the eight-node cap, and activates the first
pending node whose dependencies are resolved. This materializes specialist
output without keyword routing or semantic rewriting.

### Specialist request

```python
{
    "schema_version": "task_specialist_request.v1",
    "task_node_id": str,
    "objective": str,
    "available_evidence": list[dict[str, object]],
    "remaining_needs": list[str],
    "trusted_scope": dict[str, str],
    "coding_objective_mode": "none | read_only | propose_patch",
}
```

Specialist handlers receive this canonical task-resolution contract and map it
to the existing specialist's public IO. The mapping is owned by each handler;
the orchestrator prompt does not generate low-level arguments.

The four handler entrypoints are
`resolve_with_local_context(request, execution_context)`,
`resolve_with_public_research(request, execution_context)`,
`resolve_with_coding(request, execution_context)`, and
`resolve_with_text_computation(request, execution_context)`. Local context maps
the execution context into the existing request/context/options triplet for
`resolve_local_context(...)`; public research maps it into the existing
triplet for `resolve_complex_task(...)`; coding maps only into the frozen
public `CodingRun*Request` types and uses only the validated
`coding_objective_mode`. Text/computation moves the current two-stage
text router/generator into `specialists/text_computation.py` and reuses the
existing deterministic expression evaluator for caller-supplied numeric
expressions. It refuses web, repository, attachment, filesystem, package,
shell, database, and adapter work. Every handler returns exactly one
`TaskSpecialistResultV1` and persists no state outside its declared public IO.

### Specialist result

```python
{
    "schema_version": "task_specialist_result.v1",
    "specialist": str,
    "status": (
        "resolved"
        | "partial"
        | "incompatible"
        | "temporarily_unavailable"
        | "needs_user_input"
        | "approval_required"
        | "failed"
    ),
    "evidence": list[dict[str, object]],
    "completed_subgoals": list[str],
    "remaining_needs": list[str],
    "reason": str,
    "retryable": bool,
    "coding_run_context": dict[str, object],
}
```

`coding_run_context` is accepted only from the coding specialist. The coding
adapter projects public `CodingRunResponse` fields into exact prompt-safe
`schema_version`, `coding_run_ref`, `status`, `summary`, `limitations`,
`allowed_next_actions`, and `followup_open` fields. Other specialists must
return it absent.

### Task-resolution result

```python
{
    "schema_version": "task_resolution_result.v1",
    "status": (
        "resolved"
        | "partial"
        | "needs_user_input"
        | "approval_required"
        | "unavailable"
        | "failed"
        | "deferred"
    ),
    "prompt_safe_summary": str,
    "evidence": list[dict[str, object]],
    "completed_subgoals": list[str],
    "remaining_needs": list[str],
    "checkpoint": dict[str, object],
    "coding_run_context": dict[str, object],
}
```

`partial` requires at least one validated evidence item; completed subgoals
alone never authorize `partial`.
`deferred` requires a nonterminal checkpoint and remaining dispatch budget.

### Background worker payload

```python
{
    "schema_version": "task_orchestrator_worker_payload.v1",
    "operation": "resume_task_resolution | continue_bound_coding_run",
    "checkpoint": TaskResolutionCheckpointV1 | None,
    "coding_request": CodingRunContinueRequest | None,
}
```

Exactly one operation payload is populated. Generic promotion uses
`resume_task_resolution`. Bound coding actions use
`continue_bound_coding_run`, preserve the validated run ref, closed action,
trusted approval evidence, execution request/specs, and idempotency identity,
and deterministically preselect the coding handler without exposing those
fields to the orchestrator LLM. Both queue
`requested_worker="task_orchestrator"`; results map through the same
prompt-safe coding context and accepted-task delivery boundary.

### Accepted-task and job v2

- `accepted_task.v2` adds prompt-safe `completion_status` with
  `none`, `resolved`, `partial`, or `failed`.
- `background_work_job.v2` accepts only reviewed worker payload versions.
- The task-orchestrator checkpoint remains internal job payload state.
- Accepted-task prompts receive semantic summary, completion status, remaining
  limitations, and sanitized coding-run context only.
- Final `unavailable`, `needs_user_input`, `approval_required`, and `failed`
  resolution results map to accepted-task `completion_status="failed"` while
  retaining their exact prompt-safe result kind and limitations. `resolved`
  and `partial` map one-to-one. `incompatible` remains nonterminal until the
  route-correction budget is exhausted; exhausted incompatibility becomes
  `unavailable`. `temporarily_unavailable` defers only with remaining dispatch
  budget and otherwise becomes `unavailable`.
- v1 documents are rejected. The cutover deletes all v1 rows before runtime.

## LLM Call And Context Budget

Default context-window cap: 50k tokens.

### Before

| Stage | Normal calls | Location | Notes |
|---|---:|---|---|
| Action planning | 1 | response path | Chooses action or resolver domain and goal resolution. |
| Action/resolver authorization | 0-1 | response path | Runs for selected candidate family. |
| Local/public resolver | capability-owned | response path | Existing internal bounded loops. |
| Background router | 1 | background | Chooses coding or text worker. |
| Worker-local classifier/generator | worker-owned | background | A rejection ends the job. |

### After

| Stage | Hard cap | Location | Context |
|---|---:|---|---|
| Action planning | 1 initial plus existing structural replacements | response path | One task-resolution affordance; no specialist roster. |
| Resolver authorization | Existing cap | response path | Authorizes one task-resolution request. |
| Task orchestrator decision | 4 raw calls total per session | inline/background | Includes invalid structural candidates and bounded replacements; counter persists across promotion and queue retry. |
| Specialist execution | 4 total dispatches | inline/background | Each existing specialist retains its internal cap. |
| Final cognition recurrence | Existing resolver-cycle cap | response path/result delivery | Receives prompt-safe resolved or partial observation. |

Specialist call budget frozen for this plan:

| Specialist | Public/owned entrypoint | Route and maximum nested calls per dispatch | Prompt/deadline rule |
|---|---|---|---|
| local_context | `resolve_local_context(...)` | `RAG_PLANNER_LLM` + `RAG_SUBAGENT_LLM`; current default limits, maximum 14 stage/helper calls | Existing stage caps; start inline only with remaining deadline, 24k task payload cap. |
| public_research | `resolve_complex_task(...)` | `WEB_SEARCH_LLM`; current four-iteration/three-node-attempt/one-subagent-attempt caps, maximum 30 stage/helper calls | Existing resolver caps; deadline expiry checkpoints as temporary unavailable. |
| coding | frozen `start/continue/get_coding_run(...)` | Existing `CODING_AGENT_PM_LLM`/`CODING_AGENT_PROGRAMMER_LLM` run-owned caps; one public run operation per dispatch | New selections defer before inline invocation. Production background execution owns the public call; plan verification uses a closed stub and never executes this path. |
| text_computation | specialist-owned router/generator plus deterministic calculator | `BACKGROUND_WORK_LLM`, exactly two LLM calls for text and zero for validated calculation | Each human payload is at most 8k characters and output uses configured background-work cap. |

One task session therefore adds at most four raw orchestrator LLM calls plus
four specialist public operations. Invalid orchestrator structure consumes the
same four-call budget. Nested specialist calls are unchanged from their
existing owners and are never multiplied by queue retry. Inline code applies
the remaining wall-clock deadline to the complete public operation and starts
no dispatch with less than one second remaining; background continuation has
no inline deadline and still shares the four-dispatch ledger.

The serialized orchestrator human payload must not exceed 24,000 characters.
The system prompt remains static. Evidence projection must prefer bounded
semantic summaries and provenance over raw source payloads.

The inline service has a hard wall-clock budget from
`TASK_RESOLUTION_INLINE_BUDGET_SECONDS`, default 30 seconds. When the remaining
budget cannot safely start another specialist call, persist and defer before
launching the call. A call that reaches the remaining deadline returns a typed
temporary-unavailable observation and checkpoints without resetting counters.

Background continuation may consume the remaining dispatch budget without the
inline deadline. Existing worker lease and process-attempt limits remain
deterministic. The four-dispatch semantic budget survives every queue retry.

Before each durable specialist call, the worker persists the pending dispatch
as `started`. If the process or lease ends before a result is durably recorded,
resume converts that started dispatch into an at-most-once unavailable result
and never launches it again. A selected coding dispatch always promotes before
this transition; inline execution never calls the coding public API.

No new evaluator, judge, semantic repair, or fallback LLM stage is added.

## Change Surface

### New production package

- `src/kazusa_ai_chatbot/task_resolution/__init__.py`
- `src/kazusa_ai_chatbot/task_resolution/README.md`
- `src/kazusa_ai_chatbot/task_resolution/contracts.py`
- `src/kazusa_ai_chatbot/task_resolution/service.py`
- `src/kazusa_ai_chatbot/task_resolution/orchestrator.py`
- `src/kazusa_ai_chatbot/task_resolution/state.py`
- `src/kazusa_ai_chatbot/task_resolution/specialists/__init__.py`
- `src/kazusa_ai_chatbot/task_resolution/specialists/local_context.py`
- `src/kazusa_ai_chatbot/task_resolution/specialists/public_research.py`
- `src/kazusa_ai_chatbot/task_resolution/specialists/coding.py`
- `src/kazusa_ai_chatbot/task_resolution/specialists/text_computation.py`

### Cognition and resolver integration

- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/resolver_authorization.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py`
- `src/kazusa_ai_chatbot/cognition_resolver/state.py`
- `src/kazusa_ai_chatbot/cognition_resolver/telemetry.py`
- `src/kazusa_ai_chatbot/cognition_resolver/README.md`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`
- `src/kazusa_ai_chatbot/self_cognition/runner.py`

### Accepted-task, background work, and action contracts

- `src/kazusa_ai_chatbot/action_spec/registry.py`
- `src/kazusa_ai_chatbot/action_spec/models.py`
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`
- `src/kazusa_ai_chatbot/action_spec/execution.py`
- `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py`
- `src/kazusa_ai_chatbot/action_spec/README.md`
- `src/kazusa_ai_chatbot/accepted_task/models.py`
- `src/kazusa_ai_chatbot/accepted_task/lifecycle.py`
- `src/kazusa_ai_chatbot/accepted_task/README.md`
- `src/kazusa_ai_chatbot/background_work/models.py`
- `src/kazusa_ai_chatbot/background_work/jobs.py`
- `src/kazusa_ai_chatbot/background_work/worker.py`
- `src/kazusa_ai_chatbot/background_work/result_source.py`
- `src/kazusa_ai_chatbot/background_work/delivery.py`
- `src/kazusa_ai_chatbot/background_work/subagent/__init__.py`
- `src/kazusa_ai_chatbot/background_work/subagent/task_orchestrator.py`
- `src/kazusa_ai_chatbot/background_work/README.md`
- `src/kazusa_ai_chatbot/db/accepted_tasks.py`
- `src/kazusa_ai_chatbot/db/background_work_jobs.py`
- `src/kazusa_ai_chatbot/db/__init__.py`
- `src/kazusa_ai_chatbot/config.py`

### Delete after replacement

- `src/kazusa_ai_chatbot/background_work/router.py`
- `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`
- `src/kazusa_ai_chatbot/background_work/subagent/text_artifact.py`
- `src/kazusa_ai_chatbot/background_work/providers.py`

### Coding integration (read-only public boundary)

- Use the current frozen public exports from
  `src/kazusa_ai_chatbot/coding_agent/__init__.py`.
- Update coding-run prompt-safe result projection only where required for
  task-resolution result handoff.
- Do not edit any coding-agent file, internal prompt, action loop, repository index,
  candidate workspace, patcher, verifier, or approval logic under this plan.

### Maintenance and documentation

- `scripts/clear_background_task_history.py`
- `README.md`
- `docs/HOWTO.md`
- `docs/SUBAGENT_INTERFACES.md`
- `development_plans/reference/designs/cognition_contracts_design.md`
- `src/kazusa_ai_chatbot/brain_service/README.md`
- `src/kazusa_ai_chatbot/self_cognition/README.md`
- `src/kazusa_ai_chatbot/local_context_resolver/README.md`
- `src/kazusa_ai_chatbot/complex_task_resolver/README.md`
- `src/kazusa_ai_chatbot/rag/README.md`

### Test additions and replacements

- `tests/test_task_resolution_contracts.py`
- `tests/test_task_resolution_state.py`
- `tests/test_task_resolution_orchestrator.py`
- `tests/test_task_resolution_specialists.py`
- `tests/test_task_resolution_inline_promotion.py`
- `tests/test_task_resolution_background_resume.py`
- `tests/test_task_resolution_live_llm.py`
- `tests/test_task_resolution_persona_e2e_live_llm.py`
- `tests/test_task_resolution_cutover_live_db.py`

Existing action-selection, cognition-resolver, accepted-task, background-work,
local-context, complex-resolver, coding-run integration, persona, and prompt
contract tests must be updated or replaced where their target state changes.

## Overdesign Guardrail

- Actual problem: overlapping one-shot capability and worker choices repeatedly
  send evidence tasks to terminally incompatible workers.
- Minimal change: expose one model-facing task-resolution capability and move
  domain selection into one bounded, resumable orchestrator over four existing
  specialists.
- Ownership boundaries: cognition owns acceptance and evidence sufficiency;
  orchestrator LLM owns next-specialist semantic choice; specialists own
  domain parameters and refusal; deterministic code owns timing, limits,
  persistence, retry, permission, and delivery; dialog owns wording.
- Rejected complexity: additional specialists, universal tool schemas,
  model-controlled limits, compatibility aliases, fallback routers, dual
  paths, new evaluators, arbitrary side effects, complex-resolver
  generalization, coding-agent redesign, and separate task-session
  persistence collections.
- Evidence threshold: add another specialist or new side-effect mode only
  after a trace-backed failure demonstrates that none of the four approved
  specialists can own a required near-term task.

The checkpoint is stored in the internal background job payload after
promotion. Do not create a separate task-resolution MongoDB collection unless
future evidence proves that job-owned checkpointing cannot satisfy atomic
resume and lease recovery.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve every contract and limit in this plan.
- The responsible agent must not introduce new architecture, migration
  strategies, compatibility layers, fallback routes, extra specialists, or
  unrelated features.
- Changes outside the listed change surface require user approval before
  implementation.
- The responsible agent must search for existing equivalent behavior before
  adding helpers. Shared behavior must be moved or abstracted according to
  `py-style` rather than duplicated.
- The responsible agent may delete legacy routing code and tests listed in
  scope after replacement tests exist.
- The responsible agent must preserve all coding-agent files and integrate only
  through the frozen current public API.
- The responsible agent must not perform dependency upgrades, broad
  formatting, prompt rewrites outside the target stages, or unrelated cleanup.
- If the plan and source disagree, stop and report the exact discrepancy
  before changing the contract.
- If a required instruction is impossible, stop and report the blocker rather
  than inventing a substitute.

## Implementation Order

### Stage 0: Frozen boundary and baseline lock

1. Contract-test the frozen coding exports/types without consulting its plan or internals.
2. Reread required docs/source/tests; record Git state, two failure artifacts,
   old roster/router/v1 schemas/config defaults, and focused baselines.
3. Add and run the named target-state tests; record expected missing-contract failures.

### Stage 1: Core contracts, specialists, and inline service

1. Add package IO, exact validators, bounded ledgers, evidence merge, and README.
2. Add four handlers over declared public IO; prove refusal and coding approval.
3. Add the static orchestrator, four-dispatch loop, counters, config bounds,
   deadline/defer behavior, and no-persistence inline success.
4. Rerun the first focused command until the approved contract passes.

### Stage 2: Cognition and durable big-bang integration

1. Replace V2 local/public/generic-task affordances with `task_resolution_request`.
2. Wire authorization, inline observation, blockers, and durable promotion.
3. Add v2 task/job schemas, payload union, checkpoint updates, single worker,
   bound coding continuation, partial delivery, and deterministic future-speak.
4. Rerun the second and third focused commands until integration passes.

### Stage 3: Cleanup, documentation, and maintenance boundary

1. Delete the router, direct coding/text workers, obsolete tests, aliases, and docs.
2. Update all listed ICDs and add the exact two-collection maintenance CLI/tests.
3. Run compile/prompt/static greps and focused deterministic suites.
4. Run the full non-live regression only after focused suites pass.

### Stage 4: Live quality and database rehearsal

1. Run every named live LLM and persona function separately; inspect and save
   raw/parsed output, choices, evidence, limits, counters, and terminal result.
2. Author the indexed readable reviews and obtain user quality acceptance.
3. Run live DB rehearsal only against `_test_kazusa_live_llm`.

### Stage 5: Offline production cutover and review

1. Obtain explicit destructive-cutover authorization; stop all target writers.
2. Record counts, execute the exact maintenance command, verify zero/zero,
   start v2, and verify preserved data, first v2 task, health, and delivery.
3. Run the independent code review, remediate in-scope findings, rerun affected
   gates, record evidence, mark completed, and archive.

## Execution Model

- The parent owns orchestration, test code, baselines, verification, evidence,
  review remediation, lifecycle updates, and destructive cutover control.
- After the parent records focused failing tests, exactly one production-code
  subagent owns every planned production and documentation edit. It does not
  edit tests and closes after the planned implementation is complete.
- The parent may continue integration tests and validation while that subagent
  works, then runs all focused and broader gates against the integrated tree.
- After planned verification passes, exactly one independent review subagent
  reviews the full plan, diff, evidence, lifecycle records, and destructive
  boundary without implementing fixes.
- The parent remediates in-scope review findings and reruns affected gates. A
  contract or change-surface finding stops execution for plan amendment.
- Native subagent unavailability stops execution unless the user explicitly
  authorizes fallback execution.

## Progress Checklist

- [x] Approval gate and frozen interface contract recorded.
  - Covers: independent plan review and Stage 0 step 1.
  - Verify: status/registry agree; public-export import contract passes; no
    coding-agent file is in the planned write set.
  - Evidence: record review remediation and import result below.
  - Handoff: lock Stage 0 baselines. Sign-off: `/root, 2026-08-01`.
- [x] Stage 0 baselines and target-state failing tests locked.
  - Covers: Stage 0 steps 2-3 and establishes the Stage 1 test contract.
  - Verify: named old-roster/router/schema tests pass as baseline and new
    contract selectors fail for missing `task_resolution` symbols.
  - Evidence: save commands and expected failures below.
  - Handoff: production subagent starts core implementation. Sign-off: `/root, 2026-08-01`.
- [x] Stage 1 core contracts, specialists, orchestrator, and inline service complete.
  - Verify: first focused task-resolution test command passes; compileall and
    prompt-render checks pass.
  - Evidence: changed files, counters, deadline, refusal, and partial cases.
  - Handoff: cognition/durable integration. Sign-off: `/root, 2026-08-01`.
- [x] Stage 2 cognition cutover and durable resume complete.
  - Verify: second and third focused commands pass, including bound coding,
    approval, lease retry, partial delivery, and future-speak cases.
  - Evidence: capability, v2 schema, checkpoint, and result-source outputs.
  - Handoff: cleanup/docs/maintenance CLI. Sign-off: `/root, 2026-08-01`.
- [x] Stage 3 cleanup, documentation, and maintenance boundary complete.
  - Verify: static greps, maintenance dry-run test, compileall, and documentation
    references match the target state.
  - Evidence: grep output and exact two-collection allowlist proof.
  - Handoff: deterministic regression. Sign-off: `/root, 2026-08-01`.
- [x] Focused and full non-live deterministic verification disposition complete.
  - Verify: every focused command passes; the exact full command's unrelated
    collection blockers are isolated by the 3,378-selected-test diagnostic and
    explicitly accepted by the user.
  - Evidence: raw pytest logs, totals, blocker list, and user acceptance below.
  - Handoff: one-at-a-time live quality gates. Sign-off: `/root, 2026-08-01`.
- [x] Live LLM, persona E2E, and live-DB rehearsal complete and accepted.
  - Verify: every named case runs individually; each has a raw artifact and
    agent-authored review; DB rehearsal uses only `_test_kazusa_live_llm`.
  - Evidence: `test_artifacts/task_resolution/` index and user acceptance.
  - Handoff: offline cutover authorization. Sign-off: `/root, 2026-08-01`.
- [x] Offline production cutover and post-cutover smoke complete.
  - Verify: stopped writers, before/deleted/remaining counts, preserved
    coding/calendar rows, first v2 task, health, and delivery evidence.
  - Evidence: exact maintenance command and zero-count proof below.
  - Handoff: independent code review. Sign-off: `/root, 2026-08-01`.
- [x] Independent Code Review, remediation, evidence, and archive complete.
  - Verify: review has no unresolved finding; affected tests rerun; registry
    and archived plan say `completed`.
  - Evidence: reviewer identity, findings, fixes, residual risks, and sign-off.
  - Handoff: none. Sign-off: `/root and /root/independent_code_review,
    2026-08-01`.

## Verification

### Static and prompt checks

```powershell
venv\Scripts\python.exe -m compileall -q src\kazusa_ai_chatbot\task_resolution src\kazusa_ai_chatbot\cognition_resolver src\kazusa_ai_chatbot\background_work src\kazusa_ai_chatbot\accepted_task src\kazusa_ai_chatbot\action_spec
```

Import and render every changed prompt constant in the project venv. Verify
`.format(...)` placeholders and exact output schemas at runtime.

```powershell
rg -n "local_context_recall|public_answer_research|accepted_task_request|route_background_work|WORKER = \"coding_agent\"|WORKER = \"text_artifact\"" src\kazusa_ai_chatbot tests README.md docs
```

Run source-only zero-match greps separately for removed model-facing constants,
registrations, and worker `WORKER` declarations; `rg` exit code `1` is the
expected success. Named exceptions are limited to the four new specialist
adapter files, `tests/test_task_resolution_contracts.py` forbidden-name
fixtures, retained `accepted_coding_task_request` and
`accepted_task_status_check` tests, and historical files under
`development_plans/archive/`. Every other match is a failure requiring removal
before verification continues.

### Focused deterministic tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_task_resolution_contracts.py tests\test_task_resolution_state.py tests\test_task_resolution_orchestrator.py tests\test_task_resolution_specialists.py -q
```

```powershell
venv\Scripts\python.exe -m pytest tests\test_task_resolution_inline_promotion.py tests\test_task_resolution_background_resume.py tests\test_accepted_task_lifecycle.py tests\test_background_work_jobs.py tests\test_background_work_delivery.py -q
```

```powershell
venv\Scripts\python.exe -m pytest tests\test_action_selection_prompt_contract.py tests\test_action_selection_payload.py tests\test_cognition_core_v2_action_planning_bugfix.py tests\test_cognition_core_v2_action_authorization.py tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_l2d_contract.py tests\test_cognition_resolver_loop.py -q
```

```powershell
venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_integration.py tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_service.py -q
```

### Required deterministic cases

- Evidence-sufficient ordinary question emits no task-resolution request.
- Public URL emits `task_resolution_request`.
- Local/private recall emits `task_resolution_request`.
- Unbound repository analysis emits `task_resolution_request`, then a stubbed
  coding selection produces a durable handover without calling a coding handler.
- Supplied-text transformation emits `task_resolution_request` when accepted
  as bounded work.
- Wrong text/computation selection returns incompatible and reroutes to public
  research without entering the coding path.
- Attempted pair cannot repeat after incompatibility.
- Route correction three is rejected at the cap.
- Dispatch five is rejected at the cap.
- Same specialist/node invocation three is rejected.
- Evidence-bearing partial is accepted.
- Zero-evidence partial is rejected.
- Inline resolved/partial creates no accepted task.
- Thirty-second configured budget is read and enforced.
- Inline timeout promotes one checkpoint exactly once.
- Background resume preserves all counters.
- Queue retry does not reset semantic counters.
- Coding selection persists exact objective mode and handover fields without
  executing the coding-agent path.
- Bound coding payload validation preserves operation and approval evidence
  without invoking any coding public operation.
- Structurally invalid orchestrator output consumes the persisted four-call
  budget and cannot multiply calls after queue retry.
- A persisted `started` dispatch is not relaunched after process or lease retry.
- Evidence-bearing partial output with remaining needs creates bounded child
  nodes and continues through local-to-public dependency resolution.
- Public-to-coding dependency selection ends at the validated durable coding
  handover; the coding handler remains uncalled.
- Partial result re-enters cognition and exposes limitations.
- Future-speak remains deterministic.
- Removed v1 capability names and document versions are rejected.

### Full non-live regression

```powershell
venv\Scripts\python.exe -m pytest -q -m "not live_llm and not live_db" --ignore-glob="tests/test_coding_agent*.py"
```

### Live LLM routing cases

Run one function at a time from `tests/test_task_resolution_live_llm.py` with
`-q -s -m live_llm`:

- `test_live_original_public_url_request`;
- `test_live_public_url_retry_is_not_code`;
- `test_live_private_conversation_memory_recall`;
- `test_live_public_current_fact_research`;
- `test_live_repository_analysis_handover_only`;
- `test_live_supplied_text_transformation`;
- `test_live_local_then_public_dependency`;
- `test_live_public_then_coding_handover_only`;
- `test_live_wrong_specialist_correction`;
- `test_live_evidence_bearing_partial`;
- `test_live_zero_evidence_unavailable`.

The two handover-only cases inject a closed coding stub, assert that it is not
called, and stop at the persisted `selected` coding dispatch. No live test in
this plan starts, continues, gets, approves, cancels, or otherwise executes a
coding run.

For every run, save raw traces and author a readable review under:

```text
test_artifacts/task_resolution/
```

The readable review must include run context, exact input, planner output,
orchestrator decisions, specialist results, counters, raw and projected
evidence, final cognition behavior, deterministic validation, and human
quality notes.

### Production-wired persona E2E

Run one case at a time:

```powershell
venv\Scripts\python.exe -m pytest tests\test_task_resolution_persona_e2e_live_llm.py::test_live_inline_result_returns_grounded_dialog -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_task_resolution_persona_e2e_live_llm.py::test_live_deferred_result_reenters_cognition -q -s -m live_llm
```

Verify:

- inline resolved/partial evidence produces grounded character dialog;
- deferred acknowledgement occurs only after durable persistence;
- background completion re-enters cognition;
- dialog states partial limitations without worker/queue vocabulary;
- adapter delivery remains outside the worker.

### Live DB and cutover rehearsal

Use only `_test_kazusa_live_llm`.

```powershell
venv\Scripts\python.exe -m pytest tests\test_task_resolution_cutover_live_db.py -q -s -m live_db
```

The test must create v1-like test rows in only the two test collections, run
the maintenance boundary against the test database, verify both collections
empty, preserve seeded coding/calendar rows, then create and complete one v2
task.

### Production cutover evidence

Record:

- stopped writer processes;
- pre-delete document counts;
- exact maintenance command and confirmation phrase;
- deleted counts;
- post-delete zero counts;
- v2 index checks;
- first v2 task id and terminal status;
- proof that coding-run and calendar data remained present;
- post-cutover service health and delivery result.

## Independent Plan Review

Before approval or execution, run one independent architecture/plan review
that checks:

- every user-confirmed decision is encoded;
- the action selector has one generic task-resolution capability;
- inline/background are one resumable session path;
- partial semantics and all counters are exact;
- wrong-specialist feedback is recoverable and bounded;
- coding approval cannot be bypassed;
- data deletion targets exactly two collections;
- big-bang policy contains no compatibility path;
- the frozen current coding public interface is used and coding internals are
  absent from the change surface;
- all mandatory sections and verification gates are present.

Record findings and remediation in Execution Evidence before changing status
to approved.

## Independent Code Review

After implementation and verification:

1. Spawn one independent review subagent with the complete plan, final diff,
   deterministic test evidence, live LLM reviews, DB rehearsal evidence, and
   cutover evidence.
2. Require review of capability removal, prompt ownership, checkpoint
   atomicity, attempt accounting, partial validation, lease recovery, coding
   approval, destructive scope, privacy projection, and delivery ownership.
3. Record every finding by severity with exact file/function evidence.
4. Remediate all critical and high findings. Resolve or explicitly disposition
   medium and low findings without expanding scope.
5. Rerun affected verification after remediation.
6. Record final independent sign-off in Execution Evidence.

## Acceptance Criteria

- `task_resolution_request` is the only model-facing generic task-resolution
  capability.
- `local_context_recall`, `public_answer_research`, and generic model-facing
  `accepted_task_request` are absent from the runtime roster.
- The action planner no longer chooses a specialist or execution horizon.
- `answerable_now` means current evidence is sufficient.
- Inline task resolution uses
  `TASK_RESOLUTION_INLINE_BUDGET_SECONDS` with default 30 seconds.
- The same checkpoint moves from inline to durable background execution.
- Four dispatches, two route corrections, and two same-specialist/node
  invocations are enforced across process retries.
- Four raw orchestrator calls, including structural replacements, are enforced
  across inline promotion and process retries.
- Pending dispatch phase is durable; a resumed `started` dispatch is never
  relaunched.
- Remaining needs create at most eight dependency nodes and activate only
  nodes whose dependencies are resolved.
- Wrong-specialist incompatibility can lead to a different specialist without
  failing the accepted task.
- Evidence-bearing partial results are terminal successes and reach cognition.
- Zero-evidence partial results are rejected.
- The four approved existing specialist families are integrated through their
  public IO.
- Coding mutations and execution retain existing approval and permission
  boundaries.
- Verification never executes the known-broken coding-agent branch; coding
  acceptance is limited to frozen-interface checks and durable handover shape.
- One generic `task_orchestrator` background worker replaces generic routing.
- Direct generic coding and text-artifact worker paths are deleted.
- Future-speak scheduling remains deterministic and functional.
- Semantic objective survives handoff independently from persona rationale.
- v1 accepted-task and background-job rows are rejected by v2 runtime.
- Every existing document in `background_work_jobs` and `accepted_tasks` is
  deleted during cutover.
- Coding-run and calendar data remain intact.
- Both original production failure cases return resolved or partial public
  evidence.
- Focused deterministic, full non-live, live LLM, persona E2E, live DB,
  cutover, and independent review gates pass.
- Documentation describes only the target architecture.

## Execution Evidence

Populate during execution:

- Approval, implementation authorization, coding-interface waiver, and baseline:
  user authorized execution and froze the current coding public boundary on
  2026-08-01. Independent reviewer `/root/plan_approval_review` rejected the
  draft for eight contract/gate gaps; remediation updated the V2 resolver IO,
  specialist execution context and handlers, bound coding payload, checkpoint
  invariants, nested-call budget, change surface, greps, checklist, and exact
  two-subagent execution model. Parent self-review found no remaining open
  decision; status advanced to approved without consulting coding plan status. Frozen `CodingRun*` imports and all three public signatures passed.
- Test-contract-first, core contracts, specialist, and inline-service evidence:
  the five legacy suites passed 50/50; both `test_artifacts/background_work_20260730T*_jobs.json` exports confirm the v1 coding/text failures. The two target files produced 10 expected missing-package failures.
- Cognition cutover, durable resume, counters, and legacy deletion evidence:
  target-state focused commands passed 48/48 for inline promotion, durable
  checkpoint resume, lease loss, queue retry, accepted-task v2, background-job
  v2, partial delivery, and result-ready delivery; cognition capability and
  authorization suites passed 110/110. Retained future-speak, action-result,
  persona acknowledgement, maintenance CLI, and documentation checks passed
  33/33. The C07 deterministic handoff gate now validates one correlated v2
  `task_orchestrator` job, coding-specialist result, public coding-run
  provenance, and delivery; its positive and seven negative cases passed 8/8.
- Focused deterministic and full non-live regression results:
  the four named focused commands passed 34/34, 47/47, 110/110, and 54/54.
  Retained future-speak, action-result, persona, maintenance, self-cognition,
  and documentation checks passed 100/100. Plan compileall, six changed-prompt
  imports/renders, `git diff --check`, and AST parsing for 23 edited
  CJK-bearing Python files passed. The exact full command remains independently
  blocked during collection by a missing replay module, Asuna profile, and
  conversation-history artifact. A diagnostic run excluding those four
  collection files found 16 additional unrelated harness failures caused by
  missing replay/profile/history artifacts, a missing external checkout, and
  the pre-existing 2293-character goal prompt exceeding its 2200-character
  test cap. After excluding only those seven known unrelated harness files,
  all 3,378 selected non-coding tests completed with exit code zero. No
  task-resolution-owned test failed, and no coding public operation ran.
- Pending contract and change-surface amendment evidence:
  implementation inspection found that the approved exact checkpoint and
  orchestrator decision shapes cannot safely persist an in-flight coding
  selection across an inline deadline, bound structural replacement attempts
  across queue retries, express read-only versus patch-proposal coding starts,
  or create and activate dependency nodes for the required local-to-public and
  public-to-coding cases. Static inspection also found unreachable legacy
  `background_work/providers.py` and legacy capability references in
  `self_cognition/runner.py` plus subsystem READMEs outside the listed change
  surface. The user approved the contract and change-surface amendment on
  2026-08-01 and directed all verification to avoid executing the known-broken
  coding-agent path while retaining handover and bounded-loop coverage.
- Live LLM raw artifacts, authored reviews, and persona E2E results:
  all 11 named live routing functions ran separately and passed. The original
  public URL and its explicit non-code retry both selected public research;
  private recall selected local context; current facts selected public
  research; supplied text selected text/computation; local-to-public resolved
  two dependency nodes; a forced text incompatibility corrected to public;
  evidence-bearing partial stopped grounded at the four-call cap; and three
  zero-evidence incompatibilities became unavailable. The repository and
  public-to-repository cases stopped at durable selected coding handovers with
  `read_only` and `propose_patch` modes. The two persona functions ran
  separately and passed: inline evidence produced grounded dialog with its
  limitation, and a durable accepted-task result re-entered as a typed
  `tool_result` before visible dialog. Raw JSON and parent-authored reviews are
  indexed at `test_artifacts/task_resolution/index.md`. The user accepted the
  routing and persona quality evidence on 2026-08-01.
- Live DB rehearsal and production pre-delete counts:
  the guarded `_test_kazusa_live_llm` rehearsal first exposed a conflicting v1
  named accepted-task index (`action_kind` versus canonical v2 `task_kind`).
  Startup index reconciliation and its deterministic regression test were
  added; the affected focused gate passed 48/48. The rerun passed: before
  counts were 1/1, deleted counts were 1/1, remaining counts were 0/0,
  coding/calendar preservation markers remained 1/1, and one new v2 task
  reached `result_ready` with `completion_status="resolved"`. No coding public
  operation ran. The user accepted the unrelated full-suite blockers and
  explicitly authorized the production cutover against `asuna_core_v2` on
  2026-08-01.
- Production delete command, deleted counts, and post-delete zero counts:
  no production brain/background/delivery writer process was running on the
  execution host and MongoDB reported zero active operations on the two target
  namespaces. The count-only command reported 8 `background_work_jobs` and 8
  `accepted_tasks`. The exact authorized command was
  `venv\Scripts\python.exe scripts\clear_background_task_history.py --execute
  --confirm DELETE_BACKGROUND_WORK_JOBS_AND_ACCEPTED_TASKS` with
  `MONGODB_DB_NAME=asuna_core_v2`; it deleted 8/8 rows and verified 0/0
  remaining. A second dry run independently confirmed 0/0.
- Preserved coding-run/calendar evidence and post-cutover smoke:
  `calendar_schedules` remained 8 and `calendar_runs` remained 1,201 before
  and after deletion. The frozen coding-run boundary has no MongoDB
  `coding_runs` collection in `asuna_core_v2`. Target-state accepted-task and
  background-job indexes were ensured, including the reconciled `task_kind`
  lookup. First v2 ids were
  `task-d8df9bd589af43fc85ba6cf3cf236043` and
  `job-964a8f8b5d724d48b5b45cded66e75a9`. One production runtime tick resumed a
  selected `text_computation` handover, evaluated `21 * 2 = 42`, reached
  `resolved`, projected a typed `tool_result` episode, and marked task/job
  delivered. Its persisted checkpoint records one orchestrator call, one
  dispatch, zero corrections, and no pending dispatch. Health returned `ok`
  with a healthy DB ping; v1 counts remained 0/0; coding-handler lookups and
  coding public operations remained zero. Raw review evidence is
  `test_artifacts/task_resolution/raw/cutover_production_asuna_core_v2.json`.
- Independent plan/code reviews and remediation:
  independent reviewer `/root/independent_code_review` initially rejected the
  implementation for one high checkpoint-materialization race, one medium
  direct-import cycle, and low inline-import placement. Remediation reserved a
  stable job id, transitioned accepted tasks to pending before claimable job
  insertion, converged concurrent same-ref materializers, guarded worker start
  and terminal states, added fresh-process imports, and moved imports to module
  scope. Follow-up review then found two medium delivery-state issues and one
  stale test description. Final remediation restored normal delivery claims to
  ready/retryable result states, added a future-speak-only running completion,
  constrained job delivery terminal writes to an active claim, and made every
  missing accepted-task/job delivery write fail closed or remain retryable.
  Divergence tests cover missing claim and both missing finalization writes.
  Post-review gates passed 34/34, 56/56, 110/110, and 54/54; retained non-coding
  regressions passed 115/115; the guarded live-DB rehearsal passed 1/1; fresh
  imports, compileall, artifact JSON, removed-route greps, and diff checks
  passed. The reviewer issued final approval with no unresolved finding.
  Residual risk is the explicit at-least-once adapter side-effect boundary:
  adapter delivery and MongoDB writes cannot be globally atomic, while the new
  guards keep mismatches observable and retryable. No coding public operation
  ran, and coding-agent source remained untouched.
- Final user acceptance and completion/archive commit:
  the user approved the plan, non-coding test boundary, live-quality evidence,
  unrelated full-suite blocker disposition, and production cutover on
  2026-08-01. Production remediation smoke proved pending-before-queued
  materialization and resolved `13 * 3 = 39`; all three synthetic remediation
  smoke pairs were removed by exact id/scope after evidence capture. Final
  production counts retain the original delivered v2 pair at 1/1, v1 rows at
  0/0, and calendar counts at 8 schedules/1,201 runs. This completed record was
  moved to `archive/completed/short_term/` and registered there.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Inline timeout interrupts a specialist | Lost progress or duplicate call | Check budget before dispatch, apply per-call deadline, checkpoint typed timeout, and preserve counters. |
| Wrong specialist returns plausible success | Silent domain error | Require provenance-bearing evidence and explicit specialist ownership tests; validate terminal evidence. |
| Specialists bounce between domains | Unbounded work | Persist attempted pairs, route corrections, and fixed dispatch limits. |
| Partial becomes empty success | Misleading answer | Require validated evidence for partial. |
| Queue retry restarts work | Multiplicative calls | Persist checkpoint after every dispatch and retain counters across leases. |
| Accepted-task creation duplicates inline session | Duplicate delivery | Materialize from one stable session/idempotency identity under existing atomic accepted-task lock. |
| Coding handler bypasses approval | Unauthorized mutation | Use only public coding-run APIs and existing operation/approval validators. |
| Frozen coding export is absent or incompatible | Broken integration | Contract-test current exports first and stop without editing coding-agent files. |
| Deletion runs while writers are active | New orphan or mixed-version rows | Offline cutover with stopped writers and zero-count verification. |
| Destructive scope expands | Data loss | Exact two-collection allowlist, confirmation phrase, counts, and preserved-collection checks. |
| Removed capability remains in prompt/test | Continued selector confusion | Big-bang greps, prompt rendering, and forbidden-name tests. |
| Orchestrator prompt grows with raw evidence | Local-model degradation | 24,000-character payload cap and semantic evidence projection. |
| Background result leaks worker vocabulary | Character breaks abstraction | Accepted-task semantic projection and persona E2E review. |
| Generic router survives as fallback | Two competing paths | Delete router source, imports, tests, and docs in the same cutover. |
