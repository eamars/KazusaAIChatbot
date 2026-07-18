# Action Spec

`kazusa_ai_chatbot.action_spec` owns the modality-neutral action contract used
between cognition, selected surface handlers, private action handlers,
consolidation, and calendar-owned continuation handlers.

It is not the dispatcher, not a scheduler, and not an LLM prompt package. Its
job is to define typed action residues, validate them deterministically, project
prompt-safe capabilities, and build prompt-safe result traces after downstream
owners have handled the action.

## Runtime Boundary

```text
L2d semantic action request
  -> deterministic route materialization
  -> optional specialist-owned target judgment
  -> deterministic executable materialization and target binding
  -> ActionSpecV1 validation
  -> owner handler
       l3_text / l3_image surface handler
       memory_lifecycle private handler
       accepted_task lifecycle resolver
       background_work internal queue handler
       orchestrator future-cognition request
  -> ActionResultV1 and SurfaceOutputV1
  -> EpisodeTraceV1
  -> prompt-safe consolidation projection
```

L2d emits semantic `action_requests`. Deterministic code materializes valid
requests into `ActionSpecV1` rows by attaching schema versions, source refs,
targets, continuation defaults, handler-owned params, and idempotency metadata.
The LLM must not receive, copy, compare, or emit raw persistence ids, adapter
ids, handler ids, collection names, or schema-version fields.

## Public Contracts

Core contracts live in `models.py`:

- `ActionSpecV1`: the single graph-visible action residue.
- `ActionSourceRefV1`: trusted provenance attached by code.
- `ActionTargetV1`: trusted target binding attached by code.
- `ActionContinuationV1`: bounded contract for follow-up cognition; direct
  tool-triggered cognition is not allowed.
- `CapabilitySpecV1`: internal capability registry entry.

Trace contracts live in `results.py`:

- `ActionResultV1`: validation, scheduling, execution, or rejection outcome for
  one action.
- `SurfaceOutputV1`: text, image, private, or future surface artifact.
- `EpisodeTraceV1`: consolidation-facing episode record containing cognition
  refs, actions, action results, and surface outputs.
- `ConsolidationActionProjectionV1`: prompt-safe projection consumed by the
  consolidator.

Current runtime accepts only `cognition_mode="deliberative"`. `reflex` is a
reserved schema slot and fails validation in this implementation slice.

## Resolver Capability Requests

`ActionSpecV1` represents selected surfaces and private actions that have
survived cognition. `ResolverCapabilityRequestV1` is earlier: it represents a
bounded evidence, HIL, approval, or private self-resolution request that must
return a prompt-safe observation into another full cognition cycle before final
action selection.

This keeps the ownership line clear:

- resolver capabilities retrieve evidence or create blocked observations;
- L1 -> L2 -> L2d re-judges the turn after each observation;
- only final L2d action requests become `ActionSpecV1` rows for L3, private
  handlers, calendar handoff, or consolidation traces.

## Capabilities

`build_initial_action_capabilities()` registers the available runtime
capabilities. Prompt projection exposes only L2d-facing semantic capabilities;
internal executable capabilities stay hidden from L2d prompts.

| Capability | Owner | Visibility | Meaning |
| --- | --- | --- | --- |
| `speak` | `l3_text` | `user_visible` | Selects a text surface. L2d provides surface intent, not final wording. |
| `memory_lifecycle_update` | `memory_lifecycle_specialist` | `private` | Selects specialist review for active-commitment lifecycle changes. L2d does not choose a memory target or lifecycle decision. |
| `apply_memory_lifecycle_update` | `memory_lifecycle` | `private` | Internal executable DB update produced after specialist alias validation. It is not projected to L2d. |
| `accepted_task_request` | `accepted_task` | `private` | Model-facing delayed-work request. Deterministic code creates or reuses an accepted task, then queues the internal executor for new work. |
| `accepted_coding_task_request` | `background_work` | `private` | Model-facing durable coding-run request. Deterministic code validates a closed coding action, creates or reuses accepted-task state, and queues the `coding_agent` worker with a versioned payload. |
| `accepted_task_status_check` | `accepted_task` | `private` | Reports active accepted-task state without enqueueing new work. |
| `background_work_request` | `background_work` | `private` | Internal executable queue request produced after accepted-task lifecycle validation. It is not projected to L2d as the public delayed-work contract. |
| `future_speak` | `background_work` | `private` | Queues a deterministic accepted-task-backed worker that schedules a later self-cognition message from an exact trigger time and semantic objective. |
| `trigger_future_cognition` | `orchestrator` | `private` | Requests a later cognition cycle contract; it does not call cognition directly. |

`send_message` is intentionally absent from the L2d registry. User-visible
text is represented as `speak`, routed through the selected L3 text surface,
and delivered only through the normal live response path. User-requested
future reminders or delayed follow-up messages use `future_speak`: L2d selects
the semantic private action, deterministic execution queues the background
worker, and the worker schedules `trigger_future_cognition` so the character
decides again at execution time how to speak. Private future self-checks that
are not user-facing reminders may still use `trigger_future_cognition`
directly.

Delayed accepted work is different from delayed contact. L2d may request
`accepted_task_request` as a semantic, private, route-only delayed task with a
route reason and surface intent. Deterministic materialization builds the
trusted task seed from prompt-safe state, rejects duplicate active tasks, and
then creates the internal `background_work_request` for new work. A later
background-work router chooses only the worker; worker subagents own task
classification and artifact generation. L2d never chooses worker-local
parameters. L3 sees only accepted-task pending, already-active, result-ready,
delivered, or failure acknowledgement state. Raw job ids, adapter ids, target
ids, leases, retries, filesystem paths, credentials, worker choices, and worker
state stay out of L2d and L3 prompts.

Durable coding-agent work uses `accepted_coding_task_request` instead of the
generic delayed-work route. L2d selects one closed semantic coding action:
`start`, `revise_proposal`, `summarize`, `status`, `approve_and_verify`, or
`respond_to_blocker`, or `cancel`. Deterministic validation requires a prompt-safe
`coding_run:<run_id>` reference for revision, summary, status, approval, and
cancellation or blocker response. Each continuation must also be present in
that offered run's `allowed_next_actions`. The handler then queues
`requested_worker="coding_agent"` with a
versioned worker payload. The worker maps that payload onto
`start_coding_run(...)`, `get_coding_run(...)`, or `continue_coding_run(...)`.
Execution specs are accepted only as structured allowlisted checks or planned
inside the coding worker as `python_compileall` / focused `pytest`; shell
commands, package installation, adapter delivery, and raw filesystem paths stay
out of the L2d contract.

Prompt-safe capability projection hides `handler_id`, adapter ids, raw channel
ids, credentials, collection names, and database internals.

## Background Work Extension Boundary

The current coding-agent worker is exposed through the accepted delayed-work
capability for bounded coding and repository-analysis tasks. Future delayed
capabilities such as a complex resolver must be added as reviewed first-class
capability contracts before they can become live background work. The
action-spec layer owns the model-facing capability name, semantic purpose,
visibility, source refs, trusted target binding, validation, idempotency
material, and prompt-safe result projection. It does not own worker
implementation details.

The extension sequence is:

```text
L2d semantic capability
  -> deterministic action-spec materialization
  -> accepted-task lifecycle and duplicate rejection
  -> internal background_work_request or deterministic requested_worker handoff
  -> worker-owned execution
  -> tool_result or durable scheduled follow-up
```

L2d must not select worker-local task types, tool arguments, filesystem paths,
shell commands, resolver internals, queue ids, leases, retry policy, adapter
delivery targets, or final artifact formatting. If a worker needs those
details, the worker or its deterministic handler owns them after accepted-task
validation. L3/dialog still owns visible wording for acknowledgements,
progress reports, failures, and completed results.

Adding a future background-work capability must document its capability row,
accepted-task identity fields, worker input/output contract, permission and
side-effect policy, result-delivery policy, failure behavior, and verification
gates. The shared queue mechanics are reusable; the semantic ownership and
permission contract are not implicit.

## Target Binding

Targetful actions are resolved by deterministic code from trusted episode,
trigger-source, RAG, and repository context.

For `memory_lifecycle_update`, L2d may only request a specialist review. The
target is a cognitive episode owned by the memory lifecycle specialist. It must
not include `unit_id`, collection names, target aliases, or lifecycle decisions.

The memory lifecycle specialist receives prompt-safe aliases such as
`commitment_1`, chooses `fulfilled`, `abandoned`, `obsolete`, or `deferred`
only when the evidence is clear, and returns aliases plus prompt-safe content
anchors. Deterministic code validates the alias, resolves it to the trusted
memory unit id, and materializes `apply_memory_lifecycle_update` for execution.

No promise or commitment is retired because it is old, overdue, keyword-matched,
or visually stale. The specialist must semantically judge the lifecycle change,
and the repository must validate the resolved target.

## Attempt Ledger

Action attempts reuse `self_cognition_action_attempts`. New rows may carry
generic action-attempt metadata, while old send-message rows must remain
readable for duplicate suppression and audit compatibility.

Do not add a second action ledger collection without a separate approved plan.

Resolver HIL and approval waits also reuse this ledger. They are deterministic
pending state, not action specs and not adapter delivery:

- `resolver_pending_hil` stores a prompt-safe clarification question selected
  by cognition.
- `resolver_pending_approval` stores a prompt-safe approval summary for a
  side effect that has not been executed.

Follow-up user turns close, approve, reject, or supersede these rows only when
L2d emits a semantic pending-resolution decision. The LLM-facing prompt does
not expose pending row ids; deterministic code binds the single active pending
row into `resolver_pending_resolution` after L2d chooses `decision` and
`reason`. Deterministic code must not infer approval from keywords or execute
the prepared side effect inside the resolver stage.

## Consolidation

The consolidator receives prompt-safe episode-trace projection only. It may
learn from visible text, private action outcomes, calendar-triggered action
outcomes, and private finalization, but it must not select actions, execute
actions, call the dispatcher, create calendar runs, or trigger cognition.

`final_dialog` is represented as one text `SurfaceOutputV1`; it is no longer
the only possible evidence that a turn has consolidatable output.

## Deferred Tools

The next stage may add capabilities only through a reviewed plan that names
owner, handler, permissions, continuation behavior, trace output, tests, and
latency budget.

Reserved next-stage candidates:

- `schedule_self_check`: orchestrator-owned request to create a future
  self-cognition episode through calendar/orchestrator mechanics.
- `web_research`: retrieval-owned research action that must return evidence,
  not final dialog.
- notes/open-loop tools: memory or open-loop owner actions for durable internal
  notes and closure, not arbitrary database writes.

Image generation is also a future surface handler candidate. L3 image can
produce visual prompts/directives today, but this module does not call an
external image generation service in the current implementation.

## Forbidden Paths

- Do not let L2a or L2c emit action specs.
- Do not expose raw action envelopes or deterministic ids in L2d prompts.
- Do not make `action_spec` a second dispatcher.
- Do not let tool handlers call cognition directly.
- Do not let the consolidator execute, dispatch, schedule, or select actions.
- Do not retire promises by deterministic age cleanup.
- Do not add arbitrary shell, HTTP, file, adapter, or MongoDB tools.
- Do not let background-work workers deliver adapter text directly.
