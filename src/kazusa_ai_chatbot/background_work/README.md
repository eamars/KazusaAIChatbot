# Background Work

## Document Control

This ICD defines the generic background-work subsystem boundary for new
asynchronous jobs.

## Purpose

`background_work` owns the internal asynchronous executor behind accepted
delayed work. The model-facing lifecycle is `accepted_task`; this package keeps
queue rows, routing, worker-local task classification, generation, scheduling,
and delivery bookkeeping out of cognition and dialog prompts.

## Boundary

- L2d sees accepted-task affordances. Deterministic materialization converts a
  new `accepted_task_request` into the internal executable
  `background_work_request` only after source validation and duplicate
  rejection.
- L2d may request `future_speak` for accepted future reminders or delayed
  follow-up messages; deterministic action execution binds it to an accepted
  task and then to the `future_speak` worker.
- The background-work router emits only `action`, `worker`, and `reason`.
- Worker subagents own worker-local semantic parameters and artifacts.
- L3/dialog remain the only visible wording owners and receive accepted-task
  state, not queue or worker internals.
- Workers never send adapter text, call cognition directly, run shell or
  filesystem work, install packages, process attachments, or mutate
  persistence outside the public worker result contract.

## Public Extension Contract

New asynchronous capabilities must enter background work through one of two
stable routes:

1. Model-facing accepted work: L2d selects a semantic accepted-task capability.
   Deterministic action-spec materialization validates trusted source scope,
   rejects or reuses duplicate active work, persists an accepted-task row, and
   creates one internal `background_work_request`.
2. Internal executor work: trusted code creates a `background_work_request`
   directly only when the caller already owns the accepted-task or legacy
   lifecycle boundary.

The queue request contract is intentionally narrow. Callers provide a short
semantic `task_brief`, optional prompt-safe `source_context`, trusted source
and requester scope, `requested_delivery="send_result_when_done"`,
`max_output_chars`, and the storage timestamp. `accepted_task_id` and
`task_identity_key` are lifecycle audit fields, not worker routing inputs.
`requested_worker` plus `worker_payload` are allowed only for deterministic
handoffs where an upstream handler already validated the worker-specific
contract, such as `future_speak`. Generic delayed work should leave the worker
unset so the background-work router can choose a worker from prompt-safe worker
descriptions.

A worker is registered through `subagent.discover_background_work_workers()`.
Each worker module must expose:

```python
WORKER: str
DESCRIPTION: str

async def execute(
    decision: BackgroundWorkWorkerDecision,
    *,
    max_output_chars: int,
) -> BackgroundWorkResult: ...
```

`DESCRIPTION` is router-facing and must describe only the worker's semantic
capability. It must not mention adapter ids, persistence fields, filesystem
paths, shell commands, credentials, or hidden operational options.
`execute()` receives the selected route decision, trusted `task_brief`,
optional source summary, and optional deterministic worker payload. It returns
one `BackgroundWorkResult` with `status`, `worker`, bounded `artifact_text`,
`failure_summary`, `result_summary`, and audit-only `worker_metadata`.

Worker implementations own their local task classification, argument
extraction, validation, execution, and refusal. The generic router chooses the
worker only; it must not infer low-level parameters for a coding agent,
complex resolver, web task, filesystem task, or any future domain. If a future
worker needs domain-specific parameters, that worker must derive them inside
its own bounded prompt or deterministic validator after routing, or receive
them through a deterministic `requested_worker` handoff whose action-spec
handler already validated the fields.

Result handoff is also fixed. Text or artifact-producing workers complete the
job and allow `accepted_task_result_ready` cognition to decide whether and how
to speak. Scheduled-contact workers such as `future_speak` may set
`worker_metadata.skip_result_delivery=true` only when they create another
durable follow-up path, for example a calendar `future_cognition` run. Workers
must not dispatch messages, call the shared cognition graph directly, write
conversation rows, or generate prewritten proactive text.

## Future Worker Eligibility

The interface can support additional workers such as a complex resolver, but
those integrations require their own reviewed action capability and worker
contract before enablement. A valid future worker must define:

- semantic ownership: the class of delayed work it owns;
- model-facing entrypoint: accepted-task capability or trusted internal
  handoff;
- worker input contract: prompt-safe task brief, optional source summary, and
  any deterministic payload fields;
- output contract: bounded artifact text, result summary, failure summary, and
  audit-only metadata;
- refusal conditions: unsupported task types, missing permissions, unsafe
  side effects, or unavailable resources;
- duplicate identity material: fields used by accepted-task duplicate
  rejection before any job is queued;
- side-effect policy: whether it is read-only, writes internal artifacts, uses
  external tools, or needs explicit user permission;
- delivery policy: result-ready cognition, calendar follow-up, or no visible
  delivery;
- verification: deterministic contract tests, integration tests, and one
  real LLM test when prompts or model-facing routing are involved.

Coding-agent work enters through the registered `coding_agent` worker. That
worker delegates read-versus-write selection to the coding-agent supervisor and
returns either a code-reading answer or a code-writing proposal artifact. Any
worker that can run shell commands, edit files, install packages, browse the
web, call external tools, or apply patches needs a separate permission and
sandbox contract before it is added to the registry. The background-work queue
supplies lifecycle and handoff mechanics; it is not a general tool-permission
system.

## Workers

`subagent.text_artifact` handles bounded text artifacts. It has two separate
LLM stages:

1. Task router: chooses `coding_snippet`, `text_rewrite`, `summary`,
   `unsupported`, or `needs_user_input`.
2. Generator: produces one bounded text artifact or a failure summary.

The generic queue and router do not expose those worker-local task labels or
worker-facing task rewrites.

`subagent.future_speak` is deterministic. It receives an exact local trigger
time and a semantic continuation objective, then schedules a
`future_cognition` calendar run. It does not store final user-facing text.
The due self-cognition cycle decides again how to speak.

`subagent.coding_agent` adapts the public standalone coding-agent
`handle_background_coding_task(...)` interface. It handles accepted coding
tasks, requires `CODING_AGENT_WORKSPACE_ROOT` at execution time, and returns
bounded artifact text plus sanitized repository, evidence, and proposal
metadata. It may produce proposal artifacts, but it does not apply patches,
run project commands, install packages, or deliver adapter text.

Completed `future_speak` jobs suppress immediate background-result delivery,
because the user-facing follow-up belongs to the scheduled self-cognition
slot. This prevents the scheduling bookkeeping from creating a duplicate
message before the reminder is due.

## Persistence

Raw MongoDB access lives in `kazusa_ai_chatbot.db.background_work_jobs`.
Callers use the public queue/runtime exports from `kazusa_ai_chatbot.background_work`.
Jobs move through queued, in-progress, completed or failed, delivery in
progress, delivered, and delivery failed states.
Accepted-task ids and identity keys are copied into new internal job rows for
audit and lifecycle synchronization only. Prompt-facing progress and result
state comes from `accepted_task`, not from job ids or queue state.
