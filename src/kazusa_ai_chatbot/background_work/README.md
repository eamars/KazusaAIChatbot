# Background Work ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.background_work`
- Source of truth: v2 job models, worker runtime, and focused job/delivery tests
- Document status: current durable-worker contract

## Purpose

`background_work` owns reviewed durable jobs, lease recovery, and result-ready
handoff. It stores only `background_work_job.v2` documents and does not expose
queue mechanics to cognition or dialog.

## Boundary

The package dispatches only reviewed worker payloads. Task resolution owns
specialist selection; accepted-task state owns user-facing lifecycle; dialog
and the dispatcher own visible delivery.

## Public Interfaces

The reviewed queue API persists one v2 job, the runtime claims jobs under a
lease, and the worker loop completes or fails them through accepted-task state.
Callers do not import worker internals or dynamically register workers.

## Workers

The runtime recognizes exactly two reviewed worker names:

| Worker | Owner | Purpose |
| --- | --- | --- |
| `task_orchestrator` | task resolution | Resumes one persisted task-resolution checkpoint or one bound coding-run continuation. |
| `future_speak` | scheduling | Schedules a deterministic future cognition trigger from an exact local time and semantic objective. |

There is no generic router, no worker-discovery registry, and no direct generic
coding or text-artifact worker path. The task orchestrator chooses at most one
of the four declared specialists for each dispatch: local context, public
research, coding, or text/computation.

## Input And Output Contracts

`task_orchestrator_worker_payload.v1` accepts exactly one reviewed operation:

- `resume_task_resolution`, carrying a persisted
  `task_resolution_checkpoint.v1`; or
- `continue_bound_coding_run`, carrying a validated frozen public coding-run
  continuation request.

The bound coding operation is closed to revision, summary, status, approved
verification, blocker response, and cancellation. It retains trusted approval
evidence and cannot create a new coding run through the background queue.

`future_speak` retains its deterministic scheduling payload and does not enter
the task-resolution specialist loop.

## Runtime Flow

Claim retries resume the stored checkpoint and preserve its dispatch,
orchestrator-call, and route-correction counters. The worker checkpoints the
raw-call count before selection, then persists `pending_dispatch` as selected
and started before a handler begins; completed dispatches clear it and append
one trace row without double-counting. A recovered started dispatch becomes an
at-most-once unavailable result and is never invoked again. Terminal resolved
and evidence-bearing partial results become accepted-task result-ready state;
limitations remain prompt-safe.

Workers do not call shared cognition or adapters directly. Delivery uses the
accepted-task result source, normal cognition, dispatcher validation, and the
usual adapter path. The job's original `source_message_id` is carried
separately from the synthetic `tool-result:<task_id>` episode identity. At
accepted-task result delivery, the service resolves the original user row in
`conversation_history` and selects `reply_to_msg_id` from that source id only
when the durable server `received_at` age strictly exceeds 120 seconds or an
intervening same-channel user receipt exists before the delivery cutoff. The
synthetic tool-result identity remains provenance only and is never passed as
a reply target.

## Persistence

The reviewed maintenance command is:

```powershell
venv\Scripts\python.exe scripts\clear_background_task_history.py
venv\Scripts\python.exe scripts\clear_background_task_history.py --execute --confirm DELETE_BACKGROUND_WORK_JOBS_AND_ACCEPTED_TASKS
```

It counts or clears only `background_work_jobs` and `accepted_tasks`; it emits
counts only and verifies both are empty after execution. Run it only after
stopping every process that can write either collection.

## Failure Behavior

Lease loss, malformed payloads, and worker failures resolve through the v2 job
and accepted-task failure state. Queue retries preserve the persisted
task-resolution checkpoint and never restart its semantic dispatch budget.

## Testing Contract

Run background-work job, delivery, future-speak, task-orchestrator resume, and
accepted-task lifecycle suites. Execute the destructive maintenance boundary
only in its explicit isolated live-database rehearsal or approved offline
cutover.

## Forbidden Paths

Do not add a generic router, dynamic worker discovery, direct adapter delivery,
or direct shared-cognition invocation. Do not delete collections outside the
approved accepted-task and background-job history boundary.
