# Accepted Task ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.accepted_task`
- Source of truth: lifecycle/model contracts and focused accepted-task tests
- Document status: current v2 durable-task contract

## Purpose

`accepted_task` owns the user-scoped durable lifecycle for work accepted for
later completion. The runtime stores only `accepted_task.v2` rows and rejects
earlier document versions.

## Boundary

An accepted task is created only after deterministic validation at one of these
boundaries:

- task-resolution inline work exhausts its configured budget and promotes the
  same checkpoint to `task_orchestrator`;
- `future_speak` schedules a deterministic follow-up; or
- `accepted_coding_task_request` continues an existing bound coding run.

Task identity derives from the validated semantic objective plus trusted
requester and conversation scope. Persona rationale, route language, and raw
worker metadata are not identity inputs.

## Public Interfaces

- `create_or_return_active_accepted_task(...)` creates or reuses scoped active
  state.
- `check_accepted_task_status(...)` reads prompt-safe scoped task state.
- `load_open_coding_run_contexts_for_scope(...)` projects sanitized bound-run
  continuation context.
- Lifecycle transitions mark enqueue, running, result-ready, delivery, and
  failure state only after their reviewed counterpart succeeds.

## Persistence

Active state is created atomically with duplicate rejection and becomes pending
only after its reviewed background job exists. Task-orchestrator results map as
follows:

| Resolution result | Accepted-task completion status | Prompt-safe delivery data |
| --- | --- | --- |
| `resolved` | `resolved` | summary and validated evidence projection |
| `partial` | `partial` | summary, evidence, and remaining limitations |
| `needs_user_input`, `approval_required`, `unavailable`, `failed` | `failed` | exact result kind and remaining limitations |

The only collection is `accepted_tasks`. The offline maintenance procedure may
clear it only together with `background_work_jobs`, using the reviewed
two-collection CLI and explicit confirmation.

## Failure Behavior

Enqueue failures retain failed accepted-task state without creating an
unreviewed retry path. Result-ready work re-enters normal cognition and
dispatcher validation; a worker neither writes final dialogue nor sends an
adapter message.

Coding continuations retain only sanitized public run context:
`coding_run:<run_id>`, status, summary, limitations, allowed next actions, and
whether a follow-up is open. Mutation, cancellation, blocker response, and
verification remain governed by the frozen public coding-run lifecycle.

## Testing Contract

Run focused accepted-task lifecycle, background-work job, delivery, and
task-resolution promotion/resume tests with the project virtual environment.
Live database cutover rehearsal is explicit and uses only the isolated test
database.

## Forbidden Paths

Do not persist raw worker payloads, adapter delivery targets, credentials,
queue leases, or final dialogue in accepted-task prompt projections. Do not
clear coding runs, calendar rows, conversations, memories, or any collection
outside the approved two-collection offline maintenance boundary.
