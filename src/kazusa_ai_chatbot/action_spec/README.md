# Action Spec ICD

action_spec owns deterministic validation and execution tracing for
cognition-selected actions. Task semantic work belongs to the DSH task edge;
action specs do not select workers or interpret raw evidence.

## Current model-facing roster

| Capability | Owner | Contract |
| --- | --- | --- |
| speak | L3 surfaces | Selects a visible surface intent; L3 owns final wording. |
| memory_lifecycle_update | memory lifecycle | Requests review of an existing commitment lifecycle. |
| trigger_future_cognition | scheduler | Requests a future cognition cycle through the scheduler. |
| future_speak | background work | Schedules a deterministic future cognition trigger. |
| accepted_task_control | accepted task / DSH | Continues, summarizes, or cancels one scoped accepted task. |
| accepted_task_status_check | accepted task | Reads scoped task state without creating work. |

task_resolution_request is the typed cognition-to-DSH handoff, not a
generic action router. Deterministic resolver code owns its inline budget and
any durable promotion.

## Accepted-task control

The control input uses accepted_task_control.v1 and one of continue,
summarize, or cancel. It references only the prompt-safe accepted task and
session projection. The handler validates requester scope, binding
generation, current DSH authority, and idempotency before creating the DSH
operation. It never chooses a semantic worker or exposes queue, lease,
credential, path, or raw evidence details.

## Execution and failure

The evaluator validates one materialized action against the registry. The
executor invokes only the reviewed capability handler. Result-ready task
state goes through normal cognition, dialog, dispatcher, and adapter
boundaries; workers never send visible text directly. Invalid, stale, or
unavailable controls fail closed and remain typed.

Run action-spec, cognition action-planning, accepted-task, task-resolution,
and delivery tests with venv\Scripts\python.
