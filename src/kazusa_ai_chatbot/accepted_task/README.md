# Accepted Task ICD

accepted_task owns the user-scoped durable lifecycle for work accepted for
later completion. It stores accepted_task.v2 state and exposes only
prompt-safe projections. DSH owns the task session and its checkpoint.

## Boundary

Admission may be foreground or direct background. The model-hidden
TaskResolutionAdmissionV1 reports schema_version, accepted_task_id,
background_work_job_id, and task_session_id only while the admission is
being acknowledged. It is transient and carries no authority or checkpoint
reference.

After claim and binding, a committed
TaskResolutionResultV1(status="deferred") carries a
DshResolutionRefV1. The binding is dsh_task_binding.v1 in
dsh_task_bindings, fenced by operation_generation and revision CAS.
Recovery reuses the bound DSH thread/segment only after fresh claim-time
authority is minted.

## Public controls

accepted_task_control.v1 is the sole accepted-task control surface. Its
operations are continue, summarize, and cancel; status inspection is
read-only. Every control is scoped to the opaque accepted-task/session
binding, creates an idempotent DSH operation, and returns through normal
cognition, dialog, and dispatcher delivery.

Queue ids, leases, credentials, adapter identifiers, raw evidence,
filesystem paths, and worker payloads stay out of prompts and visible
results. Failed enqueue, lease loss, stale generations, and invalid result
projections fail closed with typed state.

## Retention and testing

The accepted-task lifecycle is retained with its reviewed background-job
partner. Historical terminal records remain immutable evidence; no converter
interprets them. Use the deterministic accepted-task, task-resolution,
background-work, and delivery tests through venv\Scripts\python. Live DB
rehearsals use an isolated test database.
