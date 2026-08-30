# Background Work ICD

background_work owns reviewed durable jobs, lease recovery, and the
result-ready handoff. Its current task payload is
task_orchestrator_worker_payload.v2; it never exposes queue mechanics to
cognition or dialog.

## Reviewed operations

| Operation | Owner | Purpose |
| --- | --- | --- |
| open_dsh_resolution | DSH task resolution | Opens a bound DSH session from an accepted task. |
| continue_dsh_resolution | DSH task resolution | Continues a committed DSH checkpoint with fresh authority. |
| future_speak | scheduler | Schedules a deterministic future cognition trigger. |

Direct background admission first returns transient
TaskResolutionAdmissionV1 with only schema_version, accepted_task_id,
background_work_job_id, and task_session_id. The worker later claims the job,
creates or verifies dsh_task_binding.v1 in dsh_task_bindings, and commits the
checkpoint before a deferred TaskResolutionResultV1 is exposed. No authority
or checkpoint is hidden in the admission observation.

## Recovery and delivery

operation_generation, revision CAS, operation idempotency, and lease fencing
protect every claim, checkpoint, continuation, and terminal result. Lease
loss, restart, sidecar fault, stale catalog digest, malformed payload, and
worker failure fail closed or recover from the durable binding. A recovered
started operation is settled according to its committed DSH state; it is not
duplicated.

Workers do not call cognition, adapters, or dispatcher delivery directly.
Accepted-task result-ready state re-enters normal cognition and dialog. The
original source message remains provenance for reply lineage and is resolved
by the service's durable delivery rules.

## Maintenance and testing

The reviewed maintenance boundary counts or clears only the approved
background-job and accepted-task collections after every writer is stopped.
Use venv\Scripts\python for job, lease, task-resolution, accepted-task, and
delivery tests. Live database maintenance is a separately authorized,
isolated rehearsal.
