# DSH Task Resolution

kazusa_ai_chatbot.task_resolution owns the single DSH task edge.
Cognition supplies a typed semantic task request when its evidence and
judgment call for work. DSH owns admission, execution, checkpointing,
recovery, and terminal projection. Dialog and the dispatcher own visible
wording and delivery.

## Admission and result contracts

TaskResolutionAdmissionV1 is transient and model-hidden. It contains
exactly schema_version, accepted_task_id, background_work_job_id, and
task_session_id. It is an observation of accepted work, not a deferred
result, not a durable authority, and not a reference to a checkpoint.

Only a committed checkpoint may produce
TaskResolutionResultV1(status="deferred"). That result carries one
DshResolutionRefV1; it never carries worker metadata or a legacy executor
reference. Terminal and evidence-bearing partial results remain typed and
prompt-safe.

## Durable binding and recovery

The task edge creates the accepted-task and background-job records in their
reviewed order. Claim-time execution creates dsh_task_binding.v1 in the
dsh_task_bindings collection, then mints fresh DSH authority for the bound
thread and segment. operation_generation and document revision CAS make
claim, checkpoint, continuation, and terminal delivery idempotent. Lease
loss, process restart, sidecar loss, stale catalog authority, and malformed
responses fail closed and recover only from the binding/checkpoint.

The worker payload is task_orchestrator_worker_payload.v2. Its operations
are open_dsh_resolution and continue_dsh_resolution. The same shared
runtime handles foreground work and direct background work; direct admission
returns the transient observation before the checkpoint exists.

## Accepted-task controls

The prompt-safe capability is accepted_task_control.v1. It allows only
continue, summarize, and cancel against the same opaque accepted task
and session. Status inspection is read-only. Controls create a new DSH
operation with fresh authority and the current binding generation; they do
not expose queue leases, raw evidence, credentials, adapters, or paths.

## Limits and ownership

Deterministic code enforces schema, budgets, idempotency, persistence order,
lease fencing, CAS, and delivery boundaries. The DSH semantic worker owns
semantic tool choice and task judgment under the pinned Standard profile.
This service is also the sole owner of the foreground inline timeout. It either
returns a terminal result or commits and projects the same session's deferred
checkpoint before control returns to cognition.
If cognition retries after an internal DSH interaction changes its persisted
state, a completed inline task returns its validated stored result. The source
scope and goal continuation must match; rewording a request during that retry
does not recreate the binding or execute the completed task again.
If that task already checkpointed and acquired a background owner, replay
returns the same checkpoint and promotion preserves the existing accepted task
and job. The worker remains responsible for its eventual completion delivery.
Caller cancellation before durable background attachment settles fenced
cancellation or an already committed terminal outcome, then joins owned local
work. An uncertain remote outcome remains a typed fault in the binding.
Cancellation during deferred promotion inspects the durable attachments:
committed accepted-task/job ownership survives; partial promotion is fenced
through the existing enqueue-failure and binding-fault boundaries.
RAG3/local-context and its prewarm owner remain the evidence path for ordinary
chat context; they are not replaced by task-resolution documentation.

No second resolver graph, specialist vocabulary, generic background router,
or direct adapter delivery may be introduced. A result-ready task returns
through normal cognition, dialog, and dispatcher delivery.

## Verification

Begin with the existing real-model foreground behavior case through the Brain,
then inspect deferred delivery and internal judgment separately. Preserve all
live LLM cases and inspect their saved evidence. The
[runtime completion record](../../../development_plans/archive/completed/bugfix/dsh_runtime_completion_plan_2026-09-05.md)
records the observed runtime repairs and recovery diagnostics. Dedicated
task-resolution suites and ownership-manifest gates are retired.

The permanent non-LLM integration probes are:

```powershell
venv\Scripts\python experiments\dsh_runtime_probe.py sidecar-lifecycle --artifact-dir <new-directory>
venv\Scripts\python experiments\dsh_runtime_probe.py brain-task-lifecycle --artifact-dir <new-directory>
venv\Scripts\python experiments\dsh_runtime_probe.py brain-task-promotion-replay --artifact-dir <new-directory>
venv\Scripts\python experiments\dsh_runtime_probe.py transport-loss --artifact-dir <new-directory>
```

Each writes `dsh_runtime_probe_result.v1`. The Mongo probes create and drop
only uniquely named guarded test databases. The promotion replay probe uses
real checkpointing, accepted-task persistence, and queue insertion, then
verifies that a paraphrased cognition retry preserves all durable owners.
