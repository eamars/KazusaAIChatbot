# Cognition Contracts — Plan 3 DSH

Status: current contract for the Brain-to-DSH task edge.

## Ownership

Cognition interprets authored input, evidence, mood, relationship, scene
pressure, and character judgment. It may emit task_resolution_request when
the character has a grounded reason to pursue bounded work. DSH owns semantic
task execution and typed progress. Deterministic code owns validation,
permission, limits, persistence, leases, CAS, and delivery mechanics. Dialog
owns final wording.

User input remains LLM-first. No pre-processing or post-processing keyword
classifier may convert an authored request into a task, permission, or
commitment. Prompt/schema output is the semantic authority; deterministic
code validates the closed contract and fails closed on structural errors.

## Task handoff contracts

The cognition-to-DSH request is task_resolution_request. Direct background
admission projects a transient, model-hidden TaskResolutionAdmissionV1 with
exactly schema_version, accepted_task_id, background_work_job_id, and
task_session_id. This observation carries no DSH authority and no
checkpoint reference.

Only a committed checkpoint can yield
TaskResolutionResultV1(status="deferred"), carrying a
DshResolutionRefV1. Terminal and partial outcomes are typed and
provenance-bearing. The model-visible result excludes queue ids, leases,
credentials, raw worker payloads, filesystem paths, and adapter objects.

The durable owner is dsh_task_binding.v1 in dsh_task_bindings. Claim-time
authority is fenced by operation_generation, lease epoch, and revision CAS.
The worker payload is task_orchestrator_worker_payload.v2 and its task
operations are open_dsh_resolution and continue_dsh_resolution. Recovery,
replay, sidecar loss, stale digest, and invalid shape fail closed or resume
from the committed binding.

## Action contracts

The model-facing action roster retains:

| Capability | Semantic owner | Visible result |
| --- | --- | --- |
| speak | L3/dialog | final wording selected by L3 |
| memory_lifecycle_update | memory lifecycle | validated lifecycle result |
| trigger_future_cognition | scheduler | scheduled future cognition |
| future_speak | scheduler/background | accepted future-speak task |
| accepted_task_control | accepted task/DSH | continue, summarize, or cancel |
| accepted_task_status_check | accepted task | read-only scoped status |

accepted_task_control.v1 addresses the same opaque accepted task and session.
Only continue, summarize, and cancel are valid operations. The handler
validates requester scope, current binding generation, fresh DSH authority,
and idempotency. It does not choose a semantic worker or expose queue state.
Status inspection never creates work.

## Result projection

Result-ready accepted-task state re-enters normal cognition, dialog,
dispatcher, and adapter delivery. Workers never write final dialogue or send
adapter messages. The result source retains trace and task provenance
separately from authored content. A deferred result is published only after
the DSH checkpoint and binding state are durable.

RAG3/local-context remains the ordinary evidence and prewarm owner. DSH
semantic evidence is bounded, provenance-bearing input; it is not persona
definition or final stance. Consolidation, scheduler, and reflection remain
outside the live response wording path.

## Catalog and safety

The DSH semantic catalog has exactly fourteen rows: the thirteen completed
Plan 2 rows plus the sole additive kazusa_inspect_public_media row. The
public-media boundary permits only HTTP(S), rechecks DNS and at most 3
redirects, enforces a 15-second timeout and 6 MiB body limit, accepts only
PNG/JPEG/GIF/WebP after MIME/magic and Pillow validation, requires dimensions
1..8192, and emits source dsh_public_media without raw bytes/base64.

The new row changes the catalog digest and therefore the segment epoch.
Eligible terminal/checkpointed V2 threads rotate to a new segment when no
interaction is open; old authority and grants fail closed. Open
pre-cutover interactions drain before rotation.

## Readiness and audit

Brain GET /runtime/dsh/health reports configured, durable_store, and
cognition_judge. Authenticated sidecar system.health must agree on route,
Standard, semantic-worker, web, Brain, catalog, policy, workspace, profile,
release, and store. The read-only drain audit is:

~~~powershell
venv\Scripts\python scripts/check_dsh_plan3_drain.py --legacy-coding-workspace-root <abs-root> --format json
~~~

The audit counts five categories and performs no writes. The active Plan 3
ledger records exact command results and residual authorization boundaries.

## Cognition observation and self-cognition delivery

The canonical process-local observation carrier is
cognition_run_observation.v1. Producer validation completes before the Brain
projects the bounded record. It contains semantic status and approved
sections, never raw model output, embeddings, raw messages, message envelopes,
database identifiers, adapter identifiers, action parameters, handler
metadata, or worker error text.

selected self-cognition `speak` uses the same shared cognition/dialog/persistence path and the runtime adapter bridge. A selected speak surface must attempt
delivery after its source-window reservation; response mechanics cannot become
semantic evidence or an orientation field.
