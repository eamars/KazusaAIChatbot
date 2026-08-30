# Subagent and Semantic Interfaces

This guide records the current Plan 3 ownership boundary. The production
route is DSH-only for multi-step task execution. LLM stages select meaning
and goals; deterministic code validates schemas, limits, authority,
persistence, leases, CAS, and delivery.

## Retained evidence owners

RAG3/local-context remains the owner of ordinary conversation, memory,
people, calendar, and approved web evidence. Its prewarm/cache owner remains
in place. A task may use the DSH semantic catalog to obtain bounded evidence,
but raw retrieval payloads do not become persona stance or visible wording.

Native DSH Standard tools and the fourteen kazusa_* semantic rows are
forwarded by the authenticated gateway. The sole additive row is
kazusa_inspect_public_media; its HTTP(S)-only safe fetch, DNS/redirect
checks, 15-second timeout, 6 MiB body limit, image decode, and bounded
dimensions are documented in the gateway ICD.

## Task handoff

The cognition-to-task contract is task_resolution_request. Direct
background admission returns transient model-hidden
TaskResolutionAdmissionV1 with exactly schema_version, accepted_task_id,
background_work_job_id, and task_session_id. This is an observation, not a
checkpoint or authority.

After claim, DSH binds the accepted task with dsh_task_binding.v1 in
dsh_task_bindings; operation_generation and revision CAS fence recovery.
Only a committed checkpoint may produce deferred
TaskResolutionResultV1 with DshResolutionRefV1. The worker payload is
task_orchestrator_worker_payload.v2, using open_dsh_resolution or
continue_dsh_resolution.

## Accepted-task controls

accepted_task_control.v1 exposes only continue, summarize, and cancel.
Status is a read-only projection. Controls reference the same opaque task and
session, obtain fresh DSH authority, and return through normal cognition,
dialog, dispatcher, and adapter delivery. Queue leases, raw worker payloads,
credentials, filesystem paths, and adapter objects never enter an LLM
prompt.

## Readiness and drain

Brain readiness is authenticated GET /runtime/dsh/health with configured,
durable_store, and cognition_judge. Sidecar system.health must report
matching route, Standard, semantic-worker, web, Brain, catalog, policy,
workspace, profile, release, and store values.

The read-only drain audit is:

~~~powershell
venv\Scripts\python scripts/check_dsh_plan3_drain.py --legacy-coding-workspace-root <abs-root> --format json
~~~

It reports the five governed categories and performs no writes. Historical
records remain historical evidence; no compatibility interpreter is added.
