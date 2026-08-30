# Agentic Resolver Architecture

Status: current implementation contract.

## System boundary

The resolver is a process-owned control plane, not a second persona or
dialog engine. The Brain provides a typed task request and receives a typed
result. The resolver validates DSH intake, operation identity, thread and
segment lineage, authority, lease, catalog digest, replay, and checkpoint
state. An authenticated DSH Standard sidecar owns session events, native
tools, semantic tools, and terminal receipts. Normal cognition, dialog, and
dispatcher delivery remain Brain owners.

~~~text
Brain task_resolution_request
  -> accepted task and background job
  -> transient TaskResolutionAdmissionV1
  -> dsh_task_binding.v1 / dsh_task_bindings
  -> claim-time authority and operation_generation
  -> DSH Standard sidecar
  -> checkpoint or terminal exhaust
  -> TaskResolutionResultV1
  -> normal Brain result-ready delivery
~~~

The same AgenticResolverRuntime serves foreground and direct-background
execution. Deterministic code owns all mechanical boundaries; the DSH
semantic worker owns task judgment and tool choice under the pinned Standard
profile. No semantic decision is reconstructed by local keyword logic.

## Admission, checkpoint, and binding

TaskResolutionAdmissionV1 is transient and model-hidden. Its exact keys are
schema_version, accepted_task_id, background_work_job_id, and
task_session_id. It is an observation of admission only; it has no authority,
no checkpoint, and no deferred result semantics.

The only deferred result is TaskResolutionResultV1 with status deferred after
a checkpoint has been committed. It contains one DshResolutionRefV1. The
worker payload is task_orchestrator_worker_payload.v2, with
open_dsh_resolution and continue_dsh_resolution operations.

Claim-time code creates or verifies dsh_task_binding.v1 in
dsh_task_bindings, then mints fresh authority. operation_generation, lease
epoch, and document revision CAS fence every mutation. A process or sidecar
restart resumes from the durable binding/checkpoint; stale authority,
duplicate operations, malformed responses, and incompatible generations fail
closed. A recovered started dispatch is settled according to committed DSH
state rather than invoked twice.

## Catalog and segment epochs

The model-visible semantic catalog contains exactly fourteen rows in canonical
order:

1. kazusa_search_conversation_history
2. kazusa_read_conversation_entries
3. kazusa_summarize_conversation_participants
4. kazusa_search_memories
5. kazusa_read_memories
6. kazusa_remember_information
7. kazusa_revise_memory
8. kazusa_change_memory_lifecycle
9. kazusa_find_people_by_name
10. kazusa_read_person_profiles
11. kazusa_recall_active_context
12. kazusa_read_calendar_context
13. kazusa_inspect_attached_media

The fourteenth row, kazusa_inspect_public_media, accepts one HTTP(S) image
URL and a question. URL credentials/fragments are rejected; DNS results in
private, loopback, link-local, multicast, reserved, and unspecified ranges
are rejected; redirects are rechecked with a maximum of 3. The timeout is
15 seconds and the body limit is 6 MiB. MIME and magic bytes must identify
PNG, JPEG, GIF, or WebP, Pillow must decode, and dimensions must be 1..8192.
Evidence source is dsh_public_media; raw bytes and base64 are excluded.

Any catalog schema change produces a new semantic catalog digest. Terminal or
checkpointed V2 threads with no open interaction rotate to a fresh segment
and receive the current digest. Authority and grants bound to an earlier
digest fail closed.

## Readiness and accepted-task controls

Brain exposes authenticated GET /runtime/dsh/health with configured,
durable_store, and cognition_judge. Sidecar system.health is ready only when
route, Standard, semantic-worker, web, Brain, catalog, policy, workspace,
profile, release, and store values match. Any mismatch blocks task admission.

accepted_task_control.v1 exposes continue, summarize, and cancel for one
opaque accepted-task/session binding. Status inspection is read-only. Each
control obtains fresh authority and current generation, then returns through
normal cognition and delivery. Queue leases, raw payloads, credentials,
filesystem paths, and adapter objects remain outside the model context.

## Retained owners and operational gate

RAG3/local-context and its prewarm/cache owner remain the ordinary chat
evidence route. Dialog owns visible wording; consolidation, scheduler,
reflection, and future_speak remain outside the live task session.

The read-only drain command is:

~~~powershell
venv\Scripts\python scripts/check_dsh_legacy_drain.py --legacy-coding-workspace-root <abs-root> --format json
~~~

It reports five governed categories and performs no writes. Runtime
configuration is governed by the exact current field inventory.
