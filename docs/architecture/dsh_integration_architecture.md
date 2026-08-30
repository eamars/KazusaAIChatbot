# DSH Integration Architecture

Status: current production integration contract. The pinned DSH Standard
profile, RPC/intake epochs, policy, store, native-tool ownership, interaction
authority, and task edge form one production runtime.

## End-to-end route

~~~text
adapter/debug client
  -> authenticated Brain intake
  -> cognition task_resolution_request
  -> accepted_task.v2 and reviewed background_work_job.v2
  -> transient TaskResolutionAdmissionV1
  -> dsh_task_binding.v1 / dsh_task_bindings
  -> claim-time operation_generation and fresh authority
  -> DSH Standard sidecar
  -> native tools + semantic catalog
  -> checkpoint or terminal exhaust
  -> TaskResolutionResultV1
  -> Brain observation/result-ready projection
  -> normal cognition/dialog/dispatcher/adapter delivery
~~~

The runtime is shared for foreground and direct-background work. The sidecar
owns DSH sessions, event persistence, receipts, native execution, and
semantic forwarding. Brain owns interaction judgment and visible surfaces.
Deterministic services own binding, CAS, limits, leases, idempotency, and
delivery ordering.

## DSH task contracts

TaskResolutionAdmissionV1 has exactly four keys:
schema_version, accepted_task_id, background_work_job_id, and task_session_id.
It is transient and model-hidden, with no authority or checkpoint reference.
Only a committed checkpoint may produce deferred
TaskResolutionResultV1 with DshResolutionRefV1.

The worker payload is task_orchestrator_worker_payload.v2. Its task
operations are open_dsh_resolution and continue_dsh_resolution. Claim-time
code writes or verifies dsh_task_binding.v1, mints current authority, and
advances operation_generation under revision CAS. Restart and lease recovery
resume from durable state; stale generation, stale authority, malformed
payload, duplicate operation, and sidecar fault fail closed.

accepted_task_control.v1 is the accepted-task control surface. continue,
summarize, and cancel are the only mutating operations; status is read-only.
Controls remain on the same opaque task/session binding and return via normal
Brain cognition and delivery.

## Fourteen-row semantic catalog

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
14. kazusa_inspect_public_media

kazusa_inspect_public_media is a storage-independent public-media tool. It
accepts HTTP(S) only, rejects credentials/fragments and special-use DNS
addresses, rechecks every redirect with a maximum of 3, and enforces a
15-second timeout plus a 6 MiB body cap. MIME/magic validation accepts only
PNG, JPEG, GIF, or WebP; Pillow decoding and dimensions 1..8192 are
required. Source dsh_public_media is retained as evidence provenance; raw
bytes and base64 are excluded from all model-facing contracts.

## Digest and segment rotation

Any catalog schema change produces a new `semantic_catalog_digest`. A terminal
or checkpointed V2 thread with no open interaction rotates to a fresh segment
carrying the current digest. Authority and one-shot grants bound to an earlier
digest fail closed after rotation.

## Readiness and recovery

Brain exposes authenticated GET /runtime/dsh/health with
configured, durable_store, and cognition_judge. The sidecar's authenticated
system.health is ready only when route, Standard, semantic-worker, web,
Brain, catalog, policy, workspace, profile, release, and store values match.
Readiness gates admission and recovery.

On sidecar loss, the durable task binding and checkpoint remain the source of
truth. Claim-time fresh authority and generation checks prevent duplicate
execution. Result-ready state returns through normal cognition/dialog
delivery, never directly from a worker to an adapter.

## Retained owners and drain

RAG3/local-context, its evidence leaves and prewarm/cache owner, dialog,
memory, consolidation, scheduler, reflection, and future_speak retain their
existing responsibilities. DSH supplies task execution and bounded semantic
evidence; it does not become a persona or final-response layer.

The read-only operational audit is:

~~~powershell
venv\Scripts\python scripts/check_dsh_legacy_drain.py --legacy-coding-workspace-root <abs-root> --format json
~~~

It counts five governed legacy categories and writes nothing.
