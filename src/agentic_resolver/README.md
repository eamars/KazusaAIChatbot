# DSH V2 Resolution Control Plane

agentic_resolver is the DSH control plane. It is the only
task-resolution execution route: the Brain admits a semantic task, the
resolver binds it to a DSH resolution session, the authenticated sidecar
executes the pinned Standard profile and semantic catalog, and the Brain
projects the validated result into normal cognition and delivery.

## Runtime boundary

The process-owned runtime validates intake, operation identity, resolution
thread and segment lineage, activation/lease fencing, catalog authority,
replay, and terminal/checkpoint/fault/cancel outcomes. The sidecar owns DSH
session events, native Standard tools, checkpoints, and terminal receipts.
The Brain owns the interaction judge, user-facing surfaces, and delivery.

TaskResolutionAdmissionV1 is a transient, model-hidden admission
observation. Its exact fields are schema_version, accepted_task_id,
background_work_job_id, and task_session_id; it grants no authority and
is never a deferred result or a checkpoint reference. A committed deferred
result is TaskResolutionResultV1(status="deferred") and carries exactly one
DshResolutionRefV1. Claim-time binding uses dsh_task_binding.v1 in
dsh_task_bindings; operation_generation and document revision CAS fence
recovery and replay.

Foreground work and direct background admission use one shared runtime.
Background admission returns the transient observation before a durable
checkpoint exists. A worker later mints fresh DSH authority at claim time,
opens or continues the bound session, and commits the checkpoint before
publishing a deferred result. A lost sidecar, lease, or stale generation
fails closed and recovers from the durable binding; it never falls back to a
retired executor.

## Model-visible catalog

The description-free semantic catalog contains exactly fourteen rows in this
canonical order:

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

The public-media tool accepts one HTTP(S) image URL and a semantic question.
It rejects credentials or fragments, blocks DNS results in private,
loopback, link-local, multicast, reserved, and unspecified ranges, rechecks
every redirect (at most 3), and enforces a 15-second timeout and 6 MiB body
limit. MIME and magic bytes must identify PNG, JPEG, GIF, or WebP; Pillow
decoding and dimensions from 1 through 8192 are required. The result is
vision evidence with source dsh_public_media; raw bytes and base64 never
enter the model contract.

Any catalog schema change produces a new digest. A terminal or checkpointed
V2 thread without an open interaction rotates to a fresh segment with the
current digest. Authority and grants bound to an earlier digest fail closed.

## Readiness, controls, and operations

Brain readiness is authenticated GET /runtime/dsh/health and reports
configured, durable_store, and cognition_judge. The authenticated sidecar
system.health response is ready only when route, Standard, semantic-worker,
web, Brain, catalog, policy, workspace, profile, release, and store values
match. Startup and recovery remain fail-closed until every value agrees.

The accepted-task capability is accepted_task_control.v1 with only
continue, summarize, and cancel operations. It projects a prompt-safe
affordance for the same opaque task/session binding; it does not expose
queue ids, leases, credentials, filesystem paths, raw evidence, or worker
payloads. Result-ready state returns through normal cognition, dialog, and
dispatcher delivery.

The read-only drain audit is:

~~~powershell
venv\Scripts\python scripts/check_dsh_legacy_drain.py --legacy-coding-workspace-root <abs-root> --format json
~~~

It counts the five governed legacy categories and performs no writes. RAG3
local-context/evidence leaves, its prewarm owner, dialog, memory,
consolidation, scheduler, reflection, and future_speak remain their own
live owners. Only the task edge changes to DSH.

## Configuration and testing

The task route uses the six AGENTIC_RESOLVER_LLM_* settings plus the
KAZUSA_DSH_* sidecar, store, gateway, and Python-executable settings. The live
RAG3 route retains its planner, subagent, and web-provider settings.

Use venv\Scripts\python for deterministic tests. Exercise contract,
controller, process-recovery, task-resolution, binding, catalog, and
readiness tests before any live-DB or live-LLM rehearsal.

`experiments/dsh_runtime_probe.py` is the reusable black-box process owner for
sidecar lifecycle, guarded Brain/task persistence, and transport-loss probes.
It records the tested revision, child PIDs and exits, observations, artifacts,
and cleanup in `dsh_runtime_probe_result.v1`; a blocked prerequisite returns
exit code 2 instead of being reported as a pass.
