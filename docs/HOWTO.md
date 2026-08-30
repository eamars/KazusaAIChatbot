# Plan 3 DSH operations

This guide describes the current production boundary after the Plan 3
cutover. Adapters remain thin: they normalize platform events, call the
Brain, and render returned surfaces. The Brain owns cognition, dialog,
persistence, and delivery. DSH is the only multi-step task execution route.

## Local setup

Use the project virtual environment for every Python command:

~~~powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
~~~

Install the Python dependencies using the repository's normal setup. The DSH
sidecar has its own frozen package-manager lockfile and is built and tested
from sidecars/dsh_resolution. Never copy credentials into command lines or
documentation.

The task route requires the six AGENTIC_RESOLVER_LLM_* settings,
KAZUSA_DSH_SIDECAR_URL, KAZUSA_DSH_RPC_TOKEN, KAZUSA_DSH_DATA_ROOT,
KAZUSA_DSH_TOOL_GATEWAY_SECRET, KAZUSA_DSH_PYTHON_EXECUTABLE, and the
absolute AGENTIC_RESOLVER_WORKSPACE_ROOT. RAG3 retains its planner, subagent,
web-provider, and prewarm settings. The old background-worker and coding
executor model-route families, workspace root, preflight switch, repair-call
limit, and repair-bundle limits are deleted configuration and must not be
restored.

## DSH-only task flow

The cognition stage emits task_resolution_request only when character
judgment, evidence, and scene pressure support task work. One shared
AgenticResolverRuntime then owns the bounded foreground and background edge.

~~~text
task_resolution_request
  -> accepted task and reviewed background job
  -> transient TaskResolutionAdmissionV1 observation
  -> claim-time dsh_task_binding.v1 in dsh_task_bindings
  -> fresh DSH authority and operation_generation
  -> open_dsh_resolution or continue_dsh_resolution
  -> sidecar Standard/native/semantic execution
  -> committed checkpoint or terminal exhaust
  -> typed TaskResolutionResultV1
  -> result-ready accepted task
  -> normal cognition/dialog/dispatcher delivery
~~~

TaskResolutionAdmissionV1 is model-hidden and transient. Its exact fields
are schema_version, accepted_task_id, background_work_job_id, and
task_session_id. It is an acknowledgement observation only: it contains no
authority, no checkpoint, and no deferred result. Direct background admission
therefore returns before checkpoint commitment.

Only a committed checkpoint may produce
TaskResolutionResultV1(status="deferred"), carrying one
DshResolutionRefV1. The worker payload is
task_orchestrator_worker_payload.v2 and supports
open_dsh_resolution and continue_dsh_resolution. Document revision CAS,
operation_generation, lease fencing, and operation idempotency protect every
claim, checkpoint, continuation, and terminal replay. Sidecar loss, stale
authority, stale catalog, malformed state, and lease loss fail closed or
recover from the durable binding; there is no alternate task executor.

The accepted-task control capability is accepted_task_control.v1. It allows
only continue, summarize, and cancel for the same opaque accepted
task/session binding. accepted_task_status_check is read-only. Controls
obtain fresh DSH authority, use the current generation, and return through
normal cognition, dialog, dispatcher, and adapter delivery. Workers never
send visible text directly.

## Readiness and drain

Before accepting task work, call the authenticated Brain endpoint:

~~~text
GET /runtime/dsh/health
~~~

The response must report configured, durable_store, and cognition_judge.
The authenticated sidecar system.health response is ready only when route,
Standard, semantic-worker, web, Brain, catalog, policy, workspace, profile,
release, and store values match. Any mismatch keeps the route unavailable.

The Plan 3 drain audit is read-only:

~~~powershell
venv\Scripts\python scripts/check_dsh_plan3_drain.py --legacy-coding-workspace-root <abs-root> --format json
~~~

It counts these five categories without changing data:

1. active legacy coding ledger rows;
2. active accepted-task rows using the retired executor;
3. pending background jobs using the retired executor;
4. open pre-cutover DSH interactions or grants; and
5. stale task bindings or unresolved recovery state.

Run it against an explicitly resolved legacy workspace root. Deployment,
process shutdown, and production data changes require separate authorization.

## Semantic catalog and public media

The DSH model-visible catalog has exactly fourteen rows. Rows 1 through 13
are the completed Plan 2 semantic contract and remain byte-identical. The
sole Plan 3 addition is kazusa_inspect_public_media.

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

The public-media tool accepts one HTTP(S) URL and one visual question. It
rejects credentials and fragments, resolves DNS before connecting, and
rejects private, loopback, link-local, multicast, reserved, and unspecified
addresses. Redirect targets are rechecked and at most 3 redirects are
allowed. The timeout is 15 seconds and the response body limit is 6 MiB.
MIME and magic bytes must identify PNG, JPEG, GIF, or WebP; Pillow must
decode the image; width and height must each be 1 through 8192. Evidence
uses source dsh_public_media. Raw bytes and base64 are never model-facing.

The semantic catalog digest changes because of the fourteenth row. A terminal
or checkpointed V2 thread with no open interaction rotates to a fresh segment
with the new digest. Old authority and grants fail closed. Open pre-cutover
interactions and grants drain before the Brain and sidecar switch.

## Retained owners

RAG3/local-context remains the ordinary chat evidence owner. It retains
conversation, memory, people, calendar, approved web evidence, and its
prewarm/cache owner. DSH semantic tools provide bounded task evidence without
becoming persona or final stance. Cognition owns character judgment and
response goals; dialog owns final wording; consolidation, scheduler,
reflection, and future_speak remain outside the live task session.

The sidecar's native Standard tools keep their existing authority, sandbox,
filesystem, shell, web, approval, and job controls. Plan 3 changes only the
semantic catalog row and the task edge described above.

## Verification commands

Focused documentation and manifest checks:

~~~powershell
venv\Scripts\python -m pytest -q tests/test_dsh_plan3_documentation.py tests/test_test_impact_manifest.py
~~~

The collection preflight may be inspected with:

~~~powershell
venv\Scripts\python -m pytest --collect-only -q tests/control_console_e2e
~~~

Full non-live collection:

~~~powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
~~~

Changed-source impact enforcement:

~~~powershell
venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run
~~~

Sidecar verification uses the pinned commands from the active plan:

~~~powershell
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution install --frozen-lockfile
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution build
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution test
~~~

Live DB nodes run explicitly:

~~~powershell
venv\Scripts\python -m pytest -m live_db -q tests/test_dsh_plan3_task_resolution_live_db.py
~~~

Live LLM nodes run one at a time with output, traces, and delivery evidence
inspected. The active Plan 3 evidence ledger records each command and
residual.

Inspect live LLM cases one at a time and retain their output and trace
evidence.

## Cognition observation and browser verification

The operator view consumes the bounded process-local
cognition_run_observation.v1 carrier. Its source-to-wire projection is
validation-only and keeps semantic status separate from historical
persistence. Use the collection gate for tests/control_console_e2e and run
the browser checks through the in-app browser or Playwright.

The expected shell exposes Overview, Debug, and Self Latest sections. Verify
HTML escaping, bounded sequence and reference fields, and zero page or console error logs. Observation sections may be available, unavailable,
invalid, or omitted according to producer status; this is not a semantic
reinterpretation. Cancellation publishes no terminal observation.
