# DSH Plan 1: Standalone Sidecar And Resolution Interface

## Summary

- Goal: replace the current standalone `agentic_resolver` implementation with
  the agreed DeepSeek Harness integration while preserving the independently
  callable `AgenticResolverRuntime.resolve(...)` entry point.
- Status: draft; executable after explicit user approval.
- Plan class: destructive standalone resolver replacement plus an additive,
  independently operated Node/TypeScript sidecar.
- Governing architecture:
  `docs/architecture/dsh_integration_architecture.md`.
- Functional support at exit: a caller can open or continue a durable
  ResolutionThread through the standalone Python entry point, exercise the
  complete control plane, and receive a validated
  `DSHResolutionExhaustV1`. The resolver profile has the generic capability
  protocol and `submit_resolution` only.
- Isolation boundary: the sidecar runs as a separate long-lived OS process in
  parallel with Kazusa Brain. Plan 1 creates no import, call, route, queue, or
  lifecycle edge from the existing Brain path.
- Compatibility policy: big-bang replacement inside `agentic_resolver`. The
  old resolver contracts, loop, skills, subagents, facades, checkpoints, and
  shared LLM streaming additions receive no alias, adapter, converter,
  fallback, or dual-run path.
- Effort estimate: six bounded implementation blocks. Effort is measured by
  functional gates and verification evidence rather than elapsed time or
  person-days.

## Three-Plan Functional Staging

| Plan | Functional support at its exit | Execution gate |
|---|---|---|
| Plan 1 | Standalone DSH sidecar, canonical intake/exhaust, durable thread/session lifecycle, control plane, terminal tool, and generic tool/evidence seam | All Plan 1 gates in this document are green while the current Brain path remains unchanged |
| Plan 2 | The standalone DSH resolver can use the accepted Kazusa leaf capabilities and DSH coding capability end to end | Every admitted tool is authorized, observable, evidence-producing, restart-safe, and validated through standalone intake-to-exhaust runs |
| Plan 3 | Kazusa Brain calls the DSH resolver in production and the old post-selector resolution designs are deleted in one cutover | Production task-resolution behavior, background continuation, cognition handoff, and fault recovery pass on the DSH-only path |

Plans execute strictly in order and one at a time. Plan 2 is refined only from
the completed Plan 1 surface. Plan 3 is refined only from the completed Plan 2
tool catalog and operational evidence.

## Confirmed Decisions

1. `src/agentic_resolver` remains the standalone Python package.
2. `from agentic_resolver import AgenticResolverRuntime` and
   `await runtime.resolve(intake)` remain the public invocation shape.
3. The request, result, construction, persistence, and lifecycle contracts
   behind that method move atomically to the DSH architecture.
4. No legacy `AgenticResolverRequestV1`, `AgenticResolverResultV1`,
   `submit_result`, skill, subagent, four-facade, or custom loop contract
   survives.
5. The old resolver-only native-tool streaming additions to
   `kazusa_ai_chatbot.llm_interface` are removed because they have no current
   Brain consumer.
6. Ordinary `LLInterface.ainvoke(...)`, `invoke(...)`, JSON output, reload,
   provider, route-report, and cache behavior remains unchanged.
7. The sidecar is a separate long-running Node/TypeScript process. Production
   Python code connects to it and never starts one process per resolution,
   thread, session, or call.
8. An operator or service supervisor starts and stops the sidecar. Test
   fixtures may launch it as a subprocess to validate the real process
   boundary.
9. The sidecar composes released public DSH packages. The DSH source checkout
   remains read-only and is not copied, patched, or vendored into Kazusa.
10. All imported DSH packages are pinned exactly to `0.1.1-rc.2` and the
    checked public API is the clean upstream revision
    `b150a551b8d465e31e418e1b2eaf5e79bbb7d28e`.
11. The sidecar uses Node `^22.19.0 || >=24` and
    `pnpm@11.7.0` with a committed sidecar-local lockfile.
12. The production bridge is JSON-RPC 2.0 over HTTP/1.1 on loopback. It binds
    only to `127.0.0.1`, exposes one `/rpc` endpoint, requires a bearer token,
    and validates a versioned request and response envelope.
13. The sidecar stores DSH session events through the released SQLite
    persistence provider at an explicitly configured path.
14. Kazusa stores ResolutionThread, segment, compatibility, and lease metadata
    in MongoDB through named `kazusa_ai_chatbot.db` helpers. Raw MongoDB access
    remains inside that package.
15. The initial resume rule is exact equality of thread identity, goal
    continuation, scope fingerprint, audience fingerprint, resolver profile,
    DSH release, model route, tool catalog digest, and policy epoch.
16. Any compatibility mismatch creates a new segment in the same thread with
    a typed rotation reason. Topic similarity alone never resumes a segment.
17. One segment has at most one unexpired Kazusa lease and one live DSH Agent
    activation.
18. `submit_resolution` is the only model-owned terminal operation.
    `deferred` is represented only by a runtime-owned checkpointed exhaust.
19. Assistant prose and private reasoning never become authoritative exhaust
    fields.
20. The evidence ledger is reconstructed from Kazusa-specific, log-only DSH
    session events. Compaction therefore cannot erase registered evidence.
21. Plan 1 exposes no Kazusa conversation, memory, person, calendar, web,
    media, text, compute, RAG, coding, skill, subagent, shell, filesystem, or
    scheduler tool.
22. Plan 1 implements the generic typed tool-result and evidence-registration
    protocol so Plan 2 can add capabilities without changing the intake,
    exhaust, session, terminal, or RPC boundary.
23. The current Brain, action selector, task resolution, RAG, cognition,
    coding, background work, adapters, dialog, scheduler, and delivery paths
    remain behaviorally unchanged.
24. Mapping `DSHResolutionExhaustV1` into Brain-owned
    `TaskResolutionResultV1` is Plan 3 work.

## Fixed Execution Ownership

This draft authorizes planning only. After an explicit user implementation
command, the parent records `approved` and then `in_progress` before the Luna
worker makes the first production edit.

Production execution uses exactly two roles:

| Role | Fixed owner | Responsibility |
|---|---|---|
| Architecture and closure | Parent agent | Owns architecture decisions, scope control, one consolidated material review, gate interpretation, plan status, and final closure |
| Implementation and verification | One persistent `/root/dsh_implementation_worker` subagent on `gpt-5.6-luna`, `max` reasoning, normal execution speed | Owns every production edit, test edit, dependency/lockfile operation, build, test execution, remediation, and pre-handoff self-review |

The Luna worker is the only implementation worker for this plan and remains
the fixed implementation worker for successor Plans 2 and 3 when they are
approved. The parent does read-only architecture and code review and consumes
the worker's command results. The parent does not perform production edits,
test remediation, or test execution. Changing this binding requires a
user-approved plan amendment.

### Maximum Two Review Iterations

1. **Iteration 1 — complete candidate.** The Luna worker completes every work
   block, runs the mapped gates, performs its own diff/style/collection review,
   and hands over one complete candidate. The parent returns one consolidated
   set of material architecture, correctness, security, lifecycle, or
   acceptance findings.
2. **Iteration 2 — final remediation, when required.** The same worker resolves
   the complete finding set, reruns every affected gate plus the final
   acceptance suite, self-checks again, and hands over the final candidate.
   The parent gives a final pass or marks the plan blocked.

When Iteration 1 has no material finding, Iteration 2 remains unused and the
parent proceeds directly to the gate decision. A third review iteration is
outside this plan. Minor formatting, import,
typing, lint, typo, test-collection, fixture, and documentation-link defects
are worker self-review responsibilities and must be corrected before each
handoff. The parent review reports consolidated material findings rather than
using a review cycle as a minor-error checklist.

### Proportionate Process Evidence

Required execution evidence is limited to:

- `git status --short` before execution and at closure;
- the plan-owned path list and final diff;
- the pinned DSH release and upstream reference revision;
- exact commands, pass/fail counts, and inspected live-case outcomes; and
- the parent gate decision.

Runtime security fingerprints and the tool-catalog digest remain mandatory
because they are part of the architecture. Workspace-file hashes,
whole-repository hashes, test-artifact hashes, redundant integrity ledgers, and
parallel review transcripts are outside this plan.

## Frozen Plan 1 Interface

### Process And Transport Boundary

```text
standalone caller
    -> AgenticResolverRuntime.resolve(DSHResolutionIntakeV1)
    -> ResolutionController
       -> Kazusa MongoDB ResolutionThread store
       -> reusable authenticated HTTP JSON-RPC client
          -> long-lived DSH resolution sidecar
             -> DSH Agent/session/tool loop
             -> durable DSH SQLite event log
             -> submit_resolution
    <- DSHResolutionExhaustV1

Kazusa Brain process ---------------- runs independently ----------------+
```

The sidecar accepts only:

- `system.health`;
- `resolution.open`;
- `resolution.continue`;
- `resolution.amend`;
- `resolution.request_checkpoint`;
- `resolution.cancel`;
- `resolution.inspect`; and
- `resolution.dispose_activation`.

Every request uses JSON-RPC `2.0`, a unique request id, protocol version
`kazusa.dsh-resolution-rpc.v1`, and the bearer token. Every response echoes the
request id and protocol version and is either a typed result or a bounded typed
error. Unknown keys, methods, versions, identities, or result shapes fail
closed. Transport exception text and credentials never enter model-visible
content or exhaust details.

Runtime configuration is explicit:

- `KAZUSA_DSH_SIDECAR_URL`: loopback `http://127.0.0.1:<port>/rpc` URL;
- `KAZUSA_DSH_RPC_TOKEN`: required non-empty bearer secret;
- `KAZUSA_DSH_SESSION_DB_PATH`: absolute writable SQLite path owned by the
  sidecar process;
- `KAZUSA_DSH_MODEL`: required DSH model id; and
- `DEEPSEEK_API_KEY` plus optional `DEEPSEEK_BASE_URL` for the
  `deepseek-official` live route.

The server and Python client both refuse non-loopback endpoints. TLS is omitted
for this loopback-only v1 transport. Runtime storage lives outside the
repository. Sidecar shutdown follows the process supervisor and flushes
sessions before closing the DSH context and HTTP listener.

### DSH Composition

The `kazusa-resolver-v1` profile composes DSH public services for:

- Cordis context and Agent service;
- Agent loop and Session service;
- SQLite session persistence;
- semantic checkpoint policy;
- DSH DeepSeek LLM route;
- Tool runtime;
- one-call-per-step policy;
- Kazusa evidence-log event extension;
- `submit_resolution`; and
- lifecycle and usage diagnostics.

The package manifest pins `@deepseek-ai/cordis` to `4.0.1`,
`@deepseek-ai/schemastery` to `3.18.1`, and these direct Harness packages to
`0.1.1-rc.2`:

- `@deepseek-ai/dsh-agent`;
- `@deepseek-ai/dsh-agent-loop`;
- `@deepseek-ai/dsh-invariants`;
- `@deepseek-ai/dsh-llm`;
- `@deepseek-ai/dsh-llm-deepseek`;
- `@deepseek-ai/dsh-scope`;
- `@deepseek-ai/dsh-session`;
- `@deepseek-ai/dsh-session-persistence`;
- `@deepseek-ai/dsh-session-persistence-sqlite`;
- `@deepseek-ai/dsh-session-checkpoint-policy`;
- `@deepseek-ai/dsh-settings`;
- `@deepseek-ai/dsh-system-prompt`; and
- `@deepseek-ai/dsh-tools`.

Transitive versions are fixed by
`sidecars/dsh_resolution/pnpm-lock.yaml`.

The sidecar uses `ctx.agents.create(...)` and
`ctx.agents.resume(...)`, `agent.followup(...)`,
`agent.steer(...)`, `agent.cancel(...)`, `agent.whenIdle()`, the
`agent/pre-step` and `agent/turn-stopping` lifecycle seams, DSH session
flush/inspect services, `AgentHandle.dispose()`, ToolRuntime registration, and
`concludeTurn()`. It does not modify DSH loop, session, persistence, LLM, tool,
or SDK source.

### Data Plane

`DSHResolutionIntakeV1` follows the governing architecture and keeps two
strictly separate objects:

- `runtime_authority`: request/thread identities, priority, deadlines, hard
  budgets, opaque capability token, trusted scope and audience fingerprints,
  resolver profile, tool catalog digest, and policy epoch; and
- `model_input`: bounded objective, constraints, success criteria, known facts,
  uncertainty, approved source literals, and approved continuation material.

Capability tokens, database identifiers, private credentials, approval
authority, ACLs, and workspace/network authority never appear in
`model_input`. The sidecar renders only `model_input` into the stable canonical
DSH waking message.

`DSHResolutionExhaustV1` has exactly three top-level kinds:

- `terminal`: a validated `submit_resolution` payload;
- `checkpointed`: a runtime-owned resumable checkpoint that maps to deferred
  only in Plan 3; or
- `runtime_fault`: a typed integration/runtime failure with no fabricated
  terminal result.

The terminal statuses are `resolved`, `partial`,
`needs_user_input`, `approval_required`, `unavailable`, and `failed`.
Status-specific required fields, evidence references, usage, segment identity,
last committed sequence, profile/catalog/scope/audience fingerprints, and
model route are validated before the exhaust crosses RPC.

### Thread, Segment, Lease, And Persistence Boundary

The MongoDB owner stores one strict `resolution_thread_store.v1` document per
thread. It contains the architecture's `ResolutionThreadRecordV1`, ordered
`ResolverSessionSegmentV1` records, a monotonic revision, and the current
lease. Named database helpers own:

- index creation;
- idempotent new-thread creation;
- explicit continuation lookup;
- exact compatibility evaluation inputs;
- segment rotation with a declared reason;
- compare-and-set lease acquisition, renewal, and release;
- sidecar session-sequence/state updates;
- terminal, checkpoint, failure, abandonment, and expiry transitions; and
- read/inspect operations.

The controller uses repository protocols and imports the narrow public
`kazusa_ai_chatbot.db.resolution_threads` owner. Mongo selectors and update
operators stay in that module. The standalone repository initialization path
creates its indexes before its first operation. The DB schemas and DB ICD
register the new collection, while the global Brain `db_bootstrap()` path
remains unchanged in Plan 1.

DSH remains the source of truth for model/tool history. The sidecar SQLite
backend persists the append-only SessionEvent log. Kazusa-specific
`kazusa/evidence` events hold bounded `StandardToolResultV1` provenance and
authorization facts outside model-visible history. On resume the sidecar
rebuilds the evidence ledger from those events before accepting a terminal
submission.

### Activation And Step Rules

- The controller acquires a segment lease before `open` or `continue` and
  releases it after the activation is disposed.
- The sidecar rejects a second live activation for the same DSH session.
- Terminal, waiting, checkpointed, canceled, and faulted calls flush the DSH
  session and dispose the activation before returning.
- A safe checkpoint stops before the next model step, flushes, returns the last
  committed sequence, and preserves the same session for cold resume.
- A hard interruption uses DSH crash repair. Unknown side-effect outcome
  remains typed and is never treated as success or blindly retried.
- One model response may contain exactly one complete tool call or one complete
  `submit_resolution` call. A multiple-call response is rejected before any
  call dispatch, receives at most two bounded regeneration attempts, and then
  fails closed.
- Terminal evidence ids must exist in the rebuilt ledger and match the current
  thread, segment, scope, audience, and authorization epoch.
- `submit_resolution` calls `concludeTurn()`, waits for idle, flushes, builds
  exhaust, and then permits activation disposal.

## Complete Change Inventory

### Create

| Path | Ownership |
|---|---|
| `sidecars/dsh_resolution/package.json`, `pnpm-lock.yaml`, `tsconfig.json`, `.gitignore`, `README.md` | Exact sidecar dependency, build, runtime, storage, and operator contract |
| `sidecars/dsh_resolution/src/main.ts` | Required configuration, process lifecycle, and loopback HTTP startup |
| `sidecars/dsh_resolution/src/contracts.ts` | Exact RPC, intake, exhaust, control, tool-result, checkpoint, and terminal validators |
| `sidecars/dsh_resolution/src/rpc.ts` | Authenticated JSON-RPC dispatch and bounded errors |
| `sidecars/dsh_resolution/src/runtime.ts` | DSH context, Agent activation map, create/resume/amend/checkpoint/cancel/inspect/dispose |
| `sidecars/dsh_resolution/src/profile.ts` | `kazusa-resolver-v1` composition, model route, one-call policy, empty semantic tool catalog |
| `sidecars/dsh_resolution/src/evidence.ts` | Log event extension, ledger rebuild, registration, and terminal evidence validation |
| `sidecars/dsh_resolution/src/submit_resolution.ts` | Only model-owned terminal tool and `concludeTurn()` handling |
| `sidecars/dsh_resolution/tests/*.spec.ts` | Direct TypeScript contract, runtime, lifecycle, process, and restart gates |
| `src/agentic_resolver/controller.py` | Resolution lifecycle, compatibility, lease, RPC, state, and exhaust orchestration |
| `src/agentic_resolver/rpc.py` | Reusable authenticated HTTP JSON-RPC client and typed transport faults |
| `src/agentic_resolver/persistence.py` | Controller-facing repository protocol and public-DB-helper adapter |
| `src/agentic_resolver/fingerprints.py` | Canonical scope, audience, profile, catalog, and policy fingerprints only |
| `src/agentic_resolver/errors.py` | Strict contract, lifecycle, lease, persistence, and transport errors |
| `src/kazusa_ai_chatbot/db/resolution_threads.py` | Raw MongoDB collection/index/CAS owner |
| `tests/test_agentic_resolver_fingerprints.py` | Fingerprint and process-overhead boundary |
| `tests/test_agentic_resolver_evidence.py` | Python contract and exhaust identity checks for sidecar-owned evidence |
| `tests/test_agentic_resolver_rpc.py` | Versioned authenticated RPC client |
| `tests/test_agentic_resolver_controller.py` | Thread, segment, lease, control-plane, and exhaust state machine |
| `tests/test_agentic_resolver_persistence.py` | Repository adapter and durable metadata behavior |
| `tests/test_agentic_resolver_decommission.py` | Legacy absence and current Brain-path non-import gate |
| `tests/test_agentic_resolver_sidecar_process.py` | Real long-lived process, restart, cold-resume, auth, and isolation gates |
| `tests/test_agentic_resolver_live_db.py` | One explicit real-Mongo lifecycle case |

### Replace In Place

| Path | Replacement |
|---|---|
| `src/agentic_resolver/__init__.py` | Export only canonical contracts, errors, controller, and `AgenticResolverRuntime` |
| `src/agentic_resolver/contracts.py` | Replace every legacy resolver DTO with the DSH architecture DTOs |
| `src/agentic_resolver/runtime.py` | Preserve `resolve(...)` while delegating to the new controller and sidecar |
| `src/agentic_resolver/README.md` | Standalone sidecar ICD, construction example, control plane, runtime configuration, and non-Brain boundary |
| `tests/test_agentic_resolver_contracts.py` | Canonical strict data-plane contract tests |
| `tests/test_agentic_resolver_runtime.py` | Preserved standalone entry point and typed exhaust tests |
| `tests/test_agentic_resolver_live_llm.py` | One live DSH terminal-exhaust case |

### Modify

| Path | Change |
|---|---|
| `src/kazusa_ai_chatbot/db/schemas.py` | Add strict thread, segment, lease, and store document types |
| `src/kazusa_ai_chatbot/db/README.md` | Register collection ownership, public helpers, lease semantics, and retention boundary |
| `src/kazusa_ai_chatbot/llm_interface/{README.md,__init__.py,contracts.py,interface.py,reload.py}` and `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py` | Remove only the unconsumed old resolver native-tool stream API and retain ordinary LLM behavior |
| `pyproject.toml` | Remove `PyYAML`, whose only consumer is the deleted legacy skill design; retain `agentic_resolver*` package discovery |
| `README.md` and `docs/HOWTO.md` | Add independent sidecar install/start/health/stop and standalone invocation instructions |
| `docs/architecture/dsh_integration_architecture.md` | Record the frozen Plan 1 transport, persistence, release pin, empty semantic catalog, and staged cutover decisions |
| `tests/test_llm_interface_reload.py` | Remove resolver-stream-specific assertions and retain ordinary reload coverage |
| `tests/test_test_impact_manifest.py` and `tests/ownership/source_test_impact_manifest.json` | Replace old resolver rows with exact new ownership and test nodes |

The resolver-originated shared LLM removal is limited to
`LLInterface.astream_tools(...)`, its native stream/tool contracts, resolver
capability declarations, provider tool-stream binding, tool-schema cache
partitioning, and resolver-only reload plumbing. The worker must prove no
remaining non-resolver import before deletion and preserve every ordinary LLM
test.

### Delete

```text
src/agentic_resolver/context_budget.py
src/agentic_resolver/integrations/
src/agentic_resolver/json_protocol.py
src/agentic_resolver/loop.py
src/agentic_resolver/model.py
src/agentic_resolver/session.py
src/agentic_resolver/skills.py
src/agentic_resolver/streaming.py
src/agentic_resolver/subagents.py
src/agentic_resolver/tools.py
resolver_skills/.gitkeep
tests/test_agentic_resolver_architecture_docs.py
tests/test_agentic_resolver_context_budget.py
tests/test_agentic_resolver_json_protocol.py
tests/test_agentic_resolver_kazusa_tools.py
tests/test_agentic_resolver_loop.py
tests/test_agentic_resolver_session.py
tests/test_agentic_resolver_skills.py
tests/test_agentic_resolver_standalone.py
tests/test_agentic_resolver_streaming.py
tests/test_agentic_resolver_subagents.py
tests/test_agentic_resolver_tools.py
tests/test_llm_interface_tool_stream.py
```

No deleted symbol, module, request shape, checkpoint shape, or facade receives a
replacement alias.

## Work Blocks And Effort Gates

| Block | Relative effort | Work | Independent completion gate |
|---|---|---|---|
| 1. Legacy replacement boundary | Medium | Delete the old resolver design and resolver-only shared LLM additions; establish the new package exports and strict contracts | Old symbols and modules are absent; ordinary LLM regressions pass; current Brain/workflow source has no resolver import |
| 2. Durable ownership | Medium | Add Mongo thread/segment/lease storage, repository adapter, fingerprints, and standalone-only index initialization | Deterministic repository/CAS tests and one real-Mongo lease/rotation case pass while global Brain bootstrap remains unchanged |
| 3. Sidecar spine | High | Add the exact DSH composition, loopback JSON-RPC server, SQLite persistence, terminal tool, and empty semantic catalog | Typecheck/build/direct TypeScript tests pass; one independent sidecar process serves health and two sessions |
| 4. Standalone controller | High | Add reusable RPC client, controller state machine, preserved runtime entry point, control methods, and typed exhaust | Python unit tests pass for open/continue/amend/checkpoint/cancel/inspect/dispose and terminal/checkpoint/fault exhaust |
| 5. Restart and adversarial lifecycle | Medium | Add cold resume, lease/live-activation exclusion, mismatch rotation, multi-call rejection, evidence validation, disconnect behavior, and crash classification | Real-process restart/security/fault tests pass with no semantic tool or Brain edge |
| 6. Operational and closure gate | Small | Final docs, manifest, exact lockfile, mapped regressions, deterministic suite, live DB, and one live LLM case | Every release blocker below is green and the parent completes final architecture sign-off |

Block completion records only its functional result and command evidence. It
does not create an additional review cycle.

## Exact Verification Matrix

### TypeScript Direct Tests

The worker implements and passes these exact Vitest titles:

```text
sidecars/dsh_resolution/tests/rpc.spec.ts > authenticates versioned request and response frames
sidecars/dsh_resolution/tests/rpc.spec.ts > rejects non-loopback unauthenticated unknown and malformed requests
sidecars/dsh_resolution/tests/contracts.spec.ts > separates runtime authority from model input
sidecars/dsh_resolution/tests/contracts.spec.ts > validates status-specific submit_resolution and exhaust
sidecars/dsh_resolution/tests/evidence.spec.ts > rebuilds authorized evidence from durable log events
sidecars/dsh_resolution/tests/evidence.spec.ts > rejects unknown cross-scope and cross-segment evidence
sidecars/dsh_resolution/tests/runtime.spec.ts > opens an agent captures terminal submit and disposes activation
sidecars/dsh_resolution/tests/runtime.spec.ts > checkpoints at pre-step and resumes the same DSH session
sidecars/dsh_resolution/tests/runtime.spec.ts > amends with steer queues continuation and cancels safely
sidecars/dsh_resolution/tests/runtime.spec.ts > rejects multi-call model steps before dispatch
sidecars/dsh_resolution/tests/lifecycle.spec.ts > rotates on scope audience profile release model catalog or policy mismatch
sidecars/dsh_resolution/tests/lifecycle.spec.ts > rejects duplicate live activation
sidecars/dsh_resolution/tests/lifecycle.spec.ts > cancels without deleting durable session
sidecars/dsh_resolution/tests/process.spec.ts > serves one long-lived independent process across multiple sessions
sidecars/dsh_resolution/tests/process.spec.ts > restarts and cold-resumes persisted session state
```

Final commands:

```powershell
pnpm --dir sidecars/dsh_resolution install --frozen-lockfile
pnpm --dir sidecars/dsh_resolution run typecheck
pnpm --dir sidecars/dsh_resolution run build
pnpm --dir sidecars/dsh_resolution test
```

### Python Deterministic Tests

The worker implements and passes these exact node IDs:

```text
tests/test_agentic_resolver_contracts.py::test_intake_v1_rejects_unknown_fields_and_runtime_authority_in_model_input
tests/test_agentic_resolver_contracts.py::test_thread_and_segment_records_validate_identity_state_and_epoch
tests/test_agentic_resolver_contracts.py::test_submit_resolution_requires_status_specific_fields
tests/test_agentic_resolver_fingerprints.py::test_authority_fingerprints_are_stable_and_do_not_hash_event_logs
tests/test_agentic_resolver_evidence.py::test_standard_tool_result_contract_binds_thread_segment_scope_and_audience
tests/test_agentic_resolver_evidence.py::test_exhaust_rejects_evidence_with_mismatched_thread_segment_scope_or_audience
tests/test_agentic_resolver_rpc.py::test_versioned_authenticated_rpc_rejects_bad_version_token_and_method
tests/test_agentic_resolver_rpc.py::test_rpc_round_trip_preserves_typed_intake_and_exhaust
tests/test_agentic_resolver_rpc.py::test_rpc_disconnect_becomes_typed_runtime_fault
tests/test_agentic_resolver_controller.py::test_open_creates_one_thread_one_segment_and_one_activation
tests/test_agentic_resolver_controller.py::test_continue_reuses_segment_only_for_same_goal_scope_audience_and_epoch
tests/test_agentic_resolver_controller.py::test_incompatible_scope_audience_profile_release_model_catalog_or_policy_rotates_segment
tests/test_agentic_resolver_controller.py::test_duplicate_execution_lease_fails_closed
tests/test_agentic_resolver_controller.py::test_amend_steers_in_flight_work_and_followup_queues_next_turn
tests/test_agentic_resolver_controller.py::test_checkpoint_cancel_inspect_and_dispose_preserve_durable_lineage
tests/test_agentic_resolver_controller.py::test_terminal_submit_maps_to_typed_terminal_exhaust
tests/test_agentic_resolver_controller.py::test_checkpoint_maps_to_runtime_owned_checkpointed_exhaust
tests/test_agentic_resolver_persistence.py::test_thread_segment_and_lease_metadata_round_trip
tests/test_agentic_resolver_persistence.py::test_cold_resume_uses_persisted_session_reference_and_revision
tests/test_agentic_resolver_persistence.py::test_expired_or_corrupt_segment_fails_closed_or_rotates
tests/test_agentic_resolver_runtime.py::test_resolve_preserves_standalone_entrypoint_and_returns_typed_exhaust
tests/test_agentic_resolver_runtime.py::test_runtime_has_no_brain_task_resolution_rag_or_coding_import_edge
tests/test_agentic_resolver_decommission.py::test_old_resolver_contracts_facades_and_aliases_are_absent
tests/test_agentic_resolver_decommission.py::test_brain_task_resolution_rag_and_coding_paths_do_not_import_or_spawn_resolver
tests/test_agentic_resolver_decommission.py::test_old_native_tool_stream_surface_is_absent
tests/test_agentic_resolver_sidecar_process.py::test_standalone_runtime_uses_one_long_lived_sidecar_across_two_resolves
tests/test_agentic_resolver_sidecar_process.py::test_brain_and_sidecar_run_independently_and_sidecar_stop_does_not_stop_brain
tests/test_agentic_resolver_sidecar_process.py::test_sidecar_restart_preserves_checkpoint_and_cold_resumes
tests/test_agentic_resolver_sidecar_process.py::test_scope_audience_profile_release_model_catalog_and_policy_mismatch_rotates
tests/test_agentic_resolver_sidecar_process.py::test_terminal_exhaust_contains_only_validated_submit_resolution_and_evidence_refs
tests/test_agentic_resolver_sidecar_process.py::test_bad_rpc_authentication_version_and_schema_fail_closed
```

The mapped deterministic command is:

```powershell
venv\Scripts\python -m pytest tests/test_agentic_resolver_contracts.py tests/test_agentic_resolver_fingerprints.py tests/test_agentic_resolver_evidence.py tests/test_agentic_resolver_rpc.py tests/test_agentic_resolver_controller.py tests/test_agentic_resolver_persistence.py tests/test_agentic_resolver_runtime.py tests/test_agentic_resolver_decommission.py tests/test_agentic_resolver_sidecar_process.py
```

### Preserved-System Regression

The worker runs:

```powershell
venv\Scripts\python -m pytest tests/test_llm_interface_contracts.py tests/test_llm_interface_openai_provider.py tests/test_llm_interface_reload.py tests/test_llm_interface_route_report.py tests/test_llm_interface_migration.py
venv\Scripts\python -m pytest tests/test_task_resolution_contracts.py tests/test_task_resolution_state.py tests/test_task_resolution_specialists.py tests/test_task_resolution_orchestrator.py tests/test_task_resolution_inline_promotion.py tests/test_task_resolution_background_resume.py
venv\Scripts\python -m pytest tests/test_test_impact_manifest.py
venv\Scripts\python -m ruff check src/agentic_resolver src/kazusa_ai_chatbot/db/resolution_threads.py tests/test_agentic_resolver_contracts.py tests/test_agentic_resolver_fingerprints.py tests/test_agentic_resolver_evidence.py tests/test_agentic_resolver_rpc.py tests/test_agentic_resolver_controller.py tests/test_agentic_resolver_persistence.py tests/test_agentic_resolver_runtime.py tests/test_agentic_resolver_decommission.py tests/test_agentic_resolver_sidecar_process.py tests/test_agentic_resolver_live_db.py tests/test_agentic_resolver_live_llm.py
venv\Scripts\python -m pytest
```

The repository-wide deterministic suite is run on each candidate handed to the
parent, at most twice. Default pytest configuration excludes live DB, live LLM,
and live internet cases.

### Explicit Live Cases

Run each live case alone and inspect its complete output:

```powershell
venv\Scripts\python -m pytest -m live_db -s tests/test_agentic_resolver_live_db.py::test_resolution_thread_store_enforces_lease_rotation_and_cold_resume_in_real_mongodb
venv\Scripts\python -m pytest -m live_llm -s tests/test_agentic_resolver_live_llm.py::test_live_standalone_sidecar_resolution_reaches_submit_resolution
```

The live LLM case uses the real sidecar, real DSH loop and persistence, real
model route, canonical intake, and terminal tool. It passes only when the
returned exhaust is terminal, schema-valid, identity-consistent, contains no
assistant prose as authority, and its activation has been disposed after a
durable flush.

## Release-Blocking Gates

| Gate | Green condition |
|---|---|
| P1-G1 — Independent process | One long-lived sidecar handles multiple sessions and restarts independently; stopping it produces a typed standalone runtime fault and leaves the Brain process/path unaffected |
| P1-G2 — Canonical interface | Every control operation plus `DSHResolutionIntakeV1` and `DSHResolutionExhaustV1` is versioned, authenticated, strict, and accepted by both Python and TypeScript contract tests |
| P1-G3 — Durable lifecycle | New, continuation, checkpoint, cold resume, mismatch rotation, cancel, inspect, terminal, crash repair, lease release, and activation disposal all preserve one coherent thread lineage |
| P1-G4 — Authority and terminal safety | Runtime authority stays outside model input; multi-call dispatch is prevented; evidence is registered and scope-bound; only validated `submit_resolution` creates terminal exhaust |
| P1-G5 — Clean replacement | The legacy resolver and resolver-only LLM stream design are absent, no compatibility surface exists, and ordinary LLM behavior passes |
| P1-G6 — Brain non-impact | Current Brain/task-resolution/RAG/cognition/coding sources neither import nor spawn the new runtime, and the mapped current-path regression suite passes |
| P1-G7 — Verification | TypeScript build/tests, mapped Python tests, lint, deterministic repository suite, real Mongo case, and individually inspected live DSH LLM case pass |

All seven gates must be green. A skipped required live case is a blocked gate,
not a pass.

## Out Of Scope

- Any Brain, action-selector, task-resolution, cognition, dialog, adapter, or
  delivery call edge.
- Kazusa semantic tools and DSH coding capability.
- Background queue ownership and `TaskResolutionResultV1` mapping.
- Decommission of the existing Brain task-resolution, RAG, internal/external
  resolver, or coding path.
- Legacy checkpoint conversion, session migration, aliases, fallbacks, shadow
  execution, or backward compatibility.
- Broad DSH retention/deletion maintenance beyond durable resume required by
  the tests.
- Changes to the retained shared-memory prewarm path.

## Closure

The Luna worker records the final owned diff and exact gate results in this
document. The parent reviews the implementation against the seven functional
gates, records the consolidated review disposition, and alone changes plan
status or archives the plan. Closure means the standalone DSH path is fully
functional through intake and exhaust while Kazusa Brain remains structurally
and behaviorally independent.
