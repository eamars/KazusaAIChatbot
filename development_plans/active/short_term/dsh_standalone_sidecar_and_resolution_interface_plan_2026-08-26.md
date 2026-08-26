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

## Architecture Review Disposition — 2026-08-26

- Accepted P1-B1 and P1-B2: supported `tool/result.meta.kazusa` receipts replace
  the proposed custom event, and the complete terminal receipt is the durable
  restart authority.
- Accepted P1-B3 and P1-B4: semantic operation identity governs admission and
  disconnect reconciliation, while every invalid action cardinality follows
  one bounded correction contract.
- Accepted lifecycle hardening: activation/lease fencing, lease renewal and
  takeover, concurrent control requests, and physical session-store epochs are
  release-gating behavior.
- Accepted forward seams: the strict profile factory prepares a separate
  least-privilege Plan 2 coding profile; bounded dependency diagnostics and
  receipt metadata preserve upgradeability and authority separation.
- Plans 2 and 3 retain their coarse functional scope. Plan 1 remains `draft`
  until explicit implementation approval.

## Three-Plan Functional Staging

| Plan | Functional support at its exit | Execution gate |
|---|---|---|
| Plan 1 | Standalone DSH sidecar, canonical intake/exhaust, durable thread/session lifecycle, control plane, terminal tool, and generic tool/evidence seam | All Plan 1 gates in this document are green while the current Brain path remains unchanged |
| Plan 2 | The standalone DSH resolver can use the accepted Kazusa leaf capabilities and a separate least-privilege DSH coding profile end to end | Every admitted tool is authorized, observable, evidence-producing, restart-safe, and validated through standalone intake-to-exhaust runs |
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
13. Every mutating RPC carries a stable semantic `operation_id` and an
    `operation_payload_digest` over its canonical immutable payload. The
    JSON-RPC `id` remains transport correlation only. Repeating an operation
    with the same digest reconciles its existing admission; reusing the ID
    with a different digest fails closed.
14. An ambiguous transport loss is reconciled through `resolution.inspect`
    before any retry. Inspection distinguishes not admitted, admitted-active,
    checkpointed, terminal, canceled, faulted, and unknown outcomes.
15. The sidecar stores DSH session events through the released SQLite
    persistence provider under the versioned store epoch
    `dsh-sqlite-0.1.1-rc.2-v1`, at
    `<KAZUSA_DSH_DATA_ROOT>/dsh/0.1.1-rc.2/sessions.sqlite`.
16. Kazusa stores ResolutionThread, segment, compatibility, operation,
    activation, and lease metadata
    in MongoDB through named `kazusa_ai_chatbot.db` helpers. Raw MongoDB access
    remains inside that package.
17. The initial resume rule is exact equality of thread identity, goal
    continuation, scope fingerprint, audience fingerprint, resolver profile,
    DSH release, session store epoch, model route, tool catalog digest, and
    policy epoch.
18. Any compatibility mismatch creates a new segment in the same thread with
    a typed rotation reason. Topic similarity alone never resumes a segment.
19. One segment has at most one unexpired Kazusa lease. Each lease acquisition
    receives a monotonically increasing `lease_epoch`, and each live DSH Agent
    activation receives a unique `activation_id`; both values fence every live
    mutating control.
20. `submit_resolution` is the only model-owned terminal operation.
    `deferred` is represented only by a runtime-owned checkpointed exhaust.
21. Assistant prose and private reasoning never become authoritative exhaust
    fields.
22. The evidence ledger is reconstructed exclusively from bounded
    `evidence_receipt_v1` objects stored in the supported
    `tool/result.meta.kazusa` namespace. Plan 1 introduces no custom DSH event
    kinds.
23. A terminal result becomes committed only after the complete bounded
    `terminal_resolution_v1` receipt is durably stored in
    `tool/result.meta.kazusa`, the turn reaches idle, and the DSH session store
    is flushed. Inspection and restart replay derive exhaust from that receipt
    without rerunning the model.
24. A valid model step selects exactly one action: one non-terminal tool or
    `submit_resolution`. Multi-call output may create denied DSH bookkeeping,
    while Kazusa executes zero tool bodies or external side effects from that
    invalid step. Multi-call, zero-call, prose-only, and empty responses share
    the same bounded corrective-regeneration budget and then fail with
    `RESOLVER_ACTION_CONTRACT_EXHAUSTED`.
25. `profile.ts` exposes a profile factory with only
    `kazusa-resolver-v1` registered in Plan 1. Plan 2 adds a separate
    least-privilege coding profile instead of widening resolver sessions.
26. Plan 1 exposes no Kazusa conversation, memory, person, calendar, web,
    media, text, compute, RAG, coding, skill, subagent, shell, filesystem, or
    scheduler tool.
27. Plan 1 implements the generic typed tool-result and evidence-registration
    protocol so Plan 2 can add capabilities without changing the intake,
    exhaust, session, terminal, or RPC boundary.
28. The current Brain, action selector, task resolution, RAG, cognition,
    coding, background work, adapters, dialog, scheduler, and delivery paths
    remain behaviorally unchanged.
29. Mapping `DSHResolutionExhaustV1` into Brain-owned
    `TaskResolutionResultV1` is Plan 3 work.

## Fixed Execution Ownership

This draft authorizes planning only. After an explicit user implementation
command, the parent records `approved` and then `in_progress` before the Luna
worker makes the first production edit.

Production execution uses exactly two roles:

| Role | Fixed owner and model | Owned surface and responsibility | Authority limit | Mandatory skills | Capability floor | Required output and gate |
|---|---|---|---|---|---|---|
| Architecture and closure | Parent agent | Architecture decisions, plan/doc amendments, scope control, one consolidated material review, gate interpretation, status, and closure | Read-only over implementation and test results; writes only plan lifecycle/architecture records authorized by this plan | `development-plan`, `local-llm-architecture` | Senior system architecture and independent release judgment | One consolidated material finding set, then the final seven-gate disposition; plan status changes remain parent-owned |
| Implementation and verification | One persistent `/root/dsh_implementation_worker` subagent on `gpt-5.6-luna`, `max` reasoning, normal execution speed | Every plan-owned production/test/doc edit, dependency/lockfile operation, build, test execution, remediation, and pre-handoff self-review | Changes stay within the complete inventory and escalate any contract or scope conflict to the parent | `development-plan`, `local-llm-architecture`, `py-style` before Python edits, and `test-style-and-execution` before test edits or runs | Production-grade TypeScript/Python, DSH integration, Mongo CAS, process lifecycle, and deterministic/live verification | Complete owned diff, exact command results, inspected live outputs, and a clean self-review; handoff occurs only after all applicable mapped gates pass |

At execution start the parent validates the Luna model/reasoning binding and
the worker acknowledges its owned surface, authority limit, required skills,
and gate outputs. This startup contract is execution setup and does not consume
a review iteration.

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

Every request uses JSON-RPC `2.0`, a unique transport request id, protocol
version `kazusa.dsh-resolution-rpc.v1`, and the bearer token. Every mutating
method also carries `operation_id` and `operation_payload_digest`; live
activation methods carry `activation_id` and `lease_epoch`. The digest is
computed from the canonical method name and immutable typed parameters,
excluding the JSON-RPC id and bearer token. Every response echoes the transport
request id and protocol version and is either a typed result or a bounded typed
error. Unknown keys, methods, versions, identities, or result shapes fail
closed. Transport exception text and credentials never enter model-visible
content or exhaust details.

The RPC server accepts lifecycle-control requests concurrently while an
`open`, `continue`, or `amend` request is awaiting DSH completion. Active
operations are addressable by operation, thread, segment, activation, and
lease identity. A duplicate operation with an equal digest joins or returns
the already admitted result. A duplicate with a different digest returns
`OPERATION_ID_REUSE_MISMATCH` without admission.

After an ambiguous disconnect, the Python client calls `resolution.inspect`
with the semantic operation identity. It handles these exact dispositions:
`not_admitted`, `admitted_active`, `checkpointed`, `terminal`, `canceled`,
`faulted`, and `unknown`. `not_admitted` permits one replay with the same
operation identity and digest; `admitted_active` attaches to the existing
operation; committed states return their stored result. `unknown` returns the
typed runtime fault `OPERATION_OUTCOME_UNCERTAIN` and leaves the operation
available for later inspection. The client never creates a new semantic
operation to resolve an ambiguous transport outcome.

Runtime configuration is explicit:

- `KAZUSA_DSH_SIDECAR_URL`: loopback `http://127.0.0.1:<port>/rpc` URL;
- `KAZUSA_DSH_RPC_TOKEN`: required non-empty bearer secret;
- `KAZUSA_DSH_DATA_ROOT`: absolute writable data root owned by the sidecar;
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
- the Kazusa action-selection contract and bounded correction policy;
- supported `tool/result.meta.kazusa` receipt emitters;
- `submit_resolution`; and
- lifecycle and usage diagnostics.

`profile.ts` exports a profile factory keyed by a strict profile id. Plan 1
registers only `kazusa-resolver-v1` and rejects unknown profiles. The factory
owns the exact plugin set, settings, system prompt, tools, limits, and policy
for that profile; it is the future seam for a separate least-privilege Plan 2
coding profile. Startup also reports one bounded diagnostic and fails closed
when the loaded DSH core/service graph contains incompatible release versions.
Health, usage, and lifecycle diagnostics are observability surfaces only;
authoritative operation, evidence, and terminal state comes from the typed
Mongo records and supported durable DSH events described below.

Receipt tests register one deterministic evidence-producing tool through a
test-local ToolRuntime fixture. That fixture is absent from the production
profile and package exports, so the released Plan 1 semantic catalog remains
empty apart from `submit_resolution`.

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
the supported `output.presentationMeta(args, value)` and `concludeTurn()` tool
seams. It does not modify DSH loop, session, persistence, LLM, tool, or SDK
source.

### Data Plane

`DSHResolutionIntakeV1` follows the governing architecture and keeps two
strictly separate objects. The canonical wire keys are:

- `runtime`: request/thread identities, priority, deadlines, hard
  budgets, opaque capability token, trusted scope and audience fingerprints,
  resolver profile, tool catalog digest, and policy epoch; and
- `model_input`: bounded objective, constraints, success criteria, known facts,
  uncertainty, approved source literals, and approved continuation material.

Capability tokens, database identifiers, private credentials, approval
authority, ACLs, and workspace/network authority never appear in
`model_input`. The sidecar renders only `model_input` into the stable canonical
DSH waking message.

The sidecar alone imports and interprets DSH Agent, Context, Session, plugin,
event, and receipt APIs. Production Python consumes only the versioned Kazusa
RPC DTOs and never parses DSH events or presentation metadata. This keeps DSH
upgrade adaptation inside the replaceable sidecar.

The smallest Plan 1 model contract is:

| Contract element | Frozen value |
|---|---|
| Semantic question | Given the bounded objective, constraints, approved facts, and model-visible prior tool presentations, select exactly one approved next action or submit the terminal resolution |
| Dynamic model input | `model_input` plus the supported DSH conversation/tool presentation history for the current segment |
| Stable system context | Resolver role, positive action procedure, exact action/tool schemas, output cardinality, and bounded correction instruction from `kazusa-resolver-v1` |
| Model output | Exactly one complete tool call; with Plan 1's empty semantic catalog, the only successful terminal action is `submit_resolution` |
| Deterministic owners | Runtime validates schemas, cardinality, permissions, budgets, evidence, receipts, persistence, retries, fencing, RPC, and execution |
| Rejected complexity | Runtime authority in prompts, hidden database schemas, Brain/RAG/coding context, helper agents, semantic prose parsing, fallback routing, and compatibility adapters |
| Evidence before expansion | Plan 1 intake-to-exhaust, correction, restart, and live gates must pass before Plan 2 adds any tool or profile |

The profile system prompt remains process-stable. Per-operation objectives,
facts, constraints, and continuation material enter only through the DSH
waking message. Correction repeats the same semantic context with a short
contract error and a hard two-attempt cap; deterministic code never invents an
action or converts prose into one.

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

### Durable Receipt Contracts

Kazusa uses only the public DSH `tool/result` event and its presentation
metadata. For every registered non-terminal tool result,
`tool/result.meta.kazusa` contains exactly one `EvidenceReceiptV1`:

```text
kind: "evidence_receipt_v1"
schema_version: "1"
call_id: string
operation_id: string
resolution_thread_id: string
segment_id: string
scope_fingerprint: string
audience_fingerprint: string
policy_epoch: string
tool_name: string
evidence_ids: list[string]
provenance: list[{evidence_id, source_kind, source_id, content_digest}]
evidence_digest: string
```

`evidence_ids` and `provenance` contain at most 64 unique entries and have the
same order and evidence identities. Provenance contains only identifiers,
declared source kind, and digests. `evidence_digest` is the canonical digest of
those ordered bounded fields. Source documents, source text, model reasoning,
bearer tokens, capability tokens, credentials, and ACL material are excluded.

For the successful `submit_resolution` tool result,
`tool/result.meta.kazusa` contains exactly one
`TerminalResolutionReceiptV1`:

```text
kind: "terminal_resolution_v1"
schema_version: "1"
call_id: string
operation_id: string
operation_payload_digest: string
request_id: string
resolution_thread_id: string
segment_id: string
activation_id: string
lease_epoch: integer >= 1
scope_fingerprint: string
audience_fingerprint: string
resolver_profile_version: "kazusa-resolver-v1"
dsh_release: "0.1.1-rc.2"
session_store_epoch: "dsh-sqlite-0.1.1-rc.2-v1"
model_route: string
tool_catalog_digest: string
policy_epoch: string
terminal: SubmitResolutionV1
terminal_digest: string
```

`terminal` is the complete validated architecture-owned
`SubmitResolutionV1` object with no extra keys; its existing field bounds are
the receipt bounds. Receipt `request_id` is the canonical intake request
identity, never the JSON-RPC transport id. `terminal_digest` is the canonical
digest of `terminal`. The enclosing DSH `tool/result` SessionEvent sequence is
the committed sequence and is not duplicated inside metadata before event
allocation. The tool output
uses DSH presentation metadata to persist the receipt alongside that public
event. A terminal exhaust is returned only after the event is durable, the
agent is idle, and the session store flush succeeds. Cold
`resolution.inspect`, duplicate-operation reconciliation, and restart recovery
reconstruct the terminal exhaust from the receipt and never depend on the
transient tool execution return value.

### Thread, Segment, Lease, And Persistence Boundary

The MongoDB owner stores one strict `resolution_thread_store.v1` document per
thread. It contains the architecture's `ResolutionThreadRecordV1`, ordered
`ResolverSessionSegmentV1` records, bounded semantic-operation records, a
monotonic document revision, a monotonic lease epoch, and the current lease.
Each segment records its `session_store_epoch`. Named database helpers own:

- index creation;
- idempotent new-thread creation;
- explicit continuation lookup;
- exact compatibility evaluation inputs;
- segment rotation with a declared reason;
- semantic-operation admission, digest comparison, disposition, and lookup;
- compare-and-set lease acquisition, epoch allocation, renewal, and release;
- sidecar session-sequence/state updates;
- terminal, checkpoint, failure, abandonment, and expiry transitions; and
- read/inspect operations.

The Mongo operation record is a controller-owned request ledger and projection,
with `prepared`, `admitted_active`, `checkpointed`, `terminal`, `canceled`, and
`faulted` states. Preparing the operation reserves its ID and immutable digest;
it does not prove DSH admission. The supported DSH waking-message identity is
the authority for `admitted_active`, and supported committed events/receipts
are the authority for later dispositions. After a normal response or
`resolution.inspect`, the controller advances the Mongo projection by CAS.
Thus a crash between preparation and DSH admission remains safely
`not_admitted`, while a crash after admission is reconstructable without a
second model entry.

Each semantic-operation record contains exactly the operation ID and payload
digest, closed method name, thread and segment IDs, optional activation ID and
lease epoch, optional supported DSH message source ID, disposition, optional
last committed sequence, optional outcome digest, and optional closed fault
code. It stores neither the immutable request payload nor model/tool content;
the terminal payload remains in the durable DSH terminal receipt.

The controller uses repository protocols and imports the narrow public
`kazusa_ai_chatbot.db.resolution_threads` owner. Mongo selectors and update
operators stay in that module. The standalone repository initialization path
creates its indexes before its first operation. The DB schemas and DB ICD
register the new collection, while the global Brain `db_bootstrap()` path
remains unchanged in Plan 1. Mongo stores lifecycle and identity metadata only;
the DSH model/tool transcript remains exclusively in the DSH session store.

DSH remains the source of truth for model/tool history. The sidecar SQLite
backend persists the append-only SessionEvent log at the path derived from the
frozen store epoch. Supported `tool/result.meta.kazusa` evidence receipts hold
bounded provenance and authorization bindings outside model-visible content.
On resume the sidecar rebuilds the evidence ledger and any terminal exhaust
from validated completed `tool/call` and `tool/result` pairs before accepting
further work. An incompatible DSH release or store format selects a fresh
versioned database path and rotates the segment
with a sanitized handoff; Plan 1 performs no cross-epoch SQLite migration.

Every admitted operation has a stable supported DSH waking-message source
identity derived from `operation_id`. The sidecar records admission before
model execution, so replay and inspection can distinguish a request that never
entered DSH from one that is active or durably committed.

### Activation And Step Rules

- The controller acquires a segment lease before `open` or `continue`, assigns
  a unique `activation_id`, and sends that ID with the allocated `lease_epoch`.
  The lease is renewed during a long-running activation and released after the
  activation is disposed.
- A takeover after lease expiry receives a strictly greater `lease_epoch`.
  `continue`, `amend`, `request_checkpoint`, `cancel`, and
  `dispose_activation` reject stale activation or lease identities before
  mutating DSH state.
- When `continue` presents a newly acquired greater epoch for a session that
  still has an older activation, the sidecar fences the old activation
  immediately, drives it to a safe stop, flushes and disposes it, and only then
  creates the new activation. Equal or lower epochs with a different
  activation ID fail closed, so two activations never execute concurrently.
- The sidecar rejects a second live activation for the same DSH session.
- Terminal, checkpointed, canceled, and faulted calls flush and freeze the live
  activation before returning their disposition. The Python controller records
  that disposition in Mongo and then sends idempotent
  `dispose_activation` with the same activation and lease fence. A controller
  failure in this interval is recoverable through inspection and later fenced
  disposal.
- A safe checkpoint stops before the next model step, flushes, returns the last
  committed sequence, and preserves the same session for cold resume.
- A hard interruption uses DSH crash repair. Unknown side-effect outcome
  remains typed and is reconciled through the semantic operation record.
- One model response may contain exactly one complete tool call or one complete
  `submit_resolution` call. A multiple-call response executes zero tool bodies
  or external side effects from that step; DSH may retain denied call/result
  bookkeeping. Multiple-call, zero-call, prose-only, and empty responses each
  receive at most two bounded corrective regenerations and then return
  `runtime_fault` with code `RESOLVER_ACTION_CONTRACT_EXHAUSTED`.
- Terminal evidence ids must exist in the rebuilt ledger and match the current
  thread, segment, scope, audience, and authorization epoch.
- `submit_resolution` validates the full payload, calls `concludeTurn()`, emits
  `terminal_resolution_v1` through DSH presentation metadata, waits for idle,
  flushes, reconstructs exhaust from the durable receipt, returns the committed
  disposition for Mongo recording, and then accepts fenced activation disposal.
- The HTTP server remains responsive to a second authenticated control request
  while the execution request is pending, allowing checkpoint and cancel to
  reach the live Agent at its next safe lifecycle boundary.

## Complete Change Inventory

### Create

| Path | Ownership |
|---|---|
| `sidecars/dsh_resolution/package.json`, `pnpm-lock.yaml`, `tsconfig.json`, `.gitignore`, `README.md` | Exact sidecar dependency, build, runtime, storage, and operator contract |
| `sidecars/dsh_resolution/src/main.ts` | Required configuration, process lifecycle, and loopback HTTP startup |
| `sidecars/dsh_resolution/src/contracts.ts` | Exact RPC, semantic-operation, intake, exhaust, fencing, receipt, checkpoint, and terminal validators |
| `sidecars/dsh_resolution/src/rpc.ts` | Concurrent authenticated JSON-RPC dispatch, semantic duplicate handling, inspection, and bounded errors |
| `sidecars/dsh_resolution/src/operations.ts` | Operation admission/digest registry, DSH message identity, reconciliation, and committed disposition lookup |
| `sidecars/dsh_resolution/src/runtime.ts` | DSH context, fenced Agent activation map, create/resume/amend/checkpoint/cancel/inspect/dispose, and store-epoch path |
| `sidecars/dsh_resolution/src/profile.ts` | Strict profile factory, `kazusa-resolver-v1` composition, dependency-graph diagnostic, model route, action policy, and empty semantic catalog |
| `sidecars/dsh_resolution/src/evidence.ts` | Supported tool-result metadata receipt emission, ledger rebuild, registration, and terminal evidence validation |
| `sidecars/dsh_resolution/src/submit_resolution.ts` | Terminal payload validation, durable terminal receipt, `concludeTurn()`, flush, and exhaust replay |
| `sidecars/dsh_resolution/tests/*.spec.ts` | Direct TypeScript contract, runtime, lifecycle, process, and restart gates |
| `src/agentic_resolver/controller.py` | Resolution lifecycle, compatibility, semantic operation identity, fenced lease renewal, RPC reconciliation, state, and exhaust orchestration |
| `src/agentic_resolver/rpc.py` | Reusable authenticated HTTP JSON-RPC client, operation inspection/reconciliation, and typed transport faults |
| `src/agentic_resolver/persistence.py` | Controller-facing repository protocol and public-DB-helper adapter |
| `src/agentic_resolver/fingerprints.py` | Canonical scope, audience, profile, catalog, and policy fingerprints only |
| `src/agentic_resolver/errors.py` | Strict contract, lifecycle, operation, fencing, persistence, and transport errors |
| `src/kazusa_ai_chatbot/db/resolution_threads.py` | Raw MongoDB collection/index, operation admission, lease-epoch, and CAS owner |
| `tests/test_agentic_resolver_fingerprints.py` | Fingerprint and process-overhead boundary |
| `tests/test_agentic_resolver_evidence.py` | Public tool-result/evidence-reference and exhaust identity checks across restart |
| `tests/test_agentic_resolver_rpc.py` | Versioned authenticated RPC, semantic idempotency, and ambiguous-outcome reconciliation |
| `tests/test_agentic_resolver_controller.py` | Thread, segment, lease, control-plane, and exhaust state machine |
| `tests/test_agentic_resolver_persistence.py` | Repository adapter and durable metadata behavior |
| `tests/test_agentic_resolver_decommission.py` | Legacy absence and current Brain-path non-import gate |
| `tests/test_agentic_resolver_sidecar_process.py` | Real long-lived process, concurrent controls, commit-before-response crash, restart, cold-resume, auth, and isolation gates |
| `tests/test_agentic_resolver_live_db.py` | One explicit real-Mongo lifecycle case |

### Replace In Place

| Path | Replacement |
|---|---|
| `src/agentic_resolver/__init__.py` | Export only canonical contracts, errors, controller, and `AgenticResolverRuntime` |
| `src/agentic_resolver/contracts.py` | Replace every legacy resolver DTO with the canonical Kazusa RPC, intake, exhaust, lifecycle, and record DTOs; expose no DSH event or receipt type |
| `src/agentic_resolver/runtime.py` | Preserve `resolve(...)` while delegating to the new controller and sidecar |
| `src/agentic_resolver/README.md` | Standalone sidecar ICD, construction example, control plane, runtime configuration, and non-Brain boundary |
| `tests/test_agentic_resolver_contracts.py` | Canonical strict data-plane contract tests |
| `tests/test_agentic_resolver_runtime.py` | Preserved standalone entry point and typed exhaust tests |
| `tests/test_agentic_resolver_live_llm.py` | One live DSH terminal-exhaust case |

### Modify

| Path | Change |
|---|---|
| `src/kazusa_ai_chatbot/db/schemas.py` | Add strict thread, segment, operation, activation, lease-epoch, and store document types |
| `src/kazusa_ai_chatbot/db/README.md` | Register collection ownership, public helpers, semantic-operation, fencing, store-epoch, and retention boundaries |
| `src/kazusa_ai_chatbot/llm_interface/{README.md,__init__.py,contracts.py,interface.py,reload.py}` and `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py` | Remove only the unconsumed old resolver native-tool stream API and retain ordinary LLM behavior |
| `pyproject.toml` | Remove `PyYAML`, whose only consumer is the deleted legacy skill design; retain `agentic_resolver*` package discovery |
| `README.md` and `docs/HOWTO.md` | Add independent sidecar install/start/health/stop and standalone invocation instructions |
| `docs/architecture/dsh_integration_architecture.md` | Record the frozen Plan 1 transport, operation identity, supported receipts, persistence/store epoch, release pin, empty semantic catalog, profile seam, and staged cutover decisions |
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
| 2. Durable ownership | Medium | Add Mongo thread/segment/operation/lease storage, monotonic lease epochs, repository adapter, fingerprints, and standalone-only index initialization | Deterministic operation-admission, digest, fencing, renewal, takeover, and CAS tests plus one real-Mongo case pass while global Brain bootstrap remains unchanged |
| 3. Sidecar spine | High | Add the profile factory, exact DSH composition, concurrent loopback JSON-RPC server, versioned SQLite store, supported evidence/terminal receipts, terminal tool, and empty semantic catalog | Typecheck/build/direct TypeScript tests pass; one independent sidecar serves health and two sessions; receipt replay and dependency-graph diagnostics pass |
| 4. Standalone controller | High | Add reusable RPC client, semantic operation/reconciliation state machine, preserved runtime entry point, fenced controls, lease renewal, and typed exhaust | Python unit tests pass for duplicate/digest behavior, ambiguous disconnect inspection, open/continue/amend/checkpoint/cancel/inspect/dispose, and terminal/checkpoint/fault exhaust |
| 5. Restart and adversarial lifecycle | High | Add cold evidence and terminal replay, commit-before-response recovery, lease/live-activation fencing, concurrent controls, mismatch/store rotation, zero/multi-call correction, evidence validation, and crash classification | Real-process restart/security/fault tests prove one admission, zero invalid-step side effects, responsive lifecycle control, and exact terminal replay with no model rerun, semantic tool, or Brain edge |
| 6. Operational and closure gate | Small | Final docs, manifest, exact lockfile, mapped regressions, deterministic suite, live DB, and one live LLM case | Every release blocker below is green and the parent completes final architecture sign-off |

Block completion records only its functional result and command evidence. It
does not create an additional review cycle.

## Test Impact And Traceability

The Python nodes below are the release-gating cross-process or owner-contract
tests. Direct Vitest nodes supplement them at the DSH ownership boundary.

| Production source path | Changed symbol or contract | Semantic owner | Exact deterministic pytest node | Supplemental direct node | Mode / regression |
|---|---|---|---|---|---|
| `sidecars/dsh_resolution/src/main.ts` | `loadConfig`, `createRpcServer`, versioned store path, shutdown flush | sidecar process boundary | `tests/test_agentic_resolver_sidecar_process.py::test_sidecar_requires_loopback_auth_data_root_model_and_versioned_store_path` | `process.spec.ts > serves one long-lived independent process across multiple sessions` | cross-process / independent-process regression |
| `sidecars/dsh_resolution/src/contracts.ts` | intake `runtime`, operation/fence frames, receipts, exhaust | sidecar contract boundary | `tests/test_agentic_resolver_sidecar_process.py::test_missing_or_invalid_terminal_receipt_never_returns_terminal_exhaust` | `contracts.spec.ts > validates exact bounded evidence and terminal receipt metadata` | direct unit plus black-box process / interface regression |
| `sidecars/dsh_resolution/src/rpc.ts` | concurrent dispatch and semantic duplicate rules | sidecar transport boundary | `tests/test_agentic_resolver_rpc.py::test_same_operation_id_and_digest_reconciles_one_admission` | `rpc.spec.ts > serves checkpoint and cancel concurrently with a pending execution request` | cross-process / replay and control regression |
| `sidecars/dsh_resolution/src/operations.ts` | `OperationRegistry.admit`, `inspect`, and committed disposition | sidecar operation boundary | `tests/test_agentic_resolver_rpc.py::test_operation_id_reuse_with_different_digest_fails_closed` | `rpc.spec.ts > inspects not admitted active and committed outcomes after transport loss` | unit plus cross-process / idempotency regression |
| `sidecars/dsh_resolution/src/runtime.ts` | `ResolutionSidecarRuntime`, activation map, fencing, store epoch | DSH lifecycle boundary | `tests/test_agentic_resolver_sidecar_process.py::test_second_http_request_checkpoints_and_cancels_pending_execution` | `runtime.spec.ts > rejects stale activation and lease epochs before DSH mutation` | cross-process / lifecycle regression |
| `sidecars/dsh_resolution/src/profile.ts` | strict `buildProfile` factory and dependency diagnostic | DSH composition boundary | `tests/test_agentic_resolver_sidecar_process.py::test_sidecar_profile_factory_and_dependency_graph_fail_closed` | `profile.spec.ts > builds only the resolver profile through the strict profile factory` | cross-process plus unit / least-privilege regression |
| `sidecars/dsh_resolution/src/evidence.ts` | `EvidenceLedger` and `evidence_receipt_v1` metadata | evidence-registration boundary | `tests/test_agentic_resolver_evidence.py::test_public_exhaust_preserves_validated_evidence_references_after_sidecar_restart` | `evidence.spec.ts > rebuilds authorized evidence from supported tool result metadata after restart` | black-box restart plus direct unit / evidence-integrity regression |
| `sidecars/dsh_resolution/src/submit_resolution.ts` | terminal validator, `terminal_resolution_v1`, commit ordering, replay | terminal-operation boundary | `tests/test_agentic_resolver_sidecar_process.py::test_kill_after_terminal_commit_before_http_response_replays_exact_exhaust` | `submit_resolution.spec.ts > commits the complete terminal receipt before returning exhaust` | cross-process fault injection / exactly-once terminal regression |
| `src/agentic_resolver/contracts.py` | canonical RPC, intake, records, and exhaust without DSH event/receipt types | Python public data-plane boundary | `tests/test_agentic_resolver_contracts.py::test_public_contracts_expose_no_dsh_event_or_receipt_types` | none | unit / sidecar-isolation regression |
| `src/agentic_resolver/controller.py` | `ResolutionController`, operation reconciliation, lease renewal and fencing | Python lifecycle boundary | `tests/test_agentic_resolver_controller.py::test_stale_activation_or_lease_epoch_rejects_every_live_control` | none | unit / state-machine regression |
| `src/agentic_resolver/rpc.py` | `DSHRpcClient.call`, `inspect_operation`, ambiguous-outcome state machine | Python transport boundary | `tests/test_agentic_resolver_rpc.py::test_disconnect_after_terminal_commit_reconciles_exact_exhaust_without_model_call` | none | unit plus cross-process / disconnect regression |
| `src/agentic_resolver/persistence.py` | `ResolutionThreadRepository` protocol and DB adapter | controller persistence boundary | `tests/test_agentic_resolver_persistence.py::test_operation_admission_is_idempotent_only_for_matching_digest` | none | unit / repository regression |
| `src/agentic_resolver/fingerprints.py` | canonical authority fingerprints and operation payload digest | authority identity boundary | `tests/test_agentic_resolver_fingerprints.py::test_operation_payload_digest_is_canonical_and_excludes_transport_fields` | none | unit / identity regression |
| `src/agentic_resolver/errors.py` | closed contract, operation, fence, and runtime-fault codes | error contract boundary | `tests/test_agentic_resolver_contracts.py::test_typed_operation_fence_and_runtime_fault_codes_are_closed` | none | unit / fault-contract regression |
| `src/agentic_resolver/runtime.py` | preserved `AgenticResolverRuntime.resolve` entry point | standalone public API boundary | `tests/test_agentic_resolver_runtime.py::test_resolve_preserves_standalone_entrypoint_and_returns_typed_exhaust` | none | unit plus cross-process / entry-point regression |
| `src/agentic_resolver/__init__.py` | canonical public exports | package boundary | `tests/test_agentic_resolver_decommission.py::test_old_resolver_contracts_facades_and_aliases_are_absent` | none | static import / compatibility-removal regression |
| `src/kazusa_ai_chatbot/db/schemas.py` | thread, segment, operation, activation, lease/store epoch types | shared DB schema boundary | `tests/test_agentic_resolver_contracts.py::test_thread_segment_operation_activation_lease_and_store_epoch_validate` | none | unit / schema regression |
| `src/kazusa_ai_chatbot/db/resolution_threads.py` | operation admission and lease-epoch CAS helpers | raw MongoDB boundary | `tests/test_agentic_resolver_persistence.py::test_thread_segment_operation_lease_epoch_and_store_epoch_round_trip` | none | repository unit plus explicit live DB / CAS regression |
| `src/kazusa_ai_chatbot/llm_interface/contracts.py` | remove resolver-only stream DTOs | retained ordinary LLM boundary | `tests/test_agentic_resolver_decommission.py::test_old_native_tool_stream_surface_is_absent` | none | static plus preserved LLM regression |
| `src/kazusa_ai_chatbot/llm_interface/__init__.py` | remove resolver-only stream exports | retained ordinary LLM package boundary | `tests/test_agentic_resolver_decommission.py::test_old_native_tool_stream_surface_is_absent` | none | static import plus preserved LLM regression |
| `src/kazusa_ai_chatbot/llm_interface/interface.py` | remove `astream_tools` | retained ordinary LLM boundary | `tests/test_agentic_resolver_decommission.py::test_old_native_tool_stream_surface_is_absent` | none | static plus preserved LLM regression |
| `src/kazusa_ai_chatbot/llm_interface/reload.py` | remove resolver-only reload plumbing | retained ordinary LLM boundary | `tests/test_llm_interface_reload.py::test_async_unload_error_retries_same_call_once` | none | unit / ordinary reload regression |
| `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py` | remove resolver-only tool-stream binding | retained provider boundary | `tests/test_llm_interface_openai_provider.py::test_provider_maps_config_to_chat_model_constructor` | none | unit / ordinary provider regression |
| `pyproject.toml` | remove legacy resolver-only `PyYAML` dependency | package/dependency boundary | `tests/test_agentic_resolver_decommission.py::test_legacy_resolver_dependency_and_package_files_are_absent` | none | static configuration regression |

### Exact Verification Matrix

#### TypeScript Direct Tests

The worker implements and passes these exact Vitest titles:

```text
sidecars/dsh_resolution/tests/rpc.spec.ts > authenticates versioned request and response frames
sidecars/dsh_resolution/tests/rpc.spec.ts > rejects non-loopback unauthenticated unknown and malformed requests
sidecars/dsh_resolution/tests/rpc.spec.ts > admits one operation for duplicate ids with the same payload digest
sidecars/dsh_resolution/tests/rpc.spec.ts > rejects operation id reuse with a different payload digest
sidecars/dsh_resolution/tests/rpc.spec.ts > inspects not admitted active and committed outcomes after transport loss
sidecars/dsh_resolution/tests/rpc.spec.ts > serves checkpoint and cancel concurrently with a pending execution request
sidecars/dsh_resolution/tests/contracts.spec.ts > separates canonical runtime from model input
sidecars/dsh_resolution/tests/contracts.spec.ts > validates status-specific submit_resolution and exhaust
sidecars/dsh_resolution/tests/contracts.spec.ts > validates exact bounded evidence and terminal receipt metadata
sidecars/dsh_resolution/tests/contracts.spec.ts > requires operation activation and lease fencing on live mutations
sidecars/dsh_resolution/tests/evidence.spec.ts > rebuilds authorized evidence from supported tool result metadata after restart
sidecars/dsh_resolution/tests/evidence.spec.ts > declares no custom session event kind in the production profile
sidecars/dsh_resolution/tests/evidence.spec.ts > excludes source content credentials capability tokens and ACLs from receipts
sidecars/dsh_resolution/tests/evidence.spec.ts > rejects unknown cross-scope and cross-segment evidence
sidecars/dsh_resolution/tests/submit_resolution.spec.ts > commits the complete terminal receipt before returning exhaust
sidecars/dsh_resolution/tests/submit_resolution.spec.ts > replays exact terminal exhaust after restart without model execution
sidecars/dsh_resolution/tests/submit_resolution.spec.ts > rejects missing or invalid terminal receipts as runtime faults
sidecars/dsh_resolution/tests/runtime.spec.ts > opens an agent captures terminal submit and disposes activation
sidecars/dsh_resolution/tests/runtime.spec.ts > checkpoints at pre-step and resumes the same DSH session
sidecars/dsh_resolution/tests/runtime.spec.ts > amends with steer queues continuation and cancels safely
sidecars/dsh_resolution/tests/runtime.spec.ts > accepts one regenerated action after a zero-call prose-only or empty step
sidecars/dsh_resolution/tests/runtime.spec.ts > returns action contract exhausted after repeated zero-call prose-only or empty steps
sidecars/dsh_resolution/tests/runtime.spec.ts > executes zero tool bodies for multi-call output and exhausts the shared correction budget
sidecars/dsh_resolution/tests/runtime.spec.ts > rejects stale activation and lease epochs before DSH mutation
sidecars/dsh_resolution/tests/profile.spec.ts > builds only the resolver profile through the strict profile factory
sidecars/dsh_resolution/tests/profile.spec.ts > fails startup for an incompatible DSH dependency graph
sidecars/dsh_resolution/tests/lifecycle.spec.ts > rotates on scope audience profile release store model catalog or policy mismatch
sidecars/dsh_resolution/tests/lifecycle.spec.ts > rejects duplicate live activation
sidecars/dsh_resolution/tests/lifecycle.spec.ts > renews a live lease and assigns a higher epoch after expired takeover
sidecars/dsh_resolution/tests/lifecycle.spec.ts > cancels without deleting durable session
sidecars/dsh_resolution/tests/process.spec.ts > serves one long-lived independent process across multiple sessions
sidecars/dsh_resolution/tests/process.spec.ts > restarts and cold-resumes evidence from the versioned session store
sidecars/dsh_resolution/tests/process.spec.ts > recovers terminal receipt when killed after commit before rpc response
sidecars/dsh_resolution/tests/process.spec.ts > reconciles admitted operation after controller restart without model re-entry
```

Final commands:

```powershell
pnpm --dir sidecars/dsh_resolution install --frozen-lockfile
pnpm --dir sidecars/dsh_resolution run typecheck
pnpm --dir sidecars/dsh_resolution run build
pnpm --dir sidecars/dsh_resolution test
```

#### Python Deterministic Tests

The worker implements and passes these exact node IDs:

```text
tests/test_agentic_resolver_contracts.py::test_intake_v1_rejects_unknown_fields_and_keeps_runtime_out_of_model_input
tests/test_agentic_resolver_contracts.py::test_thread_segment_operation_activation_lease_and_store_epoch_validate
tests/test_agentic_resolver_contracts.py::test_submit_resolution_requires_status_specific_fields
tests/test_agentic_resolver_contracts.py::test_public_contracts_expose_no_dsh_event_or_receipt_types
tests/test_agentic_resolver_contracts.py::test_typed_operation_fence_and_runtime_fault_codes_are_closed
tests/test_agentic_resolver_fingerprints.py::test_authority_fingerprints_are_stable_and_do_not_hash_event_logs
tests/test_agentic_resolver_fingerprints.py::test_operation_payload_digest_is_canonical_and_excludes_transport_fields
tests/test_agentic_resolver_evidence.py::test_standard_tool_result_contract_binds_thread_segment_scope_and_audience
tests/test_agentic_resolver_evidence.py::test_public_exhaust_preserves_validated_evidence_references_after_sidecar_restart
tests/test_agentic_resolver_evidence.py::test_production_profile_uses_no_custom_session_event_kind
tests/test_agentic_resolver_evidence.py::test_exhaust_rejects_evidence_with_mismatched_thread_segment_scope_or_audience
tests/test_agentic_resolver_rpc.py::test_versioned_authenticated_rpc_rejects_bad_version_token_and_method
tests/test_agentic_resolver_rpc.py::test_rpc_round_trip_preserves_typed_intake_and_exhaust
tests/test_agentic_resolver_rpc.py::test_same_operation_id_and_digest_reconciles_one_admission
tests/test_agentic_resolver_rpc.py::test_operation_id_reuse_with_different_digest_fails_closed
tests/test_agentic_resolver_rpc.py::test_disconnect_before_admission_inspects_then_replays_same_operation_once
tests/test_agentic_resolver_rpc.py::test_disconnect_after_admission_attaches_to_active_operation
tests/test_agentic_resolver_rpc.py::test_disconnect_after_terminal_commit_reconciles_exact_exhaust_without_model_call
tests/test_agentic_resolver_rpc.py::test_controller_restart_reconciles_admitted_operation_without_duplicate_model_entry
tests/test_agentic_resolver_rpc.py::test_unknown_operation_outcome_returns_uncertain_fault_without_new_admission
tests/test_agentic_resolver_controller.py::test_open_creates_one_thread_segment_activation_and_lease_epoch
tests/test_agentic_resolver_controller.py::test_continue_reuses_segment_only_for_same_goal_scope_audience_and_epoch
tests/test_agentic_resolver_controller.py::test_incompatible_scope_audience_profile_release_store_model_catalog_or_policy_rotates_segment
tests/test_agentic_resolver_controller.py::test_duplicate_execution_lease_fails_closed
tests/test_agentic_resolver_controller.py::test_long_activation_renews_lease_and_expired_takeover_increments_epoch
tests/test_agentic_resolver_controller.py::test_stale_activation_or_lease_epoch_rejects_every_live_control
tests/test_agentic_resolver_controller.py::test_amend_steers_in_flight_work_and_followup_queues_next_turn
tests/test_agentic_resolver_controller.py::test_checkpoint_cancel_inspect_and_dispose_preserve_durable_lineage
tests/test_agentic_resolver_controller.py::test_terminal_submit_maps_to_typed_terminal_exhaust
tests/test_agentic_resolver_controller.py::test_checkpoint_maps_to_runtime_owned_checkpointed_exhaust
tests/test_agentic_resolver_persistence.py::test_thread_segment_operation_lease_epoch_and_store_epoch_round_trip
tests/test_agentic_resolver_persistence.py::test_operation_admission_is_idempotent_only_for_matching_digest
tests/test_agentic_resolver_persistence.py::test_cold_resume_uses_persisted_session_reference_and_revision
tests/test_agentic_resolver_persistence.py::test_expired_or_corrupt_segment_fails_closed_or_rotates
tests/test_agentic_resolver_runtime.py::test_resolve_preserves_standalone_entrypoint_and_returns_typed_exhaust
tests/test_agentic_resolver_runtime.py::test_runtime_has_no_brain_task_resolution_rag_or_coding_import_edge
tests/test_agentic_resolver_decommission.py::test_old_resolver_contracts_facades_and_aliases_are_absent
tests/test_agentic_resolver_decommission.py::test_brain_task_resolution_rag_and_coding_paths_do_not_import_or_spawn_resolver
tests/test_agentic_resolver_decommission.py::test_old_native_tool_stream_surface_is_absent
tests/test_agentic_resolver_decommission.py::test_legacy_resolver_dependency_and_package_files_are_absent
tests/test_agentic_resolver_sidecar_process.py::test_standalone_runtime_uses_one_long_lived_sidecar_across_two_resolves
tests/test_agentic_resolver_sidecar_process.py::test_sidecar_requires_loopback_auth_data_root_model_and_versioned_store_path
tests/test_agentic_resolver_sidecar_process.py::test_sidecar_profile_factory_and_dependency_graph_fail_closed
tests/test_agentic_resolver_sidecar_process.py::test_brain_and_sidecar_run_independently_and_sidecar_stop_does_not_stop_brain
tests/test_agentic_resolver_sidecar_process.py::test_sidecar_restart_preserves_checkpoint_and_cold_resumes
tests/test_agentic_resolver_sidecar_process.py::test_scope_audience_profile_release_store_model_catalog_and_policy_mismatch_rotates
tests/test_agentic_resolver_sidecar_process.py::test_second_http_request_checkpoints_and_cancels_pending_execution
tests/test_agentic_resolver_sidecar_process.py::test_zero_call_and_multi_call_steps_execute_no_tool_body_and_exhaust_correction_budget
tests/test_agentic_resolver_sidecar_process.py::test_kill_after_terminal_commit_before_http_response_replays_exact_exhaust
tests/test_agentic_resolver_sidecar_process.py::test_missing_or_invalid_terminal_receipt_never_returns_terminal_exhaust
tests/test_agentic_resolver_sidecar_process.py::test_terminal_exhaust_contains_only_validated_submit_resolution_and_evidence_refs
tests/test_agentic_resolver_sidecar_process.py::test_bad_rpc_authentication_version_and_schema_fail_closed
```

The mapped deterministic command is:

```powershell
venv\Scripts\python -m pytest tests/test_agentic_resolver_contracts.py tests/test_agentic_resolver_fingerprints.py tests/test_agentic_resolver_evidence.py tests/test_agentic_resolver_rpc.py tests/test_agentic_resolver_controller.py tests/test_agentic_resolver_persistence.py tests/test_agentic_resolver_runtime.py tests/test_agentic_resolver_decommission.py tests/test_agentic_resolver_sidecar_process.py
```

#### Preserved-System Regression

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

#### Explicit Live Cases

Run each live case alone and inspect its complete output:

```powershell
venv\Scripts\python -m pytest -m live_db -s tests/test_agentic_resolver_live_db.py::test_resolution_thread_store_enforces_operation_idempotency_lease_fencing_rotation_and_cold_resume
venv\Scripts\python -m pytest -m live_llm -s tests/test_agentic_resolver_live_llm.py::test_live_standalone_sidecar_resolution_reaches_submit_resolution
```

The live LLM case uses the real sidecar, real DSH loop and persistence, real
model route, canonical intake, and terminal tool. It passes only when the
returned exhaust is terminal, schema-valid, identity-consistent, reconstructed
from a durable `terminal_resolution_v1` receipt, contains no assistant prose as
authority, and its activation has been disposed after a durable flush.

## Release-Blocking Gates

| Gate | Green condition |
|---|---|
| P1-G1 — Independent process | One long-lived sidecar handles multiple sessions, accepts concurrent lifecycle control, and restarts independently; stopping it produces a typed standalone runtime fault and leaves the Brain process/path unaffected |
| P1-G2 — Canonical and idempotent interface | Every control plus intake/exhaust is versioned, authenticated, strict, and shared by Python and TypeScript; duplicate operation identity admits once for an equal digest, rejects a changed digest, and reconciles every ambiguous disconnect through inspection |
| P1-G3 — Durable and fenced lifecycle | New, continuation, checkpoint, cold resume, mismatch/store-epoch rotation, cancel, inspect, terminal, crash repair, lease renewal/release, monotonic takeover, stale-control rejection, and activation disposal preserve one coherent thread lineage |
| P1-G4 — Authority, action, and terminal safety | Canonical `runtime` stays outside model input; invalid zero/multi-call steps execute zero tool bodies; evidence comes from supported bounded receipts and is scope-bound; a complete durable terminal receipt is the sole source of terminal exhaust, including commit-before-response restart replay |
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
