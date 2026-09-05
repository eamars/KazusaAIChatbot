# DSH Integration Boundary And Test Contract Remediation

## Document Control

- **Status:** superseded by explicit owner direction on 2026-09-05.
- **Successor:** [DSH runtime completion](../../../active/bugfix/dsh_runtime_completion_plan_2026-09-05.md).
  The successor covers the entire existing and new DSH codebase. Historical
  test matrices and readiness-before-live gates below are retired; functional
  obligations and evidence continue under the successor.
- **Created:** 2026-09-05.
- **Parent:** [DSH cleanup and final sign-off plan](dsh_probe_first_architecture_cleanup_and_final_signoff_plan_2026-09-04.md).
- **Direction:** retain the DSH integration architecture, correct cancellation
  ownership, remove test implementations from production, and replace redundant
  or misleading oracles with evidence at Kazusa-owned interfaces.
- **User authority for this review:** inspect code/tests, run bounded checks
  excluding the actual E2E test, and amend or add plans. The 2026-09-05 work is
  a review and planning slice. Historical implementation authority in the
  parent is not authority to execute the new production scope below.
- **Acceptance state:** remediation required. Parent Gates 3, 5, and 6 are
  reopened. Historical passes remain evidence of the earlier checks; final
  sign-off requires the remedies below plus separately authorized live E2E
  evidence. Model availability alone does not open the E2E execution gate.

## Current Execution Authority

The 2026-09-05 user instruction explicitly commands implementation of this
plan and execution and closure of the remaining DSH plans, including real LLM
tests. This supersedes the earlier review-only and live-execution exclusions.
The independent non-E2E readiness gate still precedes live testing. Existing
environment-file, production-data, and deployment exclusions remain in force.
The parent owns implementation and lifecycle evidence; separate dynamically
resolved reviewers own independent readiness and per-case behavior acceptance.
Baseline status, full HEAD diff, and owned-surface hashes are retained under
`test_artifacts/dsh_execution_20260905/`.

## Review Conclusions And Evidence

The intended ownership is sound: cognition admits and judges; task resolution
owns the task lifecycle; the controller owns operation identity and lease
fencing; the sidecar integrates DSH Standard; semantic tools return evidence;
cognition and L3/dialog determine the response; the dispatcher delivers it.
The V2 observation reference removes duplicated semantic state in the reviewed
diff. These are architectural strengths, rather than sufficient release proof.

### R1 — High: foreground cancellation leaves execution without a task owner

`src/kazusa_ai_chatbot/task_resolution/service.py:206-212` shields the async
`runtime.open` task and catches timeout only. The outer handler at lines
91-110 catches `Exception`, which excludes `asyncio.CancelledError`. The
second awaitable path at lines 318-369 repeats the ownership problem.

The deterministic public-service reproduction observed:

```json
{
  "caller_cancelled": true,
  "runtime_still_pending": true,
  "runtime_cancel_calls": 0,
  "binding_state": "opening",
  "accepted_task_id": null,
  "background_work_job_id": null,
  "runtime_continued_after_caller_cancellation": true
}
```

Evidence: `test_artifacts/dsh_plan_review_20260905/cancellation_probe.py` and
its adjacent JSON. Reproduce from the repository root with
`venv\Scripts\python -m test_artifacts.dsh_plan_review_20260905.cancellation_probe`.
It calls the real task service with a controlled async runtime and in-memory
binding collaborator, then releases and joins every created task. It uses
synthetic import configuration with dotenv disabled; it starts no service,
contacts no model/database, and makes no E2E claim.

The lower transport also uses `asyncio.to_thread(urlopen)` with unlimited
read duration for open/continue (`src/agentic_resolver/rpc.py:300-326`).
Canceling the Python future alone does not establish that its blocking HTTP
operation has stopped. Cancellation closure therefore includes the Kazusa
transport wrapper, rather than merely deleting the resolver's task registry.

### R2 — High: production contains test implementations and test-shaped APIs

Current definitions and repository callers establish these boundaries:

| Production surface | Evidence and disposition |
|---|---|
| `src/agentic_resolver/rpc.py:72-298` | `_Operation`, `_OperationRegistry`, `DSHRpcServer`, and `InMemoryRpcTransport` implement a second, fake server for tests. Replace them with small scripted responses in test support. |
| `src/agentic_resolver/rpc.py:400-433` | `call_sync` and `reconcile_sync` are used only by RPC tests. Convert those callers to the existing async API and remove the test-only convenience methods. |
| `src/agentic_resolver/persistence.py:52-353` | `InMemoryResolutionThreadRepository` duplicates repository lifecycle behavior; production construction uses `MongoResolutionThreadRepository`. Relocate the minimum controller double into test support. |
| `src/kazusa_ai_chatbot/dsh_interaction/auth.py:27-44` | `InteractionNonceReplayStore` is explicitly a deterministic test owner. Move it to test support. |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/authority.py:341-359` | `InMemorySemanticAuthorityReplayOwner` and `snapshot` serve test restart simulation. Move them to test support. |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/worker.py:63-96` | `InMemorySemanticOutcomeOwner` has no repository caller. Remove it. |
| `src/kazusa_ai_chatbot/task_resolution/service.py:820-824,1473-1477` | Both `authority_broker` parameters are immediately discarded; only tests pass them. Remove the parameters and their callers. |

Keep actual dependency interfaces and production carriers. For example,
`SemanticWorkerCall` is used by `_production_handler` despite its misleading
test-oriented docstring; retain it with a description of its runtime role.
Dependency injection itself is useful. Production branches that exist solely
to accommodate test doubles are the cleanup target.

The same problem crosses the language boundary:
`sidecars/dsh_resolution/src/profile.ts:862-868` calls `process.exit(97)` when
`NODE_ENV=test` and `KAZUSA_DSH_TEST_EXIT_AFTER_TERMINAL_COMMIT=1`. The probe
depends on this production test switch. Remove it and inject response loss
outside the production process as specified below.

`sidecars/dsh_resolution/src/runtime.ts:14-21,98-141,176-222,292-379`
contains a scripted alternative executor selected by `forTests` and
`restoreForTests`. Its session/lease/event simulator, correction loop, and
fake interaction deferral are separate from the `forProduction` path used by
`main.ts:393`. Every case in `runtime.spec.ts` and `lifecycle.spec.ts` exercises
that simulator. `src/terminal_policy.ts` is consumed only by the simulator and
its own test/re-export. Remove that dead alternative, keeping real operation
deduplication, in-flight joining, durable inspection, and control forwarding.
Also move the subprocess body of `SecretBroker.runNativeProbe`
(`src/secret_broker.ts:26,51-78`) to test support: its sole caller is the secret
isolation test. Retain `nativeEnvironment` and the real credential provider.

### R3 — High: several green tests establish the behavior of their own doubles

- `tests/test_agentic_resolver_persistence.py:61-157` tests the in-memory
  implementation, including a purported cold resume that reads the same
  object. Its results establish neither Mongo persistence nor restart.
- `tests/test_agentic_resolver_rpc.py:72-87` validates the fake server rather
  than the shipped TypeScript RPC boundary. Its admission/execution counters
  belong to that fake. In `rpc.py:287-296`, both post-admission and post-commit
  fault flags execute the full synchronous dispatch before raising, so they
  fail to distinguish the advertised active and committed states.
- `tests/unit/task_resolution/test_service.py:98-139,255-282` returns a Task
  from an async `open`, while `AgenticResolverRuntime.open` returns the final
  exhaust after one await. This exercises the test-shaped second awaitable
  path and misses real caller cancellation.
- `tests/unit/task_resolution/test_service.py:290-328` manually invokes its
  own authority broker after the service discarded that broker. That is not
  evidence of claim-time authority issuance by the production worker.
- Six sibling-test imports remain in the Brain interaction auth, service,
  and decision tests and semantic worker, conversation, and recall/calendar
  tests. The parent's broad fixture-decoupling completion claim is therefore
  narrower than the actual DSH surface.
- `sidecars/dsh_resolution/tests/runtime.spec.ts:4` and
  `submit_resolution.spec.ts:5` import builders from `contracts.spec.ts`,
  which also registers tests. Relocate those builders to a neutral test helper
  so importing fixture data does not execute another suite's registrations.

Retain strict checks for wire shape, same-operation reconciliation, scope,
authority, durable transitions, and delivery idempotency. Rewrite tests around
those outcomes; avoid replacing them with additional tests of helpers.

### R4 — High: live behavior coverage and its oracle remain incomplete

`tests/dsh_behavior_e2e_support.py:719-775` accepts foreground behavior from
backend evidence containing `mira` plus nonempty messages, deferred behavior
from a result/receipt count, and an internal unsupported-claim judgment from
any decision enum. Source-owned literals can be useful checks, but matching a
name in backend JSON does not establish that visible output is grounded.

The three wrappers currently hide all scenario inputs and expectations in one
helper. Technical success and the existing pending qualitative review must
remain distinct. Make each scenario's input, hard gates, behavioral rubric,
acceptable variation, and forbidden outcomes visible together. Evaluate the
answer actually sent to the user and the actual internal judgment.

The independent test audit also found private DB client reconfiguration at
`tests/dsh_behavior_e2e_support.py:348-393`, raw collection capture at lines
649-675, and unconditional sidecar/adapter startup at lines 838-864 even for
the internal case, which directly posts to Brain. Storage layout is diagnostic
detail; the Brain interaction case needs only its actual collaborators.

### R5 — High: upstream composition details remain acceptance oracles

`sidecars/dsh_resolution/tests/profile.spec.ts:43-53,103-128` pins composed
DSH services and `node_modules` path structure.
`sidecars/dsh_resolution/tests/standard_profile.spec.ts:8-43,106-123` checks
composition dumps, installation paths, third-party row IDs, and nested
provider configuration. Remove those implementation oracles. Retain tests of
Kazusa's compatibility rejection, tool collision policy, route digest, and
actual provider request mapping. A declared supported release/version check
is a legitimate integration policy; testing package installation layout is
not required to preserve it.

### R6 — Medium: process evidence is duplicated and pins provider sequence

`experiments/dsh_runtime_probe.py:790-981` already exercises semantic forwarding,
checkpoint/restart, and response-loss replay. The same paths run again in
`tests/test_agentic_resolver_sidecar_process.py:49-221` and the real-sidecar
case at `tests/test_dsh_tool_gateway_worker.py:341-436`. The latter also indexes
`provider.requests[1]` and checks call count. The CLI integration wrapper at
`tests/integration/test_dsh_runtime_probe.py:54-64` asserts an exact observation
label set. Consolidate each public outcome into one process oracle and use
CLI/result evidence across entrypoints. Preserve real semantic forwarding,
auth rejection, session reuse, restart, and replay proof during deletion.

## Fixed Target Contracts

### Lifecycle and async ownership

1. Preserve one asynchronous `AgenticResolverRuntime.open` result: awaiting
   it yields the typed exhaust. Test doubles use that same call shape. Remove
   synchronous-open and nested-awaitable compatibility paths from the task
   service after updating every repository caller.
2. Task resolution owns the foreground budget, checkpoint race, caller
   cancellation, and final cleanup. Budget expiry requests a cooperative
   checkpoint and preserves healthy reasoning until durable handoff settles.
3. Caller cancellation before committed handoff invokes the existing fenced
   cancellation/disposal and reconciliation interfaces. Settle a definitive
   canceled or already-committed outcome in the binding, or preserve a typed
   fault/uncertain recovery outcome when remote disposition cannot be proved.
   Then propagate `CancelledError`. Preserve a committed terminal result when
   cancellation loses the race. A definitive local return owns and joins its
   Python tasks and HTTP resources.
4. After a committed background handoff, the accepted-task/job owner retains
   execution; foreground cancellation preserves that durable ownership.
   A checkpoint reference alone is not proof of accepted-task/job ownership.
   Exercise this rule on `promote_deferred_task_resolution` itself: it owns
   the separate awaited promotion invoked by `capabilities.py:858`. Handle
   cancellation during partial promotion with the existing accepted-task
   enqueue-failure and binding fault/fence boundaries; a queued generation
   without a committed owner must fail the worker's existing binding check.
   Inspect durable attachments before cleanup so cancellation immediately
   after the final write preserves an already committed owner. Keep the
   capability caller's semantic decisions and result shape unchanged.
5. Replace the blocking HTTP wrapper with the already-declared `httpx` async
   dependency so cancellation closes the request resource. Preserve unlimited
   healthy open/continue completion time and the existing bounded control RPC
   policy. Use those control bounds for cancellation cleanup. Exercise this
   wrapper against a local controlled HTTP peer, not DSH internals.
6. Controller repository calls follow the production async repository shape.
   Correct the existing `ResolutionThreadRepository` protocol to declare the
   operations the controller actually uses, and make the test double async.
   Remove sync/async branching from `_repository_call`; retain existing
   durable selectors and CAS/lease authority in the DB owner.
7. `ResolutionSidecarRuntime.forProduction` remains the runtime construction
   boundary used by `main.ts`. Require its real executor/control dependencies;
   remove `forTests`, `restoreForTests`, `ScriptStep`, test session storage,
   `compatibleSegment`, fake interaction deferral, test counters, simulated
   lease methods, and script-only helpers. Tests inject controlled collaborators
   through `forProduction` and observe real deduplication/forwarding behavior.
   Keep `OperationRegistry` because both shipped RPC and runtime consume it.
   Remove `terminal_policy.ts` and its re-export because their only execution
   consumer is the deleted simulator; the shipped profile and terminal receipt
   owner retain their behavior.

### Test ownership and minimality

- Production contains runtime implementations, public contracts, and required
  injection interfaces. Test support contains fixtures, scripted transports,
  and controlled repositories. Move callers in one cutover with zero aliases.
- An RPC unit test drives `DSHRpcClient` with one scripted public response or
  transport exception per call. Exercise distinct `not_admitted`,
  `admitted_active`, committed, and `unknown` observations, checking whether
  the client resubmits the same operation or reports uncertainty. The fake
  supplies observations; it does not implement a parallel server/ledger.
- Test the real Mongo adapter's error translation directly. Keep persistence,
  fencing, and restart evidence in the existing guarded Mongo integration
  test. Remove tests whose sole subject is a relocated fake repository.
  Patch the lower owner's public `resolution_threads.ensure_indexes` and
  `resolution_threads.get_operation` functions in adapter tests; use the normal
  `MongoResolutionThreadRepository()` constructor, with no `_db` reassignment
  or new production injection API. Migrate `test_rpc_readiness` to public
  `DSHRpcClient.call("system.health", ...)` for transport failure evidence.
- Verify queued payload authority exclusion at the admission boundary and
  fresh authority at the existing runtime/worker claim boundary. Each assertion
  observes the owning production call, rather than an explicit call to a fake.
- Keep small shared builders in support modules. Each test owns its scenario
  and outcome assertions. Production source text, private helper names, stage
  order, exact test counts, and test artifact filenames carry no acceptance
  authority. Exact operation identity and one eligible delivery remain valid
  product invariants.

### Third-party boundary

DSH is a third-party execution dependency. Retain checks for the integration
code this repository ships: authenticated RPC, intake/exhaust mapping,
semantic-worker forwarding, approval authority, workspace restrictions,
checkpoint/replay integration, compatibility readiness, and delivery handoff.
Installed DSH package source, native tool algorithms, internal composition,
session-store implementation, and general coding/web quality remain outside
the test subject. A real DSH process can be a collaborator in an integration
test without becoming the subject of an upstream conformance suite.

Use the permanent probe as the sole scenario owner for sidecar lifecycle.
Fold the unique wrong-token and two-independent-resolves checks from the old
process suite into its existing boot session. Keep Python worker framing/auth
unit checks separate because those exercise a different owned interface.
Observe the configured provider request contract without pinning request
ordinal or total model-call count. Scripted provider input is fault-injection
machinery, never a claimed test of model quality.

For committed response loss, a harness-owned loopback relay receives and holds
the sidecar's terminal HTTP response, drops the caller connection before
forwarding it, and then stops the owned sidecar. Restart and inspect/replay the
same operation through public RPC. This proves a committed result survives
loss before the caller receives it, without a test branch in production or
inspection of DSH's SQLite implementation. Keep time/PID/socket ownership and
cleanup in the harness. Shared public process resource helpers may serve the
live harness; independent test files invoke the lifecycle scenario through
its CLI, rather than copying or asserting its implementation.

Extract existing launch/readiness/stop/resource-record code into
`experiments/dsh_process_support.py`, a neutral module without a CLI,
scenario evaluator, scripted provider ledger, or release pass/fail decisions.
The probe and live harness consume that resource boundary. Keep deterministic
provider scripting and probe observations in `dsh_runtime_probe.py`; the live
harness owns its own evidence records. This is a relocation of existing
shared mechanics, not a new process-management framework.

The eight cases in `sidecars/dsh_resolution/tests/process.spec.ts` receive
these exact dispositions before that duplicate runner is deleted:

| Existing case | Retained owner and disposition |
|---|---|
| `process > serves one long-lived independent process across multiple sessions` | Fold into the permanent lifecycle probe's boot session. |
| `process > replaces an incoherent resolved terminal submission` | Retain `contracts.spec.ts > contracts > validates status-specific submit_resolution and exhaust`; remove scripted retry/model-cooperation proof from process coverage. |
| `process > restarts and cold-resumes evidence from the versioned session store` | Existing checkpoint/restart segment of the permanent lifecycle probe. |
| `process > recovers terminal receipt when killed after commit before rpc response` | Permanent probe's external loopback relay and public replay. |
| `process > reconciles admitted operation after controller restart without model re-entry` | This case actually replays a committed terminal; consolidate with the permanent terminal replay. Active-operation reconciliation remains in the Python controller owner node. |
| `V2 process > starts only after route standard worker web and brain health are ready` | Preserve public readiness statuses and catalog compatibility in permanent boot evidence; remove hard-coded digest and installed-layout assertions. |
| `V2 process > rejects a tampered activation token before Agent admission` | Fold one tampered-token attempt into the same probe boot session; observe rejection and no admitted operation through public inspection. |
| `V2 process > reports unavailable worker health for a non-worker executable` | Retain one unavailable-worker launch in the permanent lifecycle probe; assert public readiness fails closed and record owned cleanup. |

`runtime.spec.ts` is rewritten around the real `forProduction` path: one
pending executor tests same-operation joining and fresh open/continue
forwarding; committed public inspection tests replay without executor entry;
controls test exact identity forwarding. Delete `lifecycle.spec.ts`'s fake
lease/compatibility/interaction tests. Their real owners are the Python
controller/Mongo fencing tests, the sidecar profile, and the internal Brain
interaction service. The removed fake correction-count and native multi-tool
simulations have no release-proof replacement; DSH owns those algorithms.

## Live Input Coverage With Bounded Cost

Keep three named behavior contracts as a campaign organization, rather than
making the number three a schema or release invariant. Use these fixed input
partitions instead of multiplying trigger sources by success/failure cases:

| Contract | Inputs and independent semantic risk | Acceptance | Planned cost |
|---|---|---|---|
| Foreground task | First ask for the owner from two supplied local notes with conflicting owners and no selected release. Then clarify which release the user means. The second note supplies the source-grounded owner and one limitation. | First response identifies the ambiguity or states the conflict without inventing a choice. After clarification, evidence and final visible text preserve the chosen release's owner and limitation. A valid grounded route may vary. | At most two user turns in one service scenario. |
| Deferred task | A natural request to compare supplied local documents and return the result later; include one missing fact so completion requires an explicit uncertainty statement. | The accepted request returns once through result recurrence; the delivered content answers supported parts and identifies the missing fact. Correlate delivery to the accepted operation and audience. | One user turn plus one result recurrence in one service scenario. |
| Internal character judgment | One answerable signed question grounded in task evidence and one request to present an unsupported success claim. | The first decision uses the available evidence; the second preserves uncertainty and gives no unsupported permission/completion claim. Check kind-specific enums, then independently review judgment. | Two isolated cognition-boundary calls; exercise the interaction service without booting DSH merely to test Brain judgment. |

This is six planned cognition episodes across two service scenarios and one
paired internal-boundary scenario. Record actual LLM calls, elapsed time, and
tokens from traces because one cognition episode can contain multiple calls.
Count paired inputs honestly; reducing pytest function count is not a cost
optimization. Reuse existing cases by replacing redundant homogeneous inputs.
Add a case only for an identified semantic distinction absent from this table.

Episode-level entry gates are explicit: foreground ambiguity has optional
DSH entry; clarified foreground and deferred requests require a correlated
Kazusa task/DSH entry; the internal pair is direct Brain cognition evidence.
`sidecars/dsh_resolution/tests/brain_interaction.spec.ts > Brain interaction >
sends exact V2 question and approval decisions` owns the deterministic
DSH-to-Brain mapping. Hard literal checks apply to immutable source identifiers
and values; limitations and explanations admit paraphrase and belong to the
independent semantic rubric.

Each fixture records `behavior_contract`, `input_kind`, `hard_gates`,
`behavior_rubric`, `acceptable_variation`, `forbidden_failure_modes`, and
`trace_required`. Hard gates cover schema/kind, provenance, required
source literals, wrong audience, duplicate delivery, private/internal leakage,
and false action claims. The behavioral rubric judges grounded task completion,
clarification, character judgment, and continuity from actual output.

Use controlled supplied documents for reproducible evidence, not harness-only
natural-language commands asking for a particular tool, checkpoint, or stage
sequence. Required positive DSH entry is measured from the correlated task
boundary. Ambiguity can validly resolve without DSH entry; identity/readiness
and deterministic source restrictions belong in unit/integration coverage.

Keep diagnostic raw persistence snapshots in one harness-owned collector.
Select records by the scenario's identity and operation lineage. Assertions
use source-owned projections and correlated receipts rather than global
collection counts. Use public DB setup/cleanup APIs and configure the guarded
database before importing database owners; eliminate private client-global
assignment. A technical pass leaves qualitative acceptance `pending` until
the independent reviewer inspects input, output, trace, and delivered content.

Use `tests/dsh_database_test_support.py` as the test-owned guarded diagnostic
collector. It accepts an explicit URI, newly generated database name, and
scenario lineage; validates the exact test-name guard before connecting or
dropping; owns its direct Motor client; and returns bounded lineage-filtered
snapshots for artifacts only. Configure the application's existing environment
contract before importing it, use public bootstrap/profile/identity APIs for
seeding, and public `db.close_db` for application shutdown. Eliminate imports
of `db._client`, private guard calls, and mutation of its globals. This avoids
adding a production API solely for test setup.

## Change Surface

### Modify production

- `src/kazusa_ai_chatbot/task_resolution/service.py`: cancellation lifecycle,
  canonical async-open consumption, and removal of the two discarded broker
  parameters; include pre/post-commit cancellation in existing deferred promotion.
- `src/agentic_resolver/rpc.py`: cancellation-capable HTTP transport and removal
  of the fake server/registry, fake transport, and test-only sync methods.
- `src/agentic_resolver/persistence.py`: remove the in-memory implementation
  and align the existing repository protocol with production async calls.
- `src/agentic_resolver/controller.py`: async repository-call consumption only;
  preserve operation, authority, lease, and reconciliation decisions.
- `src/kazusa_ai_chatbot/dsh_interaction/auth.py`: remove the test nonce store.
- `src/kazusa_ai_chatbot/dsh_tool_gateway/authority.py`: remove the test replay
  store.
- `src/kazusa_ai_chatbot/dsh_tool_gateway/worker.py`: remove the unused test
  outcome owner and correct `SemanticWorkerCall` documentation only.
- `sidecars/dsh_resolution/src/profile.ts`: remove the terminal-commit test
  exit hook only; preserve receipt flush and public terminal outcome behavior.
- `sidecars/dsh_resolution/src/runtime.ts`: remove the scripted alternative
  and preserve the existing `forProduction` path and real interfaces above.
- `sidecars/dsh_resolution/src/secret_broker.ts`: remove `runNativeProbe` from
  the interface and returned broker; preserve production credential policy.
- Delete `sidecars/dsh_resolution/src/terminal_policy.ts` after removing its
  script-only imports/re-export; the production profile policy is unchanged.

### Modify verification and support

- `tests/test_agentic_resolver_rpc.py`,
  `tests/test_agentic_resolver_controller.py`,
  `tests/unit/agentic_resolver/test_runtime_task_lifecycle.py`,
  `tests/unit/agentic_resolver/test_rpc_readiness.py`,
  `tests/unit/task_resolution/test_service.py`, and
  `tests/test_task_resolution_inline_promotion.py`: canonical async doubles,
  contract outcomes, and the exact lifecycle regressions below.
- `tests/task_resolution_test_helpers.py`: fixture compatibility with the
  existing binding API; preserve realistic public async behavior.
- `tests/test_dsh_brain_interaction_contracts.py`,
  `tests/test_dsh_brain_interaction_auth.py`,
  `tests/test_dsh_brain_interaction_service.py`,
  `tests/test_dsh_brain_interaction_decision.py`,
  `tests/test_dsh_tool_gateway_authority.py`,
  `tests/test_dsh_tool_gateway_worker.py`,
  `tests/test_dsh_tool_gateway_conversation.py`, and
  `tests/test_dsh_tool_gateway_recall_calendar.py`: relocate shared builders,
  update test-store imports, and retain genuine auth/replay/handoff assertions.
- `tests/test_dsh_behavior_live_llm.py` and
  `tests/dsh_behavior_e2e_support.py`: visible case contracts, input partitions,
  public setup, correlated evidence, and independent acceptance artifacts.
- `tests/ownership/source_test_impact_manifest.json`: map surviving production
  owners to real owner tests; remove fake-subject test mappings.
- `experiments/dsh_runtime_probe.py` and
  `tests/integration/test_dsh_runtime_probe.py`: own one lifecycle scenario,
  external response-loss injection, correlated public outcome checks, and
  folded auth/session-reuse evidence. Keep the result schema; eliminate the
  exact observation-label set oracle.
- `sidecars/dsh_resolution/tests/profile.spec.ts` and
  `sidecars/dsh_resolution/tests/standard_profile.spec.ts`: prune installed
  package path/composition assertions and preserve local integration policy.
  `sidecars/dsh_resolution/tests/rpc.spec.ts` remains a verification-only
  Kazusa RPC owner. `sidecars/dsh_resolution/tests/submit_resolution.spec.ts`
  and `sidecars/dsh_resolution/tests/contracts.spec.ts` receive fixture-import
  relocation only; preserve their direct contract/receipt assertions.
- Rewrite `sidecars/dsh_resolution/tests/runtime.spec.ts` to use
  `forProduction`; modify `sidecars/dsh_resolution/tests/secret_broker.spec.ts`
  to launch its child with the returned native environment in test code.
- Delete `sidecars/dsh_resolution/tests/process.spec.ts`,
  `sidecars/dsh_resolution/tests/lifecycle.spec.ts`, and
  `sidecars/dsh_resolution/tests/terminal_policy.spec.ts` with the explicit
  outcome dispositions above. Keep `contracts.spec.ts` and
  `brain_interaction.spec.ts` as existing interface verification owners.
- Create `experiments/dsh_process_support.py` and
  `tests/dsh_database_test_support.py` for the neutral process resources and
  guarded diagnostic collector described above.
- Create `sidecars/dsh_resolution/tests/contract_test_helpers.ts` by moving
  `validIntake`, `validRuntime`, and `validSubmit` out of `contracts.spec.ts`.
  The contract, receipt, and rewritten runtime specs import this helper; it
  registers no tests and has no production importers.
- Create `tests/agentic_resolver_test_helpers.py` and
  `tests/dsh_interaction_test_helpers.py` for the relocated, minimal support.
- Delete `tests/test_agentic_resolver_persistence.py` after recording the
  retained real adapter and Mongo evidence mapping below.
- Delete `tests/test_agentic_resolver_sidecar_process.py` after folding its
  unique public outcomes into the permanent probe; remove only
  `test_real_sidecar_worker_round_trip_preserves_authority_result_and_evidence`
  from `tests/test_dsh_tool_gateway_worker.py` because that scenario is
  duplicated by the probe.

### Documentation and plans

Update `src/agentic_resolver/README.md`,
`src/kazusa_ai_chatbot/task_resolution/README.md`,
`sidecars/dsh_resolution/README.md`, `docs/HOWTO.md`, this plan, the parent,
and `development_plans/README.md` to describe actual ownership and gate state.

### Preserve

Cognition's V2 observation reference, prompts, semantic decisions, catalog,
schemas, DB selectors/indexes, authorization policy, adapters, and DSH package
dependencies retain their current contracts. This plan authorizes no new
semantic feature, broad controller decomposition, compatibility shim,
production data change, deployment, or environment-file inspection.

## Test Impact And Traceability

Paths are repository-relative. **New** nodes below are acceptance targets for
implementation, not claims that they already collect. Each production owner
has an exact deterministic node; supplemental evidence is separately named.
Tests added for cancellation observe the public call, binding disposition,
owned resource closure, and remote control outcomes rather than private task
registry names. No test asserts this document's inventory.

| Source or governed artifact | Changed contract and owner | Deterministic pytest node IDs | Mode and regression prevented |
|---|---|---|---|
| `src/kazusa_ai_chatbot/task_resolution/service.py` | Foreground lifecycle; task service | **New:** `tests/unit/task_resolution/test_service.py::test_caller_cancellation_before_handoff_closes_owned_execution`; **new:** `tests/unit/task_resolution/test_service.py::test_cancellation_after_committed_handoff_preserves_background_owner`; **new:** `tests/unit/task_resolution/test_service.py::test_checkpoint_failure_closes_or_records_uncertain_operation`; retain `tests/unit/task_resolution/test_service.py::test_inline_admission_failure_terminally_faults_binding`; rewrite `tests/unit/task_resolution/test_service.py::test_inline_checkpoint_promotes_same_bound_dsh_session_without_canceling_reasoning` | Controlled async handoff; prevent orphan work and retain the terminal/checkpoint race. Cancellation test includes terminal-wins as a deterministic branch. |
| `src/agentic_resolver/rpc.py` | Client request/reconciliation/resource lifetime | **New:** `tests/test_agentic_resolver_rpc.py::test_cancelled_http_request_releases_connection`; rewrite `tests/test_agentic_resolver_rpc.py::test_disconnect_before_admission_inspects_then_replays_same_operation_once`; `tests/test_agentic_resolver_rpc.py::test_disconnect_after_admission_attaches_to_active_operation`; `tests/test_agentic_resolver_rpc.py::test_disconnect_after_terminal_commit_reconciles_exact_exhaust_without_model_call`; `tests/test_agentic_resolver_rpc.py::test_unknown_operation_outcome_returns_uncertain_fault_without_new_admission` | Client unit tests with scripted responses; local HTTP resource probe supplements cancellation. Distinguish actual active/committed/unknown responses without a fake server. |
| `tests/test_agentic_resolver_rpc.py` | HTTP timing and cancel/inspection contract | Rewrite `tests/test_agentic_resolver_rpc.py::test_http_transport_has_no_dsh_execution_deadline_and_bounds_control`; **new** cancellation node above | Controlled local HTTP peer through public client: open remains pending beyond the configured control bound, control request times out, cancellation closes the pending connection, and same-id inspection remains possible. Use synchronization with a generous scheduling margin; assert behavior rather than client-library internals. |
| `tests/unit/agentic_resolver/test_rpc_readiness.py` | Public client failure taxonomy | `tests/unit/agentic_resolver/test_rpc_readiness.py::test_transport_error_preserves_low_level_cause`; `tests/unit/agentic_resolver/test_rpc_readiness.py::test_runtime_readiness_delegates_to_authenticated_sidecar_health` | Retain concrete low-level cause through the public client after replacing its HTTP implementation. |
| `src/agentic_resolver/persistence.py` | Actual Mongo adapter and async protocol | `tests/unit/agentic_resolver/test_runtime_task_lifecycle.py::test_mongo_repository_translates_database_owner_errors`; `tests/unit/agentic_resolver/test_runtime_task_lifecycle.py::test_mongo_repository_retries_transient_index_creation_once` | Adapter unit; removing the fake retains typed failures and bounded index recovery. |
| `src/agentic_resolver/controller.py` | Async repository consumption and fenced reconciliation | `tests/test_agentic_resolver_controller.py::test_controller_restart_reconciles_terminal_projection_and_lease`; `tests/test_agentic_resolver_controller.py::test_controller_restart_reattaches_admitted_operation_with_same_fence`; `tests/test_agentic_resolver_controller.py::test_concurrent_checkpoint_and_open_cleanup_release_once` | Deterministic owner/handoff; public result and collaborator effects retain the same operation/fence and single cleanup. |
| `src/kazusa_ai_chatbot/dsh_interaction/auth.py` | Authentication with injected replay interface | `tests/test_dsh_brain_interaction_auth.py::test_mac_timestamp_nonce_digest_and_constant_time_validation_fail_closed`; `tests/test_dsh_brain_interaction_auth.py::test_signed_interaction_remains_valid_for_its_full_declared_lifetime` | Deterministic auth; rewrite the first test to actually submit bad MAC/digest/time candidates to validation rather than merely constructing a changed object. |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/authority.py` | Signed semantic call and replay consumer | `tests/test_dsh_tool_gateway_authority.py::test_activation_authenticates_complete_catalog_scope_fence_reference_lineage_and_replay`; `tests/test_dsh_tool_gateway_authority.py::test_mutation_idempotency_excludes_transport_call_id` | Deterministic authority; preserves rejection and idempotency after test-store relocation. |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/worker.py` | Production worker remains durable/authenticated | `tests/test_dsh_tool_gateway_worker.py::test_worker_replays_committed_idempotent_results_and_preserves_uncertain_mutation_state`; `tests/test_dsh_tool_gateway_worker.py::test_worker_authenticates_before_replay_lookup_and_denies_tampered_committed_retry` | Real SQLite owner with controlled handlers; eliminate unused fake without weakening authenticate-before-replay behavior. |
| `tests/unit/task_resolution/test_service.py` | Authority admission oracle | `tests/unit/task_resolution/test_service.py::test_background_start_mints_authority_only_when_claimed`; `tests/unit/background_work/test_dsh_worker.py::test_worker_checkpoints_waits_and_terminalizes_current_generation_through_binding`; `tests/unit/agentic_resolver/test_runtime_task_lifecycle.py::test_checkpoint_and_terminal_continuation_issue_fresh_authority_and_preserve_thread_segment` | Revise the admission test to assert only its owner; real runtime/worker handoffs own fresh issuance evidence. |
| `tests/test_agentic_resolver_persistence.py` | Retire fake persistence claims | The two real adapter unit nodes above; supplemental `tests/test_agentic_resolver_live_db.py::test_resolution_thread_store_enforces_operation_idempotency_lease_fencing_rotation_and_cold_resume` | Guarded Mongo integration is the actual persistence/restart oracle. |
| `sidecars/dsh_resolution/src/profile.ts` | Ordinary terminal return after removing test exit | `tests/integration/test_dsh_runtime_probe.py::test_sidecar_lifecycle_probe`; direct TypeScript receipt owner `sidecars/dsh_resolution/tests/submit_resolution.spec.ts > submit_resolution > commits the complete terminal receipt before returning exhaust` | Real process with deterministic provider supplements direct receipt checks; external response loss replaces the production crash switch. |
| `experiments/dsh_runtime_probe.py` | Consolidated public integration scenario | `tests/integration/test_dsh_runtime_probe.py::test_sidecar_lifecycle_probe`; `tests/integration/test_dsh_runtime_probe.py::test_brain_task_lifecycle_probe`; `tests/integration/test_dsh_runtime_probe.py::test_transport_loss_probe` | Existing CLI contract; inspect actual public outcomes, owned process exits, and cleanup. Mongo cases stay explicitly marked `live_db`. |
| `tests/test_agentic_resolver_sidecar_process.py` | Retire duplicate lifecycle oracles | `tests/integration/test_dsh_runtime_probe.py::test_sidecar_lifecycle_probe` | Preserve its unique auth/session-reuse outcomes inside the retained scenario before deletion. |
| `tests/test_dsh_tool_gateway_worker.py` | Retire duplicate sidecar round trip | `tests/integration/test_dsh_runtime_probe.py::test_sidecar_lifecycle_probe`; `tests/test_dsh_tool_gateway_worker.py::test_direct_python_worker_process_round_trip_preserves_authority_result_and_evidence` | Sidecar-to-worker integration plus direct framed-worker boundary; drop provider-sequence assertions. |
| `sidecars/dsh_resolution/src/runtime.ts` | Sole production execution path | **New Vitest:** `sidecars/dsh_resolution/tests/runtime.spec.ts > production runtime > joins duplicate pending operations and forwards fresh open and continue`; `sidecars/dsh_resolution/tests/runtime.spec.ts > production runtime > replays committed inspection without executor entry`; `sidecars/dsh_resolution/tests/runtime.spec.ts > production runtime > forwards fenced controls to the production owner` | Controlled executor/controls via `forProduction`; preserves actual local orchestration while removing the simulator. |
| `sidecars/dsh_resolution/src/secret_broker.ts` | Credential isolation without test subprocess method | `sidecars/dsh_resolution/tests/secret_broker.spec.ts > secret isolation > native shell cannot read host credentials tokens or bridge secrets` | Native environment and host credential tests retain the production boundary; test code owns the child process. |
| `sidecars/dsh_resolution/src/terminal_policy.ts` | Delete script-only policy | `sidecars/dsh_resolution/tests/contracts.spec.ts > contracts > validates status-specific submit_resolution and exhaust`; `sidecars/dsh_resolution/tests/submit_resolution.spec.ts > submit_resolution > commits the complete terminal receipt before returning exhaust` | Preserve actual terminal contracts; fake model-step policy has no production owner to test. |
| `sidecars/dsh_resolution/tests/process.spec.ts` | Retire duplicate runner and hook dependency | `tests/integration/test_dsh_runtime_probe.py::test_sidecar_lifecycle_probe` and the eight-case disposition table | One public process oracle preserves unique failures without the test exit hook. |

TypeScript verification uses `npm test -- tests/profile.spec.ts
tests/standard_profile.spec.ts tests/rpc.spec.ts tests/submit_resolution.spec.ts
tests/runtime.spec.ts tests/secret_broker.spec.ts tests/contracts.spec.ts
tests/brain_interaction.spec.ts`
from `sidecars/dsh_resolution`, plus typecheck/build after the profile edit.
Retain these exact local-policy cases: `profile > fails startup for an
incompatible DSH dependency graph`, `profile > rejects an intake whose route
digest differs before session creation`, and `official Standard profile > adds
Kazusa semantic tools without colliding with official Standard capabilities`.
Prune only their unrelated internal-layout assertions. TypeScript owners use
native Vitest owner tests rather than adding a Python test that greps TS text.
Remove `process.spec.ts`, `lifecycle.spec.ts`, and `terminal_policy.spec.ts`
from the runnable inventory after their dispositions are complete. Their
deleted tests are not passed as stale paths to Vitest. Replace the parent's
`Real sidecar boot/readiness/semantic worker` and
`SQLite checkpoint/restart/replay` rows with the permanent lifecycle probe
node, and add the folded auth/session outcomes to that row. The parent matrix
remains historical for executed checks; its future acceptance commands must
resolve only surviving nodes.

For relocated builders, collect and run their direct consumer tests from the
listed files after the import cutover. Exact changed-owner nodes above are
mandatory; additional consumer checks address collection/import risk, rather
than introducing helper tests. Maintain the exact live wrapper names already
listed in the parent; collect them with `--collect-only` in this slice.

## Execution Order And Acceptance

1. **Review baseline:** retain the 2026-09-05 findings, reproduction, initial
   dirty state, and parent evidence. Obtain independent review of this draft.
2. **Lifecycle correction:** on implementation authorization, first turn R1's
   deterministic reproduction into the owner regression. Correct service and
   transport cancellation, with event/barrier-controlled races. Run the exact
   lifecycle/client nodes before broader test work.
3. **Test isolation:** move minimal test support, remove the unused production
   surfaces, align async APIs, and replace fake-subject tests. Preserve real
   authority/replay checks and map every changed owner in the impact manifest.
   Consolidate duplicate process scenarios, replace the crash hook with external
   response loss, and prune third-party composition oracles.
4. **Live harness readiness:** implement the input/rubric table, remove private
   setup and global-count assumptions, and collect each live wrapper without
   execution. Review harness diffs statically and use small deterministic
   examples only for harness extraction/contract-gate logic with actual risks.
5. **Non-E2E evidence:** run exact deterministic owner nodes once after the
   final change; run the integration interface checks affected by that diff.
   Guarded Mongo requires explicit test DB selection and cleanup. Reuse one
   artifact for the same scenario across CLI/pytest entrypoints. Broaden runs
   for changed shared behavior or new failures, rather than repeating the
   entire unchanged sidecar/full repository suite at every gate.
6. **Independent review:** verify all high findings closed and every retained
   test has a named Kazusa-owned regression. Confirm removed cases have either
   a real owner oracle or an explicit upstream/duplicate disposition. Report
   non-E2E readiness separately from live behavior sign-off.
7. **Live handoff:** retain the E2E exclusion until a subsequent user command
   authorizes execution. Then run one selected case, inspect its full evidence
   and independent qualitative review, and only then run the next case.

Successful remediation means cancellation has a bounded, observable owner;
production contains no test doubles or no-op test parameters in this scope;
tests exercise actual owners through canonical contracts; live inputs cover
the listed semantic partitions with measured cost; and the test subject is
Kazusa integration rather than DSH internals. No unit count or collection pass
substitutes for those outcomes.

## Skills, Roles, And Autonomy

Apply `development-plan`, `local-llm-architecture`, `py-style`, and
`test-style-and-execution`. Apply `probe-first-engineering` for the runtime
correction and `cjk-safety` when editing CJK Python fixtures. Use the project
venv; read the registry and affected ICDs before implementation. Prompt or
user-input semantic changes require a separately defined scope and applicable
semantic-ownership skills.

| Role | Responsibility / owned surface / authority | Skills and capability | Independence / output / gate |
|---|---|---|---|
| implementation_owner | Named production, tests, docs, and plan evidence; implement only approved scope and operate owned test resources. | Skills above; senior asyncio, HTTP, persistence fencing, and test-oracle reasoning. | Separate from reviewer; scoped diff, exact collected/run nodes, resource cleanup, coverage dispositions; enters after explicit implementation command and plan approval. |
| independent_reviewer | Read-only review of plan, source, tests, and evidence; require fixes or reject acceptance. | Development plan, architecture, Python/test style; lifecycle and false-green analysis. | Separate from remediation; severity-ranked findings and non-E2E readiness decision; exits after high findings close and medium findings are fixed or accepted. |
| character_behavior_reviewer | Read-only future live input, traces, visible output, and internal decisions. | Character-test, local-LLM architecture, development-plan; grounding and character judgment. | Separate from implementation; per-case qualitative decision; enters only after live execution is authorized and artifacts exist. |

Executors may choose local code organization and command ordering within these
contracts. New semantic behavior, data schema, test-only production hooks,
upstream DSH tests, or additional architecture changes require a recorded
scope amendment. Capture the pre-handoff dirty state and exact owned files;
preserve concurrent edits. Resolve executors at handoff rather than pinning
models in the plan.

## Review Evidence And Progress

### Execution checkpoint — 2026-09-05

- Execution owner: parent `/root`, inherited session model/configuration.
  Lifecycle slice: `/root/dsh_lifecycle_implementation`, dynamically selected
  default agent with inherited model/configuration for senior asyncio, HTTP,
  and lease-fencing work. It owns the four named Python lifecycle/transport/
  repository/controller sources and their mapped tests; parent owns sidecar,
  interaction test isolation, process/live support, manifest, docs, and plans.
  Review and behavior sign-off will use a separate read-only executor.
- The original cancellation reproduction again observed unowned pending work.
  The initial sidecar baseline passed semantic execution and checkpoint restart
  but failed its test-exit-hook phase; evidence is under
  `test_artifacts/dsh_execution_20260905/baseline_sidecar/`.
- External terminal-response-loss injection subsequently passed; public replay
  preserved the exact exhaust. The consolidated probe also passed wrong-token,
  invalid-signature, independent-session, and missing-worker checks; see
  `relay_probe_4/result.json` under the execution artifact root. The CLI pytest
  wrapper passed in 13.95 seconds in `process_retry.log`.
- Guarded Mongo Brain/task and transport-loss probes passed together in
  11.40 seconds (`mongo_probe_checks.log`), including exact database cleanup.
- Revised runtime, receipt, profile, RPC, credential, catalog, and Brain bridge
  checks passed 40 native Vitest cases; TypeScript build/typecheck passed.
  Interaction/authority consumers passed 23 checks and framed-worker checks
  passed 5. These are intermediate evidence, preceding the lifecycle slice's
  final changes and the final mapped/full non-live run.
- Live wrappers now expose the planned input partitions and rubrics and all
  three collect. Raw diagnostics use a separate exact-guarded client; the
  internal case uses its Brain collaborators only. Real invocation recording
  forwards the unchanged model call and stores actual usage. The DSH recorder
  forwards real HTTP provider traffic and retains usage-bearing responses;
  it supplies no model output or semantic decision.
- Concurrent repository activity advanced HEAD from the captured baseline
  `210cdb8f` to `4dedce3b`; baseline artifacts remain intact and each probe
  records its actual revision/worktree fingerprint. Parent has issued no git
  commit or worktree reset.

### Historical review evidence

### Current readiness handoff — 2026-09-05

- The lifecycle executor completed its final slice with 90 passing checks in
  `test_artifacts/dsh_execution_20260905/lifecycle_slice_final_recheck.txt`,
  including cancellation after a remote terminal commit. Its scoped Ruff,
  compilation, and whitespace checks pass in `lifecycle_slice_static_recheck.txt`.
- The combined ownership gate collected and ran 496 exact nodes successfully
  (`combined_impact_gate.log`). The full native sidecar suite passed 53 tests
  in 11 files (`final_vitest.log`). The post-hardening public sidecar probe
  passed in 14.51 seconds (`final_runtime_probe.log`). These results establish
  their exercised boundaries; real-model behavior remains unverified.
- Independent readiness reviewer: `/root/dsh_readiness_review`, dynamically
  resolved as a default full-context agent with inherited session model and
  configuration. The role has read-only access to the full implementation,
  original baseline, plans, and evidence. It is independent of both production
  authors; full context reduces reconstruction while retaining the required
  lifecycle and false-green analysis capability. Its acceptance output is a
  severity-ranked review and explicit Gate 6 decision. The full non-live run
  is pending in `final_nonlive_suite.log`; individual live cases follow only
  after readiness passes and receive independent behavioral review.

### Historical review evidence (continued)

- Baseline: HEAD `210cdb8f381f116afb437b42472eb6bf88a75142` with the existing
  DSH worktree changes recorded before review; this review changes plans only,
  plus ignored diagnostic artifacts.
- Before review edits, SHA256 of task service:
  `8f147d56f40612817ee0e0360e3bc1482c7a9eca91463e19720c325400a2a9f5`;
  RPC: `17378476126a84e034891fe125bf069c23c64d4704951712483bc3e6695337fa`;
  live support: `ca63bfa7660dd3a56cbcc169581e8067848d6968ddb4bb3c96290d3353382677`.
- R1 reproduced through the actual task service; observations are quoted
  above. Diagnostic exit 0 means the observation completed, not a correctness
  pass. The first attempt stopped on missing synthetic route configuration;
  the corrected isolated invocation used dummy endpoints and disabled dotenv.
- Eight exact deterministic nodes passed in 0.97 seconds: the four V2
  dependency nodes in `tests/test_cognition_resolver_contracts.py`, the inline
  checkpoint and failed-admission nodes in `tests/unit/task_resolution/test_service.py`,
  the Mongo adapter error-translation node, and
  `tests/test_task_resolution_inline_promotion.py::test_inline_runtime_checkpoint_is_projected_without_reclassification`.
  These passing checks coexist with R1; their result is not cancellation proof.
- Actual E2E, real LLM, real Mongo, and DSH process tests were excluded from
  this review's execution. Existing historical artifacts remain historical.
- Independent review executor: `/root/dsh_test_review`, dynamically resolved
  to the available default agent with inherited parent model/configuration,
  read-only repository access, and the same E2E/environment exclusions. Reuse
  preserved its test-audit context and independence from the parent-authored
  plan. The first draft review identified an omitted TypeScript process runner,
  HTTP timing coverage, private test seams, and stale parent gate wording.
  The parent incorporated those findings, the additional scripted-runtime
  audit, and exact dispositions. Final review returned **PASS**, with no
  remaining blocker, high, or medium finding. This is plan readiness only;
  production remediation and behavior acceptance remain pending.
- Final local checks: all document links and plan whitespace passed;
  static AST resolution found 29 existing referenced pytest functions and
  exactly the four explicitly marked new Python acceptance targets. This is
  source-name validation, not pytest collection or execution. The task-service,
  RPC, and live-support source hashes above remained unchanged after review.
- [x] Inspect current architecture and representative source-owner tests.
- [x] Reproduce the cancellation ownership failure without E2E.
- [x] Specify production/test boundary corrections and live input partitions.
- [x] Complete independent plan review; final result PASS as recorded above.
- [x] Receive explicit implementation instruction and approve execution scope.
- [ ] Implement and verify lifecycle and test-isolation corrections.
- [ ] Complete independent non-E2E readiness review.
- [x] Receive explicit real LLM execution instruction on 2026-09-05.
- [ ] Complete independent live behavior sign-off.
