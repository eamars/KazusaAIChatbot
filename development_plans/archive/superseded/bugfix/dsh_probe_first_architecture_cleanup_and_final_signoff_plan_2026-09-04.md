# DSH Probe-First Architecture Cleanup And Final Sign-Off Plan

## Document Control

- **Status:** superseded by explicit owner direction on 2026-09-05.
- **Successor:** [DSH runtime completion](../../completed/bugfix/dsh_runtime_completion_plan_2026-09-05.md).
  The successor covers the entire existing and new DSH codebase. Historical
  prescriptions below are retired. The successor defines a fresh investigation
  and excludes this attempt from the next agent's decision context.
- **Created:** 2026-09-04.
- **Historical execution authority:** the earlier user request covered this
  plan's then-reviewed implementation cleanup through the real-LLM hold point.
  New production scope in the 2026-09-05 remediation draft has its own explicit
  implementation boundary and authority requirements.
- **Supersedes:**
  `dsh_operational_e2e_quality_signoff_and_transport_failure_remediation_plan_2026-08-31.md`.
- **Implementation boundary:** production, test, experiment, documentation, and
  plan-lifecycle changes named here and in the approved remediation plan.
  The current user instruction includes real LLM execution after readiness
  review. Environment-file inspection, production data mutation, and deployment
  remain outside the execution slice.
- **Current sign-off state:** withheld. The next work is the bounded
  [integration-boundary and test-contract remediation](dsh_integration_boundary_and_test_contract_remediation_plan_2026-09-05.md).
  The latest 2026-09-05 user instruction explicitly authorizes actual E2E
  execution and closure after the remaining acceptance gates pass.
- **Historical plan review:** passed on 2026-09-04 with zero blocker/high findings after
  three bounded review rounds; Gate 0 may close.
- **Historical implementation review:** passed on 2026-09-04 with zero blocker,
  high, or medium findings and supported the then-current Gate 6 closure.
  Review amendment 4 supersedes that readiness conclusion and reopens the
  non-E2E gates named above.
- **Amendment 1:** Gate 2 verification demonstrated that the service collapses
  a typed `resolver_state_contract` error to `model_contract` and exposes an
  exhausted retry as still retryable. The bounded amendment passed independent
  review and now preserves the producing error code while marking exhausted
  failures non-retryable.
- **Amendment 2:** the first full non-live Gate 6 run passed 3,384 tests and
  exposed eight stale DSH/documentation oracles plus one unrelated RAG test
  that calls an unavailable model. The DSH fixtures will be aligned with the
  already-approved semantic-ref result contract, the typed service error test
  will preserve its producer code, the managed-sidecar dependency expectation
  will be updated, and the HOWTO check will cease depending on capitalization.
  Those test-only corrections passed the final suite. The RAG failure received
  no edit in this DSH plan and remains exact residual evidence.
- **Amendment 3:** final implementation review found that the two Mongo-backed
  probe nodes lacked `live_db` markers and one L3 test imported a private
  fixture from a sibling test module. Both boundaries were corrected. The
  first post-review aggregate run then exposed a duplicated five-second raw
  child-process wait in terminal response-loss coverage. The permanent probe
  now owns one public 15-second bounded exit helper used by both probe and
  process test; the exact node and clean aggregate rerun passed.
- **Review amendment 4 (2026-09-05):** a public-service deterministic probe
  reproduced execution continuing after caller cancellation, with an `opening`
  binding and no accepted-task/background-job owner. Production still includes
  test doubles, discarded test-only parameters, and a sidecar test-exit hook.
  Independent test review found remaining DSH composition assertions,
  duplicated process tests, private DB setup, and insufficient live behavioral
  oracles. The linked draft fixes the new change surface and acceptance
  targets. This review authorizes plan changes; implementation of that new
  scope requires the execution boundary recorded in the draft. Earlier
  evidence is preserved below as historical evidence, not current closure.

## Goal

Close the DSH plan series on a smaller and inspectable architecture whose
correctness is demonstrated first through executable process behavior. Remove
test-shaped production state, duplicated lifecycle authority, nested timeout
ownership, and self-referential sign-off tests. Preserve DSH as the sole task
execution backend and preserve cognition as the owner of semantic admission,
stance, and visible behavior.

## Mandatory Skills

- `development-plan` governs scope, traceability, review, execution evidence,
  and lifecycle.
- `probe-first-engineering` requires the real-process probe before production
  redesign or broad test work.
- `local-llm-architecture` keeps semantic judgment with cognition/DSH and
  deterministic mechanics with runtime owners.
- `py-style` applies before every Python edit.
- `test-style-and-execution` applies before changing, collecting, or running
  tests and separates deterministic/process evidence from live-LLM behavior.
- `cjk-safety` applies if any edited Python string contains CJK text.

`no-prepost-user-input` is outside the current production change surface:
this plan preserves the existing character-owned interaction judgment and
changes no interpretation, acceptance, permission, or persistence decision
over user input. A later proposal to change that semantic owner must add the
skill before execution.

## Mandatory Rules

- Use `venv\Scripts\python` for Python and pytest commands.
- Use the actual built Node sidecar, official installed DSH Standard packages,
  HTTP RPC, Python semantic worker, and SQLite session store in process probes.
- A deterministic OpenAI-compatible response server may replace only the
  currently unavailable model provider; it cannot replace a runtime boundary
  named by the probe.
- The guarded Mongo controller/task probe is mandatory before the plan may
  reach a real-LLM-only hold point.
- Keep real-LLM execution locked through Gate 6.
- Make the dependency V2 change as one big-bang internal cutover with no alias,
  legacy parser, dual writer, or fallback mapper.
- Treat prompt wording, model tool sequence, and internal stage count as
  stochastic implementation detail. Stable schemas, permissions, durable
  transitions, source identity, and delivery cardinality remain strict.
- Preserve archived plan text and historical artifacts as evidence.
- Preserve unrelated work and keep every edit inside the declared surface.

## Must Do

- Establish a passing/failing real sidecar baseline before production edits.
- Replace duplicated required-evidence state with one observation reference.
- Remove the resolver loop's second DSH timeout owner and detached-task set.
- Verify the already-committed transport/readiness changes through public
  process and task-service boundaries.
- Replace self-referential sign-off topology checks with executable probes and
  direct impact-enforcement evidence.
- Reduce live-LLM evaluation to user/character behavior that deterministic
  mechanics cannot answer.
- Pass guarded Mongo, process, mapped deterministic, and full non-live gates
  before stopping for model availability.

## Deferred

- New DSH semantic tools, new trigger sources, prompt tuning, model routing,
  adapter protocol changes, and database schema migrations.
- Structural splitting of `ResolutionController` or
  `task_resolution/service.py` based only on line count. Their public behavior
  is probed here; any later internal split requires a named invariant and its
  own bounded plan amendment.
- Production database mutation, deployment, environment-file inspection, and
  real-LLM execution while the provider is unavailable.
- Compatibility support for dependency V1 or the former sign-off manifest.

## Agent Autonomy Boundaries

The implementation owner may edit only the declared files, create guarded
temporary workspaces/databases, start and stop owned local child processes,
and refine test helper placement without changing public behavior. It may
correct a probe-discovered defect only after this plan records the exact
producer, owner, changed file, and acceptance node. Changes to prompts,
semantic admission, interaction judgment, tool catalogs, durable schemas,
model routes, adapters, or production data require explicit user direction
and a plan amendment.

## Audit Corpus

The audit covers every DSH-specific plan and its current implementation:

1. `dsh_standalone_sidecar_and_resolution_interface_plan_2026-08-26.md`;
2. `dsh_semantic_tools_and_coding_capability_plan_2026-08-26.md`;
3. `dsh_brain_bigbang_cutover_and_legacy_resolution_decommission_plan_2026-08-26.md`;
4. `dsh_excluded_legacy_live_test_decommission_plan_2026-08-31.md`;
5. `dsh_integration_scope_and_test_minimality_quickfix_plan_2026-08-31.md`;
6. `dsh_phase3_focused_e2e_signoff_reset_plan_2026-08-31.md`;
7. `dsh_touched_prompt_quality_quickfix_plan_2026-08-31.md`;
8. `public_documentation_dsh_boundary_cleanup_plan_2026-08-31.md`;
9. `dsh_operational_e2e_quality_signoff_and_transport_failure_remediation_plan_2026-08-31.md`.

The associated recovery and observability boundaries were checked against
`live_response_recovery_ladder_and_no_fail_dialog_bugfix_plan_2026-08-27.md`,
`cognition_observability_icd_and_console_consistency_plan_2026-08-26.md`, the
plan registry, commits `efcb83a7` through `210cdb8f`, and the current source,
tests, and subsystem READMEs.

## System Boundary

```text
typed trigger owner
  -> cognition chooses an executable task objective
  -> one Brain task-resolution service owns timeout and durable handoff
  -> one Python control-plane operation owns Mongo thread/lease state
  -> authenticated HTTP RPC
  -> one real DSH Standard sidecar owns agent/session/SQLite state
  -> semantic gateway returns evidence
  -> one canonical task observation returns to cognition
  -> cognition commits stance and response goal
  -> L3/dialog render; dispatcher delivers
```

Ownership stays explicit:

- RAG and DSH tools return evidence.
- Cognition owns whether work is needed and how the character responds.
- Task resolution owns one operation lifecycle and its inline-to-deferred
  transition.
- The Python controller owns Mongo identity, lease, and RPC reconciliation.
- The sidecar owns DSH Standard composition and durable DSH session events.
- L3/dialog own visible wording.
- Tests observe public state transitions and process behavior; they do not
  define production topology or semantic decisions.

## Confirmed Audit Findings

### F1 — Green unit suites repeatedly failed to prove the runtime

Plan 1 initially reported passing scripted and in-memory suites while process
restart, durable persistence, and RPC exactly-once behavior were absent. Plan
2 added more than 32,000 lines in one commit, including detailed test doubles,
before later E2E runs found activation, authority, and lifecycle gaps. Plan 3
then accumulated a long live-failure amendment chain. Passing unit counts are
therefore historical evidence of contract self-consistency, not evidence that
the integrated system works.

### F2 — One fact is stored in two mutable resolver records

`ResolverObservationV1` already owns task evidence state, remaining needs,
evidence excerpts, and the goal continuation reference.
`RequiredResolverEvidenceDependencyV1` duplicates those same values plus
derived handles. The latest failed background run changed one copy and left
the other stale; the subsequent patch added cross-record alignment validation
and synchronized rewrites. This increases coupling while retaining the defect
class.

**Decision:** the observation becomes the single source of truth. The required
dependency stores only the accepted-request identity and referenced
observation identity. State, evidence, needs, surface handles, and continuation
are derived from the validated observation at each consumer.

### F3 — Two timeout owners can leave unowned work alive

`task_resolution.service.resolve_task_inline` already owns the foreground
budget, shielded sidecar operation, checkpoint race, durable binding update,
and deferred projection. The current resolver loop wraps that call in a second
timeout and retains the shielded task in a module-global set. This permits the
resolver cycle to continue while an uncorrelated task mutates durable state.

**Decision:** task resolution has one timeout owner. The resolver loop awaits
the task capability directly; the task-resolution service returns a terminal
or deferred typed observation within its own bound. Other resolver
capabilities retain the generic resolver timeout.

### F4 — Sign-off validates its own test topology

The current 3,611-line live harness, `test_dsh_signoff_contract.py`,
`test_dsh_signoff_manifest.py`, and `validate_dsh_signoff.py` encode exact case
counts, filenames, artifact filenames, hidden stage literals, and a fingerprint
of the validator itself. These checks can prove that the sign-off harness
agrees with itself; they cannot prove the production path.

**Decision:** remove the self-referential sign-off validator and exact
ten-plus-two topology contract. Keep retained artifacts as diagnostic evidence.
Final evidence is a command/result ledger tied to the tested commit, with
process exit, public response, durable lifecycle, and cleanup observations.

### F5 — The live matrix is wide where mechanics suffice and narrow where a
real process matters

Two stochastic cases per trigger source is an arbitrary count. Targetless
self-cognition non-entry, tool-result recursion closure, source identity, and
scheduled/latch projection are deterministic contracts. Requiring a model to
re-prove each twice increases runtime and tuning pressure without adding
independent mechanical evidence.

**Decision:** deterministic source-owner integration covers admission and
non-entry invariants. The later real-LLM campaign contains three user-visible
behavior paths: foreground task resolution, deferred result recurrence and
delivery, and character-owned DSH question/approval judgment. Each uses
natural input and qualitative evidence review.

### F6 — Internal framework structure is treated as a product contract

Several sidecar tests pin package-file digests, composition row counts,
internal insert roles, exact tool ordering, and copied catalog details. The
actual product contract is narrower: the official Standard profile boots, the
required public capabilities are advertised without collision, authenticated
RPC works, the session survives restart, and terminal receipts/evidence are
valid.

**Decision:** retain protocol, authority, persistence, safety, and public
catalog behavior tests. Replace composition-internal assertions with one
black-box boot/readiness/catalog probe and one restart/replay probe.

### F7 — The controller and task service mix orchestration with repeated
mapping mechanics

`ResolutionController` is a 1,483-line class and `task_resolution/service.py`
contains several 200–300-line lifecycle functions. Repeated construction of
operation identities, fence fields, binding transitions, and projection maps
makes failure handling branch-dependent.

**Decision:** line count alone does not authorize a structural rewrite. This
plan probes their public process/persistence behavior and removes test access
to private repository state. A later split requires a concrete failed
invariant and a reviewed exact abstraction; it is not an implementation-time
choice in this plan.

### F8 — The active sign-off plan asks for an unbounded proof obligation

“Every mechanical failure point” and “zero known uncovered path” cannot be a
finite acceptance contract. It encourages exhaustive mock matrices and false
confidence values. Confidence is reported from observed coverage and explicit
residual risk instead.

## Target Contracts

### Required evidence dependency V2

The big-bang internal contract is:

```text
required_resolver_evidence_dependency.v2
  schema_version
  accepted_request_handle
  observation_id
```

`RequiredResolverEvidenceDependencyV2` is the canonical `TypedDict` in
`cognition_resolver/contracts.py`. Its validator accepts exactly the three
fields above. `cognition_resolver/state.py::required_task_observation` is the
single lookup/derivation owner. It validates the whole resolver state, requires
the referenced observation to exist, requires
`capability_kind=task_resolution_request`, and requires that observation to
carry a valid evidence-state and goal-continuation carrier.

The dependency is constructed only by
`cognition_resolver/loop.py::_bind_required_evidence_dependency`. The former
blocked-state rewrite changes the referenced observation, then replaces the
dependency with a new three-field reference; it never copies semantic state.
State projection and L3 derive state, remaining needs, continuation, excerpts,
and deterministic prompt-safe handles from that observation. Handles use the
observation id and evidence ordinal and carry no semantic authority.

A missing, wrong-kind, or invalid referenced observation raises
`ResolverValidationError` at the resolver-state boundary. The loop converts
that internal contract failure before commit into
`CognitionExecutionError(error_code="resolver_state_contract",
stage="cognition_resolver", safe_checkpoint="pre_state_commit",
retryable=True)` with the current attempt count. The invalid state cannot reach
L3, action execution, persistence, or delivery. No V1 alias, fallback mapper,
dual reader, or dual writer is retained.

### Timeout and cancellation

- `task_resolution.service` is the only inline budget/checkpoint owner.
- The resolver loop creates no detached task for task resolution.
- Cancellation propagates unless the task-resolution owner has already
  committed the durable deferred handoff.
- Shutdown has no DSH task set owned by `cognition_resolver.loop`.

### Sign-off evidence

One probe result contains:

- tested git commit and dirty-state digest;
- real child process identifiers and exit dispositions;
- authenticated readiness before work;
- RPC request/result identities;
- SQLite session evidence before and after restart;
- Mongo binding/thread evidence when the guarded DB probe is available;
- inline, checkpoint/deferred, restart/replay, duplicate, cancel, and transport
  loss outcomes;
- cleanup outcome and retained logs.

It contains observations rather than a prewritten pass decision. Acceptance is
made from this plan's fixed criteria.

## Probe-First Verification Strategy

### Design hypothesis

A real built DSH sidecar, driven through authenticated HTTP RPC and a
deterministic OpenAI-compatible response server, can boot the official Standard
profile, execute a semantic-tool/terminal sequence, checkpoint, survive process
restart, and replay exactly once without patched sidecar internals.

### First executable probe

The first post-plan command is intentionally sidecar-only. It runs the existing
real-process path in
`tests/test_agentic_resolver_sidecar_process.py` for boot, semantic call,
checkpoint/restart, and terminal replay. It uses the real Node process,
official installed DSH packages, SQLite persistence, Python semantic worker,
and HTTP RPC. It does not claim Python-controller, Mongo-binding, or Brain-task
coverage. The deterministic response server replaces only the unavailable
model provider.

The result is recorded before production cleanup. A failure changes the
implementation order and is diagnosed at the first observed boundary.

### Permanent probe

Create `experiments/dsh_runtime_probe.py` as a standalone public-boundary probe
owner. Its CLI is:

```text
venv\Scripts\python experiments\dsh_runtime_probe.py <probe-name> \
  --artifact-dir <new-or-empty-directory>
```

Supported probe names are:

1. `sidecar-lifecycle` — no Mongo or real LLM required;
2. `brain-task-lifecycle` — uses
   `MongoResolutionThreadRepository`, the public task-resolution service, the
   real sidecar, the real binding owner, and a guarded uniquely named test DB;
3. `transport-loss` — terminates the owned sidecar and observes public runtime
   readiness plus the typed task-capability failure result.

The probe owns one temporary data root and workspace, records every child PID,
uses bounded readiness/shutdown deadlines, terminates then kills only its owned
children when needed, and drops only its guarded test database. It never reads
or writes `.env` directly. Its machine-readable output is
`dsh_runtime_probe_result.v1` with `probe_name`, `started_at`, `finished_at`,
`tested_revision`, `status` (`passed|failed|blocked`), `observations`,
`processes`, `artifacts`, and `cleanup`. Exit codes are 0 passed, 1 failed, and
2 prerequisite-blocked. Gate 6 requires all three probes to return 0; blocked
Mongo evidence withholds the real-LLM hold point.

## Change Surface

### Plan lifecycle

- Create and maintain this plan.
- Mark the former active operational sign-off plan superseded and move it to
  `development_plans/archive/superseded/bugfix/` after review.
- Update `development_plans/README.md`.

### Production changes

- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
- `src/kazusa_ai_chatbot/cognition_resolver/state.py`
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py`
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
- `src/kazusa_ai_chatbot/cognition_shared/contracts.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
- `src/kazusa_ai_chatbot/service.py`, limited to preserving a
  `CognitionExecutionError.error_code` in `_operational_failure_metadata` and
  making `_operational_error_response.retryable` false after the one permitted
  pre-commit retry is exhausted.

`capabilities.py` is included to preserve the existing typed distinction
between valid task failure evidence and infrastructure/transport failure while
the timeout owner changes. `service.py` is included only for the demonstrated
typed-error projection defect above; its retry limit and retry decision remain
unchanged. No prompt, task-result schema, controller, RPC, sidecar production
source, console production source, or database owner is changed under the
current reviewed scope.

### Verification-only production owners

- `src/kazusa_ai_chatbot/task_resolution/service.py`
- `src/agentic_resolver/controller.py`
- `src/agentic_resolver/runtime.py`
- `src/agentic_resolver/rpc.py`
- `src/control_console/service_registry.py`
- `src/control_console/supervisor.py`

These paths receive process or owner-test evidence and no edit. A red probe
that requires changing one of them pauses implementation until the exact
defect and source/test amendment is reviewed in this plan.

### Probe and tests

- Create `experiments/dsh_runtime_probe.py`.
- Create `tests/integration/test_dsh_runtime_probe.py` with the exact nodes
  named in the traceability matrix.
- Modify `tests/test_agentic_resolver_sidecar_process.py` to consume shared
  black-box fixtures. Remove its compatibility-rotation case because that
  belongs to controller owner tests and currently mutates repository-private
  state.
- Modify `tests/test_cognition_resolver_loop.py` for V2 dependency and
  single-timeout behavior.
- Modify `tests/test_cognition_resolver_contracts.py` for the exact V2 schema
  and fail-closed state references.
- Modify `tests/unit/nodes/test_persona_supervisor2_l3_surface.py` for
  observation-derived evidence.
- Modify mapped task-resolution/controller/process tests only when their
  public boundary changes.
- Delete `scripts/validate_dsh_signoff.py`.
- Delete `tests/test_dsh_signoff_contract.py`.
- Delete `tests/test_dsh_signoff_manifest.py`.
- Delete `tests/test_dsh_user_message_e2e_live_llm.py`,
  `tests/test_dsh_internal_thought_e2e_live_llm.py`,
  `tests/test_dsh_self_cognition_e2e_live_llm.py`,
  `tests/test_dsh_scheduled_tick_e2e_live_llm.py`,
  `tests/test_dsh_tool_result_e2e_live_llm.py`,
  `tests/test_dsh_operational_e2e_live_llm.py`, and
  `tests/dsh_trigger_source_e2e_support.py` after their reusable process
  support moves to the probe owner.
- Create `tests/dsh_behavior_e2e_support.py` and
  `tests/test_dsh_behavior_live_llm.py` with these fixed semantic contracts:
  `test_live_foreground_task_resolution_is_grounded_and_character_owned`,
  `test_live_deferred_task_result_recurs_and_delivers_once`, and
  `test_live_internal_dsh_judgment_is_character_owned`.
- Modify `sidecars/dsh_resolution/tests/profile.spec.ts` and
  `sidecars/dsh_resolution/tests/standard_profile.spec.ts` to remove installed
  file-digest and composition-row-count pinning while retaining public boot,
  compatibility, capability, and collision behavior.
- Gate 6 Amendment 2 may modify
  `tests/test_cognition_observability_docs.py`,
  `tests/test_control_console_service_registry.py`,
  `tests/test_task_resolution_background_resume.py`,
  `tests/test_task_resolution_contracts.py`,
  `tests/test_dsh_task_resolution_live_db.py`,
  `tests/test_background_work_delivery.py`,
  `tests/test_background_work_jobs.py`,
  `tests/task_resolution_test_helpers.py`,
  `tests/test_task_resolution_dialog_live_llm.py`,
  `tests/unit/background_work/test_dsh_jobs.py`,
  `tests/unit/brain_service/test_cognition_graph_projection.py`, and
  `tests/unit/db/test_task_resolution_sessions.py` only to align stale
  assertions/fixtures with the implemented public DSH contracts and remove
  cross-test-module fixture imports. No production contract changes are
  authorized by this amendment.
- Update `tests/ownership/source_test_impact_manifest.json` to exact surviving
  behavioral nodes.

### Documentation

- `src/kazusa_ai_chatbot/cognition_resolver/README.md`
- `src/kazusa_ai_chatbot/task_resolution/README.md`
- `src/agentic_resolver/README.md`
- `sidecars/dsh_resolution/README.md`
- `docs/HOWTO.md`

## Source-To-Test Traceability

| Governed source/symbol | Semantic owner | Mode | Exact deterministic/process node IDs | Exact later live node | Regression prevented |
|---|---|---|---|---|---|
| `cognition_resolver/contracts.py::RequiredResolverEvidenceDependencyV2`; `cognition_resolver/contracts.py::validate_required_resolver_evidence_dependency` | Resolver structural contract | Deterministic contract | `tests/test_cognition_resolver_contracts.py::test_required_evidence_dependency_v2_accepts_reference_only`; `tests/test_cognition_resolver_contracts.py::test_required_evidence_dependency_v2_rejects_legacy_copied_fields` | `tests/test_dsh_behavior_live_llm.py::test_live_foreground_task_resolution_is_grounded_and_character_owned`; `tests/test_dsh_behavior_live_llm.py::test_live_deferred_task_result_recurs_and_delivers_once` | Copied state returns through a legacy shape. |
| `cognition_resolver/state.py::required_task_observation`; `cognition_resolver/state.py::validate_resolver_state`; `cognition_resolver/state.py::project_resolver_context` | Resolver state and prompt-safe projection | Deterministic contract | `tests/test_cognition_resolver_contracts.py::test_required_evidence_dependency_resolves_one_task_observation`; `tests/test_cognition_resolver_contracts.py::test_required_evidence_dependency_rejects_missing_or_wrong_kind_observation` | `tests/test_dsh_behavior_live_llm.py::test_live_foreground_task_resolution_is_grounded_and_character_owned`; `tests/test_dsh_behavior_live_llm.py::test_live_deferred_task_result_recurs_and_delivers_once` | Missing or unrelated evidence reaches L3. |
| `cognition_resolver/loop.py::_bind_required_evidence_dependency`; `cognition_resolver/loop.py::_mark_existing_dependency_blocked`; `service.py::_operational_failure_metadata`; `service.py::_operational_error_response` | Resolver recurrence lifecycle and service error projection | Deterministic graph integration | `tests/test_cognition_resolver_loop.py::test_required_dependency_references_single_task_observation`; `tests/test_cognition_resolver_loop.py::test_duplicate_task_blocker_replaces_dependency_reference_without_copying_state`; `tests/test_cognition_resolver_loop.py::test_invalid_required_dependency_fails_closed_before_l3`; `tests/test_service_event_logging.py::test_resolver_state_contract_retries_once_then_settles_operational_failure` | `tests/test_dsh_behavior_live_llm.py::test_live_deferred_task_result_recurs_and_delivers_once` | A blocker updates one of two semantic copies, invalid state reaches L3, or the service erases the typed failure/discloses an exhausted retry as retryable. The loop test asserts `error_code=resolver_state_contract`, `stage=cognition_resolver`, `safe_checkpoint=pre_state_commit`, `retryable=True`, and the current attempt count; the service test asserts exactly one retry followed by `error_code=resolver_state_contract`, `attempt_count=2`, `exhausted=True`, and `retryable=False` on the second identical error. |
| `cognition_resolver/loop.py::_execute_with_timeout` | Capability timing | Deterministic graph integration | `tests/test_cognition_resolver_loop.py::test_loop_records_timeout_observation_then_returns_to_cognition`; `tests/test_cognition_resolver_loop.py::test_task_resolution_uses_task_service_timeout_without_detached_resolver_task`; `tests/test_task_resolution_inline_promotion.py::test_inline_runtime_checkpoint_is_projected_without_reclassification` | `tests/test_dsh_behavior_live_llm.py::test_live_deferred_task_result_recurs_and_delivers_once` | Nested timeout returns while an unowned task mutates durable state. |
| `cognition_resolver/capabilities.py::_execute_task_resolution_request` | Typed task/infrastructure outcome | Deterministic capability integration | `tests/unit/cognition_resolver/test_capabilities.py::test_task_resolution_transport_failure_returns_blocked_observation`; `tests/unit/cognition_resolver/test_capabilities.py::test_background_task_resolution_transport_failure_returns_blocked_observation`; `tests/unit/cognition_resolver/test_capabilities.py::test_task_resolution_database_failure_returns_blocked_observation` | `tests/test_dsh_behavior_live_llm.py::test_live_foreground_task_resolution_is_grounded_and_character_owned`; `tests/test_dsh_behavior_live_llm.py::test_live_deferred_task_result_recurs_and_delivers_once` | Infrastructure exception escapes the resolver or becomes false task success. |
| `cognition_shared/contracts.py` dependency import/dead comparison removal | L3 input contract | Deterministic contract | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_surface_projects_current_task_resolver_dependency`; `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_terminal_resolver_surface_closes_stale_pending_plan` | `tests/test_dsh_behavior_live_llm.py::test_live_foreground_task_resolution_is_grounded_and_character_owned`; `tests/test_dsh_behavior_live_llm.py::test_live_deferred_task_result_recurs_and_delivers_once` | A dead copied-field validator remains as parallel authority. |
| `nodes/persona_supervisor2_l3_surface.py::_resolver_result_continuation_ref`; `nodes/persona_supervisor2_l3_surface.py::_resolver_result`; `nodes/persona_supervisor2_l3_surface.py::_task_resolver_result` | L3 evidence input | Deterministic node integration | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_surface_projects_current_task_resolver_dependency`; `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_surface_rejects_missing_required_task_observation_before_planning`; `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_surface_rejects_mismatched_tool_result_speak_continuation` | `tests/test_dsh_behavior_live_llm.py::test_live_foreground_task_resolution_is_grounded_and_character_owned`; `tests/test_dsh_behavior_live_llm.py::test_live_deferred_task_result_recurs_and_delivers_once` | L3 consumes copied dependency semantics or detached continuation. |
| Real sidecar boot/readiness/semantic worker | Sidecar process boundary | Real process with deterministic provider | `tests/integration/test_dsh_runtime_probe.py::test_sidecar_lifecycle_probe` | `tests/test_dsh_behavior_live_llm.py::test_live_foreground_task_resolution_is_grounded_and_character_owned` | Public auth rejection, independent sessions, missing-worker readiness, and semantic forwarding remain covered. |
| SQLite checkpoint/restart/replay | DSH session persistence | Real process with deterministic provider | `tests/integration/test_dsh_runtime_probe.py::test_sidecar_lifecycle_probe` | — | Green in-memory lifecycle loses state across a real process restart. |
| Public controller plus guarded Mongo task binding | Controller/task service/persistence | Guarded live DB plus real process | `tests/integration/test_dsh_runtime_probe.py::test_brain_task_lifecycle_probe`; `tests/unit/task_resolution/test_service.py::test_inline_checkpoint_promotes_same_bound_dsh_session_without_canceling_reasoning`; `tests/test_agentic_resolver_controller.py::test_controller_restart_reconciles_terminal_projection_and_lease` | `tests/test_dsh_behavior_live_llm.py::test_live_deferred_task_result_recurs_and_delivers_once` | Sidecar proof omits the Brain binding and Mongo reconciliation boundary. |
| Actual sidecar loss | RPC/capability readiness and failure projection | Real process fault injection | `tests/integration/test_dsh_runtime_probe.py::test_transport_loss_probe`; `tests/unit/agentic_resolver/test_rpc_readiness.py::test_transport_error_preserves_low_level_cause`; `tests/unit/agentic_resolver/test_rpc_readiness.py::test_runtime_readiness_delegates_to_authenticated_sidecar_health` | `tests/test_dsh_behavior_live_llm.py::test_live_foreground_task_resolution_is_grounded_and_character_owned` | Transport loss escapes as an untyped graph crash. |
| Control-console dependency order | Local process lifecycle | Deterministic supervisor integration | `tests/test_control_console_service_registry.py::test_default_registry_manages_dsh_between_brain_and_adapters`; `tests/test_control_console_supervisor.py::test_dependency_order_requires_brain_before_adapter_and_stops_dependents` | — | Adapter starts while the managed DSH dependency is unavailable. |
| Trigger-source deterministic eligibility | Source owners | Deterministic source integration | `tests/test_self_cognition_integration.py::test_internal_latch_case_hydrates_bound_user_profile`; `tests/test_self_cognition_integration.py::test_collect_scheduled_future_cognition_cases_preserves_source_scope`; `tests/test_self_cognition_integration.py::test_collect_commitment_due_cognition_cases_projects_calendar_runs`; `tests/unit/background_work/test_result_source.py::test_dsh_result_reenters_cognition_with_exact_goal_and_evidence_provenance`; `tests/unit/cognition_resolver/test_capabilities.py::test_task_capability_uses_runtime_readiness_without_legacy_fallback` | — | A stochastic model is used to prove deterministic source identity/readiness. |
| Character-owned internal DSH judgment | Brain interaction cognition | Deterministic enactment plus real LLM judgment | `tests/test_dsh_brain_interaction_decision.py::test_brain_semantic_decision_is_enacted_without_keyword_or_post_llm_reclassification`; `tests/test_dsh_brain_interaction_service.py::test_service_returns_internal_decision_without_checkpoint_or_delivery` | `tests/test_dsh_behavior_live_llm.py::test_live_internal_dsh_judgment_is_character_owned` | Deterministic code rewrites the character decision. |
| `sidecars/dsh_resolution/tests/profile.spec.ts` public profile behavior | Sidecar profile boundary | TypeScript component integration | `sidecars/dsh_resolution/tests/profile.spec.ts > profile > builds only the resolver profile through the strict profile factory`; `sidecars/dsh_resolution/tests/profile.spec.ts > profile > fails startup for an incompatible DSH dependency graph`; `sidecars/dsh_resolution/tests/profile.spec.ts > profile > rejects an intake whose route digest differs before session creation`; `sidecars/dsh_resolution/tests/profile.spec.ts > V2 profile invariants > rejects V1 epoch while accepting the installed compatible dependency graph` | — | Compatibility remains behavioral while installed package file digests cease to be a release oracle. |
| `sidecars/dsh_resolution/tests/standard_profile.spec.ts` public Standard composition behavior | Sidecar Standard composition | TypeScript component integration | `sidecars/dsh_resolution/tests/standard_profile.spec.ts > official Standard profile > adds Kazusa semantic tools without colliding with official Standard capabilities`; `sidecars/dsh_resolution/tests/standard_profile.spec.ts > official Standard profile > publishes the Kazusa semantic capability catalog` | — | Internal Standard row counts and tool ordering cease to substitute for boot, collision, and capability behavior. |
| Changed-source mapping fail-closed | Test-impact owner | Static impact validation | `tests/test_test_impact_manifest.py::test_unmapped_changed_source_fails_closed`; `tests/test_test_impact_manifest.py::test_stale_required_node_fails_closed` | — | Removing the sign-off manifest weakens production test ownership. |

The other two live behavior contracts are
`tests/test_dsh_behavior_live_llm.py::test_live_foreground_task_resolution_is_grounded_and_character_owned`
and
`tests/test_dsh_behavior_live_llm.py::test_live_deferred_task_result_recurs_and_delivers_once`.

The exact TypeScript evidence command for the two changed specs is:

```powershell
Set-Location -LiteralPath 'sidecars/dsh_resolution'
npm test -- tests/profile.spec.ts tests/standard_profile.spec.ts
```

## Implementation Order And Gates

### Gate 0 — Plan review

- Independent reviewer checks the audit evidence, target ownership, scope, and
  probe falsifiability.
- Resolve every blocker/high finding in this document.
- Mark the plan `in_progress` and supersede the former active DSH plan.

### Gate 1 — Baseline real-process probe

- Run only the four named real-process nodes.
- Record process, RPC, SQLite, semantic-worker, and cleanup evidence.
- Diagnose any red outcome before test expansion or production refactor.

### Gate 2 — Single-source resolver state

- Replace dependency V1 with V2 in one atomic source/test cutover.
- Remove copied state, needs, handles, capability, and continuation fields.
- Derive L3 task evidence from the referenced observation.
- Run exact resolver-state, resolver-loop, cognition, and L3 owner nodes.

### Gate 3 — Single timeout owner

- Remove the resolver-loop DSH task set and shielded timeout branch.
- Prove inline terminal, inline timeout-to-checkpoint, deferred promotion,
  cancellation, and unexpected failure ownership.
- Run the real-process probe again.

### Gate 4 — Public process and persistence verification

- Add the public CLI probe and its three integration nodes.
- Run the guarded Mongo controller/task lifecycle probe and require a pass.
- Run transport loss through runtime readiness and the typed capability result.
- Remove the process test that mutates `InMemoryResolutionThreadRepository`
  private state; controller compatibility rotation remains covered by
  `tests/test_agentic_resolver_controller.py::test_incompatible_scope_audience_profile_release_store_model_catalog_or_policy_rotates_segment`.
- Treat every verification-only production path as read-only. A required
  production correction returns to plan review before edit.

### Gate 5 — Test and sign-off cleanup

- Remove meta-signoff validators and exact test-topology assertions.
- Move reusable process orchestration into the permanent probe.
- Retain the three named user/character behavior contracts without making
  suite file count or node count a product invariant.
- Keep strict deterministic tests for permissions, auth, schemas, persistence,
  idempotency, safety, cancellation, and delivery.
- Run collection and inspect the surviving test inventory.

### Gate 6 — Non-LLM closure

- Run sidecar typecheck, build, and tests.
- Run all three permanent process probes; guarded Mongo is mandatory.
- Run exact mapped deterministic/integration tests.
- Run `scripts/validate_test_impact.py --check-all`.
- Run the full non-live suite.
- Run scoped Ruff, Python compilation, and `git diff --check`.
- Obtain independent implementation review.
- Record the exact tested commit/worktree fingerprint and all residual risks.

### Gate 7 — Real-LLM behavior sign-off

The 2026-09-05 review reopens Gates 3, 5, and 6. Complete the linked remediation
scope and independent non-E2E readiness review first. The latest user command
authorizes actual E2E execution. Run each selected
live behavior scenario individually, inspect its raw trace/durable evidence
and visible behavior, obtain independent character review, and either close
the plan or return a demonstrated defect to its non-LLM owner probe. Availability
of models alone cannot advance this gate.

## Acceptance Criteria Before The Real-LLM Gate

1. The former operational sign-off plan is superseded by this single active
   DSH plan.
2. The real sidecar boots and reports authenticated readiness through the
   official Standard composition.
3. A semantic tool plus terminal resolution completes through real process
   boundaries with deterministic provider responses.
4. Checkpoint/restart and terminal-commit/response-loss replay preserve one
   durable session and one outcome.
5. Required resolver evidence has one mutable semantic owner: the referenced
   observation.
6. Task resolution has one timeout/checkpoint owner and leaves no resolver-loop
   detached task registry.
7. Transport loss produces a typed bounded outcome with truthful readiness and
   terminal durable state.
8. No test asserts an exact number of DSH live files/cases, validator-owned
   artifact filenames, hidden stage counts, or prompt-owned phrases as release
   evidence.
9. Source-owner integration proves all five trigger-source admission/non-entry
   mechanics without using a stochastic model as a unit-test oracle.
10. The remaining live evaluation covers the three behavior contracts named
    by this plan because they require semantic judgment; suite topology and
    case count are not asserted by production or deterministic tests.
11. All retained security, authority, persistence, schema, safety, idempotency,
    cancellation, and delivery contracts have direct deterministic or real-
    process evidence.
12. Full non-live verification is green, or every unrelated pre-existing
    failure is recorded with exact evidence and does not mask a DSH failure.
13. Independent review finds no blocker/high design issue in the changed DSH
    boundary.
14. Documentation describes the implemented boundary and verification path.

## Final Real-LLM Acceptance

- A natural foreground task is appropriately admitted, grounded, and rendered
  without exposing DSH/control-plane internals.
- A natural deferred task reaches one result recurrence and one eligible
  delivery without duplicate task admission.
- A DSH question or approval request receives a believable character-owned
  judgment through the internal cognition boundary.
- Each run has a successful source trace, coherent durable lineage, complete
  cleanup, and an independent qualitative review.
- Any red run invalidates final sign-off and reopens only the demonstrated
  owning boundary.

## Roles

### implementation_owner

- **Responsibility:** implement the V2 dependency cutover, single timeout
  ownership, public probe, test cleanup, documentation, and evidence ledger.
- **Owned surface:** every path under `Production changes`, `Probe and tests`,
  `Documentation`, this plan, and its registry row.
- **Authority:** edit owned files; create/drop only a guarded unique test DB;
  create/remove only probe-owned temporary roots; start/stop only probe-owned
  child processes; run non-LLM commands through Gate 6 and individually inspect
  the user-authorized real-model cases after independent readiness review.
- **Applicable skills:** `development-plan`, `probe-first-engineering`,
  `local-llm-architecture`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` when triggered.
- **Capability floor:** senior Python/TypeScript architecture, asyncio task
  ownership, process/RPC/SQLite/Mongo integration, cognition-state contracts,
  and test-oracle design.
- **Independence requirement:** cannot approve its own plan or final non-LLM
  implementation.
- **Acceptance output:** scoped diff, baseline and final probe artifacts,
  guarded Mongo evidence, exact test results, residual-risk ledger, and a
  Gate 7 handoff naming only real-LLM work.
- **Entry gate:** independent plan review passes and status becomes
  `in_progress`.
- **Exit gate:** every Gate 1–6 criterion passes and no production edit exists
  outside the reviewed surface.

### independent_reviewer

- **Responsibility:** review the executable plan before edits and the full
  non-LLM implementation/evidence after Gate 6.
- **Owned surface:** read-only plan, repository, diff, test output, probe
  artifacts, and residual-risk evidence; plan findings only.
- **Authority:** pass or fail review and require remediation. It cannot edit
  implementation or waive an acceptance gate.
- **Applicable skills:** `development-plan`, `probe-first-engineering`,
  `local-llm-architecture`, `py-style`, and `test-style-and-execution`.
- **Capability floor:** independent system architecture, runtime/process
  reasoning, cognition ownership, failure taxonomy, and false-green test
  analysis.
- **Independence requirement:** separate from the implementation owner for the
  reviewed diff; a remediated diff receives a fresh review.
- **Acceptance output:** severity-ranked findings, exact evidence, explicit
  pass/fail, and confirmation that Gate 7 is the only remaining boundary.
- **Entry gate:** plan draft for the first review; complete Gate 6 evidence for
  final review.
- **Exit gate:** zero blocker/high findings, with medium residuals either fixed
  or explicitly accepted by the user.

### character_behavior_reviewer

- **Responsibility:** after model availability, judge the three named live
  behavior contracts from natural inputs, protected traces, typed results,
  visible output, and durable lineage.
- **Owned surface:** read-only live artifacts and final review decisions.
- **Authority:** pass or fail semantic behavior; cannot edit prompts, code,
  tests, or mechanical evidence.
- **Applicable skills:** `character-test`, `local-llm-architecture`, and
  `development-plan`.
- **Capability floor:** character judgment, provenance analysis, privacy, task
  lifecycle understanding, and distinction between paraphrase and false
  completion.
- **Independence requirement:** separate from the implementation owner and
  from any later prompt-remediation executor.
- **Acceptance output:** one evidence-based decision per behavior contract and
  one overall final-signoff recommendation.
- **Entry gate:** Gate 6 passes, models are available, and each live artifact
  is complete.
- **Exit gate:** all behavior decisions are explicit; any failure returns to
  its demonstrated owner.

## Historical Gate 6 Evidence — 2026-09-04

The entries below record checks actually reported on 2026-09-04. The later
review found defects outside those tests' demonstrated coverage. Their
historical gate/sole-remaining-gate statements are superseded by Document
Control, Review amendment 4, and the linked remediation plan. Its impact
matrix replaces affected test rows when its deletion/import cutover executes.

- **Tested implementation revision:** commit
  `210cdb8f381f116afb437b42472eb6bf88a75142`, dirty worktree digest
  `sha256:8db618256d408e1ddb536b0b79ae8dca6e668f7ce7a83e735919bffca68d51f5`.
  The digest is recorded by every final probe and precedes only this
  evidence-ledger/registry update.
- **Permanent probes:** sequential CLI executions returned 0 and persisted
  `dsh_runtime_probe_result.v1` under
  `test_artifacts/dsh_gate6_20260904_final_post_review/sidecar-lifecycle`,
  `.../brain-task-lifecycle`, and `.../transport-loss`. The sidecar probe
  observed authenticated Standard boot, semantic-worker execution, SQLite
  checkpoint/cold restart, and exact terminal-commit response-loss replay.
  The guarded Mongo probes observed `resolved`/`consumed_inline` and
  ready-before-loss `failed`/`blocked`/`faulted`; both unique databases record
  `dropped` cleanup.
- **Sidecar:** `npm run typecheck` and `npm run build` returned 0; `npm test`
  passed 14 files and 103 tests.
- **Mapped verification:** the 40-node DSH deterministic/integration matrix
  passed. `scripts/validate_test_impact.py --check-all` validated 495 exact
  nodes. The two Mongo-backed permanent-probe nodes, now marked `live_db`,
  passed explicitly in 11.09 seconds. The 58 directly affected resolver/L3
  fixture tests passed after moving their shared builder out of a sibling test
  module.
- **Full non-live suite:** the final command was
  `pytest -qq --tb=short -m "not live_db and not live_llm" --deselect
  tests/test_rag_phase3_capability_agents.py::test_conversation_evidence_marks_partial_multi_target_result_unresolved`.
  It passed 3,390 tests, skipped 4, deselected 497, and returned 0 in 244.04
  seconds. A preceding post-review aggregate run passed 3,389 and exposed one
  terminal-response-loss timeout; the exact node passed in isolation, the
  duplicated raw five-second wait was replaced by the probe-owned bounded
  helper, and the clean aggregate rerun passed.
- **Static verification:** every changed Python file compiles; scoped
  `E4,E7,E9,F,I` Ruff checks and full Ruff checks for the new probe/support
  files pass; `git diff --check` passes; the DSH fixture-coupling search has no
  sibling-test imports. The three exact Gate 7 nodes compile and collect under
  `--collect-only -m live_llm` without execution.
- **Residual evidence:** when the unchanged node
  `tests/test_rag_phase3_capability_agents.py::test_conversation_evidence_marks_partial_multi_target_result_unresolved`
  was included, it attempted its real selector and failed with
  `openai.APIConnectionError: Connection error` because no model is available.
  The file and owning RAG production source are unchanged and outside this DSH
  plan, so the final DSH command deselects exactly that node. Gate 7 remains the
  only DSH acceptance boundary; expected Windows connection-reset/nonzero
  child exits during intentional response-loss injection remain visible in
  probe artifacts and are accepted only when exact durable replay passes.

## Progress

- [x] Read every DSH-specific plan and associated plan registry/history.
- [x] Inspect current DSH subsystem documentation, source ownership, test size,
  and the stopped operational diff.
- [x] Identify the duplicated evidence and split-timeout root designs.
- [x] Complete independent plan review.
- [x] Supersede the former active DSH plan and begin Gate 1.
- [x] Gate 1 passed on 2026-09-04: the four named real-process sidecar nodes
  passed in 17.97 seconds. The restart case produced one provider-server
  `ConnectionAbortedError` while the test intentionally severed the request;
  authenticated boot, semantic-worker RPC, cold resume, and exact terminal
  replay all passed.
- [x] Gate 2 passed on 2026-09-04: Amendment 1 passed independent review; the
  repeated run passed all 11 mapped V2, loop, service-retry, and L3 nodes in
  1.58 seconds. Repository search found no V1 type, schema string, or copied
  dependency-field reader/writer under `src/` or Python tests.
- [x] Gate 3 passed on 2026-09-04: five timeout/checkpoint owner nodes passed in
  1.09 seconds, the loop-owned detached task registry is absent, and the four
  real-process sidecar nodes passed again in 12.84 seconds. The intentional
  restart severed provider sockets and produced expected connection-reset/
  aborted stderr while cold resume and exact terminal replay passed.
- [x] Gate 4 passed on 2026-09-04: the permanent public CLI now owns three
  falsifiable probes and writes `dsh_runtime_probe_result.v1` artifacts with
  revision, process, observation, artifact, and cleanup evidence. Manual
  execution passed sidecar lifecycle, guarded Mongo brain/task lifecycle, and
  transport-loss classification. The first brain runs exposed and corrected
  invalid probe evidence plus a nonexistent thread convenience property; the
  first transport run exposed incomplete durable-collaborator wiring. The
  final three registered subprocess integration nodes passed in 22.07 seconds.
  The brain probe observed `resolved`, `consumed_inline`, and an active Mongo
  thread before dropping its unique test database. The transport probe
  observed ready-before-loss, typed failed/blocked evidence, and a faulted
  binding before cleanup.
- [x] Gate 5 passed on 2026-09-04: removed the meta-signoff validator, its two
  self-referential tests, six trigger/operational live wrappers, and the
  3,611-line trigger-source harness. Replaced them with three fixed behavior
  contracts plus one isolated guarded-system support module. The sidecar
  process spec now consumes the permanent probe owner and is 239 lines instead
  of 965; the five mapped real-process nodes passed in 16.43 seconds. The two
  public-behavior TypeScript specs passed 7 tests. All three Gate 7 nodes
  compile and collect without execution. Full manifest collection resolved
  495 exact required nodes, and repository-wide collection completed without
  an error. One direct live command began because the environment already set
  `KAZUSA_RUN_LIVE_LLM=1`; it was stopped without producing Gate 7 evidence,
  its exact owned process tree was terminated, and its uniquely named guarded
  database was confirmed dropped. Gate 7 remains wholly unsatisfied.
- [x] Gate 6 passed on 2026-09-04 with the evidence above and a fresh
  independent implementation review reporting zero blocker, high, or medium
  findings.
- [x] Stop with an explicit Gate 7 handoff while real models are unavailable.
- [x] 2026-09-05 review: reproduce cancellation ownership failure; review
  production/test coupling, live input/oracle quality, and third-party scope.
- [x] Record a bounded remediation draft and reopen Gates 3, 5, and 6.
- [ ] Execute the linked remediation after explicit implementation direction.
- [ ] Re-establish independent non-E2E readiness before a future authorized
  live behavior campaign.
