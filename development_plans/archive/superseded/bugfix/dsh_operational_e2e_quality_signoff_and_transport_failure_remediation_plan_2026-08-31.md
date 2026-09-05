# DSH Operational E2E Quality Sign-Off And Transport Failure Remediation Plan

## Summary

- **Goal:** replace the isolated DSH trigger matrix as the sole release oracle
  with a layered sign-off that detects sidecar startup, readiness, transport,
  graph-propagation, source-lineage, persistence, result, dialog, and delivery
  failures before DSH is declared production-ready.
- **Status:** superseded on 2026-09-04 by
  `dsh_probe_first_architecture_cleanup_and_final_signoff_plan_2026-09-04.md`.
- **Scope boundary:** DSH process ownership and readiness, task-resolution
  failure projection, terminal-failure telemetry, the ten canonical
  trigger-source live cases, and an opt-in configured-running-service canary.
- **Change direction:** retain two honest cases for each canonical trigger
  source, strengthen their shared oracle, add deterministic failure gates, add
  console-managed lifecycle coverage, and require a live Christchurch weather
  canary against the configured service without rewriting its DSH endpoint.
- **Acceptance state:** open. No DSH quality or production-readiness claim is
  permitted until every mandatory gate is green with zero skipped or xfailed
  nodes and every live artifact is inspected.
- **Execution state:** stopped for handoff on 2026-08-31. Final E2E execution
  is locked until the complete pre-E2E failure-point gate below is evidenced.
  Production sign-off is withheld.

## Binding Handoff Rules — 2026-08-31

These user-owned rules supersede any earlier plan text or execution practice
that used live E2E execution to discover DSH defects.

1. **DSH handling follows the project cognition-chain error policy without a
   separate fallback convention.** Each producing stage first uses the
   canonical JSON parser and the contract's permitted deterministic structural
   repair or bound normalization. A non-recoverable structural or semantic
   candidate becomes a typed contract error owned by that producing stage.
   The same stage performs bounded regeneration or complete replacement using
   the same semantic context, while the invalid candidate remains excluded
   from action, persistence, scheduling, dialog, and delivery. After the
   stage's explicit cap, the stage fails closed with a typed
   `CognitionExecutionError` carrying its error code, stage, attempt count,
   safe checkpoint, and retryability. The whole cognition graph may retry only
   through the existing deterministic safe-checkpoint policy; otherwise the
   service emits and settles the existing typed operational failure.
2. **The final E2E suite is never a debugging method.** Before any final E2E
   node runs, the implementation owner must map every mechanical failure point,
   implement its containment behavior, and prove that behavior through an
   exact deterministic owner test or process-integration test with controlled
   fault injection.
3. **The final E2E suite is the final production quality sign-off for DSH
   integration.** It is not a development feedback loop and is not an
   administrative checkbox used merely to close this plan. Its execution
   begins only after implementation, deterministic verification, process
   verification, full non-live verification, failure-map review, and
   independent pre-sign-off review are complete.
4. If a final E2E case fails, DSH sign-off is immediately invalid. The owner
   stops the live run, returns the failure to the pre-E2E failure map, adds the
   missing deterministic reproduction and containment proof, completes the
   remediation and non-live gates, creates a fresh code fingerprint, and only
   then starts a new final sign-off run. Repeated live reruns cannot be used to
   develop or tune the fix.

No additional user approval gate is inserted inside this already approved
scope. A future implementation owner may resume the declared work when ready,
but the objective pre-E2E gate below controls when final E2E execution becomes
eligible.

The rule above permits typed errors and typed fail-closed exhaustion. It
forbids DSH code from bypassing the cognition-chain policy through raw
exceptions, accepting or rewriting an invalid semantic candidate, allowing a
rejected candidate to reach a downstream consumer, inventing an unowned
fallback response, or retrying from an unapproved checkpoint.

The authoritative policy is `AGENTS.md::LLM Error State Handling`. Its runtime
carriers are
`src/kazusa_ai_chatbot/cognition_shared/contracts.py::CognitionExecutionError`
and the safe-checkpoint graph policy in
`src/kazusa_ai_chatbot/service.py::_can_retry_cognition_failure`. This plan
does not redefine or weaken those contracts.

## Confirmed Decisions

- The production trace `llmtrace_512da03221a94a2688fe8c87dd3a1a4c`
  is the captured regression. Cognition selected `task_resolution_request`,
  then `system.health` failed at the sidecar transport boundary and escaped as
  a graph failure with zero dialog.
- Exactly five canonical cognition trigger sources remain:
  `user_message`, `internal_thought`, `self_cognition`, `scheduled_tick`, and
  `tool_result`.
- Each source retains exactly two live source-owner E2Es. The reachable plain
  `self_cognition` producer is targetless group review and therefore proves
  two non-entry cases. `tool_result` remains recurrence-closed and proves two
  non-entry cases. The other six cases prove positive DSH entry.
- Live model behavior is judged from source-owned facts and user-visible
  behavior. Exact prompt wording, exact prose, exact internal stage counts,
  and model-selected tool choreography are not release assertions.
- Readiness, process lifecycle, typed failure behavior, telemetry truthfulness,
  durable lineage, and final status are deterministic contracts and receive
  strict assertions.
- The configured-running-service canary uses the process environment and the
  already configured Brain/DSH endpoints. It does not allocate substitute
  ports, credentials, or workspace roots.
- The user's minimum confidence requirement is an evidence gate: sign-off is
  withheld unless all known high-severity failure classes in this plan are
  closed, every deterministic and process gate passes, all twelve live DSH
  cases pass, the configured-service canary passes, artifact review finds no
  semantic failure, and zero known uncovered DSH mechanical path or
  cognition-chain error-policy violation remains. The final report states a
  calibrated confidence value of at least
  95% only when that complete evidence exists; confidence measures the
  completeness and repeatability of the proof rather than discounting any
  known failure.
- The earlier batch-first live run is retained only as historical failure
  evidence. The binding handoff rules now require all further defect discovery,
  reproduction, and remediation to occur below E2E through the pre-E2E failure
  map and deterministic fault-injection coverage.

## Scope And Change Direction

### Must Do

- Preserve the original transport exception text inside
  `RpcTransportError` so connection refusal, timeout, reset, and HTTP failure
  remain distinguishable in logs and trace evidence.
- Add one public, bounded resolver-runtime readiness probe backed by the
  authenticated sidecar `system.health` contract.
- Make Brain DSH health reflect the live sidecar probe rather than local
  object construction alone.
- Make the default control-console registry own the DSH sidecar and place it
  between Brain and adapters in the declared dependency graph.
- Convert DSH task-execution outcomes that the resolver contract defines as
  failed evidence into a blocked, prompt-safe resolver observation for renewed
  cognition. Route malformed model candidates, contradictory semantic state,
  and exhausted regeneration through the existing typed cognition-chain
  fail-closed contract instead of relabelling them as task evidence.
- Record terminal graph errors as unrecovered.
- Strengthen all ten trigger-source E2Es to require successful source traces,
  zero terminal runtime/pipeline errors, exact entry or non-entry lineage,
  typed results where applicable, and durable evidence artifacts.
- Add a sidecar-loss live E2E that replays the captured weather-shaped user
  request and requires a coherent visible failure response without graph
  failure.
- Add a configured-running-service Christchurch weather canary that first
  proves authenticated DSH readiness, then requires real DSH admission and a
  grounded terminal or durably delivered result.
- Add one sign-off manifest validator that rejects missing nodes, skipped or
  xfailed results, missing artifacts, failed technical gates, unreviewed live
  behavior, and an absent configured-service canary.
- Update the runbook and DSH/control-console ICDs to describe the actual
  managed lifecycle and mandatory sign-off sequence.

### Deferred

- Prompt tuning for one captured noun, phrase, or expected tool sequence.
- New cognition sources, new DSH capabilities, new semantic tools, and new
  adapter protocol fields.
- Compatibility aliases for the former manual sidecar lifecycle.
- Production database cleanup or deletion of canary history.
- Behavior outside the declared DSH integration boundary. The DSH sign-off
  boundary itself requires complete mechanical coverage and zero known
  cognition-chain error-policy violation.

## Target State

```text
control console
  -> starts Brain
  -> starts and owns DSH sidecar after Brain is available
  -> starts adapters only after Brain and DSH sidecar are available

Brain DSH readiness
  -> local interaction/store/judge checks
  -> authenticated sidecar system.health
  -> ready only when both boundaries agree

cognition task request
  -> DSH succeeds: typed evidence returns through cognition/dialog
  -> DSH integration fails: typed blocked observation returns through cognition
  -> invalid DSH/cognition candidate: producing stage performs bounded regeneration
  -> regeneration exhausts: typed fail-closed result carries owner and checkpoint
  -> whole graph retries only when the existing safe-checkpoint policy permits
  -> visible response remains character-owned
  -> terminal graph failure is not produced by a bounded DSH dependency error
```

The ten-source matrix proves semantic entry and recurrence ownership inside a
clean isolated stack. The lifecycle test proves the production process graph.
The configured-service canary proves the actual running endpoint and
configuration. All three are required and no one layer substitutes for
another.

## Mandatory Skills

- `development-plan` governs lifecycle, traceability, evidence, and closeout.
- `test-style-and-execution` governs deterministic versus live-LLM assertions,
  execution, artifact inspection, and failure classification.
- `py-style` governs every changed Python file.
- `local-llm-architecture` governs the semantic/deterministic boundary and
  prohibits prompt-shaped remediation.
- `llm-trace-debug` governs comparison of the captured production regression
  with the final canary evidence.

## Mandatory Rules

- Use `venv\Scripts\python` for Python and pytest commands.
- Keep final E2E execution locked until every item in `Pre-E2E Entry Gate` is
  complete and independently reviewed.
- Use deterministic unit, contract, component, and process-integration tests
  with controlled fault injection to discover, reproduce, and remediate every
  mapped DSH failure point.
- After the pre-E2E gate passes, run final live-LLM nodes individually and
  inspect each retained artifact before proceeding to the next sign-off node.
- Preserve the earlier batch-first dossier as historical evidence only; it
  supplies no sign-off credit for a later code fingerprint.
- A failed final E2E case invalidates the entire sign-off run and returns the
  work to deterministic failure mapping and remediation before any new live
  execution.
- No exact prose or prompt-owned keyword may become a pass condition.
- Input-, source-, schema-, or evidence-owned literals may be asserted only at
  the boundary that owns them.
- A source trace in `failed`, `running`, or absent state fails the case.
- A response carrying `operational_error`, a graph-error pipeline event, or a
  terminal runtime error fails the case even when another assertion passes.
- The configured-service canary may be skipped only outside sign-off. Any skip
  makes final sign-off fail.

## Pre-E2E Entry Gate

Final E2E execution remains prohibited until all items below are complete:

1. Produce a complete DSH mechanical failure-point ledger across this exact
   workflow:

   ```text
   trigger producer
     -> typed intake and source identity
     -> cognition capability selection and schema validation
     -> resolver admission, deduplication, cycle and timeout handling
     -> readiness and authenticated RPC transport
     -> binding creation and durable persistence
     -> sidecar session, capability execution and evidence production
     -> inline result or accepted background handoff
     -> resolver observation and required-evidence dependency
     -> cognition recurrence and terminal decision
     -> L3 projection and visible/private surface ownership
     -> action execution and adapter delivery
     -> background callback, promotion and recurrence
     -> terminal persistence, cleanup and service lifecycle
   ```

2. For every failure point, record the producer, trigger, exception or invalid
   state, containment owner, typed disposition, allowed state transition,
   persistence effect, visible/private behavior, telemetry evidence, cleanup
   requirement, and exact deterministic pytest node ID.
3. Prove every row with controlled fault injection. Coverage must include
   malformed and contradictory state, unavailable dependencies, authentication
   failure, refused/reset/timed-out transport, sidecar error responses,
   cancellation, duplicate/repeated requests, max-cycle handling, inline and
   background timeouts, persistence failure, callback failure, L3 projection,
   action execution, adapter delivery, cleanup, and process termination.
4. Prove the required DSH error disposition by failure class:
   - a valid DSH task-execution failure returns typed resolver evidence to the
     producing cognition stage;
   - recoverable structural or bound violations receive only the contract's
     deterministic repair or normalization;
   - every non-recoverable model candidate returns a typed contract error to
     its producing stage and triggers bounded owner regeneration;
   - every rejected candidate remains excluded from state mutation, action,
     persistence, scheduling, dialog, and delivery;
   - exhausted owner regeneration produces a typed
     `CognitionExecutionError` with complete stage, attempt, checkpoint, and
     retryability metadata;
   - a whole-chain retry occurs only when the typed error is retryable at the
     approved `pre_state_commit` checkpoint and remains within the service cap;
   - every other exhausted error follows the existing typed operational
     failure and settlement path; and
   - no raw DSH exception, contradictory state, or ad hoc fallback bypasses
     these owner and chain boundaries.
5. Collect and pass every exact mapped deterministic node, every required
   component/process node, the changed-source impact validator, and the full
   non-live suite with zero skips, xfails, hangs, leaked work, or incomplete
   cleanup.
6. Complete an independent pre-E2E review that confirms the ledger covers the
   entire declared workflow and that every row has passing deterministic
   evidence. Create a fresh code fingerprint only after this review passes.

The existing `Test Impact And Traceability` table is an initial source/test
matrix, not the complete failure-point ledger. It does not unlock E2E in its
current form.

## Execution Roles

### implementation_owner

- **Responsibility:** implement the bounded production and test changes and
  maintain this execution record.
- **Owned surface:** the exact files in `Change Surface` except the independent
  review evidence.
- **Authority:** edit the declared production, test, documentation, and plan
  files; run guarded ephemeral test databases and explicitly opt-in live
  tests; preserve existing production data.
- **Applicable skills:** all skills listed under `Mandatory Skills`.
- **Capability floor:** system-level Python/TypeScript process-boundary
  reasoning, async service testing, DSH contracts, Mongo test isolation, and
  live-LLM artifact review.
- **Independence requirement:** none.
- **Acceptance output:** scoped diff, exact deterministic results, twelve
  inspected live dossiers, configured-service canary evidence, and updated
  plan ledger.
- **Gate:** starts after baseline capture; exits only when all mapped tests are
  collected and run and no mandatory gate is red.

### signoff_reviewer

- **Responsibility:** independently inspect the final diff and evidence for
  false-green gates, overfitting, missing lifecycle coverage, and unclosed
  production risk.
- **Owned surface:** read-only access to the repository diff, plan, test
  outputs, and retained artifacts; review findings may be appended to this
  plan.
- **Authority:** pass or fail sign-off; no remediation edits.
- **Applicable skills:** `development-plan`, `test-style-and-execution`,
  `local-llm-architecture`, and `py-style` for review.
- **Capability floor:** independent architecture, test-oracle, and failure-mode
  analysis with access to all evidence.
- **Independence requirement:** must not be the executor that authors a
  remediation under review.
- **Acceptance output:** explicit pass/fail findings and a zero-known-gap
  judgment over the declared DSH mechanical boundary.
- **Gate:** begins only after implementation verification is complete; any
  finding returns remediation to `implementation_owner`, followed by a fresh
  review.

## Test Impact And Traceability

| Governed path and contract | Semantic owner | Exact deterministic pytest nodes | Supplemental live/process nodes | Mode | Regression prevented |
| --- | --- | --- | --- | --- | --- |
| `src/agentic_resolver/rpc.py::_HttpRpcTransport._send_sync` | Sidecar RPC transport | `tests/unit/agentic_resolver/test_rpc_readiness.py::test_transport_error_preserves_low_level_cause` | `tests/test_dsh_operational_e2e_live_llm.py::test_live_sidecar_loss_returns_bounded_response_without_graph_failure` | deterministic unit plus isolated live LLM/service | All connection failures collapse into an unhelpful wrapper and evade diagnosis. |
| `src/agentic_resolver/controller.py` and `runtime.py` readiness interface | Resolver control plane | `tests/unit/agentic_resolver/test_rpc_readiness.py::test_runtime_readiness_delegates_to_authenticated_sidecar_health` | `tests/test_dsh_control_console_process.py::test_console_managed_sidecar_reaches_authenticated_readiness` | deterministic unit plus real process integration | Brain or operators infer readiness without the mounted sidecar identity. |
| `src/kazusa_ai_chatbot/service.py::_dsh_interaction_health` and `/runtime/dsh/health` | Brain composition/readiness | `tests/unit/brain_service/test_dsh_task_readiness.py::test_dsh_health_requires_live_sidecar_readiness`; `tests/unit/brain_service/test_dsh_task_readiness.py::test_dsh_health_is_unavailable_when_sidecar_probe_fails` | configured-service canary node below | deterministic unit plus configured live service | Brain advertises task resolution ready while its endpoint is unreachable. |
| `src/control_console/service_registry.py::default_service_registry` | Local service lifecycle | `tests/test_control_console_service_registry.py::test_default_registry_manages_dsh_between_brain_and_adapters` | `tests/test_dsh_control_console_process.py::test_console_managed_sidecar_reaches_authenticated_readiness` | deterministic unit plus process integration | Manual/stale sidecar process is omitted from the supported startup path. |
| `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py::_execute_task_resolution_request` | Resolver capability boundary | `tests/unit/cognition_resolver/test_capabilities.py::test_task_resolution_transport_failure_returns_blocked_observation`; `tests/unit/cognition_resolver/test_capabilities.py::test_background_task_resolution_transport_failure_returns_blocked_observation` | `tests/test_dsh_operational_e2e_live_llm.py::test_live_sidecar_loss_returns_bounded_response_without_graph_failure` | patched boundary plus live LLM/service | Typed DSH integration errors escape as a graph failure and suppress dialog. |
| `src/kazusa_ai_chatbot/service.py::_process_queued_chat_item` terminal telemetry | Brain service observability | `tests/test_service_event_logging.py::test_graph_failure_records_runtime_error_and_failed_pipeline` | sidecar-loss E2E | deterministic service unit plus live service | A failed turn is incorrectly labelled recovered. |
| `tests/dsh_trigger_source_e2e_support.py` technical oracle and evidence dossier | DSH sign-off test infrastructure | `tests/test_dsh_signoff_contract.py::test_signoff_manifest_names_exact_required_live_nodes`; `tests/test_dsh_signoff_contract.py::test_case_oracle_rejects_failed_trace_and_terminal_runtime_error` | all ten trigger-source nodes below | deterministic test-infrastructure unit plus isolated live LLM/DB/service | A failed trace, graph error, or runtime error passes because the oracle checks only that execution stopped. |
| `tests/test_dsh_user_message_e2e_live_llm.py` | Public user-message entry | manifest contract node above | `::test_live_user_message_local_fact_reaches_dsh`; `::test_live_user_message_background_summary_reaches_dsh` | isolated live LLM/DB/service | Public foreground or delayed user tasks fail to enter, settle, recur, or deliver. |
| `tests/test_dsh_internal_thought_e2e_live_llm.py` | Durable internal-thought entry | manifest contract node above | `::test_live_internal_thought_file_check_reaches_dsh`; `::test_live_internal_thought_comparison_reaches_dsh` | isolated live LLM/DB/service | Identity-bound latches lose source authority or fail DSH admission. |
| `tests/test_dsh_self_cognition_e2e_live_llm.py` | Reachable plain self-cognition non-entry | manifest contract node above | `::test_live_targetless_group_review_omits_dsh_task_resolution`; `::test_live_promoted_group_review_omits_dsh_task_resolution` | isolated live LLM/DB/service | Targetless group review fabricates a user and enters DSH. |
| `tests/test_dsh_scheduled_tick_e2e_live_llm.py` | Scheduled identity-bound entry | manifest contract node above | `::test_live_commitment_due_tick_reaches_dsh`; `::test_live_scheduled_future_tick_reaches_dsh` | isolated live LLM/DB/service | Due commitment or future cognition loses user/run lineage or result delivery. |
| `tests/test_dsh_tool_result_e2e_live_llm.py` | Result recurrence closure | manifest contract node above | `::test_live_resolved_tool_result_delivers_without_recursive_dsh`; `::test_live_failed_tool_result_settles_without_recursive_dsh` | isolated live LLM/DB/service | A terminal result recursively re-enters DSH or a failed result becomes false success. |
| `tests/test_dsh_operational_e2e_live_llm.py` configured canary | Actual configured Brain/DSH boundary | manifest contract node above | `::test_live_configured_christchurch_weather_request_completes_through_dsh`; `::test_live_sidecar_loss_returns_bounded_response_without_graph_failure` | configured and isolated live LLM/service | The isolated harness masks endpoint/lifecycle failure or the captured first command fails again. |

## Change Surface

### Create

- `tests/unit/agentic_resolver/test_rpc_readiness.py`
- `tests/test_dsh_signoff_contract.py`
- `tests/test_dsh_control_console_process.py`
- `tests/test_dsh_operational_e2e_live_llm.py`
- `scripts/validate_dsh_signoff.py`

### Modify

- `src/agentic_resolver/rpc.py`
- `src/agentic_resolver/controller.py`
- `src/agentic_resolver/runtime.py`
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/brain_service/contracts.py`
- `src/control_console/service_registry.py`
- `src/control_console/supervisor.py`
- `sidecars/dsh_resolution/src/main.ts`
- `sidecars/dsh_resolution/tests/process.spec.ts`
- `tests/test_agentic_resolver_sidecar_process.py`
- `tests/dsh_trigger_source_e2e_support.py`
- the five `tests/test_dsh_*_e2e_live_llm.py` trigger-source files
- `tests/unit/brain_service/test_dsh_task_readiness.py`
- `tests/unit/cognition_resolver/test_capabilities.py`
- `tests/test_control_console_service_registry.py`
- `tests/test_control_console_supervisor.py`
- `tests/test_service_event_logging.py`
- `src/agentic_resolver/README.md`
- `src/kazusa_ai_chatbot/task_resolution/README.md`
- `src/kazusa_ai_chatbot/brain_service/README.md`
- `src/control_console/README.md`
- `docs/HOWTO.md`
- `development_plans/README.md`
- this plan

### Keep

- DSH semantic catalog, Standard profile, task-result schemas, cognition
  prompts, dialog prompts, adapter wire contracts, and the five-source enum.
- Existing component tests that remain outside the final sign-off manifest.

### Delete

- No current test is deleted by this plan. Any live test omitted from the
  final sign-off manifest must be separately classified and removed under the
  user's standing instruction rather than silently retained.

## Agent Autonomy Boundaries

- The implementation owner may choose local helper placement and artifact
  serialization mechanics within the declared files.
- The implementation owner may add a mapped deterministic node when an exact
  changed production owner otherwise lacks direct coverage; the plan matrix
  must be updated before that edit is accepted.
- Changes to prompt semantics, trigger-source meaning, DSH tool/catalog
  contracts, adapter APIs, database schemas, production data, or canary
  cleanup require a plan amendment or user decision.
- A live failure is evidence. The implementation owner classifies the owning
  boundary before editing an assertion, prompt, or runtime path.

## Verification

1. Complete and independently review the full pre-E2E failure-point ledger.
2. Collect and run every exact deterministic and process fault-injection node
   mapped by that ledger.
3. Run the project impact validator for every changed production path.
4. Run the full non-live suite and resolve every DSH-relevant failure below the
   E2E boundary.
5. Complete independent pre-E2E review and create a fresh code fingerprint.
6. Run each of the ten trigger-source live nodes individually as final sign-off
   and inspect its technical dossier, source trace, DSH lineage, callback
   evidence, and behavior review input.
7. Run and inspect the sidecar-loss sign-off node.
8. Run the configured-service Christchurch weather canary against the actual
   configured endpoint and inspect its trace/binding/result evidence.
9. Run `scripts/validate_dsh_signoff.py` against the complete retained run.
10. Obtain independent final sign-off review. Any live failure invalidates the
    run and returns execution to step 1 before another E2E attempt.

## Acceptance Criteria

- Authenticated DSH readiness becomes unavailable within the bounded probe
  interval when the sidecar is unavailable and becomes ready when the sidecar
  identity is valid.
- The default console registry owns the sidecar and prevents adapter startup
  before both Brain and sidecar dependencies are available.
- A sidecar transport failure produces a failed resolver observation and a
  coherent visible response, with no graph error, no terminal runtime event,
  and no false completion claim.
- Terminal graph errors record `recovered=false`.
- The complete pre-E2E failure-point ledger has no uncovered DSH mechanical
  edge, and every row has an exact collected and passing deterministic or
  process fault-injection test.
- Every mapped DSH error follows its exact cognition-chain disposition:
  deterministic repair where permitted, typed owner error, bounded owner
  regeneration, rejected-candidate isolation, typed fail-closed exhaustion,
  and safe-checkpoint graph retry only where the existing policy permits it.
- No mapped DSH failure bypasses the policy as a raw exception, contradictory
  downstream state, accepted invalid candidate, ad hoc semantic rewrite, or
  unapproved retry.
- Every positive-entry case has exactly one source-bound DSH binding, exactly
  one matched sidecar session, a successful source trace, and a typed terminal
  or durably delivered result grounded in retained evidence.
- Every non-entry case has a successful source trace, zero DSH bindings, zero
  sidecar sessions, and coherent visible, silent, or failure behavior.
- The configured-service canary replays the Christchurch weather request,
  proves actual authenticated readiness, reaches DSH, and produces grounded
  current-weather evidence or a durably delivered grounded result without an
  operational error.
- All exact nodes collect and pass with zero skips, xfails, hangs, leaked child
  processes, or incomplete guarded-database cleanup.
- Independent review finds no material false-green, prompt-overfit, lifecycle,
  or evidence gap.
- The final report gives a confidence value of at least 95% only after every
  criterion above is evidenced. Otherwise the report states that sign-off is
  withheld and names the blocking evidence.
- The final handoff presents the complete adapter-to-Brain-to-cognition-to-DSH
  and result-delivery workflow, maps every supported entry and non-entry route
  to its sign-off evidence, explains why the combined suite covers production
  operation, accounts for every user requirement, and derives the confidence
  value from passed evidence. Sign-off requires zero known uncovered DSH
  mechanical paths or cognition-chain error-policy violations; any known gap
  withholds sign-off.

## Progress Checklist

- [x] Capture clean worktree baseline and inspect the current DSH tests,
  runtime, process registry, readiness endpoint, and production failure trace.
- [x] Fix the target contracts and exact source-to-test matrix in this plan.
- [x] Add expected-red deterministic and operational tests.
- [x] Implement and verify the earlier resolver-to-L3 terminal-surface
  propagation slice; the mapped owner suite passed 91 tests before the latest
  unverified resolver edits.
- [x] Record two historical live passes under code fingerprint
  `sha256:f91179c93a49ab5dc356e5711b357f390bbd51423a6840118c00d6cc782f2325`:
  sidecar loss and user-message local fact.
- [x] Record the historical user-message background-summary failure under the
  same fingerprint and withhold sign-off.
- [x] Stop final E2E execution and establish the binding pre-E2E handoff gate.
- [ ] Audit the current worktree and reconcile every changed production path
  with an exact deterministic owner-test row.
- [ ] Complete the full DSH failure-point ledger and deterministic
  fault-injection matrix before any further E2E execution.
- [ ] Review and verify the latest partial resolver edits described in
  `Stopped Work And Current Verification State`.
- [ ] Complete all deterministic, process, impact, full non-live, and
  independent pre-E2E gates.
- [ ] Create a fresh code fingerprint and run the twelve-node E2E suite once as
  final production sign-off.
- [ ] Complete independent review and confidence judgment.
- [ ] Archive this plan only after every acceptance criterion passes.

## Execution Evidence

- **Baseline:** `git status --short` was clean before this plan was created.
- **Captured regression:** trace and binding evidence are retained under
  `test_artifacts/diagnostics/llm_trace_llmtrace_512da03221a94a2688fe8c87dd3a1a4c_20260831T034857Z.json`
  and
  `test_artifacts/diagnostics/llmtrace_512da03221a94a2688fe8c87dd3a1a4c_bindings.json`.
- **Runtime assignment:** the primary Codex session is the dynamic
  `implementation_owner`; it has the required repository context, filesystem,
  process, pytest, Mongo, live-LLM, and skill access. No implementation
  handoff has occurred.

## Stopped Work And Current Verification State

- **Production sign-off:** withheld. No claim that Phase 3 DSH integration is
  production-ready survives this handoff.
- **Current live evidence:** 2 of 12 cases passed, 1 of 12 failed, and 9 of 12
  were not run under fingerprint
  `sha256:f91179c93a49ab5dc356e5711b357f390bbd51423a6840118c00d6cc782f2325`.
  These artifacts are historical diagnostic evidence only and become stale
  for sign-off after any source change.
- **Historical pass artifact:**
  `test_artifacts/dsh_trigger_source_e2e/sidecar_loss_user_message_20260831T075817Z_a3e58eca`.
- **Historical pass artifact:**
  `test_artifacts/dsh_trigger_source_e2e/user_message_local_fact_20260831T080138Z_8c19a161`.
- **Historical failed artifact:**
  `test_artifacts/dsh_trigger_source_e2e/user_message_background_summary_20260831T080357Z_daa8aac1`.
  The source trace failed, the runtime/pipeline gate failed, and public chat did
  not complete.
- **Failure mechanism:** a background DSH task had a pending required-evidence
  dependency tied to its original pending observation. Repeated cognition
  requested the same task. Duplicate closure changed the dependency state to
  `blocked` while retaining the original pending `observation_id`. L3 correctly
  detected the contradictory dependency/observation pair, then raised
  `ValueError: required resolver evidence state does not match observation`,
  which terminated the public path. The later background callback still
  delivered the resolved Rowan evidence. This is a DSH cognition-chain
  error-policy violation: contradictory state reached L3, and a raw
  `ValueError` bypassed the producing owner's bounded recovery and typed
  fail-closed boundary. It is also a missing pre-E2E
  invariant/fault-injection case.
- **Partial unverified edits:**
  `src/kazusa_ai_chatbot/cognition_resolver/loop.py` currently suppresses a
  repeated request when the matching admitted background task remains pending
  and rewrites blocked dependency lineage to the blocker observation.
  `src/kazusa_ai_chatbot/cognition_resolver/state.py` currently adds strict
  dependency-to-observation alignment validation. These edits were applied
  after the failed live case and have not been collected, unit-tested,
  integration-tested, reviewed, or included in a fresh fingerprint. They carry
  no acceptance credit.
- **Worktree:** the repository contains a broad in-progress DSH diff. Preserve
  it as the execution baseline for the next owner and reconcile every changed
  source path with the failure ledger and exact deterministic tests before
  accepting any implementation work.
- **Execution stop:** no further implementation, deterministic test, process
  test, live test, or sign-off action was performed after recording this
  handoff.

## Execution Handoff — 2026-08-31

- **Plan and lifecycle:** this active bugfix plan remains `in_progress`; work is
  stopped before pre-E2E acceptance and production sign-off is withheld.
- **Released assignment:** the current primary Codex session releases the
  `implementation_owner` assignment after this documentation checkpoint.
- **Next assignment:** unassigned. Runtime executor resolution must preserve the
  existing `implementation_owner` role contract, changed worktree, binding
  handoff rules, and independent review boundary.
- **Remaining scope:** audit the stopped diff, build the complete failure-point
  ledger, add exact deterministic fault-injection coverage for every row,
  remediate all failures below E2E, pass all non-live gates, obtain independent
  pre-E2E review, create a fresh fingerprint, and execute final E2E solely as
  production sign-off.
- **Entry gate:** preserved worktree and this handoff have been read; final E2E
  remains locked.
- **Acceptance output:** a complete zero-uncovered-path failure ledger, exact
  deterministic evidence that every failure follows the existing
  cognition-chain error policy, clean process and non-live evidence, twelve
  inspected final E2E dossiers on one fresh fingerprint, and independent
  production sign-off.
- **Next checkpoint:** independent approval of the completed pre-E2E ledger and
  deterministic evidence. This is an engineering review checkpoint inside the
  approved scope, not an additional user approval request.
