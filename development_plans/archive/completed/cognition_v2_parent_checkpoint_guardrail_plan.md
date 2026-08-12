# Cognition V2 Parent-Checkpoint Guardrail Plan

## Summary

- Goal: Give the live persona cognition path one bounded second chance after
  an escaped goal-bid exhaustion by replaying the non-committing cognition
  child from an immutable canonical-input checkpoint.
- Status: completed.
- Plan class: high-risk runtime reliability and retry-boundary change.
- Cutover: bigbang for the guarded production contract; direct cognition-owner
  contracts remain unchanged and outside the guardrail.
- Scope boundary: the live persona service invocation, canonical cognition
  input connector, shared retry coordinator, parent-recovery ledger scope,
  protected outer guardrail capsule, documentation, and exact tests below.
- Acceptance state: the plan review passed, the user authorized
  implementation, GPT-5.6 Luna completed the scoped implementation, parent
  remediation closed both independent review findings, the full deterministic
  suite is green, and final GPT-5.6-sol code sign-off returned `PASS`. Plan
  closure is complete.
- Production authorization: the current user request authorizes the scoped
  production-code implementation.

## Evidence And Independent Review

The read-only MongoDB export used the repository's `database-data-pull`
workflow. The bounded artifacts are:

- [run_cognition failures, 131 rows](../../../test_artifacts/diagnostics/cognition_v2_failures_recent_24h_bounded1000.json)
- [all V2 entrypoints, 158 rows](../../../test_artifacts/diagnostics/cognition_v2_failures_all_entrypoints_recent_24h_bounded1000.json)
- [initial 100-row export](../../../test_artifacts/diagnostics/cognition_v2_failures_recent_24h.json)

The expanded `run_cognition` export covered 2026-08-11 06:48 UTC through
2026-08-12 05:46 UTC and contained 131 partial-failure capsules:

- 126 semantic-appraisal-stage failures and 3 appraisal-reduction failures;
- 4 exhausted goal-branch records across 3 captures;
- 355 bounded model-attempt records: 197 accepted, 68 recovered, 86
  regenerated, and 4 exhausted;
- every exhaustion-containing capture had a complete sibling outcome, so the
  current facade's sibling-recovery policy handled those cases;
- no terminal `run_cognition` capsule outcome in this bounded export.

The evidence supports a narrowly scoped parent replay. It does not support
retrying every current `retryable=True` failure or changing the goal owner's
three-call contract.

The independent review was requested from GPT-5.6-sol with high reasoning.
The runtime did not expose a standard service tier for that model, so its
supported default tier was used. Reviewer reference:
The first reviewer reference was
`019ff4e1-4da8-7451-aa5d-fdb1e511aaea`; its verdict was `BLOCKED` before
amendment. The amended draft was independently re-reviewed by
`019ff4f7-0742-7372-a555-f8e70438153e`; final verdict: `PASS`.
The review identified checkpoint placement, retry-token stacking, epoch
lifetime, capsule lineage, eligibility breadth, snapshot versioning,
production-owner scope, documentation coverage, and test-matrix completeness
as required corrections. This plan records the resolved decisions below.

## Confirmed Decisions

1. The parent checkpoint is the canonical `CognitionCoreInputV2` created by
   `call_cognition_subgraph` after identity resolution, mutable-state reads,
   shared-memory prewarm, and input construction complete.
2. The guardrail retries only the `run_cognition` child. It does not replay
   connector preparation, resolver capability execution, pending-resume
   persistence, actions, surfaces, delivery, or state commit.
3. The service graph and parent guardrail share one context-local,
   invocation-wide replay token. The first owner to claim it prevents the
   other owner from retrying.
4. Exactly two cognition execution epochs exist for a parent-recovery path:
   epoch `0` initial and epoch `1` parent recovery. Epoch `1` persists for all
   later cognition calls in that invocation and never resets per resolver
   cycle.
5. The parent guardrail applies only to the queued live persona path through
   `stage_1_goal_resolver`. Idle self-cognition through
   `self_cognition.runner._default_cognition_client` remains outside this
   change and retains its current direct-loop behavior.
6. Existing branch-local sibling recovery runs before the parent guardrail.
   A sibling-recovered exhaustion never claims the replay token.
7. Parent-recovery dispositions are orchestration metadata, not
   `V2AttemptDisposition` values. Existing model-attempt dispositions remain
   unchanged.
8. Unguarded attempt-ledger snapshots remain exact `v1`. A guarded invocation
   writes a separate epoch-aware `v2` aggregate in the outer guardrail capsule.

## Target State And Mechanism

### Ownership boundary

```text
service invocation
  -> bind one CognitionRetryCoordinator
  -> persona supervisor stage 1
       -> resolver loop selects the next cognition cycle
            -> call_cognition_subgraph prepares identity/state/prewarm once
            -> build canonical CognitionCoreInputV2
            -> immutable checkpoint + digest
            -> guarded run_cognition(epoch 0)
                 -> existing V2 DAG, goal owner, sibling recovery, collapse
            -> optional guarded run_cognition(epoch 1)
                 -> same canonical input, independent copy, same services
       -> capability execution between successful cognition cycles
       -> one final state commit
```

`cognition_core_v2.goal_cognition`, the dependency graph, and the facade keep
their existing semantic ownership. The new deterministic coordinator owns
only replay-token arbitration, epoch binding, checkpoint identity, and
bounded orchestration metadata.

### Canonical checkpoint

`call_cognition_subgraph` creates the checkpoint immediately after
`build_cognition_input_from_global_state(...)` returns and before the first
`run_cognition(...)` call. It retains:

- one deep-copied canonical `CognitionCoreInputV2`;
- one stable input SHA-256 digest;
- the resolver cycle index;
- the shared `CognitionCoreServicesV2` object used by both child calls; and
- the current protected trace/correlation references.

The guardrail calls the child with independent deep copies of the canonical
input. A child mutation cannot affect the replay. The checkpoint digest is
stored in bounded protected metadata; the canonical input itself, prompts,
raw model output, and state payload are never stored in the outer guardrail
capsule.

Identity resolution, mutable-state reads, shared-memory prewarm, and canonical
input construction execute once. Only the following may repeat:

- cognition model and JSON-repair calls;
- protected model-attempt tracing; and
- inner/outer failure-capsule persistence.

State reads, prewarm, resolver capabilities, pending-resume writes, action
execution, surface generation, delivery, and state commit execute once or
remain outside the replayed child.

The connector accepts the guardrail coordinator only for its non-committing
path. `commit=True` cannot enter the guarded child wrapper. The resolver loop
remains a generic recurrence controller; it receives no retry policy
parameter. `stage_1_goal_resolver` passes the context-bound coordinator to its
`commit=False` cognition-cycle closure. Idle self-cognition omits it.

### Shared replay coordinator

`cognition_resolver.guardrail.CognitionRetryCoordinator` is bound around the
entire queued service graph invocation through a `ContextVar`. Its fixed
contract is:

- owner: `none`, `service_graph`, or `parent_checkpoint`;
- parent epoch: `0` before a parent claim and `1` after a parent claim;
- maximum replay claims: exactly one;
- parent disposition: `not_attempted`, `blocked_by_service_retry`,
  `recovered`, or `exhausted`; and
- bounded trigger metadata: error code, stage, branch, cycle, and digest.

`claim_replay("service_graph")` runs at the existing service retry decision.
`claim_replay("parent_checkpoint")` runs before the second guarded child call.
Claiming is atomic within the context-local coordinator. A coordinator is
created once per service invocation and remains bound across all service graph
attempts. It is never recreated for graph attempt two.

Consequences:

- Parent recovery first: the service cannot start a whole-graph retry after
  the parent has claimed the token, including after a later post-cognition
  service failure.
- Service retry first: graph attempt two shares the original service ledger;
  the parent guardrail sees the service owner and cannot claim a parent
  replay.
- Direct connector, resolver-loop, facade, and goal-owner calls without a
  bound coordinator retain their current behavior.

### Eligibility

The parent guardrail claims its token only when all conditions hold:

- the child raised `CognitionExecutionError`;
- `safe_checkpoint == "pre_state_commit"`;
- `error_code` is exactly `goal_bid_structure_exhausted` or
  `goal_bid_provider_exhausted`;
- the error escaped the existing complete-sibling recovery; and
- the invocation-wide coordinator token is available.

Existing generic `retryable=True` failures remain service-owned. Malformed
input, invalid persisted state, cancellation, post-commit failures, provider
errors outside the two goal exhaustion codes, surface failures, and unknown
exceptions fail closed without parent replay.

### Epoch and attempt budgets

The existing graph-scope attempt policy remains unchanged. When the parent
token is claimed, the attempt ledger creates exactly one additional epoch:

| Owner family | Existing per-epoch cap | Parent-path maximum across epochs 0 and 1 | Extra normal-path calls |
|---|---:|---:|---:|
| goal producers | 3 per stage/branch | 6 per stage/branch | 0 |
| semantic appraisal | 2 per question | 4 per question | 0 |
| all 3-call V2 owners | 3 per stage/branch | 6 per stage/branch | 0 |
| all 3-call verifier owners | 3 per stage/branch | 6 per stage/branch | 0 |

The parent replay adds at most one complete cognition-child execution, using
the existing stage timeout and service model routes. It does not add a new
model or a new semantic prompt. Worst-case guarded latency is one additional
bounded cognition execution; the acceptance tests must record this call
arithmetic and the production timeout behavior. A normal successful turn
keeps its current call count.

Epoch `1` remains active for later resolver cycles after a successful parent
replay. It supplies local owner attempts within its configured cap but cannot
claim another parent replay. The total parent-path budget therefore remains
bounded across the entire service invocation.

### Terminal error contract

`cognition_resolver.guardrail.ParentRecoveryExhaustedError` subclasses
`CognitionExecutionError` and retains:

- the second child's `error_code`, `stage`, `branch_id`, and final owner
  `attempt_count`;
- `safe_checkpoint == "pre_state_commit"`;
- `retryable == False`;
- `parent_recovery_attempted == True`;
- `parent_recovery_disposition == "exhausted"`;
- bounded first-error code/stage/branch metadata; and
- the parent checkpoint digest and recovery epoch.

The exception chains from the second child error. Public `attempt_count`
remains the final failing owner's count, normally three; protected epoch
metadata carries the total six-call accounting. The service recognizes the
subclass/metadata as already guarded and produces one operational failure
without another graph retry.

### Protected outer guardrail capsule

Create `llm_tracing.guardrail_capsule` as a separate protected writer. The
outer capsule is created for an opted-in guarded cognition execution and is
persisted only when a parent failure is observed. Its exact bounded shape is:

```json
{
  "schema_version": "cognition_parent_guardrail_capsule.v1",
  "trace_id": "opaque-trace-ref",
  "guardrail_invocation_id": "opaque-invocation-ref",
  "scope": "persona_stage_1",
  "cycle_index": 0,
  "checkpoint_sha256": "bounded-digest",
  "trigger": {
    "error_code": "goal_bid_structure_exhausted",
    "stage": "goal_cognition",
    "branch_id": "ordinary_response",
    "attempt_count": 3
  },
  "parent_recovery": {
    "disposition": "recovered",
    "claimed_by": "parent_checkpoint",
    "epoch": 1,
    "max_replays": 1
  },
  "attempt_ledger": {
    "schema_version": "cognition_attempt_ledger.v2",
    "epochs": [],
    "parent_recovery": {}
  }
}
```

The implementation fills only bounded coordinates and dispositions. It emits
no checkpoint state, prompts, raw responses, model payloads, user content, or
credentials. Exact raw attempts remain in their owning inner failure capsules
and traces. The existing `failure_capsule.py` writer and unguarded
`cognition_attempt_ledger.v1` contract remain unchanged.

## Scope And Change Direction

### Must change

- Add `CognitionRetryCoordinator`, parent eligibility, checkpoint digest,
  epoch binding, and `ParentRecoveryExhaustedError` in
  `cognition_resolver/guardrail.py`.
- Add the parent-recovery epoch and separate orchestration summary to
  `cognition_core_v2/model_attempt_policy.py` while preserving the exact
  unguarded V1 snapshot and existing graph-attempt monotonicity.
- Add the optional coordinator argument to
  `nodes/persona_supervisor2_cognition.call_cognition_subgraph` and place the
  guarded `run_cognition` wrapper after canonical input construction.
- Bind one coordinator around the queued service graph and make the existing
  service retry claim the same replay token.
- Pass the context-bound coordinator from the live persona stage's
  non-committing cognition closure. Keep the resolver loop generic.
- Add `llm_tracing/guardrail_capsule.py` and document its protected schema.
- Update the resolver, cognition-core, brain-service, node, and LLM-tracing
  READMEs with the new ownership and retry boundaries.
- Add exact deterministic tests for both stacking orders, canonical-input
  reuse, epoch persistence, cancellation, concurrency, capsule lineage,
  side-effect exclusion, and direct failure preservation.
- Update the source-to-test impact manifest with one exact row per changed
  production source and governed README. Add the exact source-root and
  entry declarations for `src/kazusa_ai_chatbot/cognition_resolver/guardrail.py`,
  `src/kazusa_ai_chatbot/llm_tracing`,
  `src/kazusa_ai_chatbot/llm_tracing/guardrail_capsule.py`,
  `src/kazusa_ai_chatbot/service.py`, and
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`; map every entry to an
  exact deterministic owner test.

### Must remain unchanged

- `V2_MODEL_TOTAL_ATTEMPTS == 3` within each owner execution epoch.
- Strict goal-bid schema, evidence/role-handle authority, producer-owned
  repair, unsupported-handle rejection, and no deterministic bid synthesis.
- DAG dependency ordering, branch isolation, complete-bid sibling recovery,
  collapse validation, and fail-closed direct facade behavior.
- The generic resolver loop signature and its direct one-call behavior.
- Direct `run_goal_cognition`, direct `run_cognition`, and direct connector
  calls without an explicit coordinator.
- The idle self-cognition client and its current recurrence behavior.
- One final cognition-state commit after the live resolver returns.
- The existing service graph retry for generic typed pre-commit failures when
  it claims the shared token first.

### Deferred

- Retrying inside `parallel_executor` or retrying only the failed branch.
- Parent replay for generic `retryable=True` failures.
- More than one global replay claim or more than two cognition epochs.
- Replaying state reads, memory prewarm, capabilities, pending persistence,
  actions, surfaces, delivery, or commit.
- Persisted cross-request checkpoints, adaptive retry counts, prompt rewriting,
  fallback bid synthesis, schema relaxation, and new model routes.
- Extending the guardrail to idle self-cognition without a new approved scope.

## Mandatory Skills And Repository Rules

- `development-plan` for lifecycle, ownership, exact traceability, review,
  and execution gates.
- `local-llm-architecture` for bounded retry design, semantic/deterministic
  ownership, latency, and weak-model blast-radius review.
- `debug-llm` for protected failure evidence and human-readable replay review.
- `test-style-and-execution` for deterministic, patched, captured, live, and
  database test selection.
- `py-style` for every Python source change.
- `cjk-safety` for Python source changes containing CJK text.
- `database-data-pull` for any later read-only evidence refresh.

Execution rules:

- Use `venv\Scripts\python.exe`.
- Preserve the unrelated existing changes in
  `development_plans/README.md` and
  `development_plans/active/bugfix/group_topic_continuity_authority_fix_plan.md`.
- Keep `.env` unread.
- Use `apply_patch` for manual edits.
- Keep this plan in `draft` until independent re-review and explicit approval.
- Use big-bang contract updates; do not create compatibility aliases or
  parallel retry vocabularies.

## Execution Stages

### Stage 0: baseline and ownership freeze

Capture `git status --short`, the explicitly owned file set, the current
source-to-test manifest, exact pytest collection for every named node, and the
current README contracts. Record the pre-existing three plan/registry paths
outside the implementation diff.

Gate: the coordinator owner is the live service invocation; the connector is
the checkpoint owner; the resolver loop and idle self-cognition are outside
the retry implementation.

### Stage 1: coordinator, exception, and epoch ledger

Implement the context-local coordinator and its single atomic replay claim.
Implement the typed terminal error and the two-epoch attempt-ledger sidecar.
Keep `V2AttemptDisposition` closed and add a separate
`ParentRecoveryDisposition` contract. Preserve V1 snapshots when the parent
path is unused and emit V2 only for guarded outer-capsule aggregation.

Gate: unit tests prove token ownership, exact two-epoch lifetime, concurrent
isolation, cancellation cleanup, bounded arithmetic, and unchanged existing
graph-attempt coordinates.

### Stage 2: connector checkpoint and guarded child

Modify `call_cognition_subgraph` so all preparation completes once, then build
one canonical checkpoint and run the guarded child from independent copies.
The wrapper is entered only for `commit=False` with a coordinator. It records
the trigger, claims the parent token, switches to epoch 1, retries once, and
returns only the successful child output. It raises
`ParentRecoveryExhaustedError` after a second eligible failure.

Gate: patched connector tests prove the same canonical-input digest, one
identity/state/prewarm preparation, independent copies, no repeated
capability or commit, and correct success/failure output.

### Stage 3: service arbitration and live persona wiring

Bind one coordinator around the complete queued service graph retry loop. Make
the existing service graph retry claim `service_graph` before continuing.
Pass the bound coordinator through `stage_1_goal_resolver` into its
non-committing connector closure. Keep `_default_cognition_client` outside
the coordinator.

Gate: tests cover parent-first then service failure, service-first then goal
exhaustion, parent recovery followed by a later service failure, and existing
generic service retry behavior.

### Stage 4: protected capsule and documentation

Implement the outer guardrail capsule with the exact bounded V1 shape above.
Document the canonical checkpoint owner, shared replay token, two epoch
ledger, idle self-cognition exclusion, and inner/outer capsule lineage in:

- `src/kazusa_ai_chatbot/cognition_resolver/README.md`;
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`;
- `src/kazusa_ai_chatbot/brain_service/README.md`;
- `src/kazusa_ai_chatbot/nodes/README.md`; and
- `src/kazusa_ai_chatbot/llm_tracing/README.md`.

Gate: capsule tests prove protected bounded metadata, separate inner raw
attempt ownership, recovery success capture, failure capture, and clean
unguarded V1 snapshots.

### Stage 5: verification and independent sign-off

Run exact deterministic owner nodes and patched integration nodes in batches.
Run each selected captured/live node individually where listed below, inspect
its artifact, and record whether it used the direct owner boundary or live
persona guardrail. Run the source-to-test impact validator after the manifest
update. Route the amended plan and later implementation through an independent
reviewer; the remediation owner does not sign off its own corrections.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/cognition_resolver/guardrail.py`: coordinator,
  eligibility, checkpoint digest, epoch state, and terminal error.
- `src/kazusa_ai_chatbot/llm_tracing/guardrail_capsule.py`: protected outer
  capsule writer and bounded schema validation.
- `tests/unit/cognition_resolver/test_guardrail.py`: deterministic guardrail
  owner tests.
- `tests/unit/llm_tracing/test_guardrail_capsule.py`: capsule owner tests.
- `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py`:
  patched connector/service propagation tests.
- `tests/test_cognition_v2_parent_guardrail_docs.py`: exact README contract
  tests for every changed governed document.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/model_attempt_policy.py`: add
  epoch-aware sidecar and preserve V1 unguarded snapshots.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: place the
  guard after canonical input preparation and keep commit outside it.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: pass the bound
  coordinator from the live stage.
- `src/kazusa_ai_chatbot/service.py`: bind the coordinator and arbitrate the
  existing graph retry through its token.
- `src/kazusa_ai_chatbot/cognition_resolver/README.md`.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`.
- `src/kazusa_ai_chatbot/brain_service/README.md`.
- `src/kazusa_ai_chatbot/nodes/README.md`.
- `src/kazusa_ai_chatbot/llm_tracing/README.md`.
- `tests/test_cognition_core_v2_attempt_ledger.py`.
- `tests/test_persona_supervisor2.py`.
- `tests/test_persona_supervisor2_cognition_prewarm.py`.
- `tests/test_service_input_queue.py`.
- `tests/test_llm_tracing.py`.
- `tests/ownership/source_test_impact_manifest.json`.
- `development_plans/README.md` to keep this draft registered.

### Delete

- None.

### Keep

- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`.
- `src/kazusa_ai_chatbot/cognition_core_v2/dependency_graph.py`.
- `src/kazusa_ai_chatbot/cognition_core_v2/parallel_executor.py`.
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py` production behavior and
  signature.
- `src/kazusa_ai_chatbot/self_cognition/runner.py` production behavior.
- `src/kazusa_ai_chatbot/llm_tracing/failure_capsule.py` production contract.
- `tests/fixtures/cognition_core_v2_retry_exhaustion_cases.json`; it describes
  model-owner exhaustion and has no guarded-retry consumer.
- All direct goal-producer and direct-facade failure semantics.

## Test Impact And Traceability

### Intentional failure tests preserved outside the guardrail

These deterministic tests call a direct goal owner, direct facade, direct DAG,
direct ledger, direct capsule, or synthetic service graph. Their current
failure assertions and call counts remain fixed:

- `tests/test_cognition_core_v2_integration.py::test_required_selection_without_admitted_bid_fails_closed_before_action`;
- `tests/test_cognition_core_v2_integration.py::test_goal_structure_recovers_on_third_attempt`;
- `tests/test_cognition_core_v2_integration.py::test_required_goal_invalid_evidence_stops_before_action_planning`;
- `tests/test_cognition_core_v2_integration.py::test_required_goal_exhaustion_is_nonretryable`;
- `tests/test_cognition_core_v2_dependencies.py::test_goal_bid_schema_exhaustion_is_typed_after_three_attempts`;
- `tests/test_cognition_core_v2_dependencies.py::test_required_selection_structure_exhaustion_is_typed`;
- `tests/test_cognition_core_v2_dependencies.py::test_required_selection_invalid_evidence_fails_after_exhaustion`;
- `tests/test_cognition_core_v2_dependencies.py::test_required_branch_failure_cannot_collapse_to_silence`;
- `tests/test_cognition_core_v2_dependencies.py::test_required_branch_failure_preserves_a_complete_sibling_bid`;
- `tests/test_cognition_core_v2_dependencies.py::test_required_branch_failure_rejects_an_incomplete_sibling_bid`;
- `tests/test_cognition_core_v2_relational_willingness.py::test_ordinary_goal_exhaustion_fails_closed_before_commit`;
- `tests/test_cognition_core_v2_failures.py::test_run_cognition_capsule_preserves_original_exception`;
- `tests/test_cognition_core_v2_model_retry_continuity.py::test_v2_attempt_policy_matches_exact_owner_matrix`;
- `tests/test_cognition_core_v2_model_retry_continuity.py::test_v2_attempt_record_validation_is_bounded_and_data_only`;
- `tests/test_cognition_core_v2_prompt_budget_continuity.py::test_required_selection_producer_overflow_fails_before_call`;
- `tests/test_cognition_core_v2_quoted_message_reproduction.py::test_rebuilt_case_reproduces_goal_branch_exhaustion`;
- `tests/test_cognition_core_v2_quoted_message_reproduction.py::test_rebuilt_goal_branch_current_prompt_live_llm`;
- `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_goal_evidence_handles_not_permitted_is_rejected`;
- `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_goal_bid_fields_not_exact_is_rejected`;
- `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_goal_bid_consequences_invalid_is_rejected`;
- `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_goal_role_handles_not_permitted_is_rejected`;
- `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_relational_willingness_fields_not_exact_is_rejected`;
- `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_relational_willingness_evidence_unavailable_is_rejected`;
- `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_relational_willingness_episode_evidence_missing_is_rejected`;
- `tests/test_cognition_core_v2_attempt_ledger.py::test_goal_attempt_budget_is_monotonic_across_graph_attempts`;
- `tests/test_cognition_core_v2_attempt_ledger.py::test_goal_attempt_budgets_are_independent_by_branch`;
- `tests/test_cognition_core_v2_attempt_ledger.py::test_attempt_ledger_context_isolated_between_concurrent_calls`;
- `tests/test_service_input_queue.py::test_goal_cognition_exhaustion_skips_service_retry`.

The unchanged direct owner contracts remain collected and verified:

- `tests/unit/cognition_resolver/test_loop.py::test_loop_exposes_owned_contract`;
- `tests/unit/cognition_core_v2/test_model_attempt_policy.py::test_model_attempt_policy_exposes_owned_contract`;
- `tests/unit/nodes/test_persona_supervisor2_cognition.py::test_persona_supervisor2_cognition_exposes_owned_contract`.

The captured/direct-owner suites remain outside the guardrail:

- `tests/test_cognition_core_v2_captured_goal_failure_live_llm.py::test_captured_goal_replay_sample_1_rejects`;
- `tests/test_cognition_core_v2_captured_goal_failure_live_llm.py::test_captured_goal_replay_sample_2_rejects`;
- `tests/test_cognition_core_v2_captured_goal_failure_live_llm.py::test_captured_goal_replay_sample_3_rejects`;
- `tests/test_cognition_core_v2_captured_run_failures_live_llm.py::test_captured_run_goal_relational_willingness_repair_live_llm`;
- `tests/test_cognition_core_v2_self_improvement_schema_live_llm.py::test_plan_trace_0a04_self_improvement_schema_live_llm`;
- `tests/test_cognition_core_v2_self_improvement_schema_live_llm.py::test_plan_trace_a1a573_self_improvement_schema_live_llm`;
- `tests/test_cognition_core_v2_self_improvement_schema_live_llm.py::test_postdraft_d1138_autonomy_boundary_schema_live_llm`.

The direct required-selection live owner nodes remain explicitly named and
continue to bind their own V2 ledger:

- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_with_empty_progress_domain`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_ignores_optional_conversation_row`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_ignores_internal_evidence_row`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_accepts_one_progress_event`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_separates_progress_and_optional_rows`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_autonomy_selection_separates_progress_and_optional_rows`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_autonomy_selection_with_multiple_required_operations`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_autonomy_selection_with_empty_progress_domain`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_autonomy_selection_with_maximum_evidence_rows`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_autonomy_selection_with_progress_alias_collisions`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_autonomy_selection_with_exact_production_scene`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_autonomy_selection_with_multiple_progress_events`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_relationship_selection_with_production_state`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_autonomy_selection_with_compound_evidence_pressure`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_autonomy_selection_with_ten_visible_evidence_rows`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_autonomy_food_choice_with_stale_goal_pressure`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_parallel_food_choice_selection_contract_pressure`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_third_party_reply_ordinary_selection_contract_pressure`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_third_party_reply_autonomy_selection_contract_pressure`;
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_parallel_third_party_reply_selection_contract_pressure`.

The existing service tests retain their meaning:

- `tests/test_service_input_queue.py::test_precommit_cognition_failure_retries_once_then_succeeds`;
- `tests/test_service_input_queue.py::test_service_graph_retry_reuses_goal_attempt_ledger`;
- `tests/test_service_input_queue.py::test_goal_cognition_exhaustion_skips_service_retry`.

### New deterministic guarded-path nodes

The implementation must add these exact nodes:

- `tests/unit/cognition_resolver/test_guardrail.py::test_guardrail_exposes_owned_contract`;
- `tests/unit/cognition_resolver/test_guardrail.py::test_coordinator_claim_is_atomic_between_service_and_parent`;
- `tests/unit/cognition_resolver/test_guardrail.py::test_parent_epoch_persists_after_recovery`;
- `tests/unit/cognition_resolver/test_guardrail.py::test_parent_recovery_allows_only_goal_exhaustion_codes`;
- `tests/unit/cognition_resolver/test_guardrail.py::test_parent_recovery_rejects_postcommit_and_unknown_failures`;
- `tests/unit/cognition_resolver/test_guardrail.py::test_parent_recovery_isolated_between_concurrent_contexts`;
- `tests/unit/cognition_resolver/test_guardrail.py::test_parent_recovery_cancellation_restores_context`;
- `tests/unit/llm_tracing/test_guardrail_capsule.py::test_guardrail_capsule_exposes_owned_contract`;
- `tests/unit/llm_tracing/test_guardrail_capsule.py::test_guardrail_capsule_contains_only_bounded_metadata`;
- `tests/unit/llm_tracing/test_guardrail_capsule.py::test_guardrail_capsule_projects_adversarial_metadata`;
- `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_retry_reuses_canonical_input_digest`;
- `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_retry_runs_preparation_once`;
- `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_retry_uses_independent_input_copies`;
- `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_retry_does_not_repeat_capability_or_commit`;
- `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_recovery_failure_preserves_typed_error_and_no_side_effect`;
- `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_epoch_remains_active_on_later_resolver_cycle`;
- `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_first_blocks_later_service_retry`;
- `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_service_first_blocks_parent_recovery`;
- `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_recovery_success_then_service_failure_has_one_replay`;
- `tests/test_service_input_queue.py::test_parent_guardrail_failure_does_not_start_outer_graph_retry`;
- `tests/test_persona_supervisor2_cognition_prewarm.py::test_parent_retry_does_not_repeat_shared_memory_prewarm`;
- `tests/test_persona_supervisor2_cognition_prewarm.py::test_omitted_coordinator_does_not_use_ambient_guardrail`;
- `tests/test_persona_supervisor2.py::test_persona_stage_passes_bound_parent_coordinator`;
- `tests/test_llm_tracing.py::test_guarded_failure_capsule_contains_outer_parent_metadata`;
- `tests/test_cognition_v2_parent_guardrail_docs.py::test_resolver_readme_documents_canonical_checkpoint_owner`;
- `tests/test_cognition_v2_parent_guardrail_docs.py::test_core_readme_documents_two_epoch_owner_budget`;
- `tests/test_cognition_v2_parent_guardrail_docs.py::test_brain_service_readme_documents_shared_replay_token`;
- `tests/test_cognition_v2_parent_guardrail_docs.py::test_nodes_readme_documents_connector_guard_owner`;
- `tests/test_cognition_v2_parent_guardrail_docs.py::test_llm_tracing_readme_documents_outer_capsule_lineage`.

### Source-to-test matrix

Each row names one exact source or governed artifact path and exact pytest
nodes. No row relies on a directory-only phrase or a grouped brace path.

| Source or governed artifact | Changed symbol/contract | Semantic owner | Exact deterministic pytest nodes | Test mode | Supplemental/live node IDs | Prevented regression |
|---|---|---|---|---|---|---|
| `src/kazusa_ai_chatbot/cognition_resolver/guardrail.py` | coordinator, eligibility, checkpoint digest, epoch state, terminal error | parent guardrail owner | `tests/unit/cognition_resolver/test_guardrail.py::test_guardrail_exposes_owned_contract`; `tests/unit/cognition_resolver/test_guardrail.py::test_coordinator_claim_is_atomic_between_service_and_parent`; `tests/unit/cognition_resolver/test_guardrail.py::test_parent_epoch_persists_after_recovery`; `tests/unit/cognition_resolver/test_guardrail.py::test_parent_recovery_allows_only_goal_exhaustion_codes`; `tests/unit/cognition_resolver/test_guardrail.py::test_parent_recovery_rejects_postcommit_and_unknown_failures`; `tests/unit/cognition_resolver/test_guardrail.py::test_parent_recovery_isolated_between_concurrent_contexts`; `tests/unit/cognition_resolver/test_guardrail.py::test_parent_recovery_cancellation_restores_context` | deterministic unit; patched integration | `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_recovery_failure_preserves_typed_error_and_no_side_effect` | no broad eligibility, no recursive epoch, no context leak |
| `src/kazusa_ai_chatbot/cognition_core_v2/model_attempt_policy.py` | two-epoch sidecar and V2 guarded aggregate; V1 unchanged when unused | attempt-budget owner | `tests/unit/cognition_core_v2/test_model_attempt_policy.py::test_model_attempt_policy_exposes_owned_contract`; `tests/test_cognition_core_v2_attempt_ledger.py::test_attempt_ledger_rejects_invalid_graph_attempts`; `tests/test_cognition_core_v2_attempt_ledger.py::test_goal_attempt_budget_is_monotonic_across_graph_attempts`; `tests/test_cognition_core_v2_attempt_ledger.py::test_goal_attempt_budgets_are_independent_by_branch`; `tests/test_cognition_core_v2_attempt_ledger.py::test_attempt_ledger_snapshot_records_terminal_branch_dispositions`; `tests/test_cognition_core_v2_attempt_ledger.py::test_attempt_ledger_context_isolated_between_concurrent_calls`; `tests/test_cognition_core_v2_model_retry_continuity.py::test_v2_attempt_policy_matches_exact_owner_matrix`; `tests/test_cognition_core_v2_model_retry_continuity.py::test_v2_attempt_record_validation_is_bounded_and_data_only` | deterministic ledger/unit; patched integration | `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_epoch_remains_active_on_later_resolver_cycle` | no budget reset, overwrite, or cross-call leakage |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | connector prepares once and guards only non-committing canonical child | connector owner | `tests/unit/nodes/test_persona_supervisor2_cognition.py::test_persona_supervisor2_cognition_exposes_owned_contract`; `tests/test_cognition_chain_connector_mapping.py::test_connector_projects_protocol_owned_resolver_goal_progress`; `tests/test_persona_supervisor2_cognition_prewarm.py::test_cycle_zero_prewarm_reaches_v2_memory_evidence`; `tests/test_persona_supervisor2_cognition_prewarm.py::test_parent_retry_does_not_repeat_shared_memory_prewarm`; `tests/test_persona_supervisor2_cognition_prewarm.py::test_omitted_coordinator_does_not_use_ambient_guardrail` | deterministic connector/prewarm; patched integration | `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_retry_reuses_canonical_input_digest`; `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_retry_runs_preparation_once`; `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_retry_uses_independent_input_copies` | no repeated identity/state/prewarm reads or leaked child mutation |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | live stage passes the context-bound coordinator and commits once | persona-stage owner | `tests/test_cognition_resolver_persona_graph.py::test_persona_graph_has_one_v2_resolver_path`; `tests/test_persona_supervisor2.py::test_persona_stage_uses_canonical_resolver_loop`; `tests/test_persona_supervisor2.py::test_persona_stage_passes_bound_parent_coordinator` | deterministic graph/stage; patched integration | `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_retry_does_not_repeat_capability_or_commit` | idle/direct callers do not gain the guardrail |
| `src/kazusa_ai_chatbot/service.py` | one invocation-wide token arbitrates service and parent retries | service retry owner | `tests/test_service_input_queue.py::test_precommit_cognition_failure_retries_once_then_succeeds`; `tests/test_service_input_queue.py::test_service_graph_retry_reuses_goal_attempt_ledger`; `tests/test_service_input_queue.py::test_goal_cognition_exhaustion_skips_service_retry`; `tests/test_service_input_queue.py::test_parent_guardrail_failure_does_not_start_outer_graph_retry` | deterministic service; patched integration | `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_first_blocks_later_service_retry`; `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_service_first_blocks_parent_recovery`; `tests/integration/cognition_core_v2/test_parent_checkpoint_guardrail.py::test_parent_recovery_success_then_service_failure_has_one_replay` | no stacked whole-graph and parent retries |
| `src/kazusa_ai_chatbot/llm_tracing/guardrail_capsule.py` | protected outer capsule V1 and bounded metadata | guardrail tracing owner | `tests/unit/llm_tracing/test_guardrail_capsule.py::test_guardrail_capsule_exposes_owned_contract`; `tests/unit/llm_tracing/test_guardrail_capsule.py::test_guardrail_capsule_contains_only_bounded_metadata`; `tests/unit/llm_tracing/test_guardrail_capsule.py::test_guardrail_capsule_projects_adversarial_metadata`; `tests/test_llm_tracing.py::test_guarded_failure_capsule_contains_outer_parent_metadata` | deterministic capsule/trace | `none` | inner/outer lineage is inspectable without raw leakage |
| `src/kazusa_ai_chatbot/cognition_resolver/README.md` | live owner and canonical checkpoint documentation | resolver documentation owner | `tests/test_cognition_v2_parent_guardrail_docs.py::test_resolver_readme_documents_canonical_checkpoint_owner` | deterministic documentation | `none` | resolver docs do not claim loop-level replay |
| `src/kazusa_ai_chatbot/cognition_core_v2/README.md` | local owner cap and parent epoch documentation | cognition documentation owner | `tests/test_cognition_v2_parent_guardrail_docs.py::test_core_readme_documents_two_epoch_owner_budget` | deterministic documentation | `none` | direct three-call contract remains documented |
| `src/kazusa_ai_chatbot/brain_service/README.md` | service token and outer retry arbitration documentation | service documentation owner | `tests/test_cognition_v2_parent_guardrail_docs.py::test_brain_service_readme_documents_shared_replay_token` | deterministic documentation | `none` | service retry ownership remains explicit |
| `src/kazusa_ai_chatbot/nodes/README.md` | connector checkpoint ownership and commit boundary | nodes documentation owner | `tests/test_cognition_v2_parent_guardrail_docs.py::test_nodes_readme_documents_connector_guard_owner` | deterministic documentation | `none` | preparation and commit are outside replay |
| `src/kazusa_ai_chatbot/llm_tracing/README.md` | outer capsule lineage and protected fields | tracing documentation owner | `tests/test_cognition_v2_parent_guardrail_docs.py::test_llm_tracing_readme_documents_outer_capsule_lineage`; `tests/test_documentation_harmonization.py::test_selected_compact_module_readmes_keep_icd_sections` | deterministic documentation | `none` | protected metadata cannot be confused with raw inner capsule content |
| `tests/ownership/source_test_impact_manifest.json` | exact owner mappings for every new production source | test-impact owner | `tests/test_test_impact_manifest.py::test_unmapped_changed_source_fails_closed`; `tests/test_test_impact_manifest.py::test_stale_required_node_fails_closed` | deterministic manifest validation | `none` | no changed source lacks an exact deterministic node |

## Execution Evidence And Remediation

- Implementation owner: GPT-5.6 Luna, max reasoning, agent
  `019ff508-8446-7392-9105-fd8f3b8bd08d`, with production and governed README
  ownership limited to the approved change surface. The parent retained test,
  manifest, remediation, and plan ownership.
- First independent code review: GPT-5.6-sol, high reasoning, agent
  `019ff524-f87e-76b0-8067-1fc05d54ed25`, verdict `BLOCKED`; no reviewer files
  were changed. Findings covered ambient-coordinator guarding, unbounded outer
  capsule projection, the required prewarm regression node, missing service
  loop arbitration coverage, missing `blocked_by_service_retry` disposition,
  broad exception handling/logging, and stale lifecycle evidence.
- Parent remediation closed those findings by requiring an explicit bound
  coordinator for guarded connector calls, projecting a strict bounded outer
  capsule schema, adding the exact prewarm and ambient-coordinator tests,
  exercising service-first and parent-first arbitration through the service
  loop, recording `blocked_by_service_retry`, narrowing external exception
  handling with exception text, and updating lifecycle evidence.
- Follow-up Sol review findings: the service-first arbitration fixture still
  claimed the token inside the fake graph instead of exercising the real
  `service.py` retry branch, and the capsule accepted oversized or
  epoch-inconsistent numeric attempt coordinates. Parent remediation moved
  service-first setup to a first graph failure that causes the real service
  claim before the second graph call, asserted coordinator identity and exact
  call arithmetic, and bounded graph/local/cumulative/configured coordinates
  with stage-policy limits plus an enclosing-epoch match. The adversarial
  capsule test now covers both invalid numeric forms.
- Verification recorded before final sign-off: the 21-node impact run passed;
  the focused guardrail/prewarm/service-arbitration suite passed 31 tests; the
  service queue regression suite passed 70 tests; broader direct regressions
  passed 86 tests with 2 expected skips; persona/tracing/documentation checks
  passed 49 tests; and four captured/live cognition cases passed individually
  with inspected output. The initial full deterministic run exposed one
  public-DB-boundary violation from the new capsule import; replacing it with
  the public `DatabaseBackendError` alias made the boundary and capsule checks
  pass. The clean rerun passed 4,472 tests, skipped 16 unavailable/opt-in
  cases, deselected 1,183 live cases, and emitted 1 warning in 338.42 seconds;
  the final rerun after the second review remediation passed the same 4,472
  tests with 16 skips and 1,183 deselections in 286.93 seconds. The focused
  post-remediation suite passed 16 tests.
- Follow-up independent implementation review: GPT-5.6-sol, high reasoning,
  agent `019ff524-f87e-76b0-8067-1fc05d54ed25`, returned `PASS` with no
  residual sign-off issues.

## Agent Autonomy Boundaries

- The implementation owner may choose private helper names, local class
  decomposition, command order, and fixture construction inside the approved
  change surface.
- The implementation owner must preserve the exact coordinator ownership,
  canonical checkpoint location, eligibility set, two-epoch budget, capsule
  schema, idle exclusion, and direct-test boundary. Changes require a plan
  amendment or user decision.
- The test owner may add assertions required by the fixed acceptance criteria
  and may add deterministic fixtures without weakening failure assertions.
- The independent reviewer may inspect the amended plan or implementation and
  pass/fail the review gate. The remediation owner cannot sign off its own
  corrections.
- No role may add a fallback bid, prompt rewrite, compatibility alias,
  persisted checkpoint, extra replay token, or idle self-cognition scope.

## Verification And Acceptance Criteria

The implementation is accepted because:

1. The live connector checkpoints one canonical `CognitionCoreInputV2` after
   preparation and retries only the non-committing `run_cognition` child.
2. Identity resolution, mutable-state reads, shared-memory prewarm,
   capabilities, pending persistence, actions, surfaces, delivery, and commit
   are not repeated by the parent replay.
3. Exactly one context-local replay token arbitrates service and parent retry,
   regardless of which owner encounters its failure first.
4. Parent recovery has exactly epoch 0 and epoch 1, with epoch 1 active for
   later resolver cycles and no third epoch.
5. Only escaped pre-commit goal exhaustion codes can claim parent recovery.
6. A successful recovery returns one valid cognition result and performs one
   final state commit.
7. A failed recovery raises `ParentRecoveryExhaustedError`, preserves the
   second child error and bounded first-error metadata, and produces no commit,
   action, surface, delivery, or outer service retry.
8. Unguarded V1 ledger snapshots, three-call goal tests, strict validation,
   sibling recovery, and existing service graph retry tests pass unchanged in
   meaning and exact call arithmetic.
9. The outer guardrail capsule aggregates bounded epoch metadata while inner
   capsules retain their own exact raw attempts.
10. All changed source and governed README paths resolve to the exact
    deterministic nodes in the matrix and those nodes are collected and run.
11. An independent reviewer signs off the amended plan before implementation;
    implementation review and remediation review remain separate.

## Execution Roles And Gates

| Role | Responsibility | Owned surface | Authority | Applicable skills | Capability floor | Independence requirement | Acceptance output | Gate |
|---|---|---|---|---|---|---|---|---|
| parent plan owner | maintain scope, apply review remediation, preserve lifecycle evidence | plan, registry row, review record | edit draft plan and registry; no production implementation under draft | `development-plan`, `local-llm-architecture` | senior architecture judgment and repository context | separate independent reviewer for final sign-off | amended plan with closed decisions and exact matrix | review findings addressed; status remains draft until approval |
| implementation owner | implement coordinator, connector guard, service arbitration, ledger, and capsule | six production source modules plus exact docs in Change Surface | edit only approved production/doc paths; no semantic changes outside contract | `development-plan`, `local-llm-architecture`, `py-style`, `cjk-safety` | Python async/contextvars, LangGraph/service retry, protected tracing, test access | separate test owner and independent code reviewer | scoped diff, mapped tests, evidence artifacts | approved plan, baseline captured, owned files fixed |
| test owner | create and execute exact deterministic/patched/captured coverage | named test files and impact manifest | add tests and update mappings; preserve intentional failures | `test-style-and-execution`, `debug-llm`, `py-style` | pytest contracts, async mocks, trace artifact inspection | independent from implementation remediation | collected/run node report and inspected artifacts | implementation diff available; no live variance used as contract proof |
| independent plan reviewer | assess plan direction, ownership, contracts, matrix, and residual risk | plan and review evidence only | pass/fail plan review; no remediation edits | `development-plan`, `local-llm-architecture` | high-risk architecture review and repository inspection | independent from parent remediation owner | written verdict with exact findings | amended plan submitted; no execution authorization implied |
| independent code reviewer | assess implementation and verification against approved plan | complete implementation diff and evidence | pass/fail code review; no remediation edits | `development-plan`, `local-llm-architecture`, `test-style-and-execution` | source review, test evidence, retry/side-effect audit | independent from implementation owner | separate code-review verdict | all mapped tests run; residual risks recorded |

## Independent Plan Review Record

- Reviewer: GPT-5.6-sol, high reasoning, supported default service tier.
- Agent reference: `019ff4e1-4da8-7451-aa5d-fdb1e511aaea`.
- Initial verdict: `BLOCKED`.
- Remediation applied in this draft: connector checkpoint placement;
  invocation-wide shared token; exact epoch lifetime; goal-only eligibility;
  side-effect boundary; typed terminal error; separate orchestration
  disposition; V1/V2 snapshot policy; outer capsule; persona-only scope;
  documentation ownership; exact test nodes; complete role fields.
- Final verdict: `PASS` after independent re-review of the current draft.
- Closure state: all plan-review findings are closed, user approval moved the
  plan to `in_progress`, implementation completed, and the plan is now
  `completed` with the final code-review gate passed.

## Independent Code Review Record

- Initial implementation review: GPT-5.6-sol, high reasoning, agent
  `019ff524-f87e-76b0-8067-1fc05d54ed25`, verdict `BLOCKED`.
- Second review: same reviewer, verdict `BLOCKED`; service-first branch
  coverage and numeric capsule-coordinate bounds remained.
- Parent remediation: complete; see Execution Evidence And Remediation.
- Follow-up implementation verdict: same reviewer, `PASS` with no residual
  sign-off issues.
- Plan closure: complete after the final deterministic suite, registry update,
  and archive move.
