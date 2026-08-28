# DSH Plan 3: Brain Big-Bang Cutover And Legacy Resolution Decommission

## Summary

- Goal: replace the task-execution edge beneath the current post-selector
  cognition handover with the accepted DSH Resolution Controller, preserve the
  Brain-owned handover behavior, and delete only the superseded task executors
  in one deployment boundary.
- Status: draft; architectural boundary and preservation contract approved by
  the user on 2026-08-28; production execution remains blocked until Plan 2
  closes and the post-Plan 2 exact inventory is explicitly approved.
- Plan class: destructive Brain-path big-bang cutover and decommission.
- Governing architecture:
  `docs/architecture/dsh_integration_architecture.md`.
- Functional support at exit: Kazusa Brain retains its current cognition
  recurrence and invokes DSH for inline and background task execution,
  consumes the canonical mapped observation/result, continues through the
  existing cognition/action/dialog/delivery order, and resumes the same durable
  DSH session after checkpoint or restart.
- Compatibility policy: the canonical Brain handover remains current product
  behavior; the legacy task-resolution specialist implementation has no
  fallback, alias, translation bridge, dual run, or runtime vocabulary after
  deployment.
- Effort estimate: six functional work blocks. The preserved feature matrix is
  frozen here; the exact post-Plan 2 deletion inventory expands it without
  weakening any preservation row.

## Entry Conditions

Plan 3 may be promoted only when:

1. Plans 1 and 2 are completed and archived with all gates green.
2. The accepted storage-independent DSH semantic catalog and native Standard
   tools cover the functions required by production task resolution.
3. Real standalone evidence proves terminal, clarification, approval,
   checkpoint, cancellation, cold resume, sidecar restart, and fault behavior.
4. A current caller/callee/source/test/data/config inventory identifies every
   legacy post-selector resolution and coding surface.
5. The user approves the exact deployment drain, atomic route switch, deletion
   inventory, and final executable test matrix.

The refined plan is one executable cutover contract. It may adjust paths to the
post-Plan 2 repository, while retaining every functional gate below.

## Mandatory Skills And Rules

- Apply `development-plan` to the post-Plan 2 inventory, tests-first cutover,
  deletion evidence, review, and closure.
- Apply `local-llm-architecture` before changing P-stage, resolver observation,
  DSH intake, semantic tools, or model-facing prompts. Preserve the current
  pipeline contract and smallest execution-edge change radius.
- Apply `no-prepost-user-input` to clarification, approval, and DSH-question
  continuation. Cognition interprets the user's reply; deterministic code
  validates and persists the typed decision.
- Apply `py-style` before Python changes, `cjk-safety` when applicable, and
  `test-style-and-execution` before test creation, modification, or execution.
- Preserve the current inline budget, background worker, accepted-task
  lifecycle, dispatcher delivery, unrelated actions, and prewarm behavior as
  fixed product contracts.

## Fixed Execution Ownership

This coarse draft authorizes planning only. Its post-Plan 2 refinement requires
explicit user approval and `in_progress` status before the first Phase 1 test
edit.

| Role | Fixed owner | Responsibility |
|---|---|---|
| Architecture and closure | Parent agent | Owns cutover/deletion decisions, consolidated material review, gate interpretation, status, and final closure |
| Implementation and verification | The persistent `/root/dsh_implementation_worker` subagent on `gpt-5.6-luna`, `max` reasoning, normal execution speed | Owns all production, test, fixture, and product-documentation edits, deletions, test execution, remediation, and pre-handoff self-review |

The Plan 1/2 Luna worker remains the sole implementation worker and work
proceeds one plan at a time. The parent stays read-only for production code and
test execution. Changing this binding requires a user-approved amendment.

There are at most two review iterations:

1. The worker delivers the complete cutover/decommission candidate and all
   mapped verification. The parent returns one consolidated material finding
   set.
2. When required, the worker resolves the entire set and reruns acceptance.
   The parent passes or blocks the plan.

Minor style, lint, collection, fixture, import, typo, and link defects are
worker self-review responsibilities before handoff. There is no third review
iteration.

Process evidence remains limited to working-tree status, owned paths, final
diff/deletion inventory, exact commands/results, inspected live deployment
cases, and gate decisions. Runtime security digests remain. General source,
workspace, and artifact hashing remains outside scope.

## Mandatory Implementation Order

The post-Plan 2 refinement freezes the complete caller/callee, deletion, data,
configuration, dependency, documentation, and source-to-test change radius
before implementation begins. Execution then follows these three blocking
phases across the whole big-bang cutover:

1. **Tests first.** Create, replace, modify, or delete every planned unit,
   integration, process, static-absence, prewarm, database, live-LLM,
   real-service, adapter, deployment, and documentation test; all fixtures and
   test helpers; and every source-to-test manifest row before changing
   production code. Attempt discovery and collection for the complete frozen
   matrix. Implement the frozen advertised-feature-to-real-LLM-node coverage
   matrix below for every preserved and new Plan 3 product behavior, plus the
   one-to-five final E2E sign-off nodes named below. A collection or execution
   failure may cross the Phase 1 gate only
   when its recorded cause is the planned absence or presence of a production
   symbol that the big-bang cutover changes; syntax, fixture, node-ID, and
   unrelated collection defects are resolved within Phase 1. The Phase 1
   evidence contains the complete test/deletion diff, exact node inventory,
   and expected red results mapped to the production contracts or legacy
   surfaces they govern.
2. **Production changes second.** After the Phase 1 gate passes, implement the
   Brain caller, result mapper, background lifecycle, DSH references, database
   and configuration changes, dependency/package changes, and complete legacy
   production deletion. Run the Phase 1 tests throughout this phase. Code is
   ready for documentation when the production diff and deletion inventory
   are complete, builds and lint pass, and every applicable non-documentation
   deterministic, process, live-DB, live-LLM, real-service, adapter, and drain
   rehearsal behavior gate is green on the DSH-only task-execution edge. Run
   the complete real-LLM feature coverage matrix before remediation, collect
   all failure modes, apply one consolidated remediation pass, and verify fixes
   with the minimum faithful live reproduction and mapped deterministic nodes.
3. **Documentation last.** After the code-ready gate passes, publish the
   preserved Brain handover and DSH task-execution boundary and remove or
   replace every legacy architecture, ICD,
   operator, configuration, and usage example in the frozen documentation
   radius. Then run documentation assertions, link/static checks, the final
   mapped documentation verification, the selected one-to-five E2E real-LLM
   sign-off nodes, and the closure audit.

Production and documentation paths remain at the captured execution baseline
during Phase 1. Documentation paths remain at that baseline during Phase 2.
A phase advances only after its evidence and exit gate are recorded. Any
change-radius expansion returns execution to Phase 1 for its complete test
work before the additional production cutover work proceeds. The deployment
remains one atomic DSH task-execution release boundary; the three phases govern how that
single candidate is built.

## Target Production Boundary

```text
adapter/debug client
    -> Kazusa Brain
    -> Cognition V3 P-stage response plan
       -> action_requests -------------------------------> existing action path
       -> resolver_requests(task_resolution_request)
          -> existing cognition_resolver recurrence
             -> Resolution Controller
                -> durable ResolutionThread / DSH segment
                -> independent DSH sidecar
                -> accepted Plan 2 tools
                -> submit_resolution or runtime checkpoint/fault
             <- DSHResolutionExhaustV1
             -> typed resolver observation / TaskResolutionResultV1
          -> full cognition runs again and Brain judges sufficiency
    -> existing state commit / memory / action / L3 dialog order
    -> existing accepted-task result-ready / dispatcher / adapter delivery
```

The Brain remains the caller and owns the P-stage semantic decision, priority,
the inline-versus-background wait decision, scheduling, approval and
clarification judgment, cognition recurrence, final wording, delivery, and
durable product state. DSH owns its autonomous session, native tool selection,
Kazusa semantic-tool selection, and evidence resolution. Kazusa interfaces
through the agreed Resolution Controller operations and does not manage the
DSH reasoning loop. The sidecar remains an independently operating process and
DSH source remains untouched.

## Current Post-Selector Handover Preservation Contract

The cutover changes one execution edge, not the Brain handover around it. The
following current behaviors are mandatory retained product features. A path
may be refactored only when its replacement preserves the stated owner,
carrier, ordering, and observable result in the same atomic change.

| Current owner and source | Preserved behavior at DSH cutover |
|---|---|
| Cognition P-stage in `src/kazusa_ai_chatbot/cognition_core_v3/` and materialization in `nodes/persona_supervisor2_cognition.py` | P-stage chooses the semantic goal and keeps `action_requests` separate from `resolver_requests`. `task_resolution_request` remains a resolver capability rather than an action-spec worker selector. Deterministic code binds the exact caller-owned `goal_continuation_ref`; cognition does not choose a worker, timeout, checkpoint, or DSH tool. |
| `nodes/persona_supervisor2.stage_1_goal_resolver` and `cognition_resolver.loop.call_cognition_resolver_loop` | One selected capability executes, its typed observation is appended, and the full cognition chain runs again. Only the final replacement cognition state is committed before memory, actions, L3, dialog, or delivery. `answerable_now` still terminates optional resolver work. |
| `cognition_resolver.loop` lifecycle checks | Goal progress and deliverables survive recurrence. Required-evidence dependencies, observation ids, evidence handles, exact continuation references, repeated/failed capability prevention, and same-reference conflicts between factual answers, pending work, acknowledgements, and terminal results remain enforced. |
| `cognition_resolver.capabilities._execute_task_resolution_request` | The DSH intake receives the existing prompt-safe task execution context: cognition scene, semantic objective, continuation reference, persona/prompt context, bounded recent and wide history, conversation progress, decontextualized summary, source message lineage, timestamp, media references, workspace, and output requirement. Adapter objects, database interfaces, worker internals, and raw platform syntax remain outside the intake. |
| `cognition_resolver.pending` and the Cognition L2d/P decision | Clarification and approval remain Brain-owned semantic decisions. Pending state preserves original goal/progress, exact conversation/user/source-message lineage, expiry, and action-attempt identity. A user reply is interpreted by cognition before deterministic code applies the result. DSH questions use the Plan 2 Brain interaction bridge and this same semantic pattern. |
| `task_resolution.service.resolve_task_inline` caller behavior | The current configured inline wait budget, including the current 30-second default and its existing configuration surface, remains unchanged. It decides whether Kazusa waits for the DSH result or requests `resolution.request_checkpoint`; it does not become a DSH reasoning timeout or cancellation policy. |
| `agentic_resolver.controller` agreed interface | Kazusa uses `resolution.open`, `resolution.continue`, `resolution.amend`, `resolution.request_checkpoint`, `resolution.cancel`, `resolution.inspect`, and `resolution.dispose_activation` only. Foreground promotion requests a cooperative checkpoint at the next safe boundary and retains the same ResolutionThread/segment/session lineage. |
| `task_resolution` promotion, `accepted_task`, and `background_work` | Deferred promotion remains idempotent: accepted-task state becomes durable, is marked pending before queue insertion, and reuses the same identity after interrupted enqueue. The current worker lease/checkpoint/retry/result-ready ordering remains. Plan 3 replaces the legacy checkpoint payload with an opaque DSH thread/segment/session reference and adds the exact DSH resume operation to the existing reviewed `task_orchestrator` worker. |
| `background_work.result_source`, `background_work.delivery`, and `service.py` | Background completion is mapped from a validated typed result, creates a canonical `tool_result` cognition episode with the exact goal continuation reference and provenance, and re-enters normal cognition. Brain owns final visible wording. Dispatcher/adapter delivery, result-ready claims, retry recovery, source-message reply targeting, and delivery finalization remain unchanged. |
| `persona_supervisor2` stage order and `action_spec` | Final cognition commits before the current memory/action/dialog sequence. Pre-surface private actions, visible surfaces, L3, action result traces, consolidation, and post-turn work retain their order. `speak`, `memory_lifecycle_update`, `trigger_future_cognition`, `future_speak`, and `accepted_task_status_check` remain outside the DSH task-execution replacement. |
| Internal-thought and shared-memory prewarm paths | Internal-thought sources remain private, visible fallback actions retain caller-owned target provenance, and the shared-memory prewarm checkpoint/merge behavior remains independent and unchanged. |

### Coding Lifecycle Preservation

DSH Standard replaces the legacy coding implementation, while the Brain-owned
product lifecycle remains explicit:

| Current product behavior | DSH-era disposition |
|---|---|
| Start a new coding task from `task_resolution_request` | The cognition-resolver task edge opens or continues a DSH resolution thread; DSH Standard owns coding tools and workspace work. |
| `accepted_coding_task_request` with `revise_proposal`, `summarize`, `status`, `approve_and_verify`, `respond_to_blocker`, or `cancel` | Preserve the visible semantic actions until each is mapped to an exact DSH thread/session operation and accepted-task result. Delete the old coding-run reference vocabulary only in the same change that updates P-stage affordances, action materialization, worker payloads, pending interactions, result projection, and tests to the DSH session reference. |
| Coding clarification or approval | Brain judges the DSH question through the Plan 2 interaction bridge and may decide itself or relay to the user. The user never becomes the owner of the DSH session. |
| Coding completion or blocker | A typed DSH terminal/checkpoint/fault result enters accepted-task result-ready state and normal tool-result cognition; the worker does not author final dialogue or deliver directly. |

No coding feature may disappear because its old module is selected for
deletion. Every row requires an explicit replacement mapping or a separately
approved product-removal decision before its legacy source is deleted.

## Cutover Change Radius And Disposition

| Radius | Fixed disposition |
|---|---|
| Cognition P-stage and request materialization | Preserve semantic ownership and both request lanes; modify only the DSH interaction/result fields required by the accepted contracts. |
| `cognition_resolver` recurrence, pending state, lifecycle validation, observation projection, and visible/private fallback behavior | Preserve as the canonical post-selector handover owner. Replace only `_execute_task_resolution_request`'s legacy task-session call with the Resolution Controller adapter and canonical result mapper. |
| Legacy `task_resolution` orchestrator, state machine, four-specialist selector/adapters, and specialist prompts | Delete after the DSH edge satisfies every preserved behavior and mapped test. Retain no fallback import or checkpoint reader. |
| `local_context_resolver`, `complex_task_resolver`, post-selector RAG routes, and old external/internal resolution graphs | Delete only executable routes replaced by DSH. Retain shared-memory prewarm and every leaf/service with a surviving non-DSH caller. The frozen post-Plan 2 caller inventory decides each exact file disposition before Phase 1. |
| Legacy `coding_agent` implementation and coding-run persistence | Delete only after the coding lifecycle table has an accepted DSH mapping for every still-supported user-visible action. Historical rows follow the deployment data disposition and remain unavailable as runtime fallback. |
| `accepted_task` and `background_work` | Preserve lifecycle, scheduling, leases, retry, result-ready, and delivery owners. Replace only the legacy task checkpoint/operation variant with the DSH reference/resume variant. Preserve `future_speak`. |
| `action_spec` | Preserve unrelated action capabilities. Replace coding-continuation vocabulary atomically when the DSH mapping is ready; preserve status and visible-surface semantics. |
| Dialog, dispatcher, adapters, consolidation, scheduler, reflection, and product persistence | Keep outside the execution replacement except for the exact typed DSH result carrier required at their existing boundary. |

## Big-Bang Cutover Rules

1. Update only the task-execution edge beneath `cognition_resolver`, Resolution
   Controller composition, result mapping, background resume payload, tests,
   configuration, and ICD in one atomic release scope using the mandatory
   tests-first, production-second, documentation-last order.
2. Preserve `task_resolution_request` as the Brain-facing action name.
3. Map validated DSH terminal statuses to the canonical
   `TaskResolutionResultV1` statuses.
4. Preserve the existing inline wait budget and map its cooperative
   `resolution.request_checkpoint` exhaust to runtime-owned `deferred`,
   persisting only the opaque ResolutionThread/segment/session/revision
   reference. Add no DSH step, call, byte, deadline, or cancellation policy.
5. Route user clarification and approval responses back through normal Brain
   interpretation before explicit same-goal DSH continuation.
6. Keep background queue, priority, scheduler, dispatcher, adapter, and
   delivery authority in Kazusa.
7. Preserve the current shared-memory prewarm path and its ownership.
8. Delete the old task-resolution specialist orchestration, superseded
   post-selector RAG/internal/external execution graphs, and standalone coding
   implementation replaced by DSH. Preserve `cognition_resolver` as the
   post-selector Brain handover owner and preserve every current behavior named
   in this plan.
9. Retain every semantic service or leaf with a Plan 2 gateway or surviving
   Kazusa caller. Delete only its superseded graph/router entry points and
   vocabularies after the exact caller inventory proves the retained owner.
10. Replace obsolete tests and source-to-test manifest rows in Phase 1 while
    retaining the preservation anchors below; remove obsolete prompts, models,
    specialist state/checkpoint DTOs, coding-run payload fields,
    configuration, exports, scripts, and package dependencies in Phase 2;
    remove obsolete docs in Phase 3. All removals ship in the same cutover.
11. Convert no legacy checkpoint or active graph state into DSH state. New work
    begins on the DSH contract.
12. Keep historical records only where audit or retention policy requires
    them; no runtime code reads them as a compatibility source.

## Deployment Drain

Before the atomic deployment, Kazusa stops admitting new legacy resolution and
coding work and reaches zero active legacy inline/background executions.
Remaining stragglers follow one explicitly approved terminal administrative
disposition before deployment. The cutover then installs the DSH-only
task-execution edge and deletion set together.

This drain is an execution safety gate rather than a compatibility period.
After deployment the preserved Brain handover has one production
task-execution route.

## Coarse Decommission Surface

The refined plan must enumerate exact files from these owners:

| Owner | Cutover disposition |
|---|---|
| Brain task-resolution caller | Preserve the P-stage and `cognition_resolver` caller; replace only `_execute_task_resolution_request`'s legacy service call with the Plan 1 runtime/controller and Plan 3 result mapper |
| Legacy task-resolution orchestrator/state/specialists | Delete after all required leaf functions are owned by the Plan 2 gateway |
| Post-selector RAG and internal/external executors | Delete only superseded routing, graph, synthesis, prompt, state, export, and configuration surfaces; retain `cognition_resolver`, shared-memory prewarm, and every semantic service with a surviving caller |
| Legacy coding agent/harness | Map every supported coding lifecycle action to DSH session operations first, then delete runtime, workflow, routing, prompt, test, configuration, persistence, and package surfaces replaced by DSH coding |
| Accepted tasks/background work | Preserve the current lifecycle and worker; replace only the legacy graph checkpoint operation/payload with opaque DSH thread/segment/session/revision references and a reviewed DSH resume operation |
| Cognition handoff | Preserve recurrence and typed observation projection; consume canonical `TaskResolutionResultV1` at the same point and keep cognition judgment and dialog wording ownership unchanged |
| Database | Add or update only DSH-backed task/thread references needed by the new route; retire legacy runtime reads and indexes according to the approved data disposition |
| Test ownership | Preserve the current handover regression anchors, replace only legacy-executor nodes and manifest rows, and add the complete DSH-edge matrix in Phase 1 |
| Documentation ownership | Publish the preserved Brain handover and DSH task-execution production boundary and remove legacy executor examples in Phase 3 after the code-ready gate |

## Work Blocks And Effort Gates

These blocks are cutover scope groupings rather than chronological edit order.
Their complete test, fixture, and manifest portions execute in Phase 1; their
production and deletion portions execute in Phase 2; and their documentation
portions execute in Phase 3 under the mandatory order above.

| Block | Relative effort | Work | Independent completion gate |
|---|---|---|---|
| 1. Final ownership and drain contract | Medium | Freeze caller/callee/deletion/data/config inventory and operational drain | Every active legacy producer/consumer has one cutover disposition and zero legacy work remains at deployment |
| 2. Preserved handover and result mapping | High | Connect only `cognition_resolver`'s task-execution edge to the controller and map terminal/checkpoint/fault outcomes without changing recurrence, pending, evidence, or stage ordering | Inline resolved/partial/clarification/approval/unavailable/failed/deferred paths reach full cognition through the current typed observation boundary |
| 3. Background and lifecycle cutover | High | Replace the task-resume payload beneath the existing accepted-task/background lifecycle with opaque DSH thread/session references | Foreground promotion and background completion resume the same DSH session while current idempotency, leases, result-ready, delivery, and `future_speak` behavior remain green |
| 4. Legacy design deletion | High | Define legacy-executor test/manifest removal in Phase 1, delete superseded task-resolution specialists, post-selector RAG/internal/external execution graphs, and mapped coding implementation in Phase 2, and remove legacy documentation in Phase 3 | Static tests prove no executable legacy task route remains while every preservation anchor and surviving non-DSH caller remains green |
| 5. Production fault and behavior validation | High | Validate sidecar absence/restart, Mongo/SQLite recovery, duplicate lease, scope rotation, approval, dialog handoff, and adapter delivery | Real service tests pass on DSH only and faults remain typed without fallback to legacy code |
| 6. Final deployment and closure | Medium | Complete the Phase 2 drain/cutover rehearsal and code-ready gate, perform the Phase 3 documentation update, then run final deterministic/live suites and the closure audit | All release gates are green on the exact deployment candidate and the parent closes the preserved-handover/DSH-executor plan |

## Functional Release Gates

| Gate | Green condition |
|---|---|
| P3-G1 — Single task-execution route | Every `task_resolution_request` reaches DSH through the preserved `cognition_resolver` edge and no executable legacy task executor or fallback remains |
| P3-G2 — Handover preservation | Action/resolver lane separation, continuation refs, cognition recurrence, goal progress, evidence dependencies, lifecycle conflicts, pending interaction semantics, private/visible fallback behavior, and state/action/dialog ordering pass their retained anchors |
| P3-G3 — Brain ownership | P-stage decision, inline-versus-background wait, scheduling, approval/clarification judgment, cognition, final wording, adapter delivery, and product persistence remain Kazusa-owned; Brain interfaces with DSH only through the agreed controller operations |
| P3-G4 — Lifecycle continuity | The current 30-second inline default, cooperative checkpoint, foreground promotion, direct background, continuation, cancellation, process restart, and completion preserve the correct thread/session lineage without imposing a DSH reasoning timeout |
| P3-G5 — Complete functional parity | The accepted Plan 2 semantic catalog and DSH Standard native tools supply the required old functions; every supported coding lifecycle action has an exact DSH mapping before its old implementation is removed |
| P3-G6 — Prewarm and unrelated actions retained | Shared-memory prewarm plus `speak`, memory lifecycle, future cognition, `future_speak`, and accepted-task status behavior remain functional and independent of the deleted task executors |
| P3-G7 — Decommission complete | Legacy task-executor code, prompts, checkpoints, config, exports, dependencies, replaced tests, docs, and manifest ownership are absent or retained solely as non-runtime historical data under an explicit disposition |
| P3-G8 — Fault isolation | Sidecar failure, RPC fault, DSH crash repair, Mongo/SQLite recovery, lease conflict, scope/audience mismatch, and uncertain side effects fail or defer through typed DSH-era outcomes while current Brain recovery behavior remains intact |
| P3-G9 — Real production flow | Real Brain-service cases reach cognition/action/dialog/delivery through the DSH-only task executor for inline, background, clarification, approval, coding lifecycle, restart, and failure scenarios |
| P3-G10 — Real-LLM coverage and E2E sign-off | Every preserved and new advertised Plan 3 feature is exercised in the complete real-local-model coverage pass before remediation. Failures are consolidated, fixed in one pass, and verified with the minimum faithful real-LLM reproductions. Final sign-off uses no more than five named E2E nodes and every selected node passes behavioral inspection. |

All ten gates are release blockers. A fallback to any legacy task executor is a
failed gate, even if the visible answer appears correct.

## Test Impact And Traceability

The post-Plan 2 refinement expands this table to every changed/deleted path.
The following current nodes are immutable preservation anchors: Phase 1 may
move a node only when the replacement node asserts the same behavior at the
same semantic owner in the same test diff.

| Source or governed path | Preserved contract and semantic owner | Exact deterministic pytest nodes | Mode and regression prevented |
|---|---|---|---|
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | Persona graph owns recurrence and final cognition commit | `tests/test_cognition_resolver_persona_graph.py::test_persona_graph_has_one_v2_resolver_path`; `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_resolver_recurrence_commits_against_original_user_base` | unit/integration; prevents bypassing full cognition or committing an intermediate DSH observation |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | P-stage request materialization preserves resolver/action separation and exact continuation identity | `tests/test_cognition_resolver_loop.py::test_resolver_request_and_dependency_preserve_goal_continuation_ref`; `tests/test_cognition_resolver_loop.py::test_task_resolution_context_preserves_scene_and_continuation_ref` | deterministic integration; prevents worker selection in cognition and lost goal lineage |
| `src/kazusa_ai_chatbot/cognition_resolver/loop.py` | Cognition resolver owns observation recurrence, sufficiency termination, goal progress, evidence dependencies, lifecycle conflicts, and private/visible blocker convergence | `tests/test_cognition_resolver_loop.py::test_loop_runs_cognition_capability_then_cognition_again`; `tests/test_cognition_resolver_loop.py::test_answerable_now_terminates_without_executing_optional_resolver`; `tests/test_cognition_resolver_loop.py::test_loop_projects_goal_progress_across_iterations`; `tests/test_cognition_resolver_loop.py::test_pending_background_goal_reaches_acknowledgement_without_factual_surface`; `tests/test_cognition_resolver_loop.py::test_duplicate_final_cognition_internal_thought_stays_private` | direct owner/integration; prevents reducing the handover to selector-to-worker dispatch |
| `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py` | Task capability owns prompt-safe context projection and typed blocker/evidence mapping | `tests/test_cognition_resolver_loop.py::test_task_resolution_uses_objective_and_preserves_context`; `tests/test_cognition_resolver_loop.py::test_task_resolution_bounds_history_to_its_context_contract`; `tests/test_cognition_resolver_loop.py::test_task_resolution_user_input_result_is_blocked`; `tests/test_cognition_resolver_loop.py::test_public_evidence_projects_through_task_resolution` | direct owner/integration; prevents context loss, raw carrier exposure, or treating pending work as factual evidence |
| `src/kazusa_ai_chatbot/cognition_resolver/pending.py` | Brain cognition owns user-reply interpretation and pending goal restoration | `tests/test_cognition_resolver_loop.py::test_pending_resolution_is_applied_only_after_l2d_decision`; `tests/test_cognition_resolver_loop.py::test_hil_follow_up_can_continue_original_goal_after_answer`; `tests/test_cognition_resolver_loop.py::test_pending_resume_load_restores_original_goal_progress`; `tests/test_cognition_resolver_loop.py::test_same_message_pending_resolution_is_ignored` | direct owner/integration; prevents deterministic keyword interpretation, wrong-message resume, or lost goal progress |
| `src/kazusa_ai_chatbot/task_resolution/service.py` and the replacement DSH edge | Existing inline wait and idempotent promotion semantics | `tests/test_task_resolution_inline_promotion.py::test_inline_budget_default_is_thirty_seconds`; `tests/test_task_resolution_inline_promotion.py::test_deferred_promotion_marks_pending_before_job_insert`; `tests/test_task_resolution_inline_promotion.py::test_interrupted_enqueueing_promotion_reuses_idempotency_key`; `tests/test_task_resolution_inline_promotion.py::test_pending_promotion_repairs_missing_job_idempotently` | direct owner/integration; prevents changing the user-facing wait, duplicate jobs, or lost promotion recovery |
| `src/kazusa_ai_chatbot/accepted_task/lifecycle.py` | Accepted task owns scoped durable identity and pending/result/delivery lifecycle | `tests/test_accepted_task_lifecycle.py::test_create_or_return_active_persists_goal_continuation_ref`; `tests/test_accepted_task_lifecycle.py::test_mark_pending_records_internal_executor_ref`; `tests/test_accepted_task_lifecycle.py::test_terminal_transitions_require_running_and_delivery_claims`; `tests/test_accepted_task_lifecycle.py::test_status_check_returns_latest_active_task_for_scope` | direct owner; prevents changing accepted-task identity, transition order, or status semantics |
| `src/kazusa_ai_chatbot/background_work/subagent/task_orchestrator.py` and `src/kazusa_ai_chatbot/background_work/worker.py` | Existing worker owns lease-fenced resume/checkpoint/result-ready ordering | `tests/test_task_resolution_background_resume.py::test_terminal_snapshot_recovers_without_redispatch`; `tests/test_task_resolution_background_resume.py::test_dispatch_snapshot_persists_under_active_lease`; `tests/test_task_resolution_background_resume.py::test_queue_retry_preserves_semantic_counters_and_prior_result`; `tests/test_task_resolution_background_resume.py::test_accepted_result_is_ready_before_job_releases_lease` | direct owner/integration; prevents redispatch, lease-unsafe writes, or releasing work before durable result-ready state |
| `src/kazusa_ai_chatbot/background_work/result_source.py` | Typed accepted-task result becomes canonical prompt-safe `tool_result` evidence | `tests/test_background_work_delivery.py::test_tool_result_source_builder_creates_prompt_safe_episode`; `tests/test_background_work_delivery.py::test_result_source_preserves_typed_task_status_and_ref`; `tests/test_background_work_delivery.py::test_tool_result_source_builder_ignores_untyped_job_summary`; `tests/test_accepted_task_prompt_contract.py::test_tool_result_reenters_as_typed_evidence` | direct owner/integration; prevents untyped worker prose or lost continuation refs entering cognition |
| `src/kazusa_ai_chatbot/background_work/delivery.py` and `src/kazusa_ai_chatbot/service.py` | Brain/dispatcher own result wording, reply targeting, delivery claims, retries, and finalization | `tests/test_background_work_delivery.py::test_service_result_ready_delivery_uses_dispatcher_boundary`; `tests/test_background_work_delivery.py::test_background_reply_target_uses_original_source_on_durable_age`; `tests/test_background_work_delivery.py::test_delivery_tick_syncs_accepted_task_delivery_state`; `tests/test_background_work_delivery.py::test_delivery_tick_recovers_stale_delivery_claims_before_scan` | cross-boundary integration; prevents worker-authored dialog, synthetic reply targets, or inconsistent delivery state |
| `src/kazusa_ai_chatbot/action_spec/registry.py` and `src/kazusa_ai_chatbot/action_spec/execution.py` | Unrelated actions and visible/private surface semantics remain independent | `tests/test_action_spec_evaluator.py::test_initial_registry_contains_only_approved_runtime_capabilities`; `tests/test_action_spec_results.py::test_surface_output_preserves_surface_role_and_ref`; `tests/test_action_spec_memory_lifecycle.py::test_memory_lifecycle_execute_uses_repository_owner`; `tests/test_background_work_future_speak.py::test_future_speak_execution_enqueues_requested_worker` | direct owner/integration; prevents DSH cutover from deleting memory, scheduling, status, or surface behavior |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` shared-memory prewarm boundary | Prewarm attempt checkpoint and merge remain independent | `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_cognition_state_preserves_shared_memory_prewarm_outcome_after_merge`; `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_cognition_cancellation_publishes_no_prewarm_outcome_or_observation` | direct owner; prevents stale prewarm evidence or cancellation leakage during cutover |
| `src/kazusa_ai_chatbot/background_work/models.py` and coding-continuation replacement | Supported coding lifecycle operations remain product-visible until mapped to DSH | `tests/test_background_work_jobs.py::test_bound_coding_payload_validates_frozen_closed_request`; `tests/test_task_resolution_background_resume.py::test_bound_coding_continuation_preserves_scene_and_ref`; `tests/test_coding_agent_phase_d_action_loop_contracts.py::test_context_exposes_semantic_capabilities_without_host_details` | contract/integration; prevents silent removal of revise/status/approve/blocker/cancel behavior |

The following new cross-boundary nodes are fixed names for the refined Plan 3
matrix and must be created during Phase 1:

- `tests/test_dsh_task_resolution_cutover.py::test_cognition_resolver_task_edge_uses_dsh_and_reruns_full_cognition`
- `tests/test_dsh_task_resolution_cutover.py::test_inline_budget_requests_checkpoint_and_promotes_same_dsh_session`
- `tests/test_dsh_task_resolution_cutover.py::test_dsh_question_is_brain_judged_and_user_reply_resumes_same_goal`
- `tests/test_dsh_background_resume.py::test_existing_worker_resumes_dsh_reference_and_preserves_result_ready_order`
- `tests/test_dsh_coding_lifecycle_cutover.py::test_supported_coding_actions_map_to_dsh_session_operations`
- `tests/test_dsh_task_resolution_decommission.py::test_legacy_task_executors_are_absent_and_preserved_handover_remains`

## Real LLM Coverage And Final Sign-Off

Real-LLM coverage is defined by advertised Plan 3 product behavior:

- every P3 release-gate capability;
- every preserved behavior in the Current Post-Selector Handover Preservation
  Contract;
- each supported coding lifecycle action; and
- DSH Standard at the Plan 3 product-capability level used by the cutover,
  rather than every individual upstream helper tool.

Phase 1 implements this frozen advertised-feature-to-live-node matrix:

| Advertised or preserved feature | Exact real-LLM coverage owner |
|---|---|
| P-stage semantic goal ownership, action/resolver lane separation, `answerable_now`, and private internal-thought handling | `tests/test_dsh_plan3_feature_live_llm.py::test_live_p_stage_preserves_action_resolver_lanes_answerable_now_and_private_sources` |
| Prompt-safe DSH intake, typed observation projection, goal/evidence continuity, full cognition recurrence, and final cognition/action/dialog order | `tests/test_dsh_plan3_feature_live_llm.py::test_live_dsh_observation_reenters_full_cognition_with_goal_and_evidence_state` |
| Brain answers or rejects a DSH clarification/approval from current context without user delivery | `tests/test_dsh_plan3_feature_live_llm.py::test_live_brain_answers_dsh_question_without_user_delivery` |
| Brain relays a DSH question, interprets the user reply, and resumes the same goal and DSH session | `tests/test_dsh_plan3_feature_live_llm.py::test_live_brain_relays_dsh_question_and_resumes_same_goal_from_user_reply` |
| Current inline wait budget, cooperative checkpoint, idempotent promotion, direct background start, same-session resume, typed `tool_result`, Brain wording, and dispatcher delivery | `tests/test_dsh_plan3_feature_live_llm.py::test_live_inline_promotion_and_direct_background_resume_same_dsh_session_through_delivery` |
| Process restart, typed DSH fault/checkpoint outcomes, lease/retry recovery, and session/evidence continuity | `tests/test_dsh_plan3_feature_live_llm.py::test_live_dsh_restart_and_fault_recovery_preserve_session_and_typed_outcome` |
| New coding task plus `revise_proposal`, `summarize`, `status`, and `approve_and_verify` on the same DSH session | `tests/test_dsh_plan3_feature_live_llm.py::test_live_coding_revision_summary_status_and_approval_continue_same_dsh_session` |
| Coding `respond_to_blocker` and `cancel`, including Brain judgment and typed accepted-task results | `tests/test_dsh_plan3_feature_live_llm.py::test_live_coding_blocker_response_and_cancel_continue_same_dsh_session` |
| `speak`, `memory_lifecycle_update`, `trigger_future_cognition`, `future_speak`, and `accepted_task_status_check` remain product-visible beside the DSH cutover | `tests/test_dsh_plan3_feature_live_llm.py::test_live_unrelated_action_capabilities_remain_available_after_cutover` |
| Shared-memory prewarm remains model-visible and independent of the DSH task edge | `tests/test_shared_memory_prewarm_live_llm.py::test_live_bare_tag_prewarm_recovers_missing_generator_query` |
| DSH-only inline, background, interaction, coding, and recovery production routes after legacy executor removal | the selected `tests/test_dsh_plan3_e2e_live_llm.py` final-sign-off nodes listed below |

Each advertised feature therefore has a concrete real-LLM owner. Patched
handoff and deterministic owner tests remain useful for exact code contracts,
including static legacy absence and storage/index disposition, but they cannot
substitute for exercising the corresponding Brain-to-DSH-to-Brain product
behavior with the real local model.

The execution and remediation cycle is fixed:

1. Run every real-LLM coverage node individually with `-q -s`, inspect it, and
   record its observed behavior and failure mode before moving to the next
   node. Continue through semantic failures so the implementation owner sees
   the complete failure set. Stop only when the model backend or harness makes
   subsequent results invalid.
2. Group the complete failure set by shared root cause and apply one
   consolidated remediation pass across the approved change radius. Avoid
   one-case-at-a-time prompt or implementation tuning.
3. Verify each fix with the minimum real-LLM node that faithfully reproduces
   the failure, its mapped deterministic owner nodes, and only directly
   affected neighboring live behavior. A minor fix does not trigger the full
   repository suite or complete live matrix.
4. When a failure is not reproducible, inspect the original trace, exact
   backend/configuration, and nearest model-facing boundary, then construct the
   smallest faithful reproducer. Broad-suite runs and repeated unstructured
   model calls are not reproduction evidence.

Final sign-off uses between one and five real-LLM E2E nodes, selecting the
smallest set that covers the completed Plan 3 surface. The maximum set is:

- `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_inline_dsh_result_returns_through_cognition_action_and_dialog`
- `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_inline_budget_checkpoints_background_resume_and_delivery`
- `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_brain_judges_question_relay_reply_and_same_goal_resume`
- `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_coding_task_and_supported_continuation_complete_through_dsh`
- `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_restart_recovery_preserves_session_evidence_and_delivery`

These nodes are the final agent-behavior sign-off for the cutover. They use the
real local model through the full production boundary and judge preserved
cognition ownership, DSH execution, lifecycle continuity, visible behavior,
and delivery. A deterministic green suite cannot waive a failed E2E behavioral
judgment.

## Verification Direction

The refined plan supplies exact node IDs for:

- strict caller/result/checkpoint contracts;
- inline and background Brain integration;
- Cognition P-stage ownership and action/resolver lane separation;
- continuation, audience/scope rotation, lease, and restart;
- cognition/dialog/delivery handoff;
- coding lifecycle mapping;
- retained prewarm behavior;
- static legacy absence and dependency removal;
- database/index/config disposition;
- sidecar fault isolation;
- real MongoDB;
- real DSH LLM tool-resolution cases; and
- real service/adaptor end-to-end behavior.

Regular deterministic tests run in batches through
`venv\Scripts\python`. Each live DB, live LLM, and real-service case runs
individually with full output inspection. The final candidate also passes the
sidecar TypeScript build/tests and the repository deterministic suite.

## Out Of Scope

- A compatibility window, shadow route, dual execution, checkpoint converter,
  legacy fallback, or staged percentage rollout.
- DSH source modifications.
- Moving cognition, dialog, scheduling, approval UX, or adapter delivery into
  DSH.
- Adding Kazusa runtime management, reasoning deadlines, step/tool budgets,
  prompt guardrails, permission overlays, or a control API beyond the agreed
  Resolution Controller interface.
- Removal or redesign of the accepted shared-memory prewarm path.
- Unrelated Brain, character, memory, dialog, adapter, or control-console
  changes.

## Closure

The Luna worker records the exact cutover/deletion inventory and all functional
evidence. The parent verifies that production has one preserved
`cognition_resolver` handover with one DSH task-execution edge, that every
superseded executor is absent, and that every preservation anchor remains
green. The selected one-to-five E2E real-LLM nodes provide the final behavioral
sign-off before the parent alone marks and archives the plan. Closure is a
functional DSH cutover, not merely a successful build.
