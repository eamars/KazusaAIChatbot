# Task-resolution duplicate visible delivery bugfix plan

## Summary

- Status: `completed`
- Type: cross-boundary semantic lifecycle and contract bugfix
- Evidence: job `job-d9087f00ede5413f917930b07ad14846`, accepted task `task-d9087f00ede5413f917930b07ad14846`, parent trace `llmtrace_fdd67ee9703b4ebcadf651ed57253e7b`, child trace `llmtrace_4cce34c21aa749e389427012e14ed55c`
- Independent review: [GPT-5.6 Sol review](</C:/workspace/kazusa_ai_chatbot_dev2/test_artifacts/diagnostics/task_resolution_duplicate_visible_delivery_plan_review.md>)
- Review configuration: `gpt-5.6-sol`, high reasoning, normal speed
- Execution boundary: production implementation was explicitly authorized by the user on 2026-08-14. DeepSeek Flash owns production-code changes; the parent owns tests, text, evidence, database operations, and closure; the native reviewer is read-only; GPT-5.6 Sol performs final sign-off.

The historical production trace demonstrates a semantic double-delivery:
the parent turn answered the group-history question, then one accepted
background task produced a second visible result episode. Queue idempotency was
correct: one accepted task, one worker, one job idempotency key, and one
delivery attempt. The defect is the coexistence of two visible semantic owners,
not duplicate enqueue.

The fresh real-LLM replay is a counterexample, not a dismissal of the defect.
With the same source messages projected into a current active group scene, the
current model selected a grounded direct answer and created no task. Replaying
the historical planner payload also selected `answerable_now` in the current
run. The bug is therefore model- and projection-sensitive and requires a
deterministic contract guard in addition to prompt/evidence repair.

## Scope and constraints

### In scope

1. Use one exact shared scene representation for parent cognition, action
   planning, inline resolution, and background local-context resolution.
2. Introduce one typed goal-continuation reference that survives a new
   `tool-result:<task_id>` episode and correlates the original goal, task, and
   result without text matching.
3. Enforce a one-factual-surface lifecycle:
   - direct answer: one factual surface and zero accepted task rows;
   - accepted background work: an optional acknowledgement surface with no
     factual answer, followed later by one typed factual or status result;
   - failed or unavailable work: one objective-scoped status surface with no
     invented factual answer.
4. Preserve the existing worker -> normal cognition/dialog -> dispatcher ->
   adapter boundary.
5. Add deterministic contract/replay coverage and a one-case real-LLM
   observation with a human-readable artifact.

### Out of scope

- Queue, lease, or idempotency redesign.
- Direct worker-to-adapter or worker-to-dialog delivery.
- Model selection, temperature, or broad prompt changes unrelated to the
  evidence and surface contracts.
- Keyword routing, deterministic rewriting of user input, or mechanical
  suppression of independently grounded character speech.
- Compatibility aliases, parallel vocabularies, or a second task protocol.
- The unrelated relational-carrier changes already present in the working tree.

### Required engineering rules

- LLM stages own semantic answerability, goal selection, stance, and wording.
- Deterministic code owns typed projection, validation, persistence, limits,
  permissions, lifecycle arbitration, and delivery.
- Contradictory model output receives bounded regeneration through the existing
  canonical JSON parser/repair path. Exhaustion fails closed with typed state;
  deterministic code does not convert missing evidence into
  `answerable_now`.
- New contracts move in one big-bang change. The implementation carries no
  legacy field alias or fallback mapper.

## Reproduction and RCA

### Reproduced production sequence

1. The parent Asuna turn visibly emitted four logical messages summarizing
   千早爱音's self-praise and 阿影's `嘎嘎嘎` response.
2. The same objective was promoted to one durable background task with
   idempotency key `background_work:task-d9087f00ede5413f917930b07ad14846`.
3. The worker's `local_context` specialist returned no evidence and the task
   became unavailable/incompatible.
4. Result delivery re-entered ordinary cognition/dialog and emitted a second
   visible Asuna response about an unrelated “淑女限制/手机闹脾气”
   continuation.

Raw evidence and the inspected live runs are recorded in
[the live LLM review](</C:/workspace/kazusa_ai_chatbot_dev2/test_artifacts/diagnostics/task_resolution_duplicate_delivery_live_review.md>).

### Root cause

`persona_supervisor2.py` constructs `public_group_scene`; the cognition input
builder in `persona_supervisor2_cognition.py` consumes it as the validated
`SceneContextV2.scene_context`. `action_selection.plan_actions()` accepts a
`scene_context` parameter, but its model payload projects bids/evidence and
omits the scene context. Cognition/dialog can therefore see grounded group
facts while action planning classifies the same request as missing required
evidence.

The task persistence gate is `_execute_task_resolution_request()` in
`cognition_resolver/capabilities.py`, which calls task-resolution service
promotion. `stage_2a_background_work_enqueue()` handles selected action-spec
work and is not the task-resolution enqueue owner.

The persisted `TaskResolutionExecutionContextV1` currently contains bounded
history and requester context but no `SceneContextV2`. The worker therefore
has a weaker evidence contract than the parent turn. The later result contract
also flattens task status into summary/failure text: `ToolResultReadyV1` lacks
the continuation identity, task status, evidence state, objective, and
remaining needs required to make result wording source-bound. L3's current
resolver projection covers same-cycle observations, not the later
tool-result episode.

The resolver loop executes a background task-resolution request in the current
cycle and then allows the normal response route to continue. No typed boundary
currently rejects a same-goal pending resolver dependency combined with a
factual speak surface. That is how a parent answer and a later task result can
become two semantic owners.

### Non-cause

The evidence does not show duplicate scheduling. Existing accepted-task/job
idempotency and delivery state performed their intended duties.

## Canonical contracts

### Goal continuation reference

Add `GoalContinuationRefV1` to `src/kazusa_ai_chatbot/cognition_episode.py` as
the single cross-package contract:

```text
{
  "schema_version": "goal_continuation_ref.v1",
  "continuation_id": "goal-continuation:<sha256>",
  "source_episode_id": "<original episode id>",
  "source_message_id": "<original source message id>",
  "branch_id": "<selected bid branch>",
  "goal_ref": {
    "scope": "user | character",
    "kind": "<validated entity kind>",
    "entity_id": "<validated entity id>"
  }
}
```

The deterministic constructor hashes the canonical JSON of
`source_episode_id`, `branch_id`, and `goal_ref`; the model never authors the
identifier. The original source episode remains in the reference even when
the result gets a new `tool-result:<task_id>` episode. `source_message_id` is
lineage metadata and is copied unchanged; it is not used as the active-task
duplicate key.

The reference is required in the typed path through:

- `SelectedIntentionV2` and the normalized `CognitionCoreOutputV2`;
- `ResolverCapabilityRequestV2`, `ResolverCapabilityRequestV1`, and
  `RequiredResolverEvidenceDependencyV1`;
- materialized user-visible `speak` action specs and action results;
- `TaskResolutionExecutionContextV1`, `TaskResolutionCheckpointV1`, and
  `TaskResolutionResultV1`;
- `AcceptedTaskCreateRequest`/document and `BackgroundWorkQueueRequest`/job;
- `ToolResultReadyV1`, `ToolResultOriginV1`, the tool-result episode, and
  result-visible surface output.

Unrelated private actions carry an explicit null continuation reference. A
user-visible speak or task-resolution continuation must carry the validated
reference. Existing `task_identity_key` remains the active duplicate key and
is not replaced by the continuation identifier.

### Shared scene representation

Use the existing validated `SceneContextV2` as the sole shared representation.
It includes bounded `semantic_scene`, `public_group_scene`, participant
bindings, conversation continuity, and semantic temporal context. The same
validated object is projected into:

1. the action-planner human payload as `scene_context`;
2. `TaskResolutionExecutionContextV1.scene_context`;
3. `LocalContextResolverContextV1.scene_context` and its compact LLM payload.

The worker receives this bounded source-owned projection, not raw adapter
history or private job metadata. The public scene remains owned by
`persona_supervisor2.py` and is validated/consumed by
`persona_supervisor2_cognition.py`.

### Surface roles and result-state matrix

Add deterministic `surface_role` and the continuation reference to the
settled surface contract. The closed roles are:

| Surface role | Allowed evidence/state | Visible meaning |
|---|---|---|
| `factual_answer` | `answerable_now`, or inline evidence accepted by cognition; claims cite current/complete evidence | One direct answer for the continuation |
| `task_acknowledgement` | accepted background task, result state `pending`/`deferred` | Work was accepted and is pending; no missing-evidence facts |
| `task_result` | terminal task `resolved` or `partial`; evidence is complete or validated partial evidence | One factual result, with limitations for `partial` |
| `task_status` | `needs_user_input`, `approval_required`, `unavailable`, `failed`, or evidence state `missing`/`blocked` | Objective-scoped status or clarification; no factual answer claims |
| `ordinary` | no continuation reference and independently grounded distinct goal | Ordinary character speech remains available |

`pending` and `deferred` are never factual result material. `partial` may state
only validated evidence and its explicit remaining needs. A background
acknowledgement plus a later result is two visible surfaces but only one
factual result surface; the acceptance invariant counts those categories
separately.

## Implementation direction

### Work package A: contract plumbing

1. Add and validate `GoalContinuationRefV1`.
2. Thread the reference through V2/V1 cognition/resolver contracts, action
   materialization, task checkpoint/result, accepted-task/job, and tool-result
   episode contracts.
3. Add `surface_role` and continuation reference to `ActionSpecV1`,
   `ActionResultV1`, `SurfaceOutputV1`, and episode trace validation. The
   deterministic caller assigns the role; surface LLMs receive it as an
   authoritative semantic boundary and cannot change it.
4. Reject new task/job/result payloads missing the reference. Before cutover,
   inspect active task-resolution rows; rows that cannot receive a valid
   reference from stored lineage are terminalized as typed migration failures
   without adapter delivery. No task-id fallback reference is introduced.

### Work package B: shared evidence and planner selection

1. Keep `persona_supervisor2.py` as the scene-construction owner.
2. Project the validated `SceneContextV2` into `action_selection.py`'s bounded
   human payload and update its prompt contract to treat `public_group_scene`
   as authoritative public-scene evidence for the requested group-history
   objective.
3. Bind the continuation reference after deterministic primary-bid selection
   and before resolver/action materialization. The model still decides whether
   evidence is required.
4. Preserve the existing mutual exclusion between semantic action requests and
   resolver requests. Add deterministic validation that a resolver request,
   required-evidence state, and factual speak action cannot share the same
   continuation reference.

### Work package C: resolver one-surface arbitration

1. In `cognition_resolver/loop.py`, add a typed final-state validator after
   each cognition/capability cycle and before the state reaches the normal
   response route. It checks continuation references and surface roles rather
   than message text.
2. When cognition selects `answerable_now`, require no resolver request for the
   same reference; deterministic code validates the combination and lets the
   LLM-owned answerability decision stand.
3. When a background task is accepted, the current turn may expose one
   `task_acknowledgement` surface. The same reference cannot expose a factual
   speak surface in that episode. A mixed state receives bounded regeneration;
   after the attempt cap it fails closed as typed `task_status` with no task
   promotion and no factual answer.
4. When a tool-result episode arrives, its continuation reference is treated as
   the existing continuation. It may produce one `task_result` or `task_status`
   surface and cannot create a second task-resolution continuation for the same
   reference.
5. Distinct continuation references remain independently eligible for grounded
   character speech. This preserves character judgment rather than applying a
   global silence rule.

### Work package D: worker evidence parity and result delivery

1. Add `scene_context` and the continuation reference to the task execution
   context, checkpoint, and result. Pass the bounded scene to the local-context
   resolver contract and compact prompt projection.
2. Build `ToolResultReadyV1` from the validated stored
   `TaskResolutionResultV1`, preserving `semantic_objective`, terminal status,
   evidence state, evidence excerpts/refs, remaining needs, and continuation
   reference. Do not use only `result_summary`/`failure_summary` as the result
   authority.
3. Build the new tool-result episode with the original continuation reference
   in typed origin metadata and percept content. The result episode keeps its
   own episode ID while retaining the original lineage reference.
4. Project the typed result into cognition/L3 as a result-owned contract. L3
   wording rules are deterministic by state: resolved/partial facts come only
   from validated evidence; all other states produce objective-scoped status or
   clarification wording.
5. Keep `background_work/delivery.py` as the single normal delivery boundary.
   Its callback executes once per accepted task and receives the typed result
   episode; no worker or result-source code may call an adapter or dialog
   directly.

### Work package E: tests and live evidence

1. Add deterministic replay fixtures for the historical mixed state, including
   a parent factual-speech candidate plus a same-goal pending background
   resolver candidate. Assert fail-closed arbitration and zero task promotion.
2. Add direct, valid-background, partial-result, failed-result, and
   distinct-goal speech cases using typed contracts rather than patched LLM
   responses.
3. Split the current live harness into:
   - a historical planner-payload diagnostic that records whether the current
     model repeats the old route; and
   - a post-fix live acceptance case that accepts either direct resolution or
     the valid acknowledgement/result sequence and counts factual surfaces.
4. Run live LLM cases one at a time, inspect raw evidence, and write the
   Markdown review artifact before any subsequent live case.

## Mandatory skills

- `debug-llm` for every live run and human-readable evidence review.
- `test-style-and-execution` for deterministic/live test separation and
  execution order.
- `development-plan` for this approval and implementation boundary.
- `py-style` and `cjk-safety` for Python changes and CJK string handling.
- `local-llm-architecture` for semantic ownership and bounded pipeline design.
- `database-data-pull` and `llm-trace-debug` for any additional read-only
  database or protected-trace evidence.

## Execution roles

| Role | Responsibility / owned surface | Authority | Skills and capability floor | Independence | Acceptance gate |
|---|---|---|---|---|---|
| Contract owner | `cognition_episode.py`, V2/V1 contracts, action/surface result contracts | Deterministic validation owns schema; LLM owns semantic values | Python typing, canonical parser, CJK-safe edits; must understand cross-package exact-key validation | Reviews its own contract tests and receives resolver/surface review | Every new field has producer, consumer, validator, and direct test |
| Cognition owner | `action_selection.py`, `action_authorization.py`, `persona_supervisor2_cognition.py` | LLM owns answerability/goal; code binds refs and rejects mixed state | Prompt projection, V2 action materialization, bounded regeneration | Does not approve worker/delivery behavior | Planner sees `SceneContextV2`; same-goal factual/pending mix is rejected |
| Resolver/task owner | `cognition_resolver/loop.py`, `capabilities.py`, task-resolution contracts/state/service/orchestrator | Resolver owns observations; loop owns lifecycle arbitration; service owns durable task creation | Async lifecycle, typed observations, Mongo task boundary | Validates against independent surface/delivery tests | Direct path creates zero task; valid background path creates one task and one ack policy |
| Worker/result owner | local-context contracts/specialist, background models/jobs/worker/result source | Worker owns bounded execution result; result source owns typed result projection | Worker isolation, task result statuses, prompt-safe evidence | Does not call adapter/dialog and does not approve cognition route | Scene/ref parity and exact result-state matrix survive worker completion |
| Surface/delivery owner | L3 surface builder/stages, action result/trace, post-turn settlement | L3 owns wording; source contracts own facts/status; dispatcher owns delivery | Surface contract, dialog handoff, delivery receipt state | Verifies with resolver and worker owners' fixtures | At most one factual result surface per ref; dispatcher callback once |
| QA/review owner | deterministic replay, live LLM harness, artifacts, final plan evidence | Test contract owns observable acceptance | Live DB/LLM operation, raw evidence inspection, artifact review | Independent of production implementers; GPT-5.6 Sol review already completed | All traceability nodes pass; live artifact is inspected; no unresolved review finding |

## Exact source-to-test traceability

Every planned production source file has a direct deterministic test node. New
test node names are execution requirements.

| Planned source file | Direct deterministic test node | Required evidence |
|---|---|---|
| `src/kazusa_ai_chatbot/cognition_episode.py` | `tests/test_task_resolution_duplicate_delivery.py::test_goal_continuation_ref_survives_tool_result_episode` (new) | Stable ref survives new result episode and preserves source lineage |
| `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` | `tests/test_task_resolution_duplicate_delivery.py::test_mixed_pending_resolver_and_factual_surface_fails_contract` (new) | Exact contract rejects same-ref mixed state |
| `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py` | `tests/test_action_selection_payload.py::test_action_planning_payload_projects_authoritative_scene_context` (new) | Bounded `SceneContextV2` reaches planner; raw/private fields stay absent |
| `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py` | `tests/test_cognition_core_v2_action_authorization.py::test_same_goal_pending_resolver_cannot_derive_factual_speech_route` (new) | Route validation rejects same-ref factual/pending mix without semantic rewriting |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | `tests/test_task_resolution_duplicate_delivery.py::test_v2_coding_producer_persists_context_through_queue_to_worker` (new) | V2 accepted-coding producer carries the canonical continuation ref and typed execution context into the queue/worker path |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py` | `tests/test_action_spec_memory_lifecycle.py::test_memory_lifecycle_action_spec_declares_surface_metadata` (new) | Direct lifecycle action materialization declares ordinary/null surface metadata |
| `src/kazusa_ai_chatbot/cognition_resolver/contracts.py` | `tests/test_cognition_resolver_loop.py::test_resolver_request_and_dependency_preserve_goal_continuation_ref` (new) | V1 request, observation, and dependency retain exact ref |
| `src/kazusa_ai_chatbot/cognition_resolver/loop.py` | `tests/test_cognition_resolver_loop.py::test_pending_background_goal_reaches_acknowledgement_without_factual_surface` (new) | Resolver final-state gate prevents same-turn factual surface |
| `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py` | `tests/test_cognition_resolver_loop.py::test_task_resolution_context_preserves_scene_and_continuation_ref` (new) | Context builder carries exact scene/ref projection |
| `src/kazusa_ai_chatbot/task_resolution/contracts.py` | `tests/test_task_resolution_duplicate_delivery.py::test_task_context_requires_scene_and_continuation_ref` (new) | Missing scene/ref fails closed |
| `src/kazusa_ai_chatbot/task_resolution/state.py` | `tests/test_task_resolution_background_resume.py::test_task_checkpoint_and_result_preserve_continuation_ref` (new) | Resume checkpoint and terminal result preserve ref |
| `src/kazusa_ai_chatbot/task_resolution/orchestrator.py` | `tests/test_task_resolution_orchestrator.py::test_terminal_result_preserves_typed_status_and_evidence_state` (new) | Resolved/partial/non-success status is preserved without invention |
| `src/kazusa_ai_chatbot/task_resolution/service.py` | `tests/test_task_resolution_character_background_handoff.py::test_active_task_rejects_continuation_ref_mismatch` (new) | Active pending/enqueueing task reuse fails closed for refs A/B before enqueue or mutation |
| `src/kazusa_ai_chatbot/task_resolution/specialists/local_context.py` | `tests/test_task_resolution_character_background_handoff.py::test_local_context_specialist_receives_canonical_scene_context` (new) | Specialist receives bounded scene context and uses it as source input |
| `src/kazusa_ai_chatbot/local_context_resolver/contracts.py` | `tests/test_local_context_resolver_contracts.py::test_context_requires_bounded_scene_context` (new) | Resolver context exact contract requires validated scene |
| `src/kazusa_ai_chatbot/local_context_resolver/service.py` | `tests/test_local_context_resolver_standalone.py::test_prompt_payload_includes_bounded_scene_context` (new) | Compact prompt contains scene projection and excludes transport metadata |
| `src/kazusa_ai_chatbot/accepted_task/models.py` | `tests/test_accepted_task_lifecycle.py::test_task_document_requires_goal_continuation_ref` (new) | Accepted-task document contract carries ref and source episode |
| `src/kazusa_ai_chatbot/accepted_task/lifecycle.py` | `tests/test_accepted_task_lifecycle.py::test_create_or_return_active_persists_goal_continuation_ref` (new) | Creation preserves ref while duplicate identity remains scope/objective-based |
| `src/kazusa_ai_chatbot/background_work/models.py` | `tests/test_background_work_jobs.py::test_job_document_keeps_goal_continuation_ref` (new) | Queue/job shape includes ref and typed context |
| `src/kazusa_ai_chatbot/background_work/jobs.py` | `tests/test_background_work_jobs.py::test_enqueue_rejects_task_job_without_goal_continuation_ref` (new) | New task-resolution job missing ref fails closed |
| `src/kazusa_ai_chatbot/db/background_work_jobs.py` | `tests/test_background_work_jobs.py::test_insert_background_work_job_rejects_continuation_ref_mismatch` (new) | Duplicate idempotency key cannot return a job from a different continuation lineage |
| `src/kazusa_ai_chatbot/db/accepted_tasks.py` | `tests/test_accepted_task_lifecycle.py::test_repository_duplicate_ref_mismatch_preserves_provenance` (new) | Duplicate active task with a different continuation ref fails before provenance mutation |
| `src/kazusa_ai_chatbot/background_work/worker.py` | `tests/test_task_resolution_background_resume.py::test_completed_job_preserves_typed_result_ref` (new) | Worker terminalization stores ref/status/evidence state |
| `src/kazusa_ai_chatbot/background_work/result_source.py` | `tests/test_background_work_delivery.py::test_result_source_preserves_typed_task_status_and_ref` (new) | Result-ready episode carries objective, status, evidence, needs, and ref |
| `src/kazusa_ai_chatbot/background_work/subagent/future_speak.py` | `tests/test_background_work_future_speak.py::test_future_speak_action_spec_declares_surface_metadata` (new) | Direct future-speak action materialization declares ordinary/null surface metadata |
| `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py` | `tests/test_task_resolution_duplicate_delivery.py::test_v2_coding_producer_persists_context_through_queue_to_worker` (new) | Action-owned accepted-task and queue requests retain the exact ref and validated coding context |
| `src/kazusa_ai_chatbot/background_work/subagent/task_orchestrator.py` | `tests/test_task_resolution_background_resume.py::test_bound_coding_continuation_preserves_scene_and_ref` (new) | Bound coding continuation carries the validated scene/ref into its terminal result |
| `src/kazusa_ai_chatbot/cognition_core_v2/facade.py` | `tests/test_cognition_core_v2_contracts.py::test_v2_facade_returns_exact_one_scope_output` | Facade exposes the top-level continuation ref equal to the intention ref |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py` | `tests/test_task_resolution_duplicate_delivery.py::test_v2_coding_producer_persists_context_through_queue_to_worker` (new) | Action materialization binds coding acknowledgement role, ref, and typed context |
| `src/kazusa_ai_chatbot/task_resolution/specialists/coding.py` | `tests/test_task_resolution_background_resume.py::test_bound_coding_continuation_preserves_scene_and_ref` (new) | Coding specialist emits validated evidence only for the resolved result and preserves context |
| `src/kazusa_ai_chatbot/action_spec/models.py` | `tests/test_action_spec_models.py::test_user_visible_speak_requires_goal_continuation_ref` (new) | User-visible speak validates required ref; private unrelated action remains null |
| `src/kazusa_ai_chatbot/action_spec/execution.py` | `tests/test_action_spec_results.py::test_action_execution_rejects_malformed_spec_without_crashing` | Malformed rejected action stays prompt-safe and cannot crash trace execution |
| `src/kazusa_ai_chatbot/action_spec/results.py` | `tests/test_action_spec_results.py::test_action_result_rejects_missing_surface_metadata` (new) | Result materialization fails closed instead of silently defaulting absent metadata |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` | `tests/test_l2d_l3_surface_handoff.py::test_tool_result_surface_role_is_typed` (new) | L3 receives result-owned role and cannot drop status/ref |
| `src/kazusa_ai_chatbot/cognition_core_v2/surface.py` | `tests/test_l2d_l3_surface_handoff.py::test_pending_task_result_cannot_be_rendered_as_factual_answer` (new) | Pending/missing result is not presented as factual evidence |
| `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py` | `tests/test_l2d_l3_surface_handoff.py::test_result_status_wording_matrix_is_exact` (new) | Resolved/partial/status states map to only allowed wording requirements |
| `src/kazusa_ai_chatbot/brain_service/post_turn.py` | `tests/test_action_spec_results.py::test_settled_trace_preserves_continuation_surface_role` (new) | Episode trace keeps role/ref through settlement and fallback surface creation |
| `src/kazusa_ai_chatbot/self_cognition/runner.py` | `tests/test_self_cognition_tracking.py::test_v2_scheduled_speech_materializes_speak_action_spec` | Direct due-speech action materialization declares ordinary/null surface metadata |

Required neighboring tests remain green:

- `tests/test_cognition_core_v2_action_planning_bugfix.py::test_answerable_now_drops_optional_resolver_request`
- `tests/test_cognition_core_v2_action_planning_bugfix.py::test_task_resolution_boolean_survives_authorization_and_materialization`
- `tests/test_cognition_resolver_loop.py::test_answerable_now_terminates_without_executing_optional_resolver`
- `tests/test_cognition_resolver_loop.py::test_public_evidence_projects_through_task_resolution`
- `tests/test_cognition_resolver_loop.py::test_task_resolution_bounds_history_to_its_context_contract`
- `tests/test_l2d_l3_surface_handoff.py::test_l3_builder_projects_latest_resolver_result_as_separate_authority`
- `tests/test_task_resolution_character_background_handoff.py::test_true_direct_background_skips_inline_and_accepts_after_queue`
- `tests/test_task_resolution_character_background_handoff.py::test_empty_evidence_deferred_handoff_invents_no_partial_content`
- `tests/test_background_work_delivery.py::test_service_result_ready_delivery_uses_dispatcher_boundary`

Live diagnostic/acceptance nodes:

- `tests/test_task_resolution_duplicate_delivery_live_llm.py::test_live_replays_historical_pending_task_action_plan`: diagnostic only; it records whether the current real model repeats the historical route.
- `tests/test_task_resolution_duplicate_delivery_live_llm.py::test_live_task_resolution_duplicate_delivery_regression`: post-fix acceptance; it accepts direct or valid background routing, then asserts exactly one factual result surface and at most one acknowledgement surface for the continuation reference.

Delegation and closure policy:

- DeepSeek Flash owns every production-code edit in this plan.
- The parent owns tests, plan/artifact text, read-only inspection, live/database
  operations, and final integration.
- The native plan reviewer performs read-only implementation review.
- GPT-5.6 Sol performs the final read-only sign-off at high reasoning and
  normal speed.

## Change surface

### Modify

- The production files in the source-to-test table.
- Existing deterministic tests named as neighboring tests where their exact
  contract fixtures gain the new required fields.
- The live diagnostic harness to separate historical observation from post-fix
  acceptance.

### Create

- The new deterministic test module and named regression nodes.
- The reviewed live artifact and any per-run raw artifact required by
  `debug-llm`.
- No compatibility module or parallel production protocol.

### Delete

- No source deletion is planned.

### Keep

- `background_work/delivery.py` as the single normal brain delivery boundary.
- Accepted-task/job idempotency, lease, retry, and terminal state machines.
- Worker isolation from adapters and direct dialog calls.
- Canonical JSON parsing, bounded repair/regeneration, and fail-closed
  contract handling.
- LLM ownership of character judgment, answerability, and wording.

## Verification and rollout gates

1. Capture `git status --short` and preserve all unrelated existing changes.
2. Read the owned source/test files and update the plan's traceability table if
   implementation scope adds a source file.
3. Apply the listed skills before the corresponding Python/test/database/live
   work.
4. Run syntax/style checks for every changed Python file.
5. Run all deterministic traceability nodes in focused batches, then the
   neighboring tests. Inspect each failure before widening the batch.
6. Run the deterministic historical mixed-state replay. It must fail closed
   before task promotion and produce no factual surface.
7. Run the real-LLM historical planner diagnostic once, inspect raw output, and
   record whether the current model repeats the old route.
8. Run the post-fix group-history live case once with real LLM and live DB,
   inspect the Markdown artifact, and verify either:
   - direct path: zero accepted task rows and one factual surface; or
   - valid background path: one accepted task/job, at most one acknowledgement,
     and exactly one later factual/status result surface.
9. Verify the live database by accepted-task identity, job idempotency key,
   worker count, delivery attempts, continuation reference, and persisted
   surface roles. A second factual surface for the same ref fails the gate.
10. Before cutover, inventory active task-resolution rows. Rows missing the
    required ref are terminalized with an auditable migration failure and no
    delivery; new rows have no compatibility fallback.
11. Implementation begins only after the user explicitly commands the
    production-code change; that command was received before the recorded
    handoffs.

### Execution evidence (2026-08-14)

- Production implementation was delegated to DeepSeek Flash with bounded
  30-minute handoffs and exact owned-file snapshots. The parent made the
  test-only and evidence-only changes listed in this document.
- Focused deterministic traceability batches collected and passed: `88`,
  `140`, `105`, and `9` tests; the focused lineage runs passed `35` and `38`
  tests, and the contract-remediation focus passed `85` tests. The six-state
  typed-result matrix covers `resolved`, `partial`,
  `needs_user_input`, `approval_required`, `unavailable`, and `failed` across
  result source, cognition evidence, and L3 status projection.
- The second review findings on idempotent continuation reuse and repository
  provenance mutation were remediated by DeepSeek Flash. Parent-owned A/B
  regressions cover `pending` and `enqueueing` active-task states, duplicate-job
  insertion, and accepted-task repository matching/mismatch behavior; the
  focused lineage invocations passed `35` and `38` tests.
- A subsequent native review identified three active action-spec producers
  outside the initial matrix and a result-metadata fallback. DeepSeek Flash
  added explicit ordinary/null metadata to future-speak, memory-lifecycle,
  and self-cognition producers, made result metadata validation fail closed,
  and then added the malformed-result boundary in the trace executor. Direct
  producer regressions and rejected-result coverage are included in the
  `85`-test contract focus.
- The final DeepSeek Flash remediation handoff
  `019ffee3-be86-7821-95fc-d945c9330191` repaired the previously unexercised
  accepted-coding producer path: V2 now binds the canonical non-null
  continuation reference and a validated `TaskResolutionExecutionContextV1`,
  the handler persists both into the task-orchestrator queue request, and the
  worker receives the same context. Missing semantic surface metadata now
  fails closed. The parent-owned end-to-end regression
  `test_v2_coding_producer_persists_context_through_queue_to_worker` passed.
- The follow-up DeepSeek Flash handoff
  `019fff00-ab5b-7221-ad31-2d98d5cac566` closed the native review's remaining
  accepted-coding handler gap: the exact acknowledgement role, non-null
  continuation reference, complete execution context, and action/context
  reference equality are validated before accepted-task persistence. Parent
  regressions cover wrong surface role and mismatched reference, with the
  duplicate-delivery lineage focus now passing `38` tests.
- Syntax compilation passed with `venv\Scripts\python -m compileall -q
  src/kazusa_ai_chatbot tests`; `git diff --check` passed with only the
  repository's existing CRLF conversion warnings.
- Historical real-LLM planner diagnostic passed as an observation and recorded
  no repeat of the historical route. The parent-authored review is
  `test_artifacts/diagnostics/task_resolution_duplicate_delivery_live_review_historical_planner_diagnostic.md`.
- Post-fix real-LLM acceptance passed with zero accepted task rows and exactly
  one `factual_answer` surface. The parent-authored review is
  `test_artifacts/diagnostics/task_resolution_duplicate_delivery_live_review_post_fix_acceptance.md`.
- Pre-cutover migration terminalized the four active missing-reference rows;
  the auditable record is
  `test_artifacts/diagnostics/task_resolution_duplicate_delivery_migration.md`.
- Delegation and deterministic collection evidence is preserved in
  `test_artifacts/diagnostics/task_resolution_duplicate_delivery_execution_handoffs.md`
  and `test_artifacts/diagnostics/task_resolution_duplicate_delivery_deterministic_verification.md`.
- The complete plan-level DeepSeek Flash production ownership manifest, with
  immutable baseline references, final hashes, and the verification checkpoint,
  is `test_artifacts/diagnostics/task_resolution_duplicate_delivery_production_ownership_manifest.md`.
- The final native read-only review
  (`019fff09-2b88-7283-9577-0eea2fc971c8`) returned `READY`, and the final
  GPT-5.6 Sol read-only sign-off
  (`019fff0d-a793-7960-bbe1-a15b942e04cb`) returned `READY`; no closure gates
  remain.

## Acceptance criteria

- The historical failure is classified as semantic duplicate visible delivery,
  not duplicate queue insertion.
- The exact validated `SceneContextV2` reaches parent planner and worker local
  context through one bounded contract.
- One deterministic `GoalContinuationRefV1` correlates original goal, task,
  action/surface records, and new result episode.
- Direct grounded speech remains available and creates zero task rows.
- Same-reference pending/background resolution cannot coexist with a factual
  parent surface.
- Distinct continuation references can still produce independently grounded
  character speech.
- Background acknowledgement contains no missing-evidence factual claims.
- Resolved and partial results produce one evidence-bounded factual result;
  partial results retain explicit limitations.
- Pending, missing, blocked, needs-user-input, approval-required, failed, and
  unavailable states produce only the exact typed acknowledgement/status policy
  and cannot become unrelated free-form dialog.
- Worker delivery invokes the normal brain/dispatcher boundary once and never
  calls an adapter or dialog directly.
- Existing idempotency, lease, retry, and worker isolation tests remain green.
- All source-to-test rows have passing deterministic evidence, the live
  artifact is reviewed, and all GPT-5.6 Sol findings are incorporated.

## Review disposition

The GPT-5.6 Sol review was initially `not ready`; all findings were incorporated
through the DeepSeek Flash production handoffs and parent-owned verification.
The final native reviewer and final GPT-5.6 Sol sign-off both returned `READY`.
The implementation, deterministic/live evidence, ownership manifest, and
closure record are complete. Final status is `completed`; this plan is moved
to `development_plans/archive/completed/` as a historical execution record.

## Open questions

None. The plan chooses the existing validated `SceneContextV2`, the new typed
goal-continuation reference, the resolver capability/service enqueue boundary,
and the normal brain delivery path as the canonical ownership model.
