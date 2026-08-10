# Self-cognition group visible-reply capability contract bugfix

Date: 2026-08-11

## Summary

- Goal: restore the self-cognition group path's ability to reach the existing
  L3 speech surface while preserving deliberate silence and deterministic
  delivery controls.
- Status: completed.
- Scope boundary: Cognition V2's self-cognition response contract,
  self-cognition route/gate tracking, worker observability, documentation, and
  deterministic/live regression coverage.
- Change direction: keep `think_only` as the default starting posture, add an
  explicit semantic `stay_silent | propose_visible_reply` decision for
  targetless group self-cognition, and let deterministic policy approve or
  reject a proposal before dialog and dispatch.
- Acceptance state: the first `gpt-5.6-sol` high-reasoning review returned
  `forward`; the independent plan review returned `needs amendment, then
  forward`; the amendments below are incorporated, and the user agreed that
  the structural speech-capability defect must be fixed before any broader
  future self-cognition improvement.
- Execution state: implementation, parent verification, Terra code review,
  and Sol final review are complete; no unresolved blocker remains.

## Confirmed Decisions

1. Speech remains an always-available downstream ability. The existing
   `speak` capability stays in the action registry and the L3 materialization
   path remains the owner of final visible wording.
2. `think_only` remains the default starting posture for group
   self-cognition. It is not a permanent prohibition on a visible reply.
3. Cognition owns the semantic choice between `stay_silent` and
   `propose_visible_reply`, grounded in current evidence and character reason.
4. Cognition owns the semantic disposition; deterministic policy owns
   structured eligibility, provenance, freshness, target binding, duplicate
   reservation, permissions, persistence, and dispatch; dialog owns final
   wording; the dispatcher owns execution reporting.
5. The original investigated run is classified as a semantic contract failure
   under the new contract because it produced no dedicated self-cognition
   response proposal. A replayed proposal with no structured participation
   grounding is policy-rejected with `unresolved_target`. A separate
   `stay_silent` fixture produces `cognition_declined`; none of these outcomes
   uses an implicit missing-speech path.
6. Commitment and scheduled-future cognition keep their existing
   `scheduled_tick` route and are regression-protected by this plan.
7. No keyword router, deterministic message interpretation, broad
   `self_cognition -> visible_reply` mapping, generic unrestricted `speak`
   action request, compatibility shim, or adapter bypass is introduced.

## Independent Plan Review

The requested independent `gpt-5.6-sol` high-reasoning reviewer returned
**needs amendment, then forward**.

The reviewer verified that the supplied database and trace evidence establishes
the failure for this ordinary self-cognition planning path: `speak` was absent
from the model-facing affordances, no dedicated reply proposal existed, empty
action/resolver requests under `think_only` became silence, and the runner's
dialog branch was never entered. The evidence does not establish that every
self-cognition route can never speak; the plan therefore describes the
ordinary path's inability to represent a visible-reply proposal rather than a
universal impossibility claim.

The reviewer confirmed the ownership direction:

- Cognition owns `stay_silent | propose_visible_reply` and its evidence,
  target, and response-goal semantics.
- Deterministic policy owns structured eligibility and operational controls.
- L3/dialog owns final wording.
- The dispatcher owns execution and delivery reporting.
- Commitments remain on `scheduled_tick`.

The reviewer required removal of the unconditional high-risk/ambient veto,
structured target and participation grounding, separate semantic/policy/
execution dispositions, and an atomic duplicate reservation before dialog.
Those amendments are incorporated below.

Review evidence is preserved in
`test_artifacts/diagnostics/latest_self_cognition_dialog_review.md` and the
protected trace export
`test_artifacts/diagnostics/latest_self_cognition_full_trace.json`.

## Failure Mode And Evidence

The investigated run was:

- run: `self_cognition_run:self_cognition_trigger:0ca809b5fa63b66d8fe6c6ee`;
- trigger: `self_cognition_trigger:0ca809b5fa63b66d8fe6c6ee`;
- source: QQ group review window in channel `638473184`;
- cognition result: a high-confidence playful intention with evidence handle
  `e1`;
- action planning result: `action_requests=[]`,
  `resolver_requests=[]`, `goal_resolution=answerable_now`;
- run result: `selected_route=audit_only`, `output_mode=silent`,
  `dialog_calls=0`, `dispatch_status=not_requested`;
- protected trace result: `final_dialog_count=0`, no delivery id, no linked
  assistant conversation row, and no dialog stage;
- model-facing action roster: `accepted_task_status_check` was present and
  `speak` was absent.

The source labels were `assistant_presence=not_in_window`,
`bot_addressing=ambient_group_context`, `response_risk=high`, and
`message_recency=recent`. The source rows were a correction and joke exchange
between other participants. Silence was behaviorally defensible, but the
ordinary self-cognition planning path could not represent a dedicated
visible-reply proposal, and the resulting silence was indistinguishable from
that structural contract failure. This evidence does not claim that every
alternative self-cognition route is unable to speak.

The current ownership break is visible in:

- [`action_authorization.py`](../../../src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py), where `think_only` with no requests derives `silence`;
- [`persona_supervisor2_cognition.py`](../../../src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py), where the generic action-affordance projection skips `SPEAK_CAPABILITY`; and
- [`runner.py`](../../../src/kazusa_ai_chatbot/self_cognition/runner.py), where dialog is called only after the action-candidate route has been selected.

## Scope And Change Direction

The required first workstream is the structural capability repair:

```text
group review source
  -> Cognition V2 goal and action planning
  -> explicit stay_silent | propose_visible_reply outcome
  -> deterministic self-cognition response gates
       -> semantic disposition
       -> policy disposition and structured reason
       -> execution disposition
  -> existing L3 text surface and dialog
  -> existing self-cognition dispatcher and adapter boundary
```

The current `think_only` default remains in force until a semantic proposal is
validated and approved. An approved group proposal derives the existing
`speech` intention, materializes the existing L3-owned `speak` action surface,
and then follows the existing self-cognition action-attempt and dispatcher
path. A declined, rejected, or contract-failed proposal never invokes dialog
or dispatch, but each semantic, policy, and execution disposition is preserved
in bounded tracking and event-log metadata.

The plan changes the model-facing semantic contract without treating `speak`
as a generic executable action. The action registry continues to contain
`speak`; the explicit response decision is the self-cognition-specific bridge
that allows the L3 surface owner to be considered.

## Contracts And Data Shapes

### Semantic response decision

For targetless group self-cognition, the action-planning result carries an
additional exact object named `self_cognition_response`:

```json
{
  "decision": "stay_silent | propose_visible_reply",
  "evidence_handles": ["e1"],
  "semantic_target_handle": "current_group_scene",
  "participation_basis": "grounded_scene_intervention",
  "response_goal": "bounded semantic response goal",
  "reason": "bounded semantic reason"
}
```

The contract is fixed as follows:

- `decision` is exactly `stay_silent` or `propose_visible_reply`.
- `evidence_handles` is an array of zero to four supplied evidence handles.
- `propose_visible_reply` requires at least one current-episode evidence
  handle; `stay_silent` may use an empty array when no evidence supports an
  intervention.
- `propose_visible_reply` additionally requires a non-empty
  `semantic_target_handle` from the prompt-provided `self`,
  `current_group_scene`, or participant-role handles; a participant-targeted
  proposal must use that participant's supplied role handle.
- `propose_visible_reply` additionally requires
  `participation_basis` from the closed values `direct_address`,
  `explicit_character_reference`, or `grounded_scene_intervention`.
- `propose_visible_reply` additionally requires a non-empty `response_goal`
  capped at 300 characters.
- `reason` is a non-empty model-authored semantic explanation capped at 300
  characters for both decisions.
- The object contains no route, permission, target platform id, adapter data,
  dispatch instruction, or final dialog text.
- The object is required for a targetless group self-cognition episode with an
  admitted bid. A missing object after bounded repair is a structural contract
  failure. A group episode with no admitted bid receives deterministic
  `cognition_declined` semantic disposition with `no_admitted_bid` reason.
- Non-group and scheduled commitment episodes preserve their existing output
  contract. They do not receive a generic compatibility alias for this field.

### Three-dimensional response outcome

Self-cognition tracking records three independent closed dimensions:

- `semantic_disposition`: `cognition_declined`, `reply_proposed`, or
  `cognition_contract_failed`;
- `policy_disposition`: `not_evaluated`, `approved`, or `rejected`;
- `execution_disposition`: `not_requested`, `dialog_failed`,
  `dispatch_failed`, or `delivered`.

`policy_reason` is a closed bounded value: empty when not applicable, or one
of `stale_source`, `invalid_provenance`, `unresolved_target`,
`permission_denied`, `duplicate`, `cooldown`, or `policy_risk`.
`response_gate_codes` is a capped list of sanitized gate names. Adapter
unavailability and dialog/dispatch failures remain execution evidence rather
than semantic cognition outcomes.

The existing `selected_route` remains `audit_only` until policy approval and
`action_candidate` after approval. The run `output_mode` is `silent` until
approval and `visible_reply_candidate` after approval. Scheduled commitment
cases retain `scheduled_action_request`.

### Deterministic group gates

The policy evaluator applies these gates in order and records a closed gate
code for each material decision:

1. `response_contract`: validate the semantic response, evidence handles,
   semantic target, participation basis, and response goal.
2. `group_source_provenance`: require a `group_chat_review` source, a
   targetless group semantic scope, and a source id matching the group review
   window.
3. `recent_source`: require the typed source label
   `message_recency=recent`.
4. `participation_grounding`: validate the structured relation between
   `participation_basis`, `semantic_target_handle`, and typed source labels:
   `direct_address` requires `bot_addressing=directly_addressed`,
   `grounded_scene_intervention` requires the character to be present in the
   window or directly addressed, and `explicit_character_reference` requires
   a self/participant target plus current-episode evidence. Message-body
   interpretation remains Cognition-owned.
5. `bound_group_target`: require the already-bound delivery target to be the
   same concrete group channel; no participant becomes a delivery target.
6. `duplicate_reservation`: atomically reserve the existing source-window
   idempotency key before dialog. A reservation conflict produces policy
   `duplicate` and never enters dialog.
7. `approved_for_dialog`: permit the existing L3/dialog path only after the
   preceding gates pass.

`response_risk`, `bot_addressing`, and `assistant_presence` remain structured
policy context. No high-risk/ambient label combination is an unconditional
speech veto. Adapter availability, channel permission, write-ahead
persistence, delivery receipts, cancellation, and final dispatch remain owned
by the existing dispatcher/runtime coordination boundary. Their outcomes
remain in the execution disposition and self-cognition response trace.

## Must Do

- Add and validate the targetless group self-cognition response decision.
- Make the self-cognition `think_only` route consult that semantic decision so
  `propose_visible_reply` can reach the existing speech surface and
  `stay_silent` remains silent.
- Preserve `speak` in the canonical action registry and materialize it only
  through the existing L3-owned surface path.
- Add the deterministic group response-gate evaluator and the three
  semantic/policy/execution disposition dimensions above.
- Ensure semantic decline, policy rejection, and contract failure skip
  dialog, action-candidate creation, and dispatcher handoff while still
  recording an auditable route effect.
- Preserve existing target binding, duplicate suppression, cancellation,
  adapter capability, write-ahead persistence, delivery receipts, and
  consolidation behavior.
- Add bounded semantic, policy, execution, policy-reason, and gate-code fields
  to self-cognition tracking/event metadata without storing source text,
  candidate text, prompts, raw model output, or target ids.
- Add an atomic source-window action-attempt reservation before dialog and
  preserve the existing unique idempotency index; reservation conflicts close
  the proposal as a duplicate policy outcome.
- Update the Cognition V2, self-cognition, and event-logging ICDs and the
  architecture tests that enforce them.
- Run one individually inspected latest-case-shaped ambient review and one
  individually inspected structurally grounded group case after deterministic
  tests pass.
- Preserve the current production database diagnostic artifacts as evidence;
  do not perform a database migration.

## Deferred

- Adaptive social-risk thresholds or learned cooldowns.
- Any policy that treats response risk alone as a speech veto.
- New self-cognition trigger types or changes to commitment scheduling.
- Automatic proactive contact outside a source-bound group review.
- Generic `speak` action selection for arbitrary background workers.
- New keyword, URL, topic, or post-LLM classifiers.
- Dialog prompt or character-voice redesign.
- Adapter protocol changes or alternate delivery paths.
- Any broader future-improvement plan until this structural contract is
  implemented and its acceptance evidence is complete.

## Mandatory Skills

- `development-plan`: lifecycle, traceability, execution gates, and review.
- `local-llm-architecture`: preserve LLM semantic ownership and bounded stage
  contracts.
- `no-prepost-user-input`: keep the speech decision LLM-first and keep
  deterministic code on validation, policy, and delivery mechanics.
- `py-style`: required for every Python source or test change.
- `cjk-safety`: required for Python prompts or fixtures containing CJK text.
- `test-style-and-execution`: deterministic/live test taxonomy and execution.
- `debug-llm`: live output inspection and human-readable quality artifacts.
- `llm-trace-debug`: protected trace evidence and final-dialog boundary review.

## Mandatory Rules

- Execute only after this plan moves to `approved` or `in_progress` and the
  user explicitly authorizes implementation.
- Use `venv\Scripts\python` for Python commands and `apply_patch` for manual
  edits.
- Run every live LLM case one at a time and inspect its artifact before the
  next case.
- Preserve the existing model routes, bounded retry limits, prompt caps,
  dialog call cap, source-window cap, and dispatcher ownership.
- Pass raw model output through the canonical
  `parse_llm_json_output(...)` path and keep malformed semantic output out of
  route, persistence, dialog, and delivery.
- Keep source packet text and candidate dialog out of event-log payloads.
- Keep platform/database/delivery identity outside model-facing response
  decisions.
- Do not alter unrelated pre-existing worktree changes.

## Target State

The targetless group path has these observable outcomes:

| Semantic result | Deterministic disposition | Dialog | Dispatch |
|---|---|---:|---:|
| `stay_silent` | semantic `cognition_declined`; policy `not_evaluated`; execution `not_requested` | 0 | not requested |
| valid proposal without structured participation grounding | semantic `reply_proposed`; policy `rejected/unresolved_target`; execution `not_requested` | 0 | not requested |
| missing/invalid proposal after bounded repair | semantic `cognition_contract_failed`; policy `not_evaluated`; execution `not_requested` | 0 | not requested |
| valid proposal, all structured gates and reservation pass | semantic `reply_proposed`; policy `approved`; execution `delivered` or typed downstream failure | 1 maximum | existing dispatcher path |

For the approved branch, the data flow is:

```text
self_cognition_response.propose_visible_reply
  -> Cognition V2 intention.route = speech
  -> existing L3 speak surface materialization
  -> self-cognition action candidate
  -> dialog once
  -> dispatcher capability and permission checks
  -> write-ahead conversation row and adapter delivery
```

Commitment flow remains:

```text
scheduled_tick -> existing scheduled_action_request -> existing speak/action path
```

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`: define the bounded
  self-cognition response decision and carry it through the validated Core V2
  output contract.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`: extend the
  self-cognition action-planning prompt, parse/validate the exact response
  object, and pass its decision to route derivation. Preserve the existing
  action-request contract and keep generic speech out of executable action
  requests.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py`: make
  `think_only` consult the validated group response decision; derive
  `speech` only for `propose_visible_reply` and derive `silence` for
  `stay_silent` or absent/invalid proposals.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`: carry the validated
  response decision and structural-failure state into the Core V2 output.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: project the
  response decision, preserve the canonical `speak` registry/materializer,
  and ensure approved semantic speech reaches the existing L3 surface.
- `src/kazusa_ai_chatbot/self_cognition/models.py`: add the response
  disposition, output-mode, and bounded gate-code constants/types.
- `src/kazusa_ai_chatbot/self_cognition/tracking.py`: add the deterministic
  group response-gate evaluator, route-effect metadata, and explicit
  semantic/policy/execution dispositions while preserving existing commitment
  behavior and implementing the pre-dialog duplicate reservation.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`: evaluate the group
  response before action-attempt/dialog creation, atomically reserve the
  source-window action identity before dialog, skip dialog on non-eligible
  outcomes, and carry gate evidence into run artifacts.
- `src/kazusa_ai_chatbot/self_cognition/worker.py`: mirror bounded response
  semantic/policy/execution dispositions and gate codes into the existing
  sanitized self-cognition event recorder.
- `src/kazusa_ai_chatbot/db/self_cognition.py`: add the deterministic
  conditional action-attempt reservation operation over the existing unique
  idempotency key; preserve old row readability and avoid a migration.
- `src/kazusa_ai_chatbot/db/README.md`: document reservation ownership and
  duplicate semantics for the existing self-cognition action-attempt
  collection.
- `src/kazusa_ai_chatbot/event_logging/__init__.py` and its ICD/implementation
  modules: add only the bounded semantic/policy/execution disposition,
  policy-reason, and closed gate-code fields to the existing self-cognition
  event family.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`,
  `src/kazusa_ai_chatbot/self_cognition/README.md`,
  `src/kazusa_ai_chatbot/event_logging/README.md`: document the ownership,
  contract, dispositions, and no-raw-content telemetry boundary.
- `development_plans/README.md`: register this draft under active bugfix
  plans.

### Create

- `tests/test_self_cognition_group_visible_reply_boundary.py`: deterministic
  contract, route, gate, runner, commitment, and event metadata coverage with
  the exact nodes listed below.
- `tests/test_self_cognition_group_visible_reply_live_llm.py`: individually
  run and inspected ambient/high-risk and eligible group fixtures.
- `test_artifacts/diagnostics/self_cognition_group_visible_reply_review.md`:
  parent-authored debug-LLM quality review built from the two raw live
  artifacts. This remains diagnostic evidence, not runtime control state.

### Keep

- `src/kazusa_ai_chatbot/action_spec/registry.py` and the existing L3
  `speak` materializer as the canonical visible-surface owner.
- `src/kazusa_ai_chatbot/dispatcher/` as the only self-cognition delivery
  boundary.
- `src/kazusa_ai_chatbot/calendar_scheduler/` and commitment trigger
  collectors unchanged.
- Existing group-review source collection, participant context, target
  binding, group-review ledger, action-attempt idempotency, and consolidation
  contracts.

### Delete

- No files or persisted collections are deleted.

## Agent Autonomy Boundaries

The implementation owner may choose local helper names, function factoring,
command order, and test fixture construction within the listed files. The
implementation owner must preserve the exact semantic decision values,
dispositions, gate order, ownership boundaries, route behavior, and exclusions
specified here.

The implementation owner must request a plan amendment before changing any
of the following: commitment routes, ordinary user-response routing, adapter
contracts, persisted MongoDB shapes, model retry/cap budgets, generic action
registry semantics, or the structured participation-gate values and
semantics.

## Test Impact And Traceability

Every changed semantic owner has a deterministic unit node. The new boundary
file is part of the fixed test contract and must be created with these exact
node names:

| Source path | Changed symbol/contract | Semantic owner | Deterministic pytest node IDs | Supplemental nodes | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` | `SelfCognitionResponseDecisionV1` and Core V2 output validation | Cognition contract validator | `tests/test_self_cognition_group_visible_reply_boundary.py::test_group_self_cognition_response_contract_requires_exact_decision_shape`; `tests/test_self_cognition_group_visible_reply_boundary.py::test_group_self_cognition_response_proposal_requires_current_episode_evidence`; `tests/test_self_cognition_group_visible_reply_boundary.py::test_group_self_cognition_response_proposal_requires_target_and_goal` | `none` | deterministic unit | malformed, ungrounded, or targetless semantic speech proposals entering routing |
| `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py` | self-cognition action-planning response field and bounded repair | action-planning semantic owner | `tests/test_self_cognition_group_visible_reply_boundary.py::test_group_action_planning_requires_explicit_silence_or_reply_decision` | `tests/test_self_cognition_group_visible_reply_live_llm.py::test_live_group_self_cognition_latest_case_stays_silent`; `tests/test_self_cognition_group_visible_reply_live_llm.py::test_live_group_self_cognition_eligible_case_can_propose_reply` | deterministic unit plus one-at-a-time live LLM | implicit silence caused by an empty action request and missing speech decision |
| `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py` | `derive_action_route` for targetless group `think_only` | deterministic route owner | `tests/unit/cognition_core_v2/test_action_authorization.py::test_group_self_cognition_response_decision_derives_speech_or_silence` | `tests/test_cognition_core_v2_action_planning_bugfix.py::test_scheduled_tick_route_remains_scheduled_action_request` | deterministic unit | `think_only` hard-suppressing a validated visible-reply proposal |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | response projection and L3 `speak` materialization | Cognition-to-surface connector | `tests/unit/nodes/test_persona_supervisor2_cognition.py::test_group_self_cognition_proposal_materializes_existing_speak_surface` | `tests/test_action_spec_evaluator.py::test_evaluator_accepts_speak_surface_action_without_dispatcher_bridge` | deterministic unit | semantic speech proposal being lost before L3 |
| `src/kazusa_ai_chatbot/self_cognition/models.py` | semantic/policy/execution dispositions, policy reason, and gate-code contract | self-cognition tracking contract | `tests/test_self_cognition_group_visible_reply_boundary.py::test_response_outcome_values_are_closed` | `tests/test_self_cognition_architecture_docs.py::test_canonical_self_cognition_readme_documents_outcome_dimensions` | deterministic/static | ambiguous audit-only records that hide why no dialog occurred |
| `src/kazusa_ai_chatbot/self_cognition/tracking.py` | group response policy evaluator and route projection | self-cognition deterministic policy | `tests/test_self_cognition_group_visible_reply_boundary.py::test_group_proposal_requires_structured_participation_grounding`; `tests/test_self_cognition_group_visible_reply_boundary.py::test_high_risk_label_alone_does_not_reject_grounded_proposal`; `tests/test_self_cognition_group_visible_reply_boundary.py::test_recent_grounded_group_proposal_is_eligible`; `tests/test_self_cognition_group_visible_reply_boundary.py::test_duplicate_group_window_remains_suppressed` | `tests/test_self_cognition_tracking.py::test_classify_route_projects_v2_scheduled_speech_to_action_candidate` | deterministic unit | targetless proposal delivery, mechanical risk suppression, duplicate delivery, or commitment regression |
| `src/kazusa_ai_chatbot/self_cognition/runner.py` | pre-dialog gate, atomic reservation, and action/dialog sequencing | self-cognition episode runner | `tests/test_self_cognition_group_visible_reply_boundary.py::test_policy_rejection_skips_action_attempt_dialog_and_dispatch`; `tests/test_self_cognition_group_visible_reply_boundary.py::test_eligible_group_proposal_calls_dialog_once`; `tests/test_self_cognition_group_visible_reply_boundary.py::test_dialog_runs_only_after_atomic_duplicate_reservation` | `tests/test_self_cognition_integration.py::test_worker_no_speak_does_not_dispatch`; `tests/test_self_cognition_integration.py::test_worker_selected_speak_dispatches_to_bound_group_source_channel` | deterministic unit plus integration | dialog being skipped for eligible speech, called for rejected speech, or duplicated before reservation |
| `src/kazusa_ai_chatbot/self_cognition/worker.py` | sanitized event projection of semantic/policy/execution outcomes | self-cognition worker observability | `tests/test_self_cognition_group_visible_reply_boundary.py::test_worker_event_contains_three_dispositions_and_gate_codes_without_raw_content` | `tests/test_self_cognition_event_logging.py::test_self_cognition_event_logger_sanitizes_consolidation_outcome` | deterministic unit | production evidence losing the distinction between silence, policy rejection, contract failure, and delivery failure |
| `src/kazusa_ai_chatbot/db/self_cognition.py` | conditional source-window action-attempt reservation | deterministic persistence owner | `tests/test_self_cognition_group_visible_reply_boundary.py::test_duplicate_reservation_is_atomic_before_dialog` | `tests/test_action_spec_attempt_ledger.py::test_new_action_attempt_record_extends_existing_collection_shape` | deterministic unit/DB facade | two workers entering dialog for one source-window idempotency key |
| `src/kazusa_ai_chatbot/event_logging/__init__.py` and ICD | self-cognition semantic/policy/execution dispositions, policy reason, and gate codes | event-log contract owner | `tests/test_self_cognition_group_visible_reply_boundary.py::test_self_cognition_event_contract_rejects_raw_response_content` | `tests/test_event_logging_interface.py::test_public_event_logging_api_has_no_generic_record_event` | deterministic unit/static | raw prompt, source, candidate, or identity leakage in new telemetry |
| `src/kazusa_ai_chatbot/cognition_core_v2/README.md` and `src/kazusa_ai_chatbot/self_cognition/README.md` | capability and ownership documentation | ICD/documentation owner | `tests/test_self_cognition_architecture_docs.py::test_canonical_self_cognition_readme_documents_outcome_dimensions` | `none` | static | future maintainers reintroducing hard `think_only` silence or adapter bypass |

The live tests are supplemental evidence, not replacements for the exact
deterministic owner nodes. Each live case must run separately, preserve its
raw output under `test_artifacts/diagnostics/`, and be inspected before the
next case starts.

## Verification

Execution must capture the worktree baseline and owned file set before the
first source edit. The implementation owner then:

1. Collects every exact node in the traceability table and fails on a missing
   or stale node.
2. Runs the deterministic contract, route, connector, tracking, runner,
   worker, event-log, and documentation nodes.
3. Runs the listed integration nodes for group dispatch, no-speak behavior,
   and scheduled commitment speech.
4. Runs the two live LLM tests individually with `venv\Scripts\python`,
   inspecting the raw model response, parsed response, semantic disposition,
   policy disposition/reason, execution disposition, dialog count, and
   dispatch status after each case.
5. Writes the parent-authored
   `test_artifacts/diagnostics/self_cognition_group_visible_reply_review.md`
   with the failure baseline, both live outcomes, and any bounded model
   contract repairs.
6. Verifies the original latest-case-shaped replay has
   `cognition_contract_failed` when its observed action-planning result lacks
   the new response object; verifies a structurally targetless proposal has
   `reply_proposed + policy rejected + unresolved_target`; verifies the
   eligible fixture reaches the existing dialog and dispatcher seams.
7. Runs `venv\Scripts\python -m compileall` for changed Python paths,
   `git diff --check`, and the repository's required focused test command.
8. Performs an independent code review against this plan, resolves all
   findings inside scope, and records the final evidence before changing the
   plan lifecycle state.

## Acceptance Criteria

- The global action registry still exposes `speak`, and a validated group
  `propose_visible_reply` reaches the existing L3 `speak` materializer.
- A group `stay_silent` result records semantic `cognition_declined`, policy
  `not_evaluated`, and execution `not_requested`, calls dialog zero times, and
  requests no dispatch.
- The original captured run is classified as semantic
  `cognition_contract_failed` because its observed action-planning result has
  no dedicated response object; a replayed proposal with no structured
  participation grounding records semantic `reply_proposed`, policy
  `rejected/unresolved_target`, and execution `not_requested`.
- A malformed or exhausted response proposal cannot reach route, persistence,
  dialog, or dispatch as a visible candidate.
- A recent, structurally grounded, non-duplicate, bound group proposal calls
  dialog at most once and reaches the existing dispatcher boundary.
- High-risk or ambient labels alone never reject a proposal; a proposal is
  rejected only when the fixed structured participation/provenance/target
  policy fails.
- Duplicate prevention reserves the source-window idempotency key atomically
  before dialog; a reservation conflict produces no second dialog or
  dispatch.
- Duplicate group windows remain suppressed and cannot generate a second
  delivery.
- Scheduled commitment speech and ordinary user-response speech retain their
  existing routes and delivery ownership.
- Every semantic, policy, execution disposition and gate code is visible in
  bounded tracking, worker event metadata, and the human-readable review
  artifact.
- New telemetry contains no source packet text, candidate dialog, raw prompt,
  raw model output, platform identity, database identity, or credentials.
- All exact deterministic traceability nodes collect and pass; live results
  are individually inspected; independent code review has no unresolved
  blocking finding.

## Execution Record

Implementation and parent-owned verification completed on 2026-08-11. The
repair preserves the existing `speak` registry and L3/dialog materializer,
adds the exact targetless-group response contract, makes a valid proposal own
the speech route, applies structured provenance/freshness/participation/target
gates, reserves the source-window identity atomically before dialog, and keeps
response telemetry absent when scheduled or target-binding-only work never
evaluated a response.

Terra's independent code review initially identified six blockers: route
precedence, incomplete response validation, incomplete provenance/target
binding, duplicate inconsistency after `delivery_failed`, false non-group
contract-failure telemetry, and incomplete impact mapping. All six were
repaired. Sol's final high-reasoning review then returned `ready` with no
remaining blocker.

Verification evidence:

- Focused deterministic Cognition V2/self-cognition/event/documentation suite:
  173 passed.
- Group integration and speak-surface/evaluator regressions: 6 passed.
- Source-impact manifest gate: 28 exact nodes collected and passed.
- Individually inspected live cases: ambient `stay_silent` -> `silence`, and
  direct-address `propose_visible_reply` -> `speech`; both contracts valid.
- Targeted Ruff import checks, changed-path compilation, and `git diff --check`
  passed. Broader Ruff output retains unrelated pre-existing findings outside
  this repair's semantic changes.

Diagnostic review artifact:
`test_artifacts/diagnostics/self_cognition_group_visible_reply_review.md`.
