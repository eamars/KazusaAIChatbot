# DSH Plan 3: Brain Big-Bang Cutover And Legacy Resolution Decommission

## Summary

- **Goal:** complete the final DSH integration stage by atomically replacing
  the production task-execution edge beneath
  `cognition_resolver` with the completed Plan 1/Plan 2 DSH Standard runtime,
  preserve the current Brain-owned post-selector handover, and remove every
  superseded task, complex-resolution, RAG2-supervisor, and coding executor.
- **Status:** completed at the reviewed release-candidate boundary on
  2026-08-31. Production implementation, local verification, focused E2E
  sign-off, and independent closure review passed. P3-P4, P3-G9, and the
  deployment portion of P3-G10 were not executed by owner direction because no
  target environment or production process/data authority was supplied.
- **Scope boundary:** one DSH-only task-execution release spanning the Python
  Brain edge, durable task binding, accepted/background continuation, legacy
  executor deletion, the one required public-media semantic port, obsolete
  configuration and tests, and matching ICDs. Plan 1/Plan 2 sidecar and
  semantic-gateway ownership remain intact while their catalog/forwarding
  surfaces receive that exact additive tool. The Brain interaction bridge is
  replaced by the character-owned internal-cognition contract below. RAG3
  evidence services, dialog, adapters, consolidation, scheduler, and
  reflection remain their existing owners.
- **Change direction:** big-bang replacement. The candidate contains one
  production task route, one canonical contract vocabulary, and one
  deployment boundary. It contains no legacy fallback, dual read, shadow
  route, converter, alias, feature flag, percentage rollout, or compatibility
  package.
- **Acceptance state:** Plan 1 and Plan 2 are completed with their recorded
  gates green; the post-Plan 2 source/data/config/test inventory and all Plan 3
  design decisions are closed below. Plan 3 implementation is in progress.

## Trigger-Source E2E Sign-Off Supersession — 2026-08-31

The user approved
`development_plans/archive/completed/bugfix/dsh_phase3_focused_e2e_signoff_reset_plan_2026-08-31.md`
as the authoritative Phase 3 live-E2E release oracle. It supersedes every
earlier Plan 3 assertion, command, node cap, demonstration contract, and gate
that treats the old omnibus DSH E2E or P-stage producer probes as current
sign-off coverage. Their recorded runs remain historical failure evidence.

The current sign-off matrix has exactly two independently collectible live
nodes for each canonical cognition trigger source: positive DSH-entry proofs
for `user_message`, `internal_thought`, and `scheduled_tick`, and deliberate
non-entry proofs for the reachable targetless `self_cognition` source and the
recursion-closed `tool_result` source. Automated gates validate stable source,
readiness, lifecycle, evidence, recurrence, delivery, and cleanup contracts.
Semantic behavior is reviewed from retained dossiers without exact visible
phrases, language, opaque markers, private stage counts, or tool choreography.

By explicit user direction, the initial diagnostic pass runs all ten nodes in
one pytest invocation and preserves every reachable result. The executor makes
no mid-batch behavior change, classifies shared/systematic failure modes after
the complete batch, then fixes the smallest demonstrated owner. A production
fix requires a bounded amendment naming the exact owner and acceptance checks
before edit. Affected nodes then rerun individually with complete inspection.

The unchanged first all-ten run completed on 2026-08-31 with 2 passed and 8
failed. The focused plan now contains the authoritative case-by-case artifact
ledger and systematic diagnosis. Five shared causes were found: one transient
Mongo readiness incident affecting three cases; harness trace, shutdown, and
startup-exception handling defects; one invalid tool-result fixture; one
internal-latch profile-hydration defect; and one overly strict null-versus-
absence check for the optional fresh-response continuation carrier. No failure
supported a prompt change, source-registry change, targetless-user fabrication,
or wider DSH redesign.

The user's implementation command authorizes the focused plan's bounded
remediation amendment. Production scope is limited to:

- `self_cognition.worker._case_from_internal_action_latch`, which must hydrate
  the existing real bound user profile before shared cognition and
  consolidation; and
- `cognition_core_v3.facade._validate_plan`, which may normalize explicit null
  to absence only for the optional `fresh_ordinary`
  `pending_task_continuation` carrier when no human clarification exists; and
- `cognition_shared.state_reducers.materialize_causal_root`, which must
  reactivate an exact terminal same-source causal identity in place instead of
  appending the same deterministic id during resolver recurrence; and
- `sidecars/dsh_resolution/src/contracts.ts::validateSubmitResolution`,
  `task_resolution.contracts.validate_task_resolution_result`, and
  `task_resolution.projection.project_dsh_exhaust`, which must close the
  demonstrated terminal-status coherence gap without reclassifying semantic
  results downstream.

The third owner is supported by failed-result artifact
`tool_result_failed_20260831T010234Z_50214975`: the typed failed result and
zero-DSH recurrence closure survived, while a third valid cognition cycle
failed on `duplicate knowledge_gaps entity id` before dialog. Its exact
deterministic regression and affected live rerun are owned by the focused plan.

Non-null unauthorized continuation, non-fresh exact-field closure, task
readiness, targetless group identity, DSH schemas, prompts, model routes, and
sidecar behavior remain unchanged. Exact deterministic regressions and
individual live reruns named by the focused plan gate these corrections.

## Coding-Agent Test De-Overfitting Amendment — 2026-08-31

The user directed removal of every coding-agent-specific unit, integration,
and real-LLM test. The completed DSH cutover owns generic task execution;
Kazusa no longer carries a second coding-agent behavior contract. Earlier
Plan 2 and Plan 3 test rows, commands, matrices, and acceptance text that name
the removed nodes below are historical evidence and no longer gate Plan 3.

The removed behavioral nodes are:

- `tests/test_agentic_resolver_sidecar_process.py::test_standard_pwsh_reads_edits_runs_fixture_test_and_imports_no_kazusa_coding_agent`;
- `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_standard_mode_repairs_fixture_with_native_workspace_tools`;
- `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_outside_workspace_request_round_trips_through_brain_once`;
- `tests/test_dsh_plan2_e2e_live_llm.py::test_e2e_native_coding_repairs_and_verifies_workspace_fixture`;
- `tests/test_dsh_plan2_e2e_live_llm.py::test_e2e_brain_judgment_owns_native_approval_and_one_shot_grant`;
- `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_native_coding_uses_brain_approval_and_returns_verified_artifact`;
- `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_coding_followups_revision_summary_status_approval_blocker_and_cancel_preserve_session`;
- `tests/test_dsh_brain_interaction_live_llm.py::test_brain_cognition_decides_native_approval_without_user_relay`;
- `tests/unit/accepted_task/test_dsh_task_lifecycle.py::test_accepted_task_public_exports_exclude_coding_run_contexts`.

The coding-only `tests/fixtures/dsh_standard_coding/` fixture is deleted.
The coding-ledger-specific
`tests/unit/scripts/test_check_dsh_plan3_drain.py` module is deleted.
Mixed decommission, route, configuration, documentation, action, database,
manifest, and resolver-boundary tests retain only their non-coding contracts.
The removed-source manifest no longer carries historical coding-agent source
rows or maps them to absence-policing tests.

The retained exception is the existing result-delivery boundary. Tests may
continue to project or reject the canonical empty `coding_run_context` field
while that field remains in `TaskResolutionResultV1`, and delivery UI fixtures
may display already persisted historical worker labels. Generic DSH profile,
native-tool catalog, evidence receipt, interaction/grant, sandbox, task
lifecycle, cognition recurrence, dialog, dispatcher, and delivery tests remain
because they test DSH or delivery ownership rather than a Kazusa coding agent.
This amendment changes no production source, deployment state, or data.

## Authoritative Character-Owned DSH Interaction Amendment — 2026-08-30

The user confirmed this amendment before implementation. DSH belongs to the
character rather than the user. Every DSH `question`, native-tool `approval`,
and `plan_review` is therefore an internal character decision made by the
existing full reusable cognition loop. A DSH interaction never becomes a user
prompt, visible relay, reply-matching flow, or dedicated DSH judgment stage.

This section is the current normative interaction contract. It supersedes the
Plan 2 relay/reply design and every earlier Plan 3 clause, fixture, test, gate,
or evidence-ledger conclusion that requires `relay_to_user`,
`continue_waiting`, a delivered DSH question, a matched user reply, or a DSH
interaction checkpoint. Earlier ledger entries remain historical evidence of
the implementation path and are not current acceptance requirements.

### Closed Semantic Ownership

1. The existing full cognition loop owns character stance, sufficiency,
   judgment, resolver recurrence, and the final DSH interaction decision. The
   implementation reuses the same `stage_1_goal_resolver` recurrence and final
   cognition-commit boundary used by ordinary character cognition; it does not
   call `run_cognition(...)` as a one-pass DSH shortcut and does not add a
   DSH-only semantic stage.
2. Allowed internal decisions are exact:
   - `question`: `answer` or `reject`;
   - `approval`: `allow_once` or `reject`;
   - `plan_review`: `answer`, `allow_once`, or `reject`.
3. An internal DSH interaction produces no dialog/L3 surface and no adapter
   delivery. Its validated decision returns directly to the waiting DSH hook.
   `allow_once` still creates and consumes one exact operation-bound grant.
4. A known user-controlled prerequisite that prevents cognition from forming
   or authorizing a bounded task objective is resolved by ordinary cognition
   before DSH admission. Once a DSH task is admitted, its questions and
   approvals remain character-owned and never solicit the user.
5. The DSH interaction episode exposes the normal character, user,
   relationship, affect, conversation, goal, and relevant evidence context to
   the reusable cognition loop. The full loop may use retained evidence
   resolvers. The `task_resolution_request` capability is unavailable inside
   this internal episode so a DSH question cannot recursively start another
   DSH task.
6. The P-stage prompt states the positive decision procedure: judge from the
   character's knowledge, values, relationship, current state, task objective,
   and exact interaction semantics; answer or authorize when the character has
   sufficient reason, otherwise reject. Enum and field-shape lists alone do
   not satisfy this contract.

### Closed Mechanical Contract And Atomic Cutover

The shared Brain interaction epoch becomes `dsh_brain_interaction.v2` in one
atomic Brain/sidecar release. The candidate accepts and emits V2 only; it adds
no V1 adapter, alias, dual read, fallback mapper, or checkpoint compatibility
route. The completed Plan 2 resolution RPC/intake/profile/policy/store epochs
remain unchanged.

- The signed request retains authenticated interaction, operation, DSH
  thread/segment/activation/lease, conversation, scope, audience, profile,
  catalog, route, workspace, policy, nonce, timestamp, issuer, and digest
  identities. Model-hidden authority remains hidden from cognition.
- `transient_detail` becomes one complete bounded semantic handoff. A question
  interaction accepts exactly one complete DSH question, including its prompt
  and choices; a multi-question bundle fails closed so no question is silently
  dropped. An approval interaction includes the native tool name, DSH reason,
  and the exact recovered executable argument object. A plan review includes
  the complete bounded review request. Decision-critical semantic detail is
  rejected when it cannot fit the declared bound; it is not truncated.
- The V2 decision has exact fields `schema_version`, `interaction_id`,
  `request_digest`, `kind`, `decision`, `answer`, and `reason`. Relay-only
  `response_goal`, `relay_mode`, `checkpoint_required`, pending interaction,
  delivery-lineage, reply, and continuation fields are absent.
- Deterministic code owns authentication, strict shape and kind compatibility,
  digest/nonce/time/scope checks, idempotent audit persistence, exact one-shot
  grant creation/consumption, bounds, and fail-closed errors. It performs no
  keyword classification, semantic rewrite, default decision, or post-LLM
  override.
- Pre-cutover open V1 pending interactions and available grants remain a drain
  condition. After the zero-open drain, historical V1 rows remain inert audit
  records and the V2 runtime never interprets them as actionable state.

### Exact Amendment Change And Test Ownership

| Production owner | Required change | Exact deterministic owner |
|---|---|---|
| `sidecars/dsh_resolution/src/brain_interaction.ts`; `sidecars/dsh_resolution/src/main.ts` | Emit/consume V2, preserve complete bounded semantic detail, return internal decisions directly, and remove checkpoint callback/cancel behavior. | `sidecars/dsh_resolution/tests/brain_interaction.spec.ts` V2 question/approval/plan-review, complete-detail, multi-question fail-closed, grant, and no-checkpoint cases. |
| `src/kazusa_ai_chatbot/dsh_interaction/contracts.py`; `decision.py`; `service.py`; `__init__.py` | Replace V1 relay/reply DTO and lifecycle with strict V2 internal decision and one-shot grant/audit lifecycle. | `tests/test_dsh_brain_interaction_contracts.py`; `tests/test_dsh_brain_interaction_decision.py`; `tests/test_dsh_brain_interaction_service.py`; `tests/test_dsh_brain_interaction_persistence.py`. |
| `src/kazusa_ai_chatbot/dsh_interaction/pending.py`; `resume.py` | Delete superseded user-relay/reply owners. | `tests/test_dsh_brain_interaction_contracts.py::test_v2_contract_has_no_relay_reply_or_pending_vocabulary`; decommission assertions in `tests/test_dsh_plan3_task_resolution.py`. |
| `src/kazusa_ai_chatbot/brain_service/contracts.py`; `src/kazusa_ai_chatbot/service.py` | Expose only the authenticated V2 request route; remove checkpoint route, reply judge, delivery/continuation sink, pending-reply projection, and relay composition. | `tests/unit/service/test_dsh_task_composition.py`; `tests/test_dsh_brain_interaction_service.py`; `tests/test_relevance_turn_settlement_graph.py`. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`; `persona_supervisor2_cognition.py`; `persona_supervisor2_schema.py`; `src/kazusa_ai_chatbot/state.py` | Run DSH interactions through the full reusable cognition recurrence/commit owner, project complete semantic context, suppress recursive task resolution, and remove pending-user-reply state. | `tests/test_dsh_brain_interaction_decision.py::test_dsh_interaction_runs_full_reusable_cognition_loop_and_returns_internal_decision`; `tests/unit/nodes/test_persona_supervisor2_dsh_task_actions.py`; `tests/test_cognition_resolver_loop.py`. |
| `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`; `facade.py` | Add the positive character-first procedure and validate only the exact per-kind internal V2 decisions. | `tests/unit/cognition_core_v3/test_dsh_interaction_contract.py`; `tests/test_dsh_brain_interaction_live_llm.py`. |
| `src/kazusa_ai_chatbot/cognition_resolver/contracts.py` | Restore pre-admission ownership: a known user-controlled fact blocking a bounded objective remains ordinary human clarification; execution-time DSH questions remain internal character cognition. | `tests/unit/cognition_core_v3/test_handleless_contract.py::test_plan_packet_keeps_user_prerequisite_before_dsh_admission`. |
| `src/kazusa_ai_chatbot/db/dsh_interactions.py`; `src/kazusa_ai_chatbot/db/bootstrap.py` | Retain idempotent V2 audit and one-shot grants; remove open-reply lookup/index ownership while preserving historical V1 rows as inert data. | `tests/test_dsh_brain_interaction_persistence.py`; `tests/test_stage3_fresh_database_bootstrap.py`. |
| `src/kazusa_ai_chatbot/task_resolution/service.py`; `src/kazusa_ai_chatbot/background_work/worker.py` | Treat an internal DSH interaction as part of the active DSH call; remove waiting-for-user interaction settlement while preserving ordinary cooperative task checkpoint/restart behavior. | `tests/unit/task_resolution/test_service.py`; `tests/unit/background_work/test_dsh_worker.py`; `tests/test_dsh_plan3_task_resolution.py::test_internal_dsh_interaction_continues_without_user_wait_state`. |
| `tests/ownership/source_test_impact_manifest.json` | Replace relay/reply mappings with every exact V2 owner node and deletion assertion. | `tests/test_test_impact_manifest.py::test_manifest_contains_dsh_plan3_owner_rows`; `scripts/validate_test_impact.py` run. |

`tests/test_dsh_brain_interaction_pending.py` and
`tests/test_dsh_brain_interaction_resume.py` are deleted with their superseded
owners. Relay-specific Plan 2/Plan 3 tests are rewritten to the V2 internal
contract rather than preserved as historical compatibility tests.

### Revised Live Acceptance

The real debug-user alpha/beta node becomes
`tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_real_debug_user_prerequisite_is_resolved_before_dsh_admission`.
Turn 1 uses the existing fixed alpha/beta request and must produce an ordinary
character clarification with zero accepted task, background job, DSH binding,
DSH interaction, or file read. Turn 2 supplies `Use
plan3_real_user_e2e/beta.txt.` through ordinary `/chat`; cognition may then
admit the bounded background task. The node proves beta-only execution and
normal final delivery. It contains no DSH relay, reply-lineage, or
interaction-checkpoint assertion.

The native-coding live node proves an internal character-owned approval and
exact one-shot grant without any user prompt. The follow-up live node proves
internal question/plan-review handling without pending-user state. Every live
LLM node still runs individually with complete output and protected trace
inspection.

The same persistent Plan 3 production worker remains
`dsh_p3_implementation_worker` on `gpt-5.6-luna` with maximum reasoning. The
worker owns only the amendment source/test/documentation rows assigned in its
handoff, preserves concurrent worktree changes, and leaves the repository-root
`README.md` unchanged until the existing final P3-P3 documentation gate.

### Post-Implementation Boundary Audit Amendment — 2026-08-30

The parent audit of the first V2 implementation slice found four attributable
contract gaps. They are legitimate closure work inside the approved
character-owned architecture and are added to Plan 3 before their production
edits:

1. `waiting_for_interaction` and its transition helpers still exist in the
   accepted-task, background-job, task-binding, and worker-orchestrator
   contracts. The V2 release removes this state and every writer, reader,
   export, query, and transition. An ordinary cooperative DSH checkpoint maps
   only to `checkpointed`/queued continuation; a DSH interaction completes
   inside the active call.
2. `db.dsh_interactions.find_open_interaction(...)` is a relay-era query and is
   deleted. The V2 audit store retains only request replay, nonce, decision,
   and exact one-shot-grant ownership. Its docstrings and indexes contain no
   delivery, reply, pending, or open-interaction lifecycle vocabulary.
3. A `dsh_brain_interaction.v2` runtime cannot expose V1 Python DTO names. The
   canonical public symbols become `DshBrainInteractionRequestV2`,
   `DshBrainInteractionDecisionV2`, `DshBrainInteractionResponseV2`, and
   `DshOneShotGrantV2`; every V1 interaction symbol is removed with no alias.
   The nonce derivation domain becomes `dsh_brain_interaction.nonce.v2` in the
   same atomic Brain/sidecar cutover.
4. An internal DSH episode may advertise only resolver capabilities that
   cannot solicit the user or recursively start DSH. Its exact resolver roster
   is `self_goal_resolution` only. `task_resolution_request`,
   `human_clarification`, and `approval_preparation` are absent from its P-stage
   packet. Open-ended single questions are valid: the complete question is
   preserved, options are validated and preserved when supplied, and an empty
   or absent options list is not a rejection condition. Multi-question bundles
   and semantic-detail overflow still fail closed.

The bounded cleanup ownership is exact:

| Additional production owner | Required cleanup | Exact deterministic owner |
|---|---|---|
| `src/kazusa_ai_chatbot/accepted_task/__init__.py`; `models.py`; `lifecycle.py`; `src/kazusa_ai_chatbot/db/accepted_tasks.py` | Remove `waiting_for_interaction` accepted-task state, public helper, repository writer, and terminal/failure query alternatives. | `tests/unit/accepted_task/test_dsh_task_lifecycle.py::test_dsh_task_states_have_no_user_interaction_wait`; `tests/unit/db/test_accepted_tasks.py::test_dsh_task_updates_have_no_interaction_wait_state`. |
| `src/kazusa_ai_chatbot/background_work/models.py`; `subagent/task_orchestrator.py`; `src/kazusa_ai_chatbot/db/background_work_jobs.py` | Remove interaction-wait job state, open-interaction lookup, wait requeue, and transition branches; preserve queued cooperative checkpoint continuation. | `tests/unit/background_work/test_dsh_jobs.py::test_dsh_job_states_have_no_user_interaction_wait`; `tests/unit/background_work/test_dsh_worker.py::test_internal_dsh_interaction_never_requeues_for_user_wait`; `tests/test_dsh_plan3_task_resolution.py::test_internal_dsh_interaction_continues_without_user_wait_state`. |
| `src/kazusa_ai_chatbot/db/task_resolution_sessions.py`; `src/kazusa_ai_chatbot/db/schemas.py` | Remove interaction-wait binding state and transitions from durable validation. | `tests/unit/db/test_task_resolution_sessions.py::test_binding_followup_schemas_are_closed_without_interaction_waiting`. |
| `src/kazusa_ai_chatbot/db/dsh_interactions.py` | Delete open-interaction query and relay-era lifecycle wording. | `tests/test_dsh_brain_interaction_persistence.py::test_v2_audit_and_one_shot_grant_are_idempotent_without_reply_lookup`. |
| `src/kazusa_ai_chatbot/dsh_interaction/auth.py`; `contracts.py`; `decision.py`; `service.py`; `__init__.py`; `src/kazusa_ai_chatbot/brain_service/contracts.py`; `src/kazusa_ai_chatbot/service.py` | Rename the complete canonical DTO/grant vocabulary to V2 and remove every V1 interaction symbol. | `tests/test_dsh_brain_interaction_auth.py`; `tests/test_dsh_brain_interaction_contracts.py::test_v2_public_contract_exports_have_no_v1_alias`; `tests/test_dsh_brain_interaction_service.py`. |
| `sidecars/dsh_resolution/src/brain_interaction.ts`; `sidecars/dsh_resolution/tests/brain_interaction.spec.ts` | Use the V2 nonce domain and support complete open-ended or choice-bearing single questions. | `sidecars/dsh_resolution/tests/brain_interaction.spec.ts` V2 nonce, open-ended, choice-bearing, multi-question, and overflow cases. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`; `src/kazusa_ai_chatbot/cognition_shared/contracts.py`; `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` | Project the exact internal resolver roster, keep full-loop recurrence/commit, and validate the complete bounded semantic string without creating a pending human resolver. | `tests/test_dsh_brain_interaction_decision.py::test_dsh_interaction_full_loop_advertises_only_internal_resolvers`; `tests/unit/cognition_core_v3/test_dsh_interaction_contract.py::test_dsh_plan_packet_has_no_user_solicitation_resolver`. |
| `tests/ownership/source_test_impact_manifest.json` | Add every cleanup owner and exact node; remove mappings to deleted wait/reply behavior. | `tests/test_test_impact_manifest.py::test_manifest_contains_dsh_plan3_owner_rows`; full impact validator. |

The cleanup uses the same persistent worker and the same deterministic
acceptance boundary. It changes no documentation, live environment, database
data, deployment state, or repository-root `README.md`.

### Post-Code Documentation Consistency Amendment — 2026-08-30

The completed V2 code audit found one plan-surface omission: the authoritative
character-owned amendment changed the Brain interaction contract, but its exact
change table did not enumerate the three subsystem ICD sections that still
described the superseded user relay/reply lifecycle. This is a legitimate
documentation-boundary loophole in the amendment, so Plan 3 closes it before
the documentation edits.

| Documentation/test owner | Required correction | Acceptance owner |
|---|---|---|
| `src/kazusa_ai_chatbot/dsh_interaction/README.md` | Document `dsh_brain_interaction.v2`, the sole interaction route, character-owned internal decisions, exact per-kind decision sets, complete bounded semantic context, V2 audit/replay and one-shot-grant ownership, and the absence of user relay/reply/checkpoint state. | `tests/test_dsh_plan3_documentation.py::test_character_owned_dsh_icds_exclude_user_relay_contract`. |
| `src/kazusa_ai_chatbot/brain_service/README.md` | Replace the relay/checkpoint paragraph with the V2 internal cognition boundary and direct return to the waiting DSH hook; retain deterministic authority and grant ownership. | Same exact documentation node. |
| `src/kazusa_ai_chatbot/cognition_core_v3/README.md` | State that the existing full reusable cognition loop makes the internal character decision with only internal resolver capability, no dialog/L3 surface, and no user permission inference. | Same exact documentation node. |
| `tests/test_dsh_plan3_documentation.py` | Add one bounded static assertion over those three ICDs for the V2 epoch, character ownership, full reusable cognition loop, exact internal decisions, and absence of the retired checkpoint/relay/reply vocabulary. | Run the exact new node independently; retain the two previously recorded broader documentation-suite failures as separate P3-P3 reconciliation work. |

The same persistent `dsh_p3_implementation_worker` owns these three ICD edits
and the exact test addition. The edit preserves all unrelated documentation,
changes no production code or live state, and leaves repository-root
`README.md` unchanged. The root README remains a final product-level audit only
after all code, deterministic, and live gates are complete, and is updated only
when a finalized global product statement is materially stale.

### Authoritative Verification And Sign-Off Order Amendment — 2026-08-30

The user fixed the remaining Plan 3 gate order. This sequence supersedes every
earlier command list or evidence note that places documentation validation,
documentation remediation, independent code review, registry maintenance, or
closure bookkeeping before real-LLM E2E acceptance:

1. Complete the implementations of all five named
   `tests/test_dsh_plan3_e2e_live_llm.py` nodes. At this amendment baseline,
   the inline and real debug-user prerequisite nodes are implemented, while
   the native-coding, coding-follow-up, and public-research/media nodes are
   explicit skip stubs. The same persistent
   `dsh_p3_implementation_worker` on `gpt-5.6-luna` with maximum reasoning owns
   those test-only implementations and any test-harness correction required
   to execute the already-approved live contracts. Production behavior changes
   require a new plan amendment before editing.
2. Run each of the five real-LLM nodes separately with `-q -s`. After each
   node, inspect its complete console output, durable run artifact, child
   process logs, protected LLM trace, DSH event/receipt lineage, Mongo task/job/
   binding state, and adapter delivery evidence before starting the next node.
   A skip, deselection, missing artifact, uninspected trace, or assertion-only
   pass does not satisfy the node.
3. After all five nodes have acceptable inspected evidence, present one bounded
   E2E sign-off dossier to the user covering behavior, invariants, artifacts,
   failures/retries, and residual risk. Stop at this gate until the user gives
   explicit E2E sign-off. The live-test run itself does not imply sign-off.
4. After user E2E sign-off, run the independent read-only code review. Route
   any finding to the same persistent implementation worker within an approved
   remediation boundary, rerun the affected live/deterministic evidence, and
   obtain an independent re-review before proceeding.
5. Documentation checks are the last verification activity, after the code
   review and any remediation/re-review are green. Only then run the Plan 3
   documentation/manifest nodes, reconcile documentation findings, perform the
   final repository-root `README.md` necessity audit, and execute remaining
   evidence-ledger, registry, archive, and closure bookkeeping.

This order changes verification sequencing only. It grants no production data
operation or deployment authority; those gates retain their separately named
environment and operation requirement.

## P3-P3 Rejected E2E Failure-Mode Remediation Amendment — 2026-08-30

The user rejected the first five-node E2E attempt and required a system-level
failure-mode evaluation before any remediation handoff. The attempt produced
zero accepted nodes. This amendment records the causal boundary and authorizes
only the exact remediation below for the same persistent implementation owner.
It supersedes every earlier clause or test expectation that permits a task to
be admitted while a known user-controlled prerequisite is still missing.

### Confirmed Failure Modes And Ownership

| Failure mode | Observed evidence | Root owner | Architectural consequence |
|---|---|---|---|
| Premature task admission | The prerequisite node produced the correct visible clarification, but P also emitted `requires_user_input` plus a background `task_resolution_request` repeatedly and two durable DSH bindings were created for one continuation. | `cognition_core_v3/prompt.py` and `cognition_resolver/contracts.py` explicitly instruct P to admit a task even when a later user task parameter is required. | The model-facing contract contradicts the approved pre-admission boundary. A known user-controlled prerequisite must be resolved by ordinary cognition before any DSH task exists. |
| Contradictory P-stage candidate accepted | The native-coding and coding-control nodes emitted `answerable_now` together with a background `task_resolution_request`; the resolver loop then suppressed the request and the visible response promised work that never entered DSH. | `cognition_core_v3/facade.py` validates the enum and request rows independently and has no cross-field compatibility rule. | This is a structural contract fault. Deterministic validation must reject the candidate and use the existing bounded P-stage regeneration path; the resolver loop must not reinterpret the model's semantic decision. |
| Foreground/background meaning drift | The inline node emitted `start_in_background=true` although its DSH result was required before the current visible answer could complete. | The prompt defines background partly as work continuing outside the current visible response, which does not distinguish foreground evidence recurrence from later delivery. | `start_in_background` remains model-owned, but its semantic meanings must be exact: foreground blocks the current answer for evidence; background acknowledges accepted work now and delivers through a later normal turn. |
| Model prose used as durable identity | The prerequisite node produced paraphrased objectives under the same typed continuation and created multiple active bindings/tasks. | Task-session identity hashes the model objective with the continuation, and active accepted-task identity hashes normalized model objective without the continuation. | LLM wording is content, not authority. Task-resolution identity must derive from trusted source scope and the validated continuation reference so retries, paraphrases, and concurrent admissions converge on one active task. |
| Shared-state and evidence-finalization failure in the live harness | All five nodes used `_test_kazusa_live_llm`; later Brain workers processed stale tasks and callbacks from earlier failed nodes. Several tests asserted visible behavior before retaining the graph trace id and then terminated Brain while trace persistence was still running. | `tests/test_dsh_plan3_e2e_live_llm.py` uses one static database and case-local assertion ordering that tears down the trace producer too early. | Every case requires a fresh guarded database and failure-safe evidence finalization. Cross-run worker/delivery errors from this attempt are non-attributable and do not authorize delivery-path changes. |

The failure is therefore a distributed contract failure across semantic
guidance, deterministic contract validation, durable identity, and test
isolation. It is not a reason to add keyword classification, a user-facing DSH
question path, a resolver-loop semantic override, or a delivery fallback.

### Corrected Admission And Execution State Machine

1. Ordinary cognition first decides whether the current observation provides a
   bounded and authorizable task objective.
2. If a known user-controlled fact or choice is required to form or authorize
   that objective, P returns `goal_resolution=requires_user_input`, an ordinary
   visible clarification goal, and no `task_resolution_request`. No accepted
   task, background job, DSH binding, workspace operation, or DSH interaction
   may exist on that turn.
3. Once the user supplies the prerequisite, ordinary cognition may admit the
   now-bounded task. A P candidate containing `task_resolution_request` is
   structurally compatible only with
   `goal_resolution=requires_required_evidence`. The existing P-stage attempt
   cap and feedback-bearing regeneration replace any candidate pairing the
   task request with `answerable_now`, `requires_user_input`, or `blocked`.
   Other resolver capability combinations retain their existing contracts.
4. `start_in_background=false` means the DSH result is required evidence for
   the current visible answer: task resolution executes or checkpoints through
   the foreground recurrence before that answer completes.
   `start_in_background=true` means the character acknowledges an admitted
   bounded task now and the result is delivered later through the normal Brain
   delivery path. A missing pre-admission user prerequisite produces no task
   request and therefore has no background value.
5. After admission, every DSH question, approval, and plan review remains owned
   by the character and is resolved by the full reusable cognition loop under
   the authoritative character-owned interaction amendment. No post-admission
   interaction is propagated to the user.
6. Deterministic code validates shape, enum membership, cross-field
   compatibility, identity, persistence, limits, and authority. The P-stage LLM
   retains semantic ownership of whether the task is bounded, whether evidence
   is required, and whether an admitted task is foreground or background.

### Durable Identity Correction

The canonical identity for a task-resolution session is the trusted source
scope plus the exact validated `goal_continuation_ref`. The canonical active
accepted-task identity includes `task_kind` and trusted source scope, then uses:

- the exact validated `goal_continuation_ref` for `task_resolution`; or
- the existing normalized semantic objective for `future_speak`, whose
  scheduling identity is not a task-resolution continuation.

The model-produced objective remains persisted task content and remains
available to DSH, dialog, audit, and follow-up cognition. It is excluded from
task-resolution duplicate identity. The existing unique active-identity write
remains the concurrency owner; no alias, compatibility key, parallel index, or
fallback matcher is added. The release drain gate prevents pre-cutover active
rows with the retired identity material from crossing deployment.

### Fresh-Case Live Harness Contract

Each real-LLM E2E node creates one unique database whose name begins with the
reserved `_test_kazusa_` prefix. The Brain process accepts that database only
when the global test guard, a generic ephemeral-test guard, and the exact
database-name environment value all agree. The harness seeds the canonical
`personalities/example.json` identity into that database before starting the
Brain service and opens parent-side diagnostics against that exact database.

On success or failure, the harness performs this order:

1. recover the trace id from the response graph, or from the exact platform,
   channel, message, and user correlation when response assertions failed;
2. while Brain is still alive, wait for trace finalization and snapshot the
   trace, task, job, binding, DSH interaction, callback, and DSH event lineage;
3. write the complete UTF-8 run artifact and child logs, including the database
   name and cleanup disposition;
4. stop only the child processes owned by that case; and
5. close the parent diagnostic client and drop only that exact, guarded,
   case-created ephemeral test database after the artifact is durable.

This cleanup authority is limited to the unique local test database created by
the executing case. It grants no production data, shared test-database,
deployment, or migration operation. A fresh isolated reproduction is required
before any delivery, result-source, callback, or sidecar execution change can
enter scope.

### Exact Remediation Ownership

| Owned file | Required change | Exact acceptance owner |
|---|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` | Define all four goal-resolution meanings, the task-resolution compatibility rule, the pre-admission user-prerequisite boundary, and exact foreground/background semantics in the P packet and output contract. | `tests/unit/cognition_core_v3/test_handleless_contract.py::test_plan_packet_separates_user_prerequisite_from_task_admission`; three individual live P-stage nodes below. |
| `src/kazusa_ai_chatbot/cognition_resolver/contracts.py` | Replace the superseded “admit now and ask the user later” task semantics with the bounded-objective/pre-admission distinction. | The same handleless contract node. |
| `src/kazusa_ai_chatbot/cognition_core_v3/facade.py` | Reject a parsed P candidate when `task_resolution_request` is paired with any goal resolution other than `requires_required_evidence`, allowing the existing bounded regeneration owner to replace it. | `tests/unit/cognition_core_v3/test_handleless_contract.py::test_task_resolution_request_requires_required_evidence_resolution`; existing cognition contract-regeneration coverage. |
| `src/kazusa_ai_chatbot/task_resolution/service.py` | Derive task-session identity from trusted source scope and validated continuation identity, excluding model objective wording. | `tests/unit/task_resolution/test_service.py::test_task_session_identity_is_stable_across_objective_paraphrases`. |
| `src/kazusa_ai_chatbot/accepted_task/models.py`; `lifecycle.py` | Type and derive task-kind-specific active identity so task-resolution uses trusted scope plus continuation while future-speak retains semantic-objective identity. | `tests/unit/accepted_task/test_dsh_task_lifecycle.py::test_task_resolution_active_identity_uses_continuation_not_model_wording`; `::test_future_speak_active_identity_retains_semantic_objective`. |
| `src/kazusa_ai_chatbot/db/_client.py` | Add one generic, exact-name, reserved-prefix ephemeral test-database guard for isolated live cases. | `tests/test_live_llm_mongo_isolation.py::test_guard_allows_exact_reserved_ephemeral_database`; `::test_guard_rejects_ephemeral_database_name_mismatch`; `::test_guard_rejects_ephemeral_database_outside_reserved_prefix`. |
| `tests/test_dsh_plan3_cognition_admission_live_llm.py` | Add three independent real P-stage behavior nodes for foreground task admission, explicit background task admission, and a missing known user-controlled prerequisite. | Run each named node separately with `-q -s` and inspect the raw P output and protected trace. |
| `tests/test_dsh_plan3_e2e_live_llm.py` | Give every node a fresh guarded database, seed the character profile before startup, bind diagnostics to that database, and preserve/finalize evidence before teardown even when a behavioral assertion fails. | All five existing E2E nodes run and are inspected individually. |
| `tests/ownership/source_test_impact_manifest.json` | Add or update the exact source-to-test mappings for every production owner above, including the new database-client row. | `scripts/validate_test_impact.py`; `tests/test_test_impact_manifest.py::test_manifest_contains_dsh_plan3_owner_rows`. |

The three live P-stage nodes are exact:

1. `test_live_p_stage_foreground_task_requires_evidence_before_current_answer`
   must select `requires_required_evidence`, one
   `task_resolution_request`, and `start_in_background=false`.
2. `test_live_p_stage_explicit_background_task_requires_evidence_and_later_delivery`
   must select `requires_required_evidence`, one
   `task_resolution_request`, and `start_in_background=true`.
3. `test_live_p_stage_user_controlled_prerequisite_stays_before_task_admission`
   must select `requires_user_input` and zero `task_resolution_request` rows.

The same persistent `dsh_p3_implementation_worker` on `gpt-5.6-luna` with
maximum reasoning owns this exact slice. The worker preserves concurrent
worktree changes and changes only the files listed above plus directly required
test helpers within those test files. Resolver-loop suppression, dialog/L3,
the V2 DSH interaction bridge, delivery/result-source owners, sidecar native or
semantic execution, adapters, production configuration, deployment, data
migration, subsystem documentation, and both repository-root README files are
outside this remediation.

### Remediation Verification Order

1. Run the exact deterministic owner nodes and impact validator.
2. Run the three live P-stage nodes one at a time with `-q -s`; inspect each raw
   model candidate, any regeneration, parsed plan, and protected trace before
   proceeding.
3. Run the five full E2E nodes one at a time with `-q -s`; inspect each complete
   artifact and trace before proceeding.
4. Present the revised five-node dossier and stop for explicit user E2E
   sign-off.
5. Continue with the already-authoritative independent review, remediation and
   re-review gate.
6. Perform documentation checks, README necessity evaluation, registries, and
   closure bookkeeping last.

### P3-P4 Node 2 Cross-Turn Clarification Continuity Amendment — 2026-08-30

The first isolated run of the real debug-user prerequisite node passed the
Turn 1 pre-admission boundary and failed on Turn 2. Turn 1 produced
`requires_user_input`, one ordinary `human_clarification`, zero task request,
zero task/binding/job/interaction rows, and no file read. Turn 2 supplied the
plan-fixed ordinary `/chat` message `Use plan3_real_user_e2e/beta.txt.`. P
admitted a foreground task, even though the unresolved original request had
explicitly required background execution. DSH read only `beta.txt` and its
terminal evidence contained `PLAN3_E2E_BETA_SELECTED`, but the final visible
surface omitted that exact requested marker. The test harness then raised
`UnboundLocalError` while finalizing an unset `trace_id`, leaving the fresh test
database for recovery.

The protected Turn 2 trace and source audit establish a cross-turn ownership
failure rather than a foreground/background classification error:

1. The admission/history input and stored conversation-progress episode
   narrative still contained the original background instruction and marker
   requirement. The citeable conversation-progress event retained only the
   ask/choose/read/report workflow and dropped task mode. Conversation progress
   is lossy factual continuity and is not the canonical authority for exact
   task mode or deliverables.
2. `cognition_resolver.pending` already persists the prompt-safe original goal,
   clarification question, and goal progress. Its loader accepts a later turn
   only when adapter reply metadata points to the original user source message.
   The approved ordinary `/chat` follow-up has no such reply metadata, so the
   durable continuation is not loaded.
3. Even when a pending row is loaded, the cognition caller supplies
   `pending_resolver_resume`, `resolver_context`, and `resolver_goal_progress`,
   but Cognition V3 stores the latter two only in its workspace and exposes
   none of them to A1, A2, G, or P. The real canonical P output has no pending
   lifecycle decision field. Existing resolver-loop tests use mocked cognition
   results and therefore did not prove the production full-loop path.

The corrected architecture treats one open ordinary clarification as a typed
pending-continuation candidate, not as inherited current intent:

1. Deterministic code may select the newest unexpired `human_clarification`
   candidate by exact platform, channel, current global user, and time ordering
   on a later ordinary user turn. Explicit reply metadata remains supporting
   correlation but is not required for normal conversational adjacency. The
   candidate never authorizes work by itself.
2. The complete prompt-safe original goal, clarification question, status, and
   goal progress enter the existing reusable A1/A2/G/P loop as a bounded
   continuation lane. `current_observation` remains the sole authority for
   deciding whether the current user answered, rejected, superseded, or left
   the pending matter waiting.
3. P returns that semantic pending disposition in the same ordinary response
   plan. There is no dedicated judge stage. The model never receives or authors
   the durable resume id; deterministic code validates the disposition and
   binds it to the one selected pending row before persistence.
4. `answered` may combine the current answer with the retained original goal
   and admit the now-bounded task. `continue_waiting` admits no continuation
   task from the pending goal. `rejected` or `superseded` closes the pending
   matter according to the existing lifecycle. An unrelated same-scope message
   is therefore interpreted by cognition rather than by keyword code or an
   adapter reply heuristic.
5. The retained original goal remains available through the ensuing resolver
   recurrence so P owns both the original `start_in_background=true` constraint
   and the final marker-report requirement. Conversation-progress prompts,
   dialog/L3, and deterministic post-LLM overrides are not used to reconstruct
   or force those semantics.

The same persistent implementation worker owns this bounded correction:

| Owned file | Required change | Exact acceptance owner |
|---|---|---|
| `src/kazusa_ai_chatbot/cognition_resolver/pending.py`; `contracts.py` | Select the newest exact-scope clarification candidate without requiring adapter reply metadata; preserve current-turn authority; validate the bounded, capability-compatible lifecycle disposition. | `tests/test_cognition_resolver_loop.py` direct ordinary follow-up, unrelated-turn `continue_waiting`, answered/superseded closure, hidden-id binding, and original-goal restoration nodes. |
| `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`; `facade.py`; `contracts.py` | Project the typed pending continuation through A1/A2/G/P, add the exact ordinary P disposition field only when a candidate exists, and reject malformed, missing, incompatible, or extra dispositions through the existing bounded P regeneration path. | `tests/unit/cognition_core_v3/test_handleless_contract.py` packet visibility, no durable id, exact output shape, and pending/task cross-field nodes. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | Carry the prompt-safe pending candidate into the canonical input and bind the validated model disposition to the hidden selected resume id in the projected global state. | Existing cognition input/projection tests plus the exact handleless nodes above. |
| `tests/test_dsh_plan3_e2e_live_llm.py` | Initialize failure-path trace state before assertions, finalize recoverable evidence before child teardown, retain exact marker/background assertions, and clean the exact abandoned ephemeral database after evidence recovery. | Rerun only `test_e2e_real_debug_user_prerequisite_is_resolved_before_dsh_admission` first with complete console, artifact, protected-trace, DSH-lineage, Mongo, and delivery inspection. |
| `tests/ownership/source_test_impact_manifest.json` | Add exact mappings for every newly touched production owner and acceptance node. | `scripts/validate_test_impact.py` after affected deterministic tests. |

This amendment changes no DSH interaction ownership: after admission, DSH
questions, approvals, and plan reviews remain internal character decisions in
the full reusable cognition loop and never prompt the user. It adds no keyword
classifier, user-input pre/post processor, compatibility shim, dialog rewrite,
production database operation, deployment action, subsystem documentation, or
repository-root README change. After the exact Node 2 rerun passes and is
inspected, execution resumes at E2E Node 3 under the existing verification and
sign-off order.

### P3-P5 Cognition-Core Prompt Ownership Amendment — 2026-08-31

The parent prompt-construction audit found that Cognition V3 currently treats
authored instructions as reusable building blocks. `prompt.py` and `facade.py`
substitute shared authority/guidance constants into A1, A2, G, and P prompts,
compose ordinary and DSH plan guidance from templates, append pending-resolver
guidance conditionally, and join a reusable exact-JSON suffix onto each stage
system prompt. This makes the effective instruction dependent on assembly
order, duplicates semantic rules between system and packet guidance, and makes
one local edit change several model-call contracts without exposing each final
prompt for direct review.

The approved prompt-ownership rule is now:

1. Authored prompts are not reusable components. Each cognition model-call
   contract owns one complete literal instruction. Repeated semantic wording is
   written directly into every owning prompt. No authored prompt, guidance
   block, authority block, suffix, or repair instruction is inserted into or
   concatenated with another prompt through `.format()`, f-string prompt
   interpolation, template replacement, joining, or conditional appending.
2. Runtime state and values remain projectable. Existing typed projectors may
   filter, bound, rename, and serialize current observations, evidence,
   continuity, DSH interactions, resolver progress, output contracts, and
   bounded repair evidence into the model-facing human packet. Projection is a
   data boundary and does not authorize projecting authored instruction text.
3. A1, A2, G, ordinary P, pending-clarification P, internal-DSH P, and
   self-cognition P each receive a complete, directly reviewable semantic
   instruction appropriate to that exact call contract. Selecting one complete
   literal for a call is allowed; constructing its wording from other prompts
   is not. Model-facing packets carry projected data and contracts, not a
   second `guidance` prompt.
4. Cognition stages use the repository's structured JSON transport rather than
   `output_mode="text"`. Prompt wording does not mechanically demand JSON,
   enumerate exact field sets or types, repeat enum restrictions, require
   identifiers to be copied, reject extra fields, or restate length/list
   bounds. The projected output contract, provider transport, canonical parser,
   deterministic validators, and bounded regeneration path retain mechanical
   ownership. Prompts retain only semantic judgment, evidence-authority,
   character-ownership, and repair-meaning instructions.
5. DSH interaction identity, kind, allowed decisions, pending clarification,
   goal progress, and contract-repair evidence remain bounded projected values.
   The character cognition prompt owns the semantic decision. Deterministic
   code owns structural validation and durable identity binding; dialog and L3
   remain outside internal DSH wording.

The same persistent GPT-5.6 Terra/high implementation worker owns this bounded
prompt refactor:

| Owned file | Required change | Exact acceptance owner |
|---|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` | Replace reusable prompt fragments/templates and packet `guidance` strings with complete literal stage/variant prompts; retain typed runtime projections and output-contract data. | Cognition-core prompt contract tests prove each effective prompt is complete, projected packets contain data rather than authored guidance, and DSH/pending/self variants preserve their semantic ownership. |
| `src/kazusa_ai_chatbot/cognition_core_v3/facade.py` | Replace assembled stage system prompts and the shared JSON suffix with complete literal prompts, select only complete prompt variants, use structured JSON transport, and preserve bounded parse/validation/regeneration without projected prompt instructions. | Focused facade tests assert `json_object` transport, stable message boundaries, repair evidence projection, and unchanged typed validation behavior. |
| `tests/unit/cognition_core_v3/**`; `tests/ownership/source_test_impact_manifest.json` | Replace fragment-oriented assertions with final-prompt behavioral assertions and register every touched source owner. | Focused and adjacent deterministic suites, impact validation, compile, attributable Ruff, and `git diff --check`. |

The worker first inventories every model-facing instruction in
`cognition_core_v3`; `appraisal.py`, `contracts.py`, and `diagnostics.py` change
only if that inventory finds an actual prompt-construction owner there. No
real-LLM test runs during this refactor. After parent code audit, the exact
individual P-stage real-LLM tests run first, followed by E2E Node 2 and the
remaining E2E nodes under the existing sign-off order. Documentation, README,
registry, and closure checks remain last.

## Plan Lineage And Authority

| Plan | Lifecycle authority | Plan 3 carryover |
|---|---|---|
| [Plan 1 — standalone DSH sidecar and resolution interface](../../archive/completed/short_term/dsh_standalone_sidecar_and_resolution_interface_plan_2026-08-26.md) | Completed; all seven original gates passed. | Preserve the independently operating Node sidecar; `dsh_resolution_intake.v2`; durable Mongo ResolutionThread metadata; DSH SQLite sessions and semantic outcomes; operation idempotency and payload digests; activation/lease fencing; segment rotation; `resolution.open`, `resolution.continue`, `resolution.amend`, `resolution.request_checkpoint`, `resolution.cancel`, `resolution.inspect`, and `resolution.dispose_activation`; evidence receipts; and the closed terminal/checkpoint/fault/cancel exhaust. |
| [Plan 2 — DSH Standard semantic tools and coding capability](../../archive/completed/short_term/dsh_semantic_tools_and_coding_capability_plan_2026-08-26.md) | Completed; P2-P0 through P2-P3 and G1 through G10 passed without waiver; independent review passed. | Preserve the pinned official DSH `0.1.1-rc.2` Standard profile, `kazusa-resolver-standard-v2`, `kazusa.dsh-resolution-rpc.v2`, `dsh_resolution_intake.v2`, `dsh-standard-policy-v2`, `dsh-sqlite-0.1.1-rc.2-standard-v2`, the six-field `AGENTIC_RESOLVER_LLM_*` route, native Standard filesystem/shell/coding/jobs/tests/web/approval/sandbox ownership, all thirteen existing storage-independent `kazusa_*` semantic tools byte-for-byte, the sole `submit_resolution` terminal, autonomous multi-tool execution, and exact one-shot grants. Plan 3 supersedes the Plan 2 user-relay interaction contract with `dsh_brain_interaction.v2` under the authoritative amendment above. Plan 3 also adds only `kazusa_inspect_public_media` to port the legacy public-image function, yielding fourteen tools and a new semantic catalog digest without changing the Plan 2 resolution profile/RPC/policy/store epochs. |
| This Plan 3 | Approved planning contract. | Perform only the production edge cutover and explicitly enumerated decommission work below. |

### Predecessor Execution-Ownership Reconciliation

Plan 1's original fixed Luna worker did not complete its assignment. Its final
user-authorized parent-remediation amendment superseded that worker-only
binding and recorded the completed implementation evidence. Plan 2 then
states: the active primary engineering agent owns implementation and
verification, and the plan binds no implementation worker or model. Therefore
Plan 3 inherits no fixed executor, model, agent name, retry count, or review
iteration cap. Runtime executor resolution follows the role contracts in this
plan.

### Immutable Plan 1/Plan 2 Product Contracts

1. The Brain supplies model-visible `objective` and `facts` separately from
   model-hidden identity, workspace, route, semantic-tool, interaction, scope,
   audience, and fencing authority.
2. DSH Standard owns autonomous reasoning and native tool use. Kazusa adds no
   DSH step, tool-call, byte, reasoning-deadline, coding, command-filter,
   sandbox, or web-provider policy.
3. `submit_resolution` is the sole model-owned terminal. Its validated result
   becomes a `DSHResolutionExhaustV2`; malformed or unbound output remains out
   of Brain action, persistence, and delivery paths.
4. The full reusable Brain cognition loop owns whether the character answers
   or rejects a DSH question/review and whether the character allows once or
   rejects an exact approval. Deterministic code owns authentication, strict
   identity, one-shot consumption, persistence, and fencing. DSH interactions
   never relay to the user.
5. Production `task_resolution_request`, the accepted/background handover,
   and mapping into `TaskResolutionResultV1` were deliberately deferred by
   Plan 2 and become the atomic Plan 3 boundary.
6. Plan 2's Windows native-tool ACL variance remains a DSH Standard operational
   behavior. The Brain interaction route is the supported approval path; Plan
   3 adds no policy override.
7. Plan 2's thirteen semantic tool names, schemas, authorities, results, and
   behavior remain exact. The additive Plan 3 public-media tool is a new
   catalog row and cannot modify or alias an existing tool.

## Current Codebase Audit — 2026-08-29

| Finding | Current evidence | Plan 3 disposition |
|---|---|---|
| Production task edge is still legacy. | `cognition_resolver.capabilities._execute_task_resolution_request` calls `task_resolution.service.resolve_task_inline`, `start_task_resolution_in_background`, and `promote_deferred_task_resolution`; those functions create/run the old checkpointed four-specialist graph. | Rewrite the service beneath the same cognition capability and result-observation boundary. |
| The Plan 2 runtime is process-owned but interaction-only. | `service.py` constructs one `_dsh_resolver_runtime` and injects it into `BrainInteractionService`; `task_resolution` and the background worker cannot use it. | Inject that same runtime into the task-resolution service and background executor from the lifespan composition root. |
| Generic non-interaction checkpoint continuation is not public yet. | `ResolutionController` has the agreed operations; `AgenticResolverRuntime` publicly exposes `resolve` and interaction continuation only. | Add bounded runtime wrappers for amend, cooperative checkpoint, checkpoint continuation with fresh authority, cancel, and inspect. Preserve the controller/sidecar operation vocabulary. |
| A resumed interaction can outlive its original caller. | `_production_dsh_continue` returns a continuation result to the interaction row, while accepted/background task state has no thread binding or completion sink. | Persist a task binding before DSH activation and settle every matching interaction continuation exhaust through that binding into accepted/background result-ready state. |
| Direct background work has no DSH thread at enqueue time. | Current background priority enqueues a legacy checkpoint; DSH authority tokens are short-lived and cannot be frozen safely in a queued payload. | Queue a host-owned start specification and task-session id. Mint fresh runtime authority only when the worker claims the job, then attach the admitted thread before execution. |
| RAG3/local-context remains a live evidence owner. | `cognition_resolver.capabilities.run_rag_evidence_for_persona_state` calls `resolve_local_context`; shared-memory prewarm, consolidation, Cache2, recall, person, memory, conversation, and web evidence have surviving callers. | Retain `local_context_resolver`, Cache2, the live RAG evidence leaves, `persona_supervisor2_rag_projection.py`, and their model routes/tests. |
| The old RAG2 supervisor graph is dead production code. | Its initializer/dispatcher/evaluator/supervisor modules call only each other; `rag.quote_aware_sequence` is reached only by its retired script/tests. | Delete the exact RAG2 graph and quote-aware executable files listed below; update retained RAG documentation and mixed tests. |
| Complex-task resolver is reachable only from legacy task specialists. | External production imports are `task_resolution/specialists/public_research.py` and `text_computation.py`. | Delete the complete complex-task runtime and its dedicated tests/fixtures. |
| Public HTTP image inspection is a real legacy capability without a Plan 2 DSH owner. | `complex_task_resolver/subagent/media.py` performs bounded public URL fetch, redirect/DNS/IP validation, MIME/magic/decode/size checks, and calls the retained vision inspector. Plan 2 `kazusa_inspect_attached_media` accepts only an opaque attached-media ref, while the pinned DSH route is text-only and cannot replace this with native `read_image`. | Port the existing safety/vision boundary into the additive `kazusa_inspect_public_media` semantic tool before deleting the complex resolver. |
| Coding agent is reachable only from legacy task/coding continuation surfaces. | External production imports are the task coding specialist, the `task_orchestrator` continuation branch, coding-result projection, route reporting, action affordances, and benchmark script. | Replace the product intents with DSH task controls, then delete the complete coding runtime, benchmark, routes, fixtures, and dedicated tests. |
| Obsolete model routes are isolated. | `BACKGROUND_WORK_LLM_*` is used only by the old task orchestrator/text specialist. `CODING_AGENT_*` is used only by the old coding runtime and its continuation materialization. `RAG_PLANNER_LLM_*`, `RAG_SUBAGENT_LLM_*`, and `WEB_SEARCH_LLM_*` remain live in RAG3/evidence services. | Delete only `BACKGROUND_WORK_LLM_*` and `CODING_AGENT_*`; retain all live RAG/web route bundles. |
| Accepted/background persistence carries coding-era fields. | `accepted_task.v2`, `background_work_job.v2`, action results, cognition episodes, DB indexes, and prompt-safe action context carry `coding_run_context`, `coding_run_ref`, or `continue_bound_coding_run`. | Remove those runtime/prompt/persistence writers and readers. Retain `TaskResolutionResultV1.coding_run_context` as the predecessor-mandated empty field only; post-cutover validation rejects a non-empty value. |

### Approval-Time Executability Audit

The 2026-08-29 planning audit used baseline commit
`59357e591f762f46b7492f12be42752daff25632` and changed only this plan plus
the lifecycle registry. It established:

- 3 planned create paths are absent, and all 47 planned modify paths exist;
- all 117 production/script deletion paths exist and are unique;
- all 124 test/fixture deletion paths exist, are tracked, and are unique;
- all 36 governed config/fixture/documentation artifacts exist;
- the 327-row source/governed-artifact matrix has 327 unique exact paths and
  seven required data columns per row;
- every repository path in the plan uses forward slashes and every Markdown
  fence is paired; and
- the pinned public-media URL returned HTTP 200, `image/png`, 1,129,137 bytes,
  and the exact local SHA-256 recorded in the live-E2E section.

Planned P3-P1 nodes intentionally do not exist yet; their exact collection is
the tests-first red gate. This approval audit changed no production source,
test, fixture, database, process, or deployment state.

## Mandatory Skills And Rules

Implementation applies these repository skills before the affected work:

- `.agents/skills/development-plan` for lifecycle, evidence, amendments, and
  final sign-off;
- `.agents/skills/local-llm-architecture` for prompt-visible facts, DSH/Brain
  semantic ownership, bounded live latency, and inspectable handoffs;
- `.agents/skills/no-prepost-user-input` for user reply, permission, approval,
  continuation, and accepted-task control semantics;
- `.agents/skills/py-style` before every Python edit;
- `.agents/skills/cjk-safety` when a changed Python string contains CJK text;
- `.agents/skills/test-style-and-execution` before changing or running tests;
- `.agents/skills/character-test` before the real debug-channel interaction
  demonstration, including per-turn artifact and log inspection;
- `.agents/skills/chinese-translation` when updating `README_CN.md` or Chinese
  product prose.

Repository `AGENTS.md`, Plan 1, Plan 2, the DSH architecture ICDs, and this
plan are binding. A newly discovered production caller, durable writer, route,
or behavior expands the change radius only through a recorded plan amendment
and explicit user approval before its production edit.

## Must Do And Deferred

### Must Do

1. Route every new production `task_resolution_request` through the one shared
   Plan 2 `AgenticResolverRuntime` and the exact V2 task/binding contracts in
   this plan.
2. Preserve cognition recurrence, character-owned internal DSH interaction
   judgment, the Plan 1/Plan 2 identity/fencing/tool contracts, retained RAG3
   evidence owners, and retained `future_speak` behavior.
3. Make foreground completion, cooperative checkpoint promotion, direct
   background start, internal DSH interaction continuation, terminal replay,
   cancellation, and normal result delivery durable and exactly-once at their
   existing ownership boundaries.
4. Replace every still-supported coding-task product intent with the typed DSH
   task routes below, then remove every enumerated legacy executor, import,
   route, configuration field, fixture, test, and authoritative document.
5. Require the tests-first collection gate, exact source-to-test ownership,
   zero legacy drain, atomic readiness/smoke sequence, and independent final
   review before release sign-off.
6. Demonstrate a real two-turn user interaction through the running public
   Brain `/chat` transport and a registered debug-platform callback adapter:
   the first user turn resolves a known user-controlled prerequisite through
   ordinary character cognition before DSH admission; the second user turn
   supplies the prerequisite, background DSH work starts, and the result
   returns through normal cognition, dialog, dispatcher, and delivery.
7. Close every supported task, research, computation, media, coding,
   clarification, approval, checkpoint, follow-up, cancellation, and delivery
   capability in the functional port ledger below. Internal mechanics that are
   removed instead of ported require the exact justification and replacement
   evidence recorded below.

### Deferred DSH Integration Work

None. Plan 3 is the final DSH integration stage. Closure permits no later DSH
integration plan, unresolved port, compatibility tail, disabled capability,
or release waiver. A discovered requirement needed to satisfy a row below is
Plan 3 scope and must be added through the amendment gate before its edit; it
cannot be moved to future work. An unrelated product feature remains outside
this integration and cannot enter Plan 3.

### Final Retained Boundaries

These are complete, current owners rather than deferred integration work:

| Retained boundary | Final disposition and reason |
|---|---|
| `TaskResolutionResultV1` outward carrier | Preserve the predecessor-mandated Brain observation contract; `coding_run_context` is always `{}` and a non-empty value is rejected. A new carrier adds no missing DSH function. |
| Plan 2 sidecar, Standard profile, native-tool policy, RPC/intake epochs, model route, and thirteen semantic tools | Retain the completed, pinned DSH execution substrate and all thirteen tool contracts. Add exactly one Plan 3 public-media tool and let the canonical catalog digest reflect the fourteen-row projection. |
| Historical terminal legacy task/coding records | Retain as immutable history after the zero-active drain; the new runtime neither converts nor interprets them. Their supported live functions are ported to DSH before cutover. |
| RAG3/local-context, dialog, adapters, memory, consolidation, scheduler, reflection, and `future_speak` | Retain their live owners and exact regression anchors; Plan 3 changes only the task edge that consumes or feeds them. |
| UI | Existing adapters and control surfaces remain the user interface. The public `/chat` plus callback demonstration proves the integration without adding a DSH-specific UI. |

### Complete Functional Port Ledger

Every supported production behavior of the removed stack has one final DSH-era
owner and deterministic plus live acceptance evidence. The implementation may
delete a legacy internal function only after its row is green.

| Legacy supported function | Final Plan 3 owner | Required acceptance evidence |
|---|---|---|
| Inline task execution within the cognition budget | Shared `AgenticResolverRuntime.resolve(...)`, foreground binding, and V2-to-V1 projection | Inline deterministic integration plus real-LLM inline E2E; all terminal states recur through cognition. |
| Inline timeout promotion without lost work | Cooperative `resolution.request_checkpoint`, durable binding, and accepted/background continuation | Budget-race, checkpoint, cold-resume, and exactly-once delivery tests preserve one session. |
| Direct background task start | V2 host start specification claimed by the existing background worker, with fresh DSH authority on claim | Enqueue/claim/retry tests and the revised real two-turn debug-user E2E. |
| Task resume, replay, restart recovery, and cancellation | DSH `resolution.continue`, `resolution.inspect`, replay, lease fencing, and `resolution.cancel` wrappers | Deterministic process, live Mongo race, and sidecar restart anchors. |
| Local/private contextual evidence | Retained `local_context_resolver`, RAG3 leaves, and Plan 2 `kazusa_*` semantic tools | RAG3 retention/static checks and semantic-tool regression suites. |
| Public/current research and URL reading | DSH Standard native web tools plus storage-independent `kazusa_*` evidence tools | Public-research real-LLM E2E and native web/tool trace inspection. |
| Deterministic text computation and structured transformation | DSH Standard native computation, filesystem, and shell tools under the pinned profile | Inline/background deterministic projection tests and native coding/computation live traces. |
| Attached and public image/media inspection | Preserve Plan 2 `kazusa_inspect_attached_media`; add Plan 3 `kazusa_inspect_public_media` using the retained media-inspection service and the legacy bounded public-fetch safety checks | Fourteen-tool catalog, URL/redirect/byte/decode safety tests, and a live trace proving public-media evidence reaches the terminal without the complex resolver. |
| Complex research decomposition, follow-up work, synthesis, evidence receipts, and partial/blocker reporting | DSH autonomous multi-tool resolution, sole `submit_resolution` terminal, and `DSHResolutionExhaustV2` evidence receipts | Controller/runtime/evidence suites plus public-research live E2E. |
| Source intake from repository, local scope, pasted code, and public GitHub material | DSH Standard workspace/filesystem and native web tools; semantic source text remains in the objective/facts contract | Native coding live E2E inspects source/tool receipts and scope authority. |
| Code reading, new-file writing, existing-file modification, patching, execution, tests, verification, and bounded repair | DSH Standard native coding/filesystem/shell/test tools with character-owned full-loop approvals | Native coding internal-approval/edit/test/artifact live E2E and deterministic approval/sandbox anchors. |
| Durable coding task start and later revise, summarize, status, approve/verify, blocker response, and cancel intents | Generic DSH task binding plus `accepted_task_control`, the V2 internal interaction bridge, and existing accepted-task status action | Exact six-intent deterministic coverage plus delivered follow-up live E2E, all on one opaque DSH session and with no pending-user interaction state. |
| Pre-admission user prerequisite and execution-time DSH question/approval/plan review | Ordinary full cognition resolves a known user-controlled prerequisite before admission; the authenticated V2 bridge reuses the full cognition loop for every later internal DSH interaction | Revised two-turn debug-user E2E, full-loop internal-decision tests, exact one-shot-grant tests, and proof of zero relay/reply/checkpoint vocabulary. |
| Prompt-safe task result, cognition recurrence, final character wording, and adapter delivery | `TaskResolutionResultV1`, canonical `tool_result` episode, full cognition recurrence, dialog, dispatcher, and delivery receipt | Cross-boundary suite and visible live responses with trace/task/job/delivery lineage. |

### Justified Legacy Removals

These removals eliminate superseded mechanics or duplicate dead owners after
the functional port ledger is green; they remove no supported user function.

| Removed surface | Justification | Required proof before deletion |
|---|---|---|
| Four-specialist task selector, checkpoint, orchestrator, and specialist modules | DSH owns the same inline/background/restart/task outcomes through one canonical controller and durable session. Retaining the graph would create a second execution route. | Every task row above is green; import/decommission tests find zero caller. |
| Complex-task graph, subagent dispatcher, media helper, evidence graph, and synthesis implementation | Its public research, computation, media, decomposition, evidence, and synthesis behaviors are ported to DSH native tools, semantic tools, and terminal receipts. Its only production callers are removed task specialists. | Public research/media/computation tests pass and static import analysis finds zero surviving caller. |
| RAG2 supervisor graph and quote-aware executable | Current production cognition already uses RAG3/local-context; RAG2 is a dead duplicate graph reached only by retired scripts/tests. The live RAG3 leaves and prewarm owners remain. | Positive RAG3 retention tests and negative RAG2 import/source checks pass. |
| Kazusa coding-agent packages, coding-run ledger runtime, benchmark script, and coding-specific routes | Every supported source/read/write/patch/execute/test/repair/approval/follow-up behavior is ported to DSH Standard and generic task controls. DSH SQLite plus task bindings become the sole live durable session. | Coding and six-intent deterministic/live rows pass; the legacy drain reports zero active ledger. |
| `BACKGROUND_WORK_LLM_*` and `CODING_AGENT_*` configuration | Only removed legacy executors consume these fields; Plan 2's six-field `AGENTIC_RESOLVER_LLM_*` route is the sole task model owner. | Config/report tests prove exact removal and retained RAG/web routes. |
| Dedicated legacy tests, fixtures, and authoritative ICDs | They assert removed interfaces and would preserve a false second contract. Replacement Plan 3 nodes own every retained behavior and exact absence rule. | The deletion-artifact matrix, collection gate, manifest validation, and documentation tests pass. |
| Runtime conversion or interpretation of historical legacy records | The deployment gate drains all active work while the old release is alive. Terminal records remain historical evidence; adding a converter would create an unneeded compatibility vocabulary and data risk. | All five drain categories are zero and historical rows remain unchanged. |

## Target Ownership And Runtime Flow

```text
adapter/debug client
  -> Brain queue/intake
  -> Cognition V3 P-stage
     -> action_requests ---------------------------------> retained action path
     -> resolver_requests(task_resolution_request)
        -> cognition_resolver (retained recurrence owner)
           -> task_resolution service (Brain/DSH boundary)
              -> durable DSH task binding
              -> shared process-owned AgenticResolverRuntime
                 -> ResolutionController
                 -> independent DSH Standard sidecar
                 -> native Standard + fourteen kazusa_* semantic tools
                 -> submit_resolution | checkpoint | fault | canceled
              <- validated DSHResolutionExhaustV2
              -> deterministic TaskResolutionResultV1 projection
           -> ResolverObservationV1
        -> full cognition recurrence
  -> existing state commit -> actions -> L3/dialog
  -> existing consolidation/post-turn work

deferred/background branch
  -> accepted_task + background_work job
  -> task_orchestrator worker payload v2
  -> same durable DSH task binding and runtime thread
  -> result-ready tool_result cognition episode
  -> normal dialog -> dispatcher -> adapter delivery

DSH question/approval/plan-review branch
  -> authenticated Brain interaction V2 boundary
  -> existing full reusable cognition loop and final cognition commit
  -> answer/reject/allow_once returned internally to the waiting DSH hook
  -> exact same-thread DSH execution continues without user wait or dialog
```

### Semantic And Mechanical Ownership

| Decision or mechanism | Owner after cutover |
|---|---|
| Whether a task needs resolution; its objective, priority, response goal, and accepted-task control intent | Cognition P-stage/L2d |
| Character stance, sufficiency, DSH question/approval/plan-review judgment, recurrence, and final wording goal | Full reusable cognition loop |
| Final visible language | L3/dialog |
| DSH planning, native tool choice, semantic-tool choice, coding, public web, and terminal content | DSH Standard |
| Intake shape, fact serialization, runtime authority, identity, idempotency, fencing, task binding, leases, task state, result projection, limits, and delivery | Deterministic Python/sidecar owners |
| Evidence retrieval outside DSH live task execution | Existing RAG3/local-context and retained evidence leaves |
| Durable result wording delivery | Existing accepted-task result source, cognition, dialog, dispatcher, and adapter |

## Closed Contract Decisions

### 1. Shared Runtime Composition And Readiness

1. `service.py` constructs one `AgenticResolverRuntime` after Brain graph,
   adapter, database, and interaction owners exist. The same instance is
   injected into the Plan 2 interaction service and `task_resolution.service`.
2. Construction requires the completed Plan 2 environment bundles. The Brain
   interaction endpoint can start before the sidecar process; production task
   admission stays closed until Brain interaction health and sidecar
   `system.health` are both ready with matching V2 route, profile, catalog,
   policy, workspace, and store epochs.
3. Task capability availability uses that readiness owner. A race after
   selection produces typed `unavailable`; it never invokes a legacy path.
4. `AgenticResolverRuntime` adds generic public wrappers for
   `resolution.amend`, `resolution.request_checkpoint`, fenced continuation of
   a checkpoint or terminal thread with fresh authority, `resolution.cancel`, and
   `resolution.inspect`. The wrappers return validated mappings/exhausts and
   retain the controller's existing idempotency, lease, and segment rules.
5. DSH packages, Standard preset/native tools, profile version, policy, RPC
   schema, store epoch, and model route stay unchanged. Sidecar catalog and
   forwarding code changes only for the additive fourteenth semantic tool;
   startup requires the matching new semantic catalog digest on both sides.

### 1A. Public Media Semantic Port

Plan 3 adds exactly this storage-independent tool while preserving the original
thirteen catalog rows byte-identically:

`kazusa_search_conversation_history`, `kazusa_read_conversation_entries`,
`kazusa_summarize_conversation_participants`, `kazusa_search_memories`,
`kazusa_read_memories`, `kazusa_remember_information`,
`kazusa_revise_memory`, `kazusa_change_memory_lifecycle`,
`kazusa_find_people_by_name`, `kazusa_read_person_profiles`,
`kazusa_recall_active_context`, `kazusa_read_calendar_context`, and
`kazusa_inspect_attached_media`.

```text
name = kazusa_inspect_public_media
input_schema.additionalProperties = false
public_media_url = required non-empty HTTP(S) URL string
question = required non-empty semantic question string
```

`MediaSemanticService.inspect_public_media(...)` owns the Brain worker side.
It moves, rather than duplicates, the retired media subagent's exact bounded
intake protections: HTTP(S) only; no URL credentials or fragments; DNS
resolution rejects private, loopback, link-local, multicast, reserved, and
unspecified addresses; each of at most three redirects is resolved and checked
again; timeout is 15 seconds; a declared `Content-Length` above 6 MiB is
rejected and an undeclared/chunked response is streamed with an immediate stop
at byte 6 MiB + 1; accepted MIME/magic pairs are PNG, JPEG, GIF, and WebP;
Pillow decode must succeed; and each dimension is between 1 and 8192 pixels.
Only then may it base64-encode the image and call the retained
`media_inspection_request.v1` service with
`source=dsh_public_media`.

The semantic result uses the existing
`KazusaSemanticCapabilityResultV1`/`evidence_receipt.v2` envelope. Its entity
contains only `status`, `answer`, `source_url`, `content_type`, `byte_count`,
and `evidence_boundary_notes`; its receipt has
`source_kind=public_media`, a digest-derived opaque `semantic_ref`, and the
current authority/catalog fields. Failures expose a bounded typed code and
semantic reason without response bytes, addresses, exception text, or vision
prompt internals. The DSH sidecar forwards only the exact two input fields.

The deterministic failure map is closed:

| Condition | Semantic status | Error code |
|---|---|---|
| Missing/blank question | `invalid` | `QUESTION_REQUIRED` |
| Malformed/non-HTTP(S)/credentialed/fragment URL | `invalid` | `PUBLIC_MEDIA_URL_INVALID` |
| Any initial or redirect target resolves to a non-public address | `denied` | `PUBLIC_MEDIA_URL_DENIED` |
| Redirect lacks a target or exceeds three hops | `invalid` | `PUBLIC_MEDIA_REDIRECT_INVALID` |
| Fetch timeout | `timeout` | `PUBLIC_MEDIA_FETCH_TIMEOUT` |
| DNS/transport/HTTP failure | `unavailable` | `PUBLIC_MEDIA_FETCH_UNAVAILABLE` |
| Body exceeds 6 MiB | `invalid` | `PUBLIC_MEDIA_TOO_LARGE` |
| MIME/magic pair is unsupported or mismatched | `invalid` | `PUBLIC_MEDIA_TYPE_INVALID` |
| Decode fails or dimensions are outside bounds | `invalid` | `PUBLIC_MEDIA_DECODE_INVALID` |

For a valid fetch, compute
`media_identity=content_digest({"final_url": final_url,
"image_sha256": sha256(image_bytes)})`, use
`semantic_ref=public-media:<media_identity-without-sha256-prefix>`, and use
`receipt_id=receipt-public-media-<same-digest>`. Raw image bytes and base64
remain inside the media service call and never enter the semantic result,
receipt, task binding, trace summary, or dialog context.

The semantic catalog digest intentionally changes. Existing terminal or
checkpointed V2 threads without an open interaction continue through Plan 1's
canonical compatibility rotation into a fresh segment carrying the new
fourteen-tool digest. Old authority and grants fail closed. Deployment drains
every open pre-cutover DSH interaction/grant before the atomic Brain/sidecar
switch, so no actionable V1 relay/reply state crosses the V2 boundary.

### 2. Task Execution Context And DSH Intake

`TaskResolutionExecutionContextV2` replaces the V1 specialist context in one
release. It has these exact fields:

```text
schema_version = task_resolution_execution_context.v2
character_name
platform
channel_id
channel_type
requester_global_user_id
requester_platform_user_id
requester_display_name
source_message_id
source_platform_bot_id
source_trigger_source
source_llm_trace_id
brain_conversation_ref
scene_context
goal_continuation_ref
local_time_context
prompt_message_context
chat_history_recent
chat_history_wide
conversation_progress
persona_summary
conversation_summary
current_timestamp_utc
active_turn_platform_message_ids
active_turn_conversation_row_ids
session_media_refs
max_output_chars
```

The V2 context removes `coding_workspace_root`. The model-hidden runtime
authority obtains the canonical workspace only from
`AGENTIC_RESOLVER_WORKSPACE_ROOT`.

The four added source fields are projected only from already trusted state:
`requester_display_name=state.user_name`,
`source_platform_bot_id=state.platform_bot_id`,
`source_trigger_source=cognitive_episode.trigger_source`, and
`source_llm_trace_id=state.llm_trace_id` (the existing empty string remains
valid when no trace is authorized). `brain_conversation_ref` is exactly the
current validated `cognitive_episode.episode_id`; a promoted or follow-up task
retains that original ref from its start spec rather than substituting the
later control message.

The DSH model sees the validated `ResolverCapabilityRequestV2.semantic_goal` as
`model_input.objective`. Each fact is its literal ASCII prefix followed by the
UTF-8 text returned by the existing
`canonical_json(value).decode("utf-8")` helper, in exactly this list order.
The index column orders the list and is not emitted into the fact string.

| Index | Literal prefix | Value passed to `canonical_json` |
|---|---|---|
| 1 | `character_and_scene=` | `{"character_name": character_name, "scene_context": scene_context}` |
| 2 | `local_time=` | `local_time_context` |
| 3 | `current_message_context=` | `prompt_message_context` |
| 4 | `recent_conversation=` | `chat_history_recent` |
| 5 | `wide_conversation=` | `chat_history_wide` |
| 6 | `conversation_progress=` | `conversation_progress` |
| 7 | `persona_summary=` | `persona_summary` |
| 8 | `conversation_summary=` | `conversation_summary` |
| 9 | `active_turn_lineage=` | `{"conversation_row_ids": active_turn_conversation_row_ids, "platform_message_ids": active_turn_platform_message_ids}` |
| 10 | `attached_media_refs=` | `session_media_refs` |

Serialization clips only at the existing validated carrier bounds and performs
no semantic filtering, keyword classification, or user-text reinterpretation.
The runtime keeps platform/channel/user scope, source ids, continuation ref,
workspace, output limit, route, audience, and authority outside model facts.
The `objective_ref` is exactly
`content_digest(goal_continuation_ref)`, including its `sha256:` prefix;
`brain_conversation_ref` comes from the context. The audience object has exact
keys and values `kind=kazusa_task_resolution`,
`goal_continuation_ref_digest=objective_ref`, and
`requested_delivery=send_result_when_done`.

### 3. Durable DSH Task Binding

Create collection `dsh_task_bindings` with schema `dsh_task_binding.v1` and
one DB owner, `kazusa_ai_chatbot.db.task_resolution_sessions`. Each row has:

```text
schema_version = dsh_task_binding.v1
task_session_id                         unique opaque id
semantic_objective
goal_continuation_ref
source_scope                           validated DshTaskSourceScopeV1
state                                  queued | opening | active | checkpointed |
                                       terminal | canceled | faulted |
                                       consumed_inline
start_spec                             validated DshTaskStartSpecV1
resolution_thread_id                   null until authority preparation
segment_id                             null until authority preparation
resolution_ref                         null or DshResolutionRefV1
operation_generation                   monotonic integer, initially 0
current_accepted_task_id               null until durable promotion/follow-up
current_background_work_job_id         null until durable promotion/follow-up
latest_task_resolution_result          null or validated TaskResolutionResultV1
revision                               monotonic compare-and-set integer
created_at
updated_at
```

`DshTaskSourceScopeV1` is an exact-key object containing `platform`,
`channel_id`, `channel_type`, `requester_global_user_id`,
`requester_platform_user_id`, `source_message_id`, and
`source_platform_bot_id`, copied from the validated V2 context.

`DshTaskStartSpecV1` contains exactly:

```text
schema_version = dsh_task_start_spec.v1
resolver_request                       validated ResolverCapabilityRequestV2
execution_context                      validated TaskResolutionExecutionContextV2
model_facts                            the exact ordered ten strings above
model_facts_digest                     content_digest(model_facts)
objective_ref                          content_digest(goal_continuation_ref)
```

The resolver request `capability` is `task_resolution_request`, its
`semantic_goal` equals `semantic_objective`, its `start_in_background` boolean
equals the upstream V1 `priority == background` projection, its non-null
`goal_continuation_ref` equals the context and binding value, and the binding
source scope equals the context projection.
The start spec contains no RPC token, semantic authority token, Brain secret,
gateway secret, database handle, or adapter object.

`DshResolutionRefV1` contains exactly:

```text
schema_version = dsh_resolution_ref.v1
resolution_thread_id
segment_id
dsh_session_id
activation_id
lease_epoch
document_revision
last_committed_seq
```

Bootstrap creates these exact indexes and no TTL:

| Index name | Keys/options |
|---|---|
| `dsh_task_binding_session_unique` | unique `(schema_version, task_session_id)` |
| `dsh_task_binding_thread_unique` | unique `resolution_thread_id`, partial where it is a string |
| `dsh_task_binding_current_accepted_task_unique` | unique `current_accepted_task_id`, partial where it is a string |
| `dsh_task_binding_current_background_job_unique` | unique `current_background_work_job_id`, partial where it is a string |
| `dsh_task_binding_state_updated` | non-unique `(state, updated_at)` |

The binding state machine permits only these revision-guarded transitions:

```text
queued                  -> opening | canceled | faulted
opening                 -> checkpointed | terminal | canceled | faulted |
                           consumed_inline
checkpointed            -> active | canceled | faulted
active                  -> checkpointed | terminal | canceled | faulted
terminal                -> active | canceled | terminal
canceled                -> canceled
faulted                 -> faulted
consumed_inline         -> consumed_inline
```

The self-transitions accept only an identical idempotent replay. Attaching an
accepted task/job or reconciling a stored result updates the revision. Only a
validated DSH follow-up increments `operation_generation` and reopens
`terminal` as `active`; its accepted task/job/result writes must carry that
exact generation. A mismatched thread, segment, generation, result digest,
accepted task, job, or expected revision fails closed.

The binding is created before any sidecar activation. Runtime authority is
prepared next, and its thread/segment ids are attached before the RPC open call
so every internal interaction retains the same product binding. Attaching
a checkpoint, accepted task, job, or terminal result is idempotent and revision
guarded. A terminal result that arrives before promotion is stored on the
binding; promotion reconciles it immediately into the attached job/task. A
matching result after promotion settles the job/task exactly once.

### 4. Foreground, Direct Background, And Cooperative Promotion

| Input path | Exact behavior |
|---|---|
| `priority=now` | Create binding and stable runtime identity, start `resolution.open`, and wait for `TASK_RESOLUTION_INLINE_BUDGET_SECONDS` with the resolver coroutine shielded from cancellation. If terminal wins, map it inline and mark the binding `consumed_inline`. If the budget wins, call `resolution.request_checkpoint` on the active fence, await the committed terminal-or-checkpoint race result, and promote only a committed checkpoint. A concurrently committed terminal remains inline. |
| `priority=background` | Create-or-return the scoped accepted task first. If it is new, create binding generation 0, attach the task, mark it pending, enqueue `operation=open_dsh_resolution`, and attach the job. If it is already active, require matching goal/scope and reuse its existing DSH binding/job without creating a session. The claimed worker mints fresh short-lived authority, attaches the admitted thread, and opens DSH. Tokens are never queued. Any pre-enqueue failure moves the new accepted task to `enqueue_failed` and the new binding to `faulted`. |
| foreground checkpoint promotion | Reuse the existing binding and DSH reference; create-or-return the idempotent accepted task, mark it pending, enqueue generation 0 `operation=continue_dsh_resolution`, attach the job, and reconcile an already-arrived terminal result. If the duplicate accepted task already owns another matching DSH binding, cancel/dispose the redundant newly checkpointed session, mark its binding `canceled`, and return the existing accepted task/job. |
| delivered-task follow-up | Resolve the advertised accepted-task ref to its model-hidden `dsh_task_session_id` and require the one open follow-up affordance and a terminal binding. Continue/summarize claims that affordance by action-attempt id under compare-and-set, deterministically creates-or-returns the continuation accepted task/job, increments and attaches `operation_generation`, and enqueues `continue_dsh_resolution`; a retry resumes the same claimed operation and ids. Cancel closes the terminal follow-up locally under the same idempotency guard. Continuation issues fresh authority for the same DSH thread and segment and never opens a replacement session. |
| worker cooperative checkpoint | Persist the updated DSH ref under the job lease and return the job to queued state for a later continuation attempt. An internal DSH question/approval/plan review completes inside the active call and never creates a task wait state. |

The 30-second default remains a Brain caller-wait budget. It is not a DSH
reasoning timeout, and expiration triggers cooperative checkpoint rather than
cancellation.

### 5. DSH Exhaust To `TaskResolutionResultV1`

Plan 2 explicitly deferred this mapping to Plan 3. The V1 Brain handover stays
the stable outward carrier, with this closed mapping:

| DSH exhaust | `TaskResolutionResultV1.status` | `evidence_state` | Product disposition |
|---|---|---|---|
| terminal `resolved` | `resolved` | `complete` | Inline recurrence or accepted result-ready |
| terminal `partial` | `partial` | `partial` | Inline recurrence or accepted result-ready |
| terminal `needs_user_input` | `needs_user_input` | `pending` | Normal cognition owns visible clarification; no deterministic reply classification |
| terminal `approval_required` | `approval_required` | `pending` | Normal cognition owns visible approval goal; no grant is synthesized |
| terminal `unavailable` | `unavailable` | `missing` | Typed unavailable; no fallback |
| terminal `failed` | `failed` | `blocked` | Typed failure; no fallback |
| `checkpointed` | `deferred` | `pending` | Persist DSH ref and use accepted/background continuation |
| `runtime_fault` | `unavailable` | `missing` | Persist sanitized typed fault internally; expose bounded unavailable summary |
| `canceled` | `failed` inline; accepted task `cancelled` when control-owned | `blocked` | No result delivery for an explicitly canceled accepted task |

Projection rules are structural:

1. `semantic_objective`, `scene_context`, and `goal_continuation_ref` come from
   the validated start spec, never from the DSH terminal.
2. `prompt_safe_summary`, `completed_subgoals`, `remaining_needs`, and warnings
   come from `SubmitResolutionV2` under existing text/list bounds.
3. Findings become bounded canonical-JSON `evidence_excerpts`; deterministic
   code performs no truth correction or semantic rewriting.
4. `evidence_handles` are ordered, deduplicated semantic refs from
   `EvidenceReceiptV2`, followed by validated artifact refs, within the V1
   bound.
5. Each projected `TaskResolutionEvidenceV1` has `specialist=dsh`, summary equal
   to its semantic ref, provenance refs containing the receipt evidence id and
   content digest, and structural limitations only. Old specialist names are
   rejected after cutover.
6. `checkpoint` is empty for terminal results and contains only a validated
   `DshResolutionRefV1` for `deferred`.
7. `coding_run_context` is always `{}`. The predecessor-mandated V1 field stays
   present, while every non-empty value and every runtime reader/writer is
   rejected or deleted.

### 6. Accepted/Background Contracts

1. `accepted_task.v2` and `background_work_job.v2` remain the collection
   epochs because retained `future_speak` uses them. Plan 3 changes only new
   task-resolution writes and their task payload contract.
2. Every new DSH-backed `task_resolution` accepted-task row has model-hidden
   fields `dsh_task_session_id`, `dsh_operation_generation`,
   `dsh_followup_open`, and `dsh_followup_claim_action_attempt_id`. They are
   absent from prompt/action results. The last delivered row for a terminal
   session is the sole `dsh_followup_open=true` row; a partial unique index
   `accepted_task_open_dsh_followup_unique` on `dsh_task_session_id` enforces
   that invariant, and `accepted_task_scope_dsh_followup_lookup` supports the
   existing trusted scope fields plus `updated_at`.

   | Index name | Keys/options |
   |---|---|
   | `accepted_task_open_dsh_followup_unique` | unique `dsh_task_session_id`, partial where `schema_version=accepted_task.v2`, `task_kind=task_resolution`, and `dsh_followup_open=true` |
   | `accepted_task_scope_dsh_followup_lookup` | non-unique `(schema_version, source_platform, source_channel_id, source_channel_type, requester_global_user_id, requester_platform_user_id, dsh_followup_open, updated_at)` |

3. `DshAcceptedTaskAffordanceV1` is the only prompt-safe follow-up projection
   and contains exact keys `schema_version=dsh_accepted_task_affordance.v1`,
   `accepted_task_ref`, `task_state`, `objective_summary`, `latest_summary`,
   `allowed_next_actions`, `followup_open`, and `updated_at`. It contains no
   thread, segment, session, activation, job, workspace, or authority value.
   An active queued/running/waiting task advertises only `cancel`; a delivered
   row with a terminal binding advertises `continue|summarize|cancel`; other
   rows advertise no task control. `accepted_task_status_check` reads the
   latest active row, or the sole open DSH follow-up row when no active row
   exists.
4. A continue/summarize action claims the advertised open row with its exact
   action-attempt id, creates a new `task_resolution` accepted task and job for
   the same model-hidden session, advances the binding generation, and closes
   the old affordance. Retries with the same attempt resume the same ids;
   failure before generation attachment reopens only that claimed affordance.
   The new row becomes the sole open affordance only after its result is
   delivered. This preserves repeated same-session follow-up without reopening
   or overwriting an already delivered accepted-task record.
5. `task_orchestrator_worker_payload.v2` replaces V1 and has exact keys
   `schema_version`, `operation`, `task_session_id`, `operation_generation`,
   and `control`. `operation` is `open_dsh_resolution` or
   `continue_dsh_resolution`. `control` is null for an initial open,
   checkpoint promotion, or mechanical checkpoint retry and is a validated
   continue/summarize `AcceptedTaskControlV1` for a product follow-up.
6. The worker rejects payload V1, legacy checkpoints, coding requests, coding
   run refs, and unknown fields. Deployment drain guarantees no claimable V1
   task job remains.
7. Add `canceled` to the task-job terminal states. The V2 interaction bridge
   creates no accepted-task or background-job user-wait state.
8. Cancel is a fenced task-service control, not a second concurrent worker
   job. For queued work it prevents open and terminalizes the accepted task/job
   locally; for an active activation it invokes `resolution.cancel` on the
   current fence and then terminalizes them; for a delivered terminal session
   it closes the follow-up affordance and binding locally because the DSH
   activation is already disposed. Every path is action-attempt idempotent and
   emits no background result delivery.
9. Existing accepted-task identity, enqueue interruption recovery, worker
   lease, retry, result-ready, delivery claim, dispatcher, adapter receipt,
   and retryable delivery behavior stay intact.
10. `background_work.result_source` projects the validated V1 task result into
   the existing canonical `tool_result` cognition episode with the exact goal
   continuation ref and DSH evidence provenance. It contains
   `task_resolution_context`, not `coding_run_context`.
11. `future_speak` payloads, worker, authority, and delivery behavior remain
   unchanged.

### 7. Legacy Coding Intent Carryover

The old model capability `accepted_coding_task_request`, its
`coding_run:<run_id>` vocabulary, and its six deterministic coding-run actions
are removed atomically. Product intent is preserved through the canonical
`accepted_task_control` action and the V2 internal interaction bridge:

| Existing user/product intent | DSH-era semantic route |
|---|---|
| Start a coding task | Existing resolver capability `task_resolution_request`; DSH Standard owns native coding and verification. |
| Revise proposal | Cognition selects `accepted_task_control(operation=continue)` with the semantic revision instruction and one advertised opaque `accepted_task:<id>` ref. |
| Summarize | Cognition selects `accepted_task_control(operation=summarize)`; the background continuation asks the same DSH session for a bounded progress summary. |
| Status | Existing `accepted_task_status_check` reads the latest scoped active task or sole open DSH follow-up and its prompt-safe latest summary. |
| Approve and verify | Cognition may select `accepted_task_control(operation=continue)` with a verification goal; that control grants no permission. Every native mutation still reaches the V2 Brain bridge and the full cognition loop decides `allow_once` or `reject` internally. |
| Respond to blocker | The full cognition loop answers or rejects an execution-time DSH question internally. A later user-authored task revision may independently select `accepted_task_control(operation=continue)` with its semantic instruction. |
| Cancel | Cognition selects `accepted_task_control(operation=cancel)`; deterministic code validates the advertised task ref and idempotently cancels queued/active work or closes an already-terminal follow-up according to the exact lifecycle rule above. |

`AcceptedTaskControlV1` has exact fields:

```text
schema_version = accepted_task_control.v1
accepted_task_ref = accepted_task:<accepted_task_id>
operation = continue | summarize | cancel
instruction = required non-empty text for continue; null for summarize/cancel
```

The P-stage sees only scoped active task refs and the sole scoped open DSH
follow-up ref, each with state-derived `allowed_next_actions`. LLM cognition
interprets user wording and chooses the typed intent. Deterministic code
validates scope, advertised ref, lifecycle state, idempotency, and operation
shape; it contains no keyword acceptance, approval detector, user-text
rewrite, or post-LLM semantic override.

### 8. Stage Ordering And Visible Behavior

The cutover preserves:

1. P-stage separation of `action_requests` and `resolver_requests`;
2. one selected resolver capability per recurrence iteration;
3. exact goal-continuation, evidence dependency, conflict, and repeated-failure
   guards;
4. full cognition recurrence before state commit, memory, actions, L3, dialog,
   or delivery;
5. final cognition commit before private actions and visible surfaces;
6. Brain-owned clarification, approval, acknowledgement, result sufficiency,
   and final wording;
7. source-bound accepted-task tool-result cognition and adapter delivery;
8. internal-thought privacy and visible fallback target provenance; and
9. shared-memory prewarm's independent checkpoint/merge path.

## Big-Bang Cutover, Data, And Deployment Policy

### Code And Contract Cutover

The release simultaneously installs the DSH task edge, binding store, payload
V2, accepted-task control, V2 internal cognition interaction bridge,
configuration cleanup, legacy deletion, tests, and docs. Every new task starts
on DSH. Runtime code
reads no legacy task checkpoint, coding ledger/context, complex graph, RAG2
supervisor state, or task payload V1.

### Historical Data Disposition

| Data | Disposition |
|---|---|
| `resolution_threads`, DSH session SQLite, semantic outcome SQLite, and Plan 2 interaction rows | Retain the completed resolution epochs and retention rules. Closed threads remain historical; an eligible later continuation rotates to the new semantic catalog digest. Open V1 interaction/grant rows must drain before cutover; historical V1 rows are inert after the atomic V2 interaction cutover. |
| New `dsh_task_bindings` | Create with the V1 schema/indexes above; it is the only Brain product binding from DSH thread to accepted/background task. |
| Delivered/terminal legacy `accepted_tasks` and `background_work_jobs` rows | Retain as historical records. New runtime queries never interpret their legacy checkpoint/coding payloads. |
| Legacy coding run files/rows | Require every product ledger terminal at the drain, then retain it as historical operator data. Remove all runtime imports, loaders, prompts, and fallback reads. |
| Accepted-task follow-up indexes | Drop `accepted_task_open_coding_run_context_lookup` during atomic deployment bootstrap after the drain assertion; create the exact DSH follow-up indexes above. |
| Other conversations, memory, calendar, reflection, Cache2, and RAG collections | Retain unchanged. |

No data converter runs. Historical rows never become DSH sessions.

### Drain Gate

1. Enter a declared maintenance window on the old release by pausing every
   configured adapter and debug client before Brain intake. Record each client
   quiescent with zero outstanding Brain requests. This closes all new chat
   and therefore all new task/coding admission without inventing a task-only
   switch that the current release does not have.
2. Keep the old Brain, its background/delivery worker, and the old matching
   sidecar running while existing jobs and interactions settle. Run the
   read-only drain command until all five exact
   categories below are zero:
   - `active_legacy_accepted_tasks`: `accepted_task.v2` rows with `task_kind`
     in `task_resolution|coding_continuation` and `state` in
     `enqueueing|pending|running|result_ready|failure_ready|delivery_in_progress|delivery_retryable`;
   - `executing_legacy_task_jobs`: `background_work_job.v2` rows with
     `requested_worker=task_orchestrator`, payload schema
     `task_orchestrator_worker_payload.v1`, and `status` in
     `queued|in_progress`;
   - `undelivered_legacy_task_jobs`: the same job identity with `status` in
     `completed|failed|delivery_failed|delivery_in_progress` and
     `delivery_state` other than `delivered`;
   - `nonterminal_or_invalid_legacy_coding_runs`: files at exact product root
     `<legacy-coding-workspace-root>/coding_runs/*/run.json` whose status is
     `created|source_resolved|evidence_collected|proposal_ready|awaiting_approval|applying|verifying|repairing|blocked`,
     or whose ledger is unreadable, malformed, escapes the root, or has an
     unknown schema/status. Terminal statuses are exactly
     `completed|rejected|failed|cancelled`;
   - `open_pre_cutover_dsh_interactions`: `dsh_interaction_pending.v1` rows
     with `status` in `pending|delivered|continuation_pending` or
     `grant_status=available`. These carry the pre-Plan 3 semantic catalog
     digest and must settle or expire under the old Brain/sidecar pair before
     the additive catalog switch.
3. Reconfirm every adapter/debug client has zero outstanding requests, then
   gracefully stop the old Brain and old sidecar. Successful process
   termination establishes zero process-local inline executions and stops
   further legacy claims or old-catalog interactions.
4. Record client/process evidence, counts, and timestamps without document or
   ledger bodies. Any nonzero count or invalid ledger aborts deployment. The
   operator restores the old admission state after the maintenance decision,
   waits for normal completion, or obtains a separate explicit user decision
   for the specific unresolved task; this deployment procedure performs no
   forced state rewrite or deletion.

`future_speak` and unrelated accepted tasks are excluded by the exact filters.
`scripts/check_dsh_plan3_drain.py` is the canonical read-only command for the
five persisted categories. It uses named `db.script_operations` helpers for
the four Mongo counts and a contained read-only ledger scanner for the coding
run count.
It requires an explicit absolute `--legacy-coding-workspace-root`, emits only
schema version, timestamp, the five named counts, and `ready: true|false`, and
has no write or execute option.

```text
schema_version = dsh_plan3_drain_report.v1
generated_at = UTC storage timestamp
counts.active_legacy_accepted_tasks = non-negative integer
counts.executing_legacy_task_jobs = non-negative integer
counts.undelivered_legacy_task_jobs = non-negative integer
counts.nonterminal_or_invalid_legacy_coding_runs = non-negative integer
counts.open_pre_cutover_dsh_interactions = non-negative integer
ready = true exactly when every count is zero
```

### Deployment And Recovery

1. With all adapter/debug intake paused, the five-count drain green, and the
   old Brain/sidecar stopped, install the entire Plan 3 Brain and sidecar
   candidate, run Mongo bootstrap/index reconciliation, start the new Brain,
   then start the independently managed new sidecar.
2. Require Brain DSH interaction health and sidecar `system.health` to be ready
   with matching V2 identity before resuming adapter/debug intake.
3. Before adapter/debug intake resumes, a failed smoke gate restores and starts
   the previous complete Brain/sidecar release; no Plan 3 task has been
   admitted.
4. After adapter/debug intake resumes, recovery is roll-forward on the
   DSH-only contract.
   The old executor release is no longer a valid reader for new bindings or
   payload V2.
5. Sidecar restart, Brain restart, worker lease loss, duplicate claim, and an
   interaction reply arriving before/after promotion reconcile through the
   durable thread, binding, operation idempotency, and accepted/background
   state machines.

## Exact Production Change Surface

### Approved Execution Amendment — 2026-08-30

The user explicitly approved this bounded amendment after P3-P2 exposed three
closed-gate contradictions. It adds no compatibility route or product feature:

1. Direct background admission returns a transient, model-hidden
   `TaskResolutionAdmissionV1` carrying the accepted task, background job, and
   host task-session identities. Cognition projects only its prompt-safe
   accepted observation. `TaskResolutionResultV1(status=deferred)` remains
   exclusive to an actually committed DSH checkpoint containing an exact
   `DshResolutionRefV1`; claim-time authority and admitted thread identity are
   never synthesized at enqueue time.
2. `scripts/validate_test_impact.py` gains an explicit removed-source manifest
   vocabulary. Every planned deleted strict source maps to surviving exact
   decommission nodes, manifest validation requires each removed path to be
   absent, and any arbitrary unmapped changed or deleted source still fails
   closed.
3. `src/control_console/brain_model_routes.py` removes only the obsolete
   `background_work`, `coding_pm`, and `coding_programmer` LLM route
   descriptors. Background-work worker controls remain unchanged; the removed
   `BACKGROUND_WORK_LLM_*` and `CODING_AGENT_*` model routes do not survive in
   control-console configuration.
4. The read-only drain owner
   `src/kazusa_ai_chatbot/db/script_operations.py` may retain exactly two
   `task_orchestrator_worker_payload.v1` literals required by P3-G9: one
   filter for queued/in-progress execution rows and one filter for terminal
   rows whose delivery is not complete. A dedicated owner audit and
   deterministic test prove that both literals remain inside
   `count_dsh_plan3_drain_rows` with the exact worker, schema, status, and
   delivery predicates. No runtime worker or product path may use this
   vocabulary.

The amendment adds `scripts/validate_test_impact.py` and
`src/control_console/brain_model_routes.py` to the production/tooling modify
surface and `tests/test_control_console_service_config.py` plus the
deterministic drain-owner gate in
`tests/unit/scripts/test_check_dsh_plan3_drain.py` to the test rewrite surface.
The existing task-resolution contracts/service/cognition tests, manifest, and
impact-validator tests own the other changes.

### Create

- `src/kazusa_ai_chatbot/task_resolution/projection.py`
- `src/kazusa_ai_chatbot/db/task_resolution_sessions.py`
- `scripts/check_dsh_plan3_drain.py`

### Modify

- `src/agentic_resolver/runtime.py`
- `src/agentic_resolver/controller.py`
- `sidecars/dsh_resolution/src/brain_interaction.ts`
- `sidecars/dsh_resolution/src/main.ts`
- `sidecars/dsh_resolution/src/contracts.ts`
- `sidecars/dsh_resolution/src/profile.ts`
- `sidecars/dsh_resolution/src/semantic_gateway.ts`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/brain_service/contracts.py`
- `src/kazusa_ai_chatbot/dsh_interaction/__init__.py`
- `src/kazusa_ai_chatbot/dsh_interaction/contracts.py`
- `src/kazusa_ai_chatbot/dsh_interaction/decision.py`
- `src/kazusa_ai_chatbot/dsh_interaction/service.py`
- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/llm_interface/route_report.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/catalog.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/contracts.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/dispatch.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/media.py`
- `src/kazusa_ai_chatbot/media_inspection/contracts.py`
- `src/kazusa_ai_chatbot/task_resolution/__init__.py`
- `src/kazusa_ai_chatbot/task_resolution/contracts.py`
- `src/kazusa_ai_chatbot/task_resolution/service.py`
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
- `src/kazusa_ai_chatbot/accepted_task/__init__.py`
- `src/kazusa_ai_chatbot/accepted_task/models.py`
- `src/kazusa_ai_chatbot/accepted_task/lifecycle.py`
- `src/kazusa_ai_chatbot/background_work/__init__.py`
- `src/kazusa_ai_chatbot/background_work/models.py`
- `src/kazusa_ai_chatbot/background_work/jobs.py`
- `src/kazusa_ai_chatbot/background_work/worker.py`
- `src/kazusa_ai_chatbot/background_work/result_source.py`
- `src/kazusa_ai_chatbot/background_work/subagent/task_orchestrator.py`
- `src/kazusa_ai_chatbot/db/__init__.py`
- `src/kazusa_ai_chatbot/db/accepted_tasks.py`
- `src/kazusa_ai_chatbot/db/background_work_jobs.py`
- `src/kazusa_ai_chatbot/db/bootstrap.py`
- `src/kazusa_ai_chatbot/db/dsh_interactions.py`
- `src/kazusa_ai_chatbot/db/script_operations.py`
- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/action_spec/__init__.py`
- `src/kazusa_ai_chatbot/action_spec/registry.py`
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`
- `src/kazusa_ai_chatbot/action_spec/execution.py`
- `src/kazusa_ai_chatbot/action_spec/results.py`
- `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py`
- `src/kazusa_ai_chatbot/cognition_episode.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py`
- `src/kazusa_ai_chatbot/state.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`
- `src/kazusa_ai_chatbot/rag/__init__.py`
- `src/control_console/brain_model_routes.py`
- `scripts/validate_test_impact.py`

### Delete — Superseded DSH User Relay/Reply Owners

- `src/kazusa_ai_chatbot/dsh_interaction/pending.py`
- `src/kazusa_ai_chatbot/dsh_interaction/resume.py`

### Delete — Legacy Task/Complex/RAG2 Executors

- `src/kazusa_ai_chatbot/task_resolution/orchestrator.py`
- `src/kazusa_ai_chatbot/task_resolution/state.py`
- `src/kazusa_ai_chatbot/task_resolution/specialists/__init__.py`
- `src/kazusa_ai_chatbot/task_resolution/specialists/coding.py`
- `src/kazusa_ai_chatbot/task_resolution/specialists/local_context.py`
- `src/kazusa_ai_chatbot/task_resolution/specialists/public_research.py`
- `src/kazusa_ai_chatbot/task_resolution/specialists/text_computation.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/__init__.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/algorithmic.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/constants.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/contracts.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/graph.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/service.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/stages.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/subagent/__init__.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/subagent/algorithmic.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/subagent/evidence.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/subagent/media.py`
- `src/kazusa_ai_chatbot/complex_task_resolver/subagents.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_prompt_views.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_types.py`
- `src/kazusa_ai_chatbot/rag/quote_aware_sequence.py`
- `scripts/run_rag2_e2e_case.py`

### Delete — Legacy Coding Runtime

Every Python source under `src/kazusa_ai_chatbot/coding_agent/` is listed
individually in the source-to-test matrix below and is deleted. Its package
READMEs are deleted with the directory. Also delete:

- `scripts/run_coding_agent_benchmark.py`

### Explicitly Retain

- `src/kazusa_ai_chatbot/local_context_resolver/`
- `src/kazusa_ai_chatbot/rag/` except `quote_aware_sequence.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/shared_memory_prewarm.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/` except the four enumerated
  modifications
- `src/kazusa_ai_chatbot/dsh_interaction/`
- `src/agentic_resolver/` except the two enumerated modifications
- `sidecars/dsh_resolution/` except the three enumerated source modifications
- retained action capabilities, `future_speak`, dialog, dispatcher, adapters,
  consolidation, scheduler, reflection, and memory owners.

## Configuration And Dependency Disposition

Delete these exact configuration symbols and their tests/documentation:

```text
BACKGROUND_WORK_LLM_BASE_URL
BACKGROUND_WORK_LLM_API_KEY
BACKGROUND_WORK_LLM_MODEL
BACKGROUND_WORK_LLM_MAX_COMPLETION_TOKENS
BACKGROUND_WORK_LLM_THINKING_ENABLED
CODING_AGENT_PM_LLM_BASE_URL
CODING_AGENT_PM_LLM_API_KEY
CODING_AGENT_PM_LLM_MODEL
CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS
CODING_AGENT_PM_LLM_THINKING_ENABLED
CODING_AGENT_PROGRAMMER_LLM_BASE_URL
CODING_AGENT_PROGRAMMER_LLM_API_KEY
CODING_AGENT_PROGRAMMER_LLM_MODEL
CODING_AGENT_PROGRAMMER_LLM_MAX_COMPLETION_TOKENS
CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED
CODING_AGENT_ACTION_LOOP_LLM_BASE_URL
CODING_AGENT_ACTION_LOOP_LLM_API_KEY
CODING_AGENT_ACTION_LOOP_LLM_MODEL
CODING_AGENT_ACTION_LOOP_LLM_MAX_COMPLETION_TOKENS
CODING_AGENT_ACTION_LOOP_LLM_THINKING_ENABLED
CODING_AGENT_WORKSPACE_ROOT
CODING_AGENT_PREFLIGHT_EXECUTION
CODING_AGENT_REPAIR_MAX_CALLS
CODING_AGENT_REPAIR_BUNDLE_CHAR_LIMIT
CODING_AGENT_REPAIR_TOTAL_BUNDLE_CHAR_LIMIT
```

Retain `BACKGROUND_WORK_WORKER_*`, input/output limits, every
`AGENTIC_RESOLVER_*`/`KAZUSA_DSH_*` field, and live
`RAG_PLANNER_LLM_*`/`RAG_SUBAGENT_LLM_*`/`WEB_SEARCH_LLM_*` and web-provider
settings. `pyproject.toml` has no coding-only runtime dependency, so Plan 3
changes no dependency declaration. The package include remains
`kazusa_ai_chatbot*`; deletion removes the coding package naturally.
The public-media port reuses the already declared `httpx`, Pillow, and retained
vision-inspection route; it adds no dependency or configuration field.

## Test And Fixture File Disposition

### Create Or Rewrite As Plan 3 Owners

- `tests/unit/agentic_resolver/test_runtime_task_lifecycle.py`
- `tests/unit/task_resolution/test_contracts.py`
- `tests/unit/task_resolution/test_projection.py`
- `tests/unit/task_resolution/test_service.py`
- `tests/unit/task_resolution/test_decommission.py`
- `tests/unit/db/test_task_resolution_sessions.py`
- `tests/unit/db/test_accepted_tasks.py`
- `tests/unit/scripts/test_check_dsh_plan3_drain.py`
- `tests/unit/background_work/test_dsh_jobs.py`
- `tests/unit/background_work/test_dsh_worker.py`
- `tests/unit/background_work/test_result_source.py`
- `tests/unit/accepted_task/test_dsh_task_lifecycle.py`
- `tests/unit/action_spec/test_accepted_task_control.py`
- `tests/unit/nodes/test_persona_supervisor2_dsh_task_actions.py`
- `tests/unit/cognition_core_v3/test_dsh_task_handoff.py`
- `tests/unit/cognition_episode/test_task_result_source.py`
- `tests/unit/service/test_dsh_task_composition.py`
- `tests/unit/brain_service/test_dsh_task_readiness.py`
- `tests/unit/llm_interface/test_route_report.py`
- `tests/unit/test_config_dsh_cutover.py`
- `tests/test_dsh_plan3_task_resolution.py`
- `tests/test_dsh_plan3_task_resolution_live_db.py`
- `tests/test_dsh_plan3_e2e_live_llm.py`
- `tests/test_dsh_plan3_documentation.py`
- `tests/test_cognition_llm_producer_matrix.py`
- `tests/test_control_console_service_config.py`
- `tests/test_dsh_brain_interaction_contracts.py`
- `tests/test_dsh_brain_interaction_decision.py`
- `tests/test_dsh_brain_interaction_service.py`
- `tests/test_dsh_brain_interaction_persistence.py`
- `tests/test_dsh_brain_interaction_live_llm.py`
- `tests/test_dsh_plan2_e2e_live_llm.py`
- `tests/test_relevance_turn_settlement_graph.py`
- `tests/unit/cognition_core_v3/test_dsh_interaction_contract.py`
- `tests/unit/cognition_core_v3/test_handleless_contract.py`

Rewrite or absorb the still-valid preservation coverage from:

- `tests/test_task_resolution_contracts.py`
- `tests/test_task_resolution_inline_promotion.py`
- `tests/test_task_resolution_background_resume.py`
- `tests/test_task_resolution_cutover_live_db.py`
- `tests/test_task_resolution_persona_e2e_live_llm.py`
- `tests/test_accepted_task_lifecycle.py`
- `tests/test_accepted_task_prompt_contract.py`
- `tests/test_background_work_jobs.py`
- `tests/test_action_spec_results.py`
- `tests/unit/cognition_resolver/test_capabilities.py`
- `tests/unit/cognition_core_v3/test_handleless_contract.py`
- `tests/test_config.py`
- `tests/test_llm_interface_route_report.py`
- `tests/test_agentic_resolver_sidecar_process.py`
- `tests/test_dsh_tool_gateway_contracts.py`
- `tests/test_dsh_tool_gateway_media.py`
- `tests/test_stage3_fresh_database_bootstrap.py`
- `tests/test_stage3_fresh_database_e2e_live_llm.py`
- `tests/fixtures/stage3_fresh_database_cases.json`
- `tests/test_adapter_readable_mentions_live_llm.py`
- `tests/test_llm_time_payload_projection.py`
- `tests/test_rag_dialog_event_logging.py`
- `tests/control_console_e2e/test_page_navigation_e2e.py`
- `tests/ownership/source_test_impact_manifest.json`
- `tests/fixtures/cognition_llm_producer_matrix.json`

Rewrite these exact sidecar test owners for the additive fourteen-tool catalog
and exact forwarding shape:

- `sidecars/dsh_resolution/tests/brain_interaction.spec.ts`
- `sidecars/dsh_resolution/tests/contracts.spec.ts`
- `sidecars/dsh_resolution/tests/semantic_gateway.spec.ts`
- `sidecars/dsh_resolution/tests/standard_profile.spec.ts`

### Delete As Replaced Legacy Coverage

- `tests/test_dsh_brain_interaction_pending.py`
- `tests/test_dsh_brain_interaction_resume.py`

Delete the exact dedicated coding-agent and complex-task files in Appendix A.
The following fixture directories are grouping labels only; the tracked files
inside them and the two standalone files are enumerated individually in the
deletion-artifact matrix below:

- `tests/fixtures/coding_agent_benchmark/`
- `tests/fixtures/coding_agent_existing_source_gates/`
- `tests/fixtures/coding_agent_full_workflow/`
- `tests/fixtures/coding_agent_source_intake_signoff_cases.json`
- `tests/fixtures/complex_task_resolver_review_cases.json`

Delete these exact replaced task/RAG2 tests; each also has an individual
deletion-artifact row below:

- `tests/test_task_resolution_orchestrator.py`
- `tests/test_task_resolution_specialists.py`
- `tests/test_task_resolution_state.py`
- `tests/test_task_resolution_live_llm.py`
- `tests/test_task_resolution_background_research_e2e_live_llm.py`
- `tests/test_local_context_resolver_rag2_vs_rag3_live_llm.py`
- `tests/test_persona_supervisor2_rag_supervisor2_live.py`
- `tests/test_persona_supervisor2_rag2_integration.py`
- `tests/test_quote_aware_rag_sequence.py`
- `tests/test_quote_aware_rag_sequence_live.py`
- `tests/test_rag_finalizer_time_context.py`
- `tests/test_rag_initializer_cache2.py`
- `tests/test_rag_phase3_initializer_live_llm.py`
- `tests/test_rag_phase3_supervisor_integration.py`
- `tests/test_rag_phase4_continuation_live_llm.py`
- `tests/test_rag_prompt_contract_text.py`
- `tests/test_rag_recall_live_llm.py`

The Plan 2 DSH fixture `tests/fixtures/dsh_standard_coding/` and all Plan 2
sidecar, gateway, interaction, controller, persistence, RPC, and evidence tests
remain regression anchors. Retain `resources/avatar.png` unchanged as the
digest-pinned public-media live fixture.

## Documentation Surface

After the code-ready gate, update the subsystem, operations, architecture, and
localized documentation below:

- `README_CN.md`
- `docs/HOWTO.md`
- `docs/SUBAGENT_INTERFACES.md`
- `docs/architecture/agentic_resolver_architecture.md`
- `docs/architecture/cognition_contracts_design.md`
- `docs/architecture/dsh_integration_architecture.md`
- `src/agentic_resolver/README.md`
- `src/kazusa_ai_chatbot/task_resolution/README.md`
- `src/kazusa_ai_chatbot/accepted_task/README.md`
- `src/kazusa_ai_chatbot/action_spec/README.md`
- `src/kazusa_ai_chatbot/background_work/README.md`
- `src/kazusa_ai_chatbot/brain_service/README.md`
- `src/kazusa_ai_chatbot/dsh_interaction/README.md`
- `src/kazusa_ai_chatbot/cognition_core_v3/README.md`
- `src/kazusa_ai_chatbot/cognition_resolver/README.md`
- `src/kazusa_ai_chatbot/db/README.md`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/README.md`
- `src/kazusa_ai_chatbot/rag/README.md`
- `src/kazusa_ai_chatbot/self_cognition/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `sidecars/dsh_resolution/README.md`

The repository-root `README.md` is excluded from this early documentation
pass. Audit it only through the final gated task in P3-P3, after every
production-code change and every named live-behavior gate is complete. That
audit may change `README.md` only when a finalized product fact is materially
false or stale, and then only by minimally editing the existing global
product document in place. It must preserve the product identity, onboarding,
use cases, supported-model guidance, architecture overview, examples, design
principles, runtime layers, quick start, repository map, project status, and
license. Plan names, execution history, evidence-ledger references, temporary
gate status, and low-level migration mechanics do not belong in `README.md`.

Delete obsolete current architecture/ICD files with the removed runtime:

- `docs/architecture/coding_agent_architecture.md`
- `src/kazusa_ai_chatbot/coding_agent/README.md` and its package READMEs
- `src/kazusa_ai_chatbot/complex_task_resolver/README.md`

`docs/research/bot_cognition_apple_to_apple_comparison_zh_clean_room.md` is a
dated research record and remains unchanged. Documentation must describe the
DSH-only production route, binding/recovery state, required readiness, drain,
new task controls, retained RAG3/prewarm owners, and exact obsolete environment
fields. DSH/gateway documentation must preserve all thirteen Plan 2 tool rows,
name `kazusa_inspect_public_media` as the sole Plan 3 addition, describe its
safe-fetch/vision boundary, and record the resulting fourteen-tool catalog
digest transition and segment-rotation rule. Interaction, cognition, and Brain
service ICDs must describe `dsh_brain_interaction.v2`, full reusable cognition
ownership, exact per-kind internal decisions, one-shot grants, and the absence
of DSH user relay/reply/checkpoint state.

## Executor Autonomy Boundaries

The implementation owner may choose local function decomposition, naming of
private helpers, and test-fixture mechanics inside the exact owned paths when
all closed schemas, semantic owners, state transitions, cutover rules, and
test nodes remain unchanged. The owner records such choices in the evidence
ledger.

A new production path, caller, durable field/state, config symbol, external
effect, compatibility behavior, fallback, migration, semantic decision, or
change to a Plan 1/Plan 2 pin requires a recorded amendment and explicit user
approval before implementation. Production data operations and deployment
require the user's separate authorization for the named environment. The
independent reviewer remains read-only; remediation returns to the
implementation owner and receives a fresh independent review.

An amendment may only reconcile verified repository drift required to close an
existing functional-port, removal, safety, or release row. It cannot add an
unrelated feature, redesign a retained owner, create a compatibility tail, or
move required Plan 3 work into a later plan.

## Mandatory Implementation Order

### P3-P0 — Preflight And Baseline

1. Confirm this plan remains `approved`, both predecessor plans remain
   completed, and the worktree/status and exact inventory match this plan.
2. Read current `README.md`, `docs/HOWTO.md`, relevant subsystem ICDs, source,
   tests, and current source-test manifest. Record the baseline commit and
   unrelated failures.
3. Resolve the three runtime roles below and record each handoff.
4. Run the current deterministic collection and baseline commands before edits.

**Gate P3-P0:** exact paths, contracts, baseline, executor records, and
predecessor pins match; any drift has an approved amendment.

### P3-P1 — Tests And Manifest First

1. Add the deterministic owner tests, integration tests, live test skeletons,
   decommission assertions, data/drain assertions, doc assertion, and exact
   source-test manifest rows named below. Add the fourteen-tool cross-language
   catalog, public-media safe-fetch/result, and old-digest rotation tests before
   changing their production owners.
2. Rewrite preservation tests to the DSH contracts and remove test collection
   references to deleted legacy modules.
3. Collect all new exact nodes. Run the new deterministic nodes and record the
   expected red failures against pre-cutover production code.

**Gate P3-P1:** every planned production path has an exact collecting owner
node; red failures demonstrate the missing DSH cutover or still-present legacy
surface rather than malformed tests.

### P3-P2 — Production Cutover And Deletion

1. Implement the additive public-media semantic tool across Brain and sidecar,
   prove the original thirteen rows are byte-identical, prove the fourteen-row
   digest/rotation/safety contract, and make that port green before deleting
   the legacy media subagent.
2. Implement runtime wrappers, context/start/ref/result contracts, binding DB
   owner, and shared runtime composition.
3. Implement foreground/background execution, cooperative promotion, task
   payload V2, result source, accepted-task controls, readiness, and the
   character-owned `dsh_brain_interaction.v2` full-cognition-loop bridge.
4. Remove coding-era persistence/prompt fields and obsolete configuration.
5. Delete every exact legacy production/script/test/fixture path only after
   its functional port or justified-removal proof is green.
6. Run deterministic owner, cross-boundary, static decommission, manifest
   impact, full non-live, sidecar build/typecheck/test, and live-DB gates.

**Gate P3-P2:** the code-ready candidate has one DSH task route, the exact
fourteen-tool catalog with public-media safety green, zero legacy executor
imports, complete deterministic/live-DB evidence, and no docs edit yet.

### P3-P3 — Documentation, Live Behavior, Deployment Rehearsal, Review

1. Update/delete the exact non-root documentation surface after P3-P2 is
   green. Keep `README.md` at its pre-execution content until step 6.
2. Run the deterministic code-owner gates required before live verification.
3. Start Brain and sidecar using the Plan 2 pinned runtime; execute each named
   real-LLM node individually and inspect its trace/output. Execute the named
   real debug-user node through public HTTP and registered adapter callback
   transport, preserving its per-turn artifacts and log conclusions.
4. Rehearse the exact drain queries and admission/readiness/deployment sequence
   against the authorized environment without changing production data unless
   the user has separately authorized deployment.
5. Obtain independent review of the final diff, evidence, deletion set,
   contracts, and gate ledger. Resolve every material finding within scope.
6. After all production-code changes are complete and every named live-
   behavior gate is green, audit `README.md` against the finalized product.
   If no material product statement is false or stale, leave it unchanged and
   record that result. If a correction is necessary, make the smallest in-
   place update to the affected existing sections. Preserve the global README
   structure and exclude phase labels, plan history, evidence-ledger text,
   drain/deployment procedure, schema field inventories, CAS/lease internals,
   and other subsystem-level implementation detail.
7. Only after step 6, run the documentation nodes and full non-live suite, then
   include the final `README.md` decision and diff in independent sign-off.

**Gate P3-P3:** every functional/release gate is green without waiver; the
real debug-user evidence set is complete; every functional-port and
justified-removal row is closed; zero deferred DSH integration work remains;
the final global README audit is recorded and any necessary change is minimal;
review passes; the candidate is release-ready. Deployment execution remains
separately user-authorized.

### P3-P4 — Authorized Big-Bang Release And Post-Cutover Proof

1. Receive the user's explicit command naming the deployment environment and
   authorizing process/data operations. Record the exact service manager and
   release identity before changing state.
2. Pause every adapter and debug client, prove zero outstanding Brain requests,
   and run the exact five-category legacy/old-catalog drain while the old Brain,
   worker, and matching sidecar remain alive. Proceed only with all counts at
   zero.
3. Gracefully stop the old Brain and old sidecar, install the complete Plan 3
   Brain/sidecar candidate as one release, run Mongo bootstrap/index
   reconciliation, start the new Brain, then start the independently managed
   pinned sidecar.
4. Keep all adapter/debug intake paused until Brain `/health`, Brain
   `/runtime/dsh/health`, authenticated sidecar `system.health`, the exact V2
   identity pins, and one no-task-admission smoke are green. Restore the
   previous complete release if this pre-intake gate fails.
5. Resume all adapters/debug clients together. Run the authorized post-cutover
   public `/chat` smoke and background delivery smoke, verify new bindings and
   payload V2 only, and record a bounded observation window with zero legacy
   import/config/data-path use.
6. On any post-intake-resumption fault, keep the DSH-only data contract and roll
   forward. Record the incident, durable session state, and recovery evidence.
7. Have the independent reviewer inspect the final deployment evidence,
   catalog identity, post-cutover route, and zero-deferred-work attestation and
   issue the closing pass/block decision.

**Gate P3-P4:** the named environment runs the Plan 3 release; all five drain
counts were zero; readiness and pre/post-intake smoke passed; intake is in
its intended final state; new work uses DSH only; deployment evidence is
complete; and rollback or roll-forward disposition is recorded. Plan 3 cannot
be marked `completed` before this gate and the closing independent pass both
exist.

## Required Roles

### `p3_implementation_owner`

- **Responsibility:** implement the complete approved tests-first cutover,
  maintain scope/evidence, and prepare the deployment candidate.
- **Owned surface:** all exact create/modify/delete source, test, fixture,
  config, manifest, and documentation paths in this plan.
- **Authority:** edit only the approved surface; run local deterministic,
  sidecar, live-DB, and user-authorized real-LLM checks; propose an amendment
  for scope drift. Production data/deployment authority is excluded.
- **Applicable skills:** development-plan, local-llm-architecture,
  no-prepost-user-input, py-style, cjk-safety when triggered,
  test-style-and-execution, character-test for the live user-wire case, and
  Chinese translation when triggered.
- **Capability floor:** senior Python/TypeScript distributed-systems work,
  async cancellation/races, Mongo idempotency/leases, DSH fencing, prompt
  contracts, and exact pytest ownership.
- **Independence requirement:** may be the active primary engineering agent;
  must be distinct from `p3_independent_reviewer` for final sign-off.
- **Acceptance output:** scoped diff, updated plan gate/evidence ledger,
  baseline comparison, exact command results, live trace inspection notes,
  drain rehearsal, and residual-risk statement.
- **Gate:** P3-P0 through P3-P3 implementation-owner evidence is green without
  waiver.

### `p3_deployment_operator`

- **Responsibility:** pause all adapter/debug intake, execute the drain and
  atomic install, enforce readiness, resume intake, and own the recovery
  decision when separately authorized by the user.
- **Owned surface:** runtime processes, authorized deployment environment,
  read-only drain queries, the single obsolete index drop, and release state;
  source edits are excluded.
- **Authority:** operational actions explicitly authorized by the user for the
  named environment; record counts/health without document contents.
- **Applicable skills:** development-plan and test-style-and-execution for the
  deployment/rehearsal evidence commands.
- **Capability floor:** production process control, Mongo query/index
  operations, Brain/sidecar health interpretation, and atomic release
  recovery.
- **Independence requirement:** may be the implementation owner when the user
  grants both roles; reviewer independence still applies.
- **Acceptance output:** timestamped zero-drain report, release identity,
  before/after health, smoke results, admission state, and recovery decision.
- **Gate:** P3-P4 matches the approved sequence; all post-install
  readiness/pre-intake smokes are green before intake resumes; final
  post-intake evidence is green.

### `p3_independent_reviewer`

- **Responsibility:** independently review architecture, predecessor carryover,
  exact change radius, semantic ownership, data safety, test adequacy,
  decommission completeness, and final evidence.
- **Owned surface:** read-only final diff, plan, predecessor records, source,
  test output, traces, drain rehearsal, and documentation.
- **Authority:** issue material findings and pass or block sign-off; source
  edits and implementation ownership are excluded.
- **Applicable skills:** development-plan, local-llm-architecture,
  no-prepost-user-input, py-style, test-style-and-execution, and
  character-test when reviewing the live user-wire evidence.
- **Capability floor:** senior independent review of Python/TypeScript
  distributed runtimes, Mongo state machines, LLM semantic boundaries, and
  test traceability.
- **Independence requirement:** different active executor from
  `p3_implementation_owner`; review context must include the final candidate
  and evidence, not implementation steering ownership.
- **Acceptance output:** written findings with exact path/symbol/evidence,
  disposition of every finding, and final pass/block decision.
- **Gate:** zero unresolved material finding and explicit independent pass.

At each handoff, the parent resolves the role from currently available
project-native agents/models using capability, context, tools, verification
fit, latency, and expected total completion cost. The handoff records plan id,
role contract, remaining scope, owned paths, executor/model/configuration,
baseline, acceptance output, and gate. This plan supplies no fixed executor.

## Test Impact And Traceability

Every node below is part of P3-P1 scope. Planned files/nodes must collect
before production edits. Deterministic owner nodes are primary; integration,
live-DB, process, and real-LLM nodes are supplemental.

### Modified And New Production Owners

| Exact source or governed artifact path | Changed symbol/contract | Semantic owner | Exact deterministic pytest node ids | Supplemental integration/live node ids | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| `src/agentic_resolver/runtime.py` | generic checkpoint/amend/continue/cancel/inspect wrappers with fresh authority for checkpoint or terminal follow-up | Agentic resolver runtime | `tests/unit/agentic_resolver/test_runtime_task_lifecycle.py::test_checkpoint_and_terminal_continuation_issue_fresh_authority_and_preserve_thread_segment` | none | deterministic unit | Background/task callers bypassing fencing, reusing expired authority, or opening a replacement follow-up thread. |
| `src/agentic_resolver/controller.py` | idempotent generic checkpoint/terminal continuation, complete inspect projection, and canonical thirteen-to-fourteen catalog rotation | Resolution controller | `tests/unit/agentic_resolver/test_runtime_task_lifecycle.py::test_controller_checkpoint_terminal_control_and_replay_are_fenced_and_idempotent`; `tests/unit/agentic_resolver/test_runtime_task_lifecycle.py::test_thirteen_to_fourteen_semantic_catalog_change_rotates_segment_and_rejects_old_authority` | none | deterministic unit | Stale lease, segment drift, duplicate continuation execution, terminal thread loss, or reuse of old catalog authority. |
| `sidecars/dsh_resolution/src/contracts.ts` | exact additive `kazusa_inspect_public_media` input schema in the description-free fourteen-tool projection | DSH RPC/catalog contract owner | `tests/test_dsh_tool_gateway_contracts.py::test_plan3_public_media_tool_is_byte_identical_across_python_and_sidecar_catalogs` | `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_public_research_and_media_use_native_web_and_kazusa_semantic_evidence` | deterministic static plus live E2E | Brain/sidecar schema drift, mutation of a Plan 2 tool, or hidden extra input. |
| `sidecars/dsh_resolution/src/profile.ts` | fourteen-name semantic catalog admission and new digest publication | DSH Standard composition owner | `tests/test_agentic_resolver_sidecar_process.py::test_plan3_public_media_tool_is_advertised_with_matching_fourteen_tool_digest` | `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_public_research_and_media_use_native_web_and_kazusa_semantic_evidence` | deterministic process plus live E2E | Sidecar starts with a mismatched digest, omits the ported function, or alters native precedence. |
| `sidecars/dsh_resolution/src/semantic_gateway.ts` | exact public-media argument forwarding | DSH semantic worker proxy | `tests/test_agentic_resolver_sidecar_process.py::test_plan3_public_media_tool_forwards_only_url_and_question` | `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_public_research_and_media_use_native_web_and_kazusa_semantic_evidence` | deterministic process plus live E2E | Sidecar rewrites the URL/question, forwards extra authority fields, or cannot call the new tool. |
| `src/kazusa_ai_chatbot/task_resolution/contracts.py` | V2 context, start spec, binding/ref, accepted control, DSH-only V1 result constraints | Task-resolution contract owner | `tests/unit/task_resolution/test_contracts.py::test_dsh_start_binding_reference_and_v1_result_contracts_are_exact` | none | deterministic unit | Legacy checkpoints/coding payloads or authority secrets entering product contracts. |
| `src/kazusa_ai_chatbot/task_resolution/projection.py` | ordered facts and exhaust-to-result mapping | Task-resolution projection owner | `tests/unit/task_resolution/test_projection.py::test_dsh_exhaust_maps_to_task_result_without_semantic_reclassification` | none | deterministic unit | Semantic rewriting, wrong status/evidence mapping, or authority leakage. |
| `src/kazusa_ai_chatbot/task_resolution/service.py` | configured runtime, inline shield/checkpoint race, promotion, generation-bound start/follow-up/cancel, and internal-interaction-safe task lifecycle | Task-resolution service | `tests/unit/task_resolution/test_service.py::test_inline_checkpoint_promotes_same_bound_dsh_session_without_canceling_reasoning`; `tests/unit/task_resolution/test_service.py::test_background_start_mints_authority_only_when_claimed`; `tests/unit/task_resolution/test_service.py::test_internal_dsh_interaction_never_creates_user_wait_state`; `tests/unit/task_resolution/test_service.py::test_delivered_followup_continues_same_thread_under_next_generation` | none | deterministic unit | Canceled DSH reasoning, token persistence, pending-user state, duplicate/lost/stale result, or a replacement thread on follow-up. |
| `src/kazusa_ai_chatbot/task_resolution/__init__.py` | DSH-only public exports | Task-resolution public boundary | `tests/unit/task_resolution/test_contracts.py::test_task_resolution_public_exports_are_dsh_only` | none | deterministic unit | Legacy orchestrator/specialist API remaining importable. |
| `src/kazusa_ai_chatbot/db/task_resolution_sessions.py` | revision/generation-guarded binding persistence and lookup by thread/session/current task/job | Task-binding DB owner | `tests/unit/db/test_task_resolution_sessions.py::test_binding_generation_attach_checkpoint_terminal_and_followup_reconcile_is_revision_guarded` | none | deterministic unit | Lost continuation result, stale-generation settlement, cross-task continuation, or non-idempotent binding updates. |
| `src/kazusa_ai_chatbot/db/__init__.py` | task-binding facade exports | Database facade | `tests/unit/db/test_task_resolution_sessions.py::test_binding_repository_is_exposed_only_through_named_db_helpers` | none | deterministic unit | Ad hoc Mongo access or missing public DB owner. |
| `src/kazusa_ai_chatbot/db/bootstrap.py` | binding/follow-up indexes, V2 interaction audit/grant indexes, removal of open-reply lookup, and exact obsolete coding-context index drop | Database bootstrap | `tests/unit/db/test_task_resolution_sessions.py::test_bootstrap_creates_binding_and_dsh_followup_indexes_and_drops_only_obsolete_coding_index`; `tests/test_stage3_fresh_database_bootstrap.py::test_v2_dsh_interaction_indexes_have_no_open_reply_lookup` | none | deterministic unit | Missing uniqueness, actionable V1 reply state, lost delivered follow-up, or destructive unrelated index cleanup. |
| `src/kazusa_ai_chatbot/db/dsh_interactions.py` | idempotent V2 interaction audit and exact one-shot grant persistence without pending-user lookup | DSH interaction DB owner | `tests/test_dsh_brain_interaction_persistence.py::test_v2_audit_and_one_shot_grant_are_idempotent_without_reply_lookup` | none | deterministic unit | Duplicate grant, actionable historical V1 state, user-wait matching, or audit loss. |
| `src/kazusa_ai_chatbot/db/script_operations.py` | read-only Plan 3 drain counts, including open pre-cutover DSH interactions/grants | Maintenance DB owner | `tests/unit/scripts/test_check_dsh_plan3_drain.py::test_drain_helpers_count_only_exact_legacy_active_undelivered_and_open_old_catalog_rows`; `tests/unit/scripts/test_check_dsh_plan3_drain.py::test_drain_owner_is_the_only_v1_payload_exception_and_uses_exact_filters` | none | deterministic unit | Broad query, document disclosure, stranded old-catalog reply/grant, or unrelated future-speak blocking deployment. |
| `src/kazusa_ai_chatbot/db/schemas.py` | typed binding generation, accepted-task DSH lineage/follow-up, and canceled job schema without interaction-wait state | Database schema owner | `tests/unit/db/test_task_resolution_sessions.py::test_binding_followup_schemas_are_closed_without_interaction_waiting` | none | deterministic unit | Unbounded/unknown durable fields, pending-user state, stale generations, or state drift. |
| `src/kazusa_ai_chatbot/service.py` | shared runtime injection, readiness, authenticated V2 interaction request composition, and removal of relay/reply/checkpoint owners | Brain composition root | `tests/unit/service/test_dsh_task_composition.py::test_lifespan_injects_one_shared_runtime_into_interaction_and_task_owners`; `tests/unit/service/test_dsh_task_composition.py::test_v2_interaction_route_has_no_checkpoint_delivery_or_reply_sink` | none | deterministic unit | Split runtimes, optional production route, or a DSH interaction escaping to a user surface. |
| `src/kazusa_ai_chatbot/brain_service/contracts.py` | DSH task readiness plus strict V2 internal interaction request/response projection | Brain service contract owner | `tests/unit/brain_service/test_dsh_task_readiness.py::test_task_capability_is_available_only_when_full_dsh_runtime_is_ready`; `tests/test_dsh_brain_interaction_contracts.py::test_v2_contract_has_no_relay_reply_or_pending_vocabulary` | none | deterministic unit | Task admission while sidecar/Brain bridge identity is unavailable, mismatched, or exposing relay fields. |
| `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py` | V1-to-V2 request/context/source projection and DSH-only task service call | Cognition resolver capability owner | `tests/unit/cognition_resolver/test_capabilities.py::test_task_resolution_preserves_recurrence_and_maps_dsh_deferred_result`; `tests/unit/cognition_resolver/test_capabilities.py::test_task_resolution_v2_context_projects_trusted_source_and_original_episode_ref`; `tests/unit/cognition_resolver/test_capabilities.py::test_task_capability_uses_runtime_readiness_without_legacy_fallback` | none | deterministic unit | Changed recurrence/priority semantics, wrong interaction lineage, model-hidden workspace leakage, or fallback. |
| `src/kazusa_ai_chatbot/accepted_task/models.py` | canceled state, model-hidden DSH lineage fields, and closed task affordance without DSH user-wait state | Accepted-task contract owner | `tests/unit/accepted_task/test_dsh_task_lifecycle.py::test_dsh_task_affordance_uses_opaque_ref_for_active_and_open_followup` | none | deterministic unit | Coding refs, DSH internal ids, or pending DSH user interaction reaching cognition, or delivered follow-up disappearing. |
| `src/kazusa_ai_chatbot/accepted_task/lifecycle.py` | attach/reconcile, follow-up claim/recovery, new delivery generation, and cancel lifecycle | Accepted-task lifecycle owner | `tests/unit/accepted_task/test_dsh_task_lifecycle.py::test_dsh_promotion_and_followup_claim_preserve_one_session_with_new_delivery_row` | none | deterministic unit | Delivered rows overwritten, duplicate sessions/tasks, stuck claims, or result/control races. |
| `src/kazusa_ai_chatbot/accepted_task/__init__.py` | DSH task affordance/control public exports | Accepted-task public boundary | `tests/unit/accepted_task/test_dsh_task_lifecycle.py::test_accepted_task_public_exports_exclude_coding_run_contexts` | none | deterministic unit | Legacy coding context loader remaining public. |
| `src/kazusa_ai_chatbot/db/accepted_tasks.py` | scoped active/open-follow-up query, one-open index, compare-and-set claim/recovery, waiting/cancel/result transitions, coding index removal | Accepted-task DB owner | `tests/unit/db/test_accepted_tasks.py::test_one_open_dsh_followup_is_scoped_indexed_and_excludes_legacy_coding_context`; `tests/unit/db/test_accepted_tasks.py::test_followup_claim_recovery_and_terminal_updates_are_revision_guarded` | none | deterministic unit | Cross-scope controls, two open follow-ups, stuck claims, stale updates, or legacy index/read path. |
| `src/kazusa_ai_chatbot/background_work/models.py` | generation-bound payload V2 plus waiting/canceled states | Background-work contract owner | `tests/unit/background_work/test_dsh_jobs.py::test_task_worker_payload_v2_accepts_only_generation_bound_open_or_continue` | none | deterministic unit | Legacy checkpoint/coding request, stale generation, or unknown worker operation admission. |
| `src/kazusa_ai_chatbot/background_work/jobs.py` | V2 queue validation and generation-bound DSH session binding | Background-work queue owner | `tests/unit/background_work/test_dsh_jobs.py::test_queue_validates_binding_generation_goal_scope_and_payload_v2_exactly` | none | deterministic unit | Mismatched goal/scope/session/generation or expired token in queue. |
| `src/kazusa_ai_chatbot/background_work/worker.py` | generation-fenced DSH worker result/wait/retry/canceled transitions | Background worker | `tests/unit/background_work/test_dsh_worker.py::test_worker_checkpoints_waits_and_terminalizes_current_generation_through_binding`; `tests/unit/background_work/test_dsh_worker.py::test_worker_retry_reuses_idempotent_task_session`; `tests/unit/background_work/test_dsh_worker.py::test_worker_rejects_stale_or_canceled_generation` | none | deterministic unit | Busy retry during user wait, stale result overwrite, duplicate DSH execution, or direct delivery. |
| `src/kazusa_ai_chatbot/background_work/subagent/task_orchestrator.py` | generation-bound DSH open/continue dispatcher only | Reviewed task worker adapter | `tests/unit/background_work/test_dsh_worker.py::test_task_orchestrator_dispatches_only_generation_bound_dsh_payload_v2_operations` | none | deterministic unit | Dynamic coding import, legacy orchestrator resume, or worker-owned cancel semantics. |
| `src/kazusa_ai_chatbot/background_work/result_source.py` | DSH task context/evidence tool-result episode | Background result source | `tests/unit/background_work/test_result_source.py::test_dsh_result_reenters_cognition_with_exact_goal_and_evidence_provenance` | none | deterministic unit | Final wording in worker, coding ledger read, or lost continuation provenance. |
| `src/kazusa_ai_chatbot/background_work/__init__.py` | payload V2 public exports | Background-work public boundary | `tests/unit/background_work/test_dsh_jobs.py::test_background_work_public_exports_are_dsh_task_or_future_speak_only` | none | deterministic unit | Legacy task payload still public. |
| `src/kazusa_ai_chatbot/db/background_work_jobs.py` | waiting/queue/result transitions and payload V2 claim filters | Background-work DB owner | `tests/unit/background_work/test_dsh_jobs.py::test_job_claim_excludes_waiting_and_v1_payloads`; `tests/unit/background_work/test_dsh_jobs.py::test_terminal_sink_is_idempotent_under_delivery_state_guards` | none | deterministic unit | V1 execution, interaction spin, or duplicate result-ready transition. |
| `src/kazusa_ai_chatbot/action_spec/registry.py` | `accepted_task_control` registry/projection and legacy capability removal | Action registry | `tests/unit/action_spec/test_accepted_task_control.py::test_registry_exposes_closed_dsh_task_controls_without_legacy_coding_capability` | none | deterministic unit | Coding-run vocabulary or unbounded control decisions in prompts. |
| `src/kazusa_ai_chatbot/action_spec/evaluator.py` | exact control validation route | Action evaluator | `tests/unit/action_spec/test_accepted_task_control.py::test_evaluator_accepts_only_advertised_typed_task_control` | none | deterministic unit | Unscoped or unknown accepted-task control. |
| `src/kazusa_ai_chatbot/action_spec/execution.py` | follow-up/cancel execution and active-or-open status result projection | Action execution | `tests/unit/action_spec/test_accepted_task_control.py::test_control_claims_advertised_followup_or_cancels_without_interpreting_user_text` | none | deterministic unit | New DSH session duplication, delivered-row overwrite, or deterministic user-intent parsing. |
| `src/kazusa_ai_chatbot/action_spec/results.py` | task-resolution context result fields | Action result contract | `tests/unit/action_spec/test_accepted_task_control.py::test_action_result_projects_task_context_without_coding_context` | none | deterministic unit | Coding result carrier surviving cutover. |
| `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py` | control validation, follow-up task materialization, generation-bound continuation payload, and cancel service call | Background action handler | `tests/unit/action_spec/test_accepted_task_control.py::test_handler_binds_control_to_trusted_scope_and_task_affordance` | none | deterministic unit | Raw user approval, workspace, or internal DSH id becoming action authority. |
| `src/kazusa_ai_chatbot/action_spec/__init__.py` | canonical capability exports | Action public boundary | `tests/unit/action_spec/test_accepted_task_control.py::test_action_public_exports_use_only_accepted_task_control` | none | deterministic unit | Legacy capability import compatibility. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | load scoped active and sole open-follow-up DSH task affordances | Persona graph | `tests/unit/nodes/test_persona_supervisor2_dsh_task_actions.py::test_persona_graph_loads_only_scoped_active_or_open_followup_affordances` | none | deterministic unit | Cross-user/task prompt context, lost delivered follow-up, or coding-run lookup. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` | accepted-task action selection context | Persona state contract | `tests/unit/nodes/test_persona_supervisor2_dsh_task_actions.py::test_action_selection_context_has_closed_accepted_task_shape` | none | deterministic unit | Unvalidated DSH identity/state in model context. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | DSH task affordance prompt projection | Cognition materialization | `tests/unit/nodes/test_persona_supervisor2_dsh_task_actions.py::test_cognition_prompt_projects_opaque_task_ref_and_allowed_controls_only` | none | deterministic unit | Coding details or hidden runtime authority entering cognition. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py` | model-selected accepted-task control materialization | Cognition action materializer | `tests/unit/nodes/test_persona_supervisor2_dsh_task_actions.py::test_model_selected_control_binds_only_trusted_advertised_task_ref` | none | deterministic unit | Ref invention, deterministic keyword mapping, or operation widening. |
| `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py` | generic accepted-task continuation goal | Cognition appraisal | `tests/unit/cognition_core_v3/test_dsh_task_handoff.py::test_task_control_preserves_continuation_goal_without_coding_special_case` | none | deterministic unit | Coding-only continuation fixation after decommission. |
| `src/kazusa_ai_chatbot/cognition_episode.py` | `task_resolution_context` in tool-result episodes | Cognition episode contract | `tests/unit/cognition_episode/test_task_result_source.py::test_tool_result_episode_projects_dsh_task_context_and_provenance` | none | deterministic unit | Legacy coding context or missing DSH provenance in recurrence. |
| `src/kazusa_ai_chatbot/config.py` | obsolete route removal and retained DSH/RAG/background mechanics | Configuration owner | `tests/unit/test_config_dsh_cutover.py::test_legacy_background_and_coding_routes_are_absent_while_dsh_rag_and_worker_settings_remain` | none | deterministic unit | Startup requiring deleted routes or removal of live RAG/worker settings. |
| `src/kazusa_ai_chatbot/llm_interface/route_report.py` | route-report decommission | LLM route diagnostics | `tests/unit/llm_interface/test_route_report.py::test_route_report_omits_decommissioned_routes_and_keeps_live_routes` | none | deterministic unit | False health failures or stale coding/background route exposure. |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/catalog.py` | preserve thirteen rows byte-identically and add exact `kazusa_inspect_public_media` schema as row fourteen | Semantic catalog owner | `tests/test_dsh_tool_gateway_contracts.py::test_catalog_preserves_plan2_thirteen_and_adds_exact_public_media_tool` | `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_public_research_and_media_use_native_web_and_kazusa_semantic_evidence` | deterministic unit plus live E2E | Existing tool drift, missing public-media function, or an unplanned fifteenth tool. |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/contracts.py` | fourteen-tool common semantic result contract description | Semantic gateway contract owner | `tests/test_dsh_tool_gateway_contracts.py::test_catalog_declares_exact_fourteen_storage_independent_semantic_tools` | none | deterministic unit | Contract/documented cardinality diverges from the callable catalog. |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/dispatch.py` | dispatch public-media calls with exact two-field allowlist | Semantic gateway dispatcher | `tests/test_dsh_tool_gateway_media.py::test_public_media_dispatch_accepts_only_url_and_question` | `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_public_research_and_media_use_native_web_and_kazusa_semantic_evidence` | deterministic unit plus live E2E | Unknown fields, authority data, or wrong service method reaches the inspector. |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/media.py` | moved bounded public URL fetch plus retained visual inspection and evidence envelope | Media semantic service | `tests/test_dsh_tool_gateway_media.py::test_public_media_inspection_preserves_bounded_safe_fetch_and_visual_result`; `tests/test_dsh_tool_gateway_media.py::test_public_media_rejects_private_redirect_oversize_or_invalid_image_before_inspection` | `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_public_research_and_media_use_native_web_and_kazusa_semantic_evidence` | deterministic unit plus live E2E | SSRF, redirect bypass, oversized/invalid image inspection, capability loss, or evidence leakage. |
| `src/kazusa_ai_chatbot/media_inspection/contracts.py` | replace retired `complex_external_media` source with `dsh_public_media` | Shared media-inspection contract owner | `tests/test_dsh_tool_gateway_media.py::test_media_inspection_source_contract_accepts_dsh_public_media_and_rejects_legacy_source` | none | deterministic unit | Removed complex-resolver vocabulary remains valid or the new semantic service cannot call the retained inspector. |
| `src/kazusa_ai_chatbot/rag/__init__.py` | retained RAG3/evidence package identity | RAG package boundary | `tests/unit/task_resolution/test_decommission.py::test_retained_rag_package_has_no_rag2_runtime_claim` | none | deterministic unit | Documentation/import surface claiming retired RAG2 execution. |
| `scripts/check_dsh_plan3_drain.py` | read-only four-Mongo-count plus contained coding-ledger drain CLI and closed five-count JSON output | Deployment readiness operator tool | `tests/unit/scripts/test_check_dsh_plan3_drain.py::test_drain_cli_is_read_only_and_reports_closed_five_counts`; `tests/unit/scripts/test_check_dsh_plan3_drain.py::test_drain_cli_counts_nonterminal_and_invalid_coding_ledgers_without_exposing_content` | none | deterministic unit | Deployment uses ad hoc destructive queries, strands old-catalog interactions, ignores open/invalid legacy coding ledgers, escapes the configured root, or leaks row/ledger bodies. |

### Deletion Ownership Matrix

All rows in the next matrix use the deterministic owner nodes
`tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`
and
`tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports`.
The first proves exact file absence; the second proves no surviving production
import or configured entry point. The full per-source table follows in the
next section.

| Exact source path | Removed symbol/contract | Semantic owner | Exact deterministic pytest node ids | Supplemental integration/live node ids | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| `scripts/run_coding_agent_benchmark.py` | Legacy coding benchmark entry point deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `scripts/run_rag2_e2e_case.py` | Retired RAG2 executable entry point deletion | Retired RAG2 graph decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_action_loop/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_action_loop/actions.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_action_loop/context.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_action_loop/models.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_action_loop/parser.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_action_loop/prompts.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_action_loop/state.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_action_loop/supervisor.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_executing/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_executing/models.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_executing/runner.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_executing/supervisor.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/agent.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/github.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/local_checkout.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/managed_clone.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/managed_download.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/managed_inline.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/models.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/source_intake.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/source_resolver.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/source_scope.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_modifying/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_modifying/models.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_modifying/product_manager.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_modifying/programmer.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_modifying/supervisor.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_patching/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_patching/apply.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_patching/models.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_patching/patch_operations.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_patching/patch_validation.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_patching/patcher.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/agent.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/evidence.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/llm_config.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/master_pm.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/models.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/planner.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/product_manager.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/programmer.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/prompts.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/repository_map.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/supervisor.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/synthesizer.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_verifying/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_verifying/execution_planning.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_verifying/models.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_verifying/supervisor.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/acceptance.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/agent.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/diagnostic_trace.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/llm_config.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/models.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/package_coherence.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/product_manager.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/programmer.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/supervisor.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/synthesizer.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/workspace.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/coding_run/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/coding_run/evaluation.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/coding_run/ledger.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/coding_run/locking.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/coding_run/models.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/coding_run/supervisor.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/context_budget.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/external_evidence.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/file_agent.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/models.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/path_classification.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/repository_index/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/repository_index/builder.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/repository_index/identity.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/repository_index/models.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/repository_index/overlay.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/repository_index/regex_worker.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/repository_index/search.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/repository_index/storage.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/safety.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/supervisor.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/tools/__init__.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/tools/git.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/tools/paths.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/coding_agent/work_ledger.py` | Legacy coding executor deletion | Legacy coding runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/complex_task_resolver/__init__.py` | Legacy complex-task executor deletion | Legacy complex-task runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/complex_task_resolver/algorithmic.py` | Legacy complex-task executor deletion | Legacy complex-task runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/complex_task_resolver/constants.py` | Legacy complex-task executor deletion | Legacy complex-task runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/complex_task_resolver/contracts.py` | Legacy complex-task executor deletion | Legacy complex-task runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/complex_task_resolver/graph.py` | Legacy complex-task executor deletion | Legacy complex-task runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/complex_task_resolver/service.py` | Legacy complex-task executor deletion | Legacy complex-task runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/complex_task_resolver/stages.py` | Legacy complex-task executor deletion | Legacy complex-task runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/complex_task_resolver/subagent/__init__.py` | Legacy complex-task executor deletion | Legacy complex-task runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/complex_task_resolver/subagent/algorithmic.py` | Legacy complex-task executor deletion | Legacy complex-task runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/complex_task_resolver/subagent/evidence.py` | Legacy complex-task executor deletion | Legacy complex-task runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/complex_task_resolver/subagent/media.py` | Delete only after its public-image function and safety boundary move to the semantic gateway | Public-media port and legacy complex-task decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/test_dsh_tool_gateway_media.py::test_public_media_inspection_preserves_bounded_safe_fetch_and_visual_result`; `tests/test_dsh_tool_gateway_media.py::test_public_media_rejects_private_redirect_oversize_or_invalid_image_before_inspection` | `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_public_research_and_media_use_native_web_and_kazusa_semantic_evidence` | deterministic static/unit plus live E2E | Legacy module survives, or deletion loses public-image behavior or its safety checks. |
| `src/kazusa_ai_chatbot/complex_task_resolver/subagents.py` | Legacy complex-task executor deletion | Legacy complex-task runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py` | Retired RAG2 dispatcher deletion | Retired RAG2 graph decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py` | Retired RAG2 evaluator/finalizer deletion | Retired RAG2 graph decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py` | Retired RAG2 initializer deletion | Retired RAG2 graph decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_prompt_views.py` | Retired RAG2 prompt views deletion | Retired RAG2 graph decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` | Retired RAG2 supervisor deletion | Retired RAG2 graph decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_types.py` | Retired RAG2 graph types deletion | Retired RAG2 graph decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/rag/quote_aware_sequence.py` | Retired RAG2 quote-aware runner deletion | Retired RAG2 graph decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/task_resolution/orchestrator.py` | Legacy specialist routing/orchestration deletion | Legacy task-resolution runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/task_resolution/specialists/__init__.py` | Legacy specialist registry deletion | Legacy task-resolution runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/task_resolution/specialists/coding.py` | Legacy coding specialist deletion | Legacy task-resolution runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/task_resolution/specialists/local_context.py` | Legacy local-context specialist adapter deletion | Legacy task-resolution runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/task_resolution/specialists/public_research.py` | Legacy public-research specialist adapter deletion | Legacy task-resolution runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/task_resolution/specialists/text_computation.py` | Legacy text/computation specialist deletion | Legacy task-resolution runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |
| `src/kazusa_ai_chatbot/task_resolution/state.py` | Legacy specialist checkpoint state deletion | Legacy task-resolution runtime decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | none | deterministic static unit | Deleted executor remains importable, configured, or reachable. |

### Governed Artifacts And Documentation

| Exact governed artifact path | Changed contract | Semantic owner | Exact deterministic pytest node ids | Supplemental integration/live node ids | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| `tests/ownership/source_test_impact_manifest.json` | complete Plan 3 source rows and removal of deleted nodes | Test-impact governance | `tests/test_test_impact_manifest.py::test_manifest_contains_dsh_plan3_owner_rows`; `tests/test_test_impact_manifest.py::test_stale_required_node_fails_closed` | none | deterministic governance | Changed source lacks a collecting deterministic owner or names a deleted test. |
| `tests/fixtures/cognition_llm_producer_matrix.json` | remove coding/complex producers; retain RAG3/DSH producers | Cognition producer governance | `tests/test_cognition_llm_producer_matrix.py::test_producer_matrix_matches_current_source_owners` | none | deterministic governance | Deleted LLM owner remains advertised or retained owner disappears. |
| `tests/fixtures/stage3_fresh_database_cases.json` | DSH-only fresh-start route/config cases | Fresh-database fixture owner | `tests/test_stage3_fresh_database_bootstrap.py::test_fresh_database_fixture_uses_current_required_routes` | none | deterministic integration | Fresh installs still require deleted coding/background LLM routes. |
| `resources/avatar.png` | immutable public-media E2E fixture identity and remote digest pin | Public-media verification fixture | `tests/test_dsh_tool_gateway_media.py::test_public_media_e2e_fixture_identity_is_pinned` | `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_public_research_and_media_use_native_web_and_kazusa_semantic_evidence` | deterministic fixture plus live E2E | Live test silently changes visual input or accepts an unpinned remote asset. |
| `README.md` | final product-level audit after all code and live gates; minimal correction only if a finalized statement is materially stale, with the global document structure preserved | Project documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_describe_dsh_only_task_execution_and_retained_rag3`; `tests/test_dsh_plan3_documentation.py::test_plan3_docs_name_exact_fourteen_tool_catalog_and_public_media_boundary` | none | deferred deterministic docs, executed only at final P3-P3 step 7 | Root documentation is rewritten as a phase ledger, loses global product/onboarding sections, or advertises a materially stale finalized runtime. |
| `README_CN.md` | localized production architecture and route cleanup | Project documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_describe_dsh_only_task_execution_and_retained_rag3` | none | deterministic docs | Chinese root documentation drifts from the canonical runtime. |
| `docs/HOWTO.md` | required DSH operations, five-count drain, fourteen-tool catalog, config, test commands | Operations documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_describe_dsh_only_task_execution_and_retained_rag3`; `tests/test_dsh_plan3_documentation.py::test_plan3_docs_name_exact_fourteen_tool_catalog_and_public_media_boundary` | none | deterministic docs | Operator starts/deploys the removed route, uses obsolete settings, or misses the catalog transition. |
| `docs/SUBAGENT_INTERFACES.md` | remove legacy task/coding/complex worker interfaces | Subagent interface documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Deleted executor remains an advertised integration surface. |
| `docs/architecture/agentic_resolver_architecture.md` | production task edge, fourteen-tool catalog, and binding lifecycle | Resolver architecture | `tests/test_dsh_plan3_documentation.py::test_plan3_architecture_names_exact_v2_epochs_and_binding_flow`; `tests/test_dsh_plan3_documentation.py::test_plan3_docs_name_exact_fourteen_tool_catalog_and_public_media_boundary` | none | deterministic docs | Architecture still labels DSH as standalone-only or omits the media port. |
| `docs/architecture/cognition_contracts_design.md` | DSH task result/control handover | Cognition contract architecture | `tests/test_dsh_plan3_documentation.py::test_plan3_architecture_names_exact_v2_epochs_and_binding_flow` | none | deterministic docs | Cognition contract retains specialist/coding-run semantics. |
| `docs/architecture/dsh_integration_architecture.md` | Plan 3 production boundary, public-media tool, and catalog rotation | DSH integration architecture | `tests/test_dsh_plan3_documentation.py::test_plan3_architecture_names_exact_v2_epochs_and_binding_flow`; `tests/test_dsh_plan3_documentation.py::test_plan3_docs_name_exact_fourteen_tool_catalog_and_public_media_boundary` | none | deterministic docs | Plan 2 deferral text or stale catalog remains current after cutover. |
| `src/agentic_resolver/README.md` | fourteen-tool resolver catalog and compatibility rotation | Agentic resolver ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_name_exact_fourteen_tool_catalog_and_public_media_boundary` | none | deterministic docs | Resolver ICD retains the old cardinality or omits catalog rotation. |
| `docs/architecture/coding_agent_architecture.md` | delete obsolete architecture | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed coding runtime retains an authoritative architecture page. |
| `src/kazusa_ai_chatbot/coding_agent/README.md` | delete legacy coding ICD | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed coding runtime retains an ICD. |
| `src/kazusa_ai_chatbot/coding_agent/code_action_loop/README.md` | delete legacy action-loop ICD | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed coding action loop remains documented. |
| `src/kazusa_ai_chatbot/coding_agent/code_executing/README.md` | delete legacy execution ICD | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed coding execution remains documented. |
| `src/kazusa_ai_chatbot/coding_agent/code_fetching/README.md` | delete legacy fetch ICD | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed coding fetch remains documented. |
| `src/kazusa_ai_chatbot/coding_agent/code_modifying/README.md` | delete legacy modification ICD | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed coding modification remains documented. |
| `src/kazusa_ai_chatbot/coding_agent/code_patching/README.md` | delete legacy patch ICD | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed coding patching remains documented. |
| `src/kazusa_ai_chatbot/coding_agent/code_reading/README.md` | delete legacy reading ICD | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed coding reading remains documented. |
| `src/kazusa_ai_chatbot/coding_agent/code_verifying/README.md` | delete legacy verification ICD | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed coding verification remains documented. |
| `src/kazusa_ai_chatbot/coding_agent/code_writing/README.md` | delete legacy writing ICD | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed coding writing remains documented. |
| `src/kazusa_ai_chatbot/coding_agent/coding_run/README.md` | delete legacy run ICD | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed coding-run lifecycle remains documented. |
| `src/kazusa_ai_chatbot/coding_agent/repository_index/README.md` | delete legacy repository-index ICD | Legacy coding documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed repository-index runtime remains documented. |
| `src/kazusa_ai_chatbot/complex_task_resolver/README.md` | delete legacy complex-task ICD | Legacy complex-task documentation | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Removed complex executor remains documented. |
| `src/kazusa_ai_chatbot/task_resolution/README.md` | DSH-only task-resolution ICD | Task-resolution ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_describe_dsh_only_task_execution_and_retained_rag3` | none | deterministic docs | Package ICD disagrees with implementation. |
| `src/kazusa_ai_chatbot/accepted_task/README.md` | DSH refs/controls/waiting lifecycle | Accepted-task ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_describe_dsh_only_task_execution_and_retained_rag3` | none | deterministic docs | Accepted-task docs expose coding-run controls. |
| `src/kazusa_ai_chatbot/action_spec/README.md` | `accepted_task_control` catalog | Action-spec ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Prompt/action roster retains legacy capability. |
| `src/kazusa_ai_chatbot/background_work/README.md` | payload V2, waiting, result sink | Background-work ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_architecture_names_exact_v2_epochs_and_binding_flow` | none | deterministic docs | Worker docs permit V1 checkpoint/coding payload. |
| `src/kazusa_ai_chatbot/brain_service/README.md` | required DSH task readiness/composition | Brain service ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_architecture_names_exact_v2_epochs_and_binding_flow` | none | deterministic docs | Service readiness omits production DSH dependency. |
| `src/kazusa_ai_chatbot/db/README.md` | task-binding collection/index/retention | Database ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_architecture_names_exact_v2_epochs_and_binding_flow` | none | deterministic docs | Durable owner or data disposition is undocumented. |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/README.md` | exact fourteen-tool catalog and public-media safe-fetch/vision boundary | Semantic gateway ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_name_exact_fourteen_tool_catalog_and_public_media_boundary` | none | deterministic docs | Gateway documentation omits the port or changes a Plan 2 tool. |
| `src/kazusa_ai_chatbot/rag/README.md` | retained RAG3/evidence leaves and removed RAG2 runtime | RAG ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_describe_dsh_only_task_execution_and_retained_rag3` | none | deterministic docs | Live evidence leaves are deleted or retired graph remains claimed. |
| `src/kazusa_ai_chatbot/self_cognition/README.md` | remove coding-continuation references | Self-cognition ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_contain_no_legacy_executor_interfaces` | none | deterministic docs | Self-cognition advertises removed action vocabulary. |
| `src/kazusa_ai_chatbot/nodes/README.md` | current DSH task edge and retained RAG3 projection | Node ownership ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_describe_dsh_only_task_execution_and_retained_rag3` | none | deterministic docs | Node map routes tasks to old graphs. |
| `sidecars/dsh_resolution/README.md` | additive public-media forwarding and matching semantic catalog digest | DSH sidecar ICD | `tests/test_dsh_plan3_documentation.py::test_plan3_docs_name_exact_fourteen_tool_catalog_and_public_media_boundary` | none | deterministic docs | Sidecar documentation still claims thirteen tools or a stale digest contract. |

+### Deleted Test And Fixture Artifact Matrix

This is the exhaustive tracked-file deletion contract for the directory
groupings and test lists above. The owner test compares the repository to
these exact 124 paths; generated cache files are outside the tracked change
set.

| Exact governed artifact | Changed contract | Semantic owner | Exact deterministic pytest node IDs | Supplemental integration/live node IDs | Mode | Observable regression prevented |
|---|---|---|---|---|---|---|
| `tests/test_coding_agent_async_boundaries.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_benchmark_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_fetching.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_fetching_internet.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_image_reading_acceptance.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_interface.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase2_new_artifact_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase4_code_modifying_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase4_code_patching_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase4_interface.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase5_interface.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase5_patch_apply_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase6_code_executing_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase6_interface.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase8_interface.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase8_verify_repair_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase9_e2e_workflows.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase9_interface.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase9_run_supervisor_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase_b_execution_planning.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase_b_failure_feedback.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase_c_accepted_task_live_db.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase_c_locking.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase_c_run_context_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase_d_action_loop_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase_d_benchmark_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase_d_candidate_recovery.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase_d_coding_run_integration.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase_d_patch_operations.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_phase_d_repository_index.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_reading.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_reading_acceptance.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_reading_pm_programmer.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_source_intake.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_coding_agent_source_resolution.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_complex_task_resolver_algorithmic.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_complex_task_resolver_contracts.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_complex_task_resolver_evidence.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_complex_task_resolver_fixture.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_complex_task_resolver_graph.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_complex_task_resolver_live_llm.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_complex_task_resolver_media_subagent.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_complex_task_resolver_prompt_contract.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_complex_task_resolver_service.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_local_context_resolver_rag2_vs_rag3_live_llm.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_persona_supervisor2_rag2_integration.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_persona_supervisor2_rag_supervisor2_live.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_quote_aware_rag_sequence.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_quote_aware_rag_sequence_live.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_rag_finalizer_time_context.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_rag_initializer_cache2.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_rag_phase3_initializer_live_llm.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_rag_phase3_supervisor_integration.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_rag_phase4_continuation_live_llm.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_rag_prompt_contract_text.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_rag_recall_live_llm.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_task_resolution_background_research_e2e_live_llm.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_task_resolution_live_llm.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_task_resolution_orchestrator.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_task_resolution_specialists.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/test_task_resolution_state.py` | Remove retired legacy verification surface | Plan 3 test decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete test remains collectible or preserves a removed interface. |
| `tests/fixtures/coding_agent_benchmark/cases.jsonl` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/conftest.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_01_log_counter/README.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_01_log_counter/log_counter.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_01_log_counter/tests/test_log_counter.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/README.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/contacts_jsonl_to_csv/__init__.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/contacts_jsonl_to_csv/cli.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/contacts_jsonl_to_csv/converter.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/tests/test_cli.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/tests/test_converter.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/README.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/mdlinkcheck/__init__.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/mdlinkcheck/anchors.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/mdlinkcheck/cli.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/mdlinkcheck/scanner.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/tests/test_anchors.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/tests/test_scanner.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/README.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/issue_tracker/__init__.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/issue_tracker/api.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/issue_tracker/models.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/issue_tracker/store.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/tests/test_api.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/tests/test_store.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/README.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/__init__.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/cli.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/csv_io.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/fetch.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/html_extract.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/report.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/tests/test_cli.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/tests/test_fetch.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/tests/test_report.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_01_cli_command_discovery/README.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_01_cli_command_discovery/src/tooling/__init__.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_01_cli_command_discovery/src/tooling/cli.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_01_cli_command_discovery/src/tooling/commands.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_01_cli_command_discovery/tests/test_cli.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_02_csv_normalizer/README.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_03_counter_cli_json/README.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_03_counter_cli_json/counter_cli/__init__.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_03_counter_cli_json/counter_cli/cli.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_03_counter_cli_json/tests/test_cli.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_04_slug_normalization/README.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_04_slug_normalization/slug_tools/__init__.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_04_slug_normalization/slug_tools/slug.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_04_slug_normalization/tests/test_slug.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/README.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/release_feed/__init__.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/release_feed/cache.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/release_feed/cli.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/release_feed/feed.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/tests/test_cache.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/tests/test_cli.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_09_missing_dependency/README.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_09_missing_dependency/dep_tool/__init__.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_09_missing_dependency/dep_tool/loader.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/gate_09_missing_dependency/tests/test_yaml_dependency.py` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_full_workflow/manifest.md` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/coding_agent_source_intake_signoff_cases.json` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |
| `tests/fixtures/complex_task_resolver_review_cases.json` | Remove retired legacy fixture | Plan 3 fixture decommission | `tests/unit/task_resolution/test_decommission.py::test_legacy_test_and_fixture_artifacts_are_absent` | none | deterministic static unit | Obsolete fixture remains tracked or permits revival of a removed interface. |

## Cross-Boundary Acceptance Matrix

| Behavior | Exact node | Mode | Acceptance |
|---|---|---|---|
| Full cognition recurrence and answerable-now behavior | `tests/test_cognition_resolver_loop.py::test_loop_runs_cognition_capability_then_cognition_again`; `tests/test_cognition_resolver_loop.py::test_answerable_now_terminates_without_executing_optional_resolver` | deterministic integration | Task execution returns through the existing observation loop and final replacement cognition state. |
| Goal/progress/continuation preservation | `tests/test_cognition_resolver_loop.py::test_resolver_request_and_dependency_preserve_goal_continuation_ref`; `tests/test_cognition_resolver_loop.py::test_pending_resume_load_restores_original_goal_progress` | deterministic integration | Exact goal lineage survives recurrence and pending work. |
| DSH inline status mapping | `tests/test_dsh_plan3_task_resolution.py::test_inline_resolved_partial_and_terminal_blockers_recur_through_cognition` | deterministic integration | All six terminal statuses map and recur without legacy dispatch. |
| Foreground budget race | `tests/test_dsh_plan3_task_resolution.py::test_inline_budget_checkpoints_without_cancelling_and_promotes_same_session` | deterministic integration | Terminal wins races correctly; checkpoint promotion keeps one thread/session. |
| Direct background start | `tests/test_dsh_plan3_task_resolution.py::test_direct_background_claim_opens_dsh_and_terminal_result_uses_normal_delivery` | deterministic integration | Authority is minted on claim and result uses normal cognition/dialog delivery. |
| Internal DSH interaction continuation | `tests/test_dsh_plan3_task_resolution.py::test_internal_dsh_interaction_continues_without_user_wait_state` | deterministic integration | The full cognition-loop decision returns to the waiting DSH call without relay, pending-user state, or duplicate delivery. |
| DSH task controls | `tests/test_dsh_plan3_task_resolution.py::test_delivered_accepted_task_controls_continue_summarize_status_and_cancel_same_session` | deterministic integration | All carried-over coding intents retain one open opaque follow-up, create new delivery rows, and use the same DSH session. |
| Sidecar loss/restart | `tests/test_dsh_plan3_task_resolution.py::test_sidecar_fault_and_restart_recover_without_legacy_fallback`; `tests/test_agentic_resolver_sidecar_process.py::test_sidecar_restart_preserves_checkpoint_and_cold_resumes` | deterministic process/integration | Typed unavailable/recovery uses durable DSH state only. |
| Plan 3 catalog transition | `tests/unit/agentic_resolver/test_runtime_task_lifecycle.py::test_thirteen_to_fourteen_semantic_catalog_change_rotates_segment_and_rejects_old_authority`; `tests/test_agentic_resolver_sidecar_process.py::test_plan3_public_media_tool_is_advertised_with_matching_fourteen_tool_digest` | deterministic unit/process | Existing eligible V2 threads rotate safely while old authority/grants fail closed. |
| Public image inspection port | `tests/test_dsh_tool_gateway_media.py::test_public_media_inspection_preserves_bounded_safe_fetch_and_visual_result`; `tests/test_dsh_tool_gateway_media.py::test_public_media_rejects_private_redirect_oversize_or_invalid_image_before_inspection` | deterministic integration | Public-image capability survives without SSRF, invalid payload inspection, or the legacy complex resolver. |
| Terminal replay after response loss | `tests/test_agentic_resolver_sidecar_process.py::test_kill_after_terminal_commit_before_http_response_replays_exact_exhaust` | deterministic process | Operation replay returns the exact committed terminal. |
| V2 internal interaction and grants | `tests/test_dsh_brain_interaction_decision.py::test_dsh_interaction_runs_full_reusable_cognition_loop_and_returns_internal_decision`; `tests/test_dsh_brain_interaction_service.py::test_service_returns_internal_decision_without_checkpoint_or_delivery`; `tests/test_dsh_brain_interaction_contracts.py::test_v2_contract_has_no_relay_reply_or_pending_vocabulary` | deterministic integration | Every DSH semantic decision belongs to full character cognition; exact one-shot authority remains and user relay/reply state is absent. |
| Accepted-task state/delivery ordering | `tests/test_accepted_task_lifecycle.py::test_terminal_transitions_require_running_and_delivery_claims`; `tests/test_background_work_delivery.py::test_service_result_ready_delivery_uses_dispatcher_boundary` | deterministic integration | Result-ready precedes delivery and worker never sends text. |
| `future_speak` preservation | `tests/test_background_work_future_speak.py::test_worker_tick_dispatches_requested_future_speak_worker`; `tests/test_background_work_future_speak.py::test_future_speak_subagent_does_not_author_dialog_text` | deterministic integration | Retained worker/action behavior is unchanged. |
| Live Mongo binding, follow-up, and lease races | `tests/test_dsh_plan3_task_resolution_live_db.py::test_live_binding_promotion_followup_interaction_terminal_and_delivery_are_exactly_once`; `tests/test_dsh_plan3_task_resolution_live_db.py::test_live_one_open_followup_index_and_generation_cas_reject_duplicates`; `tests/test_dsh_plan3_task_resolution_live_db.py::test_live_worker_lease_loss_cannot_overwrite_newer_dsh_binding_revision` | live DB | Real collection/index/state/generation guards prevent duplicate or stale writes across repeated deliveries. |
| DSH-only static cutover | `tests/unit/task_resolution/test_decommission.py::test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent`; `tests/unit/task_resolution/test_decommission.py::test_runtime_import_graph_contains_no_legacy_executor_imports` | deterministic static unit | Complete legacy source/import/config deletion. |

## Real-LLM And Final E2E Sign-Off

Run the first ten-node diagnostic pass together with `-q -s`, preserve every
case dossier, and classify shared failure modes before remediation. Rerun
affected nodes separately afterward. Inspect the complete visible/silent result
plus protected trace and record source, DSH, evidence, recurrence, and delivery
lineage with secrets redacted.

| Advertised production behavior | Exact live node | Green behavior |
|---|---|---|
| User message: local fact | `tests/test_dsh_user_message_e2e_live_llm.py::test_live_user_message_local_fact_reaches_dsh` | Public `/chat` enters DSH and returns grounded task evidence through Brain-owned cognition/dialog. |
| User message: background summary | `tests/test_dsh_user_message_e2e_live_llm.py::test_live_user_message_background_summary_reaches_dsh` | Public `/chat` admits durable work and the result returns through normal recurrence and delivery. |
| Internal thought: file check | `tests/test_dsh_internal_thought_e2e_live_llm.py::test_live_internal_thought_file_check_reaches_dsh` | A durable real-user latch is claimed, consumed, and admitted to DSH with exact source lineage. |
| Internal thought: comparison | `tests/test_dsh_internal_thought_e2e_live_llm.py::test_live_internal_thought_comparison_reaches_dsh` | A second independent latch reaches DSH without fabricated user-wire input. |
| Self-cognition: targetless group review | `tests/test_dsh_self_cognition_e2e_live_llm.py::test_live_targetless_group_review_omits_dsh_task_resolution` | The real targetless producer reaches shared cognition while task resolution remains unavailable. |
| Self-cognition: promoted group review | `tests/test_dsh_self_cognition_e2e_live_llm.py::test_live_promoted_group_review_omits_dsh_task_resolution` | Reflection-owned group review preserves targetless readiness closure without fabricated identity. |
| Scheduled tick: commitment due | `tests/test_dsh_scheduled_tick_e2e_live_llm.py::test_live_commitment_due_tick_reaches_dsh` | The due commitment producer carries real-user identity into DSH and settles its source run. |
| Scheduled tick: future cognition | `tests/test_dsh_scheduled_tick_e2e_live_llm.py::test_live_scheduled_future_tick_reaches_dsh` | The scheduled-future producer reaches DSH with source/run lineage and bounded settlement. |
| Tool result: resolved | `tests/test_dsh_tool_result_e2e_live_llm.py::test_live_resolved_tool_result_delivers_without_recursive_dsh` | A resolved result recurs through cognition/delivery without opening another DSH task. |
| Tool result: failed | `tests/test_dsh_tool_result_e2e_live_llm.py::test_live_failed_tool_result_settles_without_recursive_dsh` | A failed result settles coherently without recursive task admission or false success. |

These ten nodes are the current final behavior sign-off. Component-live suites
retain direct Standard, semantic-tool, media, interaction, and dialog coverage
outside this source-integration release oracle.

### Historical Real Debug-User Demonstration Contract

The following contract explains the repeated legacy Node 2 evidence. It is
historical diagnostic material and is superseded as a Phase 3 release gate by
the trigger-source matrix above.

The named real debug-user node is the mandatory human-interaction-shaped E2E,
not a service-layer simulation. It proves the boundary between ordinary
pre-admission user clarification and character-owned DSH interactions:

1. Start the real Plan 2 DSH sidecar and a real Brain `uvicorn` process against
   an explicitly authorized isolated live Mongo database and the configured
   real LLM routes. Start a small HTTP adapter process implementing the existing
   `/send_message/capability` and `/send_message` callbacks, then register it as
   platform `debug` through `POST /runtime/adapters/register`.
2. Use one unique, stable debug user/channel/bot identity for the whole run and
   normal `debug_modes` with `listen_only=false`, `think_only=false`, and
   `no_remember=false`. The isolated database makes normal persistence safe and
   attributable.
   Create two files under the isolated DSH workspace:
   `plan3_real_user_e2e/alpha.txt` containing
   `PLAN3_E2E_ALPHA_NOT_SELECTED` and
   `plan3_real_user_e2e/beta.txt` containing
   `PLAN3_E2E_BETA_SELECTED`.
3. Before each POST, print and save the exact user text and observation target.
   Use these fixed candidates:

   ```text
   Turn 1 candidate:
   User: Please handle this in the background. In the task workspace, there are two files: plan3_real_user_e2e/alpha.txt and plan3_real_user_e2e/beta.txt. Before opening either file, ask me which one I want you to summarize. After I answer, read only that file and report its marker.
   Observation target: ordinary character clarification before DSH admission, with no task/job/binding/interaction row and no file read before the user's choice.

   Turn 2 candidate:
   User: Use plan3_real_user_e2e/beta.txt.
   Observation target: bounded background-task admission, beta-only DSH read evidence, and normal final delivery.
   ```

   Send Turn 1 only through `POST /chat` with a unique
   `platform_message_id`.
4. Assert from public response, persisted state, logs, and the real adapter
   callback that ordinary cognition selected human clarification and rendered
   the visible character question. Prove zero accepted task, background job,
   DSH binding, DSH interaction, and filesystem read for Turn 1.
5. Turn 2 supplies the fixture choice through a second ordinary `POST /chat`.
   It is normal user conversation context, not a DSH reply-lineage envelope.
6. Assert that cognition now admits one bounded direct-background task, the
   worker claims it, and DSH executes under one task/session/thread binding.
   If DSH raises a question, approval, or plan review, prove from protected
   traces that the full reusable cognition loop decides it internally and that
   no additional user message, interaction checkpoint, or adapter question is
   created.
7. Poll the registered adapter boundary for the final result and assert the
   normal `tool_result` episode, full cognition recurrence, character-authored
   dialog, dispatcher send, adapter message id, and accepted-task delivery
   state. Validate `PLAN3_E2E_BETA_SELECTED` in DSH evidence and the visible
   grounded result, and prove `PLAN3_E2E_ALPHA_NOT_SELECTED` never appears in a
   filesystem tool result.
8. Save `turn_001_request.json`, `turn_001_response.json`,
   `turn_001_log.txt`, `turn_001_delivery_callback.json`,
   `turn_002_request.json`, `turn_002_response.json`, `turn_002_log.txt`,
   `final_delivery_callback.json`, `dsh_execution_lineage.json`, and
   `behavior_audit_conclusions.md` under one timestamped
   `test_artifacts/debug_runs/<run_id>/` directory. The conclusion file records
   each turn's text, visible response, task/RAG/cognition/dialog observations,
   persistence effects, system concerns, and next-turn implication.

The node may read persisted rows and protected traces for assertions. It must
not call or patch `AgenticResolverRuntime`, `ResolutionController`, the task
service, interaction service, cognition nodes, dispatcher, database writers,
or adapter callbacks to initiate or advance the flow. It fails if either user
turn or final delivery bypasses its production HTTP and callback boundary, if
Turn 1 admits DSH work, or if any DSH interaction prompts the user. Raw
evidence stays local, secrets are redacted from the ledger, and the operator
inspects the full per-turn log slices before sign-off.

## Verification Commands

All Python commands use the project environment. The initial ten-node live-LLM
diagnostic runs as one batch; post-analysis reruns are individual. Live-DB and
deployment/drain commands run only against an explicitly authorized available
database/environment.

### P3-P0 Baseline And P3-P1 Collection

```powershell
git status --short
venv\Scripts\python -m pytest --collect-only -q
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
venv\Scripts\python -m pytest tests/test_test_impact_manifest.py -q
```

After P3-P1 creates the planned nodes:

```powershell
venv\Scripts\python -m pytest --collect-only -q tests/unit/agentic_resolver/test_runtime_task_lifecycle.py tests/unit/task_resolution tests/unit/db/test_task_resolution_sessions.py tests/unit/db/test_accepted_tasks.py tests/unit/scripts/test_check_dsh_plan3_drain.py tests/unit/background_work tests/unit/accepted_task/test_dsh_task_lifecycle.py tests/unit/action_spec/test_accepted_task_control.py tests/unit/nodes/test_persona_supervisor2_dsh_task_actions.py tests/unit/cognition_core_v3/test_dsh_task_handoff.py tests/unit/cognition_episode/test_task_result_source.py tests/unit/service/test_dsh_task_composition.py tests/unit/brain_service/test_dsh_task_readiness.py tests/unit/llm_interface/test_route_report.py tests/unit/test_config_dsh_cutover.py tests/test_agentic_resolver_sidecar_process.py tests/test_dsh_tool_gateway_contracts.py tests/test_dsh_tool_gateway_media.py tests/test_dsh_plan3_task_resolution.py tests/test_dsh_plan3_task_resolution_live_db.py tests/test_dsh_plan3_e2e_live_llm.py tests/test_dsh_plan3_documentation.py tests/test_cognition_llm_producer_matrix.py
```

### P3-P2 Deterministic, Process, Sidecar, And Live DB

```powershell
venv\Scripts\python -m pytest -q tests/unit/agentic_resolver/test_runtime_task_lifecycle.py tests/unit/task_resolution tests/unit/db/test_task_resolution_sessions.py tests/unit/db/test_accepted_tasks.py tests/unit/scripts/test_check_dsh_plan3_drain.py tests/unit/background_work tests/unit/accepted_task/test_dsh_task_lifecycle.py tests/unit/action_spec/test_accepted_task_control.py tests/unit/nodes/test_persona_supervisor2_dsh_task_actions.py tests/unit/cognition_core_v3/test_dsh_task_handoff.py tests/unit/cognition_episode/test_task_result_source.py tests/unit/service/test_dsh_task_composition.py tests/unit/brain_service/test_dsh_task_readiness.py tests/unit/llm_interface/test_route_report.py tests/unit/test_config_dsh_cutover.py tests/test_dsh_plan3_task_resolution.py tests/test_cognition_llm_producer_matrix.py
venv\Scripts\python -m pytest -q tests/test_agentic_resolver_contracts.py tests/test_agentic_resolver_controller.py tests/test_agentic_resolver_evidence.py tests/test_agentic_resolver_fingerprints.py tests/test_agentic_resolver_persistence.py tests/test_agentic_resolver_rpc.py tests/test_agentic_resolver_runtime.py tests/test_agentic_resolver_sidecar_process.py tests/test_dsh_brain_interaction_auth.py tests/test_dsh_brain_interaction_contracts.py tests/test_dsh_brain_interaction_decision.py tests/test_dsh_brain_interaction_pending.py tests/test_dsh_brain_interaction_persistence.py tests/test_dsh_brain_interaction_resume.py tests/test_dsh_brain_interaction_service.py tests/test_dsh_tool_gateway_authority.py tests/test_dsh_tool_gateway_contracts.py tests/test_dsh_tool_gateway_conversation.py tests/test_dsh_tool_gateway_media.py tests/test_dsh_tool_gateway_memory.py tests/test_dsh_tool_gateway_people.py tests/test_dsh_tool_gateway_recall_calendar.py tests/test_dsh_tool_gateway_worker.py
venv\Scripts\python -m pytest -q tests/test_cognition_resolver_contracts.py tests/test_cognition_resolver_loop.py tests/test_cognition_resolver_persona_graph.py tests/test_accepted_task_lifecycle.py tests/test_accepted_task_prompt_contract.py tests/test_background_work_delivery.py tests/test_background_work_future_speak.py tests/test_action_spec_attempt_ledger.py tests/test_action_spec_evaluator.py tests/test_action_spec_models.py tests/test_action_spec_results.py
venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution install --frozen-lockfile
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution build
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution test
venv\Scripts\python -m pytest -m live_db -q tests/test_dsh_plan3_task_resolution_live_db.py
$legacyCodingWorkspaceRoot = [Environment]::GetEnvironmentVariable("CODING_AGENT_WORKSPACE_ROOT")
if ([string]::IsNullOrWhiteSpace($legacyCodingWorkspaceRoot)) { throw "CODING_AGENT_WORKSPACE_ROOT is required for the legacy drain" }
$legacyCodingWorkspaceRoot = (Resolve-Path -LiteralPath $legacyCodingWorkspaceRoot).Path
venv\Scripts\python scripts/check_dsh_plan3_drain.py --legacy-coding-workspace-root $legacyCodingWorkspaceRoot --format json
```

### Static Decommission Checks

Each command must return no production hit except the explicitly retained
`TaskResolutionResultV1.coding_run_context` declaration/empty-value validator,
the approved read-only drain predicates covered by the dedicated owner audit,
and historical development plans/research records. The obsolete-contract scan
excludes only the approved drain owner; its next two lines positively audit and
test that narrow exception.

```powershell
rg -n "kazusa_ai_chatbot\.(coding_agent|complex_task_resolver)|task_resolution\.orchestrator|task_resolution\.specialists|persona_supervisor2_rag_(dispatch|evaluator|initializer|prompt_views|supervisor2|types)|quote_aware_sequence" src scripts
rg -n "accepted_coding_task_request|coding_run_ref|continue_bound_coding_run|task_resolution_checkpoint\.v1|task_orchestrator_worker_payload\.v1" src scripts tests --glob '!development_plans/**' --glob '!src/kazusa_ai_chatbot/db/script_operations.py'
rg -n "task_orchestrator_worker_payload\.v1" src/kazusa_ai_chatbot/db/script_operations.py
venv\Scripts\python -m pytest -q tests/unit/scripts/test_check_dsh_plan3_drain.py -k "v1_payload_exception"
rg -n "BACKGROUND_WORK_LLM_|CODING_AGENT_" src scripts README.md README_CN.md docs --glob '!docs/research/**'
rg -n "complex_external_media|exactly these thirteen|exactly the thirteen names|adds exactly thirteen|contains exactly the thirteen|supplies thirteen|all thirteen semantic tools|for the thirteen Kazusa semantic" src/agentic_resolver src/kazusa_ai_chatbot/dsh_tool_gateway src/kazusa_ai_chatbot/media_inspection sidecars/dsh_resolution/src README.md docs/HOWTO.md docs/architecture/agentic_resolver_architecture.md docs/architecture/dsh_integration_architecture.md
rg -n "from kazusa_ai_chatbot\.local_context_resolver|kazusa_ai_chatbot\.rag" src/kazusa_ai_chatbot/cognition_resolver src/kazusa_ai_chatbot/consolidation src/kazusa_ai_chatbot/service.py
```

The fifth command is a positive retention audit: expected live RAG3/evidence
imports must remain.

### P3-P3 Live LLM, User Sign-Off, Review, And Final Documentation

```powershell
venv\Scripts\python -m pytest -m live_llm -q -s tests/test_dsh_user_message_e2e_live_llm.py tests/test_dsh_internal_thought_e2e_live_llm.py tests/test_dsh_self_cognition_e2e_live_llm.py tests/test_dsh_scheduled_tick_e2e_live_llm.py tests/test_dsh_tool_result_e2e_live_llm.py
```

After the batch, classify generic/shared failure modes before remediation.
Rerun affected nodes individually, inspect all ten final dossiers, and complete
independent behavior/scope review. Documentation and bookkeeping commands run
last:

```powershell
venv\Scripts\python -m pytest -q tests/test_dsh_plan3_documentation.py tests/test_test_impact_manifest.py
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
```

## Functional Release Gates

| Gate | Green condition |
|---|---|
| P3-G1 — Predecessor fidelity | Every pinned resolution V2/Standard/profile/route/policy/store contract and all thirteen Plan 2 semantic rows remain exact; only the declared fourteenth public-media row changes the semantic catalog digest. The Plan 2 interaction relay/reply epoch is atomically superseded by the approved `dsh_brain_interaction.v2` amendment; no fixed Plan 1 worker binding is revived. |
| P3-G2 — Brain handover preservation | P-stage request lanes, resolver recurrence, goal/evidence lifecycle, final cognition commit, actions, L3/dialog, consolidation, and delivery ordering pass their exact anchors. |
| P3-G3 — DSH task execution | Foreground and direct-background work use one shared runtime, correct hidden authority, ordered model facts, native Standard tools, the exact fourteen-tool semantic catalog including the public-media port, and the sole validated terminal. |
| P3-G4 — Durable continuation | Binding-before-activation, cooperative task checkpoint, restart/replay, internal interaction continuation, lease loss, and terminal idempotency preserve one thread/session and one product result without a user-wait interaction state. |
| P3-G5 — Accepted/background lifecycle | Payload V2, waiting state, enqueue recovery, worker claims, result-ready, normal cognition/dialog delivery, and `future_speak` preservation pass. |
| P3-G6 — Brain interaction authority | Questions, approvals, and plan reviews run through the existing full reusable character cognition loop; only the exact per-kind answer/allow-once/reject decisions are accepted; one-shot grants remain exact; no relay, reply, pending-user, dialog, or user-text classifier path survives. |
| P3-G7 — Coding intent continuity | New coding, revise, summarize, status, approve/verify, blocker response, cancel, native edits/tests, and artifact result each have the exact DSH-era route and pass deterministic/live coverage. |
| P3-G8 — Complete decommission | Every exact legacy source/test/fixture/config/doc path is removed with the justified-removal proof above; static imports and manifest are clean; RAG3/prewarm/evidence leaves remain. |
| P3-G9 — Data and deployment safety | All five drain filters are zero, historical data disposition and catalog rotation are honored, only the named index is dropped, readiness gates intake, and rollback/roll-forward boundary is recorded. |
| P3-G10 — Final closure, verification, and review | Every functional-port and justified-removal row is closed; zero deferred DSH integration work remains; owner tests, cross-boundary tests, sidecar suite, manifest impact, full non-live, authorized live DB, the exact five-source/ten-node live matrix, retained first-batch evidence, systematic-failure disposition, individual remediation verification, ten independent behavior decisions, docs, deployment evidence, and independent review are green without waiver. |

All ten gates block release. A test pass without exact decommission, drain,
real-service behavior, or independent review cannot close the plan.

## Scope-Creep Exclusions

Plan 3 includes implementation, verified decommission, authorized deployment,
and post-deployment evidence. Production data and process actions begin only
after the user issues the required separate command naming the environment and
operation; the plan remains open until its required release work is executed.

The following are unrelated product changes and cannot enter this final DSH
integration stage:

- upgrading DSH `0.1.1-rc.2`, Cordis/Schemastery, the Standard preset, native
  tool policy, web provider, RPC/intake epoch, model route,
  context/completion budgets, or thinking setting, or changing the semantic
  catalog beyond the one exact additive public-media row;
- redesigning retained RAG3/local-context, shared-memory prewarm, Cache2,
  memory/conversation/person/recall/web evidence leaves, or semantic gateway
  implementations;
- changing general cognition semantics, character behavior, relevance, dialog
  voice, adapters, consolidation, scheduler, reflection, or unrelated action
  capabilities; and
- adding a DSH task/coding UI or any unrelated feature.

## Evidence Ledger And Closure

### 2026-08-29 — P3-P0 Execution Start

- Baseline commit: `59357e591f762f46b7492f12be42752daff25632`
  on `main`.
- Incoming worktree baseline: only this approved plan and
  `development_plans/README.md` were modified; both are preserved as
  user-owned planning changes.
- Implementation-owner resolution: one persistent Codex subagent using
  `gpt-5.6-luna`, maximum reasoning, and default normal-speed execution. This
  is a user-fixed Plan 3 execution constraint. The parent retains general
  direction, architecture, checkpoint, and lifecycle authority. The worker
  owns the exact P3-P1 through P3-P3 create/modify/delete/test/documentation
  surface and test execution. Independent review remains a separate executor.
- First checkpoint: P3-P0 inventory and deterministic baseline, followed by
  P3-P1 tests/manifest-first collection and expected-red evidence. Production
  source remains unchanged until the parent accepts that checkpoint.

### 2026-08-30 — P3-P1 Accepted And P3-P2 Authorized

- P3-P1 traceability is complete: all 101 exact Plan 3 nodes collect, 21
  deterministic nodes pass against the pre-cutover implementation, 72 are
  expected-red cutover contracts, and eight live nodes are deselected.
- The current full suite collects 4,006 tests (4,659 total with 653
  deselected). The retained-test import audit found and repaired 22 imports
  from two planned-deletion modules across 19 surviving tests. The AST
  survivor-import gate is green, and the exact decommission inventories remain
  117 production/script paths and 124 test/fixture paths.
- Baseline non-live evidence remains 3,925 passed, four skipped, and 644
  deselected. The sidecar typecheck is green; 13 sidecar assertions and six
  manifest structural assertions remain expected-red pending P3-P2.
- Parent architecture review accepted the executable lifecycle, durable CAS,
  projection, typed action, public-media safety, cross-boundary, and exact
  decommission owner tests. Static inspection is restricted to filesystem,
  import-graph, documentation, and configuration ownership gates.
- P3-P2 is authorized for the same persistent `gpt-5.6-luna` maximum-reasoning
  worker. The cutover remains one atomic DSH boundary: one shared runtime,
  closed V2 task context, durable generation/revision fencing, same-thread
  continuation, claim-time authority minting, the exact fourteenth public-media
  tool, no legacy fallback or user-text classifier, exact 117/124 deletion,
  and retained RAG3/prewarm/evidence leaves.

### 2026-08-30 — P3-P2 Amendment Approved

- The user explicitly approved the bounded three-part amendment recorded under
  `Exact Production Change Surface`: the transient direct-background
  admission carrier, deletion-aware fail-closed impact ownership, and removal
  of the three obsolete control-console model-route descriptors.
- Pre-amendment evidence: 3,388 non-live tests passed with only four deferred
  documentation failures; all three real-Mongo Plan 3 nodes passed; all 101
  sidecar tests passed; the 117/124 deletion inventories and import gates were
  exact; and introduced Ruff diagnostics were zero.
- P3-P2 remains open until the amendment owner tests, impact command, full
  non-live suite, sidecar suite, live-Mongo nodes, and exact scope audit are
  green. Documentation and live-LLM work remain P3-P3.

During implementation, append dated evidence beneath this section only while
the plan is `in_progress`. Record:

1. baseline commit/status and unrelated failures;
2. resolved executor/model/configuration for each role handoff;
3. P3-P0 through P3-P4 and P3-G1 through P3-G10 status;
4. exact command, exit status, duration, and concise inspected result;
5. live thread/segment/session, task/job, interaction, evidence, delivery, and
   real debug-user per-turn artifact lineage with secrets redacted;
6. drain rehearsal/deployment evidence when authorized;
7. independent findings and dispositions; and
8. residual risks.

### 2026-08-30 — P3-P2 Amendment Owner-Gate Evidence

- The shared worktree baseline remains commit
  `59357e591f762f46b7492f12be42752daff25632` with the pre-existing large P3
  cutover diff preserved. The amendment worker was the user-fixed persistent
  `gpt-5.6-luna` maximum-reasoning executor. The four documentation failures
  remain the already recorded P3-P3 residual and are not reclassified as an
  amendment failure.
- The direct-background path now returns only the transient,
  model-hidden `TaskResolutionAdmissionV1`; cognition projects the accepted
  observation without task/job/session identity, while deferred V1 results
  remain ref-backed. The impact validator now maps all 117 absent strict
  production/script sources to surviving exact decommission nodes and rejects
  unmapped strict changes. The console diff removes exactly
  `background_work`, `coding_pm`, and `coding_programmer` descriptors.
- `venv\\Scripts\\python -m pytest -q tests/unit/task_resolution/test_contracts.py tests/unit/task_resolution/test_service.py tests/unit/task_resolution/test_projection.py tests/unit/cognition_resolver/test_capabilities.py tests/test_dsh_plan3_task_resolution.py tests/test_test_impact_manifest.py tests/test_control_console_service_config.py`
  exited 0 in 1.13s: 43 passed.
- `venv\\Scripts\\python -m scripts.validate_test_impact --base-ref HEAD --run`
  exited 0 in 9.58s: 158 exact impact nodes collected and passed, including
  all 117 removed-source mappings.
- Read-only exact-scope checks against `HEAD` confirmed that the console route
  diff removes precisely the three approved descriptors and leaves every
  survivor unchanged; the manifest check confirmed 117 removed paths, all
  absent, with exact surviving node mappings.
- `venv\\Scripts\\python -m ruff check --ignore TRY004 scripts/validate_test_impact.py src/control_console/brain_model_routes.py src/kazusa_ai_chatbot/cognition_resolver/capabilities.py src/kazusa_ai_chatbot/task_resolution/__init__.py src/kazusa_ai_chatbot/task_resolution/contracts.py src/kazusa_ai_chatbot/task_resolution/service.py tests/test_control_console_service_config.py tests/test_test_impact_manifest.py tests/unit/cognition_resolver/test_capabilities.py tests/unit/task_resolution/test_contracts.py tests/unit/task_resolution/test_service.py tests/test_dsh_plan3_task_resolution.py`
  exited 0 in 0.11s; `git diff --check` exited 0 with only line-ending
  normalization warnings.
- `venv\\Scripts\\python -m pytest -m "not live_db and not live_llm" -q`
  exited 1 in 217.83s with 3,390
  passed, four skipped, and 523 deselected. The only four failures are the
  deferred documentation nodes listed in the amendment approval evidence:
  `test_plan3_docs_describe_dsh_only_task_execution_and_retained_rag3`,
  `test_plan3_docs_name_exact_fourteen_tool_catalog_and_public_media_boundary`,
  `test_plan3_docs_contain_no_legacy_executor_interfaces`, and
  `test_plan3_architecture_names_exact_v2_epochs_and_binding_flow`.
- `corepack pnpm@11.7.0 --dir sidecars/dsh_resolution build` exited 0 in
  2.01s; `corepack pnpm@11.7.0 --dir sidecars/dsh_resolution test` exited 0 in
  11.75s with 101/101 tests passed. The authorized live-DB command
  `venv\\Scripts\\python -m pytest -m live_db -q tests/test_dsh_plan3_task_resolution_live_db.py`
  exited 0 in 1.30s with all three Plan 3 nodes passed against the configured
  test MongoDB.
- P3-P2 amendment owner gates are green. P3-P2 remains an in-progress plan
  boundary because the four P3-P3 documentation nodes, named live-LLM E2Es,
  deployment/drain evidence, and independent review remain outstanding.

### 2026-08-30 — P3-P2 Py-Style Remediation Evidence

- Follow-up review corrected every amendment-added direct computed-call return
  in the task-resolution service and cognition admission projection to a named
  local followed by a simple return, preserving the exact carrier behavior and
  scope.
- `venv\\Scripts\\python -m pytest -q tests/unit/task_resolution/test_contracts.py tests/unit/task_resolution/test_service.py tests/unit/task_resolution/test_projection.py tests/unit/cognition_resolver/test_capabilities.py tests/test_dsh_plan3_task_resolution.py tests/test_test_impact_manifest.py tests/test_control_console_service_config.py`
  exited 0 in 1.17s: 43 passed.
- The amendment-file Ruff check exited 0 in 0.11s; scoped `git diff --check`
  exited 0 with only line-ending normalization warnings; and
  `venv\\Scripts\\python -m scripts.validate_test_impact --base-ref HEAD --run`
  exited 0 in 9.56s with 158 exact impact nodes passed. No new scope or
  behavioral residual was introduced.

### 2026-08-30 — P3-P3 Documentation Checkpoint Evidence

- Updated the approved Documentation Surface: README.md, README_CN.md,
  docs/HOWTO.md, docs/SUBAGENT_INTERFACES.md, the three current architecture
  documents, src/agentic_resolver/README.md, the eleven listed subsystem
  READMEs, and sidecars/dsh_resolution/README.md. Deleted exactly the
  obsolete architecture ICD plus the twelve package READMEs listed in the
  plan. The dated research record
  docs/research/bot_cognition_apple_to_apple_comparison_zh_clean_room.md
  has no diff.
- README_CN.md followed the required context map/shared glossary ledger,
  semantic draft, Chinese-first native rewrite, and bilingual
  fidelity/artifact audit. Contractual identifiers, command shapes, catalog
  rows, retained links, and diagram fences were audited and preserved. The documentation now
  describes the DSH-only route, transient TaskResolutionAdmissionV1 versus
  ref-backed deferred checkpoint, dsh_task_binding.v1 recovery and
  operation_generation/revision CAS, readiness/drain, accepted_task_control.v1,
  retained RAG3/prewarm ownership, removed configuration families, and the
  exact fourteen-row catalog with kazusa_inspect_public_media safety and
  digest/segment rotation.
- venv\\Scripts\\python -m pytest -q
  tests/test_dsh_plan3_documentation.py tests/test_test_impact_manifest.py
  exited 0 in 0.94s: 20 passed.
- venv\\Scripts\\python -m pytest -m "not live_db and not live_llm" -q
  exited 0 in 216.56s: 3,394 passed, four skipped, 523 deselected, and two
  warnings.
- venv\\Scripts\\python -m scripts.validate_test_impact --base-ref HEAD
  --run exited 0 in 9.48s: 158 exact impact-test nodes passed.
- git diff --check exited 0 with only the repository's LF-to-CRLF
  normalization warnings.
- Exact static decommission audits were run. The import/deleted-runtime
  audit exited 1 with no matches; the obsolete-contract audit exited 0 but
  still reports the pre-existing source predicates at
  src/kazusa_ai_chatbot/db/script_operations.py:128 and :134 for
  task_orchestrator_worker_payload.v1, plus intentional negative-test
  fixtures/assertions at tests/test_background_work_jobs.py:169,
  tests/test_dsh_plan3_documentation.py:55,
  tests/test_task_resolution_background_resume.py:175,
  tests/unit/action_spec/test_accepted_task_control.py:87 and :200, and
  tests/unit/background_work/test_dsh_jobs.py:248 and :331. This
  source/test residue is outside the approved docs-only slice and remains
  the exact-scope audit residual.
- The obsolete environment audit exited 1 with no matches; the stale
  catalog-wording audit exited 1 with no matches. The positive RAG3
  retention audit exited 0 and reported only the expected live imports in
  service, cognition_resolver, and consolidation. The full non-live run
  included the exact decommission test nodes, all of which passed.
- venv\Scripts\python -m pytest -q
  tests/unit/task_resolution/test_decommission.py exited 0 in 2.65s: five
  exact deletion/import/retained-RAG3 nodes passed.
- P3-P3 documentation owner gates are green. Remaining release blockers are
  the source-side obsolete-contract audit residual above, named live-LLM
  nodes, authorized drain/deployment rehearsal, and independent review.

### 2026-08-30 — P3-P2 Drain-Owner Exception Amendment Evidence

- The user approved the bounded P3-G9 exception for the existing read-only
  `count_dsh_plan3_drain_rows` owner. The source owner remains unchanged and
  retains exactly two `task_orchestrator_worker_payload.v1` predicates: the
  queued/in-progress execution filter and the terminal-undelivered filter.
  No runtime worker or product path receives this vocabulary.
- Added a deterministic AST owner gate to
  `tests/unit/scripts/test_check_dsh_plan3_drain.py`. It proves exactly two
  lexical/schema occurrences in `script_operations.py`, both inside the named
  drain function, with exact worker, collection schema, status, and
  delivery-state filters.
- Reconciled `Static Decommission Checks`: the broad obsolete-contract scan
  excludes only `src/kazusa_ai_chatbot/db/script_operations.py`; its dedicated
  positive `rg` audit reports exactly lines 128 and 134, and the owner test
  proves their complete filter shape. The revised broad scan reports only the
  seven intentional surviving test references at
  `tests/test_background_work_jobs.py:169`,
  `tests/test_dsh_plan3_documentation.py:55`,
  `tests/test_task_resolution_background_resume.py:175`,
  `tests/unit/action_spec/test_accepted_task_control.py:87` and `:200`, and
  `tests/unit/background_work/test_dsh_jobs.py:248` and `:331`.
- `venv\\Scripts\\python -m pytest -q tests/unit/scripts/test_check_dsh_plan3_drain.py`
  exited 0 in 0.99s: four owner tests passed. The dedicated command
  `venv\\Scripts\\python -m pytest -q
  tests/unit/scripts/test_check_dsh_plan3_drain.py -k
  "v1_payload_exception"` exited 0 in 0.83s: one passed and three
  deselected.
- `venv\\Scripts\\python -m pytest -q tests/test_dsh_plan3_documentation.py
  tests/test_test_impact_manifest.py` exited 0 in 0.96s: 20 passed.
  `venv\\Scripts\\python -m pytest -q
  tests/unit/task_resolution/test_decommission.py` exited 0 in 2.58s: five
  passed.
- `venv\\Scripts\\python -m scripts.validate_test_impact --base-ref HEAD
  --run` exited 0 in 12.64s: 158 exact impact-test nodes passed.
- The revised static audits exited as follows: import/deleted-runtime audit
  1 with no matches; obsolete-contract audit 0 with only the seven intentional
  test references above; positive drain audit 0 with exactly two source lines;
  obsolete-environment audit 1 with no matches; stale-catalog audit 1 with no
  matches; and positive RAG3-retention audit 0 with only expected retained
  imports. The amendment owner Ruff check exited 0 with no diagnostics, and
  `git diff --check` exited 0 with only repository LF-to-CRLF normalization
  warnings.
- `venv\\Scripts\\python -m pytest -m "not live_db and not live_llm" -q`
  exited 0 in 219.30s: 3,395 passed, four skipped, 523 deselected, and two
  warnings. The new owner test is included in this full non-live evidence.
- This amendment changed only the active plan ledger/static command and the
  drain owner test. P3-P2 retains the previously green owner, impact,
  sidecar, and authorized live-DB evidence; named live-LLM E2Es, deployment
  and drain rehearsal authorization, and independent review remain open.

### 2026-08-30 — P3-P2 Drain-Owner Manifest Reconciliation

- Adding the new owner test reference to the plan correctly exposed one
  missing ownership row: the first rerun of the documentation/manifest
  command exited 1 with 19 passed and
  `test_manifest_contains_dsh_plan3_owner_rows` failing for the new drain
  owner node. Added that exact node to the existing
  `src/kazusa_ai_chatbot/db/script_operations.py` manifest row.
- Final owner verification exited 0: drain tests, four passed in 0.93s;
  documentation and manifest tests, 20 passed in 0.97s; and decommission
  tests, five passed in 2.84s. Ruff exited 0 with no diagnostics and
  `git diff --check` exited 0 with only repository LF-to-CRLF normalization
  warnings.
- `venv\\Scripts\\python -m scripts.validate_test_impact --base-ref HEAD
  --run` exited 0 in 10.39s with 159 exact impact-test nodes passed, including
  the new owner gate and all removed-source mappings.
- The full non-live rerun after the manifest reconciliation exited 0 in
  215.09s: 3,395 passed, four skipped, 523 deselected, and two warnings.
  The exact drain-owner exception is therefore covered by the final full
  suite and manifest ownership. No product/runtime source changed in this
  reconciliation; live-LLM E2Es, deployment/drain authorization, and
  independent review remain the recorded residual gates.

### 2026-08-30 — P3-P3 First Real-LLM Gate Attempt

- Applied the `character-test` and `test-style-and-execution` workflows for
  the first P3-P3 real-LLM node. The exact command
  `venv\\Scripts\\python -m pytest -m live_llm
  tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_inline_grounded_resolution_reenters_full_cognition
  -q -s` exited 0 in 1.63s wall time (`pytest` reported 0.81s): one test
  skipped because the node body is still the explicit
  `pytest.skip("live LLM signoff is deferred until the production cutover")`.
- The selected node contains no dialog payload, service/API call, live model
  invocation, log capture, or artifact writer. Consequently no response,
  DSH terminal/evidence projection, V1 observation, cognition/dialog
  recurrence, lineage, or background-error behavior was produced or judged.
  No new raw artifact was generated; existing historical artifacts remain
  preserved under `test_artifacts/llm_traces/`.
- This is a deferred test-contract blocker rather than an environment or
  startup failure. The first P3-P3 real-LLM gate remains unexecuted and needs
  an authorized live harness/test implementation before behavioral signoff.
  No production deployment or data operation was attempted.

### 2026-08-30 — P3-P3 Inline Real-LLM Node Implementation And Startup Check

- Replaced only the first Plan 3 live skeleton node in
  `tests/test_dsh_plan3_e2e_live_llm.py`. The node uses the existing Brain
  `/runtime/dsh/health` and `/chat` boundaries, sends one real user-wire task
  request, creates one unique in-workspace evidence fixture, reads the exact
  DSH root session in SQLite read-only mode, reads the protected Mongo trace
  rows, and writes request, response, lineage, trace, and qualitative audit
  artifacts under the ignored `test_artifacts/debug_runs/` surface. The
  assertions require inline terminal state, no accepted/background identity,
  `TaskResolutionResultV1` evidence, exact thread/segment/session and Brain
  episode lineage, terminal evidence containing the fixture marker, and two
  complete `cognition_core_v3.A1/A2/G/P` passes. No other Plan 3 live node was
  changed or run.
- The exact command
  `venv\Scripts\python -m pytest -m live_llm
  tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_inline_grounded_resolution_reenters_full_cognition
  -q -s` exited 1 in 3.98s wall time. The configured Brain HTTP startup
  request failed with the redacted transport class `ConnectError` before
  `/runtime/dsh/health` returned; therefore no DSH model call, terminal
  evidence, V1 observation, cognition recurrence, dialog, or lineage behavior
  was available for judgment. The durable artifact is
  `test_artifacts/debug_runs/plan3_inline_20260830T004226Z_fe31e5f1/` and
  contains no credentials or authorization headers.
- `venv\Scripts\python -m ruff check
  tests/test_dsh_plan3_e2e_live_llm.py` exited 0 with no diagnostics. The
  implementation leaves the four remaining live nodes as their original
  deferred skeletons. Brain/sidecar startup and a rerun of this one node are
  the residual P3-P3 behavioral gate; no production deployment or data
  operation was attempted.

### 2026-08-30 — P3-P3 External Brain/Sidecar Startup Verification

- Applied the P3-P3 operator-startup boundary using only the current exported
  process environment. The exact read-only check found zero of the 19 required
  live-LLM, DSH, identity, and Mongo settings exported in the process,
  including `KAZUSA_DSH_BRAIN_URL`, `KAZUSA_DSH_SIDECAR_URL`,
  `KAZUSA_DSH_DATA_ROOT`, `KAZUSA_DSH_RPC_TOKEN`,
  `KAZUSA_DSH_BRAIN_SHARED_SECRET`, `KAZUSA_DSH_TOOL_GATEWAY_SECRET`,
  `AGENTIC_RESOLVER_WORKSPACE_ROOT`, `MONGODB_URI`, and
  `MONGODB_DB_NAME`. No URL host/port or authorized isolated database target
  could therefore be derived or verified, and no exact configured ports could
  be checked.
- The pinned local executables are present (`venv/Scripts/python.exe`,
  `src/kazusa_ai_chatbot/main.py`, and
  `sidecars/dsh_resolution/dist/src/main.js`). Since the required exported
  configuration is absent, no Brain or sidecar process was started, no PID or
  readiness response exists, and no process/log cleanup was required. The
  prior single-node rerun remains the only node execution and its
  `ConnectError` artifact is recorded above.
- The exact first-node command remains blocked at the approved local startup
  prerequisite; no `.env` was read, sourced, or used for process startup. No
  production deployment or database operation was attempted. The residual is
  an operator-provided exported environment containing the named loopback
  endpoints, pinned LLM route, DSH secrets/roots, and the already authorized
  isolated Mongo target before this one live node can be retried.

### 2026-08-30 — P3-P3 Owned Local Startup Harness And Port Blocker

- Updated only the first node in `tests/test_dsh_plan3_e2e_live_llm.py` to
  own its local execution boundary after the pytest configuration guard. It
  now validates the configured Brain and sidecar HTTP URLs as loopback,
  requires distinct explicit ports, checks each port before claiming it,
  starts the pinned sidecar and `venv\\Scripts\\python -m uvicorn
  kazusa_ai_chatbot.service:app` with hidden child-process flags, sets
  `PYTHONPATH=src`, uses `tmp_path` DSH/workspace roots and the forced
  `_test_kazusa_live_llm` database, captures timestamped stdout/stderr and
  PIDs, waits for authenticated Brain and sidecar readiness, and terminates
  only the child handles it started in `finally`. Preflight failures produce
  a redacted `startup_failure.json` artifact.
- The exact required command was rerun twice after the harness change:
  `venv\\Scripts\\python -m pytest -m live_llm
  tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_inline_grounded_resolution_reenters_full_cognition
  -q -s`. Both attempts stopped before child launch because the configured
  sidecar port `56607` is already occupied. The final run exited 1 in about
  1.20s of pytest time and reported `configured sidecar port 56607 is already
  occupied`. `Get-NetTCPConnection` identifies an existing `node.exe` PID
  `57308` running the pinned sidecar command since 2026-08-29; it is not owned
  by this test and was preserved. The redacted startup artifact is
  `test_artifacts/debug_runs/plan3_inline_20260830T005708Z_2d05a8d4/startup_failure.json`;
  it records the loopback Brain/sidecar endpoints, zero started processes,
  and no credentials. No Brain/sidecar logs or DSH/trace/behavior evidence
  could be produced because ownership preflight correctly rejected the
  occupied port.
- `venv\\Scripts\\python -m ruff check
  tests/test_dsh_plan3_e2e_live_llm.py` exited 0; `venv\\Scripts\\python -m
  py_compile tests/test_dsh_plan3_e2e_live_llm.py` exited 0;
  `venv\\Scripts\\python -m scripts.validate_test_impact --base-ref HEAD
  --run` exited 0 with 159 exact impact nodes passed; and `git diff --check`
  exited 0 with only repository LF-to-CRLF normalization warnings. No other
  live node, production deployment, or database operation was run. The
  remaining P3-P3 blocker is release/cleanup of the pre-existing PID 57308
  (or an explicit operator-owned port handoff) so this test can start both
  services itself and perform the real model/Brain/Mongo behavior gate.

### 2026-08-30 — P3-P3 First Live Node Owned-Process Completion

- Revalidated the reported orphan before cleanup: PID `57308` was still the
  exact `node.exe sidecars/dsh_resolution/dist/src/main.js` command, owned the
  loopback listener `127.0.0.1:56607`, and its recorded parent PID `51436` was
  absent. Stopped exactly PID `57308` with `Stop-Process -Id 57308 -Force` and
  verified that PID was absent afterward and no listener remained on port
  `56607`. No other process was stopped.
- The first owned-process rerun reached a real DSH terminal but exposed a
  projection defect: native `read` supplied validated terminal artifact refs
  and findings without a semantic evidence receipt. The narrow
  `src/kazusa_ai_chatbot/task_resolution/projection.py` correction projects
  bounded artifact refs as DSH evidence rows only when the receipt list is
  empty; `tests/unit/task_resolution/test_projection.py` now covers this exact
  no-receipt shape. The mapped owner command
  `venv\Scripts\python -m pytest -q tests/unit/task_resolution/test_projection.py
  tests/unit/cognition_resolver/test_capabilities.py
  tests/test_dsh_plan3_task_resolution.py` passed 14 tests.
- The live node also now follows the documented normal-adapter trace boundary:
  top-level `ChatResponse.trace_id` is empty for its non-debug platform, while
  the protected trace is taken from `cognition_graph.correlation.llm_trace_id`.
  The isolated child environment uses the approved 120-second upper bound for
  `TASK_RESOLUTION_INLINE_BUDGET_SECONDS` to avoid a slow real-model call
  racing the default foreground budget, and the node polls the established
  post-response trace finalizer before asserting terminal status.
- The exact required command was retried after each concrete correction. The
  run at `test_artifacts/debug_runs/plan3_inline_20260830T010845Z_83348b4d/`
  produced a grounded text response but failed the trace-boundary assertion;
  the run at
  `test_artifacts/debug_runs/plan3_inline_20260830T011118Z_1ca1af3e/` exposed
  an inline-budget stale-lease race; the run at
  `test_artifacts/debug_runs/plan3_inline_20260830T011344Z_c1d3523e/`
  completed the full behavior path but observed the trace row before its
  asynchronous finalization. The final exact command
  `venv\Scripts\python -m pytest -m live_llm
  tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_inline_grounded_resolution_reenters_full_cognition
  -q -s` exited 0 in 102.41s with one test passed. Its durable artifact is
  `test_artifacts/debug_runs/plan3_inline_20260830T011623Z_7cbe0685/`.
- Qualitative inspection of the final artifact found authenticated Brain and
  sidecar readiness HTTP 200, one `resolved` DSH terminal receipt, exact
  `task_session_id`/resolution-thread/segment identity in the binding and
  terminal receipt, `consumed_inline` state with no accepted task or background
  job, complete `TaskResolutionResultV1` DSH evidence, the exact fixture marker
  in the terminal evidence and visible Brain-owned dialog, and a succeeded
  protected trace with 12 steps and two complete
  `cognition_core_v3.A1/A2/G/P` passes. The owned sidecar PID `56456` and Brain
  PID `46256` were both terminated by the node's `finally` cleanup; port 8000
  and port 56607 had no listeners afterward (only normal TIME_WAIT entries).
  Sidecar stderr was empty and the Brain log contained readiness, one real
  task exchange, and grounded dialog output without a background error.
- Final checks passed: Ruff on
  `src/kazusa_ai_chatbot/task_resolution/projection.py`,
  `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`,
  `tests/unit/task_resolution/test_projection.py`,
  `tests/unit/cognition_resolver/test_capabilities.py`, and
  `tests/test_dsh_plan3_e2e_live_llm.py`; `py_compile` on the same five Python
  files; `git diff --check` (exit 0 with repository LF-to-CRLF normalization
  warnings); and
  `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run`
  (exit 0, 159 exact nodes passed in 9.67s). No other live node, production
  deployment, or production data mutation was run. The remaining P3-P3
  residual is the other four named live-LLM nodes plus the plan's deployment,
  drain authorization, and independent-review gates.

### 2026-08-30 — P3-P3 Second Real Debug-User Node Implementation And LLM Routing Blocker

- Applied `character-test`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `cjk-safety`, and
  `test-style-and-execution` to implement only
  `tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_real_debug_user_replies_to_dsh_relay_and_resumes_same_session`.
  The node owns an isolated temporary DSH workspace and `_test_kazusa_live_llm`
  database environment, starts the pinned sidecar and Brain as hidden child
  processes, registers a real loopback HTTP adapter through
  `/runtime/adapters/register`, sends both fixed candidates only through
  `/chat`, and records the required turn, relay, lineage, delivery, and
  qualitative artifacts. The node stops the processes it starts in `finally`;
  no semantic owner, adapter production path, deployment, or production data
  was modified.
- The exact command
  `venv\\Scripts\\python -m pytest -m live_llm
  tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_real_debug_user_replies_to_dsh_relay_and_resumes_same_session
  -q -s` was executed four times. The first two runs were interrupted during
  the 420-second relay wait after the public Turn 1 response and produced
  `test_artifacts/debug_runs/plan3_real_debug_user_20260830T013410Z_6fadf8d3/`
  and
  `test_artifacts/debug_runs/plan3_real_debug_user_20260830T014251Z_aa8b6449/`.
  The third run completed in 127.01s and exited 1 after the tightened 45-second
  relay wait; its artifact is
  `test_artifacts/debug_runs/plan3_real_debug_user_20260830T015219Z_20a70f2a/`.
  The fourth run exercised the post-patch failure artifact, completed in
  125.33s, and exited 1 with the same relay-wait result; its artifact is
  `test_artifacts/debug_runs/plan3_real_debug_user_20260830T015759Z_2c114ad9/`.
- All four attempts reached authenticated Brain and sidecar readiness (HTTP
  200) and executed a real Turn 1 model call. Protected cognition traces show
  two complete cognition passes, but both P-stage outputs selected
  `resolver_requests=[human_clarification]` with no
  `task_resolution_request`. The public response asked the user to choose a
  file, while the registered adapter received zero capability or delivery
  callbacks and the isolated Mongo scope contained zero DSH binding, accepted
  task, background job, or pending-interaction rows. No Turn 2 was sent, so
  reply lineage, one-shot interaction consumption, continuation, result-ready
  cognition/dialog recurrence, and beta-only delivery could not be judged.
  This is a reproducible live LLM semantic-routing contract blocker; the
  approved slice forbids changing cognition semantic owners or injecting a
  deterministic route into the public test harness.
- The fourth artifact records the exact fixed texts and observation targets,
  loopback URLs, forced isolated database name, readiness payloads, child PIDs,
  response, protected-trace-backed routing result, zero callbacks, and the
  failure/conclusion status. Each run's Brain/sidecar children were cleaned up;
  the unrelated pre-existing sidecar process was preserved. No other live node
  was implemented or run.
- `venv\\Scripts\\python -m ruff check
  tests/test_dsh_plan3_e2e_live_llm.py` exited 0; `venv\\Scripts\\python -m
  py_compile tests/test_dsh_plan3_e2e_live_llm.py` exited 0; the mapped
  deterministic owner command passed 41 tests in 0.99s; the impact validator
  passed 159 exact nodes in 9.67s; and `git diff --check` exited 0 with only
  repository LF-to-CRLF normalization warnings. The second live node stays red
  on the external model-routing contract until its owning semantic stage
  produces the required `task_resolution_request`.

### 2026-08-30 — Approved P3 Semantic-Routing Amendment And Verification

- The parent-approved amendment resolves the producer-side semantic boundary
  identified by the four preceding real debug-user attempts. A current request
  with a bounded executable task objective is now described as
  `task_resolution_request` even when execution may later ask the user a
  question. `human_clarification` is described as requesting a missing
  user-controlled fact that prevents cognition from formulating or authorizing
  a bounded task objective; a later execution-time question belongs to task
  resolution. This is a positive affordance-description change only: the
  existing `cognition_resolver.contracts.RESOLVER_CAPABILITY_SEMANTICS` map is
  already projected by `_available_resolver_affordances()` into the canonical
  P-stage packet. No enum, route, keyword classifier, post-processing override,
  fixture-specific example, or semantic owner changed, and `prompt.py` needed
  no edit.
- Added
  `tests/unit/cognition_core_v3/test_handleless_contract.py::test_plan_packet_distinguishes_task_admission_from_clarification`.
  It builds the actual P-stage packet from the resolver affordance map and
  checks both exact projected descriptions plus the generic bounded-objective,
  missing-fact, and later-question distinction without a scenario fixture.
- Changed paths for this amendment are limited to
  `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`,
  `tests/unit/cognition_core_v3/test_handleless_contract.py`, and this ledger.
  `venv\\Scripts\\python -m ruff check
  src/kazusa_ai_chatbot/cognition_resolver/contracts.py
  tests/unit/cognition_core_v3/test_handleless_contract.py` and `venv\\Scripts\\python
  -m py_compile` on the same two Python files both exited 0. The direct owner
  command covering handleless cognition, DSH task affordances, resolver
  capabilities/contracts, and Plan 3 task resolution passed 61 tests in 1.31s.
  A broader mapped command also exercised `tests/test_cognition_resolver_loop.py`;
  it passed 102 tests and retained five existing evidence-handle failures in
  the shared task-result projection path (`task result evidence handle is
  missing its semantic reference`), outside this mapping-only amendment.
- `venv\\Scripts\\python -m scripts.validate_test_impact --base-ref HEAD --run`
  exited 0 with 162 exact impact-test nodes validated. The count reflects the
  current shared Plan 3 worktree and the added deterministic owner node.
- The required single second live command was rerun exactly as
  `venv\\Scripts\\python -m pytest -m live_llm
  tests/test_dsh_plan3_e2e_live_llm.py::test_e2e_real_debug_user_replies_to_dsh_relay_and_resumes_same_session
  -q -s`; it reached authenticated Brain and sidecar readiness, then exited 1
  after 162.35s when the configured local cognition route at `127.0.0.1:8080`
  refused connections. Brain logged A1 contract exhaustion after the LLM
  connection errors and returned the typed `model_contract` operational error
  before P-stage routing, so this attempt has no protected trace id, DSH
  binding, callback, or Turn 2 lineage to judge. The complete failure artifact
  is
  `test_artifacts/debug_runs/plan3_real_debug_user_20260830T030524Z_74306838/`;
  its `run.json`, readiness, request, conclusion, and Brain/sidecar logs record
  the exact state. The node-owned sidecar PID 32548 and Brain PID 47292 were
  absent after `finally` cleanup, and ports 8000 and 56607 had no listeners.
  An independent endpoint probe also reported `127.0.0.1:8080`
  `TcpTestSucceeded=False`. No other live node, deployment, or data mutation
  was run, and `.env` was not inspected.
- The semantic-routing amendment is deterministic-owner green but remains
  live-behavior unverified until the configured local LLM route is available;
  rerun of this one named node is the next bounded action once that external
  dependency is restored.

### 2026-08-30 — P3 Task-Result Evidence Projection Fixture Reconciliation

- The five failures reported by the first broader mapped run were exactly:
  `tests/test_cognition_resolver_loop.py::test_task_resolution_uses_objective_and_preserves_context`,
  `::test_task_resolution_bounds_history_to_its_context_contract`,
  `::test_task_resolution_bounds_decontextualized_summary`,
  `::test_internal_thought_uses_unified_task_resolution_path`, and
  `::test_public_evidence_projects_through_task_resolution`. All five failed
  at the same existing Plan 3 guard in
  `_task_resolution_evidence_refs`: the test helper emitted a
  `TaskResolutionEvidenceV1.summary` containing human-readable excerpt text,
  while `evidence_handles` contained `evidence-1`.
- The approved projection contract is already explicit: `evidence_handles`
  are ordered semantic references, each projected evidence row has `summary`
  equal to its semantic reference, and human-readable findings remain in
  `evidence_excerpts`. The DSH projection owner and its dedicated unit tests
  already satisfy that contract; the omission was the migrated shared-loop
  fixture, not a production behavior gap. No production scope expansion,
  compatibility fallback, semantic-reference invention, or validator change
  was authorized or made.
- Reconciled only
  `tests/test_cognition_resolver_loop.py`: its canonical `_task_result()` now
  sets the evidence-row summary to `evidence-1` while retaining each test's
  supplied text in `evidence_excerpts`/`prompt_safe_summary`. The first
  context test additionally asserts the projected evidence id and excerpt,
  strengthening deterministic owner coverage for the V1 evidence boundary.
- The exact five-node command passed 5 tests in 1.12s. The full mapped command
  covering handleless cognition, DSH task affordances, resolver
  capabilities/contracts/loop, and Plan 3 task resolution passed 107 tests in
  1.39s. Ruff and `py_compile` on the changed test and previously amended
  Python owners exited 0; the impact validator remained green at 162 exact
  nodes; and `git diff --check` exited 0 with repository line-ending warnings.
  No live LLM node was run while `127.0.0.1:8080` remained unavailable.

### 2026-08-30 — P3-P3 Live Checkpoint Identity Projection Amendment

- The first post-routing retry of the exact second live node reached the
  intended `task_resolution_request`, admitted the DSH task, and delivered
  one authenticated relay question. The node then failed at the Brain task
  result projection with `DSH checkpoint does not contain a valid resolution
  reference`; no second turn was sent. Its durable binding already contained
  the complete `dsh_resolution_ref.v1` identity and the interaction row was
  `delivered`, so the failure is in the canonical DSH exhaust-to-V1 projection
  boundary rather than semantic routing, callback delivery, or the live
  harness.
- The sidecar/controller contract returns a checkpointed exhaust whose
  checkpoint payload is empty while the runtime-authored exhaust identity
  carries the exact thread, segment, session, activation, lease, revision, and
  sequence fields. The existing inline reference extractor already validates
  that identity shape. This bounded amendment lets the projection consume
  that same typed runtime identity when the checkpoint mapping is empty, then
  runs the unchanged strict `validate_dsh_resolution_ref` validator. It adds
  no fallback identity, semantic inference, or compatibility vocabulary.
- Add deterministic coverage for the empty-checkpoint/runtime-identity shape,
  run the mapped task-resolution and cognition owners, retry only the exact
  second live node, and retain the failed artifact and protected trace for
  qualitative comparison. The amendment remains limited to the approved DSH
  task-result projection owner and its tests; no other live node or deployment
  operation is in scope.

### 2026-08-30 — P3-P3 Semantic Priority Projection Amendment

- The subsequent exact second-node retry reached the intended
  `task_resolution_request` and delivered the DSH question, but its durable
  binding had no accepted task or background job. The protected binding
  `start_spec.resolver_request.start_in_background` was `false` even though
  the current user explicitly requested background handling. The producer
  contract exposed the capability distinction but omitted the P-stage
  background choice, while the caller materializer hardcoded `priority=now`.
- This is a bounded P3 contract omission: the plan assigns task objective,
  priority, and response goal to Cognition P/L2d, and the live contract
  requires direct background admission before the DSH question relay. Extend
  the existing typed P-stage resolver row only for
  `task_resolution_request` with the model-owned `start_in_background` boolean,
  validate it strictly, and project it into the existing V1 `priority` field.
  Non-task resolver rows keep their existing shape; omitted or inferred
  priority, keyword routing, post-processing overrides, fallback aliases, and
  scenario-specific examples remain forbidden.
- The approved owner set for this narrow projection is the P-stage prompt
  contract/validator, the existing canonical resolver materializer, and their
  deterministic tests, plus the already-approved second live node. Run the
  mapped deterministic owners and this one live node only; preserve the
  earlier failed artifact and protected trace for comparison.

### 2026-08-30 — P3-P3 Runtime Identity And Admission Repetition Amendment

- The latest exact second-node run reached the intended background
  `task_resolution_request` and delivered the current DSH relay question, but
  the worker failed while projecting its checkpoint: the sidecar checkpoint
  payload was empty and the runtime exhaust identity omitted
  `last_committed_seq`. The durable binding already held the canonical
  `dsh_resolution_ref.v1` with sequence zero, so this is an attributable
  runtime/projection contract omission. The inspected artifact is
  `test_artifacts/debug_runs/plan3_real_debug_user_20260830T041033Z_8dfa2f4a/`;
  protected trace `llmtrace_e189377c9f0648f589be2470c404de37` recorded the
  failure and callback message `b73df454b0c84c018c30aa75ad2577f6` recorded
  the delivered question.
- The same cognition turn also admitted three task rows with the same typed
  `goal_continuation_ref` digest while the model paraphrased the objective.
  The resolver loop's objective-only repetition guard therefore allowed
  duplicate admissions and stale callbacks, contradicting the direct-
  background rule that an active matching goal/scope reuses its task. This is
  a second attributable omission at the retained typed resolver-loop boundary.
- The bounded amendment permits only: completion of the runtime-authored
  typed identity needed by the existing strict checkpoint projection;
  repetition blocking when the same capability carries the same typed
  `goal_continuation_ref`; deterministic owner coverage for both; and one
  retry of the already-approved exact second live node. It adds no identity
  default beyond the already persisted binding reference, no semantic
  reclassification, keyword logic, fallback alias, compatibility route, or
  later live/deployment/data operation.

### 2026-08-30 — Character-Owned DSH Interaction Amendment Approved

- The user confirmed that DSH is owned by the character rather than the user
  and that DSH questions, approvals, and plan reviews are handled by the
  existing full reusable cognition loop. They never prompt the user.
- Code inspection established that
  `nodes.persona_supervisor2_cognition.run_dsh_interaction_cognition` currently
  calls one direct `run_cognition(...)` pass, while
  `nodes.persona_supervisor2.stage_1_goal_resolver` owns the reusable resolver
  recurrence and final cognition commit. The amendment requires the latter
  ownership and removes the one-pass DSH shortcut.
- Sidecar and Brain inspection established that the current V1 bridge drops
  later question items, hides exact executable approval arguments from
  cognition, and implements `relay_to_user` through checkpoint, delivery,
  pending-reply, and same-thread continuation owners. The atomic V2 contract
  closes those gaps with one complete bounded semantic handoff, exact internal
  decisions, and removal of the relay/reply lifecycle.
- The earlier forced semantic-routing amendment is superseded for the fixed
  alpha/beta fixture. Its explicit user-controlled file choice prevents a
  bounded authorized objective on Turn 1, so ordinary character cognition must
  clarify before DSH admission. Turn 2 may admit DSH after the user supplies
  the choice. This is the revised exact second live-node contract.
- The same persistent `dsh_p3_implementation_worker` on `gpt-5.6-luna` with
  maximum reasoning owns the production/test slice after this plan amendment.
  The repository-root `README.md` remains unchanged until all code and live
  behavior work is complete and the existing final documentation audit finds
  a material product-level correction necessary.

### 2026-08-30 — V2 Interaction First-Slice Evidence And Boundary Audit

- The persistent worker implemented the first authoritative V2 slice and
  reported 149 focused deterministic tests passing, five sidecar interaction
  tests passing, sidecar typecheck passing, 215 exact impact nodes validated,
  Python compilation and `git diff --check` passing, and no live/data/deployment
  operation. `README.md` remained clean.
- Parent source audit confirmed V2 wire traffic and removal of the Brain
  checkpoint/reply/delivery functions, but found remaining
  `waiting_for_interaction` state across accepted-task/background/binding
  owners plus `find_open_interaction(...)` in the task orchestrator. It also
  found V1 DTO class names attached to V2 schemas, user-solicitation resolvers
  still advertised inside internal DSH episodes, and a new rejection of
  open-ended DSH questions.
- The closed `Post-Implementation Boundary Audit Amendment` above records the
  exact cleanup scope and owner nodes. The same worker must make that cleanup
  green before parent acceptance of the V2 deterministic candidate.

### 2026-08-30 — Character-Owned V2 Deterministic Implementation Evidence

- The same persistent `dsh_p3_implementation_worker` completed the closed V2
  cleanup. The accepted runtime contains no DSH `waiting_for_interaction`
  state or transition, `find_open_interaction(...)` query, V1 interaction DTO
  or grant symbol, V1 nonce domain, relay/checkpoint route, pending reply, or
  DSH user-delivery owner. The canonical public vocabulary is V2 throughout.
- Internal DSH cognition advertises `self_goal_resolution` only, preserves a
  complete open-ended or choice-bearing single question, rejects a
  multi-question bundle, and returns the validated per-kind character decision
  directly to the waiting DSH hook. The sidecar uses the V2 nonce domain and
  preserves exact approval arguments and complete bounded semantic detail.
- Parent verification passed 117 focused Python tests, then the full exact
  impact validator passed all 223 selected nodes. Python compilation over all
  changed Python files, the CJK prompt AST parse, focused V2 Ruff, and
  `git diff --check` passed. Sidecar typecheck and build passed, and all five
  Brain-interaction Vitest cases passed.
- A final same-worker cleanup removed the one amendment-attributable unused
  service import. Its seven focused service/composition tests, Python
  compilation, and diff check passed. Ruff no longer reports that F401; the
  shared service file still reports its separately existing broader lint
  diagnostics, so no repository-wide Ruff pass is claimed.

### 2026-08-30 — Character-Owned V2 ICD Consistency Evidence

- Post-code audit found that three subsystem ICD sections still described the
  superseded relay/reply contract. The `Post-Code Documentation Consistency
  Amendment` above registered the missing documentation/test ownership before
  edits, and the same persistent worker changed only those three ICD sections
  plus their exact static test.
- `src/kazusa_ai_chatbot/dsh_interaction/README.md`, the DSH section of
  `src/kazusa_ai_chatbot/brain_service/README.md`, and the DSH section of
  `src/kazusa_ai_chatbot/cognition_core_v3/README.md` now describe the
  character-owned V2 internal cognition boundary. The exact new test
  `test_character_owned_dsh_icds_exclude_user_relay_contract` passed under both
  worker and parent execution, and the retired relay/checkpoint vocabulary is
  absent from those sections.
- The full documentation file currently reports three passed and two failed.
  The remaining failures are the previously recorded broader P3-P3 root
  documentation-bundle checks for legacy `coding_agent` text and the missing
  `epoch` term; they are not waived by this amendment.
- Repository-root `README.md` remains byte-for-byte unchanged in the working
  diff. Its product-level necessity audit stays deferred until the remaining
  live behavior, broader documentation, independent-review, and closure gates
  are complete.

### 2026-08-31 — P3 Node 2 Pending Task Timing Amendment Approved

- The failed prerequisite artifact established that the answered Turn 2 plan
  selected `start_in_background=false` after the original blocked request had
  explicitly required later/background delivery. The pending row retained the
  original goal and goal-progress checklist, but no typed P-authored timing
  value survived the clarification boundary. The valid false value therefore
  materialized as foreground priority, completed DSH inline, and consumed the
  binding before background delivery could begin.
- Add one closed P-authored `pending_task_timing` value: `none`, `foreground`,
  or `background`. P emits it exactly with one ordinary `human_clarification`
  request and never otherwise. It is semantic state only: Turn 1 still admits
  no task, job, binding, or DSH work. Persist the validated value in the
  big-bang `resolver_pending_resume.v2` ledger contract and project it through
  `resolver_pending_continuation.v2` without a durable identifier.
- An answered pending continuation may admit a task only when its typed
  `start_in_background` matches `foreground` or `background`; `none` forbids
  task admission. Non-answered dispositions remain ineligible. A changed user
  request uses `superseded`, closes the prior row, and enters normal cognition
  as a fresh semantic decision. Deterministic code validates typed
  cross-field consistency and routes mismatches through bounded P regeneration;
  it does not infer, default, parse, or rewrite timing from user text.
- The bounded owner set is cognition-core typed plan/output-contract
  projection and validation, resolver pending V2 contracts/persistence, the
  cognition node state projection, their direct deterministic tests, resolver
  loop integration coverage, and the source-impact manifest. Complete system
  prompt literals, dialog/L3, DSH runtime, adapters, deployment, production
  data, and repository README files remain outside this amendment. Open V1
  pending rows require a separately authorized drain or expiry operation and
  receive no compatibility interpretation or upgrade.

### 2026-08-31 — P3 Node 2 Surface Control-Plane Boundary Amendment Approved

- The prerequisite-admission artifact established that L3 copied the complete
  cognition response plan after excluding only the DSH decision. The
  P-owned `pending_task_timing` object then entered the model-facing surface
  packet and correctly failed prompt-projection validation on its raw schema.
- L3 receives an explicit allowlist of the visible response-plan semantics it
  owns: `response_goal`, `goal_resolution`, `epistemic_boundary`,
  `action_requests`, and `resolver_requests`. Pending timing and pending
  disposition remain cognition/control-plane state and never enter L3.
- The amendment changes only the L3 projection and its deterministic tests;
  it preserves prompt validation, complete system prompts, prompt wording,
  timing semantics, resolver persistence, and all other runtime owners.

### 2026-08-31 — P3 Node 2 Answer-Conditioned Pending Continuation Amendment Approved

- Artifact `prerequisite_admission_20260830T164154Z_4eebe380` proved that the
  V1 `pending_task_timing=none` value is ambiguous during a Turn 1
  clarification: it can mean no task in the current turn to the model, while
  the validator interprets it as forbidding a task after the user answers.
  Turn 2 therefore exhausted bounded P regeneration while correctly selecting
  the task after the prerequisite answer.
- Replace that carrier big-bang with the P-authored
  `pending_task_continuation.v1` object containing exactly
  `schema_version` and `on_answered_clarification`. Its closed values are
  `no_task_admission`, `foreground_task_admission`, and
  `background_task_admission`. It is required only with one ordinary
  `human_clarification`, persists only on that pending row, and remains
  model-safe continuity without durable identity.
- Resume and prompt-continuation contracts advance to V3. V1 and V2 pending
  rows fail closed without interpretation, fallback, translation, or upgrade;
  their drain or expiry remains separately authorized operational work. An
  answered continuation may request a task only when the stored disposition
  permits it and its nested `start_in_background` agrees. Non-answered and
  superseded dispositions retain the existing admission boundary.
- The bounded implementation changes resolver contracts and pending storage,
  cognition-core typed plan/validation and data-only output-contract
  projection, persona cognition/state carriage, direct deterministic and live
  fixture coverage, plus impact ownership. Complete system prompts, L3,
  dialog, DSH runtime, adapters, README files, and unrelated documentation
  remain unchanged.

### 2026-08-31 — P3 Node 2 Pending-P Output Variant Amendment Approved

- Artifact `prerequisite_admission_20260830T170143Z_71141c39` established that
  a valid stored `background_task_admission` continuation was echoed by P on
  the answered pending turn, causing bounded regeneration to reject a
  clarification-only output carrier.
- `pending_task_continuation` is authored only when P creates one ordinary
  new `human_clarification`. Pending-resolution P contracts project the stored
  continuation solely as input and omit that output field; they resolve the
  selected row through `answered`, `continue_waiting`, `rejected`, or
  `superseded` and cannot create another human-clarification request.
- Deterministic validation enforces the variant boundary and invokes existing
  bounded P regeneration without semantic rewriting. A superseded row closes,
  and a later ordinary cognition recurrence may create a new clarification.

### 2026-08-31 — P3 Node 2 Post-Answer Recurrence Variant Amendment Approved

- Artifact `prerequisite_admission_20260830T171434Z_d89d8c6d` proved that the
  open-pending P call answered the clarification and admitted the exact beta
  task in the background before the failure. The resolver loop then appended
  the task-admission observation and invoked cognition again. At that point
  `resolver_pending_resolution` still carried the already-selected disposition,
  while the open `pending_resolver_continuation` input lane was intentionally
  suppressed. P therefore received the fresh-ordinary output contract, which
  advertised the clarification-only `pending_task_continuation` field again;
  repeated candidates emitted it and exact validation exhausted. The isolated
  pending-answer live case covered only the preceding open-pending call and
  could not expose this later recurrence.
- Add one typed, non-persistent response-plan contract variant derived only
  from validated resolver lifecycle state. `fresh_ordinary` applies when no
  pending continuation or selected pending disposition is active;
  `open_pending_resolution` applies while P must decide an open pending row;
  `post_pending_resolution` applies to later cognition calls in the same
  resolver execution after P has already selected that row's disposition.
  The output-contract projection and canonical validator consume the same
  variant. The post-pending variant does not request another pending
  disposition, omits `pending_task_continuation`, and forbids a new
  `human_clarification`; it preserves task-result and other resolver evidence
  processing. After a superseded row has closed and its lifecycle state is no
  longer active, a later independent cognition execution uses
  `fresh_ordinary` and may form a new clarification normally.
- This is deterministic contract selection, not semantic judgment: P retains
  ownership of the pending disposition, task admission, and background timing.
  Invalid output remains fail-closed and goes through bounded regeneration;
  no echoed field is accepted, stripped, rewritten, or translated. The
  variant is not persisted, is not a compatibility vocabulary, and adds no
  authored prompt fragment. All complete system-prompt literals remain
  byte-identical.
- The bounded owner set is the canonical cognition input/workspace lifecycle
  projection, P output-contract selection and validation, persona cognition
  carriage, direct deterministic tests, resolver-loop production-path
  coverage, one isolated live P recurrence case, and source-impact ownership.
  Resolver durable contracts, L3/dialog, DSH runtime, adapters, deployment,
  production data, README files, and unrelated documentation remain unchanged.
  Node 2 E2E is rerun only after those deterministic and isolated live gates
  pass and their artifacts are inspected.

### 2026-08-31 — P3 Node 2 Post-Admission Capability Closure Approved

- Isolated real-P artifact
  `cognition_admission_post_pending_recurrence_076f7ec0847f424c9a918ced2cd27c7b`
  passed the structural post-pending contract in one attempt, but selected a
  second `task_resolution_request` with foreground timing. The candidate was
  semantically consistent with its advertised capabilities: the current
  observation selected beta, the resolver evidence said the background task
  remained pending, and task admission was still present in the roster. In the
  full loop the duplicate guard would prevent a second side effect, then add a
  redundant cognition cycle and risk a false visible blocker. This is a
  lifecycle capability-projection defect, not a prompt-wording failure.
- Once a task-resolution observation has established accepted continued work
  during the same answered-pending resolver execution, the
  `post_pending_resolution` P contract projects both
  `task_resolution_request` and `human_clarification` out of its actual
  capability roster. P continues to receive the typed pending task observation
  and may form the current acknowledgement or other response plan; later DSH
  completion returns through the normal independent conversation path with its
  own lifecycle state. Exact deterministic validation continues to reject an
  unavailable task or clarification request, so no duplicate candidate is
  stripped or rewritten.
- The bounded change is capability projection and its direct deterministic,
  resolver-loop, isolated real-P, and impact coverage. Complete system prompts,
  resolver persistence, task/DSH execution, L3/dialog, adapters, README files,
  and unrelated documentation remain unchanged. The isolated real-P case must
  converge without a task or clarification request before Node 2 E2E resumes.

### 2026-08-31 — P3 Node 2 Failure-Evidence Observability Gate Approved

- The failed Node 2 Turn 2 response exposed a public debug `trace_id` and a
  nested operational-error trace identity, but the harness asserted success
  before retaining that public key. Its cleanup fallback also queried trace
  runs through a user field absent from the stored trace-run schema, leaving
  the protected P attempts unavailable after the guarded database dropped.
- Failure capture retains the authoritative public debug response `trace_id`
  before outcome assertions. Only if it is absent may the harness fall back to
  the exact stored turn identity: `platform`, `platform_channel_id`, and
  `platform_message_id` for the current turn. It finalizes and writes the
  trace run and ordered trace steps while Brain and the isolated database are
  still alive.
- No cognition semantic correction may be designed from this failure until the
  retained raw failed P attempts identify the unavailable resolver capability.
  The bounded owner set is the Node 2 live-harness capture path, its focused
  deterministic artifact coverage, and this execution record; production
  cognition, prompts, DB configuration, and runtime semantics remain outside
  this gate.

### 2026-08-31 — P3 Node 2 Output-Contract Capability-Coherence Amendment Approved

- Artifact `prerequisite_admission_20260830T181712Z_c482d3fd` and protected
  trace `llmtrace_9589fd437bed4a9b98d8431ed961500a` retained the exact
  contradiction after the observability gate. On each post-pending P call,
  the visible resolver roster contained only `approval_preparation` and
  `self_goal_resolution`, while the typed output contract still advertised a
  `task_resolution_request` item variant. The local model repeatedly selected
  that unavailable task request; exact validation rejected it and bounded P
  regeneration exhausted before queued work could start.
- `resolver_request_item_variants` derives only from the same already
  projected resolver roster in the P packet. It contains `non_task` when one
  visible resolver is not `task_resolution_request`, contains the task variant
  only when that exact capability is visible, and is empty when no resolver is
  visible. Resolver-request requiredness, bounds, capability semantics,
  facade validation, retries, and lifecycle behavior remain unchanged.
- This is output-contract data coherence, not prompt tuning or semantic
  rewriting. The bounded owners are `cognition_core_v3/prompt.py`, direct
  cognition-core deterministic packet tests, and impact ownership when its
  validator requires it. The gates prove each roster/variant combination,
  post-pending coherence, unchanged eight complete system-prompt literals,
  focused deterministic tests, source impact, compile, Ruff, diff hygiene,
  and repository README cleanliness before any subsequent live Node 2 run.

### 2026-08-31 — P3 Node 2 Background-Delivery Child-Trace Observability Amendment Approved

- Artifact `prerequisite_admission_20260830T183140Z_d4d1e677` retained the
  authoritative Turn 2 parent trace
  `llmtrace_69273d37cd8b4cd79450ea6fb42da89b`, successful beta-only DSH
  resolution, and five failed delivery attempts. The child P recurrences that
  exhausted delivery validation were associated with the accepted background
  job, but their raw candidates were absent from the retained parent trace and
  were lost when the isolated database dropped.
- While Brain and the isolated database remain alive, the Node 2 failure path
  retains the public Turn 2 trace as the primary trace and discovers every
  child `llm_trace_runs` row with exact
  `source_background_work_job_id == job_id`. It orders children by stored
  `created_at` and `trace_id`, finalizes each run, and writes each run plus its
  ordered raw steps under `companion_traces` in both `failure_evidence.json`
  and `trace.json`. An unavailable job records an empty companion set.
- This is test-harness observability only. The owner set is the Node 2 live
  harness, its deterministic artifact test, and this amendment. Production
  tracing, cognition contracts, retries, prompts, runtime behavior, adapters,
  database configuration, README files, and unrelated documentation remain
  unchanged. The gate requires deterministic proof that multiple failed child
  P candidates survive artifact writing while the primary trace remains
  distinct, followed by Ruff, compile, diff hygiene, and README cleanliness.

### 2026-08-31 — P3 Node 2 Result-Delivery Lifecycle Contract Amendment Approved

- Artifact `prerequisite_admission_20260830T184918Z_6240cb27` retained the
  successful parent trace `llmtrace_b7f9a4bd2abe4fc99d21e7873640e378` and
  five failed child delivery traces: `llmtrace_853d6ef6d0ea4295b0e7aaabef3aaa71`,
  `llmtrace_40f8104c444c493e8aeb6f1a5886b7d8`,
  `llmtrace_c352b81e2a954e139d56fea765ec3318`,
  `llmtrace_4accc6a1056c409cb2cc16f3ceae949e`, and
  `llmtrace_e6aea33e8a5f4080acc2bc45c0f28d50`. All fifteen protected P
  candidates carried `pending_task_continuation: null` and failed exact
  validation after beta-only DSH resolution succeeded.
- A `tool_result` episode is an audited result-delivery cognition recurrence,
  not a fresh user request. Its transient P contract variant is
  `tool_result_delivery`: task-result evidence remains available to character
  judgment and downstream dialog, while `human_clarification`,
  `task_resolution_request`, pending-resolution fields, and
  `pending_task_continuation` are closed. Other projected non-task resolver
  capabilities remain available for character-owned internal DSH handling.
- The selector derives this variant from the validated `tool_result` trigger
  and fails closed if resolver-pending lifecycle state contradicts that
  terminal delivery episode. Typed packet projection and exact facade
  validation enforce the boundary without output rewriting, semantic
  post-processing, compatibility behavior, prompt edits, or a dedicated stage.
- Owners are the P variant contract, cognition input selector, P packet
  projection, P validator, direct deterministic tests, one isolated live P
  admission case, and source-impact records. Gates cover no model-visible
  variant metadata, coherent visible resolver variants, exact invalid-field
  rejection with bounded regeneration, ordinary/open/post continuity, prompt
  literal integrity, and deterministic validation before a later live Node 2
  rerun.

### 2026-08-31 — P3 Node 2 Turn-1 Semantic Persistence Evidence Gate Approved

- Artifact `prerequisite_admission_20260830T191414Z_8ca49b81` retained Turn 1
  trace `llmtrace_3315e36844894d868c6dc0fe5881feb9` and Turn 2 trace
  `llmtrace_d941ef4c27004bdea9b8d0b8f6df03cd`. Turn 1 produced the visible
  alpha-or-beta question with zero side effects, but the prior artifact did not
  retain either the raw Turn 1 P attempt or the durable pending-ledger row.
  Turn 2 therefore began under the fresh ordinary contract and created a new,
  semantically unrelated approval clarification instead of resuming the
  original selection question.
- Before Turn 2 is submitted, the Node 2 harness records the authoritative
  public Turn 1 trace and ordered raw steps, then requires exactly one
  `resolver_pending_hil` action-attempt row under the canonical stored trigger
  and target scope. The harness validates its V3 pending-resume payload through
  the production validator and requires the open human-clarification lifecycle,
  source identity, original goal/question, and
  `pending_task_continuation.v1.background_task_admission` carrier.
- The trace and pending row are written to named Turn 1 artifacts and copied
  into failure evidence before child-process or isolated-database cleanup. An
  absent or ambiguous row ends the node before Turn 2; visible interrogative
  prose is not a substitute for durable continuation evidence. This amendment
  changes only the Node 2 test harness, its deterministic evidence tests, and
  this execution record. Production cognition, prompts, contracts, retries,
  runtime, adapters, database configuration, and README files remain unchanged.

### 2026-08-31 — P3 Node 2 Turn-1 Ledger-Collection Ownership Correction Approved

- Artifact `prerequisite_admission_20260830T193257Z_ecfd35a4` and trace
  `llmtrace_51a82375d7544f3fa074de27daef8683` prove the Turn 1 model selected
  the alpha/beta human clarification and the required
  `background_task_admission` carrier. The evidence gate falsely reported zero
  durable rows.
- `cognition_resolver.pending.upsert_pending_resume` writes through
  `action_spec.attempt_ledger.upsert_action_attempt`. Its exported
  `ACTION_ATTEMPT_LEDGER_COLLECTION` identifies the self-cognition
  action-attempt collection; the harness queried the invented `action_attempts`
  literal. Node 2 reads and failure snapshots use the exported constant while
  preserving the exact action kind, trigger, target-scope, and validator checks.
- This is a test-harness collection-ownership correction. It owns the active
  Plan 3 entry and Node 2 evidence test only; production persistence, resolver,
  prompts, contracts, runtime, adapters, schema, and README remain unchanged.

### 2026-08-31 — P3 Node 2 Typed Result Semantic-Projection Amendment Approved

- Artifact `prerequisite_admission_20260830T193948Z_9b38943f` proves the full
  character-owned control plane: Turn 1 persisted the scoped prerequisite,
  Turn 2 admitted exactly one beta-only task, DSH resolved it, the binding and
  accepted task reached terminal delivery state, and one callback was sent.
  The callback nevertheless omitted `PLAN3_E2E_BETA_SELECTED`. The validated
  `TaskResolutionResultV1.prompt_safe_summary`, DSH terminal receipt, and
  completed subgoal retained that marker, while the evidence receipt's
  `finding` recorded only that the file had been read.
- `prompt_safe_summary` is the DSH-owned, validated, bounded semantic result.
  `evidence_excerpts`, `evidence_handles`, and evidence receipts are separate
  provenance and limitation fields; they are not a complete replacement for
  the semantic result. Background delivery currently reconstructs successful
  summaries from those provenance rows and thereby changes semantic ownership
  and discards valid result content.
- Successful and partial delivery project the validated
  `prompt_safe_summary` as the authoritative semantic summary into both the
  accepted-task delivery record and the typed tool-result cognition source.
  Evidence excerpts, handles, receipt owners, source URLs, remaining needs,
  and the exact goal-continuation reference remain separately typed and retain
  their existing provenance and bounds. Non-success status behavior remains
  unchanged. Job-level free-form summaries never become result authority.
- This is a typed data-projection correction, not prompt construction or
  tuning. The bounded owners are `background_work/result_source.py`,
  `background_work/worker.py`, their direct deterministic tests, and source
  impact ownership if required. Cognition prompt literals, task-result and DSH
  wire schemas, resolver semantics, dialog, adapters, database configuration,
  README files, deployment, and production data remain unchanged. Gates must
  prove semantic-summary preservation when an evidence receipt is less
  complete, provenance separation, non-success stability, focused regression
  coverage, Ruff, compile, impact validation, and unchanged cognition prompt
  literal hashes before Node 2 is rerun.

### 2026-08-31 — P3 Node 2 Surface Output-Contract Projection Amendment Approved

- Artifact `prerequisite_admission_20260830T195903Z_e2da4751` proves the
  typed-result correction: `PLAN3_E2E_BETA_SELECTED` reached the tool-result
  cognition direct fact, A1/A2/G, P's epistemic boundary, every raw
  content-plan candidate, and every raw dialog candidate. P validly selected a
  semantic response goal that referred to the value as the specified marker.
  The literal was lost only after downstream structural exhaustion and
  deterministic degradation.
- The surface and dialog routes request generic JSON-object transport, which
  guarantees an object but does not communicate either stage's exact typed
  shape. The content-plan packet contains no output contract, so all three
  candidates returned a nested `content_plan` list plus an extra `dialog`
  field instead of the canonical top-level semantic fields. The dialog packet
  likewise contains no output contract, so all three candidates returned
  `final_dialog` as one string instead of the canonical bounded message list.
  Their exact validators correctly rejected every candidate. The final
  deterministic degradation used the accepted P response goal, which was
  semantically sound but did not carry the literal marker.
- Each producing stage projects its existing typed output contract as bounded
  structured data in the human packet and retains it unchanged in repair
  packets. The content-plan contract describes its four canonical top-level
  fields and delivery-profile dimensions. The dialog contract describes its
  sole bounded `final_dialog` message-list field. These are model-call data
  contracts, not authored prompt fragments. Complete system-prompt literals
  and their wording remain byte-identical.
- Canonical validators continue to reject wrong fields and wrong types and use
  bounded regeneration. Deterministic code does not extract nested legacy
  shapes, discard extra fields, convert strings into message lists, or recover
  rejected semantic text into the visible result. The existing degraded path
  remains a last resort rather than an alternate semantic producer.
- The bounded owners are `cognition_shared/surface_stages.py`,
  `nodes/dialog_agent.py`, their direct deterministic tests, two isolated
  real-LLM producer cases run one at a time, and source-impact ownership when
  required. Cognition prompts and contracts, task/DSH schemas, resolver
  semantics, result projection, adapters, database configuration, README
  files, deployment, and production data remain unchanged. Gates prove
  data-only contract projection on first and repair attempts, continued exact
  rejection, marker-preserving accepted content/dialog candidates, unchanged
  complete prompt literal hashes, focused deterministic coverage, Ruff,
  compile, and impact validation before the next Node 2 rerun.

### 2026-08-31 — P3 Final Degraded Result-Delivery Amendment Approved

- The user rejected the preceding model-facing output-contract approach and
  directed the final pass to make E2E delivery work regardless of generated
  surface or dialog output quality. This amendment supersedes that approach:
  no output-contract object is added to a model packet, no system or human
  prompt is changed for this fix, and no rejected model candidate is
  normalized or mined for visible semantics.
- Artifact `prerequisite_admission_20260830T195903Z_e2da4751` proves that the
  authoritative typed tool result retained `PLAN3_E2E_BETA_SELECTED`, while
  all content-plan and dialog candidates were structurally invalid. Exact
  validators correctly rejected them. The remaining failure was the terminal
  degraded surface choosing P's paraphrased response goal as its content plan
  instead of preserving the already validated result semantic authority.
- On bounded content-plan exhaustion for a `tool_result` episode, deterministic
  degradation keeps `response_plan.response_goal` as the selected character
  intent and uses the validated `resolver_result.semantic_result` as the
  deliverable content plan when that result has succeeded. The existing
  dialog terminal projection then carries that validated upstream content
  when every dialog candidate is unusable. Resolver evidence excerpts,
  handles, state, and remaining needs remain separate provenance fields.
- Ordinary episodes and non-succeeded resolver results keep their existing
  degraded behavior. The implementation does not infer semantics from job
  summaries, evidence handles, rejected LLM text, or untyped episode data.
  Semantic selection remains cognition-owned; deterministic code only
  preserves the accepted typed result through an established failure path.
- The bounded owner set is `cognition_shared/surface.py`, its direct surface
  and dialog-fallback tests, and source-impact ownership. The interrupted
  worker's unaccepted `output_contract` changes in
  `cognition_shared/surface_stages.py`, `nodes/dialog_agent.py`, and their
  tests are removed while unrelated approved Plan 3 work is preserved.
  Gates require focused deterministic coverage, Ruff, compile, impact
  validation, all five named real-LLM E2Es run and inspected individually,
  and independent code review before documentation and closure bookkeeping.

### 2026-08-31 — User-Directed Release-Candidate Closure Boundary

- The user directed this final pass to complete E2E verification, perform code
  review, and close Plan 3 now. P3-P3 therefore closes the development plan at
  the reviewed release-candidate boundary once its code, live behavior, and
  final documentation/bookkeeping gates are green.
- The separately authorized P3-P4 production deployment remains an operational
  release action. No environment was named and no process or production-data
  authority was granted in this execution. Its unexecuted procedure remains
  recorded in the archived plan for a future explicit deployment command and
  does not create deferred development scope.
- At closure, P3-G9 and the deployment portion of P3-G10 are recorded as not
  executed by owner direction, while every development, decommission, E2E,
  review, and documentation gate must close without waiver. The closing
  attestation distinguishes a completed release candidate from a deployed
  release.

### 2026-08-31 — Trigger-Source E2E Reset Technical Completion

- The approved focused sign-off plan completed its required unchanged
  all-ten diagnostic batch before remediation. That batch produced 2 passes
  and 8 failures in 468.42 seconds; its artifacts and grouped systematic
  diagnosis remain authoritative historical evidence.
- The bounded harness, fixture, source-profile, optional-carrier, shared
  causal-state, and terminal-status-coherence corrections are implemented.
  Their final gates are green: 152 focused DSH tests, 112 direct owner tests,
  485 exact impact nodes, 101 full-sidecar tests, sidecar typecheck/build,
  changed-file compilation, exact 12-node live collection, scoped Ruff, and
  diff hygiene.
- Post-rerun review found that the original session oracle was derived from
  Brain bindings. The strengthened harness now enumerates every isolated
  sidecar SQLite store directly. All ten nodes reran individually under exact
  one-binding/one-matched-session positive gates or exact
  zero-binding/zero-session negative gates.
- The authoritative trigger-source ledger is 10/10 technically passed: six
  source-bound positive DSH-entry proofs and four deliberate zero-lineage
  non-entry proofs. The final artifacts are indexed in the focused plan; the
  six positives each contain exactly one independently matched sidecar session
  and the four negatives contain none.
- The stronger scheduled-future run exposed and retained one canonical status
  conflict before remediation. The sidecar producer and Brain result boundary
  now reject `resolved` plus remaining needs; the final live result is
  coherently `partial` with the unavailable telemetry preserved as a need.
- The first independent behavior review accepted nine authoritative dossiers
  and failed `tool_result_resolved` as a material L3 false-completion: surface
  planning and dialog extrapolated future action or risk clearance beyond the
  typed handover evidence. The focused plan cites both the authoritative
  failing artifact and the earlier retained sample that demonstrates the
  repeated mode.
- The focused plan now owns a bounded evidence-authority amendment for exactly
  the content-planning and dialog prompts, six speech-bearing matrix reruns,
  two dialog-component reruns, and fresh independent review. The technical
  oracle remains green; semantic remediation and the remaining parent closure
  gates remain open.

### 2026-08-31 — Final-Code Trigger-Source Remediation Checkpoint

- The bounded L3 evidence-authority remediation is implemented in exactly the
  approved content-planning and dialog-generator prompt owners. Scoped resolver
  completion now remains distinct from broader safety, permission, downstream
  execution, and readiness claims. Both retained dialog-component live nodes
  passed individually.
- A later scheduled-case review found deterministic loss of all but the first
  typed DSH evidence excerpt during cognition recurrence. The focused plan was
  amended before implementation. The shared capability projection now merges
  receipt-linked excerpts with the validated knowledge projection in source
  order, within the existing item/character bounds and without semantic
  synthesis. The final scheduled-commitment artifact preserves both requested
  findings.
- Every path affected by those final changes reran individually. The final-code
  ledger remains 10/10 technically passed: six positive cases have one Brain
  binding and one independently enumerated matching sidecar session; four
  recursion/readiness-closed cases have zero bindings and zero sessions.
- Final mechanical gates are green: 153 focused DSH owner tests, 33 direct L3
  owner tests, six exact remediation variants, five task-result
  contract/projection tests, 486 exact impact nodes, all changed-file
  compilation, exact 12-node live collection, scoped Ruff, and diff hygiene.
  The previously recorded 101-test sidecar suite, typecheck, and build remain
  valid because the later corrections are Python prompt/projection changes.
- The final authoritative artifacts and all retained failure/rerun reasons are
  indexed in the focused plan. Fresh independent behavior and scope review is
  now the next gate. Parent closure, registry/archive bookkeeping, and the
  zero-deferred-development-work attestation remain pending that decision.

### 2026-08-31 — Independent Behavior Pass And Closure Reconciliation

- The fresh read-only independent review accepted all ten final-code behavior
  dossiers and the final implementation scope. It found no blocker, high, or
  medium issue. Its sole low observation is a harmless tense label in the
  internal-comparison summary; the quoted evidence and responsibility change
  remain exact. The reviewer judges the ten-node matrix sufficient as the
  Phase 3 E2E sign-off layer.
- The first final-code full non-live closure run ended with 3,367 passes and
  three stale test-fixture failures. Each fixture constructed a resolved task
  result without DSH evidence and was correctly rejected by the strengthened
  terminal-coherence contract before reaching its intended owner. The focused
  plan records the systematic diagnosis before correction. The three fixtures
  now use canonical typed evidence, their exact nodes pass together, and their
  files compile and pass Ruff. The authoritative full rerun passed 3,370 tests
  with four opt-in skips and 508 live tests deselected.
- The closure command that named
  `tests/unit/scripts/test_check_dsh_plan3_drain.py`,
  `tests/unit/task_resolution/test_decommission.py`, and
  `tests/test_dsh_plan3_documentation.py` failed at collection because those
  process-only suites were intentionally deleted by the approved, completed
  DSH integration scope/test-minimality quickfix. They are historical commands,
  not missing product coverage. Current closure uses the stable
  `scripts/check_dsh_legacy_drain.py` compile/`--help` gate, active manifest
  owners, direct static audits, and the full non-live suite. The drain CLI gate
  passes; the static audit finds zero legacy runtime imports, zero obsolete
  environment or stale catalog claims, exactly two approved read-only V1
  worker-payload drain predicates, one intentional rejection fixture, and the
  expected retained RAG3 imports.
- The final README audit found its product-level ownership statements current:
  RAG3 remains evidence-only, task resolution remains separate bounded work,
  cognition owns stance, and L3/dialog own visible wording. No README edit is
  required. The completed minimality quickfix remains the authoritative
  documentation restoration record.
- Final mechanical evidence is green: the focused plan's 10/10 live matrix,
  153 focused owner tests, 33 L3 owner tests, six exact remediation variants,
  five task-result contract/projection tests, 486 impact nodes, the retained
  101-test sidecar/typecheck/build record, exact 12-node live collection,
  changed-file compilation, scoped Ruff, and diff hygiene.

### Zero-Deferred-Development-Work Attestation

Every supported function in the Complete Functional Port Ledger has a current
DSH-era owner and recorded deterministic, process, live-DB, sidecar, or live-LLM
evidence. Every Justified Legacy Removal row is closed: the retired task,
complex-resolution, RAG2, and coding executors have no live import route; the
current product retains the intended RAG3, delivery, scheduler, reflection,
memory, adapter, and dialog owners. The later test-minimality amendment removes
only plan/prose/deletion policing and coding-agent-specific tests, not current
typed authority, lifecycle, recovery, evidence, safety, or real behavior
coverage.

There is zero deferred DSH integration development work, compatibility tail,
disabled integration capability, unresolved port, or release waiver. P3-P0,
P3-P1, P3-P2, and the release-candidate portion of P3-P3 are green. P3-G1
through P3-G8 are green. The development/review portion of P3-G10 is green.

P3-P4, P3-G9, and the deployment portion of P3-G10 were not executed by owner
direction: the user named no deployment environment and authorized no
production process or data operation in this execution. Under the later
User-Directed Release-Candidate Closure Boundary, that operational procedure
is preserved for a future explicit deployment command and does not constitute
deferred development work or a waived gate. The result is a completed,
reviewed release candidate, distinct from a deployed release.

### Final Independent Closure Decision

The final read-only closure review passed with no material discrepancy. It
confirmed the canonical evidence-only test-fixture corrections, the 3,370-test
full non-live rerun, 486-node impact gate, focused/live/sidecar ledgers, the
test-minimality supersession, every functional port and justified removal, and
the zero-deferred-development-work attestation. It explicitly approved marking
this plan and the focused E2E plan completed and moving both to their completed
archives.

This plan is therefore completed and archived at the reviewed release-
candidate boundary. P3-P4, P3-G9, and deployment P3-G10 remain recorded as not
executed by owner direction, not passed or waived. A future deployment requires
a separate explicit command naming the environment and production operations.
Any later development scope uses a new plan.

## Appendix A — Exact Dedicated Legacy Test Deletion Inventory

These are the complete dedicated coding-agent and complex-task test files
present at approval time. P3-P0 requires exact equality with this list or an
approved amendment before deletion.

- `tests/test_coding_agent_async_boundaries.py`
- `tests/test_coding_agent_benchmark_contracts.py`
- `tests/test_coding_agent_fetching_internet.py`
- `tests/test_coding_agent_fetching.py`
- `tests/test_coding_agent_image_reading_acceptance.py`
- `tests/test_coding_agent_interface.py`
- `tests/test_coding_agent_phase_b_execution_planning.py`
- `tests/test_coding_agent_phase_b_failure_feedback.py`
- `tests/test_coding_agent_phase_c_accepted_task_live_db.py`
- `tests/test_coding_agent_phase_c_locking.py`
- `tests/test_coding_agent_phase_c_run_context_contracts.py`
- `tests/test_coding_agent_phase_d_action_loop_contracts.py`
- `tests/test_coding_agent_phase_d_benchmark_contracts.py`
- `tests/test_coding_agent_phase_d_candidate_recovery.py`
- `tests/test_coding_agent_phase_d_coding_run_integration.py`
- `tests/test_coding_agent_phase_d_patch_operations.py`
- `tests/test_coding_agent_phase_d_repository_index.py`
- `tests/test_coding_agent_phase2_new_artifact_contracts.py`
- `tests/test_coding_agent_phase4_code_modifying_contracts.py`
- `tests/test_coding_agent_phase4_code_patching_contracts.py`
- `tests/test_coding_agent_phase4_interface.py`
- `tests/test_coding_agent_phase5_interface.py`
- `tests/test_coding_agent_phase5_patch_apply_contracts.py`
- `tests/test_coding_agent_phase6_code_executing_contracts.py`
- `tests/test_coding_agent_phase6_interface.py`
- `tests/test_coding_agent_phase8_interface.py`
- `tests/test_coding_agent_phase8_verify_repair_contracts.py`
- `tests/test_coding_agent_phase9_e2e_workflows.py`
- `tests/test_coding_agent_phase9_interface.py`
- `tests/test_coding_agent_phase9_run_supervisor_contracts.py`
- `tests/test_coding_agent_reading_acceptance.py`
- `tests/test_coding_agent_reading_pm_programmer.py`
- `tests/test_coding_agent_reading.py`
- `tests/test_coding_agent_source_intake.py`
- `tests/test_coding_agent_source_resolution.py`
- `tests/test_complex_task_resolver_algorithmic.py`
- `tests/test_complex_task_resolver_contracts.py`
- `tests/test_complex_task_resolver_evidence.py`
- `tests/test_complex_task_resolver_fixture.py`
- `tests/test_complex_task_resolver_graph.py`
- `tests/test_complex_task_resolver_live_llm.py`
- `tests/test_complex_task_resolver_media_subagent.py`
- `tests/test_complex_task_resolver_prompt_contract.py`
- `tests/test_complex_task_resolver_service.py`
