# Scheduled Future Speech: Temporal Grounding, Content Authority, and Dispatch Gate

- Status: completed
- Plan type: active bugfix
- Execution readiness: completed under the user's explicit execution command
  and the architecture amendment recorded below
- Production-code authorization: granted by the user's explicit execution
  command; plan status remains the execution boundary
- Independent review:
  [scheduled future-speech plan review](../../../test_artifacts/diagnostics/scheduled_future_speech_plan_review.md)
- Final independent review:
  [scheduled future-speech final review](../../../test_artifacts/diagnostics/scheduled_future_speech_final_review_post_bigbang_20260815.md)
- Incident evidence:
  [impacted incident RCA](../../../test_artifacts/diagnostics/asuna_qq_480386272_impacted_incident_rca.md)

## Executive decision

The fix is a two-boundary repair:

1. The action-planning owner evaluates relative-time fidelity and current-task
   content authority before a future-speech action can be persisted.
2. The scheduled self-cognition worker evaluates the rendered candidate against
   the immutable authority immediately before dispatch.

The first boundary prevents a stale absolute timestamp from becoming durable
work. The second boundary prevents a validly timed task from emitting wording
that expands the accepted objective or relies on historical context as though
it were a current commitment.

The scheduled content gate renders once, evaluates once, and either dispatches
or suppresses. It does not perform a second full dialog render. Rejected
candidate text is removed from consolidation inputs while unrelated valid
episode evidence continues through normal consolidation.

This plan deliberately leaves queue idempotency, generic dialog verifiers,
calendar terminalization, accepted-task status redesign, catch-up policy, and
the partial-appraisal contract unchanged. They are either already-owned
boundaries or separate work requiring separate evidence.

## Mandatory skills and rules

- development-plan applies to the entire plan lifecycle, implementation
  handoffs, approval, verification, and evidence recording.
- local-llm-architecture applies to the action-planner contract, the scheduled
  content evaluator, model-facing projections, route reuse, and call budgets.
  The semantic question is LLM-owned; deterministic code owns structural
  validation, provenance, time arithmetic, limits, and execution.
- no-prepost-user-input applies to accepted future commitments and their
  authority. The parent LLM chooses the accepted objective and authority
  channel. Code validates structure, evidence-handle identity, time
  arithmetic, and permissions without keyword matching or semantic rewriting.
  The scheduled evaluator is an explicit LLM semantic judge of rendered
  wording; deterministic code maps its declared closed dimensions and does
  not infer acceptance from strings or override the parent commitment.
- debug-llm applies to every live evaluator case and review artifact. A
  passing parser or pytest result is supporting evidence, not a quality
  judgment.
- py-style applies before every Python source edit.
- test-style-and-execution applies before every test edit or test run.
  Deterministic, patched-handoff, integration, and live-LLM tests retain their
  separate responsibilities.
- cjk-safety applies when implementation edits Python prompt or test strings
  containing CJK text.
- database-data-pull applies only to a read-only incident re-check or active
  record export; use the repository virtual environment and place any export
  under test_artifacts/.

## Must Do

The implementation must:

- repair relative-time fidelity in the planner owner before persistence;
- create and carry one immutable pre-persistence scheduled authority;
- enforce deterministic future-time, source-identity, and run-identity checks;
- project bounded current authority and prompt-safe audience descriptors;
- run one scheduled semantic content gate before dispatch;
- suppress unsupported, historical-only, structurally invalid, or
  contract-mismatched candidates;
- filter rejected or undelivered candidate content from consolidation while
  preserving unrelated valid evidence;
- record authority, gate, dispatch, and admission correlation; and
- block cutover when active legacy future-speak records lack the new authority.

## Deferred

The implementation must leave these items for separately approved work:

- active-record migration or historical backfill;
- accepted-task/background-job visible-delivery lifecycle redesign;
- calendar schedule terminalization or cleanup;
- catch-up, stale-trigger, or max-lateness policy;
- a second dialog render or semantic dialog rewrite loop;
- generic dialog-verifier redesign;
- queue/idempotency redesign;
- partial-appraisal prerequisites; and
- automatic promotion of historical evidence into current authority.

## Incident and impact boundary

### Reproduction identifiers

| Item | Value |
|---|---|
| QQ group | 480386272 |
| Visible platform message | 227312230 |
| Delivery tracking | 906fd45f71694e948f60a50181e01cd1 |
| Accepted task | task-94fa616b0bd541d1b5198bf60f8aad70 |
| Background job | job-94fa616b0bd541d1b5198bf60f8aad70 |
| Calendar schedule | calendar_schedule_49f5cad88af6d137fab09c108d603717 |
| Calendar run | calendar_run_18c4e3af0554e3ca38c5edcb23704153 |
| Parent trace | llmtrace_cf2e8b28a797419dae116dba09953555 |
| Child trace | llmtrace_5128c0ef0cba47039a0a5a0adbe214b1 |
| Accepted local time | 2026-08-14 13:25 |
| Intended trigger | 2026-08-14 22:00 local time |
| Persisted stale trigger | 2025-05-23 22:00 |
| Actual visible delivery | 2026-08-14 13:47:56 local time |

The visible speech was:

~~~text
十点整！时间到啦！
嘿嘿，之前的契约可是生效了哦。现在正式宣布——补偿考核环节，启动！
之前说好的加倍补偿，你可千万别想赖掉啊。
好啦，快快快，立刻进入准备状态！
首先就是那个……厕所隔间的检查，是不是已经准备好了？
终于等到这一刻了呢，接下来就由我来主导咯！
~~~

### Direct impact radius

| Boundary | Impact | In-scope repair |
|---|---|---|
| Action planning | A relative request was converted into an old absolute date and accepted. | Planner-owned semantic temporal evaluation plus bounded planner replacement. |
| Durable future work | The stale trigger propagated through accepted task, job, calendar schedule, and calendar run. | Immutable authority created before persistence and carried through each record. |
| Scheduler execution | The worker treated the stored trigger as authoritative without checking it against the source authority. | Deterministic due and authority-identity guard. |
| Dialog content | The candidate added concrete “toilet-stall inspection” detail not present in the accepted objective. | One scheduled-authority content evaluation before dispatch. |
| Group surface | The unsupported content was delivered to group 480386272. | Audience descriptor is part of the evaluator input; operational target IDs remain deterministic metadata. |
| Consolidation | Candidate text can enter post-dispatch state before the candidate’s final admission is explicit. | Rejected or undelivered candidate content is filtered; other episode evidence remains eligible. |
| Provenance | The future-cognition source scope currently drops source_message_id; trace/episode omissions are separate propagation gaps. | Carry source episode/message/action-attempt identity; keep optional trace IDs diagnostic only. |

The incident does not prove a private-context leak. It proves that a group
surface lacked a current-task authority boundary. Historical evidence may have
supported related words such as “double compensation,” but historical evidence
was not sufficient to authorize the concrete current speech detail.

The already-passed calendar run, old records, queue retry policy, and completed
delivery lifecycle are outside the impacted repair boundary. This plan adds a
read-only cutover preflight for active legacy records; it does not migrate or
rewrite historical records.

## Root cause analysis

### Causal chain

1. The parent action planner received a relative local-time request and emitted
   the stale absolute timestamp 2025-05-23 22:00.
2. The action-planning contract checked shape, parsing, and conversion but did
   not ask the action-planning owner to judge whether the absolute date matched
   the relative expression in the current local-time context.
3. Action authorization was the only semantic boundary close to persistence.
   It owns permission for an actual effect and evidence basis; it is not the
   owner of planner timestamp regeneration. No planner replacement occurred.
4. Deterministic future-speak validation accepted a syntactically valid future
   timestamp because it did not compare the trigger to the accepted time and
   source authority.
5. The stale trigger became durable work and the scheduler later executed that
   stored trigger.
6. Scheduled cognition received a broad continuation objective without a
   bounded, immutable list of current-authority details.
7. Dialog generated a candidate containing a concrete unsupported detail.
8. No scheduled-authority content evaluator ran between candidate generation and
   deliver_selected_speak().
9. The candidate was delivered and therefore became externally irreversible.

### Non-causal or unproven observations

- The available evidence does not connect a partial appraisal failure to the
  stale date or the unsupported wording. Partial-appraisal requirements are
  deferred.
- The missing source message ID is a real propagation defect, but it does not
  prove that it caused the missing trace or episode IDs. Those fields have
  separate propagation ownership.
- The target was a group, so audience mis-grounding is a demonstrated risk.
  The incident does not establish that private evidence was leaked into the
  group speech.

## Correct end-to-end behavior

The executable runtime contract is:

~~~text
accepted user request
  -> parent cognition and action planner
  -> planner-owned temporal/authority evaluation
  -> bounded planner replacement on semantic mismatch
  -> effect authorization
  -> deterministic materialization and future-time/provenance validation
  -> immutable scheduled authority
  -> accepted task/job/calendar carrier persistence
  -> due and authority-identity check
  -> scheduled source projection
  -> scheduled cognition
  -> one dialog render
  -> one scheduled-authority content verdict
  -> dispatch or suppress
  -> settlement
  -> consolidation with rejected/undelivered candidate text removed
~~~

### Step-by-step expected behavior

1. Intake normalizes the user request into typed message, local timezone, and
   accepted timestamp fields. The adapter does not interpret raw QQ syntax in
   the brain contract.
2. The action planner uses the existing
   COGNITION_LLM_ACTION_PLANNING route and existing
   ACTION_PLANNING_ATTEMPT_LIMIT. For a future_speak row it emits:
   - the exact trigger candidate;
   - a temporal-alignment semantic value;
   - a bounded authorized-content summary; and
   - bounded evidence handles with semantic detail summaries.
3. A planner-owned evaluator validates the temporal and authority dimensions
   immediately after parsing the action plan. A semantic mismatch is returned
   to the action-planning owner through the existing bounded replacement
   prompt. The replacement uses the same context and attempt budget. A
   candidate that exhausts the budget produces no future-speak action.
4. Action authorization then evaluates whether the requested effect is allowed
   by the current character judgment and evidence. It remains an effect
   permission stage; it does not regenerate timestamps or dialog.
5. Deterministic materialization creates a
   ScheduledFutureSpeechAuthorityV1 before accepted-task persistence. The
   authority contains no IDs that are created later. The materializer validates
   source identity, evidence handles, trigger parseability, and
   trigger_at_utc > accepted_at_utc.
6. The authority is copied immutably into the accepted task, background job,
   calendar schedule, and calendar run carrier. Each carrier adds its own
   local ID without changing the authority or its deterministic authority_id.
7. The self-cognition collector claims only due work according to the existing
   scheduler. Before scheduled cognition starts:
   - now >= authority.trigger_at_utc is required;
   - the run due time must equal the authority trigger;
   - the run must reference the same authority_id; and
   - a missing or contradictory authority fails closed without dialog or
     delivery.
8. Scheduled cognition receives the current objective, bounded semantic
   authorized details, prompt-safe audience descriptor, and source context.
   It receives no platform or database IDs in model-facing text.
9. The existing dialog renderer produces one candidate under the existing
   self-cognition DIALOG_RENDER_CALL_LIMIT = 1. Its generic dialog verifiers
   retain their current ownership and behavior.
10. A scheduled-authority evaluator in nodes/dialog_agent.py evaluates the
    candidate once using the existing DIALOG_GENERATOR_LLM route and the
    existing bounded verifier-attempt limit. Canonical JSON parsing and
    structural repair are allowed inside that evaluator. Semantic rejection
    suppresses the candidate; it does not trigger a second full dialog render.
11. Deterministic code converts the evaluator dimensions plus due and identity
    checks into a closed disposition:
    - accepted candidate: dispatch through the existing delivery path;
    - rejected, malformed, unavailable, or contract-mismatched candidate:
      no dispatch and a typed suppression result.
12. Settlement records the authority, gate result, and dispatch outcome.
    Consolidation continues for valid non-dialog episode evidence. Rejected
    or undelivered candidate text is absent from final_dialog,
    surface_outputs, assistant-acceptance source views, and candidate-derived
    memory/commitment inputs.

## Target state and contracts

### 1. ScheduledAuthorityProposalV1 — action-planner output

This proposal is part of the canonical future-speak action row. It is not a
durable record and is never accepted by itself.

~~~json
{
  "schema_version": "scheduled_authority_proposal.v1",
  "temporal_alignment": "aligned",
  "authorized_content_summary": "在约定时间开始补偿考核。",
  "authorized_detail_refs": [
    {
      "evidence_handle": "ev-current-123",
      "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
      "provenance_role": "current_event"
    }
  ]
}
~~~

Rules:

- temporal_alignment is closed:
  aligned, relative_date_mismatch, past_or_not_future,
  timezone_unclear, or unavailable.
- The summary is bounded plain semantic text, not final dialog and not a
  promise to perform an unrequested concrete action.
- Each detail handle must exist in the current CognitionEvidenceV2 set.
- provenance_role is checked against the actual evidence row; the model cannot
  elevate a historical handle by merely labeling it current_event.
- Only current-episode authority roles admitted by the parent cognition may
  authorize current scheduled content. Historical memory, character-world
  context, conditional guidance, and contextual facts remain explanatory
  evidence unless the parent explicitly projects them into current authority.
- Unknown fields, missing required fields, wrong types, and malformed values
  are structural contract errors and use the existing planner replacement
  path.

### 2. ScheduledFutureSpeechAuthorityV1 — immutable pre-persistence authority

~~~json
{
  "schema_version": "scheduled_future_speech_authority.v1",
  "authority_id": "sha256-canonical-pre-persistence-payload",
  "source": {
    "source_episode_id": "episode-...",
    "source_message_id": "227312230",
    "source_action_attempt_id": "attempt-...",
    "source_llm_trace_id": "llmtrace-..."
  },
  "accepted_at": {
    "utc": "2026-08-14T01:25:00Z",
    "local": "2026-08-14T13:25:00+12:00",
    "timezone": "Pacific/Auckland"
  },
  "trigger": {
    "utc": "2026-08-14T10:00:00Z",
    "local": "2026-08-14T22:00:00+12:00",
    "timezone": "Pacific/Auckland"
  },
  "target": {
    "platform": "qq",
    "channel_type": "group",
    "audience_kind": "group"
  },
  "semantic_objective": "在约定时间开始补偿考核。",
  "authorized_content": {
    "summary": "在约定时间开始补偿考核。",
    "detail_refs": [
      {
        "evidence_handle": "ev-current-123",
        "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
        "provenance_role": "current_event"
      }
    ]
  },
  "goal_continuation_ref": "goal-..."
}
~~~

Contract rules:

- The authority is built before accepted-task, job, schedule, or run IDs
  exist.
- authority_id is deterministic over a canonical payload containing source
  episode/message/action-attempt identity, accepted time, trigger, target
  class, objective, and authorized detail references. Optional diagnostic
  source_llm_trace_id is recorded but is not permission or identity truth.
- Operational target IDs, accepted-task IDs, job IDs, schedule IDs, run IDs,
  child trace IDs, and delivery IDs are carrier metadata. They are not exposed
  in model prompts and are not used to authorize wording.
- The authority is immutable after persistence. A changed trigger or objective
  creates a new authority and a new accepted work item through the normal
  deterministic path.
- trigger.utc must be strictly later than accepted_at.utc.
- A missing source episode, message, or action-attempt identity fails closed
  before persistence.

### 3. ScheduledAuthorityCarrierV1 — later persistence envelope

Each carrier stores the exact authority plus its local correlation fields:

~~~json
{
  "authority": "<ScheduledFutureSpeechAuthorityV1>",
  "accepted_task_id": "task-...",
  "background_job_id": "job-...",
  "calendar_schedule_id": "calendar_schedule_...",
  "calendar_run_id": "calendar_run_...",
  "child_llm_trace_id": "llmtrace-...",
  "delivery_tracking_id": "906fd45f71694e948f60a50181e01cd1"
}
~~~

The carrier fields are populated only when their owning subsystem creates
them. Existing accepted-task delivered naming and background-job completion
semantics remain unchanged in this bugfix.

### 4. ScheduledSpeechSemanticVerdictV1 — one model evaluator result

The scheduled evaluator returns only semantic dimensions:

~~~json
{
  "schema_version": "scheduled_speech_semantic_verdict.v1",
  "time_claim_alignment": "aligned",
  "objective_alignment": "aligned",
  "source_grounding": "current_authority",
  "audience_alignment": "aligned",
  "execution_claim": "aligned"
}
~~~

Closed values:

| Dimension | Values |
|---|---|
| time_claim_alignment | aligned, no_claim, premature, contradictory, unavailable |
| objective_alignment | aligned, scope_expansion, contradiction, unsupported, unavailable |
| source_grounding | current_authority, historical_only, unsupported, unavailable |
| audience_alignment | aligned, mismatch, unavailable |
| execution_claim | aligned, premature, false, unavailable |

The model does not return decision, attempt, free-form reasons, or open issue
codes. The canonical parser is
kazusa_ai_chatbot.utils.parse_llm_json_output(...). Structural repair uses
the existing scheduled-evaluator attempt cap. After the cap, the candidate is
suppressed.

The evaluator owns the semantic judgment about whether the rendered wording
matches the accepted authority. Deterministic code does not keyword-match the
candidate, infer user acceptance, or rewrite a semantic value. It validates
the closed dimensions and applies the declared contract mapping to dispatch or
suppression.

### 5. Deterministic scheduled gate truth table

The worker derives the disposition:

| Deterministic prerequisites | Evaluator dimensions | Disposition |
|---|---|---|
| Authority exists, exact run identity matches, now >= due_at, candidate non-empty | time_claim_alignment is aligned or no_claim; all other dimensions are aligned/current | accepted |
| Any prerequisite fails | Any value | suppressed |
| Prerequisites pass | Any unavailable or adverse semantic value | suppressed |
| Evaluator JSON remains structurally invalid after bounded repair | No semantic verdict | suppressed |

Code-owned closed gate codes are:

~~~text
scheduled_authority_missing
scheduled_authority_invalid
scheduled_trigger_identity_mismatch
scheduled_due_not_reached
scheduled_candidate_empty
scheduled_time_claim_mismatch
scheduled_objective_mismatch
scheduled_source_not_current_authority
scheduled_audience_mismatch
scheduled_execution_claim_mismatch
scheduled_evaluator_contract_error
scheduled_evaluator_unavailable
~~~

Attempt count, raw disposition, and gate codes are generated by deterministic
code. The plan does not add stale-trigger or catch-up policy. Existing
scheduler behavior may run a due item later; this plan records actual lateness
but requires the exact authority identity and now >= due_at.

## Ownership boundaries

| Concern | Owner | Explicit non-owner |
|---|---|---|
| Relative-time interpretation and authority proposal | cognition_core_v2/action_selection.py | Action authorization, scheduler |
| Effect permission | cognition_core_v2/action_authorization.py | Dialog renderer, consolidation |
| Authority construction and carrier validation | deterministic materialization/action-spec handlers | LLM |
| Source and target metadata | action-spec execution, self-cognition source collector | Model prompts |
| Final wording | dialog agent | Consolidation |
| Scheduled wording acceptance | scheduled evaluator in nodes/dialog_agent.py plus deterministic worker gate | Dialog regeneration, consolidation |
| Dispatch | existing self-cognition delivery path | Evaluator |
| Consolidation source structure | consolidation caller/schema and existing source policy | Semantic scheduled evaluator |
| Correlation and audit | self-cognition tracking/event logging | Permission decisions |

## Latency and call budget

- Parent action planning keeps the existing
  ACTION_PLANNING_ATTEMPT_LIMIT. The semantic temporal/authority check is part
  of that loop and adds no parent LLM stage.
- Scheduled cognition keeps COGNITION_CALL_LIMIT = 1 and
  DIALOG_RENDER_CALL_LIMIT = 1. Existing dialog-generator/verifier internals
  retain their current caps and are not expanded by this plan.
- The scheduled-authority evaluator is one stage invocation per rendered
  candidate and uses the existing DIALOG_VERIFIER_ATTEMPT_LIMIT only for its
  bounded structural JSON repair. The implementation records actual attempts
  and does not increase the cap.
- Semantic rejection never triggers a second full dialog render. The common
  path therefore adds one bounded evaluator stage, with no new route or
  persistent background loop.

## Change surface

### Delete

None.

### Modify

Production and contract surfaces:

- src/kazusa_ai_chatbot/cognition_core_v2/contracts.py — add the closed
  scheduled proposal and immutable-authority contract.
- src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py — add
  planner-owned temporal/authority evaluation and bounded replacement.
- src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py — preserve the
  scheduled authority proposal through the runtime V2 action bridge.
- src/kazusa_ai_chatbot/action_spec/models.py — validate the proposal field
  only on future-speak V2 requests.
- src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py —
  materialize validated authority data before persistence.
- src/kazusa_ai_chatbot/action_spec/execution.py —
  carry authority through execution.
- src/kazusa_ai_chatbot/action_spec/handlers/background_work.py —
  validate and persist authority carriers.
- src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py —
  preserve trusted source scope.
- src/kazusa_ai_chatbot/accepted_task/models.py —
  store authority carrier fields without lifecycle redesign.
- src/kazusa_ai_chatbot/accepted_task/lifecycle.py — include the authority in
  the durable accepted-task document without changing task lifecycle states.
- src/kazusa_ai_chatbot/background_work/models.py —
  store the authority in the job contract.
- src/kazusa_ai_chatbot/background_work/jobs.py —
  preserve authority during job serialization.
- src/kazusa_ai_chatbot/background_work/subagent/future_speak.py —
  preserve authority while entering scheduled cognition.
- src/kazusa_ai_chatbot/calendar_scheduler/models.py —
  carry authority into schedule/run payloads.
- src/kazusa_ai_chatbot/self_cognition/models.py —
  type scheduled authority, verdict, and admission state.
- src/kazusa_ai_chatbot/self_cognition/projection.py —
  project semantic authority without delivery IDs.
- src/kazusa_ai_chatbot/self_cognition/sources.py —
  collect and validate authority-bound cases.
- src/kazusa_ai_chatbot/self_cognition/runner.py —
  enforce due identity and build gate-aware consolidation state.
- src/kazusa_ai_chatbot/self_cognition/worker.py —
  invoke the gate, dispatch only accepted candidates, and settle outcomes.
- src/kazusa_ai_chatbot/nodes/dialog_agent.py —
  add the scheduled semantic evaluator on the existing dialog route.
- src/kazusa_ai_chatbot/consolidation/schema.py —
  carry structural admission metadata.
- src/kazusa_ai_chatbot/self_cognition/tracking.py —
  record authority and gate correlation.
- tests/ownership/source_test_impact_manifest.json — add the scheduled
  future-speech owner tests to the governed source-to-test manifest.

Existing test surfaces listed in the test disposition ledger are modified to
add exact deterministic owner tests and cross-boundary propagation checks.

### Create

- scripts/preflight_scheduled_future_speech_contract.py — read-only active
  legacy-record cutover preflight.
- tests/test_scheduled_future_speech_contract.py — deterministic contract,
  carrier, projection, and truth-table tests.
- tests/test_scheduled_future_speech_content_gate.py — deterministic evaluator
  contract and suppression tests.
- tests/test_scheduled_future_speech_content_gate_live_llm.py — individually
  executed live evaluator cases with durable debug artifacts.
- tests/test_scheduled_future_speech_preflight.py — read-only preflight tests.

### Keep

- src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py — effect
  authorization only.
- src/kazusa_ai_chatbot/consolidation/source_policy.py — structural source
  policy only.
- src/kazusa_ai_chatbot/calendar_scheduler/repository.py — existing run
  completion and cancellation behavior; no terminalization or migration.
- src/kazusa_ai_chatbot/config.py — no new LLM route or environment contract.
- src/kazusa_ai_chatbot/action_spec/evaluator.py — generic V2 evaluation
  remains unchanged; its propagation behavior is covered by a supplemental
  contract test.
- existing queue idempotency, accepted-task status transitions, generic dialog
  verifiers, and dispatcher behavior.

## Implementation work packages

### WP0 — Freeze evidence and contract baseline

Owners:

- development_plans/active/bugfix/scheduled_future_speech_temporal_grounding_and_content_gate_plan.md
- test_artifacts/diagnostics/scheduled_future_speech_plan_review.md
- test_artifacts/diagnostics/asuna_qq_480386272_impacted_incident_rca.md

Actions:

1. Preserve incident identifiers and target text as diagnostic evidence.
2. Record the independent review verdict and every resolved blocker.
3. Keep the plan in_progress after the user's explicit execution command and
   record each subsequent amendment and handoff in this document.

Acceptance:

- The plan has no unresolved owner, route, retry, migration, or evaluator
  decision.
- No production file is modified during plan preparation.

### WP1 — Planner-owned time and content-authority evaluation

Modify:

- src/kazusa_ai_chatbot/cognition_core_v2/contracts.py
- src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py
- src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py
- src/kazusa_ai_chatbot/action_spec/models.py
- src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py

Actions:

1. Extend the canonical action-planning schema with the exact
   ScheduledAuthorityProposalV1 fields for future_speak; reject those fields
   on unrelated action kinds.
2. Add a planner-owned semantic validation step immediately after canonical
   parse and before action authorization/materialization. Its model input must
   include the original relative-time expression, accepted local datetime,
   accepted timezone, deterministic normalized trigger candidate, current
   episode evidence, and the bounded objective. Deterministic code must also
   enforce trigger_at_utc > accepted_at_utc.
3. On semantic time or authority mismatch, use the existing action-planner
   replacement prompt and ACTION_PLANNING_ATTEMPT_LIMIT. Do not add a second
   parent LLM stage.
4. Require detail handles to resolve to current evidence with an admitted
   authority role. Preserve bounded semantic summaries and provenance in the
   materialized request.
5. Keep action authorization as the effect/evidence permission boundary.
6. Have the deterministic persona-supervisor materializer construct the
   immutable authority proposal input without introducing task/job IDs.
7. Extend the existing V2 request type with the optional proposal field and
   make the validator discriminated: future_speak requires a closed proposal;
   unrelated action kinds reject that field.
8. Preserve the validated proposal when the runtime V2 bridge reconstructs the
   materialized future-speak request.

Acceptance:

- The stale 2025-05-23 candidate is replaced or rejected before persistence.
- A planner contract exhaustion returns no future-speak action.
- A historical-only detail cannot be relabeled as current authority by model
  output.
- Action authorization tests continue to judge effect permission only.
- A runtime V2 future-speak request preserves its validated proposal, while a
  proposal on an unrelated action kind fails closed before materialization.

### WP2 — Deterministic authority validation and persistence carriers

Modify:

- src/kazusa_ai_chatbot/action_spec/execution.py
- src/kazusa_ai_chatbot/action_spec/handlers/background_work.py
- src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py
- src/kazusa_ai_chatbot/accepted_task/models.py
- src/kazusa_ai_chatbot/accepted_task/lifecycle.py
- src/kazusa_ai_chatbot/background_work/models.py
- src/kazusa_ai_chatbot/background_work/jobs.py
- src/kazusa_ai_chatbot/background_work/subagent/future_speak.py
- src/kazusa_ai_chatbot/calendar_scheduler/models.py

Actions:

1. Add the typed authority carrier to the canonical future-speak action
   execution payload.
2. Validate the authority before accepted-task creation:
   source identity, trigger parse, timezone, strict future relation, target
   class, evidence handles, bounded fields, and deterministic authority ID.
3. Copy the authority unchanged into task, job, schedule, and run documents.
4. Keep local carrier IDs separate from the authority. Preserve existing
   idempotency keys and retry behavior.
5. Repair future_cognition.py so trusted source_message_id is propagated.
   Preserve optional trace IDs as diagnostics; do not claim this one field
   caused other missing fields.
6. Ensure the future-speak subagent receives the carrier and does not author
   or mutate visible speech.
7. Persist an independent immutable authority copy in the accepted-task
   document before repository insertion.
8. Preserve the existing accepted-task identity and idempotency contract, but
   reject an active duplicate whose stored authority does not equal the
   incoming authority before mutation or enqueue.

Acceptance:

- Invalid authority produces no durable accepted task.
- All four durable records have byte-equivalent canonical authority payloads.
- A carrier ID change cannot change authority_id.
- Existing task/job status transitions and queue idempotency remain unchanged.
- Source episode/message/action-attempt identity is available to the scheduled
  worker.
- An active duplicate authority mismatch fails closed without task mutation or
  job enqueue; a trigger change does not implicitly redefine task identity.

### WP3 — Due-run identity and prompt-safe source projection

Modify:

- src/kazusa_ai_chatbot/self_cognition/models.py
- src/kazusa_ai_chatbot/self_cognition/projection.py
- src/kazusa_ai_chatbot/self_cognition/sources.py
- src/kazusa_ai_chatbot/self_cognition/runner.py
- src/kazusa_ai_chatbot/self_cognition/worker.py

Actions:

1. Add typed authority and gate fields to the internal SelfCognitionCase,
   scheduled source context, and runtime artifact contracts.
2. Before cognition, validate:
   - authority is present and structurally valid;
   - run due time equals authority trigger;
   - run authority ID equals the carrier authority ID; and
   - current time is at or after the due time.
3. Leave not-yet-due work pending through the existing scheduler mechanism.
   Mark identity/contract failures with a typed run outcome and skip dialog and
   dispatch. Do not add schedule terminalization.
4. Project only semantic objective, bounded authority summaries, provenance
   roles, local-time context, and audience kind into model-facing packets.
   Keep platform IDs, channel IDs, user IDs, task IDs, and delivery IDs in
   deterministic metadata.
5. Preserve existing source context and target-binding behavior for non-
   scheduled self-cognition cases.

Acceptance:

- A clock running before the trigger does not enter dialog.
- A run with a mismatched trigger or authority ID cannot render or deliver.
- The scheduled prompt contains enough authority to judge detail grounding but
  contains no delivery identity.
- Existing group/private target-binding tests remain valid.

### WP4 — One-pass scheduled content gate

Modify:

- src/kazusa_ai_chatbot/nodes/dialog_agent.py
- src/kazusa_ai_chatbot/self_cognition/models.py
- src/kazusa_ai_chatbot/self_cognition/worker.py

Actions:

1. Add evaluate_scheduled_future_speech_content(...) beside the existing
   dialog evaluators in nodes/dialog_agent.py.
2. Reuse the existing DIALOG_GENERATOR_LLM route, its route-specific
   LLMCallConfig, DIALOG_VERIFIER_ATTEMPT_LIMIT, and canonical JSON parser.
   Add no new environment route.
3. Give the evaluator the rendered candidate, semantic objective, bounded
   authority summaries, prompt-safe audience descriptor, local due context,
   and deterministic due/identity facts.
4. Require the exact closed semantic verdict dimensions defined above.
5. Invoke the evaluator once after the one dialog render and before
   deliver_selected_speak().
6. Allow structural JSON repair within the evaluator cap. Suppress after
   exhaustion or semantic rejection. Do not call the dialog generator again.
7. Derive closed gate codes and the final disposition in deterministic worker
   code.
8. Evaluate prerequisite failures in the fixed order authority missing,
   authority invalid, authority identity mismatch, due time not reached, and
   empty candidate; evaluate semantic dimensions only after those checks pass.

Acceptance:

- A candidate containing “厕所隔间的检查” without an authorized current
  detail is suppressed.
- A candidate grounded in the exact current objective is dispatchable when all
  deterministic checks pass.
- Historical-only grounding, audience mismatch, false execution claims, and
  premature time claims are suppressed.
- The self-cognition dialog call count remains one; the scheduled evaluator
  count is one per candidate.
- Malformed evaluator output fails closed after bounded structural repair.
- Gate diagnostics preserve the fixed prerequisite-code precedence and never
  allow a semantic verdict to mask an earlier deterministic failure.

### WP5 — Consolidation admission and rejection isolation

Modify:

- src/kazusa_ai_chatbot/self_cognition/runner.py
- src/kazusa_ai_chatbot/self_cognition/worker.py
- src/kazusa_ai_chatbot/consolidation/schema.py

Keep structural and unchanged:

- src/kazusa_ai_chatbot/consolidation/source_policy.py

Actions:

1. Carry the scheduled gate result and dispatch disposition into consolidation
   state.
2. For a rejected, suppressed, or undelivered candidate, remove candidate
   dialog text from final_dialog, user-visible surface_outputs, and
   assistant-acceptance source views before consolidation.
3. Admit candidate-derived memory or commitment content only when the gate is
   accepted and dispatch status is sent.
4. Continue normal consolidation for other valid action results, episode
   evidence, affect/relationship updates, and an empty final_dialog.
5. Keep semantic text judgment out of consolidation/source_policy.py; that
   module remains a structural source-view policy.

Acceptance:

- A rejected candidate cannot become durable candidate-derived memory or a
  current commitment.
- A rejected candidate does not appear as assistant-visible dialog in
  consolidation input.
- Other valid cognition evidence still reaches its normal consolidation lanes.
- A successfully sent candidate retains its normal consolidation behavior.

### WP6 — Observability and read-only cutover preflight

Modify:

- src/kazusa_ai_chatbot/self_cognition/tracking.py
- src/kazusa_ai_chatbot/self_cognition/worker.py

Create:

- scripts/preflight_scheduled_future_speech_contract.py

Actions:

1. Emit one correlation record for authority creation, planner semantic
   replacement, deterministic persistence validation, due guard, content gate,
   dispatch, and consolidation admission.
2. Record authority_id, source episode/message/action attempt, optional trace
   IDs, trigger and accepted times, evaluator attempt count, gate dimensions,
   closed gate codes, dispatch status, and consolidation admission.
3. Add a read-only preflight that scans active future-speak task/job/schedule/
   run records for the new authority schema and required source identity.
4. Exit nonzero and print bounded sample IDs when any active legacy record is
   missing the new authority. Perform no writes, migration, terminalization,
   or status changes.

Acceptance:

- The incident can be followed from parent action attempt to delivery or
  suppression with one authority ID.
- The preflight returns success only when active records are compatible with
  the new writer.
- Completed historical records are reported but not rewritten.

### WP7 — Regression, live-quality, and documentation verification

Modify or create the tests listed in the change matrix below. Add:

1. deterministic contract and truth-table coverage;
2. action-planner replacement and exhaustion coverage;
3. persistence-carrier equality coverage;
4. due/identity and no-dialog-before-due coverage;
5. one-pass content-gate and consolidation-admission coverage;
6. source-message propagation coverage;
7. read-only preflight coverage; and
8. one live LLM gate case executed individually with its output captured in a
   debug artifact.

The live case must inspect the returned candidate, semantic verdict, gate
codes, and trace before it is considered passing. Deterministic tests may run
in batches. Live LLM and live DB cases run one at a time under the repository
testing contract.

## Explicit exclusions and follow-up work

The following are excluded from this plan:

- migration or backfill of active or historical legacy records;
- accepted-task or background-job visible-delivery status redesign;
- one-time calendar schedule cleanup or schedule terminalization;
- stale-trigger catch-up or max-lateness policy;
- a second dialog generation or semantic dialog rewrite loop;
- changes to generic dialog verifiers;
- queue idempotency redesign;
- partial-appraisal prerequisite changes;
- adapter-specific QQ parsing or delivery changes;
- promotion of historical evidence into current authority without a separately
  specified parent-cognition contract.

If the cutover preflight finds active legacy records, stop deployment and use
the separately authorized exact-record cutover plan. The cutover plan is a
big-bang deletion of the named legacy rows and does not add a compatibility or
restoration path.

## Cutover and release gates

### Cutover gates

1. Plan status is approved or in_progress, and the user has explicitly
   authorized production implementation.
2. Required code and deterministic tests pass.
3. The read-only active-record preflight reports zero incompatible active
   future-speak records.
4. The new writer and new scheduled worker are deployed as one contract;
   compatibility aliases and fallback mappers are not introduced.
5. One controlled scheduled case demonstrates:
   - correct relative-date materialization;
   - authority carrier equality;
   - due identity validation;
   - one dialog render and one gate evaluation;
   - dispatch on accepted content; and
   - suppression plus safe consolidation on rejected content.

### Big-bang boundary

The new writer, scheduled worker, and exact legacy-record deletion are one
contract. The implementation contains no backward-compatible writer, fallback
mapper, alias vocabulary, or restoration path. The separate cutover plan owns
the exact deletion and its post-delete preflight; unrelated records remain
outside scope.

## Execution roles and handoff

| Role | Owned outcome | Required evidence |
|---|---|---|
| Cognition/action owner | Planner contract, semantic time/authority evaluation, bounded replacement, and runtime V2 proposal bridging | Action-plan traces and deterministic/unit tests |
| Action-spec/persistence owner | Authority construction, strict future validation, carrier propagation, and accepted-task document persistence | Task/job/schedule/run equality assertions and duplicate-mismatch evidence |
| Self-cognition owner | Due guard, source projection, one-pass gate invocation, suppression | Worker integration traces and gate disposition |
| Dialog evaluator owner | Closed semantic verdict using existing dialog route | Evaluator contract tests and one inspected live artifact |
| Consolidation owner | Candidate admission filtering with non-dialog evidence preservation | Consolidation state before/after snapshots |
| Observability/release owner | Authority correlation and preflight | Event records and preflight output |
| Reviewer | Accuracy and scope sign-off | Review artifact and final acceptance checklist |

### Role contracts

- Cognition/action owner: owns the planner schema, prompt, semantic
  temporal/authority judgment, and bounded replacement. Applicable skills are
  local-llm-architecture, no-prepost-user-input, py-style, and
  test-style-and-execution. The capability floor is existing Core V2 action
  planning and contract-test experience. The owner may change only WP1
  surfaces. Acceptance output is an action-plan trace plus deterministic
  replacement/exhaustion tests. Gate: no invalid future-speak row reaches
  authorization or persistence.
- Action-spec/persistence owner: owns deterministic authority construction,
  strict validation, task/job/schedule/run carriers, and accepted-task document
  persistence. Applicable skills are
  development-plan, py-style, and test-style-and-execution. The capability
  floor is existing action-spec and background-work contract experience.
  Acceptance output is carrier equality and active-duplicate mismatch evidence.
  Gate: invalid authority creates no accepted task and a mismatched active
  duplicate fails before task mutation or job enqueue.
- Self-cognition owner: owns due identity, source projection, gate invocation,
  dispatch admission, and settlement projection. Applicable skills are
  local-llm-architecture, no-prepost-user-input, py-style, and
  test-style-and-execution. The capability floor is existing self-cognition
  worker and target-binding experience. Acceptance output is a worker trace
  and integration evidence. Gate: no dialog or delivery before a valid due
  authority and no delivery after rejection.
- Dialog evaluator owner: owns only the scheduled semantic verdict contract
  and evaluator prompt/handler on the existing dialog route. Applicable skills
  are local-llm-architecture, debug-llm, py-style, cjk-safety, and
  test-style-and-execution. The capability floor is existing dialog-verifier
  contract experience. Acceptance output is the closed-schema deterministic
  test and individually inspected live artifacts. Gate: one evaluator
  invocation with bounded structural repair and fail-closed exhaustion.
- Consolidation owner: owns structural admission projection and preservation
  of non-dialog evidence. Applicable skills are py-style and
  test-style-and-execution. The capability floor is existing consolidation
  source-policy experience. Acceptance output is before/after state evidence.
  Gate: rejected candidate text cannot become candidate-derived memory while
  valid independent evidence still consolidates.
- Observability/release owner: owns correlation fields and read-only
  preflight. Applicable skills are development-plan, database-data-pull when
  exporting diagnostic rows, py-style, and test-style-and-execution. The
  capability floor is existing event logging and deployment preflight
  experience. Acceptance output is event evidence and preflight output. Gate:
  active incompatible records block cutover without writes.
- Independent reviewer: owns accuracy and scope sign-off only. Applicable
  skills are development-plan, local-llm-architecture, and debug-llm. The
  reviewer must be separate from the remediation owner and may not edit the
  implementation as part of sign-off. Acceptance output is a readable review
  artifact and pass/fail verdict. Gate: all blocking findings are resolved by
  the parent and re-reviewed.

Every implementation handoff must name the exact files it owns, run the
pre-change git status --short check, preserve unrelated user changes, and
report test commands and evidence paths. A plan status does not itself
authorize production edits.

## Agent autonomy boundaries

The implementation owner may choose local function decomposition, helper
names, command order, fixture construction, and test arrangement when all
contract fields, ownership, call limits, and acceptance behavior remain
unchanged.

The implementation owner must request a plan amendment before changing:

- the semantic owner of time, authority, wording, dispatch, or consolidation;
- the authority or evaluator schemas;
- the existing LLM routes or attempt limits;
- the one-render/one-evaluator call budget;
- the carrier or source-identity boundary;
- the no-migration, no-terminalization, or no-catch-up decisions;
- the consolidation admission rule; or
- the explicit source/test change surface.

The implementation owner must not add compatibility aliases, fallback mappers,
keyword filters, a second semantic regeneration loop, migration/backfill,
schedule terminalization, or unrelated cleanup. If current code cannot satisfy
the fixed contract, execution pauses for a plan amendment or user decision.

## Execution Handoff

Initial review handoff:

- Remaining scope: revise the bugfix plan for accuracy and executability.
- Review-owned surface: plan direction, ownership, contracts, change matrix,
  and acceptance gates; no production files.
- Resolved reviewer: GPT-5.6 Sol, high reasoning, normal/default service speed,
  with no high-speed or priority override.
- Applicable skills: development-plan, local-llm-architecture, debug-llm,
  no-prepost-user-input, py-style, and test-style-and-execution.
- Completed verification: source ownership and route inspection, review
  artifact creation, plan rewrite, exact-file existence audit, and matrix
  completeness audit.
- Next checkpoint: architecture amendment and fresh implementation handoff.

Architecture amendment gate:

- Resolved reviewer: independent general architect `Helmholtz` using
  `gpt-5.6-sol`, high reasoning, read-only.
- Verdict: `AMEND`.
- Required source amendment: keep
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`,
  `src/kazusa_ai_chatbot/action_spec/models.py`, and
  `src/kazusa_ai_chatbot/accepted_task/lifecycle.py`; keep
  `src/kazusa_ai_chatbot/action_spec/evaluator.py` unchanged.
- Required contract amendment: add the proposal field to the canonical V2
  request type, preserve it through the runtime bridge, persist an independent
  accepted-task copy, and verify active duplicate authority mismatches before
  mutation or enqueue.
- Required verification amendment: canonicalize authority timestamps to UTC
  `Z`, define prerequisite-code precedence, correct the partial handoff
  failure breakdown, and update the governed source-to-test manifest.
- Disposition: the preserved partial implementation may continue after this
  plan amendment; deterministic verification remains the gate before live LLM
  evaluation.

## Fixed agent and model assignments

These are plan-scoped fixed execution constraints for this bugfix. The service
speed remains normal/default; no priority or high-speed override is used.
Agents may not silently substitute a different model for convenience. A model
change requires a plan amendment or explicit user approval before handoff.

| Stage | Agent type and model | Reasoning | Owned job | Independence and gate |
|---|---|---:|---|---|
| Fix-direction and architecture review | General subagent, gpt-5.6-sol | high | Read-only review of RCA, ownership, contracts, blast radius, and fix accuracy. | Separate from remediation; produces a readable review artifact and pass/fail verdict. |
| Production implementation | deepseek_v4_flash_0731, deepseek-v4-flash | high | Implement the approved plan across the explicitly owned production and test files. | May edit only the plan-owned surface; cannot provide final code sign-off. Uses the required acknowledgement-then-execution handoff. |
| Narrow codebase exploration | explorer, gpt-5.6-luna | high | Answer one concrete read-only source, test, or contract question needed by the implementation owner. | No edits; output must identify exact paths and symbols. |
| Read-only database/evidence pull | data_fetcher, gpt-5.6-luna | high | Export bounded diagnostic rows or verify active-record preflight inputs when explicitly required. | No writes and no semantic remediation; output is raw evidence only. |
| Live-LLM quality observation | Separate deepseek_v4_flash_0731, deepseek-v4-flash | high | Inspect one live scheduled-gate case at a time and judge groundedness, unsupported detail, and trace evidence. | Separate agent instance from the implementation worker; read-only; creates a debug review artifact. |
| Final plan/code sign-off | General subagent, gpt-5.6-sol | high | Review the completed implementation, exact mapped test collection, residual risks, and cutover evidence. | Read-only, independent from implementation and live-output review; may reject but may not remediate. |

Model rationale:

- gpt-5.6-sol is reserved for frontier-quality architecture, difficult
  semantic review, and live-output judgment.
- deepseek-v4-flash is the requested production implementation and live-output
  quality model. The two roles use separate agent instances and separate
  ownership boundaries.
- gpt-5.6-luna is limited to narrow, high-volume, read-only exploration and
  data extraction.

The DeepSeek implementation handoff uses the required two-turn
acknowledgement-then-execution protocol, explicit file ownership, workspace
polling, and a 600-second hard deadline. Every message to either DeepSeek
agent uses at least a 600-second parent wait deadline. The live-output
reviewer is a separate DeepSeek Flash instance and cannot modify production
files or provide implementation sign-off.

## Change surface and test-impact matrix

The following matrix is the implementation contract. Every source file changed
by the plan appears here, and every new test node is named. If implementation
discovers an additional source boundary, pause and update this plan before
editing it.

| Owner | Source file | Planned change | Test file and exact node(s) |
|---|---|---|---|
| Action planning | src/kazusa_ai_chatbot/cognition_core_v2/contracts.py | Add canonical scheduled proposal and authority types, including native UTC `Z` output | tests/test_scheduled_future_speech_contract.py::test_scheduled_authority_proposal_contract_is_closed; tests/test_scheduled_future_speech_contract.py::test_scheduled_authority_builder_emits_native_utc_z |
| Action planning | src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py | Planner-owned temporal/authority evaluation and bounded replacement | tests/test_cognition_core_v2_action_planning_bugfix.py::test_future_speak_temporal_mismatch_uses_existing_planner_replacement_budget; tests/test_cognition_core_v2_action_planning_bugfix.py::test_future_speak_authority_exhaustion_returns_no_action |
| Action bridge | src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py | Preserve the validated scheduled proposal through runtime V2 materialization | tests/unit/nodes/test_persona_supervisor2_cognition.py::test_future_speak_v2_bridge_preserves_validated_authority_proposal |
| Action request contract | src/kazusa_ai_chatbot/action_spec/models.py | Accept a closed proposal only on future_speak V2 requests and reject it on unrelated kinds | tests/test_scheduled_future_speech_contract.py::test_future_speak_v2_request_requires_and_preserves_closed_authority_proposal |
| Action materialization | src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py | Build pre-persistence authority input and carry validated detail provenance | tests/test_scheduled_future_speech_contract.py::test_persona_materializer_carries_validated_authority |
| Action execution | src/kazusa_ai_chatbot/action_spec/execution.py | Thread authority through validated execution | tests/test_scheduled_future_speech_contract.py::test_action_execution_passes_authority_without_carrier_ids |
| Future-speak handler | src/kazusa_ai_chatbot/action_spec/handlers/background_work.py | Validate authority before task/job persistence, copy carrier, and reject active duplicate mismatch | tests/test_background_work_future_speak.py::test_future_speak_rejects_invalid_authority_before_persistence; tests/test_background_work_future_speak.py::test_future_speak_copies_immutable_authority_to_carriers; tests/test_background_work_future_speak.py::test_future_speak_active_duplicate_rejects_authority_mismatch_before_enqueue |
| Source handler | src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py | Propagate trusted source message and authority scope | tests/test_action_spec_future_cognition.py::test_future_cognition_preserves_source_message_and_authority_scope |
| Accepted-task carrier | src/kazusa_ai_chatbot/accepted_task/models.py | Store the immutable authority carrier only | tests/test_scheduled_future_speech_contract.py::test_accepted_task_carrier_keeps_authority_immutable |
| Accepted-task lifecycle | src/kazusa_ai_chatbot/accepted_task/lifecycle.py | Persist an independent authority copy in the durable accepted-task document | tests/test_scheduled_future_speech_contract.py::test_future_speak_creation_persists_independent_authority_copy |
| Background-job carrier | src/kazusa_ai_chatbot/background_work/models.py | Store authority and source identity in job contract | tests/test_background_work_jobs.py::test_future_speak_job_preserves_authority_identity |
| Background-job carrier | src/kazusa_ai_chatbot/background_work/jobs.py | Serialize/deserialize authority without mutation | tests/test_background_work_jobs.py::test_future_speak_job_round_trip_preserves_authority |
| Future-speak subagent | src/kazusa_ai_chatbot/background_work/subagent/future_speak.py | Consume authority carrier and keep wording under scheduled cognition | tests/test_background_work_future_speak.py::test_future_speak_subagent_does_not_author_dialog_text |
| Calendar carrier | src/kazusa_ai_chatbot/calendar_scheduler/models.py | Add authority to schedule/run payload contract | tests/test_scheduled_future_speech_contract.py::test_calendar_run_carries_authority_identity |
| Self-cognition models | src/kazusa_ai_chatbot/self_cognition/models.py | Add typed scheduled authority, gate verdict, and admission fields | tests/test_scheduled_future_speech_contract.py::test_self_cognition_scheduled_models_reject_open_gate_fields |
| Source projection | src/kazusa_ai_chatbot/self_cognition/projection.py | Project bounded authority and audience descriptor without IDs | tests/test_scheduled_future_speech_contract.py::test_source_packet_projects_authority_without_delivery_ids |
| Source collection | src/kazusa_ai_chatbot/self_cognition/sources.py | Build scheduled case from carrier and enforce source identity | tests/test_self_cognition_integration.py::test_scheduled_case_carries_authority_and_source_identity |
| Self-cognition runner | src/kazusa_ai_chatbot/self_cognition/runner.py | Enforce due/identity guard and gate-aware consolidation projection | tests/test_self_cognition_integration.py::test_scheduled_case_never_renders_before_due; tests/test_self_cognition_integration.py::test_rejected_scheduled_candidate_is_removed_before_consolidation |
| Self-cognition worker | src/kazusa_ai_chatbot/self_cognition/worker.py | Invoke one gate before dispatch and settle suppression | tests/test_self_cognition_integration.py::test_scheduled_worker_dispatches_only_gate_accepted_candidate; tests/test_self_cognition_integration.py::test_scheduled_worker_suppression_preserves_other_episode_evidence |
| Dialog evaluator | src/kazusa_ai_chatbot/nodes/dialog_agent.py | Add scheduled-authority semantic evaluator using existing dialog route | tests/test_scheduled_future_speech_content_gate.py::test_gate_verdict_schema_has_only_closed_semantic_dimensions; tests/test_scheduled_future_speech_content_gate.py::test_gate_structural_failure_suppresses_after_bounded_repair |
| Consolidation schema | src/kazusa_ai_chatbot/consolidation/schema.py | Carry gate/dispatch admission metadata structurally | tests/test_consolidation_lifecycle_diagnostics.py::test_rejected_scheduled_candidate_is_not_candidate_memory_input |
| Self-cognition tracking | src/kazusa_ai_chatbot/self_cognition/tracking.py | Emit authority, gate, dispatch, and admission correlation | tests/test_self_cognition_tracking.py::test_scheduled_gate_trace_contains_authority_and_disposition |
| Cutover preflight | scripts/preflight_scheduled_future_speech_contract.py | Read-only active legacy-record check | tests/test_scheduled_future_speech_preflight.py::test_preflight_blocks_active_legacy_future_speak_records |

Test-file disposition ledger:

Create these new test files:

- tests/test_scheduled_future_speech_contract.py
- tests/test_scheduled_future_speech_content_gate.py
- tests/test_scheduled_future_speech_content_gate_live_llm.py
- tests/test_scheduled_future_speech_preflight.py

Modify these existing test files:

- tests/test_cognition_core_v2_action_planning_bugfix.py
- tests/test_action_spec_future_cognition.py
- tests/test_background_work_future_speak.py
- tests/test_background_work_jobs.py
- tests/test_self_cognition_integration.py
- tests/test_self_cognition_tracking.py
- tests/test_consolidation_lifecycle_diagnostics.py
- tests/unit/nodes/test_persona_supervisor2_cognition.py
- tests/ownership/source_test_impact_manifest.json

The live test file must define these exact nodes and execute them one at a
time:

- tests/test_scheduled_future_speech_content_gate_live_llm.py::test_current_authority_detail_is_accepted
- tests/test_scheduled_future_speech_content_gate_live_llm.py::test_incident_unsupported_detail_is_suppressed
- tests/test_scheduled_future_speech_content_gate_live_llm.py::test_historical_only_grounding_is_suppressed

Files explicitly inspected but unchanged by this plan:

- src/kazusa_ai_chatbot/action_spec/evaluator.py remains generic and unchanged;
  its proposal propagation is covered by the supplemental contract test
  `tests/test_scheduled_future_speech_contract.py::test_v2_evaluator_returns_future_speak_authority_proposal_unchanged`.
- src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py remains the
  effect permission owner.
- src/kazusa_ai_chatbot/consolidation/source_policy.py remains structural.
- src/kazusa_ai_chatbot/calendar_scheduler/repository.py receives no
  terminalization or migration behavior.
- src/kazusa_ai_chatbot/config.py receives no new LLM route.

## Test Impact And Traceability

Every row below names one repository-relative governed path, its changed
contract, semantic owner, deterministic node, supplemental evidence, test
mode, and observable regression prevented. The deterministic node is required
even when an integration or live node is also listed.

| Repository-relative path | Changed symbol, field, interface, or contract | Semantic owner | Exact deterministic pytest node IDs | Supplemental integration or live node IDs | Test mode | Observable regression prevented |
|---|---|---|---|---|---|---|
| src/kazusa_ai_chatbot/cognition_core_v2/contracts.py | ScheduledAuthorityProposalV1 and canonical UTC `Z` authority timestamps | action planning | tests/test_scheduled_future_speech_contract.py::test_scheduled_authority_proposal_contract_is_closed; tests/test_scheduled_future_speech_contract.py::test_scheduled_authority_builder_emits_native_utc_z | none | deterministic unit | Open or malformed planner authority fields and non-canonical timestamps cannot pass the action contract. |
| src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py | planner-owned temporal and authority evaluation in plan_actions | action planning | tests/test_cognition_core_v2_action_planning_bugfix.py::test_future_speak_temporal_mismatch_uses_existing_planner_replacement_budget | tests/test_cognition_core_v2_action_planning_bugfix.py::test_future_speak_authority_exhaustion_returns_no_action | patched LLM handoff | The stale 2025 trigger cannot reach authorization or persistence. |
| src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py | runtime V2 proposal-to-materializer bridge | action materialization | tests/unit/nodes/test_persona_supervisor2_cognition.py::test_future_speak_v2_bridge_preserves_validated_authority_proposal | none | deterministic unit | The runtime bridge cannot silently drop a validated scheduled authority proposal. |
| src/kazusa_ai_chatbot/action_spec/models.py | discriminated SemanticActionRequestV2 proposal field | action planning | tests/test_scheduled_future_speech_contract.py::test_future_speak_v2_request_requires_and_preserves_closed_authority_proposal | none | deterministic unit | Future-speak authority cannot be rejected by the old exact-key shape, and unrelated actions cannot carry it. |
| src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py | future_speak authority materialization bridge | action materialization | tests/test_scheduled_future_speech_contract.py::test_persona_materializer_carries_validated_authority | none | deterministic unit | Validated current-task details are not lost before action-spec creation. |
| src/kazusa_ai_chatbot/action_spec/execution.py | future_speak authority execution payload | action execution | tests/test_scheduled_future_speech_contract.py::test_action_execution_passes_authority_without_carrier_ids | none | patched handoff | Execution cannot silently replace authority with later carrier IDs. |
| src/kazusa_ai_chatbot/action_spec/handlers/background_work.py | validate_future_speak_action and enqueue carrier | persistence | tests/test_background_work_future_speak.py::test_future_speak_rejects_invalid_authority_before_persistence; tests/test_background_work_future_speak.py::test_future_speak_active_duplicate_rejects_authority_mismatch_before_enqueue | tests/test_background_work_future_speak.py::test_future_speak_copies_immutable_authority_to_carriers | deterministic integration | Invalid, mutated, or duplicate-mismatched authority cannot create or mutate durable work. |
| src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py | trusted source scope projection | provenance | tests/test_action_spec_future_cognition.py::test_future_cognition_preserves_source_message_and_authority_scope | none | deterministic unit | Scheduled cognition no longer loses the source message boundary. |
| src/kazusa_ai_chatbot/accepted_task/models.py | accepted-task authority carrier | persistence | tests/test_scheduled_future_speech_contract.py::test_accepted_task_carrier_keeps_authority_immutable | none | deterministic unit | Task-local identifiers cannot authorize different scheduled wording. |
| src/kazusa_ai_chatbot/accepted_task/lifecycle.py | durable accepted-task authority document | persistence | tests/test_scheduled_future_speech_contract.py::test_future_speak_creation_persists_independent_authority_copy | none | deterministic unit | The durable accepted-task record cannot omit or share mutable authority state. |
| src/kazusa_ai_chatbot/background_work/models.py | background job authority field | persistence | tests/test_background_work_jobs.py::test_future_speak_job_preserves_authority_identity | none | deterministic unit | Job loading cannot drop the scheduled authority. |
| src/kazusa_ai_chatbot/background_work/jobs.py | job serialization round trip | persistence | tests/test_background_work_jobs.py::test_future_speak_job_round_trip_preserves_authority | none | deterministic unit | Serialization cannot alter authority or source identity. |
| src/kazusa_ai_chatbot/background_work/subagent/future_speak.py | scheduled subagent input boundary | future-speak subagent | tests/test_background_work_future_speak.py::test_future_speak_subagent_does_not_author_dialog_text | none | deterministic unit | The subagent cannot bypass scheduled cognition to author visible speech. |
| src/kazusa_ai_chatbot/calendar_scheduler/models.py | calendar schedule/run authority carrier | calendar carrier | tests/test_scheduled_future_speech_contract.py::test_calendar_run_carries_authority_identity | none | deterministic unit | A calendar run cannot execute a different authority than the accepted task. |
| src/kazusa_ai_chatbot/self_cognition/models.py | scheduled source and gate typed fields | self-cognition | tests/test_scheduled_future_speech_contract.py::test_self_cognition_scheduled_models_reject_open_gate_fields | none | deterministic unit | Model output cannot author decision, attempt, or open issue fields. |
| src/kazusa_ai_chatbot/self_cognition/projection.py | prompt-safe scheduled source packet | self-cognition | tests/test_scheduled_future_speech_contract.py::test_source_packet_projects_authority_without_delivery_ids | none | deterministic unit | Operational target identifiers cannot enter the evaluator prompt. |
| src/kazusa_ai_chatbot/self_cognition/sources.py | scheduled case authority/source binding | self-cognition | tests/test_scheduled_future_speech_contract.py::test_source_collector_rejects_missing_authority | tests/test_self_cognition_integration.py::test_scheduled_case_carries_authority_and_source_identity | patched integration | Missing or mismatched carrier provenance cannot enter scheduled cognition. |
| src/kazusa_ai_chatbot/self_cognition/runner.py | due and run-authority identity guard, with generic future-cognition discrimination | self-cognition | tests/test_scheduled_future_speech_contract.py::test_due_guard_rejects_early_run | tests/test_self_cognition_integration.py::test_scheduled_case_never_renders_before_due; tests/test_self_cognition_integration.py::test_default_runner_allows_generic_future_cognition_without_authority | deterministic plus integration | Early or mismatched scheduled runs cannot render dialog, while valid authority-free generic cognition remains runnable. |
| src/kazusa_ai_chatbot/self_cognition/worker.py | one-pass gate and dispatch admission | self-cognition | tests/test_scheduled_future_speech_contract.py::test_scheduled_gate_truth_table_is_deterministic | tests/test_self_cognition_integration.py::test_scheduled_worker_dispatches_only_gate_accepted_candidate | deterministic plus integration | Unsupported content cannot reach delivery and accepted content is not double-rendered. |
| src/kazusa_ai_chatbot/nodes/dialog_agent.py | ScheduledSpeechSemanticVerdictV1 evaluator | dialog evaluation | tests/test_scheduled_future_speech_content_gate.py::test_gate_verdict_schema_has_only_closed_semantic_dimensions | tests/test_scheduled_future_speech_content_gate_live_llm.py::test_current_authority_detail_is_accepted; tests/test_scheduled_future_speech_content_gate_live_llm.py::test_incident_unsupported_detail_is_suppressed; tests/test_scheduled_future_speech_content_gate_live_llm.py::test_historical_only_grounding_is_suppressed | deterministic contract plus live LLM | Unsupported current-task detail and historical-only grounding are visibly suppressed. |
| src/kazusa_ai_chatbot/consolidation/schema.py | scheduled candidate admission metadata | consolidation | tests/test_scheduled_future_speech_contract.py::test_consolidation_admission_filters_rejected_candidate | tests/test_consolidation_lifecycle_diagnostics.py::test_rejected_scheduled_candidate_is_not_candidate_memory_input | deterministic plus integration | Rejected candidate text cannot become candidate-derived memory while other evidence survives. |
| src/kazusa_ai_chatbot/self_cognition/tracking.py | authority/gate/dispatch correlation record | observability | tests/test_scheduled_future_speech_contract.py::test_tracking_projection_is_deterministic | tests/test_self_cognition_tracking.py::test_scheduled_gate_trace_contains_authority_and_disposition | deterministic plus integration | RCA cannot lose the causal link between authority, gate, and delivery. |
| scripts/preflight_scheduled_future_speech_contract.py | read-only active legacy-record preflight | release | tests/test_scheduled_future_speech_preflight.py::test_preflight_blocks_active_legacy_future_speak_records | none | deterministic unit | Cutover cannot silently mix legacy active records with the new writer. |

## Progress, evidence, and handoff

Current plan state:

- [x] Incident scope, causal chain, and impact radius recorded.
- [x] Correct end-to-end workflow and ownership boundaries defined.
- [x] Initial GPT-5.6 Sol review completed at normal/default speed.
- [x] Blocking review findings incorporated.
- [x] Exact change surface and test traceability recorded.
- [x] User approval and plan promotion.
- [x] Production implementation.
- [x] Deterministic verification and exact-node collection.
- [x] Individually inspected live evaluator cases.
- [x] Read-only active-record preflight now passes after the separately
  authorized legacy-record cutover.
- [x] Independent code review completed; final post-deletion sign-off is
  recorded in the final review artifact.

Implementation is authorized and the complete handoff evidence is recorded
below. The evidence records:

- baseline git status and explicitly owned file set;
- handoff role, resolved executor/configuration, and selection rationale;
- exact commands and results;
- exact collected and executed node IDs for every changed source path;
- live debug artifact paths and human quality judgment;
- preflight output;
- deviations or residual risks; and
- independent code-review verdict.

## Execution Evidence

### Partial implementation handoff — 2026-08-14

- Role: production implementation owner.
- Resolved executor: `deepseek_v4_flash_0731` / `deepseek-v4-flash`, high
  reasoning, fixed by this plan; acknowledgement-then-execution handoff was
  completed in two bounded turns.
- Baseline: `M development_plans/README.md` and untracked target plan file;
  no implementation source or test files were changed before the handoff.
- Handoff deadline: the worker was allowed the user-requested thirty-minute
  minimum and then stopped at the concrete deadline; partial changes were
  preserved for parent review.
- Changed owned surfaces: all nineteen listed production source files, the
  preflight script, the four new scheduled-future-speech test files, and the
  seven existing test files listed in the change matrix. The plan and registry
  changes remain parent-owned lifecycle edits.
- Verification reported by the worker: `py_compile` for all 31 edited/new
  Python files passed; project import smoke test passed; the new contract test
  file reported 5 passed and 7 failed. Five failures share the UTC canonical
  representation defect: the authority builder emits `+00:00`, while the
  authority validator requires `Z`. One action-execution fixture also lacks the
  required `accepted_task_state` field, and the truth-table test exposes
  unresolved prerequisite-code precedence. Exact mapped-node collection and
  the remaining deterministic batches were not completed.
- Scope amendment: the independent architecture gate found that
  `nodes/persona_supervisor2_cognition.py`, `action_spec/models.py`, and
  `accepted_task/lifecycle.py` are required boundaries. It found
  `action_spec/evaluator.py` generic and unchanged. The user-authorized
  amendment is recorded above; no production edit has yet been made to the
  amended boundaries.
- Current disposition: implementation remains incomplete; no live-LLM
  quality evidence, operational preflight result, independent code review, or
  final sign-off has been recorded.

### Amended implementation and verification handoff — 2026-08-14

- Role: production implementation owner.
- Resolved executor: fresh `deepseek_v4_flash_0731` /
  `deepseek-v4-flash`, high reasoning, with the required acknowledgement-then-
  execution two-turn handoff. The worker completed within the monitored
  window; no hard stop was issued.
- Architecture gate applied: independent `gpt-5.6-sol` read-only review,
  verdict `AMEND`. The worker changed only the amended owned surface and
  preserved `action_spec/evaluator.py` unchanged.
- Implemented boundaries: native UTC `Z` authority canonicalization and
  hashing; discriminated V2 proposal validation; runtime proposal bridging;
  durable accepted-task authority copying; active duplicate mismatch rejection;
  fixed gate prerequisite precedence; governed manifest coverage; and the
  required deterministic fixture repairs within the owned test files.
- Parent verification: `py_compile` passed for 35 changed/new Python files;
  `git diff --check` passed. The scheduled contract, content-gate, and
  preflight test files passed `22` tests. The amended deterministic batch
  (including the persona bridge unit file) passed `190` tests.
- Governed impact verification: `venv\\Scripts\\python
  scripts\\validate_test_impact.py --base-ref HEAD --run` validated the
  manifest, collected `74` exact nodes, and completed with `68` passed and
  `6` failed. The failures are in unchanged shared fixtures outside this
  plan's owned test surface:
  `tests/test_dialog_agent.py::test_dialog_exhaustion_all_unavailable_selects_latest_valid_candidate`,
  `tests/test_dialog_agent.py::test_dialog_exhaustion_selects_highest_score_not_latest`,
  `tests/test_dialog_agent.py::test_dialog_exhaustion_ties_select_latest_attempt`,
  `tests/test_self_cognition_group_visible_reply_boundary.py::test_group_action_planning_requires_explicit_silence_or_reply_decision`,
  `tests/unit/nodes/test_dialog_agent.py::test_dialog_role_direction_rejects_selected_actor_target_reversal`,
  and `tests/unit/nodes/test_dialog_agent.py::test_terminal_candidate_opposite_polarity_is_withheld`.
  They remain unmodified pending a separate scope decision.
- Read-only database preflight command: `venv\\Scripts\\python
  scripts\\preflight_scheduled_future_speech_contract.py`. Result:
  `deployment_blocked=true`, `incompatible_active_count=5` (one active
  accepted task and four active schedules), and `historical_legacy_count=10`.
  No database writes, migration, terminalization, or backfill were performed.
- Current disposition: implementation is present, but the deterministic
  impact gate and active-record preflight are blocked. Live LLM cases,
  independent final code review, and sign-off remain pending.

### Independent code review and second architecture gate — 2026-08-14

- Independent code reviewer verdict: `REJECT`. The reviewer found that the
  current implementation is not connected and fail-closed end to end, and
  blocked live LLM review and final sign-off.
- Second architecture gate verdict: `AMEND`. This gate was limited to the
  newly discovered architectural boundary and made no workspace changes.
- Approved production amendment: the user approved this exact scope on
  2026-08-14, authorizing the next implementation handoff:
  `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`,
  `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`,
  `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py`,
  `src/kazusa_ai_chatbot/db/accepted_tasks.py`,
  `src/kazusa_ai_chatbot/self_cognition/sources.py`,
  `src/kazusa_ai_chatbot/self_cognition/worker.py`, and
  `scripts/preflight_scheduled_future_speech_contract.py`.
- Required contract repairs: connect planner output to the runtime proposal
  bridge; bind authority identity to the accepted event timestamp and local/
  UTC/timezone consistency; compare duplicate authorities atomically before
  related-source mutation; scrub accepted-but-undelivered candidates before
  consolidation; bind scheduled source and target fields; and read calendar
  authority from its production payload path.
- Required test-only baseline repair scope:
  `tests/test_dialog_agent.py`,
  `tests/unit/nodes/dialog_fixtures.py`, and
  `tests/test_self_cognition_group_visible_reply_boundary.py`. These repairs
  address the six red impact nodes and authorize no production behavior.
- Required exact nodes and manifest rows are listed in the architecture gate
  review: planner-to-runtime preservation and accepted-event context;
  timestamp consistency and identity binding; authority accepted-time and
  source-scope checks; repository atomic duplicate rejection; source-carrier
  mismatch rejection; accepted-gate delivery-failure scrubbing; undelivered
  consolidation isolation; payload-aware preflight; and the new
  `db/accepted_tasks.py` repository test.
- Legacy cutover decision: rerun the corrected preflight first. If any active
  incompatible records remain, deployment stays blocked. The separately
  authorized cutover plan deletes only the exact five legacy records with
  compare-and-delete validation; generic future cognition and terminal run
  history remain outside scope.
- Current disposition: implementation and deterministic/live verification are
  complete; final sign-off is pending the post-deletion preflight and
  independent review.

### Rejected-blocker correction — 2026-08-15

- Implemented the model-facing provenance correction in
  `cognition_core_v2/action_selection.py`. Ordinary evidence retains the
  relational vocabulary; when `future_speak` is available, evidence rows
  expose their validated `current_event`/`public_scene` authority vocabulary,
  and the prompt binds proposal detail roles to the displayed role.
- Isolated generic `trigger_future_cognition` from scheduled future speech
  using the existing structural `future_speak_background_work` source
  reference. The source collector validates and carries scheduled authority
  only for that carrier; generic authority-free cases remain projectable and
  runnable. The worker bypasses scheduled due/content-gate bookkeeping for
  generic cases, while malformed scheduled authority remains fail-closed.
- Narrowed the read-only preflight calendar schedule/run queries to the same
  structural future-speak source reference. Generic active future-cognition
  rows are therefore outside the scheduled-speech inventory.
- Verification: `py_compile` passed for the changed production and test
  modules; `git diff --check` passed. The targeted deterministic batches
  passed 164 tests and 56 additional content-gate/tracking/consolidation
  tests.
- Read-only database preflight remains blocked as expected:
  `deployment_blocked=true`, `incompatible_active_count=5` (one active
  accepted task and four active calendar schedules),
  `historical_legacy_count=10`; active calendar runs report zero incompatible
  rows after the query narrowing. No database writes were performed.
- Live LLM verification and independent re-review remain pending under the
  plan gate. No migration, backfill, terminalization, or cleanup was run.

### Final verification and independent review — 2026-08-15

- Runner remediation: `self_cognition/runner.py` now applies the scheduled
  due/authority guard only when the case carries the structural scheduled
  future-speech authority. The default runner regression
  `tests/test_self_cognition_integration.py::test_default_runner_allows_generic_future_cognition_without_authority`
  reaches cognition and emits no scheduled-gate artifact. The node is a
  required entry in the source-impact manifest.
- Static checks: `venv\\Scripts\\python -m py_compile` passed for the changed
  production/test modules; `git diff --check` passed.
- Governed deterministic impact command:
  `venv\\Scripts\\python scripts\\validate_test_impact.py --base-ref HEAD --run`.
  Result: `86 tests collected`, `86 passed`, and
  `Validated 86 exact impact-test node(s).` The output included the new
  default-runner node and every exact node mapped in the source-impact
  manifest.
- Live evaluator commands were run one case at a time with `-m live_llm`.
  All three passed on the first evaluator attempt and were manually judged:
  current-authority detail was accepted; unsupported incident detail was
  suppressed with `scheduled_objective_mismatch`; and historical-only
  grounding was suppressed with `scheduled_objective_mismatch`.
  The retained human review is
  [`scheduled_future_speech_content_gate_live_review_20260815.md`](../../../test_artifacts/llm_reviews/scheduled_future_speech_content_gate_live_review_20260815.md).
  Timestamped traces retain prompt messages, raw model output, parsed verdict,
  redacted route/model configuration, and non-empty evaluator trace IDs.
- Read-only operational preflight command before the authorized deletion:
  `venv\\Scripts\\python scripts\\preflight_scheduled_future_speech_contract.py --sample-limit 5`.
  Result at the time of the initial independent review: exit code 1,
  `deployment_blocked=true`, `incompatible_active_count=5`, and
  `historical_legacy_count=10`. The incompatible active sample contains one
  accepted task (`task-2c1831a6217342d7a5a24743d8eae669`) and four calendar
  schedules (`calendar_schedule_4944be41cc53d33b443640d10e2e7226`,
  `calendar_schedule_49f5cad88af6d137fab09c108d603717`,
  `calendar_schedule_59770930065b92900ffc676af106b457`, and
  `calendar_schedule_b812cbbd86f99e01505e319213ae0e5c`). No database writes,
  migration, backfill, cancellation, terminalization, or cleanup were run at
  that point.
- Independent final review before the deletion: a fresh `kazusa_plan_reviewer` inspected the
  current workspace and evidence. It confirmed that the previously reported
  implementation blockers are resolved, but returned `REJECT` and withheld
  sign-off because the active-record preflight gate remains blocked. The
  reviewer’s required remediation was the separately authorized exact-record
  cutover. The user then explicitly amended the operation to big-bang deletion
  with no rollback or compatibility implementation.
- Current closure state: implementation, exact deletion, and post-deletion
  verification are complete. Final plan sign-off remains open until an
  independent reviewer confirms the post-deletion state.

### Authorized legacy-record cutover — 2026-08-15

- A separate cutover plan was created and registered:
  `scheduled_future_speech_legacy_record_cutover_plan.md`.
- The initial bounded dry-run identified exactly one orphaned `future_speak`
  accepted task, four marker-bearing `future_cognition` schedules, zero linked
  background jobs, and four terminal calendar runs. The initial state-preserving
  retirement left those exact rows identifiable by the cutover reason.
- Under the user’s explicit big-bang amendment, the confirmed apply deleted the
  accepted task and all four schedule IDs with compare-and-delete filters.
- The four terminal calendar runs were preserved unchanged. Generic future
  cognition rows and unrelated collections were not touched.
- Post-deletion read-only preflight exited 0 with
  `deployment_blocked=false`, `incompatible_active_count=0`, and zero
  incompatible rows across accepted tasks, background jobs, calendar runs,
  and calendar schedules.

### Post-big-bang independent sign-off — 2026-08-15

- Independent reviewer verdict: `PASS` with no blocking findings.
- The reviewer confirmed the exact accepted task and four schedules are
  absent, the four terminal calendar runs are unchanged, and generic future
  cognition remains outside the cutover.
- The reviewer re-ran the governed exact-impact collection: 86 nodes passed.
- The reviewer re-ran the cutover and preflight tests: 10 tests passed.
- The reviewer re-ran read-only preflight with
  `deployment_blocked=false` and `incompatible_active_count=0`.
- Final artifact:
  [`scheduled_future_speech_final_review_post_bigbang_20260815.md`](../../../test_artifacts/diagnostics/scheduled_future_speech_final_review_post_bigbang_20260815.md).

## Independent reviews

The initial plan review was performed by GPT-5.6 Sol with high reasoning effort
at normal/default service speed. The reviewer was separate from the parent
remediation work and returned REJECT with nine blocking findings. The readable
record is the linked scheduled future-speech plan review artifact.

The parent has incorporated the findings into this draft. A separate reviewer
must perform code review after implementation; the implementation owner cannot
provide final sign-off for its own remediation. Code review must inspect scope,
ownership, authority immutability, model-facing projection, gate behavior,
consolidation admission, exact mapped test collection, and preflight evidence.

## Verification and acceptance

### Static and contract checks

Run from the repository root using the project virtual environment:

~~~powershell
venv/Scripts/python -m pytest tests/test_scheduled_future_speech_contract.py -q
venv/Scripts/python -m pytest tests/test_scheduled_future_speech_content_gate.py -q
venv/Scripts/python -m pytest tests/test_scheduled_future_speech_preflight.py -q
~~~

Then run the impacted deterministic batch:

~~~powershell
venv/Scripts/python -m pytest tests/test_cognition_core_v2_action_planning_bugfix.py tests/test_action_spec_future_cognition.py tests/test_background_work_future_speak.py tests/test_background_work_jobs.py tests/test_self_cognition_integration.py tests/test_self_cognition_tracking.py tests/test_consolidation_lifecycle_diagnostics.py -q

venv/Scripts/python scripts/validate_test_impact.py --base-ref HEAD --run
~~~

Before any Python edit or test edit, apply the repository py-style and
test-style-and-execution skills. Every raw LLM response is parsed through
parse_llm_json_output; no stage-local JSON repairer is introduced.

### Live LLM verification

Run only one live case at a time:

~~~powershell
venv/Scripts/python -m pytest tests/test_scheduled_future_speech_content_gate_live_llm.py -k current_authority_detail_is_accepted -q -s
~~~

Inspect and retain:

- rendered candidate;
- authority summary and detail references;
- evaluator semantic dimensions;
- deterministic gate codes and disposition;
- call/attempt counts;
- protected trace IDs; and
- consolidation admission result.

Required live cases are:

1. current-authority objective with no unsupported detail: accepted;
2. the incident-style unsupported “toilet-stall inspection” detail:
   suppressed; and
3. historical-only grounding: suppressed.

The live cases are executed individually, not as a batch, and are not a
substitute for deterministic contract tests.

### Operational preflight

Run the read-only preflight in the configured deployment environment:

~~~powershell
venv/Scripts/python scripts/preflight_scheduled_future_speech_contract.py
~~~

Expected outcomes:

- exit 0: zero active legacy future-speak records;
- exit nonzero: bounded list of incompatible active IDs and deployment
  blocked;
- no database writes under either result.

## Acceptance checklist

The fix is complete only when every item passes:

- [x] Relative “tonight at ten” requests use the correct accepted-date local
  timezone and cannot persist a stale prior-year timestamp.
- [x] Temporal mismatch is evaluated by the action-planning owner and repaired
  within the existing planner attempt cap.
- [x] Action authorization remains effect/evidence permission, not wording or
  timestamp regeneration.
- [x] An immutable pre-persistence authority exists before task/job/schedule
  IDs and survives all carrier copies unchanged.
- [x] Source episode, source message, and source action-attempt identity are
  preserved; optional trace IDs remain diagnostic.
- [x] Due execution requires now >= due_at, exact trigger equality, and exact
  authority identity.
- [x] Model-facing scheduled context contains semantic authority and audience
  descriptors without operational delivery IDs.
- [x] Dialog renders once; the scheduled evaluator runs once; no second full
  dialog render occurs after semantic rejection.
- [x] Evaluator output contains only the closed semantic dimensions.
- [x] The incident-style unsupported concrete detail is suppressed before
  delivery.
- [x] Historical-only grounding, audience mismatch, premature/contradictory
  time claims, and false execution claims suppress.
- [x] Rejected or undelivered candidate text is absent from consolidation
  visible/assistant-acceptance inputs and candidate-derived memory/commitment.
- [x] Other valid episode evidence still consolidates normally.
- [x] Authority, gate, dispatch, and admission dispositions are traceable.
- [x] Active legacy preflight passes without writes after exact legacy-record
  deletion.
- [x] Existing task lifecycle, queue idempotency, generic dialog verifiers,
  and calendar run completion behavior remain intact.
- [x] No partial-appraisal, migration, schedule-terminalization, or
  catch-up behavior was added.

## Execution handoff checklist

Before implementation begins:

- [x] User explicitly authorizes the production-code change.
- [x] This plan is promoted from draft to approved or in_progress.
- [x] git status --short is captured and unrelated changes are preserved.
- [x] README.md, docs/HOWTO.md, relevant subsystem READMEs, and every
  directly involved source/test file are re-read by the implementation owner.
- [x] The implementation owner applies py-style and
  test-style-and-execution.
- [x] No unresolved route, owner, schema, retry, migration, or compatibility
  choice remains within this plan's scope; the authorized big-bang cutover is
  recorded in its dedicated plan.

After implementation:

- [x] Record exact changed files and test commands in the execution evidence
  section of this plan.
- [x] Link deterministic and live debug artifacts.
- [x] Record preflight output and independent reviewer sign-off.
- [x] Archive this plan after production verification is complete.
