# Cognition Core V2 quoted-message evidence continuity and recurrence bugfix plan

Date: 2026-08-10

Status: completed

## Summary

- Goal: fix the two captured failure modes in the quoted-message episode:
  ordinary-response branch exhaustion during resolver recurrence, and a final
  answer that claims to answer a quoted question without carrying the quoted
  message's actual evidence.
- Scope boundary: current-turn ordinary goal contract, resolver/task-resolution
  evidence projection, action-planning answerability validation, and the
  existing text-surface/dialog handoff. Adapter delivery, persistence,
  unrelated specialists, and production database behavior remain outside the
  change.
- Change direction: keep semantic judgment with the owning LLM stages, make
  source-owned evidence explicit at each handoff, and let deterministic code
  validate provenance, evidence state, recurrence identity, and bounded
  failure behavior.
- Acceptance state: the rebuilt fixture remains a permanent failure baseline;
  new deterministic and one-at-a-time real-LLM regression gates demonstrate
  that the ordinary branch remains valid across recurrence and that a missing
  quoted fact cannot reach an answer-claiming surface or dialog.
- Implementation authority: production implementation begins only after the
  user explicitly approves this plan and the plan is moved to `approved` or
  `in_progress` in both this file and `development_plans/README.md`.

## Review Gate

This draft must be reviewed by a read-only `gpt-5.6-sol` agent with
`reasoning_effort=xhigh` and normal service speed before implementation. The
review must assess root-cause completeness, ownership boundaries, exact change
surface, contract compatibility, local-model latency, and executable test
gates. The reviewer may recommend amendments but must not edit repository
files.

Review status: completed. A read-only `gpt-5.6-sol` reviewer with
`reasoning_effort=xhigh` and normal service speed returned `REVISE`. The
blocking and major findings are incorporated below; the review record is
`test_artifacts/diagnostics/cognition_core_v2_quoted_message_plan_review_20260810.md`.
The user approved implementation on 2026-08-10; the plan entered `in_progress`
and the execution record below is complete.

## Confirmed Evidence

The production delivery id
`9bae80e4e8854854925e995552400ffb` resolves to protected trace
`llmtrace_7d8ee4de9a6c4bb697502d103ba05f3a`. The exported trace is retained
at:

`test_artifacts/diagnostics/llm_trace_llmtrace_7d8ee4de9a6c4bb697502d103ba05f3a.json`

The rebuilt case is captured in:

- `tests/fixtures/cognition_core_v2_quoted_message_case.json`
- `tests/test_cognition_core_v2_quoted_message_reproduction.py`

The fixture preserves four conversation rows, the rebuilt memory/state
projection, the exact current relay, the missing quoted-message body, the
captured goal candidate, the post-resolver observation, the action-plan
candidates, and the final dialog fragments. The deterministic and live test
artifacts under `test_artifacts/cognition_core_v2_quoted_message_reproduction/`
and `test_artifacts/llm_traces/` are raw evidence and review inputs, not
production state.

### Failure 1: ordinary-response branch exhaustion

The current episode asks whether the character can answer a question about a
quoted message. Evidence `e1` establishes that the question exists but has no
quoted-message body. The first ordinary-response candidate emits:

- `applicability=not_relationship_sensitive`;
- `current_user_relationship_state=established`;
- an ordinary response bid that otherwise cites the current episode.

The relational contract requires `not_relationship_sensitive` to pair with
`current_user_relationship_state=not_applicable` and
`stance=not_applicable`. Deterministic validation rejects the candidate. The
same branch then accepts a repaired candidate on its second cumulative
producer attempt. Resolver recurrence invokes the same ordinary producer
again; the captured mismatch consumes the third cumulative attempt and raises
`goal_bid_structure_exhausted` before action planning.

The attempt ledger is invocation-wide and keyed by the stable producing stage
and branch identity. It is deliberately not reset by graph recurrence. Local
workspace handles such as `b1` and `b2` are positional and are not branch
identities.

### Failure 2: answer loss after false resolver success

The post-resolver observation says only:

`The task resolved with validated evidence.`

Its `knowledge_we_know_so_far` contains the question itself, not the quoted
message body. The task capability had already mapped task evidence into
`evidence_refs` at its boundary, but the Cognition projection dropped those
excerpts and emitted only the generic summary; the rebuilt Cognition input
has `direct_facts=[]`. Nevertheless, action planning
repairs to:

- `goal_resolution=answerable_now`;
- `resolver_requests=[]`;
- `resolver_goal_progress=null`.

The existing resolver/surface projection carries a generic success summary but
does not carry the source-owned evidence excerpts that would answer the
question. Surface planning and dialog therefore receive an answer-now intent
without an answer-bearing fact. The live downstream replay produces the same
answerless ending, including `答案是这样的——`, while the structural dialog
verifiers pass. The verifiers currently validate shape, role direction,
surface execution claims, and other hard contract violations; they do not
prove that an answer claim is backed by the required source evidence.

## Root Cause

### A. Recurrence is re-asking an unstable ordinary-only contract

The ordinary goal producer asks the local model to emit native relationship
state even when the current request is not relationship-sensitive. The model
can copy the real relationship state (`established`) while correctly deciding
that the request is not relationship-sensitive. The strict validator rejects
that internally inconsistent pair, and the repair succeeds only after an
extra call. Because the same producer budget is shared across recurrence,
one later mismatch is terminal. This is a contract/prompt projection failure
amplified by the intentional cumulative attempt ledger, not a reason to raise
the retry cap or reset the ledger.

### B. Execution success is being treated as answer evidence

The task-resolution result, resolver observation, Cognition evidence row, and
surface resolver result collapse different facts into one generic success
summary:

1. a specialist/session may have executed;
2. a task may have produced partial or complete evidence;
3. the current user goal may or may not be answerable now.

The existing contracts do not preserve those distinctions as a typed,
source-owned handoff. The task capability already creates validated
`EvidenceRefV1` rows, but the task result contract permits a false
`resolved` result, the resolver Cognition projection ignores those evidence
excerpts and emits generic `semantic_text`, and action planning has no
deterministic evidence-state gate for `answerable_now`. The surface and
dialog handoffs also retain only capability/status/generic summary, so they
cannot prove that a final answer claim is backed by the required source
evidence.

## Relationship Between the Two Failures

The fixes are independent. Fixing Workstream A prevents the cognition graph
from stopping at `goal_bid_structure_exhausted`; it does not create the
missing quoted-message fact, repair the resolver observation, or constrain
`answerable_now`. Fixing Workstream B prevents an unsupported answer claim
after a resolver cycle; it does not make the ordinary relational-willingness
candidate valid or prevent branch exhaustion. Both workstreams are required
for the rebuilt episode to complete safely.

## Architectural Decisions

1. `task_resolution` owns the specialist/session result and its
   provenance-bearing evidence. It must distinguish complete, partial,
   pending, missing, blocked, and failed evidence states.
2. `cognition_resolver` owns the bounded observation projection. It carries
   the exact source-owned evidence excerpts and evidence state into the next
   Cognition cycle; it never turns evidence into persona, stance, or final
   wording.
3. Cognition action planning owns the semantic `goal_resolution` judgment.
   Deterministic validation may reject `answerable_now` when a trusted
   resolver evidence-state carrier says the required evidence is incomplete;
   it may not infer that state from keywords or rewrite the user's semantic
   goal.
4. Surface planning and dialog own expression and wording. They receive the
   same evidence state and bounded source excerpts and must express a missing
   fact as missing, pending, blocked, or a request for user-provided material.
   They may answer from complete source-owned evidence only.
5. The ordinary relational-willingness decision is a current-turn semantic
   decision. Once validated in cycle zero, recurrence carries it as immutable
   current-turn context; recurrence does not make a new relationship judgment
   from the existence of a resolver observation.
6. Existing `goal_resolution` values, resolver capability names, request
   limits, specialist roster, and attempt caps remain canonical. No alias,
   compatibility mapper, keyword router, final-text postprocessor, new LLM
   stage, or retry-budget increase is permitted.

## Canonical Evidence-State Contract

Scope the new carrier to `task_resolution_request` observations. Other
resolver capabilities retain their existing observation contracts; they do
not receive an artificial `not_applicable` evidence state. Add the following
conditional field to `ResolverObservationV1`:

```json
{
  "task_resolution_evidence_state": {
    "schema_version": "resolver_evidence_state.v1",
    "state": "complete",
    "remaining_needs": []
  }
}
```

For a `task_resolution_request`, the validator requires this field on every
newly built observation and rejects its absence. For every other capability,
the validator rejects the field. There is no runtime fallback that silently
maps an absent task carrier to `missing`; the rebuilt fixture gains an
explicit `state=missing` carrier while preserving all captured fields. The
raw trace and raw failure artifacts remain immutable.

`ResolverObservationV1.evidence_refs` remains the one canonical owner of
source evidence. The carrier deliberately contains no duplicate
`evidence_excerpts`. Cognition and L3 may receive a bounded, stable derived
list of prompt-safe excerpts from `evidence_refs[].excerpt`, in source order,
but the derived projection is never stored as a second source of truth.

Allowed `state` values are `complete`, `partial`, `pending`, `missing`, and
`blocked`, with this exact disposition mapping:

- task result `resolved` plus at least one usable evidence excerpt and empty
  `remaining_needs` -> `complete`;
- task result `partial` plus usable evidence and non-empty `remaining_needs`
  -> `partial`;
- task result `partial` without usable evidence or without remaining needs
  -> contract error, with no observation promotion;
- task result `deferred` -> `pending`;
- user-input, approval, unavailable-capability, specialist-failed, timeout,
  and resolver-cycle-limit dispositions -> `blocked`;
- an explicitly succeeded-looking task observation with no usable answer
  evidence -> `missing`;
- task result `resolved` without usable evidence or with non-empty
  `remaining_needs` -> contract error, with no observation promotion.

Usable evidence means a bounded task evidence row whose validated excerpt is
non-empty; a generic sentence such as “validated evidence” is not an excerpt.
The carrier is prompt-safe and contains no platform identifiers, database
identifiers, credentials, raw specialist payloads, or hidden prompt text.
Existing bounded evidence and remaining-needs limits still apply.

The resolver projection continues to return typed Cognition evidence and an
empty direct-fact list unless a capability already proves one of the existing
typed direct-fact transitions. Answer-bearing text is evidence, not an
invented `knowledge_answered` state mutation.

## Workstream A — ordinary branch recurrence stability

### A1. Make the relational decision procedure unambiguous

Modify `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`:

- Rewrite the ordinary prompt's generation procedure so it decides
  relationship sensitivity first.
- For `not_relationship_sensitive`, require the exact triple
  `not_relationship_sensitive / not_applicable / not_applicable`; do not copy
  native relationship state into this branch.
- Only relationship-sensitive requests classify the native current-user
  relationship state and choose a relationship stance.
- Place the pairing table and current-episode evidence requirement next to
  the output contract, not only in a distant prose rule.
- Keep the existing strict validator and repair contract. The validator must
  continue rejecting internally inconsistent candidates rather than coercing
  values.

### A2. Carry one validated current-turn decision through recurrence

Keep the carryover solely in `ResolverCycleStateV1`; do not add a global
persona field or modify `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`.
Define the exact `CurrentTurnRelationalWillingnessV1` transient carrier in
`cognition_resolver/contracts.py`:

```json
{
  "schema_version": "current_turn_relational_willingness.v1",
  "episode_id": "<current cognitive episode id>",
  "branch_id": "ordinary_response",
  "decision": {
    "applicability": "not_relationship_sensitive",
    "current_user_relationship_state": "not_applicable",
    "stance": "not_applicable"
  }
}
```

`episode_id` is the current `CognitiveEpisodeV1["episode_id"]` passed into
the resolver-cycle initializer. Change `new_resolver_state(...)` and
`ensure_initial_resolver_inputs(...)` to require and validate that id from
the existing `state["cognitive_episode"]`; update the three direct
`new_resolver_state` test call sites in
`tests/test_cognition_resolver_contracts.py`,
`tests/test_l2d_l3_surface_handoff.py`, and
`tests/test_cognition_core_v2_action_planning_live_llm.py` to supply the
fixture episode id. The carrier is stored in `ResolverCycleStateV1` and is
validated as a `RelationalWillingnessV2` decision immediately after cycle-zero
goal validation. It is immutable for the resolver invocation and is bound to
`branch_id=ordinary_response`, never to positional `bN` or `rN` handles.

Add `required_resolver_evidence_dependency` as a separate optional field on
`ResolverCycleStateV1`; its type is the resolver-owned
`RequiredResolverEvidenceDependencyV1` defined in Workstream B. The state
field is populated only after an accepted required resolver request and is
cleared or replaced only when that exact dependency reaches a terminal
disposition.

On recurrence, remove `relational_willingness` from the ordinary producer's
response schema and prompt. The recurrence validator validates every other
ordinary-bid field with `require_relational_willingness=False`; deterministic
assembly then inserts the stored cycle-zero `decision` into the bid and final
`CognitionCoreOutputV2`, runs the canonical relational-willingness validator,
and only then admits the result. This transports the original LLM-owned
decision without allowing a recurrence model to echo a new mismatch. A
missing or episode-mismatched carrier is a typed contract failure.

The existing three-call invocation-wide budget remains unchanged. A
recurrence must not reset or increase it. A genuinely invalid new cycle-zero
decision still fails closed; this fix only makes the validated decision
deterministically stable across recurrence.

## Workstream B — evidence-preserving answerability

### B1. Enforce task-resolution result truthfulness

Modify:

- `src/kazusa_ai_chatbot/task_resolution/contracts.py`
- `src/kazusa_ai_chatbot/task_resolution/state.py`
- `src/kazusa_ai_chatbot/task_resolution/orchestrator.py`

Implement the canonical result invariants above at the deterministic result
boundary. When a specialist says `resolved` without evidence or with
remaining needs, catch the contract error around both specialist-result
validation and `record_specialist_result(...)` in
`task_resolution/orchestrator.py`. Retain the raw failure capsule and
diagnostic reason, reject the invalid candidate and its evidence from
promotion, and terminalize the task session as `failed` through the existing
bounded failure path. Do not silently salvage it as `partial`, synthesize a
fact, or change the specialist's evidence; a separate contract is required if
partial salvage is ever desired. The rejected candidate and its evidence must
remain out of task promotion, resolver observation, scheduling, dialog, and
delivery paths. Update terminal summaries so `resolved`
means evidence-backed completion and `deferred` means continuation only.

### B2. Preserve evidence in resolver observations

Modify:

- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
- `src/kazusa_ai_chatbot/cognition_resolver/state.py`
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`

Make the task-only evidence-state carrier required for newly built
`task_resolution_request` observations and validate its relationship to
status, `evidence_refs`, and `knowledge_projection`. Update the task
observation builders for resolved, partial, deferred, failed, timeout,
duplicate-request, and cycle-limit outcomes. Non-task observation builders
must keep the existing shape and must be rejected if they contain the task
carrier.

`project_resolver_observation_for_cognition(...)` must retain the bounded
source-owned evidence excerpts by deriving them in stable order from the
validated `evidence_refs` already produced by `_task_resolution_evidence_refs`.
It must continue to preserve the observation identity and visibility map and
must not invent direct facts. `project_observations_for_cognition(...)` must
render the task evidence state, derived excerpts, and remaining needs in the
resolver context using stable semantic labels, without raw ids.

### B3. Gate action answerability on trusted evidence state

Modify:

- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`

Define and validate `RequiredResolverEvidenceDependencyV1` in
`cognition_resolver/contracts.py` as the exact transient cross-stage carrier;
`cognition_core_v2/contracts.py` consumes the same canonical type:

```json
{
  "schema_version": "required_resolver_evidence_dependency.v1",
  "accepted_request_handle": "resolver_request_1_1",
  "observation_id": "resolver_obs_<opaque-id>",
  "prompt_safe_observation_handle": "resolver_observation_1_1",
  "capability_kind": "task_resolution_request",
  "state": "missing",
  "evidence_handles": [],
  "remaining_needs": ["the quoted message body"]
}
```

The accepted-request and observation handles are allocated once by the
deterministic resolver controller from `(cycle_index, request_ordinal)` and
stored in `ResolverCycleStateV1`; they are not re-derived from the latest
observation and are not the temporary `bN`/`rN` workspace handles.
`observation_id` is code-only provenance used for exact equality checks and
is omitted from every model prompt. The prompt serializer exposes only the
prompt-safe handle, state, evidence handles, bounded derived excerpts, and
remaining needs. The dependency is absent when no accepted resolver request
is a required dependency of the current goal.

Add a validated transient `RequiredResolverEvidenceDependencyV1` input to
action planning only when the accepted resolver request is the required
dependency for the current goal. The deterministic handoff creates this
dependency at resolver-request acceptance and binds it to the exact accepted
request and resulting observation; it carries the protected observation id,
the prompt-safe observation handle, the task capability, the evidence state,
the derived evidence handles, and remaining needs. It never means “latest
resolver observation.” If no such dependency is present, unrelated RAG
evidence, an optional resolver result, and independently answerable goals are
not gated by this rule.

The action prompt must state:

- `complete` plus source-owned excerpts permits the model to consider
  `answerable_now`;
- `partial`, `pending`, `missing`, or `blocked` cannot be represented as a
  completed answer merely because the resolver execution status is
  `succeeded`;
- incomplete evidence keeps the goal on its required-evidence, user-input, or
  typed-blocked path and must not emit an unsupported answer claim;
- an empty resolver progress shell remains `null`.

Deterministic action-plan validation must reject an `answerable_now` candidate
when the explicitly bound required dependency is not `complete`, using the
existing same-stage bounded regeneration and fail-closed disposition. It must
validate the dependency's observation/evidence handle association rather than
searching free-form text, latest observations, or generic success summaries;
it must not gate independent RAG answers, optional resolver failures, or
unrelated observations, and it must not rewrite `semantic_goal`.
When the carrier is `complete`, the action planner may settle the current
goal without another optional resolver request, subject to the existing
capability/authorization rules.

### B4. Carry evidence state through surface and dialog

Modify:

- `src/kazusa_ai_chatbot/cognition_core_v2/surface.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`

Extend the task-resolution form of `SurfaceResolverResultV2` with the
validated prompt-safe observation handle, evidence state, derived bounded
source-owned excerpts, evidence handles, and remaining needs. Preserve the
existing three-field form for unrelated non-task resolver results. The
surface input carries the same internal required dependency created by action
planning; the validator compares the task result's observation/evidence
handles to that dependency and rejects omission or mismatch. Update the
surface input/output validators and all prompt payloads together.

Surface and dialog prompts must treat the evidence state as authoritative:

- complete: answer only from the supplied excerpts and preserve their
  qualifying limitations;
- partial/pending/missing: explain that the required fact is unavailable or
  still being obtained, and ask for the missing user material when that is the
  typed next step;
- blocked: express the typed blocker and next step.

The dialog semantic-fidelity verifier must receive the same source-owned
evidence state and bound dependency and reject a candidate that presents an
unavailable quoted fact as answered. This is a semantic verifier rule, not a
keyword detector or post-generation text rewrite. Existing execution-claim,
role-direction, addressee, and surface-integrity rules remain unchanged.

## Explicitly Out of Scope

- Increasing `V2_MODEL_TOTAL_ATTEMPTS`, resetting the invocation-wide ledger,
  or making an exhausted branch silently disappear.
- Adding a keyword classifier, regex answer detector, semantic post-filter,
  fallback objective, compatibility alias, or final-text replacement.
- Treating a generic resolver summary, `status=succeeded`, or a satisfied
  progress shell as proof of the quoted message body.
- Inventing a `DirectFactV2` or `knowledge_answered` transition from free text.
- Adding a resolver specialist, changing specialist routing, changing task
  budgets, changing coding-agent behavior, or changing RAG retrieval ownership.
- Changing adapter parsing, delivery receipts, persistence, consolidation,
  reflection, or production-memory writes.
- Reading `.env`, using a production database, or writing production data in
  regression tests.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/task_resolution/contracts.py`
- `src/kazusa_ai_chatbot/task_resolution/state.py`
- `src/kazusa_ai_chatbot/task_resolution/orchestrator.py`
- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
- `src/kazusa_ai_chatbot/cognition_resolver/state.py`
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/surface.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
- `src/kazusa_ai_chatbot/cognition_resolver/README.md`
- `src/kazusa_ai_chatbot/task_resolution/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `tests/fixtures/cognition_core_v2_quoted_message_case.json`
- `tests/test_cognition_core_v2_quoted_message_reproduction.py`
- `tests/test_cognition_core_v2_quoted_message_post_fix.py` (new runtime
  regression module; the baseline reproduction module remains the captured
  failure assertion)
- `tests/test_cognition_resolver_contracts.py`
- `tests/test_cognition_resolver_loop.py`
- `tests/test_cognition_core_v2_contracts.py`
- `tests/test_cognition_core_v2_action_planning_bugfix.py`
- `tests/test_cognition_core_v2_dependencies.py`
- `tests/test_cognition_core_v2_action_planning_live_llm.py`
- `tests/test_cognition_core_v2_character_carryover.py`
- `tests/test_cognition_core_v2_relational_willingness.py`
- `tests/test_cognition_core_v2_relational_willingness_live_llm.py`
- `tests/test_cognition_core_v2_relational_willingness_e2e_live_llm.py`
- `tests/test_l2d_l3_surface_handoff.py`
- `tests/test_dialog_visible_speech_and_semantic_fidelity.py`
- `tests/test_persona_supervisor2.py`
- `tests/test_persona_supervisor2_cognition_prewarm.py`
- `tests/test_task_resolution_contracts.py`
- `tests/test_task_resolution_orchestrator.py`
- `tests/test_task_resolution_inline_promotion.py`
- `tests/test_task_resolution_specialists.py`
- `development_plans/README.md`

### Keep unchanged

- The adapter and delivery contracts.
- The existing task-resolution specialist roster and budgets.
- The existing cognition attempt ledger and all configured limits.
- Direct-fact/state mutation ownership and provenance guards.
- RAG retrieval implementations and source-specific evidence ownership.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`; the new
  current-turn carrier remains inside `ResolverCycleStateV1`, which is
  already owned by the existing `resolver_state` field.
- The original protected trace export and raw failure artifacts.

## Execution Stages and Gates

### Stage 0 — baseline and approval

1. Record `git status --short` and preserve the two existing untracked
   reproduction artifacts.
2. Verify the protected trace, fixture, deterministic replay, and two live
   baseline artifacts are readable without reading `.env` or connecting to a
   production database.
3. Obtain explicit user approval, set this plan and its registry row to
   `approved`, and record the approval before production-code edits.

Gate: no production edit occurs before all three checks are recorded.

### Stage 1 — contract-first evidence state

1. Add and validate the task-only evidence-state carrier, including the
   explicit `missing` fixture state and the absence rejection for task
   observations.
2. Enforce the exact `resolved`, `partial`, and `deferred` result invariants
   at specialist-result/checkpoint/result boundaries.
3. Catch validation and recording failures together in the task orchestrator,
   preserve diagnostics, and terminalize invalid specialist results as
   `failed` without promotion.
4. Project exact source evidence from `evidence_refs` into resolver
   observations and Cognition evidence without creating direct facts.
5. Update every task observation builder and the enumerated direct-constructor
   test files; leave non-task observation shapes unchanged.

Gate: deterministic task-resolution and resolver contract suites pass, and
tests prove that generic success with empty evidence maps explicitly to
`missing`, while absent task state and false `resolved` results fail closed.

### Stage 2 — Cognition recurrence and action planning

1. Rewrite the ordinary relational decision procedure and add the typed
   current-turn carryover in `ResolverCycleStateV1`.
2. Remove relational willingness from the recurrence producer schema and
   deterministically reinsert the validated cycle-zero decision before final
   canonical validation.
3. Allocate and carry the exact required resolver dependency from accepted
   request to returned observation and action validation.
4. Reject unsupported `answerable_now` proposals only for that bound
   incomplete dependency through the existing bounded regeneration path.
5. Preserve resolver semantic goals and empty progress-shell behavior.

Gate: deterministic tests prove stable branch identity, valid carryover,
strict relational pairs, cumulative attempt accounting, exact dependency
binding, independence of unrelated RAG/optional resolver evidence, and
rejection of answer-now with incomplete required evidence.

### Stage 3 — surface/dialog handoff

1. Project task evidence state and derived excerpts into the task-resolution
   form of the existing surface resolver result, preserving the old form for
   unrelated resolver capabilities.
2. Bind the surface result to the exact required observation/evidence handles
   from Stage 2 and reject missing or mismatched bindings.
3. Update surface and dialog prompts/verifier payloads.
4. Keep complete evidence answerable and incomplete evidence explicitly
   unresolved.

Gate: deterministic handoff tests prove that no incomplete resolver result can
construct an answer-authorizing surface, while a complete source-owned result
retains its excerpts and exact dependency binding.

### Stage 4 — captured regression and live LLM gates

Run each live case separately with `venv\Scripts\python.exe`, inspect its raw
JSON before starting the next case, and author the Markdown review from the
inspected output:

0. Run the immutable captured baseline module and inspect the existing raw
   failure artifacts, preserving assertions that the original candidate was
   relationally invalid and the original downstream text was answerless.
1. rebuilt ordinary-response goal branch with the current prompt and
   reconstructed memory/history: valid relational pair and no branch
   exhaustion across one resolver recurrence, with the recurrence decision
   inserted deterministically;
2. real task-resolution specialist/orchestrator case using the rebuilt
   read-only execution context: a resolved result must carry a
   provenance-bearing answer excerpt, while an incomplete result must remain
   partial/pending/blocked;
3. real action-planning recurrence with a captured incomplete observation:
   no `answerable_now` and no unsupported final answer route;
4. real surface/dialog handoff with an incomplete observation: final wording
   states the missing/pending fact or requests the required material;
5. real surface/dialog handoff with a complete source-owned evidence result:
   the final wording preserves the supplied answer fact, allowing natural
   paraphrase but forbidding omission of the required fact.

The post-fix assertions live in the new
`tests/test_cognition_core_v2_quoted_message_post_fix.py`; the existing
`tests/test_cognition_core_v2_quoted_message_reproduction.py` remains a
failure-baseline module and is not rewritten to make the old output appear
healthy.

The five post-fix live node ids are fixed before implementation as:

- `test_post_fix_goal_branch_recurrence_live_llm`;
- `test_post_fix_task_resolution_evidence_live_llm`;
- `test_post_fix_action_planning_incomplete_dependency_live_llm`;
- `test_post_fix_surface_dialog_incomplete_live_llm`;
- `test_post_fix_surface_dialog_complete_live_llm`.

Required commands use one live test node at a time. The two immutable
baseline reproductions are run as explicit deterministic nodes so the module's
live tests are not collected accidentally:

```powershell
venv\Scripts\python.exe -m py_compile <changed Python files>
venv\Scripts\python.exe -m pytest -o addopts="" tests\test_cognition_core_v2_quoted_message_reproduction.py::test_rebuilt_case_reproduces_goal_branch_exhaustion -q
venv\Scripts\python.exe -m pytest -o addopts="" tests\test_cognition_core_v2_quoted_message_reproduction.py::test_rebuilt_case_reproduces_answer_loss_after_false_resolver_success -q
venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm <one test node> -q -s
```

The live gates must emit raw prompt/input/output/parsed-state artifacts and
an agent-authored Markdown review. A passing pytest status or schema parse is
only a harness gate; the review must judge grounding, answer completeness,
relationship correctness, and absence of unsupported claims.

Gate: all five post-fix live cases are run and inspected individually. If the
live endpoint is unavailable, execution sign-off is blocked and the plan
remains open; an unavailable endpoint is not a quality pass.

### Stage 5 — adjacent regression and independent review

1. Run the focused deterministic modules that own the changed contracts:
   `tests/test_task_resolution_contracts.py`,
   `tests/test_task_resolution_orchestrator.py`,
   `tests/test_cognition_resolver_contracts.py`,
   `tests/test_cognition_resolver_loop.py`,
   `tests/test_cognition_core_v2_contracts.py`,
   `tests/test_cognition_core_v2_action_planning_bugfix.py`,
   `tests/test_cognition_core_v2_dependencies.py`,
   `tests/test_cognition_core_v2_relational_willingness.py`,
   `tests/test_cognition_core_v2_character_carryover.py`,
   `tests/test_l2d_l3_surface_handoff.py`,
   `tests/test_dialog_visible_speech_and_semantic_fidelity.py`, and
   `tests/test_cognition_core_v2_quoted_message_post_fix.py` with live nodes
   excluded.
2. Run `git diff --check`, prompt-render checks for every changed formatted
   prompt, and the project-required Python style checks.
3. Review the complete diff against the change surface and forbidden paths.
4. Fix all in-scope findings and rerun the affected gates.
5. Update the READMEs, this plan's execution evidence, and the registry only
   after the implementation and review gates pass.

Execution result: COMPLETE. All five post-fix live gates and both original
live replay nodes passed individually. The focused deterministic gate passed
320 tests with seven live nodes excluded. The task-resolution orchestrator
and inline-promotion rerun passed 20 tests, and the rebuilt reproduction plus
post-fix deterministic nodes passed three tests with seven live nodes excluded.
The final source checks passed compilation, the selected undefined-name and
unused-import scan, and `git diff --check`.

## Acceptance Criteria

### Workstream A

- A non-relationship-sensitive current event produces the exact valid
  relational pair `not_relationship_sensitive / not_applicable /
  not_applicable` in the real goal gate.
- A relationship-sensitive event still uses the native relationship state and
  allowed stance combinations; established status does not grant unrelated
  requests relationship sensitivity.
- The validated current-turn decision survives resolver recurrence under the
  stable `ordinary_response` branch id, and the recurrence model cannot replace
  it because the field is absent from the recurrence producer schema.
- The invocation-wide three-call budget remains visible and unchanged, and
  the captured invalid candidate still fails closed when deliberately replayed
  as invalid input.
- The rebuilt live episode reaches action planning without
  `goal_bid_structure_exhausted`.

### Workstream B

- A task result cannot be `resolved` without validated evidence and no
  remaining needs.
- Task-resolution observations retain exact bounded source-owned evidence
  excerpts derived from canonical `evidence_refs` and an explicit task-only
  evidence state; generic success text alone is never complete, and absent
  task state is rejected rather than mapped silently.
- Invalid specialist `resolved` results are retained as diagnostics and
  terminalized as `failed` without evidence promotion.
- The quoted-message fixture with no quoted body and no answer evidence cannot
  produce `goal_resolution=answerable_now` for the required resolver goal.
- An empty resolver progress shell remains `null`; no invented satisfied
  checklist is accepted.
- Complete source-owned evidence reaches Cognition, surface, and dialog and
  the required answer fact survives in the final wording under the same bound
  observation/evidence dependency.
- Incomplete/pending/missing evidence reaches a natural missing/pending/user
  input response and never reaches an unsupported answer claim.
- Independently answerable RAG evidence, optional resolver outcomes, and
  unrelated observations remain unaffected by the required-dependency gate.
- Existing structural verifiers continue to pass valid outputs, while the
  semantic-fidelity verifier catches an answer claim unsupported by the typed
  evidence state.

### Architecture and safety

- No new LLM stage, keyword routing, semantic post-filter, final-text rewrite,
  retry expansion, compatibility alias, specialist, adapter change, or
  production-memory write is introduced.
- RAG and resolver evidence remain evidence and never become persona stance or
  final wording authority.
- All live test artifacts are redacted of credentials and remain in the
  ignored diagnostic area; no `.env` file is read.

## Test Matrix

| Layer | Required coverage | Test style |
| --- | --- | --- |
| Task result contract | resolved/partial/deferred invariants and empty-evidence rejection | deterministic |
| Resolver projection | task-only state, canonical evidence excerpts, visibility, no invented direct facts, absent-state rejection | deterministic |
| Goal recurrence | strict relational pairs, stable branch id, deterministic carryover, cumulative ledger | patched/deterministic |
| Action planning | exact required dependency binding; incomplete bound evidence cannot be answerable; independent evidence remains eligible | patched/deterministic |
| Surface contract | bound task evidence state/excerpts survive input/output validation; unrelated result form remains valid | deterministic |
| Dialog handoff | bound missing evidence cannot be rendered as answered; complete fact survives | patched LLM/verifier |
| Goal prompt | current rebuilt memory/history produces valid ordinary bid | real LLM, one case |
| Task specialist | source-owned evidence is produced or bounded incomplete result returned | real LLM, one case |
| End-to-end downstream | incomplete and complete resolver states produce distinct grounded wording | real LLM, one case each |

## Review and Closeout Evidence

The plan-review record must remain at
`test_artifacts/diagnostics/cognition_core_v2_quoted_message_plan_review_20260810.md`
and include the GPT-5.6 Sol verdict, blocking findings, and the amendments
adopted here. The final implementation review must verify that the plan's
deterministic carryover and exact dependency binding are present in code,
not only in prompts.

The implementation record must include:

- pre-edit and post-edit `git status --short`;
- commands and results for every deterministic gate;
- one raw JSON artifact and one inspected Markdown review per live case;
- the exact final evidence-state payload for incomplete and complete cases;
- the final dialog excerpts showing the missing-fact and answer-preservation
  behaviors;
- independent review findings, fixes, and rerun results;
- the final plan/registry status transition to `completed` only after user
  confirmation that implementation is complete.

### Completed implementation record

- Pre-edit status contained the already supplied plan/fixture/reproduction
  artifacts; those user-owned changes were preserved. Final `git status
  --short` contains only the scoped plan, cognition/resolver/task-resolution
  implementation, direct contract-test updates, the rebuilt fixture, and the
  reproduction/post-fix test modules.
- The five post-fix live artifacts and inspected reviews are paired at:
  `test_artifacts/cognition_core_v2_quoted_message_post_fix/goal_branch_recurrence_live_llm.json`
  and `reviews/goal_branch_recurrence_live_llm.md`;
  `task_resolution_evidence_live_llm.json` and its review;
  `action_planning_incomplete_dependency_live_llm.json` and its review;
  `surface_dialog_incomplete_live_llm.json` and its review; and
  `surface_dialog_complete_live_llm.json` and its review.
- The incomplete surface evidence payload was:
  `{"schema_version":"resolver_evidence_state.v1","state":"missing","evidence_excerpts":[],"evidence_handles":[],"remaining_needs":["the quoted message body"]}`.
  The complete payload carried `state=complete`, the excerpt
  `雪凪问的是：“周五下午三点在车站见。”`, and the bound evidence handle
  `resolver_evidence_surface_complete`.
- The recurrence artifact recorded no failure, three model calls, and the
  same canonical relational decision triple across both bids. The action
  artifact recorded `goal_resolution=requires_required_evidence`, one
  resolver request, and `resolver_goal_progress=null` despite the resolver's
  generic succeeded status.
- Incomplete final wording stated that the quoted content was unavailable and
  requested the question正文 without asserting the unsupported Friday/station
  fact. Complete final wording preserved `周五下午三点在车站见` and `车站`.
- The GPT-5.6 Sol plan review's blocking findings were incorporated before
  implementation. The final in-scope source scan found and fixed one missing
  orchestrator import, one undefined dialog annotation name, and unused
  imports; affected deterministic tests and the selected style checks were
  rerun successfully.

## Mandatory Skills and Repository Rules

- `development-plan` governs lifecycle, approval, scope, gates, and closeout.
- `local-llm-architecture` governs bounded local-model prompts and ownership.
- `debug-llm` governs raw live evidence and agent-authored human reviews.
- `test-style-and-execution` governs deterministic versus patched versus live
  tests and one-at-a-time live execution.
- `py-style` and `cjk-safety` apply to all Python changes.
- `llm-trace-debug` governs protected trace evidence handling.
- `no-prepost-user-input` forbids deterministic semantic routing or
  post-processing of the user's request.
