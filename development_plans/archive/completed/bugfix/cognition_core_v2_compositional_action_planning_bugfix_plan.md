# Cognition Core V2 Compositional Action Planning Bugfix Plan

## Summary

- Status: completed.
- Change class: large.
- Cutover: bigbang.
- Approval: the user explicitly instructed planning and immediate execution on
  2026-07-17.
- Parent plan: `development_plans/archive/completed/short_term/cognition_core_v2_stage_2_integration_plan.md`.
- System RCA: `test_artifacts/cognition_core_v2/action_selection_failure_rca/cognition_v2_route_selector_system_rca_and_proposal.md`.
- Production-usage evidence: `test_artifacts/cognition_core_v2/action_planning_capacity_evidence/production_action_planning_capacity_review.md`.
- Frozen proof corpora: QQ group `638473184` and QQ private user
  `673225019`, twenty chronological turns each, captured at the original input
  boundary.

This bugfix replaces the reduced V2 route selector with one semantic,
compositional action planner. The new planner preserves the production action
registry, resolver registry, HIL/approval lifecycle, delayed work, future
cognition, memory lifecycle, and three-request capacity. Immediate visible
speech becomes orthogonal to private actions, so a character acknowledgement
can coexist with as many as three private action requests without inventing a
second `speak` vocabulary. The normal action-selection LLM call count stays
unchanged.
Valid action plans retain the single-call path. One bounded retry by the same
semantic planner is permitted only after exact contract validation fails.

The same scope also contains the two technical-boundary defects exposed during
the RCA: graph errors are returned as operational errors without becoming
character history, and visual directives remain a terminal optional surface
whose environment default is disabled.

## Context

The captured private turn 20 produced a normal speech bid and a `speak`
affordance. The route model returned both `route: speech` and an action handle;
the exact validator rejected that plausible model output because speech was
represented twice. The service then persisted `杏山千纱 is busy right now,
please try again later.` as assistant history with a delivery ID. This is a
system failure, rather than a character-quality decision.

The production baseline action selector is a semantic planner. It may emit up
to three resolver requests or up to three action requests, including paired
visible acknowledgement and private/delayed work. The current V2 selector only
chooses one route and one optional capability already chosen by a goal branch.
The current V2 connector exposes only `local_context_recall`, and its reduced
resolver loop omits the baseline pending clarification, approval, goal-progress,
and general capability lifecycle.

The retained production trace window contains no `l2d_action_selection` rows,
so actual historical request cardinality cannot be measured from trace data.
Persistent production rows do prove that the broader capability surface is in
active use: accepted background/future work, HIL, approval, future cognition,
memory lifecycle, delivery retry, and scheduled follow-up records all exist.
The source contract therefore supplies the cardinality floor; the persisted
rows supply the usage proof.

Ownership remains:

```text
goal branches -> complete motives and character judgments
workspace -> primary/supporting/suppressed motive partition
action planner -> visible route plus semantic action or resolver requests
resolver/action owners -> validation, execution, persistence, permission
L3/dialog -> final visible wording
visual planner -> terminal private visual metadata only
service -> operational error delivery and conversation persistence boundary
```

## Mandatory Skills

- `development-plan`
- `local-llm-architecture`
- `no-prepost-user-input`
- `py-style`
- `cjk-safety` for Python prompts and replay fixtures containing CJK
- `test-style-and-execution`
- `debug-llm`
- `database-data-pull`
- `llm-trace-debug`
- `character-test`

## Mandatory Rules

- The parent agent owns this plan, failing tests, documentation, verification,
  replay state restoration, quality review, remediation, and sign-off.
- Exactly one production-code subagent implements the production files after
  the intended focused tests fail.
- Exactly one separate review subagent reviews the verified implementation,
  performs no production edits, and reports findings to the parent.
- LLM stages own semantic interpretation, route choice, request choice,
  acceptance, and commitment meaning. Deterministic code owns exact shapes,
  bounds, handle resolution, permissions, persistence, and execution.
- User text and generated dialog receive no keyword routing, sample-specific
  filters, or deterministic semantic rewriting.
- Prompt dynamic content is bounded JSON in `HumanMessage`; static policy is a
  triple-single-quoted system string.
- Live LLM calls run one case at a time and each result is inspected before the
  next call.
- The two frozen proofs reuse the exact original input, timestamp, production
  history boundary, memory, image, character profile, relationship state, and
  preceding replay state recorded by their manifests.
- Replay writes target the guarded `_test_kazusa_live_llm` database only; each
  scenario is restored and restoration is verified.
- Visual directives are terminal metadata. They have no downstream agent and
  default to disabled unless the environment explicitly enables them.

## Must Do

1. Remove action/resolver route authority from goal bids. Goal branches emit
   complete motives, not executable capability choices.
2. Remove the same-route restriction from workspace support selection.
3. Replace `RouteDecisionV2` with a fixed-shape semantic action-plan result.
4. Support up to three action requests or up to three resolver requests per
   plan, matching the production minimum.
5. Allow `route: speech` with zero to three non-speech action requests. Exclude
   `speak` and internal `apply_memory_lifecycle_update` from planner affordances.
6. Keep resolver and action requests mutually exclusive. `evidence` requires
   resolver requests; `action` requires action requests; `deferral` and
   `silence` require neither.
7. Anchor every selected request to an admitted bid and a prompt-local
   capability handle. Deterministic code copies that bid's target roles and
   evidence handles without reinterpreting meaning.
8. Project all executable resolver capabilities into V2 and use the established
   bounded resolver recurrence for immediate evidence, HIL, approval,
   pending-resume resolution, goal progress, duplicate detection, timeouts, and
   max-cycle behavior.
9. Preserve the uncommitted V2 replacement state between resolver cycles and
   commit only the final replacement state at the caller boundary.
10. Trace action-planner prompt, raw output, parsed output, validation status,
    and failure class under a dedicated protected stage.
11. Return graph exceptions with `content_type: operational_error`, an empty
    delivery ID, failed trace status, and zero final-dialog count. Keep the
    operational text out of assistant history, continuity, and consolidation.
12. Set the visual-directives default to disabled and assert that the visual
    stage remains terminal with no downstream agent.
13. Update the Stage 2 plan, frozen contract, execution manifest, subsystem
    READMEs, and affected tests to the one canonical contract.
14. Prove the implementation with deterministic tests, focused real-LLM action
    planner gates, and fresh sequential 20+20 frozen replays.
15. Project action affordances only when their registry-declared runtime
    context is present. Model the fixed memory-lifecycle review decision as the
    closed `active_commitment_lifecycle` value already required by its action
    spec, and expose that action only when trusted active commitments exist.
16. On an invalid action-plan object, record the failed protected trace and
    allow the same planner exactly one bounded replacement attempt containing
    the contract error and a bounded copy of its prior output. Validate the
    replacement against the unchanged exact contract; a second failure remains
    an operational error.
17. Eliminate optional free-form decisions from planner-visible no-argument
    actions. Their registry projections use one closed operational verb, while
    `semantic_goal` and `reason` retain model-owned meaning. Contract errors
    identify the prompt-local handle and exact allowed decision rule so the
    bounded replacement can correct the field without guessing.
18. Use prompt projection as the single authority for semantic-appraisal role
    handles and candidate-to-evidence bindings. Semantic question planning and
    appraisal validation must consume that canonical map rather than derive
    handles independently from evidence-id text.
19. Preserve typed speaker/addressee provenance across decontextualization,
    goal cognition, surface planning, dialog, and compliance. A drifted bid may
    not broaden action eligibility: every selected capability must match the
    current evidence's requested real effect, including accepted coding work.
20. Treat physical completion and delivery presuppositions as action narration
    regardless of grammatical person. Verbal acceptance, permission, teasing,
    and affectionate offers remain allowed; claims that the requested act was
    performed, completed, delivered, or received do not.
21. Superseded by
    `cognition_chain_responsibility_allocation_bugfix_plan.md`: authoritative
    typed participation constrains each relevance stage's first-call action
    space, and the conditional same-owner rechecks are removed. Semantic
    linkage, withdrawal, resolution, and retained-media judgment remain with
    the existing relevance owners.
22. Restore the required subjective-appraisal projection at the V2 cognition
    to consolidation boundary. Populate it from the admitted cognition reason,
    include it in the canonical consolidator state, and prove the user-memory
    lane no longer raises `KeyError: subjective_appraisals`.

## Deferred

- New action capability kinds or new resolver capability kinds.
- Increasing resolver cycle count, action execution concurrency, or scheduler
  capacity.
- Visual image generation or a local text-to-image connection.
- Adapter-specific rendering of `operational_error` beyond the existing brain
  response contract.
- Production deployment and writes outside guarded local test workflows.
- Retuning the character solely to outperform a historical baseline.

## Cutover Policy

This is a bigbang contract update. Every in-repo producer, consumer, validator,
test, and authoritative Stage 2 document moves to the canonical action-plan
shape together. The removed route-only vocabulary receives no alias, mapper,
fallback, dual field, or compatibility translation. The generic registries and
existing stored resolver/pending contracts remain canonical persistence and
execution boundaries.

## Target State

- One action-planning LLM call owns route and semantic request selection.
- Valid plans use one action-planning call; invalid plans may use one bounded
  corrective call by the same semantic owner.
- Goal branches cannot pre-empt or contradict the action planner's route.
- Immediate speech has one meaning: the visible L3/dialog route.
- A visible acknowledgement can accompany as many as three private actions.
- Action-only and evidence-only outcomes retain three-request production
  capacity.
- Adding a registry capability exposes a new affordance without changing the
  planner schema or writing keyword routing.
- HIL/approval and pending resolution retain their production lifecycle.
- Resolver recurrence sees the previous cycle's uncommitted V2 state.
- Character history contains character dialog only, not infrastructure errors.
- Visual planning is disabled by default and terminates at visual metadata.

## Design Decisions

### One semantic planning owner

Goal branches keep `intention`, `desired_outcome`, `concrete_detail`, `reason`,
`private_monologue`, roles, evidence, consequences, and confidence. Route and
capability fields are removed. Workspace collapse partitions these complete
motives and may support a primary motive with a different operational need.

The action planner receives only admitted bids, current bounded episode/evidence
context, registry-derived action and resolver affordances, and prompt-safe
resolver context. It selects route and requests in one call.

### Speech is a surface, not an action affordance

`route: speech` sends the selected intention to L3. The action registry remains
unchanged for other owners, but `speak` is omitted from this planner's
affordances. This removes the duplicated `speech`/`speak` choice that caused the
captured failure.

### Composition increases useful capacity

The baseline permits three action requests total, where visible acknowledgement
may consume one `speak` slot. V2 permits one visible speech route plus three
non-speech action requests. Action-only and resolver-only limits remain three.
This preserves baseline capability and improves visible/private composition
with a focused authorization call only on turns that propose executable
actions.

### Reuse established deterministic owners

The action planner emits semantic request proposals only. The structural
responsibility-allocation plan adds a focused semantic authorization boundary
for proposed executable actions and derives route shape deterministically.
Existing action-spec and
resolver code continue to own validation, permissions, materialization,
pending rows, execution, persistence, and delivery. The live V2 response path
uses the established full resolver recurrence rather than its reduced
local-context-only behavior.

### Runtime-valid registry affordances

The first frozen group replay exposed a second technical contract defect at
turn 3. The planner received `memory_lifecycle_update` when no active
commitment existed. Its prompt projection simultaneously used an optional
empty `decision` contract and instructed the model to provide
`review_kind=active_commitment_lifecycle`. One first response placed a value
outside the advertised decision contract and failed validation; a full-capture
retry selected the same unusable action with an empty decision, after which the
specialist skipped it because no commitment existed.

The registry projection therefore declares a bounded runtime-context
requirement for every capability. Generic connector filtering compares that
requirement with trusted runtime facts; it does not inspect user text or infer
semantics. `memory_lifecycle_update` requires active-commitment context and
uses a closed decision whose only value is the registry's existing
`active_commitment_lifecycle` enum. This keeps the capability fully available
for its production-backed use case, removes a non-executable planner choice,
and lets later capabilities add an availability requirement without changing
the planner schema or adding capability-specific prompt logic.

### Exact validation with bounded planner recovery

Frozen group turn 4 produced a semantically coherent speech-plus-future-
cognition selection, but the raw model output ended immediately after the
action array. Generic JSON syntax repair returned that partial object, which
then correctly failed the fixed five-field action-plan contract. Increasing
capacity or filling omitted semantic fields in deterministic code would either
miss the failure mode or weaken ownership.

The same action planner therefore receives at most one corrective attempt after
any parse or contract error. The retry includes only the bounded validator
error and a bounded copy of the rejected output, asks for a complete
replacement object, and passes through the same exact validator. The initial
failure and repair attempt receive distinct protected trace stages. This is a
generic local-model reliability boundary: it does not classify user input,
change action meaning, add a fallback route, or create a second semantic owner.

The clean replay later exposed the remaining optional-decision ambiguity at
group turn 13. Both initial and replacement outputs selected
`trigger_future_cognition`, then placed a prose response strategy into its
nominally optional `decision` despite the empty-default rule. Planner-visible
no-argument capabilities now use a registry-owned closed operational verb:
`background_work_request=enqueue`, `accepted_task_status_check=check`, and
`trigger_future_cognition=schedule`. The model still owns whether to select the
action and the complete semantic objective; deterministic validation owns the
capability command vocabulary. The future-cognition affordance also states
that ordinary response rehearsal and persona maintenance complete in the live
turn and are not reasons to schedule another cognition cycle.

The uninterrupted group proof then exposed an independent semantic failure at
turn 15. Protected trace `llmtrace_baf0e5cc75904cd59a8ab69596b110d7`
showed a present-scoped user preference and present-scoped cognition, followed
by an unnecessary `trigger_future_cognition` request for reply rehearsal and a
dialog clause that converted playful control into a literal future exclusivity
rule. The real compliance route accepted that unsupported clause. This is a
system boundary failure across action availability, surface grounding, final
rendering, and verification rather than a single bad phrase.

Production evidence bounds the correction. The recent 500-row action-attempt
export contains 27 `trigger_future_cognition` attempts; every one originated
from the `self_cognition_bot` private-cognition path and none originated from a
visible user-message turn. The registry therefore marks this capability with
the generic `private_cognition_source` availability context. The connector
derives that context only from canonical private episode trigger sources
`internal_thought` and `scheduled_recall`. This preserves all observed
production use and the existing scheduler/materialization owner while removing
an ungrounded visible-chat affordance. It does not inspect or classify user
text.

Source-grounded expressive creativity remains an explicit quality contract.
Possessive, controlling, jealous, exclusive, or tsundere style may make the
current wording vivid, but style alone cannot authorize a new factual claim,
future rule, obligation, or exclusivity condition. Surface planning and dialog
rendering must preserve that boundary. Compliance must audit each candidate
claim against current visible percepts, treating agreement between a drifted
surface and candidate as insufficient grounding. Explicit source-supplied
future meaning remains fully available.

The restarted group proof exposed a different cross-layer defect at turn 14.
Protected trace `llmtrace_b63b30d06cb34ba792958a4af4df40b2` preserved the
user's request for Kazusa to infer the user's bun preference through intake and
initial cognition, but the bid weakened direct inference into confirmation,
the content plan allowed either a guess or an ask-back, the renderer swapped
the preference-holder, and compliance accepted the swap. In the same trace,
the action planner selected `background_work_request` to analyze history and
prepare its current reply even though neither the source nor admitted bid
accepted delayed work. That is an unauthorized lifecycle request, not harmless
creative variation.

The corrective contract is operation- and owner-based rather than
sample-specific. When visible evidence requests an answer, inference, guess,
explanation, acceptance, refusal, or negotiation, goal cognition must preserve
that operation and its actors; a rhetorical question may add character voice
but cannot substitute for performing it. Surface planning, rendering, and
compliance apply the same actor/action/target check. The verifier must reject a
candidate that merely restates, redirects, or asks back a requested answer or
inference.

Private action requests additionally require an admitted motive whose desired
outcome needs the capability's durable or out-of-turn effect. They are not a
workspace for the planner's own reasoning, memory recall, response rehearsal,
or wording preparation. `background_work_request` remains fully available for
explicitly accepted delayed work, including repository analysis, and local
context recall remains owned by the resolver path when evidence is genuinely
missing. This preserves production capacity while preventing internal thought
from being misrouted into a durable queue.

The first private R18 restart exposed the remaining action-description root at
turn 2. Trace `llmtrace_91c902110c334edeab6a0cf3e28b2800` shows that goal
cognition treated a physical chat command as an executable body-position
change, the planner queued `background_work_request` to generate and later
present an action description, surface planning required a claim that the
movement had completed, dialog emitted that first-person completion claim, and
compliance accepted it. The bracket format was never the system root; the
pipeline incorrectly modeled the character brain and text channel as a
physical actuator.

The platform boundary is explicit: no current action capability actuates the
character's body or changes a physical scene. Cognition may decide the
character's stance toward a physical request and may choose to accept, refuse,
negotiate, tease, give permission, or issue literal spoken instructions. It
must not claim that a requested movement occurred. Action planning cannot use
background work or another capability to execute, generate, or later present a
physical-action description. Text surface and dialog preserve the stance as
literal words only. A first-person claim that the currently requested physical
movement is happening, finished, or established as a body position counts as
action narration even when it is grammatically speakable. Compliance rejects
that claim and requests the existing one-shot grounded dialog repair.

This boundary preserves extensibility: a future image generator remains a
terminal visual sibling and still cannot actuate the scene; any future real
actuator would require a new explicit registry capability, permission policy,
materialization owner, and delivery surface. No current action route is
repurposed or removed.

### Semantic appraisal uses one handle authority

The private R18 proof reached a technical failure at turn 11 with protected
trace `llmtrace_31f9e3415df5469c95abed3466f65d8f` and
`CognitionStateError: proposition subject handle is unknown`. The appraisal
model followed its supplied contract. The contract itself was impossible:
semantic question planning derived candidate handle `ce11` from evidence id
`e11`, while prompt projection assigned `ce1` to the first evidence row. The
validator accepted `ce11` because it checked the independently generated
question allowlist; deterministic reduction then rejected it because the
projection map had no such handle.

Prompt projection becomes the sole prompt-local handle owner. Semantic source
planning consumes its canonical handle map, and candidate evidence validation
resolves the same map's `candidate:<kind>:<evidence_handle>` reference. Sparse
or monotonically increasing evidence ids therefore remain exact evidence
provenance without influencing prompt-local numbering. This is a big-bang
internal contract correction: it adds no semantic fallback, handle alias,
keyword classifier, LLM retry, or extra model call.

### Typed dialogue roles survive every semantic owner

The final-code private restart exposed an actor/target reversal at turn 2.
Trace `llmtrace_fb7f85897ca844f6b35efe156d3d8cb9` shows the complete chain. The
decontextualizer changed a clear direct imperative into `杏山千纱张开腿，跨坐在我身上`
while leaving first-person ownership implicit. Goal cognition then called
Kazusa the user, changed the user-commanded action into instructions for the
user, and produced a bid for the opposite actor. The action planner selected
`accepted_coding_task_request` because that drifted bid used the word “task,”
even though current evidence described a physical chat request. Surface and
dialog preserved the inverted bid, and compliance accepted it because its
grounding input contained text without typed speaker/addressee roles.

The correction follows existing ownership. The LLM decontextualizer treats a
direct imperative as complete and preserves its implicit addressee rather than
inserting a new grammatical subject. The episode connector projects typed
`current_user` speaker and `self` addressee/imperative-subject semantics from
the user-message envelope. Goal cognition, L3, dialog, and compliance consume
those same semantic role anchors. Action planning must match a capability's
real effect against current evidence as well as the bid; generic words such as
“task” or “action” never make a capability eligible, and a drifted bid cannot
turn physical chat into coding work. Coding start, continuation, delayed work,
resolvers, and all other production capabilities remain available for evidence
that actually requests their declared effect. No user-text keyword filter,
deterministic role inference, capability removal, or added model call is used.

The final-code group restart exposed the last format-independent actuator leak
at turn 6. Trace `llmtrace_68605e06ae7948e5b5b6f87da193193a` shows correct
verbal-stance goal cognition and a zero-action speech plan, followed by dialog
`好了！拿到了就赶紧给我起来吃早饭！`. Compliance accepted it because the
actuator rule named first-person execution claims, while this wording
presupposed completion through the user's receipt. The semantic failure is the
claim that the physical affection was delivered, not its grammatical person or
punctuation.

All semantic text owners therefore distinguish a literal verbal offer such as
permission or affectionate acceptance from enactment. They reject declarative
or presuppositional claims that the requested physical act already occurred,
was completed, was delivered, or was received, including “you got it” wording.
This closes the action-description class without suppressing vivid spoken
affection, removing routes, or adding deterministic phrase filters.

### Operational failures stay operational

The brain response may return a sanitized retry message, but its content type
marks it as operational and it receives no assistant-history row or delivery
tracking ID. Failed LLM trace runs finalize with zero character dialog.

## Contracts And Data Shapes

### Goal bid

`ActionBidV2` and `GoalBidDraftV2` remove:

```text
requested_route
requested_action_kind / requested_action_handle
requested_resolver_capability / requested_resolver_handle
```

All remaining motive, provenance, target, consequence, and monologue fields are
required and validated.

### Planner result

The model emits one exact object:

```json
{
  "route": "speech|evidence|action|deferral|silence",
  "action_requests": [
    {
      "bid_handle": "b1",
      "action_handle": "a1",
      "decision": "registry-grounded semantic decision",
      "semantic_goal": "bounded semantic objective",
      "reason": "why this action advances the admitted bid"
    }
  ],
  "resolver_requests": [
    {
      "bid_handle": "b1",
      "resolver_handle": "r1",
      "semantic_goal": "bounded evidence objective",
      "reason": "why this evidence is needed"
    }
  ],
  "resolver_pending_resolution": {
    "decision": "answered|cancelled|rejected|approved",
    "reason": "bounded semantic reason"
  },
  "resolver_goal_progress": null
}
```

Arrays are always present and contain zero to three rows. Pending resolution is
always present as null or the semantic `decision`/`reason` choice; durable
`resume_id` and schema version never enter the prompt. Goal progress is always
present as null or the existing prompt-safe semantic V1 object. Deterministic
code binds a pending decision to the current validated active pending row,
rejects a decision without a matching row, and then validates the complete V1
pending-resolution contract. The projector resolves handles and produces
canonical `SemanticActionRequestV2` / `ResolverCapabilityRequestV2` rows with
`reason`, the action's required bounded `decision`, copied roles, and copied
evidence handles. Each action affordance and materialized request also carries
a deterministic `context_ref`; it is empty for unscoped actions and resolves
from the selected affordance for context-bound actions. The live persona path
loads the existing prompt-safe open coding-run contexts for the trusted turn
scope. It projects a start affordance plus one contextual accepted-coding
affordance per open run, so multiple runs remain distinguishable by handle and
the model can ground status, revision, approval, blocker, summary, and cancel
decisions in current run status without inventing a durable identifier.
Action decisions are validated against registry-projected
semantics and passed faithfully to action-spec materialization. Cognition adds
no capability-specific decision branch, route-as-decision substitution, or
hardcoded accepted-coding `start` value.

### Cognition input/output

- `CognitionCoreInputV2` adds bounded `resolver_context` and an optional
  deterministic-only validated `pending_resolver_resume`. The latter is never
  serialized into an LLM prompt or protected prompt payload.
- `CognitionCoreOutputV2` carries nullable validated
  `resolver_pending_resolution` and `resolver_goal_progress`.
- Global state receives canonical `resolver_capability_requests`, pending
  resolution, goal progress, action specs, and the selected visible route.
- Recurrent cognition consumes `cognition_state_update.replacement_state` when
  its scope and owner match the current episode; the first cycle reads storage.

### Operational response

On graph exception:

```json
{
  "messages": ["<sanitized retry text>"],
  "content_type": "operational_error",
  "delivery_tracking_id": ""
}
```

The response is delivered to the caller while persistence and character
continuity remain empty for that failed turn.

## LLM Call And Context Budget

- Goal cognition: unchanged one call per active branch, up to fourteen in
  bounded parallel execution.
- Workspace collapse: unchanged zero or one call.
- Action planning: unchanged one call; it replaces the current route-selector
  call.
- Text surface, dialog, semantic verifier, optional repair: unchanged.
- Visual planner: zero calls by default; one terminal optional call when
  explicitly enabled.
- Action-planner dynamic JSON cap: 24,000 characters.
- Bid count: primary plus admitted supporting bids only.
- Request count: three actions or three resolvers.
- Resolver context uses the existing bounded prompt-safe projection.
- The implementation must record actual prompt length and preserve existing
  route model configuration and timeout. Only the already specified one-shot
  action-plan contract repair may add a conditional call.

## Change Surface

### Delete

- `RouteDecisionV2` and its route-only validator.
- Goal-bid route/capability fields and same-route workspace filtering.
- Route-only prompt wording and tests that require duplicated speech/speak
  authority.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`
- existing accepted-task coding-run context loader wiring, without a new
  collection or query contract
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py` only as required to expose
  the full established recurrence to V2 without parallel semantic ownership.
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/brain_service/turn_settlement.py` only to close a
  current failed settled-assessment lease without inventing a semantic ignore,
  while preserving a stale newer version for reassessment.
- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py`
- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`
- `src/kazusa_ai_chatbot/consolidation/core.py`
- `src/kazusa_ai_chatbot/consolidation/schema.py`
- directly affected subsystem READMEs and HOWTO visual-default documentation
- the Stage 2 integration plan, contract spec, and execution manifest
- directly affected deterministic and live-LLM tests
- `experiments/cognition_core_v2_real_conversation_replay.py` only for a fresh
  output-root selector or review metadata; the frozen input manifests remain
  immutable.

### Create

- focused action-planning live-LLM test/harness if no existing test can express
  the fixed-shape contract
- fresh group and private post-fix 20-turn artifacts and human review documents

### Keep

- action and resolver registries
- action-spec materialization and executors
- persisted pending-resume, goal-progress, accepted-task, background-work,
  scheduler, permission, and delivery contracts
- V2 appraisal, state transition, goal graph, L3, dialog, consolidation, and
  residue semantics outside required contract projection
- visual planner as a terminal optional output with no downstream agent
- frozen historical production dialog/residue and original input manifests

## Overdesign Guardrail

- No second action planner, action-routing graph, policy DSL,
  capability-specific schema branch, automatic action-planner retry, or
  deterministic keyword classifier. One conditional recheck by the existing
  relevance stage owner is allowed only for the typed direct-address rejection
  contradiction described above.
- No sample-specific Chinese phrase or turn-ID logic.
- No new persistence collection.
- No action execution inside cognition.
- No adapter-specific action semantics.
- No model-count or context-budget increase.
- No compatibility shim for removed route-only fields.

## Agent Autonomy Boundaries

The production-code subagent may make implementation-local choices within the
listed production files while preserving the exact contracts, cardinalities,
ownership boundaries, and cutover policy. It may report a blocking design
conflict to the parent. It may not edit tests, plans, proof artifacts, frozen
manifests, or expand scope.

The parent may adjust tests and documentation to reflect the exact canonical
shape. Any change to request cardinality, LLM call count, persistence meaning,
action/resolver ownership, frozen inputs, or visual downstream behavior
requires explicit user direction.

## Implementation Order

1. Freeze git state, usage evidence, and both replay manifests.
2. Draft this plan and mark it `in_progress` under the user's execution
   authorization.
3. Amend the Stage 2 contract/manifest to supersede route-only selection.
4. Parent writes focused failing tests for contract, planner composition,
   resolver capacity, recurrence state, operational failure containment, and
   visual default/terminal behavior.
5. Parent runs the focused tests and records intended failures.
6. Spawn exactly one production-code subagent with the production change
   surface and passing criteria.
7. Parent updates plan/docs/evidence while the production owner implements.
8. Integrate and run compile, focused, integration, and full non-live tests.
9. Run focused real-LLM planner cases one at a time: captured speech case,
   speech plus delayed action, multi-action, resolver, HIL, and approval.
10. Restore and prepare the group frozen scenario; run turns 1-20 sequentially,
    inspecting every output; cleanup and verify restoration.
11. Restore and prepare the private R18 scenario; run turns 1-20 sequentially,
    inspecting every output; cleanup and verify restoration.
12. Produce baseline/V2 human review documents with per-turn independent
    quality judgments and aggregate technical/quality conclusions.
13. Spawn exactly one independent review subagent after verification.
14. Parent remediates in-scope findings and reruns affected through full gates.
15. Close evidence, mark the bugfix completed, archive it, and update Stage 2
    Checkpoint I sign-off only when every acceptance criterion passes.

## Execution Model

- Parent/architect: tests, commands, documentation, evidence, replay operations,
  quality review, remediation, and sign-off.
- Production owner: exactly one subagent, production files only.
- Independent reviewer: exactly one different subagent after implementation
  verification, review only.
- Live evidence: sequential one-case invocations with inspected output.

## Progress Checklist

- [x] Production source and persistent usage evidence collected.
- [x] Route-selector technical failure reproduced with the production model.
- [x] Exact group and private frozen manifests identified.
- [x] Executable plan drafted and execution authorized.
- [x] Stage 2 authoritative documents amended.
- [x] Focused failing tests written and intended failures recorded.
- [x] Production implementation completed by one subagent.
- [x] Focused and integration tests pass.
- [x] Open coding-run contexts are loaded and contextual continuation actions
  remain distinguishable when more than one run is active.
- [x] Focused real-LLM action-planner gates pass one at a time.
- [x] Group frozen turns 1-20 rerun and reviewed.
- [x] Private R18 turns 1-20 rerun and reviewed.
- [x] Test database restored and verified after both scenarios.
- [x] Typed direct-address discard recheck and consolidation appraisal
  projection pass deterministic and production-shaped live verification.
- [x] Full non-live regression passes.
- [x] Independent review completed.
- [x] In-scope findings remediated and verification repeated.
- [x] Plan and Stage 2 Checkpoint I closed.

## Verification

### Focused deterministic

Run with `venv\Scripts\python.exe -m pytest`:

- cognition V2 contract/dependency tests
- resolver contracts, loop, and persona-graph integration tests
- L2d/L3 surface handoff tests
- service failure/event tests
- config and visual terminal-routing tests

Required assertions include:

- speech plus three non-speech actions is valid;
- four actions, four resolvers, mixed action/resolver rows, unknown handles,
  suppressed-bid handles, and cross-bid provenance are rejected;
- action-only and evidence-only routes require requests;
- speech never needs or emits a `speak` affordance;
- all five resolver capability kinds are offered and execute through their
  existing owner;
- HIL/approval persists pending state and can coexist with visible V2 speech;
- follow-up input loads and resolves matching pending work;
- the second cognition cycle receives the first uncommitted replacement state;
- graph errors create no assistant row, delivery ID, continuity, consolidation,
  or final-dialog trace count;
- visual default is false and the visual node remains terminal.
- accepted-coding `start` remains available without an open run, while every
  continuation decision is bound through a trusted contextual affordance to
  one currently open run.
- runtime-context-free turns omit `memory_lifecycle_update`; turns with a
  trusted active commitment expose it with closed decision
  `active_commitment_lifecycle`.
- one malformed or contract-invalid model object triggers exactly one bounded
  replacement attempt; valid first responses trigger no additional call.
- planner-visible no-argument actions expose one exact registry command rather
  than an optional free-text decision slot.

### Focused real LLM

Each case runs and is inspected separately:

1. Captured private turn 20 production-shaped selector context.
2. Visible acknowledgement plus `background_work_request`.
3. Two or three independent private action requests.
4. `local_context_recall` evidence request.
5. `human_clarification` pending request.
6. `approval_preparation` pending request.

Pass requires a valid first response for each focused normal case, correct
admitted-bid grounding, correct registry handle, no duplicated speech/speak
failure, and no technical fallback in character output. A separate malformed-
shape case must recover through exactly one bounded planner replacement.

### Frozen 20+20 quality proof

For every turn, record:

- user input;
- baseline old monologue/residue;
- baseline old dialog;
- new V2 monologue/residue;
- new V2 dialog;
- new emotions/state changes;
- action route/requests/resolver status;
- analysis.

Quality is judged independently of the baseline on:

- Kazusa personality and voice;
- vividness, care, creativity, and human naturalness;
- direct responsiveness to current input;
- consistency with prior dialog and actor/action/target direction;
- absence of action-description narration;
- absence of contradictory or meaning-altering invention;
- technical completion without infrastructure text.

Minor creative elaboration is acceptable when it does not contradict current
input, previous dialog, character boundaries, or accepted commitments.

### Full regression and static checks

- compile changed Python files;
- collect affected tests;
- run the repository non-live suite using the project deselection contract;
- scan for removed route-only fields and stale visual-default documentation;
- inspect `git diff --check` and final `git status --short`.

## Independent Plan Review

The evidence and plan were parent-authored before implementation. The required
independent agent gate applies after verified code because the user instructed
immediate execution and the plan contract allocates the available subagent
roles to one production owner and one later reviewer.

## Independent Code Review

One review-only subagent receives the plan, diff, deterministic results,
focused live evidence, and both 20-turn review documents. It must check:

- baseline and optimized capacity;
- single semantic ownership;
- registry extensibility;
- pending/HIL/approval lifecycle;
- recurrence state and commit boundary;
- operational failure containment;
- visual terminal/default behavior;
- action-description and semantic-drift regressions;
- test completeness and anti-cheat compliance.

The parent owns all remediation and reruns.

## Acceptance Criteria

1. Goal bids contain no route or capability choice.
2. One planner call emits the fixed route-free action-plan proposal object;
   proposed executable actions receive a focused authorization call before
   materialization.
3. Speech plus up to three non-speech actions validates and materializes.
4. Action-only and resolver-only paths retain three-request capacity.
5. All runtime-eligible registry action capabilities except `speak` and
   internal apply are planner-visible without capability-specific branching.
6. All five existing resolver capabilities are planner-visible and execute
   through the established deterministic owner.
7. HIL, approval, pending resolution, and goal progress pass focused tests.
8. Recurrent V2 cognition preserves uncommitted state and commits only once.
9. Captured turn 20 completes with character output and no route validation
   failure.
10. Graph errors are operational responses and never character history.
11. Visual directives default disabled and remain terminal.
12. Focused normal live cases pass on their first model response, and one
    generic malformed-shape case recovers through exactly one bounded retry.
13. Both exact frozen 20-turn sequences complete sequentially with guarded
    restoration and inspectable raw artifacts.
14. Both human review documents contain every required per-turn field and an
    aggregate quality/technical assessment.
15. No new action-description narration or contradiction class is introduced;
    any surfaced regression is remediated and rerun.
16. Full non-live regression and independent review pass.
17. No keyword classifier, sample-specific patch, compatibility shim, second
    semantic owner, unconditional new LLM call, or unbounded prompt is
    introduced.
18. Runtime-ineligible actions are absent from the planner prompt, while every
    eligible production capability retains its registry-defined capacity and
    materialization owner.
19. Immediate answer and inference requests preserve their response operation
    and actor/action/target direction through cognition, surface, rendering,
    and verification.
20. Durable private actions are selected only for an admitted motive requiring
    their out-of-turn effect; planner reasoning and reply preparation never
    become queued work.
21. Physical chat requests produce only a verbal stance in the current system;
    no cognition bid, action request, surface plan, dialog, or verifier verdict
    treats the text or visual branches as a physical actuator.
22. Superseded by the structural responsibility-allocation plan: typed
    participation cannot be discarded at frontline, authoritative settled
    turns use one constrained semantic disposition call, and explicit
    redirects or grounded redundancy remain semantically rejectable.
23. Consolidation receives the required subjective-appraisal evidence from V2
    cognition, and production-shaped user-memory consolidation completes
    without a missing-field error.

## Risks

- A wider planner contract may challenge the local model. Mitigation: one
  fixed shape, prompt-local handles, bounded arrays, and focused first-response
  live gates.
- Speech-plus-action composition can accidentally double-deliver. Mitigation:
  omit `speak` from affordances and keep L3 as the only immediate text owner.
- Migrating resolver recurrence can cause extra state commits. Mitigation:
  explicit uncommitted-state and single-final-commit tests.
- Pending HIL/approval may lose visible acknowledgement. Mitigation: make V2
  speech route count as a visible surface in terminal/blocker rules.
- Frozen tests are expensive and model-variable. Mitigation: immutable inputs,
  sequential inspection, raw traces, and quality judgments that distinguish
  technical failure from acceptable creative variation.
- Operational error typing may affect callers that assume `text`. Mitigation:
  keep the existing ChatResponse shape and sanitized message while testing the
  empty delivery/persistence boundary.

## Approval Boundary

The user's 2026-07-17 instruction authorizes this plan and its production
implementation, deterministic verification, guarded local real-LLM testing,
and proof documentation. Production deployment, production writes, and new
capability kinds remain outside this authorization.

## Execution Evidence

Record commands, outcomes, live case artifacts, replay restoration guards,
review findings, remediation, and final sign-off here during execution.

- 2026-07-17: production usage exports and protected trace probe completed;
  source contract establishes capacity, persistent rows establish active
  capability usage, and retained traces contain no L2d selection rows.
- 2026-07-17: captured turn-20 route-selector failure reproduced twice with the
  production model and production-shaped prompt.
- 2026-07-17: parent red gates recorded the intended failures: missing
  `plan_actions` API at collection; five goal/workspace contract failures with
  seven existing dependency tests passing; full-resolver source assertion
  failure; operational response returned `text`; and visual directives still
  defaulted to true.
- 2026-07-17: implementation contract refinement approved: pending-resolution
  semantics remain LLM-owned, while deterministic code binds the active
  resume identifier after planning so durable identifiers remain outside the
  prompt boundary.
- 2026-07-17: frozen group turn 3 reproduced a new first-response contract
  failure. Protected trace `llmtrace_841ce99a9783444091b4acfa70132e6d`
  failed because an action decision was outside its affordance. Full-capture
  retry `llmtrace_b8e7d9d0db184f8aa4b05ddb658e386c` proved the systemic
  mismatch: `memory_lifecycle_update` was offered without active commitments,
  its optional decision contradicted the fixed `review_kind` instruction, and
  the downstream specialist skipped the selected action. The plan now requires
  registry-declared runtime availability and the canonical closed lifecycle
  decision before frozen replay restarts.
- 2026-07-17: after that correction, frozen group turn 4 failed under trace
  `llmtrace_648ab69783af4d49954bca291dae2959`. Full capture showed a coherent
  speech-plus-`trigger_future_cognition` choice whose raw JSON stopped after
  `action_requests`; deterministic JSON syntax repair salvaged an incomplete
  object and exact validation rejected its missing top-level fields. The plan
  now permits one bounded same-owner replacement attempt after contract
  failure, with exact revalidation and separate trace evidence.
- 2026-07-17: the same real turn 4 then exercised the implemented recovery
  under trace `llmtrace_cd5b5bec6dce462b9ed7ca899d7cbb1e`: the initial
  `action_planning` step failed exact validation, the sole
  `action_planning.repair` step succeeded, and the turn returned character
  dialog with nine protected trace steps and no operational response.
- 2026-07-17: clean group turn 13 trace
  `llmtrace_93334120accf49988b941a107a297d3f` showed both planner attempts
  choosing `trigger_future_cognition` while filling its optional `decision`
  with response-strategy prose. This proved the optional field was structurally
  ambiguous rather than a missing retry. The plan now requires closed registry
  verbs for every planner-visible no-argument action, precise corrective error
  text, and a future-cognition boundary excluding live-response rehearsal.
- 2026-07-17: uninterrupted group turn 15 trace
  `llmtrace_baf0e5cc75904cd59a8ab69596b110d7` showed present-scoped cognition,
  an ungrounded future-cognition rehearsal request, a generated future
  exclusivity rule, and a false-negative compliance verdict. All 27 recent
  production `trigger_future_cognition` attempts came from
  `self_cognition_bot`, so the corrective gate retains that private path and
  removes the action only from visible user-message cognition. The quality
  contract now requires source-grounded creativity and per-claim semantic
  verification.
- 2026-07-17: restarted group turn 14 trace
  `llmtrace_b63b30d06cb34ba792958a4af4df40b2` exposed two independent
  systemic failures: direct preference inference was weakened into an
  ask-back with a swapped preference-holder, and the planner selected
  `background_work_request` for current-reply preparation. The corrective
  gates now preserve requested operations and semantic roles end to end, and
  require durable action selection to be grounded in an admitted out-of-turn
  outcome rather than the planner's internal reasoning.
- 2026-07-17: first private R18 restart turn 2 trace
  `llmtrace_91c902110c334edeab6a0cf3e28b2800` proved that action-description
  failures came from an incorrect actuator model spanning cognition, planning,
  surface, dialog, and compliance. The plan now requires literal verbal stance
  only for physical chat requests and preserves future extensibility through a
  new explicit capability boundary rather than implicit text enactment.
- 2026-07-17: production-capacity refinement approved: every selected action
  carries the generic semantic `decision` required by scheduled speech and
  accepted-coding continuations; deterministic code preserves it instead of
  substituting route or hardcoded `start`.
- 2026-07-17: implementation audit found that the existing open coding-run
  loader had no production caller and `action_selection_context.coding_runs`
  was populated only by tests. The plan now wires that prompt-safe context into
  per-run affordances, restoring and improving grounded multi-run continuation
  selection without adding an LLM call or persistence contract.
- 2026-07-17: frozen source manifests locked before proof execution: group
  SHA-256 `9e62764d7bbf830164eb9a76bd365fc70f23ebcd9a62f533f98a9430773f43cb`;
  private R18 SHA-256
  `04d9241af2e1b71ad62937da164288b7f4562acda7a49b8e8d95bbaf1b90c989`.
- 2026-07-17: final group proof turns 2 and 18 both carried authoritative
  `addressed_to_global_user_ids`, a typed character mention, and a native reply
  to Kazusa, yet returned zero cognition steps and silence. The failure occurs
  before V2 cognition because the shared frontline LLM may return `discard`
  despite its typed participation contract. The plan now permits one bounded
  same-owner recheck for this exact contract contradiction while preserving
  semantic redirect and withdrawal handling.
- 2026-07-17: the first post-frontline-fix proof still produced zero cognition
  steps at exact group turn 2. Direct live frontline probes passed, and the
  eight-second service duration includes the group settlement window,
  localizing the remaining false negative to the persona-aware settled
  relevance stage. The contract now applies the same conditional same-owner
  recheck to a valid settled `ignore`, while retaining the stage's grounded
  redirect, redundancy, and retained-media ignore reasons.
- 2026-07-17: the next service proof passed turns 1-6, then exact turn 7
  `@杏山千纱 千纱妈妈` returned silence in 1.7 seconds despite authoritative
  character target and reply metadata. This proves that repeating the complete
  routing task in the recheck lets the weak local model repeat the original
  mistake by treating a direct nickname summon as lacking a request. Both
  relevance rechecks now use smaller semantic-disposition schemas and map only
  their validated closed result to the existing route.
- 2026-07-17: the same production-shaped proof log repeatedly raised
  `KeyError: 'subjective_appraisals'` in the user-memory consolidation lane.
  `CognitionCoreOutputV2` already supplies the admitted subjective reason, but
  `_build_consolidator_state` omitted the field required by
  `memory_units._json_payload`. The plan now restores that canonical projection
  and adds a regression gate.
- 2026-07-18: parent review of the Private-turn-5 and Group-turn-15 RCA fix
  found that terminal settled-relevance failures completed response futures but
  left matching coordinator turns in `ASSESSING`. The remediation adds an
  explicit operational close that preserves stale newer versions, makes the
  bounded authoritative repair consume its exact validator error and rejected
  output, and records initial/repair trace stages separately. Focused tests
  cover retry success, retry exhaustion, first-assessment wait, terminal
  operational completion, stale-lease preservation, QQ system-notice delivery,
  and branch-failure cause propagation.

## Lifecycle Closure

- Closure date: 2026-07-18.
- Final targeted review passed 14 focused checks and 208 broader affected
  checks after the Private-turn-5 and Group-turn-15 remediation review.
- The accepted failsafe preserves response-future completion, closes matching
  settled-turn leases, preserves stale newer versions, and keeps the user-
  visible busy response available only after bounded retry exhaustion.
- Group-turn-15's four resolver cycles and 275,769 ms elapsed time are a Stage
  3 efficiency case rather than an unresolved Stage 2 correctness blocker.
