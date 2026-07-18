# Cognition Chain Responsibility Allocation Bugfix Plan

## Summary

- Status: completed.
- Change class: large.
- Cutover: bigbang.
- Approval: the user explicitly instructed implementation on 2026-07-17.
- Parent plan:
  `development_plans/archive/completed/short_term/cognition_core_v2_stage_2_integration_plan.md`.
- Related plan:
  `development_plans/archive/completed/bugfix/cognition_core_v2_compositional_action_planning_bugfix_plan.md`.
- Frozen proof corpora: the existing twenty-turn QQ group `638473184`
  replay and twenty-turn QQ private-user `673225019` replay, captured at each
  original input boundary.

This bugfix replaces direct-address prompt-recheck loops with typed ownership
contracts. A separate one-shot schema repair remains available only when a
multi-option authoritative settled result violates its exact output shape.
Protocol facts constrain the relevance action space before the local LLM is
called, while the LLM retains semantic linkage, withdrawal, resolution, and
character-participation judgment. Executed, scheduled, pending, failed, and
unavailable action outcomes remain distinct through the L3 surface boundary,
so surface and dialog models no longer infer execution truth from prose.

## System RCA

### Relevance ownership defect

Canonical intake already resolves native platform reply IDs and typed target
metadata into `character`, `broadcast`, or other-participant labels. The
frontline and settled relevance prompts nevertheless expose `discard` or
`ignore` for authoritative private/direct/broadcast participation. A weak
local model can contradict the protocol fact. The current implementation then
asks the same model to revisit the same decision in a second call. That retry
changes wording and attention but leaves responsibility allocation unchanged.

Protocol ownership and semantic ownership must be separated:

- deterministic intake owns whether the character is an authoritative
  participant;
- the frontline LLM owns linkage to one supplied open turn and prelude use;
- settled relevance owns whether the admitted turn still warrants cognition;
- recipient withdrawal, already-resolved meaning, and character judgment stay
  semantic decisions;
- deterministic code owns which semantic dispositions are available from the
  supplied typed evidence and observation phase.

### Action-effect ownership defect

`ActionResultV1` already distinguishes `executed`, `scheduled`, `pending`,
`failed`, and other lifecycle statuses. Its V2 prompt projection currently
maps `executed`, `scheduled`, and `pending` to `completed`. L3 and dialog are
therefore given false completion authority and must recover the truth from
prompt prose. This causes physical-action narration and false completion
claims to be treated as a prompt-tuning problem.

The action executor remains the only authority for execution status. L3 may
express an effect as completed only from an `executed` result. Scheduled and
pending work may be acknowledged only in their actual lifecycle state. An
unexecuted user request remains a topic for verbal stance, permission,
refusal, negotiation, or instructions; it is never an action result.

### Action-selection ownership defect found by the live gate

The first post-implementation physical-request live gate reproduced a second
systemic failure. The combined action planner was asked to choose a semantic
capability, construct its request, and restate the protocol route. Despite
explicit capability text and physical-actuator prohibitions, the configured
model selected `background_work_request` to generate and later present a
physical-action description. Prompt policy therefore cannot make one
weak-model call simultaneously own proposal, eligibility, and deterministic
route shape.

The correction separates those responsibilities:

- the action planner proposes composable semantic action or resolver requests;
- a focused semantic authorization stage runs only when executable action
  requests were proposed and decides whether each proposed capability's real
  effect is grounded by its cited current evidence;
- deterministic code applies that authorization decision and derives the route
  from the canonical episode output mode plus the validated request sets;
- the planner no longer emits a duplicate route field;
- rejected executable candidates never reach action-spec materialization.

Capability authorization remains semantic and LLM-owned; deterministic code
does not classify user text. Route shape, request mutual exclusion, output-mode
compatibility, and enforcement of the authorization result are deterministic.
The focused authorization call preserves the registry-driven three-action
capacity and adds latency only on turns that actually propose an executable
action.

### Live-sequence integration defects

The first fresh frozen group replay exposed three additional responsibility
errors after the focused gates passed:

- a single authoritative fragment was offered `recipient_withdrawn` even
  though no later fragment existed to withdraw or redirect its recipient;
- the local planner was required to reproduce the resolver protocol's complete
  goal-progress ledger, so two otherwise valid resolver proposals failed on an
  omitted `deliverables` list;
- a resolver terminal `speak` action was generated correctly, but surface
  selection consulted the earlier cognition route instead of the terminal
  action spec and suppressed the reply.

The constrained disposition builder now offers recipient withdrawal only for
multi-fragment turns. Protocol code supplies and preserves the resolver ledger
shape while the planner owns only a bounded semantic delta. A validated
terminal `speak` action spec is authoritative for text-surface selection.

The replay then reproduced a separate turn-11 silence. Its cause was synthetic
clock ownership, not relevance semantics: frozen inbound messages retained
their captured timestamps, while generated assistant rows used the live wall
clock. Earlier replay replies therefore appeared after later frozen requests
and incorrectly made `already_resolved` structurally available. The replay
harness now rebinds only its generated assistant rows to each frozen source
turn after the production response is persisted. Production outbound timing
and relevance behavior remain unchanged.

The complete private replay then exposed a separate visible-content boundary
failure. The general dialog compliance model currently owns current-turn
meaning, actor direction, scope, future commitments, execution grounding, and
surface formatting in one prompt. The user subsequently approved action
description as visible roleplay in plain, bracketed, first-person, or
third-person form. A fresh private turn still failed because the surface model
twice classified plain first-person physical description as
`non_speech_surface`, including after repair.

The system correction removes action-description and punctuation quality from
the fatal surface taxonomy. The focused surface verifier now judges only false
system, tool, platform, or other character-brain capability execution against
the exact permitted-action ledger. It runs alongside semantic fidelity and
the conditional role-direction verifier. Their issue sets are combined for at
most one grounded renderer repair, and repaired output must pass every active
owner before delivery.
Deterministic code only validates verdict shape, combines exact issue lists,
and blocks an unverified repair; it does not classify user or dialog text.

### Resolver recurrence ownership defect found by the frozen live sequence

Fresh group replay run `2624f73fdc4eb0143c9d` completed turn 10 in 290.856
seconds after three `local_context_recall` executions for the same unresolved
theme. The first execution returned eight projected evidence rows, including
relevant relationship and physical-contact context. Two later executions
returned bounded blocked packets after malformed local-model output, but the
capability adapter labelled both observations `succeeded` with zero rows. The
planner then rephrased the objective, bypassing the exact-objective duplicate
check and consuming the full three-cycle recurrence budget.

The failure crosses three ownership boundaries:

- the planner proposes resolver work but also decides, without a focused
  check, whether earlier evidence has already satisfied the proposed need;
- the local-context adapter projects a blocked packet as a successful
  observation, losing execution truth before recurrence sees it;
- deterministic retry protection recognizes exact objective reuse and timeout
  only, so a failed same-capability attempt can be retried under new prose.

The system correction keeps resolver need semantic while separating it from
proposal generation. A focused resolver-authorization stage runs only when
resolver requests are proposed. It receives each request, its admitted bid,
cited current evidence, and bounded resolver context, and decides whether the
evidence need remains unresolved and materially useful. It may preserve up to
three distinct requests and may authorize a genuinely narrower follow-up even
after an earlier successful observation. Deterministic code classifies a
blocked capability packet as failed and prevents another same-capability
attempt after timeout or failure. It does not compare user text or infer
semantic equivalence from keywords.

### Selection-owner verification defect found by the private live sequence

Fresh private replay run `d218612f9305bd3b4906` stopped at turn 1 with an
unambiguous selection-owner reversal. The typed decontextualizer correctly
stated that `self` must tell `current_user` which next action `self` wants
`current_user` to perform, with `selection_required=true` and
`selection_owner_role=self`. Goal cognition instead handed control back to the
user, and the general semantic-fidelity verifier accepted dialog asking the
user to command the character.

The role contract was present and correct; the failure came from assigning
internal contradiction, direct-input conflict, general actor direction, and
nested selection ownership to one weak-model verifier. The system correction
adds a focused role-direction verifier only when a typed response operation
requires an explicit selection. It receives the candidate role frame plus the
bounded role-explicit content and response-operation fields, and judges only
response owner, selection owner, embedded actor, and embedded target. Refusal
or negotiation remains valid character judgment; delegating a required
selection to another role is rejected. The same focused owner rechecks the one
bounded repair. Deterministic code only activates the stage from the typed
`selection_required` flag and merges verdicts.

Fresh retry run `c04a380ebf8290ee3549` proved a second ownership defect. The
focused role verifier and general semantic verifier both rejected the initial
reversal, but drifted free-text `active_visible_boundaries` and
`style_guidance` instructed the repair model to ask the user for a command,
opposite to the typed current selection owner. The repair repeated the
reversal and the turn failed. A direct live retry with prompt precedence alone
still followed the drifted prose; another retry repaired the role correctly
but the surface checker then treated that prose as a fatal boundary and
crashed while formatting its explanation.

The hard-error policy does not define conformity to generated L3 prose as a
fatal error. Free-text L3 boundaries and style continue to shape ordinary
generation, while hard verification owns current-input conflict, internal
contradiction, role reversal, and false capability
execution. The one bounded repair receives typed current percepts, verified
hard issues, original dialog, and the exact permitted-action ledger, with no
L3 content, boundary, or style prose. All three focused owners recheck it.

The next private retry exposed that a final dialog verifier is too late to be
the only selection-owner defense. The typed operation was correct, but goal
cognition used strong submissive private continuity to create a bid whose
purpose was to return the required choice to the user. One live generation
passed the stochastic final checks; a diagnostic rerun rejected and repaired
the same downstream dialog. The system correction therefore verifies a typed
required selection immediately after goal cognition. A rejected goal is
regenerated from the typed operation, current evidence, affect, relationship,
character constraints, and scene context without the rejected bid or private
continuity prose, then rechecked by the same focused owner. Turns without a
typed required selection add no call. Final dialog verification remains the
independent last boundary.

### Action-plan schema containment defect found by private turn 14

The first private turn-14 execution for `我来舔干净千纱身上的` returned an
operational error. Protected trace
`llmtrace_b15e1365395a429b8b7bcf89b643580d` showed that a second-cycle action
planner output and its one repair both violated the exact resolver-request
shape. Deterministic validation raised before semantic authorization could
reject or contain the proposal, so optional action planning crashed an
otherwise valid speech turn.

The production action router already tolerates malformed individual model
rows. V2 now applies the same responsibility boundary without a compatibility
shim: deterministic code canonicalizes the known proposal envelope, ignores
unknown fields, retains valid rows, drops invalid rows individually, and caps
each request list at three. Mixed action/resolver output remains a semantic
contract error and receives the one existing whole-object replacement. If that
replacement is still unusable, planning returns an empty proposal and speech
continues. If focused action authorization cannot produce a usable replacement,
all candidates are denied. Neither path can authorize execution or reduce the
valid three-request capacity. The final private run completed the same turn
with a normal response and a succeeded protected trace.

## Ownership Boundary

```text
typed intake facts
  -> deterministic relevance mode/action-space selection
  -> one semantic frontline decision
  -> one semantic settled disposition
  -> cognition/action planning
  -> action registry validation and execution
  -> exact action lifecycle projection
  -> L3 content planning
  -> dialog wording and semantic verification
```

- RAG returns evidence.
- Cognition and relevance LLM stages own semantic judgment.
- Deterministic code owns closed action availability, validation, execution
  truth, and lifecycle projection.
- L3 and dialog own visible wording within those typed constraints.
- No deterministic user-text keyword classifier or post-hoc semantic rewrite
  is permitted.

## Mandatory Skills And Agent Contract

- `development-plan`
- `local-llm-architecture`
- `no-prepost-user-input`
- `py-style`
- `cjk-safety` when Python edits contain CJK
- `test-style-and-execution`
- `debug-llm`
- `character-test`
- `database-data-pull` and `llm-trace-debug` when database or protected trace
  evidence is retrieved

The parent agent owns this plan, production implementation, failing tests,
documentation, verification, frozen replay control, remediation, review, and
sign-off. The completed historical implementation-subagent contribution is
preserved in the evidence record; all remaining work is parent-only under the
user's updated instruction.

## Must Do

### A. Constrained frontline relevance

1. Derive an authoritative-participation mode only from typed/protocol facts:
   private scope, `character` target, `broadcast` target, or character reply.
2. For authoritative input with no open turn and no supplied prelude, admit a
   new turn without an LLM call. When supplied preludes exist, make one
   constrained semantic call whose only route is `start` and whose only
   judgment is which supplied preludes complete the current intent.
3. For authoritative input with open turns, make one constrained LLM call with
   only `start|append`; retain exact supplied-slot validation.
4. For ordinary group traffic, preserve the current semantic
   `discard|start|append` contract and bounded prompt.
5. Remove the direct-address discard recheck and its second model call.
6. Keep explicit recipient withdrawal as settled semantic judgment rather
   than deterministic text interpretation.

### B. Constrained settled relevance

7. For authoritative admitted turns, expose one closed semantic disposition:
   `proceed`, `wait`, `recipient_withdrawn`, `already_resolved`, or
   `unavailable_retained_media`.
8. Offer `wait` only while more observation time is available.
9. Offer `already_resolved` only when fresh during/after-turn history exists.
10. Offer `unavailable_retained_media` only when the media evidence status is
    `partial_media_view`.
11. Map dispositions deterministically to the existing coordinator contract:
    `proceed -> proceed`, `wait -> wait`, and terminal non-response
    dispositions to `ignore`.
12. For ordinary group traffic, preserve the current `ignore|proceed|wait`
    semantic judgment.
13. Remove the direct-address ignore recheck and its second model call.
14. Select a sole evidence-derived disposition deterministically. For several
    available dispositions, reject unavailable or malformed output and permit
    one bounded same-owner schema repair at the hard deadline. A structurally
    invalid first assessment uses the coordinator's one-time wait; repeated
    hard-deadline invalidity becomes an operational failure rather than a
    semantic ignore.

### C. Exact action lifecycle projection

15. Replace the lossy V2 `completed|failed|unavailable` result status with the
    exact surface lifecycle vocabulary `executed|scheduled|pending|failed|
    unavailable`.
16. Project only `ActionResultV1.status == executed` as `executed`.
17. Preserve `scheduled` and `pending` exactly; map rejected, validated, and
    cancelled results to `unavailable` because they authorize no completed
    external effect.
18. Preserve typed target roles and bounded semantic result summaries.
19. Update L3 content, preference, and dialog contracts so completion claims
    require an `executed` permitted action result; scheduled and pending claims
    remain bounded to their lifecycle status.
20. Preserve the existing action registry, resolver registry, three-request
    capacity, delayed work, future cognition, HIL/approval, memory lifecycle,
    terminal visual-surface behavior, and visual-agent default-disable flag.

### D. Separate proposal, authorization, and route ownership

21. Remove `route` from the action-planner model output and derive it
    deterministically from `episode.output_mode`, primary-bid presence, and
    the validated action/resolver request sets.
22. Preserve speech plus up to three private actions for `visible_reply`,
    evidence routing for resolver requests, action routing for private or
    scheduled executable work, and silence when no grounded bid exists.
23. Add one focused semantic action-authorization call only when the planner
    proposes executable actions. Require one exact decision for every proposed
    candidate and retain only authorized candidates.
24. Give the authorizer the selected registry affordance, proposed semantic
    goal, admitted-bid context, and cited current evidence. Current evidence,
    rather than drifted bid prose alone, must ground the capability's real
    effect.
25. Trace initial and repair authorization boundaries. Permit at most one
    bounded same-owner repair for a malformed authorization object.
26. Keep resolver proposal semantic in the planner and keep all action-spec
    permission, validation, execution, and persistence owners unchanged.
26a. Canonicalize optional planner proposals row by row. Preserve usable rows,
    contain an unusable replacement as an empty plan, and contain an unusable
    authorization replacement as deny-all without authorizing work or removing
    valid three-request capacity.

### E. Separate resolver proposal, authorization, and retry ownership

27. Add one focused semantic resolver-authorization call only when the planner
    proposes resolver requests. Require one exact decision for every proposed
    request and retain only authorized requests.
28. Give the resolver authorizer the proposed request, cited admitted bid,
    current evidence, and bounded resolver context. It judges whether the need
    remains unresolved and materially useful; it does not rewrite the request
    or final dialog.
29. Preserve up to three distinct resolver requests, multiple resolver
    capabilities, and semantically narrower follow-up requests after useful
    evidence. Empty resolver proposals add no model call.
30. Project a blocked local-context packet as a failed resolver observation and
    failed RAG event. Prevent a renamed same-capability retry after a timeout or
    failed observation while preserving follow-up after success.

### F. Separate nested role-direction verification

31. Add one focused semantic role-direction check only when a current visible
    percept carries `response_operation.selection_required=true`.
32. Supply only the candidate role frame and bounded current role fields:
    role-explicit content, response owner, selection owner, embedded actor, and
    embedded target. Exclude style, history, residue, and upstream content-plan
    prose.
33. Treat a clear delegation of the required selection to another role as an
    unacceptable reversal while preserving refusal, negotiation, jokes,
    ellipsis, and genuinely ambiguous readings.
34. Merge its exact verdict with the existing semantic-fidelity and surface
    verdicts, and run the same focused owner once after the bounded repair.
34a. Verify typed required-selection ownership at the goal boundary as well.
    Regenerate a rejected bid from clean typed/current context and recheck it;
    ordinary turns add no call.
    Turns without a typed required selection add no model call.

### G. Verification and proof

35. Keep free-text L3 boundaries and style as ordinary generation guidance,
    outside the fatal surface taxonomy and hard-error repair payload. Repair
    from typed current percepts, verified hard issues, original dialog, and
    exact permitted-action results, then run every focused recheck.
36. Add deterministic regression tests for every constrained action space,
    unavailable disposition, single-call bound, and exact lifecycle status.
37. Add red gates for route-free planning, output-mode route derivation,
    semantic authorization approval/rejection, no-call behavior for empty
    action proposals, and retained three-action capacity.
38. Add red gates for resolver authorization no-call behavior, initial useful
    authorization, redundant-request rejection, retained three-request
    capacity, blocked-packet failure projection, and failed retry suppression.
39. Add red gates for required-selection activation, no-call behavior without
    that typed flag, selection-owner reversal rejection, correct-role
    preservation, and post-repair recheck.
40. Add a red gate proving drifted L3 prose cannot enter hard-error repair or
    become a fatal boundary issue, while typed current-role correction and all
    focused rechecks succeed.
41. Run focused deterministic tests, then affected non-live regression tests.
42. Run live LLM cases one at a time and inspect each result before the next:
    typed direct admission/linkage, recipient withdrawal, already-resolved,
    unavailable retained media, unexecuted physical request, and scheduled or
    pending work acknowledgement.
43. Re-run the frozen twenty-turn QQ group replay and frozen twenty-turn QQ
    private replay using the same baseline/V2 comparison format and preserved
    per-turn inputs.
44. Review V2 quality against Kazusa's personality, semantic role fidelity,
    continuity, vividness, surface-format integrity, and conflict-free
    creativity. Plain first-person in-character action wording is accepted.
    Baseline similarity is evidence, not the quality objective.
43. Run anti-cheat inspection: no replay-string branches, user-ID branches,
    test-only production paths, hardcoded semantic keyword filters, hidden
    baseline output reuse, or prompt-only output patching.
44. Perform the final code review in the parent agent, remediate findings,
    repeat affected gates, update docs and plan evidence, and commit the
    verified change.
45. Give every prepared frozen replay a unique run id, delete exact prior turn
    artifacts at preparation, validate predecessor run ids, and align generated
    test-database assistant rows to the captured causal clock before the next
    historical turn.
46. Split stage-format and unsupported execution-claim review from general
    semantic dialog compliance. Run both semantic owners for the initial
    candidate and again after the single bounded repair.
47. Continue a real-sequence corpus after content drift so all content failure
    families are recorded. Stop and fix immediately only for crashes, terminal
    contract failures, or failure-caused silence.

## Must Not Do

- Do not infer authoritative participation from user text in deterministic
  code.
- Do not force visible speech merely because an admitted turn exists; settled
  semantic judgment remains active.
- Do not mechanically suppress character engagement for ordinary group
  traffic.
- Do not collapse scheduled or pending work into executed completion.
- Do not add a compatibility shim, parallel relevance vocabulary, or fallback
  mapper outside the canonical big-bang contract.
- Do not enable the visual agent by default or make it a downstream input to
  text dialog.
- Do not tune against the frozen twenty-turn strings.

## Acceptance Criteria

1. Authoritative direct/private/broadcast input cannot be discarded at
   frontline, and exact linkage still uses no more than one LLM call.
2. Ordinary group traffic retains semantic discard capacity.
3. Authoritative settled relevance uses no model call for a sole disposition;
   multi-option settlement uses one semantic call and at most one bounded
   schema repair. Only a disposition made available by typed context can pass.
4. Recipient withdrawal can yield silence through semantic judgment.
5. Fresh-history resolution and retained-media unavailability are available
   only when their structural prerequisites exist.
6. Every action lifecycle status reaches L3 without `scheduled` or `pending`
   becoming completion.
7. Only an executed result authorizes a visible completed-effect claim.
8. Existing production action selection, routing, resolver, delayed-work,
   approval, and visual-default capacities pass their regression suites.
9. Focused live gates pass one at a time without technical failure or false
   capability execution; action descriptions remain valid visible roleplay.
10. The planner emits no route field; deterministic route derivation preserves
    visible speech, action-only, resolver, deferral, and silence capacities.
11. Every proposed executable action receives one focused semantic
    authorization decision, while zero-action plans add no model call.
12. A physical chat request cannot authorize background work, accepted coding,
    or any other current non-actuator capability; an explicitly accepted
    delayed task still authorizes its production capability.
13. Both frozen twenty-turn comparisons complete with per-turn evidence and a
    written quality review.
14. Anti-cheat inspection and independent code review report no unresolved
    blocking findings.
15. Frozen replay artifacts all carry the current prepared run id, and no
    earlier generated reply can appear temporally after a later source turn.
16. Initial and repaired dialog both pass focused enactment/execution review
    and general semantic compliance before visible delivery.
17. Resolver authorization adds no call when no resolver request is proposed,
    rejects a semantically redundant request after sufficient current
    evidence, and preserves up to three genuinely distinct evidence needs.
18. Blocked local-context packets are failed observations, and a failed or
    timed-out capability cannot be retried under a renamed objective.
19. A typed required selection invokes one focused role-direction decision;
    delegating that selection to the wrong role is rejected, the corrected
    repair is rechecked, and turns without the structural gate add no call.
20. Drifted free-text L3 content, boundary, and style prose cannot enter the
    hard-error repair or define a fatal verifier issue; typed current meaning
    and exact permitted-action results remain enforced.
21. A typed character-owned required selection is verified before downstream
    action and dialog planning. A rejected goal is regenerated from typed
    current evidence and bounded character context without the rejected bid or
    private-continuity prose, then rechecked by the same focused owner.
22. Optional action proposal and authorization schema failure cannot crash an
    otherwise valid speech turn or authorize work. Valid proposal rows remain
    usable up to the existing three-request cap; unusable proposal replacement
    yields no work, and unusable authorization replacement denies all work.

## Execution Checklist

- [x] System RCA and ownership audit recorded.
- [x] User approved implementation.
- [x] Focused deterministic tests added and observed failing.
- [x] One production-code subagent assigned.
- [x] Constrained relevance contracts implemented.
- [x] Exact action lifecycle surface contract implemented.
- [x] Focused deterministic tests pass.
- [x] Affected non-live regression tests pass.
- [x] Route-free planning and focused action authorization implemented.
- [x] Focused live LLM gates passed one at a time in the preserved prior proof.
- [x] Prior frozen QQ group twenty-turn proof evidence preserved.
- [x] Prior frozen QQ private twenty-turn proof evidence preserved.
- [x] Anti-cheat inspection completed.
- [x] Parent-agent review completed and findings remediated.
- [x] Documentation, plan evidence, and scoped commit completed; hash is
  recorded in the handoff.
- [x] Fresh post-gate frozen 20+20 replay; completed after the user released
  the pre-E2E hold, with the resulting corpus carried into final Stage 2
  review.

## Execution Evidence

- Red gate: the five focused files produced 15 intended failures and 53
  passes before production edits.
- Post-fix focused and affected integration gate: 226 passed in 39.40 seconds.
- Production implementation owner:
  `/root/responsibility_allocation_implementation`.
- Parent remediation preserved external reported/observed event grounding while
  limiting the execution ledger to actions performed by the character brain.
- The first unexecuted-physical-request live gate failed: the planner selected
  `background_work_request` for a physical-action description despite prompt
  prohibitions. This is the red live evidence for separating proposal,
  semantic authorization, and deterministic route ownership.
- Fresh group replay run `9257b5becd481cd2b632` reached turn 11 before the
  strict non-silence gate stopped it. The protected trace recorded
  `already_resolved` against a turn-3 breakfast reply. The source request was
  captured on 2026-07-16 while that generated reply was persisted on
  2026-07-17, proving the mixed-clock failure. Artifact:
  `test_artifacts/cognition_core_v2/turn11_settled_silence_full_trace.json`.
- Historical replay clock regression gate: 2 passed.
- Frozen group proof completed under run `013b6fc3a38971979fb6`: 20 turns, 30
  visible fragments, 155 trace steps, and exact historical-row alignment.
- Frozen private proof completed under run `3324a92a43bcdf71081d`: 20 turns, 32
  visible fragments, 154 trace steps, and exact historical-row alignment.
- Private proof failure distribution: embodied/bodily narration and
  unsupported enactment claims recur across the explicit physical sequence;
  scope broadening or contradiction appears separately; repetitive residue
  amplifies submission and weakened agency but turn 18 demonstrates that a
  strong current-mode change can still redirect the dialog. The enactment
  boundary is in this bugfix. Residue quality remains separately observable
  and must not be conflated with action routing.
- User-directed run policy recorded on 2026-07-17: fatal runtime failure or
  failure-caused silence stops for immediate repair; content drift is logged
  while the sequence continues for systematic coverage.
- Fresh group proof run `2624f73fdc4eb0143c9d` reached turn 10. The protected
  trace contained five goal calls, four action-planning calls, and three
  `local_context_recall` executions; the turn completed in 290.856 seconds.
  The first recall projected eight rows. The following two local resolver
  calls returned blocked packets after invalid JSON, but both were projected
  as successful zero-row observations. This is the red live evidence for
  separating resolver proposal and authorization and for preserving failed
  capability truth through recurrence.
- Fresh group proof run `d11c86fb13fee650ce2d` completed 20/20 turns with one
  run id, zero operational errors, zero missing dialog or traces, and zero
  failed stages. Turn 10 completed in 61.160 seconds with no resolver proposal,
  compared with 290.856 seconds and three recalls in the failed run. Across the
  corpus there was one useful authorized recall and no repeated recall.
- Fresh private proof run `d218612f9305bd3b4906` stopped at turn 1 under the
  unacceptable-error policy. Decontextualization correctly emitted
  `selection_required=true`, `selection_owner_role=self`,
  `embedded_actor_role=current_user`, and `embedded_target_role=self`; goal
  cognition reversed that selection and the general semantic verifier returned
  aligned. This is the red live evidence for a focused typed role-direction
  owner.
- Fresh private retry run `c04a380ebf8290ee3549` stopped at turn 1 after the
  focused role and general semantic verifiers correctly rejected the initial
  selection reversal. Drifted `active_visible_boundaries` and
  `style_guidance` supplied the opposite role instruction to repair, which
  repeated the reversal and failed recheck. This is the red live evidence for
  removing drift-prone L3 prose from hard-error ownership.
- Exact production-language repair gate passed after that ownership change.
  It repaired the stopped input to a concrete self-selected next action,
  including `我想让你现在就吻过来`, and semantic, role-direction, and surface
  rechecks all returned aligned. Artifact:
  `test_artifacts/llm_traces/dialog_visible_speech_and_semantic_fidelity__`
  `focused_repair_stopped_private_role_reversal__20260717T191510073599Z.json`.
- The goal-level required-selection owner and optional action-plan containment
  were exercised through the production boundary in preserved group run
  `a5c433fa6e2ac7935ec0` and private run `91a05cf7c3118c9cc7cb`.
  Both completed 20/20. The readable turn-level evidence is
  `test_artifacts/cognition_core_v2/`
  `live_character_judgment_40_turn_review.md`; corpus-level quality judgment
  remains open for user review.
- Final focused deterministic/adjacent gate: 115 passed and four live cases
  deselected. The final full non-live gate passed 3,250, skipped two, and
  deselected 735 live/external tests in 234.20 seconds. The only warning was
  the existing Starlette/httpx deprecation warning.
- Compilation of the affected cognition, resolver, node, and focused test
  modules passed. `git diff --check` and the prompt anti-cheat contract checks
  passed.
- The fresh 20+20 replay was intentionally not started after these pre-E2E
  gates. Port 8012 is clear, the guarded replay database remains restored from
  the prior proof cleanup, and no replay service or test worker is running.

## Lifecycle Closure

The later user-authorized sequential 20+20 run superseded the recorded
pre-E2E hold. All 40 turns completed, traces were consolidated into
`test_artifacts/cognition_core_v2/fresh_40_turn_signoff/`
`cognition_v2_fresh_40_turn_monologue_dialog_review.md`, and the user accepted
the Stage 2 quality on 2026-07-18.
