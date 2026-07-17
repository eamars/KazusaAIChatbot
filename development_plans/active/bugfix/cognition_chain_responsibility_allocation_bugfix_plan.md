# Cognition Chain Responsibility Allocation Bugfix Plan

## Summary

- Status: in_progress.
- Change class: large.
- Cutover: bigbang.
- Approval: the user explicitly instructed implementation on 2026-07-17.
- Parent plan:
  `development_plans/active/short_term/cognition_core_v2_stage_2_integration_plan.md`.
- Related plan:
  `development_plans/active/bugfix/cognition_core_v2_compositional_action_planning_bugfix_plan.md`.
- Frozen proof corpora: the existing twenty-turn QQ group `638473184`
  replay and twenty-turn QQ private-user `673225019` replay, captured at each
  original input boundary.

This bugfix replaces two prompt-repair loops with typed ownership contracts.
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
action-narration detection in one prompt. It accepted real-time embodied
enactment and bodily narration in literal speech. When it does request repair,
the repaired output is delivered without any second verification. This is a
task-allocation defect rather than evidence that deterministic user-text
classification is needed.

The system correction assigns real-time enactment, bodily-performance
narration, and unsupported execution claims to one focused semantic verifier.
It runs alongside the general semantic-fidelity verifier. Their issue sets are
combined for at most one grounded renderer repair, and repaired output must
pass both owners before delivery. Deterministic code only validates verdict
shape, combines exact issue lists, and blocks an unverified repair; it does not
classify user or dialog text.

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

The parent agent owns this plan, failing tests, documentation, verification,
frozen replay control, remediation, and sign-off. Exactly one production-code
subagent implements production files after focused tests fail. Exactly one
separate review-only subagent reviews the verified implementation before
closeout.

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
14. Reject model outputs that choose an unavailable disposition; fail closed
    through the existing relevance error path.

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
26. Keep resolver choice semantic in the planner and keep all action-spec
    permission, validation, execution, and persistence owners unchanged.

### E. Verification and proof

27. Add deterministic regression tests for every constrained action space,
    unavailable disposition, single-call bound, and exact lifecycle status.
28. Add red gates for route-free planning, output-mode route derivation,
    semantic authorization approval/rejection, no-call behavior for empty
    action proposals, and retained three-action capacity.
29. Run focused deterministic tests, then affected non-live regression tests.
30. Run live LLM cases one at a time and inspect each result before the next:
    typed direct admission/linkage, recipient withdrawal, already-resolved,
    unavailable retained media, unexecuted physical request, and scheduled or
    pending work acknowledgement.
31. Re-run the frozen twenty-turn QQ group replay and frozen twenty-turn QQ
    private replay using the same baseline/V2 comparison format and preserved
    per-turn inputs.
32. Review V2 quality against Kazusa's personality, semantic role fidelity,
    continuity, vividness, action-narration prohibition, and conflict-free
    creativity. Baseline similarity is evidence, not the quality objective.
33. Run anti-cheat inspection: no replay-string branches, user-ID branches,
    test-only production paths, hardcoded semantic keyword filters, hidden
    baseline output reuse, or prompt-only output patching.
34. Run the independent review subagent, remediate findings, repeat affected
    gates, update docs and plan evidence, and commit the verified change.
35. Give every prepared frozen replay a unique run id, delete exact prior turn
    artifacts at preparation, validate predecessor run ids, and align generated
    test-database assistant rows to the captured causal clock before the next
    historical turn.
36. Split real-time embodied enactment and unsupported execution-claim review
    from general semantic dialog compliance. Run both semantic owners for the
    initial candidate and again after the single bounded repair.
37. Continue a real-sequence corpus after content drift so all content failure
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
3. Authoritative settled relevance uses one call and only a disposition made
   available by typed context.
4. Recipient withdrawal can yield silence through semantic judgment.
5. Fresh-history resolution and retained-media unavailability are available
   only when their structural prerequisites exist.
6. Every action lifecycle status reaches L3 without `scheduled` or `pending`
   becoming completion.
7. Only an executed result authorizes a visible completed-effect claim.
8. Existing production action selection, routing, resolver, delayed-work,
   approval, and visual-default capacities pass their regression suites.
9. Focused live gates pass one at a time without technical failure, false
   action completion, or prohibited action narration.
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
- [ ] Focused live LLM gates pass one at a time.
- [ ] Frozen QQ group twenty-turn proof completed.
- [ ] Frozen QQ private twenty-turn proof completed.
- [ ] Anti-cheat inspection completed.
- [ ] Independent review completed and findings remediated.
- [ ] Documentation, plan evidence, and commit completed.

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
