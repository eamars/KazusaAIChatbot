# Cognition Core V2 Frozen Replay Drift Bugfix Plan

## Summary

- Status: completed.
- Change class: large.
- Cutover: bigbang.
- Approval: the user explicitly instructed execution on 2026-07-16.
- Parent plan: `development_plans/active/short_term/cognition_core_v2_stage_2_integration_plan.md`.
- RCA evidence: `test_artifacts/cognition_core_v2/frozen_affinity_rca/qq_638473184_state_drift_rca_and_proposal.md`.
- Proof corpus: the frozen 20-turn QQ 638473184 / user 673225019 replay captured at each original input boundary.

This bugfix closes the production-path defects exposed by the frozen affinity
comparison. It preserves the user-approved V2 direction while removing state,
continuity, monologue, voice, and final-render drift. Action-description policy
is outside this plan because the user identified it as a separate issue.

## Context

The frozen replay showed that the new dialog was generally more vivid, caring,
and human, but later turns inherited false breakfast/game obligations, a
permanent relationship-connection goal, role inversions, and incomplete content
rendering. Turn 20 also failed when one appraisal batch resolved the same event
twice and evidence retention exceeded the state schema cap. Post-turn
consolidation still depended on the removed legacy `rag_result.user_image`
shape. The V2 connector exposed a bid justification as both monologue and
interaction subtext, did not consume the loaded residue window, and the final
dialog prompt discarded the character profile values supplied by its caller.

The ownership boundaries remain:

```text
current episode evidence + typed continuity -> cognition appraisal and goals
selected bid + character voice -> L3 semantic surface -> dialog renderer
completed private monologue + visible outcome -> residue recorder
canonical V2 RAG memory candidates -> consolidation
```

## Mandatory Skills

- `development-plan`
- `local-llm-architecture`
- `no-prepost-user-input`
- `py-style`
- `test-style-and-execution`
- `character-test`
- `debug-llm`
- `cjk-safety` for Python prompt edits

## Mandatory Rules

- The parent agent owns the plan, focused failing tests, verification, proof
  artifact, database/service restoration, and sign-off.
- Exactly one production-code subagent implements production changes after the
  focused tests fail for the intended reasons.
- A separate review subagent performs the post-verification code review and
  makes no production edits.
- LLM stages own semantic judgments. Deterministic code owns exact schemas,
  provenance, evidence caps, lifecycle transitions, and retry bounds.
- User text and generated dialog receive no keyword classifiers or semantic
  rewriting in deterministic code.
- Prompt changes use static triple-single-quoted system messages and dynamic
  JSON in `HumanMessage`.
- Live LLM cases run one at a time and each result is inspected.
- The frozen proof reuses the exact original input, timestamp, memory, user
  image, relationship, character profile, and prior-turn boundary for every
  turn.
- Database and service state changed by the proof is guarded, restored, and
  verified.

## Must Do

1. Make repeated terminal meanings idempotent only within the current appraisal
   batch, while preserving terminal immutability across later batches.
2. Retain bounded evidence deterministically: relationship state keeps the
   newest eight unique rows; causal entities keep their first/root row and the
   newest seven unique rows.
3. Create `relationship_connection` only from an active closeness gap, and
   satisfy an existing stale connection goal once the typed closeness gap is
   closed.
4. Add structured interaction obligations with explicit actor, action,
   beneficiary, precondition, expected outcome, lifecycle status, and source
   kind. Keep ordinary topical open loops semantically distinct.
5. Project the current event and conversation continuity into separate V2 scene
   fields, preserving structured obligation roles in the continuity projection.
6. Feed the bounded residue window only to goal-cognition branches, separate
   selected-bid reasoning from first-person private monologue, and record residue
   against bounded visible outcome and surface-boundary context.
7. Project a bounded canonical character voice context through L3 and into the
   final renderer.
8. Give the final renderer a semantic LLM compliance check for content-plan
   preservation, actor/action/target direction, and visible boundaries, with at
   most one repair call.
9. Remove consolidation dependence on `rag_result.user_image`; use the canonical
   `user_memory_unit_candidates` V2 shape in both dedup and prompt projection.
10. Give malformed goal-bid JSON one bounded LLM-owned schema repair, preserve
    initial and repaired outputs in protected traces, and fail when a required
    cognition branch remains invalid rather than collapsing it into silence.
11. Protect replay backups from overwrite until exact restoration verification
    passes; a failed replay turn triggers guarded cleanup.
12. Prove the fix with deterministic regressions, focused live LLM checks, and a
    fresh sequential run of all 20 frozen turns.

## Deferred

- Prohibiting action descriptions in visible dialog.
- Retuning the acceptable reduction in tsundere intensity.
- Recreating a literal legacy 1000-point affinity field in V2.
- Stage 3 auxiliary-console adoption.
- Any production deployment outside the guarded local proof workflow.

## Cutover Policy

This is a bigbang contract update. All in-repo producers, consumers, validators,
tests, and subsystem documentation move to the canonical shapes in one change.
There are no aliases, fallback mappers, dual output fields, or legacy
`rag_result.user_image` reads in the changed consolidation path. Existing stored
conversation-progress documents may lack the new optional obligation list; the
canonical loader projects that absence as an empty list without rewriting or
duplicating stored vocabulary.

## Target State

- One appraisal batch may reinforce a terminal result it just produced without
  attempting a second FSM transition.
- Evidence growth remains valid for arbitrarily long conversations.
- High attachment and care influence importance without manufacturing a
  permanent connection deficit.
- Generated surface suggestions never become user-owned obligations unless the
  progress recorder explicitly judges the actor and source that way.
- Current input remains authoritative and continuity remains labeled supporting
  context.
- `private_monologue` is first-person private cognition; `selected_bid_reason`
  is branch-selection rationale; interaction subtext uses the latter.
- L3 and dialog share a bounded character voice contract.
- Dialog omissions and subject swaps receive one LLM-owned repair opportunity.
- Explicitly fulfilled obligations resolve; explicit replacements retain the
  old row as superseded and the new row as active for the transition turn.
- Consolidation consumes required canonical V2 memory candidates and fails at
  the boundary when that field is malformed or absent.

## Design Decisions

1. `SceneContextV2` gains required `conversation_continuity`; `semantic_scene`
   returns to current-event ownership.
2. `CognitionCoreInputV2` gains required bounded
   `private_continuity_context`, projected only into goal-cognition branch
   payloads.
3. `GoalBidDraftV2` and `ActionBidV2` gain `private_monologue`.
   `CognitionCoreOutputV2` replaces ambiguous `residue` with
   `selected_bid_reason` and `private_monologue`.
4. `ConversationProgressPromptDoc` and stored episode state gain
   `interaction_obligations`. The recorder LLM emits exact structured rows;
   deterministic code validates enums, lengths, timestamps, and list caps only.
5. `TextSurfaceInputV2` and `TextSurfaceOutputV2` gain bounded
   `character_voice_context`; the value is a deterministic projection of the
   active character profile and remains wording-only context.
6. Dialog compliance is a semantic verifier LLM call after initial rendering.
   A negative verdict supplies bounded issues to one repair call. No loop is
   permitted.
7. The V2 RAG candidate list is the sole consolidation memory-context input.
8. Goal cognition retains strict route/capability validation. Its prompt uses an
   exact route-field matrix; invalid model JSON receives at most one semantic
   schema-repair call, and a still-failed required branch raises execution error.
9. Protected turn traces store goal initial/repair and dialog
   generator/verifier/repair model boundaries under the same trace ID.
10. Replay preparation writes a digest guard before mutation, refuses to
    overwrite an unverified backup, restores on failed turn commands, and
    compares the complete touched scope after cleanup.

## Contracts And Data Shapes

Structured obligation row:

```text
{
  actor: string,
  action: string,
  beneficiary: string,
  precondition: string,
  expected_outcome: string,
  status: active | resolved | superseded,
  source_kind: user_input | assistant_response | mutual_exchange,
  first_seen_at/age_hint: boundary-owned timestamp projection
}
```

The recorder output omits `first_seen_at` and `age_hint`; persistence and prompt
projection own those fields respectively. Empty semantic fields are allowed
only for beneficiary, precondition, and expected outcome. Actor and action are
required. At most six obligation rows are stored and projected.

The dialog verifier returns exactly:

```text
{aligned: boolean, issues: [bounded semantic issue strings]}
```

It judges content coverage, actor/action/target direction, conditions, and
visible boundaries. Deterministic code checks shape and bounds, then performs
zero or one repair call.

## Drift Closure Matrix

| Observed drift | Root cause | Closure | Proof |
|---|---|---|---|
| Turn 20 terminal transition crash | duplicate terminal meanings in one batch | batch-local terminalization set | focused deterministic test + turn 20 |
| Evidence cap crash | append-only retention | typed bounded retention | long-sequence deterministic test |
| Permanent connection pursuit | attachment/care used as deficit | closeness-gap creation/completion | relationship lifecycle test |
| Breakfast/game carry-over | free-form roleless open loops | typed obligations + separate continuity | recorder live tests + turns 9/13/17/18 |
| Subject swap | final renderer lacked semantic verification | verifier + one repair | focused mocked repair + live turns 11/12 |
| Monologue mismatch | bid reason reused as monologue and residue ignored | split contract + L2-only continuity | contract tests + proof artifact |
| Weak/static voice | profile values discarded | canonical voice projection | prompt contract + live proof |
| Content beat omission | renderer had no compliance gate | semantic compliance verifier | focused live check + turn 20 |
| Consolidation KeyError | legacy `user_image` read | canonical candidates | deterministic consolidation test + proof |

## LLM Call And Context Budget

- Appraisal, collapse, action selection, and four L3 calls remain unchanged in
  count.
- Goal-cognition context gains at most 1,000 characters of private continuity.
- Goal cognition adds zero calls for valid output and at most one schema-repair
  call for malformed output; it has no retry loop.
- L3 context gains at most 1,500 characters of character voice.
- Dialog adds one semantic verifier call per visible response and at most one
  repair call after a negative verdict.
- Verifier input is capped to the validated text-surface output and generated
  dialog; it receives no persistent state or raw RAG packet.
- Conversation progress and residue recorder call counts remain unchanged.

## Change Surface

### Delete

- Ambiguous `CognitionCoreOutputV2.residue` field and its dual projection.
- Legacy consolidation reads of `rag_result.user_image.user_memory_context`.
- Dead dialog prompt `.format(...)` arguments that never entered the prompt.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/{contracts,facade,goal_cognition,state_reducers,transition_guards,surface,surface_stages,validation_cli}.py`
- `src/kazusa_ai_chatbot/nodes/{persona_supervisor2_cognition,persona_supervisor2_l3_surface,dialog_agent}.py`
- `src/kazusa_ai_chatbot/conversation_progress/{models,policy,recorder,projection,repository,README}.py|md`
- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/internal_monologue_residue/{models,recorder,README}.py|md`
- `src/kazusa_ai_chatbot/consolidation/{core,memory_units,README}.py|md`
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
- Directly affected deterministic and live-LLM tests.
- Frozen replay experiment and final review artifact where proof capture needs
  the new explicit monologue/reason fields.

### Create

- Focused regression tests only where an existing test module has no suitable
  ownership boundary.
- Fresh post-fix proof output under
  `test_artifacts/cognition_core_v2/frozen_affinity_fix_proof/`.

### Keep

- Canonical cognitive episode shape.
- RAG evidence ownership and final-dialog wording ownership.
- Existing state FSM terminal immutability across separate episodes.
- Existing database collections and guarded replacement writes.
- The original frozen replay inputs and baseline artifacts unchanged.

## Overdesign Guardrail

- No new service, graph, database collection, compatibility layer, or generic
  obligation framework.
- One structured obligation row is the minimum shape that preserves the roles
  implicated by the failures.
- One verifier verdict and one bounded repair are the complete dialog loop.
- Character voice is one bounded projection, not a second persona engine.
- Evidence retention is one shared deterministic policy, not entity-specific
  storage infrastructure.

## Agent Autonomy Boundaries

- Production subagent: may edit only the production files named above and
  subsystem READMEs; must not edit tests, plan status, proof artifacts, database
  data, or service state.
- Parent agent: writes tests first, reviews all edits, runs verification and live
  proof, updates documentation/evidence, and restores guarded state.
- Review subagent: read-only review after parent verification; reports findings
  to the parent and makes no edits.
- Any change to frozen input data, any semantic retry beyond the approved single
  goal-bid repair and single dialog repair, or action-policy scope requires
  fresh user approval.

## Implementation Order

1. Parent writes focused failing tests for every RCA boundary and confirms the
   expected failures.
2. Production subagent implements state idempotency, evidence retention, and
   relationship-goal lifecycle.
3. Production subagent implements structured obligations and separated scene
   continuity.
4. Production subagent implements private monologue/residue contracts.
5. Production subagent implements character voice and dialog compliance repair.
6. Production subagent updates canonical consolidation consumption and subsystem
   documentation.
7. Parent updates impacted tests, runs syntax/static/focused/full verification,
   and runs live LLM cases one at a time.
8. Independent review subagent audits the verified diff; parent remediates the
   accepted prompt/repair, voice projection, trace, obligation lifecycle,
   consolidation fail-fast, and replay-restoration findings, then reruns the
   affected gates.
9. Parent guards current database/service state, runs the exact frozen 20-turn
   sequence, inspects every turn, restores state, and writes the review document.
10. Parent completes evidence and archives the plan only after every acceptance
    criterion passes.

## Execution Model

- Parent architect/test owner: `/root`.
- Production implementation owner: exactly one delegated subagent after red
  tests.
- Independent code reviewer: exactly one different delegated subagent after
  verification.
- Checkpoints are sequential because later contracts consume earlier outputs.
- After each signed major checkpoint, the parent rereads this complete plan.

## Progress Checklist

- [x] A. RCA and frozen evidence accepted as execution input.
  - Sign-off: `Codex parent architect / 2026-07-16`; user explicitly instructed execution and exact 20-turn proof.
- [x] B. Focused regression tests fail for the intended pre-fix reasons.
  - Sign-off: `Codex parent test owner / 2026-07-16`; `venv\Scripts\python -m pytest tests/test_cognition_core_v2_frozen_replay_drift.py -q` produced 10 intended failures, one at each RCA boundary, before production edits.
- [x] C. State and evidence lifecycle fixes implemented and focused tests pass.
  - Sign-off: `Codex parent architect / 2026-07-16`; terminal idempotency, typed evidence identity, eight-row retention, and closeness-gap lifecycle regressions pass.
- [x] D. Continuity and private-monologue contracts implemented and focused tests pass.
  - Sign-off: `Codex parent architect / 2026-07-16`; structured obligations, separated current/public/private continuity, bid reason, first-person monologue, and visible-outcome residue regressions pass.
- [x] E. Voice, dialog compliance, and consolidation fixes implemented and focused tests pass.
  - Sign-off: `Codex parent architect / 2026-07-16`; voice projection, content requirements, one-repair verifier, and canonical memory-candidate tests pass.
- [x] F. Scoped and full non-live regression pass.
  - Sign-off: `Codex parent test owner / 2026-07-17`; focused ownership suites passed and the final full default non-live run completed with 3,125 passed, 2 skipped, 640 deselected.
- [x] G. One-at-a-time focused live LLM gates pass and outputs are inspected.
  - Sign-off: `Codex parent live-test owner / 2026-07-17`; four individually executed live cases passed and their fresh raw traces were inspected.
- [x] H. Independent code review is resolved.
  - Sign-off: `Codex parent architect + read-only review agent / 2026-07-17`; all seven findings and both follow-up restoration findings were remediated, and the reviewer reported no remaining concrete blocker.
- [x] I. Fresh frozen 20-turn sequential proof completes with guarded restoration.
  - Sign-off: `Codex parent live-test owner / 2026-07-17`; all 20 exact frozen inputs completed sequentially with 120 protected trace steps, the dedicated PID was stopped, port 8011 was clear, and the guarded database snapshot restored exactly.
- [x] J. Review artifact, execution evidence, documentation, and archive closeout complete.
  - Sign-off: `Codex parent architect / 2026-07-17`; the 20-turn comparison review records new private monologue, residual, dialog, emotions/state, old monologue/dialog, and per-turn analysis, including five partial results and one direct-task failure rather than masking residual quality findings.

## Verification

### Focused and integration commands

- Run the directly affected cognition V2, conversation progress, residue,
  dialog, L3, consolidation, and replay-harness tests with
  `venv\Scripts\python -m pytest`.
- Run every changed test module in complete-file form after focused cases pass.

### Real LLM

- Run recorder actor/source tests one case at a time.
- Run private-monologue continuity and dialog verifier/repair cases one at a
  time.
- Store raw request/response/trace material and a readable evaluation artifact.

### Database

- Use the configured project runtime without reading `.env`.
- Snapshot all exact user/character/progress/residue rows and service state
  touched by the replay.
- Restore snapshots in `finally` paths and compare post-restore state.

### Static and full regression

- `venv\Scripts\python -m compileall` for changed Python packages.
- Project static prompt scans relevant to modified LLM paths.
- Full non-live pytest suite after scoped gates.
- `git diff --check` and a final `git status --short` audit.

### Production boundary

- Proof uses the closest production workflow available locally, including real
  configured LLM routes and MongoDB-backed memory/state loading.
- The harness captures, but does not deliver, outbound QQ messages.

## Independent Plan Review

The plan was reconciled against the Stage 2 contract, the RCA artifact, current
source, subsystem READMEs, and the frozen replay harness on 2026-07-16. It is
approved by explicit user execution instruction. There are no unresolved design
questions.

## Independent Code Review

The read-only reviewer reported seven findings after Checkpoint G: the live
goal-bid route/field mismatch and silent required-branch drop, raw numeric voice
projection, unverified replay restoration, missing repair trace steps,
under-specified obligation lifecycle transitions, two stale live `residue`
consumers, and optional handling of required consolidation candidates. The
parent remediated every finding with focused tests. A follow-up review required
guard validation before cleanup mutation and scoped trace-run/step restoration;
both were implemented and proven with a fresh prepare/cleanup drill. The final
reviewer confirmation reported no remaining blocker.

## Acceptance Criteria

- The focused terminal-event regression and original turn 20 both complete.
- No cognition state entity exceeds eight evidence refs after long sequential
  updates.
- High affinity without a closeness gap leaves no active
  `relationship_connection` goal.
- Structured obligations preserve who acts, for whom, under what condition, and
  whether the source was user, assistant, or mutual.
- Breakfast/game/state drift is absent unless grounded by the current frozen
  input or active typed obligation.
- Private monologue is first-person and semantically distinguishable from the
  bid selection reason.
- Residue continuity affects L2 goal cognition and remains absent from L1/L3
  payloads.
- Character voice reaches L3 and dialog through an inspectable bounded field.
- Dialog semantic verifier catches the seeded actor-direction failure and one
  repair produces aligned output.
- A malformed goal-bid capability field receives at most one traced LLM repair;
  a still-failed required branch cannot become semantic silence.
- Consolidation completes with a V2 RAG result that has candidate rows and no
  `user_image` field.
- All 20 frozen inputs complete in order with new monologue, new dialog, new
  state/emotions, old monologue/dialog, and per-turn analysis in the review
  document.
- Guarded database/service restoration is verified.
- Full non-live regression and independent code review pass.

## Risks

- The dialog verifier and exceptional goal-bid repair increase latency and
  model cost; their single-repair bounds prevent open loops.
- A weaker local model may over-report dialog alignment; seeded live cases and
  the frozen proof assess this directly.
- Structured obligation prompts may initially be too permissive; exact schema,
  source labels, and one-at-a-time live inspection constrain that risk.
- Bigbang output-field replacement affects diagnostics and tests; full contract
  search and regression are required.
- Replay writes can contaminate production-like state; snapshot/restore is a
  hard acceptance gate.

## Approval Boundary

This plan is authorized for local production-code implementation, tests, real
configured LLM calls, guarded database-backed replay, and review-artifact
creation. Deployment, QQ delivery, action-description policy changes, and
unrelated character retuning remain outside the authorization.

## Execution Evidence

- Red gate: `venv\Scripts\python -m pytest tests/test_cognition_core_v2_frozen_replay_drift.py -q` produced 10 intended pre-fix failures.
- Focused post-fix gate: `tests/test_cognition_core_v2_frozen_replay_drift.py` passed all 10 cases.
- Core/connector focused batch: 29 passed.
- Conversation-progress/residue focused batch: 41 passed.
- Dialog/L3/consolidation/replay-regression focused batch: 33 passed.
- Full non-live gate: `venv\Scripts\python -m pytest -q --tb=short -r fE` completed with 3,125 passed, 2 skipped, 640 deselected, and one unrelated Starlette deprecation warning.
- Static gates: changed Python packages compile; `git diff --check` passed; canonical contract scans found no stale V2 test fixture shapes or legacy consolidation `user_image` reads.
- Parent review corrected a dialog telemetry scoping defect and changed evidence identity from `source_id` alone to `(source_kind, source_id)` after full-suite evidence preservation tests exposed the collision.
- Live recorder obligation gate: `test_live_recorder_returns_string_items_for_progress_lists` passed; the emitted obligation preserved actor Kazusa, user beneficiary, fulfilled exchange precondition, active lifecycle, and `mutual_exchange` source. Trace: `test_artifacts/llm_traces/conversation_progress_recorder_identity_live_llm__returns_string_items_for_progress_lists.json`.
- Live L3 gate: `test_v2_text_surface_stage_contracts_live_llm` passed; the fresh timestamped trace includes voice context in all four prompts and four explicit content requirements.
- Live dialog gate: `test_live_dialog_generator_node_accepts_deepseek_output` passed; final dialog preserved the classification action and object under the semantic verifier.
- Live private-monologue gate: attachment case passed; capture `test_artifacts/cognition_core_v2/raw/love_attachment_1784203303141667100.json` shows first-person private monologue distinct from selected-bid rationale.
- Independent review surfaced seven findings. Focused remediation gates passed
  for the goal-bid route matrix/one repair/required failure, semantic voice
  descriptors, protected repair traces, stale live consumers, obligation
  lifecycle, canonical consolidation fail-fast, and replay backup guards.
- Live obligation resolution gate passed. Trace:
  `test_artifacts/llm_traces/conversation_progress_recorder_identity_live_llm__obligation_active_to_resolved.json`.
- Live obligation supersession gate passed after prompt-example hardening. Fresh
  trace:
  `test_artifacts/llm_traces/conversation_progress_recorder_identity_live_llm__obligation_active_to_superseded__20260716T122627995998Z.json`.
- Replay cleanup exact-state comparison passed and wrote a `restored` digest
  guard under the post-fix proof root.
- Final independent confirmation closed Checkpoint H with no remaining blocker.
- The fresh sequential proof completed all 20 turns with 120 protected trace
  steps. Review artifact:
  `test_artifacts/cognition_core_v2/frozen_affinity_fix_proof/qq_638473184_post_fix_20_turn_proof_review.md`.
- The dedicated proof service PID 14100 was stopped, port 8011 was clear, and
  exact database restoration passed with guard status `restored` and backup
  SHA-256 `dfb9e43b99ddf0ef6f9ca92e6e823513f82fc5d11a361a8f1c8348d273badc5e`.
- The post-proof focused drift gate passed all 10 cases; final `git diff
  --check` reported no content errors.
