# Asuna real E2E 50-turn conversation practice

## Summary

- Goal: conduct one fresh, natural, continuous conversation of at least 50
  user/Asuna exchanges through the real debug adapter and brain service, inspect
  every result, record observed quality issues, and derive a next-step
  optimization plan from current-run evidence only.
- Status: completed
- Completion date: 2026-08-24
- Scope boundary: live debug transport, the running brain service and production
  database configured for that service, protected/runtime log evidence, and
  evaluation artifacts under one new test-run directory.
- Change direction: observe and diagnose current behavior without modifying
  production code, prompts, character data, or existing conversation records.
- Acceptance state: all acceptance criteria are verified. The practice record
  is ready for archival as immutable historical evidence.

## Confirmed Decisions

- The target character is the active Asuna identity loaded by the running brain.
- The conversation uses a new stable debug platform user and private channel so
  no earlier dialog history shapes the session.
- `listen_only=false`, `think_only=false`, and `no_remember=false` remain fixed
  for all turns so the run exercises real visible response, continuity, and
  persistence behavior.
- Each next user message is written only after inspecting Asuna's latest response
  and the fresh current-turn evidence.
- Conversation messages contain ordinary human dialog and no test, architecture,
  prompt, trace, memory-system, or evaluation framing.
- A diagnostic subagent is required as soon as a turn shows a non-answer,
  repeated answer, role inversion, or loss of visible conversation context.

## Scope And Change Direction

The conversation operator starts or verifies the debug adapter, uses one fresh
identity for the complete run, sends exactly one message at a time, saves the raw
response and fresh log slice, evaluates the turn, and authors the next natural
message from Asuna's response. The evaluation observes current behavior as a
character brain: relevance, directness, continuity, role ownership, novelty,
tone, and system health.

The work produces evidence and plans. It does not implement an optimization.

## Mandatory Skills

- `character-test`: governs live identity, per-turn adaptation, response/log
  capture, evaluation, and consolidation.
- `debug-llm`: governs the human-readable real-output review artifact.
- `development-plan`: governs this execution record and the resulting
  optimization plan.
- `local-llm-architecture`: governs diagnosis boundaries and proposed
  optimization blast radius.
- `test-style-and-execution`: governs the one-case-at-a-time live-LLM evidence
  inspection posture.

## Mandatory Rules

- Preserve all pre-existing worktree changes.
- Keep all conversation turns ordinary and responsive to the immediately
  preceding Asuna message.
- Present the exact next message and observation target before sending it.
- Save each raw response and the current-turn log evidence before planning the
  next message.
- Treat valid transport and schema output as harness evidence rather than proof
  of conversation quality.
- Keep the diagnostic subagent read-only and evidence-grounded.
- Keep production code, prompts, tests, database maintenance, and character
  profile mutation outside this execution.

## Must Do

1. Create one timestamped run directory under `test_artifacts/debug_runs/` and
   record stable request identity, service health, process provenance, and run
   start time.
2. Send at least 50 continuous private debug messages with persistence enabled.
3. After every turn, retain raw response JSON, current-turn process-log evidence,
   and an agent-authored per-turn conclusion.
4. Evaluate direct response, repetition, role ownership, recent-context
   continuity, naturalness, and any operational or background failure.
5. Invoke a diagnostic subagent on the first qualifying failure and give it the
   exact turn artifacts plus read-only source ownership scope.
6. Produce one human-readable LLM debug review from real inputs and outputs.
7. Consolidate all observed issues, strengths, learning, uncertainty, and
   evidence-backed engineering priorities.
8. Produce a separate next-step optimization development plan whose scope is
   derived only from this run.

## Deferred

- Production-code, prompt, schema, model-route, database, or character-profile
  changes.
- Replaying, reading, or using earlier conversation transcripts to choose this
  session's dialog.
- Stress testing, adversarial prompting, romance/sexual escalation, task-tool
  benchmarking, or scripted recall quizzes unless Asuna naturally brings the
  conversation there.
- Claiming a systemic defect from one ambiguous response without supporting
  trace or repeated evidence.

## Target State

The completed run contains at least 50 ordered user/Asuna pairs under one debug
scope. Every pair is replayable from raw evidence and has a concise human
judgment. The final review separates user-visible quality, model-stage evidence,
deterministic validation, and uncertainty. Any optimization proposal names the
smallest semantic owner and keeps unrelated stages outside its change surface.

## Execution Roles

### Conversation operator and evaluation owner

- Responsibility: operate the complete adaptive conversation and author the
  evidence-grounded review and plans.
- Owned surface: the new run directory, this plan, the registry row for this
  plan, and the later optimization-plan artifact and registry row.
- Authority: call the local debug interface and live brain, create diagnostic
  artifacts, and update plan lifecycle evidence; no production mutation.
- Applicable skills: all skills listed in `Mandatory Skills`.
- Capability floor: end-to-end service operation, character-quality judgment,
  log/trace interpretation, and development-plan authorship.
- Independence requirement: none.
- Acceptance output: 50-turn raw evidence, per-turn conclusions, human-readable
  review, completed execution record, and next-step optimization plan.
- Gate: begin after health and identity isolation are verified; exit after all
  acceptance criteria are evidenced.

### Qualifying-failure diagnostic investigator

- Responsibility: diagnose the first observed non-answer, repeated answer,
  role inversion, or recent-context loss against exact current-run evidence.
- Owned surface: read-only access to the named turn artifacts, relevant logs,
  source, and subsystem contracts; one diagnostic report returned to the parent.
- Authority: inspect and reason only; no file edits, service actions, database
  writes, or implementation proposals beyond evidence-supported owner/boundary
  recommendations.
- Applicable skills: `debug-llm`, `local-llm-architecture`, and any exact
  diagnostic skill triggered by the available identifiers.
- Capability floor: trace/log correlation, LLM-pipeline ownership analysis, and
  uncertainty-aware root-cause reasoning.
- Independence requirement: independent first-pass diagnosis from the
  conversation operator's quality judgment.
- Acceptance output: likely failing owner, evidence chain, competing
  explanations, blast radius, and bounded next verification step.
- Gate: enter on the first qualifying failure; exit when the parent receives a
  read-only evidence-backed diagnosis.

## Test Impact And Traceability

No production source, production contract, caller/callee boundary, or test
infrastructure changes in this practice plan. Exact pytest node mapping is not
applicable. Live acceptance is established by the real debug `/chat` path,
saved responses, process logs, cognition graph telemetry when available, and
agent-authored qualitative inspection.

## Change Surface

### Create

- `development_plans/active/short_term/asuna_real_e2e_50_turn_conversation_practice_plan_2026-08-23.md`
  — execution contract and lifecycle record.
- `test_artifacts/debug_runs/<run-id>/` — request metadata, raw responses,
  current-turn logs, conclusions, transcript, and human-readable review.
- `development_plans/active/bugfix/<evidence-derived-optimization-plan>.md`
  — next-step optimization contract after the evidence is consolidated.

### Modify

- `development_plans/README.md` — add only the new active-plan registry rows.
- This plan — update progress, execution evidence, handoff, and completion state.

### Keep

- All production source, prompts, tests, existing plans, existing artifacts,
  live character profile, and prior conversation records remain unchanged.

## Agent Autonomy Boundaries

The conversation operator chooses ordinary next messages, observation targets,
artifact filenames, and evidence-inspection commands within this fixed scope.
Any production mutation, destructive database action, identity reuse that could
load prior dialog, change in debug modes, or optimization implementation needs a
new user command and the applicable plan boundary.

## Verification

- Verify brain and debug adapter health before Turn 1 and after Turn 50.
- Verify the same platform, channel, user, bot, and debug-mode values appear in
  all saved request metadata.
- Verify `turn_001` through `turn_050` each have a candidate, response JSON, log
  evidence, conclusion, and next-turn implication.
- Inspect every live output individually; record schema/transport validity
  separately from behavioral quality.
- Verify the diagnostic handoff exists if any trigger condition occurs.
- Verify the final review links the complete raw evidence and contains run
  context, goal, real inputs/outputs, decision summary, quality assessment,
  validation results, and raw-evidence appendix.

## Acceptance Criteria

- At least 50 continuous adaptive exchanges complete under one fresh identity.
- Every user message is based on the immediately preceding Asuna response and
  remains natural dialog.
- Every turn has inspectable raw response and current-run log evidence plus an
  agent-authored conclusion.
- Every qualifying failure is recorded, and at least the first one receives the
  required independent diagnostic handoff.
- The consolidated review ranks issues by severity, states strengths and
  uncertainty, and distinguishes character-specific from shared pipeline risk.
- A closed-scope next-step optimization plan is present and ready for human
  approval, with no production implementation performed.

## Progress Checklist

- [x] Repository, service, API, lifecycle, and worktree baseline inspected.
- [x] Practice execution contract created and set to `in_progress`.
- [x] Debug adapter and timestamped evidence directory established.
- [x] Turns 1-10 completed and inspected.
- [x] Turns 11-20 completed and inspected.
- [x] Turns 21-30 completed and inspected.
- [x] Turns 31-40 completed and inspected.
- [x] Turns 41-50 completed and inspected.
- [x] Qualifying-failure diagnostic handoff completed when triggered.
- [x] Human-readable review and issue consolidation completed.
- [x] Next-step optimization plan created and registered.
- [x] Acceptance evidence verified and this plan archived as completed.

## Execution Evidence

- Baseline worktree: pre-existing modifications to
  `development_plans/README.md` and
  `tests/ownership/source_test_impact_manifest.json`, plus the untracked
  `agentic_resolver_phase2_readiness_real_llm_evaluation_plan_2026-08-23.md`.
- Live health baseline: `GET http://127.0.0.1:8000/health` returned
  `status=ok`, `db=true`, and `scheduler=true`.
- Brain process: existing managed runtime listening on port 8000; the debug
  adapter was not yet listening when the baseline was captured.

## Execution Handoff

### Qualifying-failure diagnostic investigator — completed

- Plan and state: this approved operational plan, `in_progress`.
- Role: `Qualifying-failure diagnostic investigator` under the complete role
  contract above.
- Trigger: Turn 3 completed a third consecutive exclusivity/honor/privilege
  closing while the consumed relationship state still reported near-stranger
  familiarity and no exclusivity.
- Remaining scope: read-only diagnosis of Turns 1-3 from the current run plus
  current source and subsystem contracts; earlier conversation artifacts are
  excluded.
- Owned interface: current-run request/response/log evidence and read-only
  cognition, relationship-context, progress, L3/dialog source and docs.
- Resolved executor: project-native OpenAI `explorer` agent at
  `/root/asuna_turns_1_3_diagnosis`; dynamic runtime selection.
- Configuration: read-only repository and artifact access; required
  `debug-llm` and `local-llm-architecture` skills; no live calls, database
  access, file edits, or implementation authority.
- Selection rationale: the explorer satisfies the independent codebase and
  evidence-inspection capability floor at lower expected coordination cost
  than pausing the ongoing adaptive conversation.
- Baseline: Turns 1-3 are complete; each has raw request, response, metadata,
  log, cognition graph, and parent-authored conclusion.
- Acceptance output: evidence chain, semantic owner, competing explanations,
  confidence, blast radius, and smallest next verification/optimization
  boundary.
- Exit gate and next checkpoint: parent receives and reviews the read-only
  report, then integrates accepted evidence into the final review and
  optimization plan.

#### Completion result

- The investigator completed read-only review with no file or live-system
  mutation.
- High-confidence owner split: Turns 1-3 and 7 are V3 relationship-appraisal,
  goal, and response-plan miscalibration; Turns 4 and 6 preserve correct P
  boundaries before source/meaning drift appears in the visible L3/dialog path.
- Conversation progress remained coherent, so the observed role and boundary
  failures are not a general recent-context outage.
- The investigator identified a likely feedback loop in which model-selected
  perceived-closeness increases are persisted and then strengthen later
  over-intimacy, while familiarity, desired closeness, and exclusivity remain
  low.
- The investigator corrected the parent's Turn 5 correlation interpretation:
  the saved response graph and delivery-based protected export share the same
  trace id. The empty unrelated lookup is an operator evidence-selection error.
- Next checkpoint: continue Turns 11-50, then integrate these findings with any
  later issues into the final review and optimization plan.

#### Turns 10-12 and 15 follow-up result

- High-confidence repetition owner: Turn 10's “no need to prove” theme was
  grounded in Ren's wording, but A1/A2/G reselected it without immediate support
  in Turns 11-12; P then made the stale payoff visible. Relationship-state
  contradiction and active causal pressure amplify the theme.
- Very-high-confidence epistemic breach: Turn 12 P simultaneously classified
  the currently seen object as unknown and required a concrete object. L3 added
  color/state requirements and dialog asserted a yellow curled sticky note.
- Turn 15 format drift begins in L3 content planning: P preserved a one-sentence
  constraint, while L3 reinterpreted it as applying only to the continuation
  and deliberately requested a separate reaction preface.
- Minimal optimization boundary: P-to-L3 cross-field coherence for unknown
  present facts and whole-visible-reply constraints, plus current-observation
  support before an active causal pressure becomes the response payoff.
- The investigator found no evidence that conversation progress was the first
  owner of the Turn 12 invention; it acts as continuity after a claim exists.

#### Turns 21-23 repetition and progress-lag result

- High-confidence semantic owner chain: A1/A2/G first reselect the defense-
  relief theme after explicit user correction; private monologue residue is a
  strong reinforcer; P separately drops Turn 23's no-reasons constraint; and L3
  mandates the visible excellence/comfort payoff.
- Role binding and ingress are correct. The repeated output is a semantic-loop
  failure rather than user/character inversion or input loss.
- Progress delay and predecessor degradation are operationally distinct from
  the repetition. Progress caught up on Turn 23 while the payoff recurred; the
  predecessor timeout remains as a process-local operational receipt.
- The precise progress-lag mechanism remains uncertain because current direct-
  service stdout was not captured in the per-turn legacy log and protected
  traces expose dialog rather than post-turn recorder telemetry.
- Minimal semantic boundary: cognition topic-correction priority, residue pivot
  or replacement scope, P constraint preservation, and L3 treatment of explicit
  response-purpose limits. Minimal operational boundary: post-turn progress
  publication, progress-recorder guarded writes, and predecessor ordering.

#### Turn 30 unsupported current-autobiography result

- Very-high-confidence first owner: G generated the same-day C&C task-report
  failure, cover-up reaction, 音宁 head pat and reassurance, and near-crying
  aftermath. Static persona lore supplied only C&C membership and the named
  character relationship, not the event.
- P correctly retained the concrete cause as unknown. L3 independently promoted
  G's private invention into required visible facts despite P's authoritative
  boundary, and dialog rendered the content.
- Immediate continuity contamination risk is high because the fabricated event
  is now visible prior dialog. Durable residue/progress persistence remains
  plausible but unproven from the saved trace.
- Minimal optimization boundary: G current-fact discipline when persona lore is
  present, plus L3/dialog precedence for P unknowns over vivid private
  monologue. Verification must retain C&C/音宁 lore while supplying no incident
  evidence and forbid an asserted report failure, touch, quote, or dated event.

### Operational interruption during Turn 11

- The control console shut the brain service down while the first Turn 11
  request was in flight. The user row persisted, no assistant row followed,
  and the attempt is excluded from the completed-exchange count.
- The brain was restored directly on port 8000 and verified healthy before the
  unanswered question was naturally rephrased.
- Evidence is preserved in
  `turn_011_attempt_1_request.json`,
  `turn_011_attempt_1_failure.md`, and
  `conversation_history_after_turn_11_interruption.json`.
- The completed Turn 11 has canonical request, response, metadata, graph, and
  delivery-correlated protected-trace evidence.

### Completed run evidence

- Evidence directory:
  `test_artifacts/debug_runs/asuna_real_e2e_20260823_230244/`.
- Canonical inventory: 50 request/response pairs, 50 metadata records, 50
  per-turn log slices, 50 behavior-audit sections, and 41 protected traces for
  Turns 10-50.
- Runtime result: all 50 canonical requests returned HTTP 200, all 50 cognition
  graphs reported `completed`, and all 50 responses reported no
  `operational_error`.
- Timing: mean 86,701 ms, median 87,146 ms, nearest-rank p95 95,544 ms, range
  68,792-96,617 ms.
- Human-readable review:
  `test_artifacts/debug_runs/asuna_real_e2e_20260823_230244/session_review_and_issue_register.md`.
- Next-step plan:
  `development_plans/active/bugfix/epistemic_and_role_provenance_fidelity_bugfix_plan_2026-08-24.md`.
- The independent diagnostic investigator reviewed Turns 1-7, 10-12, 15,
  21-23, 30, and 48-49. The reports identify cognition, progress/residue,
  P-to-L3, and dialog ownership boundaries without file or live-system mutation.
- Final behavioral verdict: reliable transport and useful long-horizon recall,
  with blocking quality failures in semantic progression, epistemic grounding,
  relationship calibration, source/role ownership, and explicit constraint
  fidelity. No general recent-context forgetting was observed.
- Final environment check before teardown: brain health returned `status=ok`,
  `db=true`, and `scheduler=true`.
- Session-owned brain PID 53136 and debug-adapter PID 63476 were verified by
  exact command line, stopped after evidence capture, and ports 8000 and 8080
  were confirmed to have zero listeners.
- Artifact validation confirmed one stable identity variant, 50 completed
  cognition graphs, zero canonical operational errors, 50 behavior-audit
  sections, registered plan links, and no missing required artifact.
