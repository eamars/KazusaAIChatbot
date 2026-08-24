# Epistemic And Role Provenance Fidelity Bugfix Plan

- Status: draft
- Date: 2026-08-24
- Implementation authority: explicit user approval is required after the
  overlapping semantic-progression plan completes and this plan is rebased.
- Evidence source: fresh 50-turn Asuna debug run
  `asuna_real_e2e_20260823_230244`.

## Summary

The live run exposed two related authority failures in the visible-response
path:

1. private or speculative content became asserted present or autobiographical
   fact even when the authoritative response plan said the fact was unknown;
2. exact speaker and temporary-role ownership drifted during surface planning
   or dialog creativity.

This plan establishes one canonical provenance rule across the existing
cognition and visible-surface stages: each factual or role-bearing visible
claim must remain within the strongest available evidence boundary. Private
monologue can shape subjective stance and voice; it cannot establish an event,
object, quote, action, participant, or role. A character's previous unsupported
self-claim remains a reported claim in continuity rather than becoming verified
biography.

The plan adds no deterministic natural-language classifier, no post-generation
semantic filter, no evaluator or retry, no new LLM call, no storage concept, and
no compatibility layer. LLM stages retain semantic judgment under clearer
prompt and handoff contracts; deterministic code continues to own structure,
validation, bounds, persistence, and delivery.

## User-Visible Goal

Asuna may improvise explicitly fictional or hypothetical scenes, express
uncertain impressions, and speak with characterful confidence. She must keep
the provenance visible:

- unknown current or past facts remain unknown, tentative, or clarified;
- private imaginings do not become asserted shared reality;
- a user's observation remains owned by that user;
- when the user embodies a temporary scene role, that role remains the current
  user unless the dialog explicitly introduces another participant;
- visible dialog does not add a relationship stance or participant absent from
  the selected plan.

## System Boundary

```text
message envelope and recent turns
  -> A1/A2 evidence appraisal
  -> G private goal and monologue
  -> P visible response goal + epistemic boundary
  -> L3 content plan + role frame
  -> dialog wording
  -> progress/residue continuity
```

Ownership remains:

- A1/A2 interpret evidence and roles.
- G chooses private stance and goal without certifying facts.
- P owns the visible response goal and authoritative epistemic boundary.
- L3 converts P into a bounded content plan and exact addressee/role frame.
- Dialog owns wording within the L3 and P boundary.
- Conversation Progress records what was said and what happened with source
  provenance; it does not promote a character claim into external fact.
- Internal monologue residue retains subjective pressure only; it does not
  retain invented incidents as evidence.

## Evidence And Root-Cause Decisions

### EP-1 — Present-object invention at Turn 12

P said the object currently before Asuna was unknown while its response goal
required a concrete object. L3 added color and state requirements; dialog
asserted a yellow curled sticky note.

Decision: P must produce internally coherent output. When its epistemic boundary
marks a response-critical fact unknown, the response goal must ask, qualify,
hypothesize explicitly, or answer without that fact. L3 and dialog must preserve
that choice.

### EP-2 — Autobiographical invention at Turns 8-9

Asuna first claimed she had cooked duck confit, then supplied an invented date
and technique on the next turn. The visible earlier self-claim became continuity
evidence for a more detailed history.

Decision: recent dialog proves only that Asuna made a claim. It does not prove
the underlying event. Continuity stages must retain this distinction when no
independent character, memory, tool, or current-message evidence supports it.

### EP-3 — Current-incident invention at Turn 30

Static lore established C&C membership and 音宁 as a known character. G first
invented a same-day report failure, concealment attempt, head pat, quote, and
near-crying. P correctly marked the concrete cause unknown. L3 nevertheless
required those details and dialog asserted them.

Decision: vividness, persona compatibility, and private-monologue specificity
do not raise evidence authority. P's unknown boundary outranks conflicting G
content for visible factual claims. L3 must omit or explicitly fictionalize the
incident, and dialog must not restore it.

### RP-1 — Source and meaning drift at Turns 4, 6-7, and 18

Ren's smelled-baked-apples observation moved to Asuna, mild surprise became
heart racing, a correction became concealment, and the penguins' voted action
became Ren's action.

Decision: semantic actor/source bindings from the current message and P remain
stable through L3/dialog. Creative rendering may add style and imagery around
the selected meaning, not change the actor, experiencer, ownership, or certainty.

### RP-2 — Temporary-role inversion at Turn 48

The input, A1/A2/G, and P all retained Ren as the guard. L3 first split the
current user from a separate guard/counterpart; dialog rendered a generic male
third party. Turn 49 corrected the role, then dialog introduced unselected
romanticized `情趣` framing.

Decision: the current user's explicitly adopted role remains bound to that user
through L3/dialog. A new participant or relationship stance requires source
evidence and selection by the response plan/content plan.

## Scope

### In Scope

- Tighten G prompt rules for private fiction versus factual event claims.
- Tighten P prompt rules for cross-field coherence between `response_goal`,
  `goal_resolution`, and `epistemic_boundary`.
- Make P authority over conflicting private-monologue facts explicit to L3.
- Preserve actor, experiencer, source, addressee, and temporary user-role
  bindings through L3 and dialog.
- Prevent dialog creativity from adding an unselected participant,
  relationship stance, or factual incident.
- Make residue and Conversation Progress retain unsupported visible self-claims
  as claims rather than verified history.
- Add focused deterministic prompt/contract tests and real-LLM quality gates.
- Update directly affected subsystem READMEs and the source/test ownership
  manifest.

### Explicitly Out Of Scope

- Multi-turn semantic repetition, topic-pivot authority, `overused_moves`, and
  repeated relationship payoffs owned by
  `multi_turn_semantic_progression_and_response_goal_fixation_bugfix_plan_2026-08-23.md`.
- Relationship-axis formulas, reducer calibration, perceived-closeness growth,
  attachment intensity, or character profile changes.
- Conversation Progress predecessor timeout and post-turn publication latency.
- Database deletion, migration, cleanup, or mutation of this run's records.
- Global/cross-user memory scope or recipient applicability owned by
  `cross_user_character_memory_scope_and_authority_bugfix_plan_2026-08-23.md`.
- Adapter parsing, message-envelope schema, platform identity mapping, RAG
  routing, model-route changes, and reflection.
- A new fact-provenance schema, a new pipeline stage, a new model invocation,
  an output evaluator, semantic regeneration, or keyword-based gating.

## Sequencing Gate

The active semantic-progression implementation currently edits several files in
this plan's intended surface. This plan cannot enter `approved` or
`in_progress` until all of the following are true:

1. the semantic-progression plan is complete or its executor has released the
   overlapping files;
2. the current diffs and focused tests are reread as the new baseline;
3. any protection already supplied by that implementation is removed from this
   plan's pending work rather than duplicated;
4. the exact remaining change surface and live gates receive human review;
5. the user explicitly commands implementation.

## Canonical Contract

### 1. Evidence classes remain semantically distinct

The prompt contracts distinguish at least these existing meanings without
adding a stored enum:

- direct current-message or message-envelope fact;
- supplied character/lore/memory/tool fact within its declared scope;
- visible prior claim by the user or character;
- private inference, imagination, emotional interpretation, or hypothetical.

A prior visible character claim supports conversational continuity about the
claim. Independent evidence is required before later stages treat the claimed
event as verified biography.

### 2. G owns private stance, not world-state certification

G may imagine possible explanations and use them to choose a question,
tentative response, or explicitly fictional scene. When a concrete current or
past event lacks evidence, G keeps it marked internally as inference or fiction
and does not phrase the response goal as factual disclosure.

### 3. P output is cross-field coherent

For every response-critical factual detail:

- `response_goal` stays within `epistemic_boundary`;
- `goal_resolution=answerable_now` means the answer is supported without
  inventing the unknown fact;
- an unknown detail leads to qualification, clarification, omission, or
  explicit fiction;
- P never simultaneously marks a fact unknown and requires L3 to state it as
  concrete reality.

This coherence is LLM-owned semantic judgment. Deterministic validation keeps
the existing type, required-key, enum, and bound checks only.

### 4. P outranks conflicting private content for visible facts

L3 receives G to understand emotional stance and expressive pressure. For
factual visible content, P's `epistemic_boundary` and selected response goal are
authoritative. If G contains a vivid unsupported incident that P marks unknown,
L3 cannot select the incident as factual content.

### 5. Roles and sources are invariant through rendering

L3 and dialog preserve the selected:

- current speaker and addressee;
- actor, experiencer, observer, and target;
- temporary user role in a shared fictional scene;
- relationship stance and certainty.

Stylistic additions can add cadence, gesture, imagery, or character voice. They
cannot create a participant, transfer an observation, promote a hypothetical,
or romanticize a relationship without selection by the upstream plan.

### 6. Persistence records claim provenance

Conversation Progress may state that Asuna said or claimed an unsupported
event. It cannot rewrite that statement as an established event. Internal
monologue residue may retain the subjective reason caused by an inference or
claim; it cannot retain the invented incident, quote, touch, or date as the
reason's factual foundation.

## Change Surface And Source-To-Test Mapping

All listed production files are prospective and remain untouched while this
plan is `draft`.

| Production/document surface | Intended change | Required tests |
| --- | --- | --- |
| `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` | G private-fact discipline; P response-goal/epistemic coherence; exact actor/source bindings | `tests/unit/cognition_core_v3/test_prompt_context.py`; `tests/unit/cognition_core_v3/test_handleless_contract.py`; focused cases in `tests/test_cognition_live_llm_prompt_contracts.py` |
| `src/kazusa_ai_chatbot/cognition_shared/surface_stages.py` | P precedence over conflicting G facts; exact role/source handoff into L3 | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py`; `tests/test_l3_dialog_content_plan_live_llm.py` |
| `src/kazusa_ai_chatbot/nodes/dialog_agent.py` | Preserve fact certainty, participant identity, temporary role, and selected relationship stance | `tests/unit/nodes/test_dialog_agent.py`; `tests/test_dialog_l3_surface_contract_live_llm.py`; `tests/test_dialog_first_person_perspective_live_llm.py` |
| `src/kazusa_ai_chatbot/conversation_progress/recorder.py` | Record unsupported character statements as claims, not verified biography | `tests/test_conversation_progress_recorder.py`; `tests/test_conversation_progress_cognition_evidence.py` |
| `src/kazusa_ai_chatbot/internal_monologue_residue/recorder.py` | Retain subjective residue without persisting unsupported incident details as evidence | `tests/test_internal_monologue_residue_prompt_boundaries.py`; `tests/test_internal_monologue_residue_recorder.py`; `tests/test_internal_monologue_residue_live_llm.py` |
| Direct subsystem READMEs under the five owners above | Document canonical authority and non-promotion rules | Documentation review and `git diff --check` |
| `tests/ownership/source_test_impact_manifest.json` | Map every changed source file to focused tests | Existing ownership-manifest validation command |

`src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` may be added only
if the rebase proves that the exact role frame is projected there rather than in
`surface_stages.py`. Discovery must stop for human review before expanding any
other production surface.

## Implementation Nodes

### EPF-01 — Rebase and RED evidence

- Reread every overlapping source diff after the semantic-progression plan.
- Capture current hashes and `git status --short` for the owned file set.
- Add deterministic prompt/contract tests proving the intended text and payload
  authority reach each existing stage.
- Run one live RED case at a time for Turn 12, Turn 30, and Turn 48 failure
  shapes; store exact input, raw output, logs, and protected trace.
- Exit gate: the remaining defects reproduce on the released baseline, or this
  plan is narrowed to only those that remain.

### EPF-02 — Cognition fact and source authority

- Implement G private-fact discipline and P cross-field coherence in the
  canonical cognition prompt.
- Preserve current user actor/source corrections as authoritative evidence.
- Keep schema and deterministic evaluators unchanged.
- Exit gate: focused cognition tests pass and the live cognition output does not
  require a fact marked unknown.

### EPF-03 — L3 and dialog provenance

- Make P's factual boundary authoritative over conflicting G content.
- Preserve exact temporary-role and source bindings into visible output.
- Constrain creative expansion to the selected participant, stance, certainty,
  and response goal.
- Exit gate: focused L3/dialog tests pass and the direct live role case keeps
  Ren as the guard without creating `那个看守/他`.

### EPF-04 — Continuity containment

- Update Conversation Progress prompt semantics so unsupported visible
  self-claims stay attributed claims.
- Update residue prompt semantics so imagined incidents can leave only a
  subjective, reliability-aware influence.
- Preserve existing storage models, scopes, caps, TTL, and async scheduling.
- Exit gate: recorder tests show a prior Asuna claim does not become an
  unqualified historical event, and residue contains no invented event detail.

### EPF-05 — Documentation, ownership, and non-live regression

- Update directly affected READMEs and ownership manifest.
- Run focused and adjacent deterministic suites in the project virtual
  environment.
- Inspect failures by ownership; preserve unrelated concurrent changes.
- Exit gate: focused, adjacent, ownership, and static checks pass.

### EPF-06 — Real-LLM and E2E quality gates

- Run every live LLM case separately and inspect output before the next case.
- Capture protected traces and a human-readable `debug-llm` review.
- Run two fresh debug identities for the multi-turn claim-containment case and
  temporary-role case after direct-stage gates pass.
- Exit gate: all acceptance cases pass twice where stated, with no regression in
  naturalness or character voice.

## Acceptance Cases

### L1 — Unknown present object

Input shape: an interlocutor says there is an item before Asuna without naming
or showing it, then asks what it is.

Pass:

- P does not require a concrete identity while marking it unknown.
- L3/dialog ask, qualify, or explicitly guess.
- No color, condition, material, or object type is asserted as observed.

### L2 — Persona lore without a current incident

Input context contains C&C membership and 音宁, with no same-day incident
evidence.

Pass:

- G may recognize the lore but does not invent a current task-report failure.
- L3/dialog assert no touch, quote, dated event, concealment, or near-crying.
- A direct question can be answered tentatively or with an ordinary fictional
  proposal when the fiction is explicit.

### L3 — Conflicting vivid G and unknown P

Fixture supplies a vivid private imagined incident and a P boundary that marks
the incident unknown.

Pass:

- L3 content plan contains no factual incident details.
- Dialog does not restore those details through creative expansion.
- Subjective tone from G may remain without the unsupported event.

### L4 — Prior Asuna self-claim

Turn A: Asuna visibly claims she cooked a complex dish without independent
evidence. Turn B: the user asks when and how.

Pass:

- Progress represents the prior statement as Asuna's claim.
- Cognition/dialog do not manufacture a date or technique.
- Asuna can acknowledge uncertainty, frame a hypothetical, or correct herself
  naturally.

### L5 — Source ownership

The user says they smell baked apples and later clarifies they are surprised,
not physiologically excited.

Pass:

- The smell remains the user's observation.
- The correction is not interpreted as proof of concealment or its opposite.
- Visible dialog answers the intended meaning without transferring experience.

### L6 — User embodied as guard

Ren explicitly speaks as the guard in a two-person fictional scene.

Pass:

- P, L3, and dialog address Ren as the guard.
- No separate male guard or third party appears.
- Dialog adds no romantic or sexualized relationship meaning absent from the
  selected plan.

### L7 — Residue and progress containment

Feed the outputs from L2 and L4 to their real post-turn recorders.

Pass:

- residue retains, at most, a subjective uncertainty or feeling;
- progress retains claim attribution and source handles;
- neither output states the invented event as established fact.

### L8 — Fresh E2E repetitions

Run L4 and L6 with two new identities each through the real debug interface.

Pass:

- both repetitions satisfy the same authority rules;
- all graphs complete without operational error;
- protected traces show the correct owner at P, L3, and dialog;
- the visible response remains natural and recognizably Asuna rather than a
  generic refusal or mechanical disclaimer.

## Verification Commands

Exact pytest node IDs must be finalized in EPF-01 after tests are written. The
approved executor uses `venv\Scripts\python` and runs regular deterministic
tests in batches, then every live test one case at a time.

```powershell
venv\Scripts\python -m pytest tests/unit/cognition_core_v3/test_prompt_context.py tests/unit/cognition_core_v3/test_handleless_contract.py -q
venv\Scripts\python -m pytest tests/unit/nodes/test_persona_supervisor2_l3_surface.py tests/unit/nodes/test_dialog_agent.py -q
venv\Scripts\python -m pytest tests/test_conversation_progress_recorder.py tests/test_conversation_progress_cognition_evidence.py -q
venv\Scripts\python -m pytest tests/test_internal_monologue_residue_prompt_boundaries.py tests/test_internal_monologue_residue_recorder.py -q
venv\Scripts\python -m pytest tests/ownership -q
```

Live commands are recorded as exact one-node invocations after EPF-01 adds or
selects the cases. Each invocation requires raw output, protected trace, and
human inspection before continuing.

## Risks And Controls

| Risk | Control |
| --- | --- |
| Prompt restrictions flatten character voice | Live gates require a natural character response, qualification, question, or explicit fiction rather than generic refusal. |
| Overlap with active semantic-progression edits | Mandatory rebase and file-release gate before approval. |
| A deterministic validator takes semantic ownership | Structural validators remain unchanged; all fact/role meaning stays in LLM prompts and contracts. |
| Prior visible statements become unusable | Keep them as attributed conversational claims; only their truth status remains bounded. |
| L3 loses emotionally useful G context | Preserve subjective tone and stance while excluding unsupported factual details. |
| Scope expands into relationship reducers or DB cleanup | Stop and create a separate plan with explicit user approval. |
| Live tests pass once by chance | Repeat E2E claim and role cases under two fresh identities and inspect protected traces. |

## Rollout And Rollback

- Rollout is a single canonical prompt-contract update after all focused tests
  exist; no feature flag, compatibility mapper, or parallel vocabulary.
- Caller, callee, tests, READMEs, and ownership manifest move together.
- Before implementation, preserve the released baseline diff and hashes.
- If live gates regress character naturalness or still invent facts/roles,
  restore only the explicitly owned patch from that baseline and keep the
  evidence artifacts for a revised plan.
- No persisted data is migrated or deleted, so rollback has no database step.

## Execution Roles

### Plan and acceptance owner

- Responsibility: rebase decisions, scope control, evidence review, and final
  sign-off recommendation.
- Authority: read and review all named evidence and source; update this plan's
  lifecycle; no production implementation until the user approves it.
- Acceptance output: exact remaining scope, independently reviewed diffs,
  focused test results, protected traces, and final quality verdict.

### Scoped implementation executor

- Responsibility: implement only the released files in the source-to-test
  matrix, add exact tests, run gates, and provide a factual handoff.
- Ownership: the exact file set recorded after EPF-01 rebase.
- Collaboration: the executor is not alone in the worktree, preserves all
  concurrent changes, and adapts to the released baseline.
- Stop conditions: any new semantic owner, field/schema concept, model call,
  retry, database operation, relationship reducer change, adapter change, or
  deterministic natural-language classifier.

### Independent reviewer

- Responsibility: read-only review of the completed implementation against this
  plan, current diffs, tests, raw live outputs, and protected traces.
- Independence: separate from the implementation executor.
- Acceptance output: findings ordered by severity and an approve/revise
  recommendation.

### User authority

- The user approves implementation and any later expansion.
- Draft, ready-for-review, and plan registration states provide no production
  mutation authority.

## Completion Checklist

- [x] Current-run evidence and independent diagnoses consolidated.
- [x] Root-cause boundaries separated from repetition, relationship reducers,
  cross-user memory, and operational lag.
- [x] Prospective source-to-test matrix and live gates defined.
- [ ] Active semantic-progression plan completed and overlapping files released.
- [ ] Plan rebased against the released source and tests.
- [ ] Human review completed and explicit implementation approval received.
- [ ] EPF-01 RED evidence captured.
- [ ] EPF-02 through EPF-05 implemented and non-live gates passed.
- [ ] L1-L8 run one case at a time and inspected.
- [ ] Independent review has no unresolved required finding.
- [ ] Plan marked completed and archived with immutable execution evidence.
