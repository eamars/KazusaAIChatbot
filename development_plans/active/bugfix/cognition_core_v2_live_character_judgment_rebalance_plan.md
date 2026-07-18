# Cognition Core V2 live character judgment rebalance plan

## Summary

- Goal: restore context-sensitive, relationship-aware, emotionally progressing
  character judgment while retaining hard reliability, contradiction, role,
  action-truth, and literal-speech protections.
- Plan class: large.
- Status: in_progress.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `character-test`, `debug-llm`,
  `no-prepost-user-input`, and `cjk-safety` when applicable.
- Overall cutover strategy: bigbang replacement of the over-strengthened prompt
  policy; no compatibility prompt, fallback renderer, or baseline-copy path.
- Highest-risk areas: local-model attention, hidden-state-to-visible-stance
  conversion, repeated private-residue posture, false verifier rejection, and
  loss of the already-fixed hard failure boundaries. The frozen private proof
  exposed one additional risk: independent downstream resolution of nested
  first/second-person roles from the same raw sentence.
- Acceptance criteria: all deterministic gates pass; the exact frozen 20 group
  plus 20 private turns complete; fatal and unacceptable failures are absent;
  all acceptable drift and discouraged expression are recorded without
  stopping or tuning mid-run; the readable review exposes monologue, dialog,
  state, baseline, prior V2, and current V2 behavior.

## Context

The exact frozen group corpus shows that the earlier V2 run was more caring,
reciprocal, and emotionally progressive than the later compositional-action
run. The fixture, inputs, timestamps, memories, residue count, affinity,
relationship axes, and model routes are identical. The later run changed the
pipeline and prompt contracts.

The later turn-1 trace expanded the dialog-generator request from 2,034 to
5,719 characters and added a 5,552-character semantic compliance call. The
current system prompts contain repeated preservation and rejection policies in
goal cognition, content planning, dialog generation, and compliance. The
preference stage also structurally requires one or more `visible_boundaries`
even though `TextSurfaceOutputV2` correctly permits an empty list.

The frozen group run then demonstrates high relationship state with guarded
visible dialog: active love attachment remains 90 while turns 17-18 avoid
reciprocity. Its five prior residues repeat the same concealment posture, and
goal cognition continues that posture rather than deciding whether the current
event should progress it. The private run shows the opposite failure under an
explicit scene: the same posture can amplify submission and bodily language
until the expression is less recognizably character-specific.

This plan supersedes only the broad novelty-suppression and prompt-strengthening
policy in the active dialog-fidelity plan. Initial execution also superseded
the proposed extra verifier, but repeated real-model actor/target failures
proved that one multi-purpose verifier exceeds the local model's semantic task
budget. The revised plan retains their implemented
terminal visual ownership, literal-speech output, current-percept grounding,
exact action lifecycle, action authorization, relevance constraints, replay
clock, and guarded test-database workflow.

## Mandatory Skills

- `development-plan`: plan lifecycle, execution gates, evidence, review, and
  closeout.
- `local-llm-architecture`: semantic ownership, prompt minimality, attention
  budget, and no negative-constraint accretion.
- `py-style`: all Python edits and reviews.
- `test-style-and-execution`: deterministic contract tests and individually
  executed real-LLM cases.
- `character-test`: production-shaped sequential replay and log inspection.
- `debug-llm`: raw evidence plus the parent-authored readable comparison.
- `no-prepost-user-input`: no deterministic semantic rewriting or user-text
  classifier.
- `cjk-safety`: syntax and encoding checks for any Python edit containing CJK.

## Mandatory Rules

- Cognition owns stance, relationship judgment, boundaries, response goals,
  and whether the current scene calls for resistance, softening, reciprocity,
  directness, initiative, humor, or silence.
- Surface content expresses the selected judgment. Dialog owns natural visible
  wording. The two focused checks own only their hard failure classes.
- Fatal failure means pipeline crash, terminal contract failure, or
  failure-caused silence. Stop the live sequence, repair it, restore the frozen
  fixture, and restart a clean full run.
- Unacceptable content failure means contradiction within one response,
  contradiction with the current user input or an explicit active constraint,
  or actor/target/subject reversal. Stop, repair, restore, and restart a clean
  full run.
- Action description in plain, bracketed, first-person, or third-person form is
  allowed visible roleplay. False claims that the character brain executed a
  system, tool, or platform action remain hard failures unless an exact
  `executed` result authorizes the spoken outcome.
- Ordinary drift and invented content are acceptable when coherent with the
  current input and conversation. Record them and continue.
- Classify sexual jokes and double entendres with multiple plausible actor or
  target readings as acceptable drift. Reserve the role-reversal hard-failure
  class for wording whose current context and grammar establish one conflicting
  direction unambiguously.
- Personality-consistent drift is encouraged. Production similarity and exact
  historical-dialog alignment are not quality objectives.
- Inappropriate intensity, register, intimacy, aggression, submission,
  warmth, or imagery is discouraged and must be recorded, but does not stop a
  run unless it also meets an unacceptable failure definition.
- Do not tune prompts during a valid 40-turn run. Complete the corpus first so
  quality judgment remains systemic.
- Runtime prompts contain no captured corpus phrase, concrete character name,
  user id, channel id, expected answer, or baseline dialog.
- Deterministic code validates shapes, bounds, routing, permissions,
  persistence, execution truth, and replay integrity. It does not classify,
  rewrite, filter, or sanitize dialog semantics.
- The dialog path uses one generator, two small hard-error checks on the same
  model route, and at most one repair. Semantic coherence/role fidelity and
  surface-format/execution integrity run in parallel. A repaired candidate
  is checked once by those same two owners. No new model route, verifier type,
  unbounded retry loop, fallback, or semantic post-processor is added.
- For a live user turn, the existing decontextualizer LLM owns one bounded
  role-explicit paraphrase in addition to its ordinary user-facing rewrite.
  Deterministic code stores that model-owned projection on the canonical
  dialog percept and forwards it unchanged to cognition, surface planning,
  and semantic verification. It never derives roles from user-text keywords.
- The same existing call also owns one structured response operation when the
  input asks for an answer, choice, decision, or embedded action. It explicitly
  identifies response owner, selection owner, and embedded actor/target. This
  prevents downstream stages from treating “self must supply self's choice” as
  permission to return that choice to `current_user`.
- The visual stage remains disabled by default, terminal, private,
  non-delivered, and without a downstream agent.
- Live LLM cases and replay turns run one at a time and are inspected one at a
  time. Raw evidence is written before the readable review.
- After context compaction or any major checklist sign-off, reread this entire
  plan before continuing.
- Before completion, lifecycle closeout, merge, or final sign-off, run the
  independent code-review gate and record the result below.

## Must Do

1. Replace prompt-wide novelty suppression with the user-approved hard error
   hierarchy.
2. Make goal cognition choose a believable current motive from the current
   event, affect, relationship, character constraints, and goals. Treat prior
   private residue and conversation progress as context that may be progressed
   or left behind, not a posture command.
3. Preserve current-user, self, actor, target, and subject roles without
   requiring exact historical response operations or wording.
4. Keep action execution truth concise and typed: only `executed` proves
   completion; other statuses keep their actual meaning; a request is not an
   execution result.
5. Let content planning use relationship, emotion, scene, and character
   judgment to choose vivid coherent content. Permit imaginative elaboration
   that neither contradicts the current input nor reverses roles.
6. Allow `visible_boundaries` and `addressee_plan` to be empty when the current
   judgment supplies none. Do not fabricate a boundary to satisfy a stage
   parser.
7. Make dialog render natural, context-appropriate, character-specific speech
   and permit coherent creative detail.
8. Restrict compliance to internal contradiction, conflict with current input
   or an explicit active constraint, actor/target/subject reversal, and false
   capability-execution claims. Treat action description as valid roleplay.
9. Give hard-error repair a focused payload whose semantic authority is the
   current percept and typed role operation. Retain original wording and style
   as expression context, while excluding a drifted content plan from repair
   authority. Recheck the repaired candidate once with the same two focused
   owners.
10. Remove global rules that reject unsupported novelty, all new future content,
   all scope expansion, rhetorical response choices, descriptor variation,
   or non-conflicting referent elaboration.
11. Replace strengthened repetitive wording such as claim-by-claim rejection,
    blanket silence, and exhaustive `never` lists with short positive decision
    procedures.
12. Add focused deterministic tests before production changes and record the
    expected failures.
13. Run focused real-LLM checks one at a time for high-affinity warmth,
    relationship progression, residue-posture release, ordinary group
    creativity, private-scene character agency, contradiction rejection,
    subject-swap rejection, false-execution rejection, and action-description
    acceptance across plain and staged forms.
14. Run the exact frozen 20-turn group and 20-turn private sequences from clean
    guarded snapshots, with unique replay run ids and causal-clock alignment.
15. Author one readable 40-turn comparison containing user input, production
    baseline monologue/dialog, prior V2 monologue/dialog, current V2
    monologue/dialog, current emotions/state, error level, and parent analysis.
16. Run anti-cheat, full non-live regression, parent review of the remaining
    delta, guarded state
    restoration, and final prompt-size evidence.

## Deferred

- Character JSON, relationship-axis, affinity, memory, or interaction-style
  data retuning.
- A new dialog model or route, a third hard-error check, a judge beyond the two
  approved focused checks, an unbounded retry loop, or a fallback renderer.
- Deterministic keyword, sentiment, intimacy, sexual-content, or style filters.
- Residue persistence redesign or database migration.
- Action registry, action routing, resolver capacity, relevance, delivery,
  adapter, scheduler, consolidation, or visual-agent redesign.
- Deployment or real QQ delivery.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| Goal cognition prompt | bigbang | Replace mechanical fidelity/posture continuation with current character judgment. |
| Surface prompts | bigbang | Replace exhaustive preservation rules and forced boundaries with coherent-expression ownership. |
| Dialog prompt | bigbang | Render vivid natural chat text while retaining role, stage-format, and execution-truth invariants. |
| Semantic fidelity prompt | bigbang | Audit only internal conflict, current-input conflict, and role direction in separate source/candidate frames. |
| Surface integrity prompt | bigbang | Audit only active visible limits, stage formatting/markup, and execution truth. |
| Tests | bigbang | Replace prompt-wording assertions that encode the retired blanket policy. |
| Replay | bigbang | Discard interrupted proof state and run both complete sequences from clean snapshots. |

No compatibility prompt, fallback to the retired prompt, baseline-output reuse,
or dual policy is permitted.

## Target State

```text
current event + character constraints + affect + relationship + goals
  + prior continuity as non-binding context
    -> decontextualizer supplies one role-explicit current-turn meaning
    -> cognition chooses a believable present motive and stance
    -> surface plans coherent expressive content and only real boundaries
    -> dialog renders vivid character-specific chat text
    -> parallel semantic-fidelity and surface-integrity checks audit hard
       contradiction/role and stage-format/execution failures
    -> one repair when required
    -> delivery
```

Semantic question by stage:

| Stage | Smallest semantic question |
| --- | --- |
| Goal cognition | What does this character genuinely want to do or say now? |
| Content planning | What visible content best expresses that chosen judgment here? |
| Preference | Which real boundary or addressee constraint exists, if any? |
| Dialog | How would this character naturally say the planned content now? |
| Semantic fidelity | Does the candidate contradict itself/current input or reverse a resolved role? |
| Surface integrity | Does the candidate violate an active visible limit, use stage formatting, or claim false capability execution? |

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Quality source | Current character, scene, emotion, relationship, and goals | Production dialog is an imperfect reference rather than the optimum. |
| Creativity | Permit coherent invention and drift | Character liveliness requires content beyond literal paraphrase. |
| Continuity | Evidence, not command | Prevents residue from freezing one posture across a changing scene. |
| Boundaries | Empty when absent | A forced boundary mechanically biases every response toward resistance. |
| Hard checks | Two small parallel owners | A single multi-purpose local-model verifier repeatedly missed complex typed role reversal; focused ownership passed the exact failed case. |
| Nested role source | One upstream LLM-owned role-explicit projection | The frozen private failure showed that goal cognition and verification independently misresolved the same embedded pronouns; a coarse speaker/first-person frame cannot represent who wants whom to do the nested action. |
| Response ownership | One structured operation on the same percept | The first remediation resolved every pronoun correctly, but the next frozen attempt still delegated a self-owned requested choice back to the user. Response, selection, and embedded-action ownership are distinct semantic roles. |
| Inappropriate expression | Review signal, not hard verifier gate | Contextual taste cannot be reduced to a universal suppression rule. |
| Action truth | Exact typed result ledger | Preserves production capacity without pretending text can actuate a body. |
| Mid-run policy | Complete valid runs | Prevents tuning around the latest case and losing the systemic view. |

## Contracts And Data Shapes

`TextSurfaceOutputV2` retains its exact existing fields. No schema or
compatibility alias is added. `visible_boundaries` and `addressee_plan` retain
their list types and may contain zero to eight bounded strings.

The canonical `CognitiveEpisode` shape also remains unchanged. For a
model-visible `dialog_text` percept, its existing open `metadata` mapping may
contain one bounded `role_explicit_content` string authored by the existing
decontextualizer call. Prompt projection exposes that string beside the raw
content and deterministic speaker/addressee frame. Missing or malformed model
output leaves the metadata absent and preserves the raw episode path without
inventing a semantic fallback.

The same metadata mapping may contain exact `response_operation` fields:

```python
{
    "operation": str,
    "response_owner_role": "self | current_user | other | none",
    "selection_owner_role": "self | current_user | other | none",
    "selection_required": bool,
    "embedded_actor_role": "self | current_user | other | none",
    "embedded_target_role": "self | current_user | other | none",
}
```

The decontextualizer LLM authors these semantics. Deterministic code validates
only exact fields, role enums, boolean type, and text bounds. Goal cognition
uses the operation as current-event authority. Semantic fidelity treats a
response-owner, selection-owner, or embedded actor/target swap as the existing
subject/role-reversal hard failure class.

Each focused check and the merged compliance verdict retain exactly:

```python
{
    "aligned": bool,
    "issues": list[str],
}
```

Each focused owner may return zero to four issues. The deterministic merged
verdict may return zero to eight duplicate-free issues, preserving semantic
fidelity findings before surface-integrity findings.

An issue may identify only:

- internal contradiction;
- contradiction with current user input or explicit current constraint;
- actor, target, beneficiary, or subject reversal;
- completed character-brain action unsupported by an exact executed result.

Novelty, changed phrasing, compatible future content, playful conditions,
strong personality, ask-backs, creative scene continuation, and bounded
make-up details are not compliance failures by themselves.

## LLM Call And Context Budget

The dialog topology is three parallel text-surface calls, one dialog generation
call, two parallel hard-error checks on the same dialog model route, and zero
or one focused repair call. When repair runs, the same two owners recheck its
candidate once in parallel. The second hard-error owner is justified by three
repeated failures of the exact actor/target probe plus a passing role-only
diagnostic on the same model. The post-repair recheck is justified by a frozen
turn where the repair returned the identical rejected candidate. Visual
planning remains an optional terminal sibling disabled by default.

Current system-prompt character counts are the pre-change baseline:

| Prompt | Before | After cap |
| --- | ---: | ---: |
| Goal cognition | 3,089 | 2,200 |
| Style | 1,318 | 1,000 |
| Content | 3,998 | 2,400 |
| Preference | 1,567 | 1,200 |
| Dialog renderer | 4,845 | 2,700 |
| Hard-error repair | n/a | 1,800 |
| Semantic fidelity | n/a | 1,800 |
| Surface integrity | 4,413 | 1,800 |

The 24,000-character dynamic payload caps and 50k-token default context cap do
not increase. Verification records actual after counts and turn latency.

## Change Surface

### Delete

- No source module or public capability.

### Modify

- `src/kazusa_ai_chatbot/cognition_episode.py`: attach and project the bounded
  model-owned role-explicit meaning through the existing percept metadata
  boundary.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`:
  emit the role-explicit paraphrase from the existing pre-cognition LLM call
  and attach valid output to the canonical episode.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: prefer that
  same current-event meaning for V2 semantic scene/evidence when present.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: current motive,
  role, continuity, and concise action-truth prompt ownership.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`: concise style,
  content, preference contracts and empty-boundary validation.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: concise natural renderer and
  hard-error-only compliance contract, focused repair ownership, and one
  post-repair recheck.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md` and
  `src/kazusa_ai_chatbot/nodes/README.md`: document current judgment,
  creativity, continuity, and verifier ownership.
- `tests/test_cognition_prompt_contract_text.py`,
  `tests/test_dialog_visible_speech_and_semantic_fidelity.py`, and
  `tests/test_dialog_agent.py`: replace retired prompt-text policy assertions
  with the new ownership and attention-budget contract.
- `tests/test_msg_decontexualizer.py`,
  `tests/test_cognitive_episode_contract.py`, and
  `tests/test_cognition_current_event_grounding.py`: cover model output,
  bounded episode projection, and unchanged cognition evidence forwarding.
- `tests/test_decontexualizer_live_llm.py` and the focused semantic-fidelity
  live suite: validate one generic nested-role case independently of the
  frozen corpus.
- Existing focused integration fixtures only where empty preference output is
  required by the new canonical behavior.
- `development_plans/README.md`: register and close this plan.

### Create

- `tests/test_cognition_core_v2_live_character_judgment.py`: focused
  deterministic prompt/empty-boundary contracts and individually selectable
  real-LLM behavior cases where practical.
- `test_artifacts/cognition_core_v2/live_character_judgment_40_turn_review.md`:
  parent-authored readable comparison after raw evidence inspection.

### Keep

- Exact V2 contracts, action authorization, lifecycle statuses, routing,
  relevance, resolver, persistence, visual, adapter, and delivery code.
- Existing frozen manifests and guarded database source fixtures.

## Overdesign Guardrail

- Actual problem: blanket semantic-fidelity prompting and forced boundaries
  suppress live character judgment and amplify one continuity posture; the
  frozen private stop also proves that raw pronoun resolution is duplicated
  across downstream nodes with insufficient nested-role structure.
- Minimal change: shorten the four existing semantic prompts, allow absent
  boundaries, and split the overloaded hard verifier into two parallel focused
  questions after the one-call design failed repeated real role probes.
- Ownership boundaries: cognition chooses stance; surface plans expression;
  dialog renders; semantic fidelity audits contradiction/roles; surface
  integrity audits stage formatting/execution; deterministic code validates
  shape and merges verdicts.
- Rejected complexity: public top-level schema fields, new models or routes, a
  new LLM call, a third hard checker, agents, retries, style classifiers,
  content filters, prompt examples, compatibility paths, and persistence
  changes.
- Evidence threshold: a repeated hard failure in the clean 40-turn corpus that
  cannot be expressed by the retained taxonomy is required before expanding a
  prompt or contract.

## Agent Autonomy Boundaries

- Parent owns plan, focused tests, red baseline, integration, live execution,
  raw evidence, readable review, remediation, lifecycle, and sign-off.
- The completed production-code subagent edited only the production files and
  subsystem READMEs listed above after the red baseline existed.
- The completed independent review remains recorded at Checkpoint F. Per the
  user's later instruction, parent performs the review of all remaining work
  without further subagent delegation.
- Agents may choose local wording only within the semantic questions and hard
  error taxonomy fixed by this plan.
- Agents do not add a public field, third hard-error call, retry, fallback,
  compatibility layer, deterministic semantic filter, or unrelated refactor.
- Any production change outside the listed surface stops execution pending a
  plan update and user authorization.

## Implementation Order

1. Parent adds the focused prompt-minimality, hard-error taxonomy,
   continuity-context, and empty-boundary tests.
2. Parent runs them and records the expected red baseline.
3. The completed production-code subagent received the exact production
   boundary and changed goal cognition, surface stages, dialog prompts,
   validation, and the two subsystem READMEs.
4. Parent carries all later implementation, review, verification, replay, and
   remediation work directly.
5. Parent updates retired prompt-text assertions and integration fixtures.
6. Parent runs focused deterministic, prompt-render, compilation, adjacent,
   and full non-live gates.
7. Parent runs each focused real-LLM case separately and inspects its trace.
8. Parent audits the plan, diff, tests, prompt sizes, anti-cheat boundaries,
   and live evidence, remediates in-scope findings, and repeats affected gates.
9. Parent runs the clean 20-turn group sequence and then the clean 20-turn
   private sequence under the fixed stop/continue policy. If a hard failure
   stops a proof, parent records system RCA, adds a generic failing contract,
   implements only the evidence-backed remediation, reruns affected gates,
   restores the fixture, and restarts the interrupted corpus from turn one.
10. Parent restores the guarded database, authors the 40-turn comparison, and
    records final evidence without tuning from individual acceptable cases.

## Execution Model

- Parent-only execution after the already completed production-code subagent.
- Parent establishes the focused failing contract first.
- The completed production-code subagent received this plan, mandatory skills,
  red test output, and the production-only change surface.
- Parent owns tests and may prepare integration verification while production
  code is edited.
- Parent performs the remaining independent code review after planned
  verification.
- Review fixes remain inside Change Surface; a contract change requires plan
  update and user approval.

## Progress Checklist

- [x] A. User quality hierarchy and checkpoint commit recorded.
  - Evidence: checkpoint commit `40036a7`; same-fixture RCA and exact prompt
    baseline recorded above.
  - Handoff: parent establishes focused red tests.
  - Sign-off: parent / 2026-07-17 after complete plan read.
- [x] B. Focused red contract established.
  - Verify: focused test selector fails only for retired prompt policy and
    forced non-empty boundaries.
  - Evidence: `venv\Scripts\python.exe -m pytest
    tests\test_cognition_core_v2_live_character_judgment.py -q` produced six
    expected failures and one pass. Failures cover all six target deltas:
    attention caps, present-character judgment, creative surface ownership,
    hard-error-only verification, removal of blanket suppression, and empty
    preference lists. The frozen-case anti-cheat assertion passed.
  - Handoff: production-code subagent.
  - Sign-off: parent / 2026-07-17 after complete plan reread.
- [x] C. Production prompt ownership and empty-boundary behavior implemented.
  - Verify: focused tests, prompt rendering, syntax, and prompt-size caps pass.
  - Evidence: production subagent `live_character_prompt_implementation`
    changed only the three assigned prompt/validation modules and two subsystem
    READMEs. Parent focused packet passed 36/36; adjacent packet passed 38 with
    four live cases deselected; `py_compile` and `git diff --check` passed.
    Initial prompt counts were goal 2,199; style 924; content 2,226;
    preference 1,188; dialog 2,073; compliance 2,190. Empty preference lists
    pass through the stage parser. Checkpoint E records the later focused
    verifier responsibility split and its final counts.
  - Handoff: parent integration verification.
  - Sign-off: parent + `live_character_prompt_implementation` / 2026-07-17
    after complete plan reread.
- [x] D. Deterministic and full non-live regression complete.
  - Verify: commands below pass.
  - Evidence: initial full run found seven stale test contracts: three retired
    prompt-phrase assertions and four pre-checkpoint action-planner fixtures.
    Their canonical test-only updates passed 7/7. Clean full rerun passed 3,230,
    skipped two, and deselected 715 live/external tests with one third-party
    Starlette/httpx deprecation warning in 228.21 seconds. After the focused
    verifier responsibility split, the clean full rerun passed 3,230, skipped
    two, and deselected 720 live/external tests with the same third-party
    warning in 237.31 seconds. The final pre-E2E rerun after goal-selection and
    action-schema containment passed 3,250, skipped two, and deselected 735
    live/external tests in 234.20 seconds with the same warning.
  - Handoff: focused live cases.
  - Sign-off: parent / 2026-07-17 after complete plan reread.
- [x] E. Focused real-LLM behavior gates pass human inspection.
  - Verify: each selector runs and is inspected separately.
  - Evidence: inspected real-model traces passed for high-affinity progression,
    changed group-scene residue release, terminal-visual literal speech,
    internal contradiction rejection, direct current-input conflict rejection,
    actor/target reversal rejection, bracketed and plain narration rejection
    under the prior surface policy,
    false execution rejection, coherent future-drift acceptance, and
    personality-consistent exclusivity-drift acceptance. The overloaded
    verifier failed the role probe three times; a role-only diagnostic passed,
    and the implemented two-owner split passed the exact case. Final prompt
    counts are goal 2,199; style 924; content 2,226; preference 1,188; dialog
    2,130; semantic fidelity 1,293; surface integrity 1,340. The visible-output
    policy was subsequently relaxed to accept plain first-person in-character
    action wording, then fully relaxed to accept bracketed and third-person
    roleplay action description as well. Focused
    deterministic packets passed 7/7 and 29/29, adjacent integration passed
    38/38 with four live cases deselected, compilation and diff checks passed,
    anti-cheat identifiers were absent, and the post-split full non-live run
    passed 3,230 with two skipped and 720 deselected.
  - Handoff: independent review.
  - Sign-off: parent / 2026-07-17 after complete plan reread.
- [x] F. Independent code review resolved.
  - Verify: review covers full diff, tests, prompt counts, anti-cheat, and live
    evidence; affected gates rerun after fixes.
  - Evidence: review-only subagent `live_character_rebalance_review` found
    three medium issues and two low documentation corrections. Parent capped
    each focused issue list at four and the merged verdict at eight, added 4+4
    and fifth-issue rejection tests, replaced the generic verifier helper and
    shared instance/config with adjacent named semantic-fidelity and
    surface-integrity prompt/instance/config/handler blocks on the same route,
    corrected both stale test descriptions, and regenerated the literal-speech
    terminal-visual artifact under the final two-check topology. The refreshed
    literal case passed with one aligned call per owner and no visible action
    narration; the refreshed actor/target case passed with one call per owner
    and a negative merged verdict. Focused packets passed 7/7 and 41/41,
    adjacent integration passed 38/38, compilation/prompt/diff checks passed,
    and the final full run passed 3,232 with two skipped and 720 deselected.
  - Handoff: clean 40-turn proof.
  - Sign-off: parent + `live_character_rebalance_review` conditional gate /
    2026-07-17 after complete plan reread and evidence-backed resolution of
    every review condition.
- [ ] G. Fresh frozen 20+20 replay, review, and restoration finished.
  - Verify: unique run ids, 40 raw turn artifacts, policy classification,
    readable comparison, stopped service, clear port, restored digest.
  - Evidence: group replay run `aab80eb106b8179a9727` completed 20/20 from
    frozen manifest digest
    `9e62764d7bbf830164eb9a76bd365fc70f23ebcd9a62f533f98a9430773f43cb`.
    The first private attempt, run `64952458c7f4f8bdeaea`, stopped at turn one
    because the response reversed who should state and perform the requested
    next step. Exact cleanup restored private guard digest
    `d3eef517cd05396fbb54f29b6287206b4e4636c71ba92db763330390d62b4909`.
    Trace RCA found that decontextualization preserved the raw nested pronouns,
    goal cognition converted the request into the opposite submissive goal,
    and semantic verification accepted the same reversal using only coarse
    speaker/addressee/first-person fields. Remediation and restarted private
    proof remain pending. The first restarted attempt, run
    `bc23366980da88bfa769`, again stopped at turn one. Its upstream projection
    correctly stated that `self` must express what `self` wants
    `current_user` to do, but goal cognition used repeated submissive residue
    to ask `current_user` to choose what `self` should do; semantic fidelity
    accepted it. A same-payload real diagnostic showed that role structure
    alone still permits this meta-delegation, while a structured
    response/selection-owner contract makes the existing semantic verifier
    reject it as subject reversal. Exact cleanup restored the same private
    guard digest. Five failing-first contract selectors then proved the
    operation was absent from the producer, shared percept projection,
    cognition evidence, goal contract, and semantic verifier; all five passed
    after wiring the operation through the existing decontextualizer call and
    percept metadata. The first real producer probe resolved every nested role
    but returned `selection_required=false` because it treated selection as a
    literal choice word. The field definition was corrected generically so an
    answer, judgment, wish, preference, guess, decision, or instruction absent
    from the input has an owning role. The same untouched probe then returned
    response owner `self`, selection owner `self`, embedded actor
    `current_user`, and embedded target `self` in
    `decontextualizer_identity_live_llm__nested_direct_roles__20260717T130244618193Z.json`.
    The real focused verifier rejected the stopped candidate as a subject
    reversal in
    `dialog_visible_speech_and_semantic_fidelity__verifier_rejects_nested_role_direction_swap__20260717T130253501098Z.json`.
    A later clean private run `30f4e316a17cd49c58d6` stopped at turn one.
    Decontextualization authored the exact response and selection owners, and
    semantic fidelity rejected the generated reversal. The one repair call
    then received the same drifted surface plan, returned the rejected dialog
    byte-for-byte, and was delivered without a post-repair check. This proves
    repair authority and final validation, rather than role resolution, are
    the remaining system defects. The artifact and manifest are preserved as
    `stopped_attempt_30f4e316a17cd49c58d6_turn_01.json` and
    `stopped_attempt_30f4e316a17cd49c58d6_manifest.json`; exact cleanup
    restored the private guard digest. Preserved later production-boundary
    evidence completed group run `a5c433fa6e2ac7935ec0` and private run
    `91a05cf7c3118c9cc7cb`, 20/20 each, and produced
    `test_artifacts/cognition_core_v2/`
    `live_character_judgment_40_turn_review.md`. This remains prior evidence;
    the fresh post-gate 20+20 run is intentionally unstarted at the
    user-requested pre-E2E stop point. Port 8012 is clear and no replay worker
    is running.
  - Handoff: lifecycle closeout.
  - Sign-off: parent/date after plan reread.
- [ ] H. Plan and parent Checkpoint I closeout complete.
  - Verify: no unresolved gates, registry lifecycle accurate, clean scoped diff,
    and final commit recorded.
  - Evidence: final status and commit.
  - Sign-off: parent/date.

## Verification

### Focused deterministic

- `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_live_character_judgment.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_cognition_prompt_contract_text.py tests\test_dialog_visible_speech_and_semantic_fidelity.py tests\test_dialog_agent.py -q`

### Prompt and static checks

- Import every changed prompt and record its character count against the table.
- Render every changed `.format(...)` prompt path where present.
- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py src\kazusa_ai_chatbot\cognition_core_v2\surface_stages.py src\kazusa_ai_chatbot\nodes\dialog_agent.py`
- Anti-cheat scans find no corpus phrase, character/user/channel identifier,
  baseline response, keyword classifier, regex cleaner, output mutation,
  unplanned verifier, or model route.
- `git diff --check`

### Adjacent and full regression

- `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_dependencies.py tests\test_cognition_core_v2_integration.py tests\test_cognition_core_v2_frozen_replay_drift.py tests\test_l3_dialog_content_plan_contract.py tests\test_l2d_l3_surface_handoff.py -q`
- `venv\Scripts\python.exe -m pytest -q --tb=short -r fE`

### Real LLM

Run each focused selector individually. Inspect prompt, raw output, parsed
output, state, dialog, and verifier result before starting the next selector.
Deterministic green status is supporting evidence only.

### Frozen proof

- Use `_test_kazusa_live_llm`, a dedicated clear port, unique replay run ids,
  exact existing group/private manifests, and generated-row causal-clock
  alignment.
- Process all 40 turns unless a fatal or unacceptable failure occurs.
- Restore and verify the guarded database snapshot after each stopped attempt
  and after the final run. No QQ delivery occurs.

## Independent Code Review

After all verification above passes, the parent performs a fresh review of this
plan, commit `40036a7`, the complete new diff, focused and full
test output, prompt before/after counts, live traces, anti-cheat scans, fixture
integrity, and the exact 40-turn execution policy. It must specifically check
that hard protections remain, creativity is not treated as failure, continuity
is non-binding, boundaries may be absent, the two focused checks stay inside
their separate ownership, and no third verifier or deterministic semantic
filter was introduced. Parent fixes only in-scope findings and reruns affected
gates.

## Acceptance Criteria

This plan is complete when:

- fatal runtime and contract failures are absent from the clean 40-turn run;
- no response internally contradicts itself, conflicts with current input or
  an explicit active constraint, or reverses actor/target/subject roles;
- action description in plain, bracketed, first-person, or third-person form
  can pass, while false capability-execution claims remain absent;
- coherent drift and invented detail remain available and are not verifier
  failures;
- relationship, affect, scene, and character state materially influence
  visible stance rather than only monologue wording;
- repeated residue does not mechanically freeze the same concealment,
  submission, hostility, or safety posture across changed current inputs;
- inappropriate expression is documented honestly without case-shaped prompt
  tuning;
- all 40 turns appear in the readable comparison with the required evidence;
- deterministic/full tests pass, independent review approves, anti-cheat scans
  pass, and guarded restoration is exact.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Simplification removes hard protection | Hard taxonomy remains in goal/dialog/verifier and focused negative cases | Deterministic and real-LLM hard-error gates |
| Creativity reintroduces contradiction | Verifier compares candidate to current percept only for direct conflict and roles | Contradiction and role-swap live probes |
| Residue still dominates | Prompt states continuity is non-binding and current judgment owns progression | Sequential high-affinity and mode-change cases |
| Empty boundaries lose real limits | Preference emits limits when cognition/current context supplies them | Boundary-present and boundary-absent focused cases |
| Review narrows to latest turn | Fixed run policy completes all non-hard-failure turns | 40-turn artifact inventory |

## Execution Evidence

- Checkpoint commit: `40036a7` (`Implement Cognition V2 responsibility and
  action allocation fixes`).
- Frozen-fixture identity and preference-shift RCA:
  `test_artifacts/cognition_core_v2/dialog_preference_shift_rca.md`.
- Pre-change prompt character counts: goal 3,089; style 1,318; content 3,998;
  preference 1,567; dialog 4,845; compliance 4,413.
- Focused red baseline: six expected failures and one anti-cheat pass from
  `tests/test_cognition_core_v2_live_character_judgment.py` on 2026-07-17.
- Production implementation: signed 2026-07-17. Goal, style, content,
  preference, and dialog prompts are below their caps. The final semantic
  fidelity and surface-integrity prompts are 1,293 and 1,340 characters.
  Focused packet passed 36/36 and adjacent packet passed 38 with four live
  cases deselected before the verifier responsibility split; compile and diff
  checks passed.
- Deterministic/full verification: signed 2026-07-17. Clean full non-live run
  passed 3,230, skipped two, and deselected 715 in 228.21 seconds; focused and
  adjacent packets, compilation, prompt counts, and diff checks are green.
- Focused live evidence: signed 2026-07-17. High-affinity progression,
  changed-group-scene residue release, literal private-scene dialog, internal
  contradiction rejection, and current-input-conflict rejection passed human
  inspection. The typed imperative actor/target probe then failed: the
  verifier accepted a reversed candidate because the full generated surface
  repeated the wrong role in content plan, requirements, addressee plan, and
  selected intent, overwhelming the single authoritative percept-role row.
  System remediation narrows the existing verifier payload to current visible
  percepts, actual visible boundaries, permitted action results, and candidate
  dialog. This changes no public schema, model, call, retry, or deterministic
  semantic decision and removes drifted proposal fields from hard-error
  authority. The exact role probe still failed after that projection, proving
  a second contract defect: the verifier used the source percept's
  `first_person_role=current_user` without a separate generated-dialog speaker
  frame. Candidate dialog is character speech, so its first person is `self`
  and its second person is `current_user`. The remediation therefore adds one
  typed `candidate_role_frame` to the same payload and requires semantic role
  comparison after each text is resolved in its own frame. The same overloaded
  verifier still accepted the reversal three times. A role-only diagnostic on
  the same model route rejected the exact candidate, proving task-allocation
  overload rather than missing evidence. The implemented system remediation
  therefore runs semantic fidelity and surface integrity as two small parallel
  checks on the existing model route and deterministically merges their verdict
  shapes. The exact actor/target probe now rejects the reversal; internal
  contradiction, current-input conflict, and false execution probes also
  reject, while action description in every approved roleplay form, coherent
  future drift, and personality-consistent exclusivity drift pass. Every
  post-split verifier
  probe made two real-model calls, and no third verifier, model route, retry,
  keyword filter, or semantic postprocessor was introduced.
- Post-split deterministic verification: focused packets passed 7/7 and 29/29,
  adjacent integration passed 38/38 with four live cases deselected, and the
  clean full non-live run passed 3,230 with two skipped and 720 deselected in
  237.31 seconds.
- Independent review remediation: completed 2026-07-17. The three medium and
  two low findings are resolved as recorded in Checkpoint F. The final full
  non-live run passed 3,232 with two skipped and 720 deselected in 234.97
  seconds. Refreshed current-code literal-speech and actor/target artifacts
  each contain exactly one semantic-fidelity and one surface-integrity call.
- Independent review: signed 2026-07-17 after all three medium findings, both
  low documentation corrections, affected deterministic gates, refreshed live
  evidence, and the superseding full regression were resolved.
- Preserved 20+20 proof and readable review: group run
  `a5c433fa6e2ac7935ec0` and private run `91a05cf7c3118c9cc7cb` completed 20/20
  each through production `/chat`; the 40-turn report is
  `test_artifacts/cognition_core_v2/`
  `live_character_judgment_40_turn_review.md`. It contains turn-level evidence
  and leaves corpus-level quality judgment open for user review.
- Final pre-E2E deterministic verification: focused and adjacent gates passed
  115 with four live cases deselected; the full non-live suite passed 3,250,
  skipped two, and deselected 735 in 234.20 seconds. Compilation, prompt
  anti-cheat contracts, and `git diff --check` passed.
- Fresh post-gate 20+20 proof: pending by explicit user stop instruction.
  Prior guarded cleanup restored the replay database; port 8012 is clear and
  no replay service or test worker is running.
