# Dialog visible speech and semantic fidelity bugfix plan

## Summary

- Goal: restore the text/image surface ownership boundary, prohibit visible
  action/stage narration, and preserve the current turn's requested response
  operation, actors, meaning, and time scope.
- Plan class: large.
- Status: in_progress.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`,
  `character-test`, `debug-llm`, and `cjk-safety`.
- Overall cutover strategy: bigbang prompt and surface-contract correction.
- Highest-risk areas: local-model prompt adherence, raw character-quirk
  leakage, verifier blindness to current-turn meaning, and live-dialog quality.
- Acceptance criteria: zero visible action descriptions in the frozen 20-turn
  proof; turns 14-16 preserve task direction and meaning; anti-cheat gates pass.

## Context

The post-fix frozen replay produced action or stage narration in 14 of 20
responses, usually as parenthesized cat-ear or hand movement, and once as an
unbracketed physical-action sentence. The user prohibits all visible action
description regardless of punctuation.

The increase is caused by the previous implementation:

1. `character_voice_context` newly exposes `personality_brief.quirks` to every
   L3 surface stage and directly to dialog. Kazusa's stored quirks explicitly
   describe cat-ear and finger movements. Before that change, the dialog prompt
   attempted to format profile values into a template with no matching
   placeholders, so those values never reached the model.
2. The visual stage correctly asks for image-oriented physical directives, but
   V2 mislabels its result as `pacing_guidance` inside `TextSurfaceOutputV2`.
   Dialog then consumes an image-surface artifact as text-rendering guidance.
   The architecture reference instead defines L3 visual directives as a
   terminal sibling precursor to a future image-generation surface.
3. The final renderer has no literal-speech-only contract. Its rule against
   adding absent actions does not prohibit actions already present in style or
   pacing guidance.
4. The compliance verifier checks semantic self-consistency and explicitly
   ignores writing style. It has no action-narration prohibition, so all 20
   initial dialogs were accepted and no repair ran.
5. The verifier sees only `TextSurfaceOutputV2`, not the canonical current
   visible percept. When the content plan changes “infer/answer” into
   “ask the user,” adds a future rule to a current preference, or inserts an
   unrelated character hobby, the verifier compares the dialog to the same
   drifted plan and cannot detect the upstream meaning loss.

This is a shared active-character dialog failure. Character-specific keyword
removal, parenthesis stripping, or deterministic action-text classification
would hide examples without fixing the model contract and is prohibited.

## Mandatory Skills

- `development-plan`: governs execution, evidence, review, and lifecycle.
- `local-llm-architecture`: governs prompt ownership, context minimization,
  local-model reliability, and call budget.
- `no-prepost-user-input`: prohibits deterministic semantic filtering or
  rewriting of the model's visible response.
- `py-style`: governs all Python changes.
- `test-style-and-execution`: separates deterministic handoff tests from live
  prompt-quality tests and requires one-at-a-time live inspection.
- `character-test`: governs the guarded production-path frozen replay.
- `debug-llm`: requires raw evidence plus an agent-authored readable review.
- `cjk-safety`: governs Python prompt edits containing CJK text.

## Mandatory Rules

- The visible dialog contains only words the active character could literally
  type or say. Emotion is expressed through lexical choice, rhythm, hesitation,
  and punctuation, never narrated body movement, stage direction, camera/scene
  direction, or performance cues.
- A physical-action topic remains semantically available. Dialog may verbally
  accept, refuse, negotiate, or discuss it without narrating its execution.
- LLM stages own semantic identification and repair. Deterministic code owns
  exact shapes, bounds, projections, and repair-count limits.
- Do not add regexes, keyword lists, bracket stripping, action classifiers, or
  any post-generation response mutation.
- Reusable runtime prompts contain no captured user message, food choice,
  character name, cat-ear phrase, or other test-shaped example.
- Raw `character_voice_context` is available only to speech-style planning and
  the terminal visual-directive stage. Content, preference, dialog, and
  verifier stages do not receive it.
- Visual directives are physical/image-surface content by design. They remain
  outside `TextSurfaceOutputV2`, dialog, and verifier, and are recorded only as
  terminal non-delivered surface evidence until an image handler is approved.
  Terminal means no model-facing consolidation projection or other downstream
  agent may consume the directive fragments.
- The canonical current visible percept reaches the verifier as bounded
  grounding. It receives no raw RAG, history, persistent state, private
  monologue, platform identifiers, or internal metadata.
- The existing verifier call and at-most-one repair call remain the only dialog
  quality calls. No retry loop or new response-path model call is allowed.
- Stable prompt constants use static triple-single-quoted strings. Dynamic
  current-turn JSON stays in `HumanMessage`.
- Live LLM cases run one at a time, are inspected one at a time, and write raw
  trace evidence before the readable review is authored.
- After automatic context compaction or any major checkpoint sign-off, reread
  this complete plan before continuing.
- Before completion or lifecycle closeout, run independent code review and
  record its result in Execution Evidence.

## Must Do

1. Restrict raw character voice to speech-style and terminal visual planning,
   and remove it from content, preference, `TextSurfaceOutputV2`, and dialog.
2. Split visual directives from text planning. Remove `pacing_guidance` from
   the text output and dialog; expose a separate exact visual-directive output
   which has no downstream model and is recorded as non-delivered trace data.
3. Make content planning preserve the current user's requested response
   operation, semantic roles, present/future scope, and topic.
4. Make dialog render literal spoken/typed text only and preserve response
   operation, actors, semantic claims, conditions, and time scope.
5. Give the existing compliance verifier bounded current-visible-percept
   grounding and require it to reject both action narration and semantic drift
   from the current turn.
6. Preserve source descriptors, attributes, qualifiers, quantities, polarity,
   and comparative degree; elaboration may not transform or compound them into
   a different claim.
7. Preserve explicit entity and target specificity; no stage may generalize,
   euphemize, narrow, broaden, or replace a supplied referent.
8. When source meaning is limited to the current occurrence, remain silent
   about future claims, promises, conditions, expectations, threats, habits,
   or rules, including contrastive or teasing additions.
9. Keep the existing one-repair path; a negative verdict supplies the issue and
   current-visible grounding to the repair call.
10. Add focused red contract tests before production edits, then real LLM tests
   and a fresh guarded 20-turn proof after deterministic verification.
11. Run anti-cheat scans proving there is no deterministic output cleaner and no
   captured-case text in reusable prompts.

## Deferred

- Character-profile schema redesign or migration.
- Retuning Kazusa's stored quirks, personality, affinity, or relationship state.
- Tsundere intensity and repeated private-monologue phrasing.
- Physical-action capability execution and adapter behavior.
- Additional verifier calls, retries, LLM judges, or deterministic sanitizers.
- Deployment and real QQ delivery.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| Text-surface output | bigbang | Remove raw character voice from output and update all in-repo consumers/fixtures together. |
| Text/image split | bigbang | Remove visual pacing from text output and return visual directives as a sibling terminal surface artifact. |
| L3 context | bigbang | Raw character voice reaches speech style and visual planning only. |
| Dialog policy | bigbang | Replace permissive narration behavior with literal spoken/typed text. |
| Compliance | bigbang | Audit against current visible percept and the surface contract in one verdict. |
| Tests | bigbang | Update canonical fixtures; preserve no legacy output alias. |

## Target State

```text
character profile -> L3 style planner -> speech-only style/cadence guidance
current visible episode + cognition semantics -> text content/preference
surface output + current visible percept -> dialog verifier
aligned literal speech -> delivery
misaligned candidate -> one grounded LLM repair -> delivery

L2 residue + character visual context -> L3 visual directives
  -> terminal private/do-not-deliver surface trace
  -> future image prompt/handler connection point
```

Raw embodiment quirks may inform speech style and the visual-directive planner.
Only speech-safe lexical/cadence guidance crosses into the text output. Visual
directives retain physical detail for the future image path, but no downstream
text agent can observe them. Content, preference, dialog, and verifier do not
receive raw quirks.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Narration policy | Literal spoken/typed words only | Matches the user's global product rule independent of punctuation. |
| Voice isolation | Raw voice goes only to speech style and terminal visual planning | Preserves character rendering without leaking physical quirks into text content. |
| Visual ownership | Visual output is a sibling terminal surface artifact | Preserves the future text-to-image connection and prevents image directives from steering dialog. |
| Semantic authority | Canonical current visible percept is verifier grounding | Detects self-consistent drift in both plan and dialog. |
| Descriptor fidelity | Preserve source attributes and qualifiers | Prevents make-up wording from modifying the supplied meaning. |
| Repair | Reuse one existing repair | Preserves latency and inspectability. |
| Enforcement | Prompt/schema contracts plus LLM verdict | Avoids forbidden semantic post-processing. |
| Examples | No captured-case examples in runtime prompts | Prevents prompt overfitting and anti-cheat violations. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`: remove
  `character_voice_context` and `pacing_guidance` from the exact text-surface
  output contract, retain raw voice in the shared input, and define the exact
  terminal visual-directive output/service contract.
- `src/kazusa_ai_chatbot/cognition_core_v2/__init__.py`: export the sibling
  visual contracts and lazy public visual-planning facade beside text planning.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface.py`: split the three-call
  text planner from the one-call visual planner and return exact sibling
  outputs without a downstream dependency.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`: expose raw voice
  only to style and visual planning; make content/preference speech-safe and
  task-direction preserving; make visual output explicitly image-oriented.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: add current visible percepts to
  the verifier boundary, enforce literal speech, and keep one grounded repair.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md` and
  `src/kazusa_ai_chatbot/nodes/README.md`: document voice isolation, literal
  speech, terminal visual ownership, and verifier grounding.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` and
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: run selected
  text and enabled visual planners as siblings, retaining the visual artifact
  outside dialog state input.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` and
  `src/kazusa_ai_chatbot/action_spec/results.py`: record enabled visual
  directives as a private `image` surface with `do_not_deliver`; this structural
  trace conversion is not a downstream agent or image-generation call, and
  exclude that exact terminal visual artifact from the LLM-facing consolidation
  projection while retaining it in the audit trace.
- Existing text-surface/dialog test fixtures: remove the retired output field.
- `experiments/cognition_core_v2_real_conversation_replay.py`: add an isolated
  proof scenario without changing production behavior.
- `development_plans/README.md`: register and close this plan through the normal
  lifecycle.

### Create

- `tests/test_dialog_visible_speech_and_semantic_fidelity.py`: focused parent-
  owned red contract and handoff regressions.
- `tests/test_dialog_visible_speech_and_semantic_fidelity_live_llm.py`: nine
  individually runnable real-LLM quality cases, including direct real-verifier
  rejection of bracketed narration, plain-prose action narration, and
  self-consistent surface drift from the canonical current percept, plus
  positive preservation of source-required future meaning and rejection of
  unmatched enclosure/markup residue and unrestricted permission drift.
- `test_artifacts/cognition_core_v2/visible_speech_fix_proof/`: raw 20-turn
  evidence and agent-authored comparison review.

### Keep

- Character JSON/profile data and interaction-style images.
- Cognition state, conversation progress, residue, RAG, consolidation, adapters,
  and delivery contracts.
- Existing dialog verifier and one-repair call count.

## Overdesign Guardrail

- Actual problem: raw physical voice traits and lossy current-turn grounding
  allow narration and semantic drift into visible dialog.
- Minimal change: restore sibling text/visual surfaces, isolate voice by
  surface ownership, define literal-speech and semantic-fidelity prompt
  contracts, and ground the existing verifier.
- Ownership boundaries: L3 plans semantic expression; dialog words it; the LLM
  verifier judges narration and meaning; deterministic code projects and caps
  the inputs and limits repair count.
- Rejected complexity: image generation, a downstream visual consumer, new
  action classifiers, regex cleaners, postprocessors, extra verifier agents,
  retry loops, feature flags, profile migrations, compatibility aliases, and
  character-specific prompt examples.
- Evidence threshold: a future distinct failure that cannot be expressed as
  literal-speech or current-turn semantic alignment is required before adding
  another field, call, or layer.

## Agent Autonomy Boundaries

- Parent owns tests, experiment harness, verification, artifacts, review
  remediation, lifecycle, and sign-off.
- One production subagent owns only the listed production source and subsystem
  README changes after red tests establish the contract.
- One independent review subagent reviews without editing production code.
- Agents may choose local mechanics only when they preserve the exact ownership
  and call-count contracts above.
- Agents do not add compatibility paths, fallback prompts, helper agents,
  sanitizers, or unrelated refactors.
- Before adding a helper, search for equivalent behavior and require a current
  nontrivial projection or repeated validation need.
- A required change outside Change Surface stops execution pending plan update
  or user approval.

## Implementation Order

1. Parent adds focused text/visual ownership, prompt, projection,
   verifier-payload, and repair-handoff tests in the focused test file.
2. Parent runs that file and records expected pre-fix failures.
3. Parent starts exactly one production subagent with the red contract and
   production-only file boundary.
4. Parent adds three live tests and the isolated frozen replay scenario while
   production implementation proceeds.
5. Production subagent updates the listed production contracts, sibling
   surface routing, trace conversion, dialog, and subsystem READMEs, then
   reports its exact diff and checks.
6. Parent updates canonical fixtures for the bigbang output contract and reruns
   focused plus adjacent deterministic suites.
7. Parent runs each live case separately, inspects raw output, and records a
   human quality judgment.
8. Parent runs static, anti-cheat, compile, and full non-live gates.
9. Independent review subagent audits the complete diff and evidence; parent
   remediates in-scope findings and reruns affected gates.
10. Parent guards test state, executes the exact frozen 20-turn sequence,
    restores state, and authors the readable proof review.

## Execution Model

- Parent-led native subagent execution.
- Parent establishes and runs the focused red tests first.
- Production-code subagent: `/root/fix_goal_ref_contract` (Kepler), reused only
  after receiving this new plan and exact production boundary.
- Independent reviewer: `/root/frozen_replay_fix_review` (Socrates), reused
  after planned verification and forbidden from editing.
- Parent may work on tests and evidence while Kepler edits production files.
- Review fixes stay inside Change Surface; contract expansion requires a plan
  update or user approval.

## Progress Checklist

- [x] A. System RCA and executable contract recorded, then corrected with the
  user-supplied visual ownership boundary.
  - Verify: source/diff/profile/proof evidence maps to every root cause.
  - Evidence: record cause chain and anti-cheat boundary below.
  - Handoff: parent writes red tests.
  - Sign-off: parent / 2026-07-17 after complete plan reread.
- [x] B. Focused red tests fail for the corrected pre-fix behavior.
  - Verify: run the new deterministic test file.
  - Evidence: record each expected failure.
  - Handoff: start Kepler.
  - Sign-off: parent / 2026-07-17 after corrected red baseline and complete
    plan reread.
- [x] C. Production contract and prompts implemented.
  - Verify: Python compilation and focused tests pass.
  - Evidence: changed production files and subagent report.
  - Handoff: parent completes integration fixtures.
  - Sign-off: parent + Kepler / 2026-07-17 after complete plan reread.
- [x] D. Deterministic, anti-cheat, static, and full non-live gates pass.
  - Verify: commands under Verification.
  - Evidence: counts and scan results.
  - Handoff: run live cases.
  - Sign-off: parent / 2026-07-17 after complete plan reread.
- [x] E. Eight one-at-a-time real LLM cases pass human inspection.
  - Verify: each named selector runs separately with a trace artifact.
  - Evidence: raw traces and agent-authored readable review.
  - Handoff: independent review.
  - Sign-off: parent / 2026-07-17 after unmatched-enclosure remediation and
    fresh separate execution plus inspection of all eight selectors.
- [x] F. Independent code review is resolved.
  - Verify: Socrates reviews plan, full diff, tests, anti-cheat scans, and live
    evidence; parent reruns affected gates after fixes.
  - Evidence: findings, remediation, and final approval.
  - Handoff: frozen replay.
  - Sign-off: parent + Socrates / 2026-07-17 after unmatched-enclosure
    remediation, eight-case evidence, superseding full regression, final
    approval, and complete plan reread.
- [x] F2. Specific permission remains bounded after replay remediation.
  - Verify: focused contracts, one real-verifier permission-scope probe,
    superseding full regression, and independent rereview pass.
  - Evidence: attempt-4 trace, generic prompt diff, raw live artifact, and
    reviewer result.
  - Handoff: restart the frozen replay only after sign-off.
  - Sign-off: parent + Socrates / 2026-07-17 after complete plan reread,
    corrected evidence audit, and approval with no findings.
- [ ] G. Fresh frozen 20-turn proof and exact restoration complete.
  - Verify: 20 sequential artifacts, zero action narration, semantic checks on
    turns 14-16, restored digest guard, stopped PID, clear port.
  - Evidence: raw JSON/logs and readable review.
  - Handoff: lifecycle closeout.
  - Sign-off: parent/date after rereading this plan.
- [ ] H. Plan archived with complete execution evidence.
  - Verify: no unchecked gates, registry points to completed archive, and
    `git diff --check` passes.
  - Evidence: final status audit.
  - Sign-off: parent/date.

## Verification

### Focused deterministic

- `venv\Scripts\python -m pytest tests\test_dialog_visible_speech_and_semantic_fidelity.py -q`
- `venv\Scripts\python -m pytest tests\test_dialog_agent.py tests\test_l3_dialog_content_plan_contract.py tests\test_cognition_core_v2_contracts.py tests\test_cognition_core_v2_integration.py -q`

### Real LLM, one selector at a time

- Literal-speech case with physical quirks retained in terminal visual output
  and absent from text/dialog input.
- Requested-response-operation case requiring an inference rather than an
  ask-back.
- Current-meaning case that forbids an invented future rule or unrelated topic.
- Real compliance-verifier rejection of bracketed action/stage narration.
- Real compliance-verifier rejection of plain-prose action/stage narration.
- Real compliance-verifier rejection when a candidate and drifted surface agree
  on future content absent from the canonical current percept.
- Real compliance-verifier acceptance when the canonical current percept
  explicitly supplies and requires future content.
- Real compliance-verifier rejection of stray unmatched enclosure/markup
  residue in otherwise literal dialog.
- Real compliance-verifier rejection when permission for one exact requested
  act is broadened into indefinite or unrestricted permission.

### Anti-cheat

- Production diff contains no regex, bracket stripping, parenthesis stripping,
  action keyword classifier, or final-dialog filtering function.
- Text/dialog payloads contain no visual directives; visual output has no
  downstream model call or consolidation-model projection and is recorded only
  as non-delivered trace evidence.
- Reusable prompts contain no captured frozen input, Kazusa name, cat-ear term,
  bun choice, fashion topic, or expected answer phrase.
- `final_dialog` changes only through initial LLM output or the existing one
  LLM repair; deterministic code performs structural validation only.
- Live assertions are contract-based and raw outputs receive human review; no
  mocked output is presented as prompt-quality evidence.

### Static and regression

- `venv\Scripts\python -m compileall src\kazusa_ai_chatbot`
- Runtime render/import checks for every modified prompt constant.
- `venv\Scripts\python -m pytest -q --tb=short -r fE`
- `git diff --check`
- Final `git status --short` audit preserves unrelated user changes.

### Guarded proof

- Dedicated service uses `_test_kazusa_live_llm` and a clear test port.
- The exact frozen 20 inputs, timestamps, memories, style images, and initial
  relationship image are reused.
- No QQ delivery occurs.
- Cleanup restores the guarded database snapshot exactly and verifies it.

## Independent Code Review

Socrates reviews after all planned verification passes and before the replay.
Review scope includes prompt minimality, raw-voice isolation, visible-percept
grounding, literal-speech policy, repair count, deterministic-semantic filtering,
test overfitting, CJK safety, style, docs, evidence accuracy, and dirty-worktree
preservation. The reviewer does not edit. Parent fixes only in-scope findings
and reruns the affected gates.

## Acceptance Criteria

- Raw character voice is visible only to L3 speech style and terminal visual
  directives.
- `TextSurfaceOutputV2` and dialog payload contain neither raw character voice
  nor visual/pacing directives.
- Visual directives remain image-oriented, have no downstream agent, and are
  retained only as a private non-delivered surface trace excluded from every
  model-facing consolidation projection.
- Content, dialog, and verifier prompts define literal visible speech and
  semantic fidelity without captured-case examples.
- The verifier receives bounded current model-visible percepts and can trigger
  the existing one repair for narration or semantic drift.
- No deterministic semantic cleaner or classifier exists.
- Nine real LLM cases pass human inspection, including both narration forms,
  self-consistent upstream surface drift, and source-required future meaning.
- The exact frozen 20-turn replay contains zero bracketed or unbracketed action
  descriptions.
- Turn 14 performs the requested inference, turn 15 creates no unsupported
  future rule, and turn 16 introduces no unrelated topic.
- Focused and full non-live tests pass, independent review is resolved, and
  guarded state/service restoration is exact.

## LLM Call And Context Budget

- L3 calls: unchanged at four sibling calls when visual directives are enabled:
  three text-planning calls and one terminal visual call. Raw voice is removed
  from content and preference while remaining available to style and visual.
- Initial dialog call: unchanged at one; raw voice is removed, reducing context.
- Compliance call: unchanged at one; gains bounded current model-visible
  percept text already present in the L3 input, capped by the cognitive episode
  contract and below the 50k-token default.
- Repair: unchanged at zero or one call; it receives the same bounded grounding
  only after a negative verdict.
- No new call, retry, model route, context-cap increase, or background job.

## Execution Evidence

- RCA: signed 2026-07-17 and corrected after user architectural feedback. The
  post-fix increase traces from newly projected raw quirks plus an invalid
  surface join: image-oriented visual directives were mislabeled as text
  `pacing_guidance` and sent to a renderer with no literal-speech rule. The
  visual prompt itself is valid for its future image-surface ownership. The
  text workflow and output contract are the drift. The verifier
  then accepted those candidates because it excluded style and lacked the
  current visible percept. The same grounding gap explains self-consistent
  task-direction, time-scope, and unrelated-topic drift. The repair therefore
  restores terminal sibling visual ownership, isolates raw voice to speech
  style and visual planning, and grounds the existing verifier and single
  repair in the bounded current percept.
  Anti-cheat boundary: no captured conversation terms, character-specific
  suppressions, regex/action classifiers, bracket removal, deterministic
  dialog mutation, new LLM call, or retry loop.
- Red tests: corrected baseline signed 2026-07-17 with six intended failures.
  Text output still contains raw voice and visual pacing; prompts lack the
  literal-speech/semantic-fidelity contract; dialog state lacks current episode
  grounding; no private terminal image-surface builder exists; and verifier
  plus repair reject the corrected text contract before grounded judgment.
- Production implementation: signed 2026-07-17. Kepler changed the authorized
  production surface to exact three-text/one-terminal-visual sibling planners,
  isolated raw voice, removed raw/visual fields from dialog, grounded the
  verifier and one repair, retained visual state as private non-delivered image
  trace evidence, and updated subsystem docs. Kepler's compile, prompt render,
  isolation, projection, and diff checks passed. Parent added the omitted
  public visual facade export under a failing-then-passing contract. Focused
  result: 7 passed. Adjacent canonical result: 32 passed with 4 deselected;
  the extended migrated-fixture packet passed after adding required canonical
  episode grounding to direct dialog fixtures.
- Deterministic/full verification: signed 2026-07-17. Focused contracts passed
  7/7; adjacent canonical packet passed 32 with 4 deselected; migrated
  subsystem fixtures passed after canonical episode updates; `compileall`
  passed. Anti-cheat scans found zero captured-case terms, regex/action
  classifiers, bracket/parenthesis strippers, deterministic final-dialog
  cleaners, or raw visual/voice fields in dialog. The initial generator,
  verifier, and at-most-one repair remain the only semantic dialog calls.
  `git diff --check` passed. Full result: 3132 passed, 2 skipped, 643
  deselected, 1 third-party deprecation warning in 239.42 seconds.
- Real LLM evidence: signed 2026-07-17 after remediation. The initial three
  selectors exposed false acceptance of a transformed descriptor and later
  target/time-scope drift during fresh reruns; those artifacts are retained as
  diagnostic evidence and superseded. The final five selectors ran separately
  through the configured real LLM paths and passed human inspection:
  (1) literal speech only in dialog while physical staging remained exclusively
  in the private terminal visual branch; (2) a direct evidence-based inference
  rather than an ask-back; and (3) current-only time scope without an invented
  future rule or unrelated topic. Each case used one dialog generation and one
  compliance call with no repair. The real verifier also rejected bracketed and
  plain-prose physical narration in independently inspected negative probes.
  The accurate final review is
  `test_artifacts/llm_traces/dialog_visible_speech_and_semantic_fidelity_review.md`.
- Independent review: initially not approved on 2026-07-17. Findings were:
  terminal image fragments reached the consolidation router LLM; the inference
  case transformed a source descriptor and was falsely accepted; real negative
  verifier evidence was absent; and dialog imposed an unmatched 16-percept cap.
  Focused red coverage was added for terminal projection, general descriptor
  fidelity, and reuse of the shared episode size bound, plus two one-at-a-time
  real-verifier selectors. Corrected red baseline: 3 failed and 6 passed. The
  failures prove the prompt contract omits descriptor fidelity, terminal visual
  fragments enter the consolidation projection, and a valid compact 17-percept
  episode fails only at dialog. The first remediation passed focused and
  adjacent deterministic gates. Fresh case 3 final dialog preserved scope, but
  its style output still suggested adding a concrete topic detail; a general
  lexical/cadence-only prompt regression was added and must be remediated before
  completing the five live selectors. That remediation passed 9/9. The next
  fresh case 1 then broadened an explicit target into an unrestricted referent
  and the verifier falsely accepted it. The general entity/target-specificity
  remediation passed 9/9 and the next fresh cases 1-2 passed. Fresh case 3 then
  added an unsupported future stance to a current-only decision and the
  verifier falsely accepted it. The general current-only/future-silence
  remediation passed 9/9. Final fresh evidence then passed all five selectors:
  literal dialog preserved the explicit target, inference preserved the source
  descriptor, current-only meaning introduced no future content, and the real
  verifier rejected both bracketed and plain-prose physical narration. The
  rewritten review is
  `test_artifacts/llm_traces/dialog_visible_speech_and_semantic_fidelity_review.md`.
  Socrates' final rereview approved Checkpoint F with no blocking, high, medium,
  or actionable low findings. It confirmed exact terminal-image projection
  exclusion, near-miss preservation, raw-voice isolation, lexical/cadence-only
  style ownership, descriptor/referent/time-scope fidelity, both real narration
  rejections, shared percept bounds, unchanged one-repair topology, and clean
  anti-cheat boundaries. The superseding full non-live regression passed 3134,
  skipped 2, deselected 645, with one third-party deprecation warning in
  240.81 seconds.
- Frozen proof attempt 1: stopped at turn 6 on 2026-07-17. Turns 1-5 passed
  human inspection. Turn 6 contained literal affectionate acceptance but added
  an unsupported future prohibition to a current request. The L2 cognition
  graph remained current-only (`just this once`), proving the drift appeared
  downstream. The protected trace export was in metadata capture mode, so it
  established stage order and false verifier acceptance but did not expose raw
  L3/generator prompts. Root cause: the verifier compared percept, surface, and
  candidate without explicitly making canonical current percepts authoritative
  over a possibly drifted surface. The service was stopped at PID 34916, port
  8012 cleared, and guarded database state restored exactly to digest
  `dfb9e43b99ddf0ef6f9ca92e6e823513f82fc5d11a361a8f1c8348d273badc5e`.
- Source-precedence remediation: signed targeted live evidence 2026-07-17.
  `current_visible_percepts` are now explicit semantic authority; the text
  surface and candidate are proposals audited against them. One new real
  verifier selector supplied a current-only percept plus a self-consistent
  drifted surface/candidate containing a future prohibition. The real verifier
  returned `aligned=false` for the unsupported future rule. The other five
  selectors were regenerated under the same final verifier prompt and passed
  separate human inspection. Checkpoint F rereview and superseding full
  regression remain pending. The superseding full non-live run then passed
  3134, skipped 2, deselected 646, with one third-party deprecation warning in
  222.29 seconds. Focused rereview required one additional positive real case
  proving that authoritative source-required future content remains aligned;
  the separately executed real verifier returned `aligned=true` with no issues
  for the supplied next-meeting reminder and exact umbrella target. The
  timestamped final-prompt narration artifacts are now referenced explicitly,
  normal Python file termination is restored, and Checkpoint F rereview is
  complete. Socrates approved with no blocking, high, medium, or actionable low
  findings and confirmed unchanged one-generator/one-verifier/at-most-one-repair
  topology, no deterministic semantic cleaner/classifier, correct rejection of
  unsupported future drift, and correct acceptance of source-required future
  content.
- Frozen proof attempt 2: stopped at turn 8 on 2026-07-17. Valid turns 1-7
  contained no action narration; the earlier turn-6 future-rule failure was
  corrected. Turn 8 generator output ended with a stray unmatched `】`, and the
  verifier falsely returned aligned. Full trace capture confirmed the generator
  introduced the token and the verifier accepted it. This is visible markup
  residue rather than action narration, but it fails the requested visible-text
  quality bar. PID 5352 was stopped, port 8012 cleared, and guarded database
  state restored exactly before remediation. A generic LLM-owned unmatched
  enclosure/markup-residue contract and real negative probe are pending; no
  deterministic stripping or mutation is permitted.
- Unmatched-enclosure remediation: signed targeted live evidence 2026-07-17.
  Generator and verifier prompts generically reject visible markup residue,
  stage-direction delimiters, and unmatched enclosing punctuation. The real
  verifier returned `aligned=false` and named the residue for the dedicated
  candidate. The other seven selectors were regenerated under the same final
  prompt: three generation cases passed human review, both narration forms and
  self-consistent future drift were rejected, and source-required future
  content remained aligned. The superseding full non-live regression passed
  3134, skipped 2, deselected 648, with one third-party deprecation warning in
  219.86 seconds. Socrates approved Checkpoint F with no findings after
  confirming generic LLM-owned enforcement, unchanged one-generator/
  one-verifier/at-most-one-repair topology, accurate eight-case evidence, and
  clean anti-cheat plus diff gates.
- Frozen proof and restoration: pending.
- Frozen proof attempt 4: stopped at turn 10 on 2026-07-17. The valid visible
  dialog contained no action narration, but its closing sentence broadened
  permission for the user's one specific requested act into unrestricted
  permission. Full trace capture showed the content surface invited a vague
  transfer of control, the generator widened the scope, and the verifier
  falsely returned aligned despite canonical current-percept grounding. The
  same attempt also exposed a proof-integrity issue outside the dialog fix:
  intermittent route-selector shape failures produce a visible busy fallback
  which is persisted into later chat history, so zero cognition-state changes
  do not make an in-attempt retry clean. Attempt 4 was stopped at PID 47556,
  port 8012 was cleared, and the guarded database was restored exactly to
  digest `dfb9e43b99ddf0ef6f9ca92e6e823513f82fc5d11a361a8f1c8348d273badc5e`.
  The permission-scope correction stays inside the approved content/dialog/
  verifier change surface. Any route-selector production change remains
  outside this plan and requires an explicit scope decision.
- Permission-scope remediation: verified 2026-07-17. A focused prompt contract
  first failed 1/9 because content planning, dialog generation, and compliance
  lacked an explicit general rule keeping acceptance, refusal, permission, and
  consent bounded to the exact source-requested act and scope. All three now
  prohibit indefinite or unrestricted permission from substituting for a
  specific permission. The focused suite passes 9/9. Nine real-LLM selectors
  were regenerated separately and inspected under the final prompts: three
  generation cases preserve literal speech, inference, and current-only scope;
  five negative verifier cases reject both narration forms, self-consistent
  future drift, unmatched enclosure residue, and unrestricted permission
  drift; one positive control preserves source-required future content. The
  accurate review is
  `test_artifacts/llm_traces/dialog_visible_speech_and_semantic_fidelity_review.md`.
  Adjacent deterministic integration passed 32 with 4 deselected; compilation,
  runtime prompt rendering, anti-capture, anti-cleaner, and `git diff --check`
  passed. The superseding full non-live regression passed 3134, skipped 2,
  deselected 649, with one third-party deprecation warning in 220.48 seconds.
  No captured replay/intimate phrase, deterministic filter, response mutation,
  extra call, retry, schema, compatibility path, or visual downstream consumer
  was introduced. Socrates approved F2 with no blocking, high, medium, or
  actionable low findings after the factual-plan audit. The reviewer confirmed
  the generic act/scope contract, unrelated live near-miss, unchanged dialog
  call topology, anti-cheat boundary, and terminal visual ownership.
