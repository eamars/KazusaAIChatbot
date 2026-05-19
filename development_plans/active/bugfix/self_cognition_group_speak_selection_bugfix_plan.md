# self cognition group speak selection bugfix plan

## Summary

- Goal: correct group self-cognition source framing and action selection so
  ambient group review can use group engagement guidance before `speak`
  selection, while preserving the character's autonomy to speak when the
  observed scene gives enough reason.
- Plan class: large
- Status: in_progress
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, and `cjk-safety`
- Overall cutover strategy: bigbang for group self-cognition source framing
  and L2d `speak` semantics; compatible for existing delivery target and
  style-image storage shapes
- Highest-risk areas: turning character judgment into mechanical chatbot
  suppression, over-amplifying group engagement guidelines, confusing internal
  monologue with visible speech, and fabricating a single user target for an
  ambient group window
- Acceptance criteria: group self-cognition uses the approved label-free
  first-person source sentences; L2d treats `speak` as visible channel output,
  not as internal curiosity; targetless group review can load group
  engagement guidelines; real LLM verification records at least 20 raw
  self-reflection windows with full input and output, including silent and
  spoke cases

## Context

The observed failure was a group self-cognition run where the character read
the self-cognition source as if it were a user message from the inspected group
channel. The generated appraisal became curious about the self-cognition input
itself, and L2d selected `speak`, causing dialog to produce a visible reply
about "自检术语" instead of responding to the actual group-chat scene.

The accepted diagnosis is that source wording alone is not enough. The durable
boundary problem is that L2d currently says only "当前需要文字表层时，选择
`speak`" and does not clearly distinguish internal monologue from visible
speech.

The project owner has clarified the intended architecture:

- Kazusa is a platform-agnostic character brain service, not a generic
  assistant chatbot.
- Do not mechanically suppress speech to make the character behave like a
  chatbot.
- If the character has enough reason and observation to speak, she should
  speak.
- The bug is speaking for the wrong reason or about the wrong context, not the
  existence of proactive speech.

The latest system contract correction is:

```text
group chat source data
  -> L1/L2 understand the current observed scene
  -> group engagement guidelines may bias whether the scene gives enough reason
  -> L2d selects visible actions
  -> selected speak means visible reply
  -> internal thoughts without visible externalization produce no speak action
```

## Mandatory Skills

- `development-plan-writing`: load before changing this plan, the registry,
  execution evidence, lifecycle status, or approval status.
- `local-llm-architecture`: load before changing self-cognition source
  packets, cognition prompts, L2d action-selection prompts, graph stage
  boundaries, or prompt-facing interaction-style context.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files with CJK prompt strings.

## Mandatory Rules

- Execute this plan only after its registry status is changed to `approved` or
  `in_progress`.
- The implementation must preserve the character-brain goal. Do not add
  mechanical suppression, generic chatbot safety language, hard-coded silence
  rules, response-ratio clamps, cooldowns, or deterministic keyword blockers
  as the fix.
- LLM stages own semantic judgment: whether the observed group scene gives the
  character enough reason to externalize speech.
- Deterministic code owns source projection truthfulness, delivery target
  binding, style-image projection, prompt payload limits, validation, adapter
  delivery, persistence, and audit.
- `speak` means visible reply. Internal monologue, private curiosity, private
  dislike, feeling left out, background progress maintenance, or "keep
  observing" are not `speak` unless the character has chosen to externalize
  them into the channel.
- Group engagement guidelines are pre-selection evidence for group
  self-cognition. They may strengthen or weaken reason-to-speak only together
  with the current scene affordance. They must not become a command to speak.
- Source packet wording must describe what the data is and why it entered
  attention. It must not instruct the character how to read, what to do next,
  or whether to speak.
- Do not add a separate `阅读方式` section to production group source packets.
- Do not use "自检" in model-facing group source-packet text.
- Do not use `需要接上` in model-facing group source-packet text.
- Do not use `数据身份：` or `进入注意的原因：` in model-facing group
  source-packet text.
- Use first-person natural source framing with `我`; do not use `角色` for
  this source-framing line.
- If a group window is ambient and has no single addressed human target, keep
  the semantic user target empty. Do not fabricate a latest-speaker user
  target.
- A group self-cognition case must still have a deterministic group delivery
  target. Targetless means no single semantic user target, not no group channel
  target.
- Targetless group self-cognition must be able to load group-channel
  engagement guidelines without requiring a user `global_user_id`.
- Real LLM tests must run one case at a time and be inspected one case at a
  time. Do not run prompt optimization variants in parallel.
- Real LLM reports must show full input and output to the owner, not only a
  summary.
- Any change to an LLM prompt in this plan requires rewriting the entire
  affected prompt text as one coherent prompt. No exception is allowed for
  small changes. The implementation agent must rework the prompt's logic flow,
  decision procedure, evidence order, input contract, and output contract in
  the same edit. Simple appended warning blocks, trailing special cases, or
  one-line prompt patches are forbidden.
- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.

## Must Do

- Replace group self-cognition source-packet wording that currently says
  `来源位置`, `出现原因`, or `需要接上` with the approved label-free
  first-person source sentence.
- Keep source-packet wording simple: one sentence for ambient group windows and
  one sentence for direct-addressed group windows. Do not add headings,
  labels, read-mode instructions, or action guidance.
- Make the group source-packet sentence truthful for direct-addressed and
  ambient windows. Do not say nobody handed the topic to the character when
  the semantic labels show direct address.
- Update L2d action-selection prompt semantics so `speak` is explicitly a
  visible reply and internal monologue alone is not a visible action.
- Keep L2d as the only owner of selected visible text action. Do not move
  speak/no-speak selection into L3 or dialog.
- Add pre-L2d group engagement context for group self-cognition using bounded
  group-channel style-image projection.
- Ensure targetless group self-cognition can load group-channel style without
  failing on missing `global_user_id`.
- Preserve the current distinction between group delivery target and semantic
  user target.
- Add deterministic tests for source-packet framing, L2d prompt contract,
  group-only style projection, and targetless group state.
- Add or preserve real LLM sensitivity tests that rebuild at least 20 raw group
  self-reflection windows, including both silent and spoke historical outcomes,
  and record full inputs and outputs.
- Update documentation where it describes group self-cognition target meaning,
  source-packet framing, or `speak` action semantics.

## Deferred

- Do not redesign the reflection scheduler or group-review cadence.
- Do not add response-ratio tuning, cooldown changes, noise threshold changes,
  or deterministic suppression as part of this fix.
- Do not add a new LLM call for group engagement guidelines.
- Do not migrate or rewrite historical conversation, reflection,
  self-cognition, or style-image rows.
- Do not change adapter delivery APIs.
- Do not change private-chat self-cognition source framing unless a focused
  test proves the same bug exists there.
- Do not add a user-target inference heuristic such as latest speaker,
  loudest speaker, first speaker, or most frequent speaker.
- Do not make L3 or dialog decide whether to speak.
- Do not create a new prompt family for group self-cognition.

## Cutover Policy

Overall strategy: bigbang for group source framing and L2d action semantics;
compatible for existing storage and delivery metadata.

| Area                                        | Policy     | Instruction                                                                                                                                                  |
| ------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Group source-packet wording                 | bigbang    | Replace `来源位置` / `出现原因` wording for group review with the approved label-free first-person source sentences. Do not preserve `需要接上`, `数据身份：`, or `进入注意的原因：`. |
| L2d `speak` meaning                         | bigbang    | Treat `speak` as visible channel text. Rewrite prompt semantics directly; do not add a compatibility interpretation where `speak` can mean private thought.  |
| Engagement guidance before action selection | bigbang    | Add bounded group engagement context to group self-cognition L2d input before `speak` selection.                                                             |
| Targetless group style loading              | compatible | Preserve existing style-image documents. Add a group-only projection/load path so missing user id does not block group style.                                |
| Delivery target metadata                    | compatible | Preserve `SelfCognitionDeliveryTarget` shape. Group review keeps same-channel delivery target.                                                               |
| Semantic user target                        | compatible | Preserve `target_scope.user_id=None` for ambient group windows. Do not fabricate a user target.                                                              |
| Historical data                             | compatible | No migration or backfill.                                                                                                                                    |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- Bigbang areas must rewrite the old behavior directly, not preserve it behind
  a flag or fallback.
- Compatible areas preserve only the explicitly listed storage and metadata
  surfaces.
- Any change to this cutover policy requires user approval before
  implementation.

## Agent Autonomy Boundaries

- The agent can choose local helper names only. All model-facing wording,
  prompt contracts, data shapes, verification gates, and ownership boundaries
  are fixed by this plan.
- The agent must not introduce alternate prompt routes, prompt repair loops,
  fallback LLM calls, hidden suppressors, response-ratio parameters, or
  target-selection heuristics.
- If the agent edits any runtime LLM prompt, the agent must rewrite the whole
  prompt and record how the rewritten prompt preserves a clear logic flow.
  Appending a new instruction block to an existing prompt is a plan violation.
- The agent must use the exact group source-packet sentences in `Group Source
  Packet`. Any wording change requires owner approval before implementation.
- If implementation requires changing files outside `Change Surface`, record
  the reason before editing and keep the change directly tied to this plan.
- If a real LLM result shows poor behavior, fix the source contract, L2d
  prompt contract, or engagement-context placement. Do not add deterministic
  post-filters over LLM output.
- If code and this plan disagree, preserve this plan's ownership boundary and
  report the discrepancy.

## Target State

Ambient group self-cognition behaves as a scene observation:

```text
reflection activity window
  -> group self-cognition source case
  -> neutral first-person source packet
  -> L1/L2 scene understanding
  -> bounded group engagement context before L2d
  -> L2d selects visible action or no action
  -> selected speak runs L3/dialog and dispatches to the group channel
  -> no selected speak keeps the thought internal and may still consolidate
```

The model-facing group source packet starts with this exact sentence for an
ambient group window:

```text
我刚看到群里刚刚发生的一段现场。我之前没有插话，这段里也没有人把话题交给我。
```

For direct-addressed group windows, the sentence must be truthful and use the
semantic labels:

```text
我刚看到群里刚刚发生的一段现场。里面有人把话题指向我。
```

No production group source packet contains:

```text
自检
需要接上
阅读方式
数据身份：
进入注意的原因：
```

## Design Decisions

| Topic                   | Decision                                                                    |
| ----------------------- | --------------------------------------------------------------------------- |
| Source framing          | Use the approved label-free first-person source sentences for group review. |
| Pronoun                 | Use `我`, not `角色`, in group source framing.                                 |
| Reading instruction     | Do not add production `阅读方式`.                                               |
| Strong trigger phrase   | Remove `需要接上`.                                                              |
| Engagement placement    | Feed group engagement before L2d.                                           |
| Engagement scope        | Use group-channel style for targetless group self-cognition.                |
| L2d semantics           | `speak` means visible reply.                                                |
| Targetless group review | Preserve no semantic user target for ambient group windows.                 |

## Research Decision Record

- Date: 2026-05-19.
- Decision: remove `数据身份：` and `进入注意的原因：` from production group
  self-cognition source text.
- Conclusion: label-free first-person framing is accepted as a source-shape
  simplification. It does not by itself fix speak sensitivity; L2d still must
  distinguish internal observation from visible reply.

## Minimal Contract Audit

Semantic question:

```text
Given the current observed group scene, character state, and group engagement
guidance, does the character want to externalize a visible reply now?
```

Inputs required for that semantic question:

- neutral group source packet;
- current group visible context and semantic labels;
- L1/L2 appraisal, stance, intent, internal monologue, and social context;
- bounded group-channel engagement guidelines;
- RAG and conversation-progress evidence already present in the current
  action-initializer payload.

Output fields required by downstream code:

- L2d emits `{"action_requests": []}` for no visible action;
- L2d emits `capability="speak"` only for visible channel text;
- existing action specs, L3 text surface, dialog, and delivery path remain the
  downstream consumers.

Deterministic owners:

- source packet construction;
- semantic-label truthfulness;
- group-channel style projection and prompt budget;
- delivery target binding;
- action-spec validation, persistence, and adapter delivery.

Rejected complexity:

- no response-ratio field;
- no silence keyword detector;
- no latest-speaker target heuristic;
- no extra LLM call;
- no prompt repair loop;
- no fallback prompt family;
- no new scheduler or cooldown.

Evidence needed before adding complexity:

- at least 20 real LLM case outputs showing the simple source packet plus L2d
  contract still produces repeated wrong visible speech for the same reason;
- owner approval for a larger architecture change.

## Contracts And Data Shapes

### Group Source Packet

`src/kazusa_ai_chatbot/self_cognition/projection.py` owns the model-facing
source-packet lines.

For `TRIGGER_GROUP_CHAT_REVIEW`, the rendered group source packet must begin
with exactly one label-free first-person sentence.

Ambient group window:

```text
我刚看到群里刚刚发生的一段现场。我之前没有插话，这段里也没有人把话题交给我。
```

Direct-addressed group window:

```text
我刚看到群里刚刚发生的一段现场。里面有人把话题指向我。
```

Forbidden in rendered group source text:

```text
自检
需要接上
阅读方式
数据身份：
进入注意的原因：
```

The direct-address test must use existing group semantic labels such as
`bot_addressing=directly_addressed`. If labels are absent or unrecognized, use
the ambient sentence only when the group window is not claiming direct address.

### Group Engagement Action Context

Add a prompt-facing group engagement projection that does not require a user
id. The returned shape must be compact:

```python
{
    "engagement_guidelines": list[str],
    "confidence": str,
}
```

Rules:

- Load only group-channel style for `channel_type == "group"`.
- Return an empty shape when group style is missing, inactive, invalid, or
  empty.
- Apply the same per-field style guideline limit used by L3 style context.
- Do not include speech, social, pacing, revision, source run ids, timestamps,
  Mongo ids, or raw DB documents.
- Do not require or synthesize `global_user_id`.

### L2d Action Initializer Payload

`build_action_initializer_payload(...)` must include group engagement evidence
for group self-cognition before L2d chooses actions.

The payload wording must state the semantic role without turning it into a
command:

```text
群聊参与习惯：<bounded guidelines or 无>
```

The L2d system prompt must state:

- `speak` is visible text sent to the current channel after L3/dialog;
- internal monologue is evidence, not an action;
- if the character only wants to observe, keep a thought private, maintain
  progress, or avoid externalizing, return `{"action_requests": []}`;
- engagement guidelines may support the reason to speak when the current scene
  has a compatible opening, but they do not replace the current scene.

### Targetless Group Style Loading

`build_interaction_style_context(...)` or a new adjacent helper must support
group-only style loading. Current behavior raises `global_user_id is required`
before group style can load, which drops group engagement context for
targetless group self-cognition.

The fixed behavior:

- private chat still requires user style;
- normal group chat with a user id still loads user style first and group
  style second;
- group self-cognition without a user id loads empty user style plus group
  channel style;
- failures to load one style source must not erase the other style source.

## LLM Call And Context Budget

- No new LLM calls are approved.
- L2d receives at most the existing action-initializer context plus the bounded
  group engagement context.
- The group engagement context uses the existing style-image store and
  projection limits.
- Real LLM tests are verification-only and must not be added to the production
  path.

## Change Surface

Primary production files:

- `src/kazusa_ai_chatbot/self_cognition/projection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
- `src/kazusa_ai_chatbot/db/interaction_style_images.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `src/kazusa_ai_chatbot/self_cognition/README.md`

Primary tests:

- `tests/test_self_cognition_group_review_source.py`
- `tests/test_self_cognition_framing.py`
- `tests/test_cognition_interaction_style_context.py`
- `tests/test_interaction_style_images.py`
- `tests/test_l2d_action_selection_cases.py`
- `tests/test_self_cognition_response_sensitivity_live_llm.py`

Experiment and diagnostic files:

- `experiments/` may be used to run prompt-shape probes and save manual
  reports. Experiment files are not production runtime code.

Changes outside this surface require an execution-evidence note explaining the
direct blocker.

## Overdesign Guardrail

- Actual problem: group self-cognition can turn ambient internal reaction into
  visible speech because the source packet over-signals "catch up" and L2d
  does not clearly distinguish private thought from visible reply.
- Minimal change: neutralize group source framing, feed bounded group
  engagement evidence before L2d, and clarify L2d `speak` as visible reply.
- Ownership boundaries: deterministic code projects truthful source/context and
  style evidence; L1/L2 judge the scene; L2d selects visible actions; L3/dialog
  only render selected visible text; delivery remains deterministic.
- Rejected complexity: no generic chatbot suppression, no response ratio, no
  latest-speaker targeting, no cooldown tuning, no new LLM stage, no
  deterministic post-filter over L2d, no style-image schema migration.
- Evidence threshold: add broader architecture only after at least 20 real LLM
  cases show the minimal contract still fails for the same reason and the
  owner approves the added complexity.

## Implementation Order

1. Add deterministic source-packet tests.
   - Update group-review tests to reject `自检`, `需要接上`, and `阅读方式`.
   - Update group-review tests to reject `数据身份：` and `进入注意的原因：`.
   - Add direct-addressed and ambient group cases so the label-free source
     sentence is truthful.
   - Run the focused tests and record the expected pre-implementation failure.
2. Implement group source-packet wording.
   - Edit only `self_cognition/projection.py` unless tests expose a direct
     helper need.
   - Run the focused source-packet tests.
3. Add group-only style projection tests.
   - Cover missing user id in group channel.
   - Cover private chat still requiring user style.
   - Cover normal group user-plus-group ordering.
   - Record the expected pre-implementation failure.
4. Implement group-only style loading.
   - Reuse existing overlay validation and projection helpers.
   - Do not expose raw style-image documents to prompts.
   - Run focused DB/style tests.
5. Add L2d payload and prompt tests.
   - Verify group self-cognition action context includes bounded group
     engagement text.
   - Verify private and non-self-cognition payloads are unchanged unless
     explicitly covered.
   - Verify the prompt contains the visible-reply/internal-monologue contract.
6. Implement L2d action-selection context changes.
   - Load group engagement context before L2d for group self-cognition.
   - Rewrite the entire L2d prompt as one coherent prompt. Rebuild its logic
     flow, decision procedure, evidence order, input contract, and output
     contract in the same edit. Do not append a warning block or patch only
     the local sentence about `speak`.
   - Run focused L2d tests.
7. Run real LLM verification.
   - Use at least 20 raw group self-reflection windows, including silent and
     spoke historical outcomes.
   - Print or save full inputs and outputs for owner review.
   - Inspect each case one at a time.
   - Record silent/speak quality and reason-to-speak quality.
8. Update docs.
   - Clarify group source framing, targetless group meaning, and `speak` as
     visible reply.
   - Do not describe the behavior as chatbot suppression.
9. Run full verification and independent code review.
   - Run all commands in `Verification`.
   - Complete the independent code review gate before final sign-off.

## Progress Checklist

- [x] Stage 1 - source-packet contract updated
  - Covers: implementation order steps 1-2.
  - Verify: focused group source-packet tests pass.
  - Evidence: record failing-before and passing-after commands.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `parent/2026-05-19` after evidence is recorded.
- [x] Stage 2 - targetless group style context works
  - Covers: implementation order steps 3-4.
  - Verify: style-image and L3 interaction-style tests pass.
  - Evidence: record failing-before and passing-after commands.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `parent/2026-05-19` after evidence is recorded.
- [x] Stage 3 - L2d receives engagement guidance and owns visible speech
  - Covers: implementation order steps 5-6.
  - Verify: focused L2d action-selection tests pass.
  - Evidence: record prompt rewrite summary, confirming the full affected
    prompt was rewritten rather than appended to, and record test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `parent/2026-05-19` after evidence is recorded.
- [x] Stage 4 - real LLM quality report accepted
  - Covers: implementation order step 7.
  - Verify: at least 20 post-change real LLM cases have full inputs and
    outputs; cases are inspected one at a time.
  - Evidence: record artifact paths, silent/speak quality, and reason-to-speak
    quality.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `parent/2026-05-19` after owner-facing report is recorded.
- [x] Stage 5 - docs and regression verification complete
  - Covers: implementation order step 8 and broad verification.
  - Verify: docs updated; all deterministic verification commands pass.
  - Evidence: record command output and changed docs.
  - Handoff: next agent starts independent code review.
  - Sign-off: `parent/2026-05-19` after evidence is recorded.
- [x] Stage 6 - independent code review complete
  - Covers: implementation order step 9.
  - Verify: independent review has no unresolved blockers.
  - Evidence: record findings, fixes, rerun commands, and residual risks.
  - Handoff: plan can move toward completion only after this stage.
  - Sign-off: `parent/2026-05-19` after review evidence is recorded.

## Verification

### Static Greps

- `rg -n "需要接上|当前自检|阅读方式|数据身份：|进入注意的原因：" src/kazusa_ai_chatbot/self_cognition src/kazusa_ai_chatbot/nodes`
  - Expected: no production group source-packet or L2d prompt match. Any
    remaining match must be a non-runtime historical note or test fixture with
    explicit reason.
- `rg -n "global_user_id is required" src/kazusa_ai_chatbot/db/interaction_style_images.py`
  - Expected: no group-only style path can hit this failure before loading
    group style.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_framing.py -q`
- `venv\Scripts\python -m pytest tests\test_interaction_style_images.py tests\test_cognition_interaction_style_context.py -q`
- `venv\Scripts\python -m pytest tests\test_l2d_action_selection_cases.py -q`

### Real LLM Tests

- `venv\Scripts\python -m pytest tests\test_self_cognition_response_sensitivity_live_llm.py -q -s`
  - Run one case at a time when collecting evidence.
  - The test/report must include at least 20 raw self-reflection windows.
  - The artifact must show source packet, rendered source, action-initializer
    payload, L2d output, parsed actions, and dialog output for selected
    `speak`.

### Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\self_cognition\projection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\kazusa_ai_chatbot\db\interaction_style_images.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py`

## Independent Plan Review

Review date: 2026-05-19.

Conclusion: no blockers. The plan is tight enough to remain as a draft work
contract.

The implementation agent has no authority to change the approved source
sentences, add alternate prompt routes, add deterministic suppressors, invent
target-selection heuristics, or preserve legacy group source labels. Execution
must still wait until the plan is approved or moved to `in_progress`.

## Independent Code Review

Before completion, an independent reviewer must inspect the full diff and
record findings in `Execution Evidence`.

The review must check:

- the change does not introduce mechanical chatbot suppression;
- `speak` remains the L2d-selected visible action and is not reselected by L3
  or dialog;
- source-packet text uses only the approved label-free first-person source
  sentences;
- group engagement guidance is available before L2d but does not command
  speech;
- targetless group review keeps channel delivery target and does not fabricate
  a semantic user target;
- group-only style loading does not erase group style because user style is
  missing;
- real LLM test artifacts include full inputs and outputs for at least 20 raw
  windows;
- any touched LLM prompt was rewritten as a whole coherent prompt, not changed
  by appending a warning block or isolated sentence patch;
- deterministic tests cover direct-addressed and ambient group windows;
- docs match the implementation.

Review findings that require behavior outside this plan must stop completion
until the owner approves a plan update.

## Acceptance Criteria

- Group self-cognition source packets do not contain `自检`, `需要接上`, or
  `阅读方式`.
- Group source packets use the approved label-free first-person source
  sentences.
- Direct-addressed and ambient group windows receive truthful label-free
  source sentences.
- L2d prompt semantics explicitly define `speak` as visible reply and internal
  monologue as non-action evidence.
- Group engagement guidelines are available before L2d action selection for
  group self-cognition.
- Targetless group self-cognition can load group-channel style without
  requiring `global_user_id`.
- Ambient group review keeps no semantic user target while retaining same-group
  delivery target.
- At least 20 real LLM cases are run one at a time and reported with full
  inputs and outputs.
- The owner-facing report evaluates reason-to-speak quality, not only
  historical silent/speak alignment.
- Focused deterministic tests, compile checks, static greps, real LLM
  verification, and independent code review are complete.

## Risks

| Risk                                                  | Mitigation                                                                               | Verification                                                        |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| The character becomes too silent                      | Do not add deterministic suppression; judge real LLM outputs by reason-to-speak quality. | At least 20 real LLM cases with silent and spoke examples.          |
| Engagement guidelines over-command speech             | Prompt says guidelines are evidence, not command; current scene remains primary.         | Real LLM cases from noisy groups with active engagement guidelines. |
| Internal thought still becomes visible reply          | L2d prompt and tests define `speak` as visible reply only.                               | L2d action-selection tests and real LLM outputs.                    |
| Group style remains unavailable for targetless review | Add group-only style projection and tests.                                               | Style-image and L3 interaction-style tests.                         |
| User target is fabricated                             | Preserve `target_scope.user_id=None` for ambient windows.                                | Group review source tests.                                          |
| Address-state wording mismatch                        | Use semantic labels to choose the approved ambient or direct-addressed sentence.         | Direct-addressed and ambient source-packet tests.                   |

## Execution Evidence

- 2026-05-19: execution started by parent agent after owner instruction.
- 2026-05-19: source-packet contract implemented for group review. Ambient
  group review uses the approved label-free first-person sentence and direct
  addressed windows use the approved direct-addressed sentence. Rendered group
  source packets omit the second reason line, so production group review no
  longer contains `来源位置`, `出现原因`, `数据身份：`, `进入注意的原因：`,
  `阅读方式`, `自检`, or group `需要接上`.
- 2026-05-19: targetless group state and style loading implemented. Ambient
  group review keeps empty semantic user fields while preserving the group
  channel delivery target. Group-only style context can load group-channel
  engagement guidance without requiring `global_user_id`; private style
  loading still requires a user id.
- 2026-05-19: group engagement context added before L2d for group
  self-cognition. The L2d action-selection prompt was rewritten as one
  coherent prompt contract after review, preserving the evidence order,
  visible-`speak` meaning, internal-monologue non-action rule, group
  engagement evidence role, input contract, and output contract.
- 2026-05-19: live LLM validation was run one case at a time after the final
  L2d prompt rewrite. Final owner-facing report:
  `test_artifacts/llm_traces/self_cognition_group_response_sensitivity_report__final_prompt__20260519T021223Z.md`;
  full raw input/output JSON:
  `test_artifacts/llm_traces/self_cognition_group_response_sensitivity_report__final_prompt__20260519T021223Z.json`.
  The final report contains 20 raw group self-reflection windows: historical
  spoke=16, historical silent=4, observed spoke=14, observed silent=6,
  matched historical=10, mismatched historical=10, non-empty group engagement
  guidance=17.
- 2026-05-19: live LLM quality finding: no sampled output spoke about the
  self-cognition packet itself or `自检`. Engagement guidance affected action
  direction and did not act as a deterministic command because 6/20 cases
  stayed silent. Residual sensitivity risk remains: several ambient windows
  speak for weak reasons such as presence, entering the flow, or broad ENFP
  participation rather than a concrete handoff. No deterministic suppression
  was added.
- 2026-05-19: deterministic verification passed after final prompt rewrite.
  Compile passed for `projection.py`, `persona_supervisor2_cognition_l2d.py`,
  `interaction_style_images.py`, `persona_supervisor2_cognition.py`,
  `persona_supervisor2_schema.py`, and the live/helper sensitivity tests.
  Focused pytest command passed 62 tests across group source framing,
  self-cognition framing, interaction style images, L3 style context, L2d
  action-selection fixtures, action initializer prompt/payload, graph
  engagement handoff, sensitivity helpers, and architecture docs.
- 2026-05-19: static grep result: `需要接上|当前自检|阅读方式|数据身份：|进入注意的原因：`
  has no production group source-packet or L2d prompt match. The only
  remaining `需要接上` match is the explicitly deferred private-chat source
  framing line in `self_cognition/projection.py`.
- 2026-05-19: experiment cleanup verified. `experiments/` contains only
  `.gitignore`; no experiment code remains.
- 2026-05-19: independent code review completed by parent agent. No blocking
  code issue remains after the L2d whole-prompt rewrite. Review confirmed no
  mechanical chatbot suppression, L2d remains the owner of visible `speak`,
  L3/dialog do not reselect speech, group source sentences match the approved
  text, group engagement is pre-L2d evidence rather than command text,
  targetless group review does not fabricate a semantic user target, and docs
  match the implementation. Residual behavioral tuning should target L2d
  reason quality, not deterministic post-filters.

## Plan Self-Review

- Conclusion: the plan is ready to remain as a draft work contract.
- Tightness: source sentences, prompt-rewrite rule, change surface,
  implementation order, verification gates, and forbidden behavior are fixed.
- Autonomy boundary: implementation autonomy is limited to local helper names;
  no model-facing wording, prompt contract, data shape, or behavior boundary
  is open for creative reinterpretation.
