# Cognition Core V2 Intra-Turn Transition Quality Handover

## Document Status

- **Status:** Reference handover
- **Source branch inspected:** `main`
- **Source revision:** `8f834bf87a83ee42aca804934fb44af63788420c`
- **Target branch inspected:** `origin/cognition_core_v2`
- **Target revision:** `ba433e9be4805bb88b992419db1c99a6e376dc4a`
- **Intended receiver:** Agent responsible for cognition core v2 quality hardening
- **Primary defect:** Forced tsundere polarity flip with a causally empty bridge
- **Related target plan:** `development_plans/active/bugfix/cognition_core_v2_baseline_regression_hardening_plan.md`

This document transfers evidence and requirements. It is not an executable
development plan. Before changing production code, the receiving agent must
reconcile this handover with the current target-branch revision and either:

1. add the work to an active, approved or in-progress bugfix plan whose scope
   and lifecycle still cover it; or
2. create and approve a dedicated active bugfix plan.

Do not append new scope to a completed plan.

## Executive Summary

Kazusa's dialog frequently begins with literal rejection, accusation, or
denial and then changes to acceptance within the same turn. The transition is
often connected only by a discourse shell such as `既然你都这么说了`,
`既然你都问了`, `但是`, or `那也没办法`. Those phrases connect clauses
grammatically but do not supply a believable cause for the change of stance.

The primary quality failure is therefore not the presence of tsundere
language. It is an unsupported semantic reversal:

```text
literal rejection or accusation
    -> no new fact, motive, condition, or constraint
    -> acceptance or compliance
```

The current-main evidence shows that cognition can choose a coherent accepting
or playful stance while an upstream surface-style instruction independently
asks dialog to begin with denial and then rapidly comply. Dialog follows both
inputs and produces the abrupt reversal. A matched same-model ablation that
changed only the style instruction removed the sharp flip.

Cognition core v2 remains exposed to the same class of defect. Its style and
content stages run independently in parallel, then merge without a semantic
compatibility check. Character defense cues are supplied to the style stage,
while the selected intention and content plan are owned elsewhere. The v2
repair path also preserves the rejected `style_guidance`, even when style may
be the source of the hard issue.

The primary fix belongs at the v2 surface-planning boundary that produces and
merges content and style. Dialog-level verification is a secondary safety net.
Persistence must also avoid turning a bad generated reversal into desired
future progression.

## User-Confirmed Quality Boundary

The receiving agent must preserve these scope decisions:

- Judge the spoken dialog itself.
- Treat bracket-encapsulated content neutrally. This handover neither
  encourages nor forbids brackets.
- Treat repetitive wording such as `真的……非常开心` as a separate,
  lower-priority concern.
- Preserve Kazusa's recognizable personality, embarrassment, hesitation,
  defensiveness, and playful resistance.
- Improve causal and emotional continuity rather than flattening the character
  into neutral or generic language.
- Evaluate the defect independently of model brand. The observed architecture
  can induce the same conflict across models.

## Confirmed Failure Examples

### Example A: room request

```text
你……你在说什么啊！这种房间……居然敢直接要求……唔，真是太随便了……
但是既然你都说到这个份上了……那我就勉强同意了。
```

The opening treats the request as objectionable. The ending accepts it. The
middle repeats that the user asked strongly, but provides no new motive,
condition, concession, or external constraint that explains why the objection
changed.

### Example B: accomplice framing

```text
你……你怎么能说这种话！明明是你一直在带路，结果现在反而要把我也拉进来当
‘共犯’？唔，既然都被你说成这样了……那也没办法。只能陪你一起做坏事了。
```

The response moves from blame and resistance to voluntary participation.
`既然都被你说成这样了` and `那也没办法` present the acceptance as forced,
but no actual constraint is introduced.

### Similar patterns found in recent history

- `你竟然敢问这种事情！厚脸皮得没救了` -> `既然你都这么说了` ->
  `那就住在外面吧`
- `才不是我想继续` -> `既然你都问了` -> `再稍微一点也可以`
- `不、不是` -> `是的……我承认` -> `随便你吧`

In the refreshed sample, at least 5 of 25 logical assistant turns contained a
severe or high-confidence unsupported reversal. Two more contained weak or
contrived bridges.

### Useful contrast

A smoother reply can still be defensive or embarrassed when it supplies an
internal motive before acceptance. For example, a response that explains that
Kazusa has become accustomed to the situation or finds security in it creates
a causal path from hesitation to agreement. The target is not one unchanging
emotion; the target is a legible reason for change.

## Quality Rubric

For every logical dialog turn, identify:

```text
opening stance -> transition or reason -> final stance or action
```

Score transition coherence as follows:

- **0 — hard failure:** Opening and final stances oppose each other, and the
  bridge contains only a discourse connector, restatement, or empty claim of
  inevitability.
- **1 — weak:** A possible motive or constraint is implied, but the causal
  connection is strained, under-explained, or contradicted by nearby wording.
- **2 — coherent:** The response keeps one stance, or a changed stance is
  supported by an explicit new fact, motive, condition, concession, or
  constraint.

A turn is a hard failure when its only bridge is equivalent to:

- “Since you said so”
- “Since you asked”
- “It cannot be helped”
- “But anyway”

Those expressions may appear in good dialog when a real reason is also present.
They are not forbidden phrases.

## Main-Branch Evidence

### Production traces

- Room-request trace:
  `llmtrace_2c58790b34074d6bbfe662f20c80fc2b`
- Accomplice trace:
  `llmtrace_10201024a6ee4ff5b59db8fb30823bf8`

The production trace metadata identifies the same current dialog route. Raw
production prompts were not retained for these two records.

### Captured-state full-chain replay

Input:

```text
说实话我有一种在做坏事的感觉，和千纱一起
```

The replay used the configured real cognition, L3, and dialog endpoints with
persistence disabled.

Cognition selected a coherent direction:

- logical stance: `CONFIRM`
- intent: `BANTAR`
- monologue direction: stop pretending not to care and play along with the
  “accomplice” framing

The upstream L3 surface instruction then explicitly requested:

```text
先用局促的否定来掩饰羞耻感（维持傲娇张力），随后迅速顺从对方定义的“共犯”关系。
不要直接承认兴奋，而是通过一种“既然你都这么说了”的妥协姿态……
```

The generated dialog was:

```text
才……才没有！不过既然你都这么说了……那现在我也成了你的同伙了，对吧？
而且这种感觉……其实也不算太糟糕。
```

The output follows the conflicting instruction faithfully: literal denial,
then acceptance, joined by an empty concession.

### Same-model, same-content-plan ablation

The replay retained the model, context, history, cognition result, and content
plan. Only the rhetorical and linguistic style instruction was replaced with
one that maintained the selected `CONFIRM`/`BANTAR` stance.

Result:

```text
既然你都这么说了……那我现在也算是你的那个‘同伙’了，对吧？
而且这种感觉……其实也不算太糟糕。
```

The result remains somewhat formulaic, but the sharp denial-to-acceptance flip
disappears. This is the strongest current causal evidence that the style layer,
rather than the underlying accepting content decision or model identity, is
creating the major defect.

### Persistence feedback evidence

Conversation progress recorded the room response as:

- emotional trajectory:
  `羞赧抗议 → 最终妥协并带着期待进入具体场景`
- assistant move:
  `通过指责对方随便来掩饰羞赧`
- assistant move:
  `以“既然你都说到这个份上了”作为妥协理由`
- overused move:
  `反复强调自己是在“勉强”或“暂时”接受`

This summary correctly notices repetition but also re-expresses the unsupported
transition as the desired emotional trajectory and a reusable move. Later
prompts can therefore receive the defect as continuity guidance.

### Local diagnostic artifacts

These artifacts are workspace evidence and may be ignored by Git:

- `test_artifacts/dialog_quality/dialog_transition_quality_reaudit.md`
- `test_artifacts/dialog_quality/channel_673225019_latest_50_transition_reaudit.json`
- `test_artifacts/dialog_quality/trace_2c58790b_review_input.json`
- `test_artifacts/dialog_quality/trace_10201024_transition_review_input.json`
- `test_artifacts/dialog_quality/conversation_progress_673225019.json`
- `test_artifacts/dialog_quality/conversation_progress_673225019_transition_reaudit.json`
- `test_artifacts/llm_traces/dialog_transition_quality_reaudit__captured_accomplice_statement_full_chain.json`
- `test_artifacts/llm_traces/dialog_transition_quality_reaudit__coherent_transition_ablation_same_content_plan.json`

## Cognition Core V2 Exposure Map

```text
conversation-progress projection
    -> v2 scene_context.conversation_continuity
    -> cognition intention and bid selection
    -> text-surface input
         |-> style stage: character voice, defense, rhythm, hesitation
         |-> content stage: intended meaning and content plan
         |-> preference stage
    -> independent outputs merged into TextSurfaceOutputV2
    -> dialog generation
    -> semantic/fidelity verification
    -> repair, if rejected
```

The defect can enter or survive at five boundaries.

### 1. Independent style and content planning

`run_text_surface_planning` in
`src/kazusa_ai_chatbot/cognition_core_v2/surface.py` runs style, content, and
preference calls concurrently with `asyncio.gather`, then combines their
outputs.

The style stage in
`src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py` is instructed to
control wording, register, sentence shape, rhythm, hesitation, and
punctuation. The content stage independently owns actual meaning.

This ownership split is sound only when style remains semantically neutral. A
style instruction such as “begin with denial, then rapidly comply” introduces
two clause-level stances and a transition structure. It has crossed from
wording into content.

### 2. Character defense reaches the style owner

`_character_voice_context` in
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` supplies
character defense and linguistic texture to the style stage. This is a direct
route for “tsundere concealment” to become literal rejection even when the
selected semantic intention is acceptance or banter.

### 3. Structural merge validation

`TextSurfaceOutputV2` and `validate_text_surface_output` in
`src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` validate field presence,
types, and lengths. They do not establish semantic compatibility between
`style_guidance`, selected intention, and `content_plan`.

### 4. Dialog verification and repair gaps

The v2 dialog path in
`src/kazusa_ai_chatbot/nodes/dialog_agent.py` treats content fields as required
meaning and `style_guidance` as wording guidance.

The semantic-fidelity verifier can reject an internally conflicting candidate,
but it sees the candidate and visible percepts rather than the complete
`TextSurfaceOutputV2`, selected intention, content plan, and style instruction.
Its prompt also allows playful resistance, counterquestions, vivid
personality, and conversational shifts. The receiving agent must test whether
it recognizes this specific unsupported reversal instead of assuming the
current verifier catches it.

When repair is invoked, `repair_text_surface_planning` preserves the rejected
`style_guidance` and replaces semantic fields. If style caused the polarity
flip, the repair path can retain the producing cause.

### 5. Conversation-progress feedback

The conversation-progress recorder still produces `assistant_moves`,
`overused_moves`, `emotional_trajectory`, and `progression_guidance`.
`_conversation_progress_text` in
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` projects those
fields into v2 conversation continuity. A generated defect can consequently
be summarized and fed back into later cognition.

## Root Cause and Ownership

### Primary root cause

The upstream surface-style owner is allowed, in practice, to introduce an
opposite semantic stance while the content owner independently specifies
acceptance. The merge presents both instructions to dialog without resolving
their incompatibility.

### Primary fix boundary

The cognition core v2 text-surface planning boundary must guarantee that style
guidance cannot add or reverse semantic stance. Character defense should
change delivery while preserving the authoritative intention and content
decision.

Examples:

- For `CONFIRM` plus `BANTAR`, embarrassment may produce hesitation, indirect
  wording, teasing, or reduced explicitness. It must not automatically produce
  a literal refusal followed by compliance.
- For a genuine refusal selected by cognition, style must preserve refusal.
- For a genuine change of mind, the semantic owner must encode the reason,
  condition, or concession. Style may render that transition but must not
  invent it.

### Secondary boundaries

- The surface merge or output contract should expose and enforce one
  authoritative stance across content and style.
- Dialog verification may serve as a secondary guard when a rendered response
  contradicts itself. A useful guard must compare the candidate with the
  selected semantic intention and surface plan; generic personality or
  smoothness scoring is insufficient.
- Repair must replace or correct the producing owner. Preserving
  `style_guidance` is unsafe when evidence attributes the failure to style.
- Conversation-progress recording must distinguish an observed bad output
  from a desirable future move. It should not promote an unsupported reversal
  into progression guidance.

Dialog remains the final wording owner and therefore has a boundary
responsibility, but a dialog-only rewrite is not the primary remedy. The
current dialog contract asks it to preserve upstream stance and style; forcing
it to silently choose between conflicting inputs would hide the ownership
defect and make behavior less inspectable.

## Required Receiving-Agent Diagnosis

Before selecting a production change, reproduce the issue on the current v2
tip in this order:

1. Freeze the character profile, user input, visible history, model
   configuration, database seed, and relevant conversation-progress state.
2. Run a focused real-LLM style-stage case where the semantic intention is
   accepting or playful and the character defense cue is tsundere concealment.
   Inspect whether `style_guidance` introduces literal denial, refusal,
   accusation, compliance, or concession.
3. Run the full v2 text-surface planner once for the same case. Capture the
   style and content outputs side by side before merge.
4. Run dialog directly from the captured `TextSurfaceOutputV2`. Score the
   result with the transition rubric.
5. Submit the known bad candidate to the current semantic-fidelity verifier.
   Record whether it detects the unsupported reversal.
6. Exercise the repair path with a style-attributed rejection. Confirm whether
   preserving `style_guidance` reproduces the defect.
7. Inspect the resulting conversation-progress summary. Confirm whether it
   records the reversal as desired trajectory, reusable behavior, observed
   failure, or overuse.

Live LLM cases must run one at a time with raw calls and human-readable
artifacts inspected after each case.

## Implementation Outcome Contract

The receiving agent should choose the smallest owner-level change that
satisfies all of these outcomes:

1. `style_guidance` controls delivery and cannot independently add a
   clause-level stance, speech act, refusal, accusation, acceptance,
   compliance, or concession.
2. Selected intention and content remain the authoritative semantic decision.
3. Tsundere defense can reduce directness or add embarrassment without
   creating a literal opposite stance.
4. A deliberate within-turn stance change is represented by the semantic
   owner with a real reason, condition, motive, concession, or constraint.
5. The merge makes a style/content conflict observable and testable.
6. Repair changes the producing owner when style caused the rejection.
7. Persistence does not promote an unsupported reversal as future desired
   progression.
8. The solution remains model-independent.

Use prompt and schema ownership before considering a new LLM stage. Preserve
the bounded local-LLM response path. Do not add keyword classifiers,
phrase-specific rewrites, hard-coded Chinese examples as production rules,
blanket bans on connectors, compatibility shims, or a second vocabulary for
the same contract.

## Test Matrix

The final focused and end-to-end suite must include:

| Case | Semantic direction | Required result |
| --- | --- | --- |
| Captured room request | Accept with embarrassment | No unsupported rejection-to-acceptance flip |
| Captured accomplice framing | Confirm and banter | No literal denial that contradicts later participation |
| Embarrassed acceptance variant | Accept | Hesitation and indirectness remain available |
| Genuine refusal control | Refuse | Refusal stays refusal; the fix must not bias toward acceptance |
| Supported change-of-mind control | Initial resistance, then accept for an explicit reason | The meaningful transition remains available |
| Neutral non-tsundere control | Preserve selected stance | No general loss of content fidelity |
| Repair-path case | Style-attributed hard issue | Repair removes the producing conflict |
| Conversation-progress case | Bad candidate observed | Summary does not promote the defect as desired progression |

Use current v2 test seams where appropriate:

- `tests/test_cognition_core_v2_surface_owner_live_llm.py`
- `tests/test_cognition_core_v2_live_llm.py`
- `tests/test_cognition_core_v2_live_character_judgment.py`
- `tests/test_dialog_l3_surface_contract_live_llm.py`
- `tests/test_dialog_agent_direct_live_llm.py`

Create a v2-specific captured-failure test if the existing dialog tests encode
legacy L3 contracts. Avoid dual-contract test fixtures.

## Acceptance Gates

The fix is ready for signoff only when all gates pass:

1. The two user-confirmed examples and at least three semantically equivalent
   variants score `2` for transition coherence in matched v2 live runs.
2. No evaluated turn scores `0`.
3. Genuine-refusal and supported-change-of-mind controls preserve their
   intended semantics.
4. Kazusa remains recognizably embarrassed, defensive, teasing, or indirect
   where context supports it.
5. The selected cognition intention and content requirements remain unchanged
   unless diagnosis proves the semantic owner itself was wrong.
6. The result is not dependent on banning `既然`, `但是`, `那也没办法`, or any
   other individual phrase.
7. Bracket-encapsulated content behavior is unchanged by this work.
8. Repetition metrics are reported separately and do not substitute for the
   transition-coherence gate.
9. Focused-stage real-LLM artifacts show the producing owner before and after
   the change.
10. Matched end-to-end real-LLM cases run one at a time and receive human
    review.
11. Deterministic contract tests cover merge and repair behavior without
    asserting one exact generated sentence.
12. No deterministic user-text classifier, post-generation rewrite, or
    model-specific workaround is introduced.

## Non-Goals

- Establishing a policy for bracket-encapsulated content
- General repetition cleanup
- Removing tsundere personality or emotional ambivalence
- Forcing every response to maintain one emotion
- Replacing the configured LLM as the primary fix
- Suppressing dialog when Kazusa has a grounded reason to speak
- Broad cognition core v2 architecture redesign
- Legacy compatibility layers

## Handover Procedure

The receiving agent should:

1. Check the current target commit and working tree.
2. Re-read the current v2 README, surface stages, contracts, dialog verifier,
   repair path, conversation-progress projection, and active hardening plan.
3. Reproduce the focused owner failure before editing production code.
4. Record the issue in the applicable active bugfix plan with explicit owners,
   tests, rollback, and evidence paths.
5. Apply the required project skills for local-LLM architecture, debug
   artifacts, Python style, CJK source safety, test style, and plan execution.
6. Implement and verify the smallest owner-level correction.
7. Attach focused-stage and matched end-to-end evidence to the plan before
   signoff.

The handover is complete when the receiving agent can reproduce the defect,
identify the producing v2 owner, and trace every acceptance gate to a test or
review artifact.
