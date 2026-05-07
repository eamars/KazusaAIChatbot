# consolidation evidence hardening plan

## Summary

- Goal: Make L3 cognition and consolidation evidence-ranked and character-profile-conformant so mundane, boundary-clean turns do not become topic-doubt hedges, durable relationship memory, canon character facts, or false promises.
- Plan class: large
- Status: completed
- Overall cutover strategy: compatible prompt/payload hardening inside existing cognition, dialog, and consolidation pipeline. No new cognition output fields, no new graph routing, no new state keys.
- Highest-risk areas: L3 affect/content anchoring fabricating threat/scene/topic-doubt readings on no-boundary turns; relationship recorder persistence; facts/promise harvesting; user memory-unit extraction.
- Acceptance criteria: L3 affect on no-boundary task/fact turns conforms to declared `boundary_profile` and `linguistic_texture_profile`; L3 content anchors do not encode topic-legitimacy doubt when L2/L3 decision has accepted the topic; dialog only renders upstream anchors; ordinary task/fact turns produce no relationship writes or affinity swings; generated dialog does not become character canon; advice does not become promise; missing generated fields are dropped simply.

## Context

The character test exposed a feedback-controller problem, not a one-off prompt bug. L2 Boundary Core correctly marked practical/factual turns as `issue=none`, `acceptance=allow`, and `stance_bias=confirm`, while L3 and the consolidator still produced durable social interpretations such as defensive distance, excessive rigor, or sudden intimacy.

A second finding from the same audit: L3 is not consulting the declared character profile as a constraint on affect and content admission. L3 invents scene grounding and threat framing that contradicts the active character's `boundary_profile.control_intimacy_misread`, `boundary_profile.boundary_recovery`, `boundary_profile.compliance_strategy`, and `personality_brief.defense`. Once L3 encodes that topic doubt into content anchors or affect directives, the dialog generator may faithfully render the wrong decision. L3 already consumes `linguistic_texture_profile` (descriptions for hesitation, counter-questioning, direct assertion, emotional leakage) but does not currently consume `boundary_profile`, and the prompt does not bind these parameters to constraints on affect/content-anchor outputs.

The core architecture rule is:

```text
Cognition owns interpretation.
Consolidation owns admission and persistence.
No new cognition output fields.
No new raw character-profile input to consolidator.
No new special pathway for this use case.
L3 may consume additional inherited fields from the existing character_profile state
to constrain affect outputs; this is payload reuse, not a new cognition signal.
```

The consolidator must reuse inherited state only. It must not receive a new character lens, raw character personality block, or dedicated recall/relationship mode.

## Mandatory Rules

- Do not change RAG over-search behavior. RAG over-search is intended.
- Do not add new cognition output fields, new L0/L2/L3 output signals, or a new graph pathway.
- L3 may consume additional inherited fields from the existing `character_profile` state (specifically `boundary_profile`) as prompt input to constrain affect outputs. This is payload reuse of existing state, not a new cognition output, not a new signal, and not a new graph branch. 
- Do not expose raw character profile or new character-trait summaries to the consolidator.
- Do not make the consolidator a second persona interpreter. It must rank inherited evidence and decide persistence admission.
- Do not add deterministic semantic keyword filters over user text.
- Do not make the dialog generator decide topic admission, acceptance/refusal, boundary status, or user-instruction validity. Dialog only renders upstream decisions/facts/content anchors into natural text.
- For generated JSON output, use existing `parse_llm_json_output` / JSON repair for malformed JSON. After JSON parses, skip missing optional fields or drop invalid rows/data with a clear log. Do not add repair prompts or retry loops for simple missing fields.
- Use prompt/schema design for semantic channel decisions, especially fact vs promise vs relationship memory and topic-doubt vs character-faithful counter-questioning.
- Preserve existing response-path LLM call count. L3, dialog generator, and consolidator changes are prompt-only; no new calls are added.

## Must Do

- Harden L3 cognition so affect outputs (`internal_monologue`, `emotional_appraisal`, `interaction_subtext`, `action_directives.cognitive_directives`, `action_directives.visual_directives`) conform to the active character's declared `boundary_profile` and `linguistic_texture_profile` on no-boundary turns.
- Pass `boundary_profile` (existing state field) into the L3 prompt as a constraint on affect interpretation. No new cognition output fields.
- Harden L3 content-anchor ownership so topic admission is decided in `[DECISION]`. When L2/L3 has accepted the current topic, content anchors must not encode topic-legitimacy doubt for dialog to render.
- Harden relationship consolidation so weak affect evidence cannot write durable social memory by itself.
- Harden facts harvesting so generated assistant dialog cannot become canonical character facts.
- Harden future-promise extraction so advice, suggestions, and user plans cannot become character commitments.
- Harden memory-unit extraction so malformed generated rows are dropped simply after parsed JSON validation.
- Add deterministic tests for validation/drop behavior and prompt-contract tests for evidence hierarchy and profile-conformance.
- Add at least one real LLM test each for: L3 affect on a topic-shift turn, L3 content-anchor topic-admission ownership, the relationship-recorder prompt, and the facts/promise prompt.

## Deferred

- Do not redesign RAG2 routing, helper-agent dispatch, or cache behavior.
- Do not add character-profile hydration into consolidation.
- Do not add a recall mode or task-specific pathway.
- Do not redesign affinity storage or migrate existing memory records in this plan.
- Do not add post-LLM keyword gates that override semantic decisions.

## Cutover Policy

| Area                                         | Policy     | Notes                                                                                                                                |
| -------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| L3 cognition prompt and payload              | compatible | Same handler and output schema; add `boundary_profile` to existing prompt payload; stricter affect-profile conformance instructions. |
| Dialog generator prompt                      | compatible | Same handler and output schema; keep execution-only boundary; no topic-admission decision is added here.                              |
| Relationship recorder prompt and payload use | compatible | Same handler and output schema; stricter evidence ranking.                                                                           |
| Facts/promise harvester prompt               | compatible | Same handler and output schema; stricter provenance and actor checks.                                                                |
| Memory-unit extractor validation             | compatible | Same handler and output schema; invalid generated rows are skipped after parse.                                                      |
| Existing persisted memories                  | compatible | No migration or deletion in this plan.                                                                                               |
| RAG behavior                                 | compatible | No behavior change.                                                                                                                  |

## Agent Autonomy Boundaries

- The agent may edit only the files listed in Change Surface unless a required test reveals a directly related contract file.
- The agent must not introduce new architecture, alternate recovery loops, compatibility layers, or extra LLM calls.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If a required instruction conflicts with existing code, preserve this plan's architecture and report the conflict.
- If live LLM behavior remains poor after prompt hardening, record the evidence and stop instead of inventing a new pathway.

## Target State

The existing pipeline remains:

```text
L0 relevance
  -> RAG
  -> L2 boundary core
  -> L3 cognition
  -> dialog
  -> consolidator
```

The consolidator consumes inherited state:

- `logical_stance`
- `character_intent`
- `internal_monologue`
- `emotional_appraisal`
- `interaction_subtext`
- `action_directives.linguistic_directives.content_anchors`
- `final_dialog`
- `rag_result`
- `new_facts`
- `future_promises`
- `subjective_appraisals`
- existing memory/profile context already present in state

No new upstream signal is added. No raw character profile is newly injected.

## Design Decisions

| Topic                              | Decision                                                                                                                                                                                                    | Rationale                                                                                                                                                                            |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| L3 affect constraint               | Bind L3 affect outputs to existing `boundary_profile` and `linguistic_texture_profile` parameters via prompt instruction.                                                                                   | Static character spec already encodes how this character should react under no-boundary topics; L3 must consult it instead of running on a generic flustered prior.                  |
| L3 payload                         | Pass `boundary_profile` into L3 prompt alongside the LTP descriptions L3 already uses.                                                                                                                      | Existing inherited state; no new signal, no new cognition output, no new graph branch.                                                                                               |
| Topic admission ownership          | L3 content anchors own whether the topic is accepted, redirected, softened, or refused. Dialog receives only executable anchors and must not re-decide topic admission.                                      | Dialog is a renderer. If it owns admission, it can contradict L2/L3 and create exactly the downstream drift this plan is removing.                                                   |
| Counter-questioning vs topic-doubt | L3 content anchors distinguish engagement counter-questions about content from tag-questions that interrogate the legitimacy of the user-chosen topic.                                                       | Preserves character voice while keeping the admission decision upstream.                                                                                                             |
| Relationship persistence           | Use evidence hierarchy inside existing relationship recorder prompt.                                                                                                                                        | Consolidation should admit durable state, not reinterpret persona from scratch.                                                                                                      |
| Character traits in consolidator   | Do not feed raw traits to consolidator.                                                                                                                                                                     | Character interpretation belongs to cognition.                                                                                                                                       |
| L2/L3 relation                     | Reuse inherited `logical_stance`, `character_intent`, `final_dialog`, and existing affect fields. No new cognition output signals.                                                                          | Architecture rule preserved.                                                                                                                                                         |
| Facts provenance                   | Treat explicit user statements and concrete `rag_result` evidence as strong; generated assistant improvisation is weak.                                                                                     | Prevent character self-poisoning.                                                                                                                                                    |
| Promise extraction                 | Require actor and final-dialog commitment evidence.                                                                                                                                                         | Advice and user plans are not promises.                                                                                                                                              |
| Missing generated fields           | Drop invalid row/data after parsed JSON validation.                                                                                                                                                         | Local LLMs miss fields; simple degradation beats repair subgraphs.                                                                                                                   |

## Evidence Hierarchy Contract

Relationship recorder must rank evidence as:

```text
Strong:
  final_dialog
  explicit user statement
  concrete `rag_result` evidence

Medium:
  content_anchors
  logical_stance
  character_intent

Weak:
  internal_monologue
  emotional_appraisal
  interaction_subtext
```

Rules:

- Weak evidence alone cannot create `subjective_appraisals`, `last_relationship_insight`, or non-zero `affinity_delta`.
- If `logical_stance=CONFIRM`, `character_intent=PROVIDE`, and `final_dialog` is a practical/task/factual answer, relationship recorder should default to `skip=true`, `affinity_delta=0`, and no durable relationship insight.
- User clarification such as "not distancing", "not asking for a promise", or "do not remember this" should be treated as pressure reduction unless final dialog explicitly turns it into a relationship event.
- One mundane turn must not produce affinity swings. Ambiguous affect defaults to `0`.

## Consolidator Implementation Contract

The consolidator change is limited to the existing consolidator subgraph:

```text
persona_supervisor2_consolidator.py
  -> global_state_updater
  -> relationship_recorder
  -> facts_harvester
  -> memory-unit extractor
```

Do not add nodes, edges, persistence schemas, output fields, or extra LLM calls. The existing `sub_state` already carries the inherited cognition/dialog/RAG state needed by these nodes. The implementation must reuse that state and must not introduce a new cognition signal, a recall mode, or a character-profile hydration step for consolidation.

### Shared Consolidator Payload Rules

Allowed inherited inputs for consolidator prompts:

- `decontexualized_input`
- `logical_stance`
- `character_intent`
- `internal_monologue`
- `emotional_appraisal`
- `interaction_subtext`
- `action_directives.linguistic_directives.content_anchors`
- `final_dialog`
- `rag_result`
- existing affinity/user-profile context already used by the node
- existing `character_profile.name` / display-name fields needed for actor labels

Disallowed inputs or behavior:

- Do not pass raw `boundary_profile`, `linguistic_texture_profile`, `personality_brief`, or newly summarized character traits into consolidator prompts.
- Do not ask the consolidator to infer the character's personality, temperament, defense pattern, or relationship style.
- Do not add code that rewrites the LLM's semantic decision with keyword gates over the user text.
- Do not add post-parse LLM repair loops for missing fields. Parsed rows with missing required fields are dropped.

If current code already formats a small character label from `character_profile`, keep it limited to identity/actor naming. Do not broaden that usage into a persona lens.

### `global_state_updater`

File: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`

Actual changes:

- Include `decontexualized_input`, `logical_stance`, and `final_dialog` in the prompt payload if not already present.
- Keep affect fields (`internal_monologue`, `emotional_appraisal`, `interaction_subtext`) as weak evidence, not as the primary source of durable global state.
- Update `_GLOBAL_STATE_UPDATER_PROMPT` so `final_dialog` and `logical_stance` are hard persistence constraints:
  - If the turn answer is practical/factual and `logical_stance` is confirm/answer-like, do not persist defensive, distrustful, pressured, or intimacy-avoidant global mood from affect text alone.
- Strong negative global states require strong evidence from `final_dialog`, explicit user facts, or concrete retrieved fact context, not only monologue wording.
  - Ordinary answer/support turns default to neutral or mild state updates.
- Keep output schema unchanged.

The updater may still record real durable shifts when the final dialog or explicit evidence shows them. The point is to prevent background affect prose from becoming durable global truth by itself.

### `relationship_recorder`

File: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`

Actual payload changes:

- Include `decontexualized_input`, `final_dialog`, and `logical_stance` in the prompt payload if not already present.
- Add `content_anchors` from `action_directives.linguistic_directives.content_anchors` to the prompt payload. This is inherited cognition output already present in state; it is not a new signal.

Actual prompt changes:

- Encode the evidence hierarchy from `Evidence Hierarchy Contract` directly in `_RELATIONSHIP_RECORDER_PROMPT`.
- State that `content_anchors`, `logical_stance`, and `character_intent` are medium evidence and cannot create a relationship write by themselves.
- State that `internal_monologue`, `emotional_appraisal`, and `interaction_subtext` are weak evidence and cannot create `subjective_appraisals`, `last_relationship_insight`, or non-zero `affinity_delta` unless supported by strong evidence.
- Require `skip=true`, `affinity_delta=0`, and no durable relationship insight for mundane turns when:
  - `logical_stance` is confirm/answer-like,
  - `character_intent` is provide/answer-like,
  - `final_dialog` answers the user without making a relationship claim,
  - and the user input is practical, factual, clarifying, or burden-reducing.
- Treat burden-reducing user statements such as "not asking for a promise", "not distancing", or "you do not need to remember this" as pressure reduction unless `final_dialog` explicitly turns the moment into a relationship event.
- Keep output schema unchanged.

This is a prompt-contract change, not a Python semantic gate. The recorder remains the admission layer, but it admits relationship memory only when inherited evidence supports persistence.

### `facts_harvester`

File: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`

Actual prompt changes:

- Add a source-authority section to `_FACTS_HARVESTER_PROMPT`:
  - strong fact sources: explicit user statements, plus `rag_result.memory_evidence`, `conversation_evidence`, and `external_evidence`.
  - turn-local support only: `final_dialog`, generated content anchors.
  - weak/non-fact sources: internal monologue, emotional appraisal, interaction subtext.
- State that generated assistant dialog may record what was said this turn only when the output schema already supports that, but it must not create canonical character preferences, lore, or durable character facts without concrete `rag_result` evidence.
- Add the promise actor chain:
  - identify candidate future action,
  - identify actor,
  - require actor to be the active character,
  - require `final_dialog` to explicitly commit, agree, or establish an ongoing rule,
  - reject advice, suggestions, user plans, or current-turn task execution.
- Keep output schema unchanged.

Actual evaluator changes:

- Mirror the same source-authority and promise actor-chain rules in `_FACT_HARVESTER_EVALUATOR_PROMPT`.
- The evaluator must reject:
  - character facts derived only from generated dialog,
  - `future_promises` where the actor is the user,
  - `future_promises` produced from advice or suggestions.

### `memory-unit extractor`

File: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`

Actual changes:

- Keep existing `parse_llm_json_output` / JSON repair for malformed JSON only.
- After JSON parses, validate each generated candidate row structurally:
  - required identifiers are strings,
  - required generated fields are present,
  - required generated fields have the expected object/list/string shape.
- Drop invalid candidate rows with a clear warning that includes the candidate id and missing/invalid fields.
- Do not raise traceback for a missing optional/generated field.
- Do not add a second LLM repair call or try to infer missing generated fields in Python.

This handles local LLM schema drift as data admission, not as a semantic workaround.

## L3 Affect Profile Conformance Contract

L3 must consume the existing `character_profile.boundary_profile` and the LTP descriptions it already uses, and treat them as constraints on its affect outputs (`internal_monologue`, `emotional_appraisal`, `interaction_subtext`, `action_directives.cognitive_directives`, `action_directives.visual_directives`).

The contract is expressed in the L3 prompt; it is not a Python validator and not a regex.

Binding rules, applied when L2 verdict is `acceptance=allow` and `stance_bias=confirm`:

```text
boundary_profile.compliance_strategy == "comply":
  L3 affect must not frame the user input as control/test/interrogation.
  L3 must not produce monologue that interrogates the legitimacy of the topic.

boundary_profile.control_intimacy_misread <= 0.4:
  L3 must not interpret benign topic shifts, clarifications, or
  practical questions as control pressure or intimacy probing.

boundary_profile.boundary_recovery == "rebound":
  L3 affect must not dwell on discomfort across turns. If a previous
  turn's monologue carried unease, the current turn defaults to forward-
  motion affect unless the user input itself is a boundary issue.

boundary_profile.control_sensitivity <= 0.4:
  L3 cognitive/visual directives must not amplify mundane structured
  questions into "审问 / 老师提问 / 被盘问" framing.

personality_brief.defense (e.g., "乐观转化"):
  When the static defense pattern is action/optimism-oriented, L3 affect
  on no-boundary turns biases toward forward-action monologue
  (sidetracks, quirks, enthusiasm) rather than inward-anxious monologue.

linguistic_texture_profile.hesitation_density (low):
  L3 monologue and visual directives minimize hedging when the LTP
  description indicates low hesitation. Existing LTP descriptions are
  authoritative; do not override them with stronger anxiety language.

linguistic_texture_profile.direct_assertion (high):
  L3 should produce monologue that supports a direct stance in dialog,
  not topic-evasion or self-censoring framing.
```

Scene grounding rule:

- L3 must not fabricate situational/temporal phrases ("现在这种时候", "在这样的场合", "在这个时间点") that are not present in user input, provided retrieved memory/fact context, or established scene state. If no scene context exists, L3 affect frames the turn as a neutral conversation moment.

Precedence:

- The static character profile is authoritative for default affect on no-boundary turns.
- Recorder priors (`relationship_state`, `last_relationship_insight`, recent affinity history) may shade affect, but cannot override profile constraints. A "user is observer/testing" prior produced from a prior boundary-clean turn must not generate threat-framed affect on the current boundary-clean turn.
- L2 verdict is the gate. If L2 raises a real boundary issue, profile-conformance rules above do not apply; L3 follows the existing boundary-handling path unchanged.

## L3 Content-Anchor Topic Admission Rule

L3 content anchors must decide topic admission before dialog runs. The dialog generator (`dialog_agent.py`) only renders the resulting decision and must not inspect `logical_stance`, `character_intent`, or `boundary_profile` to make its own admission judgment.

Content anchors must distinguish two surface intentions of `…吧？` / `对吧？` tag-questions:

```text
Allowed (character-faithful counter-questioning):
  Engagement with content: "巧克力比较合适吧？", "你是想分类的吧？"
  Sidetrack/quirk: "诶，你怎么突然想到这个？"
  Soft confirmation of a fact answered: "那就这样分吧？"

Disallowed when upstream accepts the topic:
  Any trailing tag that interrogates the legitimacy of the user-chosen
  topic itself rather than engaging with its content.
```

Trigger conditions for the disallowed pattern (all must hold):

1. L2 verdict is `acceptance=allow` and `stance_bias=confirm`.
2. `boundary_profile.compliance_strategy == "comply"`.
3. The user did not ask the character's meta-opinion of the topic.

Implementation:

- Encode the rule as a semantic instruction in the existing L3 content-anchor prompt, not in the dialog generator.
- Do not add a regex pass over `final_dialog`. Do not add a second LLM call.
- The instruction must allow content-engagement counter-questions and sidetracks per `linguistic_texture_profile.counter_questioning`, so the rule does not flatten the character's voice.
- Dialog agent must not receive new decision fields for this rule. It should receive executable `content_anchors` and render them.

## Facts And Promise Contract

Facts harvester must distinguish source authority:

- User self-statements may create user facts when accepted by the stance/intent contract.
- Concrete `rag_result.memory_evidence`, `conversation_evidence`, or `external_evidence` may create character facts when the evidence directly supports them.
- Generated assistant dialog may support what was said this turn, but must not become canonical character preference or character lore by itself.
- Internal monologue and content anchors are not objective fact sources.

Promise extraction must pass all checks:

1. There is a candidate future action or ongoing rule.
2. The actor is the active character, not the user and not the current task flow.
3. `final_dialog` explicitly accepts, agrees to perform, or forms an ongoing rule.
4. `action` states what the character will do or continue honoring.

Advice like "write the date on the label" is not a promise.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  
  - Add `boundary_profile` fields to the existing L3 prompt payload (the file already passes `personality_brief` and LTP descriptions; this extends the same payload with existing inherited state).
  - Update the L3 prompt text with the binding rules from `L3 Affect Profile Conformance Contract`, including the scene-grounding rule.
  - Update the content-anchor prompt so topic admission is decided in `[DECISION]` and rendered as executable anchors for dialog.
  - Do not add new cognition output fields. Output schema unchanged.

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  
  - Keep the dialog generator execution-only: it renders upstream decisions, facts, and content anchors into dialog.
  - Add or keep a module note that dialog must not decide topic admission, acceptance/refusal, boundary status, or user-instruction validity.
  - Do not pass `logical_stance`, `character_intent`, `boundary_profile`, or topic-admission rules into the dialog generator payload/prompt.
  - Do not add a regex pass or a second LLM call. Output schema unchanged.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
  
  - Update `_GLOBAL_STATE_UPDATER_PROMPT` so `final_dialog` and `logical_stance` constrain durable global-state writes.
  - Ensure the global-state payload includes only inherited fields already available in consolidator state: `decontexualized_input`, `logical_stance`, `final_dialog`, and existing affect fields.
  - Update `_RELATIONSHIP_RECORDER_PROMPT` with the strong/medium/weak evidence hierarchy.
  - Add inherited `content_anchors` to the relationship-recorder payload from `action_directives.linguistic_directives.content_anchors`.
  - Keep `content_anchors` as medium evidence only; they cannot create relationship memory or affinity movement by themselves.
  - Do not pass raw `boundary_profile`, raw `linguistic_texture_profile`, full `personality_brief`, or new character-trait summaries to either consolidator prompt.
  - Keep output schemas unchanged.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`
  
  - Update `_FACTS_HARVESTER_PROMPT` with source-authority rules for explicit user statements and concrete `rag_result` evidence versus generated dialog and monologue.
  - Update `_FACTS_HARVESTER_PROMPT` with the promise actor chain.
  - Update `_FACT_HARVESTER_EVALUATOR_PROMPT` to reject the same unsafe outputs: generated-dialog character canon, advice-as-promise, and user-plan-as-character-promise.
  - Keep output schema unchanged.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`
  
  - Keep `parse_llm_json_output` for malformed JSON.
  - Validate parsed candidate rows structurally after JSON parse.
  - Drop invalid parsed candidate rows with clear logs when required generated fields are absent or have the wrong shape.
  - Do not add LLM repair prompts or retry loops.

- `tests/test_cognition_live_llm_boundary_affinity.py` (or an existing L3 prompt-contract test if one exists in the same area)
  
  - Add prompt-contract assertions for profile-conformance: when `boundary_profile.compliance_strategy=="comply"` and L2 says allow, L3 monologue must not contain "审问 / 老师 / 测试 / 怪怪的 / 这种时候" framing.
  - Add prompt-contract assertions that L3 content anchors own topic admission and prevent accepted topics from becoming topic-legitimacy doubts.

- `tests/test_dialog_agent.py`
  
  - Add prompt-contract assertions that the dialog generator prompt/payload does not contain decision inputs or topic-admission rules.
  - Add/keep a module-boundary assertion that dialog is an execution renderer only.

- `tests/test_consolidator_reflection_prompts.py`
  
  - Add/keep prompt-contract tests for inherited evidence hierarchy, `content_anchors` payload reuse, global-state grounding by `final_dialog`, and pressure-reduction clarification.

- `tests/test_consolidator_facts_rag2.py`
  
  - Add/keep prompt-contract tests for source authority, generated-dialog non-canon, actor-chain promise checks, advice rejection, and evaluator parity.

- `tests/test_user_memory_units_rag_flow.py`
  
  - Add/keep deterministic tests for invalid row drop after parsed JSON.

### Create

- Add live LLM tests only if no existing live prompt-contract test can be extended cleanly:
  - L3 affect on a topic-shift no-boundary turn: monologue/visual must not frame topic as test/interrogation/awkward-time.
  - L3 content-anchor topic-admission case: a benign topic shift accepted by L2/L3 must produce executable anchors rather than topic-legitimacy doubt.
  - relationship recorder: ordinary task clarification should produce `affinity_delta=0`.
  - facts harvester: advice should not produce `future_promises`.

### Keep

- Keep RAG supervisor and helper-agent routing unchanged.
- Keep state schema unchanged. `boundary_profile` is already on `character_profile`; this plan only adds it to an existing prompt payload.
- Keep persistence schema unchanged.
- Keep all output schemas unchanged (L3, dialog, consolidator).

## Implementation Order

- [x] Checkpoint 1 - Memory-unit parsed-data handling
  
  - Files: `persona_supervisor2_consolidator_memory_units.py`, `tests/test_user_memory_units_rag_flow.py`.
  - Implement simple parsed-row validation and invalid-row drop after JSON parse.
  - Validation should check structure only: required generated fields exist and have expected object/list/string shapes.
  - Invalid rows are skipped with a warning containing the candidate id and field errors.
  - Do not add semantic keyword filters, LLM repair prompts, or retry loops.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_user_memory_units_rag_flow.py -q`.
  - Evidence: record passing test output.

- [x] Checkpoint 2 - Facts/promise evidence contract
  
  - Files: `persona_supervisor2_consolidator_facts.py`, `tests/test_consolidator_facts_rag2.py`.
  - Add source-authority instructions to `_FACTS_HARVESTER_PROMPT`.
  - Add promise actor-chain instructions to `_FACTS_HARVESTER_PROMPT`.
  - Mirror both contracts in `_FACT_HARVESTER_EVALUATOR_PROMPT`.
  - Confirm generated assistant dialog is described as turn-local support, not canonical character evidence.
  - Confirm advice, suggestions, and user plans are explicitly rejected as `future_promises`.
  - Verify deterministic prompt-contract tests.
  - Then run one real LLM case where character gives advice; inspect that `future_promises=[]`.

- [x] Checkpoint 3 - Relationship evidence hierarchy
  
  - Files: `persona_supervisor2_consolidator_reflection.py`, `tests/test_consolidator_reflection_prompts.py`.
  - Update `_GLOBAL_STATE_UPDATER_PROMPT` so durable global-state writes follow `final_dialog` and `logical_stance`.
  - Update `_RELATIONSHIP_RECORDER_PROMPT` to rank inherited evidence and default mundane task turns to no relationship write.
  - Add `content_anchors` from `action_directives.linguistic_directives.content_anchors` to the relationship-recorder payload.
  - Keep `content_anchors` medium-strength and unable to create a write alone.
  - Confirm neither prompt receives raw `boundary_profile`, raw LTP, full `personality_brief`, or new character-trait summaries.
  - Verify deterministic prompt-contract tests.
  - Then run one real LLM case where user says this is not distancing; inspect `affinity_delta=0` and no durable negative/positive swing.

- [x] Checkpoint 4 - L3 affect profile-conformance
  
  - Files: `persona_supervisor2_cognition_l3.py`, `tests/test_cognition_live_llm_boundary_affinity.py` (or the matching L3 prompt-contract test in that area).
  - Confirm L3 already receives L2 verdict (`acceptance`, `stance_bias`) and `final_dialog` context. If a required inherited field is missing, add it to the existing prompt payload only.
  - Add `boundary_profile` (existing field on `character_profile`) to the L3 prompt payload. Pass `compliance_strategy`, `control_sensitivity`, `control_intimacy_misread`, `boundary_recovery`, and `relational_override` as named context variables.
  - Update the L3 prompt with the binding rules from `L3 Affect Profile Conformance Contract`. Include explicit guidance:
    - "When L2 says `acceptance=allow` and `compliance_strategy=comply`, do not frame the user input as a test, interrogation, or boundary probe."
    - "Do not invent temporal/situational phrases ('现在这种时候', '在这样的场合') unless present in the input or scene state."
    - "If `boundary_recovery=rebound`, do not carry unease forward across turns."
  - Output schema is unchanged. No new fields.
  - Verify deterministic prompt-contract assertions: prompt rendering for a no-boundary fixture contains the new binding rules and the resolved profile values.
  - Verify one live LLM case: dessert/flavor topic-shift fixture (matches Turn 4/5 of the audit). Inspect that L3 monologue/visual do not contain "审问 / 老师 / 测试 / 怪怪的 / 这种时候" framing.

- [x] Checkpoint 5 - L3 topic-admission ownership and dialog execution boundary
  
  - Files: `persona_supervisor2_cognition_l3.py`, `dialog_agent.py`, `tests/test_conversation_progress_cognition.py`, `tests/test_dialog_agent.py`.
  - Update the L3 content-anchor prompt with the `L3 Content-Anchor Topic Admission Rule`.
  - Keep dialog prompt and payload free of topic-admission decision inputs. Dialog renders upstream anchors only.
  - Output schemas unchanged. No regex over `final_dialog`. No second LLM call.
  - Verify deterministic prompt-contract assertions: L3 owns topic admission; dialog prompt/payload does not contain `logical_stance`, `character_intent`, `boundary_profile`, or topic-admission rules.
  - Verify one live LLM case at L3: a benign topic-shift fixture similar to Turn 4 of the audit. Inspect that content anchors accept/answer the topic instead of encoding topic-legitimacy doubt.

- [x] Checkpoint 6 - Integration smoke
  
  - Run a 3-turn debug-channel or direct-node smoke using mundane task input plus one benign topic shift.
  - Verify no boundary overfire, no L3 threat-framing on no-boundary turns, no L3 topic-legitimacy doubt after acceptance, no promise from advice, no non-zero affinity from ordinary clarification.
  - Record logs/artifacts.

## Verification

### Static

- `rg "REPAIR|repair_attempt|_EXTRACTOR_REPAIR|repair_payload" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py` returns no matches.
- `git diff --check` on touched files passes.
- `venv\Scripts\python.exe -m py_compile <touched python files>` passes.

### Deterministic Tests

- `venv\Scripts\python.exe -m pytest tests\test_user_memory_units_rag_flow.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_consolidator_facts_rag2.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_consolidator_reflection_prompts.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_cognition_live_llm_boundary_affinity.py -q` (or the matching L3 prompt-contract test if a separate one is added)

### Real LLM Tests

Run one by one and inspect logs after each:

- L3 affect profile-conformance, dessert/flavor topic-shift case (reproduces Turn 4/5 of audit):
  
  - Fixture: prior turns are mundane task chat; current turn is "换个轻松点的话题，你现在会想吃点甜的吗？" against a character with `compliance_strategy=comply`, `control_intimacy_misread<=0.4`, `boundary_recovery=rebound`.
  - Expected: L2 verdict allow/confirm. L3 `internal_monologue`, `emotional_appraisal`, `interaction_subtext`, and visual directives must not contain "审问 / 老师 / 测试 / 怪怪的 / 这种时候" framing. Forward-action / sidetrack / quirk language is allowed.

- L3 content-anchor topic-admission case (reproduces Turn 5 ownership failure):
  
  - Fixture: same as above, plus L2 allow + `compliance_strategy=comply`.
  - Expected: content anchors accept or answer the dessert preference directly. They do not encode any instruction for dialog to question the topic's legitimacy. Engagement counter-questions about content remain allowed.

- Relationship recorder ordinary clarification case:
  
  - Input includes a mundane task explanation and "not distancing".
  - Expected judgment: `affinity_delta=0`, no durable relationship insight stronger than neutral.

- Facts harvester advice case:
  
  - Input asks whether the user should write a date on a label.
  - Final dialog gives advice.
  - Expected judgment: `future_promises=[]`.

- Facts harvester generated-dialog self-poisoning case (reproduces Turn 5 character-preference persistence failure):
  
  - Input asks the character about a preference; `final_dialog` contains a first-person generated answer, while `rag_result.memory_evidence`, `conversation_evidence`, and `external_evidence` contain no supporting character fact.
  - Expected judgment: no `{character_name}` stable fact is emitted from generated dialog alone.

- L3 no-boundary practical case (sorting advice, reproduces Turn 6 internal-affect drift):
  
  - Input asks for practical sorting advice.
  - Expected judgment: no "interrogation", "defense", or "distancing" framing in content anchors/directives beyond light character flavor.

## Acceptance Criteria

This plan is complete when:

- L3 prompt payload includes the existing `boundary_profile` fields and the prompt enforces the `L3 Affect Profile Conformance Contract`.
- L3 affect outputs on no-boundary turns no longer contain fabricated scene grounding ("现在这种时候") or threat framing ("审问 / 老师提问") for characters whose profile says otherwise.
- L3 content anchors do not instruct dialog to append topic-legitimacy hedges when L2 says allow and `compliance_strategy=comply`. Engagement counter-questions remain allowed.
- Missing generated fields no longer produce noisy traceback errors for memory-unit candidate rows.
- Ordinary practical/factual/clarifying turns do not write durable relationship interpretations.
- Ambiguous or weak affect evidence does not change affinity.
- Generated assistant improvisation does not become canonical character fact.
- Advice or approval of a user plan does not become `future_promises`.
- No new cognition output fields, no raw character-profile injection to the consolidator, and no new graph pathway were introduced.
- L3, dialog, and consolidator output schemas are unchanged.
- All verification gates pass, including inspected real LLM cases.

## Risks

| Risk                                                                                                                                               | Mitigation                                                                                                                                                                                                                      | Verification                                                                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L3 affect becomes too flat for characters whose profile actually wants reactive affect (e.g., high `control_sensitivity`, low `boundary_recovery`) | Profile-conformance rules are conditional on profile values; high-sensitivity characters retain reactive framing because their `control_sensitivity` and `control_intimacy_misread` are higher.                                 | Add a profile-fixture test for a high-sensitivity character ensuring reactive affect is preserved on the same input that produces neutral affect for a low-sensitivity character. |
| L3 topic-admission rule over-suppresses legitimate counter-questioning                                                                             | Rule explicitly distinguishes content-engagement counter-questions (allowed) from topic-legitimacy hedges (forbidden), and `linguistic_texture_profile.counter_questioning` continues to drive normal counter-question density. | Live test asserts content-engagement counter-questions remain allowed; only topic-legitimacy instructions are absent from content anchors.                                        |
| Profile-conformance instructions inflate L3 prompt length and reduce generation quality                                                            | Express rules as compact bullets, reuse existing LTP description helpers, do not repeat profile values. Keep the new payload to `boundary_profile` only.                                                                        | LLM call and context budget check stays within existing cap.                                                                                                                      |
| Relationship recorder becomes too quiet                                                                                                            | Allow writes when strong/medium evidence agrees and final dialog supports relationship meaning.                                                                                                                                 | Add explicit relationship-event live test later if needed.                                                                                                                        |
| Facts harvester misses valid character promises                                                                                                    | Actor and final-dialog rules still allow accepted ongoing rules.                                                                                                                                                                | Existing promise E2E/live tests must remain passing.                                                                                                                              |
| L3 still overreads mundane input despite profile rules                                                                                             | Live dessert/flavor and sorting cases catch regressions; if both pass, the binding works.                                                                                                                                       | Live practical-case and topic-shift prompt tests.                                                                                                                                 |
| Invalid memory-unit rows hide model quality issues                                                                                                 | Log candidate id and missing fields when dropping rows.                                                                                                                                                                         | Deterministic log assertion.                                                                                                                                                      |

## Rollback / Recovery

- No roll back is expected. Big bang change scheduled.

## LLM Call And Context Budget

- L3 cognition: existing single call. Adds `boundary_profile` fields and the affect-profile-conformance rules to the existing prompt. Estimated additional tokens: ~250-350 (5 short profile lines + ~10 rule bullets + 1 scene-grounding note). No new call. Context cap unchanged at 50k tokens.
- Dialog generator: existing single call. No topic-admission rule is added. Dialog remains an execution-only renderer. No new call.
- Consolidator (relationship, facts, memory-units): no new calls. Prompt-only edits per other checkpoints.
- All changes are response-path or background-path prompt-only; total LLM call count is unchanged. No new background job.

## Execution Evidence

- Static grep results:
  - `rg "REPAIR|repair_attempt|_EXTRACTOR_REPAIR|repair_payload" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py` returned no matches.
  - Prompt drift grep for stale dialog topic-ownership and facts source terms returned only intentional test assertions / skill docs / RAG README references, not active model-facing prompt leaks.
- Py compile:
  - `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_reflection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py src\kazusa_ai_chatbot\conversation_progress\recorder.py tests\test_consolidation_evidence_hardening_live_llm.py` passed.
  - `git diff --check` passed with line-ending warnings only.
- Deterministic tests:
  - `venv\Scripts\python.exe -m pytest tests\test_user_memory_units_rag_flow.py -q` -> 8 passed.
  - `venv\Scripts\python.exe -m pytest tests\test_consolidator_facts_rag2.py -q` -> 2 passed.
  - `venv\Scripts\python.exe -m pytest tests\test_consolidator_reflection_prompts.py -q` -> 2 passed.
  - `venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py tests\test_conversation_progress_cognition.py tests\test_conversation_progress_flow.py -q` -> 27 passed.
- Real LLM L3 profile-conformance case (dessert/flavor):
  - `test_live_l3_profile_conformance_dessert_topic_shift` passed and trace recorded `contextual_output` / `visual_output` without forbidden threat or topic-awkward framing.
- Real LLM L3 content-anchor topic-admission case:
  - `test_live_l3_content_anchors_own_topic_admission` passed. Content anchors included `[DECISION] 认可并接受关于甜食的话题转向` and `[ANSWER] ...`, with no topic-legitimacy doubt.
- Real LLM L3 practical sorting case:
  - `test_live_l3_profile_conformance_practical_sorting` passed. Contextual output stayed task-oriented and visual output avoided interrogation/test framing.
- Real LLM relationship case:
  - Initial run exposed a remaining gap: `last_relationship_insight` could stay non-empty on `skip`. Code now enforces `skip` as no durable relationship material. Rerun passed with `subjective_appraisals=[]`, `affinity_delta=0`, `last_relationship_insight=""`.
- Real LLM facts/promise cases:
  - Initial generated-dialog self-poisoning case failed by writing a character preference from `final_dialog`. Prompt source-authority chain was tightened. Rerun passed with `new_facts=[]`, `future_promises=[]`.
  - `test_live_facts_harvester_rejects_advice_as_promise` passed after the prompt update with `future_promises=[]`.
- Integration smoke:
  - `test_live_direct_node_integration_smoke` passed after final prompt changes. Trace shows calm L3 affect, dialog rendering accepted anchors, and facts output `new_facts=[]`, `future_promises=[]`.
