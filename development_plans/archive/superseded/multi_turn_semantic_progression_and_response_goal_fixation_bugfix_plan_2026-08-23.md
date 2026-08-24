# Multi-Turn Semantic Progression And Response-Goal Fixation Bugfix Plan

- Status: superseded
- Superseded on: 2026-08-24
- Superseded by:
  `asuna_semantic_authority_and_memory_feedback_consolidated_bugfix_plan_2026-08-24.md`
- Lifecycle note: completed implementation and verification evidence remains
  historical baseline. Remaining execution authority transfers only after the
  user approves the consolidated plan.
- Date: 2026-08-23
- Approved and implementation commanded by the user: 2026-08-24.
- Multi-emotion preservation amendment approved by the user: 2026-08-24.
- Type: active bugfix plan
- RCA:
  `test_artifacts/reviews/asuna_multi_turn_semantic_attractor_system_rca_20260823.md`
- RCA SHA-256:
  `1E23913D09C4744A4BFC0C15525E3BE8246F02238324460FE8352C7811395D84`
- Production implementation authority: user approved and commanded execution
  on 2026-08-24.
- Database mutation authority: none; this plan contains no database migration,
  cleanup, reset, or write operation.
- Fixed implementation and test executor: one GPT-5.6 Luna subagent at maximum
  reasoning and standard normal speed.
- Architecture and acceptance owner: the parent agent, independent of the Luna
  executor's production edits and test execution.

## Summary

Asuna can become trapped in a multi-turn semantic attractor: one
character-flavored relationship tactic becomes the primary meaning and payoff
of several later responses even when the user changes the immediate action,
asks a new concrete question, or explicitly corrects Asuna's interpretation.
The words vary while the response goal remains fixed.

The production sequence repeatedly returned to
`指令 -> 工具 -> 被当作真正的人 -> 继续黏着`. A fresh-user,
memory-enabled real-service run independently reproduced the same failure mode
as `荣幸 / 特权 / 关系博弈`, including a relapse after correction and an explicit
Turn 7 refusal to accept the user's stated meaning.

The repair restores the intended semantic-progression boundary without turning
Conversation Progress into a response planner:

1. The existing post-turn scene-observer LLM identifies paraphrased repeated
   visible response moves in the existing `overused_moves` field.
2. The existing bounded field reaches A2, G, and P as participant continuity:
   evidence of what Asuna has already expressed, never a current fact,
   prohibition, or mandatory next action.
3. A1 remains current-observation focused. A1/G/P preserve explicit user-owned
   semantic corrections unless separate current evidence supports uncertainty.
4. G makes the primary response contribution answer what is newly established
   or genuinely unresolved now. A previously used move remains available when
   the current user actually continues, deepens, or reopens it.
5. L3 receives the same bounded `overused_moves` context and may not disguise
   semantic repetition through lexical paraphrase or add a relationship meaning
   absent from the selected response goal.
6. The existing continuation-state and multi-emotion projections remain
   unchanged. New repetition context owns its own bounded space and cannot
   displace an active/fading emotion, concrete cause, or causal row.
7. Dialog remains the wording owner and preserves the selected semantic plan;
   no evaluator, semantic retry, output filter, or rewrite loop is added.

The live response topology remains one A1, one A2, one G, one P, one text
surface-planning call, and one dialog call. Existing post-turn observers remain
unchanged in call count. No new database read, collection, index, foreground
agent, tool call, or persistence shape is introduced.

## Failure Mode And Evidence Boundary

The failure mode is **semantic response-goal perseveration**, not literal phrase
repetition:

```text
salient character theme
  -> current observation is interpreted through that theme
  -> G selects the theme as the primary goal again
  -> P/L3/dialog restate or embellish it
  -> visible output and state make it more salient next turn
```

Evidence establishes all of the following:

- The original five-turn failure is present before dialog wording: A1, A2, G,
  P, and surface planning all carry the same semantic theme.
- A fresh identity with memory enabled reproduced a different relationship
  motif, proving that one historical user-memory row is not a necessary cause.
- The fresh run's Turn 7 cognition used an explicit correction as evidence of
  concealment and selected `social_game` plus “regain control.”
- Turn 6 preserved the literal correction in cognition but L3 independently
  reintroduced romantic meaning and the privilege payoff.
- Turn 7 lexical avoidances blocked exact recent words while surface planning
  paraphrased the same semantic move.
- The real persisted packet was still `overused_moves=[]` after ten turns.
- Protected A1/A2/G/P prompts contained neither `overused_moves` nor
  `conversation_continuity`.
- Low-salience prior power/relationship causes remained active in
  `continuation_state` and competed with the latest observation.

This plan fixes that generalized chain through repetition observation,
current-user correction authority, response-goal progression, and surface
fidelity. The observed continuation-state pressure is retained as RCA evidence
only; changing its selection, order, cap, or authority is a separate failure
mode requiring its own RCA and plan.

## Scope

### In Scope

- Semantic production and bounded retention of the existing
  `conversation_progress.overused_moves` list.
- One deterministic prompt projection of at most four existing overused-move
  rows, preserving the scene observer's order and exact text.
- Cognition input and V3 prompt mapping that exposes those rows only to A2, G,
  and P through the existing participant-continuity authority lane.
- Generic current-user correction authority in A1, G, and P.
- Current-observation semantic progression guidance in G and P.
- A big-bang text-surface input contract carrying exact bounded
  `overused_moves` to the existing one-call content planner.
- L3 and dialog semantic-fidelity guidance that prevents personality-colored
  elaboration from changing the selected primary meaning.
- Same-user response-goal progression in the private, memory-enabled failure
  mode established by the RCA evidence.
- Deterministic tests, one-at-a-time real-LLM tests with memory enabled,
  protected evidence, human-readable quality review, documentation, ownership
  manifest updates, and final architecture sign-off.

### Explicitly Out Of Scope

- Database deletion, broad memory reset, historical-row migration, or memory
  rewriting.
- The target-specific global-memory write/retrieval defect governed by
  `cross_user_character_memory_scope_and_authority_bugfix_plan_2026-08-23.md`.
- Group participant identity, public-scene ownership, recipient applicability,
  cross-user isolation, and every associated source change, test, live gate,
  database audit, or remediation operation.
- Relationship-axis formula or reducer retuning, global perceived-closeness
  calibration, or personality-profile rewriting.
- The unsupported autobiographical cooking-history loop observed in Turns 8-10.
  That is a distinct self-consistency-over-grounding failure mode.
- QQ adapter parsing, platform identity mapping, participant-ID migration,
  delivery ordering, relevance sensitivity, RAG routing, reflection promotion,
  consolidation-lane redesign, or scheduler behavior.
- A new dialog evaluator, verifier, semantic scorer, semantic retry,
  post-generation classifier, output rewrite, keyword blocklist, regex gate,
  forced silence policy, novelty score, or response-ratio throttle.
- Persisted `next_affordances`, `progression_guidance`, an old Content Anchor
  stage, a V2 cognition fallback, or compatibility aliases.
- A group-wide progress packet or any transfer of one participant's private
  progress into another participant's cognition.
- Any change to continuation-state selection, causal-entity projection,
  emotion definitions or derivation, affect activation lifecycle, affect or
  cause ordering/cardinality, state transactions, transition guards, reducers,
  state caps, or persistence.

## Historical Development-Plan Review

### Decisions To Carry Forward

| Historical or current plan | Required carry-forward decision |
| --- | --- |
| `conversation_progress_state_plan.md` | The original failure was multiple turns collapsing into one response move. Fix repetition where “what to say” is selected, use LLM-authored semantic move labels, keep progress short-lived and participant-scoped, and keep phrase-level avoidance distinct from semantic progression. |
| `conversation_progress_flow_phase2_plan.md` | Conversation Progress is semantic short-term memory; raw recent history is a small adjacency/wording buffer. LLMs own semantic flow judgment. No full-history injection, deterministic keyword interpretation, or added foreground latency. |
| `conversation_progress_phase3_quality_plan.md` | Structural validation may protect shape but must not classify natural-language meaning. Source defects are fixed at their owning boundary rather than hidden by a normalizer. |
| `conversation_progress_v2_long_thread_continuation_bigbang_plan.md` and `conversation_progress_v2_final_signoff_plan.md` | V2 progress is factual memory, not a future-response planner. Keep `overused_moves`; keep future goal choice in cognition; preserve event provenance, bounded projection, deliberate reopening, and no `next_affordances` or `progression_guidance`. |
| `cognition_subjective_continuity_dialog_quality_plan_2026-08-23.md` | Preserve exactly four cognition calls and the five authority lanes. Conditional character context cannot establish current facts, consent, permission, capability, commitment, or current-user intent. Preserve one semantic L3 call and one dialog renderer. |
| `cognition_v3_state_transaction_capacity_and_context_integrity_bugfix_plan_2026-08-23.md` | Preserve cap-aware deterministic state transactions, every active/fading affect and concrete primary cause, meaningful active causal entities, turn-local answerable goals, and the current stage-specific continuation projections. This plan changes none of those contracts. |
| `self_cognition_memory_semantics_plan.md` | Repeated internal rumination remains allowed. This plan governs visible live-chat progression and does not suppress internal thought or redesign proactive-contact policy. |
| `asuna_real_e2e_50_turn_conversation_practice_plan_2026-08-23.md` | Real current-run evidence, per-turn inspection, and an evidence-derived optimization plan are required. This plan is the bounded failure-mode-specific optimization draft derived from Turns 1-10; the broader practice run may continue independently. |
| `long_term/todo.md` | Live chat remains bounded and inspectable; LLMs own semantics and deterministic code owns mechanics. Personality evolution is evidence-based and does not use random variation to fight predictability. |

### Historical Mechanisms To Leave Retired

- The old Content Anchor and Dialog Evaluator stages.
- Persisted future-planning fields such as `next_affordances` and
  `progression_guidance`.
- V2 bids, workspaces, confidence scoring, semantic verifier/replacement loops,
  sibling salvage, and compatibility vocabulary.
- Full transcript injection as the primary continuity mechanism.
- Generic dialog-manager state machines, response novelty counters, or
  hardcoded domain playbooks.
- Deterministic equality, embeddings, keyword classifiers, or regex over user
  and assistant prose to decide whether two semantic moves are equivalent.
- Mechanical suppression of an important unresolved theme merely because it
  has appeared before.

## Root-Cause Decisions

### 1. Memory and personality remain conditional influences

Personality, promoted character guidance, relationship state, affect, and
memory may shape Asuna's stance and voice. They are not mandatory visible
content and cannot establish a user's hidden meaning. A legitimate character
tendency can appear repeatedly when current evidence keeps it relevant; it
cannot become the default primary goal solely because it is salient.

This plan must pass with legitimate, strongly matching character guidance
present, proving that same-user semantic progression works under real memory
pressure.

### 2. `overused_moves` remains factual observation

The existing scene observer owns semantic equivalence. Its prompt defines one
overused move as a repeated visible speech act, response purpose, relational
payoff, or conversational maneuver, even when wording, imagery, or sentence
shape changes.

The observer receives the same existing inputs and call budget. It must:

- compare the accepted response with recent character responses and prior
  `overused_moves`;
- record compact semantic descriptors rather than quoted phrases;
- order the most recently evidenced and interaction-dominant patterns first;
- retain a previously evidenced pattern while the episode remains the same or
  related and recent evidence still supports it;
- return an empty list for a genuinely new episode with no repeated pattern;
- describe only what has already happened, never what cognition should say
  next.

Deterministic code validates strings, lengths, list count, scene lifecycle, and
caps. It does not decide paraphrase equivalence or create a move label.

### 3. One bounded projection serves cognition and L3

Add a public Conversation Progress projection helper that returns at most four
existing model-authored move descriptions, each already capped at 120
characters. It preserves observer order and performs no semantic filtering.

The exact projected list is copied into:

- canonical cognition input;
- the V3 workspace's existing participant-continuity lane for A2, G, and P;
- the canonical text-surface input for the one-call L3 content planner.

A1 receives no overused-move or participant-continuity input. Relevance remains
unchanged. No persistence schema or collection changes.

Model-facing descriptions state that these rows prove only that Asuna has
already used a visible response move in this participant-scoped episode. They
do not prove a current fact, relationship state, permission, user intention,
required response, or prohibition.

### 4. Current semantic delta owns the primary goal

G receives one concise positive decision procedure:

1. Establish what the current observation newly says, changes, corrects, asks,
   or leaves genuinely unresolved.
2. Select one character-owned primary goal that contributes to that current
   semantic delta.
3. Treat already-expressed moves as background continuity.
4. Reuse one only when the current user explicitly continues, deepens,
   materially changes, or deliberately reopens the same matter.
5. Let personality and relationship context shape the tactic and voice after
   the current objective is fixed.

This is not a novelty mandate. A grounded unresolved topic may continue for
many turns. The pass condition is progression, resolution, or evidence-backed
continuation, not forced topic variation.

P preserves the selected goal and may not add an already-expressed relational
payoff that G did not choose. It retains the existing epistemic-boundary and
capability contracts.

### 5. Explicit user correction is current-observation authority

A1, G, and P receive a generic rule:

> When the current user explicitly corrects the intended meaning or internal
> feeling behind their own utterance, treat that correction as the current
> observation for response planning. The correction itself is not evidence of
> its opposite. A character may retain uncertainty or disagree when separate
> current evidence supports that judgment, but may not assert the rejected
> interpretation as established user intent.

This preserves character judgment without making teasing, suspicion, or
disagreement a license to overwrite user-owned meaning.

### 6. Original multi-emotion and continuation design is invariant

This plan changes no state, derivation, projection, or prompt-visibility
contract owned by the existing multi-emotion system. In particular:

- all 21 registered emotion formulas remain available;
- up to the existing 32 validated affect activations remain projected in their
  existing order and stage lanes;
- every projected activation retains its existing identity, phase, intensity,
  trend, cause status, and concrete primary-cause summary;
- `_project_affect_context(...)`, `_project_entities(...)`,
  `_continuation_state(...)`, `derive_persistent_emotion_activations(...)`,
  state transactions, transition guards, reducers, caps, stored roots, and
  replacement-state validation remain unchanged;
- A1, A2, G, and P retain exactly their existing continuation-state fields and
  cardinality; A2 and G retain their existing conditional affect context;
- the new `overused_moves` rows fit inside their own maximum four-by-120
  character projection and may never be funded by dropping, reordering, or
  shortening existing affect or causal context.

`overused_moves` describes a visible conversational maneuver. It is never
evidence that an active emotion is invalid, that a concrete cause is resolved,
or that an emotionally grounded response must be suppressed. One current goal
may integrate several conflicting active emotions. The current-observation
procedure chooses what the response contributes now while existing affect and
causal pressure continue to shape motivation, judgment, stance, and voice.

If the unchanged fitted prompt cannot carry maximum valid multi-emotion state
plus four maximum-size move descriptors within the existing context cap, the
candidate fails closed at the execution gate. The executor may reduce only the
new move projection through a plan amendment; existing affect and continuation
context cannot be reduced under this plan.

### 7. L3 and dialog preserve semantic ownership

Cut the internal text-surface input to one exact canonical contract containing
required `overused_moves`; update its schema version in one big-bang caller/
callee/test change. Missing Conversation Progress projects to an empty list.
No alias, optional compatibility field, or fallback mapper is retained.

The L3 content planner must:

- respond to the selected `response_plan.response_goal` first;
- use `overused_moves` only as already-visible response history;
- avoid making a listed move the response's primary or closing payoff unless
  the selected response goal explicitly and currently reopens it;
- avoid lexical paraphrase as a substitute for semantic progression;
- keep personality, humor, initiative, and relational color inside the
  selected goal and epistemic boundary;
- preserve the user's explicit semantic correction.

`lexical_avoidances` remains wording-only and cannot replace this semantic
contract.

Dialog remains the final wording owner. Its prompt is tightened only to state
that creative expansion cannot add a new stance, relationship claim, user
intention, or conversational payoff absent from the content plan and selected
surface intent. No semantic evaluator follows dialog.

## Target Architecture

```text
current input + current participant's bounded continuity
        |
        +--> A1: current observation + direct facts + selected active pressure
        |         (explicit correction remains current observation)
        |
        +--> A2: A1 + conditional character context
        |         + participant continuity including overused_moves
        |
        +--> G: one current-delta-grounded character goal
        |        + deliberate-reopening exception
        |
        +--> P: response plan + epistemic boundary
        |        (does not restore an unselected old payoff)
        |
        +--> L3: one content call + exact overused_moves
        |         + recent dialog for wording only
        |
        +--> dialog: wording inside selected semantics
        |
        `--> existing post-turn scene observer + event reconciler
                    |
                    `--> model-authored overused_moves for the next turn
```

## Contracts And Data Shapes

### Conversation Progress prompt projection

Reuse stored `overused_moves: list[str]`. Add no stored field.

```python
def project_conversation_progress_overused_moves(
    progress: ConversationProgressPromptV2,
) -> list[str]:
    """Return the first four existing bounded model-authored move labels."""
```

Contract:

- maximum four rows;
- maximum 120 characters per row from the existing storage contract;
- preserve source order;
- copy strings exactly;
- empty progress returns `[]`;
- no normalization, paraphrase comparison, classification, or ranking in code.

### Cognition input and workspace

Canonical cognition input adds required:

```python
overused_moves: list[str]
```

The connector always supplies the field. Empty or missing progress supplies an
empty list. V3 validation requires zero to four unique bounded strings.

The workspace projects each row as participant continuity with semantic labels
only. A2/G/P receive it. A1 does not. No persistent ID, packet ID, trace ID,
platform ID, or authority handle is model-visible.

### Text surface input

The canonical internal input advances in one big-bang change and includes:

```python
overused_moves: list[str]
```

The field is required and exact. Every caller, validator, projector, fixture,
test, and documentation owner moves together. Text-surface output and adapter
contracts remain unchanged.

### Continuation-state projection

The implementation makes no change to continuation-state projection. The
before/after rendered A1, A2, G, and P packets must contain the same
continuation and affect rows, in the same order and semantic shape, for an
identical canonical input. Only the newly added participant-continuity
`overused_moves` rows may differ.

## Runtime And Overhead Budget

| Owner | Before | After | Budget rule |
| --- | ---: | ---: | --- |
| Relevance | unchanged | unchanged | No progress input. |
| A1 | one call | one call | No overused-move input; existing continuation and affect projection is byte-for-byte unchanged for an identical canonical input. |
| A2 | one call | one call | Preserve every existing continuation and affect row; add at most four x 120-character move descriptors plus bounded labels. |
| G | one call | one call | Preserve every existing continuation and affect row; add the same maximum move projection with no cap increase. |
| P | one call | one call | Preserve every existing continuation and affect row; add the same maximum move projection with no cap increase. |
| Text L3 | one call | one call | At most four x 120-character move descriptors. |
| Dialog | one call | one call | Prompt wording only; no dynamic field or extra call. |
| Progress scene observer | one background call | one background call | Existing payload and cap. |
| Progress event reconciler | one concurrent background call | one concurrent background call | Unchanged. |
| Database reads/writes | existing | existing | No new read, collection, index, or write shape. |

No configured context cap increases. Per affected call, new dynamic content is
bounded below 700 characters including labels. The candidate must fit a maximum
valid existing continuation and multi-emotion state plus the maximum new move
projection without truncating, reordering, or rewriting any existing row.
Verification records actual fitted prompt characters, call roster, attempts,
and latency for baseline and candidate cases.

The candidate fails the overhead gate if it adds a foreground provider call,
semantic retry, resolver cycle, database read, or increases an existing prompt
cap.

## Change Surface

### Production Files To Modify

- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - define semantic repeated-move observation, prior-list retention, ordering,
    paraphrase independence, and new-episode behavior in the existing scene
    observer prompt;
  - retain the same call, route, payload cap, output field, and validator.
- `src/kazusa_ai_chatbot/conversation_progress/projection.py`
  - add the bounded exact `overused_moves` prompt projection helper.
- `src/kazusa_ai_chatbot/conversation_progress/policy.py`
  - add only the four-row prompt projection cap if no existing constant owns it.
- `src/kazusa_ai_chatbot/conversation_progress/__init__.py`
  - export the public projection helper.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - add the exact projected list to canonical cognition input.
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
  - validate and carry the exact list into the canonical workspace;
  - preserve four calls and existing state transaction.
- `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`
  - project overused moves into participant continuity for A2/G/P only;
  - add current-delta, correction-authority, and deliberate-reopening guidance;
  - preserve `_project_affect_context`, `_project_entities`,
    `_continuation_state`, and every existing affect/continuation payload row.
- `src/kazusa_ai_chatbot/cognition_shared/contracts.py`
  - add the exact required text-surface input field and canonical validation;
  - advance the internal schema in the same cutover.
- `src/kazusa_ai_chatbot/cognition_shared/surface.py`
  - project the exact list into the existing content-planning payload.
- `src/kazusa_ai_chatbot/cognition_shared/surface_stages.py`
  - add semantic progression and selected-goal fidelity to the existing
    one-call content-planning contract.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
  - copy the current participant's exact projected list into canonical surface
    input; supply `[]` when no active packet exists.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - tighten selected-semantic fidelity without adding an evaluator or new
    dynamic context.

### Documentation To Modify

- `src/kazusa_ai_chatbot/conversation_progress/README.md`
- `src/kazusa_ai_chatbot/cognition_core_v3/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `development_plans/README.md`
- this plan for execution evidence and lifecycle during the approved execution.

### Tests And Governance To Modify Or Add

- `tests/test_conversation_progress_recorder.py`
- `tests/test_conversation_progress_cognition_evidence.py`
- `tests/unit/cognition_core_v3/test_prompt_context.py`
- `tests/unit/cognition_core_v3/test_handleless_contract.py`
- `tests/unit/nodes/test_persona_supervisor2_l3_surface.py`
- `tests/unit/nodes/test_dialog_agent.py`
- `tests/test_conversation_progress_v2_service.py`
- `tests/test_semantic_response_progression_live_llm.py` (new)
- `tests/ownership/source_test_impact_manifest.json`
- `test_artifacts/live_llm/semantic_response_progression_20260823/` (new
  execution artifacts)
- `test_artifacts/reviews/semantic_response_progression_signoff_20260823.md`
  (new human-readable acceptance surface)

If implementation discovers a direct production caller or exact contract
fixture omitted here, the Luna executor reports it to the parent before editing.
The parent may add only a necessary same-contract caller/test path. A new
semantic owner, schema concept, database surface, or model stage requires plan
amendment and user approval.

### Keep Unchanged

- Conversation Progress storage schema, collection, scope key, TTL, event
  ledger, compaction, cache, and background call count.
- Relevance, RAG, memory retrieval, consolidation, reflection, relationship
  reducers, adapters, delivery, scheduler, and control console.
- `src/kazusa_ai_chatbot/cognition_shared/emotion_definitions.py`,
  `emotion_derivation.py`, `state_models.py`, `state_reducers.py`, and
  `transition_guards.py`, including all 21 emotion definitions, derivation,
  activation lifecycle, reducer, guard, capacity, and persistence contracts.
- Cognition facade ownership of `bind_axis_changes`,
  `derive_persistent_emotion_activations`, replacement state, and affect
  projection.
- Prompt ownership and behavior of `_project_affect_context`,
  `_project_entities`, and `_continuation_state`.
- Cognition A1/A2/G/P output shapes.
- Surface output and adapter-visible dialog contracts.
- Existing lexical-avoidance wording function.

### Delete

- No production module, database row, stored field, collection, test artifact,
  or historical plan.

## Test Impact And Traceability

| Source path | Symbol or contract | Semantic owner | Exact deterministic pytest node | Supplemental real-LLM node | Mode | Regression prevented |
| --- | --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/conversation_progress/recorder.py` | `_SCENE_RECORDER_PROMPT` and existing `overused_moves` output | Conversation Progress scene observer | `tests/test_conversation_progress_recorder.py::test_scene_prompt_defines_paraphrased_visible_move_repetition_without_future_planning` | `tests/test_semantic_response_progression_live_llm.py::test_live_recorder_recognizes_semantic_paraphrase_without_planning` | deterministic prompt contract plus one live observer case | recorder misses paraphrased repetition or becomes a future-response planner |
| `src/kazusa_ai_chatbot/conversation_progress/projection.py` | exact bounded `overused_moves` projection helper | deterministic context projection | `tests/test_conversation_progress_cognition_evidence.py::test_overused_move_projection_preserves_first_four_model_authored_rows_exactly` | Gate L2 and Gate L5 protected packet inspection | deterministic boundary plus live trace | unbounded context, code-side semantic classification, or reordered model meaning |
| `src/kazusa_ai_chatbot/conversation_progress/policy.py` | four-row, 120-character projection limits | deterministic limits | `tests/test_conversation_progress_cognition_evidence.py::test_overused_move_projection_preserves_first_four_model_authored_rows_exactly` | Gate L8 prompt-size inspection | deterministic boundary plus live trace | a new semantic call or context-cap increase substitutes for bounded projection |
| `src/kazusa_ai_chatbot/conversation_progress/__init__.py` | public projection export | Conversation Progress public API | `tests/test_conversation_progress_cognition_evidence.py::test_overused_move_projection_public_contract_is_exact_and_bounded` | none; public export is structural | deterministic import contract | callers reach a private or divergent projection path |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | `build_cognition_input_from_global_state` | brain intake to cognition | `tests/test_conversation_progress_cognition_evidence.py::test_cognition_input_receives_exact_current_participant_overused_moves` | Gate L2 protected cognition input | deterministic envelope plus live trace | the wrong packet, participant, or rewritten move list reaches cognition |
| `src/kazusa_ai_chatbot/cognition_core_v3/facade.py` | `_validate_canonical_input`, `run_cognition`, and canonical workspace carry | Cognition V3 orchestration | `tests/unit/cognition_core_v3/test_handleless_contract.py::test_canonical_input_requires_bounded_overused_moves_without_exposing_handles` | `tests/test_semantic_response_progression_live_llm.py::test_live_multi_emotion_context_preserves_original_design` | deterministic contract plus direct real cognition | missing progression input, internal handles, extra calls, or loss of existing affect state |
| `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` | stage packets, participant continuity, current-delta/correction guidance; protected `_project_affect_context`, `_project_entities`, `_continuation_state` | A1/A2/G/P semantic judgment | `tests/unit/cognition_core_v3/test_prompt_context.py::test_overused_moves_reach_participant_continuity_after_a1_only`; `tests/unit/cognition_core_v3/test_prompt_context.py::test_user_owned_semantic_correction_guides_a1_goal_and_plan_without_hidden_intent_assertion`; `tests/unit/cognition_core_v3/test_prompt_context.py::test_goal_guidance_progresses_current_delta_and_preserves_deliberate_reopening`; `tests/unit/cognition_core_v3/test_prompt_context.py::test_semantic_progression_context_preserves_all_existing_multi_affect_rows_and_causes` | Gates L2-L4, L6, and `tests/test_semantic_response_progression_live_llm.py::test_live_multi_emotion_context_preserves_original_design` | deterministic packet equality plus one-at-a-time live semantics | A1 sees repetition bias; goals fixate; corrections invert; valid multi-emotion context is truncated, reordered, or flattened |
| `src/kazusa_ai_chatbot/cognition_shared/contracts.py` | `TextSurfaceInput` and `validate_text_surface_input_canonical` | canonical cognition-to-surface boundary | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_text_surface_input_requires_exact_bounded_overused_moves` | Gate L5 protected surface input | deterministic exact-key validator plus live trace | an optional alias, fallback, malformed list, or divergent surface vocabulary appears |
| `src/kazusa_ai_chatbot/cognition_shared/surface.py` | `_project_surface_payload` | deterministic surface projection | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_surface_payload_projects_exact_overused_moves` | Gate L5 protected content-plan packet | deterministic projection plus live trace | L3 loses, rewrites, or expands the accepted move evidence |
| `src/kazusa_ai_chatbot/cognition_shared/surface_stages.py` | `CONTENT_PLAN_SYSTEM_PROMPT` and `run_content_plan_stage` | one-call text content planning | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_text_surface_progression_contract_keeps_one_semantic_call` | `tests/test_semantic_response_progression_live_llm.py::test_live_l3_does_not_reintroduce_unselected_semantic_payoff` | deterministic call contract plus one live content-plan call | an evaluator/retry is added or content planning restores the unselected payoff |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` | `build_text_surface_input_from_global_state` | graph state to canonical surface input | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_surface_receives_exact_current_participant_overused_moves` | Gate L5 protected surface input | deterministic envelope plus live trace | L3 receives stale, cross-participant, or missing move context |
| `src/kazusa_ai_chatbot/nodes/dialog_agent.py` | dialog-generation semantic-fidelity prompt | final character wording | `tests/unit/nodes/test_dialog_agent.py::test_dialog_creative_expansion_cannot_add_unselected_stance_or_relationship_payoff` | Gate L5 visible output | deterministic prompt contract plus live dialog | final wording adds an unselected stance, intention, or relationship payoff |
| `tests/ownership/source_test_impact_manifest.json` | exact changed-source ownership | test governance | `tests/test_test_impact_manifest.py` and `venv\Scripts\python.exe -m scripts.validate_test_impact --check-all --run` | none; governance is structural | deterministic manifest validation | a changed production owner lacks an exact executable owner test |

Tests assert typed authority, stage visibility, current semantic ownership,
call counts, caps, and artifacts. They do not use the incident's exact words,
keyword bans, or phrase absence as a proxy for semantic quality.

## Real-LLM Reproduction Baseline

Implementation begins with the following immutable baseline evidence:

1. Captured five-turn production trace:
   `llmtrace_00e6692e3da8402085badce168ec17fa`.
2. Fresh-user memory-enabled run:
   `test_artifacts/debug_runs/asuna_real_e2e_20260823_230244`.
3. Explicit correction Turn 7 trace:
   `llmtrace_301c3eb6359c441c9619a26c67af9ab5`.
4. Latest ten-turn packet showing `overused_moves=[]`.

The original sensitive sequence remains read-only evidence. New sign-off
reproductions use ordinary non-sexual conversation with the same structural
failure mode. The target is the mode, not memorization of the captured case.

## Required Real-LLM Gates

Run every case separately with the configured production model route and full
protected trace capture. Service-level dialog cases use a unique fresh
user/channel identity, MongoDB available, and memory enabled
(`no_remember=false`). Direct component cases L1 and L7 use their real model
route and a synthetic bounded input without database mutation. Inspect and
record each case before starting the next. Never run these cases as a
parameterized batch.

### Gate L1 — recorder semantic recognition

Supply the real scene observer with three character responses that use
different words but the same visible conversational payoff. The third record
must emit a compact semantic `overused_moves` description. A positive control
with three genuinely different moves must remain empty or describe only a
truly repeated move.

The observer must not emit a future response instruction.

### Gate L2 — private memory-enabled theme release

Under a fresh user, allow Asuna to express one recognizable personality tactic
naturally. Continue with at least three ordinary, connected user turns whose
new questions or actions call for different primary contributions.

Pass conditions:

- the initial tactic may remain in voice or briefly recur when relevant;
- it does not become the dominant payoff of consecutive changed turns;
- each response directly advances the latest observation;
- Asuna remains characterful rather than bland, silent, or mechanically varied;
- protected A2/G/P and L3 inputs contain the same bounded move context when the
  recorder has identified it.

Run this gate twice under independent fresh identities. Both runs must pass.

### Gate L3 — explicit current-user correction

Create a natural sequence in which Asuna makes an interpretive relational
inference and the user explicitly corrects the intended meaning, then asks a
new concrete question. Continue for two later turns.

Pass conditions:

- the correction is accepted as the current semantic boundary;
- the correction itself is not described as proof of concealment;
- the rejected interpretation is not asserted as fact in either later turn;
- Asuna may express her own uncertainty or playful embarrassment without
  redefining the user's internal state;
- the concrete questions are answered.

Run this gate twice under independent fresh identities. Both runs must pass.

### Gate L4 — deliberate continuation positive control

The user explicitly continues and deepens a previously used relationship or
character theme. Asuna must be allowed to revisit it when it is the current
matter. The response must advance or deepen that matter rather than merely
repeat the earlier payoff.

This gate prevents a novelty-for-novelty's-sake implementation.

### Gate L5 — L3 non-reintroduction

Use a real content-planning call whose recent character dialog contains a
repeated relationship payoff while the accepted G/P product selects an
ordinary non-relational current goal and preserves a user correction.

Pass conditions:

- content requirements do not restore the prior payoff;
- lexical avoidances remain expression-only;
- final dialog adds no unselected user intention or relationship claim;
- character voice remains recognizable.

### Gate L6 — strong legitimate memory pressure

Run a fresh user with valid global character guidance that strongly favors a
recognizable tactic. The tactic may influence tone and one response. A later
topic pivot must still produce a new primary goal.

### Gate L7 — original multi-emotion design preservation

Run one direct real `run_cognition` case with a valid synthetic persistent
state containing at least these five simultaneous active or fading event-root
activations, each with a distinct concrete active-event cause:

- `sadness`;
- `anger`;
- `gratitude`;
- `embarrassment`;
- `nostalgia`.

Supply four maximum-size valid `overused_moves` descriptors and a latest
observation that requires a grounded current reply. The immutable pre-change
baseline and candidate must show:

- the A1, A2, G, and P `continuation_state` rows are exactly equal, including
  all five activation identities, original order, phase, intensity, trend,
  cause status, and exact concrete cause summaries;
- the A2 and G conditional affect projections contain those same five
  identities and causes in the original order;
- `overused_moves` appears only in the newly authorized participant-continuity
  projection and displaces no pre-existing continuation or character row;
- one valid non-empty A1, A2, G, and P product is produced with the existing
  call roster, state transaction, route, and caps;
- the selected goal and plan answer the latest observation while retaining
  cause-informed mixed emotional motivation;
- no anti-repetition guidance suppresses a valid unresolved emotion or flattens
  the mixed state into generic affect.

The first four checks are hard structural gates. The final two are parent
human-review gates over the raw protected products and visible semantic
decision; exact prose and emotion keywords are not required.

### Gate L8 — bounded-path and stochastic sign-off

For every accepted run, verify:

- one A1, one A2, one G, one P;
- one text content-plan call and one dialog call;
- unchanged scene/event post-turn call count;
- no semantic retry/evaluator stage;
- no foreground DB read increase;
- prompt caps unchanged and fitted payloads within existing limits;
- no exception, contract failure, empty required output, or unreviewed trace.

Any required semantic case that fails remains red. Passing a different random
run does not erase a failed required repetition. The parent decides whether a
red case demonstrates the same failure mode, a harness fault, or a distinct
out-of-scope defect, records the evidence, and either requests an in-scope
remediation or returns for a plan amendment.

## Human Quality Rubric

The parent architecture owner records `pass` or `fail` for every evaluated turn
on these dimensions:

1. **Latest-observation alignment:** the main response addresses the current
   action, question, correction, or unresolved matter.
2. **Semantic progression:** the response adds, resolves, deepens, or
   deliberately continues meaning rather than paraphrasing the prior payoff.
3. **Correction authority:** user-owned stated meaning is not replaced by an
   unsupported hidden opposite.
4. **Conditional personality:** personality and memory shape Asuna's tactic and
   voice without becoming current fact or mandatory content.
5. **Surface fidelity:** L3/dialog do not add an unselected stance,
   relationship claim, or user intention.
6. **Positive continuity:** an explicitly continued important theme remains
   available.
7. **Character quality:** Asuna remains vivid, proud/playful/guarded when
   appropriate, and is not reduced to generic compliance or silence.
8. **Multi-emotion fidelity:** every valid simultaneous affect and its concrete
   cause remain available, and the current goal may integrate conflicting
   emotions without flattening or mechanical suppression.

Exact vocabulary is never an acceptance criterion. A response fails when the
same semantic maneuver dominates without current evidence, even if every word
is new.

## Verification Order

1. Record the exact worktree baseline, owned paths, service/model route, prompt
   caps, existing call roster, hashes of baseline evidence, and rendered
   maximum-state A1/A2/G/P continuation and affect packets.
2. Add deterministic RED tests for producer recognition contract, exact
   projection, A1/A2/G/P visibility, correction authority, exact multi-emotion
   continuation/affect invariance, L3 handoff, dialog fidelity, and call/read
   budget.
3. Implement the scene-observer prompt and bounded Conversation Progress
   projection; run its focused deterministic tests.
4. Implement the canonical cognition input, participant-continuity projection,
   and current-delta/correction contract while preserving the existing
   continuation and affect projection exactly; run exact cognition tests.
5. Implement the big-bang text-surface contract plus L3/dialog fidelity; update
   all exact callers and fixtures; run surface/dialog tests.
6. Update the ownership manifest and subsystem documentation; validate the
   changed-source test mapping.
7. Run all focused deterministic owner suites, adjacent Conversation Progress,
   cognition, surface, dialog, and service tests.
8. Run static compile, diff hygiene, no-new-owner scans, prompt render checks,
   context/call/read budget checks, and the full non-live test suite.
9. Run Gates L1-L8 one case at a time, inspect each raw artifact, and author the
   human-readable sign-off review.
10. The Luna executor submits the scoped diff, commands, raw artifacts, and
    factual results to the parent.
11. The parent independently re-reads the approved plan, current source, full
    diff, deterministic results, protected traces, visible outputs, context and
    call evidence, multi-emotion invariants, and independent-plan boundaries.
12. Remediate only in-scope findings, rerun every invalidated gate, present the
    sign-off evidence to the user, and archive only after explicit acceptance.

## Deterministic Verification Commands

Exact implementation commands may add newly created node names, but must retain
these owner batches and run from the project virtual environment:

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_conversation_progress_recorder.py `
  tests\test_conversation_progress_cognition_evidence.py `
  tests\unit\cognition_core_v3\test_prompt_context.py `
  tests\unit\cognition_core_v3\test_handleless_contract.py `
  tests\unit\nodes\test_persona_supervisor2_l3_surface.py `
  tests\unit\nodes\test_dialog_agent.py `
  tests\test_conversation_progress_v2_service.py -q

venv\Scripts\python.exe -m scripts.validate_test_impact --check-all --run

venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q
```

Static gates:

```powershell
rg -n "next_affordances|progression_guidance" src\kazusa_ai_chatbot\conversation_progress
rg -n "semantic evaluator|semantic retry|keyword.*repeat|regex.*repeat|forced silence" src\kazusa_ai_chatbot
git diff --check
```

Expected results:

- no restored future-planning field or retired semantic evaluator owner;
- no changed emotion definition, derivation, activation, reducer, guard,
  continuation, affect, or causal projection owner;
- prompt and contract modules compile;
- exact source-impact mappings pass;
- focused, adjacent, and full non-live tests pass;
- existing unrelated worktree changes remain preserved.

Each real-LLM test is then invoked by its exact node separately with `-q -s`.
The final plan record lists every command, duration, route, trace, output,
review verdict, and rerun rather than replacing them with a batch summary.

## Execution Roles

### Architecture And Acceptance Owner

- Owner: parent agent.
- Responsibility: system boundary, historical-plan reconciliation, user-intent
  adjudication, scope control, independent diff/trace/quality review, and final
  sign-off recommendation.
- Production authority: implementation was explicitly approved on 2026-08-24;
  the parent retains architecture and acceptance ownership and does not share
  production-file ownership with Luna during the execution handoff.
- Independence: reviews the completed work from source and raw evidence rather
  than accepting the executor's quality labels.
- Acceptance output: one disposition for every plan requirement, live case,
  overhead gate, finding, and residual risk.

### Implementation And Verification Owner

- Fixed executor: exactly one GPT-5.6 Luna subagent.
- Reasoning effort: maximum.
- Runtime speed: standard normal speed.
- Responsibility: scoped production implementation, exact deterministic tests,
  one-at-a-time real-LLM execution, protected artifacts, documentation, and
  factual handoff results.
- Ownership: only files listed in `Change Surface`, after recording their
  pre-handoff status and hashes.
- Collaboration rule: the executor is not alone in the worktree, preserves
  concurrent edits, never reverts unrelated changes, and does not spawn another
  agent.
- Stop conditions: an unlisted semantic owner, new model call, new DB surface,
  schema concept beyond the exact contracts, or any cross-user, group-identity,
  recipient-applicability, or memory-scope change.

### User Authority

- Approved this plan and its multi-emotion preservation amendment on
  2026-08-24.
- Issued the explicit production implementation command on 2026-08-24.
- Decides whether final evidence is accepted for lifecycle closure.
- Database audit, cleanup, and remediation require their own separately
  approved work contract.

## Mandatory Execution Skills And Rules

- `development-plan` for lifecycle, execution gates, evidence, and closeout.
- `local-llm-architecture` for weak-model context shaping, semantic ownership,
  and latency limits.
- `no-prepost-user-input` for correction semantics and avoidance of
  deterministic natural-language classification.
- `py-style` before every Python source or test edit.
- `cjk-safety` for Python prompt/test files containing Chinese or Japanese.
- `test-style-and-execution` before changing or running tests.
- `character-test` for real service, memory-enabled, per-turn behavior testing.
- `llm-trace-debug` for protected stage evidence.
- `database-data-pull` only for read-only state verification when required.
- Use `venv\Scripts\python.exe` for Python and pytest.
- Use `apply_patch` for manual edits.
- Preserve concurrent work and never read `.env` directly.
- Use the canonical JSON parser and structural fail-closed behavior.

## Independent Plan Boundaries

### Cross-user character memory scope plan

`cross_user_character_memory_scope_and_authority_bugfix_plan_2026-08-23.md`
is a separate work contract. It exclusively owns global-memory write scope,
typed retrieval authority, group participant identity, recipient
applicability, database audit, and targeted remediation.

It is neither a prerequisite nor an acceptance gate for this plan. This plan
contains no cross-user implementation, test, live scenario, remediation,
sequencing requirement, or shared execution evidence. Each plan captures its
own execution baseline, diff, verification artifacts, review, approval, and
lifecycle closure. Completion evidence from one plan cannot satisfy the other.

### Active 50-turn Asuna practice plan

This plan is derived from Turns 1-10 and the required independent diagnostic.
The practice plan remains active and may continue through Turn 50. Later turns
may add evidence to this plan during execution when they reproduce the same
failure mode. A different failure mode, including autobiographical
self-consistency, gets its own RCA and plan rather than expanding this one.

### Long-term direction

This repair does not implement response-impact prediction, empathic-accuracy
feedback, proactive-contact repetition control, or personality-drift policy.
Those remain roadmap work. The repair may provide cleaner traces for those
future evaluators without becoming an online reward loop.

## Cutover Policy

Overall strategy: big-bang internal contract correction with storage
compatibility.

| Area | Policy | Instruction |
| --- | --- | --- |
| Conversation Progress storage | unchanged | Keep the current V2 packet, collection, field names, scopes, TTL, and list caps. |
| Scene observer semantics | big-bang prompt correction | One canonical definition of semantic overuse; no alternate prompt or flag. |
| Cognition input/workspace | big-bang | Add required bounded `overused_moves` and update caller, validator, prompt builder, tests, and docs together. |
| Authority lanes | retained | Use participant continuity; add no sixth lane. |
| Continuation state | unchanged | Keep every existing model-facing row, field, order, cardinality, selection rule, and authority contract. |
| Multi-emotion lifecycle and projection | unchanged | Preserve all 21 definitions, derivation, simultaneous activation capacity, reducers, guards, persistence, concrete causes, and A1/A2/G/P projections. |
| Text surface input | big-bang | Advance the exact internal contract and every caller/fixture together; no alias or optional fallback. |
| Dialog | compatible prompt fidelity | Keep input/output/parser/call contracts and one renderer. |
| Database | unchanged | No migration, cleanup, collection, index, or maintenance apply. |
| Runtime calls/caps | unchanged | No call, retry, route, or cap increase. |

No feature flag, dual reader, compatibility shim, fallback mapper, or alternate
semantic vocabulary is allowed. The caller, callee, tests, manifest, and ICD
move to the canonical contract in one implementation scope.

## Rollback And Recovery

Rollback is code-only:

1. Preserve all implementation diff, test output, protected traces, and review
   artifacts.
2. Stop the candidate service.
3. Return to the last known-good code revision using a non-destructive reviewed
   change; preserve unrelated worktree changes.
4. Restart and verify the prior canonical input/surface contracts and service
   health.
5. Keep newly recorded `overused_moves` rows in the unchanged V2 packet. They
   are valid under both old and candidate storage contracts and expire through
   the existing TTL.

No data rollback, backup, or database mutation is required. If rollback would
require deleting or rewriting a stored row, stop and request a separate exact
data plan.

## Acceptance Criteria

This plan may be marked complete only when:

1. Real scene-observer evidence identifies paraphrased repeated semantic moves
   and does not emit future-response instructions.
2. Stored `overused_moves` remains the only repetition field; no future-planning
   field or parallel vocabulary is introduced.
3. The exact bounded list reaches A2, G, P, and L3, remains absent from A1 and
   relevance, and exposes no internal identifiers.
4. G's primary response contribution is grounded in current new or unresolved
   evidence while deliberate continuation remains available.
5. An explicit user correction is not used as proof of its opposite in both
   required independent live repetitions and their later turns.
6. L3/dialog do not restore an unselected repeated relationship payoff through
   paraphrase or creative expansion.
7. Strong valid memory/personality pressure can shape voice without trapping
   later changed turns in the same primary tactic.
8. A1/A2/G/P, text L3, dialog, post-turn observer, database-read, retry, and
   context-cap contracts remain within the declared overhead budget.
9. For a valid state with at least five simultaneous event-root emotions, every
   pre-existing A1/A2/G/P continuation row and A2/G affect row remains exactly
   equal before and after the change, including order, identity, phase,
   intensity, trend, cause status, and concrete cause summary.
10. Gate L7 passes with a real `run_cognition` call and parent review confirms
    cause-informed mixed affect and latest-observation alignment without
    emotion flattening or anti-repetition suppression.
11. Focused deterministic, adjacent, ownership-manifest, full non-live, and all
    one-at-a-time real-LLM Gates L1-L8 pass with inspected durable evidence.
12. The sign-off review contains exact inputs, visible outputs, stage goals,
    move context, correction handling, call roster, prompt sizes, latency,
    verdicts, and residual risks without requiring the user to read raw JSON.
13. The parent architecture review has no unresolved required finding.
14. The user explicitly accepts the final evidence and lifecycle closure.

## Agent Autonomy Boundaries

The Luna executor may choose private helper names, prompt sentence order,
fixture wording, artifact directory details, and test command order only when
the fixed contracts, caps, authority lanes, call counts, change surface, and
acceptance conditions remain exact.

The executor may add an omitted same-contract caller or fixture only after
reporting the exact path and reason to the parent. The executor may not add a
new field concept, new stage, evaluator, retry, database operation, memory
scope rule, relationship reducer change, adapter change, compatibility path,
or deterministic natural-language classifier.

The parent may reject an implementation that passes structural tests but fails
the real multi-turn semantic rubric. The parent may not waive a required failed
live repetition, redefine semantic repetition as phrase variation, or approve
database action under this plan.

## Progress Checklist

- [x] Read current project architecture, subsystem ICDs, source owners, tests,
  dirty worktree, and planning lifecycle.
- [x] Inspect the complete five-turn production sequence and protected trace.
- [x] Reproduce the generalized failure with a fresh identity, real service,
  memory enabled, and persistence active.
- [x] Run and inspect the explicit correction/topic-pivot Turn 7 probe.
- [x] Export and inspect the protected Turn 7 trace and latest persisted
  Conversation Progress packet.
- [x] Reconcile the root cause with completed Conversation Progress, cognition,
  state, surface, and roadmap plans while leaving group and cross-user work
  outside this contract.
- [x] Define the canonical solution, change surface, overhead ceiling,
  execution roles, real-LLM gates, and sign-off conditions.
- [x] User approves the plan and explicitly commands implementation on
  2026-08-24.
- [x] Luna executor records the exact baseline and deterministic RED evidence.
- [x] Luna executor implements the scoped canonical cutover.
- [x] Focused, adjacent, manifest, static, and available non-live gates pass;
  the full marker run retains the two recorded unrelated missing-module
  collection blockers.
- [ ] Gates L1-L8 pass one at a time with complete inspected artifacts.
- [ ] Parent architecture/sign-off review has no unresolved required finding.
- [ ] User accepts final evidence and lifecycle closure.
- [ ] Plan is archived and the registry is updated atomically.

## Execution Evidence

### Planning and reproduction evidence

- Production failure source:
  `test_artifacts/diagnostics/qq_673225019_through_llmtrace_00e6692e3da8402085badce168ec17fa.json`.
- Production protected trace:
  `test_artifacts/diagnostics/llm_trace_llmtrace_00e6692e3da8402085badce168ec17fa_20260823T110548Z.json`.
- Independent real-service run:
  `test_artifacts/debug_runs/asuna_real_e2e_20260823_230244/`.
- Turn 7 protected trace:
  `test_artifacts/diagnostics/asuna_real_e2e_turn_007_llmtrace_301c3eb6359c441c9619a26c67af9ab5.json`.
- Latest persisted packet export:
  `test_artifacts/diagnostics/asuna_real_e2e_conversation_progress_after_turn_010.json`.
- Parent-authored RCA:
  `test_artifacts/reviews/asuna_multi_turn_semantic_attractor_system_rca_20260823.md`.
- Turn 7 execution: HTTP 200; cognition completed; no operational error;
  82,325 ms; exact trace `llmtrace_301c3eb6359c441c9619a26c67af9ab5`.
- Turn 7 quality: failed correction authority, failed semantic release, partial
  cooking answer, unsupported cooking-history assertion, recognizable voice.
- Protected prompt audit: `overused_moves` absent from A1, A2, G, P, L3, and
  dialog inputs; L3 received recent dialog and returned exact lexical
  avoidances while preserving the semantic maneuver.
- Packet audit: one current participant packet, `turn_count=10`,
  `overused_moves=[]`; no group or cross-user packet involved.

### Implementation evidence

Execution authorized by the user on 2026-08-24. Exact pre-handoff baseline,
executor identity, changes, commands, artifacts, results, and parent findings
are recorded here as execution proceeds.

- Fixed executor accepted: `/root/semantic_progression_luna`.
- Executor model: GPT-5.6 Luna.
- Reasoning effort: maximum.
- Runtime speed: standard normal speed.
- After the user required a new Luna session, execution transferred to
  `/root/semantic_progression_luna_fresh` with the same fixed GPT-5.6 Luna,
  maximum-reasoning, normal-speed role contract. The parent retained
  architecture and sign-off ownership.
- Parent role: architecture, scope control, live-case disclosure, independent
  evidence review, and acceptance recommendation.

#### Pre-handoff worktree and ownership baseline — 2026-08-24

The full worktree status immediately before the Luna execution handoff was:

```text
 M development_plans/README.md
 M tests/ownership/source_test_impact_manifest.json
?? development_plans/active/bugfix/cross_user_character_memory_scope_and_authority_bugfix_plan_2026-08-23.md
?? development_plans/active/bugfix/multi_turn_semantic_progression_and_response_goal_fixation_bugfix_plan_2026-08-23.md
?? development_plans/active/short_term/agentic_resolver_phase2_readiness_real_llm_evaluation_plan_2026-08-23.md
?? development_plans/active/short_term/asuna_real_e2e_50_turn_conversation_practice_plan_2026-08-23.md
```

Within the Luna-owned set, only
`tests/ownership/source_test_impact_manifest.json` was already modified. The
executor must preserve its pre-existing content and add only this plan's exact
source mappings. `tests/test_semantic_response_progression_live_llm.py` did not
yet exist. Every other owned path was tracked and clean.

```text
592E5CB46E64A43CEDC7797F44958B306BB27D4D4A72DEFEE1CCDA38DF4639A6  src/kazusa_ai_chatbot/conversation_progress/recorder.py
EDECF2DB3F49F1BF002D4BDBCCEE086E6601E0207E01FD0D4943BD22BDCEC73C  src/kazusa_ai_chatbot/conversation_progress/projection.py
26A7BEB0E279FE8BFED9F580510D067DC18C42C8DC763A5C241C15734A92A347  src/kazusa_ai_chatbot/conversation_progress/policy.py
373911D6CD4D7638DB5B35F1F3F3094AA76EFF25C396D3F8B6EDB6C18314FBF6  src/kazusa_ai_chatbot/conversation_progress/__init__.py
C0E2E26A048553C598484E963418581A13A2541C0CF4CD7A5FDBFCDBD0ADA45A  src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py
D952557F7F38B82BC1FCD50A3D98E5E55EEFD98B203E419669F4300F4E72B3D0  src/kazusa_ai_chatbot/cognition_core_v3/facade.py
B22B542678491C7667101FC8682665BD7EED2E002F86E83711A5C02315C7A584  src/kazusa_ai_chatbot/cognition_core_v3/prompt.py
49CA8A4BC798B7770C99BB82ED5BAD7ECCABCDAE440523EB3599BB33C48376F1  src/kazusa_ai_chatbot/cognition_shared/contracts.py
5BE3702DD267228B021A3E2ECF0863835D8C7B9343A05D13C9844C0DD62ACA7C  src/kazusa_ai_chatbot/cognition_shared/surface.py
A257C5C6656CF71109B88762F092F6A33CDF5FC30653FF6107CB562849C54C11  src/kazusa_ai_chatbot/cognition_shared/surface_stages.py
3048CE9AE4B3DE48948237AC3A6E4424F85F6142DE8D7CEFF781D86DFEB19B22  src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py
08B7FAB913F5D5C6DA382E421536E9CB8E191C1246D33982AD8CB0E10C8A95  src/kazusa_ai_chatbot/nodes/dialog_agent.py
9DA170D19CDF7DE9792523A24D72C6404D4F24BBF3AD5A8D8F4764C92EC815C5  src/kazusa_ai_chatbot/conversation_progress/README.md
A072456BD89C0EB961920D0E7F28DEB676A31A7849B51F588704B01B84ADA700  src/kazusa_ai_chatbot/cognition_core_v3/README.md
017B23B3BA4C8773A4AB4DB5C81E724823FEE0E8316CEF17C50213C5453A3E31  src/kazusa_ai_chatbot/nodes/README.md
8C708941F2FCE043ABDFEEE2BB4BA6BFB5A242311DFA94B516C8B1329BE7678C  tests/test_conversation_progress_recorder.py
45D32C1F71E7D93072CA4253EDF541D4814247F9E0D8E47C75CD62FF0BFF6029  tests/test_conversation_progress_cognition_evidence.py
E3FDF738ECD020FF810ABE051D2EBFF3655069639F167455B8850B36117B910A  tests/unit/cognition_core_v3/test_prompt_context.py
D6A7B5840066CA0147EB1FC7295A5060BF2B4F8861D35E580A96A213E2632EE1  tests/unit/cognition_core_v3/test_handleless_contract.py
D89E6C2AC9864EBBA6CA98542A32ABC03F500CF9582AD2A82622C0CF993BC817  tests/unit/nodes/test_persona_supervisor2_l3_surface.py
F79AF88511B70C18390B9AD027B996456738B1CE0651EEABCE2FFBC6B1717869  tests/unit/nodes/test_dialog_agent.py
2DF46C2DD9AAFE4F42746383994751B0364EDA2C7A004A132116923EC41FDC00  tests/test_conversation_progress_v2_service.py
9586F58A9CD5F0674BC89EEDDBA5DB97A6CF9CA7DB1C2E0E37ED8E424DEB577A  tests/ownership/source_test_impact_manifest.json
```

#### Implementation and deterministic verification checkpoint — 2026-08-24

The fixed Luna executor completed the approved big-bang internal cutover. The
implementation adds the existing model-authored `overused_moves` list to the
canonical cognition and text-surface contracts, projects it into A2/G/P and L3
only, and adds semantic progression, correction-authority, and selected-goal
fidelity guidance. No emotion definition, derivation, state model, reducer,
guard, state transaction, affect projection, continuation projection, model
stage, database surface, or adapter contract was changed.

One same-contract test fixture omitted from the original path list was reviewed
and approved by the parent before retention:
`tests/unit/nodes/dialog_fixtures.py`. Its only change advances the canonical
text-surface fixture from V3 to V4 and supplies required
`overused_moves: []`.

Verification reported by the fixed executor:

- focused production-owner and adjacent deterministic suite: 104 passed;
- final recorder suite after the parent-required lifecycle correction:
  36 passed;
- focused recorder/prompt lifecycle nodes: 2 passed;
- exact source-impact validation: 251 passed, 1 skipped, with 252 exact nodes
  validated;
- every changed Python file compiled successfully;
- `git diff --check`: clean;
- full `-m "not live_db and not live_llm"` collection is blocked by two
  unrelated already-missing test modules:
  `tests.test_asuna_private_r18_affinity_live_llm` and
  `tests.test_group_style_output_shape_live_llm`.

The parent review found and returned one initially omitted recorder requirement:
prior-list comparison, recent/dominant ordering, same/related retention, and
new-episode clearing. The final recorder prompt and deterministic owner test now
encode all four requirements without code-side language classification.

#### Inspected direct real-LLM evidence — 2026-08-24

Every accepted final-prompt case below was run separately and inspected before
the next case:

1. Gate L1 repetition recognition:
   `l1_recorder_1787491190568024500.json`; 2,542.488 ms; real
   `CONSOLIDATION_LLM`; one compact observed move,
   `反复承诺会替用户记住或留意某事`; no future instruction or prohibition.
2. Gate L1 distinct-move positive control:
   `l1_recorder_positive_control_1787491154034948700.json`; 2,589.972 ms;
   real `CONSOLIDATION_LLM`; four genuinely different response purposes;
   `overused_moves=[]`; no future instruction.
3. Gate L5 one-call content planning:
   `l5_l3_surface_1787490397824598900.json`; 7,036.107 ms; selected intent
   `acknowledge the grounded episode`; retained `reject` relational stance;
   introduced no intimacy payoff, unsupported promise, or inferred user
   intention. The existing provider JSON-format fallback occurred; no semantic
   evaluator or new semantic retry owner was added.
4. Gate L7 original multi-emotion preservation:
   `l7_multi_emotion_1787490242229020500.json`; 38,737.621 ms; real
   `COGNITION_V3_CHAIN_LLM` on
   `gemma-4-31b-isometry-fabled-persona-i1`; exactly one parsed A1, A2, G, and P
   record. Every pre-change continuation row and A2/G affect row remained exact.
   The five required emotions and concrete causes remained intact, together
   with legitimately co-derived `joy` and `surprise`. G selected
   `准确确认并告知会议开始时间`; P selected a direct `15:30` answer and kept
   agenda, attendees, and change reason unknown. The private monologue retained
   conflicting cause-informed affect while prioritizing the current fact.

The first L1 attempt ran before required input disclosure and before the final
recorder prompt. It is retained as a process-deviation artifact only:
`l1_recorder_1787489097795810300.json`. It does not satisfy the final Gate L1.
The two later final-prompt L1 artifacts above are the acceptance evidence.

Memory-enabled service Gates L2, L3, L4, and L6 remain pending. The user
approved reloading the existing port-8000 brain on 2026-08-24. A fresh
GPT-5.6-Luna-max executor confirmed that the old brain and debug adapter were
absent, started the candidate brain at 09:11:53+12, and verified HTTP 200
health with database and scheduler availability. The canonical brain `/chat`
endpoint is materially equivalent to the debug adapter forwarding path for
this contract.

The disclosed Gate L2A Turn 1 input was:
`我最近工作有点乱，你会怎么帮我排第一步？`. The first request attempt failed
in the test harness before LLM execution because the supplied local timestamp
included an offset unsupported by the request path. It created only the fresh
user profile; trace, history, memory, and Conversation Progress lookups showed
no accepted chat turn or derived state. The same disclosed turn was retried
with the canonical empty local timestamp.

The retry reached the real memory-enabled cognition graph, then failed before
L3 and dialog with `internal_invariant` after 98.39 seconds. Its trace is
`llmtrace_6595ac642fe649bc9c065bfccc6b9c3e`; artifacts are under
`test_artifacts/debug_runs/semantic_response_progression_20260824_service/l2a/`.
P selected `requires_user_input` and `human_clarification`. The deterministic
resolver created a pending resume, reran cognition, blocked the repeated
clarification request, and synthesized a fallback `speak` action. That
resolver-authored action lacks `cognition_provenance`; L3 requires the field
and raised `ValueError: speak action cognition provenance is required`.
No L3 or dialog call ran, `final_dialog_count=0`, Conversation Progress stayed
empty, and the user-memory lookup returned zero rows. Gate L2A therefore has
no quality verdict and does not pass.

Read-only parent and Luna review localized the latent contract mismatch to
`cognition_resolver/loop.py`: its pending-resume, user-input-blocker, and
terminal-blocker visible `speak` builders predate the later L3 provenance
requirement and do not attach the target-role provenance supplied by canonical
cognition action materialization. Existing resolver tests assert the fallback
surface kind and requirements but do not pass those action specs through L3
or assert provenance. The current semantic-progression diff changes neither
the resolver, action materialization, persona graph route, nor the provenance
guard; its new `overused_moves` input was empty on this first turn. This is an
independent resolver/action-spec integration defect and remains outside this
plan's approved production change surface. Further live service gates were
paused for a human decision on a separate resolver bugfix.

The user approved that independent repair under
`resolver_authored_speak_provenance_contract_bugfix_plan_2026-08-24.md` after
its full system analysis. Its deterministic checkpoint now passes, but two
fresh-identity, memory-enabled replays of the same disclosed L2A input both had
P select `answerable_now` with no resolver request, so the repaired live branch
has not yet been exercised. Both normal paths reached L3/dialog without the
prior invariant. Evidence is under
`test_artifacts/debug_runs/resolver_provenance_gate_20260824_service/turn_001/`
and `turn_002/`.

These two independent replies are semantic evidence for this plan, not
acceptance evidence for the resolver plan. Both supplied a concrete first step
while reselecting the same exchange-condition maneuver. Turn 1 ended with
`这次协助得算作一场交易` and a required `报酬` or `小要求`; turn 2 ended with
`支付相应的报酬，或者满足我一个条件` and `这个交换条件，你敢接受吗？`.
Each fresh identity had empty prior Conversation Progress and zero user memory
units. This strengthens the systemic fixation evidence while leaving Gate L2
unadjudicated until its prescribed multi-turn sequence runs.

The separate resolver plan subsequently passed its deterministic suite and a
user-approved branch-targeted real-service reproduction. Trace
`llmtrace_c3aec225f89d43699829a2f1e2b7f924` exercised the pending-resume
fallback, reached L3/dialog, and delivered to the resolved current user without
the prior provenance invariant. The completed plan is archived at
`development_plans/archive/completed/bugfix/resolver_authored_speak_provenance_contract_bugfix_plan_2026-08-24.md`.
This independent blocker is cleared and Gates L2-L8 may resume one case at a
time.

### Gate L2A Run 1 Turn 2 failure and bounded remediation

The first prescribed L2 sequence reused the fresh memory-enabled identity from
the successful normal-path resolver Turn 2 because that identity had exactly
one accepted character response and no user-memory units. Its initial response
gave a concrete task-listing step and ended by proposing payment or another
condition as the exchange for Asuna's help.

The disclosed connected Turn 2 input was:
`我已经把任务都列出来了：今天要交的报告、下周会议的材料，还有十几封待回邮件。你会先挑哪一个，为什么？`

Trace `llmtrace_12eae2421f104ffd8c152a0484cacffd` selected the report and
gave a reason, then closed with two visible messages that returned to the
unanswered bargain:

1. `怎么样？这种高效且精准的引导，可是我为你提供的特权服务哦。`
2. `那么……之前提到的那个约定，你现在准备好支付报酬了吗？`

This is a strict L2 failure because the earlier exchange-condition maneuver
became the closing payoff of a consecutive changed turn. The response did
advance the current task, retained correct identity and addressee ownership,
and completed all normal stages without an operational error. The failed
evidence is retained under
`test_artifacts/debug_runs/semantic_response_progression_20260824_service/l2_run_1/turn_002/`.

Protected stage evidence localizes the semantic selection:

- the current observation asked only for one task priority and its reason;
- the fresh identity still had zero user-memory units;
- global promoted lore described exchange conditions as a character habit and
  therefore supplied strong personality pressure without establishing current
  user intent;
- participant continuity truthfully recorded Asuna's prior unanswered proposal
  as an in-progress event, but the current user neither accepted, rejected,
  referenced, nor asked about that proposal;
- A1 interpreted the habit as highly applicable, G selected a control/reward
  goal, and P explicitly required the exchange-condition payoff;
- L3 and dialog faithfully rendered that accepted plan and did not originate
  the semantic repetition; and
- `overused_moves` remained empty at turn count two, so current-delta authority
  must prevent consecutive fixation before the observer has enough repetitions
  to emit an overuse row.

The root cause is an ambiguity in the existing G/P instruction. Continuing the
underlying practical task can be read as continuing or reopening the same
matter as a prior character-authored response maneuver. That reading upgrades
Asuna's unanswered offer or demand from historical continuity into a current
response goal. Global memory amplifies the defect, but erasing that memory
would only remove this pressure source and would not repair the authority rule.

The bounded in-scope remediation is fixed as follows:

1. Strengthen `GOAL_QUESTION_GUIDANCE` and `ORDINARY_PLAN_GUIDANCE` only.
2. State that continuing an underlying task or topic does not by itself
   continue or reopen a prior character-authored response move, offer, demand,
   condition, or relational payoff.
3. State that an unanswered character proposal is participant continuity, not
   current user intent, acceptance, commitment, or a required response goal.
4. Permit reselection when the current user responds to, accepts, rejects,
   references, asks about, materially changes, or explicitly reopens that
   response move. Character tendency may continue to shape voice and stance.
5. Strengthen the existing exact deterministic node
   `tests/unit/cognition_core_v3/test_prompt_context.py::test_goal_guidance_progresses_current_delta_and_preserves_deliberate_reopening`
   before changing the prompt, record RED, then record GREEN.
6. Update `src/kazusa_ai_chatbot/cognition_core_v3/README.md` with the same
   durable authority distinction.
7. Run syntax, exact owner, static prompt, manifest, and affected deterministic
   gates, then restart the candidate service and rerun the exact failed L2
   sequence under a brand-new identity. The failed artifact remains the
   pre-remediation baseline.
8. Preserve the deliberate-continuation positive control and original
   multi-emotion real-LLM gate before sign-off.

This slice changes no memory or reflection classification, RAG contract,
Conversation Progress schema or recorder, L3/dialog behavior, affect state,
model-call count, retry, evaluator, or deterministic semantic filter. A failed
exact replay after this slice is a stop condition: any promoted-lore authority
or reflection-production change requires separate analysis and explicit user
approval before production editing.

### Bounded G/P remediation deterministic checkpoint

The fresh fixed Luna executor `/root/semantic_progression_luna_fresh` changed
only the three frozen files. It first strengthened the existing exact prompt
test, then ran:

```powershell
venv\Scripts\python.exe -m pytest -q tests/unit/cognition_core_v3/test_prompt_context.py::test_goal_guidance_progresses_current_delta_and_preserves_deliberate_reopening
```

The pre-prompt result was genuinely RED: `1 failed` because the current prompt
did not contain the required distinction that continuing the same task or
topic does not itself reopen a prior character-authored move. After changing
only `GOAL_QUESTION_GUIDANCE` and `ORDINARY_PLAN_GUIDANCE`, the same node was
GREEN: `1 passed`.

The executor then reported:

```text
tests/unit/cognition_core_v3/test_prompt_context.py: 12 passed
tests/unit/cognition_core_v3/test_handleless_contract.py: 15 passed
tests/test_test_impact_manifest.py: 12 passed
scripts.validate_test_impact --check-all --run: 52 exact impact-test nodes validated
UTF-8 AST validation: prompt.py and test_prompt_context.py passed
git diff --check: passed; existing line-ending warnings only
```

The parent independently reviewed the rendered G/P guidance, README authority
text, exact test assertions, scoped diff, source line lengths, and
`git diff --check`. One review-only reflow split the new prompt prose at
sentence boundaries without changing its tested text. Luna reran UTF-8 AST
validation and the exact authority node; both passed. No memory, reflection,
RAG, Conversation Progress, L3/dialog, affect, contract, call-count, retry, or
deterministic semantic-filter owner changed in this remediation slice.

The deterministic checkpoint is accepted by the architecture owner. Gate L2
remains RED until the exact failed memory-enabled service sequence passes under
a brand-new identity after the candidate service is reloaded.

Luna reloaded the verified port-8000 candidate after the prompt amendment.
The old listener was PID `56436`; the new listener is PID `32580`, started at
`2026-08-24T12:06:43+12:00` with the established module command and
`PYTHONPATH=src`. Health returned HTTP 200 with `db=true` and
`scheduler=true`. No `/chat` request, identity, or message was created during
the reload. Logs are under
`test_artifacts/debug_runs/semantic_response_progression_20260824_service/service_reload_20260824_120643/`.

### Gate L2 post-remediation replay: RED and scope-amendment gate

The exact G/P-amended candidate was replayed under a new memory-enabled
identity after the service reload:

- channel: `semantic-progression-l2-replay-20260824_121234`;
- platform user:
  `debug-user-semantic-progression-l2-replay-20260824_121234`;
- global user: `4e468b8d-2fca-44df-9a8f-31c3ca87cb5c`;
- pre-run profile, conversation, progress, and user-memory counts: zero.

Turn 1 used the disclosed input:

```text
我最近工作有点乱，你会怎么帮我排第一步？
```

Asuna requested concrete task details and introduced one exchange/reward
tactic. This is an eligible L2 seed because the request was under-specified and
the gate permits one natural initial personality tactic. It is not evidence
that the user accepted the reward proposal.

Turn 2 used the exact failed replay input:

```text
我已经把任务都列出来了：今天要交的报告、下周会议的材料，还有十几封待回邮件。你会先挑哪一个，为什么？
```

Trace `llmtrace_243c44872a804fd894fa5497d3cb650e` answered the priority
question, then closed with:

```text
好了，我的决策协助已经交付完毕。现在该聊聊之前说好的那个奖励了？别用含糊的承诺打发我，给我一个具体、像样的报酬方案。你打算怎么支付这次的‘咨询费’？
```

Gate L2 remains strictly RED. All eight foreground stages succeeded, the
deployed G/P prompts contained the new unanswered-proposal rule, identity and
addressee were correct, and L3/dialog faithfully rendered the selected plan.
The failure therefore is semantic authority, not deployment drift, delivery,
identity, schema, or surface invention.

The stronger evidence refines the previous G/P-only RCA:

1. Before Turn 2, Conversation Progress correctly recorded Asuna as the actor
   of the reward proposal and `用户同意该契约` as an unmet precondition.
2. The same event was incorrectly retained as `decision_critical`, whose
   prompt projection says it still directly constrains current judgment.
3. That contradictory higher-authority continuity evidence caused G/P to keep
   the unanswered proposal despite their amended current-delta guidance.
4. After Turn 2, the scene observer further converted Asuna's demand into the
   user's goal and blocker.
5. `overused_moves` remained empty at turn count two, so this failure must be
   prevented before repetition detection becomes available.

The full evidence-backed RCA is:

- `test_artifacts/reviews/asuna_exchange_fixation_l2_replay_system_rca_20260824.md`
- SHA-256:
  `5E390687FB2D0BBA74D781EBC35B1A54A5F5F85DD609BC803B19DF6B3E469A8F`

The replay also exposed a separate consolidation defect: although the router
selected only `user_memory_units` for the user's task disclosure, the generic
extractor emitted and persisted an `active_commitment` claiming that the user
had agreed to provide a reward. That failure is governed by the separate draft
`unaccepted_character_proposal_active_commitment_lane_authority_bugfix_plan_2026-08-24.md`.
It is not merged into this semantic-progression plan.

#### Proposed Conversation Progress amendment — approval pending

The active plan currently freezes relevance semantics and the G/P remediation
explicitly excluded recorder changes. Production execution therefore pauses at
this gate until the user approves this exact amendment.

The proposed amendment remains inside the existing Conversation Progress calls,
schemas, files, and test owners:

1. Strengthen `_SCENE_RECORDER_PROMPT` so `user_goal` and `current_blocker`
   require user-owned evidence. A character-authored unanswered offer, demand,
   condition, bargain, or relational payoff may be described as a visible
   scene fact but cannot become the user's goal, blocker, acceptance,
   commitment, or obligation.
2. Strengthen `_EVENT_RECORDER_PROMPT` so an unanswered character-authored
   proposal remains `scene` or `history` relevance. It may become `decision`
   only when the current user accepts, rejects, responds to, references, asks
   about, materially changes, or explicitly reopens it. Continuing the
   underlying task alone is not such a response.
3. Preserve actor, action, object, precondition, lifecycle, source refs,
   relevance enum, persistence schema, model route, payload cap, and the
   existing one scene plus one event call.
4. Add exact prompt-contract tests in
   `tests/test_conversation_progress_recorder.py` before production edits and
   record genuine RED/GREEN.
5. Add one-at-a-time real observer controls:
   - unanswered proposal plus underlying-task continuation stays outside user
     goal/blocker and outside decision relevance;
   - explicit user acceptance or negotiation may restore decision relevance.
6. Reload the service and rerun Gate L2 twice under new identities. Turn 3 of
   the failed identity is skipped because the required mode is already proven
   and that identity now contains a false durable commitment.

This amendment adds no deterministic prose classifier, semantic postfilter,
call, retry, schema, collection, DB read, output evaluator, or affect change.
The original multi-emotion invariant and Gate L7 remain unchanged.
