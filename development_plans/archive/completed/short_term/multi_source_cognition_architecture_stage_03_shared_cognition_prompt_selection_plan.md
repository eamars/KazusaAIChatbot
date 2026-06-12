# multi source cognition architecture stage 03 shared cognition prompt selection plan

## Summary

- Goal: Add a source-aware cognition prompt-selection contract for L1/L2/L3
  cognition while keeping the current `/chat` prompt text and prompt payloads
  behaviorally unchanged.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` because the cognition prompt modules contain CJK strings.
- Overall cutover strategy: compatible selector insertion; current text
  `/chat` episodes select the existing prompt templates by explicit variant
  name, while future non-chat variants remain inactive references only.
- Highest-risk areas: prompt text drift, prompt payload drift, local-LLM schema
  brittleness, hidden source routing, and accidentally preparing RAG or
  consolidation behavior before their own stages.
- Acceptance criteria: L1/L2/L3 prompt handlers call the selector, current
  `/chat` prompt rendering remains equivalent, output-shape validators cover
  existing cognition outputs, and Stage 00, Stage 01, plus Stage 02 gates pass.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Stage: `stage_03`

Execution status: completed on branch
`multi-source-stage-03-prompt-selection` as of 2026-05-09.

## Context

Stage 02 moves a valid `CognitiveEpisode` through the current `/chat` graph and
cognition state without prompt or RAG consumption. Stage 03 is the next bridge:
the cognition prompt handlers become source-aware through a narrow selector,
but the only active variant remains the existing text `/chat` path.

Stage 02 is completed and merged into `main` as of 2026-05-09. Its execution
evidence lives in
`development_plans/active/short_term/multi_source_cognition_architecture_stage_02_chat_cognitive_episode_migration_plan.md`
and records the final combined verification result `43 passed`.

This stage does not tune prompt wording. It gives the graph an explicit place
to select prompt variants later, and it proves that current text `/chat`
rendering stays stable after that selector is inserted.

Prior-stage artifacts that must exist before editing:

- Stage 00 baseline test:
  `tests/test_multi_source_cognition_stage_00_regression_baseline.py`.
- Stage 00 fixture:
  `tests/fixtures/multi_source_cognition_stage_00_cases.json`.
- Stage 01 contract module:
  `src/kazusa_ai_chatbot/cognition_episode.py`.
- Stage 01 contract tests:
  `tests/test_cognitive_episode_contract.py`.
- Stage 02 completed wiring:
  `src/kazusa_ai_chatbot/service.py`,
  `src/kazusa_ai_chatbot/state.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`, and
  `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`.

Stage 02 handoff artifacts that must be preserved exactly:

- `src/kazusa_ai_chatbot/service.py` builds the episode with
  `_build_text_chat_episode_ids(...)` and
  `build_text_chat_cognitive_episode(...)`, then stores it in
  `initial_state["cognitive_episode"]`.
- `src/kazusa_ai_chatbot/state.py` exposes
  `IMProcessState.cognitive_episode`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` exposes
  `GlobalPersonaState.cognitive_episode` and
  `CognitionState.cognitive_episode`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` and
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` pass
  `cognitive_episode` through without renaming, wrapping, or nesting it.
- `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`
  proves service construction, debug-mode output-mode mapping, persona
  pass-through, and cognition pass-through.
- `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
  already asserts prompt human-message payloads do not contain
  `cognitive_episode`; Stage 03 must extend that assertion to selector
  metadata without weakening the existing check.

## Mandatory Skills

- `development-plan-writing`: preserve parent-stage scope and lifecycle.
- `local-llm-architecture`: keep source-specific prompt selection explicit and
  bounded for a weaker local model.
- `no-prepost-user-input`: do not add deterministic user-intent, preference,
  permission, or commitment interpretation while selecting prompts.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing cognition prompt Python files because they
  contain CJK strings.

## Mandatory Rules

- Before editing, read the parent ledger and confirm `stage_00`, `stage_01`,
  and `stage_02` are `completed`.
- Before editing, read Stage 00, Stage 01, and Stage 02 execution evidence and
  confirm every artifact path listed in `Context` exists.
- Treat `state["cognitive_episode"]` as a required Stage 02 artifact inside
  every LLM-backed L1/L2/L3 handler. Do not use `state.get(...)`, fallback
  episodes, synthetic episodes, or legacy-only prompt selection inside those
  handlers.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or
  final reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Do not change the literal content of existing prompt constants in
  `persona_supervisor2_cognition_l1.py`,
  `persona_supervisor2_cognition_l2.py`, or
  `persona_supervisor2_cognition_l3.py`.
- Do not change prompt human-message payload keys or values for current text
  `/chat`.
- Do not add reflection, internal-thought, image, audio, scheduled-recall,
  system-probe, or proactive runtime prompt variants.
- Do not change RAG query behavior, RAG context, RAG projection, RAG prompts,
  Cache 2 behavior, dialog behavior, consolidation behavior, persistence, or
  adapter delivery.
- Do not add live `/chat` LLM calls.
- Do not add deterministic keyword routing or semantic classification over
  user text.
- Output validation may validate structure only after each handler has built
  its existing normalized return dict. It must not reinterpret, filter, or
  rewrite LLM semantic decisions.
- Do not catch `CognitionPromptSelectionError` or
  `CognitionOutputContractError` in Stage 03. Unsupported or malformed
  contracts must fail closed during verification instead of silently falling
  back to the old prompt path.

## Prompt Text Fingerprint Guard

The current prompt constants are the active text `/chat` prompt contract for
Stage 03. The implementation agent must recompute these fingerprints before
editing L1/L2/L3 and again before completion. Any changed digest is a Stage 03
stop condition unless the diff proves only non-prompt code moved around it.

| File | Constant | SHA-256 of prompt string | Length |
|---|---|---:|---:|
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py` | `_COGNITION_SUBCONSCIOUS_PROMPT` | `93b4a80fa69aa7479d77699622aa632dd47a8515c475c91a0921bcdb302dc938` | 1768 |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` | `_COGNITION_CONSCIOUSNESS_PROMPT` | `241fb639de242e2d7fc964da922a8b0ea2ac0d9c4f5b2b762df210c34805a5e5` | 6795 |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` | `_BOUNDARY_CORE_PROMPT` | `dee7b322eb0d8637a3ee95b386560786042911cd0acca93b7c30896638ef26d1` | 5425 |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` | `_JUDGEMENT_CORE_PROMPT` | `ca4e88cc3854cbdb63372ad3b20644575ef9eb74abdc8637212fedc0ca5b3b89` | 4012 |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` | `_CONTEXTUAL_AGENT_PROMPT` | `4a2f7735c9f6b45637f329ad10581124360a24049444be43efb43cd2d802baae` | 2982 |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` | `_STYLE_AGENT_PROMPT` | `c0f66e0d744688afa4b105f20573708d295057856fa924c0102c0d5605cb6340` | 3748 |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` | `_CONTENT_ANCHOR_AGENT_PROMPT` | `9bf38821e24a561cec5c887f54432a4bff7b84131efb6c997d26edab8e0bbea0` | 7194 |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` | `_PREFERENCE_ADAPTER_PROMPT` | `f5b0363c0d1ea1f28770237d27908cbfd56a86410c7c64d9522c44e1c284f88d` | 4151 |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` | `_VISUAL_AGENT_PROMPT` | `68b1a35d43bfa28c46c91274d946faa9c7edf206f25fa414dabc822592626294` | 4337 |

## Must Do

- Add a prompt-selection module with the exact API in
  `Prompt Selector Contract`.
- Insert selector calls into every LLM-backed L1/L2/L3 cognition handler named
  in `Prompt Selector Contract`.
- Keep the selected active variant for current text `/chat` exactly
  `text_chat_user_message`.
- Add structural output-contract validation helpers for existing normalized
  L1/L2/L3 return dicts.
- Add tests proving selector decisions, unsupported source rejection, prompt
  render equivalence, human-message payload equivalence, and normalized output
  shape validation.
- Add inactive reference notes for future reflection and internal-thought
  prompt variants without wiring them into runtime.
- Rerun Stage 00, Stage 01, and Stage 02 deterministic gates.

## Deferred

- RAG episode adapter work. This belongs to Stage 04.
- Consolidation origin metadata and origin policy. These belong to Stages 05
  and 06.
- Reflection-triggered cognition, internal thought, scheduled recall,
  multimodal input expansion, proactive preview, proactive output, and adapter
  transport.
- Prompt tuning, prompt rewrites, new examples, new output schemas, or new LLM
  stages.
- Passing raw `CognitiveEpisode` or percept arrays into prompt payloads.

## Cutover Policy

Policy: `compatible`.

The selector becomes part of the current cognition prompt-render path, but for
current `/chat` text turns it selects the existing prompt constants and leaves
human-message payloads unchanged. There is no feature flag and no dual prompt
runtime path.

Rollback path: remove the selector module, remove selector calls from L1/L2/L3
handlers, remove the output-contract helper calls, and remove Stage 03 tests and
inactive reference notes. No database rollback is required.

Stop condition: if current text `/chat` prompt strings, human-message payloads,
RAG behavior, dialog behavior, consolidation behavior, response-path LLM call
count, or Stage 00 baseline changes, stop Stage 03 and create a bugfix plan.

## Agent Autonomy Boundaries

- The implementation agent must use the exact module path, public names,
  variant name, and signatures in this plan.
- Private helpers inside the selector and output-contract modules are allowed
  only for local table lookup or repeated structural validation. They must be
  deterministic, side-effect free, and must not add behavior not named in this
  plan.
- The implementation agent must use the exact handler insertion pattern in
  `Handler Insertion Contract`; do not invent wrapper layers, feature flags,
  fallback functions, or alternate call sites.
- The implementation agent must not invent additional runtime variants,
  fallback variants, feature flags, prompt templates, prompt examples, source
  labels, schema keys, or semantic routing rules.
- The implementation agent must not edit files outside `Change Surface`.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

The current text `/chat` cognition path becomes:

```text
CognitionState.cognitive_episode
-> select_cognition_prompt_variant(stage=...)
-> variant text_chat_user_message
-> existing prompt constant
-> existing human-message payload
-> existing normalized handler return dict
-> validate_cognition_output_contract(stage=..., payload=...)
```

The selector is source-aware, but runtime support remains limited to
`trigger_source="user_message"` and `input_sources=["dialog_text"]`.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Selector module | Create `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`. | Prompt selection is cognition-node ownership, not service or RAG ownership. |
| Active variant | Use exactly `text_chat_user_message`. | Gives Stage 04 and later stages a stable handoff name. |
| Unsupported source handling | Raise `CognitionPromptSelectionError`. | Non-chat triggers must not silently reuse text-chat prompts. |
| Prompt text | Existing prompt constants remain byte-for-byte unchanged. | Stage 03 is a selector insertion, not prompt tuning. |
| Output validation | Validate normalized return dicts, not raw LLM JSON. | Preserves current spelling-tolerance behavior while documenting schema contracts. |
| Validator invariance | Validator is keyed by `stage`, not by variant. Adding new variants in later stages must not change the normalized schema for an existing stage. | Locks the parent plan rule "same output schema for every variant" as a structural invariant rather than a social rule. |
| Future prompt drafts | Store as inactive reference notes only. | Keeps later source support visible without enabling it. |
| Selector signature scope | Selector accepts `trigger_source`, `input_sources`, and `output_mode` only. `visibility` and `target_scope` are deliberately not consumed in Stage 03 even though the parent plan's Prompt Strategy lists them for L2/L3. | With one active variant per stage, those fields cannot influence selection. Extending the signature is out of scope for Stage 03 and forbidden for Stage 04 (see Stage 04 Handoff). It belongs to the first stage that introduces a second active variant. |
| Allowed `output_mode` values | Accept `visible_reply`, `think_only`, `silent`. Reject `preview` and `scheduled_action_request`. | Mirrors the Stage 02 debug-mode mapping exactly: normal `/chat` produces `visible_reply`; debug `think_only` produces `think_only`; debug `listen_only` produces `silent`. `no_remember` keeps `visible_reply` and is handled outside selection. `preview` and `scheduled_action_request` belong to non-chat sources and must fail closed. |
| Rejection ordering | The selector must validate in this fixed order and raise on the first mismatch: (1) `validate_cognitive_episode`; (2) unknown `stage`; (3) unsupported `trigger_source`; (4) unsupported `input_sources`; (5) unsupported `output_mode`. | Deterministic order keeps `CognitionPromptSelectionError` messages stable across fixtures and avoids brittle test assertions. |

## Prompt Selector Contract

Create `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`.

The module must expose these public names:

```python
CognitionPromptStage = Literal[
    "l1_subconscious",
    "l2a_consciousness",
    "l2b_boundary_core",
    "l2c_judgment_core",
    "l3_contextual_agent",
    "l3_style_agent",
    "l3_content_anchor_agent",
    "l3_preference_adapter",
    "l3_visual_agent",
]

CognitionPromptVariant = Literal["text_chat_user_message"]


class CognitionPromptSelection(TypedDict):
    stage: CognitionPromptStage
    variant: CognitionPromptVariant
    prompt_key: str
    trigger_source: TriggerSource
    input_sources: list[InputSource]
    output_mode: OutputMode


class CognitionPromptSelectionError(ValueError):
    """Raised when an episode cannot select a cognition prompt variant."""


def select_cognition_prompt_variant(
    *,
    episode: CognitiveEpisode,
    stage: CognitionPromptStage,
) -> CognitionPromptSelection:
    """Select the cognition prompt variant for one stage."""
```

`select_cognition_prompt_variant` must:

- Call `validate_cognitive_episode(episode)`.
- Accept only `trigger_source="user_message"`.
- Accept only `input_sources=["dialog_text"]`.
- Accept `output_mode` values `visible_reply`, `think_only`, and `silent`.
- Raise `CognitionPromptSelectionError` for every other trigger source,
  input-source combination, output mode, or unknown stage.
- Apply the rejection checks in this fixed order and raise on the first
  mismatch: (1) `validate_cognitive_episode`; (2) unknown `stage`;
  (3) unsupported `trigger_source`; (4) unsupported `input_sources`;
  (5) unsupported `output_mode`.
- Return `variant="text_chat_user_message"`.
- Return `prompt_key=f"{stage}.text_chat_user_message"`.
- Not inspect percept content, user text, RAG evidence, profile facts,
  relationship state, debug-mode names, prompt payloads, `visibility`, or
  `target_scope`. `visibility` and `target_scope` are intentionally outside
  the Stage 03 selector contract; see `Design Decisions`.

The following handlers must call the selector before choosing their system
prompt template:

- `call_cognition_subconscious`
- `call_cognition_consciousness`
- `call_boundary_core_agent`
- `call_judgment_core_agent`
- `call_contextual_agent`
- `call_style_agent`
- `call_content_anchor_agent`
- `call_preference_adapter`
- `call_visual_agent`

`call_interaction_style_context_loader` and `call_collector` are not
LLM-backed prompt handlers and must not call the prompt selector.

Each handler must map `text_chat_user_message` to its existing prompt constant
and must not add selector metadata to the prompt payload.

## Handler Insertion Contract

Each LLM-backed handler must use the Stage 02 episode state directly:

```python
selection = select_cognition_prompt_variant(
    episode=state["cognitive_episode"],
    stage="<approved_stage_name>",
)
prompt_template = {
    "text_chat_user_message": <EXISTING_PROMPT_CONSTANT>,
}[selection["variant"]]
```

Then the handler must use `prompt_template.format(...)` in the same place where
it currently formats the existing prompt constant. The human-message payload
must remain semantically equivalent, verified by parsed payload assertions that
ignore JSON serialization ordering.

Approved stage names by handler:

| Handler | Stage name | Existing prompt constant |
|---|---|---|
| `call_cognition_subconscious` | `l1_subconscious` | `_COGNITION_SUBCONSCIOUS_PROMPT` |
| `call_cognition_consciousness` | `l2a_consciousness` | `_COGNITION_CONSCIOUSNESS_PROMPT` |
| `call_boundary_core_agent` | `l2b_boundary_core` | `_BOUNDARY_CORE_PROMPT` |
| `call_judgment_core_agent` | `l2c_judgment_core` | `_JUDGEMENT_CORE_PROMPT` |
| `call_contextual_agent` | `l3_contextual_agent` | `_CONTEXTUAL_AGENT_PROMPT` |
| `call_style_agent` | `l3_style_agent` | `_STYLE_AGENT_PROMPT` |
| `call_content_anchor_agent` | `l3_content_anchor_agent` | `_CONTENT_ANCHOR_AGENT_PROMPT` |
| `call_preference_adapter` | `l3_preference_adapter` | `_PREFERENCE_ADAPTER_PROMPT` |
| `call_visual_agent` | `l3_visual_agent` | `_VISUAL_AGENT_PROMPT` |

Forbidden handler insertion patterns:

- `state.get("cognitive_episode")`
- constructing a new `CognitiveEpisode` inside L1/L2/L3
- defaulting to `text_chat_user_message` outside
  `select_cognition_prompt_variant`
- adding `selection`, `prompt_key`, `trigger_source`, `input_sources`,
  `output_mode`, or `cognitive_episode` to the LLM human-message payload
- wrapping selector or validator calls in `try`/`except`

## Output Contract Validator

Create `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`.

The module must expose:

```python
class CognitionOutputContractError(ValueError):
    """Raised when a normalized cognition stage output is structurally invalid."""


def validate_cognition_output_contract(
    *,
    stage: CognitionPromptStage,
    payload: dict[str, object],
) -> None:
    """Validate normalized cognition output shape for one stage."""
```

The validator must validate only normalized return dicts built by existing
handlers. It must not inspect raw LLM text or change payload contents.

The validator is keyed by `stage`, not by variant. The required-keys table
below is the structural contract for every current and future variant of a
stage. Future stages that introduce a second active variant must not change
the normalized schema for an existing stage; they must add new stages or
extend the per-stage required-keys table only with explicitly approved
additive keys.

Required normalized keys:

| Stage | Required normalized keys |
|---|---|
| `l1_subconscious` | `emotional_appraisal: str`, `interaction_subtext: str` |
| `l2a_consciousness` | `internal_monologue: str`, `character_intent: str`, `logical_stance: str` |
| `l2b_boundary_core` | `boundary_core_assessment: dict` |
| `l2c_judgment_core` | `logical_stance: str`, `character_intent: str`, `judgment_note: str` |
| `l3_contextual_agent` | `social_distance: str`, `emotional_intensity: str`, `vibe_check: str`, `relational_dynamic: str`, `expression_willingness: str` |
| `l3_style_agent` | `rhetorical_strategy: str`, `linguistic_style: str`, `forbidden_phrases: list` |
| `l3_content_anchor_agent` | `content_anchors: list` |
| `l3_preference_adapter` | `accepted_user_preferences: list` |
| `l3_visual_agent` | `facial_expression: list`, `body_language: list`, `gaze_direction: list`, `visual_vibe: list` |

Each handler must call `validate_cognition_output_contract` immediately before
returning its normalized dict. Use this exact shape:

```python
result = {
    ...
}
validate_cognition_output_contract(stage="<approved_stage_name>", payload=result)
return result
```

Do not validate raw LLM JSON before the existing handler normalization logic.
Do not create a second dict for validation if the handler already has the
normalized return payload.

## Inactive Variant Notes

Create
`(removed) multi_source_cognition_stage_03_inactive_prompt_variant_notes.md`.

This reference file must describe only inactive future direction for:

- `reflection_signal` with `reflection_artifact`
- `internal_thought` with `internal_monologue`

It must state that these variants are not runtime-enabled in Stage 03, must not
contain full production prompt text, and must not authorize Stage 07 or Stage 08
work.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
- `tests/test_multi_source_cognition_stage_03_prompt_selection.py`
- `(removed) multi_source_cognition_stage_03_inactive_prompt_variant_notes.md`

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
  - Insert selector and output-contract calls without changing prompt text or
    payload keys.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
  - Insert selector and output-contract calls for L2 handlers only.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Insert selector and output-contract calls for L3 LLM-backed handlers only.
- `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
  - Extend prompt-render checks to assert selected current variant and absence
    of selector metadata in human-message payloads.
  - All assertions that exist in this file before Stage 03 must remain
    byte-equivalent. New assertions are append-only. Reordering, rewording,
    or strengthening an existing assertion is a Stage 03 stop condition.
    Renaming a fixture key, changing a regex, or moving an assertion across
    test functions counts as a change and is not allowed.
- `development_plans/active/short_term/multi_source_cognition_architecture_stage_03_shared_cognition_prompt_selection_plan.md`
  - During execution, update checklist and `Execution Evidence` only.
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
  - This review may move the `stage_03` ledger row from `draft` to `approved`.
  - During execution, update the `stage_03` ledger row to `completed` only
    after verification passes.
- `development_plans/README.md`
  - This review may move the Stage 03 registry row from `draft | blocked` to
    `approved | not_started`.
  - During execution, update the Stage 03 registry row only after completion.
    Both `Status` and `Execution` columns must move to `completed` in the same
    edit.

### Keep

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/rag/**`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_*.py`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator*.py`
- database, scheduler, dispatcher, adapter, and reflection-cycle files

## Implementation Order

1. Preflight the Stage 02 handoff.
   - Confirm every path in `Context` exists.
   - Confirm the parent ledger marks `stage_00`, `stage_01`, and `stage_02`
     as `completed`.
   - Recompute the `Prompt Text Fingerprint Guard` table before editing
     L1/L2/L3.
   - Record the preflight in `Execution Evidence`.
2. Add selector and output-contract tests.
   - Cover current text `/chat` selection, unsupported trigger rejection,
     unsupported input-source rejection, unsupported output-mode rejection,
     every approved stage name, and every normalized output shape.
   - Use Stage 01 `build_text_chat_cognitive_episode(...)` for valid selector
     fixtures.
   - Use explicit invalid `CognitiveEpisode` dictionaries for unsupported
     source tests; do not add helper behavior just to make invalid inputs pass.
   - Run the tests and record the expected missing-module or missing-symbol
     failure in `Execution Evidence`.
3. Implement selector and output-contract modules.
   - Keep them deterministic and structural.
   - Run focused module tests until they pass.
4. Insert selector and output-contract calls into L1, L2, and L3 handlers.
   - Use the exact shape from `Handler Insertion Contract`.
   - Do not edit prompt literal text.
   - Do not add selector fields to human-message payloads.
   - Do not change existing LLM variables, model configuration, message order,
     JSON payload fields, parser calls, spelling-tolerance normalization,
     logging, or return key names.
5. Extend prompt-render equivalence tests.
   - Use mocked LLMs.
   - Compare post-insertion render against the Stage 00 frozen reference in
     `tests/fixtures/multi_source_cognition_stage_00_cases.json` plus the
     existing assertions in
     `tests/test_multi_source_cognition_stage_00_regression_baseline.py`.
     The Stage 00 baseline is the only authoritative "before" snapshot;
     do not capture a fresh "before" inside the Stage 03 test run.
   - Assertions on parsed human-message payloads must ignore JSON
     serialization ordering.
   - Assert no human payload contains `cognitive_episode`, `prompt_key`,
     `trigger_source`, or `input_sources`.
6. Add inactive variant reference notes.
7. Run all verification gates.
8. Update this plan's checklist and `Execution Evidence`.
9. Update the parent ledger and registry only after every verification command
   passes.

## Progress Checklist

- [x] Preflight - Stage 02 handoff and prompt fingerprints confirmed.
  - Covers: parent ledger, Stage 00/01/02 execution evidence, Context paths,
    and `Prompt Text Fingerprint Guard`.
  - Verify: all artifact paths exist and fingerprints match this plan.
  - Evidence: recorded path confirmation and hash output in
    `Execution Evidence`.
  - Handoff: next agent starts at Stage 1.
  - Sign-off: `Codex/2026-05-09` after evidence is recorded.
- [x] Stage 1 - focused selector and output-contract tests added.
  - Covers: `tests/test_multi_source_cognition_stage_03_prompt_selection.py`.
  - Verify: focused command fails only because approved modules or symbols do
    not exist yet.
  - Evidence: recorded command and expected failure in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-09` after evidence is recorded.
- [x] Stage 2 - selector and output-contract modules implemented.
  - Covers: the two new `persona_supervisor2_cognition_*` modules.
  - Verify: focused Stage 03 tests pass and `py_compile` passes.
  - Evidence: recorded public API names and test output in
    `Execution Evidence`.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `Codex/2026-05-09` after verification and evidence are recorded.
- [x] Stage 3 - L1/L2/L3 handlers use selector and validators.
  - Covers: cognition L1/L2/L3 modules.
  - Verify: focused Stage 03 tests and prompt-render tests pass.
  - Evidence: recorded prompt-render equivalence output in
    `Execution Evidence`.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `Codex/2026-05-09` after verification and evidence are recorded.
- [x] Stage 4 - inactive variant notes added.
  - Covers: reference note path.
  - Verify: static grep shows no runtime import of the reference note and no
    non-chat variants in source.
  - Evidence: record grep result.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `Codex/2026-05-09` after verification and evidence are recorded.
- [x] Stage 5 - regression gates pass.
  - Covers: Stage 00, Stage 01, Stage 02, prompt render, and adjacent cognition
    tests.
  - Verify: every command in `Verification` passes.
  - Evidence: record exact command results.
  - Handoff: next agent updates lifecycle records.
  - Sign-off: `Codex/2026-05-09` after verification and evidence are recorded.
- [x] Stage 6 - lifecycle records updated.
  - Covers: this plan, parent ledger, and registry.
  - Verify: rows show Stage 03 completed and artifact paths are named.
  - Evidence: record parent ledger and registry confirmation.
  - Handoff: Stage 04 may be reviewed for approval; it must read this plan's
    execution evidence before implementation.
  - Sign-off: `Codex/2026-05-09` after lifecycle updates are recorded.

## Verification

### Focused Stage 03 Tests

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py
```

### Prior Stage Gates

```powershell
venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_02_chat_episode_migration.py
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py
```

### Adjacent Cognition Tests

```powershell
venv\Scripts\python -m pytest tests\test_cognition_clarification_consumers.py tests\test_cognition_interaction_style_context.py
venv\Scripts\python -m pytest tests\test_cognition_live_llm_prompt_contracts.py -m "not live_llm"
```

If the final command deselects all tests because the module is live-only, record
that result and do not run live LLM tests unless the user explicitly approves a
one-case inspected smoke run.

### Static Checks

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_output_contracts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_multi_source_cognition_stage_03_prompt_selection.py
git diff --check
rg -n "reflection_signal|internal_thought|image_observation|audio_observation|scheduled_recall|system_probe" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py
rg -n "reflection_signal|internal_thought|image_observation|audio_observation|scheduled_recall|system_probe" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py
rg -n "cognitive_episode|prompt_key|trigger_source|input_sources" tests\test_multi_source_cognition_stage_00_regression_baseline.py
```

The first `rg` is over the L1/L2/L3 modules and must produce zero matches.
Any match is a Stage 03 stop condition because it means a non-chat source
label leaked into runtime prompt text or runtime selection logic.

The second `rg` is over the selector module. Zero matches are valid and are
preferred because the selector uses a closed allow-list for current chat
sources. If matches appear, they must be only rejection literals or
unsupported-source error text. They must not appear inside prompt template
text, prompt key assembly, or runtime branch conditions. Reviewers must
classify each match as either "rejection literal" or "violation"; any
"violation" match is a Stage 03 stop condition.

The third `rg` must show only Stage 03 assertions that these fields are
absent from prompt human-message payloads.

Prompt fingerprints must also be recomputed with the repository virtual
environment and compared against `Prompt Text Fingerprint Guard`:

```powershell
@'
import ast
import hashlib
from pathlib import Path

EXPECTED = {
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py", "_COGNITION_SUBCONSCIOUS_PROMPT"): ("93b4a80fa69aa7479d77699622aa632dd47a8515c475c91a0921bcdb302dc938", 1768),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py", "_COGNITION_CONSCIOUSNESS_PROMPT"): ("241fb639de242e2d7fc964da922a8b0ea2ac0d9c4f5b2b762df210c34805a5e5", 6795),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py", "_BOUNDARY_CORE_PROMPT"): ("dee7b322eb0d8637a3ee95b386560786042911cd0acca93b7c30896638ef26d1", 5425),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py", "_JUDGEMENT_CORE_PROMPT"): ("ca4e88cc3854cbdb63372ad3b20644575ef9eb74abdc8637212fedc0ca5b3b89", 4012),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py", "_CONTEXTUAL_AGENT_PROMPT"): ("4a2f7735c9f6b45637f329ad10581124360a24049444be43efb43cd2d802baae", 2982),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py", "_STYLE_AGENT_PROMPT"): ("c0f66e0d744688afa4b105f20573708d295057856fa924c0102c0d5605cb6340", 3748),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py", "_CONTENT_ANCHOR_AGENT_PROMPT"): ("9bf38821e24a561cec5c887f54432a4bff7b84131efb6c997d26edab8e0bbea0", 7194),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py", "_PREFERENCE_ADAPTER_PROMPT"): ("f5b0363c0d1ea1f28770237d27908cbfd56a86410c7c64d9522c44e1c284f88d", 4151),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py", "_VISUAL_AGENT_PROMPT"): ("68b1a35d43bfa28c46c91274d946faa9c7edf206f25fa414dabc822592626294", 4337),
}

for (path_name, constant), (expected_digest, expected_length) in EXPECTED.items():
    tree = ast.parse(Path(path_name).read_text(encoding="utf-8"))
    value = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if constant not in names:
            continue
        value = ast.literal_eval(node.value)
        break
    if not isinstance(value, str):
        raise AssertionError(f"{constant} missing from {path_name}")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    print(f"{path_name}:{constant}:{digest}:{len(value)}")
    if digest != expected_digest or len(value) != expected_length:
        raise AssertionError(f"{constant} fingerprint changed")
'@ | venv\Scripts\python -
```

Record the output in `Execution Evidence`.

No real LLM tests are required unless explicitly approved.

## Acceptance Criteria

This plan is complete when:

- All LLM-backed L1/L2/L3 cognition handlers call
  `select_cognition_prompt_variant`.
- Current `/chat` episodes select exactly `text_chat_user_message`.
- Existing prompt constants remain text-equivalent.
- Existing prompt human-message payload keys and values remain equivalent.
- Normalized output contracts are structurally validated for every listed
  L1/L2/L3 stage.
- Unsupported non-chat sources fail closed through
  `CognitionPromptSelectionError`.
- No prompt payload receives raw `cognitive_episode`, percept arrays, selector
  metadata, or future source notes.
- RAG, dialog, consolidation, persistence, adapters, scheduler, and database
  files remain unchanged.
- Stage 00, Stage 01, and Stage 02 deterministic gates pass.
- Stage 04 can consume the recorded handoff evidence without rediscovering
  prompt-selection state.

## Data Migration

No database schema, collection, index, stored-document, cache, or migration work
is allowed or required.

## Operational Steps

No service restart, scheduler operation, adapter operation, deployment step, or
manual runtime intervention is required for local verification. The character
must keep running through the existing `/chat` path.

## Completion Artifact Contract

`stage_03` is not complete until `Execution Evidence` records:

- The prompt-selection module path and public API names.
- The output-contract module path and public API names.
- The active variant name `text_chat_user_message`.
- The L1/L2/L3 handler files changed.
- The prompt-render equivalence test path and command result.
- The Stage 00, Stage 01, and Stage 02 verification commands and results.
- Confirmation that RAG, dialog, consolidation, persistence, adapter,
  scheduler, and database files were not modified.
- Confirmation that parent ledger was updated so `stage_03` is complete.
- Confirmation that `development_plans/README.md` was updated so the Stage 03
  registry row is complete.

## Stage 04 Handoff

Stage 04 must read this plan's `Execution Evidence` before implementation and
must use these facts as its handoff inputs:

- `cognitive_episode` is present in `CognitionState` from Stage 02.
- Current cognition prompt variant is `text_chat_user_message`.
- The active selector API is
  `select_cognition_prompt_variant(episode=..., stage=...)`.
- The active output-contract API is
  `validate_cognition_output_contract(stage=..., payload=...)`.
- Prompt human-message payloads still do not contain `cognitive_episode`,
  `prompt_key`, `trigger_source`, or `input_sources`.
- Prompt selection is cognition-owned and must not be reused for RAG routing.
- RAG request construction remains inline in `persona_supervisor2.stage_1_research`
  until Stage 04 changes it.
- Stage 04 must not change prompt selector behavior or prompt output contracts.
- Stage 04 may read selector decisions as context only if its own approved plan
  explicitly says so; it must not change selector variants, prompt keys, output
  validation, or cognition prompt payloads.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Prompt text changes accidentally | Map current variant to existing constants without editing literal text. | Prompt-render equivalence tests and diff review. |
| Prompt payload grows with episode metadata | Do not add selector metadata to human-message payloads. | Payload absence assertions. |
| Local LLM sees foreign source terminology | Keep future source notes outside runtime prompts. | Static grep and prompt-render tests. |
| Output validation changes semantics | Validate normalized dict types only after existing handler normalization. | Focused output-contract tests. |
| Stage 04 boundary becomes blurry | Record explicit Stage 04 handoff and leave RAG untouched. | Change-surface review and lifecycle evidence. |

## LLM Call And Context Budget

No new LLM calls are allowed.

Before and after for current `/chat` cognition:

| Cognition area | Before | After |
|---|---:|---:|
| L1 calls | unchanged | unchanged |
| L2 calls | unchanged | unchanged |
| L3 calls | unchanged | unchanged |
| Prompt text | unchanged | unchanged |
| Human-message payload keys | unchanged | unchanged |
| Prompt context size | unchanged except negligible in-process selector metadata not sent to LLM | unchanged |

The selector must run in deterministic Python and must not call an LLM, RAG,
database, cache, scheduler, or adapter.

## Execution Evidence

Implementation started on 2026-05-09 on branch
`multi-source-stage-03-prompt-selection`.

Preflight completed:

- Artifact path check confirmed every Stage 00, Stage 01, Stage 02, and Stage
  03 context path exists:
  - `development_plans/active/short_term/multi_source_cognition_architecture_stage_00_current_chat_workflow_regression_baseline_plan.md`
  - `development_plans/active/short_term/multi_source_cognition_architecture_stage_01_cognitive_episode_contract_plan.md`
  - `development_plans/active/short_term/multi_source_cognition_architecture_stage_02_chat_cognitive_episode_migration_plan.md`
  - `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
  - `tests/fixtures/multi_source_cognition_stage_00_cases.json`
  - `src/kazusa_ai_chatbot/cognition_episode.py`
  - `tests/test_cognitive_episode_contract.py`
  - `src/kazusa_ai_chatbot/service.py`
  - `src/kazusa_ai_chatbot/state.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`
- Parent ledger confirmed `stage_00`, `stage_01`, and `stage_02` are
  `completed`; `stage_03` is the active implementation stage.
- Stage 00, Stage 01, and Stage 02 execution evidence was read. Stage 02
  records final combined verification `43 passed in 2.12s`, plus static
  checks with no prompt/RAG/dialog/consolidator consumption of
  `cognitive_episode`.
- Prompt fingerprint preflight command:

```powershell
@'
import ast
import hashlib
from pathlib import Path

EXPECTED = {
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py", "_COGNITION_SUBCONSCIOUS_PROMPT"): ("93b4a80fa69aa7479d77699622aa632dd47a8515c475c91a0921bcdb302dc938", 1768),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py", "_COGNITION_CONSCIOUSNESS_PROMPT"): ("241fb639de242e2d7fc964da922a8b0ea2ac0d9c4f5b2b762df210c34805a5e5", 6795),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py", "_BOUNDARY_CORE_PROMPT"): ("dee7b322eb0d8637a3ee95b386560786042911cd0acca93b7c30896638ef26d1", 5425),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py", "_JUDGEMENT_CORE_PROMPT"): ("ca4e88cc3854cbdb63372ad3b20644575ef9eb74abdc8637212fedc0ca5b3b89", 4012),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py", "_CONTEXTUAL_AGENT_PROMPT"): ("4a2f7735c9f6b45637f329ad10581124360a24049444be43efb43cd2d802baae", 2982),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py", "_STYLE_AGENT_PROMPT"): ("c0f66e0d744688afa4b105f20573708d295057856fa924c0102c0d5605cb6340", 3748),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py", "_CONTENT_ANCHOR_AGENT_PROMPT"): ("9bf38821e24a561cec5c887f54432a4bff7b84131efb6c997d26edab8e0bbea0", 7194),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py", "_PREFERENCE_ADAPTER_PROMPT"): ("f5b0363c0d1ea1f28770237d27908cbfd56a86410c7c64d9522c44e1c284f88d", 4151),
    ("src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py", "_VISUAL_AGENT_PROMPT"): ("68b1a35d43bfa28c46c91274d946faa9c7edf206f25fa414dabc822592626294", 4337),
}

for (path_name, constant), (expected_digest, expected_length) in EXPECTED.items():
    tree = ast.parse(Path(path_name).read_text(encoding="utf-8"))
    value = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if constant not in names:
            continue
        value = ast.literal_eval(node.value)
        break
    if not isinstance(value, str):
        raise AssertionError(f"{constant} missing from {path_name}")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    print(f"{path_name}:{constant}:{digest}:{len(value)}")
    if digest != expected_digest or len(value) != expected_length:
        raise AssertionError(f"{constant} fingerprint changed")
'@ | venv\Scripts\python -
```

Result: exit code 0. All nine prompt constant digests and lengths matched the
`Prompt Text Fingerprint Guard`.

Stage 1 focused red tests added:

- Added `tests/test_multi_source_cognition_stage_03_prompt_selection.py`.
- The test file covers:
  - current text `/chat` prompt selection for every approved stage name
  - accepted Stage 02 output modes `visible_reply`, `think_only`, and `silent`
  - unsupported trigger-source, input-source, output-mode, and unknown-stage
    rejection
  - valid normalized output shape for every approved stage
  - missing-key and wrong-type rejection for every approved output contract
  - unknown output-contract stage rejection

Expected red command:

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py -q
```

Result: 1 collection error in 0.13s. The failure is the expected
`ModuleNotFoundError` for
`kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_output_contracts`, which
has not been implemented yet.

Stage 2 selector and output-contract modules implemented:

- Added
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`.
  Public API names:
  - `CognitionPromptStage`
  - `CognitionPromptVariant`
  - `CognitionPromptSelection`
  - `CognitionPromptSelectionError`
  - `select_cognition_prompt_variant`
- Added
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`.
  Public API names:
  - `CognitionOutputContractError`
  - `validate_cognition_output_contract`
- Active variant name: `text_chat_user_message`.

Focused verification command:

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py -q
```

Result: 36 passed in 0.07s.

Static verification command:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_output_contracts.py tests\test_multi_source_cognition_stage_03_prompt_selection.py
```

Result: exit code 0.

Stage 3 handler wiring completed:

- Changed L1/L2/L3 handler files:
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- Each LLM-backed handler now calls `select_cognition_prompt_variant(...)`,
  maps `text_chat_user_message` to the existing prompt constant, and calls
  `validate_cognition_output_contract(...)` immediately before returning the
  normalized payload.
- `tests/test_multi_source_cognition_stage_00_regression_baseline.py` extends
  the prompt-render baseline with selector-call assertions and absence
  assertions for `prompt_key`, `trigger_source`, and `input_sources`.

Expected prompt-render red command before wiring:

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py::test_existing_cognition_and_dialog_prompts_render_with_mocked_llms -q
```

Result: 1 failed in 2.18s. The failure was the expected `AttributeError`
because `persona_supervisor2_cognition_l1` did not yet expose
`select_cognition_prompt_variant`.

Focused Stage 03 verification command after wiring:

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py -q
```

Result: 36 passed in 0.07s.

Prompt-render verification command after wiring:

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py::test_existing_cognition_and_dialog_prompts_render_with_mocked_llms -q
```

Result: 1 passed in 1.96s.

Syntax verification command:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_output_contracts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_multi_source_cognition_stage_00_regression_baseline.py
```

Result: exit code 0.

Prompt fingerprint guard after handler wiring: exit code 0. All nine prompt
constant digests and lengths still match the `Prompt Text Fingerprint Guard`.

Stage 4 inactive reference notes added:

- Added
  `(removed) multi_source_cognition_stage_03_inactive_prompt_variant_notes.md`.
- The reference note covers only inactive future direction for
  `reflection_signal` with `reflection_artifact` and `internal_thought` with
  `internal_monologue`. It contains no production prompt text and states that
  it does not runtime-enable those labels or authorize future stage work.

Static verification commands:

```powershell
rg -n "multi_source_cognition_stage_03_inactive_prompt_variant_notes" src
rg -n "reflection_signal|internal_thought|image_observation|audio_observation|scheduled_recall|system_probe" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py
rg -n "reflection_signal|internal_thought|image_observation|audio_observation|scheduled_recall|system_probe" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py
```

Result: all three commands returned exit code 1 with zero matches, which is the
expected Stage 4 result.

Stage 5 regression gates completed:

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py
```

Result: 36 passed in 0.07s.

```powershell
venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py
```

Result: 15 passed in 0.05s.

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_02_chat_episode_migration.py
```

Result: 5 passed in 2.28s.

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py
```

Result: 11 passed in 2.05s.

```powershell
venv\Scripts\python -m pytest tests\test_cognition_clarification_consumers.py tests\test_cognition_interaction_style_context.py
```

First result: 3 failed, 4 passed in 1.82s. The failures were stale adjacent
unit fixtures calling L2/L3 handlers directly without the Stage 02 required
`cognitive_episode` state. No production fallback was added. The test fixtures
in `tests/test_cognition_clarification_consumers.py` and
`tests/test_cognition_interaction_style_context.py` were updated to carry valid
text-chat `CognitiveEpisode` objects.

```powershell
venv\Scripts\python -m py_compile tests\test_cognition_clarification_consumers.py tests\test_cognition_interaction_style_context.py
```

Result: exit code 0.

```powershell
venv\Scripts\python -m pytest tests\test_cognition_clarification_consumers.py tests\test_cognition_interaction_style_context.py
```

Rerun result: 7 passed in 1.67s.

```powershell
venv\Scripts\python -m pytest tests\test_cognition_live_llm_prompt_contracts.py -m "not live_llm"
```

Result: exit code 1 with 17 deselected and 0 selected in 1.62s. This is the
documented live-only module case from `Verification`; no live LLM tests were
run.

Static checks:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_output_contracts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_multi_source_cognition_stage_00_regression_baseline.py tests\test_cognition_clarification_consumers.py tests\test_cognition_interaction_style_context.py
```

Result: exit code 0.

```powershell
git diff --check
```

Result: exit code 0. Git reported LF-to-CRLF working-copy warnings only; no
whitespace errors.

```powershell
rg -n "reflection_signal|internal_thought|image_observation|audio_observation|scheduled_recall|system_probe" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py
rg -n "reflection_signal|internal_thought|image_observation|audio_observation|scheduled_recall|system_probe" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py
```

Result: both commands returned exit code 1 with zero matches.

```powershell
rg -n "cognitive_episode|prompt_key|trigger_source|input_sources" tests\test_multi_source_cognition_stage_00_regression_baseline.py
```

Result: exit code 0. Matches were classified as valid Stage 02 episode fixture
construction, text-chat episode contract assertions, prompt human-message
payload absence assertions, and Stage 03 selector tracking assertions. No match
adds `cognitive_episode`, `prompt_key`, `trigger_source`, or `input_sources` to
an LLM human-message payload.

Prompt fingerprint guard after final verification: exit code 0. All nine prompt
constant digests and lengths still match the `Prompt Text Fingerprint Guard`.

Diff surface confirmation:

- Runtime source changes are limited to the approved cognition L1/L2/L3 files
  and the two new cognition contract modules.
- No RAG, dialog, consolidation, persistence, adapter, scheduler, database, or
  service file was modified.
- The only additional code test files touched outside the original Stage 03
  focused/baseline tests are adjacent cognition test fixtures needed to carry
  the now-required Stage 02 `cognitive_episode` state into direct handler
  calls.

Stage 6 lifecycle records updated:

- This plan status is `completed`.
- Parent ledger row in
  `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
  marks `stage_03` as `completed`.
- Registry row in `development_plans/README.md` marks the Stage 03 plan as
  `completed | completed`.
- Stage 04 handoff remains the `Stage 04 Handoff` section above plus this
  execution evidence.

Post-review cleanup:

- Inlined `_raise_selection_error(...)` and `_raise_output_contract_error(...)`
  into direct `raise CognitionPromptSelectionError(...)` and
  `raise CognitionOutputContractError(...)` statements to remove thin-wrapper
  private helpers.
- Updated the selector static-check wording to document the actual closed
  allow-list behavior: zero future-source-label matches are valid and
  preferred.

Post-review verification:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_output_contracts.py
```

Result: exit code 0.

```powershell
rg -n "_raise_selection_error|_raise_output_contract_error|NoReturn" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_output_contracts.py
rg -n "reflection_signal|internal_thought|image_observation|audio_observation|scheduled_recall|system_probe" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py
```

Result: both commands returned exit code 1 with zero matches.

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_multi_source_cognition_stage_00_regression_baseline.py tests\test_multi_source_cognition_stage_02_chat_episode_migration.py tests\test_cognitive_episode_contract.py tests\test_cognition_clarification_consumers.py tests\test_cognition_interaction_style_context.py
```

Result: 74 passed in 3.05s.
