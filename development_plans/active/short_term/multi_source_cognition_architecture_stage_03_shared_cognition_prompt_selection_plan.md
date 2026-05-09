# multi source cognition architecture stage 03 shared cognition prompt selection plan

## Summary

- Goal: Add a source-aware cognition prompt-selection contract for L1/L2/L3
  cognition while keeping the current `/chat` prompt text and prompt payloads
  behaviorally unchanged.
- Plan class: large
- Status: draft
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
  existing cognition outputs, and Stage 00 plus Stage 02 gates pass.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Stage: `stage_03`

Execution status: blocked until `stage_02` is completed and this plan is
explicitly approved.

## Context

Stage 02 moves a valid `CognitiveEpisode` through the current `/chat` graph and
cognition state without prompt or RAG consumption. Stage 03 is the next bridge:
the cognition prompt handlers become source-aware through a narrow selector,
but the only active variant remains the existing text `/chat` path.

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
- The implementation agent may choose private helper names inside the selector
  and output-contract modules only when the public API and behavior remain
  unchanged.
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
| Future prompt drafts | Store as inactive reference notes only. | Keeps later source support visible without enabling it. |

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
- Return `variant="text_chat_user_message"`.
- Return `prompt_key=f"{stage}.text_chat_user_message"`.
- Not inspect percept content, user text, RAG evidence, profile facts,
  relationship state, debug-mode names, or prompt payloads.

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
returning its normalized dict.

## Inactive Variant Notes

Create
`development_plans/reference/multi_source_cognition_stage_03_inactive_prompt_variant_notes.md`.

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
- `development_plans/reference/multi_source_cognition_stage_03_inactive_prompt_variant_notes.md`

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
- `development_plans/active/short_term/multi_source_cognition_architecture_stage_03_shared_cognition_prompt_selection_plan.md`
  - Update checklist and `Execution Evidence` only during execution.
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
  - Update the `stage_03` ledger row to `completed` only after verification
    passes.
- `development_plans/README.md`
  - Update the Stage 03 registry row only after completion. Both `Status` and
    `Execution` columns must move to `completed` in the same edit.

### Keep

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/rag/**`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_*.py`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator*.py`
- database, scheduler, dispatcher, adapter, and reflection-cycle files

## Implementation Order

1. Add selector and output-contract tests.
   - Cover current text `/chat` selection, unsupported trigger rejection,
     unsupported input-source rejection, unsupported output-mode rejection,
     every approved stage name, and every normalized output shape.
   - Run the tests and record the expected missing-module or missing-symbol
     failure in `Execution Evidence`.
2. Implement selector and output-contract modules.
   - Keep them deterministic and structural.
   - Run focused module tests until they pass.
3. Insert selector and output-contract calls into L1, L2, and L3 handlers.
   - Use per-handler prompt maps that map only
     `text_chat_user_message` to the existing prompt constant.
   - Do not edit prompt literal text.
   - Do not add selector fields to human-message payloads.
4. Extend prompt-render equivalence tests.
   - Use mocked LLMs.
   - Capture system prompts and human payloads before and after selector
     insertion through stable assertions.
   - Assert no human payload contains `cognitive_episode`, `prompt_key`,
     `trigger_source`, or `input_sources`.
5. Add inactive variant reference notes.
6. Run all verification gates.
7. Update this plan's checklist and `Execution Evidence`.
8. Update the parent ledger and registry only after every verification command
   passes.

## Progress Checklist

- [ ] Stage 1 - focused selector and output-contract tests added.
  - Covers: `tests/test_multi_source_cognition_stage_03_prompt_selection.py`.
  - Verify: focused command fails only because approved modules or symbols do
    not exist yet.
  - Evidence: record command and expected failure.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 - selector and output-contract modules implemented.
  - Covers: the two new `persona_supervisor2_cognition_*` modules.
  - Verify: focused Stage 03 tests pass and `py_compile` passes.
  - Evidence: record public API names and test output.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - L1/L2/L3 handlers use selector and validators.
  - Covers: cognition L1/L2/L3 modules.
  - Verify: focused Stage 03 tests and prompt-render tests pass.
  - Evidence: record prompt-render equivalence output.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - inactive variant notes added.
  - Covers: reference note path.
  - Verify: static grep shows no runtime import of the reference note and no
    non-chat variants in source.
  - Evidence: record grep result.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - regression gates pass.
  - Covers: Stage 00, Stage 01, Stage 02, prompt render, and adjacent cognition
    tests.
  - Verify: every command in `Verification` passes.
  - Evidence: record exact command results.
  - Handoff: next agent updates lifecycle records.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - lifecycle records updated.
  - Covers: this plan, parent ledger, and registry.
  - Verify: rows show Stage 03 completed and artifact paths are named.
  - Evidence: record parent ledger and registry confirmation.
  - Handoff: Stage 04 may be reviewed for approval; it must read this plan's
    execution evidence before implementation.
  - Sign-off: `<agent/date>` after lifecycle updates are recorded.

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
rg -n "reflection_signal|internal_thought|image_observation|audio_observation|scheduled_recall|system_probe" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py
rg -n "cognitive_episode|prompt_key|trigger_source|input_sources" tests\test_multi_source_cognition_stage_00_regression_baseline.py
```

The first `rg` may match unsupported-source error tests or literal lists only
inside the selector module; it must not match runtime prompt text additions in
the L1/L2/L3 modules. The second `rg` must show only Stage 03 assertions that
these fields are absent from prompt human-message payloads.

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
- Prompt selection is cognition-owned and must not be reused for RAG routing.
- RAG request construction remains inline in `persona_supervisor2.stage_1_research`
  until Stage 04 changes it.
- Stage 04 must not change prompt selector behavior or prompt output contracts.

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

Draft only. No implementation has been executed from this plan.
