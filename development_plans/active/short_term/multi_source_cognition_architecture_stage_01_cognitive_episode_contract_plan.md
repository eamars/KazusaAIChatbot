# multi source cognition architecture stage 01 cognitive episode contract plan

## Summary

- Goal: Define the neutral `CognitiveEpisode` contract and text-only `/chat`
  construction helpers without changing runtime behavior.
- Plan class: medium
- Status: draft
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` if tests add CJK strings.
- Overall cutover strategy: additive contract module and tests only; no graph,
  prompt, RAG, dialog, persistence, or consolidation path consumes the contract
  yet.
- Highest-risk areas: circular imports, overfitting the contract to `/chat`,
  leaking implementation-specific storage fields into LLM-facing concepts, and
  adding semantic keyword interpretation in code.
- Acceptance criteria: typed contracts, validators, text-only builder tests,
  and no runtime behavior changes.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Stage: `stage_01`

## Context

The parent architecture needs a source-neutral episode shape before `/chat` can
be migrated safely. This stage defines that shape but does not wire it into the
live graph. The current `/chat` workflow remains the behavior baseline from
`stage_00`.

## Mandatory Skills

- `development-plan-writing`: preserve parent-stage alignment.
- `local-llm-architecture`: keep the contract semantic and compact for a weak
  local model.
- `no-prepost-user-input`: avoid deterministic code deciding user intent,
  permissions, commitments, or relationship meaning.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding or running tests.
- `cjk-safety`: load before editing Python files that contain CJK strings.

## Mandatory Rules

- Before editing, read the parent ledger and confirm `stage_00` is completed.
- Before editing, read `stage_00` execution evidence and confirm the baseline
  test and fixture artifact paths exist.
- Do not wire `CognitiveEpisode` into the production graph in this stage.
- Do not change prompts.
- Do not change RAG input construction.
- Do not change dialog or consolidation behavior.
- Do not import `state.py` from the new contract module if that creates a
  circular import.
- Deterministic validators may validate structure, not semantic user intent.

## Must Do

- Add a new source-neutral contract module.
- Define trigger, input-source, visibility, output-mode, target-scope, percept,
  origin metadata, and episode shapes.
- Add small structural validation helpers.
- Add a text-only `/chat` episode builder that takes primitive field arguments,
  not a full `IMProcessState` dependency.
- Add tests for valid text-only `/chat` episodes and invalid structural cases.

## Deferred

- Adding `cognitive_episode` to `IMProcessState`.
- Passing episodes through `persona_supervisor2`.
- Source-aware prompt selection.
- Reflection, internal thought, image, audio, or proactive trigger support.

## Cutover Policy

There is no runtime cutover. The module is added but remains unused by the live
graph until `stage_02`.

## Agent Autonomy Boundaries

The implementation agent may choose exact helper names if they preserve the
interfaces below. The agent must not add runtime graph wiring, prompt edits, or
new source types beyond the parent-approved literals.

## Target State

The codebase has a stable internal episode contract that can represent the
current `/chat` text turn as:

```text
trigger_source=user_message
input_sources=[dialog_text]
output_mode=visible_reply
```

The contract is compact and semantic. It does not expose database internals to
the model-facing layer.

## Design Decisions

- Use `TypedDict` plus `Literal` aliases to match existing internal state style.
- Keep runtime validation as explicit helper functions returning structured
  errors or raising a narrow `ValueError`; the child implementation plan must
  pick one and use it consistently.
- Keep builders dependent on primitive fields to avoid circular imports between
  `state.py` and the new module.
- Keep compatibility projection separate from the canonical episode fields.

## Change Surface

Expected files:

- Add `src/kazusa_ai_chatbot/cognition_episode.py`.
- Add `tests/test_cognitive_episode_contract.py`.

Forbidden files:

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/state.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- prompt source files
- RAG source files
- consolidator source files

## Contract Sketch

The implementation must define these concepts:

```python
TriggerSource = Literal[
    "user_message",
    "reflection_signal",
    "internal_thought",
    "scheduled_recall",
    "system_probe",
]

InputSource = Literal[
    "dialog_text",
    "image_observation",
    "audio_observation",
    "internal_monologue",
    "reflection_artifact",
    "retrieved_memory",
]

OutputMode = Literal[
    "visible_reply",
    "silent",
    "think_only",
    "preview",
    "scheduled_action_request",
]
```

Required episode fields:

- `episode_id`
- `trigger_source`
- `input_sources`
- `output_mode`
- `percepts`
- `target_scope`
- `origin_metadata`
- `timestamp`
- `time_context`

Text-only `/chat` builder output must include one `dialog_text` percept whose
content is the current `user_input` string.

## Implementation Order

1. Add the contract module with type aliases and `TypedDict` definitions.
2. Add structural validation helper functions.
3. Add a text-only builder that accepts primitive fields used by `/chat`.
4. Add tests for valid user-message episodes.
5. Add tests for invalid empty percepts, unsupported output-mode combinations,
   missing target scope, and mismatched `input_sources`.
6. Run `stage_00` baseline tests to confirm no behavior changed.

## Progress Checklist

- [ ] Contract module added.
- [ ] Trigger and input-source literals added.
- [ ] Episode and percept shapes added.
- [ ] Structural validators added.
- [ ] Text-only `/chat` builder added.
- [ ] Unit tests added.
- [ ] Stage 00 baseline still passes.
- [ ] Completion artifacts recorded in `Execution Evidence`.
- [ ] Parent ledger updated with `stage_01` completion status.

## Verification

Required deterministic verification:

```powershell
venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py
```

Also run:

```powershell
venv\Scripts\python -m pytest tests\test_state.py tests\test_persona_supervisor2_schema.py
```

## Acceptance Criteria

- `CognitiveEpisode` can represent current text `/chat` input.
- Invalid structural episodes are rejected by deterministic validation.
- The new module has no circular import with `state.py`.
- No live graph path consumes the new contract yet.
- Stage 00 baseline still passes.
- Parent ledger can point to this stage's contract module, tests, and
  verification evidence.

## Completion Artifact Contract

`stage_01` is not complete until `Execution Evidence` records:

- The `CognitiveEpisode` contract module path.
- The text-only `/chat` builder path or function name.
- The structural validation helper paths or function names.
- The unit test path.
- The exact deterministic commands run and their result, including the
  `stage_00` baseline rerun.
- Confirmation that the parent ledger was updated so `stage_01` is complete.

## Risks

- The contract may become too broad if it tries to solve later reflection and
  multimodal stages in detail.
- A builder that imports runtime graph state directly can create circular
  dependencies.
- Validation can accidentally become semantic classification; keep it
  structural.

## LLM Call And Context Budget

No LLM calls are added. No prompt context changes are allowed.

## Glossary

- `trigger_source`: why cognition is running.
- `input_source`: what cognition is perceiving.
- `percept`: one normalized unit of perceived content.
- `output_mode`: what the cognition result is allowed to produce.

## Execution Evidence

Draft only. No implementation has been executed from this plan.
