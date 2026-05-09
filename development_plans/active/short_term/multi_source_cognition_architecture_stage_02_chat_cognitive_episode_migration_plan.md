# multi source cognition architecture stage 02 chat cognitive episode migration plan

## Summary

- Goal: Make the current `/chat` workflow build and carry a
  `CognitiveEpisode` while preserving existing behavior through legacy field
  compatibility.
- Plan class: medium
- Status: draft
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` if Python prompt or test strings with CJK content are edited.
- Overall cutover strategy: additive pass-through; the live graph receives the
  episode but existing nodes continue using legacy fields.
- Highest-risk areas: graph state shape drift, changed prompt inputs, changed
  RAG context, debug-mode regression, and accidental use of the episode inside
  cognition before prompt selection is ready.
- Acceptance criteria: `/chat` state carries `CognitiveEpisode`, all legacy
  behavior remains unchanged, and `stage_00` regression baseline passes.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Stage: `stage_02`

## Context

`stage_01` defines the neutral contract. This stage wires the contract into the
current `/chat` state path without changing how RAG, cognition, dialog, or
consolidation make decisions. The purpose is migration safety, not new
capability.

The service currently assembles `initial_state: IMProcessState` in
`src/kazusa_ai_chatbot/service.py` immediately before `_graph.ainvoke`. That is
the correct boundary for building the text-only `/chat` episode.

## Mandatory Skills

- `development-plan-writing`: preserve child-stage scope.
- `local-llm-architecture`: protect live response latency and existing prompt
  contracts.
- `no-prepost-user-input`: do not interpret user intent in deterministic episode
  mapping.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding or running tests.
- `cjk-safety`: load before editing Python files that contain CJK strings.

## Mandatory Rules

- Before editing, read the parent ledger and confirm `stage_00` and `stage_01`
  are completed.
- Before editing, read prior execution evidence and confirm baseline fixture,
  contract module, builder, and validation artifact paths exist.
- Do not change L1/L2/L3 prompt text.
- Do not change RAG query behavior.
- Do not change dialog behavior.
- Do not change consolidation write behavior.
- Do not add support for reflection, internal thought, image, audio, or
  proactive triggers.
- Do not remove legacy fields such as `user_input`, `decontexualized_input`,
  `prompt_message_context`, or `rag_result`.
- Do not add live `/chat` LLM calls.

## Must Do

- Build a text-only `CognitiveEpisode` for normal `/chat` requests.
- Add `cognitive_episode` as an optional graph-state field.
- Pass the episode through the service graph and persona graph.
- Keep existing nodes reading existing legacy fields.
- Add tests proving outputs and key state shapes remain unchanged.
- Rerun the `stage_00` regression baseline.

## Deferred

- Source-aware prompt selection.
- RAG episode adapter.
- Origin-aware consolidation policy changes.
- Non-chat trigger sources.
- Multimodal expansion beyond the current existing image-description path.

## Cutover Policy

Add the episode behind compatibility projection. Existing `/chat` behavior must
be the default and only runtime behavior. If any stage baseline fails, revert
this stage and create a bugfix plan before continuing.

## Agent Autonomy Boundaries

The implementation agent may choose the exact local variable names used to
construct the episode, but must build it at the service state-assembly boundary
and must not make cognition or prompts consume it yet.

## Target State

The `/chat` path has this shape:

```text
service.py builds legacy IMProcessState
-> service.py builds CognitiveEpisode from the same fields
-> graph carries both legacy fields and cognitive_episode
-> persona_supervisor2 passes cognitive_episode through
-> RAG, cognition, dialog, and consolidation still use legacy fields
```

## Design Decisions

- Build the episode in `service.py` after message envelope hydration and before
  `_graph.ainvoke`.
- Add `cognitive_episode: NotRequired[CognitiveEpisode]` to `IMProcessState`.
- Add `cognitive_episode: NotRequired[CognitiveEpisode]` to
  `GlobalPersonaState` and `CognitionState` only as pass-through state.
- Do not include the episode in prompt payloads in this stage.
- Do not remove or rename existing graph nodes such as `stage_1_research` or
  `stage_2_cognition`; those names are current code internals and are unrelated
  to development-plan stage numbering.

## Change Surface

Expected files:

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/state.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `tests/test_cognitive_episode_contract.py`
- `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
- possibly `tests/test_persona_supervisor2.py`
- possibly `tests/test_service_background_consolidation.py`

Forbidden changes:

- Prompt text changes.
- RAG dispatch or projection behavior changes.
- Dialog generator or evaluator prompt changes.
- Consolidator write policy changes.
- Database schema changes.

## Implementation Order

1. Import the `CognitiveEpisode` type into state/schema modules without
   creating circular imports.
2. Add optional `cognitive_episode` fields to `IMProcessState`,
   `GlobalPersonaState`, and `CognitionState`.
3. Build a text-only `/chat` episode in `service.py` from already-hydrated
   request fields.
4. Attach the episode to `initial_state`.
5. Pass `cognitive_episode` from `persona_supervisor2` initial persona state
   into cognition state as optional pass-through.
6. Add tests asserting the episode exists in state but current legacy fields are
   unchanged.
7. Run the full `stage_00` regression baseline.

## Progress Checklist

- [ ] State schemas accept optional `cognitive_episode`.
- [ ] Service builds text-only user-message episode.
- [ ] Persona graph passes episode through.
- [ ] Cognition state can carry episode without prompt use.
- [ ] Tests prove legacy fields remain unchanged.
- [ ] Stage 00 baseline passes.
- [ ] Completion artifacts recorded in `Execution Evidence`.
- [ ] Parent ledger updated with `stage_02` completion status.

## Verification

Required deterministic verification:

```powershell
venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py
venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_service_background_consolidation.py
```

Also run schema and state checks:

```powershell
venv\Scripts\python -m pytest tests\test_state.py tests\test_persona_supervisor2_schema.py
```

No real LLM tests are required unless prompt payloads unexpectedly change. If
prompt payloads change, stop and revise the plan instead of proceeding.

## Acceptance Criteria

- `/chat` builds a `CognitiveEpisode` with
  `trigger_source=user_message`, `input_sources=[dialog_text]`, and
  `output_mode=visible_reply` for normal text turns.
- Existing `/chat` outputs, targeting, delivery tracking, debug modes, RAG
  context, dialog, and consolidation behavior remain unchanged.
- No prompts consume `cognitive_episode`.
- No new live `/chat` LLM calls are added.
- Stage 00 baseline passes.
- Parent ledger can point to this stage's wiring files, tests, and baseline
  rerun evidence.

## Completion Artifact Contract

`stage_02` is not complete until `Execution Evidence` records:

- The service state-assembly file and exact function or section changed.
- The state/schema files that carry `cognitive_episode`.
- The persona/cognition pass-through files changed.
- The test paths proving legacy behavior and pass-through state.
- The exact deterministic commands run and their result, including `stage_00`
  and `stage_01` verification commands.
- Confirmation that the parent ledger was updated so `stage_02` is complete.

## Risks

- Adding the episode to state may accidentally expose it to prompt payloads.
- Passing the episode into consolidation state may increase background payload
  size later; this stage keeps the episode text-only.
- Importing types in both directions can create circular imports; keep contract
  module independent of `state.py`.

## LLM Call And Context Budget

No new LLM calls are allowed. Prompt inputs must remain unchanged. If a test
shows prompt payload drift, treat it as a regression.

## Execution Evidence

Draft only. No implementation has been executed from this plan.
