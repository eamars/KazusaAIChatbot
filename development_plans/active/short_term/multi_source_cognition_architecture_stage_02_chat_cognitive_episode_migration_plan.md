# multi source cognition architecture stage 02 chat cognitive episode migration plan

## Summary

- Goal: Make the current `/chat` workflow build and carry a
  `CognitiveEpisode` while preserving existing behavior through legacy field
  compatibility.
- Plan class: medium
- Status: approved
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` if Python prompt or test strings with CJK content are edited.
- Overall cutover strategy: compatible additive pass-through; the live graph
  receives the episode, but existing nodes continue using legacy fields.
- Highest-risk areas: graph state shape drift, changed prompt inputs, changed
  RAG context, debug-mode regression, circular imports, and accidental use of
  the episode inside cognition before prompt selection is ready.
- Acceptance criteria: `/chat` state carries a valid `CognitiveEpisode`, all
  legacy behavior remains unchanged, and the Stage 00 regression baseline
  passes after Stage 02 wiring.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Stage: `stage_02`

Execution status: approved; the implementation agent may begin at Stage 1 of
the `Progress Checklist`.

## Context

Stage 00 established the deterministic `/chat` regression baseline. Stage 01
created the source-neutral episode contract and text chat builder without
wiring it into runtime code. This stage wires that completed contract into the
current `/chat` state path without changing how RAG, cognition, dialog, or
consolidation make decisions.

The service currently assembles `initial_state: IMProcessState` in
`src/kazusa_ai_chatbot/service.py` immediately before `_graph.ainvoke`. That is
the only approved boundary for building the text-only `/chat` episode in this
stage.

Prior-stage artifacts that must exist before editing:

- Stage 00 baseline test:
  `tests/test_multi_source_cognition_stage_00_regression_baseline.py`.
- Stage 00 fixture and frozen evidence corpus:
  `tests/fixtures/multi_source_cognition_stage_00_cases.json`.
- Stage 01 contract module:
  `src/kazusa_ai_chatbot/cognition_episode.py`.
- Stage 01 contract tests:
  `tests/test_cognitive_episode_contract.py`.
- Stage 01 public API:
  `CognitiveEpisode`, `build_text_chat_cognitive_episode`,
  `project_text_chat_compatibility_fields`,
  `validate_cognitive_episode`, and
  `CognitiveEpisodeValidationError`.

## Mandatory Skills

- `development-plan-writing`: preserve child-stage scope and lifecycle.
- `local-llm-architecture`: protect live response latency and existing prompt
  contracts.
- `no-prepost-user-input`: do not interpret user intent in deterministic
  episode mapping.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files that contain CJK strings.

## Mandatory Rules

- Before editing, read the parent ledger and confirm `stage_00` and `stage_01`
  are `completed`.
- Before editing, read Stage 00 and Stage 01 execution evidence and confirm
  every artifact path listed in `Context` exists.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or
  final reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Use only the Stage 01 `build_text_chat_cognitive_episode` helper to construct
  the episode. Do not create an alternate builder, duplicate contract shape, or
  hand-build nested episode dictionaries in service code.
- Keep all semantic judgment out of deterministic episode construction. The
  builder may map explicit structural request fields and debug flags only; it
  must not classify user intent, permissions, commitments, preferences, mood,
  relationship meaning, or response desirability.
- Do not change L1/L2/L3 prompt text, prompt payload keys, or prompt examples.
- Do not change RAG query behavior, RAG context keys, RAG dispatch, RAG
  projection, Cache 2 keys, or retrieval agents.
- Do not change dialog behavior, dialog prompts, dialog evaluator rules,
  output targeting, delivery tracking, or assistant persistence.
- Do not change consolidation write behavior, fact harvesting, future-promise
  harvesting, relationship updates, mood updates, reflection updates, or cache
  invalidation.
- Do not add support for reflection, internal thought, scheduled recall, image,
  audio, proactive, or system-probe triggers.
- Do not remove legacy fields such as `user_input`, `decontexualized_input`,
  `prompt_message_context`, `rag_result`, `platform_message_id`,
  `active_turn_platform_message_ids`, or
  `active_turn_conversation_row_ids`.
- Do not add live `/chat` LLM calls.
- Do not add database schema changes, migrations, new persistence writes, new
  queues, new schedulers, new adapter behavior, or service startup behavior.

## Must Do

- Build a text-only `CognitiveEpisode` for each surviving `/chat` request at
  the service state-assembly boundary.
- Add `cognitive_episode` as an optional graph-state field.
- Pass the episode through `persona_supervisor2` and the cognition subgraph as
  inert state.
- Keep existing nodes reading existing legacy fields.
- Add focused Stage 02 tests proving the episode exists and legacy fields are
  unchanged.
- Rerun Stage 01 contract tests and the Stage 00 regression baseline.
- Record execution evidence and lifecycle updates only after verification
  passes.

## Deferred

- Source-aware prompt selection. This belongs to Stage 03.
- RAG episode adapter work. This belongs to Stage 04.
- Consolidation origin metadata threading. This belongs to Stage 05.
- Consolidator per-write origin policy. This belongs to Stage 06.
- Reflection-triggered cognition, internal thought, scheduled recall,
  multimodal expansion, proactive preview, proactive delivery, or permission
  policy.
- Any production use of `project_text_chat_compatibility_fields`.
- Any change that treats reflection, internal thought, image, or audio as a
  fake user message.

## Cutover Policy

Policy: `compatible`.

Add the episode behind compatibility projection. Existing `/chat` behavior
remains the default and only runtime behavior. The graph carries
`cognitive_episode`, but live RAG, cognition, dialog, persistence, and
consolidation continue to consume the pre-existing legacy fields.

Rollback path: remove the `cognitive_episode` imports, optional state fields,
service construction call, persona pass-through assignments, and Stage 02 tests.
No database rollback is required because this stage creates no persisted data.

Stop condition: if prompt payloads, RAG context, dialog output, targeting,
delivery tracking, persistence, consolidation, debug-mode behavior, or Stage 00
baseline behavior changes, stop Stage 02 and create a bugfix plan before
continuing multi-source cognition work.

## Agent Autonomy Boundaries

- The implementation agent may choose small private helper placement only when
  the exact contracts in this plan remain unchanged.
- The implementation agent must not choose alternate episode id formats,
  output-mode mapping rules, target-scope sources, state-key names, or
  integration boundaries.
- The implementation agent must not introduce new architecture, compatibility
  layers, fallback paths, feature flags, prompt variants, RAG adapters,
  consolidation policies, or non-chat source support.
- The implementation agent must treat every edit outside the files named in
  `Change Surface` as out of scope.
- If a required instruction is impossible, the agent must stop and report the
  blocker instead of inventing a substitute.

## Target State

The `/chat` path has this shape:

```text
service.py builds legacy IMProcessState
-> service.py builds CognitiveEpisode from the same structural fields
-> graph carries both legacy fields and cognitive_episode
-> persona_supervisor2 passes cognitive_episode through
-> cognition subgraph carries cognitive_episode as inert state
-> RAG, cognition prompts, dialog, persistence, and consolidation still use legacy fields
```

No model-facing prompt receives `cognitive_episode` in this stage.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Episode builder | Use `build_text_chat_cognitive_episode` from `src/kazusa_ai_chatbot/cognition_episode.py`. | Stage 01 owns the contract; Stage 02 must not fork it. |
| Build boundary | Build in `service.py` after `user_input`, `prompt_message_context`, active turn ids, `debug_modes`, and `time_context` are available, immediately before `initial_state` is finalized. The episode builder does not consume `promoted_reflection_context`; it is intentionally excluded from this list. | This is after message-envelope hydration and before graph invocation. |
| State field name | Use exactly `cognitive_episode`. | Keeps parent-stage terminology stable. |
| Episode ids | Use the exact deterministic formula in `Episode Construction Contract`. | Prevents implementation-agent creativity and gives tests a fixed contract. |
| Output mode | Map explicit debug flags only: `listen_only -> silent`, else `think_only -> think_only`, else `visible_reply`. | This preserves runtime behavior because the field is inert while making the episode structurally honest. |
| Target scope | Use `prompt_message_context["addressed_to_global_user_ids"]` and `prompt_message_context["broadcast"]`. | These fields describe the inbound percept target; existing response targeting remains unchanged. |
| Image attachments | Keep current image descriptions flattened into `user_input` and one `dialog_text` percept. | Multimodal percepts are deferred to Stage 09. |
| Prompt consumption | Do not include `cognitive_episode` in any prompt payload. | Stage 03 owns prompt selection and prompt contract changes. |
| RAG consumption | Do not use `cognitive_episode` to build RAG calls. | Stage 04 owns the RAG episode adapter. |

## Episode Construction Contract

`service.py` must import only these Stage 01 symbols:

```python
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    build_text_chat_cognitive_episode,
)
```

`state.py` and `persona_supervisor2_schema.py` may import only
`CognitiveEpisode` for type annotations. Use a runtime import (a plain
`from kazusa_ai_chatbot.cognition_episode import CognitiveEpisode`), not a
`TYPE_CHECKING` guard. The contract module already avoids state/schema imports,
so a runtime import will not introduce a cycle.

`service.py` must define this private helper or equivalent private helper with
the same inputs and returned values:

```python
def _build_text_chat_episode_ids(
    *,
    platform: str,
    platform_channel_id: str,
    platform_message_id: str,
    conversation_row_id: str | None,
    queue_sequence: int,
) -> tuple[str, str]:
```

The helper must implement this exact formula:

```python
message_reference = platform_message_id or conversation_row_id or f"queue-{queue_sequence}"
channel_reference = platform_channel_id or "direct"
episode_id = f"user_message:{platform}:{channel_reference}:{message_reference}"
percept_id = f"{episode_id}:dialog_text:0"
```

Truthiness contract for the `or` chains: empty string (`""`) and `None` are
both falsy and must fall through to the next term. Do not introduce additional
coercion, stripping, or normalization. Pass `req.platform_message_id` and
`req.platform_channel_id` through unchanged (they are typed `str` and may be
`""`); pass `item.conversation_row_id or None` so that the empty-string default
collapses to `None` for the helper's `str | None` parameter.

The helper must be invoked from `service.py` with these exact source bindings:

```python
episode_id, percept_id = _build_text_chat_episode_ids(
    platform=req.platform,
    platform_channel_id=req.platform_channel_id,
    platform_message_id=req.platform_message_id,
    conversation_row_id=item.conversation_row_id or None,
    queue_sequence=item.sequence,
)
```

The `build_text_chat_cognitive_episode` call must use these exact inputs:

```python
episode = build_text_chat_cognitive_episode(
    episode_id=episode_id,
    percept_id=percept_id,
    timestamp=item.timestamp,
    time_context=time_context,
    user_input=user_input,
    platform=req.platform,
    platform_channel_id=req.platform_channel_id,
    channel_type=req.channel_type,
    platform_message_id=req.platform_message_id,
    platform_user_id=req.platform_user_id,
    global_user_id=global_user_id,
    user_name=req.display_name,
    active_turn_platform_message_ids=active_turn_platform_message_ids,
    active_turn_conversation_row_ids=active_turn_conversation_row_ids,
    debug_modes=debug_modes,
    output_mode=episode_output_mode,
    target_addressed_user_ids=list(prompt_message_context["addressed_to_global_user_ids"]),
    target_broadcast=bool(prompt_message_context["broadcast"]),
)
```

`episode_output_mode` must be assigned by this priority order:

```python
if debug_modes["listen_only"]:
    episode_output_mode = "silent"
elif debug_modes["think_only"]:
    episode_output_mode = "think_only"
else:
    episode_output_mode = "visible_reply"
```

The resulting episode must be stored in `initial_state["cognitive_episode"]`.
No call site may store the result under any other key.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/service.py`
  - Add the private id helper.
  - Build the text chat episode from the exact contract above.
  - Add `"cognitive_episode": episode` to `initial_state`.
- `src/kazusa_ai_chatbot/state.py`
  - Add `cognitive_episode: NotRequired[CognitiveEpisode]` to
    `IMProcessState`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Add `cognitive_episode: NotRequired[CognitiveEpisode]` to
    `GlobalPersonaState` and `CognitionState`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Copy `state.get("cognitive_episode")` into `initial_persona_state` under
    the key `cognitive_episode` when present. Do not rename, wrap, or nest the
    value.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Copy `state.get("cognitive_episode")` into the cognition subgraph initial
    state under the key `cognitive_episode` when present. Do not rename, wrap,
    or nest the value.
- `tests/test_state.py`
  - Assert `IMProcessState` exposes optional `cognitive_episode`.
- `tests/test_persona_supervisor2_schema.py`
  - Assert `GlobalPersonaState` and `CognitionState` expose optional
    `cognitive_episode`.
- `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
  - Add Stage 02 expectations that service graph state includes the episode
    while existing legacy fields and debug behavior remain unchanged.
- `development_plans/active/short_term/multi_source_cognition_architecture_stage_02_chat_cognitive_episode_migration_plan.md`
  - Update checklist and `Execution Evidence` only during execution.
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
  - Update the `stage_02` ledger row to `completed` only after all verification
    passes.
- `development_plans/README.md`
  - Update the Stage 02 registry row only after completion. Both `Status` and
    `Execution` columns must move to `completed` in the same edit.

### Create

- `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`
  - Focused Stage 02 tests for service construction, id derivation,
    debug-mode output-mode mapping, persona pass-through, cognition pass-through,
    and prompt/RAG non-consumption.

### Keep

- `tests/test_cognitive_episode_contract.py`
  - Reuse as verification. Do not move Stage 01 contract tests into Stage 02.

### Forbidden

- Prompt source text in `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`, and
  `src/kazusa_ai_chatbot/nodes/dialog_agent.py`.
- RAG source files, including
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_*.py` and
  `src/kazusa_ai_chatbot/rag/**`.
- Consolidator source files, including
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator*.py`.
- Database schema, migration, bootstrap, cache invalidation, scheduler,
  dispatcher, adapter, and reflection-cycle files.

## Implementation Order

1. Add Stage 02 focused tests before implementation.
   - Cover the private id helper, output-mode mapping, service
     `initial_state["cognitive_episode"]`, legacy field equality, persona
     pass-through, cognition pass-through, and prompt/RAG non-consumption.
   - Run the focused test and record the expected missing-field or
     missing-helper failure in `Execution Evidence`.
2. Add optional state fields.
   - Import `CognitiveEpisode` into `state.py` and
     `persona_supervisor2_schema.py`.
   - Add `cognitive_episode: NotRequired[CognitiveEpisode]` to the approved
     state shapes.
3. Build the episode in `service.py`.
   - Add the private id helper.
   - Compute `episode_output_mode` using the exact priority order.
   - Call `build_text_chat_cognitive_episode` with the exact inputs in
     `Episode Construction Contract`.
   - Attach the resulting episode to `initial_state`.
4. Pass the episode through persona and cognition state.
   - Copy the key from `IMProcessState` to `GlobalPersonaState`.
   - Copy the key from `GlobalPersonaState` to `CognitionState`.
   - Do not add the key to any prompt payload.
5. Run focused Stage 02 tests and iterate only inside the approved change
   surface.
6. Run Stage 01 contract tests and Stage 00 baseline tests.
7. Run static greps proving no prompt, RAG, dialog, or consolidator source
   consumes `cognitive_episode`.
8. Update this plan's checklist and `Execution Evidence`.
9. Update the parent ledger and registry only after every verification command
   passes.

## Progress Checklist

- [ ] Stage 1 - focused Stage 02 tests added.
  - Covers: `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`.
  - Verify: focused command fails only because approved Stage 02 symbols or
    state fields do not exist yet.
  - Evidence: record command and failure summary in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 - state schemas accept optional `cognitive_episode`.
  - Covers: `src/kazusa_ai_chatbot/state.py` and
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`.
  - Verify: `tests/test_state.py` and `tests/test_persona_supervisor2_schema.py`.
  - Evidence: record changed fields and test output.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - service builds text-only user-message episode.
  - Covers: `src/kazusa_ai_chatbot/service.py`.
  - Verify: focused Stage 02 service tests pass.
  - Evidence: record id/output-mode mapping test output.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - persona and cognition pass-through complete.
  - Covers: `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` and
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`.
  - Verify: focused pass-through tests pass.
  - Evidence: record pass-through assertion output.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - regression and non-consumption gates pass.
  - Covers: Stage 00 baseline, Stage 01 contract tests, static greps, and
    adjacent persona/service tests.
  - Verify: every command in `Verification` passes. The only acceptable
    non-zero exit is exit code 1 from the final `rg` no-match command in
    `Static Checks`; that result must be recorded as expected. Any other
    non-zero exit must be treated as a Stage 02 failure.
  - Evidence: record exact command results.
  - Handoff: next agent updates lifecycle records.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - lifecycle records updated.
  - Covers: this plan, parent ledger, and registry.
  - Verify: rows show Stage 02 completed and artifact paths are named.
  - Evidence: record parent ledger and registry confirmation.
  - Handoff: Stage 02 is complete; Stage 03 remains blocked until separately
    approved.
  - Sign-off: `<agent/date>` after lifecycle updates are recorded.

## Verification

### Focused Stage 02 Tests

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_02_chat_episode_migration.py
```

### Prior Stage Gates

```powershell
venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py
```

### Adjacent Deterministic Tests

```powershell
venv\Scripts\python -m pytest tests\test_state.py tests\test_persona_supervisor2_schema.py
venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_service_background_consolidation.py
venv\Scripts\python -m pytest tests\test_persona_supervisor2_rag_skip_shape.py
```

### Static Checks

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py tests\test_multi_source_cognition_stage_02_chat_episode_migration.py
git diff --check
rg -n "cognitive_episode|CognitiveEpisode" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py src\kazusa_ai_chatbot\nodes\dialog_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_initializer.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_dispatch.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_projection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py
```

The final `rg` command must return no matches. Exit code 1 from that specific
no-match grep is acceptable and must be recorded as expected.

No real LLM tests are required. If prompt payloads change, stop and revise the
plan instead of proceeding.

## Acceptance Criteria

This plan is complete when:

- `/chat` builds a validated `CognitiveEpisode` with
  `trigger_source=user_message` and `input_sources=["dialog_text"]`.
- Normal text turns use `output_mode="visible_reply"`, `think_only` turns use
  `output_mode="think_only"`, and `listen_only` turns use
  `output_mode="silent"`.
- The episode target scope comes from `prompt_message_context` addressing and
  broadcast fields.
- Existing `/chat` outputs, targeting, delivery tracking, debug modes, RAG
  context, dialog, persistence, and consolidation behavior remain unchanged.
- No prompt payload consumes `cognitive_episode`.
- No RAG call consumes `cognitive_episode`.
- No new live `/chat` LLM calls are added.
- Stage 01 contract tests and Stage 00 baseline pass.
- Parent ledger and registry can point to this stage's wiring files, tests, and
  baseline rerun evidence.

## Data Migration

No database schema, collection, index, stored-document, cache, or migration work
is allowed or required.

## Operational Steps

No service restart, scheduler operation, adapter operation, deployment step, or
manual runtime intervention is required for local verification. The character
must keep running on the existing `/chat` path during and after this stage.

## Completion Artifact Contract

`stage_02` is not complete until `Execution Evidence` records:

- The service state-assembly file and exact helper/function names changed.
- The state/schema files that carry `cognitive_episode`.
- The persona and cognition pass-through files changed.
- The focused Stage 02 test path.
- The Stage 00 and Stage 01 artifact paths used as gates.
- The exact deterministic commands run and their results.
- The no-match prompt/RAG/dialog/consolidator grep result.
- Confirmation that the parent ledger was updated so `stage_02` is complete.
- Confirmation that `development_plans/README.md` was updated so the Stage 02
  registry row is complete.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Episode leaks into prompt payloads | Do not reference `cognitive_episode` in prompt handlers or prompt payload builders. | Static no-match grep and Stage 00 prompt-render baseline. |
| RAG context changes by accident | Keep Stage 04 adapter deferred and do not touch RAG files. | Static no-match grep and RAG skip/baseline tests. |
| Debug-mode behavior regresses | Map debug flags only into inert episode output mode; leave service behavior unchanged. | Focused Stage 02 debug-mode tests and Stage 00 debug-mode tests. |
| Circular imports | Keep `cognition_episode.py` independent of state modules; import only the contract type into state/schema. | `py_compile` and focused imports. |
| Agent invents non-chat support | Explicitly forbid non-chat triggers in `Deferred` and `Forbidden`. | Change-surface review and static grep. |

## LLM Call And Context Budget

No new LLM calls are allowed.

Before and after for the live `/chat` path:

| Stage | Before | After |
|---|---:|---:|
| Relevance/listen gate | unchanged | unchanged |
| Message decontextualizer | unchanged | unchanged |
| RAG initializer/dispatcher/evaluator/finalizer | unchanged | unchanged |
| L1/L2/L3 cognition | unchanged | unchanged |
| Dialog generator/evaluator | unchanged | unchanged |
| Consolidation background calls | unchanged | unchanged |

Prompt text and prompt payload keys must remain unchanged. The episode is
deterministic graph state only. If any prompt-render or capture test shows
`cognitive_episode` in a prompt payload, treat it as a Stage 02 regression.

## Execution Evidence

Draft only. No implementation has been executed from this plan.
