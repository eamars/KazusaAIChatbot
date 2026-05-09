# multi source cognition architecture stage 05 consolidation origin metadata threading plan

## Summary

- Goal: Thread `CognitiveEpisode` origin metadata into the consolidation
  subgraph state without changing current `/chat` consolidation writes.
- Plan class: large
- Status: approved
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`; apply
  `cjk-safety` before editing any Python file that contains CJK prompt text.
- Overall cutover strategy: bigbang inside `call_consolidation_subgraph(...)`.
  Normal `/chat` consolidation must require a valid text-chat
  `CognitiveEpisode`; unsupported origins fail before graph construction.
- Highest-risk areas: origin metadata leaking into prompt payloads or durable
  metadata, direct consolidation tests missing required episode state, and
  accidental Stage 06 write-policy behavior landing early.
- Acceptance criteria: `ConsolidatorState["consolidation_origin"]` exists;
  `call_consolidation_subgraph(...)` seeds it before node execution; current
  `/chat` return keys and writes are unchanged; unsupported origins fail before
  any consolidator node runs; all Verification gates pass.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Lifecycle: Stage 04 completed and was merged into `main` with commit `2131985`
on 2026-05-10. Stage 05 is approved for execution. Stage 06 remains draft and
blocked until this plan records execution evidence.

## Context

Stages 01 through 04 moved the live `/chat` path onto `CognitiveEpisode`, then
made cognition prompt selection and RAG request construction consume the episode
through narrow adapters. Consolidation is still the last major live-chat stage
that cannot see the episode origin. It currently receives only the plain
`GlobalPersonaState` fields plus `rag_result`.

This stage adds origin visibility only. It does not ask any LLM prompt to reason
about origin, does not change persistence schema, and does not change facts,
promises, relationship, mood, character image, scheduler, or cache invalidation
policy. Stage 06 owns those per-write decisions.

## Stage Handoff

### From Stage 04

Stage 04 artifacts that Stage 05 must preserve:

- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
- `tests/test_rag_cognitive_episode_adapter.py`
- `tests/test_persona_supervisor2_rag2_integration.py`
- `tests/test_persona_supervisor2_rag_skip_shape.py`
- `tests/test_rag_projection.py`
- Stage 04 execution evidence showing `889 passed, 217 deselected`.

Runtime facts Stage 05 can rely on:

- `state["cognitive_episode"]` is present on normal `/chat` graph state.
- `stage_1_research` still returns the existing `rag_result` shape.
- RAG internals still do not consume raw `CognitiveEpisode` objects.
- Dialog and consolidation were not changed by Stage 04.

Before implementation, the agent must reread Stage 04 `Execution Evidence` and
record that reread in this plan's `Execution Evidence`.

### To Stage 06

After Stage 05, Stage 06 can rely on:

- `ConsolidatorState["consolidation_origin"]` is present for current text-chat
  user-message turns.
- `consolidation_origin` contains identifiers and source labels only; it
  contains no raw user text, percept content, attachments, prompt payloads,
  `rag_result`, facts, or promises.
- Unsupported non-text-chat origins fail before consolidation nodes run.
- `consolidation_origin` is not returned inside `consolidation_metadata`.
- No per-write origin policy has been implemented.

Stage 06 must read this plan's `Execution Evidence` before approval or
execution.

## Mandatory Skills

- `development-plan-writing`: preserve staged lifecycle, handoff, and
  historical evidence.
- `local-llm-architecture`: keep origin metadata out of prompt payloads in this
  stage.
- `no-prepost-user-input`: do not add deterministic interpretation, filtering,
  or rewrite logic for user-authored facts or promises.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py` if the
  edit touches or rewrites nearby CJK prompt/test text; this plan should not
  require prompt edits.

## Mandatory Rules

- Execute only from a feature branch forked from current `main`.
- Keep edits inside the approved Change Surface.
- Use PowerShell `-LiteralPath '...'` for filesystem paths that may contain
  spaces; prefer repo-relative paths in commands.
- Do not change consolidator prompts, prompt payloads, parser contracts, or LLM
  output schemas.
- Do not add origin metadata to LLM messages.
- Do not add `consolidation_origin` or any origin field to
  `consolidation_metadata`.
- Do not change facts, promises, relationship, mood, character image,
  persistence, scheduler, cache invalidation, or write-policy behavior.
- Do not filter, reclassify, or rewrite LLM-emitted facts or promises.
- Do not add feature flags, fallback origin builders, alternate consolidation
  paths, compatibility shims, private raising-only helpers, or pass-through
  wrappers.
- `call_consolidation_subgraph(...)` must plain-index
  `global_state["cognitive_episode"]`. Do not synthesize a missing episode and
  do not use `.get()` for this field.
- Build `consolidation_origin` before constructing `StateGraph`. Unsupported
  origins must fail before any graph node can be registered or run.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.

## Must Do

- Create the consolidation origin projection module specified in `Origin
  Contract`.
- Add a top-level `consolidation_origin` field to `ConsolidatorState`.
- Build `consolidation_origin` once at the start of
  `call_consolidation_subgraph(...)`, before `StateGraph(ConsolidatorState)`.
- Add `consolidation_origin` to the initial subgraph state.
- Add focused unit tests for exact metadata projection and fail-closed origin
  rejection.
- Add integration tests proving every patched consolidation node receives
  `consolidation_origin`, unsupported origins fail before graph construction,
  and normal `/chat` return keys remain unchanged.
- Update only direct fixtures that call `call_consolidation_subgraph(...)`.
- Run every Verification command and record evidence.
- Flip lifecycle rows to `completed` only after verification passes.

## Deferred

- Per-write origin policy for any write path.
- Reflection, internal thought, scheduled recall, system probe, image, audio,
  retrieved memory, or reflection artifact origins writing memory.
- Prompt changes that ask LLMs to reason about origin.
- Persistence schema changes and new durable MongoDB metadata fields.
- Scheduler, dispatcher, autonomous-contact, or proactive-output changes.
- RAG, cognition, dialog, adapter-delivery, or message-envelope changes.

## Cutover Policy

Overall strategy: bigbang inside `call_consolidation_subgraph(...)`.

| Area | Policy | Instruction |
|---|---|---|
| Origin metadata construction | bigbang | Build `consolidation_origin` from `global_state["cognitive_episode"]` before graph construction. No fallback. |
| Normal `/chat` consolidation behavior | compatible | Preserve existing node prompts, persistence writes, return keys, and metadata shape. |
| Unsupported origins | bigbang | Raise `ConsolidationOriginError` before `StateGraph` is instantiated. Do not run nodes. |
| Direct tests | compatible | Add valid text-chat `CognitiveEpisode` fixtures only where direct calls now need them. |

Rollback path: remove the new origin module, remove the `ConsolidatorState`
field, remove the builder call and initial-state key from
`call_consolidation_subgraph(...)`, and revert Stage 05 tests/fixtures. No
database rollback is required.

## Agent Autonomy Boundaries

Allowed implementation choices:

- local test helper names;
- local variable names;
- assertion ordering inside the named tests.

Not allowed:

- any new prompt, LLM call, parser, retry loop, or prompt-render behavior;
- changing facts, promises, relationship, mood, character image, persistence,
  scheduler, dispatcher, or cache invalidation behavior;
- adding origin fields to prompt payloads or durable metadata;
- broad style rewrites of consolidator modules;
- changing files in `Keep` unless a verification failure proves this plan is
  incomplete. If that happens, stop and update this plan before continuing.

## Target State

`call_consolidation_subgraph(...)` has this shape:

```python
consolidation_origin = build_user_message_consolidation_origin(
    episode=global_state["cognitive_episode"],
)
sub_agent_builder = StateGraph(ConsolidatorState)
...
sub_state: ConsolidatorState = {
    ...
    "metadata": {},
    "consolidation_origin": consolidation_origin,
}
```

The final return value remains exactly:

```python
{
    "mood": mood,
    "global_vibe": global_vibe,
    "reflection_summary": reflection_summary,
    "subjective_appraisals": subjective_appraisals,
    "affinity_delta": affinity_delta,
    "last_relationship_insight": last_relationship_insight,
    "new_facts": new_facts,
    "future_promises": future_promises,
    "consolidation_metadata": metadata,
}
```

## Origin Contract

Create `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`.

Public names:

```python
from typing import TypedDict

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    InputSource,
    OutputMode,
    TriggerSource,
)


class ConsolidationOriginError(ValueError):
    """Raised when an episode cannot enter current consolidation."""


class ConsolidationOriginMetadata(TypedDict):
    episode_id: str
    trigger_source: TriggerSource
    input_sources: list[InputSource]
    output_mode: OutputMode
    timestamp: str
    platform: str
    platform_channel_id: str
    channel_type: str
    platform_message_id: str
    active_turn_platform_message_ids: list[str]
    active_turn_conversation_row_ids: list[str]
    current_platform_user_id: str
    current_global_user_id: str
    current_display_name: str
```

Public function:

```python
def build_user_message_consolidation_origin(
    *,
    episode: CognitiveEpisode,
) -> ConsolidationOriginMetadata:
    ...
```

Required behavior order:

1. Call `validate_cognitive_episode(episode)`.
2. If `episode["trigger_source"] != "user_message"`, raise
   `ConsolidationOriginError`.
3. If `episode["input_sources"] != ["dialog_text"]`, raise
   `ConsolidationOriginError`.
4. If `episode["output_mode"]` is not `"visible_reply"`, `"think_only"`, or
   `"silent"`, raise `ConsolidationOriginError`.
5. Return the exact metadata shape above.

Projection rules:

- `episode_id`, `trigger_source`, `input_sources`, `output_mode`,
  `timestamp`, and `time_context` source fields come from the episode.
- `platform`, `platform_channel_id`, `channel_type`,
  `current_platform_user_id`, `current_global_user_id`, and
  `current_display_name` come from `episode["target_scope"]`.
- `platform_message_id`, `active_turn_platform_message_ids`, and
  `active_turn_conversation_row_ids` come from
  `episode["origin_metadata"]`.
- List values in the returned metadata must be new lists.
- Do not include raw percept content, raw user text, `decontexualized_input`,
  prompt payloads, attachments, `rag_result`, facts, promises, or debug modes.
- Use direct `raise ConsolidationOriginError(...)` statements. Do not add a
  private raising-only helper.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Module ownership | Put origin projection beside consolidator nodes, not in RAG or cognition. | The metadata is a consolidator input contract. |
| Runtime support | Support only `trigger_source=user_message` and `input_sources=["dialog_text"]`. | Stage 05 preserves current `/chat`; later origins are not enabled. |
| Output modes | Allow `visible_reply`, `think_only`, and `silent`. | Current `/chat` can respond, think only, or choose no visible reply. |
| Prompt exposure | Do not pass origin metadata to any LLM payload. | Stage 05 is deterministic visibility only. |
| Return shape | Do not return `consolidation_origin` in `consolidation_metadata`. | Stage 06 will own policy/audit metadata if needed. |

## Change Surface

### Create

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py` —
  public origin projection contract for consolidation.
- `tests/test_consolidation_origin_metadata.py` — focused origin projection
  and integration tests.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py` —
  import `ConsolidationOriginMetadata` and add
  `consolidation_origin: ConsolidationOriginMetadata`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py` — import
  the builder, build origin before `StateGraph`, and seed the initial state.
- `tests/test_consolidator_efficiency.py` — add a valid text-chat
  `cognitive_episode` fixture to `_global_state()` because it directly calls
  `call_consolidation_subgraph(...)`.
- `development_plans/active/short_term/multi_source_cognition_architecture_stage_05_consolidation_origin_metadata_threading_plan.md` —
  record progress and execution evidence only.
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md` —
  flip `stage_05` to `completed` after verification passes.
- `development_plans/README.md` — flip Stage 05 to `completed | completed`
  after verification passes.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_dialog.py`
- `src/kazusa_ai_chatbot/rag/*.py`
- `tests/test_consolidator_facts_rag2.py`
- `tests/test_consolidator_reflection_prompts.py`
- `tests/test_service_background_consolidation.py`

## Implementation Order

1. Reread Stage 04 `Execution Evidence`.
   - File:
     `development_plans/active/short_term/multi_source_cognition_architecture_stage_04_rag_cognitive_episode_adapter_plan.md`
   - Evidence: record the Stage 04 commit and full-suite result in this plan.
2. Add focused origin tests.
   - File: `tests/test_consolidation_origin_metadata.py`
   - Tests:
     `test_build_user_message_consolidation_origin_returns_exact_metadata`,
     `test_origin_metadata_copies_list_fields`,
     `test_origin_metadata_excludes_content_and_prompt_fields`,
     `test_origin_rejects_non_user_message_trigger`,
     `test_origin_rejects_non_dialog_text_sources`,
     `test_origin_rejects_unsupported_output_mode`.
   - Verify: `venv\Scripts\python -m pytest tests\test_consolidation_origin_metadata.py -q`
   - Expected before implementation: collection error or import error because
     the module does not exist.
3. Create the origin module.
   - File:
     `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
   - Verify:
     `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin.py`
     and rerun the Step 2 test file.
4. Add `consolidation_origin` to `ConsolidatorState`.
   - File:
     `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py`
   - Verify:
     `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_schema.py`
5. Cut `call_consolidation_subgraph(...)` over to seed origin metadata.
   - File: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
   - Action: build origin before `StateGraph`, add the initial-state key, and
     leave return-value construction unchanged.
   - Verify:
     `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py`
6. Update the direct consolidation fixture.
   - File: `tests/test_consolidator_efficiency.py`
   - Action: add a helper using `build_text_chat_cognitive_episode(...)` and
     include `cognitive_episode` in `_global_state()`.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_consolidator_efficiency.py -q`
7. Add integration assertions to `tests/test_consolidation_origin_metadata.py`.
   - Tests:
     `test_call_consolidation_subgraph_threads_origin_to_all_nodes`,
     `test_unsupported_origin_fails_before_state_graph_construction`,
     `test_call_consolidation_subgraph_does_not_return_origin_metadata`.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_consolidation_origin_metadata.py tests\test_consolidator_efficiency.py -q`
8. Run the full Verification section.
9. Flip lifecycle rows only after verification passes.
10. Record execution evidence and sign off.

## Progress Checklist

- [ ] Stage 1 — Stage 04 evidence reread.
  - Covers: Step 1.
  - Verify: Stage 04 evidence names commit `2131985` and full-suite result
    `889 passed, 217 deselected`.
  - Evidence: record in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 — origin module contract proven.
  - Covers: Steps 2-3.
  - Files: origin module and `tests/test_consolidation_origin_metadata.py`.
  - Verify: focused origin tests pass after the expected red import failure.
  - Evidence: red/green results recorded.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 3 — consolidator state and graph seeding complete.
  - Covers: Steps 4-5.
  - Files: consolidator schema and graph wrapper.
  - Verify: both files compile.
  - Evidence: compile output recorded.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 4 — direct fixtures and integration assertions complete.
  - Covers: Steps 6-7.
  - Files: origin metadata tests and consolidator efficiency tests.
  - Verify: focused integration tests pass.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 5 — full verification complete.
  - Covers: Step 8.
  - Verify: every command in `Verification` passes or returns an explicitly
    allowed no-match `rg` exit code.
  - Evidence: all command results recorded.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 6 — lifecycle records flipped.
  - Covers: Steps 9-10.
  - Files: parent ledger, registry, and this plan.
  - Verify: Stage 05 rows are `completed`.
  - Evidence: row text and commit recorded.
  - Handoff: Stage 06 may be reviewed against this execution evidence.
  - Sign-off: `<agent/date>` after verification.

## Verification

Run from repository root with the project virtual environment.

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_schema.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py tests\test_consolidation_origin_metadata.py tests\test_consolidator_efficiency.py`

### Static Greps

- `rg -n "consolidation_origin|origin_metadata|trigger_source|input_sources|cognitive_episode" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_reflection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_images.py`

  Expected result: no matches. `rg` exit code `1` is acceptable and means this
  check passed. Any match means prompt, persistence, memory-unit, cache, image,
  or scheduler behavior is consuming origin metadata too early.

- `rg -n "consolidation_origin" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_schema.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py`

  Expected result: matches only in the origin module, schema field, import,
  builder call, and initial subgraph state. Matches in return-value
  construction or `metadata` output are blockers.

- `rg -n "\"consolidation_origin\"|\"trigger_source\"|\"input_sources\"" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_reflection.py`

  Expected result: no matches. `rg` exit code `1` is acceptable and means
  prompt payload dicts do not expose origin fields.

- `git diff --check`

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_consolidation_origin_metadata.py`
- `venv\Scripts\python -m pytest tests\test_consolidator_efficiency.py`

### Adjacent Consolidation Tests

- `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py tests\test_consolidator_reflection_prompts.py`
- `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py`

### Prior Stage Regression Gates

- `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_02_chat_episode_migration.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_cognition_clarification_consumers.py tests\test_cognition_interaction_style_context.py`
- `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py tests\test_persona_supervisor2_rag2_integration.py tests\test_persona_supervisor2_rag_skip_shape.py tests\test_rag_projection.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`

### Completion Review

Before merge, inspect:

- `git diff --stat`
- `git diff -- src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py`
- `git diff -- src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_schema.py`
- `git diff -- src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin.py`

The diff must show only origin metadata projection, graph-state seeding, direct
fixture updates, and lifecycle documentation.

## Acceptance Criteria

Stage 05 is complete when:

- `ConsolidationOriginMetadata` and
  `build_user_message_consolidation_origin(...)` exist with the contract above;
- `ConsolidatorState` includes `consolidation_origin`;
- `call_consolidation_subgraph(...)` builds origin metadata before graph
  construction and rejects unsupported origins before nodes run;
- every patched consolidator node receives the state field in the focused
  integration test;
- normal `/chat` user-message consolidation still returns the same output keys;
- `consolidation_origin` is not included in `consolidation_metadata`;
- facts, promises, relationship, mood, character image, persistence,
  scheduler, cache invalidation, and prompt payload behavior are unchanged;
- all Verification commands pass or have explicitly allowed no-match exit
  codes;
- parent ledger and registry rows are completed.

## Plan Self-Review

Performed before approval on 2026-05-10:

- **Coverage:** every `Must Do` maps to an implementation step and a
  verification gate.
- **Placeholder scan:** no unresolved placeholders, open choice, or broad
  "direct tests" surface remains.
- **Contract consistency:** public names, state key, file paths, test names,
  and verification commands match across sections.
- **Granularity:** each execution step has one named target and one expected
  evidence item.
- **Verification:** origin projection, graph seeding, unsupported-origin
  rejection, prompt non-consumption, output shape, prior-stage gates, and
  lifecycle flips each have a focused check.

Approval decision: approved for implementation after Stage 04 merge evidence
was verified on `main`.

## Execution Handoff

Intended execution mode: sequential implementation on a feature branch forked
from current `main`.

Next unchecked stage: Stage 1 in this plan. Required first action: reread Stage
04 `Execution Evidence`, then create the Stage 05 feature branch. The
implementation agent must not start Stage 06 work from this plan.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Origin metadata leaks into prompt payloads | Keep prompt modules in `Keep`; static-grep prompt modules | Static Greps |
| Output state shape changes | Forbid return-value changes and test return keys | Integration output-shape test |
| Unsupported future source writes memory through current policy | Build origin before graph construction and reject unsupported origins | Negative integration test |
| Stage 06 policy lands early | Explicit `Deferred`, `Keep`, and diff review | Completion Review |
| Direct tests bypass Stage 02 episode state | Update only `tests/test_consolidator_efficiency.py` fixture | Focused tests |

## Completion Artifact Contract

When Stage 05 is complete, these artifacts must exist or be updated:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
- `tests/test_consolidation_origin_metadata.py`
- `ConsolidatorState["consolidation_origin"]`
- `call_consolidation_subgraph(...)` seeding `consolidation_origin` before
  graph execution
- parent ledger row for `stage_05` flipped to `completed`
- `development_plans/README.md` Stage 05 row flipped to
  `completed | completed`
- execution evidence in this plan naming branch, commit, static checks, test
  commands, and sign-off

The completion artifact must not include prompt changes, persistence schema
changes, database changes, scheduler changes, or per-write origin policy.

## Execution Evidence

Record after implementation:

- Stage 04 evidence reread:
- Branch:
- Commit:
- Static compile:
- Static greps:
- Focused tests:
- Adjacent consolidation tests:
- Prior stage regression gates:
- Completion diff review:
- Lifecycle records:
- Sign-off:
