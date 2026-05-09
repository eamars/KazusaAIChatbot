# Multi Source Cognition Architecture Stage 05 Consolidation Origin Metadata Threading Plan

Status: draft

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Lifecycle note: this plan is blocked until Stage 04 is completed, merged, and
its execution evidence has been reread. Before approval, replace any Stage 04
test/evidence references below if the executed artifact names differ.

## Summary

Stage 05 threads source-neutral origin metadata from `CognitiveEpisode` into
the consolidation subgraph state without changing what consolidation writes.

The goal is visibility, not policy:

- deterministic consolidation code can see the origin of the current episode;
- current `/chat` user-message behavior remains identical;
- unsupported future origins fail closed before consolidation nodes run;
- facts, promises, relationship, mood, persistence, cache invalidation, and
  scheduler behavior remain unchanged.

This stage prepares Stage 06 to add per-write origin policy. Stage 05 must not
implement that policy.

## Stage Handoff

### From Stage 04

Stage 04 must be completed before this plan can be approved.

Artifacts Stage 05 expects from Stage 04:

- `state["cognitive_episode"]` is still present on normal `/chat` graph state.
- RAG request construction is centralized in
  `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`.
- `stage_1_research` still returns the existing `rag_result` shape.
- RAG internals still do not consume raw `CognitiveEpisode` objects.
- Consolidation behavior was not changed by Stage 04.

Before executing Stage 05, read the Stage 04 execution evidence and confirm:

- Stage 04 focused tests passed.
- Stage 00 through Stage 03 regression gates still passed.
- no prompt, dialog, consolidation, or RAG-internal module consumed raw
  `cognitive_episode` during Stage 04.

### To Stage 06

After Stage 05, Stage 06 can rely on:

- `ConsolidatorState["consolidation_origin"]` exists for current text-chat
  user-message turns.
- the origin metadata contains no raw user text, percept content, attachments,
  or prompt payloads.
- unsupported non-text-chat origins are rejected before consolidation nodes run.
- Stage 05 did not add origin metadata to `consolidation_metadata` output and
  did not change durable memory write policy.

Stage 06 may then design per-write origin policy using
`state["consolidation_origin"]`. That policy is out of scope for Stage 05.

## Mandatory Skills

Before execution, the implementing agent must apply:

- `.agents/skills/development-plan-writing`
- `.agents/skills/local-llm-architecture`
- `.agents/skills/no-prepost-user-input`
- `.agents/skills/py-style`
- `.agents/skills/test-style-and-execution`

If an implementation attempt edits any Python file containing CJK prompt text,
the agent must also apply `.agents/skills/cjk-safety` before editing that file.
This plan should not require prompt edits.

## Mandatory Rules

- Execute only from a feature branch forked from current `main` after Stage 04
  has been merged.
- Keep edits inside the approved Change Surface.
- Do not change consolidator prompts or prompt payloads.
- Do not add origin metadata to LLM messages in Stage 05.
- Do not change facts, promises, relationship, mood, persistence, scheduler,
  cache invalidation, or write-policy behavior.
- Do not add deterministic semantic interpretation of user input.
- Do not filter, reclassify, or rewrite LLM-emitted facts or promises.
- Do not add feature flags, fallback origin builders, or alternate
  consolidation paths.
- Do not add private raising-only helpers or pass-through wrappers. Private
  helpers are allowed only for repeated structural validation or local table
  lookup.
- `call_consolidation_subgraph` must plain-index
  `global_state["cognitive_episode"]`. Do not synthesize a missing episode and
  do not use `.get()` for this field.
- Unsupported origin types must raise before the `StateGraph` is built or any
  consolidation node can run.
- The returned result from `call_consolidation_subgraph(...)` must keep the
  existing keys. Do not add `consolidation_origin` to the returned
  `consolidation_metadata`.
- After each major checklist stage, reread this full plan before starting the
  next stage and confirm no stale assumption has appeared.

## Must Do

- Add a small origin-projection module for consolidation.
- Add a top-level `consolidation_origin` field to `ConsolidatorState`.
- Build `consolidation_origin` once at the start of
  `call_consolidation_subgraph(...)`.
- Add `consolidation_origin` to the initial subgraph state.
- Prove every existing consolidation node receives that state field without
  changing its prompt payload or write output.
- Prove unsupported episode origins fail closed before any node runs.
- Prove normal `/chat` consolidation output shape remains unchanged.

## Deferred

- Per-write policy for non-user-message origins.
- Allowing reflection, internal thought, scheduled recall, system probe, image,
  audio, retrieved memory, or reflection artifact origins to write memory.
- Prompt changes that ask the LLM to reason about origin.
- Persistence schema changes.
- New durable metadata fields in MongoDB.
- Scheduler or autonomous-contact changes.

## Cutover Policy

This is a direct structural cutover inside `call_consolidation_subgraph`.

There is no dual path and no runtime flag. Once implemented, normal `/chat`
consolidation requires a valid `CognitiveEpisode` and carries
`consolidation_origin` in the subgraph state.

## Agent Autonomy Boundaries

Allowed implementation choices:

- local variable names inside tests
- focused assertion grouping
- exact test helper names

Not allowed:

- changing prompt text to mention origin metadata
- adding origin fields to LLM JSON payloads
- adding origin fields to persistence records
- adding policy decisions based on origin
- changing fact or promise harvesting semantics
- changing the result dictionary returned by `call_consolidation_subgraph`
- broad style rewrites of consolidator modules

## Target State

The consolidator receives one additional top-level state value:

```python
sub_state: ConsolidatorState = {
    ...
    "metadata": {},
    "consolidation_origin": consolidation_origin,
}
```

`consolidation_origin` is available to deterministic nodes for future policy
work, but Stage 05 nodes must not consume it for behavior changes.

The final return shape stays:

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

Do not add `consolidation_origin` or origin fields to this return value.

## Origin Contract

Create:

`src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`

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
4. If `episode["output_mode"]` is not one of `"visible_reply"`,
   `"think_only"`, or `"silent"`, raise `ConsolidationOriginError`.
5. Return the exact metadata shape above.

The function must not include raw percept content, raw user text,
`decontexualized_input`, prompt payloads, attachments, `rag_result`, facts, or
promises in the origin metadata.

Do not add a private helper whose only job is to raise the exception. Direct
`raise ConsolidationOriginError(...)` statements are preferred here.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
- `tests/test_consolidation_origin_metadata.py`

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
- `tests/test_consolidator_efficiency.py`
- Direct consolidation tests that call `call_consolidation_subgraph(...)` and
  therefore need a valid `cognitive_episode` fixture.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
- `src/kazusa_ai_chatbot/rag/*.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_*.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_dialog.py`
- `tests/test_consolidator_facts_rag2.py`
- `tests/test_consolidator_reflection_prompts.py`

Do not modify kept files unless a verification failure proves this plan is
incomplete. If that happens, stop and update this plan before continuing.

## Completion Artifact Contract

When Stage 05 is complete, these artifacts must exist or be updated:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
- `tests/test_consolidation_origin_metadata.py`
- `ConsolidatorState["consolidation_origin"]`
- `call_consolidation_subgraph(...)` seeding `consolidation_origin` before
  graph execution
- execution evidence in this plan naming the branch, commit, static checks, and
  test commands

The completion artifact must not include prompt changes, persistence changes,
database schema changes, scheduler changes, or per-write origin policy.

## Implementation Order

1. Reread Stage 04 execution evidence and confirm this Stage 05 plan still
   names the correct artifacts.
2. Create the consolidation origin module and focused unit tests.
3. Add `consolidation_origin` to `ConsolidatorState`.
4. In `call_consolidation_subgraph`, build `consolidation_origin` before
   creating the `StateGraph`.
5. Add `consolidation_origin` to the initial subgraph state.
6. Update direct `call_consolidation_subgraph(...)` test fixtures with valid
   text-chat `CognitiveEpisode` values.
7. Add an integration test with patched consolidator nodes proving the state
   field is present and the returned output shape is unchanged.
8. Add a negative integration test proving unsupported origins fail before any
   consolidator node runs.
9. Run the full Verification section.
10. Record execution evidence only after verification completes.

## Progress Checklist

- [ ] Stage 04 execution evidence reread and this plan updated if necessary.
- [ ] Feature branch created from post-Stage-04 `main`.
- [ ] Origin module created.
- [ ] Origin unit tests written and passing.
- [ ] `ConsolidatorState` includes `consolidation_origin`.
- [ ] `call_consolidation_subgraph` seeds `consolidation_origin`.
- [ ] Direct consolidation fixtures include valid `cognitive_episode` values.
- [ ] Normal `/chat` output shape is unchanged.
- [ ] Unsupported origin negative test passes.
- [ ] Prompt/persistence non-consumption static checks pass.
- [ ] Prior stage regression gates pass.
- [ ] Execution evidence recorded.

## Verification

Run from repository root with the project virtual environment.

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_schema.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py tests\test_consolidation_origin_metadata.py tests\test_consolidator_efficiency.py`

### Static Greps

- `rg -n "consolidation_origin|origin_metadata|trigger_source|input_sources|cognitive_episode" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_reflection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py`

  Expected result: no matches. `rg` exit code `1` is acceptable and means this
  check passed. Any match means prompt, reflection, or persistence behavior is
  consuming origin metadata too early.

- `rg -n "consolidation_origin" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_schema.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py`

  Expected result: matches only in the new origin module, schema field, import,
  builder call, and initial subgraph state. Matches in return-value construction
  or `metadata` output are blockers.

- `git diff --check`

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_consolidation_origin_metadata.py`
- `venv\Scripts\python -m pytest tests\test_consolidator_efficiency.py`

### Adjacent Consolidation Tests

- `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py`
- `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py tests\test_consolidator_reflection_prompts.py`

### Prior Stage Regression Gates

- `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_02_chat_episode_migration.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_cognition_clarification_consumers.py tests\test_cognition_interaction_style_context.py`
- `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py tests\test_persona_supervisor2_rag2_integration.py tests\test_persona_supervisor2_rag_skip_shape.py tests\test_rag_projection.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`

### Completion Review

Before approval to merge, inspect:

- `git diff --stat`
- `git diff -- src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py`
- `git diff -- src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_schema.py`
- `git diff -- src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin.py`

The diff must show only origin metadata threading and tests inside the approved
surface.

## Acceptance Criteria

Stage 05 is complete when:

- `ConsolidationOriginMetadata` and
  `build_user_message_consolidation_origin(...)` exist with the contract above;
- `ConsolidatorState` includes `consolidation_origin`;
- `call_consolidation_subgraph(...)` rejects unsupported origins before nodes
  run;
- normal `/chat` user-message consolidation still returns the same output keys;
- `consolidation_origin` is not included in `consolidation_metadata`;
- facts, promises, relationship, mood, persistence, scheduler, cache
  invalidation, and prompt payload behavior are unchanged;
- all Verification commands pass or have the explicitly allowed no-match exit
  code.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Origin metadata leaks into prompt payloads | Keep prompt modules unchanged and static-grep for origin fields | Static prompt non-consumption grep |
| Output state shape changes | Forbid adding origin to return metadata | Integration output-shape test |
| Unsupported future source writes memory through current policy | Fail closed before graph construction | Negative integration test |
| Agent implements Stage 06 policy too early | Explicit deferred scope and no prompt/persistence changes | Diff review and static grep |
| Direct tests bypass Stage 02 episode state | Update direct consolidator fixtures only where needed | Focused and adjacent tests |

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
- Sign-off:
