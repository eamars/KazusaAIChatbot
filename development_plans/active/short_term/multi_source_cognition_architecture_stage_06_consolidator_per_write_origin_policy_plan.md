# multi source cognition architecture stage 06 consolidator per write origin policy plan

## Summary

- Goal: Add deterministic per-write origin policy to the consolidator so each
  durable write path has an explicit allow/deny decision before non-chat
  cognition sources are enabled.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` before editing consolidator Python files that contain CJK prompt
  text.
- Overall cutover strategy: compatible for current `/chat` user-message
  behavior; bigbang for deterministic write-policy checks inside `db_writer`.
- Highest-risk areas: changing current `/chat` writes, adding policy metadata
  to prompt payloads, disabling cache invalidation for allowed user-message
  writes, and enabling reflection/internal-thought writes before dry-run stages.
- Acceptance criteria: a policy module exists; `db_writer` gates each durable
  write category through it; `origin=user_message` behavior is unchanged;
  unsupported origins make no durable writes, no scheduler dispatches, and no
  cache invalidations; all Verification gates pass.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Lifecycle: Stage 05 completed on 2026-05-10 on branch
`stage-05-consolidation-origin-metadata-threading` with implementation commit
`de311a7`, then was merged to `main` with merge commit `75816d1`. Stage 05
evidence records `898 passed, 217 deselected` for the full deterministic suite.
This plan was reviewed against that evidence and is approved for execution from
current `main`.

## Context

Stage 05 threads `consolidation_origin` into `ConsolidatorState` without
changing behavior. Stage 06 uses that state field to make write permission
explicit at the deterministic persistence boundary.

The consolidator currently has these write/effect categories:

- character state: `upsert_character_state(...)`
- relationship insight: `update_last_relationship_insight(...)`
- user memory units: `update_user_memory_units_from_state(...)`
- scheduler dispatch: `_generate_raw_tool_calls(...)` and
  `dispatcher.dispatch(...)`
- affinity: `update_affinity(...)`
- character image: `_update_character_image(...)` and
  `upsert_character_self_image(...)`
- Cache2 invalidation: `runtime.invalidate(...)` events after successful
  writes

This plan does not add new trigger sources to runtime. Reflection and internal
thought remain blocked by Stage 05 before graph execution. Stage 06 adds the
defensive policy layer that later dry-run stages must rely on.

## Stage Handoff

### From Stage 05

Stage 06 expects these completed artifacts:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
- `ConsolidatorState["consolidation_origin"]`
- `call_consolidation_subgraph(...)` seeds `consolidation_origin` before graph
  execution.
- `tests/test_consolidation_origin_metadata.py`
- Stage 05 evidence showing unsupported origins fail before graph nodes run.

Stage 05 evidence reviewed for this approval:

- Branch: `stage-05-consolidation-origin-metadata-threading`
- Implementation commit: `de311a7`
- Merge commit on `main`: `75816d1`
- Focused origin metadata tests: `12 passed`
- Adjacent consolidation tests: `4 passed` and `18 passed`
- Prior-stage regression gates: `15 passed`, `5 passed`, `43 passed`,
  `31 passed`, and `11 passed`
- Full deterministic suite: `898 passed, 217 deselected`

### To Stage 07

After Stage 06, Stage 07 can rely on:

- Every consolidator write category has an explicit deterministic policy
  decision.
- Current `/chat` user-message origin allows all current write categories.
- Non-user-message origins are denied by default and cannot create user facts,
  future promises, relationship changes, character-state writes, scheduler
  events, character-image writes, or cache invalidations through `db_writer`.
- No prompt has received origin metadata.

Stage 07 may then build reflection-trigger dry runs knowing durable writes are
blocked unless a later approved plan changes the policy.

## Mandatory Skills

- `development-plan-writing`: preserve staged lifecycle and handoff.
- `local-llm-architecture`: keep policy deterministic and keep origin metadata
  out of prompt payloads.
- `no-prepost-user-input`: do not add code-side semantic filtering over facts,
  promises, or user instructions.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`,
  or `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`.

## Mandatory Rules

- Execute only from a feature branch forked from post-Stage-05 `main`.
- Keep edits inside the approved Change Surface.
- Use PowerShell `-LiteralPath '...'` for filesystem paths that may contain
  spaces; prefer repo-relative paths in commands.
- Do not change any consolidator prompt text or prompt payload shape.
- Do not add origin metadata to LLM messages.
- Do not add deterministic semantic interpretation of user input, facts,
  promises, memory-unit candidates, or tool-call text.
- Gate whole write/effect categories only. Do not drop, reclassify, rewrite,
  or selectively filter individual LLM-emitted facts, promises, memory-unit
  candidates, accepted preferences, or tool-call text for an allowed origin.
- Do not change the Stage 05 origin builder or broaden accepted runtime
  origins.
- Do not enable reflection, internal thought, scheduled recall, system probe,
  image, audio, retrieved memory, or reflection artifact origins to write.
- Do not add feature flags, fallback policy builders, compatibility shims,
  alternate `db_writer` paths, private raising-only helpers, or pass-through
  wrappers.
- Current `origin=user_message` behavior must stay observationally unchanged.
- Disabled origins must produce no durable writes, no task dispatch, and no
  Cache2 invalidation.
- Disabled origins must not call `get_rag_cache2_runtime()`,
  `runtime.invalidate(...)`, `_get_task_dispatcher()`,
  `_generate_raw_tool_calls(...)`, or `dispatcher.dispatch(...)`.
- `db_writer(...)` must plain-index `state["consolidation_origin"]` when
  building policy. Do not synthesize a missing origin and do not use `.get()`
  for this field.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.

## Must Do

- Create the write-policy module specified in `Policy Contract`.
- Gate the seven write/effect categories in `db_writer` through the policy:
  character state, relationship insight, user memory units, scheduler dispatch,
  affinity, character image, and Cache2 invalidation.
- Preserve all current `/chat` user-message writes and metadata for allowed
  origins.
- Add direct policy tests for allowed user-message origin and denied future
  origins.
- Add `db_writer` tests proving denied origins call no persistence, scheduler,
  image, memory-unit, or cache functions.
- Add current user-message regression tests proving existing calls still happen.
- Update every existing direct `db_writer(...)` test fixture so it includes a
  valid Stage 05 user-message `consolidation_origin`.
- Run every Verification command and record evidence.
- Flip lifecycle rows to `completed` only after verification passes.

## Deferred

- Enabling reflection-origin durable writes.
- Enabling internal-thought durable writes.
- Adding scheduler/proactive output policy.
- Adding origin metadata to prompt payloads or durable MongoDB records.
- Changing fact harvesting, promise harvesting, memory-unit extraction,
  relationship recording, character image synthesis, scheduler dispatch prompt,
  or cache invalidation semantics.
- Stage 07 reflection-trigger dry run.

## Cutover Policy

Overall strategy: compatible for current `/chat`, bigbang for `db_writer`
policy checks.

| Area | Policy | Instruction |
|---|---|---|
| Policy module | bigbang | Add one policy module and make `db_writer` use it directly. No fallback. |
| User-message writes | compatible | Existing writes, metadata keys, and cache invalidation behavior remain unchanged. |
| Unsupported origins | bigbang | Deny all write/effect categories in `db_writer`. No partial writes. |
| Runtime origin acceptance | compatible | Do not broaden Stage 05 runtime origin acceptance. |

Rollback path: remove the policy module, remove policy checks from
`db_writer`, and revert Stage 06 tests. No database migration or cleanup is
required because disabled origins must not write.

## Agent Autonomy Boundaries

Allowed implementation choices:

- local variable names inside the new policy module and tests;
- focused assertion ordering;
- additional tiny private helpers only for local table lookup when used by at
  least three write categories.

Not allowed:

- editing prompt constants;
- introducing a policy registry, feature flag, config file, dependency
  injection layer, or alternate writer;
- adding raising-only helpers, pass-through wrappers, aliases, or adapter
  layers around `build_consolidation_write_policy(...)`;
- creating modules, tests, or fixtures not listed in Change Surface;
- adding origin fields to LLM payloads or durable metadata;
- changing behavior outside `db_writer` and its direct tests;
- broad style rewrites in CJK prompt files.

## Target State

`db_writer(state)` computes a policy once:

```python
origin_policy = build_consolidation_write_policy(
    origin=state["consolidation_origin"],
)
```

Each write/effect category checks its named policy decision before running. For
current user-message origins, every current category is allowed. For all other
origins, every category is denied.

No prompt-facing function receives `origin_policy` or `consolidation_origin`.

## Policy Contract

Create:

`src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`

Public names:

```python
from typing import Literal, TypedDict

from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_origin import (
    ConsolidationOriginMetadata,
)

WritePolicyKey = Literal[
    "character_state",
    "relationship_insight",
    "user_memory_units",
    "task_dispatch",
    "affinity",
    "character_image",
    "cache_invalidation",
]


class WritePolicyDecision(TypedDict):
    allowed: bool
    reason: str


class ConsolidationWritePolicy(TypedDict):
    character_state: WritePolicyDecision
    relationship_insight: WritePolicyDecision
    user_memory_units: WritePolicyDecision
    task_dispatch: WritePolicyDecision
    affinity: WritePolicyDecision
    character_image: WritePolicyDecision
    cache_invalidation: WritePolicyDecision
```

Public function:

```python
def build_consolidation_write_policy(
    *,
    origin: ConsolidationOriginMetadata,
) -> ConsolidationWritePolicy:
    ...
```

Policy rules:

- If `origin["trigger_source"] == "user_message"`,
  `origin["input_sources"] == ["dialog_text"]`, and
  `origin["output_mode"] in {"visible_reply", "think_only", "silent"}`, every
  decision is `{"allowed": True, "reason": "user_message_dialog_text"}`.
- For every other origin, every decision is
  `{"allowed": False, "reason": "origin_not_enabled"}`.
- The function must not inspect raw percept content, prompt payloads,
  `rag_result`, facts, promises, dialog text, or user memory text.
- The function must not raise for future origin labels already accepted by the
  `ConsolidationOriginMetadata` type; it must return denied decisions.
- Use direct dictionary indexing for the listed origin fields. Do not use
  `.get()` defaults or catch exceptions inside the policy module.

## `db_writer` Integration Contract

Modify:
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`

Integration rules:

- Import only `build_consolidation_write_policy`.
- Compute `origin_policy` once at the top of `db_writer(...)` after `metadata`
  and `write_log` are initialized, using
  `state["consolidation_origin"]`.
- For user-message origins, preserve existing control flow and metadata.
- For denied origins:
  - skip `upsert_character_state(...)`;
  - skip `update_last_relationship_insight(...)`;
  - skip `update_user_memory_units_from_state(...)`;
  - skip `_generate_raw_tool_calls(...)` and `dispatcher.dispatch(...)`;
  - skip `_get_task_dispatcher()` and `_build_dispatch_context(...)`;
  - skip `update_affinity(...)`;
  - skip `_update_character_image(...)` and `upsert_character_self_image(...)`;
  - skip `get_rag_cache2_runtime()` and all `runtime.invalidate(...)` calls.
- For denied origins, set a `write_log` key to `False` only for a category that
  would otherwise have attempted a write under current logic. Do not add new
  `write_success` keys for user-message origins.
- Do not add `origin_policy` or `consolidation_origin` to prompt payloads.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Policy location | New module beside consolidator origin code. | The policy is deterministic consolidator behavior, not prompt or RAG behavior. |
| Default future origins | Deny all write/effect categories. | Later stages must explicitly approve non-chat writes. |
| User-message behavior | Allow all current categories. | Stage 06 must preserve `/chat` behavior. |
| Prompt exposure | No prompt receives origin policy. | The local LLM should not infer write permissions. |
| Metadata | Preserve user-message metadata shape. | Avoid `/chat` regression and leave audit expansion to later plans. |

## Change Surface

### Create

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
- `tests/test_consolidation_origin_policy.py`
- `tests/test_consolidator_origin_policy_db_writer.py`

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
- `tests/test_consolidator_efficiency.py` — add a valid user-message
  `consolidation_origin` fixture to the existing direct `db_writer(...)` test.
- `tests/test_db_writer_cache2_invalidation.py` — add a valid user-message
  `consolidation_origin` fixture to existing direct `db_writer(...)` Cache2
  invalidation tests.
- `development_plans/active/short_term/multi_source_cognition_architecture_stage_06_consolidator_per_write_origin_policy_plan.md`
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
- `development_plans/README.md`

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
- `src/kazusa_ai_chatbot/rag/*.py`
- cognition and dialog modules

Do not modify kept files unless a verification failure proves this plan is
incomplete. If that happens, stop and update this plan before continuing.

## Implementation Order

1. Reread Stage 05 `Execution Evidence`.
   - Expected: Stage 05 completed and parent ledger row is `completed`.
   - Evidence: record Stage 05 commit and verification result.
2. Add policy unit tests.
   - File: `tests/test_consolidation_origin_policy.py`
   - Tests:
     `test_user_message_dialog_text_allows_all_write_categories`,
     `test_reflection_signal_origin_denies_all_write_categories`,
     `test_internal_thought_origin_denies_all_write_categories`,
     `test_non_dialog_text_input_denies_all_write_categories`,
     `test_preview_output_denies_all_write_categories`.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_consolidation_origin_policy.py -q`
   - Expected before implementation: import error because the policy module
     does not exist.
3. Create the policy module.
   - File:
     `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
   - Verify:
     `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin_policy.py`
     and rerun the Step 2 tests.
4. Add denied-origin `db_writer` tests.
   - File: `tests/test_consolidator_origin_policy_db_writer.py`
   - Test:
     `test_db_writer_denied_origin_skips_all_durable_write_effects`.
   - Patch all persistence, memory-unit, scheduler, image, and cache functions
     touched by `db_writer` to fail if called.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py -q`
   - Expected before integration: the test fails because current `db_writer`
     has no policy gate.
5. Integrate policy in `db_writer`.
   - File:
     `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
   - Verify:
     `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py`
     and rerun Step 4 tests.
6. Update existing direct `db_writer(...)` fixtures.
   - Files: `tests/test_consolidator_efficiency.py` and
     `tests/test_db_writer_cache2_invalidation.py`.
   - Action: add a valid user-message `consolidation_origin` to every state
     passed directly into `db_writer(...)`.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_consolidator_efficiency.py::test_db_writer_runs_image_updaters_through_gather tests\test_db_writer_cache2_invalidation.py -q`
7. Add user-message regression tests for `db_writer`.
   - File: `tests/test_consolidator_origin_policy_db_writer.py`
   - Tests:
     `test_db_writer_user_message_origin_preserves_character_and_user_writes`,
     `test_db_writer_user_message_origin_preserves_scheduler_dispatch`,
     `test_db_writer_user_message_origin_preserves_cache_invalidation`.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py -q`
8. Run focused and adjacent consolidation tests.
9. Run the full Verification section.
10. Flip lifecycle rows only after verification passes.
11. Record execution evidence and sign off.

## Progress Checklist

- [x] Stage 1 — Stage 05 evidence reread.
  - Covers: Step 1.
  - Verify: Stage 05 row is `completed`.
  - Evidence: Stage 05 commit and verification result recorded.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex / 2026-05-10` after evidence was recorded.
- [x] Stage 2 — policy contract implemented.
  - Covers: Steps 2-3.
  - Verify: policy tests pass after expected red import failure.
  - Evidence: red/green results recorded.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `Codex / 2026-05-10` after the red import failure, compile,
    and focused policy tests passed.
- [x] Stage 3 — denied-origin writer behavior implemented.
  - Covers: Steps 4-5.
  - Verify: denied-origin `db_writer` test passes.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `Codex / 2026-05-10` after expected red writer failure,
    persistence compile, and denied-origin writer test passed.
- [x] Stage 4 — user-message regression behavior proven.
  - Covers: Steps 6-7.
  - Verify: direct fixture updates and user-message `db_writer` regression
    tests pass.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `Codex / 2026-05-10` after direct fixture and user-message
    writer regression tests passed.
- [x] Stage 5 — full verification complete.
  - Covers: Steps 8-9.
  - Verify: every command in `Verification` passes or returns an explicitly
    allowed no-match `rg` exit code.
  - Evidence: command output recorded.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `Codex / 2026-05-10` after static checks, focused tests,
    adjacent tests, prior-stage gates, completion diff review, and full
    deterministic suite passed.
- [x] Stage 6 — lifecycle records flipped.
  - Covers: Steps 10-11.
  - Verify: parent ledger and registry rows show Stage 06 completed.
  - Evidence: row text and commit recorded.
  - Handoff: Stage 07 may be reviewed against this execution evidence.
  - Sign-off: `Codex / 2026-05-10` after parent ledger and registry rows
    were flipped to completed.

## Verification

Run from repository root with the project virtual environment.

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin_policy.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py tests\test_consolidation_origin_policy.py tests\test_consolidator_origin_policy_db_writer.py tests\test_consolidator_efficiency.py tests\test_db_writer_cache2_invalidation.py`

### Static Greps

- `rg -n "consolidation_origin|origin_policy|trigger_source|input_sources" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_reflection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_images.py`

  Expected result: no matches. `rg` exit code `1` is acceptable and means
  prompt, memory-unit, and image helpers do not consume origin policy.

- `rg -n "origin_policy|consolidation_origin" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py`

  Expected result: matches only where `db_writer` imports/builds the policy and
  checks named decisions. Matches inside `_TASK_DISPATCHER_PROMPT`,
  `_generate_raw_tool_calls(...)` payload construction, or any LLM prompt
  payload dict are blockers.

- `rg -n "reflection_signal|internal_thought|scheduled_recall|system_probe|image_observation|audio_observation|reflection_artifact|retrieved_memory" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin_policy.py`

  Expected result: no matches. `rg` exit code `1` is acceptable. Runtime code
  must implement default-deny by checking the allowed user-message contract,
  not by enumerating future source labels.

- `git diff --check`

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_consolidation_origin_policy.py`
- `venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py`

### Adjacent Consolidation Tests

- `venv\Scripts\python -m pytest tests\test_consolidation_origin_metadata.py tests\test_consolidator_efficiency.py tests\test_db_writer_cache2_invalidation.py`
- `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py tests\test_consolidator_reflection_prompts.py`
- `venv\Scripts\python -m pytest tests\test_user_memory_units_rag_flow.py`
- `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py`
- `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py`
- `venv\Scripts\python -m pytest tests\test_save_conversation_invalidation.py`

### Prior Stage Regression Gates

- `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_02_chat_episode_migration.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_cognition_clarification_consumers.py tests\test_cognition_interaction_style_context.py`
- `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py tests\test_persona_supervisor2_rag2_integration.py tests\test_persona_supervisor2_rag_skip_shape.py tests\test_rag_projection.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`

### Completion Review

Before merge, inspect:

- `git diff --stat`
- `git diff -- src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py`
- `git diff -- src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin_policy.py`

The diff must show only deterministic write-policy gating and focused tests.

## Acceptance Criteria

Stage 06 is complete when:

- `build_consolidation_write_policy(...)` exists with the contract above;
- user-message dialog-text origins allow all current write/effect categories;
- future/non-dialog/unsupported origins deny every write/effect category;
- `db_writer` gates character state, relationship insight, user memory units,
  scheduler dispatch, affinity, character image, and Cache2 invalidation;
- current `/chat` user-message writer behavior is unchanged;
- denied origins produce no durable writes, no scheduler dispatch, and no cache
  invalidation;
- origin policy is not exposed to prompts or durable metadata;
- all Verification commands pass or have explicitly allowed no-match exit
  codes;
- parent ledger and registry rows are completed.

## Plan Self-Review

Performed during approval review on 2026-05-10:

- **Coverage:** every Stage 06 parent-plan scope item maps to a named
  implementation step and verification gate.
- **Placeholder scan:** no unresolved implementation choices are left in the
  executable sections; Stage 05 evidence has been carried forward from the
  completed Stage 05 plan.
- **Contract consistency:** policy keys match the seven write/effect
  categories named in Context, Must Do, Verification, and Acceptance Criteria.
- **Granularity:** steps split policy tests, policy module, denied-origin
  writer integration, direct writer fixtures, user-message regression, and
  lifecycle updates.
- **Verification:** prompt non-consumption, policy decisions, writer gating,
  current user-message behavior, adjacent consolidation, scheduler, cache, and
  prior-stage gates are covered.

Approval decision: approved for implementation after Stage 05 merge evidence
was verified on `main`.

## Execution Handoff

Intended execution mode: sequential implementation on a feature branch forked
from current `main`.

Next action: fork the Stage 06 branch, reread this plan, and start at the first
unchecked Progress Checklist stage.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| User-message writes regress | Allow all current categories and add writer regression tests | Focused writer tests and adjacent consolidation tests |
| Disabled origins still write through one path | Gate every write/effect category and patch dependencies to fail if called | Denied-origin writer test |
| Origin policy leaks into prompts | Keep prompt modules in `Keep` and static-grep payload files | Static Greps |
| Scheduler dispatch remains enabled for future origins | Gate raw tool-call generation and dispatcher dispatch | Focused writer test |
| Cache invalidation runs after denied origin | Gate invalidation event execution | Focused writer test |

## Completion Artifact Contract

When Stage 06 is complete, these artifacts must exist or be updated:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
- `tests/test_consolidation_origin_policy.py`
- `tests/test_consolidator_origin_policy_db_writer.py`
- fixture updates in `tests/test_consolidator_efficiency.py` and
  `tests/test_db_writer_cache2_invalidation.py`
- `db_writer(...)` gates the seven write/effect categories through the policy.
- parent ledger row for `stage_06` flipped to `completed`
- `development_plans/README.md` Stage 06 row flipped to
  `completed | completed`
- execution evidence in this plan naming branch, commit, static checks, test
  commands, and sign-off

The completion artifact must not include prompt changes, new runtime trigger
sources, persistence schema changes, scheduler policy changes, or proactive
output behavior.

## Execution Evidence

Record after implementation:

- Stage 05 evidence reread: Stage 05 `Execution Evidence` was reread on
  2026-05-10. It records branch
  `stage-05-consolidation-origin-metadata-threading`, implementation commit
  `de311a7`, merge commit `75816d1`, and full deterministic suite result
  `898 passed, 217 deselected`.
- Branch: `stage-06-consolidator-per-write-origin-policy`
- Commit:
- Static compile:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin_policy.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py tests\test_consolidation_origin_policy.py tests\test_consolidator_origin_policy_db_writer.py tests\test_consolidator_efficiency.py tests\test_db_writer_cache2_invalidation.py`
  passed.
- Static greps:
  prompt/helper non-consumption grep returned no matches with exit code `1`.
  Persistence grep returned only the policy import, policy build from
  `state["consolidation_origin"]`, and named policy decision checks.
  Future-source label grep returned no matches with exit code `1`.
  `git diff --check` passed with line-ending warnings only.
- Focused tests:
  `venv\Scripts\python -m pytest tests\test_consolidation_origin_policy.py -q`
  first failed as expected with `ModuleNotFoundError:
  kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_origin_policy`.
  After implementing the policy module,
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin_policy.py`
  passed and
  `venv\Scripts\python -m pytest tests\test_consolidation_origin_policy.py -q`
  passed with `5 passed`.
  `venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py -q`
  first failed as expected at `upsert_character_state(...)` because
  `db_writer` had no denied-origin gate. After integrating the policy,
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py`
  passed and
  `venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py -q`
  passed with `1 passed`.
- Adjacent consolidation tests:
  `venv\Scripts\python -m pytest tests\test_consolidator_efficiency.py::test_db_writer_runs_image_updaters_through_gather tests\test_db_writer_cache2_invalidation.py -q`
  passed with `3 passed`.
  `venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py -q`
  passed with `4 passed`.
  `venv\Scripts\python -m pytest tests\test_consolidation_origin_metadata.py tests\test_consolidator_efficiency.py tests\test_db_writer_cache2_invalidation.py`
  passed with `14 passed`.
  `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py tests\test_consolidator_reflection_prompts.py`
  passed with `4 passed`.
  `venv\Scripts\python -m pytest tests\test_user_memory_units_rag_flow.py`
  passed with `13 passed`.
  `venv\Scripts\python -m pytest tests\test_service_background_consolidation.py`
  passed with `18 passed`.
  `venv\Scripts\python -m pytest tests\test_scheduler_future_promise.py`
  passed with `4 passed`.
  `venv\Scripts\python -m pytest tests\test_save_conversation_invalidation.py`
  passed with `1 passed`.
- Prior stage regression gates:
  `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py`
  passed with `15 passed`.
  `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_02_chat_episode_migration.py`
  passed with `5 passed`.
  `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_cognition_clarification_consumers.py tests\test_cognition_interaction_style_context.py`
  passed with `43 passed`.
  `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py tests\test_persona_supervisor2_rag2_integration.py tests\test_persona_supervisor2_rag_skip_shape.py tests\test_rag_projection.py`
  passed with `31 passed`.
  `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`
  passed with `11 passed`.
  Full deterministic suite `venv\Scripts\python -m pytest` passed with
  `907 passed, 217 deselected`.
- Completion diff review:
  `git diff --stat`, persistence diff, and origin-policy diff were reviewed.
  The diff is limited to deterministic policy gating, the policy module,
  focused tests, direct writer fixture updates, and lifecycle evidence.
- Lifecycle records:
  Parent ledger row:
  `| stage_06 | multi_source_cognition_architecture_stage_06_consolidator_per_write_origin_policy_plan.md | completed | per-write origin policy tests | stage_07+ |`.
  Registry row:
  `multi_source_cognition_architecture_stage_06_consolidator_per_write_origin_policy_plan.md | completed | completed`.
- Sign-off:
  `Codex / 2026-05-10`
