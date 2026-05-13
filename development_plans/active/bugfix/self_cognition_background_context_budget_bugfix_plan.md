# self cognition background context budget bugfix plan

## Summary

- Goal: fix the two background context-overflow failure modes while preserving
  self-cognition as an enabled-by-default, memory-producing background loop.
- Plan class: large
- Status: draft
- Mandatory skills: `py-style`, `test-style-and-execution`,
  `local-llm-architecture`, `no-prepost-user-input`
- Overall cutover strategy: bigbang for self-cognition runtime semantics,
  compatible for existing live chat and reflection behavior
- Highest-risk areas: self-cognition memory ownership, consolidation origin
  policy, global-growth prompt budgeting, background LLM call count
- Acceptance criteria: global growth payloads are deterministically bounded;
  self-cognition does not call the visual agent by default; self-cognition does
  not set `no_remember`; live self-cognition can persist memory through an
  explicit bounded memory lane.

## Context

The observed incident had two separate background failures:

1. `global_character_growth.runner` failed in
   `generate_growth_candidates(...)` with `Context size has been exceeded`.
   The current implementation caps card count and per-card text, but it does
   not cap the final rendered prompt against the configured route capacity.

2. `self_cognition.worker` failed inside the shared cognition graph at
   `l3_visual_agent`. Self-cognition currently builds an internal-thought
   episode manually and does not mark visual directives as disabled, so the
   optional visual-agent LLM runs even though self-cognition does not need image
   directives to decide silence, memory, progress, or scheduled contact.

The user clarified two product decisions:

- Self-cognition must stay enabled by default.
- Self-cognition must not set `no_remember`; live self-cognition should be able
  to generate memory.

Removing `no_remember` alone is not sufficient. The current self-cognition
runner invokes cognition and optional dialog rendering, but it does not run
live-chat post-turn consolidation. This plan therefore adds an explicit,
bounded self-cognition memory lane instead of pretending a debug flag change
creates memory writes.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing prompt, graph, cognition,
  consolidation, memory, or background LLM behavior.
- `no-prepost-user-input`: load before changing memory extraction, promise
  persistence, or any path that decides whether user-facing instructions,
  preferences, commitments, or accepted actions become durable state.

## Mandatory Rules

- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Preserve `SELF_COGNITION_ENABLED=true` as the default.
- Do not set `no_remember` in self-cognition state, cognitive episode metadata,
  dry-run artifacts, or worker-generated cases.
- Disable self-cognition visual directives by default with
  `no_visual_directives`; do not disable visual directives globally for live
  user chat.
- Do not send raw self-cognition source packets, raw reflection rows, raw
  memory documents, embeddings, or full artifacts into new prompts.
- Do not add deterministic keyword rules that reinterpret user commitments,
  accepted preferences, or memory channels after an LLM output. If memory
  extraction is wrong, fix the prompt/schema and structural validation.
- Do not add retry loops, model-context increases, alternate LLM routes, or
  fallback prompts as the primary fix for prompt overflow.
- Do not route self-cognition memory through adapter sends, `/chat` synthetic
  user messages, or conversation-history rows.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Add deterministic final-prompt budgeting for global character growth before
  the candidate LLM call.
- Record global-growth prompt-budget drops in run diagnostics.
- Disable the L3 visual-agent call for self-cognition by default.
- Remove the self-cognition `no_remember` debug flag from cognition state and
  cognitive episode metadata.
- Add an explicit live self-cognition memory lane that can write user memory
  units through existing consolidation persistence contracts.
- Keep self-cognition memory writes bounded, auditable, and separate from
  adapter delivery and `/chat` conversation rows.
- Update docs and tests so `SELF_COGNITION_ENABLED` default is `true`.
- Add deterministic regression tests for both failure modes.

## Deferred

- Do not redesign the shared cognition graph.
- Do not add visual directives back to self-cognition in this plan.
- Do not change live user-chat visual-directive behavior.
- Do not change scheduler delivery, dispatcher validation, adapter callbacks,
  or direct send behavior.
- Do not change reflection promotion semantics.
- Do not batch or parallelize global character growth LLM calls.
- Do not tune model server `n_ctx` or route model settings.
- Do not migrate existing memory documents or self-cognition artifacts.

## Cutover Policy

Overall strategy: bigbang for self-cognition behavior; compatible for live chat
and reflection paths.

| Area | Policy | Instruction |
|---|---|---|
| Self-cognition visual directives | bigbang | Add `no_visual_directives` by default for self-cognition. No compatibility path that runs visual LLM in production self-cognition. |
| Self-cognition `no_remember` | bigbang | Remove the flag from self-cognition-created state and episode metadata. Do not replace it with another memory-suppression flag. |
| Self-cognition memory writes | compatible | Add a dedicated self-cognition memory lane without reusing `/chat`, adapter sends, or conversation-history writes. Existing live-chat consolidation remains unchanged. |
| Global growth prompt budgeting | bigbang | Trim candidate input before the LLM call. Do not keep an alternate unbounded path. |
| Config default | bigbang | Keep `SELF_COGNITION_ENABLED=true` and update docs/tests to match. |
| Database | compatible | Use existing user-memory-unit persistence. No data migration or new collection is approved by this plan. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- For bigbang areas, rewrite the old behavior instead of preserving an
  alternate path.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to this policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the contracts in this plan.
- The agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, extra feature flags, or
  unrelated prompt rewrites.
- The agent must treat changes outside the files listed in `Change Surface` as
  out of scope unless the plan is updated first.
- If existing helpers exactly satisfy a needed projection or validation
  contract, reuse them. Do not duplicate prompt-budget, origin-policy, or memory
  projection logic.
- Implement a narrowly scoped self-cognition memory runner that reuses only the
  facts/evaluator/db-writer contracts needed by this plan. Do not call the full
  `call_consolidation_subgraph` because it runs unrelated consolidation LLM
  nodes.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

Global character growth still performs one background candidate-generation LLM
call, but the final rendered system+human prompt is checked against a
deterministic character budget before invocation. When the prompt would exceed
budget, the module keeps the highest-priority memory cards and records how many
cards were dropped for prompt budget.

Self-cognition remains enabled by default. Self-cognition-created cognition
state has no `no_remember` flag and carries `no_visual_directives=true`.
Production self-cognition no longer invokes `l3_visual_agent`.

Live self-cognition can persist durable memory through a dedicated memory lane
after a self-cognition case completes. The memory lane writes only through
existing user-memory-unit persistence and only from bounded, prompt-safe
source/cognition/dialog artifacts. It does not call adapters, does not write
conversation rows, does not run scheduler dispatch a second time, and does not
update character image or reflection state.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Self-cognition default | Keep enabled by default. | User explicitly wants the idle loop active by default. |
| `no_remember` | Remove from self-cognition-created state. | User explicitly wants self-cognition to generate memory. |
| Visual directives | Disable for self-cognition by default. | Visual metadata is not needed for idle agency decisions and caused one overflow path. |
| Self-cognition memory owner | Add a dedicated memory lane under `self_cognition`, backed by existing consolidation persistence where safe. | The current runner does not call live post-turn consolidation, so memory needs an explicit owner. |
| Self-cognition memory write scope | Allow user-memory units and cache invalidation only. | Prevent duplicated dispatch, adapter sends, character-image writes, and unrelated global state drift. |
| Global growth budget | Budget final rendered prompt, not only row count. | The model rejects final context size, not individual card count. |
| Prompt overflow recovery | Deterministic trim/drop with diagnostics. | Local/weaker LLMs need bounded input; retries or context-size increases hide the root cause. |

## Contracts And Data Shapes

### Self-Cognition Debug Modes

Self-cognition-created `debug_modes` must use this shape:

```python
{
    "no_visual_directives": True,
}
```

Forbidden in self-cognition-created state and episode metadata:

```python
{
    "no_remember": True,
}
```

### Self-Cognition Memory Lane

Add a self-cognition memory entrypoint with this conceptual contract:

```python
async def run_self_cognition_memory_lane(
    *,
    case: dict,
    cognition_state: dict,
    cognition_output: dict,
    action_candidate: dict | None,
    dialog_output: dict | None,
    enable_memory_writes: bool,
) -> dict:
    """Return memory-lane result metadata and optionally write memory."""
```

Allowed output metadata:

```python
{
    "memory_lane_called": bool,
    "memory_writes_enabled": bool,
    "new_facts_count": int,
    "future_promises_count": int,
    "user_memory_unit_results": list[dict],
    "cache_invalidated": list[str],
    "status": "not_applicable" | "dry_run" | "written" | "skipped" | "failed",
    "error": str,
}
```

Allowed persistence categories for self-cognition origin:

```python
{
    "user_memory_units": {"allowed": True, "reason": "self_cognition_internal_thought"},
    "cache_invalidation": {"allowed": True, "reason": "self_cognition_internal_thought"},
}
```

Forbidden persistence categories for self-cognition origin:

```text
character_state
relationship_insight
task_dispatch
affinity
character_image
conversation_history
adapter_delivery
```

### Global Growth Prompt Budget

Add a deterministic prompt-budget projection with this conceptual contract:

```python
def build_budgeted_candidate_prompt_payload(
    *,
    memory_rows: Sequence[Mapping[str, Any]],
    current_trait_rows: Sequence[Mapping[str, Any]],
    prompt_char_budget: int,
    limit: int = MAX_MEMORY_CARDS,
) -> tuple[CandidatePromptPayload, dict[str, int | str]]:
    """Return a payload whose final rendered prompt fits the budget."""
```

Diagnostics must include at least:

```python
{
    "prompt_char_budget": int,
    "rendered_prompt_chars": int,
    "memory_cards_before_budget": int,
    "memory_cards_after_budget": int,
    "dropped_memory_cards_for_prompt_budget": int,
}
```

## LLM Call And Context Budget

Use `50k tokens` as the overall context-window assumption and enforce a
conservative character budget because the local route may tokenize CJK text
densely. The implementation must use
`GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET=32000` as the default. The default
must be conservative enough that 80 maximum-size cards cannot enter the growth
LLM unchanged.

| Call | Path | Before | After |
|---|---|---|---|
| Global growth candidate LLM | background reflection/growth | 1 call with up to 80 cards and no final prompt budget | 1 call with budgeted payload and drop diagnostics |
| Self-cognition L3 visual agent | background self-cognition | 1 optional visual LLM call per self-cognition case | 0 visual LLM calls for self-cognition by default |
| Self-cognition memory extraction | background self-cognition | 0 memory extraction calls | Bounded facts/evaluator memory lane calls only when `selected_route` is `action_candidate` or `progress_maintenance` and live writes are enabled |

The self-cognition memory lane must not call an LLM for cases classified as
`silent_no_write`, `audit_only`, or duplicate-suppressed handoffs. It must
record a clear `not_applicable` or `skipped` result for those cases.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/config.py`
  - Preserve `SELF_COGNITION_ENABLED=true`.
  - Add `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET` with a default of
    `32000`.

- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Remove `no_remember` from self-cognition-created state and episode
    metadata.
  - Add `no_visual_directives=true` to self-cognition-created state and episode
    metadata.
  - Thread memory-lane invocation and memory-lane artifacts without changing
    adapter or scheduler behavior.

- `src/kazusa_ai_chatbot/self_cognition/worker.py`
  - Enable live worker memory writes through the explicit memory-lane parameter.
  - Keep dry-run and test seams capable of disabling production writes.
  - Record sanitized memory-lane status in event logs or artifacts.

- `src/kazusa_ai_chatbot/self_cognition/models.py`
  - Add artifact names and status constants for memory-lane output.

- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Update default enablement to `true`.
  - Replace the old "does not update stable memory" boundary with the new
    bounded memory-lane contract.
  - State that visual directives are disabled by default for self-cognition.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
  - Add origin metadata support for self-cognition internal-thought memory
    consolidation through a narrowly scoped self-cognition origin builder.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
  - Allow only `user_memory_units` and `cache_invalidation` for self-cognition
    origin.
  - Keep all existing user-message policy unchanged.

- `src/kazusa_ai_chatbot/global_character_growth/projection.py`
  - Add final-prompt-budget payload construction and card trimming.

- `src/kazusa_ai_chatbot/global_character_growth/runner.py`
  - Use the budgeted payload before calling `generate_growth_candidates`.
  - Persist prompt-budget diagnostics in run documents.

- `src/kazusa_ai_chatbot/global_character_growth/models.py`
  - Add prompt-budget constants and diagnostics fields.

- `src/kazusa_ai_chatbot/global_character_growth/llm.py`
  - Expose a prompt-render sizing helper for tests and budgeting.

- `docs/HOWTO.md`
  - Document `SELF_COGNITION_ENABLED=true`.
  - Document that self-cognition disables visual directives by default and has
    its own memory lane.

- `tests/test_config.py`
  - Update default self-cognition expectation to `True`.

- Existing focused tests under `tests/test_self_cognition_*.py`,
  `tests/test_consolidator_origin_policy_db_writer.py`, and
  `tests/test_global_character_growth_*.py`
  - Add deterministic tests listed in `Implementation Order`.

### Create

- `src/kazusa_ai_chatbot/self_cognition/memory.py`
  - Dedicated home for the memory-lane orchestration.

### Keep

- Live `/chat` service path behavior for user-message consolidation.
- Scheduler and dispatcher validation behavior.
- Adapter delivery behavior.
- Reflection promotion behavior.
- Global character growth trait drift validation.

## Implementation Order

1. Inspect current self-cognition state building, config defaults, and relevant
   docs/tests.
   - Files: `self_cognition/runner.py`, `self_cognition/worker.py`,
     `config.py`, `self_cognition/README.md`, `docs/HOWTO.md`,
     `tests/test_config.py`.
   - Evidence: note all current `no_remember`, `no_visual_directives`, and
     `SELF_COGNITION_ENABLED` expectations.

2. Add failing tests for self-cognition debug modes.
   - Add or update tests proving self-cognition-created cognition state and
     episode metadata contain `no_visual_directives=True` and do not contain
     `no_remember`.
   - Expected before implementation: fails because `no_remember=True` is
     present and visual is not disabled.

3. Implement self-cognition debug-mode fix.
   - Update `runner.py` to remove `no_remember` and add
     `no_visual_directives`.
   - Keep the live user-chat visual config untouched.

4. Add self-cognition memory-lane contract tests.
   - Add deterministic tests proving live worker calls the memory lane with
     writes enabled after a completed action-candidate case.
   - Add deterministic tests proving dry-run or explicit disabled writes record
     memory status without DB writes.
   - Add origin-policy tests proving self-cognition origin allows only
     `user_memory_units` and `cache_invalidation`.

5. Implement the self-cognition memory lane.
   - Add `self_cognition/memory.py`.
   - Build bounded consolidation-style state from self-cognition artifacts.
   - Reuse facts/evaluator/db-writer contracts where safe.
   - Do not run scheduler dispatch from this lane.
   - Do not write conversation rows.

6. Wire the live worker to enable memory writes.
   - Worker production path passes `enable_memory_writes=True`.
   - Dry-run CLI remains no-write and records memory-lane artifacts only.
   - Event logging remains sanitized.

7. Add failing global-growth prompt-budget tests.
   - Add deterministic tests with synthetic maximum-size promoted memory rows.
   - Expected before implementation: rendered prompt exceeds the configured
     prompt budget or no diagnostics are recorded.

8. Implement global-growth prompt budgeting.
   - Build budgeted payload before `generate_growth_candidates`.
   - Persist prompt-budget diagnostics.
   - Keep the single candidate LLM call.

9. Update docs and config tests.
   - `SELF_COGNITION_ENABLED` default is documented and tested as `true`.
   - Self-cognition README describes memory lane and default visual disable.
   - HOWTO environment example matches the active default.

10. Run focused verification.
    - Run all tests listed in `Verification`.
    - Fix only failures inside this plan's change surface.

11. Run static greps.
    - Verify no self-cognition-created state still sets `no_remember`.
    - Verify self-cognition visual disable is present.
    - Verify no raw self-cognition artifacts enter memory prompts.

12. Run independent code review.
    - Follow the `Independent Code Review` gate before sign-off.

## Progress Checklist

- [ ] Stage 1 - self-cognition debug-mode contract fixed
  - Covers: implementation steps 1-3.
  - Verify: focused self-cognition tests for debug-mode state and episode
    metadata pass.
  - Evidence: record failing-before and passing-after test output.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>`.

- [ ] Stage 2 - self-cognition memory lane implemented
  - Covers: implementation steps 4-6.
  - Verify: memory-lane tests, origin-policy tests, and worker handoff tests
    pass.
  - Evidence: record changed files, write-policy behavior, and test output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>`.

- [ ] Stage 3 - global growth prompt budget implemented
  - Covers: implementation steps 7-8.
  - Verify: global-growth prompt-budget tests pass and diagnostics are recorded.
  - Evidence: record prompt-size before/after and test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>`.

- [ ] Stage 4 - docs, config expectations, and regression checks complete
  - Covers: implementation steps 9-11.
  - Verify: docs updated, config default test passes, static greps match
    expectations.
  - Evidence: record doc paths and command output.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>`.

- [ ] Stage 5 - independent code review complete
  - Covers: implementation step 12.
  - Verify: full diff reviewed against this plan and affected tests rerun after
    any review fixes.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, residual
    risks, and approval status.
  - Handoff: plan can be marked completed only after this stage is signed off.
  - Sign-off: `<agent/date>`.

## Verification

### Focused Deterministic Tests

```powershell
venv\Scripts\python -m pytest tests\test_config.py::TestSelfCognitionConfig -q
venv\Scripts\python -m pytest tests\test_self_cognition_framing.py tests\test_self_cognition_integration.py -q
venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py -q
venv\Scripts\python -m pytest tests\test_global_character_growth_runner.py tests\test_global_character_growth_prompt_contracts.py -q
```

Expected: all pass after implementation.

### Broader Regression

```powershell
venv\Scripts\python -m pytest tests\test_global_character_growth_*.py -q
venv\Scripts\python -m pytest tests\test_self_cognition_*.py -q
```

Expected: all deterministic tests pass. If the PowerShell glob does not expand
as expected, rerun using explicit file paths from `rg --files tests`.

### Static Greps

```powershell
rg -n '"no_remember": True|no_remember.*self_cognition|self_cognition.*no_remember' src\kazusa_ai_chatbot\self_cognition tests
```

Expected: no matches that set or expect `no_remember` in self-cognition-created
state. Documentation may mention that self-cognition must not set it.

```powershell
rg -n 'no_visual_directives' src\kazusa_ai_chatbot\self_cognition tests\test_self_cognition_*.py
```

Expected: matches show self-cognition disables visual directives by default and
tests assert that behavior.

```powershell
rg -n 'SELF_COGNITION_ENABLED.*false|default `false`|default false' docs src\kazusa_ai_chatbot tests
```

Expected: no stale self-cognition default-false docs or tests remain. Matches
for unrelated settings must be inspected and recorded.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/context leaks, persistence risk,
  duplicate dispatcher side effects, brittle fixtures, and avoidable blast
  radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including focused and regression tests,
  static-grep accuracy, execution evidence, and lifecycle registry updates.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only fixture/documentation
corrections. If a fix would cross the approved boundary or alter the contract,
stop and update the plan or request approval before changing code.

## Acceptance Criteria

This plan is complete when:

- Global character growth cannot send an over-budget candidate prompt under the
  configured default budget.
- Global character growth run documents include prompt-budget diagnostics.
- Self-cognition-created state and episode metadata do not contain
  `no_remember`.
- Self-cognition-created state and episode metadata contain
  `no_visual_directives=True`.
- Production self-cognition no longer invokes the L3 visual-agent LLM by
  default.
- Live self-cognition can persist memory through the explicit memory lane.
- The memory lane does not write conversation rows, call adapters, or dispatch
  scheduled tasks a second time.
- Docs and tests agree that `SELF_COGNITION_ENABLED` defaults to `true`.
- All listed verification commands pass or have documented, user-approved
  exceptions.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Self-cognition memory writes duplicate scheduler commitments | Forbid `task_dispatch` in self-cognition origin policy | Origin-policy and worker tests |
| Removing `no_remember` is mistaken for memory support | Add explicit memory lane and tests | Memory-lane tests prove write path |
| Memory lane persists semantic mistakes through deterministic filters | Use LLM output contract and structural validation only | `no-prepost-user-input` review and tests |
| Global growth drops useful cards | Record prompt-budget drop diagnostics | Global-growth runner tests inspect diagnostics |
| Visual directives are accidentally disabled for live chat | Scope `no_visual_directives` to self-cognition-created episodes | Live chat visual-control tests remain unchanged |
| New background memory calls increase worker latency | Run only for `action_candidate` and `progress_maintenance`; skip `silent_no_write`, `audit_only`, and duplicate-suppressed cases | Worker tests and LLM budget table |

## Execution Evidence

- Not started.

## Execution Handoff

Execution is not started. The next agent should begin at Stage 1 after the plan
is reviewed and approved.
