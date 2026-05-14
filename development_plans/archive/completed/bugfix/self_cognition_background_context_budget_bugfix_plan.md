# self cognition background context budget bugfix plan

## Summary

- Goal: fix the observed background context-overflow failures and align
  self-cognition defaults without designing self-cognition memory persistence.
- Plan class: large
- Status: completed
- Mandatory skills: `py-style`, `test-style-and-execution`,
  `local-llm-architecture`
- Overall cutover strategy: bigbang for the two failure-mode fixes; compatible
  for existing live chat, reflection, consolidation, and future memory work
- Highest-risk areas: background LLM prompt budgets, self-cognition debug-mode
  contract, accidental visual-agent invocation, stale default documentation
- Acceptance criteria: global growth candidate prompts are deterministically
  bounded; self-cognition does not call the visual agent by default;
  self-cognition is enabled by default in config/docs/tests;
  self-cognition-created state does not set `no_remember`; this bugfix does
  not add a memory writer or change consolidator origin policy.

## Context

The incident had two independent background failures:

1. `global_character_growth.runner` failed in
   `generate_growth_candidates(...)` with `Context size has been exceeded`.
   The current implementation caps card count and per-card text, but it does
   not cap the final rendered system+human prompt sent to the candidate LLM.

2. `self_cognition.worker` failed inside the shared cognition graph at
   `l3_visual_agent`. The shared L3 visual node already skips itself when
   `origin_metadata.debug_modes.no_visual_directives` is true, but
   self-cognition-created episodes do not set that flag. Therefore the
   optional visual LLM can run for background self-cognition even though idle
   reasoning does not need image directives.

Confirmed product decisions for this bugfix:

- Self-cognition stays enabled by default.
- Self-cognition is intended to generate memory in a later improvement.
- Therefore self-cognition-created state must not set `no_remember`.
- The full question of what self-cognition should save belongs to
  `development_plans/active/short_term/self_cognition_memory_semantics_plan.md`.

Removing `no_remember` does not by itself create memory writes in the current
self-cognition runner. This bugfix removes the memory-suppression blocker from
self-cognition-created state, but it does not add a memory writer, does not
change the consolidator graph, and does not decide memory semantics.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing prompt, graph, cognition, or
  background LLM behavior.

## Mandatory Rules

- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Preserve `SELF_COGNITION_ENABLED=true` as the default.
- In self-cognition-created cognition state and cognitive episode
  `origin_metadata.debug_modes`, include `no_visual_directives=True`.
- In self-cognition-created cognition state, cognitive episode metadata,
  dry-run artifacts, and worker-generated cases, do not set `no_remember` to
  either `True` or `False`.
- Do not disable visual directives globally for live user chat.
- Do not add self-cognition memory persistence, origin-policy expansion, DB
  writes, adapter sends, scheduler changes, or `/chat` synthetic messages in
  this bugfix.
- Do not call or modify the full live-chat consolidator graph
  `call_consolidation_subgraph` for self-cognition in this bugfix.
- Do not add retry loops, model-context increases, alternate LLM routes, or
  fallback prompts as the primary fix for prompt overflow.
- Keep prompt-budget enforcement deterministic. The candidate LLM must receive
  only an already-budgeted payload.
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
- Record global-growth prompt-budget diagnostics in every non-failed run
  document, including skipped runs.
- Disable the L3 visual-agent call for self-cognition by default.
- Remove the self-cognition `no_remember` debug flag from cognition state and
  cognitive episode metadata.
- Keep live user-chat visual-directive behavior unchanged.
- Update config, docs, and tests so `SELF_COGNITION_ENABLED` defaults to
  `true`.
- Add deterministic regression tests for both observed failure modes.

## Deferred

- Do not design or implement self-cognition memory persistence in this bugfix.
- Do not decide what self-cognition should save.
- Do not change the full live-chat consolidator graph.
- Do not change consolidation origin policy.
- Do not change live user-chat consolidation behavior.
- Do not change scheduler delivery, dispatcher validation, adapter callbacks,
  or direct send behavior.
- Do not change reflection promotion semantics.
- Do not batch or parallelize global character growth LLM calls.
- Do not tune model server `n_ctx` or route model settings.
- Do not migrate existing memory documents or self-cognition artifacts.
- Do not add new source modules.

## Cutover Policy

Overall strategy: bigbang for the two failure-mode fixes; compatible for live
chat, reflection, consolidation, and later memory work.

| Area | Policy | Instruction |
|---|---|---|
| Self-cognition visual directives | bigbang | Add `no_visual_directives=True` by default for self-cognition-created episodes. No production self-cognition compatibility path may run the visual LLM by default. |
| Self-cognition `no_remember` | bigbang | Remove the flag from self-cognition-created state and episode metadata. Do not replace it with another memory-suppression flag. |
| Self-cognition memory writes | compatible | Leave writes unimplemented in this bugfix. The short-term memory-semantics plan owns the future write lane. |
| Global growth prompt budgeting | bigbang | Trim candidate input before the LLM call. Do not keep an alternate unbounded path. |
| Config default | bigbang | Set and document `SELF_COGNITION_ENABLED=true`. |
| Database | compatible | No data migration, new collection, or persistence-policy change is approved by this bugfix. |

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
- The agent may add small private helpers inside listed files only when they
  implement the exact contracts in this plan. Do not create wrapper-only
  helpers or pass-through aliases.
- If existing helpers exactly satisfy a needed prompt-budget projection or
  validation contract, reuse them instead of duplicating logic.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

Global character growth still performs one background candidate-generation LLM
call for eligible runs, but the final rendered system+human prompt is checked
against a deterministic character budget before invocation. When the prompt
would exceed budget, the module drops lowest-priority memory cards from the end
of the already-ranked prompt-card list until the rendered prompt is within
budget or no memory cards remain. Every run document records prompt-budget
diagnostics.

Self-cognition remains enabled by default. Self-cognition-created cognition
state and cognitive episode metadata carry `no_visual_directives=True` and do
not carry `no_remember`. Production self-cognition no longer invokes
`l3_visual_agent` by default.

Self-cognition memory generation remains a product goal, but the actual memory
write contract is not implemented by this bugfix. The follow-up short-term plan
owns the taxonomy, persistence lane, and tests for what should be saved.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Self-cognition default | Keep enabled by default. | The idle loop should run without requiring an opt-in env override. |
| `no_remember` | Remove from self-cognition-created state. | Self-cognition is intended to generate memory; this flag is a memory-suppression control and should not be baked into the state contract. |
| Memory implementation | Defer to short-term memory-semantics plan. | The current incident is a context-overflow/defaults bugfix; the save taxonomy needs a wider design pass. |
| Visual directives | Disable for self-cognition by default. | Visual metadata is not needed for idle agency decisions and caused one overflow path. |
| Full consolidator graph | Do not call or modify for this bugfix. | The graph is a live post-turn consolidator and includes unrelated write lanes. |
| Global growth budget | Budget final rendered prompt, not only row count. | The model rejects final context size, not individual card count. |
| Prompt overflow recovery | Deterministic trim/drop with diagnostics. | Local/weaker LLMs need bounded input; retries or context-size increases hide the root cause. |
| Prompt-card priority | Preserve existing upstream order and drop from the tail. | The current retrieval/projection order is the only priority signal available in this bugfix. |

## Contracts And Data Shapes

### Self-Cognition Debug Modes

Self-cognition-created `debug_modes` in both the graph state and
`cognitive_episode.origin_metadata` must be exactly:

```python
{
    "no_visual_directives": True,
}
```

Self-cognition-created state and episode metadata must not include
`no_remember` at all. Both `{"no_remember": True}` and
`{"no_remember": False}` are forbidden.

### Global Growth Prompt Budget Config

Add this config constant in `src/kazusa_ai_chatbot/config.py`:

```python
GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET = _positive_int_from_env(
    "GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET",
    "32000",
)
```

Invalid zero or negative values must fail fast with
`GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET must be >= 1`.

### Global Growth Prompt Budget Types

Add this typed dict in
`src/kazusa_ai_chatbot/global_character_growth/models.py`:

```python
class PromptBudgetDiagnostics(TypedDict):
    """Auditable final prompt-size diagnostics for run documents."""

    prompt_char_budget: int
    rendered_prompt_chars_before_budget: int
    rendered_prompt_chars_after_budget: int
    memory_cards_before_prompt_budget: int
    memory_cards_after_prompt_budget: int
    dropped_memory_cards_for_prompt_budget: int
    prompt_budget_status: Literal[
        "within_budget",
        "trimmed_to_budget",
        "empty_after_budget",
    ]
```

Extend `GlobalCharacterGrowthRunResult` with:

```python
dropped_memory_cards_for_prompt_budget: int
rendered_prompt_chars_after_budget: int
```

### Global Growth Prompt Budget Helpers

Add this helper in `src/kazusa_ai_chatbot/global_character_growth/llm.py`:

```python
def count_candidate_generation_prompt_chars(
    *,
    payload: CandidatePromptPayload,
    character_name: str = "当前主体",
) -> int:
    """Return rendered system+human prompt characters for candidate generation."""
```

The helper must call `build_candidate_generation_prompt(...)` and return:

```python
len(rendered.system_prompt) + len(rendered.human_prompt)
```

Add this helper in `src/kazusa_ai_chatbot/global_character_growth/projection.py`:

```python
def build_budgeted_candidate_prompt_payload(
    *,
    memory_rows: Sequence[Mapping[str, Any]],
    current_trait_rows: Sequence[Mapping[str, Any]],
    prompt_char_budget: int,
    prompt_char_counter: Callable[[CandidatePromptPayload], int],
    limit: int = MAX_MEMORY_CARDS,
) -> tuple[CandidatePromptPayload, InputQualityDiagnostics, PromptBudgetDiagnostics]:
    """Return a candidate payload trimmed to the rendered prompt budget."""
```

Implementation requirements:

- Call `build_memory_cards(...)` once to preserve existing eligibility
  diagnostics.
- Project current traits with `project_current_traits(...)`.
- Build the candidate payload from the projected cards and traits.
- Calculate `rendered_prompt_chars_before_budget` before dropping cards.
- While the prompt has at least one memory card and
  `prompt_char_counter(payload) > prompt_char_budget`, drop one memory card
  from the end and recalculate.
- Set `prompt_budget_status` to:
  - `within_budget` when no card was dropped.
  - `trimmed_to_budget` when cards were dropped and the final prompt is within
    budget.
  - `empty_after_budget` when all cards were dropped and the final prompt still
    exceeds budget.
- Return the final payload, original input-quality diagnostics, and prompt
  budget diagnostics.

### Run Document Shape

Add a top-level run document field:

```python
"prompt_budget": PromptBudgetDiagnostics
```

For skipped runs with no eligible cards before budgeting, and for failed runs
whose exception happens before prompt budgeting, use zero numeric diagnostics,
`prompt_char_budget=GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET`, and
`prompt_budget_status="within_budget"`.

## LLM Call And Context Budget

Use `50k tokens` as the overall context-window assumption and enforce a
conservative character budget because local routes may tokenize CJK text
densely. The default is
`GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET=32000`. This is a character budget,
not an exact token budget.

| Call | Path | Before | After |
|---|---|---|---|
| Global growth candidate LLM | background reflection/growth | 1 call with up to 80 cards and no final prompt budget | 1 call with budgeted payload and drop diagnostics; 0 calls if all memory cards are dropped by prompt budget |
| Self-cognition L3 visual agent | background self-cognition | 1 optional visual LLM call per self-cognition case | 0 visual LLM calls for self-cognition by default |
| Self-cognition memory extraction | background self-cognition | 0 calls | 0 calls in this bugfix; covered by the short-term memory-semantics plan |

## Change Surface

Target ownership boundary: background self-cognition defaults and global
character-growth prompt budgeting.

### Modify

- `src/kazusa_ai_chatbot/config.py`
  - Set `SELF_COGNITION_ENABLED` default to `true`.
  - Add `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET` through
    `_positive_int_from_env(...)` with default `32000`.

- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - In `_build_cognition_state(...)`, replace the current
    `{"think_only": False, "no_remember": True}` with
    `{"no_visual_directives": True}`.
  - In `_build_cognitive_episode(...)`, replace the current
    `origin_metadata.debug_modes` with `{"no_visual_directives": True}`.

- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Update default enablement to `true`.
  - State that visual directives are disabled by default for self-cognition.
  - State that self-cognition memory persistence is planned separately, not
    implemented by this bugfix.

- `src/kazusa_ai_chatbot/global_character_growth/models.py`
  - Add `PromptBudgetDiagnostics`.
  - Extend `GlobalCharacterGrowthRunResult` with the two prompt-budget summary
    fields named in `Contracts And Data Shapes`.

- `src/kazusa_ai_chatbot/global_character_growth/llm.py`
  - Add `count_candidate_generation_prompt_chars(...)`.

- `src/kazusa_ai_chatbot/global_character_growth/projection.py`
  - Add `build_budgeted_candidate_prompt_payload(...)`.
  - Keep existing `build_candidate_prompt_payload(...)` behavior for callers
    and tests that use the non-budgeted projection directly.

- `src/kazusa_ai_chatbot/global_character_growth/runner.py`
  - Import `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET`.
  - Import `count_candidate_generation_prompt_chars(...)`.
  - Use `build_budgeted_candidate_prompt_payload(...)` before calling
    `generate_growth_candidates(...)`.
  - Validate candidates against the final budgeted payload's `memory_cards`.
  - Add `prompt_budget` to `_run_document(...)`.
  - Include the two new prompt-budget fields in `_result_from_run_doc(...)`.
  - If budget trimming leaves zero memory cards, write a skipped run document
    and do not call `generate_growth_candidates(...)`.

- `src/kazusa_ai_chatbot/global_character_growth/README.md`
  - Document `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET=32000`.
  - State that the candidate LLM receives budgeted final rendered prompts.

- `docs/HOWTO.md`
  - Document `SELF_COGNITION_ENABLED=true`.
  - Document that self-cognition disables visual directives by default.
  - Document `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET=32000` in the Global
    Character Growth section.

- `tests/test_config.py`
  - Update `TestSelfCognitionConfig.test_self_cognition_config_defaults_are_minimal`
    so the first expected line is `"True"`.
  - Add `TestGlobalCharacterGrowthConfig.test_prompt_char_budget_defaults_to_32000`.
  - Add `TestGlobalCharacterGrowthConfig.test_prompt_char_budget_fails_fast_when_invalid`.

- `tests/test_self_cognition_tracking.py`
  - Add `test_cognition_state_disables_visual_and_does_not_suppress_memory`.

- `tests/test_global_character_growth_contract.py`
  - Add `test_budgeted_candidate_prompt_payload_trims_until_rendered_prompt_fits`.
  - Add `test_budgeted_candidate_prompt_payload_reports_empty_after_budget`.

- `tests/test_global_character_growth_prompt_contracts.py`
  - Add `test_candidate_prompt_char_count_matches_rendered_prompts`.

- `tests/test_global_character_growth_runner.py`
  - Add `test_runner_uses_budgeted_payload_and_records_prompt_budget_diagnostics`.
  - Add `test_runner_skips_llm_when_prompt_budget_drops_all_cards`.

### Create

- No new source module is approved by this bugfix.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
  - Do not call or modify the full live-chat consolidator graph for
    self-cognition in this bugfix.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
  - Keep unchanged until the short-term memory-semantics plan defines a
    self-cognition origin contract.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
  - Keep unchanged until the short-term memory-semantics plan defines allowed
    persistence categories.

- Live `/chat` service path behavior for user-message consolidation.
- Scheduler and dispatcher validation behavior.
- Adapter delivery behavior.
- Reflection promotion behavior.
- Global character growth trait drift validation.

## Implementation Order

1. Add the self-cognition debug-mode failing test.
   - File: `tests/test_self_cognition_tracking.py`.
   - Add `test_cognition_state_disables_visual_and_does_not_suppress_memory`
     near `test_cognition_state_keeps_source_packet_inside_internal_percept`.
   - Test body: capture the state passed to `cognition_client`, run
     `run_self_cognition_case(...)`, assert both state-level and
     episode-level `debug_modes` are exactly `{"no_visual_directives": True}`,
     and assert neither contains `no_remember`.
   - Verify before implementation:
     `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_cognition_state_disables_visual_and_does_not_suppress_memory -q`
   - Expected before implementation: fails because `no_remember=True` is
     present and `no_visual_directives` is absent.

2. Implement the self-cognition debug-mode fix.
   - File: `src/kazusa_ai_chatbot/self_cognition/runner.py`.
   - In `_build_cognition_state(...)`, set `debug_modes` to
     `{"no_visual_directives": True}`.
   - In `_build_cognitive_episode(...)`, set
     `origin_metadata["debug_modes"]` to `{"no_visual_directives": True}`.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_cognition_state_disables_visual_and_does_not_suppress_memory -q`
   - Expected after implementation: passes.

3. Add the config failing tests.
   - File: `tests/test_config.py`.
   - Update
     `TestSelfCognitionConfig.test_self_cognition_config_defaults_are_minimal`
     to expect `"True"` for `SELF_COGNITION_ENABLED`.
   - Add `TestGlobalCharacterGrowthConfig` with
     `test_prompt_char_budget_defaults_to_32000` and
     `test_prompt_char_budget_fails_fast_when_invalid`. The first removes
     `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET` and expects stdout
     `"32000"`; the second sets it to `"0"` and expects a nonzero return code
     plus `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET must be >= 1`.
   - Verify before implementation:
     `venv\Scripts\python -m pytest tests\test_config.py::TestSelfCognitionConfig::test_self_cognition_config_defaults_are_minimal tests\test_config.py::TestGlobalCharacterGrowthConfig -q`
   - Expected before implementation: the self-cognition default assertion
     fails, and `TestGlobalCharacterGrowthConfig` fails because the config
     symbol does not exist.

4. Implement the config default and budget setting.
   - File: `src/kazusa_ai_chatbot/config.py`.
   - Change `SELF_COGNITION_ENABLED` default from `"false"` to `"true"`.
   - Add `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET` after
     `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED`.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_config.py::TestSelfCognitionConfig::test_self_cognition_config_defaults_are_minimal tests\test_config.py::TestGlobalCharacterGrowthConfig -q`
   - Expected after implementation: passes.

5. Add global-growth prompt-budget projection tests.
   - File: `tests/test_global_character_growth_contract.py`.
   - Add `test_budgeted_candidate_prompt_payload_trims_until_rendered_prompt_fits`.
     Use existing fixtures and a deterministic `prompt_char_counter` equal to
     `1000 + sum(len(card["content"]) + len(card["confidence_note"]) for card in payload["memory_cards"])`.
     Assert the final prompt is within budget, cards were dropped, and
     `prompt_budget_status == "trimmed_to_budget"`.
   - Add `test_budgeted_candidate_prompt_payload_reports_empty_after_budget`.
     Use a counter that always returns `prompt_char_budget + 1`; assert the
     payload has no `memory_cards`, status is `empty_after_budget`, and the
     dropped count equals the original eligible card count.
   - Verify before implementation:
     `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_trims_until_rendered_prompt_fits tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_reports_empty_after_budget -q`
   - Expected before implementation: fails because
     `build_budgeted_candidate_prompt_payload` does not exist.

6. Implement prompt-budget types and projection helper.
   - Files:
     - `src/kazusa_ai_chatbot/global_character_growth/models.py`
     - `src/kazusa_ai_chatbot/global_character_growth/projection.py`
   - Implement the contracts in `Global Growth Prompt Budget Types` and
     `Global Growth Prompt Budget Helpers`.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_trims_until_rendered_prompt_fits tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_reports_empty_after_budget -q`
   - Expected after implementation: passes.

7. Add and implement the prompt character counter.
   - Test file: `tests/test_global_character_growth_prompt_contracts.py`.
   - Add `test_candidate_prompt_char_count_matches_rendered_prompts`.
     Build the same small payload used by
     `test_build_candidate_prompt_renders_payload_separately`, call
     `llm.build_candidate_generation_prompt(...)`, call
     `llm.count_candidate_generation_prompt_chars(...)`, and assert the count
     equals `len(rendered.system_prompt) + len(rendered.human_prompt)`.
   - Source file: `src/kazusa_ai_chatbot/global_character_growth/llm.py`.
   - Implement `count_candidate_generation_prompt_chars(...)`.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_global_character_growth_prompt_contracts.py::test_candidate_prompt_char_count_matches_rendered_prompts -q`
   - Expected after implementation: passes.

8. Add runner integration tests for budgeted payloads.
   - File: `tests/test_global_character_growth_runner.py`.
   - Add `test_runner_uses_budgeted_payload_and_records_prompt_budget_diagnostics`.
     Patch `build_budgeted_candidate_prompt_payload` to return a known payload,
     input-quality diagnostics, and prompt-budget diagnostics; assert the LLM
     receives that payload and the run document/result expose the diagnostics.
   - Add `test_runner_skips_llm_when_prompt_budget_drops_all_cards`.
     Patch the budget helper to return `memory_cards=[]` and
     `prompt_budget_status="empty_after_budget"`; assert the LLM is not awaited
     and the skipped run document records the diagnostics.
   - Verify before implementation:
     `venv\Scripts\python -m pytest tests\test_global_character_growth_runner.py::test_runner_uses_budgeted_payload_and_records_prompt_budget_diagnostics tests\test_global_character_growth_runner.py::test_runner_skips_llm_when_prompt_budget_drops_all_cards -q`
   - Expected before implementation: fails because the runner still calls
     `build_candidate_prompt_payload(...)` and run documents have no
     `prompt_budget`.

9. Implement runner prompt-budget integration.
   - File: `src/kazusa_ai_chatbot/global_character_growth/runner.py`.
   - Replace the separate `build_memory_cards(...)` plus
     `build_candidate_prompt_payload(...)` path with
     `build_budgeted_candidate_prompt_payload(...)`.
   - Use a lambda or small local function that calls
     `count_candidate_generation_prompt_chars(payload=payload)`.
   - For candidate validation, set `memory_cards = prompt_payload["memory_cards"]`.
   - When `memory_cards` is empty after budgeting, write a skipped run document
     with summary `"No eligible reflection-promoted memory cards after prompt budget."`
     and do not call the LLM.
   - Add `prompt_budget` to every `_run_document(...)` call.
   - Add `prompt_budget` to `_run_document(...)` parameters and output.
   - Add the two prompt-budget summary fields in `_result_from_run_doc(...)`.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_global_character_growth_runner.py::test_runner_uses_budgeted_payload_and_records_prompt_budget_diagnostics tests\test_global_character_growth_runner.py::test_runner_skips_llm_when_prompt_budget_drops_all_cards -q`
   - Expected after implementation: passes.

10. Update docs.
    - Files:
      - `src/kazusa_ai_chatbot/self_cognition/README.md`
      - `src/kazusa_ai_chatbot/global_character_growth/README.md`
      - `docs/HOWTO.md`
    - Make the documented self-cognition default `true`.
    - Document default self-cognition visual disable.
    - Document that self-cognition memory persistence is deferred to the
      short-term memory-semantics plan.
    - Document `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET=32000`.
    - Verify with static greps in the `Verification` section.

11. Run focused verification.
    - Run every command under `Verification`.
    - Fix only failures inside this plan's change surface.

12. Run independent code review.
    - Follow the `Independent Code Review` gate before sign-off.

## Progress Checklist

- [x] Stage 1 - self-cognition debug-mode contract fixed
  - Covers: implementation steps 1-2.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_cognition_state_disables_visual_and_does_not_suppress_memory -q`
  - Evidence: record failing-before and passing-after output in
    `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 2 - config defaults and prompt-budget config fixed
  - Covers: implementation steps 3-4.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_config.py::TestSelfCognitionConfig::test_self_cognition_config_defaults_are_minimal tests\test_config.py::TestGlobalCharacterGrowthConfig -q`
  - Evidence: record failing-before and passing-after output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 3 - global-growth projection budget contract implemented
  - Covers: implementation steps 5-7.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_trims_until_rendered_prompt_fits tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_reports_empty_after_budget tests\test_global_character_growth_prompt_contracts.py::test_candidate_prompt_char_count_matches_rendered_prompts -q`
  - Evidence: record prompt-size diagnostics and test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 4 - global-growth runner budget integration implemented
  - Covers: implementation steps 8-9.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_global_character_growth_runner.py::test_runner_uses_budgeted_payload_and_records_prompt_budget_diagnostics tests\test_global_character_growth_runner.py::test_runner_skips_llm_when_prompt_budget_drops_all_cards -q`
  - Evidence: record run-document prompt-budget diagnostics and test output.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 5 - docs and regression checks complete
  - Covers: implementation steps 10-11.
  - Verify: all commands in `Verification` pass or have a documented,
    user-approved exception.
  - Evidence: record doc paths, focused tests, broader regression output, and
    static grep output.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `Codex/2026-05-14` after verification and evidence are recorded.

- [x] Stage 6 - independent code review complete
  - Covers: implementation step 12.
  - Verify: full diff reviewed against this plan and affected tests rerun after
    any review fixes.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, residual
    risks, and approval status.
  - Handoff: plan can be marked completed only after this stage is signed off.
  - Sign-off: `Codex/2026-05-14` after review evidence is recorded.

## Verification

### Focused Deterministic Tests

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_cognition_state_disables_visual_and_does_not_suppress_memory -q
venv\Scripts\python -m pytest tests\test_config.py::TestSelfCognitionConfig::test_self_cognition_config_defaults_are_minimal tests\test_config.py::TestGlobalCharacterGrowthConfig -q
venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_trims_until_rendered_prompt_fits tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_reports_empty_after_budget -q
venv\Scripts\python -m pytest tests\test_global_character_growth_prompt_contracts.py::test_candidate_prompt_char_count_matches_rendered_prompts -q
venv\Scripts\python -m pytest tests\test_global_character_growth_runner.py::test_runner_uses_budgeted_payload_and_records_prompt_budget_diagnostics tests\test_global_character_growth_runner.py::test_runner_skips_llm_when_prompt_budget_drops_all_cards -q
```

Expected: each command fails before its implementation stage and passes after
that stage.

### Existing Regression Tests

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_framing.py tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_self_cognition_event_logging.py tests\test_self_cognition_dry_run_cli.py -q
venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py tests\test_global_character_growth_context.py tests\test_global_character_growth_drift.py tests\test_global_character_growth_module_boundary.py tests\test_global_character_growth_prompt_contracts.py tests\test_global_character_growth_runner.py tests\test_global_character_growth_validation.py tests\test_global_character_growth_worker.py -q
venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_service.py tests\test_service_ops_status.py -q
```

Expected: all deterministic tests pass.

`tests\test_global_character_growth_live_llm.py` is a live LLM contract test
and is not part of the deterministic verification gate. Run it only when the
owner explicitly requests live LLM validation.

### Static Greps

```powershell
rg -n '"no_remember": True|no_remember.*self_cognition|self_cognition.*no_remember' src\kazusa_ai_chatbot\self_cognition tests
```

Expected: no matches that set or expect `no_remember` in self-cognition-created
state. Exit code 1 from `rg` is acceptable because zero matches are expected.
Documentation outside the searched paths may mention that self-cognition must
not set it.

```powershell
rg -n 'no_visual_directives' src\kazusa_ai_chatbot\self_cognition tests\test_self_cognition_tracking.py
```

Expected: matches show self-cognition disables visual directives by default and
tests assert that behavior.

```powershell
rg -n 'SELF_COGNITION_ENABLED.*false|default `false`|default false' docs src\kazusa_ai_chatbot tests README.md README_CN.md
```

Expected: no stale self-cognition default-false docs or tests remain. Exit code
1 from `rg` is acceptable when there are zero matches. Matches for unrelated
settings must be inspected and recorded.

```powershell
rg -n 'run_self_cognition_memory|self_cognition_memory_lane|self_cognition.*origin|origin.*self_cognition' src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\self_cognition tests
```

Expected: no new self-cognition memory writer, consolidator origin, or origin
policy implementation is introduced by this bugfix. Exit code 1 from `rg` is
acceptable when there are zero matches. Existing tracking or documentation
matches must be inspected and recorded.

```powershell
rg -n 'GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET|prompt_budget|dropped_memory_cards_for_prompt_budget' src\kazusa_ai_chatbot\global_character_growth src\kazusa_ai_chatbot\config.py tests docs\HOWTO.md
```

Expected: matches show the config, model type, projection helper, runner run
document, tests, and docs were updated.

## Independent Plan Review

Reviewer mode: same-agent independent review from a fresh-review posture after
reading the plan contract references, `development_plans/README.md`, relevant
source, docs, and tests on 2026-05-14.

Findings resolved before approval:

| Finding | Severity | Resolution |
|---|---|---|
| The previous plan was still `draft`, so it was not executable under the registry contract. | Blocker | Status changed to `approved`; registry row updated to `approved`. |
| The global-growth prompt-budget helper contract was conceptual and left the implementer to decide where prompt rendering was measured. | Blocker | Added exact helper names, signatures, diagnostics, run-document shape, and trim algorithm. |
| Test instructions referred to broad file globs and unnamed tests. | Blocker | Replaced with exact test function names and exact commands. |
| The plan risked reintroducing self-cognition memory implementation into the bugfix. | Blocker | Added explicit forbidden surfaces, static grep, and acceptance criterion proving no memory writer/origin-policy change is included. |
| Documentation surface omitted the global character growth ICD. | Non-blocking | Added `src/kazusa_ai_chatbot/global_character_growth/README.md` to the change surface. |

Review outcome: approved for implementation after the listed edits. Execution
must still follow the progress checklist and verification gates in this plan.

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
  brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including focused and regression tests,
  static-grep accuracy, execution evidence, and lifecycle registry updates.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only fixture/documentation
corrections. If a fix would cross the approved boundary or alter the contract,
stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Global character growth cannot send an over-budget candidate prompt under the
  configured default budget.
- Global character growth run documents include `prompt_budget` diagnostics.
- Global character growth public run results expose
  `dropped_memory_cards_for_prompt_budget` and
  `rendered_prompt_chars_after_budget`.
- Self-cognition-created state and episode metadata do not contain
  `no_remember`.
- Self-cognition-created state and episode metadata contain exactly
  `{"no_visual_directives": True}` for `debug_modes`.
- Production self-cognition no longer invokes the L3 visual-agent LLM by
  default.
- Docs and tests agree that `SELF_COGNITION_ENABLED` defaults to `true`.
- No self-cognition memory writer, consolidator origin, or origin-policy change
  is implemented by this bugfix.
- All listed verification commands pass or have documented, user-approved
  exceptions.
- The independent code review gate is completed and recorded in
  `Execution Evidence`.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Removing `no_remember` is mistaken for completed memory persistence | State explicitly that this bugfix removes a blocker only and defer memory semantics to the short-term plan | Acceptance criteria and static grep prove no memory writer was added |
| Global growth drops useful cards | Drop only from the end of the existing card order and record prompt-budget diagnostics | Projection and runner tests inspect diagnostics |
| System prompt alone exceeds a very small configured budget | Return `empty_after_budget`, skip the LLM, and record diagnostics | `test_budgeted_candidate_prompt_payload_reports_empty_after_budget` and runner skip test |
| Visual directives are accidentally disabled for live chat | Scope `no_visual_directives` to self-cognition-created episodes only | Self-cognition tests plus existing live-chat visual config tests |
| Memory feature remains incomplete after bugfix | Track memory semantics in the active short-term plan | Registry lists the follow-up plan |

## Execution Evidence

- Stage 1 failing-before:
  `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_cognition_state_disables_visual_and_does_not_suppress_memory -q`
  failed as expected. Assertion showed state debug modes were
  `{"think_only": False, "no_remember": True}` instead of
  `{"no_visual_directives": True}`.
- Stage 1 passing-after:
  `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_cognition_state_disables_visual_and_does_not_suppress_memory -q`
  passed with `1 passed`.
- Stage 2 failing-before:
  `venv\Scripts\python -m pytest tests\test_config.py::TestSelfCognitionConfig::test_self_cognition_config_defaults_are_minimal tests\test_config.py::TestGlobalCharacterGrowthConfig -q`
  failed as expected. The self-cognition default was still `False`,
  `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET` was absent, and invalid budget
  input did not fail.
- Stage 2 passing-after:
  `venv\Scripts\python -m pytest tests\test_config.py::TestSelfCognitionConfig::test_self_cognition_config_defaults_are_minimal tests\test_config.py::TestGlobalCharacterGrowthConfig -q`
  passed with `3 passed`.
- Stage 3 failing-before:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_trims_until_rendered_prompt_fits tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_reports_empty_after_budget -q`
  failed as expected because
  `projection.build_budgeted_candidate_prompt_payload` did not exist.
- Stage 3 prompt-counter failing-before:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_prompt_contracts.py::test_candidate_prompt_char_count_matches_rendered_prompts -q`
  failed as expected because
  `llm.count_candidate_generation_prompt_chars` did not exist.
- Stage 3 prompt-size diagnostics:
  the trim test records `rendered_prompt_chars_before_budget=3152`,
  `rendered_prompt_chars_after_budget=2076`,
  `memory_cards_before_prompt_budget=4`,
  `memory_cards_after_prompt_budget=2`, and
  `prompt_budget_status="trimmed_to_budget"` under a `2100` character budget.
- Stage 3 passing-after:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_trims_until_rendered_prompt_fits tests\test_global_character_growth_contract.py::test_budgeted_candidate_prompt_payload_reports_empty_after_budget tests\test_global_character_growth_prompt_contracts.py::test_candidate_prompt_char_count_matches_rendered_prompts -q`
  passed with `3 passed`.
- Stage 3 CJK syntax check:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\global_character_growth\llm.py`
  completed successfully after editing the CJK prompt module.
- Stage 4 failing-before:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_runner.py::test_runner_uses_budgeted_payload_and_records_prompt_budget_diagnostics tests\test_global_character_growth_runner.py::test_runner_skips_llm_when_prompt_budget_drops_all_cards -q`
  failed as expected because the runner did not call
  `build_budgeted_candidate_prompt_payload`.
- Stage 4 run-document diagnostics:
  the budgeted-payload test records run-document
  `prompt_budget.prompt_budget_status="trimmed_to_budget"`,
  `dropped_memory_cards_for_prompt_budget=1`, and
  `rendered_prompt_chars_after_budget=3000`; the budget-empty skip test
  records `prompt_budget.prompt_budget_status="empty_after_budget"`,
  `dropped_memory_cards_for_prompt_budget=2`, and no LLM call.
- Stage 4 passing-after:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_runner.py::test_runner_uses_budgeted_payload_and_records_prompt_budget_diagnostics tests\test_global_character_growth_runner.py::test_runner_skips_llm_when_prompt_budget_drops_all_cards -q`
  passed with `2 passed`.
- Stage 5 docs updated:
  `src/kazusa_ai_chatbot/self_cognition/README.md`,
  `src/kazusa_ai_chatbot/global_character_growth/README.md`, and
  `docs/HOWTO.md` now document default self-cognition enablement, default
  self-cognition visual-directive disablement, deferred self-cognition memory
  persistence, and `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET=32000`.
- Stage 5 focused verification:
  all five focused deterministic test commands under `Verification` passed
  with `1 passed`, `3 passed`, `2 passed`, `1 passed`, and `2 passed`.
- Stage 5 regression verification:
  `venv\Scripts\python -m pytest tests\test_self_cognition_framing.py tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_self_cognition_event_logging.py tests\test_self_cognition_dry_run_cli.py -q`
  passed with `36 passed`.
- Stage 5 regression verification:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py tests\test_global_character_growth_context.py tests\test_global_character_growth_drift.py tests\test_global_character_growth_module_boundary.py tests\test_global_character_growth_prompt_contracts.py tests\test_global_character_growth_runner.py tests\test_global_character_growth_validation.py tests\test_global_character_growth_worker.py -q`
  passed with `52 passed`.
- Stage 5 regression verification:
  `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_service.py tests\test_service_ops_status.py -q`
  passed with `10 passed`.
- Stage 5 static grep:
  `rg -n '"no_remember": True|no_remember.*self_cognition|self_cognition.*no_remember' src\kazusa_ai_chatbot\self_cognition tests`
  returned existing non-self-cognition-created debug-mode tests in
  `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py`,
  `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`,
  `tests/test_persona_supervisor2.py`, and `tests/test_state.py`; no matches
  were in `src/kazusa_ai_chatbot/self_cognition`.
- Stage 5 static grep:
  `rg -n 'no_visual_directives' src\kazusa_ai_chatbot\self_cognition tests\test_self_cognition_tracking.py`
  returned the self-cognition runner, README, and tracking test assertions.
- Stage 5 static grep:
  `rg -n 'SELF_COGNITION_ENABLED.*false|default `false`|default false' docs src\kazusa_ai_chatbot tests README.md README_CN.md`
  returned only `src/kazusa_ai_chatbot/service.py` log text for explicit
  disablement via `SELF_COGNITION_ENABLED=false`; no stale default-false doc or
  test expectation remained.
- Stage 5 static grep:
  `rg -n 'run_self_cognition_memory|self_cognition_memory_lane|self_cognition.*origin|origin.*self_cognition' src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\self_cognition tests`
  returned no matches, as expected.
- Stage 5 static grep:
  `rg -n 'GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET|prompt_budget|dropped_memory_cards_for_prompt_budget' src\kazusa_ai_chatbot\global_character_growth src\kazusa_ai_chatbot\config.py tests docs\HOWTO.md`
  returned the expected config, model, projection, runner, test, and docs
  matches.
- Stage 5 syntax verification:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\self_cognition\runner.py src\kazusa_ai_chatbot\global_character_growth\models.py src\kazusa_ai_chatbot\global_character_growth\projection.py src\kazusa_ai_chatbot\global_character_growth\llm.py src\kazusa_ai_chatbot\global_character_growth\runner.py tests\test_config.py tests\test_self_cognition_tracking.py tests\test_global_character_growth_contract.py tests\test_global_character_growth_prompt_contracts.py tests\test_global_character_growth_runner.py`
  completed successfully.
- Stage 6 independent code review:
  no separate reviewer was available in this session. Same-agent review was run
  from a fresh-review posture after rereading this completed plan and
  inspecting the full changed source, tests, docs, registry, and plan diff.
- Stage 6 review finding:
  one low-risk type-quality issue was found in
  `src/kazusa_ai_chatbot/global_character_growth/runner.py`; the local prompt
  character counter accepted `dict[str, Any]` even though the budget helper
  passes `CandidatePromptPayload`.
- Stage 6 review fix:
  changed the runner-local `prompt_char_counter` parameter annotation to
  `CandidatePromptPayload`.
- Stage 6 review rerun:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\global_character_growth\runner.py`
  completed successfully after the review fix.
- Stage 6 review rerun:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_runner.py::test_runner_uses_budgeted_payload_and_records_prompt_budget_diagnostics tests\test_global_character_growth_runner.py::test_runner_skips_llm_when_prompt_budget_drops_all_cards -q`
  passed with `2 passed`.
- Stage 6 diff hygiene:
  `git diff --check` returned no whitespace errors. It emitted only Git line
  ending normalization warnings for edited text files.
- Stage 6 residual risks:
  no code-review findings remain. Self-cognition memory persistence is still
  intentionally deferred to
  `development_plans/active/short_term/self_cognition_memory_semantics_plan.md`.
- Stage 6 approval status:
  approved for completed-plan archival.

## Execution Handoff

Execution is complete. The plan is ready to move from active bugfix to
`development_plans/archive/completed/bugfix/`.
