# cognition visual directives control plan

## Summary

- Goal: Add a shared control for L3 visual-directive generation so `/chat`
  can disable it from service config and self-cognition can disable it per
  internal run without exposing a new adapter/debug-client request field.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, and `cjk-safety`.
- Overall cutover strategy: compatible by default; bigbang only for the new
  disabled-mode behavior when explicitly configured or requested by an internal
  self-cognition caller.
- Highest-risk areas: accidentally exposing the control to adapters, changing
  generic `/chat` debug-mode semantics, breaking internal-thought prompt-key
  audit, adding prompt churn in CJK-heavy cognition files, and hiding an LLM
  call-count change without tests.
- Acceptance criteria: default `/chat` and internal-thought cognition still run
  the visual agent; the visual agent runs only when
  `COGNITION_VISUAL_DIRECTIVES_ENABLED` and `visual_directives_enabled` are both
  true; either control can disable visual directives; skipped visual generation
  returns the existing empty-list output shape; no adapter payload or public
  brain-service request schema exposes the new control.

## Context

The current `/chat` path accepts adapter-supplied debug modes
`listen_only`, `think_only`, and `no_remember` through
`src/kazusa_ai_chatbot/brain_service/contracts.py`. The debug adapter exposes
those fields per message. The requested new visual-generation control must not
follow that pattern for `/chat`: it is an operator/runtime config, not an
adapter-controlled per-message switch.

The L3 visual agent lives in
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` and returns
`facial_expression`, `body_language`, `gaze_direction`, and `visual_vibe`.
The L4 collector packages these values into
`action_directives.visual_directives`. Dialog generation does not consume
`visual_directives`; they are currently downstream generation metadata and
inspection output.

The self-cognition direction is represented by
`development_plans/archive/completed/short_term/self_cognition_agency_loop_plan.md` and
`development_plans/archive/superseded/self_cognition_loop_architecture.md`.
Those documents are context only. The relevant current production-ish entry
point is `src/kazusa_ai_chatbot/internal_thought_cognition.py`, which already
builds internal-thought `CognitiveEpisode` objects and calls the shared
cognition subgraph. This plan adds the visual-directive disable control to
that existing path, without implementing broader self-cognition stages.

## Mandatory Skills

- `development-plan-writing`: load before modifying this plan or its registry
  row.
- `local-llm-architecture`: load before changing cognition graph behavior or
  LLM call budgets.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` because it
  contains CJK prompt text.

## Mandatory Rules

- Preserve the existing pipeline boundary: adapter/debug client -> brain
  service -> queue/intake -> RAG -> cognition -> dialog ->
  persistence/consolidation. Do not move visual-control interpretation into
  adapters.
- Do not add `no_visual_directives` or any equivalent field to
  `ChatRequest`, `DebugModesIn`, the debug adapter UI, Discord adapter, or
  NapCat adapter.
- `/chat` must read the visual-directive behavior from config only.
- Self-cognition/internal-thought callers may pass an explicit internal
  function argument to disable visual directives for that run.
- The effective visual-agent enable condition is:
  `COGNITION_VISUAL_DIRECTIVES_ENABLED and visual_directives_enabled`. If either
  value is false, the implementation must set the internal skip state and must
  not call the visual-agent LLM.
- The visual-agent skip path must return the same structural contract as the
  current visual agent: all four visual fields are present and are lists.
- Do not rewrite `_VISUAL_AGENT_PROMPT`, prompt-selection variants, dialog
  prompts, RAG prompts, finalizers, persistence, or Cache2 behavior.
- Do not add retry loops, compatibility wrappers, fallback prompts, or extra
  LLM calls.
- Do not use deterministic code to reinterpret social intent, user
  instructions, promises, permissions, or relationship meaning.
- Use `venv\Scripts\python` for Python verification commands.
- Regular deterministic tests may run in batches. Live LLM tests must run one
  case at a time and be inspected before another live LLM test runs.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.

## Must Do

- Add config constant `COGNITION_VISUAL_DIRECTIVES_ENABLED` in
  `src/kazusa_ai_chatbot/config.py`, defaulting to enabled.
- In `/chat`, convert the config value into internal
  `debug_modes["no_visual_directives"] = True` only when the config disables
  visual directives. The `/chat` path has no per-adapter or per-request
  `visual_directives_enabled` input, so its implicit per-run value is true.
- Add optional `no_visual_directives` to the internal `DebugModes` typed dict
  in `src/kazusa_ai_chatbot/state.py`.
- In `call_visual_agent(...)`, skip `_visual_agent_llm.ainvoke(...)` when the
  current cognitive episode origin debug modes contain
  `no_visual_directives: True`.
- In `src/kazusa_ai_chatbot/internal_thought_cognition.py`, add a
  `visual_directives_enabled: bool = True` parameter to the episode builder
  and dry-run runner. Combine it with `COGNITION_VISUAL_DIRECTIVES_ENABLED`.
  When either value is false, include the same internal debug-mode flag and
  return audit prompt keys that do not claim the visual-agent prompt ran.
- Update focused tests and docs listed in this plan.
- Preserve all default behavior when the config and internal parameter are not
  changed.

## Deferred

- Do not add outbound image generation, image attachments, image files, or
  image-tool integration.
- Do not implement self-cognition agenda loops, RAG2 idle retrieval,
  proactive preview, outbox persistence, dispatcher handoff, or scheduler
  changes.
- Do not add a public `/chat` request field, adapter UI toggle, Discord flag,
  NapCat flag, or debug web control for the visual-directive setting.
- Do not remove existing `listen_only`, `think_only`, or `no_remember`
  behavior.
- Do not change prompt text, prompt fingerprints, prompt-selection variants,
  cognition output field names, dialog behavior, consolidation write policy, or
  persistence schemas.
- Do not create a new module unless a style review proves the existing owners
  cannot keep the behavior readable. The expected implementation uses existing
  modules.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Default `/chat` behavior | compatible | With `COGNITION_VISUAL_DIRECTIVES_ENABLED` unset or true, keep visual-agent execution and output shape unchanged. `/chat` uses implicit `visual_directives_enabled=True`. |
| `/chat` disabled behavior | bigbang | When config is false, skip the visual-agent LLM call directly. Do not fall back to prompt changes or downstream filtering. |
| Self-cognition/internal-thought behavior | compatible | Default parameter value keeps current behavior only when `COGNITION_VISUAL_DIRECTIVES_ENABLED` is also true. Passing `visual_directives_enabled=False` disables visual directives for that run, and config false disables it globally. |
| Brain service request schema | compatible | Do not add or change public request fields. Existing adapters remain compatible. |
| Adapter behavior | compatible | No adapter code changes are authorized. |
| Tests and docs | compatible | Add tests and docs for the new control without deleting existing debug-mode tests. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, rewrite the targeted behavior directly instead of
  preserving an old disabled-mode path.
- If an area is `compatible`, preserve only the compatibility surfaces listed
  in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The target ownership boundary is L3 cognition visual-directive generation and
  its immediate internal controls.
- The agent may choose local implementation mechanics only when they preserve
  the contracts in this plan.
- The agent must not introduce new architecture, alternate runtime controls,
  compatibility layers, fallback paths, helper wrappers, or extra features.
- The agent must treat changes outside the listed change surface as
  high-scrutiny changes and stop for approval unless the change is required to
  keep tests aligned with the approved contract.
- The agent must not perform unrelated cleanup, formatting churn, dependency
  upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

`/chat` has one service-level config switch:

```text
COGNITION_VISUAL_DIRECTIVES_ENABLED=true
```

When the value is true or unset, current behavior is unchanged. When false,
the service seeds the internal graph state with:

```python
debug_modes["no_visual_directives"] = True
```

For `/chat`, `visual_directives_enabled` is not a request or adapter field. The
service uses an implicit per-run `visual_directives_enabled=True`, so config is
the only `/chat` control surface.

For self-cognition/internal-thought, the effective enablement is:

```python
effective_visual_directives_enabled = (
    COGNITION_VISUAL_DIRECTIVES_ENABLED and visual_directives_enabled
)
```

If the effective value is false, the internal graph state receives the same
skip flag:

```python
debug_modes["no_visual_directives"] = True
```

The L3 visual agent reads the current `cognitive_episode.origin_metadata`
debug modes. If `no_visual_directives` is true, it returns:

```python
{
    "facial_expression": [],
    "body_language": [],
    "gaze_direction": [],
    "visual_vibe": [],
}
```

Self-cognition/internal-thought can disable the same behavior without changing
global config, provided global config has not already disabled it:

```python
await run_internal_thought_cognition_dry_run(
    ...,
    visual_directives_enabled=False,
)
```

The disabled path does not call `_visual_agent_llm`, does not change dialog,
and does not expose the flag through adapters.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| `/chat` control source | Use config only | User explicitly rejected per-adapter control for `/chat`. |
| Internal state carrier | Use `debug_modes["no_visual_directives"]` | Existing episode metadata already carries structured debug-mode flags through cognition and validates string-to-bool shape. |
| Public API shape | No `ChatRequest` change | Adapter clients must stay thin and compatible. |
| Self-cognition control | Add a function parameter and combine it with config | Self-cognition needs per-run control, while `COGNITION_VISUAL_DIRECTIVES_ENABLED` remains a global kill switch. The visual agent may run only when both values are true. |
| Skip location | Skip inside `call_visual_agent(...)` | The visual agent owns the visual-directive LLM call and output contract. Skipping there avoids graph rewiring. |
| Output shape | Return empty lists | This satisfies the existing `l3_visual_agent` output contract and keeps L4 collector shape stable. |
| Prompt behavior | No prompt changes | This is deterministic call gating, not prompt quality work. |

## Contracts And Data Shapes

### Config

`src/kazusa_ai_chatbot/config.py` adds:

```python
COGNITION_VISUAL_DIRECTIVES_ENABLED = os.getenv(
    "COGNITION_VISUAL_DIRECTIVES_ENABLED",
    "true",
).lower() in ("1", "true", "yes")
```

### Internal Debug Mode

`src/kazusa_ai_chatbot/state.py` updates `DebugModes`:

```python
class DebugModes(TypedDict, total=False):
    listen_only: bool
    think_only: bool
    no_remember: bool
    no_visual_directives: bool
```

This is an internal state flag. It is not added to
`brain_service.contracts.DebugModesIn`.

### Visual-Agent Skip

`call_visual_agent(state)` must read:

```python
episode = state["cognitive_episode"]
debug_modes = episode["origin_metadata"]["debug_modes"]
```

If `debug_modes.get("no_visual_directives")` is true, it must return the empty
visual-directive payload and must not invoke `_visual_agent_llm`.

### Internal-Thought Control

`build_internal_thought_cognitive_episode(...)` and
`run_internal_thought_cognition_dry_run(...)` gain:

```python
visual_directives_enabled: bool = True
```

The builder and runner must compute:

```python
effective_visual_directives_enabled = (
    COGNITION_VISUAL_DIRECTIVES_ENABLED and visual_directives_enabled
)
```

When the effective value is false, origin metadata debug modes include:

```python
{
    "think_only": True,
    "no_remember": True,
    "no_visual_directives": True,
}
```

Audit prompt keys must omit:

```text
l3_visual_agent.internal_thought_internal_monologue
```

when visual directives are disabled.

## LLM Call And Context Budget

Before this plan:

- Normal `/chat` response path runs the existing L3 visual-agent LLM call when
  the cognition subgraph reaches `call_visual_agent`.
- Internal-thought dry runs include the visual-agent prompt key and, through
  the shared cognition subgraph, run the visual-agent LLM unless a test patches
  it.

After this plan:

- Default `/chat`: unchanged call count and prompt context because
  `COGNITION_VISUAL_DIRECTIVES_ENABLED=True` and implicit
  `visual_directives_enabled=True`.
- `/chat` with `COGNITION_VISUAL_DIRECTIVES_ENABLED=false`: one fewer
  response-path cognition LLM call. No prompt context is sent to the visual
  agent.
- Default internal-thought/self-cognition: unchanged call count and prompt
  context only when `COGNITION_VISUAL_DIRECTIVES_ENABLED=True` and
  `visual_directives_enabled=True`.
- Internal-thought/self-cognition with either
  `COGNITION_VISUAL_DIRECTIVES_ENABLED=False` or
  `visual_directives_enabled=False`: one fewer background cognition LLM call.
  No prompt context is sent to the visual agent.

No new response-path calls, background calls, prompt fields, prompt templates,
context caps, or retry paths are authorized.

## Change Surface

### Create

- `development_plans/archive/completed/short_term/cognition_visual_directives_control_plan.md`
  - Owns this execution contract.

### Modify

- `development_plans/README.md`
  - Track this plan in the lifecycle registry. Completed execution moves the
    record to `archive/completed/short_term/`.
- `src/kazusa_ai_chatbot/config.py`
  - Add `COGNITION_VISUAL_DIRECTIVES_ENABLED`.
- `docs/HOWTO.md`
  - Document the config variable in the example `.env` and note that it is a
    service-level visual-directive control, not a debug-adapter toggle.
- `src/kazusa_ai_chatbot/state.py`
  - Add optional internal `no_visual_directives` to `DebugModes`.
- `src/kazusa_ai_chatbot/service.py`
  - Import the new config and merge the disabled state into internal
    `debug_modes` before building the cognitive episode.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Add the early skip in `call_visual_agent(...)`.
- `src/kazusa_ai_chatbot/internal_thought_cognition.py`
  - Add the internal per-run control, combine it with global config, and adjust
    audit prompt keys when either switch disables visual directives.
- `tests/test_config.py`
  - Verify config default and false parsing.
- `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
  - Verify `/chat` default and config-disabled state handoff.
- `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`
  - Verify the cognitive episode origin metadata carries the internal visual
    flag only when service config disables visual directives.
- `tests/test_conversation_progress_cognition.py`
  - Verify visual-agent skip returns empty lists and does not call the LLM.
- `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
  - Verify internal-thought default behavior remains when both switches are
    true, and disabled runs include the internal flag plus adjusted prompt-key
    audit when either switch is false.

### Keep

- `src/kazusa_ai_chatbot/brain_service/contracts.py`
  - No `DebugModesIn` change.
- `src/adapters/debug_adapter.py`
  - No new UI toggle or request field.
- `src/adapters/discord_adapter.py`
  - No new adapter config or debug flag.
- `src/adapters/napcat_qq_adapter.py`
  - No new adapter config or debug flag.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
  - No new prompt variant.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - No dialog behavior changes.
- Consolidation, DB, dispatcher, scheduler, RAG, and Cache2 modules
  - No persistence or execution changes.

## Implementation Order

1. Add focused tests for config parsing, service handoff, visual-agent skip,
   and internal-thought disable audit. Run the focused tests before
   implementation and record the expected failures.
2. Add `COGNITION_VISUAL_DIRECTIVES_ENABLED` to config and docs.
3. Add the internal `DebugModes` type field and service handoff from config.
4. Add the visual-agent skip in `call_visual_agent(...)`, then run the visual
   focused test.
5. Add the internal-thought parameter and audit prompt-key adjustment, then run
   the internal-thought focused tests.
6. Run all verification commands in this plan.
7. Run the independent code review gate and remediate approved findings.

## Progress Checklist

- [x] Stage 1 - test contract written
  - Covers: focused tests in `tests/test_config.py`,
    `tests/test_multi_source_cognition_stage_00_regression_baseline.py`,
    `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`,
    `tests/test_conversation_progress_cognition.py`, and
    `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`.
  - Verify: run each focused test and record the expected pre-implementation
    failure in `Execution Evidence`.
  - Handoff: next stage is config and service implementation.
  - Sign-off: `Codex/2026-05-11` after evidence is recorded.
- [x] Stage 2 - `/chat` config control complete
  - Covers: `config.py`, `service.py`, `state.py`, and `docs/HOWTO.md`.
  - Verify: focused config and service handoff tests pass.
  - Handoff: next stage is visual-agent skip implementation.
  - Sign-off: `Codex/2026-05-11` after evidence is recorded.
- [x] Stage 3 - visual-agent skip complete
  - Covers: `persona_supervisor2_cognition_l3.py`.
  - Verify: visual-agent skip test passes and py_compile passes for the CJK
    prompt file.
  - Handoff: next stage is internal-thought control.
  - Sign-off: `Codex/2026-05-11` after evidence is recorded.
- [x] Stage 4 - internal-thought control complete
  - Covers: `internal_thought_cognition.py` and Stage 08 tests.
  - Verify: internal-thought focused tests pass.
  - Handoff: next stage is full verification.
  - Sign-off: `Codex/2026-05-11` after evidence is recorded.
- [x] Stage 5 - full verification complete
  - Covers: all verification commands in this plan.
  - Verify: static grep, py_compile, focused tests, and regression tests pass.
  - Handoff: next stage is independent code review.
  - Sign-off: `Codex/2026-05-11` after evidence is recorded.
- [x] Stage 6 - independent code review complete
  - Covers: full implementation diff, tests, docs, and lifecycle record.
  - Verify: review result and any rerun commands are recorded in
    `Execution Evidence`.
  - Handoff: plan may be marked completed only after this stage passes.
  - Sign-off: `Codex/2026-05-11` after evidence is recorded.

## Verification

### Static Checks

- `rg -n "no_visual_directives|COGNITION_VISUAL_DIRECTIVES_ENABLED|visual_directives_enabled" src/adapters src/kazusa_ai_chatbot/brain_service/contracts.py`
  - Expected: no matches. Exit code 1 from `rg` is acceptable.
- `rg -n "no_visual_directives|COGNITION_VISUAL_DIRECTIVES_ENABLED|visual_directives_enabled" src/kazusa_ai_chatbot docs tests`
  - Expected: matches only in the files listed in `Change Surface`.

### Compile

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\internal_thought_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py
```

Expected: exit code 0.

### Focused Tests

```powershell
venv\Scripts\python -m pytest tests\test_config.py tests\test_conversation_progress_cognition.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py -q
```

Expected: pass.

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py tests\test_multi_source_cognition_stage_02_chat_episode_migration.py -q
```

Expected: pass.

### Regression Tests

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py tests\test_service_background_consolidation.py -q
```

Expected: pass.

### Live LLM

No live LLM test is required for this plan because no prompt text or semantic
LLM contract changes are authorized. If an implementation agent changes prompt
text despite this plan, stop and update the plan before running live LLM tests.

## Independent Plan Review

Review completed on 2026-05-11 by Codex in same-session reviewer mode. No
separate reviewer was available before approval.

Inputs inspected:

- `development_plans/README.md`
- `.agents/skills/development-plan-writing/references/plan_contract.md`
- `.agents/skills/development-plan-writing/references/execution_gates.md`
- `.agents/skills/development-plan-writing/references/cutover_policy.md`
- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/state.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/brain_service/contracts.py`
- `src/kazusa_ai_chatbot/cognition_episode.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `src/kazusa_ai_chatbot/internal_thought_cognition.py`
- Focused existing tests listed in `Change Surface`

Review findings:

- Blockers: none.
- Required pre-approval edits applied: static grep verification now includes
  `visual_directives_enabled` so adapter and public request leakage is caught.
- Architecture alignment: the approved target keeps adapters thin, keeps the
  `/chat` control in service config, keeps self-cognition control as an
  internal function parameter, and gates the L3 visual-agent LLM call through a
  deterministic internal state flag.
- Contract alignment: the visual agent can run only when
  `COGNITION_VISUAL_DIRECTIVES_ENABLED` and `visual_directives_enabled` are
  both true. Either false value disables the call and preserves the existing
  empty-list visual-directive output shape.
- Approval status: approved for implementation.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, public API leakage, adapter leakage, prompt/RAG/context
  leaks, persistence risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including focused tests, regression tests,
  static checks, execution evidence, and lifecycle notes.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only fixture/documentation
corrections. If a fix would cross the approved boundary or alter the contract,
stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `COGNITION_VISUAL_DIRECTIVES_ENABLED` defaults to true and preserves current
  `/chat` visual-directive behavior because `/chat` uses implicit
  `visual_directives_enabled=True`.
- Setting `COGNITION_VISUAL_DIRECTIVES_ENABLED=false` causes `/chat` to skip
  the L3 visual-agent LLM call and return empty visual-directive lists inside
  `action_directives.visual_directives`.
- Self-cognition/internal-thought can pass
  `visual_directives_enabled=False` to disable visual directives for that run.
- Self-cognition/internal-thought runs the visual agent only when
  `COGNITION_VISUAL_DIRECTIVES_ENABLED` and `visual_directives_enabled` are both
  true.
- Internal-thought audit prompt keys accurately omit the visual-agent key when
  visual directives are disabled.
- `ChatRequest`, `DebugModesIn`, and all adapters expose no new visual-control
  field.
- All verification commands pass and independent code review is recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Adapter API leakage | Forbid contract and adapter changes | Static grep against `src/adapters` and `brain_service/contracts.py` |
| Prompt churn in CJK-heavy file | Add only early skip logic; require `cjk-safety` and py_compile | CJK file py_compile |
| Audit claims visual prompt ran when skipped | Adjust internal-thought prompt-key list based on the flag | Stage 08 audit test |
| L4 collector receives missing fields | Return empty lists for every existing visual field | Visual-agent output contract test |
| Hidden `/chat` behavior change by default | Config defaults to enabled and `/chat` uses implicit `visual_directives_enabled=True` | Service regression tests |

## Execution Evidence

- Status: approved on 2026-05-11 after independent plan review.
- Execution start: 2026-05-11 by Codex. Plan status changed to `in_progress`;
  registry row changed to `in_progress | in_progress`.
- Stage 1 red tests:
  - `venv\Scripts\python -m pytest tests\test_config.py tests\test_conversation_progress_cognition.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py -q`
    failed as expected for the new contract: missing
    `COGNITION_VISUAL_DIRECTIVES_ENABLED`, missing early visual-agent skip, and
    missing `visual_directives_enabled` parameters/global flag in
    `internal_thought_cognition.py`.
  - The same batch also exposed unrelated existing prompt/fingerprint
    failures in `test_content_anchor_prompt_requires_fact_based_answers_without_case_example`
    and `test_text_chat_and_reflection_prompt_fingerprints_remain_stable`.
    Those are outside this plan's change surface and were not modified.
  - `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py tests\test_multi_source_cognition_stage_02_chat_episode_migration.py -q`
    failed as expected for the new contract: `service.py` has no
    `COGNITION_VISUAL_DIRECTIVES_ENABLED` attribute yet.
- Stage 2 verification:
  - `venv\Scripts\python -m pytest tests\test_config.py::TestCognitionVisualDirectivesConfig tests\test_multi_source_cognition_stage_00_regression_baseline.py::test_service_config_disabled_visual_directives_handoff tests\test_multi_source_cognition_stage_02_chat_episode_migration.py::test_service_adds_internal_visual_flag_when_config_disables_it -q`
    passed, 4 passed.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\state.py`
    passed with exit code 0.
- Stage 3 verification:
  - `venv\Scripts\python -m pytest tests\test_conversation_progress_cognition.py::test_visual_agent_skip_returns_empty_directives_without_llm -q`
    passed, 1 passed.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py`
    passed with exit code 0.
- Stage 4 verification:
  - `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_internal_thought_builder_can_disable_visual_directives tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_internal_thought_builder_obeys_global_visual_config tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_dry_run_visual_disable_omits_prompt_key_and_sets_state_flag tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_dry_run_global_visual_config_disables_prompt_key -q`
    passed, 4 passed.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\internal_thought_cognition.py`
    passed with exit code 0.
- Stage 5 verification attempted; not signed off because unrelated existing
  failures remain outside this plan's change surface:
  - `rg -n "no_visual_directives|COGNITION_VISUAL_DIRECTIVES_ENABLED|visual_directives_enabled" src/adapters src/kazusa_ai_chatbot/brain_service/contracts.py`
    returned no matches. Exit code 1 is expected for no matches.
  - `rg -n "no_visual_directives|COGNITION_VISUAL_DIRECTIVES_ENABLED|visual_directives_enabled" src\kazusa_ai_chatbot docs tests`
    returned matches only in the approved source, docs, and test files.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\internal_thought_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py`
    passed with exit code 0.
  - `git diff --check -- src/kazusa_ai_chatbot/config.py src/kazusa_ai_chatbot/state.py src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/internal_thought_cognition.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py docs/HOWTO.md tests/test_config.py tests/test_conversation_progress_cognition.py tests/test_multi_source_cognition_stage_00_regression_baseline.py tests/test_multi_source_cognition_stage_02_chat_episode_migration.py tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
    passed with exit code 0.
  - Direct plan-contract test command passed, 9 passed:
    `venv\Scripts\python -m pytest tests\test_config.py::TestCognitionVisualDirectivesConfig tests\test_conversation_progress_cognition.py::test_visual_agent_skip_returns_empty_directives_without_llm tests\test_multi_source_cognition_stage_00_regression_baseline.py::test_service_config_disabled_visual_directives_handoff tests\test_multi_source_cognition_stage_02_chat_episode_migration.py::test_service_adds_internal_visual_flag_when_config_disables_it tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_internal_thought_builder_can_disable_visual_directives tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_internal_thought_builder_obeys_global_visual_config tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_dry_run_visual_disable_omits_prompt_key_and_sets_state_flag tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_dry_run_global_visual_config_disables_prompt_key -q`.
  - Service focused command passed, 18 passed:
    `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py tests\test_multi_source_cognition_stage_02_chat_episode_migration.py -q`.
  - Full focused command failed only on unrelated existing prompt/fingerprint
    tests: `test_content_anchor_prompt_requires_fact_based_answers_without_case_example`
    and `test_text_chat_and_reflection_prompt_fingerprints_remain_stable`.
    All plan-owned visual-directive tests in that command passed.
  - Regression command failed on unrelated existing
    `test_hydrate_reply_context_keeps_adapter_supplied_metadata`. The command
    otherwise passed 69 tests and emitted the existing unawaited-coroutine
    warning from `service.py:232`.
  - Formal Stage 6 independent code review is not signed off because the
    Stage 5 verification gate is still blocked by those unrelated failures.
- Blocker RCA and fixture/baseline alignment:
  - Loaded the corresponding completed plans for the three blocking failures:
    `l3_content_anchor_open_loop_resolution_plan.md`,
    `multi_source_cognition_architecture_stage_08_internal_thought_cognition_dry_run_plan.md`,
    `character_self_words_retrieval_delivery_receipt_plan.md`,
    `multi_source_cognition_architecture_stage_09_multimodal_cognitive_input_sources_plan.md`,
    and `first_class_image_input_cognition_plan.md`.
  - Root cause for
    `test_content_anchor_prompt_requires_fact_based_answers_without_case_example`:
    the test pinned stale wording after later approved prompt refactors. The
    fact-grounding contract remained present as
    `最高优先级检索事实摘要`. Updated the deterministic assertion.
  - Root cause for
    `test_text_chat_and_reflection_prompt_fingerprints_remain_stable`: Stage 08
    prompt fingerprint baselines were stale after later approved L1/L2/L3
    prompt refactors. Refreshed all nine current byte lengths and SHA-256
    digests.
  - Root cause for
    `test_hydrate_reply_context_keeps_adapter_supplied_metadata`: the
    delivery-receipt test asserted no DB lookup, but the later first-class image
    plan intentionally uses exact reply-row lookup to hydrate quoted
    attachments. Updated the test to prove adapter-supplied text metadata
    remains authoritative while stored reply attachments are hydrated.
  - `venv\Scripts\python -m py_compile tests\test_conversation_progress_cognition.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py tests\test_service_background_consolidation.py`
    passed with exit code 0.
  - `venv\Scripts\python -m pytest tests\test_conversation_progress_cognition.py::test_content_anchor_prompt_requires_fact_based_answers_without_case_example tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_text_chat_and_reflection_prompt_fingerprints_remain_stable tests\test_service_background_consolidation.py::test_hydrate_reply_context_keeps_adapter_supplied_metadata -q`
    passed, 3 passed.
- Stage 5 verification rerun:
  - `rg -n "no_visual_directives|COGNITION_VISUAL_DIRECTIVES_ENABLED|visual_directives_enabled" src/adapters src/kazusa_ai_chatbot/brain_service/contracts.py`
    returned no matches. Exit code 1 is expected for no matches.
  - `rg -n "no_visual_directives|COGNITION_VISUAL_DIRECTIVES_ENABLED|visual_directives_enabled" src\kazusa_ai_chatbot docs tests`
    returned matches only in the approved source, docs, and test files.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\internal_thought_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_conversation_progress_cognition.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py tests\test_service_background_consolidation.py`
    passed with exit code 0.
  - `git diff --check -- src/kazusa_ai_chatbot/config.py src/kazusa_ai_chatbot/state.py src/kazusa_ai_chatbot/service.py src/kazusa_ai_chatbot/internal_thought_cognition.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py docs/HOWTO.md tests/test_config.py tests/test_conversation_progress_cognition.py tests/test_multi_source_cognition_stage_00_regression_baseline.py tests/test_multi_source_cognition_stage_02_chat_episode_migration.py tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py tests/test_service_background_consolidation.py`
    passed with exit code 0.
  - `venv\Scripts\python -m pytest tests\test_config.py tests\test_conversation_progress_cognition.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py -q`
    passed, 55 passed.
  - `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py tests\test_multi_source_cognition_stage_02_chat_episode_migration.py -q`
    passed, 18 passed.
  - `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py tests\test_service_background_consolidation.py -q`
    passed, 70 passed.
- Stage 6 independent code review:
  - No separate reviewer was available; Codex reran the review from a
    fresh-review posture after all verification commands passed.
  - Scope review passed: visual-directive implementation remains confined to
    config, internal debug-mode state, service handoff, L3 visual-agent gating,
    internal-thought controls, docs, and focused tests. No adapter or public
    `/chat` request-schema field was added.
  - Contract review passed: the effective visual-agent enable condition is
    `COGNITION_VISUAL_DIRECTIVES_ENABLED and visual_directives_enabled`; either
    false value seeds `no_visual_directives` and skips the visual LLM call.
  - Output review passed: skipped visual generation returns all four existing
    visual fields as empty lists and validates the existing L3 visual contract.
  - Regression review passed: the three stale cross-plan test failures were
    resolved by aligning tests to approved later contracts, not by weakening
    production behavior.
  - Approval status: approved. No residual blockers remain for this plan.
- Lifecycle cleanup:
  - Moved the completed plan from `active/short_term/` to
    `archive/completed/short_term/`.
  - Removed the completed plan from the Active Short-Term Plans registry table.
  - Added the completed plan to the Completed Short-Term Records registry
    table.

## Plan Self-Review

- Coverage: each `Must Do` item maps to `Implementation Order`, `Progress
  Checklist`, and `Verification`.
- Placeholder scan: no placeholders or unresolved implementation choices are
  intentionally present.
- Contract consistency: the internal flag name is consistently
  `no_visual_directives`; the config name is consistently
  `COGNITION_VISUAL_DIRECTIVES_ENABLED`; the internal self-cognition parameter
  is consistently `visual_directives_enabled`.
- Granularity: every stage names files and verification evidence.
- Verification: default behavior, disabled `/chat`, disabled internal-thought,
  adapter non-exposure, static compile, and regression coverage are included.
