# reflection flag simplification plan

## Summary

- Goal: Replace the reflection runtime controls with one positive worker flag and
  remove the separate prompt-facing reflection context flag.
- Plan class: large
- Status: draft
- Mandatory skills: `py-style`, `test-style-and-execution`,
  `local-llm-architecture`, and `development-plan-writing`
- Overall cutover strategy: bigbang
- Highest-risk areas: accidentally exposing raw reflection output, changing
  service startup semantics, stale operator documentation, and tests preserving
  the removed flags through monkeypatches.
- Acceptance criteria: `REFLECTION_CYCLE_ENABLED` is the only reflection worker
  config switch, removed flags are absent from runtime code, docs, tests, and
  other active plans, promoted reflection context is always eligible through the
  existing promoted memory boundary, and focused tests plus static greps pass.

## Context

Current reflection controls are split:

- `REFLECTION_CYCLE_DISABLED=false` starts the background reflection worker from
  FastAPI lifespan.
- `REFLECTION_CONTEXT_ENABLED=false` prevents
  `reflection_cycle.context.build_promoted_reflection_context()` from reading
  already-promoted reflection memory for normal chat.

The requested target is to align reflection config with the project's other
positive `*_ENABLED` controls and remove the redundant context gate. The safety
boundary for prompt-facing reflection context is not the removed flag. The
real boundary is:

```text
raw hourly/daily reflection runs
-> global promotion policy
-> active memory rows with source_kind="reflection_inferred"
-> bounded prompt projection
-> L2 soft global background only
```

Older plan references:

- `development_plans/archive/completed/short_term/reflection_memory_integration_stage1c_plan.md`
  introduced both `REFLECTION_CYCLE_DISABLED` and
  `REFLECTION_CONTEXT_ENABLED`. It is historical and must not be edited for new
  scope.
- `development_plans/active/short_term/reflection_driven_character_state_evolution_plan.md`
  is the older draft referenced by the user and is deleted in the current
  working tree. Use the version in Git history only as reference. Its durable
  architectural rule still applies: normal cognition may consume promoted
  reflection context, not raw reflection run documents.
- `development_plans/active/short_term/global_character_growth_from_reflection_plan.md`
  currently references `REFLECTION_CONTEXT_ENABLED`; this plan must update that
  active plan so future work does not reintroduce the removed flag.
  That plan is currently approved, so this plan authorizes only an
  administrative dependency update that replaces stale flag references with the
  new always-eligible promoted-reflection-context contract. Do not change that
  plan's status or implementation scope unless the required edit goes beyond
  stale flag vocabulary.

The current worktree contains unrelated edits. Implementation must preserve
those edits and avoid reverting deleted or modified files outside this plan's
change surface.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing reflection context exposure,
  service startup behavior, or cognition prompt-facing payload assumptions.
- `development-plan-writing`: load before changing this plan or lifecycle
  registry records.

## Mandatory Rules

- Check `git status --short` before editing and preserve unrelated user work.
- Use `venv\Scripts\python` for Python test commands.
- Do not read `.env`.
- Keep raw hourly reflection output, raw daily reflection output, reflection-run
  documents, transcripts, source message refs, user ids, and private details
  out of normal cognition prompts.
- Do not add a replacement context flag, compatibility alias, helper wrapper,
  dual-read fallback, or hidden environment variable for
  `REFLECTION_CONTEXT_ENABLED`.
- Do not preserve `REFLECTION_CYCLE_DISABLED` as a compatibility alias. The new
  worker switch is `REFLECTION_CYCLE_ENABLED`.
- Keep reflection context projection bounded to the existing lane caps unless a
  separate plan changes the cap.
- Do not change reflection prompt text, promotion validators, memory-evolution
  write semantics, scheduler semantics, or cognition L2 instructions unless
  explicitly listed in `Change Surface`.
- The edit to the approved
  `global_character_growth_from_reflection_plan.md` is limited to replacing
  stale `REFLECTION_CONTEXT_ENABLED` dependency text. If the active growth plan
  needs behavioral, schema, worker, prompt, or verification changes beyond that
  dependency update, stop and request reapproval for the growth plan before
  changing it.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Replace `REFLECTION_CYCLE_DISABLED` with
  `REFLECTION_CYCLE_ENABLED`, defaulting to `true`.
- Remove `REFLECTION_CONTEXT_ENABLED` from config, imports, docs, tests, and
  other active plans.
- Make `build_promoted_reflection_context()` always attempt the existing
  promoted-memory projection when called.
- Keep the service-level try/except fallback that converts reflection context
  load failures into `{}` without failing chat.
- Update active documentation and other active development plans to describe the
  new single worker switch and always-eligible promoted context.
- Record an operational migration note in execution evidence and final handoff
  for deployments that currently set `REFLECTION_CYCLE_DISABLED=true`,
  replacing it with `REFLECTION_CYCLE_ENABLED=false` before rollout.
- Add or update focused tests before implementation and run the verification
  commands in this plan.

## Deferred

- Do not change reflection promotion prompts or lane decisions.
- Do not change the reflection worker schedule or busy-probe behavior.
- Do not add caching for promoted reflection context.
- Do not change L2/L3/dialog prompts.
- Do not edit completed archived plans except by referencing them from this
  plan or a future evidence record.
- Do not implement global character growth, self-cognition, or interaction
  style plan work as part of this flag cleanup.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Worker flag | bigbang | Replace `REFLECTION_CYCLE_DISABLED` with `REFLECTION_CYCLE_ENABLED`. Do not read both names. |
| Context flag | bigbang | Delete `REFLECTION_CONTEXT_ENABLED`. Do not add a replacement context flag. |
| Prompt-facing context | bigbang | Promoted reflection context is always eligible when the service calls the builder. Empty result still means no prompt-visible context. |
| Docs and active plans | bigbang | Update active docs and active plans to the new flag vocabulary in the same change. |
| Deployment environment | migration | Before deploying this code, replace any operator-managed `REFLECTION_CYCLE_DISABLED=true` setting with `REFLECTION_CYCLE_ENABLED=false` and remove the old variable. The code still performs a bigbang rename and must not read the old variable. |
| Archived plans | compatible | Keep archived completed plans unchanged as historical records; grep checks may allow archive-only matches. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- Bigbang areas must delete or rewrite legacy references instead of preserving
  them.
- Any change to this cutover policy requires user approval before
  implementation.

## Agent Autonomy Boundaries

- Target ownership boundary:
  `src/kazusa_ai_chatbot/config.py`,
  `src/kazusa_ai_chatbot/service.py`,
  `src/kazusa_ai_chatbot/reflection_cycle/context.py`,
  reflection docs, active plan references, and focused tests.
- The agent may choose exact test helper names only when the tests prove the
  contracts in this plan.
- The agent must not introduce alternate feature flags, compatibility shims,
  fallback environment names, or unrelated config normalization helpers.
- If existing user edits touch the same files, inspect them and work with them.
  Do not revert unrelated changes.
- If code and plan disagree, preserve the plan's stated intent and report the
  discrepancy in `Execution Evidence`.

## Target State

Runtime config has one reflection worker switch:

```env
REFLECTION_CYCLE_ENABLED=true
```

When `REFLECTION_CYCLE_ENABLED=true`, service lifespan starts the reflection
worker. When it is `false`, service lifespan skips the worker and logs that the
worker is disabled via `REFLECTION_CYCLE_ENABLED=false`.

Prompt-facing promoted reflection context has no separate config gate. Normal
chat still receives `{}` when no active promoted rows exist or when the context
load fails. The only prompt-visible rows come from the existing promoted memory
projection.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Worker config name | Use `REFLECTION_CYCLE_ENABLED=true`. | Matches the project's positive enabled-flag style and avoids double negatives. |
| Compatibility | Do not keep `REFLECTION_CYCLE_DISABLED` or `REFLECTION_CONTEXT_ENABLED` aliases. | The user requested cleanup, and dual reads would keep the old contract alive. |
| Context exposure | Remove the context flag entirely. | Promoted memory is already gated by the reflection promotion and memory-evolution boundary. |
| Startup default | Keep reflection worker default-on. | Preserves current default behavior where `REFLECTION_CYCLE_DISABLED=false` starts the worker. |
| Failure mode | Keep chat resilient on context load errors. | Existing service behavior prevents memory-read failures from breaking live chat. |
| Active growth plan | Rewrite references to the removed context flag. | Future active work must not depend on a removed config switch. |
| Approved-plan lifecycle | Treat the growth-plan edit as an administrative dependency update. | The approved growth plan should not be returned to draft for stale flag vocabulary, but new behavior changes to it require reapproval. |
| Operator migration | Document the old-to-new env variable replacement explicitly. | Without an alias, old `REFLECTION_CYCLE_DISABLED=true` would be ignored by new code and the worker would start by default. |

## Contracts And Data Shapes

### Config Contract

`src/kazusa_ai_chatbot/config.py` must expose:

```python
REFLECTION_CYCLE_ENABLED = os.getenv(
    "REFLECTION_CYCLE_ENABLED",
    "true",
).lower() in ("1", "true", "yes")
```

It must not expose `REFLECTION_CYCLE_DISABLED` or
`REFLECTION_CONTEXT_ENABLED`.

### Service Startup Contract

`src/kazusa_ai_chatbot/service.py` must import
`REFLECTION_CYCLE_ENABLED` and use this control flow:

```python
if REFLECTION_CYCLE_ENABLED:
    _reflection_worker_handle = start_reflection_cycle_worker(
        is_primary_interaction_busy=lambda: False,
    )
else:
    logger.info(
        "Reflection cycle worker disabled via REFLECTION_CYCLE_ENABLED=false"
    )
```

### Promoted Reflection Context Contract

`src/kazusa_ai_chatbot/reflection_cycle/context.py` must expose:

```python
async def build_promoted_reflection_context(
    *,
    limit_per_lane: int = 3,
) -> PromotedReflectionContext: ...
```

The function must query only the existing approved lanes:

```python
{
    "source_kind": MemorySourceKind.REFLECTION_INFERRED,
    "source_global_user_id": "",
}
```

The two allowed `memory_type` values are
`PROMOTION_LANE_MEMORY_TYPE["lore"]` and
`PROMOTION_LANE_MEMORY_TYPE["self_guidance"]`.

The returned shape remains:

```python
{
    "promoted_lore": list[dict],
    "promoted_self_guidance": list[dict],
    "source_dates": list[str],
    "retrieval_notes": list[str],
}
```

Return `{}` when both lanes are empty.

## LLM Call And Context Budget

This plan adds no LLM calls and changes no prompt text.

Response path before:

- `REFLECTION_CONTEXT_ENABLED=false`: 0 promoted-reflection memory reads.
- `REFLECTION_CONTEXT_ENABLED=true`: 2 bounded metadata memory reads, 0 LLM
  calls, max 3 lore rows plus 3 self-guidance rows.

Response path after:

- 2 bounded metadata memory reads whenever the service calls
  `build_promoted_reflection_context()`.
- 0 new LLM calls.
- Same prompt cap: max 3 lore rows plus 3 self-guidance rows.
- On memory-read failure, service logs and uses `{}`.

The implementation must not increase lane caps or add response-path LLM calls.

## Operational Steps

This code change intentionally does not read the old
`REFLECTION_CYCLE_DISABLED` environment variable. Operators must migrate
deployment configuration before rolling out the code:

1. If a deployment currently sets `REFLECTION_CYCLE_DISABLED=true`, the
   rollout handoff must instruct the operator to replace it with
   `REFLECTION_CYCLE_ENABLED=false`.
2. If a deployment currently sets `REFLECTION_CYCLE_DISABLED=false`, the
   rollout handoff must instruct the operator to remove the old variable. The
   new default `REFLECTION_CYCLE_ENABLED=true` preserves the same worker-on
   behavior.
3. Remove `REFLECTION_CONTEXT_ENABLED` from deployment configuration. There is
   no replacement variable.
4. Do not inspect or edit local `.env` during implementation. Operators own
   environment migration outside this plan's repository edits.
5. Keep long-term docs free of removed flag names after the migration note has
   been recorded in execution evidence and final handoff.

Repository deployment templates are not currently listed in the planning scan
as containing the removed reflection flags. If verification finds a removed
flag in a repository deployment template, stop and update this plan's change
surface before editing that template.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/config.py`
  - Replace the negative worker flag with `REFLECTION_CYCLE_ENABLED`.
  - Delete `REFLECTION_CONTEXT_ENABLED`.
- `src/kazusa_ai_chatbot/service.py`
  - Import `REFLECTION_CYCLE_ENABLED`.
  - Start the reflection worker only when the new flag is true.
  - Update the operator log message.
- `src/kazusa_ai_chatbot/reflection_cycle/context.py`
  - Remove the config import.
  - Remove the `enabled` parameter and disabled early return.
  - Keep lane projection and empty-result behavior.
- `docs/HOWTO.md`
  - Replace the env example and startup note with `REFLECTION_CYCLE_ENABLED`.
  - Remove `REFLECTION_CONTEXT_ENABLED`.
- `README.md`
  - Reword reflection context language so it says promoted context is eligible
    through the promoted-memory boundary, not a separate config gate.
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
  - Replace worker flag documentation.
  - Remove the context flag from Feature Flags.
  - State that prompt-facing context is compact promoted memory only.
- `development_plans/active/short_term/global_character_growth_from_reflection_plan.md`
  - Replace all `REFLECTION_CONTEXT_ENABLED` dependency text with always-eligible
    promoted reflection context language.
  - Do not change status, scope, schema, worker behavior, prompt behavior, or
    verification beyond stale flag dependency wording.
  - Keep `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED` unchanged.
- `development_plans/README.md`
  - Add this draft plan to the active short-term registry.
- `tests/test_reflection_cycle_stage1c_reflection_context.py`
  - Remove disabled-context tests.
  - Add or keep tests proving empty lanes return `{}` and approved lanes are
    projected.
- `tests/test_reflection_cycle_stage1c_service.py`
  - Rename helper arguments from disabled to enabled.
  - Verify default/true starts the worker and false skips it.
- `tests/test_reflection_cycle_stage1c_integration.py`
  - Update static documentation expectations to `REFLECTION_CYCLE_ENABLED=true`.
- `tests/test_config.py`
  - Add config import/subprocess checks for `REFLECTION_CYCLE_ENABLED` default
    true and false parsing.
  - Add a subprocess check proving `REFLECTION_CYCLE_DISABLED=true` no longer
    disables the worker flag; `REFLECTION_CYCLE_ENABLED` remains true unless
    explicitly set false.
  - Add removed-symbol assertions for `REFLECTION_CYCLE_DISABLED` and
    `REFLECTION_CONTEXT_ENABLED`.

### Keep

- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
  - Promotion lanes and validators remain unchanged.
- `src/kazusa_ai_chatbot/memory_evolution/`
  - Memory lifecycle and retrieval contracts remain unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
  - Prompt instruction that promoted reflection context is soft global
    background remains unchanged.
- `development_plans/archive/completed/short_term/reflection_memory_integration_stage1c_plan.md`
  - Historical record only. Do not edit.

## Implementation Order

1. Update focused tests for the new config and context contracts.
   - File: `tests/test_config.py`
   - Add tests proving `REFLECTION_CYCLE_ENABLED` defaults to true and parses
     false.
   - Add tests proving removed symbols are absent.
   - File: `tests/test_reflection_cycle_stage1c_reflection_context.py`
   - Replace disabled-context behavior with empty-lane behavior.
   - Expected before implementation: tests fail because old symbols and
     function signatures still exist.
2. Update service lifecycle tests.
   - File: `tests/test_reflection_cycle_stage1c_service.py`
   - Rename `_run_lifespan(..., disabled=...)` to positive enabled semantics.
   - Expected before implementation: tests fail because service still imports
     and monkeypatches `REFLECTION_CYCLE_DISABLED`.
3. Implement config and context changes.
   - Modify `config.py` and `reflection_cycle/context.py`.
   - Run the focused config/context tests.
4. Implement service wiring changes.
   - Modify `service.py`.
   - Run service lifecycle tests.
5. Update docs, active plan references, and operational migration handoff.
   - Modify active docs and
     `global_character_growth_from_reflection_plan.md`.
   - Record old-to-new environment migration instructions in
     `Execution Evidence` and final handoff. Do not leave removed flag names in
     active docs.
   - Update registry row for this plan.
6. Run full verification gates and record evidence.
7. Run independent code review, remediate findings, and rerun affected checks.

## Progress Checklist

- [ ] Stage 1 - tests express new contracts
  - Covers: implementation steps 1 and 2.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_config.py tests\test_reflection_cycle_stage1c_reflection_context.py tests\test_reflection_cycle_stage1c_service.py -q`
  - Expected before implementation: failing tests identify old flag contracts.
  - Evidence: record failures or baseline result in `Execution Evidence`.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 2 - config, context, and service code updated
  - Covers: implementation steps 3 and 4.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_config.py tests\test_reflection_cycle_stage1c_reflection_context.py tests\test_reflection_cycle_stage1c_service.py -q`
  - Evidence: focused tests pass and changed files are listed.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 3 - docs and active plans updated
  - Covers: implementation step 5.
  - Verify:
    `rg "REFLECTION_CONTEXT_ENABLED|REFLECTION_CYCLE_DISABLED" README.md docs src tests development_plans/README.md development_plans/active -g "!development_plans/active/short_term/reflection_flag_simplification_plan.md"`
    returns no matches.
  - Evidence: static grep result recorded. This plan's own historical/context
    references and archive matches are not part of this check.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 4 - full verification complete
  - Covers: implementation step 6.
  - Verify all commands in `Verification`.
  - Evidence: command outputs and any allowed exceptions recorded.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 5 - independent code review complete
  - Covers: implementation step 7 and `Independent Code Review`.
  - Verify: review findings resolved or explicitly recorded as residual risk.
  - Evidence: review mode, findings, fixes, rerun commands, and approval status
    recorded.
  - Sign-off: `<agent/date>` after review evidence is recorded.

## Verification

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_config.py -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_reflection_context.py -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_service.py -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_integration.py -q`

### Regression Tests

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_02_chat_episode_migration.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py tests\test_persona_supervisor2_rag2_integration.py tests\test_rag_cognitive_episode_adapter.py -q`

### Static Greps

- `rg "REFLECTION_CONTEXT_ENABLED|REFLECTION_CYCLE_DISABLED" README.md docs src tests development_plans/README.md development_plans/active -g "!development_plans/active/short_term/reflection_flag_simplification_plan.md"`
  must return no matches. Exit code 1 is the expected successful no-match
  result for `rg`. This plan is excluded because it intentionally records the
  old flags as historical context and migration instructions.
- `rg "REFLECTION_CONTEXT_ENABLED|REFLECTION_CYCLE_DISABLED" development_plans/archive`
  may return historical matches only. Do not edit completed archived plans to
  force this grep to zero.
- `rg "REFLECTION_CYCLE_ENABLED" src tests docs README.md development_plans/active`
  must show current config, service, test, docs, and active-plan references.
- PowerShell deployment-template check:
  `$paths = @("Dockerfile", "docker-compose.yml", ".github") | Where-Object { Test-Path -LiteralPath $_ }; if ($paths.Count -gt 0) { rg "REFLECTION_CYCLE_DISABLED|REFLECTION_CONTEXT_ENABLED" $paths }`
  must return no matches for existing paths. Exit code 1 from `rg` is the
  expected successful no-match result. If no listed paths exist, record that as
  an allowed path-absent result in `Execution Evidence`.

### Operational Migration Check

- Confirm `Execution Evidence` and final handoff record that
  `REFLECTION_CYCLE_DISABLED=true` becomes
  `REFLECTION_CYCLE_ENABLED=false`, `REFLECTION_CYCLE_DISABLED=false` is removed,
  and `REFLECTION_CONTEXT_ENABLED` is removed with no replacement.
- Confirm `tests/test_config.py` includes a subprocess case where
  `REFLECTION_CYCLE_DISABLED=true` alone does not make
  `REFLECTION_CYCLE_ENABLED` false.

### Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\reflection_cycle\context.py`

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not draft the plan. If no separate reviewer is available, the active agent
must reread the completed Stage 1c reflection plan, the older
reflection-driven character-state draft from Git history when needed, this
plan, and relevant source/test context from a fresh-review posture.

Review scope:

- Previous reflection artifacts are named and carried forward without editing
  completed archived plans.
- The proposed scope aligns with the promoted-only reflection context boundary
  and does not expose raw reflection runs to live cognition.
- The approved growth-plan edit is limited to stale flag dependency wording.
- Verification commands can pass without matching this plan's own historical
  references to removed flags.
- The operational migration instructions make the breaking env rename explicit.
- The implementation agent has exact file paths, contracts, progress
  checkpoints, verification commands, and evidence requirements.

Record blockers, non-blocking findings, required edits, and approval status.
Approve only when blockers are resolved.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test,
  documentation, and plan artifact.
- Alignment with `Must Do`, `Deferred`, `Cutover Policy`, `Change Surface`,
  exact contracts, implementation order, verification gates, and acceptance
  criteria.
- Code quality and design risk, especially hidden compatibility aliases,
  prompt/context leakage, service startup behavior, and brittle test
  monkeypatches.
- Regression and handoff quality, including static grep expectations and
  archive-only historical references.

Fix concrete findings directly only when the fix is inside this plan's change
surface. If a finding requires a new flag, alternate compatibility behavior, or
changes outside the approved boundary, stop and request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `REFLECTION_CYCLE_ENABLED` is the only reflection cycle worker config switch
  in active source, tests, docs, and active plans other than this plan's
  historical references.
- `REFLECTION_CONTEXT_ENABLED` is absent from active source, tests, docs, and
  active plans other than this plan's historical references.
- `build_promoted_reflection_context()` no longer accepts or reads an enabled
  flag and still returns `{}` when no promoted rows exist.
- Service startup starts the worker by default and skips it only when
  `REFLECTION_CYCLE_ENABLED=false`.
- Execution evidence and final handoff describe how to migrate old environment
  settings before rollout.
- No raw reflection output becomes prompt-visible.
- Focused tests, regression tests, compile checks, static greps, and
  independent code review pass.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Old disable env var silently no longer works | Operational migration instructions require replacing `REFLECTION_CYCLE_DISABLED=true` with `REFLECTION_CYCLE_ENABLED=false`, and config tests prove the old var is ignored | Static greps, docs review, and config subprocess test |
| Prompt receives raw reflection output | Keep projection through active reflection-promoted memory only | Reflection context tests and code review |
| Live chat latency rises from always reading promoted memory | Keep 2 metadata reads, no LLM calls, and existing exception fallback | LLM/context budget and focused context tests |
| Active growth plan reintroduces removed flag | Update active plan references in this change without changing its approved implementation scope | Active-plan static grep and plan review |
| Tests monkeypatch old flags and hide drift | Rewrite tests to positive flag semantics | Service and config tests |

## Execution Evidence

- 2026-05-11 independent plan review found four blockers: impossible active-plan
  grep due to this plan's own historical references, approved growth-plan
  lifecycle ambiguity, missing operational migration gate for the breaking env
  rename, and an invalid Python lane snippet. This revision fixes those items
  by excluding this plan from removed-flag greps, constraining the growth-plan
  edit to administrative dependency wording, adding operational migration
  instructions plus a config test requirement, and replacing the invalid lane
  snippet with explicit allowed memory types.
- Implementation pending. This plan is a draft and has not been executed.
