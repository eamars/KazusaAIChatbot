# consolidation module boundary migration bugfix plan

## Summary

- Goal: Move consolidation-owned helper implementation modules from
  `kazusa_ai_chatbot.nodes` into `kazusa_ai_chatbot.consolidation` without
  changing prompts, graph behavior, persistence behavior, or LLM call count.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: bigbang module-boundary migration with no
  compatibility wrappers under `nodes`.
- Highest-risk areas: prompt text drift in CJK Python files, hidden import
  fallback paths, graph handoff state drift, and persistence/cache regression.
- Acceptance criteria: runtime and tests import consolidation helpers from
  `kazusa_ai_chatbot.consolidation`; old consolidator helper modules no longer
  exist under `nodes`; focused regression tests pass; review confirms prompt
  bodies and behavior are unchanged except import paths.

## Context

The public consolidation entrypoint already lives at
`kazusa_ai_chatbot.consolidation.core.call_consolidation_subgraph`. Runtime
callers in `service.py`, `nodes/persona_supervisor2.py`, and
`self_cognition/runner.py` use that entrypoint.

The remaining boundary defect is implementation ownership:
`consolidation/core.py` still assembles the primary consolidation graph by
importing helper modules from
`kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_*`. The previous
completed target-routing plan intentionally stopped after moving the public
entrypoint and target routing. This follow-up bugfix completes the package
boundary so consolidation-owned extraction, origin, policy, schema, image,
memory-unit, and persistence modules live with the consolidation subsystem.

This is an equivalence migration. The behavior target is:

```text
same global_state input
  -> same ConsolidatorState shape
  -> same LangGraph topology
  -> same LLM prompts and model calls
  -> same parsed outputs
  -> same write-intent validation
  -> same DB writes and Cache2 invalidation
```

No data migration is required. No production cleanup is authorized.

## Mandatory Skills

- `development-plan-writing`: load before modifying this plan or moving it
  through lifecycle states.
- `local-llm-architecture`: load before changing graph ownership, prompt
  placement, LLM-stage code layout, or background consolidation behavior.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing or moving Python files that contain CJK
  prompt text.

## Mandatory Rules

- Do not change consolidation prompt text. If a prompt body must change for any
  reason other than import paths, stop and revise this plan before continuing.
- Do not add, remove, or reorder LLM calls.
- Do not change `call_consolidation_subgraph(...)` input or return shape.
- Do not change `ConsolidatorState`, `ConsolidationOriginMetadata`,
  `ConsolidationTargetPlan`, or write-policy semantics except import paths.
- Do not add compatibility wrappers, re-export modules, fallback imports, or
  dual import paths under `kazusa_ai_chatbot.nodes`.
- Do not change RAG, cognition, dialog, action-spec, dispatcher, scheduler,
  adapter, database schema, or Cache2 behavior.
- Move files with history-preserving commands such as `git mv` where possible.
  Use `apply_patch` for manual code edits.
- For CJK prompt files, prefer path/import edits only. Do not manually rewrite
  prompt literals. Run `py_compile` after each moved CJK prompt module.
- Use `venv\Scripts\python` for Python verification commands.
- Use PowerShell `-LiteralPath` for filesystem paths where applicable.
- Do not read `.env`.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Create first-class consolidation implementation modules:
  - `src/kazusa_ai_chatbot/consolidation/schema.py`
  - `src/kazusa_ai_chatbot/consolidation/origin.py`
  - `src/kazusa_ai_chatbot/consolidation/origin_policy.py`
  - `src/kazusa_ai_chatbot/consolidation/facts.py`
  - `src/kazusa_ai_chatbot/consolidation/reflection.py`
  - `src/kazusa_ai_chatbot/consolidation/images.py`
  - `src/kazusa_ai_chatbot/consolidation/memory_units.py`
  - `src/kazusa_ai_chatbot/consolidation/persistence.py`
- Update `src/kazusa_ai_chatbot/consolidation/core.py` to import only
  consolidation-package helper modules.
- Update all source and test imports from
  `kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_*` to the new
  `kazusa_ai_chatbot.consolidation.*` paths.
- Delete the old `nodes/persona_supervisor2_consolidator*.py` helper modules,
  including the retired `persona_supervisor2_consolidator.py` stub.
- Add a deterministic module-boundary test proving the new modules exist, old
  modules are absent, and `consolidation.core` has no old helper imports.
- Run baseline tests before the migration and the same focused regression tests
  after the migration.
- Update subsystem documentation and the development-plan registry.

## Deferred

- Do not change consolidation prompts for quality, wording, examples, schema
  names, or instruction policy.
- Do not split or merge consolidator stages.
- Do not introduce a generic consolidation plugin registry or helper loader.
- Do not add node-level compatibility wrappers.
- Do not migrate RAG supervisor modules in this plan.
- Do not run live DB cleanup or mutate production data.
- Do not add live LLM tests. Prompt text changes are not authorized in this
  plan, and live LLM testing cannot be used to justify prompt drift.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Consolidation helper modules | bigbang | Move implementation from `nodes/persona_supervisor2_consolidator_*` to `consolidation/*`. Delete old helper modules. |
| Public entrypoint | compatible | Keep `kazusa_ai_chatbot.consolidation.core.call_consolidation_subgraph` unchanged. |
| Runtime callers | compatible | Existing runtime callers keep importing the public entrypoint from `consolidation.core`. |
| Tests | bigbang | Rewrite direct helper imports to the new consolidation paths. |
| Prompt and LLM behavior | compatible | Preserve prompt bodies, model configs, call order, parser use, and validation behavior. |
| Documentation | bigbang | README and subsystem ICDs describe consolidation helpers as owned by `consolidation`, not `nodes`. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- Bigbang areas must delete or rewrite legacy references instead of preserving
  them.
- Compatible areas preserve only the explicit surfaces listed in this plan.
- Any change to cutover policy requires user approval before implementation.

## Overdesign Guardrail

- Actual problem: consolidation implementation modules still live under
  `nodes`, so the subsystem boundary contradicts the public entrypoint and ICD.
- Minimal change: move existing helper modules into
  `kazusa_ai_chatbot.consolidation`, update imports, delete old modules, and
  verify behavior-equivalent graph and persistence paths.
- Ownership boundaries: `nodes` owns live persona/cognition/dialog graph
  stages; `consolidation` owns post-turn extraction, origin projection,
  write-policy validation, target-aware persistence dispatch, and helper LLM
  calls; deterministic code owns target validation and cache invalidation.
- Rejected complexity: no compatibility wrappers, loader abstraction, module
  registry, prompt refactor, retry changes, state-shape changes, feature flag,
  DB migration, or live-behavior tuning.
- Evidence threshold: add a wrapper, registry, or compatibility path only after
  a separate approved plan identifies a real external consumer that cannot
  move to the first-class consolidation module path.

## Agent Autonomy Boundaries

- The agent must preserve existing import ordering where practical and make
  only mechanical import-path edits required by the file moves.
- The agent must not introduce new architecture, alternate migration
  strategies, fallback paths, compatibility layers, or extra features.
- The agent must treat any change outside `src/kazusa_ai_chatbot/consolidation`,
  old consolidator helper modules, tests, and listed docs as out of scope.
- The agent must not perform unrelated cleanup, formatting churn, dependency
  upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, stop and report the discrepancy. Do not expand
  scope or invent a substitute implementation.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

```text
persona or self-cognition state
  -> kazusa_ai_chatbot.consolidation.core.call_consolidation_subgraph(...)
  -> kazusa_ai_chatbot.consolidation.origin
  -> kazusa_ai_chatbot.consolidation.target
  -> kazusa_ai_chatbot.consolidation.reflection
  -> kazusa_ai_chatbot.consolidation.facts
  -> kazusa_ai_chatbot.consolidation.persistence
  -> kazusa_ai_chatbot.consolidation.memory_units
  -> kazusa_ai_chatbot.consolidation.images
  -> database helpers and Cache2 invalidation
```

`kazusa_ai_chatbot.nodes` no longer contains files named
`persona_supervisor2_consolidator*.py`.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Compatibility wrappers | Do not keep wrappers under `nodes`. | The goal is a real ownership cutover. Wrappers would preserve the wrong boundary. |
| Prompt handling | Preserve prompt literals exactly. | This migration must not change LLM quality or behavior. |
| Tests as consumers | Update tests to import new module paths. | Tests should enforce the new ownership boundary, not the old implementation location. |
| Public API | Keep `call_consolidation_subgraph` as the public runtime entrypoint. | Runtime callers are already clean and should not churn. |
| Plan class | Use `large`. | The implementation is a simple migration, but the plan is large by document length and verification detail. |

## LLM Call And Context Budget

Before:

- Consolidation runs the existing background helper LLM calls from helper
  modules under `nodes`.
- Prompt inputs, prompt bodies, context caps, parser behavior, and retry limits
  are unchanged from current code.

After:

- Consolidation runs the same helper LLM calls from modules under
  `consolidation`.
- LLM call count change: zero.
- Prompt text change: zero.
- Context input change: zero.
- Latency impact: import-path only; no response-path or background-path model
  latency change is authorized.

Verification:

- Focused tests cover graph handoff, origin threading, evaluator routing,
  persistence, and cache invalidation.
- Independent code review must inspect the diff and record that prompt bodies
  were not changed beyond import/module path edits.

## Change Surface

### Create

- `tests/test_consolidation_module_boundary.py`
  - New deterministic test for package ownership.

### Move

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py`
  -> `src/kazusa_ai_chatbot/consolidation/schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
  -> `src/kazusa_ai_chatbot/consolidation/origin.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
  -> `src/kazusa_ai_chatbot/consolidation/origin_policy.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`
  -> `src/kazusa_ai_chatbot/consolidation/facts.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
  -> `src/kazusa_ai_chatbot/consolidation/reflection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
  -> `src/kazusa_ai_chatbot/consolidation/images.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`
  -> `src/kazusa_ai_chatbot/consolidation/memory_units.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
  -> `src/kazusa_ai_chatbot/consolidation/persistence.py`

### Delete

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
  - Delete the retired stub after no import references remain.

### Modify

- `src/kazusa_ai_chatbot/consolidation/core.py`
  - Update helper imports to `kazusa_ai_chatbot.consolidation.*`.
- `src/kazusa_ai_chatbot/consolidation/__init__.py`
  - Export only stable public contracts. Do not export every helper module by
    default.
- `src/kazusa_ai_chatbot/consolidation/README.md`
  - Update primary implementation files and remove references to node-owned
    helper implementation.
- `src/kazusa_ai_chatbot/nodes/README.md`
  - Remove `Consolidation inputs` ownership from `nodes`; state that completed
    persona state is handed to `consolidation`.
- `README.md`
  - Update repository map wording so `nodes` no longer lists consolidation
    stages as implementation-owned.
- Consolidation-focused tests under `tests/`
  - Update imports to the new `kazusa_ai_chatbot.consolidation.*` module paths.
- `development_plans/README.md`
  - Register this active bugfix plan while it remains active.

### Keep

- `src/kazusa_ai_chatbot/consolidation/core.py`
  - Keep public function name, input, output, graph topology, and return shape.
- `src/kazusa_ai_chatbot/consolidation/target.py`
  - Keep deterministic target planning and validation behavior unchanged.
- `src/kazusa_ai_chatbot/consolidation/group_channel.py`
  - Keep group-channel persistence helper unchanged except import formatting if
    needed.
- Runtime callers in `service.py`, `nodes/persona_supervisor2.py`, and
  `self_cognition/runner.py`
  - No import change is required because they already use `consolidation.core`.

## Implementation Order

1. Record a clean worktree baseline with `git status --short`.
2. Run baseline static import grep:
   `rg -n "kazusa_ai_chatbot\.nodes\.persona_supervisor2_consolidator" src\kazusa_ai_chatbot tests`.
3. Run the baseline focused regression suite listed in `Verification`.
4. Add `tests/test_consolidation_module_boundary.py` with failing assertions
   for the new module boundary.
5. Run the new boundary test and record the expected failure.
6. Move consolidator helper files from `nodes` to `consolidation` with
   history-preserving moves.
7. Update internal imports inside moved modules and `consolidation/core.py`.
8. Delete the retired old node stub.
9. Update test imports to the new module paths.
10. Update README and subsystem ICD documents.
11. Update `development_plans/README.md`.
12. Run `py_compile` over every moved consolidation module.
13. Run static greps proving old helper imports and old helper files are gone.
14. Run the focused regression suite listed in `Verification`.
15. Run `git diff --check`.
16. Run the independent code review gate.
17. Fix in-scope review findings and rerun affected verification commands.
18. Record execution evidence and leave the plan in `in_progress` until owner
    approval to mark completed.

## Progress Checklist

- [x] Stage 1 - baseline captured
  - Covers: steps 1-3.
  - Verify: baseline grep and focused regression suite run before migration.
  - Evidence: record command outputs in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 2 - boundary test added
  - Covers: steps 4-5.
  - Verify: `venv\Scripts\python -m pytest tests\test_consolidation_module_boundary.py -q`
    fails before implementation because new modules do not exist or old modules
    still exist.
  - Evidence: record the failing assertion summary in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 3 - helper modules moved and imports updated
  - Covers: steps 6-9.
  - Verify: `venv\Scripts\python -m py_compile` over moved modules succeeds.
  - Evidence: record moved files and compile output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 4 - docs and registry updated
  - Covers: steps 10-11.
  - Verify: static greps show active docs describe consolidation ownership
    through `kazusa_ai_chatbot.consolidation`.
  - Evidence: record grep output and docs summary.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 5 - focused regression complete
  - Covers: steps 12-15.
  - Verify: all `Verification` commands pass.
  - Evidence: record command outputs and any accepted skipped live-LLM gate.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 6 - independent code review complete
  - Covers: steps 16-18.
  - Verify: review records no blocking findings, or all in-scope findings are
    fixed and affected verification commands are rerun.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, residual
    risks, and approval status.
  - Handoff: plan completed and archived after evidence was recorded.
  - Sign-off: `Codex/2026-05-22`.

## Verification

Run from repository root with the project virtual environment.

### Baseline Before Migration

- `git status --short`
  - Expected: record current dirty files. Do not revert user-owned changes.

- `rg -n "kazusa_ai_chatbot\.nodes\.persona_supervisor2_consolidator" src\kazusa_ai_chatbot tests`
  - Expected before migration: matches in `consolidation/core.py`, helper
    modules, and tests. Record the baseline set.

- `venv\Scripts\python -m pytest tests\test_consolidation_target_routing.py tests\test_consolidator_group_channel_branch.py tests\test_consolidation_origin_metadata.py tests\test_consolidator_origin_selection.py tests\test_consolidation_origin_policy.py tests\test_consolidator_origin_policy_db_writer.py tests\test_db_writer_cache2_invalidation.py tests\test_consolidator_efficiency.py tests\test_consolidator_facts_rag2.py tests\test_consolidator_reflection_prompts.py tests\test_consolidator_source_aware_payloads.py tests\test_consolidator_character_image.py tests\test_user_memory_units_rag_flow.py tests\test_service_background_consolidation.py -q`
  - Expected before migration: pass or record unrelated pre-existing failures
    before editing.

### New Boundary Test

Add `tests/test_consolidation_module_boundary.py` with tests equivalent to:

```python
"""Tests for first-class consolidation module ownership."""

from __future__ import annotations

import importlib
import importlib.util
import inspect

from kazusa_ai_chatbot.consolidation import core


def test_consolidation_helper_modules_live_in_consolidation_package() -> None:
    module_names = [
        "schema",
        "origin",
        "origin_policy",
        "facts",
        "reflection",
        "images",
        "memory_units",
        "persistence",
    ]

    for module_name in module_names:
        module = importlib.import_module(
            f"kazusa_ai_chatbot.consolidation.{module_name}"
        )
        assert module.__name__ == f"kazusa_ai_chatbot.consolidation.{module_name}"


def test_legacy_node_consolidator_modules_are_absent() -> None:
    legacy_names = [
        "persona_supervisor2_consolidator",
        "persona_supervisor2_consolidator_schema",
        "persona_supervisor2_consolidator_origin",
        "persona_supervisor2_consolidator_origin_policy",
        "persona_supervisor2_consolidator_facts",
        "persona_supervisor2_consolidator_reflection",
        "persona_supervisor2_consolidator_images",
        "persona_supervisor2_consolidator_memory_units",
        "persona_supervisor2_consolidator_persistence",
    ]

    for legacy_name in legacy_names:
        spec = importlib.util.find_spec(
            f"kazusa_ai_chatbot.nodes.{legacy_name}"
        )
        assert spec is None


def test_consolidation_core_imports_no_legacy_node_helpers() -> None:
    source = inspect.getsource(core)

    assert "kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator" not in source
```

Run:

- `venv\Scripts\python -m pytest tests\test_consolidation_module_boundary.py -q`
  - Expected before implementation: fail.
  - Expected after implementation: pass.

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\consolidation\core.py src\kazusa_ai_chatbot\consolidation\schema.py src\kazusa_ai_chatbot\consolidation\origin.py src\kazusa_ai_chatbot\consolidation\origin_policy.py src\kazusa_ai_chatbot\consolidation\facts.py src\kazusa_ai_chatbot\consolidation\reflection.py src\kazusa_ai_chatbot\consolidation\images.py src\kazusa_ai_chatbot\consolidation\memory_units.py src\kazusa_ai_chatbot\consolidation\persistence.py`

Expected: exit code 0.

### Static Greps

- `rg -n "kazusa_ai_chatbot\.nodes\.persona_supervisor2_consolidator" src\kazusa_ai_chatbot tests`
  - Expected after migration: no matches. Exit code 1 is acceptable.

- `rg -n "persona_supervisor2_consolidator.*\.py|persona_supervisor2_consolidator" src\kazusa_ai_chatbot\nodes tests README.md src\kazusa_ai_chatbot\consolidation\README.md src\kazusa_ai_chatbot\nodes\README.md`
  - Expected after migration: no active node-owned implementation references.
    Historical archive plans are outside this grep and intentionally unchanged.

- `rg -n "from kazusa_ai_chatbot\.consolidation\.(facts|reflection|memory_units|persistence|origin|origin_policy|schema|images)" tests src\kazusa_ai_chatbot`
  - Expected after migration: tests and consolidation internals import new
    module paths.

- `rg -n "CONSOLIDATION_LLM|_PROMPT|ainvoke" src\kazusa_ai_chatbot\consolidation\core.py src\kazusa_ai_chatbot\consolidation\target.py`
  - Expected: no matches in `core.py` or `target.py` that add new LLM work.
    LLM prompt and `ainvoke` matches remain only in helper modules that already
    owned those calls before the move.

- `git diff --check`
  - Expected: no whitespace errors. Existing line-ending warnings must be
    recorded if present.

### Focused Regression Tests

- `venv\Scripts\python -m pytest tests\test_consolidation_module_boundary.py tests\test_consolidation_target_routing.py tests\test_consolidator_group_channel_branch.py -q`

- `venv\Scripts\python -m pytest tests\test_consolidation_origin_metadata.py tests\test_consolidator_origin_selection.py tests\test_consolidation_origin_policy.py -q`

- `venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py tests\test_db_writer_cache2_invalidation.py tests\test_consolidator_efficiency.py -q`

- `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py tests\test_consolidator_reflection_prompts.py tests\test_consolidator_source_aware_payloads.py tests\test_consolidator_character_image.py -q`

- `venv\Scripts\python -m pytest tests\test_user_memory_units_rag_flow.py tests\test_service_background_consolidation.py -q`

Expected: all pass, or unrelated pre-existing failures are recorded before
implementation and confirmed unchanged.

### Live LLM Gate

No live LLM test is authorized or required because prompt bodies must remain
unchanged.

If any prompt body changes, stop execution and remove the prompt change. If the
prompt change cannot be removed, stop and request a new approved plan. Do not
run live LLM tests as a substitute for preserving prompt text exactly.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test,
  documentation, and command artifact.
- CJK prompt safety: prompt literals were not manually rewritten, and moved CJK
  prompt modules compile.
- Module-boundary correctness: no old consolidator helper implementation,
  wrapper, fallback import, or test import remains under `nodes`.
- Behavioral equivalence: graph topology, state shape, prompt bodies, parser
  calls, write validation, DB helper calls, and Cache2 invalidation are
  unchanged except module paths.
- Plan alignment: `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, implementation order, verification gates, and acceptance criteria.
- Regression quality: baseline and post-migration focused tests are recorded,
  and no live LLM gate was run because prompt bodies and LLM call behavior are
  unchanged.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only documentation or
test-import corrections. If a fix would cross the approved boundary or alter
the contract, stop and update the plan or request approval before changing
code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Independent Plan Review

Run this gate before approval or execution. Prefer a reviewer that did not
draft the plan. If no separate reviewer is available, the active agent must
reread this plan, the development-plan registry, README, HOWTO, consolidation
README, nodes README, and the completed consolidation target-routing plan from
a fresh-review posture.

Review scope:

- The plan completes the consolidation module boundary without changing
  consolidation behavior.
- The plan rejects node-level wrappers and fallback imports.
- The plan protects CJK prompt text and local-LLM behavior.
- The verification suite includes baseline, boundary, static, focused
  regression, and independent code-review gates.
- The plan gives enough exact paths, commands, and expected results for an
  implementation agent to execute without inventing scope.

Record blockers, non-blocking findings, required edits, and approval status.
Approve only when blockers are resolved.

### 2026-05-22 Review Result

- Reviewer mode: active agent fresh-review; no separate reviewer was available.
- Inputs reviewed: current git status, development-plan registry, repository
  README, HOWTO, consolidation README, nodes README, active source/test import
  grep, and the completed consolidation target-routing plan.
- Blocking findings: none after approval edits in this plan.
- Required approval edits applied: status changed to `approved`; stale plan
  class row corrected to `large`; live LLM fallback language tightened to
  prohibit prompt drift; agent autonomy language tightened so implementation is
  mechanical and non-discretionary.
- Approval status: approved for execution on 2026-05-22 as a simple
  module-boundary migration with no prompt change, no LLM call change, no
  wrapper, and no behavior change.

## Acceptance Criteria

This plan is complete when:

- Consolidation helper implementation modules live under
  `src/kazusa_ai_chatbot/consolidation/`.
- No source or test file imports
  `kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_*`.
- No `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator*.py` file
  remains.
- `consolidation.core.call_consolidation_subgraph(...)` keeps the same public
  input and output contract.
- The new module-boundary test passes.
- Baseline and post-migration focused regression tests are recorded.
- Static greps prove old active imports and node-owned helper references are
  gone.
- Documentation states that consolidation owns helper extraction, origin,
  policy, schema, image, memory-unit, and persistence implementation.
- Independent code review finds no blocking issues.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Prompt drift changes LLM behavior | Move files, change imports only, inspect diff for prompt body changes | Code review plus compile and focused tests |
| Hidden old import path remains | Delete old modules and add boundary test | Static grep and `find_spec` absence test |
| Graph handoff state changes | Keep `core.py` graph topology and `ConsolidatorState` unchanged | Origin metadata and efficiency tests |
| Persistence or cache invalidation regresses | Move persistence code without semantic edits | DB writer and Cache2 invalidation tests |
| Tests keep enforcing old boundary | Rewrite test imports and add no-old-module assertion | Module-boundary test and static grep |

## Execution Evidence

### 2026-05-22 Stage 1 Baseline

- `git status --short`: existing plan approval edits were present before
  migration (`development_plans/README.md` modified and this plan untracked).
- `rg -n "kazusa_ai_chatbot\.nodes\.persona_supervisor2_consolidator" src\kazusa_ai_chatbot tests`:
  baseline matches were present in `consolidation/core.py`, old helper modules,
  and consolidation-focused tests.
- `venv\Scripts\python -m pytest tests\test_consolidation_target_routing.py tests\test_consolidator_group_channel_branch.py tests\test_consolidation_origin_metadata.py tests\test_consolidator_origin_selection.py tests\test_consolidation_origin_policy.py tests\test_consolidator_origin_policy_db_writer.py tests\test_db_writer_cache2_invalidation.py tests\test_consolidator_efficiency.py tests\test_consolidator_facts_rag2.py tests\test_consolidator_reflection_prompts.py tests\test_consolidator_source_aware_payloads.py tests\test_consolidator_character_image.py tests\test_user_memory_units_rag_flow.py tests\test_service_background_consolidation.py -q`:
  100 passed in 2.82s.

### 2026-05-22 Stage 2 Boundary Test Red

- Added `tests/test_consolidation_module_boundary.py`.
- `venv\Scripts\python -m pytest tests\test_consolidation_module_boundary.py -q`:
  failed as expected with three assertions covering missing
  `kazusa_ai_chatbot.consolidation.schema`, importable legacy node
  consolidator modules, and legacy node helper imports in
  `consolidation.core`.

### 2026-05-22 Stage 3 Helper Migration

- Production worker moved all helper modules from `nodes` to
  `consolidation`, updated production imports, and deleted the retired node
  stub. The worker did not edit tests, docs, or plans.
- Moved files:
  - `facts.py`
  - `images.py`
  - `memory_units.py`
  - `origin.py`
  - `origin_policy.py`
  - `persistence.py`
  - `reflection.py`
  - `schema.py`
- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\consolidation\core.py src\kazusa_ai_chatbot\consolidation\schema.py src\kazusa_ai_chatbot\consolidation\origin.py src\kazusa_ai_chatbot\consolidation\origin_policy.py src\kazusa_ai_chatbot\consolidation\facts.py src\kazusa_ai_chatbot\consolidation\reflection.py src\kazusa_ai_chatbot\consolidation\images.py src\kazusa_ai_chatbot\consolidation\memory_units.py src\kazusa_ai_chatbot\consolidation\persistence.py`:
  exit code 0.
- `venv\Scripts\python -m pytest tests\test_consolidation_module_boundary.py -q`:
  3 passed in 1.49s.

### 2026-05-22 Stage 4 Documentation And Registry

- `README.md` now describes `nodes` as persona/cognition/dialog stages and
  `consolidation` as durable consolidation helpers, target routing, and ICD.
- `src\kazusa_ai_chatbot\nodes\README.md` now describes completed persona
  state as a consolidation handoff.
- `src\kazusa_ai_chatbot\consolidation\README.md` now lists the first-class
  helper modules under `consolidation` and removes node-helper compatibility
  language.
- `development_plans\README.md` records this plan as `in_progress`.
- `rg -n "Consolidation handoff|consolidation/schema.py|consolidation/facts.py|Durable consolidation helpers" README.md src\kazusa_ai_chatbot\consolidation\README.md src\kazusa_ai_chatbot\nodes\README.md`:
  found the expected updated ownership text.

### 2026-05-22 Stage 5 Verification

- `rg -n "kazusa_ai_chatbot\.nodes\.persona_supervisor2_consolidator" src\kazusa_ai_chatbot tests`:
  no matches; exit code 1.
- `rg -n "persona_supervisor2_consolidator.*\.py|persona_supervisor2_consolidator" src\kazusa_ai_chatbot\nodes tests README.md src\kazusa_ai_chatbot\consolidation\README.md src\kazusa_ai_chatbot\nodes\README.md`:
  no matches; exit code 1.
- `rg -n "from kazusa_ai_chatbot\.consolidation\.(facts|reflection|memory_units|persistence|origin|origin_policy|schema|images)" tests src\kazusa_ai_chatbot` and companion consolidation package import scan:
  matches confirmed source and tests import the new module paths.
- `rg -n "CONSOLIDATION_LLM|_PROMPT|ainvoke" src\kazusa_ai_chatbot\consolidation\core.py src\kazusa_ai_chatbot\consolidation\target.py`:
  found the existing non-LLM `sub_graph.ainvoke(...)` in `core.py`. This was
  not a new LLM call. Follow-up diff check
  `git diff -U0 -- src\kazusa_ai_chatbot\consolidation\core.py src\kazusa_ai_chatbot\consolidation\target.py | rg -n "^[+-].*(CONSOLIDATION_LLM|_PROMPT|ainvoke)"`:
  no matches; exit code 1.
- `venv\Scripts\python -m pytest tests\test_consolidation_module_boundary.py tests\test_consolidation_target_routing.py tests\test_consolidator_group_channel_branch.py -q`:
  12 passed in 1.44s.
- `venv\Scripts\python -m pytest tests\test_consolidation_origin_metadata.py tests\test_consolidator_origin_selection.py tests\test_consolidation_origin_policy.py -q`:
  24 passed in 1.54s.
- `venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py tests\test_db_writer_cache2_invalidation.py tests\test_consolidator_efficiency.py -q`:
  14 passed in 1.53s.
- `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py tests\test_consolidator_reflection_prompts.py tests\test_consolidator_source_aware_payloads.py tests\test_consolidator_character_image.py -q`:
  16 passed in 1.54s.
- `venv\Scripts\python -m pytest tests\test_user_memory_units_rag_flow.py tests\test_service_background_consolidation.py -q`:
  37 passed in 2.49s.
- Supplemental deterministic import-path batch:
  `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py tests\test_llm_time_payload_projection.py tests\test_memory_writer_information_flow_contracts.py tests\test_memory_writer_prompt_contracts.py tests\test_multi_source_cognition_image_input.py tests\test_persona_supervisor2_schema.py tests\test_reflection_cycle_prompt_contracts.py -q`:
  68 passed in 2.03s after correcting the moved `memory_units.py` path in
  `tests\test_reflection_cycle_prompt_contracts.py`.
- Live LLM execution was not run because prompt bodies and LLM behavior are
  unchanged. Collection-only check
  `venv\Scripts\python -m pytest tests\test_consolidation_evidence_hardening_live_llm.py tests\test_memory_writer_perspective_live_llm.py tests\test_temporal_relative_terms_live_llm.py tests\test_user_memory_units_live_llm.py --collect-only -q -m live_llm`:
  30 tests collected in 1.83s.
- `git diff --check`: exit code 0 with existing LF-to-CRLF working-copy
  warnings only.

### 2026-05-22 Stage 6 Independent Code Review

- Reviewer mode: separate review subagent.
- Approval status: approved.
- Blocking findings: none.
- Non-blocking findings: none.
- Review verification:
  - `git status --short`
  - `git diff --stat`
  - `git diff HEAD --find-renames`
  - static greps for old node consolidator imports and files: no matches
  - `venv\Scripts\python -m pytest tests\test_consolidation_module_boundary.py -q`:
    3 passed
  - `venv\Scripts\python -m py_compile` over moved consolidation modules:
    exit code 0
  - `git diff --check`: exit code 0 with LF-to-CRLF working-copy warnings
    only
  - prompt and LLM-call diff scan over `consolidation/core.py` and
    `consolidation/target.py`: no changed prompt or LLM invocation lines
- Reviewer residual risks:
  - full focused regression was not independently rerun by the reviewer, but
    the active agent recorded the focused regression evidence in Stage 5
  - LF-to-CRLF working-copy warnings remain commit hygiene noise, with no
    whitespace errors
  - final commit must include unstaged import rewrites and untracked
    boundary/plan files together
- Active-agent supplemental broad-suite check:
  - A non-live suite attempt showed unrelated failures outside this migration
    surface.
  - `tests\test_multi_source_cognition_stage_07_reflection_dry_run.py::test_text_chat_prompt_fingerprints_remain_stable`
    fails on `_COGNITION_SUBCONSCIOUS_PROMPT` byte-count fingerprint in
    untouched cognition prompt and test files.
  - `tests\test_self_cognition_tracking.py::test_runner_executes_private_lifecycle_action_for_consolidation`
    fails while loading the configured local model through
    `internal_monologue_residue`; no migration-owned files are in the failing
    stack before the external model request.
  - `git diff` confirms neither failing test path nor the failing production
    prompt/LLM paths were modified by this migration.
- Fixes after review: none required.
- Final closure decision: completed because all plan acceptance criteria and
  focused migration verification gates passed, independent review approved, and
  broad-suite residuals were confirmed outside the approved migration surface.
- Post-archive focused closeout:
  `venv\Scripts\python -m pytest tests\test_consolidation_module_boundary.py tests\test_consolidation_target_routing.py tests\test_consolidator_group_channel_branch.py tests\test_consolidation_origin_metadata.py tests\test_consolidator_origin_selection.py tests\test_consolidation_origin_policy.py tests\test_consolidator_origin_policy_db_writer.py tests\test_db_writer_cache2_invalidation.py tests\test_consolidator_efficiency.py tests\test_consolidator_facts_rag2.py tests\test_consolidator_reflection_prompts.py tests\test_consolidator_source_aware_payloads.py tests\test_consolidator_character_image.py tests\test_user_memory_units_rag_flow.py tests\test_service_background_consolidation.py -q`:
  103 passed in 2.88s.

## Execution Handoff

This plan is completed and archived. New scope must use a new active plan.
