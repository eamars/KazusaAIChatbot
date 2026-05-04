# character reflection cycle stage 1 plan

## Summary

- Goal: Supersede the combined Stage 1 plan with three smaller independent plans.
- Plan class: small
- Status: superseded
- Mandatory skills: `development-plan-writing`
- Overall cutover strategy: no implementation is authorized by this superseded plan.
- Highest-risk areas: implementation agents accidentally using this old combined scope.
- Acceptance criteria: active Stage 1 work uses the split 1a, 1b, and 1c plans listed below.

## Context

This file previously combined reflection-cycle evaluation, memory evolution, and reflection-to-memory integration into one broad Stage 1 plan. That combined scope is no longer approved.

The approved split is:

- [character_reflection_cycle_stage1a_plan.md](C:/workspace/kazusa_ai_chatbot/development_plans/character_reflection_cycle_stage1a_plan.md): read-only reflection-cycle evaluation against recent data and real LLM behavior.
- [memory_evolution_stage1b_plan.md](C:/workspace/kazusa_ai_chatbot/development_plans/memory_evolution_stage1b_plan.md): database/search/seeding changes for evolving global persistent memory.
- [reflection_memory_integration_stage1c_plan.md](C:/workspace/kazusa_ai_chatbot/development_plans/reflection_memory_integration_stage1c_plan.md): bigbang integration that connects approved reflection output to evolved memory.

## Mandatory Skills

- `development-plan-writing`: load before modifying this superseded index.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread the active split plan it is executing, not this superseded file.
- Do not implement from this file.
- Do not copy broad combined scope from this file into an implementation task.

## Must Do

- Use Stage 1a for reflection-cycle work.
- Use Stage 1b for memory-evolution work.
- Use Stage 1c for integration work after Stage 1a approval and Stage 1b completion.

## Deferred

- All implementation is deferred to the active split plans.

## Cutover Policy

| Area | Policy | Notes |
|---|---|---|
| This combined plan | bigbang | Replaced by the three split plans. |

## Agent Autonomy Boundaries

- Implementation agents must treat this file as a pointer only.

## Target State

```text
stage 1a read-only reflection evaluation
stage 1b memory evolution/search/seeding
stage 1c integration of approved 1a and completed 1b
```

## Current Validation Snapshot

| Stage | Current implementation state | Approval state |
|---|---|---|
| Stage 1a | Read-only module, DB interface, prompt contracts, CLI, ICD, deterministic tests, and real 48-hour monitored-channel artifact are implemented in the workspace | Completed and signed off in `character_reflection_cycle_stage1a_plan.md` |
| Stage 1b | No `memory_evolution` package or reset/reseed CLI exists in the workspace | Not started and unsigned |
| Stage 1c | No production repository, worker, lore-promotion module, production CLI, or feature flags exist in the workspace | Blocked on Stage 1a approval and Stage 1b completion |

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Combined Stage 1 | Superseded | Scope was too broad and high risk. |
| Stage split | 1a, 1b, 1c | Keeps evaluation, memory DB work, and integration independently reviewable. |

## Change Surface

None. This file authorizes no code changes.

## Implementation Order

1. Execute Stage 1a.
2. Execute Stage 1b independently.
3. Execute Stage 1c only after Stage 1a approval and Stage 1b completion.

## Progress Checklist

- [ ] This superseded file is ignored by implementation agents.

## Verification

- Confirm active work references one of the split plan files.

## Acceptance Criteria

- No implementation is performed from this file.

## Execution Evidence

- Superseded by:
  - `development_plans/character_reflection_cycle_stage1a_plan.md`
  - `development_plans/memory_evolution_stage1b_plan.md`
  - `development_plans/reflection_memory_integration_stage1c_plan.md`
