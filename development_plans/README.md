# development plans registry

This directory contains current implementation planning work, historical plan
records, and the living long-term roadmap. Agents must read this registry
before opening an active plan.

## Directory Contract

| Path | Purpose | Execution rule |
|---|---|---|
| `long_term/` | Living long-term development direction. | Use as context only; promote work into an active plan before implementation. |
| `active/short_term/` | Current short-term development plans. | Execute only plans whose `Status` is `approved` or `in_progress`. |
| `active/bugfix/` | Current bugfix and quality-fix plans. | Execute only plans whose `Status` is `approved` or `in_progress`. |
| `archive/completed/` | Completed execution records. | Historical lookup only; do not append new scope. |
| `archive/superseded/` | Plans replaced by newer plans. | Historical lookup only; follow the superseding plan. |

Completed and superseded records remain under `archive/` as historical
material. Reference and triage records are removed from the current planning
surface; stable architecture references live under `docs/architecture/`.

## Promotion Rule

Long-term roadmap items become implementation work only through promotion:

```text
long_term/todo.md
  -> active/short_term/<specific_plan>.md
  -> active/bugfix/<specific_bugfix_plan>.md
  -> archive/completed/... after execution evidence is recorded
```

An active plan becomes historical when its execution and verification are
complete. Move it into the appropriate archive category and preserve its
recorded evidence.

## Current Documents

### Long-Term Direction

| Document | Type | Status |
|---|---|---|
| [todo.md](long_term/todo.md) | Living long-term development roadmap | active |

### Active Short-Term Plans

| Document | Type | Status |
|---|---|---|
| [coding_agent_assessment_gap_phase_d_plan.md](active/short_term/coding_agent_assessment_gap_phase_d_plan.md) | Coding-agent persistent JSON action-loop and repository-index migration | in_progress |

### Active Bugfix Plans

| Document | Type | Status |
|---|---|---|
| [task_resolution_duplicate_visible_delivery_plan.md](active/bugfix/task_resolution_duplicate_visible_delivery_plan.md) | Task-resolution semantic duplicate visible delivery and evidence-contract repair | approved |

### Completed Bugfix Plans

| Document | Type | Status |
|---|---|---|
| [selected_operation_nested_role_scope_bugfix_plan.md](archive/completed/selected_operation_nested_role_scope_bugfix_plan.md) | Selected-operation nested role scope and dialog evaluator false-positive fix | completed |
| [dialog_language_and_model_output_contract_cleanup_plan.md](archive/completed/dialog_language_and_model_output_contract_cleanup_plan.md) | Final-dialog language delegation and deterministic model-output contract cleanup | completed |
| [group_topic_continuity_authority_fix_plan.md](archive/completed/group_topic_continuity_authority_fix_plan.md) | Group multi-user topic-continuity authority and weak-local-model quality fix | completed |
| [cognition_v2_parent_checkpoint_guardrail_plan.md](archive/completed/cognition_v2_parent_checkpoint_guardrail_plan.md) | Cognition V2 parent-checkpoint guardrail and bounded recovery epoch | completed |
| [cognition_v2_group_ownership_terminalization_bugfix_plan.md](archive/completed/cognition_v2_group_ownership_terminalization_bugfix_plan.md) | Cognition V2 group ownership terminalization and role-domain bugfix | completed |
| [cognition_v2_relational_carrier_recurrence_binding_plan.md](archive/completed/cognition_v2_relational_carrier_recurrence_binding_plan.md) | Cognition V2 relational-carrier recurrence binding regression hardening | completed |

## Working-Tree Policy

- Executable implementation work belongs under `active/short_term/` or
  `active/bugfix/`.
- Long-term direction belongs in `long_term/todo.md`.
- Stable architecture and contract references belong under `docs/architecture/`.
- The independent future architecture document belongs at
  `docs/FUTURE_ARCHITECTURE.md`.
- Completed and superseded plans belong under `archive/` as immutable
  historical records.
