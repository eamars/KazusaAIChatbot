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
| [standalone_agentic_resolver_first_pass_plan_2026-08-23.md](active/short_term/standalone_agentic_resolver_first_pass_plan_2026-08-23.md) | Standalone thinking-enabled native-tool streaming resolver with JSON protocol, external skills, and non-recursive same-runtime subagents | draft |

### Superseded Plans

| Document | Type | Status |
|---|---|---|
| [cognition_v3_cache_affine_semantic_chain_bigbang_plan.md](archive/superseded/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md) | Former parallel cache-affine semantic-chain projection and partial execution record | superseded by `cognition_v3_hybrid_agentic_loop_reconciliation_plan.md` |
| [cognition_v3_hybrid_agentic_loop_static_architecture_review_2026-08-21.md](archive/superseded/cognition_v3_hybrid_agentic_loop_static_architecture_review_2026-08-21.md) | Static findings for the retired hybrid cognition chain | superseded by the handleless cognition cutover |

### Completed Short-Term Plans

| Document | Type | Status |
|---|---|---|
| [cognition_v3_handleless_model_contract_bigbang_plan_2026-08-22.md](archive/completed/cognition_v3_handleless_model_contract_bigbang_plan_2026-08-22.md) | Handleless cognition model-contract big-bang cutover and Gate 7 closure | completed |
| [unified_llm_native_structured_output_default_plan.md](archive/completed/unified_llm_native_structured_output_default_plan.md) | Lightweight native JSON-object default and prompt cleanup | completed |
| [cognition_v3_hybrid_agentic_loop_reconciliation_plan.md](archive/completed/cognition_v3_hybrid_agentic_loop_reconciliation_plan.md) | Cognition V3 single-lane hybrid agentic-loop reconciliation and V3-only cutover | completed |
| [cognition_v3_hybrid_agentic_loop_reconciliation_plan_handover_2026-08-21.md](archive/completed/cognition_v3_hybrid_agentic_loop_reconciliation_plan_handover_2026-08-21.md) | Gate 7/8 execution and closure handover | completed |
| [cognition_v2_runtime_decommission_after_v3_cutover_plan_2026-08-22.md](archive/completed/cognition_v2_runtime_decommission_after_v3_cutover_plan_2026-08-22.md) | V2 cognition-engine removal and V3-only cutover closure | completed |
| [cognition_v2_stale_axes_and_validator_policy_plan.md](archive/completed/cognition_v2_stale_axes_and_validator_policy_plan.md) | Stale relationship-axis maintenance and semantic/structural cognition admission policy | completed |
| [control_console_v3_cutover_startup_and_config_cleanup_plan_2026-08-22.md](archive/completed/control_console_v3_cutover_startup_and_config_cleanup_plan_2026-08-22.md) | Control-console startup repair, production verification, and V3-only environment cleanup | completed |

### Active Bugfix Plans

| Document | Type | Status |
|---|---|---|

### Completed Bugfix Plans

| Document | Type | Status |
|---|---|---|
| [cognition_v3_state_transaction_capacity_and_context_integrity_bugfix_plan_2026-08-23.md](archive/completed/cognition_v3_state_transaction_capacity_and_context_integrity_bugfix_plan_2026-08-23.md) | Cognition V3 state capacity, lifecycle, commit lineage, and prompt-context integrity correction | completed |
| [unified_llm_json_schema_fallback_no_text_bugfix_plan_2026-08-23.md](archive/completed/unified_llm_json_schema_fallback_no_text_bugfix_plan_2026-08-23.md) | Unified LLM JSON Schema fallback without text correction | completed |
| [cognition_v3_first_pass_appraisal_structural_exhaustion_diagnosis_2026-08-22.md](archive/completed/cognition_v3_first_pass_appraisal_structural_exhaustion_diagnosis_2026-08-22.md) | First-pass cognition appraisal structural-exhaustion diagnosis | completed |
| [cognition_v2_semantic_admission_original_contract_restoration_plan.md](archive/completed/cognition_v2_semantic_admission_original_contract_restoration_plan.md) | Cognition V2 semantic-admission original-contract restoration | completed |
| [dialog_final_generator_evaluator_decommission_plan.md](archive/completed/dialog_final_generator_evaluator_decommission_plan.md) | Final dialog evaluator/verifier and scheduled semantic-evaluator decommission | completed |
| [selected_operation_nested_role_scope_bugfix_plan.md](archive/completed/selected_operation_nested_role_scope_bugfix_plan.md) | Selected-operation nested role scope and dialog evaluator false-positive fix | completed |
| [dialog_language_and_model_output_contract_cleanup_plan.md](archive/completed/dialog_language_and_model_output_contract_cleanup_plan.md) | Final-dialog language delegation and deterministic model-output contract cleanup | completed |
| [group_topic_continuity_authority_fix_plan.md](archive/completed/group_topic_continuity_authority_fix_plan.md) | Group multi-user topic-continuity authority and weak-local-model quality fix | completed |
| [cognition_v2_parent_checkpoint_guardrail_plan.md](archive/completed/cognition_v2_parent_checkpoint_guardrail_plan.md) | Cognition V2 parent-checkpoint guardrail and bounded recovery epoch | completed |
| [cognition_v2_group_ownership_terminalization_bugfix_plan.md](archive/completed/cognition_v2_group_ownership_terminalization_bugfix_plan.md) | Cognition V2 group ownership terminalization and role-domain bugfix | completed |
| [cognition_v2_semantic_appraisal_boundary_recovery_bugfix_plan.md](archive/completed/cognition_v2_semantic_appraisal_boundary_recovery_bugfix_plan.md) | Cognition V2 semantic-appraisal boundary recovery and typed failure disposition | completed |
| [cognition_v2_relational_carrier_recurrence_binding_plan.md](archive/completed/cognition_v2_relational_carrier_recurrence_binding_plan.md) | Cognition V2 relational-carrier recurrence binding regression hardening | completed |
| [task_resolution_duplicate_visible_delivery_plan.md](archive/completed/task_resolution_duplicate_visible_delivery_plan.md) | Task-resolution semantic duplicate visible delivery and evidence-contract repair | completed |
| [required_selection_prompt_contract_and_evaluator_relaxation_plan.md](archive/completed/required_selection_prompt_contract_and_evaluator_relaxation_plan.md) | Required-selection prompt contract and evaluator relaxation | completed |
| [scheduled_future_speech_temporal_grounding_and_content_gate_plan.md](archive/completed/scheduled_future_speech_temporal_grounding_and_content_gate_plan.md) | Scheduled future-speech temporal grounding, provenance, pre-dispatch content gate, and consolidation safety | completed |
| [scheduled_future_speech_legacy_record_cutover_plan.md](archive/completed/scheduled_future_speech_legacy_record_cutover_plan.md) | Exact legacy scheduled-future-speech big-bang deletion | completed |

## Working-Tree Policy

- Executable implementation work belongs under `active/short_term/` or
  `active/bugfix/`.
- Long-term direction belongs in `long_term/todo.md`.
- Stable architecture and contract references belong under `docs/architecture/`.
- The independent future architecture document belongs at
  `docs/FUTURE_ARCHITECTURE.md`.
- Completed and superseded plans belong under `archive/` as immutable
  historical records.
