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
| [dsh_standalone_sidecar_and_resolution_interface_plan_2026-08-26.md](active/short_term/dsh_standalone_sidecar_and_resolution_interface_plan_2026-08-26.md) | Plan 1 executable standalone DSH sidecar, canonical resolution interface, durable lifecycle, and old resolver replacement | draft; executable after user approval |
| [dsh_semantic_tools_and_coding_capability_plan_2026-08-26.md](active/short_term/dsh_semantic_tools_and_coding_capability_plan_2026-08-26.md) | Plan 2 coarse standalone Kazusa semantic tools and DSH-native coding capability expansion | draft; refine after Plan 1 closure |
| [dsh_brain_bigbang_cutover_and_legacy_resolution_decommission_plan_2026-08-26.md](active/short_term/dsh_brain_bigbang_cutover_and_legacy_resolution_decommission_plan_2026-08-26.md) | Plan 3 coarse DSH-only Brain big-bang cutover and legacy resolution/coding design decommission | draft; refine after Plan 2 closure |

### Superseded Plans

| Document | Type | Status |
|---|---|---|
| [agentic_resolver_phase2_readiness_real_llm_evaluation_plan_2026-08-23.md](archive/superseded/agentic_resolver_phase2_readiness_real_llm_evaluation_plan_2026-08-23.md) | Retired standalone-first four-facade resolver readiness evaluation | superseded by the renewed agentic resolver target architecture |
| [cognition_v3_cache_affine_semantic_chain_bigbang_plan.md](archive/superseded/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md) | Former parallel cache-affine semantic-chain projection and partial execution record | superseded by `cognition_v3_hybrid_agentic_loop_reconciliation_plan.md` |
| [cognition_v3_hybrid_agentic_loop_static_architecture_review_2026-08-21.md](archive/superseded/cognition_v3_hybrid_agentic_loop_static_architecture_review_2026-08-21.md) | Static findings for the retired hybrid cognition chain | superseded by the handleless cognition cutover |
| [multi_turn_semantic_progression_and_response_goal_fixation_bugfix_plan_2026-08-23.md](archive/superseded/multi_turn_semantic_progression_and_response_goal_fixation_bugfix_plan_2026-08-23.md) | Partial semantic-progression implementation and immutable execution evidence | superseded by the consolidated Asuna semantic-authority plan |
| [unaccepted_character_proposal_active_commitment_lane_authority_bugfix_plan_2026-08-24.md](archive/superseded/unaccepted_character_proposal_active_commitment_lane_authority_bugfix_plan_2026-08-24.md) | False-commitment lane RCA and proposed contract | superseded by the consolidated Asuna semantic-authority plan |
| [cross_user_character_memory_scope_and_authority_bugfix_plan_2026-08-23.md](archive/superseded/cross_user_character_memory_scope_and_authority_bugfix_plan_2026-08-23.md) | Cross-user memory scope, typed authority, audit, and remediation design | superseded by the consolidated Asuna semantic-authority plan |

### Completed Short-Term Plans

| Document | Type | Status |
|---|---|---|
| [cognition_observability_icd_and_console_consistency_plan_2026-08-26.md](archive/completed/short_term/cognition_observability_icd_and_console_consistency_plan_2026-08-26.md) | Canonical cognition observability ICD, explicit prewarm outcome, and consistent control-console graph cutover | completed |
| [asuna_real_e2e_50_turn_conversation_practice_plan_2026-08-23.md](archive/completed/short_term/asuna_real_e2e_50_turn_conversation_practice_plan_2026-08-23.md) | Fresh-identity adaptive 50-turn real debug conversation practice, issue register, and optimization handoff | completed |
| [standalone_agentic_resolver_first_pass_plan_2026-08-23.md](archive/completed/short_term/standalone_agentic_resolver_first_pass_plan_2026-08-23.md) | Standalone thinking-enabled native-tool streaming resolver with JSON protocol, external skills, and non-recursive same-runtime subagents | completed |
| [cognition_v3_full_chain_native_chinese_prompt_migration_plan_2026-08-23.md](archive/completed/short_term/cognition_v3_full_chain_native_chinese_prompt_migration_plan_2026-08-23.md) | Canonical A1/A2/G/P native-Chinese prompt migration and end-to-end verification | completed |
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
| [gemma4_thinking_disable_enforcement_gap_bugfix_plan_2026-08-25.md](active/bugfix/gemma4_thinking_disable_enforcement_gap_bugfix_plan_2026-08-25.md) | Gemma 4 provider-side thinking-disable enforcement gap | in_progress |
| [epistemic_and_role_provenance_fidelity_bugfix_plan_2026-08-24.md](active/bugfix/epistemic_and_role_provenance_fidelity_bugfix_plan_2026-08-24.md) | Independent unsupported-fact, private-residue, and temporary-role provenance fidelity plan from the 50-turn Asuna run | draft; separate ownership and approval boundary |
| [live_response_generation_failure_modes_problem_statement_2026-08-27.md](active/bugfix/live_response_generation_failure_modes_problem_statement_2026-08-27.md) | Live-response LLM generation failure-mode problem statement | draft; problem statement only; non-executable |

### Completed Bugfix Plans

| Document | Type | Status |
|---|---|---|
| [qwen_alias_thinking_disable_enforcement_bugfix_plan_2026-08-25.md](archive/completed/bugfix/qwen_alias_thinking_disable_enforcement_bugfix_plan_2026-08-25.md) | Qwen-family alias provider-side thinking-disable enforcement | completed; deterministic and real-LLM gates passed on 2026-08-25 |
| [asuna_semantic_authority_and_memory_feedback_consolidated_bugfix_plan_2026-08-24.md](archive/completed/bugfix/asuna_semantic_authority_and_memory_feedback_consolidated_bugfix_plan_2026-08-24.md) | Consolidated semantic progression, proposal ownership, commitment admission, cross-user learned-memory authority, audit, and real-LLM execution record | closed with residual failures on 2026-08-25; execution complete, functional sign-off RED |
| [resolver_authored_speak_provenance_contract_bugfix_plan_2026-08-24.md](archive/completed/bugfix/resolver_authored_speak_provenance_contract_bugfix_plan_2026-08-24.md) | Resolver-authored pending, user-input, and terminal visible-speak target provenance repair | completed; deterministic and user-approved real-service fallback gates passed on 2026-08-24 |
| [cognition_v3_consolidation_interaction_subtext_handoff_bugfix_plan_2026-08-23.md](archive/completed/bugfix/cognition_v3_consolidation_interaction_subtext_handoff_bugfix_plan_2026-08-23.md) | Cognition V3 required subjective-state handoff and zero-exception post-turn consolidation repair | completed |
| [cognition_subjective_continuity_dialog_quality_plan_2026-08-23.md](archive/completed/bugfix/cognition_subjective_continuity_dialog_quality_plan_2026-08-23.md) | First-person cognition continuity, epistemic dialog-boundary, and exact action-authority quality correction | completed |
| [cognition_full_protected_trace_capture_bugfix_plan_2026-08-23.md](archive/completed/cognition_full_protected_trace_capture_bugfix_plan_2026-08-23.md) | Successful Cognition V3 and L3 full protected trace capture restoration | completed |
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
