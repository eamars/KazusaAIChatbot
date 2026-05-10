# development plans registry

This directory separates long-term direction from executable short-term and
bugfix plans.

Agents must read this registry before scanning individual plans.

## Directory Contract

| Path | Purpose | Execution rule |
|---|---|---|
| `long_term/` | Living long-term development direction. | Never implement directly from this folder. Promote work into a short-term or bugfix plan first. |
| `active/short_term/` | Short-term development plans that are draft, approved, or in progress. | Execute only plans whose `Status` is `approved` or `in_progress`. |
| `active/bugfix/` | Bugfix or quality-fix plans that are draft, approved, or in progress. | Execute only plans whose `Status` is `approved` or `in_progress`. |
| `archive/completed/` | Closed historical execution records. | Historical lookup only. Do not append new scope. |
| `archive/superseded/` | Plans replaced by newer plans. | Do not execute. Follow the superseding plan instead. |
| `reference/` | Design notes and supporting references that are not execution contracts. | Use as context only. |
| `triage/` | Legacy files whose lifecycle or plan contract is not yet normalized. | Do not execute until classified and moved. |

## Promotion Rule

Long-term roadmap items become implementation work only through promotion:

```text
long_term/todo.md
  -> active/short_term/<specific_plan>.md
  -> active/bugfix/<specific_bugfix_plan>.md
  -> archive/completed/... after execution evidence is recorded
```

## Long-Term Direction

| Document | Type | Status |
|---|---|---|
| [todo.md](long_term/todo.md) | Living long-term development plan | active |

## Active Short-Term Plans

| Plan | Status | Execution |
|---|---|---|
| [multi_source_cognition_architecture_plan.md](active/short_term/multi_source_cognition_architecture_plan.md) | approved | staged |
| [multi_source_cognition_architecture_stage_00_current_chat_workflow_regression_baseline_plan.md](active/short_term/multi_source_cognition_architecture_stage_00_current_chat_workflow_regression_baseline_plan.md) | completed | completed |
| [multi_source_cognition_architecture_stage_01_cognitive_episode_contract_plan.md](active/short_term/multi_source_cognition_architecture_stage_01_cognitive_episode_contract_plan.md) | completed | completed |
| [multi_source_cognition_architecture_stage_02_chat_cognitive_episode_migration_plan.md](active/short_term/multi_source_cognition_architecture_stage_02_chat_cognitive_episode_migration_plan.md) | completed | completed |
| [multi_source_cognition_architecture_stage_03_shared_cognition_prompt_selection_plan.md](active/short_term/multi_source_cognition_architecture_stage_03_shared_cognition_prompt_selection_plan.md) | completed | completed |
| [multi_source_cognition_architecture_stage_04_rag_cognitive_episode_adapter_plan.md](active/short_term/multi_source_cognition_architecture_stage_04_rag_cognitive_episode_adapter_plan.md) | completed | completed |
| [multi_source_cognition_architecture_stage_05_consolidation_origin_metadata_threading_plan.md](active/short_term/multi_source_cognition_architecture_stage_05_consolidation_origin_metadata_threading_plan.md) | completed | completed |
| [multi_source_cognition_architecture_stage_06_consolidator_per_write_origin_policy_plan.md](active/short_term/multi_source_cognition_architecture_stage_06_consolidator_per_write_origin_policy_plan.md) | completed | completed |
| [multi_source_cognition_architecture_stage_07_reflection_trigger_cognition_dry_run_plan.md](active/short_term/multi_source_cognition_architecture_stage_07_reflection_trigger_cognition_dry_run_plan.md) | completed | completed |
| [multi_source_cognition_architecture_stage_08_internal_thought_cognition_dry_run_plan.md](active/short_term/multi_source_cognition_architecture_stage_08_internal_thought_cognition_dry_run_plan.md) | completed | completed |
| [multi_source_cognition_architecture_stage_09_multimodal_cognitive_input_sources_plan.md](active/short_term/multi_source_cognition_architecture_stage_09_multimodal_cognitive_input_sources_plan.md) | completed | completed |
| [multi_source_cognition_architecture_stage_10_permissioned_proactive_output_plan.md](active/short_term/multi_source_cognition_architecture_stage_10_permissioned_proactive_output_plan.md) | draft | blocked |
| [reflection_driven_character_state_evolution_plan.md](active/short_term/reflection_driven_character_state_evolution_plan.md) | draft | deferred |

## Active Bugfix Plans

| Plan | Status | Execution |
|---|---|---|

## Reference Documents

| Document | Type |
|---|---|
| [rag_cache2_design.md](reference/designs/rag_cache2_design.md) | Design reference |

## Triage

No triage files are currently classified.

## Evaluation Passes

### 2026-05-08 implementation reconciliation

This pass compared stale active and triage plans against source, docs, and
focused deterministic tests. Plans already implemented in code were moved to
completed history; legacy drafts replaced by later plans were moved to
superseded history.

| Outcome | Plans |
|---|---|
| Completed by actual implementation | `character_local_time_context_plan.md`, `character_self_words_retrieval_delivery_receipt_plan.md`, `conversation_progress_phase3_quality_plan.md`, `conversation_progress_state_plan.md`, `get_db_private_boundary_plan.md`, `group_chat_noise_relevance_plan.md`, `llm_routing_migration_plan.md`, `memory_evidence_scoped_user_continuity_plan.md`, `native_shape_boundary_hardening_plan.md` |
| Still unfinished as of 2026-05-08 | `character_profile_runtime_state_split_plan.md` |
| Superseded legacy drafts | `rag_supervisor2_inner_loop_agents_plan.md`, `short_circuit_early_stop_plan.md` |

## Archive

Completed and superseded records live under `archive/`. Use them for historical
lookup, rationale, and execution evidence. New work must not be added to archived
plans.

### Completed Bugfix Records

| Plan |
|---|
| [cognition_silence_short_circuit_and_dialog_evaluator_quality_plan.md](archive/completed/bugfix/cognition_silence_short_circuit_and_dialog_evaluator_quality_plan.md) |
| [rag_active_turn_conversation_row_exclusion_plan.md](archive/completed/bugfix/rag_active_turn_conversation_row_exclusion_plan.md) |
| [temporal_grounding_rag_episode_state_plan.md](archive/completed/bugfix/temporal_grounding_rag_episode_state_plan.md) |

### Completed Short-Term Records

| Plan |
|---|
| [cache2_agent_stats_health_plan.md](archive/completed/short_term/cache2_agent_stats_health_plan.md) |
| [character_local_time_context_plan.md](archive/completed/short_term/character_local_time_context_plan.md) |
| [character_profile_runtime_state_split_plan.md](archive/completed/short_term/character_profile_runtime_state_split_plan.md) |
| [character_reflection_cycle_stage1a_plan.md](archive/completed/short_term/character_reflection_cycle_stage1a_plan.md) |
| [character_self_words_retrieval_delivery_receipt_plan.md](archive/completed/short_term/character_self_words_retrieval_delivery_receipt_plan.md) |
| [cognition_state_integrity_plan.md](archive/completed/short_term/cognition_state_integrity_plan.md) |
| [consolidation_evidence_hardening_plan.md](archive/completed/short_term/consolidation_evidence_hardening_plan.md) |
| [conversation_progress_flow_phase2_plan.md](archive/completed/short_term/conversation_progress_flow_phase2_plan.md) |
| [conversation_progress_phase3_quality_plan.md](archive/completed/short_term/conversation_progress_phase3_quality_plan.md) |
| [conversation_progress_state_plan.md](archive/completed/short_term/conversation_progress_state_plan.md) |
| [get_db_private_boundary_plan.md](archive/completed/short_term/get_db_private_boundary_plan.md) |
| [global_input_queue_plan.md](archive/completed/short_term/global_input_queue_plan.md) |
| [group_chat_noise_relevance_plan.md](archive/completed/short_term/group_chat_noise_relevance_plan.md) |
| [identity_free_memory_output_contract_plan.md](archive/completed/short_term/identity_free_memory_output_contract_plan.md) |
| [interaction_style_image_plan.md](archive/completed/short_term/interaction_style_image_plan.md) |
| [live_context_runtime_facts_plan.md](archive/completed/short_term/live_context_runtime_facts_plan.md) |
| [llm_routing_migration_plan.md](archive/completed/short_term/llm_routing_migration_plan.md) |
| [memory_evidence_scoped_user_continuity_plan.md](archive/completed/short_term/memory_evidence_scoped_user_continuity_plan.md) |
| [memory_evolution_stage1b_plan.md](archive/completed/short_term/memory_evolution_stage1b_plan.md) |
| [message_coalescing_queue_module_plan.md](archive/completed/short_term/message_coalescing_queue_module_plan.md) |
| [native_shape_boundary_hardening_plan.md](archive/completed/short_term/native_shape_boundary_hardening_plan.md) |
| [prompt_safe_message_context_plan.md](archive/completed/short_term/prompt_safe_message_context_plan.md) |
| [rag1_decommission_plan.md](archive/completed/short_term/rag1_decommission_plan.md) |
| [rag2_phase4_continuation_plan.md](archive/completed/short_term/rag2_phase4_continuation_plan.md) |
| [rag_cache2_persistent_initializer_plan.md](archive/completed/short_term/rag_cache2_persistent_initializer_plan.md) |
| [rag_current_turn_exclusion_plan.md](archive/completed/short_term/rag_current_turn_exclusion_plan.md) |
| [rag_phase3_development_plan.md](archive/completed/short_term/rag_phase3_development_plan.md) |
| [rag_reply_mention_and_vague_input_plan.md](archive/completed/short_term/rag_reply_mention_and_vague_input_plan.md) |
| [recall_agent_plan.md](archive/completed/short_term/recall_agent_plan.md) |
| [reflection_memory_integration_stage1c_plan.md](archive/completed/short_term/reflection_memory_integration_stage1c_plan.md) |
| [role_vocabulary_contract_cleanup_plan.md](archive/completed/short_term/role_vocabulary_contract_cleanup_plan.md) |
| [service_module_separation_stage1_plan.md](archive/completed/short_term/service_module_separation_stage1_plan.md) |
| [typed_message_envelope_stage2_plan.md](archive/completed/short_term/typed_message_envelope_stage2_plan.md) |
| [user_memory_unit_rolling_window_plan.md](archive/completed/short_term/user_memory_unit_rolling_window_plan.md) |

### Superseded Records

| Plan |
|---|
| [character_reflection_cycle_stage1_plan.md](archive/superseded/character_reflection_cycle_stage1_plan.md) |
| [rag_supervisor2_inner_loop_agents_plan.md](archive/superseded/rag_supervisor2_inner_loop_agents_plan.md) |
| [short_circuit_early_stop_plan.md](archive/superseded/short_circuit_early_stop_plan.md) |
