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

| Document | Type | Status |
|---|---|---|
| [l2d_action_router_prompt_separation_plan.md](active/short_term/l2d_action_router_prompt_separation_plan.md) | Large top-level action-router prompt and boundary separation plan | draft |
| [media_descriptor_cache_plan.md](active/short_term/media_descriptor_cache_plan.md) | Medium media descriptor cache using Cache2 persistent layer | draft |
| [rag3_router_interpreter_poc_experiment_plan.md](active/short_term/rag3_router_interpreter_poc_experiment_plan.md) | Medium experiment and POC design plan | draft |

## Supporting Experiment Records

| Document | Type | Status | Supports |
|---|---|---|---|
| [rag2_recall_quality_experiment_plan.md](reference/evidence/rag2_recall_quality_experiment_plan.md) | Experiment decision and supporting evidence | reference evidence | [rag2_cognition_ready_evidence_plan.md](archive/completed/short_term/rag2_cognition_ready_evidence_plan.md) |

## Active Bugfix Plans

| Document | Type | Status |
|---|---|---|
| [cross_thread_image_contamination_bugfix_plan.md](active/bugfix/cross_thread_image_contamination_bugfix_plan.md) | Small observation-only bugfix draft for QQ group image/thread source contamination | draft |
| [qq_replied_image_description_unavailable_queue_prune_bugfix_plan.md](active/bugfix/qq_replied_image_description_unavailable_queue_prune_bugfix_plan.md) | Small observation-only bugfix draft for replied image description loss after queue pruning | draft |
| [rag2_public_output_contract_leak_bugfix_plan.md](active/bugfix/rag2_public_output_contract_leak_bugfix_plan.md) | Large RAG2 prompt/evidence contract bugfix plan | draft |
| [self_other_inversion_personality_question_bugfix_plan.md](active/bugfix/self_other_inversion_personality_question_bugfix_plan.md) | Small observation-only bugfix draft for Chinese self/other referent inversion | draft |

## Reference Documents

| Document | Type |
|---|---|
| [action_spec_effector_expansion_architecture.md](reference/designs/action_spec_effector_expansion_architecture.md) | Architecture reference |
| [cognition_contracts_design.md](reference/designs/cognition_contracts_design.md) | Authoritative contract reference |
| [cognition_core_evolution_progression.md](reference/designs/cognition_core_evolution_progression.md) | Architectural progression |
| [rag_cache2_design.md](reference/designs/rag_cache2_design.md) | Design reference |
| [rag_hybrid_search_architecture_decision.md](reference/designs/rag_hybrid_search_architecture_decision.md) | Design reference |
| [self_cognition_loop_architecture.md](reference/designs/self_cognition_loop_architecture.md) | Architecture reference |
| [self_cognition_reasoning_basis.md](reference/designs/self_cognition_reasoning_basis.md) | Reasoning basis |
| [self_cognition_tracking_icd.md](reference/designs/self_cognition_tracking_icd.md) | Interface control document |
| [cognition_prompt_chain_side_by_side_comparison_20260519.md](reference/evidence/cognition_prompt_chain_side_by_side_comparison_20260519.md) | Supporting evidence |
| [cognition_prompt_chain_previous20_equivalence_check_20260519.md](reference/evidence/cognition_prompt_chain_previous20_equivalence_check_20260519.md) | Supporting evidence |
| [cognition_prompt_chain_previous20_input_output_20260519.md](reference/evidence/cognition_prompt_chain_previous20_input_output_20260519.md) | Supporting evidence |
| [rag2_recall_quality_experiment_plan.md](reference/evidence/rag2_recall_quality_experiment_plan.md) | Supporting experiment evidence |
| [self_cognition_rag_resolver_evidence_review_20260601.md](reference/evidence/self_cognition_rag_resolver_evidence_review_20260601.md) | Supporting real LLM evidence |
| [multi_source_cognition_stage_03_inactive_prompt_variant_notes.md](reference/multi_source_cognition_stage_03_inactive_prompt_variant_notes.md) | Reference notes |

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
| [character_self_image_rolling_state_bugfix_plan.md](archive/completed/bugfix/character_self_image_rolling_state_bugfix_plan.md) |
| [cognition_silence_short_circuit_and_dialog_evaluator_quality_plan.md](archive/completed/bugfix/cognition_silence_short_circuit_and_dialog_evaluator_quality_plan.md) |
| [consolidation_module_boundary_migration_bugfix_plan.md](archive/completed/bugfix/consolidation_module_boundary_migration_bugfix_plan.md) |
| [consolidator_facts_prompt_budget_bugfix_plan.md](archive/completed/bugfix/consolidator_facts_prompt_budget_bugfix_plan.md) |
| [conversation_progress_identity_leakage_bugfix_plan.md](archive/completed/bugfix/conversation_progress_identity_leakage_bugfix_plan.md) |
| [decontextualizer_scope_users_referent_bugfix_plan.md](archive/completed/bugfix/decontextualizer_scope_users_referent_bugfix_plan.md) |
| [dialog_anchor_authority_stale_history_bugfix_plan.md](archive/completed/bugfix/dialog_anchor_authority_stale_history_bugfix_plan.md) |
| [dialog_evaluator_guess_owner_boundary_bugfix_plan.md](archive/completed/bugfix/dialog_evaluator_guess_owner_boundary_bugfix_plan.md) |
| [generic_cognition_prompt_migration_plan.md](archive/completed/bugfix/generic_cognition_prompt_migration_plan.md) |
| [history_media_projection_image_boundary_plan.md](archive/completed/bugfix/history_media_projection_image_boundary_plan.md) |
| [l3_content_anchor_open_loop_resolution_plan.md](archive/completed/bugfix/l3_content_anchor_open_loop_resolution_plan.md) |
| [llm_semantic_descriptor_validation_bugfix_plan.md](archive/completed/bugfix/llm_semantic_descriptor_validation_bugfix_plan.md) |
| [lm_studio_model_unload_retry_bugfix_plan.md](archive/completed/bugfix/lm_studio_model_unload_retry_bugfix_plan.md) |
| [memory_lifecycle_specialist_routing_plan.md](archive/completed/bugfix/memory_lifecycle_specialist_routing_plan.md) |
| [no_due_commitment_lifecycle_resolution_plan.md](archive/completed/bugfix/no_due_commitment_lifecycle_resolution_plan.md) |
| [rag2_cognition_identity_evidence_content_bugfix_plan.md](archive/completed/bugfix/rag2_cognition_identity_evidence_content_bugfix_plan.md) |
| [quote_aware_rag_sequence_plan.md](archive/completed/bugfix/quote_aware_rag_sequence_plan.md) |
| [qq_face_projection_empty_input_guard_bugfix_plan.md](archive/completed/bugfix/qq_face_projection_empty_input_guard_bugfix_plan.md) |
| [rag_active_turn_conversation_row_exclusion_plan.md](archive/completed/bugfix/rag_active_turn_conversation_row_exclusion_plan.md) |
| [rag_hybrid_search_time_config_plan.md](archive/completed/bugfix/rag_hybrid_search_time_config_plan.md) |
| [rag_memory_evidence_remember_me_inner_path_bugfix_plan.md](archive/completed/bugfix/rag_memory_evidence_remember_me_inner_path_bugfix_plan.md) |
| [rag_conversation_evidence_current_episode_boundary_bugfix_plan.md](archive/completed/bugfix/rag_conversation_evidence_current_episode_boundary_bugfix_plan.md) |
| [rag_retrieval_top_k_embedding_tuning_plan.md](archive/completed/bugfix/rag_retrieval_top_k_embedding_tuning_plan.md) |
| [reflection_global_promotion_replay_bugfix_plan.md](archive/completed/bugfix/reflection_global_promotion_replay_bugfix_plan.md) |
| [resolver_image_only_empty_input_bugfix_plan.md](archive/completed/bugfix/resolver_image_only_empty_input_bugfix_plan.md) |
| [self_cognition_background_context_budget_bugfix_plan.md](archive/completed/bugfix/self_cognition_background_context_budget_bugfix_plan.md) |
| [self_cognition_character_global_id_config_bugfix_plan.md](archive/completed/bugfix/self_cognition_character_global_id_config_bugfix_plan.md) |
| [self_cognition_dialog_state_contract_bugfix_plan.md](archive/completed/bugfix/self_cognition_dialog_state_contract_bugfix_plan.md) |
| [self_cognition_group_speak_selection_bugfix_plan.md](archive/completed/bugfix/self_cognition_group_speak_selection_bugfix_plan.md) |
| [self_cognition_sleep_period_plan.md](archive/completed/bugfix/self_cognition_sleep_period_plan.md) |
| [self_cognition_speak_delivery_bugfix_plan.md](archive/completed/bugfix/self_cognition_speak_delivery_bugfix_plan.md) |
| [task_dispatcher_json_contract_bugfix_plan.md](archive/completed/bugfix/task_dispatcher_json_contract_bugfix_plan.md) |
| [temporal_grounding_rag_episode_state_plan.md](archive/completed/bugfix/temporal_grounding_rag_episode_state_plan.md) |
| [time_source_boundary_bugfix_plan.md](archive/completed/bugfix/time_source_boundary_bugfix_plan.md) |

### Completed Short-Term Records

| Plan |
|---|
| [cache2_agent_stats_health_plan.md](archive/completed/short_term/cache2_agent_stats_health_plan.md) |
| [background_artifact_handoff_poc_plan.md](archive/completed/short_term/background_artifact_handoff_poc_plan.md) |
| [character_local_time_context_plan.md](archive/completed/short_term/character_local_time_context_plan.md) |
| [character_profile_runtime_state_split_plan.md](archive/completed/short_term/character_profile_runtime_state_split_plan.md) |
| [character_reflection_cycle_stage1a_plan.md](archive/completed/short_term/character_reflection_cycle_stage1a_plan.md) |
| [character_self_words_retrieval_delivery_receipt_plan.md](archive/completed/short_term/character_self_words_retrieval_delivery_receipt_plan.md) |
| [cognition_llm_stage_reconnection_plan.md](archive/completed/short_term/cognition_llm_stage_reconnection_plan.md) |
| [cognition_visual_directives_control_plan.md](archive/completed/short_term/cognition_visual_directives_control_plan.md) |
| [cognition_state_integrity_plan.md](archive/completed/short_term/cognition_state_integrity_plan.md) |
| [consolidator_text_dispatch_decommission_plan.md](archive/completed/short_term/consolidator_text_dispatch_decommission_plan.md) |
| [consolidation_evidence_hardening_plan.md](archive/completed/short_term/consolidation_evidence_hardening_plan.md) |
| [consolidation_target_routing_architecture_plan.md](archive/completed/short_term/consolidation_target_routing_architecture_plan.md) |
| [conversation_progress_flow_phase2_plan.md](archive/completed/short_term/conversation_progress_flow_phase2_plan.md) |
| [conversation_progress_phase3_quality_plan.md](archive/completed/short_term/conversation_progress_phase3_quality_plan.md) |
| [conversation_progress_state_plan.md](archive/completed/short_term/conversation_progress_state_plan.md) |
| [dialog_mention_target_user_plan.md](archive/completed/short_term/dialog_mention_target_user_plan.md) |
| [event_logging_observability_plan.md](archive/completed/short_term/event_logging_observability_plan.md) |
| [first_class_image_input_cognition_plan.md](archive/completed/short_term/first_class_image_input_cognition_plan.md) |
| [get_db_private_boundary_plan.md](archive/completed/short_term/get_db_private_boundary_plan.md) |
| [global_character_growth_from_reflection_plan.md](archive/completed/short_term/global_character_growth_from_reflection_plan.md) |
| [global_input_queue_plan.md](archive/completed/short_term/global_input_queue_plan.md) |
| [group_chat_user_style_image_plan.md](archive/completed/short_term/group_chat_user_style_image_plan.md) |
| [group_chat_noise_relevance_plan.md](archive/completed/short_term/group_chat_noise_relevance_plan.md) |
| [identity_free_memory_output_contract_plan.md](archive/completed/short_term/identity_free_memory_output_contract_plan.md) |
| [internal_monologue_residue_lifecycle_plan.md](archive/completed/short_term/internal_monologue_residue_lifecycle_plan.md) |
| [interaction_style_image_plan.md](archive/completed/short_term/interaction_style_image_plan.md) |
| [live_context_runtime_facts_plan.md](archive/completed/short_term/live_context_runtime_facts_plan.md) |
| [llm_routing_migration_plan.md](archive/completed/short_term/llm_routing_migration_plan.md) |
| [l2d_l3_surface_handoff_plan.md](archive/completed/short_term/l2d_l3_surface_handoff_plan.md) |
| [l2d_router_split_and_background_ack_plan.md](archive/completed/short_term/l2d_router_split_and_background_ack_plan.md) |
| [memory_evidence_scoped_user_continuity_plan.md](archive/completed/short_term/memory_evidence_scoped_user_continuity_plan.md) |
| [memory_evolution_stage1b_plan.md](archive/completed/short_term/memory_evolution_stage1b_plan.md) |
| [message_coalescing_queue_module_plan.md](archive/completed/short_term/message_coalescing_queue_module_plan.md) |
| [modality_neutral_action_spec_effector_expansion_plan.md](archive/completed/short_term/modality_neutral_action_spec_effector_expansion_plan.md) |
| [multi_source_cognition_architecture_plan.md](archive/completed/short_term/multi_source_cognition_architecture_plan.md) |
| [multi_source_cognition_architecture_stage_00_current_chat_workflow_regression_baseline_plan.md](archive/completed/short_term/multi_source_cognition_architecture_stage_00_current_chat_workflow_regression_baseline_plan.md) |
| [multi_source_cognition_architecture_stage_01_cognitive_episode_contract_plan.md](archive/completed/short_term/multi_source_cognition_architecture_stage_01_cognitive_episode_contract_plan.md) |
| [multi_source_cognition_architecture_stage_02_chat_cognitive_episode_migration_plan.md](archive/completed/short_term/multi_source_cognition_architecture_stage_02_chat_cognitive_episode_migration_plan.md) |
| [multi_source_cognition_architecture_stage_03_shared_cognition_prompt_selection_plan.md](archive/completed/short_term/multi_source_cognition_architecture_stage_03_shared_cognition_prompt_selection_plan.md) |
| [multi_source_cognition_architecture_stage_04_rag_cognitive_episode_adapter_plan.md](archive/completed/short_term/multi_source_cognition_architecture_stage_04_rag_cognitive_episode_adapter_plan.md) |
| [multi_source_cognition_architecture_stage_05_consolidation_origin_metadata_threading_plan.md](archive/completed/short_term/multi_source_cognition_architecture_stage_05_consolidation_origin_metadata_threading_plan.md) |
| [multi_source_cognition_architecture_stage_06_consolidator_per_write_origin_policy_plan.md](archive/completed/short_term/multi_source_cognition_architecture_stage_06_consolidator_per_write_origin_policy_plan.md) |
| [multi_source_cognition_architecture_stage_07_reflection_trigger_cognition_dry_run_plan.md](archive/completed/short_term/multi_source_cognition_architecture_stage_07_reflection_trigger_cognition_dry_run_plan.md) |
| [multi_source_cognition_architecture_stage_08_internal_thought_cognition_dry_run_plan.md](archive/completed/short_term/multi_source_cognition_architecture_stage_08_internal_thought_cognition_dry_run_plan.md) |
| [multi_source_cognition_architecture_stage_09_multimodal_cognitive_input_sources_plan.md](archive/completed/short_term/multi_source_cognition_architecture_stage_09_multimodal_cognitive_input_sources_plan.md) |
| [multi_source_cognition_architecture_stage_10_permissioned_proactive_output_plan.md](archive/completed/short_term/multi_source_cognition_architecture_stage_10_permissioned_proactive_output_plan.md) |
| [native_shape_boundary_hardening_plan.md](archive/completed/short_term/native_shape_boundary_hardening_plan.md) |
| [napcat_qq_adapter_modularization_face_catalog_plan.md](archive/completed/short_term/napcat_qq_adapter_modularization_face_catalog_plan.md) |
| [outbound_adapter_channel_allowlist_plan.md](archive/completed/short_term/outbound_adapter_channel_allowlist_plan.md) |
| [prompt_prefix_and_input_format_optimization_plan.md](archive/completed/short_term/prompt_prefix_and_input_format_optimization_plan.md) |
| [prompt_safe_message_context_plan.md](archive/completed/short_term/prompt_safe_message_context_plan.md) |
| [qq_adapter_readable_mentions_plan.md](archive/completed/short_term/qq_adapter_readable_mentions_plan.md) |
| [rag1_decommission_plan.md](archive/completed/short_term/rag1_decommission_plan.md) |
| [rag_2_1_initializer_subagent_contract_plan.md](archive/completed/short_term/rag_2_1_initializer_subagent_contract_plan.md) |
| [rag2_cognition_ready_evidence_plan.md](archive/completed/short_term/rag2_cognition_ready_evidence_plan.md) |
| [rag2_mainline_fusion_recall_quality_plan.md](archive/completed/short_term/rag2_mainline_fusion_recall_quality_plan.md) |
| [rag2_phase4_continuation_plan.md](archive/completed/short_term/rag2_phase4_continuation_plan.md) |
| [rag_agent_package_reorganization_plan.md](archive/completed/short_term/rag_agent_package_reorganization_plan.md) |
| [rag_cache2_persistent_initializer_plan.md](archive/completed/short_term/rag_cache2_persistent_initializer_plan.md) |
| [rag_current_turn_exclusion_plan.md](archive/completed/short_term/rag_current_turn_exclusion_plan.md) |
| [rag_phase3_development_plan.md](archive/completed/short_term/rag_phase3_development_plan.md) |
| [rag_reply_mention_and_vague_input_plan.md](archive/completed/short_term/rag_reply_mention_and_vague_input_plan.md) |
| [recall_agent_plan.md](archive/completed/short_term/recall_agent_plan.md) |
| [reflection_attached_group_self_cognition_plan.md](archive/completed/short_term/reflection_attached_group_self_cognition_plan.md) |
| [reflection_flag_simplification_plan.md](archive/completed/short_term/reflection_flag_simplification_plan.md) |
| [reflection_memory_integration_stage1c_plan.md](archive/completed/short_term/reflection_memory_integration_stage1c_plan.md) |
| [reflection_phase_scheduled_group_review_plan.md](archive/completed/short_term/reflection_phase_scheduled_group_review_plan.md) |
| [resolver_default_mainline_cutover_plan.md](archive/completed/short_term/resolver_default_mainline_cutover_plan.md) |
| [role_vocabulary_contract_cleanup_plan.md](archive/completed/short_term/role_vocabulary_contract_cleanup_plan.md) |
| [self_cognition_agency_loop_plan.md](archive/completed/short_term/self_cognition_agency_loop_plan.md) |
| [self_cognition_group_mention_delivery_plan.md](archive/completed/short_term/self_cognition_group_mention_delivery_plan.md) |
| [self_cognition_group_review_participant_context_plan.md](archive/completed/short_term/self_cognition_group_review_participant_context_plan.md) |
| [self_cognition_memory_semantics_plan.md](archive/completed/short_term/self_cognition_memory_semantics_plan.md) |
| [self_cognition_rag_resolver_evidence_plan.md](archive/completed/short_term/self_cognition_rag_resolver_evidence_plan.md) |
| [searxng_mcp_phaseout_plan.md](archive/completed/short_term/searxng_mcp_phaseout_plan.md) |
| [service_module_separation_stage1_plan.md](archive/completed/short_term/service_module_separation_stage1_plan.md) |
| [typed_message_envelope_stage2_plan.md](archive/completed/short_term/typed_message_envelope_stage2_plan.md) |
| [unconditional_shared_memory_prewarm_plan.md](archive/completed/short_term/unconditional_shared_memory_prewarm_plan.md) |
| [universal_calendar_scheduler_plan.md](archive/completed/short_term/universal_calendar_scheduler_plan.md) |
| [user_style_engagement_consumer_plan.md](archive/completed/short_term/user_style_engagement_consumer_plan.md) |
| [user_memory_unit_rolling_window_plan.md](archive/completed/short_term/user_memory_unit_rolling_window_plan.md) |
| [web_agent3_replacement_plan.md](archive/completed/short_term/web_agent3_replacement_plan.md) |

### Superseded Records

| Plan |
|---|
| [character_reflection_cycle_stage1_plan.md](archive/superseded/character_reflection_cycle_stage1_plan.md) |
| [conversation_graph_recent_context_plan.md](archive/superseded/conversation_graph_recent_context_plan.md) |
| [cognition_preserving_goal_resolver_production_plan.md](archive/superseded/cognition_preserving_goal_resolver_production_plan.md) |
| [graph_rag_recall_experiment_plan.md](archive/superseded/graph_rag_recall_experiment_plan.md) |
| [goal_resolver_poc_plan.md](archive/superseded/goal_resolver_poc_plan.md) |
| [rag_supervisor2_inner_loop_agents_plan.md](archive/superseded/rag_supervisor2_inner_loop_agents_plan.md) |
| [short_circuit_early_stop_plan.md](archive/superseded/short_circuit_early_stop_plan.md) |
