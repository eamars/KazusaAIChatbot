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
| (none) | No active short-term execution plan | none |

## Supporting Experiment Records

| Document | Type | Status | Supports |
|---|---|---|---|
| [rag2_recall_quality_experiment_plan.md](reference/designs/rag2_recall_quality_experiment_plan.md) | Experiment decision and supporting evidence | reference evidence | [rag2_cognition_ready_evidence_plan.md](archive/completed/short_term/rag2_cognition_ready_evidence_plan.md) |

## Active Bugfix Plans

| Document | Type | Status |
|---|---|---|
| [rag2_public_output_contract_leak_bugfix_plan.md](active/bugfix/rag2_public_output_contract_leak_bugfix_plan.md) | Large RAG2 prompt/evidence contract bugfix plan | draft |

## Reference Documents

| Document | Type |
|---|---|
| [action_spec_effector_expansion_architecture.md](reference/designs/action_spec_effector_expansion_architecture.md) | Architecture reference |
| [coding_agent_architecture.md](reference/designs/coding_agent_architecture.md) | Architecture reference |
| [coding_agent_phase9_run_supervisor_architecture.md](reference/designs/coding_agent_phase9_run_supervisor_architecture.md) | Directional architecture reference |
| [coding_agent_phase10_repository_scale_reading_architecture.md](reference/designs/coding_agent_phase10_repository_scale_reading_architecture.md) | Directional architecture reference |
| [coding_agent_phase2_new_artifact_gating_tests.md](reference/designs/coding_agent_phase2_new_artifact_gating_tests.md) | Supporting verification procedure and pass criteria |
| [codex_single_agent_source_guide.md](reference/designs/codex_single_agent_source_guide.md) | Codex single-agent source reference |
| [cognition_contracts_design.md](reference/designs/cognition_contracts_design.md) | Authoritative contract reference |
| [cognition_core_evolution_progression.md](reference/designs/cognition_core_evolution_progression.md) | Architectural progression |
| [kazusa_parallel_cognition_architecture.md](reference/designs/kazusa_parallel_cognition_architecture.md) | Architecture reference |
| [rag_cache2_design.md](reference/designs/rag_cache2_design.md) | Design reference |
| [rag_hybrid_search_architecture_decision.md](reference/designs/rag_hybrid_search_architecture_decision.md) | Design reference |
| (removed) cognition_prompt_chain_side_by_side_comparison_20260519.md | Supporting evidence |
| (removed) cognition_prompt_chain_previous20_equivalence_check_20260519.md | Supporting evidence |
| (removed) cognition_prompt_chain_previous20_input_output_20260519.md | Supporting evidence |
| [rag2_recall_quality_experiment_plan.md](reference/designs/rag2_recall_quality_experiment_plan.md) | Supporting experiment evidence |
| (removed) self_cognition_rag_resolver_evidence_review_20260601.md | Supporting real LLM evidence |

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

### 2026-07-02 active lifecycle cleanup

This pass compared active plan statuses against plan files and source/test
evidence, then moved completed short-term records out of `active/`.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `control_console_auto_model_discovery_picker_plan.md`, `control_console_brain_model_route_config_plan.md`, `control_console_cognition_debug_visibility_plan.md`, `llm_trace_observability_and_retrieval_plan.md`, `web_agent3_source_availability_bigbang_plan.md` |
| Completed by user-approved fallback execution without subagent | `web_agent3_bilibili_source_subagent_plan.md` |
| At that time, kept active draft because required implementation artifacts were absent | `coding_agent_phase1_fetching_reading_plan.md`; superseded by the 2026-07-07 coding-agent active plan refresh |
| Kept active draft because the bugfix remains unexecuted | `rag2_public_output_contract_leak_bugfix_plan.md`; source still contains the planned forbidden RAG public-output phrases in `persona_supervisor2_rag_evaluator.py` |

### 2026-07-04 RAG3 active plan refresh

The active RAG3 router/interpreter POC draft was removed after the user chose a
bigbang RAG 3 local-context resolver direction aligned with the complex task
resolver architecture.

| Outcome | Plans |
|---|---|
| Removed from active short-term | `rag3_router_interpreter_poc_experiment_plan.md` |
| Added to active short-term | `rag3_local_context_resolver_bigbang_plan.md` |

### 2026-07-04 RAG3 production cutover completion

The approved RAG3 local-context resolver bigbang plan completed after source
hydration, Cache2 integration, one-at-a-time real LLM verification, full
non-live regression, independent review remediation, and documentation
closeout.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `rag3_local_context_resolver_bigbang_plan.md` |

### 2026-07-07 coding-agent active plan refresh

This pass compared the active coding-agent short-term files against the current
codebase, completed Phase 2 and Phase 3 records, and the coding-agent ICDs.

| Outcome | Plans |
|---|---|
| Refreshed active draft against current codebase | `coding_agent_phase2_5_security_boundary_plan.md`; current production flow uses review-package materialization, while legacy validation helper code still exposes generated-test execution |
| Completed by user-approved fallback execution without subagent | `coding_agent_phase2_5_security_boundary_plan.md`; removed the generated-test execution helper boundary, aligned coding-agent architecture/ICDs, and verified the inert review-materialization boundary |
| Moved from active short-term to completed archive | `coding_agent_phase2_chat_input_queue_role_io_contract.md`; this was Gate 02 supporting role-contract evidence for completed Phase 2, not an executable active plan |
| Removed stale active registry row | `coding_agent_phase1_fetching_reading_plan.md`; the file is absent from active short-term and Phase 1 implementation records are already archived as completed |

### 2026-07-08 coding-agent Phase 4 completion

The coding-agent Phase 4 code modifying and patching plan completed after the
no-subagent execution path added the `code_modifying` and `code_patching`
boundaries, deterministic contract tests, role-level live LLM evidence, five
public E2E live LLM gates, final review remediation, and documentation
closeout.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `coding_agent_phase4_code_modifying_and_patching_plan.md` |

### 2026-07-08 coding-agent Phase 6 completion

The coding-agent Phase 6 code executing plan completed after user-approved
fallback execution without subagents. The implementation added a bounded
`code_executing` direct API for Phase 5 managed apply workspaces, deterministic
safety tests, one-at-a-time live LLM execution gates, no-subagent review
remediation, documentation updates, and lifecycle closeout.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `coding_agent_phase6_code_executing_plan.md` |

### 2026-07-08 coding-agent Phase 8 completion

The coding-agent Phase 8 verify/repair loop plan completed after user-approved
fallback execution without subagents. The implementation added the direct
trusted `verify_and_repair_code_change(...)` API, deterministic repair
contracts, six one-at-a-time real LLM gates with committed raw/review evidence,
review remediation, documentation updates, and archived closeout evidence.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `coding_agent_phase8_verify_repair_loop_plan.md` |

## Archive

Completed and superseded records live under `archive/`. Use them for historical
lookup, rationale, and execution evidence. New work must not be added to archived
plans.

### Completed Bugfix Records

| Plan |
|---|
| [adapter_semantic_identity_boundary_and_memory_pollution_plan.md](archive/completed/bugfix/adapter_semantic_identity_boundary_and_memory_pollution_plan.md) |
| [character_state_lane_integrity_plan.md](archive/completed/bugfix/character_state_lane_integrity_plan.md) |
| [character_self_image_rolling_state_bugfix_plan.md](archive/completed/bugfix/character_self_image_rolling_state_bugfix_plan.md) |
| [conversation_episode_state_lane_lifecycle_plan.md](archive/completed/bugfix/conversation_episode_state_lane_lifecycle_plan.md) |
| [control_console_functional_remediation_plan.md](archive/completed/bugfix/control_console_functional_remediation_plan.md) |
| [control_console_information_architecture_remediation_plan.md](archive/completed/bugfix/control_console_information_architecture_remediation_plan.md) |
| [control_console_ui_e2e_acceptance_test_plan.md](archive/completed/bugfix/control_console_ui_e2e_acceptance_test_plan.md) |
| [cognition_silence_short_circuit_and_dialog_evaluator_quality_plan.md](archive/completed/bugfix/cognition_silence_short_circuit_and_dialog_evaluator_quality_plan.md) |
| [coding_agent_source_intake_resolution_plan.md](archive/completed/bugfix/coding_agent_source_intake_resolution_plan.md) |
| [coding_agent_inline_source_bundle_bugfix_plan.md](archive/completed/bugfix/coding_agent_inline_source_bundle_bugfix_plan.md) |
| [consolidation_module_boundary_migration_bugfix_plan.md](archive/completed/bugfix/consolidation_module_boundary_migration_bugfix_plan.md) |
| [consolidator_facts_prompt_budget_bugfix_plan.md](archive/completed/bugfix/consolidator_facts_prompt_budget_bugfix_plan.md) |
| [consolidator_lane_router_memory_pollution_bigbang_plan.md](archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md) |
| [conversation_progress_identity_leakage_bugfix_plan.md](archive/completed/bugfix/conversation_progress_identity_leakage_bugfix_plan.md) |
| [cross_thread_image_contamination_bugfix_plan.md](archive/completed/bugfix/cross_thread_image_contamination_bugfix_plan.md) |
| [decontextualizer_scope_users_referent_bugfix_plan.md](archive/completed/bugfix/decontextualizer_scope_users_referent_bugfix_plan.md) |
| [dialog_anchor_authority_stale_history_bugfix_plan.md](archive/completed/bugfix/dialog_anchor_authority_stale_history_bugfix_plan.md) |
| [dialog_evaluator_decommission_plan.md](archive/completed/bugfix/dialog_evaluator_decommission_plan.md) |
| [dialog_evaluator_guess_owner_boundary_bugfix_plan.md](archive/completed/bugfix/dialog_evaluator_guess_owner_boundary_bugfix_plan.md) |
| [dialog_one_bubble_layout_contract_bugfix_plan.md](archive/completed/bugfix/dialog_one_bubble_layout_contract_bugfix_plan.md) |
| [generic_pipeline_cancellation_channel_guard_plan.md](archive/completed/bugfix/generic_pipeline_cancellation_channel_guard_plan.md) |
| [generic_cognition_prompt_migration_plan.md](archive/completed/bugfix/generic_cognition_prompt_migration_plan.md) |
| [group_scene_digest_explicit_participants_bugfix_plan.md](archive/completed/bugfix/group_scene_digest_explicit_participants_bugfix_plan.md) |
| [history_media_projection_image_boundary_plan.md](archive/completed/bugfix/history_media_projection_image_boundary_plan.md) |
| [interaction_style_images_lane_data_integrity_plan.md](archive/completed/bugfix/interaction_style_images_lane_data_integrity_plan.md) |
| [l3_content_anchor_open_loop_resolution_plan.md](archive/completed/bugfix/l3_content_anchor_open_loop_resolution_plan.md) |
| [llm_semantic_descriptor_validation_bugfix_plan.md](archive/completed/bugfix/llm_semantic_descriptor_validation_bugfix_plan.md) |
| [l3_dialog_content_plan_contract_bugfix_plan.md](archive/completed/bugfix/l3_dialog_content_plan_contract_bugfix_plan.md) |
| [lm_studio_model_unload_retry_bugfix_plan.md](archive/completed/bugfix/lm_studio_model_unload_retry_bugfix_plan.md) |
| [logical_dialog_message_receipt_plan.md](archive/completed/bugfix/logical_dialog_message_receipt_plan.md) |
| [memory_lifecycle_specialist_routing_plan.md](archive/completed/bugfix/memory_lifecycle_specialist_routing_plan.md) |
| [no_due_commitment_lifecycle_resolution_plan.md](archive/completed/bugfix/no_due_commitment_lifecycle_resolution_plan.md) |
| [rag2_cognition_identity_evidence_content_bugfix_plan.md](archive/completed/bugfix/rag2_cognition_identity_evidence_content_bugfix_plan.md) |
| [quote_aware_rag_sequence_plan.md](archive/completed/bugfix/quote_aware_rag_sequence_plan.md) |
| [qq_face_projection_empty_input_guard_bugfix_plan.md](archive/completed/bugfix/qq_face_projection_empty_input_guard_bugfix_plan.md) |
| [qq_replied_image_description_unavailable_queue_prune_bugfix_plan.md](archive/completed/bugfix/qq_replied_image_description_unavailable_queue_prune_bugfix_plan.md) |
| [rag_active_turn_conversation_row_exclusion_plan.md](archive/completed/bugfix/rag_active_turn_conversation_row_exclusion_plan.md) |
| [rag_hybrid_search_time_config_plan.md](archive/completed/bugfix/rag_hybrid_search_time_config_plan.md) |
| [rag_memory_evidence_remember_me_inner_path_bugfix_plan.md](archive/completed/bugfix/rag_memory_evidence_remember_me_inner_path_bugfix_plan.md) |
| [rag_conversation_evidence_current_episode_boundary_bugfix_plan.md](archive/completed/bugfix/rag_conversation_evidence_current_episode_boundary_bugfix_plan.md) |
| [rag_retrieval_top_k_embedding_tuning_plan.md](archive/completed/bugfix/rag_retrieval_top_k_embedding_tuning_plan.md) |
| [reflection_group_scene_digest_self_cognition_bugfix_plan.md](archive/completed/bugfix/reflection_group_scene_digest_self_cognition_bugfix_plan.md) |
| [reflection_global_promotion_replay_bugfix_plan.md](archive/completed/bugfix/reflection_global_promotion_replay_bugfix_plan.md) |
| [resolver_image_only_empty_input_bugfix_plan.md](archive/completed/bugfix/resolver_image_only_empty_input_bugfix_plan.md) |
| [self_cognition_background_context_budget_bugfix_plan.md](archive/completed/bugfix/self_cognition_background_context_budget_bugfix_plan.md) |
| [self_cognition_group_digest_context_evidence_bugfix_plan.md](archive/completed/bugfix/self_cognition_group_digest_context_evidence_bugfix_plan.md) |
| [self_cognition_character_global_id_config_bugfix_plan.md](archive/completed/bugfix/self_cognition_character_global_id_config_bugfix_plan.md) |
| [self_cognition_dialog_state_contract_bugfix_plan.md](archive/completed/bugfix/self_cognition_dialog_state_contract_bugfix_plan.md) |
| [self_cognition_group_thread_subject_boundary_bugfix_plan.md](archive/completed/bugfix/self_cognition_group_thread_subject_boundary_bugfix_plan.md) |
| [self_cognition_group_speak_selection_bugfix_plan.md](archive/completed/bugfix/self_cognition_group_speak_selection_bugfix_plan.md) |
| [self_cognition_sleep_period_plan.md](archive/completed/bugfix/self_cognition_sleep_period_plan.md) |
| [self_cognition_speak_delivery_bugfix_plan.md](archive/completed/bugfix/self_cognition_speak_delivery_bugfix_plan.md) |
| [self_other_inversion_personality_question_bugfix_plan.md](archive/completed/bugfix/self_other_inversion_personality_question_bugfix_plan.md) |
| [shared_memory_lane_data_integrity_plan.md](archive/completed/bugfix/shared_memory_lane_data_integrity_plan.md) |
| [task_dispatcher_json_contract_bugfix_plan.md](archive/completed/bugfix/task_dispatcher_json_contract_bugfix_plan.md) |
| [temporal_grounding_rag_episode_state_plan.md](archive/completed/bugfix/temporal_grounding_rag_episode_state_plan.md) |
| [text_chat_current_event_grounding_bugfix_plan.md](archive/completed/bugfix/text_chat_current_event_grounding_bugfix_plan.md) |
| [user_memory_units_lane_data_integrity_plan.md](archive/completed/bugfix/user_memory_units_lane_data_integrity_plan.md) |
| [user_profiles_lane_data_integrity_plan.md](archive/completed/bugfix/user_profiles_lane_data_integrity_plan.md) |
| [time_source_boundary_bugfix_plan.md](archive/completed/bugfix/time_source_boundary_bugfix_plan.md) |

### Completed Short-Term Records

| Plan |
|---|
| [backend_control_console_development_plan.md](archive/completed/short_term/backend_control_console_development_plan.md) |
| [backend_control_console_web_test_plan.md](archive/completed/short_term/backend_control_console_web_test_plan.md) |
| [background_artifact_handoff_poc_plan.md](archive/completed/short_term/background_artifact_handoff_poc_plan.md) |
| [background_work_semantic_lifecycle_plan.md](archive/completed/short_term/background_work_semantic_lifecycle_plan.md) |
| [cache2_agent_stats_health_plan.md](archive/completed/short_term/cache2_agent_stats_health_plan.md) |
| [channel_name_semantic_projection_plan.md](archive/completed/short_term/channel_name_semantic_projection_plan.md) |
| [character_local_time_context_plan.md](archive/completed/short_term/character_local_time_context_plan.md) |
| [character_profile_runtime_state_split_plan.md](archive/completed/short_term/character_profile_runtime_state_split_plan.md) |
| [coding_agent_phase0_fetching_plan.md](archive/completed/short_term/coding_agent_phase0_fetching_plan.md) |
| [coding_agent_phase1_code_reading_final_plan.md](archive/completed/short_term/coding_agent_phase1_code_reading_final_plan.md) |
| [coding_agent_phase1_real_repo_retrieval_remediation_plan.md](archive/completed/short_term/coding_agent_phase1_real_repo_retrieval_remediation_plan.md) |
| [coding_agent_phase2_chat_input_queue_role_io_contract.md](archive/completed/short_term/coding_agent_phase2_chat_input_queue_role_io_contract.md) |
| [coding_agent_phase2_5_security_boundary_plan.md](archive/completed/short_term/coding_agent_phase2_5_security_boundary_plan.md) |
| [coding_agent_phase2_code_writing_plan.md](archive/completed/short_term/coding_agent_phase2_code_writing_plan.md) |
| [coding_agent_phase3_background_worker_integration_plan.md](archive/completed/short_term/coding_agent_phase3_background_worker_integration_plan.md) |
| [coding_agent_phase4_code_modifying_and_patching_plan.md](archive/completed/short_term/coding_agent_phase4_code_modifying_and_patching_plan.md) |
| [coding_agent_phase5_patch_apply_plan.md](archive/completed/short_term/coding_agent_phase5_patch_apply_plan.md) |
| [coding_agent_phase6_code_executing_plan.md](archive/completed/short_term/coding_agent_phase6_code_executing_plan.md) |
| [coding_agent_phase7_existing_source_planning_plan.md](archive/completed/short_term/coding_agent_phase7_existing_source_planning_plan.md) |
| [coding_agent_phase8_verify_repair_loop_plan.md](archive/completed/short_term/coding_agent_phase8_verify_repair_loop_plan.md) |
| [character_reflection_cycle_stage1a_plan.md](archive/completed/short_term/character_reflection_cycle_stage1a_plan.md) |
| [character_self_words_retrieval_delivery_receipt_plan.md](archive/completed/short_term/character_self_words_retrieval_delivery_receipt_plan.md) |
| [cognition_chain_module_separation_plan.md](archive/completed/short_term/cognition_chain_module_separation_plan.md) |
| [cognition_llm_stage_reconnection_plan.md](archive/completed/short_term/cognition_llm_stage_reconnection_plan.md) |
| [cognition_visual_directives_control_plan.md](archive/completed/short_term/cognition_visual_directives_control_plan.md) |
| [cognition_state_integrity_plan.md](archive/completed/short_term/cognition_state_integrity_plan.md) |
| [daily_affect_settling_plan.md](archive/completed/short_term/daily_affect_settling_plan.md) |
| [complex_task_resolver_capability_plan.md](archive/completed/short_term/complex_task_resolver_capability_plan.md) |
| [control_console_auto_model_discovery_picker_plan.md](archive/completed/short_term/control_console_auto_model_discovery_picker_plan.md) |
| [control_console_brain_model_route_config_plan.md](archive/completed/short_term/control_console_brain_model_route_config_plan.md) |
| [control_console_cognition_debug_visibility_plan.md](archive/completed/short_term/control_console_cognition_debug_visibility_plan.md) |
| [control_console_entity_information_architecture_plan.md](archive/completed/short_term/control_console_entity_information_architecture_plan.md) |
| [control_console_live_logs_plan.md](archive/completed/short_term/control_console_live_logs_plan.md) |
| [control_console_runtime_service_config_plan.md](archive/completed/short_term/control_console_runtime_service_config_plan.md) |
| [consolidator_text_dispatch_decommission_plan.md](archive/completed/short_term/consolidator_text_dispatch_decommission_plan.md) |
| [consolidation_evidence_hardening_plan.md](archive/completed/short_term/consolidation_evidence_hardening_plan.md) |
| [consolidation_target_routing_architecture_plan.md](archive/completed/short_term/consolidation_target_routing_architecture_plan.md) |
| [conversation_progress_flow_phase2_plan.md](archive/completed/short_term/conversation_progress_flow_phase2_plan.md) |
| [conversation_progress_phase3_quality_plan.md](archive/completed/short_term/conversation_progress_phase3_quality_plan.md) |
| [conversation_progress_state_plan.md](archive/completed/short_term/conversation_progress_state_plan.md) |
| [dialog_mention_target_user_plan.md](archive/completed/short_term/dialog_mention_target_user_plan.md) |
| [dialog_message_sequence_delivery_plan.md](archive/completed/short_term/dialog_message_sequence_delivery_plan.md) |
| [documentation_harmonization_bigbang_plan.md](archive/completed/short_term/documentation_harmonization_bigbang_plan.md) |
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
| [inline_delivery_mentions_plan.md](archive/completed/short_term/inline_delivery_mentions_plan.md) |
| [live_context_runtime_facts_plan.md](archive/completed/short_term/live_context_runtime_facts_plan.md) |
| [llm_trace_observability_and_retrieval_plan.md](archive/completed/short_term/llm_trace_observability_and_retrieval_plan.md) |
| [llm_routing_migration_plan.md](archive/completed/short_term/llm_routing_migration_plan.md) |
| [llm_interface_backend_abstraction_plan.md](archive/completed/short_term/llm_interface_backend_abstraction_plan.md) |
| [l2_affinity_willingness_boundary_plan.md](archive/completed/short_term/l2_affinity_willingness_boundary_plan.md) |
| [l2d_action_router_prompt_separation_plan.md](archive/completed/short_term/l2d_action_router_prompt_separation_plan.md) |
| [l2d_l3_surface_handoff_plan.md](archive/completed/short_term/l2d_l3_surface_handoff_plan.md) |
| [l2d_router_split_and_background_ack_plan.md](archive/completed/short_term/l2d_router_split_and_background_ack_plan.md) |
| [media_descriptor_cache_plan.md](archive/completed/short_term/media_descriptor_cache_plan.md) |
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
| [past_dialog_cognition_residual_plan.md](archive/completed/short_term/past_dialog_cognition_residual_plan.md) |
| [prompt_prefix_and_input_format_optimization_plan.md](archive/completed/short_term/prompt_prefix_and_input_format_optimization_plan.md) |
| [prompt_safe_message_context_plan.md](archive/completed/short_term/prompt_safe_message_context_plan.md) |
| [qwen_thinking_support_plan.md](archive/completed/short_term/qwen_thinking_support_plan.md) |
| [qq_adapter_readable_mentions_plan.md](archive/completed/short_term/qq_adapter_readable_mentions_plan.md) |
| [rag1_decommission_plan.md](archive/completed/short_term/rag1_decommission_plan.md) |
| [rag_2_1_initializer_subagent_contract_plan.md](archive/completed/short_term/rag_2_1_initializer_subagent_contract_plan.md) |
| [rag2_cognition_ready_evidence_plan.md](archive/completed/short_term/rag2_cognition_ready_evidence_plan.md) |
| [rag2_mainline_fusion_recall_quality_plan.md](archive/completed/short_term/rag2_mainline_fusion_recall_quality_plan.md) |
| [rag2_phase4_continuation_plan.md](archive/completed/short_term/rag2_phase4_continuation_plan.md) |
| [rag3_local_context_resolver_bigbang_plan.md](archive/completed/short_term/rag3_local_context_resolver_bigbang_plan.md) |
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
| [universal_chat_history_llm_projection_plan.md](archive/completed/short_term/universal_chat_history_llm_projection_plan.md) |
| [user_style_engagement_consumer_plan.md](archive/completed/short_term/user_style_engagement_consumer_plan.md) |
| [user_memory_unit_rolling_window_plan.md](archive/completed/short_term/user_memory_unit_rolling_window_plan.md) |
| [web_agent3_bilibili_source_subagent_plan.md](archive/completed/short_term/web_agent3_bilibili_source_subagent_plan.md) |
| [web_agent3_source_availability_bigbang_plan.md](archive/completed/short_term/web_agent3_source_availability_bigbang_plan.md) |
| [web_agent3_search_attempt_expansion_and_resolver_evidence_decomposition_plan.md](archive/completed/short_term/web_agent3_search_attempt_expansion_and_resolver_evidence_decomposition_plan.md) |
| [web_agent3_replacement_plan.md](archive/completed/short_term/web_agent3_replacement_plan.md) |

### Superseded Records

| Plan |
|---|
| [character_reflection_cycle_stage1_plan.md](archive/superseded/character_reflection_cycle_stage1_plan.md) |
| [conversation_graph_recent_context_plan.md](archive/superseded/conversation_graph_recent_context_plan.md) |
| [cognition_preserving_goal_resolver_production_plan.md](archive/superseded/cognition_preserving_goal_resolver_production_plan.md) |
| [graph_rag_recall_experiment_plan.md](archive/superseded/graph_rag_recall_experiment_plan.md) |
| [goal_resolver_poc_plan.md](archive/superseded/goal_resolver_poc_plan.md) |
| [self_cognition_loop_architecture.md](archive/superseded/self_cognition_loop_architecture.md) |
| [self_cognition_reasoning_basis.md](archive/superseded/self_cognition_reasoning_basis.md) |
| [self_cognition_tracking_icd.md](archive/superseded/self_cognition_tracking_icd.md) |
| [rag_supervisor2_inner_loop_agents_plan.md](archive/superseded/rag_supervisor2_inner_loop_agents_plan.md) |
| [short_circuit_early_stop_plan.md](archive/superseded/short_circuit_early_stop_plan.md) |
| [coding_agent_phase2_code_writing_plan_superseded_20260623.md](archive/superseded/coding_agent_phase2_code_writing_plan_superseded_20260623.md) |
| [visual_descriptor_seeded_reference_images_plan.md](archive/superseded/visual_descriptor_seeded_reference_images_plan.md) |

