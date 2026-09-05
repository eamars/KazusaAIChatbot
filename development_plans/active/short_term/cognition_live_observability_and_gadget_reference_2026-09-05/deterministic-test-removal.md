# Completed deterministic test removal

Date: 2026-09-05. Status: complete for the separately authorized preparation step. The [production development plan](../cognition_live_observability_and_gadget_plan_2026-09-05.md) remains draft.

Removed all 327 identified affected deterministic test definitions across 48 files before production development: 21 whole files deleted, with selected functions removed from 27 other files. Existing real LLM cases and shared support were preserved. No replacement tests or production changes were made.

## Scope and identification

The user explicitly requested identification and deletion of every deterministic test impacted by the live cognition observation and gadget development, while leaving real LLM tests as is. The scope is the affected tests for this plan; unrelated deterministic tests remain.

Inspected 404 Python test/support files containing 3,285 test definitions at baseline commit `f1c6f06f5b781391bf80d7df45b2c3b5fbda2b99`. Matched the plan's changed functions, observation DTOs, producer/consumer contracts, console markup and rendering, source/document assertions, fake Brain fixtures, and test-helper dependencies. Reviewed the candidates at function level rather than deleting every test that imports a large service module.

Deterministic coverage includes patched model calls, controlled stage responses, fake HTTP/browser integration, static source/DOM assertions and read-only console layout checks. Classification follows actual test behavior and markers: the fake-Brain test named `test_live_debug_and_self_views_share_observation_section_layout` was deterministic; the ten marked real-model cases in the mixed decontextualizer module were preserved.

Pure tests for unchanged prompt/input schemas, state transactions, capability validation, RAG retrieval, persistence, adapter behavior, and unrelated console functions remain where they do not exercise a changed boundary or depend on a retired contract. All 15 former EXISTING test references in the plan's matrix are included in this removal.

Counts refer to Python test functions/methods, before parameter expansion. The [machine-readable inventory](deterministic-test-removal.json) contains every removed node ID, the identified dependency/reason, baseline line spans, before/after file hashes, and preservation evidence.

## Preservation and verification

- All 383 remaining Python test/support files parse successfully.
- The remaining definition inventory contains exactly 2,958 tests: the original set minus the 327 selected removals.
- SHA-256 checks confirm the 65 whole files selected for live-suite/support protection are unchanged. This protection set also includes mixed or support modules preserved conservatively.
- All ten real LLM functions in `tests/test_decontextualizer_referents.py`, including their decorators, retain their exact bytes. Its six deterministic cases were removed.
- All retained top-level definitions in edited files match the baseline content, including 188 helper/class definitions. This preserves the `_input` helper in `tests/unit/cognition_core_v3/test_handleless_contract.py` used by a real LLM suite.
- The other 356 Python test/support files retain their full original hashes. Shared fixtures and `conftest.py` files remain intact.
- Static import inspection found zero remaining imports to deleted test modules.
- The production source diff is empty. Both HTML references and their screenshots retain the hashes in [manifest.json](manifest.json).

Verification used syntax parsing, exact definition-set comparison, import inspection, hashes and retained-source comparison. Pytest cases and real LLM requests were not executed for this deletion-only preparation; these checks establish removal and preservation, rather than runtime product behavior.

## Implementation handoff

The [amended test matrix](test-impact.md) retires the former EXISTING gates. It still describes future NEW acceptance checks for the draft development plan. Implement those from observable lifecycle, transport, topology, disclosure and failure outcomes after the early producer-to-browser probe. The deleted implementation assertions supply no requirement to restore old structure.

Preserve the existing real LLM cases and support. Four live integration suites retain observation reset/fixture references identified in the matrix; any later fixture-only changes require separate identification and authorization. Keep the original HTML mockups frozen as implementation references. Production development remains pending.

## File inventory

| Test file | Removed definitions | Remaining definitions | Action |
|---|---:|---:|---|
| `tests/control_console_e2e/test_cognition_observability_e2e.py` | 3 | 0 | File deleted |
| `tests/control_console_e2e/test_debug_chat_e2e.py` | 3 | 0 | File deleted |
| `tests/control_console_e2e/test_error_paths_e2e.py` | 1 | 0 | File deleted |
| `tests/control_console_e2e/test_page_navigation_e2e.py` | 1 | 6 | Selected functions removed |
| `tests/control_console_e2e/test_running_console_signoff_e2e.py` | 1 | 0 | File deleted |
| `tests/control_console_e2e/test_stage3_fresh_database_e2e.py` | 1 | 0 | File deleted |
| `tests/control_console_e2e/test_visual_product_acceptance_e2e.py` | 1 | 0 | File deleted |
| `tests/test_cognition_observability_docs.py` | 6 | 0 | File deleted |
| `tests/test_cognition_preference_adapter.py` | 2 | 0 | File deleted |
| `tests/test_cognition_resolver_loop.py` | 20 | 12 | Selected functions removed |
| `tests/test_cognition_resolver_persona_graph.py` | 1 | 0 | File deleted |
| `tests/test_console_debug_chat.py` | 3 | 0 | File deleted |
| `tests/test_console_lookup_limits.py` | 1 | 0 | File deleted |
| `tests/test_control_console_bootstrap.py` | 6 | 0 | File deleted |
| `tests/test_control_console_cognition_debug_visibility.py` | 4 | 8 | Selected functions removed |
| `tests/test_control_console_contracts.py` | 2 | 0 | File deleted |
| `tests/test_control_console_kazusa_client.py` | 5 | 0 | File deleted |
| `tests/test_control_console_redaction.py` | 2 | 0 | File deleted |
| `tests/test_control_console_repository.py` | 1 | 18 | Selected functions removed |
| `tests/test_control_console_review_edges.py` | 1 | 7 | Selected functions removed |
| `tests/test_control_console_stream.py` | 9 | 0 | File deleted |
| `tests/test_control_console_web_surface.py` | 10 | 8 | Selected functions removed |
| `tests/test_conversation_progress_group_scene.py` | 1 | 16 | Selected functions removed |
| `tests/test_decontextualizer_referents.py` | 6 | 10 | Selected functions removed |
| `tests/test_dialog_mention_target_user.py` | 3 | 0 | File deleted |
| `tests/test_memory_lifecycle_specialist.py` | 6 | 5 | Selected functions removed |
| `tests/test_msg_decontextualizer.py` | 23 | 4 | Selected functions removed |
| `tests/test_multi_source_cognition_image_input.py` | 1 | 2 | Selected functions removed |
| `tests/test_rag_dialog_event_logging.py` | 2 | 3 | Selected functions removed |
| `tests/test_real_history_personality_fixture_contract.py` | 2 | 6 | Selected functions removed |
| `tests/test_reflection_cycle_stage1c_service.py` | 7 | 2 | Selected functions removed |
| `tests/test_self_cognition_event_logging.py` | 3 | 2 | Selected functions removed |
| `tests/test_self_cognition_group_review_source.py` | 1 | 32 | Selected functions removed |
| `tests/test_self_cognition_integration.py` | 36 | 20 | Selected functions removed |
| `tests/test_self_cognition_tracking.py` | 25 | 25 | Selected functions removed |
| `tests/test_service_background_consolidation.py` | 23 | 12 | Selected functions removed |
| `tests/test_service_event_logging.py` | 7 | 1 | Selected functions removed |
| `tests/test_service_input_queue.py` | 35 | 29 | Selected functions removed |
| `tests/unit/brain_service/test_cognition_graph_projection.py` | 10 | 0 | File deleted |
| `tests/unit/cognition_core_v3/test_handleless_contract.py` | 2 | 20 | Selected functions removed |
| `tests/unit/cognition_core_v3/test_stage_recovery.py` | 11 | 0 | File deleted |
| `tests/unit/cognition_core_v3/test_state_transaction.py` | 3 | 8 | Selected functions removed |
| `tests/unit/cognition_observability/test_contracts.py` | 2 | 0 | File deleted |
| `tests/unit/cognition_observability/test_projection.py` | 11 | 0 | File deleted |
| `tests/unit/nodes/test_dialog_agent.py` | 8 | 1 | Selected functions removed |
| `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py` | 6 | 8 | Selected functions removed |
| `tests/unit/nodes/test_persona_supervisor2_l3_surface.py` | 7 | 16 | Selected functions removed |
| `tests/unit/nodes/test_persona_supervisor2_schema.py` | 2 | 2 | Selected functions removed |

## Exact removed definitions

### tests/control_console_e2e/test_cognition_observability_e2e.py

- `tests/control_console_e2e/test_cognition_observability_e2e.py::test_live_debug_and_self_views_share_observation_section_layout`
- `tests/control_console_e2e/test_cognition_observability_e2e.py::test_prewarm_and_context_sources_render_status_counts_and_omissions`
- `tests/control_console_e2e/test_cognition_observability_e2e.py::test_canonical_sequence_and_reference_edges_render`

### tests/control_console_e2e/test_debug_chat_e2e.py

- `tests/control_console_e2e/test_debug_chat_e2e.py::test_debug_chat_sends_to_brain_and_updates_history_and_graph`
- `tests/control_console_e2e/test_debug_chat_e2e.py::test_debug_chat_click_shows_live_running_state_before_response`
- `tests/control_console_e2e/test_debug_chat_e2e.py::test_debug_chat_abort_renders_network_failure_without_graph`

### tests/control_console_e2e/test_error_paths_e2e.py

- `tests/control_console_e2e/test_error_paths_e2e.py::test_debug_chat_error_paths_are_visible_and_actionable`

### tests/control_console_e2e/test_page_navigation_e2e.py

- `tests/control_console_e2e/test_page_navigation_e2e.py::test_each_sidebar_page_has_connected_or_explicitly_gated_state`

### tests/control_console_e2e/test_running_console_signoff_e2e.py

- `tests/control_console_e2e/test_running_console_signoff_e2e.py::test_running_console_information_contract_matrix`

### tests/control_console_e2e/test_stage3_fresh_database_e2e.py

- `tests/control_console_e2e/test_stage3_fresh_database_e2e.py::test_stage3_fresh_database_graph_and_debug_handoff`

### tests/control_console_e2e/test_visual_product_acceptance_e2e.py

- `tests/control_console_e2e/test_visual_product_acceptance_e2e.py::test_desktop_visual_acceptance_for_cards_buttons_and_branding`

### tests/test_cognition_observability_docs.py

- `tests/test_cognition_observability_docs.py::test_icd_and_runtime_docs_name_one_brain_service_contract_owner`
- `tests/test_cognition_observability_docs.py::test_process_local_observation_and_future_persisted_chain_run_are_distinct`
- `tests/test_cognition_observability_docs.py::test_howto_documents_canonical_observation_and_browser_checks`
- `tests/test_cognition_observability_docs.py::test_runtime_readmes_document_prewarm_and_observation_carriers`
- `tests/test_cognition_observability_docs.py::test_icd_documents_relevance_diagnostic_envelope_flow`
- `tests/test_cognition_observability_docs.py::test_icd_documents_live_response_recovery_dispositions`

### tests/test_cognition_preference_adapter.py

- `tests/test_cognition_preference_adapter.py::test_surface_projection_owns_visible_boundaries_deterministically`
- `tests/test_cognition_preference_adapter.py::test_surface_projection_has_no_keyword_based_user_input_adapter`

### tests/test_cognition_resolver_loop.py

- `tests/test_cognition_resolver_loop.py::test_invalid_required_dependency_fails_closed_before_l3`
- `tests/test_cognition_resolver_loop.py::test_pending_background_goal_reaches_acknowledgement_without_factual_surface`
- `tests/test_cognition_resolver_loop.py::test_answerable_now_terminates_without_executing_optional_resolver`
- `tests/test_cognition_resolver_loop.py::test_answerable_now_is_independent_of_unresolved_conversation_source`
- `tests/test_cognition_resolver_loop.py::test_loop_records_timeout_observation_then_returns_to_cognition`
- `tests/test_cognition_resolver_loop.py::test_task_resolution_uses_task_service_timeout_without_detached_resolver_task`
- `tests/test_cognition_resolver_loop.py::test_loop_blocks_same_capability_retry_after_timeout`
- `tests/test_cognition_resolver_loop.py::test_duplicate_final_cognition_internal_thought_stays_private`
- `tests/test_cognition_resolver_loop.py::test_max_cycle_internal_thought_request_stays_private`
- `tests/test_cognition_resolver_loop.py::test_hil_blocked_observation_persists_pending_and_reenters_cognition`
- `tests/test_cognition_resolver_loop.py::test_hil_repeated_after_pending_surfaces_pending_question`
- `tests/test_cognition_resolver_loop.py::test_hil_pending_without_action_surfaces_pending_question`
- `tests/test_cognition_resolver_loop.py::test_same_message_pending_resolution_is_ignored`
- `tests/test_cognition_resolver_loop.py::test_same_message_terminal_action_closes_pending_resolution`
- `tests/test_cognition_resolver_loop.py::test_resolver_telemetry_is_sanitized_and_stage_readable`
- `tests/test_cognition_resolver_loop.py::test_resolver_human_readable_trace_is_prompt_safe`
- `tests/test_cognition_resolver_loop.py::test_approval_blocked_observation_persists_pending_without_side_effect`
- `tests/test_cognition_resolver_loop.py::test_pending_resolution_is_applied_only_after_l2d_decision`
- `tests/test_cognition_resolver_loop.py::test_pending_unrelated_turn_can_continue_waiting_without_task_admission`
- `tests/test_cognition_resolver_loop.py::test_user_input_blocker_converges_after_one_final_cognition`

### tests/test_cognition_resolver_persona_graph.py

- `tests/test_cognition_resolver_persona_graph.py::test_persona_graph_has_one_v2_resolver_path`

### tests/test_console_debug_chat.py

- `tests/test_console_debug_chat.py::test_debug_chat_returns_brain_unavailable_without_cognition_when_stopped`
- `tests/test_console_debug_chat.py::test_debug_chat_uses_live_unmanaged_brain_endpoint`
- `tests/test_console_debug_chat.py::test_debug_chat_rejects_stale_unowned_brain_conflict`

### tests/test_console_lookup_limits.py

- `tests/test_console_lookup_limits.py::test_lookup_routes_enforce_pagination_redaction_and_no_embeddings`

### tests/test_control_console_bootstrap.py

- `tests/test_control_console_bootstrap.py::test_bootstrap_wraps_canonical_observations_with_view_metadata`
- `tests/test_control_console_bootstrap.py::test_overview_api_returns_only_owner_aggregates`
- `tests/test_control_console_bootstrap.py::test_bootstrap_returns_initial_state_session_csrf_services_and_stream_url`
- `tests/test_control_console_bootstrap.py::test_bootstrap_projects_live_health_without_overview_duplication`
- `tests/test_control_console_bootstrap.py::test_bootstrap_projects_live_health_when_brain_is_unmanaged`
- `tests/test_control_console_bootstrap.py::test_bootstrap_does_not_query_brain_for_stale_unowned_conflict`

### tests/test_control_console_cognition_debug_visibility.py

- `tests/test_control_console_cognition_debug_visibility.py::test_static_surface_exposes_semantic_v2_owner_panels`
- `tests/test_control_console_cognition_debug_visibility.py::test_static_renderers_tolerate_missing_optional_panel_targets`
- `tests/test_control_console_cognition_debug_visibility.py::test_static_renderers_do_not_write_inner_html_through_raw_selectors`
- `tests/test_control_console_cognition_debug_visibility.py::test_static_shell_dom_access_uses_guarded_helpers`

### tests/test_control_console_contracts.py

- `tests/test_control_console_contracts.py::test_service_contracts_reject_extra_fields_and_unbounded_strings`
- `tests/test_control_console_contracts.py::test_console_response_contract_uses_view_envelopes_for_bootstrap_and_debug`

### tests/test_control_console_kazusa_client.py

- `tests/test_control_console_kazusa_client.py::test_kazusa_client_reads_health_and_posts_debug_chat`
- `tests/test_control_console_kazusa_client.py::test_client_validates_canonical_cognition_observation_without_reprojection`
- `tests/test_control_console_kazusa_client.py::test_client_raises_protocol_error_for_invalid_observation_version`
- `tests/test_control_console_kazusa_client.py::test_client_rejects_invalid_latest_observation_without_reconstruction`
- `tests/test_control_console_kazusa_client.py::test_debug_client_returns_direct_response_observation_without_latest_fetch`

### tests/test_control_console_redaction.py

- `tests/test_control_console_redaction.py::test_responses_exclude_secrets_prompts_embeddings_env_values_and_raw_messages`
- `tests/test_control_console_redaction.py::test_canonical_observation_sections_bypass_legacy_semantic_reprojection`

### tests/test_control_console_repository.py

- `tests/test_control_console_repository.py::test_character_posture_consumes_canonical_context_observation_section`

### tests/test_control_console_review_edges.py

- `tests/test_control_console_review_edges.py::test_stream_parsing_replay_and_status_events_cover_failure_branches`

### tests/test_control_console_stream.py

- `tests/test_control_console_stream.py::test_stream_wait_returns_immediately_after_shutdown_signal`
- `tests/test_control_console_stream.py::test_stream_wait_times_out_when_shutdown_is_not_signaled`
- `tests/test_control_console_stream.py::test_summary_stream_emits_bounded_service_event_cursor_and_gap_payload`
- `tests/test_control_console_stream.py::test_stream_gap_forces_bootstrap_refetch`
- `tests/test_control_console_stream.py::test_numeric_cursor_before_replay_window_returns_gap`
- `tests/test_control_console_stream.py::test_stream_poll_appends_graph_invalidation_for_new_latest_run`
- `tests/test_control_console_stream.py::test_stream_poll_ignores_missing_run_id_and_client_errors`
- `tests/test_control_console_stream.py::test_stream_uses_nested_observation_run_id_and_ignores_empty_id`
- `tests/test_control_console_stream.py::test_stream_iterator_emits_graph_invalidation_and_heartbeat`

### tests/test_control_console_web_surface.py

- `tests/test_control_console_web_surface.py::test_debug_api_wraps_canonical_observation_without_reprojection`
- `tests/test_control_console_web_surface.py::test_latest_and_debug_protocol_errors_map_to_exact_view_availability`
- `tests/test_control_console_web_surface.py::test_cognition_observation_renderer_uses_contract_labels_and_shared_detail_layout`
- `tests/test_control_console_web_surface.py::test_renderer_accepts_unknown_producer_section_without_js_catalog`
- `tests/test_control_console_web_surface.py::test_debug_loading_and_error_states_are_separate_from_cognition_graph`
- `tests/test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs`
- `tests/test_control_console_web_surface.py::test_event_stream_refresh_does_not_reconnect_stream`
- `tests/test_control_console_web_surface.py::test_web_api_outputs_for_logs_events_audit_character_and_debug_error`
- `tests/test_control_console_web_surface.py::test_auth_optional_mode_and_invalid_hash_rejections`
- `tests/test_control_console_web_surface.py::test_app_uses_live_debug_chat_timeout`

### tests/test_conversation_progress_group_scene.py

- `tests/test_conversation_progress_group_scene.py::test_persona_supervisor_reports_group_scene_success_non_group_and_failure`

### tests/test_decontextualizer_referents.py

- `tests/test_decontextualizer_referents.py::test_decontextualizer_prompt_requires_character_name_and_identity_safe_examples`
- `tests/test_decontextualizer_referents.py::test_decontextualizer_projects_chat_history_as_transcript_lines`
- `tests/test_decontextualizer_referents.py::test_unresolved_reference_referent_flows`
- `tests/test_decontextualizer_referents.py::test_reply_excerpt_resolved_referent_flows`
- `tests/test_decontextualizer_referents.py::test_mixed_referents_are_preserved`
- `tests/test_decontextualizer_referents.py::test_malformed_referents_preserve_input_after_bounded_retry`

### tests/test_dialog_mention_target_user.py

- `tests/test_dialog_mention_target_user.py::test_dialog_generator_preserves_inline_tag_without_delivery_context`
- `tests/test_dialog_mention_target_user.py::test_dialog_generator_does_not_require_mention_flag`
- `tests/test_dialog_mention_target_user.py::test_dialog_agent_returns_no_mention_flag`

### tests/test_memory_lifecycle_specialist.py

- `tests/test_memory_lifecycle_specialist.py::test_post_surface_review_uses_final_dialog_and_direct_rows`
- `tests/test_memory_lifecycle_specialist.py::test_post_surface_review_leaves_ambiguous_dessert_open`
- `tests/test_memory_lifecycle_specialist.py::test_handler_consumes_route_and_materializes_apply_action`
- `tests/test_memory_lifecycle_specialist.py::test_provider_exhaustion_degrades_to_skipped_lifecycle_context`
- `tests/test_memory_lifecycle_specialist.py::test_specialist_repair_prompt_carries_contract_error`
- `tests/test_memory_lifecycle_specialist.py::test_post_surface_review_provider_failure_returns_empty_update`

### tests/test_msg_decontextualizer.py

- `tests/test_msg_decontextualizer.py::test_decontextualizer_attaches_role_explicit_meaning_to_episode`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_leaves_accepted_task_episode_source_owned`
- `tests/test_msg_decontextualizer.py::test_multimedia_descriptor_updates_prompt_context_and_current_row`
- `tests/test_msg_decontextualizer.py::test_multimedia_descriptor_continues_when_vision_llm_fails`
- `tests/test_msg_decontextualizer.py::test_multimedia_descriptor_retries_malformed_objects_before_cache`
- `tests/test_msg_decontextualizer.py::test_multimedia_descriptor_exhaustion_does_not_cache_fallback`
- `tests/test_msg_decontextualizer.py::test_multimedia_descriptor_ignores_invalid_cached_object`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_filtered_history_recreates_group_referent_loss`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_full_history_surfaces_group_referent_evidence`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_returns_modified_input`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_returns_original_when_not_modified`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_fallback_on_llm_error`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_fallback_on_malformed_json`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_trace_records_bounded_validation_error`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_actual_capsule_captures_validation_error`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_recovers_on_third_contract_attempt`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_repair_receives_exact_nested_field_error`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_forwards_reply_context_to_llm`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_projects_group_name_into_channel_topic_text`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_forwards_scope_users_as_neutral_identity_table`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_parses_unresolved_reference_signal`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_missing_new_fields_uses_original_input`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_llm_exception_warns_with_input_preview`

### tests/test_multi_source_cognition_image_input.py

- `tests/test_multi_source_cognition_image_input.py::test_quoted_image_description_enters_prompt_and_cognition_context`

### tests/test_rag_dialog_event_logging.py

- `tests/test_rag_dialog_event_logging.py::test_dialog_generator_records_llm_metadata_without_generated_text`
- `tests/test_rag_dialog_event_logging.py::test_dialog_agent_records_quality_without_dialog_text`

### tests/test_real_history_personality_fixture_contract.py

- `tests/test_real_history_personality_fixture_contract.py::test_private_monologue_uses_only_canonical_reasoning_node`
- `tests/test_real_history_personality_fixture_contract.py::test_private_monologue_fails_closed_when_canonical_node_is_missing`

### tests/test_reflection_cycle_stage1c_service.py

- `tests/test_reflection_cycle_stage1c_service.py::test_lifespan_starts_reflection_worker_by_default`
- `tests/test_reflection_cycle_stage1c_service.py::test_lifespan_does_not_start_reflection_worker_when_explicitly_disabled`
- `tests/test_reflection_cycle_stage1c_service.py::test_lifespan_stops_reflection_worker_on_shutdown`
- `tests/test_reflection_cycle_stage1c_service.py::test_lifespan_starts_self_cognition_worker_only_when_enabled`
- `tests/test_reflection_cycle_stage1c_service.py::test_lifespan_starts_calendar_scheduler_worker_by_default`
- `tests/test_reflection_cycle_stage1c_service.py::test_lifespan_does_not_start_calendar_scheduler_when_disabled`
- `tests/test_reflection_cycle_stage1c_service.py::test_reflection_probe_ignores_chat_queue_state`

### tests/test_self_cognition_event_logging.py

- `tests/test_self_cognition_event_logging.py::test_worker_records_target_binding_failure_without_dispatch_text`
- `tests/test_self_cognition_event_logging.py::test_worker_records_empty_source_collection_reason`
- `tests/test_self_cognition_event_logging.py::test_worker_synthesizes_missing_target_binding_failure_metadata`

### tests/test_self_cognition_group_review_source.py

- `tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_rejects_malformed_source_context`

### tests/test_self_cognition_integration.py

- `tests/test_self_cognition_integration.py::test_prepared_commitment_state_contains_public_group_scene`
- `tests/test_self_cognition_integration.py::test_worker_tick_marks_future_cognition_run_completed`
- `tests/test_self_cognition_integration.py::test_default_runner_allows_generic_future_cognition_without_authority`
- `tests/test_self_cognition_integration.py::test_worker_tick_marks_commitment_due_run_completed`
- `tests/test_self_cognition_integration.py::test_worker_tick_skips_stale_commitment_due_run`
- `tests/test_self_cognition_integration.py::test_worker_tick_skips_future_cognition_slot_when_claim_fails`
- `tests/test_self_cognition_integration.py::test_worker_tick_marks_state_contract_error_calendar_run_failed`
- `tests/test_self_cognition_integration.py::test_worker_tick_marks_unexpected_calendar_case_error_failed`
- `tests/test_self_cognition_integration.py::test_worker_selected_speak_dispatches_to_private_channel`
- `tests/test_self_cognition_integration.py::test_worker_selected_speak_dispatches_to_bound_group_source_channel`
- `tests/test_self_cognition_integration.py::test_worker_channel_capability_failure_blocks_before_history_write`
- `tests/test_self_cognition_integration.py::test_worker_missing_delivery_target_blocks_before_dialog`
- `tests/test_self_cognition_integration.py::test_worker_missing_delivery_target_blocks_without_adapter_provider`
- `tests/test_self_cognition_integration.py::test_worker_records_target_binding_failed_and_skips_calendar_run`
- `tests/test_self_cognition_integration.py::test_worker_selected_speak_never_records_not_requested`
- `tests/test_self_cognition_integration.py::test_worker_no_speak_does_not_dispatch`
- `tests/test_self_cognition_integration.py::test_worker_adapter_failure_marks_delivery_failed`
- `tests/test_self_cognition_integration.py::test_worker_empty_dialog_text_marks_delivery_failed`
- `tests/test_self_cognition_integration.py::test_worker_duplicate_suppression_marks_duplicate_suppressed`
- `tests/test_self_cognition_integration.py::test_worker_tick_records_state_contract_error_without_tick_failure`
- `tests/test_self_cognition_integration.py::test_worker_default_path_requests_production_consolidation_without_files`
- `tests/test_self_cognition_integration.py::test_worker_default_path_runs_prepared_case_without_state_contract_error`
- `tests/test_self_cognition_integration.py::test_worker_default_path_applies_consolidation_without_dispatch_or_files`
- `tests/test_self_cognition_integration.py::test_worker_default_path_records_action_without_dispatch`
- `tests/test_self_cognition_integration.py::test_worker_tick_loads_prior_attempts_before_running_case`
- `tests/test_self_cognition_integration.py::test_worker_tick_blocks_unbound_case_before_candidate_render`
- `tests/test_self_cognition_integration.py::test_worker_tick_suppresses_duplicate_due_occurrence_from_prior_attempts`
- `tests/test_self_cognition_integration.py::test_worker_tick_uses_attempt_updates_between_cases`
- `tests/test_self_cognition_integration.py::test_worker_tick_defers_when_primary_interaction_is_busy`
- `tests/test_self_cognition_integration.py::test_worker_tick_defers_pipeline_cancelled_case`
- `tests/test_self_cognition_integration.py::test_worker_tick_defer_requeues_claimed_source_calendar_run`
- `tests/test_self_cognition_integration.py::test_worker_tick_releases_pipeline_handle_when_claim_raises`
- `tests/test_self_cognition_integration.py::test_worker_tick_pauses_before_collection_for_affect_settling`
- `tests/test_self_cognition_integration.py::test_scheduled_case_never_renders_before_due`
- `tests/test_self_cognition_integration.py::test_scheduled_worker_dispatches_only_gate_accepted_candidate`
- `tests/test_self_cognition_integration.py::test_scheduled_worker_scrubs_gate_accepted_candidate_after_delivery_failure`

### tests/test_self_cognition_tracking.py

- `tests/test_self_cognition_tracking.py::test_default_self_cognition_client_uses_resolver_loop`
- `tests/test_self_cognition_tracking.py::test_before_due_commitment_writes_progress_route_without_action_candidate`
- `tests/test_self_cognition_tracking.py::test_past_due_contact_decision_writes_action_attempt_and_candidate_without_handoff`
- `tests/test_self_cognition_tracking.py::test_build_self_cognition_case_artifacts_does_not_write_files`
- `tests/test_self_cognition_tracking.py::test_runner_apply_consolidation_uses_empty_dialog_without_render`
- `tests/test_self_cognition_tracking.py::test_runner_consolidates_no_action_cognition_without_dialog`
- `tests/test_self_cognition_tracking.py::test_runner_does_not_call_dialog_for_intent_only_no_speak`
- `tests/test_self_cognition_tracking.py::test_runner_skips_dialog_for_private_only_actions_without_directives`
- `tests/test_self_cognition_tracking.py::test_runner_rejects_explicit_visible_route_without_speak`
- `tests/test_self_cognition_tracking.py::test_runner_executes_private_lifecycle_action_for_consolidation`
- `tests/test_self_cognition_tracking.py::test_runner_routes_lifecycle_intent_through_specialist_before_execution`
- `tests/test_self_cognition_tracking.py::test_runner_does_not_execute_private_actions_by_default`
- `tests/test_self_cognition_tracking.py::test_runner_reuses_dialog_render_for_action_and_consolidation`
- `tests/test_self_cognition_tracking.py::test_contact_decision_without_candidate_marker_uses_dialog_candidate`
- `tests/test_self_cognition_tracking.py::test_selected_speak_self_cognition_runs_l3_before_dialog`
- `tests/test_self_cognition_tracking.py::test_dialog_output_without_inline_tag_omits_group_action_delivery_mention`
- `tests/test_self_cognition_tracking.py::test_dialog_inline_tag_builds_group_action_delivery_mention`
- `tests/test_self_cognition_tracking.py::test_duplicate_contact_decision_suppresses_same_due_occurrence`
- `tests/test_self_cognition_tracking.py::test_duplicate_contact_decision_suppresses_active_prior_attempt_statuses`
- `tests/test_self_cognition_tracking.py::test_group_review_suppresses_prior_delivery_failed_attempt`
- `tests/test_self_cognition_tracking.py::test_group_noise_rejected_without_rag_or_action`
- `tests/test_self_cognition_tracking.py::test_group_chat_review_starts_without_preloaded_rag`
- `tests/test_self_cognition_tracking.py::test_topic_followup_contact_decision_writes_action_candidate`
- `tests/test_self_cognition_tracking.py::test_cognition_state_keeps_source_packet_inside_internal_percept`
- `tests/test_self_cognition_tracking.py::test_cognition_state_disables_visual_and_does_not_suppress_memory`

### tests/test_service_background_consolidation.py

- `tests/test_service_background_consolidation.py::test_chat_queues_background_consolidation_for_mapping_state`
- `tests/test_service_background_consolidation.py::test_chat_response_uses_true_reply_feature_from_graph`
- `tests/test_service_background_consolidation.py::test_chat_response_adds_inline_delivery_mentions_without_channel_gate`
- `tests/test_service_background_consolidation.py::test_chat_response_reply_feature_keeps_inline_delivery_mentions`
- `tests/test_service_background_consolidation.py::test_chat_response_adds_multiple_inline_delivery_mentions`
- `tests/test_service_background_consolidation.py::test_chat_response_keeps_inline_mention_when_scope_repeats_current_user`
- `tests/test_service_background_consolidation.py::test_chat_response_preserves_message_sequence_for_inline_mentions`
- `tests/test_service_background_consolidation.py::test_chat_response_tracks_deliverable_assistant_row`
- `tests/test_service_background_consolidation.py::test_chat_response_waits_for_assistant_persistence`
- `tests/test_service_background_consolidation.py::test_chat_response_omits_tracking_id_when_no_message`
- `tests/test_service_background_consolidation.py::test_chat_cognition_silence_skips_user_visible_work`
- `tests/test_service_background_consolidation.py::test_chat_consolidates_private_action_result_without_dialog`
- `tests/test_service_background_consolidation.py::test_post_turn_lifecycle_iterates_after_productive_passes`
- `tests/test_service_background_consolidation.py::test_post_turn_lifecycle_skips_structural_blockers`
- `tests/test_service_background_consolidation.py::test_chat_runs_post_turn_lifecycle_before_progress_and_consolidation`
- `tests/test_service_background_consolidation.py::test_next_chat_waits_until_background_consolidation_finishes`
- `tests/test_service_background_consolidation.py::test_no_remember_skips_consolidation_but_releases_after_other_writes`
- `tests/test_service_background_consolidation.py::test_graph_failure_does_not_stop_queue_worker`
- `tests/test_service_background_consolidation.py::test_background_consolidation_refreshes_cached_character_state`
- `tests/test_service_background_consolidation.py::test_build_graph_preserves_consolidation_state_from_supervisor`
- `tests/test_service_background_consolidation.py::test_build_graph_preserves_persona_no_response`
- `tests/test_service_background_consolidation.py::test_build_graph_skips_episode_state_loader_when_relevance_declines`
- `tests/test_service_background_consolidation.py::test_chat_listen_only_drops_before_graph`

### tests/test_service_event_logging.py

- `tests/test_service_event_logging.py::test_enqueue_suppresses_routine_accepted_queue_event`
- `tests/test_service_event_logging.py::test_process_queued_item_suppresses_routine_success_events`
- `tests/test_service_event_logging.py::test_graph_failure_records_runtime_error_and_failed_pipeline`
- `tests/test_service_event_logging.py::test_precommit_cognition_conflict_retries_graph_once`
- `tests/test_service_event_logging.py::test_resolver_state_contract_retries_once_then_settles_operational_failure`
- `tests/test_service_event_logging.py::test_user_persistence_failure_keeps_failure_telemetry`
- `tests/test_service_event_logging.py::test_lifespan_records_process_and_resource_events`

### tests/test_service_input_queue.py

- `tests/test_service_input_queue.py::test_first_settled_contract_failure_uses_bounded_wait`
- `tests/test_service_input_queue.py::test_final_settled_contract_failure_returns_operational_error`
- `tests/test_service_input_queue.py::test_native_reply_promotes_for_intervening_group_answer`
- `tests/test_service_input_queue.py::test_native_reply_true_survives_later_fragments`
- `tests/test_service_input_queue.py::test_native_reply_reaches_single_fragment_response`
- `tests/test_service_input_queue.py::test_native_reply_is_false_without_visible_dialog`
- `tests/test_service_input_queue.py::test_native_reply_promotes_after_delay_threshold`
- `tests/test_service_input_queue.py::test_native_reply_delay_promotion_requires_strictly_greater_than_threshold`
- `tests/test_service_input_queue.py::test_native_reply_promotion_skips_private_scope`
- `tests/test_service_input_queue.py::test_native_reply_promotion_requires_platform_message_id`
- `tests/test_service_input_queue.py::test_native_reply_stays_false_without_promotion_condition`
- `tests/test_service_input_queue.py::test_postcommit_degraded_dialog_uses_normal_delivery_path`
- `tests/test_service_input_queue.py::test_bot_continuity_follows_successful_visible_persistence`
- `tests/test_service_input_queue.py::test_failed_assistant_persistence_does_not_record_bot_continuity`
- `tests/test_service_input_queue.py::test_chat_enqueue_commits_receipt_before_queue_admission`
- `tests/test_service_input_queue.py::test_worker_consumes_precommitted_receipt_without_duplicate`
- `tests/test_service_input_queue.py::test_listen_only_worker_attaches_trace_to_precommitted_receipt`
- `tests/test_service_input_queue.py::test_listen_only_drop_keeps_precommitted_receipt_without_duplicate`
- `tests/test_service_input_queue.py::test_shutdown_drain_keeps_precommitted_receipt`
- `tests/test_service_input_queue.py::test_native_reply_fails_closed_without_durable_receipt_time`
- `tests/test_service_input_queue.py::test_enqueue_requests_same_scope_background_cancellation`
- `tests/test_service_input_queue.py::test_cancelled_enqueue_wait_keeps_foreground_handle`
- `tests/test_service_input_queue.py::test_worker_saves_dropped_messages_before_next_graph`
- `tests/test_service_input_queue.py::test_dropped_message_never_invokes_graph`
- `tests/test_service_input_queue.py::test_worker_suppresses_graph_when_surviving_user_save_fails`
- `tests/test_service_input_queue.py::test_worker_drops_listen_only_without_pruning_active_plain`
- `tests/test_service_input_queue.py::test_worker_saves_collapsed_messages_before_graph`
- `tests/test_service_input_queue.py::test_private_frontline_sees_complete_coalesced_logical_input`
- `tests/test_service_input_queue.py::test_worker_aborts_survivor_when_collapsed_save_fails`
- `tests/test_service_input_queue.py::test_worker_derives_graph_input_from_message_envelope`
- `tests/test_service_input_queue.py::test_enqueue_skips_empty_no_content_turn`
- `tests/test_service_input_queue.py::test_worker_keeps_image_only_turn_on_graph_path`
- `tests/test_service_input_queue.py::test_worker_keeps_collapsed_non_empty_content_on_graph_path`
- `tests/test_service_input_queue.py::test_worker_resolves_cross_user_envelope_targets`
- `tests/test_service_input_queue.py::test_worker_preserves_collapsed_image_input`

### tests/unit/brain_service/test_cognition_graph_projection.py

- `tests/unit/brain_service/test_cognition_graph_projection.py::test_service_publishes_canonical_observation_without_legacy_graph_helpers`
- `tests/unit/brain_service/test_cognition_graph_projection.py::test_live_terminal_status_mapping_and_cancellation_are_exact`
- `tests/unit/brain_service/test_cognition_graph_projection.py::test_failed_run_uses_current_attempt_prewarm_checkpoint`
- `tests/unit/brain_service/test_cognition_graph_projection.py::test_cognition_contract_exhaustion_preserves_typed_error_code`
- `tests/unit/brain_service/test_cognition_graph_projection.py::test_pre_commit_contract_exhaustion_triggers_one_checkpoint_replay`
- `tests/unit/brain_service/test_cognition_graph_projection.py::test_post_commit_degradations_do_not_trigger_replay`
- `tests/unit/brain_service/test_cognition_graph_projection.py::test_attempt_diagnostics_reducer_concatenates_within_bound`
- `tests/unit/brain_service/test_cognition_graph_projection.py::test_initial_process_state_seeds_relevance_diagnostics_before_persona`
- `tests/unit/brain_service/test_cognition_graph_projection.py::test_service_consumes_only_settlement_outcome_diagnostics`
- `tests/unit/brain_service/test_cognition_graph_projection.py::test_brain_response_contract_uses_canonical_cognition_observation`

### tests/unit/cognition_core_v3/test_handleless_contract.py

- `tests/unit/cognition_core_v3/test_handleless_contract.py::test_canonical_cognition_calls_a1_a2_g_p_once_with_subjective_outputs`
- `tests/unit/cognition_core_v3/test_handleless_contract.py::test_canonical_cognition_completes_without_input_evidence`

### tests/unit/cognition_core_v3/test_stage_recovery.py

- `tests/unit/cognition_core_v3/test_stage_recovery.py::test_stage_regenerates_with_exact_contract_error_and_rejected_candidate`
- `tests/unit/cognition_core_v3/test_stage_recovery.py::test_object_valued_response_goal_converges_after_one_feedback_attempt`
- `tests/unit/cognition_core_v3/test_stage_recovery.py::test_url_only_ordinary_response_goal_is_recovered_as_p_contract_error`
- `tests/unit/cognition_core_v3/test_stage_recovery.py::test_post_pending_hil_echo_regenerates_without_reopening_pending`
- `tests/unit/cognition_core_v3/test_stage_recovery.py::test_tool_result_delivery_carrier_echo_regenerates_to_result_plan`
- `tests/unit/cognition_core_v3/test_stage_recovery.py::test_url_only_self_cognition_response_goal_is_recovered_as_p_contract_error`
- `tests/unit/cognition_core_v3/test_stage_recovery.py::test_provider_failure_consumes_one_attempt_and_regenerates`
- `tests/unit/cognition_core_v3/test_stage_recovery.py::test_stage_exhaustion_raises_retryable_pre_commit_execution_error`
- `tests/unit/cognition_core_v3/test_stage_recovery.py::test_rejected_attempt_records_contract_fault_before_disposition`
- `tests/unit/cognition_core_v3/test_stage_recovery.py::test_regeneration_is_skipped_below_the_remaining_deadline_floor`
- `tests/unit/cognition_core_v3/test_stage_recovery.py::test_appraisal_family_key_order_is_normalized_without_regeneration`

### tests/unit/cognition_core_v3/test_state_transaction.py

- `tests/unit/cognition_core_v3/test_state_transaction.py::test_character_state_transaction_advances_timestamp_and_validates_final_affect`
- `tests/unit/cognition_core_v3/test_state_transaction.py::test_character_noop_transaction_advances_timestamp_strictly`
- `tests/unit/cognition_core_v3/test_state_transaction.py::test_cognition_turn_deadline_bounds_full_chain`

### tests/unit/cognition_observability/test_contracts.py

- `tests/unit/cognition_observability/test_contracts.py::test_observation_contract_rejects_unknown_fields_invalid_references_and_over_budget_payloads`
- `tests/unit/cognition_observability/test_contracts.py::test_observation_contract_enforces_truthful_record_counts_statuses_and_utc_serialization`

### tests/unit/cognition_observability/test_projection.py

- `tests/unit/cognition_observability/test_projection.py::test_live_projection_reports_all_shared_memory_prewarm_dispositions`
- `tests/unit/cognition_observability/test_projection.py::test_context_sources_share_one_detail_shape_and_budget`
- `tests/unit/cognition_observability/test_projection.py::test_public_group_scene_projects_discriminator_headers_and_status`
- `tests/unit/cognition_observability/test_projection.py::test_conversation_progress_invalid_headers_affect_section_status`
- `tests/unit/cognition_observability/test_projection.py::test_self_source_uses_stable_wire_field_keys_and_order`
- `tests/unit/cognition_observability/test_projection.py::test_self_action_results_fallback_requires_a_valid_empty_result_list`
- `tests/unit/cognition_observability/test_projection.py::test_self_visible_message_precedence_fails_closed_and_counts_source_rows`
- `tests/unit/cognition_observability/test_projection.py::test_live_and_self_projections_share_exact_section_catalog`
- `tests/unit/cognition_observability/test_projection.py::test_projection_uses_closed_source_field_mapping_and_invalid_row_counts`
- `tests/unit/cognition_observability/test_projection.py::test_projection_excludes_protected_and_operational_fields`
- `tests/unit/cognition_observability/test_projection.py::test_projection_emits_only_canonical_sequence_and_reference_edges`

### tests/unit/nodes/test_dialog_agent.py

- `tests/unit/nodes/test_dialog_agent.py::test_dialog_retry_carries_rejected_candidate_and_contract_error`
- `tests/unit/nodes/test_dialog_agent.py::test_inline_task_source_url_survives_dialog_normalization`
- `tests/unit/nodes/test_dialog_agent.py::test_missing_required_source_url_is_appended_without_regeneration`
- `tests/unit/nodes/test_dialog_agent.py::test_unexpected_source_url_is_removed_before_degradation`
- `tests/unit/nodes/test_dialog_agent.py::test_dialog_delivers_newest_retained_candidate_after_structural_exhaustion`
- `tests/unit/nodes/test_dialog_agent.py::test_dialog_projects_content_plan_when_no_candidate_survives`
- `tests/unit/nodes/test_dialog_agent.py::test_dialog_never_raises_on_provider_exhaustion`
- `tests/unit/nodes/test_dialog_agent.py::test_oversized_visible_percepts_bound_url_scan_without_failing_dialog`

### tests/unit/nodes/test_persona_supervisor2_cognition_commit.py

- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_cognition_state_preserves_shared_memory_prewarm_outcome_after_merge`
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_cognition_records_noneligible_prewarm_without_starting_worker`
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_cognition_cancellation_publishes_no_prewarm_outcome_or_observation`
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_resolver_recurrence_commits_against_original_user_base`
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_cognition_state_version_conflict_is_retryable_before_surface`
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_persona_character_commit_reads_canonical_state_projection`

### tests/unit/nodes/test_persona_supervisor2_l3_surface.py

- `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_terminal_dialog_preserves_degraded_tool_result_semantic_result`
- `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_tool_result_reaches_the_dialog_payload`
- `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_text_surface_uses_one_content_call_and_deterministic_preference_projection`
- `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_successful_content_plan_is_not_discarded_by_deterministic_projection`
- `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_handler_binds_state_trace_for_text_and_visual_calls`
- `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_unexpected_visual_failure_is_omitted_and_text_surface_is_returned`
- `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_visual_cancellation_still_propagates`

### tests/unit/nodes/test_persona_supervisor2_schema.py

- `tests/unit/nodes/test_persona_supervisor2_schema.py::test_persona_supervisor_returns_attempt_diagnostics`
- `tests/unit/nodes/test_persona_supervisor2_schema.py::test_persona_supervisor_returns_only_new_diagnostic_delta`

