# Cognition Core V2 Stage 3 Change Radius

## Summary

- Record role: mandatory exact Create/Modify/Delete/Keep inventory for
  [cognition_core_v2_stage_3_system_adoption_plan.md](cognition_core_v2_stage_3_system_adoption_plan.md).
- Execution companion:
  [cognition_core_v2_stage_3_execution_manifest.md](cognition_core_v2_stage_3_execution_manifest.md).
- Plan class: high_risk_migration.
- Status: in_progress.
- Authority: the parent plan governs; user-authorized implementation is active;
  this file grants no production-data authority.
- Change-control rule: amend this file and obtain plan approval before
  changing an unlisted production or test path.

## User Quality Sign-off Record — 2026-07-22

The user approved the Phase 3 artifact set, including the retained real-LLM
emotion, abuse-boundary, role, mechanical-path, bounded-error, and consolidated
Chinese dialog/monologue evidence. This inventory remains `in_progress` with
the parent plan until the separately tracked external Browser acceptance and
remaining lifecycle completion checks are closed.

## Exact Change Surface

### Create

```text
src/kazusa_ai_chatbot/character_profile.py
src/kazusa_ai_chatbot/db/internal_action_latches.py
src/kazusa_ai_chatbot/db/post_turn_lifecycle.py
tests/stage3_fresh_database.py
tests/test_character_profile_seed.py
tests/test_internal_action_latches.py
tests/test_post_turn_lifecycle_record.py
tests/test_stage3_trigger_source_cutover.py
tests/test_stage3_fresh_database_bootstrap.py
tests/test_stage3_documentation_links.py
tests/test_stage3_fresh_database_e2e_live_llm.py
tests/test_stage3_background_reasoning_live_db.py
tests/test_stage3_auxiliary_v2_contract.py
tests/control_console_e2e/test_stage3_fresh_database_e2e.py
tests/fixtures/stage3_fresh_database_cases.json
tests/fixtures/cognition_core_v2_abuse_to_sadness_e2e_cases.json
tests/fixtures/cognition_core_v2_crying_sadness_e2e_cases.json
tests/fixtures/cognition_core_v2_high_attachment_abuse_e2e_cases.json
tests/fixtures/cognition_core_v2_secondary_crying_e2e_cases.json
tests/fixtures/cognition_core_v2_verbal_abuse_boundary_e2e_cases.json
tests/test_cognition_core_v2_abuse_to_sadness_dialog_e2e_live_llm.py
tests/test_cognition_core_v2_abuse_to_sadness_e2e_live_llm.py
tests/test_cognition_core_v2_abuse_to_sadness_mechanical.py
tests/test_cognition_core_v2_crying_sadness_e2e_live_llm.py
tests/test_cognition_core_v2_high_attachment_abuse_e2e_live_llm.py
tests/test_cognition_core_v2_secondary_crying_e2e_live_llm.py
tests/test_cognition_core_v2_verbal_abuse_boundary_e2e_live_llm.py
test_artifacts/cognition_core_v2/stage_3/
```

`tests/fixtures/stage3_fresh_database_cases.json` is tracked test input. It has
schema version `stage3_fresh_database_cases.v1`, source artifact paths, and 40
ordered cases with `case_id`, `sequence`, `turn_index`, sanitized `input_text`,
`target_scope_fixture`, and typed technical expectations. Generated per-case
traces/reports under `test_artifacts/cognition_core_v2/stage_3/` are ignored
unless the repository's existing artifact policy explicitly tracks the named
final review document; protected traces always remain in their protected
owner.

### Delete

```text
src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py
src/kazusa_ai_chatbot/internal_thought_cognition.py
src/kazusa_ai_chatbot/proactive_output/__init__.py
src/kazusa_ai_chatbot/proactive_output/contracts.py
src/kazusa_ai_chatbot/proactive_output/policy.py
src/kazusa_ai_chatbot/proactive_output/outbox.py
src/kazusa_ai_chatbot/proactive_output/README.md
tests/test_multi_source_cognition_stage_07_reflection_dry_run.py
tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py
tests/test_multi_source_cognition_stage_10_proactive_policy.py
tests/test_multi_source_cognition_stage_10_proactive_outbox.py
```

Delete only after all production/test imports and equivalent runtime ownership
tests have moved.

### Modify — Bootstrap And Deployment

```text
Dockerfile — set deployed absolute profile path
src/kazusa_ai_chatbot/config.py — required profile path and startup validation
src/kazusa_ai_chatbot/service.py — startup order, runtime owners, one settlement caller
src/kazusa_ai_chatbot/db/__init__.py — public bootstrap/profile exports
src/kazusa_ai_chatbot/db/bootstrap.py — native indexes and no legacy fresh collections
src/kazusa_ai_chatbot/db/character.py — insert/verify static seed boundary
src/kazusa_ai_chatbot/db/schemas.py — seed/native singleton typing
src/scripts/load_character_profile.py — shared validator; explicit maintenance only
```

### Modify — Episode, Cognition, Action, Surface, Trace

```text
src/kazusa_ai_chatbot/cognition_episode.py
src/kazusa_ai_chatbot/cognition_core_v2/contracts.py
src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py
src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py
src/kazusa_ai_chatbot/cognition_core_v2/facade.py
src/kazusa_ai_chatbot/cognition_core_v2/diagnostics.py
src/kazusa_ai_chatbot/cognition_core_v2/state_models.py
src/kazusa_ai_chatbot/cognition_core_v2/validation_cli.py
src/kazusa_ai_chatbot/cognition_core_v2/surface.py
src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py
src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py
src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py
src/kazusa_ai_chatbot/cognition_core_v2/transition_guards.py
src/kazusa_ai_chatbot/cognition_resolver/contracts.py
src/kazusa_ai_chatbot/cognition_resolver/loop.py
src/kazusa_ai_chatbot/cognition_resolver/capabilities.py
src/kazusa_ai_chatbot/cognition_resolver/pending.py
src/kazusa_ai_chatbot/cognition_resolver/state.py
src/kazusa_ai_chatbot/llm_interface/contracts.py
src/kazusa_ai_chatbot/llm_interface/route_report.py
src/kazusa_ai_chatbot/action_spec/__init__.py
src/kazusa_ai_chatbot/action_spec/models.py
src/kazusa_ai_chatbot/action_spec/registry.py
src/kazusa_ai_chatbot/action_spec/evaluator.py
src/kazusa_ai_chatbot/action_spec/execution.py
src/kazusa_ai_chatbot/action_spec/results.py
src/kazusa_ai_chatbot/nodes/persona_supervisor2.py
src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py
src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py
src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py
src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py
src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py
src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py
src/kazusa_ai_chatbot/nodes/dialog_agent.py
src/kazusa_ai_chatbot/brain_service/post_turn.py
src/kazusa_ai_chatbot/state.py
```

Only contract projection/ownership changes are allowed in the V2 and dialog
files; Stage 2 emotion/personality tuning is preserved.

Review reconciliation: the V2 surface/transition contract modules and the
accepted-task prompt contract test are direct Stage 3 consumers identified by
the independent implementation review; they are included here without adding
new runtime capability or product scope.

### Modify — Evidence, Persistence, Progress

```text
src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py
src/kazusa_ai_chatbot/rag/prompt_projection.py
src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py
src/kazusa_ai_chatbot/consolidation/core.py
src/kazusa_ai_chatbot/consolidation/schema.py
src/kazusa_ai_chatbot/consolidation/origin.py
src/kazusa_ai_chatbot/consolidation/origin_policy.py
src/kazusa_ai_chatbot/consolidation/source_policy.py
src/kazusa_ai_chatbot/consolidation/target.py
src/kazusa_ai_chatbot/consolidation/images.py
src/kazusa_ai_chatbot/consolidation/memory_units.py
src/kazusa_ai_chatbot/conversation_progress/models.py
src/kazusa_ai_chatbot/conversation_progress/recorder.py
src/kazusa_ai_chatbot/internal_monologue_residue/recorder.py
src/kazusa_ai_chatbot/event_logging/recording.py
```

### Modify — Grounded Background Producers

```text
src/kazusa_ai_chatbot/self_cognition/runner.py
src/kazusa_ai_chatbot/self_cognition/__init__.py
src/kazusa_ai_chatbot/self_cognition/models.py
src/kazusa_ai_chatbot/self_cognition/projection.py
src/kazusa_ai_chatbot/self_cognition/delivery.py
src/kazusa_ai_chatbot/self_cognition/sources.py
src/kazusa_ai_chatbot/self_cognition/worker.py
src/kazusa_ai_chatbot/self_cognition/tracking.py
src/kazusa_ai_chatbot/background_work/__init__.py
src/kazusa_ai_chatbot/background_work/jobs.py
src/kazusa_ai_chatbot/background_work/result_source.py
src/kazusa_ai_chatbot/background_work/delivery.py
src/kazusa_ai_chatbot/background_work/worker.py
src/kazusa_ai_chatbot/background_work/subagent/__init__.py
src/kazusa_ai_chatbot/background_work/subagent/future_speak.py
src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py
src/kazusa_ai_chatbot/accepted_task/__init__.py
src/kazusa_ai_chatbot/accepted_task/models.py
src/kazusa_ai_chatbot/accepted_task/lifecycle.py
src/kazusa_ai_chatbot/db/accepted_tasks.py
src/kazusa_ai_chatbot/db/self_cognition.py
src/kazusa_ai_chatbot/calendar_scheduler/handlers.py
src/kazusa_ai_chatbot/calendar_scheduler/worker.py
src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py
src/kazusa_ai_chatbot/action_spec/handlers/accepted_task.py
src/kazusa_ai_chatbot/action_spec/handlers/background_work.py
src/kazusa_ai_chatbot/reflection_cycle/affect_settling.py
src/kazusa_ai_chatbot/reflection_cycle/worker.py
```

Reflection workers/repositories otherwise keep their current offline ownership.

### Modify — Operations, Console, Scripts

```text
src/kazusa_ai_chatbot/brain_service/contracts.py
src/kazusa_ai_chatbot/brain_service/health.py
src/kazusa_ai_chatbot/event_logging/snapshots.py
src/kazusa_ai_chatbot/event_logging/status.py
src/kazusa_ai_chatbot/llm_tracing/__init__.py
src/control_console/contracts.py
src/control_console/kazusa_client.py
src/control_console/repository.py
src/control_console/app.py
src/control_console/redaction.py
src/control_console/stream.py
src/control_console/static/index.html
src/control_console/static/console.js
src/control_console/static/console.css
src/scripts/export_user_profile.py
src/scripts/export_character_state.py
src/scripts/export_user_image.py
src/scripts/identify_user_image.py
src/scripts/user_state_snapshot.py
src/scripts/character_state_snapshot.py
src/scripts/audit_user_profiles_lane.py
src/scripts/audit_character_state_lane.py
src/scripts/fetch_ops_status.py
src/scripts/count_project_artifacts.py
src/scripts/export_llm_trace.py
src/scripts/export_event_log.py
src/scripts/export_dialog_trace_review_input.py
```

Trace/console changes expose bounded source, capability, terminal status,
latency, and retry fields under existing redaction/access rules.

### Modify — Primary Tests

```text
tests/conftest.py
tests/cognition_core_v2_test_helpers.py
tests/live_llm_mongo.py
tests/test_live_llm_mongo_isolation.py
tests/test_cognitive_episode_contract.py
tests/test_cognition_chain_connector_mapping.py
tests/test_cognition_core_v2_alignment_gates.py
tests/test_cognition_core_v2_action_planning_live_llm.py
tests/test_cognition_core_v2_benchmark.py
tests/test_cognition_core_v2_dependencies.py
tests/test_cognition_core_v2_projection.py
tests/test_cognition_core_v2_state.py
tests/test_cognition_core_v2_integration.py
tests/test_cognition_core_v2_live_llm.py
tests/test_cognition_resolver_loop.py
tests/test_action_selection_prompt_contract.py
tests/test_action_spec_evaluator.py
tests/test_action_spec_future_cognition.py
tests/test_action_spec_models.py
tests/test_action_spec_results.py
tests/test_accepted_task_lifecycle.py
tests/test_accepted_task_prompt_contract.py
tests/test_background_work_future_speak.py
tests/test_background_work_delivery.py
tests/test_coding_agent_background_run_contracts.py
tests/test_coding_agent_phase3_handoff_e2e.py
tests/test_coding_agent_phase3_live_e2e.py
tests/test_self_cognition_integration.py
tests/test_self_cognition_tracking.py
tests/test_self_cognition_event_logging.py
tests/test_self_cognition_architecture_docs.py
tests/test_rag_cognitive_episode_adapter.py
tests/test_consolidation_origin_metadata.py
tests/test_consolidation_origin_policy.py
tests/test_consolidation_source_policy.py
tests/test_consolidation_lane_router_contract.py
tests/test_consolidation_target_routing.py
tests/test_consolidation_memory_write_use_cases_live_llm.py
tests/test_consolidator_origin_policy_db_writer.py
tests/test_consolidator_origin_selection.py
tests/test_consolidator_source_aware_payloads.py
tests/test_internal_monologue_residue_recorder.py
tests/test_msg_decontexualizer.py
tests/test_conversation_progress_history_policy.py
tests/test_past_dialog_cognition_prompt_boundaries.py
tests/test_service_background_consolidation.py
tests/test_service_health.py
tests/test_service_ops_status.py
tests/test_service_event_logging.py
tests/test_db.py
tests/test_config.py
tests/test_llm_interface_contracts.py
tests/test_llm_interface_route_report.py
tests/test_script_db_boundary.py
tests/test_user_state_snapshot.py
tests/test_character_state_snapshot.py
tests/test_control_console_contracts.py
tests/test_control_console_repository.py
tests/test_control_console_review_edges.py
tests/test_control_console_cognition_debug_visibility.py
tests/test_control_console_cognition_graph.py
tests/test_control_console_kazusa_client.py
tests/test_control_console_redaction.py
tests/test_control_console_web_surface.py
tests/control_console_e2e/fake_brain.py
tests/control_console_e2e/fake_services.py
tests/control_console_e2e/test_cognition_graph_e2e.py
tests/control_console_e2e/test_debug_chat_e2e.py
tests/control_console_e2e/test_live_database_owner_pages_e2e.py
tests/control_console_e2e/test_page_navigation_e2e.py
```

This primary list includes every current direct test occurrence of a forbidden
trigger or retired scaffold. Checkpoint A must reproduce the list before edit;
an unlisted current match is a plan-review blocker and requires an inventory
amendment before that file changes. Mechanical vocabulary changes do not
authorize unrelated expectation rewrites.

The `CognitiveEpisodeV1` big-bang contract also reaches every current direct
episode consumer below. This is a coverage overlay; paths already listed above
or in the real-LLM group remain one planned edit, not duplicate scope.

```text
tests/cognition_core_v2_test_helpers.py
tests/test_action_spec_evaluator.py
tests/test_action_spec_future_cognition.py
tests/test_action_spec_memory_lifecycle.py
tests/test_action_spec_results.py
tests/test_background_work_delivery.py
tests/test_background_work_future_speak_live_llm.py
tests/test_background_work_jobs.py
tests/test_calendar_scheduler_models.py
tests/test_coding_agent_background_run_contracts.py
tests/test_coding_agent_full_workflow_integration_live_llm.py
tests/test_coding_agent_phase3_handoff_e2e.py
tests/test_coding_agent_phase3_live_e2e.py
tests/test_cognition_chain_connector_mapping.py
tests/test_cognition_core_v2_contracts.py
tests/test_cognition_core_v2_action_planning_bugfix.py
tests/test_cognition_core_v2_frozen_replay_drift.py
tests/test_cognition_current_event_grounding.py
tests/test_cognition_interaction_style_context.py
tests/test_cognition_live_llm_prompt_contracts.py
tests/test_cognition_resolver_contracts.py
tests/test_cognition_resolver_l2d_contract.py
tests/test_cognition_resolver_loop.py
tests/test_cognitive_episode_contract.py
tests/test_consolidation_lane_bigbang_integration.py
tests/test_consolidation_lane_router_contract.py
tests/test_consolidation_origin_metadata.py
tests/test_consolidation_target_routing.py
tests/test_consolidator_efficiency.py
tests/test_consolidator_origin_selection.py
tests/test_conversation_progress_cognition.py
tests/test_conversation_progress_flow_live_llm.py
tests/test_conversation_progression_live_llm.py
tests/test_db_writer_cache2_invalidation.py
tests/test_decontexualizer_live_llm.py
tests/test_dialog_agent.py
tests/test_dialog_generator_live_llm_contract.py
tests/test_dialog_mention_target_user.py
tests/test_dialog_visible_speech_and_semantic_fidelity_live_llm.py
tests/test_dialog_visible_speech_and_semantic_fidelity.py
tests/test_e2e_live_llm.py
tests/test_image_cognition_options_live_llm.py
tests/test_internal_monologue_residue_recorder.py
tests/test_kazusa_victory_anchor_live_llm.py
tests/test_l2d_action_selection_cases.py
tests/test_l2d_action_selection_live_llm.py
tests/test_l2d_l3_surface_handoff.py
tests/test_l2d_quiet_monologue_live_llm.py
tests/test_l2d_unknown_context_resolver_live_llm.py
tests/test_l3_dialog_content_plan_contract.py
tests/test_l3_dialog_content_plan_live_llm.py
tests/test_local_context_resolver_integration.py
tests/test_local_context_resolver_live_llm.py
tests/test_memory_lifecycle_specialist_live_llm.py
tests/test_memory_lifecycle_specialist.py
tests/test_msg_decontexualizer.py
tests/test_multi_source_cognition_image_input.py
tests/test_multi_source_cognition_stage_02_chat_episode_migration.py
tests/test_multi_source_cognition_stage_07_reflection_dry_run.py
tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py
tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py
tests/test_persona_supervisor2_rag_skip_shape.py
tests/test_persona_supervisor2_rag2_integration.py
tests/test_persona_supervisor2_schema.py
tests/test_persona_supervisor2.py
tests/test_persona_supervisor2_action_selection.py
tests/test_rag_cognitive_episode_adapter.py
tests/test_rag_dialog_event_logging.py
tests/test_self_cognition_duplicate_response_live_llm.py
tests/test_self_cognition_group_review_source.py
tests/test_self_cognition_integration.py
tests/test_self_cognition_response_sensitivity_live_llm.py
tests/test_self_cognition_tracking.py
tests/test_service_background_consolidation.py
tests/test_shared_memory_prewarm.py
tests/test_state.py
```

### Modify — Quality/Real-LLM Regression Tests

```text
tests/test_background_work_future_speak_live_llm.py
tests/test_boundary_core_sensitivity_live_llm.py
tests/test_coding_agent_full_workflow_integration_live_llm.py
tests/test_cognition_core_v2_frozen_replay_drift.py
tests/test_cognition_core_v2_live_character_judgment.py
tests/test_cognition_live_llm_prompt_contracts.py
tests/test_cognition_prompt_contract_text.py
tests/test_consolidation_evidence_hardening_live_llm.py
tests/test_conversation_progress_flow_live_llm.py
tests/test_conversation_progression_live_llm.py
tests/test_dialog_agent_direct_live_llm.py
tests/test_dialog_anchor_boundary_live_llm.py
tests/test_dialog_generator_live_llm_contract.py
tests/test_dialog_inline_mentions_live_llm.py
tests/test_dialog_l3_surface_contract_live_llm.py
tests/test_dialog_message_sequence_live_llm.py
tests/test_image_cognition_options_live_llm.py
tests/test_l2d_action_selection_live_llm.py
tests/test_l2d_quiet_monologue_live_llm.py
tests/test_l2d_unknown_context_resolver_live_llm.py
tests/test_l3_dialog_content_plan_live_llm.py
tests/test_local_context_resolver_full_matrix_live_llm.py
tests/test_memory_writer_perspective_live_llm.py
tests/test_multi_source_cognition_image_input.py
tests/test_relevance_cross_channel_failure_live_llm.py
tests/test_relevance_reply_to_bot_live_llm.py
tests/test_relevance_sensitivity_live_llm.py
tests/test_user_memory_units_live_llm.py
```

### Modify — Documentation And Lifecycle

```text
README.md
README_CN.md
docs/HOWTO.md
src/control_console/README.md
src/scripts/README.md
src/kazusa_ai_chatbot/accepted_task/README.md
src/kazusa_ai_chatbot/action_spec/README.md
src/kazusa_ai_chatbot/background_work/README.md
src/kazusa_ai_chatbot/brain_service/README.md
src/kazusa_ai_chatbot/calendar_scheduler/README.md
src/kazusa_ai_chatbot/cognition_core_v2/README.md
src/kazusa_ai_chatbot/cognition_resolver/README.md
src/kazusa_ai_chatbot/consolidation/README.md
src/kazusa_ai_chatbot/conversation_progress/README.md
src/kazusa_ai_chatbot/db/README.md
src/kazusa_ai_chatbot/event_logging/README.md
src/kazusa_ai_chatbot/internal_monologue_residue/README.md
src/kazusa_ai_chatbot/llm_interface/README.md
src/kazusa_ai_chatbot/llm_tracing/README.md
src/kazusa_ai_chatbot/nodes/README.md
src/kazusa_ai_chatbot/rag/README.md
src/kazusa_ai_chatbot/reflection_cycle/README.md
src/kazusa_ai_chatbot/relevance/README.md
src/kazusa_ai_chatbot/self_cognition/README.md
development_plans/README.md
development_plans/reference/designs/cognition_contracts_design.md
development_plans/reference/designs/action_spec_effector_expansion_architecture.md
development_plans/reference/designs/cognition_core_evolution_progression.md
development_plans/reference/designs/coding_agent_architecture.md
development_plans/reference/documentation_harmonization_audit_report.md
development_plans/active/short_term/cognition_core_v2_stage_3_system_adoption_plan.md
development_plans/active/short_term/cognition_core_v2_stage_3_execution_manifest.md
development_plans/active/short_term/cognition_core_v2_stage_3_change_radius.md
development_plans/active/short_term/cognition_core_v2_stage_4_production_database_migration_plan.md
```

The authoritative reference changes to align the five-source roster,
EpisodeTraceV2 settlement, nine-kind action roster, and their ownership with
the approved Stage 3 contract.

### Keep / Verify Unchanged

```text
src/adapters/
personalities/asuna.json
personalities/example.json
personalities/kazusa.json
personalities/qingche.json
src/kazusa_ai_chatbot/db/_client.py
src/kazusa_ai_chatbot/db/users.py
src/kazusa_ai_chatbot/dispatcher/
src/kazusa_ai_chatbot/llm_interface/providers/
src/kazusa_ai_chatbot/rag/conversation_evidence/
src/kazusa_ai_chatbot/rag/live_context/
src/kazusa_ai_chatbot/rag/memory_evidence/
src/kazusa_ai_chatbot/rag/person_context/
src/kazusa_ai_chatbot/rag/recall/
src/kazusa_ai_chatbot/rag/web_agent3/
src/kazusa_ai_chatbot/memory_evolution/
src/kazusa_ai_chatbot/global_character_growth/
src/kazusa_ai_chatbot/coding_agent/
```

Adapter assertions may be updated only in tests; adapter production code keeps
its existing typed transport contract. Character JSON content is unchanged.

### Stage-4-only Legacy Allowlist

```text
src/kazusa_ai_chatbot/db/script_operations.py
src/scripts/_lane_cleanup.py
src/scripts/migrate_scheduled_events_to_calendar_scheduler.py
```

These remain unchanged and unreachable from Stage 3 normal runtime.
