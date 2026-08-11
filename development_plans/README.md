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
| [coding_agent_assessment_gap_phase_d_plan.md](active/short_term/coding_agent_assessment_gap_phase_d_plan.md) | High-risk coding-agent migration plan for a generic JSON action loop, persistent repository index, exploration-cap removal, and delete/rename | in progress |

## Completed Short-Term Plans

| Document | Type | Status |
|---|---|---|
| [v1_release_readiness_plan.md](archive/completed/short_term/v1_release_readiness_plan.md) | v1.0.0 release identity, deterministic contract cleanup, packaging, documentation, and local tag cutover | completed |
| [development_plan_test_impact_traceability_and_cognition_unit_structure_bigbang_plan.md](archive/completed/short_term/development_plan_test_impact_traceability_and_cognition_unit_structure_bigbang_plan.md) | Big-bang development-plan test-impact contract, cognition source ownership manifest, and mirrored deterministic unit-test structure | completed |
| [character_owned_content_judgment_cutover_plan.md](archive/completed/short_term/character_owned_content_judgment_cutover_plan.md) | Cutover removing application-owned semantic safety/refusal policy and relationship gating while preserving character-owned judgment, expression continuity, and operational guards | completed |
| [cognition_parallel_neutral_weak_tilt_plan.md](archive/completed/short_term/cognition_parallel_neutral_weak_tilt_plan.md) | Branch-owned semantic intent guidance for generic Cognition V2 bid generation; real-model quality explicitly deferred | completed |
| [august_change_alignment_audit_and_remediation_plan.md](archive/completed/short_term/august_change_alignment_audit_and_remediation_plan.md) | Repository-wide Aug 1+ change alignment audit, bounded remediation, and baseline feature regression map | completed |
| [legacy_llm_configuration_cleanup_plan.md](archive/completed/short_term/legacy_llm_configuration_cleanup_plan.md) | Removal of ten legacy local environment keys and three stale verifier Compose bindings | completed |
| [task_resolution_character_background_handoff_plan.md](archive/completed/short_term/task_resolution_character_background_handoff_plan.md) | Character-selected task-resolution background handoff and partial-result continuation | completed |
| [cognition_graph_multi_source_latest_run_plan.md](archive/completed/short_term/cognition_graph_multi_source_latest_run_plan.md) | Medium source-neutral latest cognition publication and transport contract change | completed |

## Supporting Experiment Records

| Document | Type | Status | Supports |
|---|---|---|---|
| [rag2_recall_quality_experiment_plan.md](reference/designs/rag2_recall_quality_experiment_plan.md) | Experiment decision and supporting evidence | reference evidence | [rag2_cognition_ready_evidence_plan.md](archive/completed/short_term/rag2_cognition_ready_evidence_plan.md) |

## Active Bugfix Plans

| Document | Type | Status |
|---|---|---|
| [cognition_core_v2_selected_response_operation_role_contract_bugfix_plan_20260810.md](archive/completed/bugfix/cognition_core_v2_selected_response_operation_role_contract_bugfix_plan_20260810.md) | Post-selection response-operation role contract repair for required-selection dialog exhaustion | completed |
| [cognition_surface_score_ranking_followup_plan_20260810.md](active/bugfix/cognition_surface_score_ranking_followup_plan_20260810.md) | Evidence-gated surface quality ranking follow-up for bounded retry exhaustion | draft |
| [self_cognition_trigger_state_contract_recovery_bugfix_plan.md](archive/completed/bugfix/self_cognition_trigger_state_contract_recovery_bugfix_plan.md) | Group-chat and commitment self-cognition V2 state-contract recovery | completed |
| [reflection_recursive_root_timestamp_canonicalization_bugfix_plan.md](archive/completed/bugfix/reflection_recursive_root_timestamp_canonicalization_bugfix_plan.md) | Recursive reflection root timestamp canonicalization and guarded group-ledger recovery | completed |
| [cognition_core_v2_goal_bid_live_llm_regression_remediation_20260810.md](archive/completed/bugfix/cognition_core_v2_goal_bid_live_llm_regression_remediation_20260810.md) | Goal-bid live-LLM failure classification, frozen-fixture recovery, prompt remediation, and 95% regression gate | completed |
| [cognition_core_v2_quoted_message_evidence_and_recurrence_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_quoted_message_evidence_and_recurrence_bugfix_plan.md) | Quoted-message resolver evidence continuity, answerability gating, and ordinary-branch recurrence stability | completed |
| [llm_trace_console_full_web_correlation_surface_bugfix_plan.md](archive/completed/bugfix/llm_trace_console_full_web_correlation_surface_bugfix_plan.md) | Full web-console correlation ID visibility across the existing mapped views | completed |
| [background_work_jobs_console_job_id_visibility_bugfix_plan.md](archive/completed/bugfix/background_work_jobs_console_job_id_visibility_bugfix_plan.md) | Existing Background Work Jobs card job-ID visibility in the web console | completed |
| [llm_trace_console_correlation_gap_bugfix_plan.md](archive/completed/bugfix/llm_trace_console_correlation_gap_bugfix_plan.md) | Protected trace correlation retrieval, web availability mapping, and forward action/background/future-cognition id propagation | completed |
| [cognition_core_v2_goal_schema_and_semantic_repair_compaction_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_goal_schema_and_semantic_repair_compaction_bugfix_plan.md) | Non-ordinary goal-schema isolation and semantic repair-envelope compaction | completed |
| [dialog_third_party_target_binding_and_addressee_fidelity_bugfix_plan.md](archive/completed/bugfix/dialog_third_party_target_binding_and_addressee_fidelity_bugfix_plan.md) | Ephemeral third-party target binding and final-dialog addressee fidelity | completed |
| [cognition_core_v2_action_planning_request_fidelity_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_action_planning_request_fidelity_bugfix_plan.md) | Action-planning request fidelity and retrieval-goal preservation | completed |
| [background_task_result_blocker_detail_delivery_bugfix_plan.md](archive/completed/bugfix/background_task_result_blocker_detail_delivery_bugfix_plan.md) | Explicit task-result blocker detail in background delivery | completed |
| [cognition_size_limit_truncation_and_fallback_scan_plan.md](archive/completed/bugfix/cognition_size_limit_truncation_and_fallback_scan_plan.md) | Project-wide context-size scan and deterministic truncation fallback | completed |
| [durable_ingress_native_reply_intervening_message_bugfix_plan.md](archive/completed/bugfix/durable_ingress_native_reply_intervening_message_bugfix_plan.md) | Durable inbound ordering for /chat and original-message background replies | completed |
| [background_tool_result_delivery_current_episode_evidence_bugfix_plan.md](archive/completed/bugfix/background_tool_result_delivery_current_episode_evidence_bugfix_plan.md) | Background tool-result evidence authority and visible result-delivery regression | completed |
| [background_coding_event_loop_starvation_bugfix_plan.md](archive/completed/bugfix/background_coding_event_loop_starvation_bugfix_plan.md) | Background coding async-boundary responsiveness and deterministic regression coverage | completed |
| [relevance_native_reply_review_remediation_plan.md](archive/completed/bugfix/relevance_native_reply_review_remediation_plan.md) | Post-review deterministic native-reply coverage, ID validation, and ICD wording remediation | completed |
| [relevance_native_reply_monotonic_delivery_plan.md](archive/completed/bugfix/relevance_native_reply_monotonic_delivery_plan.md) | Deterministic `/chat` native-reply monotonic delivery and delay promotion | completed |
| [cognition_core_v2_generation_contract_prompt_projection_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_generation_contract_prompt_projection_bugfix_plan.md) | High-risk Core V2 prompt, model-facing contract, and diagnostic replay bugfix | completed |
| [control_console_web_availability_followup_plan.md](archive/completed/bugfix/control_console_web_availability_followup_plan.md) | Follow-up bugfix for Control Console operational projections and static asset availability | completed |
| [cognition_core_v2_relational_willingness_gradient_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_relational_willingness_gradient_bugfix_plan.md) | Core V2 post-prewarm relational-willingness contract for stranger rejection, lover acceptance, scoped-memory authority, and an explicit intermediate gradient | completed |
| [cognition_core_v2_short_horizon_global_state_composition_bigbang_plan.md](archive/completed/bugfix/cognition_core_v2_short_horizon_global_state_composition_bigbang_plan.md) | Post-identity V2 plan for transient global posture, causal relationship context, scoped style composition, and runtime/UI proof | completed |
| [character_identity_growth_contract_recovery_bugfix_plan.md](archive/completed/bugfix/character_identity_growth_contract_recovery_bugfix_plan.md) | Follow-up bugfix for identity-growth model contracts, provenance indices, review copying, and bounded retry recovery | completed |
| [qq_group_public_scene_response_ordering_bugfix_plan.md](archive/completed/bugfix/qq_group_public_scene_response_ordering_bugfix_plan.md) | QQ multi-user public-scene composition and same-group response-ordering bugfix | completed |
| [cognition_core_v2_first_pass_robustness_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_first_pass_robustness_bugfix_plan.md) | Core V2 first-pass generation grounding, evaluator feedback, per-stage token/character budget rebalance, and deterministic failure-path recovery bugfix plan | completed |
| [cognition_core_v2_context_fade_and_sleep_phase_plan.md](archive/completed/bugfix/cognition_core_v2_context_fade_and_sleep_phase_plan.md) | Deterministic age-based discard of group-scene turns and conversation-progress topics before projection, plus cognition V2 ownership of the character sleep phase and morning refresh | completed |

## Reference Documents

| Document | Type |
|---|---|
| [action_spec_effector_expansion_architecture.md](reference/designs/action_spec_effector_expansion_architecture.md) | Architecture reference |
| [CODING_AGENT_CAPABILITY_ASSESSMENT.md](reference/designs/CODING_AGENT_CAPABILITY_ASSESSMENT.md) | Coding-agent capability assessment reference |
| [coding_agent_assessment_gap_gate_06_10_failure_modes.md](reference/designs/coding_agent_assessment_gap_gate_06_10_failure_modes.md) | Supporting real LLM failure-mode evidence |
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

### 2026-07-09 coding-agent Phase 9 executable plan approval

The directional coding-agent run supervisor architecture was promoted into an
approved active short-term execution plan. Five real LLM run-supervisor gates
were prepared ahead of implementation, and plan review tightened the contract
with closed run objectives, closed continuation actions, stable ledger
schemas, and explicit evidence requirements.

| Outcome | Plans |
|---|---|
| Added to active short-term | `coding_agent_phase9_run_supervisor_plan.md` |

### 2026-07-09 coding-agent Phase 9 completion

The coding-agent Phase 9 run supervisor plan completed after user-approved
execution without subagents. The implementation added workspace-local durable
run ledgers, public `start_coding_run(...)`, `continue_coding_run(...)`, and
`get_coding_run(...)` APIs, deterministic workflow tests, five one-at-a-time
real LLM gates with raw/review evidence, repair-context remediation for the
hard seeded gate, no-subagent code review remediation, and documentation
closeout.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `coding_agent_phase9_run_supervisor_plan.md` |

### 2026-07-09 coding-agent pre-integration hardening completion

The coding-agent full workflow integration test plan and pre-integration
hardening plan completed after user-approved execution without subagents. The
implementation made the Phase 5-9 workflow reachable from the L2d
accepted-task/background-worker entrypoint, added five executable real LLM
workflow gates, remediated live-gate failures around local source hints and
prompt-safe run reference recovery, reran deterministic regressions, and updated
architecture and ICD documents.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `coding_agent_full_workflow_integration_test_plan.md`, `coding_agent_pre_integration_hardening_plan.md` |

### 2026-07-10 coding-agent assessment gap Phase A completion

The coding-agent assessment gap Phase A plan completed after user-approved
execution without subagents. The implementation added existing-source
`create_file` support, source-backed mixed create/edit routing, source-free
package coherence and alignment gates, durable alignment projection, revision
package-state preservation, and deterministic regressions. Live Gate 02 and
Gate 10 reruns passed after targeted hardening.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `coding_agent_assessment_gap_phase_a_plan.md` |
| Kept active draft follow-up plans | `coding_agent_assessment_gap_phase_b_plan.md`, `coding_agent_assessment_gap_phase_c_plan.md` |

### 2026-07-10 coding-agent assessment gap Phase C completion

The coding-agent assessment gap Phase C plan completed after user-approved
fallback execution without subagents. The implementation added durable typed
blocker continuation, prompt-safe accepted-task run context, trusted approval
provenance, ordered kernel locks, a 30-case benchmark seam, five individually
inspected live outer-loop gates, final remediation for context sanitization, and
lifecycle closeout.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `coding_agent_assessment_gap_phase_c_plan.md` |

### 2026-07-16 relevance turn-settlement completion

The relevance DAG and first-ready settlement cutover completed after
parent-only review remediation, deterministic regressions, and twenty
individually inspected real-LLM gates. The closeout also verified conservative
private-message behavior, shared-route workload bounds, and test realism.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `relevance_turn_settlement_dag_plan.md` |

### 2026-07-16 relevance input-scope robustness completion

The follow-up bugfix completed after production-shaped live LLM gates, local
model prompt-load remediation, private/group scope separation, slot-reference
validation, interleaved-history projection, deterministic regressions, and a
parent-only independent review.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to completed archive | `relevance_input_scope_robustness_bugfix_plan.md` |

### 2026-07-16 relevance native-reply anchor hardening completion

The follow-up bugfix defined the existing settled native-reply Boolean, added a
deterministic response-owner/latest-fragment delivery guard, preserved private
and whole-group behavior, and verified the control-console observation surface.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to completed archive | `relevance_native_reply_anchor_guard_bugfix_plan.md` |

### 2026-08-05 relevance native-reply monotonic delivery completion

The follow-up delivery bugfix completed after the bounded DeepSeek
implementation handoff, deterministic service and adapter regressions, parent
diff review, and lifecycle closeout. The final `/chat` flag now preserves the
existing graph latch and may be promoted deterministically for qualifying group
owner-mismatch or delayed responses without changing the public response schema
or proactive delivery paths.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to completed archive | `relevance_native_reply_monotonic_delivery_plan.md` |

### 2026-08-05 native-reply review remediation completion

The post-review remediation completed after deterministic edge-case coverage,
whitespace-only message-ID validation, HOWTO clarification, focused and
affected regressions, and parent closeout review. The archived implementation
plan remains unchanged.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to completed archive | `relevance_native_reply_review_remediation_plan.md` |

### 2026-07-18 Cognition Core V2 Stage 2 closure

The user accepted the Stage 2 release candidate after the sequential fresh
group/private 20+20 review and the targeted Private-turn-5/Group-turn-15
technical remediation review. The parent plan, frozen companions, and
completed bugfix satellites moved to completed history. The earlier strict
zero-action-narration dialog plan moved to superseded history because the later
accepted policy uses organic discouragement and pass-through model variation.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `cognition_core_v2_stage_2_integration_plan.md`, `cognition_core_v2_stage_2_contract_spec.md`, `cognition_core_v2_stage_2_execution_manifest.md` |
| Moved from active bugfix to completed archive | `action_selection_context_contract_bugfix_plan.md`, `cognition_chain_responsibility_allocation_bugfix_plan.md`, `cognition_core_v2_compositional_action_planning_bugfix_plan.md`, `cognition_core_v2_live_character_judgment_rebalance_plan.md`, `runtime_prompt_chinese_and_dialog_surface_guidance_plan.md` |
| Moved from active bugfix to superseded archive | `dialog_visible_speech_and_semantic_fidelity_bugfix_plan.md` |
| Activated planning boundary | `cognition_core_v2_stage_3_system_adoption_plan.md`, its mandatory execution and change-radius companions, and the Stage 4 production-database placeholder |

### 2026-07-23 Cognition Core V2 Stage 3 closure

Stage 3 completed after the repository-wide non-live collection, affected
console/API/browser gates, static/document checks, fresh-database and real-LLM
evidence review, failure-mode screenshot acceptance, independent review
remediation, and explicit user closure. The in-app Browser had no session; the
user accepted the system-Chrome Playwright screenshots as the visual artifact,
and that environment disposition remains recorded in the archived plans.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `cognition_core_v2_stage_3_system_adoption_plan.md`, `cognition_core_v2_stage_3_execution_manifest.md`, `cognition_core_v2_stage_3_change_radius.md` |
| Stage 4 handoff input | `cognition_core_v2_stage_4_production_database_migration_plan.md` is approved as a one-off database migration; `kazusa_bot_core` stays read-only and later application activation remains user-owned |

### 2026-07-26 Cognition Core V2 transition-coherence completion

The V2-only transition-coherence bugfix completed after unified content and
delivery ownership, contextual semantic-fidelity verification, full surface
repair, accepted-surface propagation, deterministic and live quality gates,
parent review remediation, and explicit user quality sign-off.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to completed archive | `cognition_core_v2_intra_turn_transition_coherence_bugfix_plan.md` |

### 2026-07-26 active plan status cleanup

The active directories were reconciled against each plan's declared status,
completion checklist, execution evidence, and independent review record.
Stale active-table rows pointing to archived records were removed.
Completed plan records and their coding-agent gate-review evidence moved to
completed history. Draft and in-progress plans remain active.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `coding_agent_full_workflow_hardening_plan_2.md`, `cognition_graph_semantic_observability_plan.md` |
| Moved from active bugfix to completed archive | `asuna_private_r18_affinity_harness_plan.md`, `real_history_personality_comparison_fixture_bugfix_plan.md` |
| Moved supporting completion records to completed archive | `coding_agent_full_workflow_hardening_plan_2_llm_gate_reviews.md`, `coding_agent_final_integration_gate_reviews.md` |

### 2026-07-26 Cognition Core V2 baseline hardening accepted closure

The user explicitly accepted the plan's remaining functional and local-LLM
quality residuals and directed closure. The archived record preserves its
unchecked historical gates rather than representing them as completed.
Clean-state deployability passed against the configured `.env`: native database
bootstrap, caller-supplied local profile seeding, package-only neutral example
content, ready vector indexes, standalone health and worker checks, one live
HTTP turn, and a final clean reset and reseed all completed successfully. The
unavailable local container engine remains an accepted external verification
item.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to completed archive by explicit user acceptance with recorded residuals | `cognition_core_v2_baseline_regression_hardening_plan.md` |

### 2026-07-27 Cognition Core V2 retry continuity completion

The V2-only retry-exhaustion bugfix completed after a 17-owner policy cutover,
bounded retries, stage-owned degraded delivery, normal-path service
verification, four one-at-a-time real-model gates, independent review
remediation, and final scope/static regression gates. Service and adapter
production code remained at baseline.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to completed archive | `cognition_core_v2_retry_exhaustion_continuity_bugfix_plan.md` |

### 2026-07-27 Cognition Core V2 model-assignment evaluation closure

The user explicitly closed the completed 384-sample model-assignment
experiment despite its procedural and input-shaping flaws. The archived record
retains its rejected independent-review gate, the rejected aggregate
recommendation, the valid technical call evidence, the `0/192` wrong-target
robustness result, and the user-defined goal criterion results. Production
assignment moved to a separate approved stage-routing plan with exact
environment and Control Console contracts.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive with accepted limitations | `cognition_core_v2_model_assignment_quality_evaluation_plan.md` |
| Added to active short-term | `cognition_core_v2_stage_llm_endpoint_routing_plan.md` |

### 2026-07-27 Cognition Core V2 stage endpoint routing completion

The stage-owned Core V2 route cutover completed after deterministic routing,
configuration, diagnostics, Control Console, browser, and full non-live
verification. The final regression also restored baseline owner-matrix coverage
for the concurrent character-name cutover's dispatcher and message-envelope
documentation paths; this was test-only and left routing behavior unchanged.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `cognition_core_v2_stage_llm_endpoint_routing_plan.md` |

### 2026-07-27 brain-owned adapter character-name completion

The big-bang runtime-adapter cutover made the active process-local brain
profile the only character display-name authority. Discord and NapCat now
receive the required name through registration and heartbeat, enforce it for
platform-bot mention and reply labels, and preserve platform display names
only for human identities. Dispatcher and background output use current brain
identity without rewriting historical conversation rows.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `brain_owned_adapter_character_name_bigbang_plan.md` |

### 2026-07-28 Cognition Core V2 Stage 4 invalidation

| Outcome | Plan |
|---|---|
| Moved from active short-term to superseded archive | `cognition_core_v2_stage_4_production_database_migration_plan.md`; invalidated by user before execution |

### 2026-07-30 Conversation Progress V2 closeout replacement

The user replaced the long in-progress implementation plan with a focused
final-signoff contract. Historical implementation detail remains preserved in
the superseded record. The active successor requires a human-readable semantic
proof of the original failure and an explicitly qualified maximum-turn
projection without requiring raw-data review.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to superseded archive | `conversation_progress_v2_long_thread_continuation_bigbang_plan.md` |
| Added to active bugfix | `conversation_progress_v2_final_signoff_plan.md` |

### 2026-07-31 Cognition Core V2 P0 context reconnection completion

The native V2 conversation graph again consumes cycle-zero globally shared
memory evidence, private past-dialog residual at goal cognition only, and
bounded group-engagement guidance for eligible self-cognition goal/action
judgment. Focused and broader regressions, one-at-a-time live traces, and
independent review remediation completed before user sign-off.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to completed archive | `cognition_core_v2_p0_context_reconnection_bugfix_plan.md` |

### 2026-07-31 Cognition Core V2 prewarm mention-content completion

Cycle-zero shared-memory prewarm now removes only exact typed active-character
mention tokens from its retrieval task, searches the remaining authored
content, and skips retrieval for mention-only turns. Focused and connector
regressions plus independent review completed before archival.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to completed archive | `cognition_core_v2_prewarm_mention_content_query_bugfix_plan.md` |

### 2026-08-01 relevance evidence-grounded admission completion

The relevance admission bugfix completed after evidence-grounded recipient and
character-state validation, focused deterministic and one-at-a-time real-LLM
verification, full non-live regression, independent review remediation, and
explicit user sign-off.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to completed archive | `relevance_evidence_grounded_admission_over_sensitivity_bugfix_plan.md` |

### 2026-08-01 Conversation Progress V2 final sign-off closure

The Conversation Progress V2 final sign-off work completed with its recorded
semantic handoff, maximum-turn projection, regression evidence, and explicit
user approval to archive the plan.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to completed archive | `conversation_progress_v2_final_signoff_plan.md` |

### 2026-08-01 RAG2 public-output contract plan closure

The RAG2 public-output contract plan was closed as no longer relevant before
execution, with no implementation, review, or verification action taken.

| Outcome | Plans |
|---|---|
| Moved from active bugfix to superseded archive | `rag2_public_output_contract_leak_bugfix_plan.md` |

### 2026-08-01 unified task-resolution orchestrator completion

The approved big-bang migration completed with one model-facing
`task_resolution_request`, one bounded resumable orchestrator over four
specialists, v2 accepted-task/background-job persistence, deterministic
future-speak retention, normal result-ready delivery, and exact offline
task-history cutover. Focused, live-LLM, persona, guarded live-DB, production
smoke, and independent-review remediation gates passed. The coding execution
branch remained closed throughout verification.

| Outcome | Plans |
|---|---|
| Moved from active short-term to completed archive | `unified_task_resolution_orchestrator_bigbang_plan.md` |

### 2026-08-03 Cognition Core V2 short-horizon state composition completion

The post-identity V2 global-state composition plan completed after the final
DeepSeek remediation, authoritative deterministic regression, independent
review approval, explicit user acceptance of the documented local-model,
behavioral, and guarded-database exceptions, and final Stage 11 closeout.

| Outcome | Plan |
|---|---|
| Moved from active bugfix to completed archive | `cognition_core_v2_short_horizon_global_state_composition_bigbang_plan.md` |

### 2026-08-04 Cognition Core V2 first-pass robustness accepted closure

The first-pass robustness bugfix completed its production implementation,
deterministic verification, baseline fixture correction, one-at-a-time live
LLM evidence collection, human-readable monologue/dialog review, and
independent code review. The user explicitly approved lifecycle closure while
accepting the recorded Stage 18 live harness, database-precondition,
brief-reply-boundary, and baseline-latency residuals. The archived plan keeps
Stage 18 unchecked as historical evidence rather than representing those
residuals as passing.

| Outcome | Plan |
|---|---|
| Moved from active bugfix to completed archive by explicit user acceptance with recorded residuals | `cognition_core_v2_first_pass_robustness_bugfix_plan.md` |

### 2026-08-11 relevance pre-active answer attribution lifecycle closure

The settled-relevance pre-active answer attribution bugfix has a completed
execution record: all five stages are checked off, deterministic and live
verification evidence is recorded, and the independent review disposition is
approved for closeout. The active registry entry was stale; the plan is now
preserved as completed historical evidence.

| Outcome | Plan |
|---|---|
| Moved from active bugfix to completed archive | `relevance_pre_active_answer_attribution_bugfix_plan.md` |

## Archive

Completed and superseded records live under `archive/`. Use them for historical
lookup, rationale, and execution evidence. New work must not be added to archived
plans.

### Completed Bugfix Records

| Plan |
|---|
| [self_cognition_group_visible_reply_capability_bugfix_plan_20260811.md](archive/completed/bugfix/self_cognition_group_visible_reply_capability_bugfix_plan_20260811.md) |
| [adapter_semantic_identity_boundary_and_memory_pollution_plan.md](archive/completed/bugfix/adapter_semantic_identity_boundary_and_memory_pollution_plan.md) |
| [action_selection_context_contract_bugfix_plan.md](archive/completed/bugfix/action_selection_context_contract_bugfix_plan.md) |
| [character_state_lane_integrity_plan.md](archive/completed/bugfix/character_state_lane_integrity_plan.md) |
| [character_self_image_rolling_state_bugfix_plan.md](archive/completed/bugfix/character_self_image_rolling_state_bugfix_plan.md) |
| [conversation_episode_state_lane_lifecycle_plan.md](archive/completed/bugfix/conversation_episode_state_lane_lifecycle_plan.md) |
| [conversation_progress_v2_final_signoff_plan.md](archive/completed/bugfix/conversation_progress_v2_final_signoff_plan.md) |
| [control_console_functional_remediation_plan.md](archive/completed/bugfix/control_console_functional_remediation_plan.md) |
| [control_console_information_architecture_remediation_plan.md](archive/completed/bugfix/control_console_information_architecture_remediation_plan.md) |
| [control_console_ui_e2e_acceptance_test_plan.md](archive/completed/bugfix/control_console_ui_e2e_acceptance_test_plan.md) |
| [cognition_chain_responsibility_allocation_bugfix_plan.md](archive/completed/bugfix/cognition_chain_responsibility_allocation_bugfix_plan.md) |
| [cognition_core_v2_baseline_regression_hardening_plan.md](archive/completed/bugfix/cognition_core_v2_baseline_regression_hardening_plan.md) |
| [cognition_core_v2_first_pass_robustness_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_first_pass_robustness_bugfix_plan.md) |
| [cognition_core_v2_character_identity_growth_bigbang_plan.md](archive/completed/bugfix/cognition_core_v2_character_identity_growth_bigbang_plan.md) |
| [cognition_core_v2_compositional_action_planning_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_compositional_action_planning_bugfix_plan.md) |
| [cognition_goal_capability_and_workspace_relevance_bugfix_plan.md](archive/completed/bugfix/cognition_goal_capability_and_workspace_relevance_bugfix_plan.md) |
| [cognition_core_v2_failure_capsule_plan.md](archive/completed/bugfix/cognition_core_v2_failure_capsule_plan.md) |
| [cognition_core_v2_live_character_judgment_rebalance_plan.md](archive/completed/bugfix/cognition_core_v2_live_character_judgment_rebalance_plan.md) |
| [cognition_core_v2_p0_context_reconnection_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_p0_context_reconnection_bugfix_plan.md) |
| [cognition_core_v2_prewarm_mention_content_query_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_prewarm_mention_content_query_bugfix_plan.md) |
| [cognition_core_v2_intra_turn_transition_coherence_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_intra_turn_transition_coherence_bugfix_plan.md) |
| [cognition_core_v2_prompt_budget_and_failure_containment_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_prompt_budget_and_failure_containment_bugfix_plan.md) |
| [cognition_core_v2_relational_authority_transfer_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_relational_authority_transfer_bugfix_plan.md) |
| [cognition_core_v2_short_horizon_global_state_composition_bigbang_plan.md](archive/completed/bugfix/cognition_core_v2_short_horizon_global_state_composition_bigbang_plan.md) |
| [cognition_core_v2_retry_exhaustion_continuity_bugfix_plan.md](archive/completed/bugfix/cognition_core_v2_retry_exhaustion_continuity_bugfix_plan.md) |
| [cognition_core_v2_semantic_appraisal_partial_failure_mitigation_plan.md](archive/completed/bugfix/cognition_core_v2_semantic_appraisal_partial_failure_mitigation_plan.md) |
| [asuna_private_r18_affinity_harness_plan.md](archive/completed/bugfix/asuna_private_r18_affinity_harness_plan.md) |
| [real_history_personality_comparison_fixture_bugfix_plan.md](archive/completed/bugfix/real_history_personality_comparison_fixture_bugfix_plan.md) |
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
| [relevance_input_scope_robustness_bugfix_plan.md](archive/completed/bugfix/relevance_input_scope_robustness_bugfix_plan.md) |
| [relevance_evidence_grounded_admission_over_sensitivity_bugfix_plan.md](archive/completed/bugfix/relevance_evidence_grounded_admission_over_sensitivity_bugfix_plan.md) |
| [relevance_pre_active_answer_attribution_bugfix_plan.md](archive/completed/bugfix/relevance_pre_active_answer_attribution_bugfix_plan.md) |
| [required_selection_partial_recovery_bugfix_plan.md](archive/completed/bugfix/required_selection_partial_recovery_bugfix_plan.md) |
| [reflection_group_scene_digest_self_cognition_bugfix_plan.md](archive/completed/bugfix/reflection_group_scene_digest_self_cognition_bugfix_plan.md) |
| [reflection_global_promotion_replay_bugfix_plan.md](archive/completed/bugfix/reflection_global_promotion_replay_bugfix_plan.md) |
| [relevance_native_reply_anchor_guard_bugfix_plan.md](archive/completed/bugfix/relevance_native_reply_anchor_guard_bugfix_plan.md) |
| [resolver_image_only_empty_input_bugfix_plan.md](archive/completed/bugfix/resolver_image_only_empty_input_bugfix_plan.md) |
| [runtime_prompt_chinese_and_dialog_surface_guidance_plan.md](archive/completed/bugfix/runtime_prompt_chinese_and_dialog_surface_guidance_plan.md) |
| [semantic_appraisal_terminal_transition_reliability_plan.md](archive/completed/bugfix/semantic_appraisal_terminal_transition_reliability_plan.md) |
| [settled_relevance_logical_history_projection_bugfix_plan.md](archive/completed/bugfix/settled_relevance_logical_history_projection_bugfix_plan.md) |
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
| [august_change_alignment_audit_and_remediation_plan.md](archive/completed/short_term/august_change_alignment_audit_and_remediation_plan.md) |
| [backend_control_console_development_plan.md](archive/completed/short_term/backend_control_console_development_plan.md) |
| [backend_control_console_web_test_plan.md](archive/completed/short_term/backend_control_console_web_test_plan.md) |
| [background_artifact_handoff_poc_plan.md](archive/completed/short_term/background_artifact_handoff_poc_plan.md) |
| [background_work_semantic_lifecycle_plan.md](archive/completed/short_term/background_work_semantic_lifecycle_plan.md) |
| [brain_owned_adapter_character_name_bigbang_plan.md](archive/completed/short_term/brain_owned_adapter_character_name_bigbang_plan.md) |
| [cache2_agent_stats_health_plan.md](archive/completed/short_term/cache2_agent_stats_health_plan.md) |
| [channel_name_semantic_projection_plan.md](archive/completed/short_term/channel_name_semantic_projection_plan.md) |
| [character_local_time_context_plan.md](archive/completed/short_term/character_local_time_context_plan.md) |
| [character_profile_runtime_state_split_plan.md](archive/completed/short_term/character_profile_runtime_state_split_plan.md) |
| [coding_agent_assessment_gap_phase_a_plan.md](archive/completed/short_term/coding_agent_assessment_gap_phase_a_plan.md) |
| [coding_agent_final_integration_gate_reviews.md](archive/completed/short_term/coding_agent_final_integration_gate_reviews.md) |
| [coding_agent_full_workflow_hardening_plan_2.md](archive/completed/short_term/coding_agent_full_workflow_hardening_plan_2.md) |
| [coding_agent_full_workflow_hardening_plan_2_llm_gate_reviews.md](archive/completed/short_term/coding_agent_full_workflow_hardening_plan_2_llm_gate_reviews.md) |
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
| [cognition_graph_semantic_observability_plan.md](archive/completed/short_term/cognition_graph_semantic_observability_plan.md) |
| [cognition_core_v2_stage_1_validation_plan.md](archive/completed/short_term/cognition_core_v2_stage_1_validation_plan.md) |
| [cognition_core_v2_stage_2_contract_spec.md](archive/completed/short_term/cognition_core_v2_stage_2_contract_spec.md) |
| [cognition_core_v2_stage_2_execution_manifest.md](archive/completed/short_term/cognition_core_v2_stage_2_execution_manifest.md) |
| [cognition_core_v2_stage_2_integration_plan.md](archive/completed/short_term/cognition_core_v2_stage_2_integration_plan.md) |
| [cognition_core_v2_stage_3_change_radius.md](archive/completed/short_term/cognition_core_v2_stage_3_change_radius.md) |
| [cognition_core_v2_stage_3_execution_manifest.md](archive/completed/short_term/cognition_core_v2_stage_3_execution_manifest.md) |
| [cognition_core_v2_stage_3_system_adoption_plan.md](archive/completed/short_term/cognition_core_v2_stage_3_system_adoption_plan.md) |
| [cognition_core_v2_model_assignment_quality_evaluation_plan.md](archive/completed/short_term/cognition_core_v2_model_assignment_quality_evaluation_plan.md) |
| [cognition_core_v2_stage_llm_endpoint_routing_plan.md](archive/completed/short_term/cognition_core_v2_stage_llm_endpoint_routing_plan.md) |
| [cognition_llm_stage_reconnection_plan.md](archive/completed/short_term/cognition_llm_stage_reconnection_plan.md) |
| [cognition_visual_directives_control_plan.md](archive/completed/short_term/cognition_visual_directives_control_plan.md) |
| [cognition_state_integrity_plan.md](archive/completed/short_term/cognition_state_integrity_plan.md) |
| [daily_affect_settling_plan.md](archive/completed/short_term/daily_affect_settling_plan.md) |
| [complex_task_resolver_capability_plan.md](archive/completed/short_term/complex_task_resolver_capability_plan.md) |
| [control_console_auto_model_discovery_picker_plan.md](archive/completed/short_term/control_console_auto_model_discovery_picker_plan.md) |
| [control_console_brain_model_route_config_plan.md](archive/completed/short_term/control_console_brain_model_route_config_plan.md) |
| [control_console_cognition_debug_visibility_plan.md](archive/completed/short_term/control_console_cognition_debug_visibility_plan.md) |
| [control_console_entity_information_architecture_plan.md](archive/completed/short_term/control_console_entity_information_architecture_plan.md) |
| [control_console_information_contract_v2_remediation_plan.md](archive/completed/short_term/control_console_information_contract_v2_remediation_plan.md) |
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
| [rag3_subagent_framework_and_media_inspection_bigbang_plan.md](archive/completed/short_term/rag3_subagent_framework_and_media_inspection_bigbang_plan.md) |
| [rag_agent_package_reorganization_plan.md](archive/completed/short_term/rag_agent_package_reorganization_plan.md) |
| [rag_cache2_persistent_initializer_plan.md](archive/completed/short_term/rag_cache2_persistent_initializer_plan.md) |
| [rag_current_turn_exclusion_plan.md](archive/completed/short_term/rag_current_turn_exclusion_plan.md) |
| [rag_phase3_development_plan.md](archive/completed/short_term/rag_phase3_development_plan.md) |
| [rag_reply_mention_and_vague_input_plan.md](archive/completed/short_term/rag_reply_mention_and_vague_input_plan.md) |
| [recall_agent_plan.md](archive/completed/short_term/recall_agent_plan.md) |
| [relevance_turn_settlement_dag_plan.md](archive/completed/short_term/relevance_turn_settlement_dag_plan.md) |
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
| [unified_task_resolution_orchestrator_bigbang_plan.md](archive/completed/short_term/unified_task_resolution_orchestrator_bigbang_plan.md) |
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
| [conversation_progress_v2_long_thread_continuation_bigbang_plan.md](archive/superseded/conversation_progress_v2_long_thread_continuation_bigbang_plan.md) |
| [cognition_preserving_goal_resolver_production_plan.md](archive/superseded/cognition_preserving_goal_resolver_production_plan.md) |
| [cognition_core_v2_stage_4_production_database_migration_plan.md](archive/superseded/cognition_core_v2_stage_4_production_database_migration_plan.md) |
| [dialog_visible_speech_and_semantic_fidelity_bugfix_plan.md](archive/superseded/dialog_visible_speech_and_semantic_fidelity_bugfix_plan.md) |
| [graph_rag_recall_experiment_plan.md](archive/superseded/graph_rag_recall_experiment_plan.md) |
| [goal_resolver_poc_plan.md](archive/superseded/goal_resolver_poc_plan.md) |
| [self_cognition_loop_architecture.md](archive/superseded/self_cognition_loop_architecture.md) |
| [self_cognition_reasoning_basis.md](archive/superseded/self_cognition_reasoning_basis.md) |
| [self_cognition_tracking_icd.md](archive/superseded/self_cognition_tracking_icd.md) |
| [rag_supervisor2_inner_loop_agents_plan.md](archive/superseded/rag_supervisor2_inner_loop_agents_plan.md) |
| [rag2_public_output_contract_leak_bugfix_plan.md](archive/superseded/rag2_public_output_contract_leak_bugfix_plan.md) |
| [short_circuit_early_stop_plan.md](archive/superseded/short_circuit_early_stop_plan.md) |
| [coding_agent_phase2_code_writing_plan_superseded_20260623.md](archive/superseded/coding_agent_phase2_code_writing_plan_superseded_20260623.md) |
| [visual_descriptor_seeded_reference_images_plan.md](archive/superseded/visual_descriptor_seeded_reference_images_plan.md) |
