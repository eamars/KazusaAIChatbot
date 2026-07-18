# Cognition Core V2 Stage 2 Execution Manifest

## Document Control

- Parent plan: `cognition_core_v2_stage_2_integration_plan.md`
- Contract: `cognition_core_v2_stage_2_contract_spec.md`
- Status: completed.
- Record role: frozen companion to the completed parent plan.

### 2026-07-17 Checkpoint I corrective packet

The active
`development_plans/archive/completed/bugfix/cognition_core_v2_compositional_action_planning_bugfix_plan.md`
is a mandatory Checkpoint I remediation packet. Its production-capacity,
compositional planner, full resolver lifecycle, operational-error containment,
disabled visual-default, focused live-LLM, frozen 20+20, restoration, and
independent-review gates must pass before final Stage 2 closure. Its big-bang
contract supersedes route-only action-selection requirements in earlier
packets.
- Purpose: exact change surface, test ownership, database isolation, commands,
  checkpoints, evidence paths, and handoffs
- Change rule: path, test, database, or checkpoint changes require a parent-plan
  update before the affected checkpoint begins

## 1. Release-Candidate Boundary

The checked-out source branch is the Stage 2 release candidate. The deployed
service and production MongoDB are external systems and are never contacted by
Stage 2 commands. Source coexistence is permitted only while V2 is being built:

- callers continue using V1 until Checkpoint F;
- Checkpoint F performs the one caller switch to V2;
- after Checkpoint F there is one active V2 call path while the unreferenced V1
  package still exists on disk;
- Checkpoint H deletes V1 after isolated live evidence passes;
- no feature flag, dual invocation, fallback import, adapter, or translation
  layer is created.

The Checkpoint A V1 comparison baseline uses synthetic DB-free inputs and a
scripted deterministic LLM double. Checkpoint G performs the real V1/V2 model
latency comparison from guarded synthetic seed data while V1 remains available.

## 2. Exact Change Surface

### Create

| Path | Required symbol/ownership |
|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v2/state_models.py` | All persistent V2 types, defaults, caps, scalar validation, and state-scope validation from the contract spec. |
| `src/kazusa_ai_chatbot/cognition_core_v2/transition_guards.py` | Goal/threat/event/gap FSMs, event-comparison guards, direct-fact table, emotion begin/sustain/fade guards, retention protection, and delta allowlist. |
| `src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py` | `plan_semantic_questions(...)` with at most six unique target-path owners. |
| `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py` | Central numeric/time/id/control-to-semantic projection and prompt-sentinel validation seam. |
| `src/kazusa_ai_chatbot/cognition_core_v2/surface.py` | Public `run_text_surface_planning(...)` orchestrator. |
| `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py` | Four stage-local L3 prompt/LLM/handler blocks ported from V1. |
| `tests/live_llm_mongo.py` | Guard assertion, idempotent synthetic seed, unique owner ids, and singleton snapshot fixture. |
| `tests/fixtures/cognition_core_v2_mongo_seed.json` | One native-V2 character singleton, two users, histories, memories, active causes, and shared evidence. |
| `tests/test_cognition_core_v2_alignment_gates.py` | RCA tripwires and two-phase boundary-shape tests. |
| `tests/test_cognition_core_v2_projection.py` | Semantic bands, duration labels, sentinel leak checks, and free-form descriptor acceptance. |
| `tests/test_cognition_core_v2_failures.py` | Every failure/commit row from the contract spec. |
| `tests/test_cognition_core_v2_integration.py` | Facade, state commit order, V2 L3, action/resolver, and service smoke tests. |
| `tests/test_live_llm_mongo_isolation.py` | Exact-name guard, pre-import binding, idempotent seed, owner isolation, singleton restore, and no-xdist checks. |

### Modify: core package

| Path | Exact responsibility |
|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v2/__init__.py` | Export only V2 contracts and the two public entrypoints. |
| `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` | Replace every V1 import/type with the exact V2 public, bid, diagnostic, service, and surface contracts. |
| `src/kazusa_ai_chatbot/cognition_core_v2/facade.py` | Replace `run_cognition_chain` with `run_cognition(input_payload, services)`. |
| `src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py` | Implement fixed reducer order and one-scope `StateUpdateV2`. |
| `src/kazusa_ai_chatbot/cognition_core_v2/emotion_definitions.py` | Encode the frozen 21-row registry and decay rates. |
| `src/kazusa_ai_chatbot/cognition_core_v2/emotion_derivation.py` | Recompute/validate activation cache and lifecycle. |
| `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py` | Run one scoped question per call; return prompt handles, propositions, and allowlisted deltas. |
| `src/kazusa_ai_chatbot/cognition_core_v2/branch_activation.py` | Use the frozen fourteen goal branches and `MAX_GOAL_BRANCHES = 14`. |
| `src/kazusa_ai_chatbot/cognition_core_v2/dependency_graph.py` | Validate appraisal/branch dependencies, missing refs, and cycles. |
| `src/kazusa_ai_chatbot/cognition_core_v2/parallel_executor.py` | Remove the semaphore; preserve dependency readiness, timings, failures, and isolated result slots. |
| `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` | Emit complete `ActionBidV2` without persistent-goal mutation. |
| `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py` | Validate the prompt-local admitted-handle partition, map it internally, and copy complete primary/supporting/competing bids without field synthesis. |
| `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py` | Route-only selection with action availability validation. |
| `src/kazusa_ai_chatbot/cognition_core_v2/output_projection.py` | Emit one state update and semantic affect/relationship/diagnostic projections. |
| `src/kazusa_ai_chatbot/cognition_core_v2/diagnostics.py` | Bounded protected two-phase, appraisal, branch, collapse, latency, and failure diagnostics. |
| `src/kazusa_ai_chatbot/cognition_core_v2/validation_cli.py` | Replace V1 facade use with V2 fixture-driven lifecycle/benchmark commands. |
| `src/kazusa_ai_chatbot/cognition_core_v2/README.md` | Publish the two public APIs, state ownership, two-phase flow, failure boundary, and prompt rules. |

### Modify: DB and bootstrap

| Path | Exact responsibility |
|---|---|
| `src/kazusa_ai_chatbot/db/schemas.py` | Add embedded V2 user/character state types; remove runtime affinity and prose-affect fields. |
| `src/kazusa_ai_chatbot/db/users.py` | Add acquaintance default, validated state read, and one-document replacement; remove affinity/relationship-insight APIs. |
| `src/kazusa_ai_chatbot/db/character.py` | Add validated singleton V2 state read/replacement; remove prose-affect mutation. |
| `src/kazusa_ai_chatbot/db/__init__.py` | Export V2 state facades and remove legacy exports. |
| `src/kazusa_ai_chatbot/db/bootstrap.py` | Create missing character default and validate existing native-V2 state. |
| `src/kazusa_ai_chatbot/db/_client.py` | When `KAZUSA_TEST_DB_GUARD=1`, reject every name except `_test_kazusa_live_llm` before client creation. |
| `src/kazusa_ai_chatbot/db/README.md` | Replace affinity/profile persistence documentation with the native V2 user/character state contract. |
| `src/kazusa_ai_chatbot/config.py` | Remove affinity constants; retain ordinary DB config; add only contract-spec V2 constants. |
| `src/kazusa_ai_chatbot/utils.py` | Remove `build_affinity_block` and route callers to V2 semantic projections. |

### Modify: one canonical upstream path

| Path | Exact responsibility |
|---|---|
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | Replace V1 builders/services/apply functions with V2 input, services, call, and output application. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` | Build `TextSurfaceInputV2` and call the V2 surface API. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` | Replace V1 chain fields with exact V2 input/output/working-state fields. |
| `src/kazusa_ai_chatbot/state.py` | Carry V2 episode state through the top-level graph. |
| `src/kazusa_ai_chatbot/cognition_resolver/contracts.py` | Replace V1 output dependencies with V2 intention, requests, progress, and working state. |
| `src/kazusa_ai_chatbot/cognition_resolver/state.py` | Carry episode-local V2 state across recurrence. |
| `src/kazusa_ai_chatbot/cognition_resolver/loop.py` | Resume V2 cognition without DB reload and commit only the final recurrence. |
| `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py` | Return typed observations/direct facts without state authority. |
| `src/kazusa_ai_chatbot/cognition_resolver/telemetry.py` | Emit bounded V2 request/progress/status fields. |
| `src/kazusa_ai_chatbot/cognition_resolver/README.md` | Replace V1 resolver contract description. |
| `src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py` | Build a character-scoped V2 episode with promoted evidence. |
| `src/kazusa_ai_chatbot/internal_thought_cognition.py` | Build a character-scoped V2 episode without treating residue prose as facts. |
| `src/kazusa_ai_chatbot/background_work/result_source.py` | Emit typed accepted-task outcomes. |
| `src/kazusa_ai_chatbot/calendar_scheduler/handlers.py` | Emit typed scheduler outcomes and source identity. |
| `src/kazusa_ai_chatbot/self_cognition/group_review_participant_context.py` | Replace affinity projection with semantic relationship context. |
| `src/kazusa_ai_chatbot/self_cognition/models.py` | Carry V2 scope and state result. |
| `src/kazusa_ai_chatbot/self_cognition/projection.py` | Use V2 semantic projections. |
| `src/kazusa_ai_chatbot/self_cognition/sources.py` | Emit trusted typed source events only. |
| `src/kazusa_ai_chatbot/self_cognition/runner.py` | Call the canonical V2 persona path and persist the single mutable scope. |
| `src/kazusa_ai_chatbot/self_cognition/worker.py` | Preserve current schedule/delivery boundaries with V2 results. |
| `src/kazusa_ai_chatbot/self_cognition/README.md` | Replace affinity-lane documentation with the V2 semantic relationship/state boundary. |

### Modify: RAG/context and downstream runtime

| Path | Exact responsibility |
|---|---|
| `src/kazusa_ai_chatbot/rag/person_context/workers/relationship.py` | Return semantic V2 relationship evidence instead of affinity rank. |
| `src/kazusa_ai_chatbot/rag/prompt_projection.py` | Remove affinity fields and use bounded V2 semantic relationship projection. |
| `src/kazusa_ai_chatbot/rag/cache2_policy.py` | Replace affinity dependency keys with V2 relationship-state dependency. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` | Consume the V2 RAG relationship projection. |
| `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py` | Replace affinity context with semantic relationship context. |
| `src/kazusa_ai_chatbot/nodes/dialog_agent.py` | Consume `TextSurfaceOutputV2`; retain final wording ownership. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py` | Materialize V2 semantic action requests through existing permission checks. |
| `src/kazusa_ai_chatbot/nodes/README.md` | Replace V1 layer, affinity, and prose-affect examples with the canonical V2 flow and payload boundaries. |
| `src/kazusa_ai_chatbot/action_spec/models.py` | Accept V2 action-request fields without changing executable action authority. |
| `src/kazusa_ai_chatbot/action_spec/evaluator.py` | Validate V2 requested action against availability and permission. |
| `src/kazusa_ai_chatbot/action_spec/execution.py` | Return typed action outcomes for later cognition. |
| `src/kazusa_ai_chatbot/action_spec/results.py` | Project typed outcomes without authoring emotion state. |
| `src/kazusa_ai_chatbot/brain_service/graph.py` | Order V2 cognition, single-scope persistence, L3/action/dialog, and terminal handling. |
| `src/kazusa_ai_chatbot/brain_service/post_turn.py` | Remove emotion/relationship authoring; retain memory, progress, and residue work. |
| `src/kazusa_ai_chatbot/service.py` | Load/refresh V2 character state and preserve service error/no-message behavior. |
| `src/kazusa_ai_chatbot/consolidation/core.py` | Remove affinity/prose-affect write lanes. |
| `src/kazusa_ai_chatbot/consolidation/lane_router.py` | Remove legacy lanes while retaining memory/self-image lanes. |
| `src/kazusa_ai_chatbot/consolidation/origin_policy.py` | Remove affinity target policy. |
| `src/kazusa_ai_chatbot/consolidation/persistence.py` | Stop authoring V2 cognition state from LLM output. |
| `src/kazusa_ai_chatbot/consolidation/reflection.py` | Stop prose mood/relationship writes. |
| `src/kazusa_ai_chatbot/consolidation/schema.py` | Remove legacy lane contracts. |
| `src/kazusa_ai_chatbot/consolidation/target.py` | Remove legacy targets. |
| `src/kazusa_ai_chatbot/consolidation/images.py` | Replace affinity input with semantic relationship projection. |
| `src/kazusa_ai_chatbot/consolidation/README.md` | Remove affinity/prose-affect writer documentation and state that consolidation cannot author V2 cognition state. |
| `src/kazusa_ai_chatbot/reflection_cycle/affect_settling.py` | Keep schedule/run key; replace two LLM calls with deterministic `sleep_recovery`. |
| `src/kazusa_ai_chatbot/reflection_cycle/README.md` | Document deterministic sleep recovery and run artifact. |
| `src/kazusa_ai_chatbot/event_logging/models.py` | Add bounded V2 component/branch/state-commit event fields. |
| `src/kazusa_ai_chatbot/event_logging/sanitization.py` | Redact raw state, ids, numbers promised semantic-only, prompt text, and private bids. |
| `src/kazusa_ai_chatbot/event_logging/schemas.py` | Register bounded V2 event schemas. |
| `src/kazusa_ai_chatbot/event_logging/snapshots.py` | Store bounded graph/diagnostic summaries for Stage 3 consumption. |

### Modify: test infrastructure and exact regression owners

| Path | Required change |
|---|---|
| `tests/conftest.py` | After `load_dotenv`, set exact test DB and guard before any `kazusa_ai_chatbot` import; add session close fixture. |
| `pytest.ini` | Add `norecursedirs = tests/fixtures/coding_agent_full_workflow` because it is a fixture project, not a pytest suite. |
| `tests/test_cognition_core_v2_contracts.py` | Replace V1 facade assertions with exact V2 API/state/bid/surface contracts. |
| `tests/test_cognition_core_v2_state.py` | Test exact defaults, scopes, caps, FSMs, pruning, restart reload, and one-scope output. |
| `tests/test_cognition_core_v2_emotion_lifecycle.py` | Test all registry formulas and natural begin/sustain/fade/negative cases. |
| `tests/test_cognition_core_v2_dependencies.py` | Test two-phase readiness, no semaphore, unique slots, dependency failures, and fourteen-branch activation. |
| `tests/test_cognition_core_v2_benchmark.py` | Through Checkpoint G, keep scripted and guarded real-model V1/V2 comparisons; at Checkpoint H remove the V1 import/comparison node and retain V2 call/context/latency/state-I/O regression cases plus recorded G evidence. |
| `tests/test_cognition_core_v2_live_llm.py` | Replace Stage 1 cases with the exact Stage 2 `live_llm` plus `live_db` cases below; every case loads seeded V2 state through the guard. |
| `tests/fixtures/cognition_core_v2_emotion_lifecycle_cases.json` | Replace injected root strengths and generic fade text with typed natural begin/sustain/fade/negative cause sequences for all twenty-one registry emotions. |
| `tests/fixtures/cognition_core_v2_benchmark_cases.json` | Replace Stage 1 V1-compatible payloads with scripted equivalent V1/V2 cases through Gate G and V2-only retained cases at Gate H. |
| `tests/test_cognition_chain_connector_mapping.py` | Replace V1 mapping with V2 persona mapping. |
| `tests/test_cognition_resolver_contracts.py` | Replace V1 resolver payload expectations. |
| `tests/test_cognition_resolver_loop.py` | Test working-state carry, pending terminal commit, and no DB reload. |
| `tests/test_l2d_l3_surface_handoff.py` | Replace V1 L2d/L3 payload with V2 intention/surface contract. |
| `tests/test_dialog_agent.py` | Assert dialog receives no raw state and owns final wording. |
| `tests/test_reflection_affect_settling.py` | Test schedule retention, idempotency, deterministic recovery, and zero LLM calls. |
| `tests/test_service_background_consolidation.py` | Test consolidation cannot author cognition state. |
| `tests/test_self_cognition_integration.py` | Test character/user scope selection and one-document commit. |
| `tests/test_action_spec_models.py` | Test V2 semantic request mapping. |
| `tests/test_action_spec_evaluator.py` | Test permission/availability stays deterministic. |
| `tests/test_action_spec_results.py` | Test typed outcome re-entry. |
| `tests/test_event_logging_interface.py` | Test bounded V2 component metadata. |
| `tests/test_service_event_logging.py` | Test protected V2 state/branch redaction. |
| `tests/fixtures/rag_agent_package_prompt_baseline.json` | Replace affinity prompt fields with semantic V2 relationship fields. |

### Checkpoint I change-surface reconciliation

The Checkpoint H alignment review found committed Stage 2 changes outside the
original literal path tables. These paths are required to complete the frozen
caller, telemetry, consolidation, lifecycle, and regression contracts; they
are therefore part of the exact Stage 2 surface for independent review and
remediation. This reconciliation adds no compatibility path, public contract,
LLM call, persistence lane, or production cutover operation.

| Path | Bounded Stage 2 responsibility |
|---|---|
| `llm_test_helpers.py` | Keep shared live-test helper behavior compatible with the V2 test taxonomy and retained trace contract. |
| `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py` | Preserve the validated V2 coding-task action payload during deterministic accepted-task materialization. |
| `src/kazusa_ai_chatbot/consolidation/source_policy.py` | Remove retired affect/relationship writer-source policy while preserving retained memory sources. |
| `src/kazusa_ai_chatbot/event_logging/__init__.py` | Export the bounded V2 telemetry recording/status surface. |
| `src/kazusa_ai_chatbot/event_logging/recording.py` | Record bounded V2 component, resolver, route, and state-commit events without raw state or prompt payloads. |
| `src/kazusa_ai_chatbot/event_logging/status.py` | Include bounded V2 event families in aggregate status without exposing protected data. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | Wire the canonical V2 persona flow, terminal handling, and one-scope commit ordering at the existing graph owner. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py` | Preserve memory-lifecycle specialist behavior after the V2 action/surface handoff. |
| `src/kazusa_ai_chatbot/reflection_cycle/models.py` | Align deterministic sleep-recovery run metadata with the frozen run artifact. |
| `src/kazusa_ai_chatbot/self_cognition/tracking.py` | Track V2 self-cognition route outcomes without retired V1/affinity fields. |
| `src/scripts/sanitize_memory_writer_perspective.py` | Keep the maintenance sanitizer importable after removal of retired relationship-insight fields; it remains outside runtime cognition authority. |
| `tests/cognition_core_v2_test_helpers.py` | Own reusable deterministic V2 test fixtures without hiding behavioral assertions. |

The following retained regression owners were also changed to replace legacy
V1/affinity assumptions or to preserve their existing subsystem behavior after
the V2 big-bang switch. The parent owns all Checkpoint I test remediation in
these paths:

```text
tests/control_console_e2e/test_visual_product_acceptance_e2e.py
tests/test_accepted_task_prompt_contract.py
tests/test_action_selection_media_affordance.py
tests/test_background_work_delivery.py
tests/test_calendar_scheduler_active_commitments.py
tests/test_coding_agent_background_run_contracts.py
tests/test_coding_agent_full_workflow_integration_live_llm.py
tests/test_coding_agent_image_reading_acceptance.py
tests/test_coding_agent_phase_c_run_context_contracts.py
tests/test_cognition_preference_adapter.py
tests/test_cognition_resolver_persona_graph.py
tests/test_cognition_stage_connection_live_llm.py
tests/test_control_console_config_routes.py
tests/test_db_public_boundary.py
tests/test_documentation_harmonization.py
tests/test_event_logging_status.py
tests/test_l2d_action_selection_live_llm.py
tests/test_l2d_unknown_context_resolver_live_llm.py
tests/test_l3_dialog_content_plan_contract.py
tests/test_llm_time_payload_projection.py
tests/test_memory_writer_prompt_contracts.py
tests/test_multi_source_cognition_stage_03_prompt_selection.py
tests/test_persona_supervisor2_action_selection.py
tests/test_rag_projection.py
tests/test_self_cognition_duplicate_response_live_llm.py
tests/test_self_cognition_response_sensitivity_live_llm.py
```

### Rewrite-in-place legacy-reference test inventory

The parent owns rewriting every path below. Each path currently contains an
affinity or prose-affect dependency but also tests behavior retained by V2. The
rewrite preserves that behavior and replaces only its legacy state/prompt
contract:

| Legacy test dependency | Mandatory disposition |
|---|---|
| V1 input/output/stage import | Replace with the exact V2 public/surface contract; pure V1 contract coverage moves to the named V2 contract/alignment tests. |
| Affinity prompt/rank | Replace with `SemanticRelationshipProjectionV2` and assert raw relationship numbers stay outside the prompt. |
| Affinity DB default/update/clamp | Replace with acquaintance native-state creation, guarded relationship-axis delta, cap, and one-document replacement tests. |
| `last_relationship_insight` | Replace with semantic relationship projection assertions; consolidation tests assert that consolidation cannot author the underlying V2 state. |
| `global_vibe`/`reflection_summary`/prose mood writer | Replace with native character cognition-state read/replacement and deterministic sleep-recovery assertions. |
| Affinity-only helper/class/function | Remove that local block after its behavior is covered by `test_cognition_core_v2_projection.py` or `test_cognition_core_v2_state.py`; retain the rest of the file. |
| Retained test identifier containing a removed term | Rename by exact vocabulary: `affinity` to `relationship_state`, `last_relationship_insight` to `semantic_relationship_projection`, and `global_vibe`/`reflection_summary` to `character_cognition_state`. |

The parent performs these rewrites with Checkpoint F's caller switch so the
retained tests exercise V2 before Checkpoint G. Checkpoint H removes the four
obsolete files and verifies the final static inventory.

```text
tests/fixtures/rag_agent_package_prompt_baseline.json
tests/test_action_selection_payload.py
tests/test_action_selection_prompt_contract.py
tests/test_background_work_future_speak_live_llm.py
tests/test_boundary_core_sensitivity_live_llm.py
tests/test_coding_agent_phase3_handoff_e2e.py
tests/test_coding_agent_phase3_live_e2e.py
tests/test_cognition_chain_connector_mapping.py
tests/test_cognition_clarification_consumers.py
tests/test_cognition_core_v2_benchmark.py
tests/test_cognition_core_v2_contracts.py
tests/test_cognition_core_v2_live_llm.py
tests/test_cognition_current_event_grounding.py
tests/test_cognition_interaction_style_context.py
tests/test_cognition_live_llm_prompt_contracts.py
tests/test_cognition_live_llm.py
tests/test_cognition_prompt_contract_text.py
tests/test_cognition_referents_live_llm.py
tests/test_cognition_resolver_l2d_contract.py
tests/test_cognition_resolver_loop.py
tests/test_config.py
tests/test_consolidation_evidence_hardening_live_llm.py
tests/test_consolidation_lane_bigbang_integration.py
tests/test_consolidation_lane_router_contract.py
tests/test_consolidation_lifecycle_diagnostics.py
tests/test_consolidation_memory_write_use_cases_live_llm.py
tests/test_consolidation_origin_metadata.py
tests/test_consolidation_origin_policy.py
tests/test_consolidation_target_routing.py
tests/test_consolidator_character_image.py
tests/test_consolidator_efficiency.py
tests/test_consolidator_group_channel_branch.py
tests/test_consolidator_origin_policy_db_writer.py
tests/test_consolidator_origin_selection.py
tests/test_consolidator_reflection_prompts.py
tests/test_consolidator_source_aware_payloads.py
tests/test_conversation_progress_cognition.py
tests/test_conversation_progress_flow_live_llm.py
tests/test_conversation_progress_history_policy.py
tests/test_conversation_progression_live_llm.py
tests/test_db_writer_cache2_invalidation.py
tests/test_db.py
tests/test_dialog_agent_direct_live_llm.py
tests/test_dialog_agent.py
tests/test_dialog_anchor_boundary_live_llm.py
tests/test_dialog_first_person_perspective_live_llm.py
tests/test_dialog_generator_live_llm_contract.py
tests/test_dialog_inline_mentions_live_llm.py
tests/test_dialog_l3_surface_contract_live_llm.py
tests/test_dialog_mention_target_user.py
tests/test_dialog_message_sequence_live_llm.py
tests/test_e2e_live_llm.py
tests/test_global_character_growth_replay.py
tests/test_image_cognition_options_live_llm.py
tests/test_internal_monologue_residue_prompt_boundaries.py
tests/test_kazusa_victory_anchor_live_llm.py
tests/test_l2d_l3_surface_handoff.py
tests/test_l2d_quiet_monologue_live_llm.py
tests/test_l3_dialog_content_plan_live_llm.py
tests/test_local_context_resolver_integration.py
tests/test_memory_writer_database_sanitizer.py
tests/test_memory_writer_perspective_live_llm.py
tests/test_msg_decontexualizer.py
tests/test_multi_source_cognition_image_input.py
tests/test_multi_source_cognition_stage_00_regression_baseline.py
tests/test_multi_source_cognition_stage_02_chat_episode_migration.py
tests/test_multi_source_cognition_stage_07_reflection_dry_run.py
tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py
tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py
tests/test_past_dialog_cognition_prompt_boundaries.py
tests/test_persona_relevance_agent.py
tests/test_persona_supervisor2_cognition_prewarm.py
tests/test_persona_supervisor2_rag_skip_shape.py
tests/test_persona_supervisor2_rag2_integration.py
tests/test_persona_supervisor2_schema.py
tests/test_persona_supervisor2.py
tests/test_rag_agent_package_prompt_stability.py
tests/test_rag_cognitive_episode_adapter.py
tests/test_rag_dialog_event_logging.py
tests/test_reflection_affect_settling_live_llm.py
tests/test_reflection_affect_settling.py
tests/test_relevance_reply_to_bot_live_llm.py
tests/test_relevance_sensitivity_live_llm.py
tests/test_self_cognition_delivery_target.py
tests/test_self_cognition_group_review_participant_context.py
tests/test_self_cognition_group_review_source.py
tests/test_self_cognition_integration.py
tests/test_self_cognition_memory_lifecycle_live_llm.py
tests/test_self_cognition_tracking.py
tests/test_service_background_consolidation.py
tests/test_service_input_queue.py
tests/test_shared_memory_prewarm.py
tests/test_user_profile_agent.py
tests/test_utils.py
```

At Checkpoint A the parent reruns the legacy-reference inventory grep. Any new
match is added to this manifest by plan update before its file is edited.

### Delete

| Path | Deletion gate |
|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v2/state_store.py` | DB-backed V2 state tests pass and no import remains. |
| `src/kazusa_ai_chatbot/cognition_chain_core/` | Checkpoint G live evidence passes and caller grep is zero. |
| `tests/test_cognition_chain_core_action_selection.py` | V2 action/collapse replacement tests pass. |
| `tests/test_cognition_chain_core_contracts.py` | V2 public contract tests pass. |
| `tests/test_cognition_live_llm_affinity_willingness.py` | V2 relationship/personality live cases pass. |
| `tests/test_cognition_live_llm_boundary_affinity.py` | V2 boundary/personality live cases pass. |

The rewrite-in-place inventory above and the four deletion rows are the complete
Stage 2 disposition of tests that currently contain affinity or V1 terminology.
Checkpoint A freezes that disposition before implementation begins.

### Keep unchanged in Stage 2

- Adapter delivery implementations and platform message normalization.
- Mongo URI configuration and connection creation outside the activated test
  guard.
- Memory collections and memory-evolution contracts.
- Existing scheduler timing, worker trigger, and run repository boundaries.
- Existing LLM provider/interface implementation and provider-owned parallel
  request capacity.
- Production migration, deployment, restart, and data conversion tooling.

### Exact Stage 3 residual allowlist

Affinity/V1/prose-affect text may remain after Stage 2 only in these auxiliary
paths, which Stage 3 must classify as change or retire:

```text
src/control_console/contracts.py
src/control_console/kazusa_client.py
src/control_console/repository.py
src/control_console/app.py
src/control_console/redaction.py
src/control_console/stream.py
src/control_console/static/index.html
src/control_console/static/console.js
src/control_console/static/console.css
src/control_console/README.md
src/kazusa_ai_chatbot/db/script_operations.py
src/scripts/_lane_cleanup.py
src/scripts/export_user_profile.py
src/scripts/export_character_state.py
src/scripts/export_user_image.py
src/scripts/identify_user_image.py
src/scripts/user_state_snapshot.py
src/scripts/character_state_snapshot.py
src/scripts/audit_user_profiles_lane.py
src/scripts/audit_character_state_lane.py
src/scripts/README.md
tests/test_control_console_repository.py
tests/test_control_console_review_edges.py
tests/control_console_e2e/test_page_navigation_e2e.py
tests/test_script_db_boundary.py
tests/test_user_state_snapshot.py
tests/test_character_state_snapshot.py
```

Stage 3 also owns the already-registered auxiliary documentation adoption
surface below. These files may retain explanatory V1 layer names, affinity,
prose-affect, or pre-V2 operator instructions during Stage 2, but they remain
outside runtime authority and must not be used as a Stage 2 implementation
contract:

```text
README.md
docs/HOWTO.md
src/kazusa_ai_chatbot/action_spec/README.md
src/kazusa_ai_chatbot/accepted_task/README.md
src/kazusa_ai_chatbot/background_work/README.md
src/kazusa_ai_chatbot/brain_service/README.md
src/kazusa_ai_chatbot/coding_agent/README.md
src/kazusa_ai_chatbot/coding_agent/coding_run/README.md
src/kazusa_ai_chatbot/complex_task_resolver/README.md
src/kazusa_ai_chatbot/event_logging/README.md
src/kazusa_ai_chatbot/internal_monologue_residue/README.md
src/kazusa_ai_chatbot/llm_interface/README.md
src/kazusa_ai_chatbot/past_dialog_cognition/README.md
src/kazusa_ai_chatbot/rag/README.md
```

No other `src` or `tests` match is permitted at Stage 2 completion.

## 3. Database Isolation Contract

`tests/conftest.py` performs this order:

1. call `load_dotenv(override=False)` using the existing behavior;
2. set `MONGODB_DB_NAME=_test_kazusa_live_llm`;
3. set `KAZUSA_TEST_DB_GUARD=1`;
4. import project modules only after steps 1-3;
5. close the cached client at session completion.

`db._client.get_db()` checks the guard immediately before creating a client.
When the guard is active and the imported database name differs from the exact
allowlisted name, it raises `DatabaseTestGuardError`. The guard has no alternate
name and no prefix matching. Normal runtime behavior is unchanged when the
guard variable is absent.

`src/kazusa_ai_chatbot/db/_client.py` remains the only source or test path that
constructs `AsyncIOMotorClient`/`MongoClient`. Checkpoints B and H run
`rg -n "AsyncIOMotorClient|MongoClient\(" src tests --glob "*.py"` and require
all matches to be import/type/construction lines in that one file.

Database-backed Stage 2 tests:

- carry `live_db`; real-LLM cases also carry `live_llm`;
- run without pytest-xdist; the helper fails if `PYTEST_XDIST_WORKER` exists;
- use owner id `s2-<sanitized-test-node>-<uuid-hex>` for every mutable user,
  channel, event, task, memory, schedule, and action row;
- query mutable rows by that owner id or an exact shared-seed id;
- use an idempotent seed helper that creates missing fixed seed documents and
  validates existing documents without overwriting them;
- snapshot the exact singleton character document before a writing test and
  restore it in `finally`;
- perform no collection drop, database drop, broad delete, or routine reseed.

The idempotent shared seed contains exactly:

| Seed id | Content |
|---|---|
| `character_state`, singleton `_id="global"` | Contract-spec character production default with native V2 empty causal lists |
| `user_profiles`, user `seed-s2-acquaintance` | Exact acquaintance V2 state and bounded neutral profile data |
| `user_profiles`, user `seed-s2-established` | Familiarity `70`, positive regard `60`, trust `60`, attachment `70`, desired closeness `70`, perceived closeness `60`, care `70`, boundary safety `50`, exclusivity `60`, unresolved injury `0`, salience `60`; other lists empty |
| `conversation_history`, rows `seed-s2-jealousy`, `seed-s2-guilt`, `seed-s2-relief` | Bounded synthetic episode text and source metadata used by the three full-pipeline cases |
| `conversation_history`, rows `seed-s2-emotion-<emotion_id>-<phase>` | One bounded synthetic begin/sustain/fade/negative sequence for each registry emotion |
| `memory`, rows `seed-s2-nostalgia`, `seed-s2-promoted-reflection` | Synthetic promoted memory/reflection evidence with no production-derived content |
| `accepted_tasks`, row `seed-s2-task-result` | Typed synthetic accepted-task result for direct-fact and recurrence cases |
| fixture-only row `seed-s2-action-result` | Typed synthetic ActionSpec result; the test still loads its owner/state/evidence from Mongo before use |

The seed helper validates exact schema/version/content hashes and creates a
missing row with `$setOnInsert`; an existing mismatch fails the test setup.
Mutable tests copy the needed seed state into their unique owner document and
leave shared seed rows unchanged. Character-scoped tests use the singleton
snapshot/restore rule instead of cloning the singleton.

## 4. Fixed Real-LLM Manifest

Each command runs alone. The parent inspects and signs its raw trace before
running the next command.

### Scoped appraisal and full-pipeline cases

```powershell
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_jealousy_scoped_appraisal_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_guilt_scoped_appraisal_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_relief_scoped_appraisal_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_ambiguous_user_meaning_uses_semantic_lane_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_jealousy_full_pipeline_live_llm_db -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_guilt_full_pipeline_live_llm_db -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_relief_full_pipeline_live_llm_db -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_group_user_scope_full_pipeline_live_llm_db -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_media_observation_full_pipeline_live_llm_db -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_resolver_recurrence_full_pipeline_live_llm_db -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_self_cognition_character_scope_live_llm_db -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_accepted_task_result_full_pipeline_live_llm_db -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_same_ambiguous_events_report_cross_model_variance_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_benchmark.py::test_v1_v2_latency_profile_live_llm_db -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_e2e_live_llm.py::test_live_graph_relationship_state_negative_delta_for_hostile_input -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_e2e_live_llm.py::test_live_graph_relationship_state_no_change_for_neutral_transactional_input -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_e2e_live_llm.py::test_live_graph_relationship_state_positive_delta_for_warm_appreciation -q -s -m "live_llm and live_db"
```

The cross-model case injects `COGNITION_LLM` as the primary binding and
`BOUNDARY_CORE_LLM` as the comparison binding and asserts distinct resolved
model identifiers. Equal identifiers block the case until the operator supplies
two distinct existing route configurations; no code/config fallback is added.

### Twenty-one natural-language lifecycle cases

`tests/test_cognition_core_v2_live_llm.py` contains one plain test function per
emotion. The parent runs these commands separately and in this order:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_joy_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_fear_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_anger_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_sadness_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_disgust_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_surprise_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_love_attachment_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_compassion_empathy_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_gratitude_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_jealousy_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_envy_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_pride_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_shame_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_guilt_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_embarrassment_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_curiosity_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_awe_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_nostalgia_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_loneliness_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_relief_lifecycle_live_llm -q -s -m "live_llm and live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_live_llm.py::test_ennui_existential_angst_lifecycle_live_llm -q -s -m "live_llm and live_db"
```

Each trace records the semantic question, semantic projection, raw/parsed
output, accepted/rejected propositions and deltas, deterministic state before
and after, derived activation, branch bids, final intention, model id, call
count, and latency. Passing pytest structure is insufficient; the parent adds
a human-readable groundedness/semantic judgment before sign-off. Every case
also records `_test_kazusa_live_llm`, its unique owner id, and singleton restore
result when character state is written.

## 5. Execution Checkpoints

The same production-code owner receives a read-only alignment packet at
Checkpoint A and sequential bounded implementation packets through Checkpoint
H. The original native subagent is unavailable; the user explicitly authorized
the current Codex parent agent to take over that production-owner role. The
current Codex agent is the sole production implementation owner for B/C
remediation, later packets, and any Checkpoint I remediation, and remains
available until Checkpoint I sign-off. The parent owns every test, command,
evidence artifact, architect calibration, and sign-off. The takeover owner
owns implementation self-calibration and in-scope production remediation.

This is an execution-model deviation only. Frozen contracts, change surface,
test ownership, checkpoint order, evidence requirements, and the rule that B/C
must be aligned before D remain unchanged.

Every checkpoint uses this pair:

```text
test_artifacts/cognition_core_v2/stage_2/calibration/<checkpoint_id>_implementation.md
test_artifacts/cognition_core_v2/stage_2/calibration/<checkpoint_id>_architect.md
```

Both records use `Contract Hashes`, `Expected Outcome Rows`, `Actual State`,
`Changed Files`, `Commands and Results`, `Forbidden-State Checks`, `Remaining
Distance`, `Deviation and RCA`, `Status`, and `Sign-off`. The implementation
agent rereads all three contract documents before each packet and records their
SHA-256 hashes. Its record states what its own packet changed and where it
diverged. The architect record checks
that claim against parent-run tests, static checks, evidence, and the frozen
outcomes. `Status` is exactly `aligned`, `drift_detected`, or `blocked`; both
records must be `aligned` before handoff.

### Checkpoint A — contract and baseline lock

- Files: parent plan, contract spec, execution manifest, Stage 1 RCA and evidence.
- Agent action: parent starts the one production implementation subagent with a
  read-only packet, records its canonical id, and requires its Gate A
  implementation self-calibration before any production edit.
- Actions: record current V1 scripted synthetic benchmark, known 14-pass/6-fail adjacent
  V1 result, and known fixture-project collection issue; create the checkpoint
  ledger and exact affected-test inventory.
- Verify:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath development_plans\active\short_term\cognition_core_v2_stage_2_integration_plan.md
Get-FileHash -Algorithm SHA256 -LiteralPath development_plans\active\short_term\cognition_core_v2_stage_2_contract_spec.md
Get-FileHash -Algorithm SHA256 -LiteralPath development_plans\active\short_term\cognition_core_v2_stage_2_execution_manifest.md
venv\Scripts\python -m pytest tests\test_cognition_core_v2_benchmark.py -q
venv\Scripts\python -m pytest --collect-only --ignore=tests\fixtures\coding_agent_full_workflow -m "not live_db and not live_llm and not live_internet" -q
```

- Expected: benchmark completes or records a named baseline blocker; collection
  has no error outside the ignored fixture project.
- Calibration: map the frozen contracts to `S2-O1` through `S2-O10`; report
  future evidence rather than claiming unimplemented outcomes.
- Evidence: `baseline/`, `contracts/gate_a.md`,
  `calibration/gate_a_implementation.md`, `calibration/gate_a_architect.md`.
- Handoff: start Checkpoint B only after architect sign-off.
- Sign-off: `<architect/date>`.

### Checkpoint B — isolated DB harness and schema

- Production packet: `state_models.py`, DB schema/facade/bootstrap/client files.
- Parent test packet: Mongo helper/seed/isolation tests plus V2 contract/state tests.
- Expected before implementation: missing V2 models/facades/guard tests fail.
- Verify:

```powershell
venv\Scripts\python -m pytest tests\test_live_llm_mongo_isolation.py tests\test_cognition_core_v2_contracts.py tests\test_cognition_core_v2_state.py -q
```

- Expected after: exact-name rejection, native seed validation, acquaintance and
  character defaults, the complete caller-to-scope matrix, one-scope state,
  caps, pruning, restart reload, owner isolation, and singleton restore pass.
- Calibration: `S2-O2`, `S2-O4`, `S2-O8`.
- Evidence: `persistence/gate_b.md`, `calibration/gate_b_implementation.md`,
  `calibration/gate_b_architect.md`.
- Handoff: Checkpoint C.
- Status: `aligned`.
- Sign-off: `Codex parent architect / 2026-07-15`; takeover implementation and
  architect calibration records are aligned.

### Checkpoint C — deterministic reducers and twenty-one lifecycles

- Production packet: state reducers, transition guards, definitions,
  derivation, elapsed decay, and sleep recovery.
- Parent test packet: lifecycle, failure, and affect-settling tests.
- Expected before implementation: new formula/FSM/one-scope/failure cases fail.
- Verify:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_core_v2_state.py tests\test_cognition_core_v2_emotion_lifecycle.py tests\test_cognition_core_v2_failures.py tests\test_reflection_affect_settling.py -q
```

- Expected after: exact goal FSM; event outcomes; every emotion begin/sustain/
  fade/negative guard; threat/event/knowledge-gap FSMs; mixed emotion; user
  decay; the exact producer/fact direct-transition table; deterministic
  idempotent sleep recovery; zero settling LLM calls.
- Calibration: `S2-O2`, `S2-O3`, `S2-O4`.
- Evidence: `lifecycle/gate_c.md`, `calibration/gate_c_implementation.md`,
  `calibration/gate_c_architect.md`.
- Handoff: Checkpoint D. Branch/DAG work remains unopened until sign-off.
- Status: `aligned`.
- Sign-off: `Codex parent architect / 2026-07-15`; takeover implementation and
  architect calibration records are aligned.

### Checkpoint D — two-phase appraisal and projection

- Production packet: semantic source planner, appraisal, central projection,
  immediate/final facade flow, and diagnostics.
- Parent test packet: projection, alignment-gate, and failure tests.
- Expected before implementation: preliminary-state/branch and sentinel tests fail.
- Verify:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_core_v2_projection.py tests\test_cognition_core_v2_alignment_gates.py tests\test_cognition_core_v2_failures.py -q
```

- Expected after: deterministic work proceeds before semantic completion;
  six-family source selection and unique target ownership; zero-axis candidates
  persist only through guarded create; all numeric state is interpreted;
  free-form descriptors are accepted; direct user-language meaning cannot enter
  the direct-fact lane.
- Calibration: `S2-O3`, `S2-O5` and every Stage 1 appraisal/projection RCA row.
- Evidence: `contracts/gate_d.md`, `calibration/gate_d_implementation.md`,
  `calibration/gate_d_architect.md`.
- Handoff: Checkpoint E.
- Sign-off: `<architect/date>`.

### Checkpoint E — goal DAG, parallel execution, collapse, and public facades

- Production packet: branch activation, dependency graph, executor, goal
  cognition, workspace, action selection, output projection, facade, and L3.
- Parent test packet: dependency, contract, alignment, failure, and integration tests.
- Expected before implementation: fourteen-branch, no-semaphore, complete-bid,
  provenance, public API, and V2 L3 tests fail.
- Verify:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_core_v2_dependencies.py tests\test_cognition_core_v2_contracts.py tests\test_cognition_core_v2_alignment_gates.py tests\test_cognition_core_v2_failures.py tests\test_cognition_core_v2_integration.py -q
```

- Expected after: dependency-ready overlap, unique slots, isolated failures,
  exact question-to-new-goal dependencies including the jealousy schedule,
  complete bids, provenance-safe collapse, route validation, two public V2
  APIs, and no V1 import inside V2.
- Calibration: `S2-O5`, `S2-O6`, `S2-O7` and every branch/collapse RCA row.
- Evidence: `parallelism/gate_e.md`, `calibration/gate_e_implementation.md`,
  `calibration/gate_e_architect.md`.
- Handoff: Checkpoint F.
- Sign-off: `<architect/date>`.

### Checkpoint F — single caller switch and runtime integration

- Production packet: every exact upstream, RAG/context, action, brain,
  consolidation, reflection, and event-logging modify path in Section 2.
- Parent test packet: connector, resolver, action, dialog, consolidation,
  self-cognition, event-logging tests, and the complete rewrite-in-place
  legacy-reference inventory in Section 2.
- Expected before implementation: integration tests use V1 or fail on missing V2 fields.
- Verify:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_chain_connector_mapping.py tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_loop.py tests\test_cognition_core_v2_integration.py tests\test_action_spec_models.py tests\test_action_spec_evaluator.py tests\test_action_spec_results.py tests\test_l2d_l3_surface_handoff.py tests\test_dialog_agent.py tests\test_service_background_consolidation.py tests\test_self_cognition_integration.py tests\test_event_logging_interface.py tests\test_service_event_logging.py -q
```

- Expected after: all runtime callers use V2; state commits before downstream
  surface/action work; recurrence carries working state; dialog/consolidation
  cannot author causal state; event diagnostics remain bounded.
- Calibration: `S2-O1`, `S2-O2`, `S2-O5`, `S2-O7`.
- Evidence: `integration/gate_f.md`, `calibration/gate_f_implementation.md`,
  `calibration/gate_f_architect.md`.
- Handoff: Checkpoint G; V1 stays unreferenced but present until live sign-off.
- Sign-off: `<architect/date>`.

### Checkpoint G — isolated real-LLM, DB, and performance evidence

- Parent action: run every command in Section 4 one at a time, inspect each
  trace, then run the benchmark and test-DB service smoke.
- Verify:

```powershell
venv\Scripts\python -m pytest tests\test_live_llm_mongo_isolation.py -q -m "live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_benchmark.py -q
venv\Scripts\python -m pytest tests\test_cognition_core_v2_integration.py::test_test_database_private_chat_smoke -q -s -m "live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_integration.py::test_test_database_resolver_recurrence_smoke -q -s -m "live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_integration.py::test_test_database_self_cognition_smoke -q -s -m "live_db"
venv\Scripts\python -m pytest tests\test_cognition_core_v2_integration.py::test_test_database_accepted_task_result_smoke -q -s -m "live_db"
```

- Expected: exact test DB recorded; owner scope and singleton restore pass; all
  live cases have inspected reviews; exact call/latency/overlap/context/state
  I/O/failure data exists. No agent-selected value/cost threshold is applied.
- Calibration: `S2-O3` through `S2-O9`.
- Evidence: `live_llm/gate_g.md`, `performance/gate_g.md`,
  `calibration/gate_g_implementation.md`, `calibration/gate_g_architect.md`.
- Handoff: Checkpoint H only after architect sign-off.
- Sign-off: `Codex parent architect / 2026-07-15`; exact G live/model/DB/performance packet and both calibration records are aligned. H and I remain open.

### Checkpoint H — legacy deletion and full release-candidate regression

- Production packet: delete V1, Stage 1 store, and remaining production
  affinity/prose-affect code.
- Parent test packet: delete the four exact obsolete test files, remove the
  completed Checkpoint G V1 benchmark node, and remove any legacy reference
  left outside the already-rewritten Checkpoint F inventory.
- Static verify:

```powershell
rg -n "from kazusa_ai_chatbot\.cognition_chain_core|import kazusa_ai_chatbot\.cognition_chain_core|CognitionChain(Input|Output|Services)V1|run_cognition_chain" src tests
rg -n "active_incidents|IncidentState" src tests
rg -n "affinity|Affinity|last_relationship_insight|global_vibe|reflection_summary" src tests
```

- Expected: first two searches return zero matches with `rg` exit code `1`.
  The third search returns matches only at the exact Stage 3 residual paths in
  Section 2; every other match blocks sign-off.
- Regression verify:

```powershell
venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot tests
venv\Scripts\python -m pytest --collect-only -q
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest -q --ignore=tests\control_console_e2e\test_page_navigation_e2e.py --ignore=tests\test_control_console_repository.py --ignore=tests\test_control_console_review_edges.py --ignore=tests\test_script_db_boundary.py --ignore=tests\test_user_state_snapshot.py --ignore=tests\test_character_state_snapshot.py
git diff --check
```

- Expected: compile and collection pass; pytest uses `pytest.ini` to exclude
  live tests and the fixture project. The diagnostic full run may fail only at
  the six exact Stage 3 test paths excluded by the following command, and each
  such failure is copied into the Stage 3 handoff. The Stage 2-owned command
  must pass with zero failures. Every other runtime/core failure blocks sign-off.
- Calibration: `S2-O1`, `S2-O2`, `S2-O8`, `S2-O9`, `S2-O10`.
- Evidence: `integration/gate_h.md`, `calibration/gate_h_implementation.md`,
  `calibration/gate_h_architect.md`.
- Handoff: keep the Codex takeover production owner available for Checkpoint I
  remediation; Checkpoint I remains open.
- Sign-off: `Codex takeover production owner / Codex parent architect / 2026-07-15`.

### Checkpoint I — independent review, report, and closure

- Parent starts exactly one independent review subagent after Checkpoint H.
- Reviewer receives the approved parent plan, both companions, Stage 1 RCA,
  full diff, commands/results, all calibration records, live reviews, DB guard
  evidence, and Stage 3 residual list.
- Reviewer checks architecture, state ownership, all formulas/FSMs, two-phase
  readiness, prompts, failures, DB isolation, change surface, deletions,
  tests, and overdesign. Reviewer implements no fix.
- Parent sends each in-scope production finding to the same production subagent
  as a bounded remediation packet, owns any test-only remediation, and reruns
  the affected checkpoint. The implementation subagent updates Gate I
  self-calibration after remediation. Contract/scope findings reopen the plan.
- Parent sends the remediation diff and rerun evidence back to the same
  independent reviewer through a follow-up task. Reviewer closure and architect
  Gate I calibration are both required before sign-off.
- Verify: repeat Checkpoint H static/regression commands after remediation.
- Calibration: compare actual system with all `S2-O1` through `S2-O10`.
- Evidence: `reviews/gate_i.md`, `calibration/gate_i_implementation.md`,
  `calibration/gate_i_architect.md`, and final `reviews/value_cost_quality.md`.
- Candidate identity and durable evidence summary:
  - base HEAD `3a1247320cb016d4a9d5d24a1300fd46cfdbe8af`;
  - source/test diff git hash `4ed2538b6a0b46a4820cb6a01901226d87bfb9ec`;
  - normal and `--noconftest` collection both report `3112/3750`
    selected with `638` marker-deselected;
  - exact full regression reports `3110 passed, 2 skipped, 638
    deselected`; the six-path Stage 3 diagnostic regression reports `3071
    passed, 2 skipped, 638 deselected`;
  - the final affected deterministic remediation packets report Checkpoint B
    `33 passed, 3 deselected`, Checkpoint D `40 passed`, Checkpoint E `53
    passed, 4 deselected`, and the bounded surface/dialog packet `31 passed`;
    the earlier unaffected Checkpoint C and F packets remain green;
  - guarded Mongo isolation reports `3 passed, 3 deselected`; the four private,
    resolver, self-cognition, and accepted-task DB smokes each report `1
    passed`;
  - refreshed scoped, full-pipeline, surface, lifecycle, relationship, and
    cross-model commands in Section 4 pass one at a time. The comparison model
    records one expected fail-closed appraisal for an unsupported
    `goal_supersession` lacking a distinct object handle; the primary model has
    six successful appraisals;
  - the corrected ten-sample V2 performance summaries are: ordinary minimal
    mean/p95 `38249.8/42527.5 ms`, single-emotion goal
    `47682.7/50608.0 ms`, mixed-goal conflict `46542.1/52155.7 ms`, and
    maximum bounded parallelism `50709.1/52733.0 ms`. All `40` samples have
    zero run, parse, and validation failures; each records four protected
    orchestration/state events. Maximum-parallelism mean LLM overlap ratio is
    `0.7796`;
  - exact V1 and incident scans return exit `1`; the residual vocabulary scan
    returns only paths in the Stage 3 allowlist; compile and `git diff --check`
    pass.
  - final affected trace linkage and qualitative review:
    - surface trace
      `test_artifacts/llm_traces/cognition_core_v2_stage_2__v2_text_surface_stage_contracts__20260715T174036596112Z.json`,
      SHA-256
      `B4A6EAD68DEC2472C64C6B2BAB495DEAE2267F9C75365E5A3C635686FB64A23A`;
      all four adjacent stage-local prompt/LLM/handler blocks returned exact
      bounded fields, kept visible boundaries separate from addressee handling,
      and authored no final dialogue;
    - ambiguous-appraisal trace
      `test_artifacts/cognition_core_v2/raw/ambiguous-user-meaning_1784136529692538800.json`,
      SHA-256
      `083DBED750A728698AB93614CF7545D5CB5CC8D9C588B518F10F10DABC09CE50`;
      all six appraisals succeeded, recorded zero failures, and every candidate
      proposition/delta cited its originating `e1` evidence;
    - full jealousy trace
      `test_artifacts/cognition_core_v2/raw/jealousy-full-pipeline_1784136643438635300.json`,
      SHA-256
      `123E7CC41F7183BDEA04BB0E93E623E6DB6E632F28C5A9364E80B175422C8BDC`;
      all six appraisals succeeded with zero failures and zero candidate/evidence
      binding mismatches;
    - performance artifact SHA-256 values are
      `F1CE6F96170BEF9B7E691F0F58D2F90DB28022686F8C19F39509C2232ACDA879`
      ordinary,
      `4569EF3C28CF74EAB33291BA59BD37D9C3FB368B43B495FC89CB4646AD515510`
      single-emotion,
      `48A6EC96A7D986F9E14146914E07C2CD646BA26AF5E4A26D09F400742B379E80`
      mixed-goal, and
      `FAFE384B30E4E045E1F11771CF855F313395545F807167CE941AAFE220A12B03`
      maximum parallelism. These forty samples precede the final bounded
      surface-layout, evidence-binding, and seed-hash remediation; those fixes
      add no LLM call or concurrency cap, and the affected current-candidate
      live cases above were refreshed individually.
- Handoff: present exact measurements and limitations to the user. The user's
  Stage 3/cutover decision does not change whether completed Stage 2 work meets
  its technical acceptance criteria.
- Sign-off: `/root/checkpoint_i_independent_review / Codex parent architect /
  2026-07-16`; independent verdict `APPROVED`, no remaining closure blockers;
  the user-authorized production-owner takeover role is closed.

## 6. Production-State Evidence Boundary

Stage 2 evidence proves:

- no command used a production database name;
- the test guard rejected alternate names;
- execution logs contain only `_test_kazusa_live_llm`;
- the diff contains no migration/deployment/restart operation;
- the release candidate is not deployed by this plan.

Statements about the external deployed service or production database require a
separate operator/user confirmation. Stage 2 agents do not inspect those systems.

## Lifecycle Closure

This execution manifest closed with the Stage 2 parent on 2026-07-18. Its
production-state boundary remains binding: Stage 3 uses an isolated fresh test
database, while Stage 4 owns production migration and cutover.
