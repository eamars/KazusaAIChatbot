# group topic continuity authority fix

## Summary

- Goal: Fix the S6-S8 group multi-user current-event authority collapse so the weak local model stays anchored to the current public scene and the addressed user's continuity while preserving useful character judgment, private per-user continuity, and bounded latency.
- Status: completed
- Archive disposition: completed on 2026-08-13 after independent final sign-off
- Scope boundary: `conversation_progress`, the cognition input projection, V2 goal cognition, typed internal-monologue residue lifecycle, continuity diagnostics, post-turn reconciliation, their ICDs, and the named deterministic/live verification artifacts. S1/S2 private/public surface failures and S4 reality-state failures receive characterization and non-expansion gates here; they are not claimed as fixed by this plan.
- Change direction: Shape public context deterministically first, make evidence-lane authority explicit in goal cognition, give the residue recorder an LLM-owned typed disposition, cut the residue contract over in one canonical v2 change, and diagnose progress persistence with trace-keyed reconciliation before changing its write algorithm.
- Acceptance state: Verified and independently signed off under the final owner amendments; Gates A-D completed and Gate E owner-waived with no latency-regression claim.

## Evidence And Confirmed Decisions

The plan is based on:

- [QQ group RCA](../../../test_artifacts/llm_debug/qq_group_54369546_2026-08-12_root_cause_analysis.md)
- [topic-continuity review](../../../test_artifacts/llm_debug/qq_group_54369546_2026-08-12_topic_continuity_review.md)
- [failure-mode addendum](../../../test_artifacts/llm_debug/qq_group_54369546_2026-08-12_failure_mode_addendum.md)
- Protected trace and diagnostic exports under `test_artifacts/diagnostics/` and `test_artifacts/llm_debug/`.
- The independent `gpt-5.6-sol` maximum-reasoning review completed for this RCA.

Confirmed decisions:

1. The generalized failure is authority collapse across context lanes. `cognition_core_v2.goal_cognition` owns the primary semantic selection for S7 and S8; public-scene shaping and stale continuity are deterministic contributors.
2. Promoted reflection is an amplifier. S7 cited a reward strategy from promoted guidance, and the cognition projection currently replaces each row's source timestamp with the current turn timestamp.
3. S6 is a weaker visible-contamination proof. Its selected ordinary goal kept protection/rescue primary; reward appeared in private or suppressed branches. S6 remains a future-residue and ordering regression case.
4. The 50K per-call baseline remains fixed. `GOAL_COGNITION_PROMPT_CAP` remains 36,000 characters. No context increase is authorized by this plan.
5. No new foreground LLM call, arbitration agent, semantic retry, keyword topic suppressor, forced-silence gate, group-wide persisted packet, or compatibility vocabulary is introduced.
6. The independent `gpt-5.6-sol` review found the draft non-approvable until residue disposition, telemetry schema, migration/rollback, manifest ownership, captured failure coverage, anchor fitting, typed guidance authority, live all-lane replay, interruption reconciliation, and capacity gates were made executable. Those corrections are mandatory plan content below.
7. This plan deliberately fixes and accepts S6-S8 authority collapse only. S1/S2 and S4 remain named residual failure modes with characterization gates and require a separate approved scope before any broader “generalized failure modes fixed” claim.

## Scope And Change Direction

The target flow is:

```text
current trigger
  -> deterministic public-scene anchor selection
  -> typed evidence-lane projection with temporal provenance
  -> one goal-cognition primary objective
  -> existing facade/action/surface/dialog path
```

The semantic question stays small: **what is the character's one primary response objective for this current event, and which subordinate actions serve that same objective?**

The resulting authority order is fixed:

1. The current episode and typed response operation define the current event, addressee, actor, and target direction.
2. The transient public group scene defines observable shared facts and visible participant order.
3. Current-user conversation progress may continue or constrain the same concrete matter.
4. Private residue may influence motive, tone, or hesitation, but cannot establish an external fact or create a new topic.
5. Promoted lore may provide character/world context. Promoted self-guidance is conditional guidance for tactics only and cannot satisfy current-event evidence requirements.
6. Every surface-bearing goal field remains causally attached to the one primary objective. Ordered sub-actions are allowed only when they serve that objective.

The selector uses the neutral term **participant continuity anchors**, not
“causal pair”. The newest current-user turn and the newest explicitly addressed
assistant turn are protected independently. A test may call them a causal pair
only when the assistant row is chronologically after the selected user row and
has a matching `reply_context` or an explicit single-user address. Multi-address
and out-of-order rows remain ambient/protected context without an asserted reply
relationship.

## Mandatory Skills

- `development-plan` for lifecycle, ownership, exact traceability, and execution gates.
- `local-llm-architecture` for weak-model context shaping, prompt vocabulary, latency, and call-count limits.
- `debug-llm` for captured-trace replay, human-readable quality review, and real-output evidence.
- `test-style-and-execution` for deterministic, patched, exported-replay, and one-at-a-time live tests.
- `py-style` for every Python source change.
- `cjk-safety` for Python prompt changes containing Chinese/Japanese text.
- `character-test` for live character behavior checks through the debug or service path.

## Mandatory Rules

- Keep adapters, intake, settlement, relevance gating, RAG retrieval, workspace collapse, action planning, L3 wording, dialog verification, and delivery outside the first behavior slice.
- Keep `ConversationProgressScope` keyed by platform, channel, and current `global_user_id`. No participant's private progress or residue may enter another participant's cognition input.
- Keep public group context transient. Do not add a persisted group continuity packet or group-wide private residue lane.
- Keep deterministic code responsible for scope, identity, caps, temporal labels, persistence, cache invalidation, and event-log sanitization. Keep semantic objective selection in goal cognition.
- Preserve the existing goal output schema, route selection, attempt ledger, retry caps, and call count.
- Use semantic descriptors for model-facing age/noise information; do not expose database telemetry, raw storage IDs, or diagnostic counters as model instructions.
- Run regular deterministic and patched tests in batches. Run every real-LLM test individually, inspect its durable artifact, and record a human-readable quality judgment.
- Keep production behavior unchanged until the plan is approved and the user explicitly authorizes implementation.

## Must Do

### Stage 0: bounded continuity diagnostics before persistence redesign

Add a dedicated `continuity_boundary` event family rather than overloading the
existing `database_operation` payload. The public recorder is keyword-only and
the ICD, typed model, sanitizer, exports, and snapshot compatibility are
updated together. The exact bounded payload is:

- `boundary`: `progress_load`, `progress_record`, `residue_load`,
  `residue_record`, or `reflection_projection`;
- `status`: `started`, `succeeded`, `skipped`, `contract_failed`,
  `provider_failed`, `persistence_failed`, `guarded_write_lost`,
  `cache_not_published`, `unknown`, or `reconciled`;
- `scope_kind`: `user_thread`, `group_scene`, `private`, or `targetless`;
- bounded counts: `candidate_count`, `selected_count`, `packet_turn_count`,
  `protected_anchor_count`, and `rendered_chars`;
- bounded labels: `packet_age`, `source_age`, `recorder_disposition`,
  `write_disposition`, `cache_disposition`, and `barrier_disposition`;
- opaque `trace_ref`, `correlation_ref`, and `operation_ref` values after the
  existing short-reference sanitization. References are links, never content.

Each field has an explicit enum, integer bound, or maximum length in the ICD.
Unknown or unavailable state is represented as `unknown`; omitted fields are
not silently interpreted as success. The event contains no message bodies,
dialog, prompt text, private residue text, raw model output, database filters,
document bodies, user identifiers, or credentials. It is best-effort
observability: event-write success never proves a database write, and an event
failure never changes the response path.

Instrument progress load/record, residue load/record, reflection projection,
and the post-turn wrapper. The existing packet/source references remain the
source of truth for reconciliation. A progress operation is keyed by its
`llm_trace_id`; after interruption, the reconciler performs a deterministic
read for that trace reference and emits `reconciled` with
`reconciled_written` or `reconciled_absent`. An unresolved state remains
`unknown` and blocks delivery evidence.

Gate: exported or event-log evidence distinguishes skipped, contract/provider
failure, persistence failure, guarded-write loss, cache disposition, unknown,
reconciled, and successful write. The stage does not change the progress write
algorithm.

### Stage 1: deterministic public-scene shaping and reflection provenance

Implement a participant-aware group ambient selector before the existing six-turn cap:

- Always retain the current trigger.
- For a non-empty current group user, reserve at most two participant continuity anchors: the newest preceding user turn authored by that user, and the newest assistant turn explicitly addressed to that user. Do not label the two rows a causal pair unless the chronological/reply-link rule in the scope contract is satisfied.
- A broadcast assistant row without an explicit address is not a current-user anchor.
- A multi-address assistant row is not a single-user causal anchor unless the
  current user is the sole explicit addressee and the reply link is valid.
- Fill remaining ambient capacity with the newest public turns from all participants, preserving chronological rendering and the current `GROUP_SCENE_MAX_TURNS` and `GROUP_SCENE_MAX_RENDERED_CHARS` limits.
- Protect the reserved anchors during final character fitting. The fit order is
  (1) drop unprotected ambient turns, (2) preserve a non-empty trigger, (3)
  preserve a non-empty current-user anchor, and (4) preserve a non-empty
  explicitly addressed assistant anchor. Only then may bounded visible text be
  shortened. If the protected semantic minima cannot fit the hard cap, return
  a typed `protected_minimum_unfit` projection failure and use the existing
  fail-closed/degraded scene path; never silently truncate a protected text to
  empty or claim the anchor survived.
- For targetless group self-cognition, pass an explicit empty current-user identity and retain the existing newest-ambient behavior.
- For private channels, bypass public-group anchor selection entirely and keep
  the existing private per-user history path. Add targetless-group and
  private-channel controls to the same owner suite.
- Pass the typed current-user identity from `service.py` through `ConversationProgressRuntime.load`, `persona_supervisor2.py`, and `build_group_scene_context`.

Preserve promoted reflection source provenance:

- Use each promoted row's valid `updated_at` as its evidence time; omit a row with invalid provenance rather than stamping it with the current turn.
- Keep promoted lore as character/world context evidence.
- Add a required typed `authority` field to the canonical `CognitionEvidenceV2`
  contract and update every constructor/projection in `cognition_core_v2` in the
  same change. Use a closed enum that includes `current_event`,
  `public_scene`, `participant_continuity`, `private_motive_only`,
  `character_world_context`, `conditional_character_guidance`, and
  `contextual_fact_only`. The field is deterministic metadata, never an LLM
  decision and never inferred from free text.
- Keep promoted self-guidance in the existing bounded evidence carrier with
  `authority=conditional_character_guidance`. It may influence tactics after
  objective selection, cannot satisfy current-event evidence requirements, and
  cannot create a topic or establish a current fact.
- Preserve the existing private residue lane and do not send promoted guidance to appraisal, workspace, dialog, or L3.

Gate: deterministic fixtures prove that current-user causal anchors survive newer multi-user noise, public order remains chronological, broadcast output is not misattributed, and reflection timestamps/guidance lanes retain their authority boundaries.

### Stage 2: one-objective goal arbitration contract

Update the ordinary, required-selection, recurrence, and repair goal prompts with one concise positive decision procedure:

1. Establish one primary objective from the current episode, typed response operation, and observable public scene.
2. Continue a progress item only when it is the same concrete matter; otherwise leave it as supplemental context.
3. Use private residue as motive, tone, or hesitation only. Treat hypothetical residue as a hypothesis, not as an observed event.
4. Use conditional self-guidance only for tactics after the primary objective is fixed.
5. Keep `selection`/`intention`, `reason`, `desired_outcome`, `concrete_detail`, and `expected_consequences` causally attached to that objective.
6. Allow ordered sub-actions only when they serve that objective, such as protecting an injured participant before neutralizing the threat and returning.

Carry a bounded temporal provenance field with each projected conversation-progress evidence row. The field contains the source `occurred_at` and a deterministic age descriptor; it does not change goal output authority. The typed `authority` field from `CognitionEvidenceV2` is the only source of lane authority; prompt prose may explain it but may not replace it.

Gate: prompt-contract tests prove current-episode/public-scene authority, non-authoritative residue/guidance, temporal provenance, and one-objective wording for every goal mode. No additional model call is added.

### Stage 3: typed residue disposition, scoped barrier, and canonical cutover

Give the residue recorder an explicit LLM-owned disposition. The only accepted
recorder object is:

```json
{"disposition":"append|replace_scope|clear_scope","residue_text":"..."}
```

The semantic stage chooses the disposition; deterministic code validates,
normalizes only bounded text, applies scope and source invariants, and performs
the write. The rules are strict: `append` requires non-empty text and keeps a
related short-lived reason; `replace_scope` requires non-empty text and retires
older rows for the exact scope because the current episode replaces that
branch; `clear_scope` requires empty text and means that no prior residue in
that exact scope remains relevant. A missing/unknown disposition, conflicting
text/disposition, invalid scope, or invalid type is a contract error and uses
the recorder's existing bounded regeneration/fail-closed path. Deterministic
code never infers a topic pivot from empty text or keywords.

Persist only canonical v2 rows with required `schema_version`, `operation_id`,
and `disposition`. `operation_id` is a deterministic digest of the exact
character/platform/channel/scope and completed cognitive-episode ID; the
completed episode ID is required for a recordable operation; if it is missing,
the recorder returns a typed contract failure and performs no write. The
operation ID is unique. A duplicate operation with the same payload returns the
existing result without another row. A duplicate operation with a different
payload is a typed conflict and fails closed. Add a configured
`INTERNAL_MONOLOGUE_RESIDUE_RETENTION_HOURS` default of 48 hours, a bounded
`purge_at`, and a TTL index.

This is a big-bang schema and reader cutover. The canonical `InternalMonologueResidueV2Doc`
is the only storage type, writer input, and reader result. Rows missing the v2
contract are excluded from the database query and deterministic selector; no
legacy append inference, backfill routine, optional-field alias, or parallel
reader is retained. Deployment requires an operational preflight confirming
that the collection contains only canonical rows; cleanup of any pre-existing
non-canonical rows is a separately approved data operation, not runtime
compatibility code. For each exact scope, the reader sorts newest first, loads
at most the existing window size, and applies the newest `replace_scope` or
`clear_scope` row as a barrier: `replace_scope` is projected as the newest
residue; `clear_scope` is not projected and hides older rows. Later `append`
rows remain visible. A barrier older than the bounded window cannot affect the
visible window because at least the full window of newer rows already wins.

Rollback retains the same canonical schema and barrier-aware reader. It stops
the v2 writer only if the runtime can still consume canonical rows; it never
returns to a pre-v2 reader and never invents a legacy default to make a partial
rollback appear clean.

Gate: deterministic loader, recorder, projection, DB, schema, and integration
tests prove append/replace/clear semantics, same-user clearing, cross-user and
group-scope isolation, duplicate idempotency, conflicting-operation failure,
canonical-row rejection, TTL/index configuration, restart recovery, and
positive same-scope carryover after a replacement or clear.

### Stage 4: captured failure matrix, live validation, and independent sign-off

- Add a sanitized structured fixture for the captured S1, S2, S4, S6, S7, and S8 cases. It includes source trace references, redaction manifest, character identity/route identity, hashes of raw inputs, all-lane state (current episode, public scene, progress, residue, promoted guidance), and expected hard gates. Keep production trace exports as linked raw evidence, not embedded source strings.
- Replay S1/S2/S4 as characterization/non-expansion cases and S6/S7/S8 as fix cases through deterministic context projection and goal prompt payload building. The fixture must reproduce the complete competing-lane condition; a public-message-only synthetic replay is insufficient.
- Run the captured all-lane S1/S2/S4/S6/S7/S8 live cases one at a time with the local production model route. Produce a human-readable review under `test_artifacts/llm_debug/` containing run context, input, raw/parsed output, validation, quality judgment, before/after comparison, latency, call count, and raw-evidence paths.
- Run three independent repetitions of S8 and the highest-risk all-lane S7 case. Record stochastic variation; every repetition must pass privacy and hard-contract gates, and the quality rubric must pass the agreed threshold.
- Run real recorder cases for related continuation (`append`), scene pivot (`replace_scope`), no continuation (`clear_scope`), and private-boundary behavior. The recorder output and persisted disposition are reviewed as real model evidence.
- Run the named existing multi-user isolation and cross-scope continuity tests individually when they are real-LLM tests.
- Characterize S1/S2/S4 explicitly: this plan requires no new leakage, wrong
  addressee or semantic owner, contradiction, or severe measurable worsening
  attributable to the changed surfaces, but it does not claim to solve their
  baseline defect. Limited ungrounded detail is documented as residual quality
  evidence and is not an automatic failure. Any attempt to claim those failures
  fixed requires a new approved plan.
- Have an independent reviewer inspect the diff, exact mapped test collection, deterministic results, exported replay artifacts, live artifacts, prompt character counts, and residual risks.

Gate: all hard acceptance criteria below pass, each changed source path maps to a collected exact pytest node, every live artifact is inspected, and the independent reviewer signs off. A green pytest status alone is insufficient for live quality acceptance.

### Progress-writer follow-up boundary

Stage 0 records the actual frozen-packet/write disposition and performs only a
trace/operation-keyed read reconciliation after an ambiguous interruption. This
plan does not change guarded replacement, locking, sequencing, retry, or
persistence ownership. A progress-writer algorithm fix is a separate approved
plan amendment whose scope must be based on the captured disposition rather than
a guessed race.

## Deferred

- Guarded-write sequencing, per-scope leases, retry policy, or packet-write
  reconciliation changes. The bounded observational read reconciliation named
  in Stage 0 is in scope and does not mutate the packet algorithm.
- The actual S1/S2 private/public surface fix and S4 reality-state correction. This plan adds characterization and non-expansion gates only; those failure modes remain residual and require separate ownership, evidence, and approval before implementation.
- Any adapter, QQ wire, intake, settlement, relevance, RAG, workspace, action-planning, delivery, or database-key redesign.
- Any group-wide persisted continuity or group-wide private residue.
- Any increase to the 50K context baseline or the 36K goal-stage cap.
- A new arbitration model, new semantic retry, keyword topic suppression, regex silence gate, or forced-silence policy.
- Unrelated cleanup, compatibility aliases, fallback mappers, or parallel prompt vocabularies.

## Target State

### Public group-scene contract

`ConversationProgressRuntime.load` receives an explicit `preserve_group_public_anchors` boolean from the service based on the typed channel kind. The selector uses the existing assembled logical turns and `scope.global_user_id`; it does not read another user's progress or residue.

`build_group_scene_context` receives an explicit `current_global_user_id` string. Its transient turn shape gains deterministic-only `protected_for_current_user` metadata, which is removed from the rendered prompt. The public prompt remains bounded by the existing six-turn and 1,800-character caps and contains only visible names, roles, order, reply/address labels, and text. Protected semantic minima are non-empty; an unfit protected set returns a typed degraded result instead of an empty anchor.

The selector's protected rows are participant continuity anchors. A causal
relationship is asserted only when the selected assistant row is after the
selected user row and has a valid `reply_context` or sole explicit address.
Targetless groups preserve newest-ambient behavior; private channels bypass the
group selector.

### Goal evidence-lane contract

The goal input contains separate current episode, public group scene, participant continuity, private residue, conversation-progress evidence, promoted lore, and conditional self-guidance authority lanes within the existing evidence carrier. Every `CognitionEvidenceV2` row has required deterministic `authority` metadata from the closed contract enum. Evidence rows preserve source time. Only current episode and observable public scene establish current external facts. The goal output schema and downstream selected-bid authority remain unchanged.

### Residue lifecycle contract

Residue rows remain exact `user_thread` or `group_scene` scope records in `internal_monologue_residue_state`. Canonical rows carry `schema_version`, deterministic `operation_id`, and the LLM-owned `disposition` (`append`, `replace_scope`, or `clear_scope`). Rows without those fields are rejected by the canonical reader and writer. The visible residue projection contains only non-empty rows at or after the newest exact-scope barrier. `replace_scope` is visible as the new row; `clear_scope` is not visible. Rows have bounded TTL retention and transition-idempotent writes.

### Runtime constraint contract

The common path keeps the existing LLM call count, model routes, attempt caps, foreground latency shape, `GOAL_COGNITION_PROMPT_CAP=36000`, and effective 50,000-character/route context baseline. The selector, timestamp projection, barrier, cache, canonical persistence, and telemetry work is deterministic or background-owned. Baseline and candidate runs use the same route and fixture; candidate p95 foreground latency must be no more than 10% above baseline p95, and any absolute route timeout or call/retry increase is a hard failure.

## Execution Roles

### `runtime_implementation_owner`

- Responsibility: Implement the fixed behavior and contract changes within the named change surface.
- Owned surface: `src/kazusa_ai_chatbot/conversation_progress/**`, `src/kazusa_ai_chatbot/internal_monologue_residue/**`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`, `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`, `src/kazusa_ai_chatbot/service.py`, `src/kazusa_ai_chatbot/brain_service/post_turn.py`, their ICDs, mapped tests, fixture, and ownership manifest rows.
- Authority: May edit only the stated files and local tests/docs required by the contracts. May choose local helper decomposition and command order. May not change deferred areas, model routing, call count, context caps, persistence algorithm, or semantic ownership.
- Applicable skills: `py-style`, `cjk-safety`, `local-llm-architecture`, `test-style-and-execution`, `development-plan`.
- Capability floor: Strong Python/MongoDB contract reasoning, prompt-boundary analysis for a weaker local model, exact pytest mapping, and access to protected replay artifacts.
- Independence requirement: none for implementation; final quality review must be performed by a different role.
- Acceptance output: source diff, updated ICDs/manifest, deterministic test evidence, and a handoff listing every changed path and mapped node.
- Gate: approved plan, explicit implementation authorization, clean baseline captured, required skills loaded, and no unclassified worktree changes in the owned surface.

### `quality_verification_owner`

- Responsibility: Execute deterministic, patched, exported-replay, and real-LLM verification and author the human-readable quality artifact.
- Owned surface: Test commands, test artifacts, replay fixtures/results, and review notes; no production source edits.
- Authority: May mark checks pass/fail and request remediation. May not alter prompts, source, tests, fixtures, or acceptance criteria.
- Applicable skills: `debug-llm`, `test-style-and-execution`, `character-test`, `local-llm-architecture`.
- Capability floor: Able to inspect raw model input/output, distinguish harness/contract/quality results, run live cases one at a time, and assess group continuity and character judgment.
- Independence requirement: separate executor from the implementation owner.
- Acceptance output: durable raw evidence, human-readable review, exact pytest collection/run results, prompt-size report, and residual-risk list.
- Gate: implementation handoff identifies completed source-to-test rows and all required fixtures/artifacts are present.

### `independent_signoff_reviewer`

- Responsibility: Independently review scope alignment, ownership, regressions, evidence quality, and final acceptance.
- Owned surface: Full plan, implementation diff, mapped tests, live/replay artifacts, and verification report; no remediation edits.
- Authority: Pass or fail the plan gate and request a plan amendment or remediation.
- Applicable skills: `development-plan`, `debug-llm`, `local-llm-architecture`, `test-style-and-execution`.
- Capability floor: Independent architecture and test-traceability review with enough context to challenge authority, isolation, capacity, and character-quality claims.
- Independence requirement: Must not be the implementation owner or the person who remediates findings.
- Acceptance output: signed review with findings, exact evidence references, and explicit residual-risk disposition.
- Gate: all hard acceptance criteria and exact mapped pytest nodes have been executed; live artifacts have been inspected individually.

### `release_verification_owner`

- Responsibility: Perform live preflight, release/cutover checks, interruption
  recovery, and single-revision rollback verification.
- Owned surface: Runtime configuration evidence, service/model/MongoDB
  preflight output, deployment checklist, checkpoint ledger, and rollback
  evidence; no source or prompt edits.
- Authority: May block deployment for unavailable route, wrong character/model
  identity, missing database, skipped/deselected live nodes, unresolved
  telemetry, or rollback ambiguity. May not waive a quality or privacy gate.
- Acceptance output: exact preflight command/results, database name and test
  isolation evidence, route/context configuration, checkpoint ledger, and
  rollback drill result.
- Independence requirement: separate from the implementation owner; may be the
  quality owner only if the signoff reviewer remains independent.

Executor and model selection remain runtime-owned. No permanent agent, model, or dispatch choreography is fixed by this plan.

## Test Impact And Traceability

Every new node below is part of the implementation contract. New nodes must be created with the exact repository-relative ID shown here; existing nodes listed as supplemental remain required regression coverage.

| Source or governed artifact | Changed symbol/contract | Semantic owner | Exact deterministic pytest node IDs | Supplemental/replay/live nodes | Test mode | Regression prevented |
| --- | --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/conversation_progress/history.py` | participant-aware pre-cap logical-turn selector | conversation-progress history owner | `tests/test_conversation_progress_history_policy.py::test_group_scene_selection_preserves_current_user_anchors_before_recent_cap` | `tests/test_conversation_progress_v2_live_llm.py::test_live_group_stale_ambient_is_absent_from_stage_zero_prompt` | deterministic unit; live supplemental | Newer users cannot evict the current user's participant continuity anchors before scene projection. |
| `src/kazusa_ai_chatbot/conversation_progress/projection.py` | `build_group_scene_context`, `_fit_group_scene_turns`, protected-anchor fitting | public group-scene owner | `tests/test_conversation_progress_group_scene.py::test_group_scene_reserves_current_user_causal_anchors_under_newer_noise`; `tests/test_conversation_progress_group_scene.py::test_group_scene_anchor_requires_explicit_assistant_address`; `tests/test_conversation_progress_group_scene.py::test_group_scene_final_fit_keeps_protected_anchors_within_render_cap` | `tests/test_conversation_progress_v2_regression.py::test_captured_group_topic_continuity_fixture_preserves_public_continuity_anchors` | deterministic unit; exported replay | Prevents S8-style eviction while retaining chronological multi-user noise and hard caps. |
| `src/kazusa_ai_chatbot/conversation_progress/models.py` | deterministic-only protected anchor and diagnostics fields | conversation-progress contract owner | `tests/test_conversation_progress_v2_contract.py::test_group_scene_anchor_contract_is_prompt_safe` | `tests/test_conversation_progress_group_scene.py::test_group_scene_prompt_uses_semantic_labels_without_metadata` | deterministic contract | Prevents source IDs or protection metadata from entering the model-facing scene. |
| `src/kazusa_ai_chatbot/conversation_progress/runtime.py` | `ConversationProgressRuntime.load`, diagnostics, group-anchor mode | conversation-progress runtime owner | `tests/test_conversation_progress_runtime.py::test_load_group_scene_preserves_current_user_anchors_before_ambient_cap`; `tests/test_conversation_progress_runtime.py::test_progress_diagnostics_expose_packet_age_and_anchor_counts`; `tests/test_conversation_progress_runtime.py::test_progress_diagnostics_classify_guarded_write_outcomes`; `tests/test_conversation_progress_runtime.py::test_interrupted_record_does_not_publish_uncommitted_cache_state`; `tests/test_conversation_progress_runtime.py::test_lost_guarded_write_does_not_publish_packet_to_cache` | `tests/test_conversation_progress_v2_service.py::test_ordinary_response_path_adds_no_llm_call` | patched/deterministic | Preserves call count/cache safety while making frozen packets, partial outcomes, and anchor loss observable. |
| `src/kazusa_ai_chatbot/service.py` | `load_conversation_episode_state` passes explicit group-anchor mode | brain-service context loader | `tests/test_conversation_progress_v2_service.py::test_service_load_passes_group_anchor_mode_and_keeps_user_scope` | `tests/test_persona_supervisor2.py::test_persona_supervisor_projects_group_scene_alongside_user_history` | patched handoff | Prevents channel-mode ambiguity and preserves per-user progress scope. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | group-scene call carries typed current user | persona context owner | `tests/test_persona_supervisor2.py::test_persona_supervisor_passes_current_user_to_group_scene` | `tests/test_qq_group_public_scene_live_llm.py::test_live_participant_branch_isolation` | deterministic handoff; live supplemental | Prevents the selector from guessing the current participant or collapsing participants. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | `_promoted_reflection_evidence`, source-time-preserving guidance metadata | cognition input projection owner | `tests/unit/nodes/test_persona_supervisor2_cognition.py::test_promoted_reflection_preserves_source_updated_at`; `tests/unit/nodes/test_persona_supervisor2_cognition.py::test_promoted_self_guidance_is_goal_only_conditional_context` | `tests/test_cognition_core_v2_frozen_replay_drift.py::test_connector_separates_current_event_continuity_and_private_residue` | deterministic unit; exported replay | Prevents S7 reward strategy from becoming current-turn evidence and prevents timestamp recency fabrication. |
| `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` | goal prompt authority order and `_conversation_progress_evidence` temporal projection | goal-cognition semantic owner | `tests/unit/cognition_core_v2/test_goal_cognition.py::test_conversation_progress_evidence_preserves_temporal_provenance`; `tests/unit/cognition_core_v2/test_goal_cognition.py::test_goal_prompt_declares_one_primary_current_scene_objective`; `tests/test_cognition_prompt_contract_text.py::test_goal_prompt_labels_private_residue_and_guidance_as_non_authoritative` | `tests/test_cognition_core_v2_prompt_budget_continuity.py::test_goal_prompt_fits_maximum_evidence_without_duplication`; `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_captured_s8_group_noise_replay_keeps_injury_foreground`; `tests/test_cognition_core_v2_live_character_judgment.py::test_live_goal_progresses_high_affinity_guarded_continuity`; `tests/test_cognition_core_v2_live_character_judgment.py::test_live_goal_releases_stale_residue_for_changed_group_scene`; `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_separates_progress_and_optional_rows`; `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_accepts_one_progress_event`; `tests/test_cognition_core_v2_captured_run_failures_live_llm.py::test_captured_run_goal_relational_willingness_repair_live_llm`; `tests/test_cognition_core_v2_goal_capability_live_llm.py::test_live_captured_online_search_goal_preserves_required_evidence` | deterministic prompt contract; exported replay; real LLM one case at a time | Prevents a weak model from promoting stale residue/guidance into the current objective without increasing context or calls, across ordinary, required, recurrence, and repair branches. |
| `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` | required closed `CognitionEvidenceV2.authority` metadata and validated authority projection | cognition contract owner | `tests/unit/cognition_core_v2/test_contracts.py::test_cognition_evidence_requires_closed_typed_authority`; `tests/unit/cognition_core_v2/test_contracts.py::test_promoted_reflection_projects_conditional_guidance_authority` | `tests/test_cognition_core_v2_frozen_replay_drift.py::test_connector_separates_current_event_continuity_and_private_residue` | deterministic contract | Prevents free-text or source-kind ambiguity from promoting guidance, residue, or lore into current-event authority. |
| `src/kazusa_ai_chatbot/internal_monologue_residue/recorder.py` | v2 residue record result and fail-closed persistence status | residue lifecycle owner | `tests/test_internal_monologue_residue_recorder.py::test_record_completed_episode_writes_scoped_clear_barrier_for_empty_output`; `tests/test_internal_monologue_residue_recorder.py::test_empty_residue_write_failure_is_not_reported_as_cleared` | `tests/test_internal_monologue_residue_integration.py::test_post_turn_records_internal_monologue_residue_in_background` | deterministic/patched | Prevents a partial write from being reported as cleared and retains the existing exact-scope error shield while the disposition contract is applied. |
| `src/kazusa_ai_chatbot/internal_monologue_residue/recorder.py` | typed `append`/`replace_scope`/`clear_scope` recorder disposition and bounded regeneration | residue lifecycle owner | `tests/test_internal_monologue_residue_recorder.py::test_recorder_requires_typed_disposition`; `tests/test_internal_monologue_residue_recorder.py::test_recorder_disposition_text_invariants_are_fail_closed`; `tests/test_internal_monologue_residue_recorder.py::test_record_completed_episode_operation_id_is_stable` | `tests/test_internal_monologue_residue_live_llm.py::test_live_residue_related_continuation_selects_append`; `tests/test_internal_monologue_residue_live_llm.py::test_live_residue_scene_pivot_selects_replace_scope`; `tests/test_internal_monologue_residue_live_llm.py::test_live_residue_no_continuation_selects_clear_scope`; `tests/test_internal_monologue_residue_live_llm.py::test_live_residue_private_boundary_does_not_narrate_residue` | deterministic/patched; real LLM one case at a time | Prevents deterministic code from guessing a semantic pivot and prevents non-empty crisis residue from surviving a model-owned replacement decision. |
| `src/kazusa_ai_chatbot/internal_monologue_residue/loader.py` | barrier-aware scope selection and load status | residue loader owner | `tests/test_internal_monologue_residue_loader.py::test_select_residue_window_stops_at_scoped_empty_clear_barrier`; `tests/test_internal_monologue_residue_loader.py::test_clear_barrier_keeps_other_user_residue_isolated` | `tests/test_internal_monologue_residue_loader.py::test_load_residue_context_can_disable_read_telemetry` | deterministic unit | Preserves exact user/group scope while hiding only older rows in the cleared or replaced scope. |
| `src/kazusa_ai_chatbot/internal_monologue_residue/projection.py` | empty marker exclusion from prompt projection | residue projection owner | `tests/test_internal_monologue_residue_projection.py::test_project_residue_window_ignores_clear_barrier_rows` | `tests/test_internal_monologue_residue_projection.py::test_project_residue_window_prefers_newer_rows_when_budget_is_tight` | deterministic unit | Prevents a marker from becoming visible character text. |
| `src/kazusa_ai_chatbot/internal_monologue_residue/models.py` | `ResidueLoadResult`/`ResidueRecordResult` clear status fields | residue contract owner | `tests/test_internal_monologue_residue_recorder.py::test_empty_residue_result_has_cleared_contract`; `tests/test_internal_monologue_residue_loader.py::test_clear_barrier_load_reports_cleared_status` | `tests/test_internal_monologue_residue_integration.py::test_residue_row_uses_configured_character_id_fallback` | deterministic contract | Keeps lifecycle state inspectable without leaking residue content. |
| `src/kazusa_ai_chatbot/db/internal_monologue_residue.py` | v2 operation idempotency, conflict detection, and bounded retention index | residue persistence owner | `tests/test_internal_monologue_residue_database.py::test_residue_operation_is_transition_idempotent_and_conflict_safe`; `tests/test_internal_monologue_residue_database.py::test_residue_indexes_include_operation_uniqueness_and_purge_ttl` | `tests/test_internal_monologue_residue_integration.py::test_post_turn_records_internal_monologue_residue_in_background` | deterministic DB-contract test | Prevents duplicate markers/replacements, ambiguous retries, and unbounded marker growth. |
| `src/kazusa_ai_chatbot/db/schemas.py` | required canonical residue document fields and source references | residue persistence contract owner | `tests/test_internal_monologue_residue_database.py::test_v2_residue_schema_requires_disposition_operation_and_retention` | `tests/test_internal_monologue_residue_loader.py::test_noncanonical_rows_are_excluded_from_the_residue_window` | deterministic contract | Prevents malformed or non-canonical residue rows from reaching persistence or prompt context. |
| `src/kazusa_ai_chatbot/config.py` | bounded 48-hour residue retention setting | runtime configuration owner | `tests/test_internal_monologue_residue_database.py::test_residue_retention_config_is_bounded_and_documented` | none | deterministic config contract | Prevents reset markers and replacement rows from growing without a bounded lifecycle. |
| `src/kazusa_ai_chatbot/brain_service/post_turn.py` | trace-linked final progress disposition telemetry | post-turn observability owner | `tests/test_conversation_progress_v2_service.py::test_post_turn_emits_trace_linked_progress_disposition`; `tests/test_conversation_progress_v2_service.py::test_post_turn_preserves_trace_link_when_diagnostic_event_write_fails` | `tests/test_service_event_logging.py::test_process_queued_item_suppresses_routine_success_events` | patched/event contract | Distinguishes a frozen packet's actual post-turn outcome from an unproven race hypothesis, including diagnostic-write interruption. |
| `src/kazusa_ai_chatbot/event_logging/models.py`, `recording.py`, `__init__.py` | typed `continuity_boundary` event family, sanitizer, public recorder, and exports | event logging contract owner | `tests/test_event_logging_interface.py::test_continuity_boundary_recorder_is_keyword_only_and_exported`; `tests/test_event_logging_interface.py::test_continuity_boundary_payload_is_bounded_and_text_free`; `tests/test_event_logging_interface.py::test_continuity_boundary_unknown_status_is_not_success` | `tests/test_service_event_logging.py::test_progress_disposition_telemetry_is_trace_linked_and_sanitized` | deterministic event contract | Makes partial/unknown continuity outcomes inspectable without leaking user content or overclaiming database state. |
| `src/kazusa_ai_chatbot/event_logging/README.md` | continuity event ICD, forbidden fields, and source ownership | event logging documentation owner | `tests/test_event_logging_interface.py::test_continuity_boundary_event_documentation_matches_contract` | none | static/documentation | Prevents future instrumentation from overloading `database_operation` or logging prompt/residue content. |
| `src/kazusa_ai_chatbot/conversation_progress/README.md` | public group-scene and diagnostic ICD | conversation-progress documentation owner | `tests/test_conversation_progress_stage12_architecture.py::test_conversation_progress_documentation_preserves_public_scene_boundary` | none | static/documentation | Prevents future code from persisting group state or treating progress as planner authority. |
| `src/kazusa_ai_chatbot/internal_monologue_residue/README.md` | empty marker and guidance boundary ICD | residue documentation owner | `tests/test_internal_monologue_residue_prompt_boundaries.py::test_internal_monologue_residue_documents_clear_barrier_and_goal_only_scope` | none | static/documentation | Prevents empty-result semantics and private-lane ownership from regressing. |
| `src/kazusa_ai_chatbot/cognition_core_v2/README.md` | public-scene, temporal provenance, and one-objective authority ICD | cognition documentation owner | `tests/test_cognition_core_v2_prompt_contract_guidance.py::test_goal_prompt_documents_one_objective_evidence_authority` | none | static/documentation | Keeps future prompt changes aligned with the weak-model authority contract. |
| `src/kazusa_ai_chatbot/brain_service/README.md` | post-turn diagnostic ownership and transient group-scene boundary | brain-service documentation owner | `tests/test_service_event_logging.py::test_progress_disposition_telemetry_is_trace_linked_and_sanitized` | none | static/event contract | Prevents unlinked post-turn failures and accidental persisted group continuity. |
| `tests/fixtures/qq_group_topic_continuity_regression.json` | captured S1/S2/S4/S6/S7/S8 all-lane interleavings, redactions, hashes, identity, and hard gates | regression fixture owner | `tests/test_conversation_progress_v2_regression.py::test_captured_group_topic_continuity_fixture_preserves_public_continuity_anchors` | `tests/test_qq_group_public_scene_live_llm.py::test_live_group_crisis_anchor_beats_other_user_noise`; `tests/test_qq_group_public_scene_live_llm.py::test_live_group_reward_control_remains_playful`; `tests/test_qq_group_public_scene_live_llm.py::test_live_group_same_user_continuity_survives_unrelated_noise`; captured S1/S2/S4/S6/S7/S8 live nodes listed in Gate D | deterministic fixture; real LLM one case at a time | Covers the in-scope authority collapse and characterizes the out-of-scope modes without claiming them fixed. |
| `tests/ownership/source_test_impact_manifest.json` | ownership rows for conversation-progress, residue, and post-turn surfaces | test-governance owner | `tests/test_test_impact_manifest.py::test_manifest_contains_group_topic_continuity_owner_rows` | none | deterministic governance | Prevents production changes from bypassing exact owner tests. |

Existing live isolation nodes remain required and must pass individually:

- `tests/test_e2e_live_llm.py::test_live_chat_multi_user_photo_thread_keeps_user_intents_separated`
- `tests/test_e2e_live_llm.py::test_live_chat_multi_user_quantization_thread_keeps_xuezhang_bound_to_haodieyou`
- `tests/test_e2e_live_llm.py::test_live_chat_multi_user_understanding_thread_keeps_joke_and_self_definition_separate`
- `tests/test_e2e_live_llm.py::test_live_chat_multi_user_preferences_remain_isolated_across_suffix_english_and_switch`
- `tests/test_conversation_progress_v2_live_llm.py::test_live_interleaved_group_multifragment_continuation`
- `tests/test_conversation_progress_v2_live_llm.py::test_live_group_stale_ambient_is_absent_from_stage_zero_prompt`
- `tests/test_short_horizon_state_composition_e2e_live_llm.py::test_private_event_changes_next_group_turn`
- `tests/test_short_horizon_state_composition_e2e_live_llm.py::test_group_event_changes_next_private_turn`

## Risk Register And Regression Controls

| Failure mode | Regression risk | Required detection | Delivery control |
| --- | --- | --- | --- |
| Current-user causal anchors are evicted by newer group noise or final character fitting. | S8-like competing outputs return because the model sees ambient novelty but loses the injury/recovery pair. | History/projection owner tests, captured S6/S7/S8 replay, and the live crisis-anchor case. | Any lost required anchor is a hard failure; no live quality pass can waive it. |
| A broadcast assistant row or another participant's addressed row is attributed to the current user. | The character answers the wrong person or inherits another participant's branch. | Explicit-address deterministic tests, public-address live cases, and participant-branch isolation. | Broadcast rows remain ambient only; any target or private-continuity misbinding blocks delivery. |
| User-scoped progress or residue crosses participant boundaries. | Private facts, relationship state, or unresolved residue contaminate another user's response. | Scope-key unit tests, residue barrier tests, existing multi-user isolation tests, and live per-user cases. | A privacy/scope failure is fail-closed and invalidates all later gates. |
| Current-event authority collapses with residue, promoted guidance, lore, or stale continuity. | S7/S8 competing objectives, unsupported death/punishment premises, or reward strategy replacing the actual event. | Prompt-contract tests, source-time tests, captured replay, and live S6/S7/S8-shaped cases. | The current episode/public scene must own the objective; stale or conditional lanes may affect tactics or tone only. |
| The fix overcorrects into silence or flat, generic responses. | Grounded reward negotiation, ask-backs, or character judgment disappear in ordinary group conversation. | Live reward-control, public-topic-pivot, and noise-only behavior cases plus human review. | A supported current reason to speak must remain available; no keyword suppression or forced-silence behavior is accepted. |
| Context shaping increases prompt size, calls, retries, or foreground latency. | The weak local model becomes less reliable or the group path becomes operationally unavailable. | Prompt-budget, call-count, runtime-diagnostic, and latency evidence. | The 36K goal cap, 50K baseline, existing routes, attempt caps, and foreground call shape are hard invariants. |
| Reflection timestamps or residue barriers fabricate recency or clear the wrong scope. | Old guidance is treated as current evidence, or a valid user's continuity disappears. | Provenance/age unit tests, exact-scope residue tests, and positive same-scope carryover tests. | Invalid provenance is omitted; barrier writes affect only the exact scope; any over-clear or stale-source failure blocks. |
| A non-empty crisis residue survives because an empty-only marker cannot express a scene pivot. | The loader hides only a future empty marker while older reward/attack residue remains and competes with the current episode. | Real recorder disposition cases, `replace_scope` loader tests, duplicate/conflict tests, and captured all-lane S7/S8 replay. | The LLM owns `append`/`replace_scope`/`clear_scope`; deterministic validation rejects ambiguity; replacement is a scoped barrier and is idempotent. |
| A non-canonical residue row remains in the collection or canonical cutover is interrupted. | A malformed or pre-v2 row reaches prompt context, or a partial rollout invents a disposition and changes continuity. | Canonical schema/reader rejection tests, collection preflight, restart/reconciliation tests, and rollback drill. | Query and selector accept only canonical v2 rows; deployment blocks on non-canonical storage; rollback retains the canonical reader and never invents a legacy default. |
| Diagnostics claim fields the event contract cannot carry or overclaim a commit after cancellation. | RCA evidence is incomplete or a partial/unknown write is accepted as success. | Event-family schema/sanitizer tests, cancellation-after-commit reconciliation, and trace-linked artifacts. | Use the explicit `continuity_boundary` family; unknown is a blocking state and durable packet/operation references reconcile it. |
| Protected-anchor fitting truncates a required text to empty. | The selector reports a retained anchor while the weak model receives no injury/recovery text. | Text-level minimum assertions, over-cap protected-set failure test, targetless/private controls. | Drop ambient first; require non-empty trigger/anchors; return typed degraded result when minima cannot fit. |
| Captured S1/S2/S4 reality is treated as solved by an S6-S8 change. | Delivery claims are broader than evidence and downstream surface failures remain unowned. | Direct all-lane characterization replays and before/after non-expansion comparison. | The plan's acceptance claim is explicitly S6-S8 only; S1/S2/S4 remain residual until a separate approved plan. |
| Progress persistence or diagnostic emission is interrupted or partially completed. | A frozen packet is mistaken for a successful write, cache publishes uncommitted state, or the failure cannot be attributed. | Injected guarded-write/cache tests, trace-linked disposition tests, and exported diagnostic evidence. | Unknown, partial, or unlinked outcomes are incomplete—not passing—and require recovery from the last checkpoint. |

The implementation owner must preserve this register in the execution evidence.
Each failed control records the failing case, the observed boundary, the owner,
and the next checkpoint; it is not reclassified as acceptable variation.

## Change Surface

### Delete

- No production module, storage collection, adapter path, or existing user-isolation test is deleted.
- Remove the legacy residue document type, legacy backfill routine, legacy append inference, and old compatibility assertions. Replace the old empty-residue no-write assertion with the typed disposition/barrier contract.

### Modify

- `src/kazusa_ai_chatbot/conversation_progress/history.py`: add the pre-cap current-user anchor selector.
- `src/kazusa_ai_chatbot/conversation_progress/models.py`: add deterministic-only anchor metadata and required bounded diagnostics fields.
- `src/kazusa_ai_chatbot/conversation_progress/projection.py`: reserve/protect anchors and preserve render caps.
- `src/kazusa_ai_chatbot/conversation_progress/runtime.py`: pass group-anchor mode, compute bounded diagnostics, and keep cache/write behavior unchanged.
- `src/kazusa_ai_chatbot/service.py`: pass channel-derived group-anchor mode and carry correlation/trace references for post-turn observation.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: pass the typed current user into group-scene projection.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: preserve promoted source timestamps and mark self-guidance as conditional guidance in the existing evidence carrier.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` and `goal_cognition.py`: add validated typed evidence authority, preserve temporal provenance, and update the one-objective prompt procedure without changing the goal output schema or call policy.
- `src/kazusa_ai_chatbot/internal_monologue_residue/models.py`, `loader.py`, `projection.py`, and `recorder.py`: implement typed disposition, canonical exact-scope barriers, and statuses.
- `src/kazusa_ai_chatbot/db/internal_monologue_residue.py` and `src/kazusa_ai_chatbot/db/schemas.py`: implement operation idempotency, conflict detection, the required canonical persisted fields, TTL/index configuration, and bounded retention.
- `src/kazusa_ai_chatbot/config.py`: add the bounded residue-retention setting and preserve the existing prompt/context limits.
- `src/kazusa_ai_chatbot/event_logging/models.py`, `recording.py`, `__init__.py`, and `README.md`: add the explicit text-free `continuity_boundary` event family and public contract.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`: emit trace-linked sanitized progress disposition telemetry and reconcile interrupted operations.
- The four named subsystem READMEs: reconcile the new contracts.
- `tests/ownership/source_test_impact_manifest.json`: add exact ownership/test rows for the new behavior.

### Create

- `tests/fixtures/qq_group_topic_continuity_regression.json`: captured S1/S2/S4/S6/S7/S8 all-lane fixture with redaction manifest, hashes, and route/character identity.
- `tests/test_internal_monologue_residue_live_llm.py`: one-at-a-time real recorder quality cases.
- The exact new pytest nodes named in `Test Impact And Traceability`.
- During verification, a human-authored review artifact under `test_artifacts/llm_debug/` for the live/replay quality evidence.

### Keep

- `ConversationProgressScope` per-user/channel keying.
- Transient `public_group_scene` projection and existing group turn/render caps.
- Current model routes, output schemas, retry/attempt caps, and foreground latency shape.
- RAG evidence ownership, workspace collapse, action planning, L3/dialog wording ownership, verifier boundaries, adapters, intake, settlement, and delivery.
- Character judgment: the plan preserves grounded speaking, negotiation, and playful reward behavior when the current scene actually supports them.

### Ownership manifest requirements

Before implementation, update `tests/ownership/source_test_impact_manifest.json`
in the same change set. Its `source_roots` must include every exact production
source path this plan permits to change, including:

- `src/kazusa_ai_chatbot/conversation_progress/**`;
- `src/kazusa_ai_chatbot/internal_monologue_residue/**`;
- `src/kazusa_ai_chatbot/db/internal_monologue_residue.py` and
  `src/kazusa_ai_chatbot/db/schemas.py`;
- `src/kazusa_ai_chatbot/event_logging/__init__.py`, `models.py`, and
  `recording.py`;
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` and
  `goal_cognition.py`;
- `src/kazusa_ai_chatbot/config.py`;
- `src/kazusa_ai_chatbot/service.py`,
  `src/kazusa_ai_chatbot/brain_service/post_turn.py`, and the two persona
  supervisor modules.

Every Python source file under those roots must have an exact manifest entry
with a required deterministic owner node; directory coverage alone is not
enough. The validator's changed-source check is a delivery gate, and a plan
matrix row alone is insufficient. If implementation discovers an additional
source path, it stops, adds the path/root/owner row, and reruns collection
before editing that path. The final evidence records the manifest diff and the
exact `validate_test_impact --base-ref HEAD --run` result.

## Cutover Policy

Overall strategy: bigbang for the in-memory selector, cognition projection,
typed goal authority, canonical residue schema/reader/writer, and an explicitly
bounded telemetry addition.

| Area | Policy | Instruction |
| --- | --- | --- |
| Public-scene selection | bigbang | Replace newest-only group selection with participant-aware protected-anchor selection directly. |
| Goal evidence lanes | bigbang | Replace unlabelled promoted self-guidance projection with the fixed conditional-guidance authority label. |
| Residue reader/schema | bigbang | Deploy one required `InternalMonologueResidueV2Doc` contract; query and selector reject rows without the canonical schema, disposition, operation, and retention fields. |
| Residue writer lifecycle | bigbang | Use typed `append`/`replace_scope`/`clear_scope` directly; no legacy writer, optional-field alias, or compatibility vocabulary remains. |
| Existing non-empty residue | operational preflight | Require the collection to contain only canonical rows; any non-canonical data blocks deployment and is handled by a separately approved data cleanup. |
| Residue rollback | canonical reader only | Stop the writer only while retaining the canonical reader and barrier semantics; never roll back to a pre-v2 reader or fabricate missing fields. |
| Continuity diagnostics | additive typed event family | Add `continuity_boundary` with bounded labels/counts and no content; event failure is best effort and cannot report a DB write as successful. |
| Progress persistence algorithm | deferred | Capture the actual disposition first; do not cut over a guessed locking or retry design in this plan. |
| Tests and ownership manifest | bigbang | Replace obsolete contract assertions and add the new exact owner rows. |

## Interruption, Recovery, And No-Partial-Acceptance

The delivery state is checkpointed in this fixed order:

1. `baseline_captured`: worktree status, owned-file set, plan revision, and
   protected evidence paths are recorded.
2. `owner_tests_collected`: every exact mapped deterministic node is collected;
   unmapped, stale, deselected, or missing nodes stop the work.
3. `owner_tests_passed`: all mandatory deterministic and patched nodes pass.
4. `replay_passed`: the sanitized fixture and S6/S7/S8 replay hard gates pass,
   with the human-readable review artifact authored from the raw evidence.
5. `live_passed`: every mandatory real-LLM case has been run individually,
   inspected, and accepted against the hard gates and behavioral rubric.
6. `signoff_passed`: the independent reviewer accepts scope, evidence,
   residual risk, and the production diff.

Interruption rules are mandatory:

- An interrupted command, timeout, process exit, or unavailable model leaves
  the current checkpoint incomplete. It never becomes a pass by omission.
- Resume from the last completed checkpoint after recapturing `git status`,
  the owned-file diff, plan revision, and artifact inventory. Re-run the
  interrupted gate and every later gate that depends on it.
- Any production-source, prompt, fixture, or test change after a completed
  gate invalidates that gate and all dependent evidence. The owner reruns the
  exact mapped nodes before proceeding.
- A real-LLM transport or harness failure is preserved as raw evidence and
  classified before a single-case rerun. A semantic quality or privacy failure
  blocks delivery; rerunning it cannot erase the failed result.
- A database or telemetry result that is unknown, partial, duplicated, or not
  trace-linked blocks acceptance. The owner resolves the state with the
  deterministic failure-path tests and records the disposition before retrying.
- No production cutover occurs before `signoff_passed`. If a released revision
  later violates a hard gate, the deployment owner rolls back that single
  revision through the normal release mechanism while retaining database rows
  and all evidence; no destructive data cleanup is part of this plan.
- Recovery preserves pre-existing user worktree changes and never uses a
  broad reset or cleanup as an interruption remedy.

Interruption-specific persistence handling is explicit. The writer derives a
stable `operation_id` before the DB call. If cancellation occurs before the
result is returned, the post-turn worker records `unknown` and the recovery
step queries by `operation_id`/`llm_trace_id`; it must resolve to
`reconciled_written` or `reconciled_absent` before the checkpoint can pass. A
telemetry write that is interrupted after the DB commit is reconstructed from
the durable row/packet reference, not guessed from an absent event. Duplicate
recovery attempts are idempotent and never publish cache state until the
underlying guarded write is confirmed.

## Agent Autonomy Boundaries

The implementation owner may choose local helper names, function decomposition, deterministic sorting mechanics, test fixture construction details, and command order within the named paths. It must preserve the exact authority order, anchor rules, caps, scope keys, marker semantics, prompt/output schema, and call/latency constraints.

The implementation owner must request a plan amendment before changing a
persisted schema beyond the v2 fields explicitly named here, adding any event
family beyond `continuity_boundary`, adding a new LLM stage or retry, changing
`ConversationProgressScope`, changing the 50K/36K limits, modifying deferred
owners, or weakening any isolation/acceptance gate. The named residue cutover
and event family are already in scope. The owner must stop and report any
source/contract conflict rather than silently add a compatibility path.

## Verification

Before implementation, capture `git status --short` and the explicit owned-file baseline. Preserve unrelated user changes.

Run deterministic and patched nodes from the matrix in batches using `venv\\Scripts\\python -m pytest`. Confirm every mapped node is collected before accepting its source row. The implementation owner must run the exact owner unit nodes; a broader suite cannot substitute for an uncollected mapped node.

Run exported S6/S7/S8 replay nodes with the sanitized fixture and inspect the projected scene, goal payload, lane provenance, admitted objective, and hard-gate results. Keep raw protected trace paths in the review artifact rather than copying private prompt/output content into the fixture.

Run the new and existing real-LLM nodes one at a time with the local production route. After each case, inspect the durable trace/review output before starting the next case. The human-readable review must separate harness/validation status from quality judgment and record:

- current trigger, public anchors, participant roles, and per-user continuity source;
- raw and parsed goal output, selected objective, evidence handles, and expected consequences;
- whether stale residue/guidance changed the topic or only affected tone;
- whether the response remains characterful when a grounded reason to speak exists;
- prompt character counts, model route, attempt count, latency, and any degraded verifier state;
- regressions, acceptable variation, and residual evidence gaps.

Run the existing multi-user isolation nodes individually, followed by the independent reviewer. The reviewer must verify no cross-user private continuity, no broadcast misattribution, no context-cap increase, no new foreground call, and no hidden persistence change.

## Mandatory Delivery Gates

The following gates are delivery-blocking and execute in order. A passing
broader suite, a green harness result, or a favorable single output cannot
waive an earlier gate.

### Gate A: baseline, ownership, and collection

- The plan is `approved` or `in_progress`, implementation authorization is
  recorded, and the pre-existing worktree baseline is preserved.
- `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run`
  passes after the implementation diff exists.
- Every exact deterministic node in `Test Impact And Traceability` is
  collected. A missing, stale, deselected, or `xfail` owner node blocks
  delivery.

### Gate B: deterministic owner and contract coverage

Every exact deterministic node in the matrix must pass, including the new
history, public-scene, runtime, service, persona, reflection, goal, residue,
post-turn, documentation, fixture, and manifest nodes. The following existing
boundary tests are additionally mandatory:

- `tests/test_conversation_progress_history_policy.py::test_history_caps_match_approved_contract`
- `tests/test_conversation_progress_history_policy.py::test_newest_ten_interaction_turns_preserve_chronology`
- `tests/test_conversation_progress_history_policy.py::test_prompt_projection_keeps_complete_newest_turns_within_budget`
- `tests/test_conversation_progress_history_policy.py::test_prompt_projection_never_exposes_protected_row_or_trace_ids`
- `tests/test_conversation_progress_group_scene.py::test_group_scene_merges_trigger_and_labels_relative_order`
- `tests/test_conversation_progress_group_scene.py::test_group_scene_redacts_ids_and_resolves_visible_names`
- `tests/test_conversation_progress_group_scene.py::test_group_scene_applies_field_and_participant_caps`
- `tests/test_conversation_progress_group_scene.py::test_group_scene_prompt_drops_old_turns_and_keeps_trigger`
- `tests/test_conversation_progress_group_scene.py::test_group_scene_prompt_uses_semantic_labels_without_metadata`
- `tests/test_conversation_progress_group_scene.py::test_group_scene_skips_malformed_ambient_rows_and_keeps_trigger`
- `tests/test_conversation_progress_group_scene.py::test_group_scene_render_cap_is_non_fatal_for_oversized_context`
- `tests/test_conversation_progress_group_scene.py::test_group_scene_renderer_enforces_turn_cap_for_shaped_context`
- `tests/test_conversation_progress_group_scene.py::test_group_scene_drops_aged_turns_and_never_drops_trigger`
- `tests/test_conversation_progress_group_scene.py::test_group_scene_filter_is_shared_with_stage_zero_history`
- `tests/test_conversation_progress_runtime.py::test_lost_guarded_write_does_not_publish_packet_to_cache`
- `tests/test_conversation_progress_runtime.py::test_progress_diagnostics_classify_guarded_write_outcomes`
- `tests/test_conversation_progress_runtime.py::test_interrupted_record_does_not_publish_uncommitted_cache_state`
- `tests/test_conversation_progress_v2_service.py::test_service_load_passes_v2_scope_bot_and_current_row_ids`
- `tests/test_conversation_progress_v2_service.py::test_ordinary_response_path_adds_no_llm_call`
- `tests/test_conversation_progress_v2_service.py::test_post_turn_preserves_trace_link_when_diagnostic_event_write_fails`
- `tests/test_persona_supervisor2.py::test_persona_supervisor_scopes_history_before_cognition`
- `tests/test_persona_supervisor2.py::test_persona_supervisor_projects_group_scene_alongside_user_history`
- `tests/test_persona_supervisor2.py::test_persona_supervisor_degrades_when_group_scene_projection_fails`
- `tests/test_persona_supervisor2.py::test_persona_supervisor_keeps_private_public_scene_empty`
- `tests/test_conversation_progress_cognition_evidence.py::test_evidence_occurred_at_uses_the_event_own_timestamp`
- `tests/test_conversation_progress_cognition_evidence.py::test_evidence_occurred_at_is_not_the_episode_timestamp`
- `tests/test_internal_monologue_residue_recorder.py::test_empty_residue_write_failure_is_not_reported_as_cleared`

### Gate C: replay and failure-mode coverage

- `tests/test_conversation_progress_v2_regression.py::test_captured_group_topic_continuity_fixture_preserves_public_continuity_anchors`
- `tests/test_cognition_core_v2_frozen_replay_drift.py::test_connector_separates_current_event_continuity_and_private_residue`
- `tests/test_cognition_core_v2_trace_failure_mode_matrix.py::test_captured_s8_group_noise_replay_keeps_injury_foreground`
- Every replay hard gate verifies current trigger retention, anchor identity,
  participant scope, lane provenance, one primary objective, absence of
  contradictory or ownership-changing premises, and unchanged prompt/call
  limits. Limited ungrounded embellishment is recorded for review instead of
  failing the replay by itself.
- The reviewer reads the replay payload and output evidence; a deterministic
  replay pass alone is not a quality judgment.

### Gate D: real local-LLM regression and quality coverage

Before collecting a live node, the `release_verification_owner` runs a live
preflight: verify the service health endpoint, exact character identity, exact
local model route and effective 50,000 context setting from the trace/config,
MongoDB connectivity and explicit isolated test database, artifact directory
writability, protected fixture availability, and the debug-channel permission
for the selected identities. A missing preflight item blocks live execution.

Because `pytest.ini` excludes live markers by default, collection and execution
must use the explicit override below. Every case is run separately; a skipped,
deselected, xfailed, or fixture-setup failure is a gate failure, not a pass:

```powershell
$env:PYTHONPATH="src"
venv\Scripts\python.exe -m pytest -o addopts= -m live_llm --collect-only -q tests/<live_file>.py::<node>
venv\Scripts\python.exe -m pytest -o addopts= -m live_llm tests/<live_file>.py::<node> -q -s
```

Replace the placeholders with one exact node per command and preserve the
collection/run output in the artifact. Inspect the raw trace, parsed output,
contract result, and human quality review before starting the next node.

The mandatory captured all-lane cases are:

- `tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s1_private_surface_characterization`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s2_public_boundary_characterization`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s4_reality_correction_characterization`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s6_final_surface_keeps_triage_counteraction_return_order`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s7_all_lanes_keep_crisis_foreground`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s8_all_lanes_keep_injury_foreground_repeat_1`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s8_all_lanes_keep_injury_foreground_repeat_2`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_captured_s8_all_lanes_keep_injury_foreground_repeat_3`

The S1/S2/S4 nodes are characterization/non-expansion gates, not claims that
this plan fixes those baseline failures. The S6-S8 nodes must reproduce the
current episode, unrelated public users, current-user progress, real residue,
promoted guidance, and final visible serialization; a synthetic public-only
case does not satisfy the gate.

The mandatory new focused group controls are:

- `tests/test_qq_group_public_scene_live_llm.py::test_live_group_crisis_anchor_beats_other_user_noise`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_group_reward_control_remains_playful`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_group_same_user_continuity_survives_unrelated_noise`

The changed goal branches also require real-model evidence:

- `tests/test_cognition_core_v2_live_character_judgment.py::test_live_goal_progresses_high_affinity_guarded_continuity`
- `tests/test_cognition_core_v2_live_character_judgment.py::test_live_goal_releases_stale_residue_for_changed_group_scene`
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_separates_progress_and_optional_rows`
- `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_accepts_one_progress_event`
- `tests/test_cognition_core_v2_captured_run_failures_live_llm.py::test_captured_run_goal_relational_willingness_repair_live_llm`
- `tests/test_cognition_core_v2_goal_capability_live_llm.py::test_live_captured_online_search_goal_preserves_required_evidence`

The residue recorder also requires the four one-at-a-time cases named in the
traceability table. They use the real recorder route and inspect both model
disposition and persisted result; patched tests alone do not prove recorder
quality.

The mandatory existing quality shield is:

- `tests/test_qq_group_public_scene_live_llm.py::test_live_public_target_distinct`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_parallel_addresses`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_public_topic_pivot`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_participant_branch_isolation`
- `tests/test_qq_group_public_scene_live_llm.py::test_live_noise_only_silence`
- `tests/test_conversation_progress_v2_live_llm.py::test_live_original_failure_progress_semantic_handoff`
- `tests/test_conversation_progress_v2_live_llm.py::test_live_asuna_houjing_long_thread_regression`
- `tests/test_conversation_progress_v2_live_llm.py::test_live_interleaved_group_multifragment_continuation`
- `tests/test_conversation_progress_v2_live_llm.py::test_live_group_stale_ambient_is_absent_from_stage_zero_prompt`
- `tests/test_conversation_progress_v2_live_llm.py::test_live_private_stale_progress_is_pruned_before_cognition`
- `tests/test_cognition_core_v2_p0_context_reconnection_live_llm.py::test_live_reply_residual_reaches_goal_only`
- `tests/test_cognition_core_v2_p0_context_reconnection_live_llm.py::test_live_group_self_cognition_uses_one_advisory_projection`
- `tests/test_e2e_live_llm.py::test_live_chat_multi_user_photo_thread_keeps_user_intents_separated`
- `tests/test_e2e_live_llm.py::test_live_chat_multi_user_quantization_thread_keeps_xuezhang_bound_to_haodieyou`
- `tests/test_e2e_live_llm.py::test_live_chat_multi_user_understanding_thread_keeps_joke_and_self_definition_separate`
- `tests/test_e2e_live_llm.py::test_live_chat_multi_user_preferences_remain_isolated_across_suffix_english_and_switch`
- `tests/test_short_horizon_state_composition_e2e_live_llm.py::test_private_event_changes_next_group_turn`
- `tests/test_short_horizon_state_composition_e2e_live_llm.py::test_group_event_changes_next_private_turn`

Each live case must satisfy all three layers:

- Harness: no exception, non-empty result, valid trace, parseable required
  output, correct route, and durable raw/parsed evidence.
- Hard contract: no private leakage, wrong participant/addressee or semantic
  owner, conflicting primary objectives, unauthorized action claim, prompt-cap
  violation, or changed call/retry shape.
- Quality rubric: the objective follows the current event, subordinate actions
  serve that objective, grounded reasons to speak remain available, and the
  character remains coherent and non-generic. The reviewer records this
  judgment in Markdown; exact wording is not asserted.

Live-LLM acceptance uses a tolerant weak-model threshold. Unless the output
contains a conflict, severe degradation, or altered semantic ownership, the
reviewer records it as a pass. Ungrounded facts, hallucinated details, vague
referents, and imperfect grounding are acceptable to a limited degree when
they remain non-conflicting, preserve the owning objective/participant/lane,
and do not severely degrade the response. These imperfections remain visible
in the review artifact as residual quality notes; they are not silently erased
or converted into deterministic expected-answer checks.

For this threshold, disagreement with a user's requested tactic or with a
conversation-progress ordering is not itself a conflict or ownership change.
Conversation progress remains supplemental evidence rather than planner
authority, and the character retains ownership of stance and tactical order.
The response passes when it preserves the same participant and primary
objective while choosing a different tactic, unless its own claims/actions are
mutually incompatible or the result is severely degraded. A categorical or
ungrounded rationale for that tactic is recorded as hallucination/residual
quality under the rule above.

For the final visible S6 surface, the hard ordering is triage/protection first,
counteraction second, and return/continuation third; the output must not contain
two contradictory primary locations or resurrect the reward branch as a
competing objective. For the S1/S2/S4 characterization cases, compare the
candidate with the captured baseline and block any new leakage, wrong
addressee/owner, contradiction, or severe measurable quality regression.
Limited hallucination or ungrounded detail is a documented residual and passes
when it does not cross those boundaries.

One hard-contract failure blocks delivery. A harness failure is not a pass and
requires an isolated recovery run. A quality concern is recorded with the raw
evidence; only a conflict, severe degradation, or altered ownership blocks
sign-off under the live-LLM threshold. Lesser grounding or hallucination
concerns pass with an explicit residual note.

### Gate E: capacity, interruption, and final sign-off

- Prompt character counts, model route, attempt count, foreground call count,
  and latency evidence remain within the fixed runtime contract. The candidate
  must retain every required lane and protected anchor, remain at or below
  `GOAL_COGNITION_PROMPT_CAP=36,000`, report the unchanged effective 50,000
  context setting, keep the same call/retry count, and have p95 foreground
  latency no more than 110% of the pre-change baseline on identical captured
  inputs. Missing baseline measurements block comparison rather than being
  treated as zero.
- All interruption rules in `Interruption, Recovery, And No-Partial-Acceptance`
  are satisfied; no incomplete checkpoint is accepted.
- The independent reviewer signs the final diff, exact collection output,
  deterministic results, replay review, live review, and residual-risk list.

## Development Guidance

Implement and verify in this order:

1. Add or update the deterministic owner tests for the public-scene selector,
   typed identity propagation, protected text minima, typed evidence authority,
   residue disposition/barrier/idempotency, event-family contract, and
   interruption reconciliation. Make these tests fail against the old behavior
   before changing production code.
2. Implement the canonical residue schema, barrier-aware selection, operation
   identity, TTL/index checks, and restart reconciliation as one big-bang
   contract. Remove old writers, optional-field aliases, legacy backfill, and
   legacy append parsing before the canonical gate is declared complete.
3. Implement the smallest deterministic public-scene and scope changes. Keep
   objective selection in goal cognition, keep authority typed, and keep all
   prompt-facing metadata semantic, bounded, and source-time aware. Assert
   protected text content, not only retained turn objects.
4. Add patched handoff tests for service-to-runtime-to-persona propagation,
   post-turn disposition reporting, cancellation, and cache safety. Use patched
   model responses only for state handoff, error, and retry mechanics; never use
   them as evidence that the prompt or recorder quality works.
5. Add the text-free `continuity_boundary` event family and instrument it as
   best-effort observability. Verify that an event-write failure preserves the
   response path and that an unknown DB state reconciles from a durable trace or
   operation reference.
6. Run the exact owner nodes after each semantic owner is changed, then run
   the complete deterministic Gate B set before touching live cases. Validate
   the manifest after the final source set is known.
7. Build the sanitized S1/S2/S4/S6/S7/S8 all-lane fixture and inspect the
   projected scene, evidence authority, residue disposition, final visible
   ordering, and hard gates before making a real model call.
8. Run the Gate D cases one at a time after live preflight on the unchanged
   local production route. Inspect raw and parsed output, prompt/version
   metadata, route, attempts, latency, call count, persistence disposition, and
   the character-quality rubric after every case.

Development must preserve the weak-model capacity boundary: compact context
shaping is preferred over more context, no new foreground LLM call or semantic
retry is added, and no prompt instruction may turn stale residue, reflection,
or another user's public row into current-event authority. Exact wording is
not a live assertion; target ownership, grounded objective, privacy, supported
speaking reason, and coherent subordinate actions are the acceptance signals.

When implementation and evidence disagree, the developer stops at the current
checkpoint, records the mismatch and affected owner, and requests a plan
amendment. The developer does not add a fallback mapper, compatibility alias,
keyword suppressor, silent retry, or unrelated downstream fix to make the gate
green.

## Acceptance Criteria

1. Under ten or more interleaved public turns, the trigger, newest current-user public turn, and newest assistant turn explicitly addressed to that user survive the current six-turn/1,800-character scene caps as non-empty participant continuity anchors. A causal relationship is asserted only when the chronological/reply-link rule is true.
2. The remaining scene capacity retains newest other-user public traffic in chronological order; group noise remains useful rather than being globally suppressed.
3. A broadcast assistant row without explicit address is visible public context but is not attributed to the current user's participant continuity.
4. User A's cognition input contains only A's progress and A's user-thread residue; B/C progress and residue remain absent. Public B/C turns remain visible only through the bounded public scene.
5. S6 keeps protection/survival primary, permits characterful subordinate counteraction, and does not surface reward as an unrelated objective.
6. S7 may respond to the attacker while unresolved injury/death remains the shared foreground; promoted reward guidance cannot become current-event evidence.
7. S8 keeps the current bleeding/injury statement under the injury/recovery
   objective and does not replace it with a conflicting fake-death or
   punishment objective. Ungrounded embellishment may pass when it remains
   subordinate and does not severely degrade the response. Anger or
   possessiveness may remain a subordinate character reaction.
8. A calm scene with an explicit current reward discussion still permits playful reward negotiation. The fix does not mechanically suppress reward content.
9. A valid residue result owns an explicit `append`, `replace_scope`, or `clear_scope` disposition. `replace_scope` and `clear_scope` hide older rows only in the exact current scope; `clear_scope` projects no text; a later non-empty same-scope append is loadable; duplicate operation retries are idempotent and conflicting retries fail closed.
10. Every recordable progress turn exposes a trace/operation-linked disposition, including skipped, contract/provider failure, write failure, guarded-write loss, unknown, reconciled, cache outcome, and success. A missing event is reconciled from durable source references before acceptance.
11. Every `CognitionEvidenceV2` row carries closed typed authority metadata. Promoted evidence preserves its source timestamp/age; conditional self-guidance cannot satisfy current-event evidence requirements and does not reach appraisal, workspace, L3, dialog, or adapters as a current-event fact.
12. The goal prompt remains at or below 36,000 characters, the configured 50K baseline remains unchanged, model-call count and retry caps remain unchanged, and no new foreground latency-producing stage exists.
13. All deterministic owner nodes are collected and pass; all exported replay hard gates pass; every real-LLM case has an inspected human-readable artifact; existing multi-user isolation nodes pass individually; captured S1/S2/S4 characterization shows no new attributable breakage.
14. Independent sign-off confirms the diff is limited to this plan, deferred persistence/decontextualizer/dialog changes were not smuggled into the patch, and residual uncertainty is recorded without weakening acceptance.
15. Gates A through E complete in order with no skipped, xfailed, interrupted, unknown, or unreviewed mandatory result.
16. Every mandatory live case passes its harness and hard-contract gates; every quality judgment is present in the human-authored review artifact and is independently accepted.
17. Any interruption resumes from the last completed checkpoint and re-runs all invalidated dependent gates before acceptance; no partial source, test, trace, cache, or database state is treated as delivered.
18. A post-release hard regression has a documented single-revision rollback path that preserves evidence and database rows, keeps the canonical barrier-aware residue reader in place until v2 rows expire, and introduces no destructive cleanup.
19. The S6 final visible surface preserves triage/protection, counteraction, and return/continuation order without contradictory primary locations or a competing reward objective.
20. The candidate's p95 foreground latency on identical captured inputs is at most 110% of the pre-change baseline, with unchanged effective 50,000 context configuration, 36,000 goal prompt cap, model route, call count, and retry caps.

## Progress Checklist

- [ ] Stage 0 diagnostics implemented and linked to trace/correlation.
- [ ] Stage 0 `continuity_boundary` ICD, sanitizer, recorder, exports, and trace/operation reconciliation implemented; evidence distinguishes progress outcomes without overclaiming commits.
- [ ] Stage 1 participant-aware public-scene selector implemented and deterministic tests pass.
- [ ] Stage 1 protected text minima, targetless-group behavior, and private-channel bypass are tested.
- [ ] Stage 1 reflection timestamp and conditional-guidance projection implemented and tested.
- [ ] `CognitionEvidenceV2.authority` typed metadata is required, validated, and mapped across all constructors.
- [ ] Stage 2 one-objective goal contract implemented across ordinary, required-selection, recurrence, and repair prompts.
- [ ] Stage 3 canonical residue cutover, typed disposition, operation idempotency, conflict handling, TTL, and rollback contract implemented and tested.
- [ ] Stage 4 sanitized all-lane S1/S2/S4/S6/S7/S8 fixture and exported replays pass hard gates.
- [ ] Stage 4 live cases, including three S8 repetitions and real recorder cases, run individually with inspected quality artifacts.
- [ ] Live preflight verifies service, character/model route, 50,000 context setting, isolated MongoDB, permissions, and artifact output.
- [ ] Baseline/candidate prompt, call-count, retry, and p95 latency comparison passes.
- [ ] Existing multi-user isolation and cross-scope continuity nodes pass individually.
- [ ] Gates A-E complete with no uncollected, skipped, interrupted, unknown, or unreviewed mandatory result.
- [ ] Interruption/recovery evidence records the last completed checkpoint and any invalidated gates.
- [ ] Independent code/quality review passes.
- [ ] S1/S2/S4 are documented as characterization-only residual modes with no attributable regression; no broader fix claim is made.
- [ ] Any progress-writer follow-up is recorded as a separate plan decision; this plan does not silently change the guarded progress persistence algorithm.

## Handover Document — 2026-08-13 (replacement)

This is the continuation checkpoint for the next parent agent. The plan
remains `in_progress`; acceptance remains unverified; no partial live run,
green deterministic batch, or model output is a delivery sign-off.

### Scope and ownership

- Active plan:
  `development_plans/active/bugfix/group_topic_continuity_authority_fix_plan.md`.
- Preserve the existing dirty worktree and inspect attribution before every
  edit. Do not reset or revert unrelated user changes. The pre-existing
  worktree baseline was already dirty before this execution.
- Parent responsibility: execute tests, preserve raw evidence, report observed
  results, coordinate independent semantic review, and obtain final sign-off.
  The parent does not convert keyword presence into a semantic quality verdict.
- Persistent primary implementation agent: DeepSeek Flash, fresh thread
  `019ff8a9-bb36-7ea3-839f-5687e7b5982c` (Gauss). Its acknowledgement passed
  and its current bounded execution handoff is the anti-cheating prompt/test
  cleanup. The next parent should wait on this thread with a hard deadline of
  at least 1,800,000 ms and preserve the 30-minute minimum wait.
- Independent reviewer already consulted: DeepSeek Pro thread
  `019ff897-99d0-7ff1-86c5-8e4d4eaa2c63` (Russell). Its report is read-only
  evidence, not final sign-off. A fresh Pro review is required after the
  anti-cheating cleanup and its dependent evidence reruns.
- Final sign-off agent: GPT-5.6 Sol, high reasoning, normal speed. The final
  agent must review the final diff, exact test evidence, raw/parsed live
  artifacts, baseline comparison, anti-cheating scan, and all residual risk.

### Verified checkpoint evidence

- Broad deterministic owner-oriented run, excluding `live_llm` and `live_db`:
  `509 passed, 2 skipped, 80 deselected in 29.11s`. The two skips remain
  unexplained and therefore do not satisfy a strict no-skips Gate B claim.
- Final impact validation after the current source/test set:
  `scripts.validate_test_impact --base-ref HEAD --run` validated 93 exact
  nodes, and `--check-all` validated 183 exact nodes. These supersede the
  stale handover statement that the impact validator was still failing.
- Python compilation of the changed Python set passed for 88 files. `git
  diff --check` reports only the repository's LF/CRLF conversion warnings.
- The runtime prompt scan before the pending cleanup found no direct fixture
  names or residue cue strings in `src/kazusa_ai_chatbot`, but the independent
  Pro review found a separate circularity: the goal prompt and ordered crisis
  metadata repeat exact visible crisis wording that is also used by a
  deterministic S6 cue-position helper. This is the active remediation scope.

### Live execution ledger

The following are execution observations, not parent semantic judgements:

- Direct graph isolation: latest photo-thread passed in 203.18s and latest
  quantization-thread passed in 70.08s. The understanding-thread passed in
  111.93s. Earlier setup failures were MongoDB connection closures and remain
  setup failures, not passes.
- Ordinary `/chat` preference isolation: latest correctly typed private/group
  harness run passed in 347.99s. The earlier empty English response was traced
  to the test harness omitting `ChatRequest.channel_type` for `dm`; the fixture
  now supplies the canonical private scope. No runtime prompt was changed for
  this repair.
- Conversation progress: the latest Asuna long-thread run passed in 74.07s
  with accepted scene/event recorder dispositions. The latest interleaved
  multifragment run failed in 5.13s because the live event output attempted a
  non-terminal transition on a completed event without explicit reopening.
  Deterministic contract tests preserve this fail-closed rule.
- Residue: scene-pivot and no-continuation cases previously produced their
  required typed dispositions. The latest related-continuation run returned
  `replace_scope` instead of the test's requested `append` disposition; the
  latest private-boundary run persisted forbidden residue text. Both remain
  live quality/contract failures and are not repaired by changing assertions.
- Short-horizon private-to-group and group-to-private runs have latest passing
  structural receipts after timestamp normalization and operational projection
  fixes. Earlier timeout and rejected-write artifacts remain superseded
  evidence and must not be deleted.
- P0 residual context has a hard leakage failure in a prior live artifact;
  P0 group-self latest rerun passed after test timestamp fixture repair. The
  changed-goal required-selection, relational-willingness, and online-search
  cases passed. The two character replay cases remain blocked by the missing
  `test_artifacts/cognition_core_v2/real_conversation_replay/production_character_state.json`.
- S1/S2/S4, S6/S7/S8, focused controls, residue, P0, e2e, and short-horizon
  artifacts exist in `test_artifacts`, but a complete set of independently
  authored human-readable semantic review Markdown files is not currently
  present. Existing raw/parsed artifacts remain the source evidence; no live
  pass is accepted solely from a keyword or non-empty assertion.

### Independent review findings to carry forward

- The Pro reviewer recommended `NOT ready for final GPT-5.6 Sol sign-off`.
- Confirmed blockers: missing Gate E identical-input baseline/p95 comparison,
  incomplete human-readable semantic review artifacts, unresolved live residue
  failures, the interleaved recorder failure, missing character replay input,
  and the prompt/test crisis-wording circularity.
- The Pro report's earlier claim that impact validation was failing is stale;
  the parent reran it successfully as recorded above. Its scope-drift finding
  for `character_carryover.py` and `state_projection.py` remains open for the
  final reviewer to classify against the plan's Change Surface. No production
  revert is authorized by this handover.
- The Pro review also observed raw `p1` handles in some earlier S8 visible
  artifacts. The next reviewer must classify this against the final visible
  surface and hard redaction contract rather than treating absence/presence of
  a string as the semantic quality decision.

### Active Flash handoff

The current Flash task is limited to removing test-answer steering while
preserving the generic semantic authority contract. It must:

1. Replace exact crisis wording in the runtime goal prompt and ordered
   contract metadata with neutral conceptual stage identifiers or wording.
2. Remove positional keyword/cue gates from the deterministic S6 harness.
3. Keep the live S6/S7/S8 tests as raw/parsed evidence and non-topic hard
   boundary checks; semantic coherence belongs to the independent reviewer.
4. Run focused deterministic prompt/contract/harness tests, compilation,
   collection, and the anti-cheating scan, then report exact changed files.

After the handoff returns, the next parent must inspect its diff, rerun all
invalidated deterministic owner nodes, and rerun every dependent live case in
Gate D one at a time. No semantic prompt or harness change is accepted without
fresh raw artifacts and a new independent Pro review.

### Required next-parent sequence

1. Wait for Flash thread `019ff8a9-bb36-7ea3-839f-5687e7b5982c` before making
   any edit in its owned scope. Review the diff, prompt scan, compile output,
   and focused deterministic results.
2. Run the exact deterministic owner set and both impact-validator commands
   after the cleanup. Preserve their output. Treat any skipped mandatory node
   as blocking.
3. Collect and execute every invalidated live node one at a time with an
   isolated guarded MongoDB database. Inspect raw prompt, parsed output,
   contract result, route, attempt count, latency, persistence disposition,
   and visible response after each case.
4. Produce evidence-index Markdown files containing raw/parsed paths and
   observed contract/operational results. Obtain semantic quality judgements
   from the new DeepSeek Pro reviewer; the parent records those judgements
   without inventing its own.
5. Capture a same-fixture baseline and candidate comparison for effective
   50,000 context, 36,000 goal cap, model route, call/retry counts, and p95
   foreground latency. Missing baseline blocks Gate E.
6. Resolve the missing replay artifact, P0 leakage result, residue failures,
   scope-drift classification, and S8 handle observation. Then request a
   fresh GPT-5.6 Sol final review. Update checklist/status only after that
   agent explicitly signs off Gates A–E.

### Non-negotiable boundaries

- S1/S2/S4 remain characterization-only; no broader fix claim is permitted.
- No keyword gate, expected answer, fixture-specific cue, semantic retry,
  fallback mapper, or deterministic rewrite of model wording may enter any
  prompt or delivery stage.
- A harness/setup failure, timeout, skipped node, contract failure, missing
  review artifact, or unmeasured baseline is not a pass.
- Do not read `.env`, mutate production data, delete prior artifacts, or reset
  the dirty worktree. Preserve interrupted and superseded evidence.

## Closure Attempt Evidence — 2026-08-13

This section supersedes the operational next-parent sequence above while
preserving that sequence as historical handoff evidence. The plan remains
`in_progress`; Gates D and E are incomplete, so it is not eligible for archive.

### Anti-cheating remediation and deterministic verification

- The implementation handoff removed fixture-shaped S6 ordering language and
  ordered scenario metadata from the runtime goal prompt. The remaining prompt
  contract is generic: typed evidence authority, source provenance, one primary
  objective, and coherent subordinate actions.
- The deterministic live harness now enforces only non-empty visible output and
  protected internal-identifier boundaries. Live S6/S7/S8 nodes preserve raw and
  parsed evidence for semantic review instead of encoding expected topic words
  or answer positions.
- Parent scans found no S6/S7/S8 scenario vocabulary or ordered-answer metadata
  in the runtime goal prompt or dialog prompt surfaces.
- All 76 exact deterministic nodes named by the plan collected and passed with
  zero skips. The changed deterministic test surface completed with `662 passed,
  2 skipped, 98 deselected`; the two skips were optional day-wide trace-inventory
  checks and were not among the 76 mandatory nodes.
- `scripts.validate_test_impact --base-ref HEAD --run` passed 93 exact nodes, and
  `scripts.validate_test_impact --base-ref HEAD --check-all` validated 183 exact
  nodes. Compilation of the changed Python surface passed.

### Guarded live preflight and protected evidence

- Live verification used the explicit isolated database
  `_test_group_topic_continuity_20260813` with all repository test guards set.
  The current-workspace service health check returned `200` with database and
  scheduler health true.
- The local route was `gemma-4-31b-fable-5-agent-distill`; the loaded effective
  context was `50176`, and `GOAL_COGNITION_PROMPT_CAP` remained `36000`.
- The previously missing protected replay fixture was recovered byte-for-byte
  from a clean sibling checkout of the same repository remote and restored at
  `test_artifacts/cognition_core_v2/real_conversation_replay/production_character_state.json`.
  Its SHA-256 is
  `239FC1071F4B6E6812D30E753B3D64298FBAC1C18FF79CFBFB2D814FCE8B7B4B`.
- Shared/production database rows were preserved. The isolated live database had
  no residue rows at preflight. The parent stopped the port-8011 verification
  service after Gate D stopped.

### Gate D independent verdict

- Case 1, S1 private-surface characterization, collected exactly one node and
  passed pytest in 155.09 seconds. Independent review accepted it only as a
  characterization/non-expansion result and recorded sensitive-phrase
  restatement plus internal-handle trace-hygiene risks.
- Case 2, S2 public-boundary characterization, collected exactly one node and
  passed pytest in 216.19 seconds, but independent raw-evidence review failed it.
  The final response asserted that an undefined group question had an
  affirmative answer even though the public scene contained no yes/no
  proposition and the admitted objective said the concrete question first had
  to be identified. This is an unsupported conclusion and an incoherent
  objective, so it fails the hard-contract and semantic-quality layers.
- The verifier stopped without a rerun. Gate D nodes 3 through 39 remain
  uncollected and unrun in this attempt, as required by the stop-on-first-failure
  rule.
- Authoritative readable evidence is under
  `test_artifacts/llm_debug/group_topic_continuity_gate_d/`, especially
  `gate_d_current_attempt_evidence_index.md`,
  `case_01_s1_private_surface_review.md`, and
  `case_02_s2_public_boundary_review.md`. Raw S2 evidence is under
  `test_artifacts/llm_debug/qq_group_public_scene/captured_s2_public_boundary/`.

### Closure disposition

- Gate D is failed and incomplete. S2 remains an explicitly deferred production
  fix surface, so changing its semantic behavior would expand this plan beyond
  the approved S6-S8 implementation scope.
- Gate E also remains blocked because no valid pre-change/candidate p95
  foreground-latency comparison on identical captured inputs exists. Existing
  unrelated or differently shaped latency samples cannot substitute for that
  baseline.
- Earlier unresolved live residue, interleaved-recorder, P0 leakage, trace-handle,
  and scope-classification findings remain residual blockers unless a later
  authorized attempt produces complete superseding evidence.
- Closure requires an explicit scope/acceptance decision for the S2 failure (or
  a separately approved S2 plan), followed by a fresh ordered Gate D run,
  identical-input Gate E baseline/candidate measurements, and independent final
  sign-off. No current evidence supports marking this plan `completed`.

### User-authorized live-LLM acceptance update — 2026-08-13

The user explicitly revised the live-LLM pass threshold after the first Gate D
attempt stopped at S2. The authoritative rule is now the tolerant weak-model
threshold stated in Gate D: conflicts, severe degradation, and altered semantic
ownership fail; limited ungrounded facts or hallucination pass with documented
residual notes. This change supersedes the earlier automatic-failure treatment
of the S2 vague affirmative answer. The independent quality owner must
re-evaluate the preserved S2 evidence under this rule before resuming at case 3.
No source, prompt, deterministic test, fixture, route, call, retry, or context
contract is changed by this acceptance update.

## Final Owner Amendments And Closure Evidence — 2026-08-13

This section is the authoritative final amendment. It supersedes conflicting
scope, closed-enum, Gate D, Gate E, and closure statements above while retaining
the earlier text as historical execution evidence.

### Accepted final scope and contract variances

The owner explicitly accepted the independent review findings and includes the
following existing final-diff surfaces in this plan:

- `continuity_boundary.boundary` additionally permits `post_turn`, because the
  post-turn wrapper is an instrumented continuity reconciliation boundary.
- `continuity_boundary.status` additionally permits `empty`, which distinguishes
  a successfully inspected boundary with zero projected rows from `skipped`.
- The bounded source-evidence wording in the existing dialog generator and
  semantic-fidelity prompts is included. It compacts the source-owned
  `evidence_state`, `evidence_excerpts`, `evidence_handles`,
  `prompt_safe_observation_handle`, and `remaining_needs` rules; it does not add
  a route, model call, retry, output field, or scenario-specific instruction.
- `state_projection.py` uses an operational entity salience floor of `25` for
  projecting already-committed active causal entities. This aligns visibility
  with the native retention floor and is accepted as a global projection
  behavior change.
- `character_carryover.py` normalizes accepted evidence and reducer timestamps
  to canonical UTC-Z text before native state application. Malformed timestamps
  remain excluded or fail the bounded reduction path. This is accepted as part
  of the final state-continuity contract.

These amendments expand the final implementation-owner surface to those exact
files and behaviors. They do not authorize further dialog, global projection,
carryover, event-family, or compatibility work.

### Gate D final disposition

The fresh final-source V3 ledger completed all 39 ordered live cases. The
independent reviewer accepted 29 cases directly, and the owner explicitly
accepted 10 cases after reviewing their exact inputs, visible outputs, admitted
goals or typed persistence/state results, and the reviewer's concern. There are
zero unrun or pending cases. The authoritative evidence index is:

`test_artifacts/llm_debug/group_topic_continuity_gate_d_final_v3/gate_d_final_v3_evidence_index.md`

The user-accepted cases are 6, 8, 11, 18, 21, 27, 29, 37, 38, and 39. Their
original harness failures or independent concerns remain in the case reviews;
acceptance does not erase them. Material residuals include weak-model factual or
participant drift, `replace_scope` in the related-continuation case, private
literal retention in an expiring user scope, lossy progress fields, a language
preference violation, diagnostic-capsule retention, and `route_invalid`
short-horizon receipts without a committed state transition.

The owner's final live-quality bottom line is participant/topic coherence:
limited hallucination, harshness, lossy typed detail, or temporary weak-model
interpretation can pass when participant-facing responses remain coherent and
do not make a drastic contradictory topic jump. A reviewer-recommended pass no
longer requires a separate owner pause; reviewer-recommended failures were
presented with exact evidence and decided by the owner.

### Gate E owner waiver

The owner explicitly waived Gate E and directed that latency evaluation be
skipped because it is not useful for this closure. Gate E is therefore recorded
as **waived**, not passed. No pre-change/candidate p95 comparison was performed,
and this plan makes no claim that foreground latency is unchanged or
non-regressive. The configured route context (`50176` observed), goal prompt cap
(`36000`), route identity, and existing call/retry contracts were checked in
the live preflight and deterministic contracts, but they are not a substitute
for a latency comparison.

### Final deterministic evidence

- The plan's 76 exact deterministic nodes collected and passed with zero skips.
- The expanded changed deterministic surface completed with `662 passed`, two
  optional inventory skips, and `98 deselected` live cases.
- The final mapped impact run passed all 93 exact nodes.
- The final complete impact mapping collected all 183 exact nodes.
- Focused final goal/prompt verification passed 83 tests; composed prompts
  remained below `GOAL_COGNITION_PROMPT_CAP=36000`.
- Compilation and `git diff --check` passed; reported line-ending notices do not
  represent diff errors.
- Runtime and harness anti-cheating scans found no captured scenario IDs,
  fixture-answer vocabulary, topic keyword gate, ordered answer template, or
  canned visible response in the final prompt/harness changes.

### Closure rule

With this explicit amendment, scope classification and closed-enum variances
are resolved by owner authority. Gates A-D are complete under the documented
acceptance decisions, Gate E is owner-waived, and no latency-regression claim is
permitted. Final completion and archive still require an independent reviewer
to confirm that this amendment resolves the previously reported blockers and
that no new blocker exists.

### Independent final sign-off

The independent final reviewer re-read the amended contract and final diff and
returned **PASS**. The reviewer confirmed that the owner amendment resolves the
closed-enum and scope findings, that no production source changed after the V3
Gate D run, that all 39 live cases have final dispositions, and that the final
deterministic and anti-cheating evidence matches the shipped source. No new
blocker was found.

This plan is complete and archived. Gates A-D are complete under the recorded
owner decisions; Gate E is explicitly waived. The closure makes no foreground
latency or latency-regression claim. All serious residuals and superseded raw
evidence remain linked from the final V3 evidence index.
