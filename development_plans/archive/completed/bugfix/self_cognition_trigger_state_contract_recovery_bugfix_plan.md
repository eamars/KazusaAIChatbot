# self-cognition trigger state-contract recovery bugfix plan

## Summary

- Goal: restore the group-chat and commitment self-cognition trigger paths on
  the current V2 branch, from source selection through shared cognition and
  the existing delivery boundary.
- Status: completed.
- Scope boundary: self-cognition source projection and runner-owned V2 state
  preparation; reflection remains the group cadence owner and calendar remains
  the commitment cadence owner.
- Change direction: upgrade self-cognition inputs to the current V2 contract
  at the source/runner boundary and provide the canonical immutable
  interaction-style snapshot before Cognition Core V2 runs.
- Acceptance state: implementation, deterministic verification, and the fresh
  guarded reflection-to-self-cognition group-ledger gate are complete.

## Context And Evidence

The read-only database diagnosis is recorded in
[`test_artifacts/diagnostics/self_cognition_trigger_review.md`](../../../../test_artifacts/diagnostics/self_cognition_trigger_review.md).
It was collected on branch `main`, HEAD `2526803c`, against the production
MongoDB without writes.

The evidence establishes that the trigger sources still select cases, while
the current runner fails at the V2 state boundary:

| Source | Evidence | Current failure boundary |
| --- | --- | --- |
| Group chat | 72-hour exports contain 500 rows in QQ group `638473184` and 222 rows in `54369546`; recent review ledgers contain concrete `group_activity_window:*` case ids. | The latest 300 review-window rows contain 158 `review_failed`, 134 `coalesced_skipped`, and no `reviewed`; runtime errors include 221 `KeyError: 'current_thread'` rows and 78 `interaction style turn snapshot is required` rows. |
| Commitment | 20 `commitment_due_cognition` runs include one completed run and one failed run; the scheduler-to-case handoff has worked. | The failed run records non-retryable `failure_summary.error='public_group_scene'`; four currently exported active commitments all have `due_at=null`, so none is currently due. |
| Historical delivery | One Aug 2 commitment run reached `self_cognition_processed` and selected an action candidate. | Its separate outbound attempt ended `delivery_failed` with `adapter_send_failed:ValueError`; this plan verifies that boundary after state recovery but does not change adapter delivery. |

The code correlation is:

1. `reflection_cycle.worker._run_group_self_cognition_review_for_scope` owns
   group-window selection and passes a concrete case to the normal worker.
2. `self_cognition.sources._build_group_review_case` supplies only a legacy
   partial `conversation_progress` mapping, while
   `project_conversation_progress_scene` directly requires V2 fields such as
   `current_thread`.
3. `self_cognition.runner._build_cognition_state` omits both
   `public_group_scene` and `interaction_style_context`.
4. Current Cognition Core V2 correctly fails closed when those required
   projections are missing. Commit `06fd4622` made the group style snapshot
   mandatory; restoring the removed cognition-side fallback would hide the
   caller contract defect.

The existing deterministic source, calendar, and reflection seam tests pass,
but they do not run the default self-cognition runner against the V2 state
contract. The plan adds that missing cross-boundary proof.

## Scope And Change Direction

The implementation will make the following single cutover:

- Source collectors keep `conversation_progress` for genuine current-user
  participant continuity only. Group review and scheduled-future graph states
  use `conversation_progress=None`; their semantic window and continuation
  data move into a typed `SelfCognitionCase.source_context` union owned by the
  self-cognition source boundary.
- `self_cognition.projection` renders `source_context` and visible rows through
  explicit prompt-safe allowlists. It excludes V2 scaffolding, row ids,
  platform/database ids, delivery metadata, and scheduler metadata while
  preserving the existing semantic digest, participant, thread, evidence, and
  continuation content.
- The self-cognition runner performs one deterministic preparation step before
  the shared cognition client. It always places `public_group_scene` in the
  graph state, using the canonical group-scene projection for a group review
  window and an empty string when no current public group scene exists. It
  loads one `interaction_style_turn_snapshot.v1` through
  `db.build_interaction_style_context` for every private/group target that can
  reach text-surface rendering and for every targetless group self-cognition
  case.
- The immutable style snapshot remains the only carrier for
  `group_engagement_action_context`; cognition continues to derive the
  bounded advisory projection from that snapshot. LLM stages retain semantic
  ownership of stance, reason to speak, action selection, and wording.
- The group review worker continues to select one fresh window and the
  commitment path continues to claim and re-read due calendar runs. No second
  group collector, raw active-commitment polling path, scheduler trigger kind,
  or adapter shortcut is introduced.

This is a big-bang runtime-contract cutover. The removed partial progress
shape and missing-state path are replaced directly; no compatibility mapper,
legacy fallback, cognition-side style fallback, retry loop, or feature flag is
added.

## Mandatory Skills

- `development-plan`: governs this plan's lifecycle, review, approval, and
  execution handoff.
- `local-llm-architecture`: applies before changing prompt-facing context,
  cognition state preparation, or shared V2 stage inputs.
- `py-style`: applies before editing Python source.
- `cjk-safety`: applies before editing Python files containing Chinese text.
- `test-style-and-execution`: applies before adding or running tests; live LLM
  cases run one at a time with their output inspected.
- `debug-llm`: applies to the post-fix live cognition smoke and its
  human-readable evidence artifact.
- `database-data-pull`: applies to the read-only post-fix MongoDB evidence
  pull; no database write is part of this plan.

## Mandatory Rules

- Do not edit production source until the plan is approved or in progress and
  the user explicitly starts implementation.
- Capture `git status --short`, the owned file set, and the clean execution
  baseline before the first implementation edit.
- Use `venv\Scripts\python` for Python commands and `apply_patch` for manual
  edits. Do not read `.env`.
- Preserve the current Cognition Core V2 fail-fast checks. Required fields are
  prepared by the owning caller; they are not filled with semantic defaults in
  `persona_supervisor2_cognition.py`.
- Keep raw platform ids, database ids, delivery targets, scheduler ids, and
  source row ids out of model-facing progress and scene projections.
- Keep targetless group review targetless: `global_user_id`,
  `platform_user_id`, and addressed-user ids remain empty.
- Treat RAG as evidence only. It does not synthesize the current group scene,
  choose a route, or decide whether the character should speak.
- Do not modify historical failed ledger rows or calendar runs as part of the
  code fix. Fresh post-fix windows and isolated deterministic due-run fixtures
  provide acceptance evidence.

## Must Do

- Replace the group-review partial progress mapping with typed source context;
  retain existing bounded participant, digest, thread, and
  conversation-evidence source hydration outside participant continuity.
- Replace the scheduled-future legacy progress carrier with typed source
  context containing only the semantic continuation objective and safe mode.
- Add runner-owned preparation for `public_group_scene` and the immutable
  interaction-style snapshot before the default cognition path.
- Exercise a real group-review case and a commitment case through the default
  state-to-cognition-input boundary with deterministic LLM seams.
- Preserve current reflection group cadence, calendar due-run ownership,
  target binding, action-attempt idempotency, consolidation, and dispatcher
  ownership.
- Update the self-cognition ICD to document the V2 state-preparation contract
  and the fact that group engagement guidance comes from the style snapshot.
- Verify fresh group-window completion and a due commitment run after the fix;
  record any adapter delivery failure as a separate residual.

## Deferred

- Preserve Cognition Core V2 prompt topology, projection strictness, prompts,
  semantic appraisal, goal selection, action authorization, L3 wording, and
  dialog verification. A narrow provenance correction classifies the existing
  `scheduler_event` trigger as current-episode evidence; it does not change a
  semantic contract or prompt topology and is covered by a focused unit test.
- The independent reflection recursive-root metadata correction is tracked in
  the linked
  [`reflection_recursive_root_timestamp_canonicalization_bugfix_plan.md`](reflection_recursive_root_timestamp_canonicalization_bugfix_plan.md).
  This plan consumes its guarded group-ledger evidence without changing the
  self-cognition ownership boundary or rewriting historical production rows.
- No automatic replay, repair, or deletion of historical `review_failed`,
  `coalesced_skipped`, or failed calendar rows.
- No adapter implementation, send retry, delivery fallback, or private-channel
  lookup change.
- No live-LLM prompt-quality tuning. The post-fix live smoke proves that the
  repaired state reaches the existing model path; semantic quality changes
  require a separate plan.

## Target State

### Source case contract

`SelfCognitionCase.conversation_progress`, when present, is a genuine
`conversation_progress_prompt.v2` mapping for current-user continuity. Group
review and scheduled-future cases set it to `None`; neither source is a
participant-continuity episode.

`SelfCognitionCase.source_context` is nullable at the case boundary. It is
required for group-review and scheduled-future cases and is one of these exact
prompt-safe shapes:

```python
{
    "schema_version": "self_cognition_group_source_context.v1",
    "context_kind": "group_chat_review",
    "group_activity_window": {
        "source": str,
        "window_start": str,
        "window_end": str,
        "semantic_labels": dict[str, str],
    },
    "participant_context": dict | None,
    "thread_reference_context": dict | None,
    "group_scene_digest": {"digest": str, "summary": str} | None,
    "conversation_evidence": list[str],
}
```

```python
{
    "schema_version": "self_cognition_scheduled_source_context.v1",
    "context_kind": "scheduled_future_cognition",
    "continuation_objective": str,
    "continuation_mode": str,
}
```

Optional group fields are omitted when their source builder returns no valid
value. The bounded case `visible_context` remains source evidence, but the
model-facing projection keeps only `role`, `display_name`, localized
`timestamp`, and `body_text`.

### Prepared cognition state

Before `_default_cognition_client` runs, the runner passes a state containing:

- `conversation_progress`: `None` for group-review and scheduled-future
  source cases; a canonical V2 prompt mapping only for a real current-user
  continuity source;
- `source_context`: the typed group or scheduled source context, carried into
  the source packet but never into participant continuity;
- `public_group_scene`: a bounded string, always present; empty only when no
  current public group scene exists;
- `interaction_style_context`: the exact
  `interaction_style_turn_snapshot.v1` from
  `build_interaction_style_context` for send-capable private/group targets and
  targetless group review;
- existing episode, source, target, residue, action, and history fields
  unchanged.

After successful target binding, the runner loads exactly one style snapshot
for each bound private/group case that can reach a text surface and exactly
one for each targetless group review. For a targetless group review, the style
snapshot is loaded with an empty user id and the current group scope. Cognition
and L3 receive the same snapshot value/object. Cognition receives the resulting
`group_engagement_action_context` only through the snapshot. The group-scene
string is built from chronological prompt-safe `visible_context` rows: the
newest valid row is the structural trigger and earlier rows are ambient. The
runner supplies empty identity, address, reply, and roster fields, creates no
additional semantic message, and calls `build_group_scene_context` followed by
`project_group_scene_prompt`. The resulting string contains no ids and is not
copied into `conversation_progress`.

## Contracts And Data Shapes

- `source_context` is the sole source-owned carrier for group-window metadata
  and scheduled continuation semantics. Its projection is explicit and
  prompt-safe.
- `public_group_scene` is a caller-owned prompt projection, not persisted
  character state and not RAG evidence.
- `interaction_style_context` is immutable per self-cognition case and is
  reused by Cognition V2 and L3 in that case.
- A missing, wrong-schema, or malformed style snapshot field is a typed
  state-preparation failure recorded by the existing per-case worker failure
  path. Valid snapshots whose individual style sources have `status="missing"`
  or `status="failed"` remain usable immutable snapshots with empty overlays.
  The runner does not turn those valid source statuses into a cognition-side
  fallback.
- Group engagement guidance remains advisory. It cannot create a topic,
  permission, route, response ratio, or deterministic silence rule.
- Commitment due handling remains `calendar_runs` -> re-read memory unit ->
  self-cognition case -> shared cognition. The memory unit is still the
  authority at execution time.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| Self-cognition source context | bigbang | Replace partial/legacy group and scheduled carriers with the typed `source_context` union; reserve `conversation_progress` for real participant continuity. |
| Runner state | bigbang | Prepare the complete caller-owned V2 state before cognition. |
| Cognition Core V2 | preserve | Keep the existing strict consumer contract unchanged. |
| Persisted MongoDB state | read-only verification | Do not migrate, replay, or mutate historical rows. |
| Adapter delivery | preserve | Verify the existing handoff and report separate failures. |
| Tests | bigbang | Rewrite assertions for the V2 source/state contract and add direct owner tests. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/self_cognition/sources.py`
  - Add the typed group/scheduled `source_context` shapes and populate them
    from existing bounded source data.
  - Set group-review and scheduled-future `conversation_progress` to `None`;
    preserve current-user continuity only where it genuinely exists.
  - Keep calendar claims, memory-unit re-reads, source refs, visible context,
    target binding, and group source ownership unchanged.
- `src/kazusa_ai_chatbot/self_cognition/models.py`
  - Define the `SelfCognitionSourceContext` union and carry it through case and
    source-packet contracts.
- `src/kazusa_ai_chatbot/self_cognition/projection.py`
  - Project `source_context` and visible rows through explicit allowlists.
  - Remove legacy reads from `conversation_progress` for group metadata and
    preserve scheduled forbidden-fragment checks.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Add the runner-owned asynchronous preparation step, exact chronological
    group-scene mapping, style-snapshot load, and complete state handoff.
  - Keep `_default_cognition_client` and Cognition Core V2 invocation semantics
    unchanged.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
  - Include the already-emitted scheduler trigger evidence in the current
    episode provenance allowlist so a due commitment can complete the existing
    relational validation path.
- `src/kazusa_ai_chatbot/reflection_cycle/repository.py` *(linked
  prerequisite; see the separate plan)*
  - Canonicalize repeated settled roots by earliest normalized `captured_at`
    while retaining fail-closed checks for correlation, local-date, and scope
    metadata conflicts.
- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Document the required V2 progress, public-scene, and interaction-style
    inputs for self-cognition cases.
- `tests/test_self_cognition_group_review_source.py`
  - Add direct source-context, allowlist, group-scene, and prepared-state
    regressions; update targetless-group assertions to use the prepared-state
    seam.
- `tests/test_self_cognition_integration.py`
  - Add scheduled source-context and commitment-state regressions; verify the
    default worker path no longer records the observed state-contract errors.
- `tests/test_reflection_cycle_stage1c_worker.py`
  - Preserve the cross-boundary assertion that a reflection-selected group
    case reaches the self-cognition runner and records its selected window.
- `tests/test_reflection_cycle_stage1c_repository.py` *(linked prerequisite;
  see the separate plan)*
  - Verify timestamp-only duplicate roots collapse to the earliest timestamp,
    distinct roots remain distinct, and identity metadata conflicts fail closed.
- `tests/test_cognition_core_v2_p0_context_reconnection_live_llm.py`
  - Migrate the direct group self-cognition fixture to the prepared state and
    injected immutable style snapshot.
- `tests/test_l2d_action_selection_live_llm.py`
  - Migrate the direct self-cognition fixture away from the legacy partial
    progress mapping.
- `tests/test_self_cognition_response_sensitivity_live_llm.py`
  - Migrate group source-context fixtures to the typed source-context lane.
- `tests/test_self_cognition_duplicate_response_live_llm.py`
  - Migrate duplicate-response group fixtures to the typed source-context
    lane.
- `tests/test_stage3_fresh_database_e2e_live_llm.py` and
  `tests/cognition_baseline_worker.py`
  - Migrate retained scheduled/group direct callers to the typed source
    context and `conversation_progress=None` contract.
- `tests/unit/cognition_core_v2/test_contracts.py`
  - Add the focused scheduler-event provenance regression.

### Create

- `tests/test_self_cognition_trigger_state_live_llm.py`
  - Add one DB-free real-LLM group state-contract smoke and one DB-free
    real-LLM commitment state-contract smoke. Each case uses an injected
    immutable style snapshot and patched persistence seams.

### Delete

- Delete legacy group metadata reads from `conversation_progress`, the legacy
  scheduled-future progress carrier, and raw visible-row fields from the
  model-facing source packet. No source module or persisted collection is
  deleted.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` strict
  `current_thread`, `public_group_scene`, and style-snapshot checks.
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py` group selection,
  coalescing, ledger, and same-scope coordination ownership; its propagation
  test remains a preserved cross-boundary contract.
- `src/kazusa_ai_chatbot/calendar_scheduler/` commitment trigger ownership,
  leases, stale-run handling, and terminal transitions.
- `src/kazusa_ai_chatbot/dispatcher/` delivery validation and adapter bridge.

## Test Impact And Traceability

The following matrix is part of the execution contract. New node names are
fixed implementation targets and must be collected exactly as written. The
production rows identify the semantic owner of each changed contract. The
linked reflection prerequisite owns its repository normalization rows in its
own traceability matrix.

| Source or governed artifact | Changed symbol/contract | Semantic owner | Exact deterministic pytest node(s) | Supplemental nodes | Mode | Regression prevented |
| --- | --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/self_cognition/models.py` | `SelfCognitionSourceContext` union and case/source-packet field contract | self-cognition case model | `tests/test_self_cognition_group_review_source.py::test_source_context_contract_accepts_group_and_scheduled_shapes` | `tests/test_self_cognition_integration.py::test_collect_commitment_due_cognition_cases_projects_calendar_runs` | deterministic unit | Prevents untyped legacy metadata from crossing the source-to-runner boundary. |
| `src/kazusa_ai_chatbot/self_cognition/projection.py` | `_visible_context` and source-context prompt projection allowlists | self-cognition model-facing projection | `tests/test_self_cognition_group_review_source.py::test_source_packet_allowlist_strips_ids_and_v2_scaffolding` | `tests/test_self_cognition_group_review_source.py::test_group_review_source_packet_uses_active_group_review_contract` | deterministic unit/source seam | Prevents platform/database ids, delivery metadata, scheduler metadata, and V2 scaffold fields from reaching the model. |
| `src/kazusa_ai_chatbot/self_cognition/sources.py` | `_build_group_review_case` and `_build_scheduled_future_cognition_case` typed source-context construction | self-cognition source collector | `tests/test_self_cognition_group_review_source.py::test_group_review_case_projects_typed_source_context`; `tests/test_self_cognition_integration.py::test_collect_scheduled_future_cognition_cases_projects_typed_source_context` | `tests/test_self_cognition_integration.py::test_collect_commitment_due_cognition_cases_projects_calendar_runs` | deterministic unit/source seam | Prevents `KeyError: 'current_thread'` from partial group progress and preserves scheduled continuation semantics without misusing participant continuity. |
| `src/kazusa_ai_chatbot/self_cognition/runner.py` | prepared `public_group_scene`, one immutable `interaction_style_context`, and source-specific progress handoff | self-cognition runner/state boundary | `tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_contains_v2_scene_and_style_contract`; `tests/test_self_cognition_integration.py::test_prepared_commitment_state_contains_public_group_scene` | `tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_builds_v2_cognition_input` | deterministic unit/cross-boundary seam | Prevents `interaction style turn snapshot is required` and `KeyError: 'public_group_scene'` in the default runner. |
| `src/kazusa_ai_chatbot/self_cognition/projection.py` | runtime validation of progress/source-context shapes before model projection | self-cognition source-contract boundary | `tests/test_self_cognition_group_review_source.py::test_source_context_validator_rejects_wrong_group_schema` | `tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_rejects_malformed_source_context` | deterministic contract test | Prevents malformed typed source state from reaching V2 as an untyped or partial mapping. |
| `src/kazusa_ai_chatbot/self_cognition/runner.py` | chronological newest-row group-scene construction | self-cognition runner/state boundary | `tests/test_self_cognition_group_review_source.py::test_group_scene_uses_chronological_newest_row_as_trigger` | `tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_reaches_strict_v2_connector` | deterministic/cross-boundary seam | Prevents stale row ordering from becoming the structural group trigger. |
| `src/kazusa_ai_chatbot/self_cognition/runner.py` | immutable style snapshot identity across Cognition and L3 | self-cognition runner/state boundary | `tests/test_self_cognition_group_review_source.py::test_prepared_style_snapshot_is_reused_by_dialog_handoff` | `tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_reaches_strict_v2_connector` | deterministic cross-boundary seam | Prevents Cognition and L3 from receiving different style snapshots. |
| `src/kazusa_ai_chatbot/self_cognition/runner.py` | malformed style-snapshot failure and unavailable-source status handling | self-cognition style-state validator | `tests/test_self_cognition_group_review_source.py::test_interaction_style_snapshot_rejects_bad_contract`; `tests/test_self_cognition_group_review_source.py::test_interaction_style_snapshot_allows_unavailable_source_status` | `tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_contains_v2_scene_and_style_contract` | deterministic contract test | Prevents invalid snapshots from reaching V2 while preserving usable `missing` and `failed` source overlays. |
| `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` | `scheduler_event` current-episode evidence classification | Cognition Core V2 provenance contract | `tests/unit/cognition_core_v2/test_contracts.py::test_scheduler_events_are_current_episode_evidence` | `tests/test_self_cognition_integration.py::test_prepared_commitment_state_reaches_strict_v2_input` | deterministic unit/cross-boundary seam | Prevents a valid commitment-due trigger from failing relational evidence validation. |
| `src/kazusa_ai_chatbot/self_cognition/runner.py` | default worker preparation and cognition-input construction for a repaired case | self-cognition worker-to-cognition boundary | `tests/test_self_cognition_integration.py::test_worker_default_path_runs_prepared_case_without_state_contract_error` | `tests/test_reflection_cycle_stage1c_worker.py::test_group_review_passes_adapter_registry_provider_to_self_cognition` | deterministic integration seam | Prevents a source case from failing after selection because the normal worker path receives an incomplete V2 state. |
| `src/kazusa_ai_chatbot/reflection_cycle/worker.py` *(preserved cross-boundary contract; repository normalization is owned by the linked plan)* | selected group case propagation and terminal ledger status | reflection group-review orchestrator | none | `tests/test_reflection_cycle_stage1c_worker.py::test_group_review_passes_adapter_registry_provider_to_self_cognition`; `tests/test_reflection_cycle_stage1c_worker.py::test_group_review_records_target_binding_failed_terminal_row` | deterministic integration seam | Prevents a repaired case from being constructed but not propagated or acknowledged by the group ledger. |
| `src/kazusa_ai_chatbot/self_cognition/README.md` | self-cognition V2 state-preparation ICD | self-cognition contract owner | `tests/test_self_cognition_group_review_source.py::test_self_cognition_readme_documents_v2_state_contract` | none | deterministic static/document contract | Prevents documentation from reintroducing a legacy state shape or cognition-side fallback. |

Every changed semantic production owner has a direct deterministic owner node.
The live database and live-LLM checks below supplement this matrix and do not
replace the owner nodes. The existing direct live callers that construct
self-cognition state are migration targets and are listed separately so they
cannot silently keep the legacy partial mapping.

### Live and fixture traceability

| Exact pytest node | Purpose | Mode | Evidence required |
| --- | --- | --- | --- |
| `tests/test_cognition_core_v2_p0_context_reconnection_live_llm.py::test_live_group_self_cognition_uses_one_advisory_projection` | Migrate the direct group fixture to the prepared state and injected immutable style snapshot. | live LLM, one case | Human inspection of the debug artifact; no database writes. |
| `tests/test_l2d_action_selection_live_llm.py::test_l2d_live_routes_real_active_commitment_lifecycle_update` | Migrate the direct commitment fixture away from the legacy partial progress mapping. | live LLM, one case | Action/route evidence remains separate from delivery evidence. |
| `tests/test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_group_response_sensitivity` | Verify group response-sensitivity behavior receives the repaired state. | live LLM, one case | Output and trace inspected for state-contract errors. |
| `tests/test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_ambient_group_l2d_does_not_speak` | Preserve the ambient-group boundary with the repaired scene and style inputs. | live LLM, one case | Silence is judged from character reason and trace, not a deterministic gate. |
| `tests/test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_cat_side_thread_subject_boundary` | Preserve side-thread subject-boundary behavior after source-context migration. | live LLM, one case | Trace and model-facing projection inspected. |
| `tests/test_self_cognition_duplicate_response_live_llm.py::test_live_sc_duplicate_response_window_with_own_reply` | Preserve duplicate-response handling with the typed group source context. | live LLM, one case | No duplicate action claim without persistence evidence. |
| `tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_scheduled_tick_source` | Verify the retained scheduled direct caller uses the typed scheduled source context. | live LLM/DB, one case | Environment-gated output and state-contract result. |
| `tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_group_review_promoted_reflection` | Verify the retained reflection-selected direct caller uses the typed group source context. | live LLM/DB, one case | Environment-gated output and state-contract result. |
| `tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_group_review_worker_records_reviewed_ledger` | Verify the real reflection group worker deduplicates repeated settled roots and records a fresh reviewed ledger row after self-cognition. | live LLM/DB, one case | Guarded artifact plus independent reflection and ledger exports. |
| `tests/test_self_cognition_trigger_state_live_llm.py::test_live_group_review_state_contract_reaches_model` | Prove the group state contract reaches the existing real model path. | DB-free live LLM, one case | `test_artifacts/diagnostics/self_cognition_trigger_state_live_group.json` and a human-readable review artifact. |
| `tests/test_self_cognition_trigger_state_live_llm.py::test_live_commitment_state_contract_reaches_model` | Prove the commitment state contract reaches the existing real model path. | DB-free live LLM, one case | `test_artifacts/diagnostics/self_cognition_trigger_state_live_commitment.json` and a human-readable review artifact. |

The new live smoke injects a fixed immutable style snapshot and patches only
persistence/delivery seams needed to keep the test database-free. It does not
replace the deterministic due-run fixture or the read-only production
observation.

## Agent Autonomy Boundaries

The implementation owner may choose local helper placement, async sequencing,
and bounded test fixtures within the listed files while preserving the fixed
state shapes and ownership. The implementation owner must not change V2
consumer contracts, add defaults for required semantic fields, move group or
commitment cadence, introduce a compatibility layer, or alter delivery policy.

If the current source data cannot be projected into the stated prompt-safe
shape without exposing ids or fabricating a scene, execution pauses for a plan
amendment. A failing live LLM quality result is recorded as evidence and does
not authorize prompt or semantic-policy changes under this plan.

## Verification

### Baseline and exact deterministic collection

Before implementation, capture the branch, commit, clean/dirty state, and
explicitly owned file set. After implementation, collect every fixed owner and
supplemental node below; a missing node is a verification failure.

```powershell
git status --short
git branch --show-current
git rev-parse --short HEAD
venv\Scripts\python -m pytest --collect-only -q `
  tests/test_self_cognition_group_review_source.py::test_source_context_contract_accepts_group_and_scheduled_shapes `
  tests/test_self_cognition_group_review_source.py::test_source_packet_allowlist_strips_ids_and_v2_scaffolding `
  tests/test_self_cognition_group_review_source.py::test_group_review_case_projects_typed_source_context `
  tests/test_self_cognition_integration.py::test_collect_scheduled_future_cognition_cases_projects_typed_source_context `
  tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_contains_v2_scene_and_style_contract `
  tests/test_self_cognition_integration.py::test_prepared_commitment_state_contains_public_group_scene `
  tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_builds_v2_cognition_input `
  tests/test_self_cognition_integration.py::test_worker_default_path_runs_prepared_case_without_state_contract_error `
  tests/test_reflection_cycle_stage1c_worker.py::test_group_review_passes_adapter_registry_provider_to_self_cognition `
  tests/test_reflection_cycle_stage1c_worker.py::test_group_review_records_target_binding_failed_terminal_row `
  tests/test_self_cognition_group_review_source.py::test_group_review_source_packet_uses_active_group_review_contract `
  tests/test_self_cognition_integration.py::test_collect_commitment_due_cognition_cases_projects_calendar_runs `
  tests/test_self_cognition_group_review_source.py::test_self_cognition_readme_documents_v2_state_contract `
  tests/test_self_cognition_group_review_source.py::test_source_context_validator_rejects_wrong_group_schema `
  tests/test_self_cognition_group_review_source.py::test_group_scene_uses_chronological_newest_row_as_trigger `
  tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_rejects_malformed_source_context `
  tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_reaches_strict_v2_connector `
  tests/test_self_cognition_group_review_source.py::test_prepared_style_snapshot_is_reused_by_dialog_handoff `
  tests/test_self_cognition_group_review_source.py::test_interaction_style_snapshot_rejects_bad_contract `
  tests/test_self_cognition_group_review_source.py::test_interaction_style_snapshot_allows_unavailable_source_status `
  tests/test_self_cognition_integration.py::test_prepared_commitment_state_reaches_strict_v2_input `
  tests/unit/cognition_core_v2/test_contracts.py::test_scheduler_events_are_current_episode_evidence
venv\Scripts\python -m pytest `
  tests/test_self_cognition_group_review_source.py::test_source_context_contract_accepts_group_and_scheduled_shapes `
  tests/test_self_cognition_group_review_source.py::test_source_packet_allowlist_strips_ids_and_v2_scaffolding `
  tests/test_self_cognition_group_review_source.py::test_group_review_case_projects_typed_source_context `
  tests/test_self_cognition_integration.py::test_collect_scheduled_future_cognition_cases_projects_typed_source_context `
  tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_contains_v2_scene_and_style_contract `
  tests/test_self_cognition_integration.py::test_prepared_commitment_state_contains_public_group_scene `
  tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_builds_v2_cognition_input `
  tests/test_self_cognition_integration.py::test_worker_default_path_runs_prepared_case_without_state_contract_error `
  tests/test_reflection_cycle_stage1c_worker.py::test_group_review_passes_adapter_registry_provider_to_self_cognition `
  tests/test_reflection_cycle_stage1c_worker.py::test_group_review_records_target_binding_failed_terminal_row `
  tests/test_self_cognition_group_review_source.py::test_group_review_source_packet_uses_active_group_review_contract `
  tests/test_self_cognition_integration.py::test_collect_commitment_due_cognition_cases_projects_calendar_runs `
  tests/test_self_cognition_group_review_source.py::test_self_cognition_readme_documents_v2_state_contract `
  tests/test_self_cognition_group_review_source.py::test_source_context_validator_rejects_wrong_group_schema `
  tests/test_self_cognition_group_review_source.py::test_group_scene_uses_chronological_newest_row_as_trigger `
  tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_rejects_malformed_source_context `
  tests/test_self_cognition_group_review_source.py::test_prepared_group_review_state_reaches_strict_v2_connector `
  tests/test_self_cognition_group_review_source.py::test_prepared_style_snapshot_is_reused_by_dialog_handoff `
  tests/test_self_cognition_group_review_source.py::test_interaction_style_snapshot_rejects_bad_contract `
  tests/test_self_cognition_group_review_source.py::test_interaction_style_snapshot_allows_unavailable_source_status `
  tests/test_self_cognition_integration.py::test_prepared_commitment_state_reaches_strict_v2_input `
  tests/unit/cognition_core_v2/test_contracts.py::test_scheduler_events_are_current_episode_evidence -q
```

### Adjacent deterministic regression

Run the existing source, worker, reflection, calendar, event, and delivery
boundaries after the fixed nodes pass:

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py tests\test_self_cognition_integration.py tests\test_reflection_cycle_activity_windows.py tests\test_reflection_cycle_stage1c_worker.py tests\test_calendar_scheduler_active_commitments.py -q
venv\Scripts\python -m pytest tests\test_self_cognition_event_logging.py tests\test_self_cognition_delivery_target.py -q
```

Run `git diff --check`. If a source path is added to
`tests/ownership/source_test_impact_manifest.json`, run the repository's
required impact validation and preserve its output with the execution
artifacts.

### DB-free live LLM smoke

Apply `debug-llm` and `test-style-and-execution`. Run one live case per
process, inspect the output and trace before starting the next case, and keep
the smoke bounded to the existing model path. The two new state-contract
smokes are required:

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_trigger_state_live_llm.py::test_live_group_review_state_contract_reaches_model -q
venv\Scripts\python -m pytest tests\test_self_cognition_trigger_state_live_llm.py::test_live_commitment_state_contract_reaches_model -q
```

The live fixture injects an immutable `interaction_style_turn_snapshot.v1`,
patches persistence and delivery seams, performs no MongoDB writes, and emits
these artifacts:

- `test_artifacts/diagnostics/self_cognition_trigger_state_live_group.json`
- `test_artifacts/diagnostics/self_cognition_trigger_state_live_commitment.json`
- `test_artifacts/diagnostics/self_cognition_trigger_state_live_review.md`

Then run each migrated existing live caller separately, with output inspected
after each process:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_core_v2_p0_context_reconnection_live_llm.py::test_live_group_self_cognition_uses_one_advisory_projection -q
venv\Scripts\python -m pytest tests\test_l2d_action_selection_live_llm.py::test_l2d_live_routes_real_active_commitment_lifecycle_update -q
venv\Scripts\python -m pytest tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_group_response_sensitivity -q
venv\Scripts\python -m pytest tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_ambient_group_l2d_does_not_speak -q
venv\Scripts\python -m pytest tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_cat_side_thread_subject_boundary -q
venv\Scripts\python -m pytest tests\test_self_cognition_duplicate_response_live_llm.py::test_live_sc_duplicate_response_window_with_own_reply -q
```

The live results prove state reachability and preserve character-quality
evidence; they do not authorize prompt tuning or deterministic response
suppression.

### Bounded read-only production observation

Use `database-data-pull` and the repository's protected trace procedure. This
gate has two independent parts:

1. Use a deterministic due-commitment fixture for the required commitment
   trigger proof. The fixture must claim a due calendar run, re-read its
   memory unit, and exercise the repaired runner without writing to production
   MongoDB.
2. Use a read-only production observation for current runtime health. Run the
   ops status pull once before the observation and once after one fresh group
   phase plus its following worker tick:

```powershell
venv\Scripts\python -m scripts.fetch_ops_status --hours 1 --json
venv\Scripts\python -m scripts.export_collection self_cognition_group_review_windows --filter "{\"channel_type\":\"group\"}" --sort "{\"reviewed_at\":-1}" --limit 50
venv\Scripts\python -m scripts.export_collection calendar_runs --filter "{\"run_type\":\"commitment_due_cognition\"}" --sort "{\"created_at\":-1}" --limit 50
venv\Scripts\python -m scripts.fetch_ops_status --hours 1 --json
```

The production observation is limited to 30 minutes, includes one fresh
group-review phase and the following worker tick, and performs no database
writes, replay, repair, or historical-row mutation. It must report whether a
fresh ledger row reaches `reviewed` or deliberate terminal
`target_binding_failed`, and whether any new state-contract error appears.
The current database snapshot has no due commitment; the observation therefore
does not wait indefinitely for a natural due event and cannot substitute for
the deterministic due fixture.

If cognition selects visible speech, inspect the protected trace, action
attempt, dispatcher receipt, and adapter result separately. An adapter failure
remains a delivery residual, not a cognition success claim. Preserve sanitized
JSON and a human-readable review artifact under
`test_artifacts/diagnostics/`; ordinary logs contain no raw ids or message
bodies.

### Static contract checks

Review the following searches after the tests pass:

```powershell
rg -n "conversation_progress" src\kazusa_ai_chatbot\self_cognition
rg -n "platform_message_id|database_message_id|scheduler_id|delivery_target" src\kazusa_ai_chatbot\self_cognition\projection.py
rg -n "current_thread|public_group_scene|interaction_style_context" src\kazusa_ai_chatbot\self_cognition\runner.py src\kazusa_ai_chatbot\self_cognition\projection.py
```

The first search must show `conversation_progress` used only for genuine
participant continuity or explicit contract tests. The second search must be
empty for model-facing projection code. The third search must show complete
runner preparation and no projection-side fallback. Re-run `git diff --check`
after the static review.

## Acceptance Criteria

The plan is complete only when all of the following are evidenced:

- Every exact owner node is collected and passes, followed by the adjacent
  deterministic suites.
- `SelfCognitionCase.source_context` validates the exact group and scheduled
  shapes, and group/scheduled cases carry `conversation_progress=None`.
- The model-facing source packet keeps only the declared semantic allowlists;
  no platform/database row id, delivery field, scheduler field, or V2
  scaffold reaches the model.
- The group scene uses the newest valid chronological visible row as the
  structural trigger, earlier rows as ambient context, canonical scene
  projection, empty identity/address/reply/roster fields, and no added
  semantic message or participant-continuity copy.
- One immutable style snapshot is prepared per case, reused by Cognition and
  L3, and valid source overlays with `status="missing"` or `status="failed"`
  remain usable. Missing, wrong-schema, or malformed snapshot state fails at
  preparation with a typed case error.
- A group-review case reaches `build_cognition_input_from_global_state`
  without `current_thread`, `public_group_scene`, or interaction-style
  contract errors.
- A deterministic due commitment case reaches the same boundary without
  `public_group_scene` failure, and its calendar run remains correctly
  completed, skipped, or failed according to the semantic outcome.
- The group review ledger records a fresh selected window as `reviewed` after
  a successful worker case; failed cases remain visibly failed rather than
  being marked reviewed.
- Targetless group cases preserve an empty semantic user target and carry the
  group engagement projection only from the canonical style snapshot.
- No post-fix self-cognition runtime error contains `KeyError: 'current_thread'`,
  `KeyError: 'public_group_scene'`, or `interaction style turn snapshot is
  required` for the exercised sources.
- The two DB-free real-LLM state-contract smokes complete one case at a time
  with inspected artifacts, and each migrated direct live caller uses the
  prepared state contract.
- The bounded read-only production observation reports the fresh group ledger
  outcome and post-fix runtime errors without writing or replaying any row.
- The live evidence distinguishes cognition completion, action selection,
  dispatcher handoff, and adapter delivery; no layer is declared healthy from
  an earlier layer's event alone.
- The independent plan review findings are resolved in this document, and any
  independent code review findings during execution are recorded and rerun.

## Implementation Order

1. Capture the execution baseline and confirm the explicitly owned file set.
2. Add the model/projection/source contract tests and update source
   constructors to emit the typed source-context union.
3. Add the runner preparation seam and state-field tests, including the
   canonical style-snapshot call and group-scene projection.
4. Update the self-cognition ICD, migrate all listed direct live fixtures, and
   complete the source, worker, and calendar deterministic regressions. Execute
   the linked reflection prerequisite before the guarded group-ledger run.
5. Run the exact owner-node collection, owner nodes, adjacent suites, and
   static checks.
6. Run the two DB-free live state-contract smokes and each migrated live caller
   one case at a time; preserve the evidence artifacts.
7. Run the bounded read-only production observation and the deterministic due
   commitment fixture; preserve separate trigger, cognition, action, and
   delivery evidence.
8. Perform independent code review, remediate only in-scope findings, rerun
   affected nodes, and update this plan's execution evidence.

## Progress Checklist

- [x] Draft created from current branch and database evidence.
- [x] Independent GPT-5.6 Sol plan review completed; verdict was `REVISE`.
- [x] Review findings incorporated; plan status is `approved` and execution is
  ready.
- [x] Implementation baseline captured by the execution owner.
- [x] Source, runner, projection, provenance, and fixture contract changes
  implemented.
- [x] Exact owner nodes and adjacent deterministic suites pass.
- [x] DB-free live LLM evidence and bounded read-only production evidence
  recorded.
- [x] Independent implementation and final review completed.
- [x] Linked reflection repository remediation and deterministic regressions
  implemented after the final-review blocker.
- [x] Fresh guarded reflection-to-self-cognition group-ledger outcome and full
  lifecycle closeout.

## Execution Evidence

Implementation was delegated to GPT-5.6 Luna (`max`, normal service speed)
with exclusive ownership of the self-cognition production and ICD files. The
parent maintained tests and review. GPT-5.6 Terra (`max`, normal service speed)
performed the independent implementation review and identified direct legacy
callers, missing runtime source-context validation, and the need for a strict
connector proof. Those findings were remediated in scope. The final GPT-5.6
Sol reviewer (`high`, normal service speed) confirmed the architecture and
returned `BLOCKED` only because the production closure gate and its fresh
ledger transition were not evidenced.

Final deterministic evidence:

- The fixed owner list collected exactly 23 nodes and passed: `23 passed in
  1.07s`.
- The adjacent source, integration, reflection, and calendar suites passed:
  `134 passed in 1.46s`.
- Event logging and delivery-target suites passed: `13 passed in 0.92s`.
- Python compilation and `git diff --check` passed.
- The two DB-free real-LLM trigger smokes passed individually. Both accepted
  `cognition_core_input.v2`, completed all seven cognition stages, and emitted
  no warnings. Their review is recorded in
  [`self_cognition_trigger_state_live_review.md`](../../../../test_artifacts/diagnostics/self_cognition_trigger_state_live_review.md).

Live caller dispositions:

- The P0 group replay reached setup but stopped on the existing immutable
  seed/profile conflict.
- The L2D commitment replay skipped because no production active commitment
  case was available.
- Group response sensitivity stopped because the guarded database had fewer
  than the required 20 production-derived cases; the ambient group case
  skipped because its production-derived window was unavailable.
- The cat side-thread replay reached the repaired cognition path and wrote a
  trace, but its separate semantic-quality assertion still found one invented
  subject phrase in the internal monologue. This remains outside the state
  contract fix.
- Duplicate-response replay skipped because its historical window was absent.
- Stage 3 scheduled and promoted-group callers skipped because the guarded
  environment requires `MONGODB_DB_NAME='_test_kazusa_core_v2'`.
- The retained baseline worker source compiles and its CLI contract was
  checked with `--help`; no baseline input fixture was available for execution.

Read-only production evidence is recorded in
[`self_cognition_trigger_postfix_observation_review.md`](../../../../test_artifacts/diagnostics/self_cognition_trigger_postfix_observation_review.md).
Self-cognition was enabled but inactive (`runs=0`, `dispatch_accepted=0`),
the current commitment export contained zero due runs, and no fresh
`reviewed` or `target_binding_failed` group row was created. The latest stored
state-contract error predates the observation. The separate recursive-root
metadata issue is now remediated by the linked reflection plan. The required
write-capable guarded test-database worker run completed against
`_test_kazusa_core_v2`, and
production mutation remains outside both plans. The guarded artifact shows two
hourly runs sharing one settled root, a daily derivative with one earliest root
timestamp, one processed self-cognition case with zero failures, and a fresh
`reviewed` group ledger row (plus the expected older `coalesced_skipped` row).
The artifact is
[`focused_group_review_worker_ledger.json`](../../../../test_artifacts/cognition_core_v2/stage_3/focused_live/focused_group_review_worker_ledger.json);
independent exports are recorded in
[`reflection_group_worker_postfix_guarded.json`](../../../../test_artifacts/diagnostics/reflection_group_worker_postfix_guarded.json)
and
[`self_cognition_group_review_postfix_guarded_reviewed.json`](../../../../test_artifacts/diagnostics/self_cognition_group_review_postfix_guarded_reviewed.json).

## Independent Plan Review

Requested and completed gate: GPT-5.6 Sol, xhigh reasoning, normal/default
service speed. The reviewer inspected the direction, ownership, V2 state
approach, scope exclusions, traceability matrix, and live acceptance gates.

Review verdict: `REVISE before approval`.

Blocking findings and disposition:

1. The first draft overloaded `conversation_progress` with group-window turns
   and scheduled continuation. The final plan replaces those carriers with a
   typed `SelfCognitionSourceContext` union and reserves
   `conversation_progress` for genuine current-user participant continuity;
   group and scheduled cases use `None`.
2. The first draft omitted the model/projection ownership boundary and could
   expose `platform_message_id` through copied visible rows. The final plan
   adds `models.py` and `projection.py`, requires explicit allowlists, and
   excludes ids, delivery metadata, scheduler metadata, and V2 scaffolding.
3. The first draft under-specified public-scene construction and style-snapshot
   preparation. The final plan fixes the chronological newest-row structural
   trigger algorithm, canonical scene projection, empty targetless identity
   fields, one immutable snapshot per case, and typed malformed-snapshot
   failure semantics while retaining usable `missing`/`failed` source
   overlays.
4. The first draft had stale test-node names and incomplete direct-live-caller
   migration. The final plan corrects
   `test_group_review_records_target_binding_failed_terminal_row`, marks the
   reflection worker row as preserved, and lists the direct group/commitment
   live fixtures and exact owner nodes.
5. The first draft treated a natural due commitment as a live gate even though
   the current database has no due commitment. The final plan separates the
   deterministic due fixture, DB-free real-LLM smokes, and a bounded read-only
   production observation with no database writes or unbounded wait.
6. The evidence report link was corrected to resolve from the active bugfix
   plan directory.

All blocking findings are resolved in this final plan. The plan owner marked the
plan `approved` after incorporating the review; no implementation or database
mutation occurred during planning.

## Independent Final Implementation Review

GPT-5.6 Sol (`high`, normal service speed) reviewed the completed diff,
deterministic evidence, live artifacts, migrated caller dispositions, and
read-only production exports. Verdict: `BLOCKED` for lifecycle closure.

The reviewer confirmed that the implementation direction is sound: typed
group/scheduled source context, strict runtime validation, prompt-safe
allowlists, chronological group scenes, one shared immutable style snapshot,
and the narrow scheduler-event provenance correction preserve the intended
ownership boundaries. The reviewer also confirmed that both DB-free real-LLM
smokes reached and completed the strict V2 path.

The reviewer requested three evidence remediations. The final exact owner run
was repeated after the last test change with 23 collected and passing nodes;
the adjacent suites were repeated with 134 and 13 passing nodes; and direct
style-snapshot tests now cover malformed state plus `missing`/`failed` source
statuses. Live callers that were unavailable are recorded with their exact
environment or fixture gate above. The remaining blocker is the fresh
production group-ledger transition: creating it requires a write-capable
worker/phase run, while this plan's production observation explicitly forbids
database mutation. The separate cat subject-boundary failure remains a
prompt-quality follow-up and is not folded into this state-contract plan.

## Execution Handoff

Implementation, deterministic verification, and the fresh guarded
test-database ledger transition are complete. The linked reflection plan
records the exact repository correction and guarded evidence; production
mutation remains out of scope. The combined plan is ready for archival.

## Post-review Blocking Remediation

The parent formed an independent proposal before requesting architectural
guidance: preserve the completed self-cognition contract, repair the separate
reflection repository blocker in a guarded test database, and use the resulting
reflection-to-self-cognition run as fresh ledger evidence. GPT-5.6 Sol (`high`,
normal service speed) approved that direction and confirmed that
`captured_at` is temporal provenance rather than root identity. The remediation
therefore retains the earliest normalized timestamp for duplicate roots while
requiring identical `correlation_id`, `character_local_date`, and `scope_kind`.

The production reflection change and its focused tests are owned by the linked
plan [`reflection_recursive_root_timestamp_canonicalization_bugfix_plan.md`](reflection_recursive_root_timestamp_canonicalization_bugfix_plan.md).
This plan consumed the fresh guarded group-ledger result and the combined
self-cognition acceptance gate is closed. Both plans are ready for archival.
