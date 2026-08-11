# cognition surface score ranking follow-up plan 20260810

## Summary

- Goal: determine whether a numeric evaluator score can safely rank multiple
  structurally valid surface candidates after bounded retries, while keeping
  the existing normal response and deterministic fallback behavior.
- Plan class: medium, evidence-gated quality experiment with a conditional
  production follow-up; no database migration.
- Status: completed.
- Acceptance state: the original five-stage score-fallback proposal was
  independently rejected by a gpt-5.6-sol high-reasoning reviewer. This
  revised plan records the review decision and narrows the candidate scope to
  two surface-quality experiments. The user explicitly authorized execution
  on 2026-08-11. The required evidence gate then failed: both owner artifacts
  have zero trace-backed contexts and unaccepted placeholder thresholds. The
  parent therefore retained the baseline first-valid/degraded runtime and
  closed the production score cutover as unapproved; descriptor and workspace
  boundary fixes remain active. Final gpt-5.6-sol signoff approved no-cutover
  historical closeout on 2026-08-11.
- Decision: zero current owners can adopt the completed dialog fix directly
  under their current contracts. Two owners are defensible research
  candidates: surface_content_plan and
  surface_dialog_compliance_repair.
- Score contract: use a finite floating-point score in [0.0, 1.0], with a
  separately calibrated threshold for each owner. Boolean, missing, non-finite,
  out-of-range, or producer-self-assigned scores are invalid.
- Confidence RCA: Cognition V2 intentionally uses bounded string confidence
  descriptors such as `"high"` for semantic context, while numeric `[0.0,
  1.0]` values are evaluator-owned `score`s. The distinction is intentional;
  the boundary documentation and workspace projection are under-specified.
  This plan resolves that ambiguity without converting descriptors into
  ranking signals or introducing a persistence migration.
- Safety boundary: structural validity, evidence handles, role direction,
  relational willingness, addressee identity, action truth, authorization,
  persistence, queue state, delivery state, and state integrity remain hard
  gates. A score never overrides them.
- Role-direction finding: the reviewed dialog evidence showed a user-aligned
  response and no role reversal. The incident was a false rejection/exhaustion
  path, not a role-reversal behavior change. This plan preserves the existing
  role-direction verifier and does not add a role-direction rewrite.

## Scope And Change Direction

The Cognition V2 model-owner matrix has sixteen owners. Four are dialog owners,
leaving twelve non-dialog owners. The twelve count is a policy-matrix count,
not a count of stages that can safely use score bidding.

The completed dialog plan established a valid pattern for a comparable,
side-effect-free visible candidate: retain hard-eligible candidates, allow an
immediate threshold pass, and select the highest eligible score after the
attempt cap. That pattern is not directly reusable by the other owners because
their current retry loops stop at the first structurally valid result or
terminate with an already-controlled fallback.

This follow-up has two phases:

1. Collect and evaluate evidence for surface_content_plan and
   surface_dialog_compliance_repair. The evidence phase must show at least
   two structurally valid, hard-eligible, semantically distinguishable
   candidates for the same owner and context before a production ranking
   change is considered.
2. If the evidence gates pass, implement owner-local score selection for those
   two surface owners under a separately approved production execution. If the
   evidence gates fail, retain the current first-valid, degraded, or prior
   validated-surface behavior and close the experiment without a production
   score change.

The included owners have these boundaries:

- surface_content_plan may rank visible content-plan candidates only after
  canonical cognition truth, selected action truth, target/evidence handles,
  relational willingness, and runtime limits pass deterministic validation.
  The selected plan cannot change cognition stance, authorize an action, or
  mutate state.
- surface_dialog_compliance_repair may rank repair candidates only after the
  existing dialog semantic-fidelity, role-direction, surface-integrity, and
  lexical contracts identify the candidate as hard-eligible. The selected
  repair cannot change the authoritative intent, action, target, addressee,
  relational willingness, or role-direction result.

The following five-stage proposal is explicitly rejected for production scope:

- goal_bid_structure: its validator owns evidence handles, target roles,
  required selection coverage, and relational willingness. A failed bid is not
  a comparable candidate.
- action_planning: its proposal controls intention, resolver requests, and
  pending-resolution semantics. Its empty fallback is safer than selecting an
  uncalibrated plan.
- surface_preference: its contract primarily preserves empty visible
  boundaries and authoritative addressee rows; score ranking adds little
  value and risks identity drift.

The remaining seven non-dialog owners also remain outside score fallback:
image_descriptor, message_decontextualizer, semantic_appraisal,
workspace_collapse, action_authorization, resolver_authorization, and
surface_visual. Their current skip, preserve-input, no-change, deterministic
collapse, deny, or optional-degrade outcomes are the correct ownership-specific
behavior.

No repository-wide retry or scoring abstraction is introduced. The proposed
surface experiment is an owner-local quality-ranking contract, not a general
rule that every failed open or close stage should choose its highest bid.

## Confidence And Score RCA

### Finding

The V2 source and prompt contracts contain two different concepts that share
the word confidence in some field names:

| Concept | Current representation | Owner and permitted use |
|---|---|---|
| `daily_confidence` | closed string enum such as `medium` or `high` | Interaction-style update eligibility; upstream style policy only. |
| `confidence` | bounded string descriptor, optionally empty for group context | Goal/action semantic context and protected observation; never numeric ranking, thresholding, authorization, or delivery gating. |
| `score` | finite float in `[0.0, 1.0]` | Independent dialog or approved surface evaluator quality signal; eligible candidates only. |

The Cognition V2 contracts type goal bids, action bids, branch
observations, and group-engagement context `confidence` as strings. Goal
prompts explicitly require a string and prohibit numeric confidence. The
interaction-style source likewise calls its overlay value a semantic
descriptor, while keeping `daily_confidence` separate. The numeric values
observed in dialog quality checks are named `score`, not V2 `confidence`.

The scoped V2 search found no numeric `confidence` field in the cognition
contracts or direct V2 dialog surfaces. Other subsystems have independent
numeric confidence contracts, such as RAG initializer cache confidence; they
are outside this plan and must not be silently unified with evaluator scores.

### Root cause

The type distinction is intentional, but the vocabulary boundary is
under-specified. The goal/action descriptor is copied through action-planning
context, workspace candidate context, and branch observations. In particular,
workspace collapse currently sends `confidence` beside candidate fields while
its prompt asks the model to compare candidate quality. That creates a
plausible path for a producer self-description such as `"high"` to be treated
as a quality bid, even though deterministic V2 code does not currently use it
for ranking or threshold decisions. Existing bounded-string validation also
proves type and length but does not name the descriptor's non-ranking
semantics.

This is a contract-clarity defect, not evidence that a numeric confidence was
accidentally substituted for `"high"`, and not evidence of role reversal. The
reviewed dialog response remained aligned with the user; the separate failure
was the false-rejection/exhaustion path already covered by the role-direction
plan.

### Resolution in this plan

1. Keep the established `confidence` field on current V2 and persisted
   interaction-style boundaries, but define and document it as a semantic
   `confidence descriptor`. Renaming it and migrating persisted overlays is a
   separate schema decision, not an implicit compatibility alias in this
   experiment.
2. Reject numeric, boolean, non-string, and over-bounded confidence values at
   the existing V2 contract boundaries. Add prompt-contract language stating
   that the descriptor is advisory context and cannot be compared as a score.
3. Remove confidence descriptors from the workspace candidate-quality
   comparison projection. Workspace collapse will receive the event,
   provenance, intention, reason, and other fields required for its own
   relevance partition, but no producer self-assessed quality signal.
4. Keep confidence descriptors available only where the owning semantic stage
   has an explicit advisory-context need. Deterministic action selection,
   thresholding, retry ranking, authorization, persistence, queue, and
   delivery code will never read them as numeric or ordinal evidence.
5. Make `score` the only quality-ranking vocabulary in the two proposed
   surface experiments and the existing dialog evaluator. No producer emits
   its own score, and no confidence descriptor is converted into one.

The change therefore fixes the ambiguity at the contract and prompt boundary
while preserving the existing public/persisted shape and the separate
interaction-style eligibility contract.

## Mandatory Skills

- development-plan: plan lifecycle, execution gates, evidence, and independent
  review.
- llm-trace-debug: protected trace retrieval and correlation of candidate,
  evaluator, and exhaustion evidence.
- debug-llm: human-readable local-LLM evaluation artifacts and one-at-a-time
  live-case inspection.
- local-llm-architecture: model ownership, evaluator separation, prompt
  boundaries, latency, and bounded call budgets.
- py-style: every Python production or test edit during approved execution.
- cjk-safety: every Python prompt or fixture edit containing CJK text.
- test-style-and-execution: deterministic pytest structure and live-LLM
  execution policy.

## Mandatory Rules

- Production implementation requires this plan to be in_progress and requires
  explicit user authorization. The current execution retains the evidence and
  calibration gates defined below.
- Use venv\Scripts\python.exe for Python and pytest commands. Use apply_patch
  for manual file edits. Keep .env outside inspection.
- Preserve the current three-attempt cap and the current owner-specific retry
  policy. Do not generate extra candidates merely to make a score comparison
  possible.
- Pass every evaluator response through
  kazusa_ai_chatbot.utils.parse_llm_json_output(...) before semantic
  validation. Do not add a stage-local JSON parser or repairer.
- The producing surface stage emits only its normal candidate. A separate,
  semantically scoped evaluator owns the score. The producer cannot self-score
  its own candidate.
- Reject a score when it is missing, a boolean, non-numeric, non-finite,
  outside [0.0, 1.0], or accompanied by unknown evaluator keys.
- Keep structural, evidence, role, addressee, relational-willingness,
  action-truth, permission, persistence, queue, delivery, and state-integrity
  checks deterministic and authoritative. A high score cannot make an
  ineligible candidate eligible.
- Treat provider outage, evaluator outage, malformed evaluator output, and
  all-structurally-invalid attempts using the current typed fallback path. Do
  not convert an unavailable evaluator into score 0.0 and do not crash merely
  because ranking is unavailable.
- Keep raw candidates, evaluator output, evidence, and provider failures in
  protected diagnostic artifacts. Public response shapes and adapter delivery
  remain unchanged.
- Treat V2 `confidence` as a bounded semantic descriptor only. Never use it as
  an ordinal, numeric, threshold, or tie-breaking value; only evaluator-owned
  `score` may participate in quality ranking.
- Keep the existing `daily_confidence` interaction-style eligibility enum
  separate from both V2 confidence descriptors and evaluator scores.
- Preserve the existing role-direction, semantic-fidelity,
  surface-integrity, lexical, relational-willingness, and canonical-intent
  checks. Deterministic code must not inspect evaluator prose with keywords or
  rewrite an LLM-owned semantic verdict.
- Keep the healthy-path and worst-case producer/evaluator call counts visible
  in the evidence. The maximum is the existing producer attempt cap plus one
  evaluator call per hard-eligible candidate, never more than three producer
  calls and three evaluator calls for one owner invocation.
- Run live local-LLM cases one at a time and inspect each raw result before
  accepting it as calibration evidence.
- Do not change goal_bid_structure, action_planning, surface_preference,
  authorization, persistence, queue, delivery, or state semantics as part of
  this plan.

## Must Do

- Record the clean implementation baseline and the current owner-policy matrix
  before any approved implementation work.
- Build a trace-backed corpus for each included owner. Each context must have
  at least two candidates from the same bounded invocation, both structurally
  valid and hard-eligible, with a human-reviewable semantic quality
  difference. A context with only malformed, unavailable, or hard-invalid
  candidates does not qualify as ranking evidence.
- Keep separate corpus, evaluator prompt, score calibration, threshold, and
  acceptance artifacts for surface_content_plan and
  surface_dialog_compliance_repair.
- Use independent owner-specific evaluator prompts that receive canonical
  stage context and one candidate at a time. Evaluator output must include a
  score and bounded evidence-bearing blocking issues from the closed owner
  contract. It must not receive permission to alter canonical fields.
- Calibrate each owner threshold against local-LLM performance on a held-out
  context set. The report must show pairwise ordering accuracy, threshold
  false-accept and false-reject counts, provider/evaluator failure behavior,
  call counts, and latency.
- Require a minimum pilot corpus of thirty contexts per included owner, with
  at least twenty contexts for calibration and ten disjoint contexts held out.
  Use two independent human labels per candidate comparison and record
  adjudication for disagreements.
- Promote a numeric owner threshold only when held-out ordering accuracy is at
  least 80 percent and the selected threshold produces no hard-integrity false
  acceptances. Record the actual threshold as an experiment output; do not
  reuse the dialog threshold automatically.
- When production ranking is approved, retain hard-eligible candidates in an
  owner-local ledger, return immediately when the owner threshold is reached,
  and after the cap select the maximum score with deterministic tie-breaking:
  higher score first, then later attempt index, then stable candidate digest.
- When no scored candidate exists, preserve the existing owner fallback. A
  valid unscored candidate may only use the owner’s current baseline fallback
  behavior; it must never be presented as a score-selected candidate.
- Add deterministic tests for score parsing, bounds, boolean rejection,
  hard-gate exclusion, threshold selection, highest-score exhaustion, tie
  breaking, evaluator outage, provider outage, all-invalid exhaustion, and
  canonical-field preservation.
- Add deterministic confidence-boundary tests for numeric rejection, prompt
  descriptor labeling, workspace exclusion from quality comparison, and
  action-planning advisory-only propagation. Cover the separate
  `daily_confidence` interaction-style enum without treating it as a V2 score.
- Add cross-boundary tests proving that a selected surface remains subject to
  dialog semantic-fidelity, role-direction, surface-integrity, lexical, and
  relational-willingness checks.
- Update the ownership manifest, subsystem documentation, calibration report,
  and execution evidence in the same approved change.

## Deferred

- Keep goal_bid_structure, action_planning, and surface_preference out of score
  fallback and out of the first experiment.
- Keep image_descriptor, message_decontextualizer, semantic_appraisal,
  workspace_collapse, action_authorization, resolver_authorization, and
  surface_visual out of score fallback.
- Keep dialog generation and its existing score/evaluator contract unchanged;
  the completed dialog plan is historical evidence, not an extension target.
- Keep a repository-wide score/retry helper, shared threshold, shared
  evaluator prompt, compatibility shim, feature flag, and alternate score scale
  outside the plan.
- Keep malformed structure, missing required fields, unsupported handles,
  conflicting fields, provider unavailability, permission uncertainty,
  persistence failure, queue failure, delivery failure, and state-integrity
  failure outside highest-score selection.
- Keep provider routing, model settings, database schemas, persistence timing,
  adapter behavior, background work, reflection, consolidation, RAG, and task
  resolver retry behavior unchanged.
- Keep the established persisted interaction-style overlay field name and
  schema unchanged. A future `confidence_descriptor` rename would require a
  separately approved big-bang migration with all readers, writers, and stored
  data updated together.
- Keep role-reversal remediation outside this plan. The evidence finding is
  that the reviewed dialog response aligned with the user and did not reverse
  roles; any new role-direction defect requires its own evidence and plan.

## Confirmed Decisions

- The score scale is [0.0, 1.0] as a finite float, not a boolean and not an
  integer-only rubric.
- Thresholds are owner-specific and learned from local-LLM calibration with
  held-out contexts. The dialog threshold is not copied to surface owners.
- The only research candidates are surface_content_plan and
  surface_dialog_compliance_repair.
- The direct same-fix count is zero under the current contracts. The research
  candidate count is two.
- Score selection is permitted only among hard-eligible, side-effect-free,
  structurally valid candidates.
- The current fallback remains authoritative when no score-selected candidate
  exists.
- No generic retry abstraction or semantic post-processor is part of the
  change.
- `confidence` is a descriptor-only field; `score` is the only evaluator
  ranking field. The plan does not normalize, map, or numerically infer one
  from the other.
- Workspace collapse remains outside score fallback, but its candidate prompt
  projection is included solely to remove producer confidence from quality
  comparison.

## Cutover Policy

Overall strategy: evidence-first, then a bounded big-bang change limited to the
two surface owners if every gate passes; baseline-preserving otherwise.

| Area | Policy | Instruction |
|---|---|---|
| Evidence collection | additive | Collect protected traces and evaluation artifacts without changing live production behavior. |
| Evaluator contract | owner-local | Define and validate separate content-plan and compliance-repair contracts. |
| Thresholds | calibrated | Record one numeric threshold per included owner from held-out local-LLM evidence. |
| Surface selection | bounded | Rank only hard-eligible candidates within the current attempt cap. |
| No score available | baseline-preserving | Use the existing owner fallback or retain the prior validated surface. |
| Cognition/action/authorization | baseline-preserving | Keep their current structural and fail-closed behavior. |
| Role direction | baseline-preserving | Keep the existing typed verifier and semantic checks unchanged. |
| Public delivery | baseline-preserving | Preserve public response shapes, persistence, queue, delivery, and adapter contracts. |

## Contracts And Data Shapes

The production candidate ledger is diagnostic and owner-local. It is not a
public API and it is not persisted as user or character state.

Each hard-eligible candidate record contains:

~~~text
owner: surface_content_plan | surface_dialog_compliance_repair
attempt_index: integer in [0, 2]
candidate: validated owner payload
score: finite float in [0.0, 1.0]
blocking_issues: list with at most three typed issue records
selection: threshold | highest_after_cap | baseline_unscored
candidate_digest: stable digest of the bounded candidate payload
~~~

The independent evaluator response is a strict JSON object:

~~~json
{
  "score": 0.0,
  "blocking_issues": [
    {
      "kind": "owner-defined closed issue kind",
      "evidence": "bounded reference to the candidate or typed check"
    }
  ]
}
~~~

The closed blocking-issue kinds are fixed before calibration. For
surface_content_plan they are topic_mismatch, intent_coverage_gap,
unsupported_claim, format_mismatch, and unnecessary_content. For
surface_dialog_compliance_repair they are residual_violation, new_violation,
over_repair, voice_discontinuity, and repair_scope_mismatch. These issue kinds
are quality-only and cannot represent structural, role-direction,
evidence-handle, addressee, relational-willingness, action-truth, permission,
persistence, queue, delivery, or state-integrity decisions; those remain
deterministic hard gates. The evaluator has no field that can rewrite the
candidate or declare a hard-invalid candidate valid.

### Confidence descriptor boundary

The existing V2 wire field remains named `confidence` for this plan, with the
following canonical meaning:

~~~text
confidence: bounded string descriptor; semantic context only; never a score
score: finite float in [0.0, 1.0]; evaluator-owned quality signal only
daily_confidence: interaction-style eligibility enum; not a V2 quality signal
~~~

Goal, action, group-engagement, and branch-observation validators must reject a
numeric or boolean `confidence`. Workspace candidate-quality payloads must not
include it. Any remaining semantic-stage prompt that receives the descriptor
must label it as advisory context and must not instruct the model to rank or
threshold it. Existing persisted interaction-style overlay data remains
descriptor-shaped and is covered by its current validator and regression
tests.

Selection is deterministic:

1. Parse and validate the evaluator response through the canonical parser.
2. Exclude candidates with structural, typed semantic, or blocking issues.
3. Return the first candidate whose owner-specific score reaches the calibrated
   threshold.
4. After the existing attempt cap, choose the highest eligible score; resolve
   ties by later attempt index and then stable candidate digest.
5. If no scored candidate exists, use the existing fallback and record the
   typed disposition.

The threshold is an artifact of the owner’s held-out calibration report, not a
hard-coded global constant. A missing threshold blocks production cutover.

## Target State

The two included surface stages have explicit quality-ranking behavior only
after the evidence gate passes:

~~~text
producer attempt
  -> canonical parse and owner validation
  -> deterministic hard-eligibility gate
  -> independent owner evaluator
  -> finite [0,1] score validation
  -> threshold return, bounded retry, or highest-score exhaustion
  -> existing surface/fallback and unchanged dialog boundary
~~~

The target state has these invariants:

- A malformed or hard-invalid candidate never enters the score ledger.
- A provider or evaluator failure never becomes a numeric score.
- A score cannot change cognition truth, stance, action authorization, target,
  addressee, role direction, relational willingness, persistence, queue,
  delivery, or state.
- Exhaustion with at least one scored candidate returns the deterministic
  highest-score candidate rather than raising an avoidable quality-exhaustion
  error.
- Exhaustion with no scored candidate retains the existing deterministic
  fallback and its current observability.
- The normal path stops at the threshold and stays within the current producer
  attempt cap.
- The dialog stage continues to perform its existing independent semantic,
  role-direction, surface-integrity, and lexical checks.

## Test Impact And Traceability

The following matrix is the required execution scope. New test node IDs are
fixed names for the tests to be added; they are not permission to expand the
semantic owner set.

| Repository path | Changed symbol/contract | Semantic owner | Deterministic pytest node IDs | Supplemental integration/live nodes | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py | baseline first-valid retry and fallback retained after the score evidence gate failed | surface-stage quality selection | tests/unit/cognition_core_v2/test_surface_stages.py::test_surface_stages_exposes_owned_contract | tests/test_cognition_core_v2_model_retry_continuity.py::test_text_surface_retry_or_validated_degraded_projection; tests/test_cognition_core_v2_model_retry_continuity.py::test_dialog_surface_repair_exhaustion_retains_valid_surface; tests/test_cognition_core_v2_surface_score_bidding_live_llm.py::test_live_surface_content_score_orders_candidates; tests/test_cognition_core_v2_surface_score_bidding_live_llm.py::test_live_dialog_compliance_repair_score_orders_candidates | deterministic unit, regression integration, one-at-a-time live gate reserved | Prevents an uncalibrated score path from changing the existing surface fallback. |
| src/kazusa_ai_chatbot/cognition_core_v2/surface.py | run_text_surface_planning, repair_text_surface_planning, and degraded-surface routing | canonical surface truth and fallback | tests/unit/cognition_core_v2/test_surface.py::test_surface_score_fallback_preserves_canonical_intent_and_limits; tests/unit/cognition_core_v2/test_surface.py::test_surface_output_preserves_relational_willingness_v2; tests/unit/cognition_core_v2/test_surface.py::test_surface_exposes_owned_contract | tests/integration/cognition_core_v2/test_relational_stance_propagation.py::test_relational_stance_preserves_polarity_through_surface_and_dialog; tests/test_cognition_core_v2_model_retry_continuity.py::test_degraded_text_surface_projects_only_validated_v2_truth | deterministic unit and integration | Prevents score selection from changing authoritative cognition truth, relational willingness, limits, or downstream dialog inputs. |
| src/kazusa_ai_chatbot/cognition_core_v2/workspace.py | workspace candidate prompt projection excludes producer confidence from quality comparison | workspace relevance partition | tests/unit/cognition_core_v2/test_workspace.py::test_workspace_collapse_does_not_rank_by_confidence_descriptor; tests/unit/cognition_core_v2/test_workspace.py::test_workspace_exposes_owned_contract | none | deterministic unit | Prevents a free-form producer descriptor from becoming an implicit quality bid. |
| src/kazusa_ai_chatbot/cognition_core_v2/contracts.py | confidence descriptor type and non-ranking boundary for V2 contracts | canonical V2 contract validator | tests/unit/cognition_core_v2/test_contracts.py::test_goal_and_action_confidence_reject_numeric_values; tests/unit/cognition_core_v2/test_contracts.py::test_group_confidence_rejects_numeric_values; tests/unit/cognition_core_v2/test_contracts.py::test_contracts_exposes_owned_contract | tests/test_cognition_core_v2_contracts.py::test_group_context_rejects_every_invalid_shape_and_bound | deterministic unit and integration | Prevents numeric confidence from crossing a semantic contract boundary. |
| src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py | goal prompt and selection contract labels confidence as descriptor-only | cognition stance and goal owner | tests/unit/cognition_core_v2/test_goal_cognition.py::test_goal_prompt_labels_confidence_as_descriptor; tests/unit/cognition_core_v2/test_goal_cognition.py::test_goal_cognition_exposes_owned_contract | tests/test_cognition_core_v2_prompt_contract_guidance.py::test_nonordinary_generic_goal_prompt_excludes_relational_contract | deterministic unit and prompt-contract regression | Prevents goal self-assessment from being mistaken for evaluator score. |
| src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py | action-planning projection retains confidence only as labeled advisory context | action proposal owner | tests/unit/cognition_core_v2/test_action_selection.py::test_action_planning_keeps_confidence_descriptor_advisory; tests/unit/cognition_core_v2/test_action_selection.py::test_action_selection_exposes_owned_contract | tests/test_cognition_core_v2_action_planning_bugfix.py::test_action_plan_exhaustion_returns_empty_control_output | deterministic unit and regression integration | Prevents descriptor propagation from becoming numeric action ranking or authorization. |
| src/kazusa_ai_chatbot/reflection_cycle/interaction_style.py and src/kazusa_ai_chatbot/db/interaction_style_images.py | existing semantic overlay confidence and separate daily eligibility confidence | interaction-style semantic context | tests/test_interaction_style_images.py::test_validate_interaction_style_overlay_accepts_semantic_confidence; tests/test_interaction_style_images.py::test_validate_interaction_style_overlay_caps_confidence_descriptor; tests/test_interaction_style_images.py::test_validate_interaction_style_overlay_rejects_confident_empty_overlay | none | deterministic regression | Prevents the persisted interaction-style descriptor or daily eligibility enum from being conflated with V2 evaluator score. |
| src/kazusa_ai_chatbot/nodes/dialog_agent.py | existing evaluator score remains the sole dialog quality-ranking field | dialog quality evaluator | tests/unit/nodes/test_dialog_agent.py::test_dialog_score_is_numeric_quality_signal; tests/unit/nodes/test_dialog_agent.py::test_numeric_score_rejects_boolean_and_out_of_range_values | tests/test_dialog_agent.py::test_dialog_exhaustion_selects_highest_score_not_latest; tests/test_dialog_agent.py::test_dialog_exhaustion_all_unavailable_selects_latest_valid_candidate; tests/test_dialog_agent.py::test_dialog_exhaustion_ties_select_latest_attempt | deterministic unit and regression integration | Prevents confidence naming from weakening the established numeric dialog score contract. |
| tests/ownership/source_test_impact_manifest.json | ownership entries for the changed surface-stage and surface contracts | test-coverage ownership | tests/unit/cognition_core_v2/test_surface_stages.py::test_surface_stages_exposes_owned_contract; tests/unit/cognition_core_v2/test_surface.py::test_surface_exposes_owned_contract | none | deterministic manifest-backed unit coverage | Prevents a production surface change from landing without exact owner-level test traceability. |
| tests/test_cognition_core_v2_surface_score_bidding_live_llm.py | new owner-specific calibration gates and evidence artifact checks | local-LLM quality evaluation | tests/unit/cognition_core_v2/test_surface_score_bidding.py::test_calibration_artifact_requires_owner_specific_threshold; tests/unit/cognition_core_v2/test_surface_score_bidding.py::test_calibration_artifact_rejects_boolean_or_nonfinite_score | tests/test_cognition_core_v2_surface_score_bidding_live_llm.py::test_live_surface_content_score_orders_candidates; tests/test_cognition_core_v2_surface_score_bidding_live_llm.py::test_live_dialog_compliance_repair_score_orders_candidates | deterministic contract unit plus one-at-a-time live | Prevents production thresholds from being copied from dialog or accepted without local-model ordering and failure evidence. |
| src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py | current goal-bid contract | cognition stance and goal owner | tests/unit/cognition_core_v2/test_goal_cognition.py::test_goal_cognition_exposes_owned_contract | none | unchanged safety regression | Prevents the experiment from treating invalid or incomplete cognition bids as score-comparable surface candidates. |
| src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py | current action-plan contract | action proposal owner | tests/unit/cognition_core_v2/test_action_selection.py::test_non_accepting_stance_suppresses_downstream_effects; tests/unit/cognition_core_v2/test_action_selection.py::test_action_selection_exposes_owned_contract | tests/test_cognition_core_v2_action_planning_bugfix.py::test_action_plan_exhaustion_returns_empty_control_output; tests/test_cognition_core_v2_action_authorization.py::test_empty_action_plan_adds_no_authorization_call | unchanged safety regression | Prevents score selection from turning an action-planning fallback into an execution or resolver request. |
| src/kazusa_ai_chatbot/cognition_core_v2/model_attempt_policy.py | exact 16-owner attempt matrix | retry-policy owner | tests/unit/cognition_core_v2/test_model_attempt_policy.py::test_model_attempt_policy_exposes_owned_contract | tests/test_cognition_core_v2_model_retry_continuity.py::test_v2_attempt_policy_matches_exact_owner_matrix | unchanged policy regression | Prevents accidental expansion from two evidence-gated surface owners to all twelve non-dialog owners. |

The existing dialog role-direction and continuity tests remain required
supplemental evidence. The plan does not add a role-reversal test because the
reviewed incident did not demonstrate role reversal; it adds preservation
coverage around the existing verifier boundary instead.

## Change Surface

### Delete

- None. The experiment removes no retry path, evaluator, fallback, or public
  contract.

### Modify

- src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py: add only the two
  owner-local score-ledger paths after evidence and threshold approval;
  preserve current provider, parser, structural validation, and fallback
  behavior.
- src/kazusa_ai_chatbot/cognition_core_v2/surface.py: route a selected surface
  candidate without changing canonical cognition projection or public response
  shape.
- src/kazusa_ai_chatbot/cognition_core_v2/workspace.py: remove the producer
  confidence descriptor from the candidate-quality prompt projection while
  preserving workspace relevance and partition semantics.
- src/kazusa_ai_chatbot/cognition_core_v2/contracts.py: make the descriptor
  versus score boundary explicit in validation documentation and regression
  coverage; preserve the existing public field shape.
- src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py and
  action_selection.py: label confidence as advisory semantic context and
  preserve their existing ownership, retry, action, and authorization
  semantics.
- tests/unit/cognition_core_v2/test_surface_stages.py: add deterministic score
  contract, selection, tie, and exhaustion tests.
- tests/unit/cognition_core_v2/test_surface.py: add canonical-field and
  fallback-preservation tests.
- tests/unit/cognition_core_v2/test_workspace.py,
  test_contracts.py, test_goal_cognition.py, and test_action_selection.py:
  add confidence-descriptor type, prompt, projection, and advisory-context
  regressions.
- tests/unit/nodes/test_dialog_agent.py: add explicit numeric-score versus
  confidence naming regressions for the existing dialog evaluator.
- tests/ownership/source_test_impact_manifest.json: record exact tests for the
  changed source owners.
- src/kazusa_ai_chatbot/cognition_core_v2/README.md and
  src/kazusa_ai_chatbot/nodes/README.md: document the two-owner,
  evidence-gated boundary, confidence descriptor versus score vocabulary, and
  explicit exclusions.
- development_plans/README.md: register this draft under active bugfix plans
  until it is approved, superseded, or archived.

### Create

- experiments/cognition_surface_score_bidding/README.md: fixed calibration
  protocol, corpus rules, score schema, threshold procedure, and artifact
  naming.
- experiments/cognition_surface_score_bidding/content_plan_candidates.jsonl:
  protected-evidence references and bounded candidate summaries for the
  content-plan owner.
- experiments/cognition_surface_score_bidding/compliance_repair_candidates.jsonl:
  protected-evidence references and bounded candidate summaries for the
  compliance-repair owner.
- experiments/cognition_surface_score_bidding/calibration_report.md:
  held-out ordering, threshold, failure, call-count, and latency evidence.
- experiments/cognition_surface_score_bidding/thresholds.json: the two
  owner-specific thresholds and calibration metadata after acceptance.
- tests/unit/cognition_core_v2/test_surface_score_bidding.py: deterministic
  score and calibration-artifact contract tests.
- tests/test_cognition_core_v2_surface_score_bidding_live_llm.py: one-at-a-time
  live local-LLM cases for the two included owners, only after the evidence
  phase is authorized.

### Keep

- src/kazusa_ai_chatbot/nodes/dialog_agent.py and its existing evaluator,
  threshold, candidate, and role-direction behavior.
- src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py,
  action_selection.py, model_attempt_policy.py, authorization modules,
  decontextualization, appraisal, visual surface, and all durable/stateful
  retry owners: preserve semantic ownership and retry/action behavior apart
  from the explicit descriptor-labeling changes above.
- Workspace collapse: keep it out of score fallback and preserve its
  relevance/partition contract; only its candidate prompt projection changes
  to remove confidence as a quality-comparison input.
- Existing deterministic degraded text, prior validated-surface retention,
  optional visual skip, empty action plan, deny, no-change, and preserve-input
  fallbacks.
- Public response shapes, persistence, queue, delivery, adapter, provider,
  database, and scheduler contracts.

## Agent Autonomy Boundaries

The execution agent may choose local Python structure, test fixtures, protected
artifact serialization, prompt wording within the fixed owner boundaries, and
the mechanics of deterministic candidate ordering.

The execution agent must not:

- promote the plan, edit production code while the plan is draft, or treat a
  failed evidence gate as approval;
- add a third score-enabled owner or modify an excluded owner;
- change the [0.0, 1.0] scale, threshold procedure, tie order, attempt cap,
  evaluator independence, or hard-gate list;
- introduce a generic retry/scoring helper, compatibility layer, feature flag,
  semantic post-processor, or hidden call multiplier;
- allow evaluator prose or score to change role direction, cognition stance,
  action authorization, addressee, persistence, queue, delivery, or state;
- replace the current fallback with a new fallback when no score is available.

If the current code cannot satisfy these contracts without changing an excluded
owner or public boundary, execution stops and records a plan amendment request.

## Runtime Or Resource Constraints

- Producer calls remain bounded by the current owner cap: at most three per
  invocation.
- Evaluator calls are bounded to one per hard-eligible candidate and at most
  three per invocation. No extra producer attempts are requested to populate a
  ledger.
- Evidence records baseline and candidate producer calls separately from
  evaluator calls, plus p50/p95 latency for each included owner. A production
  cutover requires explicit owner acceptance of the measured normal and
  worst-case overhead.
- Candidate and evaluator payloads remain within the current surface-stage
  prompt/output limits. Raw candidate text and evaluator prose stay in
  protected artifacts, not public responses.

## Verification

Verification proceeds in this order:

1. Static and baseline audit: record the base SHA, clean status, owner matrix,
  current attempt caps, current fallbacks, and the completed dialog plan.
   Confirm that the role-direction finding is user alignment rather than
   reversal, and audit every V2 confidence field and numeric score field for
   owner, type, and permitted use.
2. Evidence phase: collect the minimum corpus for both owners, inspect raw
   local-LLM outputs one case at a time, and produce separate calibration
   reports. Exclude any candidate that fails deterministic validation or hard
   integrity gates.
3. Calibration gate: label candidate comparisons independently, hold out
   contexts by conversation case, calculate ordering and threshold metrics, and
   record owner-specific thresholds. A missing, unstable, or
   non-discriminating score blocks production implementation.
4. Deterministic implementation verification: after explicit approval, run the
   exact unit nodes in the traceability matrix, then run the affected V2
   regression nodes. Inspect failures and score/fallback dispositions.
5. Live verification: run the two named live cases individually with the local
   model, inspect every raw evaluator and selected-candidate artifact, and
   compare against the held-out calibration report. Live failure blocks
   cutover.
6. Boundary audit: prove that no excluded owner or public boundary changed, all
  invalid/unavailable paths retain current behavior, and role-direction checks
  still govern the final dialog surface. Prove that confidence descriptors do
  not enter workspace quality comparison and that `score` remains the only
  quality-ranking field.
7. Independent code review: obtain a native reviewer after implementation and
   before completion. Resolve every finding in scope and record the final
   verdict in this plan.

## Acceptance Criteria

The plan may move from draft to approved implementation only when:

- The revised scope is accepted as two research candidates and zero direct
  same-fix owners.
- Each included owner has at least thirty trace-backed contexts with at least
  two hard-eligible, semantically distinguishable candidates per context,
  twenty calibration contexts, and ten disjoint held-out contexts.
- Each candidate comparison has two independent human labels or a recorded
  adjudication, and held-out score ordering accuracy is at least 80 percent.
- Each owner has a separately recorded finite threshold in [0.0, 1.0]; the
  threshold has no hard-integrity false acceptance in held-out evidence.
- Deterministic tests reject boolean, missing, non-finite, out-of-range, and
  unknown-key evaluator results.
- Deterministic tests prove invalid and blocking candidates cannot enter the
  ledger, threshold selection returns immediately, highest-score exhaustion is
  deterministic, and ties use the fixed order.
- Deterministic tests prove numeric or boolean confidence is rejected at V2
  boundaries, confidence is labeled as advisory descriptor context, workspace
  does not compare it as candidate quality, and interaction-style
  `daily_confidence` remains separate from evaluator score.
- Provider outage, evaluator outage, all-invalid attempts, and no-score
  exhaustion preserve the current owner fallback without an avoidable crash.
- Canonical cognition truth, relational willingness, limits, action truth,
  target/addressee identity, role direction, and public response shape remain
  unchanged through selected and degraded surfaces.
- Existing dialog role-direction, semantic-fidelity, surface-integrity,
  lexical, action-authorization, and relational-propagation regressions pass.
- Normal and worst-case producer/evaluator calls and latency are recorded and
  explicitly accepted before cutover.
- The independent code reviewer accepts scope, contracts, tests, and residual
  risk, and the implementation diff is contained to this plan’s Change
  Surface.

## Progress Checklist

- [x] Read the development-plan registry and governing plan contract.
- [x] Inspect the completed dialog score-bidding plan and current V2 owner
  matrix.
- [x] Obtain independent gpt-5.6-sol review of the five-stage proposal.
- [x] Narrow the scope to two evidence-gated surface research candidates.
- [x] Scope and perform the Cognition V2 confidence-versus-score RCA; add the
  descriptor-only resolution and workspace boundary to this plan.
- [x] Capture the execution baseline and record the trace-corpus gate as
  blocked: neither owner had a trace-backed corpus.
- [x] Evaluate the owner-specific calibration gate; zero contexts and zero
  score samples keep both thresholds placeholder-only and unaccepted.
- [x] Obtain explicit user approval and promote plan status before production
  edits.
- [x] Add deterministic descriptor, workspace, fallback, and artifact-contract
  tests within the allowlist; retain the baseline runtime after the evidence
  gate failed.
- [x] Run the reserved live gate nodes one at a time; both explicitly skip
  because calibration is blocked, and no artifacts are accepted.
- [x] Complete native implementation review, parent remediation, and final
  gpt-5.6-sol boundary audit.

## Execution Evidence

Planning evidence captured on 2026-08-10:

- Worktree baseline: 2526803c (main synchronized with origin/main), clean
  before this plan was created.
- Prior completed plan:
  development_plans/archive/completed/bugfix/dialog_score_bidding_and_role_direction_bugfix_plan_20260810.md.
- Current owner matrix: sixteen total owners, twelve non-dialog owners.
- Independent reviewer: gpt-5.6-sol, high reasoning, normal speed.
- Reviewer verdict: REJECT for the original five-stage production proposal;
  surface_content_plan and surface_dialog_compliance_repair were retained only
  as evidence-gated research candidates.
- Confidence RCA amendment on 2026-08-11: V2 confidence values are
  intentionally bounded semantic descriptors; numeric quality values are
  evaluator-owned scores. The ambiguity is a contract/prompt-boundary defect,
  with workspace candidate projection as the concrete misuse risk. No numeric
  V2 confidence field was found.
- Deterministic confidence-related audit coverage: 11 existing contract and
  interaction-style tests passed on 2026-08-10; no live LLM test was required
  for the type/ownership RCA.
- The workspace contains a separate in-progress role-operation contract
  change. Its files and plan remain preserved; confidence implementation must
  be sequenced with that change where ownership overlaps.
- No production source or test implementation was changed while producing this
  draft or this confidence amendment.

Execution baseline captured on 2026-08-11 before the delegated production
handoff:

- Base SHA: `500d5d18b87db977a6079d6732ab2b3ab89574e7`.
- Pre-task branch: `main`, synchronized with `origin/main`; the worktree was
  clean before parent-owned test and plan updates. The separate role-operation
  plan archive move was preserved as unrelated workspace state.
- Targeted deterministic baseline: 41 passed in 7.78 seconds across the
  surface, workspace, contract, goal, action, dialog, and interaction-style
  regression files.
- Parent test-first checkpoint: collection after adding the score-contract
  nodes failed only because the production score helper symbols were not yet
  present. This is the expected red checkpoint for the delegated implementation.
- Initial evidence audit found no trace-backed corpus for either included owner
  in the workspace. No candidate or score sample was synthesized; the
  experiment artifacts record the blocked state explicitly.
- Parent-owned exploratory live probes ran one at a time against the local
  endpoint and produced one producer plus one evaluator response for each
  owner before the evidence-gated rollback. Those two raw artifacts were
  inspected but are not calibration evidence: they contain no candidate
  comparison, labels, held-out split, latency distribution, or threshold
  acceptance.
- Native implementation review by the project reviewer on 2026-08-11 returned
  `REJECT`. The parent resolved the blocking production findings by restoring
  the baseline `surface_stages.py` runtime, retaining confidence-descriptor
  and workspace-boundary changes, and updating the live gates to remain
  explicitly skipped while calibration is blocked.
- Post-remediation deterministic verification: 70 passed in 25.54 seconds
  across the affected unit, interaction-style, and retry-continuity suites;
  `py_compile` and `git diff --check` passed. The reserved live nodes remain
  skipped by their explicit calibration blocker.
- Traceability verification: all 36 mapped nodes resolve; the plan-contract
  and ownership checks passed 14 tests.
- Final gpt-5.6-sol high-reasoning review at normal speed on 2026-08-11:
  `APPROVE`. The reviewer independently reran the affected suite with 70
  passes, confirmed the stale-node replacements, Chinese descriptor wording,
  restored required-selection wording and `del messages`, byte-identical
  baseline surface runtime, zero/placeholder/unaccepted artifacts, clean
  diff checks, and preservation of the unrelated relevance-plan archive move.

Exact post-remediation deterministic command:

~~~powershell
& '.\venv\Scripts\python.exe' -m pytest tests\unit\cognition_core_v2\test_surface_stages.py tests\unit\cognition_core_v2\test_surface.py tests\unit\cognition_core_v2\test_workspace.py tests\unit\cognition_core_v2\test_contracts.py tests\unit\cognition_core_v2\test_goal_cognition.py tests\unit\cognition_core_v2\test_action_selection.py tests\unit\cognition_core_v2\test_surface_score_bidding.py tests\unit\nodes\test_dialog_agent.py tests\test_interaction_style_images.py tests\test_cognition_core_v2_model_retry_continuity.py -q
~~~

Result: `70 passed in 25.54s`.

## Independent Plan Review

The requested independent plan review was performed by a native
gpt-5.6-sol high-reasoning subagent at normal speed on 2026-08-10.

The review rejected the original proposal because the five proposed owners do
not currently retain multiple hard-eligible comparable candidates or possess a
validated ranking signal. It also found that four proposed owners already
finish with controlled fallbacks rather than crashing. The review required:

- narrowing production scope to the two surface-quality research candidates;
- excluding goal bids, action planning, and preference preservation;
- independent owner-specific evaluators with finite [0,1] scores and
  evidence-bearing blocking issues;
- held-out local-LLM calibration, latency/call budgets, and exact test
  traceability; and
- preservation of provider outage, all-invalid, role, state, permission, and
  delivery behavior.

Those conditions are incorporated above. The revised plan was draft until the
user explicitly authorized execution on 2026-08-11 and the parent promoted it
to `in_progress`. Evidence, calibration, deterministic verification, and
independent implementation review remain completion gates.

The confidence-versus-score RCA and workspace projection boundary were added
on 2026-08-11 after that review. They are evidence-backed scope amendments,
not an independent approval of production implementation. The implementation
reviewer must verify the descriptor-only contract, the absence of confidence
from workspace quality comparison, the interaction-style schema boundary, and
the added test traceability before the plan can close.

## Independent Code Review

The project-native reviewer inspected the first implementation and returned
`REJECT` on 2026-08-11. Findings were: unapproved placeholder thresholds were
active, valid low-score candidates caused extra producer calls, blocking
issues entered the ledger, deterministic coverage was incomplete, and the
selection/evaluator helpers generalized across the two owners. The parent
restored the baseline surface runtime and recorded the evidence-gate failure;
the score cutover is not active.

The parent applied the remediation and reran the exact affected suite,
traceability checks, compilation, diff checks, and both reserved live nodes.
The final gpt-5.6-sol high-reasoning reviewer at normal speed returned
`APPROVE` on 2026-08-11 with no remaining blockers. This plan closes as a
no-cutover historical record: it does not claim calibrated score ranking or an
accepted production implementation. Any future score-ranking change requires
a new user-approved evidence package.
