# Cognition Core V2 residual Appraisal and goal-bid contract bugfix plan 20260811

## Summary

- Goal: resolve the residual Cognition V2 semantic-Appraisal boundary failures
  and goal-bid contract failures left after
  `cognition_core_v2_atomic_appraisal_finalization_and_terminal_eligibility_bugfix_plan_20260811.md`, while preserving LLM ownership of semantic judgment and every strict state guard.
- Plan class: high-risk, evidence-gated Cognition V2 bugfix covering
  Appraisal reduction classification, prompt-domain projection, goal bidding,
  deterministic regression, and live-LLM quality gates. No database migration
  or public response-shape change is planned.
- Status: completed.
- Execution boundary: the user explicitly authorized execution on 2026-08-11.
  Production implementation is limited to the fixed change surface and the
  acceptance gates below. The real-LLM test contract updates recorded below
  were completed before this execution checkpoint.
- Review state: the initial GPT-5.6 Sol proposal review was **PASS WITH
  AMENDMENTS**. The follow-up baseline evaluation and historical execution
  review were **FAIL / NOT SIGNED OFF** because their pre-recovery cohorts did
  not satisfy the fixed gates. The final post-change execution record satisfies
  every live gate, and the final GPT-5.6 Sol review returned **PASS — SIGN OFF**
  with no blocking corrections.
- Production implementation completed by GPT-5.6 Luna (`gpt-5.6-luna`,
  reasoning `max`, normal/default speed) in the four plan-owned files:
  `semantic_appraisal.py`, `facade.py`, `semantic_source_planner.py`, and
  `goal_cognition.py`. Directly coupled prompt-contract corrections in
  `action_selection.py`, `character_carryover.py`, and
  `nodes/dialog_agent.py` were included during the final deterministic and
  live-regression recovery. The implementation passed the final independent
  review gate.
- Luna completed a second bounded remediation in
  `semantic_source_planner.py` and `goal_cognition.py` after the independent
  Sol review: the undeclared planner field was removed and repair prompts now
  use `goal_output_contract` as their sole top-level field/type authority.

### Test-only pre-execution updates

The live-test contract was updated before production-plan execution:

- [required-selection live tests](../../../tests/test_cognition_core_v2_required_selection_live_llm.py)
  now require the dynamic goal_output_contract, exact six-field selected
  response operations, separated role/evidence domains, and durable
  operation-contract diagnostics.
- [ordinary goal capability live test](../../../tests/test_cognition_core_v2_goal_capability_live_llm.py)
  now requires the same dynamic contract for a non-selection ordinary goal.
- [new Appraisal contract live tests](../../../tests/test_cognition_core_v2_appraisal_contract_live_llm.py)
  cover exact target-path projection, cap accounting, domain separation, and
  singular nullable micro-Appraisal output.

The pre-fix live baseline is intentionally recorded as evidence: the
singular nullable Appraisal probe passed; the dense Appraisal probe failed on
the missing permitted_target_paths field; the required-selection probe
observed both an outer-operation repetition and a missing goal_output_contract;
and the ordinary goal probe's semantic quality judge passed while its dynamic
contract gate failed. These failures are expected to close only after the
approved production changes are executed and rerun.

### Fixed change direction

1. P0: retain the producer-local Appraisal reduction as a non-authoritative
   compatibility preflight, classify `CognitionStateError` separately before
   generic `ValueError`, preserve accepted prefixes, and leave full state
   commit to the facade finalization boundary.
2. P1: project exact, cap-safe Appraisal target paths with separate role,
   evidence, and path domains; preserve the archived lifecycle filter and
   strict reducers.
3. P2: add a compact dynamic goal-output contract beside the current allowed
   domains for every bidding mode, including exact selected-operation fields,
   handle-set separation, relational pairings, and deterministic recurrence
   carry-forward.
4. P3: add deterministic coverage, a fixed 71-row live-bidding evidence
   ledger, an exact 21-node Appraisal replay gate, and independent agent
   evaluation with cohort floors.

No alternate fix remains open. Retry-cap increases, guard weakening,
terminal normalization, aliases, evidence dropping, semantic mappers, and
deterministic replacement of LLM judgment are rejected.

## Evidence baseline

### Raw failure evidence

The refreshed read-only export window is local
`2026-08-11 00:00:00` through `16:17:02 +12:00`, or UTC
`2026-08-10T12:00:00Z` through `2026-08-11T04:17:02Z`.

- 194 cognition run documents.
- 1,417 ordinary trace-step documents.
- 117 full failure-capsule documents and 110 unique capsule trace IDs.
- 286 failed semantic-Appraisal attempts across 109 unique traces.
- 62 traces with an Appraisal failure event and 47 recovered-only traces.
- 97 failed goal-bid validation attempts across 67 traces.
- 106/109 affected Appraisal traces ended in `partial_failure`; 3/109 ended
  in `terminal_failure`.

Authoritative artifacts:

- [RCA and fix proposal](../../../test_artifacts/diagnostics/cognition_appraisal_remaining_failures_20260811_rca_and_fix_proposal.md)
- [GPT-5.6 Sol review and baseline verdict](../../../test_artifacts/diagnostics/cognition_appraisal_remaining_failures_20260811_sol_review.md)
- [canonical 109-trace inventory](../../../test_artifacts/diagnostics/cognition_appraisal_trace_inventory_20260811.md)
- [full failure capsules](../../../test_artifacts/diagnostics/cognition_failure_capsules_today_20260811_latest_full.json)
- [ordinary trace steps](../../../test_artifacts/diagnostics/cognition_llm_steps_today_20260811_latest.json)
- [cognition runs](../../../test_artifacts/diagnostics/cognition_llm_runs_today_20260811_latest.json)
- [bidding live-LLM review](../../../test_artifacts/diagnostics/cognition_v2_bidding_llm_test_review_20260811.md)
- [final immutable 71-row bidding ledger](../../../test_artifacts/diagnostics/cognition_v2_bidding_llm_evidence_ledger_20260812_final.md)
- [supplemental captured replay evidence](../../../test_artifacts/diagnostics/cognition_v2_supplemental_captured_replay_evidence_20260812.md)
- [prompt-coupled live evidence](../../../test_artifacts/diagnostics/cognition_v2_prompt_coupled_live_review_20260812.md)
- [final GPT-5.6 Sol signoff](../../../test_artifacts/diagnostics/cognition_v2_remaining_appraisal_and_goal_bid_sol_review_20260812_final.md)
- [Appraisal boundary artifacts](../../../test_artifacts/cognition_core_v2_appraisal_boundary/)

The raw corpus predates the currently present archived lifecycle/source
filtering changes. It is evidence for failure families and boundary pressure,
not a post-fix incidence denominator.

### Residual failure families

| Family | Attempts | Traces | RCA disposition |
|---|---:|---:|---|
| resolved knowledge gap cannot transition | 79 | 76 | deterministic state incompatibility misclassified as producer contract error |
| micro-Appraisal fields not exact | 42 | 27 | producer schema adherence |
| terminal event cannot transition | 31 | 20 | deterministic state incompatibility misclassified as producer contract error |
| selected role contains unknown evidence handle | 28 | 28 | role/evidence domain adherence |
| unowned `knowledge_gaps.*.uncertainty` path | 37 | 37 | path-domain adherence; corpus is pre-filter |
| invalid semantic role value | 10 | 7 | enum/role contract adherence |
| question-specific unknown role handles | 14 | 14 | role-domain adherence |
| unowned relationship/drive path prefix | 14 | 11 | non-compositional path contract |
| evidence-retention overflow | 4 | 2 | deterministic capacity error misclassified as contract error |
| remaining binding/proposition/delta/finalization errors | 27 | 20 | producer contract or bounded finalization rejection |

Categories overlap. The complete trace-ID sets are the 62 + 47 disjoint lists
in the canonical inventory. The user-provided anchor
`llmtrace_ade6709d51614339b56b303de5c058c6` is included there.

### Bidding baseline and agent verdict

The exact 71-node primary cohort was collected one node per process. First-pass
results were:

| Cohort | First-pass result | Required floor | Sol verdict |
|---|---:|---:|---|
| goal capability | 1/1 | 1/1 | PASS |
| relational willingness | 13/13 | 13/13 | PASS |
| required selection | 15/20 | 19/20 | FAIL |
| action planning/authorization | 34/35 | 34/35 | PASS with instability |
| workspace collapse | 2/2 | 2/2 | PASS |
| **primary total** | **65/71 (91.55%)** | **at least 68/71 (95.8%)** | **FAIL** |

The original first-pass action failure is
`test_c03_action_planning_selects_local_recall_from_connector_state`; its
durable capture selected `speech` and left the goal blocked where the test
requires the evidence/local-recall route. Focused repeats additionally failed
the C03 case and
`test_o04_action_planning_selects_local_recall_from_frozen_e2e_state` once
each. These repeats are instability evidence and do not replace the original
34/35 numerator.

The four supplemental captured replays were 0/4 in this execution, so their
separate hard gate also fails. Samples 1-3 observed `deflect` instead of the
fixture's `reject`; this remains an LLM-owned character-judgment mismatch,
not a deterministic schema repair opportunity. The captured-run repair case
was blocked by a missing historical export and remains an infrastructure /
harness failure.

The separate Appraisal replay harness did not produce valid live evidence:
13 nodes lacked the historical 2026-08-04 export, one A1a573 replay observed
an earlier unknown-role error instead of the fixture's expected unowned-path
error, two capacity nodes were explicitly skipped, two exhaustion nodes lacked
`target_2304_llm_trace.json`, and three near-cap replays no longer reproduced
their captured initial failure. These are gate failures, not passes.

The affected deterministic run was 152 passed, 2 skipped, and 7 failed. The
mapped source and contract nodes passed. Three failures are stale near-cap
prompt-continuity expectations that assert older error ordering; four live-DB
smoke failures are caused by the existing `seed_shared_documents` hash
mismatch for `character_state:{'_id':'global'}`. The prompt-length guidance
node also remains a pre-existing failure because the unchanged
`REQUIRED_SELECTION_GOAL_PROMPT` is 2,624 characters against its 2,600 target.
These failures are retained as evidence and are not represented as production
regressions fixed by changing expectations.

## Root-cause analysis

### P0: Appraisal state errors cross the producer contract boundary

`semantic_appraisal._appraise_semantic_item` performs canonical JSON parsing,
singular-item validation, cumulative-prefix validation, and then a producer-
local trial reduction. The strict reducers correctly raise
`CognitionStateError` when a valid-looking item is incompatible with the
current state, for example a resolved knowledge gap, terminal event, or
retention-capacity conflict. `CognitionStateError` inherits from `ValueError`.
The broad producer contract handler therefore records these state conflicts as
model contract errors and spends the bounded regeneration attempt.

This is a latent boundary defect and repair-budget/family-availability
amplifier. It is not safe to call it the dominant post-fix fatal classifier
from this pre-fix corpus. The reducer guards in `state_reducers.py` and
`transition_guards.py` are the correct authority and remain strict.

### P1: Appraisal path projection encourages composition under attention pressure

The current prompt separates role and evidence domains, but grouped state-path
domains and large handle lists still leave the model to compose
`state_field.handle.axis` strings. The observed malformed prefixes and
cross-domain handles are contract violations, not authorization for aliases or
path rewriting. Adding full exact-path lists without synchronous budget
fitting would create a second failure: prompt overflow or truncation near the
20,000-character limit.

### P2: Goal-bid projection is not compactly mode-local

The initial prompts already state exact fields, descriptor-only confidence,
relational pairings, and `selection_required`. The missing element is a
compact machine-readable projection of the active mode's exact contract beside
the current allowed/required handle sets. This leaves the model repeatedly
omitting selected-operation fields, using role handles as evidence, selecting
absent roles instead of empty arrays, emitting non-sensitive relational tuples,
and exhausting the bounded repair loop in dense evidence cases.

The invalid seeded `embedded_target_role` failures occur while constructing a
required response-operation fixture before the goal model can repair the bid.
The execution ledger must classify these as fixture/producer-contract defects
or model behavior from the raw input/output, while retaining them as fixed-
denominator hard failures.

### Ownership boundary

- LLM stages own semantic Appraisal, character judgment, goal intention,
  relational applicability/state/stance, and bid semantics.
- Deterministic cognition owns canonical parsing, exact structure, domains,
  evidence binding, state safety, bounded attempts, persistence, and
  observability.
- The facade owns authoritative full Appraisal finalization and state commit.
- Workspace admits only complete live-eligible bids; action planning owns
  capability and execution routing.

## Proposed implementation

### P0 — typed Appraisal preflight degradation

Owners: `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py` and
`src/kazusa_ai_chatbot/cognition_core_v2/facade.py`.

1. Keep canonical `parse_llm_json_output`, singular item validation, handle /
   evidence / path binding, enums, bounds, and text limits in the producer.
2. Retain producer-local reduction as a non-authoritative compatibility
   preflight so accepted-prefix behavior does not change accidentally.
3. Catch `CognitionStateError` before the generic `ValueError` handler. Record
   a typed bounded state-incompatibility disposition containing question,
   item/finalization step, bounded error text, retry count, and final
   disposition. Do not ask the model to regenerate for this class.
4. Preserve an already accepted micro-item prefix. When the first item
   conflicts, omit that question family through a typed degradable result.
   The rejected item remains out of state, persistence, workspace, action, and
   delivery paths.
5. Keep `facade._reduce_appraisals_with_isolation` as the only authoritative
   full-question reduction/finalization and commit boundary. It preserves the
   accepted prefix and records `semantic_appraisal_reduction_rejected` for
   full finalization incompatibility.
6. Keep attempt caps, strict terminal transitions, resolved-gap guards,
   candidate-source checks, evidence-retention limits, and state-capacity
   checks unchanged in semantic authority.

### P1 — exact, cap-safe Appraisal domains

Owners: `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py` and
`src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py`.

- Add exact per-item `permitted_target_paths`, each retaining the complete
  `state_field.handle.axis` spelling.
- Make `_fit_appraisal_payload` prune state handles and their exact paths as a
  single unit. The projected contract must remain within the existing prompt
  budget after every pruning operation; no partial path, path alias, or silent
  evidence drop is allowed.
- Keep separate named `permitted_role_handles`, `evidence_handles`, and path
  domains with an explicit no-crossing rule.
- Keep `proposition` and `delta` as mandatory nullable singular members;
  unsupported values are `null`, not omitted or arrays.
- Preserve candidate-origin evidence mapping and strict source binding.
- Preserve the archived goal-outcome lifecycle filter and the current strict
  terminal guards. Do not introduce a compatibility vocabulary.

### P2 — dynamic goal-bid contract projection

Owner: `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` and its
direct contract tests.

- Keep stable mode schema and semantic ownership in the system prompt.
- Add a compact current-run `goal_output_contract` adjacent to the dynamic
  domain sets in the human payload for ordinary, non-ordinary,
  required-selection, and recurrence modes.
- Project exact top-level fields, types, bounds, and descriptor-only string
  confidence from the canonical validator.
- Project selected response operations with exactly six fields, including
  `selection_required`; reject omissions and unknown fields without
  deterministic backfilling.
- Project `allowed_role_handles`, `allowed_evidence_handles`,
  `required_evidence_handles`, and `current_episode_evidence_handles` as
  separate sets. A role handle cannot appear in an evidence array and an
  evidence handle cannot appear in a target-role array.
- State that unavailable target roles are represented by an empty target-role
  array, never by an invented alias.
- Project the relational-willingness pairings initially. Non-sensitive is
  exactly `not_relationship_sensitive/not_applicable/not_applicable`; the
  sensitive path uses a real relationship state plus one ordered stance.
- Carry validated relational willingness deterministically through recurrence;
  do not make recurrence regenerate an already validated relational tuple.
- Reuse the same dynamic contract in repair feedback, adding observed fields
  and types without changing semantic context or attempt caps.

### P3 — evidence ledger and review contract

Owners: plan execution harness, `debug-llm`, and independent GPT-5.6 Sol
review.

Create
`test_artifacts/diagnostics/cognition_v2_bidding_llm_evidence_ledger_20260811.md`
with exactly 71 rows. Every row must contain:

- exact pytest node ID and cohort;
- first-pass process identity and timestamp;
- pytest result and hard-contract result;
- explicit agent behavior verdict;
- durable artifact path and trace/correlation ID when produced;
- fixture/harness classification when the node cannot be attributed to model
  behavior; and
- branch-level results for every parallel or multi-call node.

Focused reruns are separate rows in an instability appendix and cannot replace
the first-pass 71-row denominator. Missing artifacts, skips, infrastructure
blocks, and pytest failures remain non-passes.

## Change surface and ownership

### Planned production files

- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`: typed
  preflight disposition and exact path projection / budget fitting.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`: preserve and expose the
  authoritative full-finalization boundary and typed rejection evidence.
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py`:
  preserve lifecycle filtering while supplying exact eligible paths.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: dynamic initial
  goal-output contract, repair reuse, selection and recurrence projections.

### Files explicitly kept strict

- `src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/transition_guards.py`

These files remain unchanged in semantic authority. No aliasing, terminal
normalization, evidence deletion, cap increase, or generic evaluator may be
added.

### Planned tests and artifacts

- `tests/unit/cognition_core_v2/test_semantic_appraisal.py`: first/later
  `CognitionStateError` disposition, retry count, prefix preservation, and
  first-family omission.
- `tests/test_cognition_core_v2_semantic_terminalization.py`: finalization
  rejection and cross-Appraisal composition.
- `tests/test_cognition_core_v2_trace_failure_mode_matrix.py`: terminal,
  resolved-gap, exact micro-field, evidence/role, and goal-bid matrix.
- `tests/unit/cognition_core_v2/test_semantic_source_planner.py`: exact-path
  domains, lifecycle eligibility, and cap-safe pruning.
- `tests/unit/cognition_core_v2/test_goal_cognition.py`: initial/repair
  contract equality across modes, six-field selection, role/evidence domains,
  relational pairings, and recurrence carry-forward.
- `tests/test_cognition_core_v2_prompt_budget_continuity.py`: near-cap exact
  path projection and revised typed error ordering, with no assertion broadening.
- `tests/test_cognition_core_v2_failures.py`: typed bounded dispositions and
  fail-closed downstream behavior.
- Exact integration regressions:
  `tests/unit/cognition_core_v2/test_workspace.py::test_workspace_collapse_does_not_rank_by_confidence_descriptor`;
  `tests/unit/cognition_core_v2/test_action_selection.py::test_selected_intention_preserves_selected_response_operation`;
  `tests/unit/cognition_core_v2/test_action_selection.py::test_non_accepting_stance_suppresses_downstream_effects`;
  `tests/test_cognition_core_v2_integration.py::test_stale_nonordinary_goal_bid_is_dropped_before_collapse`;
  `tests/test_cognition_core_v2_integration.py::test_v2_facade_commits_before_surface_and_preserves_complete_bid`.
- Human-readable RCA, review, raw exports, canonical inventory, and the
  71-row ledger under `test_artifacts/diagnostics/`.

## Test Impact And Traceability

| Source path and changed contract | Semantic owner | Exact deterministic nodes | Live or boundary nodes | Regression prevented |
|---|---|---|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`: typed state-incompatibility disposition, singular item contract, exact-path projection, and cap fitting | semantic Appraisal producer | `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_semantic_appraisal_exposes_owned_contract`; `tests/test_cognition_core_v2_semantic_terminalization.py::test_appraisal_classifies_state_incompatibility_without_retry`; `tests/test_cognition_core_v2_semantic_terminalization.py::test_appraisal_preserves_accepted_prefix_before_later_state_conflict`; `tests/test_cognition_core_v2_prompt_budget_continuity.py::test_appraisal_exhaustion_returns_the_accepted_prefix`; `tests/test_cognition_core_v2_prompt_budget_continuity.py::test_captured_near_cap_repairs_fit_the_existing_budget` | model contract errors consuming retries for deterministic state conflicts, malformed composed paths, and non-singular micro-items |
| `src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py`: lifecycle-eligible candidates and exact delta-path domains | semantic source planner | `tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_goal_outcome_filters_terminal_handles_and_delta_paths`; `tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_semantic_source_planner_exposes_owned_contract`; `tests/test_cognition_core_v2_prompt_budget_continuity.py::test_irreducible_appraisal_context_is_omitted_per_question` | `tests/test_cognition_core_v2_appraisal_contract_live_llm.py::test_live_appraisal_projects_exact_paths_without_crossing_domains` | terminal or unowned state paths entering the model-facing question domain |
| `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: dynamic mode-local goal contract, six-field selected operation, domain sets, relational pairings, and recurrence carry | goal-bid producer | `tests/unit/cognition_core_v2/test_goal_cognition.py::test_goal_cognition_exposes_owned_contract`; `tests/unit/cognition_core_v2/test_goal_cognition.py::test_required_selection_emits_selected_response_operation`; `tests/unit/cognition_core_v2/test_goal_cognition.py::test_goal_prompt_labels_confidence_as_descriptor`; `tests/test_cognition_core_v2_prompt_budget_continuity.py::test_required_selection_budget_preserves_mandatory_evidence`; `tests/test_cognition_core_v2_prompt_budget_continuity.py::test_required_selection_regeneration_feedback_counts_toward_cap` | `tests/test_cognition_core_v2_goal_capability_live_llm.py::test_live_captured_online_search_goal_preserves_required_evidence`; all 13 relational-willingness, 20 required-selection, and 35 action-planning/authorization nodes in the Stage 2 cohort | omitted dynamic fields, role/evidence handle crossover, invented target aliases, invalid relational tuples, and selected-operation wrapper repetition |
| `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`: authoritative appraisal reduction isolation and downstream admission | cognition facade and finalization boundary | `tests/test_cognition_core_v2_semantic_terminalization.py::test_final_reduction_isolates_one_residual_invalid_appraisal`; `tests/test_cognition_core_v2_semantic_terminalization.py::test_final_reduction_preserves_cross_appraisal_composition`; `tests/test_cognition_core_v2_integration.py::test_v2_facade_commits_before_surface_and_preserves_complete_bid`; `tests/test_cognition_core_v2_integration.py::test_appraisal_retry_or_omission_preserves_cognition`; `tests/test_cognition_core_v2_failures.py::test_appraisal_collection_records_original_failure_cause` | exact 21-node residual Appraisal cohort in Stage 4; 2 workspace-collapse nodes in Stage 2 | rejected appraisal candidates reaching state, persistence, workspace, action, or delivery |

The four planned production paths above are the complete Luna ownership set.
`state_reducers.py` and `transition_guards.py` remain strict and unchanged in
semantic authority. The existing test-only changes are pre-execution baseline
work and remain part of the parent review surface.

## Historical execution results

The implementation and verification record is now available:

- [71-row bidding ledger](../../../test_artifacts/diagnostics/cognition_v2_bidding_llm_evidence_ledger_20260811.md)
  contains exactly 71 original first-pass rows. The corrected immutable
  revision is [ledger revision 2](../../../test_artifacts/diagnostics/cognition_v2_bidding_llm_evidence_ledger_20260811_revision2.md).
- The mapped deterministic nodes for the four production files passed. The
  historical broader affected deterministic run was 152 passed, 2 skipped,
  and 7 failed; those failures were the stale near-cap and live-DB fixture
  cases described in the evidence baseline. The separate prompt-length
  guidance node also failed on its pre-existing 2,600-character target. After
  the final Sol amendments and the dependent test-contract update, the final
  selected affected deterministic suite is **160 passed, 4 deselected**;
  `py_compile` and `git diff --check` also pass.
- The original primary live cohort was 65/71 (91.55%). After correcting the
  three executable fixture defects and rerunning all 71 nodes, revision 2 is
  70/71 (98.59%): goal capability 1/1, relational willingness 13/13, required
  selection 20/20, action planning/authorization 34/35, and workspace collapse
  2/2.
- The corrected required-selection fixtures now pass with model calls and
  contract evidence. The sole primary action failure remains the known C03
  local-recall model-quality failure; the action floor still passes.
- The supplemental captured replay cohort is 0/4. Three samples returned
  `deflect` against fixture expectations of `reject`; the fourth lacked its
  historical capture export.
- The separate 21-node Appraisal replay cohort is not valid executable replay
  evidence: historical exports are missing for most nodes, the A1a573 replay
  asserts an obsolete error ordering, two capacity nodes are explicit skips,
  two exhaustion nodes lack `target_2304_llm_trace.json`, and three near-cap
  replays no longer reproduce their captured initial failures. These are
  blocked/non-pass results, never passes.
- The first [independent GPT-5.6 Sol execution review](../../../test_artifacts/diagnostics/cognition_v2_bidding_llm_test_execution_20260811_sol_review.md)
  returned **NOT SIGNED OFF — FAIL** and its implementation amendments were
  applied by the bounded Luna remediation. The historical [GPT-5.6 Sol review](../../../test_artifacts/diagnostics/cognition_v2_bidding_llm_test_execution_20260811_final_sol_review.md)
  recorded the pre-recovery gate failure. The final [GPT-5.6 Sol signoff](../../../test_artifacts/diagnostics/cognition_v2_remaining_appraisal_and_goal_bid_sol_review_20260812_final.md)
  reviewed the frozen recovery evidence and returned **PASS — SIGN OFF**.

## Execution amendment — final recovery and verification 2026-08-12

The historical results above remain the pre-recovery evidence record. The
recovery execution is frozen in the following artifacts:

- [immutable primary 71-row ledger](../../../test_artifacts/diagnostics/cognition_v2_bidding_llm_evidence_ledger_20260812_final.md)
- [supplemental captured replay ledger](../../../test_artifacts/diagnostics/cognition_v2_supplemental_captured_replay_evidence_20260812.md)
- [prompt-coupled live evidence](../../../test_artifacts/diagnostics/cognition_v2_prompt_coupled_live_review_20260812.md)
- [final Sol review](../../../test_artifacts/diagnostics/cognition_v2_remaining_appraisal_and_goal_bid_sol_review_20260812_final.md)
- [final evidence hash manifest](../../../test_artifacts/diagnostics/cognition_v2_final_evidence_hash_manifest_20260812.txt)

The remaining failure modes were handled through the owning semantic and
boundary contracts. Required selection with no admitted bid now fails closed
at `workspace_collapse` with the typed
`required_selection_without_admitted_bid` error before action planning,
committable output, L3 projection, or delivery. Rejected Appraisal candidates
remain outside state, persistence, workspace, action, and delivery; the 21
residual nodes use the public `facade.run_cognition` replay boundary with
commit/L3 spies and before/after control comparison. State-incompatibility
dispositions are typed no-repair outcomes.

### Final live and deterministic gates

| Gate | Result |
|---|---:|
| goal capability | 1/1 |
| relational willingness | 13/13 |
| required selection | 20/20 |
| action planning/authorization | 35/35 |
| workspace collapse | 2/2 |
| **primary bidding cohort** | **71/71 (100%)** |
| supplemental captured replays | 4/4 |
| prompt-coupled live cohort | 13/13 |
| residual Appraisal cohort | 21/21 |
| focused deterministic suite | 147 passed, 4 deselected |
| full eligible deterministic suite | exited 0; 4457 collected, 1182 deselected |
| `py_compile` / `git diff --check` | clean |

The required primary threshold is 68/71 (95%); the frozen result exceeds it
by three nodes. The 71-node runner used one fresh process per exact node with
no retry. The multi-call branches are represented explicitly in the ledger.

### Final review disposition

GPT-5.6 Sol returned **PASS — SIGN OFF** and identified no remaining technical
remediation or blocking correction. The active plan therefore moves to the
completed lifecycle archive.

## Verification plan

### Stage 0 — preflight and ledger preparation

1. Capture `git status --short`, current source/test baseline, and the exact
   71-node collection output.
2. Keep the raw daily exports immutable and regenerate the canonical inventory
   from the raw capsule export. The two sets must compare exactly as 62 + 47 =
   109 and remain disjoint.
3. Create the 71-row ledger schema before live execution. Do not claim a node
   pass from a console summary without a durable artifact or an explicit
   harness classification.

### Stage 1 — deterministic implementation gates

Run the focused deterministic suite with
`venv\\Scripts\\python.exe`. Every required node must pass after the change.
The three current near-cap failures are recorded as baseline evidence and may
be updated only to assert the new typed contract disposition and exact error
ordering.

### Stage 2 — exact primary bidding cohort

Execute exactly these 71 live nodes, one process at a time, with output
inspected before advancing:

- 1 goal capability node from
  `tests/test_cognition_core_v2_goal_capability_live_llm.py`;
- all 13 relational-willingness nodes from
  `tests/test_cognition_core_v2_relational_willingness_live_llm.py`;
- all 20 required-selection nodes from
  `tests/test_cognition_core_v2_required_selection_live_llm.py`;
- all 35 action-planning/authorization nodes from
  `tests/test_cognition_core_v2_action_planning_live_llm.py`; and
- both workspace-collapse nodes from
  `tests/test_cognition_core_v2_workspace_live_llm.py`.

The exact node list is captured by `--collect-only -m live_llm` and copied into
the 71-row ledger. A multi-call node passes only when every branch passes.

### Stage 3 — supplemental goal replay cohort

Run separately and require 4/4 hard-gate passes:

- `tests/test_cognition_core_v2_captured_goal_failure_live_llm.py::test_captured_goal_replay_sample_1_rejects`
- `tests/test_cognition_core_v2_captured_goal_failure_live_llm.py::test_captured_goal_replay_sample_2_rejects`
- `tests/test_cognition_core_v2_captured_goal_failure_live_llm.py::test_captured_goal_replay_sample_3_rejects`
- `tests/test_cognition_core_v2_captured_run_failures_live_llm.py::test_captured_run_goal_relational_willingness_repair_live_llm`

Any `deflect`/`reject` fixture expectation must be justified by the source
contract before the row can be marked hard-contract failure.

### Stage 4 — exact residual Appraisal cohort

The Appraisal replay denominator is exactly 21 nodes, separate from the 71
bidding nodes. Refresh or provide the historical capture inputs before
execution; missing fixtures are blocked failures, not skips or passes.

The 16 trace-failure nodes are:

- `test_semantic_delta_path_not_owned_live_llm`
- `test_selected_evidence_unknown_handle_live_llm`
- `test_terminal_event_transition_rejected_live_llm`
- `test_candidate_origin_evidence_missing_live_llm`
- `test_semantic_role_value_invalid_live_llm`
- `test_current_run_event_agency_role_value_invalid_live_llm`
- `test_a1a573_goal_threat_unowned_path_live_llm`
- `test_resolved_knowledge_gap_transition_rejected_live_llm`
- `test_selected_roles_unknown_handle_live_llm`
- `test_semantic_proposition_subject_kind_mismatch_live_llm`
- `test_semantic_proposition_object_handle_not_permitted_live_llm`
- `test_delta_reason_invalid_live_llm`
- `test_semantic_delta_type_invalid_live_llm`
- `test_semantic_micro_appraisal_fields_not_exact_live_llm`
- `test_captured_trace_8d0d4295_capacity_path_live_llm`
- `test_captured_trace_9164e957_capacity_path_live_llm`

The 5 exhaustion nodes are:

- `test_moral_identity_contract_exhaustion_live_llm`
- `test_goal_threat_outcome_contract_exhaustion_live_llm`
- `test_a1a573_near_cap_semantic_repair_reaches_live_llm`
- `test_caad1a_near_cap_semantic_repair_reaches_live_llm`
- `test_df6eb4_near_cap_semantic_repair_reaches_live_llm`

All 21 must either produce the new bounded typed Appraisal disposition or
complete with the expected valid semantic result. No state, persistence,
workspace, action, or delivery mutation may occur on a rejected candidate.

### Stage 5 — independent agent evaluation and signoff

Forward the completed ledger and durable artifacts to GPT-5.6 Sol for explicit
per-node and per-cohort behavior verdicts. The evaluator must preserve:

- fixed 71-node primary denominator;
- 71/71 harness and hard-contract zero-tolerance;
- at least 68/71 explicit agent PASS;
- floors of 1/1, 13/13, 19/20, 34/35, and 2/2;
- 4/4 supplemental hard-gate passes; and
- a valid 21/21 Appraisal replay cohort.

The implementation is not complete when only the aggregate percentage passes;
any hard-contract, unauthorized ownership transfer, unbounded retry, stale-
goal admission, role/evidence leak, or invalid relational tuple fails signoff.

## Acceptance criteria

The plan may move from draft to execution only after explicit user direction
and the required plan lifecycle approval. The implementation may move to
complete only when all of the following hold:

1. Deterministic source and integration tests pass, including the new typed
   Appraisal dispositions and exact prompt-budget assertions.
2. The 71-row ledger is complete and auditable, with all 71 nodes attempted
   one at a time and no skipped, missing, infrastructure-blocked, or pytest-
   failed row counted as pass.
3. GPT-5.6 Sol records at least 68/71 explicit behavior passes and every
   cohort floor passes.
4. Every hard-contract and harness gate passes with zero tolerance.
5. All four supplemental replays pass without inflating the primary score.
6. All 21 separate Appraisal nodes provide valid executable evidence and pass
   their bounded disposition/side-effect contract.
7. No change weakens `state_reducers.py` or `transition_guards.py`, changes
   semantic ownership, increases retries/caps, introduces aliases, or writes
   a rejected candidate to downstream state.

## Rejected alternatives and out of scope

- Removing the producer preflight entirely: it could change accepted-prefix
  behavior and discard a whole family because of one later incompatible item.
- Treating `CognitionStateError` as a generic LLM contract error or spending a
  regeneration attempt on it.
- Increasing any attempt or context cap to improve the percentage.
- Rewriting paths, aliasing handles, dropping evidence, normalizing terminal
  state, or inventing target roles deterministically.
- Adding a generic LLM judge, semantic refusal gate, compatibility mapper, or
  parallel contract vocabulary.
- Declaring the current 59/71 baseline, stale replay harness, or skipped
  capacity nodes as a pass.
- Production implementation, test-source edits, fixture refresh, database
  migration, or deployment in this draft-plan turn.

## Execution Handoff

- Lifecycle transition: `draft` -> `in_progress`, authorized by the user's
  explicit execution request on 2026-08-11, then `in_progress` -> `completed`
  after the frozen gates and GPT-5.6 Sol signoff on 2026-08-12.
- Fixed executor: GPT-5.6 Luna (`gpt-5.6-luna`), reasoning effort `max`,
  normal/default service tier, runtime agent `Averroes`
  (`019fef6c-8575-71d1-9e0e-816886bd9d04`). The fixed executor owns only the
  four production paths listed in the Test Impact And Traceability table.
- Parent authority: preserve the pre-existing worktree changes, inspect the
  production diff, run the mapped deterministic nodes, run the required
  one-at-a-time live cohorts, and record the ledger and independent review.
- Baseline before production handoff: modified
  `development_plans/README.md`,
  `tests/test_cognition_core_v2_goal_capability_live_llm.py`, and
  `tests/test_cognition_core_v2_required_selection_live_llm.py`; untracked
  `tests/test_cognition_core_v2_appraisal_contract_live_llm.py` and this plan.
  No planned production source path was modified at handoff.
- Entry gate: preserve all strict reducers, transition guards, attempt caps,
  semantic ownership, and the test-only live probes. Exit gate: source diff
  matches the fixed change direction, every mapped deterministic node is
  collected and passing, and no live or independent-review acceptance gate is
  represented as passing without evidence. This exit gate is satisfied by the
  final evidence and signoff artifacts linked above.
