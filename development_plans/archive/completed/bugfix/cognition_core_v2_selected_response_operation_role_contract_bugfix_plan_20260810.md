# cognition core v2 selected response operation role contract bugfix plan 20260810

## Summary

- Goal: prevent required-selection dialog turns from exhausting all dialog
  candidates when the selected concrete action has a different actor/target
  direction from the input-level response operation.
- Plan class: cross-stage semantic contract bugfix covering decontextualization,
  Cognition V2 selection, L3 handoff, and dialog role verification.
- Status: completed.
- Scope boundary: repair the typed role-operation contract and its propagation;
  preserve the completed dialog score-bidding policy and all existing hard
  gates.
- Change direction: retain the input-level `response_operation` as episode
  provenance and add a canonical post-selection
  `selected_response_operation` carried from required-selection goal cognition
  to the dialog role verifier.
- Acceptance state: the delegated implementation and review gates are closed.
  The GPT-5.6 Sol final review returned `APPROVE`; its required corrections
  were incorporated and the final scoped checks passed.

## RCA And Review Disposition

The protected trace for
`chat:qq:ch_73987da21ae6b88a:798924750` and
`llmtrace_f92ea2563c6a42078d620c21b57e7740` shows that the dialog score plan
was active. The runtime made three bounded dialog-generation attempts and
raised `dialog generator exhausted candidates without an eligible candidate`.
Each candidate received the hard issue
`typed_operation_role_reversal`, so `_select_best_dialog_candidate` had no
eligible record to rank. The `0.50` threshold did not cause this failure.

The episode carried this input-level operation:

```json
{
  "response_owner_role": "当前角色",
  "selection_owner_role": "当前角色",
  "selection_required": true,
  "embedded_actor_role": "当前角色",
  "embedded_target_role": "当前用户"
}
```

The cognition bid selected a reward action performed by the user for the
character. The selected action direction is therefore
`当前用户 -> 当前角色`, while the response and selection owners remain
`当前角色`. The current dialog verifier receives the episode operation and
does not receive the concrete operation selected by cognition. The failure is
a cross-stage authority collision: one tuple is being used for both the
outer response/selection act and the embedded action chosen inside that act.

The semantic verifier contract error during repair is secondary. The current
metadata-oriented trace does not expose enough raw response content to specify
its behavioral repair safely. This plan records a full-capture replay as a
diagnostic prerequisite and keeps any semantic-verifier behavior change out of
scope.

## Confirmed Decisions

- The dialog score threshold, geometric aggregation, three-attempt cap, hard
  role gate, and fail-closed exhaustion behavior remain unchanged.
- A role-direction hard issue remains ineligible for score ranking. Numeric
  scoring never overrides a typed ownership conflict.
- Semantic interpretation stays LLM-owned. Deterministic code validates,
  compares, carries, limits, and records typed fields; it does not infer roles
  from keywords or rewrite user input before or after an LLM stage.
- The new canonical field is named `selected_response_operation` and reuses
  the exact `DialogResponseOperation` shape.
- The active
  `cognition_surface_score_ranking_followup_plan_20260810.md` remains a
  separate draft and is not extended with this role-contract work.

## Scope And Change Direction

### Input-level operation

`response_operation` remains the decontextualizer-owned description of the
current episode. It records who responds, who owns an unspecified answer or
choice, and any embedded actor/target direction that the input itself fixes.
The prompt must distinguish an embedded action from wrapper verbs such as
“decide”, “say”, and “tell”. When the input does not fix an embedded action,
the operation keeps the corresponding role values at `无` instead of inventing
a direction from the response wrapper.

### Post-selection operation

Required-selection goal cognition emits `selected_response_operation` with the
concrete action represented by the selected goal. Its fixed ownership fields
for this incident are:

```json
{
  "response_owner_role": "当前角色",
  "selection_owner_role": "当前角色",
  "selection_required": true,
  "embedded_actor_role": "当前用户",
  "embedded_target_role": "当前角色"
}
```

The `operation` text describes the selected reward action and remains
model-owned. Deterministic validation enforces these invariants:

- `response_owner_role`, `selection_owner_role`, and
  `selection_required` remain equal to the input-level operation.
- Every non-`无` input actor or target role is preserved exactly.
- Cognition may resolve an input role that is `无` only when its selected
  action supplies that role; it may not replace a known role with another
  role.
- The selected operation is a complete validated object before it enters a
  bid, intention, surface input, or dialog verifier.
- Actor and target remain independently valid role fields, matching the
  existing `DialogResponseOperation` contract. Actionless selections use
  `无` for both fields; one-endpoint actions may keep `无` for the ungrounded
  endpoint. No new pairing rejection is introduced.

### Propagation and authority

The selected operation travels through the following single semantic path:

```text
episode response_operation
  -> required_selection_operations
  -> goal bid selected_response_operation
  -> selected intention selected_response_operation
  -> TextSurfaceInputV2 selected_response_operation
  -> dialog role-direction verifier
```

The surface model receives the existing prompt projection and does not get to
rewrite this control-only carrier. The dialog verifier consumes the selected
operation as its role-direction authority for required-selection turns. A
missing or conflicting selected operation fails at the Cognition boundary
after the existing bounded regeneration policy, rather than consuming dialog
attempts with stale role evidence.

## Mandatory Skills

- `development-plan`: plan lifecycle, exact source-to-test traceability,
  approval gates, and execution evidence.
- `local-llm-architecture`: LLM semantic ownership, deterministic validation,
  bounded latency, and stage responsibility.
- `no-prepost-user-input`: preserve LLM-first user-input interpretation and
  prohibit deterministic semantic correction.
- `llm-trace-debug`: protected trace retrieval, full-capture replay, and
  correlation of upstream and dialog evidence.
- `debug-llm`: human-readable trace and live-LLM regression artifacts.
- `py-style`: every Python source or test edit.
- `cjk-safety`: every Python prompt or fixture edit containing CJK text.
- `test-style-and-execution`: deterministic owner tests and one-at-a-time live
  LLM execution.

## Mandatory Rules

- Production implementation requires explicit user authorization and this
  plan promoted to `approved` or `in_progress`. Draft status authorizes
  planning and evidence definition only.
- Capture the execution baseline and explicitly owned file set before any
  implementation. Preserve the existing worktree changes.
- Use `venv\\Scripts\\python.exe` for Python and pytest commands. Use
  `apply_patch` for manual edits. Keep `.env` outside inspection.
- Use `parse_llm_json_output(...)` for all new or changed LLM JSON boundaries.
  Do not add a stage-local parser, keyword classifier, semantic normalizer, or
  compatibility alias.
- Keep the existing V2 attempt cap for goal cognition and the existing dialog
  producer/verifier cap. Invalid or conflicting selected operations stay out
  of dialog, persistence, scheduling, and delivery.
- Record expected/actual role fields, producing stage, attempt, and
  disposition in sanitized diagnostics. Protected trace evidence retains both
  the input-level and selected operations when full capture is enabled.
- Run live cases one at a time and inspect each result and its human-readable
  review artifact before accepting it as evidence.

## Must Do

1. Attempt one protected full-trace replay of the supplied input. If the
   historical trace exposes the raw semantic-verifier response, record it; if
   its metadata capture cannot recover that response, record the limitation and
   run a fresh self-contained semantic-verifier control with full capture. If
   either path reveals an independent verifier defect, record it as separate
   triage evidence without changing verifier behavior in this plan.
2. Update the decontextualizer prompt contract and deterministic schema
   guidance so wrapper response acts and embedded action roles are distinct.
   Add the reward-offer example and an actionless-selection example.
3. Add and validate `selected_response_operation` in required-selection goal
   output. Preserve fixed non-`无` roles, carry the field through the complete
   bid and selected intention, and fail closed after the current bounded
   regeneration cap when the field is malformed or conflicting.
4. Carry the selected operation into `TextSurfaceInputV2`, validate it at the
   L3 boundary, omit it from model-facing surface projection, and pass it to
   dialog role verification unchanged.
5. Change only dialog role-operation authority. Keep the score threshold,
   aggregate calculation, hard issue classification, repair ownership,
   candidate ledger, and exhaustion selection behavior unchanged.
6. Add deterministic owner tests and cross-boundary propagation tests from
   decontextualization through dialog. Retain regression coverage for true
   selection-owner transfer, true actor/target reversal, unavailable semantic
   verification, below-threshold eligible candidates, score tie-breaking, and
   all-hard-invalid exhaustion.
7. Add a one-at-a-time live regression for the supplied reward-offer input and
   produce a human-readable review artifact containing the input operation,
   selected operation, candidate verdicts, and delivery outcome.
8. Update the cognition and nodes README contracts and extend
   `tests/ownership/source_test_impact_manifest.json` for the newly governed
   `cognition_episode.py` and decontextualizer source boundaries.

## Progress Checklist

- [x] Independent Sol RCA/proposal review completed and incorporated.
- [x] User authorized implementation and delegated review sequence.
- [x] Baseline status, required documentation, owned production paths, and
  source hashes captured.
- [x] Luna production implementation complete within the owned source set.
- [x] Parent deterministic, patched-handoff, and integration tests complete.
- [x] Terra independent code review complete and findings resolved.
- [x] Sol final review complete and acceptance decision recorded.
- [x] Full-capture semantic-verifier control and explicit historical-replay
  triage artifact complete.
- [x] Plan acceptance evidence and lifecycle registry closed.

## Deferred

- Any semantic-verifier prompt, parser, retry, or fallback behavior change
  until the full-capture replay identifies the exact malformed contract.
- Any dialog score threshold, aggregation, hard-gate, attempt-cap, or
  candidate-selection policy change.
- Any generic cross-owner retry/scoring abstraction or compatibility shim.
- Any deterministic parsing or rewriting of the user’s Chinese input.
- Changes to adapters, RAG, memory, persistence, queue intake, delivery,
  scheduler, database schema, or action authorization.
- Extending the active surface score-ranking draft with role-direction scope.

## Execution Evidence

- Luna completed the production implementation within the explicitly owned
  eight-source write set. The parent added and reviewed the deterministic
  contract, propagation, prompt-boundary, role-verifier, integration, live,
  documentation, and ownership tests.
- The final scoped deterministic suite passed: `143 passed` across the V2
  unit, node, prompt, decontextualizer, dialog, L2D/L3 handoff, terminal
  integration, and test-impact manifest nodes.
- The neighboring deterministic slice passed `632`, skipped `2` trace-inventory
  cases, and exposed four unrelated database smoke failures. All four fail in
  `seed_shared_documents` because the live test database contains a
  `character_state:{'_id': 'global'}` content-hash mismatch; the other 20
  integration tests in that file pass. The two known failures in
  `tests/test_cognition_core_v2_live_character_judgment.py` remain outside
  this diff and concern pre-existing prompt-length/wording assertions.
- One-at-a-time live evidence passed for decontextualization, selected
  dialog-role verification, terminal-candidate degraded delivery, and the
  required-selection goal producer. The goal producer initially exposed
  handle-token role output and then a generic operation wrapper; after the
  prompt contract correction, the final full-capture run passed after one
  bounded confidence repair with concrete selected operation text and correct
  roles. The fresh full-capture semantic control passed with one raw
  semantic-verifier call, one surface-verifier call, and a typed semantic
  hard-error verdict.
  The original seeded semantic control could not start because its unrelated
  `personalities/kazusa.json` fixture is absent; the self-contained control is
  recorded in `test_artifacts/llm_reviews/dialog_selected_response_operation_review_20260810.md`.
- The protected original exports contain 22 steps but no raw dialog semantic
  verifier response. The review artifact records that historical limitation,
  the fresh control evidence, and the separate triage disposition.
- Final closure checks passed: the scoped deterministic suite is `143 passed`,
  `py_compile` passed for the eight changed production sources, the ownership
  manifest is valid JSON, and `git diff --check` returned no whitespace
  errors.

## Delegated Implementation And Review

- GPT-5.6 Luna, max reasoning, normal-speed service: production implementation
  completed with no edits outside the assigned source set.
- GPT-5.6 Terra, max reasoning, normal-speed service: `APPROVE WITH REQUIRED
  CORRECTIONS`; all closure-blocking findings were corrected and the affected
  deterministic/live nodes were rerun.
- GPT-5.6 Sol, high reasoning, normal-speed service: final `APPROVE`. The
  reviewer confirmed the four role-enum prompt paths, concrete selected
  operation text, production diagnostic projector reuse, fixed-field live
  assertions, propagation, unchanged score/gate behavior, and accurate plan
  evidence. Residual risks are model-compliance variance in active/recurrence
  live paths and the unavailable historical semantic-verifier payload; both
  are explicitly covered as evidence limitations.

## Final Review

- Final reviewer: GPT-5.6 Sol, high reasoning, normal-speed service tier,
  read-only.
- Verdict: `APPROVE` with no remaining code corrections.
- Closure basis: final deterministic suite, production compilation, ownership
  manifest validation, diff hygiene, and the documented live full-capture
  controls all passed. The original trace remains a diagnosis of the stale
  operation authority collision; it is not claimed as a byte-for-byte replay
  because the protected metadata lacks the raw semantic-verifier response.

## Target State

The canonical contracts have these conditional fields:

- `GoalBidDraftV2.selected_response_operation`: present for a
  required-selection draft; absent for branches without a selected operation.
- `ActionBidV2.selected_response_operation`: copied from the validated draft
  when present; never reconstructed from free-text intention.
- `SelectedIntentionV2.selected_response_operation`: copied from the admitted
  primary bid when present.
- `TextSurfaceInputV2.selected_response_operation`: copied from the selected
  intention and validated before surface planning and dialog.

The selected operation is control-only. Surface content planning may describe
the selected intention but cannot mutate the role carrier. Dialog receives the
selected operation for required-selection role verification and continues to
use the original episode evidence for semantic fidelity and other independent
checks.

## Change Surface

### Delete

None.

### Modify

- `src/kazusa_ai_chatbot/cognition_episode.py`: extend the canonical
  response-operation validation helpers with selected-operation preservation
  and conflict checks.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py`:
  clarify wrapper-versus-embedded-action semantics and add positive/negative
  prompt examples.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`: carry and validate
  the selected operation in goal bids, selected intention, and text-surface
  input.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: emit the field
  for required-selection output, validate it, preserve fixed roles, and carry
  it into `ActionBidV2`.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`: copy the
  admitted bid’s selected operation into `SelectedIntentionV2` unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: build and
  validate the selected-operation field in the canonical surface input.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface.py`: exclude the
  control-only selected operation from the model-facing surface projection.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: consume the selected
  operation for required-selection role verification and preserve all existing
  hard gates and score-bidding behavior.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md` and
  `src/kazusa_ai_chatbot/nodes/README.md`: document the two operation layers,
  ownership, propagation, and fail-closed behavior.
- `tests/ownership/source_test_impact_manifest.json`: register the direct
  `cognition_episode.py` and decontextualizer ownership entries and their
  exact deterministic nodes.
- Existing direct and integration test files named in the traceability matrix:
  extend them with the new role-carrier and propagation assertions.

### Create

- `tests/unit/cognition_core_v2/test_cognition_episode.py`: direct validator
  tests for selected-operation preservation and conflict rejection.
- `test_artifacts/diagnostics/dialog_selected_response_operation_replay_20260810.json`:
  full-capture replay evidence.
- `test_artifacts/llm_reviews/dialog_selected_response_operation_review_20260810.md`:
  human-readable live regression review.

### Keep

- `DIALOG_PASS_SCORE_THRESHOLD = 0.50`, geometric score aggregation, the
  three-attempt producer cap, hard role/actor-target gates, and deterministic
  tie-breaking.
- The completed dialog score-bidding plan as historical scope.
- The active surface score-ranking follow-up as a separate draft.
- Adapter delivery and all persistence, authorization, queue, and scheduler
  contracts.

## Contracts And Data Shapes

`selected_response_operation` uses the exact fields below and no aliases:

```json
{
  "operation": "当前用户为当前角色提供所选择的奖励",
  "response_owner_role": "当前角色",
  "selection_owner_role": "当前角色",
  "selection_required": true,
  "embedded_actor_role": "当前用户",
  "embedded_target_role": "当前角色"
}
```

The operation text is semantic content selected by the LLM. The four role
fields and boolean are deterministic contract data after validation. A
selected operation that changes a known upstream actor, target, response
owner, or selection owner is rejected as a typed contract error. A selected
operation that is missing for a required-selection bid is rejected before L3.

## Agent Autonomy Boundaries

The implementation agent may choose helper names, local function decomposition,
trace field placement, and test fixture construction when those choices retain
the exact field name, ownership rules, propagation path, and acceptance
criteria above.

The implementation agent must request a plan amendment before changing the
actor/target `无` pairing rule, adding a fallback that uses the stale
episode-level tuple, modifying the dialog hard gate, changing score selection,
repairing the semantic verifier, or expanding into another subsystem.

## Test Impact And Traceability

Each production semantic owner has a deterministic direct owner node. New
nodes below are exact names to create or preserve during implementation.

| Source or governed artifact | Changed symbol or contract | Semantic owner | Exact deterministic pytest node IDs | Supplemental / live nodes | Mode and regression prevented |
|---|---|---|---|---|---|
| `src/kazusa_ai_chatbot/cognition_episode.py` | `DialogResponseOperation` validation and selected-operation preservation | episode role-operation contract owner | `tests/unit/cognition_core_v2/test_cognition_episode.py::test_selected_response_operation_preserves_fixed_roles`; `tests/unit/cognition_core_v2/test_cognition_episode.py::test_selected_response_operation_rejects_conflicting_roles` | `none` | unit; prevents known actor/target or ownership fields being silently replaced |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py` | `response_operation` prompt contract | message decontextualization owner | `tests/test_msg_decontextualizer.py::test_decontextualizer_prompt_explains_reply_ellipsis_decision_owner` | `tests/test_decontextualizer_live_llm.py::test_live_decontextualizer_reward_offer_preserves_user_actor_character_target` | deterministic prompt contract plus one-at-a-time live; prevents wrapper “tell/decide” semantics from becoming the embedded action |
| `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` | `GoalBidDraftV2`, `ActionBidV2`, `SelectedIntentionV2`, `TextSurfaceInputV2`, and validators | cognition contract validator | `tests/unit/cognition_core_v2/test_contracts.py::test_selected_response_operation_has_exact_contract`; `tests/unit/cognition_core_v2/test_contracts.py::test_selected_response_operation_rejects_missing_required_fields` | `tests/test_l2d_l3_surface_handoff.py::test_selected_response_operation_reaches_dialog_unchanged`; `tests/integration/cognition_core_v2/test_terminal_dialog_candidate.py::test_reward_offer_required_selection_delivers_visible_dialog` | unit and integration; prevents field loss or shape drift across public V2 contracts |
| `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` | required-selection prompt, `validate_selection_goal_draft`, and bid mapping | goal cognition owner | `tests/unit/cognition_core_v2/test_goal_cognition.py::test_required_selection_emits_selected_response_operation`; `tests/unit/cognition_core_v2/test_goal_cognition.py::test_required_selection_rejects_fixed_role_conflict` | `tests/test_l2d_l3_surface_handoff.py::test_selected_response_operation_reaches_dialog_unchanged`; `tests/integration/cognition_core_v2/test_terminal_dialog_candidate.py::test_reward_offer_required_selection_delivers_visible_dialog` | unit and integration; prevents cognition from selecting an action whose typed roles conflict with the episode |
| `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py` | selected intention construction from admitted bid | action selection owner | `tests/unit/cognition_core_v2/test_action_selection.py::test_selected_intention_preserves_selected_response_operation` | `none` | unit; prevents deterministic selection from dropping or reconstructing the selected operation |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` | `build_text_surface_input_from_global_state` | L3 surface handoff owner | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_surface_preserves_selected_response_operation` | `tests/test_l2d_l3_surface_handoff.py::test_selected_response_operation_reaches_dialog_unchanged` | unit and integration; prevents L3 from delivering stale or incomplete role authority |
| `src/kazusa_ai_chatbot/cognition_core_v2/surface.py` | `_project_surface_payload` | surface prompt-boundary owner | `tests/unit/cognition_core_v2/test_surface.py::test_surface_prompt_omits_selected_response_operation` | `none` | unit; prevents a control carrier from being rewritten by surface generation |
| `src/kazusa_ai_chatbot/nodes/dialog_agent.py` | `_required_selection_role_operations`, `_verify_dialog_role_direction`, and dialog state handoff | dialog verifier owner | `tests/unit/nodes/test_dialog_agent.py::test_dialog_role_direction_uses_selected_response_operation`; `tests/unit/nodes/test_dialog_agent.py::test_dialog_role_direction_rejects_selected_actor_target_reversal`; existing `tests/test_dialog_agent.py::test_dialog_exhaustion_selects_highest_score_not_latest`; existing `tests/test_dialog_agent.py::test_dialog_exhaustion_all_unavailable_selects_latest_valid_candidate` | `tests/test_dialog_visible_speech_and_semantic_fidelity.py::test_role_direction_verifier_owns_required_selection`; `tests/test_dialog_visible_speech_and_semantic_fidelity_live_llm.py::test_live_reward_request_preserves_user_actor_character_target` | unit, integration, and one-at-a-time live; prevents the score layer from ranking stale-role candidates while preserving hard reversal rejection |
| `src/kazusa_ai_chatbot/cognition_core_v2/README.md` and `src/kazusa_ai_chatbot/nodes/README.md` | operation-layer ownership and propagation documentation | architecture/documentation owner | `tests/test_cognition_core_v2_prompt_contract_guidance.py::test_selected_response_operation_contract_is_documented` | `none` | deterministic documentation contract; prevents future stages from conflating input and selected operations |
| `tests/ownership/source_test_impact_manifest.json` | strict source-root and owner mapping | test ownership registry | `tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary`; `tests/test_test_impact_manifest.py::test_stale_required_node_fails_closed` | `none` | unit; prevents changed semantic owners from losing exact direct tests |
| `test_artifacts/diagnostics/dialog_selected_response_operation_replay_20260810.json` | full-capture replay evidence | trace-debug evidence owner | `tests/test_llm_trace_export.py::test_build_trace_export_groups_failure_capsules` | `tests/test_llm_trace_export.py::test_build_trace_export_uses_trace_id` | deterministic export plus live evidence; identifies the secondary verifier contract defect without prescribing an unsupported fix |

Existing score-bidding, role-transfer, actor-target-reversal, semantic-unavailable,
and dialog-exhaustion nodes remain regression gates even when their files are
not modified.

## Verification

- Before implementation, record `git status --short`, the exact owned file
  set, and the current deterministic baseline.
- Run every exact deterministic node in the matrix and confirm collection by
  node ID. A broader passing suite does not replace a missing mapped node.
- Run the decontextualizer live case, the cognition-to-dialog replay, and a
  semantic-verifier full-capture control one at a time. Inspect the raw result,
  sanitized events, protected trace, and human-readable review artifact after
  each case. If the supplied historical trace has no raw semantic-verifier
  payload, record that limitation and keep the control evidence separate from
  the historical replay claim.
- Verify the supplied input reaches visible dialog with
  `response_owner_role=当前角色`, `selection_owner_role=当前角色`,
  `embedded_actor_role=当前用户`, and `embedded_target_role=当前角色` in the
  selected operation.
- Verify malformed, missing, and conflicting selected operations exhaust at
  the Cognition boundary within the existing cap and never reach dialog.
- Verify true selection-owner transfer and true actor/target reversal remain
  hard-ineligible, while a soft below-threshold candidate without hard issues
  remains eligible for the existing score fallback.
- Run the source/test ownership validation and `git diff --check` before plan
  handoff.

## Acceptance Criteria

- The supplied reward-offer role direction reaches visible dialog through the
  decontextualizer live case and the cognition-to-dialog integration path. The
  historical trace cannot be replayed byte-for-byte because its dialog
  semantic-verifier response is absent from protected metadata; this is
  recorded as an evidence limitation rather than an unverified replay claim.
- The selected operation records current-character response and selection
  ownership and the user-to-character embedded reward direction.
- Every known input role is preserved; only an input role explicitly marked
  `无` can be resolved by the selected-operation LLM stage.
- Missing, malformed, or conflicting selected operations regenerate within the
  existing cognition cap and fail closed before dialog after exhaustion.
- Existing dialog hard gates reject genuine ownership transfer and
  actor/target reversal, regardless of numeric score.
- Existing score fallback still selects an eligible candidate below the
  threshold when no hard issue exists and still honors deterministic tie rules.
- Sanitized diagnostics identify the expected and actual role fields and
  disposition; protected full-capture evidence retains the input and selected
  operations.
- The secondary semantic-verifier contract failure has either a captured raw
  historical replay or an explicit unavailable-replay disposition backed by a
  fresh full-capture control; this plan does not claim a behavioral fix for it.
- No adapter, database, persistence, queue, delivery, RAG, authorization, or
  scheduler behavior changes.

## Execution Decisions And Closure

- The user explicitly authorized execution, and the plan is now `completed`.
- The actor/target decision is resolved by retaining independent enum fields:
  actionless operations use `无`/`无`, and one-endpoint operations preserve
  `无` only for the ungrounded endpoint.
- The historical raw semantic-verifier replay is unavailable in the supplied
  trace metadata. A fresh full-capture semantic-verifier control and its
  explicit triage disposition satisfy the diagnostic checkpoint; if it
  identifies a separate producer/evaluator contract defect, record the
  evidence and open a separate plan rather than expanding this bugfix.

## Independent Plan Review

- Reviewer: GPT-5.6 Sol, xhigh reasoning, normal-speed service tier, read-only.
- Review result: `APPROVE WITH REQUIRED CORRECTIONS`.
- Incorporated corrections: distinguish pre-selection and post-selection
  operations; add `selected_response_operation` to the bid/intention/surface
  carrier; make dialog consume the post-selection authority; add exact
  propagation and negative tests; keep the semantic-verifier issue separate;
  and register the added source boundaries.
