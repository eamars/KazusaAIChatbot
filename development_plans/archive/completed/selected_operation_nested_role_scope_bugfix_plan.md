# selected operation nested role scope bugfix

## Summary

- Goal: stop semantically correct dialog from being rejected when a character-owned agreement, request, or condition contains a nested action performed by another role.
- Status: completed
- Scope boundary: required-selection goal prompt semantics, dialog role-direction prompt semantics, their ownership documentation, exact mapped unit tests, and captured-trace live regressions.
- Change direction: bind `selected_response_operation` to the concrete embedded action after wrapper verbs are removed, and reject role direction only when that same selected action is unambiguously reversed.
- Acceptance state: user authorized production implementation, fixed DeepSeek Flash execution, and parent-agent sign-off.

## Confirmed Decisions

1. `response_owner_role` and `selection_owner_role` identify who responds and chooses; they do not automatically identify the actor of a nested requested action.
2. `selected_response_operation.operation` describes one concrete selected embedded action after agreement, selection, telling, requesting, confirming, desire, and condition wrappers are removed.
3. A character-owned request such as “I want you to do X to me” keeps the character as response and selection owner while binding the current user as actor of X and the character as target of X.
4. A candidate may contain compatible wrapper speech, conditions, consequences, and secondary actions whose grammatical actor differs from the selected embedded action. Those additions are not role reversal by themselves.
5. The role verifier reports `typed_operation_role_reversal` only when the candidate unambiguously assigns the same selected embedded action to the opposite actor/target pair.
6. Ambiguous, omitted, multi-clause, request, desire, imperative, negotiation, and condition readings receive the existing high-score treatment unless candidate text proves a reversal.
7. `DIALOG_PASS_SCORE_THRESHOLD`, attempt caps, typed schemas, validator ownership, hard-error eligibility, model routes, and deterministic fail-closed behavior remain unchanged.
8. No deterministic keyword parser, semantic postprocessor, compatibility path, alias, or fallback is introduced.

## Scope And Change Direction

The corrected flow is:

```text
required selection
  -> goal model removes response/request wrappers
  -> selected operation types the concrete embedded action
  -> dialog may express that action through desire/request/condition wording
  -> role verifier compares the same embedded action across the tuple and candidate
  -> only an unambiguous same-action actor/target reversal is a hard issue
```

This is a prompt-contract correction within existing semantic owners. Goal cognition continues to own selected operation meaning. Dialog role verification continues to own selection transfer and selected-action role reversal. Deterministic code continues to validate shape, bounds, provenance, eligibility, and delivery.

## Cutover Policy

Overall strategy: bigbang.

Replace the two prompt contracts and their tests directly. Preserve no legacy wrapper-level interpretation and add no alternate evaluator mode.

## Mandatory Skills

- `development-plan` for scope, execution handoff, exact traceability, and sign-off.
- `py-style` for every Python change.
- `cjk-safety` for Chinese prompt and test literals.
- `test-style-and-execution` for deterministic owner tests and one-at-a-time real-LLM regressions.
- `debug-llm` for captured-trace live artifacts and behavioral inspection.

## Mandatory Rules

- Keep LLM stages as semantic owners; deterministic code must not infer role direction from words or syntax.
- Preserve the canonical JSON parser and existing typed schemas.
- Keep prompt constants triple-single-quoted and adjacent to their existing model handler boundaries.
- Run Python syntax checks immediately after editing CJK-bearing Python files.
- Preserve every pre-existing and concurrent worktree change outside the owned files.
- Keep the patch surgical and avoid unrelated formatting, refactoring, or prompt rewrites.

## Must Do

1. Strengthen every required-selection goal prompt and repair instruction that governs `selected_response_operation` so wrapper and nested action actors cannot be conflated.
2. Strengthen the dialog role-direction prompt so compatible response wrappers, requests, desires, imperatives, negotiations, conditions, consequences, and secondary clauses do not become hard role reversals.
3. Retain strict rejection for explicit selection-owner transfer and unambiguous reversal of the same selected embedded action.
4. Add direct deterministic owner tests for both prompt contracts.
5. Update the source-test impact manifest with the new exact owner nodes.
6. Convert the three supplied trace replay cases from reproducing rejection to asserting acceptance above the existing pass threshold with no typed violation.
7. Run and inspect one captured-trace case at a time, then run one existing true-reversal live case to prove the mitigation remains bounded.
8. Update subsystem ownership documentation to state the wrapper-versus-embedded-action rule.

## Deferred

- New selected-operation schemas, multiple typed operation arrays, tuple confidence, or semantic provenance fields.
- Deterministic semantic validation of goal prose or monologue.
- Score-threshold, retry-budget, aggregation, delivery, persistence, consolidation, adapter, or model-route changes.
- Reconstruction of production dialog text that the metadata trace did not retain.
- Changes to unrelated relational-carrier recurrence work or task-resolution delivery work already present in the worktree.

## Target State

1. Goal prompts state that the nested predicate owns `embedded_actor_role` and `embedded_target_role`, while agreement/request/selection speech remains owned by response and selection fields.
2. Dialog role verification compares the typed tuple only to the same selected embedded action in candidate text.
3. The three exact preserved spans from `llmtrace_5eab4c8aee004cc59680fac47c0685d9` pass role-direction evaluation under their full character-owned request semantics.
4. Existing explicit same-action reversal and selection delegation cases remain rejected.
5. No production threshold or deterministic semantic behavior changes.

## Execution Roles

### implementation_owner

- Responsibility: implement the two prompt-contract corrections, documentation, mapped deterministic tests, and captured-trace live regression updates.
- Owned surface: `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`, `src/kazusa_ai_chatbot/nodes/dialog_agent.py`, `src/kazusa_ai_chatbot/cognition_core_v2/README.md`, `src/kazusa_ai_chatbot/nodes/README.md`, `tests/unit/cognition_core_v2/test_goal_cognition.py`, `tests/unit/nodes/test_dialog_agent.py`, `tests/ownership/source_test_impact_manifest.json`, and `tests/test_dialog_trace_5eab4c8aee004cc59680fac47c0685d9_live_llm.py`.
- Authority: may edit only the owned surface and run its exact syntax, deterministic, and live checks; may choose local wording and test arrangement that preserve the fixed contracts.
- Applicable skills: `development-plan`, `py-style`, `cjk-safety`, `test-style-and-execution`, `debug-llm`.
- Capability floor: production prompt-contract editing, typed role semantics, pytest ownership mapping, CJK-safe Python editing, and real-LLM artifact inspection.
- Independence requirement: separate from final sign-off.
- Acceptance output: scoped diff, changed-file list, syntax results, exact deterministic results, individual live results, and residual-risk note.
- Gate: this in-progress plan, captured baseline, explicit owned file set, two-turn DeepSeek handoff, and no edits outside ownership.
- Plan-scoped fixed execution constraint: `deepseek_v4_flash_0731` with its fixed DeepSeek-V4-Flash configuration, selected explicitly by the user. Only the user may change this executor constraint.

### parent_signoff_owner

- Responsibility: independently review the implementation diff and evidence against this plan, run final mapped verification, inspect each live artifact, and issue pass or fail.
- Owned surface: read-only review of all implementation-owned files and artifacts; may update this plan and registry lifecycle evidence only after the implementation gate passes.
- Authority: may reject the implementation, identify findings, run checks, and sign off; may not silently expand or reinterpret the implementation contract.
- Applicable skills: `development-plan`, `py-style`, `cjk-safety`, `test-style-and-execution`, `debug-llm`.
- Capability floor: cross-stage architecture review, source/test traceability, prompt-quality judgment, and real-LLM behavioral inspection.
- Independence requirement: must not author the production fix being signed off.
- Acceptance output: evidence-backed sign-off verdict, exact commands/results, artifact judgment, and residual risk.
- Gate: DeepSeek implementation completed within scope; every changed production source has a collected and passing mapped unit node; individual live results are inspected; no unresolved blocking finding remains.

## Test Impact And Traceability

| Repository path | Changed symbol or contract | Semantic owner | Exact deterministic pytest node IDs | Supplemental live node IDs | Test mode | Regression prevented |
| --- | --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` | required-selection prompt and repair instructions for concrete embedded-action typing | required-selection goal cognition | `tests/unit/cognition_core_v2/test_goal_cognition.py::test_required_selection_prompt_separates_wrapper_and_embedded_action_roles` | `tests/test_cognition_core_v2_required_selection_live_llm.py::test_live_required_selection_accepts_one_progress_event` | deterministic owner plus one-at-a-time live LLM | Prevents agreement/request wrappers from becoming the actor/target of the selected nested action. |
| `src/kazusa_ai_chatbot/nodes/dialog_agent.py` | `_V2_DIALOG_ROLE_DIRECTION_PROMPT` same-action reversal boundary | dialog role-direction verifier | `tests/unit/nodes/test_dialog_agent.py::test_role_direction_prompt_allows_compatible_nested_request_actions` | `tests/test_dialog_trace_5eab4c8aee004cc59680fac47c0685d9_live_llm.py::test_live_trace_candidate_one_role_direction_verdict`; `tests/test_dialog_trace_5eab4c8aee004cc59680fac47c0685d9_live_llm.py::test_live_trace_candidate_two_role_direction_verdict`; `tests/test_dialog_trace_5eab4c8aee004cc59680fac47c0685d9_live_llm.py::test_live_trace_candidate_three_role_direction_verdict`; `tests/test_dialog_trace_5eab4c8aee004cc59680fac47c0685d9_live_llm.py::test_live_trace_true_same_action_reversal_is_rejected` | deterministic owner plus one-at-a-time live LLM | Prevents wrapper/request scope false positives while retaining rejection of an explicit same-action reversal. |
| `src/kazusa_ai_chatbot/cognition_core_v2/README.md` | selected operation wrapper-versus-embedded-action ownership | Cognition V2 ICD | `tests/unit/cognition_core_v2/test_goal_cognition.py::test_required_selection_prompt_separates_wrapper_and_embedded_action_roles` | none | deterministic documentation contract | Prevents documentation from reintroducing wrapper-level endpoint ownership. |
| `src/kazusa_ai_chatbot/nodes/README.md` | role verifier same-action comparison boundary | dialog node ICD | `tests/unit/nodes/test_dialog_agent.py::test_role_direction_prompt_allows_compatible_nested_request_actions` | none | deterministic documentation contract | Prevents future evaluator tightening from rejecting compatible nested actions. |

## Change Surface

### Delete

None.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: clarify concrete embedded-action endpoint ownership in initial, recurrence, and repair prompt contracts.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: narrow hard reversal to unambiguous reversal of the same selected embedded action and explicitly preserve compatible wrapper/request layers.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`: document selected embedded-action typing.
- `src/kazusa_ai_chatbot/nodes/README.md`: document role-verifier scope and mitigation boundary.
- `tests/unit/cognition_core_v2/test_goal_cognition.py`: add the exact mapped goal prompt owner node.
- `tests/unit/nodes/test_dialog_agent.py`: add the exact mapped dialog prompt owner node.
- `tests/ownership/source_test_impact_manifest.json`: register both exact owner nodes.
- `tests/test_dialog_trace_5eab4c8aee004cc59680fac47c0685d9_live_llm.py`: change the three captured regressions to acceptance contracts while preserving full raw artifacts.

### Create

None.

### Keep

- Typed selected-operation schema and deterministic validation code.
- Dialog score threshold, aggregation, attempts, hard-issue ineligibility, and fail-closed delivery.
- Semantic-fidelity and surface-integrity verifier ownership.
- All unrelated working-tree files and plans.

## Agent Autonomy Boundaries

The implementation owner may choose exact prompt phrasing, test assertion decomposition, and documentation placement within the owned files. It must preserve the confirmed decisions, exact test node names, schema, thresholds, routes, and exclusions. A need for new fields, deterministic semantic parsing, threshold changes, or edits outside the owned surface stops execution and requires a plan amendment or user decision.

## Verification

1. Capture `git status --short`, `git rev-parse HEAD`, and hashes of every owned file before handoff.
2. After every CJK-bearing Python edit, run an AST syntax check with `venv\Scripts\python`.
3. Collect and run the two exact new deterministic owner nodes.
4. Run `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run` after all source edits.
5. Run each of the three captured trace live nodes individually with `-m live_llm`, inspecting its raw artifact before starting the next.
6. Run the existing true nested-reversal live node individually and inspect its artifact.
7. Run the goal required-selection live node individually and inspect its artifact for concrete embedded-action role quality.
8. Review the complete diff for scope, prompt adjacency, CJK safety, style, test mapping, and preservation of strict true-reversal behavior.

## Acceptance Criteria

- Both exact new deterministic owner nodes collect and pass.
- Source-test impact validation passes for every changed production path.
- Each of the three preserved trace spans receives a role score at or above `DIALOG_PASS_SCORE_THRESHOLD` with no typed violation.
- The existing explicit nested same-action reversal remains below threshold with a typed violation.
- The required-selection live output describes a concrete embedded action rather than only agreement, selection, telling, requesting, or confirming wrappers.
- No threshold, schema, route, retry, deterministic semantic, persistence, or delivery change appears in the diff.
- Parent review finds no unresolved blocking issue and records final sign-off.

## Progress Checklist

- [x] RCA confirmed against full trace context and monologue.
- [x] Production implementation explicitly authorized by the user.
- [x] Fixed DeepSeek Flash executor and parent sign-off owner confirmed.
- [x] Baseline and owned-file hashes captured.
- [x] DeepSeek acknowledgement completed.
- [x] DeepSeek implementation completed within scope.
- [x] Exact deterministic owner nodes collected and passed.
- [x] Captured and true-reversal live nodes run and inspected one at a time.
- [x] Source-test impact validation passed.
- [x] Parent sign-off completed.

## Execution Evidence

- Source trace: `llmtrace_5eab4c8aee004cc59680fac47c0685d9`.
- RCA review: `test_artifacts/llm_traces/dialog_trace_5eab4c8aee004cc59680fac47c0685d9_review.md`.
- Pre-implementation real verifier reproduction: all three preserved spans scored `0.0` as `typed_operation_role_reversal` when conditioned on the flattened tuple.
- Baseline: `ae9e68baac2e935956f29ff740e3942e1211e560`; the owned file set and hashes were captured before handoff, and unrelated worktree files remained preserved.
- Implementation executor: DeepSeek Flash subagent `019ffb89-c821-7ab1-8cb2-261855ecc1d4` completed the bounded prompt, documentation, mapped-test, and manifest patch; the parent independently reviewed and tested it.
- Exact deterministic owner tests: `2 passed`.
- Source-impact gate: `27 passed`; every changed production source retained an exact passing owner node.
- Live goal evidence: `test_live_required_selection_accepts_one_progress_event` passed and emitted a concrete operation whose response/selection owner is the current character, embedded actor is the current user, and embedded target is the current character.
- Historical role replays: candidates one, two, and three each scored `1.0` with no violations after the selected operation was corrected to the monologue-consistent user-to-character nested action.
- Strict live control: `test_live_trace_true_same_action_reversal_is_rejected` scored `0.0` with `typed_operation_role_reversal` for an explicit character-to-user reversal of that same action.
- Full-context audit: all three preserved spans were classified as aligned with the goal. The overall audit remained `underdetermined` because the metadata trace retained only each offending span rather than the full surrounding generated message.
- Parent sign-off: PASS. The patch changes semantic prompt ownership only; thresholds, schemas, attempt caps, model routes, deterministic validators, persistence, and delivery behavior are unchanged.

## Execution Handoff

- Role: `implementation_owner`.
- Executor resolution: plan-scoped fixed execution constraint `deepseek_v4_flash_0731`; user-selected, eligible for the bounded prompt/test patch and local verification.
- Selection rationale: the user explicitly required one DeepSeek Flash subagent; the scope is narrow, prompt-centric, fully file-bounded, and independently signable by the parent.
- Final checkpoint: implementation, verification, and parent sign-off completed; this plan is archived as historical evidence.
