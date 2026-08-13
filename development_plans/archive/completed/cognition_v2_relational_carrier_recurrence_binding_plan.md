# cognition v2 relational-carrier recurrence binding bugfix

## Summary

- Goal: preserve the validated episode identity when an ordinary-response recurrence enters goal cognition, preventing the historical `current_turn_relational_carrier_invalid` failure caused by context loss.
- Status: completed
- Scope boundary: Cognition V2 branch-context propagation, recurrence validation, deterministic ownership tests, captured-trace live regression tests, and their review artifacts.
- Change direction: use the validated cognition input payload as the sole episode-identity owner, retain fail-closed recurrence validation, and make the historical ownership error permanently test-visible.
- Acceptance state: execution complete; independent plan and final native review approved with no blocking findings or acceptance gaps.

## Confirmed Decisions

1. `payload["episode"]["episode_id"]` is the canonical episode identity for the Cognition V2 branch handler.
2. The mutable `state` argument passed to `_branch_handler` is not an episode container and cannot supply recurrence identity.
3. The relational carrier remains transient, exact-schema, ordinary-branch-only, and episode-bound.
4. Missing episode binding fails before an LLM call with `attempt_count=0`, `safe_checkpoint="pre_state_commit"`, and `retryable=False`.
5. Recurrence materialization continues to require at least one current-episode evidence handle.
6. Sibling-branch recovery behavior remains unchanged; the ordinary branch must be correctly dispatched before recovery policy is evaluated.
7. The original production ownership correction is already present in commit `ff85eae7`; this plan closes the regression-verification and plan-review gap around that fix.
8. Both supplied traces are classified as missing episode binding. Missing current-episode evidence is a separate synthetic goal-owner contract case.

## Scope And Change Direction

The corrected flow is:

```text
validated cognition input
  -> payload-owned episode identity
  -> isolated branch context with _episode_id
  -> current-turn carrier validation
  -> ordinary recurrence goal call
  -> carrier materialization using current-episode evidence
```

The branch handler must derive `_episode_id` from the same validated payload
that supplied the carrier and evidence. It must not inspect mutable cognition
state for an episode object. The goal owner keeps the existing bounded
validation and typed fail-closed result. The plan does not change the LLM
prompt, model route, relational semantics, sibling recovery, persistence, or
delivery behavior.

## Mandatory Skills

- `development-plan` for lifecycle, ownership, exact traceability, and review gates.
- `llm-trace-debug` for protected trace evidence and historical capsule review.
- `debug-llm` for real-LLM artifacts and agent-authored human-readable review.
- `test-style-and-execution` for deterministic ownership tests and one-at-a-time live tests.
- `py-style` for Python changes.

## Mandatory Rules

- Preserve the current canonical payload-to-context ownership correction.
- Keep raw trace exports, prompts, model outputs, and user data under local diagnostic artifacts; do not copy secrets into source or plan text.
- Keep real-LLM tests one case at a time and inspect each JSON artifact and Markdown review.
- Keep deterministic tests independent of live model availability.
- Keep production changes unimplemented until the user explicitly authorizes implementation.
- Do not introduce compatibility aliases, fallback episode sources, prompt changes, semantic retries, or new sibling-branch recovery behavior.

## Must Do

1. Verify and preserve the payload-owned episode binding in `_branch_handler`.
2. Add deterministic goal-owner coverage for missing episode binding and missing current-episode evidence, including exact typed failure fields.
3. Retain the facade ownership test proving the validated input episode reaches recurrence context.
4. Retain the captured-carrier live tests for both supplied trace cases as missing-binding replays, with raw model-call evidence emitted as JSON only; the inspecting agent authors the Markdown reviews afterward.
5. Verify that a valid carrier with matching episode binding and current-episode evidence reaches a real ordinary recurrence call without this error.
6. Keep the checked-in replay fixture deidentified while preserving schema, branch, cycle, provenance, and expected failure fields; keep exact trace values in ignored diagnostic exports.
7. Register this plan in `development_plans/README.md` before approval or execution.
8. Record the protected-trace evidence, historical cause, verification commands, review outcome, and any residual risk in the execution record after implementation.

## Deferred

- Changes to relational willingness semantics or relationship-state inference.
- Changes to evidence authority classification or resolver cycle scheduling.
- Changes to sibling-branch recovery, final reduction, action planning, surface generation, persistence, or delivery.
- Broad trace-schema redesign or new error-code taxonomy beyond the current contract.
- Full production-dialog replay through adapters and live MongoDB; the root clause is pre-model and is covered at the Cognition V2 owner boundary.

## Target State

The following invariants hold:

1. For every ordinary branch invocation, `context["_episode_id"]` equals `payload["episode"]["episode_id"]`.
2. A supplied recurrence carrier is validated against that same episode ID.
3. A malformed or mismatched carrier is kept out of model calls, state mutation, persistence, scheduling, dialog, and delivery.
4. A valid recurrence carrier is materialized only when its cited handles include current-episode evidence from `episode`, `scheduler_event`, or `tool_result`.
5. The typed failure retains `error_code="current_turn_relational_carrier_invalid"`, `stage="goal_cognition"`, `attempt_count=0`, `safe_checkpoint="pre_state_commit"`, and `retryable=False` for deterministic precondition failures.

## Execution Roles

### implementation_owner

- Responsibility: implement regression hardening and produce verification evidence for the already-landed ownership correction.
- Owned surface: `tests/ownership/source_test_impact_manifest.json`, `tests/unit/cognition_core_v2/test_facade.py`, `tests/unit/cognition_core_v2/test_goal_cognition.py`, `tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py`, `tests/fixtures/cognition_core_v2_relational_carrier_failure_cases.json`, and this plan's execution evidence. `src/kazusa_ai_chatbot/cognition_core_v2/facade.py` and `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` are read-only verification surfaces.
- Authority: may modify only the owned test, fixture, manifest, and plan surfaces after explicit implementation authorization; may run local deterministic and live tests; may update this plan's execution evidence. Production-source edits require a plan amendment and separate user authorization.
- Applicable skills: `development-plan`, `llm-trace-debug`, `debug-llm`, `test-style-and-execution`, `py-style`.
- Capability floor: able to trace typed contracts across facade, goal owner, evidence validators, and live test artifacts.
- Independence requirement: separate from the plan-review role.
- Acceptance output: source diff, exact mapped pytest results, raw live artifacts, human-readable reviews, and execution evidence.
- Gate: approved plan, explicit user implementation authorization, clean baseline comparison, and no unresolved plan-review blocker.

### independent_plan_reviewer

- Responsibility: inspect this plan for root-cause fidelity, ownership, scope, contract consistency, and executable verification.
- Owned surface: read-only review of this plan, the RCA artifact, supplied trace exports, relevant source, and mapped tests; no file edits.
- Authority: may issue blocking and non-blocking findings and a review verdict; may not authorize production implementation or remediate findings.
- Applicable skills: `development-plan`, `llm-trace-debug`, `debug-llm`.
- Capability floor: GPT 5.6-sol with high reasoning, repository context, and access to the cited evidence.
- Independence requirement: must be independent of the implementation owner.
- Acceptance output: written review with evidence-backed findings, required amendments, and a verdict.
- Gate: draft plan and RCA are complete; exit requires every blocking finding to be addressed in the plan or explicitly accepted by the user.

## Test Impact And Traceability

| Repository path | Changed symbol or contract | Semantic owner | Exact deterministic pytest node IDs | Supplemental live node IDs | Test mode | Regression prevented |
| --- | --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/cognition_core_v2/facade.py` | `_branch_handler` episode-to-context propagation | Cognition V2 facade; current source is a read-only verified fix | `tests/unit/cognition_core_v2/test_facade.py::test_branch_handler_carries_input_episode_into_recurrence_context` | `tests/test_cognition_core_v2_quoted_message_post_fix.py::test_post_fix_goal_branch_recurrence_live_llm`; `tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py::test_live_f408_carrier_failure_without_episode_binding`; `tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py::test_live_fdd_carrier_failure_without_episode_binding` | deterministic plus one-at-a-time live LLM | Prevents recurrence from entering goal cognition with a valid carrier but no episode identity. |
| `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` | `_run_goal_cognition` recurrence precondition and `_materialize_recurrence_relational_willingness` evidence gate | ordinary-response goal owner; current source is a read-only verified contract | `tests/unit/cognition_core_v2/test_goal_cognition.py::test_goal_cognition_rejects_recurrence_without_episode_binding`; `tests/unit/cognition_core_v2/test_goal_cognition.py::test_goal_cognition_rejects_recurrence_without_current_episode_evidence` | `tests/test_cognition_core_v2_quoted_message_post_fix.py::test_post_fix_goal_branch_recurrence_live_llm`; `tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py::test_live_f408_carrier_failure_without_episode_binding`; `tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py::test_live_fdd_carrier_failure_without_episode_binding` | deterministic plus one-at-a-time live LLM | Prevents invalid recurrence carriers from reaching model or state-commit paths and preserves current-episode provenance requirements. |
| `tests/fixtures/cognition_core_v2_relational_carrier_failure_cases.json` | deidentified captured-carrier schema, branch, provenance, and expected fields | trace-replay evidence boundary | `tests/unit/cognition_core_v2/test_goal_cognition.py::test_relational_carrier_replay_fixture_is_deidentified_and_complete` | `tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py::test_live_f408_carrier_failure_without_episode_binding`; `tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py::test_live_fdd_carrier_failure_without_episode_binding` | fixture validation plus one-at-a-time live LLM | Prevents replay tests from exposing protected identities or drifting from the supplied failure contract. |
| `tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py` | live captured-carrier reproductions and JSON evidence emission | LLM regression harness | none | `tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py::test_live_f408_carrier_failure_without_episode_binding`; `tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py::test_live_fdd_carrier_failure_without_episode_binding` | one-at-a-time real LLM | Prevents deterministic-only coverage from missing model-boundary timing while keeping human review outside test code. |

## Change Surface

### Delete

None.

### Modify

- `tests/ownership/source_test_impact_manifest.json`: register the direct facade and goal-owner unit nodes plus live supplements.
- `tests/unit/cognition_core_v2/test_goal_cognition.py`: add exact deterministic tests named in the traceability matrix and validate the deidentified fixture.

### Create

- `tests/fixtures/cognition_core_v2_relational_carrier_failure_cases.json`: checked-in, sanitized summaries of the two supplied trace cases.
- `tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py`: real-LLM reproductions with raw JSON evidence; the inspecting agent authors the Markdown reviews outside test code.
- `test_artifacts/llm_debug/cognition_v2_relational_carrier_root_clause_analysis.md`: human-readable RCA linked to protected evidence.

### Keep

- Existing payload validation and exact relational-carrier schemas.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py::_branch_handler` at current HEAD, which already binds from `payload["episode"]["episode_id"]`.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` recurrence validation at current HEAD.
- Older timestamped test-generated Markdown files under `test_artifacts/cognition_core_v2_relational_carrier_failure_live_llm/reviews/` are historical and excluded from execution evidence; the authoritative reviews are the `__agent_review.md` files.
- Existing sibling recovery and final reduction behavior.
- Existing `test_post_fix_goal_branch_recurrence_live_llm` control coverage.
- Protected trace exports under `test_artifacts/diagnostics/` as local evidence only.

## Agent Autonomy Boundaries

The implementation owner may choose local test helpers, fixture loading details,
artifact naming, and assertion organization within the listed files. The owner
must preserve the canonical payload ownership, typed error fields, live-call
boundaries, and exclusions above. The owner must request a plan amendment or
user decision before changing production semantics, error taxonomy, recovery
policy, persistence, prompt content, model routing, or the listed file surface.

## Verification

1. Capture `git status --short`, the current commit, and the explicitly owned file set before implementation.
2. Run the exact deterministic node IDs in the traceability matrix and confirm collection is not deselected.
3. Run each live node individually with `-m live_llm`, inspect its JSON artifact, and author or update its Markdown review from the real prompt/output evidence.
4. Run Ruff on every modified Python test or production file.
5. Re-read the RCA and compare the final diff against the root clause: payload-owned episode identity, no state fallback, and no semantic expansion.

## Acceptance Criteria

- The branch handler always passes the validated payload episode ID into recurrence semantic context.
- Deterministic tests prove the propagation and both goal-owner fail-closed paths with exact typed fields.
- Both supplied trace cases remain reproducible at their corresponding boundary with real LLM execution and durable review artifacts.
- For each supplied replay, `resolver_cycle_index=1`, `e1/source_kind=episode`, `attempt_count=0`, and unchanged model-call count after removing `_episode_id` are asserted.
- A valid carrier with matching episode identity and current-episode evidence completes ordinary recurrence without `current_turn_relational_carrier_invalid`.
- No prompt, model-route, relationship-semantic, sibling-recovery, persistence, or delivery behavior changes outside this plan occur.
- The independent plan review has no unresolved blocking finding, and all accepted findings are recorded in the plan's review evidence.

## Independent Plan Review

The requested independent review was resolved as a plan-scoped fixed execution
constraint: GPT 5.6-sol, high reasoning, default/normal service speed,
read-only authority. The reviewer inspected the RCA, protected trace evidence,
current and historical facade code, fixture, live harness, manifest, and exact
test nodes.

Initial verdict: `needs changes`. Findings required trace-to-test fidelity,
agent-owned Markdown reviews, deidentified fixture values, corrected
production-surface authority, accurate traceability mappings, and registry
registration. Those findings were addressed in the fixture, deterministic and
live tests, RCA, manifest, plan ownership sections, and registry.

Final verdict: `approved`. Blocking findings: none. Non-blocking finding:
older pre-amendment generated Markdown artifacts remain beside the authoritative
`__agent_review.md` files and are explicitly historical and excluded from
execution evidence. Residual risk is low: the shared error code remains broad,
and sibling recovery can mask an ordinary-branch failure in an otherwise
successful invocation.

## Execution Evidence

### Execution Handoff: production binding slice

- Plan lifecycle: `in_progress`; user-authorized implementation execution.
- Role: `implementation_owner`, delegated production slice.
- Executor: native DeepSeek Flash subagent `019ffd45-b9e3-7b63-8066-d31bd685203e` (`deepseek_v4_flash_0731`, `deepseek-v4-flash`, high reasoning); the acknowledgement turn completed before this execution turn.
- Resolution mode: user-fixed model assignment for this production slice.
- Owned files: `src/kazusa_ai_chatbot/cognition_core_v2/facade.py` and `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` only.
- Baseline: commit `8970474b9b0e946566d192217757ca688d6c9c4f`; pre-existing untracked paths were recorded and remain outside this slice.
- Authority: preserve payload-owned episode identity, fail-closed typed recurrence validation, exact prompt/model/recovery/persistence/delivery behavior, and no compatibility or fallback episode source.
- Acceptance gate: inspect the approved plan and subsystem contracts, make the smallest necessary production correction or report that the correction is already present, and return changed paths plus focused verification evidence.
- Next checkpoint: compare the worker diff with the baseline, run the exact mapped deterministic nodes, review the live replay artifacts, and obtain independent native sign-off.
- Execution amendment: the user explicitly authorized this bounded production-source handoff despite the original read-only verification wording; the approved semantic direction and exclusions remained fixed, and the worker accepted no production diff because the canonical correction was already present.

- Root-clause RCA: `test_artifacts/llm_debug/cognition_v2_relational_carrier_root_clause_analysis.md`.
- Initial reviewer handoff: GPT 5.6-sol, high reasoning, normal/default service speed, read-only review; final verdict `needs changes`.
- Remediation confirmation: same independent reviewer, read-only; final verdict `approved`, no blocking findings.
- Deterministic owner/governance nodes: the four exact plan-mapped nodes passed; two additional direct facade/goal contract nodes also passed (6 total in the focused owner run).
- Live private replay: passed individually with raw artifact `test_artifacts/cognition_core_v2_relational_carrier_failure_live_llm/captured_private_missing_episode_binding__1786660848450278100.json` and authoritative review `test_artifacts/cognition_core_v2_relational_carrier_failure_live_llm/reviews/captured_private_missing_episode_binding__agent_review.md`; the control made one model call and removing `_episode_id` added none.
- Live group replay: passed individually with raw artifact `test_artifacts/cognition_core_v2_relational_carrier_failure_live_llm/captured_group_missing_episode_binding__1786660994431736900.json` and authoritative review `test_artifacts/cognition_core_v2_relational_carrier_failure_live_llm/reviews/captured_group_missing_episode_binding__agent_review.md`; the control made one model call and removing `_episode_id` added none.
- Valid recurrence control: passed individually with raw artifact `test_artifacts/cognition_core_v2_quoted_message_post_fix/goal_branch_recurrence_live_llm.json` and agent-authored review `test_artifacts/cognition_core_v2_quoted_message_post_fix/reviews/goal_branch_recurrence_live_llm.md`; no typed failure, memory/history rebuild passed, and the relational decision remained carried.
- Ruff and Python syntax checks: modified test surfaces pass Ruff and syntax checks; owned production files compile successfully but retain 7 pre-existing Ruff findings (facade import ordering and six goal-cognition `TRY004` findings), with zero production diff so no out-of-scope cleanup was applied.
- Test-only hygiene: removed one extra blank line in `tests/unit/cognition_core_v2/test_facade.py` to satisfy Ruff; no semantic test change.
- Production source worktree state: unchanged from the baseline; current payload-owned binding remains the verified `ff85eae7` fix.
- Independent final review: native Kazusa plan reviewer `019ffd4e-1d8f-7c90-acda-0b476fa5c4cd`, read-only, verdict `approved`, no critical/high/medium/low findings and no acceptance gaps.
- Final residual risk: seven pre-existing production Ruff findings remain; the shared carrier-invalid error code aggregates multiple precondition failures, and sibling recovery can mask an ordinary-branch failure.
- Lifecycle disposition: execution evidence is complete and this plan is ready to move to `development_plans/archive/completed/`.
