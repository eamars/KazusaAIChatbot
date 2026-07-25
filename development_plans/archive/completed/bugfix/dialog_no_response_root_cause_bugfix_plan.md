# dialog no-response root-cause bugfix

## Summary

- Goal: eliminate the confirmed dialog-compliance and decontextualizer
  regeneration causes of user-visible operational-error responses, and correct
  the replay evidence harness that obscured turn chronology and raw evidence.
- Plan class: large.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `cjk-safety`,
  `test-style-and-execution`, `debug-llm`, and `character-test`.
- Overall cutover strategy: bigbang replacement of the faulty repair
  contracts and test-harness behavior.
- Highest-risk areas: semantic ownership between text surface and dialog,
  post-cognition-commit failure metadata, local-model repair latency, and
  protected trace retention.
- Acceptance criteria: focused deterministic contracts pass; individually
  inspected real-LLM cases produce valid dialog; typed failures expose the
  correct owner and checkpoint; harness chronology and full-trace assertions
  pass; independent review has no unresolved findings.

## Context

The completed
`asuna_private_r18_affinity_harness_plan.md` produced three no-dialog
operational failures across forty turns. The RCA in
`test_artifacts/cognition_core_v2/asuna_private_r18_affinity_replay/e2e_20260725_full_01/comparison_review.md`
confirmed two distinct production failure families:

- default turn 01 and high-affinity turn 02 exhausted the dialog semantic
  compliance boundary after one repair;
- default turn 19 exhausted the decontextualizer response-operation contract
  after returning the same invalid candidate twice.

The dialog repair removed the validated text-surface output and asked the
wording owner to reconstruct character stance from raw percepts. This crossed
the cognition/L3/dialog ownership boundary. The semantic verifier also left
valid character refusal, negotiation, and conditions ambiguous under its
current-input conflict rule.

The decontextualizer validator converted specific nested
`CognitiveEpisodeValidationError` details into one generic message. Its retry
used two consecutive human messages, while the dedicated repair instruction
was unused. The local model therefore lacked a clear prior-assistant candidate
and actionable field correction.

The replay harness supplied historical fixture timestamps for new user rows
while assistant rows used runtime timestamps. Timestamp sorting eventually
produced assistant-only recent windows. The child also retained metadata-only
LLM traces. These defects invalidate causal interpretation of the late-turn
failure and prevent candidate-level review.

The user authorized production implementation and focused real-LLM
verification. Full twenty-turn E2E execution remains outside this plan's
verification run.

On 2026-07-25, the user explicitly approved expanding the focused-verifier
retry and call budget after repeated real-model `Issues` key drift exposed a
recoverable post-commit no-dialog seam.

Focused verification then confirmed a lexical producer-contract collision:
the shared `issues` spelling across verifier prompts reinforced the model's
memorized `Issues` variant even after exact correction. Semantic fidelity now
owns the exact external field `hard_errors`, while role direction and surface
integrity retain `issues`. The semantic field is normalized to the internal
aggregate only after exact validation; no alias is accepted.

## Mandatory Skills

- `development-plan`: governs this execution record, stage sign-off, and
  independent review.
- `local-llm-architecture`: governs LLM ownership, prompt scope, retry count,
  context caps, and latency.
- `no-prepost-user-input`: keeps role and response semantics LLM-owned while
  deterministic code validates structure only.
- `py-style`: governs every production and test Python edit.
- `cjk-safety`: governs Chinese prompt and fixture edits plus immediate syntax
  validation.
- `test-style-and-execution`: separates deterministic contracts from
  one-at-a-time real-LLM quality evidence.
- `debug-llm`: requires raw evidence plus an agent-authored readable review.
- `character-test`: governs focused character-response inspection and test
  database isolation.

## Mandatory Rules

- The parent owns test contracts, verification, evidence, review remediation,
  lifecycle updates, and final sign-off.
- One production-code subagent owns the planned production Python changes
  after focused failing contracts are recorded.
- One independent review subagent reviews the completed diff and evidence
  after verification.
- LLM stages retain semantic judgment. Deterministic code owns exact shapes,
  enums, limits, attempt caps, byte-identity detection, persistence, and
  failure projection.
- Dialog renders a validated L3 stance. L3 owns replacement of content,
  requirements, visible boundaries, and addressee semantics after a verified
  hard failure.
- A character-owned refusal, negotiation, or condition remains compatible
  with a direct user request unless it reverses a typed actor, target, response
  owner, or selection owner.
- Decontextualizer repair uses the canonical `parse_llm_json_output(...)`
  entrypoint and the existing two-attempt cap.
- Prompt constants use triple-single-quoted strings and named `.format(...)`
  composition only.
- CJK-bearing Python files receive immediate AST or `py_compile` validation.
- Real-LLM tests run one case at a time, with each raw artifact inspected
  before the next case.
- Full twenty-turn E2E execution stays deferred throughout this plan.
- Existing unrelated worktree changes remain preserved.
- After automatic context compaction, the active agent rereads this entire
  plan before continuing.
- After signing off a major checklist stage, the active agent rereads this
  entire plan before starting the next stage.
- Before lifecycle completion, the parent runs the Independent Code Review
  gate and records its outcome in Execution Evidence.

## Must Do

- Add focused failing tests before production implementation.
- Replace dialog-only semantic reconstruction with one L3-owned semantic
  surface replacement followed by one second dialog candidate. The L3
  producer may use the canonical two-attempt structural-regeneration cap
  without creating an additional semantic replacement or dialog candidate.
- Keep the existing two-dialog-candidate hard cap.
- Give semantic fidelity, role direction, and surface integrity separate
  two-attempt structural-regeneration contracts. Each producer reuses its own
  unchanged semantic context and may make one replacement call only after its
  first parsed candidate fails exact structural validation.
- Preserve each verifier's local prompt/model/payload/parser/validator block;
  do not hide the three semantic owners behind a generic invocation helper.
- Use exact `aligned` plus `hard_errors` as the semantic-fidelity producer
  contract; role-direction and surface-integrity retain exact `aligned` plus
  `issues`. Accept no compatibility alias or case variant.
- Record both rejected and accepted verifier attempts in protected trace
  evidence, cap the rejected candidate included in repair feedback, and expose
  stage-specific typed exhaustion with checkpoint `post_cognition_commit`.
- Make the semantic fidelity prompt explicitly preserve valid refusal,
  negotiation, and conditions.
- Raise a typed dialog compliance exhaustion carrying error code, stage,
  attempt count, checkpoint, and retryability.
- Project typed dialog failure metadata through the service boundary.
- Preserve exact nested response-operation validation errors in
  decontextualizer repair feedback.
- Use a system/human/assistant/human message sequence for decontextualizer
  regeneration.
- Detect a byte-identical second invalid decontextualizer candidate and expose
  an `unchanged_candidate` exhaustion code.
- Correct replay request chronology using service receive time while retaining
  source timestamps in fixture evidence.
- Assert per-turn persisted conversation ordering in the harness.
- Enable and assert full protected trace capture in the replay child.
- Leave human-readable quality review authorship with the inspecting agent.
- Update subsystem documentation for the new owner-repair and failure
  contracts.
- Run all verification commands listed in this plan except the explicitly
  deferred full E2E command.

## Deferred

- Full high/default twenty-turn E2E replay execution.
- Production database writes or production-memory character testing.
- Retries beyond the approved two-attempt L3, decontextualizer, and focused
  verifier structural-regeneration paths.
- Keyword rules, deterministic semantic correction, and safety suppression.
- Model-route changes, token-budget increases, and dependency upgrades.
- Broad dialog, cognition, persistence, or adapter refactors.
- Candidate-specific prompt examples copied from the failed R18 turns.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Dialog hard repair | bigbang | Replace percept-only wording repair with L3-owned semantic surface replacement and one second render. |
| Dialog failure metadata | bigbang | Emit the typed dialog compliance exhaustion directly; retain no generic alias path. |
| Focused verifier structure | bigbang | Each semantic, role-direction, and surface-integrity producer regenerates one structurally invalid verdict in the same context and otherwise fails with its own typed error. |
| Decontextualizer repair | bigbang | Replace consecutive-human retry construction with the canonical role sequence and exact validation detail. |
| Replay clock | bigbang | Use service receive time for model-facing chronology; retain source time only in fixture/artifact provenance. |
| Trace capture | bigbang | Require full protected capture for future replay children. |
| Human review | bigbang | Keep scripts data-only and author readable assessment after evidence inspection. |

Cutover enforcement:

- Each area moves directly to the listed contract.
- Compatibility aliases, dual repair paths, and fallback retry modes stay
  absent.
- A cutover-policy change requires renewed user approval.

## Target State

The common successful dialog path retains its present stages and call count.
When a focused verifier rejects the first candidate:

```text
validated cognition and TextSurfaceInputV2
  -> L3 semantic surface replacement
  -> validated replacement TextSurfaceOutputV2
  -> second dialog render
  -> same focused verifiers
  -> deliver or typed dialog_compliance_contract_exhausted
```

The L3 replacement returns complete `content_plan`,
`content_requirements`, `visible_boundaries`, and `addressee_plan` fields.
The original validated style, selected intent, permitted action results, and
runtime capability limits remain unchanged. Dialog receives the replacement
surface, the original candidate, and bounded verifier issues; it changes
wording without inventing a new character stance.

Each focused verifier keeps one local semantic stage:

```text
system contract
  -> human candidate/evidence payload
  -> canonical parse and exact verdict validation
  -> on structural error only:
       rejected assistant candidate
       -> human correction with exact structural error
       -> one complete replacement verdict
  -> validated verdict or typed stage-specific exhaustion
```

This structural regeneration does not re-render dialog, add another semantic
verifier owner, or create a third dialog candidate.

The decontextualizer regeneration path becomes:

```text
original system contract
  -> original human semantic input
  -> rejected assistant candidate
  -> human correction containing the exact field error
  -> complete replacement object
```

An unchanged second invalid candidate terminates with
`message_decontextualizer_unchanged_candidate_exhausted`. A different second
invalid candidate retains
`message_decontextualizer_contract_exhausted`.

Future replay requests retain the source timestamp in their source-message
artifact while leaving `ChatRequest.local_timestamp` empty so the service
creates an interleaved runtime clock. Every turn asserts that each assistant
row follows its trace-owning user row. Replay children set
`LLM_TRACE_CAPTURE_MODE=full` and assert relevant steps contain raw messages,
raw response text, and parsed output before restoration.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Repair owner | L3 replaces semantic surface fields | Cognition/L3 own stance; dialog owns wording. |
| Repair cardinality | One L3 replacement and one second dialog candidate | Preserves bounded latency and two-candidate failure behavior. |
| Style handling | Preserve the original validated style | Style was not a semantic failure owner. |
| Surface repair context | Use authoritative input, rejected surface semantics, and bounded verifier issues | Gives the weak local model actionable context without raw state. |
| Refusal semantics | State the reusable boundary in semantic fidelity | Prevents a valid character judgment from becoming a false contradiction. |
| Verifier structural repair | One local regeneration per focused producer after exact validation failure | Repairs casing/shape drift without changing semantic ownership or the common-path call count. |
| Semantic producer vocabulary | Exact `hard_errors`, normalized internally after validation | Removes the observed shared-`issues` lexical collision without accepting a compatibility alias. |
| Terminal error | Typed `DialogComplianceContractError` | Retains compatibility with existing dialog catches while exposing owner metadata. |
| Nested validation | Propagate the original validation message | Field-level feedback is already produced by the canonical validator. |
| Unchanged repair | Compare exact response text deterministically | This classifies retry progress without interpreting semantics. |
| Replay time | Empty `local_timestamp` | The service receive clock naturally preserves user/assistant order. |
| Full traces | Child environment plus per-turn assertion | Configuration alone is insufficient proof of retained evidence. |

## Contracts And Data Shapes

`GlobalPersonaState` and `DialogAgentState` gain an optional
`text_surface_input_v2: TextSurfaceInputV2`. Production speech paths populate
it before dialog. A successful first dialog candidate does not require the
field; a rejected candidate requires it for owner-correct replacement.

The L3 semantic repair stage returns exactly:

```python
{
    "content_plan": str,
    "content_requirements": list[str],
    "visible_boundaries": list[str],
    "addressee_plan": list[str],
}
```

`DialogComplianceContractError` carries:

```python
error_code = "dialog_compliance_contract_exhausted"
stage = "dialog_compliance"
attempt_count = 2
safe_checkpoint = "post_cognition_commit"
retryable = False
```

The service projects these fields without exposing candidate text or verifier
issue prose.

Focused verifier structural exhaustion carries one of:

```python
error_code = "dialog_semantic_fidelity_contract_exhausted"
stage = "dialog.semantic_fidelity"

error_code = "dialog_role_direction_contract_exhausted"
stage = "dialog.role_direction"

error_code = "dialog_surface_integrity_contract_exhausted"
stage = "dialog.surface_integrity"
```

Each carries `attempt_count = 2`,
`safe_checkpoint = "post_cognition_commit"`, and `retryable = False`.

The semantic-fidelity producer returns exactly:

```python
{
    "aligned": bool,
    "hard_errors": list[str],
}
```

Role-direction and surface-integrity retain their exact stage-owned `issues`
shapes. Deterministic validation rejects `issues`, `Issues`, and every other
alias at the semantic producer boundary.

## LLM Call And Context Budget

| Stage | Before | After | Bound |
|---|---|---|---|
| Common dialog | One generator plus focused verifiers | Unchanged when verifier output is valid; one owning verifier may make one structural replacement call | Two attempts per focused verifier; no cross-verifier retry |
| Rejected dialog | One wording repair plus verifier recheck | One L3 semantic replacement stage, one second render, and the same verifier recheck | Up to two structural attempts per producing stage; two dialog candidates; repair payloads remain bounded |
| Decontextualizer common path | One call | Unchanged | Existing completion cap |
| Decontextualizer invalid path | One regeneration | One role-correct regeneration | Two attempts total |

The new L3 repair is response-path work only after a verified hard rejection.
It reuses the configured cognition/surface route and completion cap. Its
projection omits raw persistent state and character voice, preserves the
24,000-character prompt cap, and fails closed when the reduced owner packet
cannot fit.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/surface.py`: expose the L3 semantic
  replacement facade and merge validated output.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`: add the bounded
  surface-owned repair prompt and exact result validator.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: retain the
  canonical text-surface input and bind the repair owner to existing services.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: type the
  optional retained surface input.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: pass the retained
  surface input into dialog.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: request L3 replacement,
  render the second candidate from it, clarify verifier semantics, and raise
  the typed terminal error.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py`:
  preserve exact validation feedback, use correct message roles, and classify
  unchanged regeneration.
- `src/kazusa_ai_chatbot/service.py`: project typed dialog failure metadata.
- `tests/test_dialog_visible_speech_and_semantic_fidelity.py`: deterministic
  dialog ownership and terminal-error contracts.
- `tests/test_msg_decontextualizer.py`: exact nested feedback, message-role,
  and unchanged-candidate contracts.
- `tests/test_service_cognition_graph.py`: service metadata projection.
- `tests/test_asuna_private_r18_affinity_harness_contract.py`: replay clock,
  trace-mode, chronology, and data-only-controller contracts.
- `tests/test_asuna_private_r18_affinity_live_llm.py`: runtime chronology and
  trace assertions.
- `tests/run_asuna_private_r18_affinity_replay.py`: full trace child config and
  data-only execution.
- `tests/test_dialog_visible_speech_and_semantic_fidelity_live_llm.py`: focused
  real-model repair evidence.
- `tests/test_dialog_agent.py`: align adjacent prompt assertions with the
  canonical L3-owned replacement payload.
- `tests/test_l3_dialog_content_plan_contract.py`: require the retained
  `TextSurfaceInputV2` in the adjacent L3 handoff contract.
- `tests/test_decontextualizer_referents.py`: align the adjacent identical
  malformed-candidate expectation with the typed unchanged disposition.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md` and
  `src/kazusa_ai_chatbot/nodes/README.md`: document final ownership and failure
  behavior.
- `development_plans/README.md`: register and close this plan.

### Create

- `development_plans/active/bugfix/dialog_no_response_root_cause_bugfix_plan.md`:
  execution contract and evidence.
- Focused raw and readable LLM artifacts below
  `test_artifacts/cognition_core_v2/`.

### Keep

- Cognition state schemas and persistence ordering.
- Existing model routes and completion-token caps.
- Adapter delivery, action execution, and consolidation behavior.
- The completed harness plan as historical evidence.

## Overdesign Guardrail

- Actual problem: bounded repairs lose semantic ownership and actionable
  validation feedback, focused verifiers terminate on recoverable structural
  drift, and the replay harness corrupts chronological evidence.
- Minimal change: one surface-owned semantic replacement, one correctly
  constructed decontextualizer regeneration, one local structural replacement
  per malformed focused-verifier result, typed failure projection, and
  corrected test chronology/capture.
- Ownership boundaries: LLM stages choose semantics; L3 owns surface meaning;
  dialog owns wording; deterministic code validates, caps, classifies byte
  identity, persists, and projects failures.
- Rejected complexity: extra agents, open retry loops, keyword classifiers,
  compatibility shims, model-route changes, new configuration flags, and
  persistence rollback.
- Evidence threshold: repeated final-source `Issues` key drift and explicit
  user approval authorize the focused-verifier structural retry only. Further
  retries or semantic fallback require a separate plan.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics that
  preserve every contract in this plan.
- New architecture, fallback paths, compatibility layers, model routes, and
  optional modes remain outside scope.
- Changes outside the listed surface require a plan amendment and explicit
  user approval.
- Existing equivalent helpers and validators take precedence over new
  duplicates.
- Review-only corrections may update listed tests, prompts, docs, and plan
  evidence when they preserve the fixed contracts.
- An impossible instruction stops execution with the blocker recorded.

## Implementation Order

1. Add deterministic dialog tests proving that a rejected candidate requests
   L3 replacement, preserves style/action truth, and exposes typed exhaustion.
2. Add deterministic decontextualizer tests proving exact nested feedback,
   assistant-role candidate placement, and unchanged-candidate classification.
3. Add deterministic service and harness tests for metadata, runtime clock,
   chronology, full trace mode, and data-only output.
4. Run these focused tests and record their expected pre-fix failures.
5. Start the production-code subagent with production files only.
6. Implement the L3 semantic repair stage and retained input handoff.
7. Implement dialog repair, refusal boundary, and typed terminal error.
8. Implement decontextualizer regeneration and unchanged-candidate behavior.
9. Implement service failure projection.
10. Parent applies the test-only harness corrections.
11. Run focused deterministic tests, adjacent regressions, syntax checks, and
    prompt-render checks.
12. Run the selected real-LLM cases one at a time and inspect each artifact.
13. Author the readable focused verification review from raw evidence.
14. Run independent code review, remediate in-scope findings, and repeat
    affected verification.
15. Add failing focused-verifier structural regeneration, message-role,
    trace, call-count, typed exhaustion, and service-projection contracts.
16. Implement one local structural regeneration in each focused verifier and
    rerun deterministic verification.
17. Run final-source default and high real-model probes individually, inspect
    each artifact, and update the readable review.
18. Reactivate the existing independent reviewer for the approved expansion.
19. Record evidence, update docs, complete lifecycle state, and archive the
    plan.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution
  evidence, review remediation, lifecycle updates, and final sign-off.
- Parent establishes and runs the focused failing test contract first.
- Production-code subagent: exactly one native subagent; owns only the listed
  production source and subsystem README changes; closes after implementation.
- Parent owns the harness and test edits while the production subagent works.
- Independent code-review subagent: exactly one native subagent after
  verification; reviews only and reports findings.

## Progress Checklist

- [x] Stage 1 - focused contracts and failing baseline established.
  - Covers implementation steps 1-4.
  - Verify the named focused pytest nodes fail for the intended missing
    contracts.
  - Evidence records exact failing assertions.
  - Handoff starts production implementation.
  - Sign-off: `/root`, 2026-07-25; eleven focused tests failed at the
    intended missing contracts before production implementation.
- [x] Stage 2 - production ownership and failure contracts implemented.
  - Covers steps 5-9.
  - Verify focused dialog, decontextualizer, service, syntax, and prompt-render
    checks pass.
  - Evidence records changed production files and subagent report.
  - Handoff starts harness correction.
  - Sign-off: `/root`, 2026-07-25; production subagent Beauvoir completed
    the ten-file owner-contract cutover, and the parent verified all focused
    and adjacent deterministic contracts.
- [x] Stage 3 - harness and focused live verification complete.
  - Covers steps 10-13.
  - Verify harness contracts and adjacent deterministic suites pass; run real
    LLM nodes separately and inspect raw evidence.
  - Evidence links the agent-authored review and raw artifacts.
  - Handoff starts independent review.
  - Sign-off: `/root`, 2026-07-25; final deterministic batch passed
    `97 passed, 10 deselected`, all static gates passed, and individually
    inspected default and high final-source real-model probes returned valid
    visible dialog with aligned exact-shape verdicts.
- [x] Stage 4 - independent review and closeout complete.
  - Covers steps 14-19.
  - Verify review findings are resolved and affected commands rerun.
  - Evidence records reviewer identity, findings, remediation, and approval.
  - Handoff is final user delivery.
  - Sign-off: `/root`, 2026-07-25; Mendel approved the final scoped diff
    with no unresolved findings after style and symmetric regression-evidence
    remediation. Final verification passed and the plan moved to completed
    history.

## Verification

### Focused deterministic tests

- `venv\Scripts\python.exe -m pytest tests/test_dialog_visible_speech_and_semantic_fidelity.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_msg_decontextualizer.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_service_cognition_graph.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_asuna_private_r18_affinity_harness_contract.py -q`

### Adjacent deterministic regressions

- `venv\Scripts\python.exe -m pytest tests/test_dialog_agent.py tests/test_l3_dialog_content_plan_contract.py tests/test_decontextualizer_referents.py -q`

### Syntax and prompt rendering

- `venv\Scripts\python.exe -m py_compile` over every changed Python file.
- Render the decontextualizer, dialog semantic verifier, and L3 repair prompt
  constants successfully.
- `rg -n "_validated_response_operation|_render_review" src tests` returns no
  runtime helper or scripted Markdown review matches.

### Focused real LLM

- Run the repaired default-turn-01 role/selection case individually.
- Run the high-turn-02 direct action/fidelity case individually.
- Inspect each raw artifact before starting the next case.
- Produce an agent-authored Markdown review containing run context, real input
  and output, behavior assessment, deterministic validation, and raw paths.

### Excluded command

- `tests/run_asuna_private_r18_affinity_replay.py run-all` remains unexecuted.

## Independent Code Review

After all verification passes, create one review subagent. Review:

- the active plan and complete implementation diff;
- Python/CJK/prompt policy compliance;
- L3/dialog/decontextualizer ownership and retry cardinality;
- typed metadata and checkpoint accuracy;
- harness chronology, protected trace capture, and data-only output;
- deterministic and real-LLM evidence quality;
- preservation of unrelated working changes.

The parent resolves findings inside the listed change surface and reruns
affected checks. A finding that changes the fixed contract or expands scope
requires a plan update and user approval.

## Acceptance Criteria

This plan is complete when:

- the first hard-invalid dialog candidate receives one L3-owned semantic
  replacement and one second render;
- valid refusal, negotiation, or conditions pass semantic fidelity;
- a second hard-invalid candidate raises
  `dialog_compliance_contract_exhausted` with stage `dialog_compliance`,
  attempt count `2`, and checkpoint `post_cognition_commit`;
- each focused verifier repairs one structurally invalid parsed verdict in a
  system/human/assistant/human sequence without changing its semantic payload;
- two invalid verdicts from one focused verifier expose that producer's typed
  `*_contract_exhausted` error with attempt count `2` and checkpoint
  `post_cognition_commit`;
- decontextualizer repair receives the exact response-operation field error in
  a system/human/assistant/human sequence;
- a byte-identical second invalid candidate exposes
  `message_decontextualizer_unchanged_candidate_exhausted`;
- future replay user/assistant rows remain chronologically interleaved;
- replay children require full raw trace fields;
- controller code emits data artifacts rather than human judgment prose;
- all listed deterministic checks pass;
- each selected real-LLM case is individually inspected and judged acceptable;
- independent review has no unresolved findings;
- full E2E execution remains deferred.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Failure-path latency grows | Add one surface replacement stage only after verified rejection, bound its structural regeneration to two attempts, and retain two dialog candidates | Call-count tests and focused live timing |
| Surface repair changes stance | Use authoritative cognition input, preserve style/action fields, and revalidate exact output | Payload and merge tests |
| Refusal wording weakens role checks | Keep typed actor/target/selection reversal as hard errors | Focused verifier regressions |
| Repair candidate is prompt-like | Place it in an assistant message and state its data role in the correction contract | Message-role test |
| Trace assertions expose protected text publicly | Store full data only in guarded test artifacts | Harness and artifact inspection |
| Test harness writes review prose | Remove scripted Markdown generation | Static grep and controller tests |
| Focused verifier emits structurally invalid verdict | Use the user-approved one-replacement structural contract in the owning verifier; retain exact semantic context and two dialog candidates | Invalid-to-valid, two-invalid, call-count, role-order, trace, and focused live checks |

## Execution Evidence

- Pre-fix focused failures: `venv\Scripts\python.exe -m pytest` over eleven
  named focused nodes returned eleven failures. Missing evidence was exact:
  refusal/negotiation verifier wording; `repair_text_surface_planning`;
  `repair_text_surface_for_dialog`; `DialogComplianceContractError`;
  assistant-role decontextualizer repair; unchanged-candidate error code;
  typed service projection; service-time replay clock; chronology assertion;
  full trace child mode; and removal of scripted `_render_review`.
- Independent review found that the new post-commit L3 replacement path
  bypassed the canonical surface producer's two-attempt structural
  regeneration contract. The path now uses that canonical helper with an
  explicit `post_cognition_commit` checkpoint. Focused invalid-to-valid and
  two-invalid tests pass, and the exhausted result reports
  `surface_dialog_compliance_repair_contract_exhausted`, stage
  `surface.dialog_compliance_repair`, attempt count `2`, and
  `retryable=False`.
- Production-code subagent: Beauvoir (`/root/production_fix`) changed only the
  ten approved production/doc files. Its focused run passed 52 tests; its
  adjacent run identified three stale parent-owned assertions, which the
  parent updated to the big-bang contract.
- Change-surface reconciliation: the active plan now lists the three adjacent
  test files changed to align their retired expectations with the same
  user-authorized big-bang contracts. This records the already-required
  adjacent verification work without expanding production scope.
- Deterministic verification: the final combined focused and adjacent batch
  passed `97 passed, 10 deselected`. This includes invalid-to-valid,
  system/human/assistant/human role ordering, unchanged semantic context,
  rejected/accepted protected traces, bounded rejected output, and typed
  two-invalid exhaustion for all three focused verifiers.
- Syntax and prompt rendering: all eighteen changed Python files compiled;
  decontextualizer repair, dialog semantic verifier, focused-verifier
  structural repair, and L3 repair prompts rendered as non-empty strings;
  `git diff --check` passed. Runtime source has no
  `_validated_response_operation` or `_render_review` helper.
- Focused real-LLM evidence: the final-source default selection probe returned
  `蚝爹油，过来，看着我。` with one call per applicable stage and an aligned
  exact-shape verdict at
  `test_artifacts/llm_traces/dialog_visible_speech_and_semantic_fidelity__owner_repair_default_turn_01_selection__20260725T051857795661Z.json`.
  The final-source high direct-action probe returned
  `蚝爹油，既然你都这么说了……那我就依你一次。` with exact semantic and
  surface verdicts at
  `test_artifacts/llm_traces/dialog_visible_speech_and_semantic_fidelity__owner_repair_high_turn_02_direct_action__20260725T051926796347Z.json`.
  The role-direction stage correctly made zero calls for the non-selection
  case, whose actor/target direction remains owned by semantic fidelity.
- Focused evidence labels: the user texts are captured replay inputs; original
  dialog, rejected surfaces, intentions, repair issues, and role metadata are
  synthetic reconstruction fixtures. The runs are real-model contract probes,
  not original-candidate reproductions or performance proof.
- Agent-authored live review:
  `test_artifacts/cognition_core_v2/dialog_no_response_root_cause_bugfix/focused_real_llm_review.md`
  records exact commands, route/model names, code/prompt revision, accepted
  behavior, consolidated root causes, deterministic evidence, and residual
  risk.
- Prior independent review: Mendel (`/root/independent_review`) verified the
  L3 structural-regeneration correction, topology cleanup, style fixes,
  reduced selection-role projection, and non-selection semantic ownership,
  then withheld approval solely because the verifier retry expansion had not
  yet been authorized. The user approved that expansion; the existing
  reviewer will be reactivated against the completed contract and evidence.
- Residual risk: the high-case L3 result copied the prompt's prohibition on
  generic safety/content/politeness boundaries into the private
  `visible_boundaries` data. It did not enter visible dialog or alter the
  character decision and is retained as a non-fatal private-surface quality
  risk.
- Scope decision resolved: the user approved a canonical two-attempt
  structural regeneration contract for all three focused verifier stages,
  with unchanged semantic context, no extra dialog candidate, bounded trace
  evidence, and typed post-commit exhaustion metadata.
- Structural producer resolution: semantic fidelity now emits exact
  `hard_errors`, while role direction and surface integrity emit exact
  `issues`. This big-bang boundary eliminated the observed shared-vocabulary
  collision in both final-source real-model probes without adding aliases.
- Independent closeout review: Mendel (`/root/independent_review`) approved
  the completed diff with no unresolved Critical, High, Medium, or Low
  findings. It verified owner-local verifier loops, exact producer cutover,
  L3/dialog/decontextualizer ownership, typed service metadata, harness
  chronology/full traces, policy compliance, and evidence labeling.
- Review remediation: new computed returns now use named locals under P-005
  and N-006; replay adapter methods carry durable P-003 docstrings; role and
  surface invalid-to-valid tests now assert the same rejected-output cap,
  exact correction, and failed-sequence-zero/accepted-sequence-one evidence as
  semantic fidelity.
- Final post-remediation gate: the listed seven-file deterministic batch
  passed `97 passed, 10 deselected`; all eighteen changed Python files
  compiled; all four modified prompt contracts rendered; and
  `git diff --check` passed. The reviewer separately confirmed the same
  deterministic result and a post-remediation `9 passed` harness run.
- Execution exclusion confirmed: neither the full twenty-turn E2E replay nor
  any additional real-model call ran during closeout.
