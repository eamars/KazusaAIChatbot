# cognition core v2 semantic appraisal partial failure mitigation plan

## Summary

- Goal: eliminate the original semantic-appraisal partial-failure modes proven
  by delivery run `97dcb5bf43cc42248bc3d009b42fab9c` and make future protected
  failures expose their original cause chain.
- Plan class: large.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `llm-trace-debug`, and `debug-llm`.
- Overall cutover strategy: bigbang for prompt-local event handles, appraisal
  contracts, repair budgeting, route defaults, and failure-capsule schema.
- Highest-risk areas: prompt quality on local models, current-batch provenance
  retention, prompt-handle fixtures, and protected trace diagnostics.
- Acceptance criteria: the four original failed families can reach bounded
  repair or accepted reduction, current-batch citations survive reduction,
  event and evidence handles are unambiguous, the six appraisal route defaults
  are 3,072 completion tokens, and failure events retain a bounded cause chain.

## Approved Architecture Amendment - 2026-08-01

The user explicitly expanded the implementation after real-route replay showed
that one family response still coordinated too many independent output items.
This amendment supersedes the earlier one-response-path/no-extra-call wording:

- Each semantic family runs a bounded micro-appraisal loop on its existing
  configured route.
- One loop call uses singular nullable `proposition` and `delta` fields, so it
  can contain at most one of each by structure rather than array-count advice.
- An accepted empty item terminates the family; otherwise deterministic code
  accumulates accepted items and removes emitted delta paths from later turns.
- Exact repeats of an accepted proposition signature or emitted delta path are
  suppressed before validation. An item with no novel component is the same
  bounded no-progress terminator as a null/null response.
- Deterministic code wraps accepted singular items into the unchanged public
  aggregate schema and derives selected evidence and role handles from their
  actual proposition and delta citations.
- The model-facing item omits aggregate-only explanation metadata.
  Deterministic code derives the bounded public explanation from authored
  `semantic_value` and `reason` text, or uses a fixed empty-item audit message.
- Each item retains one initial attempt and at most one complete-replacement
  repair. The family loop is capped by the eight-item contract.
- No new model, provider, semantic evaluator, fallback mapper, or public output
  schema is introduced.
- The parent agent owns this user-approved amendment after the original single
  production worker completed; the required independent reviewer remains one.

## Context

Protected trace `llmtrace_e726f5c3e21d43359f4fa9ffe82db673` showed six
parallel semantic appraisals. One succeeded initially, one succeeded after
repair, and four were omitted. The original failure modes were:

- candidate event, threat, or knowledge-gap handles were used without citing
  the exact evidence handle that created the candidate;
- relationship evidence was accepted by the model and then evicted by the
  eight-row retention policy before deterministic delta validation;
- goal/threat/outcome returned ten propositions although the validator permits
  at most eight and the prompt did not expose that bound;
- four repair requests exceeded the same 8,000-character cap used for initial
  appraisal input, so the second model call was never attempted;
- event handle `e1` and evidence handle `e1` occupied different namespaces but
  appeared as the same token in one model-facing payload;
- surfaced partial-failure dispositions retained the terminal code while the
  original validation error remained separated in attempt records.

The target module is Cognition Core V2 semantic appraisal and deterministic
reduction. Configuration defaults and protected tracing change only where the
approved mitigations require their owning boundaries to change.

## Mandatory Skills

- `development-plan`: govern this execution contract, stage sign-off, review,
  evidence, and lifecycle closeout.
- `local-llm-architecture`: keep the semantic question compact, preserve LLM
  semantic ownership, and keep limits and validation deterministic.
- `py-style`: govern every Python production and test edit.
- `cjk-safety`: govern the Chinese appraisal prompt edit and immediate syntax
  verification.
- `test-style-and-execution`: keep deterministic tests strict and run every
  real-LLM gate individually with output inspection.
- `llm-trace-debug`: use the protected trace export as the captured-failure
  replay source.
- `debug-llm`: author a human-readable before/after review from real model
  output before claiming prompt-quality completion.

## Mandatory Rules

- Preserve the pipeline boundary: evidence planning supplies prompt-safe
  sources, semantic appraisal judges meaning, deterministic reducers validate
  and apply state, and tracing records protected diagnostics.
- Keep each micro-appraisal item at one normal call plus at most one repair.
  Bound each family to eight items and retain its existing model route.
- Parse every raw appraisal through `parse_llm_json_output(...)` before semantic
  validation. Repairs request a complete replacement and cannot invent facts.
- Keep stable contract instructions in the static Chinese `SystemMessage` and
  current question, evidence, state, and candidate-origin facts in the
  `HumanMessage`.
- Define edited prompt constants with triple-single-quoted strings and render
  prompt templates only with named `.format(...)` placeholders.
- Use UTF-8 for CJK file checks and run `py_compile` immediately after editing
  `semantic_appraisal.py`.
- Keep deterministic tests strict. Run real-LLM tests one case at a time,
  inspect each durable trace, and author the human-readable comparison review.
- Use `venv\Scripts\python.exe` for Python and pytest commands.
- Preserve the user-owned `.codex/config.toml` change and all unrelated
  worktree content.
- Keep `.env` unread and unchanged. This plan changes code-owned defaults and
  documented configuration examples; explicit deployment overrides remain
  operator-owned.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing.
- After signing off any major progress stage, the parent or active execution
  agent must reread this entire plan before starting the next stage.
- Before completion or lifecycle closeout, the parent must run the Independent
  Code Review gate and record its result in Execution Evidence.
- Execute through parent-led native subagents: exactly one production-code
  worker followed by exactly one independent code-review worker.

## Must Do

- Preserve the union of evidence cited by all accepted current-batch deltas for
  each mutable target before filling remaining capacity with historical rows.
- Put a compact candidate-to-origin-evidence map in the non-removable question
  contract and remove the duplicate removable candidate list from appraisal
  state projection.
- State singular nullable output rules, exact handle-field domains, the exact
  candidate-origin citation rule, and the eight-item family bound in the
  retained micro-appraisal question contract.
- Give repair input a separate 10,000-character dynamic cap and truncate the
  invalid candidate against the residual capacity after canonical payload and
  exact contract error are accounted for.
- Change prompt-local persistent event handles from `e1..eN` to `ev1..evN` in
  one clean cutover. Keep evidence handles `e1..eN` and candidate handles
  `ce1..ceN`, `ct1..ctN`, and `ck1..ckN` unchanged.
- Set the code-owned default completion budget for all six appraisal routes to
  3,072 tokens and update the HOWTO examples to the same value.
- Replace the protected failure-capsule v1 shape with v2 and attach a bounded,
  redacted exception cause chain to producer-owned failure events.
- Record detailed appraisal collection and reduction failures while preserving
  the existing validated public observability codes.
- Add focused deterministic regressions and update all affected prompt-handle
  fixtures without changing evidence handles that happen to use `eN` tokens.
- Replay captured production-shaped appraisal inputs against the configured
  real routes one case at a time and author a Markdown before/after review.
- Replace broad family-output generation with the approved bounded
  micro-appraisal loop and add deterministic accumulation, termination,
  duplicate, and call-count regressions.
- Derive per-item selected handle metadata from actual structured use and state
  the signed-integer delta representation explicitly.

## Deferred

- Keep model assignment and provider selection unchanged.
- Keep the semantic proposition vocabulary, delta axes, state limits, and
  evidence cardinality cap unchanged.
- Keep public cognition output and warning codes unchanged.
- Keep route-specific deployment overrides outside source control unchanged.
- Keep RAG, goal cognition, workspace collapse, action planning, dialog,
  persistence, adapter, and scheduler behavior unchanged.
- Keep the current two-attempt policy unchanged per micro-appraisal item.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Event handles | bigbang | Emit and accept only `ev1..evN` for prompt-local persistent events. |
| Appraisal prompt | bigbang | Use one explicit sparse-output and candidate-origin contract. |
| Repair budget | bigbang | Use the 10,000-character repair cap and residual-aware candidate slice. |
| Route defaults | bigbang | Use 3,072 for all six appraisal defaults and documentation examples. |
| Failure capsule | bigbang | Persist `cognition_failure_capsule.v2` with bounded event cause chains. |
| Tests | bigbang | Update event-handle fixtures in place; preserve evidence `eN` handles. |

Cutover enforcement:

- Rewrite legacy event-handle expectations instead of adding aliases.
- Preserve only the handle namespaces and public output codes explicitly named
  in this plan.
- Any cutover-policy change requires user approval before implementation.

## Target State

One appraisal receives a compact question contract that distinguishes
`evN` persistent event handles, `eN` evidence handles, and `ceN`/`ctN`/`ckN`
candidate handles. Candidate origin evidence remains visible even when
supplemental state is dropped for the initial prompt budget. A structurally
invalid first response can reach the existing second attempt because repair
payload construction reserves canonical input first and uses only residual
capacity for the invalid candidate. Accepted current-batch deltas retain their
cited source rows before deterministic transition guards execute.

Protected partial failures retain both the stable public failure code and an
ordered bounded chain of exception type and redacted message in the v2 capsule.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Current evidence | Pin the accepted batch union per target, then retain newest historical rows in remaining slots. | Every applied delta must keep its complete provenance under the existing eight-row cap. |
| Candidate origin | Put `candidate_origin_evidence` inside `question`. | Question content is retained while supplemental state is the first budget-reduction surface. |
| Candidate state | Remove `causal_candidates` from appraisal state payload. | The origin map is the only model-facing candidate contract and avoids duplication. |
| Repair cap | Use 10,000 dynamic characters, canonical payload first, error second, candidate residual last. | All four measured failed repair payloads can reach the second call without raising the initial cap. |
| Event namespace | Use `evN`. | It is visually and structurally distinct from evidence `eN` and candidate `ceN`. |
| Completion budget | Use 3,072 tokens for six appraisal route defaults. | Each call emits at most one short proposition and one short delta. |
| Cause chain | Add at most four ordered exception entries to each marked failure event. | It links terminal containment to the original contract or reducer failure without exposing public output. |
| Micro output shape | Request singular nullable `proposition` and `delta` fields, then wrap them into the aggregate arrays deterministically. | Array output repeatedly invited the model to batch three items despite a one-item instruction; singular fields make cardinality structural. |
| Selection metadata | Derive selected handles from accepted proposition and delta fields before validation. | Selection is deterministic provenance bookkeeping; omitting it from model output prevents copied permission lists from masking semantic contract errors. |
| Explanation metadata | Derive it from accepted item text instead of requesting an independent model field. | The real route returned `null` twice despite valid singular semantic content; explanation has no state authority and should not invalidate that content. |
| Loop progress | Suppress exact accepted proposition signatures and emitted delta paths before validation; terminate if no novel component remains. | The local model repeated item 1 on item 2 even with exclusion metadata; repetition is a deterministic no-progress signal rather than a reason to discard the accepted family result. |
| Event identity | Permit role-signature fallback matching only against active events; retain exact canonical-ID matching for every status. | A new candidate event with a distinct source-derived ID was falsely matched to a resolved historical event through a generic relationship role, making a valid terminal assertion impossible. |
| Delta scalar | Require a JSON integer from -40 through 40 in prompt and validation errors. | The real route repeatedly emitted the string `"+0.05"`; an explicit representation and unit contract removes that ambiguity. |

## Contracts And Data Shapes

The question payload adds:

```python
"candidate_origin_evidence": {
    "ce1": "e1",
    "ct1": "e1",
    "ck1": "e1",
}
```

Each model-facing micro call returns exactly:

```python
{
    "question_id": str,
    "proposition": dict | None,
    "delta": dict | None,
}
```

Deterministic code converts that internal producer shape into the existing
`SemanticAppraisalResultV2` arrays and selected-handle fields before contract
validation and accumulation.

Only candidate handles permitted for that question appear. The mapping is
derived from private `handle_to_ref` bindings and authorized evidence handles.

Prompt-local handle namespaces after cutover:

| Kind | Handle |
|---|---|
| persistent event | `ev1..evN` |
| evidence | `e1..eN` |
| candidate event | `ce1..ceN` |
| candidate threat | `ct1..ctN` |
| candidate knowledge gap | `ck1..ckN` |

Failure event v2 adds this optional field only when an exception is supplied:

```python
"cause_chain": [
    {"type": "CognitionContextLimitError", "message": str},
    {"type": "ValueError", "message": str},
]
```

The chain is outermost-first, capped at four entries, cycle-safe, and redacted
through the session's protected secret set.

## LLM Call And Context Budget

| Call | Before | After | Blocking and limits |
|---|---|---|---|
| Initial appraisal | One response-path call; 8,000 dynamic characters; route default 8,192 completion tokens. | Up to eight serial singular-item calls per family; each has an 8,000-character dynamic cap and 3,072 completion tokens. | The six families still start in parallel; each family terminates on empty or exact-repeat output. |
| Appraisal repair | Existing second attempt; same 8,000 dynamic-character cap caused pre-call overflow. | At most one replacement call per item; 10,000 dynamic characters with residual-aware invalid-candidate truncation and 3,072 completion tokens. | Worst case is sixteen calls per family; no item receives more than two attempts. |

The default 50,000-token context-window cap remains unchanged. The character
caps are conservative pre-tokenization limits. Canonical question/evidence is
retained ahead of repair candidate text.

## Change Surface

### Modify production

- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`: explicit
  prompt contract, candidate-origin question mapping, and repair budget.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py`: per-target
  current-batch evidence union and retention order.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py`: emit `evN`
  persistent event handles.
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py`: use the
  exact `ev` event namespace in family ownership.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`: record detailed protected
  appraisal collection and reduction failure events.
- `src/kazusa_ai_chatbot/llm_tracing/failure_capsule.py`: v2 cause-chain shape.
- `src/kazusa_ai_chatbot/config.py`: six appraisal route defaults.
- `src/kazusa_ai_chatbot/cognition_core_v2/validation_cli.py`: update its
  prompt-local event fixture if the cutover reaches that fixture.

### Modify tests and documentation

- `tests/test_cognition_core_v2_prompt_budget_continuity.py`
- `tests/test_cognition_core_v2_semantic_terminalization.py`
- `tests/test_cognition_prompt_contract_text.py`
- `tests/test_cognition_core_v2_stage_model_routing.py`
- `tests/test_llm_tracing.py`
- `tests/test_llm_trace_export.py`
- `tests/test_cognition_core_v2_failures.py`
- `tests/cognition_core_v2_test_helpers.py` and affected Cognition Core V2 test
  fixtures that encode persistent event handle `eN`.
- `docs/HOWTO.md`
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
- `src/kazusa_ai_chatbot/llm_tracing/README.md`
- `test_artifacts/diagnostics/llm_trace_appraisal_mitigation_review_97dcb5bf43cc42248bc3d009b42fab9c.md`

### Create

- This active bugfix plan and the human-readable real-LLM comparison review.

### Keep

- All public Cognition Core V2 input/output schemas and stable warning codes.
- The original protected export and RCA artifacts.
- The user-owned `.codex/config.toml` modification.

## Overdesign Guardrail

- Actual problem: multiple scoped appraisals were omitted because prompt
  contracts, reducer retention, repair budgeting, handle namespaces, and
  failure disposition did not align with their deterministic validators.
- Revised minimum after real-route evidence: align the original boundaries and
  replace broad family generation with bounded singular calls, without adding
  model routes, semantic evaluators, or public outputs.
- Ownership boundaries: the LLM judges semantic meaning; deterministic code
  owns handles, counts, provenance retention, repair limits, configuration,
  state application, and trace persistence.
- Rejected complexity: additional retries, verifier models, automatic origin
  evidence invention, weaker validation, larger initial prompts, compatibility
  aliases, feature flags, and model reassignment.
- Evidence threshold: a new observed failure with protected trace evidence and
  focused reproduction is required before expanding these boundaries again.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve every contract in this plan.
- The responsible agent must not introduce new architecture, alternate
  cutovers, aliases, fallback paths, or extra features.
- Changes outside the named production files require a concrete compile or
  test failure proving the cutover reaches that file.
- Before adding a helper, search for equivalent behavior and use an existing
  contract when present.
- Keep edits surgical and avoid unrelated cleanup, formatting churn,
  dependency updates, or prompt rewrites.
- If this plan and current code disagree materially, stop and report the exact
  discrepancy before changing the contract.

## Implementation Order

1. Parent adds focused failing tests for batch evidence retention, candidate
   origin placement, explicit prompt bounds, residual-aware repair, `evN`
   handles, six 3,072 route defaults, and v2 cause chains.
2. Parent runs each focused deterministic test and records the expected
   baseline failure or current contradictory behavior.
3. Parent starts exactly one production-code subagent with ownership of the
   production files listed above and the frozen test contract.
4. Parent updates impacted test fixtures and documentation while the worker
   implements production code in its forked workspace.
5. Parent reviews and integrates the worker patch, then runs syntax, prompt
   rendering, focused tests, and the affected non-live regression suite.
6. Parent runs captured production-shaped real-LLM appraisal cases one at a
   time, inspects output, and authors the comparison review artifact.
7. Parent starts exactly one independent code-review subagent after planned
   verification passes, remediates in-scope findings, and reruns affected gates.
8. Parent records evidence, marks this plan completed, moves it to the completed
   bugfix archive, and updates the registry.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review remediation, lifecycle updates, and final sign-off.
- Parent establishes and runs the focused test contract before production
  implementation begins.
- Production-code subagent: exactly one native `kazusa_plan_worker`; owns only
  the listed production Python files; does not edit tests; closes after planned
  production changes are complete.
- Parent may update test fixtures, documentation, static checks, and evidence
  while the production worker edits its disjoint production scope.
- Independent code-review subagent: exactly one native
  `kazusa_plan_reviewer`; starts after verification, reviews only, and reports
  findings without implementing fixes.

## Progress Checklist

- [x] Stage 1 - focused test contract established.
  - Covers: implementation steps 1-2.
  - Verify: named focused tests fail for the expected pre-fix behavior.
  - Evidence: record commands and failure reasons below.
  - Handoff: start the production worker at Stage 2.
  - Sign-off: parent, 2026-08-01; all eight focused cases failed for their
    expected pre-fix contract gaps.
- [x] Stage 2 - production contracts implemented.
  - Covers: implementation steps 3-4.
  - Verify: CJK syntax check, `py_compile`, and focused tests pass.
  - Evidence: record worker identity, changed files, and focused results.
  - Handoff: proceed to regression and real-LLM verification.
  - Sign-off: parent, 2026-08-01; the production worker patch and parent
    containment/provenance corrections compile, and all 11 focused cases pass.
- [x] Stage 3 - regression and real-LLM quality gates complete.
  - Covers: implementation steps 5-6.
  - Verify: affected non-live suite passes; each real case is run and inspected
    individually; comparison review exists.
  - Evidence: record commands, artifact paths, and qualitative judgment.
  - Handoff: start independent review.
  - Sign-off: parent, 2026-08-01; 137 affected tests and 373 broad non-live
    Cognition/tracing tests pass, all six individually inspected real-route
    replays succeed, and the before/after review is complete.
- [x] Stage 4 - independent code review and closeout complete.
  - Covers: implementation steps 7-8.
  - Verify: reviewer approves after in-scope findings are remediated and
    affected commands rerun.
  - Evidence: record reviewer identity, findings, fixes, reruns, residual risk,
    and lifecycle paths.
  - Handoff: final user report.
  - Sign-off: parent, 2026-08-01; the one independent review found no P0/P1
    issue, all two P2 findings and one P3 finding were remediated, and the
    post-remediation affected and broad suites pass.

## Verification

### Static and syntax

- `venv\Scripts\python.exe -m py_compile` on every changed Python file exits
  zero.
- Runtime rendering of `SEMANTIC_APPRAISAL_PROMPT` and a maximum appraisal
  payload succeeds and remains within its declared cap.
- `rg -n 'active_events\.e[0-9]|"e[0-9]+".*"kind": "event"' src tests`
  returns no persistent-event-handle matches; evidence-handle matches remain
  allowed.
- `rg -n 'cognition_failure_capsule\.v1' src tests` returns no matches.

### Focused deterministic tests

- Run the new and changed cases in
  `tests/test_cognition_core_v2_prompt_budget_continuity.py` individually.
- Run the current-batch retention case in
  `tests/test_cognition_core_v2_semantic_terminalization.py` individually.
- Run prompt contract and stage routing cases individually.
- Run v2 cause-chain cases in `tests/test_llm_tracing.py` and
  `tests/test_cognition_core_v2_failures.py` individually.

### Regression

- `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_prompt_budget_continuity.py tests\test_cognition_core_v2_semantic_terminalization.py tests\test_cognition_prompt_contract_text.py tests\test_cognition_core_v2_stage_model_routing.py tests\test_cognition_core_v2_failures.py tests\test_llm_tracing.py tests\test_llm_trace_export.py -q`
- Run all non-live Cognition Core V2 tests reached by the handle cutover.

### Real LLM

- Replay each of the six captured appraisal families one at a time through its
  configured route.
- Inspect raw output, parsed output, validation disposition, repair behavior,
  proposition and delta counts, candidate-origin citations, and reduction.
- Author the required before/after Markdown review from the real output and
  link the protected raw evidence path.

## Independent Code Review

Run this gate after all planned verification passes and before lifecycle
closeout. Start one native `kazusa_plan_reviewer` with this plan, the complete
diff excluding the user-owned `.codex/config.toml` change, test outputs, static
results, and the real-LLM review artifact.

Review scope:

- project style, CJK safety, prompt contract clarity, local-model suitability,
  and test execution compliance;
- exact alignment with Must Do, Deferred, cutover, change surface, context
  budgets, and acceptance criteria;
- current-batch evidence invariants, handle namespace correctness, repair-call
  reachability, route default ownership, cause-chain redaction, and public
  output stability;
- fixture precision, regression coverage, execution evidence, and absence of
  unrelated changes.

The parent fixes concrete in-scope findings and reruns affected checks. A
finding that changes a frozen contract or leaves the named change surface
requires a plan update and user approval.

## Acceptance Criteria

This plan is complete when:

- the reproduced relationship reduction keeps its current source evidence and
  applies without `semantic delta evidence handle is unknown`;
- candidate use in any structured field requires and receives its exact origin
  evidence citation;
- the prompt states sparse output and maximum eight-item contracts;
- all four formerly oversized repair requests can construct and reach the
  second attempt within 10,000 dynamic characters;
- persistent event handles are `evN` everywhere in the active prompt contract;
- all six appraisal route defaults and HOWTO examples use 3,072 completion
  tokens;
- protected failure capsules use v2 and preserve bounded cause chains while
  public observability codes remain unchanged;
- focused and affected non-live tests pass;
- six real-LLM cases are individually inspected and the comparison review finds
  no recurrence of the original failure modes;
- independent code review is complete with no unresolved in-scope finding.

## Execution Evidence

- Baseline trace: `test_artifacts/diagnostics/llm_trace_delivery_97dcb5bf43cc42248bc3d009b42fab9c.json`.
- RCA: `test_artifacts/diagnostics/llm_trace_rca_97dcb5bf43cc42248bc3d009b42fab9c.md`.
- Focused baseline tests: eight individual pytest cases reproduced missing
  `candidate_origin_evidence`, `eN` persistent-event handles, repair
  pre-call context overflow, relationship current-evidence eviction, missing
  prompt count/sparse rules, missing 3,072 default, absent capsule exception
  argument/v2 shape, and absent appraisal cause-chain failure event.
- Production worker: `kazusa_plan_worker`
  `019fbce2-4262-7e43-a2df-11f3e1721121` (Hooke), completed and closed.
- Changed files: the seven planned production Python files, eight focused or
  cutover test files, three subsystem/user documents, the registry, this plan,
  and the ignored replay harness. `validation_cli.py` and shared test helpers
  required no event-handle update.
- Static and syntax results: `py_compile` exited zero for every changed Python
  file and the ignored replay harness after the CJK prompt edit.
- Focused deterministic results: 11 passed, including candidate origins,
  `evN`, residual repair, relationship and causal retention, reducer isolation,
  prompt rules, route defaults, capsule v2, and original-cause recording.
- Regression results: 137 passed in the affected suite; 373 passed and 169
  live tests were deselected in the broad Cognition/tracing suite. The broad
  gate excluded the documented unrelated replay-clock and live-character-
  judgment modules.
- Real-LLM raw evidence: six
  `test_artifacts/diagnostics/semantic_appraisal_replay_97dcb5bf43cc42248bc3d009b42fab9c_*.json`
  artifacts; all six succeeded, five with one accepted item plus bounded
  termination and one with an immediate valid empty result. Every emitted
  delta was an integer and every result trial-reduced successfully.
- Human-readable comparison review:
  `test_artifacts/diagnostics/llm_trace_appraisal_mitigation_review_97dcb5bf43cc42248bc3d009b42fab9c.md`.
- Independent reviewer: `kazusa_plan_reviewer`
  `019fbd26-c099-7eb1-88d7-8410e8f7c06a` (Darwin), completed and closed.
- Review findings and remediation: no P0 or P1 findings. Two P2 findings were
  remediated by correcting the documented attempt cap and adding explicit
  cause-chain truncation, cycle, and context-fallback redaction regressions.
  One P3 style finding was remediated by assigning computed values to named
  locals before return. The parent reran reviewer-targeted tests (4 passed),
  the affected suite (137 passed), and the broad suite (373 passed, 169
  deselected); no in-scope finding remains unresolved.
- Residual risks: the live route reports an operator-owned 8,192-token
  completion override despite the 3,072 code default; singular output keeps
  responses bounded. Worst-case family latency is sixteen calls, while the
  captured final replays required one or two calls.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Local model omits output after stricter wording | Explicitly state sparse empty lists are valid and inspect six live families. | One-at-a-time real-LLM review. |
| Handle cutover changes evidence fixtures accidentally | Update only refs whose kind is event or paths under `active_events`. | Static grep plus affected non-live suite. |
| Pinned batch evidence exceeds eight rows | Question evidence is bounded at eight; reject any reducer state that cannot retain the complete current union. | Focused maximum-batch retention test. |
| Cause chain leaks credentials | Apply existing protected redaction to every chain message. | Secret-redaction tracing test. |
| 3,072 completion tokens truncate valid output | Maximum-output real-LLM case and parsed-contract inspection. | Individual maximum-family live gate. |
