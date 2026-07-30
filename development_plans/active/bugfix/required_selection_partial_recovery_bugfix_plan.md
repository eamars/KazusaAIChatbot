# Required-Selection Partial-Recovery Bugfix Plan

Status: draft

Plan class: large

Cutover: bigbang

## Summary

Reproduce the required-selection contract failure with the configured real
31B cognition model before changing production code. Stop the work if a
bounded set of production-shaped live probes does not produce a usable,
model-authored failure.

When the reproduction gate passes, remove the unused `selection_kind` output
field, add a narrow deterministic normalization boundary for structurally
recoverable candidate data, and let cognition continue when a required branch
fails but another branch has already produced a complete valid model-authored
bid. Preserve bounded regeneration and the current typed failure when no
usable candidate or alternate bid exists.

This plan follows the completed
`required_selection_contract_separation_bugfix_plan.md`; it does not reopen
the dense-model routing, relation-matrix removal, or three-attempt producer
policy established there.

## Context

The production failure associated with
`chat:qq:ch_1f677493d7a52025:1634535291` and trace
`llmtrace_0cd78d39a55e48f6ae8efb662262eb70` reached the required-selection
goal producer successfully and then exhausted its contract attempts:

- relevance and decontextualization succeeded;
- the required-selection route used
  `gemma-4-31b-fable-5-agent-distill`;
- two service graph attempts each ran three local attempts for both
  `ordinary_response` and `autonomy_boundary`;
- all twelve goal outputs were recorded as `contract_error`;
- the required `ordinary_response` failure raised
  `goal_bid_structure_exhausted` before workspace collapse;
- the metadata trace omitted raw model output and parsed candidates, so it
  proves the failure class but not the exact invalid field;
- an adjacent same-channel trace recovered after regeneration, proving that
  the route and model were loaded and that the failure is context-sensitive.

The current producer has two all-or-nothing surfaces:

1. `validate_selection_goal_draft(...)` requires an exact eight-field object,
   even though `selection_kind` is discarded by
   `_selection_goal_draft_to_goal_bid(...)`.
2. `_raise_for_failed_required_branches(...)` raises before workspace
   collapse, even when another branch has already authored a complete valid
   bid.

The existing trace evidence is sufficient to design a reproduction, but the
production code change is gated on capturing a new real-model output that
demonstrates a recoverable instance of this failure class.

## Mandatory Skills

Execution must read and apply these skills in full before the corresponding
work:

- `.agents/skills/development-plan/SKILL.md`
- `.agents/skills/local-llm-architecture/SKILL.md`
- `.agents/skills/debug-llm/SKILL.md`
- `.agents/skills/test-style-and-execution/SKILL.md`
- `.agents/skills/no-prepost-user-input/SKILL.md`
- `.agents/skills/py-style/SKILL.md`
- `.agents/skills/cjk-safety/SKILL.md`

After context compaction, reread this entire plan and every mandatory skill
before continuing. Reread the entire plan after the reproduction signoff and
before final verification.

## Mandatory Rules

- Run live LLM tests one case at a time and inspect the durable artifact after
  every case.
- Use `venv\Scripts\python.exe` for Python and pytest.
- Pass every raw model response through
  `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` before contract
  evaluation.
- Keep the 31B dense route and configured endpoint unchanged. The failing run
  already proves that the new worker and model route were loaded.
- Keep semantic judgment in the required-selection LLM producer.
- Deterministic code may remove unusable structure only under the recovery
  matrix in this plan. It may not invent, paraphrase, copy, or default a
  selection, reason, private monologue, target, confidence, citation, or
  consequence.
- Forward a recovered candidate only after it passes the same canonical strict
  validator used by an unchanged candidate.
- Keep the existing local three-attempt cap and service safe-retry boundary.
- Add no JSON-repair LLM, verifier LLM, classifier LLM, or extra healthy-path
  call.
- Preserve the typed fail-closed result when no complete candidate or complete
  alternate bid exists.
- Use one canonical producer contract. Add no alias fields, compatibility
  mapper, dual schema, or legacy execution path.
- Preserve unrelated user changes in the worktree.

## Must Do

- Add three fixed, production-shaped real LLM probes that retain raw model
  output and validator diagnostics.
- Make successful pre-fix reproduction a hard implementation gate.
- Stop before production-code edits if no probe produces a usable failure.
- Record the exact model-authored defect and map it to the fixed recovery
  matrix before implementation.
- Remove the unused `selection_kind` producer field.
- Normalize only recoverable structural defects.
- Continue with complete alternate model bids when a required branch fails.
- Keep total-invalid execution fail-closed.
- Add deterministic contract, integration, retry-continuity, and prompt-budget
  tests.
- Re-run the reproduced and adjacent live cases after the fix.
- Produce a human-readable pre-fix/post-fix LLM review artifact.
- Complete an independent code review after verification.

## Deferred

- Any model or endpoint change.
- Any `.env` change.
- Any change to the 31B dense required-selection route.
- Any increase to local or service retry limits.
- General goal-bid schema redesign outside required selection.
- Changes to RAG, cognition state, action planning, dialog wording, or adapter
  delivery.
- Recovery of missing or invalid model-authored semantic decisions.
- Historical trace reconstruction beyond the already exported RCA evidence.

## Cutover Policy

Use a big-bang contract cutover:

- the prompt, validator, mapper tests, and prompt-budget assertions move to one
  seven-field required-selection contract in the same change;
- the previous eight-field contract has no runtime fallback;
- `selection_kind` is removed rather than aliased or translated;
- structural recovery runs once at the canonical producer boundary before
  strict validation;
- required-branch failure handling moves to the new
  valid-alternate-bid rule in the same change;
- rollback is a source revert of this complete change, not a runtime flag or
  dual path.

## Target State

```text
real required-selection response
  -> canonical deterministic JSON parsing
  -> narrow structural normalization
       -> unchanged candidate
       -> recovered candidate with recorded recovery codes
       -> unrecoverable candidate, regenerate within current cap
  -> strict seven-field validation
  -> complete ActionBidV2
  -> branch execution
       -> required branch valid: normal collapse
       -> required branch failed + complete sibling bid: collapse sibling bids
       -> no complete bid: existing typed safe retry/failure
```

No `None`, partial dictionary, or invalid `ActionBidV2` crosses into workspace
collapse. “Nullify” means omitting a noncanonical field or invalid optional
list row from the normalized candidate. It does not mean forwarding `None` in
a field required by the next-stage contract.

## Design Decisions

### Root-cause hypothesis to test

The candidate root cause is an overconstrained all-or-nothing boundary, not a
worker-loading failure:

- the prompt asks the model to emit `selection_kind`, but the mapper discards
  it;
- exact-key validation rejects an otherwise complete candidate because of any
  additional field;
- list validation rejects the whole candidate when valid required data
  coexists with a bad optional row;
- facade pre-collapse validation discards complete sibling bids when the
  required ordinary branch exhausts.

The live reproduction must prove at least one of these recovery surfaces before
implementation proceeds.

### Usable reproduction definition

A live probe is usable only when all of the following are true:

1. The configured provider returns nonempty raw model output.
2. The failure is a parser or required-selection contract failure rather than
   a provider, timeout, route, or endpoint failure.
3. The raw and parsed output are preserved in the test artifact with the exact
   validator error.
4. At least one rejected candidate either:
   - retains valid model-authored `selection`, `reason`,
     `private_monologue`, `confidence`, every mandatory evidence citation, and
     at least one valid consequence, while failing only a recovery-matrix row;
     or
   - belongs to an execution where the required branch failed and a sibling
     branch produced a complete valid `ActionBidV2`.

If all three fixed probes finish without such evidence, execution stops. The
agent records the outputs, leaves production code untouched, and returns the
plan to draft amendment.

### Canonical seven-field producer contract

The required-selection model owns exactly:

```text
selection
reason
private_monologue
target_role_handles
evidence_handles
expected_consequences
confidence
```

`selection_kind` is removed because it has no downstream consumer. Branch
identity and required-selection activation remain deterministic inputs, not
model-authored output fields.

### Fixed recovery matrix

The normalizer applies these rules in order and records a stable recovery code
for each change:

| Input defect | Disposition | Recovery code |
|---|---|---|
| Parsed value is not an object | unrecoverable | `selection_not_object` |
| Unknown top-level keys coexist with all seven canonical fields | remove unknown keys | `extra_fields_removed` |
| A canonical field is missing | unrecoverable | `required_field_missing` |
| `selection`, `reason`, `private_monologue`, or `confidence` has surrounding whitespace | strip surrounding whitespace | `semantic_text_trimmed` |
| A required semantic string is wrong-type, empty after trimming, or over its bound | unrecoverable | `semantic_text_invalid` |
| `target_role_handles` has exact duplicate valid handles | deduplicate in first-seen order | `target_handles_deduplicated` |
| A target handle is wrong-type or outside the allowed role set | unrecoverable | `target_handle_invalid` |
| `evidence_handles` has duplicate valid handles | deduplicate in first-seen order | `evidence_handles_deduplicated` |
| An evidence row is wrong-type, empty, or outside the allowed evidence set | drop that row | `invalid_optional_evidence_removed` |
| Required operation or progress evidence is absent after evidence normalization | unrecoverable | `required_evidence_missing` |
| `expected_consequences` is not a list | unrecoverable | `consequences_not_list` |
| A consequence row is wrong-type, empty after trimming, or over 240 characters | drop that row | `invalid_consequence_removed` |
| Valid consequence rows have surrounding whitespace | trim them | `consequence_trimmed` |
| No valid consequence remains or more than eight valid rows remain | unrecoverable | `consequences_invalid` |

The strict validator runs after normalization. A candidate that still fails is
unrecoverable for that attempt and follows current bounded regeneration.

### Branch-level partial recovery

Rename `_raise_for_failed_required_branches(...)` to
`_raise_for_unrecoverable_required_branch_failures(...)` and apply this exact
rule:

- if a required branch failed and `execution.results` contains no complete
  mapping bid, raise the current typed `CognitionExecutionError`;
- if a required branch failed and `execution.results` contains one or more
  complete mapping bids, preserve the failure record, append
  `required_branch_recovered_by_valid_bid:<branch_id>` to execution warnings,
  and continue;
- workspace collapse receives only the complete bids already returned by
  model-owned branches;
- deterministic code creates no substitute ordinary-response bid.

### Observability

For a successfully normalized candidate:

- record the goal trace step with `parse_status="recovered"`;
- preserve the normalized parsed output under the existing full-capture trace
  policy;
- include stable recovery codes in the full-capture parsed-output diagnostic
  envelope used by the live test artifact;
- retain `parse_status="succeeded"` for candidates requiring no normalization.

Metadata-only production traces remain bounded and do not gain raw prompt or
raw output storage.

## Contracts And Data Shapes

Add one private normalizer in `goal_cognition.py`:

```python
def _normalize_selection_goal_draft(
    parsed: object,
    *,
    evidence_handles: set[str],
    role_handles: set[str],
    required_evidence_handles: set[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    ...
```

The function returns a new dictionary and ordered unique recovery-code tuple.
It mutates no parsed object. It raises `ValueError` with one stable
unrecoverable code when a rule cannot preserve the model-authored semantic
core.

Call order in `run_goal_cognition(...)` is fixed:

1. `parse_llm_json_output(response_text, deterministic_only=True)`
2. `_normalize_selection_goal_draft(...)`
3. `validate_selection_goal_draft(...)`
4. `_selection_goal_draft_to_goal_bid(...)`
5. existing complete `ActionBidV2` construction

No public state schema, database document, queue envelope, or adapter contract
changes.

## LLM Call And Context Budget

| Surface | Before | After |
|---|---:|---:|
| Required-selection local attempts per branch | up to 3 | up to 3 |
| Service safe graph retry | 1 | 1 |
| Added repair/evaluator calls | 0 | 0 |
| Healthy-path call count | 1 per selected branch | unchanged |
| Required-selection system prompt | eight-field contract | shorter seven-field contract |
| Aggregate prompt cap | current cap | unchanged |

Candidate recovery may end a local retry sequence earlier. Complete sibling-bid
recovery may avoid a service graph retry. Fully unrecoverable output retains
the existing worst-case call count and typed failure.

## Change Surface

### Production

- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
  - simplify the prompt contract;
  - add structural normalization;
  - record recovered trace status;
  - preserve strict post-normalization validation.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`
  - replace unconditional required-branch failure with the complete
    alternate-bid rule.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
  - document candidate recovery, alternate-bid continuation, and the remaining
    fatal boundary.

### Tests

- `tests/test_cognition_core_v2_required_selection_live_llm.py`
- `tests/test_cognition_core_v2_dependencies.py`
- `tests/test_cognition_core_v2_integration.py`
- `tests/test_cognition_core_v2_model_retry_continuity.py`
- `tests/test_cognition_core_v2_prompt_budget_continuity.py`
- `tests/test_service_input_queue.py`

### Evidence

- `test_artifacts/llm_traces/` for raw per-case live traces.
- `test_artifacts/reviews/required_selection_partial_recovery_live_llm_review.md`
  for the agent-authored pre-fix/post-fix comparison.

### Lifecycle

- this plan;
- `development_plans/README.md`.

## Overdesign Guardrail

- One private normalizer in the existing producer module.
- One canonical seven-field schema.
- One warning string for branch-level recovery.
- No new service, graph stage, model route, feature flag, persistence schema,
  compatibility layer, or generic recovery framework.
- No attempt to make arbitrary malformed semantic content recoverable.
- Do not broaden the change beyond a defect observed by the pre-fix live gate
  and covered by the fixed matrix.

## Agent Autonomy Boundaries

The executing agent may:

- add the three named live probes and their artifact capture;
- implement only the fixed recovery matrix;
- make mechanical test and documentation updates required by the cutover;
- stop early when the reproduction gate fails.

The executing agent must stop and request a plan amendment when:

- no usable failure is reproduced;
- the reproduced defect is outside the fixed recovery matrix;
- recovery would require inventing or defaulting semantic content;
- the fix would require a model, endpoint, `.env`, retry-cap, public schema, or
  persistence change;
- a complete sibling bid cannot safely enter the existing collapse contract;
- unrelated worktree changes overlap a target file and cannot be preserved.

## Implementation Order

1. Re-read this plan, mandatory skills, `README.md`, `docs/HOWTO.md`,
   `src/kazusa_ai_chatbot/cognition_core_v2/README.md`, the completed
   required-selection separation plan, current git status, and every target
   source/test file.
2. Add only the live reproduction cases and test-side diagnostic capture.
3. Run the three pre-fix probes in the fixed order below, one at a time,
   stopping after the first usable reproduction:
   1. `test_live_third_party_reply_ordinary_selection_contract_pressure`
   2. `test_live_third_party_reply_autonomy_selection_contract_pressure`
   3. `test_live_parallel_third_party_reply_selection_contract_pressure`
4. Inspect each raw artifact immediately. Write the pre-fix evidence and exact
   recovery-matrix classification into the review artifact.
5. Apply the reproduction hard gate:
   - usable reproduction: continue;
   - no usable reproduction after all three probes: stop with no production
     edit;
   - defect outside the matrix: stop for plan amendment.
6. Add deterministic red tests for the reproduced candidate defect, every
   adjacent recovery-matrix row, and the branch-level complete-sibling case.
7. Implement the seven-field prompt and validator cutover plus the private
   normalizer in `goal_cognition.py`.
8. Run Python parse/compile checks for the edited Python file immediately
   because it contains CJK prompt text.
9. Implement the complete-alternate-bid continuation in `facade.py`.
10. Run focused deterministic tests and repair only regressions within this
    plan.
11. Update the cognition-core README.
12. Run each reproduced and adjacent live LLM case separately and inspect its
    artifact.
13. Complete the full non-live cognition-core regression, service boundary
    tests, static checks, and diff inspection.
14. Run one independent code-review subagent, remediate valid findings, and
    repeat affected verification.
15. Record commands, artifact paths, model route, pre-fix defect, post-fix
    disposition, test results, review findings, and final signoff in this plan.

## Execution Model

Execution uses parent-led native subagents:

- the parent owns the reproduction gate, live-test inspection, stop decision,
  final integration, and plan evidence;
- after the reproduction gate passes, one focused implementation subagent may
  modify the production and deterministic test surface;
- the parent runs all live LLM tests and final verification;
- one fresh independent review subagent inspects the final diff and evidence.

Only one production-code subagent works at a time. Subagents receive this plan,
the fixed matrix, exact file ownership, and explicit instruction to preserve
unrelated changes.

If native subagents are unavailable at an implementation or review gate,
execution stops until the user explicitly authorizes a fallback execution
model.

## Progress Checklist

- [x] Pre-edit repository and instruction refresh complete.
- [x] Three fixed live probes added without production edits.
- [x] Pre-fix probes run one at a time with artifacts inspected.
- [ ] Usable failure reproduced and mapped to the fixed matrix.
- [ ] Reproduction hard gate signed off.
- [ ] Deterministic red tests added.
- [ ] Seven-field producer cutover implemented.
- [ ] Structural recovery matrix implemented.
- [ ] Complete-alternate-bid continuation implemented.
- [ ] Focused deterministic tests pass.
- [ ] Cognition-core documentation updated.
- [ ] Reproduced live case passes after the fix.
- [ ] Adjacent live cases pass one at a time.
- [ ] Full non-live regression and static checks pass.
- [ ] Independent review complete and findings resolved.
- [ ] Execution evidence and acceptance criteria signed off.

## Verification

### Pre-fix live reproduction gate

Run each command separately and inspect the newly written artifact before
running the next command:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_required_selection_live_llm.py::test_live_third_party_reply_ordinary_selection_contract_pressure -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_required_selection_live_llm.py::test_live_third_party_reply_autonomy_selection_contract_pressure -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_required_selection_live_llm.py::test_live_parallel_third_party_reply_selection_contract_pressure -q -s
```

The live fixture must record:

- case name and production trace inspiration;
- rendered system and human messages;
- model route and resolved model name;
- raw response for every attempt;
- canonical parsed output or parser error;
- exact current-validator error;
- required citation set;
- recovery-matrix classification;
- branch result and graph result.

### Deterministic focused verification

```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_dependencies.py tests\test_cognition_core_v2_integration.py tests\test_cognition_core_v2_model_retry_continuity.py tests\test_cognition_core_v2_prompt_budget_continuity.py tests\test_service_input_queue.py -m "not live_llm and not live_db" -q
```

Required deterministic assertions:

- each recovery-matrix row has a positive or negative contract test;
- all recovered candidates pass the canonical strict validator;
- no test observes a partial or `None`-filled `ActionBidV2`;
- missing semantic fields, invalid semantic strings, invalid target handles,
  missing mandatory citations, and zero valid consequences remain
  unrecoverable;
- total invalidity still raises `goal_bid_structure_exhausted`;
- required ordinary failure plus a complete autonomy bid reaches collapse;
- required ordinary failure with zero complete bids still raises;
- healthy candidates remain unchanged and record `succeeded`;
- normalized candidates record `recovered`;
- call counts and retry caps remain unchanged;
- the prompt stays within the current aggregate cap.

### Post-fix live verification

Re-run the exact reproduced command first. Then run the other two live cases
one at a time. Each must preserve complete model-authored semantics and produce
one of these valid outcomes:

- unchanged valid candidate;
- matrix-covered recovered candidate;
- unrecoverable attempt followed by a valid bounded regeneration;
- failed required branch followed by collapse of a complete sibling bid.

No accepted outcome may contain a fabricated semantic default.

### Broader regression

```powershell
$cognitionV2Tests = Get-ChildItem -LiteralPath 'tests' -Filter 'test_cognition_core_v2*.py' | ForEach-Object { $_.FullName }
venv\Scripts\python.exe -m pytest $cognitionV2Tests -m "not live_llm and not live_db" -q
venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py -m "not live_llm and not live_db" -q
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py src\kazusa_ai_chatbot\cognition_core_v2\facade.py tests\test_cognition_core_v2_required_selection_live_llm.py
git diff --check
rg -n "selection_kind" src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py tests\test_cognition_core_v2_required_selection_live_llm.py
rg -n "_raise_for_failed_required_branches" src\kazusa_ai_chatbot\cognition_core_v2 tests
```

The two final `rg` commands must return no matches in the active producer and
test surface.

### Human-readable LLM review

The review artifact must compare:

- the original production failure class;
- the newly reproduced raw defect;
- pre-fix validator disposition;
- exact normalization performed;
- post-normalization strict-validation result;
- downstream branch/collapse result;
- semantic preservation judgment;
- adjacent-case results;
- model and endpoint identity;
- observed attempts and latency.

## Independent Code Review

The independent reviewer receives:

- this full plan;
- the final diff;
- the production RCA artifact path;
- the pre-fix/post-fix live review artifact;
- deterministic and live test commands/results.

The review must specifically check:

- semantic ownership is preserved;
- recovery cannot invent or default model decisions;
- recovery-matrix implementation exactly matches this plan;
- strict validation always follows normalization;
- unknown target handles and missing required citations remain fatal;
- complete sibling bids are validated before required-branch recovery;
- total-invalid behavior remains fail-closed;
- no compatibility schema or second JSON parser was added;
- no new LLM call or prompt-budget regression was introduced;
- CJK prompt source parses correctly;
- tests prove both candidate-level and branch-level partial recovery.

Any valid finding is remediated and affected verification is rerun before
signoff.

## Risks

- Real-model stochasticity may not reproduce a recoverable defect. The bounded
  three-case hard gate prevents an unproven production change.
- Dropping an invalid evidence row could hide missing provenance. Mandatory
  operation and progress citations are rechecked after normalization.
- Dropping an unknown extra field could hide conflicting content. Only the
  seven canonical fields are authoritative; the prompt explicitly states that
  extras are non-authoritative.
- Continuing after required ordinary failure could speak from an unrelated
  bid. Continuation requires an already complete branch bid, and the existing
  workspace collapse retains final semantic selection ownership.
- Prompt edits can introduce CJK quoting or syntax errors. Immediate compile
  checks and the CJK skill are mandatory.

## Execution Evidence

Populate during execution:

- Pre-fix git state: clean except for the newly drafted plan and registry row.
- Reproduction commands: the three commands listed under
  `Pre-fix live reproduction gate`, run separately in their listed order.
- Reproduction artifact paths:
  `cognition_core_v2_required_selection_live_llm__third_party_reply_ordinary_contract_pressure.json`,
  `cognition_core_v2_required_selection_live_llm__third_party_reply_autonomy_contract_pressure.json`,
  `cognition_core_v2_required_selection_live_llm__parallel_third_party_reply_autonomy.json`,
  and
  `cognition_core_v2_required_selection_live_llm__parallel_third_party_reply_ordinary.json`
  under `test_artifacts/llm_traces/`.
- Usable failure and matrix row: none. All four real model invocations parsed
  and passed the current strict validator on their first attempt.
- Reproduction gate decision: failed on 2026-07-30. Execution stopped before
  production edits and the plan returned to draft amendment.
- Production files changed: none.
- Deterministic test results: live test file compiled successfully; no
  implementation test batch was authorized after the failed gate.
- Post-fix live commands and artifacts: not applicable; no fix was made.
- Model and endpoint: route `COGNITION_LLM_GOAL_ORDINARY_RESPONSE`, model
  `gemma-4-31b-fable-5-agent-distill`; endpoint credentials were not read.
- Full regression result:
- Static-check result:
- Independent reviewer: not reached because implementation was not authorized.
- Review findings and remediation: not applicable.
- Final git diff summary: live diagnostic capture and three reproduction
  probes, this execution record, registry status, and the agent-authored review
  artifact only.

## Acceptance Criteria

- A usable real-LLM failure was reproduced before any production edit.
- If no usable failure was reproduced, execution stopped with production code
  untouched.
- The reproduced defect is explicitly mapped to a fixed recovery-matrix row.
- The required-selection producer uses the canonical seven-field contract.
- `selection_kind` is absent from the active producer contract.
- Recoverable candidates preserve model-authored semantic fields and pass
  strict validation before becoming `ActionBidV2`.
- Invalid semantic fields, invalid target handles, missing required evidence,
  and zero valid consequences remain unrecoverable.
- A complete sibling model bid can continue to workspace collapse after a
  required branch failure.
- Zero complete bids retain the current typed safe-retry/fail-closed behavior.
- Healthy-path and worst-case LLM call caps do not increase.
- The reproduced live case and adjacent live cases pass one at a time with
  inspected artifacts.
- Focused and full non-live regressions pass.
- The independent review has no unresolved findings.
- Documentation and execution evidence describe the final contract accurately.
