# cognition core v2 goal schema and semantic repair compaction bugfix plan

## Summary

- Goal: remove the observed non-ordinary goal-schema contamination and prevent
  the observed semantic-appraisal repair envelope from exceeding its existing
  cap.
- Status: completed.
- Plan class: bounded Cognition Core V2 prompt and repair-contract bugfix.
- Scope boundary: existing prompt and repair-message owners in
  `goal_cognition.py` and `semantic_appraisal.py`, their focused tests,
  production-shaped live replay coverage, and the owning Cognition README.
- Change direction: replace the shared non-ordinary goal prompt contract with
  branch-specific prompt material and compact exact-field repair feedback;
  compact only the validator-owned `; permitted paths:` semantic error suffix
  while preserving the failed rule, offending path, protected full error, and
  existing cap.
- Acceptance state: the user directed execution after the independent
  `gpt-5.6-sol` high-reasoning review confirmed the two workstreams address the
  observed root causes. Execution, deterministic and live verification,
  parent review, and independent code review are complete within the amended
  scope below.

## Evidence Basis

The plan is based on the following protected evidence and independent review:

- Delivery tracking ID `0a04c1db64e24dd7870cd3d865179f37` resolved to
  `llmtrace_0ef8aa8da3784e0c8a8b65b6b16defdd`, Cognition invocation
  `324fc8b199144308a514ed4732c5aa9a`.
- Delivery tracking ID `a1a573b590a3494786c4edebdee55342` resolved to
  `llmtrace_93482f08e4a74aa5af90adc6e6f5918a`, Cognition invocation
  `8cb6b0f2ed994832ba05706a89c968e8`.
- Both traces exhausted `self_improvement` after three candidates containing
  the nine generic goal fields plus `relational_willingness`.
- `self_improvement` is optional; only `ordinary_response` is required. The
  invalid self-improvement candidates did not reach action planning.
- The `a1a573...` trace also contains an independent semantic-appraisal
  repair-envelope overflow after the model proposed an unowned
  `knowledge_gaps.k7.uncertainty` path.
- The independent review is preserved at
  `test_artifacts/diagnostics/llm_trace_cognition_independent_subagent_review_20260808.md`.
- The parent-authored RCA is preserved at
  `test_artifacts/diagnostics/llm_trace_cognition_rca_20260808.md`.
- The post-draft protected retrieval found 87 Cognition trace runs, 43 traces
  with failed goal attempts, 70 failed goal-model attempts, five terminal
  failure episodes, and six terminal failed branches. The complete evidence is
  preserved in:
  `test_artifacts/diagnostics/cognition_goal_bid_postdraft_trace_runs.json`,
  `test_artifacts/diagnostics/cognition_goal_bid_postdraft_failed_attempts_all.json`,
  and
  `test_artifacts/diagnostics/cognition_goal_bid_postdraft_failure_capsules_all_goal.json`.
- All six terminal branches ended in `goal_bid_structure_exhausted` and belong
  to the same non-ordinary exact-field contamination family. Two of those
  terminal traces also contain the semantic context-limit recurrence:
  `llmtrace_caad1a9370cf4d859e8ea6233f1e473d` and
  `llmtrace_df6eb45b1bfc405fa0e781baa7ce8d76`.
- The other 45 post-draft failures were ordinary relational/evidence contract
  failures that recovered through the intended bounded contract boundary and
  did not create terminal goal exhaustion. They remain regression controls,
  not additional production root causes. The parent-authored categorization is
  preserved at
  `test_artifacts/diagnostics/cognition_goal_bid_postdraft_failure_review_20260809.md`.

The protected traces do not contain a runtime Git SHA. The archived August
plan records `423f6573` as its Cognition V2 baseline. The final prompt,
validator, and semantic-appraisal source changes cannot be attributed to the
later overlay solely from trace timing; the plan therefore fixes the observed
behavior at its current owning boundaries without changing attribution claims.

## Independent Review Outcome

The parent obtained a read-only review from `gpt-5.6-sol` at high reasoning
effort before execution. The reviewer confirmed that Workstream A addresses
the root cause: the shared goal prompt exposes the ordinary-only
`relational_willingness` schema while non-ordinary validation correctly rejects
it. The reviewer also confirmed that Workstream B addresses the root cause:
the validator-owned permitted-path enumeration duplicates the existing
`allowed_values` authority and pushes the repair envelope over its cap.

The review required these amendments, now binding on execution:

- record the complete post-draft failure population and distinguish terminal
  schema contamination from recoverable ordinary contract failures;
- compact only the validator-owned `; permitted paths:` suffix, preserve the
  failed rule and exact offending path, preserve every other error form, and
  capture the complete `str(exc)` before projection;
- derive exact-field key facts from the parsed mapping and, for the captured
  contamination shape, require `missing_top_level_fields == []` and
  `unexpected_top_level_fields == ["relational_willingness"]`;
- remove both the complete `invalid_draft` value and `invalid_draft` vocabulary
  from model-facing exact-field repair instructions while preserving candidate
  context for other repair classes;
- put prompt vocabulary, key sets, payload shape, and character budgets in
  deterministic tests, and give every real-LLM case its own test function;
- treat preserved first candidates followed by live repairs as explicitly
  labeled captured-candidate/live-repair hybrids, with separate reachability
  and repeated-unowned-path purposes;
- verify the `a1a573`, `caad1a`, and `df6eb4` semantic envelopes independently
  under the existing 24,000-character cap.

## Scope And Change Direction

The implementation is two independently verifiable workstreams in one
bugfix plan. They must remain separate in source ownership, tests, evidence,
and failure classification.

### Workstream A: non-ordinary goal-schema isolation

The current generic goal system prompt contains detailed relational-willingness
instructions that apply only to `ordinary_response`, while the same prompt is
used by non-ordinary generic active-goal branches such as `self_improvement`.
The validator correctly rejects `relational_willingness` for those branches,
but the model-facing contract presents the field as salient. The repair
request also repeats relational terminology and echoes the malformed draft.

The target behavior is:

- ordinary generic goals retain the exact relational-willingness contract;
- non-ordinary generic goals receive a separate static prompt with only the
  nine generic goal fields;
- required-selection prompts retain their existing ordinary and active-branch
  contracts;
- each repair request reuses the same branch-specific system contract as its
  initial request;
- an exact-field repair reports the observed, missing, and unexpected top-level
  keys instead of echoing the complete malformed candidate to the model;
- for the captured contamination shape, the parsed key sets are explicit:
  `missing_top_level_fields == []` and
  `unexpected_top_level_fields == ["relational_willingness"]`;
- model-facing exact-field repair instructions contain no complete
  `invalid_draft` value and no `invalid_draft` vocabulary;
- the protected trace continues to retain the complete raw response, parsed
  candidate, and validation error;
- deterministic validation, output fields, retry limits, branch activation,
  optional-branch isolation, and downstream bid filtering remain unchanged.

### Workstream B: semantic-appraisal repair-envelope compaction

The current semantic repair message includes a validation error whose permitted
path suffix duplicates the already-projected `allowed_values` path domains.
For the `a1a573...` failure, this duplicated text pushed the repair request
past the existing 24,000-character repair ceiling before the second model call.

The target behavior is:

- retain the existing 20,000-character initial and 24,000-character repair
  ceilings;
- keep the offending field/path and the single failed rule in model-facing
  repair feedback;
- remove only the validator-owned `; permitted paths:` suffix from the error
  text when the same authority is already present in `allowed_values`;
- preserve every contract-error form that does not contain that exact
  validator-owned suffix unchanged;
- capture the complete original `str(exc)` before constructing model-facing
  feedback;
- preserve the complete original validation error in the protected trace;
- preserve the existing permitted-path authority and strict validator;
- invoke the bounded repair call when the compacted request fits the existing
  ceiling;
- leave a repeated invalid semantic path as a producer-owned contract failure,
  not as an accepted or broadened path.

The plan does not increase the cap, add retries, relax path ownership, or alter
semantic state transitions.

## Confirmed Decisions

| Topic | Decision |
|---|---|
| Runtime cutover | Big-bang replacement of the affected prompt and repair-message behavior; no feature flag, dual prompt path, or compatibility alias. |
| Goal output contract | Keep the existing nine generic fields; `ordinary_response` alone adds `relational_willingness`; required-selection contracts remain unchanged. |
| Goal validation | Keep exact-key validation and all existing type, bound, handle, evidence, and relational validators. |
| Goal exact-field repair | Send deterministic observed/missing/unexpected key facts; do not send the complete malformed draft or the `invalid_draft` token for this error class. |
| Goal retry policy | Keep the invocation-scoped three-call producer budget and all existing branch dispositions. |
| Semantic repair cap | Keep the existing 20,000/24,000 character ceilings. |
| Semantic repair authority | Keep `allowed_values` as the sole model-facing domain authority; compact only the validator-owned `; permitted paths:` suffix and retain the full original error in protected evidence. |
| Failure-family separation | Do not combine the goal-schema fix with branch activation, dialog, delivery, or semantic state-transition changes. |
| Runtime provenance | Capture source SHA, prompt hashes, route/model identity, and repair-payload lengths in the sign-off evidence; do not change the protected trace schema in this plan. |
| Rollback | Revert the affected source commit for either workstream; no runtime switch or persisted-data migration is introduced. |

## Mandatory Skills

Execution must load and apply these skills before touching the governed
surface:

- `development-plan`, including `references/plan_contract.md`,
  `references/execution_gates.md`, and `references/cutover_policy.md`.
- `local-llm-architecture` before editing prompts, model-facing payloads, or
  repair context.
- `debug-llm` before running live LLM replays or authoring human-readable
  quality reviews.
- `py-style` before editing Python.
- `cjk-safety` before editing Python prompt text containing CJK content.
- `test-style-and-execution` before adding, changing, or running tests.
- `python-venv` before running Python or pytest.
- `no-prepost-user-input` before changing any prompt that shapes semantic
  interpretation of user-directed content.
- `llm-trace-debug` when retrieving or replaying protected Cognition evidence.

## Mandatory Rules

- Change only the owned prompt and repair boundaries listed in `Change
  Surface`.
- Preserve RAG evidence ownership, Cognition semantic ownership, dialog
  wording ownership, persistence ownership, and adapter delivery ownership.
- Keep deterministic code responsible for validation, field limits, handle
  authority, path authority, retry accounting, persistence, and failure
  disposition.
- Keep the model responsible for semantic goal judgment and semantic appraisal.
- Route all parsed model output through the existing canonical JSON parser.
  Do not add a parser, semantic post-processor, field-deletion fallback, or
  keyword correction.
- Keep `SystemMessage` contract text static for each branch. Put current-run
  evidence, state, and repair facts in the human message.
- Keep all prompt constants in triple-single-quoted strings and preserve the
  existing Simplified Chinese prompt language.
- Do not introduce plan names, migration terms, implementation topology, or
  test-shaped expected answers into runtime prompts.
- Use `venv\Scripts\python.exe` for Python and pytest commands.
- Do not read `.env`.
- Run live LLM cases one at a time and inspect every raw artifact before the
  next case.
- Keep protected prompt/output evidence out of sanitized event logs, public
  responses, and operational status.
- Preserve unrelated worktree changes.

## Must Do

### 1. Freeze the baseline

Before any implementation:

- record `git status --short` and the current source commit;
- record the current hashes and lengths of the three affected prompt families;
- record the two protected trace paths, trace IDs, invocation IDs, and known
  failure dispositions;
- record the current deterministic test expectations for prompt text, goal
  repair payloads, semantic repair caps, and failure-matrix controls;
- record the absence of runtime source SHA in the supplied traces;
- create a raw sign-off manifest listing every deterministic and live case,
  its expected disposition, and its evidence path.

The baseline is evidence only. It must not be treated as authorization to
change production code.

### 2. Isolate non-ordinary generic goal prompts

Within `goal_cognition.py`:

- retain the existing ordinary generic prompt as the ordinary contract;
- add one canonical non-ordinary generic prompt for active branches without
  typed required selection;
- ensure the non-ordinary system prompt contains no relational-willingness
  field, schema, or ordinary-only decision rule;
- select the ordinary prompt only when the branch requires ordinary relational
  willingness; select the non-ordinary prompt for generic active branches;
- retain the current required-selection prompt selection, including its
  ordinary relational contract and active-branch no-relational contract;
- split repair instruction material so non-ordinary generic repairs contain no
  relational-willingness instruction, while ordinary repairs retain the
  existing contract;
- keep the repair system prompt identical to the initial branch-specific
  system prompt;
- extend exact-field repair feedback with sorted observed, missing, and
  unexpected top-level key sets;
- omit the complete `invalid_draft` value from model-facing feedback when the
  failure is `goal bid draft fields are not exact`;
- omit the `invalid_draft` token from the exact-field repair instruction and
  preserve candidate context for all other repair classes;
- retain complete raw response and parsed output in protected trace evidence;
- leave `validate_goal_bid_draft`, `require_relational_willingness`, branch
  activation, model routes, and the attempt ledger unchanged.

The resulting self-improvement repair request must not make an
ordinary-only relational schema salient through its stable instructions or its
exact-field repair facts.

### 3. Compact semantic repair errors

Within `semantic_appraisal.py`:

- keep `_appraisal_repair_messages` as the repair-message owner;
- project the model-facing `contract_error` into a compact form that retains
  the failed rule and offending field/path;
- remove the redundant permitted-path or allowed-domain suffix when that
  domain is already present in `allowed_values`;
- preserve the existing `allowed_values` structure and permitted path
  projection as the only model-facing authority;
- preserve the complete original `str(exc)` in the protected trace
  `validation_error` field and failure capsule;
- keep invalid-candidate truncation and aggregate-cap accounting deterministic;
- ensure the observed `a1a573...` repair shape fits below the existing
  24,000-character ceiling and reaches a second model invocation;
- keep an unowned path rejected when the model repeats it, with no path
  broadening or deterministic semantic rewrite.

Do not change `contracts.py`, `transition_guards.py`, state reducers, or the
semantic question dependency graph for this workstream.

### 4. Add deterministic regression coverage

Add or update focused tests to prove:

- ordinary generic prompts retain `relational_willingness` and its exact
  repair contract;
- non-ordinary generic prompts and their repair instructions do not contain
  the ordinary-only relational schema;
- exact-field repair feedback reports observed/missing/unexpected keys and
  does not echo the malformed candidate;
- a stubbed non-ordinary model that first returns an extra relational field
  receives branch-correct repair feedback and can return a valid nine-field
  candidate on the next call;
- strict validation still rejects an extra field when it reaches the
  validator;
- optional self-improvement exhaustion remains isolated and does not place an
  invalid bid into action planning;
- ordinary relational-willingness recovery remains unchanged;
- the semantic repair error retains the offending path while omitting the
  validator-owned `; permitted paths:` suffix only;
- the `a1a573`, `caad1a`, and `df6eb4` near-cap semantic repair payloads each fit
  under the existing 24,000-character limit and invoke repair;
- a repeated unowned semantic path remains a contract failure rather than an
  accepted delta;
- failure-matrix, ledger, prompt-budget, dependency, contract, and trace
  capture controls remain valid.

### 5. Run production-shaped live verification

Run each case one at a time with the project virtual environment and inspect
the raw protected output before proceeding:

- each captured `self_improvement` generic goal input from the supplied traces
  in its own test function;
- one preserved post-draft `autonomy_boundary` contamination input in its own
  test function;
- ordinary relational-willingness replay from the existing captured-goal live
  coverage;
- `q:goal_threat_outcome` unowned-path replay from the existing semantic
  failure-mode coverage;
- separate near-cap semantic repair replays corresponding to `a1a573...`,
  `caad1a...`, and `df6eb4...`.

Live acceptance expectations are fixed:

- the non-ordinary goal case must not reproduce the named
  `relational_willingness` exact-field failure;
- the ordinary case must retain a valid relational-willingness result;
- the semantic near-cap case must reach a repair model call without
  `CognitionContextLimitError`;
- if the semantic model repeats the unowned path after repair, the result is
  recorded as a bounded semantic contract failure, not converted into a pass;
- no invalid goal bid may enter action planning or dialog.

### 6. Author review and sign off

Author a human-readable `debug-llm` review from the raw outputs. It must
separate observed input/output, deterministic validation, quality judgment,
failure disposition, and residual risk. The review must link the raw protected
exports and record source SHA, prompt hashes, route/model identity, attempt
coordinates, repair-payload lengths, and live case outcomes.

An independent code review must compare the final diff with this plan. The
review must verify that validators, branch activation, retry budgets, model
routes, protected trace schema, action planning, dialog, persistence, and
delivery were not changed.

## Execution Sign-off

- Focused deterministic gate: 84 passed, 19 deselected.
- Captured deterministic near-cap gate: separate a1a573, caad1a, and df6eb4
  cases each reached exactly two stubbed calls with repair packets under the
  existing 24,000-character cap.
- Live gate: three non-ordinary goal hybrids, the copied ordinary captured
  relational replay, the ordinary search control, the unowned semantic-path
  replay, and all three near-cap semantic hybrids passed their stated bounded
  dispositions. Raw evidence is reviewed in
  `test_artifacts/diagnostics/cognition_core_v2_goal_schema_semantic_repair_review_20260808.md`.
- Broader deterministic family: 527 passed, 2 skipped, 237 deselected, with
  one pre-existing unrelated `PREFERENCE_SYSTEM_PROMPT` attention-cap
  assertion recorded as a waiver in the review artifact.
- Compilation and `git diff --check` passed.
- Independent `kazusa_plan_reviewer` verdict: PASS; no blocking findings.
- Deferred validators, branch activation, retry budgets, routes, trace schema,
  action planning, dialog, persistence, permissions, delivery, and semantic
  transition owners remain unchanged.

## Deferred

- Do not relax exact-key validation or any semantic path/handle validator.
- Do not delete unknown model fields into an accepted bid.
- Do not change `branch_activation.py`, `facade.py`, `model_attempt_policy.py`,
  `contracts.py`, or `transition_guards.py`.
- Do not change the three-call goal budget, appraisal attempt count, or graph
  retry policy.
- Do not increase either semantic appraisal cap.
- Do not add a provider structured-output dependency, new LLM stage, verifier,
  parser, compatibility alias, fallback mapper, or semantic post-processor.
- Do not change RAG retrieval, evidence selection, state transitions,
  workspace collapse, action planning, authorization, dialog wording,
  persistence, scheduler, adapters, database behavior, or delivery.
- Do not rewrite historical protected traces.
- Do not add runtime source-SHA fields to the trace schema in this plan;
  capture provenance in the sign-off manifest and execution environment.
- Do not classify a remaining semantic contract failure as fixed merely because
  the repair context now fits.
- Do not perform unrelated cleanup or formatting.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Non-ordinary generic goal prompt | bigbang | Replace the shared prompt use with the branch-specific contract; retain no legacy non-ordinary path. |
| Goal exact-field repair feedback | bigbang | Replace malformed-draft echoing with key-set feedback for the exact-field error; retain no parallel feedback shape. |
| Semantic repair error text | bigbang | Compact duplicated domain text in the existing repair payload; retain the same authority and cap. |
| Output schemas and validators | compatible | Preserve the existing output and validator contracts exactly; no migration is required. |
| Protected trace storage | compatible | Preserve existing trace fields and protected raw evidence; no schema migration or public exposure is added. |
| Tests | bigbang | Replace assertions for the old prompt/repair presentation and add the new contract controls. |

Rollback is a source revert of the affected workstream. No feature flag, dual
prompt path, persisted-data migration, or runtime compatibility layer is
introduced.

## Target State

```text
generic goal branch
  ordinary_response
    -> ordinary system prompt
    -> relational-willingness repair contract
    -> exact nine generic fields plus relational_willingness

  active non-ordinary branch
    -> non-ordinary system prompt
    -> non-ordinary repair contract
    -> exact nine generic fields only
    -> key-set repair facts for exact-field errors

semantic appraisal repair
  invalid model result
    -> protected full raw response and full validation error
    -> compact model-facing failed rule and offending path
    -> existing allowed_values domains
    -> existing 24,000-character repair ceiling
    -> bounded repair call or typed contract disposition
```

No invalid candidate is accepted, rewritten into a bid, sent to action
planning, or sent to dialog. Valid sibling and ordinary behavior remains
available through the existing optional-branch disposition.

## Contracts And Data Shapes

### Goal output contracts

The output schemas remain unchanged:

- generic non-ordinary bid: `intention`, `desired_outcome`,
  `concrete_detail`, `reason`, `private_monologue`, `target_role_handles`,
  `evidence_handles`, `expected_consequences`, and `confidence`;
- ordinary generic bid: the same nine fields plus
  `relational_willingness`;
- required-selection bids: their existing selection contract and relational
  field rules remain unchanged.

### Goal exact-field repair feedback

For `goal bid draft fields are not exact`, the model-facing feedback contains:

- `required_top_level_fields`;
- `field_types`;
- `observed_top_level_fields`;
- `missing_top_level_fields`;
- `unexpected_top_level_fields`;
- existing allowed evidence/role domains and current-episode handles;
- the existing validation error without a complete malformed candidate.

The protected trace retains the complete raw candidate and parsed output. The
new key-set fields are diagnostic structure, not semantic output fields.

### Semantic repair feedback

The model-facing repair payload retains the existing top-level repair shape:
`repair_instruction`, `contract_error`, and `allowed_values`. Only
`contract_error` presentation changes: it keeps the failed rule and offending
field/path and removes the validator-owned `; permitted paths:` suffix when it
duplicates `allowed_values`. Other contract-error forms remain unchanged. The
protected trace retains the unprojected validation error.

## Change Surface

### Delete

- No production module or validator is deleted.
- The shared non-ordinary use of the ordinary-only goal prompt is removed as a
  behavior, not retained as a fallback.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
  - add the canonical non-ordinary generic prompt;
  - select branch-correct system and repair instructions;
  - add exact-field key-set repair feedback;
  - preserve all output validation and attempt behavior.
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`
  - compact redundant model-facing contract-error text;
  - preserve protected full error capture and existing cap accounting.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
  - document branch-specific generic goal schemas and compact semantic repair
    feedback without changing the public Stage 2 contract.
- `tests/test_cognition_core_v2_prompt_contract_guidance.py`
  - assert branch-specific prompt and repair vocabulary and exact-field facts.
- `tests/test_cognition_core_v2_prompt_budget_continuity.py`
  - cover the near-cap semantic repair shape and preserved cap behavior.
- `tests/test_cognition_core_v2_dependencies.py`
  - align existing repair-payload assertions with key-set feedback while
    retaining ordinary relational assertions.
- `tests/test_cognition_core_v2_trace_failure_modes_live_llm.py`
  - preserve the unowned semantic-path negative control and inspect the
    compact repair attempt.
- `tests/test_cognition_core_v2_semantic_appraisal_exhaustion_live_llm.py`
  - distinguish repair-envelope reachability from any later semantic contract
    exhaustion and cover each of the three captured near-cap envelopes in its
    own inspected case.
- `development_plans/README.md`
  - register this plan under active bugfix plans with status `in_progress`.

### Create

- `tests/test_cognition_core_v2_self_improvement_schema_live_llm.py`
  - separate one-at-a-time production-shaped live replays for the two observed
    generic self-improvement inputs and one post-draft autonomy-boundary input,
    with raw prompt/output capture and downstream bid-set inspection.
- `test_artifacts/diagnostics/cognition_core_v2_goal_schema_semantic_repair_signoff_manifest_20260808.json`
  - raw case inventory, expected dispositions, source/prompt hashes, and
    runtime provenance.
- `test_artifacts/diagnostics/cognition_core_v2_goal_schema_semantic_repair_review_20260808.md`
  - parent-authored human-readable quality review.
- `test_artifacts/diagnostics/cognition_core_v2_goal_schema_semantic_repair_plan_review_20260809.md`
  - parent-preserved `gpt-5.6-sol` root-cause review and amendments.

### Keep

- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py` as strict semantic
  validation authority.
- `src/kazusa_ai_chatbot/cognition_core_v2/branch_activation.py` as branch
  activation and dependency authority.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py` as orchestration and
  optional/required branch disposition authority.
- `src/kazusa_ai_chatbot/cognition_core_v2/model_attempt_policy.py` as the
  invocation-scoped retry ledger authority.
- `src/kazusa_ai_chatbot/cognition_core_v2/transition_guards.py` and state
  reducers as semantic state-transition authorities.
- Existing ordinary relational-willingness, required-selection, failure-matrix,
  trace-capture, action-planning, dialog, and delivery tests as regression
  controls.
- All protected raw evidence and the archived August plan as immutable
  diagnostic/history inputs.

## Agent Autonomy Boundaries

The implementation owner may choose local function names, prompt section order,
test helper organization, and command order within the stated change surface.

The implementation owner must not change:

- output schemas, enum values, evidence/role/path authority, or validators;
- attempt counts, retry ownership, branch activation, or dispositions;
- model routes, environment configuration, protected trace schema, or public
  surfaces;
- the semantic meaning of a goal, appraisal, or relational decision;
- any file in `Deferred` or any unlisted production file.

If the compact error projection cannot preserve the offending path and the
failed rule within the existing repair envelope, execution pauses for a plan
amendment. The owner must not increase the cap or broaden allowed values
silently.

## Verification

### Baseline gate

- clean/current `git status --short` captured;
- current commit, prompt lengths/hashes, and failure evidence recorded;
- no `.env` inspection;
- sign-off manifest frozen before post-change tests.

### Deterministic gate

- Run the focused prompt-guidance, prompt-budget, dependency, contract,
  failure-matrix, attempt-ledger, and trace-capture tests.
- Assert 100% pass for the new branch-specific prompt and repair tests.
- Assert ordinary relational-willingness and required-selection tests retain
  their existing outcomes.
- Assert the semantic near-cap repair construction invokes the repair boundary
  without context-limit failure for all three captured envelopes.

### Live gate

- Run every new non-ordinary replay one case at a time.
- Run ordinary relational recovery one case at a time.
- Run the unowned semantic-path case and each near-cap semantic repair case
  one at a time.
- Inspect raw messages, raw outputs, parsed outputs, validation errors, attempt
  dispositions, branch phase, action-planning bid set, and final dialog.
- Record a human quality judgment separately from deterministic pass/fail.

### Broader regression gate

- Run the affected Cognition Core V2 deterministic family after focused tests.
- Run `git diff --check` and compile every edited Python file through the
  project virtual environment.
- Confirm no changes to branch activation, facade, ledger, validators, routes,
  persistence, dialog, adapter, or event-log raw-data boundaries.

## Acceptance Criteria

The user has accepted execution of this amended scope by explicitly directing
the parent to take the lead and execute after review. The plan can move from
`in_progress` to `completed` only when all of the following are evidenced:

- The non-ordinary generic goal system prompt and repair instructions contain
  no ordinary-only relational-willingness contract.
- The ordinary generic and required-selection contracts remain valid and
  preserve their relational behavior.
- An exact-field non-ordinary failure produces observed/missing/unexpected key
  feedback without echoing the malformed candidate to the model.
- The strict validator still rejects unsupported fields and no deterministic
  code deletes or rewrites model-authored semantics into acceptance.
- The invocation-scoped three-call budget and existing optional-branch
  disposition remain unchanged.
- The `a1a573...` semantic repair shape reaches a bounded repair model call
  without `CognitionContextLimitError` under the existing 24,000-character
  cap.
- The `caad1a...` and `df6eb4...` semantic repair shapes also reach bounded
  repair model calls under the same cap.
- An unowned semantic path remains rejected if the model repeats it; no path
  domain is broadened.
- Invalid goal candidates remain absent from action planning and dialog.
- All focused deterministic gates pass; the affected Cognition V2 regression
  suite passes with no unexplained new failure.
- Each live case has protected raw evidence, a source/prompt/model provenance
  record, and a parent-authored human-readable quality review.
- Independent code review confirms the final diff stays inside `Change
  Surface` and preserves every item in `Deferred`.
- The lifecycle registry and plan status are updated only after the acceptance
  evidence is complete.

## Progress Checklist

- [x] User directs execution after the high-reasoning `gpt-5.6-sol` review and
      accepts the amended scope.
- [x] Baseline commit, worktree state, prompt hashes, evidence paths, and
      sign-off manifest are recorded.
- [x] Workstream A deterministic prompt/schema isolation is implemented and
  verified.
- [x] Workstream B deterministic semantic repair compaction is implemented
  and verified.
- [x] Workstream A live replay is run and inspected.
- [x] Workstream B live replay is run and inspected.
- [x] Broader Cognition V2 regression verification is complete.
- [x] Parent-authored review and independent code review are complete.
- [x] Registry and lifecycle status are updated at closeout.
