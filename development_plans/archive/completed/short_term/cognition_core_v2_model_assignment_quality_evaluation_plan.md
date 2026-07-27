# cognition core v2 model assignment quality evaluation plan

## Summary

- Goal: determine the quality-safe assignment of each Cognition Core V2 model caller to the configured dense baseline or MoE candidate endpoint.
- Plan class: large.
- Status: completed.
- Mandatory skills: development-plan, local-llm-architecture, debug-llm,
  character-test, database-data-pull, test-style-and-execution, and py-style.
- Overall cutover strategy: compatible, test-only evaluation with unchanged production routing and database state.
- Highest-risk areas: model-identity confounds, reviewer bias, caller interactions, database mutation, and tracked character-data leakage.
- Acceptance criteria: preserve the completed 384-sample corpus, its technical
  evidence, the rejected semantic aggregate, and the user's corrected
  stage-level criteria as one closed historical record.

## Closure Disposition

- The user explicitly directed closure on 2026-07-27 despite the evaluation
  flaws and accepted the corpus as limited historical evidence.
- The 384 terminal runs, 3,308 captured configured-factor calls, contract
  results, repair counts, and local raw artifacts remain valid factual
  evidence.
- The original `Q00` all-one-route recommendation is not accepted as a
  trustworthy production assignment. Its aggregate semantic judgment was
  rejected because the test input shaping and four-factor scoring did not
  support fair stage-level allocation decisions.
- Under the user-defined wrong-target robustness criterion, the system passed
  `0/192` runs: no configuration rejected the incorrect addressed-character
  input at the system boundary.
- Under the user-defined allocation-sensitive goal criterion, the current
  `COGNITION_LLM` route passed `24/24` observations and the current
  `BOUNDARY_CORE_LLM` route passed `10/24`.
- The collapse binding had zero observed calls. This plan therefore supplies
  no direct quality evidence for collapse routing.
- Production stage routing is owned by
  [cognition_core_v2_stage_llm_endpoint_routing_plan.md](../../../active/short_term/cognition_core_v2_stage_llm_endpoint_routing_plan.md).
  That plan may use the valid stage-level observations and the user's explicit
  ownership requirements; it must not present this plan's rejected aggregate
  as ground truth.

## Context

`CognitionCoreServicesV2` exposes four model bindings inside the exact
`run_cognition(...)` intake-to-output boundary:

1. `appraisal_config`
2. `goal_cognition_config`
3. `collapse_config`
4. `action_selection_config`

The first three currently share `COGNITION_LLM`. The fourth uses
`BOUNDARY_CORE_LLM`. Surface planning, dialog generation, memory lifecycle,
internal-monologue residue recording, resolver recurrence, persistence, and
adapter delivery sit outside this evaluation boundary.

Sanitized runtime diagnostics identify the configured cognition route as the
31B dense baseline and the boundary route as the 26B MoE candidate. Both are
simultaneously available, and existing service injection permits test-only
per-binding substitution without production routing changes.

A read-only discovery pass confirmed:

- the configured active-character singleton contains the required static
  profile fields and persisted cognition state;
- the selected real conversation rows exist in the configured database;
- every selected source user has a persisted cognition state;
- eight non-duplicate, target-addressed scenarios are frozen by row identity
  in the ignored local manifest
  `test_artifacts/cognition_model_assignment/source_case_selection.json`.

The local manifest and all character/model evidence remain under ignored
`test_artifacts/`; tracked artifacts use active-character-neutral schemas.

This plan measures output quality only. Caller counts are descriptive load
evidence; concurrency, latency, saturation, and executor tuning remain a
post-quality evaluation.

## Mandatory Skills

- `development-plan`: govern lifecycle, evidence, review, and closeout.
- `local-llm-architecture`: preserve Core V2 ownership without prompt or contract drift.
- `debug-llm`: capture real inputs and outputs, separate technical validation
  from semantic judgment, and require agent-authored human reviews.
- `character-test`: evaluate scenario response and active-character
  consistency from real local state while keeping persistent data unchanged.
- `database-data-pull`: use configured read-only database paths and keep
  sensitive exports local.
- `test-style-and-execution`: execute one real-LLM sample at a time and inspect
  it before continuing.
- `py-style`: govern every Python harness and deterministic-test change.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing.
- After signing off any major progress-checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status change, merge, or sign-off, the
  parent must run the Independent Code Review gate and record its result in
  Execution Evidence.
- Parent-led execution uses the current harness's native subagent capability
  for independent review. Native review unavailability blocks final sign-off
  unless the user explicitly authorizes fallback review.
- Database interaction is read-only. Snapshot preparation uses existing public
  read helpers. Matrix execution reads frozen local snapshots and makes no
  database connection.
- Database bootstrap, profile seeding, row creation, updates, replacement,
  deletion, and collection cleanup are fatal scope violations.
- The configured active profile and persisted character/user cognition states
  are copied exactly at snapshot time. Every matrix cell receives a deep copy
  of the same validated `CognitionCoreInputV2` for that case.
- Source messages, typed target data, timestamps, and bounded prior context are
  preserved. Missing ephemeral upstream artifacts remain absent rather than
  receiving invented RAG, resolver, decontextualizer, or progress content.
- The matrix boundary is exactly `run_cognition(input_payload, services)`.
  Surface, dialog, persistence, resolver recurrence, and adapter paths remain
  outside the run.
- The only experimental variable per service binding is its selected model
  profile; endpoint, identifier, and credential switch atomically as one route.
  Prompt text, payload, role contract, temperature, sampling values,
  completion budget, thinking setting, parser, validation, and attempt caps
  remain the binding's current production values.
- The dense 31B output is the baseline reference. Reviewers also judge every
  output independently against the exact input and model-visible character
  constraints; baseline output is not treated as semantic ground truth.
- Exactly eight frozen real scenarios, sixteen full-factorial assignments, and
  three repetitions are required.
- Real-LLM samples execute sequentially. One `run-next` invocation executes one
  matrix sample and then exits.
- The parent inspects and records a disposition for each raw sample before the
  next sample runs.
- Human semantic judgment belongs to the parent agent. Scripts may validate
  shape, blind labels, calculate arithmetic aggregates, and render evidence;
  scripts must not assign semantic scores or author quality conclusions.
- A transport failure may receive one exact technical retry after endpoint
  availability is rechecked. Both attempts remain in evidence. Model-contract
  or semantic failures remain observed results and receive no replacement run.
- Any prompt, parser, production-code, generation-setting, case-input, or
  snapshot change invalidates the matrix and requires a fresh full run.
- Tracked files remain character-neutral. Character profile content, source
  dialog, model prompts, raw responses, blinded reviews, and recommendation
  evidence stay in ignored local artifacts.
- Raw API keys remain excluded from logs, artifacts, diagnostics, and reports.
- Existing unrelated worktree changes remain preserved and outside this plan.

## Must Do

- Freeze eight validated Core V2 input snapshots from the existing local
  manifest and configured database using read-only public helpers.
- Build the complete 16-cell assignment matrix and execute three repetitions
  of every case/cell pair, yielding exactly 384 semantic samples.
- Capture every stage call, raw response, parsed result, repair use, validated
  output, and failure disposition.
- Review every sample, compare each candidate with the 31B baseline, and
  calculate caller main and pairwise interaction effects.
- Produce a quality-qualified assignment set, one recommended mapping, and the
  minimal future route split required to express it.
- Preserve all active-character-specific evidence locally.

## Deferred

- Concurrent-turn load testing, endpoint saturation, latency percentiles,
  executor tuning, queue policy, and throughput verification belong to the
  post-quality performance plan.
- Production route constants, environment variables, Control Console route
  entries, runtime service builders, and deployment configuration remain
  unchanged in this plan.
- Surface/dialog, memory/residue, RAG/resolver recurrence, persistence, and
  adapters remain outside the evaluated boundary.
- Prompt tuning, schema changes, parser changes, retry changes, and generation
  parameter tuning require separate evidence and authorization.
- Additional profiles, synthetic scenarios, database fixtures, and data
  migration remain outside this evaluation.
- Fine-grained splitting inside `action_selection_config` remains deferred
  unless the completed four-binding matrix shows that this aggregated binding
  alone blocks a quality-safe high-offload assignment.

## Cutover Policy

Overall strategy: compatible, test-only evaluation.

| Area | Policy | Instruction |
|---|---|---|
| Production Core V2 routing | compatible | Preserve current route bindings throughout the evaluation. |
| Evaluation harness | compatible | Add a standalone test harness that injects model profiles through the existing four service fields. |
| Database | compatible | Read and freeze existing state; keep persistent documents unchanged. |
| Character artifacts | compatible | Store all profile-specific inputs and outputs under ignored local artifacts. |
| Future route split | migration | Create a separate implementation plan only after the quality recommendation is accepted. |

Cutover enforcement:

- The evaluation harness remains unreachable from production service
  entrypoints.
- Matrix completion authorizes a recommendation artifact, not a runtime
  routing mutation.
- The later production plan uses the exact accepted mapping and adds only the
  route boundaries required to express it.

## Target State

The completed evaluation contains:

- one immutable local case snapshot set with eight validated Core V2 inputs;
- one ledger containing 384 terminal sample rows;
- one raw artifact per sample with complete model and validation evidence;
- one human-authored review per sample;
- one blinded-label key retained separately from review artifacts;
- one aggregate matrix report with per-cell and per-caller quality effects;
- one recommendation that maps appraisal, goal cognition, collapse, and
  boundary/action selection independently to the dense or MoE endpoint;
- one minimal future configuration proposal aligned with the selected mapping;
- one explicit handoff to parallelism and throughput validation.

The recommendation is quality-constrained. It identifies the assignment that
moves the largest observed share of Core V2 calls to the MoE endpoint while
passing every quality gate. Observed call share is a deterministic tie-breaker,
not a throughput measurement.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Evaluation boundary | Use only `run_cognition(...)` input through validated output. | This is the user-approved Core V2 intake/exhaust boundary. |
| Assignment granularity | Use the four existing `CognitionCoreServicesV2` fields. | They are the current inspectable caller ownership boundary and need no production edit for evaluation. |
| Matrix shape | Run the complete two-level, four-factor matrix. | All 16 cells are tractable and expose cross-stage interactions that one-caller swaps miss. |
| Scenario source | Use eight frozen rows from current configured data. | This preserves real character-state and interaction pressure without seeding. |
| Repetitions | Run three repetitions per case and cell. | This is the user-selected repetition count and exposes local-model variance. |
| Input stability | Freeze one validated payload per case before the first model call. | Every model assignment sees byte-equivalent semantic input and initial state. |
| Execution order | Use a deterministic balanced rotation and execute one sample per command. | This limits ordering bias while preserving one-at-a-time inspection. |
| Baseline | Use the configured 31B dense model as the reference envelope. | The user selected it as baseline while defining suitability rather than a single correct answer. |
| Semantic judge | Use blinded parent-agent review with a fixed rubric. | A judge LLM would add another model dependency and hide qualitative reasoning. |
| Failure ownership | Keep contract failures separate from reaction-quality scores. | Technical completion does not prove character suitability. |
| Assignment selection | Maximize MoE-observed call share only among quality-qualified cells. | This supports later throughput work without claiming unmeasured parallel performance. |
| Runtime split | Recommend only the minimal explicit route split needed by the winning cell. | Existing single-route behavior remains adequate when all cognition callers choose one endpoint. |

## Contracts And Data Shapes

### Factor Contract

Factor order is fixed:

```text
A = appraisal_config
G = goal_cognition_config
C = collapse_config
B = action_selection_config
```

Each factor has exactly two levels:

```text
D = configured 31B dense baseline
M = configured 26B MoE candidate
```

### Full Assignment Matrix

| Cell | A | G | C | B |
|---|---|---|---|---|
| `Q00` | D | D | D | D |
| `Q01` | D | D | D | M |
| `Q02` | D | D | M | D |
| `Q03` | D | D | M | M |
| `Q04` | D | M | D | D |
| `Q05` | D | M | D | M |
| `Q06` | D | M | M | D |
| `Q07` | D | M | M | M |
| `Q08` | M | D | D | D |
| `Q09` | M | D | D | M |
| `Q10` | M | D | M | D |
| `Q11` | M | D | M | M |
| `Q12` | M | M | D | D |
| `Q13` | M | M | D | M |
| `Q14` | M | M | M | D |
| `Q15` | M | M | M | M |

`Q00` is the dense baseline. `Q15` is the complete MoE candidate.

### Local Source Manifest

The ignored local source manifest uses schema
`cognition_model_assignment_source_cases.v1`, declares `case_count=8`, and
stores `case_id`, platform/channel/message row identity, and one neutral
scenario-dimension label for each case.

The snapshot command creates a local privacy-token list and requires exactly eight unique source rows. Each row must:

- exist as a non-empty persisted user message;
- resolve to an existing user profile;
- carry an existing persisted `cognition_state`;
- have a timestamp and typed channel/user identity;
- have a bounded preceding context window available from persisted rows.

### Frozen Case Snapshot

Each ignored `cognition_model_assignment_case.v1` snapshot stores the case and
source identity; SHA-256 digests for source, profile, character state, user
state, and validated input; the complete `CognitionCoreInputV2`; and a review
projection containing current input, model-visible character constraints,
initial affect/relationship, and bounded prior context.

Snapshot construction uses:

- `get_character_profile()`;
- `get_user_profile(...)`;
- `get_conversation_by_platform_message_id(...)`;
- `get_conversation_history(...)`;
- `build_user_message_episode(...)`;
- `project_conversation_history_for_llm(...)`;
- `build_cognition_input_from_global_state(...)`;
- `validate_cognition_core_input(...)`.

Each case is a current-state counterfactual: only snapshot-time persisted fields,
the unchanged source message, and at most eight strictly earlier context rows enter; temporal mismatch remains review evidence.

### Model Profile Substitution

For each factor, the harness starts from that factor's production
`LLMCallConfig`. It replaces only:

- `base_url`;
- `api_key`;
- `model`.

The selected values come from the configured dense or MoE route. The harness
retains the factor's stage name, route-local generation policy, temperature,
top-p, top-k, completion budget, presence penalty, and thinking setting.

### Run Ledger

The ledger contains exactly:

```text
8 cases * 16 cells * 3 repetitions = 384 semantic samples
```

Each row records `sample_id`, case, repetition, cell, input digest, terminal
status, artifact path, inspection status, and technical retry count. Status is
one of `pending`, `running`, `completed`, `transport_failed`, or
`contract_failed`; inspection is `pending`, `accepted`, or `finding_recorded`.

For block index `i` from 0 through 23, cell order is the matrix order rotated
by `(5 * i) mod 16`; odd-numbered blocks reverse the rotated order. This gives
each case/repetition block all 16 assignments and distributes early/late
positions without runtime randomness.

### Raw Sample Artifact

Every sample artifact records immutable input/model digests, a blinded label,
ordered stage calls with prompts/raw/parsed values and repair/failure evidence,
validated output and its reaction-bearing projections, structural disposition,
exception details, and descriptive per-factor call counts. The exact
factor-to-model mapping stays in the separate unblinding key.

### Human Review Contract

The reviewer scores five dimensions from 0 through 4:

| Dimension | Question |
|---|---|
| Input responsiveness | Does the reaction engage the salient meaning and pressure in the current input? |
| Character/state consistency | Is the reaction compatible with model-visible profile constraints and current persisted state? |
| Situational suitability | Is the reaction plausible, nuanced, and proportionate for this scenario? |
| Role/evidence grounding | Are speaker, target, ownership, and evidence direction preserved? |
| Cross-stage coherence | Do appraisal, bids, collapse, intention, and action judgment form one coherent reaction? |

Score anchors:

- `4`: strongly suitable and specific;
- `3`: suitable with minor weakness;
- `2`: usable with visible compromise;
- `1`: materially mismatched;
- `0`: critical contradiction, character inversion, role reversal, or
  ungrounded reaction.

Each review also records one baseline-relative verdict:

```text
better | equivalent | minor_loss | material_loss | critical_loss
```

The reviewer writes a concise rationale grounded in exact input/output fields.
A sample without output receives human-assigned zeros and `critical_loss`, with
its structural cause recorded. Labels stay blinded through all 384 rationales.

### Quality Qualification Gate

A matrix cell is quality-qualified only when all conditions hold across its
24 samples:

- every sample reaches a terminal recorded technical disposition;
- unrecovered model-contract failures equal zero;
- critical-loss verdicts equal zero;
- no case has a median three-repetition verdict of `material_loss` or worse;
- aggregate mean score is at least 90% of `Q00` aggregate mean;
- mean input-responsiveness score is within 0.25 points of `Q00`;
- mean character/state-consistency score is within 0.25 points of `Q00`;
- at least 80% of paired verdicts are `better` or `equivalent`;
- any isolated `material_loss` occurs in at most one of 24 samples and does
  not repeat for the same case.

`Q00` is reported even when it contains weak reactions. A candidate may exceed
the baseline; the report preserves that improvement rather than forcing
surface similarity.

### Recommendation Rule

The aggregate step:

1. filters to quality-qualified cells;
2. calculates each caller's main quality effect and pairwise interaction
   effects;
3. calculates baseline-observed call share assigned to MoE for each qualified
   cell;
4. selects the qualified cell with the largest MoE call share;
5. breaks a tie by higher aggregate quality, then fewer production route
   groups, then lower matrix cell id;
6. maps the selected cell to the four caller bindings;
7. states that measured throughput remains pending the post-quality plan.

When only `Q00` qualifies, the recommendation retains the dense model for all
four callers. When a mixed cell wins, the future route proposal introduces
explicit route ownership only where the selected mapping differs.

## LLM Call And Context Budget

Production before and after this plan:

- response-path call count: unchanged;
- background call count: unchanged;
- production prompts and context limits: unchanged.

Evaluation budget:

- Core V2 executions: exactly 384 semantic samples;
- sample concurrency: one;
- appraisals: up to six configured-factor calls per sample;
- goal cognition: up to fourteen branches, with bounded schema and
  required-selection repair, conservatively capped at 56 configured-factor
  calls per sample;
- collapse: one configured-factor call per sample;
- boundary/action selection: conservatively capped at 50 configured-factor
  calls per sample, including required-selection verification, action
  planning, and authorization attempts;
- conservative configured-factor ceiling: 113 calls per sample and 43,392
  calls for the full matrix;
- JSON syntax-repair calls remain on the fixed configured repair route and are
  recorded separately from the four factors;
- review judge calls: zero.

The harness uses existing stage payload caps. No model-facing payload may
exceed the existing 50,000-character default plan cap or a stricter stage cap.
Snapshot context is bounded before the first run and remains byte-stable.

## Change Surface

### Create

- `tests/cognition_core_v2_model_assignment_matrix.py`
  - Character-neutral CLI with `preflight`, `snapshot`, `initialize-ledger`,
    `run-next`, `status`, `verify-ledger`, `build-review-queue`, and
    `aggregate` commands.
  - Public test-harness entrypoints for matrix enumeration, model-profile
    substitution, one-sample execution, and structural aggregation.
- `tests/test_cognition_core_v2_model_assignment_matrix.py`
  - Deterministic contracts for 16-cell enumeration, 384-row ledger,
    balanced ordering, exact config substitution, input immutability,
    read-only snapshot calls, one-sample execution, blinding, and arithmetic
    aggregation.
- `development_plans/active/short_term/cognition_core_v2_model_assignment_quality_evaluation_plan.md`
  - Executable evaluation contract and evidence ledger.

### Modify

- `development_plans/README.md`
  - Register this draft under active short-term plans.

### Local Ignored Artifacts

- `test_artifacts/cognition_model_assignment/`: local source manifest,
  privacy-token list, snapshots, runs, reviews, ledger, unblinding key, matrix
  report, and final recommendation.

### Keep

- All files under `src/kazusa_ai_chatbot/` remain unchanged.
- `.env`, service overrides, Control Console route configuration, and
  deployment files remain unchanged.
- Database collections and documents remain unchanged.

## Overdesign Guardrail

- Actual problem: the project lacks evidence showing which Core V2 caller
  bindings can use the higher-capacity MoE model without material reaction
  quality loss.
- Minimal change: use the existing injected service fields to run one complete
  two-model factorial matrix over frozen real inputs and author a human-reviewed
  recommendation.
- Ownership boundaries: Core V2 prompts own semantic judgment; deterministic
  harness code owns immutable inputs, assignment, validation, ledger state,
  and arithmetic; the parent reviewer owns quality judgment; later runtime
  configuration owns production route mapping.
- Rejected complexity: runtime dynamic routers, fallback chains, judge models,
  prompt tuning, model escalation, per-character production routing,
  fine-grained prompt-call routers, database fixtures, surface/dialog
  evaluation, and concurrency load generation.
- Evidence threshold: a finer split inside the aggregated boundary binding
  requires completed matrix evidence showing that the whole binding must stay
  dense while one repeatedly observed internal call family is both high-load
  and plausibly quality-safe on MoE.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve every schema, count, command, factor, gate, and artifact
  boundary in this plan.
- The responsible agent uses existing public projection, DB-read, Core V2,
  diagnostics, and LLM-interface functions before introducing a new helper.
- Any new helper must remove non-trivial repetition, isolate a named matrix
  contract, or match an established harness pattern.
- Production modules, prompts, route config, database helpers, and runtime
  behavior remain outside implementation authority.
- The eight source identities, sixteen cells, three repetitions, rubric,
  qualification thresholds, and recommendation rule remain fixed.
- An impossible snapshot, missing source row, unavailable model, or attempted
  persistent write blocks execution and receives an explicit report.
- Scope expansion requires a new or updated approved plan before code changes.

## Implementation Order

1. Establish deterministic matrix contracts.
   - File: `tests/test_cognition_core_v2_model_assignment_matrix.py`.
   - Add tests for factor order, all 16 cells, three repetitions, eight cases,
     384 unique sample ids, and balanced order.
   - Expected before implementation: collection fails because the harness
     module is absent.

2. Establish immutable input and read-only snapshot contracts.
   - Add tests that permit only the named public DB read helpers during
     snapshot construction.
   - Add tests that matrix execution loads local snapshots and has no DB
     dependency.
   - Add tests that every cell receives the same `input_digest` and a deep copy
     of the payload.

3. Implement the character-neutral matrix harness.
   - File: `tests/cognition_core_v2_model_assignment_matrix.py`.
   - Implement exact CLI commands and schemas from this plan.
   - Reuse `build_cognition_core_services()` and dataclass replacement for
     factor model substitution.
   - Wrap the existing LLM invoker only to capture ordered evidence.

4. Verify deterministic harness behavior.
   - Run focused tests and `py_compile`.
   - Record the expected-failure and passing evidence.

5. Run read-only preflight and freeze case snapshots.
   - Verify both configured models are simultaneously listed by their endpoint.
   - Verify the active profile and every selected source/user state.
   - Build and validate exactly eight immutable Core V2 inputs.
   - Close database resources before matrix execution.

6. Initialize the 384-row ledger.
   - Record snapshot and model-profile digests.
   - Generate the exact balanced sample order.
   - Verify uniqueness and counts before any model call.

7. Execute and inspect all real-LLM samples.
   - Run `run-next` once.
   - Inspect its input, calls, output, parse/repair behavior, and failure state.
   - Record `accepted` or `finding_recorded`.
   - Repeat until the ledger has 384 inspected terminal samples.
   - Stop immediately on a harness invariant, data mutation attempt, repeated
     transport failure, or changed snapshot/config digest.

8. Complete blinded semantic review.
   - Present one blinded sample at a time.
   - Score all five dimensions and write the rationale.
   - Complete all 384 reviews before unblinding.
   - Add the baseline-relative verdict after comparing the reviewed output with
     that case/repetition's `Q00` result.

9. Aggregate the matrix and author the recommendation.
   - Run structural/arithmetic aggregation.
   - Inspect main effects, interaction effects, case medians, failure rows, and
     qualification gates.
   - Apply the fixed recommendation rule.
   - Author the local quality report and endpoint assignment recommendation.
   - Include the minimal route split and post-quality throughput handoff.

10. Run regression and static verification.
    - Run the focused deterministic suite and relevant existing benchmark
      collection tests.
    - Verify production source and persistent data remained outside the diff.

11. Run Independent Code Review.
    - Start one native review subagent after all verification passes.
    - Remediate findings within the approved test/document surface.
    - Repeat affected verification and record approval.

12. Complete lifecycle sign-off.
    - Present the recommendation and residual scope limits to the user.
    - Mark the plan completed and archive it only after explicit user
      acceptance.

## Execution Model

- Parent agent owns test-contract establishment, harness implementation,
  database read-only preflight, every one-at-a-time real-LLM execution,
  per-sample inspection, semantic review, aggregation, evidence, lifecycle
  updates, and final sign-off.
- Production-code subagent work is inapplicable because production source
  remains unchanged.
- Independent code-review subagent: exactly one native subagent after planned
  verification; it reviews the plan, test/document diff, local evidence
  summaries, privacy boundary, and recommendation arithmetic without
  implementing fixes.
- The parent remediates in-scope review findings and reruns affected checks.
- Native review capability unavailability blocks final sign-off unless the
  user explicitly authorizes fallback execution.

## Progress Checklist

- [x] Stage 1 - deterministic matrix contract established
  - Covers: steps 1-4 and both planned test harness files.
  - Verify: focused tests and `py_compile` pass.
  - Evidence: record the expected failure, changed files, and passing output.
  - Handoff/sign-off: `Codex/2026-07-27`; reread before Stage 2.

- [x] Stage 2 - read-only local snapshots frozen
  - Covers: implementation step 5.
  - Verify: exactly eight cases validate; profile, source, state, and input
    digests are present; DB resources close successfully.
  - Evidence: record the content-free preflight summary and snapshot paths.
  - Handoff/sign-off: `Codex/2026-07-27`; reread before Stage 3.

- [x] Stage 3 - 384-row execution ledger initialized
  - Covers: implementation step 6.
  - Verify: eight cases, sixteen cells, three repetitions, unique sample ids,
    and balanced order.
  - Evidence: ledger verification output recorded.
  - Handoff/sign-off: `Codex/2026-07-27`; reread before Stage 4.

- [x] Stage 4 - all sequential real-LLM samples executed and inspected
  - Covers: implementation step 7.
  - Verify: 384 terminal samples and 384 inspection dispositions; every
    artifact input digest matches its case snapshot.
  - Evidence: ledger summary, technical failures, retries, and artifact root.
  - Handoff/sign-off: `Codex/2026-07-27`; Stage 5 remains unstarted under the
    evidence-collection-only execution boundary.

- [x] Stage 5 - blinded human quality review complete
  - Covers: implementation step 8.
  - Verify: 384 five-dimension reviews and baseline-relative verdicts; blinding
    key remained separate until independent scores were complete.
  - Evidence: review count and reviewer sign-off recorded.
  - Handoff/sign-off: `Codex/2026-07-27`; reread before Stage 6.

- [x] Stage 6 - matrix recommendation authored
  - Covers: implementation step 9.
  - Verify: every cell has qualification results; caller effects and
    interactions reconcile with source reviews; one mapping is selected by the
    fixed rule.
  - Evidence: local report and recommendation paths recorded.
  - Handoff/sign-off: `Codex/2026-07-27`; reread before Stage 7.

- [x] Stage 7 - regression and privacy gates pass
  - Covers: implementation step 10.
  - Verify: deterministic tests, collection checks, diff checks, local-artifact
    checks, and database write-boundary tests pass.
  - Evidence: command outputs and changed-file inventory recorded.
  - Handoff/sign-off: `Codex/2026-07-27`; reread before Stage 8.

- [ ] Stage 8 - independent code review approved
  - Covers: implementation step 11.
  - Verify: review subagent approves plan alignment, code quality, evidence,
    privacy, and recommendation arithmetic after remediation.
  - Evidence: reviewer identity, findings, fixes, rerun commands, residual
    risks, and approval recorded.
  - Handoff/sign-off: remains unchecked; reviewer rejection and the accepted
    procedural deviation are preserved in Execution Evidence.

- [x] Stage 9 - user accepts limited evidence closure
  - Covers: implementation step 12.
  - Verify: explicit user direction closes the corpus despite its flaws,
    rejects the aggregate recommendation as assignment authority, and records
    the successor routing plan.
  - Evidence: closure disposition, final status, registry update, and archive
    location recorded.
  - Sign-off: `Codex/2026-07-27`.

## Verification

### Deterministic Tests

- `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_model_assignment_matrix.py -q`
  - Expected: all matrix, snapshot, ledger, blinding, immutability, and
    aggregation contracts pass.
- `venv\Scripts\python.exe -m py_compile tests\cognition_core_v2_model_assignment_matrix.py tests\test_cognition_core_v2_model_assignment_matrix.py`
  - Expected: exit code zero.
- `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_benchmark.py -q -m "not live_llm and not live_db"`
  - Expected: retained benchmark utility contracts pass.

### Read-Only Preflight

- `venv\Scripts\python.exe tests\cognition_core_v2_model_assignment_matrix.py preflight --case-manifest test_artifacts\cognition_model_assignment\source_case_selection.json`
  - Expected: two model profiles available, active profile valid, eight source
    rows valid, and eight existing user cognition states present.
- `venv\Scripts\python.exe tests\cognition_core_v2_model_assignment_matrix.py snapshot --case-manifest test_artifacts\cognition_model_assignment\source_case_selection.json`
  - Expected: eight validated snapshots and no database write operation.

### Ledger

- `venv\Scripts\python.exe tests\cognition_core_v2_model_assignment_matrix.py initialize-ledger`
  - Expected: exactly 384 unique pending rows.
- `venv\Scripts\python.exe tests\cognition_core_v2_model_assignment_matrix.py status`
  - Expected before execution: `pending=384`; expected after execution and
    inspection: `terminal=384`, `inspected=384`.
- `venv\Scripts\python.exe tests\cognition_core_v2_model_assignment_matrix.py verify-ledger`
  - Expected: all counts, digests, cell coverage, repetition coverage, and
    artifact references pass.

### Sequential Real LLM

- `venv\Scripts\python.exe tests\cognition_core_v2_model_assignment_matrix.py run-next`
  - Execute exactly once per sample.
  - Expected: one sample becomes terminal and the process exits.
  - The parent inspects and dispositions that artifact before repeating.

### Reviews And Aggregation

- `venv\Scripts\python.exe tests\cognition_core_v2_model_assignment_matrix.py build-review-queue`
  - Expected: one blinded review item per terminal sample and no model identity
    in reviewer-facing files.
- `venv\Scripts\python.exe tests\cognition_core_v2_model_assignment_matrix.py aggregate`
  - Expected: arithmetic succeeds only after 384 human reviews and emits all
    qualification gates without generating semantic judgments.

### Static And Privacy Checks

- `git diff --check`
  - Expected: no whitespace errors.
- `git status --short`
  - Expected planned tracked changes are limited to the plan registry and the
    two test harness files, while any pre-existing unrelated changes remain
    separately identified.
- `git diff --name-only -- src/kazusa_ai_chatbot`
  - Expected: this plan contributes no production-source changes.
- `rg -n -F -f test_artifacts\cognition_model_assignment\tracked_forbidden_tokens.txt development_plans\README.md development_plans\active\short_term\cognition_core_v2_model_assignment_quality_evaluation_plan.md tests\cognition_core_v2_model_assignment_matrix.py tests\test_cognition_core_v2_model_assignment_matrix.py`
  - Expected: zero matches; `rg` exit code 1 is the successful zero-match
    result.
- `git check-ignore test_artifacts/cognition_model_assignment/source_case_selection.json`
  - Expected: the local manifest is ignored.

## Independent Code Review

Run this gate after every verification command passes and before final
recommendation sign-off. The parent creates one independent code-review
subagent through native subagent capability. The reviewer implements no fixes.

Review scope:

- project and mandatory-skill compliance for the harness, tests, plan, and
  registry;
- exact four-factor matrix and 384-sample ledger arithmetic;
- proof that model substitution changes only model endpoint identity;
- proof that matrix execution cannot access or mutate MongoDB;
- one-at-a-time execution and inspection evidence;
- separation of structural aggregation from human semantic judgment;
- character-data locality, secret redaction, and tracked-file neutrality;
- qualification-gate arithmetic and final mapping selection;
- absence of runtime routing, prompt, parser, or production behavior changes;
- completeness of the post-quality throughput handoff.

The parent fixes findings inside the approved test/document surface and reruns
affected verification. A finding requiring production code, contract changes,
new data, or altered quality thresholds blocks sign-off pending a plan update
and user approval.

Record reviewer identity, findings, fixes, rerun commands, residual risks, and
approval status in Execution Evidence.

## Acceptance Criteria

This plan is complete when:

- eight real configured-data cases are frozen through read-only access;
- all sixteen assignments run three times for every case;
- all 384 samples have raw artifacts and human inspection dispositions;
- all 384 samples have completed five-dimension semantic reviews;
- each matrix cell has baseline-relative quality and qualification results;
- caller main effects and material interaction effects are documented;
- one quality-safe assignment maps all four Core V2 callers to explicit model
  endpoints;
- the recommendation includes the smallest future route split needed to
  express that mapping;
- the report clearly reserves throughput, concurrency, and executor tuning for
  the post-quality evaluation;
- production source, service configuration, prompts, database state, and
  tracked character content remain unchanged;
- deterministic verification and independent code review pass;
- the user explicitly accepts the recommendation before lifecycle closeout.

Accepted closure exception:

- The user closed this plan with Stage 8 still rejected and without accepting
  the original recommendation.
- Completion means the experiment and its limitations are historically
  settled. It does not mean every original acceptance criterion passed.
- The successor routing plan owns production implementation and verification.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Baseline output is itself weak | Score every output independently before baseline comparison. | Independent dimension scores and baseline-relative verdicts. |
| Caller interactions hide single-stage effects | Execute all 16 full-factorial cells. | Matrix coverage and interaction-effect report. |
| Database state changes during a long run | Freeze validated local inputs once; execute the matrix without DB access. | Snapshot digests and DB-free run tests. |
| Reviewer learns model identity | Store the unblinding key separately until scores are complete. | Blinding tests and review evidence. |
| Character material enters git | Keep all local inputs, outputs, and reports under ignored artifacts; grep tracked harnesses. | Ignore check, static character-name grep, and code review. |
| Transport instability is confused with model quality | Allow one recorded exact transport retry and keep contract failures as results. | Ledger retry and failure dispositions. |
| Quality-only results are presented as throughput proof | Use call share only as a tie-breaker and require a later load plan. | Recommendation wording and review gate. |
| Matrix execution becomes excessively large | Preserve the fixed 384-sample design and resumable one-sample ledger. | Status and ledger verification. |

## Execution Evidence

- Discovery git status: clean worktree before execution on 2026-07-27.
- Read-only profile/data preflight: the first command exposed two neutral
  provenance fields in the accepted local manifest; the root validator was
  aligned without changing any case identity, and all 7 focused contracts
  passed. The rerun reported `status=ready`, 8 cases, 2 model profiles from
  1 provider, and `database_closed=true` on 2026-07-27.
- Deterministic expected failure: focused pytest collection failed on 2026-07-27 with the planned missing `tests.cognition_core_v2_model_assignment_matrix` module.
- Deterministic passing tests: focused matrix suite passed 7 tests; both
  harness files passed `py_compile`; retained non-live benchmark suite passed
  3 tests with 4 live tests deselected; `git diff --check` passed and the
  production-source diff remained empty on 2026-07-27.
- Snapshot result: 8 frozen case snapshots validated from disk with distinct
  input and snapshot digests, snapshot-set digest
  `09f8c2dd2c98684be581e00376e94636ee25d09abcb716174d3c7b811b26dd2a`,
  and `database_closed=true`. Local paths are
  `test_artifacts/cognition_model_assignment/snapshots/index.json`,
  `snapshots/active_profile.json`, `snapshots/scenario_01.json` through
  `snapshots/scenario_08.json`, and `tracked_forbidden_tokens.txt`. Ignore
  checks passed and the tracked privacy-token scan returned zero matches.
- Ledger initialization: `initialize-ledger`, `status`, and `verify-ledger`
  each reported a valid 384-row ledger with `pending=384`, `running=0`,
  `terminal=0`, `inspected=0`, and 384 reconciled unblinding assignments.
  Content-free balance inspection confirmed 8 cases with 48 rows each,
  16 cells with 24 rows each, 3 repetitions with 128 rows each, 384 unique
  sample ids, 384 unique blind labels, and all 24 blocks containing every
  cell exactly once.
- Real-LLM execution summary: Stage 4 completed on 2026-07-27 with 384
  sequential terminal samples and 384 reconciled technical inspection
  sidecars. Final ledger verification reported `completed=384`,
  `contract_failed=0`, `transport_failed=0`, `technical_retry_count=0` for
  every sample, `pending=0`, `running=0`, and unchanged snapshot-set and route
  profile digests. The local evidence contains 384 run artifacts and 384
  inspection artifacts, all with disposition `accepted`. Across the captured
  runs there were 384 attempts, 3,308 configured-factor calls, 0 JSON-repair
  calls, 0 deterministic parse failures, 0 attempt failures, and 0
  factor-call failures. The configured-factor call totals were 2,530
  appraisal, 384 goal-cognition, 0 collapse, and 394 action-selection calls;
  the observed range was 8 through 17 configured-factor calls per sample.
  Raw artifacts, inspection sidecars, and the factual capture log remain under
  ignored local path `test_artifacts/cognition_model_assignment/`. No semantic
  review, comparison, aggregation, qualification, conclusion, or assignment
  decision was performed.
- Human review summary: Stage 5 completed on 2026-07-27 with 384 unique
  parent-authored, five-dimension judgments completed in exact blinded queue
  order before the assignment key was opened. The independent corpus passed
  count, uniqueness, order, score-bound, rationale, and premature-verdict
  checks. Each sample was then compared with its matching case/repetition
  `Q00` output, yielding 85 `better`, 178 `equivalent`, 56 `minor_loss`,
  62 `material_loss`, and 3 `critical_loss` verdicts; all 24 `Q00`
  self-comparisons were `equivalent`. Deterministic merge validation confirmed
  384 canonical review files with exact score, rationale, and verdict
  preservation. Local evidence paths are
  `test_artifacts/cognition_model_assignment/review_drafts.jsonl`,
  `blinded_independent_reviews.md`, `review_verdicts.jsonl`, `reviews/`, and
  `completed_human_reviews.md`.
- Original fixed-rule qualification output: `Q00` only. All 15
  candidate-containing cells
  failed one or more fixed gates. `Q01`, `Q03`, and `Q07` were the closest
  candidates but each exceeded the global material-loss cap. The user rejected
  this aggregate as a fair basis for stage allocation.
- Original fixed-rule assignment output: `Q00`, with `appraisal_config`,
  `goal_cognition_config`, `collapse_config`, and
  `action_selection_config` all assigned to the configured dense baseline.
  This output is retained only to reconcile the historical aggregate artifact;
  it is not an accepted production recommendation. The collapse caller had
  zero observed calls, so no direct candidate evidence exists for that
  binding.
- User-corrected robustness evidence: the wrong addressed-character cases are
  valid failure-mode inputs. A pass requires system-level rejection before
  cognition proceeds. All 192 such runs continued through speech or evidence
  routing, so every matrix cell failed this criterion.
- User-corrected goal evidence: for the allocation-sensitive quantity case, a
  pass requires identifying that the active character receives two. The
  current `COGNITION_LLM` route passed all 24 observations; the current
  `BOUNDARY_CORE_LLM` route passed 10 of 24.
- Allocation limit: the four-factor aggregate combines distinct internal
  stages and cannot answer how many specific parallel stages can move between
  endpoints. Only the factual call counts, contract behavior, and explicit
  user-scored criteria above carry into the successor plan.
- Local report paths:
  `test_artifacts/cognition_model_assignment/matrix_aggregate.json`,
  `quality_report.md`, and `recommendation.json`. Deterministic reconciliation
  confirmed the selected cell, qualified set, all four binding levels, all 16
  cell means and paired-verdict ratios, caller main effects, and pairwise
  interactions against the aggregate artifact.
- Static/privacy checks: final `py_compile`, 7 focused matrix tests, and 3
  retained non-live benchmark tests passed with 4 live tests deselected. The
  focused suite covered exact endpoint substitution, read-only snapshot
  helpers, DB-free sample execution, input immutability, and aggregation.
  Final ledger verification retained 384 completed and inspected samples with
  unchanged route/snapshot digests; aggregate replay retained 16 cells,
  `Q00` as the sole qualified and selected cell, and 384 reviews.
  `git diff --check` passed with line-ending notices only. Tracked changes are
  limited to the registry, active plan, harness, and focused test; the
  production-source diff is empty. All checked local evidence paths are
  ignored and `git ls-files` reports zero tracked files under the artifact
  root. The tracked privacy-token scan returned zero matches. An artifact scan
  parsed 1,173 JSON files and scanned 1,180 text artifacts with zero raw
  credential-key occurrences, zero secret-pattern matches, and zero JSON parse
  failures.
- Independent review: native reviewer `Feynman`
  (`019fa1ce-7532-7560-80d1-64bdef6fe483`) completed the initial read-only
  review and remediation re-review on 2026-07-27. The initial review
  reconciled the 384-sample arithmetic, reviews, qualification gates, caller
  effects, interactions, final `Q00` mapping, endpoint-only substitution,
  local evidence boundary, empty production-source diff, and deferred
  throughput scope. It reported three findings:
  - Stage 4 used automated deterministic structural inspection and sidecar
    creation between sequential samples, while parent-authored semantic
    inspection occurred later in Stage 5. This differs from the plan's
    required parent raw-sample inspection before each next sample.
  - CLI error reporting could emit untrusted exception text when route
    identity loading failed.
  - The focused DB-free execution test guarded harness-local helpers but did
    not independently prohibit database dependencies inside the Core V2
    runtime package.
  Parent remediation replaced CLI exception text with a fixed
  credential-safe boundary message and added deterministic coverage proving
  secret-bearing exceptions remain absent from CLI output. A package-wide AST
  regression test now rejects direct Core V2 imports from the public database
  package, Motor, or PyMongo. Post-remediation verification passed 9 focused
  tests and `py_compile`. The same reviewer confirmed both code/test findings
  resolved and reported no new findings. Residual technical limits are
  transitive or dynamic database imports outside the static guard, zero
  observed collapse calls, eight-scenario/three-repetition coverage, and
  subjective human judgment. Approval status remains `REJECT`; the user
  accepted this residual and retained the existing corpus without rerun.
- User acceptance: on 2026-07-27 the user explicitly directed closure despite
  the flaws. The corpus and valid factual observations are accepted as a
  completed historical record. The original aggregate recommendation remains
  rejected, and the successor stage-routing plan owns production assignment.
