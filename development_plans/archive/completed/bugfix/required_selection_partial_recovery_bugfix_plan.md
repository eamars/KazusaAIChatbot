# Required-Selection Partial-Recovery Bugfix Plan

Status: completed

Plan class: high_risk_migration

Cutover: bigbang

## Summary

Implement the approved `COGNITION-V2-BID-EXHAUST-ARCH` decision from
`development_plans/archive/completed/short_term/august_change_alignment_audit_and_remediation_plan.md`.
The cutover establishes one monotonic goal-producer attempt ledger across
local regeneration and the service's clean graph retry, removes the unused
`selection_kind` field, keeps unsupported model-authored values fail-closed,
and preserves complete validated sibling bids when one required branch fails.

The user's 2026-08-08 instruction authorizes execution and architecture
decisions by the parent execution owner. Execution, implementation, review,
verification, and evidence are parent-owned without delegation or
execution-time questions.

## Approval And Dependency Resolution

- Approved for execution: 2026-08-08 through the user's instruction to fully
  execute the parent August alignment plan and make the required architecture
  decisions.
- Parent dependency: `COGNITION-V2-BID-EXHAUST-ARCH` is approved below and in
  the parent audit plan before production changes begin.
- Historical predecessor:
  `development_plans/archive/completed/bugfix/required_selection_contract_separation_bugfix_plan.md`.
- The earlier 2026-07-30 execution stopped before production edits after three
  live probes produced valid first-attempt candidates. That evidence remains
  historical and is retained under `Execution Evidence`.
- New protected captures from 2026-08-07 and 2026-08-08 provide exact current
  candidate failures, including unauthorized evidence handles and relational
  willingness pairing errors. They satisfy the evidence gate for the failure
  family without treating stochastic repetition as an architecture gate.

## Mandatory Skills And Repository Rules

Execution applies these skills in full:

- `.agents/skills/development-plan/SKILL.md`
- `.agents/skills/local-llm-architecture/SKILL.md`
- `.agents/skills/debug-llm/SKILL.md`
- `.agents/skills/test-style-and-execution/SKILL.md`
- `.agents/skills/no-prepost-user-input/SKILL.md`
- `.agents/skills/py-style/SKILL.md`
- `.agents/skills/cjk-safety/SKILL.md`

Use `venv\Scripts\python.exe`. Run live LLM tests one case at a time and
inspect each durable artifact before continuing. Preserve unrelated worktree
changes. Keep `.env` unread for this bugfix.

## Evidence And Root Cause

The historical failure associated with
`chat:qq:ch_1f677493d7a52025:1634535291` and
`llmtrace_0cd78d39a55e48f6ae8efb662262eb70` reached required-selection goal
cognition, then made twelve goal calls: two service graph attempts multiplied
three local attempts across both `ordinary_response` and
`autonomy_boundary`. Every call ended in `contract_error`, and the required
ordinary branch raised `goal_bid_structure_exhausted` before workspace
collapse.

The historical metadata export lacks raw candidate values. Later authorized
full failure capsules establish the current failure family:

- `llmtrace_a5997476b97640b4af5e0786244b1676` records a repaired relational
  willingness mismatch;
- `llmtrace_899ea885f64b402cb2df6ef1d4e35783` records an unavailable evidence
  handle and adjacent semantic contract failures;
- `llmtrace_cb1de2895b4a4987be79cc8530ab6f5c` records an unauthorized goal
  evidence handle followed by a relational willingness mismatch, with the
  ordinary branch succeeding on cumulative attempt three;
- the production-shaped third-party required-selection replay reached the
  current dense goal owner and completed after bounded regeneration.

The architecture defects are deterministic:

1. The local three-attempt counter restarts when the service repeats the
   graph, multiplying the producer's declared bound.
2. `selection_kind` is required by the prompt and validator but discarded by
   the mapper, so it is not model-owned product meaning.
3. Goal cognition currently accepts a candidate after deleting unsupported
   evidence or role handles. Current repository rules classify unsupported
   handles as non-recoverable semantic-contract values; regeneration belongs
   to the producing LLM stage.
4. The facade escalates a required branch failure before considering a
   complete validated sibling bid.
5. Protected capsules retain raw attempts but do not expose the shared
   invocation/graph/local/cumulative budget coordinates or final branch
   disposition needed to diagnose multiplied exhaustion.

## Mandatory Architecture Decisions

### Semantic and deterministic ownership

- The goal-cognition LLM owns selection, reasons, private monologue, targets,
  evidence citations, consequences, confidence, and relational willingness.
- `parse_llm_json_output(...)` owns canonical deterministic JSON cleanup.
- The goal validator owns exact fields, types, bounds, handle authority,
  evidence coverage, and relational pairings.
- Goal cognition owns repair feedback and bounded complete regeneration.
- The monotonic ledger owns call counting only; it never changes a candidate.
- The facade owns required/optional branch continuation and escalation.
- Workspace collapse receives complete validated `ActionBidV2` objects only.

### Canonical required-selection contract

The model returns exactly these seven fields, plus
`relational_willingness` for `ordinary_response`:

```text
selection
reason
private_monologue
target_role_handles
evidence_handles
expected_consequences
confidence
```

`selection_kind` is removed from the prompt, repair feedback, validator,
fixtures, and tests in the same cutover. There is no alias or legacy mapper.

### Candidate disposition matrix

No stage-local normalizer rewrites model-authored candidate values. The
canonical parser may repair JSON transport syntax under its existing contract;
the strict validator then applies this matrix:

| Candidate condition | Attempt disposition | Next action |
|---|---|---|
| Exact schema and all values valid | `accepted` | Construct the complete bid. |
| Canonical parser repaired syntax and the strict candidate is valid | `recovered` | Construct the complete bid and retain repair evidence. |
| Missing or unknown field, including legacy `selection_kind` | `regenerate` or `exhausted` | Request a complete replacement while budget remains; otherwise fail the branch. |
| Wrong type, empty/over-bound prose, invalid enum or relational pairing | `regenerate` or `exhausted` | Request a complete replacement while budget remains; otherwise fail the branch. |
| Duplicate, wrong-type, unknown, or unauthorized role/evidence handle | `regenerate` or `exhausted` | Preserve provenance authority; never delete or broaden the cited value into acceptance. |
| Required operation/progress evidence missing | `regenerate` or `exhausted` | Preserve required coverage; never fabricate or copy a citation. |
| Invalid, empty, or over-bound consequence list | `regenerate` or `exhausted` | Preserve model ownership; never delete rows to manufacture validity. |

The previous `_degraded_selection_goal_draft(...)` and
`_degraded_goal_bid_draft(...)` handle-filtering acceptance paths are removed.
`V2_MODEL_OWNER_POLICIES["goal_bid_structure"]` declares an exhausted
disposition of `unrecoverable`.

### Monotonic attempt ledger

One invocation-local ledger is bound before the first service graph attempt
and reused by the one permitted clean graph retry. Direct `run_cognition(...)`
calls create an equivalent one-graph scope. Its producer key is
`(stage="goal_bid_structure", branch_id)`.

An attempt is consumed immediately before `llm.ainvoke(...)`, including a
provider-error call. The reservation records:

```text
cognition_invocation_id
graph_attempt
branch_id
producing_stage
local_attempt
cumulative_producer_attempt
configured_limit
attempt_disposition
```

The per-key configured limit remains three. A service retry can reuse only
unused producer calls. It cannot reset the counter.

| Scenario | Graph 1 calls | Graph 2 calls | Cumulative bound | Terminal result |
|---|---:|---:|---:|---|
| Goal succeeds on first call; later retryable stage fails | 1 | up to 2 remaining | 3 | Later stage succeeds or its owner fails. |
| Goal succeeds on second call; later retryable stage fails | 2 | up to 1 remaining | 3 | Later stage succeeds or its owner fails. |
| Goal consumes all three calls and fails | 3 | 0 | 3 | Non-retryable branch exhaustion. |
| Provider errors consume all three calls | 3 | 0 | 3 | Non-retryable provider exhaustion. |
| Direct facade invocation | up to 3 | not applicable | 3 | Accepted or typed exhaustion. |

The service retains its one clean graph retry for other typed pre-commit
failures. Goal exhaustion is `retryable=False` because its producer budget is
already consumed.

### Branch recovery matrix

| Branch state | Facade disposition | Downstream visibility |
|---|---|---|
| Every required branch succeeded | `accepted` | All eligible complete bids may reach collapse. |
| Required branch failed and at least one complete validated sibling bid exists | `accepted_degraded` / `recovered_by_sibling` | Preserve failure record and warning; only complete sibling bids reach collapse. |
| Required branch failed and no complete validated sibling bid exists | `exhausted` | Raise typed non-retryable `CognitionExecutionError` before collapse. |
| Optional branch failed | Existing optional-isolation behavior | Only complete results continue. |

The warning is
`required_branch_recovered_by_valid_bid:<branch_id>`. Deterministic code does
not synthesize an ordinary-response bid or promote a partial mapping.

### Protected observability

The existing full failure capsule remains the authorized raw-content boundary.
Each captured goal attempt gains the monotonic budget coordinates and attempt
disposition. Each promoted capsule gains a bounded ledger snapshot with final
branch dispositions. Ordinary metadata traces remain raw-free; event logs,
public responses, and operational status receive no raw candidate content.

## Target State

```text
service cognition invocation
  -> bind one invocation ledger
  -> graph attempt 1
       -> goal branch reserves cumulative call
       -> canonical JSON parsing
       -> strict seven-field validation
            -> valid complete bid
            -> producer-owned regeneration while budget remains
            -> typed branch exhaustion at cumulative limit
  -> required branch policy
       -> all required valid: normal collapse
       -> required failed + complete sibling: sibling-only collapse
       -> required failed + zero complete bids: typed terminal failure
  -> optional clean graph retry
       -> same ledger; only unused producer calls are available
```

No invalid or incomplete bid reaches workspace collapse, action planning,
dialog, persistence, or delivery.

## Change Surface

### Production

- `src/kazusa_ai_chatbot/cognition_core_v2/model_attempt_policy.py`
  - add the invocation-local ledger and protected snapshot contract;
  - align the goal exhausted disposition.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
  - reserve cumulative attempts;
  - remove `selection_kind` and handle-filtering degraded acceptance;
  - record typed accepted/recovered/exhausted dispositions.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`
  - own direct-call ledger scope;
  - continue with validated sibling bids under the fixed branch matrix.
- `src/kazusa_ai_chatbot/service.py`
  - bind one ledger across initial and clean-retry graph invocations.
- `src/kazusa_ai_chatbot/llm_tracing/__init__.py` and
  `src/kazusa_ai_chatbot/llm_tracing/failure_capsule.py`
  - carry bounded attempt coordinates and ledger/branch disposition snapshots
    inside protected capsules.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
  - document the canonical contract, cumulative bound, and branch policy.

### Tests

- `tests/test_cognition_core_v2_dependencies.py`
- `tests/test_cognition_core_v2_failures.py`
- `tests/test_cognition_core_v2_integration.py`
- `tests/test_cognition_core_v2_model_retry_continuity.py`
- `tests/test_cognition_core_v2_prompt_budget_continuity.py`
- `tests/test_cognition_core_v2_required_selection_live_llm.py`
- `tests/test_llm_tracing.py`
- `tests/test_service_input_queue.py`

### Evidence and lifecycle

- `test_artifacts/reviews/required_selection_partial_recovery_live_llm_review.md`
- parent August audit artifact and this plan
- `development_plans/README.md`

## Deferred

- Model, endpoint, or `.env` changes.
- Changes to the dense required-selection route.
- More than one service graph retry or more than three cumulative goal calls.
- Public API, persistence schema, queue, adapter, RAG, action-planning, dialog,
  or visible-wording redesign.
- A compatibility schema for `selection_kind`.
- A second JSON parser, verifier LLM, classifier LLM, or new healthy-path call.

## Implementation Order

1. Record the approved architecture and evidence gate in both active plans and
   update their registry statuses.
2. Add deterministic red tests for cumulative call accounting, no-budget graph
   retry, strict unsupported-handle rejection, seven-field cutover, required
   sibling continuation, zero-bid failure, and protected ledger capture.
3. Implement the ledger and tracing data shape.
4. Bind it across service graph attempts and direct facade calls.
5. Implement goal reservation/dispositions, remove `selection_kind`, and
   remove handle-filtering degraded acceptance.
6. Compile CJK-bearing Python immediately.
7. Implement the facade branch matrix.
8. Run focused deterministic verification and remediate scoped failures.
9. Update the Cognition V2 README.
10. Run the production-shaped live required-selection replay and adjacent
    cases one at a time; inspect and author the human-readable review.
11. Run the broader deterministic cognition/service/trace suites and static
    checks.
12. Perform parent-owned diff and architecture review, resolve findings, and
    record closeout evidence.

## Progress Checklist

- [x] Mandatory repository and skill context read.
- [x] Historical and current protected evidence reviewed.
- [x] Parent architecture dependency approved.
- [x] Monotonic budget and branch matrices recorded.
- [x] Parent-only execution model recorded.
- [x] Deterministic red tests added and observed failing for the intended reason.
- [x] Ledger and protected-capture contract implemented.
- [x] Seven-field producer cutover implemented.
- [x] Handle-filtering degraded acceptance removed.
- [x] Complete-sibling continuation implemented.
- [x] Focused deterministic tests pass.
- [x] Cognition V2 documentation updated.
- [x] Production-shaped live replay passes and artifact is inspected.
- [x] Adjacent live cases pass individually and are inspected.
- [x] Full non-live regression and static checks pass.
- [x] Parent-owned final review has no unresolved findings.
- [x] Execution evidence and acceptance criteria signed off.

## Verification

### Deterministic focused checks

```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_dependencies.py tests\test_cognition_core_v2_failures.py tests\test_cognition_core_v2_integration.py tests\test_cognition_core_v2_model_retry_continuity.py tests\test_cognition_core_v2_prompt_budget_continuity.py tests\test_llm_tracing.py tests\test_service_input_queue.py -m "not live_llm and not live_db" -q
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\cognition_core_v2\model_attempt_policy.py src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py src\kazusa_ai_chatbot\cognition_core_v2\facade.py src\kazusa_ai_chatbot\llm_tracing\failure_capsule.py src\kazusa_ai_chatbot\service.py
```

Required assertions:

- graph attempt two observes the remaining per-branch budget;
- a branch that consumed three calls performs zero further calls;
- attempt records contain invocation, graph, local, cumulative, limit, and
  disposition fields;
- goal exhaustion is typed, pre-commit, and non-retryable;
- unsupported handles never become accepted by deterministic deletion;
- required failure plus a complete mapping sibling reaches collapse;
- required failure plus zero complete bids raises before collapse;
- no partial bid reaches collapse;
- `selection_kind` is absent from active prompt/validator/test fixtures;
- healthy-path model-call and prompt caps do not increase.

### Live LLM checks

Run one at a time and inspect each newly written artifact:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_required_selection_live_llm.py::test_live_third_party_reply_ordinary_selection_contract_pressure -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_required_selection_live_llm.py::test_live_third_party_reply_autonomy_selection_contract_pressure -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_required_selection_live_llm.py::test_live_parallel_third_party_reply_selection_contract_pressure -q -s
```

The review records the case, route/model, rendered messages, raw response,
parsed candidate, exact validation result, required citations, attempt
coordinates, final branch/cognition disposition, latency, and semantic-quality
judgment. Secret configuration values remain excluded.

### Broader checks

```powershell
$cognitionV2Tests = Get-ChildItem -LiteralPath 'tests' -Filter 'test_cognition_core_v2*.py' | ForEach-Object { $_.FullName }
venv\Scripts\python.exe -m pytest $cognitionV2Tests -m "not live_llm and not live_db" -q
venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py tests\test_llm_tracing.py -m "not live_llm and not live_db" -q
git diff --check
rg -n "selection_kind|_degraded_selection_goal_draft|_degraded_goal_bid_draft|_raise_for_failed_required_branches" src\kazusa_ai_chatbot\cognition_core_v2 tests
```

The final `rg` result must contain no active producer or assertion that retains
the superseded contracts; unrelated historical plan text is excluded.

## Parent-Owned Review

The parent execution owner reviews the complete diff and evidence for:

- one monotonic budget per producing stage and branch;
- no counter reset on graph retry;
- exact semantic ownership and canonical parser use;
- no unsupported-handle deletion into acceptance;
- complete validation before sibling recovery;
- typed zero-bid exhaustion before downstream stages;
- protected-only raw content and redacted ordinary telemetry;
- no compatibility path, new model call, prompt-budget increase, or public
  schema change;
- CJK source validity and adequate deterministic/live coverage.

## Risks And Mitigations

- ContextVar scope can leak between turns if reset incorrectly. Bind around
  each graph invocation and direct facade call with token-based restoration;
  prove sequential and concurrent isolation.
- A graph retry after an early goal success has fewer remaining calls. This is
  intentional: the configured producer cap belongs to the cognition
  invocation, not each orchestration replay.
- Sibling continuation can produce silence or a non-ordinary action. Require a
  complete validated mapping bid and retain the failed required branch in
  diagnostics; workspace remains the final selection owner.
- Removing deterministic handle deletion can surface more typed failures.
  Producer repair feedback keeps the exact allowed handles and the live replay
  verifies bounded recovery.

## Execution Evidence

### Historical 2026-07-30 gate

- Three fixed live probes were run separately using
  `gemma-4-31b-fable-5-agent-distill`.
- Four real model invocations parsed and passed the then-current strict
  validator on their first attempt.
- The gate stopped before production edits and returned this plan to draft.

### 2026-08-08 architecture amendment and execution

- Parent plan: completed under explicit user execution authority.
- Protected evidence:
  `test_artifacts/diagnostics/cognition_v2_bid_exhaust_architecture_capture.md`
  and its linked protected JSON exports.
- Decision: replace independent local/service counters with the monotonic
  per-stage/per-branch ledger; remove unused `selection_kind`; keep invalid
  semantic/handle values fail-closed; continue only complete sibling bids.
- Execution model: parent-only under the user's 2026-08-08 directive.
- Production changes: implemented in `model_attempt_policy.py`, `contracts.py`,
  `goal_cognition.py`, `facade.py`, `service.py`, and protected tracing. The
  cutover removes `selection_kind`, removes filtered degraded acceptance,
  exposes strict public bid validation, preserves only complete validated
  siblings, and carries one invocation ledger across graph retries.
- Deterministic red evidence: the protected bid-exhaust capture and focused
  ledger/partial-recovery cases reproduced the unbounded/reset and sibling-loss
  failure contracts before their owning implementation was accepted.
- Focused verification: 217 passed with 4 deselected; the dedicated ledger and
  routing slice passed 21 tests; the Cognition V2 family passed 519 tests with
  2 precise skips and 230 deselections.
- Final non-live verification: 4,198 passed, 26 skipped, and 1,123 deselected
  in 236.47 seconds, with one nonblocking Starlette deprecation warning.
- Static verification: 41 changed Python files compiled, `git diff --check`
  passed, and the retired-contract scan returned zero active matches.
- Live LLM verification: ordinary, autonomy, and parallel ordinary/autonomy
  required-selection cases passed individually. The ordinary case recorded an
  unauthorized-handle rejection followed by a valid second candidate; the
  parallel case recorded isolated invocation IDs and counters.
- Human-readable live review:
  `test_artifacts/reviews/required_selection_partial_recovery_live_llm_review.md`
  (SHA-256
  `E24CE01310832C402E4AC4448DC44BE2335C8D911B9C16C339DD4A0B079C0654`).
- Parent review: accepted with no unresolved finding. The exhaustive audit is
  `test_artifacts/diagnostics/august_change_alignment_review.md`.

## Acceptance Criteria

- [x] Required-selection uses one canonical seven-field producer contract.
- [x] The three-call goal budget is cumulative across graph retries per branch.
- [x] Goal exhaustion is non-retryable and cannot trigger additional goal calls.
- [x] Every protected failed attempt records the approved budget coordinates and
  final branch disposition.
- [x] Unsupported semantic values and handles are regenerated or rejected; they
  are never rewritten into acceptance.
- [x] A complete sibling bid survives a required branch failure with typed warning
  and diagnostics; zero complete bids fail before workspace collapse.
- [x] Invalid bids never reach action planning, dialog, persistence, or delivery.
- [x] Production-shaped live replay and adjacent cases pass individually with
  inspected human-readable evidence.
- [x] Focused and broader deterministic suites pass.
- [x] Parent review has no unresolved finding.
- [x] README, archived plans, registry, evidence artifacts, and code describe one
  consistent contract.
