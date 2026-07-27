# Semantic appraisal terminal transition reliability plan

## Summary

- Goal: prevent semantic-appraisal terminal assertions from contradicting the
  reducer contract or aborting an otherwise valid character turn.
- Plan class: medium.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`, and `character-test`.
- Overall cutover strategy: bigbang contract replacement inside Cognition Core
  V2; no aliases or compatibility vocabulary.
- Highest-risk areas: terminal state semantics, reducer atomicity, appraisal
  retry ownership, degraded continuation, and local-model prompt clarity.
- Acceptance criteria: terminal and explicit nonterminal assertions are
  mechanically valid, handles remain in their declared domains, invalid
  candidates use bounded replacement, and the captured QQ E2E returns normal
  character dialog without semantic terminal contradiction.

## Context

Production traces for QQ user `673225019` contain repeated failures with
`CognitionStateError: goal completion evidence is required`. The exact turn
`chat:qq:ch_732699d6699040ae:197517893` was reproduced through the complete
service queue with the captured private-history window, user cognition state,
character state, durable user memory, original QQ identity, and real configured
LLM routes.

The reproduced appraisal emitted `completion_meaning` for goal `g1` while its
own semantic prose said the goal was still in progress. Structural validation
accepted the row because the proposition vocabulary was a list of bare,
cross-entity tokens and validation intentionally did not interpret free-form
prose. The reducer treated the token as terminal authority and called
`transition_goal(..., satisfied)`, but it did not first establish the required
`progress == 100` invariant. The transition guard correctly rejected the
incomplete state. That reducer exception occurred after parallel preliminary
goal cognition and outside the appraisal producer's bounded replacement loop,
so it escaped to the service and became the user-visible operational notice.

The replay also observed one malformed `relationship_social` candidate. Its
existing bounded replacement succeeded on the next attempt. This is recorded
as a recovered contract failure and requires regression preservation rather
than a new production path.

Evidence is in the numbered raw artifacts and their E2E replay review.

Attempt 05 completed normally yet reproduced a false `event_completed` and
three relationship failures using evidence `e1` as subject. Two real-model
experiments resolved them with `outcome_pending` and handle field domains.
Review found reducer/ledger gaps. Attempt 08 then exposed an undefined `self`
role-handle meaning and generic repair feedback that repeated one invalid
Chinese role label through all attempts.

## Mandatory Skills

- `development-plan`: govern this plan, checkpoints, evidence, and review.
- `local-llm-architecture`: preserve LLM semantic ownership and local-model
  latency/context limits.
- `py-style`: load before every production or test Python edit.
- `cjk-safety`: validate edited Python files containing Chinese prompt text.
- `test-style-and-execution`: use test-first changes and run live LLM cases one
  at a time with artifact inspection.
- `debug-llm`: keep separate raw and human-readable E2E evidence.
- `character-test`: verify the final full service path and all stage failures.

## Mandatory Rules

- LLM stages own semantic terminal judgment; deterministic code owns exact
  enum/target validation, state invariants, bounded retries, persistence, and
  failure containment.
- Replace ambiguous terminal vocabulary with explicit affirmative,
  entity-specific kinds. Do not preserve aliases for `completion_meaning` or
  `resolution_meaning`.
- Prompt changes define positive proposition semantics in one compact mapping.
  Do not add a list of negative examples, keyword rules, prose classifiers, or
  repeated prohibitions.
- Do not parse `semantic_value` or `explanation` text to override the model's
  structured semantic decision.
- Use the existing appraisal attempt cap. Do not add a model call, stage,
  service retry, route, or completion-budget increase.
- A reducer-compatibility rejection belongs to the producing appraisal's
  existing replacement loop. Exhaustion remains an omitted appraisal with
  typed diagnostics.
- A residual final-reduction rejection must discard only the offending copied
  appraisal result and continue from the last validated state.
- Keep the transition guards strict. Establish terminal invariants before
  invoking them; do not weaken or bypass them.
- Preserve the recovered malformed-JSON/handle replacement behavior observed
  in the E2E replay.
- Use `venv\Scripts\python.exe`. Run real LLM tests individually and inspect
  both raw and review artifacts.
- Preserve all unrelated worktree changes.
- After any automatic context compaction, reread this entire plan before
  continuing.
- After signing off a major checklist stage, reread this entire plan before the
  next stage.
- Before completion or lifecycle archival, run the Independent Code Review gate
  and record it in Execution Evidence.
- Execute through the parent-led native subagent model below.

## Must Do

- Replace the two ambiguous cross-entity terminal proposition kinds with exact
  affirmative goal, threat, event, and knowledge-gap kinds.
- Include compact positive semantics for proposition kinds and `self`/
  `current_user`, an explicit pending outcome, and field-handle domains.
- Validate proposition-kind and subject-kind compatibility structurally.
- Validate a structurally accepted appraisal by reducing it against a copied
  native state before returning it from the producer.
- Atomically establish terminal axes for existing and new causal candidates
  before transition guards and preserve them after accepted batch deltas.
- Isolate residual rejection through cumulative accepted-batch replay so
  cross-appraisal composition and the preliminary response remain intact.
- Add deterministic tests for vocabulary, target compatibility, atomic
  terminalization, producer-owned replacement, and residual isolation.
- Re-run the exact full E2E replay until one post-fix run is inspected.
- Record every failed or recovered stage in the replay ledger and review.

## Deferred

- No changes to relevance, RAG, dialog wording, adapters, database schemas,
  persistence formats, model routing, or service-level operational notices.
- No new evaluator, helper agent, retry tier, feature flag, compatibility shim,
  fallback vocabulary, or legacy alias.
- No attempt to eliminate every malformed local-model response; bounded
  replacement remains the intended handling.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Proposition vocabulary | bigbang | Replace ambiguous terminal tokens everywhere in source and tests. |
| Reducer dispatch | bigbang | Dispatch only entity-specific terminal kinds. |
| Appraisal retry | compatible | Preserve the existing attempt count and replacement message shape. |
| Service response | compatible | Preserve operational errors for genuine unrecoverable service failures. |
| Persistent data | compatible | Keep existing cognition-state schema and stored entity/status vocabulary. |

Cutover enforcement:
- Update producer, validator, reducer, tests, and documentation in one scope.
- Static verification must find no active source/test use of the retired
  terminal proposition kinds.
- Any cutover-policy change requires user approval.

## Target State

The `goal_threat_outcome` family offers entity-specific terminal assertions
plus `outcome_pending` for a subject that remains nonterminal. The payload
defines each kind/handle domain; validators enforce terminal target kinds.

A valid terminal assertion establishes its terminal axes on a copied state and
then passes the unchanged FSM guard:

- goal completion sets `progress = 100`, then transitions to `satisfied`;
- threat resolution sets `residual_pressure = 0`, then transitions to
  `resolved`;
- event completion or repair sets `repair_need = 0`, with repair also setting
  `reparability = 100`, then transitions to `resolved`;
- knowledge answer sets `uncertainty = 0`, then transitions to `resolved`.

Each appraisal candidate is reducer-validated before the producer accepts it.
Reducer incompatibility becomes a normal contract replacement attempt. If all
attempts fail, the existing appraisal-exhaustion path omits that question.
`outcome_pending` carries semantic observation only and performs no state
transition or candidate materialization.
Final reduction cumulatively replays accepted results from the original state,
preserving composition while recording and omitting one residual failure.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Semantic vocabulary | Use `goal_completed`, `event_completed`, `threat_resolved`, `event_repaired`, and `knowledge_answered` alongside existing goal release/supersession kinds. | Entity-specific affirmative names remove the observed topic-versus-assertion ambiguity. |
| Prompt contract | Add positive kind/role-handle meanings, `outcome_pending`, field domains, and exact repair errors. | Gives the local model actionable structured channels without negative-rule accumulation. |
| Terminal axes | Set canonical terminal axes before unchanged guards and reassert their postconditions after deltas. | A terminal assertion is typed completion evidence and remains authoritative for the completed batch. |
| Producer validation | Trial-reduce each result against a copied preliminary state inside `appraise_semantic_question`. | Keeps repair with the semantic producer and never commits trial state. |
| Final containment | Re-reduce the cumulative accepted prefix from the original state. | Isolates one invalid row while preserving cross-appraisal composition. |
| Companion malformed response | Preserve existing bounded replacement and add evidence/tests. | Attempt 03 proves the current handling already recovers correctly. |

## Contracts And Data Shapes

`question_proposition_kinds("goal_threat_outcome")` returns:

```text
goal_release
goal_supersession
goal_completed
event_completed
threat_resolved
event_repaired
knowledge_answered
outcome_pending
```

The appraisal question payload adds:
```python
"proposition_kind_semantics": {
    "<kind>": "<compact positive assertion meaning>",
},
"handle_field_domains": {"<handle field>": ["<permitted handle>"]},
```

`appraise_semantic_question(...)` receives the validated preliminary mutable
state as a required argument. It never serializes that native state directly;
only the existing bounded projection enters the model payload.

Residual reduction failure uses the diagnostic code:
```text
semantic_appraisal_reduction_rejected
```

No persistent schema, service API, or adapter contract changes.

## LLM Call And Context Budget

| Stage | Before | After |
|---|---|---|
| Semantic appraisal normal path | One call per selected family; up to the existing total-attempt cap on invalid output. | Same call count and cap. |
| Semantic appraisal invalid reducer candidate | Could escape after one structurally valid call. | Uses remaining attempts from the existing cap; exhaustion is omitted. |

- Response path, model routes, and completion caps remain unchanged.
- The compact vocabulary/domain map stays inside the existing 8,000-character
  payload cap; truncation order, routes, and model limits remain unchanged.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py`:
  expose terminal/nonterminal vocabulary and positive kind semantics.
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`: include the
  semantic map, enforce target-kind compatibility, and trial-reduce candidates
  inside the existing attempt loop.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py`: dispatch exact
  terminal kinds and terminalize existing or newly created candidates.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`: pass preliminary state
  to producers and isolate failure with cumulative batch replay.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`: document exact terminal
  semantics and degraded containment.
- Existing focused tests: replace retired kinds and cover explicit pending/domain
  behavior.

### Create

- `tests/test_cognition_core_v2_semantic_terminalization.py`: focused contract,
  reducer, retry, and containment tests.
- Raw/review evidence under `test_artifacts/llm_traces/`.

## Overdesign Guardrail

- Actual problem: an ambiguous terminal semantic token can be structurally
  accepted and then violate a strict reducer guard, aborting the turn.
- Minimal change: exact affirmative tokens, atomic terminal axes, producer-owned
  trial reduction, and per-appraisal final isolation.
- Ownership boundaries: the LLM selects semantic terminal meaning;
  deterministic code validates token/target shape, establishes invariants,
  guards transitions, and contains invalid candidates.
- Rejected complexity: prose classifiers, extra evaluators/retries, and shims.
- Evidence threshold: a new reproduced failure outside these exact terminal
  contracts is required before expanding the architecture.

## Agent Autonomy Boundaries

- The parent owns tests, verification, evidence, plan lifecycle, review fixes,
  and sign-off.
- The production-code subagent owns only the five production/documentation
  files listed under Modify and does not edit tests.
- The review subagent reviews only and does not implement fixes.
- Equivalent existing behavior must be reused rather than duplicated.
- Changes outside the listed surface, new APIs, or altered model budgets require
  a plan update and user authority.
- Unrelated cleanup, dependency changes, and prompt rewrites are prohibited.

## Implementation Order

1. Parent adds focused failing tests for the exact vocabulary, atomic goal
   completion, target-kind validation, trial-reduction replacement, and residual
   isolation.
2. Parent runs the focused tests and records expected pre-implementation
   failures.
3. Parent starts exactly one production-code subagent with this plan and the
   focused test contract.
4. The subagent updates planner, appraisal, reducer, facade, and Cognition Core
   V2 documentation only.
5. Parent reruns focused tests, then relevant cognition failure, integration,
   routing, and frozen-replay tests.
6. Parent runs the exact full live E2E replay once at a time and inspects raw
   calls, validation stages, event log, graph result, persistence, and response.
7. Parent starts exactly one independent code-review subagent, remediates
   in-scope findings, and reruns affected verification.

## Execution Model

- Parent-led native subagent execution is required.
- Exactly one production-code subagent starts after the focused failing test is
  established and closes after production edits.
- Exactly one independent code-review subagent starts after planned
  verification passes and reports findings without editing.
- If native subagents are unavailable, execution stops unless the user
  explicitly authorizes fallback execution.

## Progress Checklist

- [x] Stage 1 - focused test contract established.
  - Verify: focused tests fail for the old vocabulary/reducer behavior.
  - Evidence: record command and failure summaries below.
  - Handoff: start production-code subagent.
  - Sign-off: parent confirmed nine production-contract failures.
- [x] Stage 2 - production contract implemented and refined.
  - Verify: focused tests pass and retired tokens are absent from active source.
  - Evidence: record changed files and commands below.
  - Handoff: run integration and live E2E gates.
  - Sign-off: 21 focused/repeated-terminal tests pass after attempt 08 fix.
- [x] Stage 3 - deterministic and live E2E verification complete.
  - Verify: all commands below pass; live artifact has normal character text
    and no target runtime error.
  - Evidence: record raw/review paths and failure ledger.
  - Handoff: start independent code review.
  - Sign-off: 84 tests and attempt 10 passed with every failure ledgered.
- [x] Stage 4 - independent code review complete.
  - Verify: review has no unresolved blocker and affected commands are rerun.
  - Evidence: record reviewer, findings, fixes, and approval below.
  - Handoff: archive the completed plan.
  - Sign-off: Laplace approved with no remaining findings.

## Verification

### Focused tests
```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_semantic_terminalization.py -q
```

### Regression tests
```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_failures.py tests\test_cognition_core_v2_stage_model_routing.py tests\test_cognition_core_v2_frozen_replay_drift.py tests\test_cognition_core_v2_integration.py -q
```

### Static checks
```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\cognition_core_v2\semantic_source_planner.py src\kazusa_ai_chatbot\cognition_core_v2\semantic_appraisal.py src\kazusa_ai_chatbot\cognition_core_v2\state_reducers.py src\kazusa_ai_chatbot\cognition_core_v2\facade.py tests\test_cognition_core_v2_semantic_terminalization.py
rg -n "completion_meaning|resolution_meaning" src\kazusa_ai_chatbot\cognition_core_v2 tests
```

The `rg` command must return zero active source/test matches. Exit code 1 is the
expected zero-match result.

### Live E2E

Run the guarded replay individually through
`test_artifacts/diagnostics/test_completion_meaning_e2e_replay_live_llm.py`.
The inspected artifact must show:

- `trace_run.status == "succeeded"`;
- normal text content with no `operational_error`;
- a persisted assistant row and normal post-turn lifecycle;
- no `CognitionStateError: goal completion evidence is required`;
- every recovered or failed validation stage present in the failure ledger.

## Independent Code Review

After all verification passes, start one independent review subagent. Review the
approved plan, complete diff, focused/regression/live evidence, prompt
minimality, reducer invariants, trial-state isolation, final containment,
worktree preservation, and absence of compatibility vocabulary. The parent may
fix only findings inside this change surface, reruns affected verification, and
records approval before completion.

## Acceptance Criteria

This plan is complete when:

- the exact E2E failure no longer reaches the service operational-error path;
- terminal proposition kinds are affirmative and entity-specific;
- existing and candidate terminal assertions satisfy FSM guards atomically;
- reducer-incompatible candidates use bounded producer replacement;
- exhaustion or residual rejection omits only the failed appraisal;
- recovered malformed appraisal output remains recovered;
- deterministic and individual live tests pass;
- independent code review has no unresolved blocker.

## Risks
| Risk | Mitigation | Verification |
|---|---|---|
| Model uses a terminal token to report ongoing state | Explicit `outcome_pending` structured channel. | Exact real-model replay. |
| Terminal assertion mutates the wrong entity kind | Exact proposition-target matrix. | Focused validator tests. |
| Trial validation mutates live state | Reduce only deep-copied state and discard trial output. | State immutability test. |
| Isolation loses composed candidate/delta state | Cumulative accepted-prefix replay. | Cross-appraisal composition test. |
| Prompt becomes over-constrained | One positive semantic map; no negative examples. | Diff review and prompt-size check. |

## Execution Evidence

- Pre-fix exact E2E: attempt 03 reproduced the production stack and operational
  response; attempt 02 reached the same prompt/state and completed normally,
  demonstrating sampling sensitivity.
- Companion malformed relationship handle recovered on `repair_1`.
- Stage 1 focused test: nine expected contract failures; no setup failure.
- Stage 2 refinement: 21 focused/repeated-terminal tests pass after all fixes.
- Stage 3: 84 deterministic tests pass; attempts 09 and 10 pass the exact E2E.
- Stage 4: Laplace approved the final diff with no remaining findings.
