# cognition core v2 prompt budget and failure containment bugfix plan

## Summary

- Goal: eliminate the repeated semantic-appraisal context overflow at its
  aggregate serialization root and prevent equivalent pre-invocation prompt-cap
  failures from bypassing existing V2 degraded-continuation contracts.
- Plan class: large.
- Status: completed and archived.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`, and `character-test`.
- Overall cutover strategy: bigbang replacement of the internal prompt-visible
  appraisal permission shape and direct correction of owner-local cap
  dispositions; no compatibility vocabulary or parallel path.
- Highest-risk areas: prompt authority equivalence, evidence truncation,
  required goal cognition, action denial, surface degradation, and production
  regression fidelity.
- Acceptance criteria: the production-derived eight-evidence-row case reaches the
  configured epistemic appraisal model below 8,000 characters, maximum-shape
  deterministic cases remain bounded, and every audited degradable
  pre-invocation cap uses its existing skip, fallback, deny, or degraded
  surface instead of the operational-error path.

## Context

Two production QQ turns for user `257629823` failed with the same
`CognitionContextLimitError`, stack fingerprint, and
`cognition_failure:context_limit` outcome. A read-only production-derived
focused run reproduced the failure with one current episode row, two fixed
production memories, and five production conversation packets.

The failing epistemic appraisal had 1,432 characters of semantic evidence but
an 8,684-character minimum serialized payload against an 8,000-character cap.
The prompt contained the same evidence text at top level and inside
`state.evidence`, plus 87 fully expanded delta paths. The failure occurred
before the configured LLM was invoked.

The completed retry-continuity work already defines the correct owner outcomes:
semantic appraisal omission, workspace branch-order fallback, empty action
plan, deny-all authorization, degraded text surface, and optional visual
omission. The current code starts several of those outcomes only after entering
an LLM attempt loop. Prompt-cap checks executed before those loops can still
escape to the service and produce an adapter-visible operational notice.

The repository-wide live-path scan found the following equivalent gaps:

- semantic-appraisal aggregate payload overflow;
- goal-cognition aggregate prompt overflow and duplicated evidence;
- required-selection verifier and repair prompt overflow;
- workspace-collapse prompt overflow;
- action-planning prompt overflow;
- action-authorization prompt overflow;
- resolver-authorization prompt overflow;
- content, preference, repair, and visual surface prompt overflow.

Dialog semantic-verifier context overflow already returns `unavailable` and is
the correct contained reference behavior. Prompt message-envelope overflow is
an intentional required-addressing safety failure and remains outside this V2
change. Background coding and growth prompt limits do not enter the current
character reply surface.

## Mandatory Skills

- `development-plan`: govern the approved scope, checkpoints, evidence,
  independent review, and lifecycle closeout.
- `local-llm-architecture`: preserve local-model context, call-count, semantic
  ownership, and latency boundaries.
- `py-style`: load before every production or test Python edit.
- `cjk-safety`: protect Chinese prompt source and run immediate syntax checks.
- `test-style-and-execution`: establish deterministic red contracts first and
  run live LLM verification one case at a time.
- `debug-llm`: preserve raw evidence and author a separate readable quality
  review after the live run.
- `character-test`: enforce explicit production-data scope and read-only
  focused verification.

## Mandatory Rules

- Deterministic code owns serialization budgets, truncation, exact path
  authority, retry limits, skip/fallback/deny dispositions, and delivery
  containment. LLM stages retain semantic appraisal, goal, action, and surface
  judgment.
- Keep top-level appraisal `evidence` as the only prompt-visible semantic
  evidence registry. Remove appraisal `state.evidence`.
- Keep the complete exact `permitted_delta_paths` set private for deterministic
  validation. Project one compact grouped domain to the model and preserve
  exact canonical `target_path` output.
- Preserve source order. Reduce supplemental state first, then truncate
  lower-priority evidence rows from the end while preserving both ends and a
  non-empty semantic floor.
- Never expose a handle to the model without its retained evidence row and
  exact deterministic validator authority.
- Preserve every current character, user, source, permission, persistence,
  state, and action authorization boundary.
- Preserve normal-path LLM call counts and all existing character-route
  assignments. Add no model call, provider retry, cap increase, feature flag,
  or service retry.
- Use the existing owner disposition for pre-invocation cap exhaustion:
  appraisal omits, workspace selects the stable complete bid, action planning
  returns no work, authorization denies, required-selection verification
  becomes unavailable, and surface planning degrades or omits visual output.
- Keep malformed required canonical input, invalid persistent state, failed
  commit invariants, and total dialog-generator unavailability as
  unrecoverable errors.
- Keep service, adapters, databases, persistence schemas, RAG routing, and
  dialog wording outside the production change surface.
- Use `venv\Scripts\python.exe`.
- Run regular deterministic tests in batches. Run the production-data real LLM
  case once and inspect it before any further live call.
- Preserve all pre-existing unrelated worktree changes.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off a major checklist stage, reread this entire plan before
  starting the next stage.
- Before completion, lifecycle changes, or sign-off, run the Independent Code
  Review gate and record its result in Execution Evidence.
- Execute through the parent-led native subagent model in this plan.

## Must Do

- Add production-derived deterministic tests that fail on the current
  `8,684 > 8,000` epistemic appraisal boundary.
- Add maximum-shape and exact cap-boundary tests for the appraisal serializer.
- Remove duplicated appraisal evidence.
- Replace prompt-visible expanded delta paths with grouped field, handle, and
  axis domains while retaining exact private validation.
- Add aggregate evidence truncation after supplemental state reduction.
- Make residual irreducible appraisal context omit only that appraisal.
- Remove duplicated goal-cognition evidence and fit the complete serialized
  goal payload under its existing 24,000-character cap.
- Apply established cap dispositions to required-selection, workspace, action,
  authorization, resolver, and surface owners.
- Add deterministic tests proving those cap dispositions invoke no operational
  response path and authorize no side effects.
- Run the focused read-only production-data epistemic appraisal through its
  real configured LLM route once after deterministic verification.
- Inspect and author a readable before/after review from the raw live artifact.
- Update the Cognition Core V2 README with aggregate-budget and pre-invocation
  containment contracts.

## Deferred

- Do not change the 8,000-, 12,000-, 16,000-, 18,000-, 24,000-, or
  50,000-character caps.
- Do not reduce global RAG retrieval, alter RAG planning, or repair the separate
  relationship-memory routing drift.
- Do not change service or adapter operational-response contracts.
- Do not suppress required message addressing, malformed persistent state,
  commit failures, or total dialog-generator failure.
- Do not add a generic model invocation wrapper, prompt framework, alternate
  serializer, compatibility mapper, alias field, fallback prompt, or extra
  evaluator.
- Do not modify databases or send a new production chat message.
- Do not refactor unrelated prompts, state models, reducers, persistence,
  consolidation, relevance, or dialog wording.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Appraisal evidence | bigbang | Keep one top-level semantic-text registry; remove the duplicate state registry. |
| Appraisal delta authority | bigbang | Replace expanded prompt paths with grouped domains in one prompt update; retain exact private validation. |
| Evidence budgeting | bigbang | Fit aggregate serialized payloads through deterministic supplemental trimming and evidence truncation. |
| Existing owner fallbacks | compatible | Reuse the exact skip, fallback, deny, degraded text, and visual omission outcomes already defined. |
| Service and adapters | compatible | Preserve current API and delivery behavior; contained V2 failures never reach it. |
| Persistent data | compatible | Preserve all schemas and writes; this plan performs no migration. |

Cutover enforcement:

- Update prompt construction, deterministic validation tests, and documentation
  together.
- Remove the old prompt-visible `permitted_delta_paths` shape without aliases.
- Preserve only the exact compatibility surfaces listed above.
- Any cutover-policy change requires user approval.

## Target State

Semantic appraisal owns one complete serialized 8,000-character packet:

```text
question contract
  + one evidence registry
  + authorized state projection
  <= 8,000 characters
```

The model receives grouped delta domains:

```python
{
    "permitted_delta_path_domains": [
        {
            "state_field": "active_events",
            "handles": ["ce1", "ce2"],
            "axes": ["comparison_gap", "memory_warmth"],
        },
    ],
}
```

The model still returns:

```text
active_events.ce1.comparison_gap
```

The validator checks that value against the unchanged private exact-path set.

Evidence appears once. Supplemental state rows are removed in stable reverse
priority. If the complete packet still exceeds the cap, evidence rows remain
handle-addressable and their semantic text is middle-truncated from the lowest
priority row backward. If the packet cannot fit after every allowed reduction,
only that appraisal is omitted with typed diagnostics.

Goal cognition uses one top-level evidence registry, removes the duplicated
projection registry, trims supplemental projected state, and applies the same
stable evidence-text fitting mechanics under 24,000 characters. It preserves
the current episode and source order.

Every equivalent pre-invocation cap enters the outcome already owned by its
stage. No action or resolver request is authorized from a skipped model call.
Text surface overflow yields a validated degraded surface; visual overflow
yields no visual directives.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Appraisal budget owner | The complete serialized appraisal builder owns the cap. | Local producer caps cannot prove aggregate fit. |
| Evidence registry | One top-level registry. | Removes the reproduced duplication without losing handle provenance. |
| Delta authority | Compact prompt domain plus private exact set. | Preserves strict validation while eliminating repeated mechanical strings. |
| Truncation order | Supplemental state, then lowest-priority evidence text. | Preserves current episode and source ranking. |
| Truncation shape | Preserve head, tail, handle, source kind, and a non-empty floor. | Retains semantic clues and exact provenance. |
| Irreducible appraisal | Omit the question. | Appraisals are independent and already degradable. |
| Required goal cognition | Fit the prompt rather than invent a deterministic goal. | Character stance remains LLM-owned. |
| Workspace overflow | Use existing branch-order fallback. | All bids are already valid and complete. |
| Action and authorization overflow | Empty plan and deny all. | Preserves side-effect safety and visible reply continuity. |
| Surface overflow | Raise the existing typed owner failure. | Existing caller already turns typed text failure into degraded output and visual failure into omission. |

## Contracts And Data Shapes

Create an internal deterministic budget helper in
`cognition_core_v2/prompt_budget.py`:

```python
class PromptBudgetError(ValueError):
    """Required prompt structure cannot fit after permitted reduction."""


def fit_evidence_texts_to_budget(
    payload: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    *,
    text_field: str,
    maximum_chars: int,
    minimum_text_chars: int,
) -> str:
    """Serialize or middle-truncate low-priority evidence until it fits."""
```

The helper receives caller-owned copied rows already present in `payload`.
It preserves row count, row order, handles, source kinds, and non-text fields.
It returns deterministic `json.dumps(..., ensure_ascii=False,
sort_keys=True)` output. It raises `PromptBudgetError` only after every row is
at the declared floor and the full payload remains over budget.

Appraisal uses:

```text
SEMANTIC_APPRAISAL_PROMPT_CAP = 8000
MIN_PROMPT_EVIDENCE_TEXT_CHARS = 96
```

Goal cognition uses:

```text
GOAL_COGNITION_PROMPT_CAP = 24000
MIN_PROMPT_EVIDENCE_TEXT_CHARS = 96
```

No public API, persistent schema, service response, evidence handle, or
canonical target-path contract changes.

## LLM Call And Context Budget

| Owner | Before | After |
|---|---|---|
| Six appraisals | One normal call per selected family; pre-call overflow aborts the turn. | Same calls; every call packet is at most 8,000 characters; irreducible family skips. |
| Goal cognition | One call per selected branch; duplicated evidence and aggregate overflow can fail before invocation. | Same calls; one evidence registry and at most 24,000 characters. |
| Required-selection verifier/repair | Existing bounded calls; cap failure escapes before invocation. | Same bounded calls; cap failure uses unavailable/latest-valid disposition without a call. |
| Workspace collapse | Up to three attempts; cap failure escapes before invocation. | Same normal attempts; cap failure uses zero-call stable fallback. |
| Action planning | Up to three attempts; cap failure escapes before invocation. | Same normal attempts; cap failure uses zero-call empty plan. |
| Action/resolver authorization | Up to three attempts; cap failure escapes before invocation. | Same normal attempts; cap failure uses zero-call deny-all. |
| Text/visual surface | Up to three attempts; untyped cap failure escapes. | Same normal attempts; cap failure becomes typed degraded text or visual omission. |

No response-path call is added. No completion budget, provider route, or
context cap increases. Character-based measurement uses exact rendered JSON
length, matching production.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/cognition_core_v2/prompt_budget.py`: shared
  deterministic evidence-text budget mechanic only.
- `tests/test_cognition_core_v2_prompt_budget_continuity.py`: production-derived
  appraisal, maximum-shape, cap disposition, and no-call safety tests.
- `tests/fixtures/cognition_core_v2_prompt_budget_production_case.json`:
  sanitized production-derived evidence text and incident shape.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`: one evidence
  registry, grouped delta domains, aggregate fitting, and prompt guidance.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`: isolate residual
  appraisal budget exhaustion as one omitted question.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: remove evidence
  duplication, fit aggregate prompt, and contain required-selection cap paths.
- `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py`: route pre-call cap to
  the existing stable fallback.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`: route pre-call
  cap to the existing empty plan.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py`: deny every
  candidate on pre-call cap.
- `src/kazusa_ai_chatbot/cognition_core_v2/resolver_authorization.py`: deny
  every candidate on pre-call cap.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`: classify prompt
  cap through the existing typed surface failure owner.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`: document aggregate
  budget ownership and pre-call containment.
- `experiments/focused_context_limit_625857543_live_llm.py`: convert the
  historical focused boundary runner into a post-fix single-target,
  read-only verification while retaining incident references.
- `development_plans/README.md`: register and close this plan.

### Keep

- `src/kazusa_ai_chatbot/service.py` and all adapters.
- RAG planning, retrieval counts, database code, state schema, reducers,
  persistence, consolidation, and dialog prompt/wording.
- Existing character and route configuration.

## Overdesign Guardrail

- Actual problem: independently bounded prompt producers compose into
  over-cap packets, and pre-call cap errors bypass established V2 degraded
  outcomes.
- Minimal change: one canonical evidence truncation mechanic, one appraisal
  evidence registry, compact path authority, goal evidence deduplication, and
  direct wiring to existing owner fallbacks.
- Ownership boundaries: LLMs judge semantics; deterministic builders own
  serialization, authority, truncation, and cap disposition; service and
  adapters only deliver completed outcomes.
- Rejected complexity: new calls, retries, cap increases, global RAG reduction,
  provider wrappers, feature flags, compatibility fields, and service catches.
- Evidence threshold: a separately reproduced non-V2 prompt failure or an
  irreducible required-addressing case is required before expanding this
  boundary.

## Agent Autonomy Boundaries

- Parent owns plan lifecycle, tests, fixtures, experiments, deterministic and
  live verification, evidence, review remediation, and sign-off.
- Exactly one production-code subagent owns only the ten production files
  listed under Create/Modify, including the subsystem README. It does not edit
  tests, experiments, plans, or artifacts.
- Exactly one independent review subagent reviews only and implements no fixes.
- The responsible agent may choose local loop mechanics only when the exact
  contracts above remain unchanged.
- Search existing helpers before adding a function. The new budget helper may
  contain only the shared non-trivial evidence truncation behavior.
- Changes outside the listed production surface, new public interfaces, or
  altered failure taxonomy require a plan update and user authority.
- Unrelated cleanup, formatting churn, prompt rewrites, dependency changes, and
  schema changes are prohibited.
- If the plan and source disagree, preserve this plan's stated outcome and
  report the discrepancy.
- If a required instruction is impossible, stop and report the blocker.

## Implementation Order

1. Parent adds the production-derived fixture and focused deterministic test
   file covering appraisal reproduction, maximum shape, residual omission,
   goal fitting, and each pre-call owner disposition.
2. Parent runs the focused file and records the expected failures against the
   current implementation.
3. Parent starts exactly one production-code subagent with this plan, mandatory
   skills, red-test output, and the listed production ownership boundary.
4. Production subagent adds the budget helper, fixes appraisal and goal
   payloads, wires existing owner dispositions, and updates the V2 README.
5. Parent reruns the focused tests, compiles every changed Python file, renders
   affected prompts, and runs affected deterministic regressions.
6. Parent updates the focused production-data harness to run only the formerly
   failing epistemic appraisal and save a new raw post-fix artifact.
7. Parent runs that real LLM case once against read-only production data,
   inspects the raw prompt/output/validation trace, and authors a readable
   review.
8. Parent starts exactly one independent code-review subagent with the plan,
   full diff, and verification evidence.
9. Parent remediates in-scope findings, reruns affected checks, records
   evidence, archives the completed plan, and updates the registry.

## Execution Model

- Parent agent owns orchestration, tests, fixtures, verification, evidence,
  review remediation, lifecycle updates, and final sign-off.
- Parent establishes and runs the focused red test before production work.
- Production-code subagent: exactly one native subagent, started after the red
  contract; owns production source and subsystem README only; closes after
  planned production edits.
- Parent may continue experiment and verification preparation while the
  production subagent edits its disjoint file set.
- Independent code-review subagent: exactly one native subagent after planned
  verification; reviews the plan, diff, tests, and evidence; makes no edits.
- Native subagent unavailability stops execution unless the user explicitly
  approves fallback execution.

## Progress Checklist

- [x] Stage 1 - focused failure contracts established.
  - Covers: implementation steps 1-2.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_prompt_budget_continuity.py -q`.
  - Evidence: record exact expected failures and current payload sizes.
  - Handoff: start production-code subagent.
  - Sign-off: parent agent, 2026-07-28. Focused red result:
    13 expected failures and 5 control passes; the production-derived
    epistemic packet failed with `CognitionContextLimitError` before its model
    boundary, and every adjacent failure matched its audited preflight gap.
- [x] Stage 2 - production budget and containment contracts implemented.
  - Covers: implementation steps 3-4.
  - Verify: focused tests pass; changed production Python compiles.
  - Evidence: changed files, prompt sizes, truncation counts, and no-call
    dispositions.
  - Handoff: start deterministic regression and live harness update.
  - Sign-off: parent agent, 2026-07-28. All changed production Python
    compiled; focused green result was 18 passed. The production-derived
    epistemic payload reached the model boundary at 6,627 characters with
    eight evidence handles and exact reconstruction of all 87 private paths.
- [x] Stage 3 - deterministic and one-case live verification complete.
  - Covers: implementation steps 5-7.
  - Verify: all commands below pass and readable live review is authored.
  - Evidence: commands, raw artifact, validation trace, and review path.
  - Handoff: start independent code review.
  - Sign-off: parent agent, 2026-07-28. Affected regression: 194 passed
    and 4 deselected. Initial full regular V2 regression: 304 passed and 150
    deselected. Shared prompt contracts: 11 passed. Changed Python compiled
    and `git diff --check` passed. The one-case run queried production identity,
    profile, memory, turn, and history rows for QQ `257629823`, reconstructed
    the pre-failure cognition state from the preserved replay artifact, and
    called the configured epistemic model once with a 6,642-character packet.
    It used zero retries and invoked neither a database mutation path nor a
    production delivery path. Its parseable first candidate received the typed
    contract disposition `resolved knowledge gap cannot transition`; the
    original context-limit failure was absent.
- [x] Stage 4 - independent review and lifecycle closeout complete.
  - Covers: implementation steps 8-9.
  - Verify: no unresolved review finding; affected checks rerun.
  - Evidence: reviewer identity, findings, fixes, residual risks, registry and
    archive paths.
  - Handoff: final user report.
  - Sign-off: parent agent, 2026-07-28. Independent reviewer Maxwell
    (`019fa79d-3328-7e03-9400-544baa72b567`) reported one high, three medium,
    and one low finding. Parent remediation removed residual goal duplication,
    budgeted every audited repair attempt, restored malformed appraisal input
    to invariant failure, strengthened exact/max-shape tests, and corrected
    live-evidence provenance. Focused remediation moved from 9 failed/19 passed
    to 28 passed. Final affected regression was 194 passed/4 deselected; final
    full V2 regression was 314 passed/150 deselected; prompt contracts were
    11 passed. Static, JSON, diff, and changed-path gates passed. No review
    finding remains unresolved.

## Verification

### Focused red/green contract

```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_prompt_budget_continuity.py -q
```

Required assertions:

- the production-derived epistemic packet reaches the LLM boundary at no more
  than 8,000 characters;
- semantic evidence appears in one registry;
- compact domains reconstruct the private exact path set;
- maximum-shape appraisal families fit or omit independently;
- low-priority evidence text is truncated while handles and order remain;
- a residual irreducible appraisal records one failure and warning;
- 32 maximum-size valid evidence rows fit goal cognition at no more than
  24,000 characters without duplicated evidence;
- required-selection overflow retains the latest valid bid;
- workspace overflow selects the first stable complete bid;
- action planning overflow returns no work and `blocked`;
- action and resolver authorization overflow deny all and make no model call;
- text surface overflow returns validated degraded output;
- visual surface overflow is a typed optional-stage failure.
- every audited repair overflow stops before the next model call and uses the
  same owner disposition as initial overflow;
- malformed appraisal state/evidence projections remain invariant failures;
- exact-cap packets pass and cap-plus-one packets take the declared failure
  path.

### Affected deterministic regression

```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_semantic_terminalization.py tests\test_cognition_core_v2_failures.py tests\test_cognition_core_v2_alignment_gates.py tests\test_cognition_core_v2_integration.py tests\test_cognition_core_v2_action_planning_bugfix.py tests\test_cognition_core_v2_action_authorization.py tests\test_cognition_core_v2_model_retry_continuity.py tests\test_cognition_core_v2_transition_coherence.py tests\test_service_cognition_graph.py tests\test_service_input_queue.py -m "not live_llm and not live_db" -q
```

### Full regular V2 regression

```powershell
$v2Tests = Get-ChildItem -LiteralPath 'tests' -Filter 'test_cognition_core_v2*.py' | Select-Object -ExpandProperty FullName
venv\Scripts\python.exe -m pytest $v2Tests -m "not live_llm and not live_db" -q
```

### Static and prompt checks

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\cognition_core_v2\prompt_budget.py src\kazusa_ai_chatbot\cognition_core_v2\semantic_appraisal.py src\kazusa_ai_chatbot\cognition_core_v2\facade.py src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py src\kazusa_ai_chatbot\cognition_core_v2\workspace.py src\kazusa_ai_chatbot\cognition_core_v2\action_selection.py src\kazusa_ai_chatbot\cognition_core_v2\action_authorization.py src\kazusa_ai_chatbot\cognition_core_v2\resolver_authorization.py src\kazusa_ai_chatbot\cognition_core_v2\surface_stages.py tests\test_cognition_core_v2_prompt_budget_continuity.py tests\test_cognition_core_v2_action_planning_bugfix.py experiments\focused_context_limit_625857543_live_llm.py
git diff --check
```

The focused contract test and live artifact validator inspect JSON field
locations directly: prompt-visible `state.evidence` and expanded
`permitted_delta_paths` must be absent, while the private deterministic
question retains the exact canonical path set.

### One-case real LLM verification

```powershell
venv\Scripts\python.exe experiments\focused_context_limit_625857543_live_llm.py
```

Run once after deterministic gates. Required result:

- database `asuna_core_v2`, read-only;
- QQ user `257629823`, profile, memory, turn, and history loaded from production;
- pre-failure cognition state loaded from the hash-identified production replay
  artifact and its owner verified against the target global user;
- seven focused RAG rows plus the current episode;
- one call to
  `COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY`;
- rendered human payload at or below 8,000 characters;
- parseable, structurally validated model output or a model-owned bounded
  contract exhaustion after the model was called;
- no `CognitionContextLimitError`;
- no database mutation path and no production delivery path invoked.

The parent inspects the raw artifact and authors
`test_artifacts/diagnostics/failure_625857543_postfix_boundary_live_review.md`
under the `debug-llm` contract.

## Independent Code Review

After all verification passes, start one independent review subagent. Provide
this plan, the full diff, red/green output, regression output, prompt-size
evidence, and live raw/review artifacts.

Review scope:

- exact aggregate budget ownership and deterministic cap arithmetic;
- evidence row order, handle authority, single-registry invariant, truncation
  floor, and no raw/private leakage;
- compact-domain equivalence to private exact path validation;
- no prompt-visible compatibility field or alias;
- required goal semantics remain LLM-owned;
- every pre-call cap maps to its established omission, fallback, deny, degraded
  text, or visual omission;
- no cap increases, added calls, service/adapter changes, database writes,
  global RAG suppression, or unrelated refactors;
- Python/CJK style, prompt rendering, test realism, and live evidence quality;
- preservation of pre-existing unrelated worktree changes.

The parent fixes findings only inside the approved change surface and reruns
affected verification. A finding requiring a new contract or outside file
stops execution for plan update or user authority.

## Acceptance Criteria

This plan is complete when:

- the production-derived focused case no longer raises before the epistemic
  model call;
- its rendered appraisal payload is at most 8,000 characters;
- appraisal evidence text is serialized once;
- prompt-visible delta authority is compact and exact private validation is
  unchanged;
- evidence outside aggregate budget is deterministically truncated in source
  priority order;
- maximum-shape appraisal and goal packets remain under their existing caps;
- every audited degradable pre-invocation cap follows its owner fallback;
- action and resolver overflow authorize no work;
- text and visual cap overflow never reach the operational-error response;
- normal-path call counts, routes, schemas, persistence, and adapter behavior
  remain unchanged;
- focused, affected, full V2, static, prompt, one-case live, and independent
  review gates pass;
- execution evidence is recorded and the plan is archived completed.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Compact domains confuse the local model | Explicit positive composition rule and unchanged canonical output validation | one-case live appraisal and contract tests |
| Truncation removes decisive context | Preserve source order, current episode, handles, head/tail, and non-empty floor | production fixture and truncation-order tests |
| Goal prompt still exceeds cap | Remove duplicate evidence, trim supplemental state, then apply aggregate evidence fitting | 32-row maximum test |
| Fallback authorizes work | Empty/deny-only control dispositions | no-call and empty-result assertions |
| Surface cap becomes silent | Existing validated degraded text remains visible; only visual is optional | public surface tests |
| Change broadens beyond V2 | Exact production allowlist and independent review | changed-path audit and `git diff --check` |

## Execution Evidence

- User explicitly requested implementation of the root-cause fix and the
  adjacent user-visible error-spill audit on 2026-07-28.
- Pre-implementation production evidence:
  `test_artifacts/diagnostics/failure_625857543_root_cause_analysis.md`.
- Pre-implementation focused raw evidence:
  `test_artifacts/diagnostics/failure_625857543_focused_boundary_live_llm.json`.
- Stage 1 red command:
  `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_prompt_budget_continuity.py -q`.
- Stage 1 red result: 13 failed and 5 passed. The five passing controls were
  the non-epistemic appraisal families; all expected fix contracts were red.
- Stage 2 focused green result: 18 passed in 0.90 seconds. Maximum goal
  evidence fit at 24,000 characters, and all audited skip, stable fallback,
  empty-plan, deny-all, degraded-text, and typed-visual outcomes made zero
  model calls at their initial over-cap boundary.
- Production-code subagent: Copernicus
  (`019fa789-b764-76d0-8496-c3c34729f946`), closed after editing only the
  approved production surface.
- Independent reviewer: Maxwell
  (`019fa79d-3328-7e03-9400-544baa72b567`), closed after read-only review.
  The initial verdict withheld approval for five findings:
  - high: goal cognition still duplicated role summaries/projection data and
    could exceed 24,000 characters at a complete valid maximum;
  - medium: workspace, action, authorization, appraisal, and surface repairs
    lacked aggregate next-attempt budgets;
  - medium: malformed appraisal projection was mislabeled as degradable context
    exhaustion;
  - medium: maximum and exact-boundary tests were not sufficiently complete;
  - low: the live harness overstated direct cognition-state DB provenance and
    runtime write/send counters.
- Parent remediation established nine expected focused failures against 19
  passing controls, then reached 28 focused passes in 0.96 seconds. The final
  complete valid goal packet is exactly 24,000 characters. All six complete
  appraisal-family maxima, exact 8,000/8,001 appraisal boundaries, exact
  24,000/24,001 goal boundaries, malformed projection invariants, and initial
  plus repair owner dispositions pass.
- Final affected deterministic result: 194 passed and 4 deselected in 2.44
  seconds.
- Final full regular V2 result: 314 passed and 150 deselected in 5.67 seconds.
- Final shared prompt-contract result: 11 passed in 0.80 seconds.
- Static result: every changed Python source, test, and focused harness
  compiled; fixture and live-artifact JSON parsed; `git diff --check` passed
  with line-ending notices only.
- Changed-path result: exactly ten approved Cognition Core V2 production paths,
  zero unexpected runtime paths, and zero service/adapter paths.
- Post-fix production-data raw artifact:
  `test_artifacts/diagnostics/failure_625857543_postfix_boundary_live_llm.json`.
- Post-fix human LLM review:
  `test_artifacts/diagnostics/failure_625857543_postfix_boundary_live_review.md`.
- Live result: exact QQ identity/profile/relationship memory, failed turn, and
  focused history rows loaded read-only from `asuna_core_v2`; the pre-failure
  cognition state came from
  `failure_625857543_production_readonly_live_reproduction.json`, SHA-256
  `13fb81bee2730ee48f4c96e8d582d8826c2fe6a9d7a6b1ee99962c352b1bd590`,
  with its owner verified against the target global user. One 15,230 ms target
  call used a 6,642-character packet and returned 2,784 characters.
  Deterministic validation rejected a delta against resolved `k3`; the direct
  one-attempt harness recorded typed contract exhaustion and zero retries. Its
  code path invoked neither a production database mutation API nor a production
  delivery API.
- Residual risks: the local model can still propose invalid terminal-state
  transitions, contained by bounded regeneration and per-appraisal omission;
  irreducible required goal/addressing/state/commit failures remain typed by
  design; the separately observed relationship-memory RAG routing drift remains
  deferred because it has a different owner and failure contract.
- Completed plan archive:
  `development_plans/archive/completed/bugfix/cognition_core_v2_prompt_budget_and_failure_containment_bugfix_plan.md`.
- Initial worktree contains unrelated plan-registry and Stage 4 lifecycle
  changes; these remain preserved.
