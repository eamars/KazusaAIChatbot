# cognition core v2 retry exhaustion continuity bugfix plan

## Summary

- Goal: make every recoverable or degradable model failure inside the live
  Cognition Core V2 chain finish as a normal response, while reserving the
  adapter-visible operational notice for an unrecoverable pipeline failure.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `llm-trace-debug`, `debug-llm`, `py-style`, `cjk-safety`,
  `test-style-and-execution`
- Overall cutover strategy: bigbang inside the approved V2 response path;
  baseline-preserving outside that path.
- Baseline: branch `cognition_core_v2`; implementation base SHA
  `5e5e23a2424153059f5906e25ba37b0bc8095a44`. The planning SHA was
  `87cd6df5a869dc78b6e1ed48dd1d580db9d5f6da`; the intervening control-console
  completion commit did not touch this plan's V2 production allowlist.
- Highest-risk areas: retry amplification, accepting malformed structured
  control output, weakening character judgment, losing post-commit dialog,
  and accidental changes outside the accepted V2 radius.
- Acceptance criteria: every V2-path model producer and verifier has a bounded
  outcome; recoverable failures retry; degradable failures return normal
  character output or a typed no-op; only unrecoverable failures reach the
  existing operational-error response; the implementation production diff is
  a subset of this plan's V2 allowlist.

## Context

Two production turns in the same QQ channel reached
`dialog_compliance_contract_exhausted`:

| Correlation id | Trace id | Duration | Rejected check |
|---|---|---:|---|
| `chat:qq:ch_732699d6699040ae:921358597` | `llmtrace_57ad37855a714d0798a8d2a2960a00d6` | 140,063 ms | role direction |
| `chat:qq:ch_732699d6699040ae:1573849126` | `llmtrace_b18ae4eea17b4c6a8b8399bc0f3bd33d` | 71,656 ms | role direction |

Both turns had two non-empty, structurally valid dialog candidates.
Semantic-fidelity and surface-integrity verification passed. Role direction
mistook an information request or character-commanded user action for transfer
of the character's required decision. `dialog_agent.py` then raised after the
second candidate, and the brain service replaced the available character text
with an operational notice after cognition state had committed.

The required ownership boundary is:

```text
brain-service admitted turn
  -> V2-related decontextualization
  -> Cognition Core V2 producers and deterministic reduction
  -> V2 action proposal/authorization
  -> V2 text and optional visual surface
  -> V2-related dialog generation and focused verification
  -> existing normal service persistence and ChatResponse
  -> unchanged adapter delivery
```

This plan defines “all model retries” as every model producer, semantic check,
and verifier reachable inside that boundary, including a currently single-shot
producer when its failure can be recovered locally. It excludes upstream
relevance settlement, provider implementation, generic JSON repair, RAG worker
internals, standalone resolvers, consolidation, reflection, residue,
background work, self-cognition scheduling, calendar work, coding agents, the
brain-service generic error classifier, and adapters.

The accepted V2 architectural baseline is recorded by:

- `development_plans/archive/completed/short_term/cognition_core_v2_stage_3_change_radius.md`
- `development_plans/archive/completed/bugfix/cognition_core_v2_baseline_regression_hardening_plan.md`
- `development_plans/archive/completed/bugfix/cognition_core_v2_intra_turn_transition_coherence_bugfix_plan.md`

The Stage 3 radius identifies `cognition_core_v2`, its persona connector,
surface, and dialog as V2 owners while keeping adapters, provider internals,
RAG worker packages, and coding agents unchanged. This plan uses that boundary
and narrows it further to files required for retry and degraded delivery.

## Mandatory Skills

- `development-plan`: approval, execution, review, lifecycle, and sign-off.
- `local-llm-architecture`: V2 model ownership, retry, prompt, and latency.
- `llm-trace-debug`: protected evidence for the two incidents.
- `debug-llm`: one-at-a-time real-model tests and readable review artifacts.
- `py-style`: every Python production or test edit.
- `cjk-safety`: Python prompt and fixture edits containing Chinese text.
- `test-style-and-execution`: every test edit and test command.

## Mandatory Rules

- Begin production implementation only after explicit approval and status
  change to `approved`.
- Use `venv\Scripts\python.exe`; use `apply_patch` for manual edits; preserve
  unrelated worktree changes; keep `.env` outside inspection.
- Capture `implementation_base_sha`, exact `git status --short`, and SHA-256
  fingerprints for every pre-existing dirty file before the first test edit.
- Require every production path changed after that capture to appear under
  `Change Surface / Modify` or `Create`. Preserve each pre-existing dirty file
  outside this plan byte-for-byte relative to its captured fingerprint.
- Keep `src/adapters/**`, `src/kazusa_ai_chatbot/service.py`,
  `src/kazusa_ai_chatbot/brain_service/**`, provider internals, generic JSON
  repair, relevance, RAG, resolver packages, background systems, persistence,
  and coding agents at baseline.
- Pass every JSON-producing response through
  `kazusa_ai_chatbot.utils.parse_llm_json_output(...)`.
- Keep retry selection, attempt caps, structural validation, state safety,
  permissions, execution, persistence, and delivery eligibility deterministic.
  Keep semantic repair and visible wording with the producing model owner.
- Use one data-only V2 attempt policy. Keep each model call beside its local
  prompt, payload, parser, validator, and trace capture; do not create a generic
  invocation wrapper.
- Use three total producer attempts as the V2 failure-path minimum. Existing
  paths already using three or more total semantic candidates retain their cap.
- Keep healthy-path call counts unchanged. Attempt two or three runs only
  after a provider, parse, structural, or typed semantic failure.
- Keep repair input bounded to canonical stage context, the latest bounded
  invalid candidate when structurally useful, and exact typed feedback. Do not
  accumulate an attempt conversation.
- Treat visible bounded text and structured control differently. A
  structurally valid dialog may finish as `accepted_degraded`; malformed
  action, resolver, authorization, state, permission, or persistence output
  never enters its downstream effect path.
- Preserve the newest structurally valid dialog candidate. Verifier failure or
  disagreement cannot erase it.
- Keep role-direction decisions typed. Prose supplies bounded model feedback
  only; deterministic code does not classify issue prose with substrings,
  keywords, or regular expressions.
- Let recoverable failures retry and degradable failures finish normally.
  Allow the existing operational-error path only when the V2 owner proves that
  no validated state/control fallback and no bounded visible candidate exist.
- Treat corrupt canonical input/state, failed required state commit, internal
  invariant violation, and total model unavailability with no owned fallback
  as unrecoverable. These are the only V2 outcomes allowed to escape to the
  generic service failure boundary.
- Keep degraded dialog on the existing normal assistant persistence, delivery
  tracking, episode settlement, and adapter path with
  `operational_error=None`.
- Keep raw candidates, prompts, verifier prose, and provider errors in
  protected trace storage. Existing protected/internal diagnostic slots receive
  only bounded stage, count, failure-kind, selected-attempt, and disposition
  fields; public response shapes remain unchanged.
- Preserve the 50k-token planning cap and every smaller stage character cap.
- Reread this entire plan after context compaction and after signing off each
  major checklist stage.
- Run the `Independent Code Review` gate and record it before completion,
  lifecycle movement, merge, or sign-off.

## Must Do

- Freeze and record the implementation baseline before edits.
- Add one data-only V2 attempt policy and an exact V2-path owner registry.
- Cover every V2-path model call, including image description,
  decontextualization, appraisal, goal generation and selection, workspace
  collapse, action/resolver authorization, text/visual surface, dialog
  generation, and all three focused dialog verifiers.
- Raise one- and two-attempt recoverable producer paths to three total
  attempts; retain bounded paths already at three or more.
- Preserve stage-owned degraded values after attempt exhaustion.
- Correct role direction so explicit decision transfer and exact typed
  actor/target reversal remain violations, while information requests,
  reports, standby instructions, and character-selected user actions remain
  aligned.
- Add a third dialog candidate and deliver it after semantic exhaustion when
  it is bounded and non-empty. When attempt three yields no usable text,
  deliver candidate two, then candidate one.
- Convert exhausted focused verifiers to typed `unavailable` results so they
  cannot delete a valid producer candidate.
- Ensure V2 surface failure still yields a validated degraded text surface
  from already validated cognition truth; optional visual failure skips visual
  output while text continues.
- Keep invalid action/resolver plans empty or blocked and invalid
  authorization denied.
- Prove through service integration tests that recovered and degraded V2
  outcomes produce normal `ChatResponse` text or the existing semantic no-op,
  never an operational notice.
- Prove through one unrecoverable injected invariant case that the existing
  service operational response remains reachable as the last resort.
- Run a final baseline-radius audit and fail sign-off on every production path
  outside the allowlist.

## Deferred

- Keep the repository-wide retry policy outside this plan.
- Keep provider unload replay, model routing, failover, circuit breaking,
  backoff, and model generation settings unchanged.
- Keep relevance settlement, RAG and resolver worker loops, consolidation,
  reflection, residue, background jobs, self-cognition, calendar, and coding
  retry behavior unchanged.
- Keep `ChatResponse`, brain-service error contracts, adapter behavior,
  database schemas, indexes, and persistence timing unchanged.
- Keep legacy cognition-chain behavior unchanged.
- Keep semantic keyword filters, deterministic dialog rewriting, compatibility
  aliases, feature flags, and parallel fallback paths outside this plan.

## Cutover Policy

Overall strategy: bigbang inside the V2 response path and baseline-preserving
outside it.

| Area | Policy | Instruction |
|---|---|---|
| V2 attempt policy | bigbang | Use one three-attempt minimum for every recoverable V2 producer and verifier. |
| Existing V2 loops at three or more | compatible | Preserve their existing cap and change only terminal disposition where listed. |
| Dialog semantic exhaustion | bigbang | Deliver the newest bounded candidate as `accepted_degraded`; end the normal path without raising compliance exhaustion. |
| Structured control exhaustion | compatible safety | Preserve empty, denied, blocked, skipped, or required-invariant behavior. |
| V2 text-surface exhaustion | bigbang | Project one validated degraded surface from canonical V2 truth and continue to dialog. |
| Optional visual exhaustion | bigbang | Skip visual output and preserve text response. |
| Service and adapter | baseline-preserving | Preserve code and public contracts; V2 prevents recoverable/degraded failures from reaching them. |
| Outside V2 | baseline-preserving | Preserve every retry cap and terminal behavior outside the allowlist. |
| Tests/docs | bigbang | Replace V2 fail-closed exhaustion expectations with the recovered/degraded/unrecoverable matrix in the same change. |

## Cutover Policy Enforcement

- Apply all V2 policy changes in one reviewed implementation.
- Add no compatibility flag, alias code, dual retry path, or global helper.
- Preserve only the compatibility surfaces explicitly marked above.
- Require user approval for any policy or change-surface expansion.

## Target State

### Outcome ladder

```text
1. accepted
   valid first candidate -> normal pipeline

2. recovered
   bounded retry produces a valid candidate -> normal pipeline

3. accepted_degraded
   attempts exhaust, but the owner has validated prior semantic state,
   a safe typed fallback, or bounded visible text -> normal pipeline

4. unrecoverable
   canonical state/invariant/commit is invalid, or every model attempt fails
   with no owned state/control/text fallback -> existing operational response
```

`recovered` and `accepted_degraded` never produce
`content_type="operational_error"`. An adapter-visible operational notice is
evidence that the request reached outcome four.

### V2 owner matrix

| V2 owner | Current total opportunity | Planned total opportunity | Exhausted disposition |
|---|---:|---:|---|
| image descriptor | 1 | 3 | existing typed unavailable description; continue |
| message decontextualizer | 2 | 3 | original normalized input; omit uncertain role projection |
| semantic appraisal | 1 | 3 | omit that appraisal and retain existing warning path |
| goal-bid structure | 2 | 3 | drop optional invalid branch; required zero-valid branch requests the existing one clean graph retry, then becomes unrecoverable |
| required-selection alignment | initial + 2 replacements; single-shot check | same semantic candidate ledger; up to 3 attempts per provider/contract-failed check | newest structurally valid bid is `accepted_degraded` |
| workspace collapse | 1 | 3 | choose the highest-priority already valid complete bid by existing branch order |
| action planning | 2 | 3 | empty requests with blocked/no-work semantics |
| action and resolver authorization | 2 | 3 | deny all candidates |
| content/preference surface stages | 2 | 3 | validated degraded surface projected from selected V2 intention |
| dialog-compliance surface replacement | 2 | 3 | retain the latest valid surface |
| optional visual surface | 2 | 3 | omit visual directives |
| dialog generator structure/provider | 1 candidate call at each semantic round | 3 bounded producer opportunities | newest structurally valid dialog, otherwise unrecoverable |
| semantic/role/surface verifier structure | 2 per verifier | 3 per verifier | verifier status `unavailable`; candidate remains eligible |
| dialog semantic candidates | 2 | 3 | candidate three, else candidate two, else candidate one |

The workspace fallback selects among complete model-authored bids and invents
no bid content. The text-surface fallback copies the validated selected
intention, action truth, and runtime limits; it supplies only neutral delivery
instructions and empty optional preference lists.

### Dialog sequence

```text
candidate 1
  -> aligned: deliver normally
  -> negative/unavailable check: retain candidate and repair

candidate 2
  -> aligned: deliver normally
  -> negative/unavailable check: retain candidate and render terminal attempt

candidate 3
  -> bounded non-empty text: deliver as accepted_degraded
  -> no usable text: deliver candidate 2, else candidate 1
  -> no candidate at any attempt: unrecoverable
```

The third candidate receives canonical V2 surface truth and typed remaining
violation kinds. It receives no rejected dialog transcript and no accumulated
attempt conversation. Candidate three is terminal and receives no verifier
calls; the three generator opportunities are one shared ledger rather than a
nested three-attempt loop per semantic candidate.

### Fatal boundary

The V2 path may raise to the brain service only for:

- invalid canonical episode or validated persistent state;
- a required cognition branch with zero structurally valid candidate and no
  valid alternative bid after its local cap and the existing safe clean-graph
  retry;
- failed state commit or invalid post-commit invariant;
- total provider/model unavailability across the bounded cap when no earlier
  valid state, surface, or dialog candidate exists; or
- an unexpected internal exception outside the classified model-failure
  taxonomy.

Provider, parse, structure, semantic-verifier, and optional-stage exhaustion
with an owned fallback finish inside V2.

### Normal degraded service outcome

```python
ChatResponse(
    messages=["<bounded character dialog>"],
    content_type="text",
    delivery_tracking_id="<normal tracking id>",
    operational_error=None,
)
```

The assistant row, logical message index, trace id, delivery tracking, episode
settlement, and adapter rendering remain the existing normal behavior.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Scope | V2 core plus its direct decontextualizer, surface, persona, and dialog owners | Satisfies continuity without changing shared systems outside the accepted V2 radius. |
| Service handling | Keep `service.py` unchanged | Recovery belongs to the producing V2 owner; the generic service remains the unrecoverable boundary. |
| Attempt minimum | Three total producer attempts | Adds one failure-only opportunity with unchanged healthy latency. |
| Existing longer loops | Preserve their cap | Prevents nested amplification. |
| Retry implementation | Shared data policy, local stage loops | Keeps prompts, parsing, validation, and semantic repair visible at each owner. |
| Dialog fallback | Deliver newest bounded text after retry exhaustion | A semantic disagreement is less severe than deleting available character speech. |
| Verifier exhaustion | Mark unavailable | A verifier contract failure cannot invalidate producer structure. |
| Role semantics | Reject explicit owner transfer or exact operation reversal only | Preserves natural information requests and character-directed user actions. |
| Surface fallback | Project from validated V2 intention and action truth | Continues dialog without inventing state or authorizing work. |
| Control fallback | Empty/deny/blocked | Preserves permissions and side-effect safety. |
| Operational response | Unrecoverable only | Makes adapter-visible failure an invariant/availability alarm. |

## Contracts And Data Shapes

Create a data-only V2 module:

```python
V2_MODEL_TOTAL_ATTEMPTS = 3
V2_VERIFIER_TOTAL_ATTEMPTS = 3

V2AttemptFailureKind = Literal[
    "provider",
    "parse",
    "structure",
    "semantic",
    "verifier_unavailable",
]

V2AttemptDisposition = Literal[
    "accepted",
    "recovered",
    "accepted_degraded",
    "retry_graph",
    "empty",
    "denied",
    "skipped",
    "unrecoverable",
]

class V2AttemptRecord(TypedDict):
    stage: str
    failure_kind: V2AttemptFailureKind | None
    attempt_count: int
    total_attempt_limit: int
    selected_attempt: int | None
    disposition: V2AttemptDisposition
    safe_checkpoint: Literal["pre_state_commit", "post_cognition_commit"]
```

The record contains no prompt, user text, candidate text, verifier prose,
provider detail, identifier, credential, action argument, or persistent row.

The degraded surface projection has one exact interface:
`build_degraded_text_surface(input_payload: TextSurfaceInputV2) ->
TextSurfaceOutputV2`.

It validates input, copies `intention.intention` into `content_plan`, supplies
one neutral content requirement, uses empty `visible_boundaries` and
`addressee_plan`, supplies a bounded neutral five-field delivery profile, and
copies selected intent, permitted action results, and runtime limits. It then
passes through `validate_text_surface_output(...)`.

The dialog verifier aggregate preserves each owner:

```python
{
    "semantic_fidelity": {
        "status": "aligned" | "misaligned" | "unavailable",
        "issues": list[str],
    },
    "role_direction": {
        "status": "aligned" | "misaligned" | "unavailable",
        "violations": list[{
            "kind": (
                "selection_owner_transfer"
                | "typed_operation_role_reversal"
            ),
            "evidence": str,
            "explanation": str,
        }],
    },
    "surface_integrity": {
        "status": "aligned" | "misaligned" | "unavailable",
        "issues": list[dict[str, str]],
    },
}
```

No `ChatResponse`, adapter, database, action-spec, resolver packet, or
`CognitionCoreOutputV2` public field changes.

## LLM Call And Context Budget

The 50k-token planning cap and existing smaller character caps remain fixed.

| Path | Before | After | Healthy path | Failure path |
|---|---:|---:|---|---|
| image descriptor | 1 | up to 3 | unchanged | two failure-only calls |
| decontextualizer | up to 2 | up to 3 | unchanged | one bounded repair |
| appraisal/workspace | 1 | up to 3 | unchanged | two bounded repairs |
| goal structure | up to 2 | up to 3 | unchanged | one bounded repair |
| required-selection alignment | initial + up to 2 replacements; single-shot check | same semantic candidate ledger; failed check up to 3 | unchanged | two failure-only verifier calls per failed check; degraded terminal disposition |
| action/authorization | up to 2 | up to 3 | unchanged | one bounded repair |
| each surface producer | up to 2 | up to 3 | unchanged | one bounded repair |
| each focused verifier | up to 2 | up to 3 | unchanged | one structure repair |
| dialog semantic candidates | 2 | 3 | unchanged | one terminal renderer |
| typical dialog semantic rejection | about 9 calls | about 10 calls | unchanged | one terminal generator call |
| worst malformed dialog/verifiers | about 16 calls | hard cap about 24 | unchanged | rare structure-repair expansion |
| service clean graph retry | one retry after typed safe pre-commit failure | unchanged | unchanged | remains the last recovery layer when a required V2 stage has no local fallback |

The existing provider-internal unload replay remains unchanged and may occur
inside one stage attempt. Record logical stage attempts and underlying provider
call counts separately. Attempt prompts use canonical context plus latest
bounded feedback and stay within existing caps.

## Change Surface

Target ownership boundary: Cognition Core V2 and its direct live connector,
surface, dialog, and tests.

### Create

- `src/kazusa_ai_chatbot/cognition_core_v2/model_attempt_policy.py`: data-only
  limits, failure kinds, dispositions, and bounded record validation.
- `tests/test_cognition_core_v2_model_retry_continuity.py`: owner registry,
  attempt limits, terminal matrix, and change-radius guard.
- `tests/fixtures/cognition_core_v2_retry_exhaustion_cases.json`: sanitized
  role-direction and forced-exhaustion cases.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/__init__.py`: public V2 policy
  exports.
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`: three
  attempts and typed omission after exhaustion.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: three
  structural attempts and last-valid semantic degradation.
- `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py`: three attempts and
  existing-order complete-bid fallback.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`: three
  attempts and existing empty/blocked fallback.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py`: three
  attempts for action and resolver authorization, then deny.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`: three attempts,
  last-valid retention, and typed failure disposition.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface.py`: validated degraded
  text-surface projection and visual skip.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`: carry bounded V2
  degraded warnings and preserve required-invariant classification.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`: document the outcome
  ladder and fatal boundary.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py`:
  image/decontext attempts and original-input fallback.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: continue
  with degraded text surface and omit failed optional visual output.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: corrected typed role checks,
  candidate ledger, third candidate, verifier-unavailable handling, and
  degraded selection.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: retain degraded
  surface/dialog on the normal episode and delivery path.
- `src/kazusa_ai_chatbot/nodes/README.md`: document V2 continuity behavior.
- `tests/test_cognition_core_v2_failures.py`
- `tests/test_cognition_core_v2_contracts.py`
- `tests/test_cognition_core_v2_alignment_gates.py`
- `tests/test_cognition_core_v2_action_authorization.py`
- `tests/test_cognition_core_v2_action_planning_bugfix.py`
- `tests/test_cognition_core_v2_dependencies.py`
- `tests/test_cognition_core_v2_integration.py`
- `tests/test_cognition_core_v2_transition_coherence.py`
- `tests/test_msg_decontextualizer.py`
- `tests/test_dialog_agent.py`
- `tests/test_dialog_visible_speech_and_semantic_fidelity.py`
- `tests/test_dialog_visible_speech_and_semantic_fidelity_live_llm.py`
- `tests/test_cognition_core_v2_surface_owner_live_llm.py`
- `tests/test_service_cognition_graph.py`
- `tests/test_service_input_queue.py`
- `development_plans/README.md`
- this plan

### Delete

- The previous unapproved repository-wide draft filename is replaced by this
  V2-scoped plan.

### Keep

- `src/kazusa_ai_chatbot/service.py` and
  `src/kazusa_ai_chatbot/brain_service/**`
- `src/adapters/**`
- `src/kazusa_ai_chatbot/llm_interface/**` and
  `src/kazusa_ai_chatbot/utils.py`
- `src/kazusa_ai_chatbot/relevance/**`
- `src/kazusa_ai_chatbot/rag/**`,
  `src/kazusa_ai_chatbot/local_context_resolver/**`, and
  `src/kazusa_ai_chatbot/complex_task_resolver/**`
- persistence, consolidation, reflection, residue, background, self-cognition,
  calendar, coding-agent, control-console, and legacy cognition production code
- model routes, generation settings, public APIs, database state, and adapter
  delivery contracts
- `tests/fixtures/cognition_llm_producer_matrix.json`

## Overdesign Guardrail

- Actual problem: recoverable/degradable V2 model exhaustion discards usable
  character output and reaches the adapter's fatal surface.
- Minimal change: one V2-local data policy, one extra failure-only attempt for
  short V2 producers, stage-owned safe fallbacks, and newest-dialog delivery.
- Ownership boundaries: V2 models own semantic repair; dialog owns wording;
  deterministic V2 code owns attempt limits, validation, fallback selection,
  action denial, and normal-path eligibility; service/adapters retain existing
  delivery ownership.
- Rejected complexity: repository-wide policy, shared invocation wrapper,
  provider changes, dynamic retry, failover, feature flags, compatibility
  aliases, database state, adapter changes, and new background behavior.
- Evidence threshold: a wider policy requires a separately approved plan with
  failures from a named non-V2 owner and its own baseline radius.

## Agent Autonomy Boundaries

- Preserve the exact owner matrix, fatal definition, and production allowlist.
- Choose local helper names only inside listed files and only when they
  implement an explicit contract in this plan.
- Search for equivalent V2-local behavior before adding a helper; keep prompts,
  model instances, payloads, parsers, and validators visible at their owner.
- Treat every new production path outside the allowlist as a blocker requiring
  plan amendment and user approval.
- Preserve pre-existing dirty files outside this plan at captured fingerprints.
- Keep unrelated cleanup, formatting, dependency, prompt, and schema work out
  of execution.
- Preserve this plan's normal/degraded/unrecoverable distinction when source
  and plan wording conflict; record the discrepancy before proceeding.
- Stop with a precise blocker when a mandatory instruction cannot be met.

## Implementation Order

1. Capture base SHA, dirty status, fingerprints, V2 model call inventory,
   current call counts, and current failing service/dialog behavior.
2. Add focused failing tests for the attempt policy, every owner disposition,
   the two role cases, candidate-three delivery, surface fallback, visual skip,
   and `operational_error=None`.
3. Create the V2-local data policy and owner registry; verify that it contains
   every scoped model call and no non-V2 owner.
4. Update decontextualization, appraisal, goal, workspace, action, and
   authorization loops and fallbacks.
5. Update surface attempts, degraded surface projection, and optional visual
   containment.
6. Update typed dialog verification, candidate retention, third rendering, and
   normal persona-graph handoff.
7. Run focused deterministic tests and correct only plan-scoped failures.
8. Run the injected recovered/degraded/unrecoverable service matrix.
9. Run real-model cases one at a time and inspect each trace before continuing.
10. Run affected regression, prompt render, call-budget, static radius, and
    pre-existing fingerprint gates.
11. Run independent code review, remediate scoped findings, and rerun affected
    verification.
12. Record evidence, sign off acceptance, update lifecycle status, and archive.

## Execution Model

- Parent agent owns test contracts, orchestration, integration, verification,
  evidence, review remediation, lifecycle updates, and final sign-off.
- Parent establishes and records focused failing tests before implementation.
- Production-code subagent: exactly one native subagent, started after the test
  contract; owns listed production files only and closes after implementation.
- Parent may continue integration and static verification while the production
  subagent works.
- Independent code-review subagent: exactly one native subagent after planned
  verification; reviews the plan, full diff, baseline evidence, and tests and
  implements no fixes.
- Native subagent unavailability pauses execution until the user explicitly
  approves a fallback execution model.

## Progress Checklist

- [x] Stage 1 — baseline and failing contracts frozen.
  - Covers: steps 1-2.
  - Verify: recorded SHA/status/fingerprints, owner inventory, and red tests.
  - Evidence: baseline manifest and exact expected failures.
  - Handoff: start Stage 2 at the V2 data policy.
  - Sign-off: parent / 2026-07-27 after full plan reread.
- [x] Stage 2 — V2 core retry and safe control dispositions complete.
  - Covers: steps 3-4.
  - Verify: policy, appraisal, goal, workspace, action, and authorization tests.
  - Evidence: exact 17-owner registry; three-attempt producer constants;
    appraisal omission, required-goal graph retry, newest valid selection bid,
    branch-order workspace fallback, empty action plan, and deny-all
    authorization assertions; 89 deterministic tests passed.
  - Handoff: start Stage 3 at V2 surface.
  - Sign-off: parent / 2026-07-27 after full plan reread.
- [x] Stage 3 — surface, dialog, and normal delivery continuity complete.
  - Covers: steps 5-8.
  - Verify: surface/dialog/service focused tests and operational-error spy.
  - Evidence: 145 focused deterministic tests passed with 4 intentional
    marker deselections; the injected degraded dialog returned tracked text,
    persisted the assistant row, settled lifecycle/continuity, and never
    called the operational response builder; zero usable dialog candidates
    retained the existing operational response.
  - Handoff: start Stage 4 real-model and regression verification.
  - Sign-off: parent / 2026-07-27 after full plan reread.
- [x] Stage 4 — real-model, regression, and radius gates pass.
  - Covers: steps 9-10.
  - Verify: one-at-a-time live cases, affected batches, prompt/call/radius gates.
  - Evidence: four one-at-a-time real-model cases passed with trace inspection
    and a readable review artifact; focused verification produced 184 passed
    and 4 intentional marker deselections; affected verification produced 143
    passed and 4 intentional marker deselections; the full regular V2 batch
    produced 239 passed and 150 intentional live/database deselections; prompt
    and call-budget checks produced 14 passed and 95 deselections. The final
    inventory maps 14 local model-call sites to the exact 17-owner registry;
    one additional `.ainvoke` is V2 subgraph execution. Static gates found 14
    approved production files, zero unexpected production files, zero
    service/adapter changes, 25 compilable and AST-valid changed Python files,
    and a clean `git diff --check`.
  - Handoff: start Stage 5 independent review.
  - Sign-off: parent / 2026-07-27 after full plan reread.
- [x] Stage 5 — independent review and lifecycle closeout complete.
  - Covers: steps 11-12.
  - Verify: findings resolved, affected checks rerun, acceptance signed.
  - Evidence: the single independent reviewer requested changes for two high,
    two medium, and one low finding. All findings were remediated inside the
    approved V2 surface: required-selection verifier retries, exact descriptor
    validation and cache eligibility, role-evidence grounding, appraisal
    invariant/cancellation propagation, and subsystem documentation. The
    remediation contracts produced 9 passed; the dependency ledger produced
    17 passed; the final focused batch produced 189 passed and 4 marker
    deselections; the affected batch produced 145 passed and 4 marker
    deselections; the service outcome matrix produced 3 passed; and the final
    regular V2 collection produced 244 passed and 150 intentional
    live/database deselections. Two affected real-model ownership cases passed
    again one at a time with fresh trace inspection. Final static scope was 15
    approved production paths, zero unexpected production paths, zero
    service/adapter changes, 26 compilable and AST-valid Python paths, a
    17-owner registry, and a clean `git diff --check`.
  - Handoff: archive completed record.
  - Sign-off: parent / 2026-07-27 after full plan reread.

## Verification

### Static scope and contract gates

- Record `git rev-parse HEAD` as `implementation_base_sha`.
- Record exact initial `git status --short` and SHA-256 fingerprints for every
  pre-existing dirty path.
- Run `git diff --name-only <implementation_base_sha> --` at sign-off.
  - Expected: every plan-owned production path is listed under `Change
    Surface`; pre-existing unrelated paths retain their captured fingerprints;
    no new adapter, service, brain-service, provider, relevance, RAG, resolver,
    persistence, background, reflection, coding, or console production path.
- Run:

```powershell
rg -n "\.ainvoke\(|\.invoke\(" src/kazusa_ai_chatbot/cognition_core_v2 src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py src/kazusa_ai_chatbot/nodes/dialog_agent.py
```

  - Expected: every model call maps exactly once to the V2 owner registry.
- Run `git diff --check`.
  - Expected: exit code `0`.
- Compile every changed Python file with `venv\Scripts\python.exe -m
  py_compile`.
  - Expected: exit code `0`.
- Render every changed prompt builder with canonical fixtures.
  - Expected: valid `.format(...)`, stable system prompt, bounded payload, and
    exact output contract.

### Focused deterministic tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_model_retry_continuity.py tests/test_cognition_core_v2_failures.py tests/test_cognition_core_v2_contracts.py tests/test_cognition_core_v2_alignment_gates.py tests/test_cognition_core_v2_action_authorization.py tests/test_cognition_core_v2_integration.py tests/test_cognition_core_v2_transition_coherence.py tests/test_msg_decontextualizer.py tests/test_dialog_agent.py tests/test_dialog_visible_speech_and_semantic_fidelity.py tests/test_service_cognition_graph.py -q
```

Required assertions:

- attempts one and two fail and attempt three succeeds as `recovered`;
- each exhausted degradable owner returns its exact typed fallback;
- required zero-valid structural state requests one safe clean-graph retry and
  becomes `unrecoverable` only after that retry exhausts;
- the two traced role patterns are aligned;
- explicit selection-owner transfer remains misaligned before terminal
  degradation;
- candidate three is delivered, with candidate two/one fallback ordering;
- malformed/unavailable verifiers preserve valid dialog;
- text-surface exhaustion produces a validated degraded surface;
- visual exhaustion omits visual output and preserves text;
- invalid action/resolver output authorizes no work;
- recovered and degraded service cases return
  `content_type="text"`, normal tracking, and `operational_error=None`;
- one injected invariant failure reaches the unchanged operational response;
- degraded dialog is persisted and settled through the normal path.

### Affected regression

```powershell
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_action_planning_bugfix.py tests/test_cognition_core_v2_action_authorization.py tests/test_cognition_core_v2_contracts.py tests/test_cognition_core_v2_integration.py tests/test_cognition_core_v2_transition_coherence.py tests/test_dialog_agent.py tests/test_dialog_visible_speech_and_semantic_fidelity.py tests/test_service_cognition_graph.py -m "not live_llm and not live_db" -q
```

### Real-model tests

Run one case, inspect its protected trace and readable output, then start the
next:

```powershell
venv\Scripts\python.exe -m pytest tests/test_dialog_visible_speech_and_semantic_fidelity_live_llm.py -m live_llm -k "information_request_preserves_selection_owner" -q -s
venv\Scripts\python.exe -m pytest tests/test_dialog_visible_speech_and_semantic_fidelity_live_llm.py -m live_llm -k "character_commanded_user_action_preserves_selection_owner" -q -s
venv\Scripts\python.exe -m pytest tests/test_dialog_visible_speech_and_semantic_fidelity_live_llm.py -m live_llm -k "terminal_candidate_is_deliverable_after_semantic_exhaustion" -q -s
venv\Scripts\python.exe -m pytest tests/test_cognition_core_v2_surface_owner_live_llm.py -m live_llm -k "degraded_surface_preserves_character_dialog" -q -s
```

Save a readable review under `test_artifacts/llm_debug/` with attempt counts,
candidate text, typed verifier outcomes, selected candidate, disposition, and
human judgment.

## Independent Code Review

Run after all verification passes and before lifecycle completion. The parent
creates one independent code-review subagent.

Review scope:

- final production diff is a strict subset of the V2 allowlist;
- pre-existing unrelated dirty paths retain captured fingerprints;
- every scoped model call has one attempt owner and terminal disposition;
- healthy-path call counts remain unchanged and nested calls remain bounded;
- recovered/degraded cases never reach the generic service error function;
- only the exact unrecoverable taxonomy can escape V2;
- dialog candidate retention, surface fallback, persistence, settlement, and
  tracking use the normal path;
- invalid structure never authorizes action, resolver, permission, scheduling,
  persistence, database, or adapter effects;
- typed role checks preserve natural questions and commanded user actions
  while detecting explicit owner transfer;
- raw prompts, candidates, verifier prose, identifiers, credentials, and
  provider errors remain outside public telemetry;
- no global helper, compatibility path, feature flag, provider change, service
  change, adapter change, or unrelated refactor appears.

The parent fixes findings only inside the approved surface, reruns affected
checks, and records findings and remediation in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- every scoped V2 model producer and verifier appears exactly once in the owner
  registry;
- one- and two-attempt recoverable V2 stages use three total attempts and
  existing longer paths retain their cap;
- every recoverable injected failure reaches a valid retry result;
- every degradable injected failure returns an exact owner fallback;
- the two traced role-direction patterns no longer exhaust dialog compliance;
- dialog semantic exhaustion delivers the newest bounded candidate;
- recovered and degraded V2 turns never return an operational error;
- only injected unrecoverable state/invariant/no-candidate cases reach the
  existing operational response;
- degraded visible output follows normal persistence, tracking, settlement,
  and adapter delivery;
- structured control safety remains validated-only;
- healthy-path behavior and call count remain baseline-equivalent;
- the final production path set is a subset of the approved V2 allowlist and
  every outside baseline fingerprint is unchanged;
- focused, regression, prompt, static, real-model, and independent-review gates
  pass;
- evidence is recorded and lifecycle status is completed.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Failure latency increases | Third attempts are failure-only and bounded | call-count and duration assertions |
| Nested provider replay multiplies calls | Provider behavior stays fixed; record logical and underlying calls separately | forced provider-failure counts |
| Degraded dialog has a semantic defect | Two semantic rounds precede terminal acceptance; newest candidate and typed diagnostics retained | incident and exhaustion review |
| Role check becomes permissive | Exact typed transfer/reversal violation kinds | positive/negative role matrix |
| Structured output bypasses safety | Control owners retain strict validation and empty/deny/blocked terminals | action/resolver integration tests |
| Neutral surface fallback weakens style | Copy selected intention and validated context; use fallback only after three failures | surface and live dialog review |
| Scope leaks outside V2 | Base SHA, dirty fingerprints, exact allowlist, static audit, independent review | final path/fingerprint gate |

## Execution Evidence

- Execution started 2026-07-27 after explicit user approval.
- Implementation base:
  `5e5e23a2424153059f5906e25ba37b0bc8095a44`.
- Pre-test-edit dirty manifest:
  `M development_plans/README.md`; `??` this plan.
- Pre-test-edit SHA-256:
  `development_plans/README.md` =
  `4943FCE49FC4A8FFC0B990049696700BD22D281F96C46A46635F8A65F032F900`;
  this plan =
  `AB62004D826CF2AE2CEEFF7570D092B6115C03F4456B5D0D4F2FB4E006EEC5F1`.
- Red contract: `test_cognition_core_v2_model_retry_continuity.py` compiled;
  its isolated run produced seven expected failures covering the missing policy,
  three-attempt limits, degraded surface, typed role verdict, unavailable
  verifier, and terminal third candidate.
- Stage 2 deterministic verification:
  `test_cognition_core_v2_failures.py`,
  `test_cognition_core_v2_alignment_gates.py`,
  `test_cognition_core_v2_action_authorization.py`,
  `test_cognition_core_v2_action_planning_bugfix.py`,
  `test_cognition_core_v2_dependencies.py`, and
  `test_cognition_core_v2_integration.py` produced 89 passed and 4 intentional
  marker deselections. The two additional test paths above complete the
  already-approved V2 action/goal verification surface and add no production
  scope.
- Stage 3 focused verification produced 145 passed and 4 intentional marker
  deselections. It covered three-candidate ordering, verifier structure and
  provider exhaustion, exact typed role violations, owner-preserving verifier
  aggregates, degraded text surface, optional visual omission, original-input
  decontextualization, and unexpected-invariant propagation.
- The deterministic service-path case in `test_service_input_queue.py` proved
  that a bounded degraded dialog returns `content_type="text"` with a normal
  delivery tracking id and `operational_error=None`, persists the assistant
  message, records bot continuity, and invokes lifecycle settlement without
  calling `_operational_error_response`. This test-only path completes the
  approved service integration requirement and adds no production scope.
- A zero-candidate `DialogGenerationContractError` remained service-recognized
  with attempt count 3 and produced the unchanged operational response,
  preserving the last-resort fatal boundary.
- Stage 4 deterministic verification:
  - the amended focused command produced 184 passed and 4 intentional marker
    deselections;
  - the amended affected command produced 143 passed and 4 intentional marker
    deselections;
  - the full regular `test_cognition_core_v2*.py` batch produced 239 passed and
    150 intentional live/database deselections;
  - prompt-render and call-budget checks produced 14 passed and 95
    deselections;
  - the explicit service recovered/degraded/unrecoverable matrix produced 3
    passed.
- Four real-model cases ran one at a time and passed after individual protected
  trace inspection: information-request ownership, character-commanded user
  action ownership, terminal-candidate delivery after two typed semantic
  rejections, and dialog generation from a degraded V2 text surface. The
  readable review is
  `test_artifacts/llm_debug/cognition_core_v2_retry_exhaustion_continuity_live_review.md`.
  It records candidate text, typed outcomes, selected attempts, dispositions,
  and human judgment, including the terminal candidate's explicit semantic
  degradation.
- The final model-call inventory contains 14 local call sites representing the
  exact 17 logical owners. Generic authorization represents action and resolver
  authorization; the generic surface stage represents content, preference,
  dialog-compliance repair, and visual owners; the two required-selection
  source sites share one candidate-ledger owner. The remaining `.ainvoke` is
  V2 subgraph execution rather than a model call.
- The final pre-review radius gate found 30 changed paths, including 14
  approved production paths and 25 Python paths. Every changed Python file
  passed `py_compile` and AST parsing, `git diff --check` passed, unexpected
  production paths were zero, and service/adapter production changes were
  zero.
- The required independent reviewer returned two high, two medium, and one low
  finding:
  - required-selection verifier provider/contract failures accepted a candidate
    after one failed check despite the three-attempt owner policy;
  - image descriptor structure was under-validated and exhausted fallback
    objects could enter cache;
  - role-direction evidence was not required to quote candidate text;
  - unexpected appraisal exceptions were reduced to optional appraisal
    warnings; and
  - subsystem READMEs retained obsolete one/two-attempt behavior.
- Parent remediation remained within approved V2 files:
  - required-selection verification now retries provider and contract failures
    up to three times per check and becomes unavailable only after exhaustion,
    retaining the newest structurally valid candidate;
  - image descriptors require the exact five fields and types, only validated
    results enter cache, and stale malformed cache rows trigger a fresh bounded
    attempt;
  - typed role-direction violation evidence must be an exact substring of the
    candidate dialog;
  - only the two typed semantic-appraisal exhaustion errors degrade; internal
    invariants and task cancellation propagate to the existing fatal boundary;
    and
  - both subsystem READMEs now describe the final attempt and degradation
    contracts.
- The remediation red contract produced 9 expected failures before production
  fixes. After remediation it produced 9 passed. A cancellation-specific
  follow-up produced 2 passed together with the internal-invariant case.
- Post-review deterministic verification:
  - focused V2/dialog/service verification produced 189 passed and 4
    intentional marker deselections;
  - the required-selection dependency ledger produced 17 passed;
  - affected regression produced 145 passed and 4 intentional marker
    deselections;
  - the explicit recovered/degraded/unrecoverable service matrix produced 3
    passed; and
  - the final regular V2 collection produced 244 passed and 150 intentional
    live/database deselections.
- The two affected real-model role cases ran again one at a time after evidence
  grounding. Each used one verifier call and returned `aligned=true` with no
  violations. Fresh traces were inspected and appended to the readable review
  artifact.
- Final radius and static evidence: 31 changed paths, 15 approved production
  paths, 26 Python paths, zero unexpected production paths, zero
  service/adapter production changes, an exact 17-owner registry, successful
  `py_compile` and AST parsing for every changed Python path, and clean
  `git diff --check`.
- Residual risks are bounded and accepted by the plan: failure-only attempts
  add latency; required-selection contract/provider failures may use up to
  three verifier calls per semantic candidate; a terminal third dialog can be
  semantically degraded after two verified repairs; and policy-to-call-site
  conformance is enforced by deterministic tests rather than a global runtime
  invocation wrapper. Healthy-path call counts and public response contracts
  remain unchanged.
- All acceptance criteria are satisfied. Lifecycle closeout moves this record
  to completed bugfix history.

## Approval Gate

The user approved this V2-only plan on 2026-07-27. Execution completed inside
the approved V2 change surface on 2026-07-27.
