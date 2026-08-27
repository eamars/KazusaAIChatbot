# Live Response Recovery Ladder And No-Fail Dialog Bugfix Plan

## Summary

- Goal: give every critical live-response model stage between intake and dialog
  generation one uniform recovery ladder, and make dialog generation incapable
  of terminating a committed turn without visible character text.
- Status: completed
- Scope boundary: frontline relevance, settled relevance, local-context
  resolution, Cognition V3 A1/A2/G/P, memory lifecycle, L3 text and visual
  surface, dialog generation, and the brain-service failure classification and
  checkpoint-replay boundary.
- Change direction: replace per-stage ad-hoc failure handling with one declared
  four-tier ladder (deterministic recovery, feedback-bearing regeneration,
  degradation to an available sibling or deterministic product, pre-commit
  checkpoint replay), and remove the dialog stage's failure exit.
- Acceptance state: parent independent plan-scope sign-off PASS with residual
  repository failures recorded.
- Execution authority: explicit user approval is recorded in the execution
  amendment below; this plan remains bounded by its stated scope.
- Supersedes as a change contract:
  `development_plans/active/bugfix/live_response_generation_failure_modes_problem_statement_2026-08-27.md`
  (problem statement only; retained as the incident record).

## Execution Amendment And Handoff — 2026-08-27

- Fixed executor: `/root/live_response_recovery_impl`, model `gpt-5.6-luna`,
  maximum reasoning, normal-speed operation. All implementation and test roles
  are serialized through this executor.
- Independent reviewer: parent `/root`; the parent owns review and sign-off
  and does not remediate implementation findings.
- User-approved executor waiver: the dialog and service implementation roles
  may share this fixed executor so the one-subagent constraint is honored.
- Execution baseline: clean worktree at
  `016061fe81ee2af31bfffbebf954c5ac51ec3ecf`.
- Local-context parser decision: all local-context stages reuse
  `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` and its existing JSON
  repair path unchanged. Structured output owns JSON structure; this plan adds
  recovery only for stage content and contract validity. This amendment does
  not authorize changes to `utils.py`, JSON-repair internals, or structured
  output transport.

## Terminal Deliverability Amendment And Handoff — 2026-08-27

- User-approved remediation scope reopens only the Delivery 1/2 boundaries:
  `cognition_shared/contracts.py`, `cognition_core_v3/facade.py`,
  `nodes/dialog_agent.py`, and their explicitly mapped deterministic tests.
  Ownership-manifest production rows and all Delivery 3+ surfaces remain
  reserved for their declared deliveries.
- One shared deterministic terminal-text invariant is added: a semantic seed
  must retain at least one non-whitespace authored character after HTTP URL
  tokens are removed. It is mechanical deliverability validation, not a
  semantic evaluator, and adds no schema field, prompt text, model call,
  route configuration, compatibility behavior, or fallback vocabulary.
- P response goals, including ordinary and self-cognition plans, use that
  invariant and route URL-only products through the existing bounded typed
  P recovery and pre-commit replay. The validated `selected_surface_intent`
  uses the same invariant.
- Dialog applies source-token normalization to every structurally valid
  candidate, including when the allowed source set is empty. T1-normalized
  acceptance records `parse_status="normalized"`, `status="succeeded"`,
  protected `status="parsed"`, and contract-event `status="normalized"`.
  Terminal projection is limited to accepted candidate, newest retained
  candidate, sanitized `content_plan`, then sanitized validated
  `selected_surface_intent`.
- Accepted-degraded dialog results carry exactly one existing
  `episode_attempt_diagnostic.v1` row and propagate it through the
  `dialog_agent` return mapping. Normal and T1-normalized results carry no
  degradation row. Quality events use the real `llm_trace_id`.
- The fixed executor remains `/root/live_response_recovery_impl` using
  `gpt-5.6-luna` at maximum reasoning and normal speed; parent `/root`
  remains the independent reviewer. The canonical
  `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` repair path,
  structured-output transport, route configuration, prompts, attempt caps,
  and caller-owned invariant failures remain unchanged.

## Scope And Change Direction

The live response path currently mixes four unrelated failure disciplines. Two
owners already implement the intended behavior
(`cognition_shared/surface_stages.py::_run_surface_stage` and
`nodes/persona_supervisor2_msg_decontextualizer.py::call_msg_decontextualizer`):
bounded attempts whose repair request carries the exact contract error and the
rejected candidate, followed by a deterministic degradation the caller owns.
Every other governed stage implements some strict subset: one attempt with no
feedback, retries with no feedback, deterministic fallback with no retry, or no
containment at all.

The change direction is to normalize all governed stages onto the existing
`_run_surface_stage` discipline, and to close the dialog stage so that a
committed turn always produces visible text.

Included:

- One declared recovery ladder with fixed tier semantics and fixed dispositions.
- Cognition V3 per-stage bounded regeneration with contract feedback, replacing
  the current single call whose validator runs outside the attempt boundary.
- Typed cognition failures that the brain service can classify and replay.
- A dialog generator with no failure exit on the visible path.
- Provider-exception containment for every governed stage that currently lets a
  provider exception leave the stage.
- Degradation for every optional post-commit stage.
- An observable degradation record on the settled episode trace.

Excluded: see `Deferred`.

Dependencies: none. This plan does not depend on and is not blocked by the
active DSH plans or the `epistemic_and_role_provenance_fidelity` plan.

## Confirmed Decisions

1. Dialog generation has no failure exit on the visible path.
   `DialogGenerationContractError` is deleted, not retained behind a flag.
2. No runtime semantic evaluator, semantic score, or evaluator-driven repair is
   reintroduced anywhere. The decommission recorded in
   `archive/completed/dialog_final_generator_evaluator_decommission_plan.md`
   stands. "Semantic degradation" is handled only to the extent deterministic
   contract checks can observe it.
3. Deterministic recovery is allowed only where it adds no semantics: key-order
   normalization, clamping a declared bound on non-assertive free text, and
   exact-literal fidelity repair. Wrong types, unknown keys, unsupported enum
   values, missing required fields, and conflicting fields stay non-recoverable
   and go to feedback-bearing regeneration, matching `AGENTS.md`
   "LLM Error State Handling".
4. `epistemic_boundary` is never clamped. Truncating it weakens a restriction
   and would let downstream wording assert more than cognition allowed. It
   regenerates instead. `private_monologue` is clamped because it is private and
   carries no assertion authority.
5. Regeneration is used only where no deterministic recovery exists. Frontline
   relevance and non-authoritative settled relevance already own an
   information-complete deterministic decision for contract-invalid output, so
   they keep exactly one semantic attempt and gain regeneration only for
   provider failures.
6. Tier 4 checkpoint replay reuses the existing single service-level graph
   replay and its existing `COGNITION_SAFE_RETRY_LIMIT = 1` bound. No new replay
   machinery and no new checkpoint store. Replay is available only when
   `safe_checkpoint == "pre_state_commit"`.
7. Post-cognition-commit stages never replay. They terminate at tier 3.
8. Per-stage provider-enforced JSON Schema is out of scope. The shared
   `json_schema` transport in
   `llm_interface/providers/openai_compatible.py` intentionally sends a
   permissive `{"type": "object", "additionalProperties": true}` schema; making
   it per-stage is a separate architectural change.
9. Degradation dispositions are recorded on the existing
   `episode_attempt_diagnostic.v1` carrier, which `brain_service/post_turn.py`
   already reads from the graph result and
   `brain_service/cognition_observation_projection.py` already projects. No new
   observability schema is introduced.

## Static Analysis Findings

Read against the working tree at commit `4e6a6215`. The problem-statement
inventory was treated as a claim set and re-derived from source.

### Confirmed

| Claim | Evidence |
|---|---|
| FM-06 one-shot cognition stages | `cognition_core_v3/facade.py::_call_once` makes one `services.llm.ainvoke` call and re-raises provider exceptions; `_run_cognition` calls it once per stage. |
| FM-04 parse success recorded before stage validation | `_call_once` writes `parse_status="succeeded"` and protected `status="parsed"` and returns; `validate_canonical_appraisal`, `_validate_goal`, and `_validate_plan` run afterwards in `_run_cognition`. |
| FM-03 and the observed P incident | `_validate_plan` calls `_bounded_text(raw["response_goal"], "response goal")`; an object value raises `CanonicalContractError`. `max_actions = 2 if str(raw["response_goal"]).strip() else 3` accepts the object before the type check, so the failure is the bounded-text rule exactly as reported. |
| FM-05 generic classification | `CanonicalContractError(ValueError)` and `AppraisalContractError(ValueError)` are not `CognitionContractError` subclasses, so `service.py::_operational_failure_metadata` reaches its `isinstance(exc, ValueError)` branch and reports `internal_invariant`. |
| FM-07 frontline provider failure escapes | `relevance/frontline_relevance_agent.py::frontline_relevance_agent` awaits `ainvoke` with no handler. |
| FM-08 settled provider failure escapes | `relevance/persona_relevance_agent.py::relevance_agent` awaits `ainvoke` at three sites with no handler. |
| FM-09 local-context provider failure escapes | `local_context_resolver/stages.py` awaits `ainvoke` in all four stages with no handler; `service.py::resolve_local_context` catches only `(LocalContextValidationError, ValueError)`. |
| FM-10 memory lifecycle failure escapes | `nodes/persona_supervisor2_memory_lifecycle.py::_invoke_memory_lifecycle_specialist` has no handler around `ainvoke`, `normalize_memory_lifecycle_output`, or `materialize_memory_lifecycle_actions`. |
| FM-11 text surface degrades | `cognition_shared/surface.py::_run_text_surface_planning` catches `CognitionExecutionError` and returns `build_degraded_text_surface`. |
| FM-12 unexpected visual failure escapes | `nodes/persona_supervisor2_l3_surface.py::_run_l3_text_surface_handler` drops the visual result only when it is a `CognitionExecutionError` whose `stage == "surface.visual"`; every other `BaseException` is re-raised even though visual output is optional. |
| FM-13 dialog exhaustion raises | `nodes/dialog_agent.py::dialog_generator` raises `DialogGenerationContractError` when no candidate is accepted. |
| FM-14 no dialog semantic stage | Confirmed; the generator performs JSON, message-shape, and source-URL checks only. |
| FM-15 post-commit ordering | `nodes/persona_supervisor2.py` commits in `stage_1_goal_resolver`, enqueues in `stage_2a_background_work_enqueue`, and executes pre-surface actions in `call_action_subgraph` before `call_l3_text_surface_handler` and `dialog_agent`. |
| FM-16 post-turn consumers contained | Confirmed at the service owner. |

### Corrected

| Claim | Correction |
|---|---|
| FM-02 "a response that still cannot yield the required JSON object is a structural contract error" | `utils.py::parse_llm_json_output` never raises; it logs and returns `{}`. Every governed stage therefore fails at its own validator, not at the parser. `facade.py::_call_once` converts the empty result into `CanonicalContractError` explicitly; frontline and settled reach their validators with `{}` and use their deterministic fallbacks. The parse boundary is already uniform. |
| FM-07 "frontline contains deterministic fallback decisions for invalid parsed output" | True for validation and, because of the correction above, also true for unparseable output. The real frontline gap is provider-exception containment only. |
| FM-13 "three producer opportunities" | Three attempts exist, but all three send byte-identical requests. `_render_dialog_candidate` returns a classified failure kind that `dialog_generator` discards (`generated_dialog, _ = await ...`). There is no mechanism by which attempt 2 or 3 can differ from attempt 1 for a deterministic model fault, which is consistent with the 3/3 live reproduction recorded for the P stage. |
| FM-01 "some owners contain provider exceptions inside a bounded attempt loop" | The contained set is exactly `surface_stages._run_surface_stage`, `dialog_agent._render_dialog_candidate`, `msg_decontextualizer.call_msg_decontextualizer`, and the image-descriptor loop. Every other governed stage is uncontained. |

### Added by this analysis

| ID | Finding |
|---|---|
| SA-01 | `cognition_core_v3/appraisal.py::validate_canonical_appraisal` rejects a correct product on JSON object key order: `tuple(raw) != families`. Object key order carries no semantics, so this is a validator defect that fails a recoverable product. |
| SA-02 | `nodes/dialog_agent.py::_current_visible_percepts` raises `StateContractError` when the serialized visible percepts exceed 24000 characters. Its only consumer is `_completed_tool_result_source_urls`; the percepts never enter the dialog prompt. The bound is vestigial and can terminate a committed turn for a reason unrelated to dialog. |
| SA-03 | `_dialog_source_url_issues` rejects a candidate that omits every required URL and a candidate that contains any URL outside the required set. Both are exact-literal faults with deterministic repairs available (append an allowed token, remove an unsupported token) that the current code does not attempt. |
| SA-04 | `run_cognition` wraps the chain in `asyncio.wait_for` and lets a bare `asyncio.TimeoutError` escape. `service.py::_operational_failure_metadata` classifies it as `provider_transient` with `retryable=True`, but `_can_retry_cognition_failure` requires a `CognitionExecutionError`, so the reported retryability is never honored. |
| SA-05 | `cognition_resolver/loop.py::call_cognition_resolver_loop` has no containment around `call_cognition_subgraph_func`. A cognition failure on cycle N discards every capability observation already collected in cycles 0..N-1 even though nothing was committed. |
| SA-06 | `cognition_shared/model_attempt_policy.py` exposes `reserve_v2_model_attempt`, `record_v2_attempt_disposition`, and `record_v2_branch_disposition`, none of which are called from production. Only `V2_MODEL_TOTAL_ATTEMPTS` is consumed. The ledger is not a usable carrier for new attempt accounting. |
| SA-07 | `relevance/persona_relevance_agent.py` already owns a deterministic authoritative decision path (`_deterministic_authoritative_decision`, `_authoritative_settled_assessment`) used when only one disposition is available. It is not used when the model exhausts its repair, so an available deterministic answer is discarded in favor of an operational failure. |
| SA-08 | `local_context_resolver/service.py` already owns `_deterministic_synthesis_response` and a `blocked` node status. Neither is used on stage exhaustion. |

## Mandatory Skills

| Skill | When it applies |
|---|---|
| `.agents/skills/py-style` | Before every `.py` edit in this plan. N-002 and N-003 bound the new handlers: catch the existing typed provider tuple `(OpenAIError, httpx.HTTPError, ConnectionError, OSError, RuntimeError, TimeoutError)` and the typed contract classes, never `except Exception`. P-002 bounds the `try` body to the risky call. |
| `.agents/skills/test-style-and-execution` | Before adding or running any test. Deterministic ladder behavior uses patched-LLM handoff tests; prompt convergence uses live-LLM tests run one at a time with inspected output. |
| `.agents/skills/development-plan` | Execution, handoff, checkpoint, and evidence rules for this plan. |
| `.agents/skills/python-venv` | Use `venv\Scripts\python` for every command. |
| `.agents/skills/llm-trace-debug` | When interpreting a live-LLM gate or a retained protected trace. |

## Mandatory Rules

1. No compatibility shim, alias module, fallback mapper, or parallel vocabulary.
   Caller, callee, tests, and the ownership manifest move together.
2. No broad exception handler. Every new containment names its exception
   classes. `asyncio.CancelledError` and `PipelineCancelled` always propagate.
3. Degradation never invents semantic content. A degraded product may only
   reorder, clamp a declared bound, drop an unsupported literal, insert an exact
   allowed literal, copy an already-validated upstream field, or select an
   already-produced candidate.
4. A repair request never carries a new semantic instruction. It carries the
   original stage system prompt, the original packet, and a `contract_repair`
   block.
5. Deterministic internal-invariant failures stay fatal. Caller-owned input
   faults (`_validate_canonical_input`, `_prepare_state_transaction`,
   `build_text_surface_input_from_global_state`, `validate_cognition_state`,
   `CognitionContextLimitError`) are bugs and must keep raising. The ladder
   governs model-product faults and provider faults only.
6. Every governed stage records one attempt row per attempt through
   `llm_tracing.record_llm_trace_step` with the true `parse_status`, `status`,
   `attempt_index`, and `validation_error`, and never records `succeeded` for an
   attempt a stage validator later rejected.
7. Trace and event writes must not change semantic execution. Keep the existing
   pattern of isolating trace writes from the stage result.

## Must Do

1. Replace `facade._call_once` with a bounded cognition stage runner that owns
   parse, stage validation, deterministic recovery, feedback-bearing
   regeneration, per-attempt tracing with the true disposition, and typed
   exhaustion.
2. Normalize appraisal family slots by key set, not key order (SA-01).
3. Clamp an over-length `private_monologue` and record the normalization; leave
   every other bound to regeneration.
4. Raise `CognitionExecutionError` with
   `error_code="cognition_<stage>_contract_exhausted"`,
   `stage="cognition_core_v3.<stage>"`, `attempt_count=3`,
   `safe_checkpoint="pre_state_commit"`, `retryable=True` on stage exhaustion.
5. Convert the cognition turn-deadline expiry into
   `CognitionExecutionError(error_code="cognition_turn_deadline_exhausted",
   stage="cognition_core_v3", safe_checkpoint="pre_state_commit",
   retryable=False)` (SA-04).
6. Make the cognition stage runner deadline-aware: do not start a regeneration
   attempt when the remaining turn budget is below
   `_COGNITION_ATTEMPT_TIME_FLOOR_SECONDS`; stop and raise the typed exhaustion
   instead of being cancelled mid-attempt.
7. Contain cognition failures inside `call_cognition_resolver_loop` so a
   mid-recurrence failure preserves the collected `resolver_state` observations
   in the raised typed failure's diagnostics instead of discarding them silently
   (SA-05).
8. Delete `DialogGenerationContractError` and every reference to it, including
   the `service.py` import and its branch in `_operational_failure_metadata`.
9. Give the dialog generator the full ladder: source-URL fidelity normalization,
   three feedback-bearing attempts, newest-retained-candidate degradation, and a
   terminal deterministic projection of
   `text_surface_output_v2["content_plan"]`.
10. Replace the vestigial 24000-character raise in `_current_visible_percepts`
    with a bounded scan that cannot fail dialog (SA-02).
11. Widen the L3 visual drop rule so any non-cancellation visual failure is
    omitted while text surface and dialog continue (FM-12).
12. Contain and degrade both memory-lifecycle entry points
    (`call_memory_lifecycle_update_handler`,
    `call_post_surface_memory_lifecycle_review`) to a `skipped` lifecycle
    context that preserves non-lifecycle action specs.
13. Add a shared bounded runner in `local_context_resolver/stages.py` used by all
    four stages, and per-stage degradation in
    `local_context_resolver/service.py`: blocked packet for the planner, node
    `blocked` plus continued traversal for the active node,
    `should_collapse: false` for collapse review, and
    `_deterministic_synthesis_response` for synthesis (SA-08).
14. Contain provider failures in `frontline_relevance_agent` with two attempts
    and terminate on the existing deterministic decision.
15. Contain provider failures in `relevance_agent` for all three call sites, and
    degrade authoritative repair exhaustion to
    `_deterministic_authoritative_decision` when a disposition is available
    (SA-07). Keep `SettledRelevanceContractError` for the case where no
    disposition is available and for the repair-input cap.
16. Plumb `attempt_diagnostics` from the persona graph to the settled episode
    trace with an additive reducer, and emit one row per degraded or exhausted
    governed stage.
17. Update `tests/ownership/source_test_impact_manifest.json` for every strict
    source path this plan changes.
18. Document the ladder tiers and dispositions in
    `docs/architecture/cognition_observability_icd.md`.

## Deferred

- Per-stage provider-enforced JSON Schema output transports.
- Any runtime semantic evaluator, semantic score, sibling salvage, goal-bid
  exhaustion, or unavailable-goal state.
- Feedback-bearing repair for the image descriptor loop. It is already bounded
  and already degrades to `summary_status="unavailable"`.
- Post-turn consumers (conversation progress, internal-monologue residue,
  consolidation). Already contained at the service owner.
- Changing `COGNITION_SAFE_RETRY_LIMIT`, `V2_MODEL_TOTAL_ATTEMPTS`,
  `turn_deadline_seconds`, or any route configuration.
- Removing the unused `model_attempt_policy` ledger functions (SA-06).
- Any change to prompt semantics, character judgment, response sensitivity, or
  the visible-content authority contract.
- Adding `src/kazusa_ai_chatbot/relevance/**` or
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py` to
  `source_roots` in the ownership manifest. Their exact nodes are fixed by this
  plan's `Test Impact And Traceability` section instead.

## Target State

### The Live Response Recovery Ladder

Every governed stage applies these tiers in order. A stage may skip a tier only
where this plan says the tier does not exist for that stage.

| Tier | Name | Meaning | Recorded disposition |
|---|---|---|---|
| T1 | `recover` | Deterministic normalization that adds no semantics. | `normalized` |
| T2 | `regenerate` | Bounded same-context regeneration whose request carries the exact contract error and the bounded rejected candidate. A provider failure consumes one attempt and uses the no-candidate reason. | `regenerate` |
| T3 | `degrade` | Deliver an already-available product: a sibling path result, a retained earlier candidate, or a deterministic projection of already-validated upstream truth. | `accepted_degraded` or `skipped` |
| T4 | `replay` | Raise a typed `CognitionExecutionError(retryable=True, safe_checkpoint="pre_state_commit")` so the existing single service-level graph replay re-runs from the settled-relevance and prewarm checkpoint. | `retry_graph` |

Terminal rule: a stage with no T3 and no T4 may raise a typed failure. The
dialog stage has a T3 and no failure exit, so it never raises on the visible
path.

### Per-stage ladder assignment

| Stage | Owner | T1 | T2 limit | T3 | T4 | May terminate the turn |
|---|---|---|---|---|---|---|
| frontline relevance | `relevance/frontline_relevance_agent.py` | existing `_start_decision` / `_discard_decision` | provider only, 2 | deterministic decision | no | no |
| settled relevance, non-authoritative | `relevance/persona_relevance_agent.py` | existing `_ignore_decision` | provider only, 2 | `_ignore_decision` | no | no |
| settled relevance, authoritative | same | none | existing 2 (initial plus repair) plus provider | `_deterministic_authoritative_decision` | no | only when no disposition is available |
| local-context planner | `local_context_resolver/stages.py`, `service.py` | none | 3 | blocked packet | no | no |
| local-context active node | same | subagent source rows already merged | 3 | node `blocked`, traversal continues | no | no |
| local-context collapse review | same | none | 3 | `should_collapse: false` | no | no |
| local-context synthesis | same | none | 3 | `_deterministic_synthesis_response` | no | no |
| cognition A1 and A2 | `cognition_core_v3/facade.py`, `appraisal.py` | family key-set normalization | 3 | none | yes | yes, typed, pre-commit |
| cognition G | `facade.py` | `private_monologue` clamp | 3 | none | yes | yes, typed, pre-commit |
| cognition P | `facade.py` | none | 3 | none | yes | yes, typed, pre-commit |
| memory lifecycle, pre-surface and post-surface | `nodes/persona_supervisor2_memory_lifecycle.py` | existing normalizer tolerance | 3 | `skipped` context, non-lifecycle specs preserved | no | no |
| text surface content plan | `cognition_shared/surface_stages.py`, `surface.py` | existing | existing 3 | existing `build_degraded_text_surface` | no | no |
| visual surface | `nodes/persona_supervisor2_l3_surface.py` | existing | existing 3 | omit visual output | no | no |
| dialog generator | `nodes/dialog_agent.py` | percept bound, source-URL fidelity normalization for every candidate | 3 | newest retained candidate, sanitized `content_plan`, then sanitized `selected_surface_intent` projection | no | never |

### Cognition stage runner

`facade._call_once` becomes `_run_cognition_stage`:

```text
_run_cognition_stage(
    *, services, stage, packet, validator, deadline_monotonic,
) -> object
```

- `packet` is the built stage packet. `validator` is the stage's product
  validator (`validate_canonical_appraisal` bound to its families,
  `_validate_goal`, or `_validate_plan` bound to `self_cognition` and
  `capabilities`).
- Attempt limit is `_COGNITION_STAGE_ATTEMPT_LIMIT = V2_MODEL_TOTAL_ATTEMPTS`.
- Attempt 1 sends the existing messages. Attempt `n > 1` sends the same stage
  system prompt and a human message whose content is the original packet plus:

```json
{
  "contract_repair": {
    "repair_instruction": "<cognition stage repair instruction>",
    "reason": "<no-candidate or failed-contract reason>",
    "contract_error": "<exact validator message, capped at 500 chars>",
    "invalid_candidate": "<bounded rejected raw output, capped at 8000 chars>"
  }
}
```

- Provider failure: record `parse_status="provider_error"`, `status="failed"`,
  protected `status="provider_error"`; build the no-candidate repair request;
  continue.
- Empty or non-object parse: record `parse_status="contract_error"`,
  `status="failed"`, protected `status="contract_fault"`; build the
  failed-contract repair request with the raw output; continue.
- Validator rejection: same recording as parse rejection, with the validator
  message as `contract_error`.
- T1 normalization is applied by the validator, not by the runner. A normalized
  product is accepted; the runner records `parse_status="normalized"` and emits
  `record_model_contract_event(violation_kind=..., repair_used=False,
  status="normalized")`.
- Acceptance: record `parse_status="succeeded"`, `status="succeeded"`, protected
  `status="parsed"`, `attempt_index=<attempt>`, and return the validated
  product. On a first-attempt clean product this is byte-identical to today's
  recording, so `attempt_index == 1` and protected `status == "parsed"` remain
  true.
- Exhaustion or an insufficient remaining deadline: raise
  `CognitionExecutionError` per `Must Do` items 4 and 6.

`_run_cognition` computes one monotonic deadline from
`services.turn_deadline_seconds` and passes it to every stage. The outer
`asyncio.wait_for` stays as the hard bound and its expiry is converted per
`Must Do` item 5.

### Dialog terminal contract

`dialog_generator` returns
`{"final_dialog": [...], "text_surface_output_v2": ...}` in every reachable
case and adds exactly one `attempt_diagnostics` row only for an
accepted-degraded result. Ordered resolution:

1. For each of three attempts, render a candidate. Attempt `n > 1` includes a
   `contract_repair` block in the payload with the same four fields as the
   cognition runner, carrying the rejected candidate and the exact structural or
   source-URL failure text.
2. On every structurally valid candidate, including when
   `required_source_urls` is empty, apply source-token T1 in this order and
   re-check:
   - remove every URL token absent from `required_source_urls` from every
     message, and drop a message that becomes blank;
   - if no allowed token remains and appending one keeps the total within
     `DIALOG_CANDIDATE_MAX_CHARS`, append the first `required_source_urls` token
     to the last message on its own line.
   A candidate that passes after normalization is accepted and recorded with
   `parse_status="normalized"`, `status="succeeded"`, protected
   `status="parsed"`, and
   `record_model_contract_event(violation_kind="source_url_fidelity",
   repair_used=True, status="normalized")`.
3. Retain every structurally valid candidate that still fails after
   normalization, newest last.
4. If no attempt was accepted and a retained candidate exists, deliver the
   newest retained candidate. Record
   `record_dialog_quality_event(quality_status="accepted_degraded",
   failure_codes=["source_url_fidelity"])` and one `attempt_diagnostics` row.
5. If no retained candidate exists, deliver the sanitized
   `[surface_output["content_plan"]]` when it retains authored text after URL
   removal; otherwise deliver the sanitized
   `[surface_output["selected_surface_intent"]]`. The selected intent is the
   existing upstream P semantic authority and is validated by the shared
   terminal-text invariant. Strip surrounding whitespace and remove any URL
   token absent from `required_source_urls`.
   Record `record_dialog_quality_event(quality_status="accepted_degraded",
   failure_codes=["deterministic_surface_projection"])` and one
   `attempt_diagnostics` row.

`StateContractError` remains for internal invariants: a missing or invalid
`text_surface_output_v2`, a non-object `text_surface_input`, and an invalid
accepted surface. Those are caller bugs, not model faults.

### Degradation record

New reducer and field:

```python
# src/kazusa_ai_chatbot/state.py
MAX_EPISODE_ATTEMPT_DIAGNOSTICS = 16

def append_attempt_diagnostics(
    current: list[EpisodeAttemptDiagnosticV1] | None,
    update: list[EpisodeAttemptDiagnosticV1] | None,
) -> list[EpisodeAttemptDiagnosticV1]:
    """Concatenate stage-attempt diagnostics within the retained bound."""
```

`IMProcessState` and both persona state TypedDicts in
`nodes/persona_supervisor2_schema.py` gain:

```python
attempt_diagnostics: Annotated[
    list[EpisodeAttemptDiagnosticV1],
    append_attempt_diagnostics,
]
```

`persona_supervisor2` adds
`"attempt_diagnostics": results.get("attempt_diagnostics", [])` to its return
mapping. `brain_service/post_turn.py` already reads the key; no change there.

Row shape for a degraded or exhausted governed stage:

```json
{
  "schema_version": "episode_attempt_diagnostic.v1",
  "stage": "dialog_generation",
  "error_code": "dialog_source_url_degraded",
  "attempt_count": 3,
  "safe_checkpoint": "post_cognition_commit",
  "retryable": false,
  "final_status": "accepted_degraded"
}
```

Fixed `error_code` values introduced by this plan:

`cognition_a1_contract_exhausted`, `cognition_a2_contract_exhausted`,
`cognition_g_contract_exhausted`, `cognition_p_contract_exhausted`,
`cognition_turn_deadline_exhausted`, `dialog_source_url_degraded`,
`dialog_surface_projection_degraded`, `memory_lifecycle_skipped`,
`surface_visual_omitted`, `local_context_planner_blocked`,
`local_context_node_blocked`, `local_context_collapse_skipped`,
`local_context_synthesis_degraded`, `settled_relevance_deterministic_degraded`,
`frontline_relevance_deterministic_degraded`.

Fixed `final_status` values: `accepted_degraded`, `skipped`, `exhausted`,
`retry_graph`, `normalized`.

## Contracts And Data Shapes

1. Cognition stage repair packet: the original packet mapping plus one
   `contract_repair` key whose value has exactly
   `{repair_instruction, reason, contract_error, invalid_candidate}`. No other
   key is added or removed. The `output_contract` key stays as produced by
   `cognition_core_v3/prompt.py`.
2. Dialog repair payload: the existing dialog payload plus one `contract_repair`
   key with the same four fields.
3. Local-context repair packet: the existing stage payload plus one
   `contract_repair` key with the same four fields. Each stage reuses
   `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` and its existing JSON
   repair path; `utils.py`, JSON-repair internals, and structured-output
   transport remain unchanged. The local-context recovery change covers stage
   content and contract validity only.
4. Memory-lifecycle repair packet: the existing `_specialist_prompt_payload` or
   `_post_surface_specialist_prompt_payload` result plus one `contract_repair`
   key with the same four fields.
5. Settled and frontline provider containment adds no new payload shape.
6. `TextSurfaceOutputV2`, `TextSurfaceInput`, `CanonicalCognitionOutput`,
   `EpisodeTraceV2`, `EpisodeAttemptDiagnosticV1`, `SurfaceOutputV1`,
   `ActionResultV1`, and every route configuration are unchanged.

## Runtime Or Resource Constraints

| Constraint | Source | Value after this plan |
|---|---|---|
| Cognition chain wall clock per pass | `CognitionChainServicesV3.turn_deadline_seconds` | unchanged, default 240s, range 30-600s |
| Cognition model calls per pass | `_COGNITION_STAGE_ATTEMPT_LIMIT` | 4 on a clean pass, at most 12 |
| Cognition wall clock per turn including replay | `COGNITION_SAFE_RETRY_LIMIT = 1` | at most `2 x turn_deadline_seconds`, the same theoretical bound the existing `version_conflict` replay already permits |
| Regeneration attempt floor | `_COGNITION_ATTEMPT_TIME_FLOOR_SECONDS` | 20s, a new named constant; a regeneration attempt is not started below it |
| Repair error text cap | new cognition, dialog, local-context, and lifecycle constants | 500 chars, matching `SURFACE_STAGE_ERROR_CAP` |
| Rejected candidate cap in a repair prompt | new per-owner constants | 8000 chars, matching `SURFACE_STAGE_REPAIR_OUTPUT_CAP` |
| Dialog visible text cap | `DIALOG_CANDIDATE_MAX_CHARS` | unchanged, 12000 |
| Required source URL cap | `_MAX_REQUIRED_SOURCE_URLS` | unchanged, 8 |
| Retained attempt diagnostics | `MAX_EPISODE_ATTEMPT_DIAGNOSTICS` | 16 |
| Frontline and settled attempts | new per-owner constants | 2 |
| Local-context and lifecycle attempts | new per-owner constants set from `V2_MODEL_TOTAL_ATTEMPTS` | 3 |

## Execution Roles

### `cognition_stage_recovery_owner`

- Responsibility: the Cognition V3 per-stage ladder, typed cognition failures,
  and deadline attribution.
- Owned surface: `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`,
  `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py`,
  `src/kazusa_ai_chatbot/cognition_resolver/loop.py`,
  `tests/unit/cognition_core_v3/test_stage_recovery.py`,
  `tests/unit/cognition_core_v3/test_state_transaction.py`,
  `tests/unit/cognition_resolver/test_loop.py`,
  `tests/test_cognition_v3_response_goal_contract_live_llm.py`.
- Authority: refactor `_call_once` into the stage runner, add the T1
  normalizations named in `Must Do` 2 and 3, add the typed failures, and contain
  the resolver loop. May not change stage prompts, packet contents,
  `output_contract`, attempt limits, or route configuration.
- Applicable skills: `py-style`, `test-style-and-execution`, `llm-trace-debug`,
  `python-venv`.
- Capability floor: must read `cognition_core_v3/README.md`,
  `cognition_shared/surface_stages.py::_run_surface_stage`, and
  `archive/completed/cognition_v3_handleless_model_contract_bigbang_plan_2026-08-22.md`
  before editing, and must be able to run the named live-LLM node one case at a
  time against a configured provider.
- Independence requirement: none from the other implementation roles; separate
  from `independent_review_owner`.
- Acceptance output: the deterministic nodes in its owned test files pass, and
  the live node `test_live_captured_p_stage_converges_within_attempt_cap` passes
  with a retained artifact under
  `test_artifacts/diagnostics/cognition_v3_response_goal_contract_live_llm/`.
- Gate: entry requires an approved plan status and a captured execution
  baseline. Exit requires every mapped node collected and run, plus the live
  artifact.

### `dialog_no_fail_owner`

- Responsibility: the dialog stage's inability to fail on the visible path.
- Owned surface: `src/kazusa_ai_chatbot/nodes/dialog_agent.py`,
  `tests/unit/nodes/test_dialog_agent.py`,
  `tests/test_dialog_generator_live_llm_contract.py`.
- Authority: delete `DialogGenerationContractError`, add the ladder, add the
  source-URL normalizations, and replace the vestigial percept bound. May not
  add a semantic evaluator, change `_V2_DIALOG_GENERATOR_PROMPT` semantics,
  change the attempt limit, or change the accepted surface contract.
- Applicable skills: `py-style`, `test-style-and-execution`, `cjk-safety`,
  `python-venv`.
- Capability floor: must read the module docstring,
  `cognition_shared/surface.py`, and
  `archive/completed/dialog_final_generator_evaluator_decommission_plan.md`
  before editing.
- Independence requirement: none from the other implementation roles.
- Acceptance output: the deterministic nodes in its owned test files pass, and
  no reference to `DialogGenerationContractError` remains in `src/` or `tests/`.
- Gate: exit requires a demonstrated non-raising path for provider exhaustion,
  structural exhaustion, and source-URL exhaustion.

### `optional_stage_containment_owner`

- Responsibility: post-commit optional stages and the local-context resolver
  never terminate a turn.
- Owned surface:
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`,
  `src/kazusa_ai_chatbot/local_context_resolver/stages.py`,
  `src/kazusa_ai_chatbot/local_context_resolver/service.py`,
  `tests/test_memory_lifecycle_specialist.py`,
  `tests/unit/nodes/test_persona_supervisor2_l3_surface.py`,
  `tests/test_local_context_resolver_recovery.py`,
  `tests/test_local_context_resolver_standalone.py`.
- Authority: add the shared local-context stage runner, the per-stage
  degradations, the lifecycle containment, and the widened visual drop rule. May
  not change the local-context packet contracts, node or artifact validators,
  cache keys, or the text-surface degradation already in place.
- Applicable skills: `py-style`, `test-style-and-execution`, `python-venv`.
- Capability floor: must read `local_context_resolver/README.md` and
  `nodes/README.md` before editing.
- Independence requirement: none from the other implementation roles.
- Acceptance output: the deterministic nodes in its owned test files pass.
- Gate: exit requires each of the four local-context degradations and both
  lifecycle entry points demonstrated by a mapped node.

### `intake_relevance_containment_owner`

- Responsibility: frontline and settled relevance provider containment and
  authoritative degradation.
- Owned surface: `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py`,
  `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`,
  `tests/test_frontline_relevance_agent.py`,
  `tests/test_persona_relevance_agent.py`.
- Authority: add provider containment and the authoritative T3 degradation, and
  update the two existing nodes whose contract this changes. May not change
  prompt text, projection budgets, evidence validation, the number of semantic
  attempts for contract-invalid output, or the service-side settled failure
  handling.
- Applicable skills: `py-style`, `test-style-and-execution`, `python-venv`.
- Capability floor: must read `service.py::_process_settlement_lease` and
  `runtime_coordination` turn-settlement ownership before editing.
- Independence requirement: none from the other implementation roles.
- Acceptance output: the deterministic nodes in its owned test files pass,
  including the renamed malformed-JSON node.
- Gate: exit requires that
  `test_frontline_direct_open_turn_rejects_discard_without_retry` still passes
  unchanged, proving contract-invalid frontline output still costs exactly one
  model call.

### `service_classification_and_observability_owner`

- Responsibility: failure classification, replay gating, degradation record
  plumbing, the ownership manifest, and the ICD.
- Owned surface: `src/kazusa_ai_chatbot/service.py`,
  `src/kazusa_ai_chatbot/state.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`,
  `tests/ownership/source_test_impact_manifest.json`,
  `docs/architecture/cognition_observability_icd.md`,
  `tests/unit/brain_service/test_cognition_graph_projection.py`,
  `tests/unit/nodes/test_persona_supervisor2_schema.py`,
  `tests/test_cognition_observability_docs.py`.
- Authority: remove the `DialogGenerationContractError` import and branch, add
  the reducer and state field, add the persona return key, update the manifest
  rows, and add the ICD section. May not change `COGNITION_SAFE_RETRY_LIMIT`,
  the operational notice text, the operational error contract, or the queue and
  settlement control flow.
- Applicable skills: `py-style`, `test-style-and-execution`, `python-venv`.
- Capability floor: must read `brain_service/post_turn.py`,
  `brain_service/cognition_observation_projection.py`, and
  `scripts/validate_test_impact.py` before editing.
- Independence requirement: must not be the same executor as
  `dialog_no_fail_owner` for the `DialogGenerationContractError` removal, so the
  deletion is verified from both sides.
- Acceptance output: the deterministic nodes in its owned test files pass, and
  `venv\Scripts\python scripts\validate_test_impact.py` reports no ownership
  error for the changed source set.
- Gate: exit requires a demonstrated `model_contract` classification for
  cognition exhaustion and a demonstrated single replay.

### `relevance_diagnostic_envelope_owner` — Delivery 5 expansion

- Responsibility: carry relevance T3 degradation metadata through the existing
  pending-turn, assessment lease, persona graph, and settled episode boundary.
- Owned surface: the Delivery 5 files listed in the approved expansion
  handoff, including both relevance producers, the shared relevance envelope,
  `brain_service/turn_settlement.py`, the service/state/persona carriers, the
  ownership manifest, ICD, and their mapped tests.
- Authority: change only the outer metadata envelope and its deterministic
  carrier path. The nested `FrontlineDecision` and
  `SettledRelevanceDecision`, their validators, protected trace behavior,
  queue ordering, deadlines, stale-version handling, claim, and response
  semantics remain exact.
- Applicable skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `python-venv`, and `cjk-safety`.
- Independence requirement: the fixed executor is shared across all
  implementation roles under the user's explicit one-subagent waiver; parent
  `/root` is the independent reviewer and does not implement remediation.
- Gate: every Delivery 5 node in the expansion matrix is collected and passes,
  the manifest ownership check is clean for each newly listed strict source,
  and no relevance diagnostic row is produced for a T1 result or a discarded
  turn without a settled episode.

### `independent_review_owner`

- Responsibility: independent code review and plan-conformance sign-off.
- Owned surface: review findings only; no production edits.
- Authority: pass or fail each acceptance criterion and require remediation. May
  not remediate its own findings.
- Applicable skills: `py-style` (review workflow), `development-plan`
  (`references/execution_gates.md`), `test-style-and-execution`.
- Capability floor: must be able to read the full execution diff against the
  recorded baseline and to re-run every mapped deterministic node.
- Independence requirement: must not be an executor of any implementation role
  in this plan.
- Acceptance output: a written review recording, per acceptance criterion, the
  verdict and the evidence, plus a residual-risk list.
- Gate: entry requires all implementation roles reporting complete with
  evidence. Exit requires either a full pass or a recorded remediation handoff
  to a role other than itself.

## Test Impact And Traceability

| Source path | Changed symbol or contract | Semantic owner | Deterministic pytest node IDs | Supplemental node IDs | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v3/facade.py` | `_call_once` to `_run_cognition_stage`, typed exhaustion, deadline attribution, per-attempt disposition, shared terminal-text invariant on ordinary and self-cognition P response goals | V3 cognition facade owner | `tests/unit/cognition_core_v3/test_stage_recovery.py::test_stage_regenerates_with_exact_contract_error_and_rejected_candidate`, `tests/unit/cognition_core_v3/test_stage_recovery.py::test_object_valued_response_goal_converges_after_one_feedback_attempt`, `tests/unit/cognition_core_v3/test_stage_recovery.py::test_url_only_ordinary_response_goal_is_recovered_as_p_contract_error`, `tests/unit/cognition_core_v3/test_stage_recovery.py::test_url_only_self_cognition_response_goal_is_recovered_as_p_contract_error`, `tests/unit/cognition_core_v3/test_stage_recovery.py::test_provider_failure_consumes_one_attempt_and_regenerates`, `tests/unit/cognition_core_v3/test_stage_recovery.py::test_stage_exhaustion_raises_retryable_pre_commit_execution_error`, `tests/unit/cognition_core_v3/test_stage_recovery.py::test_rejected_attempt_records_contract_fault_before_disposition`, `tests/unit/cognition_core_v3/test_stage_recovery.py::test_regeneration_is_skipped_below_the_remaining_deadline_floor`, `tests/unit/cognition_core_v3/test_state_transaction.py::test_cognition_turn_deadline_bounds_full_chain`, `tests/unit/cognition_core_v3/test_handleless_contract.py::test_canonical_cognition_calls_a1_a2_g_p_once_with_subjective_outputs` | `tests/test_cognition_v3_response_goal_contract_live_llm.py::test_live_captured_p_stage_converges_within_attempt_cap` | deterministic unit plus live-LLM | A provider or typed stage-content contract fault receives bounded same-context feedback; exhaustion remains typed and pre-commit |
| `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py` | `validate_canonical_appraisal` family-slot comparison by key set | V3 appraisal semantics owner | `tests/unit/cognition_core_v3/test_stage_recovery.py::test_appraisal_family_key_order_is_normalized_without_regeneration`, `tests/unit/cognition_core_v3/test_state_transaction.py::test_unresolved_continuation_replaces_prior_response_goal_exactly` | none | deterministic unit | A correct appraisal product rejected because the model emitted the family keys in another order |
| `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py` | none; validated unchanged | V3 internal execution contract owner | `tests/unit/cognition_core_v3/test_handleless_contract.py::test_canonical_stage_packets_are_handleless_and_disjoint` | none | deterministic unit | Silent drift of the canonical output or state-projection contract while the facade is refactored |
| `src/kazusa_ai_chatbot/cognition_shared/contracts.py` | shared terminal-text seed invariant and `selected_surface_intent` validation | V3 shared contract owner | `tests/unit/cognition_core_v3/test_handleless_contract.py::test_canonical_stage_packets_are_handleless_and_disjoint`, `tests/unit/cognition_core_v3/test_handleless_contract.py::test_terminal_text_seed_requires_authored_text_after_http_url_removal`, `tests/unit/cognition_core_v3/test_handleless_contract.py::test_selected_surface_intent_uses_the_shared_terminal_text_invariant` | none | deterministic unit | A URL-only semantic seed reaches a terminal visible surface or downstream dialog fallback |
| `src/kazusa_ai_chatbot/cognition_resolver/loop.py` | cognition-failure containment preserving collected observations | cognition_resolver owner | `tests/unit/cognition_resolver/test_loop.py::test_cycle_failure_preserves_collected_observations_in_typed_failure`, `tests/unit/cognition_resolver/test_loop.py::test_loop_exposes_owned_contract` | `tests/test_cognition_resolver_loop.py::test_user_input_blocker_converges_after_one_final_cognition` | deterministic unit | A mid-recurrence cognition failure discards every capability observation already collected before any commit |
| `src/kazusa_ai_chatbot/nodes/dialog_agent.py` | `DialogGenerationContractError` deleted, ladder added, bounded percept scan, source-URL normalization for empty and non-empty allowed sets, terminal projection, protected dispositions, and `attempt_diagnostics` propagation | dialog generator owner | `tests/unit/nodes/test_dialog_agent.py::test_dialog_retry_prompt_carries_rejected_candidate_and_contract_error`, `tests/unit/nodes/test_dialog_agent.py::test_dialog_agent_exposes_owned_contract`, `tests/unit/nodes/test_dialog_agent.py::test_missing_required_source_url_is_appended_without_regeneration`, `tests/unit/nodes/test_dialog_agent.py::test_unexpected_source_url_is_removed_before_degradation`, `tests/unit/nodes/test_dialog_agent.py::test_dialog_delivers_newest_retained_candidate_after_structural_exhaustion`, `tests/unit/nodes/test_dialog_agent.py::test_dialog_projects_content_plan_when_no_candidate_survives`, `tests/unit/nodes/test_dialog_agent.py::test_dialog_never_raises_on_provider_exhaustion`, `tests/unit/nodes/test_dialog_agent.py::test_oversized_visible_percepts_bound_url_scan_without_failing_dialog` | `tests/test_dialog_generator_live_llm_contract.py::test_live_dialog_source_url_feedback_converges`, `tests/test_dialog_mention_target_user.py::test_dialog_agent_returns_no_mention_flag` | deterministic unit plus live-LLM | A committed turn produces no visible character text because dialog exhausted its identical attempts |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` | visual failure drop rule widened, `attempt_diagnostics` row on omission | surface owner | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_unexpected_visual_failure_is_omitted_and_text_surface_is_returned`, `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_visual_cancellation_still_propagates`, `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_handler_binds_state_trace_for_text_and_visual_calls`, `tests/unit/nodes/test_persona_supervisor2_l3_surface.py::test_l3_surface_preserves_relational_willingness` | none | deterministic unit | An optional visual-surface failure terminates a committed turn that has a valid text surface |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py` | provider and contract containment, `skipped` degradation for both entry points | memory lifecycle specialist owner | `tests/test_memory_lifecycle_specialist.py::test_provider_exhaustion_degrades_to_skipped_lifecycle_context`, `tests/test_memory_lifecycle_specialist.py::test_specialist_repair_prompt_carries_contract_error`, `tests/test_memory_lifecycle_specialist.py::test_post_surface_review_provider_failure_returns_empty_update`, `tests/test_memory_lifecycle_specialist.py::test_handler_consumes_route_and_materializes_apply_action` | `tests/test_memory_lifecycle_specialist_live_llm.py::test_live_tiramisu_fulfilled` | deterministic unit plus live-LLM | An optional post-commit lifecycle specialist failure terminates the turn before surface planning and dialog |
| `src/kazusa_ai_chatbot/local_context_resolver/stages.py` | shared bounded stage runner with feedback repair | local-context semantic node owner | `tests/test_local_context_resolver_recovery.py::test_stage_repair_prompt_carries_contract_error`, `tests/test_local_context_resolver_recovery.py::test_stage_provider_failure_is_contained_and_retried`, `tests/test_local_context_resolver_standalone.py::test_stage_prompts_keep_source_field_and_time_boundaries`, `tests/test_local_context_resolver_standalone.py::test_local_context_stages_reuse_canonical_llm_json_parser` | `tests/test_local_context_resolver_live_llm.py::test_live_resolves_local_context_packet` | deterministic unit plus live-LLM | A provider exception in any resolver stage leaves the resolver and reaches the cognition recurrence caller |
| `src/kazusa_ai_chatbot/local_context_resolver/service.py` | per-stage degradation for planner, active node, collapse, and synthesis | local-context resolver service owner | `tests/test_local_context_resolver_recovery.py::test_planner_provider_exhaustion_returns_blocked_packet`, `tests/test_local_context_resolver_recovery.py::test_active_node_exhaustion_blocks_one_node_and_continues_traversal`, `tests/test_local_context_resolver_recovery.py::test_collapse_exhaustion_defaults_to_no_collapse`, `tests/test_local_context_resolver_recovery.py::test_synthesis_exhaustion_uses_deterministic_synthesis`, `tests/test_local_context_resolver_standalone.py::test_node_artifact_binds_code_owned_metadata` | none | deterministic unit | One unresolved evidence node or one failed synthesis discards the whole local-evidence packet |
| `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py` | provider containment with two attempts, deterministic terminal decision | frontline relevance owner | `tests/test_frontline_relevance_agent.py::test_frontline_provider_exhaustion_starts_authoritative_turn`, `tests/test_frontline_relevance_agent.py::test_frontline_provider_exhaustion_discards_non_authoritative_turn`, `tests/test_frontline_relevance_agent.py::test_frontline_direct_open_turn_rejects_discard_without_retry`, `tests/test_frontline_relevance_agent.py::test_frontline_agent_uses_structural_parser_and_returns_decision` | none | deterministic unit | A provider outage at intake fails the queue item instead of using the deterministic admission decision |
| `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py` | provider containment at three call sites, authoritative T3 degradation | settled relevance owner | `tests/test_persona_relevance_agent.py::test_relevance_agent_malformed_json_degrades_to_available_disposition`, `tests/test_persona_relevance_agent.py::test_relevance_agent_raises_when_no_authoritative_disposition_remains`, `tests/test_persona_relevance_agent.py::test_settled_provider_exhaustion_degrades_without_raising`, `tests/test_persona_relevance_agent.py::test_non_authoritative_provider_exhaustion_returns_ignore`, `tests/test_persona_relevance_agent.py::test_authoritative_bad_evidence_does_not_trigger_repair`, `tests/test_persona_relevance_agent.py::test_relevance_repair_feedback_identifies_missing_required_field` | `tests/test_settled_relevance_captured_failure_live_llm.py::test_live_replays_qq_480386272_settled_relevance_failure` | deterministic unit plus live-LLM | An authoritative turn with an available deterministic disposition is dropped as an operational failure |
| `src/kazusa_ai_chatbot/service.py` | `DialogGenerationContractError` import and branch removed, cognition exhaustion classification and replay | brain service cognition graph and runtime owner | `tests/unit/brain_service/test_cognition_graph_projection.py::test_cognition_contract_exhaustion_maps_to_model_contract_error_code`, `tests/unit/brain_service/test_cognition_graph_projection.py::test_pre_commit_contract_exhaustion_triggers_one_checkpoint_replay`, `tests/unit/brain_service/test_cognition_graph_projection.py::test_post_commit_degradations_do_not_trigger_replay`, `tests/unit/brain_service/test_cognition_graph_projection.py::test_failed_run_uses_current_attempt_prewarm_checkpoint`, `tests/unit/brain_service/test_cognition_graph_projection.py::test_legacy_cognition_graph_projection_symbols_are_absent_from_production` | `tests/test_service_background_consolidation.py::test_chat_response_tracks_deliverable_assistant_row` | deterministic unit | A model contract exhaustion reported as `internal_invariant` with no replay, and a dead dialog exhaustion branch left behind |
| `src/kazusa_ai_chatbot/state.py` | `append_attempt_diagnostics`, `MAX_EPISODE_ATTEMPT_DIAGNOSTICS`, `attempt_diagnostics` field | brain service state owner | `tests/unit/brain_service/test_cognition_graph_projection.py::test_attempt_diagnostics_reducer_concatenates_within_bound` | none | deterministic unit | Two degraded stages in one turn overwrite each other's degradation record |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` | `attempt_diagnostics` field on both persona states | persona schema owner | `tests/unit/nodes/test_persona_supervisor2_schema.py::test_persona_states_carry_attempt_diagnostics` | none | deterministic unit | A degradation recorded inside the persona graph never reaches the settled trace |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | `attempt_diagnostics` in the return mapping | persona stage owner | `tests/unit/nodes/test_persona_supervisor2_schema.py::test_persona_supervisor_returns_attempt_diagnostics`, `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_persona_character_commit_reads_canonical_state_projection` | none | deterministic unit | The persona graph drops the degradation rows at its boundary |
| `tests/ownership/source_test_impact_manifest.json` | updated rows for every changed strict source | ownership manifest owner | `tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary`, `tests/test_test_impact_manifest.py::test_manifest_accepts_an_explicit_package_init_source_root` | none | deterministic unit | A changed strict source with no exact owning node |
| `docs/architecture/cognition_observability_icd.md` | ladder tiers, dispositions, and `error_code` vocabulary documented | observability ICD owner | `tests/test_cognition_observability_docs.py::test_icd_documents_live_response_recovery_dispositions`, `tests/test_cognition_observability_docs.py::test_icd_and_runtime_docs_name_one_brain_service_contract_owner` | none | static text | Degradation dispositions appear in the console with no documented meaning |

## Change Surface

### Delete

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py` -> `DialogGenerationContractError`
  and its docstring `Raises:` entry. The dialog stage has no failure exit.
- `src/kazusa_ai_chatbot/service.py` -> the `DialogGenerationContractError`
  import and its entry in the `_operational_failure_metadata` isinstance tuple.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py` -> the 24000-character
  `StateContractError` raise in `_current_visible_percepts`, replaced by a
  bounded scan (SA-02).

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py` - replace `_call_once`
  with `_run_cognition_stage`; add the repair-message builder, the stage attempt
  and cap constants, the deadline floor, and the typed exhaustion; move stage
  validation and disposition recording inside the attempt boundary; convert the
  deadline expiry.
- `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py` - compare family slots
  by key set and iterate the canonical order.
- `src/kazusa_ai_chatbot/cognition_shared/contracts.py` - add the shared
  terminal-text seed invariant and apply it to `selected_surface_intent`.
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py` - contain the cognition
  call so a mid-recurrence failure carries the collected observations.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py` - add the ladder, the
  `contract_repair` payload, the source-URL normalizations, candidate retention,
  the terminal projection through sanitized `content_plan`, then sanitized
  validated `selected_surface_intent`, and the degradation records.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` - widen the
  visual drop rule and record the omission.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py` - add
  the bounded attempt loop with feedback and the `skipped` degradation for both
  entry points.
- `src/kazusa_ai_chatbot/local_context_resolver/stages.py` - add the shared
  bounded stage runner with feedback repair and route all four stages through
  it.
- `src/kazusa_ai_chatbot/local_context_resolver/service.py` - add the four
  per-stage degradations.
- `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py` - add provider
  containment and the deterministic terminal decision.
- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py` - add provider
  containment at three call sites and the authoritative T3 degradation.
- Delivery 5 relevance expansion: add the relevance-owned
  `RelevanceEvaluationEnvelope` contract and package export/documentation;
  change both relevance agents to emit that envelope with only T3 diagnostic
  rows; extend `brain_service/turn_settlement.py` with ordered pending-turn
  and lease metadata; and carry the bounded rows through the existing service,
  state, persona, manifest, and ICD boundaries. The nested decision objects,
  validators, prompts, parser options, protected traces, queue/lease/claim
  control flow, and episode schema remain unchanged.
- `src/kazusa_ai_chatbot/service.py` - classification and replay per the tables
  above.
- `src/kazusa_ai_chatbot/state.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` - the
  `attempt_diagnostics` carrier.
- `tests/ownership/source_test_impact_manifest.json` - rows for every changed
  strict source.
- `docs/architecture/cognition_observability_icd.md` - the ladder section.
- `tests/unit/cognition_core_v3/test_state_transaction.py`,
  `tests/unit/cognition_core_v3/test_handleless_contract.py`,
  `tests/unit/cognition_resolver/test_loop.py`,
  `tests/unit/nodes/test_dialog_agent.py`,
  `tests/unit/nodes/test_persona_supervisor2_l3_surface.py`,
  `tests/unit/nodes/test_persona_supervisor2_schema.py`,
  `tests/unit/brain_service/test_cognition_graph_projection.py`,
  `tests/test_memory_lifecycle_specialist.py`,
  `tests/test_frontline_relevance_agent.py`,
  `tests/test_persona_relevance_agent.py`,
  `tests/test_local_context_resolver_standalone.py`,
  `tests/test_cognition_observability_docs.py`,
  `tests/test_cognition_v3_response_goal_contract_live_llm.py`,
  `tests/test_dialog_generator_live_llm_contract.py` - per the traceability
  matrix.

### Create

- `tests/unit/cognition_core_v3/test_stage_recovery.py`
- `tests/test_local_context_resolver_recovery.py`
- Delivery 5 proposes `src/kazusa_ai_chatbot/relevance/contracts.py` as the
  sole new production contract module because both relevance agents require
  one shared nested envelope without a downstream ownership inversion or an
  import cycle. Its exact manifest row and contract node are listed in the
  Delivery 5 expansion matrix.

### Keep

- `src/kazusa_ai_chatbot/cognition_shared/surface_stages.py` and
  `src/kazusa_ai_chatbot/cognition_shared/surface.py` - already implement the
  ladder; they are the reference pattern and are not edited.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py` -
  already implements the ladder for both the decontextualizer and the image
  descriptor.
- `src/kazusa_ai_chatbot/utils.py::parse_llm_json_output` - already the single
  canonical parse boundary and already non-raising.
- `src/kazusa_ai_chatbot/cognition_shared/model_attempt_policy.py` - only
  `V2_MODEL_TOTAL_ATTEMPTS` is consumed; the unused ledger stays untouched.
- `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` - no prompt or packet
  change.
- `src/kazusa_ai_chatbot/llm_interface/**` - no transport change.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py` and
  `brain_service/cognition_observation_projection.py` - already consume
  `attempt_diagnostics`.
- `src/kazusa_ai_chatbot/action_spec/results.py` -
  `EpisodeAttemptDiagnosticV1` already admits the new rows.

## Agent Autonomy Boundaries

The responsible executor may decide locally:

- Function decomposition inside an owned module, helper naming, constant
  placement, and the order of work inside its own surface.
- The exact wording of new Simplified-Chinese repair instructions, provided they
  add no semantic instruction beyond "return the stage's exact JSON object and
  preserve the original semantic judgment".
- Fixture construction, patch targets, and the internal structure of new tests,
  provided the node IDs in the traceability matrix are the ones collected.
- Log and event message text.

The executor must request a plan amendment or a user decision before:

- Adding, removing, or renaming any ladder tier, disposition, or `error_code`.
- Changing an attempt limit, the turn deadline, `COGNITION_SAFE_RETRY_LIMIT`, or
  a route configuration.
- Changing any prompt's semantic content, packet contents, or `output_contract`.
- Adding a semantic evaluator, a score, a sibling branch, or a second replay.
- Introducing a compatibility layer, an alias, or a feature flag.
- Making any deterministic internal-invariant failure non-fatal.
- Adding a new observability schema instead of using
  `episode_attempt_diagnostic.v1`.
- Adding a path to `source_roots` in the ownership manifest.

If the plan and the code disagree, record the conflict and stop. Do not
reinterpret the ladder.

## Verification

1. Capture the execution baseline: `git status --short` and
   `git rev-parse HEAD`, plus the explicitly owned file set per role.
2. Ownership gate:
   `venv\Scripts\python scripts\validate_test_impact.py` for the changed source
   set; confirm every mapped node is collected.
3. Deterministic gate, batched:
   `venv\Scripts\python -m pytest tests/unit -q` and
   `venv\Scripts\python -m pytest tests/test_frontline_relevance_agent.py tests/test_persona_relevance_agent.py tests/test_memory_lifecycle_specialist.py tests/test_local_context_resolver_recovery.py tests/test_local_context_resolver_standalone.py tests/test_cognition_observability_docs.py tests/test_test_impact_manifest.py -q`.
4. Regression gate, batched: `venv\Scripts\python -m pytest -q`. Unrelated
   pre-existing failures are identified and reported, not fixed, without a user
   decision.
5. Absence gate:
   `venv\Scripts\python -m pytest tests/unit/nodes/test_dialog_agent.py -q`
   plus a repository search proving no `DialogGenerationContractError` or
   `dialog_generator_exhausted` reference remains under `src/` or `tests/`.
6. Live-LLM gate, one case at a time with the output inspected before the next:
   - `venv\Scripts\python -m pytest tests/test_cognition_v3_response_goal_contract_live_llm.py::test_live_captured_p_stage_converges_within_attempt_cap -q -s -m live_llm`
   - `venv\Scripts\python -m pytest tests/test_dialog_generator_live_llm_contract.py::test_live_dialog_source_url_feedback_converges -q -s -m live_llm`
   - `venv\Scripts\python -m pytest tests/test_settled_relevance_captured_failure_live_llm.py::test_live_replays_qq_480386272_settled_relevance_failure -q -s -m live_llm`
   - `venv\Scripts\python -m pytest tests/test_memory_lifecycle_specialist_live_llm.py::test_live_tiramisu_fulfilled -q -s -m live_llm`

   Each run retains a durable artifact containing the case input, model config,
   raw output, parsed output, attempt dispositions, and a judgment note.
7. Independent code review by `independent_review_owner` against every
   acceptance criterion, with findings recorded and remediation handed to a
   different role.

## Acceptance Criteria

1. For every governed stage in the per-stage table, a mapped deterministic node
   demonstrates each tier the table assigns to it, and demonstrates that the
   stage does not raise where the table says it may not.
2. A cognition stage whose first product fails its validator produces a second
   request whose `contract_repair.contract_error` equals the validator message
   and whose `contract_repair.invalid_candidate` contains the rejected raw
   output.
3. An object-valued `response_goal` on attempt 1 followed by a string-valued
   `response_goal` on attempt 2 yields a validated response plan and a completed
   cognition output, with two recorded P attempts whose dispositions are
   `contract_error` then `succeeded`.
4. No cognition attempt is recorded as `succeeded` when its stage validator
   rejected the product. Protected chain records use `parsed` only for accepted
   attempts and `contract_fault` for rejected ones.
5. Cognition stage exhaustion reaches the service as
   `error_code="cognition_<stage>_contract_exhausted"`,
   `stage="cognition_core_v3.<stage>"`, `retryable=True`,
   `safe_checkpoint="pre_state_commit"`; the service performs exactly one graph
   replay; a second exhaustion produces an operational response with
   `exhausted=True` and that same `error_code`, never `internal_invariant`.
6. A cognition turn-deadline expiry reaches the service as
   `error_code="cognition_turn_deadline_exhausted"` with `retryable=False` and
   triggers no replay.
7. `dialog_generator` returns a non-empty `final_dialog` for every combination
   of three provider failures, three structurally invalid outputs, three
   source-URL-failing outputs, and any mix of them. No test can construct a
   model-side input that makes it raise.
8. A candidate that omits every required source URL is accepted after the exact
   allowed token is appended, with no extra model call. A candidate containing a
   URL outside the required set has that token removed before any degradation,
   including when the allowed set is empty. T1-normalized acceptance records
   `parse_status="normalized"`, `status="succeeded"`, protected
   `status="parsed"`, and contract-event `status="normalized"`, with no
   degradation diagnostic; accepted-degraded retained and projection paths each
   carry exactly one fixed dialog diagnostic row.
9. An oversized visible-percept projection does not fail dialog.
10. A visual-surface failure of any non-cancellation type yields a returned text
    surface, a normal dialog, and one `surface_visual_omitted` diagnostic row.
11. A memory-lifecycle provider or contract exhaustion yields a `skipped`
    lifecycle context, preserves the non-lifecycle action specs, and lets
    surface planning and dialog run.
12. Each of the four local-context stage exhaustions yields its assigned
    degradation, and no provider exception leaves `resolve_local_context`.
13. Frontline contract-invalid output still costs exactly one model call.
    Frontline provider exhaustion yields `start` for authoritative evidence and
    `discard` otherwise, and never fails the queue item.
14. Settled authoritative repair exhaustion yields the deterministic disposition
    when one is available, and raises `SettledRelevanceContractError` only when
    none is.
15. A turn with two degraded stages retains both diagnostic rows in the settled
    episode trace, ordered by occurrence and bounded by
    `MAX_EPISODE_ATTEMPT_DIAGNOSTICS`.
16. `scripts/validate_test_impact.py` reports no ownership error for the changed
    source set, and every node in the traceability matrix is collected and run.
17. The ICD documents every tier, disposition, and `error_code` this plan
    introduces, and its docs node passes.
18. No compatibility shim, alias, feature flag, semantic evaluator, or second
    replay was introduced, confirmed by the independent code review against the
    baseline diff.

## Progress Checklist

| # | Outcome | Owner surface | Status | Evidence or next checkpoint |
|---|---|---|---|---|
| 1 | Cognition stage runner, T1 normalizations, typed failures, deadline attribution | `cognition_core_v3/**`, `cognition_resolver/loop.py` | completed | Terminal-deliverability evidence: 17 amended deterministic nodes collected and passing via `venv\Scripts\python.exe`, including ordinary/self-cognition P URL-only recovery and shared/selected-intent invariant checks; edited Python CJK syntax checks pass; `git diff --check` clean; parent re-review pending |
| 2 | Dialog no-fail contract and source-URL normalization | `nodes/dialog_agent.py` | completed | Terminal-deliverability evidence: 8 amended dialog nodes collected and passing; full `tests/unit/nodes/test_dialog_agent.py -q` passed (12); mention-target node passed; empty-allowed URL removal, normalized trace/protected/event dispositions, mixed provider/structural/source degradation, selected-intent projection, fixed diagnostics, and dialog-agent propagation covered; removed-symbol absence search, edited Python CJK syntax checks, and `git diff --check` clean; parent re-review pending |
| 3 | Optional post-commit and local-context containment | `nodes/persona_supervisor2_memory_lifecycle.py`, `nodes/persona_supervisor2_l3_surface.py`, `local_context_resolver/**` | completed | Delivery 3 final evidence: 17 mapped deterministic nodes collected and passing via `venv\Scripts\python.exe`; all four owned test files passed (53). Real queued producer counts cover planner provider exhaustion (3), active-node invalid-content recovery plus traversal (4: 3 rejected + next valid), collapse malformed-candidate exhaustion (3), synthesis malformed-candidate exhaustion (3), and direct stage provider recovery (3). All four stage functions and the shared runner require a pure `candidate_validator`, and owned fakes invoke it; planner/node authoritative service validation/materialization and strict collapse/synthesis candidate checks run inside producer-attempt recovery; collapse exhaustion records `final_status=skipped`. Canonical parser usage is present with no stage-local parser, duplicate partial validators, optional callback fallback, handler shim, or new broad catch; all 8 edited Python files pass `py_compile`, import ordering passes, and `git diff --check` is clean; no live-LLM or full-repository suite run; parent re-review pending |
| 4 | Intake relevance containment and authoritative degradation | `relevance/**` | completed | Delivery 4 evidence: all 10 exact mapped deterministic nodes collected and passing via `venv\Scripts\python.exe`; both owned relevance files passed in full (53). Frontline and settled provider exhaustion each made exactly 2 identical producer calls and returned their existing deterministic decisions; settled authoritative malformed output used the existing initial-plus-repair cap and deterministic disposition, while no available disposition preserved `SettledRelevanceContractError`; contract-invalid frontline and non-authoritative settled outputs remained one-call paths and now trace the approved T1 fallback as `normalized`/`succeeded`; authoritative settled rejections remain `invalid`/`failed`. Truthful traces record provider attempts as `provider_error`/`failed` at indices 1 and 2, accepted model outcomes at their actual index, and deterministic degradation as a distinct `accepted_degraded` step with no raw output. CJK-safe UTF-8 AST/`py_compile`, imports, import ordering, broad-catch scan, and `git diff --check` pass; no prompt/parser-option/route/decision-contract changes and no live/full-repository suite run. Per the approved terminal boundary, Delivery 4 emits, attaches, and persists no `attempt_diagnostics`; Delivery 5 must transport `frontline_relevance_deterministic_degraded` and `settled_relevance_deterministic_degraded` rows through its owned coordinator/state/service carrier boundary; parent review pending |
| 5 | Service classification, replay gating, diagnostics carrier, manifest, ICD, and approved relevance envelope expansion | `relevance/**`, `brain_service/turn_settlement.py`, `service.py`, `state.py`, `nodes/persona_supervisor2*.py`, manifest, ICD | completed | Delivery 5 implementation and deterministic evidence complete; exact mapped nodes and owned regression batch pass, impact ownership validation is clean, and parent explicitly accepted the delivery |
| 6 | Live-LLM gates | named live nodes | completed | All four required live gates passed with retained artifacts; the dialog gate observed first-attempt `source_url_fidelity` rejection and later bounded acceptance; Delivery 6 was parent-accepted and Delivery 7 closeout is recorded below |
| 7 | Independent code review and sign-off | review findings | completed | Delivery 7 verification and parent independent plan-scope sign-off PASS recorded below; two unrelated prompt-context failures and two unrelated full-suite collection failures remain accepted as separate work |

### Delivery 4 Evidence And Delivery 5 Handoff — 2026-08-27

- The exact mapped node set collected and passed (10/10) is:
  `test_frontline_provider_exhaustion_starts_authoritative_turn`,
  `test_frontline_provider_exhaustion_discards_non_authoritative_turn`,
  `test_frontline_direct_open_turn_rejects_discard_without_retry`,
  `test_frontline_agent_uses_structural_parser_and_returns_decision`,
  `test_relevance_agent_malformed_json_degrades_to_available_disposition`,
  `test_relevance_agent_raises_when_no_authoritative_disposition_remains`,
  `test_settled_provider_exhaustion_degrades_without_raising`,
  `test_non_authoritative_provider_exhaustion_returns_ignore`,
  `test_authoritative_bad_evidence_does_not_trigger_repair`, and
  `test_relevance_repair_feedback_identifies_missing_required_field`.
- The full owned deterministic files passed 53/53. Frontline provider
  exhaustion made two identical calls and selected `start` for authoritative
  evidence or `discard` otherwise. Settled non-authoritative exhaustion made
  two identical calls and selected `ignore`; settled authoritative provider or
  repair exhaustion selected the existing deterministic disposition when one
  was available and retained the typed contract error when none was available.
- Trace assertions cover true provider, invalid, normalized, repaired, and
  deterministic dispositions with actual attempt indices. Frontline and
  non-authoritative settled T1 fallbacks are recorded as
  `normalized`/`succeeded` after one model call; authoritative settled
  rejections stay `invalid`/`failed`. A deterministic degradation trace has no
  invented raw model output and uses the real `llm_trace_id` when present.
- The approved Delivery 4 boundary decision defers diagnostics transport.
  These relevance modules emit, attach, and persist no `attempt_diagnostics`.
  Delivery 5 must carry the fixed
  `frontline_relevance_deterministic_degraded` and
  `settled_relevance_deterministic_degraded` rows through its owned
  coordinator/state/service carrier boundary.
- Edited-Python CJK-safe UTF-8 AST/`py_compile`, import, import-ordering,
  broad-catch, and `git diff --check` gates passed. Live-LLM and full-repository
  suites remain pending for their declared deliveries; parent `/root` review
  remains pending.

## Delivery 5 Scope-Expansion Amendment And Preflight — 2026-08-27

- User approval explicitly expands Delivery 5 to carry the two Delivery 4
  relevance T3 diagnostic rows through the existing turn-settlement, brain
  graph, persona, and episode-trace path. Delivery 5 implementation remains
  pending parent approval of this amendment; no production or test file was
  changed during this preflight.
- Execution ownership remains the fixed single executor
  `/root/live_response_recovery_impl`, using `gpt-5.6-luna` at maximum
  reasoning and normal speed. Parent `/root` remains the independent reviewer
  and does not remediate. The user's one-subagent constraint waives any
  separate dialog/service or relevance/settlement executor requirement; all
  Delivery 5 implementation and deterministic testing are serialized through
  the fixed executor.
- Read-only baseline for this amendment remains HEAD
  `016061fe81ee2af31bfffbebf954c5ac51ec3ecf` with the prior Delivery 1-4
  changes preserved in the shared worktree. The existing canonical parser,
  protected trace path, decision validators, queue/lease ordering, graph
  routing, route configuration, packet contracts, and post-turn trace
  constructor remain unchanged.

### Canonical Relevance Evaluation Envelope

Delivery 5 makes the proposed new
`src/kazusa_ai_chatbot/relevance/contracts.py` the sole definition owner for
the two existing decision `TypedDict`s and the new envelope. This is a
relocation of unchanged type contracts, not an alias, duplicate declaration,
compatibility vocabulary, or semantic change. The producer modules import
these definitions; their validators remain in the producer modules and retain
their exact key and semantic behavior. `relevance/__init__.py` re-exports the
canonical definitions for callers.

The relocated decision declarations remain exactly as they are in the current
producer modules. In particular, `FrontlineDecision.prelude_targets` remains
`list[Literal["prelude_1", "prelude_2"]]`; every other relocated annotation,
literal, required key, and value type is copied with exact identity and no
widening:

```python
class FrontlineDecision(TypedDict):
    intake_action: Literal["discard", "start", "append"]
    append_target: Literal["none", "open_1", "open_2", "open_3"]
    prelude_targets: list[Literal["prelude_1", "prelude_2"]]
    reason: str

class SettledRelevanceDecision(TypedDict):
    response_action: Literal["ignore", "proceed", "wait"]
    reason_to_respond: str
    use_reply_feature: bool
    channel_topic: str
    indirect_speech_context: str
```

The shared envelope has exactly these two keys:

```python
class RelevanceEvaluationEnvelope(TypedDict):
    decision: FrontlineDecision | SettledRelevanceDecision
    attempt_diagnostics: list[EpisodeAttemptDiagnosticV1]
```

- The `decision` value is the existing exact `FrontlineDecision` or
  `SettledRelevanceDecision`, nested without added keys. The existing decision
  validators receive only `envelope["decision"]`; their accepted key sets and
  semantic rules remain unchanged.
- `attempt_diagnostics` is a separate bounded metadata list. A T1 normalized
  relevance result emits an empty list. A Delivery 4 T3 deterministic
  degradation emits exactly one existing
  `episode_attempt_diagnostic.v1` row with the fixed relevance error code; no
  provider-attempt row is promoted into the episode carrier.
- The single `state.append_attempt_diagnostics` reducer/normalizer and
  `MAX_EPISODE_ATTEMPT_DIAGNOSTICS = 16` are the only bound/merge authority.
  Every merge concatenates in occurrence order and applies exactly
  `combined[-MAX_EPISODE_ATTEMPT_DIAGNOSTICS:]`. The most recent 16 rows
  survive in their chronological order; when 17 rows are combined, rows 2
  through 17 survive and row 1 is displaced. This keeps terminal
  dialog/surface/cognition outcomes discoverable after many earlier
  frontline/settled degradations; earlier individual relevance attempts
  remain available in protected traces. No other cap, slice, equality
  deduplication policy, alias, or parallel diagnostic vocabulary is permitted.
- `should_respond`, turn identity/version, protected trace data, and other
  operational values are not envelope keys. Settled response fields are read
  from the nested decision at the service boundary, so no flattened decision
  compatibility shape is retained.

### Exact Metadata Flow And Ownership

The coordinator APIs are changed in one big-bang contract update; no raw
decision fallback remains:

```python
FrontlineEvaluator = Callable[
    [Mapping[str, Any]],
    Awaitable[RelevanceEvaluationEnvelope],
]
SettledEvaluator = Callable[
    [AssessmentLease, Mapping[str, Any]],
    Awaitable[RelevanceEvaluationEnvelope],
]

async def evaluate_frontline(...) -> RelevanceEvaluationEnvelope: ...
async def apply_frontline_decision(
    fragment: PersistedChatFragment,
    envelope: RelevanceEvaluationEnvelope,
) -> FrontlineOutcome: ...
async def evaluate_settled(...) -> RelevanceEvaluationEnvelope: ...
async def apply_settled_decision(
    lease: AssessmentLease,
    envelope: RelevanceEvaluationEnvelope,
) -> SettlementOutcome: ...

@dataclass(frozen=True, slots=True)
class SettlementOutcome:
    ...
    attempt_diagnostics: tuple[EpisodeAttemptDiagnosticV1, ...] = ()
```

`_PendingTurn.attempt_diagnostics` is the mutable bounded list owned by the
coordinator; `AssessmentLease.attempt_diagnostics` is its immutable tuple
snapshot. `SettlementOutcome.attempt_diagnostics` is also an immutable tuple
snapshot, populated only for a current claimable `proceed`. No envelope,
decision, or diagnostic list is stored in a service global or passed through a
second side channel.

Each evaluator callback and `evaluate_*` result is a fully validated envelope:
the outer shape and the nested stage decision are validated before the result
crosses the evaluator boundary. Each `apply_*` call revalidates only
`envelope["decision"]` with the unchanged producer validator and owns metadata
application. The
`SettlementOutcome` gains one `attempt_diagnostics` field containing the
merged bounded rows only for the current claimable `proceed`; all other
outcomes carry an empty tuple. This is the sole service handoff for current
settled diagnostics. Service passes that field separately to
`_process_queued_chat_item`; all response fields come from the nested decision,
and service performs no second merge or cap.

1. `frontline_relevance_agent.py` returns the envelope. The FIFO coordinator
   validates only the nested decision and stores the envelope rows on the
   pending turn through `append_attempt_diagnostics`. A `start` stores rows; a
   valid `append` appends rows after the existing rows. An invalid append
   target or a discard, including append-with-no-valid-target, drops that
   envelope's rows. Collapsed private fragments that are attached without a
   separate relevance evaluation add no rows. The already-recorded protected
   LLM trace remains authoritative for a discard.
2. `_PendingTurn` and `AssessmentLease` in
   `brain_service/turn_settlement.py` carry the bounded frontline rows. The
   settled-state projection exposes the lease rows as metadata without adding
   them to the relevance prompt or changing the settled packet. A stale lease
   never merges its rows into the current version and returns no diagnostic
   handoff.
3. `persona_relevance_agent.py` returns a second envelope containing only its
   settled-stage rows. On a matching lease, `apply_settled_decision` merges
   the lease rows followed by the settled envelope rows exactly once. A
   `wait` stores that merged list on the pending turn; the next lease carries
   it once, and a later settled evaluation appends after it. `ignore` drops
   the carrier with the completed turn. `proceed` returns the current merged
   list once in `SettlementOutcome.attempt_diagnostics`.
4. `_process_settlement_lease` consumes the nested settled decision for the
   existing response fields and passes only the outcome's separate diagnostic
   field to the initial `IMProcessState`. The aggregate enters
   `IMProcessState.attempt_diagnostics` before persona and dialog work. The
   pending-turn, lease, wait, stale-version, deadline, FIFO, and atomic claim
   behavior remains unchanged.
5. `persona_supervisor2_schema.py` declares the reducer-backed field on both
   persona states. The top-level `IMProcessState` already contains the
   inherited relevance rows before persona is invoked. `persona_supervisor2`
   initializes the internal persona graph's `attempt_diagnostics` field to an
   empty list, so the subgraph reducer accumulates only diagnostics newly
   produced by persona and dialog nodes. It returns that normalized downstream
   delta list directly. Inherited rows are not copied into the internal graph;
   there is no positional slicing, equality deduplication, side channel, or
   second cap. The top-level `IMProcessState` reducer is the one and only merge
   and bound operation combining inherited relevance rows with the persona
   delta. Thus two inherited rows plus one persona degradation produce three
   rows once, while 16 inherited rows plus a new terminal row retain inherited
   rows 2 through 16 followed by the new row exactly once.
6. The existing brain graph routing, atomic claim, FIFO settlement worker, and
   post-turn completion remain the same. `brain_service/post_turn.py` already
   reads the final graph result's `attempt_diagnostics`, so a claimable
   `proceed` writes the bounded rows into the existing `episode_trace.v2`
   carrier without a new schema. A settled `ignore` and a frontline
   `discard` have no settled episode and therefore persist no episode
   diagnostic row; their protected relevance traces remain available.

The preflight confirms that `brain_service/graph.py`,
`brain_service/post_turn.py`, `brain_service/cognition_observation_projection.py`,
`action_spec/results.py`, and `runtime_coordination/**` already provide the
required downstream contract or unchanged control flow and do not need edits.
`persona_supervisor2_cognition.py` projects semantic cognition input explicitly
and does not consume this operational metadata, so it also remains outside the
Delivery 5 source set. The existing `relevance/README.md` and package facade
describe raw decision returns; they must be updated with the envelope contract
when the new public type is implemented.

### Expanded Delivery 5 Owned Surface And Role

The strict production and documentation ownership for the approved expansion
is:

- `src/kazusa_ai_chatbot/relevance/contracts.py` (sole definition owner for
  the unchanged `FrontlineDecision` and `SettledRelevanceDecision` types plus
  the new envelope; producer-local duplicate declarations are removed).
- `src/kazusa_ai_chatbot/relevance/__init__.py` and
  `src/kazusa_ai_chatbot/relevance/README.md` (public type export and exact
  caller-contract documentation).
- `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py` and
  `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py` (envelope
  emission; existing decision validation and trace semantics unchanged).
- `src/kazusa_ai_chatbot/brain_service/turn_settlement.py` (envelope
  validation, pending-turn/lease storage, ordered merge, and nested decision
  validation).
- `src/kazusa_ai_chatbot/service.py` (service adapters, settled envelope
  consumption, and the initial graph-state metadata carrier only).
- `src/kazusa_ai_chatbot/state.py` (the sole bounded reducer/normalizer,
  constant, and `IMProcessState` field).
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` and
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` (persona state and
  return propagation).
- `tests/ownership/source_test_impact_manifest.json` and
  `docs/architecture/cognition_observability_icd.md` (strict-source
  ownership rows and documented envelope/disposition flow).

The existing service classification/replay authority remains limited to its
declared Delivery 5 boundary; this amendment adds only the relevance metadata
carrier work above. The prior `Deferred` statement about adding relevance
paths to `source_roots` is superseded only for the explicitly listed Delivery 5
files: the manifest must add explicit file roots and exact entries for the two
relevance agents, the new relevance contract, `relevance/__init__.py`,
`brain_service/turn_settlement.py`, and `state.py`. No broad relevance or
brain-service directory root is authorized.

### Delivery 5 Source-To-Test Matrix Additions

The following exact nodes are the acceptance contract for every newly changed
strict source. Existing Delivery 4 nodes named below are strengthened to assert
the nested decision envelope and the T1-empty/T3-one-row distinction; they do
not create a second semantic path.

| Source path | New or strengthened contract | Exact deterministic node IDs |
|---|---|---|
| `src/kazusa_ai_chatbot/relevance/contracts.py` | sole definitions of unchanged decision types plus the two-key nested envelope; no producer-local duplicates | `tests/test_relevance_turn_settlement.py::test_relevance_exports_canonical_decision_types_without_producer_duplicates`, `tests/test_relevance_turn_settlement.py::test_relevance_evaluation_envelope_has_nested_decision_and_diagnostics_only` |
| `src/kazusa_ai_chatbot/relevance/__init__.py` | package facade re-exports the one canonical decision and envelope types | `tests/test_relevance_turn_settlement.py::test_relevance_exports_canonical_decision_types_without_producer_duplicates`, `tests/test_relevance_turn_settlement.py::test_relevance_evaluation_envelope_has_nested_decision_and_diagnostics_only` |
| `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py` | T3 provider exhaustion emits one nested-decision envelope row; T1 fallback emits no row; validator uses relocated canonical type | `tests/test_frontline_relevance_agent.py::test_frontline_provider_exhaustion_starts_authoritative_turn`, `tests/test_frontline_relevance_agent.py::test_frontline_provider_exhaustion_discards_non_authoritative_turn`, `tests/test_frontline_relevance_agent.py::test_frontline_agent_fails_closed_on_unsupplied_model_slot`, `tests/test_relevance_turn_settlement.py::test_relevance_exports_canonical_decision_types_without_producer_duplicates` |
| `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py` | settled T3 emits one nested-decision envelope row; T1 normalized output emits no row; validator uses relocated canonical type | `tests/test_persona_relevance_agent.py::test_settled_provider_exhaustion_degrades_without_raising`, `tests/test_persona_relevance_agent.py::test_non_authoritative_provider_exhaustion_returns_ignore`, `tests/test_persona_relevance_agent.py::test_relevance_agent_fails_closed_on_invented_evidence_ref`, `tests/test_relevance_turn_settlement.py::test_relevance_exports_canonical_decision_types_without_producer_duplicates` |
| `src/kazusa_ai_chatbot/brain_service/turn_settlement.py` | mandatory envelope APIs, nested-only validators, ordered start/append/settled merge, exact stale/wait/discard lifecycle, FIFO/lease/claim preservation, and most-recent-16 bound | `tests/test_relevance_turn_settlement.py::test_frontline_start_accumulates_relevance_diagnostics_in_order`, `tests/test_relevance_turn_settlement.py::test_frontline_append_accumulates_relevance_diagnostics_in_order`, `tests/test_relevance_turn_settlement.py::test_frontline_append_with_invalid_target_drops_diagnostics`, `tests/test_relevance_turn_settlement.py::test_relevance_validators_receive_only_nested_decision`, `tests/test_relevance_turn_settlement.py::test_settled_relevance_diagnostics_append_after_frontline`, `tests/test_relevance_turn_settlement.py::test_stale_settled_lease_diagnostics_do_not_pollute_current_version`, `tests/test_relevance_turn_settlement.py::test_wait_diagnostics_carry_once_before_next_settled_append`, `tests/test_relevance_turn_settlement.py::test_relevance_diagnostics_are_bounded_to_sixteen_in_occurrence_order`, `tests/test_relevance_turn_settlement.py::test_discarded_turn_keeps_protected_trace_without_episode_diagnostics`, `tests/test_relevance_turn_settlement.py::test_envelope_preserves_fifo_lease_and_claim_control_flow` |
| `src/kazusa_ai_chatbot/service.py` | settled envelope reaches initial graph state and final episode carrier through only `SettlementOutcome.attempt_diagnostics`, without changing settlement control flow | `tests/unit/brain_service/test_cognition_graph_projection.py::test_initial_process_state_seeds_relevance_diagnostics_before_persona`, `tests/test_relevance_turn_settlement.py::test_service_consumes_only_settlement_outcome_diagnostics`, `tests/test_relevance_turn_settlement.py::test_settled_relevance_diagnostics_reach_episode_trace`, `tests/test_relevance_turn_settlement.py::test_discarded_turn_keeps_protected_trace_without_episode_diagnostics` |
| `src/kazusa_ai_chatbot/state.py` | one reducer/normalizer retains the most recent chronological 16 rows, displacing older rows at the bound; the reducer node asserts the latest 16 remain in chronological order | `tests/unit/brain_service/test_cognition_graph_projection.py::test_attempt_diagnostics_reducer_concatenates_within_bound`, `tests/unit/brain_service/test_cognition_graph_projection.py::test_initial_process_state_seeds_relevance_diagnostics_before_persona` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` | both persona states carry the reducer-backed metadata field | `tests/unit/nodes/test_persona_supervisor2_schema.py::test_persona_states_carry_attempt_diagnostics`, `tests/unit/nodes/test_persona_supervisor2_schema.py::test_persona_supervisor_diagnostic_delta_respects_sixteen_row_cap` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | persona graph starts an empty internal diagnostic accumulator and returns only its normalized downstream delta; the top-level reducer merges inherited rows once, and the cap node asserts the oldest inherited row is displaced while the new terminal row is retained exactly once | `tests/unit/nodes/test_persona_supervisor2_schema.py::test_persona_supervisor_returns_attempt_diagnostics`, `tests/unit/nodes/test_persona_supervisor2_schema.py::test_persona_supervisor_returns_only_new_diagnostic_delta`, `tests/unit/nodes/test_persona_supervisor2_schema.py::test_persona_supervisor_diagnostic_delta_respects_sixteen_row_cap`, `tests/test_relevance_turn_settlement.py::test_settled_relevance_diagnostics_reach_episode_trace` |
| `tests/ownership/source_test_impact_manifest.json` | explicit roots and entries cover all expanded strict sources | `tests/test_test_impact_manifest.py::test_manifest_covers_relevance_diagnostic_source_boundary`, `tests/test_test_impact_manifest.py::test_manifest_covers_turn_settlement_diagnostic_source_boundary` |
| `docs/architecture/cognition_observability_icd.md` | relevance T3 error codes, envelope flow, bound, and discard/no-episode rule | `tests/test_cognition_observability_docs.py::test_icd_documents_relevance_diagnostic_envelope_flow` |

The required acceptance assertions are therefore explicit: frontline start
and append retain rows in occurrence order; both unchanged validators receive
only nested decisions; settled rows follow frontline rows; the reducer keeps
the most recent 16 rows in chronological order; a final persona episode
retains two rows and never exceeds 16; inherited 16 rows displace the oldest
row when one new persona terminal row arrives, with that new row retained once;
a discarded turn retains its protected trace but no episode carrier row; and
the existing FIFO, lease, wait, stale-version, claim, and proceed/ignore
routing tests continue to pass unchanged except for envelope fixtures.

### Delivery 5 Preflight Boundary And Status

- Current-code conflicts recorded by this preflight are: the exact decision
  `TypedDict` declarations are still duplicated in the two producer modules;
  both agents return raw/flattened decision mappings rather than one nested
  envelope; `TurnSettlementCoordinator` validates those raw mappings and has
  no diagnostic storage on `_PendingTurn` or `AssessmentLease`; its
  `SettlementOutcome` has no diagnostic handoff; `service.py` seeds no
  relevance diagnostic field into `IMProcessState`; both persona state
  schemas lack the field; and the strict manifest has no entries for
  relevance sources, `turn_settlement.py`, or `state.py`.
- The existing post-turn consumer already reads
  `graph_result["attempt_diagnostics"]`, and the graph/projection/settlement
  control-flow audit found no required source change there. The only
  implementation boundaries requiring deliberate proof are the stale-lease
  no-merge rule, the `SettlementOutcome`-only service handoff, and LangGraph's
  inherited state re-emission: the persona graph must return downstream delta
  rows rather than re-emit its inherited relevance rows, and the exact-two and
  cap-boundary episode tests are the gates against duplication.
- At preflight time, Delivery 5 remained `not started`; implementation, manifest
  edits, ICD edits, and deterministic evidence were pending parent approval of
  this amendment. The implementation evidence below supersedes that preflight
  status. No live-LLM nodes were run in this preflight.

### Delivery 5 Implementation Evidence — 2026-08-27

- The approved D5 implementation was executed by the fixed sole executor
  `/root/live_response_recovery_impl` using `gpt-5.6-luna` at maximum reasoning
  and normal speed. Parent `/root` remains the independent reviewer. No
  production, test, or documentation paths outside the approved D5 ownership
  were added; the manifest also records the already-changed Delivery 3
  `persona_supervisor2_memory_lifecycle.py` source because the repository impact
  validator requires its four exact plan-mapped tests.
- The implementation matches the approved contracts: canonical relevance
  decision types and the exact two-key nested envelope; T1-empty/T3-one-row
  relevance metadata; ordered pending-turn/lease/wait/stale/discard/proceed
  carrier lifecycle; the sole state reducer retaining the most recent
  chronological 16 rows; `SettlementOutcome.attempt_diagnostics` as the only
  service handoff; cognition `model_contract` classification and one
  pre-commit checkpoint replay; and persona's empty-internal-accumulator,
  downstream-delta return. Manifest, README, and ICD records match those
  boundaries.
- Exact D5 mapped deterministic acceptance set: `45` nodes collected and
  passed, including the restored original ICD node
  `test_icd_documents_live_response_recovery_dispositions` and the strengthened
  cognition replay-label assertion. The complete owned deterministic regression
  command before the supplemental ownership extension was:
  `venv\Scripts\python -m pytest tests/test_frontline_relevance_agent.py tests/test_persona_relevance_agent.py tests/test_relevance_turn_settlement.py tests/unit/brain_service/test_cognition_graph_projection.py tests/unit/nodes/test_persona_supervisor2_schema.py tests/test_test_impact_manifest.py tests/test_cognition_observability_docs.py -q`
  Result: `120 passed in 2.69s`.
- Repository impact validation command:
  `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD`
  Result: `91` exact impact-test nodes collected and validated with exit code
  `0`.
- Edited-Python CJK-safe syntax gate passed with `venv\Scripts\python -m
  py_compile` over all nine edited D5 production modules and seven edited D5
  test modules. Canonical import check passed with `D5 imports OK`.
- Read-only scope scans confirm the decision `TypedDict` declarations exist
  only in `relevance/contracts.py`; removed `DialogGenerationContractError`
  and `dialog_generator_exhausted` references are absent from `src/` and
  `tests/`; and the D5 carrier is present only at the declared coordinator,
  service, state, and persona boundaries. `git diff --check` passed for every
  D5-owned path.
- The parent-review remediation changed only the ICD ladder documentation,
  its exact vocabulary test, and the cognition retry log label; no recovery,
  parser, transport, decision, or carrier behavior changed. The ICD now names
  T1/T2/T3/T4, all fixed final statuses and error codes, protected-trace versus
  episode-diagnostic ownership, and the exact pre-commit-only `model_contract`
  replay gate.
- Parent explicitly extended D5 test ownership to
  `tests/test_service_background_consolidation.py`. Its shared
  `_patch_chat_dependencies` relevance doubles now return the exact nested
  two-key envelopes with empty `attempt_diagnostics`; the other direct
  relevance monkeypatch in that file is not on the service producer/coordinator
  boundary and remains semantically unchanged.
- The updated D5 deterministic collection contains `45` required nodes plus
  `1` parent-owned supplemental node (`46` total), including
  `test_icd_documents_live_response_recovery_dispositions`. The required
  `45`-node set passed; the supplemental focused node
  `tests/test_service_background_consolidation.py::test_chat_response_tracks_deliverable_assistant_row`,
  passed (`1 passed in 1.79s`), and the full background-consolidation file was
  collected but could not complete because its existing DB-backed post-turn
  residue path blocked during `test_chat_queues_background_consolidation_for_mapping_state`;
  the same bounded run reached `120` prior passes before the timeout. The
  complete eight-file batch has the same supplemental-file blocker; no
  production or unrelated test behavior was changed to bypass it.
- No live-LLM tests or full-repository test suite were run. Delivery 5 is
  implementation-complete and pending parent independent review; Delivery 6
  live gates and Delivery 7 sign-off remain not started.

### Delivery 6 Live-LLM Evidence — 2026-08-27

- This is the resumed execution after an interruption. No Delivery 6 file
  edits or live nodes had completed before the interruption; the resumed run
  used only the four authorized live-test files and this plan's D5/D6 records.
  No production or deterministic test file was changed, and no raw protected
  trace content is included in this plan record.
- Gate A passed with:
  `venv\Scripts\python -m pytest tests/test_cognition_v3_response_goal_contract_live_llm.py::test_live_captured_p_stage_converges_within_attempt_cap -q -s -m live_llm`
  (`1 passed in 8.41s`). Artifact:
  `test_artifacts/diagnostics/cognition_v3_response_goal_contract_live_llm/response_goal_1787814240138571400.json`.
  The real `COGNITION_V3_CHAIN_LLM` / `gemma4-4090` call produced one
  `succeeded`/`parsed` P attempt and a non-empty validated `response_goal`.
  `recovery_branch_exercised=false` is recorded truthfully; no contract
  recovery was claimed for this direct success.
- Gate B was run with the exact command
  `venv\Scripts\python -m pytest tests/test_dialog_generator_live_llm_contract.py::test_live_dialog_source_url_feedback_converges -q -s -m live_llm`
  three times before the bounded remediation. The first two artifacts
  (`test_artifacts/llm_traces/dialog_generator_live_llm_contract__source_url_feedback_converges.json`
  and the timestamped `__20260827T072815591620Z.json`) each record three real
  `provider_error` attempts caused by the configured `gemma4-4090` route
  consuming its 25,000-token completion limit. The structured-URL artifact
  (`test_artifacts/llm_traces/dialog_generator_live_llm_contract__source_url_feedback_converges__20260827T073230142580Z.json`)
  records three real `contract_error`/`contract_fault` attempts with
  `dialog output messages exceed text bound`; no source-fidelity rejection or
  later accepted candidate was observed. The supplemental existing short-node
  probe was not an acceptance node and failed before model invocation because
  its stale fixture omitted the now-required `epistemic_boundary` field.
- The bounded dialog-only remediation then executed the same exact command
  four additional times after stimulus-only adjustments. This exceeded the
  requested three redesigned-probe allowance by one execution; no further
  exploratory or confirmation model call was started after the fourth
  remediation artifact. The post-remediation artifacts are retained under
  `test_artifacts/llm_traces/` and contain API-key-free model configuration,
  protected per-attempt evidence, parsed summaries, final result, capture mode,
  and judgment:
  - `...__20260827T074333704845Z.json`: URL length 5,498; three
    `contract_error`/`contract_fault` attempts for non-exact output fields;
    `source_rejection_observed=false`,
    `later_acceptance_observed=false`.
  - `...__20260827T074503609850Z.json`: URL length 6,018; one
    `normalized`/`parsed` attempt where T1 appended the required URL within
    the cap; no source-fidelity rejection or later acceptance.
  - `...__20260827T074814101161Z.json`: URL length 10,763; one
    `normalized`/`parsed` attempt where T1 appended the required URL within
    the cap; no source-fidelity rejection or later acceptance.
  - `...__20260827T075236320680Z.json`: URL length 11,218; attempt 1 is
    `contract_error`/`contract_fault` with
    `source_url_fidelity`, while attempts 2 and 3 are
    `contract_error`/`contract_fault` with
    `dialog output messages exceed text bound`; the exact URL was present in
    those over-bound candidates but neither was accepted. The artifact records
    `source_rejection_observed=true` and
    `later_acceptance_observed=false`, with final projection lacking the
    required source token.
- The final stimulus asserted before the model call that the 12,000-character
  cap was respected: the concise repaired candidate measured 11,239
  characters, JSON framing measured 23 characters, and the explicit safety
  margin was 512 characters (total 11,774); the authored-length floor was
  1,000 characters, making URL append exceed the cap. The real route remained
  `DIALOG_GENERATOR_LLM` / `gemma4-4090` with `max_completion_tokens=25000`.
  The strict named dialog feedback gate therefore remains unproven: a real
  source-fidelity rejection was observed, but no later real candidate was
  accepted within the three-attempt cap. D6 remains in progress.
- Parent then authorized one additional exact execution after the bounded
  URL-budget correction, using the same command and no additional probe:
  `venv\Scripts\python -m pytest tests/test_dialog_generator_live_llm_contract.py::test_live_dialog_source_url_feedback_converges -q -s -m live_llm`.
  The artifact
  `test_artifacts/llm_traces/dialog_generator_live_llm_contract__source_url_feedback_converges__20260827T084916481146Z.json`
  records the API-key-free `DIALOG_GENERATOR_LLM` / `gemma4-4090` route and
  one real `normalized`/`parsed` attempt. Its parsed candidate measured
  11,696 characters without the exact 10,820-character URL; T1 appended the
  required URL and returned two final messages totaling 11,690 characters.
  `source_rejection_observed=false` and
  `later_acceptance_observed=false`, so the strict gate failed at its required
  retry-count assertion and no feedback path was exercised.
- The corrected pre-call budget in that artifact is explicit and model-output
  agnostic: first-response reference 1,373 plus one separator plus URL gives
  12,194 (over the 12,000 cap); repair reference 1,149 plus URL gives 11,969,
  with 23 JSON-framing characters giving 11,992; the concise repaired
  candidate plus its 512-character safety margin totals 11,376. No further
  real-LLM call was started after this parent-authorized follow-up. D6 remains
  in progress.
- Parent then authorized up to three serialized repetitions with the unchanged
  10,820-character stimulus. The first repetition passed, so the remaining
  repetitions were not run:
  `venv\Scripts\python -m pytest tests/test_dialog_generator_live_llm_contract.py::test_live_dialog_source_url_feedback_converges -q -s -m live_llm`
  (`1 passed in 114.61s`). The retained artifact is
  `test_artifacts/llm_traces/dialog_generator_live_llm_contract__source_url_feedback_converges__20260827T085300905884Z.json`.
  It records the API-key-free `DIALOG_GENERATOR_LLM` / `gemma4-4090` route,
  `source_rejection_observed=true`, and
  `later_acceptance_observed=true`. Attempt 1 was
  `contract_error`/`contract_fault` with
  `source_url_fidelity`, 1,423 parsed dialog characters, no exact URL, and
  one unexpected URL token. Attempt 2 was `succeeded`/`parsed`, 11,957 parsed
  dialog characters with the exact URL and no unexpected URL, plus the exact
  four-key `contract_repair` block. The final result was one non-empty
  11,944-character message containing the exact URL. No third repetition or
  additional model call was made after this pass; protected raw evidence
  remains in the local artifact and is not reproduced here. D6 is complete
  pending Delivery 7 independent review.
- Gate C initially exposed a live-test assertion mismatch after a real
  `proceed` result correctly skipped the optional second QQ assessment. After
  narrowing that assertion to the existing branch, the exact command
  `venv\Scripts\python -m pytest tests/test_settled_relevance_captured_failure_live_llm.py::test_live_replays_qq_480386272_settled_relevance_failure -q -s -m live_llm`
  passed (`1 passed in 3.62s`). Final artifact:
  `test_artifacts/llm_traces/settled_relevance_captured_failure_live_llm__qq_480386272_message_731909695__20260827T073409053468Z.json`.
  The real `RELEVANCE_AGENT_LLM` / `gemma4-4090` call made one accepted
  `succeeded`/`succeeded` attempt, returned the exact nested two-key envelope,
  and carried `attempt_diagnostics=[]` (the T1/valid no-row rule).
- Gate D passed with:
  `venv\Scripts\python -m pytest tests/test_memory_lifecycle_specialist_live_llm.py::test_live_tiramisu_fulfilled -q -s -m live_llm`
  (`1 passed in 4.16s`). Artifact:
  `test_artifacts/llm_traces/memory_lifecycle_specialist_live_llm__tiramisu_fulfilled.json`.
  The real `COGNITION_LLM` / `gemma4-4090` call made one accepted
  `succeeded`/`succeeded` lifecycle attempt, materialized `unit-tiramisu`, and
  recorded the required case input, API-key-free model configuration,
  attempt/raw/parsed evidence, final result, capture mode, and judgment note.
- All retained artifacts expose only API-key-free configuration metadata in
  this plan. Protected raw/parsed evidence remains in the local artifact
  paths under the configured `full` capture mode and is not reproduced in the
  handoff. The four live-test files pass `py_compile`, their imports pass, and
  `git diff --check` is clean for the authorized live-test and plan paths.
- Parent acceptance record: all four named real-LLM gates pass. The strict
  dialog artifact `...__20260827T085300905884Z.json` proves attempt 1
  `contract_error`/`contract_fault` `source_url_fidelity`, followed by
  attempt 2 `succeeded`/`parsed` with the exact four-key `contract_repair`,
  exact required URL, and terminal visible output. D6 changed no canonical
  JSON parser or repair path. The scoped delivery paths were the four
  authorized live-test files plus this plan and their generated artifacts;
  no production or deterministic-test files changed. The previously
  disclosed redesigned-probe allowance was exceeded by one run; that process
  deviation did not expand file, model, or production scope. D6 is
  parent-accepted, and Delivery 7 is next. No deterministic or full-repository
  suite was run in this delivery.

## Delivery 7 Verification And Independent Review — 2026-08-27

### Verification Evidence

- Baseline and executor state: HEAD was
  `016061fe81ee2af31bfffbebf954c5ac51ec3ecf`; the root virtual environment
  reported Python `3.14.6`; the changed-path inventory remained stable across
  the verification run. The fixed executor was
  `/root/live_response_recovery_impl` using `gpt-5.6-luna` at maximum
  reasoning and normal speed, with parent `/root` retaining independent review
  authority. No unplanned executor changes were introduced.
- Ownership and mapped-node gate:
  `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run`
  collected and ran 91 mapped nodes. 89 passed and 2 failed, with no
  ownership, mapping, or collection error. The unchanged failures were
  `tests/unit/cognition_core_v3/test_prompt_context.py::test_response_content_provider_is_reply_content_fact_across_all_stages`
  and
  `tests/unit/cognition_core_v3/test_prompt_context.py::test_a2_existential_drive_keeps_character_experience_distinct_from_user_state`.
  Both directly exercise unchanged `cognition_core_v3/prompt.py` contracts
  and remain outside this plan's explicit Keep boundary.
- Unit gate:
  `venv\Scripts\python -m pytest tests/unit -q` produced 155 passed and the
  same 2 unchanged prompt-context failures in 6.08s.
- Exact integration gate:
  `venv\Scripts\python -m pytest tests/test_frontline_relevance_agent.py tests/test_persona_relevance_agent.py tests/test_memory_lifecycle_specialist.py tests/test_local_context_resolver_recovery.py tests/test_local_context_resolver_standalone.py tests/test_cognition_observability_docs.py tests/test_test_impact_manifest.py -q`
  produced 108 passed in 0.96s.
- Absence gate:
  `venv\Scripts\python -m pytest tests/unit/nodes/test_dialog_agent.py -q`
  produced 12 passed in 0.72s before the final dialog remediation and 12
  passed in 0.76s after it. The repository search for
  `DialogGenerationContractError|dialog_generator_exhausted` under `src/`
  and `tests/` returned zero matches.
- Full regression gate:
  `venv\Scripts\python -m pytest -q` stopped during collection with two
  unrelated missing test-module imports:
  `tests.test_asuna_private_r18_affinity_live_llm` and
  `tests.test_group_style_output_shape_live_llm`. The result was 2 collection
  errors, 1 skipped, 616 deselected in 2.58s. No plan-owned node failure was
  exposed by this run; full regression execution is incomplete because of
  these repository collection issues.
- Live-LLM gates were run individually and inspected as recorded in Delivery 6:
  the cognition gate passed in 8.41s, the strict dialog feedback gate passed
  in 114.61s with the retained artifact
  `test_artifacts/llm_traces/dialog_generator_live_llm_contract__source_url_feedback_converges__20260827T085300905884Z.json`,
  the settled QQ gate passed in 3.62s, and the memory gate passed in 4.16s.
  The existing artifact records remain the protected evidence source; their
  raw bodies are not copied into this plan.
- Final integrity: `git diff --check` passed with line-ending warnings only.
  The worktree path inventory remained stable except for the reviewed
  `nodes/dialog_agent.py` remediation and this authorized plan lifecycle move.

### Independent Acceptance-Criterion Verdict

| # | Verdict | Review conclusion |
|---|---|---|
| 1 | PASS | The mapped deterministic owner matrix and live gates cover the assigned ladder tiers; 89 of 91 mapped nodes passed, with the two unrelated unchanged prompt-context failures qualified under criterion 16. |
| 2 | PASS | Cognition feedback carries the validator error and rejected raw candidate in the exact bounded repair block. |
| 3 | PASS | The P object-to-string recovery path produces a validated completed plan with truthful two-attempt trace records. |
| 4 | PASS | Rejected products are recorded as contract faults; accepted products alone use parsed/succeeded dispositions. |
| 5 | PASS | Cognition contract exhaustion maps to public `model_contract` and permits exactly one pre-state-commit replay. |
| 6 | PASS | Turn-deadline exhaustion is non-retryable and does not replay. |
| 7 | PASS | Dialog remains non-empty across the bounded provider, structural, source-fidelity, and mixed-failure paths. |
| 8 | PASS | URL normalization, T1 trace dispositions, retained/projection degradation diagnostics, and the final authored-text boundary are aligned; the reviewed URL-only replacement branch is removed. |
| 9 | PASS | Oversized visible percepts are bounded by the URL scan without failing dialog. |
| 10 | PASS | Non-cancellation visual failures omit only the visual result and preserve text/dialog with the fixed diagnostic. |
| 11 | PASS | Memory provider/contract exhaustion skips lifecycle context while preserving non-lifecycle action specifications. |
| 12 | PASS | All four local-context exhaustion paths degrade to their assigned products and contain provider failures. |
| 13 | PASS | Frontline contract-invalid output remains one-call T1 normalization; provider exhaustion preserves authoritative start/discard behavior. |
| 14 | PASS | Settled authoritative repair exhaustion uses an available deterministic disposition and preserves the typed failure when none exists. |
| 15 | PASS | Ordered relevance and downstream diagnostics are additive and bounded to the latest 16 rows in the episode path. |
| 16 | PASS with qualification | All 91 mapped nodes were collected and ran with no ownership/mapping/collection error; the two unchanged prompt-context nodes failed because of contracts outside this plan's Keep boundary. |
| 17 | PASS | The ICD documents T1/T2/T3/T4, fixed dispositions/error codes, protected-trace versus episode ownership, and the pre-commit replay rule. |
| 18 | PASS | Independent review found no compatibility shim, alias, feature flag, semantic evaluator, route/attempt-limit change, or second replay. |

### Independent Review Conclusions

- The canonical `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` entry
  point and existing repair path are reused and unchanged; no stage-local JSON
  parser or repairer was introduced.
- Recovery is limited to structural/content-contract handling. No semantic
  scoring, sibling salvage, goal-bid exhaustion, or unavailable-goal state was
  introduced.
- Deterministic internal invariants still fail closed; optional/model failures
  alone use the approved degradation paths.
- Service replay is exactly one safe pre-state-commit replay; deadline and
  post-commit paths do not replay.
- Diagnostics are ordered, additive, and bounded, and the relevance envelope
  scope expansion follows the approved amendment.
- All 19 changed production source paths and governed documentation/manifest
  paths are within the approved and amended change surface.

### Review Finding, Remediation, And Re-review

- Finding: `_normalize_dialog_source_urls` accepted a URL-only visible result
  after unsupported-URL removal by replacing an empty dialog with the first
  required URL, outside the approved dialog terminal contract.
- Remediation by Luna: removed that empty-dialog URL replacement and required
  a non-empty normalized last message before the allowed-URL append path, only
  in `src/kazusa_ai_chatbot/nodes/dialog_agent.py`.
- Re-review: PASS. The UTF-8 AST check passed; the dialog unit file passed 12
  tests; the impact result remained 89/91 with the same two unrelated failures;
  the removed-symbol search returned zero matches; and `git diff --check`
  passed.

### Residual Risks Accepted For Plan-Scope Closure

- The two unchanged prompt-context unit mismatches remain outside scope.
- Full-suite execution is incomplete because of the two unrelated missing test
  modules.
- The URL-only remediation has parent code review and the existing 12-test
  dialog suite; no new deterministic test was added under the user's
  restriction.
- The previously disclosed one-run redesigned-probe allowance overrun remains
  a process deviation without file, model, or production scope expansion.
- Live dialog recovery remains model/stimulus dependent even though the strict
  required artifact passed.

These residuals are accepted for this plan-scope closure and remain separate
work. No fix is implied or authorized by this closeout.

Parent `/root` independent plan-scope verdict: PASS. Delivery 7 is complete,
the plan is closed, and the archived record is historical; any residual
repository failures remain separately tracked rather than silently expanded
into this plan.

## Independent Plan Review

Required before approval, because this plan changes a completed contract
(`cognition_v3_handleless_model_contract_bigbang_plan_2026-08-22.md` removed
semantic retries) and deletes a service-recognized failure class. The reviewer
confirms:

- The restored cognition regeneration is structural only and reintroduces no
  sibling salvage, goal-bid exhaustion, unavailable-goal state, or semantic
  scoring.
- Every T1 normalization named here adds no semantics.
- No degradation path can mask a deterministic internal-invariant bug.
- The worst-case cognition wall clock is bounded and stated.
- The traceability matrix is complete for every changed source path.

## Independent Code Review

Authority: pass or fail each acceptance criterion, and require remediation; may
not remediate. Independence: must not be an executor of any implementation role.
Evidence required for closure: the baseline diff, the ownership-gate output,
every mapped deterministic node result, the four live artifacts, and a written
verdict with a residual-risk list.
