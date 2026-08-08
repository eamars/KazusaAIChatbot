# august change alignment audit and remediation plan

## Summary

- Goal: review every repository change made from 2026-08-01 through the
  execution handoff, including completed documented plans, active plan work,
  undocumented change clusters, and the pre-existing uncommitted worktree
  change; remediate code, tests, and documentation until the repository matches
  the project direction and its ownership contracts.
- Execution owner: gpt5.6 sol, with the implementation mechanics left to the
  execution owner inside this fixed scope.
- Status: completed
- Scope boundary: the tracked commit range beginning at 2026-08-01 00:00
  Pacific/Auckland time through the approved execution baseline, the working
  tree at that baseline, and older code or tests that remain relevant to an
  affected contract or failure.
- Change direction: build a complete change-to-contract matrix, inspect every
  affected boundary, repair the smallest owning surface when behavior or
  quality is misaligned, and leave an evidence-backed disposition for every
  failure.
- End goal: every in-scope change has an architecture and plan verdict, every
  required code/test/documentation repair is applied, and every remaining test
  or environment failure is fixed or explicitly justified with evidence.
- Final comparison: after remediation and verification, the parent execution
  owner performs a read-only comparison of the pinned baseline and Cognition
  V2 branch, producing
  `test_artifacts/diagnostics/cognition_v2_baseline_feature_regression_map.md`
  with missing features and legacy configuration references highlighted for
  user decision.
- Acceptance state: completed and signed off on 2026-08-08 against the pinned
  execution baseline, exhaustive audit artifact, regression map, and recorded
  deterministic/live evidence.

## Scope And Change Direction

This plan is a repository-wide alignment audit and bounded remediation pass.
The execution owner reviews the actual diff and runtime behavior rather than
accepting commit messages, plan status, or green test assertions as proof of
alignment.

The audit covers:

- all commits from the fixed 2026-08-01 cutoff through the execution handoff;
- all tracked and untracked worktree changes present at the handoff;
- the committed `tests/test_e2e_live_llm.py` addition at the execution HEAD,
  which remains part of the audit and receives the same review;
- all 22 completed plan records during the window;
- the active Cognition Core V2 context-fade/sleep work and other older active
  code whose contracts are exercised by the audited changes;
- undocumented production, test, documentation, configuration, console,
  script, fixture, and agent-workflow changes in the same window;
- relevant pre-cutoff callers, schemas, persistence code, prompts, tests, and
  documentation required to establish the affected contract.

The target architecture remains:

    adapter/debug client -> brain service -> queue/intake -> relevance/evidence
    -> cognition -> dialog/surfaces -> persistence/consolidation/scheduler

The audit applies these ownership rules:

- adapters normalize platform events and render returned surfaces;
- brain service and queue/intake own typed ingress, ordering, deadlines, and
  delivery receipts;
- relevance decides interaction admission and settlement semantics;
- RAG and task-resolution specialists return evidence or bounded task results;
- cognition owns stance, boundaries, character judgment, action need, and
  response goals;
- dialog and L3 surfaces own final visible wording and rendering;
- deterministic code owns validation, persistence, permissions, limits,
  scheduling, cache invalidation, and adapter delivery feasibility;
- background work owns durable jobs, leases, worker execution, and result-ready
  handoff without directly invoking adapters or shared cognition;
- reflection and consolidation maintain continuity outside the live wording
  path;
- the Control Console remains a separate operational surface and does not own
  brain semantics.

The execution owner updates code when it is misaligned with these contracts or
when the implementation quality is poor. The execution owner updates tests
when their assertions, fixtures, taxonomy, or coverage are deprecated,
misaligned, brittle, or unable to prove the current contract. Documentation is
updated when it contradicts the accepted runtime behavior or leaves an audited
boundary undocumented.

## Confirmed Decisions

- The audit cutoff is 2026-08-01 00:00:00 +1200. The approved execution
  baseline fixes the end commit, worktree state, and plan-registry snapshot.
- gpt5.6 sol owns the complete audit, remediation, verification, evidence, and
  closeout within this plan. After verification and before closeout, the same
  parent execution owner performs the scoped read-only comparison defined
  below. That comparison produces evidence and does not authorize runtime,
  test, or configuration edits.
- The owner resolves ordinary uncertainty from repository evidence and records
  external limitations with the command, error, affected contract, attempted
  recovery, and remaining risk. The owner continues the audit across all
  remaining in-scope items.
- Existing user changes are preserved as baseline input. At approval the
  worktree contains the modified plan registry and this untracked plan; the
  `tests/test_e2e_live_llm.py` addition is already committed at the execution
  HEAD and remains in the audit inventory.
- The approved baseline is the scope boundary. Any unrelated concurrent change
  discovered after the baseline is recorded separately and excluded from
  completion claims until it receives an explicit review row.
- Code changes are made only within the inventory-defined audit surface and
  directly relevant older callers or tests. New product features, migrations,
  compatibility shims, and speculative abstractions remain outside the audit.
- During the main audit, `.env` and other secret-bearing configuration remain
  unread. At the final regression-mapping gate only, the user-authorized parent
  execution owner may inspect `.env` read-only to enumerate legacy
  feature/configuration references. Secret values remain redacted and never
  enter the generated artifact.
- The user's 2026-08-08 parent-only execution directive supersedes earlier
  delegation language: no subagent participates and no execution-time question
  is required. The parent execution owner makes the bounded architectural and
  remediation decisions from the approved plans and repository evidence.
- Live LLM tests run one case at a time and receive individual output
  inspection. Deterministic tests run in batches. Live database tests run only
  through their explicit project markers when MongoDB is available.
- A passing parser, schema check, or pytest command proves only the mechanical
  gate. LLM behavior receives a human-readable review grounded in captured
  input, raw output, parsed output, route, state, and contract evidence.

## Documented Execution Records

The following completed records are in the audit set. Their archived plan
content, execution evidence, implementation diff, and tests are reviewed
together rather than treated as independent proof:

### Completed 1 Aug

- development_plans/archive/completed/bugfix/relevance_evidence_grounded_admission_over_sensitivity_bugfix_plan.md
- development_plans/archive/completed/bugfix/conversation_progress_v2_final_signoff_plan.md
- development_plans/archive/completed/short_term/unified_task_resolution_orchestrator_bigbang_plan.md

### Completed 2 Aug

- development_plans/archive/completed/bugfix/cognition_core_v2_semantic_appraisal_partial_failure_mitigation_plan.md
- development_plans/archive/completed/bugfix/cognition_goal_capability_and_workspace_relevance_bugfix_plan.md
- development_plans/archive/completed/bugfix/character_identity_growth_contract_recovery_bugfix_plan.md
- development_plans/archive/completed/bugfix/qq_group_public_scene_response_ordering_bugfix_plan.md

### Completed 3 Aug

- development_plans/archive/completed/bugfix/cognition_core_v2_short_horizon_global_state_composition_bigbang_plan.md
- development_plans/archive/completed/bugfix/control_console_web_availability_followup_plan.md

### Completed 4 Aug

- development_plans/archive/completed/bugfix/cognition_core_v2_first_pass_robustness_bugfix_plan.md
- development_plans/archive/completed/bugfix/cognition_core_v2_relational_authority_transfer_bugfix_plan.md
- development_plans/archive/completed/bugfix/cognition_core_v2_relational_willingness_gradient_bugfix_plan.md

### Completed 5 Aug

- development_plans/archive/completed/bugfix/cognition_core_v2_generation_contract_prompt_projection_bugfix_plan.md
- development_plans/archive/completed/bugfix/background_coding_event_loop_starvation_bugfix_plan.md
- development_plans/archive/completed/bugfix/background_tool_result_delivery_current_episode_evidence_bugfix_plan.md
- development_plans/archive/completed/bugfix/relevance_native_reply_monotonic_delivery_plan.md
- development_plans/archive/completed/bugfix/relevance_native_reply_review_remediation_plan.md

### Completed 6 Aug

- development_plans/archive/completed/bugfix/cognition_size_limit_truncation_and_fallback_scan_plan.md
- development_plans/archive/completed/bugfix/durable_ingress_native_reply_intervening_message_bugfix_plan.md

### Completed 7 Aug

- development_plans/archive/completed/bugfix/cognition_core_v2_action_planning_request_fidelity_bugfix_plan.md
- development_plans/archive/completed/short_term/task_resolution_character_background_handoff_plan.md
- development_plans/archive/completed/bugfix/background_task_result_blocker_detail_delivery_bugfix_plan.md

The following active records are relevant context and receive contract review
where their code is in the audit radius:

- development_plans/active/bugfix/cognition_core_v2_context_fade_and_sleep_phase_plan.md
- development_plans/active/short_term/coding_agent_assessment_gap_phase_d_plan.md

## Undocumented Change Inventory

The initial history scan identified these change clusters without a completed
plan record directly attached to the commit. The execution owner verifies this
inventory against the handoff snapshot and assigns each cluster a review ID in
the audit matrix:

| Commit | Change cluster | Required review boundary |
| --- | --- | --- |
| faeb1fe9 | Degraded Cognition Core V2 bids reaching dialog | Cognition failure disposition, dialog ownership, surface preservation |
| c738c8bb | Forced default model removal | Configuration ownership, route completeness, clean startup behavior |
| e6e0f7cb | Control Console conversation-progress loader contract | Console API projection and stale-schema handling |
| 1dc655f5 | Resolver search error correction | Capability ownership, error propagation, task-resolution evidence |
| 3dbb8f0e | LLM completion fallback raised to 25,000 tokens | Budget ownership, local-model latency, prompt-budget contracts |
| fe1dbf7d | Web-search behavior correction | Web-agent source boundary, specialist routing, result evidence |
| 7fd2fa60 | Current-event grounding test addition | Test intent, fixture grounding, live/deterministic taxonomy |
| 3be764f9 | Short-horizon focused-test freeze | Test freeze validity and plan acceptance evidence |
| 7f31963a | Bounded repair routine correction | Contract repair ownership, retry bounds, semantic preservation |
| 483d9562 | Development-plan skill workflow modernization | Agent governance, plan lifecycle, execution-authority consistency |
| 6a6b5a9 | Cognition trace-failure test rebuild | Trace evidence quality, failure-case coverage, live-test reviewability |
| 662cbf8f | Cognition Core V2 fix attempts | Prompt contract, semantic appraisal, test and evidence alignment |
| f5831c66 | Relevance behavior correction | Evidence-grounded admission and settled-response ownership |
| 423f6573 | `tests/test_e2e_live_llm.py` third-party-history rejection addition | Public /chat to local-context RAG routing, trace quality, live-test contract |
| worktree baseline | Plan-registry row and this audit plan | Lifecycle authority, approved scope, baseline fidelity |

The agent also reviews commits that changed only plans, registry files, or
merge metadata because lifecycle state and implementation evidence must agree.

## Mandatory Skills

The execution agent reads and applies these skills before changing the governed
surface:

- development-plan for the change contract, checkpoints, evidence, and
  lifecycle closeout;
- local-llm-architecture for bounded local-model design, semantic ownership,
  prompt context, routing, and failure recovery;
- no-prepost-user-input for accepted commands, preferences, permissions,
  commitments, task resolution, and post-LLM semantic passthrough;
- py-style, including both positive and negative constraint references, for
  every Python file reviewed or edited;
- cjk-safety for Python source or tests containing CJK prompt/input content;
- test-style-and-execution for test review, changes, and all test runs;
- debug-llm for live/local LLM evidence, human-readable quality review, and
  trace artifact handling;
- control-console-web-development for Control Console API/static frontend
  review, browser validation, and stale-asset handling when that surface is
  touched.

The agent uses the project virtual environment at venv/Scripts/python and
reads the relevant subsystem ICD before changing its ownership boundary.

## Execution Guardrails

- Keep the work within the approved baseline and the inventory-defined audit
  surface. The owner may repair any affected code, test, fixture, or
  documentation that is required to reach the end goal.
- Preserve the project’s LLM-first semantic ownership. Deterministic code may
  validate, normalize structurally, persist, authorize, limit, schedule,
  invalidate, and deliver; it does not infer or rewrite the model’s semantic
  judgment to make a test pass.
- Route malformed structured output through the canonical
  kazusa_ai_chatbot.utils.parse_llm_json_output(...) boundary before semantic
  evaluation. Preserve raw supported keys and values; keep repair and bounded
  regeneration at the producing semantic stage.
- Keep RAG and local-context output as evidence. Keep cognition responsible
  for stance, boundaries, character judgment, and response goals. Keep dialog
  responsible for wording and visible rendering.
- Keep accepted-task and background-work persistence deterministic and typed.
  Keep worker delivery behind the brain/delivery boundary and preserve
  source, target, permission, idempotency, and receipt checks.
- Keep the Control Console separate from brain semantics, use bounded redacted
  projections, and validate changed static surfaces in a fresh and relevant
  browser state when applicable.
- Preserve test taxonomy: deterministic tests prove deterministic contracts,
  patched LLM tests prove graph handoffs, and real LLM tests prove model-facing
  behavior with individually inspected evidence.
- Keep failures visible. A test is removed, skipped, xfailed, weakened, or
  reclassified only when the agent records the obsolete contract or external
  condition and adds or preserves the current contract coverage.
- Keep historical plan records and raw evidence intact. Correct registry or
  documentation drift through the current plan and its audit artifact.
- Keep Python prompt constants and rendering aligned with the project prompt
  rules, and use explicit UTF-8 handling for CJK-bearing files and artifacts.

## Must Do

1. Establish the approved baseline and regenerate the complete tracked,
   untracked, documented, and undocumented change inventory.
2. Map every changed path to a completed plan, active dependency, undocumented
   cluster, or lifecycle-only record, with an owner, contract, evidence source,
   and final disposition.
3. Compare each affected implementation and test family with the root
   architecture, affected subsystem ICDs, and the referenced plan contract.
4. After the architecture and downstream-plan gates are recorded, progressively
   remediate misaligned or poor code, deprecated or misleading tests, and
   contradictory documentation at the smallest owning boundary.
5. Preserve semantic ownership, bounded local-model behavior, deterministic
   validation, persistence, permissions, limits, delivery, and redaction
   contracts throughout remediation.
6. Verify each repaired cluster, then verify the affected suites and the final
   repository-wide deterministic surface. Inspect changed live-LLM evidence
   individually and cover required live-DB or console surfaces.
7. Produce the human-readable audit review at
   test_artifacts/diagnostics/august_change_alignment_review.md, including the
   full matrix, repairs, quality judgments, failure dispositions, and residual
   risks.
8. At the final workflow gate, pin the baseline and Cognition V2 refs and
   perform the parent-owned read-only regression mapping.
9. Review the generated map, incorporate its evidence and decision queue into
   the audit record, then update this plan’s checklist and execution evidence
   and close the plan only when the end goal and acceptance criteria are
   evidenced.

## Required Execution Order

The plan remains a draft until the registry status is promoted to
`approved` or `in_progress`. After promotion, execute the work in this order:

1. Freeze the approved baseline: commit range, worktree state, registry
   snapshot, secret-read boundary, and concurrent-change classification.
2. Regenerate the complete change inventory and map every path to an owner,
   contract, evidence source, and plan or undocumented review row.
3. Read the root architecture, affected subsystem ICDs, active dependent
   plans, and relevant pre-cutoff callers/tests. Capture protected traces and
   classify the observed failure before choosing a repair.
4. Complete `COGNITION-V2-BID-EXHAUST-ARCH`: define the owner boundary,
   monotonic attempt ledger, branch recovery matrix, capture mode, and typed
   terminal dispositions. Record the budget matrix as an execution gate.
5. Amend and separately approve the downstream
   `required_selection_partial_recovery_bugfix_plan.md` so its local
   producer-validation and branch-recovery work matches the architecture
   decision.
6. Implement production changes at the owning boundaries first, then update
   deterministic, patched-LLM, and real-LLM tests to prove the resulting
   contract. Keep test edits from manufacturing recovery.
7. Reconcile subsystem documentation and the audit matrix, then run focused
   deterministic checks, broader non-live checks, and individually inspected
   live-LLM checks. Run live-DB and console checks only for audited surfaces.
8. Author the human-readable audit review and disposition every failure and
   residual risk with the completed verification evidence.
9. Pin explicit `baseline_ref` and `cognition_v2_ref` values, perform the
   parent-owned comparison below, inspect the generated regression map, and
   record every missing or changed feature and every legacy `.env` config
   reference for user decision.
10. Perform final self-review from the approved baseline and update
    checklist/status/registry together at closeout.

## Final Parent-Owned Regression Mapping

This comparison occurs after the audited repairs and verification are complete
and immediately before final plan closeout. It is a read-only evidence task.
The parent execution owner maps migration regressions separately from any
future implementation decision.

The parent execution owner supplies explicit comparison refs and records their
resolved commits so the handoff uses immutable comparison inputs:

- `baseline_ref`: the approved pre-Cognition-V2 baseline branch or commit,
  together with its resolved commit;
- `cognition_v2_ref`: the approved Cognition V2 branch or commit, together with
  its resolved commit;
- the final audit matrix, relevant architecture/ICD records, affected source,
  tests, documentation, and captured diagnostic artifacts.

The handoff request must state that the agent must not infer either ref from
the current checkout, branch name, or commit history. The parent records both
refs and their resolved commits in the plan and in the generated artifact
before the agent starts.

The parent execution owner performs the following read-only work:

- inventory the migration-relevant features present in the baseline;
- locate the corresponding Cognition V2 implementation, route, graph stage,
  prompt/model configuration, persistence path, adapter path, test coverage,
  and documentation;
- classify each baseline feature as `present_equivalent`,
  `present_behavior_changed`, `missing_in_v2`, `renamed_or_moved`,
  `legacy_config_unmapped`, or `indeterminate_requires_decision`;
- identify regressions separately from intentional behavior changes and
  explicitly cite the source, test, plan, or trace evidence for each row;
- inspect `.env` read-only as an authorized input source, using a
  value-suppressing method, and classify every feature-related legacy key by
  its legacy target, Cognition V2 mapping, and status. Record key names and
  redacted status only; never copy values, credentials, tokens, connection
  strings, or other secret material into the report, logs, or messages;
- make no changes to production code, tests, `.env`, or other configuration as
  part of this comparison. Any proposed repair becomes a separately authorized
  decision or plan after the user reviews the evidence.

The required output is
`test_artifacts/diagnostics/cognition_v2_baseline_feature_regression_map.md`.
It must be readable without protected raw traces or `.env` values and contain:

- the pinned refs, comparison date, scope, and method;
- a complete feature matrix with feature ID, baseline location and behavior,
  Cognition V2 location and behavior, classification, evidence references,
  related `.env` key names, and decision owner;
- a separately highlighted `Migration Regressions` section covering every
  `missing_in_v2` and unintended `present_behavior_changed` row;
- a separately highlighted `Legacy .env Configuration Requiring Decision`
  section covering every legacy key that points to an unmapped or uncertain
  feature, with values redacted;
- evidence limitations, confidence, and a concise list of user decisions
  needed before any remediation plan is authorized.

Handoff acceptance:

- both comparison refs are pinned and recorded before the handoff runs;
- every scoped baseline feature has a classification and an evidence pointer;
- missing, behavior-changed, renamed/moved, and legacy-unmapped items are
  separately visible rather than collapsed into a generic pass/fail result;
- every feature-related legacy `.env` key is classified or explicitly marked
  indeterminate, and no secret value appears in the artifact;
- the comparison produces no source, test, or configuration diff;
- the parent execution owner reviews and links the artifact from this plan;
  any implementation decision is deferred until the user reviews the map.

## Deferred

- Leave new product capabilities, unrelated refactors, new storage schemas,
  database migrations, model-route redesigns, and adapter feature work for a
  separate approved plan.
- Leave broad pre-August cleanup untouched unless it is a direct caller,
  contract owner, test dependency, or failure source for an audited change.
- Leave generated traces, screenshots, and raw live outputs outside committed
  source unless an existing project artifact contract requires them; link them
  from the human-readable review instead.
- Leave `.env` values, API keys, credentials, and unbounded private data outside
  the audit evidence. The final comparison may record legacy configuration key
  names, target features, redacted status, and evidence references only.
- Leave active plans in their current lifecycle state unless this audit
  produces the evidence required for a separate lifecycle update; this plan
  records alignment and remediation, not automatic archival of other plans.

## Target State

At closeout, the repository has one inspectable alignment record for every
change in the audit set. Each record contains:

    change_id
    commit_or_worktree_path
    documented_plan_or_undocumented_cluster
    owning_boundary
    project_contract
    observed_behavior
    architecture_verdict
    plan_verdict
    code_verdict
    test_verdict
    repair_or_justification
    verification_evidence
    residual_risk

The runtime has a single semantic owner for each decision. Deterministic
checks reject invalid structure and unsafe operations without overriding valid
model meaning. Test assertions reflect the current contract and remain
inspectable. Every remaining failure has a concrete justification, evidence,
scope owner, and next action; no failure is concealed by a blanket skip,
weakened assertion, or unrecorded environment assumption.

## Contracts And Data Shapes

The audit uses these contract checkpoints:

- MessageEnvelope is the platform-neutral inbound boundary; brain code
  consumes typed fields rather than raw Discord, QQ, or debug-wire syntax.
- Relevance and turn settlement own admission, response ownership, deadlines,
  and native-reply base semantics. Delivery may apply only the documented
  deterministic promotion rules.
- Local context, RAG3, web specialists, and task-resolution specialists emit
  bounded evidence or typed task results. They do not set persona stance or
  write final visible dialog.
- Cognition Core V2 stages retain their declared route, prompt, input, output,
  parser, retry, and validation ownership. Required-selection and action
  planning preserve complete model-authored semantics and bounded recovery.
- Action specs select only registered deterministic capabilities and preserve
  the accepted-task/background-work contract. They do not become a generic
  background router or specialist chooser.
- Accepted tasks and background jobs preserve typed lifecycle states,
  source/target identity, result provenance, idempotency, and delivery receipt
  semantics.
- Conversation progress, sleep phase, reflection, self-cognition, and
  character growth preserve their separate temporal, persistence, and
  semantic ownership boundaries.
- Protected LLM traces and event logs remain redacted, bounded, separately
  governed, and non-authoritative for normal cognition.
- Control Console projections remain bounded, redacted, authenticated where
  required, and separate from brain service semantics.

## Architecture Follow-up Task: Cognition Core V2 Bid-Exhaust Recovery

This task is captured from protected traces during the audit. It records the
architecture work to perform; this capture pass does not implement the
runtime or test fix.

Task ID: `COGNITION-V2-BID-EXHAUST-ARCH`

Dependency gate:

- `development_plans/archive/completed/bugfix/required_selection_partial_recovery_bugfix_plan.md`
  was executed downstream of this architecture task. Its strict candidate
  contract, shared retry ledger, branch disposition, protected-capture
  contract, and sibling-bid recovery are complete.
- Before implementation, the execution record must contain one budget matrix
  covering local producer attempts, service-graph attempts, cumulative model
  calls, and the disposition at each boundary. The matrix must state whether
  a clean graph retry can reuse remaining producer budget; the implementation
  must not infer that relationship from two independent counters.

Task:

- Establish one monotonic attempt ledger across goal-local model attempts and
  service-graph retries. A graph retry must consume the same invocation
  budget and must not multiply the producing stage's bounded attempt limit.
  Record the cognition invocation, graph attempt, branch, producing stage,
  local attempt, cumulative producer attempts, configured limit, and final
  branch disposition in the protected diagnostic record.
- Define and implement an architecture-owned failure disposition for goal
  cognition when a required-selection bid reaches its bounded attempt limit.
  The disposition must distinguish a branch-local contract failure from a
  whole-cognition failure, explicitly apply the required/optional branch
  policy, and prevent workspace collapse or dialog from consuming an
  invalid or incomplete bid.
- Keep candidate validation, repair feedback, and bounded regeneration owned
  by the goal-cognition stage. Keep orchestration responsible for branch
  continuation, required-branch escalation, and the typed terminal result.
  Preserve the rule that deterministic code validates structure and limits but
  does not rewrite the model's semantic selection.
- Trace the failure boundary with enough protected evidence to identify the
  rejected field: branch, stage, route/model, attempt index, parse status,
  validation error, redacted raw response, parsed candidate, retry feedback,
  and final branch/cognition disposition. Metadata-only bid-exhaust exports
  are insufficient for root-cause attribution. Use the existing authorized
  full protected failure-capsule path for this evidence; keep ordinary
  metadata capture raw-free and keep protected content out of event logs,
  public responses, and operational status.
- Reconcile evidence provenance and semantic pairing failures as separate
  contract classes, including unauthorized evidence handles, exact selection
  field drift, relational-willingness state/stance mismatches, question-owned
  semantic delta paths, invalid semantic-value bounds, and resolved knowledge
  gap transitions. The repair design must preserve evidence ownership rather
  than silently broaden permitted handles or paths.
- Add or update individually inspected real-LLM replay coverage only after
  the architecture decision is approved. The coverage must prove bounded
  attempts, explicit accepted/degraded/exhausted disposition, sibling-branch
  handling, and preservation of a valid downstream boundary.

Architecture acceptance for this task:

- The owning boundary and branch-level recovery matrix are documented before
  implementation.
- The local-attempt/service-graph-attempt budget matrix and its monotonic
  ledger invariant are recorded before implementation.
- The downstream required-selection plan is amended to consume this decision
  before either plan authorizes production changes.
- A bid-exhaust trace contains the rejected candidate and contract evidence,
  or records the exact protected-capture limitation and residual risk.
- No invalid bid reaches workspace collapse, action planning, dialog, or
  persistence; no test-only assertion manufactures recovery.
- A real-LLM replay demonstrates the same production-shaped input through the
  current goal owner and records the final typed disposition.

### Approved Architecture Decision — 2026-08-08

Ownership decision:

- goal cognition remains the only semantic owner of bids, complete
  regeneration, repair feedback, and candidate validation;
- orchestration owns required/optional branch continuation and typed
  escalation;
- the service owns the clean graph retry, but it cannot reset a producer's
  invocation budget;
- the protected failure capsule owns exact failure evidence. Ordinary trace,
  event-log, response, and status surfaces remain raw-free.

Monotonic budget invariant:

- one ledger is bound to the full cognition invocation before graph attempt
  one and reused by graph attempt two;
- its producer key is `(goal_bid_structure, branch_id)` and its configured
  limit remains three;
- a call is consumed immediately before model invocation, including provider
  failures;
- a clean graph retry may use only calls left under the same producer key;
- exhaustion after cumulative call three is non-retryable and makes no call in
  a later graph attempt.

| Scenario | Graph attempt 1 | Graph attempt 2 | Cumulative goal calls per branch | Disposition |
| --- | ---: | ---: | ---: | --- |
| Goal succeeds on call one; a later retryable owner fails | 1 | up to 2 remaining | 3 maximum | Later owner succeeds or fails under its own policy. |
| Goal succeeds on call two; a later retryable owner fails | 2 | up to 1 remaining | 3 maximum | Later owner succeeds or fails under its own policy. |
| Goal structure/provider exhausts | 3 | 0 | 3 | Typed non-retryable branch exhaustion. |
| Direct facade invocation | up to 3 | not applicable | 3 maximum | Accepted or typed exhaustion. |

Candidate-contract decision:

- remove unused `selection_kind` from the prompt, repair feedback, validator,
  fixtures, and tests as one big-bang seven-field cutover;
- keep the canonical JSON parser as the only deterministic syntax-repair
  boundary;
- treat unknown/missing fields, wrong types, invalid bounds, unsupported or
  duplicate handles, missing required citations, invalid consequences, and
  relational-willingness pairing errors as producer-owned regeneration or
  exhaustion;
- remove both goal handle-filtering degraded-acceptance helpers. Deterministic
  code does not delete unsupported semantic values to manufacture a valid bid;
- declare `goal_bid_structure` exhaustion `unrecoverable`.

Branch recovery matrix:

| Branch result | Orchestration result | Downstream boundary |
| --- | --- | --- |
| Required branches complete | `accepted` | Complete eligible bids reach collapse. |
| Required branch failed and a complete validated sibling exists | `accepted_degraded` / `recovered_by_sibling` | Preserve failure and warning; only complete siblings reach collapse. |
| Required branch failed and no complete validated sibling exists | `exhausted` | Raise typed non-retryable error before collapse. |
| Optional branch failed | Existing isolated failure | Complete results continue. |

Protected records attach cognition invocation, graph attempt, branch,
producing stage, local attempt, cumulative producer attempt, configured limit,
attempt disposition, and final branch disposition. The downstream plan was
amended and promoted to `in_progress` under the user's explicit execution
authority before production edits.

## Change Surface

The concrete file set is the union of the Git inventory and the directly
relevant callers/tests identified during the audit. The principal surfaces are:

### Modify

- src/kazusa_ai_chatbot/accepted_task/, action_spec/, background_work/,
  brain_service/, and task_resolution/ for typed action, durable-job,
  result, ingress, and delivery alignment;
- src/kazusa_ai_chatbot/cognition_core_v2/, cognition_resolver/, nodes/,
  conversation_progress/, relevance/, complex_task_resolver/,
  local_context_resolver/, and rag/ for semantic ownership, prompt,
  evidence, context, and resolver alignment;
- src/kazusa_ai_chatbot/db/, reflection_cycle/, self_cognition/,
  character_identity_growth/, llm_tracing/, and config.py for persistence,
  temporal, trace, route, and operational contract alignment;
- src/kazusa_ai_chatbot/coding_agent/ for the audited background coding
  boundaries and async behavior;
- src/control_console/ for the audited console API/static projections;
- tests/ and tests/fixtures/ for test sanitation, replacement coverage,
  deterministic contracts, live evidence, and regression alignment;
- README.md, README_CN.md, docs/HOWTO.md, affected subsystem READMEs,
  development_plans/README.md, and the current plan for documentation
  reconciliation;
- .codex/ and .agents/skills/development-plan/ only for direct governance
  drift found in the Aug 1+ changes and required to make the execution record
  internally consistent.

### Create

- test_artifacts/diagnostics/august_change_alignment_review.md as the
  agent-authored human-readable audit artifact;
- test_artifacts/diagnostics/cognition_v2_baseline_feature_regression_map.md as
  the final parent-owned baseline-to-Cognition-V2 comparison artifact;
- focused tests or fixtures only where an audited contract lacks a minimal
  proof and the new coverage belongs to the existing test family.

### Delete

- deprecated tests, dead compatibility branches, or duplicate helpers only
  after the audit records the canonical replacement, proves current coverage,
  and confirms that removal belongs to the audited contract.

### Keep

- completed and superseded plan records, historical raw traces, existing user
  changes, and evidence needed for before/after review;
- public API and storage contracts that already match the project direction;
- live behavior that is character-grounded and contract-valid even when its
  wording differs from a brittle historical assertion.

## Runtime Or Resource Constraints

- Use venv/Scripts/python for Python commands and the repository’s existing
  dependency set. Package installation is outside this plan.
- Keep deterministic verification batched and live LLM verification
  sequential, individually inspected, and durably traced.
- Keep live response and retry paths bounded. A repair must preserve or reduce
  the current call, retry, context, and delivery limits unless an existing
  audited contract explicitly requires the limit change.
- Keep browser validation local and bounded. If the Control Console is touched,
  record the served checkout, URL, browser path, session state, page identity,
  console/page errors, interaction proof, and fresh/stale asset result.
- Keep database operations read-only unless an existing explicitly marked test
  requires isolated setup or teardown; record Mongo availability and the exact
  marker used.
- The final parent-owned comparison is read-only and uses explicit baseline and
  Cognition V2 refs. It may inspect `.env` only for the
  user-authorized legacy-feature mapping, and it must redact values from every
  artifact and message.

## Progress Checklist

- [x] Handoff baseline, commit range, worktree patch, and plan registry captured.
- [x] Full tracked and untracked path inventory regenerated from Git.
- [x] All 22 completed plans mapped to changed paths, evidence, and verdicts.
- [x] Active-plan and relevant pre-cutoff dependency surfaces mapped.
- [x] Undocumented change clusters assigned review IDs and dispositions.
- [x] Root architecture and affected subsystem ICD audit completed.
- [x] `COGNITION-V2-BID-EXHAUST-ARCH` architecture decision, budget matrix,
      recovery matrix, and downstream-plan amendment are complete.
- [x] Production code quality and semantic-ownership audit completed.
- [x] Test taxonomy, fixture, assertion, deletion, and skip audit completed.
- [x] Code, test, and documentation repairs completed within scope.
- [x] Deterministic verification completed and every failure dispositioned.
- [x] Changed live-LLM cases run individually, inspected, and reviewed.
- [x] Required live-DB and Control Console checks completed or justified.
- [x] Human-readable audit artifact authored from real evidence.
- [x] Final parent-owned baseline-to-Cognition-V2 comparison completed.
- [x] Baseline/V2 refs and the feature regression map are reviewed and linked.
- [x] Legacy `.env` feature references are classified with redacted evidence.
- [x] Final status/diff/docs consistency review completed.
- [x] User-facing closeout and lifecycle update prepared.

## Execution Ownership

gpt5.6 sol owns the complete review and remediation loop. It selects the local
mechanics and verification breadth that fit the affected risk while preserving
the fixed scope, contracts, exclusions, and end goal in this plan.

At the final workflow gate, gpt5.6 sol pins the two comparison refs, performs
the read-only feature/configuration comparison, authors and reviews its
Markdown artifact, and keeps any requested fixes outside this evidence task.
No delegated implementation, review, or comparison role participates under the
user's 2026-08-08 parent-only execution directive.

The owner resolves ordinary blockers from repository evidence and records an
external limitation when local remediation cannot remove it. Each limitation
records the affected contract, attempted recovery, exact evidence, and residual
risk; it does not convert an unverified result into a pass. The owner keeps
semantic decisions in LLM prompts/contracts, deterministic ownership in
validators/executors, and all new scope outside this plan.

## Verification

### Inventory and static checks

- Re-run git log --since='2026-08-01 00:00:00' with path and status output,
  git diff --name-status, and git status --short at the handoff and final
  checkpoints.
- Confirm every changed path has a review-matrix row and every row has an
  owning contract and final disposition.
- Run git diff --check, compile every changed Python file with the project
  interpreter, and run relevant repository static/documentation checks.
- Scan for deleted or bypassed validators, duplicate parsers, direct adapter
  calls from workers/graph code, raw platform identifiers in model context,
  undocumented fallback defaults, plan-specific Python comments, and
  deterministic post-LLM semantic overrides.

### Deterministic and patched-LLM checks

- Run focused suites for each repaired boundary first.
- Run the full non-live suite with the project marker contract after focused
  repairs.
- Run patched LLM handoff tests for graph/state/worker routing and error paths.
- Confirm every failure is fixed, is a genuine external/environment condition
  with evidence, or is a documented intentional contract result with current
  coverage. The final report lists all three categories separately.

### Real LLM checks

- Run each changed or newly added live_llm case one at a time.
- Inspect the durable trace after every case before launching the next case.
- Author or update the human-readable review with run context, input,
  rendered/parsed output, route/state decisions, quality judgment, validation,
  and raw evidence paths.
- Judge groundedness, semantic ownership, target/source fidelity, character
  judgment, continuity, refusal/unsupported behavior, and user-visible
  quality in addition to schema and pytest status.

### Live database and console checks

- Run explicitly marked live-DB checks when MongoDB is available, using isolated
  test data and the existing project setup/teardown boundary.
- For affected Control Console surfaces, use the project console workflow and
  validate the exact served URL in a fresh context plus the relevant stale-tab
  or hard-reload path. Record screenshots only when the changed surface or
  existing acceptance contract requires them.

## Final Self-Review And Closeout

The execution owner performs a final self-review from the completed diff and
audit artifact after all repairs. The review uses the approved baseline, not
the post-repair state alone, and checks:

- every Aug 1+ change is represented;
- every completed plan’s scope and acceptance evidence matches its code;
- every undocumented cluster has an explicit verdict;
- ownership boundaries and semantic authority remain intact;
- tests prove current behavior without hiding failures;
- no compatibility, fallback, feature, migration, or unrelated cleanup was
  introduced;
- residual failures are specific, reproducible, and justified;
- documentation and plan registry state describe the final repository.

## Acceptance Criteria

- [x] The final audit matrix covers every tracked commit path and worktree path
  from the fixed cutoff through handoff, plus every directly relevant older
  dependency.
- [x] All 22 completed plan records are explicitly reconciled against their
  implementation, tests, documentation, and recorded residuals.
- [x] The active context-fade/sleep work, coding-agent dependency surface, and all
  undocumented change clusters are reviewed rather than omitted for lacking a
  completed plan file.
- [x] Misaligned or poor production code is repaired at its owning boundary, with
  focused regression evidence and affected-suite verification.
- [x] Deprecated, brittle, misaligned, duplicate, or failure-hiding tests are
  replaced, corrected, or removed with the canonical coverage recorded.
- [x] Every deterministic failure in the final scope is eliminated or explicitly
  justified, and the complete non-live suite has a recorded result. A
  justification includes the exact failure, why it is external or contract-
  intentional, the evidence that supports it, and the remaining risk.
- [x] Every changed live-LLM case has an individually inspected trace and a
  human-readable quality judgment; every remaining live failure has a precise
  evidence-backed justification.
- [x] Required live-DB and Control Console checks pass or have explicit external
  limitation records with exact commands, errors, and residual risk.
- [x] Changed Python compiles, prompt rendering checks pass, CJK-bearing files
  preserve UTF-8 safety, and git diff --check passes.
- [x] The final diff contains no unreviewed path, no unowned semantic decision, no
  hidden skip/xfail used to manufacture green status, and no test-only rewrite
  that masks a production defect.
- [x] `COGNITION-V2-BID-EXHAUST-ARCH` has a recorded budget matrix, retry-ledger
  invariant, branch recovery matrix, protected-capture contract, and approved
  downstream-plan alignment before its production changes are verified.
- [x] The final parent-owned regression map pins the baseline and
  Cognition V2 refs, covers every scoped baseline feature, highlights missing
  or behavior-regressed features, classifies legacy `.env` keys with redacted
  evidence, and records user decisions separately from implementation work.
- [x] `test_artifacts/diagnostics/august_change_alignment_review.md` is complete,
  readable without raw JSON inspection, and links all material raw evidence.
- [x] `test_artifacts/diagnostics/cognition_v2_baseline_feature_regression_map.md`
  is complete, readable without protected raw traces or `.env` values, and
  reviewed by the parent execution owner.
- [x] This plan’s checklist, execution evidence, residual-risk record, registry
  row, and final user-facing closeout agree before status changes to completed.

## Draft Plan Self-Review

Self-review updated on 2026-08-08 after protected-trace capture.

- The original draft over-specified agent choreography. It now identifies
  gpt5.6 sol as the execution owner and limits the requested implementation
  agent to a final, read-only regression-mapping evidence handoff.
- The original draft allowed the audit boundary to drift. The approved
  execution baseline now fixes the end commit, worktree, registry snapshot,
  and treatment of later unrelated changes.
- The original draft distributed the outcome across several sections. The
  Summary now states the end goal directly: every in-scope change receives an
  architecture and plan verdict, required repairs are applied, and every
  remaining failure is fixed or evidence-backed and justified.
- The original draft used an overly broad pass claim for tests. Acceptance now
  requires every deterministic failure to be eliminated or justified with the
  exact failure, contract or external reason, evidence, and residual risk.
- The original draft called the same-owner closeout an independent review. The
  section now names it Final Self-Review And Closeout and checks the approved
  baseline against the completed diff and audit artifact.
- The original scope, 22 completed-plan references, undocumented change
  inventory, architecture contracts, remediation authority, and progressive
  verification requirements remain intact.
- The protected bid-exhaust evidence now has an explicit architecture task,
  downstream-plan dependency, monotonic-budget gate, and execution order.
- Metadata-only and full protected capture responsibilities are separated, and
  the action-planning trace is labeled as adjacent evidence rather than a
  bid-exhaust reproduction.
- The final baseline-to-Cognition-V2 mapping is placed after verification and
  before closeout, with explicit ref pinning, redacted `.env` inspection, and a
  user-decision queue separated from remediation authority.

## Execution Evidence

### Approved Execution Baseline — 2026-08-08

- User approval: the user directed full execution and delegated architectural
  decisions to the execution owner on 2026-08-08.
- Branch: `cognition_core_v2`.
- Tracked cutoff parent: `7e7c1617a6773e20a9ac585b0190444ad1f16935`
  (`2026-07-31T04:30:22+12:00`).
- Execution HEAD: `423f6573bd1085f5f4d492213f28e047727b9b50`
  (`2026-08-07T23:33:06+12:00`).
- Audited tracked range:
  `7e7c1617a6773e20a9ac585b0190444ad1f16935..423f6573bd1085f5f4d492213f28e047727b9b50`.
- Handoff worktree: modified `development_plans/README.md` plus untracked
  `development_plans/active/short_term/august_change_alignment_audit_and_remediation_plan.md`;
  no production or test worktree patch was present.
- Handoff SHA-256: registry
  `2666DBD2A28B8C3AC24E36348D8BD3FFD7D368E9155276606D3C3AF289BFC12F`;
  audit plan
  `1093D39A539B38EE225F8DDD7F55D183778EFFC6671DA613C199671B1A16AE0E`.
- Secret boundary: `.env` and secret-bearing configuration remain unread
  during the main audit. Only the final read-only regression-mapping gate may
  enumerate redacted `.env` key names under the plan's explicit authority.
- Execution ownership: the user's 2026-08-08 directive requires parent-only
  execution without subagents or execution-time questions; all architecture,
  implementation, review, verification, and evidence work remains with the
  parent execution owner.
- Concurrent-change rule: any path change after this snapshot is compared with
  this baseline and receives an attributable audit row before inclusion in a
  completion claim.

### Protected Cognition Core V2 Capture — 2026-08-08

The protected trace query used a post-runtime-change window beginning at
`2026-08-07T07:00:00Z`. The exports below are local diagnostic artifacts and
remain outside committed source as required by this plan.

| Trace | Evidence | Disposition |
| --- | --- | --- |
| `llmtrace_0cd78d39a55e48f6ae8efb662262eb70` | Historical required-selection failure: 15 steps, including 12 goal-cognition contract-error steps across ordinary/autonomy initial plus two regenerations. The metadata export has empty raw response text and empty parsed candidates. | Bid-exhaust baseline; exact rejected field is unresolved. |
| `llmtrace_a5997476b97640b4af5e0786244b1676` | `2026-08-07T11:16:59Z`; semantic-value bound failure, question-owned delta-path rejection, and ordinary relational-willingness mismatch; ordinary regeneration succeeded. | Partial failure with bounded recovery. |
| `llmtrace_899ea885f64b402cb2df6ef1d4e35783` | `2026-08-07T11:29:12Z`; resolved knowledge-gap transition, exact semantic-item field shape, unknown evidence-handle, and ordinary relational-willingness failures; repairs succeeded. | Partial failure with bounded recovery. |
| `llmtrace_cb1de2895b4a4987be79cc8530ab6f5c` | `2026-08-07T11:34:05Z`; question-owned delta path, ordinary unauthorized evidence handle, relational-willingness mismatch, and semantic micro-appraisal field shape; ordinary repair succeeded on attempt three. | Partial failure with bounded recovery at the current limit. |
| `llmtrace_e86e2bda365e49aca2d0ad54fb0fd066` | `2026-08-06T22:15:46Z`; action-planning request-fidelity run with an ordinary goal contract error repaired before action planning succeeded. | Adjacent completed-plan evidence; not a bid-exhaust reproduction. |

Raw JSON exports:

- `test_artifacts/diagnostics/cognition_v2_bid_exhaust_current.json`
- `test_artifacts/diagnostics/cognition_v2_recent_partial_failure.json`
- `test_artifacts/diagnostics/cognition_v2_recent_recovered_failure.json`
- `test_artifacts/diagnostics/cognition_v2_post_change_partial_failure.json`
- `test_artifacts/diagnostics/cognition_v2_action_planning_current.json`

Human-readable capture review:

- `test_artifacts/diagnostics/cognition_v2_bid_exhaust_architecture_capture.md`

Capture interpretation:

- The bid-exhaust evidence fails at the Cognition Core V2 goal contract
  boundary after RAG/evidence preparation; the export does not support a
  claim that RAG selected the wrong evidence.
- The newer full capsules show that contract failures are currently being
  repaired in bounded attempts, but the same run can still carry partial
  semantic failures before workspace collapse and action planning.
- The metadata-only bid-exhaust trace cannot identify whether the root cause
  was exact-field drift, unauthorized handles, relational-willingness pairing,
  or another candidate-shape error. Treat that as a protected-capture gap and
  residual risk for `COGNITION-V2-BID-EXHAUST-ARCH`.

Capture-only replay observations:

- A production-shaped third-party required-selection input reached the dense
  ordinary goal route and returned a bid after two model calls; the durable
  local artifact records the raw responses and route.
- Recent semantic failure replays reached current repair successfully. One
  replay produced a different first contract error before repair, which is
  evidence that the failure family is reproducible while the exact model
  candidate remains variable.
- The temporary live probes produced durable raw-boundary artifacts under
  `test_artifacts/llm_traces/` and replay artifacts under
  `test_artifacts/cognition_core_v2_*`; these are capture evidence only and
  are not source changes.
- These observations are evidence for the follow-up task, not acceptance of a
  runtime fix. The test-source edits used during probing were removed from the
  worktree after capture.

### Bid-Exhaust Architecture Approval — 2026-08-08

- Decision owner: parent execution owner under the user's explicit instruction
  to make architecture decisions and fully execute the plan.
- Attempt budget: three cumulative goal-model calls per producing stage and
  branch across the entire cognition invocation; graph retries reuse only the
  remaining budget.
- Exhaustion: typed pre-state-commit and non-retryable after the cumulative
  limit; the service retry remains available to other eligible owners.
- Candidate contract: canonical JSON parsing followed by strict validation;
  no deterministic deletion of unsupported model-authored handles or values.
- Required branch policy: preserve complete validated siblings with a typed
  degraded/recovered disposition; fail before collapse when none exist.
- Downstream plan:
  `development_plans/archive/completed/bugfix/required_selection_partial_recovery_bugfix_plan.md`
  implemented, verified, signed off, and completed.
- Execution model: parent-only, including implementation, review, tests, and
  evidence.

### Final Baseline-to-Cognition-V2 Regression Mapping — completed 2026-08-08

- `baseline_ref`: `8f834bf87a83ee42aca804934fb44af63788420c`,
  the frozen approved baseline controller revision.
- `cognition_v2_ref`: `423f6573bd1085f5f4d492213f28e047727b9b50`
  plus the reviewed 48-path remediation overlay and one deliberate test
  deletion, manifest SHA-256
  `C28658B912E0D3625D598ABCFBB6A90AF7D03DD623B7FA195391C91F6FBEF97E`.
- Output:
  `test_artifacts/diagnostics/cognition_v2_baseline_feature_regression_map.md`,
  SHA-256
  `19D59F5B9E4FDC166C30DB3CDB43964A34985844EDAD46CE33CD72ABC61D57BB`.
- Result: 48 feature rows classify C01-C20, O01-O10, nine moved baseline
  selectors, and seven topology/deployment groups. No missing feature or
  unintended behavior regression remains. Ten legacy configuration keys are
  isolated for a future user decision, with every value redacted.
- Status: accepted. The comparison itself produced no production, test,
  `.env`, or runtime-configuration change.

### Final Audit And Remediation Signoff — 2026-08-08

- Inventory: 42 commits, 324 unique tracked-range paths, and 348 unique paths
  across the handoff, attributable remediation, and lifecycle archive moves
  are enumerated in
  `test_artifacts/diagnostics/august_change_alignment_review.md`, SHA-256
  `0C8F80268C2BB0E1080DB950172A8277BCC12BA45068EAE03F71D0FAC844091F`.
- Plan mapping: all 22 completed records, two active/pre-cutoff dependencies,
  13 undocumented clusters, lifecycle metadata, and the R01 remediation have
  explicit owners and dispositions.
- Architecture: invocation-scoped cumulative attempt accounting, strict
  seven-field bids, complete-sibling recovery, typed exhaustion, and failure
  capsule v3 are implemented without a compatibility shim or semantic
  post-processing override.
- Adjacent repairs: deployment carryover bindings, service-owned interaction
  style, consolidation origins, console lookup context, lazy task-handler
  imports, portable live fixtures, and unique cutover fixtures are complete.
- Focused verification: 217 passed with 4 deselected; ledger/routing passed 21;
  the Cognition V2 family passed 519 with 2 precise skips and 230 deselections.
- Integrated non-live verification: 4,198 passed, 26 skipped, and 1,123
  deselected in 236.47 seconds; the sole warning is a nonblocking Starlette
  TestClient deprecation.
- Static verification: 41 changed Python files compiled, `git diff --check`
  passed, and retired-contract/parser/comment scans were clean.
- Live LLM: ordinary, autonomy, and parallel ordinary/autonomy cases passed
  individually with inspected raw messages, candidates, validation, routes,
  ledger coordinates, and semantic quality.
- Live DB: four Cognition V2 smokes and the task-resolution cutover rehearsal
  passed; marker cleanup and seed restoration were verified.
- Control Console: the exact checkout served successfully and all seven
  affected navigation E2E cases passed. The supplemental in-app browser had no
  connected backend, leaving only manual fresh-tab/hard-reload observation as
  an external limitation.
- Residual decision: removal of ten legacy `.env` keys and three stale verifier
  Compose bindings remains a separately authorized configuration cleanup.
- Final parent review: accepted with no unresolved in-scope finding.
