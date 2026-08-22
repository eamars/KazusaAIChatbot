# Cognition V3 Hybrid Agentic Loop Reconciliation And Big-Bang Replacement Plan

## Summary

- **Status:** completed on 2026-08-22 after Gate 8 V3-only cutover,
  owner-directed narrow post-cutover verification, and final parent audit.
  Execution began in progress on 2026-08-19 after explicit owner production-
  implementation authorization. Approved on 2026-08-19 by the owner. Historical independent
  technical-contract review passed
  on 2026-08-19. The later owner-directed execution-governance amendment and
  total-context budget clarification are parent-reviewed. The later owner-
  directed execution-integrity, semantic-failure, diagnostic-order, and data-
  provenance amendment is also parent-reviewed. The 2026-08-20 owner
  execution amendment permits delegation and fixes one reusable subagent as
  the sole production-code implementation executor.
  This document is an approved implementation contract, amended on 2026-08-21
  by the owner's Gate 7 input-flow direction. The owner
  authorized implementation on 2026-08-19, the exact closure goal is active,
  Gate 0 passed on 2026-08-19T12:18:06Z, Gate 1 passed on
  2026-08-19T15:29:12Z, and Gate 2 is in progress.
  The owner amended cutover on 2026-08-22 to remove the V2 runtime and end
  dual-engine support. The executable decommission boundary is recorded in
  `cognition_v2_runtime_decommission_after_v3_cutover_plan_2026-08-22.md`;
  its V3-only direction supersedes this plan's V2 runtime-rollback clauses.
- **Plan class:** high-risk cognition architecture reconciliation, LLM prompt
  and orchestration replacement, persistence addition, and evidence-gated
  engine cutover.
- **Execution constraint:** the root parent owns scope, the plan-closure goal,
  executor handoffs, checkpoints, evidence, gate decisions, review, cutover,
  and lifecycle closure. Exactly one reusable subagent is the sole executor
  permitted to modify production code during the remaining implementation
  flight. The parent may perform read-only inspection and may edit plans,
  evidence, tests, governed artifacts, and documentation; production-code
  remediation returns to the same implementation subagent whenever practical
  to preserve context and cache affinity.
- **Authored:** 2026-08-19 from repository HEAD `047bed95` on branch
  `feature/cognition_core_v3_cache_affine`.
- **Governing architecture:**
  [`docs/architecture/cognition_v3_hybrid_agentic_loop_architecture.md`](../../../docs/architecture/cognition_v3_hybrid_agentic_loop_architecture.md),
  designated by the owner as the de-facto source of truth for this work. The
  architecture document remains non-executable on its own; this plan resolves
  its implementation choices and supplies execution authority only after the
  approvals above.
- **Supersedes:**
  [`archive/superseded/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md`](../../archive/superseded/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md).
  Its execution log and partial implementation are historical evidence. None
  of its parallel-chain topology, isolated goal transcripts, per-stage V3
  routes, or skipped baseline decisions carry into this plan.
- **Goal:** replace the current partial V3 semantic executor with one
  append-only, serialized primary cognition chain; retain V2 semantic products,
  contracts, deterministic state mechanics, and public payloads; isolate L1,
  authorization, and optional JSON repair on one genuinely distinct sidecar
  lane; continue resolver cycles from an in-memory chain session; and prove
  semantic parity plus cache/latency improvement before selecting V3 by
  default.
- **Change direction:** big-bang replacement of the existing V3 internals.
  There is one canonical V3 implementation after the change. There is no
  compatibility flag for the superseded V3 topology, no parallel V3 executor,
  no stage-route mapper, and no fallback from a failed V3 invocation to V2.
- **External boundary:** `CognitionCoreInputV2`, `CognitionCoreOutputV2`,
  cognition state, the 21-emotion derivation, the 11-axis relationship model,
  resolver request/observation contracts, text/visual surface payloads,
  morning refresh, and validator semantics stay unchanged. The injected V3
  services object becomes lane-scoped as required by the architecture.
- **Cutover:** V2 stays the default through implementation and all evidence
  gates. V3 becomes the default only after deterministic parity, live quality,
  serving-overflow, context-estimator, cache-affinity, resolver-continuation,
  control-console, and performance acceptance all pass. Rollback is the single
  selector value `COGNITION_CORE_ENGINE=v2`; no data migration is involved.
- **Current deterministic evidence:** before this plan was written, the
  existing partial-V3 selector/connector suite passed `60` tests in `0.85s`.
  Those tests prove that the superseded topology is internally green; they do
  not prove alignment with the governing architecture.

## Current State And Gap Assessment

### System-level finding

The current V3 implementation is not an unfinished form of the target loop.
It implements the superseded plan's core premise: several parallel,
cache-affine semantic chains joined through deterministic checkpoints. The
target architecture changes the unit of cache affinity from a semantic owner
to the whole cognition invocation. Reconciliation therefore requires replacing
the V3 orchestration, prompt geometry, routing contract, repair behavior, and
resolver recurrence model together. Incremental preservation of the current
topology would retain the architectural defect.

### What is already reusable

1. `CognitionCoreInputV2`, `CognitionCoreOutputV2`, state models, transition
   guards, reducers, emotion definitions/derivation, relationship maintenance,
   branch activation order, morning refresh, surface contracts, and public
   validators remain the canonical substrate.
2. The closed `v2`/`v3` process selector, the one-final-commit connector flow,
   the external resolver loop, and the current V2 default are safe seams to
   retain.
3. The current V3 package supplies useful scaffolding for typed stage records,
   diagnostics, prompt tests, and source-to-test ownership. Its semantic-stage
   implementations may be rewritten in place.
4. The existing V2 baseline fixtures, cognition comparison utilities, trace
   infrastructure, event logging, cognition graph endpoint, and control-console
   renderer are reusable infrastructure, subject to the explicit extensions in
   this plan.
5. The current V3 authorization projection is directionally correct because
   authorization is already isolated from the model-visible cognition
   transcript. It must be rebound to the sidecar lane and exact V2 validators.

### Gap matrix

| Area | Current implementation | Required ideal state | Required disposition |
|---|---|---|---|
| Primary topology | `execution.start_wave(...)` launches multiple appraisal and goal tasks. | One serialized A1→A2→I1→G1a→G1b→I2→W1→P1 chain on one lane. | Replace executor and registry; remove parallel-wave behavior and tests. |
| Prompt geometry | One static prompt and transcript per semantic chain. | One byte-stable system head containing only stage-independent rules and character identity; each dynamic carrier appears once with its first consuming stage in one append-only invocation transcript. | Replace prompt/transcript ownership. |
| Appraisal | Three parallel first-wave chains plus a later terminal-outcome call; custom aggregate arrays. | Fixed A1/A2 stages, exact per-family proposition/delta arrays, deterministic V2 replay reduction, direct singleton recovery. | Replace V3 appraisal contract and bridge. |
| State reduction | Provisional reductions occur between current parallel chains. | One I1 interlude after grouped appraisals, using V2 reducers/maintenance/emotion derivation verbatim. | Centralize in I1; remove topology-specific provisional bridges. |
| Goal generation | Preliminary goals run beside appraisal and sibling goal transcripts are isolated. | Ordinary G1a after I1, then registry-ordered G1b whose branches see the accepted ordinary bid and earlier sibling content. | Replace goal ordering and visibility. |
| Required selection | Partial contract reuse exists. | The exact V2 specialized required-selection validator, fixed role binding, evidence coverage, and deterministic-only parsing. | Reuse the canonical V2 pure validator/materializer. |
| Relationship-sensitive collapse | Current helpers retain some authority behavior. | Generate G1b normally, then deterministic ordinary-primary collapse; W1 is skipped and effects are denied downstream. | Preserve helper, move it into I2/W1 control flow. |
| Workspace | Separate model call and current V3 schema. | W1 appears in the same chain only for two or more complete, non-sensitive bids; zero/one bid short-circuits. | Rewrite model boundary and validator binding. |
| Planning | Separate action-planning transcript. | P1 is the next primary-chain question and emits the exact V2 planning envelope. | Rewrite as a chain step. |
| Authorization | Fresh call exists but may share any mapped V2 stage route/model. | X1/X2 use fresh minimal context on one distinct sidecar lane; no sidecar means deny all. | Replace service/routing contract and enforce lane identity. |
| L1 subconscious | Absent. | Optional asynchronous `L1ResidueV1`, advisory only, joined without waiting at A1 or G1a. | Add a sidecar owner and closed schema. |
| Resolver recurrence | Every external resolver cycle re-enters the complete V3 facade. | Cycle N reattaches to the episode session and appends observation→delta appraisal→bid revision→fresh P1. | Add bounded session registry and continuation path. |
| Lane isolation | Parallel calls can interleave on the same local model; route checkpoints accept model changes. | FIFO one invocation at a time for each primary `(base_url, model)` lane; no foreign request between chain steps. | Add process-local lane coordinator; delete route checkpoints. |
| Service injection | V3 consumes `CognitionCoreServicesV2` and maps stages to twelve configs. | Exact `CognitionChainServicesV3` with chain and optional sidecar bindings. | Replace service construction at selector seam. |
| Configuration | V3 indirectly depends on all V2 stage bundles. | V3 consults only `COGNITION_V3_CHAIN_LLM_*` and optional `COGNITION_V3_SIDECAR_LLM_*`; V2 routes remain V2-only. | Make cognition route loading engine-conditional. |
| Context budget | UTF-8 byte checks and per-owner prompt caps; no full-chain ledger. | Calibrated CJK-aware estimator, 50k normal/65k conditional total request ceilings with per-step completion reservation, one deterministic re-anchor, typed overflow. | Replace budget model. |
| Repair | An ephemeral request can contain the rejected draft. | Tail rollback removes the failing assistant answer; retry repeats the same question with a monotonic error appendix and never sends the rejected draft as input. | Replace repair transcript construction. |
| JSON repair | Global repair route behavior is not lane-scoped. | Canonical parser first; optional repair uses the sidecar config; required selection stays deterministic-only. | Extend canonical parser injection and V3 call discipline. |
| Turn deadline | Per-stage timeout only. | One configured 240-second turn deadline checked between steps, subordinate per-step timeouts, terminal decision on accepted products. | Add deadline owner and tests. |
| Public API | V3 exports only part of the V2 Stage 2 facade and several internal diagnostic names. | Same public entrypoint names/payload validators as V2; V3-specific public addition is the services dataclass. | Rebuild V3 package facade. |
| Output parity | Current README records empty role assignments and prompt-fitting differences as accepted limitations. | No documented parity limitations; role assignments and all output/state carriers match V2 contracts. | Remove limitations and close gaps before acceptance. |
| Diagnostics | Context-local stage/config rows only. | Ordinary trace steps plus protected full transcript, persisted `cognition_chain_run.v1`, sanitized aggregate event, and engine descriptor. | Extend tracing, DB, event logging, service, and console. |
| Control console | Existing graph has no chain-run schema. | Read-only chain run panel sourced from persisted records through existing service/console paths; no prompt exposure. | Add bounded additive contracts and rendering. |
| Baseline evidence | The former plan explicitly skipped its required Gate 1; no sealed V2/V3 architecture comparison exists. | A frozen current-V2 control and current-partial-V3 audit precede target production edits. | Reinstate a mandatory baseline gate; a skip blocks execution. |
| Tests | Current V3 tests assert parallelism, isolated goals, route checkpoints, and byte caps. | Tests assert one lane, sibling-visible ordered goals, tail rollback, session continuation, token ledger, sidecar failure, and persisted observability. | Rewrite conflicting nodes and impact manifest entries. |

## Scope And Change Direction

### In scope

1. The implementation behind `cognition_core_v3.run_cognition(...)`.
2. V3 prompt, anchor, transcript, execution, appraisal, goal, collapse,
   planning, authorization, diagnostics, context ledger, L1, and session
   ownership.
3. Minimal public extraction of pure V2 validators/materializers required to
   keep V2 and V3 on one semantic contract.
4. V3 lane configuration, connector service construction, and selector
   metadata.
5. Canonical JSON-parser injection needed for sidecar-owned repair.
6. Protected chain transcript capture, sanitized chain-run persistence,
   cognition-chain event telemetry, brain-service read projection, and the
   existing control-console cognition view.
7. Deterministic, integration, live-quality, long-context, overflow, and
   performance evidence; documentation and source-test impact registration.
8. V3 default cutover after every gate passes.

### Change radius

- **Core cognition:** high; the full current V3 runtime is replaced.
- **Shared V2 semantic substrate:** low-to-medium; pure helpers become public
  canonical APIs, while V2 behavior and payloads remain unchanged.
- **LLM/config/connector:** medium; one optional `LLMCallConfig` field, two V3
  route bundles, engine-specific service construction.
- **Persistence/telemetry:** medium; one transient diagnostic collection and
  one sanitized event family, with no cognition-state migration.
- **Brain service/control console:** low-to-medium; additive read-only fields
  and one bounded panel on the existing cognition page.
- **Tests/docs:** high; superseded topology assertions are replaced and live
  comparison evidence is added.

### Big-bang policy

The current V3 execution path changes in one coherent implementation slice.
Caller, callee, tests, ownership manifest, ICDs, configuration, and console
schema move to the canonical boundary together. The 2026-08-22 owner cutover
amendment removes the separately importable V2 runtime through the companion
decommission plan. No alias modules, legacy V3 flags, fallback mappers, dual
V3 schemas, or adapter translations are introduced.

## Mandatory Skills

The responsible executor reads and applies these skills directly before its
matching work and re-applies triggered skills after compaction recovery or a
handoff:

1. `development-plan` for eligibility, checkpoints, evidence, parent audit,
   sign-off, and plan status changes.
2. `local-llm-architecture` before changing prompt geometry, stage ownership,
   routing, context budgeting, recurrence, or latency behavior.
3. `py-style` before creating or editing any Python file.
4. `cjk-safety` before writing CJK content in Python prompts or fixtures.
5. `no-prepost-user-input` before reviewing goal, willingness, required-
   selection, or user-instruction handling. Semantic user-input judgment stays
   LLM-owned; deterministic code validates contracts and permissions.
6. `test-style-and-execution` before creating, changing, or running tests.
7. `debug-llm` for every real-model invocation, prompt comparison, trace
   inspection, quality review, and performance artifact.
8. `python-venv` before environment verification or dependency work. Use
   `venv\Scripts\python`; this plan adds no dependency.
9. `control-console-web-development` for the brain-service/console contract,
   frontend change, browser validation, cache/stale-JavaScript checks, and
   screenshot sign-off.
10. `database-data-pull` whenever a missing reproduction, calibration, or
    quality input must be copied read-only from production under Decision 49.

## Mandatory Rules

1. **Fixed single production-implementation subagent constraint:** the root
   parent maintains one coherent workspace, goal, scope, and evidence record
   from the current checkpoint through archive closure. Exactly one reusable
   subagent owns all remaining production-code edits. The parent supplies
   bounded handoffs with explicit owned files, acceptance checks, applicable
   skills, baseline state, and next checkpoint, reviews every returned diff,
   and reuses that subagent for remediation while it remains available. The
   parent may directly change plans, execution evidence, tests, fixtures,
   governed artifacts, and documentation. Any production-code correction is
   handed back to the same implementation subagent.
2. **Goal contract:** immediately after owner approval and explicit production
   authorization, the parent checks the thread goal state and creates the goal
   `Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all
   Gates 0-8, cutover evidence, final audit, and archive completion` when no
   unfinished goal exists. The goal has no token budget. It remains active
   across turns and compactions and becomes complete only after every
   Acceptance Criterion is evidenced and the plan is archived as completed.
3. **Question-free flight:** execution flight begins when the goal above is
   created and ends at completed archive closure or a genuinely blocked goal
   disposition. During that interval the parent makes every in-scope decision
   directly and proceeds without requesting user input. Decision precedence is
   current system/developer/`AGENTS.md` safety and authorization rules; the
   owner's fixed single-production-subagent, goal, question-free, and
   compaction rules here; the rest of this approved
   plan; the governing architecture; canonical V2 contracts; current
   source/tests/ICDs; then the smallest deterministic implementation that
   preserves those authorities.
4. When discovery exposes an unlisted dependency inside the approved goal and
   architecture, the parent amends the plan, Change Surface, exact impact row,
   and gate evidence before editing that dependency, performs a fresh parent
   contract audit, and continues. It preserves all thresholds, exclusions,
   public contracts, semantic owners, permission boundaries, and cutover
   rules. A requirement for new external authority, credentials, destructive
   production mutation, or architecture outside this precedence leaves the
   applicable gate open and is recorded as a blocker without a mid-flight
   question.
5. **Compaction recovery:** after every conversation compaction or restored
   summary, the parent performs the recovery checklist before any edit, test,
   live call, DB operation, deployment action, or other state-changing work:
   read the current goal; run `git status --short`; re-read Summary, Scope And
   Change Direction, Mandatory Skills, Mandatory Rules, Change Surface, Agent
   Autonomy Boundaries, the current Execution Gate, Acceptance Criteria,
   Progress Checklist, Execution Evidence, and Execution Continuity;
   re-read the Confirmed Decisions, Contracts And Data Shapes, Runtime Or
   Resource Constraints, Test Impact rows, governing-architecture sections,
   subsystem ICDs, source, and tests relevant to the active work item; and
   re-apply every triggered skill. The parent records the compaction event,
   sections/references read, goal state, worktree state, current gate, completed
   evidence, and next checkpoint in Execution Evidence.
6. Read `git status --short`, `README.md`, `docs/HOWTO.md`, this plan, the
   governing architecture, the relevant subsystem README files, source, and
   tests before each implementation checkpoint. Preserve unrelated changes.
7. The owner authorizes read-only execution-readiness probes of the current
   service, production MongoDB, configured primary/sidecar LLM endpoints, model
   availability, context-window declarations, and operator surfaces. Use
   existing loaded configuration, project scripts, and protected diagnostic
   boundaries; record only sanitized endpoint/model fingerprints and aggregate
   health. Preflight probes create no semantic DB writes, model-state changes,
   deliveries, actions, schedules, or user-visible output.
8. Production edits require explicit owner authorization in addition to plan
   approval. A draft or approved plan alone is insufficient.
9. Use `apply_patch` for manual edits. Use the project virtual environment.
10. Preserve the architecture ownership split: RAG/resolver return evidence;
    cognition chooses stance and goals; dialog/L3 own wording; deterministic
    code owns validation, state transitions, budgets, permissions, scheduling,
    persistence, and delivery eligibility.
11. Every raw model response first enters
    `kazusa_ai_chatbot.utils.parse_llm_json_output(...)`. Repair may restore
    syntax/object shape only. It cannot invent, delete into acceptance, or
    semantically correct a model decision.
12. Every chain question and assistant answer is bounded. The model never
    chooses the next step, tool, retry, route, budget tier, session action, or
    re-anchor.
13. Live LLM tests run one pytest node and one case at a time. Inspect and write
    the human-readable artifact before starting the next case.
14. The current partial V3 is diagnostic evidence only. Passing current tests
    cannot waive a target-architecture acceptance criterion.
15. A failed gate stops progression. The parent records evidence, applies the
    precedence and amendment process above, and resumes only after the fixed
    gate contract is satisfiable. It preserves the architecture, thresholds,
    case labels, and gate order.
16. **Evaluation integrity / never cheat:** every implementation, prompt,
    fixture, test, run, artifact, score, and report must measure the runtime
    contract honestly. Cheating is any manipulation whose purpose is to make a
    known test, case, score, or gate pass without improving the general runtime
    contract. This includes, without limitation: test names, case ids, fixture
    text, expected answers, assertion predicates, rubric language, or test-
    related instructions in a runtime prompt; test-only production branches;
    hidden input/output rewriting; outcome-conditioned reruns; cherry-picked
    seeds or trials; discarded semantic failures; weakened/deleted/skipped/
    `xfail` assertions; score rewriting; selective evidence omission; and a
    fallback used only by the harness. General prompt guidance is admissible
    only when it states a reusable semantic rule grounded in the architecture,
    contains no test metadata, and is checked against a distinct countercase.
17. **Two-consecutive-local-semantic-failure reset:** when the same semantic
    owner and behavior contract fail on two immediately consecutive eligible
    real-local-model invocations after structural parsing and hard validation
    succeed, the parent stops same-case reruns and prompt patching. It performs
    the root-cause and smallest-component procedure in Confirmed Decision 46
    before another full-path attempt. A retry counter or prompt tweak aimed
    only at passing the failing case does not satisfy this reset.
18. **Inherited V2 semantic failure:** the parent first attempts an
    architecture-grounded V3 correction for every Gate-1-sealed V2 model-
    semantic baseline defect. A residual inherited semantic failure does not
    block delivery when it satisfies the exact 95.00% policy in Confirmed
    Decisions 47-48. Deterministic contract, state, provenance, privacy,
    permission, authorization, and effect-safety defects remain hard failures
    and cannot consume the semantic allowance.
19. **Component-first live diagnosis:** a full real-LLM E2E test is a final
    cooperation/acceptance check, not a diagnosis method. After an E2E failure,
    the parent uses its trace only to locate the first failing semantic owner,
    proves deterministic boundaries, reproduces the exact stage packet in one
    real-LLM node test, inspects its human-readable artifact, and verifies
    patched handoffs before running the full E2E path again.
20. **Production-data fallback:** when required reproduction, calibration, or
    quality evidence is absent locally, the parent copies the smallest
    sufficient source data from production through the repository's read-only
    export/protected-trace paths. It records exact correlation/filter, time
    window, count, redactions, and SHA-256; keeps raw sensitive exports under
    `test_artifacts/`; derives only a minimal sanitized fixture when a committed
    fixture is required; and performs no production mutation or global-latest
    substitution.

## Must Do

1. Seal a current-V2 functional and live comparison baseline before target
   production edits. Record current partial-V3 topology evidence separately.
2. Replace the current V3 runtime with the exact primary-chain sequence and
   exact off-chain owners in this plan.
3. Reuse the V2 deterministic state and semantic validation owners; remove
   copied or weakened V3 contract logic.
4. Enforce primary and sidecar lane identity, FIFO, thinking, context-window,
   and non-interleaving rules mechanically.
5. Implement the total 50k baseline/conditional 65k ledger, calibrated
   estimator, per-step completion reservation, pre-anchor fitting, one
   re-anchor, and typed context-limit disposition.
6. Implement resolver session continuation and cold-rebuild safety without
   changing the external resolver contract or commit order.
7. Persist and expose the exact sanitized observability contracts in this
   plan, with best-effort writes that never alter cognition behavior.
8. Replace conflicting tests and ownership mappings; retain unchanged V2
   regression coverage.
9. Prove V2 semantic/output compatibility, target chain invariants, and live
   quality/performance gates before cutover.
10. Update every affected ICD and operator document in the same change.
11. Deliver the bulk V3 cutover once the exact 95.00% semantic floor and every
    hard gate pass; keep any accepted inherited residual visible and address it
    only after the bulk deliverable is established.

## Deferred

1. Converting relevance, settlement, Stage 0 decontextualization,
   conversation-progress recording, consolidation, reflection, L3 surface,
   dialog, or delivery into agentic chains.
2. Moving surface/dialog routes to protect cross-turn cognition cache. This
   plan measures cross-turn reuse but does not expand the Stage 2 boundary.
3. More than one primary lane, channel sharding, distributed lane leasing, or
   multi-process session sharing.
4. New tools, actions, resolver capabilities, reflexes, semantic fields,
   emotion formulas, relationship axes, branch kinds, or persistence meaning.
5. V2 package retirement is no longer deferred. The 2026-08-22 owner-approved
   companion decommission plan is required before Gate 8 closure.
6. Console prompt editing, live stepping, injection, mutation, or protected
   transcript display.
7. LLM-authored compaction or transcript summarization.
8. A cognition-state data migration. None is required.
9. Up to three inherited V2 semantic-failure trials accepted by Decision 48
   are follow-up quality work after the bulk V3 cutover. They do not block this
   plan's delivery or closure; each must either be fixed after cutover with the
   affected gates rerun or transferred with its complete evidence into a new
   active bugfix plan before this plan is archived.

## Confirmed Decisions

### Engine and public boundary

1. V3 accepts and returns the exact V2 public payloads. V3 re-exports
   `run_text_surface_planning`, `run_visual_surface_planning`,
   `run_character_morning_refresh`, `validate_cognition_input`, and
   `validate_cognition_core_output` from the canonical V2 implementation.
2. `cognition_core_v3.__all__` exposes those entrypoints,
   `run_cognition`, and `CognitionChainServicesV3`. Current V3 internal
   topology/config diagnostic types cease to be public package exports.
3. V3 retains the existing V2 error classes, accepted/recovered/
   accepted-degraded/unrecoverable ladder, output validation, and one-final-
   commit connector ordering.
4. The selector remains process-level and closed to `v2` and `v3`. A V3
   error propagates through its typed boundary and never invokes V2.

### Canonical shared helper extraction

The following names are the complete shared-helper extraction. Each change is
an atomic rename of the present V2 implementation plus all V2 call sites; V3
imports the renamed owner directly. No copied validator, adapter, alias, or
second vocabulary remains:

| Owning module | Current private name | Canonical public name | Exact ownership |
|---|---|---|---|
| `cognition_core_v2.semantic_appraisal` | `_canonicalize_semantic_appraisal_item` | `canonicalize_semantic_appraisal_item` | maps one singular model item into the existing aggregate shape |
| `cognition_core_v2.semantic_appraisal` | `_validate_semantic_boundary_candidate` | `validate_semantic_boundary_candidate` | validates carriers/evidence/domain and returns the existing accepted-item pair |
| `cognition_core_v2.semantic_appraisal` | `_merge_semantic_appraisal_item` | `merge_semantic_appraisal_item` | merges one validated item without semantic reinterpretation |
| `cognition_core_v2.goal_cognition` | `_selection_goal_draft_to_goal_bid` | `selection_goal_draft_to_goal_bid` | materializes the existing required-selection bid and fixed roles |
| `cognition_core_v2.workspace` | `_validate_partition` | `validate_workspace_partition` | validates the exact primary/supporting/suppressed handle partition |
| `cognition_core_v2.action_selection` | `_validate_action_plan_decision` | `validate_action_plan_decision` | validates and materializes the complete existing V2 P1 decision |
| `cognition_core_v2.action_authorization` | `_validate_authorization_decisions` | `validate_authorization_decisions` | validates exact candidate coverage and boolean decisions for X1/X2 |

Existing public `validate_selection_goal_draft`, `validate_goal_bid_draft`, and
`collapse_authoritative_relational_bid` remain canonical. V3 owns its lane
invocation and attempt accounting, and reuses the two existing authorization
prompt constants plus `validate_authorization_decisions`; it does not reuse the
V2 service-bound `invoke_semantic_authorizer(...)` loop.

### Primary chain

5. One primary lane holds one invocation lock from its first chain request
   through P1 or the terminal continuation tail. Authorization may run on the
   sidecar while the primary lock remains owned; no primary foreign call may
   enter during that interval.
6. The exact cold sequence is A1, A2, I1, G1a, optional G1b, I2, conditional
   W1, P1, off-chain X1/X2, and O. Deterministic short-circuits retain the
   step identifier with `status=skipped` in observability but append no model
   message.
7. The model receives one system message and alternating user/assistant
   messages. There are no tool-role messages and no assistant prefill.
8. The system head uses one compact canonical JSON encoding
   (`ensure_ascii=False`, `sort_keys=True`, separators `(',', ':')`) and
   contains only the stage-independent `engine_manual` followed by
   `character_identity`. The first non-skipped cognition question carries the
   observation packet exactly once. Each later carrier first appears with its
   owning stage. No timestamp, invocation id, trace id, random id, route
   name, retry counter, or stage-specific repair text appears in the system
   head. Dynamic state, scene, evidence, affordances, and episode fields never
   enter the system message.
9. Evidence and the typed conversation frame first appear at A1; character
   constraints, relationship state, and relation-facing domains first appear
   at A2; deterministic state plus goal-only continuity first appear at G1;
   bid indexes first appear at W1; and action/resolver capabilities first
   appear at P1. Private continuity, past-dialog cognition context, character
   sleep phase, group-engagement context, required-selection operations, and
   branch guidance remain goal-scoped. All carriers stay visible downstream
   through the append-only transcript after their first consumer.
10. The primary model's accepted raw assistant text is retained byte-for-byte
    in the session transcript after canonical parsing and semantic validation.
    Rebuild uses those exact accepted bytes. Deterministic interlude products
    are rendered canonically in the next user question.

### Appraisal stages and products

11. The frozen family order is:
    `event_agency`, `goal_threat_outcome`,
    `epistemic_comparison_memory`, `relationship_social`,
    `moral_identity`, `existential_drive`.
12. A1 is the fixed world-facing stage and A2 is the fixed relation-facing
    stage. The runtime grouping setting and 1/2/3/6 registry are removed; the
    stage boundary is semantic rather than tunable.
13. Each planned family returns exactly `propositions` and `deltas`. Each
    array retains the V2 maximum of eight rows and the same V2 fields, kinds,
    handles, paths, and numeric domains. Empty arrays mean no additional
    supported product. Model-facing `question_id` echoes, nullable product
    pairs, null-null terminators, and exact-repeat loop instructions are
    omitted. Canonical V2 aggregate validation/reduction and every downstream
    emotion, relationship, goal, planning, authorization, and resolver owner
    remain unchanged.
14. One A1/A2 request consumes attempt one for every listed family. Structural
    failure rolls back that answer and directly asks each affected family once
    through an explicit owning-stage singleton recovery. That request consumes
    the family's second and final attempt. There is no same-group retry or
    topology ladder. Accepted sibling families are never regenerated; an
    exhausted optional family is omitted with its existing typed warning.
15. The model-facing system anchor contains only stage-independent rules.
    Every stage question carries its local objective, exact output schema,
    relevant state subset, candidate index, allowed handles, proposition-kind
    meanings, evidence domain, and writable paths adjacent to the answer.
16. Valid but reducer-rejected appraisal products remain visible in the accepted
    assistant answer. I1 lists only qualitative accepted/rejected counts and
    state bands; the rejected item never mutates state.

### Goals, collapse, and planning

17. G1a emits only the ordinary bid and owns exactly one
    `relational_willingness.v2`. Typed required-selection uses the V2
    specialized selection form and deterministic-only JSON parsing.
18. G1b emits one bid for each active branch in frozen
    `branch_order_key(...)` order, capped by the unchanged V2 roster limit.
    The top-level array order must equal the question roster. There are no
    winner, score, priority, rank, or collapse fields in G1a/G1b.
19. G1b runs even when G1a is relationship-sensitive. I2 then applies the
    existing deterministic ordinary-primary sensitive collapse and downstream
    effect denial; this preserves competing-bid observability.
20. A structurally malformed G1 answer is removed from the next request.
    Goal generation has three total requests for each participating branch.
    A G1b request consumes one attempt for every branch listed in that request.
    On final exhaustion, deterministic code may omit an entire invalid
    optional branch and append a canonical assistant projection containing
    only independently validated sibling bids. It may not repair fields inside
    an invalid bid. Required-branch recovery uses the unchanged V2 complete-
    sibling rule and warning; zero complete eligible bids fails before state
    commit.
21. W1 runs only when I2 has at least two complete bids and the turn is not
    relationship-sensitive. Zero bids follow the typed required failure; one
    bid is selected deterministically; sensitive turns use decision 19.
22. P1 emits the exact V2 intention, up to three action requests, up to three
    resolver requests, goal resolution, pending/progress carriers,
    `start_in_background`, selected operation, targetless-group response, and
    expression/residue inputs required by output projection. P1 receives no
    runtime permission to execute work.
23. `goal_resolution == answerable_now` suppresses optional resolver requests
    before authorization. Non-accepting relational stances use the unchanged
    effect-suppression contract.

### Sidecar and L1

24. A sidecar is either absent or one complete route bundle. Its normalized
    `(base_url, model)` must differ from the primary lane. A partial bundle or
    identical lane fails service construction.
25. Without a sidecar, L1 is skipped, JSON repair is deterministic-only, and
    action/resolver authorization denies every request. P1 can still select
    speech, silence, or private handling under the normal contract.
26. `subconscious_enabled` defaults to `False` and becomes effective only with
    a sidecar. Enabling it adds no new public input/output field.
27. L1 starts after canonical input validation and before waiting for the
    primary lane. It receives only current percept text, qualitative affect
    bands, a bounded boundary summary, and the supplied evidence-handle list.
    Its exact output is `L1ResidueV1` in the Contracts section.
28. The harness checks `task.done()` without awaiting immediately before A1.
    If a valid result exists, A1 receives it. Otherwise it checks once before
    G1a. A result still unavailable is cancelled/dropped. L1 failure, timeout,
    malformed output, or cancellation adds a warning and never delays or
    changes deterministic control flow.
29. L1 is advisory narrative. It cannot create evidence, facts, stance,
    willingness, a branch, a permission, an action, a resolver request, or a
    response route. Its handles must be a subset of supplied evidence handles.

### Resolver recurrence and sessions

30. The external resolver loop and `CognitionCoreInputV2` recurrence fields
    stay unchanged. V3 owns an internal process-local session registry keyed
    by a SHA-256 digest of episode id, state scope, and validated owner
    identity. Raw owner ids do not enter diagnostics.
31. Every successful cold invocation stores the exact accepted transcript,
    typed products, original input digest partitions, last output, the expected
    next mutable state, the next admissible cycle index, attempt ledger, budget
    ledger, re-anchor token, roster, and expiry. For the normal cold input at
    cycle `0`, the next admissible index is `1`; a cold rebuild at input index
    `N` stores `N + 1`. It has one owner at a time.
32. A cycle reattaches only when all conditions hold: episode and scope match;
    cycle index equals the stored next admissible value; mutable state equals
    the immediately preceding validated output's
    `state_update.replacement_state`; every immutable input field and digest
    partition matches its cold-session value; the prior relational-willingness
    carrier matches; prior evidence
    remains an exact ordered prefix; and every added row is a validated
    `resolver_observation` projection with a new handle. Any mismatch records
    `session_rebuilt_input_divergence` and executes a cold invocation from the
    supplied canonical input.
33. Reattached cycle order is R-observation, R-delta-appraisal, deterministic
    reduction, R-bid-revision in the existing roster order, I2, conditional
    W1, fresh P1, X1/X2, and O. Delta appraisal may cite only newly appended
    evidence handles plus persistent entity/role handles needed by the exact
    V2 contract. It cannot re-appraise unrelated old evidence.
34. The revision roster is the stable union of the prior participating roster
    and newly active final branches, sorted by `branch_order_key(...)`.
    Ordinary is always first and carries forward the original current-turn
    willingness. Every participating bid is revised because new resolver
    evidence is visible to the semantic owners.
35. A terminal resolver cycle still performs the short R decision tail so the
    observation re-enters cognition. It does not rerun the cold A1/A2 anchor
    path.
36. Session capacity is `256`. Expired and least-recently-used idle sessions
    are evicted. TTL is
    `COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS * COGNITION_RESOLVER_MAX_CYCLES + COGNITION_V3_TURN_DEADLINE_SECONDS + 30`.
    This is at least the maximum capability waiting period. Concurrent claims
    on one session permit the first claimant; later claims cold-rebuild and
    record `session_rebuilt_concurrent_owner`.
37. The existing `cognition_resolver.guardrail` becomes engine-neutral only at
    its service pass-through type. It declares `ServicesT = TypeVar(...)`;
    `ParentCognitionRunner[ServicesT]` accepts
    `(CognitionCoreInputV2, ServicesT)` and
    `run_guarded_cognition(..., services: ServicesT,
    runner: ParentCognitionRunner[ServicesT])` passes that same object through
    unchanged. The guardrail imports neither services dataclass. Eligibility,
    one replay token, checkpoint digest, error codes, safe checkpoint, and
    `CognitionCoreInputV2` / `CognitionCoreOutputV2` stay unchanged.
38. V3 reserves every model attempt through the existing epoch-aware
    `cognition_core_v2.model_attempt_policy` context using its stable V2
    semantic-owner/branch coordinate. The V3 chain/session attempt ledger is a
    projection of those reservations plus chain step ids, never an independent
    resettable authority. Service-graph retries consume unused calls in the
    current epoch; the one parent-checkpoint replay uses the existing epoch 1;
    resolver cycles retain that epoch. A cold rebuild, session miss, or
    recurrence reattachment never resets an attempt count. V2 behavior and
    ledger schemas remain unchanged.

### Failure, budget, and cutover

39. Tail repair appends a deterministic error appendix to the original user
    question, removes the rejected assistant candidate, and resends the
    extended question. Appendices are monotonic and include attempt index,
    typed error code, exact field path, expected contract/domain, and permitted
    handles. They exclude rejected raw text. Repeated identical output or two
    consecutive empty outputs consumes the remaining local attempt budget.
40. Primary model/provider failure follows the stage disposition. Sidecar
    failure follows decision 25 for that call. Cancellation propagates after
    owned tasks are cancelled and session ownership is released.
41. `COGNITION_V3_TURN_DEADLINE_SECONDS` defaults to `240`, accepts `30..600`,
    and is checked before every new model request, repair, fallback group,
    sidecar authorization request, re-anchor, and recurrence model step. An
    expired deadline prevents new model work but never skips deterministic
    validation/reduction of already accepted products, session release, output
    validation, or commit-owned cleanup. Expiry produces the best
    contract-valid terminal decision from accepted products when that stage
    owns a degraded disposition; required-selection/zero-bid/state invariant
    failures remain unrecoverable. L1 is observed only with `task.done()` and
    is never awaited for the deadline.
42. The normal total request-window ceiling is 50,000 tokens, including the
    owning step's reserved completion cap. Before every primary request, admit
    only when `estimated_prompt_tokens + step_max_completion_tokens <= 50_000`.
    Resolver recurrence or an oversized evidence registry that cannot fit the
    normal ceiling may activate the 65,000 total ceiling once when the declared
    serving window is at least 65,000. A 50,000–64,999 serving window runs the
    normal tier only; extension pressure continues through projection,
    re-anchor, then typed context-limit disposition without a mid-turn model
    reload. The configured primary context window must be at least 50,000.
43. One re-anchor token is shared by degeneration recovery and context
    pressure. A second request for either purpose raises the typed V2
    `CognitionContextLimitError` with the owning stage disposition.
44. V2 remains default until the Cutover Gate. Performance or quality evidence
    unavailable on a real eligible model blocks cutover; fixture-only evidence
    cannot waive the gate.

### Evaluation integrity, semantic failure, diagnosis, and data provenance

45. **Integrity contract.** Runtime prompts contain only the general semantic
    contract needed in production. They contain no pytest vocabulary, test or
    case identifier, fixture-only phrase, expected decision, score/rubric
    instruction, known regression answer, or development-process instruction.
    Production code cannot branch on test execution, manifest ids, fixture
    hashes, or expected outputs. Tests and the comparison harness preserve the
    canonical input and every eligible raw/parsed result; they cannot rewrite a
    model answer, select a favorable seed, rerun because of semantic quality,
    discard an unfavorable trial, weaken an assertion, or hide an artifact.
    The parent audits prompt diffs against the fixed manifests and records why
    every new instruction is a reusable architecture rule. A prompt correction
    derived from a failure must also pass one distinct countercase whose nouns,
    surface form, and desired semantic choice differ from the triggering case.
46. **Two-failure root-cause procedure and diagnostic order.** Two consecutive
    eligible local-model semantic failures for the same owner/behavior contract
    create one `local_semantic_reset.v1` record. Eligibility means the provider
    returned normally, canonical parsing completed, the stage input was valid,
    and the failure is behavioral rather than a harness or hard-boundary error.
    `Consecutive` means the next two eligible calls to that owner/contract in
    the engine currently being evaluated; interleaved control-engine calls and
    deterministic checks do not reset the sequence, while one eligible semantic
    success does.
    Before another E2E attempt, the parent:
    1. freezes both inputs, raw outputs, parsed outputs, prompt/version/model
       fingerprints, ledger positions, and reviewer judgments;
    2. steps back from the individual expected answer and audits the semantic
       question, ownership boundary, prompt load, context visibility/order,
       schema ambiguity, model/route capability, upstream data quality, and
       deterministic validator/materializer behavior;
    3. states a root-cause hypothesis and the smallest discriminating change or
       experiment that would falsify it;
    4. proves parser, validator, reducer, carrier, and handoff mechanics with
       deterministic or patched tests;
    5. reproduces the exact captured stage input through one real-LLM node test,
       runs and reviews that node individually, then runs one distinct
       countercase; and
    6. advances through a live subgraph/graph test only after the owning node
       works, and through the full real-LLM E2E path only after the smaller
       stage and patched handoffs work.
    An E2E artifact may localize the first failure but cannot be used as the
    iterative diagnostic harness. The reset closes only with the root-cause
    record, evidence-backed correction/disposition, individual stage artifacts,
    and countercase result.
47. **Inherited-failure classification and first correction.** Gate 1 creates
    an immutable `v2_semantic_baseline_defects.v1` registry before target V3
    production edits. A defect is inherited only when the same fixed case,
    behavior contract, and rubric dimension score `0` in at least two of the
    three sealed V2 trials; a single V2 miss, a post-implementation discovery,
    or a merely similar failure cannot be relabeled inherited. For an inherited
    model-behavior defect, the parent first attempts a general V3 correction in
    the V3 prompt geometry, context shaping, grouping, or orchestration and
    proves it with Decision 46. Contract preservation means preserving shape,
    ownership, validation boundaries, and deterministic authority; it does not
    require reproducing a known bad V2 semantic judgment. When evidence instead
    identifies a shared deterministic V2 validator/reducer/materializer defect,
    the parent amends the plan surface if needed and fixes the canonical shared
    owner plus V2 callers and V3 consumers atomically with direct-owner and
    propagation tests. A V3-only shim, mapper, exception, or test-specific
    prompt is forbidden.
48. **Exact semantic acceptance floor.** The fixed quality run contains exactly
    `24 cases * 3 V3 trials = 72 V3 trials`; every completed eligible trial
    remains in the denominator. A V3 trial is a semantic success only when its
    behavior contract is satisfied and every applicable behavioral dimension
    scores at least `1`. Acceptance requires the exact rational threshold
    `semantic_successes / 72 >= 0.95`, which means at least `69` semantic
    successes and at most `3` semantic failures. Every residual failed trial
    must match a Decision-47 baseline-defect case/contract/dimension, have a
    completed V3 correction attempt and root-cause record, and enter the
    deferred residual register with owner, evidence, consequence, and follow-up
    test. Any new V3-only semantic failure blocks Gate 7 regardless of the
    aggregate rate. Any hard schema, state, role/target, evidence/provenance,
    privacy, permission, availability-claim, relationship-stance, required-
    literal, authorization, or effect-safety failure blocks immediately and
    cannot use the three-trial allowance. Baseline-defect dimensions remain in
    raw scores and the 72-trial rate, but are excluded by their presealed ids
    from comparative non-regression means; every baseline-clean capability
    group and the overall baseline-clean V3 mean must be at least V2. Report
    all unfiltered means alongside the gated calculation.
49. **Production-data copy contract.** Data absence cannot be filled with a
    convenient mock when the required real source exists in production. The
    parent first uses the matching repository export (`export_dialog_trace_review_input`,
    `export_llm_trace`, `export_chat_history`, `export_user_profile`,
    `export_user_memories`, `export_character_state`, or bounded
    `export_collection`) through configured read-only access. Trace-led pulls
    begin with the protected exact-correlation manifest and never select the
    globally newest row. Each `production_data_extract.v1` record contains the
    source script, collection or protected source surface, exact non-secret
    correlation/filter, UTC window, projected fields, row count, redactions,
    raw-artifact path/hash, and sanitized-fixture path/hash when one is derived.
    Credentials, endpoints, embeddings, unrelated rows, and raw identifiers
    stay out unless the semantic contract specifically requires that field.
    Raw exports remain access-limited and uncommitted under `test_artifacts/`;
    only reviewed, minimal, pseudonymized derivatives may enter fixtures. The
    copy is evidence input only and never writes back to production.

## Target State

### Ownership picture

```text
adapter/debug client
  -> brain service / queue / intake / RAG / Stage 0
  -> persona connector builds canonical CognitionCoreInputV2
  -> closed engine selector
       v2 -> existing CognitionCoreServicesV2 and V2 executor
       v3 -> CognitionChainServicesV3
               sidecar: optional L1
               primary FIFO lane:
                 anchor -> A1 -> A2 -> I1 -> G1a -> G1b -> I2
                        -> W1 when eligible -> P1
               sidecar: X1/X2 authorization and optional JSON repair
               resolver recurrence:
                 append observation -> delta appraisal -> revise bids -> P1
               O: exact CognitionCoreOutputV2 projection and validation
  -> external resolver executes at most one admitted request and re-enters
  -> one final cognition-state commit
  -> unchanged action/L3/dialog/persistence/delivery path
```

### Volatility-ordered message head

The chain begins with exactly two messages:

1. One `SystemMessage` with `engine_manual` followed by
   `character_identity`. `engine_manual` contains the chain procedure,
   semantic ownership, output discipline, failure rules, and closed contract
   vocabularies and contains no run data. `character_identity` contains
   `core`, `personality`, `boundaries`, and `self_image`, in that order. The
   system bytes are identical for identical engine release and character
   identity and exclude all mutable state, relationship, scene, episode,
   evidence, affordance, trace, deadline, attempt, and route values.
2. One first `HumanMessage` containing, in order,
   `constraints_and_operational_state`, `relationship_and_mutable_state`,
   `episode_and_scene`, `evidence_and_affordances`, and the A1 question. The
   four data sections retain the meanings defined by the governing
   architecture. When no A1 family is planned, this message carries the next
   model-owned question instead; when the invocation has no model-owned
   question, no chain request is issued.

Every section is an exact object with a fixed key set. Empty optional values use
their canonical empty list/object/string rather than omitting keys. Handle
registries retain their input order. Unordered mappings sort by key. Numeric
state not authorized for model visibility remains behind deterministic
qualitative projection. Every later row alternates assistant/user; interlude
notices are prefixed to the next user question rather than emitted as a
standalone consecutive user message.

### Canonical step registry

| Step id | Owner | Input visibility | Output | Calls |
|---|---|---|---|---:|
| `A1` | primary chain | anchor + world-family questions + optional ready L1 | per-family V2 micro-item batches | 0 or 1 plus one repair; fallback groups bounded separately |
| `A2` | primary chain | accepted A1 + relation-family questions | per-family V2 micro-item batches | 0 or 1 plus one repair; fallback groups bounded separately |
| `I1` | deterministic | accepted appraisal items + original state | replacement-state candidate, maintenance, 21 emotions, ≤600-char notice | 0 |
| `G1a` | primary chain | prior transcript + goal-only carriers + optional ready L1 | ordinary bid, willingness, optional selected operation | 1..3 |
| `G1b` | primary chain | accepted ordinary bid + ordered active roster/guidance | ordered active-branch bids | 0..3 |
| `I2` | deterministic | complete bids + active goals/current matter | admitted candidates and collapse eligibility | 0 |
| `W1` | primary chain | complete bids already present | selected bid handle + bounded reason | 0..3 |
| `P1` | primary chain | selected/supporting bids + affordances | exact planning envelope | 1..3 |
| `X1` | sidecar | fresh minimal action authorization packet | exact V2 authorization decisions | 0..3 |
| `X2` | sidecar | fresh minimal resolver authorization packet | exact V2 authorization decisions | 0..3 |
| `R<n>` | primary chain | prior transcript + new observation | delta appraisal, revisions, fresh plan | bounded by owning appraisal/goal/plan caps |
| `O` | deterministic | validated typed products | `CognitionCoreOutputV2` | 0 |

Interlude notices are included in the next user message because the transcript
must alternate user/assistant roles. A skipped model step does not add a dummy
message.

### Completion caps

Per-step configs are produced with `dataclasses.replace(...)` from their lane
config. Endpoint, credential, model, sampling, thinking, and lane identity are
unchanged. Only `stage_name`, `max_completion_tokens`, and the remaining
bounded timeout change.

| Owner | Cap |
|---|---:|
| A1/A2 or fallback appraisal group | 4,096 |
| G1a/G1b or recurrence bid revision | 8,192 |
| W1 | 2,048 |
| P1 or recurrence P1 | 8,192 |
| L1 | 1,024 |
| X1/X2 | 1,024 |
| JSON repair | minimum of failed owner's cap and 8,192 |

Both configured lane caps must be at least the greatest derived cap they may
serve. Thinking is disabled for every derived V3 config.

## Contracts And Data Shapes

### V3 services

```python
@dataclass(frozen=True)
class CognitionChainServicesV3:
    llm: LLMInvoker
    chain_lane: LLMCallConfig
    sidecar_lane: LLMCallConfig | None
    subconscious_enabled: bool = False
    turn_deadline_seconds: int = 240
```

Construction validates exact dataclass fields, a non-empty chain route, chain
thinking disabled, `context_window_tokens >= 50000`, each step completion cap,
normal total-ceiling admission, conditional extended-tier availability only
when `context_window_tokens >= 65000`, sidecar completeness, sidecar thinking
disabled, sidecar/chain identity inequality, required lane caps, and the
`30..600` turn-deadline range. The selected connector injects the turn
deadline from `CognitionV3RouteSettingsV1`; the
facade never rereads route environment. API keys remain excluded from repr,
diagnostics, persistence, and cache identity.

Configuration is engine-conditional and exact:

| Setting | Required/default when V3 is selected |
|---|---|
| `COGNITION_V3_CHAIN_LLM_BASE_URL` | required non-empty URL |
| `COGNITION_V3_CHAIN_LLM_API_KEY` | required non-empty credential |
| `COGNITION_V3_CHAIN_LLM_MODEL` | required non-empty model |
| `COGNITION_V3_CHAIN_LLM_MAX_COMPLETION_TOKENS` | default `8192`, minimum `8192` |
| `COGNITION_V3_CHAIN_LLM_CONTEXT_WINDOW_TOKENS` | required, minimum `50000`; values `>=65000` enable the conditional extended tier |
| `COGNITION_V3_CHAIN_LLM_THINKING_ENABLED` | default `false`; `true` rejected |
| `COGNITION_V3_SIDECAR_LLM_BASE_URL` / `API_KEY` / `MODEL` | all absent or all non-empty |
| `COGNITION_V3_SIDECAR_LLM_MAX_COMPLETION_TOKENS` | default `8192`, minimum `8192` when sidecar exists |
| `COGNITION_V3_SIDECAR_LLM_THINKING_ENABLED` | default `false`; `true` rejected |
| `COGNITION_V3_SUBCONSCIOUS_ENABLED` | default `false`; requires sidecar when `true` |
| `COGNITION_V3_TURN_DEADLINE_SECONDS` | default `240`; range `30..600` |

The loaders return these closed immutable settings shapes before any
`LLMCallConfig` is constructed:

```python
@dataclass(frozen=True)
class CognitionRouteSettingV1:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    max_completion_tokens: int
    thinking_enabled: bool
    context_window_tokens: int | None

@dataclass(frozen=True)
class CognitionV3RouteSettingsV1:
    chain: CognitionRouteSettingV1
    sidecar: CognitionRouteSettingV1 | None
    subconscious_enabled: bool
    turn_deadline_seconds: int
```

`load_cognition_v2_route_settings()` returns a mapping with exactly these
twelve keys: `appraisal_event_agency`, `appraisal_relationship_social`,
`appraisal_moral_identity`, `appraisal_goal_threat_outcome`,
`appraisal_epistemic_comparison_memory`, `appraisal_existential_drive`,
`goal_ordinary_response`, `goal_active_branch`, `workspace_collapse`,
`action_planning`, `action_authorization`, and `resolver_authorization`;
`context_window_tokens=None` on every row.
`load_cognition_v3_route_settings()` returns `CognitionV3RouteSettingsV1`.
Both functions validate their selected family completely and fail startup with
the missing selected variable name; neither reads the inactive family.

`config.py` parses `COGNITION_CORE_ENGINE` before cognition route variables.
It defines two exact engine-selected loaders,
`load_cognition_v2_route_settings()` and
`load_cognition_v3_route_settings()`. Only the selected loader reads its
environment family; no inactive cognition route constant, alias, or
`LLMCallConfig` is constructed.
`persona_supervisor2_cognition.build_cognition_core_services()` returns the
selected engine's services object and constructs configs inside that branch;
it no longer constructs twelve module-level V2 configs before selection.

When V2 is selected, all twelve existing V2 cognition bundles remain required,
the V3 variables are not read, and `CognitionCoreServicesV2` is returned. When
V3 is selected, the complete V3 chain bundle is required, the sidecar bundle is
all-absent or all-present, the twelve V2 cognition stage variables are not
read, and `CognitionChainServicesV3` is returned. Shared non-core routes,
including surface/dialog, character carry-over, and the global JSON-repair
route used by non-V3 callers, retain their current requiredness.
`COGNITION_LLM_BASE_URL`, `COGNITION_LLM_API_KEY`, and
`COGNITION_LLM_MODEL` are explicitly a retained required shared non-core route
for both engines; its existing completion/thinking settings remain shared. It
is not one of the twelve V2 core stage bundles because it still serves the
connector's non-core carry-over path,
`internal_monologue_residue.recorder`, and
`nodes.persona_supervisor2_memory_lifecycle`, all imported by the real service
graph. Those consumers remain unchanged. The V3 chain never invokes this
generic route and uses the sidecar injection below instead of the global repair
config.
`llm_interface.route_report` reports only the selected engine's cognition
routes plus shared routes, including `COGNITION_LLM`. The route report labels
it `shared_non_core`, never `v2_cognition`. `control_console.brain_model_routes`
registers both families for deployment editing, keeps `COGNITION_LLM` in its
shared required group, and marks only the selected core family active in the
runtime descriptor; it never requires inactive credentials for a read.

The startup contract is proven in a subprocess that imports
`kazusa_ai_chatbot.service`, outside the repository directory and without
dotenv loading, with `COGNITION_CORE_ENGINE=v3`, all current retained shared
route variables (including `COGNITION_LLM_*`) plus the V3 chain bundle present,
and all 36 base-url/API-key/model variables for the twelve V2 stage bundles
absent. The inverse V2 subprocess keeps the twelve V2 plus shared routes,
removes every V3 route variable, and imports the same real service graph.

### LLM call config extension

`LLMCallConfig` gains one optional field:

```python
context_window_tokens: int | None = None
```

It appears after existing defaulted fields so current constructors remain
source-compatible. `LLInterface` transports no provider field for it; it is a
caller-owned serving-boundary declaration used by V3 guards and route
diagnostics. V2 configs retain `None`.

### Canonical JSON repair injection

The canonical parser remains the only JSON parse entrypoint. Extend the two
existing functions with these exact optional keyword-only parameters:

```python
def parse_json_with_llm(
    broken_string: str,
    *,
    expected_output_format: str | None = None,
    repair_trace_hook: JsonRepairTraceHook | None = None,
    repair_llm: LLMInvoker | None = None,
    repair_config: LLMCallConfig | None = None,
) -> dict: ...

def parse_llm_json_output(
    raw_output: str,
    *,
    expected_output_format: str | None = None,
    deterministic_only: bool = False,
    repair_trace_hook: JsonRepairTraceHook | None = None,
    repair_llm: LLMInvoker | None = None,
    repair_config: LLMCallConfig | None = None,
) -> dict: ...
```

`repair_llm` and `repair_config` are both supplied or both omitted. An injected
pair is passed by `parse_llm_json_output(...)` to `parse_json_with_llm(...)`
only after deterministic parsing fails. An omitted pair preserves the existing
global `JSON_REPAIR_LLM` behavior for V2 and all non-V3 callers. V3 passes the
sidecar `llm/config` pair for repair-permitted chain steps; when the sidecar is
absent, or for required selection, V3 passes `deterministic_only=True` and no
pair. The repair response is parsed deterministically and can repair only
syntax, wrapper, or object shape. A repaired object retains the original raw
assistant answer in the protected attempt record but appends the canonical
compact repaired object as the accepted assistant transcript row, so later
chain steps never consume malformed syntax or content invented outside the
supported raw keys/values.

### L1 residue

```python
class L1ResidueV1(TypedDict):
    schema_version: Literal["l1_residue.v1"]
    emotional_appraisal: str       # 1..120 chars
    interaction_subtext: str       # 0..200 chars
    salience_hints: list[str]      # 0..4 supplied evidence handles
    risk_flags: list[Literal[
        "boundary_pressure",
        "coercion_or_control",
        "privacy_or_secrecy",
        "physical_harm",
        "self_harm",
        "sexual_boundary",
        "relationship_rupture",
        "identity_conflict",
        "evidence_conflict",
    ]]
```

Lists are duplicate-free. Unknown fields/flags/handles fail the L1 contract
and drop the residue.

### Chain transcript and step record

The runtime owns an internal `ChainTranscriptV1` containing immutable message
rows, accepted typed products, a token ledger, attempt ledger, deadline, and
one re-anchor token. Mutation APIs are limited to `append_question`,
`accept_answer`, `rollback_tail_answer`, `append_interlude_to_next_question`,
and `reanchor`. No public mutable message list is exposed.

Each step record has exact keys:

```text
step_id, stage_kind, lane_kind, sidecar_stream_kind, status, attempt_count,
duration_ms, queue_wait_ms, in_flight_at_start,
prompt_chars, new_suffix_chars, estimated_prompt_tokens,
reserved_completion_tokens, estimated_total_context_tokens,
active_total_ceiling_tokens, extension_available, extension_used,
estimated_new_suffix_tokens, declared_shared_prefix_chars, cache_class,
parse_status, repair_count, disposition, warning_codes
```

`lane_kind` is `primary`, `sidecar`, or `deterministic`; `cache_class` is
`cold`, `changed_tail`, `reanchored`, `sidecar_isolated`, or `not_applicable`.
`sidecar_stream_kind` is `l1`, `json_repair`, `action_authorization`,
`resolver_authorization`, or `not_applicable`; queue/concurrency fields are
zero for deterministic steps and primary steps without a queue wait.
No prompt, output, evidence text, endpoint, key, raw id, or state document is
present.

### Session record

`ChainSessionV1` is process-local and never serialized to MongoDB. Exact
fields are:

```text
schema_version, session_key_digest, episode_id_digest, scope,
immutable_input_digest, original_evidence_digest,
expected_mutable_state_digest, expected_willingness_digest,
expected_cycle_index, accepted_messages,
accepted_products, accepted_evidence, current_roster, attempt_ledger, token_ledger,
last_cycle_delta_digest,
reanchor_used, last_output, created_monotonic, last_used_monotonic,
expires_monotonic, owner_token
```

The session contains prompt text because it is an in-memory performance
carrier. It never enters event logs, console payloads, or ordinary output.

### Recurrence digest and allowed cycle delta

Reattachment starts only after `validate_cognition_core_input(...)` succeeds.
The field classification is exhaustive and mechanically locked to
`CognitionCoreInputV2.__required_keys__ | CognitionCoreInputV2.__optional_keys__`:

| Exact `CognitionCoreInputV2` field | Classification | Reattachment rule |
|---|---|---|
| `schema_version` | immutable | exact equality |
| `episode` | immutable and session-key source | canonical value equality, including every nested field |
| `state_scope` | immutable and session-key source | exact equality |
| `mutable_state` | constrained cycle carrier | must canonically equal the immediately preceding validated `state_update.replacement_state`; no other mutation is admissible |
| `character_constraints` | immutable | canonical value equality |
| `character_identity_context` | immutable | canonical value equality |
| `character_operational_context` | immutable optional | presence and canonical value equality |
| `relationship_context` | immutable optional | presence and canonical value equality |
| `evidence` | append-only cycle carrier | prior accepted list must be an exact canonical prefix; exactly one new validated `resolver_observation` row is allowed per reattachment |
| `direct_facts` | immutable | canonical value equality |
| `available_actions` | immutable | canonical value equality and order |
| `available_resolver_capabilities` | immutable | canonical value equality and order |
| `resolver_context` | cycle carrier | validated bounded string may change; it is included in the cycle-delta digest |
| `runtime_capability_limits` | immutable optional | presence, order, and value equality |
| `resolver_goal_progress` | cycle carrier optional | must equal the prior accepted output's `resolver_goal_progress`, including absence |
| `required_resolver_evidence_dependency` | cycle carrier optional | exact validator plus selected-request, goal-continuation, and new observation-id binding |
| `current_turn_relational_willingness` | immutable optional | presence and digest must equal `expected_willingness_digest` |
| `resolver_cycle_index` | cycle carrier optional | normalized absent cold value is `0`; reattachment must equal `expected_cycle_index`, which exclusively means the next admissible input index |
| `pending_resolver_resume` | cycle carrier optional | exact validator; any present observation/resume binding must match the newly appended evidence row |
| `scene_context` | immutable | canonical value equality, including time and participant bindings |
| `private_continuity_context` | immutable | exact string equality |
| `past_dialog_cognition_context` | immutable optional | presence and exact string equality; resolver-RAG enrichment therefore cold-rebuilds |
| `group_engagement_action_context` | immutable optional | presence and canonical value equality |

The immutable projection has every immutable field as
`{"present": bool, "value": <validated value or null>}` in the table order,
plus `original_evidence_digest`. Canonical encoding is UTF-8 JSON with
`ensure_ascii=False`, `sort_keys=True`, and separators `(',', ':')`; the digest
is lowercase SHA-256. Missing optional values and present empty values are
different. Floats, non-string mapping keys, non-finite numbers, and values not
already admitted by the V2 validator fail before hashing.

After a cold or reattached invocation produces a fully validated terminal
output, the session requires
`output.state_update.expected_previous_state == incoming.mutable_state` and
stores the canonical value and SHA-256 of
`output.state_update.replacement_state` as the only expected state for the next
input. Reattachment compares both the digest and canonical value from
`last_output`, so a digest alone never authorizes state. A repeated old state,
an independently loaded DB state, a partially applied replacement, or any
mutation not equal to that exact validated replacement cold-rebuilds. This is
the resolver loop's authorized state evolution; all other cross-cycle state
mutation remains a connector breach.

On a cold session, `accepted_evidence` is the complete validated evidence list
and `original_evidence_digest` is its canonical digest. On reattachment, the
incoming evidence must equal `accepted_evidence` row-for-row and byte-for-byte
under canonical encoding, followed by exactly one row whose
`evidence_ref.source_kind == "resolver_observation"`, whose `source_id` is
non-empty and absent from all accepted evidence, and whose handle is the next
canonical `e<N>` handle. Any inserted, removed, reordered, renumbered, or
mutated prior row, any additional non-resolver evidence, or more/less than one
new row triggers `session_rebuilt_input_divergence`. This deliberately sends a
resolver cycle that adds RAG/past-dialog evidence through the semantically
identical cold path.

The cycle-delta projection contains only the new evidence row,
`mutable_state`, `resolver_context`, `resolver_goal_progress`,
`required_resolver_evidence_dependency`, `resolver_cycle_index`, and
`pending_resolver_resume`, each with explicit presence. It uses the same
canonical encoding and is stored as `last_cycle_delta_digest`. After a valid
reattachment, the session replaces `accepted_evidence` with the full incoming
list, replaces its expected state with the validated replacement, and sets
`expected_cycle_index = incoming_resolver_cycle_index + 1` only after the
terminal output passes validation. The stored index always denotes the next
admissible input; admission requires equality and never adds another offset.
Repeated, skipped, decreasing, or otherwise out-of-order indices cold-rebuild.
A field absent from the table, a future public input field not
added to the table/test, an invalid allowed-delta binding, or any immutable
change always cold-rebuilds and records the exact divergent field name.

### Persisted cognition chain run

Create collection `cognition_chain_runs` with one sanitized document per V3
invocation:

```text
schema_version = cognition_chain_run.v1
chain_run_id
engine = v3
run_id
llm_trace_id
cognition_invocation_id
source_kind = live | self_cognition | debug | unknown
chain_model_name
sidecar_model_name
subconscious_enabled
appraisal_group_count
started_at
completed_at
terminal_disposition
steps[]                         # capped at 96 exact step records
ledger {
  declared_context_window_tokens,
  normal_total_ceiling_tokens,
  extended_total_ceiling_tokens,
  active_total_ceiling_tokens,
  extension_available,
  extension_used,
  max_estimated_prompt_tokens,
  max_reserved_completion_tokens,
  max_estimated_total_context_tokens,
  reanchor_used
}
sidecar {
  l1_stream_count,
  json_repair_call_count,
  action_auth_attempt_count,
  resolver_auth_attempt_count,
  queue_wait_ms_total,
  max_in_flight,
  l1_preempted_by_repair,
  cancellation_count
}
session_events[]                # capped at 16 closed values
degradation_markers[]           # capped at 32 closed values
warning_codes[]                 # capped at 32 sanitized codes
expires_at
```

Indexes are `chain_run_id` unique,
`(run_id, llm_trace_id, completed_at)`,
`(cognition_invocation_id, completed_at)`, `(engine, started_at)`, and TTL on
`expires_at`. Retention uses `AUDIT_LOG_TTL_DAYS`. Writes are best-effort and
bounded by the event-log write timeout policy; a persistence failure emits one
sanitized local warning and never changes the cognition result.

DB public helpers are exact:

```python
async def save_cognition_chain_run(document: Mapping[str, object]) -> bool: ...
async def get_cognition_chain_run(
    *, run_id: str, llm_trace_id: str
) -> dict[str, object] | None: ...
```

`chain_run_id` is generated once by the V3 invocation as
`cogchain_<uuid4 hex>` and is retained by every retry of that diagnostic write.
The write helper validates the complete schema and performs an idempotent
upsert by `chain_run_id`. An existing row is accepted only when its immutable
`run_id`, `llm_trace_id`, and `cognition_invocation_id` equal the candidate;
any conflict returns `False` and emits a sanitized local warning. The read
helper requires non-empty exact `run_id` and `llm_trace_id`, queries their
intersection, sorts by `completed_at` descending to select the terminal/latest
cycle for that graph, and returns only the bounded projection. An absent key,
no exact match, a cross-key mismatch, or read failure returns `None`; no global
latest lookup exists. Raw Mongo access stays inside `db`.

### Protected transcript

Add `cognition_chain_transcript.v1` as one protected `llm_trace_steps` row per
V3 invocation when protected tracing is enabled. In `full` mode it contains
the exact system/user/assistant messages, rejected candidate only inside its
owning attempt record, parser/validator outcomes, interlude products, config
identities without secrets, and terminal disposition. In `metadata` mode the
same row retains message hashes, lengths, step metadata, and dispositions
without message content. In `off` mode it emits no row. It uses
`DEBUG_LOG_TTL_DAYS`, the existing trace access boundary, and background
best-effort persistence. Existing `llm_trace_steps` document shape and
ordinary per-call rows remain unchanged.

### Event log family

Add family `cognition_chain` with public keyword-only recorder
`record_cognition_chain_event(...)`. Allowed arguments are:

```text
run_id, cognition_invocation_id, terminal_disposition, chain_model_name,
sidecar_model_name, step_count, repair_count, cold_start_count,
prompt_chars_total, new_suffix_chars_total, prefix_share_ratio,
max_estimated_prompt_tokens, max_reserved_completion_tokens,
max_estimated_total_context_tokens, active_total_ceiling_tokens,
extension_available, extension_used,
reanchor_used, session_disposition, duration_ms, deadline_ms,
deadline_consumption_ratio, l1_stream_count, json_repair_call_count,
action_auth_attempt_count, resolver_auth_attempt_count,
sidecar_queue_wait_ms_total, sidecar_max_in_flight,
l1_preempted_by_repair, sidecar_cancellation_count, warning_codes, occurred_at
```

The recorder accepts no generic payload/metrics mapping. It persists no raw
prompt, answer, message, evidence, state, user id, endpoint, or credential.
The V3 diagnostics owner is added to the approved instrumentation table.

### Brain-service and console contract

`OpsLatestCognitionGraphResponse` gains optional
`cognition_chain_run: dict[str, Any] | None` and
`self_cognition_chain_run: dict[str, Any] | None`. The service route resolves
each row independently only when its owning graph contains non-empty `run_id`
and `llm_trace_id`, then awaits the exact intersected DB read. A graph with a
missing key, an exact-match miss, or a read failure receives `None`; it never
borrows the other graph's row or a process-global latest row. The existing
graph response remains available when either diagnostic read fails.

The existing runtime-status response gains one typed optional field named
`cognition_engine`, whose descriptor is:

```text
schema_version = cognition_engine_descriptor.v1
engine_id
chain_model_name
sidecar_model_name
sidecar_enabled
subconscious_enabled
appraisal_group_count
chain_context_window_tokens
normal_budget_tokens = 50000
extended_budget_tokens = 65000
turn_deadline_seconds
```

`normal_budget_tokens` and `extended_budget_tokens` are total request-window
ceilings. For each step, the ledger reserves that step's completion cap inside
the active ceiling and records prompt estimate, completion reservation, total
reservation, tier, declared serving window, and whether the extended tier is
available. Availability is derived mechanically as
`chain_context_window_tokens >= extended_budget_tokens`; the console does not
infer it from a model name or provider claim.

For V2, model names are the stable sorted unique set of its configured stage
models, chain-specific numeric fields are zero, and sidecar fields are false or
empty. The descriptor is built from selected configuration at the service
boundary. It contains no endpoint or credential and performs no runtime query
into the cognition engine.

The console adds a strict `CognitionChainRunSnapshot` projector and one panel
under each existing response/self-cognition graph view. Each panel consumes
only its paired `cognition_chain_run` or `self_cognition_chain_run`. The panel
shows engine/model names, ordered step/status/timing rows, ledger usage,
prefix-share/cache class, session events, warnings, and terminal disposition.
It renders `status=not_reported` on missing or mismatched correlation and never
falls back to another row. It renders no raw prompt, answer, evidence, ids
beyond existing closed run references, endpoint, or credential. The console
never imports cognition or DB modules.

The console runtime overview projects the engine descriptor as a read-only
configuration summary beside the chain panel. Unknown future additive fields
are ignored; unknown schema versions produce `status=not_reported` rather than
an inferred configuration.

## Runtime Or Resource Constraints

### Lane coordination

1. `lane.py` owns a process-local primary registry keyed by `(id(llm),
   normalized base_url, model)` and one `asyncio.Lock` plus FIFO ticket counter
   per key.
2. The primary lock covers the complete primary sequence and each recurrence
   tail. Cancellation releases ownership in `finally`.
3. The same module owns a separate `SidecarCoordinator` registry keyed by the
   sidecar `(id(llm), normalized base_url, model)`. It has one FIFO
   `asyncio.Lock`, so V3 issues at most one sidecar model request at a time per
   resident sidecar. Sidecar/primary construction-time identity inequality
   keeps that stream off the primary lane. Fresh-context sidecar calls remain
   cache-indifferent; serialization is the reference deployment's one
   multiplexed stream, not a cache-affinity claim.
4. Logical sidecar admission is exact: one L1 producer per cold invocation;
   zero or one X1 action-authorization producer and zero or one X2
   resolver-authorization producer per cognition cycle; and zero or one JSON
   repair call per raw assistant candidate whose deterministic parse failed.
   X1 and X2 run in that order and each retains the V2 maximum of three model
   attempts and deny-all exhaustion. A JSON repair has one call and no retry;
   it consumes no new semantic attempt but is legal only while the owning
   producer's already-reserved attempt is live. It cannot run after owner
   exhaustion. Thus repair's invocation maximum is mechanically bounded by
   the sum of raw candidates admitted by the existing producer attempt ledger,
   never by a resettable repair ledger.
5. L1 claims the sidecar first and may overlap primary work. Before a JSON
   repair claims the sidecar, the invocation cancels any unfinished L1 task,
   awaits its `CancelledError`/`finally` lock release, records
   `l1_preempted_by_repair`, then admits repair. Repair therefore preempts L1;
   they never coexist. The A1/G1 join/drop rules remain unchanged when no
   repair occurs. X1/X2 start only after P1 and after L1 is joined or dropped,
   so neither overlaps L1 or repair inside one invocation.
6. V3 calls the synchronous canonical repair path only in a worker thread while
   holding a sidecar claim, after an initial `deterministic_only=True` parse
   failed. The repair config timeout is the lesser of the owning stage timeout
   and remaining turn deadline. The worker future is shielded: if its awaiting
   task is cancelled, cleanup retains the sidecar claim, drains the bounded
   provider call to completion, discards its result, then propagates
   cancellation. The lock is never released while the provider thread is still
   using the sidecar. The injected sidecar `llm/config` pair remains the only
   model repair route; no V3 synchronous caller invokes it outside this claim.
7. Invocation cancellation stops admission, cancels the owned L1 and native
   async authorization tasks, removes queued tickets, drains any already
   running synchronous repair as specified above, awaits every owned task, and
   releases sidecar then primary ownership in `finally`. Cancellation never
   triggers a semantic regeneration by itself. A sidecar provider failure
   releases its claim before applying L1-drop, JSON-unrepaired, or X1/X2
   deny-all disposition; subsequent bounded producer streams may still claim
   the lane.
8. A task cannot reacquire a lane it owns. Recursive primary or sidecar claims
   are typed execution errors. Deadline checks occur before ticket admission;
   deadline expiry removes a queued ticket without disturbing FIFO order.
9. Every sidecar step records `queue_wait_ms`, `in_flight_at_start` (always
   `1`), and its logical stream kind. Per invocation diagnostics record
   `l1_stream_count`, `json_repair_call_count`, `action_auth_attempt_count`,
   `resolver_auth_attempt_count`, `sidecar_queue_wait_ms_total`,
   `sidecar_max_in_flight`, `l1_preempted_by_repair`, and
   `sidecar_cancellation_count`. Tests prove every admission limit, FIFO order,
   preemption/cancellation cleanup, maximum sidecar concurrency one, and that
   no sidecar request enters the primary request sequence.

### Deterministic estimator

The initial estimator is fixed:

```text
base_units =
  CJK_codepoint_count
  + ceil(non_CJK_utf8_byte_count / 4)
  + 16 * message_count
  + 32
estimate = ceil(base_units * CALIBRATION_MULTIPLIER)
```

`CJK_codepoint_count` includes Han, Hiragana, Katakana, Hangul, and CJK
punctuation ranges named in `budget.py`. The calibration command uses a frozen
48-payload corpus: 12 anchor-only, 12 A/G tails, 12 repair/long-context, and 12
resolver-observation payloads. It records server-reported prompt tokens and
chooses `CALIBRATION_MULTIPLIER` as the next 0.05 above the maximum
`actual/base_units` ratio, with a minimum of `1.00`. The chosen numeric constant
and calibration artifact digest are committed together. Acceptance requires
zero underestimates on the 48 calibration payloads and a separate 16-payload
holdout, plus median overestimate no greater than 35%. Failure blocks the plan;
the fixed formula remains authoritative for the parent.

### Budget ladder

1. Before the first request, fit once using the V2 order: remove supplemental
   goal-only context from the anchor; reduce scene, constraints, and identity
   at their existing semantic floors; then middle-truncate evidence text no
   lower than 96 characters. Preserve handles, provenance, source order,
   roles, timestamps, current-event facts, identity core, and boundaries.
2. Start every invocation with a 50,000 total request-window ceiling. Before
   each request, reserve the owning step's completion cap and require
   `estimated_prompt_tokens + reserved_completion_tokens <= 50000`.
3. When resolver recurrence or an oversized evidence registry cannot fit the
   normal tier, switch once to the 65,000 total ceiling only if the declared
   serving window is at least 65,000. Keep the extended tier active for the
   rest of that invocation. A smaller declared serving window records
   `extension_unavailable` and continues the bounded ladder without reloading
   or swapping a model mid-turn.
4. If an incoming resolver observation still does not fit, apply its existing
   bounded prompt projection. Record the truncation.
5. If pressure remains, consume the shared re-anchor token. Rebuild a tighter
   anchor and deterministic digest from validated appraisal receipts, I1 state
   notice, complete bids, collapse selection, last planning envelope,
   authorization verdicts, and bounded resolver observations. No raw reasoning
   prose or LLM summary enters the digest.
6. Remaining pressure raises `CognitionContextLimitError`. No previously sent
   message is edited in place.

### Serving-window defense

1. Client code refuses every request whose estimate plus completion cap exceeds
   the declared context window.
2. Before production cutover, run a dedicated one-at-a-time overflow probe
   against the exact deployed primary route. It sends a non-semantic synthetic
   payload above the declared window and requires an explicit provider
   rejection. A success response, silent rolling window, missing token usage,
   or inconclusive result blocks cutover.
3. Record backend build/model checksum, declared window, payload estimate,
   provider outcome, and timestamp in the protected validation artifact. Never
   run the over-window probe on a normal chat turn.

### Cache invariants

1. The system message bytes stay identical for identical fitted anchors.
2. A same-session continuation's prior message list is a byte-identical prefix
   of the next request, except after the one explicit tail rollback or
   re-anchor.
3. No primary foreign call occurs between requests.
4. Thinking is disabled and no assistant prefill is used.
5. Prefix metrics use exact serialized character hashes and new-suffix lengths;
   backend cache counters, when present, supplement rather than replace this
   proof.
6. Primary and sidecar model residency is an operator prerequisite for a
   sidecar-enabled deployment. Model swapping observed during the performance
   gate fails the environment fingerprint.

## Cutover Policy

- **Strategy:** big-bang V3 replacement behind the existing closed engine
  selector, followed by evidence-gated default cutover. V2 remains the whole-
  engine rollback path.
- **Pre-cutover default:** `COGNITION_CORE_ENGINE=v2`.
- **Cutover commit:** after Gate 7 acceptance, change the documented/default
  selector value to `v3` in one reviewed commit and retain explicit `v2`.
- **Data:** create only transient diagnostic indexes. Cognition state documents
  are byte-compatible; no backfill or migration runs.
- **Rollback trigger:** any state/output validation regression, authorization
  breach, role/target inversion, repeated context overflow, session divergence,
  new V3-only semantic failure, semantic failure outside the sealed inherited-
  residual register, material live-quality regression beyond the accepted
  Decision-48 allowance, or performance threshold breach during observation.
- **Rollback action:** deploy `COGNITION_CORE_ENGINE=v2`, restart the process so
  the selector binds V2, and retain V3 diagnostic rows for review. No record is
  rewritten or deleted.
- **Observation:** run at least 100 eligible V3 turns across user, group,
  resolver, self-cognition, and required-selection paths. Require zero hard
  contract/permission/state/effect failures, no new V3-only semantic failure,
  and no unexplained p95 regression before marking the plan complete. A sealed
  inherited residual remains follow-up evidence rather than a rollback trigger.
- **V2 retirement:** outside this plan.

## Execution Roles

### Root execution owner

- **Responsibility:** close the complete plan. The parent owns readiness
  probing, baseline sealing, scope and handoff control, fixtures/tests,
  live-model evidence, scoring, final code audit,
  evaluation-integrity auditing, semantic root-cause resets, production-data
  provenance, inherited-residual disposition, cutover, observation,
  documentation, plan status, and archive closure.
- **Owned surface:** plans, lifecycle records, governed artifacts, protected
  test artifacts, read-only readiness boundaries, named V3 runtime-environment
  bundles, verification, and review. Production source paths are read-only to
  this role and are assigned to the production implementation role below.
  Existing credentials remain in
  protected configuration and may be referenced under new V3 variable names;
  evidence records keys and sanitized fingerprints only. Keep paths remain
  read-only except when the parent first records an architecture-required plan
  amendment and exact impact row under Mandatory Rule 4.
- **Authority:** after owner approval and explicit production authorization,
  assign the complete Change Surface, directly edit non-production portions of
  it, run all deterministic/live/browser checks,
  perform sanitized read-only service/DB/LLM probes, apply the specified
  diagnostic schema/index additions, load or reload only the sealed candidate
  model identities to the plan-required context and simultaneous residency,
  deploy/cut over only at Gate 8, remediate findings, update the plan/registry,
  and archive on complete evidence. The authority excludes new semantic
  capabilities, weakened thresholds, compatibility fallbacks, credential
  disclosure, destructive production data operations, and effects outside the
  approved cutover/runbook.
- **Applicable skills:** every skill in Mandatory Skills, triggered at its
  named boundary and re-applied after compaction recovery.
- **Capability floor:** senior system ownership across Python/async local-LLM
  orchestration, V2/V3 cognition and resolver contracts, Mongo/index safety,
  API/telemetry redaction, frontend/browser QA, protected live evidence,
  deterministic statistics, deployment, rollback, and lifecycle management.
- **Independence requirement:** the parent does not author production-code
  changes and reviews the production subagent's diff and evidence. Blinded
  quality scoring and calculation verification remain distinct sealed parent
  passes. The earlier independent plan review remains historical pre-execution
  evidence and is represented only as such.
- **Acceptance output:** sealed baseline and environment fingerprint; complete
  scoped source/test/docs diff; all exact mapped nodes collected and passing;
  live quality/performance/overflow artifacts and sealed parent score sheet;
  production-data provenance records; inherited-defect and residual registers;
  every triggered local-semantic reset; evaluation-integrity audit; console/
  browser evidence; severity-ordered final parent audit and remediation re-
  check; 100-turn observation; rollback proof; completed checklist; and the
  plan archived under `development_plans/archive/completed/`.
- **Gate:** Gate 0 entry requires historical plan-review closure, owner
  approval, and explicit production authorization. Gate 0 then creates the
  closure goal, captures baseline status, and completes the readiness probe.
  The parent role exits only when Gates 0–8 and every Acceptance Criterion pass
  and the goal is marked complete after archive closure.

### Production implementation owner — plan-scoped fixed constraint

- **Responsibility:** implement and remediate the remaining production-code
  surface in bounded checkpoints, together with the directly coupled tests
  needed to prove each changed production boundary.
- **Owned surface:** only the explicit production and test files named in each
  parent handoff, within this plan's Create/Modify surface. Ownership returns
  to the parent at every checkpoint.
- **Authority:** modify assigned production and coupled test files, run the
  specified deterministic checks, and report exact diffs, results, deviations,
  and remaining risks. The role cannot change architecture, plan thresholds,
  gate meanings, cutover policy, credentials, production data, or lifecycle
  state.
- **Applicable skills:** every Mandatory Skill triggered by the assigned file
  set or verification activity.
- **Capability floor:** senior Python/async and local-LLM pipeline reasoning,
  exact contract preservation, repository-aware testing, and safe operation in
  a dirty shared worktree.
- **Independence requirement:** one reusable subagent performs all production
  code changes. It preserves concurrent work and never reverts edits outside
  its assigned slice. The root parent remains the accepting reviewer.
- **Resolved executor constraint:** for production-code handoffs after the
  2026-08-20 owner amendment below, use one reusable `gpt-5.6-luna` subagent
  with maximum reasoning at normal speed. A different model requires a later
  owner amendment.
- **Acceptance output:** a scoped production/test diff, exact mapped test
  evidence, style/static results, and a concise handback naming unresolved
  issues and the next safe checkpoint.
- **Gate:** entry requires a parent handoff containing the current baseline,
  exact owned file set, applicable skills and contracts, completed evidence,
  and acceptance checks. Exit requires parent inspection and acceptance of the
  returned diff and evidence.

The root parent advances the logical checkpoints sequentially and reuses the
same production implementation subagent across checkpoints whenever it remains
available. Review, live evidence, cutover, observation, and lifecycle closure
remain parent-owned.

## Test Impact And Traceability

### Required test ownership changes

Update `tests/ownership/source_test_impact_manifest.json` so every changed or
new production owner maps to at least one exact deterministic node below.
Remove entries whose contract names parallel waves, isolated sibling
transcripts, route checkpoints, or UTF-8 byte budgets. `deterministic unit`,
`deterministic integration`, `static contract`, and `patched E2E` are regular
non-live pytest modes. Gate 6 runs `pytest --collect-only -q` for every mapped
node and `scripts.validate_test_impact --run`; a missing node, unmapped changed
path, wildcard path, or nonzero collection result fails the gate.

### Core and shared-owner impact matrix

| Exact repository-relative source path | Semantic owner / changed contract | Exact deterministic pytest node | Mode | Regression prevented |
|---|---|---|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v3/__init__.py` | V3 public facade / exact V2 entrypoints plus V3 services | `tests/unit/cognition_core_v3/test_public_api.py::test_v3_exports_exact_v2_entrypoints_and_services` | deterministic unit | partial or internal-only public API |
| `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py` | V3 boundary / closed step, L1, session, run schemas | `tests/unit/cognition_core_v3/test_contracts.py::test_v3_contracts_reject_unknown_fields_types_and_enums` | deterministic unit | permissive or divergent payload shapes |
| `src/kazusa_ai_chatbot/cognition_core_v3/registry.py` | V3 topology / exact serial chain and frozen grouping maps | `tests/unit/cognition_core_v3/test_registry.py::test_registry_exposes_exact_serial_chain_and_groupings` | deterministic unit | old parallel-wave topology or unstable order |
| `src/kazusa_ai_chatbot/cognition_core_v3/transcript.py` | V3 transcript / append, accept, rollback, interlude, re-anchor | `tests/unit/cognition_core_v3/test_transcript.py::test_tail_rollback_preserves_prefix_and_excludes_rejected_candidate` | deterministic unit | rejected drafts or rewritten accepted bytes entering context |
| `src/kazusa_ai_chatbot/cognition_core_v3/execution.py` | V3 executor / one primary chain and epoch-aware attempt use | `tests/unit/cognition_core_v3/test_execution.py::test_executor_runs_one_serial_primary_chain_and_preserves_attempt_epochs` | deterministic unit | primary interleave or retry-budget reset |
| `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py` | appraisal / grouped A1/A2 plus exact V2 item reduction | `tests/unit/cognition_core_v3/test_appraisal.py::test_grouped_appraisal_uses_exact_v2_micro_item_contract_and_reduction` | deterministic unit | lost families, duplicate reduction, or invented semantics |
| `src/kazusa_ai_chatbot/cognition_core_v3/goal_cognition.py` | goals / G1a then ordered visible G1b and required selection | `tests/unit/cognition_core_v3/test_goal_cognition.py::test_g1a_precedes_sibling_visible_ordered_g1b` | deterministic unit | sibling isolation, ranking, or role inversion |
| `src/kazusa_ai_chatbot/cognition_core_v3/workspace.py` | workspace / zero-one-many and sensitive collapse | `tests/unit/cognition_core_v3/test_workspace.py::test_workspace_short_circuits_and_sensitive_collapse_match_v2` | deterministic unit | invalid arbitration or loss of ordinary stance authority |
| `src/kazusa_ai_chatbot/cognition_core_v3/action_selection.py` | planning / exact V2 P1 and sidecar X1/X2 | `tests/unit/cognition_core_v3/test_action_selection.py::test_p1_and_sidecar_authorization_match_v2_contracts` | deterministic unit | unbounded requests or unauthorized effects |
| `src/kazusa_ai_chatbot/cognition_core_v3/diagnostics.py` | observability / bounded redacted correlated records | `tests/unit/cognition_core_v3/test_diagnostics.py::test_chain_diagnostics_are_bounded_correlated_and_redacted` | deterministic unit | secret/raw-content disclosure or uncorrelated telemetry |
| `src/kazusa_ai_chatbot/cognition_core_v3/facade.py` | orchestration / final V2 output and one terminal disposition | `tests/unit/cognition_core_v3/test_facade.py::test_run_cognition_serial_chain_returns_valid_v2_output_without_partial_commit` | deterministic unit | partial state commit or invalid output projection |
| `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` | prompt owner / dynamic question packets, bounded carriers, and evaluation-integrity boundary | `tests/unit/cognition_core_v3/test_prompt.py::test_prompt_questions_are_bounded_contract_oriented_and_dynamic`; `tests/unit/cognition_core_v3/test_prompt.py::test_runtime_prompts_exclude_test_fixture_rubric_and_expected_answer_metadata` | deterministic unit | hidden carrier leakage, prompt-restatement drift, or test-tailored prompt cheating |
| `src/kazusa_ai_chatbot/cognition_core_v3/anchor.py` | anchor owner / byte-stable manual and identity system head | `tests/unit/cognition_core_v3/test_anchor.py::test_system_head_excludes_dynamic_turn_data_and_is_byte_stable` | deterministic unit | volatile scene/evidence in `SystemMessage` |
| `src/kazusa_ai_chatbot/cognition_core_v3/budget.py` | budget owner / calibrated estimator, total 50k/65k request ceilings, per-step completion reservation, re-anchor, overflow | `tests/unit/cognition_core_v3/test_budget.py::test_budget_ledger_calibration_extension_reanchor_and_overflow` | deterministic unit | prompt-plus-completion overflow, premature extension, or unbounded compaction |
| `src/kazusa_ai_chatbot/cognition_core_v3/lane.py` | lane owner / FIFO primary lock plus single-stream sidecar coordinator | `tests/unit/cognition_core_v3/test_lane.py::test_primary_lane_is_fifo_single_owner_and_sidecar_cannot_interleave`; `tests/unit/cognition_core_v3/test_lane.py::test_sidecar_stream_serializes_l1_repair_and_authorization_with_fixed_caps`; `tests/unit/cognition_core_v3/test_lane.py::test_l1_repair_preemption_and_cancellation_release_sidecar_fifo` | deterministic unit | foreign primary call, concurrent sidecar calls, cap bypass, or leaked cancellation ownership |
| `src/kazusa_ai_chatbot/cognition_core_v3/session.py` | session owner / exhaustive input digest, expected replacement state/index, exact reattach, cold rebuild, TTL/LRU | `tests/unit/cognition_core_v3/test_session.py::test_session_reattaches_exactly_and_cold_rebuilds_without_attempt_reset`; `tests/unit/cognition_core_v3/test_session.py::test_session_digest_classifies_every_input_field_and_rejects_each_unapproved_mutation`; `tests/unit/cognition_core_v3/test_session.py::test_session_accepts_prior_replacement_and_rejects_other_mutable_state`; `tests/unit/cognition_core_v3/test_session.py::test_session_cycle_index_accepts_zero_one_two_and_rejects_repeated_skipped_or_out_of_order` | deterministic unit | unclassified cycle drift, stale/foreign state, off-by-one recurrence, mixed episodes, or refreshed attempt authority |
| `src/kazusa_ai_chatbot/cognition_core_v3/subconscious.py` | L1 owner / advisory nonblocking bounded handles | `tests/unit/cognition_core_v3/test_subconscious.py::test_l1_is_advisory_nonblocking_and_handle_bounded` | deterministic unit | L1 delay, invented evidence, or semantic authority |
| `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py` | canonical V2 substrate / three exact public micro-item helpers | `tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_public_micro_item_helpers_preserve_existing_v2_contract` | deterministic unit | copied V3 validators or V2 semantic drift |
| `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py` | canonical V2 substrate / public selection materializer | `tests/unit/cognition_core_v2/test_goal_cognition.py::test_public_selection_materializer_preserves_roles_and_evidence` | deterministic unit | changed required-selection roles/evidence |
| `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py` | canonical V2 substrate / public partition validator | `tests/unit/cognition_core_v2/test_workspace.py::test_public_workspace_partition_validator_preserves_exact_partition` | deterministic unit | inconsistent V2/V3 partition rules |
| `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py` | canonical V2 substrate / public P1 validator/materializer | `tests/unit/cognition_core_v2/test_action_selection.py::test_public_action_plan_validator_preserves_existing_normalization` | deterministic unit | V2 output normalization drift |
| `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py` | canonical V2 substrate / public exact boolean validator | `tests/unit/cognition_core_v2/test_action_authorization.py::test_public_authorization_validator_requires_exact_boolean_coverage` | deterministic unit | missing/extra candidates or truthy coercion |
| `src/kazusa_ai_chatbot/llm_interface/contracts.py` | LLM boundary / local context-window declaration only | `tests/test_llm_interface_contracts.py::test_call_config_declares_context_window_without_transporting_it` | deterministic unit | invented streaming/provider timing contract |
| `src/kazusa_ai_chatbot/llm_interface/route_report.py` | route diagnostics / selected core family plus retained shared non-core routes | `tests/test_llm_interface_route_report.py::test_route_report_includes_only_selected_cognition_engine_routes_and_shared_generic_cognition_route` | deterministic unit | inactive credentials becoming startup requirements or generic route disappearing |
| `src/kazusa_ai_chatbot/utils.py` | canonical JSON parser / injected repair pair | `tests/test_utils.py::test_parse_llm_json_output_uses_injected_repair_pair_only_after_deterministic_failure` | deterministic unit | V3 using global repair route or stage-local parser |
| `src/kazusa_ai_chatbot/config.py` | configuration / engine-first lazy core loaders with retained shared `COGNITION_LLM` | `tests/test_config.py::test_v3_service_import_succeeds_with_shared_routes_and_without_twelve_v2_stage_bundles`; `tests/test_config.py::test_v2_service_import_succeeds_without_v3_routes` | subprocess deterministic | selected engine blocked by inactive core routes or live non-core consumer missing its shared route |
| `src/kazusa_ai_chatbot/cognition_core_selector.py` | engine selector / closed choice and no fallback | `tests/unit/test_cognition_core_selector.py::test_v3_failure_never_invokes_v2` | deterministic unit | hidden V2 fallback |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | connector / branch-local selected service construction | `tests/unit/nodes/test_persona_supervisor2_cognition.py::test_connector_builds_selected_engine_services_without_inactive_routes` | deterministic unit | eager twelve-route construction or double commit |
| `src/kazusa_ai_chatbot/cognition_resolver/guardrail.py` | resolver guardrail / engine-neutral service pass-through and existing epoch ledger | `tests/unit/cognition_resolver/test_guardrail.py::test_guardrail_passes_engine_neutral_services_and_preserves_v3_attempt_epochs` | deterministic unit | V2-only service typing or checkpoint retry reset |

### Persistence, service, and console impact matrix

| Exact repository-relative source path | Semantic owner / changed contract | Exact deterministic pytest node | Mode | Regression prevented |
|---|---|---|---|---|
| `src/kazusa_ai_chatbot/db/cognition_chain_runs.py` | chain-run store / idempotent ID and exact dual correlation | `tests/test_db_cognition_chain_runs.py::test_chain_run_upsert_is_idempotent_and_rejects_correlation_conflict`; `tests/test_db_cognition_chain_runs.py::test_chain_run_read_requires_exact_run_and_trace_ids` | deterministic unit | duplicate record or cross-episode latest read |
| `src/kazusa_ai_chatbot/db/__init__.py` | DB public API / exact chain-run helper exports | `tests/test_db_cognition_chain_runs.py::test_db_exports_exact_chain_run_helpers` | static contract | unowned DB entrypoint |
| `src/kazusa_ai_chatbot/db/bootstrap.py` | DB bootstrap / unique, correlation, invocation, engine, TTL indexes | `tests/test_db_cognition_chain_runs.py::test_chain_run_indexes_match_retention_and_correlation_contract` | deterministic unit | missing uniqueness, query support, or TTL |
| `src/kazusa_ai_chatbot/llm_tracing/chain_transcript.py` | protected trace / off-metadata-full capture modes | `tests/test_llm_chain_transcript.py::test_chain_transcript_capture_obeys_off_metadata_full_modes` | deterministic unit | ordinary-path raw prompt exposure |
| `src/kazusa_ai_chatbot/llm_tracing/__init__.py` | trace facade / scoped transcript export | `tests/test_llm_chain_transcript.py::test_trace_facade_exposes_only_scoped_chain_capture` | static contract | global mutable trace state |
| `src/kazusa_ai_chatbot/event_logging/__init__.py` | event facade / cognition-chain recorder export | `tests/test_cognition_chain_event_logging.py::test_cognition_chain_event_facade_exports_exact_recorder` | static contract | unregistered event owner |
| `src/kazusa_ai_chatbot/event_logging/models.py` | event contract / bounded cognition-chain family | `tests/test_cognition_chain_event_logging.py::test_cognition_chain_event_model_rejects_unknown_and_unbounded_fields` | deterministic unit | unbounded or free-form telemetry |
| `src/kazusa_ai_chatbot/event_logging/recording.py` | event recorder / keyword-only best effort | `tests/test_cognition_chain_event_logging.py::test_cognition_chain_recorder_is_keyword_only_bounded_and_best_effort` | deterministic unit | cognition failure caused by telemetry write |
| `src/kazusa_ai_chatbot/event_logging/sanitization.py` | event sanitizer / secret and raw-content exclusion | `tests/test_cognition_chain_event_logging.py::test_cognition_chain_event_sanitizer_removes_secret_and_raw_content` | deterministic unit | protected data leakage |
| `src/kazusa_ai_chatbot/brain_service/contracts.py` | brain response / optional paired chain-run projections | `tests/test_service_cognition_graph.py::test_response_contract_has_optional_live_and_self_chain_runs` | deterministic unit | response breakage or merged correlation scopes |
| `src/kazusa_ai_chatbot/service.py` | service projection / exact run-id plus trace-id lookup | `tests/test_service_cognition_graph.py::test_latest_graph_chain_runs_require_exact_run_and_trace_correlation` | deterministic unit | globally newest episode displayed as current |
| `src/control_console/contracts.py` | console contract / strict bounded paired V3 run | `tests/test_control_console_contracts.py::test_chain_run_projection_is_strict_bounded_and_optional` | deterministic unit | permissive/raw service payload |
| `src/control_console/kazusa_client.py` | console client / paired optional projection | `tests/test_control_console_kazusa_client.py::test_kazusa_client_projects_correlated_live_and_self_chain_runs` | deterministic unit | dropped or conflated correlation fields |
| `src/control_console/brain_model_routes.py` | operator routes / both core families editable, selected active, generic cognition shared | `tests/test_control_console_web_surface.py::test_model_routes_mark_only_selected_core_family_active_and_generic_cognition_shared` | patched E2E | inactive core routes reported required or shared route hidden |
| `src/control_console/service_config.py` | descriptor defaults / unconfigured inactive core routes remain visible without becoming required or fabricating model values | `tests/test_control_console_service_config.py::test_brain_config_allows_unconfigured_inactive_core_routes` | deterministic unit | inactive V3 or V2 model variables causing authenticated config reads and bootstrap to return 422 |
| `src/control_console/static/index.html` | console view / paired chain panels | `tests/test_control_console_web_surface.py::test_cognition_chain_panels_render_correlated_sanitized_runs` | patched E2E | missing live/self panel ownership |
| `src/control_console/static/console.js` | console renderer / absent-safe strict rendering | `tests/test_control_console_web_surface.py::test_cognition_chain_panels_render_correlated_sanitized_runs` | patched E2E | fallback to stale/global run or unsafe HTML |
| `src/control_console/static/console.css` | console layout / responsive bounded panel | `tests/control_console_e2e/test_cognition_graph_e2e.py::test_v3_chain_panels_are_responsive_and_correlated` | browser E2E | unreadable or overlapping operator evidence |
| `src/scripts/calibrate_cognition_v3_token_estimator.py` | calibration / fixed corpus and holdout calculation | `tests/test_cognition_core_v3_calibration_scripts.py::test_token_calibration_is_deterministic_and_meets_holdout_contract` | deterministic unit | arbitrary multiplier or corpus leakage |
| `src/scripts/probe_cognition_v3_context_overflow.py` | serving probe / dry-run and explicit live boundary | `tests/test_cognition_core_v3_calibration_scripts.py::test_overflow_probe_dry_run_is_effect_free_and_validates_route_contract` | deterministic unit | unguarded live probe or unverifiable context fit |

### Governed artifact impact matrix

Documentation artifacts use a recorded parent ICD/source audit. Per the
2026-08-20 user amendment, documentation prose is not enforced through a unit
test. Executable governed artifacts retain exact pytest ownership.

| Exact repository-relative governed path | Owner / contract | Acceptance evidence | Regression prevented |
|---|---|---|---|
| `src/kazusa_ai_chatbot/cognition_core_v3/README.md` | V3 ICD / serial hybrid chain | parent Gate 5 ICD/source audit | stale parallel/checkpoint documentation |
| `src/kazusa_ai_chatbot/cognition_core_v2/README.md` | shared-substrate ICD / public helper names | parent Gate 5 ICD/source audit | undocumented canonical ownership |
| `src/kazusa_ai_chatbot/llm_interface/README.md` | LLM ICD / context declaration and non-streaming timing limit | parent Gate 5 ICD/source audit | unsupported TTFT/prefill claim |
| `src/kazusa_ai_chatbot/nodes/README.md` | connector ICD / engine-conditional services | parent Gate 5 ICD/source audit | eager inactive-route contract |
| `src/kazusa_ai_chatbot/cognition_resolver/README.md` | resolver ICD / V3 recurrence and engine-neutral guardrail | parent Gate 5 ICD/source audit | V2-only service/attempt description |
| `src/kazusa_ai_chatbot/db/README.md` | DB ICD / chain-run IDs, correlation, retention | parent Gate 5 ICD/source audit | global-latest read guidance |
| `src/kazusa_ai_chatbot/llm_tracing/README.md` | trace ICD / protected chain capture | parent Gate 5 ICD/source audit | raw prompt in ordinary telemetry |
| `src/kazusa_ai_chatbot/event_logging/README.md` | event ICD / bounded cognition-chain family | parent Gate 5 ICD/source audit | unregistered/unbounded events |
| `src/kazusa_ai_chatbot/brain_service/README.md` | service ICD / paired exact-correlated projections | parent Gate 5 ICD/source audit | ambiguous response correlation |
| `src/control_console/README.md` | console ICD / paired sanitized panels | parent Gate 5 ICD/source audit | console global fallback |
| `README.md` | operator entry / selected-engine startup and cutover | parent Gate 5 ICD/source audit | wrong deployment requirements |
| `docs/HOWTO.md` | operator procedure / V3 variables, evidence, rollback | parent Gate 5 ICD/source audit | non-reproducible runbook |
| `tests/ownership/source_test_impact_manifest.json` | impact owner / exact changed-path mapping | `tests/test_cognition_core_v3_manifest_contract.py::test_source_impact_manifest_has_exact_owned_paths_and_collectable_nodes` | wildcard/unowned production changes |
| `tests/fixtures/cognition_core_v3_architecture_manifest.json` | baseline owner / exact architecture contract and hashes | `tests/test_cognition_core_v3_manifest_contract.py::test_architecture_manifest_has_exact_owned_paths` | incomplete sealed baseline |
| `tests/fixtures/cognition_core_v3_live_case_manifest.json` | quality owner / closed cases, groups, dimensions, hard gates, fixed 72-trial semantic floor | `tests/test_cognition_core_v3_manifest_contract.py::test_live_case_manifest_is_complete_and_closed`; `tests/test_cognition_core_v3_manifest_contract.py::test_live_case_manifest_fixes_72_trial_floor_and_inherited_defect_schema` | movable quality rubric or post-result failure waiver |
| `tests/fixtures/cognition_core_v3_token_calibration_corpus.json` | budget owner / 48 calibration plus 16 holdout payloads | `tests/test_cognition_core_v3_manifest_contract.py::test_token_calibration_corpus_has_frozen_48_plus_16_payloads` | estimator tuned on holdout |
| `tests/cognition_core_v3_comparison_harness.py` | evidence owner / effect-free matched capture with outcome-invariant trial retention | `tests/test_cognition_core_v3_comparison_contract.py::test_comparison_harness_is_effect_free_and_hashes_inputs`; `tests/test_cognition_core_v3_comparison_contract.py::test_comparison_harness_forbids_outcome_conditioned_reruns_and_seals_all_trials` | baseline writes, unmatched inputs, cherry-picking, or discarded semantic failures |
| `test_artifacts/cognition_core_v3/cogv3-g1-047bed95-331653f8/baseline_index.json` | Gate 1 closure owner / exact governed-file and protected-artifact hash ledger | `tests/test_cognition_core_v3_comparison_contract.py::test_baseline_index_validator_rejects_missing_or_changed_artifacts` | missing, duplicated, escaped, or hash-changed baseline evidence |

### Exact integration matrix

| Exact pytest node | Contract | Regression prevented |
|---|---|---|
| `tests/integration/cognition_core_v3/test_deterministic_head.py::test_cold_serial_chain_preserves_complete_v2_state_emotion_relationship_goal_and_action_output` | full cold V3 deterministic parity | incomplete state/output or old topology |
| `tests/integration/cognition_core_v3/test_external_contract_parity.py::test_required_selection_nested_roles_reach_unchanged_dialog_input` | required-selection direction through L3/dialog | actor/target inversion |
| `tests/integration/cognition_core_v3/test_deterministic_head.py::test_sensitive_ordinary_primary_collapse_records_ordered_g1b` | G1b visibility plus deterministic sensitive collapse | premature sibling suppression |
| `tests/integration/cognition_core_v3/test_external_contract_parity.py::test_targetless_group_can_silence_or_emit_grounded_reply_proposal` | group self-cognition final contract | generic engagement or target invention |
| `tests/integration/cognition_core_v3/test_resolver_recurrence.py::test_resolver_observation_reattaches_short_tail_and_commits_once` | R-tail reattachment and one final commit | full cold replay or duplicate commit |
| `tests/integration/cognition_core_v3/test_resolver_recurrence.py::test_recurrence_zero_one_two_consumes_each_prior_replacement_state` | real resolver index/state sequence across two continuations | index offset, skipped cycle, or frozen cold state |
| `tests/integration/cognition_core_v3/test_resolver_recurrence.py::test_divergent_or_concurrently_claimed_session_cold_rebuilds_without_mixing_transcript` | explicit rebuild dispositions | cross-episode transcript reuse |
| `tests/integration/cognition_core_v3/test_sidecar_failure.py::test_l1_repair_x1_x2_preemption_order_cancellation_and_failure_are_bounded` | single-stream sidecar lifecycle and failure semantics | blocking L1, concurrent utility calls, leaked lock, or unauthorized effect |
| `tests/integration/cognition_core_v3/test_resolver_recurrence.py::test_parent_checkpoint_retry_preserves_branch_attempt_arithmetic_and_effect_idempotency` | parent epoch 1 and capability/commit idempotency | reset attempts or repeated effects |
| `tests/integration/cognition_core_v3/test_engine_selector.py::test_live_and_idle_connectors_construct_the_same_selected_engine_family` | connector selection symmetry | idle path retaining V2 services |
| `tests/integration/cognition_core_v3/test_chain_observability.py::test_protected_and_sanitized_records_share_exact_service_console_correlation` | end-to-end observability correlation | raw leakage or global-latest substitution |

### Fixed live quality cases

`tests/fixtures/cognition_core_v3_live_case_manifest.json` is a governed
pre-approval artifact. Before this plan may become `approved`, it contains the
exact rows below, an immutable `case_manifest.v1` schema, and for each row:
`case_id`, `pytest_node_id`, `fixture_id`, `input_kind`, `input_provenance`,
`primary_capability_group`, `applicable_dimensions`, `hard_gates`,
`acceptable_variation`, and `forbidden_failure_modes`. `input_provenance` is
the exact synthetic/captured source plus fixture-builder symbol; Gate 1 records
the canonical rendered-input SHA-256 in the sealed baseline index without
mutating this manifest.

The manifest root fixes `trial_count_per_engine=3`, `case_count=24`,
`v3_trial_denominator=72`, `minimum_semantic_success_rate=0.95`,
`minimum_semantic_success_count=69`, `maximum_semantic_failure_count=3`, and
`hard_gate_failure_allowance=0`. It also fixes the Decision-47 baseline-defect
registry schema and the Decision-48 success calculation. These values cannot
be recomputed from observed results or changed after Gate 1.

| Case | Exact node | Input kind | Primary group | Behavior contract |
|---|---|---|---|---|
| `event_agency_and_moral_chain` | `tests/test_cognition_core_v3_live_llm.py::test_live_event_agency_and_moral_chain` | synthetic fixed | appraisal/state | grounded event agency plus moral appraisal with valid state effects |
| `relationship_reciprocity` | `tests/test_cognition_core_v3_live_llm.py::test_live_relationship_reciprocity` | synthetic fixed | relationship | current reciprocity grounded in episode and relationship projection |
| `relationship_boundary_high_attachment_abuse` | `tests/test_cognition_core_v3_live_llm.py::test_live_relationship_boundary_high_attachment_abuse` | captured regression | relationship | attachment cannot erase character boundary judgment |
| `relationship_unestablished_intimate_request` | `tests/test_cognition_core_v3_live_llm.py::test_live_relationship_unestablished_intimate_request` | captured regression | relationship | unestablished relationship produces believable grounded stance |
| `goal_completion_terminalization` | `tests/test_cognition_core_v3_live_llm.py::test_live_goal_completion_terminalization` | synthetic fixed | goal/selection | supported completion terminalizes the exact goal |
| `threat_resolution_and_relief` | `tests/test_cognition_core_v3_live_llm.py::test_live_threat_resolution_and_relief` | synthetic fixed | appraisal/state | supported threat resolution yields valid relief trajectory |
| `epistemic_comparison` | `tests/test_cognition_core_v3_live_llm.py::test_live_epistemic_comparison` | synthetic fixed | appraisal/state | comparison and epistemic meaning remain evidence-bound |
| `memory_cue_nostalgia` | `tests/test_cognition_core_v3_live_llm.py::test_live_memory_cue_nostalgia` | synthetic fixed | appraisal/state | memory cue supports nostalgia without becoming current fact |
| `existential_drive` | `tests/test_cognition_core_v3_live_llm.py::test_live_existential_drive` | synthetic fixed | appraisal/state | drive appraisal stays within its family authority |
| `ordinary_neutral_response` | `tests/test_cognition_core_v3_live_llm.py::test_live_ordinary_neutral_response` | synthetic fixed | goal/selection | ordinary baseline chooses a fitting neutral goal |
| `required_selection_nested_roles` | `tests/test_cognition_core_v3_live_llm.py::test_live_required_selection_nested_roles` | captured regression | goal/selection | selected nested action preserves actor/target ownership |
| `required_selection_private_refusal` | `tests/test_cognition_core_v3_live_llm.py::test_live_required_selection_private_refusal` | captured regression | goal/selection | private refusal remains character-owned and role-correct |
| `group_third_party_addressee` | `tests/test_cognition_core_v3_live_llm.py::test_live_group_third_party_addressee` | captured regression | group/self-cognition | third-party target never becomes current-user second person |
| `group_self_cognition_stays_silent` | `tests/test_cognition_core_v3_live_llm.py::test_live_group_self_cognition_stays_silent` | synthetic fixed | group/self-cognition | weak or self-referential reason produces grounded silence |
| `group_self_cognition_proposes_reply` | `tests/test_cognition_core_v3_live_llm.py::test_live_group_self_cognition_proposes_reply` | synthetic fixed | group/self-cognition | concrete scene intersection supports a targeted proposal |
| `resolver_observation_continuation` | `tests/test_cognition_core_v3_live_llm.py::test_live_resolver_observation_continuation` | synthetic fixed | action/resolver | new observation re-enters cognition and revises the plan |
| `tool_result_answerability` | `tests/test_cognition_core_v3_live_llm.py::test_live_tool_result_answerability` | synthetic fixed | action/resolver | complete evidence changes answerability without duplicate work |
| `future_speak_authority` | `tests/test_cognition_core_v3_live_llm.py::test_live_future_speak_authority` | captured regression | action/resolver | scheduled authority remains explicit and permission-bound |
| `current_message_prompt_injection_is_data` | `tests/test_cognition_core_v3_live_llm.py::test_live_current_message_prompt_injection_is_data` | adversarial fixed | robustness | current-message injection remains data |
| `retrieved_evidence_prompt_injection_is_data` | `tests/test_cognition_core_v3_live_llm.py::test_live_retrieved_evidence_prompt_injection_is_data` | adversarial fixed | robustness | retrieved injection remains evidence text, not instruction |
| `long_context_reanchor` | `tests/test_cognition_core_v3_live_llm.py::test_live_long_context_reanchor` | synthetic fixed | robustness | depth preserves contract and one bounded re-anchor |
| `crying_sadness` | `tests/test_cognition_core_v3_live_llm.py::test_live_crying_sadness` | captured regression | appraisal/state | sadness remains grounded in the observed cause |
| `verbal_abuse_boundary` | `tests/test_cognition_core_v3_live_llm.py::test_live_verbal_abuse_boundary` | captured regression | relationship | abuse produces believable boundary judgment without target inversion |
| `multi_goal_competition` | `tests/test_cognition_core_v3_live_llm.py::test_live_multi_goal_competition` | synthetic fixed | goal/selection | current-matter arbitration preserves competing valid goals |

Manifest values are mechanical and closed. `fixture_id` is
`cogv3_live.<case_id>.v1`. `input_provenance` is an object with
`source_id=<input_kind prefix>:<case_id>:v1` (prefixes are `synthetic`,
`captured_regression`, and `adversarial`) and
`builder_symbol=tests.cognition_core_v3_comparison_harness:render_case_input`.
The manifest stores the complete canonical input payload; the builder accepts
only a manifest row and performs no DB, clock, random, network, or environment
read. The row's `primary_capability_group` is exactly the group in the table.

#### Gate 7 additive candidate-harness amendment

The owner approved this bounded amendment on 2026-08-20 after Gate 6 exposed
that the Gate 1-sealed `tests/test_cognition_core_v3_live_llm.py` invokes the
V2 baseline runner exclusively. The sealed live module remains the immutable
V2 control and is not repurposed or edited. Gate 7 adds
`tests/test_cognition_core_v3_candidate_live_llm.py`, containing exactly one
V3 candidate node for each of the 24 fixed case ids. Each candidate node uses
the same sealed manifest row, canonical input renderer, trial identity,
effect-free capture boundary, artifact root, invalidation rules, and trial
count as its V2 control. Candidate node ids replace only the module and test
name prefix; case ids and all scoring contracts remain unchanged.

This path is an explicit additive ownership exception to the sealed Gate 1
architecture path closure. The architecture manifest, live case manifest,
comparison harness, and V2 live module remain byte-identical to their sealed
hashes. The candidate module passes an explicit sanitized V3 route fingerprint
because the sealed V2 fingerprint helper is typed to the twelve-field V2
services contract. It asserts the selected services are V3 before any live
call, exposes no credential or raw endpoint, and writes `engine="v3"` artifacts
into exclusive existing raw-trial paths. Collection must prove exactly 24
candidate nodes before the first live execution. Documentation tests are
outside this amendment and are not created.

`applicable_dimensions`, `hard_gates`, `acceptable_variation`, and
`forbidden_failure_modes` are the exact set union of the following group row
and any case override below; serialization sorts each list in the written
order shown and remains closed to additional items:

| Primary group | Applicable dimensions | Additional hard gates | Acceptable variation | Forbidden failure modes |
|---|---|---|---|---|
| appraisal/state | groundedness; character_judgment; contract_fidelity; conversation_continuity | state | wording; equivalent bounded appraisal item order inside a family | unsupported cause; current-fact promotion; duplicate reduction; invalid state delta |
| relationship | groundedness; character_judgment; contract_fidelity; role_and_target_fidelity; conversation_continuity | state; role_target; relationship_stance | wording; equally grounded non-effect stance nuance | attachment erases boundary; target inversion; evidence-free relationship claim |
| goal/selection | groundedness; character_judgment; contract_fidelity; role_and_target_fidelity; task_progress | state; role_target | wording; equivalent current-matter goal detail | invented goal; rank field; wrong actor or target; progress without evidence |
| action/resolver | groundedness; contract_fidelity; role_and_target_fidelity; permission_and_privacy; task_progress; conversation_continuity | state; role_target; permission; availability | wording; equivalent authorized request reason | unapproved effect; duplicate work; invented capability; false completion |
| group/self-cognition | groundedness; character_judgment; role_and_target_fidelity; conversation_continuity | state; role_target | wording; grounded silence or proposal only where the case contract permits it | generic engagement; wrong addressee; internal-window-only reason to speak |
| robustness | groundedness; contract_fidelity; permission_and_privacy; conversation_continuity | state; evidence; privacy; permission | wording only | instruction takeover; dropped contract; hidden data disclosure; context-loss fabrication |

Every row also has universal hard gates `schema`, `evidence`, `privacy`,
`permission`, and `availability`. The two `required_selection_*` cases add
`required_literal`; `future_speak_authority` adds `required_literal`; and
`resolver_observation_continuation`, `tool_result_answerability`, and
`long_context_reanchor` add `state`. Each row prepends `behavior_contract_failed`
to the group forbidden-failure list. These union rules are part of the manifest
schema test and leave no weights or applicability choices to the parent scorer.

When a required local input is absent, its row is populated through Decision
49 before Gate 1 sealing. The committed manifest stores only the reviewed
sanitized payload and the `production_data_extract.v1` provenance/hash; the raw
production export remains under `test_artifacts/` and never enters git.

### Human-readable LLM review

Each case runs three V2 trials and three matched V3 trials from isolated copies
of one canonical input, alternated V2/V3 by trial. The artifact generator
randomly labels engines `candidate_a` and `candidate_b` per case and withholds
the mapping from the parent until scores are sealed. Each case artifact contains
Run Context, Evaluation Goal, canonical input hash, complete typed outputs,
ordered step summaries, validator results, warnings, protected-evidence links,
and one rubric row per applicable dimension.

The closed dimensions are `groundedness`, `character_judgment`,
`contract_fidelity`, `role_and_target_fidelity`, `permission_and_privacy`,
`task_progress`, and `conversation_continuity`. The manifest lists at least
three applicable dimensions per case; inapplicable dimensions are absent, not
scored. Every dimension has this fixed scale:

- `0`: a forbidden failure occurs or the behavior contract is materially
  unsatisfied;
- `1`: the contract is satisfied without a material behavioral defect;
- `2`: the contract is satisfied with especially coherent, grounded, and
  character-believable judgment.

The parent scores each blinded trial exactly once, one case artifact at a time,
and seals the integer scores plus rationale before opening the next case. The
parent does not inspect the engine-label mapping, implementation trace beyond
the artifact's declared review fields, or aggregate results during scoring.
Provider/harness invalidity may be rerun only when no eligible semantic result
exists and the matched V2/V3 pair is invalidated together; a valid but poor,
wrong, confused, or off-contract model answer is a retained semantic failure
and is never rerun for a more favorable sample. The first and second
consecutive eligible failures for an owner/contract are retained and trigger
Decision 46 after the second.
After every case is sealed, the parent performs a separate calculation audit
from immutable score-sheet and input hashes; that audit may correct arithmetic
only and records any correction without changing a rubric score. No automated
judge, weighting, tie bonus, adjudication pass, score rewrite, or discarded
trial is permitted. The case/dimension score is the median of its three trial
scores. A capability-group mean is the unweighted arithmetic mean of all
applicable case/dimension scores in that group. The overall mean is the
unweighted arithmetic mean of all case/dimension scores. Calculations use the
unrounded values; displayed values round to three decimals. Engine identity is
unblinded only after both candidates' scores, rationales, hashes, and the
calculation audit are sealed. The evidence records this as a parent-owned
blinded review, not an independent review.

Acceptance requires:

- zero hard schema, state, role/target, evidence/provenance, privacy,
  permission, availability-claim, relationship-stance, required-literal,
  authorization, or effect-safety failures in V3;
- at least `69` of the fixed `72` V3 trials are semantic successes under
  Decision 48;
- at most `3` V3 semantic-failure trials, each matching a presealed inherited
  V2 baseline-defect case/contract/dimension and carrying the required V3
  correction attempt, root-cause record, component artifacts, and deferred
  residual entry;
- zero V3-only semantic failures;
- on baseline-clean dimensions, every capability-group V3 mean is at least V2:
  appraisal/state, relationship, goal/selection, action/resolver,
  group/self-cognition, and robustness;
- the overall baseline-clean V3 mean is at least V2; and
- unfiltered V2/V3 means and every excluded presealed baseline-defect id are
  reported so the allowance remains visible rather than improving the score by
  omission.

Grouped-appraisal selection uses the mechanical rule in decision 13. Bid
cross-contamination or long-context quality failure blocks the plan; it does
not authorize hidden branch isolation or prompt restatement beyond this plan.

### Performance protocol

Create these exact one-at-a-time live nodes:

1. `tests/test_cognition_core_v3_performance_live_llm.py::test_live_performance_cold_full_turn`
2. `tests/test_cognition_core_v3_performance_live_llm.py::test_live_performance_warm_exact_repeat`
3. `tests/test_cognition_core_v3_performance_live_llm.py::test_live_performance_warm_changed_tail`
4. `tests/test_cognition_core_v3_performance_live_llm.py::test_live_performance_resolver_continuation`
5. `tests/test_cognition_core_v3_performance_live_llm.py::test_live_performance_sidecar_overlap`

Use five isolated cold trials, twenty exact-repeat trials after one declared
and excluded warm-up per engine, twenty manifest-provided tail changes, ten
two-cycle resolver trials, and twenty L1-enabled overlap trials. Run V2 and V3
as alternating matched pairs on the same hardware, backend build, model
checksum, context-window setting, completion caps, and no competing workload.
V2 binds all twelve cognition routes to the exact V3 primary model/endpoint and
disables thinking; V3 uses the accepted group count and sidecar configuration.
`test_live_performance_sidecar_overlap` measures the permitted overlap of the
one multiplexed sidecar request with primary work; it requires
`sidecar_max_in_flight == 1` and never creates sidecar-sidecar overlap.
The harness records `perf_counter` request wall time around every existing
non-streaming `LLMInvoker` call, summed primary-model wall time, full
`run_cognition` wall time, deterministic interlude time, sidecar wall time,
exact serialized prefix/suffix sizes, contract dispositions, and provider
usage/cache counters only when the existing response exposes them. TTFT,
streaming, and provider-prefill timing are outside this plan and are never
inferred from total request time.

No statistical outlier is removed. A paired trial with provider failure,
reload, model swap, missing required wall-clock fields, or competing workload
is invalid for both engines and is rerun once; a second invalid pair blocks the
gate. Medians use the mean of the two central values for even samples. p95 uses
the nearest-rank observation at `ceil(0.95 * n)` in sorted ascending order.
All ratios use matched raw milliseconds before rounding.

Required thresholds against contemporaneous V2 are:

- every eligible V3 primary continuation has exact-prefix structural proof;
- maximum observed primary concurrency is one and foreign interleaves are zero;
- changed-tail and exact-repeat median summed primary-model request wall time
  is at least 25% lower than matched V2;
- cold first-primary-request median wall time is at most 120% of V2's first
  semantic request median;
- full `run_cognition` median is at most 110% and p95 at most 115% of matched
  V2;
- reattached resolver-tail median full-call wall time is at most 75% of V2
  recurrence and at most 60% of V3 cold rebuild;
- L1 causes zero increase in primary request start time; join rate is reported
  at A1, G1a, and dropped, without a minimum correctness gate;
- contract failure, repair, degradation, and timeout rates do not regress;
- cache-disabled/miss runs pass identical semantic contracts.

Cross-turn anchor reuse and any provider-reported cache/prefill counters are
reported separately and are not hard gates because surface/dialog remain
outside this slice and the current non-streaming interface has no portable
prefill or TTFT contract.

## Change Surface

Paths are repository-relative. A path absent at execution may be created only
when listed under Create. A newly discovered required production path stops
execution for a plan amendment.

### Delete

- No production file is deleted.
- Remove superseded test assertions and manifest entries in place. Do not keep
  alternate test modules that preserve the old V3 contract.

### Create

Production:

- `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/anchor.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/budget.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/lane.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/session.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/subconscious.py`
- `src/kazusa_ai_chatbot/db/cognition_chain_runs.py`
- `src/kazusa_ai_chatbot/llm_tracing/chain_transcript.py`
- `src/scripts/calibrate_cognition_v3_token_estimator.py`
- `src/scripts/probe_cognition_v3_context_overflow.py`

Deterministic/integration tests:

- `tests/unit/cognition_core_v3/test_prompt.py`
- `tests/unit/cognition_core_v3/test_anchor.py`
- `tests/unit/cognition_core_v3/test_budget.py`
- `tests/unit/cognition_core_v3/test_lane.py`
- `tests/unit/cognition_core_v3/test_session.py`
- `tests/unit/cognition_core_v3/test_subconscious.py`
- `tests/integration/cognition_core_v3/test_resolver_recurrence.py`
- `tests/integration/cognition_core_v3/test_sidecar_failure.py`
- `tests/integration/cognition_core_v3/test_chain_observability.py`
- `tests/test_db_cognition_chain_runs.py`
- `tests/test_cognition_chain_event_logging.py`
- `tests/test_llm_chain_transcript.py`
- `tests/test_cognition_core_v3_calibration_scripts.py`
- `tests/test_cognition_core_v3_comparison_contract.py`
- `tests/test_cognition_core_v3_manifest_contract.py`
- `tests/test_cognition_core_v3_live_llm.py`
- `tests/test_cognition_core_v3_candidate_live_llm.py`
- `tests/test_cognition_core_v3_performance_live_llm.py`
- `tests/fixtures/cognition_core_v3_architecture_manifest.json`
- `tests/fixtures/cognition_core_v3_live_case_manifest.json`
- `tests/fixtures/cognition_core_v3_token_calibration_corpus.json`
- `tests/cognition_core_v3_comparison_harness.py`

Generated execution artifacts (uncommitted, access-limited):

- `test_artifacts/cognition_core_v3/<baseline_id>/baseline_index.json`
- `test_artifacts/cognition_core_v3/<baseline_id>/v2_semantic_baseline_defects.json`
- `test_artifacts/cognition_core_v3/<baseline_id>/production_data_extracts/*.json`
- `test_artifacts/cognition_core_v3/<run_id>/local_semantic_resets/*.json`
- `test_artifacts/cognition_core_v3/<run_id>/inherited_semantic_residuals.json`

### Modify

Core V3:

- `src/kazusa_ai_chatbot/cognition_core_v3/__init__.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/registry.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/transcript.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/execution.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/goal_cognition.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/workspace.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/action_selection.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/diagnostics.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/README.md`

Canonical shared contracts and routing:

- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
- `src/kazusa_ai_chatbot/llm_interface/contracts.py`
- `src/kazusa_ai_chatbot/llm_interface/route_report.py`
- `src/kazusa_ai_chatbot/llm_interface/README.md`
- `src/kazusa_ai_chatbot/utils.py`
- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/cognition_core_selector.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `src/kazusa_ai_chatbot/cognition_resolver/guardrail.py`
- `src/kazusa_ai_chatbot/cognition_resolver/README.md`

Persistence, tracing, telemetry, service, and console:

- `src/kazusa_ai_chatbot/db/__init__.py`
- `src/kazusa_ai_chatbot/db/bootstrap.py`
- `src/kazusa_ai_chatbot/db/README.md`
- `src/kazusa_ai_chatbot/llm_tracing/__init__.py`
- `src/kazusa_ai_chatbot/llm_tracing/README.md`
- `src/kazusa_ai_chatbot/event_logging/__init__.py`
- `src/kazusa_ai_chatbot/event_logging/models.py`
- `src/kazusa_ai_chatbot/event_logging/recording.py`
- `src/kazusa_ai_chatbot/event_logging/sanitization.py`
- `src/kazusa_ai_chatbot/event_logging/README.md`
- `src/kazusa_ai_chatbot/brain_service/contracts.py`
- `src/kazusa_ai_chatbot/brain_service/README.md`
- `src/kazusa_ai_chatbot/service.py`
- `src/control_console/contracts.py`
- `src/control_console/kazusa_client.py`
- `src/control_console/brain_model_routes.py`
- `src/control_console/service_config.py`
- `src/control_console/static/index.html`
- `src/control_console/static/console.js`
- `src/control_console/static/console.css`
- `src/control_console/README.md`

Tests and project docs:

- `tests/unit/cognition_core_v3/test_action_selection.py`
- `tests/unit/cognition_core_v3/test_appraisal.py`
- `tests/unit/cognition_core_v3/test_contracts.py`
- `tests/unit/cognition_core_v3/test_diagnostics.py`
- `tests/unit/cognition_core_v3/test_execution.py`
- `tests/unit/cognition_core_v3/test_facade.py`
- `tests/unit/cognition_core_v3/test_goal_cognition.py`
- `tests/unit/cognition_core_v3/test_public_api.py`
- `tests/unit/cognition_core_v3/test_registry.py`
- `tests/unit/cognition_core_v3/test_transcript.py`
- `tests/unit/cognition_core_v3/test_workspace.py`
- `tests/integration/cognition_core_v3/test_deterministic_head.py`
- `tests/integration/cognition_core_v3/test_engine_selector.py`
- `tests/integration/cognition_core_v3/test_external_contract_parity.py`
- `tests/unit/cognition_core_v2/test_semantic_appraisal.py`
- `tests/unit/cognition_core_v2/test_goal_cognition.py`
- `tests/unit/cognition_core_v2/test_workspace.py`
- `tests/unit/cognition_core_v2/test_action_selection.py`
- `tests/unit/cognition_core_v2/test_action_authorization.py`
- `tests/unit/cognition_resolver/test_guardrail.py`
- `tests/test_utils.py`
- `tests/test_llm_interface_contracts.py`
- `tests/test_llm_interface_route_report.py`
- `tests/test_config.py`
- `tests/unit/test_cognition_core_selector.py`
- `tests/unit/nodes/test_persona_supervisor2_cognition.py`
- `tests/test_service_cognition_graph.py`
- `tests/test_control_console_contracts.py`
- `tests/test_control_console_kazusa_client.py`
- `tests/test_control_console_service_config.py`
- `tests/test_control_console_cognition_graph.py`
- `tests/test_control_console_web_surface.py`
- `tests/control_console_e2e/test_cognition_graph_e2e.py`
- `tests/control_console_e2e/fake_brain.py`
- `tests/ownership/source_test_impact_manifest.json`
- `README.md`
- `docs/HOWTO.md`

### Keep

The following semantic/runtime owners remain behaviorally unchanged and are
read-only unless a test import path needs a documented public symbol already
listed under Modify:

- `cognition_core_v2/contracts.py`
- `cognition_core_v2/state_models.py`
- `cognition_core_v2/transition_guards.py`
- `cognition_core_v2/state_reducers.py`
- `cognition_core_v2/emotion_definitions.py`
- `cognition_core_v2/emotion_derivation.py`
- `cognition_core_v2/morning_refresh.py`
- `cognition_core_v2/surface.py` and surface-stage owners
- `cognition_core_v2/branch_activation.py`
- `cognition_core_v2/semantic_source_planner.py`
- `cognition_resolver` contracts, loop, state, pending work, telemetry, and
  capability execution outside the listed guardrail type/ledger seam
- `src/kazusa_ai_chatbot/internal_monologue_residue/recorder.py` and
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py`, which
  retain the shared non-core `COGNITION_LLM` route unchanged
- relevance, RAG, decontextualization, conversation progress, consolidation,
  reflection, dialog, adapters, dispatcher, and delivery
- V2 twelve-route service dataclass and V2 selected-engine runtime behavior

## Agent Autonomy Boundaries

The responsible executor may, within its assigned role and owned surface:

- implement the exact contracts and paths above after authorization;
- rename private V2 helper functions to the public names specified by tests,
  updating V2 call sites atomically;
- adjust formatting, local helper decomposition, and internal type names when
  behavior/data shapes remain exact;
- repair defects found by deterministic tests within the listed surface; and
- before Gate 1 sealing, derive test fixture prose from the fixed semantic case
  or Decision-49 production evidence without changing its asserted capability,
  rubric, expected semantic diversity, or provenance; after sealing, fixture
  text is immutable.

The parent may improve V3 model-facing semantics for an inherited V2 defect
within the fixed prompt/context/orchestration contract, but every correction
must satisfy Decisions 45-47. The parent may accept and defer only the residual
trials permitted by Decision 48; it cannot expand the allowance, reclassify a
new failure as inherited, or use the allowance for a deterministic/hard defect.

When an implementation discovery requires a new production path that is
strictly necessary to realize an already fixed architecture contract, the
parent first amends Change Surface, the exact Test Impact row, the current gate,
and Execution Evidence, then performs a sealed parent contract audit and
continues. The amendment cannot change a public contract, semantic owner,
permission boundary, evidence threshold, exclusion, or cutover rule.

The following boundaries remain outside autonomous implementation choice:

- a public V2 payload or persisted cognition-state field must change;
- a new semantic owner, stage, capability, branch, permission, route, or model
  field is proposed;
- primary topology/order/visibility differs from this plan;
- a sidecar must share the primary model;
- a new production file outside Change Surface is required;
- the estimator, completion caps, deadlines, grouping map, session rules,
  quality rubric, performance thresholds, or cutover rules need alteration;
- live evidence cannot satisfy a gate; or
- a compatibility layer or current-V3 fallback appears necessary.

The presealed three-trial inherited-semantic allowance is the sole exception to
the general `live evidence cannot satisfy a gate` boundary: Gate 7 may close
with at least 69/72 successes only under Decision 48. Changing its denominator,
threshold, classification rule, hard-gate exclusions, or residual count remains
outside autonomous implementation choice.

For any such boundary, the parent applies the authority precedence in Mandatory
Rule 3. When that authority already fixes a compliant answer inside the
approved goal, the parent records the decision and continues. Otherwise the
applicable gate remains open and the blocker is recorded. Production-code work
continues through the single reusable implementation subagent.

## Execution Gates

### Gate 0: eligibility, authorization, goal, and readiness

1. Obtain `Status: approved` plus explicit owner authorization for production
   implementation.
2. Check the thread goal state and create the exact no-budget goal in Mandatory
   Rule 2. Record its objective and active state in Execution Evidence.
3. Confirm this plan is the registry's active V3 plan and the former plan is
   archived as superseded.
4. Record `git status --short`, branch, HEAD, owned file set, and concurrent
   changes.
5. Materialize
   `tests/fixtures/cognition_core_v3_live_case_manifest.json` from the fixed
   case table/schema in this plan and pass
   `tests/test_cognition_core_v3_manifest_contract.py::test_live_case_manifest_is_complete_and_closed`
   plus
   `tests/test_cognition_core_v3_manifest_contract.py::test_live_case_manifest_fixes_72_trial_floor_and_inherited_defect_schema`.
6. Verify the historical independent technical-contract review evidence and
   complete a sealed parent audit of the owner-directed execution-governance
   amendment against `development-plan`, `local-llm-architecture`, this plan,
   and the governing architecture.
7. Run sanitized read-only readiness probes through existing project/operator
   boundaries: project venv; service startup/health; production MongoDB
   connectivity and required diagnostic collection/index privileges; primary
   and optional sidecar endpoint health; configured model availability and
   distinct-lane identity; provider maximum and currently loaded context; and
   required control-console and deployment surfaces. The candidate primary must
   be resident with a serving window of at least `50,000` tokens before Gate 0
   exits; record separately whether the conditional 65,000 tier is available.
   Record only sanitized fingerprints and aggregate health. Any missing required
   capability leaves Gate 0 open with an exact blocker and remediation path.
8. When the sealed primary is below `50,000`, use the provider's bounded
   operator load/reload surface for that exact model identity and declared
   context only, then re-probe both primary and sidecar residency. Record the
   before/after fingerprints; any foreign-model eviction, swapping, or resource
   instability leaves Gate 0 open.

Exit: plan approved, production authorization and active closure goal recorded,
worktree baseline captured, owner amendment audit sealed, and every required
readiness probe passes. Otherwise no production edit starts.

### Gate 1: sealed baseline and architecture manifest

1. Materialize and seal the fixed current V2 inputs, validators, deterministic
   output hashes, live case manifest, rubric/calculation rules, performance
   protocol, calibration corpus, and environment-fingerprint schema declared
   by this plan. Record the plan SHA-256; no rubric, weight, node, threshold, or
   scoring-protocol choice is made in this gate.
2. For every missing reproduction/calibration/quality input, use Decision 49
   to copy the smallest sufficient production source through a read-only
   repository export, inspect it, and seal its provenance/hash before deriving
   a sanitized fixture. A synthetic approximation cannot replace available
   production evidence merely because it is easier to construct.
3. Extend the comparison harness in effect-free mode: no state commit, action
   execution, resolver capability execution, surface delivery, DB semantic
   write, adapter delivery, or scheduler effect.
4. Capture current V2 deterministic and patched-LLM paths and a separate
   current-partial-V3 topology report demonstrating the gaps above.
5. Run the 24 V2 live cases only in an eligible real-model environment, one
   case at a time, three trials each, and write reviews. If the eligible
   environment is unavailable, seal deterministic inputs but keep Gate 1 open.
6. Before any target V3 production edit, classify only the V2 case/contract/
   dimension cells whose sealed median is `0` into
   `v2_semantic_baseline_defects.v1`, with all three trial ids, scores,
   rationales, raw-artifact hashes, semantic owner, consequence, and authority
   source. Hard-boundary failures are recorded separately and keep Gate 1 open;
   they never become inherited semantic allowances.
7. Seal hashes under one `baseline_id`; in a separate parent audit pass, verify
   100% manifest path closure and artifact hashes before implementation begins.

Exit: immutable V2 baseline, inherited-semantic-defect registry, production-
data provenance records where needed, and target architecture manifest are
sealed. Unlike the superseded execution, this gate cannot be skipped.

### Gate 2: shared contracts, lane, anchor, and budget

1. Add public pure V2 validator/materializer APIs by renaming the current
   internal owners and updating V2 callers; prove V2 tests unchanged.
2. Add `context_window_tokens`, engine-conditional route loading, exact V3
   services construction, lane coordinator, prompt/anchor, transcript, and
   budget modules.
3. Bind the exact V3 chain and sidecar runtime bundles to the sealed Gate 0
   candidate fingerprints through protected configuration, declare the chain
   window at its verified loaded value of at least `50,000`, record whether it
   enables the 65,000 tier, and rerun the selected-engine startup matrix for
   both normal-only and extension-capable configurations. Keep credentials out
   of artifacts.
4. Implement canonical parser injection and protected tracing support.
5. Collect and run every Gate 2 node mapped in the Core and shared-owner impact
   matrix plus `tests/unit/cognition_core_v2/`; a missing mapped node fails.
6. Run token-estimator calibration and holdout; commit the selected multiplier
   and artifact digest.

Exit: shared substrate has no V2 behavior regression; all infrastructure
contracts pass; estimator satisfies its gate.

### Gate 3: cold primary chain

1. Implement A1/A2 grouping/fallback, I1 replay reduction, G1a/G1b, I2, W1,
   P1, failure repair, deadline, and O projection.
2. Remove parallel-wave/isolated-goal/route-checkpoint behavior from source and
   tests.
3. Prove exact state/emotion/relationship/output parity with patched model
   responses and exact-prefix structural invariants.

Exit: one cold V3 invocation produces a fully valid V2 output across every
registered deterministic path with observed primary concurrency one.

### Gate 4: sidecar and recurrence

1. Implement L1, lane-scoped JSON repair, X1/X2, absent/failing-sidecar
   dispositions, session registry, and R recurrence.
2. Prove exact reattachment and every cold-rebuild reason.
3. Run resolver, guardrail, live/idle connector, one-commit, authorization,
   required-selection, and group-self-cognition integration tests.

Exit: resolver observations continue the chain safely; sidecar failures grant
no authority; no duplicate preparation, effect, or commit occurs.

### Gate 5: observability and console

1. Implement protected transcript, chain-run DB schema/indexes, event family,
   service projection, console contract/client/panel, and documentation.
2. Prove best-effort storage failure cannot affect cognition.
3. Collect and run every Gate 5 node mapped in the Persistence, service, and
   console impact matrix, then start the approved local console test
   environment, validate with Browser, inspect console errors, responsive
   layout, redaction, and screenshot evidence.

Exit: complete sanitized chain observability is readable through existing
operator paths; protected content remains protected; visual sign-off passes.

### Gate 6: full deterministic and impact verification

1. Run `scripts.validate_test_impact` from the captured baseline and resolve
   every mapped node.
2. Run all V3 unit/integration tests, all changed shared-owner tests, unchanged
   V2 cognition/resolver/node/service/event/DB/console suites, and repository
   static checks.
3. Run the full regular test suite. Classify any unrelated pre-existing
   failure with evidence; no changed-boundary failure may remain.

Exit: deterministic suite green, impact coverage 100%, no undocumented
production source change.

### Gate 7: live quality, serving, and performance

1. Run the serving-overflow probe and long-context tests against the exact
   candidate route with the normal active ceiling fixed at 50,000 total tokens.
2. Exercise the 65,000 total tier deterministically. If a fixed eligible live
   case activates it, load the sealed primary to at least 65,000 between
   isolated test blocks, freeze the new fingerprint, and rerun that extended
   block; model load/reload is forbidden during an invocation or performance
   block. If no fixed live case needs extension, retain the 50,000 deployment
   baseline and record the extended tier as inactive/not loaded rather than
   inflating the baseline.
3. Run the matched 24-case V2/V3 quality protocol and human review.
4. On the first live semantic failure, retain its result and localize the
   owner. On two consecutive eligible failures for the same owner/contract,
   execute Decision 46 completely. Run no further full real-LLM E2E attempt
   for that failure until the stage node and patched handoffs pass. Apply
   Decisions 45 and 47 to every correction.
5. Evaluate the fixed A1/A2 product contract and direct singleton recovery;
   there is no appraisal topology selection pass.
6. Calculate the fixed `semantic_successes / 72` rate, verify at least `69`
   successes, verify every residual against the presealed inherited-defect
   registry, and create the deferred residual register for up to three
   accepted inherited semantic failures. A V3-only or hard failure blocks.
7. Run the five performance nodes under the fixed protocol.
8. In a separate sealed parent calculation-audit pass, verify calculations,
   raw artifacts, and the environment fingerprint without changing rubric
   scores.

Exit: the exact 95.00% semantic floor and baseline-clean non-regression gates
pass, every accepted inherited residual is visible and deferred, every hard
quality/overflow/context/cache/recurrence/performance threshold passes, and no
evaluation-integrity finding remains. An unavailable/inconclusive environment
leaves the gate open.

### Gate 8: cutover, observation, closure

1. Change the default selector to V3, update operator docs, and deploy with the
   accepted route/grouping/calibration configuration.
2. Run startup, health, one direct debug case, one user case, one group case,
   one resolver case, one required-selection case, and one self-cognition case.
3. Observe at least 100 eligible turns and review chain-run/event aggregates.
   Any new semantic concern follows the component-first Decision-46 ladder;
   accepted Gate 7 inherited residuals remain visible in the residual register
   and do not themselves block cutover. Any hard regression or new V3-only
   semantic failure triggers rollback and keeps Gate 8 open.
4. After the bulk cutover and observation are accepted, revisit each inherited
   residual once. Apply an architecture-grounded correction only when it stays
   within this plan and all affected component, quality, and performance gates
   can be rerun; otherwise create a new active bugfix draft containing the
   complete residual record, exact owner/path/nodes, and acceptance target.
   Closing or transferring the residual is evidence work and does not revoke
   the accepted 95.00% delivery threshold.
5. Freeze the complete diff and evidence, perform the Final Parent Code Audit,
   remediate every blocker, major, and minor finding, rerun affected mapped
   checks, refreeze, and repeat the audit until no finding remains open.
6. Record commits, tests, artifacts, environment fingerprint, observation
   summary, rollback readiness, and final sign-off; move the plan to
   `archive/completed/` only then and mark the closure goal complete.

Exit: V3 is the sole runtime, evidence is accepted, the parent audit is signed
off, deployment-revision rollback is recorded, both plans are archived, the
goal is complete, and no required work remains.

## Verification

Use `venv\Scripts\python` for all Python commands. Exact node names added by
implementation must match this plan and the impact manifest.

Baseline/current-state command already executed during plan authorship:

```powershell
venv\Scripts\python -m pytest tests\unit\cognition_core_v3 tests\integration\cognition_core_v3 tests\unit\test_cognition_core_selector.py tests\unit\nodes\test_persona_supervisor2_cognition.py::test_connector_calls_selected_engine_with_canonical_input tests\unit\nodes\test_persona_supervisor2_cognition.py::test_selected_engine_output_commits_once tests\test_config.py::test_cognition_core_engine_accepts_only_v2_or_v3 tests\test_config.py::test_cognition_core_engine_default_matches_cutover_state -q
```

Recorded result: `60 passed in 0.85s` on 2026-08-19.

Implementation verification sequence:

```powershell
venv\Scripts\python -m pytest tests\unit\cognition_core_v2 -q
venv\Scripts\python -m pytest tests\unit\cognition_core_v3 -q
venv\Scripts\python -m pytest tests\integration\cognition_core_v3 -q
venv\Scripts\python -m pytest tests\unit\cognition_resolver tests\unit\nodes -q
venv\Scripts\python -m pytest tests\test_llm_interface_contracts.py tests\test_utils.py tests\test_config.py tests\unit\test_cognition_core_selector.py -q
venv\Scripts\python -m pytest tests\test_db_cognition_chain_runs.py tests\test_cognition_chain_event_logging.py tests\test_service_cognition_graph.py -q
venv\Scripts\python -m pytest tests\test_control_console_contracts.py tests\test_control_console_kazusa_client.py tests\test_control_console_cognition_graph.py tests\test_control_console_web_surface.py tests\control_console_e2e\test_cognition_graph_e2e.py -q
venv\Scripts\python -m scripts.validate_test_impact --base-ref <gate-1-baseline-commit> --run
venv\Scripts\python -m pytest -q
```

Live commands run one exact node at a time and are followed by artifact
inspection before the next command. Calibration and overflow scripts first run
with `--help`, then their documented preflight/dry-run mode, then one explicit
live execution against the candidate route.

Live diagnosis and acceptance use this fixed escalation order:

1. Read the failed artifact/trace only far enough to identify the first failing
   owner and freeze its exact input/output/configuration evidence.
2. Run the exact deterministic parser, validator, reducer, carrier, and patched
   handoff nodes for that owner.
3. Run one exact real-LLM node test for the captured stage packet and inspect
   its human-readable artifact.
4. After a justified correction, run that stage node and one distinct
   countercase individually; after two consecutive eligible failures, complete
   `local_semantic_reset.v1` before proceeding.
5. Run the smallest live subgraph/graph cooperation node that crosses the
   repaired boundary.
6. Run the full real-LLM E2E node once as acceptance evidence. Its failure
   returns diagnosis to step 1; repeated full-E2E probing is prohibited.

Every semantic output remains evidence even when it fails. Environmental
invalidity and semantic failure are recorded separately, and only the former
permits the fixed matched-pair rerun described by the protocol.

## Acceptance Criteria

1. The implementation matches every governing-architecture invariant and
   every Confirmed Decision in this plan.
2. The current parallel/isolated/checkpoint V3 topology has no executable
   source, configuration, test, or documentation path.
3. Public input/output/state/surface/morning-refresh contracts are V2-identical;
   full deterministic parity is proven.
4. State reduction, relationship maintenance, 21 emotions, role assignments,
   selected operation, goal progress, action/resolver requests, and targetless-
   group response are complete and valid.
5. One primary lane is serialized, append-only, thinking-free, and protected
   from foreign interleaving; sidecar identity is mechanically distinct.
6. Repair, exhaustion, provider failure, sidecar absence, deadline, context
   pressure, cancellation, and session divergence follow their exact bounded
   dispositions and authorize no invalid state/effect.
7. Resolver recurrence uses the short continuation tail when safe and cold
   rebuilds with an explicit warning otherwise; one final commit remains true.
8. The estimator, total 50k baseline and conditional total 65k extension,
   per-step completion reservation, serving-overflow probe, one re-anchor, and
   long-context behavior pass.
9. Protected transcript, persisted chain run, event aggregate, service
   response, and console panel are correlated, bounded, redacted, and
   best-effort.
10. Source-test impact coverage is 100%; deterministic/full suites pass.
11. Live quality reaches at least `69/72` semantic successes (exact threshold
    `>= 0.95`), has at most three residual failures all preclassified as
    inherited V2 semantic defects with completed V3 correction attempts and
    deferred records, has zero V3-only semantic failures, and has zero hard
    schema/state/provenance/privacy/permission/authorization/effect failures.
12. Every baseline-clean capability-group and overall V3 mean is at least V2;
    all unfiltered quality results remain reported; all performance thresholds
    pass on eligible matched evidence with sealed parent blinded scoring and a
    separate calculation audit.
13. V3-only cutover and a fresh post-decommission 100-turn observation
    complete with zero hard regression; deployment-revision rollback is
    recorded and no V2 runtime selector remains.
14. Runtime prompts, production branches, fixtures, tests, reruns, artifacts,
    scoring, and reporting pass the evaluation-integrity audit with no test-
    tailored instruction, hidden branch, cherry-picking, or suppressed result.
15. Every two-consecutive-local-semantic-failure trigger has a closed root-
    cause record and component-first evidence; full real-LLM E2E tests were
    used only after smaller owners and handoffs passed.
16. Every missing-data event used a reviewed read-only production export and
    complete `production_data_extract.v1` provenance, or recorded evidence that
    the required production source genuinely did not exist.
17. README/ICDs/HOWTO describe the implemented contract exactly.
18. The Final Parent Code Audit has no open blocker, major, or minor finding.

## Progress Checklist

- [x] Governing architecture read and reconciled against current source.
- [x] Superseded plan and execution log audited.
- [x] Current partial-V3 gaps and reusable seams recorded.
- [x] Current targeted deterministic suite recorded (`60 passed`).
- [x] Independent plan review complete (PASS; reviewed SHA-256
  `2431E632B2DF4482A391AD6234F461A47B0F230483893655DF726AB8AB285367`).
- [x] Owner-directed execution governance, goal, question-free flight,
  compaction recovery, and readiness-probe rules encoded and parent-audited.
- [x] Owner amended execution on 2026-08-20 to permit delegation and require
  one reusable subagent for all remaining production-code changes.
- [x] Owner clarified 50k/65k as total request-window ceilings; completion
  reservation, readiness, extension, and overflow rules are parent-audited.
- [x] Owner fixed evaluation-integrity, two-consecutive semantic reset,
  inherited-V2 95.00% acceptance, component-first live diagnosis, and read-only
  production-data provenance rules; parent contract audit complete.
- [x] Owner plan approval recorded (2026-08-19).
- [x] Explicit production implementation authorization recorded.
- [x] Exact plan-closure goal created and active.
- [x] Gate 0 sanitized readiness probes pass.
- [x] Gate 1 baseline and architecture manifest sealed.
- [x] Gate 2 shared infrastructure and calibration accepted.
- [x] Gate 3 cold primary chain accepted.
- [x] Gate 4 sidecar and recurrence accepted.
- [x] Gate 5 observability and console accepted.
- [x] Gate 6 deterministic/impact verification accepted.
- [x] Gate 7 live quality/serving/performance accepted under the owner's
  2026-08-22 critical-only semantic and retained-evidence amendment.
- [ ] Gate 8 cutover/observation/final parent audit accepted.
- [ ] Plan archived as completed.
- [ ] Plan-closure goal marked complete after archive closure.

## Execution Evidence

### Plan-review evidence

- Independent-review baseline `git status --short` contained the owner's
  registry/supersession changes and this untracked replacement plan:
  `M development_plans/README.md`, deleted active superseded plan, untracked
  replacement plan, and untracked archived superseded plan. Those concurrent
  owner changes were preserved.
- Branch: `feature/cognition_core_v3_cache_affine`.
- HEAD: `047bed95` (`Fix complete-bid validation to accept role reference dicts`).
- Pre-remediation plan SHA-256:
  `B5FD43B3CF7DAD388FB2A95C3B04EF8626DB3A0880F136E1E5E8368317BB8A7A`.
- The plan audit read the governing architecture, former plan, plan registry,
  root README/HOWTO, V2/V3/resolver/node/LLM/DB/event/control-console contracts,
  current V3 source/tests, selector, connector, trace boundary, and cognition
  graph read path.
- Deterministic targeted suite: `60 passed in 0.85s`.
- Production source was not modified during plan authorship.
- Independent read-only review completed four full-document rounds. Final
  verdict: PASS on review-input SHA-256
  `2431E632B2DF4482A391AD6234F461A47B0F230483893655DF726AB8AB285367`,
  with zero blocker, major, or minor findings.
- The owner-directed execution-governance amendment started from plan SHA-256
  `837EAAD683FEF0EF12BCDA48D1E9C5A4FFBE3C2F6626B41B283EECC7CB7084DB`.
  The parent re-read the applicable development-plan and local-LLM architecture
  skills, plan governance/gate/evidence/continuity sections, and governing-
  architecture §§4.1–5.4 after compaction; then it verified all 23 public input
  fields, 58 exact source impact rows, 24 live cases, and 113 exact pytest node
  references. This amendment changed execution governance only and modified no
  production or test file. Gate 0 captures the approved plan hash after
  status/authorization are recorded.
- The owner's total-context clarification started from plan SHA-256
  `18CAF981A2E7FBCBF417712BE7F1B1F481F98D8523DD8A350FE93AF07502ADA1`.
  The parent re-read architecture G7, §9.5, §10, and §11.1 plus every plan
  budget/config/readiness/test/acceptance occurrence. The corrected contract
  starts at a 50,000-token total request ceiling, reserves the owning completion
  inside it, and conditionally extends the total ceiling to 65,000. This is a
  plan-only correction that restores the owner's architecture meaning; no
  production or test file was modified.
- The owner's evaluation-integrity and semantic-failure amendment started from
  plan SHA-256
  `296753D96BA07C1582EF987D8F03E4B60D24B05E320E6796BD223D9B2B4F56A1`.
  The parent re-read the development-plan, local-LLM architecture, debug-LLM,
  database-data-pull, and test-style/execution skills; the plan registry; the
  governing architecture's prompt, failure, budget, and parity sections; the
  V2/V3/resolver ICDs; and the plan's rules, failure, live review, gate,
  acceptance, evidence, and continuity sections. The amendment makes runtime
  and evaluation cheating auditable, stops case-chasing after two consecutive
  eligible local semantic failures, fixes the diagnostic ladder from owner node
  to E2E acceptance, freezes a 69/72 semantic-success floor with zero hard-
  failure allowance, and defines protected read-only production-data copying.
  It changes only this plan and preserves the architecture's deterministic
  owners, public contracts, permission boundaries, and fail-closed mechanics.

### Approval record

- **Decision date:** 2026-08-19.
- **Owner decision:** plan approved.
- **Execution instruction:** implementation is explicitly deferred; this
  approval does not authorize production-code edits, test/fixture creation,
  live probes, deployment actions, or Gate 0 execution.
- **Lifecycle update:** plan and registry status changed from `draft` to
  `approved`.
- **Goal state:** the plan-closure goal was not created because Mandatory Rule
  2 requires both approval and explicit production implementation
  authorization.
- **Execution state:** Gate 0 remains unopened. No readiness probe, production
  or test edit, model call, database operation, or implementation command ran
  as part of approval recording.
- **Approval-input plan SHA-256:**
  `25749D10AE023AFFF6E936C63F3A0D8B2FA28987DA021D1472F9AADB2BD3EA31`.

### Pre-approval read-only readiness snapshot

This 2026-08-19 snapshot is diagnostic input to Gate 0, not a substitute for
the post-approval Gate 0 fingerprint:

- `venv\Scripts\python.exe` is available and reports Python `3.14.6`.
- The configured production MongoDB accepted `ping` and collection-list reads;
  27 collection names were visible. `cognition_chain_runs` is absent, as
  expected before implementation. The endpoint currently exposes no
  authenticated user/role and server authorization is undeclared, so a
  read-only metadata check infers open access; Gate 0 revalidates the exact
  diagnostic create/index capability before any schema action.
- All 14 configured cognition routes reached their OpenAI-compatible `/models`
  endpoint and every configured model identity was present. One provider
  exposes eight model identities; the current cognition routes use two distinct
  identities and both report `state=loaded`, so a separate primary/sidecar pair
  is available without a new provider.
- Both configured cognition models report `max_context_length=262144` and
  `loaded_context_length=50176`. That loaded value satisfies the corrected
  50,000-token total baseline, so model reload is not a Gate 0 prerequisite.
  The conditional 65,000 tier is currently unavailable; Gate 7 reloads the
  sealed primary between isolated test blocks only if fixed live evidence
  activates that tier. Provider maximum capacity is sufficient.
- The planned `COGNITION_V3_CHAIN_LLM_*` and
  `COGNITION_V3_SIDECAR_LLM_*` bundles are absent before their Gate 2 loaders
  exist. Gate 2 binds them to the sealed Gate 0 candidate fingerprints and
  declares the verified loaded context window.
- The configured brain-service `/health` endpoint and control-console root were
  not listening during the snapshot. Gate 0 starts and checks the authorized
  local operator processes; their stopped pre-execution state is not treated as
  a contract defect.

### Execution authorization record

- **Authorization date:** 2026-08-19.
- **Owner instruction:** `Read development_plans/active/short_term/cognition_v3_hybrid_agentic_loop_reconciliation_plan.md and start to execute the plan`.
- **Authorization disposition:** explicit production implementation
  authorization accepted under Mandatory Rules 2 and 8; execution flight and
  Gate 0 opened.
- **Goal objective:** `Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion`.
- **Goal state:** active, with no token budget.

### Gate 0 opening evidence

```text
date/time: 2026-08-19T11:49:05Z
gate: Gate 0 - eligibility, authorization, goal, and readiness (in progress)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: no compaction; read the complete plan, development-plan skill and execution/cutover references, local-llm-architecture, python-venv, test-style-and-execution, py-style and both constraint references, governing architecture, registry, root README/HOWTO operational sections, and current V3 ICD/source-test inventory; next checkpoint is Gate 0 manifest materialization and sanitized readiness probes
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803; pre-existing owner changes were M development_plans/README.md, D development_plans/active/short_term/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md, ?? development_plans/active/short_term/cognition_v3_hybrid_agentic_loop_reconciliation_plan.md, and ?? development_plans/archive/superseded/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md
owned files: complete Create/Modify and governed-artifact surface in this plan; Gate 0 initially owns development_plans/README.md, this plan, tests/fixtures/cognition_core_v3_live_case_manifest.json, and tests/test_cognition_core_v3_manifest_contract.py
commands: read-only plan/document/source inventory; get_goal; create_goal; git status/branch/HEAD/hash capture
test/artifact results: pending Gate 0 manifest contract nodes
readiness-probe sanitized fingerprint and health: pending
live environment fingerprint: pending
active total context ceiling, completion reservation, and extension state: Gate 0 requires 50000 total tokens; pending candidate re-probe; conditional 65000 tier pending
production-data extract ids, exact provenance hashes, and redactions: none required at Gate 0 opening
evaluation-integrity audit result: opening audit found fixed cases/rubric in the plan and no Gate 0 runtime prompt change; complete parent amendment audit pending
local semantic reset ids and smallest-component evidence: none; no real-model semantic invocation has run
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 1/Gate 7
decisions mechanically applied from plan: parent-only execution, exact goal, question-free flight, baseline preservation, V2 default, big-bang V3 boundary, 50000/65000 total ceilings
unexpected findings: Gate 0 manifest and its contract test are absent as expected and must be created
sealed parent audit pass and findings: pending Gate 0 completion
exit disposition: Gate 0 open
ending commit and git status: pending
```

### Gate 0 completion evidence

```text
date/time: 2026-08-19T12:18:06Z
gate: Gate 0 - eligibility, authorization, goal, and readiness (passed)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: one automatic compaction; resumed from the retained Gate 0 checkpoint without restarting; rechecked the exact manifest rules, Decision 47-48 clauses, recurrence field table, Gate 0 exit, and historical review record; next checkpoint is Gate 1 architecture manifest, effect-free comparison harness, and sealed V2 baseline
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803; preserved pre-existing owner changes M development_plans/README.md, D development_plans/active/short_term/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md, ?? this replacement plan, and ?? development_plans/archive/superseded/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md
owned files: Gate 0 added tests/fixtures/cognition_core_v3_live_case_manifest.json and tests/test_cognition_core_v3_manifest_contract.py; lifecycle/evidence updates remain in development_plans/README.md and this plan; the ignored generator is test_artifacts/cognition_core_v3/gate0/build_live_case_manifest.py
commands: project-venv and import probes; sanitized MongoDB/server-capability probe; configured model /api/v0/models probe; local brain and console startup plus HTTP checks; canonical-input generation and validation; SHA-256 capture; 23-field parent audit; exact two-node pytest run; production-source diff audit
test/artifact results: tests/test_cognition_core_v3_manifest_contract.py::test_live_case_manifest_is_complete_and_closed and tests/test_cognition_core_v3_manifest_contract.py::test_live_case_manifest_fixes_72_trial_floor_and_inherited_defect_schema both passed; 2 passed in 0.69s; all 24 stored canonical inputs passed validate_cognition_core_input
readiness-probe sanitized fingerprint and health: venv Python 3.14.6 with fastapi/httpx/motor/pymongo/pytest importable; selected runtime remains cognition engine v2; brain /health HTTP 200 with status=ok, db=true, scheduler=true; control-console root and static asset HTTP 200, session endpoint HTTP 200, and protected bootstrap/model-route endpoints correctly returned 401 without authentication
live environment fingerprint: all 14 configured cognition routes resolved to endpoint SHA-256 prefix 94afb24309b0b0eb; provider exposed 8 models; distinct candidate primary and sidecar identities were resident and loaded; each reported provider maximum context 262144 and loaded context 50176; no model reload or foreign-model eviction occurred
active total context ceiling, completion reservation, and extension state: verified resident normal total ceiling 50000; completion remains reserved inside that total by the approved contract; conditional 65000 tier unavailable at the current 50176 loaded window and remains disabled
MongoDB diagnostic capability: URI SHA-256 prefix a58fbe781aae28a4; database-name SHA-256 prefix 03408146245d055d; ping passed; 27 collections visible; cognition_chain_runs correctly absent before implementation; zero authenticated users/roles and undeclared server authorization support the bounded open-access create/index capability inference; probe performed no mutation
production-data extract ids, exact provenance hashes, and redactions: none required; every captured regression input was available from reviewed repository fixtures/tests; no raw production data entered git
evaluation-integrity audit result: manifest is immutable case_manifest.v1 with 24 cases, three trials per engine, fixed V3 denominator 72, exact 69-success floor, maximum three presealed inherited semantic failures, and zero hard-gate allowance; no semantic model invocation, favorable rerun, fixture rewrite from an outcome, or prompt patch occurred in Gate 0
manifest/test hashes: tests/fixtures/cognition_core_v3_live_case_manifest.json SHA-256 CAEC964CBC421882F4CA1CC63A66F55B8B046BDF44044FE30CDE73E3DA21BA22; tests/test_cognition_core_v3_manifest_contract.py SHA-256 C5C228D0DBB0CF42DD6709736BCCAEC97F4DB34E1D02E24CBD60CC6FD90FA5CA
local semantic reset ids and smallest-component evidence: none; no eligible real-model semantic invocation has run
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 1/Gate 7
decisions mechanically applied from plan: parent-only execution; active exact closure goal; question-free flight; fixed 24-case table and group rubric; Decision 47 inherited-defect identity and deadline; Decision 48 rational 19/20 threshold; V2 remains default; target V3 production edits remain prohibited until Gate 1 seals
unexpected findings: the first in-memory generator command exceeded the Windows command-line length limit before execution; the ignored bounded generator plus chunked apply_patch transfer resolved it. Canonical validation caught and corrected one storage-equivalent +00:00 versus terminal-Z timestamp in the generated tool-result state before manifest materialization. Neither finding changed a behavior contract, source runtime, or public data shape
sealed parent audit pass and findings: PASS. Historical independent review evidence records four full-document rounds and final PASS on SHA-256 2431E632B2DF4482A391AD6234F461A47B0F230483893655DF726AB8AB285367 with zero blocker, major, or minor findings. The execution-governance amendment was re-audited against the fully read development-plan and local-LLM-architecture skills, this plan, and the governing architecture. Mechanical closure found exactly 23 recurrence-table fields equal to CognitionCoreInputV2 annotations with zero missing/extra/duplicate fields, 24 unique case ids, 24 unique fixture ids, 24 unique exact live-node ids, and no production-source diff
exit disposition: Gate 0 passed; Gate 1 opened. No target V3 production edit is authorized by gate order until the immutable V2 baseline and inherited-defect registry seal
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; Gate 0 additions are ?? tests/fixtures/cognition_core_v3_live_case_manifest.json and ?? tests/test_cognition_core_v3_manifest_contract.py; pre-existing owner lifecycle changes remain preserved; production source diff is empty
```

### Gate 1 compaction-recovery evidence

```text
date/time: 2026-08-19T15:08:32Z
gate: Gate 1 - sealed baseline and architecture manifest (in progress)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: two automatic Gate 1 compactions occurred after completion of the 72 eligible current-V2 live trials, the second while restoring the first compaction. The parent re-read the goal and worktree state; plan Summary, Scope, Mandatory Skills, Mandatory Rules, Confirmed Decisions 1-49, Target State, Contracts And Data Shapes, Runtime Or Resource Constraints, Cutover Policy, Test Impact And Traceability, Change Surface, Agent Autonomy Boundaries, Gate 1, Verification, Acceptance Criteria, Progress Checklist, Execution Evidence, Final Parent Code Audit, and Parent Execution Continuity; governing architecture sections 4.1-12.3; root README/HOWTO cognition, runtime, setup, service, and testing sections; the plan registry; V2, current-partial-V3, resolver, nodes, and LLM-interface ICDs; current V2/V3 facade, registry, executor, connector service construction, Gate 1 harness, manifest contract, live-case test, architecture manifest, and comparison contract. The parent re-applied development-plan plus execution/cutover references, local-llm-architecture, py-style plus both constraint references, cjk-safety, no-prepost-user-input, test-style-and-execution, debug-llm, and python-venv. The next checkpoint is the independent Gate 1 path/hash closure audit and immutable inherited-defect registry seal
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803; preserved owner lifecycle changes M development_plans/README.md, D development_plans/active/short_term/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md, ?? this plan, and ?? development_plans/archive/superseded/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md; Gate 0/Gate 1 additions remain the seven untracked tests/fixtures/harness paths listed by git status
owned files: Gate 1 owns tests/cognition_core_v3_comparison_harness.py, tests/test_cognition_core_v3_comparison_contract.py, tests/test_cognition_core_v3_live_llm.py, tests/test_cognition_core_v3_manifest_contract.py, the three cognition_core_v3 governed fixtures, this evidence record, and ignored protected artifacts under test_artifacts/cognition_core_v3/cogv3-g1-047bed95-331653f8
commands: read-only recovery reads and SHA-256 inventory; current goal query; git status; project-venv existence check; protected artifact count and reset inventory
test/artifact results: baseline id cogv3-g1-047bed95-331653f8 currently contains 74 raw V2 attempt artifacts, exactly 72 eligible trial reviews, two retained matched-pair invalidation records for resolver_observation_continuation trial 1, and six local_semantic_reset.v1 directories with 34 supporting JSON/Markdown/JUnit files; the 72 eligible trials cover 24 fixed cases times three trials
readiness-probe sanitized fingerprint and health: unchanged from Gate 0 completion; project venv remains present
live environment fingerprint: unchanged from Gate 0 and the sealed V2 trial artifacts; eligible resident real-local-model environment supplied every retained semantic result
active total context ceiling, completion reservation, and extension state: normal total ceiling 50000 remains available; conditional 65000 tier remains unavailable at loaded context 50176
production-data extract ids, exact provenance hashes, and redactions: none required; all fixed inputs remained available from governed local fixtures
evaluation-integrity audit result: every eligible V2 semantic result remains retained. The only state change made after the first compaction and before the full recovery re-read completed was the ignored human-authored review test_artifacts/cognition_core_v3/cogv3-g1-047bed95-331653f8/reviews/v2/multi_goal_competition__trial-3.md, sealed from the already-completed eligible raw artifact SHA-256 53bfbf7273fc8093faf18fa1acd1e9f53786439b616f1fa70da02703189e06e3. Recovery then restarted before any additional edit, test, live call, DB operation, or production action. The review supplies the final missing fixed-trial score and changes neither raw evidence nor runtime behavior
local semantic reset ids and smallest-component evidence: six closed current-V2 reset ids are v2_event_agency_third_party_actor, v2_reciprocity_pronoun_role_direction, v2_neutral_schedule_user_actor, v2_future_speak_schema_literal, v2_verbal_abuse_duplicate_event_identity, and v2_verbal_abuse_existential_actor_direction; each directory retains its local_semantic_reset.json, deterministic mechanics result, exact stage reproduction, distinct countercase, and agent-authored diagnostic review, with the applicable discriminating experiment retained
semantic successes/72, inherited residual ids, and baseline-clean means: Gate 1 current-V2 control scoring is complete; exact inherited case/contract/dimension cells and means remain pending the separate calculation audit. Gate 7 V3 successes/72 and residual ids remain pending
decisions mechanically applied from plan: parent-only execution; exact active no-budget goal; question-free flight; immutable fixed manifest and scoring protocol; effect-free direct-facade capture; retain-all semantic evidence; Decision 46 component-first resets; Decision 47 pre-edit inherited classification; V2 default; target V3 production edit barrier remains active
unexpected findings: the recovery process itself compacted once while reading the large nodes ICD; the second recovery used bounded exact line ranges and current file hashes. The premature review write is disclosed above and is confined to ignored evidence already produced before compaction
sealed parent audit pass and findings: recovery audit passed for continuity, authority, current gate, and worktree ownership. Target V3 production diff remains empty. Gate 1 exit remains open pending the independent artifact/path/hash audit and immutable defect registry
exit disposition: Gate 1 remains in progress
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; production source diff remains empty; current plan entry SHA-256 before this evidence append was 331653f8cd8c42a60620ea267e1af3d157d8cf03855295665f79a922b3f16fd0
```

### Gate 1 baseline-index path amendment

- **Discovery:** Gate 1 step 7 and the fixed live-case contract require one
  sealed baseline index containing canonical input and artifact hashes, while
  the original generated-artifact list named the defect registry but omitted
  the final index path.
- **Authority and scope:** Mandatory Rule 4 permits the parent to add a path
  strictly necessary to realize an already fixed contract. The amendment adds
  only the exact ignored path
  `test_artifacts/cognition_core_v3/cogv3-g1-047bed95-331653f8/baseline_index.json`
  and its deterministic tests-only validator node. It changes no production
  path, semantic owner, payload, rubric, threshold, permission, cutover rule,
  or retained evidence.
- **Parent contract audit:** PASS against Gate 1 steps 1 and 7, the fixed
  live-case baseline-index requirement, the development-plan execution-gate
  reference, and the architecture parity boundary. The pre-amendment plan
  SHA-256 was
  `77DB8CAC3B03AE62470DB6CE6848C3D07EE8E2BE287901E9949B3470E3E75655`.
- **Next checkpoint:** implement and collect the exact validator node, build
  the final index from already sealed evidence, then perform the separate
  100% path/hash audit.

### Gate 1 third compaction-recovery evidence

```text
date/time: 2026-08-19T15:22:46Z
gate: Gate 1 - sealed baseline and architecture manifest (in progress)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: one automatic Gate 1 compaction occurred while apply_patch was adding the tests-only baseline-index validator. The parent re-read the active goal and worktree; plan Summary, Scope And Change Direction, Mandatory Skills, Mandatory Rules, Confirmed Decisions 1-49, Target State, Contracts And Data Shapes, Runtime Or Resource Constraints, Cutover Policy, Test Impact And Traceability, Change Surface, Agent Autonomy Boundaries, Gate 1, Verification, Acceptance Criteria, Progress Checklist, Execution Evidence, Final Parent Code Audit, and Parent Execution Continuity; governing architecture sections 4.1-12.3; the plan registry; root README/HOWTO cognition, architecture, runtime, service, setup, and testing sections; current V2, partial-V3, resolver, nodes, and LLM-interface ICDs; current V2/V3 facades, V3 registry/executor, connector service construction, comparison harness, comparison contract, live-case test, manifest contract, architecture manifest, protected path fingerprints, pre-live index, and inherited-defect registry. The parent re-applied development-plan plus execution/cutover references, local-llm-architecture, python-venv, test-style-and-execution, py-style plus both constraint references, cjk-safety, debug-llm, and no-prepost-user-input. The next checkpoint is the exact baseline-index validator node, followed by regenerated path fingerprints and the sealed final index
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803; preserved owner lifecycle changes M development_plans/README.md, D development_plans/active/short_term/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md, ?? this plan, and ?? development_plans/archive/superseded/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md; the seven Gate 0/Gate 1 untracked tests/fixtures/harness paths remain owned by this execution
owned files: Gate 1 owns tests/cognition_core_v3_comparison_harness.py, tests/test_cognition_core_v3_comparison_contract.py, tests/test_cognition_core_v3_live_llm.py, tests/test_cognition_core_v3_manifest_contract.py, the three governed cognition_core_v3 fixtures, this evidence record, and protected artifacts under test_artifacts/cognition_core_v3/cogv3-g1-047bed95-331653f8
commands: get_goal; git status and production-source diff check; complete bounded recovery reads; project-venv existence check; SHA-256 and interrupted-patch boundary inspection
test/artifact results: the interrupted apply_patch committed atomically and left one complete validate_baseline_index function plus constants/helpers in tests/cognition_core_v3_comparison_harness.py at SHA-256 1ef6279fb30fd8f2ad3341d3b52ee7f8eaaa5b4514ef3e001876becfcdc3b48d; the mapped comparison-contract node remains to be added and collected; the sealed baseline still contains 72 eligible V2 trials, 72 parent-authored reviews, two retained invalid attempts, two matched-pair invalidations, six closed semantic resets, and seven presealed inherited defect cells
readiness-probe sanitized fingerprint and health: unchanged from Gate 0 and prior Gate 1 evidence; venv\\Scripts\\python.exe remains present
live environment fingerprint: unchanged from the sealed eligible V2 evidence; resident primary/sidecar identities and the 50176-token loaded window remain the last authorized fingerprint
active total context ceiling, completion reservation, and extension state: normal total ceiling 50000 remains the sealed baseline; completion reservation stays inside the total; conditional 65000 extension remains unavailable at the sealed 50176 loaded window
production-data extract ids, exact provenance hashes, and redactions: zero; every fixed Gate 1 input came from governed local sources
evaluation-integrity audit result: recovery inspected existing retained artifacts only. The baseline-index validator measures complete path/hash/count closure and contains no runtime prompt, semantic-output rewrite, score mutation, favorable rerun, fixture choice, or production branch
local semantic reset ids and smallest-component evidence: the six previously closed ids remain v2_event_agency_third_party_actor, v2_reciprocity_pronoun_role_direction, v2_neutral_schedule_user_actor, v2_future_speak_schema_literal, v2_verbal_abuse_duplicate_event_identity, and v2_verbal_abuse_existential_actor_direction, each with its retained deterministic, exact-stage, distinct-countercase, and parent-authored review evidence
semantic successes/72, inherited residual ids, and baseline-clean means: V2 control evidence remains 72 eligible trials; seven inherited case/contract/dimension cells are sealed in v2_semantic_baseline_defects.v1; V3 successes, residuals, and comparative means remain Gate 7 work
decisions mechanically applied from plan: parent-only execution; active exact closure goal; question-free flight; immutable Gate 1 inputs/reviews/defects; complete baseline path/hash closure; V2 default; target V3 production edit barrier remains active
unexpected findings: the interrupted tool output reflected display truncation rather than a partial filesystem write. Read-only inspection found the validator complete through its final defect-registry check, while the companion mapped pytest node had not yet been written
sealed parent audit pass and findings: recovery continuity audit passed. Current plan SHA-256 before this evidence append was 60882dc1e56cfb3ba631a3c96968993ce0a669a2f5863cbb4ddaf80aed9ca108; production source diff remains empty; Gate 1 exit remains open for validator verification and final index closure
exit disposition: Gate 1 remains in progress
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; owner lifecycle changes and seven Gate additions remain preserved; production source diff remains empty
```

### Gate 1 completion evidence

```text
date/time: 2026-08-19T15:29:12Z
gate: Gate 1 - sealed baseline and architecture manifest (passed)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: the third Gate 1 compaction recovery is recorded immediately above and remained current through seal completion; next checkpoint is Gate 2 canonical shared-helper renames plus lane/config/anchor/transcript/budget infrastructure
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803; preserved owner lifecycle changes M development_plans/README.md, D development_plans/active/short_term/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md, ?? this plan, and ?? development_plans/archive/superseded/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md; seven Gate 0/Gate 1 tests/fixtures/harness paths remain execution-owned
owned files: Gate 1 sealed tests/cognition_core_v3_comparison_harness.py, tests/test_cognition_core_v3_comparison_contract.py, tests/test_cognition_core_v3_live_llm.py, tests/test_cognition_core_v3_manifest_contract.py, tests/fixtures/cognition_core_v3_architecture_manifest.json, tests/fixtures/cognition_core_v3_live_case_manifest.json, tests/fixtures/cognition_core_v3_token_calibration_corpus.json, this evidence record, and protected baseline cogv3-g1-047bed95-331653f8
commands: exact live trials and parent-authored review sequence recorded in protected artifacts; V2 deterministic and current-partial-V3 topology captures; py_compile; Ruff; exact comparison/manifest pytest suites; final baseline-index generation; separate read-only 24x3 review/raw/defect/path/hash calculation audit
test/artifact results: py_compile passed for all four Gate 1 Python owners; Ruff passed for the harness and three Gate 1 tests; comparison and manifest contracts passed 10/10 in 0.83s, including tests/test_cognition_core_v3_comparison_contract.py::test_baseline_index_validator_rejects_missing_or_changed_artifacts. Current V2 deterministic capture records 139 passed and two explicitly classified pre-existing failures: goal-progress prompt assertion drift and surface-score calibration fixture owner-set drift. Current partial V3 topology capture records 53 passed and the ten target gap codes. The final baseline index contains 192 protected pre-index artifact rows: 72 eligible V2 raw trials, two invalidated raw attempts, 72 reviews, two matched-pair invalidations, six local semantic resets, 28 reset-support artifacts, three deterministic reports, six baseline-governance artifacts, and one defect registry
readiness-probe sanitized fingerprint and health: inherited unchanged from Gate 0; service, MongoDB, provider, model-residency, and control-console readiness were already sealed before live capture
live environment fingerprint: eligible real-local-model environment produced every retained V2 semantic result; sealed route endpoint SHA-256 prefix 94afb24309b0b0eb, distinct loaded primary/sidecar model identities, provider maximum context 262144, and loaded context 50176
active total context ceiling, completion reservation, and extension state: 50000 total-token normal ceiling remains available with completion reserved inside the total; the 65000 conditional tier remains unavailable at loaded context 50176 and stays disabled
production-data extract ids, exact provenance hashes, and redactions: zero; every fixed captured-regression, synthetic, adversarial, calibration, and quality input was available from reviewed local governed sources
evaluation-integrity audit result: PASS. Exactly 74 V2 attempts remain: 72 eligible semantic results plus two resolver fixture-invalid attempts with semantic_result_available=false and complete matched-pair invalidation attestations. Exactly 72 reviews join one-to-one to the eligible 24-case x 3-trial grid by raw SHA-256. Every valid but poor output remains retained. The final index validates every governed current hash except the deliberately frozen Gate 1 entry-plan hash, every protected path/hash/size, every canonical input hash, all 116 owned paths, and the fixed counts. No prompt/test branch, score rewrite, semantic rerun, favorable sample selection, or omitted result entered the baseline
local semantic reset ids and smallest-component evidence: six closed ids are v2_event_agency_third_party_actor, v2_reciprocity_pronoun_role_direction, v2_neutral_schedule_user_actor, v2_future_speak_schema_literal, v2_verbal_abuse_duplicate_event_identity, and v2_verbal_abuse_existential_actor_direction. Each retains local_semantic_reset.v1, deterministic mechanics, exact-stage reproduction, distinct countercase, parent-authored review, and any discriminating experiment
semantic successes/72, inherited residual ids, and baseline-clean means: current V2 control has 51 semantic successes and 21 semantic failures across 72 eligible trials. The immutable inherited registry contains seven exact median-zero cells: event_agency_and_moral_chain/groundedness; relationship_reciprocity/role_and_target_fidelity; ordinary_neutral_response/role_and_target_fidelity; future_speak_authority/contract_fidelity; future_speak_authority/task_progress; verbal_abuse_boundary/contract_fidelity; verbal_abuse_boundary/role_and_target_fidelity. Unfiltered V2 capability means are action/resolver 1.500, appraisal/state 1.500, goal/selection 1.680, group/self-cognition 1.667, relationship 1.150, robustness 1.833, overall 1.532 across 111 cells. Baseline-clean V2 means are action/resolver 1.688, appraisal/state 1.565, goal/selection 1.750, group/self-cognition 1.667, relationship 1.353, robustness 1.833, overall 1.635 across 104 cells after excluding only those seven presealed cells. Gate 7 V3 successes and inherited residual dispositions remain pending
decisions mechanically applied from plan: parent-only execution; immutable current-V2 baseline; Decision 46 component-first resets; Decision 47 inherited-defect deadline and identity; Decision 48 raw-score retention and comparative exclusions; effect-free facade capture; V2 remains default; target V3 production edit barrier lifts only with this Gate 1 pass
unexpected findings: the missing final-index path was resolved through the recorded Mandatory Rule 4 amendment before creation. Two resolver fixture-invalid attempts and their two successive corrections remain visible and created no inherited allowance. The two pre-existing deterministic V2 test failures remain baseline evidence and are outside the target V3 production change
sealed parent audit pass and findings: PASS. Baseline id cogv3-g1-047bed95-331653f8; Gate 1 entry plan SHA-256 331653f8cd8c42a60620ea267e1af3d157d8cf03855295665f79a922b3f16fd0; current plan SHA-256 before this completion append a26172bfa0ac4c79fae51b4f5f5cbe9d9ba3610009cf437377c36153cd12ecd1; architecture manifest f5a40bedc1fc33f16221f1c2e33c1017426f7580c5d1422eafbdfb5d261f0259; live manifest 2c3c04ad168d6adb4493c6d6c81479b5c08ff1d594038c24b8bfe1d062a8cd3f; token corpus cc8715b0022243c8f2b59120bfaf508ca075e0aa38e805e25d781a226d4688d7; comparison harness 29bfb403f0790f41e73487ed8d2ce708896fe404e2e7c5cafc30d73460b66bff; comparison contract a8730223f14caabd2799775cbd286022e8648ff9e2ade00ce0ddd6e592addcf7; live test b89bab4f64732280156079037ec14d1f9db689b9d86361c2407cba8ca41b9791; manifest contract a23346b1b1fdd40792b1c5bd5bf9b3e4472aba28a85492b5fd685a0b58a398aa; path fingerprint c04651e53403b42627b0ed1c5364571546410b6eb670b9909533c380df12e741; defect registry 57affca1b0e0790b35ae2d089caeadb31a4acdad823534db46c376e4c9ee790f; final baseline index 7342b30664784c6d43591058d181524d31054e3c1601d4a5daa9854716057ca7. Production source diff remains empty
exit disposition: Gate 1 passed; immutable baseline and architecture manifest sealed; Gate 2 opened
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; owner lifecycle changes and the seven Gate additions remain preserved; target production source diff is empty at the Gate 2 entry boundary
```

### Gate 2 compaction-recovery evidence

```text
date/time: 2026-08-19T15:34:46Z
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: a conversation compaction occurred while the Gate 1 completion evidence patch was being applied. The patch landed cleanly with the Gate 1 checklist marked complete, a closed evidence fence, and Gate 2 opened. The parent re-read the plan Summary; Scope And Change Direction; Mandatory Skills and Rules; Confirmed Decisions 1-49; Target State; V3 services, routing, LLM config, parser, transcript, and recurrence contracts; Runtime Or Resource Constraints; Change Surface; Agent Autonomy Boundaries; Gate 2; Verification; Acceptance Criteria; Progress Checklist; Execution Evidence; and Parent Execution Continuity. The parent also re-read development_plans/README.md; applicable README.md and docs/HOWTO.md cognition/testing guidance; governing-architecture sections 1-11; the development-plan execution-gates and cutover references; local-llm-architecture; python-venv; py-style and both constraint references; cjk-safety; test-style-and-execution; debug-llm; and no-prepost-user-input. Active-source inspection covered the five canonical V2 helper owners, their current call sites, exact mapped direct-owner tests, and additional tests importing those private names. Next checkpoint is the exact atomic rename of the seven V2 helper symbols plus all V2/test call sites, followed by collection and execution of the five mapped owner nodes and the full tests/unit/cognition_core_v2 suite
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803; preserved owner lifecycle changes M development_plans/README.md, D development_plans/active/short_term/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md, ?? this plan, ?? development_plans/archive/superseded/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md; seven Gate 0/Gate 1 test/fixture/harness additions remain execution-owned; git diff under src is empty
owned files for the first Gate 2 checkpoint: src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py, goal_cognition.py, workspace.py, action_selection.py, action_authorization.py; their five tests/unit/cognition_core_v2 direct-owner modules; tests/test_cognition_clarification_consumers.py; tests/test_cognition_core_v2_action_planning_bugfix.py; tests/test_cognition_core_v2_trace_failure_mode_matrix.py; src/kazusa_ai_chatbot/cognition_core_v2/README.md; tests/ownership/source_test_impact_manifest.json; and this plan evidence record
commands and read-only results: get_goal confirmed the exact active no-budget objective; git status and HEAD/branch checks confirmed no production-source diff; plan patch inspection confirmed Gate 1 completion; current plan SHA-256 before this recovery append was c54baa8434199117ff8bc2d8a8af0f3e5a0cdb326a0df87030576ce6e14facbf; venv/Scripts/python.exe exists
completed evidence retained: Gate 0 passed; Gate 1 baseline cogv3-g1-047bed95-331653f8 remains sealed with final baseline index SHA-256 7342b30664784c6d43591058d181524d31054e3c1601d4a5daa9854716057ca7, exact 24x3 eligible V2 grid, seven inherited median-zero cells, six closed local semantic resets, zero eligible hard-boundary failures, and a passed independent parent path/hash audit
active total context ceiling, completion reservation, and extension state: sealed loaded primary context remains 50176; the 50000 total-token normal ceiling is available with completion reserved inside the total; the conditional 65000 tier remains unavailable
decisions mechanically applied from plan: production work starts only after the sealed Gate 1 boundary; the first slice is rename-only with no alias or copied implementation; every current V2 caller and test reference moves atomically; V2 behavior and payloads remain unchanged; exact mapped owner nodes must collect and pass before this slice closes
unexpected findings: the plan Summary still states Gate 1 is in progress although the authoritative checklist and completion evidence show Gate 1 passed; update that lifecycle sentence to Gate 2 in progress after this recovery record. Existing tests outside tests/unit/cognition_core_v2 import two of the private helper names and therefore belong to the atomic call-site rename
exit disposition: recovery complete; Gate 2 remains in progress; production edits may resume at the canonical V2 helper-rename checkpoint
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; owner lifecycle changes and the seven Gate additions remain preserved; production source diff remains empty
```

### Gate 2 canonical shared-helper checkpoint evidence

```text
date/time: 2026-08-19T15:40:34Z
gate: Gate 2 - canonical V2 shared-helper extraction checkpoint (passed)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: exact plan-closure goal remains active with no token budget
starting boundary: Gate 1 sealed, Gate 2 recovery recorded, and no production-source diff existed before this checkpoint
changed production owners: cognition_core_v2/semantic_appraisal.py, goal_cognition.py, workspace.py, action_selection.py, and action_authorization.py; changes are exact symbol renames plus their own V2 call sites. cognition_core_v2/README.md now documents the seven canonical owners
canonical names: canonicalize_semantic_appraisal_item, validate_semantic_boundary_candidate, merge_semantic_appraisal_item, selection_goal_draft_to_goal_bid, validate_workspace_partition, validate_action_plan_decision, validate_authorization_decisions
atomic call-site result: every V2 runtime caller, direct-owner test, tests/test_cognition_clarification_consumers.py, tests/test_cognition_core_v2_action_planning_bugfix.py, and tests/test_cognition_core_v2_trace_failure_mode_matrix.py uses the canonical name. Repository search over cognition_core_v2 plus tests finds zero superseded private names. No alias, copied implementation, adapter, fallback, or behavior change was introduced
source-test ownership: tests/ownership/source_test_impact_manifest.json maps each of the five changed production owners to the exact Gate 2 node fixed by the plan
syntax/style evidence: venv Python py_compile passed for all five source owners and eight directly changed test modules. Ruff import-order checks passed for every changed test module. A full Ruff scan of the large existing V2 modules continues to report pre-existing TRY004/import-order/FLY002 findings outside this rename-only diff; this checkpoint does not perform drive-by behavior or style changes
exact mapped collection: all five required node ids collected, 5 tests collected in 0.69s
exact mapped execution: all five required owner nodes passed, 5 passed in 0.66s
full V2 owner suite: 133 passed and exactly two failed in 1.06s. The failures are the sealed Gate 1 baseline rows test_goal_progress_model_output_omits_protocol_metadata and test_calibration_artifact_requires_owner_specific_threshold; neither changed owner logic nor the helper extraction contributes to either assertion drift. All five new owner nodes passed inside the suite
additional renamed-import suites: 67 passed and one failed in 0.98s. The failure test_relational_willingness_fields_not_exact_is_rejected exercises unchanged validate_goal_bid_draft error precedence and is outside every renamed implementation; the import/call-site changes passed throughout all three modules
evaluation-integrity result: PASS. Tests invoke the same implementations under their canonical names and assert existing contracts; no expected semantic answer, fixture, production branch, or score changed
exit disposition: canonical V2 substrate extraction accepted; Gate 2 continues with LLM config, engine-lazy route settings, CognitionChainServicesV3, lane, anchor, prompt, transcript, budget, parser injection, protected tracing, and estimator calibration
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; preserved owner lifecycle and Gate 1 additions remain present; production diff is limited to the five rename owners and their V2 ICD
```

### Gate 2 configuration-checkpoint compaction-recovery evidence

```text
date/time: 2026-08-19T15:49:35Z
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: one conversation compaction occurred while inspecting the LLM provider transport boundary for the configuration/service-construction slice. The parent re-read the active goal and worktree; plan Summary, Scope And Change Direction, Mandatory Skills, Mandatory Rules, Confirmed Decisions 1-49, Target State, Contracts And Data Shapes, Runtime Or Resource Constraints, Cutover Policy, Test Impact And Traceability, Change Surface, Agent Autonomy Boundaries, Gate 2, Verification, Acceptance Criteria, Progress Checklist, Execution Evidence, Final Parent Code Audit, and Parent Execution Continuity; governing-architecture sections 4.1-12.3; the plan registry; applicable root README/HOWTO cognition, route, service-startup, and testing guidance; current V3, nodes, and LLM-interface ICDs; and the config, LLM call contract/provider/report, selector, connector service construction, V3 contracts/facade bindings, and exact mapped config/LLM/report/connector/V3-contract tests. The parent re-applied development-plan plus execution/cutover references, local-llm-architecture, python-venv, py-style plus both constraint references, cjk-safety, test-style-and-execution, debug-llm, and no-prepost-user-input. The next checkpoint is the exact LLMCallConfig context declaration, engine-first immutable route settings loaders, CognitionChainServicesV3 validation, branch-local selected service construction, and selected-family route diagnostics, followed by collection and execution of every mapped node for those owners
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803; preserved owner lifecycle changes, seven Gate 0/Gate 1 additions, and the accepted Gate 2 helper-rename source/test/ICD changes are present exactly as listed by git status; no configuration/service-construction source edit exists at this recovery boundary
owned files for this checkpoint: src/kazusa_ai_chatbot/config.py; src/kazusa_ai_chatbot/llm_interface/contracts.py; src/kazusa_ai_chatbot/llm_interface/route_report.py; src/kazusa_ai_chatbot/cognition_core_v3/contracts.py; src/kazusa_ai_chatbot/cognition_core_v3/__init__.py; src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py; their exact mapped tests in tests/test_config.py, tests/test_llm_interface_contracts.py, tests/test_llm_interface_route_report.py, tests/unit/cognition_core_v3/test_contracts.py, and tests/unit/nodes/test_persona_supervisor2_cognition.py; tests/ownership/source_test_impact_manifest.json; affected ICD rows; and this evidence record
commands and read-only results: get_goal confirmed the exact active no-budget objective; git status confirmed only preserved lifecycle, Gate 1, and accepted helper-checkpoint changes; current pre-append plan SHA-256 was 4975f9779249681f82c6771ccfa30832d58069a70aef713da2d344f25969f299; venv/Scripts/python.exe remains present; repository searches found the current eager twelve-route reads only in config.py and persona_supervisor2_cognition.py, with route-report and V2 live-test consumers recorded for contract preservation
completed evidence retained: Gates 0 and 1 remain passed and sealed; the canonical V2 shared-helper checkpoint remains accepted with all five exact owner nodes passing, no superseded private names, and no helper alias or copied implementation
active total context ceiling, completion reservation, and extension state: sealed loaded primary context remains 50176; the 50000 total-token normal ceiling is available with completion reserved inside the total; the conditional 65000 tier remains unavailable
decisions mechanically applied from plan: COGNITION_CORE_ENGINE is parsed before core route settings; only the selected V2 or V3 family is read; shared non-core COGNITION_LLM remains required; V2 retains exactly twelve settings rows with no context declaration; V3 uses one required chain row plus one all-absent/all-present sidecar row; connector configs are constructed only inside the selected branch; context_window_tokens remains caller-owned and is not transported to the provider; no inactive-family alias, fallback, or LLMCallConfig is constructed
unexpected findings: current route diagnostics omit the required character-carryover shared route and have no selected-family label; the exact planned report must add selected core rows and label generic COGNITION_LLM as shared_non_core. The existing V2 live test reads two active V2 config constants; this checkpoint preserves those constants only when V2 is selected while removing them entirely from V3 startup, so the live test remains valid without a compatibility shim or unlisted test edit
evaluation-integrity audit result: PASS for this recovery. No model invocation, prompt change, fixture mutation, semantic rerun, score change, or test-result suppression occurred
exit disposition: recovery complete; Gate 2 remains in progress and configuration/service-construction edits may begin
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; worktree scope is unchanged apart from this required evidence append
```

### Gate 2 configuration-block compaction-recovery evidence

```text
date/time: 2026-08-19T16:02:31Z
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: a conversation compaction occurred while an apply_patch operation was replacing config.py's eager twelve-route base/API/model block with the selected-engine branch. The parent recovered the exact active goal and worktree, then re-read the plan Summary; Scope And Change Direction; Mandatory Skills; Mandatory Rules; Must Do and Deferred boundaries; V3 service, route-loader, LLMCallConfig, parser, and runtime-resource contracts; Test Impact rows; Change Surface; Agent Autonomy Boundaries; Gate 2; Acceptance Criteria; Progress Checklist; prior Execution Evidence; and Parent Execution Continuity. The parent re-read the governing architecture's volatility, chain, deterministic-interlude, lane/service, failure, interface, context-budget, cache, and cutover sections; the plan registry; applicable root README/HOWTO route, startup, and testing guidance; current V3, nodes, and LLM-interface ICD sections; and the active config, LLM contract/provider/report, V3 services/facade boundary, connector construction, manifest mappings, and all six exact checkpoint tests. The parent re-applied development-plan plus execution/cutover references, local-llm-architecture, python-venv, py-style plus both complete constraint references, cjk-safety, test-style-and-execution, debug-llm, and no-prepost-user-input. The next checkpoint is removal of the remaining duplicate V2 completion/thinking reads, branch-local connector construction, selected-family route diagnostics, source-test mapping and ICD updates, followed by exact collection and execution
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803; preserved owner lifecycle changes, seven Gate 0/Gate 1 additions, the accepted helper checkpoint, and the in-progress configuration/service source and test files are present in the recovered git status
owned files for this checkpoint: src/kazusa_ai_chatbot/config.py; src/kazusa_ai_chatbot/llm_interface/contracts.py; src/kazusa_ai_chatbot/llm_interface/route_report.py; src/kazusa_ai_chatbot/cognition_core_v3/contracts.py; src/kazusa_ai_chatbot/cognition_core_v3/__init__.py; src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py; their exact mapped tests in tests/test_config.py, tests/test_llm_interface_contracts.py, tests/test_llm_interface_route_report.py, tests/unit/cognition_core_v3/test_contracts.py, and tests/unit/nodes/test_persona_supervisor2_cognition.py; tests/ownership/source_test_impact_manifest.json; affected V3/nodes/LLM-interface/operator ICD rows; and this evidence record
commands and read-only results: get_goal confirmed the exact active no-budget objective; git status recovered the complete changed-path boundary; current pre-append plan SHA-256 was 2826c9c717e8ac7e6e2b3ae8dc25ecb83536c4a18507512672efa653a63ce761; source inspection confirmed the interrupted apply_patch landed completely through `else: _COGNITION_V3_ROUTE_SETTINGS = load_cognition_v3_route_settings()` at the selected-engine branch
completed evidence retained: Gates 0 and 1 remain passed and sealed; the canonical V2 shared-helper checkpoint remains accepted; the pre-change exact service-construction node remains recorded as an expected ImportError before CognitionChainServicesV3 existed; the LLMCallConfig declaration, immutable route settings loaders, selected config branch, V3 service validator, and their six exact tests remain present for completion of this checkpoint
active total context ceiling, completion reservation, and extension state: sealed loaded primary context remains 50176; the 50000 total-token normal ceiling is available with completion reserved inside the total; the conditional 65000 tier remains unavailable
decisions mechanically applied from plan: selected-engine route loading remains the only core configuration authority; shared non-core routes stay required; V2 retains exactly twelve rows and V3 retains one chain plus an optional complete sidecar; context_window_tokens stays caller-owned; connector construction will occur only inside the selected branch; diagnostics will expose the selected core family and label generic COGNITION_LLM shared_non_core
unexpected findings: the interrupted replacement is complete, while the older downstream V2 max-completion/thinking parsing block remains at the later configuration section and would still read the inactive family under V3. That exact duplicate block is the first resumed edit. The route report and connector are still on their eager V2 implementations, matching the planned remaining checkpoint scope
evaluation-integrity audit result: PASS for this recovery; activity was limited to goal/status recovery, contract/source/test/ICD inspection, skill reapplication, and this required evidence append
exit disposition: recovery complete; Gate 2 remains in progress and configuration/service-construction edits may resume at the verified duplicate-block boundary
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; worktree scope is unchanged apart from this required evidence append
```

### Gate 2 selected-engine configuration and services checkpoint evidence

```text
date/time: 2026-08-19T16:13:23Z
gate: Gate 2 - selected-engine configuration and services checkpoint (passed)
parent executor: /root, fixed parent-only execution constraint
starting boundary: Gate 1 and the canonical V2 helper checkpoint remained sealed; the immediately preceding compaction recovery reopened this exact configuration/service slice at HEAD 047bed9500111e44872b96c5445b6a64686f5803
implemented surface: LLMCallConfig now declares optional caller-owned context_window_tokens after existing defaults and provider construction omits it; config.py parses COGNITION_CORE_ENGINE before core routes and loads only the selected exact V2 or V3 immutable settings family; V3 validates chain/sidecar completeness, 8192 minimum lane caps, 50000 minimum chain context, thinking-off, closed 1/2/3/6 grouping, 30..600 deadline, and subconscious-sidecar dependency; CognitionChainServicesV3 validates exact lane bindings and normalized endpoint/model inequality; the connector constructs the selected core service object inside its branch while retaining the shared non-core COGNITION_LLM config consumed by L3; route diagnostics include only the selected core family, character carry-over, and all retained shared routes, with COGNITION_LLM labelled shared_non_core
source-test ownership: tests/ownership/source_test_impact_manifest.json now owns config.py, llm_interface/contracts.py, llm_interface/route_report.py, cognition_core_v3/contracts.py, and persona_supervisor2_cognition.py through the six exact plan nodes plus one supplemental closed-config node; the manifest remains valid JSON
collection and exact tests: all six mapped nodes collected exactly (6 collected in 0.79s). The final exact/supplemental run passed 7/7 in 5.84s. The V3 real-service import node separately passed both 50176 normal-only and 65000 extension-capable declared contexts in 3.63s; the inverse V2 service import passed with every V3 route absent
adjacent deterministic evidence: tests/test_config.py, tests/test_llm_interface_contracts.py, tests/test_llm_interface_route_report.py, tests/test_cognition_core_v2_stage_model_routing.py, tests/unit/cognition_core_v3/test_contracts.py, and tests/unit/nodes/test_persona_supervisor2_cognition.py produced 116 passes and two unchanged sealed-HEAD fixture failures. test_goal_branches_reuse_their_own_route_for_repairs_and_trace emits schema_version inside model-owned relational_willingness although HEAD already rejects that code-owned field; test_selection_producer_retry_reuses_goal_route_and_trace emits code-owned selected_response_operation fields although HEAD already exposes only writable fields. Neither call stack enters a changed configuration/service owner. The shared generic L3 binding and all other stage-route checks pass
syntax and style evidence: py_compile passed all five changed production Python modules and the CJK-bearing connector/test immediately after edits; git diff --check is clean. Mechanical import ordering was repaired in the three changed tests. Focused Ruff on the complete large modules continues to report inherited UP035/UP037/FLY002 in unchanged LLM/V3 contract lines and inherited import-order/SIM114/BLE001/PLR0402 findings outside this checkpoint's changed logic; no new focused route/config/test import finding remains
ICD/operator evidence: README.md, docs/HOWTO.md, llm_interface/README.md, and nodes/README.md now state engine-conditional routes, the caller-owned non-transported context declaration, the required shared generic route, selected-family diagnostics, normal/extended window settings, and branch-local service construction
active total context ceiling, completion reservation, and extension state: the sealed candidate declares 50176 and enables the 50000 total-token normal tier; the 65000 startup configuration is mechanically accepted when declared, while the sealed Gate 0 deployment remains normal-only and records extension unavailable
unexpected finding and disposition: persona_supervisor2_l3_surface imports the connector's shared _cognition_llm_config. The plan explicitly retains COGNITION_LLM for this non-core carry-over consumer, so the shared binding was restored in the listed connector owner while every one of the twelve V2 core configs remains branch-local. Both startup nodes then passed
evaluation-integrity audit result: PASS; tests, expected failures, and route evidence are reported unfiltered, no fixture or assertion was weakened, and no inactive-family alias/fallback was introduced
exit disposition: selected-engine configuration, V3 services construction, shared generic carry-over, and selected-family route diagnostics are accepted; Gate 2 remains in progress with lane, anchor/prompt, transcript, budget/estimator calibration, parser injection, and protected tracing still open
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; preserved owner lifecycle and Gate 1 files remain present; the new configuration/service, mapped test, manifest, and ICD diffs are confined to the approved Change Surface
```

### Gate 2 infrastructure-slice compaction-recovery evidence

```text
date/time: 2026-08-19T16:20:44Z
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: a conversation compaction occurred while the selected-engine configuration and services checkpoint evidence was being appended. The parent recovered the active goal and complete changed-path worktree, verified that the interrupted evidence patch landed completely, and re-read this plan's Summary; Scope And Change Direction; Mandatory Skills; Mandatory Rules; Must Do and Deferred boundaries; Confirmed Decisions for engine/public ownership, shared helpers, primary chain, grouping, goals, sidecar, recurrence, failure, budget, and evaluation integrity; target ownership, volatility head, step registry, completion caps, V3 services, LLM config, parser injection, transcript, protected trace, lane, estimator, budget, serving, and cache contracts; exact Gate 2 impact rows; Change Surface; Agent Autonomy Boundaries; Gate 2; Verification; Acceptance Criteria; Progress Checklist; current Execution Evidence; and Parent Execution Continuity. The parent re-read the governing architecture's overview, anchor volatility/byte stability/scoping, serial chain, deterministic interludes, lane and authorization isolation, sidecar/recurrence, failure ladder, interface, total-context ledger, re-anchor, cache invariants, routing disposition, and cutover sections; the plan registry; root startup/routing/testing guidance; V2 shared-substrate, current V3, connector, LLM-interface, and protected-tracing ICDs; the current V3 contracts/registry/transcript/execution boundary, V2 input and prompt-budget owners, canonical JSON parser and tests, trace facade/store and tests, frozen calibration corpus/manifest, source-test ownership mappings, and existing FIFO coordination pattern. The parent re-applied development-plan plus execution-gates/cutover references, local-llm-architecture, python-venv, test-style-and-execution, py-style plus both complete constraint references, cjk-safety, debug-llm, and no-prepost-user-input. The next checkpoint is a test-first infrastructure slice covering lane, prompt/anchor, transcript, budget/estimator, canonical repair injection, and protected chain transcript before their exact mapped implementation
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803; preserved owner lifecycle changes, sealed Gate 1 fixtures/artifacts, accepted public-helper changes, and accepted selected-engine configuration/service changes are present in the recovered status
owned files for the next checkpoint: src/kazusa_ai_chatbot/cognition_core_v3/lane.py; prompt.py; anchor.py; budget.py; transcript.py; src/kazusa_ai_chatbot/utils.py; src/kazusa_ai_chatbot/llm_tracing/chain_transcript.py; llm_tracing/__init__.py; src/scripts/calibrate_cognition_v3_token_estimator.py; their exact mapped tests; tests/ownership/source_test_impact_manifest.json; the V3, LLM tracing, and operator ICD rows; the frozen calibration corpus as read-only sealed input; and this evidence record
commands and read-only results: get_goal confirmed the exact active no-budget objective; git status recovered the complete changed-path boundary; branch and HEAD match the sealed baseline; venv/Scripts/python.exe exists; all newly listed Gate 2 infrastructure source/test paths remain absent except the superseded transcript.py and its old topology tests; current pre-append plan SHA-256 was 1fcd8e7a47ba1a3b2afb9d9c4cf8ce44a05c94f817818e2e82eb66561b4c208f
completed evidence retained: Gates 0 and 1 remain passed and sealed; the canonical V2 shared-helper checkpoint and selected-engine configuration/services checkpoint remain passed; the selected-engine evidence record is complete through its ending status and exit disposition
active total context ceiling, completion reservation, and extension state: the sealed loaded primary context remains 50176; the 50000 total-token normal ceiling is available with the owning completion reservation inside the total; the conditional 65000 tier remains unavailable in the sealed deployment
decisions mechanically applied from plan: the next slice uses one primary FIFO registry key of id(llm), normalized endpoint, and model; one distinct serialized sidecar registry; one stable system head followed by volatility-ordered dynamic input; alternating append-only messages with tail rollback and monotonic error appendices; the fixed CJK-aware estimator formula and frozen 48-plus-16 corpus; 50000 normal and conditional 65000 total ceilings; one shared re-anchor token; canonical parser injection only after deterministic failure; and protected off/metadata/full chain capture with ordinary surfaces remaining sanitized
unexpected findings: the current transcript owner and tests still implement the superseded per-chain cache-domain/checkpoint vocabulary, so this checkpoint replaces that contract directly. Existing llm_tracing stores arbitrary protected step documents and already owns capture-mode, TTL, and best-effort persistence mechanics, allowing the new chain transcript row to reuse that store without a new DB production path. The frozen calibration corpus contains messages only and retains no observed token results, so calibration output and its digest remain separate evidence as required
evaluation-integrity audit result: PASS for this recovery; activity was confined to goal/status recovery, complete contract/skill/source/test/ICD inspection, and this mandatory evidence append
exit disposition: recovery complete; Gate 2 remains in progress and the test-first infrastructure slice may begin
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; worktree scope is unchanged apart from this required evidence append
```

### Gate 2 lane-creation compaction-recovery evidence

```text
date/time: 2026-08-19T16:31:09Z
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: a conversation compaction occurred while apply_patch was creating src/kazusa_ai_chatbot/cognition_core_v3/lane.py. The parent recovered the active goal and full changed-path worktree before any further edit or test, then verified the exact interrupted source, test, and manifest state. The parent re-read this plan's Summary; Scope And Change Direction; Mandatory Skills; Mandatory Rules; lane, sidecar-admission, cancellation, deadline, diagnostics, and estimator contracts; exact lane Test Impact row; Change Surface; Agent Autonomy Boundaries; Gate 2; Verification; Acceptance Criteria; Progress Checklist; current Execution Evidence; Final Parent Code Audit; and Parent Execution Continuity. The parent re-read the governing architecture's overview, chain, deterministic-interlude, primary/sidecar lane, authorization, L1, loop-census, recurrence, repair, attempt, provider, serving-window, deadline, concurrency, interface, budget, re-anchor, and cache-affinity sections; development_plans/README.md; applicable root README/HOWTO routing and startup guidance; V2, V3, node, and LLM-interface ICD boundaries; the exact LLMCallConfig and URL-normalization owners; the complete recovered lane source and complete three-node lane test; the source-test manifest row; and searched existing FIFO/condition/coordinator owners before retaining this lane-specific implementation. The parent re-applied development-plan plus execution-gates/cutover references, local-llm-architecture, python-venv, test-style-and-execution, py-style plus both complete constraint references, cjk-safety, debug-llm, and no-prepost-user-input. The next checkpoint is a surgical queued-cleanup exception correction, syntax/manifest checks, exact collection, and execution of all three mapped lane nodes
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803; the recovered status contains the preserved owner lifecycle changes, sealed Gate 1 additions, accepted Gate 2 helper/configuration changes, new lane source/test, and its manifest mapping; no unrelated path was altered during recovery
owned files for this checkpoint: src/kazusa_ai_chatbot/cognition_core_v3/lane.py; tests/unit/cognition_core_v3/test_lane.py; tests/ownership/source_test_impact_manifest.json; and this plan evidence record
commands and read-only results: get_goal confirmed the exact active no-budget objective; git status recovered every changed path; branch and HEAD match the sealed baseline; venv/Scripts/python.exe exists; lane.py exists at 16740 bytes with SHA-256 2c0fdb3c97db46e2c46febe0d94f8fd90a33bd5c3cd5e984ed75dc5725c0aafe; test_lane.py has SHA-256 94642ae8368ac588e685885734dd62dc38fd5f19ca5ada442bddc98fb17d10d5; the pre-append plan SHA-256 is 70cb6d26c6cc25982ca1d3a1eb500a44a6c0d2329fb9fbe6733167f79b523b0d
completed evidence retained: Gates 0 and 1 remain passed and sealed; the canonical V2 helper and selected-engine configuration/services checkpoints remain accepted; the lane's exact test-first collection previously failed as expected with ImportError while the production module was absent; the interrupted patch landed completely rather than partially
active total context ceiling, completion reservation, and extension state: the sealed loaded primary context remains 50176; the 50000 total-token normal ceiling is available with completion reserved inside the total; the conditional 65000 tier remains unavailable in the sealed deployment
decisions mechanically applied from plan: the recovered implementation uses separate process-local primary and sidecar registries keyed by id(llm), normalized endpoint, and model; FIFO task tickets; complete-claim ownership; typed non-reentrancy and deadline failures; queued cancellation removal; maximum one sidecar request; fixed L1, repair, X1, and X2 admissions; and L1 cancel-and-drain before repair
unexpected findings: the interrupted source is complete but its queued-ticket cleanup catches BaseException, which violates py-style N-002 for application logic. The smallest compliant repair is to catch the explicit asyncio.CancelledError and LaneDeadlineError paths that can escape the wait boundary; no architecture or public-contract change is needed
test/artifact results: no Python command or test ran during this recovery. The retained pre-change red result is one collection error with no tests collected because lane.py did not yet exist
evaluation-integrity audit result: PASS for this recovery; work was limited to goal/status recovery, complete contract/skill/source/test/ICD inspection, hashing, and this mandatory evidence append; no result, fixture, prompt, model output, or acceptance threshold changed
exit disposition: recovery complete; Gate 2 remains in progress and the lane checkpoint may resume at the explicit exception-cleanup boundary
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; worktree scope is unchanged apart from this required evidence append
```

### Gate 2 lane-coordination checkpoint evidence

```text
date/time: 2026-08-19T16:42:56Z
gate: Gate 2 - primary and sidecar lane-coordination checkpoint (passed)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
starting boundary: the immediately preceding compaction recovery verified that the interrupted lane source landed completely and identified one broad queued-cleanup catch before any resumed test or edit
implemented surface: lane.py now owns separate process-local primary and sidecar registries keyed by id(llm), normalized base URL, and exact model while retaining the invoker to prevent id recycling; one FIFO non-reentrant condition/lock per key; complete-claim ownership; deadline checks before admission, during waiting, and after wakeup; queued-ticket removal; active-release cleanup; one serialized sidecar stream; exact L1/X1/X2/repair admission; canonical live-attempt coordinate binding through the shared V2 attempt ledger; one repair per live producer attempt and raw candidate; ordered X1 then X2 with the canonical three-attempt cap; L1 cancel-and-drain; and all eight required per-invocation sidecar diagnostics with task-deduplicated cancellation accounting
source-test ownership: tests/ownership/source_test_impact_manifest.json maps lane.py to the three exact Gate 2 nodes fixed by the plan; the manifest parses as valid JSON
test-first evidence: the original pre-source collection failed with ImportError as expected because lane.py was absent. After the recovered first implementation passed its initial three tests, parent audit expanded those same mapped nodes for deadline expiry, queued cancellation, live canonical-attempt binding, repair-after-exhaustion rejection, authorization order/caps, and exact diagnostic fields. The first expanded red command was terminated after the absent invocation_state API left held test tasks waiting; the harness was made exception-safe with one-second event bounds, and the repeated red run completed in 2.82s with all three nodes failing specifically on the absent invocation_state contract. No red result was suppressed or reclassified
collection and deterministic results: all three exact mapped nodes collect. The final exact run passed 3/3 in 0.71s: primary FIFO/separate sidecar/deadline cleanup; serialized sidecar plus live-attempt repair and authorization caps; and L1 preemption plus queued/active cancellation release
syntax and style evidence: venv Python py_compile passed lane.py and test_lane.py; focused Ruff passed both files with zero findings; git diff --check is clean apart from informational working-copy line-ending warnings on existing changed files
artifact hashes: lane.py SHA-256 4d9dee414135ad9d98a8fb333520f5e27890049d03f32734f1d8df1f27551b32; test_lane.py SHA-256 e3931da0b695d1475bd084ff90af1172cafc27d6a818fba968b1d1c67ab8010a; source-test manifest SHA-256 c9f6e83776e0bf7fb515b23d01c6d7e16d6bafb975207bb9b26a2c6d938b7386; pre-append plan SHA-256 63dc8e05be2db2b8a33b6db3ef5d9143d92e77ad50cadc987f52e9b5356ff070
decisions mechanically applied from plan: primary and sidecar identities remain distinct at services construction; sidecar serialization is cache-indifferent; no task can recursively claim either lane; deadlines and cancellation preserve FIFO; JSON repair consumes no semantic attempt and is admitted only against a currently started canonical producer attempt; a second repair cannot be created by changing candidate identity; X1/X2 cannot create a second branch-local producer for one cycle; and diagnostics count actual admitted calls rather than reservations
unexpected findings and dispositions: BaseException cleanup was narrowed to asyncio.CancelledError and LaneDeadlineError under py-style N-002. A post-wakeup deadline race was closed with a second deadline check inside queued cleanup. The id(llm) registry initially retained no invoker and could permit identity recycling after object collection; each coordinator now retains its owning invoker for the process-local registry lifetime. All three findings stayed inside the listed lane owner and fixed contract
evaluation-integrity audit result: PASS; deterministic failures and the terminated hanging red harness are reported explicitly, no fixture or assertion was weakened, and no model/live/DB operation occurred
exit disposition: lane coordination checkpoint accepted; Gate 2 remains in progress with prompt/anchor, transcript, budget/calibration, parser injection, and protected chain transcript still open
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; owner lifecycle changes, sealed baseline additions, accepted earlier Gate 2 changes, and the lane source/test/manifest/evidence additions remain preserved within the approved Change Surface
```

### Gate 2 prompt/anchor-contract compaction-recovery evidence

```text
date/time: 2026-08-19T16:58:26Z
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: a conversation compaction occurred during a read-only audit of the canonical V2 producer schemas after the first prompt/anchor implementation and exact deterministic tests had completed. The interrupted command performed no edit. Before any further edit or test, the parent recovered the active goal and worktree, then re-read this plan's Summary; Scope And Change Direction; Mandatory Skills; Mandatory Rules; Confirmed Decisions for the public boundary, primary chain, appraisal, goals, recurrence, failure, budget, and evaluation integrity; Target State ownership, volatility-ordered message head, registry, and caps; Runtime Or Resource Constraints; exact prompt/anchor Test Impact rows; Change Surface; Agent Autonomy Boundaries; Gate 2; Verification; Acceptance Criteria; Progress Checklist; current Execution Evidence; Final Parent Code Audit; and Parent Execution Continuity. The parent re-read development_plans/README.md; applicable README.md and docs/HOWTO.md architecture, route, prompt-budget, sleep-phase, startup, and testing guidance; governing-architecture sections 3, 4.1-4.3, 5.1, 6, 9.1, 10, and 11.1-11.3; the V2 and current-V3 ICDs; the complete recovered anchor.py, prompt.py, and their complete exact mapped tests; ownership-manifest rows; canonical V2 semantic-appraisal question/schema/validator, goal-output/required-selection/relational-willingness schema/validator, workspace partition, P1 planning envelope, and their direct-owner tests. The parent re-applied development-plan plus execution-gates/cutover references, local-llm-architecture, python-venv, test-style-and-execution, py-style plus both complete constraint references, cjk-safety, debug-llm, and no-prepost-user-input. The next checkpoint is test-first correction from caller-authored instruction/output contracts to a closed semantic contract-name registry whose minimal pointer is code-owned, with the complete reusable V2 schemas and vocabularies moved into the byte-stable engine manual
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803. git status recovered the preserved owner lifecycle changes, sealed Gate 1 artifacts, accepted Gate 2 helper/configuration/lane changes, and the four new prompt/anchor source/test files plus their ownership-manifest rows. README.md, docs/HOWTO.md, affected V2/V3/LLM/node ICDs, accepted shared/config source/tests, and Gate fixtures remain present as execution-owned changes; no prompt/anchor file is partial or missing
owned files for the resumed checkpoint: src/kazusa_ai_chatbot/cognition_core_v3/anchor.py; src/kazusa_ai_chatbot/cognition_core_v3/prompt.py; tests/unit/cognition_core_v3/test_anchor.py; tests/unit/cognition_core_v3/test_prompt.py; tests/ownership/source_test_impact_manifest.json; and this plan evidence record
commands and read-only results: get_goal confirmed the exact active no-budget objective; venv/Scripts/python.exe exists; the ownership manifest maps anchor.py to its exact one node and prompt.py to its exact two nodes; pre-append plan SHA-256 f926944e02726cc21f0b807832062573b33352f2b8c690027bb6691b3dc11901; anchor.py 22d24d25123e95be9ef5618efefe536f21d184744562eca53ef94bcf64f22996; prompt.py 84e09ead102f9ff5b4b299563b15e735db39f6dcdf930496512408b5525805a8; test_anchor.py 590e43c8c13ec23a4b2837768f5761e521916f41ed505aec6aced07cd22e8b2a; test_prompt.py f295e587a920941752fa6f202c1d1ecf7757bc1b17e899699f6eaa4028758930; source-test manifest 7baeeb548c18f98df576cf127ebfefa94de3d1e148c35396525fb34222cc2885
completed evidence retained: Gates 0 and 1 remain passed and sealed; canonical V2 helper, selected-engine configuration/services, and lane checkpoints remain accepted. The prompt/anchor test-first collection originally failed with two ImportErrors while the modules were absent. After the first implementation, the three exact mapped nodes passed 3/3 in 0.66s; immediate CJK py_compile checks passed after each source edit; focused Ruff passed all four files after __all__ ordering repair; and one canonical real-fixture render succeeded after the first-packet projection correctly held character_sleep_phase for the later goal question. The retained successful render had system_chars=1577, human_chars=3746, system sections engine_manual then character_identity, and the exact four dynamic sections then question. No live LLM or DB operation occurred
active total context ceiling, completion reservation, and extension state: the sealed loaded primary context remains 50176; the 50000 total-token normal ceiling is available with the owning completion reservation inside the total; the conditional 65000 tier remains unavailable in the sealed deployment
contract audit finding and disposition: the first implementation is coherent but not accepted. ChainQuestion currently permits an arbitrary caller instruction and output_contract, repeats reusable schema text in each dynamic user question, and leaves ENGINE_MANUAL as general guidance only. That violates the governing architecture's rule that the system manual owns all reusable output contracts/closed vocabularies while stage questions are one-to-three-line pointers plus current-run payload. The parent will first change the exact prompt tests to require a closed registered semantic contract name, a code-owned pointer, no dynamic output_contract, unknown-name rejection, and static manual presence for every registered contract; it will capture the resulting red failures before editing either CJK-bearing production file. Per-input handles, writable selected-operation fields, planned family domains, roster, affordances, and other current-run facts remain dynamic payload rather than being frozen into the manual
evaluation-integrity audit result: PASS for recovery and retained evidence. The structural integrity checks inspect field names rather than user prose, the system/pointer text contains no sealed case/fixture/node identifiers or test/rubric/expected-answer instructions, and the planned correction generalizes the production architecture instead of encoding a known quality answer
exit disposition: recovery complete; Gate 2 remains in progress; prompt/anchor acceptance is explicitly withheld until the registered-pointer/system-manual contract turns red then green under all three exact mapped nodes and a fresh runtime render/style audit
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; this record is the only recovery mutation and all pre-existing execution-owned changes remain preserved
```

### Gate 2 prompt-safe carrier compaction-recovery evidence

```text
date/time: 2026-08-19T17:16:21Z
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: /root, fixed parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: a conversation compaction occurred during the read-only audit that was locating the canonical prompt-safe episode projection. The interrupted command only read cognition_core_v2/facade.py and performed no edit or test. Before any further mutation, the parent recovered the goal and complete changed-path worktree; verified the prompt, anchor, mapped tests, manifest, and plan hashes/timestamps; re-read this plan's Summary, Scope And Change Direction, Mandatory Skills, Mandatory Rules, Confirmed Decisions 1-49, Target State, Contracts And Data Shapes, Runtime Or Resource Constraints, exact prompt/anchor Test Impact rows, Change Surface, Agent Autonomy Boundaries, Gate 2, Verification, Acceptance Criteria, Progress Checklist, current Execution Evidence, Final Parent Code Audit, and Parent Execution Continuity; re-read governing-architecture sections 3, 4.1-4.3, 5.1, 6, 9.1, 10, and 11.1-11.3; re-read development_plans/README.md, applicable root README/HOWTO architecture, route, prompt-budget, sleep-phase, startup, and testing guidance, and the V2, current-V3, nodes, and LLM-interface ICD boundaries; and re-read the complete recovered prompt.py, anchor.py, their exact mapped tests, ownership rows, canonical V2 state/relationship/scene/episode projections, facade carrier construction, cognition-episode model-visible projection, connector scene builder, and their relevant tests. The parent re-applied development-plan plus execution-gates/cutover references, local-llm-architecture, python-venv, test-style-and-execution, py-style plus both complete constraint references, cjk-safety, debug-llm, and no-prepost-user-input. The next checkpoint is a test-first exact first-packet carrier contract and recursive private-metadata rejection, followed by the smallest prompt-owner correction and fresh canonical render
starting commit and git status: branch feature/cognition_core_v3_cache_affine; HEAD 047bed9500111e44872b96c5445b6a64686f5803. The recovered status contains the preserved owner lifecycle changes, sealed Gate 1 artifacts, accepted Gate 2 helper/configuration/lane changes, and the four prompt/anchor source/test additions plus their manifest rows. No prompt/anchor file is partial or missing; the interrupted read left the worktree unchanged
owned files for the resumed checkpoint: src/kazusa_ai_chatbot/cognition_core_v3/anchor.py; src/kazusa_ai_chatbot/cognition_core_v3/prompt.py; tests/unit/cognition_core_v3/test_anchor.py; tests/unit/cognition_core_v3/test_prompt.py; tests/ownership/source_test_impact_manifest.json; and this plan evidence record
commands and read-only results: get_goal confirmed the exact active no-budget objective; venv/Scripts/python.exe exists; pre-append plan SHA-256 14d2ec755c6467c85759d557aef0d18f56b93c885f12fbef1760b5c621928217; anchor.py SHA-256 edc22a1c2abaf8fd32d9105921c38e560dffe9fca45b0a0e549653723a92cf13; prompt.py SHA-256 357b40df6bf96eab5a8a02958d7f03ed035264f986c37c314ef0a61f48658cdf; test_anchor.py SHA-256 8f107f0a997966ee3bf26fe4a1d3dbdfec354149bb3a0dcedd7c8288a7f1420c; test_prompt.py SHA-256 38b8589d9b651a0b2a66909a35d1602ff438ba626b5d3569408cf870be0034a3; source-test manifest SHA-256 7baeeb548c18f98df576cf127ebfefa94de3d1e148c35396525fb34222cc2885. git diff --check reported only informational LF-to-CRLF working-copy warnings
completed evidence retained: Gates 0 and 1 remain passed and sealed; canonical V2 helper, selected-engine configuration/services, and lane checkpoints remain accepted. The registered-contract prompt/manual correction is present. Its exact three mapped nodes passed 3/3 in 0.66s, immediate CJK py_compile checks and focused Ruff passed, and a canonical render produced system sections engine_manual then character_identity and human sections constraints_and_operational_state, relationship_and_mutable_state, episode_and_scene, evidence_and_affordances, then question. No live LLM or DB operation occurred
active total context ceiling, completion reservation, and extension state: the sealed loaded primary context remains 50176; the 50000 total-token normal ceiling is available with the owning completion reservation inside the total; the conditional 65000 tier remains unavailable in the sealed deployment
contract audit finding and disposition: prompt/anchor acceptance remains withheld. The successful render used raw relationship_context and raw episode carriers, while prompt.py validated only each outer section as a mapping. Those inputs contain durable relationship_id, source_id, platform/channel/message/user identifiers, origin metadata, and other deterministic provenance that must stay outside model input, and arbitrary inner keys could bypass the intended fixed section contract. Existing canonical owners resolve the boundary without a new architecture: project_state_for_prompt(...) supplies qualitative constraints, relationship, mutable state, evidence, character identity, and operational context; project_model_visible_percepts(...) strips deterministic percept source ids; SceneContextV2 is the prompt-safe semantic scene and will be copied without character_sleep_phase for the first appraisal packet, with participant_bindings normalized to its canonical empty list; and the existing stable prompt alias current_cognitive_episode supplies episode identity instead of the raw episode_id. The first packet will use exact section/interior key sets and reject raw ids/provenance field names recursively while preserving identical words inside legitimate user prose
evaluation-integrity audit result: PASS for recovery. The retained green result is reported but not accepted as the carrier checkpoint; the newly found privacy/shape defect is preserved, no assertion was weakened, no runtime prompt contains test metadata, and no model output, fixture, score, or threshold changed
exit disposition: recovery complete; Gate 2 remains in progress; the prompt-safe carrier contract may proceed test-first, and prompt/anchor acceptance stays open until red/green evidence, syntax/style checks, exact mapped collection/execution, and a real canonical-input render prove no raw episode or relationship carrier can enter the model packet
ending commit and git status: HEAD remains 047bed9500111e44872b96c5445b6a64686f5803; this record is the only recovery mutation and all pre-existing execution-owned changes remain preserved
```

### Gate 2 budget/transcript/parser checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent resumed after the prompt-safe carrier checkpoint and re-read the governing architecture transcript/budget/re-anchor sections, exact plan contracts, current V3 transcript/lane/contracts, canonical parser, manifest, and impacted tests. Next checkpoint is protected chain-transcript capture and token-estimator calibration/holdout, then exact Gate 2 mapped-node collection
starting commit and git status: HEAD c8b0881c; modified prompt.py, transcript.py, utils.py, test_prompt.py, test_transcript.py, test_utils.py, manifest; new budget.py and test_budget.py
owned files: src/kazusa_ai_chatbot/cognition_core_v3/budget.py; src/kazusa_ai_chatbot/cognition_core_v3/transcript.py; src/kazusa_ai_chatbot/utils.py; tests/unit/cognition_core_v3/test_budget.py; tests/unit/cognition_core_v3/test_transcript.py; tests/test_utils.py; tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: exact pytest nodes for prompt/anchor/budget/transcript and parser injection; py_compile; focused Ruff; manifest JSON validation
test/artifact results: prompt/anchor 3/3, transcript 5/5 including the new tail-rollback node, budget 1/1, full regular test_utils.py 17 passed, manifest contract 4/4; focused Ruff passed on the V3 files; py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged; no model invocation occurred
active total context ceiling, completion reservation, and extension state: unchanged from the sealed Gate 0/1 evidence
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice. All changes are deterministic structural contracts or parser plumbing; no runtime prompt, fixture answer, rubric, score, rerun, or model-output handling changed
local semantic reset ids and smallest-component evidence: none; no eligible real-model invocation occurred
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: chain-level append-only transcript with tail rollback, interlude queue, and one re-anchor; CJK-aware 50k/65k budget ledger with per-step completion reservation and serving-window defense; injected sidecar repair pair only after deterministic parse failure
unexpected findings: pre-existing broad-Ruff warnings in utils.py and CJK punctuation RUF001 findings are outside this slice; focused style checks used the project-specific files that are clean
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 2 closure remains pending
exit disposition: Gate 2 remains in progress
ending commit and git status: HEAD c8b0881c; modified/new files limited to the approved Change Surface
```

### Gate 2 protected tracing and calibration checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent resumed after the budget/transcript/parser checkpoint and implemented protected chain-transcript capture, calibration/holdout calculation, and overflow-probe dry-run. Next checkpoint is exact Gate 2 mapped-node collection and the remaining cold-chain/sidecar/recurrence gates
starting commit and git status: HEAD c8b0881c; new/modified files are limited to llm_tracing, V3 budget/transcript/parser, scripts, tests, and manifest within the approved Change Surface
owned files: src/kazusa_ai_chatbot/llm_tracing/chain_transcript.py; src/kazusa_ai_chatbot/llm_tracing/__init__.py; src/scripts/calibrate_cognition_v3_token_estimator.py; src/scripts/probe_cognition_v3_context_overflow.py; tests/test_llm_chain_transcript.py; tests/test_cognition_core_v3_calibration_scripts.py; tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: exact pytest nodes for chain transcript, calibration scripts, llm tracing, manifest contract, and prior V3 owners; py_compile; focused Ruff; manifest JSON validation
test/artifact results: chain-transcript capture 2/2, calibration scripts 2/2, existing llm tracing 18/18, manifest contract 4/4, prior V3 owners 10/10; combined 26 passed in the tracing/manifest batch
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged; no model invocation occurred
active total context ceiling, completion reservation, and extension state: unchanged from the sealed Gate 0/1 evidence
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice. Protected capture and calibration code are deterministic and contain no runtime prompt, fixture answer, rubric, score, rerun, or model-output handling
local semantic reset ids and smallest-component evidence: none; no eligible real-model invocation occurred
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: protected off/metadata/full chain-transcript capture; scoped trace facade; deterministic calibration with zero-underestimate holdout; effect-free overflow-probe dry run
unexpected findings: broad-Ruff BLE001 findings remain pre-existing in the tracing facade; the new capture function uses the project-specific bounded exception tuple
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 2 closure remains pending
exit disposition: Gate 2 remains in progress
ending commit and git status: HEAD c8b0881c; modified/new files limited to the approved Change Surface
```

### Gate 2 impact validation and calibration preflight evidence

```text
date/time: 2026-08-20
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent resolved the manifest source-root and facade-entry gaps, validated the ownership manifest programmatically, collected and ran the exact changed-impact nodes, and executed the deterministic calibration/overflow preflight. Next checkpoint is real-model calibration when the eligible environment is available, then Gate 3 cold-chain replacement
starting commit and git status: HEAD c8b0881c; changes remain limited to the approved Gate 2 Change Surface
owned files: tests/ownership/source_test_impact_manifest.json; scripts/validate_test_impact.py read-only; test_artifacts/cognition_core_v3/synthetic_calibration_observations.json ignored preflight artifact; this plan evidence record
commands: python -m scripts.validate_test_impact --base-ref HEAD; python -m scripts.validate_test_impact --base-ref HEAD --run; python -m scripts.calibrate_cognition_v3_token_estimator --observations test_artifacts/cognition_core_v3/synthetic_calibration_observations.json; python -m scripts.probe_cognition_v3_context_overflow --route-name COGNITION_V3_CHAIN_LLM --context-window-tokens 50000 --payload-char-count 260000
test/artifact results: manifest validation returned zero errors; changed-impact collection found 9 exact nodes and all 9 passed; synthetic calibration selected multiplier 1.0 with zero underestimates and accepted=true; overflow dry-run estimated 65048 tokens and reported payload_exceeds_declared_window=true
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged; synthetic observations are a deterministic preflight, not server-reported token evidence
active total context ceiling, completion reservation, and extension state: unchanged from the sealed Gate 0/1 evidence
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice. Preflight observations were generated deterministically and are not represented as live calibration evidence
local semantic reset ids and smallest-component evidence: none; no eligible real-model invocation occurred
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: exact source-test impact validation, changed-node collection/execution, deterministic calibration preflight, and effect-free overflow dry run
unexpected findings: --check-all exceeds the Windows command-line length limit; changed-node impact validation is the usable Gate 2 collection path until Gate 6 full collection is chunked
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 2 closure remains pending real-model calibration and remaining gates
exit disposition: Gate 2 remains in progress
ending commit and git status: HEAD c8b0881c; modified/new files limited to the approved Change Surface
```

### Gate 2 full V3 unit and impact-validator evidence

```text
date/time: 2026-08-20
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent fixed the ownership manifest source-root/facade gaps, ran the changed-path exact impact validator with execution, ran the full V3 unit suite, and executed the synthetic calibration/overflow preflight. Next checkpoint is real-model calibration and holdout when the eligible environment is available, then Gate 3
starting commit and git status: HEAD c8b0881c; changes remain within the approved Gate 2 Change Surface
owned files: tests/ownership/source_test_impact_manifest.json; test_artifacts/cognition_core_v3/synthetic_calibration_observations.json ignored preflight artifact; this plan evidence record
commands: python -m scripts.validate_test_impact --base-ref HEAD --run; python -m pytest tests/unit/cognition_core_v3 -q; python -m scripts.calibrate_cognition_v3_token_estimator --observations test_artifacts/cognition_core_v3/synthetic_calibration_observations.json; python -m scripts.probe_cognition_v3_context_overflow ...
test/artifact results: changed-path impact validation collected and ran 9 exact nodes (9 passed); full V3 unit suite passed 52/52; synthetic calibration selected 1.0 with accepted=true; overflow dry-run estimated 65048 tokens above the declared 50000 window
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged; synthetic observations are deterministic preflight evidence only
active total context ceiling, completion reservation, and extension state: unchanged from the sealed Gate 0/1 evidence
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; no runtime prompt, score, rerun, or model output was changed or hidden
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: exact changed-node impact execution, full V3 deterministic suite, deterministic calibration/overflow preflight
unexpected findings: --check-all exceeds the Windows command-line length limit; the changed-node path remains the usable Gate 2 verification surface until Gate 6 chunking
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 2 closure remains pending real-model calibration
exit disposition: Gate 2 remains in progress
ending commit and git status: HEAD c8b0881c; modified/new files limited to the approved Change Surface
```

### Gate 2 V2 shared-substrate regression evidence

```text
date/time: 2026-08-20
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent ran the complete V2 unit suite and classified the two known failures against the sealed Gate 1 baseline records. Next checkpoint is the real-model calibration/collection, then Gate 3
starting commit and git status: HEAD c8b0881c; no V2 production source was modified in this slice
owned files: this plan evidence record
commands: venv\Scripts\python -m pytest tests\unit\cognition_core_v2 -q
test/artifact results: 133 passed and 2 failed. The failures are the sealed Gate 1 baseline drift rows test_goal_progress_model_output_omits_protocol_metadata and test_calibration_artifact_requires_owner_specific_threshold; neither changed V2 production source was involved in the Gate 2 helper extraction
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; failures were not hidden or reclassified
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: full V2 unit suite regression check for shared-substrate extraction
unexpected findings: none new; the same two baseline drift rows remain visible
sealed parent audit pass and findings: bounded checkpoint accepted; Gate 2 remains in progress
exit disposition: Gate 2 remains in progress
ending commit and git status: HEAD c8b0881c; no V2 source change
```

### Gate 3 transition checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 - cold primary chain (not yet in progress)
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: the parent completed Gate 2 deterministic infrastructure verification and inspected the current registry/execution/facade topology. The superseded parallel-wave topology is still fully present and must be replaced as one coherent slice with the serial A1→A2→I1→G1a→G1b→I2→W1→P1 chain
starting commit and git status: HEAD c8b0881c; no Gate 3 production edit has started
owned files: this plan evidence record
commands: none state-changing; read-only topology inspection only
test/artifact results: no Gate 3 implementation or tests exist yet
readiness-probe sanitized fingerprint and health: none
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: not applicable to this read-only transition checkpoint
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: Gate 3 requires removing parallel-wave/isolated-goal/route-checkpoint behavior from source and tests in one coherent replacement
unexpected findings: the old facade remains coupled to the old registry/execution symbols, so the serial replacement cannot be split into isolated registry-only edits without leaving dual schemas
sealed parent audit pass and findings: bounded checkpoint accepted; Gate 2 remains open and Gate 3 has not begun
exit disposition: Gate 3 remains pending
ending commit and git status: HEAD c8b0881c; no Gate 3 source change
```

### Gate 2 collection and selector-regression evidence

```text
date/time: 2026-08-20
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent chunked the complete manifest node collection, removed one stale node, fixed the selector subprocess to use the V3 configuration helper, and reran the exact changed-node validator. Next checkpoint is real-model calibration and Gate 3
starting commit and git status: HEAD c8b0881c; changes remain limited to Gate 2 verification and test ownership
owned files: tests/ownership/source_test_impact_manifest.json; tests/unit/test_cognition_core_selector.py; test_artifacts/gate2_manifest_nodes.json ignored helper artifact; this plan evidence record
commands: chunked pytest --collect-only over 430 manifest nodes; pytest exact shared-owner batch; pytest selector regression; scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: all 430 manifest nodes collected after removing the stale semantic-terminalization node; changed-node impact validation 9/9 passed; shared-owner batch 112 passed and one selector regression fixed to pass; V2 and V3 unit suites already recorded separately
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; no semantic output, rerun, or hidden failure was introduced
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: complete manifest collection, stale node removal, exact selector subprocess config, changed-node execution
unexpected findings: --check-all exceeds the Windows command-line length limit, so the parent used chunked collection
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 2 closure remains pending real-model calibration
exit disposition: Gate 2 remains in progress
ending commit and git status: HEAD c8b0881c; changes limited to Gate 2 verification and test ownership
```

### Gate 2 full-suite collection classification

```text
date/time: 2026-08-20
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent attempted the default full suite and classified the collection error as a pre-existing live-LLM fixture defect outside the Gate 2 change boundary. Next checkpoint is real-model calibration and Gate 3
starting commit and git status: HEAD c8b0881c; no production boundary was changed by the failed collection attempt
owned files: this plan evidence record
commands: venv\Scripts\python -m pytest -q
test/artifact results: collection interrupted by NameError in tests/test_cognition_core_v2_transition_coherence_live_llm.py at line 73: _CAPTURED_ACCOMPLICE_INPUT is not defined. This file was not touched by the Gate 2 work and is a pre-existing live-LLM fixture defect. One identity-growth live cohort also skipped normally due to an unavailable replay manifest.
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; no result was hidden or changed
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: unrelated pre-existing collection failure is classified separately from changed-boundary failures
unexpected findings: default pytest collection is blocked by an unrelated live fixture before deterministic selection
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 2 closure remains pending
exit disposition: Gate 2 remains in progress
ending commit and git status: HEAD c8b0881c; no production boundary changed
```

### Chain-session registry checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 2/4 infrastructure - process-local chain-session registry
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent implemented the independent process-local ChainSessionV1 and registry, exhaustive immutable/cycle classification, prior-replacement state check, and next-admissible cycle index. Next checkpoint is L1 subconscious or the Gate 3 serial-chain replacement
starting commit and git status: HEAD c8b0881c; new session.py/test_session.py and manifest mapping only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/session.py; tests/unit/cognition_core_v3/test_session.py; tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_session.py -q; ruff; py_compile; scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: session tests 4/4 passed; exact changed-node impact validation 13/13 passed; focused Ruff and py_compile passed; manifest JSON valid
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic structural implementation only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: exact ChainSessionV1 fields, exhaustive immutable/cycle classification, prior validated replacement state, expected_cycle_index means next admissible input
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 2/3/4 closure remains pending
exit disposition: chain-session registry accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; new session source/test and manifest mapping within Change Surface
```

### Advisory L1 residue checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 4 infrastructure - advisory L1 residue
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent implemented the closed nonblocking L1ResidueV1 validator with bounded supplied-handle membership. Next checkpoint is the Gate 3 serial-chain replacement or remaining sidecar integration
starting commit and git status: HEAD c8b0881c; new subconscious.py/test_subconscious.py and manifest mapping only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/subconscious.py; tests/unit/cognition_core_v3/test_subconscious.py; tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_subconscious.py -q; ruff; py_compile; scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: L1 test 1/1 passed; exact changed-node impact validation 14/14 passed; focused Ruff and py_compile passed; manifest JSON valid
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic structural implementation only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: exact L1ResidueV1 fields and limits, closed risk flags, supplied-handle subset, duplicate-free lists, no invented authority
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 2/3/4 closure remains pending
exit disposition: advisory L1 residue accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; new subconscious source/test and manifest mapping within Change Surface
```

### Persisted chain-run checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 5 infrastructure - persisted sanitized chain-run owner
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent implemented the chain-run persistence owner, bootstrap indexes, and public DB exports. Next checkpoint is the remaining serial-chain replacement and service/console projection
starting commit and git status: HEAD c8b0881c; new db/cognition_chain_runs.py, test_db_cognition_chain_runs.py, db/__init__.py/bootstrap edits, and manifest mappings
owned files: src/kazusa_ai_chatbot/db/cognition_chain_runs.py; src/kazusa_ai_chatbot/db/__init__.py; src/kazusa_ai_chatbot/db/bootstrap.py; tests/test_db_cognition_chain_runs.py; tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: pytest tests/test_db_cognition_chain_runs.py -q; ruff on new files; py_compile; scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: chain-run tests 4/4 passed; exact changed-node impact validation 20/20 passed; focused Ruff and py_compile passed; manifest JSON valid
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic persistence implementation only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: idempotent chain_run_id upsert, immutable correlation conflict rejection, exact dual-key read, exact indexes and TTL, public DB facade exports
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 2/3/5 closure remains pending
exit disposition: persisted chain-run owner accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; new DB source/test and manifest mapping within Change Surface
```

### Serial topology registry checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - serial topology registry
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added the exact serial A1..P1 chain and frozen 1/2/3/6 appraisal grouping maps to the registry while retaining old symbols until the facade cutover. Next checkpoint is the serial executor and facade replacement
starting commit and git status: HEAD c8b0881c; registry.py/test_registry.py/manifest changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/registry.py; tests/unit/cognition_core_v3/test_registry.py; tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_registry.py -q; ruff; scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: registry tests 2/2 passed; exact changed-node impact validation 21/21 passed; focused Ruff and manifest JSON valid
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic structural implementation only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: frozen family order event_agency, goal_threat_outcome, epistemic_comparison_memory, relationship_social, moral_identity, existential_drive; exact 1/2/3/6 grouping maps
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: serial topology registry accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; registry serial additions within Change Surface
```

### Serial primary-chain executor checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - serial primary-chain executor
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added a serial-chain runner to execution.py that runs exact serial steps under shared attempt arithmetic while retaining the old wave symbols until the facade cutover. Next checkpoint is the serial facade replacement
starting commit and git status: HEAD c8b0881c; execution.py/test_execution.py/manifest changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/execution.py; tests/unit/cognition_core_v3/test_execution.py; tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_execution.py -q; ruff; py_compile; scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: execution tests 4/4 passed; exact changed-node impact validation 22/22 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic implementation only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: one serial primary chain, exact step identity, shared attempt ledger without reset
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: serial primary-chain executor accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; execution additions within Change Surface
```

### Serial primary-chain facade cutover pending checkpoint

```text
date/time: 2026-08-20
gate: Gate 3 - cold primary chain (in progress, cutover pending)
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added the serial registry and executor primitives and verified them independently. The old facade still imports the parallel-wave symbols, so the final serial facade replacement must be one coherent edit. Next checkpoint is the serial facade and old-path removal
starting commit and git status: HEAD c8b0881c; no facade source change in this slice
owned files: this plan evidence record
commands: read-only facade/producer inspection only
test/artifact results: no facade implementation change; current V3 unit suite remains green at 59/59
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: not applicable to read-only pending-checkpoint record
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: keep the facade cutover coherent rather than introducing a parallel V3 executor
unexpected findings: the old producer helpers are deeply coupled to the old transcript/wave topology and require replacement with the serial chain
sealed parent audit pass and findings: bounded checkpoint accepted; Gate 3 facade cutover remains pending
exit disposition: Gate 3 remains in progress
ending commit and git status: HEAD c8b0881c; no facade source change
```

### Cognition-chain event logging checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 5 infrastructure - cognition-chain event family
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added the bounded cognition-chain event family, recorder, sanitizer, schema, and facade export. Next checkpoint is the serial facade replacement or service/console projection
starting commit and git status: HEAD c8b0881c; event_logging model/recording/sanitization/schema/facade and test/manifest changes only
owned files: src/kazusa_ai_chatbot/event_logging/__init__.py; src/kazusa_ai_chatbot/event_logging/models.py; src/kazusa_ai_chatbot/event_logging/recording.py; src/kazusa_ai_chatbot/event_logging/sanitization.py; src/kazusa_ai_chatbot/event_logging/schemas.py; tests/test_cognition_chain_event_logging.py; tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: pytest tests/test_cognition_chain_event_logging.py -q; ruff; py_compile; scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: cognition-chain event tests 4/4 passed; exact changed-node impact validation 31/31 passed; focused Ruff and py_compile passed; manifest JSON valid
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic telemetry implementation only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: exact allowed cognition-chain fields, keyword-only best-effort recorder, no generic payload, secret/raw-content exclusion
unexpected findings: none within this slice
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 5 closure remains pending
exit disposition: cognition-chain event family accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; event-logging additions within Change Surface
```

### Brain-service chain-run projection checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 5 infrastructure - brain-service chain-run projection
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added paired optional live/self chain-run fields to the latest-graph response and exact dual-key DB resolution. Next checkpoint is the serial facade replacement or control-console projection
starting commit and git status: HEAD c8b0881c; brain_service/contracts.py, service.py, test_service_cognition_graph.py, and manifest changes only
owned files: src/kazusa_ai_chatbot/brain_service/contracts.py; src/kazusa_ai_chatbot/service.py; tests/test_service_cognition_graph.py; tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: pytest exact service graph nodes -q; py_compile; scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: new service graph nodes 2/2 passed; exact changed-node impact validation 38/38 passed; py_compile passed; manifest JSON valid
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic projection implementation only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: paired optional fields, exact run_id+llm_trace_id lookup, no global latest fallback
unexpected findings: service.py retains pre-existing broad-Ruff findings; the new projection code is py_compile-clean and covered by exact nodes
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 5 closure remains pending
exit disposition: brain-service chain-run projection accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; service projection additions within Change Surface
```

### Control-console chain-run projection checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 5 infrastructure - control-console chain-run projection
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added a strict optional CognitionChainRunSnapshot and paired client projection. Next checkpoint is the serial facade replacement or console rendering/browser validation
starting commit and git status: HEAD c8b0881c; control_console/contracts.py, kazusa_client.py, tests, and manifest changes only
owned files: src/control_console/contracts.py; src/control_console/kazusa_client.py; tests/test_control_console_contracts.py; tests/test_control_console_kazusa_client.py; tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: pytest exact console nodes -q; ruff; py_compile; scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: new console nodes 2/2 passed; exact changed-node impact validation 40/40 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic projection implementation only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: strict optional chain-run snapshot, exact paired live/self projection, no raw payload passthrough
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 5 closure remains pending browser/panel work
exit disposition: control-console chain-run projection accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; console contract/client additions within Change Surface
```

### Control-console selected-family route checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 5 infrastructure - control-console operator route family
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added V3 route descriptors and selected-family active marking while keeping generic COGNITION_LLM shared. Next checkpoint is the serial facade replacement or static panel/browser work
starting commit and git status: HEAD c8b0881c; brain_model_routes.py, test_control_console_web_surface.py, and manifest changes only
owned files: src/control_console/brain_model_routes.py; tests/test_control_console_web_surface.py; tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: pytest exact route node -q; ruff on new source; scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: route test 1/1 passed; exact changed-node impact validation 41/41 passed; focused Ruff on brain_model_routes passed with only pre-existing test-file inline-import warning
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic route projection only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: both core families editable, selected family active, generic cognition shared_non_core
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 5 closure remains pending panel/browser work
exit disposition: selected-family route projection accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; console route additions within Change Surface
```

### First-packet producer checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - canonical first-packet producer
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added build_first_packet_sections to prompt.py, using project_model_visible_percepts and canonical projected carriers. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/prompt.py; tests/unit/cognition_core_v3/test_prompt.py; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_prompt.py -q; ruff; scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: prompt tests 3/3 passed; exact changed-node impact validation 41/41 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic projection only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: canonical projection owners feed the validated first packet
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: first-packet producer accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py additions within Change Surface
```

### Serial-chain message assembler checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - serial-chain message assembler
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added build_initial_chain_messages pairing the static anchor with the validated first user packet. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/prompt.py; tests/unit/cognition_core_v3/test_prompt.py; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_prompt.py -q; ruff
test/artifact results: prompt tests 4/4 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic message assembly only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: one system head plus one first human packet
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: serial-chain message assembler accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py additions within Change Surface
```

### Grouped appraisal question-payload checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - grouped appraisal question payload
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added build_appraisal_question_payload for grouped A1/A2 questions. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/prompt.py; tests/unit/cognition_core_v3/test_prompt.py; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_prompt.py -q; ruff
test/artifact results: prompt tests 5/5 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic payload construction only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: grouped appraisal payload from planned family questions plus optional advisory L1
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: grouped appraisal question-payload builder accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py additions within Change Surface
```

### Deterministic I1 notice checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - deterministic I1 notice
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added a bounded deterministic I1 notice builder to the facade. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; facade.py/test_facade.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/facade.py; tests/unit/cognition_core_v3/test_facade.py; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_facade.py::test_i1_notice_is_bounded_and_deterministic -q; py_compile
test/artifact results: I1 notice test 1/1 passed; py_compile passed; broad Ruff on facade retains pre-existing import/type findings
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic notice construction only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: bounded 600-char deterministic I1 notice
unexpected findings: none within this slice
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: deterministic I1 notice accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; facade.py/test_facade.py additions within Change Surface
```

### Serial harness checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - serial harness
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added SerialChainHarness owning the append-only transcript, attempt ledger, and budget reference. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; execution.py/test_execution.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/execution.py; tests/unit/cognition_core_v3/test_execution.py; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_execution.py -q; ruff
test/artifact results: execution tests 5/5 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic harness behavior only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: one owned append-only transcript with rollback/interludes and shared attempt ledger
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: serial harness accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; execution.py/test_execution.py additions within Change Surface
```

### Serial harness step-runner checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - serial harness step runner
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added run_serial_harness_step reserving shared attempts and updating the append-only transcript. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; execution.py/test_execution.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/execution.py; tests/unit/cognition_core_v3/test_execution.py; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_execution.py -q; ruff
test/artifact results: execution tests 6/6 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic step runner only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: shared attempt reserve per serial step, transcript acceptance, fail-closed typed failure
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: serial harness step-runner accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; execution.py/test_execution.py additions within Change Surface
```

### Serial model-step invoker checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - serial model-step invoker
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added invoke_serial_model_step, sending the exact static head plus append-only transcript and parsing structured output. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; execution.py/test_execution.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/execution.py; tests/unit/cognition_core_v3/test_execution.py; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_execution.py -q; ruff
test/artifact results: execution tests 7/7 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic invoker behavior only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: exact static head plus append-only transcript, deterministic-only parse option
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: serial model-step invoker accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; execution.py/test_execution.py additions within Change Surface
```

### Serial initial-context builder checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - serial initial-context builder
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added _build_serial_initial_context, reusing the canonical V2 deterministic head, goal-cognition identity, and new prompt-safe first-packet producer. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; facade.py and test_deterministic_head.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/facade.py; tests/integration/cognition_core_v3/test_deterministic_head.py; this plan evidence record
commands: pytest exact serial-initial-context node -q; py_compile
test/artifact results: serial-initial-context test 1/1 passed; py_compile passed; broad Ruff on facade retains pre-existing findings
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic context construction only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: canonical projected state, goal-cognition identity, and prompt-safe percepts for the cold-chain head
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: serial initial-context builder accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; facade.py/test additions within Change Surface
```

### Grouped appraisal sequence builder checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - grouped appraisal sequence builder
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added build_grouped_appraisal_questions using the frozen registry grouping maps. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/prompt.py; tests/unit/cognition_core_v3/test_prompt.py; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_prompt.py -q; ruff
test/artifact results: prompt tests 6/6 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic question mapping only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: frozen 1/2/3/6 appraisal grouping maps
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: grouped appraisal sequence builder accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py additions within Change Surface
```

### Registered step payload-builder checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - registered step payload builders
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added goal, active-group, workspace, and action-plan payload builders. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/prompt.py; tests/unit/cognition_core_v3/test_prompt.py; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_prompt.py -q; ruff
test/artifact results: prompt tests 7/7 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic payload construction only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: code-owned payload constructors for every serial question owner
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: registered step payload builders accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py additions within Change Surface
```

### Serial question-sequence builder checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - serial question sequence
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added build_serial_question_sequence mapping the deterministic head into registered serial questions. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/prompt.py; tests/unit/cognition_core_v3/test_prompt.py; this plan evidence record
commands: pytest tests/unit/cognition_core_v3/test_prompt.py -q; ruff
test/artifact results: prompt tests 8/8 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic question ordering only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: A1/G1a/G1b/W1/P1 registered question order
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: serial question-sequence builder accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; prompt.py/test_prompt.py additions within Change Surface
```

### Serial question-sequence invoker checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - serial question-sequence invoker
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added invoke_serial_question_sequence driving registered questions through the primary lane. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; execution.py/test_execution.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/execution.py; tests/unit/cognition_core_v3/test_execution.py; this plan evidence record
commands: pytest exact serial-sequence node -q; ruff
test/artifact results: serial-sequence invoker test 1/1 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic invocation order only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: exact registered question order with shared attempt reservation
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: serial question-sequence invoker accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; execution.py/test_execution.py additions within Change Surface
```

### Serial step-payload wiring checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - serial step-payload wiring
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added _build_serial_step_payloads connecting the deterministic head to registered serial questions. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; facade.py/prompt.py/test_deterministic_head.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/facade.py; src/kazusa_ai_chatbot/cognition_core_v3/prompt.py; tests/integration/cognition_core_v3/test_deterministic_head.py; this plan evidence record
commands: pytest exact serial-payload node -q; py_compile
test/artifact results: serial step-payload test 1/1 passed; py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic payload wiring only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: current-run payloads are closed and private-metadata-free at the ChainQuestion boundary
unexpected findings: initial wiring used empty projected carriers; semantic enrichment remains in the facade orchestration checkpoint
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: serial step-payload wiring accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; facade/prompt/test additions within Change Surface
```

### Grouped appraisal validator checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 preparation - grouped appraisal validator
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. The parent added validate_grouped_appraisal_output enforcing exact family keys and micro-item shape. Next checkpoint remains the serial facade replacement
starting commit and git status: HEAD c8b0881c; appraisal.py/test_appraisal.py changes only
owned files: src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py; tests/unit/cognition_core_v3/test_appraisal.py; this plan evidence record
commands: pytest exact grouped-validator node -q; ruff
test/artifact results: grouped-validator test 1/1 passed; focused Ruff and py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic structural validation only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: exact planned-family coverage, question_id binding, proposition/delta object-or-null, max eight items
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 3 closure remains pending
exit disposition: grouped appraisal validator accepted for this checkpoint
ending commit and git status: HEAD c8b0881c; appraisal.py/test_appraisal.py additions within Change Surface
```

### Execution evidence template

Each gate appends:

```text
date/time:
gate:
parent executor:
goal objective and state:
compaction recovery event, reread sections/references, and next checkpoint:
starting commit and git status:
owned files:
commands:
test/artifact results:
readiness-probe sanitized fingerprint and health:
live environment fingerprint:
active total context ceiling, completion reservation, and extension state:
production-data extract ids, exact provenance hashes, and redactions:
evaluation-integrity audit result:
local semantic reset ids and smallest-component evidence:
semantic successes/72, inherited residual ids, and baseline-clean means:
decisions mechanically applied from plan:
unexpected findings:
sealed parent audit pass and findings:
exit disposition:
ending commit and git status:
```

### Gate 2 prompt-safe carrier checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 2 - shared contracts, lane, anchor, and budget (in progress)
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice; the parent resumed at the recorded prompt-safe carrier checkpoint and re-read the active goal, plan registry, root README/HOWTO, prompt/anchor source/tests, py-style constraints, test execution guidance, CJK safety, local-LLM boundaries, and no-prepost-user-input guidance before editing. Next checkpoint is fresh canonical-input rendering against the real V2 projection owners, then continuing the remaining Gate 2 transcript/budget/parser/tracing/calibration work
starting commit and git status: HEAD c8b0881c Partial implementation; changes are limited to src/kazusa_ai_chatbot/cognition_core_v3/prompt.py and tests/unit/cognition_core_v3/test_prompt.py
owned files: src/kazusa_ai_chatbot/cognition_core_v3/prompt.py; tests/unit/cognition_core_v3/test_prompt.py
commands: venv\Scripts\python -m pytest tests\unit\cognition_core_v3\test_prompt.py tests\unit\cognition_core_v3\test_anchor.py -q; venv\Scripts\python -m py_compile ...; venv\Scripts\python -m ruff check ...
test/artifact results: exact mapped prompt/anchor nodes passed 3/3; py_compile and Ruff passed; git diff --check reported only informational LF-to-CRLF working-copy warnings
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged; no model invocation occurred
active total context ceiling, completion reservation, and extension state: unchanged from the sealed Gate 0/1 evidence
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice. The change adds deterministic structural rejection only; it contains no runtime prompt text, fixture answer, rubric, score, rerun, or model-output handling
local semantic reset ids and smallest-component evidence: none; no eligible real-model invocation occurred
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: exact first-packet interior key sets for character constraints, operational context, relationship, visible percept rows, and evidence rows; recursive private-metadata rejection retained
unexpected findings: none beyond the existing informational line-ending warnings
sealed parent audit pass and findings: this checkpoint is bounded to the prompt-safe carrier contract and is not a full Gate 2 closure audit
exit disposition: prompt-safe carrier contract accepted for this checkpoint; Gate 2 remains in progress
ending commit and git status: HEAD c8b0881c; only the two prompt.py files are modified in this slice
```

## Historical Independent Plan Review

- **Status:** historical technical-contract pass; no blocker, major, or minor
  finding remained at the reviewed SHA. The later owner-directed execution-
  governance amendment is covered by the parent audit recorded below.
- **Initial review date:** 2026-08-19.
- **Reviewer:** `/root/independent_plan_review`, independent read-only
  `kazusa_plan_reviewer`; the reviewer authored none of this plan or its
  remediation.
- **Initial verdict:** fail.
- **Initial blocker/major findings and dispositions:**

  | Finding | Remediation in this revision |
  |---|---|
  | Dynamic scene/evidence occupied the system message. | System head is closed to byte-stable manual plus identity; the first user turn owns every dynamic field. |
  | Test Impact used shorthand paths and uncollectable coverage descriptions. | Per-path matrices now name semantic owner, exact pytest node, mode, and prevented regression; Gate 6 collects every node and rejects wildcard/unmapped paths. |
  | TTFT/prefill gates exceeded the non-streaming LLM response contract. | Performance evidence is limited to `perf_counter` wall times around the existing invoker; TTFT/prefill are explicitly out of scope and never inferred. |
  | Console/service could fall back to a globally latest V3 run. | DB and service require exact non-empty `run_id` plus `llm_trace_id`; missing/mismatched/read-failure states project `None` with no fallback. |
  | Resolver guardrail was V2-service typed and V3 attempt epochs were unsettled. | The guardrail uses a generic pass-through service type; V3 binds and projects the existing epoch-aware attempt ledger without reset across replay/recurrence/rebuild. |
  | Inactive engine route bundles remained eager startup requirements. | Engine-first loaders and branch-local services construction now define V2-without-V3 and V3-without-all-twelve-V2 startup contracts and exact tests. |
  | Persisted write/read identity and correlation semantics were incomplete. | Invocation-owned UUID, idempotent upsert, immutable-key conflict handling, correlation indexes, and exact dual-key reads are fixed. |
  | Live quality cases, dimensions, weights, and review resolution were movable. | Exact nodes/cases/groups, mechanical manifest values, closed dimensions/rubric, unweighted calculations, parent-only blinding, arithmetic-only audit, and tie handling are fixed before approval; adjudication is excluded. |
  | Resolver ICD was absent from the documentation surface. | `src/kazusa_ai_chatbot/cognition_resolver/README.md` and its exact documentation contract test are in scope. |

- **Second review date and verdict:** 2026-08-19, fail after confirming all nine
  initial findings closed.
- **Second-round findings and dispositions:**

  | Finding | Remediation in this revision |
  |---|---|
  | Recurrence did not exhaustively classify/digest every public input field. | A closed one-row-per-field table, canonical presence-sensitive UTF-8 JSON/SHA-256 encoding, exact evidence suffix and cycle-delta rules, fail-cold rule for future/unclassified fields, and mutation coverage for every field now govern admission. |
  | V3 startup omitted live shared consumers of generic `COGNITION_LLM_*`. | The generic route is explicitly retained as shared non-core for both engines, its consumers remain unchanged, route/console/docs contracts label it shared, and real-service subprocess imports prove V3 without the twelve V2 stage bundles and V2 without V3 routes. |
  | Sidecar overlap/caps/cancellation were not mechanically closed. | One FIFO multiplexed sidecar stream now has exact L1/X1/X2/repair admission caps, repair preemption, shielded synchronous-repair drain, cancellation ordering, instrumentation, and deterministic concurrency/failure nodes. |

- **Third review input and verdict:** SHA-256
  `E585F6ABC2191F87C224818FE6E2B76D65C64BCFCB4CF6213CE37D3464543508`,
  fail; shared-route and sidecar findings closed, recurrence partially open.
- **Third-round findings and dispositions:**

  | Finding | Remediation in this revision |
  |---|---|
  | `expected_cycle_index` alternated between current/next meanings and added an extra `+1`. | It now exclusively means the next admissible input: cold input `N` stores `N+1`, admission requires equality, success stores `incoming+1`, and exact `0→1→2`, repeated, skipped, decreasing, and out-of-order tests are owned. |
  | The digest froze `mutable_state` at its cold value despite the resolver consuming the prior replacement. | `mutable_state` is now a constrained cycle carrier equal only to the immediately prior validated `state_update.replacement_state`; the session compares canonical value plus digest, advances after valid terminal output, and has positive/negative unit and real-loop integration nodes. |

- **Final re-review:** PASS on 2026-08-19 over SHA-256
  `2431E632B2DF4482A391AD6234F461A47B0F230483893655DF726AB8AB285367`.
  The reviewer reported zero blocker, major, or minor findings and explicitly
  closed all initial, second-round, and third-round findings.
- **Post-review owner amendments:** after that reviewed SHA, the owner fixed
  execution ownership to the root parent, prohibited mid-flight questions and
  delegation, required an explicit plan-closure goal, required compaction
  recovery re-reads, authorized sanitized readiness probes, and replaced
  implementation-time independent review/handoff mechanics with separate
  sealed parent passes. The owner then clarified that 50,000 and 65,000 are
  total context ceilings rather than prompt-only allowances. The parent re-read
  the `development-plan` and `local-llm-architecture` skills, governing-
  architecture G7/§9.5/§10/§11.1, and every affected plan section, then audited
  both amendments. The total-ceiling correction changes the derived budget
  mechanics to restore the governing architecture while preserving source/test
  Change Surface, live cases, rubric, performance thresholds, gate order,
  permission boundaries, and cutover contract. The owner subsequently added
  the evaluation-integrity, two-consecutive semantic reset, inherited-V2 exact
  95.00% floor, component-first diagnosis, and production-data-copy rules. The
  parent audited that amendment against the development-plan, local-LLM,
  debug-LLM, database-data-pull, and test-execution contracts plus the governing
  architecture. The 95.00% allowance applies only to at most three presealed
  inherited model-semantic trials; hard boundaries retain zero tolerance. The
  That no-subagent rule was superseded by the owner's 2026-08-20 execution
  amendment. Historical evidence above remains factual; current execution uses
  the single reusable production implementation subagent defined in Execution
  Roles.
- **Residual gated preconditions:** explicit production-code authorization,
  goal creation, Gate 0 readiness, Gate 1 baseline sealing, and
  an eligible stable real-model environment for Gates 1 and 7. These remain
  gate conditions, not accepted review defects or waivers.
- **Approval rule:** owner approval was recorded on 2026-08-19. The owner's
  same-turn instruction withholds implementation authorization. The parent
  creates the closure goal and begins Gate 0 only after a later explicit
  implementation command; historical reviewer participation remains complete
  in the evidence above.

## Final Parent Code Audit

- **Status:** pending implementation and Gate 7 evidence.
- After Gate 7 and Gate 8 steps 1–3, the parent freezes the complete diff and
  evidence, performs the Compaction Recovery checklist, and re-reads the
  governing architecture, applicable skills, this complete plan, source-test
  impact output, deterministic/full tests, protected live reviews, blinded
  score sheets, performance calculations, overflow/calibration artifacts,
  console evidence, observation evidence, and cutover runbook.
- The parent audits in a distinct sealed pass for hidden compatibility paths,
  semantic contract drift, primary-lane interleave, unsafe sidecar behavior,
  prompt-data leakage, test/rubric/fixture metadata in runtime prompts, test-
  only production branches, outcome-conditioned reruns, suppressed failures,
  post-result inherited-failure reclassification, wrong 69/72 arithmetic,
  missing semantic-reset or production-data provenance, missing source
  ownership, skipped component-first/live gates, unverifiable performance
  claims, and documentation/rollback drift.
- Findings are recorded by blocker/major/minor severity. The parent remediates
  every finding, reruns affected exact mapped nodes and gates, refreezes the
  evidence, and performs a fresh audit pass. Gate 8 closure requires zero open
  findings; this checkpoint is parent-owned and carries no independence claim.

## Execution Continuity

The root parent retains orchestration across the complete flight. After approval
and explicit production authorization, it creates the exact closure goal in
Mandatory Rule 2 and starts Gate 0. This plan, the governing architecture, the
archived superseded plan, the sealed Gate 1 baseline, and Execution Evidence
remain the continuity package across turns. At every checkpoint the parent
updates the plan before advancing. After each compaction it completes and
records the Mandatory Rule 5 recovery re-read before further work. During the
execution flight it resolves in-scope decisions through the fixed authority
order, hands every production-code mutation to the same reusable implementation
subagent while available, reviews each returned checkpoint, preserves every
semantic failure as evidence, applies component-first diagnosis, and records
genuine blockers while retaining question-free execution. The exact 95.00%
floor permits bulk delivery before follow-up repair of accepted inherited
residuals; each residual is fixed or transferred to a new active bugfix draft
before archive closure. The parent marks the goal complete only after Gate 8
acceptance and completed-plan archive closure.

### 2026-08-20 execution-governance amendment

```text
date/time: 2026-08-20
gate: execution-governance amendment before continuation
parent executor: /root
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
starting commit and git status: dirty shared worktree containing the ongoing V3 implementation and evidence listed by git status
owned files: this plan only
decision: owner removed the parent-only implementation constraint, permitted subagents, and fixed exactly one reusable subagent as the sole production-code implementation executor
review boundary: root parent retains scope, handoffs, evidence, verification, review, cutover, and lifecycle closure; production-code edits and remediations are assigned to the same subagent whenever available
historical evidence disposition: prior parent-only records remain unchanged as factual execution history and no longer govern future checkpoints
exit disposition: amendment accepted; execution may continue under the Root execution owner and Production implementation owner role contracts
```

### 2026-08-20 production-executor model amendment

```text
date/time: 2026-08-20
gate: execution-governance amendment during Gate 3/4 review
parent executor: /root
owner decision: all production-code changes from this checkpoint forward use a gpt-5.6-luna subagent with maximum reasoning at normal speed
prior executor disposition: /root/v3_implementation completed its assigned Terra checkpoints and remains historical implementation evidence; it receives no further production-code handoff
fixed reusable executor rule: root creates one Luna production implementation subagent because the model change makes replacement necessary, then reuses that Luna subagent for subsequent production edits while available
review and lifecycle boundary: unchanged; root retains plan/evidence/verification/review/cutover/archive authority and independently accepts every returned production checkpoint
exit disposition: amendment accepted before the next production mutation
```

### 2026-08-20 V3 runtime-policy injection amendment

```text
date/time: 2026-08-20
gate: Gate 2 service-injection contract amendment
authorizer: user
root reviewer: /root
authorized change: extend CognitionChainServicesV3 with appraisal_group_count and turn_deadline_seconds, validate their closed domains, inject both values from the already selected CognitionV3RouteSettingsV1 connector branch, and consume them in the V3 facade without rereading route environment
architectural disposition: explicit dependency injection is accepted; hidden context-local policy and duplicate route loading are excluded
downstream contract: configured grouping replaces hard-coded group_count=2 in cold and recurrence appraisal; one invocation-wide monotonic deadline is checked at every Decision 41 admission boundary; Decision 36 session TTL uses the injected deadline term
production implementation owner: /root/v3_luna_implementation, the reusable gpt-5.6-luna max-reasoning normal-speed production subagent
exit disposition: amendment accepted; Gate 2 runtime propagation may proceed after the active bounded static-remediation slice
```

### Gate 5 control-console config-read scope amendment

```text
date/time: 2026-08-20
gate: Gate 5 partial - inactive cognition-route config read
parent executor: /root
reproduction: GET /api/services/brain/config returned 422 with `cognition_v3_chain_llm_model: value must not be empty` while V2 was selected; the browser E2E recorded the same 422 resource failure
scope finding: brain_model_routes.py registered both V2 and V3 families as required descriptor defaults, while the generic service-config owner had no representation for an intentionally unconfigured inactive route
plan amendment: add src/control_console/service_config.py and tests/test_control_console_service_config.py to Change Surface and map the exact deterministic owner node test_brain_config_allows_unconfigured_inactive_core_routes
fixed contract: both core families remain visible/editable, only the selected required family is required, optional/inactive routes expose an honest unconfigured state, and no placeholder model value, route alias, or runtime fallback is introduced
next checkpoint: one reusable production implementation subagent owns the bounded service-config/route fix and coupled tests; root reviews the diff and reruns the two reproduced failures
```

### Gate 5 control-console config-read checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 5 partial - inactive cognition-route config read accepted
root reviewer: /root
production implementation owner: /root/v3_implementation, the fixed reusable production-code subagent
owned files: src/control_console/brain_model_routes.py; src/control_console/service_config.py; tests/test_control_console_service_config.py; tests/test_control_console_web_surface.py; tests/ownership/source_test_impact_manifest.json
root cause and disposition: route descriptors treated both V2 and V3 core model families as simultaneously required; requiredness is now derived from the selected engine for config reads and route projection while shared routes remain required and optional sidecar routes remain optional
exact deterministic verification: `venv\Scripts\python -m pytest tests\test_control_console_service_config.py::test_brain_config_allows_unconfigured_inactive_core_routes tests\test_control_console_web_surface.py::test_model_routes_mark_only_selected_core_family_active_and_generic_cognition_shared tests\test_control_console_web_surface.py::test_character_identity_surface_and_pace_api_contract -q` -> 3 passed
exact browser E2E verification: `venv\Scripts\python -m pytest tests\control_console_e2e\test_cognition_graph_e2e.py::test_native_v2_failure_states_render_with_typed_evidence -q -s` -> 1 passed
impact verification: `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run` -> 63 exact mapped tests collected and passed
static artifact verification: changed Python sources/tests compiled; ownership manifest parsed as JSON; owned diff check passed
style review: Ruff reported only pre-existing findings in the touched modules and no new finding attributable to this checkpoint
live environment and evaluation-integrity disposition: no live LLM call occurred; the browser E2E used its patched deterministic service boundary
sealed root audit: accepted after one remediation returned to the same production subagent so projected route metadata uses selected-engine requiredness as well as config-field loading
exit disposition: the reproduced 422 failures are closed; Gate 5 remains open for its remaining API/event/transcript/service evidence
next checkpoint: reconcile Gate 4 recurrence status against the current facade/session implementation and add the missing bounded recurrence tail with the same production subagent
```

### Serial baseline recovery checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 recovery checkpoint
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: restored serialized V3 registry/execution/prompt/appraisal/contracts/facade and V3 fixtures after an accidental file truncation; next checkpoint is G1b/W1/recurrence parity
starting commit and git status: working tree dirty; V3 serial path restored and deterministic suites green
owned files: V3 serialized source and test files listed above; plan evidence record
commands: venv\Scripts\python -m pytest tests\unit\cognition_core_v3 -q; venv\Scripts\python -m pytest tests\integration\cognition_core_v3 -q
test/artifact results: 50 V3 unit tests passed; 10 V3 integration tests passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this recovery slice; deterministic serial path and test fixtures only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: serial registry, serial execution, V3 lane services, and serial prompt carriers
unexpected findings: accidental truncation required restoration; no semantic change was introduced
sealed parent audit pass and findings: recovery accepted; full Gate 3 closure remains pending
exit disposition: serial baseline recovered and green
ending commit and git status: working tree dirty; serial baseline recovered
```

### Serial G1b/W1 checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 3 - active-branch and workspace serial steps
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. Added G1b materialization and serial W1 partition; next checkpoint is resolver recurrence continuation
starting commit and git status: serial baseline recovered; facade.py modified in this slice
owned files: src/kazusa_ai_chatbot/cognition_core_v3/facade.py; this plan evidence record
commands: venv\Scripts\python -m pytest tests\unit\cognition_core_v3 -q; venv\Scripts\python -m pytest tests\integration\cognition_core_v3 -q; py_compile
test/artifact results: 50 V3 unit tests passed; 10 V3 integration tests passed; py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; deterministic G1b validation and serial W1 validation/materialization
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: active roster order, exact workspace partition handles, deterministic fallback
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; resolver recurrence remains open
exit disposition: G1b/W1 serial steps accepted for this checkpoint
ending commit and git status: facade.py modified; V3 serial baseline remains green
```

### Gate 6 impact-validation checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 6 partial - exact source-test impact validation
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. Updated ownership manifest to serial node names and ran impact validator; next checkpoint remains resolver recurrence tail and full deterministic suite
starting commit and git status: serial baseline recovered and V3 fixtures migrated
owned files: tests/ownership/source_test_impact_manifest.json; this plan evidence record
commands: venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run
test/artifact results: Validated 62 exact impact-test node(s)
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; manifest and exact node collection only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: exact mapped source-test nodes updated to current serial topology
unexpected findings: none
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 6 remains pending full regular suite
exit disposition: impact validation accepted for this checkpoint
ending commit and git status: manifest updated; serial baseline remains green
```

### Gate 6 full-suite attempt evidence

```text
date/time: 2026-08-20
gate: Gate 6 attempt - full regular suite
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. Attempted full pytest; next checkpoint remains classify full-suite pre-existing failures and run a bounded regular suite
starting commit and git status: serial baseline green before this attempt
owned files: this plan evidence record
commands: venv\Scripts\python -m pytest -q; venv\Scripts\python -m pytest -q --ignore=tests\test_cognition_core_v2_transition_coherence_live_llm.py
test/artifact results: first full pytest collection failed on pre-existing NameError `_CAPTURED_ACCOMPLICE_INPUT` in tests/test_cognition_core_v2_transition_coherence_live_llm.py; bounded full run excluding that file exceeded the 300s command timeout
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred in this checkpoint
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; no result suppression, pre-existing collection failure recorded separately
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: full regular suite attempt, not relying on focused suite as full-suite proof
unexpected findings: full pytest collection fails in an existing live-LLM module; ignoring that module still did not finish within 300s
sealed parent audit pass and findings: bounded checkpoint accepted; full regular-suite evidence remains incomplete
exit disposition: full regular suite not yet proven; no changed-boundary failure observed before timeout
ending commit and git status: unchanged
```

### Gate 4 session-persistence checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 4 partial - chain-session persistence integration
parent executor: sole root owner, parent-only execution constraint
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
compaction recovery event, reread sections/references, and next checkpoint: none in this slice. Added cold session creation, recurrence reattachment classification, and terminal session advance to _run_serial_cognition; next checkpoint remains short recurrence tail
starting commit and git status: serial baseline recovered; facade.py modified
owned files: src/kazusa_ai_chatbot/cognition_core_v3/facade.py; this plan evidence record
commands: venv\Scripts\python -m pytest tests\unit\cognition_core_v3 -q; venv\Scripts\python -m pytest tests\integration\cognition_core_v3 -q; py_compile
test/artifact results: 50 V3 unit tests passed; 10 V3 integration tests passed; py_compile passed
readiness-probe sanitized fingerprint and health: no live service, DB, LLM, or browser operation occurred
live environment fingerprint: unchanged
active total context ceiling, completion reservation, and extension state: unchanged
production-data extract ids, exact provenance hashes, and redactions: none required
evaluation-integrity audit result: PASS for this slice; process-local session carrier and diagnostics only
local semantic reset ids and smallest-component evidence: none
semantic successes/72, inherited residual ids, and baseline-clean means: pending Gate 7
decisions mechanically applied from plan: session key, cold store, digest reattachment, and terminal advance
unexpected findings: recurrence still cold-runs because the short re-decision tail is not implemented yet
sealed parent audit pass and findings: bounded checkpoint accepted; full Gate 4 remains pending recurrence tail
exit disposition: session persistence accepted for this checkpoint
ending commit and git status: facade.py modified; serial baseline remains green
```

### Gate 4 sidecar and recurrence internal checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 4 partial - V3-internal sidecar and recurrence accepted
root reviewer: /root
production implementation owner: /root/v3_implementation, the fixed reusable production-code subagent
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
owned implementation surface: src/kazusa_ai_chatbot/cognition_core_v3/action_selection.py; execution.py; facade.py; prompt.py; session.py; subconscious.py; utils.py; V3 unit/integration tests; tests/ownership/source_test_impact_manifest.json
implemented contract: complete cold and recurrence primary sequences retain FIFO ownership; L1 starts before primary admission and is observed without waiting at A1/G1a; one sidecar coordinator and admission ledger serialize L1, canonical JSON repair, X1, and X2; X1 finishes before X2; absent, failed, or malformed sidecar work denies authority without suppressing otherwise valid speech planning; exact session reattachment, prior-replacement consumption, live admitted-roster advancement, divergent/concurrent cold rebuild, and parent epoch arithmetic are covered
exact Gate 4 verification: tests/integration/cognition_core_v3/test_sidecar_failure.py::test_l1_repair_x1_x2_preemption_order_cancellation_and_failure_are_bounded plus the complete tests/integration/cognition_core_v3/test_resolver_recurrence.py module -> 6 passed
adjacent deterministic verification: tests/unit/cognition_core_v3 -> 50 passed; tests/integration/cognition_core_v3 -> 16 passed
impact verification: venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run -> 68 exact mapped tests collected and passed
static review: the changed Gate 4 functional surface compiles and executes; the broader V3 Ruff F/I sweep reports eleven inherited current-worktree findings, including the existing undefined ChainOutcome annotation in appraisal.py, which remain mandatory Gate 6 remediation and prevent a repository-static seal
evaluation-integrity disposition: deterministic and patched-LLM orchestration only; no live LLM call, semantic scoring change, fixture-conditioned production branch, or production-data write occurred
sealed root code audit: PASS for the V3-internal sidecar/session/recurrence slice. The root inspected lane admission, L1 cancellation/drain, lane-scoped parser injection, fresh authorization calls, exact boolean materialization, session claim/release, and primary ownership directly
exit disposition: this bounded internal checkpoint is accepted. Gate 4 remains open for substantive live/idle connector, resolver guardrail, external one-final-state-commit/effect-idempotency, required-selection, and group-self-cognition integration evidence. Gate 2 remains open for configured appraisal-group and turn-deadline propagation
next checkpoint: the same production subagent owns the missing exact Gate 3/4 acceptance nodes and external connector lifecycle proof within the plan-listed connector/guardrail surface
```

### Gate 3/4 connector and recurrence acceptance checkpoint evidence

```text
date/time: 2026-08-20
gates: Gate 3 deterministic contract surface complete; Gate 4 deterministic sidecar, recurrence, and connector lifecycle accepted
root reviewer: /root
production implementation owners: /root/v3_implementation for the historical Gate 3/4 implementation; /root/v3_luna_implementation for the subsequent bounded Python-style remediation under the production-executor model amendment
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
implemented contract: cold and recurrence paths execute the exact serial V3 chain; required-selection nested-role drafts reach the unchanged dialog boundary; sensitive ordinary-primary bids record ordered G1b and collapse after I2; targetless group turns can silence or emit a grounded reply proposal; live and idle connectors construct the selected engine family; recurrence preserves one action and resolver across parent epochs and performs one terminal external commit-boundary call
canonical helper boundary: V2 build_goal_output_contract is public and all V2/V3 callers use it; zero _build_goal_output_contract occurrences remain
exact acceptance verification: the five planned Gate 3 matrix nodes plus the recurrence and parent-retry nodes collected eight parametrized cases and passed
adjacent deterministic verification: tests/unit/cognition_core_v3 plus tests/integration/cognition_core_v3 -> 73 passed; tests/unit/cognition_resolver plus tests/unit/nodes -> 35 passed
impact verification: venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run -> 101 exact mapped tests collected and passed
static review: the root found and closed the private V2-helper import and the py-style P-005/N-006 wrapper-return findings; the remaining broader V3 Ruff findings stay assigned to Gate 6
evaluation-integrity disposition: deterministic and patched-LLM orchestration only; no live LLM call, semantic scoring change, fixture-conditioned production branch, or production-data write occurred
sealed root code audit: PASS for the completed Gate 3/4 deterministic and connector lifecycle surface
exit disposition: Gate 4 deterministic acceptance is complete, with formal checklist sealing ordered after Gate 2 and Gate 3 predecessor closure. Gate 3 remains dependent on Gate 2 configured appraisal-group and turn-deadline propagation
next checkpoint: complete the remaining Gate 5 control-console chain panels, exact documentation/observability nodes, and browser evidence with the reusable gpt-5.6-luna production subagent
```

### Gate 5 control-console chain-panel checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 5 partial - paired sanitized chain panels accepted
root reviewer: /root
production implementation owner: /root/v3_luna_implementation, the reusable gpt-5.6-luna max-reasoning normal-speed production subagent
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
implemented contract: bootstrap, Overview, and Debug expose strict bounded chain-run snapshots; one service read projects exact-correlated live/self graph and chain pairs; mismatch and absence fail closed to not_reported without a global or stale fallback; the static renderer allowlists and escapes every approved field; missing values including step_count render not_reported; mobile and desktop layouts remain bounded
focused deterministic verification: console contracts, client, web surface, and manifest contract -> 23 passed; Ruff F/I, JavaScript syntax, and git diff checks passed
exact Gate 5 browser E2E: tests/control_console_e2e/test_cognition_graph_e2e.py::test_v3_chain_panels_are_responsive_and_correlated -> 1 passed
browser environment: the in-app Browser runtime was selected first for http://127.0.0.1:8765, but no backend was connected and agent.browsers.list() returned empty after the prescribed recovery check; the plan-mapped repository Playwright E2E was used as the explicit fallback
visual evidence: v3_chain_panels_overview_correlated.png shows distinct mobile live/self rows with exact run and trace correlation and no overlap; v3_chain_panels_debug_absent_safe.png shows the desktop Debug absent-safe table with every missing field rendered not_reported; the root inspected both screenshots
redaction evidence: the renderer exposes only status, chain_run_id, run_id, llm_trace_id, cognition_invocation_id, chain_model_name, sidecar_model_name, terminal_disposition, started_at, completed_at, step_count, and warning_codes; the exact test rejects raw_prompt, raw_output, endpoint leakage, and cross-run substitution
sealed root code and visual audit: PASS for the control-console panel slice
exit disposition: this bounded checkpoint is accepted. Gate 5 remains open for the exact protected/sanitized cross-owner observability integration node and governed documentation contract
next checkpoint: add the missing Gate 5 observability and documentation acceptance nodes and align only the stale governed ICD sections with the same Luna worker
```

### Gate 5 observability and governed-document checkpoint evidence

```text
date/time: 2026-08-20
gate: Gate 5 deterministic observability surface accepted; formal sequential seal pending predecessor gates
root reviewer: /root
production implementation owner: /root/v3_luna_implementation, the reusable gpt-5.6-luna max-reasoning normal-speed production subagent
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
implemented contract: protected full-mode transcript content stays behind the protected trace boundary; the cognition_chain event, persisted cognition_chain_run.v1 row, brain-service live/self projection, and console projection contain bounded sanitized fields only; DB and service reads require exact run_id plus llm_trace_id; mismatched/global-latest substitution fails closed; trace, event, and DB write failures remain best effort and do not propagate into cognition-facing completion
exact Gate 5 integration: tests/integration/cognition_core_v3/test_chain_observability.py::test_protected_and_sanitized_records_share_exact_service_console_correlation -> 1 passed
adjacent deterministic verification: tracing, event, DB, service graph, console contract, and console client suites -> 30 passed including the exact integration node
impact/static verification: Ruff F/I passed for the new integration test; scripts.validate_test_impact --base-ref HEAD validated 102 exact nodes; git diff --check passed with line-ending notices only
governed documentation amendment: per the user's 2026-08-20 instruction, no documentation unit test exists or is referenced. The root manually audited every governed ICD against its source owner and the approved target contract; the impact matrix records parent Gate 5 ICD/source audit evidence instead of a prose unit test
manual ICD/source audit findings closed: removed the obsolete empty-role-assignment parity limitation; documented selected-engine service construction, serial primary/single-stream sidecar topology, recurrence and one terminal commit, V2 public helper reuse, context/timing ownership, exact DB/service/console correlation, protected versus sanitized observability, and operator cutover/rollback procedures
sealed root code and documentation audit: PASS for the protected/sanitized correlation, persistence, service pairing, console panel, and governed-document ownership slice
exit disposition: this Gate 5 slice is accepted. Gate 5 remains open for the plan-required cognition_engine_descriptor.v1 runtime-status field and its read-only console overview projection. The checklist remains ordered behind Gate 2, Gate 3, and Gate 4; configured appraisal grouping and turn-deadline claims become runtime-current only when Gate 2 propagation is closed
next checkpoint: propagate the authorized Gate 2 appraisal grouping and turn deadline, then implement the bounded Gate 5 engine-descriptor service/console slice before sequential sealing
```

### Gate 2 runtime-policy closure and sequential Gate 3/4 seal

```text
date/time: 2026-08-20
gates: Gate 2 accepted; Gate 3 accepted; Gate 4 accepted
root reviewer: /root
production implementation owner: /root/v3_luna_implementation, the reusable gpt-5.6-luna max-reasoning normal-speed production subagent
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
implemented amendment: CognitionChainServicesV3 owns validated appraisal_group_count and turn_deadline_seconds; the selected V3 connector injects both from its one already-loaded route-settings object; cold and recurrence appraisal grouping use the injected count; each public invocation creates one absolute monotonic deadline; FIFO primary/sidecar admission, semantic attempts, canonical JSON repair, L1, X1, and X2 all receive the same deadline and clamp provider timeout to the smaller of route timeout and remaining turn time; deterministic cleanup and validation remain available after model admission closes; chain-session expiry uses COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS * COGNITION_RESOLVER_MAX_CYCLES + services.turn_deadline_seconds + 30
production hashes: contracts.py 629E8E223B87851D4A2F5744770AECB534F0D8D5795BE20C259B751F631E5960; execution.py AF85A5C78D21BA6579D988D378D292C467D39EC851CA307A53661BFC379FDB09; facade.py E0DD783966209EE02204A677D4BE180D6542650B043729AA8A234F80DE801E6D; persona_supervisor2_cognition.py 5744916274730A198CB11CA34BBE27206180E28C4C7CE9EB2E19397A59AA45DE
subagent verification: 103 owned tests passed; 16 focused sidecar/recurrence tests passed; Ruff F/I, compileall, 111-node impact validation, and git diff --check passed
independent root verification: tests/unit/cognition_core_v3, tests/unit/nodes/test_persona_supervisor2_cognition.py, deterministic-head integration, and resolver-recurrence integration -> 70 passed; focused Ruff F/I and compileall passed; git diff --check passed with line-ending notices only
sealed root code audit: PASS. The root inspected public deadline construction, cold and recurrence propagation, every direct model and repair call, FIFO admission, timeout clamping, configured grouping, connector injection, transcript rehydration, and exact session TTL
sequential predecessor seal: the previously accepted Gate 3 cold-chain and Gate 4 sidecar/recurrence/connector evidence no longer have an open Gate 2 dependency. Gates 2, 3, and 4 are accepted in order
evaluation-integrity disposition: deterministic and patched-LLM verification only; no live LLM call, semantic scoring change, production-data read, or production-data write occurred
exit disposition: Gates 2, 3, and 4 accepted; Gate 5 remains open only for cognition_engine_descriptor.v1 and its read-only console overview projection
next checkpoint: implement and verify the bounded Gate 5 engine-descriptor service/console slice with the same Luna production subagent
```

### Gate 5 cognition-engine descriptor closure

```text
date/time: 2026-08-20
gate: Gate 5 accepted
root reviewer: /root
production implementation owner: /root/v3_luna_implementation, the reusable gpt-5.6-luna max-reasoning normal-speed production subagent
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
implemented contract: config.py exposes the already-loaded selected cognition route settings without environment reread or engine query and defensively copies the mutable V2 mapping; OpsRuntimeStatusResponse adds the strict optional cognition_engine_descriptor.v1 field; the service boundary renders deterministic selected V2/V3 model and chain-policy values without endpoint, API key, credential, or secret; V2 uses its stable sorted unique model set and zeroed chain-only fields; V3 uses exact selected chain/sidecar/group/context/budget/deadline values; the console allowlists exact fields, ignores additive future fields, fails unknown or incomplete schemas to not_reported, escapes all rendered values, and exposes one read-only Overview configuration card without switching controls
contract remediation: the root audit rejected broad numeric-only validation. CognitionEngineDescriptorResponse now closes engine_id to v2/v3, rejects booleans for numeric fields, closes grouping to 0/1/2/3/6, requires all V2 disabled/zero invariants, and enforces V3 budget, context, deadline, sidecar/model coupling, and subconscious-sidecar dependency
subagent verification: 92 owned tests passed; exact responsive browser E2E passed; 121 impact nodes passed; Ruff F/I, compileall, node --check, and git diff --check passed
independent root deterministic verification: config, runtime status, console contract/client/web, and exact cross-owner observability suites -> 98 passed with one Starlette/httpx deprecation warning; exact browser E2E -> 1 passed; expected best-effort tracing/event backend warnings remained non-propagating
root static verification: Ruff F/I passed; compileall passed; node --check passed; git diff --check passed with working-copy line-ending notices only; repository search found no descriptor projection of API key, endpoint, credential, or secret fields and no runtime engine switch
browser environment and visual audit: the in-app Browser backend remained unavailable from the earlier Gate 5 recovery check, so the approved repository Playwright fallback was rerun with external temp isolation. The root inspected v3_chain_panels_overview_correlated.png and v3_chain_panels_debug_absent_safe.png; the mobile Overview shows distinct live/self/engine cards with the exact configured values and no horizontal overflow, while the desktop Debug panel renders every absent chain field as not_reported
environmental test note: one initial root rerun placed pytest basetemp under the repository, so three subprocess config tests legitimately discovered the repository dotenv and failed their empty-environment assertions. The same tests passed under normal external temp isolation; this harness-geometry result is retained and is not a product failure
production hashes: config.py 553ABB9892DE311FED4F4BF27378BCD4984A042075C69BDEF6E54944B8230C3E; brain_service/contracts.py 3F3F1B40FB7B7EE84DA3905F0B601198CD1BBDD38CF6EE661B2D1105ED5E0464; service.py E511C93457F2D1B21C2CFE943353810A34091BB475AAB7510B7C0AFD530CFB6B; control_console/app.py 720CA37511C2BFF46DF4852F4010DE347D0723CE04BCD9B88AEE8CA3360416C9; static/index.html 133A8CF30805C5E192D090EEED1A8E9B4EB0E0E4E19ED829D008ADD13DFC3543; static/console.js FDC0A1D252CC349329E3C5684C1DADB5E03A3E08FC4C90A065B8BD8BC5AB2540
sealed root code and visual audit: PASS. All Gate 5 observability, persistence, service, documentation, console, redaction, absent-state, and responsive-layout requirements are accepted
exit disposition: Gate 5 accepted; proceed to Gate 6 full deterministic, impact, and static verification
```

### Gate 6 deterministic, impact, static, and regular-suite closure

```text
date/time: 2026-08-20
gate: Gate 6 accepted
root reviewer: /root
production implementation owner: /root/v3_luna_implementation, reused for the bounded V2 import-order cleanup under the owner's gpt-5.6-luna max-reasoning normal-speed rule
goal objective and state: Close cognition_v3_hybrid_agentic_loop_reconciliation_plan.md through all Gates 0-8, cutover evidence, final audit, and archive completion; active; no token budget
fixed verification sequence: V2 unit -> 133 passed with the two Gate 1 baseline failures retained; V3 unit -> 50 passed; V3 integration -> 24 passed; resolver/node -> 35 passed; shared parsing/config/selector -> 107 passed; persistence/event/service graph -> 22 passed; console -> 33 passed with one Starlette/httpx deprecation warning
impact verification: scripts.validate_test_impact --base-ref 047bed9500111e44872b96c5445b6a64686f5803 --run collected and passed all 165 exact mapped nodes
collection disposition: unmodified full collection still fails on the pre-existing NameError _CAPTURED_ACCOMPLICE_INPUT in tests/test_cognition_core_v2_transition_coherence_live_llm.py; explicit exclusion yields clean collection of 4,825 selected regular tests from 5,845 collected with 1,020 live-mode deselections
regular-suite disposition: the complete unit/integration/control-console-E2E segment passed 287 tests with 3 skips and only the two sealed V2 failures deselected; bounded top-level segmentation produced 4,386 additional passes. Unchanged failures were retained and classified by exact owner family: action-spec fixtures missing surface_role; stale V2 intention, continuation, scene-context, conversation-progress, task-resolution, and reproduction fixtures; one missing captured diagnostic export; one migration-script clock audit; and one unchanged baseline owner-matrix omission. No changed cognition V3 boundary failed
environmental stall: tests/test_service_background_consolidation.py::test_chat_queues_background_consolidation_for_mapping_state times out alone. The external artifact C:\Temp\kazusa_gate6_bg_stall.log records an idle Windows asyncio I/O loop with Mongo monitor threads active and no Python frame executing changed cognition logic. This reproduces the earlier pre-existing full-suite timeout; changed service graph/status/input-queue coverage passed 115 focused tests outside that stalled legacy module
changed-boundary remediation: the newly added COGNITION_V3_CHAIN_LLM and COGNITION_V3_SIDECAR_LLM routes were missing from two console test catalogs. The tests now assert 28 routes, 89 derived fields, and the exact two-route Cognition Core V3 group; 5 exact console nodes pass. No documentation unit test was created or modified for this plan
static remediation: the reused Luna production agent applied import-only Ruff ordering fixes in cognition_core_v2/action_selection.py and workspace.py. Final changed-file Ruff F/I, compileall, node --check, git diff --check, and stale V3 topology search pass
final hashes: cognition_core_v2/action_selection.py 07BBEE660448D29D6F003DD019DF19E47CE78143E9602869656C4A6501A3E8C7; cognition_core_v2/workspace.py D7149CBED945739C7374860957E1498E60C0BFCB3FDCED447E5B2DEB115542C4; test_control_console_brain_model_routes.py E8F90B34579F1CFA176A6B48393869AD4BFE8610FA95CF63D079F4FF0C13E28A; test_control_console_config_routes.py 2DA322BC3F8C5FD2353B13A9FC546B2ACFF861480CF1EBF9B9D265B1CE206EAE
evaluation-integrity disposition: deterministic, patched-LLM, and browser-harness verification only; no live semantic score, production-data read, production-data write, hidden production branch, or result suppression occurred. Every baseline failure and environmental stall remains visible
sealed root audit: PASS for every changed production boundary and its mapped deterministic evidence; impact coverage is 100%; no undocumented production source change remains
exit disposition: Gate 6 accepted; Gate 7 remains unopened pending a bounded test-contract amendment because the sealed 24-case live module invokes only the V2 baseline runner and the five-node performance module named by the architecture manifest does not exist
next checkpoint: obtain owner approval for the exact Gate 7 test-contract amendment, then run live nodes one at a time with artifact inspection
```

### Gate 7 additive test-contract amendment approval

```text
date/time: 2026-08-20
owner decision: approved
amendment scope: add tests/test_cognition_core_v3_candidate_live_llm.py for the exact 24 V3 candidate cases and create the already-declared tests/test_cognition_core_v3_performance_live_llm.py for the five fixed performance nodes
sealed boundary: tests/fixtures/cognition_core_v3_architecture_manifest.json, tests/fixtures/cognition_core_v3_live_case_manifest.json, tests/cognition_core_v3_comparison_harness.py, and tests/test_cognition_core_v3_live_llm.py remain byte-identical to their Gate 1 hashes
ownership exception: the candidate live module is additive and Gate 7-only; it reuses the immutable case rows and effect-free trial contract without changing scoring, rerun, denominator, or hard-gate rules
executor constraint: production remediation, if live evidence exposes a production defect, remains owned by the single reusable gpt-5.6-luna max-reasoning normal-speed subagent; test harness and parent-authored review artifacts remain root-owned
documentation-test disposition: no documentation unit test may be created; any such test discovered during this gate is removed
entry disposition: Gate 7 opened for harness implementation, collection sealing, readiness probes, one-node-at-a-time live execution, artifact inspection, blinded scoring, performance evidence, and parent calculation audit
```

### Gate 7 batch-first live-execution amendment

The owner changed Gate 7 execution order on 2026-08-21. The parent must first
run the complete fixed Gate 7 real-LLM inventory without interleaving further
production remediation: the already-sealed V2 controls, all fixed V3 candidate
trials, the five performance nodes, and the serving/long-context evidence.
Every node still runs one at a time, every raw result remains retained, and no
valid semantic result is rerun or suppressed. After the full inventory is
frozen, the parent performs one failure-mode collection pass, groups findings
by semantic or deterministic owner, and then remediates those owner groups one
by one through the applicable component-first ladder before any verification
rerun.

This amendment changes scheduling only. It preserves the sealed manifests,
three-trial denominator, blinding, score rubric, hard gates, exact 69/72 floor,
performance thresholds, serving requirements, countercase requirement,
production-code ownership, and evaluation-integrity rules. The earlier
two-failure stop in Gate 7 is superseded only for completing this batch-first
initial evidence flight; every triggered `local_semantic_reset.v1` stays open
and must close before Gate 7 acceptance. The interrupted dynamic appraisal-
domain prompt correction was removed before the batch began, leaving the last
reviewed first-packet remediation as the candidate baseline. Documentation
prose remains audited by the parent and has no documentation unit test.

### Gate 7 input-flow architecture amendment — 2026-08-21

The owner directed Gate 7 to preserve every function-level cognition feature
while reconsidering the V2 model-call shape before further prompt iteration.
This amendment supersedes the earlier appraisal Decisions 12–15 and the Gate 7
2→3→6 selection step. It retains the A1, A2, I1, G1a, G1b, I2, W1, P1, X1,
X2, R, and O owners; all six appraisal families; every V2 proposition and
delta domain; per-family capacity; the 21-emotion derivation; all 11
relationship axes; goals, bids, required selection, workspace collapse,
planning, authorization, resolver recurrence, state validation, and external
V2 input/output contracts.

The retained live evidence establishes the boundary:

- The sealed legacy-shaped A1 reproduction `exact.json` used two provider
  calls, 6,373 then 6,464 provider prompt tokens, and about 36 seconds. Both
  responses assigned the reporting `current_user` as actor of a reported
  third-party event and used an out-of-family epistemic proposition kind;
  bounded repair repeated the ownership error and exhausted structurally.
- The answer-leakage-audited architecture probe used the same frozen case,
  route, and model with one evidence occurrence, typed conversation roles,
  candidate handles, local family domains, and no character/relationship,
  capability, resolver, or duplicate-scene carrier. It used one provider call,
  1,255 provider prompt tokens, and 5,298 ms. Its semantic direction improved:
  both propositions used candidate event `ce1`, with no invented
  `current_user` actor assignment.
- The probe still failed structurally because the inherited nullable pair led
  the model to emit scalar `delta: 1` beside a proposition. The sealed probe
  artifact is
  `test_artifacts/cognition_core_v3/cogv3-g1-047bed95-331653f8/local_semantic_resets/v3_appraisal_first_packet_carrier/architecture_probe.json`;
  file SHA-256
  `b2a0dd80f62315dd4896bc22fcc06d466f670e428ad6b94baaf8ccc8573ad159`.
  Its deterministic harness module passed 21/21 tests before the live call.

The resulting contract decision is architectural rather than case-specific
prompt tuning:

1. A1 and A2 are fixed semantic stages. The runtime grouping control and
   1/2/3/6 topology registry leave the canonical runtime.
2. Each planned family returns exact independent `propositions` and `deltas`
   arrays, each retaining the V2 eight-row maximum and V2 field vocabulary.
   This removes only the model-facing micro-loop ceremony: repeated
   `question_id`, nullable proposition/delta pairs, null-null sentinels, and
   exact-repeat termination.
3. One grouped attempt consumes attempt one for its listed families. A
   structural failure rolls back and routes directly to one singleton recovery
   attempt per affected family. This preserves family isolation and the V2
   two-attempt cap without a same-group repair or topology ladder.
4. The system anchor retains character identity and universal semantic/JSON
   rules. Exact stage contracts and local domains sit adjacent to the current
   question.
5. Dynamic input follows first-consumer ownership: observation evidence and
   typed conversation frame at A1; character/relationship context at A2;
   reduced qualitative state and goal carriers at G1; bid index at W1;
   capabilities and resolver availability at P1; new observation deltas at R.
   Each semantic evidence text appears once in the transcript.
6. Deterministic code remains the only owner of parsing, validation, attempt
   accounting, state reduction, emotions, relationship mutation, permissions,
   execution, persistence, and delivery. It maps accepted family arrays into
   the canonical V2 aggregate and invokes the existing V2 validators/reducers.

Implementation proceeds in bounded checkpoints through the same reusable
gpt-5.6-luna maximum-reasoning production worker: first appraisal product
geometry and recovery; then first-consumer carrier projection and compact
stage-independent anchor; then obsolete topology/configuration and dead-layer
excision. Each checkpoint requires its exact unit/integration nodes, focused
Ruff, compilation, diff check, and parent code audit. A smallest-component
live verification follows deterministic acceptance; the full Gate 7 rerun
remains governed by the sealed rerun and blinding rules.

### Gate 7 critical-only acceptance and retained-evidence amendment — 2026-08-22

The owner replaced the semantic pass rule and the post-remediation execution
requirement after reviewing the complete failure census. A semantic trial now
fails Gate 7 only for role reversal, material internal self-conflict, or a
boundary/safety conflict. Other rubric weaknesses remain visible diagnostic
warnings. Structural schema, provenance, privacy, permission, authorization,
effect, persistence, and delivery failures retain zero tolerance.

The owner also directed the parent to retain prior passing tests and accept a
remediated failure owner from its narrow post-fix verification without
repeating the full passing inventory. This is an explicit evidence-policy
override, not a mutation of the sealed Gate 1 rubric or prior blinded reviews.
The original detailed scores, raw artifacts, mappings, and arithmetic audit
remain immutable and diagnostic.

The sealed relaxed batch result before the final role fix was `65/72` V3
critical passes. Its seven failures were four role reversals and the three
`future_speak_authority` self-conflicts. The final structural correction places
the typed five-field dialogue-role binding beside its matching current-event
evidence, carries the same binding into goal stages, excludes memory/tool
evidence from inheritance, removes private source identity from the prompt,
and fails unknown/private carrier fields closed. It does not parse pronouns or
rewrite model semantics deterministically.

The two owner-selected narrow live checks passed after that source change:

- `existential_drive:1` accepted `self` as subject and experiencer and
  persisted an empty event `role_refs`; artifact SHA-256
  `7b68815bd3a5dfae07ff6870a7c24293c9453660653c9b0816af6c9526cc4e40`.
- `event_agency_and_moral_chain:3` invented no self/current-user event role and
  persisted an empty event `role_refs`; artifact SHA-256
  `b638b925faeeabd02ad8a49d8d8d6f1a14cb28c459686fa1013f6279789636bd`.

Both trials were eligible, input/output-validator clean, input-unchanged, and
prefix-exact. Deterministic V3 unit/integration verification passed `181`
tests. The correction therefore substitutes passes for the four trials in the
single role-reversal owner group. The effective accepted result is `69/72`
(`95.833%`). The three `future_speak_authority` trials remain explicitly
reported `self_conflict` failures and consume the full allowed residual count;
they are not relabeled as passes.

The input-flow optimization preserved all cognition stages, six appraisal
families, emotion/relationship axes, goals, selection, planning,
authorization, resolver, and persistence features. The 21-emotion reducer
still persists `emotion_id`, `primary_root`, `root_refs`, and `cause_status`
for active affect and exposes `cause_summary`; emotion-plus-cause is therefore
preserved in V3.

For retained performance evidence, the pre-compaction cold run already passed
eligibility, full median/p95 ratios, primary median reduction, prefix
exactness, single-primary ownership, and foreign-interleave checks. Its sole
failure was the first-primary ratio (`1.9936519198167852`). The compact
first-consumer design reduced the observed first request from `16,294`
characters/`6,151` provider prompt tokens to `10,253` characters/`3,295`
provider prompt tokens while retaining exact prefixes. Previously passing
warm, changed-tail, resolver, concurrency, sidecar, long-context, and serving
evidence is retained. The corrected serving probe directly produced provider
context rejection; artifact SHA-256
`20a8b1838b8b201b3ba5f58e45336b33e6da43a984513e11a0f230ad0284fc57`.
Under the owner's retained-evidence instruction, these repaired owner groups
are accepted without full performance-node repetition.

The attempted new full baseline
`cogv3-g7-evidence-role-official-20260822` was stopped before its first call
and produced no artifact. The older `cogv3-g7-role-binding-final-20260822`
batch predates the evidence-local source change and is retained only as stale
diagnostic evidence. The authoritative acceptance record is
`test_artifacts/cognition_core_v3/cogv3-g7-rerun-20260822/gate7_owner_accepted_disposition.json`.

Gate 7 is accepted.

### Gate 8 V3-only cutover and closure — 2026-08-22

The owner superseded dual-engine support and the planned 100-turn observation
with a bounded closure policy: delete the V2 engine and its tests, retain prior
passing evidence, and run only narrow post-cutover live checks for the repaired
failure owners. No documentation-test or broad unit-test expansion is part of
closure.

The selector, V2 route/service branches, and
`src/kazusa_ai_chatbot/cognition_core_v2/` are absent. Active semantic, state,
affect, surface, resolver, and persistence owners now live in
`cognition_core_v3` or `cognition_shared`. Protocol identifiers ending in
`V2` or `.v2` remain data-contract versions rather than an alternate engine.

Post-decommission V3 import/config smoke passed with
`CognitionChainServicesV3` and `fixed_a1_a2`. Three owner-selected live cases
then passed sequentially with eligible, input-unchanged, effect-free artifacts
and strict input/output validators:

- `group_self_cognition_proposes_reply`, SHA-256
  `276cf57cdf9a348bae9dad96bb9be2124a04e54bb1da9c0bff92ea7072d0af0e`;
- `event_agency_and_moral_chain`, SHA-256
  `fcb565723f42f30538d1010c6339b42e063411b3b9232b9be4f213e03cf825d3`;
- `memory_cue_nostalgia`, SHA-256
  `9b34a1a4abdee517717a951c2a1a049234213ecd7fe783d33cff0642af67b240`.

No role reversal, material self-conflict, boundary/safety conflict, structural,
privacy, permission, provenance, effect, or persistence failure occurred.
Emotion identity plus cause remains represented by `emotion_id`,
`primary_root`, `root_refs`, `cause_status`, and public `cause_summary`.

Gate 8 and the complete plan are accepted. Deployment rollback uses the
pre-cutover revision/configuration (`2a22d2381efa4596cb800204c4905c3a4dace33b`)
or another known-good deployment revision; no in-process V2 selector remains.
