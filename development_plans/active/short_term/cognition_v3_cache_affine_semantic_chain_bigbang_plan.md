# Cognition V3 Cache-Affine Semantic Chain Big-Bang Plan

## Summary

- Goal: create Cognition V3 as a bounded cache-affine semantic-chain engine
  that preserves the complete Cognition Core V2 input, output, state,
  emotion, relationship, goal, action, resolver, and surface handoff
  contracts while replacing the parallel-appraisal execution topology.
- Status: draft.
- Plan class: high-risk cognition architecture and local-LLM behavior change.
- Scope boundary: a new `cognition_core_v3` semantic execution package, one
  process-level V2/V3 selector, the live and idle connector import reached
  through `persona_supervisor2_cognition`, exact tests, comparison artifacts,
  documentation, and test-impact registration.
- Change direction: implement V3 as a separate engine in one coherent change;
  preserve the V2-shaped public substrate; evaluate V2 and V3 explicitly;
  move the production default to V3 only after every gate in this plan passes.
- Evidence-gated workflow: after the predecessor bugfix is frozen, capture an
  immutable V2 baseline that exercises every registered finite internal path
  and failure disposition; implement V3; replay the exact frozen corpus and
  protocol through V3; compare contract accuracy, semantic quality,
  robustness, cold/warm latency, and cache behavior; cut over only after the
  comparison proves non-regression plus directional improvement.
- Overall cutover strategy: big-bang V3 implementation and production-default
  cutover, with an explicitly compatible module selector during the
  evaluation and observation period.
- Acceptance state: architecture, parity map, failure policy, change surface,
  and test contract are drafted. Implementation and test execution require a
  later user approval and explicit production-code authorization.
- Retirement boundary: this plan retires neither the V2 package nor the
  V2-shaped state and public contracts. A later decommission plan may remove
  the V2 semantic executor after V3 production evidence is accepted. The
  shared V2-shaped substrate remains until that later plan resolves its many
  downstream imports.
- Dependency: complete and archive
  `development_plans/active/bugfix/cognition_v2_semantic_appraisal_boundary_recovery_bugfix_plan.md`
  first. Its completed commit, typed appraisal-failure disposition matrix,
  evidence-preservation invariant, and exact tests become the V2 baseline for
  V3 parity. V3 implementation cannot overlap its V2 source edits.

## Confirmed Decisions

1. Cognition V3 is an execution-engine version over the unchanged
   `CognitionCoreInputV2`, `CognitionCoreOutputV2`, `CognitionCoreServicesV2`,
   `cognition_state.v2`, `TextSurfaceInputV2`, and `TextSurfaceOutputV2`
   contracts.
2. V2 and V3 are separate importable modules. One closed startup selector
   binds exactly one engine per process. A V3 failure never falls back to V2,
   and normal production turns never execute both engines.
3. The effect-free comparison harness selects an engine explicitly per run:
   Gate 1 captures V2, and Gate 3 replays V3 from an independent copy of the
   same canonical input. Gate 3 also runs V2 and V3 as separate, alternating
   current-environment performance blocks so latency is judged against a
   contemporaneous control. Comparison mode has no state commit, action
   execution, resolver capability execution, surface delivery, persistence,
   or adapter effect.
4. V3 reuses the stable deterministic V2-shaped substrate: contracts, state
   models, transition guards, reducers, emotion definitions and derivation,
   state and output projections, semantic source planning, prompt-budget
   primitives, attempt-ledger policy, morning refresh, character carry-over,
   and L3 surface contracts.
5. V3 owns new semantic model stages and orchestration. It does not call the
   V2 semantic appraiser, V2 goal producer, V2 workspace collapse, V2 action
   planner/authorizers, V2 facade, or V2 parallel executor.
6. The six appraisal families remain complete. V3 groups them into three
   parallel first-wave chains and one post-checkpoint terminal-outcome stage:
   `event_agency -> moral_identity`, `relationship_social`,
   `epistemic_comparison_memory -> existential_drive`, then
   `goal_threat_outcome` after provisional accepted-state reduction.
7. The relationship chain remains isolated. Current-turn
   `relational_willingness.v2` remains owned by `ordinary_response`, which
   executes in parallel with first-wave appraisal chains from the same
   preliminary state.
8. Preliminary active-goal branches remain semantically isolated parallel
   chains. Newly activated final branches run in dependency-ready parallel
   waves after final appraisal reduction. Sibling goal outputs never enter one
   another's model transcript.
9. Workspace collapse, action planning, action authorization, and resolver
   authorization remain ordered semantic owners. Each safety or permission
   boundary starts a fresh prompt and consumes only the validated typed output
   of its predecessor.
10. The deterministic orchestrator owns chain selection, stage order,
    dependencies, caps, route identity, validation, retry eligibility,
    checkpointing, cancellation, failure disposition, reduction, permissions,
    and commit eligibility. The LLM never selects or creates stages.
11. Accepted same-owner output may continue a transcript. Rejected raw output
    is visible only to its bounded local repair request. Successful repair or
    deterministic normalization forces a canonical checkpoint before the next
    semantic stage.
12. A cache-domain change forces a canonical checkpoint. A cache domain is
    the normalized backend URL, hashed credential identity, backend kind,
    model, thinking/chat-template strategy, and static system-prompt hash.
13. Cache availability affects latency only. Every request is reconstructable
    from canonical input and accepted typed state, and a cache miss cannot
    change routing, validation, permissions, persistence, or output semantics.
14. The existing twelve Cognition Core service fields and route bundles remain
    unchanged. V3 uses each semantic owner's current model configuration.
15. V3 preserves the existing parent-checkpoint replay eligibility, global
    attempt-ledger arithmetic, typed failure codes required by the connector,
    protected trace/failure-capsule behavior, and one final compare-and-replace
    state commit.
16. L3 text planning, preference planning, visual planning, dialog wording,
    consolidation, persistence, resolver capabilities, adapters, and delivery
    remain outside the V3 semantic-engine implementation.
17. V3 ports the completed predecessor bugfix's exact appraisal disposition:
    only `candidate_origin_missing` and explicitly declared
    `producer_handle_domain_invalid` failures receive bounded producer
    replacement; semantic ownership, proposition-kind, target, delta-owner,
    permission/effect, state/FSM, and unknown validation failures remain
    terminal. Candidate-origin replacement preserves every valid existing
    citation and adds the canonical origin at selected and nested evidence
    levels.
18. The V2 baseline gate completes before the first V3 production-source
    edit. "Every possible internal path" means every finite registered
    control-flow edge, stage/branch eligibility outcome, typed validator and
    reducer disposition, bounded retry/exhaustion route, permission/effect
    route, connector source route, replay/resolver lifecycle route, and commit
    result. The Cartesian product of unrelated choices is not a distinct path
    unless their interaction changes control flow or ownership.
19. The baseline corpus, path registry, harness code, model/configuration
    fingerprint, scoring rubric, timing protocol, and V2 raw evidence are
    content-addressed and immutable for one `baseline_id`. A correction to any
    of them creates a new baseline version and reruns the complete V2 capture
    before V3 comparison resumes.
20. V3 replay uses the exact baseline case IDs and canonical input hashes. It
    uses the same model routes, model artifacts, backend strategy, generation
    parameters, prompt-budget settings, machine class, and cold/warm procedure
    as a contemporaneous V2 performance-control run. The historical V2 timing
    remains longitudinal evidence; an environment-fingerprint difference
    makes only the historical timing delta non-comparable.
21. Sign-off requires all registered V2 and target V3 paths to have evidence,
    complete approved V2-to-V3 mapping, every hard contract and safety gate to
    pass, no capability-group quality regression, a strictly positive
    aggregate semantic-quality delta in at least one predeclared V3 objective
    group, and every fixed performance gate to pass. Equal quality everywhere
    is parity evidence rather than evidence that V3 improves in the intended
    direction.
22. A failed or invalid comparison keeps V2 as the production default and V3
    available only for explicit evaluation. Remediation changes V3 or creates
    a newly versioned baseline; it never edits historical baseline evidence or
    weakens a gate after results are known.

## Minimal Current Contract

```text
Semantic question:
Produce the same complete character cognition decision and replacement state
through bounded ordered semantic chains whose accepted local context can reuse
an exact model prefix.

Inputs required:
The exact validated CognitionCoreInputV2, existing stage model configurations,
typed evidence and handles, preliminary state projection, identity,
relationship and character operational context, resolver state, and existing
action/resolver affordances.

Output fields required:
The exact validated CognitionCoreOutputV2, including intention, admitted and
supporting bids, replacement state update, affect, relationship,
relational-willingness, action and resolver requests, goal progress,
expression policy, diagnostics, observability, and private residue.

Deterministic owners:
Stage registry, scheduling, cache-domain comparison, prompt fitting,
checkpointing, JSON parsing, structural validation, trial reduction, path
ownership, attempt caps, failure disposition, authorization boundaries,
output validation, state commit, and external effects.

Rejected complexity:
Autonomous stage creation, LLM-selected routing, hidden provider conversation
state as correctness authority, raw sibling-output sharing, raw rejected-output
carry-over, keyword routing over user text, semantic post-processing that
rewrites model decisions, automatic V2 fallback, dual state writes, synthetic
user-visible warm-up turns, and provider-specific cache correctness logic.

Evidence required:
Exact deterministic owner tests, connector and resolver propagation tests,
an exhaustive registered-path closure report, immutable same-input V2/V3
side-by-side live local-model review, prompt-injection review, captured failure
replay, and controlled cold/warm/cache-domain performance evidence.
```

## Current V2 Functionality And V3 Parity Map

| Current V2 capability | Current owner | V3 mapping and required invariant |
| --- | --- | --- |
| Exact public input/output and services | `cognition_core_v2.contracts` and package entrypoints | V3 consumes and returns the same objects. No extra output keys, renamed fields, aliases, or adapters. |
| User/character mutable-state separation | `state_models.py`, connector, DB facades | Reuse unchanged state resolution and one final compare-and-replace commit. |
| Eleven native relationship/affinity dimensions and maintenance | `RelationshipStateV2`, relationship reducer and projection | Preserve familiarity, positive regard, trust, attachment, desired/perceived closeness, care, boundary safety, exclusivity, unresolved injury, salience, evidence, date reinforcement, idempotency, and decay. The legacy untyped affinity scalar remains outside cognition. |
| Twenty-one emotion families | `emotion_definitions.py`, `emotion_derivation.py` | Reuse exact formulas, roots, begin/sustain/fade lifecycle, thresholds, reinforcement, trends, timestamps, and action tendencies. |
| Character drives, standards, meaning, and sleep recovery | state models, reducers, morning refresh | Preserve eight drives, standards, meaning state, elapsed decay, sleep recovery, and state-scope rules. |
| Typed causal state | events, threats, goals, knowledge gaps, transition guards | Preserve entity caps, handles, roles, evidence retention, terminal immutability, candidate materialization, comparison identity, and guarded transitions. |
| Six semantic appraisal families | source planner, semantic appraiser | Preserve each question vocabulary, evidence visibility, permitted handle domain, target-path ownership, proposition/delta cardinality, micro-item cap, and trial reduction. V3 changes only scheduling and accepted-context continuation. |
| Fourteen goal kinds and branch bids | branch activation and goal cognition | Preserve branch registry, goal ownership, route/capability matrix, complete-bid contract, active-goal recurrence, required branch policy, and first-person private monologue. |
| Ordinary response baseline | ordinary goal owner | Preserve universal ordinary branch execution and its authority as the neutral current-event response bid. |
| Current-turn relational willingness | ordinary goal owner and workspace | Preserve exact `relational_willingness.v2`, relationship-state descriptor, stance vocabulary, evidence binding, fail-closed behavior, and authoritative collapse/suppression. Relationship appraisal remains a different owner. |
| Required-selection role operation | decontextualizer input carrier, goal owner, output carrier | Preserve selected operation, writable/code-owned fields, actor/target roles, progress evidence, complete selection coverage, and downstream immutable propagation. |
| Group participant and addressee ownership | episode-local `pN` bindings and goal/surface contracts | Preserve third-party identity isolation, target semantics, second-person permission, display-name rendering, and delivery-recipient separation. |
| Group self-cognition speech judgment | goal and action planning | Preserve `stay_silent`/`propose_visible_reply`, evidence handles, participation basis, target domain, advisory engagement context, and deterministic route derivation. |
| Workspace competition | workspace collapse | Preserve ordinary baseline, persistent-goal matter matching, complete-bid-only admission, authoritative relational collapse, supporting bids, and no confidence-score ranking. |
| Action and resolver selection | action planning | Preserve goal answerability, action/resolver request shapes, pending resolution, continuation refs, scheduled authority, group-self-cognition precedence, and capability-neutral goal ownership. |
| Permission boundaries | action and resolver authorization | Preserve fresh isolated prompts, permitted handles, runtime capability limits, denial disposition, and zero unauthorized side effects. |
| Affect/relationship/expression output | output projection | Preserve semantic affect, semantic relationship, expression policy, selected bid reason, private residue, and changed-state receipt. |
| Resolver recurrence | `cognition_resolver` and connector | Preserve cognition -> one capability -> typed observation -> cognition, cycle cap, no intermediate commit, answerability suppression, pending state, and final single commit. |
| L3 text/preference/visual surface | V2 surface and node connector | Reuse unchanged after a validated V3 output. Content and preference remain parallel; visual remains an optional sibling. |
| Self-cognition, tool result, scheduled and internal sources | episode/connector/self-cognition runner | Preserve every canonical source, output-mode constraint, scheduled authority, targetless group behavior, and source-bound delivery. |
| Prompt caps and bounded regeneration | prompt-budget owners and attempt ledger | Preserve owner caps, call arithmetic, zero-call over-cap dispositions, provider retry behavior, parent checkpoint arithmetic, and the completed typed appraisal-repair matrix from the predecessor bugfix. |
| Diagnostics and protected evidence | observability, LLM tracing, failure capsules | Preserve exact public diagnostic shape and add V3-only chain detail to protected trace metadata rather than the public output. |
| Morning refresh and operational carry-over | V2 deterministic public functions | Reuse unchanged. V3 semantic execution cannot alter their persisted-state contracts. |

## Current V2 Parallel And Sequential Topology Baseline

The baseline audit finds these complete explicit concurrency regions on the
live cognition and immediate L3 path. The V2 source contains no other
`asyncio.create_task` or `asyncio.gather` region inside `cognition_core_v2`.

| Region | Current V2 parallel execution | V3 disposition |
| --- | --- | --- |
| Connector preparation, outside the engine | On resolver cycle zero, eligible shared-memory RAG prewarm runs while identity and mutable-state preparation proceeds. The connector joins prewarm before building the canonical cognition input. | Unchanged connector-owned optimization. It remains outside V3 prompts, stage routing, and semantic state ownership. |
| Core first wave | Every planned semantic appraisal question runs as its own task. At the same time, the preliminary goal dependency graph runs as one sibling task; V2 strips preliminary branch dependencies, so all ready ordinary/active preliminary branches fan out concurrently. | Replace only the all-question appraisal fan-out with the registered V3 appraisal chains. Keep ordinary and preliminary active-goal chains isolated and parallel with V3 first-wave chains. |
| Appraisal task internals | Each appraisal task executes its own bounded micro-item loop, provider retry, contract replacement, and trial reduction sequentially. No micro-items within one question run in parallel. | Continue accepted same-owner stages sequentially in one cache-affine transcript; keep replacement bounded and checkpoint after repair/normalization. |
| Goal dependency execution | `execute_dependency_graph` starts every currently ready branch concurrently, waits for first completion, then releases the next dependency-ready wave. Branches in one wave are parallel; dependent waves are sequential. | Preserve this readiness-wave model for newly activated final goals, with sibling transcript isolation and deterministic registry order. |
| Text surface, after cognition | Content-plan and preference stages run concurrently and join before exact `TextSurfaceOutputV2` projection. | Unchanged and outside V3 semantic execution. |
| L3 terminal surface | When visual directives are enabled, the text-surface pipeline and visual-surface pipeline run as siblings; text runs alone when visual output is disabled. | Unchanged and outside V3 semantic execution. |

The V2 sequential spine, which the baseline path registry must exercise in
order, is:

```text
validate canonical input
  -> elapsed/direct-fact state update
  -> deterministic preliminary goals and prompt projection
  -> plan appraisal questions and preliminary branches
  -> join appraisal fan-out + preliminary goal fan-out
  -> cumulative isolated appraisal reduction
       (semantic deltas -> relief transitions -> state/affect update
        -> deterministic goals -> validation)
  -> final relationship maintenance
  -> user-scope state/goal reconciliation and validation when applicable
  -> run dependency-ready final goal waves
  -> filter stale/incomplete bids
  -> authoritative relational collapse OR workspace LLM collapse
  -> action planner
  -> action authorization when proposed
  -> resolver authorization when proposed and semantically required
  -> route/intention/state/output projection and validation
  -> resolver recurrence when requested, bounded by the existing cycle policy
  -> one final compare-and-replace commit
  -> unchanged L3 surface planning and delivery
```

Within each V2 LLM owner, attempts are sequential: original request, optional
bounded replacement after an eligible failure, then accepted result or typed
exhaustion. Action and resolver proposals are mutually exclusive by contract;
their fresh authorization boundaries therefore remain ordered conditional
stages rather than a parallel region.

## Target Architecture

```text
canonical CognitionCoreInputV2
  -> exact input validation
  -> deterministic elapsed update, direct facts, goals, handles, projection
  -> PARALLEL WAVE A
       chain causal_normative
         event_agency
           -> validate / bounded local repair / trial reduction
         moral_identity
           -> validate / bounded local repair / trial reduction

       chain relationship
         relationship_social micro-items
           -> validate / bounded local repair / trial reduction

       chain epistemic_meaning
         epistemic_comparison_memory
           -> validate / bounded local repair / trial reduction
         existential_drive
           -> validate / bounded local repair / trial reduction

       isolated ordinary_response goal chain
       isolated preliminary active-goal chains
  -> deterministic accepted-prefix provisional reduction
  -> goal_threat_outcome terminal-outcome stage
       -> validate / bounded local repair / trial reduction
  -> deterministic final reduction
       -> relationship maintenance exactly once
       -> causal/goal lifecycle reconciliation
       -> emotion derivation exactly once
  -> PARALLEL WAVE B
       newly activated dependency-ready goal chains
  -> complete-bid join
  -> fresh workspace collapse boundary
  -> fresh action-planning boundary
  -> fresh action-authorization boundary when applicable
  -> fresh resolver-authorization boundary when applicable
  -> exact CognitionCoreOutputV2 projection and validation
```

### Appraisal Chain Registry

| Chain | Ordered stages | Visibility | Failure continuation |
| --- | --- | --- | --- |
| `causal_normative` | `event_agency`, `moral_identity` | Later stage receives the accepted canonical local state and accepted bounded semantic summary. | Each stage is optional. Exhaustion records the failure, restores the latest accepted checkpoint, and allows the next independently valid stage to run. |
| `relationship` | `relationship_social` | Full same-owner accepted continuation only. No ordinary-goal, sibling-appraisal, or action transcript enters this chain. | Exhaustion omits the family with typed diagnostics; existing relationship state and maintenance rules remain authoritative. |
| `epistemic_meaning` | `epistemic_comparison_memory`, `existential_drive` | Existential appraisal receives only accepted epistemic state and summary. | Each stage is optional and independently valid from the chain root. |
| `terminal_outcome` | `goal_threat_outcome` after provisional reduction | Receives a fresh canonical projection of the provisional accepted state and original authorized evidence. | Exhaustion omits terminal-outcome appraisal and retains the provisional accepted state. |

The registry is immutable at runtime. Evidence visibility still decides whether
an already registered stage has a question. User text, model output, cache
state, timing, and provider errors cannot create, remove, reorder, or reroute a
stage.

### Transcript Contract

Normal accepted continuation:

```text
System(static chain contract)
Human(stage 1 current facts)
Assistant(stage 1 accepted candidate)
Human(stage 2 request + compact accepted stage 1 projection)
Assistant(stage 2 accepted candidate)
```

Local structural repair:

```text
System(static chain contract)
Human(stage request)
Assistant(latest bounded invalid candidate)
Human(exact contract error + exact allowed values + replacement instruction)
Assistant(complete replacement)
```

After repair, normalization, route-domain change, or context pressure:

```text
System(static chain contract)
Human(canonical accepted checkpoint + next stage current facts)
```

The checkpoint contains accepted typed propositions, deltas, semantic summaries,
and the prompt-safe state projection required by the next owner. It contains no
rejected candidate, validator prose from a previous stage, sibling output, raw
model trace, provider metadata, permissions, adapter fields, or hidden state.

### Cache Contract

- Every appraisal chain uses one byte-identical static V3 appraisal system
  prompt. Question kind, semantic question, evidence, handles, state
  projection, and accepted predecessor context live only in the human tail;
  chain isolation is separate message history rather than a prompt variant.
- Every ordinary, active, and final goal chain uses one byte-identical static
  V3 goal system prompt. Goal kind, branch context, relationship state, current
  evidence, and accepted appraisal projection live only in the human tail.
- Workspace, action planning, action authorization, and resolver authorization
  retain separate owner-specific system prompts because their semantic and
  safety contracts differ. They never borrow an appraisal or goal transcript
  merely to preserve a cache prefix.
- Static instructions, schema, enum meanings, positive decision procedure,
  and stable examples occupy the exact system-message prefix.
- Episode text, evidence, state, handles, stage request, prior accepted
  projection, resolver observations, and repair facts remain in human-message
  tails.
- Same-domain accepted continuation preserves the complete prior message
  prefix byte-for-byte.
- A route-domain mismatch starts a fresh request from a canonical checkpoint;
  semantic continuity remains intact while backend KV continuity ends.
- A provider cache hit, miss, eviction, model reload, or absent cache counter
  produces the same messages, validation, state, and output.
- Synthetic prompt-warm-up calls and user-visible warm-up turns remain outside
  this plan. The first real parallel wave exposes one byte-identical reusable
  prefix per actual appraisal or goal cache domain; measured cold and warm
  behavior decides acceptance without making backend reuse a correctness
  dependency.

### Goal And Safety Isolation

- `ordinary_response` and every active branch use independent transcripts.
- A branch receives current episode evidence, permitted state/identity/
  relationship context, its active goal, and accepted appraisal summaries
  allowed by the existing branch contract.
- A branch receives no sibling candidate, confidence, private monologue, or
  collapse ranking.
- Workspace sees complete validated bids only.
- Action planning sees the admitted intention and declared affordances; it
  receives no appraisal or goal raw transcript.
- Action and resolver authorization each receive only their typed proposal,
  permitted evidence/handles, and deterministic availability context.
- A non-accepting relationship-sensitive stance continues to suppress
  effectful action/resolver requests before authorization.

## Local-LLM Failure Modes And Required Dispositions

| Failure mode | Detection owner | Required V3 disposition |
| --- | --- | --- |
| Weak model loses early instructions in a growing transcript | Prompt-cap/checkpoint owner plus live review | Checkpoint before the stage cap and project only accepted compact context. Keep each semantic question narrow. |
| Prior output anchors or contaminates an independent judgment | Registry and visibility validator | Keep independent families/branches in separate chains and expose only declared accepted projections. |
| Prompt injection appears in current text, evidence, or an accepted semantic string | Prompt projection and boundary prompts | Treat dynamic content as delimited evidence; retain handles/provenance; keep scheduler and permissions deterministic; start fresh at safety boundaries. |
| Rejected output pollutes later stages | Transcript owner | Permit it only in the local repair request; force a canonical scrub checkpoint before continuation. |
| Parsed output has missing/unknown keys, wrong types, unsupported enums, handles, or paths | Stage structural validator | Request one complete bounded replacement under the existing owner attempt cap; keep the candidate out of reduction and effects. |
| JSON remains malformed after canonical parser/allowed repair | Producing stage | Consume the stage attempt budget and apply that owner's existing exhausted disposition. |
| Candidate-bearing proposition/delta omits its canonical origin evidence | Typed appraisal validator | Classify `candidate_origin_missing`; request bounded producer replacement; require every valid prior citation plus the canonical origin at selected and nested evidence levels. |
| Generated handle violates an explicitly declared question-local field domain | Typed appraisal validator | Classify `producer_handle_domain_invalid`; request bounded producer replacement; deterministic code selects no substitute. |
| Semantic ownership, proposition-kind, target, delta-owner, duplicate/conflict, permission/effect, state/FSM, or unknown validator failure occurs | Typed appraisal validator or trial reducer | Classify the exact terminal/unknown disposition from the completed predecessor matrix, issue no producer-repair call, and keep the candidate out of state and effects. |
| Model repeats an accepted micro-item or emits an empty item | Appraisal stage | Terminate that family successfully with its accepted prefix. |
| Model invents a stage, route, dependency, permission, capability, or cache instruction | Registry and validators | Reject the unsupported field/value. Deterministic topology and permission owners remain unchanged. |
| First stage of a chain exhausts | Chain executor | Restore the last accepted checkpoint and run the next registered stage when it is independently valid from the root. |
| Required ordinary goal exhausts | Goal owner/facade | Preserve the current complete-sibling recovery rule and typed pre-state-commit failure when no complete valid recovery exists. |
| Optional goal branch exhausts | Branch executor | Isolate the branch, record diagnostics, and admit only complete sibling bids. |
| Cross-chain outputs conflict or target the same path | Path-owner validator and final reducer | Fail the conflicting result slot closed; preserve other accepted slots and fixed reduction order. |
| Provider timeout, cancellation, model crash, or unload | `LLInterface`, stage executor, parent guardrail | Preserve provider reload retry, cancellation propagation, global attempt ledger, typed failure, and zero effects before complete output. |
| Cache is absent or evicted | Provider/backend | Continue correctly from full messages. Record performance evidence without changing semantic behavior. |
| Cache domain changes between related stages | Transcript owner | Start a fresh canonical checkpoint on the new route; retain no cross-provider hidden state. |
| Cumulative context exceeds the stage cap | Prompt-budget owner | Checkpoint and refit using the existing stable reduction order. Apply the existing zero-call owner disposition if required content still cannot fit. |
| Action/resolver proposal hallucinates availability or parameters | Authorization owners and action materializer | Deny invalid rows; grant no execution from prose or prior transcript. |
| Parent checkpoint replays V3 | Existing coordinator plus V3 attempt ledger integration | Replay only `run_cognition` from the immutable canonical input, preserve epoch/call arithmetic, and repeat no preparation, capability, action, surface, commit, or delivery work. |
| Final state base is stale | Existing DB compare-and-replace owner | Fail the one final commit closed; perform no silent merge or second write. |
| Resolver recurrence compounds a weak decision | Existing resolver cap and V3 output validator | Execute at most one admitted capability per cycle, project typed evidence, retain current-turn relational carrier, and stop at the existing cycle/final lifecycle gates. |
| Semantically poor but structurally valid judgment | Human-reviewed real-LLM comparison | Record the real input/output and quality regression. Improve the owning prompt/contract in V3; add no deterministic semantic rewrite or hidden evaluator. |

## Evidence-Gated Development Workflow

The workflow is ordered and irreversible inside one baseline version:

```text
complete V2 predecessor fix
  -> freeze V2 code/config/corpus/harness/rubric
  -> exercise and capture every registered V2 path
  -> seal immutable V2 baseline
  -> implement V3 with V2 still selected by default
  -> replay the exact sealed baseline through V3
  -> compare robustness + contract accuracy + semantic quality + performance
  -> independent review and human-owner sign-off
  -> switch the production default to V3
```

### Gate 0: Baseline Eligibility

- Complete and archive the predecessor V2 appraisal-recovery bugfix. Record
  its commit and exact passing test nodes.
- Confirm the V2 production package has no later uncommitted or unreviewed
  source change. Record the complete worktree status and explicitly owned
  files.
- Freeze the baseline harness, case manifest, path registry, rubrics, model
  and route configuration, prompt budgets, generation parameters, backend
  strategy, and environment fingerprint before the first V3 production edit.
- Derive `baseline_id` from the V2 commit, baseline-manifest hash,
  harness-code hash, model/configuration fingerprint, and rubric hash. The
  baseline index records each component rather than relying on a mutable name.
- Gate 0 fails if the predecessor plan is active, the V2 source is dirty, an
  input lacks a canonical hash, or any required model/configuration identity
  is unavailable.

### Gate 1: Capture The Exhaustive V2 Baseline

The baseline has two complementary evidence layers:

1. Deterministic and patched-LLM path evidence exercises the complete finite
   control-flow graph and all typed failure dispositions. This layer proves
   route, ownership, isolation, retry, reduction, permission, effect, and
   commit behavior without depending on model luck.
2. Real local-LLM evidence uses production traces and captured failures where
   available, plus explicitly labelled realistic fixtures for missing
   semantic situations. This layer measures actual judgment quality and
   latency; fixtures alone cannot establish production performance.

The path manifest expands every row below into stable `scenario_id` values and
engine-scoped `path_id` values. Every scenario names its preconditions,
canonical input or patched producer sequence, V2 expected edges and terminal
disposition, predeclared V3 target edges and disposition, required artifacts,
and exact pytest node. Each V2 path is classified as `preserved_shared`,
`replaced_topology`, or `external_unchanged` and maps to at least one target
V3 or unchanged path. Gate 1 V2 coverage closes only when every registered V2
path has evidence and no observed V2 runtime edge is absent from the manifest.
The same frozen scenario set must also reach every registered target V3 path
in Gate 3.

| Path class | Exhaustive registered coverage required |
| --- | --- |
| Appraisal eligibility and topology | Question absent/present for all six appraisal families; accepted, empty, repeated-item completion, optional exhaustion, provisional reduction, terminal-outcome execution/omission, and every declared sequential edge and parallel-wave join. |
| Goal topology | Ordinary response, every one of the fourteen goal kinds, inactive/active/newly activated states, complete/incomplete bid, required-selection success/exhaustion, sibling isolation, dependency-ready waves, and workspace admission/suppression. |
| Typed appraisal disposition | Every repairable and terminal class from the completed predecessor matrix, successful replacement, replacement exhaustion, valid-citation preservation, unknown failure, and proof that terminal classes make zero repair calls. |
| Structure and parsing | Canonical parser success, deterministic syntax cleanup, allowed JSON repair, irreparable structure, missing/unknown keys, wrong types, enum/handle/path failures, normalization success/failure, and owner attempt exhaustion. |
| Bounded execution | Provider success, configured provider reload retry, timeout, cancellation, model error, zero-call prompt over-cap, stage cap, global attempt cap, optional-owner failure, required-owner failure, and parent checkpoint replay arithmetic. |
| Transcript and cache isolation | Accepted same-domain continuation, rejected-candidate local repair visibility, post-repair scrub checkpoint, context-pressure checkpoint, route-domain split, cache hit/miss/eviction equivalence, sibling non-visibility, and fresh workspace/action/resolver safety boundaries. |
| Relationship, emotion, and state | Every relationship/affinity carrier and maintenance path, accepting/non-accepting relational willingness, all twenty-one emotion-family derivation/lifecycle paths, goal/threat/event/knowledge transition guards, elapsed update, morning refresh, and character/user scope separation. |
| Action and resolver | No proposal, valid/invalid action proposal, valid/invalid resolver proposal, permission grant/deny, unavailable capability, answerability suppression, one-capability recurrence, cycle cap, and zero effect before complete authorization. |
| Episode and connector source | Direct and group chat, group addressee/third-party roles, self-cognition speak/silence, tool result, resolver observation, scheduled/future-speak authority, targetless internal source, live connector, idle connector, and first-cycle shared-memory prewarm ineligible/success/failure/cancellation join paths. |
| Output, surface, and persistence | Visible response, silence, exact V2 output projection, protected diagnostics, private-residue handling, text content/preference success and degraded joins, visual disabled/enabled/degraded paths, successful one-time compare-and-replace commit, stale-base rejection, and no commit/action/delivery in comparison mode. |

This is exhaustive path coverage, not an assertion that the natural-language
input space is finite. A Cartesian combination is added as its own `path_id`
when interaction can change stage visibility, reduction, safety, effect, or
output; otherwise each independent edge and disposition is exercised once or
more without multiplying unrelated combinations.

Gate 1 execution order is fixed:

1. Collect the exact existing V2 deterministic and integration nodes named by
   the baseline manifest.
2. Run the deterministic and patched-LLM path matrix and write raw path traces.
3. Run each V2 live-quality case as a separate pytest node. Each case performs
   three independent model trials under the frozen generation configuration;
   inspect and review all three trials before starting the next case.
4. Capture V2 performance using five independent cold backend/model-load
   trials, twenty measured warm exact-repeat trials after one real effect-free
   setup request, twenty measured changed-tail trials, and ten mixed-route
   transitions. Record the setup request as evidence and exclude it from warm
   statistics.
5. Write the V2 path-closure report, deterministic contract results, raw live
   outputs, agent-authored quality reviews, and timing summary.
6. Seal the V2 baseline index with content hashes. The indexed V2 files remain
   immutable; Gate 3 creates V3 and comparison sibling files plus a separate
   comparison index without rewriting the baseline index.

Gate 1 passes only when path coverage is 100%, all required evidence files
match the index, every live trial has a readable review, and known V2 defects
are explicitly recorded with a predeclared V3 acceptance expectation. The V2
output is comparison evidence, not automatically the ideal semantic answer.

### Gate 2: Implement V3 Against The Frozen Contract

- Keep V2 as the configured production default and leave the sealed V2
  package unchanged.
- Implement only the V3, selector, connector-import, tests, and documentation
  surfaces listed by this plan.
- Use the baseline manifest as an immutable external acceptance contract. V3
  code may add source-mirrored owner tests, while the baseline inputs, path
  IDs, rubrics, and thresholds remain fixed.
- Run V3 deterministic owner tests and patched integration tests during
  implementation. These development tests do not replace the formal replay.
- If implementation reveals a missing V2 path or a faulty baseline fixture,
  stop V3 comparison, version the manifest, recapture the complete V2
  baseline, and seal a new `baseline_id` before continuing.

### Gate 3: Replay And Compare V3

- Run the same harness commit in V3 replay mode. It rejects a case when its
  scenario/case ID, canonical input hash, rubric hash, route/configuration
  fingerprint, or predeclared path mapping differs from the sealed baseline.
- Run every frozen deterministic and patched scenario through V3. Produce a
  V3 path-closure report against the predeclared target V3 `path_id` set and a
  mapping report proving that every V2 path is preserved, intentionally
  replaced by the approved topology, or owned by an unchanged external stage.
- Run the same twenty-four live-quality cases, three trials per case, one
  pytest node and one fully inspected case at a time. Preserve trial-level raw
  output; do not collapse model variance into a single preferred example.
- Score semantic output under blind `A`/`B` engine labels when the reviewer did
  not execute the implementation. Unblind only after the per-trial rubric and
  hard gates are recorded, then write the side-by-side comparison.
- Run the exact frozen performance protocol for both engines on the current
  environment in separate cache epochs and alternating engine-order blocks.
  Use current V2 as the acceptance control and retain historical V2 timing as
  a drift reference. When the hardware, backend build, model checksum, driver,
  load policy, or measurement instrumentation differs from the Gate 1
  fingerprint, mark only the historical timing delta non-comparable. The
  current V2 run supplements rather than replaces the immutable semantic V2
  baseline.
- Produce four separate comparison surfaces: deterministic contract accuracy,
  semantic quality, robustness/path coverage, and performance/cache behavior.
  A favorable aggregate cannot hide a safety, permission, isolation, state,
  or capability-group regression.

Contract accuracy is the percentage of deterministic cases that match their
predeclared schema, ownership, route, state-transition, permission, effect,
and failure-disposition expectations. V3 requires 100%. Semantic quality uses
the fixed 0/1/2 rubric: every V3 hard gate passes, no V3 dimension scores `0`,
each capability-group mean is at least its V2 baseline, the overall V3 mean is
strictly greater than V2, and at least one of `appraisal/state`,
`goal/selection`, or `robustness` improves by at least 0.20 on the 0-2 scale.
Performance uses the thresholds in Performance Comparison Contract.

### Gate 4: Sign-Off And Cutover

The evidence package contains the sealed V2 index, final comparison index, V2
and V3 path-closure reports, same-input hash report, contract-accuracy summary,
trial-level raw evidence, agent-authored quality comparisons, performance
report, source-test mapping, implementation diff, and independent reviews.

Sign-off requires all of these conclusions:

- **Robust design:** 100% V2 and target V3 registered-path coverage, complete
  approved path mapping, bounded failure behavior, exact isolation/checkpoint
  behavior, no unauthorized effects, and no external-contract drift.
- **Preserved character brain:** emotion, relationship/affinity, goal,
  boundary, self-cognition, action/resolver, state, and surface carriers pass
  the parity gates.
- **Improved direction:** V3 meets the fixed positive semantic delta and warm
  cache/prompt-processing improvement while satisfying cold and end-to-end
  latency bounds.
- **Operational safety:** V2 remains explicitly selectable, V3 has no runtime
  fallback or dual effects, rollback is a process-level selector change, and
  the production smoke protocol is complete.
- **Independent judgment:** the LLM quality/performance reviewer and code
  reviewer pass independently, followed by explicit human-owner authorization
  for the production-default change.

Any failed, missing, or invalid gate keeps the production default on V2. V3
remediation and full affected-gate replay continue under this plan; a change
to baseline evidence, topology, public contracts, or acceptance thresholds
requires a plan amendment and a newly sealed V2 baseline.

### Baseline And Comparison Artifact Contract

```text
test_artifacts/cognition_v3_baseline/<baseline_id>/
  baseline_index.json
  path_registry.json
  v2/path_coverage.json
  v2/deterministic/<path_id>.json
  v2/live/<case_id>/<trial_id>.json
  v2/performance/<case_id>.json
  v3/path_coverage.json
  v3/deterministic/<path_id>.json
  v3/live/<case_id>/<trial_id>.json
  v3/performance/<case_id>.json
  comparison/current_v2_performance/<case_id>.json
  comparison/input_identity.json
  comparison/path_mapping.json
  comparison/contract_accuracy.json
  comparison/semantic_quality.json
  comparison/performance.json
  comparison/comparison_index.json
```

Harnesses and tests write raw JSON, JSONL, CSV, logs, and hashes only. The
reviewing agent writes human-readable Markdown under
`test_artifacts/llm_reviews/cognition_v3/<baseline_id>/`. Artifacts record
redactions and contain no credentials, raw API keys, private platform IDs, or
unprotected hidden prompts.

## Cutover Policy

Overall strategy: bigbang for the V3 implementation and production-default
switch. The user-required engine coexistence is the only compatible runtime
area before later V2 retirement.

| Area | Policy | Instruction |
| --- | --- | --- |
| V3 semantic execution package | bigbang | Create the complete target topology directly. Retain no V3 legacy parallel-appraisal path, topology flag, alias stage vocabulary, or internal V2 semantic fallback. |
| Public cognition/state/surface contracts | compatible | Preserve the exact V2-shaped contracts throughout coexistence. This is the module boundary required by the user, not a translation shim. |
| Engine selection | compatible | Select exactly `v2` or `v3` once at process startup through a closed configuration value. Unknown values stop configuration. V3 failure never invokes V2. |
| Evaluation harness | compatible | Capture V2 and replay V3 as separate explicit effect-free runs over independent copies and exact canonical hashes. In Gate 3, run alternating current-environment V2/V3 performance blocks in separate cache epochs and use current V2 as the latency control. Store separate raw outputs and timing evidence. |
| Production default | bigbang | Keep V2 as the default during implementation/evaluation. After all acceptance gates and independent sign-off pass, change the default to V3 in one cutover while retaining an explicit V2 selector value for the observation period. |
| Persistent cognition state | compatible | Read and write exactly `cognition_state.v2`; add no migration, dual write, shadow state, or V3-only persisted field. |
| L3, dialog, resolver capabilities, consolidation and delivery | compatible | Consume the unchanged public output and remain unchanged except documentation references needed to describe engine selection. |
| V2 executor retirement | deferred | Create a separate approved decommission plan after V3 production evidence is accepted. That plan resolves shared-substrate naming/imports and removes the selector's V2 value. |

## Mandatory Skills

- `development-plan`: before approving, executing, reviewing, or closing this
  plan.
- `local-llm-architecture`: before changing stage responsibilities, prompts,
  chain visibility, cache layout, model routing, failure policy, or latency
  design.
- `py-style`: before creating or modifying Python source.
- `test-style-and-execution`: before adding/changing tests or running any
  deterministic, patched-LLM, integration, performance, or live-LLM case.
- `debug-llm`: before creating comparison artifacts or running each live local
  model case. Human-readable quality reviews remain agent-authored.
- `no-prepost-user-input`: before changing how user requests, boundaries,
  choices, permissions, accepted work, relational willingness, or commitments
  are interpreted or carried.
- `cjk-safety`: whenever a changed Python file contains Chinese or Japanese
  text.

## Mandatory Rules

- Capture `git status --short`, the baseline commit, and the exact owned file
  set before the first production edit.
- Preserve pre-existing and concurrent work outside the owned file set.
- Keep the selector closed, deterministic, process-scoped, and effect-free.
- Keep production on one engine per process. Comparison belongs only to the
  isolated harness.
- Reuse V2-shaped deterministic state and external contracts without copying
  or renaming them during coexistence.
- Give V3 independent semantic prompts, parsers, stage functions, and facade;
  import no V2 semantic executor listed in Confirmed Decision 5.
- Keep every normal chain finite. The registry stage count, micro-item count,
  per-owner attempt count, resolver-cycle count, and parent replay count remain
  hard deterministic bounds.
- Keep LLM semantic judgment intact. Deterministic code validates shape,
  provenance, paths, FSMs, limits, permissions, state, and effects; it does not
  rewrite a valid model stance into another semantic decision.
- Preserve emotion and relationship behavior as first-class acceptance gates,
  not incidental output fields.
- Start from the completed predecessor bugfix commit and preserve its typed
  appraisal failure matrix and evidence-preservation rules exactly.
- Preserve current-turn relational-willingness ownership in the ordinary goal
  branch.
- Preserve evidence authority. Retrieved, reflected, historical, resolver,
  tool, and private-residue context cannot become current fact, persona stance,
  permission, or final wording merely by entering a transcript.
- Keep prompt-stable instructions in system messages and per-run facts in
  human-message tails.
- Record cache-domain hashes and timing metadata without credentials, raw API
  keys, private identifiers, or model-facing hidden instructions.
- Run deterministic tests in batches only after exact collection succeeds.
- Run live LLM tests one node at a time, inspect its complete artifact, author
  its readable review, and resolve suspicious behavior before the next node.
- Treat parser/schema success as harness evidence only. Human review owns
  semantic-quality acceptance.
- Require independent plan review before approval and independent code review
  before production-default cutover.
- Keep implementation authorization distinct from plan approval.

## Must Do

- Before any V3 production edit, create and freeze the baseline path registry,
  case manifests, effect-free capture/replay harness, scoring rubric, timing
  protocol, and artifact-integrity contract.
- Exercise every registered V2 control-flow edge and terminal disposition,
  capture raw deterministic, patched-LLM, live-quality, and performance
  evidence, close path coverage at 100%, and seal the V2 `baseline_id`.
- Create the complete `cognition_core_v3` package and public `run_cognition`
  entrypoint.
- Add exact chain contracts, fixed registry, transcript/checkpoint owner,
  bounded executor, V3 appraiser, V3 goal producer, V3 workspace collapse, V3
  action planner/authorizers, V3 diagnostics, and V3 facade.
- Preserve every V2 capability and external carrier in the parity map.
- Port the completed predecessor bugfix's typed appraisal-failure classes,
  repair eligibility, citation-preservation validation, exhaustion metadata,
  and terminal no-repair classes into V3.
- Add a closed `COGNITION_CORE_ENGINE` setting accepting exactly `v2` and
  `v3`.
- Add one selector module that resolves the configured module once and exposes
  the selected `run_cognition` callable.
- Change the persona cognition connector's engine import only. Its canonical
  input builder, service construction, resolver handoff, commit, and output
  projection remain unchanged.
- Prove idle self-cognition reaches the same selector through the existing
  connector rather than adding a separate V3 path.
- Add deterministic source-mirrored unit tests for every V3 production owner.
- Add patched-LLM integration tests for selector, connector, resolver,
  parent-replay, state-commit, relational stance, required selection, action,
  and surface handoffs.
- Add a fixed 24-case live quality manifest and one unparameterized pytest node
  per case. Capture and inspect three independent model trials per engine per
  case.
- Add four one-at-a-time performance comparison nodes covering cold full
  appraisal, warm exact repeat, warm changed tail, and mixed-route checkpoint
  with the fixed sample counts and environment controls in Gate 1.
- Replay V3 with the exact sealed scenario/case IDs, canonical input hashes,
  predeclared V2-to-V3 path mappings, harness, route/configuration fingerprint,
  rubric, and timing protocol.
- Produce raw V2/V3 JSON evidence plus an agent-authored V2 baseline review
  and final side-by-side Markdown review per live case, together with path-
  closure, input-identity, contract-accuracy, semantic-quality, and
  performance comparison summaries.
- Update documentation and the source-test impact manifest.
- Keep V2 production-default selection until all gates pass.
- Switch the production default to V3 only after deterministic, integration,
  live-quality, performance, plan-review, code-review, and explicit human-
  owner sign-off gates pass.

## Deferred

- Deleting or renaming `cognition_core_v2`.
- Renaming `CognitionCoreInputV2`, `CognitionCoreOutputV2`,
  `CognitionCoreServicesV2`, `cognition_state.v2`, or downstream V2-shaped
  carriers.
- Moving shared deterministic V2-shaped substrate into a version-neutral
  package.
- Changing MongoDB documents, indexes, migrations, DB facade behavior, or
  state commit policy.
- Changing L3 text/preference/visual prompts, dialog generation,
  consolidation, reflection, scheduler, adapters, delivery, action handlers,
  task-resolution specialists, RAG, or web agents.
- New model routes, provider session APIs, persistent provider conversation
  identifiers, explicit LM Studio cache-control APIs, speculative decoding,
  batching, or synthetic prompt warm-up calls.
- New semantic evaluator/verifier LLMs or deterministic semantic correction.
- Dynamic chain construction, model-selected stage ordering, recursive agent
  spawning, arbitrary tools, or unbounded recurrence.
- V2 executor decommission and removal of the selector's `v2` value.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/cognition_core_v3/__init__.py`: exact V3 engine public
  entrypoint.
- `src/kazusa_ai_chatbot/cognition_core_v3/README.md`: V3 ownership, topology,
  public substrate, failure, cache, and testing ICD.
- `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py`: internal chain,
  stage, visibility, checkpoint, cache-domain, result, and failure contracts.
- `src/kazusa_ai_chatbot/cognition_core_v3/registry.py`: immutable chain and
  stage order with explicit dependencies and visibility.
- `src/kazusa_ai_chatbot/cognition_core_v3/transcript.py`: exact-prefix
  message assembly, accepted projection, repair-local invalid candidate,
  checkpoint scrub, route-domain comparison, and context-cap fitting.
- `src/kazusa_ai_chatbot/cognition_core_v3/execution.py`: bounded stage and
  parallel-chain execution, cancellation, global attempt-ledger integration,
  and typed result slots.
- `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py`: V3 appraisal prompt,
  micro-items, validation, trial reduction, accepted local state, and terminal
  outcome stage.
- `src/kazusa_ai_chatbot/cognition_core_v3/goal_cognition.py`: isolated
  ordinary, active, and required-selection goal producers with unchanged bid
  and relational-willingness output contracts.
- `src/kazusa_ai_chatbot/cognition_core_v3/workspace.py`: V3 complete-bid
  collapse and authoritative relational disposition.
- `src/kazusa_ai_chatbot/cognition_core_v3/action_selection.py`: V3 action
  planning plus fresh action/resolver authorization boundaries and unchanged
  output contracts.
- `src/kazusa_ai_chatbot/cognition_core_v3/diagnostics.py`: exact public V2
  diagnostic/observability projection plus protected V3 chain trace metadata.
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`: complete V3 orchestration
  and final exact output validation.
- `src/kazusa_ai_chatbot/cognition_core_selector.py`: closed process-level
  module selector with no runtime fallback.
- `tests/unit/cognition_core_v3/test_public_api.py`: exact V3 public export.
- `tests/unit/cognition_core_v3/test_contracts.py`: internal contract
  validation.
- `tests/unit/cognition_core_v3/test_registry.py`: topology, dependencies, and
  visibility.
- `tests/unit/cognition_core_v3/test_transcript.py`: prefix, scrub, checkpoint,
  cache-domain, and prompt-cap behavior.
- `tests/unit/cognition_core_v3/test_execution.py`: concurrency, attempts,
  cancellation, and bounded failure.
- `tests/unit/cognition_core_v3/test_appraisal.py`: six-family semantics,
  ordered chain handoff, typed disposition, and reduction.
- `tests/unit/cognition_core_v3/test_goal_cognition.py`: ordinary, active,
  required-selection, relationship, and sibling-isolation behavior.
- `tests/unit/cognition_core_v3/test_workspace.py`: complete-bid and
  authoritative relational collapse.
- `tests/unit/cognition_core_v3/test_action_selection.py`: planning,
  authorization, suppression, resolver, and selected-operation behavior.
- `tests/unit/cognition_core_v3/test_diagnostics.py`: public diagnostic parity
  and protected metadata.
- `tests/unit/cognition_core_v3/test_facade.py`: whole-engine ordering, parity,
  finalization, and partial-failure isolation.
- `tests/unit/test_cognition_core_selector.py`: selector and no-fallback tests.
- `tests/integration/cognition_core_v3/test_external_contract_parity.py`:
  public contract, target-path mapping, completion-order, and carrier parity.
- `tests/integration/cognition_core_v3/test_parent_checkpoint_guardrail.py`:
  parent replay arithmetic and effect isolation.
- `tests/integration/cognition_core_v3/test_relational_stance_propagation.py`:
  unchanged surface/dialog relational polarity.
- `tests/integration/cognition_core_v3/test_resolver_recurrence.py`: resolver
  round trip, typed evidence, and one final commit.
- `tests/integration/cognition_core_v3/test_engine_selector.py`: live/idle
  selected-engine convergence.
- `tests/integration/cognition_core_v3/test_self_cognition.py`: idle and group
  self-cognition selected-engine behavior.
- `tests/cognition_core_v3_comparison_harness.py`: effect-free raw evidence
  runner with sealed `capture-v2` and `replay-v3` modes, canonical input/hash
  enforcement, and independent engine input copies.
- `tests/fixtures/cognition_core_v3_baseline_manifest.json`: immutable path
  registry, exact scenario/node/case IDs, canonical inputs and hashes,
  V2 expected paths, predeclared V3 target paths, explicit path mappings and
  dispositions, environment/configuration fingerprint requirements, sample
  counts, and artifact schema.
- `tests/fixtures/cognition_core_v3_live_case_manifest.json`: fixed quality
  cases, source kind, hard gates, rubric, acceptable variation, forbidden
  failures, and trace requirements.
- `tests/test_cognition_core_v3_baseline_contract.py`: baseline eligibility,
  path completeness, hash integrity, sealed V2 immutability, and exact V3
  replay identity tests.
- `tests/test_cognition_core_v3_baseline_path_replay.py`: manifest-driven
  deterministic and patched-LLM V2 path capture plus same-scenario V3 target-
  path replay.
- `tests/test_cognition_core_v3_comparison_contract.py`: comparison-harness and
  manifest deterministic contract tests.
- `tests/test_cognition_core_v3_live_llm.py`: twenty-four separate live local
  model quality nodes.
- `tests/test_cognition_core_v3_performance_live_llm.py`: four separate
  cold/warm/cache-domain performance nodes.
- `tests/test_cognition_core_v3_docs.py`: static V3 documentation parity test.

### Modify

- `src/kazusa_ai_chatbot/config.py`: add closed engine selection; retain every
  current cognition model route and budget.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: import the
  selected `run_cognition`; preserve all other connector behavior.
- `tests/test_config.py`: exact selector configuration tests.
- `tests/unit/nodes/test_persona_supervisor2_cognition.py`: selected-engine,
  canonical-input, and one-commit connector tests.
- `tests/ownership/source_test_impact_manifest.json`: register every new and
  changed production owner with exact deterministic nodes.
- `README.md`: describe V2/V3 engine selection and V3 target topology without
  changing external architecture ownership.
- `docs/HOWTO.md`: document `COGNITION_CORE_ENGINE`, evaluation-only use, V3
  test execution, and default-cutover behavior.
- `src/kazusa_ai_chatbot/nodes/README.md`: replace V2-only connector wording
  with selected-engine wording while keeping input/output contracts exact.
- `development_plans/README.md`: register this active draft.

### Keep

- All production files under `src/kazusa_ai_chatbot/cognition_core_v2/`
  unchanged during V3 implementation and evaluation after the predecessor
  bugfix is completed and archived.
- `src/kazusa_ai_chatbot/cognition_resolver/` unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` unchanged.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py` unchanged.
- DB, action handlers, self-cognition, consolidation, reflection, scheduler,
  task resolution, RAG, adapters, and service entrypoints unchanged.
- Existing V2 deterministic, integration, live, and captured-replay tests
  retained as baseline evidence.

### Delete

- None in this plan.

## Test Impact And Traceability

| Source or governed artifact | Changed contract and semantic owner | Exact deterministic pytest node IDs | Supplemental integration/live node IDs | Mode | Regression prevented |
| --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/cognition_core_v3/__init__.py` | V3 public engine entrypoint | `tests/unit/cognition_core_v3/test_public_api.py::test_v3_exports_exact_engine_entrypoint` | `tests/integration/cognition_core_v3/test_external_contract_parity.py::test_v3_accepts_and_returns_exact_v2_shaped_contracts` | deterministic + patched integration | Public symbol or signature drift. |
| `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py` | Chain/stage/checkpoint/cache-domain contracts | `tests/unit/cognition_core_v3/test_contracts.py::test_chain_contracts_reject_unknown_fields_and_values` | none | deterministic | Unbounded or ambiguous internal state. |
| `src/kazusa_ai_chatbot/cognition_core_v3/registry.py` | Fixed topology and visibility | `tests/unit/cognition_core_v3/test_registry.py::test_registry_exposes_exact_appraisal_and_goal_topology` | `tests/integration/cognition_core_v3/test_external_contract_parity.py::test_stage_order_is_deterministic_under_completion_reordering` | deterministic + patched integration | Model-created stages, order drift, or sibling leakage. |
| `src/kazusa_ai_chatbot/cognition_core_v3/transcript.py` | Prefix continuation, repair scrub, checkpoint, cache domain, cap | `tests/unit/cognition_core_v3/test_transcript.py::test_accepted_same_domain_stage_extends_exact_prefix`; `tests/unit/cognition_core_v3/test_transcript.py::test_rejected_candidate_is_scrubbed_before_next_stage`; `tests/unit/cognition_core_v3/test_transcript.py::test_cache_domain_change_forces_canonical_checkpoint`; `tests/unit/cognition_core_v3/test_transcript.py::test_context_pressure_checkpoints_before_owner_cap` | `tests/test_cognition_core_v3_performance_live_llm.py::test_live_performance_warm_changed_tail` | deterministic + live performance | Cache-prefix churn, invalid-output pollution, cross-model hidden state, or overflow. |
| `src/kazusa_ai_chatbot/cognition_core_v3/execution.py` | Bounded chain executor and attempt/cancellation policy | `tests/unit/cognition_core_v3/test_execution.py::test_executor_runs_parallel_chains_and_serial_registered_stages`; `tests/unit/cognition_core_v3/test_execution.py::test_executor_preserves_global_attempt_caps`; `tests/unit/cognition_core_v3/test_execution.py::test_executor_cancels_owned_tasks_without_partial_effects` | `tests/integration/cognition_core_v3/test_parent_checkpoint_guardrail.py::test_v3_parent_replay_preserves_epoch_and_call_arithmetic` | deterministic + patched integration | Infinite loops, retry reset, task leaks, or duplicate effects. |
| `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py` | Six-family appraisal semantics, shared static prefix, and trial reduction | `tests/unit/cognition_core_v3/test_appraisal.py::test_v3_appraisal_preserves_six_family_domains`; `tests/unit/cognition_core_v3/test_appraisal.py::test_all_appraisal_chains_use_one_byte_identical_static_system_prompt`; `tests/unit/cognition_core_v3/test_appraisal.py::test_causal_and_epistemic_chains_expose_only_accepted_predecessor_context`; `tests/unit/cognition_core_v3/test_appraisal.py::test_terminal_outcome_runs_from_provisional_accepted_state`; `tests/unit/cognition_core_v3/test_appraisal.py::test_optional_stage_exhaustion_preserves_accepted_prefix` | `tests/test_cognition_core_v3_live_llm.py::test_live_event_agency_and_moral_chain`; `tests/test_cognition_core_v3_live_llm.py::test_live_goal_completion_terminalization` | deterministic + live LLM | Lost appraisal family, system-prefix fragmentation, wrong path/handle, bad chain handoff, or terminalization regression. |
| `src/kazusa_ai_chatbot/cognition_core_v3/goal_cognition.py` | Ordinary/active/required-selection goal bids, shared static prefix, and relational willingness | `tests/unit/cognition_core_v3/test_goal_cognition.py::test_all_goal_chains_use_one_byte_identical_static_system_prompt`; `tests/unit/cognition_core_v3/test_goal_cognition.py::test_ordinary_goal_remains_relational_willingness_owner`; `tests/unit/cognition_core_v3/test_goal_cognition.py::test_sibling_goal_transcripts_are_isolated`; `tests/unit/cognition_core_v3/test_goal_cognition.py::test_required_selection_preserves_fixed_roles_and_progress_evidence`; `tests/unit/cognition_core_v3/test_goal_cognition.py::test_required_goal_exhaustion_preserves_existing_fail_closed_contract` | `tests/test_cognition_core_v3_live_llm.py::test_live_required_selection_nested_roles`; `tests/test_cognition_core_v3_live_llm.py::test_live_relationship_boundary_high_attachment_abuse` | deterministic + live LLM | Goal-prefix fragmentation, wrong stance owner, sibling anchoring, role inversion, or unsafe acceptance. |
| `src/kazusa_ai_chatbot/cognition_core_v3/workspace.py` | Complete-bid collapse and relational authority | `tests/unit/cognition_core_v3/test_workspace.py::test_workspace_preserves_ordinary_relational_authority`; `tests/unit/cognition_core_v3/test_workspace.py::test_workspace_admits_only_complete_current_matter_bids` | `tests/integration/cognition_core_v3/test_relational_stance_propagation.py::test_v3_relational_stance_preserves_polarity_through_unchanged_surface_and_dialog` | deterministic + patched integration | Confidence ranking, stale-goal admission, or stance polarity loss. |
| `src/kazusa_ai_chatbot/cognition_core_v3/action_selection.py` | Action planning, goal resolution, and isolated authorizations | `tests/unit/cognition_core_v3/test_action_selection.py::test_non_accepting_stance_suppresses_effects`; `tests/unit/cognition_core_v3/test_action_selection.py::test_action_and_resolver_authorizers_receive_fresh_minimal_context`; `tests/unit/cognition_core_v3/test_action_selection.py::test_invalid_authority_proposal_denies_all_effects`; `tests/unit/cognition_core_v3/test_action_selection.py::test_selected_operation_and_goal_progress_are_preserved` | `tests/integration/cognition_core_v3/test_resolver_recurrence.py::test_v3_resolver_request_round_trip_preserves_typed_evidence_and_one_commit` | deterministic + patched integration | Unauthorized effects, boundary pollution, or resolver lifecycle drift. |
| `src/kazusa_ai_chatbot/cognition_core_v3/diagnostics.py` | Exact public diagnostics and protected chain metadata | `tests/unit/cognition_core_v3/test_diagnostics.py::test_v3_public_diagnostics_match_v2_contract`; `tests/unit/cognition_core_v3/test_diagnostics.py::test_protected_chain_metadata_excludes_secrets_and_rejected_content` | none | deterministic | Public output drift or protected-data leakage. |
| `src/kazusa_ai_chatbot/cognition_core_v3/facade.py` | Complete V3 orchestration and final output | `tests/unit/cognition_core_v3/test_facade.py::test_facade_runs_exact_two_appraisal_waves_and_goal_waves`; `tests/unit/cognition_core_v3/test_facade.py::test_facade_applies_relationship_maintenance_and_emotion_derivation_once`; `tests/unit/cognition_core_v3/test_facade.py::test_facade_preserves_complete_v2_output_shape`; `tests/unit/cognition_core_v3/test_facade.py::test_facade_isolates_partial_chain_failure_before_state_commit` | `tests/integration/cognition_core_v3/test_external_contract_parity.py::test_v3_preserves_emotion_relationship_goal_and_action_carriers`; `tests/test_cognition_core_v3_live_llm.py::test_live_multi_goal_competition` | deterministic + patched/live | Topology drift, duplicated state updates, missing features, or partial output commit. |
| `src/kazusa_ai_chatbot/cognition_core_selector.py` | Closed process-level engine selection and no fallback | `tests/unit/test_cognition_core_selector.py::test_selector_resolves_exact_v2_and_v3_modules`; `tests/unit/test_cognition_core_selector.py::test_selector_rejects_unknown_engine`; `tests/unit/test_cognition_core_selector.py::test_v3_failure_never_invokes_v2` | `tests/integration/cognition_core_v3/test_engine_selector.py::test_live_and_idle_connectors_share_one_selected_engine` | deterministic + patched integration | Per-turn switching, unknown imports, or automatic fallback. |
| `src/kazusa_ai_chatbot/config.py` | `COGNITION_CORE_ENGINE` closed startup setting | `tests/test_config.py::test_cognition_core_engine_accepts_only_v2_or_v3`; `tests/test_config.py::test_cognition_core_engine_default_matches_cutover_state` | none | deterministic | Invalid configuration or unintended default. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | Selected engine call with unchanged connector/commit | `tests/unit/nodes/test_persona_supervisor2_cognition.py::test_connector_calls_selected_engine_with_canonical_input`; `tests/unit/nodes/test_persona_supervisor2_cognition.py::test_selected_engine_output_commits_once` | `tests/integration/cognition_core_v3/test_parent_checkpoint_guardrail.py::test_v3_parent_replay_repeats_no_preparation_capability_or_commit`; `tests/integration/cognition_core_v3/test_self_cognition.py::test_idle_self_cognition_uses_selected_v3_engine` | deterministic + patched integration | Connector duplication, V2 hard-wiring, replay effects, or idle-path split. |
| `tests/cognition_core_v3_comparison_harness.py` | Effect-free sealed V2 capture and exact V3 replay owner | `tests/test_cognition_core_v3_baseline_contract.py::test_capture_v2_mode_imports_only_v2_engine`; `tests/test_cognition_core_v3_baseline_contract.py::test_replay_v3_requires_exact_baseline_identity`; `tests/test_cognition_core_v3_comparison_contract.py::test_comparison_uses_independent_inputs_and_disables_all_effects`; `tests/test_cognition_core_v3_comparison_contract.py::test_comparison_emits_raw_evidence_only` | four performance nodes and twenty-four quality nodes listed below | deterministic + live LLM | Baseline drift, shadow commits, action execution, or script-authored quality claims. |
| `tests/fixtures/cognition_core_v3_baseline_manifest.json` | Exhaustive registered-path mapping, artifact-integrity, and immutable evidence contract | `tests/test_cognition_core_v3_baseline_contract.py::test_baseline_manifest_covers_every_registered_v2_and_target_v3_path`; `tests/test_cognition_core_v3_baseline_contract.py::test_baseline_manifest_has_unique_scenario_case_and_engine_path_ids`; `tests/test_cognition_core_v3_baseline_contract.py::test_baseline_manifest_maps_every_v2_path_to_an_approved_v3_or_unchanged_path`; `tests/test_cognition_core_v3_baseline_contract.py::test_baseline_identity_hashes_all_governed_inputs_and_protocols`; `tests/test_cognition_core_v3_baseline_contract.py::test_sealed_baseline_index_resolves_every_required_artifact_hash`; `tests/test_cognition_core_v3_baseline_contract.py::test_v3_evidence_cannot_replace_v2_baseline_artifacts`; `tests/test_cognition_core_v3_baseline_path_replay.py::test_capture_v2_exercises_every_manifest_v2_path`; `tests/test_cognition_core_v3_baseline_path_replay.py::test_replay_v3_exercises_frozen_scenarios_and_every_target_path` | `tests/integration/cognition_core_v3/test_external_contract_parity.py::test_v3_replay_closes_target_paths_and_complete_v2_mapping` | deterministic + patched integration | Missing internal path, topology substitution without approval, historical evidence rewrite, untraceable artifact, or post-baseline corpus/rubric change. |
| `tests/fixtures/cognition_core_v3_live_case_manifest.json` | Fixed quality corpus and rubric | `tests/test_cognition_core_v3_comparison_contract.py::test_live_manifest_has_exact_case_ids_and_complete_review_contracts` | twenty-four live quality nodes listed below | deterministic + live LLM | Case drift, missing feature coverage, or prompt-shaped assertions. |
| `tests/ownership/source_test_impact_manifest.json` | V3 source-to-test enforcement | `tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary`; `tests/test_test_impact_manifest.py::test_manifest_accepts_an_explicit_package_init_source_root` | none | deterministic | Unmapped V3 production paths. |
| `src/kazusa_ai_chatbot/cognition_core_v3/README.md` | V3 ICD and parity map | `tests/test_cognition_core_v3_docs.py::test_v3_readme_maps_every_preserved_v2_capability` | none | static deterministic | Undocumented ownership or feature omission. |
| `README.md` | Top-level selected-engine architecture | `tests/test_cognition_core_v3_docs.py::test_root_readme_documents_v3_selection_and_unchanged_boundary` | none | static deterministic | Top-level architecture drift. |
| `docs/HOWTO.md` | Engine configuration and test runbook | `tests/test_cognition_core_v3_docs.py::test_howto_documents_engine_selection_and_one_case_live_workflow` | none | static deterministic | Unsafe or missing operator workflow. |
| `src/kazusa_ai_chatbot/nodes/README.md` | Connector ownership wording | `tests/test_cognition_core_v3_docs.py::test_nodes_readme_documents_selected_engine_and_exact_contracts` | none | static deterministic | V2-only connector documentation after hot-plug. |

## Fixed Live Quality Cases

`tests/test_cognition_core_v3_live_llm.py` contains these exact separate nodes.
During Gate 1 each node captures three V2 trials from its canonical input;
during Gate 3 the same node and harness commit capture three V3 trials from an
independent copy with the same canonical hash. All trials for one node are
inspected before another live node runs:

1. `test_live_event_agency_and_moral_chain`
2. `test_live_relationship_reciprocity`
3. `test_live_relationship_boundary_high_attachment_abuse`
4. `test_live_relationship_unestablished_intimate_request`
5. `test_live_goal_completion_terminalization`
6. `test_live_threat_resolution_and_relief`
7. `test_live_epistemic_comparison`
8. `test_live_memory_cue_nostalgia`
9. `test_live_existential_drive`
10. `test_live_ordinary_neutral_response`
11. `test_live_required_selection_nested_roles`
12. `test_live_required_selection_private_refusal`
13. `test_live_group_third_party_addressee`
14. `test_live_group_self_cognition_stays_silent`
15. `test_live_group_self_cognition_proposes_reply`
16. `test_live_resolver_observation_continuation`
17. `test_live_tool_result_answerability`
18. `test_live_future_speak_authority`
19. `test_live_current_message_prompt_injection_is_data`
20. `test_live_retrieved_evidence_prompt_injection_is_data`
21. `test_live_long_context_checkpoint`
22. `test_live_crying_sadness`
23. `test_live_verbal_abuse_boundary`
24. `test_live_multi_goal_competition`

The manifest labels each case as `production_trace`, `captured_failure`, or
`realistic_fixture`. Performance conclusions use only production-trace or
captured-production inputs. Realistic fixtures support semantic coverage and
cannot establish cache or latency performance.

## Performance Comparison Contract

`tests/test_cognition_core_v3_performance_live_llm.py` contains these exact
separate nodes:

1. `test_live_performance_cold_full_appraisal_turn`
2. `test_live_performance_warm_exact_repeat`
3. `test_live_performance_warm_changed_tail`
4. `test_live_performance_mixed_route_checkpoint`

Each node records, when exposed by the backend:

- baseline ID, engine, trial ID, case ID, canonical input hash, harness hash,
  environment fingerprint, route/configuration fingerprint, route, model,
  static-prompt hash, and cache-domain hash;
- cold/warm state and cache preparation procedure;
- request count and maximum engine concurrency;
- prompt characters/tokens and completion tokens;
- per-stage prompt-processing duration and TTFT;
- end-to-end cognition duration;
- validator rejection, repair, checkpoint, route-domain split, and exhausted
  counts; and
- exact input identity and raw artifact paths.

The cold node records five independent backend/model-load trials per engine.
The exact-repeat node records one real effect-free setup request and twenty
measured warm trials per engine. The changed-tail node records twenty measured
manifest-provided tail changes per engine. The mixed-route node records ten
measured route transitions per engine. The setup request is evidence and is
excluded from warm statistics. Gate 3 repeats the complete protocol for V2 and
V3 in separate cache epochs and alternating engine-order blocks. Performance
acceptance uses this contemporaneous V2 control; the Gate 1 historical timing
remains a drift reference and is labelled non-comparable when its environment
fingerprint differs.

Performance acceptance requires all of the following against the
contemporaneous V2 control on the fixed captured performance corpus:

- V3 warm aggregate prompt-processing duration is lower than V2.
- V3 warm exact-repeat and changed-tail median TTFT are no higher than V2.
- V3 cold first-stage median TTFT is at most 110% of V2.
- V3 end-to-end median and p95 cognition duration are at most 115% of V2.
- Every eligible same-domain continuation proves exact-prefix extension by
  message hash, regardless of whether the backend exposes a cache-hit counter.
- Mixed-route execution proves a checkpoint and correct output rather than
  pretending to retain a cross-domain KV prefix.
- A cache-disabled or cache-miss run produces output that passes the same
  deterministic contract and safety gates.

A failed performance gate keeps V2 as the production default and keeps this
plan `in_progress`. Remediation remains inside the V3 engine and may not weaken
feature, safety, isolation, or contract gates.

## Human-Readable LLM Review Contract

For every live quality and performance node:

1. The test/harness writes Gate 1 V2 raw evidence under the immutable `v2/`
   subtree, Gate 3 V3 evidence under `v3/`, and the contemporaneous V2 timing
   control under `comparison/current_v2_performance/`, exactly as declared by
   the artifact contract.
2. During Gate 1, the executing agent reads all three V2 quality trials, or
   the complete V2 performance sample set, and authors the baseline review.
   During Gate 3, it reads the complete matched V2 and V3 inputs, outputs,
   parsed state, stage traces, validation results, and timing evidence.
3. The agent writes the Gate 1 review to
   `test_artifacts/llm_reviews/cognition_v3/<baseline_id>/<case_id>/v2_baseline.md`
   and the Gate 3 review to the sibling `comparison.md`. Each contains Run
   Context, Evaluation Goal, Input, Output or side-by-side Output, Decision/
   Behavior Summary, Quality Notes, Validation, and Raw Evidence links.
4. Hard gates cover schema, role/target ownership, evidence grounding,
   privacy, permission, unavailable-action claims, relationship stance,
   required literals, and state invariants.
5. Each behavioral dimension receives `0=failed`, `1=acceptable`, or
   `2=strong`, with evidence from the displayed real output.
6. V3 acceptance requires no hard-gate failure, no `0` behavioral score, no
   safety/privacy/permission score below V2, and a V3 aggregate score at least
   equal to V2 inside each capability group: appraisal/state, relationship,
   goal/selection, action/resolver, group/self-cognition, and robustness. The
   overall V3 mean must be strictly greater than V2, and at least one of
   appraisal/state, goal/selection, or robustness must improve by at least
   0.20 on the 0-2 scale.
7. Schema validity or pytest success alone cannot accept a case.

## Verification Sequence

1. Verify the predecessor appraisal-recovery plan is completed and archived.
   Capture the clean post-fix V2 commit, worktree status, exact passing
   predecessor nodes, and owned file set.
2. Create and review the baseline path registry, case manifests, capture/
   replay harness, rubric, environment fingerprint, performance protocol, and
   artifact-integrity tests. Freeze their hashes before any V3 production edit.
3. Collect every exact V2 deterministic and patched node named by the path
   registry. Run the full V2 path matrix and resolve missing or unregistered
   observed paths until the V2 path-closure report reaches 100%.
4. Run each of the twenty-four V2 live-quality nodes separately, three trials
   per node, inspect all trial traces, and author the V2 baseline review before
   continuing to the next node.
5. Run the four V2 performance nodes separately under the fixed five/twenty/
   twenty/ten measurement protocol and review each artifact.
6. Seal and independently verify the V2 baseline index, hashes, path closure,
   live reviews, timing summary, known-defect register, and `baseline_id`.
7. Complete V3 production source and source-mirrored deterministic tests while
   production remains configured for V2 and the V2 package/baseline remain
   unchanged.
8. Run exact test-impact collection; resolve every new or changed production
   path to the exact deterministic nodes in this plan, then run those nodes.
9. Run unchanged V2 deterministic owner tests plus V3 patched integration
   tests for selector, connector, resolver, parent replay, state commit,
   relational stance, required selection, self-cognition, actions, and
   unchanged L3/dialog propagation.
10. Replay every sealed deterministic and patched baseline `scenario_id`
    through V3. Require exact input identity, 100% target V3 path closure,
    100% contract accuracy, and a complete approved mapping for every V2
    `path_id`.
11. Run each of the twenty-four V3 live-quality nodes separately, three trials
    per node, inspect every trace, complete blind rubric scoring, unblind, and
    author the side-by-side review before the next node.
12. Run each of the four performance nodes separately for alternating V2 and
    V3 blocks on the current environment, using separate cache epochs. Use the
    current V2 run for acceptance deltas and label historical V2 timing
    comparable or non-comparable from its environment fingerprint.
13. Produce and review the contract-accuracy, semantic-quality, robustness,
    and performance summaries. Resolve every failed gate through V3-only
    remediation and rerun every affected gate without changing the baseline.
14. Complete an independent code and evidence review against this plan, the
    baseline seal, implementation diff, exact test evidence, live reviews,
    performance evidence, and forbidden paths.
15. Obtain explicit human-owner V3 sign-off and production-default cutover
    authorization, then change the default to V3.
16. Rerun selector/config nodes, complete external-contract integration nodes,
    and run one ordinary, one relationship-boundary, and one resolver-
    continuation live smoke individually.
17. Record cutover evidence and retain explicit V2 selection for the later
    observation and decommission plan.

## Execution Roles

### Baseline Contract And Capture Owner

- Responsibility: enumerate the finite V2 path graph, create the governed
  manifests and raw-evidence harness, capture deterministic/live/performance
  V2 evidence, and seal the content-addressed baseline before V3 production
  implementation begins.
- Owned surface: `tests/cognition_core_v3_comparison_harness.py`,
  `tests/fixtures/cognition_core_v3_baseline_manifest.json`,
  `tests/fixtures/cognition_core_v3_live_case_manifest.json`,
  `tests/test_cognition_core_v3_baseline_contract.py`,
  `tests/test_cognition_core_v3_comparison_contract.py`,
  `tests/test_cognition_core_v3_baseline_path_replay.py`,
  `tests/test_cognition_core_v3_live_llm.py`,
  `tests/test_cognition_core_v3_performance_live_llm.py`, and raw V2 artifacts
  under the active `baseline_id`.
- Authority: edit those surfaces until Gate 1 seals the baseline, run the
  authorized V2 deterministic/patched/live/performance nodes, and correct an
  incomplete baseline only by creating a new version. It has read-only
  authority over the V2 production package and no authority to add V3
  production source during Gate 1.
- Applicable skills: `development-plan`, `test-style-and-execution`,
  `debug-llm`, `local-llm-architecture`, `py-style`, and `cjk-safety` when
  applicable to changed Python fixtures or harnesses.
- Capability floor: cognition V2 control-flow audit, deterministic path
  coverage, local-model trace capture, performance measurement, artifact
  hashing, privacy-safe evidence handling, and pytest collection.
- Independence requirement: baseline contracts and rubrics receive an
  independent seal review before implementation; after sealing, every role
  treats V2 artifacts and governed hashes as read-only.
- Acceptance output: immutable baseline index, 100% V2 path closure, exact
  collection evidence, V2 raw live/performance evidence, readable V2 quality
  reviews, and known-defect register.
- Gate: starts after predecessor completion, plan approval, and explicit
  baseline/test evidence execution authorization; exits only when Gate 1
  passes and the baseline is independently sealed.

### Architecture And Implementation Owner

- Responsibility: implement the exact V3 topology, selector, connector import,
  prompts, validators, tests, documentation, and cutover mechanics.
- Owned surface: V3 production package, selector, configuration, connector
  import, source-mirrored V3 tests, patched V3 integrations, source-test
  impact manifest, and documentation. The sealed baseline manifests, harness,
  V2 artifacts, agent-authored review artifacts, and independent-review
  evidence remain outside its write authority.
- Authority: edit the owned files, choose local code decomposition inside the
  fixed contracts, and run authorized deterministic/patched verification.
  It cannot change topology, public contracts, cutover policy, performance
  thresholds, semantic ownership, baseline identity, rubric, path registry,
  or deferred scope.
- Applicable skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `no-prepost-user-input`, `cjk-safety`, and
  `debug-llm` when preparing live evidence.
- Capability floor: senior Python async architecture, local chat-model prompt
  design, typed contracts, state reducers/FSMs, concurrency/cancellation,
  prompt caching, pytest, and protected trace handling.
- Independence requirement: none for implementation; it cannot provide final
  independent code sign-off.
- Acceptance output: scoped diff, updated checklist, exact deterministic and
  integration evidence, raw live/performance evidence, and resolved review
  findings.
- Gate: starts only after plan approval and explicit implementation
  authorization and Gate 1 baseline sealing; exits only when all
  implementation-owned acceptance criteria are evidenced.

### LLM Quality And Performance Reviewer

- Responsibility: run or inspect each authorized live node one at a time and
  author the required three-trial V2 baseline and side-by-side V2/V3 quality/
  performance reviews.
- Owned surface: raw ignored artifacts and agent-authored Markdown reviews
  under `test_artifacts`; read-only access to source, prompts, fixtures, and
  traces.
- Authority: judge pass/fail against the fixed rubric and report regressions;
  it cannot edit production code or lower gates.
- Applicable skills: `debug-llm`, `test-style-and-execution`,
  `local-llm-architecture`, and `character-test` only when a later approved
  execution explicitly uses real multi-turn character-service behavior.
- Capability floor: local-model output interpretation, character-brain
  judgment, prompt-injection analysis, relationship/boundary semantics,
  performance evidence, and trace inspection.
- Independence requirement: separate from the executor that implements any
  remediation resulting from its findings.
- Acceptance output: one complete readable artifact per node with all trial
  evidence, blind scores and unblinding record, plus capability-group and
  overall directional-improvement gate summaries.
- Gate: real model and required data are available; exits only after every
  reviewed node has a supported judgment and linked raw evidence.

### Independent Plan, Baseline, And Code Reviewer

- Responsibility: review the draft architecture and execution contract before
  approval, seal baseline integrity before implementation, then review code,
  tests, live evidence, performance evidence, and cutover readiness against
  the approved plan.
- Owned surface: read-only access to the complete diff and evidence.
- Authority: return a separate `PASS` or `FAIL` for plan approval, baseline
  sealing, and final code/evidence sign-off with severity-ranked findings; it
  cannot remediate its own findings.
- Applicable skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `debug-llm`, and `no-prepost-user-input`.
- Capability floor: independent senior review of async LLM pipelines, state
  contracts, permissions, retries, caching, concurrency, and test evidence.
- Independence requirement: different executor from the implementation owner
  and every remediation executor whose changes it signs off.
- Acceptance output: recorded plan review, baseline-seal review, and final
  code/evidence review, each with no unresolved critical or high finding; the
  final review includes an explicit cutover recommendation.
- Gate: reviews this complete draft before approval, seals the complete V2
  baseline before implementation, then reviews the complete implementation
  diff and comparison evidence; any remediation triggers a new independent
  sign-off pass over affected gates.

## Agent Autonomy Boundaries

The implementation owner may choose local function names, private helper
decomposition, test setup, and command order that preserve every fixed
contract and exact change surface. It may add no production path, model route,
state field, persisted record, compatibility adapter, fallback, semantic
evaluator, or deferred feature.

After Gate 1, no role may rewrite the sealed V2 evidence, path registry,
canonical inputs, rubric, protocol, or governed hashes. A discovered baseline
defect creates a new version and repeats Gate 1; a V3 failure is never resolved
by relabelling, deleting, or weakening the historical case.

A conflict between this plan and current source, an inability to preserve the
external contract, a required change outside the listed surface, a change to
chain topology/visibility, or a proposed gate reduction requires a plan
amendment and user decision before that work continues.

## Acceptance Criteria

- A content-addressed V2 baseline is sealed before the first V3 production
  edit and resolves every governed input, protocol, configuration, raw
  artifact, review, and source commit by hash.
- The finite V2 path registry covers every declared control-flow edge and
  terminal disposition, reports 100% evidence closure, and has no
  unregistered observed runtime edge.
- The frozen manifest predeclares every target V3 path and maps every V2 path
  as preserved, replaced by the approved V3 topology, or external and
  unchanged; no path remains unmapped.
- V3 replays the exact baseline scenario/case IDs, canonical input hashes,
  path mapping, harness, rubric, route/configuration fingerprint, and valid
  performance protocol without changing V2 historical evidence.
- A separate final comparison index hashes every V3 raw artifact, input-
  identity result, metric summary, readable review, and independent sign-off
  used for the cutover decision.
- The complete V2 functionality map has a tested V3 counterpart or an
  explicitly unchanged shared owner.
- The predecessor appraisal-recovery plan is completed and archived, and V3
  matches its typed repair/terminal disposition and evidence-preservation
  contract.
- V3 accepts and returns the exact V2-shaped public contract and passes the
  unchanged canonical validators.
- User/character state scope, one final compare-and-replace commit, resolver
  recurrence, parent replay, actions, L3, dialog, consolidation, and delivery
  remain externally unchanged.
- All twenty-one emotions and every native relationship/affinity dimension,
  maintenance rule, and projection remain deterministic and covered.
- All six appraisal families, fourteen goal kinds, ordinary relational
  willingness, required selection, group/self-cognition, action/resolver
  authorization, scheduled authority, and prompt-cap dispositions are
  preserved.
- The fixed chain topology, exact stage order, accepted-only visibility,
  rejected-output scrub, cache-domain checkpoint, and hard iteration limits
  pass deterministic tests.
- All appraisal chains share one byte-identical static appraisal system prompt
  and all goal chains share one byte-identical static goal system prompt;
  stage/branch facts remain in isolated dynamic tails.
- V3 has no import of a V2 semantic executor listed in Confirmed Decision 5.
- Selector configuration chooses one module per process and supplies no
  automatic fallback or dual execution.
- Every changed production path is present in the source-test impact manifest
  with collected exact deterministic nodes.
- Every deterministic and patched integration gate passes; deterministic
  contract accuracy, target V3 path closure, and V2-path mapping completeness
  all equal 100%.
- Every live trial passes its hard gates and human-review scoring contract;
  no capability-group mean regresses, the overall V3 quality mean is strictly
  greater than V2, and one predeclared objective group improves by at least
  0.20 on the 0-2 scale.
- Every performance gate passes.
- Documentation matches the implemented topology and operator workflow.
- Independent code review passes with no unresolved critical or high finding.
- The human owner explicitly signs off the V3 evidence package and authorizes
  the production-default change.
- Production-default cutover smoke evidence passes while explicit V2 selection
  remains available for the later retirement decision.

## Progress Checklist

- [ ] Plan approved and baseline/test evidence execution explicitly
      authorized.
- [ ] Predecessor V2 appraisal-recovery bugfix completed, archived, and frozen
      as the baseline commit.
- [ ] Baseline path registry, predeclared V2-to-target-V3 path mapping, case
      manifests, harness, rubric, timing protocol, environment fingerprint,
      and governed hashes reviewed and frozen.
- [ ] Every registered V2 deterministic and patched path exercised; V2 path
      closure is 100% with no unregistered observed edge.
- [ ] Twenty-four V2 quality cases captured at three trials each, individually
      inspected, and reviewed.
- [ ] V2 cold/warm/changed-tail/mixed-route performance baseline captured under
      the fixed sample protocol.
- [ ] V2 baseline index sealed and independently verified before any V3
      production-source edit.
- [ ] V3 production implementation explicitly authorized after baseline
      sealing.
- [ ] V3 contracts, registry, transcript, and executor complete.
- [ ] V3 appraisal, goal, workspace, action, diagnostics, and facade complete.
- [ ] Selector/config/connector integration complete with V2 still default.
- [ ] Source-mirrored deterministic tests and impact manifest complete.
- [ ] Patched integration and external-contract parity tests complete.
- [ ] Exact frozen baseline scenarios replayed through V3 with 100% target V3
      path closure, contract accuracy, V2-path mapping completeness, and
      input-identity proof.
- [ ] Twenty-four V3 quality cases captured at three trials each, individually
      inspected, blind-scored, unblinded, and compared with V2.
- [ ] Four V3 performance cases individually run against a contemporaneous V2
      control with separate cache epochs and alternating engine-order blocks;
      historical timing comparability is recorded.
- [ ] Final comparison index seals all V3 evidence, summaries, readable
      reviews, and sign-off records without modifying the V2 baseline index.
- [ ] Every feature, quality, safety, robustness, and performance gate passes,
      including the fixed positive directional-improvement threshold.
- [ ] Independent code review passes.
- [ ] Human-owner V3 sign-off and production-default authorization recorded.
- [ ] Production default changes to V3 and cutover smoke passes.
- [ ] Execution evidence and handoff for later V2 retirement recorded.

## Independent Plan Review

Before approval, the Independent Plan, Baseline, And Code Reviewer checks this
draft for complete V2 functionality coverage, exact V3 topology, public-
contract preservation, module-selector justification, local-model failure
handling, test traceability, exhaustive finite-path baseline closure,
predeclared V2-to-V3 path mapping, baseline immutability, exact scenario/input
replay identity, accuracy and directional-improvement gates, performance
validity, scope exclusions, and retirement separation. Findings are resolved
in the draft before status changes to `approved`.
