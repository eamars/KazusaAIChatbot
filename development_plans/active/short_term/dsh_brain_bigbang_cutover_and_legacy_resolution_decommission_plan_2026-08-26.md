# DSH Plan 3: Brain Big-Bang Cutover And Legacy Resolution Decommission

## Summary

- Goal: route production `task_resolution_request` through the accepted DSH
  Resolution Controller and delete the superseded post-selector resolution
  designs in one deployment boundary.
- Status: draft; coarse successor plan; not executable until Plan 2 closes and
  this document is refined and explicitly approved.
- Plan class: destructive Brain-path big-bang cutover and decommission.
- Governing architecture:
  `docs/architecture/dsh_integration_architecture.md`.
- Functional support at exit: Kazusa Brain invokes the DSH-only resolver for
  inline and background resolution, consumes the canonical mapped result,
  continues through cognition/dialog/delivery, and resumes durable DSH
  sessions after checkpoint or restart.
- Compatibility policy: no legacy caller, resolver, graph, checkpoint,
  fallback, alias, translation bridge, dual run, or backward-compatible
  vocabulary remains after deployment.
- Effort estimate: six functional work blocks. The exact deletion and test
  matrix is frozen from the completed Plans 1 and 2 implementation.

## Entry Conditions

Plan 3 may be promoted only when:

1. Plans 1 and 2 are completed and archived with all gates green.
2. The accepted DSH tool catalog and its authority manifests cover the
   functions required by production task resolution.
3. Real standalone evidence proves terminal, clarification, approval,
   checkpoint, cancellation, cold resume, sidecar restart, and fault behavior.
4. A current caller/callee/source/test/data/config inventory identifies every
   legacy post-selector resolution and coding surface.
5. The user approves the exact deployment drain, atomic route switch, deletion
   inventory, and final executable test matrix.

The refined plan is one executable cutover contract. It may adjust paths to the
post-Plan 2 repository, while retaining every functional gate below.

## Fixed Execution Ownership

This coarse draft authorizes planning only. Its post-Plan 2 refinement requires
explicit user approval and `in_progress` status before production execution.

| Role | Fixed owner | Responsibility |
|---|---|---|
| Architecture and closure | Parent agent | Owns cutover/deletion decisions, consolidated material review, gate interpretation, status, and final closure |
| Implementation and verification | The persistent `/root/dsh_implementation_worker` subagent on `gpt-5.6-luna`, `max` reasoning, normal execution speed | Owns all production/test edits, deletions, deployment fixtures, test execution, remediation, and pre-handoff self-review |

The Plan 1/2 Luna worker remains the sole implementation worker and work
proceeds one plan at a time. The parent stays read-only for production code and
test execution. Changing this binding requires a user-approved amendment.

There are at most two review iterations:

1. The worker delivers the complete cutover/decommission candidate and all
   mapped verification. The parent returns one consolidated material finding
   set.
2. When required, the worker resolves the entire set and reruns acceptance.
   The parent passes or blocks the plan.

Minor style, lint, collection, fixture, import, typo, and link defects are
worker self-review responsibilities before handoff. There is no third review
iteration.

Process evidence remains limited to working-tree status, owned paths, final
diff/deletion inventory, exact commands/results, inspected live deployment
cases, and gate decisions. Runtime security digests remain. General source,
workspace, and artifact hashing remains outside scope.

## Target Production Boundary

```text
adapter/debug client
    -> Kazusa Brain
    -> action selector
    -> task_resolution_request
    -> Resolution Controller
       -> durable ResolutionThread / DSH segment
       -> independent DSH sidecar
       -> accepted Plan 2 tools
       -> submit_resolution or runtime checkpoint/fault
    <- DSHResolutionExhaustV1
    -> TaskResolutionResultV1
    -> cognition
    -> dialog / action / background delivery
```

The Brain remains the caller and owns action selection, priority, scheduling,
approval and clarification UX, cognition, final wording, delivery, and durable
product state. DSH owns bounded semantic tool selection and evidence
resolution. The sidecar remains an independently managed process and DSH
source remains untouched.

## Big-Bang Cutover Rules

1. Update the Brain caller, Resolution Controller composition, result mapping,
   background resume path, tests, configuration, and ICD in one atomic scope.
2. Preserve `task_resolution_request` as the Brain-facing action name.
3. Map validated DSH terminal statuses to the canonical
   `TaskResolutionResultV1` statuses.
4. Map checkpointed exhaust to runtime-owned `deferred` and persist only the
   opaque ResolutionThread/segment/session/revision reference.
5. Route user clarification and approval responses back through normal Brain
   interpretation before explicit same-goal DSH continuation.
6. Keep background queue, priority, scheduler, dispatcher, adapter, and
   delivery authority in Kazusa.
7. Preserve the current shared-memory prewarm path and its ownership.
8. Delete the old task-resolution orchestration and every post-selector RAG,
   internal resolver, external resolver, specialist DAG, and standalone coding
   harness design replaced by the accepted DSH/tool path.
9. Retain only bounded leaf implementations that the Plan 2 tool gateway owns.
   Their old graph/router entry points and vocabularies are deleted.
10. Remove obsolete prompts, models, state/checkpoint DTOs, accepted-task
    payload fields, configuration, exports, tests, docs, manifests, scripts,
    and package dependencies in the same cutover.
11. Convert no legacy checkpoint or active graph state into DSH state. New work
    begins on the DSH contract.
12. Keep historical records only where audit or retention policy requires
    them; no runtime code reads them as a compatibility source.

## Deployment Drain

Before the atomic deployment, Kazusa stops admitting new legacy resolution and
coding work and reaches zero active legacy inline/background executions.
Remaining stragglers follow one explicitly approved terminal administrative
disposition before deployment. The cutover then installs the DSH-only route
and deletion set together.

This drain is an execution safety gate rather than a compatibility period.
After deployment there is one production route.

## Coarse Decommission Surface

The refined plan must enumerate exact files from these owners:

| Owner | Cutover disposition |
|---|---|
| Brain task-resolution caller | Replace implementation edge with the Plan 1 runtime/controller and Plan 3 result mapper |
| Legacy task-resolution orchestrator/state/specialists | Delete after all required leaf functions are owned by the Plan 2 gateway |
| Post-selector RAG and internal/external resolvers | Delete routing, graph, synthesis, prompt, state, export, and configuration surfaces; retain the explicitly accepted prewarm path |
| Legacy coding agent/harness | Delete runtime, workflow, approval glue, routing, prompt, test, configuration, and package surfaces replaced by DSH coding |
| Accepted tasks/background work | Replace legacy graph checkpoint payloads with opaque DSH thread/segment/session/revision references while retaining Kazusa scheduling authority |
| Cognition handoff | Consume canonical `TaskResolutionResultV1` only; keep cognition judgment and dialog wording ownership unchanged |
| Database | Add or update only DSH-backed task/thread references needed by the new route; retire legacy runtime reads and indexes according to the approved data disposition |
| Documentation and test ownership | Publish the DSH-only production boundary and remove legacy examples, test nodes, and manifest rows |

## Work Blocks And Effort Gates

| Block | Relative effort | Work | Independent completion gate |
|---|---|---|---|
| 1. Final ownership and drain contract | Medium | Freeze caller/callee/deletion/data/config inventory and operational drain | Every active legacy producer/consumer has one cutover disposition and zero legacy work remains at deployment |
| 2. Brain caller and result mapping | High | Connect `task_resolution_request` to the controller and map terminal/checkpoint/fault outcomes | Inline resolved/partial/clarification/approval/unavailable/failed/deferred paths reach cognition through one canonical result |
| 3. Background and lifecycle cutover | High | Move accepted-task/background resume, priority, cancellation, restart, and completion to opaque DSH thread/session references | Foreground promotion and background completion resume the same safe DSH session and retain Kazusa scheduling/delivery authority |
| 4. Legacy design deletion | High | Delete superseded task-resolution, post-selector RAG/internal/external resolver, specialist DAG, and coding harness surfaces | Static imports, package exports, config, tests, docs, and manifests contain no executable legacy route |
| 5. Production fault and behavior validation | High | Validate sidecar absence/restart, Mongo/SQLite recovery, duplicate lease, scope rotation, approval, dialog handoff, and adapter delivery | Real service tests pass on DSH only and faults remain typed without fallback to legacy code |
| 6. Final deployment and closure | Medium | Execute the approved drain/cutover rehearsal, final deterministic/live suites, docs, and closure audit | All release gates are green on the exact deployment candidate and the parent closes the DSH-only plan |

## Functional Release Gates

| Gate | Green condition |
|---|---|
| P3-G1 — Single production route | Every `task_resolution_request` reaches the DSH controller and no executable legacy route or fallback remains |
| P3-G2 — Brain ownership | Action selection, priority, scheduling, approval/clarification UX, cognition, final wording, adapter delivery, and product persistence remain Kazusa-owned |
| P3-G3 — Lifecycle continuity | Inline, foreground-to-background promotion, direct background, continuation, checkpoint, cancellation, process restart, and completion preserve the correct thread/session lineage |
| P3-G4 — Complete functional parity | The accepted Plan 2 tool catalog supplies the required old functions without retaining old resolver/DAG/coding designs |
| P3-G5 — Prewarm retained | The approved shared-memory prewarm path remains functional, tested, and independent of the deleted post-selector resolver graphs |
| P3-G6 — Decommission complete | Legacy code, prompts, checkpoints, config, exports, dependencies, tests, docs, and manifest ownership are absent or retained solely as non-runtime historical data under an explicit disposition |
| P3-G7 — Fault isolation | Sidecar failure, RPC fault, DSH crash repair, Mongo/SQLite recovery, lease conflict, scope/audience mismatch, and uncertain side effects fail or defer through typed DSH-era outcomes |
| P3-G8 — Real production flow | Real Brain-service cases reach cognition/dialog/delivery through the DSH-only resolver for inline, background, clarification, approval, restart, and failure scenarios |

All eight gates are release blockers. A fallback to any legacy resolver is a
failed gate, even if the visible answer appears correct.

## Verification Direction

The refined plan supplies exact node IDs for:

- strict caller/result/checkpoint contracts;
- inline and background Brain integration;
- action-selector ownership;
- continuation, audience/scope rotation, lease, and restart;
- cognition/dialog/delivery handoff;
- retained prewarm behavior;
- static legacy absence and dependency removal;
- database/index/config disposition;
- sidecar fault isolation;
- real MongoDB;
- real DSH LLM tool-resolution cases; and
- real service/adaptor end-to-end behavior.

Regular deterministic tests run in batches through
`venv\Scripts\python`. Each live DB, live LLM, and real-service case runs
individually with full output inspection. The final candidate also passes the
sidecar TypeScript build/tests and the repository deterministic suite.

## Out Of Scope

- A compatibility window, shadow route, dual execution, checkpoint converter,
  legacy fallback, or staged percentage rollout.
- DSH source modifications.
- Moving cognition, dialog, scheduling, approval UX, or adapter delivery into
  DSH.
- Removal or redesign of the accepted shared-memory prewarm path.
- Unrelated Brain, character, memory, dialog, adapter, or control-console
  changes.

## Closure

The Luna worker records the exact cutover/deletion inventory and all functional
evidence. The parent verifies that production has one DSH-only resolution route
and that every superseded design is absent, then alone marks and archives the
plan. Closure is a functional DSH cutover, not merely a successful build.
