# complex task resolver capability plan

## Summary

- Goal: Add and prove a distinct `complex_task_resolver` specialist module for
  complicated questions, then bring the L2d-facing capability contract forward
  before the comprehensive real LLM review.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: standalone specialist first, then a narrow
  user-approved L2d capability-contract cutover before comprehensive real LLM
  testing; broad live workflow enablement remains blocked until the user
  accepts the review evidence.
- Highest-risk areas: confusing this specialist with the existing cognition
  resolver, unbounded recursive decomposition, lower-layer retrieval freshness
  misuse, semantic collapse mistakes, LLM-only arithmetic, prompt bloat,
  response-path latency, review-time web/RAG/LLM dependency availability, and
  dialog treating task-resolver artifacts as persona wording.
- Acceptance criteria: the specialist returns a validated prompt-safe packet,
  L2d exposes the agreed semantic capability names, `public_answer_research`
  routes to the complex resolver, `local_context_recall` routes to existing
  RAG2 recall, comprehensive real LLM review artifacts show sufficient
  robustness, and broad live workflow enablement remains user-gated.

## Context

The existing `kazusa_ai_chatbot.cognition_resolver` package is the live
cognition recurrence controller. It initializes `ResolverCycleStateV1`, runs
the L1/L2/L2d cognition stack, executes one immediate
`ResolverCapabilityRequestV1` per cycle, records `ResolverObservationV1` rows,
and projects prompt-safe observations back into the next cognition pass.

This plan adds a separate specialist module. The new module is not the
cognition resolver, not a replacement for `stage_1_goal_resolver`, and not a
dialog generator. It is intended to become the L2d-selected public answer
research capability while RAG2 remains the local context recall capability.

The standalone validation path is:

```text
deterministic tests / standalone review harness
  -> complex_task_resolver builds or reuses a task graph
  -> specialist returns ComplexTaskResolutionPacketV1
  -> review artifact records graph, collapse decisions, lower-layer cache
     metadata if present, node outputs, final packet, and human judgment
  -> user explicitly decides whether L2d contract integration may begin
```

The agreed L2d integration path to bring forward before comprehensive real LLM
review is:

```text
L1/L2/L2d cognition
  -> L2d emits resolver_capability_request(kind=public_answer_research)
  -> cognition_resolver executes one bounded specialist call
  -> complex_task_resolver builds or reuses a task graph
  -> specialist returns ComplexTaskResolutionPacketV1
  -> ResolverObservationV1 stores prompt-safe packet summary
  -> next L1/L2/L2d pass sees the packet
  -> L2d selects speak or no-response/private finalization
  -> L3/dialog renders final character wording from the packet
```

The companion L2d capability is `local_context_recall`. It routes to existing
RAG2 through `run_rag_evidence_for_persona_state(...)` and owns local/private
memory, relationship, profile, and conversation recall. The former
L2d-visible `web_evidence` path collapses into `public_answer_research`.
The former L2d-visible `rag_evidence` path is renamed to
`local_context_recall` without renaming the underlying RAG2 package.

This integration contract may be implemented before the comprehensive real LLM
review only after explicit user instruction. Broad live enablement, rollout,
and any claim that the new resolver supersedes existing behavior remain blocked
until the review gate passes and the user accepts the evidence.

The user expectation driving this plan:

- complex questions become a top-down graph of tasks and subtasks;
- each specialist cycle addresses one active task node at a time;
- resolved node results are collected bottom-up into a final answer;
- semantically equivalent branches can collapse;
- same-run duplicate branches can reuse resolved graph results by collapse;
- retrieval caching and freshness are owned below the resolver by RAG,
  WebAgent, or source layers and may be surfaced as subagent metadata;
- the component should eventually be strong enough to reduce reliance on
  cognition-chain-driven multi-cycle reasoning for complex questions, but that
  replacement is not part of this plan.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing resolver capability
  contracts, complex-task prompts, LLM call budgets, graph decomposition,
  lower-layer cache metadata handling, RAG/tool handoff, or dialog handoff.
- `debug-llm`: load before running live LLM graph decomposition, collapse, or
  synthesis cases, and before writing human-readable quality artifacts.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files or tests that contain CJK
  string literals.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or final
  reporting.
- After signing off any major checklist stage, reread this entire plan before
  starting the next stage.
- Do not execute implementation steps while `Status` is `draft`.
  Implementation requires user approval and status `approved` or
  `in_progress`.
- Keep the new module distinct from `kazusa_ai_chatbot.cognition_resolver`.
  The existing cognition resolver remains the recurrence controller and owns
  `ResolverCycleStateV1`, pending HIL/approval, duplicate request blocking,
  max-cycle blocking, and resolver telemetry.
- Reuse existing typed dictionaries and validators where their semantics match.
  Do not force reuse when a current type would distort ownership or hide the
  task-graph contract.
- The new specialist must not produce final visible dialog. It returns a
  dialog-ready answer packet for review. L2d and L3/dialog decide whether and
  how to speak.
- The canonical L2d capability names are `public_answer_research` and
  `local_context_recall`. Do not introduce `answer_investigation` as a runtime
  capability name.
- `public_answer_research` must route to
  `complex_task_resolver.resolve_complex_task(...)`. It owns public/current/
  external/source-bound answer investigation, including work that previously
  appeared to L2d as `web_evidence`.
- `local_context_recall` must route to existing RAG2 via
  `run_rag_evidence_for_persona_state(...)`. It owns local memory,
  relationship, profile, conversation, and private/contextual recall.
- Do not keep `rag_evidence` or `web_evidence` as L2d-visible canonical
  capabilities after the integration cutover. No compatibility aliases,
  fallback mappers, or parallel vocabularies are allowed unless the user
  explicitly approves a different cutover policy.
- Passing deterministic tests, focused tests, or live LLM tests is not
  sufficient authorization for broad live enablement. The comprehensive real
  LLM review must be completed, the user must inspect or explicitly accept the
  evidence, and the user must instruct any rollout beyond the narrow L2d
  capability-contract path.
- The new specialist must not execute platform delivery, scheduler actions,
  database writes, filesystem operations, shell commands, arbitrary HTTP tools,
  or adapter callbacks.
- Deterministic code owns graph shape validation, node caps, depth caps,
  timeouts, status mapping, prompt-safe projection, and propagation of
  lower-layer cache metadata without interpreting it as resolver-owned reuse.
- LLM stages own decomposition, active-node semantic resolution, collapse
  judgment over bounded candidate sets, and bottom-up synthesis text.
- Arithmetic, scheduling math, budget math, weighted scoring, token/call
  estimates, duplicate-branch call-savings estimates, and benchmark
  normalization must be
  executed by a typed deterministic algorithmic subagent. LLM stages may request
  the operation and provide semantic context, but they must not be treated as
  the calculation authority.
- Do not add arbitrary expression execution, shell execution, notebook
  execution, or a free-form symbolic-math engine in Phase 1. The algorithmic
  subagent may evaluate a caller-prepared expression only after AST validation,
  with empty `__builtins__`, safe numeric helpers, and schema-validated request
  and result envelopes.
- Do not keyword-classify raw user input in Python to decide semantic task
  decomposition, task satisfaction, user intent, or final answerability.
- Do not encode test-case keywords, product names, case ids, expected node
  paths, expected statuses, expected final-answer text, or fixture-specific
  hints in deterministic code or runtime prompts. Review fixtures are
  human/harness metadata only; they must not be injected into model prompts or
  used to route, repair, or rewrite resolver output.
- Minimum viable answers, expected final answers, and
  performance-reference summaries in review fixtures are metadata only; they
  are for AI/human judgment after a run, not prompt input, routing input, or
  deterministic answer-rewrite input.
- Do not expose raw platform ids, database ids, adapter wire syntax, raw tool
  payloads, lower-layer cache keys, embeddings, raw prompt text, or internal
  worker names in cognition-visible resolver context.
- New specialist LLM calls must have hard caps, timeout behavior, and a
  documented context budget before approval.
- Before any live LLM inspection or comprehensive real LLM review, run and
  record a review-environment preflight covering the resolver LLM route,
  JSON repair route, web search/read availability, network access, and any
  RAG/Mongo/embedding dependency used by that review run. Missing dependencies
  must become explicit per-case blockers or skipped-live-evidence notes; they
  must not be counted as resolver quality passes.
- Live LLM tests must run one case at a time with output inspected.
- For each real LLM test, the parent agent must perform a per-case failure
  mode analysis before moving to another case. The analysis must consolidate
  the root cause from real run evidence, sit back and review the failure mode
  against the planned architecture and local-LLM constraints, then recommend
  either a focused fix or an architectural fix. The live test harness may write
  raw structured run evidence only; the parent agent authors the
  human-readable review in plan evidence or a separate review document. Do not
  advance to the next live case while this analysis is missing or unresolved.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` uses parent-led native subagent execution
  unless the user explicitly approves fallback execution.

## Must Do

- Create a new `kazusa_ai_chatbot.complex_task_resolver` package with its own
  README and public entrypoint.
- Keep the public complex-resolver entrypoint self-contained and callable
  through declared IO only.
- Document and implement the L2d-facing capability names as
  `public_answer_research` and `local_context_recall` before the comprehensive
  real LLM review, when the user instructs that implementation step.
- Keep all new task-graph contracts under the new module.
- Add `ComplexTaskGraphV1`, `ComplexTaskNodeV1`, and
  `ComplexTaskResolutionPacketV1` contracts with validators. The resolver must
  not define a semantic node-cache entry contract.
- Implement bounded graph traversal where each internal specialist iteration
  addresses one active task node.
- Add a standalone validation runtime boundary for stage LLM invokers, clocks,
  and structural limits. These runtime dependencies are not prompt-facing data
  and must not allow tests or review harnesses to configure internal subagent
  rosters, node routing, prompt variants, graph paths, or expected answers.
- Add a standard complex-task subagent IO contract under the new module,
  following the same public IO principle as `WebAgent3.run(...)`: a bounded
  task/request plus context enters the subagent, and a resolved/result/attempts
  plus lower-layer cache-metadata/trace envelope comes back.
- Add a bounded node-resolution loop inside the complex-task resolver. The
  outer graph traversal still selects one active node at a time, but each
  selected node may record prompt-safe attempt observations and retry with a
  refined action before the resolver advances to another node.
- The node-resolution loop must support general problem solving, not only web
  search refinement. Allowed next actions include refining evidence requests,
  disambiguating entities, expanding the graph, repairing subagent IO,
  reviewing source conflict, revising deterministic calculation requests,
  synthesizing a partial answer, asking for user-owned input, or blocking the
  node with explicit reasons.
- Add a bounded executable follow-up task channel inside the complex resolver.
  Prose `recommended_next_iteration` remains a semantic projection for
  cognition and review; deterministic code must execute only structured
  `followup_tasks` emitted by resolver-local LLM stages.
- If an active node produces structured local follow-up tasks, create bounded
  child nodes under that node and resolve them through normal graph traversal.
  Follow-up child creation must obey fixed `max_depth`, `max_nodes`, and
  per-source follow-up caps, and cap exhaustion must become semantic lacking
  knowledge rather than unbounded recursion.
- If bottom-up synthesis produces structured top-level follow-up tasks that
  are executable by resolver-owned subagents or decomposition, create bounded
  root-level follow-up nodes and continue traversal before returning the final
  packet. User clarification, unavailable external access, persona judgment,
  and final dialog decisions remain outside resolver ownership and must be
  returned semantically instead of executed.
- Prompt-facing resolver stages must receive compact prior-attempt summaries
  for the active node so they can avoid repeating the same failed action and
  can justify the next local action. These summaries are internal resolver
  observations, not fixture hints and not cognition-visible dialog text.
- Add an internally owned evidence subagent boundary for active `evidence_need`
  nodes. It may call existing public WebAgent3/RAG entrypoints through module
  code, records dependency availability, and returns bounded source-backed
  facts or explicit unavailable/blocker status.
- Add a deterministic algorithmic subagent for `algorithmic_task` nodes. It
  must cover Phase 1 arithmetic review cases without relying on LLM arithmetic:
  schedule/duration math, budget math, weighted scoring, token/call estimates,
  duplicate-branch call savings, percentage/range comparisons, and benchmark
  normalization over already-sourced numbers.
- Implement bounded semantic collapse over a small deterministic candidate set.
- Do not implement a resolver-owned semantic node cache. Same-run reuse is
  handled by graph collapse, and retrieval caching remains in RAG, WebAgent,
  or source layers.
- Provide a compact prompt-safe packet projection helper for review artifacts
  and Stage 5 `resolver_context` observation projection.
- Add focused deterministic tests for contracts, graph validation, same-run
  collapse behavior, lower-layer cache metadata passthrough,
  evidence-subagent unavailable paths, algorithmic subagent IO and operation
  outputs, packet projection, and standalone orchestration.
- Add live LLM inspection tests for decomposition and synthesis, marked
  `live_llm`, one case at a time, after the L2d-facing capability vocabulary
  and routing contract are in place.
- For Stage 5 L2d tool-selection quality, use real LLM tests focused on L2d
  route choice from frozen prompt-safe state to expected capability output.
  Do not treat patched LLM or deterministic tests as evidence that L2d will
  semantically pick the right tool. Deterministic tests in this stage exist
  only to remove or update legacy old-capability expectations and keep the
  code contract executable.
- Add the dedicated real-LLM review fixture:
  `tests/fixtures/complex_task_resolver_review_cases.json`.
- Add fixture coverage validation that compares all fixture `required_stages`,
  statuses, and case categories against the comprehensive review artifact.
- Add a comprehensive real LLM review artifact before any broad live workflow
  enablement is considered.
- Update docs for the new module, L2d capability vocabulary, and remaining
  rollout gate.

## Deferred

- Do not replace `stage_1_goal_resolver` or the existing cognition resolver
  recurrence in this plan.
- Do not remove `ResolverGoalProgressV1`, HIL, approval, or self-goal resolver
  capabilities in this plan.
- Do not move L1/L2/L2d cognition into the new module.
- Do not route `public_answer_research` to anything except the public complex
  resolver entrypoint.
- Do not route `local_context_recall` to anything except the existing RAG2
  persona evidence entrypoint.
- Do not add `answer_investigation` to any runtime enum, prompt, allowed
  capability list, resolver capability dispatcher, test fixture, or review
  artifact as a canonical capability name.
- Do not enable broad live workflow rollout before the comprehensive real LLM
  review is accepted by the user.
- Do not let the new module emit `ActionSpecV1` or select final surfaces.
- Do not add persistent MongoDB cache storage, new collections, new indexes, or
  data migrations in Phase 1.
- Do not integrate coding-agent execution, shell tools, filesystem writes, or
  background-work job execution into Phase 1.
- Do not use task-graph output as durable memory directly; consolidation
  remains the only durable write owner.
- Do not broaden RAG2 helper-agent contracts or rewrite RAG routing as part of
  this plan.
- Do not add a compatibility shim that makes task graph payloads look like
  `rag_result`.
- Do not add feature flags for alternate architectures unless a later plan
  explicitly approves rollout control.

## Cutover Policy

Overall strategy: bigbang L2d capability-vocabulary cutover after standalone
module hardening, with user-gated broad live workflow enablement.

| Area | Policy | Instruction |
|---|---|---|
| Existing cognition resolver | compatible | Preserve recurrence, state, pending, trace, and terminal behavior while changing only the declared capability vocabulary and dispatcher targets. |
| New complex-task module | additive | Add a separate package and public entrypoint. |
| Resolver capability enum | bigbang | Replace L2d-visible `rag_evidence` and `web_evidence` with `local_context_recall` and `public_answer_research`; do not keep aliases. |
| Resolver observation | additive | Add only the prompt-safe complex-task packet summary required for next-cycle cognition. Do not expose internal graph prompts or raw trace payloads. |
| L2d prompt | bigbang | Present positive affordance descriptions for `public_answer_research` and `local_context_recall`; remove canonical `rag_evidence` and `web_evidence` wording. |
| RAG | compatible | Keep the underlying RAG2 implementation and route `local_context_recall` through approved public/persona entrypoints. |
| Web evidence | bigbang | Collapse L2d-visible `web_evidence` into `public_answer_research`; WebAgent3 remains an internal evidence provider under complex resolver. |
| Dialog | compatible | Receive content through existing selected `speak` and L3 content-plan path. |
| Cache | lower-layer only | Add no resolver-owned cache, persistent storage, or migration in this plan. |
| Broad live enablement | deferred | Do not treat the integrated capability as accepted for rollout until comprehensive real LLM review passes and the user accepts the evidence. |

## Target State

The completed behavior before broad live enablement is:

```text
L2d selects one semantic resolver capability
  -> public_answer_research calls complex_task_resolver for public/current/
     external answer investigation
  -> local_context_recall calls existing RAG2 for local/private context recall
  -> complex_task_resolver decomposes and resolves a bounded graph when used
  -> structured internal follow-up tasks are executed inside the resolver when
     graph limits and ownership allow
  -> result packet states semantic knowledge known, still lacking, and
     recommended next directions without judging answerability
  -> review artifact records packet, graph path, collapse decisions,
     lower-layer cache metadata if present, and human judgment
  -> comprehensive real LLM review proves or rejects the integrated path
```

The new module is a specialist, not a second persona. Its public answer packet
is factual and structural:

- what was resolved;
- what could not be resolved;
- which facts are source-backed;
- which assumptions or inferences were used;
- which subtasks collapsed;
- what final answer content is safe for dialog to render.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| L2d capability names | Use `public_answer_research` and `local_context_recall` as the canonical L2d-visible resolver capability names. | The names describe semantic ownership without negative prompt rules or implementation jargon. |
| Public answer research | Route `public_answer_research` to `complex_task_resolver.resolve_complex_task(...)`, not to direct RAG or WebAgent calls. | The complex resolver owns multi-stage public/current answer investigation and may invoke WebAgent3 internally. |
| Local context recall | Route `local_context_recall` to existing RAG2 via `run_rag_evidence_for_persona_state(...)`. | RAG2 still owns private/local memory, relationship, profile, and conversation recall. |
| Capability collapse | Remove L2d-visible `web_evidence`; rename L2d-visible `rag_evidence` to `local_context_recall` through a bigbang prompt/contract update. | This collapses overlapping public evidence retrieval into answer research while preserving RAG2 for local context. |
| Integration stage order | Bring the L2d capability contract and narrow routing boundary before comprehensive real LLM review. | The live review should exercise the final L2d-facing contract instead of a standalone-only harness. |
| Module boundary | Create `kazusa_ai_chatbot.complex_task_resolver`. | Keeps it distinguishable from `cognition_resolver` and allows independent contracts/tests. |
| Existing type reuse | Reuse `EvidenceRefV1` where semantics match; document future compatibility with `ResolverCapabilityRequestV1` and `ResolverObservationV1` without changing them in Phase 1. | The standalone module should not distort current cognition resolver contracts before it is proven. |
| New graph type | Use a node map with stable ids, not nested dynamic dictionaries. | Collapse, provenance, and validation need stable node identity. |
| Final result | Return `ComplexTaskResolutionPacketV1`, not final dialog. | Dialog ownership remains in L3/dialog. |
| Cache boundary | The complex resolver has no resolver-owned semantic cache. Retrieval caching, freshness, keys, and invalidation stay in RAG, WebAgent, or source layers and may be reported only as subagent metadata. | A resolver-level semantic cache would duplicate RAG2/WebAgent ownership and create non-deterministic matching, freshness, and privacy risks. |
| Collapse | Run collapse only over deterministic candidate sets from existing graph nodes in the same resolver run. | Avoids broad semantic comparison across arbitrary history. |
| Tool use | Production code may call existing helper agents only through their declared IO, including WebAgent3 `run(task, context, max_attempts)`. | RAG/WebAgent remains evidence owner; the new resolver composes answers from evidence while L2d remains only the semantic capability selector. |
| Public options | Callers may provide only structural resolver limits through `ComplexTaskResolverOptionsV1`. Stage LLM invokers, clocks, prompt variants, subagent rosters, graph paths, expected answers, and node routing are not public IO. | The module must be production-tappable as a self-contained specialist rather than completed by a review harness. |
| Evidence subagent | Active `evidence_need` nodes use a production WebAgent3-backed subagent boundary that reports resolved, partial, unavailable, or failed evidence. | The fixture exercises web retrieval, but missing SearXNG/network/RAG dependencies must be explicit blockers returned by the internal subagent rather than hidden prompt behavior or harness-owned prefetch. |
| Subagent IO | All resolver subagents use one standard envelope: request/task plus context in, `resolved`, `status`, `result`, `attempts`, lower-layer `cache` metadata, and `trace` out. | This mirrors the existing WebAgent3 helper-agent principle while adapting it to typed task-graph nodes and deterministic review artifacts. |
| Node-resolution loop | Add a resolver-owned bounded loop around active-node resolution. Each pass appends a typed, prompt-safe node attempt observation and either commits a terminal node update, expands the node, invokes a subagent, or retries with a refined local action until the node-attempt cap is reached. | The current one-shot active-node resolver cannot recover from blocked search, ambiguous entities, malformed subagent IO, wrong calculation shape, source conflict, or premature partial answers. This reuses the cognition resolver's observation-recurrence idea without importing cognition-chain ownership. |
| Loop ownership | The graph traversal loop owns which node is active; the node-resolution loop owns local refinement; the evidence subagent may own bounded search/read refinement; the algorithmic subagent remains deterministic single-shot. | This keeps recursion, specialist IO, and deterministic calculation responsibilities separate and inspectable. |
| Executable follow-ups | Add structured `followup_tasks` as resolver-internal control data emitted by active-node and synthesis stages. Do not execute prose `recommended_next_iteration`. | The resolver should act on work it can handle cheaply, while preserving semantic projections for cognition and avoiding keyword-based or prose-parsing control flow. |
| Local follow-up placement | Active-node follow-up tasks become child nodes under the source node. | This preserves top-down graph semantics and lets parent nodes synthesize from their own follow-up children. |
| Top-level follow-up placement | Synthesis follow-up tasks become bounded children under the root node and trigger another traversal pass before final packet return. | This lets the standalone resolver consume its own actionable research directions instead of forcing the heavier cognition loop to retry with less context. |
| Follow-up limits | Enforce `max_depth`, `max_nodes`, and per-source follow-up caps; cap exhaustion is reported as lacking knowledge and boundary notes. | Recursive execution must be useful and inspectable without allowing infinite node creation. |
| Algorithmic calculations | Add a deterministic algorithmic subagent for arithmetic-like work. Do not rely on LLM reasoning for calculations. | The review fixture includes schedules, budgets, weighted scores, token/call estimates, duplicate-branch savings, and benchmark normalization; local LLM arithmetic is not reliable enough to own those outputs. |
| Expression evaluation | Reject arbitrary expression evaluators, shell execution, and broad symbolic-math engines. Permit only caller-prepared numeric expressions through AST validation, empty `__builtins__`, and safe numeric helpers. | The reference calculator shape is useful, but it must remain a deterministic subagent interface, not a generic execution capability. |
| Coding agent | Defer direct coding-agent integration. | Coding-agent execution has separate safety, file, and latency boundaries. |
| Replacement path | No replacement of the cognition resolver in Phase 1. | The new module first proves specialist value through standalone review. |

## Contracts And Data Shapes

### Standalone Request

Create a new standalone request type under `complex_task_resolver`:

```python
ComplexTaskResolverRequestV1 = {
    "schema_version": "complex_task_resolver_request.v1",
    "objective": str,
    "reason": str,
    "source": "test | review_harness | live_llm_review",
    "priority": "normal | review",
}
```

The request objective must describe the full complex question, not one narrow
retrieval slot. It must not include raw platform ids, adapter wire syntax, or
backend-only debug fields.

### L2d Capability Requests

The canonical L2d-visible resolver capabilities are:

```python
PUBLIC_ANSWER_RESEARCH_CAPABILITY = "public_answer_research"
LOCAL_CONTEXT_RECALL_CAPABILITY = "local_context_recall"
```

`public_answer_research` maps to the public complex resolver entrypoint.
`local_context_recall` maps to the existing RAG2 persona evidence entrypoint.
`answer_investigation`, `web_evidence`, and `rag_evidence` are not canonical
L2d-visible names after the cutover.

### Public Entrypoint

Create:

```python
async def resolve_complex_task(
    request: ComplexTaskResolverRequestV1,
    context: ComplexTaskResolverContextV1,
    options: ComplexTaskResolverOptionsV1 | None = None,
) -> ComplexTaskResolutionPacketV1:
    ...
```

Only tests and standalone review harnesses should call this entrypoint in
Phase 1. Other modules must not import internal planner, graph, prompt, or
collapse helpers.

### Standalone Options

Create a public structural-options object under
`complex_task_resolver`:

```python
ComplexTaskResolverOptionsV1 = {
    "schema_version": "complex_task_resolver_options.v1",
    "limits": dict[str, int],
}
```

The options object is optional and contains only structural limits such as
`max_iterations`, `max_nodes`, `max_depth`, `max_node_attempts`, and
`max_subagent_attempts`. It is not projected into prompts. Deterministic
service code owns applying these limits.

The options object must reject caller-supplied LLM stage invokers, clocks,
subagent registries, prompt variant selectors, node routers, expected graph
paths, or expected answers. Resolver-local stages and subagents are created
and selected by the module itself. Deterministic tests may monkeypatch internal
stage handlers to test graph plumbing, but real LLM tests and review harnesses
must use only declared public IO.

### Node Attempt Observations

Each `ComplexTaskNodeV1` owns a bounded attempt ledger:

```python
ComplexTaskNodeAttemptV1 = {
    "schema_version": "complex_task_node_attempt.v1",
    "attempt_index": int,
    "action": (
        "resolve_direct | expand_node | call_subagent | refine_search | "
        "disambiguate_entity | repair_subagent_request | "
        "revise_calculation_request | review_source_conflict | "
        "synthesize_partial | ask_user_input | block"
    ),
    "status": "planned | resolved | partial | blocked | cannot_answer | invalid",
    "input_summary": str,
    "result_summary": str,
    "blockers": list[str],
    "next_action": str,
}
```

The attempt ledger is prompt-facing only through compact semantic active-node
context that excludes graph identifiers, attempt counters, and operational
state fields.
It is not external configuration, not a caller-provided graph path, and not
fixture metadata. It exists so the resolver can say, for example, "the first
search was blocked; refine the query", "the entity is ambiguous; split it",
"the calculation request had prose instead of an expression; repair the IO",
or "the available branches conflict; expand a conflict-review child".

The loop terminates when a terminal node update, node expansion, or subagent
result is applied, or when the bounded attempt cap is exhausted. Exhaustion
must produce a blocked or partial node with the recorded blockers rather than
silently returning an empty answer.

### Executable Follow-Up Tasks

Resolver LLM stages may request bounded executable follow-up work only through
semantic `continuation_tasks`:

```python
{
    "objective": str,
    "work_type": "subtask | public_evidence | calculation | synthesis",
    "reason": str,
}
```

Deterministic service code maps those semantic tasks into the internal control
contract:

```python
ComplexTaskFollowupTaskV1 = {
    "schema_version": "complex_task_followup_task.v1",
    "objective": str,
    "kind": "subtask | evidence_need | algorithmic_task | synthesis",
    "reason": str,
}
```

Internal `followup_tasks` may appear only after deterministic mapping beside
an internal node update or node attempt, and in bottom-up synthesis processing
beside the semantic packet fields. Deterministic service code validates every
row before creating graph nodes.

Follow-up tasks are resolver-internal control data. The service must never
parse or execute prose from `recommended_next_iteration`. If a stage provides
only prose recommendations, those recommendations remain semantic output for
cognition and review. If a stage provides structured follow-up tasks and
limits allow execution, the service creates pending graph nodes from those
tasks and records trace events.

Active-node follow-up tasks are created as children of the active node.
Synthesis follow-up tasks are created as children of the root node. Both paths
must enforce graph limits and a per-source cap so one stage output cannot
create unbounded children. When limits reject a follow-up task, the source node
or final packet must preserve the rejected objective as lacking knowledge and
add an evidence-boundary note explaining the graph limit.

### Standalone Context

Create:

```python
ComplexTaskResolverContextV1 = {
    "schema_version": "complex_task_resolver_context.v1",
    "conversation_summary": str,
    "persona_context_summary": str,
    "time_context": dict[str, object],
    "available_evidence": list[EvidenceRefV1],
}
```

This context is a compact review/test input. It is not `GlobalPersonaState` and
must not carry raw platform ids or full backend state.

### Complex Task Subagent IO Contract

Create a standard resolver-local subagent interface. It follows the same
public IO principle as WebAgent3: callers provide a bounded task/request and
context, and the subagent returns a small resolved/result envelope with attempts
and cache metadata. It is adapted for complex-task graph nodes by using typed
request/result payloads rather than a free-form `str` result.

```python
class ComplexTaskSubagentV1(Protocol):
    async def run(
        self,
        task: ComplexTaskSubagentRequestV1,
        context: ComplexTaskSubagentContextV1,
        max_attempts: int = 1,
    ) -> ComplexTaskSubagentResultV1:
        ...

ComplexTaskSubagentRequestV1 = {
    "schema_version": "complex_task_subagent_request.v1",
    "node_id": str,
    "subagent": "algorithmic | evidence",
    "action": str,
    "objective": str,
    "payload": dict,
    "constraints": dict,
}

ComplexTaskSubagentContextV1 = {
    "root_question": str,
    "parent_chain_summary": str,
    "sibling_summaries": list[str],
    "available_evidence": list[EvidenceRefV1],
    "time_context": dict[str, object],
}

ComplexTaskSubagentResultV1 = {
    "schema_version": "complex_task_subagent_result.v1",
    "resolved": bool,
    "status": "resolved | partial | invalid | unavailable | failed",
    "result": dict,
    "attempts": int,
    "cache": {
        "enabled": bool,
        "hit": bool,
        "cache_name": str,
        "reason": str,
    },
    "trace": dict[str, object] | list[str],
    "unresolved_items": list[str],
}
```

Subagent rules:

- The graph/service layer selects subagents from the module-owned internal
  registry.
- Subagents must not receive fixture expected answers, expected traces,
  failure-mode hints, raw platform ids, backend credentials, or runtime route
  handles.
- Subagents may receive node ids and bounded graph summaries because those are
  internal review artifacts, not user-facing evidence.
- `max_attempts` is structural and deterministic. The algorithmic subagent
  ignores retries and runs once; evidence may use bounded attempts only inside
  its provider contract.
- The result `trace` must be concise, deterministic, and safe for review
  artifacts. It must not contain hidden chain-of-thought or raw prompts.

### Deterministic Algorithmic Subagent

Create an `AlgorithmicSubagent` under the new module. It owns arithmetic-like
execution for `algorithmic_task` graph nodes and must not call an LLM. The
subagent only evaluates a caller-prepared numeric expression. The caller node
owns unit interpretation, unit conversion, operation selection, and expression
construction before invoking the subagent.

```python
ComplexTaskSubagentRequestV1 = {
    "schema_version": "complex_task_subagent_request.v1",
    "node_id": str,
    "subagent": "algorithmic",
    "action": "evaluate_expression",
    "objective": str,
    "payload": {
        "expression": str,
        "label": str,
    },
    "constraints": dict,
}

ComplexTaskSubagentResultV1 = {
    "schema_version": "complex_task_subagent_result.v1",
    "resolved": bool,
    "status": "resolved | partial | blocked | invalid | failed",
    "result": {
        "label": str,
        "expression": str,
        "result_repr": str,
        "result_str": str,
        "result_type": str,
        "display": str,
    },
    "attempts": int,
    "cache": dict,
    "trace": dict,
    "unresolved_items": list[str],
}
```

Supported behavior:

- `evaluate_expression`: evaluate one AST-validated Python expression using
  numbers, arithmetic operators, safe math helpers, `Decimal`, `Fraction`,
  public `math` attributes, and public `statistics` attributes.
- Return only a numeric or boolean result envelope. Reject non-numeric results,
  unknown names, private attributes, assignment, imports, lambdas,
  comprehensions, loops, and unsupported call targets.
- Preserve the expression and result in the subagent envelope so graph nodes
  can summarize deterministic arithmetic without hidden LLM reasoning.

Numeric safety:

- Validate the expression AST before evaluation and execute with empty
  `__builtins__`.
- Do not import modules dynamically, spawn subprocesses, use notebooks, call
  shell tools, call external math services, or call an LLM.
- Do not perform unit conversion or infer qualitative scoring rubrics inside
  the algorithmic subagent. If units or scoring are ambiguous, the caller node
  must block or create a prerequisite node instead of sending a guessed
  expression.

### Review Evidence Provider

Create a review-only evidence subagent/protocol under the new module. The core
graph may request evidence for an active `evidence_need` node, but only the
standalone service/review harness may satisfy that request through the standard
subagent envelope:

```python
ComplexTaskEvidenceRequestV1 = {
    "schema_version": "complex_task_evidence_request.v1",
    "node_id": str,
    "query": str,
    "evidence_kind": "web | rag | supplied_context",
    "freshness_requirement": "stable | session | volatile",
}

ComplexTaskEvidenceResultV1 = {
    "schema_version": "complex_task_evidence_result.v1",
    "status": "resolved | partial | unavailable | failed",
    "summary": str,
    "source_backed_facts": list[str],
    "evidence_refs": list[EvidenceRefV1],
    "unresolved_items": list[str],
    "dependency_status": dict[str, str],
}
```

The provider may call `WebAgent3` or approved RAG public entrypoints only from
tests or review harnesses. It must not call adapters, scheduler, dispatcher,
L2d, dialog, shell tools, coding-agent execution, or arbitrary MCP tools. If
`SEARXNG_URL`, network access, model route configuration, MongoDB, embeddings,
or another selected dependency is unavailable, it returns
`status="unavailable"` or `status="partial"` with `dependency_status` details
instead of fabricating evidence.

### ComplexTaskNodeV1

```python
{
    "schema_version": "complex_task_node.v1",
    "node_id": str,
    "parent_id": str | None,
    "depth": int,
    "objective": str,
    "node_kind": (
        "root | subtask | evidence_need | algorithmic_task | synthesis"
    ),
    "status": (
        "pending | expanded | resolving | resolved | blocked | "
        "cannot_answer | collapsed"
    ),
    "children": list[str],
    "result_summary": str,
    "answer_text": str,
    "cannot_answer_reason": str,
    "source_backed_facts": list[str],
    "assumptions_or_inferences": list[str],
    "evidence_refs": list[EvidenceRefV1],
    "source_observation_ids": list[str],
    "collapsed_into": str | None,
    "attempts": list[ComplexTaskNodeAttemptV1],
}
```

### ComplexTaskGraphV1

```python
{
    "schema_version": "complex_task_graph.v1",
    "root_node_id": str,
    "active_node_id": str,
    "nodes": dict[str, ComplexTaskNodeV1],
    "traversal_order": list[str],
    "collapse_events": [
        {
            "from_node_id": str,
            "to_node_id": str,
            "reason": str,
        }
    ],
    "max_nodes": int,
    "max_depth": int,
}
```

Validation rules:

- every `node_id` is unique;
- `root_node_id` and `active_node_id` exist;
- every parent and child reference exists;
- no cycle is allowed;
- depth is bounded by `max_depth`;
- node count is bounded by `max_nodes`;
- collapsed nodes must point to an existing non-collapsed node;
- answer-bearing nodes must include `status="resolved"` or
  `status="cannot_answer"`;
- raw ids and backend terms are rejected from prompt-facing summaries.

### ComplexTaskResolutionPacketV1

```python
{
    "schema_version": "complex_task_resolution_packet.v1",
    "root_question": str,
    "investigation_summary": str,
    "knowledge_we_know_so_far": list[str],
    "knowledge_still_lacking": list[str],
    "recommended_next_iteration": list[str],
    "evidence_boundary_notes": list[str],
    "graph": ComplexTaskGraphV1,
    "trace_summary": dict[str, object],
}
```

This packet is a semantic evidence bundle, not an answerability judgment.
`recommended_next_iteration` gives evidence directions for cognition to
consider, not a command to continue searching. Resolver-owned continuation is
represented only by validated internal `followup_tasks`, which are not exposed
as final packet control instructions.

### Cache Ownership Boundary

The complex resolver must not define `ComplexTaskCacheEntryV1`, node cache
keys, freshness labels, or a resolver-owned semantic cache. Similar research
tasks do not have deterministic semantic matching at this layer, and RAG2 plus
source agents already own retrieval caching and invalidation where those
concepts are meaningful.

Resolver behavior is:

- same-run duplicate answers are handled by graph collapse over bounded
  candidates;
- lower-layer cache metadata may appear in `ComplexTaskSubagentResultV1.cache`
  for trace/debug review;
- the resolver does not decide source freshness from cache keys;
- raw lower-layer cache keys are not projected to cognition-visible context;
- if evidence caching needs to improve, change RAG/WebAgent/source-layer code,
  not `complex_task_resolver`.

### ResolverObservationV1 Projection

`public_answer_research` observations carry a prompt-safe
`knowledge_projection` copied from the complex resolver packet:

- investigation summary;
- knowledge known so far;
- knowledge still lacking;
- recommended next iteration;
- evidence boundary notes.

The observation `status` remains a capability-execution status only. It must
not encode whether the original user goal is answered, partial, or unresolvable.
Cognition reads the semantic projection and decides whether to answer, request
more evidence, ask the user, or stop.

## LLM Call And Context Budget

Default context cap: 50k tokens.

Standalone Phase 1 maximum:

| Stage | Calls | Context inputs | Cap policy |
|---|---:|---|---|
| Graph planner | 1 | original objective, compact review context, optional relevant RAG summary | input under 18k chars; output max nodes 8 |
| Active node resolver | 0-3 | one active node, parent chain, sibling summaries, existing evidence packet | at most 3 active node attempts per capability call |
| Algorithmic subagent | 0 LLM | typed operation payload and bounded semantic context | deterministic execution only; invalid/missing operands return invalid or partial |
| Evidence subagent | 0-2 | active evidence node query plus compact semantic context only | at most 2 WebAgent3/RAG evidence calls per resolver run; missing dependency returns partial/unavailable |
| Collapse reviewer | 0-2 | active node and deterministic same-run candidate nodes only | at most 2 collapse checks; no broad search |
| Bottom-up synthesizer | 1-2 | resolved leaf summaries, unresolved items, facts, assumptions, and optional structured follow-up tasks | input under 16k chars; second call only after executable synthesis follow-ups are added and processed |

Hard Phase 1 caps:

- `max_nodes=8`
- `max_depth=3`
- `max_iterations=4`
- `max_rag_calls=2`
- `max_node_attempts=3`
- structured follow-up tasks obey `max_nodes`, `max_depth`, and per-source
  follow-up caps
- one JSON repair pass per LLM stage only when parsing fails
- bounded node-resolution retries only through typed node-attempt observations;
  no unbounded semantic retry or answer-text polishing loop

If the specialist cannot complete inside these caps, return
semantic known/lacking knowledge, recommendations, and boundary notes rather
than expanding the review run beyond its caps.

### Review Environment Dependencies

Before live inspection or comprehensive review, the review harness must record:

- resolver stage LLM route/config used for planner, node resolver, collapse,
  and synthesis;
- JSON repair route/config if parser repair is enabled;
- whether `web_read` direct URL reads are available from the Kazusa process;
- whether `web_search` is registered through `SEARXNG_URL`;
- whether outbound network access is available for the requested public
  sources;
- whether MongoDB and embeddings are available when the review selects full
  RAG2 retrieval rather than WebAgent3-only evidence;
- whether any generic MCP server is enabled. MCP must be recorded as absent or
  unused for this plan unless a later user-approved plan explicitly adds it to
  a resolver subagent.

Missing dependencies do not authorize deterministic hints, fixture-answer
injection, broader retries, arbitrary browser automation, or live workflow
integration. They only authorize a per-case `partial`, `cannot_answer`,
`needs_user_input`, or dependency-blocked review result with the blocker
recorded in the artifact.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/complex_task_resolver/README.md`
  - Owns the new specialist ICD and states how it differs from
    `cognition_resolver`.
- `src/kazusa_ai_chatbot/complex_task_resolver/__init__.py`
  - Exports only public contracts and `resolve_complex_task`.
- `src/kazusa_ai_chatbot/complex_task_resolver/contracts.py`
  - Defines graph, node, packet, subagent envelopes, validators, and projection
    helpers.
- `src/kazusa_ai_chatbot/complex_task_resolver/graph.py`
  - Owns deterministic graph mutation, traversal, active-node selection, and
    bottom-up collection helpers.
- `src/kazusa_ai_chatbot/complex_task_resolver/subagents.py`
  - Owns resolver-local evidence subagent wrapping, dependency failure result
    shape, and unavailable/partial evidence handling.
- `src/kazusa_ai_chatbot/complex_task_resolver/algorithmic.py`
  - Owns deterministic algorithmic subagent operations for duration, schedule,
    budget, weighted score, token/call, duplicate-branch call-savings,
    percentage/range, and benchmark-normalization calculations.
- `src/kazusa_ai_chatbot/complex_task_resolver/stages.py`
  - Owns graph-planning, active-node resolution, collapse-review, and
    bottom-up synthesis prompts plus JSON parsing.
- `src/kazusa_ai_chatbot/complex_task_resolver/service.py`
  - Owns the public orchestration entrypoint.
- `tests/helpers/complex_task_resolver_review.py` or equivalent local harness
  helper
  - Builds review inputs and writes raw structured run-evidence artifacts
    without wiring runtime cognition. Human-readable quality review remains
    agent-authored after inspection.
- `tests/test_complex_task_resolver_contracts.py`
- `tests/test_complex_task_resolver_graph.py`
- `tests/test_complex_task_resolver_evidence.py`
- `tests/test_complex_task_resolver_algorithmic.py`
- `tests/test_complex_task_resolver_service.py`
- `tests/test_complex_task_resolver_live_llm.py`
- `tests/fixtures/complex_task_resolver_review_cases.json`
  - Stores synthetic review cases, expected observable graph traces, final
    answer contracts, and anti-cheat metadata. The fixture is not prompt
    input.
- `test_artifacts/complex_task_resolver/`
  - Stores comprehensive real LLM review artifacts when the user instructs
    live review.

### Modify

- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
  - Replace canonical L2d-visible `rag_evidence` and `web_evidence`
    capability names with `local_context_recall` and
    `public_answer_research`.
- `src/kazusa_ai_chatbot/cognition_chain_core/action_selection.py`
  - Replace resolver affordance descriptions with positive semantic
    descriptions for `public_answer_research` and `local_context_recall`.
- `src/kazusa_ai_chatbot/cognition_chain_core/action_selection_prompt.py`
  - Keep capability selection data-driven from
    `capabilities.resolver_affordances`; update stable prose only if needed to
    remove stale `web_evidence`/`rag_evidence` assumptions.
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
  - Route `public_answer_research` to
    `complex_task_resolver.resolve_complex_task(...)`.
  - Route `local_context_recall` to existing
    `run_rag_evidence_for_persona_state(...)`.
- `src/kazusa_ai_chatbot/cognition_resolver/state.py`
  - Project only prompt-safe complex-task packet summaries needed by the next
    cognition pass. Do not project raw graph prompts, raw traces, source-cache
    keys, or internal subagent details.
- `src/kazusa_ai_chatbot/cognition_resolver/README.md`
  - Document the new L2d-visible capability names and routing ownership.
- `src/kazusa_ai_chatbot/cognition_chain_core/README.md`
  - Document the L2d capability vocabulary if the current README names the old
    evidence capabilities.
- `development_plans/README.md`
  - Register this draft plan while it remains under discussion.
- Documentation must distinguish the narrow L2d contract integration from
  broad live workflow enablement.

### Broad Live Enablement - Deferred

These changes are intentionally outside Phase 1:

- Turning the integrated path into an accepted production replacement for all
  public/current evidence routing.
- Deprecating RAG2 itself or removing the underlying RAG2 entrypoint.
- Removing `local_context_recall` after `public_answer_research` exists.
- Routing coding-agent execution through the complex resolver.
- Letting dialog consume raw graph state or subagent traces.

### Keep

- Keep `stage_1_goal_resolver` as the live persona resolver entrypoint.
- Keep existing RAG2, dialog, consolidation, adapter, scheduler, and
  background-work behavior unchanged.
- Keep `ResolverGoalProgressV1` for cognition-maintained goal progress. The
  new task graph does not replace it in Phase 1.

## Overdesign Guardrail

- Actual problem: complicated user questions need a bounded specialist that can
  decompose subtasks, collect results, collapse duplicate same-run paths, and
  return a complete answerability packet before dialog.
- Minimal change: add one separate specialist module with typed graph
  contracts, then update the existing L2d resolver capability vocabulary and
  dispatcher to call either the complex resolver or existing RAG2.
- Ownership boundaries: `complex_task_resolver` owns public/current answer
  investigation; RAG2 owns local/private context recall; L2d owns semantic
  capability choice; cognition resolver owns bounded capability execution and
  observation projection; dialog owns final wording.
- Rejected complexity: resolver-owned semantic cache, direct coding-agent
  execution, arbitrary tools, action-spec emission, scheduler work, adapter delivery,
  replacing `stage_1_goal_resolver`, broad prompt rewrites, compatibility
  aliases for old capability names, unbounded recursion, and multi-minute
  agent loops.
- Evidence threshold: add deferred complexity only after deterministic and
  live LLM evidence shows Phase 1 cannot answer target complex questions within
  caps while preserving source-backed facts and dialog handoff quality.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper mechanics only when they
  preserve the public contracts in this plan.
- The responsible agent must not introduce alternate architectures, fallback
  call paths, persistent storage, compatibility shims, or extra features.
- Changes outside `complex_task_resolver`, cognition resolver capability
  contracts/dispatcher/projection, L2d action-selection capability
  descriptions, tests, review artifacts, and plan documentation are out of
  scope unless the user explicitly approves them.
- The responsible agent must search for existing validators, JSON parsers,
  evidence refs, prompt projection helpers, and RAG entrypoints before adding
  new equivalents.
- If an existing type has matching semantics, reuse it. If semantics differ,
  create a new type and document why reuse was rejected.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, broad refactors, or prompt rewrites.
- If this plan and code disagree, preserve this plan's stated architecture and
  report the discrepancy before changing production code.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Parent adds focused contract tests.
   - Files:
     - `tests/test_complex_task_resolver_contracts.py`
     - `tests/test_complex_task_resolver_graph.py`
     - `tests/test_complex_task_resolver_evidence.py`
     - `tests/test_complex_task_resolver_algorithmic.py`
   - Expected before implementation: fail with missing module or missing
     contract symbols.

2. Parent starts one production-code subagent after focused tests exist.
   - Ownership boundary: production code under
     `src/kazusa_ai_chatbot/complex_task_resolver/` only.

3. Production-code subagent creates the new module contracts, graph helpers,
   standard subagent IO, review-only evidence subagent, and deterministic
   algorithmic subagent.
   - Implement validators before LLM prompts.
   - Verify contract, graph, evidence, and algorithmic tests pass before prompt
     integration.

4. Parent adds standalone service and review-harness tests.
   - Files:
     - `tests/test_complex_task_resolver_service.py`
     - review harness helper tests if the harness is not covered by service
       tests.
   - Expected before implementation: fail because the public orchestration
     entrypoint and review artifact writer do not exist.

5. Production-code subagent implements planner, active-node resolver, collapse,
   subagent dispatch, and synthesizer prompts.
   - Use static system prompts, current-run human payloads, and existing JSON
     parsing helpers.
   - Keep prompt projection compact and role-neutral.

6. Parent runs focused deterministic tests, service tests, review-harness
   tests, review dependency preflight tests, prompt-render checks, and static
   integration-boundary greps.

7. Parent implements the narrow L2d capability-contract cutover after explicit
   user instruction.
   - Replace L2d-visible `rag_evidence`/`web_evidence` with
     `local_context_recall`/`public_answer_research` in resolver contracts and
     action-selection affordances.
   - Route `local_context_recall` to existing RAG2.
   - Route `public_answer_research` to the complex resolver public entrypoint.
   - Update or remove legacy L2d tests that invoke `rag_evidence`,
     `web_evidence`, or `answer_investigation` as valid L2d capability output.
   - Add or update real LLM L2d tests for direct route-choice evidence from
     frozen prompt-safe input to expected resolver capability output.
   - Do not add compatibility aliases or old-name fallback mappers.

8. Parent adds live LLM inspection cases against the L2d-facing contract.
   - File: `tests/test_complex_task_resolver_live_llm.py`
   - Case source:
     `tests/fixtures/complex_task_resolver_review_cases.json`
   - Cases must run one at a time and produce review artifacts.
   - Runtime prompts must receive only the user question and normal review
     context, never expected graph traces, expected statuses, expected final
     answers, or forbidden failure modes from the fixture.

9. Parent performs the comprehensive real LLM review only after the user
   explicitly instructs the review.
   - Required artifact:
     `test_artifacts/complex_task_resolver/comprehensive_real_llm_review.md`
   - The artifact must include each raw input, expected challenge, graph,
     traversal path, lower-layer cache metadata if present, collapse decisions,
     evidence dependency status, node outputs, final packet, answerability
     judgment, quality judgment, failures, and whether the case is acceptable
     for future L2d integration.
   - The artifact must include every fixture case or an explicit dependency
     blocker for that case, plus a coverage table for fixture categories,
     expected statuses, and `required_stages`.

10. Parent updates docs.
    - New module README.
    - Development-plan README if lifecycle status changes are needed.
    - Development-plan execution evidence.

11. Parent runs the full verification gate.

12. Parent starts one independent code-review subagent after verification
    passes.

13. Parent remediates review findings inside approved scope and reruns affected
    tests.

14. Parent stops before broad live enablement or deprecation of any remaining
    capability.
    - Record whether the comprehensive real LLM review is sufficient.
    - Ask for explicit user instruction before rollout, removal of additional
      capabilities, or any workflow beyond the narrow L2d contract path.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after focused
  tests exist; owns production code changes only; does not edit tests unless
  the parent explicitly directs it; closes after planned production code changes
  are complete, excluding review fixes.
- Parent agent may continue service tests, review-harness tests, L2d contract
  tests, regression tests, and validation work while the production-code
  subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - contract tests established
  - Covers: implementation step 1.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py -q`
  - Evidence: record expected missing-symbol or failure output.
  - Sign-off: `Codex / 2026-06-30` after expected missing-module failure was
    recorded in `Execution Evidence`.

- [x] Stage 2 - module contracts, graph helpers, and subagents implemented
  - Covers: implementation steps 2-3.
  - Verify: same Stage 1 pytest command passes.
  - Evidence: record changed files and test output.
  - Sign-off: `Codex / 2026-06-30` after evidence was recorded.

- [x] Stage 3 - standalone service and review harness implemented
  - Covers: implementation steps 4-5.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_service.py -q`
  - Evidence: record service orchestration, packet projection, and artifact
    writer test output.
  - Sign-off: `Codex / 2026-06-30` after evidence was recorded.

- [x] Stage 4 - LLM planner/resolver/collapse/synthesizer implemented
  - Covers: implementation steps 5-6.
  - Scope: implement the bounded active-node resolution loop, typed node
    attempt observations, prompt-facing prior-attempt projection, resolver-owned
    evidence/algorithmic subagent invocation, collapse review, and bottom-up
    synthesis support. Do not add L2d integration.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_service.py -q`
  - Evidence: record node-loop contract coverage, LLM call budget,
    prompt-render output including prior attempts, and static integration-boundary
    grep output.
  - Sign-off: `Codex / 2026-06-30` after evidence was recorded.

- [x] Stage 5 - L2d capability contract integrated
  - Covers: implementation step 7.
  - Scope: add the canonical L2d-visible capability names
    `public_answer_research` and `local_context_recall`; route them to the
    complex resolver and existing RAG2 respectively; reject old canonical names
    without compatibility aliases.
  - Verify:
    - Legacy old-capability L2d tests have been updated or removed.
    - Real LLM L2d route-choice tests run one case at a time and demonstrate
      the expected capability for `public_answer_research`,
      `local_context_recall`, and non-resolver/speak cases.
    - Deterministic tests touched in this stage pass only as contract
      executability checks, not as evidence of L2d semantic route quality.
  - Evidence: record exact modified files, old-name cleanup grep output,
    one-case-at-a-time real LLM L2d trace paths, observed capability choices,
    and per-case judgment.
  - Sign-off: `Codex / 2026-06-30` after evidence was recorded.

- [x] Stage 5A - executable follow-up tasks implemented
  - Covers: approved follow-up execution scope added on 2026-06-30.
  - Scope: add `ComplexTaskFollowupTaskV1`, teach active-node and synthesis
    prompts to emit structured `followup_tasks`, create bounded child nodes
    from active-node follow-ups, create bounded root children from synthesis
    follow-ups, trace follow-up creation/rejection, and preserve prose
    `recommended_next_iteration` as semantic-only output.
  - Verify:
    - Focused deterministic tests prove validator behavior, node-level
      follow-up child creation, synthesis-level follow-up continuation,
      limit rejection, and the rule that prose recommendations alone are not
      executed.
    - One real LLM test is run through public module IO and inspected to prove
      the model can emit the intended structured follow-up output when the
      input correctly requires it.
  - Evidence: record changed files, deterministic test commands, real LLM trace
    path, observed follow-up task output, graph nodes created, and whether the
    result satisfies the intended contract.
  - Sign-off: `Codex / 2026-06-30` after evidence was recorded and the
    independent review gate approved production readiness for this stage.

- [x] Stage 6 - live LLM inspection cases completed
  - Covers: implementation step 8.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py -q -k "<single_case_name>" -s`
    for each approved case, one case at a time.
  - Evidence: record fixture case id, reviewed artifact paths, per-case
    judgment, and confirmation that fixture expected traces/final answers were
    not injected into prompts.
  - Sign-off: `Codex / 2026-07-01` after the 32-case review artifact and
    per-case traces were recorded and the user accepted the current evidence
    as good enough for this plan closeout.

- [x] Stage 7 - comprehensive real LLM review completed under user instruction
  - Covers: implementation step 9.
  - Verify:
    `test_artifacts/complex_task_resolver/comprehensive_real_llm_review.md`
    exists and includes raw inputs, graph traces, lower-layer cache metadata if
    present, collapse decisions, final packets, failures, and
    broad-live-enablement judgment.
  - Evidence: record the user's review instruction, artifact path, per-case
    outcome, and whether the user accepted the evidence as sufficient to
    consider broad live enablement.
  - Sign-off: `Codex / 2026-07-01` after
    `test_artifacts/complex_task_resolver/comprehensive_32_case_review.md`
    and `test_artifacts/complex_task_resolver/failure_mode_fix_tracker.md`
    were reviewed and accepted as sufficient for this plan closeout.

- [x] Stage 8 - docs and regression verification complete
  - Covers: implementation steps 10-11.
  - Verify all commands in `Verification`.
  - Evidence: record command outputs and any accepted warnings.
  - Sign-off: `Codex / 2026-07-01` after final focused regression,
    compile, diff-check, and documentation review evidence was recorded.

- [x] Stage 9 - standalone independent code review complete
  - Covers: standalone production-readiness review before the L2d contract
    cutover.
  - Verify review findings are resolved or explicitly accepted, and affected
    tests are rerun.
  - Evidence: record reviewer, findings, fixes, rerun commands, residual risks,
    and approval status.
  - Sign-off: `Codex / 2026-06-30` after follow-up independent review approved
    standalone production readiness and evidence was recorded.

- [x] Stage 10 - final independent code review complete
  - Covers: implementation steps 12-13 after the L2d capability contract,
    live review, docs, and verification gates are complete.
  - Verify review findings are resolved or explicitly accepted, and affected
    tests are rerun.
  - Evidence: record reviewer, findings, fixes, rerun commands, residual risks,
    and approval status.
  - Sign-off: `Codex / 2026-07-01` after final independent review by
    `Bernoulli` approved the final diff with no findings.

- [x] Stage 11 - broad live enablement remains blocked pending explicit user
      instruction
  - Covers: implementation step 14.
  - Verify no dialog, adapter, scheduler, coding-agent, filesystem, shell, or
    broad live workflow path consumes raw complex-task internals.
  - Evidence: record that the only L2d integration is the declared
    `public_answer_research`/`local_context_recall` capability boundary and
    that no rollout/deprecation beyond this boundary occurred.
  - Sign-off: `Codex / 2026-07-01` after static review confirmed this plan
    delivered only the declared `public_answer_research` and
    `local_context_recall` boundary, with no adapter, dialog, scheduler,
    filesystem, shell, or broad live rollout.

## Verification

Use `venv\Scripts\python` for Python commands.

### Static Greps

- `rg "answer_investigation" src\kazusa_ai_chatbot\cognition_resolver src\kazusa_ai_chatbot\cognition_chain_core src\kazusa_ai_chatbot\nodes`
  - Expected: no matches.
- `rg "rag_evidence|web_evidence" src\kazusa_ai_chatbot\cognition_resolver src\kazusa_ai_chatbot\cognition_chain_core`
  - Expected after Stage 5: no canonical L2d capability names, affordance
    descriptions, or dispatcher branches remain. Historical comments must be
    removed or rewritten to the new vocabulary unless they are in archived
    execution evidence outside runtime code.
- `rg "public_answer_research|local_context_recall" src\kazusa_ai_chatbot\cognition_resolver src\kazusa_ai_chatbot\cognition_chain_core`
  - Expected after Stage 5: matches only in resolver contracts, action
    selection affordances, dispatcher routing, prompt-safe projection, tests,
    and docs named in this plan.
- `rg "complex_task_result" src\kazusa_ai_chatbot\cognition_resolver src\kazusa_ai_chatbot\cognition_chain_core src\kazusa_ai_chatbot\nodes`
  - Expected: no matches unless Stage 5 implements an explicitly named
    prompt-safe observation field; raw graph or trace payload fields are never
    allowed.
- `rg "complex_task_resolver" src\kazusa_ai_chatbot\cognition_resolver src\kazusa_ai_chatbot\cognition_chain_core src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\dialog`
  - Expected after Stage 5: matches only in
    `cognition_resolver/capabilities.py` or approved tests/docs for routing
    `public_answer_research` to the public entrypoint. No dialog, adapter,
    scheduler, or node surface path may import complex resolver internals.
- `rg "ctr_0|expected_graph_trace|minimum_viable_answer|expected_final_answer|performance_reference_summary|forbidden_failure_modes" src\kazusa_ai_chatbot`
  - Expected in Phase 1: no matches. Fixture case ids, expected traces,
    minimum viable answers, expected final answers, performance references,
    and failure-mode hints must not appear in production code or runtime
    prompts.
- `rg "task_brief|worker|handler_id|platform_user_id|platform_channel_id" src\kazusa_ai_chatbot\complex_task_resolver`
  - Expected: no prompt-facing payload leaks. Internal trusted scope helpers, if
    any, must be documented and not projected to LLM prompts.
- `rg "from kazusa_ai_chatbot\.(dispatcher|calendar_scheduler|cognition_resolver|cognition_chain_core|nodes|dialog)|from adapters|import subprocess|create_subprocess|MCP_SERVERS|mcp" src\kazusa_ai_chatbot\complex_task_resolver`
  - Expected: no matches. The complex resolver must not call live
    workflow modules, adapters, scheduler, dialog, shell execution, or arbitrary
    MCP tooling.
- `rg "exec\(|subprocess|create_subprocess|LLInterface|ainvoke|SystemMessage|HumanMessage" src\kazusa_ai_chatbot\complex_task_resolver\algorithmic.py`
  - Expected in Phase 1: no matches. The algorithmic subagent must not call
    LLMs, shell execution, subprocesses, or notebooks.
- Review `src\kazusa_ai_chatbot\complex_task_resolver\algorithmic.py` for one
  restricted `eval(...)` call guarded by AST validation, safe globals, and
  empty `__builtins__`.
- `rg "recommended_next_iteration.*followup|followup.*recommended_next_iteration|search\\(|startswith\\(|in recommended_next_iteration" src\kazusa_ai_chatbot\complex_task_resolver`
  - Expected after Stage 5A: no production control flow that parses prose
    recommendations into executable follow-up tasks. Matches are allowed only
    in prompt text, docs, tests, or validation messages that state the
    semantic-only rule.

### Fixture Validation

```powershell
venv\Scripts\python -m json.tool tests\fixtures\complex_task_resolver_review_cases.json > $null
```

The fixture must contain exactly 32 cases. Review harnesses may read
`user_question`, `case_id`, and review metadata for artifact labeling, but must
not include expected traces, expected statuses, minimum viable answers,
expected final answers, performance-reference summaries, expected subagent
calls, or forbidden failure modes in model prompts.

The fixture validator must also emit a coverage summary containing:

- case count by `category`;
- case count by `expected_review_outcome`;
- all distinct `required_stages`;
- confirmation that every case has `minimum_viable_answer`,
  `expected_final_answer`, and `forbidden_failure_modes`.

The comprehensive review artifact must later map every emitted category,
status, and required stage to at least one inspected case or to an explicit
dependency blocker.

### Focused Tests

```powershell
venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py -q
venv\Scripts\python -m pytest tests\test_complex_task_resolver_graph.py -q
venv\Scripts\python -m pytest tests\test_complex_task_resolver_evidence.py -q
venv\Scripts\python -m pytest tests\test_complex_task_resolver_algorithmic.py -q
venv\Scripts\python -m pytest tests\test_complex_task_resolver_service.py -q
```

Stage 5A must include focused deterministic coverage for:

- `ComplexTaskFollowupTaskV1` validator acceptance/rejection;
- active-node `followup_tasks` creating child nodes under the source node;
- synthesis `followup_tasks` creating root-level follow-up nodes and continuing
  traversal before final packet return;
- follow-up limit rejection preserving lacking knowledge and boundary notes;
- prose-only `recommended_next_iteration` never creating graph nodes.

### Review Dependency Preflight

Before any live LLM case:

```powershell
venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_review_dependency_preflight -q -s
```

The preflight must write or print a structured dependency record for:

- resolver stage LLM configuration;
- JSON repair configuration, if enabled;
- `web_read` availability;
- `web_search`/`SEARXNG_URL` availability;
- outbound network availability for public sources;
- MongoDB and embedding availability if full RAG2 evidence is selected;
- MCP availability, which must be absent or unused unless a later plan approves
  it.

If preflight fails, do not run the comprehensive review. If only optional web
search is unavailable, live cases that require search must be recorded as
dependency-blocked or partial, not passed.

### Resolver And Cognition Non-Integration Regression

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_loop.py -q
venv\Scripts\python -m pytest tests\test_action_selection_prompt_contract.py tests\test_persona_supervisor2_action_selection.py -q
```

These regressions should pass without any new complex-task resolver capability
appearing in L2d or cognition resolver behavior.

### Live LLM Inspection

Run each live case separately:

```powershell
venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py -q -k "<single_case_name>" -s
```

Required live case families:

- multi-part planning question;
- evidence-dependent comparison;
- question with duplicate subtask paths that should collapse;
- question blocked by missing user-owned input;
- simple question that the standalone specialist should mark out of scope or
  not worth complex decomposition;
- adversarial or ambiguous complex question where the correct result is
  partial, cannot_answer, or needs_user_input.

For every live case, record the following before running the next case:

- observed behavior and exact failing or passing boundary;
- failure modes, including schema failures, source-quality failures,
  hallucinated facts, arithmetic mistakes, collapse mistakes, lower-layer cache
  metadata misuse, answerability misclassification, or dependency blockers;
- consolidated root cause grounded in real prompt/output/parser evidence;
- architecture review against this plan's boundaries, especially local-LLM
  reliability, deterministic validation ownership, subagent IO ownership,
  evidence ownership, and anti-cheat rules;
- recommended fix classification:
  - `focused`: a scoped prompt, test, validator, or harness change that does
    not change the architecture or public contract;
  - `architectural`: a change to stage boundaries, structured-output
    strategy, schema-repair path, subagent responsibility, or graph contract
    that must be planned before continuing broad live review;
- decision on whether the next live case may proceed.

### Comprehensive Real LLM Review

This gate runs only after the user explicitly instructs it. It is required
before broad live workflow enablement, rollout, or deprecation beyond the
narrow `public_answer_research` and `local_context_recall` L2d contract.

The review artifact path is:

```text
test_artifacts/complex_task_resolver/comprehensive_real_llm_review.md
```

The review case source is:

```text
tests/fixtures/complex_task_resolver_review_cases.json
```

The artifact must include:

- raw user question and context summary;
- why the case is complex;
- expected answerability class before execution;
- fixture category and required-stage coverage;
- full graph or clipped graph with node ids, statuses, and traversal order;
- active-node outputs;
- RAG/evidence references used, if any;
- dependency status for web search/read, RAG, MongoDB, embeddings, and network
  where relevant;
- lower-layer cache metadata and freshness blockers, if present;
- collapse candidates, collapse decision, and reason;
- final `ComplexTaskResolutionPacketV1`;
- human quality judgment;
- robustness issue list;
- explicit judgment on whether the behavior is safe to expose to future L2d
  production traffic beyond the controlled review path.

### Broad Regression

```powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
```

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Alignment with `Must Do`, `Deferred`, `Change Surface`, exact contracts,
  implementation order, verification gates, and acceptance criteria.
- Whether `complex_task_resolver` remains distinct from `cognition_resolver`.
- Whether L2d/cognition-resolver integration is limited to the declared
  `public_answer_research` and `local_context_recall` capability boundary.
- Whether no dialog, adapter, scheduler, coding-agent, filesystem, shell, or
  broad live workflow integration has been added.
- Whether existing typed dictionaries are reused only where semantics match.
- Whether prompt/RAG/context projections avoid raw ids, backend terms, worker
  internals, and final-dialog leakage.
- Whether fixture case ids, keywords, expected graph traces, expected statuses,
  minimum viable answers, expected final answers, performance-reference
  summaries, or forbidden failure modes are absent from deterministic routing
  code and runtime prompts.
- Whether same-run collapse is scope-safe and does not carry stale lower-layer
  retrieval results as current facts.
- Whether graph collapse is bounded and traceable.
- Whether live review dependency preflight, evidence-subagent unavailable
  paths, and fixture category/status/stage coverage are present in artifacts.
- Whether algorithmic subagent operations are deterministic, typed, tested,
  free of LLM calls, and free of arbitrary expression execution.
- Whether comprehensive real LLM evidence is sufficient for local/weaker model
  risk and for considering broad live enablement beyond the L2d contract path.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface. If a fix changes the public contract or adds new
scope, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `complex_task_resolver` exists as a separate documented package.
- The specialist exposes a standalone `resolve_complex_task` entrypoint and
  accepts `ComplexTaskResolverRequestV1` plus
  `ComplexTaskResolverContextV1` plus optional structural
  `ComplexTaskResolverOptionsV1`.
- The specialist returns a validated `ComplexTaskResolutionPacketV1` through
  declared module IO and the L2d `public_answer_research` routing boundary.
- Evidence subagent behavior is bounded, WebAgent3-backed through declared IO,
  tested for unavailable dependencies, and never calls live workflow modules,
  adapters, scheduler, dialog, shell tools, or arbitrary MCP tools.
- The deterministic algorithmic subagent follows the standard subagent IO
  contract, covers the fixture arithmetic families, uses caller-prepared
  numeric expressions, and contains no LLM calls, arbitrary expression
  execution, shell execution, notebook execution, or broad symbolic-math
  engine. The only permitted expression path is the AST-validated
  `evaluate_expression` action with empty `__builtins__` and safe numeric
  helpers.
- The existing cognition resolver remains the only recurrence controller.
- L2d exposes `public_answer_research` and `local_context_recall` as the
  canonical resolver capability names.
- `public_answer_research` routes to `resolve_complex_task(...)`; former
  L2d-visible `web_evidence` work is collapsed into this path.
- `local_context_recall` routes to existing RAG2 persona evidence; former
  L2d-visible `rag_evidence` work is represented by this clearer name.
- No `answer_investigation`, `web_evidence`, or `rag_evidence` compatibility
  aliases remain in runtime L2d capability vocabulary after the cutover.
- No persona graph, dialog, adapter, scheduler, coding-agent, filesystem,
  shell, or broad live workflow code consumes raw complex-task internals.
- The new task graph uses stable node ids, bounded depth, bounded node count,
  node statuses, collapse events, and bottom-up synthesis.
- Resolver-owned semantic cache is absent; same-run collapse and lower-layer
  cache metadata boundaries are implemented.
- Existing RAG, dialog, background-work, consolidation, adapter, and scheduler
  behavior remain unchanged.
- Focused deterministic tests, standalone service tests, integration-boundary
  greps, review dependency preflight, resolver regressions, and approved live
  LLM inspection cases pass or have recorded accepted blockers.
- The review fixture
  `tests/fixtures/complex_task_resolver_review_cases.json` contains 32
  concrete cases. Each case has a `minimum_viable_answer` for AI-judge
  acceptance and a Codex-authored `expected_final_answer`; both are used only
  as review metadata, not as prompt or deterministic-routing hints.
- Comprehensive real LLM review has been performed after explicit user
  instruction, and the artifact records every fixture case or explicit
  dependency blocker, category/status/required-stage coverage, dependency
  status, and whether the module is robust enough for broad live enablement.
- Broad live enablement remains blocked until the user gives a separate
  explicit instruction after reviewing or accepting the real LLM evidence.
- Independent code review has no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Specialist becomes a second cognition resolver | Separate package, docs, public entrypoint, and no ownership of `ResolverCycleStateV1` recurrence | Code review and README boundary checks |
| Prompt bloat from graph projection | Keep full graph in validated state and project compact packet summaries | Projection tests and live LLM artifacts |
| Lower-layer cache metadata is mistaken for resolver-owned answer reuse | Resolver does not own cache keys or semantic reuse; same-run reuse uses bounded graph collapse only | Contract tests, collapse tests, and review artifacts |
| Collapse merges distinct subtasks | Deterministic candidate set and traceable LLM collapse reason | Collapse tests and live duplicate-path case |
| Response latency grows too high | Hard LLM/tool/node caps; return partial/cannot-answer on cap | Service timeout tests and live review artifacts |
| Dialog treats packet as final voice | Dialog receives only selected `speak` content after L2d; raw complex-task graph state is never a dialog input | Static integration-boundary greps and code review |
| Premature broad routing occurs before robustness is proven | Only the narrow L2d capability contract is brought forward; broad live enablement remains blocked by real LLM review and user acceptance | Stage 11 broad-live block sign-off |
| Reuse distorts existing contracts | Reuse `EvidenceRefV1` only where semantics match; document future resolver wrapper without changing current contracts | Contract tests and code review |
| Live review hides missing external dependencies | Run dependency preflight, record web/RAG/LLM/Mongo/embedding/network status, and mark affected cases dependency-blocked or partial | Preflight artifact, comprehensive review coverage table, and evidence-subagent tests |
| Evidence subagent becomes accidental runtime integration | Keep subagent rosters module-owned, allow tests to read traces only, and forbid dialog/adapter/scheduler calls from complex resolver internals | Static integration-boundary greps, evidence tests, and independent code review |
| LLM arithmetic produces plausible wrong numbers | Route arithmetic-like nodes through the deterministic algorithmic subagent and require caller-prepared expressions plus calculator safety tests | Algorithmic tests, no-LLM/no-shell grep, AST-validation review, and live review artifact checks |

## Execution Evidence

### 2026-06-29 draft

- User clarified that the new resolver is not a replacement for the current
  cognition-integrated resolver at first.
- User clarified the new module should eventually sit at the same level as
  RAG/coding-like specialists, be callable by L2d, handle complicated
  questions and challenges, and return enough answerability information for
  cognition to proceed to dialog.
- User clarified that the complex-task resolver must not be routed to L2d or
  the rest of the workflow before it is proven robust.
- User required comprehensive real LLM review, instructed by the user, before
  any integration with L2d or the live workflow.
- Revised draft decision: implement a standalone/shadow
  `complex_task_resolver` module first, with no L2d/runtime integration in
  this plan.
- Revised draft decision: reuse `EvidenceRefV1` where semantics match, add new
  standalone request/context/task-graph contracts, and document current
  resolver request/observation reuse only as a future integration design.
- User requested 30 concrete review cases covering web retrieval,
  consolidation, arithmetic, summarization, coding-agent-style work, collapse,
  lower-layer retrieval freshness, and unresolvable tasks.
- Draft fixture added:
  `tests/fixtures/complex_task_resolver_review_cases.json`.
- Draft anti-cheat rule added: fixture keywords, case ids, expected graph
  traces, expected statuses, minimum viable answers, expected final answers,
  performance-reference summaries, and forbidden failure modes must not be
  injected into runtime prompts or deterministic routing, repair, or rewrite
  logic.
- User requested one additional GPU inference performance reference case based
  on a ChatGPT extended-thinking example comparing RTX 5090 and R9700 for
  Qwen3.6 and Gemma 4 Q4 local inference.
- Draft fixture now contains 31 review cases.
- Reference-answer quality pass corrected the
  `ctr_002_codex_claude_docs_conflict` source framing against current official
  Codex CLI and Claude Code/Agent SDK documentation, and removed evaluator-style
  wording from `expected_final_answer` values so those fields are direct
  Codex-authored reference answers.
- Second-pass plan review against the 31-case fixture found that web retrieval,
  source conflict, benchmark normalization, and dependency-sensitive cases
  required an explicit review-only evidence subagent, runtime dependency
  boundary, dependency preflight, and fixture coverage table. The plan was
  updated so missing web/RAG/LLM/Mongo/embedding/network dependencies become
  recorded blockers or partial results rather than hidden failures or passes.
- User required arithmetic and benchmark-style calculation to avoid LLM
  reasoning. The plan now requires a reusable deterministic algorithmic
  subagent under `complex_task_resolver`, using the same standard subagent IO
  envelope as evidence subagents and forbidding arbitrary expression
  evaluation, shell execution, and LLM calls inside calculation execution.
- Superseding decision: Phase 1 has no resolver-owned semantic cache. Retrieval
  cache behavior remains in RAG, WebAgent, or source layers; same-run answer
  reuse is graph collapse only.
- User explicitly instructed execution to start on 2026-06-30, with a new
  branch in the current workspace, one production-code subagent, one
  `gpt-5.5` `xhigh` independent code-review subagent, and one-at-a-time real
  LLM review beginning with case 01.
- This plan is now in progress for standalone Phase 1 implementation only.

### 2026-06-30 execution

- Created branch `complex-task-resolver-execution` in the current workspace.
  No separate checkout or worktree was created.
- Contract correction: `public_answer_research` and the standalone complex
  resolver now return semantic evidence sections instead of top-level
  answerability status. The complex resolver provides knowledge known so far,
  knowledge still lacking, evidence-boundary notes, and recommended evidence
  directions; cognition owns the judgment about answering, retrying, asking, or
  stopping.
- Stage 4 loop-gap review found that the implementation had an outer graph
  traversal loop but no bounded active-node resolution recurrence. The plan now
  requires typed node attempt observations, compact prompt-facing prior-attempt
  projection, resolver-owned local next-action selection, bounded specialist
  retry semantics for evidence, deterministic single-shot arithmetic, and
  explicit node blocking or partial status when the loop cap is exhausted.
- Stage 1 contract tests added:
  `tests/test_complex_task_resolver_contracts.py`,
  `tests/test_complex_task_resolver_graph.py`,
  `tests/test_complex_task_resolver_evidence.py`, and
  `tests/test_complex_task_resolver_algorithmic.py`.
- Stage 1 expected failure recorded:
  `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py -q`
  failed during collection because
  `kazusa_ai_chatbot.complex_task_resolver` does not exist yet.
- One production-code subagent implemented the standalone package under
  `src/kazusa_ai_chatbot/complex_task_resolver/` only.
- Stage 2 verification passed:
  `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py -q`
  reported `15 passed`.
- Parent added the standalone service entrypoint, runtime contract,
  prompt-safe packet projection, fixture coverage test, and live review test
  harness.
- Stage 3 verification passed:
  `venv\Scripts\python -m pytest tests\test_complex_task_resolver_service.py -q`
  reported `2 passed`.
- Focused resolver suite passed:
  `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_service.py -q`
  reported `35 passed` after Stage 4 loop implementation.
- Stage 4 implemented the bounded active-node resolution loop. Each node now
  owns `ComplexTaskNodeAttemptV1` records, `_compact_node(...)` projects the
  most recent attempts into the next active-node prompt, loop exhaustion
  produces an explicit blocked node, and `trace_summary` records
  `node_attempt_count` plus a compact read-only attempt log.
- Stage 4 prompt-facing facilities were updated: the live node-resolver prompt
  now documents the non-terminal `node_attempt` output shape and instructs the
  resolver stage to use prior attempts rather than repeating a blocked local
  action.
- Stage 4 subagent boundary was updated: `ComplexTaskEvidenceSubagent` records
  bounded evidence availability attempts and reports unavailable dependencies
  without caller-supplied subagent configuration; `AlgorithmicSubagent` remains
  deterministic single-shot and only evaluates caller-prepared expressions.
- Stage 4 final-answer handling now preserves a useful synthesizer brief while
  appending source-grounding limits; source-backed facts are still withheld
  when structured evidence handles are missing.
- Fixture coverage validation passed:
  `venv\Scripts\python -m pytest tests\test_complex_task_resolver_fixture.py -q`
  reported `32` cases with all required `minimum_viable_answer`,
  `expected_final_answer`, and `forbidden_failure_modes` fields present.
- Static non-integration and anti-cheat greps returned no matches for
  fixture ids, expected-answer hints, GPU case keywords, or review-reference
  fields in `src\kazusa_ai_chatbot\complex_task_resolver`. The broader test
  grep only found the expected negative tests and live-review fixture metadata
  exclusion list.
- Static non-integration grep returned no matches for `answer_investigation`
  outside `src\kazusa_ai_chatbot\complex_task_resolver`, confirming that no
  L2d, cognition resolver, dialog, adapter, scheduler, or live workflow route
  was connected in Stage 4.
- Live review dependency preflight passed:
  `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_review_dependency_preflight -q -s -m live_llm`.
  Resolver LLM, JSON repair LLM, web read, SearXNG-backed web search, and
  public network were available; RAG/Mongo/embedding were explicitly not
  selected; MCP was unused.
- Live case 01 was run and failed before node resolution:
  `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_review_case_01_agent_harness_comparison -q -s -m live_llm`
  failed because the live planner repeatedly emitted invalid graph-node
  payloads that did not validate as `complex_task_node.v1`.
- Case 01 failure artifact:
  `test_artifacts/complex_task_resolver/case_01_agent_harness_comparison_failed.md`.
  Current judgment: the service failed closed correctly, but prompt-only graph
  planning is not robust enough for live review; next architecture work should
  add an explicit schema-repair or structured-output stage rather than further
  prompt nudging or deterministic semantic repair using fixture hints.

### 2026-06-30 production boundary correction

- User clarified that self-contained means production-tappable through
  declared IO. The complex resolver may call other production helper agents
  through their declared IO, but callers and real LLM tests must not provide
  resolver LLM stages, prompt variants, subagent rosters, graph paths, or
  expected answers.
- Public IO was corrected from the review-runtime scaffold to:
  `resolve_complex_task(request, context, options=None)`, where
  `ComplexTaskResolverOptionsV1` contains only structural limits.
- Production-owned LLM stage handlers now live under
  `src/kazusa_ai_chatbot/complex_task_resolver/stages.py` for graph planning,
  active-node resolution, collapse review, and bottom-up synthesis.
- `ComplexTaskEvidenceSubagent` now calls WebAgent3 through its declared
  `run(task, context, max_attempts)` IO and preserves lower-layer cache
  metadata. The explicit unavailable subagent remains only for dependency
  blocker tests and deterministic service-test isolation.
- The real LLM harness no longer owns stage prompts, stage invokers, or
  pre-resolver web evidence collection. It calls only public resolver IO and
  inspects returned packet/trace material.
- Focused baseline before correction:
  `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py::test_options_contract_rejects_external_stage_configuration tests\test_complex_task_resolver_evidence.py::test_evidence_subagent_calls_web_agent3_declared_io -q`
  failed during collection because `COMPLEX_TASK_RESOLVER_OPTIONS_VERSION` did
  not exist.
- Corrected deterministic verification passed:
  `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_service.py tests\test_complex_task_resolver_fixture.py -q`
  reported `32 passed`.
- Compile verification passed:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\complex_task_resolver\__init__.py src\kazusa_ai_chatbot\complex_task_resolver\contracts.py src\kazusa_ai_chatbot\complex_task_resolver\service.py src\kazusa_ai_chatbot\complex_task_resolver\stages.py src\kazusa_ai_chatbot\complex_task_resolver\subagents.py tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_service.py tests\test_complex_task_resolver_live_llm.py`.
- Static boundary greps passed with no matches for the retired runtime
  contract, test-owned live prompts, test-owned stage invokers, L2d routing,
  fixture answer hints in production, adapter/scheduler/dialog imports, MCP
  use, shell execution, or algorithmic-subagent LLM calls.
- Cognition resolver non-integration regression passed:
  `venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_loop.py -q`
  reported `54 passed`.
- Action-selection regression command:
  `venv\Scripts\python -m pytest tests\test_action_selection_prompt_contract.py tests\test_persona_supervisor2_action_selection.py -q`
  reported `26 passed, 1 failed`. The failing test is
  `test_action_selection_prompt_follows_cognition_prompt_structure`, which
  asserts `ACTION_ROUTER_PROMPT` contains `# 输入格式`. This is outside the
  complex-task resolver change surface and was not changed under this plan.
- No new live LLM case was run after the production boundary correction.
  Live LLM case execution remains paused until the user commands a specific
  case.

#### Case 01 failure mode analysis

- Observed behavior:
  - Dependency preflight passed for resolver LLM, JSON repair LLM, public
    network, `web_read`, and SearXNG-backed `web_search`.
  - The review harness collected web evidence before resolver execution.
  - The standalone service stopped at `ComplexTaskGraphV1` validation before
    active-node resolution, collapse review, bottom-up synthesis, or final
    packet production.
- Failure modes:
  - The planner emitted a generic DAG-like node schema with fields such as
    `id`, `type`, `description`, `dependencies`, `inputs`, and `outputs`
    instead of `complex_task_node.v1`.
  - After prompt hardening, the planner still emitted invalid contract details:
    schema typo `complex_task_node.v01`, plain-string evidence references, and
    missing or misspelled required list fields such as
    `assumptions_or_inferences`.
  - The planner attempted to perform evidence synthesis during decomposition,
    creating resolved-looking nodes and citations during a stage that should
    only create a pending task graph.
- Consolidated root cause:
  - The current live graph-planning stage asks a local LLM to satisfy two
    responsibilities at once: semantic decomposition and exact wire-contract
    generation for a strict nested graph schema. The model treats the request
    as a generic planning task and approximates a DAG schema, then drifts into
    premature evidence synthesis when review evidence is present in context.
    This is a local-LLM schema reliability failure, not a missing dependency
    and not a deterministic service bug.
- Architecture review:
  - The service behavior is correct: deterministic validation owns graph shape
    and fails closed before invalid state can enter node resolution or
    synthesis.
  - The architecture expectation that "LLM stages own decomposition" remains
    valid only if decomposition output is separated from the strict graph wire
    contract or passed through an explicit structured-output/schema-repair
    boundary.
  - Further negative prompt accretion is not a good fix. The failure repeated
    after three planner-boundary attempts, which indicates the stage boundary
    is too brittle for the target local model.
  - Deterministic semantic repair that infers missing graph meaning from case
    keywords or fixture expectations is forbidden by the anti-cheat rule and
    would undermine review evidence.
- Recommended fix classification:
  - Architectural fix required before continuing broad live review.
  - Add an explicit planner output strategy: either a smaller semantic
    decomposition contract that deterministic code maps into
    `ComplexTaskGraphV1`, or a dedicated structured-output/schema-repair stage
    that is itself validated, traced, and included in review artifacts.
  - Keep deterministic repair limited to mechanical schema completion only
    when semantics are already explicit in the model output; do not infer
    missing subtasks, statuses, evidence, answers, or collapse decisions from
    fixture metadata or keywords.
- Decision:
  - Do not run case 02 until the planner boundary is revised and case 01 is
    rerun with a valid graph packet or an explicitly accepted blocker.

#### Case 01 fix and rerun analysis

- Local LLM input/output rule applied:
  - The live planner no longer has to emit the strict
    `ComplexTaskGraphV1`/`ComplexTaskNodeV1` wire shape. It emits a compact
    semantic task list, and deterministic service code maps that list into the
    validated graph contract.
  - The semantic planner output may hint at `algorithmic_task`, but the graph
    mapper normalizes that hint to `subtask` unless a later node-resolution
    stage emits a typed algorithmic subagent request. This prevents a local LLM
    from creating deterministic arithmetic nodes without operands.
  - The synthesizer receives semantic `answerability_issues`; deterministic
    packet assembly still owns final status downgrades and unresolved-item
    projection.
- Focused fixes implemented:
  - Planner semantic decomposition mapping into validated graph state.
  - Collapse review candidate narrowing: synthesis nodes are not collapse
    candidates, and collapse is limited to resolved same-kind nodes.
  - Packet answerability reducer: unresolved graph nodes force `partial`
    instead of `answered`.
  - Source-grounding reducer: evidence nodes that produce source-backed facts
    without structured evidence handles or source observation ids force
    `partial`.
  - Live case status assertion: the review test now compares the final packet
    status to fixture metadata after execution; fixture metadata remains
    excluded from all prompts and runtime stage payloads.
- Verification:
  - Deterministic resolver suite:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_service.py tests\test_complex_task_resolver_fixture.py -q`
    reported `24 passed`.
  - Live case 01 rerun:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_review_case_01_agent_harness_comparison -q -s -m live_llm`
    passed with `CASE_01_STATUS=partial`.
  - Assertion-backed live case 01 rerun:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_review_case_01_agent_harness_comparison -q -s -m live_llm`
    passed with `CASE_01_STATUS=partial` after the test was tightened to
    assert the expected status.
  - Final assertion-backed live graph:
    ```mermaid
    graph TD
      root["root\nroot\nexpanded"]
      task_1["task_1\nevidence_need\nresolved"]
      task_2["task_2\nevidence_need\nresolved"]
      task_3["task_3\nsynthesis\nresolved"]
      root --> task_1
      root --> task_2
      root --> task_3
    ```
- Observed behavior after fix:
  - The resolver returns a validated packet instead of failing at graph
    construction.
  - The packet status is `partial`, matching the case 01 expected
    answerability class.
  - The final packet carries explicit unresolved evidence gaps for review
    gates and failure recovery in its answer text, and deterministic packet
    assembly adds the unresolved item for missing structured evidence handles.
  - No collapse events occurred, and no synthesis node was collapsed into an
    evidence node.
- Remaining failure modes:
  - The evidence path still relies on prose from WebAgent3 review evidence,
    not structured `EvidenceRefV1` or source-observation handles.
  - Some live evidence claims are only weakly traceable in the artifact,
    including AgentRouter/OpenClaw/Codex/Hermes relationship claims.
  - The graph can now complete its synthesis node within the iteration cap,
    but source-grounding remains insufficient for a full `answered` status.
- Consolidated root cause after fix:
  - Case 01 is no longer blocked by graph schema generation. The remaining
    limitation is evidence traceability and source quality: the current review
    harness passes web evidence as compact prose, so deterministic code cannot
    verify which final facts are backed by primary or structured sources.
- Architecture review:
  - The current result is consistent with the planned architecture: the module
    should answer only when all required branches are resolved and evidence
    can be traced. In this run, both conditions are false, so `partial` is the
    correct answerability class.
  - This validates the need for the planned review-only evidence subagent to
    return bounded source-backed facts with structured evidence handles before
    case 01 can ever be considered fully `answered`.
- Fix classification:
  - Focused fix completed for case 01 robustness: graph schema reliability,
    collapse safety, algorithmic-kind normalization, and answerability status
    reduction.
  - Architectural follow-up remains before broad review sign-off: add
    structured evidence handles from the evidence path so source-grounding
    checks can distinguish fully grounded answers from partial evidence.
- Decision:
  - Case 01 is accepted as a `partial` live review result after the focused
    fixes. Do not claim the case is fully answered.
  - Do not route the resolver to L2d or the live workflow. Continue real LLM
    review one case at a time only after this evidence is reviewed.

#### Case 02 run and failure mode analysis

- Harness update:
  - Added
    `tests/test_complex_task_resolver_live_llm.py::test_live_review_case_02_codex_claude_docs_conflict`.
  - Generalized the live review helper so each case writes its own artifact
    and trace while still running one case at a time.
  - Fixed review preflight wording so it no longer says "case 01" for later
    cases.
- Verification before live run:
  - `venv\Scripts\python -m py_compile tests\test_complex_task_resolver_live_llm.py`
    passed.
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py --collect-only -q -m live_llm`
    collected dependency preflight, case 01, and case 02.
- Live case 02 command:
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_review_case_02_codex_claude_docs_conflict -q -s -m live_llm`
    passed structurally with `CASE_02_STATUS=partial`.
- Trace and artifact:
  - Trace:
    `test_artifacts/llm_traces/complex_task_resolver_live_llm__case_02_codex_claude_docs_conflict.json`.
  - Artifact:
    `test_artifacts/complex_task_resolver/case_02_codex_claude_docs_conflict.md`.
- Mermaid graph from the run:
  ```mermaid
  graph TD
    root["root\nroot\nexpanded"]
    task_1["task_1\nevidence_need\nresolved"]
    task_2["task_2\nsubtask\nresolved"]
    task_3["task_3\nevidence_need\ncollapsed"]
    task_4["task_4\nsubtask\ncollapsed"]
    task_5["task_5\nsynthesis\npending"]
    root --> task_1
    root --> task_2
    root --> task_3
    root --> task_4
    root --> task_5
    task_3 -. collapsed .-> task_1
    task_4 -. collapsed .-> task_2
  ```
- Observed behavior:
  - The packet status matched the fixture expected status, `partial`.
  - The answer disclosed source-access limits for both official-source paths.
  - The graph collapsed the source-conflict branch into the official-doc
    verification branch, and collapsed source-reliability review into the
    architectural extraction branch.
  - The final synthesis branch remained pending.
- Task dissection review:
  - The resolver split the question into reasonable trunks: official
    documentation verification, agent-loop detail extraction, source conflict
    identification, source reliability review, and final synthesis.
  - This is the correct distribution for case 02. The user asked for current
    documentation, comparison of agent-loop models, and source disagreement
    handling, and the graph created separate branches for those concerns.
  - The collapsed source-conflict and source-reliability branches had already
    produced content before collapse, so collapse did not prevent those tasks
    from being addressed.
  - The pending synthesis branch and source-access blockers correctly kept the
    packet `partial`.
- Failure modes:
  - External source availability prevented a full answer. OpenAI documentation
    was blocked by Cloudflare during retrieval, and Claude documentation could
    not be fully read due size limits.
  - Secondary/snippet-derived claims were still listed as `source_backed_facts`.
    Under the case-02 task-dissection review lens, this is not a decomposition
    failure; it is a source-grounding limitation already reflected by `partial`.
- Consolidated root cause:
  - The resolver's task dissection was adequate. The incomplete answer comes
    from retrieval evidence quality and source access, not poor decomposition.
  - The evidence path still carries WebAgent3 prose rather than structured
    source handles, so deterministic code cannot distinguish official,
    primary, snippet, and secondary evidence at the fact level.
- Architecture review:
  - The `partial` status is correct and the decomposition behavior is accepted
    for case 02.
  - The graph created the right work trunks and distributed the task
    reasonably.
  - Source-trust limitations remained visible in the packet instead of being
    hidden behind an `answered` status.
- Fix classification:
  - No focused task-dissection fix is required before case 03.
  - Architectural follow-up remains: add structured source handles and source
    type labels to the review evidence path.
- Decision:
  - Case 02 is accepted as a `partial` task-dissection pass. The source-access
    and structured-evidence limitations remain known Phase 1 evidence-path
    limitations, but they do not block moving to case 03 under the current
    review goal.

#### Case 03 run and failure mode analysis

- Harness update:
  - Added
    `tests/test_complex_task_resolver_live_llm.py::test_live_review_case_03_runtime_versions`.
  - The live review helper continued to write one artifact and one trace per
    case, with fixture metadata excluded from prompts and runtime stage
    payloads.
- Verification before live run:
  - `venv\Scripts\python -m py_compile tests\test_complex_task_resolver_live_llm.py`
    passed.
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py --collect-only -q -m live_llm`
    collected dependency preflight, case 01, case 02, and case 03.
- Live case 03 command:
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_review_case_03_runtime_versions -q -s -m live_llm`
    failed the expected-status assertion with `CASE_03_STATUS=partial`.
- Trace and artifact:
  - Trace:
    `test_artifacts/llm_traces/complex_task_resolver_live_llm__case_03_runtime_versions.json`.
  - Artifact:
    `test_artifacts/complex_task_resolver/case_03_runtime_versions.md`.
- Mermaid graph from the run:
  ```mermaid
  graph TD
    root["root\nroot\nexpanded"]
    task_1["task_1\nevidence_need\nresolved"]
    task_2["task_2\nevidence_need\ncollapsed"]
    task_3["task_3\nevidence_need\ncollapsed"]
    task_4["task_4\nsynthesis\nresolved"]
    root --> task_1
    root --> task_2
    root --> task_3
    root --> task_4
    task_2 -. collapsed .-> task_1
    task_3 -. collapsed .-> task_1
  ```
- Observed behavior:
  - The packet status was `partial`, while the fixture expected `answered`.
  - The graph separated broad runtime-version extraction, Node.js LTS
    identification, MongoDB conservative-target identification, and final
    synthesis.
  - Node.js and MongoDB evidence branches were collapsed into the broad
    extraction branch after resolution.
  - The final answer distinguished latest/current from conservative target in
    broad terms, but omitted required exact version facts for the fixture.
- Task dissection review:
  - The initial split was directionally reasonable for the question.
  - The distribution was not sufficient to produce a complete answer because
    exact-version branches were treated as resolved even when the concrete
    Node.js LTS and MongoDB current/conservative release numbers were missing.
  - Collapse did not erase generated branch text, but it hid that the
    exact-version branches had only partial evidence.
- Failure modes:
  - The packet failed the hard status gate: `partial` instead of `answered`.
  - The answer omitted the fixture-required exact current/latest and
    conservative target versions.
  - The evidence collector used weak or secondary prose evidence for a
    date-sensitive official-version question.
  - The deterministic source-grounding reducer also forced `partial` because
    source-backed facts had no structured evidence handles or source
    observation ids.
- Consolidated root cause:
  - Case 03 failed because the current review evidence path is not strong
    enough for exact current-version questions requiring official release
    facts. The graph can create plausible branches, but the branches can be
    marked resolved with incomplete facts, and collapse can then obscure that
    incompleteness.
- Architecture review:
  - This is primarily an evidence-path and answerability-boundary failure, not
    a basic graph-construction failure.
  - The `partial` status is safer than incorrectly claiming `answered`, but it
    means the module does not yet satisfy case 03.
  - Current-version questions need structured official-source handles and
    fact-level source typing before they can pass as fully answered.
- Fix classification:
  - Focused fix needed before claiming case 03 robust: exact-version evidence
    branches must not be marked `resolved` when they lack the requested
    concrete version numbers.
  - Architectural follow-up needed for case 03 to pass as `answered`: the
    review evidence path must return structured official-source facts and
    handles instead of only prose summaries.
- Decision:
  - Case 03 is not accepted.
  - Do not run case 04 until the user decides whether to implement the focused
    exact-version branch fix now or record case 03 as an unresolved Phase 1
    limitation.

#### Case 32 run and failure mode analysis

- Harness update:
  - Added
    `tests/test_complex_task_resolver_live_llm.py::test_live_review_case_32_emergency_power_subagents`.
  - Added a case-32 live path that registered resolver-local `evidence` and
    `algorithmic` subagents through recording wrappers so actual subagent
    invocation was observable in the artifact and trace.
  - This harness registration is now classified as an interface violation:
    real LLM tests must use declared public module IO only and may inspect
    subagent invocation through read-only traces after execution.
  - Added `expected_subagent_calls` to the fixture anti-cheat list and live
    prompt-payload leak check.
- Verification before live run:
  - `venv\Scripts\python -m py_compile tests\test_complex_task_resolver_live_llm.py`
    passed.
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py --collect-only -q -m live_llm`
    collected dependency preflight, cases 01, 02, 03, and 32.
- First live case 32 command under the obsolete operation-specific calculator
  interface:
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_review_case_32_emergency_power_subagents -q -s -m live_llm`
    failed before artifact writing because the local LLM emitted an invalid
    operation-specific `algorithmic` payload.
- Interface correction:
  - Replaced the operation-specific calculator interface with
    `evaluate_expression`.
  - The node resolver must interpret units and prepare one numeric expression;
    the algorithmic subagent only evaluates the expression.
  - The correction did not inject expected answers, expected subagent calls,
    fixture node ids, product answers, or hidden review metadata into the
    prompt.
- Second live case 32 command:
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_review_case_32_emergency_power_subagents -q -s -m live_llm`
    wrote the artifact and trace, returned `CASE_32_STATUS=partial`, then
    failed the hard subagent gate because `algorithmic` was not observed.
- Trace and artifact:
  - Trace:
    `test_artifacts/llm_traces/complex_task_resolver_live_llm__case_32_emergency_power_subagents.json`.
  - Artifact:
    `test_artifacts/complex_task_resolver/case_32_emergency_power_subagents.md`.
- Mermaid graph from the accepted artifact-producing run:
  ```mermaid
  graph TD
    root["root\nroot\nexpanded"]
    task_1["task_1\nsubtask\nresolved"]
    task_2["task_2\nevidence_need\nblocked"]
    task_3["task_3\nsynthesis\nresolved"]
    task_4["task_4\nevidence_need\nresolved"]
    root --> task_1
    root --> task_2
    root --> task_3
    root --> task_4
  ```
- Observed subagent calls:
  - `evidence` was called once for current NZ product availability and returned
    unavailable.
  - `algorithmic` was not called in the artifact-producing run.
  - `trace_summary.subagent_calls` was `1`.
- Observed behavior:
  - The packet status matched the fixture expected status, `partial`.
  - The graph stayed flat at depth 1; no second- or third-tier nodes were
    created.
  - Arithmetic was calculated inside LLM-generated node text, not by the
    deterministic algorithmic subagent.
  - The final answer recommended a specific product while also carrying
    unresolved evidence/source-grounding items.
- Failure modes:
  - Missing required `algorithmic` subagent invocation.
  - No recursive decomposition into child or grandchild nodes.
  - LLM-owned arithmetic remained possible for arithmetic-like branches.
  - Product recommendation wording was too confident relative to the evidence
    subagent's unavailable result.
- Consolidated root cause:
  - The current resolver implementation supports recursive graph validation
    but not recursive graph execution. Semantic planner output is still mapped
    to root-level children only, and active node resolution can update one node
    or call one subagent but cannot expand a complicated active node into
    bounded child work before resolving it.
- Architecture review:
  - Case 32 exposes the precise gap in the proposed architecture: recursive
    dissection and leaf-level subagent ownership are design requirements, but
    the current implementation only demonstrates flat decomposition plus
    optional per-node subagent dispatch.
  - This is not a web-source availability failure. Source availability affects
    the product branch, but the missing recursive depth and missing algorithmic
    subagent call are resolver architecture failures.
- Fix classification:
  - Focused harness fix completed: case 32 is a version-controlled live LLM
    test entry that records observed subagent calls.
  - Architectural fix required: add an explicit node-expansion response path
    for active nodes to create bounded child nodes up to `max_depth`.
  - Architectural fix required: arithmetic-like leaf nodes with available
    operands must be routed through the deterministic algorithmic subagent
    instead of being calculated by LLM prose.
  - Architectural fix required: final synthesis must treat unavailable
    evidence-subagent results as product-fact blockers, not as support for a
    confident product recommendation.
- Decision:
  - Case 32 is not accepted.
  - Do not treat the current resolver as recursive or subagent-robust. The
    result should drive the next architecture fix before this case is rerun.

### 2026-06-30 case 32 closure pass

- Root-cause fixes applied:
  - Added bounded active-node expansion so complicated nodes can recursively
    create child tasks under `max_depth` and `max_nodes`.
  - Replaced operation-specific arithmetic payloads with
    `evaluate_expression` over a caller-prepared numeric expression.
  - Added one bounded algorithmic-node repair call. If an algorithmic node is
    answered with prose, the resolver asks the same node resolver stage for the
    declared subagent IO once, then still fails closed if the repair does not
    produce a valid request.
  - Kept unit conversion outside the algorithmic subagent. The node resolver
    prompt states that W is multiplied by hours, while Wh is already energy and
    must be added once.
  - Removed unstructured pre-collected web output from live resolver prompt
    context. Review web output remains in artifacts, but product facts must
    come through resolver evidence nodes before they can support source-backed
    claims.
  - Added synthesis dependency gating so synthesis nodes cannot resolve while
    earlier prerequisite branches remain pending, blocked, or unanswerable.
  - Added graph fact hygiene so untraced evidence facts are downgraded from
    source-backed facts to assumptions before packet projection.
- Deterministic verification:
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_fixture.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_service.py -q`
    passed with 33 tests.
- Live case 32 command:
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_review_case_32_emergency_power_subagents -q -s -m live_llm`
    passed.
- Accepted case 32 artifact:
  - `test_artifacts/complex_task_resolver/case_32_emergency_power_subagents.md`
- Accepted case 32 trace:
  - `test_artifacts/llm_traces/complex_task_resolver_live_llm__case_32_emergency_power_subagents__20260630T011721871965Z.json`
- Accepted Mermaid graph:
  ```mermaid
  graph TD
    root["root\nroot\nexpanded"]
    task_1["task_1\nsubtask\nexpanded"]
    task_1_1["task_1_1\nalgorithmic_task\nresolved"]
    task_1_2["task_1_2\nevidence_need\nblocked"]
    task_2["task_2\nsynthesis\nblocked"]
    task_3["task_3\nsynthesis\nblocked"]
    root --> task_1
    root --> task_2
    root --> task_3
    task_1 --> task_1_1
    task_1 --> task_1_2
  ```
- Observed accepted behavior:
  - The graph recursed to depth 2 and resolved one arithmetic leaf through the
    internal `algorithmic` subagent.
  - The arithmetic expression was
    `(45 * 6) + 12 + (12 * 6) + 60 = 414`, correctly treating the laptop's
    `60 Wh` battery as energy rather than multiplying it by two hours.
  - Evidence subagent calls remained unavailable in the standalone review
    environment. This is an external evidence-provider availability blocker,
    not an arithmetic or graph-recursion failure.
  - Synthesis nodes remained blocked behind unresolved product-evidence
    prerequisites, and the final packet stayed `partial` with source-backed
    facts withheld.
- Residual risk:
  - Case 32 proves the architecture path for recursive decomposition,
    resolver-local subagent invocation, source-grounding downgrade, and
    dependency-aware synthesis blocking. It does not prove product evidence
    retrieval quality because the review evidence provider remains
    unavailable by design in this pass.
- Decision:
  - Case 32 is accepted as an architectural pass for the resolver framework.
  - Do not connect `answer_investigation` to L2d until the remaining real LLM
    review cases are run one at a time and inspected under the same
    failure-mode analysis rule.

### 2026-06-30 production-readiness hardening pass

- Independent review pass 1 was negative. Findings:
  - Blocker: the public entrypoint could still raise on malformed internal
    planner/stage output instead of returning a declared
    `ComplexTaskResolutionPacketV1`.
  - Blocker/process gate: comprehensive real LLM review and rejected case 03
    remain unresolved, so L2d integration and full Phase 1 completion remain
    blocked.
  - High: package root exported internal subagents and graph traversal helpers,
    weakening the public IO boundary.
- Remediation:
  - `resolve_complex_task(request, context, options=None)` now wraps public
    input validation and internal execution with a production failure boundary.
    Invalid public input returns a validated `failed` packet with
    `failure_stage=input_validation`; malformed internal stage output returns a
    validated `failed` packet with `failure_stage=internal_resolution`.
  - Planner semantic task validation now reports missing task fields through
    `ComplexTaskValidationError` rather than raw dictionary exceptions.
  - Package-root exports were narrowed to public constants, validators,
    projection helper, `ComplexTaskValidationError`, and
    `resolve_complex_task`; internal subagents and traversal helpers are
    imported only from their implementation modules by deterministic tests.
  - The plan/options contract was corrected to document
    `ComplexTaskResolverOptionsV1["limits"]` as `dict[str, int]`, matching the
    validator.
- Focused verification:
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_service.py::test_resolve_complex_task_returns_failed_packet_on_internal_error -q`
    passed. The test intentionally logs the internal planner validation error
    and asserts the returned packet is `status=failed`.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\complex_task_resolver\__init__.py src\kazusa_ai_chatbot\complex_task_resolver\service.py src\kazusa_ai_chatbot\complex_task_resolver\contracts.py src\kazusa_ai_chatbot\complex_task_resolver\stages.py src\kazusa_ai_chatbot\complex_task_resolver\subagents.py src\kazusa_ai_chatbot\complex_task_resolver\algorithmic.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_service.py tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_fixture.py`
    passed.
- Deterministic resolver suite:
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_service.py tests\test_complex_task_resolver_fixture.py -q`
    reported `33 passed`.
- Static boundary verification:
  - `rg -n "AlgorithmicSubagent|ComplexTaskEvidenceSubagent|UnavailableEvidenceSubagent|find_next_active_node" src\kazusa_ai_chatbot\complex_task_resolver\__init__.py`
    returned no matches.
  - `rg -n "COMPLEX_TASK_RESOLVER_RUNTIME_VERSION|ComplexTaskResolverRuntimeV1|validate_complex_task_resolver_runtime|resolve_complex_task\([^\n]*runtime|_collect_web_evidence|_LiveJsonStageInvoker" src\kazusa_ai_chatbot\complex_task_resolver tests development_plans\active\short_term\complex_task_resolver_capability_plan.md`
    returned no matches.
  - `rg -n "expected_final_answer|minimum_viable_answer|performance_reference_summary|expected_graph_trace|forbidden_failure_modes|ctr_0|RTX5090|R9700|Qwen3\.6|gemma4" src\kazusa_ai_chatbot\complex_task_resolver`
    returned no matches.
  - `rg -n "answer_investigation|complex_task_resolver|resolve_complex_task" src\kazusa_ai_chatbot --glob "!src/kazusa_ai_chatbot/complex_task_resolver/**" --glob "!**/__pycache__/**"`
    returned no matches.
- Residual gates:
  - Stage 5 and Stage 6 remain open because the remaining real LLM cases have
    not been run under user command.
  - Stage 7 remains open because broad regression still includes the unrelated
    action-selection prompt contract failure
    `test_action_selection_prompt_follows_cognition_prompt_structure`
    (`26 passed, 1 failed` in
    `tests/test_action_selection_prompt_contract.py`
    and `tests/test_persona_supervisor2_action_selection.py`).
  - Stage 9 is not signed because integration remains intentionally absent and
    must stay blocked pending future user instruction.

### 2026-06-30 independent review pass 2 remediation

- Independent review pass 2 was negative. Findings:
  - Blocker: production still accepted undeclared full-graph planner output and
    legacy collapse `node_updates`/`collapse_events` shortcut output.
  - Blocker: public structural options were positive integers but not capped,
    and case 32 live-test options exceeded the planned Phase 1 caps.
  - High: subagent execution discarded public resolver context before calling
    evidence helpers, and prose-only evidence could still support an
    `answered` packet when no `source_backed_facts` were present.
  - Medium: the active plan still had stale context, trace, file-layout, and
    fixture-count wording.
  - Medium: the live LLM harness generated Markdown review reports instead of
    raw structured evidence.
- Remediation:
  - Removed production support for planner full-graph shortcut output; planner
    LLM output must now be semantic `tasks` mapped by deterministic service
    code.
  - Removed production support for collapse-stage `node_updates` and raw
    `collapse_events`; collapse LLM output must now be a semantic
    `collapse_decision`.
  - Added `OPTION_LIMIT_CAPS` in the public options validator:
    `max_iterations=4`, `max_nodes=8`, `max_depth=3`,
    `max_node_attempts=3`, and `max_subagent_attempts=1`.
  - Updated live case 32 configuration to stay within the hard caps:
    `max_nodes=8`, `max_depth=3`, and `max_iterations=4`.
  - Passed validated resolver context into resolver-local subagents, including
    `time_context`, `available_evidence`, parent-chain summary, and sibling
    summaries.
  - Tightened source-grounding answerability checks so a resolved or collapsed
    `evidence_need` node with prose result material but no evidence refs or
    source observation ids blocks `answered` status unless structured
    evidence is available in context.
  - Changed live LLM case artifacts from harness-authored Markdown to raw
    structured JSON evidence with the Mermaid graph included as data. Human
    review remains agent-authored after inspection.
  - Updated the plan ICD sections for context shape, subagent context,
    trace-summary type, file layout, cap name, fixture case count, and live
    review artifact ownership.
- New deterministic coverage:
  - Public options reject values over hard caps.
  - Planner full-graph shortcut returns a validated `failed` packet.
  - Legacy collapse node-update shortcut returns a validated `failed` packet.
  - Evidence subagent receives public resolver context and prose-only evidence
    keeps the packet `partial`.
- Verification:
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py::test_options_contract_rejects_external_stage_configuration tests\test_complex_task_resolver_service.py::test_resolve_complex_task_rejects_planner_graph_shortcut tests\test_complex_task_resolver_service.py::test_resolve_complex_task_rejects_legacy_collapse_update_shape tests\test_complex_task_resolver_service.py::test_evidence_subagent_receives_public_resolver_context -q`
    reported `4 passed`.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\complex_task_resolver\__init__.py src\kazusa_ai_chatbot\complex_task_resolver\service.py src\kazusa_ai_chatbot\complex_task_resolver\contracts.py src\kazusa_ai_chatbot\complex_task_resolver\stages.py src\kazusa_ai_chatbot\complex_task_resolver\subagents.py src\kazusa_ai_chatbot\complex_task_resolver\algorithmic.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_service.py tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_fixture.py tests\test_complex_task_resolver_live_llm.py`
    passed.
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_service.py tests\test_complex_task_resolver_fixture.py -q`
    reported `36 passed`.
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py --collect-only -q -m live_llm`
    collected 5 tests without executing real LLM cases.
- Static verification:
  - Production shortcut grep returned no matches:
    `rg -n "COMPLEX_TASK_RESOLVER_RUNTIME_VERSION|ComplexTaskResolverRuntimeV1|validate_complex_task_resolver_runtime|_collect_web_evidence|_LiveJsonStageInvoker|node_updates" src\kazusa_ai_chatbot\complex_task_resolver`.
  - Planner graph shortcut grep returned no matches across production code and
    tests.
  - Plan/live-harness stale wording grep returned no matches:
    `rg -n "max_nodes=12|max_iterations=8|case_.*\.md|Write one human-readable|human-readable artifacts|resolver_context_summary|time_context: str|trace_summary\": list|31\s+concrete cases|max_internal_iterations" tests\test_complex_task_resolver_live_llm.py development_plans\active\short_term\complex_task_resolver_capability_plan.md`.
  - Root export, fixture-answer leakage, and external integration greps
    returned no matches.

### 2026-06-30 independent review approval

- Follow-up independent review after pass-2 remediation returned no blocking
  findings.
- Reviewer confirmed the prior blockers were resolved:
  - planner shortcut graph output is rejected in favor of semantic `tasks`;
  - collapse requires `collapse_decision`;
  - public options are capped;
  - subagents receive validated resolver context;
  - prose-only evidence nodes downgrade answerability;
  - live harness case 32 stays within caps and writes raw JSON artifacts;
  - static checks found no production fixture-answer leakage, retired shortcut
    paths, package-root exports of internal subagents/helpers, or external
    L2d/live workflow wiring.
- Reviewer noted one non-blocking documentation cleanup about `eval` wording;
  the acceptance text was corrected to permit only AST-validated
  `evaluate_expression` with empty `__builtins__` and safe numeric helpers.
- Review decision:
  - Stage 8 code-review gate is signed for standalone production readiness.
  - Scope is limited to the standalone module/public packet boundary.
  - Comprehensive real LLM review remains unexecuted except for previously
    recorded individual cases and must resume only under user command.
  - L2d/live workflow integration remains blocked pending separate explicit
    user instruction.

### 2026-06-30 L2d integration sequencing update

- User approved bringing L2d integration steps forward before the comprehensive
  real LLM review.
- Architecture decision recorded:
  - `public_answer_research` is the L2d-visible public/current/external answer
    research capability and routes to
    `complex_task_resolver.resolve_complex_task(...)`.
  - `local_context_recall` is the L2d-visible local/private/context recall
    capability and routes to existing RAG2 via
    `run_rag_evidence_for_persona_state(...)`.
  - Former L2d-visible `web_evidence` collapses into
    `public_answer_research`.
  - Former L2d-visible `rag_evidence` is represented by the clearer
    `local_context_recall` name; the underlying RAG2 package is not renamed.
  - `answer_investigation` is rejected as a runtime capability name.
- Plan sequencing changed:
  - Stage 5 is now the narrow L2d capability-contract integration gate.
  - Live LLM inspection cases and the comprehensive real LLM review run after
    the L2d-facing contract exists, so the review exercises the final
    capability vocabulary.
  - Broad live enablement, rollout, and additional deprecation remain blocked
    until the real LLM evidence is accepted by the user.
- This evidence entry supersedes older execution notes that said all L2d
  integration must remain absent until after real LLM review. It does not
  authorize production-code changes by itself; implementation still requires
  the user's explicit execution instruction.

### 2026-06-30 Stage 5 L2d capability contract integration

- User narrowed Stage 5 proof to real LLM tests focused on L2d route choice:
  given correct L2d input, L2d must pick the correct output capability. Other
  tests are not quality proof for this stage except as contract executability
  checks.
- Production changes:
  - `src/kazusa_ai_chatbot/cognition_resolver/contracts.py` now accepts
    `local_context_recall` and `public_answer_research` as canonical
    L2d-visible resolver capabilities.
  - `src/kazusa_ai_chatbot/cognition_chain_core/action_selection.py` presents
    positive affordance descriptions for local/private recall versus
    public/current/external answer research.
  - `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py` routes
    `local_context_recall` to existing RAG2 persona evidence and routes
    `public_answer_research` to
    `kazusa_ai_chatbot.complex_task_resolver.resolve_complex_task(...)`
    through declared complex-resolver request/context/options IO.
  - `src/kazusa_ai_chatbot/complex_task_resolver/contracts.py` declares `l2d`
    as a valid complex-resolver request source.
  - `src/kazusa_ai_chatbot/self_cognition/runner.py` updates resolver
    evidence-count bookkeeping to the final capability names.
  - Capability documentation updated in cognition-resolver, RAG, and
    complex-resolver READMEs.
- Legacy L2d tests:
  - Updated L2d/cognition-resolver tests that formerly expected
    `rag_evidence` or `web_evidence` as valid L2d outputs.
  - Old-name grep over focused L2d route tests now matches only
    prompt-forbidden assertions, not accepted L2d route outputs.
- Focused deterministic contract checks:
  - `venv\Scripts\python -m py_compile` over changed production Python files
    passed.
  - `venv\Scripts\python -m pytest tests\test_action_selection_prompt_contract.py tests\test_cognition_chain_core_action_selection.py tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_l2d_contract.py tests\test_cognition_resolver_loop.py::test_public_answer_research_uses_complex_resolver -q`
    passed: 29 passed.
- Real LLM L2d route-choice tests, run one case at a time:
  - Local/private recall:
    `tests\test_l2d_unknown_context_resolver_live_llm.py::test_l2d_local_reference_first_iteration_emits_local_context_recall`
    passed. Observed `resolver_kinds=["local_context_recall"]`.
    Trace:
    `test_artifacts\llm_traces\l2d_unknown_context_resolver_live_llm__local_reference_hao_die_you_no_loaded_context__20260630T072207128816Z.json`.
    Failure-mode analysis: the original ambiguous "unknown term" fixture led
    L2d to choose `public_answer_research`, which was architecturally
    reasonable because the prompt did not expose the supposedly local/private
    fact. The fixture was corrected to a true prior-chat local reference, and
    the affordance descriptions were tightened without negative prompt rules.
  - Non-resolver visible reply:
    `tests\test_l2d_unknown_context_resolver_live_llm.py::test_l2d_gibberish_routes_to_speak_confusion`
    passed. Observed `action_kinds=["speak"]`, `resolver_kinds=[]`.
    Trace:
    `test_artifacts\llm_traces\l2d_unknown_context_resolver_live_llm__gibberish_keyboard_mash__20260630T072135887636Z.json`.
    Failure-mode analysis: the first run returned no action because the test
    hid `speak` from `capabilities.action_affordances`; this was a harness
    error, not an L2d routing failure. The frozen L2d input now exposes
    `speak` when the expected route is a visible reply.
  - Public answer research:
    `tests\test_l2d_unknown_context_resolver_live_llm.py::test_l2d_internet_term_emits_evidence_request`
    passed. Observed `resolver_kinds=["public_answer_research"]`.
    Trace:
    `test_artifacts\llm_traces\l2d_unknown_context_resolver_live_llm__internet_term_nanimi.json`.
    Failure-mode analysis: the case is a public/searchable phrase, so
    `public_answer_research` is the correct route and confirms the replacement
    for the former L2d-visible web evidence path.
- Residual observation:
  - The local-reference live case logs `L2d dropped invalid goal progress:
    missing_user_inputs: expected list`. This is route-adjacent L2d output
    quality debt, but it does not affect the selected capability and is outside
    the user's Stage 5 route-choice-only criterion.
  - A broader deterministic batch against unrelated prompt-contract files
    produced 3 pre-existing prompt-text failures while 108 tests passed. These
    failures are not part of the Stage 5 route-choice proof and were not
    addressed in this cutover.

### 2026-06-30 Stage 5A executable follow-up tasks

- User added two execution requirements:
  - if a local node produces resolver-owned structured recommendations, create
    bounded child nodes to execute and collect the answer;
  - if top-level synthesis produces resolver-owned structured
    recommendations, the complex resolver should address them internally
    within graph limits before returning to cognition.
- Plan and ICD changes:
  - Added `ComplexTaskFollowupTaskV1` as an internal executable control
    contract.
  - Recorded that prose `recommended_next_iteration` is semantic-only output
    for cognition/review and must never be parsed or executed by deterministic
    code.
  - Documented active-node follow-ups as source-node children, synthesis
    follow-ups as root-level children, and all follow-up creation as bounded by
    max iterations, max nodes, max depth, and per-source caps.
- Production implementation:
  - Production subagent `Hubble`
    (`019f184d-f5a9-7860-aea5-15c0c5a117dc`) implemented the initial Stage 5A
    code changes under `src/kazusa_ai_chatbot/complex_task_resolver/`.
  - `contracts.py` now validates `complex_task_followup_task.v1`.
  - `stages.py` exposes the structured `followup_tasks` prompt channel for
    active-node and bottom-up synthesis stages.
  - `service.py` creates active-node follow-up child nodes, creates synthesis
    follow-up root children with one additional traversal pass, records
    follow-up creation/rejection events, and rejects follow-ups that cannot be
    traversed inside remaining graph limits.
  - The algorithmic operand provenance path was corrected so
    `recommended_next_iteration` text cannot support deterministic arithmetic
    operands.
- Deterministic verification:
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_service.py -q`
    reported `47 passed`.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\complex_task_resolver\contracts.py src\kazusa_ai_chatbot\complex_task_resolver\__init__.py src\kazusa_ai_chatbot\complex_task_resolver\stages.py src\kazusa_ai_chatbot\complex_task_resolver\service.py tests\test_complex_task_resolver_service.py tests\test_complex_task_resolver_live_llm.py`
    passed.
  - Static recommendation-control grep:
    `rg "recommended_next_iteration.*followup|followup.*recommended_next_iteration|search\(|startswith\(|in recommended_next_iteration" src\kazusa_ai_chatbot\complex_task_resolver`
    matched only unrelated `startswith` validation checks and the prompt
    example in `stages.py`; no production control flow parsed prose
    recommendations.
  - New focused coverage includes follow-up contract validation, active-node
    child creation, synthesis continuation, limit rejection, prose-only
    recommendation non-execution, iteration-cap rejection, non-string semantic
    row tolerance, and arithmetic provenance rejection for recommendation-only
    operands.
- Real LLM public-IO proof:
  - Command:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_followup_tasks_are_emitted_and_executed -q -s -m live_llm`.
  - Trace:
    `test_artifacts\llm_traces\complex_task_resolver_live_llm__stage_5a_followup_tasks__20260630T115719884281Z.json`.
  - Observed follow-up events:
    - active node `task_2` created synthesis child `task_2_1` to compare
      `1140 Wh` demand against `1500 Wh` capacity;
    - active node `task_2_1` created algorithmic child `task_2_1_1` to
      calculate total router and laptop energy use.
  - Resolver output included sourced numeric facts for `120 W`, `8 hours`,
    `60 W`, `3 hours`, `1500 Wh`, and deterministic calculation
    `(120 * 8) + (60 * 3) = 1140 Wh`.
  - Contract judgment: the public-IO real LLM test proves the model can emit
    structured `followup_tasks` and the resolver executes them internally.
  - Residual quality observation: the final packet still listed
    `pending: Synthesize whether the combined load fits within the power
    station budget` even though the summary and known facts contained enough
    comparison context. This is a model/prompt-quality residual for later
    case-level review, not a failure of the Stage 5A executable-follow-up
    control channel.
- Independent review:
  - First Stage 5A review rejected approval because arithmetic provenance could
    use semantic `recommended_next_iteration` text. This was fixed by removing
    that field from `_algorithmic_source_node_text(...)` and adding
    `test_resolve_complex_task_blocks_algorithmic_recommendation_provenance`.
  - Follow-up independent review by `Sartre`
    (`019f185c-4e71-7521-be34-84c8f07a1ba2`) returned no findings and
    approved Stage 5A for production readiness in the reviewed scope.
  - Reviewer independently re-ran the deterministic suite (`47 passed`),
    py-compile, and the recommendation-control grep, and agreed that the
    residual live trace synthesis gap is later prompt/model quality debt
    rather than a Stage 5A control-channel blocker.

### 2026-07-01 semantic LLM projection and subagent registry hardening

- User identified that resolver LLM prompts and outputs still exposed
  deterministic content such as schema markers, graph identifiers,
  source-node handles, operational state, trace/cache vocabulary, and hardcoded
  subagent capability text.
- Fixes implemented:
  - Replaced hardcoded resolver-local subagent prompt text with a
    WebAgent3-style module-owned discovery registry under
    `complex_task_resolver.subagent`.
  - Moved subagent ownership, supported actions, default actions, and factories
    into registry metadata consumed by the service path.
  - Rewrote resolver LLM prompts so active-node, collapse, and synthesis stages
    request semantic decisions, semantic candidate text, and semantic
    `continuation_tasks` rather than internal graph/subagent/follow-up
    envelopes.
  - Added deterministic mappers from semantic LLM output to internal graph
    updates, subagent request envelopes, follow-up task contracts, and collapse
    targets.
  - Removed graph identifiers, attempt counters, operational state, raw trace
    summary, raw evidence contract versions, and structural options from
    prompt-facing input projections.
  - Added key-level validation that rejects deterministic transport fields in
    semantic LLM outputs.
  - Updated the module README to document typed public/internal contracts
    versus semantic LLM-stage projection.
- Focused tests added or updated:
  - `tests/test_complex_task_resolver_prompt_contract.py`
  - `tests/test_complex_task_resolver_service.py`
- Verification:
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_prompt_contract.py tests\test_complex_task_resolver_service.py -q`
    passed with `39 passed`.
  - `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_fixture.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_prompt_contract.py tests\test_complex_task_resolver_service.py -q`
    passed with `57 passed`.
  - `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\complex_task_resolver tests\test_complex_task_resolver_prompt_contract.py`
    passed.
- Independent review:
  - Reviewer `Bacon`
    (`019f18a2-4abf-7ad2-9401-f29b33aa5c3d`) returned blocking findings:
    production LLM stage outputs could still bypass the semantic-output guard
    by using internal envelopes; subagent result metadata could leak
    status/cache vocabulary into semantic prompt projections; subagent repair
    prompt payloads included internal normalized responses; tests did not catch
    the production-path bypass; and `self_cognition/README.md` still mentioned
    the old `rag_evidence` capability name.
  - Remediation completed:
    production node/synthesis/collapse handlers now reject internal envelopes;
    patched deterministic tests may still use internal envelopes only when
    stage handlers are monkeypatched; subagent-derived semantic boundary notes
    no longer project status/cache wording; repair prompts receive a semantic
    previous-attempt summary instead of internal response objects; prompt
    contract tests cover production-normalizer rejection, repair context, and
    subagent notes; self-cognition documentation uses `local_context_recall`
    and `public_answer_research`.
  - Post-remediation verification:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_fixture.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_prompt_contract.py tests\test_complex_task_resolver_service.py -q`
    passed with `57 passed`.
  - Follow-up independent review by `Bacon`
    (`019f18a2-4abf-7ad2-9401-f29b33aa5c3d`) returned no remaining blocking
    findings. The reviewer independently re-ran the focused quick suite
    (`39 passed`) and full focused resolver suite (`57 passed`) and confirmed
    the production semantic-output guards, subagent boundary note cleanup,
    repair prompt projection, registry-owned subagent capability text, prompt
    contract coverage, and self-cognition documentation remediation.
- Signoff:
  - The resolver now enforces the local LLM projection rule for its production
    prompt path: deterministic code owns schema, graph bookkeeping,
    operational state, source-node provenance, trace/debug material,
    lower-layer cache metadata, and internal subagent envelopes.
  - Public typed IO remains versioned and debug traces remain available in the
    returned packet for read-only inspection.
  - Real LLM review remains gated and must be run one case at a time under user
    command.

### 2026-07-01 case 04 live review remediation

- Case:
  - `ctr_004_local_llm_gpu_buy`
  - Command:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_review_case_04_local_llm_gpu_buy -q -s -m live_llm`
- Initial failure:
  - The resolver failed before useful investigation with
    `ComplexTaskValidationError: collapse semantic output must not use graph
    targets`.
  - Root cause: `_review_collapse(...)` created its no-candidate fallback using
    internal `target_node_id` even though the production collapse guard accepts
    only semantic `matching_candidate` output.
  - Fix: no-candidate fallback now uses `matching_candidate: ""`.
  - Regression coverage:
    `test_no_candidate_collapse_fallback_uses_semantic_shape`.
- Second failure:
  - The live harness passed but produced no evidence. The active
    `public_evidence` node repeatedly expanded into an equivalent
    `public_evidence` child, consuming graph limits before the evidence
    subagent ran.
  - Root cause: resolver-local subagent fallback covered direct prose updates
    for owned nodes, but not unproductive single-child self-expansion.
  - Fix: when a node owned by a resolver-local subagent expands into one child
    with the same objective and node kind, the service treats that as
    unproductive decomposition and invokes the owned subagent fallback.
  - Regression coverage:
    `test_owned_evidence_node_self_expansion_uses_subagent_fallback`.
- Final live result:
  - Case 04 passed the harness and produced one resolved evidence subagent call.
  - The resolver identified candidate GPU facts for RTX 5090, RTX 4090,
    RX 7900 XTX, RTX 5070 Ti, Apple M5 Max/Ultra, and DGX-class options.
  - Quality judgment: output is improved but still partial. It lacks current
    US prices, power/TDP comparison, performance-per-dollar comparison, and a
    final budget recommendation.
  - Human-readable review:
    `test_artifacts\complex_task_resolver\case_04_local_llm_gpu_buy_review.md`.
- Verification:
  - Focused deterministic resolver suite passed with `59 passed`.
  - `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\complex_task_resolver tests\test_complex_task_resolver_prompt_contract.py tests\test_complex_task_resolver_service.py`
    passed.
  - `git diff --check` passed.
  - Prompt forbidden-token grep against
    `src\kazusa_ai_chatbot\complex_task_resolver\stages.py` returned no
    matches.

### 2026-07-01 final independent review and closeout

- User accepted the current complex-resolver behavior as good enough and
  requested final review, cleanup, commit, and plan closure.
- Evidence accepted for closeout:
  - `test_artifacts/complex_task_resolver/comprehensive_32_case_review.md`
  - `test_artifacts/complex_task_resolver/failure_mode_fix_tracker.md`
  - per-case JSON artifacts under `test_artifacts/complex_task_resolver/`
- Independent review:
  - Reviewer `Bernoulli` (`019f1b8d-138b-7af0-bc52-82f803f72fd7`)
    initially found three Important blockers: traversal cap drift, missing
    run-level evidence-call cap, and semantic continuation accepting internal
    `kind`.
  - Fixes: `max_iterations=4` was restored; `MAX_RAG_CALLS_PER_RUN=2` was
    added with `evidence_calls` trace and budget result; semantic continuation
    tasks now require `work_type`; stale L2d mocked packet data was updated to
    the current node semantic IO.
  - Reviewer minor fail-fast issue around optional `evidence_calls` lookup was
    fixed to use required trace indexing.
  - Final reviewer verdict: approved, no findings.
- Verification:
  - `git diff --check` passed.
  - Compile verification passed for the complex resolver, WebAgent3, and
    focused test modules.
  - Focused resolver/WebAgent3 suite passed with `131 passed`.
  - L2d/cognition boundary suite passed with `30 passed`.
  - Targeted post-review regression passed with `3 passed`.
- Closure:
  - Stages 6 and 7 are accepted based on user-approved good-enough review
    evidence; residual live-LLM partials and failures are tracked and do not
    authorize broad live rollout.
  - Stage 11 remains a completed block: broad live enablement is still blocked
    unless a future explicit plan and user command changes it.
