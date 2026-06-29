# complex task resolver capability plan

## Summary

- Goal: Add and prove a distinct standalone `complex_task_resolver` specialist
  module for complicated questions before any L2d or live workflow routing is
  allowed.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: standalone/shadow validation first; integration is
  explicitly deferred until a comprehensive real LLM review is completed and
  the user instructs the next integration step.
- Highest-risk areas: confusing this specialist with the existing cognition
  resolver, unbounded recursive decomposition, cache staleness, semantic
  collapse mistakes, LLM-only arithmetic, prompt bloat, response-path latency,
  review-time web/RAG/LLM dependency availability, and dialog treating
  task-resolver artifacts as persona wording.
- Acceptance criteria: the standalone specialist returns a validated
  prompt-safe packet through a test/review harness, comprehensive real LLM
  review artifacts show sufficient robustness, no L2d/runtime integration has
  been added, and all current resolver/RAG/dialog ownership boundaries remain
  intact.

## Context

The existing `kazusa_ai_chatbot.cognition_resolver` package is the live
cognition recurrence controller. It initializes `ResolverCycleStateV1`, runs
the L1/L2/L2d cognition stack, executes one immediate
`ResolverCapabilityRequestV1` per cycle, records `ResolverObservationV1` rows,
and projects prompt-safe observations back into the next cognition pass.

This plan adds a separate specialist module. The new module is not the
cognition resolver, not a replacement for `stage_1_goal_resolver`, and not a
dialog generator. It is intended to become a capability peer to demand-driven
RAG evidence, but this plan does not connect it to L2d or the live workflow.

The Phase 1 validation path is:

```text
deterministic tests / standalone review harness
  -> complex_task_resolver builds or reuses a task graph
  -> specialist returns ComplexTaskResolutionPacketV1
  -> review artifact records graph, collapse/cache decisions, node outputs,
     final packet, and human judgment
  -> user explicitly decides whether a later integration plan may begin
```

The future integration path, outside this plan, is expected to be:

```text
L1/L2/L2d cognition
  -> L2d emits resolver_capability_request(kind=complex_task_resolution)
  -> cognition_resolver executes one bounded specialist call
  -> complex_task_resolver builds or reuses a task graph
  -> specialist returns ComplexTaskResolutionPacketV1
  -> ResolverObservationV1 stores prompt-safe packet summary
  -> next L1/L2/L2d pass sees the packet
  -> L2d selects speak or no-response/private finalization
  -> L3/dialog renders final character wording from the packet
```

That future path must not be implemented until the review gate in this plan is
complete and the user explicitly instructs integration.

The user expectation driving this plan:

- complex questions become a top-down graph of tasks and subtasks;
- each specialist cycle addresses one active task node at a time;
- resolved node results are collected bottom-up into a final answer;
- semantically equivalent branches can collapse;
- node results are cached and may be reused for similar future nodes;
- the component should eventually be strong enough to reduce reliance on
  cognition-chain-driven multi-cycle reasoning for complex questions, but that
  replacement is not part of this plan.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing resolver capability
  contracts, complex-task prompts, LLM call budgets, graph decomposition, cache
  behavior, RAG/tool handoff, or dialog handoff.
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
  dialog-ready answer packet for review. Future L2d and L3/dialog integration
  will decide whether and how to speak only after a later user-approved plan.
- Do not add `complex_task_resolution` to L2d prompts, allowed resolver
  capabilities, action-selection contracts, cognition resolver routing,
  persona graph wiring, or any live response workflow in this plan.
- Passing deterministic tests, focused tests, or live LLM tests is not
  sufficient authorization for integration. The comprehensive real LLM review
  must be completed, the user must inspect or explicitly accept the evidence,
  and the user must instruct the integration step before any L2d/workflow
  connection begins.
- The new specialist must not execute platform delivery, scheduler actions,
  database writes, filesystem operations, shell commands, arbitrary HTTP tools,
  or adapter callbacks.
- Deterministic code owns graph shape validation, node caps, depth caps, cache
  key construction, cache invalidation policy, timeouts, status mapping, and
  prompt-safe projection.
- LLM stages own decomposition, active-node semantic resolution, collapse
  judgment over bounded candidate sets, and bottom-up synthesis text.
- Arithmetic, scheduling math, budget math, weighted scoring, token/call
  estimates, cache-savings estimates, and benchmark normalization must be
  executed by a typed deterministic algorithmic subagent. LLM stages may request
  the operation and provide semantic context, but they must not be treated as
  the calculation authority.
- Do not add arbitrary expression evaluation, Python `eval`, shell execution,
  notebook execution, or a free-form symbolic-math engine in Phase 1. The
  algorithmic subagent must expose explicit operations with schema-validated
  operands and deterministic outputs.
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
  payloads, cache keys, embeddings, raw prompt text, or internal worker names
  in cognition-visible resolver context.
- New specialist LLM calls must have hard caps, timeout behavior, and a
  documented context budget before approval.
- Before any live LLM inspection or comprehensive real LLM review, run and
  record a review-environment preflight covering the resolver LLM route,
  JSON repair route, web search/read availability, network access, and any
  RAG/Mongo/embedding dependency used by that review run. Missing dependencies
  must become explicit per-case blockers or skipped-live-evidence notes; they
  must not be counted as resolver quality passes.
- Live LLM tests must run one case at a time with output inspected.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` uses parent-led native subagent execution
  unless the user explicitly approves fallback execution.

## Must Do

- Create a new `kazusa_ai_chatbot.complex_task_resolver` package with its own
  README and public entrypoint.
- Keep the Phase 1 entrypoint standalone and callable only from deterministic
  tests, local review harnesses, and live LLM inspection tests.
- Document the future `complex_task_resolution` resolver-capability wrapper as
  a non-executed integration design, but do not implement or expose it.
- Keep all new task-graph contracts under the new module.
- Add `ComplexTaskGraphV1`, `ComplexTaskNodeV1`,
  `ComplexTaskResolutionPacketV1`, and `ComplexTaskCacheEntryV1` contracts with
  validators.
- Implement bounded graph traversal where each internal specialist iteration
  addresses one active task node.
- Add a review-only runtime dependency boundary for stage LLM invokers,
  evidence retrieval, clocks, and limits. These runtime dependencies are not
  prompt-facing data and must be injected by tests or review harnesses.
- Add a standard complex-task subagent IO contract under the new module,
  following the same public IO principle as `WebAgent3.run(...)`: a bounded
  task/request plus context enters the subagent, and a resolved/result/attempts
  plus cache/trace envelope comes back.
- Add a review-only evidence subagent boundary for active `evidence_need`
  nodes. It may call existing public WebAgent3/RAG entrypoints only from tests
  or review harnesses, records dependency availability, and returns bounded
  source-backed facts or explicit unavailable/blocker status.
- Add a deterministic algorithmic subagent for `algorithmic_task` nodes. It
  must cover Phase 1 arithmetic review cases without relying on LLM arithmetic:
  schedule/duration math, budget math, weighted scoring, token/call estimates,
  cache savings, percentage/range comparisons, and benchmark normalization
  over already-sourced numbers.
- Implement bounded semantic collapse over a small deterministic candidate set.
- Implement a process-local node cache for Phase 1.
- Provide a compact prompt-safe packet projection helper for review artifacts
  and future integration, but do not project into `resolver_context` yet.
- Add focused deterministic tests for contracts, graph validation, cache reuse,
  collapse behavior, evidence-subagent unavailable paths, algorithmic subagent
  IO and operation outputs, packet projection, and standalone orchestration.
- Add live LLM inspection tests for decomposition and synthesis, marked
  `live_llm`, one case at a time.
- Add the dedicated real-LLM review fixture:
  `tests/fixtures/complex_task_resolver_review_cases.json`.
- Add fixture coverage validation that compares all fixture `required_stages`,
  statuses, and case categories against the comprehensive review artifact.
- Add a comprehensive real LLM review artifact before any integration work is
  considered.
- Update docs for the new module and its explicit non-integration status.

## Deferred

- Do not replace `stage_1_goal_resolver` or the existing cognition resolver
  recurrence in this plan.
- Do not remove `ResolverGoalProgressV1`, `rag_evidence`, `web_evidence`,
  HIL, approval, or self-goal resolver capabilities.
- Do not move L1/L2/L2d cognition into the new module.
- Do not route the complex-task resolver from L2d, `cognition_resolver`,
  `persona_supervisor2`, action selection, dialog, adapters, schedulers, or
  any live workflow in this plan.
- Do not add `complex_task_resolution` to any runtime enum, prompt, allowed
  capability list, or resolver capability dispatcher in this plan.
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

Overall strategy: standalone/shadow module first, with user-gated future
integration.

| Area | Policy | Instruction |
|---|---|---|
| Existing cognition resolver | compatible | Preserve current recurrence, state, pending, trace, and terminal behavior. |
| New complex-task module | additive | Add a separate package and public entrypoint. |
| Resolver capability enum | deferred | Do not add `complex_task_resolution` until a later user-approved integration plan. |
| Resolver observation | deferred | Do not add `complex_task_result` to `ResolverObservationV1` in Phase 1. Keep projection helpers local to the specialist. |
| L2d prompt | deferred | Do not expose the specialist to L2d until after comprehensive real LLM review and explicit user instruction. |
| RAG | compatible | Use existing RAG helper only through approved public/persona entrypoints. |
| Dialog | compatible | Receive content through existing selected `speak` and L3 content-plan path. |
| Cache | process-local v1 | Add no persistent storage or migration in this plan. |

## Target State

The completed Phase 1 behavior is:

```text
Standalone harness receives a complex multi-part question
  -> complex_task_resolver decomposes and resolves a bounded graph
  -> result packet states answered, partial, cannot_answer, or needs_user_input
  -> review artifact records packet, graph path, cache/collapse decisions,
     and human judgment
  -> no live workflow consumes the packet in this plan
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
| Integration path | Phase 1 has no runtime integration. Future integration is expected to use a `complex_task_resolution` resolver capability, not an `ActionSpecV1` action. | The module must prove robustness before L2d can route to it. |
| Module boundary | Create `kazusa_ai_chatbot.complex_task_resolver`. | Keeps it distinguishable from `cognition_resolver` and allows independent contracts/tests. |
| Existing type reuse | Reuse `EvidenceRefV1` where semantics match; document future compatibility with `ResolverCapabilityRequestV1` and `ResolverObservationV1` without changing them in Phase 1. | The standalone module should not distort current cognition resolver contracts before it is proven. |
| New graph type | Use a node map with stable ids, not nested dynamic dictionaries. | Collapse, cache, provenance, and validation need stable node identity. |
| Final result | Return `ComplexTaskResolutionPacketV1`, not final dialog. | Dialog ownership remains in L3/dialog. |
| Cache scope | Phase 1 uses process-local LRU node cache only. | Avoids DB migration and stale persistent answers while proving the contract. |
| Collapse | Run collapse only over deterministic candidate sets from existing graph/cache nodes. | Avoids broad semantic comparison across arbitrary history. |
| Tool use | Phase 1 may use existing RAG evidence through approved public/persona RAG entrypoints in tests or review harnesses only. | RAG remains evidence owner; the new resolver composes answers from evidence without becoming a live routing owner. |
| Review runtime dependencies | Stage LLM invokers, subagent registry, clock, and limit settings are injected by tests or review harnesses through a non-prompt-facing runtime object. | The standalone module needs real LLM/web/algorithmic review without adding new production route wiring or exposing operational handles to prompts. |
| Evidence subagent | Active `evidence_need` nodes use a review-only subagent boundary that reports resolved, partial, unavailable, or failed evidence. | The fixture exercises web retrieval, but missing SearXNG/network/RAG dependencies must be explicit blockers rather than hidden prompt behavior. |
| Subagent IO | All resolver subagents use one standard envelope: request/task plus context in, `resolved`, `status`, `result`, `attempts`, `cache`, and `trace` out. | This mirrors the existing WebAgent3 helper-agent principle while adapting it to typed task-graph nodes and deterministic review artifacts. |
| Algorithmic calculations | Add a deterministic algorithmic subagent for arithmetic-like work. Do not rely on LLM reasoning for calculations. | The review fixture includes schedules, budgets, weighted scores, token/call estimates, cache savings, and benchmark normalization; local LLM arithmetic is not reliable enough to own those outputs. |
| Expression evaluation | Reject arbitrary expression evaluators, Python `eval`, shell execution, and broad symbolic-math engines in Phase 1. | Typed operations are easier to validate, test, cache, explain, and keep safe. |
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

### Future Capability Request

A later integration plan may add:

```python
COMPLEX_TASK_RESOLUTION_CAPABILITY = "complex_task_resolution"
```

That integration must not be implemented in this plan. The future wrapper is
documented only to keep the standalone packet shape compatible with a likely
L2d-selected resolver capability.

### Public Entrypoint

Create:

```python
async def resolve_complex_task(
    request: ComplexTaskResolverRequestV1,
    context: ComplexTaskResolverContextV1,
    runtime: ComplexTaskResolverRuntimeV1,
) -> ComplexTaskResolutionPacketV1:
    ...
```

Only tests and standalone review harnesses should call this entrypoint in
Phase 1. Other modules must not import internal planner, cache, graph, prompt,
or collapse helpers.

### Standalone Runtime

Create a non-prompt-facing runtime dependency object under
`complex_task_resolver`:

```python
ComplexTaskResolverRuntimeV1 = {
    "schema_version": "complex_task_resolver_runtime.v1",
    "planner_llm": LLMInvoker,
    "node_resolver_llm": LLMInvoker,
    "collapse_llm": LLMInvoker,
    "synthesizer_llm": LLMInvoker,
    "subagents": dict[str, ComplexTaskSubagentV1],
    "limits": dict[str, int | float],
    "clock": Callable[[], datetime],
}
```

The runtime object is constructed only by deterministic tests, local review
harnesses, or live LLM inspection tests. It is not serialized, not projected
into prompts, and not a production config surface. Phase 1 must not add new
live workflow route wiring merely to construct this object.

### Standalone Context

Create:

```python
ComplexTaskResolverContextV1 = {
    "schema_version": "complex_task_resolver_context.v1",
    "conversation_summary": str,
    "persona_context_summary": str,
    "available_evidence": list[EvidenceRefV1],
    "resolver_context_summary": str,
    "time_context": str,
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
    "schema_version": "complex_task_subagent_context.v1",
    "root_question": str,
    "parent_chain_summary": str,
    "sibling_summaries": list[str],
    "available_evidence": list[EvidenceRefV1],
    "time_context": str,
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
    "trace": list[str],
    "unresolved_items": list[str],
}
```

Subagent rules:

- The graph/service layer selects subagents from `runtime["subagents"]`.
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
execution for `algorithmic_task` graph nodes and must not call an LLM.

```python
AlgorithmicOperationV1 = {
    "schema_version": "complex_task_algorithmic_operation.v1",
    "operation": (
        "duration_sum | schedule_fit | budget_allocation | "
        "weighted_score | token_budget | cache_savings | "
        "percentage_change | range_compare | benchmark_normalization"
    ),
    "operands": dict,
    "units": dict[str, str],
    "rounding": {
        "mode": "none | half_up | floor | ceil",
        "decimal_places": int,
    },
    "required_outputs": list[str],
}

AlgorithmicResultV1 = {
    "schema_version": "complex_task_algorithmic_result.v1",
    "operation": str,
    "resolved_values": dict,
    "formulas": list[str],
    "assumptions_used": list[str],
    "warnings": list[str],
    "display_summary": str,
}
```

Supported Phase 1 behavior:

- `duration_sum`: add task and break durations, return total duration and
  component breakdown.
- `schedule_fit`: combine duration math with a start/end window and return
  fit status, finish time, slack/overrun, and a deterministic schedule when
  ordered tasks are provided.
- `budget_allocation`: add fixed and optional costs, compare against a budget,
  and return remaining/overrun amount.
- `weighted_score`: validate weights, map supplied numeric scores, compute
  weighted totals, rank candidates, and report ties. It must not invent a
  qualitative-to-numeric rubric unless the rubric is supplied by a previous
  semantic node.
- `token_budget`: compute call counts, context sizes, completion budgets, and
  candidate reduction savings.
- `cache_savings`: compute node count, hit count/rate, uncached calls, and saved
  calls.
- `percentage_change` and `range_compare`: compute ratios/ranges from already
  sourced numbers and preserve uncertainty ranges.
- `benchmark_normalization`: normalize already-sourced benchmark numbers by
  unit, quantization label, backend label, and range. It must not fabricate
  missing benchmark values.

Numeric safety:

- Use `decimal.Decimal` for money, percentages, weighted scores, and benchmark
  normalization where decimal precision matters.
- Use `datetime`/`timedelta` for schedules and durations.
- Reject missing units, incompatible units, non-finite numbers, negative
  durations where not meaningful, and unsupplied qualitative score mappings.
- Do not parse or execute arbitrary expressions. Do not use Python `eval`,
  subprocesses, notebooks, shell tools, or external math services.

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
    "cache_key": str,
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
    "status": "answered | partial | cannot_answer | needs_user_input | failed",
    "root_question": str,
    "final_answer_brief": str,
    "source_backed_facts": list[str],
    "assumptions_or_inferences": list[str],
    "unresolved_items": list[str],
    "user_input_needed": list[str],
    "dialog_handoff_requirements": list[str],
    "graph": ComplexTaskGraphV1,
    "trace_summary": list[str],
}
```

`final_answer_brief` is not final dialog. In Phase 1 it is content for review
and future dialog handoff only.

### ComplexTaskCacheEntryV1

```python
{
    "schema_version": "complex_task_cache_entry.v1",
    "cache_key": str,
    "objective_summary": str,
    "scope_summary": str,
    "answer_text": str,
    "source_backed_facts": list[str],
    "assumptions_or_inferences": list[str],
    "evidence_refs": list[EvidenceRefV1],
    "freshness_label": "stable | session | volatile",
    "dependency_labels": list[str],
    "created_at_utc": str,
    "last_used_at_utc": str,
}
```

Phase 1 cache rules:

- process-local only;
- keyed from normalized objective summary, current-user scope label, channel
  scope label, evidence dependency labels, and freshness label;
- no persistent rows;
- no cross-user private memory reuse;
- no reuse for volatile live/current facts unless the same resolver run created
  the entry;
- cache hits create a normal node result with trace text indicating cache use,
  but raw cache keys are not projected to cognition.

### Future ResolverObservationV1 Extension

Do not add this field in Phase 1. A later user-approved integration plan may
add optional:

```python
"complex_task_result": ComplexTaskResolutionPacketV1
```

Future projection into `resolver_context` should include only:

- status;
- root question;
- final answer brief;
- unresolved item count and short summaries;
- source-backed fact summaries;
- assumptions or inferences;
- dialog handoff requirements.

In Phase 1 the full graph and compact projection remain inside the standalone
packet and review artifacts only. Nothing is written to
`resolver_state["observations"]`, and nothing is projected into live cognition
prompts.

## LLM Call And Context Budget

Default context cap: 50k tokens.

Standalone Phase 1 maximum:

| Stage | Calls | Context inputs | Cap policy |
|---|---:|---|---|
| Graph planner | 1 | original objective, compact review context, optional relevant RAG summary | input under 18k chars; output max nodes 8 |
| Active node resolver | 0-3 | one active node, parent chain, sibling summaries, existing evidence packet | at most 3 active node attempts per capability call |
| Algorithmic subagent | 0 LLM | typed operation payload and bounded semantic context | deterministic execution only; invalid/missing operands return invalid or partial |
| Evidence subagent | 0-2 | active evidence node query plus compact semantic context only | at most 2 WebAgent3/RAG evidence calls per resolver run; missing dependency returns partial/unavailable |
| Collapse reviewer | 0-2 | active node and deterministic candidate nodes/cache entries only | at most 2 collapse checks; no broad search |
| Bottom-up synthesizer | 1 | resolved leaf summaries, unresolved items, facts, assumptions | input under 16k chars |

Hard Phase 1 caps:

- `max_nodes=8`
- `max_depth=3`
- `max_internal_iterations=4`
- `max_rag_calls=2`
- `max_cache_hits=4`
- one JSON repair pass per LLM stage only when parsing fails
- no retry loop for semantically unsatisfactory output

If the specialist cannot complete inside these caps, return
`status="partial"` or `status="cannot_answer"` with explicit blockers rather
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
  - Defines graph, node, packet, cache entry, validators, and projection
    helpers.
- `src/kazusa_ai_chatbot/complex_task_resolver/graph.py`
  - Owns deterministic graph mutation, traversal, active-node selection, and
    bottom-up collection helpers.
- `src/kazusa_ai_chatbot/complex_task_resolver/cache.py`
  - Owns process-local node cache and cache eligibility.
- `src/kazusa_ai_chatbot/complex_task_resolver/evidence.py`
  - Owns the review-only evidence subagent, dependency preflight result shape,
    and unavailable/partial evidence handling.
- `src/kazusa_ai_chatbot/complex_task_resolver/algorithmic.py`
  - Owns deterministic algorithmic subagent operations for duration, schedule,
    budget, weighted score, token/call, cache-savings, percentage/range, and
    benchmark-normalization calculations.
- `src/kazusa_ai_chatbot/complex_task_resolver/planner.py`
  - Owns graph-planning prompt and parser.
- `src/kazusa_ai_chatbot/complex_task_resolver/node_resolver.py`
  - Owns active-node resolution and delegates evidence or algorithmic needs
    only through injected resolver subagents.
- `src/kazusa_ai_chatbot/complex_task_resolver/collapse.py`
  - Owns bounded semantic collapse prompt and deterministic candidate
    selection.
- `src/kazusa_ai_chatbot/complex_task_resolver/synthesizer.py`
  - Owns bottom-up packet synthesis.
- `src/kazusa_ai_chatbot/complex_task_resolver/service.py`
  - Owns the public orchestration entrypoint.
- `tests/helpers/complex_task_resolver_review.py` or equivalent local harness
  helper
  - Builds review inputs and writes human-readable artifacts without wiring
    runtime cognition.
- `tests/test_complex_task_resolver_contracts.py`
- `tests/test_complex_task_resolver_graph.py`
- `tests/test_complex_task_resolver_cache.py`
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

- No production module outside `src/kazusa_ai_chatbot/complex_task_resolver/`
  should be modified in Phase 1.
- No `cognition_resolver`, `cognition_chain_core`, `persona_supervisor2`,
  dialog, adapter, scheduler, or live workflow file should import or call the
  new module in this plan.
- `development_plans/README.md`
  - Register this draft plan while it remains under discussion.
- Documentation may reference the future integration path only as deferred,
  user-gated work.

### Future Integration Surface - Deferred

These changes are intentionally outside Phase 1:

- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
  - Future: add the new capability enum and optional observation field.
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
  - Future: route `complex_task_resolution` to the public entrypoint.
- `src/kazusa_ai_chatbot/cognition_resolver/state.py`
  - Future: project compact complex-task observation summaries into
    `resolver_context`.
- `src/kazusa_ai_chatbot/cognition_chain_core`
  - Future: expose when L2d may request the new resolver capability.
- Resolver and L2d tests
  - Future: cover request acceptance, observation projection, and
    next-cycle cognition behavior.

### Keep

- Keep `stage_1_goal_resolver` as the live persona resolver entrypoint.
- Keep existing RAG2, dialog, consolidation, adapter, scheduler, and
  background-work behavior unchanged.
- Keep `ResolverGoalProgressV1` for cognition-maintained goal progress. The
  new task graph does not replace it in Phase 1.

## Overdesign Guardrail

- Actual problem: complicated user questions need a bounded specialist that can
  decompose subtasks, collect results, reuse/cache node answers, collapse
  duplicate paths, and return a complete answerability packet before dialog.
- Minimal change: add one separate specialist module with typed graph
  contracts and a standalone review harness; keep the existing cognition
  resolver untouched.
- Ownership boundaries: `complex_task_resolver` owns graph resolution in
  standalone review; RAG evidence, if used, remains evidence only;
  deterministic code owns validation/cache/caps; future L2d/dialog ownership
  is documented but not wired.
- Rejected complexity: persistent cache, direct coding-agent execution,
  arbitrary tools, action-spec emission, scheduler work, adapter delivery,
  replacing `stage_1_goal_resolver`, broad prompt rewrites, compatibility
  shims, unbounded recursion, and multi-minute agent loops.
- Evidence threshold: add deferred complexity only after deterministic and
  live LLM evidence shows Phase 1 cannot answer target complex questions within
  caps while preserving source-backed facts and dialog handoff quality.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper mechanics only when they
  preserve the public contracts in this plan.
- The responsible agent must not introduce alternate architectures, fallback
  call paths, persistent storage, compatibility shims, or extra features.
- Changes outside `complex_task_resolver`, tests, review artifacts, and plan
  documentation require strong justification and must not create runtime
  integration in Phase 1.
- The responsible agent must search for existing validators, JSON parsers,
  evidence refs, prompt projection helpers, cache helpers, and RAG entrypoints
  before adding new equivalents.
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
     - `tests/test_complex_task_resolver_cache.py`
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
   - Verify contract, graph, cache, evidence, and algorithmic tests pass before
     prompt integration.

4. Parent adds standalone service and review-harness tests.
   - Files:
     - `tests/test_complex_task_resolver_service.py`
     - review harness helper tests if the harness is not covered by service
       tests.
   - Expected before implementation: fail because the public orchestration
     entrypoint and review artifact writer do not exist.

5. Production-code subagent implements planner, active-node resolver, collapse,
   cache, subagent dispatch, and synthesizer prompts.
   - Use static system prompts, current-run human payloads, and existing JSON
     parsing helpers.
   - Keep prompt projection compact and role-neutral.

6. Parent runs focused deterministic tests, service tests, review-harness
   tests, review dependency preflight tests, prompt-render checks, and static
   non-integration greps.

7. Parent adds live LLM inspection cases.
   - File: `tests/test_complex_task_resolver_live_llm.py`
   - Case source:
     `tests/fixtures/complex_task_resolver_review_cases.json`
   - Cases must run one at a time and produce review artifacts.
   - Runtime prompts must receive only the user question and normal review
     context, never expected graph traces, expected statuses, expected final
     answers, or forbidden failure modes from the fixture.

8. Parent performs the comprehensive real LLM review only after the user
   explicitly instructs the review.
   - Required artifact:
     `test_artifacts/complex_task_resolver/comprehensive_real_llm_review.md`
   - The artifact must include each raw input, expected challenge, graph,
     traversal path, cache decisions, collapse decisions, evidence dependency
     status, node outputs, final packet, answerability judgment, quality
     judgment, failures, and whether the case is acceptable for future L2d
     integration.
   - The artifact must include every fixture case or an explicit dependency
     blocker for that case, plus a coverage table for fixture categories,
     expected statuses, and `required_stages`.

9. Parent updates docs.
    - New module README.
    - Development-plan README if lifecycle status changes are needed.
    - Development-plan execution evidence.

10. Parent runs the full verification gate.

11. Parent starts one independent code-review subagent after verification
    passes.

12. Parent remediates review findings inside approved scope and reruns affected
    tests.

13. Parent stops before any integration work.
    - Record whether the comprehensive real LLM review is sufficient.
    - Ask for explicit user instruction before any future L2d/workflow
      integration plan or code change.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after focused
  tests exist; owns production code changes only; does not edit tests unless
  the parent explicitly directs it; closes after planned production code changes
  are complete, excluding review fixes.
- Parent agent may continue service tests, review-harness tests, static
  non-integration checks, regression tests, and validation work while the
  production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - contract tests established
  - Covers: implementation step 1.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_cache.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py -q`
  - Evidence: record expected missing-symbol or failure output.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 2 - module contracts, graph helpers, and subagents implemented
  - Covers: implementation steps 2-3.
  - Verify: same Stage 1 pytest command passes.
  - Evidence: record changed files and test output.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 3 - standalone service and review harness implemented
  - Covers: implementation steps 4-5.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_service.py -q`
  - Evidence: record service orchestration, packet projection, and artifact
    writer test output.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 4 - LLM planner/resolver/collapse/synthesizer implemented
  - Covers: implementation steps 5-6.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_graph.py tests\test_complex_task_resolver_cache.py tests\test_complex_task_resolver_evidence.py tests\test_complex_task_resolver_algorithmic.py tests\test_complex_task_resolver_service.py -q`
  - Evidence: record LLM call budget, prompt-render output, and static
    non-integration grep output.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 5 - live LLM inspection cases completed
  - Covers: implementation step 7.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py -q -k "<single_case_name>" -s`
    for each approved case, one case at a time.
  - Evidence: record fixture case id, reviewed artifact paths, per-case
    judgment, and confirmation that fixture expected traces/final answers were
    not injected into prompts.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 6 - comprehensive real LLM review completed under user instruction
  - Covers: implementation step 8.
  - Verify:
    `test_artifacts/complex_task_resolver/comprehensive_real_llm_review.md`
    exists and includes raw inputs, graph traces, cache/collapse decisions,
    final packets, failures, and future-integration judgment.
  - Evidence: record the user's review instruction, artifact path, per-case
    outcome, and whether the user accepted the evidence as sufficient to
    consider a future integration plan.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 7 - docs and regression verification complete
  - Covers: implementation steps 9-10.
  - Verify all commands in `Verification`.
  - Evidence: record command outputs and any accepted warnings.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 8 - independent code review complete
  - Covers: implementation steps 11-12.
  - Verify review findings are resolved or explicitly accepted, and affected
    tests are rerun.
  - Evidence: record reviewer, findings, fixes, rerun commands, residual risks,
    and approval status.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 9 - integration remains blocked pending explicit user instruction
  - Covers: implementation step 13.
  - Verify static non-integration greps still show no runtime wiring.
  - Evidence: record that no L2d/cognition/dialog workflow imports or routes
    to `complex_task_resolver`.
  - Sign-off: `<agent/date>` after evidence is recorded.

## Verification

Use `venv\Scripts\python` for Python commands.

### Static Greps

- `rg "complex_task_resolution" src\kazusa_ai_chatbot\cognition_resolver src\kazusa_ai_chatbot\cognition_chain_core src\kazusa_ai_chatbot\nodes`
  - Expected in Phase 1: no matches.
- `rg "complex_task_result" src\kazusa_ai_chatbot\cognition_resolver src\kazusa_ai_chatbot\cognition_chain_core src\kazusa_ai_chatbot\nodes`
  - Expected in Phase 1: no matches.
- `rg "complex_task_resolver" src\kazusa_ai_chatbot\cognition_resolver src\kazusa_ai_chatbot\cognition_chain_core src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\dialog`
  - Expected in Phase 1: no matches.
- `rg "ctr_0|expected_graph_trace|minimum_viable_answer|expected_final_answer|performance_reference_summary|forbidden_failure_modes" src\kazusa_ai_chatbot`
  - Expected in Phase 1: no matches. Fixture case ids, expected traces,
    minimum viable answers, expected final answers, performance references,
    and failure-mode hints must not appear in production code or runtime
    prompts.
- `rg "task_brief|worker|handler_id|platform_user_id|platform_channel_id" src\kazusa_ai_chatbot\complex_task_resolver`
  - Expected: no prompt-facing payload leaks. Internal trusted scope helpers, if
    any, must be documented and not projected to LLM prompts.
- `rg "from kazusa_ai_chatbot\.(dispatcher|calendar_scheduler|cognition_resolver|cognition_chain_core|nodes|dialog)|from adapters|import subprocess|create_subprocess|MCP_SERVERS|mcp" src\kazusa_ai_chatbot\complex_task_resolver`
  - Expected in Phase 1: no matches. The standalone resolver must not call live
    workflow modules, adapters, scheduler, dialog, shell execution, or arbitrary
    MCP tooling.
- `rg "eval\(|exec\(|subprocess|create_subprocess|LLInterface|ainvoke|SystemMessage|HumanMessage" src\kazusa_ai_chatbot\complex_task_resolver\algorithmic.py`
  - Expected in Phase 1: no matches. The algorithmic subagent must be
    deterministic Python calculation only, with no arbitrary expression
    execution and no LLM calls.

### Fixture Validation

```powershell
venv\Scripts\python -m json.tool tests\fixtures\complex_task_resolver_review_cases.json > $null
```

The fixture must contain exactly 31 cases. Review harnesses may read
`user_question`, `case_id`, and review metadata for artifact labeling, but must
not include expected traces, expected statuses, minimum viable answers,
expected final answers, performance-reference summaries, or forbidden failure
modes in model prompts.

The fixture validator must also emit a coverage summary containing:

- case count by `category`;
- case count by `expected_status`;
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
venv\Scripts\python -m pytest tests\test_complex_task_resolver_cache.py -q
venv\Scripts\python -m pytest tests\test_complex_task_resolver_evidence.py -q
venv\Scripts\python -m pytest tests\test_complex_task_resolver_algorithmic.py -q
venv\Scripts\python -m pytest tests\test_complex_task_resolver_service.py -q
```

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

### Comprehensive Real LLM Review

This gate runs only after the user explicitly instructs it. It is required
before any later L2d/workflow integration plan.

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
- cache-hit decisions and freshness reasoning;
- collapse candidates, collapse decision, and reason;
- final `ComplexTaskResolutionPacketV1`;
- human quality judgment;
- robustness issue list;
- explicit judgment on whether the behavior is safe to expose to future L2d
  routing.

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
- Whether no L2d, cognition resolver, dialog, adapter, scheduler, or live
  workflow integration has been added.
- Whether existing typed dictionaries are reused only where semantics match.
- Whether prompt/RAG/context projections avoid raw ids, backend terms, worker
  internals, and final-dialog leakage.
- Whether fixture case ids, keywords, expected graph traces, expected statuses,
  minimum viable answers, expected final answers, performance-reference
  summaries, or forbidden failure modes are absent from deterministic routing
  code and runtime prompts.
- Whether cache reuse is scope-safe and does not persist stale answers.
- Whether graph collapse is bounded and traceable.
- Whether live review dependency preflight, evidence-subagent unavailable
  paths, and fixture category/status/stage coverage are present in artifacts.
- Whether algorithmic subagent operations are deterministic, typed, tested,
  free of LLM calls, and free of arbitrary expression execution.
- Whether comprehensive real LLM evidence is sufficient for local/weaker model
  risk and for considering a later integration plan.

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
  `ComplexTaskResolverContextV1` plus a non-prompt-facing
  `ComplexTaskResolverRuntimeV1`.
- The specialist returns a validated `ComplexTaskResolutionPacketV1` through
  tests and review harnesses only.
- Review-only evidence subagent behavior is bounded, tested for unavailable
  dependencies, and never calls live workflow modules, adapters, scheduler,
  dialog, shell tools, or arbitrary MCP tools.
- The deterministic algorithmic subagent follows the standard subagent IO
  contract, covers the fixture arithmetic families, uses typed operands and
  explicit formulas, and contains no LLM calls, `eval`, shell execution,
  notebook execution, or arbitrary expression engine.
- The existing cognition resolver remains the only recurrence controller.
- No L2d, cognition resolver, persona graph, dialog, adapter, scheduler, or
  live workflow code routes to the new module.
- The new task graph uses stable node ids, bounded depth, bounded node count,
  node statuses, collapse events, and bottom-up synthesis.
- Process-local node cache reuse is implemented with scope and freshness
  guards.
- Existing RAG, dialog, background-work, consolidation, adapter, and scheduler
  behavior remain unchanged.
- Focused deterministic tests, standalone service tests, non-integration
  greps, review dependency preflight, resolver regressions, and approved live
  LLM inspection cases pass or have recorded accepted blockers.
- The review fixture
  `tests/fixtures/complex_task_resolver_review_cases.json` contains 31
  concrete cases. Each case has a `minimum_viable_answer` for AI-judge
  acceptance and a Codex-authored `expected_final_answer`; both are used only
  as review metadata, not as prompt or deterministic-routing hints.
- Comprehensive real LLM review has been performed after explicit user
  instruction, and the artifact records every fixture case or explicit
  dependency blocker, category/status/required-stage coverage, dependency
  status, and whether the module is robust enough to consider a future L2d
  integration plan.
- Integration remains blocked until the user gives a separate explicit
  instruction after reviewing or accepting the real LLM evidence.
- Independent code review has no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Specialist becomes a second cognition resolver | Separate package, docs, public entrypoint, and no ownership of `ResolverCycleStateV1` recurrence | Code review and README boundary checks |
| Prompt bloat from graph projection | Keep full graph in validated state and project compact packet summaries | Projection tests and live LLM artifacts |
| Cache returns stale or private answer | Process-local cache only; scope/freshness guards; no cross-user private reuse | Cache tests |
| Collapse merges distinct subtasks | Deterministic candidate set and traceable LLM collapse reason | Collapse tests and live duplicate-path case |
| Response latency grows too high | Hard LLM/tool/node caps; return partial/cannot-answer on cap | Service timeout tests and live review artifacts |
| Dialog treats packet as final voice | No dialog integration in Phase 1; packet says content only | Static non-integration greps and code review |
| Premature L2d routing occurs before robustness is proven | Runtime integration is deferred and blocked by static greps, plan gates, and explicit user approval requirement | Stage 9 non-integration sign-off |
| Reuse distorts existing contracts | Reuse `EvidenceRefV1` only where semantics match; document future resolver wrapper without changing current contracts | Contract tests and code review |
| Live review hides missing external dependencies | Run dependency preflight, record web/RAG/LLM/Mongo/embedding/network status, and mark affected cases dependency-blocked or partial | Preflight artifact, comprehensive review coverage table, and evidence-subagent tests |
| Evidence subagent becomes accidental runtime integration | Inject subagents only through tests/review harnesses and forbid L2d/dialog/adapter/scheduler calls | Static non-integration greps, evidence tests, and independent code review |
| LLM arithmetic produces plausible wrong numbers | Route arithmetic-like nodes through the deterministic algorithmic subagent and require formulas plus operation tests | Algorithmic tests, no-LLM/no-eval grep, and live review artifact checks |

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
  cache reuse, and unresolvable tasks.
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
- Draft decision: Phase 1 cache is process-local only; persistent cache is
  deferred to avoid database migration and stale answer risk.
- This draft is not approved for implementation.
