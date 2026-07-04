# rag3 local context resolver bigbang plan

## Summary

- Goal: Replace the RAG2 initializer/slot supervisor with RAG 3, a
  local-context resolver aligned with the `complex_task_resolver` architecture
  while preserving cognition's prompt-facing `rag_result` evidence contract.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, and `test-style-and-execution`.
- Overall cutover strategy: two-step execution. Step 1 implements and
  evaluates standalone RAG 3 through the final public IO without tapping it
  into production. Step 2 performs a bigbang production cutover that replaces
  RAG2 after standalone evidence and user approval are recorded. Do not
  preserve `unknown_slots`, prefix dispatch, prose slot dependencies, refined
  natural-language initializer re-entry, or compatibility compilation back to
  RAG2 slots. Preserve the downstream `rag_result` projection surface.
- Highest-risk areas: replacing a live response-path evidence system,
  preserving local-model latency, avoiding a universal subagent abstraction,
  carrying graph dependencies without raw ids in prompts, preventing local
  memory/private evidence leaks, preserving current-user and active-character
  speaker scope, and keeping cognition/dialog ownership unchanged.
- Acceptance criteria: Step 1 produces a standalone RAG 3 implementation,
  stable public IO, and human-readable real-LLM review artifacts for quality,
  performance, and efficiency. Step 2 then makes RAG 3 the only production
  local-context recall path; the RAG2 initializer/supervisor files are removed
  or retired from production imports; cognition still receives sanitized
  `rag_result`; RAG2 real-LLM initializer cases are represented as RAG 3
  graph/packet tests; deterministic and selected live-LLM checks pass; and
  independent code review approves the implementation before lifecycle
  completion.

## Context

The current RAG2 path is slot driven:

```text
local_context_recall
  -> quote-aware RAG wrapper
  -> rag_initializer
  -> unknown_slots: list[str]
  -> prefix dispatcher
  -> helper capability agent
  -> evaluator
  -> optional refined-query initializer re-entry
  -> finalizer
  -> project_known_facts
  -> rag_result
```

The active investigation around `#napcat` exposed a representative failure:
the evidence existed, but the search task over-focused on the addressed active
character name instead of the command anchor. Renaming memory improved the
knowledge row but did not address the architectural issue. The current
initializer carries retrieval intent, capability route, dependencies, speaker
scope, anchors, and retry strategy in one prose-oriented LLM stage. Later
stages recover from malformed focus by spending loop budget.

The user rejected an incremental RAG3 router/interpreter POC and requested a
bigbang design aligned with `complex_task_resolver`. The intended conclusion
is that local context recall and public answer research solve the same
evidence-resolution problem over different source domains:

```text
ambiguous goal
  -> bounded graph
  -> one active node at a time
  -> source-owned evidence collection
  -> known/lacking/boundary packet
  -> cognition judges what it means
```

`complex_task_resolver` owns public answer research. RAG 3 will own local
context recall. The two systems must align on graph lifecycle, semantic
knowledge packet shape, traversal caps, collapse/dedup review, and prompt-safe
projection discipline. They must not become one universal runtime abstraction
or one interchangeable subagent family.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  lifecycle-updating, or signing off this plan.
- `local-llm-architecture`: load before changing RAG, resolver, prompt,
  graph, LLM-call, context-budget, evidence, cognition, or dialog behavior.
- `debug-llm`: load before creating live LLM review artifacts, inspecting
  traces, comparing RAG2/RAG3 behavior, or judging prompt quality.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files or tests containing CJK string
  literals.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- This plan is `in_progress` for Step 1 only. Do not execute Step 2
  production cutover until explicit user approval is recorded after Step 1
  evidence.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` uses parent-led native subagent execution. If
  native subagents are unavailable, stop unless the user explicitly approves
  fallback execution.
- Preserve the character-brain ownership boundary: RAG 3 returns evidence;
  cognition decides stance and action; dialog renders selected visible text;
  consolidation owns durable writes after the completed episode.
- Do not make RAG mandatory before cognition. RAG 3 remains the implementation
  of the L2d-selected `local_context_recall` capability plus the existing
  bounded shared-memory prewarm lane when retained by this plan.
- The RAG 3 public IO is identical in standalone evaluation and production:
  `resolve_local_context(request, context, options=None)` accepts
  `LocalContextResolverRequestV1`, `LocalContextResolverContextV1`, and
  `LocalContextResolverOptionsV1`, and returns
  `LocalContextResolutionPacketV1`.
- Step 1 must not modify `local_context_recall`, shared-memory prewarm,
  adapter delivery, cognition production wiring, dialog production wiring, or
  any production caller to use RAG 3. Step 1 may add an isolated importable
  package, tests, standalone evaluation harnesses, raw evidence artifacts, and
  human-readable review artifacts.
- Step 1 standalone evaluation must call the same public entrypoint and read
  the same `LocalContextResolutionPacketV1` that Step 2 will use in
  production. Do not create experiment-only request, context, option, or packet
  shapes.
- Step 2 production cutover must not start until Step 1 deterministic tests,
  standalone real-LLM review artifacts, quality/performance/efficiency
  summary, and explicit user approval are recorded in `Execution Evidence`.
- Do not expose raw adapter syntax, database ids, storage rows, embeddings,
  cache keys, raw UTC timestamps, prompt text, or raw graph ids in
  cognition-visible evidence.
- Do not ask LLM stages to generate MongoDB filters, index names, embedding
  dimensions, cache keys, raw worker internals, adapter delivery decisions,
  persistence decisions, or final dialog wording.
- Deterministic code owns graph identity, traversal caps, validation,
  dependency binding, request/result schema checks, prompt-safety projection,
  cache mechanics, persistence boundaries, and service wiring.
- LLM stages own semantic decomposition, semantic evidence judgment, semantic
  blocker explanation, and semantic synthesis into known/lacking/boundary rows.
- Do not add compatibility shims, dual production paths, fallback to RAG2
  `unknown_slots`, alias modules, or translation bridges to preserve old
  supervisor shapes.
- Do not create a universal base class or runtime registry shared between RAG
  3 and `complex_task_resolver`. Align architecture and vocabulary while
  keeping family-local contracts.
- Live LLM tests must run one case at a time with output inspected. Do not run
  live LLM cases in batches.

## Must Do

- Delete or retire the active RAG3 router/interpreter POC plan and make this
  the only active RAG3 plan.
- Implement standalone RAG 3 first behind
  `resolve_local_context(request, context, options=None)` without tapping it
  into live `local_context_recall`, prewarm, cognition, dialog, or adapter
  paths.
- Use the same request, context, options, public entrypoint, and
  `LocalContextResolutionPacketV1` output for standalone evaluation and later
  production cutover.
- Create a new local-context resolver package with graph lifecycle aligned to
  `complex_task_resolver`.
- Run standalone real-LLM evaluation over the accepted RAG2 behavior matrix.
  Evaluate quality, performance, and efficiency from raw evidence and write
  human-readable review artifacts before production cutover.
- After Step 1 evidence and user approval are recorded, replace the production
  RAG2 supervisor entrypoint used by `local_context_recall` with the RAG 3
  local-context resolver in one bigbang cutover.
- Remove production use of `rag_initializer`, `unknown_slots`, prefix
  dispatcher routing, RAG2 evaluator/finalizer loop control, and refined-query
  initializer re-entry.
- Preserve the existing prompt-facing `rag_result` evidence surface consumed by
  cognition and consolidation unless a field is explicitly replaced in this
  plan.
- Preserve the existing local evidence source packages where they remain the
  source owners: conversation evidence, memory evidence, person context, live
  context, recall, and web evidence.
- Convert the RAG2 initializer real-LLM behavior requirements into RAG 3
  graph/packet tests rather than keeping initializer tests as canonical.
- Add deterministic contract, graph, traversal, projection, subagent, and
  integration tests before production implementation.
- Add selected live LLM review cases for the graph planner and active-node
  resolver. Run them one at a time and write human-readable debug artifacts.
- Update README, HOWTO, RAG/local-context resolver docs, subagent interface
  docs, and development plan registry to reflect the new active architecture.

## Deferred

- Do not redesign cognition, dialog, memory lifecycle, consolidation,
  adapters, scheduler, reflection, self-cognition, or accepted-task behavior.
- Do not implement a public web research replacement here; public/external
  answer research remains owned by `complex_task_resolver`.
- Do not merge RAG 3 and `complex_task_resolver` into one runtime package.
- Do not add new MongoDB collections, indexes, or migrations unless execution
  discovers that a cache or trace persistence contract cannot remain
  process-local. Such discovery requires a plan update before implementation.
- Do not add generic MCP, shell, filesystem, notebook, package-install, or
  adapter-send tools to RAG 3.
- Do not raise live response-path iteration or timeout caps to hide retrieval
  weakness.
- Do not use better embeddings or rerankers as the primary fix for wrong
  query focus. Retrieval quality improvements need a separate plan.

## Cutover Policy

Overall strategy: two-step standalone-first execution, followed by a bigbang
production cutover.

| Area | Policy | Instruction |
|---|---|---|
| RAG 3 standalone package | compatible isolated Step 1 | Add standalone RAG 3 package, tests, and evaluation harness without wiring production callers to it. |
| RAG 3 public IO | bigbang-stable contract | Use the exact same `resolve_local_context(request, context, options=None)` input and `LocalContextResolutionPacketV1` output in Step 1 and Step 2. Do not create experiment-only shapes. |
| RAG supervisor architecture | Step 2 bigbang | Replace the slot-driven RAG2 supervisor with RAG 3 local-context resolver in one production cutover after Step 1 evidence and user approval are recorded. |
| RAG2 `unknown_slots` state | bigbang | Remove as production control state. Do not compile RAG 3 graphs back to slots. |
| RAG2 initializer and dispatcher | bigbang | Remove or retire from production imports after callers move to RAG 3. |
| Downstream `rag_result` | compatible retained surface | Preserve cognition/consolidation prompt-facing evidence shape to keep non-RAG stages stable. |
| Local evidence source packages | migration | Reuse or move existing source-owned retrieval code behind RAG 3 subagents without changing their source semantics. |
| Complex task resolver | compatible sibling | Do not route local/private recall through public answer research. Align graph lifecycle only. |
| Cache2 helper behavior | migration | Keep source-level cache mechanics where still owned by retrieval helpers. Remove durable initializer strategy cache use. |
| Tests | two-step replacement | Step 1 adds standalone RAG 3 contract, graph, packet, and live-LLM review tests. Step 2 replaces RAG2 initializer-route and supervisor tests with production RAG 3 integration coverage. |
| Documentation | two-step replacement | Step 1 creates standalone local-context resolver docs. Step 2 updates production architecture docs to present RAG 3 as the production local-context recall path. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative dual-path strategy by default.
- If an area is `bigbang`, delete or rewrite legacy references instead of
  preserving them.
- If an area is `compatible isolated Step 1`, keep RAG 3 callable only through
  standalone tests, standalone evaluation harnesses, and direct developer
  invocation. Do not route live message handling through it.
- If an area is `bigbang-stable contract`, every caller and test must use the
  named public IO; alternate DTOs, slot compilers, or adapter wrappers are
  forbidden.
- If an area is `migration`, follow the exact implementation and cleanup gates
  in this plan.
- If an area is `compatible retained surface`, preserve only the named
  downstream surface; do not preserve old supervisor internals.
- Any change to a cutover policy requires user approval before implementation.

## Target State

Step 1 standalone target:

```text
standalone evaluation harness or focused test
  -> resolve_local_context(request, context, options=None)
  -> graph planner
  -> active node resolver
  -> local context subagent
  -> collapse/dedup review
  -> bottom-up synthesis
  -> LocalContextResolutionPacketV1
  -> human-readable debug review artifact
```

Step 1 must produce the final production packet, including `rag_result`, even
though production callers do not consume it yet.

Step 2 completed production architecture:

```text
local_context_recall
  -> resolve_local_context(request, context, options=None)
  -> graph planner
  -> active node resolver
  -> local context subagent
  -> collapse/dedup review
  -> bottom-up synthesis
  -> LocalContextResolutionPacketV1
  -> project_local_context_packet(...)
  -> rag_result
  -> next cognition resolver cycle
```

RAG 3 is a local-context resolver, not a generic assistant and not a public
research resolver. Its source domains are:

- conversation history and adjacent/reply context;
- durable shared memory and current-user scoped memory units;
- person/profile/relationship context;
- active recall, commitments, plans, and current episode position;
- live context for current time/date/weather/opening/current facts;
- web evidence only when a local-context node needs URL/public content
  retrieval and the capability boundary allows it.

RAG 3 returns a prompt-safe evidence packet with explicit known facts, missing
facts, recommended next evidence directions, and evidence boundary notes.
Cognition remains the only owner of whether the evidence answers the original
goal, whether Kazusa should speak, and how the visible response should behave.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Architecture model | Model RAG 3 after `complex_task_resolver` graph traversal. | The two systems solve the same evidence-resolution problem over different source domains. |
| Execution phasing | Build standalone RAG 3 first, then perform one production bigbang cutover. | This gives real LLM evidence before touching the live response path while preserving the user's bigbang replacement requirement. |
| Runtime package | Create `kazusa_ai_chatbot.local_context_resolver`. | The name states source ownership and avoids pretending RAG 3 is a generic public-research resolver. |
| Public entrypoint | `resolve_local_context(request, context, options=None)`. | Mirrors complex resolver public IO while remaining family-local. |
| Interface consistency | Standalone evaluation and production cutover use the same public request/context/options and packet. | Prevents a POC shape from drifting away from the production contract. |
| Cutover gate | Step 2 starts only after Step 1 evidence and explicit user approval are recorded. | The production replacement is high risk and must be justified by observed quality/performance/efficiency. |
| Downstream surface | Preserve `rag_result`. | Cognition and consolidation already consume this prompt-safe evidence surface. |
| Legacy supervisor state | Remove `unknown_slots` and slot prefix dispatch. | Prose slots are the fragile dependency carrier being replaced. |
| Dependency carrier | Use graph node ids internally and prompt-safe semantic aliases externally. | Deterministic code owns graph identity; LLM prompts see semantic dependencies, not raw ids. |
| Subagent family | Add local-context resolver subagents with typed request/result envelopes. | Aligns with complex resolver while keeping RAG/private source contracts separate. |
| Shared abstraction | Do not add a universal complex/RAG subagent base or registry bridge. | The subagent guide explicitly keeps families related but separate. |
| Prewarm | Keep only a bounded shared-memory prewarm if it can call the RAG 3 memory node path without creating a resolver observation. | Preserves current first-cycle behavior while avoiding the old full RAG-first path. |
| Cache | Remove durable initializer strategy cache from the new path. | There is no initializer strategy in RAG 3. Source-level cache remains source-owned. |

## Contracts And Data Shapes

### Interface Consistency Rule

The standalone harness, live LLM review tests, and production L2d integration
must all call:

```python
resolve_local_context(request, context, options=None)
```

They must all pass `LocalContextResolverRequestV1`,
`LocalContextResolverContextV1`, and `LocalContextResolverOptionsV1` and read
`LocalContextResolutionPacketV1`. No experiment-only request, context, option,
packet, slot compiler, or compatibility wrapper may be introduced.

### `LocalContextResolverRequestV1`

```python
{
    "schema_version": "local_context_resolver_request.v1",
    "objective": str,
    "source": "standalone_eval|l2d|prewarm|test|live_llm_review",
    "reason": str,
    "priority": "normal|high",
}
```

### `LocalContextResolverContextV1`

```python
{
    "schema_version": "local_context_resolver_context.v1",
    "character_name": str,
    "platform": str,
    "platform_channel_id": str,
    "global_user_id": str,
    "user_name": str,
    "local_time_context": dict,
    "prompt_message_context": dict,
    "chat_history_recent": list[dict],
    "chat_history_wide": list[dict],
    "conversation_progress": dict,
}
```

The context must be prompt-safe before any LLM stage sees it. It must not
include raw adapter wire syntax, raw message ids, database ids, embeddings,
binary payloads, credentials, callback URLs, or unbounded history.

### `LocalContextResolverOptionsV1`

```python
{
    "schema_version": "local_context_resolver_options.v1",
    "max_iterations": int,
    "max_nodes": int,
    "max_depth": int,
    "max_node_attempts": int,
    "max_subagent_attempts": int,
}
```

Default limits must be no looser than the current response-path budget without
explicit user approval. The initial target is:

```text
max_iterations <= 4
max_nodes <= 8
max_depth <= 3
max_node_attempts <= 2
max_subagent_attempts <= 1
```

### `LocalContextNodeV1`

```python
{
    "schema_version": "local_context_node.v1",
    "node_id": str,
    "node_kind": (
        "conversation_evidence"
        "|external_evidence"
        "|live_context"
        "|memory_evidence"
        "|person_context"
        "|recall_evidence"
        "|scoped_memory"
        "|subtask"
        "|synthesis"
    ),
    "objective": str,
    "parent_id": str | None,
    "children": list[str],
    "depends_on": list[str],
    "consumes": dict[str, str],
    "produces": list[str],
    "status": "pending|resolving|resolved|blocked|cannot_answer|collapsed",
    "investigation_summary": list[str],
    "knowledge_we_know_so_far": list[str],
    "knowledge_still_lacking": list[str],
    "recommended_next_iteration": list[str],
    "evidence_boundary_notes": list[str],
    "attempts": list[dict],
    "collapsed_into": str | None,
}
```

`depends_on`, `consumes`, and `produces` are deterministic graph fields. LLM
stages may describe semantic dependencies, but service code maps them into
validated graph references.

### `LocalContextArtifactV1`

```python
{
    "schema_version": "local_context_artifact.v1",
    "artifact_id": str,
    "artifact_type": (
        "conversation_ref|external_ref|live_context_ref|memory_ref|"
        "person_ref|recall_ref|semantic_packet"
    ),
    "producer_node_id": str,
    "summary": str,
    "projection_payload": dict,
    "source_policy": str,
    "prompt_visible": bool,
}
```

Artifacts may contain trace-only source refs in `projection_payload`, but
prompt-facing projection must strip raw storage ids and raw adapter ids before
cognition receives evidence.

### `LocalContextResolutionPacketV1`

```python
{
    "schema_version": "local_context_resolution_packet.v1",
    "investigation_summary": list[str],
    "knowledge_we_know_so_far": list[str],
    "knowledge_still_lacking": list[str],
    "recommended_next_iteration": list[str],
    "evidence_boundary_notes": list[str],
    "rag_result": dict,
    "graph": dict,
    "trace_summary": dict,
}
```

`rag_result` is the existing cognition/consolidation-facing evidence shape.
`graph` and `trace_summary` are debug/supervisor material and must not become
semantic evidence for cognition.

### Local Context Subagent Request

```python
{
    "schema_version": "local_context_subagent_request.v1",
    "node_id": str,
    "subagent": str,
    "action": str,
    "objective": str,
    "payload": dict,
    "constraints": dict,
}
```

Each subagent must refuse out-of-domain work and return a bounded result. It
must not decide persona stance or final wording.

## LLM Call And Context Budget

Before:

- RAG2 initializer: one response-path planner LLM call when full RAG runs.
- RAG2 helper agents: zero or more response-path subagent LLM calls depending
  on capability and worker.
- RAG2 evaluator/finalizer/continuation: response-path LLM calls may occur
  during evidence summarization and refined-query re-entry.
- Loop cap: four dispatch iterations.

After:

- RAG 3 graph planner: one response-path LLM call per local-context resolver
  run unless no retrieval is required and deterministic caller suppresses the
  capability.
- Active-node resolver: at most one LLM call per active node attempt.
- Collapse review: at most one LLM call per traversed node when collapse
  candidates exist; deterministic skip when no candidates exist.
- Bottom-up synthesis: one LLM call after traversal or deterministic synthesis
  when no evidence was collected.
- Local context subagents: reuse source-owned LLM calls inside bounded
  subagent contracts where the current source packages already need semantic
  argument generation or judging.

Step 1 standalone evaluation budget:

- Standalone runs use the same caps and public IO as production-intended RAG
  3.
- Each standalone live-LLM case must record planner calls, active-node
  resolver calls, collapse-review calls, synthesis calls, source-subagent
  calls, wall-clock duration, node count, graph depth, resolved/blocked node
  count, and whether a comparable RAG2 baseline was run.
- The human-readable review artifact must judge quality from real input and
  output, not from schema validity alone.
- Step 1 may be slower than production target while isolated, but Step 2
  cannot proceed unless the review artifact states whether the observed call
  count and duration fit the live chatbot response-path budget or identifies a
  plan update requirement.

Step 2 production budget:

- Production cutover must not raise response-path caps from the approved RAG 3
  defaults without explicit user approval.
- Production integration must preserve bounded blocking behavior for the live
  chat response path.

Context budget:

- Use 50k tokens as the default cap.
- Planner prompt inputs must use semantic projections: objective, current
  message context, compact recent history descriptors, local time context,
  active character role, current user role, and available source domains.
- Node resolver prompt inputs must include only the active node, parent chain
  summary, sibling summaries, known artifacts, recent attempts for that node,
  and bounded context.
- Do not include raw graph bookkeeping, raw trace records, raw storage rows,
  or unbounded chat history in LLM prompts.
- If a prompt cannot fit under budget after clipping, the node must block with
  an evidence boundary note rather than silently broadening search.

Latency rule:

- RAG 3 must reduce wrong-path retries rather than raise response-path caps.
- Any proposal to increase `COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS`,
  resolver max cycles, local context max iterations, max nodes, or max node
  attempts requires explicit user approval and live latency evidence.

## Change Surface

Step 1 creates and validates standalone RAG 3 only. Step 2 performs the
production wiring and RAG2 retirement after Step 1 evidence and user approval
are recorded.

### Delete

- `development_plans/active/short_term/rag3_router_interpreter_poc_experiment_plan.md`
  - Removed because the user rejected the compatible router/interpreter POC in
    favor of a bigbang complex-resolver-aligned RAG 3 plan.

### Create

- `development_plans/active/short_term/rag3_local_context_resolver_bigbang_plan.md`
  - New active RAG3 draft.
- `src/kazusa_ai_chatbot/local_context_resolver/`
  - New family-local resolver package. Step 1 owns the standalone package and
    public entrypoint.
- `src/kazusa_ai_chatbot/local_context_resolver/README.md`
  - ICD for RAG 3 local context resolver.
- `tests/test_local_context_resolver_contracts.py`
  - Contract validation tests.
- `tests/test_local_context_resolver_graph.py`
  - Graph traversal, dependency, collapse, and blocked-node tests.
- `tests/test_local_context_resolver_projection.py`
  - `LocalContextResolutionPacketV1` to `rag_result` projection tests.
- `tests/test_local_context_resolver_integration.py`
  - Cognition resolver capability integration tests.
- `tests/test_local_context_resolver_live_llm.py`
  - One-case-at-a-time live LLM review tests.
- `tests/test_local_context_resolver_standalone.py`
  - Standalone public-IO tests proving the harness and future production
    caller use the same request/context/options and packet.
- `test_artifacts/local_context_resolver/rag3_standalone_live_llm_review.md`
  - Agent-authored Step 1 quality review produced from real raw evidence.
- `test_artifacts/local_context_resolver/rag3_efficiency_summary.json`
  - Raw Step 1 call-count, duration, graph-size, and baseline-comparison
    evidence.

### Modify

- `development_plans/README.md`
  - Replace the old active RAG3 entry with this plan.
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
  - Step 2 only: route `local_context_recall` to the RAG 3 public entrypoint.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Step 2 only: preserve public helper behavior while delegating
    local-context evidence to RAG 3.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Step 2 only: preserve or rewire bounded shared-memory prewarm through the
    RAG 3 memory path without creating a resolver observation.
- `src/kazusa_ai_chatbot/rag/README.md`
  - Mark RAG2 supervisor as retired and source packages as retained evidence
    providers when applicable.
- `docs/SUBAGENT_INTERFACES.md`
  - Add local-context resolver subagent family vocabulary without creating a
    universal subagent abstraction.
- `README.md` and `docs/HOWTO.md`
  - Replace production RAG2 architecture text with RAG 3 local-context
    resolver text.
- `tests/test_rag_phase3_initializer_live_llm.py`
  - Step 2 only: replace canonical initializer live-LLM route expectations
    with RAG 3 graph/packet live-LLM expectations.
- `tests/test_rag_phase3_real_conversation_live_llm.py`
  - Step 2 only: carry real-conversation-derived route expectations into RAG
    3 behavior cases.
- `tests/test_rag_phase3_supervisor_integration.py`
  - Step 2 only: replace RAG2 slot-supervisor integration expectations with
    RAG 3 production integration expectations.
- `tests/test_persona_supervisor2_rag_supervisor2_live.py`
  - Step 2 only: replace live supervisor expectations with RAG 3 production
    behavior checks.
- `tests/test_rag_phase4_continuation_live_llm.py`
  - Step 2 only: remove refined-query continuation as canonical behavior and
    replace retained quality cases with graph dependency or lacking-evidence
    packet checks.
- `tests/test_rag_projection.py`
  - Step 2 only: keep `rag_result` projection coverage while replacing
    slot-supervisor trace expectations with RAG 3 packet projection
    expectations.

### Retire From Production Imports

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
- `src/kazusa_ai_chatbot/rag/quote_aware_sequence.py`

Remove these files only after greps prove no production import remains and the
replacement tests cover their former public behavior.

### Keep

- `src/kazusa_ai_chatbot/rag/conversation_evidence/`
- `src/kazusa_ai_chatbot/rag/memory_evidence/`
- `src/kazusa_ai_chatbot/rag/person_context/`
- `src/kazusa_ai_chatbot/rag/live_context/`
- `src/kazusa_ai_chatbot/rag/recall/`
- `src/kazusa_ai_chatbot/rag/web_agent3/`

These packages remain source-owned retrieval providers unless implementation
discovers an import boundary that requires moving code into
`local_context_resolver/subagent/`. Moving code must preserve source semantics
and tests.

## Overdesign Guardrail

- Actual problem: RAG2 local-context recall relies on an overloaded initializer
  and prose slot dependencies, causing wrong search focus and fragile
  multi-hop retrieval under local-model constraints.
- Minimal change: replace only the local-context recall supervisor
  architecture with a complex-resolver-aligned graph resolver while preserving
  downstream `rag_result` and existing source-owned retrieval packages.
- Ownership boundaries: LLMs own semantic graph planning, active-node semantic
  decisions, and packet synthesis; deterministic code owns graph validation,
  dependency binding, caps, source execution, cache mechanics, projection,
  persistence boundaries, and service wiring.
- Rejected complexity: universal resolver base class, shared complex/RAG
  subagent registry, RAG2 slot compatibility compiler, dual production paths,
  fallback to old initializer, raw backend parameter generation by planner,
  higher live caps, generic tools, new DB collections, and unrelated cognition
  or dialog rewrites.
- Evidence threshold: add any rejected complexity only after deterministic
  and live LLM RAG 3 tests show the simpler graph resolver cannot satisfy the
  accepted RAG2 behavior matrix and the user approves a follow-up plan.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside
  `local_context_resolver`, RAG local-context recall wiring, and listed docs
  as high-scrutiny changes. Updating an existing module outside the target
  module or introducing a new code path, prompt, or variable requires strong
  justification in `Change Surface` before implementation.
- The responsible agent may remove code from the existing RAG2 supervisor path
  after greps and tests prove the replacement contract is wired.
- The responsible agent must search the codebase for existing equivalent
  behavior before implementing helpers. If equivalent behavior exists, move or
  reuse it at the proper ownership boundary rather than duplicating it.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors unless explicitly
  listed in `Must Do`.
- If this plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

### Step 1 - Standalone RAG 3

1. Parent establishes standalone public-IO and contract tests for
   `LocalContextResolverRequestV1`, `LocalContextResolverContextV1`,
   `LocalContextResolverOptionsV1`, `LocalContextNodeV1`,
   `LocalContextArtifactV1`, and `LocalContextResolutionPacketV1`.
2. Parent records baseline expected failures for the missing package, missing
   public entrypoint, and missing standalone packet.
3. Parent starts one production-code subagent with this approved plan, the
   mandatory skills, the focused standalone test contract, and the
   standalone-only production-code ownership boundary.
4. Production-code subagent creates `local_context_resolver` contracts,
   validators, constants, graph helpers, public entrypoint, and packet
   projection without editing production callers.
5. Parent adds graph traversal, packet projection, and standalone harness tests
   while the subagent implements standalone contracts.
6. Production-code subagent implements graph planner, active-node resolver,
   collapse review, bottom-up synthesis, local-context subagent registry, and
   projection behind the public entrypoint.
7. Parent runs focused deterministic standalone tests and records results.
8. Parent runs selected standalone real-LLM cases one at a time through the
   public entrypoint, inspects raw evidence, and writes the human-readable
   debug review artifact.
9. Parent records Step 1 quality, performance, efficiency, raw evidence paths,
   and approval status in `Execution Evidence`.
10. Parent stops before Step 2 unless explicit user approval for production
    cutover is recorded after Step 1 evidence.

### Step 2 - Bigbang Production Cutover

11. Parent rereads this plan and establishes production integration tests for
    cognition resolver `local_context_recall` and shared-memory prewarm.
12. Parent records expected failures or current RAG2 baseline behavior for the
    production integration tests.
13. Parent starts one production-code subagent with this approved plan, Step 1
    evidence, the mandatory skills, the focused production integration
    contract, and the RAG2 replacement ownership boundary.
14. Production-code subagent rewires production local-context recall to RAG 3,
    preserves the `rag_result` projection, and retires RAG2 supervisor imports.
15. Parent converts or replaces RAG2 initializer live-LLM tests with RAG 3
    graph/packet live-LLM cases.
16. Parent runs focused deterministic tests, integration tests, static greps,
    and selected one-at-a-time live LLM cases.
17. Parent updates docs and lifecycle registry.
18. Parent starts independent code-review subagent after planned verification
    passes.
19. Parent fixes review findings within the approved change surface and reruns
    affected verification before final sign-off.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it; closes after planned
  production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  validation work, documentation, evidence, and plan-progress updates while the
  production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - standalone public-IO contract baseline established
  - Covers: implementation steps 1-2.
  - Verify: focused standalone and contract tests fail for missing RAG 3
    package, public entrypoint, or packet.
  - Evidence: record command output and expected failure in
    `Execution Evidence`.
  - Handoff: next agent starts Stage 2.
  - Sign-off: `Codex/2026-07-04` after verification and evidence are recorded.
- [x] Stage 2 - standalone local-context resolver implemented
  - Covers: implementation steps 3-7.
  - Verify: standalone, contract, graph, and projection tests pass without
    modifying production callers.
  - Evidence: record changed files, commands, and results.
  - Handoff: next agent starts Stage 3.
  - Sign-off: `Codex/2026-07-04` after verification and evidence are recorded.
- [x] Stage 3 - standalone real-LLM evaluation complete
  - Covers: implementation steps 8-9.
  - Verify: selected live LLM cases run one at a time through
    `resolve_local_context`, raw evidence is captured, and a human-readable
    review artifact evaluates quality, performance, and efficiency.
  - Evidence: record raw evidence paths, review artifact path, efficiency
    summary path, and user approval status.
  - Handoff: next agent stops unless user approval for Stage 4 is recorded.
  - Sign-off: `Codex/2026-07-04` after verification and evidence are recorded.
- [x] Stage 4 - production cutover approval recorded
  - Covers: implementation step 10.
  - Verify: Step 1 evidence is present and explicit user approval for Step 2
    production cutover is recorded in `Execution Evidence`.
  - Evidence: record the approval message or linked decision.
  - Handoff: next agent starts Stage 5.
  - Sign-off: `Codex/2026-07-04` after verification and evidence are recorded.
- [x] Stage 5 - production local-context recall cutover complete
  - Covers: implementation steps 11-16.
  - Verify: cognition resolver integration tests and converted RAG 3 behavior
    tests pass.
  - Evidence: record greps proving no production import uses RAG2 initializer
    or slot supervisor.
  - Handoff: next agent starts Stage 6.
  - Sign-off: `Codex/2026-07-04` after verification and evidence are recorded.
- [x] Stage 6 - documentation and lifecycle updated
  - Covers: implementation step 17.
  - Verify: `git diff --check` and static documentation greps listed in
    `Verification`.
  - Evidence: record documentation files changed and grep results.
  - Handoff: next agent starts Stage 7.
  - Sign-off: `Codex/2026-07-04` after verification and evidence are recorded.
- [x] Stage 7 - independent code review complete
  - Covers: implementation steps 18-19.
  - Verify: independent review reports no unresolved blockers, and affected
    tests are rerun after any fixes.
  - Evidence: record review findings, fixes, rerun commands, and approval
    status.
  - Handoff: plan can be marked completed only after this is signed off.
  - Sign-off: `Codex/2026-07-04` after verification and evidence are recorded.

## Verification

Use `venv\Scripts\python.exe` for Python commands.

### Static Greps

- `rg "unknown_slots|rag_initializer|resolved in slot|refined_query" src\kazusa_ai_chatbot tests`
  - Expected after cutover: no production-path references to removed RAG2
    supervisor control. Allowed matches must be limited to archive docs,
    explicit migration notes, or tests proving removal.
- `Test-Path development_plans\active\short_term\rag3_router_interpreter_poc_experiment_plan.md`
  - Expected after this draft: `False`.
- `Get-ChildItem development_plans\active -Recurse -Filter '*rag3*.md' | Select-Object -ExpandProperty Name`
  - Expected after this draft: exactly
    `rag3_local_context_resolver_bigbang_plan.md`.
- `rg "local_context_resolver" src\kazusa_ai_chatbot tests docs README.md development_plans`
  - Expected after implementation: production entrypoint, tests, and docs are
    present.

### Step 1 Standalone Tests

- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_standalone.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_contracts.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_graph.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_projection.py -q`

Expected after Step 1: all listed tests pass without modifying production
callers.

### Step 2 Production Integration Tests

- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_integration.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_supervisor_integration.py tests\test_rag_projection.py -q`

Expected after Step 2: all listed tests pass with `local_context_recall`
wired to RAG 3 and no production import depending on the RAG2 initializer or
slot supervisor.

### Step 2 Existing Regression Tests

- `venv\Scripts\python.exe -m pytest tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_persona_graph.py tests\test_cognition_resolver_l2d_contract.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_shared_memory_prewarm.py tests\test_persona_supervisor2_cognition_prewarm.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py tests\test_rag_projection.py -q`

### Step 1 Live LLM Review

Run one case at a time through the same public entrypoint used by production
cutover, then inspect trace output and write the agent-authored review
artifact:

- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "standalone_current_time" -s`
- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "standalone_exact_phrase" -s`
- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "standalone_current_user_url" -s`
- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "standalone_scoped_memory" -s`
- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "standalone_napcat_command_anchor" -s`

Required Step 1 artifacts:

- `test_artifacts/local_context_resolver/rag3_standalone_live_llm_review.md`
  - Expected: human-readable review with run context, inputs, outputs,
    decisions, quality notes, validation, raw evidence paths, and human
    attention points.
- `test_artifacts/local_context_resolver/rag3_efficiency_summary.json`
  - Expected: raw structured evidence for call counts, wall-clock duration,
    node counts, graph depth, resolved/blocked counts, and RAG2 baseline
    comparison availability.

### Step 2 Live LLM Regression

Run selected production-wired cases one at a time after the bigbang cutover.
Each case must still use `resolve_local_context` through the production
`local_context_recall` path and must have raw evidence plus a human-readable
review artifact when quality is being judged.

- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "production_current_time" -s`
- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "production_exact_phrase" -s`
- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "production_current_user_url" -s`
- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "production_scoped_memory" -s`
- `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "production_napcat_command_anchor" -s`

### Static Quality

- `git diff --check`
- `venv\Scripts\python.exe -m py_compile <changed python files>`

## Independent Plan Review

Run this gate before approval or implementation. Prefer a reviewer that did
not draft the plan. If no separate reviewer is available, the drafting agent
must reread this plan, the RAG2 README, complex task resolver README,
cognition resolver README, subagent interface guide, and relevant tests from a
fresh-review posture.

Review scope:

- The plan cleanly removes the obsolete RAG3 router/interpreter POC and leaves
  only one active RAG3 plan.
- The two-step execution model is unambiguous: standalone implementation and
  real-LLM evaluation first, production bigbang cutover only after evidence
  and explicit user approval.
- The same public IO is used in standalone evaluation, live LLM review, and
  production cutover.
- The architecture aligns with `complex_task_resolver` without creating a
  universal resolver/subagent abstraction.
- RAG 3 remains local-context recall and does not absorb public answer
  research.
- The plan gives full, concrete instructions for execution agents: contracts,
  change surface, implementation order, verification gates, progress
  checklist, and evidence requirements.
- Agent creativity is tightly bounded: no unresolved choices, broad verbs,
  optional fallbacks, compatibility shims, private helper freedom, or unowned
  side paths remain.
- Stage boundaries are explicit between RAG 3, cognition, dialog,
  consolidation, adapters, and complex resolver.

Record blockers, non-blocking findings, required edits, questions, and
approval status. Approve only when blockers are resolved.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including prior-stage artifacts, focused and
  regression tests, execution evidence, next-stage handoff notes, and
  path-safe commands.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- This draft or a superseding approved plan is the only active RAG3 plan.
- Step 1 standalone RAG 3 exists behind
  `resolve_local_context(request, context, options=None)` and returns
  `LocalContextResolutionPacketV1` including `rag_result`.
- Step 1 standalone real-LLM review artifacts evaluate quality, performance,
  and efficiency from real raw evidence.
- Step 2 production cutover begins only after Step 1 evidence and explicit
  user approval are recorded.
- RAG 3 local-context resolver is the only production implementation behind
  `local_context_recall`.
- RAG2 initializer, prose slot queue, prefix dispatcher, evaluator/finalizer
  supervisor loop, and refined-query initializer re-entry are absent from
  production imports.
- Downstream cognition and consolidation still receive sanitized `rag_result`
  evidence.
- Graph dependencies are carried through typed node/artifact references, not
  prose slot numbers.
- RAG 3 tests cover the real-LLM initializer behavior matrix: current time,
  active agreement recall, exact phrase provenance, active-character self
  words, current-user URL recall, scoped current-user memory, named person
  impression, official address memory, and `#napcat` command-anchor retrieval.
- Deterministic tests, selected live LLM tests, static greps, and
  documentation checks listed in `Verification` pass.
- Independent code review has no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Live latency grows because graph traversal adds LLM calls. | Keep caps no looser than current RAG2 budget and measure live LLM cases one at a time. | Live LLM trace artifacts and call-count review. |
| The plan accidentally creates a universal resolver abstraction. | Keep `local_context_resolver` and `complex_task_resolver` family-local; align lifecycle only. | Subagent docs and code review. |
| Existing source packages depend on slot text. | Add typed subagent request adapters or move source logic behind typed interfaces without compiling graphs to slots. | Source package tests and static greps. |
| Standalone RAG 3 drifts from production needs. | Require Step 1 and Step 2 to share the same public IO, packet shape, caps, and review artifacts. | Standalone public-IO tests and production integration tests. |
| `rag_result` projection leaks graph internals. | Keep graph and trace material separate from prompt-facing evidence. | Projection tests and sanitizer tests. |
| Current-user or active-character speaker scope regresses. | Encode scope as typed constraints and artifacts. | Converted live LLM cases and conversation evidence regression tests. |
| Shared-memory prewarm reintroduces hidden RAG-first behavior. | Keep prewarm bounded, shared-memory-only, and not a resolver observation. | Prewarm tests. |
| RAG2 removal breaks health/cache docs or operational status. | Update docs and retain source-level cache metrics where still valid. | README/HOWTO greps and service tests. |

## Execution Evidence

### 2026-07-04 draft creation

- User requested removal of all active RAG3 plans and a new RAG 3 plan based
  on the complex-resolver-aligned local-context resolver discussion.
- Current active RAG3 plan found and removed:
  `development_plans/active/short_term/rag3_router_interpreter_poc_experiment_plan.md`.
- New active RAG3 draft created:
  `development_plans/active/short_term/rag3_local_context_resolver_bigbang_plan.md`.
- Context read before drafting:
  - `README.md`
  - `docs/HOWTO.md`
  - `development_plans/README.md`
  - `docs/SUBAGENT_INTERFACES.md`
  - `src/kazusa_ai_chatbot/rag/README.md`
  - `src/kazusa_ai_chatbot/complex_task_resolver/README.md`
  - `src/kazusa_ai_chatbot/cognition_resolver/README.md`
  - `src/kazusa_ai_chatbot/nodes/README.md`
  - `.agents/skills/development-plan/references/plan_contract.md`
  - `.agents/skills/development-plan/references/cutover_policy.md`
  - `.agents/skills/development-plan/references/execution_gates.md`
- This draft is not approved for implementation.

### 2026-07-04 requirements update and plan review

- User added two requirements:
  - Preserve the RAG 3 input and output interface across standalone
    evaluation and production cutover.
  - Execute the work in two steps: standalone RAG 3 with real-LLM
    quality/performance/efficiency evaluation first, then full RAG2
    replacement in a bigbang production cutover.
- Plan updated to make `resolve_local_context(request, context, options=None)`
  and `LocalContextResolutionPacketV1` the shared Step 1 and Step 2 contract.
- Plan updated to forbid Step 1 production wiring and to require Step 1
  evidence plus explicit user approval before Step 2 starts.
- Drafting-agent plan review performed against `development-plan`,
  `local-llm-architecture`, `debug-llm`, `plan_contract.md`,
  `cutover_policy.md`, and `execution_gates.md`.
- Review findings addressed:
  - Verification mixed standalone tests with production integration tests.
    Fixed by splitting Step 1 standalone tests, Step 2 production integration
    tests, Step 2 regression tests, Step 1 live-LLM review, and Step 2
    live-LLM regression.
  - Cutover policy table still described tests and documentation as a single
    bigbang phase. Fixed by changing both rows to two-step replacement.
  - Change surface used an imprecise current-test reference. Fixed by naming
    the exact RAG2 initializer, supervisor, continuation, and projection test
    files that Step 2 must convert.
  - Step 2 live-LLM regression lacked exact commands. Fixed by adding
    production-wired one-case-at-a-time pytest commands.
- Review status: no unresolved plan-review blockers remain. This draft still
  requires user approval before implementation.

### 2026-07-04 Step 1 execution approval

- User approved this plan and requested execution of Step 1 only.
- User required exactly one production-code subagent for production code
  changes and exactly one review subagent for code review. Parent agent owns
  tests, verification, evidence, and review remediation.
- Execution boundary: implement standalone RAG 3 only. Do not wire RAG 3 into
  production `local_context_recall`, shared-memory prewarm, cognition, dialog,
  adapters, or production callers during this step.

### 2026-07-04 Stage 1 baseline

- Added focused Step 1 tests:
  - `tests/test_local_context_resolver_standalone.py`
  - `tests/test_local_context_resolver_contracts.py`
  - `tests/test_local_context_resolver_graph.py`
  - `tests/test_local_context_resolver_projection.py`
  - `tests/test_local_context_resolver_live_llm.py`
- Baseline command:
  `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_standalone.py tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_graph.py tests\test_local_context_resolver_projection.py -q`
- Expected baseline result recorded: collection fails with
  `ModuleNotFoundError: No module named 'kazusa_ai_chatbot.local_context_resolver'`.
- Stage 1 sign-off: `Codex/2026-07-04`.

### 2026-07-04 Stage 2 standalone implementation

- Production-code subagent:
  `019f2afa-6e7f-7c42-91bc-5d5cf15b34d8` (`Galileo`) implemented the
  standalone package under `src/kazusa_ai_chatbot/local_context_resolver/**`.
- Parent review/remediation:
  - Added a deterministic collapse test proving prompt-safe `candidate_ref`
    values map back to internal graph nodes without exposing `node_id` to the
    collapse prompt payload.
  - Updated collapse review to use `target_candidate_ref` in the LLM-facing
    contract and deterministic internal id mapping in service code.
  - Added deterministic skip for collapse review when no resolved same-kind
    candidates exist.
  - Dropped planner-produced `synthesis` rows before graph construction and
    kept final synthesis owned by the service.
  - Added prompt-facing projection sanitation and deduplication for
    `rag_result` payloads.
- Changed standalone production files:
  - `src/kazusa_ai_chatbot/local_context_resolver/__init__.py`
  - `src/kazusa_ai_chatbot/local_context_resolver/constants.py`
  - `src/kazusa_ai_chatbot/local_context_resolver/contracts.py`
  - `src/kazusa_ai_chatbot/local_context_resolver/graph.py`
  - `src/kazusa_ai_chatbot/local_context_resolver/service.py`
  - `src/kazusa_ai_chatbot/local_context_resolver/stages.py`
  - `src/kazusa_ai_chatbot/local_context_resolver/README.md`
- Deterministic verification:
  - `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_standalone.py -q`
    -> 6 passed.
  - `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_contracts.py -q`
    -> 4 passed.
  - `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_graph.py -q`
    -> 3 passed.
  - `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_projection.py -q`
    -> 1 passed.
  - Combined focused suite:
    `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_standalone.py tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_graph.py tests\test_local_context_resolver_projection.py -q`
    -> 14 passed.
- Syntax/static verification:
  - `venv\Scripts\python.exe -m py_compile` over all changed package and test
    Python files -> passed.
  - `git diff --check` -> passed with existing CRLF warning for
    `development_plans/README.md`.
  - `rg -n 'local_context_resolver' src\kazusa_ai_chatbot | rg -v 'src\\kazusa_ai_chatbot\\local_context_resolver'`
    -> no matches; no production caller wiring added.
  - `rg -n 'target_node_id|unknown_slots|rag_initializer|resolved in slot|refined_query|except Exception|except:' src\kazusa_ai_chatbot\local_context_resolver tests\test_local_context_resolver_standalone.py tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_graph.py tests\test_local_context_resolver_projection.py tests\test_local_context_resolver_live_llm.py`
    -> no matches.
  - `Test-Path development_plans\active\short_term\rag3_router_interpreter_poc_experiment_plan.md`
    -> `False`.
  - `Get-ChildItem development_plans\active -Recurse -Filter '*rag3*.md' | Select-Object -ExpandProperty Name`
    -> `rag3_local_context_resolver_bigbang_plan.md`.
- Stage 2 sign-off: `Codex/2026-07-04`.

### 2026-07-04 Stage 3 standalone live-LLM evaluation

- Live LLM note: repo `pytest.ini` deselects `live_llm` by default, so each
  planned one-case command was run with explicit `-m live_llm`.
- Commands run one case at a time:
  - `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "standalone_current_time" -s -m live_llm`
    -> passed; raw evidence:
    `test_artifacts/local_context_resolver/raw/standalone_current_time.json`.
  - `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "standalone_exact_phrase" -s -m live_llm`
    -> passed; raw evidence:
    `test_artifacts/local_context_resolver/raw/standalone_exact_phrase.json`.
  - `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "standalone_current_user_url" -s -m live_llm`
    -> passed; raw evidence:
    `test_artifacts/local_context_resolver/raw/standalone_current_user_url.json`.
  - `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "standalone_scoped_memory" -s -m live_llm`
    -> passed; raw evidence:
    `test_artifacts/local_context_resolver/raw/standalone_scoped_memory.json`.
  - `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_live_llm.py -q -k "standalone_napcat_command_anchor" -s -m live_llm`
    -> passed; raw evidence:
    `test_artifacts/local_context_resolver/raw/standalone_napcat_command_anchor.json`.
- Efficiency summary:
  `test_artifacts/local_context_resolver/rag3_efficiency_summary.json`.
  Final observed totals:
  - current time: 3 LLM calls, 11.938s.
  - exact phrase: 5 LLM calls, 17.967s.
  - current-user URL: 5 LLM calls, 17.502s.
  - scoped memory: 4 LLM calls, 12.465s.
  - `#napcat`: 5 LLM calls, 23.562s.
- Agent-authored live review:
  `test_artifacts/local_context_resolver/rag3_standalone_live_llm_review.md`.
- Quality result:
  - `#napcat` direct-address case preserved `#napcat` as the command anchor
    and retrieved the `napcat` memory rule.
  - Current time, exact phrase, current-user URL, and scoped memory cases
    produced the expected local-context evidence.
  - Final live raw artifacts include per-stage raw model outputs, parsed
    outputs, model names, routes, and model-facing input payloads.
  - Final live raw artifacts passed a scan for forbidden model-input and
    prompt-facing `rag_result` metadata: no `message_id`,
    `source_message_id`, `scope_global_user_id`, platform channel id, raw
    `timestamp`, or raw UTC timestamp keys found.
  - Remaining non-blocking issue: near-duplicate conversation evidence can
    still appear when the LLM emits semantically similar rows with different
    fields.
  - Remaining production-cutover risk: standalone live durations are
    11.938-23.562s, so Step 2 needs explicit latency review before production
    replacement.
- User approval status for Stage 4 / Step 2 production cutover: not recorded.
- Stage 3 sign-off: `Codex/2026-07-04`.

### 2026-07-04 Step 1 independent review remediation

- Independent review subagent:
  `019f2b1b-aa5a-78d1-948e-cde6ca3ae96a` (`Rawls`), review-only, no file
  edits.
- Review result before remediation: not approved.
- Findings addressed:
  - Prompt-safe context was not enforced before LLM calls. Fixed by applying
    recursive prompt-payload sanitation inside `_compact_context` before
    planner, active-node, collapse, and synthesis stages receive context.
  - Active plan contract had stale node and artifact enum values. Fixed the
    `LocalContextNodeV1`, `LocalContextArtifactV1`, and local subagent request
    contract snippets in this plan.
  - JSON repair could add hidden LLM calls not counted in efficiency evidence.
    Fixed RAG 3 stage parsing to use deterministic JSON parsing only, without
    `parse_llm_json_output` / `JSON_REPAIR_LLM`.
  - Collapse could hide blocked evidence. Fixed collapse application so only
    resolved active nodes may be collapsed.
  - Live artifacts lacked raw per-stage model outputs. Fixed live evidence to
    include stage traces with raw model output, parsed output, model, route,
    and model-facing input payload for each LLM stage.
- Review-remediation tests:
  - `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_standalone.py tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_graph.py tests\test_local_context_resolver_projection.py -q`
    -> 14 passed.
  - `venv\Scripts\python.exe -m py_compile` over all changed package and test
    Python files -> passed.
  - `git diff --check` -> passed with existing CRLF warning for
    `development_plans/README.md`.
  - `rg -n 'local_context_resolver' src\kazusa_ai_chatbot | rg -v 'src\\kazusa_ai_chatbot\\local_context_resolver'`
    -> no matches; no production caller wiring added.
  - `rg -n 'parse_llm_json_output|JSON_REPAIR_LLM|target_node_id|unknown_slots|rag_initializer|resolved in slot|refined_query|except Exception|except:' src\kazusa_ai_chatbot\local_context_resolver tests\test_local_context_resolver_standalone.py tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_graph.py tests\test_local_context_resolver_projection.py tests\test_local_context_resolver_live_llm.py`
    -> no matches.
  - Live raw artifact scan -> stage traces present and no forbidden metadata
    in stage input payloads or final `rag_result`.
- Stage 7 remains unchecked because Step 2 production cutover stages are not
  approved or executed in this Step 1-only request.

### 2026-07-04 RAG2 vs RAG3 real-LLM comparison goal

- User requested a full real-LLM comparison between current RAG2 and RAG3,
  with standalone RAG3 blockers fixed before continuing comparison evidence.
- Comparison scope:
  - RAG2 side: current live `rag_initializer` route baseline, one real LLM
    call per case, with prompt/output captured.
  - RAG3 side: standalone `resolve_local_context(request, context, options)`
    public IO over equivalent prompt-safe supplied context, with packet and
    per-stage raw traces captured.
  - RAG3 remains unwired from production `local_context_recall`.
- Added comparison live-LLM tests:
  `tests/test_local_context_resolver_rag2_vs_rag3_live_llm.py`.
- Ran nine accepted behavior-matrix cases one at a time with `-m live_llm`:
  current time, active agreement recall, exact phrase provenance,
  active-character self words, current-user URL recall, scoped current-user
  memory, named person impression, official address memory, and `#napcat`
  command-anchor retrieval.
- RAG3 blocker found:
  - `active_agreement_recall` initially returned a bounded blocked packet
    because the synthesis stage emitted JSON with a raw control character
    inside a string literal.
- RAG3 blocker fix:
  - `src/kazusa_ai_chatbot/local_context_resolver/stages.py` now performs
    deterministic control-character escaping inside JSON strings after normal
    parsing fails. No `JSON_REPAIR_LLM` path was added.
  - `tests/test_local_context_resolver_standalone.py` now covers the parser
    case.
- Validation after fix:
  - Focused parser unit test -> passed.
  - `active_agreement_recall` real-LLM comparison rerun -> passed and produced
    usable RAG3 agreement evidence instead of a blocked packet.
  - Deterministic suite:
    `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_standalone.py tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_graph.py tests\test_local_context_resolver_projection.py -q`
    -> 15 passed.
  - `venv\Scripts\python.exe -m py_compile` over changed RAG3 package/tests
    -> passed.
  - `git diff --check` over changed RAG3 package/tests -> passed.
- Comparison artifacts:
  - Raw per-case JSON:
    `test_artifacts/local_context_resolver/comparison/raw/*.json`.
  - Aggregate JSON:
    `test_artifacts/local_context_resolver/comparison/rag2_vs_rag3_summary.json`.
  - Agent-authored report:
    `test_artifacts/local_context_resolver/comparison/rag2_vs_rag3_real_llm_report.md`.
- Comparison result:
  - RAG2 route checks passed 7/9 cases.
  - RAG2 failed the `#napcat` command-anchor case by focusing on
    `Recall: retrieve active_episode_agreement relevant to checking NapCat
    status` instead of the `#napcat` memory command anchor.
  - RAG2 also exposed a route-surface issue where the active-agreement raw
    `Recall:` output normalized into a non-prefix `召回候选：...` slot.
  - RAG3 produced usable evidence for all nine cases after the parser fix and
    handled `#napcat`, official address, scoped current-user memory, exact
    phrase provenance, and current-user URL recall better than the RAG2
    initializer route baseline.
  - RAG3 remaining optimization targets: field classification consistency
    (`third_party_profiles`, `external_evidence`, `recall_evidence`),
    temporal-grounding restraint when only message timestamps are present,
    over-broad missing-detail rows, and latency.

### 2026-07-04 RAG3 internal-knowledge optimization goal

- User requested optimization of standalone RAG3 so internal knowledge
  handling is better than current RAG2 based on the comparison result, with a
  stop condition of at most two attempts per real-LLM test case.
- Production boundary:
  - RAG3 remains standalone.
  - No `local_context_resolver` production wiring was added outside
    `src/kazusa_ai_chatbot/local_context_resolver`.
- Optimization changes:
  - `src/kazusa_ai_chatbot/local_context_resolver/stages.py` now instructs
    active-node and synthesis stages that chat row `local_time` values are
    message timestamps only; current date/time must come from
    `local_time_context`.
  - `stages.py` now gives explicit artifact/source ownership for
    `memory_ref`, `conversation_ref`, `person_ref`, `recall_ref`, and
    `external_ref`, reducing mixed evidence fields.
  - `stages.py` now tells RAG3 to leave `knowledge_still_lacking` empty for
    confirmation, provenance, quote, URL, speaker, or command-definition
    objectives once the requested anchor is found.
  - `src/kazusa_ai_chatbot/local_context_resolver/service.py` now normalizes
    non-empty strings for semantic list fields such as `produces` into
    one-item lists while keeping `attempts` strict.
  - `service.py` now strips prompt-unsafe keys such as `local_time` and raw
    timestamps from final `rag_result`, redacts embedded metadata/id/timestamp
    string values, and removes the generic synthesis-to-`memory_evidence`
    fallback.
  - `service.py` now normalizes stage artifact-type aliases such as
    `third_party_profiles` to canonical source-owned artifact types such as
    `person_ref`.
  - `tests/test_local_context_resolver_rag2_vs_rag3_live_llm.py` now evaluates
    RAG3 quality targets, prompt-facing leakage, missing rows, and unexpected
    non-empty source fields instead of only recording targets.
- Tests added/updated:
  - `test_stage_prompts_keep_source_field_and_time_boundaries`.
  - `test_node_update_accepts_single_semantic_string_lists`.
  - `test_prompt_payload_sanitizes_string_values_and_local_time`.
  - `test_rag_result_does_not_fallback_synthesis_into_memory`.
  - `test_artifact_type_accepts_stage_semantic_aliases`.
- Real-LLM optimization reruns:
  - `active_agreement_recall`: 2 attempts, strict quality passed on attempt 2;
    active agreement landed in `recall_evidence`, no inferred current-time row,
    no missing background.
  - `current_user_url_recall`: 2 attempts, strict quality passed on attempt 2;
    URL provenance landed in `conversation_evidence`, not memory or recall,
    and `local_time` was absent from final `rag_result`.
  - `exact_phrase_provenance`: 1 optimization rerun, strict quality passed;
    exact quote stayed in `conversation_evidence`, not recall.
  - `official_address_memory`: 1 optimization rerun, strict quality passed;
    durable address stayed in `memory_evidence`.
  - `named_person_impression`: 2 attempts, capped with a blocked packet on
    attempt 2 because the model emitted `artifact_type:
    third_party_profiles`; deterministic alias normalization now covers the
    exact blocked shape, but no third live rerun was taken.
  - `active_character_self_words`: 2 attempts, passed after second prompt
    tightening; optional project-detail missing rows removed. Not rerun under
    strict harness because the attempt cap was already reached.
  - `scoped_current_user_memory`: 2 attempts, passed after schema tolerance
    fix for single-string `produces`; no raw user-id leakage. Not rerun under
    strict harness because the attempt cap was already reached.
  - `napcat_command_anchor`: 2 attempts, capped. Attempt 2 preserved the main
    targets and showed no missing target or timestamp/id leakage in the failed
    assertion, but failed strict quality, likely because of remaining
    source-field cleanliness around recall duplication. No third live rerun was
    taken.
- Current comparison aggregate after optimization:
  - RAG2 route checks: 7/9 passed.
  - Strict RAG3 quality checks saved in summary: 4 passed, 1 failed, 4 not
    rerun under strict harness because of the attempt cap or unchanged scope.
  - Saved strict pass cases: `active_agreement_recall`,
    `current_user_url_recall`, `exact_phrase_provenance`, and
    `official_address_memory`.
  - Saved strict failure: `named_person_impression`, with deterministic fix
    applied after the capped run.
- Validation:
  - `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_standalone.py tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_graph.py tests\test_local_context_resolver_projection.py -q`
    -> 20 passed.
  - `venv\Scripts\python.exe -m py_compile` over standalone RAG3 package and
    tests -> passed.
  - `git diff --check` -> passed with existing CRLF warning for
    `development_plans/README.md`.
  - Banned-pattern scan over standalone package/tests -> no matches.
  - Production-wiring scan -> no `local_context_resolver` references outside
    the standalone package.
  - Artifact consistency check -> nine raw artifacts present; RAG2 7/9; saved
    strict RAG3 quality 4/9.
- Optimization report:
  `test_artifacts/local_context_resolver/comparison/rag3_internal_knowledge_optimization_report.md`.
- Residual non-blocking risks:
  - `named_person_impression` needs one future live validation pass if the user
    authorizes exceeding the two-attempt cap.
  - `napcat_command_anchor` still needs future strict source-field cleanliness
    validation if the user authorizes exceeding the two-attempt cap.
  - RAG3 latency remains materially higher than the RAG2 initializer baseline.
  - Step 2 production cutover remains unexecuted and requires separate
    approval.

### 2026-07-04 RAG3 full-matrix optimization attempt

- User requested another RAG3 optimization attempt with a broader live-LLM
  harness, useful group-history retrieval seeds, a 100% pass goal, and
  minimized RAG3 LLM calls without violating LLM-first prompt rules.
- Retrieved QQ group `638473184` history and generated prompt-safe seed
  windows:
  `test_artifacts/local_context_resolver/rag3_seed_windows_638473184.json`.
  Useful windows covered `#napcat` teaching behavior, adjacent NapCat bot
  responses, Volcengine URL shares, reply-parent GPU context, and topic
  participant context.
- Added full-matrix live harness:
  `tests/test_local_context_resolver_full_matrix_live_llm.py`.
  It covers 19 standalone RAG3 cases over live context, recall, conversation
  evidence, current-user scope, scoped memory, durable memory, person context,
  command-anchor retrieval, group-history adjacency, external evidence, and
  multi-hop phrase/person/link dependency.
- Baseline failures found:
  - current-time live context omitted the exact `09:30` value and emitted an
    unsupported `live_evidence` projection key;
  - scoped user memories were projected into `memory_evidence` instead of
    `user_memory_unit_candidates`;
  - conversation-owned NapCat rows were duplicated into `recall_evidence`;
  - one topic-participant run blocked on malformed node-stage JSON before raw
    failed output was captured;
  - exact phrase and several real-history cases over-decomposed and spent
    unnecessary calls;
  - two test expectations were corrected where the objective did not actually
    require the over-strict source field.
- Optimization changes:
  - Planner prompt now prefers one source-domain node when one supplied source
    can satisfy speaker, quote, URL, and adjacent-context needs together.
  - Planner prompt now keeps command/tag/direct-address/chat-event work out of
    recall and keeps command behavior out of person context unless profile or
    impression is explicitly requested.
  - Node prompt now makes scoped `user_memory_units` and live-context projection
    ownership explicit.
  - Fully resolved graphs now use deterministic bottom-up aggregation of
    node-owned LLM semantic rows instead of an extra final synthesis LLM call.
  - Projection now normalizes scoped memory rows, conversation rows emitted in
    recall fields, recall agreement duplication, and model-emitted live
    evidence before producing prompt-facing `rag_result`.
  - Failed stage parses now record raw model output in stage traces.
- Final live full-matrix result:
  - 19/19 cases passed.
  - Total RAG3 LLM calls: 40.
  - Average RAG3 LLM calls: 2.105 per case.
  - 17 cases used 2 calls; 2 cases used 3 calls.
  - Final summary:
    `test_artifacts/local_context_resolver/full_matrix/rag3_full_matrix_summary.json`.
  - Human-readable report:
    `test_artifacts/local_context_resolver/full_matrix/rag3_full_matrix_optimization_report.md`.
- Verification:
  - Focused deterministic suite:
    `venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_standalone.py tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_graph.py tests\test_local_context_resolver_projection.py -q`
    -> 23 passed.
  - `venv\Scripts\python.exe -m py_compile` over changed standalone RAG3
    package files and tests -> passed.
  - Production-wiring grep for `local_context_resolver` outside the standalone
    package -> no matches.
  - Banned-pattern grep for JSON repair LLM and RAG2 slot-control patterns in
    the standalone package/focused tests -> no matches.
  - Step 2 production cutover remains unexecuted and requires separate
    approval.

### 2026-07-04 Stage 4 production cutover approval

- Approval source: user requested execution of Step 2 in the current Codex
  turn: "Based on the current status, I need you to start executing step 2 per
  plan that phase out the RAG2, replaced with RAG3."
- Approval scope:
  - Start the bigbang production cutover that replaces production
    `local_context_recall` with RAG 3.
  - Verify close to the production workflow with deterministic, E2E, and
    one-at-a-time real LLM tests.
  - Improve coverage and prevent regressions before final sign-off.
- Stage 4 sign-off: `Codex/2026-07-04`.

### 2026-07-04 Stage 5 production cutover execution

- Production cutover changes:
  - `cognition_resolver.capabilities.run_rag_evidence_for_persona_state`
    calls `resolve_local_context(...)` and projects
    `LocalContextResolutionPacketV1` through `project_local_context_packet`.
  - Shared-memory prewarm uses the RAG3 public entrypoint with source
    `prewarm` and filters scoped user-memory rows out of prewarm output.
  - `persona_supervisor2` no longer imports or wraps the RAG2 quote-aware
    supervisor for production recall.
  - Service lifespan no longer hydrates the RAG initializer cache; it hydrates
    media descriptor cache only.
  - `db_bootstrap()` no longer purges or prunes the retired RAG2 initializer
    cache during service startup.
  - Empty RAG payload construction in cognition resolver state no longer
    imports the RAG2 projection helper.
- RAG3 projection fixes made during cutover:
  - `rag_result.user_image.user_memory_context` remains present in empty and
    projected payloads.
  - `supervisor_trace.dispatched[*].source_refs` preserves private
    conversation row refs for past-dialog cognition when raw conversation
    artifacts contain row ids, while prompt-visible evidence strips those ids.
  - `user_memory_unit_candidates` is counted in local-context observation
    telemetry and prompt-safe summaries.
- Focused deterministic verification:
  - `venv\Scripts\python -m pytest tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_graph.py tests\test_local_context_resolver_projection.py tests\test_local_context_resolver_standalone.py tests\test_local_context_resolver_integration.py -q`
    -> 25 passed.
  - `venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_persona_graph.py tests\test_cognition_resolver_l2d_contract.py tests\test_shared_memory_prewarm.py tests\test_persona_supervisor2_cognition_prewarm.py -q`
    -> 69 passed.
  - `venv\Scripts\python -m pytest tests\test_past_dialog_cognition_rag_integration.py tests\test_past_dialog_cognition_context.py tests\test_past_dialog_cognition_prompt_boundaries.py -q`
    -> 15 passed.
  - `venv\Scripts\python -m pytest tests\test_rag_dialog_event_logging.py tests\test_multi_source_cognition_stage_00_regression_baseline.py::test_rag_skip_preserves_full_projected_shape tests\test_documentation_harmonization.py::test_howto_startup_order_matches_service_lifespan -q`
    -> 9 passed.
  - `venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_persona_supervisor2_schema.py tests\test_persona_supervisor2_action_selection.py tests\test_multi_source_cognition_stage_02_chat_episode_migration.py -q`
    -> 52 passed.
  - `venv\Scripts\python -m pytest tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_service_health.py tests\test_service_event_logging.py tests\test_reflection_cycle_stage1c_service.py tests\test_rag_cache2_persistent.py -q`
    -> 92 passed.
  - After removing the last startup initializer-cache maintenance hook:
    `venv\Scripts\python -m pytest tests\test_db.py -q`
    -> 71 passed, 13 deselected.
  - After the same startup cleanup:
    `venv\Scripts\python -m pytest tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_service_health.py tests\test_service_event_logging.py tests\test_reflection_cycle_stage1c_service.py tests\test_rag_cache2_persistent.py tests\test_documentation_harmonization.py::test_howto_startup_order_matches_service_lifespan -q`
    -> 93 passed.
- Static verification:
  - Production-path grep for RAG2 supervisor/import/cache symbols across
    `cognition_resolver`, `persona_supervisor2`, `service`, and
    `db_bootstrap`, and `local_context_resolver` found no production RAG2
    initializer, slot supervisor, or initializer-cache maintenance usage.
  - `Test-Path development_plans\active\short_term\rag3_router_interpreter_poc_experiment_plan.md`
    -> `False`.
  - `Get-ChildItem development_plans\active -Recurse -Filter '*rag3*.md' | Select-Object -ExpandProperty Name`
    -> `rag3_local_context_resolver_bigbang_plan.md`.
  - `venv\Scripts\python -m py_compile` over changed production and test
    Python files -> passed.
  - `git -C C:\workspace\kazusa_ai_chatbot diff --check` -> passed with
    line-ending normalization warnings only.
  - `venv\Scripts\python -m pytest tests\test_db_public_boundary.py::test_production_code_outside_db_has_no_raw_mongo_operations tests\test_local_context_resolver_standalone.py::test_stage_json_parser_escapes_control_characters_inside_strings -q`
    -> RAG3 parser test passed; DB boundary still failed only on the
    pre-existing unrelated
    `src\kazusa_ai_chatbot\coding_agent\code_writing\patch_operations.py`
    `.find(` token.
- Production-wired live LLM verification:
  - `venv\Scripts\python -m pytest tests\test_local_context_resolver_live_llm.py -k "production_scoped_memory" -s -m live_llm`
    -> passed; 2 stage calls; 6.56s; prompt-safe summary now reports
    `Local context evidence succeeded with 1 projected rows.`
  - `venv\Scripts\python -m pytest tests\test_local_context_resolver_live_llm.py -k "production_exact_phrase" -s -m live_llm`
    -> passed; 2 stage calls; 9.65s; projected the `Mika` /
    `blue comet marker` conversation evidence row.
  - Existing production live review artifact updated:
    `test_artifacts/local_context_resolver/rag3_production_live_llm_review.md`.
- Full default regression run:
  - `venv\Scripts\python -m pytest -q`
    -> 2844 passed, 2 skipped, 496 deselected, 7 failed.
  - One added full-suite failure,
    `tests/test_coding_agent_image_reading_acceptance.py::test_target_image_reading_question_returns_evidence_backed_answer`,
    passed on isolated rerun:
    `venv\Scripts\python -m pytest tests\test_coding_agent_image_reading_acceptance.py::test_target_image_reading_question_returns_evidence_backed_answer -q`
    -> 1 passed.
  - Persistent failures were outside the RAG3 cutover files and focused suites:
    `tests/test_action_selection_prompt_contract.py` two prompt contract
    checks, `tests/test_cognition_prompt_contract_text.py` one prompt text
    check, `tests/test_db_public_boundary.py` one unrelated coding-agent
    `.find(` boundary token, and two prompt fingerprint baseline tests under
    `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py` and
    `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`.
- Stage 5 sign-off: `Codex/2026-07-04`.

### 2026-07-04 Stage 6 documentation and ICD update

- Updated production architecture documentation:
  - `README.md`
  - `docs/HOWTO.md`
  - `docs/SUBAGENT_INTERFACES.md`
  - `src/kazusa_ai_chatbot/local_context_resolver/README.md`
  - `src/kazusa_ai_chatbot/cognition_resolver/README.md`
  - `src/kazusa_ai_chatbot/rag/README.md`
  - `src/kazusa_ai_chatbot/self_cognition/README.md`
- Documentation now presents RAG3 as production `local_context_recall`, marks
  RAG2 initializer/supervisor behavior as retired from production, removes
  startup initializer-cache hydration from HOWTO, and documents trace-only
  conversation source refs for private past-dialog cognition consumers.
- Stage 6 sign-off: `Codex/2026-07-04`.

### 2026-07-04 Stage 7 independent code review and remediation

- Independent review subagent:
  `019f2be9-3f2f-7d63-a7bf-216229482851` (`Hooke`), review-only, no file
  edits.
- Review result before remediation: not approved.
- Findings addressed:
  - Blocking: RAG3 projection dropped conversation source refs needed by
    past-dialog cognition. Fixed by preserving trace-only refs under
    `supervisor_trace.dispatched[*].source_refs` while stripping row ids from
    prompt-visible evidence.
  - Medium: `cognition_resolver.state` still imported the RAG2
    `project_known_facts` helper to build empty RAG results. Fixed by building
    the retained empty RAG surface directly with `empty_user_memory_context`.
  - Low: local-context retrieval telemetry ignored
    `user_memory_unit_candidates`. Fixed `_retrieval_count` to include scoped
    user-memory candidate rows.
- Post-review verification:
  - Focused review-fix tests -> 8 passed.
  - Local-context production/integration suite -> 25 passed.
  - Cognition/prewarm suite -> 69 passed.
  - Past-dialog cognition source-ref suite -> 15 passed.
  - Event logging/documentation targeted suite -> 9 passed.
  - Production-wired live `production_scoped_memory` and
    `production_exact_phrase` cases reran one at a time and passed.
- Review approval status after remediation: no unresolved RAG3 cutover
  blockers remain. Residual full-suite failures are recorded above as outside
  the RAG3 cutover surface.
- Stage 7 sign-off: `Codex/2026-07-04`.

### 2026-07-04 RAG3 replacement assurance plan

- Keep the replacement gate centered on production ownership:
  `local_context_recall` and shared-memory prewarm must call RAG3 public IO;
  production code must not import RAG2 initializer, slot supervisor,
  dispatcher, evaluator/finalizer loop control, durable initializer-cache
  hydration, or initializer-cache startup maintenance.
- Regression safety set for future changes:
  - static RAG2 production-path greps;
  - local-context contract/graph/projection/standalone/integration suites;
  - cognition resolver, prewarm, past-dialog source-ref, event logging, persona
    supervisor, and service lifecycle suites;
  - one-at-a-time production-wired live LLM cases for current time, exact
    phrase, current-user URL, scoped memory, and `#napcat`;
  - default pytest tracking with unrelated residual failures kept visible, not
    masked by the RAG3 plan.
- E2E confidence boundary:
  focused service and queue tests exercise the production-adjacent workflow,
  while full browser/control-console E2E tests in the default suite also
  passed during the final run. The two opt-in live-service/database E2E tests
  remain skipped by repository markers.

### 2026-07-04 Stage 8 Cache2 integration

- Implemented process-local Cache2 reuse for RAG3 planner and active-node
  stages without caching final RAG packets, dialog text, or raw graph ids.
- Cache policy:
  - Planner entries are long-lived and keyed by prompt/model stage identity,
    capability scope, normalized objective, request source, limits, and coarse
    context shape. Exact live time and exact chat-history text are excluded so
    repeated resolver sequences can reuse the plan when the prompt and
    capability contract have not changed.
  - Active-node entries are exact evidence-stage entries keyed by prompt/model
    identity, node contract, compact context digest, dependency digest, scope,
    and limits.
  - Stable local evidence domains use no TTL and rely on Cache2 dependency
    invalidation, LRU eviction, or process restart. Live-context entries use a
    short TTL. External-evidence entries use a longer but finite TTL.
- Runtime support:
  - Cache2 entries now support optional TTL expiration. Missing TTL means no
    time-based expiry.
  - Cache2 stats include expiration count.
  - Shared cache-key text normalization moved to the Cache2 runtime layer for
    RAG3 and retained helper reuse.
- Production telemetry now marks local-context recall events as cache hits when
  the RAG3 trace summary reports planner or active-node cache reuse.
- Validation:
  - `venv\Scripts\python -m pytest tests\test_cache2_agent_stats.py tests\test_persistent_memory_cache_invalidation.py tests\test_media_descriptor_cache.py tests\test_local_context_resolver_cache.py tests\test_local_context_resolver_standalone.py -q`
    -> 37 passed.
  - `venv\Scripts\python -m pytest tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_graph.py tests\test_local_context_resolver_projection.py tests\test_local_context_resolver_standalone.py tests\test_local_context_resolver_integration.py tests\test_local_context_resolver_cache.py -q`
    -> 28 passed.
  - `venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_persona_graph.py tests\test_cognition_resolver_l2d_contract.py tests\test_shared_memory_prewarm.py tests\test_persona_supervisor2_cognition_prewarm.py tests\test_service_health.py -q`
    -> 70 passed.
  - `venv\Scripts\python -m py_compile` over changed production and test
    Python files -> passed.
  - `git diff --check` -> passed with line-ending normalization warnings only.
- Live LLM cache-performance validation:
  - Added `tests/test_local_context_resolver_cache_live_llm.py`, a one-case
    live harness that clears Cache2, runs the same production local-context
    recall input cold, reruns it warm in the same Python process, and writes
    raw JSON evidence.
  - Ran five production-wired cases individually with `-m live_llm`: current
    time, exact phrase, current-user URL, scoped memory, and `#napcat`.
  - The two prior production sign-off cases, `production_exact_phrase` and
    `production_scoped_memory`, both passed with identical cold/warm
    `rag_result` output.
  - Aggregate cold runtime was 39.539342s across 11 LLM stage calls. Aggregate
    warm runtime was 0.003271s across 0 LLM stage calls. Warm cache-hit
    telemetry was true for all five cases.
  - Human-readable report:
    `test_artifacts/local_context_resolver/rag3_cache2_live_llm_review.md`.

### 2026-07-04 Stage 9 source-backed gap closure and final verification

- Final gap closure scope:
  - RAG3 production active nodes now hydrate source-backed evidence on cache
    misses through the existing source-owned helpers:
    `MemoryEvidenceAgent`, `ConversationEvidenceAgent`,
    `PersonContextAgent`, and `RecallAgent`.
  - `source_hydration.py` projects sanitized `source_context` rows and
    deterministic artifacts into the active-node resolver path. Prompt-visible
    memory rows keep semantic content, memory name, source kind/type/status,
    and scoped-memory provenance only; retrieval scores, cache keys,
    embeddings, raw rows, and platform ids are stripped.
  - `cognition_resolver.capabilities` passes current timestamp, optional
    current platform message id, active-turn platform message ids,
    active-turn conversation row ids, and `source_hydration_enabled=True` to
    RAG3 production calls.
  - Active-node Cache2 keys include the source-hydration enablement flag.
    Warm active-node cache hits skip both source hydration and the active-node
    LLM for that node.
  - Cache invalidation dependencies for scoped memory, recall, and person
    context were broadened to match global user-profile and character-state
    invalidation events.
  - Artifact alias normalization now accepts scoped/user-memory aliases emitted
    by live local LLM runs and rebinds `producer_node_id` to the active node.
  - `tests/test_db_public_boundary.py` now detects nested raw Mongo patterns
    such as `db.collection.find(...)`.
  - The persona RAG helper context builder now tolerates missing
    `platform_message_id` in narrow helper-test states while preserving
    production ids when present.
- Independent review:
  - Review subagent: `019f2c92-324f-74f1-9e30-7a89cb39c5b3` (`Kant`),
    review-only.
  - Review result before remediation: not approved.
  - Findings addressed:
    - Critical: production RAG3 active nodes were source-context-only and had
      no path to DB-backed memory/conversation/person/recall source agents.
      Fixed with static source hydration and source-backed artifacts.
    - High: Cache2 invalidation dependencies for user-global data were too
      channel scoped. Fixed scoped memory, recall, and person-context
      dependency scopes.
    - Low: DB public-boundary tests missed nested raw Mongo calls. Fixed the
      detector and added direct coverage.
  - Review status after remediation: no unresolved RAG3 production blockers.
- One-at-a-time live LLM verification:
  - Standalone RAG3 cases all passed: current time, exact phrase,
    current-user URL, scoped memory, and `#napcat` command anchor.
  - Production RAG3 cases all passed: current time, exact phrase,
    current-user URL, scoped memory, and `#napcat` command anchor.
  - RAG2-vs-RAG3 comparison cases all passed: current time, active agreement
    recall, exact phrase provenance, active-character self words,
    current-user URL recall, scoped current-user memory, named person
    impression, official address memory, and `#napcat` command anchor.
  - Cache2 production live cases all passed: current time, exact phrase,
    current-user URL, scoped memory, and `#napcat` command anchor. Warm runs
    produced identical `rag_result` outputs, zero LLM traces, and cache-hit
    telemetry.
  - Latest `#napcat` cache artifact shows cold source hydration retrieved the
    real durable memory entry named `napcat`; warm run returned in
    0.000705s with planner and active-node cache hits.
- Focused deterministic verification after remediation:
  - Local-context contract/graph/projection/integration/standalone/cache/source
    hydration bundle:
    `venv\Scripts\python -m pytest tests/test_local_context_resolver_contracts.py tests/test_local_context_resolver_graph.py tests/test_local_context_resolver_projection.py tests/test_local_context_resolver_integration.py tests/test_local_context_resolver_standalone.py tests/test_local_context_resolver_cache.py tests/test_local_context_resolver_source_hydration.py -q`
    -> 31 passed.
  - Cognition, prewarm, and event boundary bundle:
    `venv\Scripts\python -m pytest tests/test_cognition_resolver_l2d_contract.py tests/test_cognition_resolver_contracts.py tests/test_cognition_resolver_loop.py tests/test_persona_supervisor2_cognition_prewarm.py tests/test_persona_supervisor2_rag_skip_shape.py tests/test_rag_dialog_event_logging.py -q`
    -> 72 passed.
  - Cache2 focused bundle:
    `venv\Scripts\python -m pytest tests/test_cache2_agent_stats.py tests/test_db_writer_cache2_invalidation.py tests/test_rag_cache2_persistent.py -q`
    -> 13 passed.
  - Former full-suite failures were isolated and resolved or classified:
    coding-agent image-reading acceptance passed in isolation and in the
    final exact failed-set rerun; RAG helper integration failures were fixed
    by the optional message-id boundary change and snapshot update.
  - Exact failed-set rerun:
    `venv\Scripts\python -m pytest tests/test_coding_agent_image_reading_acceptance.py::test_target_image_reading_question_returns_evidence_backed_answer tests/test_persona_supervisor2_rag2_integration.py::test_rag_evidence_helper_calls_rag3_and_projects_payload tests/test_persona_supervisor2_rag2_integration.py::test_rag_evidence_request_shape_snapshot tests/test_persona_supervisor2_rag2_integration.py::test_rag_evidence_passes_empty_reply_context_to_wrapper tests/test_persona_supervisor2_rag2_integration.py::test_rag_evidence_runs_for_mixed_referents -q`
    -> 5 passed.
- Final deterministic regression:
  - `venv\Scripts\python -m pytest -m "not live_db and not live_llm and not live_internet" -q`
    -> 2860 passed, 2 skipped, 501 deselected.
- Static verification:
  - `venv\Scripts\python -m py_compile` over changed and untracked Python
    files -> passed.
  - `git diff --check` -> passed with line-ending normalization warnings only.
  - Production-path grep for RAG2 initializer/supervisor/refined-query symbols
    across cognition resolver, persona supervisor, service, DB bootstrap, and
    local-context resolver returned no matches.
- Documentation closeout:
  - `src/kazusa_ai_chatbot/local_context_resolver/README.md` now documents the
    implemented static source-hydration bridge, optional source-hydration
    context fields, cache behavior, and distinction from future dynamic
    subagent protocols.
  - `src/kazusa_ai_chatbot/rag/README.md` continues to identify RAG2 as
    retired from production while documenting retained helper/source-cache
    ownership.
- Residual risk:
  - No known RAG3 cutover blockers remain.
  - Opt-in live DB/service E2E tests remain skipped by repository markers in
    the default non-live suite.
  - RAG2 helper modules remain in the repository for retained source helpers,
    historical tests, and comparison coverage, but production
    `local_context_recall` no longer uses the RAG2 initializer/supervisor
    path.
- Final plan status: completed and ready for archive.
