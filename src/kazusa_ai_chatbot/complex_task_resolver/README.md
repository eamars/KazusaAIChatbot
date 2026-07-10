# Complex Task Resolver ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.complex_task_resolver`
- Runtime role: public answer research specialist
- L2d capability name: `public_answer_research`
- Companion L2d recall capability: `local_context_recall`
- Integration status: standalone module is implemented; L2d contract
  integration must route only through declared public IO and remains gated by
  a future explicit rollout plan and user command before broad live enablement.
- Related docs: [Cognition Resolver ICD](../cognition_resolver/README.md),
  [RAG 2](../rag/README.md), [web_agent3 ICD](../rag/web_agent3/README.md),
  [Action Spec](../action_spec/README.md)

This document is the module-level integration contract. Tests, review harnesses,
and future workflow integration must use this boundary rather than reaching into
implementation details.

## Purpose

The complex task resolver investigates answer gaps that are too broad for one
ordinary L2d evidence request or direct visible reply. It decomposes a user goal
into a bounded task graph, resolves one active node at a time, invokes
resolver-local specialist subagents when needed, reviews duplicate or
equivalent branches, and synthesizes a prompt-safe semantic knowledge packet.

The module returns structured factual review output. It does not decide whether
the original user goal is answered, partial, blocked, or worth retrying. L2d,
cognition, and L3/dialog remain responsible for semantic judgment, whether to
speak, and how visible text should be rendered.

## L2d Boundary

The agreed L2d-visible resolver capability for this module is:

```text
public_answer_research
```

L2d should use `public_answer_research` when the original user goal needs
public, current, external, or source-bound answer investigation, especially
when several dependent steps are required before a reliable answer can be
selected. Examples include public evidence retrieval, deterministic
calculation over sourced numbers, comparison, feasibility review, conflict
review, and synthesis.

The companion L2d capability is:

```text
local_context_recall
```

`local_context_recall` is not owned by this package. It routes to RAG3 through
`resolve_local_context(...)` and owns local memory,
relationship, profile, prior conversation, and private/contextual recall.

L2d should not use `public_answer_research` for:

- a single missing fact that the current cognition cycle can already answer;
- local memory, relationship, profile, or conversation recall, which belongs
  to `local_context_recall`;
- user-owned missing input, which belongs to `human_clarification`;
- approval before a side effect, which belongs to `approval_preparation`;
- private internal-source goal handling, which belongs to
  `self_goal_resolution`;
- final visible wording, which belongs to `speak` and L3/dialog.

When this boundary is enabled, the former L2d-visible `web_evidence` capability
is collapsed into `public_answer_research`; WebAgent3 remains an internal
evidence provider under the resolver. The former L2d-visible `rag_evidence`
capability is represented by `local_context_recall`; the underlying RAG2
package and entrypoint remain unchanged.

Do not add `answer_investigation`, `web_evidence`, or `rag_evidence` as
canonical runtime aliases for this boundary. The capability cutover is intended
to be a single vocabulary change across caller, callee, tests, and ICD.

Broad live enablement is still gated. Do not route dialog, adapters,
scheduler, coding-agent execution, filesystem work, shell work, database
writes, or raw graph traces through this module.

## Public Module IO

Standalone callers use the public resolver entrypoint:

```python
await resolve_complex_task(request, context, options=None)
```

The stable public inputs are:

- `ComplexTaskResolverRequestV1`: root objective, source, reason, and priority.
  The declared source values are `test`, `review_harness`,
  `live_llm_review`, and `l2d`.
- `ComplexTaskResolverContextV1`: compact prompt-safe context, time context,
  and review evidence provided by the caller.
- optional `ComplexTaskResolverOptionsV1`: structural limits only, such as
  maximum graph iterations, nodes, depth, node attempts, and subagent attempts.

The stable output is:

- `ComplexTaskResolutionPacketV1`: prompt-safe investigation summary,
  `knowledge_we_know_so_far`, `knowledge_still_lacking`,
  `recommended_next_iteration`, `evidence_boundary_notes`, graph state, and
  trace summary.

Real LLM tests and review harnesses must call the module only through this
public IO. They may read returned graph and trace data for debugging, but they
must not configure internal LLM stage invokers, node routing, prompt variants,
subagent rosters, expected graph paths, or expected answers from outside the
module. Test-only monkeypatching of internal stage handlers is allowed for
deterministic plumbing tests, but it is not part of public module IO and must
not be used by real LLM review cases.

## Internal Ownership

The module owns:

- graph planning and validation;
- resolver-local LLM stage prompts, model calls, JSON parsing, and validation;
- active-node traversal and recursion;
- bounded active-node resolution attempts and prompt-safe attempt observations;
- node expansion under bounded depth and node caps;
- structured executable follow-up task creation under bounded graph limits;
- graph traversal order and traversal recording;
- semantic collapse review over bounded candidates;
- same-run duplicate handling through bounded semantic collapse;
- lower-layer cache metadata propagation through subagent result envelopes;
- deterministic arithmetic through a typed algorithmic subagent;
- evidence subagent wrapping for resolver-local evidence needs;
- bottom-up synthesis into semantic sections on
  `ComplexTaskResolutionPacketV1`;
- prompt-safe projection and read-only trace material.

The deterministic arithmetic subagent evaluates only caller-prepared numeric
expressions. The node resolver owns unit interpretation, unit conversion,
operation selection, and expression construction before invoking the
algorithmic subagent.

Every internal `evaluate_expression` request must declare operand provenance
in its payload:

```json
{
  "expression": "(51.6 - 38.2) / 38.2 * 100",
  "label": "performance_delta_percent",
  "input_values": [
    {
      "label": "candidate_a_tps",
      "value": "51.6",
      "source_node_id": "task_1",
      "source_text": "candidate A measured 51.6 tokens/sec"
    },
    {
      "label": "candidate_b_tps",
      "value": "38.2",
      "source_node_id": "task_2",
      "source_text": "candidate B measured 38.2 tokens/sec"
    }
  ],
  "formula_constants": [
    {
      "value": "100",
      "purpose": "percentage conversion"
    }
  ]
}
```

The service validates this structurally before calling the calculator: every
numeric literal in the expression must be declared as either an `input_values`
operand or a `formula_constants` entry, each input value must quote an existing
graph node projection, and the quoted text must include the declared value.
This does not judge whether the source is true; it prevents hidden arithmetic
operands from entering deterministic calculation. If a required operand is not
present in the graph, the node must stay blocked or lacking instead of
calculating from an assumption.

The LLM-facing arithmetic decision does not emit graph identifiers. It quotes
the source text visible in the prompt projection. The service deterministically
matches that quoted source text back to the graph node and fills the internal
provenance field before validating the subagent request.

If an algorithmic node is first answered with prose, the resolver may perform
one bounded repair call to request the declared subagent IO. It must still fail
closed if the repair does not produce a valid `evaluate_expression` request.

Each selected active node may run a bounded local resolution loop before the
resolver advances to another node. The loop records compact observations such
as the attempted action, result summary, blockers, and recommended next local
action. Those observations are projected only into later resolver-stage prompts
for the same node and into read-only review traces. They are not external
configuration, fixture hints, final dialog, or cognition-visible state.

Resolver stages may also emit structured executable follow-up tasks when the
resolver itself can address the missing work. This is the only executable
follow-up channel:

```json
{
  "schema_version": "complex_task_followup_task.v1",
  "objective": "bounded executable task",
  "kind": "subtask|evidence_need|algorithmic_task|synthesis",
  "reason": "why this task is needed"
}
```

Active-node follow-up tasks become child nodes under the source node.
Bottom-up synthesis follow-up tasks become bounded root-level child nodes and
trigger another traversal pass before final packet return when graph limits
allow it. All follow-up creation obeys structural resolver limits, including
maximum nodes, maximum depth, maximum traversal iterations, and per-source
follow-up caps. Rejected follow-ups are preserved as lacking knowledge and
evidence-boundary notes.

The resolver must never parse or execute prose from
`recommended_next_iteration`. Prose recommendations remain semantic projection
for cognition and review only. If work should execute inside the resolver, the
LLM stage must emit `followup_tasks`.

The graph traversal loop, node-resolution loop, and subagent loops have separate
ownership:

- graph traversal chooses one pending node at a time;
- node resolution decides whether to update, expand, invoke a subagent, retry
  locally, ask for user-owned input, or block;
- public evidence node resolution expands narrower children first when one
  node still combines independent public targets or fact dimensions;
- evidence subagents may perform bounded search/read refinement when an
  evidence backend is available, and otherwise report dependency blockers;
- the algorithmic subagent is deterministic single-shot and only evaluates the
  caller-prepared expression after the resolver has handled units and
  conversion.

The module does not own:

- L2d semantic action selection;
- `ResolverCycleStateV1` recurrence;
- pending HIL or approval rows;
- final visible dialog wording;
- adapter delivery or delivery receipts;
- scheduler actions;
- database writes;
- arbitrary shell, filesystem, notebook, package, or generic MCP execution.
- retrieval-result caching, cache keys, cache invalidation, or freshness
  storage; those belong to RAG, web/source agents, or their backing stores.

## Subagent Boundary

Resolver-local subagents are internal module implementation. External callers
must not provide a subagent registry to change behavior for real LLM tests or
future L2d execution.

Internal subagents follow the existing Kazusa helper-agent shape in spirit:

```python
async def run(
    task: ComplexTaskSubagentRequestV1,
    context: dict[str, object],
    max_attempts: int = DEFAULT_SUBAGENT_MAX_ATTEMPTS,
) -> ComplexTaskSubagentResultV1:
    ...
```

Each subagent must return a bounded result envelope and refuse out-of-domain
work. Deterministic code validates every request and result before graph state
is updated.

Resolver-local subagents are registered through module-owned discovery under
`complex_task_resolver.subagent`. Each registration declares the semantic
capability description, supported actions, owned node kinds, default action,
and factory. LLM prompts receive the generated semantic capability list; the
service owns capability validation, default-action selection, and dispatch.

The `media` subagent owns `media_inspection_task` nodes. It fetches only
public HTTP(S) image URLs through deterministic DNS/IP, redirect, timeout,
MIME, byte, magic-byte, and decoder/dimension validation, then calls the
shared image-only media-inspection service. It returns bounded visual evidence
and source boundaries only; raw bytes and fetch internals do not enter graph
prompts or `available_evidence`.

## LLM Stage Projection

Public resolver IO and internal graph/subagent contracts remain typed and
versioned. LLM stages do not author those deterministic envelopes.

Prompt inputs are semantic projections only: active node objective, work type,
semantic knowledge rows, recent attempt summaries, compact context, and
semantic evidence excerpts. Prompt inputs do not include graph bookkeeping,
raw trace summaries, cache metadata, or schema fields.

Prompt outputs are also semantic. The active-node stage emits semantic
decisions such as expansion, knowledge recording, subagent use, or local
continuation. The synthesis stage emits semantic knowledge sections and
optional continuation tasks. The collapse stage identifies a duplicate by
candidate semantic text. Deterministic code then maps those semantic outputs
into graph nodes, follow-up task contracts, subagent request envelopes,
collapse events, traversal state, and read-only traces.

If a semantic LLM output contains deterministic transport keys such as schema
markers, graph identifiers, attempt counters, operational state fields, trace
fields, or cache fields, the resolver rejects that output before using it.

## Semantic Knowledge Contract

The resolver must finish each standalone run with a semantic knowledge packet:

- `investigation_summary`: compact summary of what the resolver investigated;
- `knowledge_we_know_so_far`: evidence-backed or explicitly bounded knowledge
  gathered so far;
- `knowledge_still_lacking`: unresolved facts, missing source coverage,
  unavailable inputs, or structural blockers;
- `recommended_next_iteration`: evidence directions the next cognition cycle
  may consider, not commands to keep searching;
- `evidence_boundary_notes`: source, traceability, tool, or availability
  limits that cognition must preserve when deciding what to do next.

The semantic sections are synthesized bottom-up from traversed node results.
The module must preserve unresolved branches and blockers rather than silently
flattening them into confident facts.

Each `ComplexTaskNodeV1` carries the same local semantic projection shape:

- `investigation_summary`
- `knowledge_we_know_so_far`
- `knowledge_still_lacking`
- `recommended_next_iteration`
- `evidence_boundary_notes`

Node `status` remains operational traversal state only. A resolved node means
the assigned node step completed; it does not mean the original user question
is fully answered. Later resolver LLM stages consume these node semantic
projection fields rather than answer-shaped node text.

## Testing Contract

Deterministic tests may patch LLM stage outputs to verify graph plumbing,
validation, traversal, subagent-result merging, collapse handling,
semantic knowledge projection, lower-layer cache metadata passthrough, and
prompt-safe projection.

Real LLM tests must:

- run one case at a time;
- use only declared public module IO;
- emit durable artifacts with prompt inputs, parsed outputs, graph state,
  subagent traces, final packet, and human or AI review notes;
- inspect behavior qualitatively, not only pytest status;
- avoid deterministic keyword or fixture hints in production code and prompts;
- avoid externally configuring internal subagents, prompt variants, graph
  paths, expected node names, or expected answers.

Case 32 and later recursive/subagent cases must prove that the module itself
can create deeper task graphs and invoke the appropriate internal subagents.
A test that passes only because the harness injected the correct subagent
roster, prompt variant, or graph path is not a valid architectural pass.
