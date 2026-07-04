# Local Context Resolver ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.local_context_resolver`
- Runtime role: production local/private context evidence resolver, also known
  as RAG3
- Interface boundary: stable public IO around
  `resolve_local_context(request, context, options=None)`
- Current integration status: production `local_context_recall` and
  shared-memory prewarm are wired through the stable public IO
- Production caller status: cognition resolver capability execution and
  first-cycle shared-memory prewarm call `resolve_local_context(...)`; dialog,
  adapters, and persistence consume only the retained projected evidence
- Source evidence: `contracts.py`, `service.py`, `stages.py`, `graph.py`,
  `constants.py`, and the focused `tests/test_local_context_resolver_*.py`
  suites

This README is the module ICD for implemented RAG3 behavior. If a
development plan, report, or historical artifact disagrees with this file,
read the code and tests first, then update this ICD to match implemented
behavior.

## Purpose

The local-context resolver turns one bounded local-context objective into a
prompt-safe evidence packet. It is aligned with the complex-resolver shape:

```text
objective
  -> bounded semantic graph
  -> one active evidence node at a time
  -> source-owned artifacts
  -> known/lacking/boundary packet
  -> retained rag_result projection
```

The resolver returns evidence. It does not decide whether Kazusa should speak,
what stance she should take, or how final visible text should be worded.
Cognition and dialog keep those responsibilities.

## Ownership Boundary

This package owns:

- public request, context, options, graph, artifact, packet, and future
  subagent contracts;
- structural validation for all public IO and graph objects;
- graph planning, deterministic graph construction, bounded traversal, and
  dependency-safe active-node selection;
- active-node semantic evidence extraction from prompt-safe supplied context;
- optional duplicate-node collapse review;
- deterministic or LLM-backed bottom-up packet synthesis;
- prompt-safe `rag_result` projection and metadata redaction;
- live review trace records and raw efficiency counters.

This package does not own:

- platform adapter syntax, mention parsing, delivery receipts, or rendering;
- MongoDB query construction, embeddings, cache invalidation, or persistence
  writes;
- cognition stance, action selection, response-gating judgment, or final
  dialog wording;
- consolidation, scheduler, reflection, or accepted-task lifecycle;
- cognition stance, dialog wording, adapter delivery, or persistence writes
  after projected evidence has been returned.

## Public Interfaces

Callers use one public resolver entrypoint:

```python
await resolve_local_context(request, context, options=None)
```

Stable public input contracts:

- `LocalContextResolverRequestV1`
- `LocalContextResolverContextV1`
- `LocalContextResolverOptionsV1`

Stable public output contract:

- `LocalContextResolutionPacketV1`

Projection entrypoint:

```python
project_local_context_packet(packet)
```

`project_local_context_packet(...)` returns only `packet["rag_result"]`.
`graph` and `trace_summary` are debug and supervisor material, not prompt-facing
evidence.

The package also exports validators for each public contract:

- `validate_local_context_resolver_request`
- `validate_local_context_resolver_context`
- `validate_local_context_resolver_options`
- `validate_local_context_node`
- `validate_local_context_graph`
- `validate_local_context_artifact`
- `validate_local_context_resolution_packet`
- `validate_local_context_subagent_request`
- `validate_local_context_subagent_result`

## Architecture

The implemented resolver has a direct standalone interface and four
resolver-local LLM stages. These are stage agents inside `stages.py`, not a
package-discovered source-subagent registry. `LocalContextSubagentV1` is a
typed future handler protocol; no current service path dispatches to dynamic
subagent modules.

```mermaid
flowchart TD
    C0["Caller<br/>l2d local_context_recall, prewarm,<br/>standalone eval, live LLM review"]
    C1["resolve_local_context(...)<br/>public stable IO"]
    C2["contract validation<br/>request/context/options"]
    C3["blocked packet builder<br/>input or local-resolution failure"]
    C4["LocalContextResolutionPacketV1<br/>known/lacking/boundary + rag_result + graph + trace_summary"]
    C5["project_local_context_packet(...)<br/>prompt-facing rag_result only"]

    C0 --> C1 --> C2
    C2 -->|invalid input| C3 --> C4
    C2 -->|valid input| P0
    S8 --> C4 --> C5

    subgraph Planner["Graph planner stage"]
        P0["compact prompt-safe context<br/>strip raw ids, timestamps, embeddings, trace data"]
        P1["Graph Planner LLM<br/>plan_local_context_graph(...)<br/>RAG_PLANNER_LLM"]
        P2["planner task validator<br/>1..max_nodes-1 semantic tasks<br/>alias node_kind vocabulary"]
        P3["deterministic graph builder<br/>root synthesis node + task_N evidence nodes"]
        P0 --> P1 --> P2 --> P3
    end

    subgraph Traversal["Bounded graph traversal"]
        T0["find_next_active_node(...)<br/>child order plus dependency status"]
        T1{"pending node exists<br/>and iteration cap remains?"}
        T2["active_node_id selected"]
        T3["Active Node Resolver LLM<br/>resolve_local_context_node(...)<br/>RAG_SUBAGENT_LLM"]
        T4["node_update applier<br/>status and semantic rows"]
        T5["artifact validator<br/>bind active_node alias<br/>normalize artifact_type aliases"]
        T6{"same-kind resolved<br/>collapse candidates?"}
        T7["Collapse Review LLM<br/>review_local_context_collapse(...)<br/>RAG_SUBAGENT_LLM"]
        T8["deterministic collapse applier<br/>candidate_ref -> internal node id"]
        T9["trace counters refresh<br/>iterations, node counts, collapse counts"]
        P3 --> T0 --> T1
        T1 -->|yes| T2 --> T3 --> T4 --> T5 --> T6
        T6 -->|no| T9 --> T0
        T6 -->|yes| T7 --> T8 --> T9 --> T0
        T1 -->|no| S0
    end

    subgraph Synthesis["Packet synthesis and projection"]
        S0{"all unresolved nodes absent<br/>and node rows/artifacts exist?"}
        S1["deterministic bottom-up synthesis<br/>reuse node-owned rows<br/>0 extra LLM calls"]
        S2["Bottom-Up Synthesis LLM<br/>synthesize_local_context_packet(...)<br/>RAG_SUBAGENT_LLM"]
        S3["semantic synthesis normalizer<br/>fallback to node rows when fields are empty"]
        S4["rag_result base<br/>RAG2-compatible retained surface"]
        S5["artifact projection merge<br/>source-owned payloads only"]
        S6["projection normalization<br/>scoped memory, live context,<br/>conversation/recall/external placement"]
        S7["prompt sanitizer<br/>strip raw ids, timestamps, metadata, cache keys"]
        S8["packet validator<br/>LocalContextResolutionPacketV1"]
        S0 -->|yes| S1 --> S4
        S0 -->|no| S2 --> S3 --> S4
        S4 --> S5 --> S6 --> S7 --> S8
    end

    subgraph Trace["Review-only trace material"]
        X0["stage trace records<br/>input payload, raw model output,<br/>parsed output or parse_error"]
        X1["trace_summary counters<br/>planner, active-node, collapse,<br/>synthesis, subagent calls"]
    end

    P1 -.records.-> X0
    T3 -.records.-> X0
    T7 -.records.-> X0
    S2 -.records.-> X0
    T9 -.updates.-> X1
    S8 -.includes.-> X1
```

The normal optimized path for a single-source resolved objective is:

```text
planner LLM
  -> active-node LLM
  -> deterministic synthesis and projection
```

That path is two LLM calls. Collapse review runs only when a resolved
same-kind candidate exists. Bottom-up synthesis LLM runs only when deterministic
node-row synthesis is not sufficient, such as unresolved or blocked graph
state.

## Resolver-Local Stage Agents

| Stage agent | Function | Route | Input | Output | Validation owner | Side effects |
|---|---|---|---|---|---|---|
| Graph planner | `plan_local_context_graph(payload)` | `RAG_PLANNER_LLM` | Request, compact context, option limits | JSON `tasks` with objective and node kind | `_planner_tasks`, `_graph_from_planner_response`, graph validators | None beyond stage trace capture |
| Active node resolver | `resolve_local_context_node(payload)` | `RAG_SUBAGENT_LLM` | Request, compact context, active node, dependency context, limits | `node_update` plus source-owned artifacts | `_apply_active_node_response`, `_validated_artifact_for_node`, artifact validators | None beyond stage trace capture |
| Collapse reviewer | `review_local_context_collapse(payload)` | `RAG_SUBAGENT_LLM` | Active node and prompt-safe same-kind candidates | `collapse_decision` with `target_candidate_ref` | `_apply_collapse_response` maps prompt ref to internal node id | None beyond stage trace capture |
| Bottom-up synthesizer | `synthesize_local_context_packet(payload)` | `RAG_SUBAGENT_LLM` | Resolved/unresolved node summaries and limits | Packet semantic row fields | `_semantic_synthesis_response`, packet validators | None beyond stage trace capture |

All stage agents use deterministic JSON parsing without a JSON-repair LLM.
Raw control characters inside JSON strings are escaped deterministically after
normal parsing fails. If parsing still fails, the stage records a failed trace
row with raw model output and raises a bounded validation error.

## Future Subagent Protocol

`contracts.py` defines `LocalContextSubagentV1`,
`LocalContextSubagentRequestV1`, and `LocalContextSubagentResultV1` for future
source-owned handlers. This is a resolver-local protocol, not a shared
subagent abstraction and not a current registry.

Current contract fields:

| Category | Current contract |
|---|---|
| Family name | Local-context resolver source handler |
| Owning package | `kazusa_ai_chatbot.local_context_resolver` |
| Runtime purpose | Future bounded source-owned retrieval for one local-context node |
| Registry or discovery | None implemented today |
| Identifier | `subagent` in `LocalContextSubagentRequestV1` |
| Supported actions | `action` string, validated as a non-empty semantic action |
| Input contract | `node_id`, `subagent`, `action`, `objective`, `payload`, `constraints` |
| Output contract | `resolved`, `status`, `result`, `attempts`, `cache`, `trace`, `unresolved_items` |
| Validation owner | `validate_local_context_subagent_request` and `validate_local_context_subagent_result` |
| Enablement | None implemented today |
| Cache behavior | Result envelope has `cache`; no current source handler uses it |
| Trace or audit | Result envelope has bounded `trace`; service `subagent_calls` counter exists |
| Side-effect boundary | Future source handlers may return evidence only; no stance, dialog, adapter delivery, arbitrary persistence, or shell/tool execution |
| Required tests | Contract tests plus source-handler tests before any implementation is wired |

Until a source-handler registry is implemented, diagrams and callers must not
claim that RAG3 dispatches to concrete conversation, memory, person, recall,
live, or web subagent modules. Today, active-node LLM resolution reads supplied
prompt-safe context and emits source-owned artifacts.

## Input And Output Contracts

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
    "original_user_request": str,  # optional
}
```

The caller may carry platform and user ids in the public context envelope for
validation and future integration, but `_compact_context(...)` strips
prompt-unsafe ids, raw timestamps, storage rows, embeddings, cache keys, and
trace fields before any LLM stage sees the context.

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

Default limits:

```python
{
    "max_iterations": 3,
    "max_nodes": 8,
    "max_depth": 3,
    "max_node_attempts": 2,
    "max_subagent_attempts": 1,
}
```

Hard caps:

```python
{
    "max_iterations": 4,
    "max_nodes": 8,
    "max_depth": 3,
    "max_node_attempts": 2,
    "max_subagent_attempts": 1,
}
```

Behavior injection fields such as `planner_llm`, `node_resolver_llm`,
`collapse_llm`, `synthesizer_llm`, `subagents`, and `clock` are rejected.

### `LocalContextNodeV1`

Allowed node kinds:

- `conversation_evidence`
- `external_evidence`
- `live_context`
- `memory_evidence`
- `person_context`
- `recall_evidence`
- `scoped_memory`
- `subtask`
- `synthesis`

Allowed statuses:

- `pending`
- `resolving`
- `resolved`
- `blocked`
- `cannot_answer`
- `collapsed`

Node ids are deterministic service-owned ids such as `root` and `task_1`.
LLM-facing payloads receive compact semantic node projections, not raw graph
internals.

### `LocalContextArtifactV1`

Allowed artifact types:

- `conversation_ref`
- `external_ref`
- `live_context_ref`
- `memory_ref`
- `person_ref`
- `recall_ref`
- `semantic_packet`

Artifacts are source-owned evidence. `producer_node_id` is bound
deterministically; the active-node alias `active_node` is allowed in LLM output
and mapped to the internal active node id before validation.

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
    "graph": LocalContextGraphV1,
    "trace_summary": dict,
}
```

`rag_result` preserves the retained prompt-facing RAG2-compatible surface:

```python
{
    "answer": str,
    "user_image": dict,
    "user_memory_unit_candidates": list,
    "character_image": dict,
    "third_party_profiles": list,
    "memory_evidence": list,
    "recall_evidence": list,
    "conversation_evidence": list,
    "external_evidence": list,
    "supervisor_trace": dict,
}
```

## Runtime Flow

1. `resolve_local_context(...)` validates request, context, and options. Invalid
   input returns a bounded blocked packet instead of raising to callers.
2. `_plan_graph(...)` calls the graph planner LLM and maps semantic task rows
   into a strict `LocalContextGraphV1`.
3. `_run_graph_traversal(...)` repeatedly selects the next dependency-ready
   pending node through `find_next_active_node(...)`, resolves it, applies
   artifacts, and optionally reviews same-kind collapse candidates.
4. `_synthesize_packet(...)` uses deterministic synthesis when resolved
   node-owned rows are sufficient. Otherwise it calls the bottom-up synthesis
   LLM and normalizes the result.
5. `_rag_result_from_artifacts(...)` builds a fresh retained `rag_result`,
   merges prompt-visible artifacts, normalizes source fields, projects
   structured live context, sanitizes prompt-facing rows, and deduplicates
   repeated payload items.
6. `validate_local_context_resolution_packet(...)` validates the final packet
   before returning it.

## Projection Rules

Projection is source-owned and prompt-safe.

- `memory_ref` writes durable/shared memory to `memory_evidence`.
- `memory_ref` rows sourced from `user_memory_units` or user-scoped memory move
  to `user_memory_unit_candidates`.
- `conversation_ref` writes chat messages, speakers, exact phrases, URL
  provenance, direct-address anchors, reply context, and nearby dialog to
  `conversation_evidence`.
- `conversation_ref` artifacts cannot populate `recall_evidence`.
- Raw `conversation_ref` projection payloads may carry trace-only source refs
  such as `conversation_row_id` or `_id`. These refs are copied only to
  `rag_result["supervisor_trace"]["dispatched"][*]["source_refs"]` for private
  past-dialog cognition consumers, then stripped from prompt-visible evidence.
- `recall_ref` writes active agreements, commitments, plans, and episode state
  to `recall_evidence`; when recall evidence is present, duplicate
  `conversation_evidence` rows from that same artifact are dropped.
- `person_ref` writes profile, identity, relationship, or impression evidence
  to `third_party_profiles`, `user_image`, or `character_image`.
- `external_ref` writes supplied public URL or web-content evidence to
  `external_evidence`.
- `live_context_ref` has no dedicated retained top-level field; structured
  live context is projected into `conversation_evidence` with source
  `live_context`.

The sanitizer strips forbidden keys and embedded metadata such as raw message
ids, adapter ids, platform ids, conversation row ids, database ids, raw UTC
timestamps, cache keys, trace ids, embeddings, and local storage timestamps.

## Configuration

LLM stages use route-specific configuration through `LLInterface`:

- Graph planner: `RAG_PLANNER_LLM`
- Active-node resolver: `RAG_SUBAGENT_LLM`
- Collapse reviewer: `RAG_SUBAGENT_LLM`
- Bottom-up synthesizer: `RAG_SUBAGENT_LLM`

Stage generation defaults in `constants.py`:

- `STAGE_LLM_TEMPERATURE = 0.1`
- `STAGE_LLM_TOP_P = 0.7`

The package does not read environment variables directly; route configuration
is imported through `kazusa_ai_chatbot.config`.

## Persistence And Side Effects

The standalone resolver has no direct persistence writes. It does not upsert
memory, update conversation history, write scheduler rows, send adapter
messages, mutate Cache2, or execute external tools.

Stage trace records are process-local review material returned by
`drain_stage_trace_records()` in `stages.py`. They are used by live LLM review
tests and are not part of prompt-facing `rag_result`.

## Failure Behavior

- Public input validation failure returns a bounded blocked packet with
  `failure_stage = "input_validation"` in `trace_summary`.
- Local graph, stage, artifact, or packet validation failure returns a bounded
  blocked packet with `failure_stage = "local_resolution"`.
- Bounded blocked packets contain safe missing-knowledge rows and an empty
  evidence surface except `supervisor_trace`.
- LLM JSON parsing uses deterministic parsing only. It may extract the outer
  JSON object and escape raw control characters inside JSON strings, but it
  does not call `JSON_REPAIR_LLM`.
- Failed stage parsing records raw output and a `parse_error` in stage trace
  records before raising a validation error.
- Collapse review is skipped deterministically when no same-kind resolved
  candidate exists.
- Collapse application is ignored unless the active node is resolved and the
  model-selected `target_candidate_ref` maps to a valid same-kind candidate.

## Observability

`trace_summary` includes raw counters needed for review:

- `iterations`
- `node_count`
- `max_depth_observed`
- `resolved_node_count`
- `blocked_node_count`
- `planner_calls`
- `active_node_calls`
- `collapse_calls`
- `synthesis_calls`
- `subagent_calls`
- `collapse_count`

Stage trace records include:

- stage name and prompt id;
- route name and model;
- model-facing input payload;
- raw model output;
- parsed output, or `parse_error` when deterministic parsing fails.

`trace_summary` and stage traces are diagnostic material. They must not become
cognition evidence or dialog input.

## Testing Contract

Use `venv\Scripts\python.exe`.

Deterministic package tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_local_context_resolver_standalone.py tests\test_local_context_resolver_contracts.py tests\test_local_context_resolver_graph.py tests\test_local_context_resolver_projection.py -q
```

Standalone live LLM review tests must be run one case at a time with
`-m live_llm`. Current live review files include:

- `tests/test_local_context_resolver_live_llm.py`
- `tests/test_local_context_resolver_rag2_vs_rag3_live_llm.py`
- `tests/test_local_context_resolver_full_matrix_live_llm.py`

The full-matrix live harness covers the current RAG2 behavior matrix plus
additional RAG3 cases for real group-history adjacency, `#napcat`, scoped user
memory, URL provenance, external content, topic participants, reply-parent
context, and cascaded phrase/person/link dependency.

## Forbidden Paths

- Do not add a second production recall path, fallback to the retired RAG2
  supervisor, or dual-run compatibility bridge around this package.
- Do not add an experiment-only request, context, options, packet, or adapter
  wrapper shape.
- Do not compile RAG3 graphs back into RAG2 `unknown_slots`.
- Do not introduce compatibility shims, alias modules, fallback mappers, or
  dual production paths to preserve RAG2 supervisor internals.
- Do not expose graph ids, raw stage traces, raw database ids, adapter ids,
  platform ids, cache keys, embeddings, prompt text, raw wire syntax, or raw
  timestamps in `rag_result`.
- Do not ask LLM stages to generate MongoDB filters, index names, embedding
  settings, adapter delivery behavior, persistence decisions, or final visible
  dialog.
- Do not use JSON repair LLM calls or unbounded retry loops for malformed
  local-context stage output.
- Do not treat retrieved evidence as persona stance, character judgment, or
  final user-visible wording.
- Do not claim dynamic source-subagent registry behavior until concrete
  source-handler modules, discovery, dispatch, and tests exist.

## Change Control

Any change to public contracts, node kinds, artifact types, prompt-facing
projection, LLM stage prompts, traversal caps, trace fields, or production
wiring must update this ICD in the same change.

Prompt changes require focused deterministic tests and selected one-case-at-a-
time live LLM review. Production wiring changes must preserve the public IO
described here and keep `rag_result` as the only prompt-facing evidence
surface.
