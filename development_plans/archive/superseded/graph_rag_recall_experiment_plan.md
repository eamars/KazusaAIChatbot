# graph_rag_recall_experiment_plan

## Lifecycle

- Status: superseded.
- Superseded on 2026-05-23 by the current RAG2 recall-quality production
  direction in
  `development_plans/active/short_term/rag2_cognition_ready_evidence_plan.md`.
- This document is historical context only. Do not execute it unless a future
  plan explicitly revives graph RAG work.

## Summary

- Goal: Build an isolated experiment that tests whether a GraphRAG retrieval path improves memory and conversation-history recall accuracy over the current Recall, conversation-evidence, and memory-evidence workers.
- Plan class: large
- Status: superseded
- Mandatory skills: `development-plan`, `local-llm-architecture`, `database-data-pull`, `debug-llm`, `py-style`, `cjk-safety`
- Overall cutover strategy: compatible experiment only; no production runtime cutover in this plan.
- Highest-risk areas: LLM extraction quality, temporal fact validity, source provenance, group-chat entity resolution, shared/world memory authority, misleading graph traversal, and false confidence from non-human-readable command output.
- Acceptance criteria: The experiment produces real-data baseline-vs-GraphRAG evidence, human-readable quality review, and a go/no-go recommendation for a later production plan.

## Context

Kazusa's current RAG2 path already separates the initializer, dispatcher,
specialist agents, deterministic execution, evaluator, and finalizer. The
observed product risk is narrower: the character does not reliably recall
conversation history, active agreements, and scoped memory when the current
retrievers depend on vector search, keyword search, short bounded transcript
fallbacks, or recent volatile progress alone.

Current ownership:

- `src/kazusa_ai_chatbot/rag/recall_agent.py` reconciles active agreements,
  durable commitments, scheduled events, progress, and a bounded recent-history
  fallback.
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py` selects search,
  filter, or aggregate conversation workers.
- `src/kazusa_ai_chatbot/rag/conversation_search_agent.py` runs hybrid
  semantic plus literal-anchor conversation retrieval.
- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py` selects scoped user
  memory or shared persistent memory retrieval.
- `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py` exposes current
  conversation and memory vector or keyword search tools.

Current GraphRAG research and implementations point to a different retrieval
shape: extract entities and relationships into a graph, keep provenance to raw
source text, then retrieve by entity, relationship, temporal validity, graph
neighborhood, and raw source refs. Microsoft GraphRAG local search combines an
AI-extracted knowledge graph with raw document chunks for entity-specific
questions. Graphiti emphasizes temporal context graphs with fact validity,
episode provenance, and hybrid traversal for evolving agent memory. HippoRAG
uses knowledge graphs plus Personalized PageRank-style retrieval to reduce
iterative search cost for multi-hop memory questions. LightRAG adds dual-level
low-level and high-level graph retrieval with incremental update pressure in
mind.

This plan tests those ideas in Kazusa without changing production behavior.

Research references:

- Microsoft GraphRAG query engine: https://microsoft.github.io/graphrag//query/overview/
- Microsoft GraphRAG local search: https://microsoft.github.io/graphrag/query/local_search/
- Graphiti temporal context graph: https://github.com/getzep/graphiti
- HippoRAG: https://arxiv.org/abs/2405.14831
- LightRAG: https://arxiv.org/abs/2410.05779

## Mandatory Skills

- `development-plan`: load before editing this plan, approving it, or executing it.
- `local-llm-architecture`: load before changing any RAG prompt, graph query
  prompt, LLM extraction prompt, or retrieval contract.
- `database-data-pull`: load before pulling real conversation, memory, profile,
  or shared memory rows from MongoDB.
- `debug-llm`: load before running live LLM extraction, query planning,
  retrieval comparison, or writing the review artifact.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before writing Python files containing Chinese or Japanese strings.

## Mandatory Rules

- After automatic context compaction, reread this entire plan before continuing
  implementation, verification, handoff, lifecycle updates, or final reporting.
- After signing off a major checklist stage, reread this entire plan before
  starting the next stage.
- Do not read `.env`; use project scripts and settings normally loaded by the
  repo commands.
- Use `venv\Scripts\python.exe` for Python commands.
- Use real MongoDB data for the quality evaluation. Synthetic rows may only
  validate parser/schema mechanics and must not be used as performance evidence.
- No unit tests are required for experiment code. Use experiment CLIs, artifact
  validation, raw trace inspection, and the agent-authored review instead.
- Live LLM evaluation cases must run one case at a time with logs or trace artifacts
  inspected before the next case runs.
- Scripts may write raw JSON, JSONL, CSV, and logs only. The human-readable
  quality review must be authored by the agent after inspecting raw evidence.
- Agent-authored relevance review is the primary quality factor. Numeric
  metrics support the review but do not decide the outcome by themselves.
- The reviewing agent must actively compare each input query and semantic slot
  against each retrieved output and its raw source refs. A retrieval result is
  not high quality unless the raw data directly supports the needed fact.
- Do not modify production RAG behavior in this plan.
- Do not add production database collections, production indexes, service flags,
  runtime fallbacks, or live-response-path calls.
- Keep the initializer contract unchanged. The experiment may consume existing
  `Recall:`, `Conversation-evidence:`, and `Memory-evidence:` slots, but it
  must not make the initializer generate graph backend parameters.
- Deterministic code owns schema validation, source-id handling, traversal
  limits, scoring arithmetic, and artifact writing.
- LLM calls own bounded semantic extraction and graph-query interpretation only.
- Do not ask the local LLM to infer raw MongoDB schema, hidden collection
  meaning, or operational routing behavior from examples.

## Must Do

- Create an isolated GraphRAG experiment under `experiments/graph_rag_recall/`.
- Pull real conversation, scoped user-memory, and shared/world-memory data through read-only database export
  paths into `test_artifacts/graph_rag_recall/inputs/`.
- Build a graph index from real conversation rows, scoped user memory rows, and shared/world memory rows.
- Preserve source provenance from every graph fact back to conversation row ids,
  platform message ids, user memory unit ids, shared memory unit ids,
  timestamps, speaker ids, and raw text.
- Compare the GraphRAG path against current RAG2 baseline workers on the same
  real cases and the same semantic slots.
- Measure source-hit accuracy, unresolved rate, false-positive rate, search
  load, LLM call count, and prompt-facing evidence size as supporting metrics.
- Use agent-authored relevance review as the primary quality decision for each
  case and for the final go/no-go recommendation.
- Write an agent-authored Markdown review under
  `test_artifacts/graph_rag_recall/graph_rag_recall_review.md`.
- End with a go/no-go recommendation for a later production integration plan.

## Deferred

- Do not replace current recent history behavior.
- Do not change `trim_history_dict`.
- Do not change RAG initializer routing.
- Do not change production Recall, conversation-evidence, memory-evidence, or
  persistent memory search workers.
- Do not add Neo4j, external graph database infrastructure, or managed GraphRAG
  services.
- Do not add graph-derived memory writes to production collections.
- Do not promote the experiment into cognition or dialog.

## Cutover Policy

Overall strategy: compatible experiment only.

| Area | Policy | Instruction |
|---|---|---|
| Production RAG2 path | compatible | Keep existing runtime behavior untouched. |
| Experiment graph index | compatible | Store only JSON artifacts under `test_artifacts/graph_rag_recall/`. |
| Database | compatible | Read real data only. Do not create production collections or indexes. |
| Verification | compatible | Use experiment CLIs, artifact validation, live checks, and agent-authored review without changing existing RAG tests. |
| Future production adoption | migration | Requires a separate approved plan after this experiment proves value. |

## Cutover Policy Enforcement

- The responsible execution agent must keep this plan experiment-only.
- Any production runtime change requires a separate user-approved plan.
- Any persistent database schema or index change requires a separate
  user-approved plan.

## Target State

The completed experiment has two runnable retrieval paths over the same real
case set:

1. Current baseline path:
   - RAG initializer produces semantic slots.
   - Current Recall, conversation-evidence, and memory-evidence workers execute.
   - Raw outputs, source refs, LLM calls, search calls, and evidence payloads are
     captured.

2. GraphRAG candidate path:
   - The same semantic slots feed an experimental graph query planner.
   - A temporal graph index returns fact edges and source episodes.
   - Deterministic source fetch reconstructs evidence from raw conversation and
     memory refs.
   - Raw outputs, graph traversal trace, LLM calls, search calls, and evidence
     payloads are captured.

The final review states whether GraphRAG improves recall accuracy enough to
justify production integration and names the specific consumer that should be
changed first in a later plan.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Experiment boundary | Put all new code under `experiments/graph_rag_recall/` | Keeps production RAG stable while allowing real DB and live LLM work. |
| Graph shape | Use temporal entities, fact edges, and source episodes | Chat memory changes over time and needs provenance back to exact rows. |
| Input source | Use real `conversation_history`, `user_memory_units`, and `memory` exports | The question is retrieval quality on real Kazusa data, not synthetic graph mechanics. The shared/world `memory` collection must be present because current RAG often recalls it. |
| Query input | Reuse existing initializer slots | Isolates retriever quality from top-level route generation. |
| Primary retrieval | Use graph seeds, traversal, temporal validity, and source refs | Evaluates an alternative to document vector plus keyword search as the main candidate source. |
| Vector/lexical role | Baseline only in the main comparison | The current implementation uses vector search plus lexical keyword/regex paths and neighbor expansion, not a dedicated BM25 scorer. The experiment must show whether graph structure reduces search load. |
| Graph storage | JSON artifacts, not production MongoDB collections | Read-only, portable, inspectable, and easy to delete after the POC. |
| Human review | Agent-authored Markdown from raw traces is the primary quality gate | Matches the LLM debug contract and avoids hiding quality behind passing tests or aggregate metrics. |

## Contracts And Data Shapes

### Graph Index Artifact

Path pattern:

```text
test_artifacts/graph_rag_recall/index/<scope_id>_graph_index.json
```

Shape:

```json
{
  "schema_version": "graph_rag_recall.v1",
  "source_scope": {
    "platform": "qq",
    "platform_channel_id": "string",
    "scope_kind": "private | group"
  },
  "episodes": [
    {
      "episode_id": "string",
      "source_type": "conversation_history | user_memory_units | memory",
      "source_refs": [
        {
          "conversation_row_id": "string",
          "platform_message_id": "string",
          "user_memory_unit_id": "string",
          "shared_memory_unit_id": "string",
          "memory_name": "string",
          "lineage_id": "string",
          "timestamp": "string"
        }
      ],
      "speaker": {
        "global_user_id": "string",
        "platform_user_id": "string",
        "display_name": "string",
        "role": "user | assistant | system"
      },
      "text": "source text with attachment descriptions when available"
    }
  ],
  "entities": [
    {
      "entity_id": "string",
      "entity_type": "person | topic | object | place | plan | commitment | message | memory | event",
      "canonical_name": "string",
      "aliases": ["string"],
      "source_episode_ids": ["string"]
    }
  ],
  "facts": [
    {
      "fact_id": "string",
      "subject_entity_id": "string",
      "predicate": "string",
      "object_entity_id": "string",
      "object_text": "string",
      "fact_type": "agreement | preference | claim | mention | reply | plan | memory_fact | topic_link",
      "status": "active | historical | superseded | uncertain",
      "valid_from": "string",
      "valid_to": "string",
      "confidence": 0.0,
      "source_episode_ids": ["string"]
    }
  ]
}
```

### Graph Retrieval Request

```json
{
  "case_id": "string",
  "original_query": "string",
  "slot": "Recall: | Conversation-evidence: | Memory-evidence:",
  "platform": "qq",
  "platform_channel_id": "string",
  "global_user_id": "string",
  "current_timestamp_utc": "string"
}
```

### Real Data Selection Contract

`experiments/graph_rag_recall/real_data_export.py` must select transcript
scopes deterministically from `conversation_history`:

- Read QQ rows without embeddings.
- Start with rows from the last 14 days. If no qualifying private-like and
  group-like scopes exist, expand the read window to 30 days, then to all
  available QQ history.
- For each `platform_channel_id`, count user-role rows and distinct non-empty
  human speaker ids using `global_user_id` first and `platform_user_id` second.
- Select the private-like scope as the newest scope with exactly one distinct
  human speaker and at least 50 user-role rows.
- Select the group-like scope as the newest scope with at least three distinct
  human speakers and at least 100 user-role rows.
- Write the selected scope metadata, counts, and time window to
  `test_artifacts/graph_rag_recall/inputs/selected_scopes.json`.
- If either scope class is absent after reading all QQ history, stop with a
  clear error and do not create partial comparison results.
- Export scoped user memory for each selected private/group participant when a
  `global_user_id` is available, using the existing `export_user_memories`
  path without embeddings.
- Export shared/world memory from the `memory` collection using the existing
  `scripts.export_memory` path without embeddings.
- The first shared/world memory export must use `--status active --limit 500`.
  If that returns fewer than 20 rows, also export the latest 500 unfiltered
  rows and record that fallback in `selected_scopes.json`.
- Do not filter shared/world memory by `source_global_user_id`,
  `source_kind`, or `memory_type` for the main experiment; the point is to test
  whether graph retrieval can keep global memory evidence related to the same
  conversation and recall cases without relying on broad vector recall.

### Graph Evidence Pack

```json
{
  "resolved": true,
  "selected_summary": "string",
  "facts": [
    {
      "fact_id": "string",
      "fact_type": "string",
      "status": "string",
      "claim": "string",
      "source_episode_ids": ["string"]
    }
  ],
  "source_refs": [
    {
      "source_type": "conversation_history | user_memory_units | memory",
      "conversation_row_id": "string",
      "platform_message_id": "string",
      "user_memory_unit_id": "string",
      "shared_memory_unit_id": "string",
      "memory_name": "string",
      "lineage_id": "string",
      "timestamp": "string",
      "display_name": "string",
      "text": "string"
    }
  ],
  "graph_trace": {
    "seed_entities": ["string"],
    "visited_entities": ["string"],
    "visited_facts": ["string"],
    "score_breakdown": [{"id": "string", "score": 0.0, "reasons": ["string"]}]
  },
  "load": {
    "llm_calls": 0,
    "vector_calls": 0,
    "keyword_calls": 0,
    "graph_nodes_visited": 0,
    "graph_edges_visited": 0
  }
}
```

### Agent Review Rubric

The agent-authored review must inspect each case side by side. The reviewing
agent must read the real input, the semantic slot, current-path output,
GraphRAG output, and the raw source rows referenced by both outputs.

For every case, the review must assign these judgments:

```json
{
  "case_id": "string",
  "primary_quality_winner": "current_path | graph_path | tie_correct | tie_bad | inconclusive",
  "input_output_relevance": {
    "current_path": "direct | partial | adjacent | irrelevant | unresolved",
    "graph_path": "direct | partial | adjacent | irrelevant | unresolved"
  },
  "source_support": {
    "current_path": "source directly supports output | source weakly supports output | source contradicts output | no source",
    "graph_path": "source directly supports output | source weakly supports output | source contradicts output | no source"
  },
  "source_authority_match": {
    "current_path": "correct_source_class | wrong_source_class | mixed",
    "graph_path": "correct_source_class | wrong_source_class | mixed"
  },
  "temporal_match": {
    "current_path": "current_or_correct_time | stale | unclear | not_time_sensitive",
    "graph_path": "current_or_correct_time | stale | unclear | not_time_sensitive"
  },
  "review_reason": "short explanation grounded in displayed source refs"
}
```

Definitions:

- `direct`: the output answers the input/slot using source rows that directly
  contain or entail the needed fact.
- `partial`: the output contains useful evidence but misses an important
  actor, relationship, time, or source class.
- `adjacent`: the output is about a related topic but does not answer the input.
- `irrelevant`: the output does not support the input.
- `unresolved`: the path returned no usable evidence.
- `tie_bad`: both paths fail, even if one has better-looking prose.

The final recommendation must be based first on this rubric. Aggregate metrics
such as hit rate and call count are secondary evidence.

## LLM Call And Context Budget

- Production live path before this plan: unchanged.
- Production live path after this plan: unchanged.
- Offline index extraction: uses `RAG_SUBAGENT_LLM`; batch by source rows with
  a hard input cap of 35,000 characters per extraction call.
- Offline graph query planning: uses `RAG_SUBAGENT_LLM`; one call per evaluated
  slot, capped to the original query, one slot, short case metadata, and a
  compact entity-name inventory.
- Baseline runs: use current RAG2 calls as implemented.
- GraphRAG retrieval after index creation: no vector calls and no keyword
  search calls in the primary candidate path.
- Context cap: keep each experiment LLM prompt under an estimated 50,000-token
  window using character budgets and deterministic clipping.
- Blocking behavior: all experiment calls run from CLI commands; no service
  request path is blocked.

## Change Surface

### Create

- `experiments/graph_rag_recall/README.md`: operator instructions and artifact map.
- `experiments/graph_rag_recall/__init__.py`: package marker.
- `experiments/graph_rag_recall/schema.py`: dataclasses or TypedDicts for
  graph index, graph request, evidence pack, and metrics.
- `experiments/graph_rag_recall/validate_artifacts.py`: CLI validator for
  exported inputs, graph indexes, run outputs, source refs, and comparison JSON.
- `experiments/graph_rag_recall/real_data_export.py`: read-only export runner
  for deterministic private and group scopes, scoped user memory, and
  shared/world memory selected from real DB activity.
- `experiments/graph_rag_recall/extractor.py`: LLM-backed episode-to-graph
  extractor with strict JSON parsing and validation.
- `experiments/graph_rag_recall/index_store.py`: JSON index read/write and
  provenance lookup helpers.
- `experiments/graph_rag_recall/retriever.py`: deterministic graph seed,
  traversal, temporal scoring, and evidence-pack builder.
- `experiments/graph_rag_recall/baseline_runner.py`: current RAG2 baseline
  runner for the same cases.
- `experiments/graph_rag_recall/graph_runner.py`: GraphRAG candidate runner for
  the same cases.
- `experiments/graph_rag_recall/run_live_cases.py`: CLI orchestrator that runs
  one named live LLM/live DB evaluation case at a time and writes raw traces.
- `experiments/graph_rag_recall/compare_runs.py`: raw metric comparison writer.
- `experiments/graph_rag_recall/cases.jsonl`: real-data case definitions with
  source ids and expected evidence refs after data export inspection.
- `test_artifacts/graph_rag_recall/`: generated input, index, run, and review artifacts.

### Modify

- `development_plans/README.md`: register this active draft plan.

### Keep

- `src/kazusa_ai_chatbot/rag/recall_agent.py`: unchanged.
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`: unchanged.
- `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`: unchanged.
- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`: unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`: unchanged.

## Overdesign Guardrail

- Actual problem: Kazusa's current recall path can miss or mis-rank relational
  conversation and memory evidence when the needed source is not recovered by
  vector search, keyword search, or short transcript fallback.
- Minimal change: create an offline GraphRAG experiment that builds typed graph
  evidence from real rows and compares it against the current retrievers.
- Ownership boundaries: LLM extraction creates candidate semantic graph facts;
  deterministic code validates, scores, traverses, preserves provenance, and
  writes artifacts; RAG production agents stay unchanged.
- Rejected complexity: no production graph store, no service flag, no runtime
  dual-read, no graph memory writes, no new initializer route, no managed graph
  provider, and no graph-to-cognition integration.
- Evidence threshold: production integration requires real-data review showing
  better source-hit accuracy or materially lower search load without a higher
  false-positive rate.

## Agent Autonomy Boundaries

- The responsible agent may choose local Python implementation mechanics only
  when they preserve the contracts in this plan.
- The responsible agent must not introduce production behavior changes,
  persistent database schema changes, runtime feature flags, or alternate RAG
  dispatch routes.
- The responsible agent must keep experiment code under
  `experiments/graph_rag_recall/` and generated artifacts under
  `test_artifacts/graph_rag_recall/`.
- The responsible agent must search the codebase before duplicating existing
  DB export, trace-writing, JSON parsing, or LLM utility behavior.
- If equivalent behavior exists, reuse it or import it through the approved
  public boundary.
- If the plan and code disagree, preserve the plan's experiment-only intent and
  report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Establish deterministic schemas and artifact validation.
   - Add `experiments/graph_rag_recall/schema.py`.
   - Add `experiments/graph_rag_recall/validate_artifacts.py`.
   - Verify CLI availability with
     `venv\Scripts\python.exe -m experiments.graph_rag_recall.validate_artifacts --help`.

2. Add read-only real-data export.
   - Add `experiments/graph_rag_recall/real_data_export.py`.
  - Export the most recent QQ private-like scope with at least 50 user rows,
     the most recent QQ group-like scope with at least 100 user rows, scoped
     user memory for available participants, and shared/world `memory` rows.
   - Write exports to `test_artifacts/graph_rag_recall/inputs/`.
   - Verify the command prints record counts and writes JSON without embeddings.

3. Add graph extraction.
   - Add `experiments/graph_rag_recall/extractor.py`.
   - Run one live LLM extraction case at a time.
   - Write raw extraction traces to `test_artifacts/graph_rag_recall/raw_runs/`.
   - Validate every entity and fact has source provenance before accepting it
     into the index.

4. Add graph index storage.
   - Add `experiments/graph_rag_recall/index_store.py`.
   - Validate one generated index with
     `venv\Scripts\python.exe -m experiments.graph_rag_recall.validate_artifacts --kind index --path test_artifacts\graph_rag_recall\index\<scope_id>_graph_index.json`.

5. Add graph retriever.
   - Add `experiments/graph_rag_recall/retriever.py`.
   - Verify graph traversal returns source refs, not only summaries, by running
     one named dry-run case through `experiments.graph_rag_recall.graph_runner`
     and inspecting the raw JSON output.

6. Add baseline and graph runners.
   - Add `experiments/graph_rag_recall/baseline_runner.py`.
   - Add `experiments/graph_rag_recall/graph_runner.py`.
   - Use the same `cases.jsonl` for both runners.
   - Capture current RAG2 evidence, graph evidence, and load metrics.

7. Add live LLM and live DB evaluation runner.
   - Add `experiments/graph_rag_recall/run_live_cases.py`.
   - Run one case at a time with `--case <case_id>` and inspect the trace
     before the next case.

8. Compare raw outputs and write review.
   - Add `experiments/graph_rag_recall/compare_runs.py`.
   - Write raw comparison JSON.
   - Author `test_artifacts/graph_rag_recall/graph_rag_recall_review.md` by
     inspecting raw inputs, baseline outputs, graph outputs, source refs,
     validation, metric deltas, and the agent review rubric.

9. Run independent code review gate.
   - Review experiment code, raw artifacts, and Markdown review against
     this plan.
   - Fix findings inside the approved experiment boundary and rerun affected checks.

## Execution Model

- Parent agent owns orchestration, artifact validation, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes deterministic artifact contracts and validation CLI
  before production-like experiment code is written.
- Production-code subagent: exactly one native subagent, started after the
  focused artifact contract is established; owns experiment production code only.
- Parent agent may continue real-data exports, artifact validation, and review
  artifact preparation while the production-code subagent edits experiment code.
- Independent code-review subagent: exactly one native subagent after planned
  verification passes; reviews plan alignment, diff, artifacts, and
  evidence; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - deterministic artifact contracts established.
  - Covers: implementation steps 1, 4, and 5.
  - Verify: `venv\Scripts\python.exe -m experiments.graph_rag_recall.validate_artifacts --help`.
  - Evidence: record CLI output and the artifact schema contract in `Execution Evidence`.
  - Sign-off: completed for schema, CLI, and validation contract.

- [x] Stage 2 - real data exported and case set created.
  - Covers: implementation step 2.
  - Verify: `venv\Scripts\python.exe -m experiments.graph_rag_recall.real_data_export`.
  - Evidence: record exported file paths, scope summary, message counts,
    scoped user-memory counts, shared/world-memory counts, and excluded fields.
  - Sign-off: completed for initial real-data export.

- [ ] Stage 3 - GraphRAG index extraction complete.
  - Covers: implementation steps 3 and 4.
  - Verify: run each live LLM extraction case one at a time and inspect traces.
  - Evidence: record trace paths, accepted fact counts, rejected fact counts, and provenance validation.
  - Sign-off: pending.

- [ ] Stage 4 - baseline and graph runs complete.
  - Covers: implementation steps 6 and 7.
  - Verify: run live LLM/live DB evaluation cases one case at a time with
    `venv\Scripts\python.exe -m experiments.graph_rag_recall.run_live_cases --case <case_id>`.
  - Evidence: record raw run paths, per-case baseline output, per-case graph output, and load metrics.
  - Sign-off: pending.

- [ ] Stage 5 - human-readable review complete.
  - Covers: implementation step 8.
  - Verify: review artifact includes run context, input, output, side-by-side
    relevance judgments, primary quality winner per case, decision summary,
    quality notes, validation, and raw evidence paths.
  - Evidence: record `test_artifacts/graph_rag_recall/graph_rag_recall_review.md`.
  - Sign-off: pending.

- [ ] Stage 6 - independent code review complete.
  - Covers: implementation step 9.
  - Verify: review findings are recorded and affected checks are rerun after fixes.
  - Evidence: record review outcome and residual risks in `Execution Evidence`.
  - Sign-off: pending.

## Verification

### Static Checks

- `rg 'conversation[_]graph' development_plans/active/short_term/graph_rag_recall_experiment_plan.md experiments/graph_rag_recall`
  - Expected result: no matches. Exit code 1 from `rg` is acceptable.
- `rg ('TO' + 'DO|TB' + 'D|ma' + 'ybe|choose ' + 'one|option ' + 'A|option ' + 'B') development_plans/active/short_term/graph_rag_recall_experiment_plan.md`
  - Expected result: no matches. Exit code 1 from `rg` is acceptable.

### Artifact Validation

- `venv\Scripts\python.exe -m experiments.graph_rag_recall.validate_artifacts --kind inputs --path test_artifacts\graph_rag_recall\inputs`
- `venv\Scripts\python.exe -m experiments.graph_rag_recall.validate_artifacts --kind index --path test_artifacts\graph_rag_recall\index`
- `venv\Scripts\python.exe -m experiments.graph_rag_recall.validate_artifacts --kind runs --path test_artifacts\graph_rag_recall\raw_runs`

### Real Data Export

- `venv\Scripts\python.exe -m experiments.graph_rag_recall.real_data_export`
  writes real inputs under `test_artifacts/graph_rag_recall/inputs/` and
  excludes embedding vectors. The exported inputs must include
  `conversation_history`, `user_memory_units`, and `memory` source files.

### Live LLM And Live DB Evaluation

Run each case individually:

- `venv\Scripts\python.exe -m experiments.graph_rag_recall.run_live_cases --case graph_extraction_private_scope`
- `venv\Scripts\python.exe -m experiments.graph_rag_recall.run_live_cases --case graph_extraction_group_scope`
- `venv\Scripts\python.exe -m experiments.graph_rag_recall.run_live_cases --case graph_retrieval_compares_recall_case`
- `venv\Scripts\python.exe -m experiments.graph_rag_recall.run_live_cases --case graph_retrieval_compares_conversation_case`
- `venv\Scripts\python.exe -m experiments.graph_rag_recall.run_live_cases --case graph_retrieval_compares_memory_case`

### Review Artifact

- `test_artifacts/graph_rag_recall/graph_rag_recall_review.md` exists and
  includes real input excerpts, raw output summaries, source refs,
  side-by-side input-output relevance judgments, primary quality winner per
  case, validation status, and raw evidence paths.

## Independent Code Review

Run this gate after all verification commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python,
  documentation, and command artifact.
- Experiment-only boundary enforcement.
- Source provenance completeness for every graph fact and evidence item.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- LLM debug artifact quality and whether raw JSON is not being used as the main
  human review surface.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Real DB data was exported for at least one private-like QQ scope and one
  group-like QQ scope.
- Shared/world memory rows from the `memory` collection were exported without
  embeddings and included in graph extraction.
- The GraphRAG index contains entities, facts, episodes, temporal status, and
  source provenance derived from real data.
- Baseline and GraphRAG candidate retrieval run on the same real cases.
- The comparison reports source-hit accuracy, unresolved rate, false-positive
  rate, LLM calls, vector calls, keyword calls, graph traversal load, and
  prompt-facing evidence size.
- The agent-authored review is the primary quality decision and explicitly
  judges whether each path's output is directly relevant to the input and
  supported by raw source data.
- The human-readable review explains whether GraphRAG improved recall quality,
  worsened it, or only shifted cost from query time to offline extraction.
- The review recommends one of these outcomes: no production integration,
  Recall-only integration plan, conversation-evidence integration plan,
  memory-evidence integration plan, or broader graph-memory architecture plan.
- No production RAG behavior changed.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| LLM extractor invents facts | Require source refs and reject facts without provenance | Provenance validation counts rejected facts |
| Group chat entity ambiguity | Use stable `global_user_id`, platform ids, and display names when available | Review group cases with source refs |
| Graph traversal returns plausible but wrong links | Return raw source rows with every evidence pack | Source-hit and false-positive review |
| Query-time cost moves to offline extraction | Track offline extraction calls separately from query calls | Comparison metrics include both |
| Better summaries hide worse evidence | Evaluate source refs, not only prose summaries | Ground truth case refs in `cases.jsonl` |
| Experiment leaks into production | Keep code under `experiments/` and run static checks | `git diff` and review gate |

## Execution Evidence

- Plan status:
  - 2026-05-23: User approved implementation start. Status moved from
    `draft` to `in_progress`.
  - Native subagent tooling was discoverable, but the current harness only
    permits spawning subagents when the user explicitly asks for subagents.
    The parent agent proceeded locally inside the experiment-only boundary
    and will keep the independent-review gate as a manual/code-review step
    unless explicit subagent authorization is provided later.
- Static grep results:
  - `rg "conversation[_]graph" development_plans/active/short_term/graph_rag_recall_experiment_plan.md experiments/graph_rag_recall`
    returned no matches.
  - `rg ('TO' + 'DO|TB' + 'D|ma' + 'ybe|choose ' + 'one|option ' + 'A|option ' + 'B') development_plans/active/short_term/graph_rag_recall_experiment_plan.md experiments/graph_rag_recall`
    returned no matches.
- Artifact validation results:
  - `venv\Scripts\python.exe -m experiments.graph_rag_recall.validate_artifacts --kind inputs --path test_artifacts\graph_rag_recall\inputs`
    passed with 51 JSON files: 2 conversation exports, 47 user-memory exports,
    and 1 shared-memory export.
  - `venv\Scripts\python.exe -m experiments.graph_rag_recall.validate_artifacts --kind index --path test_artifacts\graph_rag_recall\index`
    passed after duplicate source episode ids were rejected by validation and
    deduped during deterministic index build.
  - `venv\Scripts\python.exe -m experiments.graph_rag_recall.validate_artifacts --kind runs --path test_artifacts\graph_rag_recall\raw_runs`
    passed.
  - `venv\Scripts\python.exe -m experiments.graph_rag_recall.validate_artifacts --kind comparison --path test_artifacts\graph_rag_recall\comparison`
    passed.
- Real data export:
  - `venv\Scripts\python.exe -m experiments.graph_rag_recall.real_data_export`
    selected QQ private scope `673225019` and QQ group scope `227608960`
    from the last 14 days.
  - Exported 1200 private conversation rows, 1200 group conversation rows,
    259 active `user_memory_units` rows across 47 users, and 182 active shared
    `memory` rows.
  - Generated 5 real-data cases in
    `test_artifacts\graph_rag_recall\inputs\cases.generated.jsonl`.
  - Export validation found no `embedding` fields.
- Live LLM/live DB evaluation results:
  - Current RAG2 baseline was run for the five generated cases one case at a
    time through `experiments.graph_rag_recall.run_live_cases`.
  - Graph candidate retrieval was run for the five generated cases with
    deterministic graph query planning against deterministic provenance
    indexes. Full LLM graph extraction and LLM graph query planning remain open.
  - Unbounded private LLM extraction with
    `venv\Scripts\python.exe -m experiments.graph_rag_recall.run_live_cases --case graph_extraction_private_scope`
    produced the first batch trace but did not complete in an acceptable
    diagnostic window. The run was stopped manually. This is evidence that full
    graph extraction must be offline, incremental, and budgeted before any
    production-facing design.
  - Bounded private LLM extraction with `--max-llm-batches 1` completed:
    1637 episodes, 457 entities, 1641 facts, 31 available LLM batches, 1 batch
    run, 4 accepted sampled LLM facts, and 6 rejected sampled LLM facts.
  - Bounded group LLM extraction with `--max-llm-batches 1` completed:
    1629 episodes, 502 entities, 1640 facts, 32 available LLM batches, 1 batch
    run, 11 accepted sampled LLM facts, and 1 rejected sampled LLM fact.
  - Graph candidate retrieval was rerun for all five generated cases with live
    LLM graph query planning enabled. The latest comparison reported baseline
    source-hit accuracy `0.6` versus graph source-hit accuracy `1.0`, baseline
    unresolved rate `0.2` versus graph unresolved rate `0.0`, and baseline
    false-positive rate `0.2` versus graph false-positive rate `0.0`.
  - Agent review found that graph retrieval improved Recall and scoped
    user-memory source hits, but over-expanded exact conversation evidence and
    ranked the group exact-message target only third. Conversation-evidence is
    not ready for graph replacement based on this stage.
- Raw artifact paths:
  - Inputs: `test_artifacts\graph_rag_recall\inputs\`
  - Indexes: `test_artifacts\graph_rag_recall\index\`
  - Raw runs: `test_artifacts\graph_rag_recall\raw_runs\`
  - Deterministic comparison:
    `test_artifacts\graph_rag_recall\comparison\graph_rag_recall_20260523T030009000891Z0000.json`
  - Bounded LLM extraction plus LLM query-planning comparison:
    `test_artifacts\graph_rag_recall\comparison\graph_rag_recall_20260523T032057366495Z0000.json`
- Human-readable review path:
  - `test_artifacts\graph_rag_recall\graph_rag_recall_review.md`
- Independent code review:
  - Pending. Native subagent execution requires an explicit user request under
    the current harness policy; no subagent review was run in this pass.
