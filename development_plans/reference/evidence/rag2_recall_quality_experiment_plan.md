# rag2_recall_quality_experiment_plan

## Lifecycle

- Status: reference evidence.
- Archived from active short-term work on 2026-05-23 after the experiment
  decision was promoted into
  `development_plans/active/short_term/rag2_cognition_ready_evidence_plan.md`.
- This document is not an executable implementation plan. Use it only for
  experiment rationale, case history, and supporting evidence.

## Summary

- Goal: Compare recall/search performance between current RAG2 and experiment-only improved search candidates over the same real database cases.
- Plan class: large
- Status: reference evidence
- Mandatory skills: `development-plan`, `local-llm-architecture`, `database-data-pull`, `debug-llm`, `py-style`, `cjk-safety`
- Overall cutover strategy: compatible experiment only; no production runtime, prompt, schema, index, database, or collection changes.
- Highest-risk areas: unfair case selection, false-positive evidence, scoped memory/source confusion, image-only conversation rows, active commitment state confusion, unreadable comparison output, evidence rows that are retrieved but not summarized, raw UTC timestamp leakage, adapter-wire CQ leakage, and `rag_result` evidence that is too raw for cognition to consume reliably.
- Acceptance criteria: The experiment produces a readable blind current-RAG2-vs-improved-search comparison on real data, judged only from visible input/output plus cost/load deltas after output quality is assessed.

## Context

Kazusa's current RAG2 path is the production baseline:

```text
stage_1_research
  -> call_quote_aware_rag_supervisor
  -> rag_initializer
  -> rag_dispatcher
  -> top-level capability worker
  -> evaluator
  -> finalizer
  -> project_known_facts
  -> rag_result
```

The experiment does not try to redesign that pipeline. It compares the current
pipeline against narrower search improvements that could later be implemented
inside existing RAG2 subagents if they prove better.

Core comparison:

```text
same real input
current RAG2 result
experiment-only improved search result
blind agent-authored quality comparison
cost/load comparison
```

Relevant existing modules:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/rag/quote_aware_sequence.py`
- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/persistent_memory_search_agent.py`
- `src/kazusa_ai_chatbot/rag/user_memory_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/recall_agent.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`
- `src/kazusa_ai_chatbot/db/memory.py`
- `src/kazusa_ai_chatbot/db/memory_evolution.py`
- `src/kazusa_ai_chatbot/db/user_memory_units.py`
- `src/kazusa_ai_chatbot/db/conversation.py`

Graph, DAG, GraphRAG, and conversation-graph work are out of scope for this
plan.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing experiment prompt shape,
  RAG2 invocation shape, candidate search contracts, or LLM context budgets.
- `database-data-pull`: load before exporting real conversation, shared memory,
  scoped user memory, conversation progress, or scheduled-event data.
- `debug-llm`: load before running live RAG2, running live LLM-backed search
  candidates, comparing outputs, or writing the human-readable review.
- `py-style`: load before editing Python experiment files.
- `cjk-safety`: load before writing Python files containing Chinese or Japanese
  strings.

## Mandatory Rules

- After automatic context compaction, reread this entire plan before continuing
  implementation, verification, handoff, lifecycle updates, or final reporting.
- After signing off a major checklist stage, reread this entire plan before
  starting the next stage.
- This is an experiment-only plan. Do not modify production source, production
  prompts, production configuration, production tests, service startup, runtime
  adapters, or live response-path behavior.
- Do not create a new database, collection, index, cache, or persistent store.
- Do not write to existing MongoDB collections. The experiment reads existing
  collections and writes filesystem artifacts only.
- Allowed database sources are existing `memory`, `user_memory_units`,
  `conversation_history`, `conversation_progress`, and scheduled-event data
  when an active-commitment case requires it.
- Do not read `.env`. Use project scripts and settings normally loaded by repo
  commands.
- Use `venv\Scripts\python.exe` for Python commands.
- Use real MongoDB data for the generated inputs and outputs. Synthetic rows
  may validate harness mechanics only and must not be counted as recall-quality
  evidence.
- Do not create, keep, or maintain unit tests for experiment code in this plan.
  Verification is command execution, artifact validation, trace inspection, and
  agent-authored review.
- Scripts may emit raw JSON, JSONL, CSV, and logs only. The human-readable
  Markdown review must be authored by the agent after inspecting raw evidence.
- Every comparison report must follow the `debug-llm` review shape: run
  context, evaluation goal, real input, raw/interpreted outputs,
  decision/behavior summary, quality notes, validation, and raw evidence paths.
- Scripts must not assign qualitative evaluation labels such as `correct`,
  `partial`, `miss`, `high`, `low`, `false_positive`, `winner`, `adopt`,
  `reject`, or `narrow-fix`. Those judgments belong only in the
  agent-authored review.
- Scripts must not filter baseline or improved-search outputs by judged
  relevance or quality. Candidate search code may apply bounded retrieval
  limits, but it must write enough raw selected evidence and source refs for
  the reviewing agent to evaluate the result directly.
- Do not truncate input text, current RAG2 output, improved search output, or
  model output in the review artifact. If a field is too long for a table, put
  the full text in a per-case subsection.
- Agent-authored relevance review is the primary quality factor. Numeric
  metrics support the review but do not decide quality alone.
- Current RAG2 must be evaluated as the production baseline, including its
  current initializer, dispatcher, workers, evaluator, finalizer, and
  projection.
- Improved search candidates must remain experiment-only read paths. They must
  not change current RAG2 behavior.
- The initializer remains semantic-only. The experiment must not require the
  initializer to generate MongoDB filters, backend parameters, or search
  implementation details.

## Must Do

- Create an isolated experiment package under
  `experiments/rag2_recall_quality/`.
- Create raw output directories under `test_artifacts/rag2_recall_quality/`.
- Build real-data input cases from database exports.
- Include the explicit input matrix in this plan's required case mix and in
  the generated `case_matrix.jsonl` summary.
- Run current RAG2 end-to-end as the baseline for every case.
- Run experiment-only improved search candidates for every case using the same
  input and the same allowed database sources.
- Compare baseline and improved-search outputs side by side.
- Evaluate each case with a blind agent-authored input/output comparison in the
  Markdown review. Do not generate evaluation judgments from code.
- When a user-reviewed blind run exposes output-shape weaknesses, rerun the
  same input cases after improving only experiment-side evidence formatting and
  summarization. Keep the same raw current-RAG2 baseline artifacts unless the
  baseline invocation itself is intentionally rerun and recorded.
- Project evidence timestamps to configured local wall-clock time before they
  appear in prompt-facing or human-facing improved-search output. Use
  second-level precision; keep raw UTC timestamps only in trace artifacts.
- Strip adapter-wire CQ syntax from prompt-facing improved-search evidence.
  Use typed `body_text`, reply excerpts, and attachment descriptions as the
  semantic source, following the completed prompt-safe history projection
  direction.
- Capture enough trace data to judge quality and cost:
  - full input;
  - baseline slots, dispatched workers, selected evidence, and projected
    `rag_result`;
  - improved-search query strategy, selected rows, source refs, and evidence;
  - runner-observed LLM calls, embedding calls, DB/search calls, candidate
    counts, and prompt-facing evidence size.
- Write raw structured artifacts only from scripts.
- Write an agent-authored Markdown review at
  `test_artifacts/rag2_recall_quality/rag2_recall_quality_review.md`.
- End with a direct summary of blind IO wins/losses. Candidate adoption
  decisions require a later plan because this blind comparison hides source
  truth by design.

## Deferred

- Do not use DAG, graph, GraphRAG, conversation graph, or `conversation_graph`
  as the candidate retrieval path.
- Do not replace recent history.
- Do not change `conversation_progress`.
- Do not change `rag_initializer`, `rag_dispatcher`, top-level capability
  agents, worker prompts, production projection, cognition prompts, or dialog.
- Do not add production database collections, indexes, flags, caches, or
  compatibility layers.
- Do not write or preserve experiment unit-test files for this plan. If a
  draft `test_*.py`, `*_test.py`, or `tests/**` file is created during
  execution, remove it before sign-off.
- Do not produce CSV or JSON as the main human review surface.
- Do not promote any finding into production code in this plan.

## Cutover Policy

Overall strategy: compatible experiment only.

| Area | Policy | Instruction |
|---|---|---|
| Production RAG2 path | compatible | Invoke existing code as baseline only; do not modify runtime behavior. |
| Improved search candidates | compatible | Implement under `experiments/rag2_recall_quality/` only. |
| Artifacts | compatible | Write raw and review output only under `test_artifacts/rag2_recall_quality/`. |
| Database | compatible | Read existing collections only; do not create or write databases, collections, or indexes. |
| Future production fix | separate plan | Create a later approved bugfix or short-term plan only after this experiment proves a candidate improvement. |

## Target State

The completed experiment answers one question:

> Which visible output is more useful for the same real input: current RAG2 or
> experiment-only improved search, and what quality/cost tradeoff remains?

For each case, the final review shows:

```text
Input
Current RAG2 result
Improved search result
Blind quality comparison
Better visible output
Reason
Cost/load delta
```

The final recommendation must report blind IO wins/losses and identify which
failure modes deserve a later production plan.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Experiment boundary | Use `experiments/rag2_recall_quality/` | Keeps all new code outside production runtime. |
| Baseline | Current quote-aware RAG2 end-to-end path | Measures the behavior Kazusa uses today. |
| Candidate path | Experiment-only search improvements over existing DB data | Compares concrete search changes without changing production. |
| Data storage | Filesystem artifacts only | Avoids new databases, collections, indexes, and runtime state. |
| Case selection | Curated real-data matrix | The user needs realistic inputs, not random inputs with unclear intent. |
| Quality gate | Blind agent-authored side-by-side review | The main question is visible output quality, not only schema validity or gold-source match. |
| User correction on `rag2-alt-003` | Count the richer improved-search output as preferred | The user judged that the added evidence and insight are more valuable to cognition than the plainer current-RAG2 answer. |
| Known loss on `rag2-alt-010` | Treat it as an evidence-summarization and thread-pruning failure | The improved path retrieved the right rows but polluted the summary with unrelated neighboring group messages. |
| Timestamp projection | Improved prompt-facing output uses configured local time with second precision | Raw UTC and microseconds are not cognition-friendly or human-readable evidence. |
| CQ projection | Improved prompt-facing output must not quote raw CQ syntax | Adapter wire syntax is not the brain contract; use the same prompt-safe projection direction as recent history fixes. |
| RAG-to-cognition contract | Improved output must become cognition-ready facts, not only evidence rows | RAG returns evidence, but cognition needs concise source-backed facts to interpret stance and response goals. |
| Production action | Deferred | This plan measures candidates; it does not ship fixes. |

## Improved Search Candidates

Each candidate is an experiment-only read path. Candidate output must preserve
source ids, source collection, full source text, and a prompt-facing evidence
summary.

| Candidate | Source | Search behavior |
|---|---|---|
| `shared_memory_literal_metadata` | `memory` | Active rows only. Search `memory_name`, `memory_name_contains`, `memory_type`, `source_kind`, `authority`, literal anchors, and then vector fallback when needed. |
| `user_memory_scoped_literal` | `user_memory_units` | Current `global_user_id` only. Search `unit_type`, `status`, literal terms over `fact`, `subjective_appraisal`, and `relationship_signal`, then vector fallback when needed. |
| `user_memory_state_lookup` | `user_memory_units` | Current `global_user_id` only. Search active and non-active rows when the question asks whether an item is still owed, completed, or already closed. Preserve status, completion, cancellation, and archive fields. |
| `conversation_media_literal` | `conversation_history` | Search full message text, attachment descriptions, reply excerpts, speaker, platform/channel, and time bounds when present. Preserve media descriptions in selected evidence. |
| `recall_active_strict` | `user_memory_units`, `conversation_progress`, scheduled events | Prioritize active commitments and open loops. Reject completed, cancelled, fulfilled, or stale evidence unless the input explicitly asks about past completion. |

## Input Matrix

The 10-case input matrix was the first experiment batch. It is retained as raw
debugging history only. It must not be used as the current decision surface
because its expected-source framing can bias the comparison.

| Case type | Count | Source family used to construct input | Improved-search candidate |
|---|---:|---|---|
| Shared/world memory | 2 | `memory` | `shared_memory_literal_metadata` |
| Scoped user memory | 2 | `user_memory_units` | `user_memory_scoped_literal` |
| Active commitment/open loop | 2 | `user_memory_units`, `conversation_progress`, scheduled events when present | `recall_active_strict` |
| Historical conversation event | 2 | `conversation_history` | `conversation_media_literal` |
| Media-heavy conversation | 1 | `conversation_history` attachment/reply fields | `conversation_media_literal` |
| Negative/unknown | 1 | no known support in inspected data | `none` |

Random chat inputs are not a substitute for this matrix. Random cases may be
added only in a later experiment after the blind IO matrix is reviewed.

## Contextual Recall Batch

The first generated batch used isolated current inputs. That is not the real
shape of RAG2. RAG2 receives a decontextualized current input plus narrowed
recent conversation context. The second batch therefore tests that input shape
directly.

The initial contextual review used expected DB source rows as the evaluation
frame. That is now considered biased for deciding current RAG2 versus the
improved search candidate, because the improved candidate design was built
around those source anchors.

The decision artifact for this batch is now a blind input/output comparison.
The reviewer sees only:

```json
{
  "case_id": "stable case id",
  "raw_input": "raw user-facing follow-up",
  "rag_facing_input": "decontextualized query handed to RAG",
  "output_a": "anonymous output from one method",
  "output_b": "anonymous output from the other method"
}
```

Blind artifact path:

```text
test_artifacts/rag2_recall_quality/inputs/contextual_blind_io_comparison.jsonl
```

Blind review path:

```text
test_artifacts/rag2_recall_quality/contextual_blind_io_subagent_review.md
```

The blind review must not use expected sources, distractor sources, source ids,
search anchors, DB exports, prior review text, or production code. It compares
only the visible input and the two visible outputs. Routing remains outside the
main experimental variable; any routing behavior observed in Output A or Output
B is judged only as it affects the visible output quality.

The contextual cases remain hard, source-grounded follow-ups constructed from
real DB rows, but gold rows are not shown to the blind reviewer and are not used
as the judging frame.

### User Review Conclusion And Improvement Scope

The second 10-case contextual run is not rejected. The user corrected the
interpretation of `rag2-alt-003`: the improved output should be preferred
because it provides more information and insight for downstream cognition than
the plain current-RAG2 answer. With that correction, the second-batch visible
output result is:

```text
formatted improved search: 9 cases preferred
current RAG2 baseline: 1 case preferred
tie: 0
unclear: 0
```

The remaining loss, `rag2-alt-010`, is not a retrieval failure. The improved
search found the oxygen-sensor image row and the immediate follow-up, but the
formatter summarized unrelated nearby Flipper messages as if they belonged to
the same follow-up. This identifies the next experiment target:

```text
improved retrieval evidence
  -> prompt-safe local-time projection
  -> answer-oriented evidence summarization
  -> cognition-ready source-backed facts
```

The next rerun must use the same `contextual_case_matrix_alt10.jsonl` inputs and
must compare current RAG2 against the improved formatter after these fixes:

- add a concise direct conclusion before evidence details;
- add evidence summaries that preserve source-backed facts rather than dumping
  raw rows as the primary output;
- keep enough full evidence text for human review without truncating the report;
- convert evidence timestamps from storage UTC to configured local time with
  second precision;
- remove raw CQ syntax from reply excerpts and prompt-facing evidence;
- keep raw UTC timestamps, raw selected rows, and source ids only in JSON trace
  artifacts;
- evaluate whether the output shape satisfies the RAG-to-cognition contract:
  cognition should receive direct facts, uncertainty, and supporting evidence
  rather than needing to infer the answer from raw transcript rows.

This work remains experiment-only. It does not modify production RAG2,
cognition, dialog, prompts, database schema, indexes, or runtime behavior.

## Final Experiment Decision

The experiment decision is to productionize the formatted improved-search V2
behaviors inside existing RAG2 rather than adopt DAG, graph RAG, or a new
retrieval architecture.

Decision basis:

- The same-input improved V2 blind-style run preferred formatted improved
  search in 10/10 real database cases.
- The user reviewed the visible output and preferred the formatted improved
  shape because it gives cognition direct facts, supporting evidence, and
  useful uncertainty instead of making downstream agents infer from raw rows.
- The decisive gaps were output shape, prompt-safe projection, evidence
  summarization, timestamp presentation, and bounded search behavior. The run
  did not prove a need for graph storage or a new RAG pipeline.

Production ownership:

- New production work belongs to
  `development_plans/active/short_term/rag2_cognition_ready_evidence_plan.md`.
- Experiment code under `experiments/rag2_recall_quality/` is design reference
  only. Production source and tests must not import it.
- The implementation agent should inspect the experiment behavior in
  `improved_search_runner.py` and `format_improved_outputs.py`, then reimplement
  the approved behavior through production RAG2 modules and existing
  prompt-safe projection helpers.

## Contracts And Data Shapes

### Case Matrix

Path:

```text
test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl
```

Each row:

```json
{
  "case_id": "rag2-memory-001",
  "case_type": "shared_memory | user_memory | active_commitment | conversation_event | media_conversation | negative_unknown",
  "platform": "qq",
  "platform_channel_id": "string",
  "channel_type": "private | group",
  "global_user_id": "string",
  "input_text": "full user input text",
  "candidate": "shared_memory_literal_metadata | user_memory_scoped_literal | conversation_media_literal | recall_active_strict | none",
  "expected_sources": [
    {
      "source_collection": "memory | user_memory_units | conversation_history | conversation_progress | scheduled_events",
      "source_id": "string",
      "source_timestamp_utc": "string",
      "full_source_text": "full raw source text or readable projection"
    }
  ],
  "quality_question": "What should a correct retrieval result prove?"
}
```

Minimum matrix:

- 2 shared/world memory cases from `memory`;
- 2 current-user durable memory cases from `user_memory_units`;
- 2 active commitment or open-loop cases;
- 2 historical conversation event cases;
- 1 image-only or attachment-heavy conversation case;
- 1 negative/unknown case where confident evidence should not be returned.

### Baseline Trace

Path pattern:

```text
test_artifacts/rag2_recall_quality/baseline/<case_id>.json
```

Required fields:

```json
{
  "case_id": "rag2-memory-001",
  "input_text": "full user input text",
  "decontextualized_input": "full decontextualized text",
  "initializer_slots": ["Memory-evidence: ..."],
  "dispatches": [],
  "rag_supervisor_result": {},
  "projected_rag_result": {},
  "observed_load": {
    "loop_count": 0,
    "llm_calls": "not_exposed | integer",
    "embedding_calls": "not_exposed | integer",
    "db_or_search_calls": "not_exposed | integer",
    "candidate_count": "not_exposed | integer",
    "evidence_char_count": 0
  },
  "artifact_version": "rag2_recall_quality.v2"
}
```

### Improved Search Trace

Path pattern:

```text
test_artifacts/rag2_recall_quality/improved_search/<case_id>.json
```

Required fields:

```json
{
  "case_id": "rag2-memory-001",
  "candidate": "shared_memory_literal_metadata",
  "input_text": "full user input text",
  "query_strategy": {
    "literal_terms": ["string"],
    "filters": {},
    "fallbacks_used": ["keyword | metadata | vector | recency"]
  },
  "selected_evidence": [
    {
      "source_collection": "memory",
      "source_id": "string",
      "source_timestamp_utc": "string",
      "full_source_text": "full row text",
      "prompt_evidence": "full prompt-facing evidence text",
      "score": 0.0,
      "metadata": {}
    }
  ],
  "observed_load": {
    "llm_calls": 0,
    "embedding_calls": 0,
    "db_or_search_calls": 0,
    "candidate_count": 0,
    "evidence_char_count": 0
  },
  "artifact_version": "rag2_recall_quality.v2"
}
```

### Comparison Review

Path:

```text
test_artifacts/rag2_recall_quality/rag2_recall_quality_review.md
```

Required per-case section:

```markdown
## Case <case_id>

### Input

### Output A

### Output B

### Blind Quality Comparison

| Dimension | Output A | Output B | Better | Reason |
|---|---|---|---|---|

### Load Notes

### Decision After Mapping
```

### Agent Evaluation Matrix

The reviewing agent must fill this matrix in the Markdown review for every case
using only the visible input and outputs. Scripts must not prefill these fields,
compute winners, or drop evidence based on these dimensions.

| Dimension | Output A judgment | Output B judgment | Better | Agent reasoning required |
|---|---|---|---|---|
| Directness | `high`, `medium`, `low`, or `wrong` | `high`, `medium`, `low`, or `wrong` | `A`, `B`, `tie`, or `unclear` | Does the output answer the visible user request directly? |
| Specificity | `high`, `medium`, `low`, or `wrong` | `high`, `medium`, `low`, or `wrong` | `A`, `B`, `tie`, or `unclear` | Does the output contain concrete details needed to respond? |
| Caution | `appropriate`, `overconfident`, or `too_minimal` | `appropriate`, `overconfident`, or `too_minimal` | `A`, `B`, `tie`, or `unclear` | Does the output avoid unsupported certainty while remaining useful? |
| Noise | `low`, `medium`, or `high` | `low`, `medium`, or `high` | `A`, `B`, `tie`, or `unclear` | Does the output include distracting or unrelated material? |
| Context usefulness | `high`, `medium`, `low`, or `wrong` | `high`, `medium`, `low`, or `wrong` | `A`, `B`, `tie`, or `unclear` | Would cognition/dialog have enough visible material to answer well? |
| Final-answer readiness | `ready`, `needs_synthesis`, or `unusable` | `ready`, `needs_synthesis`, or `unusable` | `A`, `B`, `tie`, or `unclear` | Is the output already answer-like, or only evidence fragments? |
| Cost/load | observed counters or `not_exposed` | observed counters or `not_exposed` | `A`, `B`, `tie`, or `unclear` | Compare only after quality is judged from visible input/output. |

The final per-case decision must be agent-authored as one of:

- `A`: Output A is better from visible IO only.
- `B`: Output B is better from visible IO only.
- `tie`: both outputs are equivalent for this case.
- `unclear`: visible IO is insufficient to decide.

Only after the blind review is complete may the coordinator map Output A and
Output B back to implementation names.

## LLM Call And Context Budget

- Case matrix construction must use database reads and deterministic sampling;
  it must not require LLM calls.
- Baseline runs use existing RAG2 calls and existing supervisor loop caps.
- Improved search candidates may use embeddings when existing DB APIs require
  them for vector fallback.
- Improved search candidates must not add multi-turn agent loops.
- Runner-observed counters must be recorded for both paths. When current RAG2
  does not expose a counter, record `not_exposed` rather than inventing a value.
- The first batch is exactly 10 cases. Expanding beyond 10 cases requires a new
  user instruction.

## Change Surface

Create:

- `experiments/rag2_recall_quality/__init__.py`
- `experiments/rag2_recall_quality/README.md`
- `experiments/rag2_recall_quality/schema.py`
- `experiments/rag2_recall_quality/case_matrix_builder.py`
- `experiments/rag2_recall_quality/baseline_runner.py`
- `experiments/rag2_recall_quality/improved_search_runner.py`
- `experiments/rag2_recall_quality/format_improved_outputs.py`
- `experiments/rag2_recall_quality/compare_runs.py`
- `experiments/rag2_recall_quality/artifact_validation.py`
- `experiments/rag2_recall_quality/run_case_batch.py`

Generated at execution time:

- `test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl`
- `test_artifacts/rag2_recall_quality/baseline/*.json`
- `test_artifacts/rag2_recall_quality/improved_search/*.json`
- `test_artifacts/rag2_recall_quality/formatted_improved_search/*.json`
- `test_artifacts/rag2_recall_quality/comparisons/*.json`
- `test_artifacts/rag2_recall_quality/rag2_recall_quality_review.md`

Modify:

- `development_plans/README.md` only to register or update this plan.

Forbidden:

- `src/kazusa_ai_chatbot/**`
- `tests/**`
- existing `experiments/graph_rag_recall/**`
- production configuration files
- database migrations, index setup, or collection creation

## Overdesign Guardrail

- Keep the experiment focused on baseline RAG2 versus improved search
  candidates.
- Do not introduce a new retrieval architecture.
- Do not create a general experiment framework.
- Do not add production abstractions.
- Do not optimize aggregate metrics before the per-case comparison is readable.
- Prefer explicit JSON artifacts plus an agent-authored review over hidden
  in-memory state.

## Agent Autonomy Boundaries

- The execution agent may add helper functions inside
  `experiments/rag2_recall_quality/` when they only support approved artifacts
  and commands.
- The execution agent may adjust artifact field ordering for readability while
  preserving every required field.
- The execution agent must record pre-existing dirty working-tree paths before
  execution and avoid touching them.
- The execution agent must stop and request approval before modifying
  production source, prompts, config, indexes, tests, existing graph experiment
  files, or database schema.
- The execution agent must stop and request approval if real database access is
  unavailable.
- The execution agent must stop and request approval if live LLM routes are not
  configured enough to run the baseline.

## Implementation Order

1. Create the experiment package and README.
   - Files: `experiments/rag2_recall_quality/__init__.py`,
     `experiments/rag2_recall_quality/README.md`.
   - Verify: `Get-ChildItem -LiteralPath 'experiments/rag2_recall_quality'`.
2. Define artifact schemas and path helpers.
   - File: `experiments/rag2_recall_quality/schema.py`.
   - Verify: `venv\Scripts\python.exe -m py_compile experiments/rag2_recall_quality/schema.py`.
3. Implement real-data case matrix construction.
   - File: `experiments/rag2_recall_quality/case_matrix_builder.py`.
   - Command:
     `venv\Scripts\python.exe -m experiments.rag2_recall_quality.case_matrix_builder --limit 10 --output test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl`.
   - Verify: output contains exactly 10 cases and all required case types.
4. Implement current RAG2 baseline trace capture.
   - File: `experiments/rag2_recall_quality/baseline_runner.py`.
   - Command:
     `venv\Scripts\python.exe -m experiments.rag2_recall_quality.baseline_runner --cases test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl --case-id <case_id>`.
   - Verify: one baseline JSON exists and contains full input, slots,
     dispatches, supervisor result, projected `rag_result`, and observed load.
5. Implement improved search candidate runner.
   - File: `experiments/rag2_recall_quality/improved_search_runner.py`.
   - Command:
     `venv\Scripts\python.exe -m experiments.rag2_recall_quality.improved_search_runner --cases test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl --case-id <case_id>`.
   - Verify: one improved-search JSON exists and contains full input, query
     strategy, selected source evidence, source refs, and observed load.
6. Implement machine-readable comparison assembly.
   - File: `experiments/rag2_recall_quality/compare_runs.py`.
   - Command:
     `venv\Scripts\python.exe -m experiments.rag2_recall_quality.compare_runs --cases test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl`.
   - Verify: comparison JSON exists for each case. It must not contain agent
     prose quality judgments, correctness labels, relevance labels, winners,
     or candidate adoption decisions.
7. Implement artifact validation.
   - File: `experiments/rag2_recall_quality/artifact_validation.py`.
   - Command:
     `venv\Scripts\python.exe -m experiments.rag2_recall_quality.artifact_validation --root test_artifacts/rag2_recall_quality`.
   - Verify: validator reports required artifacts present, no review-critical
     text fields truncated, and no experiment unit-test files present.
8. Implement batch runner.
   - File: `experiments/rag2_recall_quality/run_case_batch.py`.
   - Command:
     `venv\Scripts\python.exe -m experiments.rag2_recall_quality.run_case_batch --cases test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl --limit 10`.
   - Verify: baseline, improved-search, and comparison artifacts exist for all
     10 cases.
9. Run one live case through baseline and improved-search paths.
   - Verify: inspect the raw artifacts before running the full batch.
10. Run the 10-case batch and artifact validation.
    - Verify: validation passes and all expected artifacts exist.
11. Write the agent-authored review.
    - File:
      `test_artifacts/rag2_recall_quality/rag2_recall_quality_review.md`.
    - Verify: the review points to the blind IO comparison, and does not use
      expected-source matching as the current decision surface.
12. Write the blind IO subagent review.
    - File:
      `test_artifacts/rag2_recall_quality/contextual_blind_io_subagent_review.md`.
    - Verify: the reviewer compared only visible input, Output A, and Output B.

## Execution Model

This plan is draft and must not be executed until the user approves it.

After approval, execution is parent-led. The parent owns artifact inspection,
live-LLM sequencing, review authorship, verification, lifecycle updates, and
final sign-off. A code-writing subagent may implement only the experiment files
under `experiments/rag2_recall_quality/`. No production-code subagent is
authorized because production source changes are out of scope.

Independent code review is required before completion. The review scope is the
experiment package, generated artifact contract, command behavior, database
read-only behavior, comparison fairness, and proof that production source was
not changed by this plan.

## Progress Checklist

- [x] Plan approved by the user before execution.
- [x] Pre-existing dirty working-tree paths recorded.
- [x] Experiment package created under `experiments/rag2_recall_quality/`.
- [x] Artifact schemas and path helpers compile.
- [x] Real-data 10-case matrix generated with all required case types.
- [x] One baseline RAG2 case run and inspected.
- [x] One improved-search case run and inspected.
- [x] Comparison assembly implemented and inspected.
- [x] Artifact validator implemented and run.
- [x] 10-case batch run completed.
- [x] Agent-authored Markdown review reset to the blind IO decision surface.
- [x] Blind IO subagent comparison completed.
- [ ] Independent code review completed.
- [x] Execution evidence recorded and plan status updated.

## Verification

Required commands after implementation:

```powershell
venv\Scripts\python.exe -m py_compile experiments/rag2_recall_quality/schema.py
venv\Scripts\python.exe -m experiments.rag2_recall_quality.case_matrix_builder --limit 10 --output test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl
venv\Scripts\python.exe -m experiments.rag2_recall_quality.run_case_batch --cases test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl --limit 10
venv\Scripts\python.exe -m experiments.rag2_recall_quality.format_improved_outputs --cases test_artifacts/rag2_recall_quality/inputs/contextual_case_matrix.jsonl --limit 12 --output test_artifacts/rag2_recall_quality/inputs/contextual_formatted_blind_io_comparison.jsonl
venv\Scripts\python.exe -m experiments.rag2_recall_quality.artifact_validation --root test_artifacts/rag2_recall_quality
Get-ChildItem -LiteralPath 'experiments/rag2_recall_quality' -Recurse -File | Where-Object { $_.Name -match '(^test_|_test\.py$)' }
git status --short
```

Expected verification results:

- `py_compile` succeeds.
- Case matrix contains exactly 10 real-data cases.
- Batch runner creates baseline, improved-search, and comparison artifacts for
  every case.
- Artifact validation reports required fields present and no truncated
  review-critical text fields.
- Artifact validation reports that generated JSON/JSONL artifacts do not
  contain script-authored qualitative evaluation labels or winners.
- The experiment test-file check prints no files.
- `git status --short` shows no new plan-caused modifications outside allowed
  paths. Pre-existing dirty paths must match the recorded pre-execution dirty
  list.

No unit test command is required or allowed for this experiment plan.

## Independent Code Review

Before completion, run an independent review with this scope:

- Confirm only allowed files changed.
- Confirm scripts read existing MongoDB data only.
- Confirm scripts do not create or write databases, collections, indexes, or
  caches.
- Confirm scripts do not generate the human-readable Markdown review.
- Confirm scripts do not generate qualitative evaluation labels, winners, or
  candidate adoption decisions.
- Confirm full text fields are preserved in artifacts and review.
- Confirm current RAG2 and improved search candidates are compared on the same
  visible inputs.
- Confirm quality judgments are grounded in displayed raw input/output only for
  the blind decision artifact.
- Confirm implementation names are mapped back only after blind review is
  completed.

Record findings, fixes, rerun commands, and residual risk in execution
evidence before marking the plan completed.

## Acceptance Criteria

- New code exists only under `experiments/rag2_recall_quality/`.
- Generated artifacts exist only under `test_artifacts/rag2_recall_quality/`.
- No experiment unit-test files are created or retained by this plan.
- No new database, collection, index, cache, or persistent store is created.
- No existing MongoDB collection is written by this plan.
- No production source, production prompt, production config, production test,
  graph experiment, database schema, or runtime behavior is changed by this
  plan.
- The case matrix includes real examples from `memory`, `user_memory_units`,
  active commitment or open-loop state, historical conversation, media-heavy
  conversation, and negative/unknown recall.
- Every case shows current RAG2 result and improved-search result side by side.
- The blind IO review is readable without opening JSON or CSV first.
- The blind IO review contains a case-by-case A/B decision and reason.
- No script-authored artifact contains correctness, relevance, winner, or
  adoption/rejection judgments.
- The final recommendation reports visible-output wins/losses. Candidate
  adoption/rejection is deferred to a later production plan.

## Risks

- Case selection can bias the improved search path if expected sources are used
  as the judging frame. The decision review must use blind input/output only.
- Current RAG2 may not expose all load counters. The runner must record
  `not_exposed` for unavailable counters and still compare exposed loop count,
  candidate count, and evidence size.
- Live LLM routes may be unavailable. The execution agent must stop and report
  the route blocker instead of substituting synthetic model output.
- Improved search can be more useful as evidence but weaker as a final answer.
  Blind review must distinguish evidence quality from final-answer readiness.

## Execution Evidence

### 2026-05-23 start

- User approved starting experiment execution.
- Pre-existing dirty working-tree paths before experiment code changes:
  - `development_plans/README.md`
  - `src/kazusa_ai_chatbot/reflection_cycle/worker.py`
  - `src/kazusa_ai_chatbot/self_cognition/worker.py`
  - `tests/test_reflection_cycle_stage1c_worker.py`
  - `tests/test_self_cognition_event_logging.py`
  - `development_plans/active/short_term/graph_rag_recall_experiment_plan.md`
  - `development_plans/active/short_term/rag2_recall_quality_experiment_plan.md`

### 2026-05-23 execution

- Created experiment package:
  - `experiments/rag2_recall_quality/__init__.py`
  - `experiments/rag2_recall_quality/README.md`
  - `experiments/rag2_recall_quality/schema.py`
  - `experiments/rag2_recall_quality/case_matrix_builder.py`
  - `experiments/rag2_recall_quality/baseline_runner.py`
  - `experiments/rag2_recall_quality/improved_search_runner.py`
  - `experiments/rag2_recall_quality/compare_runs.py`
  - `experiments/rag2_recall_quality/artifact_validation.py`
  - `experiments/rag2_recall_quality/run_case_batch.py`
- Generated real-data matrix:
  - Command:
    `venv\Scripts\python.exe -m experiments.rag2_recall_quality.case_matrix_builder --limit 10 --output test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl`
  - Result: 10 cases with counts `shared_memory=2`, `user_memory=2`,
    `active_commitment=2`, `conversation_event=2`,
    `media_conversation=1`, `negative_unknown=1`.
- Ran one-case smoke batch for `rag2-recall-001` and inspected output.
  - Finding: current RAG2 retrieved the target shared-memory row but expanded
    into adjacent boundary memories.
  - Read-only caveat: this first smoke run exposed normal production
    telemetry/cache write paths before guards were installed.
- Added experiment-local read-only runtime guards in `baseline_runner.py`:
  - `event_logging.record_rag_stage_event`
  - `persona_supervisor2_rag_initializer.record_initializer_hit`
  - `persona_supervisor2_rag_initializer.upsert_initializer_entry`
  - `persona_supervisor2_rag_supervisor2.record_initializer_hit`
  - `persona_supervisor2_rag_supervisor2.upsert_initializer_entry`
- Ran guarded full batch:
  - Command:
    `venv\Scripts\python.exe -m experiments.rag2_recall_quality.run_case_batch --cases test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl --limit 10`
  - Result: wrote 10 complete case comparisons.
- Improved the experiment-only candidate path after inspecting raw artifacts:
  - Stripped prompt framing before literal search.
  - Preserved attachment description anchors.
  - Stopped after the first useful literal hit.
  - Scoped active commitment search through keyword/vector filters before
    recency fallback.
- Reran improved-search artifacts and comparisons:
  - Command:
    `venv\Scripts\python.exe -m experiments.rag2_recall_quality.improved_search_runner --cases test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl --limit 10`
  - Command:
    `venv\Scripts\python.exe -m experiments.rag2_recall_quality.compare_runs --cases test_artifacts/rag2_recall_quality/inputs/case_matrix.jsonl --limit 10`
- Wrote agent-authored review:
  - `test_artifacts/rag2_recall_quality/rag2_recall_quality_review.md`
  - Superseded: the initial review used expected-source evidence as the
    decision frame and was later reset after user feedback.

### 2026-05-23 blind IO correction

- Removed expected-source evaluation from the primary review surface.
- Replaced the contextual decision artifact with blind input/output review:
  - `test_artifacts/rag2_recall_quality/inputs/contextual_blind_io_comparison.jsonl`
  - `test_artifacts/rag2_recall_quality/contextual_blind_io_subagent_review.md`
- Spawned a subagent with the explicit instruction to read only the blind IO
  JSONL and not read expected sources, DB exports, prior reviews, source code,
  or the database.
- Blind result after mapping:
  - Current RAG2 baseline: 5 cases preferred.
  - Experiment-only improved search output: 7 cases preferred.
  - Tie: 0.
  - Unclear: 0.
- Reset `test_artifacts/rag2_recall_quality/rag2_recall_quality_review.md` to
  point to the blind IO review as the current decision surface.

### 2026-05-23 formatted blind IO rerun

- User challenged the previous conclusion that current RAG2 remained better
  for synthesis-heavy cases, noting that output-oriented formatting should
  address that weakness.
- Added experiment-only formatted improved output assembly:
  - `experiments/rag2_recall_quality/format_improved_outputs.py`
  - `test_artifacts/rag2_recall_quality/formatted_improved_search/*.json`
  - `test_artifacts/rag2_recall_quality/inputs/contextual_formatted_blind_io_comparison.jsonl`
- Formatter behavior:
  - preserves improved-search retrieval evidence;
  - adds a direct conclusion before evidence;
  - uses narrowed real context rows for group thread evidence flow;
  - formats negative no-evidence cases as a cautious no-evidence answer;
  - treats completion cues inside active milestone/objective rows as closure
    evidence instead of relying only on `status`.
- First formatted blind subagent run reported improved output preferred in
  12/12 cases, but flagged `rag2-context-006` because the formatter still
  labeled an active milestone as unfinished despite closure evidence.
- Fixed the state-label rule and reran the formatted blind review with a fresh
  subagent reading only the regenerated blind IO JSONL.
- Latest formatted blind result after mapping:
  - Current RAG2 baseline: 1 case preferred.
  - Formatted improved search output: 11 cases preferred.
  - Tie: 0.
  - Unclear: 0.
- Remaining loss:
  - `rag2-context-010`: current RAG2 gave a fuller broad nearby-conversation
    summary, while formatted improved output was cleaner but less complete for
    wider follow-up context.
- Wrote agent-authored debug review:
  - `test_artifacts/rag2_recall_quality/contextual_formatted_blind_io_subagent_review.md`
- Updated the primary review pointer:
  - `test_artifacts/rag2_recall_quality/rag2_recall_quality_review.md`

### 2026-05-23 second 10-case blind run

- User requested 10 more different examples and a readable `debug-llm` style
  blind test report.
- Pulled a fresh real-data export batch:
  - `test_artifacts/rag2_recall_quality/second_batch/private_673225019_recent.json`
  - `test_artifacts/rag2_recall_quality/second_batch/group_905393941_recent.json`
  - `test_artifacts/rag2_recall_quality/second_batch/user_673225019_memories_raw.json`
- Added experiment-only helpers:
  - `experiments/rag2_recall_quality/contextual_alt_case_matrix_builder.py`
  - `experiments/rag2_recall_quality/blind_io_randomizer.py`
- Updated `artifact_validation.py` to skip raw diagnostic DB exports under
  `second_batch/`, matching the existing `brainstorm/` behavior.
- Generated and ran 10 new contextual cases:
  - Command:
    `venv\Scripts\python -m experiments.rag2_recall_quality.contextual_alt_case_matrix_builder`
  - Command:
    `venv\Scripts\python -m experiments.rag2_recall_quality.run_case_batch --cases test_artifacts/rag2_recall_quality/inputs/contextual_case_matrix_alt10.jsonl --limit 10`
- Generated formatted outputs and randomized A/B labels:
  - `test_artifacts/rag2_recall_quality/inputs/contextual_alt10_formatted_io_unshuffled.jsonl`
  - `test_artifacts/rag2_recall_quality/inputs/contextual_alt10_blind_io.jsonl`
  - `test_artifacts/rag2_recall_quality/inputs/contextual_alt10_blind_mapping.json`
- Wrote a full Markdown raw IO appendix so the user does not need to read JSON:
  - `test_artifacts/rag2_recall_quality/contextual_alt10_blind_io_full_outputs.md`
- Wrote agent-authored debug review:
  - `test_artifacts/rag2_recall_quality/contextual_alt10_blind_review.md`
- Updated the primary review pointer:
  - `test_artifacts/rag2_recall_quality/rag2_recall_quality_review.md`
- Latest second-batch blind result after mapping:
  - Current RAG2 baseline: 2 cases preferred.
  - Formatted improved search output: 8 cases preferred.
  - Tie: 0.
  - Unclear: 0.
- Main residual issues:
  - improved search still under-synthesizes some correct evidence rows;
  - group-thread formatter can include unrelated nearby messages;
  - `rag2-alt-002` missed the separate precision-priority memory even though it
    found the Information Graph / GraphRAG memory.

### 2026-05-23 verification

- Compile command succeeded:
  `venv\Scripts\python.exe -m py_compile experiments/rag2_recall_quality/schema.py experiments/rag2_recall_quality/case_matrix_builder.py experiments/rag2_recall_quality/baseline_runner.py experiments/rag2_recall_quality/improved_search_runner.py experiments/rag2_recall_quality/compare_runs.py experiments/rag2_recall_quality/artifact_validation.py experiments/rag2_recall_quality/run_case_batch.py`
- Artifact validation succeeded:
  - Command:
    `venv\Scripts\python.exe -m experiments.rag2_recall_quality.artifact_validation --root test_artifacts/rag2_recall_quality`
  - Result:
    `case_count=10`, `json_artifact_count=32`, `status=valid`.
- Experiment test-file check returned no files:
  `Get-ChildItem -LiteralPath 'experiments/rag2_recall_quality' -Recurse -File | Where-Object { $_.Name -match '(^test_|_test\.py$)' }`
- Experiment DB-write scan found only the no-op guard assignments:
  `rg -n "insert_|insert_many|update_|delete_|drop_|create_index|create_collection|upsert|record_rag_stage_event|record_initializer_hit|prune_|purge_" experiments/rag2_recall_quality -g "*.py"`
- Current `git status --short` reports only plan/registry paths because
  `experiments/` and `test_artifacts/` are ignored in this repository:
  - `M development_plans/README.md`
  - `?? development_plans/active/short_term/graph_rag_recall_experiment_plan.md`
  - `?? development_plans/active/short_term/rag2_recall_quality_experiment_plan.md`
- Independent code review remains pending.

### 2026-05-23 second-batch verification

- Compile command succeeded:
  `venv\Scripts\python -m py_compile experiments\rag2_recall_quality\contextual_alt_case_matrix_builder.py experiments\rag2_recall_quality\blind_io_randomizer.py experiments\rag2_recall_quality\artifact_validation.py experiments\rag2_recall_quality\format_improved_outputs.py`
- Artifact validation succeeded:
  - Command:
    `venv\Scripts\python -m experiments.rag2_recall_quality.artifact_validation --root test_artifacts/rag2_recall_quality`
  - Result:
    `case_count=10`, `json_artifact_count=129`, `status=valid`.
- Second-batch artifact presence check succeeded:
  - `contextual_case_matrix_alt10.jsonl`: 10 rows.
  - `contextual_alt10_blind_io.jsonl`: 10 rows.
  - `contextual_alt10_blind_mapping.json`: 10 mapped cases.
  - `contextual_alt10_blind_review.md`: present.
  - `contextual_alt10_blind_io_full_outputs.md`: present.
- Experiment test-file check returned no files:
  `Get-ChildItem -LiteralPath 'experiments/rag2_recall_quality' -Recurse -File | Where-Object { $_.Name -match '(^test_|_test\.py$)' }`
- Experiment DB-write scan still found only the no-op guard assignments in
  `baseline_runner.py`.
- Current `git status --short` reports:
  - `M development_plans/README.md`
  - `?? development_plans/active/short_term/graph_rag_recall_experiment_plan.md`
  - `?? development_plans/active/short_term/rag2_recall_quality_experiment_plan.md`
- `git status --short --ignored experiments/rag2_recall_quality test_artifacts/rag2_recall_quality`
  confirms the experiment package and artifacts are ignored by git:
  - `!! experiments/rag2_recall_quality/`
  - `!! test_artifacts/`

### 2026-05-23 improved V2 same-input rerun

- User reviewed the second-batch blind report and corrected the judgment:
  - `rag2-alt-003` should prefer the improved output because it provides
    richer evidence and insight for cognition.
  - `rag2-alt-010` is the remaining weakness because improved search retrieved
    the right evidence but summarized unrelated neighboring group messages.
  - Improved output must add better evidence summarization, local
    second-precision timestamps, CQ-safe projection, and an explicit
    RAG-to-cognition contract check.
- Updated this plan with the corrected conclusion and improvement scope.
- Changed experiment-only search/formatting:
  - `experiments/rag2_recall_quality/improved_search_runner.py`
    continues user-memory keyword search across multiple literal anchors until
    the evidence limit is reached, so a broad topic hit does not hide a
    separate preference memory.
  - `experiments/rag2_recall_quality/format_improved_outputs.py`
    adds answer-oriented summaries, local second-precision timestamp
    projection, CQ stripping, `<image>...</image>` evidence blocks, and
    same-speaker image follow-up pruning.
- Reran the same 10 `contextual_case_matrix_alt10.jsonl` inputs:
  - Command:
    `venv\Scripts\python.exe -m experiments.rag2_recall_quality.improved_search_runner --cases test_artifacts/rag2_recall_quality/inputs/contextual_case_matrix_alt10.jsonl --limit 10`
  - Command:
    `venv\Scripts\python.exe -m experiments.rag2_recall_quality.format_improved_outputs --cases test_artifacts/rag2_recall_quality/inputs/contextual_case_matrix_alt10.jsonl --limit 10 --output test_artifacts/rag2_recall_quality/inputs/contextual_alt10_improved_v2_formatted_io_unshuffled.jsonl`
  - Command:
    `venv\Scripts\python.exe -m experiments.rag2_recall_quality.blind_io_randomizer --input test_artifacts/rag2_recall_quality/inputs/contextual_alt10_improved_v2_formatted_io_unshuffled.jsonl --output test_artifacts/rag2_recall_quality/inputs/contextual_alt10_improved_v2_blind_io.jsonl --mapping-output test_artifacts/rag2_recall_quality/inputs/contextual_alt10_improved_v2_blind_mapping.json --seed 2026052302`
- Wrote agent-authored debug review with full visible input and output in the
  report body:
  - `test_artifacts/rag2_recall_quality/contextual_alt10_improved_v2_blind_review.md`
- Updated the primary review pointer:
  - `test_artifacts/rag2_recall_quality/rag2_recall_quality_review.md`
- Latest same-input improved V2 blind-style result after mapping:
  - Current RAG2 baseline: 0 cases preferred.
  - Formatted improved search V2: 10 cases preferred.
  - Tie: 0.
  - Unclear: 0.
- Specific fixes observed:
  - `rag2-alt-002` now includes the separate precision-priority memory.
  - `rag2-alt-003` now has both direct answer and richer cognition-useful
    relationship evidence.
  - `rag2-alt-010` no longer includes unrelated Flipper messages and now keeps
    only the oxygen-sensor image plus same-speaker follow-up.
  - Formatted `rag2-alt-*` outputs no longer contain `[CQ:...]` or raw ISO UTC
    microsecond timestamps.
- Verification:
  - `venv\Scripts\python.exe -m py_compile experiments\rag2_recall_quality\format_improved_outputs.py experiments\rag2_recall_quality\improved_search_runner.py`
    succeeded.
  - `venv\Scripts\python.exe -m experiments.rag2_recall_quality.artifact_validation --root test_artifacts/rag2_recall_quality`
    succeeded with `case_count=10`, `json_artifact_count=132`,
    `status=valid`.
  - `Get-ChildItem -LiteralPath 'experiments/rag2_recall_quality' -Recurse -File | Where-Object { $_.Name -match '(^test_|_test\.py$)' }`
    returned no files.
  - `Get-ChildItem -LiteralPath 'test_artifacts/rag2_recall_quality/formatted_improved_search' -Filter 'rag2-alt-*.json' | Select-String -Pattern '\[CQ:|T\d\d:\d\d:\d\d\.'`
    returned no matches.

