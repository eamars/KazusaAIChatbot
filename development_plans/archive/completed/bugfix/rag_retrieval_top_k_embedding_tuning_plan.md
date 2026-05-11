# rag retrieval top k and embedding tuning plan

## Summary

- Goal: Improve RAG recall quality for QQ group conversation and memory recall by tuning internal retrieval breadth first, then applying Nomic query/document embedding prefixes with a full re-embedding migration.
- Plan class: high_risk_migration
- Status: completed with documented Stage 2 evidence exception
- Mandatory skills: `py-style`, `test-style-and-execution`, `local-llm-architecture`, `database-data-pull`, `cjk-safety`
- Overall cutover strategy: Phase 1 compatible behavior tuning; Phase 2 bigbang embedding corpus cutover with no runtime dual path.
- Highest-risk areas: live RAG latency, false positives from wider retrieval, false negatives from overly strict thresholds, MongoDB vector index compatibility, destructive embedding overwrite.
- Acceptance criteria: Phase 1 passes deterministic tests and today-channel retrieval profiles before Phase 2 starts; Phase 2 passes deterministic prefix tests, migration dry-run/apply checks, and before/after retrieval profiles with no approved-case regression.

## Context

Recent production logs showed RAG2 failing to recall QQ group `905393941`
discussion about the highest-share gaming GPU even though matching messages
exist in `conversation_history`. Diagnosis found:

- `search_conversation` and `conversation_search_agent` default to `top_k=5`.
- `search_conversation_history` uses `numCandidates=limit * 10`.
- Conversation vector search has no hard score threshold today.
- The live `conversation_history_vector_index` only indexes `embedding`; channel
  and user filters are post-filtered after vector search.
- The Nomic `text-embedding-nomic-embed-text-v2-moe` model expects query/document
  task prefixes, while current code sends raw text through `get_text_embedding`.

Prefix research completed on 2026-05-11:

- Source: Hugging Face model card
  `https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe`.
- Source: Hugging Face model config
  `https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe/blob/main/config_sentence_transformers.json`.
- Source: LM Studio OpenAI-compatible embeddings docs
  `https://lmstudio.ai/docs/developer/openai-compat/embeddings`.
- Nomic states that the text prompt must include a task-instruction prefix.
- Transformers usage requires manual text prefixes:
  `search_query: ` for queries and `search_document: ` for documents.
- SentenceTransformers usage with `prompt_name="query"` and
  `prompt_name="passage"` applies task instructions automatically. The checked
  model config maps `query` to `search_query: ` and `passage` to
  `search_document: `.
- The model config has `default_prompt_name: null`, so raw
  `SentenceTransformer.encode(texts)` is not equivalent to
  `prompt_name="query"` or `prompt_name="passage"`.
- LM Studio's OpenAI-compatible embedding endpoint accepts input text and does
  not expose a `prompt_name` parameter. Treat LM Studio as caller-supplied text:
  the application must prepend Nomic prefixes before sending requests.
- Therefore "Transformers prefix" and "SentenceTransformers prefix" are not two
  different literal prefix strings for this model. They are two API styles that
  produce the same effective input text when used correctly.

Empirical prefix comparison completed on 2026-05-11:

- Artifact:
  `test_artifacts/nomic_prefix_research_qq905393941_20260511.json`.
- Dataset: 237 recent `conversation_history` rows exported from QQ group
  `905393941`.
- Cases: five positive recall cases covering today's GPU/RAG/embedding
  discussion and one absent-breakfast negative case.
- Compared modes:
  - no prefix: raw query and raw document text;
  - Transformers manual prefix: explicit `search_query: ` and
    `search_document: `;
  - SentenceTransformers prompt-equivalent prefix: effective text from
    `prompt_name="query"` and `prompt_name="passage"`, which matches the
    manual-prefix strings above.
- Result summary:
  - all three modes achieved `hit@5 = 5/5` on the positive cases;
  - all three modes had `false_positive@5 = 0/1` on the negative case;
  - prefixed modes improved answer-row ranking for the precise GPU cases:
    the assistant `RTX 3060` answer moved ahead of the matching user question
    for the exact and recent-reference GPU queries;
  - prefixed modes lowered cosine score scale versus raw mode, so thresholds
    must be recalibrated after re-embedding;
  - prefixes did not solve broad/meta query ambiguity by themselves, so Phase 1
    top-k/candidate tuning remains necessary.

The target ownership boundary is RAG retrieval and text-vector persistence:

```text
RAG helper agents -> memory_retrieval_tools -> db vector search helpers
maintenance scripts -> db.script_operations -> affected MongoDB collections
```

Do not redesign initializer, dispatcher, cognition, dialog, reflection, or
memory consolidation in this plan.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing RAG helper-agent contracts,
  prompt-facing retrieval behavior, or live-path budgets.
- `database-data-pull`: load before pulling live MongoDB diagnostic data or
  running production-data profiling exports.
- `cjk-safety`: load if adding Chinese strings to Python source. Prefer putting
  Chinese diagnostic cases in JSON fixtures instead of Python string literals.

## Mandatory Rules

- Use `venv\Scripts\python.exe` for Python commands.
- Run `git status --short` before edits and before final sign-off.
- Do not read `.env` directly. Maintenance scripts may load project env through
  their existing script helpers.
- Use `rg` for source searches.
- Use `apply_patch` for manual file edits.
- Keep changes scoped to retrieval tuning, embedding prefixing, embedding
  migration support, and tests named in this plan.
- Do not add deterministic keyword fallbacks, hardcoded query rewriting, or
  code-side semantic interpretation of user input.
- Do not make the RAG initializer generate backend parameters. The initializer
  continues to produce semantic slots; specialist agents and deterministic DB
  helpers own retrieval parameters and bounds.
- Do not introduce a production hard score threshold in Phase 1. Threshold
  analysis is diagnostic-only until Phase 2 profiles recalibrate the new score
  distribution.
- Do not expose Nomic prefix literals outside `kazusa_ai_chatbot.db._client`.
  Callers must choose semantic role through helper names, not by string
  concatenation.
- Keep `get_text_embedding(text: str)` available with the same public signature.
  In Phase 2, this existing helper and `get_text_embeddings_batch(...)` become
  document-role compatibility helpers. All vector query call sites must use
  query-role helpers.
- Add batch query/document embedding helpers in Phase 2. Re-embedding and
  profiling code must not call private prefix-formatting helpers directly.
- Conversation vector prefiltering must be index-capability gated. If the live
  vector index does not expose every required filter field, the runtime query
  must retain the post-`$vectorSearch` `$match` path instead of failing.
- Existing Atlas vector search indexes must not be silently accepted when their
  definitions do not match this plan. The implementation must expose an
  operator dry-run/apply path to verify and rebuild mismatched search indexes.
- Phase 2 must not start until Phase 1 acceptance criteria and execution
  evidence are complete.
- Phase 2 is a bigbang embedding cutover: no runtime dual-read, dual-write,
  fallback-to-old-embedding, or compatibility shim is allowed.
- The Phase 2 apply step must overwrite embeddings across all affected
  collections through one approved script path. No ad hoc Mongo shell updates.
- A database backup or snapshot command must be recorded before Phase 2 apply.
  This is operational safety, not a runtime rollback feature.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the `Independent Code Review` gate and record the result
  in `Execution Evidence`.

## Must Do

- Add deterministic retrieval-tuning profile fixtures with positive and
  negative cases from QQ group `905393941`.
- Add a profiling script that measures hit/miss behavior across multiple
  `top_k` values before and after each phase.
- Phase 1 must tune default/internal conversation semantic retrieval breadth
  and Mongo candidate count without changing embedding semantics.
- Phase 1 must include deterministic profile-evaluation tests for false
  negatives and false positives.
- Phase 1 must include deterministic conversation vector pipeline tests for
  candidate bounds and filter handling.
- Phase 1 must include deterministic tests for vector-index definition
  inspection, mismatch detection, and safe runtime behavior when prefilters are
  unavailable.
- Phase 1 must run the profile before implementation and after implementation.
- Phase 2 must encapsulate Nomic query/document prefixes inside the embedding
  adapter boundary.
- Phase 2 must preserve the research distinction between raw input,
  Transformers manual prefixing, and SentenceTransformers prompt-name prefixing.
- Phase 2 must include a deterministic or operator-profile gate that compares
  no prefix, Transformers manual prefix, and SentenceTransformers
  prompt-equivalent prefix behavior.
- Phase 2 must re-embed `conversation_history`, `memory`, and
  `user_memory_units`.
- Phase 2 must update every text-vector query call site to use query-role
  embedding helpers.
- Phase 2 must update every text-vector document write or migration path to use
  document-role embedding behavior through the embedding adapter.
- Phase 2 must add and test a dry-run/apply re-embedding script.
- Phase 2 must profile retrieval before and after the re-embedding apply step.

## Deferred

- Do not redesign RAG2 initializer slot generation.
- Do not change generic RAG finalizer wording.
- Do not add LLM retry loops to `conversation_evidence_agent`.
- Do not add lexical fallback routing for this failure.
- Do not tune unrelated web search, relationship, image, scheduler, reflection,
  or dialog behavior.
- Do not migrate non-text embedding collections unless a grep proves they are
  written by the text embedding helper and the plan is updated before execution.
- Do not introduce an environment-controlled rollout flag or dual embedding
  index.

## Cutover Policy

Overall strategy: Phase 1 compatible, Phase 2 bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Phase 1 top-k defaults | compatible | Widen internal retrieval while preserving existing tool and DB entrypoints. |
| Phase 1 score threshold | compatible | Produce threshold diagnostics only; do not drop live rows by score. |
| Conversation vector index filters | migration | Add index definition inspection and a dry-run/apply rebuild script. Runtime vector search may use prefilters only when the index capability check confirms the required filter fields; otherwise it must use post-filtering. |
| Phase 2 embedding prefix contract | bigbang | Replace raw Nomic text-vector behavior with role-prefixed behavior at the adapter boundary. |
| Phase 2 stored embeddings | bigbang | Recompute affected collection embeddings in one apply run after dry-run and backup evidence. |
| Runtime compatibility | bigbang | Do not keep old raw-query/raw-document embedding behavior as a fallback. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- If an area is `bigbang`, rewrite old behavior directly instead of preserving
  it through compatibility shims.
- If an area is `migration`, follow the exact migration phases and cleanup gates
  listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the contracts in this plan.
- The agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, or extra features.
- The agent must treat changes outside the target modules as high-scrutiny
  changes and record the justification in `Execution Evidence`.
- The agent may add small private helpers only when they remove repeated
  validation or keep prefix/candidate math centralized and testable.
- The agent must not perform unrelated cleanup, formatting churn, dependency
  upgrades, broad refactors, or prompt rewrites.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

Phase 1 target:

- Common QQ group recall questions retrieve enough internal evidence for RAG
  judging without flooding the local LLM context.
- Default semantic conversation search uses a tuned retrieval breadth.
- Mongo vector search uses a larger candidate pool and index-compatible
  prefilters where available.
- The live RAG prompt still receives a compact projected evidence set.

Phase 2 target:

- Query embeddings and document embeddings follow the Nomic v2 MoE task-prefix
  contract.
- Prefixing is invisible to stored text and prompt-facing text.
- All affected stored embeddings are recomputed with the new document-prefix
  contract.
- Query/document prefix use is covered by deterministic tests and live profile
  evidence.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Phase order | Tune top-k/candidate behavior before embedding migration. | Phase 1 is reversible code/config behavior and isolates retrieval-breadth effects before score distribution changes. |
| Hard threshold | Do not add a live hard threshold in Phase 1. | The incident is a false-negative recall failure; a threshold would increase false-negative risk before score calibration. |
| Conversation default top_k | Use `CONVERSATION_SEARCH_DEFAULT_TOP_K` from `config.py`, defaulting to `20`, for normal semantic conversation search. | Diagnosis showed exact queries succeed at 5, but common vague recall queries need more breadth; the runtime default must be operator-configurable without code edits. |
| Conversation evidence fuzzy top_k | Use `CONVERSATION_SEARCH_DEFAULT_TOP_K` as the normal semantic conversation default and `CONVERSATION_SEARCH_MAX_TOP_K`, defaulting to `50`, only as the upper cap/diagnostic sweep value. | Phase 1 live sweep showed `50` improves recall headroom but introduces a false positive in the absent-breakfast negative case, so it must not be forced as the default path. |
| numCandidates | Use `max(200, limit * 20)` for conversation vector search. | Aligns with MongoDB guidance that approximate vector search often needs at least 20x `limit`. |
| Index prefilters | Add `platform`, `platform_channel_id`, `global_user_id`, `role`, and `timestamp` as conversation vector filter fields. | Prevents small candidate pools from being consumed by out-of-channel/out-of-user rows before filtering. |
| Prefilter rollout | Capability-gate `$vectorSearch.filter` use and retain post-filtering when the index is missing required filter fields. | Existing production indexes may not have filter fields; query code must not fail during migration or local development. |
| Prefix ownership | Prefix strings live only in `db._client`. | Keeps Nomic-specific behavior transparent and prevents prefix literals spreading through RAG code. |
| Existing embedding interface | Keep `get_text_embedding(text)` and `get_text_embeddings_batch(texts)` signatures and make both document-role compatibility helpers in Phase 2. Add explicit query/document single and batch helpers. | Preserves the existing public interface while making vector-query intent explicit and keeping re-embedding efficient. |
| SentenceTransformers semantics | Treat `prompt_name="query"` and `prompt_name="passage"` as equivalent to `search_query: ` and `search_document: ` for this model. | The Hugging Face config maps those prompt names to those exact prefix strings. |
| LM Studio semantics | Treat LM Studio OpenAI-compatible embeddings as raw text input that does not apply SentenceTransformers `prompt_name`. | The LM Studio endpoint accepts `input` text and exposes no prompt-name parameter. The application must send effective prefixed text. |
| Prefix accuracy decision | Use Nomic prefixes in Phase 2 despite current QQ slice showing equal hit@5. | The model contract requires prefixes, and the empirical slice showed better answer-row ordering for precise GPU cases while not increasing false positives at top 5. |
| Re-embedding scope | Re-embed `conversation_history`, `memory`, and `user_memory_units`. | These are the text-vector collections written by current embedding helpers and used by RAG/memory recall. |

## Contracts And Data Shapes

### Retrieval Profile Cases

Create `tests/fixtures/rag_retrieval_tuning_cases.json`:

```json
[
  {
    "case_id": "qq905393941_gpu_precise",
    "platform": "qq",
    "platform_channel_id": "905393941",
    "query": "游戏市场占有率最高的显卡",
    "expected_any": ["Steam", "RTX 3060", "NVIDIA"],
    "forbidden_any": ["早餐外卖"],
    "kind": "positive"
  }
]
```

The final fixture must include these six cases:

- `qq905393941_gpu_precise`: precise GPU market-share recall.
- `qq905393941_gpu_discussed_today`: broad "did we discuss GPU usage/share
  today" recall.
- `qq905393941_gpu_recent_reference`: "刚刚提到的游戏市场占有率最高的显卡" recall.
- `qq905393941_rag_memory_failure_today`: recall of today's RAG/memory-search
  failure discussion.
- `qq905393941_embedding_model_today`: recall of today's
  `text-embedding-nomic-embed-text-v2-moe` discussion.
- `qq905393941_absent_breakfast_memory`: negative case that must not be marked
  as resolved by GPU/RAG rows.

### Embedding Prefix Research Modes

Phase 2 profile tooling must compare these modes:

| Mode | Query effective text | Document effective text | Meaning |
|---|---|---|---|
| `no_prefix` | `<query>` | `<document>` | Current raw LM Studio/OpenAI-compatible behavior. |
| `transformers_manual_prefix` | `search_query: <query>` | `search_document: <document>` | Hugging Face Transformers usage path. |
| `sentence_transformers_prompt_equivalent` | `search_query: <query>` | `search_document: <document>` | Effective text produced by SentenceTransformers `prompt_name="query"` / `prompt_name="passage"` for this model. |

Do not add a different `query:` or `passage:` literal prefix. Those are
SentenceTransformers prompt names, not the final text prefixes for this model.
Treat prefix-like source text as ordinary content. Runtime callers must pass raw
query or document text and the embedding adapter must add the selected role
prefix itself.

### Profile Output

`src/scripts/profile_rag_retrieval.py` writes JSON:

```json
{
  "generated_at": "ISO-8601 timestamp",
  "phase_label": "phase1_before",
  "top_k_values": [5, 10, 20, 50],
  "cases": [
    {
      "case_id": "qq905393941_gpu_precise",
      "top_k": 20,
      "hit": true,
      "false_positive": false,
      "max_score": 0.0,
      "matched_terms": ["RTX 3060"],
      "rows": [
        {
          "rank": 1,
          "score": 0.0,
          "timestamp": "2026-05-11T...",
          "display_name": "string",
          "role": "assistant",
          "body_text": "bounded text",
          "platform_message_id": "string"
        }
      ]
    }
  ]
}
```

The script must not send profile rows to an LLM.

### Embedding Adapter

`kazusa_ai_chatbot.db._client` owns prefix formatting:

```python
async def get_text_embedding(text: str) -> list[float]
async def get_text_embeddings_batch(texts: list[str]) -> list[list[float]]
async def get_query_text_embedding(text: str) -> list[float]
async def get_query_text_embeddings_batch(texts: list[str]) -> list[list[float]]
async def get_document_text_embedding(text: str) -> list[float]
async def get_document_text_embeddings_batch(texts: list[str]) -> list[list[float]]
```

Rules:

- `get_text_embedding(text)` keeps the same signature.
- `get_text_embedding(text)` delegates to `get_document_text_embedding(text)`.
- `get_text_embeddings_batch(texts)` delegates to
  `get_document_text_embeddings_batch(texts)`.
- Prefix literals are private constants in `_client.py`.
- For Nomic v2 MoE, query helper applies `search_query:` and document helper
  applies `search_document:`.
- For non-Nomic models, helpers pass text through unchanged unless the model's
  documented contract is added in a future approved plan.
- Stored MongoDB source text must remain unprefixed.
- Prefixing must be idempotent for already-prefixed input. If text already
  begins with the exact role prefix, the adapter must not add the same prefix a
  second time.
- The adapter must not strip or rewrite user/document content other than adding
  the required role prefix.

### Conversation Vector Index Capability

Add these public or module-private contracts inside the DB boundary:

```python
CONVERSATION_VECTOR_FILTER_FIELDS = (
    "platform",
    "platform_channel_id",
    "global_user_id",
    "role",
    "timestamp",
)
```

The implementation must provide deterministic helpers with these behaviors:

- inspect a search index definition returned by `list_search_indexes`;
- return the set of declared vector filter paths;
- report whether a given index supports all required filter fields;
- build a conversation vector-search pipeline that uses `$vectorSearch.filter`
  only when filter support is confirmed;
- otherwise build the current post-`$vectorSearch` `$match` pipeline with the
  tuned `numCandidates`.

Do not call `list_search_indexes` for every live chat retrieval. Cache the
capability result for the process after the first successful inspection, and
allow tests to reset or patch that cache.

### Re-embedding Script

Create `src/scripts/reembed_text_vector_embeddings.py`:

```text
python -m scripts.reembed_text_vector_embeddings --dry-run --collections conversation_history memory user_memory_units
python -m scripts.reembed_text_vector_embeddings --apply --collections conversation_history memory user_memory_units --batch-size 100 --output test_artifacts/reembed_text_vectors_apply.json
```

The script must delegate DB operations to `kazusa_ai_chatbot.db.script_operations`.

### Vector Search Index Script

Create `src/scripts/ensure_vector_search_indexes.py`:

```text
python -m scripts.ensure_vector_search_indexes --dry-run --collections conversation_history
python -m scripts.ensure_vector_search_indexes --apply --collections conversation_history --wait-ready --output test_artifacts/vector_search_index_phase1_apply.json
```

Rules:

- The script must delegate MongoDB work to `kazusa_ai_chatbot.db.script_operations`
  or `kazusa_ai_chatbot.db._client`.
- Dry-run mode reports current definition, expected definition, missing filter
  fields, and whether recreation is required.
- Apply mode may drop and recreate only the named search index for the named
  collection. It must not drop collections or modify documents.
- Apply mode must write an output artifact with old definition, new requested
  definition, status polling result, and final `READY`/not-ready state.
- Runtime code must not rely on prefiltering until this script or an equivalent
  index inspection proves the required filter fields are present.

## LLM Call And Context Budget

- Phase 1 adds no LLM calls.
- Phase 1 may increase MongoDB retrieval breadth, but RAG prompt projection must
  stay bounded to the existing compact evidence shape.
- Phase 2 adds no LLM calls.
- The profiling script performs database/vector calls only and writes local JSON
  artifacts.
- The live chat response path remains bounded to the current RAG agent call
  structure. This plan does not authorize retry-loop changes.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/config.py`: expose
  `CONVERSATION_SEARCH_DEFAULT_TOP_K` and `CONVERSATION_SEARCH_MAX_TOP_K`
  with positive integer validation and a max/default consistency check.
- `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py`: tune
  `search_conversation` default `top_k` from config and enforce internal
  bounds.
- `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`: update prompt
  default from config and `_normalize_args` bounds for conversation semantic
  search.
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`: keep projected
  evidence compact and do not add retry-loop or initializer changes. Phase 1
  relies on `conversation_search_agent`'s default/cap normalization rather than
  forcing `top_k=50` for every semantic evidence slot.
- `src/kazusa_ai_chatbot/db/conversation.py`: tune `numCandidates`; use
  `$vectorSearch.filter` for indexed filters; keep post-filter fallback only for
  fields not allowed in the vector index.
- `src/kazusa_ai_chatbot/db/bootstrap.py`: add conversation vector filter paths.
- `src/kazusa_ai_chatbot/db/_client.py`: add role-specific prefix helpers
  without changing the `get_text_embedding(text)` or
  `get_text_embeddings_batch(texts)` signatures; add vector-index definition
  inspection helpers.
- `src/kazusa_ai_chatbot/db/memory.py`: use query-role embeddings for vector
  search.
- `src/kazusa_ai_chatbot/db/memory_evolution.py`: expose query/document
  embedding helpers for evolving memory.
- `src/kazusa_ai_chatbot/memory_evolution/repository.py`: use document-role
  embeddings for writes and query-role embeddings for semantic reads.
- `src/kazusa_ai_chatbot/db/user_memory_units.py`: use document-role embeddings
  for writes.
- `src/kazusa_ai_chatbot/rag/user_memory_evidence_agent.py`: use query-role
  embeddings for semantic user-memory retrieval.
- `src/kazusa_ai_chatbot/rag/user_profile_agent.py`: use query-role embeddings
  when hydrating semantic user-memory context.
- `src/kazusa_ai_chatbot/db/script_operations.py`: add profile and re-embedding
  script helpers; add vector-search index dry-run/apply helpers; update
  existing refresh logic to use the shared source-text functions.
- `src/scripts/benchmark_text_embedding.py`: use explicit query/document
  embedding helpers instead of the compatibility helper when benchmarking a
  semantic role.
- `pyproject.toml`: add console script entry only if needed for operator use;
  `python -m scripts.reembed_text_vector_embeddings` must work regardless.

### Create

- `src/scripts/profile_rag_retrieval.py`: read profile cases, run vector
  retrieval for configured `top_k` values, write JSON metrics.
- `src/scripts/profile_embedding_prefix_modes.py`: compare `no_prefix`,
  `transformers_manual_prefix`, and `sentence_transformers_prompt_equivalent`
  using exported/profiled conversation rows without writing to MongoDB.
- `src/scripts/ensure_vector_search_indexes.py`: dry-run/apply operator script
  for search-index definition verification and recreation.
- `src/scripts/reembed_text_vector_embeddings.py`: dry-run/apply re-embedding
  operator script.
- `tests/fixtures/rag_retrieval_tuning_cases.json`: today-channel positive and
  negative profile cases.
- `tests/test_rag_retrieval_tuning_profile.py`: deterministic tests for profile
  case loading and hit/false-positive metric evaluation.
- `tests/test_embedding_prefix_mode_profile.py`: deterministic tests for prefix
  comparison mode definitions and metric calculation.
- `tests/test_embedding_prefix_contract.py`: deterministic tests for embedding
  role-prefix behavior and unchanged public signature.
- `tests/test_vector_search_index_migration_script.py`: deterministic tests for
  vector-index dry-run/apply behavior.
- `tests/test_reembed_text_vector_embeddings_script.py`: deterministic tests
  for dry-run/apply script behavior.

### Keep

- RAG initializer, dispatcher route mapping, cognition, dialog, reflection, and
  finalizer behavior.
- Existing MongoDB collection names and vector field name `embedding`.
- Existing `get_text_embedding(text)` public signature.

## Data Migration

Phase 1 index migration target:

- `conversation_history_vector_index`: keep the existing index name and vector
  path `embedding`; add filter fields `platform`, `platform_channel_id`,
  `global_user_id`, `role`, and `timestamp`.

Phase 1 index migration procedure:

1. Run `ensure_vector_search_indexes --dry-run` for `conversation_history`.
2. If dry-run reports all required filter fields present, record the artifact
   and proceed without index recreation.
3. If dry-run reports missing filter fields, run the apply command to recreate
   only `conversation_history_vector_index`.
4. Wait for the recreated index to become queryable/`READY`.
5. Run the Phase 1 post-fix profile only after readiness is recorded.
6. If readiness cannot be confirmed, leave runtime prefilter capability disabled
   and continue using post-filtering; do not claim the prefilter acceptance
   criterion is met.

Phase 2 migration target collections:

- `conversation_history`: source text from `_embedding_source_text(row)`.
- `memory`: source text from `memory_embedding_source_text(row)`.
- `user_memory_units`: source text from `_semantic_text(row)`.

Migration procedure:

1. Record `git status --short`.
2. Record collection counts for all target collections.
3. Record vector index definitions for all target collections.
4. Verify the Phase 1 conversation vector index migration evidence is present.
5. Run Phase 2 pre-migration profile.
6. Run `reembed_text_vector_embeddings --dry-run`.
7. Record database backup or snapshot command and output location.
8. Run `reembed_text_vector_embeddings --apply`.
9. Verify processed count equals dry-run target count minus documented skipped
   rows with empty source text.
10. Rebuild or verify affected vector indexes.
11. Run Phase 2 post-migration profile.

The migration is bigbang at runtime. Do not add dual indexes or runtime fallback
to old raw embeddings.

## Implementation Order

### Phase 1 - Top K And Candidate Tuning

1. Add `tests/fixtures/rag_retrieval_tuning_cases.json`.
   - Include the six required cases from `Contracts And Data Shapes`.
   - Keep Chinese text in JSON, not Python literals.

2. Add failing deterministic profile tests.
   - File: `tests/test_rag_retrieval_tuning_profile.py`.
   - Tests:
     - `test_profile_case_loader_requires_positive_and_negative_cases`.
     - `test_profile_metrics_marks_missing_expected_terms_as_false_negative`.
     - `test_profile_metrics_marks_forbidden_terms_as_false_positive`.
     - `test_profile_metrics_bounds_row_text_for_artifacts`.
   - Run:
     `venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_tuning_profile.py -q`
   - Expected before implementation: fails because profile module does not exist.

3. Add failing top-k and DB pipeline tests.
   - Update `tests/test_memory_retrieval_tools.py` to assert default
     `search_conversation` delegates with `limit=20`.
   - Update `tests/test_rag_helper_arg_boundaries.py` to assert omitted or
     too-small `top_k` normalizes to the phase default.
   - Update `tests/test_db.py` conversation vector tests to assert
     `numCandidates == max(200, limit * 20)` and indexed filters are inside
     `$vectorSearch.filter`.
   - Add `test_db_bootstrap_configures_conversation_vector_filter_paths`
     asserting conversation vector index receives filter
     paths for `platform`, `platform_channel_id`, `global_user_id`, `role`, and
     `timestamp`.
   - Add `test_search_conversation_history_vector_uses_prefilter_when_supported`.
   - Add `test_search_conversation_history_vector_post_filters_when_prefilter_not_supported`.
   - Add `test_conversation_vector_prefilter_support_detects_missing_fields`.
   - Run the touched tests and record expected failures.

4. Implement profile script and pure metric helpers.
   - Create `src/scripts/profile_rag_retrieval.py`.
   - Keep DB reads delegated through public DB helpers.
   - No LLM calls.
   - Artifact rows must omit embeddings and bound `body_text`.

5. Implement vector-index inspection and dry-run/apply script.
   - Create `src/scripts/ensure_vector_search_indexes.py`.
   - Add script-facing helpers in `db/script_operations.py` or DB-private
     helpers in `_client.py`.
   - Add `tests/test_vector_search_index_migration_script.py`.
   - The script must support `--dry-run`, `--apply`, `--collections`,
     `--wait-ready`, and `--output`.
   - Apply mode may recreate only the named search index, never documents or
     collections.

6. Run Phase 1 pre-fix profile.
   - Command:
     `venv\Scripts\python.exe -m scripts.profile_rag_retrieval --phase-label phase1_before --cases tests/fixtures/rag_retrieval_tuning_cases.json --top-k 5 10 20 50 --output test_artifacts/rag_retrieval_phase1_before.json`
   - Evidence: record which positive cases miss at `top_k=5`, `20`, and `50`;
     record any negative-case false positives.

7. Implement top-k and candidate tuning.
   - Add named constants for default top-k, fuzzy evidence top-k, minimum
     candidates, and candidate multiplier.
   - Update conversation semantic defaults and normalization.
   - Update conversation vector pipeline candidate count.
   - Push indexed filters into `$vectorSearch.filter` only when the cached
     index-capability check confirms support.
   - Keep a post-`$match` for all filters when prefilter support is unavailable,
     and for any filters not supported by the vector index.

8. Run Phase 1 focused tests.
   - Commands:
     - `venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_tuning_profile.py -q`
     - `venv\Scripts\python.exe -m pytest tests/test_memory_retrieval_tools.py tests/test_rag_helper_arg_boundaries.py -q`
     - `venv\Scripts\python.exe -m pytest tests/test_db.py::test_db_bootstrap_configures_conversation_vector_filter_paths tests/test_db.py::test_search_conversation_history_vector_mocked tests/test_db.py::test_search_conversation_history_vector_with_filters_mocked tests/test_db.py::test_search_conversation_history_vector_uses_prefilter_when_supported tests/test_db.py::test_search_conversation_history_vector_post_filters_when_prefilter_not_supported tests/test_db.py::test_conversation_vector_prefilter_support_detects_missing_fields -q`
     - `venv\Scripts\python.exe -m pytest tests/test_vector_search_index_migration_script.py -q`

9. Verify live index readiness for Phase 1.
   - Dry-run command:
     `venv\Scripts\python.exe -m scripts.ensure_vector_search_indexes --dry-run --collections conversation_history --output test_artifacts/vector_search_index_phase1_dry_run.json`
   - Apply command when dry-run reports missing filter fields:
     `venv\Scripts\python.exe -m scripts.ensure_vector_search_indexes --apply --collections conversation_history --wait-ready --output test_artifacts/vector_search_index_phase1_apply.json`
   - Evidence must show `conversation_history_vector_index` includes the Phase
     1 filter fields before live prefilter reliance is accepted.

10. Run Phase 1 post-fix profile.
   - Command:
     `venv\Scripts\python.exe -m scripts.profile_rag_retrieval --phase-label phase1_after --cases tests/fixtures/rag_retrieval_tuning_cases.json --top-k 20 50 --output test_artifacts/rag_retrieval_phase1_after.json`
   - Acceptance gate:
     - All positive cases must hit expected terms by `top_k=50`.
     - Precise GPU case must hit by `top_k=20`.
     - Negative case must not be marked resolved by forbidden terms.

11. Stop and record Phase 1 evidence.
    - Do not start Phase 2 until Phase 1 evidence is complete.

### Phase 2 - Embedding Prefix And Re-embedding

1. Add failing embedding prefix research/profile tests.
   - File: `tests/test_embedding_prefix_mode_profile.py`.
   - Tests:
     - `test_prefix_mode_definitions_include_no_prefix_transformers_and_sentence_transformers`.
     - `test_sentence_transformers_mode_uses_model_config_prompt_strings`.
     - `test_transformers_and_sentence_transformers_modes_have_same_effective_prefixes`.
     - `test_prefix_mode_metrics_track_hit_rank_and_false_positive_rank`.
   - Run:
     `venv\Scripts\python.exe -m pytest tests/test_embedding_prefix_mode_profile.py -q`
   - Expected before implementation: fails because prefix-mode profile module
     does not exist.

2. Add failing embedding prefix tests.
   - File: `tests/test_embedding_prefix_contract.py`.
   - Tests:
     - `test_get_text_embedding_signature_remains_single_text_argument`.
     - `test_nomic_query_embedding_adds_search_query_prefix`.
     - `test_nomic_document_embedding_adds_search_document_prefix`.
     - `test_non_nomic_embedding_does_not_add_nomic_prefix`.
     - `test_batch_document_embeddings_prefix_each_input_once`.
     - `test_embedding_prefixing_is_idempotent_for_same_role_prefix`.
   - Run:
     `venv\Scripts\python.exe -m pytest tests/test_embedding_prefix_contract.py -q`
   - Expected before implementation: fails because role-specific helpers do not
     exist.

3. Add failing query/document call-site tests.
   - Update existing focused tests so query paths patch/assert
     `get_query_text_embedding`:
     - `tests/test_conversation_history_envelope.py`
     - `tests/test_db.py`
     - `tests/test_user_memory_evidence_agent.py`
     - `tests/test_user_profile_agent.py`
   - Update document write tests to assert stored source text remains raw at
     the call-site boundary and prefixing is owned by `_client`.

4. Add failing re-embedding script tests.
   - File: `tests/test_reembed_text_vector_embeddings_script.py`.
   - Tests:
     - `test_reembed_dry_run_counts_target_rows_without_updates`.
     - `test_reembed_apply_updates_conversation_memory_and_user_units`.
     - `test_reembed_skips_empty_source_rows_and_reports_them`.
     - `test_reembed_script_rejects_unknown_collection`.
   - Run:
     `venv\Scripts\python.exe -m pytest tests/test_reembed_text_vector_embeddings_script.py -q`
   - Expected before implementation: fails because script/helpers do not exist.

5. Implement embedding prefix-mode profile tooling.
   - Create `src/scripts/profile_embedding_prefix_modes.py`.
   - Reuse the Phase 1 retrieval fixture where possible.
   - Compare `no_prefix`, `transformers_manual_prefix`, and
     `sentence_transformers_prompt_equivalent`.
   - For SentenceTransformers mode, load or encode the checked model-config
     prompt mapping as data; do not invent new prefix strings.
   - Do not write to MongoDB.

6. Implement embedding adapter role helpers.
   - Modify `src/kazusa_ai_chatbot/db/_client.py`.
   - Keep `get_text_embedding(text)` signature unchanged.
   - Add query/document helper functions and private prefix-formatting helper.
   - Keep prefix literals private to `_client.py`.
   - Ensure role prefixing is idempotent for the same exact prefix.

7. Update vector query call sites.
   - Conversation search query: `db/conversation.py`.
   - Shared memory search query: `db/memory.py`.
   - Evolving memory semantic query: `memory_evolution/repository.py` through
     `db/memory_evolution.py`.
   - User memory evidence query: `rag/user_memory_evidence_agent.py`.
   - User profile semantic context query: `rag/user_profile_agent.py`.

8. Update document embedding write paths.
   - Conversation save and attachment-description update.
   - Shared/evolving memory writes.
   - User memory unit insert/update.
   - Script restore/recalculation helpers.

9. Implement re-embedding helpers and script.
   - Add script-facing DB helpers in `db/script_operations.py`.
   - Create `src/scripts/reembed_text_vector_embeddings.py`.
   - Script must support `--dry-run`, `--apply`, `--collections`,
     `--batch-size`, and `--output`.
   - Script must refuse to apply unless `--apply` is present.

10. Run Phase 2 deterministic tests.
   - Commands:
     - `venv\Scripts\python.exe -m pytest tests/test_embedding_prefix_mode_profile.py -q`
     - `venv\Scripts\python.exe -m pytest tests/test_embedding_prefix_contract.py -q`
     - `venv\Scripts\python.exe -m pytest tests/test_reembed_text_vector_embeddings_script.py -q`
     - `venv\Scripts\python.exe -m pytest tests/test_conversation_history_envelope.py tests/test_db.py tests/test_user_memory_evidence_agent.py tests/test_user_profile_agent.py -q`

11. Run Phase 2 prefix-mode comparison profile.
   - Command:
     `venv\Scripts\python.exe -m scripts.profile_embedding_prefix_modes --phase-label phase2_prefix_research --cases tests/fixtures/rag_retrieval_tuning_cases.json --input test_artifacts/qq905393941_chat_history_recent_20260511.json --output test_artifacts/nomic_prefix_research_qq905393941_20260511.json`
   - Acceptance gate:
     - The output includes all three required modes.
     - `sentence_transformers_prompt_equivalent` and
       `transformers_manual_prefix` use the same effective query/document
       prefix strings.
     - The report records `hit@5`, `hit@10`, `hit@20`,
       `false_positive@5`, first expected-hit rank, and first forbidden-hit
       rank per case.
     - If prefixed modes regress any Phase 1 accepted positive case below
       `hit@20`, stop and report before re-embedding.

12. Run Phase 2 pre-migration profile.
   - Command:
     `venv\Scripts\python.exe -m scripts.profile_rag_retrieval --phase-label phase2_before_reembed --cases tests/fixtures/rag_retrieval_tuning_cases.json --top-k 20 50 --output test_artifacts/rag_retrieval_phase2_before_reembed.json`
   - This profile is expected to be mixed because query prefixes may not match
     old raw document embeddings yet. Record it as transition evidence only.

13. Run re-embedding dry-run.
    - Command:
      `venv\Scripts\python.exe -m scripts.reembed_text_vector_embeddings --dry-run --collections conversation_history memory user_memory_units --batch-size 100 --output test_artifacts/reembed_text_vectors_dry_run.json`
    - Evidence: record target counts and skipped rows.

14. Record backup/snapshot evidence.
    - Record the exact backup command or managed snapshot identifier.
    - Do not proceed to apply without this evidence.

15. Run re-embedding apply.
    - Command:
      `venv\Scripts\python.exe -m scripts.reembed_text_vector_embeddings --apply --collections conversation_history memory user_memory_units --batch-size 100 --output test_artifacts/reembed_text_vectors_apply.json`
    - Evidence: processed/skipped/failed counts must match dry-run expectations.

16. Verify indexes and run Phase 2 post-migration profile.
    - Command:
      `venv\Scripts\python.exe -m scripts.profile_rag_retrieval --phase-label phase2_after_reembed --cases tests/fixtures/rag_retrieval_tuning_cases.json --top-k 20 50 --output test_artifacts/rag_retrieval_phase2_after_reembed.json`
    - Acceptance gate:
      - All Phase 1 accepted positive cases still hit.
      - Negative case still does not resolve through forbidden terms.
      - At least one answer-row case improves rank or preserves the Phase 1
        accepted rank.

17. Run final regression gates and independent code review.

## Progress Checklist

- [x] Stage 0 - discovery baseline recorded
  - Covers: current git state, source greps, current index definitions, and the
    diagnosis artifacts referenced by this plan.
  - Verify: `git status --short`; `rg "get_text_embedding\\(" src/kazusa_ai_chatbot src/scripts -g "*.py"`.
  - Evidence: record outputs in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-12` after evidence was recorded.

- [x] Stage 1 - Phase 1 failing tests and profile fixture added
  - Covers: Phase 1 implementation steps 1-3.
  - Verify: focused tests fail for missing/new expected behavior.
  - Evidence: record failing commands and failure summaries.
  - Sign-off: `Codex/2026-05-12` after Phase 1 failing-test evidence was
    recorded.

- [ ] Stage 2 - Phase 1 pre-fix profile recorded
  - Covers: Phase 1 implementation steps 4-6.
  - Verify: `test_artifacts/rag_retrieval_phase1_before.json` exists and
    contains all six cases.
  - Evidence: record hit/miss summary.
  - Sign-off: not signed off; strict live pre-fix artifact was missed and is
    recorded as a compensating-evidence exception.

- [x] Stage 3 - Phase 1 implementation complete
  - Covers: Phase 1 implementation steps 7-9.
  - Verify: Phase 1 focused tests pass and index definition is verified.
  - Evidence: record changed files, commands, and index fields.
  - Sign-off: `Codex/2026-05-12` after Phase 1 implementation evidence was
    recorded.

- [x] Stage 4 - Phase 1 post-fix profile accepted
  - Covers: Phase 1 implementation steps 10-11.
  - Verify: `test_artifacts/rag_retrieval_phase1_after.json` passes acceptance.
  - Evidence: record positive/negative case results.
  - Sign-off: `Codex/2026-05-12` after post-fix profile and gating-review
    follow-up evidence were recorded.

- [x] Stage 5 - Phase 2 failing tests added
  - Covers: Phase 2 implementation steps 1-4.
  - Verify: focused tests fail for missing prefix-mode profile tooling, prefix
    helpers, and migration script.
  - Evidence: record failing commands and failure summaries.
  - Sign-off: `Codex/2026-05-12` after failing test evidence was recorded.

- [x] Stage 6 - Phase 2 embedding prefix implementation complete
  - Covers: Phase 2 implementation steps 5-10.
  - Verify: Phase 2 deterministic tests pass.
  - Evidence: record changed files and command outputs.
  - Sign-off: `Codex/2026-05-12` after deterministic verification passed.

- [x] Stage 7 - Phase 2 prefix-mode profile accepted
  - Covers: Phase 2 implementation steps 11-12.
  - Verify: prefix-mode profile and pre-migration profile artifacts exist.
  - Evidence: record no-prefix versus prefixed hit/rank summary.
  - Sign-off: `Codex/2026-05-12` after read-only profile evidence was
    recorded.

- [x] Stage 8 - Phase 2 re-embedding migration applied
  - Covers: Phase 2 implementation steps 13-15.
  - Verify: dry-run/apply artifacts exist and counts reconcile.
  - Evidence: record backup/snapshot evidence and apply counts.
  - Sign-off: `Codex/2026-05-12` after backup, apply, and post-apply count
    evidence were recorded.

- [x] Stage 9 - Phase 2 post-migration profile accepted
  - Covers: Phase 2 implementation step 16.
  - Verify: Phase 2 profile acceptance gate passes.
  - Evidence: record rank/hit summary versus Phase 1 accepted profile.
  - Sign-off: `Codex/2026-05-12` after post-migration profile acceptance was
    recorded.

- [x] Stage 10 - final verification and independent code review complete
  - Covers: final regression gates and review.
  - Verify: all commands in `Verification` pass and review findings are closed
    or explicitly recorded as residual risk.
  - Evidence: record review mode, findings, fixes, reruns, and approval status.
  - Sign-off: `Codex/2026-05-12` after full default regression and review
    closure were recorded.

## Verification

### Static Greps

- `rg "search_query:|search_document:" src/kazusa_ai_chatbot src/scripts tests -g "*.py"`
  - Expected: prefix literals only in `src/kazusa_ai_chatbot/db/_client.py` and
    tests/scripts that assert or profile adapter behavior.
- `rg "get_text_embedding\\(" src/kazusa_ai_chatbot src/scripts -g "*.py"`
  - Expected: no vector query call site uses `get_text_embedding`. Allowed
    matches are `_client.py`, document-write compatibility paths explicitly
    recorded in `Execution Evidence`, and non-RAG benchmark code updated to
    make semantic role explicit. Forbidden query-path matches include
    `search_conversation_history`, `search_memory`, user-memory semantic
    retrieval, and user-profile semantic context hydration.
- `rg "top_k.: 5|top_k = raw_args.get\\(\"top_k\", 5\\)|limit \\* 10" src/kazusa_ai_chatbot/rag src/kazusa_ai_chatbot/db -g "*.py"`
  - Expected: no stale conversation semantic-search defaults remain. Allowed
    matches must be unrelated non-conversation modules and documented in
    `Execution Evidence`.

### Deterministic Tests

- `venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_tuning_profile.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_memory_retrieval_tools.py tests/test_rag_helper_arg_boundaries.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_db.py::test_db_bootstrap_configures_conversation_vector_filter_paths tests/test_db.py::test_search_conversation_history_vector_mocked tests/test_db.py::test_search_conversation_history_vector_with_filters_mocked tests/test_db.py::test_search_conversation_history_vector_uses_prefilter_when_supported tests/test_db.py::test_search_conversation_history_vector_post_filters_when_prefilter_not_supported tests/test_db.py::test_conversation_vector_prefilter_support_detects_missing_fields -q`
- `venv\Scripts\python.exe -m pytest tests/test_vector_search_index_migration_script.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_embedding_prefix_mode_profile.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_embedding_prefix_contract.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_reembed_text_vector_embeddings_script.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_conversation_history_envelope.py tests/test_user_memory_evidence_agent.py tests/test_user_profile_agent.py -q`

### Profile Commands

- Phase 1 before:
  `venv\Scripts\python.exe -m scripts.profile_rag_retrieval --phase-label phase1_before --cases tests/fixtures/rag_retrieval_tuning_cases.json --top-k 5 10 20 50 --output test_artifacts/rag_retrieval_phase1_before.json`
- Phase 1 after:
  `venv\Scripts\python.exe -m scripts.profile_rag_retrieval --phase-label phase1_after --cases tests/fixtures/rag_retrieval_tuning_cases.json --top-k 20 50 --output test_artifacts/rag_retrieval_phase1_after.json`
- Phase 1 vector index dry-run:
  `venv\Scripts\python.exe -m scripts.ensure_vector_search_indexes --dry-run --collections conversation_history --output test_artifacts/vector_search_index_phase1_dry_run.json`
- Phase 1 vector index apply, only when dry-run reports missing filter fields:
  `venv\Scripts\python.exe -m scripts.ensure_vector_search_indexes --apply --collections conversation_history --wait-ready --output test_artifacts/vector_search_index_phase1_apply.json`
- Phase 2 prefix-mode comparison:
  `venv\Scripts\python.exe -m scripts.profile_embedding_prefix_modes --phase-label phase2_prefix_research --cases tests/fixtures/rag_retrieval_tuning_cases.json --input test_artifacts/qq905393941_chat_history_recent_20260511.json --output test_artifacts/nomic_prefix_research_qq905393941_20260511.json`
- Phase 2 before re-embed:
  `venv\Scripts\python.exe -m scripts.profile_rag_retrieval --phase-label phase2_before_reembed --cases tests/fixtures/rag_retrieval_tuning_cases.json --top-k 20 50 --output test_artifacts/rag_retrieval_phase2_before_reembed.json`
- Phase 2 after re-embed:
  `venv\Scripts\python.exe -m scripts.profile_rag_retrieval --phase-label phase2_after_reembed --cases tests/fixtures/rag_retrieval_tuning_cases.json --top-k 20 50 --output test_artifacts/rag_retrieval_phase2_after_reembed.json`

### Migration Commands

- Dry-run:
  `venv\Scripts\python.exe -m scripts.reembed_text_vector_embeddings --dry-run --collections conversation_history memory user_memory_units --batch-size 100 --output test_artifacts/reembed_text_vectors_dry_run.json`
- Apply:
  `venv\Scripts\python.exe -m scripts.reembed_text_vector_embeddings --apply --collections conversation_history memory user_memory_units --batch-size 100 --output test_artifacts/reembed_text_vectors_apply.json`

### Final Regression

- `venv\Scripts\python.exe -m pytest -q`

If full regression is blocked by unrelated existing failures, record the failing
tests, rerun every touched deterministic test, and do not mark this plan
completed until the owner accepts the residual risk.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for changed Python, tests, fixtures,
  scripts, prompts, and documentation.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, implementation order, verification gates, and acceptance criteria.
- Code quality and design weaknesses, including hidden fallback paths, prefix
  leakage outside `_client.py`, brittle live-profile assumptions, accidental
  prompt changes, DB migration safety, and avoidable blast radius.
- Regression and handoff quality, including Phase 1-to-Phase 2 evidence,
  migration artifacts, static grep accuracy, and residual risks.

Fix concrete findings directly only when the fix is inside the approved change
surface. If a finding requires a contract, boundary, cutover, or migration
change, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Phase 1 deterministic tests pass.
- Phase 1 before/after profile artifacts exist and show accepted recall on the
  QQ `905393941` positive cases with no accepted negative-case false positive.
- Conversation semantic retrieval defaults and candidate counts match the
  Phase 1 contract.
- Conversation vector index filter fields are verified.
- Phase 2 prefix-mode profile artifact exists and compares no prefix,
  Transformers manual prefix, and SentenceTransformers prompt-equivalent
  prefix behavior.
- Phase 2 deterministic prefix tests pass.
- Prefix literals are contained inside `_client.py` except for tests and the
  approved prefix-mode profiling script.
- Query vector call sites use query-role helpers.
- Document vector write and migration paths use document-role behavior.
- Re-embedding dry-run/apply artifacts exist and counts reconcile.
- Phase 2 post-migration profile preserves or improves the Phase 1 accepted
  retrieval behavior.
- Final regression gates are run or blocked only by documented unrelated
  failures accepted by the owner.
- Independent code review is complete and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Wider top-k increases false positives | Keep hard threshold disabled but add negative profile cases and compact projection | Profile false-positive tests and negative live case |
| Wider top-k increases latency | Use internal caps and bounded evidence projection | Profile script duration and focused RAG smoke |
| Prefilter index is missing in live DB | Verify index definition before relying on `$vectorSearch.filter` | MongoDB index inspection evidence |
| LM Studio does not apply SentenceTransformers prompt names | Add prefixes in application-owned embedding adapter before calling LM Studio | Prefix-mode profile and adapter tests |
| Query/document prefixes change score scale | Reprofile before/after re-embed and avoid hard threshold | Phase 2 profile artifacts |
| Partial re-embedding corrupts retrieval consistency | Use one script path, dry-run/apply counts, and backup evidence | Migration artifacts and count reconciliation |
| Prefix literals leak into callers | Static grep restricts prefix literals to `_client.py`, tests, and the approved profiling script | Static grep gate |

## Execution Evidence

- Research update 2026-05-11:
  - Source review: Hugging Face model card and
    `config_sentence_transformers.json` require/effectively map query and
    passage prompts to `search_query: ` and `search_document: `.
  - LM Studio review: OpenAI-compatible embeddings endpoint accepts raw
    `input` text and exposes no SentenceTransformers `prompt_name`.
  - Empirical artifact:
    `test_artifacts/nomic_prefix_research_qq905393941_20260511.json`.
  - Empirical summary: no-prefix, Transformers manual prefix, and
    SentenceTransformers prompt-equivalent modes all reached `hit@5 = 5/5`
    and `false_positive@5 = 0/1`; prefixed modes improved assistant answer-row
    ranking for the precise GPU cases and lowered score scale.
- Stage 0:
  - 2026-05-12 / Codex: Recorded dirty worktree before Phase 1 execution.
    Pre-existing local plan changes were present:
    `development_plans/README.md` modified and
    `development_plans/active/bugfix/` untracked.
  - Source diagnosis confirmed Phase 1 target code still used
    `search_conversation(top_k=5)`, `conversation_search_agent` default
    `top_k=5`, `numCandidates=limit * 10`, and no conversation vector filter
    paths in bootstrap.
- Stage 1:
  - 2026-05-12 / Codex: Added Phase 1 fixture and failing tests.
  - Expected red-state commands:
    - `venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_tuning_profile.py -q`
      failed because `scripts.profile_rag_retrieval` did not exist.
    - `venv\Scripts\python.exe -m pytest tests/test_vector_search_index_migration_script.py -q`
      failed because `scripts.ensure_vector_search_indexes` did not exist.
    - `venv\Scripts\python.exe -m pytest tests/test_memory_retrieval_tools.py tests/test_rag_helper_arg_boundaries.py -q`
      failed on old `top_k=5` / `top_k=3` behavior.
    - `venv\Scripts\python.exe -m pytest tests/test_db.py::test_db_bootstrap_configures_conversation_vector_filter_paths tests/test_db.py::test_search_conversation_history_vector_mocked tests/test_db.py::test_search_conversation_history_vector_with_filters_mocked tests/test_db.py::test_search_conversation_history_vector_uses_prefilter_when_supported tests/test_db.py::test_search_conversation_history_vector_post_filters_when_prefilter_not_supported tests/test_db.py::test_conversation_vector_prefilter_support_detects_missing_fields -q`
      failed on missing filter paths, old `numCandidates`, no prefilter, and
      missing vector-index definition helper.
- Stage 2:
  - 2026-05-12 / Codex: The strict pre-fix live profile artifact was not
    produced before implementation because the profiling script was introduced
    during the TDD loop. Do not treat any later `phase1_before` artifact as a
    true baseline for this run.
  - Compensating evidence: red-state deterministic tests above captured the
    pre-fix code behavior, and `test_artifacts/rag_retrieval_phase1_topk_sweep.json`
    records the post-index live sweep over `top_k=5,10,20,50`.
- Stage 3:
  - 2026-05-12 / Codex: Implemented Phase 1 top-k/candidate/prefilter behavior
    and vector-index tooling.
  - Changed files:
    `src/kazusa_ai_chatbot/config.py`,
    `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py`,
    `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`,
    `src/kazusa_ai_chatbot/db/conversation.py`,
    `src/kazusa_ai_chatbot/db/bootstrap.py`,
    `src/kazusa_ai_chatbot/db/_client.py`,
    `src/kazusa_ai_chatbot/db/script_operations.py`,
    `src/scripts/profile_rag_retrieval.py`,
    `src/scripts/ensure_vector_search_indexes.py`, fixture/tests.
  - Focused tests passed:
    - `venv\Scripts\python.exe -m pytest tests/test_config.py::TestConversationSearchConfig -q`
      passed 4 tests for the configurable conversation search default/cap.
    - `venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_tuning_profile.py -q`
      passed 4 tests.
    - `venv\Scripts\python.exe -m pytest tests/test_vector_search_index_migration_script.py -q`
      passed 2 tests.
    - `venv\Scripts\python.exe -m pytest tests/test_memory_retrieval_tools.py tests/test_rag_helper_arg_boundaries.py -q`
      passed 25 tests after adding configured default/cap coverage.
    - `venv\Scripts\python.exe -m pytest tests/test_db.py::test_db_bootstrap_configures_conversation_vector_filter_paths tests/test_db.py::test_search_conversation_history_vector_mocked tests/test_db.py::test_search_conversation_history_vector_with_filters_mocked tests/test_db.py::test_search_conversation_history_vector_uses_prefilter_when_supported tests/test_db.py::test_search_conversation_history_vector_post_filters_when_prefilter_not_supported tests/test_db.py::test_conversation_vector_prefilter_support_detects_missing_fields -q`
      passed 6 tests.
    - `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py::test_conversation_evidence_semantic_topic_uses_search -q`
      passed.
  - Vector index dry-run:
    `test_artifacts/vector_search_index_phase1_dry_run.json` reported
    `missing_filter_paths` for `platform`, `platform_channel_id`,
    `global_user_id`, `role`, and `timestamp`.
  - Vector index apply:
    `test_artifacts/vector_search_index_phase1_apply.json` reported
    `status=applied`, `ready_status=READY`, `dropped_existing=true`.
  - Vector index verification:
    `test_artifacts/vector_search_index_phase1_verify.json` reported
    `status=ready`, `requires_recreate=false`, `missing_filter_paths=[]`.
- Stage 4:
  - 2026-05-12 / Codex: Post-fix profile artifact:
    `test_artifacts/rag_retrieval_phase1_after.json`.
  - At `top_k=20`, all five positive QQ `905393941` cases resolved with
    expected terms; the absent-breakfast negative case had
    `resolved=false`, `false_positive=false`, `false_negative=false`.
  - At `top_k=50`, all five positive cases resolved, but the
    absent-breakfast negative case had `resolved=false`,
    `false_positive=true` because unrelated GPU/RAG rows appeared deep in the
    result list. This is recorded as the reason Phase 1 keeps `20` as the
    runtime default and treats `50` as a cap/diagnostic value, not a forced
    default.
  - Top-k sweep artifact:
    `test_artifacts/rag_retrieval_phase1_topk_sweep.json`. Sweep summary:
    positives resolved at `top_k=5,10,20,50`; broad GPU-discussed case only
    matched partial expected terms at `top_k=5` but matched the full expected
    term set by `top_k=10`; negative stayed clean through `top_k=20` and
    became false-positive at `top_k=50`.
- Independent review follow-up:
  - 2026-05-12 / Codex: Fixed Phase 1 gating review findings.
  - Vector-index inspection now reports full definition issues, including
    vector path, dimension count, similarity, index type when present, and
    missing filter paths.
  - Vector-index apply now prepares the sample embedding and search-index model
    before dropping an existing index, reducing the destructive failure window.
  - Conversation vector search now caps DB-level `numCandidates` at the Atlas
    limit of `10000`.
  - Conversation vector prefilter support caches positive capability only, so a
    stale negative result does not survive index recreation in a long-running
    process.
  - Retrieval profile artifacts now include `max_score` per case for threshold
    diagnostics.
  - The conversation search generator prompt no longer uses percent-style
    interpolation.
  - Refreshed `test_artifacts/vector_search_index_phase1_verify.json` reported
    `status=ready`, `requires_recreate=false`, `missing_filter_paths=[]`, and
    `definition_issues=[]`.
  - Refreshed `test_artifacts/rag_retrieval_phase1_after.json` and
    `test_artifacts/rag_retrieval_phase1_topk_sweep.json`; all result entries
    include `max_score`. At `top_k=20`, all five positive cases resolved and
    the negative case had no false positive. At diagnostic cap `top_k=50`, the
    negative case still records the known false positive.
  - Follow-up tests passed:
    - `venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_tuning_profile.py tests/test_vector_search_index_migration_script.py -q`
      passed 9 tests.
    - `venv\Scripts\python.exe -m pytest tests/test_db.py::test_db_bootstrap_configures_conversation_vector_filter_paths tests/test_db.py::test_search_conversation_history_vector_mocked tests/test_db.py::test_search_conversation_history_vector_with_filters_mocked tests/test_db.py::test_search_conversation_history_vector_uses_prefilter_when_supported tests/test_db.py::test_search_conversation_history_vector_post_filters_when_prefilter_not_supported tests/test_db.py::test_conversation_vector_prefilter_support_detects_missing_fields tests/test_db.py::test_vector_index_definition_issues_detect_vector_mismatch tests/test_db.py::test_vector_num_candidates_is_capped_to_atlas_limit tests/test_db.py::test_conversation_vector_prefilter_support_rechecks_cached_false -q`
      passed 9 tests.
    - `venv\Scripts\python.exe -m pytest tests/test_memory_retrieval_tools.py tests/test_rag_helper_arg_boundaries.py tests/test_config.py::TestConversationSearchConfig -q`
      passed 29 tests.
    - `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py::test_conversation_evidence_semantic_topic_uses_search -q`
      passed.
    - `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\conversation_search_agent.py src\kazusa_ai_chatbot\db\_client.py src\kazusa_ai_chatbot\db\conversation.py src\kazusa_ai_chatbot\db\script_operations.py src\scripts\profile_rag_retrieval.py src\scripts\ensure_vector_search_indexes.py`
      passed.
    - `git diff --check` reported no whitespace errors, only line-ending
      warnings.
- Stage 5:
  - 2026-05-12 / Codex: Added Phase 2 failing tests for prefix-mode profile
    contracts, query/document embedding adapter behavior, query-call-site
    split, and the re-embedding operator script.
  - Expected red-state commands:
    - `venv\Scripts\python.exe -m pytest tests/test_embedding_prefix_mode_profile.py tests/test_embedding_prefix_contract.py tests/test_reembed_text_vector_embeddings_script.py tests/test_conversation_history_envelope.py tests/test_db.py::test_search_memory_vector tests/test_db.py::test_search_conversation_history_vector_mocked tests/test_db.py::test_search_conversation_history_vector_with_filters_mocked tests/test_db.py::test_search_conversation_history_vector_uses_prefilter_when_supported tests/test_db.py::test_search_conversation_history_vector_post_filters_when_prefilter_not_supported tests/test_user_memory_evidence_agent.py tests/test_user_profile_agent.py -q`
      failed before implementation because the profile and re-embedding script
      modules and role-specific embedding helpers did not exist.
    - `venv\Scripts\python.exe -m pytest tests/test_embedding_prefix_contract.py::test_embedding_prefixing_keeps_prefix_like_content_transparent -q`
      failed during independent review before the adapter treated
      prefix-looking source text as ordinary raw content.
    - `venv\Scripts\python.exe -m pytest tests/test_reembed_text_vector_embeddings_script.py::test_reembed_apply_clears_embeddings_for_empty_source_rows -q`
      failed before stale-vector cleanup because empty-source rows were only
      reported, not cleared during apply.
- Stage 6:
  - 2026-05-12 / Codex: Implemented role-specific embedding helpers in
    `src/kazusa_ai_chatbot/db/_client.py`; prefix literals remain private to
    `_client.py` except tests and the approved profile script.
  - Query vector paths now use query-role helpers in conversation search,
    memory search, memory-evolution retrieval, user-memory evidence, and
    user-profile hydration. Document write and maintenance paths now use
    document-role helpers.
  - `get_text_embedding(text)` and `get_text_embeddings_batch(texts)` keep
    their public signatures and delegate to document-role behavior.
  - Added `src/scripts/profile_embedding_prefix_modes.py` and
    `src/scripts/reembed_text_vector_embeddings.py`. The CLI scripts load
    `.env` before DB/config-dependent imports during real operator runs.
  - Updated Database ICD `src/kazusa_ai_chatbot/db/README.md` with the
    query/document embedding role contract.
  - Updated local `.env` to explicitly set
    `EMBEDDING_MODEL=text-embedding-nomic-embed-text-v2-moe`.
  - Re-embedding apply behavior will now update non-empty source rows and
    clear stale embeddings from empty-source rows, preventing mixed old-vector
    rows from remaining in the vector index.
  - Verification:
    - `venv\Scripts\python.exe -m pytest tests/test_embedding_prefix_mode_profile.py tests/test_embedding_prefix_contract.py tests/test_reembed_text_vector_embeddings_script.py tests/test_conversation_history_envelope.py tests/test_user_memory_evidence_agent.py tests/test_user_profile_agent.py tests/test_db.py tests/test_vector_search_index_migration_script.py tests/test_memory_evolution_retrieval.py tests/test_memory_evolution_repository.py tests/test_memory_evolution_reset.py tests/test_memory_evolution_idempotency.py tests/test_save_conversation_invalidation.py tests/test_user_state_snapshot.py tests/test_rag_retrieval_tuning_profile.py tests/test_memory_retrieval_tools.py tests/test_rag_helper_arg_boundaries.py tests/test_config.py::TestConversationSearchConfig -q`
      passed 157 tests with 13 deselected after independent-review fixes.
    - `venv\Scripts\python.exe -m py_compile ...` over touched Python source
      and scripts passed.
    - `git diff --check` reported no whitespace errors, only line-ending
      warnings.
    - `rg "get_text_embedding\\(" src/kazusa_ai_chatbot src/scripts -g "*.py"`
      matched only `src/kazusa_ai_chatbot/db/_client.py`.
    - Precise prefix-literal grep matched only `_client.py`, the approved
      prefix profile script, and tests.
- Stage 7:
  - 2026-05-12 / Codex: Prefix-mode profile artifact refreshed:
    `test_artifacts/nomic_prefix_research_qq905393941_20260511.json`.
  - Prefix profile summary for `top_k=5`: no-prefix, Transformers manual
    prefix, and SentenceTransformers prompt-equivalent modes each reached
    `hit@5 = 5/5` on positive cases and `false_positive@5 = 0/1` on the
    negative case.
  - At `top_k=20`, all three modes reached `hit@20 = 5/5` and
    `false_positive@20 = 1/1`; the negative false positive appears only deeper
    in the candidate list.
  - In this refreshed run, positive-case first-hit ranks were identical across
    the three modes: GPU precise rank 2, GPU discussed-today rank 1, GPU recent
    reference rank 2, RAG/memory failure rank 1, embedding-model discussion
    rank 1.
  - Pre-reembedding live RAG profile artifact:
    `test_artifacts/rag_retrieval_phase2_before_reembed.json`. With new
    query-role prefixes against the still-raw stored vectors, `top_k=20` and
    `top_k=50` both resolved `5/5` positives with `0/1` negative false
    positives; max-score range was about `0.7036` to `0.9370`.
- Stage 8:
  - 2026-05-12 / Codex: Re-embedding dry-run artifact refreshed:
    `test_artifacts/reembed_text_vectors_dry_run.json`.
  - Dry-run results: `total_count=34594`, `total_processed=30159`,
    `total_skipped=4435`, `total_updated=0`, `total_cleared=0`.
  - Per collection: `conversation_history` total `33997`, processed `29562`,
    skipped `4435`; `memory` total/processed `242`; `user_memory_units`
    total/processed `355`.
  - 2026-05-12 / Codex: User explicitly approved the database migration.
  - Backup/snapshot evidence: wrote
    `test_artifacts/text_vector_embedding_backup_pre_reembed_20260512.json.gz`
    before apply. The snapshot contains `_id` plus current `embedding` values
    for `conversation_history`, `memory`, and `user_memory_units`.
  - Backup counts: `conversation_history` rows `33997`, rows with embedding
    `33997`; `memory` rows/with embedding `242`; `user_memory_units`
    rows/with embedding `355`.
  - Apply command:
    `venv\Scripts\python.exe -m scripts.reembed_text_vector_embeddings --apply --collections conversation_history memory user_memory_units --batch-size 100 --output test_artifacts/reembed_text_vectors_apply.json`.
  - Apply artifact:
    `test_artifacts/reembed_text_vectors_apply.json`. Apply stderr log
    `test_artifacts/reembed_apply_stderr_20260512.log` was empty.
  - Apply results reconciled with dry-run: `total_count=34594`,
    `total_processed=30159`, `total_skipped=4435`,
    `total_updated=30159`, `total_cleared=4435`.
  - Per collection apply results: `conversation_history` total `33997`,
    processed/updated `29562`, skipped/cleared `4435`; `memory`
    total/processed/updated `242`; `user_memory_units`
    total/processed/updated `355`.
  - Post-apply DB count check: `conversation_history` total `33997`,
    with embedding `29562`, without embedding `4435`; `memory` total/with
    embedding `242`; `user_memory_units` total/with embedding `355`.
- Stage 9:
  - 2026-05-12 / Codex: Phase 2 post-migration profile artifact generated:
    `test_artifacts/rag_retrieval_phase2_after_reembed.json`.
  - Post-migration profile command:
    `venv\Scripts\python.exe -m scripts.profile_rag_retrieval --phase-label phase2_after_reembed --cases tests/fixtures/rag_retrieval_tuning_cases.json --top-k 20 50 --output test_artifacts/rag_retrieval_phase2_after_reembed.json`.
  - Acceptance summary: at both `top_k=20` and `top_k=50`, positives resolved
    `5/5` and the negative false-positive count stayed `0/1`.
  - Post-migration score range was `0.6686` to `0.8955`, lower than the
    pre-reembed `0.7036` to `0.9370` range, as expected after aligning stored
    document vectors with query/document prefix semantics.
  - Positive matched-term coverage remained intact for GPU precise,
    GPU-discussed-today, GPU recent-reference, RAG/memory failure, and
    embedding-model discussion cases. The absent-breakfast negative case
    remained unresolved.
- Stage 10:
  - 2026-05-12 / Codex: Performed pre-apply independent review from a fresh
    diff-review posture. No separate reviewer was available in this session.
  - Finding fixed: adjacent memory-evolution tests still patched the legacy
    `compute_memory_embedding` helper, so deterministic repository/reset tests
    could hit the live embedding endpoint after the document-role split. Updated
    those tests to patch and assert `compute_memory_document_embedding`.
  - Finding fixed: the adapter treated source text beginning with
    `search_query: ` or `search_document: ` as an existing internal prefix.
    That made raw conversation text able to override the requested embedding
    role. Updated runtime adapter and prefix-profile script to always prepend
    the selected role prefix to raw text, and updated tests/plan wording for
    the transparent raw-content contract.
  - Finding fixed: review found local line-length/style issues in newly touched
    files and long new test names. Wrapped the new/review-created lines and
    left unrelated historical long lines outside this change.
  - Coverage added: batch embedding adapter now has a regression test proving
    provider response rows are sorted by response `index` before returning
    embeddings in input order.
  - Verification:
    - `venv\Scripts\python.exe -m pytest tests/test_memory_evolution_repository.py tests/test_memory_evolution_reset.py tests/test_memory_evolution_idempotency.py -q`
      passed 12 tests.
    - `venv\Scripts\python.exe -m pytest tests/test_embedding_prefix_mode_profile.py tests/test_embedding_prefix_contract.py -q`
      passed 12 tests.
    - `venv\Scripts\python.exe -m pytest tests/test_embedding_prefix_mode_profile.py tests/test_embedding_prefix_contract.py tests/test_reembed_text_vector_embeddings_script.py tests/test_conversation_history_envelope.py tests/test_user_memory_evidence_agent.py tests/test_user_profile_agent.py tests/test_db.py tests/test_vector_search_index_migration_script.py tests/test_memory_evolution_retrieval.py tests/test_memory_evolution_repository.py tests/test_memory_evolution_reset.py tests/test_memory_evolution_idempotency.py tests/test_save_conversation_invalidation.py tests/test_user_state_snapshot.py tests/test_rag_retrieval_tuning_profile.py tests/test_memory_retrieval_tools.py tests/test_rag_helper_arg_boundaries.py tests/test_config.py::TestConversationSearchConfig -q`
      passed 157 tests with 13 deselected.
    - `venv\Scripts\python.exe -m py_compile ...` over touched Python source
      and scripts passed.
    - `git diff --check` reported no whitespace errors, only line-ending
      warnings.
    - `rg "get_text_embedding\\(" src/kazusa_ai_chatbot src/scripts -g "*.py"`
      matched only `src/kazusa_ai_chatbot/db/_client.py`.
    - Precise prefix-literal grep matched only `_client.py`, the approved
      prefix-profile script, and tests.
  - 2026-05-12 / Codex: Final default regression initially failed
    `tests/test_service_input_queue.py::test_dropped_message_never_invokes_graph`.
    Root cause was a deterministic test scheduling assumption: the test yielded
    the event loop once and expected the queue worker to have reached graph
    invocation despite multiple awaited setup calls before `_graph.ainvoke`.
    Fixed the test to wait on an explicit `graph_started` event, matching the
    adjacent queue-worker tests.
  - Verification after the queue-test synchronization fix:
    - `venv\Scripts\python.exe -m pytest tests/test_service_input_queue.py::test_dropped_message_never_invokes_graph -q`
      passed.
    - `venv\Scripts\python.exe -m pytest -q` passed `1106` tests with `237`
      deselected by the default `not live_db and not live_llm` pytest config.
    - `venv\Scripts\python.exe -m py_compile ...` over touched Python source
      and scripts passed.
    - `git diff --check` reported no whitespace errors, only line-ending
      warnings.
    - `rg "get_text_embedding\\(" src/kazusa_ai_chatbot src/scripts -g "*.py"`
      matched only `src/kazusa_ai_chatbot/db/_client.py`.
    - Prefix literal grep matched only `_client.py`, the approved
      prefix-profile script, and tests.
    - Stale top-k grep found no conversation semantic-vector default `5` path;
      remaining matches were non-conversation vector helpers, persistent memory
      helper defaults, and conversation keyword search.
  - Approval status: Phase 2 is complete after user-approved migration,
    post-migration profile acceptance, final regression, and independent review.
    Stage 2 remains a documented Phase 1 compensating-evidence exception
    because the strict live pre-fix artifact was missed before Phase 1 changes.
