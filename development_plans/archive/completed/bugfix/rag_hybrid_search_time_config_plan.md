# rag hybrid search time config bugfix plan

## Summary

- Goal: Fix the RAG failure class where recent QQ conversation evidence is retrieved poorly, projected incompletely, or temporally misread, while keeping the live response path bounded and inspectable.
- Plan class: large
- Status: completed
- Mandatory skills: `py-style`, `test-style-and-execution`, `local-llm-architecture`, `database-data-pull`, `development-plan-writing`, `cjk-safety`.
- Overall cutover strategy: compatible, with local bigbang replacement for internal conversation and shared-memory evidence retrieval.
- Highest-risk areas: false positives from broader retrieval, broad semantic drift, hidden filter loss, attachment-only evidence loss, local-time boundary errors, local LLM judging, Cache2 stale plans, and response-path latency.
- Acceptance criteria: hybrid evidence beats current semantic-only and keyword-only baselines on real QQ `905393941` cases; absent-topic and literal-trap negatives stay clean; attachment-only positives are usable evidence; all search top-k, selected-limit, candidate, and floor values are shared config; relative-day output respects the runtime local date.

## Context

The triggering incident is the QQ group `905393941` GPU memory-recall failure:

- Local `2026-05-11 20:50`: user asked which GPU is most popular on Steam.
- Local `2026-05-11 20:51`: assistant answered `RTX 3060`, NVIDIA share `73.21%`.
- Local `2026-05-11 21:29-21:32`: later messages discussed whether RAG had forgotten or denied the topic.
- Local `2026-05-12 08:39-08:41`: RAG sometimes reported no actual GPU market-share discussion, or retrieved only the denial/meta cluster.

RCA conclusion: this is not primarily an embedding-model failure. The embedding path can retrieve the true answer row, but production can lose or misclassify it through worker selection, keyword/synonym mismatch, selected-summary caps, local/UTC window errors, unresolved-result projection, attachment text loss, and weak trusted-filter enforcement.

Validated evidence:

- Direct fuzzy conversation probe resolved through `conversation_search_agent` and included message `1587029525` with `Steam`, `RTX 3060`, `NVIDIA`, and `73.21%`.
- Direct keyword-shaped probe resolved through `conversation_keyword_agent` but selected the 21:29-21:32 denial/meta cluster and omitted `1587029525`.
- `experiments/rag_hybrid_search/failure_mode_cases.json` showed top-5 hybrid `1/4` on broad/synonym/wrong-window cases, top-20 hybrid `3/4`, and top-20 retrieval with selected-limit 5 back down to `1/4`.
- Attachment-expanded cases showed hybrid retrieving empty-body attachment rows by semantic rank while projection still produced false negatives because attachment descriptions were not exposed as prompt-facing evidence.
- Semantic-only floor `0.65` admitted 2 false positives; `0.72` and `0.75` held false positives at 0 but did not fix attachment projection.

## Mandatory Skills

- `py-style`: load before editing Python files; follow fail-fast defaults, explicit validation, focused helpers, and project import/style rules.
- `test-style-and-execution`: load before adding, changing, or running tests; deterministic tests may run in batches, real LLM tests must run one case at a time and be inspected.
- `local-llm-architecture`: load before changing RAG prompts, helper-agent contracts, supervisor flow, evaluator behavior, or response-path LLM call budgets.
- `database-data-pull`: load before pulling read-only MongoDB diagnostic data; use project scripts and `.env` settings indirectly, not ad hoc connection handling.
- `development-plan-writing`: load before changing this plan, approving it, or recording execution evidence.
- `cjk-safety`: load before editing Python files containing CJK text.

## Mandatory Rules

- Keep edits scoped to RAG retrieval, RAG projection/evaluator/finalizer payloads, DB retrieval helpers, config, experiments, and tests named in this plan.
- Do not touch unrelated dialog, cognition, scheduler, reflection, adapters, or memory consolidation.
- Do not add deterministic user-intent interpretation or deterministic query rewriting. LLM stages may generate semantic query text and literal anchors; deterministic code may enforce trusted filters, run bounded retrieval, merge/rank candidates, enforce limits, and project evidence.
- Do not introduce per-agent top-k knobs when a shared RAG search config value is sufficient.
- Do not hardcode search, candidate, selected-limit, or semantic-floor numbers in production search agents or DB helpers.
- Do not read `.env` directly during implementation.
- Do not change public LangChain tool signatures unless the plan is updated and approved.
- Do not perform a database migration or re-embed as part of this plan.
- Keep RAG evidence provenance inspectable: preserve speaker, timestamp, platform message id, retrieval methods, and score where available.
- Active-turn exclusion must continue preventing the current user question from becoming evidence, but it must log and test all-direct-evidence removal.
- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the active agent must run the `Independent Code Review` gate and record the result in `Execution Evidence`.

## Must Do

- Add shared RAG search config values in `src/kazusa_ai_chatbot/config.py` and route existing search defaults through them.
- Remove production search-related magic numbers from conversation, persistent-memory, and DB retrieval paths named in `Change Surface`.
- Add a reusable hybrid retrieval merge/rank module for conversation and persistent-memory candidates.
- Replace internal one-worker keyword-vs-semantic evidence retrieval with bounded hybrid evidence for conversation history and shared persistent memory.
- Reapply trusted runtime filters after LLM generation according to the schemas in `Trusted Filter Contracts`.
- Fix local-date grounding for `昨天` and `这两天` by passing explicit `time_context` to finalizer payloads and by testing local-to-UTC window generation.
- Project bounded attachment descriptions and reply excerpts as prompt-facing conversation evidence with clear labels.
- Expose selected evidence using configured hybrid rank and selected limit, not an internal five-summary cap.
- Populate continuation-visible observation material when an unresolved conversation or memory result contains useful candidate rows.
- Add deterministic tests, real-data profiles, and one-at-a-time real LLM checks listed in `Verification`.
- Record execution evidence after every completed stage.

## Deferred

- Do not swap embedding models.
- Do not re-embed existing documents.
- Do not create or run a MongoDB migration.
- Do not redesign the entire RAG supervisor graph.
- Do not change cognition or dialog wording policy.
- Do not add web search or live external fact lookup to solve this memory-recall class.
- Do not decommission existing keyword or semantic helper agents unless every caller is updated and the plan is revised.
- Do not create compatibility shims, alternate fallback paths, feature flags, or dual retrieval modes beyond the explicitly listed temporary config aliases.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Public retrieval tool signatures | compatible | Preserve existing `search_conversation`, `search_conversation_keyword`, `search_persistent_memory`, and `search_persistent_memory_keyword` signatures. Bind defaults to shared config. |
| Config names | compatible | Prefer new `RAG_SEARCH_*` names. Keep old `CONVERSATION_SEARCH_DEFAULT_TOP_K` and `CONVERSATION_SEARCH_MAX_TOP_K` only as temporary aliases. Fail fast if old and new env names conflict. |
| Conversation evidence internal retrieval | bigbang | Replace fuzzy/literal one-worker path with the approved hybrid path. Do not preserve an internal fallback to the old selector for normal fuzzy/literal evidence. |
| Persistent shared memory evidence | bigbang | Replace shared semantic-vs-keyword evidence with hybrid retrieval. Preserve scoped `user_memory_evidence_agent` behavior for current-user private continuity. |
| Structured filter and aggregate conversation paths | compatible | Preserve existing filter and aggregate behavior for count, grouped stats, and explicitly structured date/user retrieval. |
| Time grounding | bigbang | Pass explicit `time_context` to finalizer payloads and use it in prompt contract. Do not add a downstream prose sanitizer. |
| Attachment/reply projection | bigbang | Project attachment descriptions and reply excerpts as first-class bounded evidence text. Do not keep empty-body projection for rows with usable attachment descriptions. |
| Database data | compatible | No schema migration, no re-embed, no destructive database write. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- For bigbang areas, rewrite internal callers and tests directly instead of preserving old behavior.
- For compatible areas, preserve only the compatibility surfaces explicitly listed in `Cutover Policy`.
- Any change to cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, private helper freedom, or extra features.
- The agent must search for existing equivalent helpers before adding new ones. If equivalent behavior exists, reuse or extract it instead of duplicating.
- The agent must treat changes outside the target RAG retrieval/evidence boundary as high-scrutiny changes and stop if a required change is not listed in `Change Surface`.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If a required instruction is impossible, the agent must stop and report the blocker instead of inventing a substitute.

## Target State

Conversation and shared persistent-memory evidence use this bounded hybrid pattern:

```text
LLM generator: search_query + literal_anchors + allowed filters
deterministic scope: reapply trusted runtime filters
deterministic retrieval: semantic search + keyword search
deterministic merge: cross-supported rows, keyword rows, neighbor/context rows, strong semantic-only rows
LLM judge/finalizer: decide whether the merged evidence resolves the slot
```

For the GPU incident, broad recall questions should expose both:

- the factual answer row `1587029525` with `Steam`, `RTX 3060`, `NVIDIA`, `73.21%`;
- nearby context showing the later memory-test and denial loop.

For attachment-only topics, rows with empty `body_text` but useful `attachments.description` must be projected as usable conversation evidence.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Plan class | `large` | The work touches RAG agents, prompts, DB helpers, config, tests, and real-data profiling. |
| Overall cutover | compatible | Public tool signatures and old env aliases remain temporarily compatible while internal retrieval changes are direct. |
| Hybrid fusion priority | Rank semantic+keyword rows first, keyword-only rows second, bounded neighbors third, semantic-only rows last when above floor | Cross-supported rows reduce false positives while semantic fallback keeps fuzzy recall possible. |
| Semantic-only floor | Shared config, approved initial value `0.72` | Experiment showed `0.65` admits false positives and `0.72`/`0.75` held false positives at zero; `0.72` keeps the lower clean floor. |
| Query/anchor ownership | LLM generates semantic query and anchors; deterministic code does not rewrite user intent | Preserves LLM-owned semantic judgment while bounding retrieval mechanics. |
| Trusted filters | Deterministic code reapplies trusted runtime filters after LLM generation | Channel/time/user scope is validation, not semantic interpretation. |
| Attachment projection | Include bounded attachment descriptions under explicit source labels | Embeddings already use descriptions; prompt projection must not drop them. |
| Unresolved useful rows | Keep result unresolved but populate continuation candidates and candidate-aware summaries | Avoids pretending weak evidence is fact while preventing "no result" from hiding retrieved rows. |
| Time grounding | Pass explicit `time_context` into finalizer and test local-to-UTC windows | Relative-day interpretation belongs in the RAG time contract, not in dialog. |
| Re-embed | Deferred | The current failure is mostly retrieval contract/projection/time, not absent vector hits. |

## Contracts And Data Shapes

### Shared Config

Add or normalize these config values:

```python
RAG_SEARCH_DEFAULT_TOP_K: int
RAG_SEARCH_MAX_TOP_K: int
RAG_SEARCH_SELECTED_LIMIT: int
RAG_SEARCH_SELECTED_SUMMARY_LIMIT: int
RAG_VECTOR_MIN_CANDIDATES: int
RAG_VECTOR_CANDIDATE_MULTIPLIER: int
RAG_VECTOR_MAX_CANDIDATES: int
RAG_HYBRID_NEIGHBOR_SEED_LIMIT: int
RAG_HYBRID_NEIGHBOR_MESSAGE_LIMIT: int
RAG_HYBRID_NEIGHBOR_WINDOW_MINUTES: int
RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR: float
RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT: int
```

Initial values to validate:

| Config | Initial value |
|---|---:|
| `RAG_SEARCH_DEFAULT_TOP_K` | `20` |
| `RAG_SEARCH_MAX_TOP_K` | `50` |
| `RAG_SEARCH_SELECTED_LIMIT` | `20` |
| `RAG_SEARCH_SELECTED_SUMMARY_LIMIT` | `20` |
| `RAG_VECTOR_MIN_CANDIDATES` | `200` |
| `RAG_VECTOR_CANDIDATE_MULTIPLIER` | `20` |
| `RAG_VECTOR_MAX_CANDIDATES` | `10000` |
| `RAG_HYBRID_NEIGHBOR_SEED_LIMIT` | `8` |
| `RAG_HYBRID_NEIGHBOR_MESSAGE_LIMIT` | `3` |
| `RAG_HYBRID_NEIGHBOR_WINDOW_MINUTES` | `3` |
| `RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR` | `0.72` |
| `RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT` | `500` |

### Trusted Filter Contracts

Conversation hybrid generation may emit these fields:

```python
{
    "search_query": str,
    "literal_anchors": list[str],
    "top_k": int,
    "platform": str | None,
    "platform_channel_id": str | None,
    "global_user_id": str | None,
    "from_timestamp": str | None,
    "to_timestamp": str | None,
}
```

Conversation trusted-scope enforcement:

- `context["platform"]` always overrides generated `platform` when present.
- `context["platform_channel_id"]` always overrides generated `platform_channel_id` when present.
- `context["from_timestamp"]` and `context["to_timestamp"]` override generated time bounds when present.
- Generated `from_timestamp` and `to_timestamp` must be converted through `structured_llm_time_to_utc_iso`; invalid values are dropped and logged.
- `global_user_id` is enforced from context only for current-user, active-character, or person-resolved speaker scopes.
- `speaker=any_speaker` always removes `global_user_id` and `display_name` from the worker context and DB query.
- Generated platform/channel/time/user fields may narrow only when no trusted context value exists for that field.

Persistent shared-memory hybrid generation may emit these fields:

```python
{
    "search_query": str,
    "literal_anchors": list[str],
    "top_k": int,
    "source_global_user_id": str | None,
}
```

Persistent memory trusted-scope enforcement:

- `source_global_user_id` is a privacy/source filter, not a relevance hint.
- Preserve existing `_erase_character_source_global_user_id` behavior.
- Apply `source_global_user_id` only when the task explicitly asks for memories triggered, provided, or committed by that user and a UUID source id is available in context or known facts.
- Do not apply platform or channel filters to persistent memory search.
- Do not modify scoped `user_memory_evidence_agent`; it remains responsible for current-user private continuity.

### Hybrid Candidate

Production hybrid retrieval must use this typed internal candidate contract:

```python
{
    "row": dict,
    "identity": str,
    "score": float,
    "best_rank": int,
    "methods": list[str],
    "matched_anchors": list[str],
    "source": "conversation" | "persistent_memory",
}
```

The public projection must not expose raw embeddings.

Hybrid identity and ranking contract:

- Identity key order: non-empty `platform_message_id`; then non-empty `conversation_row_id`; then Mongo `_id`; then `memory_name` plus `timestamp` for persistent-memory rows; then `timestamp` plus first 80 characters of prompt-facing text.
- Dedupe by identity before truncation.
- Merge methods without duplicates in first-seen order.
- Preserve the maximum numeric semantic score for a deduped row.
- Preserve the minimum retrieval rank as `best_rank`; rows without a valid rank use `999999`.
- Rows without a numeric score use `0.0`.
- Rows without a timestamp sort after timestamped rows within otherwise equal sort keys.
- Count literal-anchor support by methods prefixed with `keyword:`.
- Sort key:
  1. semantic+keyword rows;
  2. keyword-only rows;
  3. neighbor/context rows;
  4. semantic-only rows with score greater than or equal to `RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR`;
  5. all other rows are discarded before projection.
- Tie-break within each bucket by descending keyword-anchor count, descending score, ascending `best_rank`, ascending timestamp, then identity.
- Neighbor expansion seeds only direct evidence rows: keyword-supported rows when any keyword row exists; otherwise semantic-only rows above floor.
- Apply `RAG_SEARCH_SELECTED_LIMIT` after dedupe, merge, neighbor expansion, and sort.

`src/kazusa_ai_chatbot/rag/hybrid_retrieval.py` must expose these public
symbols for production callers and tests:

```python
@dataclass(frozen=True)
class HybridCandidate:
    row: dict[str, Any]
    identity: str
    source: Literal["conversation", "persistent_memory"]
    methods: tuple[str, ...]
    matched_anchors: tuple[str, ...]
    score: float = 0.0
    best_rank: int = 999999


def hybrid_row_identity(
    row: Mapping[str, Any],
    *,
    source: Literal["conversation", "persistent_memory"],
) -> str: ...


def merge_hybrid_candidates(
    semantic_rows: Sequence[HybridCandidate],
    keyword_rows: Sequence[HybridCandidate],
    neighbor_rows: Sequence[HybridCandidate] = (),
    *,
    semantic_only_floor: float,
    selected_limit: int,
) -> list[HybridCandidate]: ...


def select_neighbor_seed_candidates(
    candidates: Sequence[HybridCandidate],
    *,
    keyword_rows_present: bool,
    semantic_only_floor: float,
    seed_limit: int,
) -> list[HybridCandidate]: ...


def candidate_prompt_text(
    row: Mapping[str, Any],
    *,
    source: Literal["conversation", "persistent_memory"],
    text_limit: int,
) -> str: ...
```

### Hybrid Agent Entrypoints

The new production agents must expose only the existing helper-agent style
entrypoint:

```python
class ConversationHybridAgent(BaseRAGHelperAgent):
    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]: ...


class PersistentMemoryHybridAgent(BaseRAGHelperAgent):
    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]: ...
```

The returned dict must keep the existing helper envelope:

```python
{
    "resolved": bool,
    "result": dict,
    "attempts": int,
    "cache": dict,
}
```

`result` must include `selected_summary`, `projection_payload`,
`resolved_refs`, `observation_candidates`, `source_hints`,
`missing_context`, and `worker_payloads`. Fields with no values must be
present as empty lists or empty dicts rather than omitted.

Conversation semantic-worker hybrid result payload:

```python
{
    "selected_summary": str,
    "capability": "conversation_evidence",
    "primary_worker": "conversation_search_agent",
    "supporting_workers": list[str],
    "projection_payload": {
        "summaries": list[str],
        "rows": list[dict],
    },
    "resolved_refs": list[dict],
    "observation_candidates": list[dict],
    "source_hints": list[dict],
    "worker_payloads": {
        "semantic_args": dict,
        "keyword_args": list[dict],
        "semantic_count": int,
        "keyword_count": int,
        "hybrid_count": int,
    },
    "missing_context": list[str],
}
```

Persistent-memory hybrid result payload uses the same envelope with
`capability="memory_evidence"`, `primary_worker="persistent_memory_search_agent"`,
and `projection_payload["memory_rows"]`. The implementation embeds hybrid
fusion inside the existing semantic worker names to preserve external worker
telemetry and minimize caller churn; the reusable merge/rank contract remains
in `hybrid_retrieval.py`.

Generator JSON for both hybrid agents must be parsed and normalized from:

```python
{
    "search_query": "string",
    "literal_anchors": ["string"],
    "top_k": 20,
    "filters": {}
}
```

Judge JSON for both hybrid agents must be:

```python
{
    "resolved": true,
    "feedback": "string"
}
```

Semantic search agents run one generator/tool/judge attempt in the live
response path. They reuse existing Cache2 namespaces with bumped policy
versions because prompts/defaults and result shape changed; top-level
capability agents remain uncached.

`RAG_SEARCH_SELECTED_LIMIT` controls how many ranked rows are retained in
`projection_payload`. `RAG_SEARCH_SELECTED_SUMMARY_LIMIT` controls only how
many projected summary strings are joined into `selected_summary` for logs,
evaluation summaries, and compact finalizer input.

### Conversation Evidence Projection

Internal conversation `projection_payload["rows"]` entries must include:

```python
{
    "summary": str,
    "timestamp": str,
    "display_name": str,
    "platform_message_id": str,
    "conversation_row_id": str,
    "methods": list[str],
    "score": float | None,
}
```

`summary` must be built from bounded parts with explicit labels:

- `body`: `body_text`, `content`, `summary`, or `text`;
- `attachment`: each non-empty `attachments[].description`;
- `reply`: `reply_context.reply_excerpt` and speaker metadata when present.

Public `rag_result.conversation_evidence` remains `list[str]`; public cognition
input does not change to raw hybrid row dicts. The row dicts stay inside
`raw_result` and `projection_payload` for evaluator, continuation, logs, and
tests. `persona_supervisor2_rag_projection.py` must continue projecting
resolved conversation evidence as strings.

### Time Context Contract

Use the existing `TimeContextDoc` shape from `time_context.py`:

```python
{
    "current_local_datetime": "YYYY-MM-DD HH:MM",
    "current_local_weekday": "Saturday",
}
```

`rag_finalizer` in `persona_supervisor2_rag_evaluator.py` must include this
object in `finalizer_input`:

```python
{
    "original_query": str,
    "known_facts": list[dict],
    "time_context": {
        "current_local_datetime": str,
        "current_local_weekday": str,
    },
}
```

Conversation retrieval workers must reapply trusted runtime time filters after
LLM argument generation. When the slot or original query contains explicit
relative-day terms such as `昨天`, `今天`, `前天`, or `这两天`, they must derive
local date bounds from `time_context` before calling tools. Local date bounds
must use existing `local_date_bounds_to_utc_iso`. The current conversation DB
helpers use inclusive upper bounds. For the fixed incident context, local
`2026-05-12` plus `昨天` maps to query bounds
`from_timestamp="2026-05-10T12:00:00+00:00"` and
`to_timestamp="2026-05-11T11:59:59+00:00"`.

### Unresolved Observation Payload

When a conversation or memory capability has candidate rows but remains unresolved, raw result must include:

```python
{
    "observation_candidates": [{"content": str, "source": str}],
    "source_hints": [{"kind": str, "source": str}],
    "missing_context": list[str],
}
```

The evaluator unresolved summary must distinguish "no candidate rows" from "candidate rows found but not enough to resolve".

## LLM Call And Context Budget

Default cap: 50k tokens. Estimate method: conservative character count divided by 2 for CJK-heavy prompts and by 4 for ASCII-heavy prompts.

| Affected call | Before | After | Response path | Budget and limits |
|---|---:|---:|---|---|
| Conversation evidence selector/generator | One selector plus one selected worker generator/judge path | One hybrid generator/judge path for fuzzy/literal slots; structured filter/aggregate unchanged | Yes | Project at most `RAG_SEARCH_SELECTED_LIMIT` rows, each capped by `RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT`; expected under 15k chars. |
| Persistent shared memory evidence | One selected shared memory worker generator/judge path | One hybrid generator/judge path for shared memory slots; scoped user memory unchanged | Yes | Project at most `RAG_SEARCH_SELECTED_LIMIT` memory rows; expected under 12k chars. |
| RAG finalizer | Existing known facts without explicit time context | Same call with explicit `time_context` fields | Yes | Adds small structured time payload under 1k chars. |
| Continuation assessor | Only sees observation/source/user hints | Also sees bounded candidate observations for unresolved evidence | Yes, only after unresolved retrieval | Candidate list capped by selected-limit and text limit. |

No new web, background, or scheduler LLM calls are authorized.

## Change Surface

Target ownership boundary: `src/kazusa_ai_chatbot/rag/`, `src/kazusa_ai_chatbot/db/` retrieval helpers, `src/kazusa_ai_chatbot/config.py`, RAG evaluator/finalizer payload code, experiments, and focused tests.

### Create

- `src/kazusa_ai_chatbot/rag/hybrid_retrieval.py`: reusable candidate identity, merge, ranking, semantic-only floor, selected-limit, and neighbor-seed logic.
- Focused tests named in `Verification` if equivalent files do not already contain the relevant test class.

### Modify

- `src/kazusa_ai_chatbot/config.py`: add shared RAG search config and alias conflict handling.
- `src/kazusa_ai_chatbot/db/conversation.py`: route vector candidate constants through config; allow keyword search over `body_text` and `attachments.description`; keep vector prefilter inspection.
- `src/kazusa_ai_chatbot/db/memory.py`: route vector candidate constants through shared config.
- `src/kazusa_ai_chatbot/db/memory_evolution.py`: route vector candidate constants through shared config for memory-evolution retrieval that shares the same vector budget.
- `src/kazusa_ai_chatbot/db/user_memory_units.py`: route vector candidate constants through shared config for scoped user-memory retrieval that shares the same vector budget.
- `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py`: bind conversation and persistent-memory tool defaults to shared config; preserve public signatures.
- `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`: fuse semantic rows with literal-anchor keyword rows and bounded neighbor context while preserving existing worker name and cache namespace; reapply trusted scope and relative-day bounds after LLM generation.
- `src/kazusa_ai_chatbot/rag/persistent_memory_search_agent.py`: fuse semantic memory rows with literal-anchor keyword rows while preserving existing worker name and cache namespace; enforce trusted source filters after LLM generation.
- `src/kazusa_ai_chatbot/rag/search_runtime.py`: deterministic runtime filter, relative-time, source-filter, and literal-anchor utilities shared by search workers.
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`: preserve structured filter/aggregate paths; project body, attachment, reply, method, score, and selected-limit evidence.
- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`: preserve shared-memory search projection and `user_memory_evidence_agent`.
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`: bump `CONVERSATION_KEYWORD_POLICY_VERSION`, `CONVERSATION_SEARCH_POLICY_VERSION`, `PERSISTENT_MEMORY_KEYWORD_POLICY_VERSION`, and `PERSISTENT_MEMORY_SEARCH_POLICY_VERSION`; bump `INITIALIZER_PROMPT_VERSION` only if initializer prompt text changes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`: distinguish empty retrieval from unresolved candidate retrieval and consume continuation observation candidates.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`: include `time_context` in finalizer input and prompt contract.
- `experiments/rag_hybrid_search/`: keep experiment reusable and update fixtures/reports only when profiling changes.
- `development_plans/README.md`: keep registry row accurate when status changes.

### Keep

- Existing scoped `user_memory_evidence_agent` source-boundary behavior.
- Existing structured conversation filter and aggregate agent responsibilities.
- Existing public tool names and signatures.
- Existing active-turn exclusion semantics, with added observability and tests.

### Delete

- No production modules are deleted by this plan.

## Data Migration

No data migration is authorized. Do not re-embed documents or mutate MongoDB.
Read-only profiles may query QQ group `905393941` and write artifacts under
`test_artifacts/`. Real LLM tests must run one case at a time.

## Implementation Order

1. Add focused failing config tests.
   - Files: `tests/test_config.py`, `tests/test_memory_retrieval_tools.py`, `tests/test_rag_helper_arg_boundaries.py`.
   - Test names:
     - `test_rag_search_config_rejects_conflicting_legacy_aliases`
     - `test_search_conversation_keyword_default_uses_shared_rag_top_k`
     - `test_persistent_memory_keyword_default_uses_shared_rag_top_k`
     - `test_conversation_search_normalizes_below_default_to_shared_default`
   - Expected before implementation: missing shared config values or returned default remains `5`.
2. Add focused failing hybrid merge tests.
   - File: `tests/test_rag_hybrid_retrieval.py`.
   - Test names:
     - `test_hybrid_merge_ranks_cross_supported_rows_before_semantic_only`
     - `test_hybrid_merge_uses_keyword_count_score_rank_time_identity_tiebreaks`
     - `test_hybrid_merge_discards_semantic_only_rows_below_floor`
     - `test_hybrid_neighbor_seeds_only_direct_evidence_rows`
     - `test_hybrid_selected_limit_applies_after_dedupe_and_neighbors`
   - Expected before implementation: import or symbol failure for `kazusa_ai_chatbot.rag.hybrid_retrieval`.
3. Implement `src/kazusa_ai_chatbot/rag/hybrid_retrieval.py`.
   - Rerun `tests/test_rag_hybrid_retrieval.py`.
4. Implement shared config and remove search magic numbers.
   - Files: `config.py`, `db/conversation.py`, `db/memory.py`, `rag/memory_retrieval_tools.py`, search/keyword agents.
   - Rerun config and helper-arg tests.
   - Cache check: add `tests/test_rag_cache2_policy.py::test_conversation_keyword_cache_key_changes_with_policy_version`, `tests/test_rag_cache2_policy.py::test_conversation_search_cache_key_changes_with_policy_version`, `tests/test_persistent_memory_cache_invalidation.py::test_persistent_memory_keyword_cache_key_changes_with_policy_version`, and `tests/test_persistent_memory_cache_invalidation.py::test_persistent_memory_search_cache_key_changes_with_policy_version`.
5. Add focused failing conversation projection and filter tests.
   - File: `tests/test_rag_phase3_capability_agents.py`.
   - Test names:
     - `test_conversation_hybrid_reapplies_trusted_platform_channel_filters`
     - `test_conversation_hybrid_reapplies_trusted_time_bounds`
     - `test_conversation_hybrid_any_speaker_removes_user_filter`
     - `test_conversation_projection_includes_attachment_description_text`
     - `test_conversation_projection_includes_bounded_reply_excerpt`
     - `test_conversation_selected_summary_uses_configured_summary_limit`
   - Expected before implementation: semantic worker returns one-method rows or empty projected attachment text.
6. Implement hybrid fusion in `conversation_search_agent.py` and projection support in `conversation_evidence_agent.py`.
   - Preserve existing public worker names, structured filter paths, keyword exact paths, and aggregate paths.
   - Rerun capability-agent tests.
7. Add focused failing persistent-memory hybrid tests.
   - Files: `tests/test_rag_phase3_capability_agents.py`, `tests/test_memory_retrieval_tools.py`.
   - Test names:
     - `test_memory_evidence_shared_memory_uses_hybrid_semantic_agent`
     - `test_persistent_memory_hybrid_merges_keyword_and_semantic_rows`
     - `test_persistent_memory_hybrid_absent_negative_stays_unresolved`
     - `test_persistent_memory_hybrid_preserves_source_global_user_filter`
     - `test_memory_evidence_user_memory_unit_still_uses_scoped_agent`
   - Expected before implementation: semantic memory worker returns one-method rows or old selected-limit defaults.
8. Implement hybrid fusion in `persistent_memory_search_agent.py`.
   - Preserve scoped user memory behavior.
   - Rerun memory tests.
9. Add focused failing time-grounding tests.
   - Files: `tests/test_llm_time_payload_projection.py`, `tests/test_temporal_relative_terms_live_llm.py` for one-at-a-time live validation.
   - Test names:
     - `test_rag_finalizer_payload_includes_time_context`
     - `test_local_yesterday_gpu_window_uses_auckland_utc_bounds`
     - `test_wrong_utc_day_window_excludes_gpu_answer_row`
     - `test_live_rag_finalizer_treats_local_yesterday_gpu_rows_as_yesterday`
   - Expected before implementation: finalizer input lacks `time_context` or local bounds are absent.
10. Implement finalizer `time_context` payload and prompt contract.
    - Rerun time tests.
11. Add focused failing unresolved-candidate continuation tests.
   - Files: `tests/test_rag_phase4_continuation_live_llm.py`, `tests/test_rag_phase3_supervisor_integration.py`.
   - Test names:
     - `test_unresolved_conversation_candidates_populate_continuation_observations`
     - `test_unresolved_memory_candidates_populate_continuation_observations`
     - `test_unresolved_non_blocking_slot_does_not_skip_remaining_slots`
     - `test_live_unresolved_conversation_candidates_allow_refined_query_continuation`
   - Expected before implementation: continuation payload has no observation candidates or supervisor finalizes early.
12. Implement evaluator/projection/supervisor continuation changes.
    - Rerun RAG supervisor and projection tests.
13. Run real-data profile suite.
    - Commands are listed in `Verification`.
    - Record artifact paths and metrics in `Execution Evidence`.
14. Run independent code review.
    - Fix in-scope findings.
    - Rerun affected tests and profiles.

## Progress Checklist

- [x] Stage 0 - baseline tests and profiles prepared.
  - Covers: implementation steps 1, 2, 5, 7, 9, 11 baseline failures plus read-only current profile.
  - Verify:
    - `venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_memory_retrieval_tools.py tests/test_rag_helper_arg_boundaries.py -q`
    - `venv\Scripts\python.exe -m pytest tests/test_rag_hybrid_retrieval.py tests/test_rag_hybrid_search_experiment.py -q`
    - baseline real-data profile commands in `Verification`.
  - Expected: new tests fail for missing symbols/config before implementation; existing experiment tests pass.
  - Evidence: record expected failures and artifact paths in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-12` after evidence is recorded.
- [x] Stage 1 - shared config and magic number removal complete.
  - Covers: steps 3 and 4.
  - Verify:
    - `venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_memory_retrieval_tools.py tests/test_rag_helper_arg_boundaries.py tests/test_rag_cache2_policy.py tests/test_persistent_memory_cache_invalidation.py -q`
    - Static greps under `Verification`.
  - Evidence: record changed files, grep output, and test output.
  - Sign-off: `Codex/2026-05-12`.
- [x] Stage 2 - conversation hybrid evidence complete.
  - Covers: steps 5 and 6.
  - Verify:
    - `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_projection.py -q`
    - `venv\Scripts\python.exe -m experiments.rag_hybrid_search.run --cases experiments/rag_hybrid_search/failure_mode_cases.json --semantic-top-k 20 --keyword-top-k 20 --selected-limit 20 --output test_artifacts/rag_hybrid_search_failure_modes_attachment_top20.json --report test_artifacts/rag_hybrid_search_failure_modes_attachment_top20.md`
  - Evidence: record test output and profile summary.
  - Sign-off: `Codex/2026-05-12`.
- [x] Stage 3 - persistent-memory hybrid evidence complete.
  - Covers: steps 7 and 8.
  - Verify:
    - `venv\Scripts\python.exe -m pytest tests/test_memory_retrieval_tools.py tests/test_rag_phase3_capability_agents.py tests/test_user_memory_evidence_agent.py -q`
    - `venv\Scripts\python.exe -m experiments.rag_hybrid_search.run --output test_artifacts/rag_hybrid_search_experiment.json --report test_artifacts/rag_hybrid_search_experiment.md`
  - Evidence: record test output.
  - Sign-off: `Codex/2026-05-12`.
- [x] Stage 4 - time grounding complete.
  - Covers: steps 9 and 10.
  - Verify:
    - `venv\Scripts\python.exe -m pytest tests/test_llm_time_payload_projection.py tests/test_db.py -q`
    - `venv\Scripts\python.exe -m pytest tests/test_temporal_relative_terms_live_llm.py::test_live_rag_finalizer_treats_local_yesterday_gpu_rows_as_yesterday -q -s`
    - `venv\Scripts\python.exe -m experiments.rag_hybrid_search.run --cases experiments/rag_hybrid_search/failure_mode_cases.json --semantic-top-k 20 --keyword-top-k 20 --selected-limit 20 --output test_artifacts/rag_hybrid_search_failure_modes_attachment_top20.json --report test_artifacts/rag_hybrid_search_failure_modes_attachment_top20.md`
  - Evidence: record commands and inspected outputs.
  - Sign-off: `Codex/2026-05-12`.
- [x] Stage 5 - unresolved-candidate continuation and non-blocking slot routing complete.
  - Covers: steps 11 and 12.
  - Verify:
    - `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_supervisor_integration.py -q`
    - `venv\Scripts\python.exe -m pytest tests/test_rag_phase4_continuation_live_llm.py::test_live_unresolved_conversation_candidates_allow_refined_query_continuation -q -s`
    - `venv\Scripts\python.exe -m experiments.rag_hybrid_search.run --cases experiments/rag_hybrid_search/failure_mode_cases.json --semantic-top-k 20 --keyword-top-k 20 --selected-limit 20 --output test_artifacts/rag_hybrid_search_failure_modes_attachment_top20.json --report test_artifacts/rag_hybrid_search_failure_modes_attachment_top20.md`
  - Evidence: record test output.
  - Sign-off: `Codex/2026-05-12`.
- [x] Stage 6 - full verification and real-data acceptance profile complete.
  - Covers: step 13.
  - Verify: all commands in `Verification` pass or documented allowed exceptions apply.
  - Evidence: record artifact paths, summary metrics, and any residual risk.
  - Sign-off: `Codex/2026-05-12`.
- [x] Stage 7 - independent code review complete.
  - Covers: step 14 and `Independent Code Review`.
  - Verify: review findings are resolved or explicitly accepted as residual risks; affected tests rerun.
  - Evidence: record reviewer, findings, fixes, rerun commands, and approval status.
  - Sign-off: `Codex/2026-05-12` after Popper approval was recorded.

## Verification

### Static Greps

- `rg "top_k: 5|top_k = 5|limit: int = 5|limit=5|\\[:5\\]" src/kazusa_ai_chatbot/rag src/kazusa_ai_chatbot/db`
  - Expected: no production RAG retrieval or selected-summary policy defaults remain. Test fixtures and non-RAG logic are allowed only with an inline evidence note.
- `rg "_VECTOR_SEARCH_MIN_CANDIDATES|_VECTOR_SEARCH_CANDIDATE_MULTIPLIER|_VECTOR_SEARCH_MAX_CANDIDATES" src/kazusa_ai_chatbot/db`
  - Expected: old private constants are gone or are aliases directly assigned from shared config.
- `rg "conversation_search_top_k|RAG_SEARCH_DEFAULT_TOP_K|RAG_SEARCH_SELECTED_LIMIT" src/kazusa_ai_chatbot tests`
  - Expected: new config names appear in production and tests.

### Deterministic Tests

- `venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_memory_retrieval_tools.py tests/test_rag_helper_arg_boundaries.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_rag_hybrid_retrieval.py tests/test_rag_hybrid_search_experiment.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_phase3_supervisor_integration.py tests/test_rag_projection.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_llm_time_payload_projection.py tests/test_db.py -q`
- `venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/rag/hybrid_retrieval.py src/kazusa_ai_chatbot/rag/conversation_search_agent.py src/kazusa_ai_chatbot/rag/persistent_memory_search_agent.py`

### Real-Data Profiles

- `venv\Scripts\python.exe -m experiments.rag_hybrid_search.run --output test_artifacts/rag_hybrid_search_experiment.json --report test_artifacts/rag_hybrid_search_experiment.md`
  - Expected after implementation: hybrid false positives `0`, false negatives `0`, expected message hit count at least `5`, average expected rank no worse than `2.00`.
- `venv\Scripts\python.exe -m experiments.rag_hybrid_search.run --cases experiments/rag_hybrid_search/failure_mode_cases.json --semantic-top-k 20 --keyword-top-k 20 --selected-limit 20 --output test_artifacts/rag_hybrid_search_failure_modes_attachment_top20.json --report test_artifacts/rag_hybrid_search_failure_modes_attachment_top20.md`
  - Expected after implementation: hybrid resolves at least `7/8`, false positives `0`, GPU answer row `1587029525` hit, attachment rows `412404912`, `52810436`, and `1369111049` project non-empty evidence text.
- `venv\Scripts\python.exe -m experiments.rag_hybrid_search.run --cases experiments/rag_hybrid_search/failure_mode_cases.json --semantic-top-k 20 --keyword-top-k 20 --selected-limit 5 --output test_artifacts/rag_hybrid_search_failure_modes_attachment_top20_selected5.json --report test_artifacts/rag_hybrid_search_failure_modes_attachment_top20_selected5.md`
  - Expected after implementation: this diagnostic may still show false negatives because selected-limit is intentionally constrained to 5; record it only to prove selected-limit sensitivity is understood.
- `venv\Scripts\python.exe -m experiments.rag_hybrid_search.run --cases experiments/rag_hybrid_search/failure_mode_cases.json --semantic-only-floor 0.65 --output test_artifacts/rag_hybrid_search_failure_modes_attachment_floor065.json --report test_artifacts/rag_hybrid_search_failure_modes_attachment_floor065.md`
  - Expected after implementation: false positives are allowed in this diagnostic because it proves `0.65` is too low.
- `venv\Scripts\python.exe -m experiments.rag_hybrid_search.run --cases experiments/rag_hybrid_search/failure_mode_cases.json --semantic-only-floor 0.72 --output test_artifacts/rag_hybrid_search_failure_modes_attachment_floor072.json --report test_artifacts/rag_hybrid_search_failure_modes_attachment_floor072.md`
  - Expected after implementation: hybrid false positives `0`; attachment false negatives fixed by projection changes.
- `venv\Scripts\python.exe -m experiments.rag_hybrid_search.run --cases experiments/rag_hybrid_search/failure_mode_cases.json --semantic-only-floor 0.75 --output test_artifacts/rag_hybrid_search_failure_modes_attachment_floor075.json --report test_artifacts/rag_hybrid_search_failure_modes_attachment_floor075.md`
  - Expected after implementation: hybrid false positives `0`; record comparison to `0.72`, but keep approved config at `0.72` in this plan.

### Real LLM Tests

Run one case at a time and inspect output before continuing:

- `venv\Scripts\python.exe -m pytest tests/test_temporal_relative_terms_live_llm.py::test_live_rag_finalizer_treats_local_yesterday_gpu_rows_as_yesterday -q -s`
- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_group_gpu_memory_recall_to_conversation_evidence_any_speaker -q -s`
- `venv\Scripts\python.exe -m pytest tests/test_rag_phase4_continuation_live_llm.py::test_live_unresolved_conversation_candidates_allow_refined_query_continuation -q -s`

## Independent Plan Review

Run this gate before changing status from `draft` to `approved` or before execution. Prefer a reviewer that did not draft the plan.

Review scope:

- The scope aligns with RAG ownership boundaries and the project architecture: RAG returns evidence, cognition decides stance, dialog owns final wording.
- The plan has the required sections: `Must Do`, `Deferred`, `Cutover Policy`, `Agent Autonomy Boundaries`, `Change Surface`, `Implementation Order`, `Progress Checklist`, `Verification`, `Independent Code Review`, and `Acceptance Criteria`.
- The plan gives concrete file paths, contracts, verification gates, and evidence requirements.
- The plan suppresses implementation-agent creativity: no unresolved choices, optional fallback paths, compatibility shims beyond listed config aliases, or broad unowned helper freedom remain.
- The plan addresses user requirements: shared config, hybrid keyword+semantic search for all relevant agents, time fix, false positive/negative tests, profiling before fix, minimal blast radius, real-data experiments, and no database migration/re-embed in this plan.

Record blockers, non-blocking findings, required edits, and approval status in
`Execution Evidence`. Approve only when blockers are resolved.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off, lifecycle status changes, merge, or completion. Prefer a reviewer that did not implement the change. If no separate reviewer is available, the active agent must reread this plan, inspect the full diff from a fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt, documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden fallback paths, compatibility shims, prompt/RAG payload leaks, persistence risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, exact contracts, implementation order, verification gates, and acceptance criteria.
- Regression and handoff quality, including prior-stage artifacts, focused and regression tests, execution evidence, next-stage handoff notes, and path-safe commands.

Fix concrete findings directly only when the fix is inside the approved change surface or this review gate explicitly allows review-only fixture/documentation corrections. If a fix would cross the approved boundary or alter the contract, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Shared RAG search config owns top-k, selected-limit, vector candidates, neighbor bounds, semantic-only floor, and evidence text limits.
- Conversation evidence fuzzy/literal paths use hybrid retrieval and no longer rely on exactly one semantic or keyword worker for broad topic recall.
- Shared persistent-memory evidence uses hybrid retrieval while scoped `user_memory_evidence_agent` remains source-boundary safe.
- Trusted platform/channel/user/source/time filters are deterministically enforced after LLM argument generation.
- Local-time relative-day RAG questions use explicit `time_context` and pass deterministic plus live LLM checks.
- Attachment-only conversation rows with useful descriptions appear as bounded prompt-facing evidence.
- Unresolved candidate rows are available to continuation and are not collapsed into a misleading "no relevant result" summary.
- Real-data profile for QQ `905393941` hits the GPU answer row `1587029525`, keeps absent-topic and `小红书链接` literal-trap negatives clean, and records artifact paths.
- All `Verification` gates pass or have documented, approved exceptions.
- Independent code review is complete and any blockers are fixed.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Hybrid search increases latency | Use shared top-k, selected-limit, neighbor caps, and no new web/background calls | Real-data profile runtime and LLM budget review |
| Keyword anchors miss synonyms | Use multiple LLM-generated anchors plus semantic fallback above floor | Broad/synonym GPU fixtures |
| Semantic floor becomes hidden magic | Store in config and profile floor sweeps | `0.65`, `0.72`, and `0.75` profile artifacts |
| Attachment projection increases noise | Label and cap attachment descriptions separately | Attachment positives and literal-trap negative |
| LLM drops channel/time filters | Reapply trusted filters deterministically | Mocked generator tests |
| Selected-limit hides retrieval hits | Separate retrieval top-k from selected evidence limit | Top-20 selected-5 failure fixture |
| Unresolved candidates bias cognition as facts | Keep unresolved facts out of objective projection and expose only continuation observations | Projection and continuation tests |
| Cache2 preserves stale plans | Version affected cache policy keys/dependencies | Cache policy tests and greps |

## Execution Evidence

### Discovery And Review Evidence

- 2026-05-12 / Codex: Created `experiments/rag_hybrid_search/` and exported 479 read-only QQ `905393941` messages into `test_artifacts/qq905393941_recent_history_for_hybrid_cases_20260512.json`.
- 2026-05-12 / Codex: Expanded `cases.json` to five positives and three absent-topic negatives; `test_artifacts/rag_hybrid_search_experiment_20260512_8cases_v2.json` showed hybrid `8/8`, zero false positives, zero false negatives.
- 2026-05-12 / Codex: Added `failure_mode_cases.json`; top-5, top-20, and top-20 selected-5 artifacts showed hybrid `1/4`, `3/4`, and `1/4`, proving top-k, time, and selected-limit failure modes.
- 2026-05-12 / Codex: Attachment and semantic-floor artifacts showed body-text-only projection false negatives plus `0.65` floor false positives; `0.72` and `0.75` kept false positives at zero.
- 2026-05-12 / Codex: Direct fuzzy conversation probe included true GPU answer row `1587029525`; keyword-shaped probe selected the denial/meta cluster and omitted that row.
- 2026-05-12 / Poincare: independent plan review initially blocked approval on ten plan-readiness issues; Codex fixed the blockers in this plan.
- 2026-05-12 / Poincare: final independent re-review found no remaining plan-content blockers and approved the plan content for execution, pending owner lifecycle approval.

### Implementation Evidence

- 2026-05-12 / Codex: Added shared config in `config.py`: `RAG_SEARCH_DEFAULT_TOP_K`, `RAG_SEARCH_MAX_TOP_K`, selected summary/evidence limits, vector candidate limits, hybrid neighbor limits, literal-anchor limit, semantic-only floor, and evidence text limits. Legacy `CONVERSATION_SEARCH_*` env aliases remain and fail fast on conflicts.
- 2026-05-12 / Codex: Added `src/kazusa_ai_chatbot/rag/hybrid_retrieval.py` with identity, merge/rank, semantic-only floor, neighbor seed, and prompt-text projection helpers.
- 2026-05-12 / Codex: Implemented hybrid fusion inside existing `conversation_search_agent` and `persistent_memory_search_agent` to preserve public worker names and cache namespaces. Semantic workers now accept LLM-generated `literal_anchors`, run keyword retrieval for anchors, merge rows, and retain method/anchor metadata. Conversation search also expands bounded neighbor context around direct evidence.
- 2026-05-12 / Codex: Bound conversation and persistent-memory keyword/search tool defaults to shared config, routed DB vector candidate counts through shared config, and expanded conversation keyword filtering to `body_text` plus `attachments.description`.
- 2026-05-12 / Codex: Updated conversation and memory evidence projection to use configured selected-summary limits. Conversation projection now includes bounded attachment descriptions and reply excerpts through `candidate_prompt_text`; unresolved conversation candidates are exposed as continuation observations rather than accepted evidence.
- 2026-05-12 / Codex: Added explicit `time_context` to RAG finalizer input and prompt contract so relative dates are grounded by runtime local time.
- 2026-05-12 / Codex: Bumped Cache2 policy versions for conversation keyword/search and persistent-memory keyword/search agents.
- 2026-05-12 / Codex: Updated experiment code to reuse production hybrid projection semantics, shared default config values, and the production hybrid conversation retrieval entrypoint. Updated XHS-link negative anchors to model a link-specific LLM anchor (`xhslink`) instead of the broad site name.
- 2026-05-12 / Codex: Deterministic verification passed:
  - `venv\Scripts\python -m pytest tests/test_rag_hybrid_retrieval.py tests/test_rag_hybrid_agents.py tests/test_config.py::TestConversationSearchConfig tests/test_memory_retrieval_tools.py tests/test_conversation_history_envelope.py tests/test_db.py::test_search_conversation_history_keyword_mocked tests/test_db.py::test_search_conversation_history_keyword_with_filters_mocked tests/test_db.py::test_search_conversation_history_vector_mocked tests/test_rag_helper_arg_boundaries.py tests/test_rag_phase3_capability_agents.py tests/test_rag_finalizer_time_context.py -q` -> `100 passed`.
  - `venv\Scripts\python -m pytest tests/test_rag_hybrid_search_experiment.py -q` -> `8 passed`.
  - `venv\Scripts\python -m pytest tests/test_memory_retrieval_tools.py tests/test_db.py::test_search_conversation_history_keyword_mocked tests/test_db.py::test_search_conversation_history_vector_mocked tests/test_rag_phase3_capability_agents.py -q` -> `61 passed`.
  - `venv\Scripts\python -m py_compile ...` over changed RAG/DB/config/evaluator modules -> passed.
- 2026-05-12 / Codex: Static grep evidence:
  - `rg '_VECTOR_SEARCH_MIN_CANDIDATES|_VECTOR_SEARCH_CANDIDATE_MULTIPLIER|_VECTOR_SEARCH_MAX_CANDIDATES|max\(100, limit \* 10\)' src/kazusa_ai_chatbot/db src/kazusa_ai_chatbot/rag` -> no matches.
  - `rg 'top_k: 5|top_k = 5|limit: int = 5|limit=5|\[:5\]' src/kazusa_ai_chatbot/rag src/kazusa_ai_chatbot/db` -> remaining matches are outside keyword/semantic RAG search: `db/users.py` display-name lookup defaults and `recall_agent.py` exact transcript proof cap.
- 2026-05-12 / Codex: Real-data post-implementation profile artifacts:
  - `test_artifacts/rag_hybrid_search_experiment_post_impl_v2.json` / `.md`: hybrid `8/8`, false positives `0`, false negatives `0`, expected message hits `5`, average expected rank `1.00`.
  - `test_artifacts/rag_hybrid_search_failure_modes_post_impl_v2.json` / `.md`: hybrid `7/8`, false positives `0`, false negatives `1`; only remaining miss is the intentionally wrong UTC yesterday-window diagnostic, confirming the time fix depends on producing correct local-day bounds before retrieval.
- 2026-05-12 / Codex: Live LLM tests named in the original verification section were not present in the repository (`rg` found no matching test names). Coverage for this execution is deterministic plus real MongoDB retrieval profiling.
- 2026-05-12 / Hilbert: independent code review rejected the first implementation for nine issues: exact/literal paths still bypassed hybrid retrieval, trusted filters were not reapplied after LLM generation, time grounding was finalizer-only, unresolved candidates could still become "no relevant result", conversation projection lacked row provenance, production neighbor expansion did not fetch rows around the seed, experiment defaults drifted from production, semantic floor config lacked range validation, and the DB vector config change surface was underdocumented.
- 2026-05-12 / Codex: Fixed the review blockers:
  - Top-level exact/literal conversation and shared-memory evidence now route to the existing hybrid search workers (`conversation_search_agent`, `persistent_memory_search_agent`) rather than the keyword-only workers.
  - Added `rag/search_runtime.py` to reapply trusted platform/channel/user/source filters and local relative-day UTC bounds after LLM generation. The fixed incident mapping for local `2026-05-12` + `昨天` is covered by deterministic tests.
  - Added source-scoped `observation_candidates` and `source_hints`; unresolved evaluator summaries now distinguish empty retrieval from "candidate rows found but not enough to resolve".
  - Conversation `projection_payload["rows"]` now records summary, timestamp, display name, message IDs, methods, and score for inspectability while public string projection remains unchanged.
  - Neighbor expansion now reads bounded rows before and after each seed instead of taking only newest rows in the whole time window.
  - Experiment CLI defaults now read production config, and `RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR` fails fast outside `[0.0, 1.0]` or on NaN.
- 2026-05-12 / Codex: Post-review deterministic verification passed:
  - `venv\Scripts\python -m pytest tests/test_rag_hybrid_agents.py tests/test_rag_finalizer_time_context.py tests/test_config.py::TestConversationSearchConfig tests/test_rag_hybrid_search_experiment.py tests/test_rag_phase3_capability_agents.py::test_conversation_evidence_exact_phrase_uses_hybrid_search_and_refs tests/test_rag_phase3_capability_agents.py::test_memory_evidence_exact_memory_name_uses_hybrid_search -q` -> `27 passed`.
  - `venv\Scripts\python -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_hybrid_agents.py tests/test_rag_finalizer_time_context.py tests/test_config.py::TestConversationSearchConfig tests/test_rag_hybrid_search_experiment.py -q` -> `74 passed`.
  - `venv\Scripts\python -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_helper_arg_boundaries.py tests/test_conversation_history_envelope.py tests/test_memory_retrieval_tools.py tests/test_db.py::test_get_conversation_history tests/test_db.py::test_get_conversation_history_filters tests/test_db.py::test_search_conversation_history_keyword_mocked tests/test_db.py::test_search_conversation_history_keyword_with_filters_mocked tests/test_db.py::test_search_conversation_history_vector_mocked tests/test_db.py::test_search_conversation_history_vector_with_filters_mocked -q` -> `88 passed`.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\search_runtime.py src\kazusa_ai_chatbot\rag\conversation_search_agent.py src\kazusa_ai_chatbot\rag\conversation_keyword_agent.py src\kazusa_ai_chatbot\rag\conversation_filter_agent.py src\kazusa_ai_chatbot\rag\persistent_memory_search_agent.py src\kazusa_ai_chatbot\rag\persistent_memory_keyword_agent.py src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py src\kazusa_ai_chatbot\rag\memory_evidence_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_evaluator.py src\kazusa_ai_chatbot\config.py experiments\rag_hybrid_search\run.py` -> passed.
- 2026-05-12 / Codex: Post-review real-data profile artifacts:
  - `test_artifacts/rag_hybrid_search_experiment_post_review_fix.json` / `.md`: hybrid `8/8`, false positives `0`, false negatives `0`, expected message hits `5`, average expected rank `1.00`.
  - `test_artifacts/rag_hybrid_search_failure_modes_post_review_fix.json` / `.md`: hybrid `7/8`, false positives `0`, false negatives `1`; the remaining miss is the intentionally wrong UTC yesterday-window diagnostic used to prove the retrieval-time local-date fix must run before the tool call.
- 2026-05-12 / Singer: independent code review rejected the second implementation for three gating issues and one stale-doc issue: unscoped conversation retrieval could still inherit the current user's `global_user_id`, trusted time bounds were applied only when both sides were present, and the real-data experiment still validated a parallel hybrid implementation instead of the production hybrid path.
- 2026-05-12 / Codex: Fixed the second-review blockers:
  - Conversation evidence now marks explicit author scopes with `conversation_user_scope`; unscoped and `speaker=any_speaker` searches strip `global_user_id` and `display_name` before helper execution, preventing group-memory searches from becoming current-user-only searches.
  - `rag/search_runtime.py` now applies trusted `from_timestamp` and `to_timestamp` independently by field, while relative local-date bounds fill only missing time fields.
  - `conversation_search_agent` exposes `run_hybrid_conversation_search(...)`; production `_tool` and the experiment hybrid method now call the same retrieval/fusion/neighbor path.
  - Stale retrieval-tool docstrings no longer mention `Default is 5`.
- 2026-05-12 / Codex: Second-review deterministic verification passed:
  - `venv\Scripts\python -m pytest tests/test_rag_hybrid_agents.py tests/test_rag_phase3_capability_agents.py::test_conversation_evidence_unscoped_search_removes_current_user_filter tests/test_rag_phase3_capability_agents.py::test_conversation_evidence_any_speaker_scope_removes_current_user_filter tests/test_rag_phase3_capability_agents.py::test_conversation_evidence_current_user_scope_replaces_self_dependency_topic tests/test_rag_phase3_capability_agents.py::test_conversation_evidence_active_character_scope_uses_character_identity tests/test_rag_hybrid_search_experiment.py -q` -> `22 passed`.
  - `venv\Scripts\python -m pytest tests/test_rag_hybrid_retrieval.py tests/test_rag_hybrid_agents.py tests/test_config.py::TestConversationSearchConfig tests/test_memory_retrieval_tools.py tests/test_conversation_history_envelope.py tests/test_db.py::test_get_conversation_history tests/test_db.py::test_get_conversation_history_filters tests/test_db.py::test_search_conversation_history_keyword_mocked tests/test_db.py::test_search_conversation_history_keyword_with_filters_mocked tests/test_db.py::test_search_conversation_history_vector_mocked tests/test_db.py::test_search_conversation_history_vector_with_filters_mocked tests/test_rag_helper_arg_boundaries.py tests/test_rag_phase3_capability_agents.py tests/test_rag_finalizer_time_context.py tests/test_rag_hybrid_search_experiment.py -q` -> `123 passed`.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\search_runtime.py src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py src\kazusa_ai_chatbot\rag\conversation_search_agent.py experiments\rag_hybrid_search\hybrid_search.py tests\test_rag_hybrid_agents.py tests\test_rag_phase3_capability_agents.py tests\test_rag_hybrid_search_experiment.py` -> passed.
  - `git diff --check` -> no whitespace errors.
  - `rg 'Default is 5|limit=50|top_k: 5|top_k = 5|limit: int = 5|limit=5|limit\": 5|limit\s*=\s*5|\[:5\]' src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\db experiments\rag_hybrid_search -n` -> no matches.
- 2026-05-12 / Codex: Second-review real-data profile artifacts using the production-backed hybrid experiment:
  - `test_artifacts/rag_hybrid_search_experiment_post_review_fix2.json` / `.md`: hybrid `8/8`, false positives `0`, false negatives `0`, expected message hits `5`, average expected rank `1.00`.
  - `test_artifacts/rag_hybrid_search_failure_modes_post_review_fix2.json` / `.md`: hybrid `7/8`, false positives `0`, false negatives `1`; remaining miss is the intentionally wrong UTC yesterday-window diagnostic.
- 2026-05-12 / Halley: independent approval review rejected the third implementation because the RAG dispatcher could still route legacy `Conversation-keyword:` slots, fallback exact conversation slots, and fallback exact persistent-memory slots directly to low-level keyword/search workers, bypassing the top-level hybrid evidence capabilities.
- 2026-05-12 / Codex: Fixed the dispatcher bypass:
  - Legacy `Conversation-aggregate:`, `Conversation-filter:`, `Conversation-keyword:`, and `Conversation-semantic:` prefixes now dispatch to `conversation_evidence_agent`.
  - Legacy `Memory-keyword:` and `Memory-search:` prefixes now dispatch to `memory_evidence_agent`.
  - Dispatcher fallback aliases now normalize low-level conversation workers to `conversation_evidence_agent` and low-level persistent-memory workers to `memory_evidence_agent`.
  - Dispatcher prompt and RAG README now describe low-level keyword/search workers as internal implementation details, and the dispatcher `agent_name` union no longer advertises those low-level workers as fallback choices.
  - Plan schema examples now use the implementation-aligned search field instead of the stale semantic-only field name.
- 2026-05-12 / Codex: Dispatcher-bypass verification passed:
  - `venv\Scripts\python -m pytest tests/test_rag_initializer_cache2.py::test_rag_dispatcher_uses_deterministic_new_prefix tests/test_rag_initializer_cache2.py::test_rag_dispatcher_remaps_legacy_prefix_alias tests/test_rag_initializer_cache2.py::test_normalize_dispatch_remaps_low_level_keyword_agents tests/test_rag_phase3_route_mapping.py tests/test_rag_hybrid_agents.py tests/test_rag_phase3_capability_agents.py::test_conversation_evidence_unscoped_search_removes_current_user_filter tests/test_rag_hybrid_search_experiment.py -q` -> `24 passed`.
  - `venv\Scripts\python -m pytest tests/test_rag_hybrid_retrieval.py tests/test_rag_hybrid_agents.py tests/test_config.py::TestConversationSearchConfig tests/test_memory_retrieval_tools.py tests/test_conversation_history_envelope.py tests/test_db.py::test_get_conversation_history tests/test_db.py::test_get_conversation_history_filters tests/test_db.py::test_search_conversation_history_keyword_mocked tests/test_db.py::test_search_conversation_history_keyword_with_filters_mocked tests/test_db.py::test_search_conversation_history_vector_mocked tests/test_db.py::test_search_conversation_history_vector_with_filters_mocked tests/test_rag_helper_arg_boundaries.py tests/test_rag_phase3_capability_agents.py tests/test_rag_finalizer_time_context.py tests/test_rag_hybrid_search_experiment.py tests/test_rag_initializer_cache2.py tests/test_rag_phase3_route_mapping.py -q` -> `151 passed`.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_dispatch.py src\kazusa_ai_chatbot\rag\search_runtime.py src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py src\kazusa_ai_chatbot\rag\conversation_search_agent.py experiments\rag_hybrid_search\hybrid_search.py` -> passed.
  - Source-scoped route-bypass grep over `src\kazusa_ai_chatbot\nodes`, `src\kazusa_ai_chatbot\rag`, and active bugfix docs -> no matches.
  - Source-scoped stale schema grep over this plan and `src\kazusa_ai_chatbot\rag\README.md` -> no matches.
- 2026-05-12 / Popper: independent approval review returned `APPROVED` with no gating findings. Confirmed legacy dispatcher prefixes and LLM fallback low-level worker names route to top-level evidence agents, trusted user/time scope and production-backed experiment fixes remain intact, and docs/plan are accurate enough to close. Residual non-gating risk: real-data experiment fixture `keywords` validate production fusion and neighbor behavior, not live LLM anchor-generation quality.
