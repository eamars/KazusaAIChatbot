# RAG 2

`kazusa_ai_chatbot.rag` is the helper-agent and Cache 2 layer for the current RAG 2 retrieval system.

RAG 2 turns a user query into bounded factual evidence for persona cognition. It resolves identities, searches conversation history and persistent memory, reads profile-like state, performs narrow factual reductions, and can delegate to web search when external information is needed.

It is not a reply generator, not a persona reasoning layer, not a relationship judge, and not long-term memory consolidation. Its job is to transform data into evidence. Cognition decides what that evidence means for Kazusa's stance, tone, and final response.

## System Boundary

The runtime entry point is the progressive supervisor:

```python
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor2 import call_rag_supervisor
```

`call_rag_supervisor(...)` returns:

```python
{
    "answer": str,
    "known_facts": list[dict],
    "unknown_slots": list[str],
    "loop_count": int,
}
```

The persona graph does not pass this raw shape directly to cognition. `stage_1_research` projects `known_facts` into `rag_result`, the compact cognition/consolidation payload:

```python
{
    "answer": str,
    "user_image": dict,
    "user_memory_unit_candidates": list[dict],
    "character_image": dict,
    "third_party_profiles": list[str],
    "memory_evidence": list[dict],
    "recall_evidence": list[dict],
    "conversation_evidence": list[str],
    "external_evidence": list[dict],
    "supervisor_trace": {
        "loop_count": int,
        "unknown_slots": list[str],
        "dispatched": list[dict],
    },
}
```

This split is intentional. RAG 2 keeps raw evidence and agent telemetry available inside the supervisor, then exposes only the evidence shape downstream prompts need.

## Runtime Lifecycle

```text
persona_supervisor2
  -> stage_0_msg_decontexualizer
  -> stage_1_research
       call_rag_supervisor(original_query, character_name, context)
       -> rag_initializer
            decomposes the query into ordered unknown_slots
       -> rag_dispatcher
            chooses one specialist helper agent for the current slot
       -> rag_executor
            delegates to that helper agent
       -> rag_evaluator
            summarizes the result, records provenance, drains the slot
       -> repeat dispatcher/executor/evaluator until:
            unknown_slots is empty, no valid dispatch remains, or loop cap is reached
       -> rag_finalizer
            produces a compact factual synthesis
       project_known_facts(...)
       -> state["rag_result"]
  -> stage_2_cognition
  -> stage_3_action
```

The loop is slot-driven. `unknown_slots` is the work queue and the stop condition: slots are drained one at a time, and each completed attempt appends a fact row to `known_facts`.

The default loop cap is eight dispatch iterations. This prevents open-ended agentic search while still allowing multi-hop queries such as resolving a pronoun, finding a referenced conversation object, then reading external content.

## Design Intention

RAG 2 is built for a local, latency-sensitive LLM runtime. The architecture keeps broad planning separate from low-level retrieval details:

- The initializer decides what facts are missing.
- The dispatcher chooses the semantic capability for the next missing fact.
- A specialist helper agent owns argument generation for one retrieval domain.
- Deterministic code executes tools, validates structure, applies cache policy, and records provenance.
- The evaluator summarizes the result for downstream consumers.
- The finalizer synthesizes factual context without speaking as Kazusa.

This avoids asking one prompt to understand every backend schema, generate perfect database parameters, judge results, and produce a final answer in a single step.

## Initializer And Dispatch

The initializer receives the decontextualized user query, the active character name, and runtime context such as platform, channel, current user, timestamp, and recent chat hints. It does not call storage backends. It produces an ordered list of `unknown_slots`, where each slot describes one missing fact.

Slots are intentionally semantic and dependency-aware. New initializer output
uses top-level capability prefixes:

```python
[
    "Conversation-evidence: retrieve exact phrase '版权保护一直都是play的一环' to identify the speaker",
    "Person-context: retrieve profile/impression for speaker found in slot 1",
    "Conversation-evidence: retrieve messages from the user resolved in slot 2 containing a URL",
    "Web-evidence: retrieve public web content for the URL found in slot 3",
]
```

The slot order matters. Later slots can refer to earlier results, such as "the user resolved in slot 1" or "the URL found in slot 3". This is how RAG 2 handles multi-hop requests without forcing one agent to solve the whole query.

The dispatcher handles one slot at a time. For each loop, it receives:

```python
{
    "current_slot": str,
    "known_facts": list[dict],
    "context": dict,
}
```

The dispatcher first checks the slot prefix. Recognized prefixes map directly
to one top-level capability or compatibility worker without calling the
dispatcher LLM:

| Slot prefix | Primary agent |
|---|---|
| `Live-context:` | `live_context_agent` |
| `Conversation-evidence:` | `conversation_evidence_agent` |
| `Memory-evidence:` | `memory_evidence_agent` |
| `Person-context:` | `person_context_agent` |
| `Web-evidence:` | `web_search_agent2` |
| `Recall:` | `recall_agent` |

Legacy worker prefixes remain accepted as compatibility aliases:

| Compatibility prefix | Primary agent |
|---|---|
| `Identity:` | `user_lookup_agent` |
| `User-list:` | `user_list_agent` |
| `Relationship:` | `relationship_agent` |
| `Profile:` | `user_profile_agent` |
| `Conversation-aggregate:` | `conversation_aggregate_agent` |
| `Conversation-filter:` | `conversation_filter_agent` |
| `Conversation-keyword:` | `conversation_keyword_agent` |
| `Conversation-semantic:` | `conversation_search_agent` |
| `Memory-search:` | `persistent_memory_search_agent` |
| `Web-search:` | `web_search_agent2` |

If a slot has no recognized prefix, the dispatcher falls back to semantic
routing rules for compatibility. Normal generated slots should use the
top-level prefixes so the dispatcher path is deterministic.

The executor then delegates to the chosen helper agent with a concise task description and the accumulated `known_facts`. The evaluator summarizes the result, attaches source policy metadata, appends a fact row, and removes the slot from `unknown_slots`.

## Retrieval Capabilities

RAG 2 groups helper agents by semantic ownership:

- **Live context** resolves target/scope for changing external facts such as weather, temperature, opening status, schedules, prices, and exchange rates, then delegates to web search. Memory may be used only for stable target/scope lookup, not as the live value source.
- **Conversation evidence** searches or filters historical messages by semantic query, keyword, structured filters, or aggregate questions.
- **Memory evidence** is scope-aware. It searches durable shared/world/common-sense/character facts by semantic query or exact identifiers, and it can also retrieve current-user scoped continuity from `user_memory_units` when the slot is about private continuity rather than profile or relationship state.
- **Person context** resolves users, enumerates users, reads user or character profile bundles, and ranks relationship-like profile state.
- **Recall retrieval** reconciles active agreements, ongoing promises, plans, open loops, and current-episode state.
- **External retrieval** performs web search and URL reads when the required fact is outside local storage.

Each helper agent exposes the same high-level contract:

```python
async def run(
    task: str,
    context: dict,
    max_attempts: int = 3,
) -> dict
```

Supervisor-compatible results include:

```python
{
    "resolved": bool,
    "result": object,
    "attempts": int,
    "cache": dict,
}
```

Many retrieval helpers use a generator -> tool -> judge inner loop. The generator produces arguments only for that helper's tool, the tool call is deterministic, and the judge decides whether the slot is resolved or whether another attempt should refine the arguments.

## Helper Agent Roles

The dispatcher-visible top-level capability agents are:

| Agent | Responsibility |
|---|---|
| `live_context_agent` | Resolves target/scope for live external facts, then delegates to `web_search_agent2`. It refuses missing location/target instead of guessing. |
| `conversation_evidence_agent` | Chooses the appropriate conversation worker for exact phrases, fuzzy topics, structured filters, counts, URLs, and speaker provenance. |
| `memory_evidence_agent` | Chooses among shared semantic memory, shared exact-memory lookup, and scoped current-user continuity retrieval. Natural-language home/address questions use shared semantic memory search; literal memory identifiers use shared keyword search; current-user private continuity uses `user_memory_evidence_agent`. |
| `person_context_agent` | Chooses identity, profile, user-list, relationship, or the approved display-name to profile chain. |
| `recall_agent` | Reconciles active agreements, promises, plans, and current-episode state from scoped volatile sources. |

Top-level capability results keep structured handoff inside RAG 2:

```python
{
    "selected_summary": str,
    "capability": str,
    "primary_worker": str,
    "source_policy": str,
    "resolved_refs": list[dict],
    "projection_payload": dict,
    "worker_payloads": dict,
    "evidence": list[str],
    "missing_context": list[str],
    "conflicts": list[str],
}
```

`resolved_refs` is the approved cross-slot channel for person IDs, message
provenance, URLs, locations, and memory refs. `projection_payload` is the
approved projection channel into public `rag_result` fields. `worker_payloads`
is trace/debug material only.

The reusable worker agents are:

| Agent | Responsibility |
|---|---|
| `user_lookup_agent` | Resolves one display name to a user profile identity and `global_user_id`. It is for "who is this named person?" questions, not message-content search. |
| `user_list_agent` | Enumerates users by display-name predicates or participant metadata, such as names that equal, contain, start with, or end with a literal value. |
| `user_profile_agent` | Reads a full user or character profile bundle after an identity is already known. It should not be used to discover unknown identities. |
| `relationship_agent` | Produces factual rankings from stored relationship/profile state, such as top or bottom relationship-like scores. It returns evidence, not persona judgment. |
| `conversation_aggregate_agent` | Computes factual aggregates over conversation history, such as message counts, speaker rankings, or who mentioned a literal term most often. |
| `conversation_filter_agent` | Retrieves conversation rows by structured filters: known user, channel, timestamp range, display name, or requested message count. It is preferred when concrete filters exist. |
| `conversation_keyword_agent` | Searches message content for exact strings, URLs, filenames, proper nouns, and phrases that must appear verbatim. |
| `conversation_search_agent` | Performs semantic conversation recall when the topic is fuzzy or the exact wording is unknown. It is the fallback for conversation content after filters and keywords are unsuitable. |
| `persistent_memory_keyword_agent` | Searches durable memories by exact keyword or phrase, useful for tags, event names, and proper nouns. |
| `persistent_memory_search_agent` | Searches durable memories semantically for impressions, commitments, facts, and other remembered knowledge when exact wording is unknown. |
| `user_memory_evidence_agent` | Searches `user_memory_units` for the current `global_user_id` only. It uses vector retrieval when available, explicit literal lexical retrieval for exact continuity anchors, and bounded recency fallback. Returned rows keep `source_system`, `scope_type`, `scope_global_user_id`, `authority`, `truth_status`, and `origin`. |
| `web_search_agent2` | Searches or reads public web content when the requested fact cannot come from local profiles, memories, or conversation history. |

Most agents are evidence retrievers. They should answer "what was found?" rather than "what should Kazusa think about it?" The only ranking-style agents still return factual rankings from stored data or message counts; interpretation remains downstream.

## Evidence Shape

Each fact recorded by the supervisor keeps both operational and provenance information:

```python
{
    "slot": str,
    "agent": str,
    "resolved": bool,
    "summary": str,
    "raw_result": object,
    "attempts": int,
    "cache": dict,
    "fact_source": dict,
}
```

`summary` is the normal prompt-facing evidence for search-like agents. `raw_result` is retained where structure matters, especially user and character profile bundles. The projection layer applies a hybrid policy:

- structured profile/image bundles remain structured,
- scoped `user_memory_units` rows surfaced through `Memory-evidence:` remain in `memory_evidence` with scope metadata preserved and are also appended to `rag_result.user_memory_unit_candidates` for consolidation merge/evolve reuse,
- conversation evidence is summarized,
- shared memory evidence is summarized,
- live/external evidence keeps URL-bearing text,
- recall evidence stays structured because downstream stages inspect fields such as status and temporal scope.

## Scope And Provenance

`Memory-evidence:` now spans two durable-memory sources with different authority boundaries:

- shared `memory` rows for official facts, common sense, seeded lore, and other non-user-scoped durable memory,
- scoped `user_memory_units` rows for current-user continuity.

When the source is `user_memory_units`, projected evidence preserves:

- `source_system="user_memory_units"`,
- `scope_type="user_continuity"`,
- `scope_global_user_id=<current global_user_id>`,
- `authority="scoped_continuity"` unless the stored row already provides a stronger value,
- `truth_status="character_lore_or_interaction_continuity"` unless the stored row already provides a stronger value,
- `origin="consolidated_interaction"` unless the stored row already provides a stronger value.

This keeps current-user private continuity distinct from shared/global durable memory and lets the consolidator reuse the exact surfaced units instead of creating duplicates.

## Future Direction

Long-term global lore promotion remains future work. The current implementation does not promote scoped `user_memory_units` continuity into shared/global lore automatically. Any future promotion pipeline must preserve provenance and conflict handling instead of flattening user continuity into unscoped durable memory.

## Cache 2

Cache 2 is the only RAG cache. Its hot serving layer is a process-local LRU used by the helper-agent layer and initializer strategy lookup.

Helper-agent result cache entries are intentionally session-local:

- helper-agent entries disappear on service restart,
- helper-agent results are not written through to MongoDB,
- correctness depends on dependency-based invalidation rather than long TTLs,
- web search and final answers are not cached.

Top-level capability agents are not cached. Their cache metadata reports
`enabled=false` and `reason="capability_orchestrator_uncached"`. They may call
existing cacheable workers, whose Cache 2 policies remain source-aware.

The `rag_initializer` strategy cache is the one durable exception. Current-version
initializer rows are persisted in `rag_cache2_persistent_entries`, loaded into
the Python LRU during service startup, and ranked by `hit_count desc,
updated_at desc` so the most useful paths survive restart. Normal request-path
cache reads still check only the Python LRU. A new cacheable initializer path
stores in memory first and then schedules a best-effort MongoDB upsert; an
initializer memory hit schedules a best-effort hit-count update. LRU eviction
does not delete the persistent row.

Cache entries declare the data scopes they depend on, such as `user_profile`, `character_state`, or `conversation_history`. Durable write paths emit invalidation events after successful writes. The runtime evicts cached entries whose dependencies overlap the event scope.

```text
helper agent result
  -> cache entry with CacheDependency(...)

durable write
  -> CacheInvalidationEvent(...)

Cache 2 runtime
  -> dependency/event overlap
  -> evict stale entries
```

Conversation retrieval is cached only for closed historical windows with both `from_timestamp` and `to_timestamp`. Recent or open-ended conversation queries are deliberately not cached because new messages can make them stale immediately.

## Integration With Cognition

RAG 2 runs before cognition in `persona_supervisor2`.

Cognition reads `rag_result`, not raw RAG supervisor state. The intended division is:

- RAG 2 answers "what evidence exists?"
- Cognition answers "what does this mean for Kazusa right now?"
- Dialog generation answers "how should Kazusa say it?"

RAG evidence should not encode persona stance, emotional interpretation, or user-facing wording. If a retrieved fact is ambiguous, RAG should preserve uncertainty in evidence or summary form rather than resolving it as character intent.

## Integration With Consolidation

The consolidator also reads the projected `rag_result`.

RAG 2 retrieval does not itself write new durable knowledge. Durable writes happen later in the background consolidation path after cognition and dialog have completed. Cache invalidation is tied to those successful durable writes, not to whether a RAG result was used by cognition.

`recall_evidence` is operational provenance for agreements, promises, plans, and
current episode progress. A progress-only recall can guide the current reply but
does not by itself authorize a durable character fact. The consolidator should
require user input, durable memory, conversation evidence, or external evidence
before turning recalled progress into stable knowledge.

This keeps ownership clear:

- RAG helper agents retrieve evidence.
- Cognition and dialog may use that evidence.
- The consolidator decides what durable state changes are accepted.
- Cache 2 invalidates from durable write events.

## Semantic Ownership

LLMs own semantic retrieval decisions:

- what slots a query needs,
- which helper capability should handle a slot,
- how to frame semantic or keyword queries,
- whether a helper result resolves the slot,
- how to summarize retrieved evidence.

Deterministic code owns mechanics:

- graph routing and loop caps,
- registry membership,
- schema-shaped result handling,
- cache keys and invalidation scopes,
- prompt-facing projection,
- text clipping and structural budgets,
- database/tool execution.

Do not move natural-language interpretation into regexes, keyword classifiers, or post-hoc code filters. If a semantic boundary is wrong, fix the relevant prompt, helper-agent contract, or projection policy.

## Operational Notes

RAG 2 replaced the legacy RAG 1 / Cache 1 path. There is no compatibility adapter between the old `research_facts` shape and the current `rag_result` shape.

Cache 2 exposes per-agent hit/miss statistics through service health data. These counters are operational telemetry only; they should not affect retrieval or cognition decisions.

The most important behavior checks are:

- progressive supervisor integration,
- initializer Cache 2 reuse, startup hydration, and persistent hit counting,
- `rag_result` projection,
- Cache 2 invalidation from conversation saves,
- Cache 2 invalidation from consolidator writes,
- helper-agent focused tests,
- live RAG supervisor smoke tests.

Live LLM tests are evidence-producing checks. Inspect their logs and returned evidence, not only their pass/fail status.

Production observability follows a split log policy:

- INFO keeps key operational breadcrumbs: initializer slots, dispatch
  `agent/task/route_source`, top-level capability
  `resolved/primary_worker/missing_context/selected_summary/cache reason`,
  evaluator fact summaries, and finalizer answers.
- DEBUG keeps supporting detail: `resolved_refs`, `projection_payload`,
  `worker_payloads`, raw context, raw payloads, cache keys, and compacted
  LLM-facing views.

Large raw capability payloads remain available in supervisor state and trace
artifacts, but evaluator/finalizer and downstream delegate prompts receive
compacted views so local LLM context windows are not consumed by full profile or
worker payloads.
