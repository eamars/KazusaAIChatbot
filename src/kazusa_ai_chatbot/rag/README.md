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
    "character_image": dict,
    "third_party_profiles": list[str],
    "memory_evidence": list[dict],
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

Slots are intentionally semantic and dependency-aware:

```python
[
    "Identity: look up display name '小钳子' to get global_user_id",
    "Profile: retrieve full user profile for the user resolved in slot 1",
    "Conversation-keyword: find messages from the user resolved in slot 1 containing a URL",
    "Web-search: retrieve the content at the URL found in slot 3",
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

The dispatcher first checks the slot prefix. Recognized prefixes map directly to specialist agents:

| Slot prefix | Primary agent |
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

If a slot has no recognized prefix, the dispatcher falls back to semantic routing rules: identity resolution first, then user enumeration, profile reads when a `global_user_id` is already known, exact conversation search, aggregate conversation questions, structured conversation filters, fuzzy conversation search, persistent-memory search, and finally web search.

The executor then delegates to the chosen helper agent with a concise task description and the accumulated `known_facts`. The evaluator summarizes the result, attaches source policy metadata, appends a fact row, and removes the slot from `unknown_slots`.

## Retrieval Capabilities

RAG 2 groups helper agents by semantic ownership:

- **Identity and profile retrieval** resolves users, enumerates users, reads user or character profile bundles, and ranks relationship-like profile state.
- **Conversation retrieval** searches or filters historical messages by semantic query, keyword, structured filters, or aggregate questions.
- **Persistent memory retrieval** searches durable memory records by semantic query or exact keyword.
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

The dispatcher-visible helper agents are:

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
- conversation evidence is summarized,
- persistent-memory and external evidence keep short clipped evidence text,
- dispatch trace remains available as retrieval telemetry, not as factual proof.

This keeps cognition grounded without flooding prompts with raw search hits.

## Cache 2

Cache 2 is the only RAG cache. It is a process-local, session-scoped LRU used by the helper-agent layer and initializer strategy lookup.

Cache 2 is intentionally not a durable memory store:

- entries disappear on service restart,
- there is no MongoDB write-through cache collection,
- correctness depends on dependency-based invalidation rather than long TTLs,
- web search and final answers are not cached.

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
- initializer Cache 2 reuse,
- `rag_result` projection,
- Cache 2 invalidation from conversation saves,
- Cache 2 invalidation from consolidator writes,
- helper-agent focused tests,
- live RAG supervisor smoke tests.

Live LLM tests are evidence-producing checks. Inspect their logs and returned evidence, not only their pass/fail status.
