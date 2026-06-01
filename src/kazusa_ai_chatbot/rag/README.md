# RAG 2

`kazusa_ai_chatbot.rag` is the helper-agent and Cache 2 layer for the current RAG 2 retrieval system.

RAG 2 turns a user query into bounded factual evidence for persona cognition. It resolves identities, searches conversation history and persistent memory, reads profile-like state, performs narrow factual reductions, and can delegate to web search when external information is needed.

It transforms data into evidence for cognition. Cognition decides what that
evidence means for Kazusa's stance, tone, and final response.

## System Boundary

The live persona graph reaches RAG through the cognition resolver's evidence
capability. The retained helper boundary is:

```python
from kazusa_ai_chatbot.nodes.persona_supervisor2 import run_rag_evidence_for_persona_state
```

That helper wraps the quote-aware RAG supervisor and projects the result into
the compact persona payload. The lower-level supervisor returns:

```python
{
    "answer": str,
    "known_facts": list[dict],
    "unknown_slots": list[str],
    "loop_count": int,
}
```

The persona graph does not pass this raw shape directly to cognition.
`run_rag_evidence_for_persona_state(...)` projects `known_facts` into
`rag_result`, the compact cognition/consolidation payload:

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
Continuation metadata, when present, is kept under
`supervisor_trace.dispatched[*].continuation`; it is not public evidence for
cognition.

Public memory, recall, and conversation evidence is formatted for cognition
with the same semantic order:

```text
结论：直接事实答案，或明确说明没有找到证据。
上下文：
- 来源或发言人在配置的本地时间：可读的支撑内容。
不确定性：仍然存在的不确定、冲突，或“无”。
```

Prompt-facing generated RAG wording is Chinese-first because the primary chat
data, retrieval anchors, and cognition consumer are Chinese. Stable code-level
route tokens, JSON keys, source text, URLs, filenames, model labels, and quoted
user content keep their original spelling.

Prompt-facing evidence must not expose raw adapter wire syntax, raw attachment
URLs, storage ids, embeddings, binary payloads, source rows, or raw UTC
timestamps. Source ids and raw refs stay in `supervisor_trace` or helper
payloads for debugging, not in the primary evidence text consumed by
cognition.

## Prompt-Facing Safety Sanitization Policy

RAG safety checks protect the cognition prompt from raw storage, adapter, and
binary material. They must not make the live chat graph fragile. The
operational rule is:

```text
Unsafe prompt-facing RAG evidence degrades RAG evidence; it must not crash the
turn.
```

The projection boundary owns this policy. `project_known_facts(...)` converts
supervisor `known_facts` into `rag_result`, then applies prompt-facing
sanitization before cognition receives the payload. The safety policy covers
the public evidence fields:

- `answer`
- `third_party_profiles`
- `memory_evidence`
- `recall_evidence`
- `conversation_evidence`
- `external_evidence`

`supervisor_trace` is not public evidence. It may contain compact provenance,
dispatch metadata, continuation metadata, and capped recovery incident labels
for debugging. Cognition prompt projections strip recovery incident labels from
`supervisor_trace` so local LLM stages do not receive sanitization internals as
semantic evidence.

Sanitization follows a recover-first order:

1. Preserve safe evidence unchanged. `external_evidence[*].url` is validated as
   a URL field, not scanned as prose; valid `http` and `https` URLs may contain
   ordinary query parameters such as `url=` and UUID-like path segments.
2. Strip or rewrite recoverable source markers from prose using the shared
   public RAG evidence sanitizer. Trace-only provenance markers such as
   `evidence_time=` are handled by this shared boundary, not by
   source-specific projection exceptions.
3. Drop raw adapter wire lines such as CQ message syntax. CQ material should be
   stopped by adapters before RAG; if it reaches RAG evidence, it is ignored
   rather than shown to cognition.
4. Drop only the unsafe line when a multi-line evidence block contains both
   usable evidence and raw unsafe material.
5. Blank optional malformed external URL fields when the surrounding external
   evidence text is still usable.
6. Drop an evidence item when every useful public field in that item is unsafe
   or empty after recovery.
7. Empty the public RAG evidence fields if residual unsafe material remains
   after recovery.

All recovery incidents are operational signals, not user-facing facts. The
projection layer logs a warning and records capped labels under
`rag_result.supervisor_trace.safety_recovery`. Repeated incidents indicate an
upstream projection or helper-agent contract bug that should be fixed, but the
current user turn should continue with partial or empty RAG evidence.

This policy applies only at the prompt-facing evidence boundary. Raw helper
payloads may still keep source ids, refs, and worker material for trace/debug
use while they remain outside cognition-visible evidence fields.

## Cognitive Episode Adapter

`cognitive_episode_adapter.py` owns the text-chat projection from
`CognitiveEpisode` into the current RAG request boundary.
`build_text_chat_rag_request(...)` is the public entrypoint. RAG request
construction is centralized there; the rest of RAG continues to receive
`original_query`, `character_name`, and `context` rather than raw
`CognitiveEpisode` payloads.

## Package Boundary

RAG helper agents are organized by dispatcher-visible capability. Public
callers import the top-level capability class from the package:

```python
from kazusa_ai_chatbot.rag.conversation_evidence import ConversationEvidenceAgent
from kazusa_ai_chatbot.rag.memory_evidence import MemoryEvidenceAgent
from kazusa_ai_chatbot.rag.person_context import PersonContextAgent
from kazusa_ai_chatbot.rag.live_context import LiveContextAgent
from kazusa_ai_chatbot.rag.recall import RecallAgent
```

The global RAG dispatcher remains an explicit registry. It does not discover
RAG packages dynamically. The only local source discovery in RAG remains inside
`web_agent3/subagent`, where source modules are part of the web helper's own
provider router.

Package ownership is:

| Package | Owns |
|---|---|
| `conversation_evidence/` | Conversation-history evidence selection, projection, active-turn exclusion, and conversation workers. |
| `memory_evidence/` | Durable memory evidence selection, projection, shared memory workers, and scoped current-user memory evidence. |
| `person_context/` | Person/profile selector, projection, identity lookup, user listing, profile, relationship, and image hydration workers. |
| `live_context/` | Runtime date/time facts, live external target resolution, and web delegation for changing public facts. |
| `recall/` | Active agreement, commitment, scheduled-event, current-episode, and gated history recall. |
| `web_agent3/` | Public web search/read routing and source-local web subagents. |

Shared RAG utility modules stay flat unless a capability move directly requires
a local import update. Shared utilities include Cache 2 policy/runtime,
cognitive episode projection, evidence formatting, hybrid retrieval, memory
retrieval tools, prompt projection, quote-aware sequence handling, search
runtime, and user-memory-unit retrieval.

Split rule: when a helper grows beyond a single inspectable ownership boundary,
split inside its capability package first. Do not add global auto-discovery,
compatibility import shims, fallback import paths, new helper modes, or prompt
behavior changes as part of a structural split.

## Runtime Lifecycle

```text
persona_supervisor2
  -> stage_0_msg_decontexualizer
  -> stage_1_goal_resolver
       L1 -> L2 -> L2d cognition cycle
       L2d selects `rag_evidence` or `web_evidence` only when evidence is needed
       call_rag_supervisor(original_query, character_name, context)
       -> rag_initializer
            decomposes the query into ordered unknown_slots
       -> rag_dispatcher
            chooses one specialist helper agent for the current slot
       -> rag_executor
            delegates to that helper agent
       -> rag_evaluator
            summarizes the result, records provenance, drains the slot,
            and may run a bounded refined-query initializer re-entry from
            unresolved observation material
       -> repeat dispatcher/executor/evaluator until:
            unknown_slots is empty, no valid dispatch remains, or loop cap is reached
       -> rag_finalizer
            produces a compact factual synthesis
       project_known_facts(...)
       -> resolver observation with state["rag_result"]
       next resolver cycle consumes the projected evidence
  -> stage_3_action
```

The RAG loop is slot-driven. `unknown_slots` is the work queue and the stop
condition: slots are drained one at a time, and each completed attempt appends
a fact row to `known_facts`. The persona graph decides whether to enter this
loop through L2d action selection, not through a mandatory pre-cognition step.

The default loop cap is four dispatch iterations. This prevents open-ended
agentic search and keeps normal chatbot latency bounded. Future RAG
optimization should reduce the need for repeated loops instead of raising this
cap.

## Observation-Driven Continuation

RAG 2 can perform a bounded refined-query re-entry after an unresolved
retrieval when that retrieval returned observation material with useful
direction. The continuation path refines the query and reuses the existing
initializer/dispatcher/helper flow:

```text
unresolved retrieval result
  -> observation candidates or source hints
  -> continuation refiner
  -> should_continue=false: stop RAG normally
  -> should_continue=true: produce a self-contained refined query
  -> existing rag_initializer(refined_query) with Cache 2 active
  -> existing dispatcher and specialist agents execute initializer slots
```

Observation candidates are rows or snippets that did not answer the current
slot but may improve the next initializer pass. For example, a durable memory
row may say that latest prices, benchmark scores, schedules, or driver state
should come from fresh retrieval. That row is not accepted answer evidence for
the concrete question, but it can be folded into a refined natural-language
query.

The continuation refiner returns only:

```python
{
    "should_continue": bool,
    "refined_query": str,
    "reason": str,
}
```

`reason` is trace-only. The refiner emits a self-contained natural-language
query; initializer and dispatcher stages own slots, agent choice, and backend
parameters. When `should_continue` is true, `refined_query` is self-contained
because Cache 2 keys the initializer from its normal query and context inputs.

Continuation is capped by `MAX_CONTINUATION_DECISIONS_PER_RAG_RUN` and the
existing four-loop supervisor cap. Rejected candidates and continuation
metadata stay in supervisor/debug trace. Public evidence remains limited to
accepted `rag_result` fields.

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
| `Web-evidence:` | `web_agent3` |
| `Recall:` | `recall_agent` |

Legacy worker prefixes remain accepted as compatibility aliases:

| Compatibility prefix | Primary agent |
|---|---|
| `Identity:` | `user_lookup_agent` |
| `User-list:` | `user_list_agent` |
| `Relationship:` | `relationship_agent` |
| `Profile:` | `user_profile_agent` |
| `Conversation-aggregate:` | `conversation_evidence_agent` |
| `Conversation-filter:` | `conversation_evidence_agent` |
| `Conversation-keyword:` | `conversation_evidence_agent` |
| `Conversation-semantic:` | `conversation_evidence_agent` |
| `Memory-keyword:` | `memory_evidence_agent` |
| `Memory-search:` | `memory_evidence_agent` |
| `Web-search:` | `web_agent3` |

If a slot has no recognized prefix, the dispatcher falls back to semantic
routing rules for compatibility. Normal generated slots should use the
top-level prefixes so the dispatcher path is deterministic.

The executor then delegates to the chosen helper agent with a concise task description and the accumulated `known_facts`. The evaluator summarizes the result, attaches source policy metadata, appends a fact row, and removes the slot from `unknown_slots`.

## Retrieval Capabilities

RAG 2 groups helper agents by semantic ownership:

- **Live context** resolves target/scope for changing external facts such as weather, temperature, opening status, schedules, prices, and exchange rates, then delegates to web search. Memory may be used only for stable target/scope lookup, not as the live value source.
- **Conversation evidence** searches or filters historical messages by semantic query, keyword, structured filters, or aggregate questions.
- **Memory evidence** is scope-aware. It searches durable shared/world/common-sense/character facts by semantic query or exact identifiers, and it can also retrieve current-user scoped continuity from `user_memory_units` when the slot is about private continuity, recognition, accepted preferences, user-specific lore, or prior shared interactions rather than profile or relationship state.
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
| `live_context_agent` | Resolves target/scope for live external facts, then delegates to `web_agent3`. It refuses missing location/target instead of guessing. |
| `conversation_evidence_agent` | Chooses the appropriate conversation worker for hybrid exact/fuzzy evidence, structured filters, counts, URLs, and speaker provenance. |
| `memory_evidence_agent` | Chooses among shared hybrid memory retrieval and scoped current-user continuity retrieval. Natural-language home/address questions and literal memory identifiers both use shared hybrid memory search; current-user recognition, private continuity, accepted preferences, user-specific lore, and prior shared interactions use `user_memory_evidence_agent`. |
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
    "observation_candidates": list[dict],
    "source_hints": list[dict],
}
```

`resolved_refs` is the approved cross-slot channel for person IDs, message
provenance, URLs, locations, and memory refs. `projection_payload` is the
approved projection channel into public `rag_result` fields. `worker_payloads`
is trace/debug material only.
`observation_candidates` is also trace/internal material only. It carries
unresolved candidate rows that may guide continuation, not accepted answer
evidence. `source_hints` gives compact provenance for those unresolved
observations.

The reusable worker agents are:

| Agent | Responsibility |
|---|---|
| `user_lookup_agent` | Resolves one display name to a user profile identity and `global_user_id`. It is for "who is this named person?" questions, not message-content search. |
| `user_list_agent` | Enumerates users by display-name predicates or participant metadata, such as names that equal, contain, start with, or end with a literal value. |
| `user_profile_agent` | Reads a full user or character profile bundle after an identity is already known. Identity discovery belongs to person-context helpers. |
| `relationship_agent` | Produces factual rankings from stored relationship/profile state, such as top or bottom relationship-like scores. It returns evidence, not persona judgment. |
| `conversation_aggregate_agent` | Computes factual aggregates over conversation history, such as message counts, speaker rankings, or who mentioned a literal term most often. |
| `conversation_filter_agent` | Retrieves conversation rows by structured filters: known user, channel, timestamp range, display name, or requested message count. It is preferred when concrete filters exist. |
| `conversation_keyword_agent` | Lower-level exact-string search worker used by tools and hybrid fusion. Top-level conversation evidence does not route literal recall here directly. The shared search config controls default and maximum result counts. |
| `conversation_search_agent` | Performs hybrid conversation recall for fuzzy topics and literal anchors. Its LLM generator may emit a semantic query plus literal anchors; deterministic code reapplies trusted platform/channel/time filters plus explicit author-scope filters, runs bounded semantic and keyword retrieval, fuses candidates, and expands narrow neighboring context around accepted evidence. |
| `persistent_memory_keyword_agent` | Lower-level exact-keyword memory worker used by hybrid fusion. Top-level shared-memory evidence does not route literal recall here directly. |
| `persistent_memory_search_agent` | Performs hybrid durable-memory recall for exact and fuzzy shared memory evidence. It fuses semantic memory rows with literal-anchor keyword rows, enforces trusted source filters, and keeps scoped `user_memory_evidence_agent` separate. |
| `user_memory_evidence_agent` | Searches `user_memory_units` for the current `global_user_id` only. It uses vector retrieval when available, explicit literal lexical retrieval for exact continuity anchors, and bounded recency fallback. Returned rows keep `source_system`, `scope_type`, `scope_global_user_id`, `authority`, `truth_status`, and `origin`. |
| `web_agent3` | Searches or reads public web content when the requested fact belongs outside local profiles, memories, or conversation history. |

Most agents are evidence retrievers. They should answer "what was found?" rather than "what should Kazusa think about it?" The only ranking-style agents still return factual rankings from stored data or message counts; interpretation remains downstream.

### Hybrid Retrieval

Conversation and shared persistent-memory recall now use bounded hybrid
retrieval internally for exact and fuzzy evidence paths. Public tool signatures
and the public `rag_result.conversation_evidence` string projection stay
stable. Hybrid row metadata is internal `raw_result` / `projection_payload`
evidence for evaluator, continuation, logging, and tests.

Hybrid fusion ranks semantic+keyword rows first, keyword-only rows second,
neighbor/context rows third, and semantic-only rows last when they clear the
configured semantic-only floor. Search top-k, selected evidence limits, vector
candidate counts, neighbor windows, semantic-only floor, and evidence text caps
come from shared `RAG_SEARCH_*` / `RAG_HYBRID_*` / `RAG_VECTOR_*` config values
rather than per-agent literals.

Conversation projection includes bounded attachment descriptions and reply
excerpts, so rows whose body text is empty can still become usable evidence
when attachments carried the remembered content. Image summaries are rendered
as escaped `<image>...</image>` blocks in prompt-facing text; raw CQ syntax,
raw URLs, and binary fields are stripped before the RAG tool output reaches an
LLM.

Conversation evidence can also produce bounded relation packets for adjacent
message questions. The initializer may mark stable internal relation contracts
such as `relation=previous_message`, `relation=next_message`, or
`relation=reply_parent`. The search helper annotates neighboring rows with
their relation to the seed message, and `conversation_evidence_agent` reduces
the seed plus relation rows into packet summaries under
`projection_payload.packets`. Public `conversation_evidence` prefers those
packet summaries so cognition sees the answered relation, while raw row IDs
and full row details remain trace/debug material.

Relative-day conversation retrieval is grounded before tool execution when the
runtime `time_context` is available. For example, a local "yesterday" slot uses
the character-local previous date converted to UTC query bounds instead of
trusting the LLM to invent timestamp filters.

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
    "continuation": dict,  # optional on unresolved facts
}
```

`summary` is the worker-level retrieval summary. `raw_result` is retained
where structure matters, especially user and character profile bundles. The
projection layer applies a hybrid policy:

- structured profile/image bundles remain structured,
- scoped `user_memory_units` rows surfaced through `Memory-evidence:` remain
  in `memory_evidence` with scope metadata preserved and are also appended to
  `rag_result.user_memory_unit_candidates` for consolidation merge/evolve
  reuse,
- conversation evidence becomes formatted conclusion/context/uncertainty
  text, while raw message refs stay trace-only,
- shared and scoped memory evidence becomes formatted
  conclusion/context/uncertainty dictionaries,
- live/external evidence keeps URL-bearing text,
- recall evidence stays structured because downstream stages inspect fields
  such as `primary_source`, but prompt-facing recall summaries and evidence
  lines use the same formatted evidence style.

## Scope And Provenance

`Memory-evidence:` now spans two durable-memory sources with different authority boundaries:

- shared `memory` rows for official facts, common sense, seeded lore, and other non-user-scoped durable memory,
- scoped `user_memory_units` rows for current-user continuity, recognition,
  accepted preferences, user-specific lore, and prior shared interactions.

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

RAG 2 is called from the cognition resolver only when L2d selects an evidence
capability.

Cognition reads `rag_result`, not raw RAG supervisor state. The intended division is:

- RAG 2 answers "what evidence exists?"
- Cognition answers "what does this mean for Kazusa right now?"
- Dialog generation answers "how should Kazusa say it?"

RAG evidence preserves facts, uncertainty, and source context. Cognition owns
persona stance, emotional interpretation, and user-facing intent.

Formatted evidence is still evidence, not a persona decision. Cognition should
consume the conclusion, support, and uncertainty directly, then decide stance
and response goals from the wider cognition state.

## Integration With Consolidation

The consolidator also reads the projected `rag_result`.

RAG 2 retrieval produces evidence. Durable writes happen later in the
background consolidation path after cognition and dialog have completed. Cache
invalidation is tied to successful durable writes.

`recall_evidence` is operational provenance for agreements, promises, plans, and
current episode progress. A progress-only recall can guide the current reply.
Stable knowledge comes from the consolidator using approved durable evidence
sources.

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

Natural-language interpretation belongs in prompts, helper-agent contracts, and
projection policy. Deterministic code owns routing mechanics, budgets, cache
policy, and structural validation.

## Operational Notes

RAG 2 replaced the legacy RAG 1 / Cache 1 path. The current downstream payload
is `rag_result`.

Cache 2 exposes per-agent hit/miss statistics through service health data.
These counters are operational telemetry for operators and dashboards.

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
