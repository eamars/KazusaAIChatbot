# rag phase3 development plan

## Summary

- Goal: Reorganize RAG2 retrieval around semantic top-level capability agents so the initializer keeps multi-hop dependency planning but no longer chooses low-level search mechanics such as keyword vs semantic, profile vs relationship, or memory vs recent conversation target resolution.
- Plan class: large
- Status: approved
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`, `py-style`, `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: one approved Phase 3 execution in this fixed order: baseline tests, `LiveContextAgent`, `ConversationEvidenceAgent`, `MemoryEvidenceAgent`, `PersonContextAgent`, dispatcher/projection integration, initializer prompt/cache cutover, then full regression.
- Highest-risk areas: invalidating initializer strategy cache, increasing response latency through nested helper calls, breaking existing cascaded RAG2 paths, over-broad "memory manager" behavior, and accidentally moving RAG responsibilities into cognition or consolidation.
- Acceptance criteria: current complicated RAG2 chains still work through coarser semantic slots; live external fact questions route through target/scope resolution; conversation, memory, and person/profile searches are delegated to top-level capability agents; old worker agents remain available as reusable tooling; no cognition layer changes are required.

## Context

RAG2 currently asks the initializer to emit detailed physical retrieval slots such as:

```text
Conversation-keyword -> Identity -> Profile -> Conversation-keyword -> Web-search
```

That shape preserves dependency order, but it overloads the local planner with source mechanics. Recent real conversation analysis for QQ user `673225019` showed repeated pressure in five classes:

- active agreement and episode recall, now addressed by `Recall:`,
- live external facts such as weather, temperature, opening status, schedules, and prices,
- current episode/location state such as "where are you" and "what next",
- durable character/world facts such as the official address,
- cascaded historical evidence such as "who said phrase X, what link did they post, and what does the link contain?"

The real-conversation audit artifacts used to derive Phase 3 coverage are:

- `test_artifacts/rag2_reorg_chat_qq_673225019_last24h.json` — 122 messages from the scoped 24-hour conversation.
- `test_artifacts/rag2_reorg_user_image_qq_673225019.json` — scoped user image/profile summary.
- `test_artifacts/rag2_reorg_user_memories_qq_673225019.json` — scoped prompt-facing user memory categories.
- `test_artifacts/rag2_reorg_conversation_progress_qq_673225019.json` — scoped current episode progress.

Tests added by this plan must be based on compact cases derived from these artifacts and the user-provided logs. Do not commit the full raw transcript as a test fixture. Commit only the minimum user query, short context summary, expected route prefix, and reviewer note needed to reproduce the routing pressure.

The completed Recall plan already proved the preferred pattern:

```text
initializer emits one semantic slot
dispatcher maps the slot to one helper
helper owns source arbitration and low-level collectors
projection exposes bounded evidence to cognition
```

This phase generalizes that pattern without broadening Recall.

The completed `recall_agent_plan.md` intentionally left adjacent RAG improvements for later plans:

- real-time fact routing needs location/source resolution before memory or web lookup,
- conversation search needs topic, speaker, date-window, and anti-self-hit controls,
- memory source hygiene needs clearer source labels and authority boundaries.

This plan is that later RAG2 organization plan. It is a new approved plan; it does not amend completed plans.

## Mandatory Skills

- `development-plan-writing`: load before editing this plan, after context compaction, and before approving or changing checklist stages.
- `local-llm-architecture`: load before editing RAG2 prompts, dispatcher routing, helper-agent boundaries, cache policy, source arbitration, or LLM call contracts.
- `py-style`: load before editing Python production or test files.
- `test-style-and-execution`: load before writing or running deterministic, integration, prompt-render, or live LLM tests.
- `cjk-safety`: load before editing Python files that contain CJK examples or prompt text.

## Mandatory Rules

- Preserve the current RAG2 stage contract: initializer plans semantic slots and dependency order; dispatcher selects a helper by prefix; helper owns low-level source logic; deterministic code validates and executes; evaluator/finalizer summarize evidence.
- The initializer must not produce database parameters, Mongo filters, exact timestamps, memory-type filters, source-precedence rankings, or worker-agent names.
- The initializer must still support cascaded multi-hop plans through ordered slot dependencies such as "slot 1" and "the user resolved in slot 2".
- Do not implement generic primary/secondary route hedging.
- Do not create a generic "memory manager" or all-purpose retrieval super-agent.
- Do not broaden `RecallAgent` into memory search, profile lookup, relationship ranking, live external facts, or ordinary conversation search.
- Do not make top-level capability agents call each other recursively. They may reuse worker/tooling agents directly.
- Do not make any top-level capability run every possible source unconditionally. Each capability must have bounded source-selection rules and explicit refusal/unresolved behavior.
- Do not add deterministic keyword or regex classification over raw user text outside the LLM initializer or specialist LLM extractors. Deterministic code may validate structured capability outputs and source metadata.
- Deterministic parsing of the initializer's structured slot text is allowed inside a capability agent. Deterministic inference over `original_query` or other raw user text remains forbidden outside the LLM initializer or specialist LLM extractors.
- Do not edit cognition L2/L3 files in this phase. If a focused test proves the existing public `rag_result` contract cannot carry the evidence, stop and revise this plan before touching cognition.
- Do not change consolidation persistence schemas, conversation-progress recorder behavior, user-memory schema, conversation-history schema, relevance behavior, dialog behavior, or generic RAG2 retry semantics.
- Do not add a new graph stage, DB collection, scheduler behavior, environment variable, background task, or persistent helper-agent cache namespace in this phase.
- Do not reorder implementation stages. If the fixed order becomes impossible, stop and revise this plan before continuing.
- The `Capability Interfaces` section is the approved caller/callee contract. Do not change the `run(...)` surface, constructor surface, slot prefixes, context requirements, result keys, projection targets, worker/tooling tables, or refusal rules without revising this plan.
- Structured cross-slot handoff must use `known_facts[*].raw_result.resolved_refs`; free-text summaries are not an approved dependency channel for IDs, URLs, locations, or message provenance.
- Top-level capability agents are not cached in v1. They may call existing cacheable worker agents. Their own `cache` metadata must report `enabled=false` with a capability-orchestrator reason.
- Existing worker agents remain importable and testable. They must remain dispatcher-visible as compatibility aliases until a later cleanup plan explicitly removes those routes.
- Any initializer prompt change must bump `INITIALIZER_PROMPT_VERSION`. Any dispatcher-visible agent roster/prefix change must bump the agent registry version used in initializer cache keys.
- Runtime prompt-render checks are required for every changed prompt string that is formatted with `.format(...)`, f-strings, or JSON examples.
- Live LLM tests must be run one at a time and inspected one at a time.
- Preserve RAG2 logging observability: INFO must retain key operational outputs (`initializer unknown_slots`, dispatch `agent/task/route_source`, top-level capability `resolved/primary_worker/missing_context/selected_summary/cache reason`, evaluator fact summary, finalizer answer); DEBUG must hold supporting detail (`resolved_refs`, `projection_payload`, `worker_payloads`, raw payloads, context, cache keys). Do not move existing INFO breadcrumbs to DEBUG or log full raw transcripts/profile payloads at INFO.
- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.

## Must Do

- Add four dispatcher-visible top-level capability agents:
  - `live_context_agent`,
  - `conversation_evidence_agent`,
  - `memory_evidence_agent`,
  - `person_context_agent`.
- Keep `recall_agent` as the existing top-level active agreement / promise / episode-state capability.
- Keep `web_search_agent2` as the web worker and as a compatibility web route.
- Add new initializer slot prefixes:
  - `Live-context:`,
  - `Conversation-evidence:`,
  - `Memory-evidence:`,
  - `Person-context:`,
  - `Web-evidence:`.
- Keep these existing prefixes accepted by the dispatcher as compatibility aliases:
  - `Identity:`,
  - `User-list:`,
  - `Relationship:`,
  - `Profile:`,
  - `Conversation-aggregate:`,
  - `Conversation-filter:`,
  - `Conversation-keyword:`,
  - `Conversation-semantic:`,
  - `Memory-search:`,
  - `Web-search:`.
- Update the initializer prompt so new generated slots use the top-level capability prefixes, while examples still prove cascaded dependency order.
- Update dispatcher mapping so new prefixes route to top-level agents and old prefixes route to existing worker agents.
- Update projection so top-level agent results map into existing public `rag_result` fields:
  - `live_context_agent` and `web_search_agent2` -> `external_evidence`,
  - `conversation_evidence_agent` -> `conversation_evidence`,
  - `memory_evidence_agent` -> `memory_evidence`,
  - `person_context_agent` active/current-user profile payloads -> `user_image`,
  - `person_context_agent` active-character self-profile payloads -> `character_image`,
  - `person_context_agent` third-party profile, user-list, relationship, or `selected_summary`-only person evidence payloads -> `third_party_profiles`,
  - `recall_agent` -> `recall_evidence`.
- Add deterministic and live LLM tests mapping existing initializer and full RAG2 cases from old paths to new top-level paths.
- Add real-conversation-derived live LLM tests based on the QQ `673225019` Kazusa episode. These tests must cover live weather/opening questions, address recall/confirmation, active agreement recall, episode-position recall, and exact conversation evidence boundaries.
- Update RAG README documentation and this plan's execution evidence after implementation.

## Deferred

- Do not remove legacy worker-agent prefixes in this plan.
- Do not persist top-level capability-agent results in Cache2.
- Do not create a new vector index, Mongo collection, migration, backfill, or scheduler integration.
- Do not add parallel all-source fanout except where already approved inside `RecallAgent`.
- Do not change `RecallAgent` source order or collector set under this plan. If existing Recall tests reveal a compatibility issue, stop and revise this plan before changing Recall.
- Do not add image-aware direct RAG or binary attachment consumption.
- Do not modify cognition L2/L3 prompts.
- Do not modify consolidator persistence schemas.
- Do not implement admin UI, health endpoint changes, or cache-inspection APIs.
- Do not remove old tests; update them to assert the new path while preserving the same behavioral intent.

## Cutover Policy

Overall strategy: one approved Phase 3 execution with staged containment and a fixed implementation order. The implementation agent must not reorder the stages or integrate a later capability before the earlier stage's focused tests and checklist evidence are recorded. The stage gates are: baseline route tests first, standalone capability unit tests in the required order, dispatcher/projection integration tests, initializer prompt/cache cutover tests, live LLM route tests one by one, then full RAG2 live regression.

| Area | Policy | Instruction |
|---|---|---|
| New capability modules | compatible | Add modules under `src/kazusa_ai_chatbot/rag/` and test them directly before dispatcher registration. |
| Existing worker agents | compatible | Keep current public files/classes and dispatcher aliases. They become reusable tooling under top-level capabilities. |
| Dispatcher registry | bigbang at route cutover | Register new prefixes, add deterministic prefix dispatch before the dispatcher LLM fallback, and bump agent registry version in the same stage. |
| Initializer prompt/cache | bigbang at prompt cutover | Replace old generated-prefix examples with top-level capability examples and bump `INITIALIZER_PROMPT_VERSION`. Old cached strategies with legacy prefixes remain executable because aliases survive; the bump is defensive and makes new strategies preferred. |
| Projection payload | compatible | Reuse existing public `rag_result` fields; do not add new cognition-required fields. |
| Cache2 helper results | compatible | Top-level capability agents are uncached; existing worker caches keep their current policies. |
| Live web behavior | compatible | `web_search_agent2` remains the executor for public web content; `Live-context` only resolves live target/scope and delegates. |
| Tests/docs | migration | Existing tests are remapped to new prefixes and old compatibility aliases are tested separately. |

Rollback is code-only:

1. Restore the previous initializer prompt and bump `INITIALIZER_PROMPT_VERSION` again.
2. Remove new dispatcher prefix mappings.
3. Leave new modules unused or delete them if no longer referenced.
4. Keep legacy worker-agent prefixes, which remain compatible with the current system.
5. Rerun RAG initializer/cache tests, RAG projection tests, Recall tests, and one representative live RAG2 chain.

## Agent Autonomy Boundaries

Implementation agents may:

- create private helpers only inside the four new capability-agent modules, only for task normalization, listed-worker calls, standard result formatting, and test injection,
- add or modify tests only in the test files listed in `Change Surface` or new `tests/test_rag_phase3_*.py` files that cover the exact stages in this plan,
- update only `src/kazusa_ai_chatbot/rag/README.md` and this plan's `Execution Evidence` section for documentation,
- edit the Stage 7 initializer/dispatcher prompt text only to express the approved prefixes, old-to-new mapping, and listed route examples.

Implementation agents must not:

- reorder the fixed stage sequence: baseline tests, `LiveContextAgent`, `ConversationEvidenceAgent`, `MemoryEvidenceAgent`, `PersonContextAgent`, dispatcher/projection, initializer cutover, full regression,
- change the approved capability interface, output shape, projection mapping, source tables, worker/tooling paths, or refusal behavior,
- read context keys outside the capability's declared key list,
- add a fifth top-level capability not listed here,
- add a new public source, collection, persistence schema, graph stage, or environment variable,
- delete or rename existing worker-agent modules,
- edit existing worker-agent modules under this plan; if a worker compatibility bug blocks a stage, stop and revise this plan before changing that worker,
- make top-level agents call each other,
- integrate a partially tested capability into the dispatcher before its focused unit tests and route-mapping tests pass,
- make the initializer emit both old and new prefixes for the same evidence need,
- add fallback/self-repair loops to the RAG2 supervisor,
- move source-precedence rules back into the initializer,
- infer user meaning through deterministic raw-text keyword rules,
- treat source timestamp recency as the only authority model,
- edit RAG evaluator/finalizer prompts,
- touch cognition L2/L3 files without plan revision,
- claim live LLM behavior is proven by mocked tests.

If implementation reveals a missing architectural decision, stop and revise this plan before coding past it.

## Target State

The desired RAG2 search path is:

```text
RAG2 initializer
  -> emits ordered semantic capability slots
RAG2 dispatcher
  -> maps slot prefix to one top-level capability agent
top-level capability agent
  -> selects one bounded worker/tool path for its domain
worker/tooling agent
  -> generates low-level query parameters for one source/tool
deterministic executor
  -> validates, queries, caps, and returns evidence
RAG2 evaluator/finalizer/projection
  -> exposes compact evidence to cognition
```

The initializer remains responsible for cross-capability dependency order:

```text
Conversation-evidence -> Person-context -> Conversation-evidence -> Web-evidence
```

The initializer is no longer responsible for intra-capability mechanics:

```text
keyword vs semantic
filter vs aggregate
user profile vs relationship rank
exact memory vs semantic memory
character default location vs recent user-stated location
web search vs URL fetch
```

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Capability layer | Add semantic top-level agents above existing worker agents | Reduces initializer prompt burden while preserving multi-hop planning. |
| Recall | Keep as-is and narrow | Recall already owns active agreement / promise / episode-state arbitration. |
| Live facts | Add `Live-context:` and `live_context_agent` | Weather/current-status failures need target/scope resolution before web. |
| Conversation evidence | Add `Conversation-evidence:` and `conversation_evidence_agent` | The initializer should not choose keyword vs semantic vs filter vs aggregate. |
| Memory evidence | Add `Memory-evidence:` and `memory_evidence_agent` | Durable fact retrieval should own exact-vs-semantic memory decisions. |
| Person context | Add `Person-context:` and `person_context_agent` | Identity/profile/user-list/relationship routes are all person-centric and should be one semantic capability. |
| Web route | Add `Web-evidence:` as new generated prefix while keeping `Web-search:` alias | New initializer vocabulary is semantic; existing web helper stays reusable. |
| Dispatcher prefix routing | Make recognized prefixes deterministic before LLM fallback | Removes one planner LLM hop for normal slots and makes route contracts testable. |
| Cross-slot handoff | Add `resolved_refs` to top-level raw results | Cascaded slots need structured IDs, URLs, locations, and message provenance; summaries are not reliable handoff data. |
| Projection contract | Add `projection_payload` to top-level raw results | Projection reads one normalized shape instead of guessing from worker-specific payloads. |
| Caching | Do not cache top-level capability results v1 | Prevents stale orchestration results; worker caches remain source-aware. |
| Projection | Reuse existing `rag_result` fields | Avoids cognition prompt changes and limits blast radius. |
| Legacy prefixes | Keep dispatcher aliases | Allows rollback and preserves old cached/manual slot compatibility until cleanup. |

## Capability Interfaces

This section is the reviewed and approved interface between the RAG2 supervisor and the four new top-level capability agents. The implementation agent must not invent extra public arguments, return fields required by downstream consumers, graph state keys, dispatcher JSON fields, or projection destinations.

All new top-level capability agents use the standard helper surface:

```python
class <Capability>Agent(BaseRAGHelperAgent):
    def __init__(
        self,
        *,
        cache_runtime: RAGCache2Runtime | None = None,
    ) -> None:
        ...

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        raise NotImplementedError
```

Constructor contract:

- Each class constructor accepts only the optional `cache_runtime` keyword used by tests and existing Cache2 patterns.
- Each class calls `BaseRAGHelperAgent.__init__` with its fixed `name` and `cache_name=""`.
- No public constructor options, feature flags, environment variables, or persistent cache namespace are approved.

Caller contract:

- The dispatcher calls the capability through the existing `BaseRAGHelperAgent.run(task, context, max_attempts)` surface.
- `task` is a plain string derived from the initializer slot and dispatcher task text. The agent must not require a structured task object.
- `context` is the existing delegate context from `_build_delegate_context`: runtime context plus `known_facts`, `original_query`, and `current_slot`.
- All capabilities may read only `known_facts`, `original_query`, and `current_slot` plus the capability-specific context keys declared below.
- The implementation must not require new runtime context keys. If a declared key required for a source lookup is missing, return `resolved=false` with `missing_context` rather than changing the graph state.
- `known_facts[*].raw_result.resolved_refs` is the only approved structured dependency channel between ordered slots. Summaries may be displayed, but they must not be parsed for IDs, URLs, locations, or message provenance.

Execution contract:

- A top-level capability runs once and returns `attempts=1`; it must not implement its own retry loop.
- A top-level capability selects one primary worker/tool path for the request.
- Only two multi-worker chains are approved: `LiveContextAgent` target/scope resolution -> `web_search_agent2`, and `PersonContextAgent` `user_lookup_agent` -> `user_profile_agent`.
- Any other request that needs more than one worker/helper execution must return unresolved and explain the missing handoff in `missing_context`.
- Top-level capabilities must not call each other.
- If a capability returns `resolved=false`, the existing RAG evaluator records that fact, drains the slot, and the finalizer surfaces the missing context. Do not add dispatcher retries, repair loops, alternate-route fallback, or graph-state changes.

Top-level capability result shape:

```json
{
  "resolved": true,
  "result": {
    "selected_summary": "short factual answer or evidence summary",
    "capability": "live_context | conversation_evidence | memory_evidence | person_context",
    "primary_worker": "worker agent name or deterministic collector name",
    "supporting_workers": ["optional worker names"],
    "source_policy": "short authority/freshness explanation",
    "resolved_refs": [],
    "projection_payload": {},
    "worker_payloads": {},
    "evidence": [],
    "missing_context": [],
    "conflicts": []
  },
  "attempts": 1,
  "cache": {
    "enabled": false,
    "hit": false,
    "reason": "capability_orchestrator_uncached"
  }
}
```

`resolved=false` uses the same shape with empty `selected_summary` and a non-empty `missing_context` or source-policy explanation.

Structured handoff contract:

- `resolved_refs` is a list of normalized references for later slots.
- Allowed `ref_type` values are `person`, `message`, `url`, `location`, and `memory`.
- Person refs use `{"ref_type": "person", "role": "speaker|profile_owner|current_user|active_character", "global_user_id": "...", "display_name": "..."}`.
- URL refs use `{"ref_type": "url", "role": "posted_url|source_url", "url": "..."}`.
- Location refs use `{"ref_type": "location", "role": "target_location|character_default|user_recent", "text": "..."}`.
- Message refs use `{"ref_type": "message", "platform_message_id": "...", "timestamp": "...", "global_user_id": "...", "display_name": "..."}`.
- `worker_payloads` is trace/debug material only. Downstream slots and projection must use `resolved_refs` and `projection_payload`.

Projection payload contract:

- `conversation_evidence_agent`: `projection_payload.summaries` is a list of prompt-facing evidence strings.
- `memory_evidence_agent`: `projection_payload.memory_rows` is a list of raw memory rows with `content` preserved.
- `live_context_agent`: `projection_payload.external_text` and `projection_payload.url` carry web/live evidence; `web_search_agent2` keeps the existing free-text projection branch.
- `person_context_agent`: `projection_payload.profile_kind` is `current_user`, `active_character`, `third_party`, `user_list`, or `relationship`; `owner_global_user_id` is set when one profile owner is known; `profile` carries the raw `user_profile_agent` payload including `_user_memory_units` and `self_image` when present; `summary` carries list/ranking/person-summary evidence.
- Projection switches on top-level `agent` and `projection_payload`; it must not parse `selected_summary` to reconstruct IDs or worker internals.

### LiveContextAgent

Semantic ownership:

- live external facts that change with real time: weather, temperature, public opening status, schedules, prices, exchange rates, current event status.
- Declared context keys: `platform`, `platform_channel_id`, `platform_user_id`, `global_user_id`, `user_name`, `chat_history_recent`, `chat_history_wide`, `character_profile`.

Input slot examples:

```text
Live-context: answer current temperature for explicit location Auckland
Live-context: answer current temperature for the active character's location
Live-context: answer current weather for the current user's location if recently stated
Live-context: answer current opening status for Christchurch Adventure Park this weekend
```

Worker/tooling paths:

| Need | Worker/tool |
|---|---|
| explicit target in task | deterministic structured extraction from the `Live-context:` task |
| active character stable default location | `persistent_memory_keyword_agent` or `persistent_memory_search_agent` as a worker |
| current user's recently stated location | `conversation_search_agent` or `conversation_filter_agent` as a worker with same-user scope |
| public live fact | `web_search_agent2` |
| approved two-worker chain | target/scope worker -> `web_search_agent2` |

Refusal/unresolved behavior:

- If no trusted target/scope can be resolved, return `resolved=false` with `missing_context=["location" or "target"]`.
- Do not silently substitute character location for user-local questions.
- Do not search persistent memory for the live value itself; memory may only resolve stable target/scope.
- Calls to persistent memory workers must be labelled in `source_policy` as `target_scope_lookup`; all other durable memory retrieval belongs to `MemoryEvidenceAgent`.

### ConversationEvidenceAgent

Semantic ownership:

- evidence from chat history, including exact phrases, URLs, filenames, speaker provenance, recent messages, fuzzy topics, counts, rankings, and grouped message statistics.
- Declared context keys: `platform`, `platform_channel_id`, `platform_user_id`, `global_user_id`, `user_name`, `chat_history_recent`, `chat_history_wide`.

Input slot examples:

```text
Conversation-evidence: find who said "版权保护一直都是play的一环"
Conversation-evidence: find URL posted by the user resolved in slot 2
Conversation-evidence: retrieve recent messages from the user resolved in slot 1
Conversation-evidence: count recent messages mentioning cookie管理器 by user
```

Worker/tooling paths:

| Need | Worker/tool |
|---|---|
| exact phrase, URL, filename, literal term | `conversation_keyword_agent` |
| fuzzy topic or topic-to-speaker | `conversation_search_agent` |
| known user/time/count retrieval | `conversation_filter_agent` |
| counts/rankings/grouped stats | `conversation_aggregate_agent` |

Refusal/unresolved behavior:

- If the task asks for active agreement/current episode state, return incompatible intent; initializer should have used `Recall:`.
- If the task asks for durable world facts, return incompatible intent; initializer should have used `Memory-evidence:`.
- Prevent current-question self-hit where the task asks for prior evidence or origin evidence.
- Speaker, message, and URL findings that later slots may need must be emitted in `resolved_refs`.

### MemoryEvidenceAgent

Semantic ownership:

- durable shared/world/common-sense/character facts and curated memory evidence.
- Declared context keys: `platform`, `platform_channel_id`, `global_user_id`, `user_name`, `character_profile`.

Input slot examples:

```text
Memory-evidence: retrieve durable evidence about the active character's official address
Memory-evidence: retrieve common-sense evidence relevant to choosing walk vs drive for a 50 meter trip
Memory-evidence: retrieve curated knowledge about vibe coding
```

Worker/tooling paths:

| Need | Worker/tool |
|---|---|
| exact named fact/tag/proper noun | `persistent_memory_keyword_agent` |
| fuzzy concept/common-sense/character-world fact | `persistent_memory_search_agent` |

Refusal/unresolved behavior:

- Do not handle active agreements/promises/current episode state; use `Recall:`.
- Do not handle person relationship/profile impressions; use `Person-context:`.
- Do not handle live external facts; use `Live-context:`.
- Memory workers called by this agent are for durable evidence only, not live target/scope lookup.

### PersonContextAgent

Semantic ownership:

- person identity, profile, user image, character self-profile, user-list predicates, relationship ranking, and person-specific impression/compatibility context.
- Declared context keys: `platform`, `platform_user_id`, `global_user_id`, `user_name`, `character_profile`.

Input slot examples:

```text
Person-context: retrieve profile/impression for display name 小钳子
Person-context: retrieve active character self-profile
Person-context: list users whose display names end with 子
Person-context: rank users by active character relationship from top limit 1
Person-context: resolve/read profile for the speaker found in slot 1
```

Worker/tooling paths:

| Need | Worker/tool |
|---|---|
| display-name identity | `user_lookup_agent` |
| full user or character profile | `user_profile_agent` |
| display-name -> profile chain | `user_lookup_agent` -> `user_profile_agent` |
| display-name predicate enumeration | `user_list_agent` |
| relationship ranking | `relationship_agent` |

Refusal/unresolved behavior:

- If a task first needs to find an unknown speaker by message content, return incompatible intent; initializer should put `Conversation-evidence:` before `Person-context:`.
- Do not search persistent memory directly for person impressions unless the existing profile/user-memory worker returns that data through the profile contract.
- Identity/profile results that later slots may need must be emitted in `resolved_refs`.

## Old-To-New Route Mapping

| Existing path | New top-level path |
|---|---|
| `Identity -> Profile` | `Person-context` |
| `User-list` | `Person-context` |
| `Relationship` | `Person-context` |
| `Conversation-keyword` | `Conversation-evidence` |
| `Conversation-semantic` | `Conversation-evidence` |
| `Conversation-filter` | `Conversation-evidence` |
| `Conversation-aggregate` | `Conversation-evidence` |
| `Memory-search` | `Memory-evidence` |
| `Memory-search stable location -> Web-search live value` | `Live-context` |
| `Conversation-semantic recent user location -> Web-search live value` | `Live-context` |
| `Web-search explicit URL/topic` | `Web-evidence` |
| `Recall` | `Recall` |

Important cascaded mappings:

| Query class | Old chain | New chain |
|---|---|---|
| "who said phrase, what link did they post, what is in the link" | `Conversation-keyword -> Identity -> Conversation-keyword -> Web-search` | `Conversation-evidence -> Person-context -> Conversation-evidence -> Web-evidence` |
| "speaker of 5090/qwen27b, impression, recent interaction" | `Conversation-keyword -> Identity -> Profile -> Conversation-filter` | `Conversation-evidence -> Person-context -> Conversation-evidence` |
| "named person yesterday said AI phrase" | `Identity -> Profile -> Conversation-keyword` | `Person-context -> Conversation-evidence` |
| "character-local current temperature" | `Memory-search -> Web-search` | `Live-context` |
| "today's agreement" | `Recall` | `Recall` |

## Change Surface

### Create

| Path | Purpose |
|---|---|
| `src/kazusa_ai_chatbot/rag/live_context_agent.py` | Top-level live external fact target/scope resolver and web delegator. |
| `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py` | Top-level chat-history evidence capability over existing conversation workers. |
| `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py` | Top-level durable memory evidence capability over existing memory workers. |
| `src/kazusa_ai_chatbot/rag/person_context_agent.py` | Top-level person/profile/relationship capability over existing person workers. |
| `tests/test_rag_phase3_capability_agents.py` | Deterministic capability-agent unit tests with patched workers. |
| `tests/test_rag_phase3_initializer_live_llm.py` | One-at-a-time live initializer routing tests for new prefixes. |
| `tests/test_rag_phase3_real_conversation_live_llm.py` | Real-LLM route tests derived from the QQ `673225019` Kazusa conversation and user image/progress artifacts. |
| `tests/fixtures/rag_phase3_real_conversation_cases.json` | Compact sanitized case fixture derived from the scoped real conversation; no full transcript dump. |

### Modify

| Path | Change |
|---|---|
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` | Register new prefixes/agents, add deterministic prefix dispatch before dispatcher LLM fallback, update initializer and dispatcher prompts, keep old prefix aliases. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py` | Map top-level `projection_payload` fields into existing public `rag_result` fields; keep old worker-agent projection branches for compatibility aliases. |
| `src/kazusa_ai_chatbot/rag/cache2_policy.py` | Bump initializer prompt and agent registry versions; add explicit non-cache policy labels for all four top-level capability agents. |
| `src/kazusa_ai_chatbot/rag/README.md` | Document top-level capability layer and worker/tooling reuse. |
| `tests/test_rag_initializer_cache2.py` | Update prompt-contract assertions and version checks. |
| `tests/test_persona_supervisor2_rag_supervisor2_live.py` | Update expected/manual path descriptions and add one new cascaded route trace for phrase -> person -> link -> web. |
| `tests/test_rag_projection.py` | Add projection fixtures for new top-level agents, `resolved_refs` handoff payloads, and old worker-agent compatibility aliases. |

### Keep Unchanged

- Cognition L2/L3 files.
- Consolidator persistence schemas.
- Conversation progress recorder.
- Existing worker-agent modules. If a test proves a narrow worker compatibility bug, stop and revise this plan before changing that worker.
- Existing DB schemas and indexes.

## LLM Call And Context Budget

| Stage | Current | Target | Notes |
|---|---:|---:|---|
| RAG initializer | 1 call, long prompt | 1 call, shorter prompt | Prompt should contain only top-level capability rules and high-value cascaded examples. |
| RAG dispatcher | 1 LLM call per slot today | 0 LLM calls for recognized prefixes | Deterministic prefix mapping is required for new and legacy prefixes; dispatcher LLM remains only as fallback for unrecognized slots. |
| Top-level capability | new | usually 1 selector LLM call | Deterministic shortcut is expected only when the slot contains a structured marker already sufficient for a worker call. |
| Worker agents | existing loops | existing loops | Reused workers keep their own attempt caps. |
| LiveContext | new | usually 1 selector + target resolver when required + web | Must not run memory + conversation + web blindly. |
| Finalizer/evaluator | unchanged | unchanged | Do not redesign summarizers in this phase. |

The expected common path is:

```text
initializer LLM
deterministic dispatcher prefix match
one top-level capability selector LLM
one worker/helper execution, or one approved resolver worker plus final worker
evaluator/finalizer
```

Only `LiveContextAgent` target/scope -> web and `PersonContextAgent` lookup -> profile are approved two-worker paths. Other multi-worker paths return unresolved and require plan revision.

## Implementation Order

### Stage 1: Baseline Mapping And Failing Tests

1. Add deterministic route-mapping fixtures that encode the old-to-new table in this plan.
2. Add live initializer tests for:
   - active agreement remains `Recall:`,
   - exact phrase routes to `Conversation-evidence:` and not `Recall:`,
   - character-local current temperature routes to `Live-context:`,
   - user-local current temperature routes to `Live-context:` and does not silently use character location,
   - cascaded phrase -> person -> link -> web route uses the four new top-level prefixes.
3. Add compact real-conversation cases derived from the QQ `673225019` Kazusa episode:
   - `christchurch_weekend_weather`: "对了周末基督城天气怎么样？" -> `Live-context:`.
   - `amusement_park_opening`: "先说好啊可不是怂哦，就只是想看看到时候游乐场开不开门哼" -> `Live-context:`.
   - `recent_address_confirmation`: "你刚刚才把地址发给我，123 Example Street没错吧" -> `Conversation-evidence:`.
   - `official_address_memory`: the persistent-memory address query from the provided log -> `Memory-evidence:`.
   - `today_agreement`: "早上好呀，还记得今天的约定么" -> `Recall:`.
   - `episode_position_next_step`: "那我们接下来去哪儿" -> `Recall:` with `episode_position` intent.
   - `exact_phrase_boundary`: "谁说过'约定就是约定'？" -> `Conversation-evidence:` and not `Recall:`.
4. Run the live tests one at a time before prompt edits and record expected failures.

### Stage 2: LiveContextAgent Standalone Module

1. Implement `LiveContextAgent` with patched-worker deterministic tests.
2. Cover live external fact classes: weather, temperature, opening status, schedules, and prices.
3. Verify location/source resolution refuses unresolved user-local live facts instead of silently falling back to character memory.
4. Verify memory-worker calls are labelled `target_scope_lookup` and never retrieve the live value itself.
5. Verify `resolved_refs` contains the target location and `projection_payload` contains external evidence fields.
6. Verify the agent returns the standard helper shape and `capability_orchestrator_uncached` cache metadata.

### Stage 3: ConversationEvidenceAgent Standalone Module

1. Implement `ConversationEvidenceAgent` with patched-worker deterministic tests.
2. Cover exact phrase, semantic topic, structured filter, aggregate/count, speaker/date-window, anti-self-hit, and cascaded phrase -> person -> link handoff cases.
3. Verify the agent selects existing conversation worker/tooling agents internally without exposing keyword/semantic/filter/aggregate mechanics to the initializer.
4. Verify speaker, message, and URL outputs are emitted in `resolved_refs` for later slots.
5. Verify `projection_payload.summaries` carries prompt-facing conversation evidence.
6. Verify the agent returns the standard helper shape and `capability_orchestrator_uncached` cache metadata.

### Stage 4: MemoryEvidenceAgent Standalone Module

1. Implement `MemoryEvidenceAgent` with patched-worker deterministic tests.
2. Cover durable character/world facts, user memory units, shared memory/common-sense entries, and source-authority boundaries.
3. Verify the agent selects existing memory worker/tooling agents internally without exposing memory-type or physical source filters to the initializer.
4. Verify `projection_payload.memory_rows` preserves raw memory `content` rows for projection.
5. Verify live target/scope lookup remains outside this agent.
6. Verify the agent returns the standard helper shape and `capability_orchestrator_uncached` cache metadata.

### Stage 5: PersonContextAgent Standalone Module

1. Implement `PersonContextAgent` with patched-worker deterministic tests.
2. Cover identity lookup, user list, profile/image, relationship, third-party profile, and cascaded person-reference cases.
3. Cover the approved `user_lookup_agent` -> `user_profile_agent` chain for display-name profile requests.
4. Verify identity/profile outputs are emitted in `resolved_refs` for later slots.
5. Verify `projection_payload` covers current user profile, active character profile, third-party profile, user-list summary, and relationship summary.
6. Verify the agent selects existing person/profile/relationship worker tooling internally without exposing profile-vs-relationship mechanics to the initializer.
7. Verify the agent returns the standard helper shape and `capability_orchestrator_uncached` cache metadata.

### Stage 6: Dispatcher And Projection Integration

1. Register new prefix mappings in the dispatcher.
2. Add deterministic prefix-to-agent dispatch for every new prefix and every legacy alias before the dispatcher LLM fallback.
3. Keep old prefix mappings intact.
4. Add new agent names to the allowed dispatcher union.
5. Update projection to consume top-level `projection_payload` fields.
6. Keep old worker-agent projection branches intact for compatibility aliases.
7. Add deterministic tests proving recognized prefixes do not call the dispatcher LLM.
8. Add projection tests for `resolved_refs`, `projection_payload`, unresolved `missing_context`, and legacy aliases.
9. Add deterministic logging-level tests proving key route/capability outputs stay at INFO and verbose payloads stay at DEBUG.
10. Bump agent registry version.
11. Run dispatcher/projection deterministic tests.

### Stage 7: Initializer Prompt Cutover

1. Rewrite initializer prompt around top-level capabilities.
2. Keep examples for every cascaded class in the old-to-new mapping.
3. Remove detailed prompt obligations that make the initializer choose keyword vs semantic/filter/aggregate or profile vs relationship internals.
4. Bump `INITIALIZER_PROMPT_VERSION`.
5. Confirm old cached strategies with legacy prefixes still dispatch through compatibility aliases.
6. Run runtime prompt-render checks.
7. Run deterministic initializer prompt/cache tests.
8. Run live initializer tests one by one and inspect traces.

### Stage 8: Full RAG2 Regression And Documentation

1. Run focused deterministic RAG2 tests.
2. Run representative full live RAG2 cases one by one:
   - person impression,
   - named person recent topic,
   - exact phrase speaker,
   - cascaded speaker -> profile -> recent interaction,
   - XHS/URL link content,
   - Kazusa official address.
3. Run the real-conversation-derived live initializer tests one by one and inspect the trace for each case listed in Stage 1.
4. Add deterministic integration coverage for live weather/opening status with the web worker patched so the test does not depend on public network availability.
5. Update `rag/README.md`.
6. Record execution evidence in this plan.
7. Status is already `approved`; mark `in_progress` when implementation starts and `completed` only after evidence is recorded.

## Progress Checklist

- [ ] Stage 1: Baseline route mapping and expected-failure live tests recorded.
- [ ] Stage 1b: Real-conversation-derived QQ `673225019` live LLM route cases recorded as baseline expected-failure or route-pressure evidence.
- [ ] Stage 2: `LiveContextAgent` implemented standalone and deterministic tests pass.
- [ ] Stage 3: `ConversationEvidenceAgent` implemented standalone and deterministic tests pass.
- [ ] Stage 4: `MemoryEvidenceAgent` implemented standalone and deterministic tests pass.
- [ ] Stage 5: `PersonContextAgent` implemented standalone and deterministic tests pass.
- [ ] Stage 6: Deterministic prefix dispatch and projection integration complete, old aliases preserved, registry version bumped.
- [ ] Stage 7: Initializer prompt cutover complete, prompt version bumped, live initializer tests inspected.
- [ ] Stage 8: Full RAG2 regression, docs, and execution evidence complete.

## Verification

### Static Greps

- `rg "Conversation-evidence|Memory-evidence|Live-context|Person-context|Web-evidence" src tests development_plans`
- `rg "resolved_refs|projection_payload|capability_orchestrator_uncached" src tests`
- `rg "deterministic prefix|prefix-to-agent|dispatcher LLM" src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py tests`
- `rg "Conversation-keyword|Conversation-semantic|Conversation-filter|Conversation-aggregate" src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
  - Expected: old prefixes remain in dispatcher compatibility table, not as preferred generated initializer examples after cutover.
- `rg "src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py|src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py" <changed-file-list>`
  - Expected: no cognition L2/L3 changes.

### Compile

```powershell
venv\Scripts\python.exe -m py_compile `
  src\kazusa_ai_chatbot\rag\live_context_agent.py `
  src\kazusa_ai_chatbot\rag\conversation_evidence_agent.py `
  src\kazusa_ai_chatbot\rag\memory_evidence_agent.py `
  src\kazusa_ai_chatbot\rag\person_context_agent.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_projection.py
```

### Deterministic Tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py -q
venv\Scripts\python.exe -m pytest tests\test_rag_projection.py tests\test_rag_initializer_cache2.py -q
venv\Scripts\python.exe -m pytest tests\test_rag_recall_agent.py -q
```

### Live LLM Tests

Run one at a time and inspect trace artifacts:

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_recall_live_llm.py::test_live_initializer_routes_active_agreement_to_recall -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_exact_phrase_to_conversation_evidence -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_character_local_temperature_to_live_context -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_user_local_temperature_to_live_context_without_character_fallback -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_initializer_live_llm.py::test_live_initializer_preserves_cascaded_phrase_person_link_web_chain -q -s -m live_llm
```

Each trace must include input query, raw initializer output, parsed slots, expected top-level prefixes, and reviewer judgment.

### Real Conversation Live LLM Tests

Run one at a time and inspect trace artifacts. These tests are mandatory because they are based on the actual Kazusa conversation that motivated this plan:

```powershell
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_real_conversation_live_llm.py::test_real_conversation_christchurch_weather_routes_to_live_context -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_real_conversation_live_llm.py::test_real_conversation_amusement_park_opening_routes_to_live_context -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_real_conversation_live_llm.py::test_real_conversation_recent_address_confirmation_routes_to_conversation_evidence -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_real_conversation_live_llm.py::test_real_conversation_official_address_routes_to_memory_evidence -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_real_conversation_live_llm.py::test_real_conversation_today_agreement_routes_to_recall -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_real_conversation_live_llm.py::test_real_conversation_episode_next_step_routes_to_recall -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_rag_phase3_real_conversation_live_llm.py::test_real_conversation_exact_phrase_boundary_routes_to_conversation_evidence -q -s -m live_llm
```

Each trace must include the compact fixture case id, the user query, the route expected from this plan, raw initializer output, parsed slots, and a reviewer judgment. The fixture must include only compact snippets and must not include the full 122-message transcript.

### Full RAG2 Live Regression

Run representative existing live RAG2 tests one by one. Passing pytest structure is not enough; inspect traces for route quality:

- person impression routes through `Person-context:`,
- recent person topic routes through `Person-context:` then `Conversation-evidence:`,
- exact phrase speaker routes through `Conversation-evidence:`,
- cascaded phrase/profile/interaction route preserves dependency order,
- link content route ends with `Web-evidence:`,
- address query routes through `Memory-evidence:`,
- active agreement still routes through `Recall:`.

## Acceptance Criteria

This plan is complete when:

- The initializer prompt emits top-level capability prefixes for new strategies.
- Existing cascaded RAG2 cases still preserve dependency order.
- Cascaded slots use `known_facts[*].raw_result.resolved_refs` for structured IDs, URLs, locations, and message provenance; no downstream slot parses free-text summaries for handoff.
- The dispatcher resolves recognized new and legacy prefixes deterministically without calling the dispatcher LLM.
- The initializer no longer needs prompt rules for choosing keyword vs semantic/filter/aggregate, profile vs relationship internals, or memory-vs-conversation target resolution for live facts.
- `Live-context:` resolves explicit live targets, character-local target dependencies, and user-local target dependencies without falling back to memory as the live value source.
- `Conversation-evidence:` correctly delegates exact, semantic, filter, and aggregate cases to existing conversation workers.
- `Memory-evidence:` correctly delegates exact and semantic durable memory cases to existing memory workers.
- `Person-context:` correctly delegates identity, profile, user-list, relationship, and the approved display-name -> profile chain to existing person workers.
- `Recall:` remains narrow and unchanged in responsibility.
- Old worker prefixes still dispatch for compatibility.
- Projection reads top-level `projection_payload` fields, preserves old worker projection branches for aliases, uses existing public `rag_result` fields, and leaves cognition L2/L3 files unchanged.
- Unresolved top-level capability results with `missing_context` are recorded by the existing evaluator/finalizer path without new retries or alternate fallback.
- RAG2 logs preserve user-debuggable key outputs at INFO while keeping refs, payloads, raw context, and cache internals at DEBUG.
- Deterministic tests pass.
- Real LLM initializer and full RAG2 traces are inspected and judged acceptable.
- Real-conversation-derived live LLM traces from the QQ `673225019` Kazusa episode are inspected and judged acceptable.
- Execution evidence is recorded in this plan.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Initializer loses complex chain ability | Keep dependency-order examples and live cascaded tests | Cascaded phrase -> person -> link -> web live test |
| Latency increases | Dispatcher prefix routing is deterministic; top-level agents normally use one selector LLM and a capped worker path; no unconditional fanout | Agent tests assert worker call counts and dispatcher tests assert no LLM call for recognized prefixes |
| Cascaded ID handoff breaks | Top-level results expose `resolved_refs` and downstream slots read that field from `known_facts` | Deterministic handoff tests for speaker -> profile and speaker -> URL chains |
| Projection loses profile internals | Top-level results expose `projection_payload.profile` with `_user_memory_units`, `self_image`, and owner ID preserved | Projection tests for current user, active character, and third-party profile payloads |
| Top-level agents become a mega-agent | Four fixed semantic capabilities; no top-level recursion | Static review and autonomy boundaries |
| Live facts still route to memory | `Live-context` owns target/scope and memory is only for stable target dependencies | Live temperature tests and deterministic missing-location tests |
| Cache invalidation confusion | Top-level agents uncached; workers keep existing Cache2 policies | Cache metadata tests |
| Old tests become misleading | Old-to-new route table is encoded in tests | Deterministic mapping tests |
| Cognition behavior regresses | Projection reuses existing public fields; no L2/L3 edits | Projection tests and no-changed-file grep |

## Execution Evidence

Not started. Fill this section during implementation with:

- plan reread log,
- changed files,
- prompt version and agent registry version changes,
- prompt-render command output,
- static grep results,
- deterministic test results,
- live LLM, real-conversation, and full RAG2 regression trace paths and judgments,
- known residual risks.

## Glossary

- **Top-level capability agent:** dispatcher-visible RAG helper that owns one semantic evidence class, such as live context, conversation evidence, memory evidence, person context, or recall.
- **Worker/tooling agent:** existing narrow helper such as `conversation_keyword_agent` or `user_profile_agent` that generates low-level parameters for one retrieval source/tool.
- **Capability slot:** initializer output prefix that names the evidence class needed, not the physical query method.
- **Compatibility alias:** old initializer/dispatcher prefix that remains executable for rollback, manual use, or legacy cached strategy safety.
