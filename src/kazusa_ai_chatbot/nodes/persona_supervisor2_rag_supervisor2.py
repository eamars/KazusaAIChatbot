"""Progressive RAG supervisor (experimental).

RAG responsibility boundary:
  - Fetch evidence from conversation history, user/profile stores, memories,
    and external sources.
  - Resolve identities, enumerate entities, filter records, deduplicate rows,
    preserve source metadata, and perform simple factual reductions such as
    counts, rankings, and short extractive summaries.
  - Return compact factual context for downstream cognition.

RAG must NOT:
  - Answer on behalf of the character.
  - Provide opinions, relationship judgments, motives, emotional stance, or
    persona-specific interpretation.
  - Decide final user-facing tone or strategy.

In short: RAG transforms data into evidence; cognition transforms evidence
into meaning.

Resolves complex multi-hop queries such as
"他上次说的那个链接里有什么信息么" through:

  Initializer -> Dispatcher -> Executor -> Evaluator -> Dispatcher -> ...

The Initializer decomposes the query into ordered ``unknown_slots`` (e.g.
["人物指代: 解析'他'", "对象指代: 找 URL", ...]). Each loop iteration
targets one slot, runs one tool call, then drains the slot from the list.
``unknown_slots`` emptying IS the stop condition — no ``should_stop`` flag.

Message flow:
  ``messages`` — consumed only by the Dispatcher for short dispatch history.
  ``known_facts`` — consumed only by the Evaluator and Finalizer.
"""

import asyncio
import datetime
import json
import logging
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from kazusa_ai_chatbot.config import (
    RAG_PLANNER_LLM_API_KEY,
    RAG_PLANNER_LLM_BASE_URL,
    RAG_PLANNER_LLM_MODEL,
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.db.rag_cache2_persistent import (
    record_initializer_hit,
    upsert_initializer_entry,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.rag.cache2_policy import (
    INITIALIZER_AGENT_REGISTRY_VERSION,
    INITIALIZER_CACHE_NAME,
    INITIALIZER_PROMPT_VERSION,
    INITIALIZER_STRATEGY_SCHEMA_VERSION,
    build_initializer_cache_key,
)
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.rag.conversation_aggregate_agent import ConversationAggregateAgent
from kazusa_ai_chatbot.rag.conversation_evidence_agent import ConversationEvidenceAgent
from kazusa_ai_chatbot.rag.conversation_filter_agent import ConversationFilterAgent
from kazusa_ai_chatbot.rag.conversation_keyword_agent import ConversationKeywordAgent
from kazusa_ai_chatbot.rag.conversation_search_agent import ConversationSearchAgent
from kazusa_ai_chatbot.rag.live_context_agent import LiveContextAgent
from kazusa_ai_chatbot.rag.memory_evidence_agent import MemoryEvidenceAgent
from kazusa_ai_chatbot.rag.person_context_agent import PersonContextAgent
from kazusa_ai_chatbot.rag.persistent_memory_keyword_agent import PersistentMemoryKeywordAgent
from kazusa_ai_chatbot.rag.persistent_memory_search_agent import PersistentMemorySearchAgent
from kazusa_ai_chatbot.rag.prompt_projection import (
    project_runtime_context_for_llm,
    project_tool_result_for_llm,
)
from kazusa_ai_chatbot.rag.recall_agent import RecallAgent
from kazusa_ai_chatbot.rag.relationship_agent import RelationshipAgent
from kazusa_ai_chatbot.rag.user_list_agent import UserListAgent
from kazusa_ai_chatbot.rag.user_lookup_agent import UserLookupAgent
from kazusa_ai_chatbot.rag.user_profile_agent import UserProfileAgent
from kazusa_ai_chatbot.rag.web_search_agent import WebSearchAgent
from kazusa_ai_chatbot.time_context import format_timestamp_for_llm
from kazusa_ai_chatbot.utils import get_llm, log_list_preview, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)

_MAX_LOOP_COUNT = 8



class ProgressiveRAGState(TypedDict):
    """Working state for the progressive RAG supervisor.

    Fields:
        original_query: The user's natural-language question.
        context: Optional auxiliary fields (platform, channel, ...).
        unknown_slots: Ordered slots still needing resolution; drains to empty.
        current_slot: The slot being targeted in the current iteration.
        known_facts: Slot results; each entry has slot, agent, resolved, summary, raw_result, attempts.
        messages: LangGraph message log — tool-call protocol only.
        initializer_cache: Cache metadata for the initializer strategy lookup.
        loop_count: Safety cap counter.
        final_answer: Synthesised answer from the finalizer.
    """

    original_query: str
    character_name: str
    context: dict
    unknown_slots: list[str]
    current_slot: str
    known_facts: list[dict]
    messages: Annotated[list, add_messages]
    initializer_cache: dict
    current_dispatch: dict
    last_agent_result: dict
    loop_count: int
    final_answer: str


RAGAgentCallable = Callable[..., Awaitable[dict]]


class RAGFactSource(TypedDict):
    """Deterministic source policy for facts returned by one RAG agent."""

    source_kind: str
    source_system: str
    consolidation_policy: str
    can_consolidate_as_new_knowledge: bool


class RAGAgentRegistryEntry(TypedDict):
    """Registry entry for a RAG agent and its provenance metadata."""

    agent: RAGAgentCallable
    fact_source: RAGFactSource





# ── Initializer ────────────────────────────────────────────────────

# NOTE FOR FUTURE AGENTS:
# This prompt runs on a local/weaker LLM, so examples are semantic boundary
# anchors, not decoration. Add examples only when they teach a new routing
# boundary or fix a recurring confusion. Prefer one concise precedence rule over
# many near-duplicate examples. Keep RAG slots factual: fetch evidence, do not
# answer or interpret on behalf of cognition.
_INITIALIZER_PROMPT = """\
You are a search-strategy planner. The character you serve is named {character_name}.
Decompose original_query into an ordered list of atomic retrieval slots.
Each slot is a DATA TARGET to look up, not an action, answer, analysis, or persona move.
Use only the top-level capability prefixes listed in this prompt.

## Rule 0 — Character name
If {character_name} appears as the person being ADDRESSED (e.g. "{character_name}, what do you think…"),
do NOT create a slot for that name.
Only create a slot for {character_name} when it IS the subject of data being retrieved
(e.g. "what did {character_name} say about…").

## Rule 0b — Evidence-dependency gate
Before adding any slot, ask: "Would the next cognition/action stage be unable to
respond safely without fetched evidence?"

If no fetched evidence is needed, return an empty slot list. Do not retrieve
person, memory, or conversation evidence for greetings, thanks, welcome-back
messages, praise, social acknowledgement, or other routine interaction acts.

Retrieve evidence when the answer needs durable facts, current facts, active
agreements, conversation provenance, profile/impression context, relationship
ranking, or web content.

## Rule 1 — Hard constraints
- Slots are data targets only. No drafting, summarising, analysing, or reasoning.
- Stop when all required facts are accounted for. The next model answers.
- One fact per slot — no "and / then / also" inside a single slot. Split if needed.
- Never invent facts absent from original_query.
- Do not add adjacent facts that are merely useful. For example, do not add
  weather for an opening-status question, traffic for a meeting-time question,
  or price for an availability question unless the user explicitly asked.
- Preserve explicit count limits from original_query, such as "3条", "last 5",
  or "recent 10", inside the conversation slot text.
- RAG gathers evidence only. Do not create slots for final judgment, persona stance, or answer wording.
- Do not generate low-level worker routes. The capability agent chooses keyword
  search, semantic search, filters, aggregates, profile reads, relationship reads,
  memory exact search, memory semantic search, or target/scope lookup internally.

## Rule 2 — Live context present-tense facts
Live-context owns present-tense facts needed for the current turn.
This includes:
- active-character current local time, current local date, and current local weekday,
- current user local time only if runtime context already provides it,
- live external facts that require fresh public evidence, such as current weather
  or temperature, live prices, exchange rates, market quotes, live scores,
  opening status, current availability, latest news, or other explicitly current
  public facts.

Use one `Live-context:` slot for every present-tense fact. The live-context
capability either answers from current runtime context or resolves target/scope
internally and then fetches fresh public evidence.
For external live slots, do not split character-location or user-location
lookup into separate memory/conversation slots. The live-context capability
owns target/scope resolution for those external live slots.

Each live slot must correspond to one live fact type directly requested by the
user. If the user asks whether a venue is open, do not also add weather,
temperature, travel, price, or schedule slots unless those facts were requested.

Runtime-backed live slots are limited to active-character current local
time/date/weekday, plus current-user local time only when runtime context is
already configured. Bare current-time questions are active-character runtime
facts; do not attach `unknown location` to them.

For external live facts, write the live slot so the target source is clear:
- explicit target or location supplied by the user,
- the active character's location,
- the current user's location if recently stated,
- unknown location/target when the query lacks enough scope.

This rule overrides memory defaults and backend wording. If original_query says
"search memory" but asks about current weather, temperature, prices, exchange
rates, opening status, latest public facts, or current local time/date/weekday,
still use `Live-context:`.

## Rule 3 — Recall active agreements and episode state
Use Recall when the user asks what was agreed, promised, planned, left unresolved,
or where the current episode left off. Recall is for active agreements, ongoing
promises, current plans, open loops, and current-episode state.
Apply this rule after present-tense live facts and before memory or conversation
defaults. Do not search conversation merely because the query contains words
like "约定", "promise", "plan", or "agreed".
Recall slot modes are fixed:
- active_episode_agreement: current/today/now/upcoming active agreement or plan.
- durable_commitment: ongoing accepted promise or obligation.
- episode_position: where the current episode left off, unresolved loops, or next step.
- exact_agreement_history: when or how an agreement was originally made.
Do not use Recall for exact quote, URL, filename, or "who said this exact phrase"
requests. Those are conversation evidence. Do not use Recall for world knowledge,
durable character/world facts, present-tense live facts, profile impressions, or
relationship ranking.

## Rule 4 — Person context
Use `Person-context:` for person/profile/relationship/user-list evidence:
- active character self-description or self-introduction,
- current user profile or durable profile facts,
- named person identity, impression, compatibility, relationship, or profile,
- relationship ranking over users,
- enumerating users by display-name or profile predicates,
- retrieving a profile for a speaker/person resolved in an earlier slot.

If the query first needs to discover an unknown speaker by content, start with
`Conversation-evidence:` and only then add `Person-context:` if profile or
impression evidence is needed.

## Rule 5 — Conversation evidence
Use `Conversation-evidence:` for evidence from chat history:
- exact phrases, quoted messages, URLs, filenames, or literal anchors,
- who said/posted/mentioned something,
- recent/fuzzy conversation topics,
- messages from current_user, active_character, any_speaker, or a person
  resolved in an earlier slot,
- counts, totals, rankings, or grouped message statistics.

For author scope, append exactly one speaker field:
speaker=current_user, speaker=active_character, speaker=any_speaker, or
speaker=person resolved in slot N. Use the slot-N form only for a person
produced by an earlier slot.

Do not use conversation evidence for active agreement recall; use Recall.
Do not use conversation evidence for durable official/world facts; use
Memory-evidence.

## Rule 6 — Memory evidence
Use `Memory-evidence:` for durable memory/world/common-sense evidence:
- official or stable character/world facts,
- the active character's official address or stable home/location,
- shared/common-sense knowledge that may enrich an answer,
- durable user memory facts or accepted preferences,
- object, place, concept, or non-human topic knowledge.

Do not use memory evidence for live external values, active agreements, or
person relationship/profile reads.

If original_query already contains all factual premises and the remaining work
is common-sense recommendation, planning, preference, or opinion, generate one
`Memory-evidence:` slot for the main non-live topic. Return an empty list only
for pure arithmetic, tautology, or trick questions where memory is irrelevant.

## Rule 7 — Web evidence
Use `Web-evidence:` only when the user asks to fetch or inspect a public web
page/topic that is not a live/current fact, or when a previous slot found a URL
whose content must be retrieved.

Current weather, current temperature, opening status, live prices, exchange
rates, schedules, current availability, and latest news are `Live-context:`,
not direct web evidence.

## Rule 8 — Context pre-check
Read the context object before generating any slot.
If global_user_id is already present in context and the user asks about the
current user's profile or durable person context, use
`Person-context: retrieve current user profile`.
Do not create Person-context merely to bind current_user for conversation
history; use speaker=current_user in the Conversation-evidence slot.
If a pronoun (他/她/你/他们) clearly refers to context user_name, write that
the person comes from context in the `Person-context:` slot.

## Conflict resolution — choose the structural evidence source
When two patterns seem possible, choose the more structural source:
- Present-tense live facts → Live-context.
- Active agreement, promise, plan, open loop, or current episode state → Recall.
- Person/profile/relationship/user-list subject → Person-context.
- Chat-history content, exact phrase, speaker, URL from a message, or message stats → Conversation-evidence.
- Durable official/world/common-sense/object facts → Memory-evidence.
- Public webpage or URL content that is not current/live → Web-evidence.

## Slot format — ALWAYS use one of these exact prefixes
When a slot depends on a specific earlier slot, write "resolved in slot N" (e.g. "slot 1", "slot 3").

- "Live-context: answer active character current local <time / date / weekday>"
- "Live-context: answer current user local time if configured"
- "Live-context: answer current <weather / temperature / opening status / price / exchange rate / schedule / availability / latest fact> for <explicit location/target X | the active character's location | the current user's location if recently stated | unknown location/target>"
- "Conversation-evidence: retrieve <exact phrase / URL / recent messages / topic / count/ranking> [speaker=current_user | speaker=active_character | speaker=any_speaker | speaker=person resolved in slot N] [to identify the speaker] [time/count limit]"
- "Memory-evidence: retrieve durable evidence about <official fact / address / common-sense topic / world fact / user memory topic>"
- "Person-context: retrieve <active character profile / current user profile / profile/impression for display name X / profile for speaker found in slot N / relationship ranking / user list predicate>"
- "Recall: retrieve <active_episode_agreement / durable_commitment / episode_position / exact_agreement_history> relevant to <topic>"
- "Web-evidence: retrieve public web content for <explicit URL/topic | URL found in slot N>"

## Pattern gallery
Examples below are boundary anchors, not an exhaustive routing table.
Generalize from the rules above when the wording differs.

### 0. Character relationship preference / ranking
Query: "<character mention>你最喜欢谁？"
   → Person-context owns relationship/user ranking; preserve count.
   ["Person-context: rank users by active character relationship from top limit 1"]

### 1. Named person → impression or compatibility
Query: "<character mention>你觉得<named user>这个人怎么样"  (character_name=<active character>)
   → <character mention> is addressee, skip. Person impression read needs person context.
   ["Person-context: retrieve profile/impression for display name <named user>"]

### 1c. Provided facts → durable/common-sense memory
Query: "我想洗车，我家距离洗车店只有 50 米，请问你推荐我走路去还是开车去呢？"
   → All facts provided. Common-sense recommendation → memory evidence on the topic.
   ["Memory-evidence: retrieve durable evidence about 洗车, short walking distance, or nearby activities"]

### 1d. Character self-profile request
Query: "<character mention>能做一个自我介绍么"  (character_name=<active character>)
   → The requested action needs character self-profile evidence. Retrieve the character's profile.
   ["Person-context: retrieve active character profile"]

### 1e. Routine interaction act with no evidence dependency
Query: "<character mention><character mention>欢迎回来"  (character_name=<active character>)
   → Greeting/welcome interaction. No person, memory, conversation, recall, live, or web evidence is needed.
   []

### 1f. Live context present-tense facts
Query: "现在几点？"
   → Bare current-time question. Use active-character runtime local time.
   ["Live-context: answer active character current local time"]

### 1g. Recall active agreements and episode state
Query: "早上好呀，还记得今天的约定么？"
   → User asks what was agreed for the active/current episode. Use Recall, not keyword self-hit.
   ["Recall: retrieve active_episode_agreement relevant to today's agreement"]

### 2. Named person → event or message history
Query: "<named user>前两天欺负你了么"
   → Specific past event involving a person: get person context, then time-bounded conversation evidence.
  ["Person-context: retrieve profile/impression for display name <named user>",
   "Conversation-evidence: retrieve messages from 2 days ago speaker=person resolved in slot 1"]

### 3. Named person → specific past quote
Query: "<named user>昨天说的AI那句是什么"
  → Named person filter, then exact/literal conversation evidence.
  ["Person-context: resolve display name <named user>",
   "Conversation-evidence: retrieve messages containing exact term 'AI', sent yesterday speaker=person resolved in slot 1"]

### 4. Direct content search, no follow-up
Query: "最近有人提到cookie管理器吗"
  → No named person. Single conversation evidence slot.
  ["Conversation-evidence: retrieve recent messages mentioning exact term 'cookie管理器' speaker=any_speaker"]

### 4b. Enumerate users by display-name predicate
Query: "所有以'子'结尾的用户"
  → User metadata predicate. Do not search message content.
  ["Person-context: list users whose display names end with '子'"]

### 4c. Factual aggregate over conversation history
Query: "最近谁发言最多"
  → Count messages by user. This is factual evidence, not interpretation.
  ["Conversation-evidence: count recent messages by user"]

### 5. Find speaker by exact phrase → get their profile
Query: "那个说5090能跑qwen27b的人，你对他有什么印象"
  → Find message first (no named person), then resolve speaker identity, then profile.
  → Human subject: impression uses Person-context, not memory evidence.
  ["Conversation-evidence: retrieve exact terms '5090' and 'qwen27b' to identify the speaker",
   "Person-context: retrieve profile/impression for speaker found in slot 1"]

### 6. Find speaker by topic → get their profile and message history
Query: "最近在聊版权保护的那个人，他平时都在群里聊些什么"
  → Topic search to find speaker, then person context, then that person's messages.
  ["Conversation-evidence: retrieve recent messages about 版权保护 to identify the speaker",
   "Person-context: retrieve profile/impression for speaker found in slot 1",
   "Conversation-evidence: retrieve recent messages speaker=person resolved in slot 2"]

### 7. Named person → find their URL → fetch URL content
Query: "<named user>发的那个小红书链接，里面写的是什么"
  → Resolve the named person, then find URL from that user, then fetch URL content.
  ["Person-context: resolve display name <named user>",
   "Conversation-evidence: retrieve messages containing a 小红书 URL speaker=person resolved in slot 1",
   "Web-evidence: retrieve public web content for the URL found in slot 2"]

### 8. Current user → find URL → fetch content
Query: "我上次发的那个链接里有什么信息"
  → The current user comes from context. Do not create Person-context just to bind it.
  ["Conversation-evidence: retrieve messages containing a URL speaker=current_user",
   "Web-evidence: retrieve public web content for the URL found in slot 1"]

### 9. Two named people → compare profiles
Query: "<named user A>和<named user B>这两个人，你对他们各有什么印象"
  → Two independent person-context reads.
  ["Person-context: retrieve profile/impression for display name <named user A>",
   "Person-context: retrieve profile/impression for display name <named user B>"]

### 10. Find speaker by exact phrase → find their URL → fetch content
Query: "说版权保护是play一环的那个人，他发过什么链接，链接里是什么内容"
  → Exact phrase to find speaker, then identity, then find their URL, then fetch it.
  ["Conversation-evidence: retrieve exact phrase '版权保护一直都是play的一环' to identify the speaker",
   "Person-context: retrieve profile/impression for speaker found in slot 1",
   "Conversation-evidence: retrieve messages containing a URL speaker=person resolved in slot 2",
   "Web-evidence: retrieve public web content for the URL found in slot 3"]

### 11. Durable official character fact
Query: "你家的官方地址是什么？"
  → Stable official character/world fact. Use durable memory, not recent conversation.
  ["Memory-evidence: retrieve durable evidence about the active character's official address"]

## Input format
{{
    "original_query": "user's question",
    "context": auxiliary info
}}

## Generation Procedure
1. Read `original_query` and first decide whether it asks for a present-tense live fact.
   If yes, apply Rule 2 before any memory default or backend wording in the query.
2. Decide whether the query asks to recall an active agreement, promise, plan,
   open loop, or current episode state. If yes, apply Rule 3 before memory or
   conversation search defaults.
3. Decide whether downstream cognition truly needs fetched evidence.
4. If evidence is needed, identify atomic data targets and order dependencies.
5. Apply the routing rules and conflict-resolution rules before writing slots.
6. Use only the allowed slot prefixes and preserve explicit counts, names, times, URLs, and exact phrases.
7. Return an empty list when the response can be handled without retrieval.

## Output format
Return valid JSON only:
{{
    "unknown_slots": ["slot 1", "slot 2", ...]
}}
"""
_initializer_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_PLANNER_LLM_MODEL,
    base_url=RAG_PLANNER_LLM_BASE_URL,
    api_key=RAG_PLANNER_LLM_API_KEY,
)
_MIN_INITIALIZER_CACHE_CONFIDENCE = 0.5


def _initializer_cache_status(
    *,
    hit: bool,
    reason: str,
    cache_key: str,
) -> dict[str, Any]:
    """Build cache metadata for the RAG initializer.

    Args:
        hit: Whether the initializer strategy was served from cache.
        reason: Machine-readable cache outcome.
        cache_key: Exact Cache 2 key used for lookup.

    Returns:
        Metadata dict stored in progressive RAG state.
    """
    return_value = {
        "enabled": True,
        "hit": hit,
        "cache_name": INITIALIZER_CACHE_NAME,
        "reason": reason,
        "cache_key": cache_key,
    }
    return return_value


def _normalize_initializer_slots(raw_slots: object) -> list[str]:
    """Normalize an initializer slot payload into a list of strings.

    Args:
        raw_slots: Value from LLM JSON or cached strategy payload.

    Returns:
        List of non-empty slot strings.
    """
    if not isinstance(raw_slots, list):
        return_value = []
        return return_value
    return_value = [slot.strip() for slot in raw_slots if isinstance(slot, str) and slot.strip()]
    return return_value


def _read_cached_initializer_slots(cached: object) -> list[str] | None:
    """Validate a cached initializer strategy payload.

    Args:
        cached: Payload returned from Cache 2.

    Returns:
        Cached slots when the strategy is valid and confident enough, otherwise
        None so the caller falls back to live initialization.
    """
    if not isinstance(cached, dict):
        return None

    raw_confidence = cached.get("confidence", 0.0)
    if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
        return None
    if raw_confidence < _MIN_INITIALIZER_CACHE_CONFIDENCE:
        return None

    raw_slots = cached.get("unknown_slots")
    if not isinstance(raw_slots, list):
        return None
    return_value = _normalize_initializer_slots(raw_slots)
    return return_value


def _initializer_cache_result(unknown_slots: list[str]) -> dict[str, Any]:
    """Build the persisted and in-memory initializer cache payload.

    Args:
        unknown_slots: Slot strategy produced by the live initializer.

    Returns:
        Cache payload containing slots and confidence.
    """

    return_value = {
        "unknown_slots": list(unknown_slots),
        "confidence": 1.0,
    }
    return return_value


def _initializer_cache_metadata() -> dict[str, Any]:
    """Build operational metadata for initializer cache entries.

    Returns:
        Metadata describing the initializer stage and version constants.
    """

    return_value = {
        "stage": "rag_initializer",
        "initializer_prompt_version": INITIALIZER_PROMPT_VERSION,
        "agent_registry_version": INITIALIZER_AGENT_REGISTRY_VERSION,
        "strategy_schema_version": INITIALIZER_STRATEGY_SCHEMA_VERSION,
    }
    return return_value


async def _write_initializer_cache(
    *,
    cache_key: str,
    unknown_slots: list[str],
) -> None:
    """Store one initializer strategy payload in Cache 2.

    Args:
        cache_key: Exact Cache 2 key for this query/context signature.
        unknown_slots: Slot strategy produced by the live initializer.
    """
    await get_rag_cache2_runtime().store(
        cache_key=cache_key,
        cache_name=INITIALIZER_CACHE_NAME,
        result=_initializer_cache_result(unknown_slots),
        dependencies=[],
        metadata=_initializer_cache_metadata(),
    )


async def rag_initializer(state: ProgressiveRAGState) -> dict:
    """Decompose original_query into an ordered list of unknown slots.

    Args:
        state: Initial state with original_query, character_name, and context.

    Returns:
        Partial state update with unknown_slots populated.
    """
    character_name = state.get("character_name", "")
    context = state.get("context", {})
    cache_key = build_initializer_cache_key(
        original_query=state["original_query"],
        character_name=character_name,
        context=context,
    )
    cached = await get_rag_cache2_runtime().get(
        cache_key,
        cache_name=INITIALIZER_CACHE_NAME,
        agent_name="rag_initializer",
    )
    cached_slots = _read_cached_initializer_slots(cached)
    if cached_slots is not None:
        logger.info(
            f"RAG2 initializer output: unknown_slots={log_list_preview(cached_slots)}"
        )
        logger.debug(
            f'RAG2 initializer metadata: cache_hit=True query={log_preview(state["original_query"])} '
            f"cache_key={cache_key}"
        )
        asyncio.create_task(record_initializer_hit(cache_key))
        return_value = {
            "unknown_slots": cached_slots,
            "initializer_cache": _initializer_cache_status(
                hit=True,
                reason="hit",
                cache_key=cache_key,
            ),
        }
        return return_value

    system_prompt = SystemMessage(content=_INITIALIZER_PROMPT.format(character_name=character_name))
    llm_context = project_runtime_context_for_llm(context)
    user_input = {
        "original_query": state["original_query"],
        "context": llm_context,
    }
    human_message = HumanMessage(content=json.dumps(user_input, ensure_ascii=False))

    response = await _initializer_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)

    cacheable_result = isinstance(result, dict) and isinstance(
        result.get("unknown_slots"), list
    )
    if not isinstance(result, dict):
        result = {}

    unknown_slots = _normalize_initializer_slots(result.get("unknown_slots", []))
    if cacheable_result:
        await _write_initializer_cache(cache_key=cache_key, unknown_slots=unknown_slots)
        asyncio.create_task(
            upsert_initializer_entry(
                cache_key=cache_key,
                result=_initializer_cache_result(unknown_slots),
                metadata=_initializer_cache_metadata(),
            )
        )

    logger.info(
        f"RAG2 initializer output: unknown_slots={log_list_preview(unknown_slots)}"
    )
    logger.debug(
        f'RAG2 initializer metadata: cache_hit=False query={log_preview(state["original_query"])} '
        f"cacheable={cacheable_result} raw={log_preview(result)}"
    )
    return_value = {
        "unknown_slots": unknown_slots,
        "initializer_cache": _initializer_cache_status(
            hit=False,
            reason="miss_stored" if cacheable_result else "miss_not_cacheable",
            cache_key=cache_key,
        ),
    }
    return return_value


# ── Dispatcher ─────────────────────────────────────────────────────
_RAG_SUPERVISOR_AGENT_REGISTRY: dict[str, RAGAgentRegistryEntry] = {
    "live_context_agent": {
        "agent": LiveContextAgent().run,
        "fact_source": {
            "source_kind": "external",
            "source_system": "live_context",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_evidence_agent": {
        "agent": ConversationEvidenceAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "memory_evidence_agent": {
        "agent": MemoryEvidenceAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "memory",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "person_context_agent": {
        "agent": PersonContextAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "user_lookup_agent": {
        "agent": UserLookupAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "user_list_agent": {
        "agent": UserListAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles_or_conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "user_profile_agent": {
        "agent": UserProfileAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "relationship_agent": {
        "agent": RelationshipAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "web_search_agent2": {
        "agent": WebSearchAgent().run,
        "fact_source": {
            "source_kind": "external",
            "source_system": "web",
            "consolidation_policy": "eligible_external_knowledge",
            "can_consolidate_as_new_knowledge": True,
        },
    },
    "conversation_aggregate_agent": {
        "agent": ConversationAggregateAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_filter_agent": {
        "agent": ConversationFilterAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_keyword_agent": {
        "agent": ConversationKeywordAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_search_agent": {
        "agent": ConversationSearchAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "persistent_memory_keyword_agent": {
        "agent": PersistentMemoryKeywordAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "memory",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "persistent_memory_search_agent": {
        "agent": PersistentMemorySearchAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "memory",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "recall_agent": {
        "agent": RecallAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "recall",
            "consolidation_policy": "operational_recall_evidence",
            "can_consolidate_as_new_knowledge": False,
        },
    },
}

_PREFIX_DISPATCH_TABLE: tuple[tuple[str, str, int], ...] = (
    ("Live-context:", "live_context_agent", 1),
    ("Conversation-evidence:", "conversation_evidence_agent", 1),
    ("Memory-evidence:", "memory_evidence_agent", 1),
    ("Person-context:", "person_context_agent", 1),
    ("Web-evidence:", "web_search_agent2", 3),
    ("Identity:", "user_lookup_agent", 3),
    ("User-list:", "user_list_agent", 3),
    ("Relationship:", "relationship_agent", 3),
    ("Profile:", "user_profile_agent", 3),
    ("Conversation-aggregate:", "conversation_aggregate_agent", 3),
    ("Conversation-filter:", "conversation_filter_agent", 3),
    ("Conversation-keyword:", "conversation_keyword_agent", 3),
    ("Conversation-semantic:", "conversation_search_agent", 3),
    ("Memory-search:", "persistent_memory_search_agent", 3),
    ("Recall:", "recall_agent", 1),
    ("Web-search:", "web_search_agent2", 3),
)

_DISPATCHER_PROMPT = '''\
You are a RAG Dispatcher. For each slot, select exactly one inner-loop retrieval agent and produce a concise task description for it.

## Agent Roster

- `live_context_agent`: Top-level present-tense live context capability.
  Answers runtime-backed current local time/date/weekday directly. For weather,
  temperature, opening status, schedules, prices, exchange rates, or current
  public status, resolves target/scope and then delegates to web.
  Use for `Live-context:` slots.

- `conversation_evidence_agent`: Top-level conversation-history evidence capability.
  Chooses keyword, semantic, filter, or aggregate conversation worker internally.
  Use for `Conversation-evidence:` slots.

- `memory_evidence_agent`: Top-level durable memory evidence capability.
  Chooses exact or semantic persistent-memory worker internally.
  Use for `Memory-evidence:` slots.

- `person_context_agent`: Top-level person/profile/relationship capability.
  Chooses identity, profile, user-list, or relationship worker internally.
  Use for `Person-context:` slots.

- `user_lookup_agent`: Direct user-profile lookup by display name.
  Queries the user_profiles collection (NOT conversation history). Returns global_user_id.
  Use as the ONLY choice when the slot is about resolving who a named person is.

- `user_list_agent`: User enumeration by display-name predicates or participant metadata.
  Handles listing users whose display names equal, contain, start with, or end with a literal value.
  Use for "all users whose names..." and similar user-list questions.

- `user_profile_agent`: Reads a user's full profile from the user-profile store.
  Use ONLY when global_user_id is already present in known_facts. Never for unknown identities.

- `relationship_agent`: Ranks profiled users by the character's relationship data.
  Use for `Relationship:` slots. The agent extracts its own ranking parameters.

- `conversation_filter_agent`: Structured filter over conversation history.
  Handles: fetching messages from a known user (by global_user_id); filtering by channel, time range, or message count.
  Prefer over search agents whenever structural filters are available.

- `conversation_aggregate_agent`: Factual aggregate over conversation history.
  Handles: counts and rankings grouped by user, optionally filtered by literal keyword, known user, channel, and time window.
  Use for "who spoke most", "how many messages", and "who mentioned X most" questions.
  It returns evidence only, not opinions or persona interpretation.

- `conversation_keyword_agent`: Exact-string search over message content.
  Handles: URLs, filenames, exact phrases, proper nouns that must appear verbatim.

- `conversation_search_agent`: Semantic similarity search over message content.
  Handles: fuzzy topic recall when exact wording is unknown.
  Use ONLY when filter and keyword agents are not applicable.

- `persistent_memory_keyword_agent`: Exact-keyword search over persistent memories.
  Handles: tags, event names, proper nouns that must appear verbatim.

- `persistent_memory_search_agent`: Semantic search over persistent memories.
  Handles durable memory evidence relevant to answering the slot when exact wording is unknown.

- `recall_agent`: Reconciles active agreements, ongoing promises, current plans,
  open loops, and current-episode state from scoped progress, active commitments,
  pending scheduled events, and gated history proof. Use for `Recall:` slots.

- `web_search_agent2`: Public internet search.
  Use ONLY when information cannot exist in local conversation history or persistent memory.

## Slot prefix → agent mapping (check this FIRST — overrides everything below)

Slots produced by the initializer start with a prefix that maps directly to an agent.
Match the prefix literally and use the mapped agent without further deliberation:

| Slot prefix                  | Agent                            |
|------------------------------|----------------------------------|
| "Live-context: ..."          | `live_context_agent`             |
| "Conversation-evidence: ..." | `conversation_evidence_agent`    |
| "Memory-evidence: ..."       | `memory_evidence_agent`          |
| "Person-context: ..."        | `person_context_agent`           |
| "Web-evidence: ..."          | `web_search_agent2`              |
| "Identity: ..."              | `user_lookup_agent`              |
| "User-list: ..."             | `user_list_agent`                |
| "Relationship: ..."          | `relationship_agent`             |
| "Profile: ..."               | `user_profile_agent`             |
| "Conversation-aggregate: ..."| `conversation_aggregate_agent`   |
| "Conversation-filter: ..."   | `conversation_filter_agent`      |
| "Conversation-keyword: ..."  | `conversation_keyword_agent`     |
| "Conversation-semantic: ..." | `conversation_search_agent`      |
| "Memory-search: ..."         | `persistent_memory_search_agent` |
| "Recall: ..."                | `recall_agent`                   |
| "Web-search: ..."            | `web_search_agent2`              |

## Fallback decision sequence — use only when slot has no recognised prefix

Evaluate top to bottom, pick the first match:

1. Slot resolves a named person's identity (display_name → global_user_id)?
   → `user_lookup_agent`.

2. Slot enumerates users by display-name pattern or participant metadata?
   → `user_list_agent`.

3. Slot needs a user's full profile AND global_user_id is already in known_facts?
   → `user_profile_agent`.

4. Slot targets a literal string (URL, filename, exact phrase) inside messages?
   → `conversation_keyword_agent`.

5. Slot asks for counts, rankings, or grouped message statistics?
   → `conversation_aggregate_agent`.

6. Slot needs messages from a known user (global_user_id available in known_facts)?
   → `conversation_filter_agent`.

7. Slot needs semantic/fuzzy recall over conversation history?
   → `conversation_search_agent`.

8. Slot targets persistent memory with a known exact keyword?
   → `persistent_memory_keyword_agent`.

9. Slot targets persistent memory semantically?
   → `persistent_memory_search_agent`.

10. Slot asks what was agreed, promised, planned, left unresolved, or where the
    current episode left off?
   → `recall_agent`.

11. Slot requires public internet data?
   → `web_search_agent2`.

## Input

{{
    "current_slot": "the slot to resolve this turn",
    "known_facts": [{{"slot": "...", "agent": "...", "resolved": true/false, "summary": "concise fact summary", "attempts": 1}}, ...],
    "context": runtime info (platform, channel, timestamp, etc.)
}}

## Generation Procedure
1. Read `current_slot` and match its prefix against the prefix-to-agent table first.
2. If there is no recognized prefix, use the fallback decision sequence.
3. Build a concise task for the chosen inner-loop agent, preserving dependency references such as "slot N".
4. Pass through only relevant context needed by the chosen agent.

## Output

Return valid JSON only:
{{
    "agent_name": "{agent_name_union}",
    "task": "task description for the chosen agent",
    "context": {{}},
    "max_attempts": 3
}}
'''
_dispatcher_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_PLANNER_LLM_MODEL,
    base_url=RAG_PLANNER_LLM_BASE_URL,
    api_key=RAG_PLANNER_LLM_API_KEY,
)


def build_rag_fact_source_map() -> dict[str, dict[str, Any]]:
    """Build the deterministic fact-source map for RAG known_facts.

    The consolidator can combine this map with each known fact's ``agent``
    field to decide whether a fact came from internal memory/history/profile
    stores or from an external source.

    Returns:
        Dict keyed by agent name with source and consolidation policy metadata.
    """
    return_value = {
        agent_name: dict(entry["fact_source"])
        for agent_name, entry in _RAG_SUPERVISOR_AGENT_REGISTRY.items()
    }
    return return_value


def _dispatch_from_known_prefix(current_slot: str) -> dict[str, Any] | None:
    """Build a dispatch directly from a recognized slot prefix.

    Args:
        current_slot: The ordered slot selected for this loop iteration.

    Returns:
        Normalized dispatch payload, or ``None`` when no prefix matches.
    """

    if not isinstance(current_slot, str):
        return None

    for prefix, agent_name, max_attempts in _PREFIX_DISPATCH_TABLE:
        if current_slot.startswith(prefix):
            dispatch = {
                "agent_name": agent_name,
                "task": current_slot.strip(),
                "context": {},
                "max_attempts": max_attempts,
                "route_source": "deterministic_prefix",
            }
            return dispatch

    return_value = None
    return return_value


async def rag_dispatcher(state: ProgressiveRAGState) -> dict:
    """Generate one plain JSON dispatch targeting the first unknown slot.

    Args:
        state: Current state with unknown_slots and known_facts.

    Returns:
        Partial state update with dispatcher JSON recorded in state,
        current_slot set, and loop_count incremented.
    """
    current_slot = state["unknown_slots"][0]
    prefix_dispatch = _dispatch_from_known_prefix(current_slot)
    if prefix_dispatch is not None:
        logger.info(
            f'RAG2 dispatch output: agent={prefix_dispatch["agent_name"]} '
            f"task={log_preview(prefix_dispatch['task'])} "
            f'route_source={prefix_dispatch["route_source"]}'
        )
        logger.debug(
            f'RAG2 dispatch metadata: loop={state.get("loop_count", 0) + 1} '
            f"slot={log_preview(current_slot)} "
            f'max_attempts={prefix_dispatch["max_attempts"]} '
            f"dispatch_context={log_preview(prefix_dispatch['context'])}"
        )
        return_value = {
            "messages": [
                AIMessage(
                    content=json.dumps(prefix_dispatch, ensure_ascii=False)
                )
            ],
            "current_slot": current_slot,
            "current_dispatch": prefix_dispatch,
            "loop_count": state.get("loop_count", 0) + 1,
        }
        return return_value

    system_prompt = SystemMessage(
        content=_DISPATCHER_PROMPT.format(
            agent_name_union=_build_agent_name_union(),
        )
    )
    llm_context = project_runtime_context_for_llm(state.get("context", {}))
    user_input = {
        "current_slot": current_slot,
        "known_facts": _known_facts_llm_view(state.get("known_facts", [])),
        "context": llm_context,
    }
    human_message = HumanMessage(content=json.dumps(user_input, ensure_ascii=False, default=str))

    # Only the last two messages — enough to preserve immediate loop history
    # without polluting the current slot with distant attempts.
    recent_messages = state["messages"][-2:] if len(state["messages"]) >= 2 else state["messages"]

    response = await _dispatcher_llm.ainvoke([system_prompt, human_message] + recent_messages)
    dispatch = parse_llm_json_output(response.content)
    if not isinstance(dispatch, dict):
        dispatch = {}
    normalized_dispatch = _normalize_dispatch(dispatch, current_slot)
    logger.info(
        f'RAG2 dispatch output: agent={normalized_dispatch["agent_name"] or "<invalid>"} '
        f"task={log_preview(normalized_dispatch['task'])} "
        f'route_source={normalized_dispatch["route_source"]}'
    )
    logger.debug(
        f'RAG2 dispatch metadata: loop={state.get("loop_count", 0) + 1} '
        f"slot={log_preview(current_slot)} "
        f'max_attempts={normalized_dispatch["max_attempts"]} '
        f'route_source={normalized_dispatch["route_source"]} '
        f"dispatch_context={log_preview(normalized_dispatch['context'])} "
        f"raw={log_preview(dispatch)}"
    )

    return_value = {
        "messages": [AIMessage(content=response.content)],
        "current_slot": current_slot,
        "current_dispatch": normalized_dispatch,
        "loop_count": state.get("loop_count", 0) + 1,
    }
    return return_value


# ── Executor ───────────────────────────────────────────────────────


def _build_delegate_context(state: ProgressiveRAGState, dispatch: dict) -> dict:
    """Merge runtime context, dispatcher hints, and known facts for one agent call."""
    delegate_context = dict(state.get("context", {}))
    raw_dispatch_context = dispatch.get("context", {})
    if isinstance(raw_dispatch_context, dict):
        delegate_context.update(raw_dispatch_context)
    delegate_context["known_facts"] = _known_facts_llm_view(
        state.get("known_facts", []),
    )
    delegate_context["original_query"] = state.get("original_query", "")
    delegate_context["current_slot"] = state.get("current_slot", "")
    return delegate_context



def _build_agent_name_union() -> str:
    """Render the dispatcher's allowed ``agent_name`` values as a union string."""
    return_value = " | ".join(_RAG_SUPERVISOR_AGENT_REGISTRY)
    return return_value


def _normalize_dispatch(raw_dispatch: dict, current_slot: str) -> dict:
    """Normalize dispatcher JSON into a safe executable dispatch payload."""
    raw_agent_name = raw_dispatch.get("agent_name", "")
    agent_name = raw_agent_name.strip() if isinstance(raw_agent_name, str) else ""
    if agent_name not in _RAG_SUPERVISOR_AGENT_REGISTRY:
        agent_name = ""

    raw_task = raw_dispatch.get("task", "")
    fallback_task = current_slot if isinstance(current_slot, str) else ""
    task = raw_task.strip() if isinstance(raw_task, str) else ""
    task = task or fallback_task
    context = raw_dispatch.get("context", {})
    if not isinstance(context, dict):
        context = {}

    raw_max_attempts = raw_dispatch.get("max_attempts", 3)
    if isinstance(raw_max_attempts, int) and not isinstance(raw_max_attempts, bool) and raw_max_attempts > 0:
        max_attempts = raw_max_attempts
    else:
        max_attempts = 3

    raw_route_source = raw_dispatch.get("route_source", "dispatcher_llm")
    if isinstance(raw_route_source, str) and raw_route_source.strip():
        route_source = raw_route_source.strip()
    else:
        route_source = "dispatcher_llm"

    return_value = {
        "agent_name": agent_name,
        "task": task,
        "context": context,
        "max_attempts": max_attempts,
        "route_source": route_source,
    }
    return return_value


def _serialize_agent_result(result: dict) -> str:
    """Serialize one agent result for dispatcher history."""
    return_value = json.dumps(result, ensure_ascii=False, default=str)
    return return_value


def _agent_result_info_view(result: dict) -> dict:
    """Build an INFO-safe view of an agent result.

    Args:
        result: Raw helper-agent result from the executor.

    Returns:
        Compact operational fields suitable for INFO logging. Heavy payloads
        such as resolved refs, projection payloads, and worker payloads remain
        available in DEBUG logs and in graph state.
    """

    raw_result = result.get("result")
    if isinstance(raw_result, dict) and "capability" in raw_result:
        compact_result = {
            "capability": raw_result.get("capability", ""),
            "primary_worker": raw_result.get("primary_worker", ""),
            "supporting_workers": raw_result.get("supporting_workers", []),
            "missing_context": raw_result.get("missing_context", []),
            "selected_summary": raw_result.get("selected_summary", ""),
        }
    else:
        compact_result = raw_result

    raw_cache = result.get("cache", {})
    cache = raw_cache if isinstance(raw_cache, dict) else {}
    info_view = {
        "agent": result.get("agent", ""),
        "resolved": bool(result.get("resolved", False)),
        "attempts": result.get("attempts", 0),
        "cache": {
            "enabled": cache.get("enabled", False),
            "hit": cache.get("hit", False),
            "reason": cache.get("reason", ""),
        },
        "result": compact_result,
    }
    return info_view


_LLM_SUMMARY_TEXT_LIMIT = 400
_LLM_SUMMARY_LIST_LIMIT = 5
_LLM_SUMMARY_REF_LIMIT = 8
_LLM_SUMMARY_PROFILE_UNIT_LIMIT = 4


def _clip_llm_summary_text(
    value: object,
    *,
    limit: int = _LLM_SUMMARY_TEXT_LIMIT,
) -> str:
    """Clip a field before sending it to a local summarizer/finalizer LLM.

    Args:
        value: Arbitrary evidence field to render.
        limit: Maximum number of characters kept from the rendered value.

    Returns:
        A string capped to the requested length.
    """

    text = str(value)
    if len(text) > limit:
        text = f"{text[:limit]}..."
    return_value = text
    return return_value


def _compact_memory_unit_rows(rows: object) -> list[object]:
    """Project memory/profile rows to the fields useful for summary prompts.

    Args:
        rows: Optional list of user-memory rows or memory-like dictionaries.

    Returns:
        A capped list with heavy metadata removed and text fields clipped.
    """

    if not isinstance(rows, list):
        return_value: list[object] = []
        return return_value

    _TIME_KEYS = {"updated_at", "timestamp"}
    compact_rows: list[object] = []
    for row in rows[:_LLM_SUMMARY_PROFILE_UNIT_LIMIT]:
        if not isinstance(row, dict):
            compact_rows.append(_clip_llm_summary_text(row))
            continue

        compact_row: dict[str, object] = {}
        for key in (
            "unit_type",
            "fact",
            "subjective_appraisal",
            "relationship_signal",
            "status",
            "updated_at",
            "timestamp",
            "memory_name",
            "content",
        ):
            if key in row:
                value = _clip_llm_summary_text(row[key])
                if key in _TIME_KEYS:
                    value = format_timestamp_for_llm(str(value))
                compact_row[key] = value
        compact_rows.append(compact_row)

    return_value = compact_rows
    return return_value


def _compact_user_memory_context(memory_context: object) -> dict[str, object]:
    """Cap user-memory context sections for evaluator/finalizer prompts.

    Args:
        memory_context: Profile memory context from a profile payload.

    Returns:
        A dictionary with the standard memory sections and clipped rows.
    """

    if not isinstance(memory_context, dict):
        return_value: dict[str, object] = {}
        return return_value

    compact_context: dict[str, object] = {}
    for section in (
        "stable_patterns",
        "recent_shifts",
        "objective_facts",
        "milestones",
        "active_commitments",
    ):
        if section in memory_context:
            compact_context[section] = _compact_memory_unit_rows(
                memory_context[section],
            )

    return_value = compact_context
    return return_value


def _compact_profile_for_llm(profile: object) -> dict[str, object]:
    """Build a bounded profile view for local LLM summary stages.

    Args:
        profile: Raw profile or character image payload.

    Returns:
        Profile identity fields plus capped memory sections.
    """

    if not isinstance(profile, dict):
        return_value: dict[str, object] = {}
        return return_value

    compact_profile: dict[str, object] = {}
    for key in (
        "global_user_id",
        "display_name",
        "name",
        "description",
        "gender",
        "age",
        "birthday",
        "backstory",
    ):
        if key in profile:
            compact_profile[key] = _clip_llm_summary_text(profile[key])

    if "self_image" in profile:
        compact_profile["self_image"] = _clip_llm_summary_text(
            profile["self_image"],
        )

    memory_context = profile.get("user_memory_context")
    compact_context = _compact_user_memory_context(memory_context)
    if compact_context:
        compact_profile["user_memory_context"] = compact_context
    else:
        memory_units = profile.get("_user_memory_units")
        compact_units = _compact_memory_unit_rows(memory_units)
        if compact_units:
            compact_profile["_user_memory_units"] = compact_units

    return_value = compact_profile
    return return_value


def _compact_projection_payload_for_llm(payload: object) -> dict[str, object]:
    """Build a summary-safe view of a top-level capability projection payload.

    Args:
        payload: The capability result projection payload.

    Returns:
        A compact payload that preserves prompt-facing facts without heavy raw
        worker/profile internals.
    """

    if not isinstance(payload, dict):
        return_value: dict[str, object] = {}
        return return_value

    compact_payload: dict[str, object] = {}
    for key in (
        "profile_kind",
        "owner_global_user_id",
        "summary",
        "external_text",
        "url",
    ):
        if key in payload:
            compact_payload[key] = _clip_llm_summary_text(payload[key])

    if "profile" in payload:
        compact_payload["profile"] = _compact_profile_for_llm(
            payload["profile"],
        )

    summaries = payload.get("summaries")
    if isinstance(summaries, list):
        compact_payload["summaries"] = [
            _clip_llm_summary_text(item)
            for item in summaries[:_LLM_SUMMARY_LIST_LIMIT]
        ]

    memory_rows = payload.get("memory_rows")
    compact_memory_rows = _compact_memory_unit_rows(memory_rows)
    if compact_memory_rows:
        compact_payload["memory_rows"] = compact_memory_rows

    return_value = compact_payload
    return return_value


def _compact_raw_result_for_llm(raw_result: object) -> object:
    """Remove heavy internals before a raw result enters a local LLM prompt.

    Args:
        raw_result: Agent raw result stored in known_facts.

    Returns:
        The original raw result for ordinary worker payloads, or a compact
        view for top-level capability/profile payloads.
    """

    projected_result = project_tool_result_for_llm(raw_result)
    if not isinstance(projected_result, dict):
        return projected_result

    if "capability" in projected_result:
        compact_result: dict[str, object] = {
            "capability": projected_result.get("capability", ""),
            "primary_worker": projected_result.get("primary_worker", ""),
            "supporting_workers": projected_result.get("supporting_workers", []),
            "source_policy": projected_result.get("source_policy", ""),
            "missing_context": projected_result.get("missing_context", []),
            "conflicts": projected_result.get("conflicts", []),
            "selected_summary": projected_result.get("selected_summary", ""),
        }

        evidence = projected_result.get("evidence")
        if isinstance(evidence, list):
            compact_result["evidence"] = [
                _clip_llm_summary_text(item)
                for item in evidence[:_LLM_SUMMARY_LIST_LIMIT]
            ]

        refs = projected_result.get("resolved_refs")
        if isinstance(refs, list):
            compact_result["resolved_refs"] = refs[:_LLM_SUMMARY_REF_LIMIT]

        projection_payload = projected_result.get("projection_payload")
        compact_result["projection_payload"] = _compact_projection_payload_for_llm(
            projection_payload,
        )
        return_value: object = compact_result
        return return_value

    if (
        "user_memory_context" in projected_result
        or "_user_memory_units" in projected_result
        or "self_image" in projected_result
    ):
        return_value = _compact_profile_for_llm(projected_result)
        return return_value

    return projected_result


def _known_facts_llm_view(known_facts: object) -> list[dict[str, object]]:
    """Compact previous facts before sending them back through an LLM.

    Args:
        known_facts: Facts accumulated by the evaluator.

    Returns:
        A list preserving slot summaries and compact raw results.
    """

    if not isinstance(known_facts, list):
        return_value: list[dict[str, object]] = []
        return return_value

    compact_facts: list[dict[str, object]] = []
    for fact in known_facts:
        if not isinstance(fact, dict):
            continue

        compact_fact = {
            "slot": fact.get("slot", ""),
            "agent": fact.get("agent", ""),
            "resolved": bool(fact.get("resolved", False)),
            "summary": _clip_llm_summary_text(fact.get("summary", "")),
            "raw_result": _compact_raw_result_for_llm(fact.get("raw_result")),
            "attempts": fact.get("attempts", 0),
        }
        compact_facts.append(compact_fact)

    return_value = compact_facts
    return return_value


async def rag_executor(state: ProgressiveRAGState) -> dict:
    """Execute the delegate-agent call produced by the dispatcher.

    Args:
        state: Current state whose dispatch payload names the agent to run.

    Returns:
        Partial state update with the normalized agent result.
    """
    dispatch = _normalize_dispatch(
        state.get("current_dispatch", {}),
        state.get("current_slot", ""),
    )
    agent_name = dispatch["agent_name"]

    if not agent_name:
        result = {
            "agent": "",
            "resolved": False,
            "result": "Dispatcher failed to choose a valid agent.",
            "attempts": 0,
        }
        logger.info(
            f"RAG2 agent output: result={log_preview(result)}"
        )
        logger.debug(
            f'RAG2 agent metadata: slot={log_preview(state.get("current_slot", ""))} '
            f"agent=<invalid> resolved=False attempts=0"
        )
        return_value = {
            "last_agent_result": result,
            "messages": [
                HumanMessage(
                    content=_serialize_agent_result(_agent_result_info_view(result)),
                    name="agent_result",
                )
            ],
        }
        return return_value

    agent = _RAG_SUPERVISOR_AGENT_REGISTRY[agent_name]["agent"]
    try:
        result = await agent(
            task=dispatch["task"],
            context=_build_delegate_context(state, dispatch),
            max_attempts=dispatch["max_attempts"],
        )
        if not isinstance(result, dict):
            result = {
                "resolved": False,
                "result": str(result),
                "attempts": dispatch["max_attempts"],
            }
        result.setdefault("agent", agent_name)
    except Exception as exc:
        logger.exception(f'Error executing agent {agent_name}: {exc}')
        result = {
            "agent": agent_name,
            "resolved": False,
            "result": f"{type(exc).__name__}: {exc}",
            "attempts": 0,
        }

    info_view = _agent_result_info_view(result)
    logger.info(f"RAG2 agent output: result={log_preview(info_view)}")
    logger.debug(
        f'RAG2 agent metadata: slot={log_preview(state.get("current_slot", ""))} '
        f'agent={agent_name} resolved={bool(result.get("resolved", False))} '
        f'attempts={result.get("attempts", 0)} raw={log_preview(result)}'
    )
    return_value = {
        "last_agent_result": result,
        "messages": [
            HumanMessage(
                content=_serialize_agent_result(info_view),
                name="agent_result",
            )
        ],
    }
    return return_value


# ── Evaluator ──────────────────────────────────────────────────────

_EVALUATOR_SUMMARIZER_PROMPT = '''\
你是一个槽位结果提炼器。给定槽位任务描述和原始工具结果，提炼出一段简洁的中文事实摘要，供后续检索代理和最终回答器使用。

# 生成步骤
1. 先读取 `slot` 和 `agent`，确认本次工具结果回答的是哪一个槽位。
2. 读取 `raw_result`，只提炼其中已经存在的事实、标识符和可引用来源。
3. 参考 `known_facts`，避免重复已经总结过的槽位结论。
4. 如果 `resolved` 为 false 或 `raw_result` 缺少可用信息，只说明本次来源没有返回什么，不要扩大结论。

# 摘要要求
- 保留对后续步骤有用的关键标识符（global_user_id、display_name、URL 等）
- 如果内容是对话记录，列出 1-5 条最相关的消息摘要（说话人 + 关键内容）
- 如果内容是用户画像或持久记忆，提炼关键事实
- 如果槽位未解决（resolved: false），简洁说明本次检索的来源没有返回什么
- 如果 raw_result 为空，不要推断先前槽位失败；只有 known_facts 明确显示先前槽位 unresolved 时才可这样说

# 输入格式
human payload 是以下 JSON：
{
    "slot": "当前槽位任务描述",
    "agent": "执行该槽位的 agent 名称",
    "resolved": true,
    "raw_result": "工具原始输出，可以是 dict/list/string/null",
    "known_facts": [{"slot": "...", "agent": "...", "resolved": true, "summary": "...", "raw_result": "...", "attempts": 1}]
}

# Recall 结果
- 如果 `agent` 是 `recall_agent` 且 `raw_result.selected_summary` 存在，必须保留该 selected_summary 的核心内容。
- 可补充 `primary_source` 与 `supporting_sources`，但不要把 progress-only recall 当成长期事实来源。

# 输出格式
- 不超过 200 字，纯文本，无 JSON 外壳
'''

_EVALUATOR_SUMMARIZER_USER_PROFILE_PROMPT = '''\
你是一个用户/角色画像槽位结果提炼器。给定 Profile 槽位、原始画像结果、以及先前身份解析结果，提炼一段简洁中文事实摘要。

# 字段语义（必须遵守）
- user_memory_context：当前用户的统一记忆投影。每条记录都包含 fact、subjective_appraisal、relationship_signal，可分别作为事实锚点、角色的主观评价、未来互动信号。
- objective_facts、milestones、active_commitments 若出现，应来自 user_memory_context 内部分类，不再作为旧的独立画像来源处理。
- 如果 raw_result 包含 name/description/gender/age/birthday/backstory/self_image，且不包含 user_memory_context：
  这是角色自己的公开资料或自我画像。按“角色自身资料”总结，不要当成第三方用户画像。

# 生成步骤
1. 先读取 `slot`、`agent` 与 `known_facts`，确认本次 profile 结果对应当前用户、第三方用户还是角色自身。
2. 如果 `raw_result` 包含 `user_memory_context`，按五类记忆单元总结：先写事实锚点，再写角色的主观评价和关系信号。
3. 如果 `raw_result` 是角色公开资料或 `self_image`，按角色自身资料总结，不要写成用户画像。
4. 只使用 `raw_result` 中已有的信息；未知字段保持未知，不要补全。

# 摘要要求
- 保留 global_user_id、display_name 等对后续步骤有用的标识。
- 明确区分“目标用户是谁”、事实锚点是什么、角色的主观评价是什么。
- 只总结 raw_result 中已有的信息，不要补全未知信息。

# 输入格式
human payload 是以下 JSON：
{
    "slot": "当前 Profile 槽位任务描述",
    "agent": "user_profile_agent",
    "resolved": true,
    "raw_result": {
        "global_user_id": "用户 UUID",
        "display_name": "用户显示名",
        "user_memory_context": {
            "stable_patterns": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间YYYY-MM-DD HH:MM"}],
            "recent_shifts": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间YYYY-MM-DD HH:MM"}],
            "objective_facts": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间YYYY-MM-DD HH:MM"}],
            "milestones": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间YYYY-MM-DD HH:MM"}],
            "active_commitments": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间YYYY-MM-DD HH:MM", "due_at": "可选本地到期时间YYYY-MM-DD HH:MM", "due_state": "no_due_date | future_due | due_today | past_due | unknown_due_date"}]
        }
    },
    "known_facts": [{"slot": "...", "agent": "...", "resolved": true, "summary": "...", "raw_result": "...", "attempts": 1}]
}

# 输出格式
- 不超过 220 字，纯文本，无 JSON 外壳。
'''

_evaluator_summarizer_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)

async def _summarize_agent_result(
    slot: str,
    agent_name: str,
    resolved: bool,
    raw_result: object,
    known_facts: list[dict],
) -> str:
    """Distil a resolved agent result into a concise fact summary for downstream agents.

    Only called when resolved=True. Unresolved slots receive a deterministic
    template in rag_evaluator and never reach this function.

    Args:
        slot: The slot description that was being resolved.
        agent_name: Inner-loop agent that produced the raw result.
        resolved: Whether the inner-loop agent judged the slot as resolved.
        raw_result: Native tool output from the inner-loop agent (dict, list, str, or None).
        known_facts: Facts resolved before this slot.

    Returns:
        A concise Chinese-language summary of the key facts extracted.
    """
    if agent_name == "user_profile_agent":
        prompt = _EVALUATOR_SUMMARIZER_USER_PROFILE_PROMPT
    else:
        prompt = _EVALUATOR_SUMMARIZER_PROMPT

    compact_raw_result = _compact_raw_result_for_llm(raw_result)
    compact_known_facts = _known_facts_llm_view(known_facts)
    system_prompt = SystemMessage(content=prompt)
    human_message = HumanMessage(
        content=json.dumps(
            {
                "slot": slot,
                "agent": agent_name,
                "resolved": resolved,
                "raw_result": compact_raw_result,
                "known_facts": compact_known_facts,
            },
            ensure_ascii=False,
            default=str,
        )
    )
    response = await _evaluator_summarizer_llm.ainvoke([system_prompt, human_message])
    return_value = response.content.strip()
    return return_value


async def rag_evaluator(state: ProgressiveRAGState) -> dict:
    """Record the inner-agent verdict for the current slot, then drain it.

    The slot is always removed from unknown_slots regardless of success. The
    evaluator normalizes the agent result, calls the summarizer to produce a
    concise fact summary, and stores both the summary and raw result.

    Args:
        state: Current state after the executor ran.

    Returns:
        Partial state update with the current slot removed from unknown_slots
        and the agent verdict (with summary and raw_result) appended to known_facts.
    """
    agent_result = state.get("last_agent_result", {})
    if not isinstance(agent_result, dict):
        agent_result = {}

    slot = state.get("current_slot", "")
    resolved = bool(agent_result.get("resolved", False))
    raw_result = agent_result.get("result")
    known_facts = state.get("known_facts", [])
    agent_name = str(agent_result.get("agent", ""))

    if resolved:
        summary = await _summarize_agent_result(
            slot,
            agent_name,
            resolved,
            raw_result,
            known_facts,
        )
    else:
        summary = f"检索未返回相关结果。槽位: {slot}"

    new_fact = {
        "slot": slot,
        "agent": agent_name,
        "resolved": resolved,
        "summary": summary,
        "raw_result": raw_result,
        "attempts": int(agent_result.get("attempts", 0) or 0),
    }

    remaining_slots = list(state.get("unknown_slots", []))[1:]
    logger.info(
        f'RAG2 fact: slot={log_preview(slot)} agent={agent_name or "<none>"} '
        f"resolved={resolved} summary={log_preview(summary)}"
    )
    logger.debug(
        f'RAG2 fact metadata: attempts={new_fact["attempts"]} '
        f"remaining_slots={len(remaining_slots)}"
    )
    logger.debug(f"RAG2 fact detail: raw_result={log_preview(raw_result)}")

    return_value = {
        "unknown_slots": remaining_slots,
        "known_facts": known_facts + [new_fact],
    }
    return return_value


# ── Finalizer ──────────────────────────────────────────────────────

_FINALIZER_PROMPT = '''\
你是一个事实总结员。请根据 `known_facts` 生成简短事实摘要。

# 生成步骤
1. 先读取 `original_query`，确认本次摘要需要覆盖的事实类型。
2. 按顺序读取 `known_facts`，只使用 resolved 槽位中的 summary 和 raw_result。
3. 如果 user_profile_agent 的 raw_result 包含 user_memory_context，区分 fact、subjective_appraisal、relationship_signal 三种语义。
4. 如果某个必要槽位 unresolved，只说明缺少该槽位信息。
5. 如果 agent="recall_agent"，优先使用 raw_result.selected_summary 总结约定/承诺/进度事实。
6. 输出一段短的事实摘要；说话人、来源、时间和引用都应来自 `known_facts` 中可见内容。

# 准则
- 围绕 `original_query` 需要的事实组织摘要，不要复述查找过程。
- 如果 known_facts 为空，说明本次 RAG 没有需要检索的外部/内部事实；不要说“缺少关于该问题的具体信息”。
- 如果某个槽位未能解决（resolved: false），如实告知缺少哪一部分信息。
- 不要把某个来源没有检索结果扩大成“没有任何记录/没有互动记录”；只能说明实际查询过的来源没有返回结果。
- 引用来源 URL 或对话来源时尽量保留。
- 对 conversation evidence，按“可见来源/说话人标签 + 时间 + 内容”的方式摘要；没有可见标签时使用“说话人”。
- 引用对话原文时，保留原文内部的人称，不要改写引用内容。
- 当 known_facts 中 agent="user_profile_agent" 且 raw_result 包含 user_memory_context：
  fact 是事实锚点，subjective_appraisal 是画像来源的主观评价，relationship_signal 是未来互动信号。
  回答时不要把 subjective_appraisal 误写成目标用户自己的感受。
- 当 known_facts 中 agent="user_profile_agent" 且 raw_result 是公开资料或 self_image：
  这是 self_image 或公开资料对应的主体资料。回答自我资料问题时，可以使用这些公开资料；
  不要误写成第三方用户画像。
- 当 known_facts 中 agent="recall_agent" 且 raw_result 包含 selected_summary：
  这是当前约定/承诺/进度的已仲裁回忆结果。直接使用 selected_summary 回答，不要改搜关键字或把它改写成长期设定。

# 输入格式
{
    "original_query": "用户原始问题",
    "known_facts": [{"slot": ..., "agent": ..., "resolved": ..., "summary": "简洁事实摘要", "raw_result": "原始工具输出（如需引用原文）", "attempts": ...}, ...]
}

# 输出格式
请直接返回一段自然语言事实摘要（纯文本，无 JSON 外壳）。
- no markdown formatting
- preserve visible source/speaker labels
- no broad interpretation beyond short extractive summaries
'''
_finalizer_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def rag_finalizer(state: ProgressiveRAGState) -> dict:
    """Synthesise the final answer from all collected slot results.

    Args:
        state: Final state after all slots have been processed.

    Returns:
        Partial state update with final_answer set.
    """
    system_prompt = SystemMessage(content=_FINALIZER_PROMPT)
    finalizer_input = {
        "original_query": state["original_query"],
        "known_facts": _known_facts_llm_view(state.get("known_facts", [])),
    }
    human_message = HumanMessage(content=json.dumps(finalizer_input, ensure_ascii=False, default=str))

    response = await _finalizer_llm.ainvoke([system_prompt, human_message])
    logger.info(f"RAG2 finalizer output: answer={log_preview(response.content)}")
    logger.debug(
        f'RAG2 finalizer metadata: query={log_preview(state["original_query"])} '
        f'facts={len(state.get("known_facts", []))}'
    )
    return_value = {"final_answer": response.content}
    return return_value


# ── Routing ────────────────────────────────────────────────────────

def _route_after_initializer(state: ProgressiveRAGState) -> str:
    """Skip to finalizer if the initializer produced no slots."""
    return_value = "dispatch" if state.get("unknown_slots") else "finalize"
    return return_value


def _route_after_dispatcher(state: ProgressiveRAGState) -> str:
    """Execute if the dispatcher produced a valid agent name, else finalize."""
    dispatch = _normalize_dispatch(
        state.get("current_dispatch", {}),
        state.get("current_slot", ""),
    )
    if dispatch["agent_name"]:
        return "execute"
    return "finalize"


def _route_after_evaluator(state: ProgressiveRAGState) -> str:
    """Stop immediately if the last slot failed; loop if slots remain under cap."""
    known_facts = state.get("known_facts", [])
    if known_facts and not known_facts[-1].get("resolved", True):
        return "finalize"
    if state.get("unknown_slots") and state.get("loop_count", 0) < _MAX_LOOP_COUNT:
        return "loop"
    return "finalize"


# ── Graph entry-point ──────────────────────────────────────────────

async def call_rag_supervisor(
    original_query: str,
    character_name: str = "",
    context: dict | None = None,
) -> dict:
    """Run the progressive RAG supervisor over a single query.

    Args:
        original_query: User's natural-language question.
        character_name: Display name of the active character. Used by
            the initializer to distinguish addressee references from subject
            references so it does not generate spurious slots for the character.
        context: Optional auxiliary fields (platform/channel/target user UUID).

    Returns:
        Dict with keys ``answer``, ``known_facts``, ``unknown_slots``
        (any that were not drained), and ``loop_count``.
    """
    runtime_context = context or {}
    builder = StateGraph(ProgressiveRAGState)

    builder.add_node("rag_initializer", rag_initializer)
    builder.add_node("rag_dispatcher", rag_dispatcher)
    builder.add_node("rag_executor", rag_executor)
    builder.add_node("rag_evaluator", rag_evaluator)
    builder.add_node("rag_finalizer", rag_finalizer)

    builder.add_edge(START, "rag_initializer")
    builder.add_conditional_edges(
        "rag_initializer",
        _route_after_initializer,
        {"dispatch": "rag_dispatcher", "finalize": "rag_finalizer"},
    )
    builder.add_conditional_edges(
        "rag_dispatcher",
        _route_after_dispatcher,
        {"execute": "rag_executor", "finalize": "rag_finalizer"},
    )
    builder.add_edge("rag_executor", "rag_evaluator")
    builder.add_conditional_edges(
        "rag_evaluator",
        _route_after_evaluator,
        {"loop": "rag_dispatcher", "finalize": "rag_finalizer"},
    )
    builder.add_edge("rag_finalizer", END)

    graph = builder.compile()

    initial_state: ProgressiveRAGState = {
        "original_query": original_query,
        "character_name": character_name,
        "context": runtime_context,
        "unknown_slots": [],
        "current_slot": "",
        "known_facts": [],
        "messages": [],
        "initializer_cache": {},
        "current_dispatch": {},
        "last_agent_result": {},
        "loop_count": 0,
        "final_answer": "",
    }

    logger.debug(
        f'RAG2 request metadata: platform={runtime_context.get("platform", "")} '
        f'channel={runtime_context.get("platform_channel_id", "") or "<dm>"} '
        f'user={runtime_context.get("global_user_id", "")} '
        f"character={log_preview(character_name)} "
        f'history_recent={len(runtime_context.get("chat_history_recent", []))} '
        f'history_wide={len(runtime_context.get("chat_history_wide", []))} '
        f"query={log_preview(original_query)} "
        f"context={log_preview(runtime_context)}"
    )
    result = await graph.ainvoke(initial_state)
    known_facts = result.get("known_facts", [])
    unknown_slots = result.get("unknown_slots", [])
    loop_count = result.get("loop_count", 0)
    final_answer = result.get("final_answer", "")

    logger.debug(
        f'RAG2 summary metadata: platform={runtime_context.get("platform", "")} '
        f'channel={runtime_context.get("platform_channel_id", "") or "<dm>"} '
        f'user={runtime_context.get("global_user_id", "")} '
        f"loop_count={loop_count} known_facts={len(known_facts)} "
        f"unknown_slots={len(unknown_slots)} answer={log_preview(final_answer)} "
        f"facts={log_preview(known_facts)} "
        f"remaining_slots={log_list_preview(unknown_slots)}"
    )

    return_value = {
        "answer": final_answer,
        "known_facts": known_facts,
        "unknown_slots": unknown_slots,
        "loop_count": loop_count,
    }
    return return_value


async def test_main():
    """Simple debug entry-point."""
    try:
        await mcp_manager.start()
    except Exception as exc:
        logger.exception(f"MCP manager failed to start — web tools will be unavailable: {exc}")

    # Dummy GlobalPersonaState-equivalent fields
    character_profile = {
        "name": "<active character>",
        "description": "一个温柔的AI角色",
    }
    user_profile = {
        "affinity": 800,
        "display_name": "<current user>",
    }

    result = await call_rag_supervisor(
        original_query="<character mention><character mention>欢迎回来",
        character_name=character_profile["name"],
        context={
            "platform": "qq",
            "platform_channel_id": "902317662",
            "user_name": user_profile.get("display_name", ""),
            "current_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )

    print("=" * 80)
    print(f"Answer:\n{result['answer']}")
    print("-" * 80)
    print(f"Loop count:       {result['loop_count']}")
    print(f"Remaining slots:  {result['unknown_slots']}")
    print("-" * 80)
    print("Known facts:")
    print(json.dumps(result["known_facts"], ensure_ascii=False, indent=2, default=str))

    await mcp_manager.stop()


if __name__ == "__main__":
    asyncio.run(test_main())
