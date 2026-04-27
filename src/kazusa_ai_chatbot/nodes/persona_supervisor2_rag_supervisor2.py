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
from kazusa_ai_chatbot.rag.conversation_filter_agent import ConversationFilterAgent
from kazusa_ai_chatbot.rag.conversation_keyword_agent import ConversationKeywordAgent
from kazusa_ai_chatbot.rag.conversation_search_agent import ConversationSearchAgent
from kazusa_ai_chatbot.rag.persistent_memory_keyword_agent import PersistentMemoryKeywordAgent
from kazusa_ai_chatbot.rag.persistent_memory_search_agent import PersistentMemorySearchAgent
from kazusa_ai_chatbot.rag.relationship_agent import RelationshipAgent
from kazusa_ai_chatbot.rag.user_list_agent import UserListAgent
from kazusa_ai_chatbot.rag.user_lookup_agent import UserLookupAgent
from kazusa_ai_chatbot.rag.user_profile_agent import UserProfileAgent
from kazusa_ai_chatbot.rag.web_search_agent import WebSearchAgent
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

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
Each slot is a DATA TARGET to look up — never an action, analysis, or task to perform.

## Rule 0 — Character name
If {character_name} appears as the person being ADDRESSED (e.g. "{character_name}, what do you think…"),
do NOT create a slot for that name.
Only create a slot for {character_name} when it IS the subject of data being retrieved
(e.g. "what did {character_name} say about…").

## Rule 0b — Character relationship preference / ranking
Use Relationship only for character-to-users preference/ranking queries over unnamed
users ("who do you like most", "has someone liked", "most hated", "top N").
Named-person relationship reads remain Identity → Profile under Rule 3.

## Rule 0c — Character self-profile requests
If the user asks {character_name} to talk about herself / themselves / "you" / "yourself"
as the DATA SUBJECT, retrieve the character's own profile. This includes Chinese phrasing
such as "你自己", "聊聊你自己", "介绍一下你自己", "说说你自己", or "你是什么样的人".
Do NOT treat these as already-known conversational requests.
Generate:
  Slot 1: "Identity: look up display name '{character_name}' to get global_user_id"
  Slot 2: "Profile: retrieve full user profile for the user resolved in slot 1"

## Rule 1 — Hard constraints
- Slots are data targets only. No drafting, summarising, analysing, or reasoning.
- Stop when all required facts are accounted for. The next model answers.
- One fact per slot — no "and / then / also" inside a single slot. Split if needed.
- Never invent facts absent from original_query.
- Preserve explicit count limits from original_query, such as "3条", "last 5",
  or "recent 10", inside the conversation slot text.
- RAG gathers evidence only. Do not create slots for final judgment, persona stance, or answer wording.

## Rule 1a — Evidence-dependency gate
Before adding any slot, ask: "Would the next cognition/action stage be unable to respond safely
without this fetched evidence?"

If the answer is no, return an empty slot list. Do not retrieve Profile, Identity, Memory-search,
or Conversation evidence just because the query contains a name, greeting, praise, welcome,
thanks, social acknowledgement, or other interaction act.

Use Profile only when the follow-up answer needs profile-derived evidence, such as:
- character self-description or self-introduction
- opinion, relationship, compatibility, or impression about a person
- remembered facts, commitments, diaries, user image, or stable profile facts

Do not use Profile for routine interaction acts where the response can be produced from the
conversation act itself.

## Rule 1b — Default Memory-search for self-contained queries
If original_query already contains all factual premises needed and the remaining work is
common sense, planning, preference, recommendation, or opinion, generate one Memory-search
slot targeting the query's main topic.

The character may hold relevant world knowledge, safety rules, or personal experience in
persistent memory that enriches the answer. Always check, regardless of whether the user
explicitly asks to recall memory.
The Memory-search slot should target the topic, not restate the common-sense question itself.

Do NOT add Web-search or structured-data slots (Identity, Profile, Conversation-*) merely
because the query mentions location, distance, travel, shopping, or a real-world action.

Exception — return empty slots only when the query is pure arithmetic, a tautology, or a
trick question where memory is structurally irrelevant (e.g. "what is 2+2").

## Rule 2 — Context pre-check
Read the context object before generating any slot.
If global_user_id is already present in context, skip the Identity slot for that person.
If a pronoun (他/她/你/他们) clearly refers to the person in context user_name, treat user_name as the resolved display name and generate an Identity slot for it.

## Rule 3 — Identity-first, Profile-always for person subjects
When a NAMED PERSON is the starting point, begin with Identity.
When that person is the primary relational subject (their behavior, interactions,
events, impressions, or relationship with the character), insert Profile immediately
after Identity and before any secondary slots.

Standard order: Identity → Profile → [secondary slots if needed]

For pure impression or relationship-read queries ("怎么样", "合得来么", "what do you
think of X"), Profile alone is sufficient — no secondary slot needed.
Never use Memory-search for person-relationship data; relationship data lives in
the profile, not in persistent memory.

Exception: when the person is purely a content filter to locate a URL or exact
quoted phrase with no relational dimension, the Profile slot may be omitted.

## Rule 5 — User-list pattern
Use this when the query asks to enumerate users by display-name pattern, profile/user metadata,
or observed participant metadata. Do NOT use Conversation-keyword for display-name predicates.

  Applies to:
  - "all users whose name ends with <suffix>" → User-list
  - "users whose display names contain <term>" → User-list
  - "who in this channel has a name starting with <prefix>" → User-list

## Rule 6 — Content-first pattern (no named starting person)
Use this when the query seeks content BY TOPIC, URL, EXACT PHRASE, or KEYWORD.
Generate a content retrieval slot FIRST. The speaker identity comes from that result.
If the query asks for MORE about that identified speaker, append Identity + content slots AFTER.
If the query asks for counts, rankings, totals, or "most/least", use Rule 7 instead.

  Applies to:
  - "Who said <exact phrase>?" → Conversation-keyword [→ Identity → Profile → Filter if follow-up needed]
  - "Who posted that <link type>?" → Conversation-keyword
  - "Who has been talking about <topic>?" → Conversation-semantic [→ Identity → more if needed]
  - "Has anyone mentioned <term>?" → Conversation-keyword

## Rule 7 — Conversation aggregate pattern
Use this when the query asks for factual counts, rankings, or grouped message statistics.
These slots compute evidence only; do not use them for opinions or persona interpretation.

  Applies to:
  - "Who spoke the most recently?" → Conversation-aggregate
  - "How many messages did <named user> send?" → Identity then Conversation-aggregate
  - "Who mentioned <literal term> most often?" → Conversation-aggregate

## Conflict resolution — choose the structural evidence source
When two patterns seem possible, choose the more structural source:
- Character relationship preference/ranking over users → Relationship, not Profile.
- Display-name or user metadata predicates → User-list, not Conversation-keyword.
- Counts, totals, rankings, "most", or "least" → Conversation-aggregate, not Conversation-keyword/semantic.
- Person is primary relational subject → Identity + Profile always (Rule 3), then secondary slots; never Memory-search for person-relationship data.
- Impression or opinion about an OBJECT, concept, or non-human topic → Memory-search.
- All facts provided, common-sense or opinion query → Memory-search on the topic (Rule 1b default); empty slots only for pure arithmetic or tautologies.
- Exact quoted phrases, URLs, filenames, or literal content anchors → Conversation-keyword.
- Fuzzy topics without exact wording → Conversation-semantic.

## Slot format — ALWAYS use one of these exact prefixes
When a slot depends on a specific earlier slot, write "resolved in slot N" (e.g. "slot 1", "slot 3").

- "Identity: look up display name '<name>' to get global_user_id"
- "User-list: list users whose display names <equals / contain / start with / end with> '<value>' [from known profiles / observed conversation participants / both]"
- "Relationship: rank users by character relationship [top / bottom] [limit N / existence check]"
- "Profile: retrieve full user profile for the user resolved in slot N"
- "Conversation-aggregate: count/rank messages by user [containing '<literal term>'] [from the user resolved in slot N] [recent / today / yesterday / all]"
- "Conversation-filter: retrieve [recent / yesterday's / last N] messages from the user resolved in slot N"
- "Conversation-keyword: find messages containing <exact phrase or term> [from the user resolved in slot N]"
- "Conversation-semantic: find recent messages about <topic> [from the user resolved in slot N]"
- "Memory-search: search persistent memory for impressions or opinions about a topic, concept, or non-human subject"
- "Web-search: search the web for <description of target URL or topic from slot N>"

## Pattern gallery

### 0. Character relationship preference / ranking (1 slot)
Queries: "千纱你最喜欢谁？", "你最喜欢谁？", "千纱有喜欢的人了么", "千纱最讨厌谁？", "千纱最喜欢的三个人"
  → Relationship over users; top means high affinity, bottom means low affinity; preserve count.
  ["Relationship: rank users by character relationship from top, limit 1"]
  ["Relationship: check whether a top-ranked relationship candidate exists"]

### 1. Named person → impression or compatibility (2 slots)
Query: "千纱你觉得小钳子这个人怎么样"  (character_name=千纱)
  → 千纱 is addressee, skip. Impression/compatibility read → Identity + Profile only (Rule 3).
  → Same routing for "合得来么", "什么印象", "怎么样" — no secondary slot, no Memory-search.
  ["Identity: look up display name '小钳子' to get global_user_id",
   "Profile: retrieve full user profile for the user resolved in slot 1"]

### 1c. Provided facts → Memory-search on topic (1 slot)
Query: "我想洗车，我家距离洗车店只有 50 米，请问你推荐我走路去还是开车去呢？"
  → All facts provided. Common-sense recommendation → default Memory-search on the topic (Rule 1b).
  → Explicit hints ("根据你的记忆回答") and no-hint queries route identically.
  ["Memory-search: search persistent memory for experiences or impressions related to 洗车 or nearby activities"]

### 1d. Character self-profile request (2 slots)
Query: "千纱聊聊你自己"  (character_name=千纱)
  → 千纱 is addressed, but "你自己" is also the data subject. Retrieve the character's profile.
  ["Identity: look up display name '千纱' to get global_user_id",
   "Profile: retrieve full user profile for the user resolved in slot 1"]

Query: "千纱能做一个自我介绍么"  (character_name=千纱)
  → The requested action needs character self-profile evidence. Retrieve the character's profile.
  ["Identity: look up display name '千纱' to get global_user_id",
   "Profile: retrieve full user profile for the user resolved in slot 1"]

### 1e. Routine interaction act with no evidence dependency (0 slots)
Query: "千纱千纱欢迎回来"  (character_name=千纱)
  → Greeting/welcome interaction. No profile, memory, identity, or conversation evidence is needed.
  []

Query: "千纱辛苦啦"  (character_name=千纱)
  → Social acknowledgement. The next stage can respond directly without retrieval.
  []

### 2. Named person → event or message history (3 slots)
Query: "小钳子前两天欺负你了么"
  → Specific past event: Identity → Profile (Rule 3) → time-bounded Conversation-filter.
  ["Identity: look up display name '小钳子' to get global_user_id",
   "Profile: retrieve full user profile for the user resolved in slot 1",
   "Conversation-filter: retrieve messages from the user resolved in slot 1 from 2 days ago"]

Query: "蚝爹油最近在聊什么"
  → Recent messages: same structure, open-ended filter. Preserve explicit counts if present.
  ["Identity: look up display name '蚝爹油' to get global_user_id",
   "Profile: retrieve full user profile for the user resolved in slot 1",
   "Conversation-filter: retrieve recent messages from the user resolved in slot 1"]

### 3. Named person → specific past quote (3 slots)
Query: "小钳子昨天说的AI那句是什么"
  → Named person → identity-first, then profile (Rule 3), then keyword search.
  ["Identity: look up display name '小钳子' to get global_user_id",
   "Profile: retrieve full user profile for the user resolved in slot 1",
   "Conversation-keyword: find messages from the user resolved in slot 1 containing 'AI', sent yesterday"]

### 4. Direct content search, no follow-up (1 slot)
Query: "最近有人提到cookie管理器吗"
  → No named person. Single keyword slot.
  ["Conversation-keyword: find recent messages mentioning 'cookie管理器'"]

Query: "最近在聊版权保护的是谁"
  → No named person. Single semantic slot.
  ["Conversation-semantic: find recent messages about 版权保护 (copyright protection)"]

### 4b. Enumerate users by display-name predicate (1 slot)
Query: "所有以'子'结尾的用户"
  → User metadata predicate. Do not search message content.
  ["User-list: list users whose display names end with '子' from known profiles"]

### 4c. Factual aggregate over conversation history (1 slot)
Query: "最近谁发言最多"
  → Count messages by user. This is factual evidence, not interpretation.
  ["Conversation-aggregate: count recent messages by user"]

Query: "最近谁提到cookie管理器最多"
  → Count messages containing the literal term by user.
  ["Conversation-aggregate: count recent messages by user containing 'cookie管理器'"]

### 5. Find speaker by exact phrase → get their profile (3 slots)
Query: "那个说5090能跑qwen27b的人，你对他有什么印象"
  → Find message first (no named person), then resolve speaker identity, then profile.
  → Human subject: impression always resolves to Profile, not Memory-search.
  ["Conversation-keyword: find messages containing '5090' and 'qwen27b' to identify the speaker",
   "Identity: look up display name of the person found in slot 1 to get global_user_id",
   "Profile: retrieve full user profile for the user resolved in slot 2"]

### 6. Find speaker by topic → get their profile and message history (4 slots)
Query: "最近在聊版权保护的那个人，他平时都在群里聊些什么"
  → Semantic search to find speaker, then identity, then profile (Rule 3b), then messages.
  ["Conversation-semantic: find recent messages about 版权保护 to identify the speaker",
   "Identity: look up display name of the person identified in slot 1 to get global_user_id",
   "Profile: retrieve full user profile for the user resolved in slot 2",
   "Conversation-filter: retrieve recent messages from the user resolved in slot 2"]

### 7. Named person → find their URL → fetch URL content (3 slots)
Query: "落郇之源发的那个小红书链接，里面写的是什么"
  → Named person → identity-first. Then find URL from that user. Then fetch URL content.
  ["Identity: look up display name '落郇之源' to get global_user_id",
   "Conversation-keyword: find messages from the user resolved in slot 1 containing a 小红书 URL",
   "Web-search: retrieve the content at the 小红书 URL found in slot 2"]

### 8. Pronoun resolved from context → find URL → fetch content (3 slots)
Query: "他上次说的那个链接里有什么信息"  (context has user_name='小钳子', '他' refers to current user)
  → '他' maps to context user_name. Identity lookup, then find URL, then fetch.
  ["Identity: look up display name '小钳子' (from context user_name, resolving '他') to get global_user_id",
   "Conversation-keyword: find messages from the user resolved in slot 1 containing a URL",
   "Web-search: retrieve the content at the URL found in slot 2"]

### 9. Two named people → compare profiles (4 slots)
Query: "小钳子和蚝爹油这两个人，你对他们各有什么印象"
  → Two independent identity+profile chains (human subjects → Profile, not Memory-search).
  → Slot 4 depends on slot 3, NOT slot 1.
  ["Identity: look up display name '小钳子' to get global_user_id",
   "Profile: retrieve full user profile for the user resolved in slot 1",
   "Identity: look up display name '蚝爹油' to get global_user_id",
   "Profile: retrieve full user profile for the user resolved in slot 3"]

### 10. Find speaker by exact phrase → find their URL → fetch content (4 slots)
Query: "说版权保护是play一环的那个人，他发过什么链接，链接里是什么内容"
  → Exact phrase to find speaker, then identity, then find their URL, then fetch it.
  ["Conversation-keyword: find messages containing '版权保护一直都是play的一环' to identify the speaker",
   "Identity: look up display name of the person found in slot 1 to get global_user_id",
   "Conversation-keyword: find messages from the user resolved in slot 2 containing a URL",
   "Web-search: retrieve the content at the URL found in slot 3"]

## Input format
{{
    "original_query": "user's question",
    "context": auxiliary info
}}

## Output format
Return valid JSON only:
{{
    "unknown_slots": ["slot 1", "slot 2", ...]
}}
"""
_initializer_llm = get_llm(temperature=0.0, top_p=1.0)
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
    return {
        "enabled": True,
        "hit": hit,
        "cache_name": INITIALIZER_CACHE_NAME,
        "reason": reason,
        "cache_key": cache_key,
    }


def _normalize_initializer_slots(raw_slots: object) -> list[str]:
    """Normalize an initializer slot payload into a list of strings.

    Args:
        raw_slots: Value from LLM JSON or cached strategy payload.

    Returns:
        List of non-empty slot strings.
    """
    if not isinstance(raw_slots, list):
        return []
    return [str(slot).strip() for slot in raw_slots if str(slot).strip()]


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
    return _normalize_initializer_slots(raw_slots)


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
        result={
            "unknown_slots": list(unknown_slots),
            "confidence": 1.0,
        },
        dependencies=[],
        metadata={
            "stage": "rag_initializer",
            "initializer_prompt_version": INITIALIZER_PROMPT_VERSION,
            "agent_registry_version": INITIALIZER_AGENT_REGISTRY_VERSION,
            "strategy_schema_version": INITIALIZER_STRATEGY_SCHEMA_VERSION,
        },
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
        logger.info("Initializer cache hit: %s", cached_slots)
        return {
            "unknown_slots": cached_slots,
            "initializer_cache": _initializer_cache_status(
                hit=True,
                reason="hit",
                cache_key=cache_key,
            ),
        }

    system_prompt = SystemMessage(content=_INITIALIZER_PROMPT.format(character_name=character_name))
    user_input = {
        "original_query": state["original_query"],
        "context": context,
    }
    human_message = HumanMessage(content=json.dumps(user_input, ensure_ascii=False))

    response = await _initializer_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(str(response.content))

    cacheable_result = isinstance(result, dict) and isinstance(
        result.get("unknown_slots"), list
    )
    if not isinstance(result, dict):
        result = {}

    unknown_slots = _normalize_initializer_slots(result.get("unknown_slots", []))
    if cacheable_result:
        await _write_initializer_cache(cache_key=cache_key, unknown_slots=unknown_slots)

    logger.info("Initializer slots: %s", unknown_slots)
    return {
        "unknown_slots": unknown_slots,
        "initializer_cache": _initializer_cache_status(
            hit=False,
            reason="miss_stored" if cacheable_result else "miss_not_cacheable",
            cache_key=cache_key,
        ),
    }


# ── Dispatcher ─────────────────────────────────────────────────────
_RAG_SUPERVISOR_AGENT_REGISTRY: dict[str, RAGAgentRegistryEntry] = {
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
}

_DISPATCHER_PROMPT = '''\
You are a RAG Dispatcher. For each slot, select exactly one inner-loop retrieval agent and produce a concise task description for it.

## Agent Roster

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
  Handles: impressions, opinions, commitments, facts when exact wording is unknown.

- `web_search_agent2`: Public internet search.
  Use ONLY when information cannot exist in local conversation history or persistent memory.

## Slot prefix → agent mapping (check this FIRST — overrides everything below)

Slots produced by the initializer start with a prefix that maps directly to an agent.
Match the prefix literally and use the mapped agent without further deliberation:

| Slot prefix                  | Agent                            |
|------------------------------|----------------------------------|
| "Identity: ..."              | `user_lookup_agent`              |
| "User-list: ..."             | `user_list_agent`                |
| "Relationship: ..."          | `relationship_agent`             |
| "Profile: ..."               | `user_profile_agent`             |
| "Conversation-aggregate: ..."| `conversation_aggregate_agent`   |
| "Conversation-filter: ..."   | `conversation_filter_agent`      |
| "Conversation-keyword: ..."  | `conversation_keyword_agent`     |
| "Conversation-semantic: ..." | `conversation_search_agent`      |
| "Memory-search: ..."         | `persistent_memory_search_agent` |
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

10. Slot requires public internet data?
   → `web_search_agent2`.

## Input

{{
    "current_slot": "the slot to resolve this turn",
    "known_facts": [{{"slot": "...", "agent": "...", "resolved": true/false, "summary": "concise fact summary", "attempts": 1}}, ...],
    "context": runtime info (platform, channel, timestamp, etc.)
}}

## Output

Return valid JSON only:
{{
    "agent_name": "{agent_name_union}",
    "task": "task description for the chosen agent",
    "context": {{}},
    "max_attempts": 3
}}
'''
_dispatcher_llm = get_llm(temperature=0.0, top_p=1.0)


def build_rag_fact_source_map() -> dict[str, dict[str, Any]]:
    """Build the deterministic fact-source map for RAG known_facts.

    The consolidator can combine this map with each known fact's ``agent``
    field to decide whether a fact came from internal memory/history/profile
    stores or from an external source.

    Returns:
        Dict keyed by agent name with source and consolidation policy metadata.
    """
    return {
        agent_name: dict(entry["fact_source"])
        for agent_name, entry in _RAG_SUPERVISOR_AGENT_REGISTRY.items()
    }


async def rag_dispatcher(state: ProgressiveRAGState) -> dict:
    """Generate one plain JSON dispatch targeting the first unknown slot.

    Args:
        state: Current state with unknown_slots and known_facts.

    Returns:
        Partial state update with dispatcher JSON recorded in state,
        current_slot set, and loop_count incremented.
    """
    current_slot = state["unknown_slots"][0]

    system_prompt = SystemMessage(
        content=_DISPATCHER_PROMPT.format(
            agent_name_union=_build_agent_name_union(),
        )
    )
    user_input = {
        "current_slot": current_slot,
        "known_facts": state.get("known_facts", []),
        "context": state.get("context", {}),
    }
    human_message = HumanMessage(content=json.dumps(user_input, ensure_ascii=False, default=str))

    # Only the last two messages — enough to preserve immediate loop history
    # without polluting the current slot with distant attempts.
    recent_messages = state["messages"][-2:] if len(state["messages"]) >= 2 else state["messages"]

    response = await _dispatcher_llm.ainvoke([system_prompt, human_message] + recent_messages)
    dispatch = parse_llm_json_output(str(response.content))
    if not isinstance(dispatch, dict):
        dispatch = {}

    return {
        "messages": [AIMessage(content=str(response.content))],
        "current_slot": current_slot,
        "current_dispatch": dispatch,
        "loop_count": state.get("loop_count", 0) + 1,
    }


# ── Executor ───────────────────────────────────────────────────────


def _build_delegate_context(state: ProgressiveRAGState, dispatch: dict) -> dict:
    """Merge runtime context, dispatcher hints, and known facts for one agent call."""
    delegate_context = dict(state.get("context", {}))
    raw_dispatch_context = dispatch.get("context", {})
    if isinstance(raw_dispatch_context, dict):
        delegate_context.update(raw_dispatch_context)
    delegate_context["known_facts"] = list(state.get("known_facts", []))
    delegate_context["original_query"] = state.get("original_query", "")
    delegate_context["current_slot"] = state.get("current_slot", "")
    return delegate_context



def _build_agent_name_union() -> str:
    """Render the dispatcher's allowed ``agent_name`` values as a union string."""
    return " | ".join(_RAG_SUPERVISOR_AGENT_REGISTRY)


def _normalize_dispatch(raw_dispatch: dict, current_slot: str) -> dict:
    """Normalize dispatcher JSON into a safe executable dispatch payload."""
    agent_name = str(raw_dispatch.get("agent_name", "")).strip()
    if agent_name not in _RAG_SUPERVISOR_AGENT_REGISTRY:
        agent_name = ""

    task = str(raw_dispatch.get("task", "")).strip() or current_slot
    context = raw_dispatch.get("context", {})
    if not isinstance(context, dict):
        context = {}

    raw_max_attempts = raw_dispatch.get("max_attempts", 3)
    if isinstance(raw_max_attempts, int) and not isinstance(raw_max_attempts, bool) and raw_max_attempts > 0:
        max_attempts = raw_max_attempts
    else:
        max_attempts = 3

    return {
        "agent_name": agent_name,
        "task": task,
        "context": context,
        "max_attempts": max_attempts,
    }


def _serialize_agent_result(result: dict) -> str:
    """Serialize one agent result for dispatcher history."""
    return json.dumps(result, ensure_ascii=False, default=str)

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
        return {
            "last_agent_result": result,
            "messages": [HumanMessage(content=_serialize_agent_result(result), name="agent_result")],
        }

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
        logger.exception("Error executing agent %s", agent_name)
        result = {
            "agent": agent_name,
            "resolved": False,
            "result": f"{type(exc).__name__}: {exc}",
            "attempts": 0,
        }

    return {
        "last_agent_result": result,
        "messages": [HumanMessage(content=_serialize_agent_result(result), name="agent_result")],
    }


# ── Evaluator ──────────────────────────────────────────────────────

_EVALUATOR_SUMMARIZER_PROMPT = '''\
你是一个槽位结果提炼器。给定槽位任务描述和原始工具结果，提炼出一段简洁的中文事实摘要，供后续检索代理和最终回答器使用。

# 摘要要求
- 保留对后续步骤有用的关键标识符（global_user_id、display_name、URL 等）
- 如果内容是对话记录，列出 1-5 条最相关的消息摘要（说话人 + 关键内容）
- 如果内容是用户画像或持久记忆，提炼关键事实
- 如果槽位未解决（resolved: false），简洁说明本次检索的来源没有返回什么
- 如果 raw_result 为空，不要推断先前槽位失败；只有 known_facts 明确显示先前槽位 unresolved 时才可这样说
- 不超过 200 字，纯文本，无 JSON 外壳
'''

_EVALUATOR_SUMMARIZER_USER_PROFILE_PROMPT = '''\
你是一个用户/角色画像槽位结果提炼器。给定 Profile 槽位、原始画像结果、以及先前身份解析结果，提炼一段简洁中文事实摘要。

# 字段语义（必须遵守）
- 如果 raw_result 包含 character_diary：这些是角色写在目标用户档案下的主观日记，作者/感受主体是角色，不是目标用户。
  - 可以写“角色日记显示，角色对该用户/这次互动感到……”
  - 禁止写成“该用户感到心虚/防线动摇/尴尬/不安”，除非日记原文明确说这是用户的感受。
- objective_facts：关于目标用户的客观事实。
- active_commitments：与目标用户相关、已被接受的承诺或持续规则。
- user_image：第三人称滚动画像，描述目标用户表现以及角色对目标用户的感知。
- 如果 raw_result 包含 name/description/gender/age/birthday/backstory/self_image，且不包含 character_diary/objective_facts/active_commitments/user_image：
  这是角色自己的公开资料或自我画像。按“角色自身资料”总结，不要当成第三方用户画像。

# 摘要要求
- 保留 global_user_id、display_name 等对后续步骤有用的标识。
- 明确区分“目标用户是谁”和“角色日记是谁的主观视角”。
- 只总结 raw_result 中已有的信息，不要补全未知信息。
- 不超过 220 字，纯文本，无 JSON 外壳。
'''

_evaluator_summarizer_llm = get_llm(temperature=0.0, top_p=1.0)

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

    system_prompt = SystemMessage(content=prompt)
    human_message = HumanMessage(
        content=json.dumps(
            {
                "slot": slot,
                "agent": agent_name,
                "resolved": resolved,
                "raw_result": raw_result,
                "known_facts": known_facts,
            },
            ensure_ascii=False,
            default=str,
        )
    )
    response = await _evaluator_summarizer_llm.ainvoke([system_prompt, human_message])
    return str(response.content).strip()


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

    return {
        "unknown_slots": remaining_slots,
        "known_facts": known_facts + [new_fact],
    }


# ── Finalizer ──────────────────────────────────────────────────────

_FINALIZER_PROMPT = '''\
你是一个总结员。请根据已收集的事实回答用户的原始问题。

# 准则
- 直接回答用户的原始问题，不要复述查找过程。
- 如果 known_facts 为空，说明本次 RAG 没有需要检索的外部/内部事实；不要说“缺少关于该问题的具体信息”。
- 如果某个槽位未能解决（resolved: false），如实告知缺少哪一部分信息。
- 不要把某个来源没有检索结果扩大成“没有任何记录/没有互动记录”；只能说明实际查询过的来源没有返回结果。
- 引用来源 URL 或对话来源时尽量保留。
- 当 known_facts 中 agent="user_profile_agent" 且 raw_result 包含 character_diary：
  character_diary 是角色写在目标用户档案下的主观日记，作者/感受主体是角色，不是目标用户。
  禁止把日记里的“心虚、防线动摇、尴尬、不安”等感受归因给目标用户，除非原文明确说这是目标用户的感受。
- 当 known_facts 中 agent="user_profile_agent" 且 raw_result 是角色自身公开资料或 self_image：
  这是角色自己的资料。回答角色自我资料问题时，可以使用这些公开资料；不要误写成第三方用户画像。

# 输入格式
{
    "original_query": "用户原始问题",
    "known_facts": [{"slot": ..., "agent": ..., "resolved": ..., "summary": "简洁事实摘要", "raw_result": "原始工具输出（如需引用原文）", "attempts": ...}, ...]
}

# 输出
请直接返回一段自然语言回复（纯文本，无 JSON 外壳）。
- no markdown formatting
- no persona voice
- no final answer
- no broad interpretation beyond short extractive summaries
'''
_finalizer_llm = get_llm(temperature=0.2, top_p=0.9)


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
        "known_facts": state.get("known_facts", []),
    }
    human_message = HumanMessage(content=json.dumps(finalizer_input, ensure_ascii=False, default=str))

    response = await _finalizer_llm.ainvoke([system_prompt, human_message])
    return {"final_answer": response.content}


# ── Routing ────────────────────────────────────────────────────────

def _route_after_initializer(state: ProgressiveRAGState) -> str:
    """Skip to finalizer if the initializer produced no slots."""
    return "dispatch" if state.get("unknown_slots") else "finalize"


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
        character_name: Display name of the character (e.g. "千纱"). Used by
            the initializer to distinguish addressee references from subject
            references so it does not generate spurious slots for the character.
        context: Optional auxiliary fields (platform/channel/target user UUID).

    Returns:
        Dict with keys ``answer``, ``known_facts``, ``unknown_slots``
        (any that were not drained), and ``loop_count``.
    """
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
        "context": context or {},
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

    logger.info(f"RAG Query: {original_query}")
    result = await graph.ainvoke(initial_state)

    return {
        "answer": result.get("final_answer", ""),
        "known_facts": result.get("known_facts", []),
        "unknown_slots": result.get("unknown_slots", []),
        "loop_count": result.get("loop_count", 0),
    }


async def test_main():
    """Simple debug entry-point."""
    try:
        await mcp_manager.start()
    except Exception:
        logger.exception("MCP manager failed to start — web tools will be unavailable")

    # Dummy GlobalPersonaState-equivalent fields
    character_profile = {
        "name": "千纱",
        "description": "一个温柔的AI角色",
    }
    user_profile = {
        "affinity": 800,
        "display_name": "小钳子",
    }

    result = await call_rag_supervisor(
        original_query="千纱千纱欢迎回来",
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
