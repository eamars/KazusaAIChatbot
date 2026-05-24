"""Initializer and Cache2 strategy lookup for the RAG supervisor."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_PLANNER_LLM_API_KEY,
    RAG_PLANNER_LLM_BASE_URL,
    RAG_PLANNER_LLM_MODEL,
)
from kazusa_ai_chatbot.db.rag_cache2_persistent import (
    record_initializer_hit,
    upsert_initializer_entry,
)
from kazusa_ai_chatbot.rag.cache2_policy import (
    INITIALIZER_AGENT_REGISTRY_VERSION,
    INITIALIZER_CACHE_NAME,
    INITIALIZER_PROMPT_VERSION,
    INITIALIZER_STRATEGY_SCHEMA_VERSION,
    build_initializer_cache_key,
)
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.rag.prompt_projection import project_runtime_context_for_llm
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_types import (
    ProgressiveRAGState,
)
from kazusa_ai_chatbot.utils import (
    get_llm,
    log_list_preview,
    log_preview,
    parse_llm_json_output,
)

logger = logging.getLogger(__name__)


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
- Write generated RAG2 control text in English. Preserve exact names, quoted
  phrases, URLs, filenames, code/model labels, and literal search anchors in
  their original source language.
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

If the user asks about the active character's own prior wording, claims,
quotes, opinions, or self-authored statements, use Conversation-evidence with
speaker=active_character. In these self-word requests, "you" or "你" means the
active character even when the character name is not present. Do not use this
route for user-self questions such as "我刚才说什么了？" or "What did I say
earlier?"; those ask about the current user's words.

Do not use conversation evidence for active agreement recall; use Recall.
Do not use conversation evidence for durable official/world facts; use
Memory-evidence.

## Rule 6 — Memory evidence
Use `Memory-evidence:` for durable memory/world/common-sense evidence:
- official or stable character/world facts,
- the active character's official address or stable home/location,
- shared/common-sense knowledge that may enrich an answer,
- durable user memory facts, accepted preferences, current-user private
  continuity, recognition, or prior shared interactions,
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
If global_user_id is present and the subject is current-user continuity,
recognition, or prior shared interactions, use
`Memory-evidence:` with current-user scoped wording.
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
- Current-user private continuity, recognition, or prior shared interactions → Memory-evidence.
- Durable official/world/common-sense/object facts → Memory-evidence.
- Public webpage or URL content that is not current/live → Web-evidence.

## Slot format — ALWAYS use one of these exact prefixes
When a slot depends on a specific earlier slot, write "resolved in slot N" (e.g. "slot 1", "slot 3").

- "Live-context: answer active character current local <time / date / weekday>"
- "Live-context: answer current user local time if configured"
- "Live-context: answer current <weather / temperature / opening status / price / exchange rate / schedule / availability / latest fact> for <explicit location/target X | the active character's location | the current user's location if recently stated | unknown location/target>"
- "Conversation-evidence: retrieve <exact phrase / URL / recent messages / topic / count/ranking> [speaker=current_user | speaker=active_character | speaker=any_speaker | speaker=person resolved in slot N] [to identify the speaker] [time/count limit]"
- "Memory-evidence: retrieve current-user private continuity and prior shared interactions with the active character"
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

### 8b. Active character self-word retrieval
Query: "你之前是不是说过那个项目要延期？"
  → The user asks for the active character's own prior claim. Use active-character conversation evidence.
  ["Conversation-evidence: retrieve prior active-character claim about the project being delayed speaker=active_character"]

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
