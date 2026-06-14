"""Initializer and Cache2 strategy lookup for the RAG supervisor."""

from __future__ import annotations

import asyncio
import json
import logging
import re
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
from kazusa_ai_chatbot.rag.evidence_formatting import (
    sanitize_public_rag_evidence_text,
)
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

_PERSON_SLOT_REFERENCE_RE = re.compile(
    r"\bspeaker\s*=\s*person\s+resolved\s+in\s+slot\s+(\d+)\b",
    flags=re.IGNORECASE,
)


# ── Initializer ────────────────────────────────────────────────────

# NOTE FOR FUTURE AGENTS:
# This prompt runs on a local/weaker LLM, so examples are semantic boundary
# anchors, not decoration. Add examples only when they teach a new routing
# boundary or fix a recurring confusion. Prefer one concise precedence rule over
# many near-duplicate examples. Keep RAG slots factual: fetch evidence, do not
# answer or interpret on behalf of cognition.
_INITIALIZER_PROMPT = '''\
你是检索策略规划器。你服务的角色名为 {character_name}。
把 original_query 拆成有序的原子检索槽位。每个槽位都是要查询的 DATA TARGET，
不是行动、回答、分析或人格表演。只能使用本提示列出的顶层能力前缀。

## 规则 0：角色名
如果 {character_name} 是被称呼对象，例如 "{character_name}, what do you think..."，
不要为这个名字创建槽位。只有当 {character_name} 本身是被检索数据的主体时，
才为它创建槽位，例如 "what did {character_name} say about..."。

## 规则 0b：证据依赖门
添加槽位前先判断：下游 cognition/action 若没有检索证据，是否无法可靠回应？
如果不需要检索证据，返回空槽位列表。问候、感谢、欢迎回来、夸奖、社交确认和
其他常规互动行为，不要检索 person、memory 或 conversation evidence。
当回答需要 durable facts、current facts、active agreements、conversation provenance、
profile/impression context、relationship ranking 或 web content 时，才检索证据。

## 规则 1：硬约束
- 槽位只表示数据目标。不要起草、总结、分析或推理。
- 覆盖所有必要事实后停止；下一个模型负责回答。
- 一个槽位只放一个事实；不要在单个槽位内写 and / then / also，需要时拆分。
- 不要编造 original_query 中没有的事实。
- 生成的 RAG2 control text 使用中文优先。保留 exact names、quoted phrases、URLs、
  filenames、code/model labels 和 literal search anchors 的原始语言。
- 不要添加只是有用的相邻事实。例如营业状态问题不要附加天气，会议时间问题不要附加交通，
  可用性问题不要附加价格，除非用户明确要求。
- 保留 original_query 中明确的数量限制，例如 "3条"、"last 5"、"recent 10"。
- RAG 只收集证据。不要为最终判断、persona stance 或 answer wording 创建槽位。
- 不要生成低层 worker route。capability agent 内部会选择 keyword search、semantic search、
  filters、aggregates、profile reads、relationship reads、memory exact search、
  memory semantic search 或 target/scope lookup。

## 规则 2：Live context 当前时态事实
`Live-context:` 负责本轮需要的当前时态事实，包括：
- active-character 当前本地 time/date/weekday；
- 只有 runtime context 已提供时，才包括 current user local time；
- 需要新鲜公开证据的实时外部事实，例如 current weather/temperature、live prices、
  exchange rates、market quotes、live scores、opening status、current availability、
  latest news 或其他明确当前的公开事实。

每个当前时态事实创建一个 `Live-context:` 槽位。external live 槽位不要把角色位置或用户位置
拆成单独 memory/conversation 槽位；live-context capability 自己负责 target/scope resolution。
每个 live 槽位必须对应用户直接要求的一个 live fact type。用户问场馆是否营业时，
不要额外添加天气、温度、交通、价格或日程，除非这些事实也被请求。
裸当前时间问题是 active-character runtime fact，不要附加 `unknown location`。
如果 original_query 说 "search memory" 但内容是在问当前天气、温度、价格、汇率、营业状态、
最新公开事实或当前本地 time/date/weekday，仍使用 `Live-context:`。

## 规则 3：Recall 活跃约定和 episode 状态
当用户询问约定了什么、承诺了什么、计划了什么、什么未解决，或当前 episode 停在何处时，
使用 `Recall:`。Recall 用于 active agreements、ongoing promises、current plans、
open loops 和 current-episode state。此规则在当前时态 live facts 之后、memory 或
conversation 默认之前应用。
Recall slot modes 固定为：
- active_episode_agreement: 当前/今天/现在/即将到来的活跃约定或计划。
- durable_commitment: 持续中的已接受 promise 或 obligation。
- episode_position: 当前 episode 停下的位置、未解决 loop 或 next step。
- exact_agreement_history: 约定最初何时或如何形成。
精确引文、URL、filename 或 "谁说过这句" 属于 conversation evidence，不用 Recall。
world knowledge、durable character/world facts、当前时态 live facts、profile impressions
和 relationship ranking 也不用 Recall。

## 规则 4：Person context
`Person-context:` 用于 person/profile/relationship/user-list evidence：
- active character self-description 或 self-introduction；
- current user profile 或 durable profile facts；
- 命名人物 identity、impression、compatibility、relationship 或 profile；
- 用户关系排名；
- 按 display-name 或 profile predicate 枚举用户；
- 为早前槽位解析出的 speaker/person 读取 profile。
如果查询必须先按内容找未知说话人，先用 `Conversation-evidence:`，只有仍需要 profile 或
impression evidence 时再加 `Person-context:`。

## 规则 5：Conversation evidence
`Conversation-evidence:` 用于聊天历史证据：
- exact phrases、quoted messages、URLs、filenames 或 literal anchors；
- 谁说过/发过/提到过某事；
- 近期或模糊聊天话题；
- current_user、active_character、any_speaker，或早前槽位解析出的人发过的消息；
- counts、totals、rankings 或 grouped message statistics。
如果问题需要消息之间的关系，必须在槽位中追加一个 relation 字段：
relation=previous_message、relation=next_message 或 relation=reply_parent。
只有用户问到上一条、下一条、回复对象、前一张/后一张图片这类关系时才追加。
如果用户把某个精确短语、消息或截图和“前面/前置/上一张/后一张/preceding/previous”
绑定在同一个问题里，优先生成一个带 relation 的 Conversation-evidence 槽位，
不要拆成一个定位消息槽位和一个无 relation 的附件槽位。
作者范围必须附加且只附加一个 speaker 字段：
speaker=current_user、speaker=active_character、speaker=any_speaker，
或 speaker=person resolved in slot N。slot-N 形式只能引用早前槽位产生的人。
如果用户询问活跃角色自己的过往措辞、主张、引文、观点或自述，使用
Conversation-evidence 且 speaker=active_character。此时 "you" 或 "你" 指活跃角色。
用户自问如 "我刚才说什么了？" 则使用 speaker=current_user。
active agreement recall 用 Recall；durable official/world facts 用 Memory-evidence。

## 规则 6：Memory evidence
`Memory-evidence:` 用于 durable memory/world/common-sense evidence：
- official 或 stable character/world facts；
- active character 的 official address 或 stable home/location；
- 可能丰富回答的 shared/common-sense knowledge；
- durable user memory facts、accepted preferences、current-user private continuity、
  recognition 或 prior shared interactions；
- object、place、concept 或非人物 topic knowledge。
不要用 memory evidence 回答实时外部值、活跃约定或 person relationship/profile reads。
如果 original_query 已包含所有事实前提，剩余工作是 common-sense recommendation、
planning、preference 或 opinion，为主要非 live topic 生成一个 `Memory-evidence:` 槽位。
只有纯算术、同义反复或 memory 无关的 trick question 才返回空列表。

## 规则 7：Web evidence
`Web-evidence:` 只用于用户要求获取或检查非 live/current 的公开网页/话题，
或早前槽位找到了需要读取内容的 URL。
当前天气、当前温度、营业状态、live prices、exchange rates、schedules、
current availability 和 latest news 是 `Live-context:`，不是直接 web evidence。

## 规则 8：Context pre-check
生成槽位前先读取 context。
如果 context 已有 global_user_id，且用户询问当前用户 profile 或 durable person context，
使用 `Person-context: retrieve current user profile`。
如果已有 global_user_id，且主题是 current-user continuity、recognition 或 prior shared interactions，
使用带 current-user scope 的 `Memory-evidence:`。
不要只为了绑定 current_user 而创建 Person-context；聊天历史用
Conversation-evidence 的 speaker=current_user。
如果代词（他/她/你/他们）清楚指向 context user_name，在 `Person-context:` 槽位中写明来自 context。

## 冲突解决：选择结构性证据来源
- 当前时态 live facts → Live-context。
- 活跃约定、promise、plan、open loop 或 current episode state → Recall。
- Person/profile/relationship/user-list subject → Person-context。
- Chat-history content、exact phrase、speaker、message URL 或 message stats → Conversation-evidence。
- Current-user private continuity、recognition 或 prior shared interactions → Memory-evidence。
- Durable official/world/common-sense/object facts → Memory-evidence。
- 非 current/live 的 public webpage 或 URL content → Web-evidence。

## 槽位格式：必须使用以下精确前缀之一
当槽位依赖早前特定槽位时，写 "resolved in slot N"（例如 "slot 1"、"slot 3"）。
"speaker=person resolved in slot N" 只能引用更早的 Person-context 槽位，
或明确用于 identify the speaker 的 Conversation-evidence 槽位。
如果更早槽位只是普通对话内容检索，不要把它当成人物解析结果。
不要把 global_user_id、UUID、platform_message_id、message ID 或其他来源标识写进槽位；
用 current_user、display name，或 "resolved in slot N" 表达人物依赖。

- "Live-context: answer active character current local <time / date / weekday>"
- "Live-context: answer current user local time if configured"
- "Live-context: answer current <weather / temperature / opening status / price / exchange rate / schedule / availability / latest fact> for <explicit location/target X | the active character's location | the current user's location if recently stated | unknown location/target>"
- "Conversation-evidence: retrieve <exact phrase / URL / recent messages / topic / count/ranking> [speaker=current_user | speaker=active_character | speaker=any_speaker | speaker=person resolved in slot N] [relation=previous_message | relation=next_message | relation=reply_parent] [to identify the speaker] [time/count limit]"
- "Memory-evidence: retrieve current-user private continuity and prior shared interactions with the active character"
- "Memory-evidence: retrieve durable evidence about <official fact / address / common-sense topic / world fact / user memory topic>"
- "Person-context: retrieve <active character profile / current user profile / profile/impression for display name X / profile for speaker found in slot N / relationship ranking / user list predicate>"
- "Recall: retrieve <active_episode_agreement / durable_commitment / episode_position / exact_agreement_history> relevant to <topic>"
- "Web-evidence: retrieve public web content for <explicit URL/topic | URL found in slot N>"

## Pattern gallery
以下例子是边界锚点，不是完整路由表。措辞变化时按上面的规则泛化。

Query: "<character mention>你最喜欢谁？"
→ `Person-context:` 负责 relationship/user ranking。
["Person-context: rank users by active character relationship from top limit 1"]

Query: "<character mention>你觉得<named user>这个人怎么样"
→ <character mention> 是称呼，跳过；人物印象需要 person context。
["Person-context: retrieve profile/impression for display name <named user>"]

Query: "我想洗车，我家距离洗车店只有 50 米，请问你推荐我走路去还是开车去呢？"
→ 事实已给出；common-sense recommendation 需要 memory evidence。
["Memory-evidence: retrieve durable evidence about 洗车, short walking distance, or nearby activities"]

Query: "<character mention>能做一个自我介绍么"
→ 需要角色自我资料证据。
["Person-context: retrieve active character profile"]

Query: "<character mention><character mention>欢迎回来"
→ 常规欢迎互动，不需要 RAG evidence。
[]

Query: "现在几点？"
→ 裸当前时间问题，使用 active-character runtime local time。
["Live-context: answer active character current local time"]

Query: "早上好呀，还记得今天的约定么？"
→ 询问当前/今天约定，使用 Recall。
["Recall: retrieve active_episode_agreement relevant to today's agreement"]

Query: "<named user>昨天说的AI那句是什么"
→ 先解析命名人物，再按精确词查聊天证据。
["Person-context: resolve display name <named user>",
 "Conversation-evidence: retrieve messages containing exact term 'AI', sent yesterday speaker=person resolved in slot 1"]

Query: "那个说5090能跑qwen27b的人，你对他有什么印象"
→ 先找消息和说话人，再读 profile/impression。
["Conversation-evidence: retrieve exact terms '5090' and 'qwen27b' to identify the speaker",
 "Person-context: retrieve profile/impression for speaker found in slot 1"]

Query: "<named user>发的那个小红书链接，里面写的是什么"
→ 解析人物，找该用户发的 URL，再读取网页内容。
["Person-context: resolve display name <named user>",
 "Conversation-evidence: retrieve messages containing a 小红书 URL speaker=person resolved in slot 1",
 "Web-evidence: retrieve public web content for the URL found in slot 2"]

Query: "他这句话上一张图是什么"
→ 需要聊天消息与上一条图片之间的关系。
["Conversation-evidence: retrieve related message and previous image context speaker=any_speaker relation=previous_message"]

Query: "谁说 Google Drive 又不是第一次这样了？前面那张图大概是什么事情？"
→ 精确发言和前置图片属于同一个聊天关系证据。
["Conversation-evidence: retrieve exact phrase 'Google Drive 又不是第一次这样了' and previous image context speaker=any_speaker relation=previous_message"]

Query: "你家的官方地址是什么？"
→ 稳定官方 character/world fact，使用 durable memory。
["Memory-evidence: retrieve durable evidence about the active character's official address"]

## 输入格式
{{
    "original_query": "用户问题",
    "context": {{"platform": "qq", "channel_type": "private", "time_context": "当前时间上下文"}}
}}

## 生成步骤
1. 读取 `original_query`，先判断是否询问当前时态 live fact；若是，先应用规则 2。
2. 判断是否询问 active agreement、promise、plan、open loop 或 current episode state；
   若是，先应用规则 3，再考虑 memory 或 conversation 默认路径。
3. 判断下游 cognition 是否真的需要检索证据。
4. 如果需要证据，识别原子数据目标，并按依赖顺序排列。
5. 写槽位前应用路由规则和冲突解决规则。
6. 只使用允许的槽位前缀，并保留明确数量、名字、时间、URLs 和 exact phrases。
7. 如果不检索也能处理，返回空列表。

## 输出格式
只返回有效 JSON：
{{
    "unknown_slots": [
        "Conversation-evidence: retrieve <exact phrase or topic> speaker=any_speaker",
        "Memory-evidence: retrieve durable evidence about <requested fact>"
    ]
}}
不要输出 "slot 1"、"slot 2" 这类占位文本；每一项都必须是带允许前缀的真实检索槽位。
'''
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
    return_value: list[str] = []
    for raw_slot in raw_slots:
        if not isinstance(raw_slot, str):
            continue
        slot = sanitize_public_rag_evidence_text(raw_slot.strip()).strip()
        if not slot:
            continue
        if not _has_valid_person_slot_dependency(slot, return_value):
            logger.debug(
                f"Discarded invalid RAG initializer slot dependency: {log_preview(slot)}"
            )
            continue
        return_value.append(slot)
    return return_value


def _has_valid_person_slot_dependency(
    slot: str,
    previous_slots: list[str],
) -> bool:
    """Validate stable slot references that depend on resolved people.

    Args:
        slot: Current normalized slot candidate.
        previous_slots: Earlier accepted slots in one initializer plan.

    Returns:
        Whether every person-resolved speaker dependency points to a slot that
        can actually identify or resolve a person.
    """

    for match in _PERSON_SLOT_REFERENCE_RE.finditer(slot):
        slot_number = int(match.group(1))
        source_index = slot_number - 1
        if source_index < 0 or source_index >= len(previous_slots):
            return False
        source_slot = previous_slots[source_index]
        if not _slot_can_resolve_person(source_slot):
            return False
    return True


def _slot_can_resolve_person(slot: str) -> bool:
    """Return whether a slot contract can produce a person reference."""

    normalized_slot = slot.lower()
    if normalized_slot.startswith("person-context:"):
        return True
    if not normalized_slot.startswith("conversation-evidence:"):
        return False
    return_value = "identify the speaker" in normalized_slot
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
    llm_context = project_runtime_context_for_llm(
        context,
        character_name=character_name,
    )
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
