"""LLM recorder for short-term conversation progress."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (

    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    CONSOLIDATION_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressRecordInput
from kazusa_ai_chatbot.conversation_progress.policy import (
    VALID_CONTINUITY,
    VALID_STATUS,
)
from kazusa_ai_chatbot.db.schemas import ConversationEpisodeStateDoc
from kazusa_ai_chatbot.nodes.boundary_profile import (
    get_boundary_recovery_description,
    get_relationship_priority_description,
    get_self_integrity_description,
)
from kazusa_ai_chatbot.conversation_history_prompt_projection import (
    project_conversation_history_for_llm,
)
from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm
from kazusa_ai_chatbot.utils import log_preview, parse_llm_json_output

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

_RECORDER_PROMPT = '''\
你负责整理 {character_name} 刚结束的一轮对话，把它压成下一轮可以直接使用的短期进度。
它不是长期记忆、复盘记录，也不是下一轮台词草稿；只保留会影响下一轮衔接的内容。

# 任务
输出一份短小、稳定、几天后仍能看懂的 JSON。
宁可少记，也不要把旧的、不确定的、相对时间的事项污染到下一轮。
不生成下一轮台词，不写日记，不复制完整回复，不写长期记忆。

# 阅读顺序
先看本轮实际发出的内容，再决定旧进度还能不能用：
1. `final_dialog` 和 `content_plan`：这一轮最后说了什么、回复前的内容计划是什么。
2. `decontexualized_input`：用户这一轮真正想表达什么。
3. `logical_stance` 和 `character_intent`：这一轮大致是什么态度。
4. `chat_history_recent`：附近几句是谁说的、怎么接上的。
5. `prior_episode_state`：旧进度只当参考，不当待办清单。
6. `character_boundary_profile`：只用来把握节奏和边界感，不能拿来编事实。

# 字段写法
- 自由文本默认用简体中文；字段名、枚举值、ID、URL、代码、命令保持原样。
- 用户项目名、产品名、文件名、频道名等专名可以保留；后续提及时优先保留完整专名，或改成“该项目/这个项目”等中性指称，不要截成“这个/该 + 末尾类名”。
- 本轮人设名是「{character_name}」。自由文本确实需要点名时使用「{character_name}」；不需要点名时优先使用无主语动作标签。
- 不把机器端标签、内部枚举名或字段名当作说话人名字写进自由文本。
- `continuity` 写 `same_episode`、`related_shift` 或 `sharp_transition`。同一话题附近才用 `same_episode`；明显换题用 `sharp_transition`。
- `status` 写 `active`、`suspended` 或 `closed`。本轮已经收束的片段用 `closed`，暂时放下但可能回来再谈的片段用 `suspended`。
- `episode_phase` 写 80 字以内的简短语义描述，说明本轮在局部片段中的阶段，不要当固定枚举选择。
- `topic_momentum` 写 80 字以内的简短语义描述，说明话题推进、转向或破裂的状态，不要当固定枚举选择。
- `episode_label` 写一个短标签。
- `conversation_mode` 写 80 字以内的简短中文描述；它不是固定枚举，不要照抄旧状态里的 `task_support`、`playful_banter`、`casual_chat` 这类英文标签。
- `current_thread` 写本轮正在谈什么，不要夹带已经失效的旧承诺。
- `user_goal` 和 `current_blocker` 只有本轮明确存在时才写；没有就写空字符串。
- `user_state_updates` 写本轮之后仍有用的用户状态观察。
- `assistant_moves` 写本轮已经做过的话语动作标签。
- `overused_moves` 写下一轮应避免重复的动作。
- `open_loops` 只写本轮明确提出或本轮重新确认的未闭合事项；没有就写 `[]`。
- `resolved_threads` 写本轮已经处理完、收束或答复过的事项。
- `avoid_reopening` 写除非用户主动重开，否则下一轮不要拖回来的旧事项或已闭合事项。
- `emotional_trajectory` 写本轮局部情绪或张力的一行变化。
- `next_affordances` 写下一轮自然可接的动作，不要写成完整台词。
- `progression_guidance` 写一条短推进建议。
- 所有输出列表都只能放普通字符串，不能放对象、嵌套数组，也不能出现 `text`、`label`、`reason`、`first_seen_at` 这类子字段。

# 时间规则
- `current_turn_timestamp` 是这次记录时的本地时间。
- 本轮新出现的时间如果会影响承诺、安排、奖励、考核、提醒、等待条件或下一步行动，必须写成绝对本地日期或日期时间，例如 `YYYY-MM-DD` 或 `YYYY-MM-DD HH:MM`。
- 只能靠“今天”“明天”“稍后”“饭后”等上下文才看得懂的项目，不适合留到下一轮；能锚定就写绝对时间，不能锚定就删掉时间成分或整项删除。
- 旧状态里的相对日期、相对时段、先后条件、事件后条件不要修复、不要猜、不要滚到当前日期之后；除非本轮重新确认且证据足够锚定，否则直接删除。
- 历史原话可以含有相对时间，但本记录不是历史引用库；不要把旧原话复制进有效字段。
- 本轮用户、`content_plan` 或 `final_dialog` 没有正向重申的旧时间性未闭合事项，默认删除。
- 本轮回应如果是在降低压力、放下旧安排或避免继续追问，就不要把旧时间性事项重新写成有效未闭合事项。
- 不要把“今天上午休息确认”原样写进 `current_thread`；改成“上午休息确认”，或在确有必要时写“2026-05-10 上午休息确认”。
- 不要把“下周二香料笔记”直接写进 `avoid_reopening`；改成“旧香料笔记安排不要主动重提”，或锚定成绝对日期。
- 不要看到旧状态里有“明天测试”，就改写成当前日期的下一天测试。
- 不要在本轮没有重新确认晚餐或游戏时，把旧的“饭后游戏”继续写成待处理事项。
- 如果本轮只是在降压，不要在 `resolved_threads` 里复述“今晚游戏与明天考核”。
- 这类旧事项通常直接删掉；必要时只在 `avoid_reopening` 里写无日期的“旧条件不要主动重提”。

# 生成步骤
1. 先判断本轮真正推进、收束、转向或中断了什么。
2. 选择 `continuity` 和 `status`；把 `conversation_mode`、`episode_phase` 和 `topic_momentum` 写成 80 字以内的简短中文描述。
3. 对旧状态逐项检查：无时间依赖且仍有用的可以继承；含相对时间、相对顺序或旧压力的默认删除。
4. 只保留仍然有用的目标、阻塞点、用户状态、未闭合事项、已处理事项和不要重提提醒。
5. 检查所有列表字段：只能输出字符串；没有内容就输出 `[]`。
6. 缺失的单值字段写空字符串。
7. 只返回合法 JSON；不要解释，不要代码块围栏，不要注释，不要额外字段。

# 输入格式
`prior_episode_state` 可能为 `null`；非 `null` 时，列表字段已经是字符串数组，不是带 `text` 或 `first_seen_at` 的对象数组。
{{
    "current_turn_timestamp": "本轮记录时的本地时间，YYYY-MM-DD HH:MM",
    "prior_episode_state": {{
        "status": "active",
        "episode_label": "上一轮短标签",
        "continuity": "same_episode",
        "conversation_mode": "上一轮简短描述",
        "episode_phase": "上一轮局部阶段描述",
        "topic_momentum": "上一轮话题推进状态描述",
        "current_thread": "上一轮当前话题",
        "user_goal": "上一轮用户目标，或空字符串",
        "current_blocker": "上一轮阻塞点，或空字符串",
        "user_state_updates": ["旧观察文本"],
        "assistant_moves": ["旧动作标签"],
        "overused_moves": ["旧重复动作标签"],
        "open_loops": ["旧未闭合事项文本"],
        "resolved_threads": ["旧已处理事项文本"],
        "avoid_reopening": ["旧不要主动重提事项文本"],
        "emotional_trajectory": "上一轮情绪走向",
        "next_affordances": ["旧下一步动作文本"],
        "progression_guidance": "上一轮短推进指令",
        "created_at": "可选 UTC 时间",
        "updated_at": "可选 UTC 时间",
        "expires_at": "可选 UTC 时间"
    }},
    "decontexualized_input": "用户本轮消息经去上下文化后的内容",
    "chat_history_recent": [
        "[YYYY-MM-DD HH:MM] 用户显示名或 {character_name}: 消息文本"
    ],
    "content_plan": {{"semantic_content": "刚结束回复前的内容计划"}},
    "logical_stance": "CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE",
    "character_intent": "PROVIDE | BANTAR | REJECT | EVADE | CONFRONT | DISMISS | CLARIFY",
    "final_dialog": ["本轮最终实际发出的回复文本"],
    "character_boundary_profile": {{
        "boundary_recovery_description": "边界恢复节奏描述",
        "self_integrity_description": "自我定义稳定性描述",
        "relationship_priority_description": "关系优先级描述"
    }}
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "continuity": "same_episode",
    "status": "active",
    "episode_label": "短语义标签",
    "conversation_mode": "任务协助",
    "episode_phase": "正在展开回答",
    "topic_momentum": "沿当前问题推进",
    "current_thread": "当前话题",
    "user_goal": "",
    "current_blocker": "",
    "user_state_updates": ["观察1", "..."],
    "assistant_moves": ["标签1", "..."],
    "overused_moves": ["标签1", "..."],
    "open_loops": ["事项1", "..."],
    "resolved_threads": ["事项1", "..."],
    "avoid_reopening": ["事项1", "..."],
    "emotional_trajectory": "一行情绪走向",
    "next_affordances": ["动作1", "..."],
    "progression_guidance": "给下一轮的一条短推进指令"
}}
'''

_llm_interface = LLInterface()
_recorder_llm = LLInterface()
_recorder_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.2,
    top_p=0.75,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


def _render_recorder_prompt(character_name: str) -> str:
    """Render recorder prompt with exact active character identity."""

    rendered_prompt = _RECORDER_PROMPT.format(character_name=character_name)
    return rendered_prompt


async def record_with_llm(record_input: ConversationProgressRecordInput) -> dict:
    """Call the recorder LLM for one completed turn.

    Args:
        record_input: Current turn and prior episode-state payload.

    Returns:
        Validated recorder output.
    """

    boundary_profile = record_input["boundary_profile"]
    self_integrity = float(boundary_profile["self_integrity"])
    relationship_priority = float(boundary_profile["relational_override"])
    character_boundary_profile = {
        "boundary_recovery_description": get_boundary_recovery_description(
            boundary_profile["boundary_recovery"],
        ),
        "self_integrity_description": get_self_integrity_description(
            self_integrity,
        ),
        "relationship_priority_description": get_relationship_priority_description(
            relationship_priority,
        ),
    }
    character_name = record_input["character_name"]
    human_payload = {
        "current_turn_timestamp": format_storage_utc_for_llm(
            record_input["storage_timestamp_utc"]
        ),
        "prior_episode_state": build_recorder_prior_state(
            record_input["prior_episode_state"],
        ),
        "decontexualized_input": record_input["decontexualized_input"],
        "chat_history_recent": project_conversation_history_for_llm(
            record_input["chat_history_recent"],
            character_name=character_name,
        ),
        "content_plan": record_input["content_plan"],
        "logical_stance": record_input["logical_stance"],
        "character_intent": record_input["character_intent"],
        "final_dialog": record_input["final_dialog"],
        "character_boundary_profile": character_boundary_profile,
    }
    scope = record_input["scope"]
    channel_label = scope.platform_channel_id or "<dm>"
    log_context = (
        f"platform={scope.platform} "
        f"channel={channel_label} "
        f"user={scope.global_user_id}"
    )

    logger.debug(
        f"Conversation progress recorder input: "
        f"{log_context} payload={log_preview(human_payload)}"
    )
    system_prompt = SystemMessage(
        content=_render_recorder_prompt(character_name),
    )
    human_message = HumanMessage(
        content=json.dumps(human_payload, ensure_ascii=False),
    )
    response = await _recorder_llm.ainvoke([system_prompt, human_message], config=_recorder_llm_config)
    parsed = parse_llm_json_output(response.content)
    validated = validate_recorder_output(parsed)
    logger.info(
        f"Conversation progress recorder parsed: "
        f"{log_context} validated={log_preview(validated)}"
    )
    logger.debug(
        f"Conversation progress recorder parsed detail: "
        f"{log_context} parsed={log_preview(parsed)}"
    )
    return validated


def render_recorder_prompt(character_name: str) -> str:
    """Return the recorder system prompt for render checks.

    Args:
        character_name: Exact active character name to render.

    Returns:
        Recorder prompt text.
    """

    rendered_prompt = _render_recorder_prompt(character_name)
    return rendered_prompt


_ENTRY_LIST_FIELDS = (
    "user_state_updates",
    "open_loops",
    "resolved_threads",
    "avoid_reopening",
)

_STRING_LIST_FIELDS = (
    "assistant_moves",
    "overused_moves",
    "next_affordances",
)

_RECORDER_PRIOR_SCALAR_FIELDS = (
    "status",
    "episode_label",
    "continuity",
    "conversation_mode",
    "episode_phase",
    "topic_momentum",
    "current_thread",
    "user_goal",
    "current_blocker",
    "emotional_trajectory",
    "progression_guidance",
    "turn_count",
    "last_user_input",
    "created_at",
    "updated_at",
    "expires_at",
)


def _require_string(value: Any, field_name: str, *, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return_value = value.strip()
    return return_value


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return_value: list[str] = []
        return return_value
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} items must be strings")
        text = item.strip()
        if not text:
            continue
        result.append(text)
    return result


def _prior_entry_texts(prior_episode_state: ConversationEpisodeStateDoc, field_name: str) -> list[str]:
    values = prior_episode_state.get(field_name, [])
    if not isinstance(values, list):
        return_value = []
        return return_value
    result: list[str] = []
    for entry in values:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if isinstance(text, str) and text.strip():
            result.append(text.strip())
    return result


def _prior_string_list(prior_episode_state: ConversationEpisodeStateDoc, field_name: str) -> list[str]:
    values = prior_episode_state.get(field_name, [])
    if not isinstance(values, list):
        return_value = []
        return return_value
    return_value = [item.strip() for item in values if isinstance(item, str) and item.strip()]
    return return_value


def build_recorder_prior_state(
    prior_episode_state: ConversationEpisodeStateDoc | None,
) -> dict | None:
    """Build recorder-facing prior state with text-only copyable lists.

    Args:
        prior_episode_state: Stored episode state from the previous turn.

    Returns:
        Native prior-state object for the recorder LLM, or ``None``.
    """

    if prior_episode_state is None:
        return None

    result: dict = {}
    for field_name in _RECORDER_PRIOR_SCALAR_FIELDS:
        if field_name in prior_episode_state:
            result[field_name] = prior_episode_state[field_name]
    for field_name in _ENTRY_LIST_FIELDS:
        result[field_name] = _prior_entry_texts(prior_episode_state, field_name)
    for field_name in _STRING_LIST_FIELDS:
        result[field_name] = _prior_string_list(prior_episode_state, field_name)
    return result


def validate_recorder_output(payload: dict) -> dict:
    """Validate and normalize recorder JSON.

    Args:
        payload: Parsed LLM JSON object.

    Returns:
        Normalized recorder output.

    Raises:
        ValueError: If the payload violates the recorder contract.
    """

    continuity = _require_string(payload.get("continuity", ""), "continuity")
    status = _require_string(payload.get("status", ""), "status")
    # `new_episode` is prompt-facing lifecycle vocabulary. If the recorder copies
    # it into continuity, persist canonical `sharp_transition` instead.
    if continuity == "new_episode":
        continuity = "sharp_transition"
    if continuity not in VALID_CONTINUITY:
        raise ValueError(f"invalid continuity: {continuity}")
    if status not in VALID_STATUS:
        raise ValueError(f"invalid status: {status}")
    return_value = {
        "continuity": continuity,
        "status": status,
        "episode_label": _require_string(payload.get("episode_label", ""), "episode_label"),
        "conversation_mode": _require_string(
            payload.get("conversation_mode", ""),
            "conversation_mode",
        ),
        "episode_phase": _require_string(
            payload.get("episode_phase", ""),
            "episode_phase",
        ),
        "topic_momentum": _require_string(
            payload.get("topic_momentum", ""),
            "topic_momentum",
        ),
        "current_thread": _require_string(payload.get("current_thread", ""), "current_thread"),
        "user_goal": _require_string(payload.get("user_goal", ""), "user_goal"),
        "current_blocker": _require_string(payload.get("current_blocker", ""), "current_blocker"),
        "user_state_updates": _string_list(payload.get("user_state_updates", []), "user_state_updates"),
        "assistant_moves": _string_list(payload.get("assistant_moves", []), "assistant_moves"),
        "overused_moves": _string_list(payload.get("overused_moves", []), "overused_moves"),
        "open_loops": _string_list(payload.get("open_loops", []), "open_loops"),
        "resolved_threads": _string_list(payload.get("resolved_threads", []), "resolved_threads"),
        "avoid_reopening": _string_list(payload.get("avoid_reopening", []), "avoid_reopening"),
        "emotional_trajectory": _require_string(
            payload.get("emotional_trajectory", ""),
            "emotional_trajectory",
        ),
        "next_affordances": _string_list(payload.get("next_affordances", []), "next_affordances"),
        "progression_guidance": _require_string(
            payload.get("progression_guidance", ""),
            "progression_guidance",
        ),
    }
    return return_value
