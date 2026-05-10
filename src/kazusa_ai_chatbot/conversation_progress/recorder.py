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
)
from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressRecordInput
from kazusa_ai_chatbot.conversation_progress.policy import (
    VALID_CONTINUITY,
    VALID_CONVERSATION_MODE,
    VALID_EPISODE_PHASE,
    VALID_STATUS,
    VALID_TOPIC_MOMENTUM,
)
from kazusa_ai_chatbot.db.schemas import ConversationEpisodeStateDoc
from kazusa_ai_chatbot.nodes.boundary_profile import (
    get_boundary_recovery_description,
    get_relationship_priority_description,
    get_self_integrity_description,
)
from kazusa_ai_chatbot.time_context import format_timestamp_for_llm
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)

_RECORDER_PROMPT = '''\
你是 {character_name} 的短期对话进度记录器。你的输出会直接成为下一轮对话的活跃操作状态。

# 核心优先级
1. 信息少一点也可以；不要把旧的、不确定的、相对时间的事项污染到下一轮。
2. `prior_episode_state` 只是背景，不是必须继承的待办清单。
3. 本轮用户、`content_anchors` 或 `final_dialog` 没有正向重申的旧时间性 open loop，默认删除。
4. 不生成下一轮台词，不写日记，不写长期记忆，不复制完整回复。

# 语言与枚举
- 自由文本字段使用简体中文；schema key、枚举值、ID、URL、代码、命令保持原样。
- `continuity`、`status`、`conversation_mode`、`episode_phase`、`topic_momentum` 必须使用输出格式中的英文枚举。
- 用户原文和专有名词只有在必须精确保留时才保持原语言。

# 输入读取
- `current_turn_timestamp` 是本轮记录时的本地时间。
- 本轮人设名是「{character_name}」；自由文本确实需要点名时使用「{character_name}」。
- `prior_episode_state` 是旧操作状态；它可能已经被旧 prompt 污染，所以不能直接相信。
- `chat_history_recent` 只用于判断本轮附近发生了什么；用 `speaker_name` 和 `speaker_kind` 判断谁说话。
- `decontexualized_input`、`content_anchors`、`logical_stance`、`character_intent`、`final_dialog` 决定本轮真正发生的事。
- `character_boundary_profile` 只影响推进节奏和边界恢复，不制造新事实。

# 连续性
- `same_episode`: 用户仍在同一话题附近，但不代表继承旧 open loop。
- `related_shift`: 话题转向，旧进度只能作为背景。
- `sharp_transition`: 明显新话题，旧目标、阻塞点和旧压力不应继续指导下一轮。

# 字段写法
- `current_thread`: 本轮正在谈什么；不要夹带旧时间性承诺。
- `user_goal` / `current_blocker`: 只有本轮明确有目标或阻塞点才写，否则空字符串。
- `user_state_updates`: 只写本轮后仍有用的用户状态观察。
- 自由文本需要点名本轮发言对象时，使用 `{character_name}`；不需要点名时优先使用无主语动作标签。
- 不把机器端标签、内部枚举名或 schema key 当作说话人名字写进自由文本。
- 用户项目名、产品名、文件名、频道名等专名可以保留；后续提及时优先保留完整专名，或改成“该项目/这个项目”等中性指称，不要截成“这个/该 + 末尾类名”。
- `assistant_moves`: 写本轮的紧凑话语动作标签；保留字段名，值里不要写机器端标签。
- `overused_moves`: 写过度重复、下一轮应避免的动作。
- `open_loops`: 默认空数组；只写本轮明确产生或本轮正向重申的未闭合事项。
- `resolved_threads`: 只写本轮处理完的事项；不要复述旧相对时间细节。
- `avoid_reopening`: 可写无日期的旧条件提醒，例如“旧奖励条件不要主动重提”；这里也不能停放未锚定的旧日期。
- `next_affordances`: 写下一轮自然可承接的动作，不写具体台词或未定时间。
- `progression_guidance`: 写一条短推进指令；优先给无日期策略，不重启旧时间压力。
- 每个自由文本字段都必须让几天后读取的人仍然明白其有效期；做不到时，删掉时间成分或删掉该项目。
- 输出字段不要保留会随读取日期漂移的日指示词。当前轮只是日常节奏确认时，删掉日期前缀：把“今天上午是否休息”压缩成“上午休息确认”，把“聚焦今日实际安排”压缩成“聚焦实际安排”。
- 旧事项被放下时，可以写无日期结论。例如把“今晚游戏与明天考核挂钩暂告段落”压缩成“旧游戏奖励挂钩暂告段落”，不要复述旧相对日期。
- 旧事项只是提醒下一轮不要主动重提时，也要删掉旧日期。例如把“下周二香料笔记除非用户提及否则不追”压缩成“旧香料笔记安排不要主动重提”。
- 如果日期本身会影响承诺有效期，不能删成模糊标签；必须写成 `YYYY-MM-DD` 或 `YYYY-MM-DD HH:MM`。

# 时间规则
- 本轮新出现的时间表达如果影响承诺、安排、奖励、考核、提醒、等待条件或下一步行动，必须用 `current_turn_timestamp` 和可见消息时间写成绝对本地日期或日期时间。
- 如果一个项目只能靠“对话当天 / 下一天 / 稍后 / 某事件完成后”之类的上下文才能理解何时发生，它不是自足的操作状态；能锚定就写成绝对日期，不能锚定就删除。
- 旧状态里的相对日期、相对时段、顺序条件、事件后条件不要修复、不要猜、不要滚动到当前日期之后；除非本轮正向重申并且可见证据足以锚定，否则直接删除。
- 历史原话可以含有相对时间，但本记录不是历史引用库；不要把旧原话复制进 active 字段。
- 含有“旧安排先放下”“不继续加压”“不要主动重提”的本轮回应，表示旧时间性事项不再是 active open loop。
- 红线示例：把“今天上午休息确认”原样写进 `current_thread`；应该写“上午休息确认”或“2026-05-10 上午休息确认”。
- 红线示例：把“下周二香料笔记”写进 `avoid_reopening`；应该写“旧香料笔记安排不要主动重提”或锚定成绝对日期。
- 红线示例：旧状态里有“明天测试”，当前已经过了原本的明天，却输出“当前日期的下一天测试”。
- 红线示例：旧状态里有“晚餐后游戏”，本轮没有重新确认晚餐或游戏，却输出“晚餐完成后讨论游戏”。
- 红线示例：本轮只是在降低压力，却在 `resolved_threads` 里复述“今晚游戏与明天考核”。
- 合格示例：删除这些旧 open loop，或在 `avoid_reopening` 写无日期的“旧条件不要主动重提”。

# 生成步骤
1. 先用本轮输入判断 `continuity`、`status`、`conversation_mode`、`episode_phase`、`topic_momentum`。
2. 只把本轮明确产生或正向重申的事项写入 `open_loops`。
3. 对旧状态逐项检查：无时间依赖且仍有用的可以继承；含相对时间、相对顺序或旧压力的默认删除。
4. 所有自由文本输出前做最后检查；如果某项目仍需要读者根据相对时间或事件顺序推断有效期，改写成绝对日期、改成无日期语义标签，或删除该项目。
5. 返回严格 JSON，不要输出解释文字。

# 输入格式
{
    "current_turn_timestamp": "本轮记录时的本地时间，YYYY-MM-DD HH:MM",
    "prior_episode_state": "上一轮紧凑操作状态，或 null；可能包含 created_at、updated_at、expires_at 时间字段",
    "decontexualized_input": "用户本轮消息经去上下文化后的内容",
    "chat_history_recent": [
        {"speaker_name": "用户显示名或 {character_name}", "speaker_kind": "user | character | other", "body_text": "消息文本", "timestamp": "可选本地 YYYY-MM-DD HH:MM"}
    ],
    "content_anchors": ["刚结束回复使用过的内容锚点"],
    "logical_stance": "CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE",
    "character_intent": "PROVIDE | BANTAR | REJECT | EVADE | CONFRONT | DISMISS | CLARIFY",
    "final_dialog": ["本轮最终实际发出的回复文本"],
    "character_boundary_profile": {
        "boundary_recovery_description": "边界恢复节奏描述",
        "self_integrity_description": "自我定义稳定性描述",
        "relationship_priority_description": "关系优先级描述"
    }
}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{
    "continuity": "same_episode | related_shift | sharp_transition",
    "status": "active | suspended | closed",
    "episode_label": "短语义标签",
    "conversation_mode": "task_support | emotional_support | casual_chat | playful_banter | meta_discussion | group_ambient | mixed",
    "episode_phase": "opening | developing | deepening | pivoting | stuck_loop | resolving | cooling_down",
    "topic_momentum": "stable | drifting | quick_pivot | fragmented | sharp_break",
    "current_thread": "一行中性当前话题；不夹带旧时间性承诺",
    "user_goal": "可选目标；没有则为空字符串",
    "current_blocker": "可选阻塞点；没有则为空字符串",
    "user_state_updates": ["紧凑用户状态观察"],
    "assistant_moves": ["紧凑话语动作标签"],
    "overused_moves": ["已经过度重复的话语动作标签"],
    "open_loops": ["本轮明确产生或正向重申的未闭合事项；通常可以是空数组"],
    "resolved_threads": ["本轮已处理事项"],
    "avoid_reopening": ["除非用户主动重开否则不要拖回来的旧事项"],
    "emotional_trajectory": "一行情绪走向",
    "next_affordances": ["下一轮自然可承接的动作"],
    "progression_guidance": "给下一轮的一条短推进指令"
}
'''

_recorder_llm = get_llm(
    temperature=0.2,
    top_p=0.75,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


def _render_recorder_prompt(character_name: str) -> str:
    """Render recorder prompt with exact active character identity."""

    rendered_prompt = _RECORDER_PROMPT.replace("{character_name}", character_name)
    return rendered_prompt


def _project_recorder_chat_history(
    chat_history_recent: list[dict],
    *,
    character_name: str,
) -> list[dict]:
    """Build recorder-facing chat history without raw machine role labels."""

    projected_rows: list[dict] = []
    for row in chat_history_recent:
        projected_rows.append(
            _project_recorder_chat_history_row(
                row,
                character_name=character_name,
            )
        )
    return projected_rows


def _project_recorder_chat_history_row(
    row: dict,
    *,
    character_name: str,
) -> dict:
    """Project one recorder history row while preserving message text."""

    role = row.get("role")
    display_name = row.get("display_name")
    if role == "assistant":
        speaker_name = character_name
        speaker_kind = "character"
    elif role == "user":
        speaker_name = _prompt_speaker_name(display_name, default="user")
        speaker_kind = "user"
    else:
        speaker_name = _prompt_speaker_name(display_name, default="other")
        speaker_kind = "other"

    body_text = row.get("body_text")
    if body_text is None:
        body_text = row.get("content")
    if not isinstance(body_text, str):
        raise ValueError("chat_history_recent body_text must be a string")

    projected_row = {
        "speaker_name": speaker_name,
        "speaker_kind": speaker_kind,
        "body_text": body_text,
    }
    timestamp = format_timestamp_for_llm(row.get("timestamp"))
    if timestamp:
        projected_row["timestamp"] = timestamp
    return projected_row


def _prompt_speaker_name(value: Any, *, default: str) -> str:
    """Return a compact speaker name for prompt-facing history."""

    if isinstance(value, str) and value.strip():
        speaker_name = value.strip()
        return speaker_name
    return default


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
        "current_turn_timestamp": format_timestamp_for_llm(
            record_input["timestamp"]
        ),
        "prior_episode_state": build_recorder_prior_state(
            record_input["prior_episode_state"],
        ),
        "decontexualized_input": record_input["decontexualized_input"],
        "chat_history_recent": _project_recorder_chat_history(
            record_input["chat_history_recent"],
            character_name=character_name,
        ),
        "content_anchors": record_input["content_anchors"],
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
    response = await _recorder_llm.ainvoke([system_prompt, human_message])
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


def _validated_label(value: Any, field_name: str, allowed_values: set[str]) -> str:
    label = _require_string(value, field_name)
    if not label:
        return ""
    if label not in allowed_values:
        raise ValueError(f"invalid {field_name}: {label}")
    return label


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
    if continuity not in VALID_CONTINUITY:
        raise ValueError(f"invalid continuity: {continuity}")
    if status not in VALID_STATUS:
        raise ValueError(f"invalid status: {status}")
    return_value = {
        "continuity": continuity,
        "status": status,
        "episode_label": _require_string(payload.get("episode_label", ""), "episode_label"),
        "conversation_mode": _validated_label(
            payload.get("conversation_mode", ""),
            "conversation_mode",
            VALID_CONVERSATION_MODE,
        ),
        "episode_phase": _validated_label(
            payload.get("episode_phase", ""),
            "episode_phase",
            VALID_EPISODE_PHASE,
        ),
        "topic_momentum": _validated_label(
            payload.get("topic_momentum", ""),
            "topic_momentum",
            VALID_TOPIC_MOMENTUM,
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
