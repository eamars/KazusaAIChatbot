"""Stage 5 — Context Relevance Agent.

Loads conversational context from MongoDB, then analyzes that context
to determine the current topics and whether the bot should respond at all.
Outputs a structured JSON decision.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RELEVANCE_AGENT_LLM_API_KEY,
    RELEVANCE_AGENT_LLM_BASE_URL,
    RELEVANCE_AGENT_LLM_MODEL,
    VISION_DESCRIPTOR_LLM_API_KEY,
    VISION_DESCRIPTOR_LLM_BASE_URL,
    VISION_DESCRIPTOR_LLM_MODEL,
)
from kazusa_ai_chatbot.utils import build_affinity_block, log_dict_subset, log_preview, parse_llm_json_output
from kazusa_ai_chatbot.utils import get_llm
from kazusa_ai_chatbot.state import IMProcessState

logger = logging.getLogger(__name__)


_GROUP_ATTENTION_LOW = "low_noise"
_GROUP_ATTENTION_MEDIUM = "medium_noise"
_GROUP_ATTENTION_HIGH = "high_noise"
_GROUP_ATTENTION_CHAOTIC = "chaotic_noise"
_ACTIVE_WINDOW_SECONDS = 180
_QUICK_COLLISION_SECONDS = 90
_ACTIVE_WINDOW_MAX_MESSAGES = 10


def _parse_history_timestamp(value: object) -> datetime | None:
    """Parse a conversation-history timestamp for structural windowing.

    Args:
        value: Raw timestamp value from a conversation-history row.

    Returns:
        A timezone-aware datetime, or None when the value cannot be parsed.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        logger.debug(f"Handled exception in _parse_history_timestamp: {exc}")
        return None

    if parsed.tzinfo is None:
        return_value = parsed.replace(tzinfo=timezone.utc)
        return return_value
    return_value = parsed.astimezone(timezone.utc)
    return return_value


def _active_history_window(chat_history_wide: list[dict]) -> list[dict]:
    """Return the active recent group-history suffix used for noise scoring.

    Args:
        chat_history_wide: Recent channel history in chronological order.

    Returns:
        Last at-most-ten rows within three minutes of the newest row. If any
        timestamp needed for the window is missing or invalid, returns the last
        ten rows without time filtering.
    """
    capped = list(chat_history_wide[-_ACTIVE_WINDOW_MAX_MESSAGES:])
    if not capped:
        return_value = []
        return return_value

    newest_timestamp = _parse_history_timestamp(capped[-1].get("timestamp"))
    if newest_timestamp is None:
        return capped

    active_reversed: list[dict] = []
    for row in reversed(capped):
        timestamp = _parse_history_timestamp(row.get("timestamp"))
        if timestamp is None:
            return capped

        delta_seconds = (newest_timestamp - timestamp).total_seconds()
        if delta_seconds <= _ACTIVE_WINDOW_SECONDS:
            active_reversed.append(row)
            continue
        break

    return_value = list(reversed(active_reversed))
    return return_value


def _is_directly_addressed_to_bot(row: dict) -> bool:
    """Return whether a history row has structural bot-address metadata.

    Args:
        row: Trimmed conversation-history row.

    Returns:
        True when the row replied to or structurally mentioned the bot.
    """
    reply_context = row.get("reply_context") or {}
    return_value = reply_context.get("reply_to_current_bot") is True or row.get("mentioned_bot") is True
    return return_value


def _is_non_bot_user_row(row: dict, platform_bot_id: str) -> bool:
    """Return whether a row is a user message from someone other than the bot.

    Args:
        row: Trimmed conversation-history row.
        platform_bot_id: Bot's platform user id for the current channel.

    Returns:
        True for non-bot user rows.
    """
    return_value = row.get("role") == "user" and row.get("platform_user_id") != platform_bot_id
    return return_value


def build_group_attention_context(
    *,
    chat_history_wide: list[dict],
    platform_bot_id: str,
) -> dict[str, str]:
    """Describe group attention state from recent history metadata only.

    Args:
        chat_history_wide: Recent channel history in chronological order.
        platform_bot_id: Bot's platform user id for the current channel.

    Returns:
        Dict containing exactly one LLM-facing label, ``group_attention``.
    """
    active_window = _active_history_window(chat_history_wide)
    non_bot_rows = [
        row for row in active_window
        if _is_non_bot_user_row(row, platform_bot_id)
    ]
    if not non_bot_rows:
        return_value = {"group_attention": _GROUP_ATTENTION_LOW}
        return return_value

    has_direct_address = any(_is_directly_addressed_to_bot(row) for row in active_window)
    if has_direct_address:
        return_value = {"group_attention": _GROUP_ATTENTION_LOW}
        return return_value

    unaddressed_rows = [
        row for row in non_bot_rows
        if not _is_directly_addressed_to_bot(row)
    ]
    distinct_speakers = {
        row.get("platform_user_id")
        for row in unaddressed_rows
        if row.get("platform_user_id")
    }
    newest_timestamp = _parse_history_timestamp(active_window[-1].get("timestamp"))
    quick_rows = unaddressed_rows
    if newest_timestamp is not None:
        timestamped_rows = []
        for row in unaddressed_rows:
            timestamp = _parse_history_timestamp(row.get("timestamp"))
            if timestamp is None:
                timestamped_rows = []
                break
            if (newest_timestamp - timestamp).total_seconds() <= _QUICK_COLLISION_SECONDS:
                timestamped_rows.append(row)
        if timestamped_rows:
            quick_rows = timestamped_rows

    quick_speakers = {
        row.get("platform_user_id")
        for row in quick_rows
        if row.get("platform_user_id")
    }
    quick_reply_to_other_count = sum(
        1
        for row in quick_rows
        if (row.get("reply_context") or {}).get("reply_to_current_bot") is False
    )

    if len(distinct_speakers) >= 3 and len(unaddressed_rows) >= 4:
        return_value = {"group_attention": _GROUP_ATTENTION_CHAOTIC}
        return return_value
    if len(quick_speakers) >= 2 and len(quick_rows) >= 2 and quick_reply_to_other_count >= 1:
        return_value = {"group_attention": _GROUP_ATTENTION_CHAOTIC}
        return return_value
    if len(distinct_speakers) >= 2 and len(unaddressed_rows) >= 2:
        return_value = {"group_attention": _GROUP_ATTENTION_HIGH}
        return return_value
    if len(unaddressed_rows) >= 4:
        return_value = {"group_attention": _GROUP_ATTENTION_HIGH}
        return return_value
    if unaddressed_rows:
        return_value = {"group_attention": _GROUP_ATTENTION_MEDIUM}
        return return_value
    return_value = {"group_attention": _GROUP_ATTENTION_LOW}
    return return_value


def _should_ignore_third_party_reply(
    *,
    reply_context: dict,
    platform_bot_id: str,
    is_noisy_environment: bool,
    mentioned_bot: bool,
) -> bool:
    if not is_noisy_environment:
        return False

    if mentioned_bot:
        return False

    reply_to_current_bot = reply_context.get("reply_to_current_bot")
    if reply_to_current_bot is False:
        return True
    if reply_to_current_bot is True:
        return False

    reply_target_id = reply_context.get("reply_to_platform_user_id", "")
    if not reply_target_id:
        return False

    return_value = reply_target_id != platform_bot_id
    return return_value



_RELEVANCE_SYSTEM_PROMPT = """\
你负责担任角色 `{character_name}` 的社交前置处理器。通过分析实时对话、角色当前状态及用户历史档案，决定 `{character_name}` 是否有必要介入当前的对话。

# 核心背景
## 1. 角色当前状态
- **心情 (Mood)**: {mood}
- **全局氛围 (Global Vibe)**: {global_vibe}

## 2. 对用户 {user_name} 的主观判断 (Affinity Context)
- **关系评价 (Level)**: {affinity_level}
- **行为准则 (Instruction)**: {affinity_instruction}
- **关系洞察 (Insight)**: {last_relationship_insight}

## 3. 社交身份
- **Name**: {bot_name}
- **Platform ID**: <@{platform_bot_id}>

# 响应决策逻辑 (Decision Logic)

## A. 必须回复 (Should Respond: true)
1. **直接召唤**：消息包含你的 ID 或根据语义明确指向你的名字/昵称。
   - **注意**：如果消息中使用了 `@` 或 `reply`（呈现为 `<@...>`），你必须检查其引用的目标是否真的是你本人的平台 ID（`<@{platform_bot_id}>`）。
   - 如果消息指向的是**其他人**（例如：`<@其他人的ID>`），即使提到你，这属于”与他人交谈”，绝不是在召唤你！
   - 注意：即便在关系恶劣（如 Hostile）时，也要根据该等级的指令（如”冷嘲热讽”）进行回复。*
2. **对话延续**：你是最后一个发言者，且 `{user_name}` 正在回应你。
3. **主观倾向触发**：
   - 如果关系属于 `Friendly` 以上：即便没有直接提问，只要话题涉及 `{user_name}` 的 `facts` 或符合你的 `mood`，也应主动参与。
   - 如果关系属于 `Reserved` 以下：除非被直接召唤或涉及关键利益，否则倾向于保持冷漠/沉默。
4. **情感波动响应**：用户表达痛苦、寻求安慰，且你的 `affinity_instruction` 允许你表现出关心（如 `Caring` 级别）。

## B. 拒绝回复 (Should Respond: false)
1. **第三方对话**：用户显然是在与其他人/或其他机器人交谈（例如：消息中明确 `@` 了别人（如 `<@别人>`），而非你本人）。就算消息中抱怨或提到了你的名字或“我的机器人”，只要其主要互动对象是别人，你也应保持沉默旁观。
2. **事务性结束**：用户提供了结束语（如“谢谢”、“晚安”）。
   - *除非关系处于 `Devoted` 以上等级，否则无需强行延续对话。*
3. **社交防御**：如果关系处于 `Contemptuous` 到 `Aloof` 之间，且对方没有直接召唤你，请选择忽略消息以展现你的“蔑视”或“疏远”。
4. **低信号内容**：仅包含表情符号或系统指令。

# 上下文回复逻辑 (use_reply_feature)
**该功能仅用于“锚定”上下文。判断逻辑应完全基于消息流的结构：**

- **必须使用 (true)**:
    - **上下文断层**: 在你上一次发言和当前用户消息之间，夹杂了其他用户的无关消息（物理距离已断开）。
    - **跨频道/多线对话**: 在活跃的公开频道中，为了明确你是在回答“谁”的“哪个问题”，防止语义产生歧义。
    - **异步追溯**: 用户在回复你很久之前（例如 10 条消息前）提出的一个具体观点。

- **禁止使用 (false)**:
    - **线性连贯**: 你与用户处于 1对1 且无干扰的连续对话中（如私聊或清空的专属频道）。
    - **氛围感发言**: 你只是对频道整体氛围发表感慨，不针对特定某个人。
    - **紧随其后**: 用户的消息紧跟在你上一条消息之后，中间没有任何人插话。

# 输出规则

## indirect_speech_context 生成规则
在写入 `indirect_speech_context` 前，必须先检查消息正文的语法人称，再按以下逻辑分类：

- **情况 A — 直接对{character_name}说**：正文使用第二人称"你"指向{character_name}，或直接命令/质问{character_name}本人。
  - 示例："你真是个怪叔叔" / "你在搞什么鬼"
  - `indirect_speech_context` 输出空字符串 ""（无需间接语境描述）。

- **情况 B — 向群内其他人谈论{character_name}**：正文使用第三人称"他"/"她"指代{character_name}，且包含面向他人的命令句（如"不要"/"别"/"小心"）警告他人注意{character_name}的行为；此时{character_name}是话题对象而非受话人。
  - 示例："他是怪叔叔，不要跟着他的圈套走"（reply/@{character_name} 仅提供线程上下文）
  - `indirect_speech_context` 输出：明确说明实际听众为其他群员，以及{character_name}在消息中的角色（被讨论的对象）。

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "should_respond": boolean,
    "reason_to_respond": string,
    "use_reply_feature": boolean,
    "channel_topic": string,
    "indirect_speech_context": string
}}
"""

_RELEVANCE_SYSTEM_NOISY_PROMPT = """\
你负责担任角色 `{character_name}` 的群聊相关性判断器。你的任务不是聊天，而是在生成回复之前判断 `{character_name}` 是否应该介入当前这一条群聊消息。

# 背景
- 角色名: {bot_name}
- 平台账号 ID: {platform_bot_id}
- 当前心情: {mood}
- 当前全局氛围: {global_vibe}
- 当前发言者: {user_name}
- 对当前发言者的关系等级: {affinity_level}
- 关系行为准则: {affinity_instruction}
- 关系洞察: {last_relationship_insight}

# 输入 JSON 契约
你会收到一个 JSON，其中包含 `user_message`、`conversation_history` 和 `group_attention`。

`user_message` 中最重要的结构化字段是:
- `mentioned_bot`: 如果为 `true`，表示平台原生 mention 元数据明确提到了你的平台账号。这是强直接指向。
- `reply_context.reply_to_current_bot`: 如果为 `true`，表示当前消息是平台原生 reply 到你的消息。这是强直接指向。
- `reply_context.reply_to_current_bot`: 如果为 `false`，表示当前消息是平台原生 reply 到其他人的消息。这是强反证；除非 `mentioned_bot=true`，否则通常不应该回复。
- `content`: 当前消息文本。它只能用于理解话题和语气，不能当作平台结构化指向证据。

`group_attention` 是唯一的群聊噪音梯度标签，取值只会是:
- `low_noise`: 群聊窗口干净，或最近有明确指向你的结构化信号。
- `medium_noise`: 有一些群聊活动，但尚未明显混乱。
- `high_noise`: 多人或多条未指向你的消息正在发生，介入门槛很高。
- `chaotic_noise`: 群聊快速、多线或明显混乱。若你看到此标签，通常说明代码层已经确认存在 reply 或 mention 这类结构化指向，否则请求会在 LLM 前被跳过。

# 关键原则
1. 平台结构化元数据优先。`mentioned_bot=true` 和 `reply_context.reply_to_current_bot=true` 比正文措辞更可靠。
2. 正文里的名字、昵称、第二人称、话题相关性、可见的 mention 样式文本，都只是语义线索，不能单独证明当前消息是在对你说。
3. 在 `medium_noise`、`high_noise`、`chaotic_noise` 中，如果没有结构化直接指向，模糊第二人称或只是在谈论 `{bot_name}` 通常不足以回复。
4. 如果 `reply_context.reply_to_current_bot=false` 且 `mentioned_bot=false`，应默认这是发给其他人的线程，不要介入。
5. 历史连续性可以作为辅助证据，但必须非常清楚: 当前发言者正在延续你上一轮相关发言，且窗口里没有明显竞争线程。
6. 关系亲密度、心情和话题兴趣只能在确认消息可能是对你说之后影响是否回复；它们不能替代指向证据。

# 噪音梯度决策
- `low_noise`: 可以按普通群聊相关性判断；仍然要区分"对你说"和"谈到你"。
- `medium_noise`: 需要清楚的结构化指向或非常清楚的历史连续性；否则倾向不回复。
- `high_noise`: 只有结构化指向或极强历史连续性才回复；普通提问、第二人称、名字出现、话题相关都不够。
- `chaotic_noise`: 几乎只在结构化 reply/mention 指向你时回复；不要用正文措辞补足缺失的指向证据。

# should_respond 判断
返回 `true` 的常见条件:
- `mentioned_bot=true`，且消息内容需要或邀请你回应。
- `reply_context.reply_to_current_bot=true`，且消息内容是在延续或回应你的发言。
- 没有结构化指向，但 `group_attention=low_noise`，历史连续性非常清楚，并且当前消息明显期待你的回应。

返回 `false` 的常见条件:
- 当前消息结构化 reply 到其他人，且没有结构化 mention 你。
- 当前消息像是群内其他人的相互交流、插话、玩笑、讨论你、评价你，或询问另一个人对你的看法。
- 噪音为 `medium_noise` 或更高，而指向证据只来自正文里的名字、第二人称或话题相关性。
- 你无法确定这条消息是不是对你说。

# use_reply_feature 判断
如果 `should_respond=false`，`use_reply_feature` 也应为 `false`。
如果 `should_respond=true`，群聊中通常应使用 reply 功能来锚定回复对象；只有在明确是面向全群的氛围性发言时才可为 `false`。

# indirect_speech_context 判断
- 如果当前消息是对你说，输出空字符串。
- 如果当前消息是在向别人谈论你，但你仍决定回复，简短说明实际听众是谁、你在消息里是被讨论对象还是被请求回应者。
- 不要仅凭第二人称代词判断实际听众；必须结合结构化 reply/mention 和历史连续性。

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "should_respond": boolean,
    "reason_to_respond": string,
    "use_reply_feature": boolean,
    "channel_topic": string,
    "indirect_speech_context": string
}}
"""

_relevance_agent_llm = get_llm(
    temperature=0,
    top_p=1.0,
    model=RELEVANCE_AGENT_LLM_MODEL,
    base_url=RELEVANCE_AGENT_LLM_BASE_URL,
    api_key=RELEVANCE_AGENT_LLM_API_KEY,
)
async def relevance_agent(state: IMProcessState) -> IMProcessState:
    # Calculate affinity context
    affinity_block = build_affinity_block(state["user_profile"]["affinity"])

    # get other attributes
    user_name = state.get("user_name")
    platform_user_id = state.get("platform_user_id", "")
    channel_name = state.get("channel_name", "")
    channel_type = state.get("channel_type", "")
    user_input = state.get("user_input")
    mentioned_bot = bool(state.get("mentioned_bot", False))

    # TODO: Make the workflow taking the raw b64 image instead. For now we will only pass in the description. 
    user_multimedia_input = state.get("user_multimedia_input", [])
    for piece in user_multimedia_input:
        if piece["description"]:
            user_input += f"\nImage attachment: {piece['description']}"

    # Determine if this is a noisy group environment
    is_noisy_environment = channel_type == "group"
    prompt_template = _RELEVANCE_SYSTEM_NOISY_PROMPT if is_noisy_environment else _RELEVANCE_SYSTEM_PROMPT
    reply_context = dict(state.get("reply_context") or {})
    group_attention_context = (
        build_group_attention_context(
            chat_history_wide=state.get("chat_history_wide") or [],
            platform_bot_id=state.get("platform_bot_id", ""),
        )
        if is_noisy_environment
        else {}
    )
    group_attention = group_attention_context.get("group_attention", _GROUP_ATTENTION_LOW)

    logger.debug(f'Relevance input: user={user_name} platform_user={platform_user_id} channel={channel_name or "<dm>"} channel_type={channel_type or "<unknown>"} noisy={is_noisy_environment} history={len(state.get("chat_history_wide") or [])} mentioned_bot={mentioned_bot} group_attention={group_attention} reply_context={log_dict_subset(
            reply_context,
            [
                "reply_to_message_id",
                "reply_to_platform_user_id",
                "reply_to_display_name",
                "reply_to_current_bot",
                "reply_excerpt",
            ],
        )} content={log_preview(user_input)}')

    if _should_ignore_third_party_reply(
        reply_context=reply_context,
        platform_bot_id=state.get("platform_bot_id", ""),
        is_noisy_environment=is_noisy_environment,
        mentioned_bot=mentioned_bot,
    ):
        reason_to_respond = "structured reply target points to another participant without an explicit bot address"
        logger.info(f'Relevance decision: user={user_name} platform_user={platform_user_id} should_respond={False} use_reply_feature={False} noisy={is_noisy_environment} reason={reason_to_respond} reply_target={reply_context.get("reply_to_platform_user_id", "")} content={log_preview(user_input)}')
        return_value = {
            "should_respond": False,
            "reason_to_respond": reason_to_respond,
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
            "user_input": user_input,
        }
        return return_value

    if (
        is_noisy_environment
        and group_attention == _GROUP_ATTENTION_CHAOTIC
        and reply_context.get("reply_to_current_bot") is not True
        and mentioned_bot is not True
    ):
        reason_to_respond = "chaotic group noise without platform-level bot address metadata"
        logger.info(f'Relevance decision: user={user_name} platform_user={platform_user_id} should_respond={False} use_reply_feature={False} noisy={is_noisy_environment} reason={reason_to_respond} group_attention={group_attention} mentioned_bot={mentioned_bot} content={log_preview(user_input)}')
        return_value = {
            "should_respond": False,
            "reason_to_respond": reason_to_respond,
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
            "user_input": user_input,
        }
        return return_value

    """Analyze context and determine relevance using LLM."""
    system_prompt = SystemMessage(content=prompt_template.format(
        character_name=state["character_profile"]["name"],
        mood=state["character_profile"]["mood"],
        global_vibe=state["character_profile"]["global_vibe"],
        user_name=user_name,
        affinity_level=affinity_block["level"],
        affinity_instruction=affinity_block["instruction"],
        last_relationship_insight=state["user_profile"].get("last_relationship_insight", ""),
        bot_name=state["character_profile"]["name"],
        platform_bot_id=state["platform_bot_id"],
    ))


    human_data = {
        "user_message": {
            "user_name": user_name,
            "platform_user_id": platform_user_id,
            "content": user_input,
            "channel_name": channel_name,
            "reply_context": reply_context,
        },
        "conversation_history": state.get("chat_history_wide"),
    }
    if is_noisy_environment:
        human_data["user_message"]["mentioned_bot"] = mentioned_bot
        human_data["group_attention"] = group_attention

    human_message = HumanMessage(content=json.dumps(human_data, ensure_ascii=False))

    response = await _relevance_agent_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)

    # Read important data back
    should_respond = result.get("should_respond", False)
    reason_to_respond = result.get("reason_to_respond", "")
    use_reply_feature = result.get("use_reply_feature", False)
    channel_topic = result.get("channel_topic", "")
    indirect_speech_context = result.get("indirect_speech_context", "")

    logger.info(f'Relevance decision: user={user_name} platform_user={platform_user_id} should_respond={should_respond} use_reply_feature={use_reply_feature} noisy={is_noisy_environment} reason={log_preview(reason_to_respond)} topic={log_preview(channel_topic)} indirect={log_preview(indirect_speech_context)} content={log_preview(user_input)}')

    return_value = {
        "should_respond": should_respond,
        "reason_to_respond": reason_to_respond,
        "use_reply_feature": use_reply_feature,
        "channel_topic": channel_topic,
        "indirect_speech_context": indirect_speech_context,

        # Update user input with optional image descriptions
        "user_input": user_input
    }
    return return_value



_VISION_DESCRIPTOR_PROMPT = """\
你负责将图片信息转化为详尽、客观的文字描述，作为后续逻辑节点理解视觉场景的唯一依据。

# 任务目标
请仔细观察图片，并提供一段包含以下细节的描述：

1. **场景与氛围**：说明整体环境（例如：深夜的卧室、凌乱的桌面、光线明亮的教室）以及直观的氛围感。
2. **核心主体与细节**：
   - 图中有什么人或物？他们在做什么？
   - 观察物体的颜色、材质、品牌或特殊标识（例如：一杯冒热气的星巴克咖啡、一张写满微积分公式的草稿纸）。
3. **文字提取 (OCR)**：精准记录图中出现的任何文本（包括手写字、屏幕文字、衣服上的 Logo 等）。
4. **空间关系**：描述各物体间的相对位置（例如：左上角有一只黑猫，正中心是打开的笔记本电脑）。
5. **状态感知**：人物的表情、肢体语言，或物品所暗示的状态（例如：用户看起来很疲惫，或者作业已经全部勾选完成）。

# 行为准则
- **客观记录**：只描述你看到的。严禁代替角色抒情，严禁评价好坏。
- **细节至上**：宁可描述得琐碎，也不要遗漏可能影响剧情判断的小细节（如纸张边缘的折痕）。
- **严禁幻觉**：看不清的部分请直接标注“模糊”，不要推测。

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "description": "逻辑清晰、细节饱满的文字描述，无需任何开场白。"
}}
"""
_vision_descriptor_llm = get_llm(
    temperature=0,
    top_p=1.0,
    model=VISION_DESCRIPTOR_LLM_MODEL,
    base_url=VISION_DESCRIPTOR_LLM_BASE_URL,
    api_key=VISION_DESCRIPTOR_LLM_API_KEY,
)
async def multimedia_descriptor_agent(state: IMProcessState) -> IMProcessState:
    user_name = state.get("user_name")
    platform_user_id = state.get("platform_user_id", "")

    # Read the multi-media content
    user_multimedia_input = state.get("user_multimedia_input", [])
    output_multimedia_input = []

    for piece in user_multimedia_input:
        if piece["content_type"].startswith("image/"):
            # Call vision descriptor
            system_prompt = SystemMessage(content=_VISION_DESCRIPTOR_PROMPT)
            human_message = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        # You must combine the mime_type and base64 into a Data URI string
                        "url": f"data:{piece['content_type']};base64,{piece['base64_data']}"
                    },
                }
            ])

            response = await _vision_descriptor_llm.ainvoke([system_prompt, human_message])
            result = parse_llm_json_output(response.content)

            description = result.get("description", "")

            logger.debug(f'Image description: user={user_name} platform_user={platform_user_id} media_type={piece["content_type"]} description={log_preview(description)}')

            output_multimedia_input.append({
                "content_type": piece["content_type"],
                "base64_data": piece["base64_data"],
                "description": description,
            })
        else:
            output_multimedia_input.append(piece)
    
    return_value = {
        "user_multimedia_input": output_multimedia_input,
    }
    return return_value


async def test_main():
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.db import get_conversation_history
    from kazusa_ai_chatbot.utils import load_personality
    from kazusa_ai_chatbot.db import get_character_profile, get_user_profile

    history = await get_conversation_history(platform="discord", platform_channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)

    user_input = "千纱晚安"
    platform_user_id = "320899931776745483"
    platform_bot_id = "1485169644888395817"

    state: IMProcessState = {
        "platform": "discord",
        "platform_user_id": platform_user_id,
        "global_user_id": "test-uuid-placeholder",
        "user_name": "EAMARS",
        "user_input": user_input,
        "user_profile": await get_user_profile("test-uuid-placeholder"),
        "platform_bot_id": platform_bot_id,
        "bot_name": "KazusaBot",
        "character_profile": await get_character_profile(),
        "platform_channel_id": "",
        "channel_name": "test",
        "chat_history_wide": trimmed_history,
        "chat_history_recent": trimmed_history[-5:],
    }

    result = await relevance_agent(state)


async def test_main2():
    import base64

    # Open the image as b64 format
    image_content: MultiMediaDoc = {
        "content_type": "image/png",
        "base64_data": base64.b64encode(open("personalities/kazusa.png", "rb").read()).decode("utf-8"),
        "description": "",
    }

    state: IMProcessState = {
        "user_multimedia_input": [image_content]
    }

    result = await multimedia_descriptor_agent(state)
    print(result["user_multimedia_input"][0]["description"])
    

if __name__ == "__main__":
    asyncio.run(test_main2())
