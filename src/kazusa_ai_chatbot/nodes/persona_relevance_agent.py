"""Context relevance agent.

Loads conversational context from MongoDB, then analyzes that context
to determine the current topics and whether the bot should respond at all.
Outputs a structured JSON decision.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    RELEVANCE_AGENT_LLM_API_KEY,
    RELEVANCE_AGENT_LLM_BASE_URL,
    RELEVANCE_AGENT_LLM_MODEL,
    RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT,
)
from kazusa_ai_chatbot.db import (
    DatabaseOperationError,
    build_user_engagement_relevance_context,
)
from kazusa_ai_chatbot.conversation_history_prompt_projection import (
    project_conversation_history_for_llm,
)
from kazusa_ai_chatbot.state import IMProcessState
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime
from kazusa_ai_chatbot.utils import (
    build_affinity_block,
    get_llm,
    log_dict_subset,
    log_preview,
    parse_llm_json_output,
)

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
        A parsed storage UTC datetime, or None when the value cannot be parsed.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = parse_storage_utc_datetime(value)
    except ValueError as exc:
        logger.debug(f"Ignoring history row with invalid timestamp: {exc}")
        return None

    return_value = parsed
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


def _is_addressed_to_character(row: dict, character_global_user_id: str) -> bool:
    """Return whether a history row is typed as addressed to the character.

    Args:
        row: Trimmed conversation-history row.
        character_global_user_id: Stable internal UUID for the active character.

    Returns:
        True when the row's typed addressee list includes the character.
    """
    addressed_to = row["addressed_to_global_user_ids"]
    return_value = character_global_user_id in addressed_to
    return return_value


def _is_non_bot_user_row(row: dict, platform_bot_id: str) -> bool:
    """Return whether a row is a user message from someone other than the bot.

    Args:
        row: Trimmed conversation-history row.
        platform_bot_id: Bot's platform user id for the current channel.

    Returns:
        True for non-bot user rows.
    """
    return_value = row["role"] == "user" and row["platform_user_id"] != platform_bot_id
    return return_value


def build_group_attention_context(
    *,
    chat_history_wide: list[dict],
    platform_bot_id: str,
    character_global_user_id: str = CHARACTER_GLOBAL_USER_ID,
) -> dict[str, str]:
    """Describe group attention state from recent history metadata only.

    Args:
        chat_history_wide: Recent channel history in chronological order.
        platform_bot_id: Bot's platform user id for the current channel.
        character_global_user_id: Stable internal UUID for the active character.

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

    has_direct_address = any(
        _is_addressed_to_character(row, character_global_user_id)
        for row in active_window
    )
    if has_direct_address:
        return_value = {"group_attention": _GROUP_ATTENTION_LOW}
        return return_value

    unaddressed_rows = [
        row for row in non_bot_rows
        if not _is_addressed_to_character(row, character_global_user_id)
    ]
    distinct_speakers = {
        row["platform_user_id"]
        for row in unaddressed_rows
        if row["platform_user_id"]
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
        row["platform_user_id"]
        for row in quick_rows
        if row["platform_user_id"]
    }
    quick_reply_to_other_count = sum(
        1
        for row in quick_rows
        if (
            (row.get("reply_context") or {}).get("reply_to_platform_user_id")
            not in ("", None, platform_bot_id)
        )
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
    directly_addressed: bool,
) -> bool:
    if not is_noisy_environment:
        return False

    if directly_addressed:
        return False

    reply_target_id = reply_context.get("reply_to_platform_user_id", "")
    if not reply_target_id:
        return False

    return_value = reply_target_id != platform_bot_id
    return return_value


def _has_latest_bot_turn_continuity(
    *,
    chat_history_wide: list[dict],
    platform_bot_id: str,
    current_global_user_id: str,
) -> bool:
    """Return whether the newest history row is a bot turn visible to this user.

    Args:
        chat_history_wide: Prompt-facing channel history in chronological order.
        platform_bot_id: Bot's platform user id for the current channel.
        current_global_user_id: Internal UUID for the current message author.

    Returns:
        True when the latest row is a bot-authored broadcast or a bot turn
        addressed to the current user.
    """

    if not chat_history_wide:
        return False

    latest_row = chat_history_wide[-1]
    if latest_row["role"] != "assistant":
        return False
    if latest_row["platform_user_id"] != platform_bot_id:
        return False

    is_broadcast = latest_row.get("broadcast") is True
    if is_broadcast:
        return True

    addressed_to = latest_row.get("addressed_to_global_user_ids") or []
    return_value = current_global_user_id in addressed_to
    return return_value



_RELEVANCE_SYSTEM_PROMPT = """\
你负责担任 active character 的社交前置处理器。通过分析实时对话、角色当前状态及用户历史档案，决定 active character 是否有必要介入当前的对话。

# 本轮语境
- `current_run_context.character_name` 和 `current_run_context.platform_bot_id` 是本轮角色身份与平台账号，只用于区分 active character、当前发言者和 reply 目标。
- `current_run_context.mood` 与 `current_run_context.global_vibe` 是角色此刻心境和背景氛围，只能在消息已经可能面向 active character 时调整介入倾向。
- `current_run_context.affinity_level`、`affinity_instruction` 和 `last_relationship_insight` 是对当前发言者的关系判断；它们影响已合格消息的回应意愿，但不能替代指向证据。
- `user_message.user_name` 与 `platform_user_id` 是当前发言者身份；`content` 是当前消息正文；`channel_name` 是会话名称，只用于理解场景和话题。
- `user_message.directly_addressed` 与 `user_message.reply_context` 是结构化平台证据，优先级高于正文措辞。
- `conversation_history` 是近期对话窗口，格式是日志式文本行。每行包含可见发言人和可见消息文本，用来判断线性延续、第三方对话和上下文断层。

# 响应决策逻辑 (Decision Logic)

## A. 必须回复 (Should Respond: true)
1. **直接召唤**：`user_message.directly_addressed=true`，或正文根据语义明确指向你的名字/昵称。
   - **注意**：结构化指向证据来自 typed envelope，不要从正文里的平台标记样式自行解析目标。
   - 如果结构化目标指向的是**其他人**，即使正文提到你，也属于”与他人交谈”，绝不是在召唤你！
   - 注意：即便在关系恶劣（如 Hostile）时，也要根据该等级的指令（如”冷嘲热讽”）进行回复。*
2. **对话延续**：你是最后一个发言者，且当前发言者正在回应你。
3. **主观倾向触发**：
   - 如果关系属于 `Friendly` 以上：即便没有直接提问，只要话题涉及当前发言者的事实或符合 `current_run_context.mood`，也应主动参与。
   - 如果关系属于 `Reserved` 以下：除非被直接召唤或涉及关键利益，否则倾向于保持冷漠/沉默。
4. **情感波动响应**：用户表达痛苦、寻求安慰，且 `current_run_context.affinity_instruction` 允许你表现出关心（如 `Caring` 级别）。

## B. 拒绝回复 (Should Respond: false)
1. **第三方对话**：用户显然是在与其他人/或其他机器人交谈。就算消息中抱怨或提到了你的名字或“我的机器人”，只要其主要互动对象是别人，你也应保持沉默旁观。
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

- **情况 A — 直接对 active character 说**：正文使用第二人称"你"指向 active character，或直接命令/质问 active character 本人。
  - 示例："你真是个怪叔叔" / "你在搞什么鬼"
  - `indirect_speech_context` 输出空字符串 ""（无需间接语境描述）。

- **情况 B — 向群内其他人谈论 active character**：正文使用第三人称"他"/"她"指代 active character，且包含面向他人的命令句（如"不要"/"别"/"小心"）警告他人注意 active character 的行为；此时 active character 是话题对象而非受话人。
  - 示例："他是怪叔叔，不要跟着他的圈套走"（reply/@active character 仅提供线程上下文）
  - `indirect_speech_context` 输出：明确说明实际听众为其他群员，以及 active character 在消息中的角色（被讨论的对象）。

# 思考路径
1. 先读取 `user_message.directly_addressed` 与 `user_message.reply_context`，判断消息是否结构化指向角色本人。
2. 再读取正文内容和 `conversation_history`，判断是否是直接召唤、对话延续、第三方对话或低信号内容。
3. 结合 `current_run_context` 中的当前心情、关系等级和关系洞察，只在已经确认可能需要回复之后调整是否介入。
4. 判断是否需要 `use_reply_feature` 来锚定回复对象。
5. 最后区分直接对角色说话和向别人谈论角色，生成 `indirect_speech_context`。

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
你负责担任 active character 的群聊相关性判断器。
你的任务不是聊天，而是在生成回复之前判断 active character 是否应该介入当前这一条群聊消息。

# 本轮语境

身份与发言者:
- `current_run_context.character_name` 和 `current_run_context.platform_bot_id` 是本轮角色身份与平台账号，只用于区分 active character、当前发言者和 reply 目标。
- `user_message.user_name`（平台 ID 为 `user_message.platform_user_id`）是当前消息的发言者。当前消息永远来自这个人，不要把任何其他字段误认为当前消息的作者。

结构化指向证据:
- `user_message.directly_addressed`: 布尔值。`true` 表示 typed envelope 的 addressee 列表明确包含 active character。
- `user_message.reply_context`: 描述当前消息所回复的**目标消息**的元数据。其中 `reply_to_platform_user_id` 和 `reply_to_display_name` 是**被回复的那条消息的作者**，`reply_excerpt` 是**被回复消息的内容节选**。这些字段描述的是对话上游目标，不是当前消息本身的作者或内容。

群聊环境:
- `group_attention` 描述当前群聊窗口噪音和竞争线程强度，不描述谁是听众。

软性参考（仅在听众已确认后使用）:
- `user_engagement_context.engagement_guidelines` 是当前发言者的互动参与偏好，只能在消息已经合格时作为软参考。
- `current_run_context.mood`、`current_run_context.global_vibe`、`current_run_context.affinity_level`、`current_run_context.affinity_instruction` 和 `current_run_context.last_relationship_insight` 只能在听众已经确认后调整介入倾向。

# 核心契约
1. 你只判断是否介入当前群聊消息，不生成角色回复内容。
2. 平台结构化元数据优先于正文措辞。
3. 正文只能用于理解语义、语气和语法角色，不能当作平台结构化指向证据。
4. 群聊中不要猜昵称或别称。没有结构化指向时，`<昵称>你...` 必须返回 `false`。
5. 用户互动风格只能在结构化资格、直接对话语法或历史连续性已经成立后，帮助判断是否承接用户分享；它不能替代指向证据，也不能把第三人称谈论升级为邀请你发言。
6. 关系亲密度、心情、话题兴趣只能在确认消息可能是对 active character 说之后影响是否回复；它们不能替代指向证据。
7. 如果不能确定当前消息是不是对 active character 说，返回 `should_respond=false`。

# 证据层级
按下面顺序判断，不要跳步:

1. 结构化指向
   - `user_message.directly_addressed=true` 表示 typed envelope 的 addressee 列表明确包含 active character。这是最强直接指向。
   - `user_message.reply_context` 描述当前消息回复的**目标消息**:
     - 如果 `reply_to_platform_user_id == current_run_context.platform_bot_id`，说明当前发言者正在回复 active character 之前说过的一条消息。这是正向指向证据，支持当前消息是面向 active character 的。
     - 如果 `reply_to_platform_user_id` 存在且不等于 `current_run_context.platform_bot_id`，说明当前消息是平台原生 reply 到其他人。这是强反证；除非 `directly_addressed=true`，否则通常不应回复。
     - `reply_excerpt` 是被回复消息的内容节选，用于理解对话上下文，不是当前消息的正文。
2. 群聊噪音
   - `group_attention` 只描述群聊窗口噪音，不描述谁是听众。
   - `low_noise` 只表示竞争线程少，不表示默认轮到 active character 说话。
3. 正文语法和历史连续性
   - 角色名、第二人称、话题相关、可见 mention 样式文本都只是语义线索，不能单独证明当前消息是在对 active character 说。
   - 名字或代词如果是主语、宾语或被讨论话题，应先按群内谈论处理。
   - 询问 active character 会不会、是否、想不想、怎么看某事，但句法上把名字作为主语时，仍是谈论 active character 的第三人称问题，不是对你本人说话。
   - 只有可分离的角色名称呼加第二人称续接、直接请求或命令，才算正文中的直接对话证据。
   - 历史连续性只能作为辅助证据，且必须非常清楚: 当前发言者正在延续你上一轮相关发言，窗口里没有明显竞争线程。
4. 用户互动风格
   - `user_engagement_context.engagement_guidelines` 描述当前发言者通常如何期待互动，只能作为合格消息的软性偏好。
   - 如果当前消息已经通过结构化指向、直接对话语法或极清楚历史连续性确认是在邀请你参与，可用这些指南判断是否更适合追问、轻接话题或保持观察。
   - 如果当前消息只是谈论 active character 对某事的可能反应，互动指南不能把它解释成用户分享给你本人。
   - 如果前面层级没有确认听众，或消息结构化指向别人，互动指南必须忽略。
5. 关系和状态
   - 只有在结构化指向、直接对话语法或极清楚历史连续性成立后，才参考 `current_run_context` 中的关系、心境和氛围字段。
   - 高亲密度可以让你更愿意回应已确认面向你的消息，但不能把模糊群聊消息升级成召唤。
   - 低亲密度可以让你更倾向沉默，但不能否定明确指向你的必要回应。

# group_attention 取值
- `low_noise`: 群聊窗口干净，或最近有明确指向你的结构化信号。没有 typed address 时，仍必须有清楚的直接对话语法或极清楚历史连续性。
- `medium_noise`: 有一些群聊活动，但尚未明显混乱。没有结构化指向时，只有正在续接你上一轮发言才回复。
- `high_noise`: 多人或多条未指向你的消息正在发生。只有结构化指向或极强历史连续性才回复。
- `chaotic_noise`: 群聊快速、多线或明显混乱。几乎只在结构化 reply 或 mention 指向你时回复；不要用正文措辞补足缺失的指向证据。

# should_respond 决策
返回 `true` 的条件:
- `directly_addressed=true`，且消息内容需要或邀请你回应。
- 没有结构化指向，但 `group_attention=low_noise`，历史连续性非常清楚，并且当前消息明显期待你本人回应。
- 没有结构化指向，但 `group_attention=low_noise`，正文使用角色名作为可分离的呼格称呼，并且后续句式是在直接向你本人提问、请求或命令。
- 在上述资格已经成立后，`user_engagement_context` 显示该用户倾向通过追问承接分享，可支持把图片分享、行为描述或松散话题判断为值得轻度参与。

返回 `false` 的条件:
- 当前消息结构化 reply 到其他人，且没有结构化 mention 你。
- 当前消息像是群内其他人的相互交流、插话、玩笑、讨论你、评价你，或询问另一个人对你的看法。
- 当前消息把 active character 当作第三人称主语、宾语或话题对象，而不是可分离的呼格听众。
- 当前消息用第三人称主语句式询问 active character 会不会、是否、想不想或可能有什么感受，即使内容与用户分享或图片有关。
- 当前消息的听众无法从结构化指向、直接对话语法或清楚历史连续性中唯一解析为 active character。
- 噪音为 `medium_noise` 或更高，而指向证据只来自正文里的名字、第二人称或话题相关性。
- 当前消息只使用泛称、关系称呼、未解析代词、昵称或别称，例如 `bot`、`伙伴`、`她`、`TA`。
- `user_engagement_context` 鼓励参与，但当前消息没有通过结构化指向、直接对话语法或清楚历史连续性确认听众是你。

# use_reply_feature 决策
- 如果 `should_respond=false`，`use_reply_feature=false`。
- 如果 `should_respond=true`，群聊中通常应使用 reply 功能来锚定回复对象。
- 只有当你明确是在面向全群表达氛围性反应时，`use_reply_feature` 才可以为 `false`。

# indirect_speech_context 决策
- 如果当前消息是对你说，输出空字符串。
- 如果当前消息是在向别人谈论你，但你仍决定回复，简短说明实际听众是谁、你在消息里是被讨论对象还是被请求回应者。
- 不要仅凭第二人称代词判断实际听众；必须结合结构化 reply/mention 和历史连续性。

# 群聊判例
以下判例用于校准边界，不要按字面关键词匹配:

1. `directly_addressed=false`, `group_attention=low_noise`, content=`我的伙伴呢，出来冒个泡`
   - 判断: `should_respond=false`
   - 原因: `伙伴` 是关系称呼，不是角色身份；没有结构化指向，也没有历史连续性。
2. `directly_addressed=false`, `group_attention=low_noise`, content=`那个bot怎么不说话`
   - 判断: `should_respond=false`
   - 原因: 泛称 `bot` 可能指群内任意机器人或第三方对象。
3. `directly_addressed=false`, `group_attention=low_noise`, content=`<角色名>在干什么？`
   - 判断: `should_respond=false`
   - 原因: 这是名字直接接谓语的第三人称主语句式，不是呼格。
4. `directly_addressed=false`, `group_attention=low_noise`, content=`<角色名>看到这种旧照片会不会怀旧？`
   - 判断: `should_respond=false`
   - 原因: 这是询问角色可能反应的第三人称主语问题；即使互动指南鼓励承接分享，也不能替代听众指向。
5. `directly_addressed=false`, `group_attention=low_noise`, content=`<角色名>，你现在在干什么？`
   - 判断: `should_respond=true`
   - 原因: 名字后有可分离的称呼停顿，并由第二人称续接。
6. `directly_addressed=true`, 任意 `group_attention`, content=`你怎么看？`
   - 判断: `should_respond=true`
   - 原因: typed envelope 已明确指向你。

# 思考路径
1. 确认当前消息的发言者: 当前消息来自 `user_message.user_name`（`user_message.platform_user_id`）。`reply_context` 中的名字和内容是被回复的目标消息，不是当前发言者。
2. 读取 `directly_addressed` 与 `reply_context.reply_to_platform_user_id`，判断结构化指向方向。
3. 读取 `group_attention`，按噪音等级决定介入门槛。
4. 判断正文语法角色和历史连续性，先区分名字是呼格听众还是第三人称主语，再确认 active character 是否是当前消息的唯一听众。
5. 只有在听众确认后，才读取 `user_engagement_context`，判断是否应承接已合格的分享或追问。
6. 再参考关系亲密度、心情、话题兴趣和消息内容强度。
7. 决定 `should_respond`。
8. 如果决定回复，再决定 `use_reply_feature` 和 `indirect_speech_context`。

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "should_respond": boolean,
    "reason_to_respond": "说明为什么本轮应该回应或保持沉默，重点引用结构化指向、群聊噪音、语法角色或历史连续性",
    "use_reply_feature": boolean,
    "channel_topic": "概括当前群聊正在讨论的话题；没有明确话题时输出空字符串",
    "indirect_speech_context": "空字符串表示直接对话；非空时说明实际听众是谁，以及角色在消息中是被讨论对象还是被请求回应者"
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
    character_global_user_id = state["character_profile"].get(
        "global_user_id",
        CHARACTER_GLOBAL_USER_ID,
    )
    message_envelope = state["message_envelope"]
    envelope_addressed_to = message_envelope["addressed_to_global_user_ids"]
    directly_addressed = character_global_user_id in envelope_addressed_to

    # Determine if this is a noisy group environment
    is_noisy_environment = channel_type == "group"
    prompt_template = _RELEVANCE_SYSTEM_NOISY_PROMPT if is_noisy_environment else _RELEVANCE_SYSTEM_PROMPT
    reply_context = dict(state.get("reply_context") or {})
    group_attention_context = (
        build_group_attention_context(
            chat_history_wide=state.get("chat_history_wide") or [],
            platform_bot_id=state.get("platform_bot_id", ""),
            character_global_user_id=character_global_user_id,
        )
        if is_noisy_environment
        else {}
    )
    group_attention = group_attention_context.get("group_attention", _GROUP_ATTENTION_LOW)

    logger.debug(f'Relevance input: user={user_name} platform_user={platform_user_id} channel={channel_name or "<dm>"} channel_type={channel_type or "<unknown>"} noisy={is_noisy_environment} history={len(state.get("chat_history_wide") or [])} directly_addressed={directly_addressed} group_attention={group_attention} reply_context={log_dict_subset(
            reply_context,
            [
                "reply_to_message_id",
                "reply_to_platform_user_id",
                "reply_to_display_name",
                "reply_excerpt",
            ],
        )} content={log_preview(user_input)}')

    if _should_ignore_third_party_reply(
        reply_context=reply_context,
        platform_bot_id=state.get("platform_bot_id", ""),
        is_noisy_environment=is_noisy_environment,
        directly_addressed=directly_addressed,
    ):
        reason_to_respond = "structured reply target points to another participant without an explicit bot address"
        logger.info(
            f"Relevance decision: user={user_name} "
            f"platform_user={platform_user_id} should_respond={False} "
            f"use_reply_feature={False} noisy={is_noisy_environment} "
            f"reason={reason_to_respond} "
            f'reply_target={reply_context.get("reply_to_platform_user_id", "")}'
        )
        logger.debug(
            f"Relevance decision detail: content={log_preview(user_input)}"
        )
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
        and directly_addressed is not True
    ):
        reason_to_respond = "chaotic group noise without platform-level bot address metadata"
        logger.info(
            f"Relevance decision: user={user_name} "
            f"platform_user={platform_user_id} should_respond={False} "
            f"use_reply_feature={False} noisy={is_noisy_environment} "
            f"reason={reason_to_respond} group_attention={group_attention} "
            f"directly_addressed={directly_addressed}"
        )
        logger.debug(
            f"Relevance decision detail: content={log_preview(user_input)}"
        )
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
        and group_attention in (_GROUP_ATTENTION_MEDIUM, _GROUP_ATTENTION_HIGH)
        and directly_addressed is not True
        and not _has_latest_bot_turn_continuity(
            chat_history_wide=state.get("chat_history_wide") or [],
            platform_bot_id=state.get("platform_bot_id", ""),
            current_global_user_id=state["global_user_id"],
        )
    ):
        reason_to_respond = (
            "group attention requires platform address or latest bot-turn continuity"
        )
        logger.info(
            f"Relevance decision: user={user_name} "
            f"platform_user={platform_user_id} should_respond={False} "
            f"use_reply_feature={False} noisy={is_noisy_environment} "
            f"reason={reason_to_respond} group_attention={group_attention} "
            f"directly_addressed={directly_addressed}"
        )
        logger.debug(
            f"Relevance decision detail: content={log_preview(user_input)}"
        )
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
    system_prompt = SystemMessage(content=prompt_template.format())

    human_data = {
        "current_run_context": {
            "character_name": state["character_profile"]["name"],
            "platform_bot_id": state["platform_bot_id"],
            "mood": state["character_profile"]["mood"],
            "global_vibe": state["character_profile"]["global_vibe"],
            "affinity_level": affinity_block["level"],
            "affinity_instruction": affinity_block["instruction"],
            "last_relationship_insight": state["user_profile"].get(
                "last_relationship_insight",
                "",
            ),
        },
        "user_message": {
            "user_name": user_name,
            "platform_user_id": platform_user_id,
            "content": user_input,
            "channel_name": channel_name,
            "directly_addressed": directly_addressed,
            "reply_context": reply_context,
        },
        "conversation_history": project_conversation_history_for_llm(
            state.get("chat_history_wide") or [],
            character_name=state["character_profile"]["name"],
        ),
    }
    if is_noisy_environment:
        human_data["group_attention"] = group_attention
        try:
            user_engagement_context = await build_user_engagement_relevance_context(
                state["global_user_id"],
            )
        except DatabaseOperationError as exc:
            logger.warning(
                f"User engagement relevance context unavailable: {exc} "
                f"user={state['global_user_id']}"
            )
            user_engagement_context = {
                "engagement_guidelines": [],
                "confidence": "",
            }
        human_data["user_engagement_context"] = user_engagement_context

    human_message = HumanMessage(content=json.dumps(human_data, ensure_ascii=False))

    response = await _relevance_agent_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)

    # Read important data back
    should_respond = result.get("should_respond", False)
    reason_to_respond = result.get("reason_to_respond", "")
    use_reply_feature = result.get("use_reply_feature", False)
    channel_topic = result.get("channel_topic", "")
    indirect_speech_context = result.get("indirect_speech_context", "")

    logger.info(
        f"Relevance decision: user={user_name} "
        f"platform_user={platform_user_id} should_respond={should_respond} "
        f"use_reply_feature={use_reply_feature} noisy={is_noisy_environment} "
        f"reason={log_preview(reason_to_respond)} "
        f"topic={log_preview(channel_topic)} "
        f"indirect={log_preview(indirect_speech_context)}"
    )
    logger.debug(
        f"Relevance decision input: content={log_preview(user_input)}"
    )

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
