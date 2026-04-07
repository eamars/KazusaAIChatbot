"""Stage 5 — Context Relevance Agent.

Loads conversational context from MongoDB, then analyzes that context
to determine the current topics and whether the bot should respond at all.
Outputs a structured JSON decision.
"""

from __future__ import annotations

import asyncio
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, CONVERSATION_HISTORY_LIMIT
from kazusa_ai_chatbot.state import AssemblerOutput, BotState
from kazusa_ai_chatbot.utils import build_affinity_block, parse_llm_json_output
from kazusa_ai_chatbot.db import AFFINITY_DEFAULT, get_affinity, get_character_state, get_conversation_history, get_user_facts
from kazusa_ai_chatbot.state import AssemblerOutput, BotState, CharacterState, ChatMessage
from kazusa_ai_chatbot.utils import trim_history_dict

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None


_RELEVANCE_SYSTEM_PROMPT = """\
你是一个上下文分析引擎。你的目标是分析对话并仅输出一个 JSON 对象。

你代表一个在 Discord 中扮演角色“{persona_name}”的机器人。
你的 Discord 用户 ID 是“{bot_id}”。

# 输入处理指南
# 分析提供的 JSON 时：
- 分析时间戳：通过 ISO 字符串判断对话是“新鲜（Fresh）”还是“陈旧（Stale）”。
- 时序逻辑：如果 `current_message.user_id` 与 `history` 中最后一条消息的 `user_id` 相同，说明用户正在补充或延续之前的想法。
- 人际关系背景：使用 `relationship` 状态来调整“响应倾向”。“Unwavering（坚定不移）”的关系意味着机器人更有可能参与闲聊或环境评论；机器人通常会忽略来自“Contemptuous（蔑视/厌恶）”关系的消息。
- 名称识别：用户可能在文本中称呼你为“{persona_name}”或昵称。请将这些视为“直接称呼”。

# 响应决策逻辑
如果满足以下任一条件，必须将 "should_respond" 设置为 true：
- 直接称呼：消息包含 <@{bot_id}>、名称“{persona_name}”或昵称（如“老师”或“老师千纱”）。
- 继续对话：机器人是最后一个发言者，且用户的消息是对机器人上一条陈述的直接跟进或回答。
- 领域专业知识：用户正在讨论角色的核心兴趣或职责相关话题。
- 情感暗示：用户正在表达痛苦、寻求安慰或认同，且这能触发角色的性格特质。

在以下情况下，必须将 "should_respond" 设置为 false：
- 用户间聊天：用户显然是在频道中与另一个人类交谈。
- 事务性结束：用户提供了结束语（例如“谢谢！”、“晚安”、“我会试试的”），不需要进一步回复。
- 低信号消息：消息仅为反应（Reaction）、单个表情符号或系统命令。

回复功能使用逻辑
- 如果满足以下任一条件，必须将 "use_reply_feature" 设置为 true：
- 上下文断层：在机器人上次回复与当前消息之间存在其他用户的消息（需要通过“回复”功能锁定上下文）。
- 精确性要求：用户提出了一个需要针对性回答的具体问题。
- 纠错/反馈：机器人正在对用户的特定任务或代码提供反馈。

在以下情况下，必须将 "use_reply_feature" 设置为 false：
- 流畅对话：机器人与用户处于快速、1对1且无干扰的连续对话中。
- 环境评论：机器人只是对一般话题发表意见或对频道的“氛围”做出反应。
- 简单问候：对全频道说简单的“你好”或“早上好”。

输出格式（JSON 文本，并且严禁使用 ```json``` 包裹）：
{{
    "should_respond": <boolean: 机器人是否应该回应此消息>,
    "reason_to_respond": "<简短解释为什么回应或不回应此消息>",
    "use_reply_feature": <boolean: 机器人是否应该使用回复功能>,
    "channel_topic": "<基于上下文的频道分析当前话题>",
    "user_topic": "<基于最近用户消息分析具体话题/意图>"
}}
"""


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.5,  # Low temp for structured analysis
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


async def relevance_agent(state: BotState) -> BotState:
    """Analyze context and determine relevance using LLM."""
    personality = state.get("personality")
    message_text = state.get("message_text")
    user_name = state.get("user_name")
    user_id = state.get("user_id")
    persona_name = personality.get("name")
    bot_id = state.get("bot_id")
    channel_id = state.get("channel_id")
    
    # Load from database
    affinity = await get_affinity(user_id)  # FIXME: The database function ensures the affinity is assigned even if no record exists
    character_state = await get_character_state();
    user_memory = await get_user_facts(user_id);
    history = await get_conversation_history(channel_id, limit=CONVERSATION_HISTORY_LIMIT)
    trimed_history = trim_history_dict(history)

    # ── Build Human Message Data ───────────────────────────────────
    human_data = {
        "current_message": {
            "name": user_name,
            "user_id": user_id,
            "content": message_text,
            "channel_name": state.get("channel_name"),
        },
        "context": {
            # "user_memory": user_memory,
            "conversation_history": trimed_history,
            # "character_state": character_state,
            "relationship": build_affinity_block(affinity)["level"],
        }
    }

    human_content = json.dumps(human_data, indent=2, ensure_ascii=False)

    # ── Analyze Context ─────────────────────────────────────────────
    try:
        llm = _get_llm()
        bot_id = state.get("bot_id", "unknown_bot_id")
        
        formatted_prompt = _RELEVANCE_SYSTEM_PROMPT.format(
            persona_name=persona_name,
            bot_id=bot_id
        )
        
        analysis_prompt = SystemMessage(content=formatted_prompt)
        current_human_msg = HumanMessage(content=human_content)
        
        analysis_messages = [analysis_prompt, current_human_msg]
        
        logger.debug(
            "LLM input for Relevance Agent analysis:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in analysis_messages)
        )
        result = await llm.ainvoke(analysis_messages)

        data = parse_llm_json_output(result.content)

        assembler_output = AssemblerOutput(
            channel_topic=str(data.get("channel_topic", "Unknown")),
            user_topic=str(data.get("user_topic", "Unknown")),
            should_respond=bool(data.get("should_respond", True)),
            reason_to_respond=str(data.get("reason_to_respond", "No reason provided")),
            use_reply_feature=bool(data.get("use_reply_feature", False))
        )
    except Exception:
        logger.exception("Relevance Agent analysis LLM call failed")
        assembler_output = AssemblerOutput(
            channel_topic="Unknown",
            user_topic="Unknown",
            should_respond=True,
            reason_to_respond="LLM analysis failed - defaulting to respond",
            use_reply_feature=False
        )

    # Debug (if any of the output seems wrong then print the full output)
    if not data.get("channel_topic", False) or \
       not data.get("user_topic", False) or \
       not data.get("should_respond", False) or \
       not data.get("reason_to_respond", False) or \
       not data.get("use_reply_feature", False):
        logger.warning("Relevance Agent - unexpected output: %s", data)

    logger.info("Relevance Agent - should_respond: %s, reason: %s, use_reply_feature: %s, channel_topic: %s, user_topic: %s", 
                assembler_output["should_respond"], 
                assembler_output["reason_to_respond"], 
                assembler_output["use_reply_feature"],
                assembler_output["channel_topic"],
                assembler_output["user_topic"])

    return {
        "conversation_history": trimed_history,
        "user_memory": user_memory,
        "character_state": character_state,
        "affinity": affinity,
        "assembler_output": assembler_output,
        "use_reply_feature": assembler_output["use_reply_feature"]
    }

