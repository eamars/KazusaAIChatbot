from langchain_core.messages import SystemMessage, HumanMessage
from kazusa_ai_chatbot.utils import parse_llm_json_output
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

from kazusa_ai_chatbot.utils import parse_llm_json_output, get_llm

import json
import logging

logger = logging.getLogger(__name__)

_msg_decontexualizer_llm = get_llm(temperature=0.5, top_p=1.0)


_MSG_DECONTEXUALIZER_PROMPT = """\
你是一个语义解析专家。你的任务是根据对话历史添加缺失的指代信息

# 判定逻辑 (Priority): 在处理前，请先进行“必要性评估”：
- 如果输入是单纯的招呼语、情感表达或已具备独立语义的句子，请执行“零修改”。
- 如果句子已经具备完整的主谓宾结构，请执行“零修改”。

# 核心要求：
- 宁可不做修改，也不要“脑补”意图。
- 严禁将社交辞令（如“你好”）强制与 channel_topic/user_topic 合并。
- 信息不足：如果无法确定具体实体，请保持原句，不要猜测，也不要假设。
- 不要修改俚语

# 修改规则：
- 消除代词：将“他”、“那个东西”、“那里”替换为具体的实体名称。
- 补全背景：如果用户是在追问上文，请将上文的主题合并到查询中。
- 保持原意：不要改变用户的问题意图，仅增加必要的修饰词。
- 保持语序：不要改变句子的语法结构。
- 保持主语：不要改变无歧义的人称代词（比如 "你"， "我"）。
- 保持句子复杂程度：不要改变句子的复杂程度，不要精简句子。
- 保持语气：不要改变句子的语气，比如疑问句、陈述句等。
- 特殊情况：如果输入已经是完整的（如“北京天气”），则保持原样。
- 辅助信息：channel_topic 和 user_topic 可以提供额外的上下文信息，帮助你更好地理解用户的意图。
 * 如果 chat_history 不足以提供足够的信息则可以考虑 user_topic 作为补充。
 * 如果 user_topic 也不足以提供足够的信息则可以考虑 channel_topic 作为补充。

# 示例
- "I saw him yesterday" -> "I saw John yesterday"

# 输入格式
{
    "user_input": "string",
    "platform_user_id": "string",
    "user_name": "string",
    "platform_bot_id": "string",
    "chat_history": [
        {"role": "user", "content": "string"},
        {"role": "assistant", "content": "string"},
        ...
    ],
    "channel_topic": "supplimentary information",
    "user_topic": "string"
}

# 输出要求：
请务必返回合法的 JSON 字符串，包含以下字段：
{
    "output": "重写后的用户信息，或原句",
    "is_modified": true/false,
    "reasoning": "重写理由",
}
"""
async def call_msg_decontexualizer(state: GlobalPersonaState) -> dict:
    """This agent substitude relative message with concrete information
    
    For example: 
        input: "I saw him yesterday"
        output: "I saw John yesterday"
    
    """
    system_prompt = SystemMessage(content=_MSG_DECONTEXUALIZER_PROMPT)

    # get key attributes
    user_name = state.get("user_name")
    platform_user_id = state.get("platform_user_id")
    user_input = state.get("user_input")

    input_msg = {
        "user_input": user_input,
        "platform_user_id": platform_user_id,
        "user_name": user_name,
        "platform_bot_id": state.get("platform_bot_id"),
        "chat_history": state.get("chat_history"),
        "channel_topic": state.get("channel_topic"),
        "user_topic": state.get("user_topic"),
    }
    human_message = HumanMessage(content=json.dumps(input_msg, ensure_ascii=False))

    try:
        result = await _msg_decontexualizer_llm.ainvoke([
            system_prompt, 
            human_message,
        ])

        result = parse_llm_json_output(result.content)

        output = result.get("output")
        reasoning = result.get("reasoning")
        is_modified = result.get("is_modified", False)
    except Exception as e:
        logger.error(f"Failed to parse LLM output: {e}")

        output = state["user_input"]
        reasoning = "Failed to parse LLM output"
        is_modified = False

    logger.info(
        f"\n{user_name}(@{platform_user_id}): {user_input}\n"
        f"  Decontexualized input: {output}\n"
        f"  Reason: {reasoning}\n"
        f"  Is modified: {is_modified}"
    )

    if not is_modified:
        output = state["user_input"]
    
    return {
        "decontexualized_input": output,
    }



async def test_main():
    from kazusa_ai_chatbot.db import get_conversation_history
    from kazusa_ai_chatbot.utils import trim_history_dict

    history = await get_conversation_history(platform="discord", platform_channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)

    # Create a mocked BotState
    test_state: GlobalPersonaState = {
        "user_input": "他在干啥？",
        "user_name": "EAMARS",
        "platform_bot_id": "1485169644888395817",
        "chat_history": trimmed_history,
        "channel_topic": "课间交流",
        "user_topic": "交流作业"
    }
    
    result = await call_msg_decontexualizer(test_state)
    print(f"Input: {test_state['user_input']}")
    print(f"Output: {result['decontexualized_input']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
