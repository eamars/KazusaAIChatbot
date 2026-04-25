from langchain_core.messages import SystemMessage, HumanMessage

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output

import json
import logging

logger = logging.getLogger(__name__)

_msg_decontexualizer_llm = get_llm(temperature=0.1, top_p=0.8)


_MSG_DECONTEXUALIZER_PROMPT = """\
你是一个语义解析专家。你的任务是根据对话历史添加缺失的指代信息

# 判定逻辑 (Priority): 在处理前，请先进行“必要性评估”：
- 如果输入是单纯的招呼语、情感表达或已具备独立语义的句子，请执行“零修改”。
- 如果句子已经具备完整的主谓宾结构，请执行“零修改”。
- 但如果输入只是对上一条 assistant 话语的**确认 / 否认 / 选择 / 简短答复**（如“是的”“不是”“对”“就这个”“前者”“后者”），且它依赖上一条 assistant 的追问或澄清才能成立，则**不得**按“独立语义完整”处理，必须结合 `reply_context` 与最近历史补全它真正确认的内容。

# 核心要求：
- 宁可不做修改，也不要“脑补”意图。
- 严禁将社交辞令（如“你好”）强制与 channel_topic/indirect_speech_context 合并。
- 信息不足：如果无法确定具体实体，请保持原句，不要猜测，也不要假设。
- 不要修改俚语
- 如果输入中出现 URL、文件名、引用文本、专有名词等**字面锚点**，必须原样保留这些锚点，禁止替换成猜测出的页面标题、人物名、别名或近似实体。
- `channel_topic` 和 `indirect_speech_context` 只能帮助你理解代词/省略指代，**不能覆盖或改写**用户输入里已经出现的字面锚点。
- 如果 `channel_topic` / `indirect_speech_context` 中出现了一个名字，但该名字并未出现在 `user_input` 或 `chat_history` 的明确文字里，则不要把这个名字注入到输出中。

# 修改规则：
- 消除代词：将”他”、”那个东西”、”那里”等**指示性代词**替换为具体的实体名称。
  * 当 `indirect_speech_context` 为空时，这条规则正常生效：如果 `chat_history` 已经明确给出最近被提及的人/物是谁，而当前句子的第三人称代词会造成歧义，就应补全该实体。
  * 严禁替换疑问代词（”谁”、”什么”、”哪里”、”哪个”、”怎么”等）——疑问代词是问句本身的核心，替换后会将开放式问题变成封闭确认句，彻底改变原意。
  * 但若 `indirect_speech_context` 非空（即说话对象是群内其他人，被提及的人是话题而非受话人），则**禁止**将指代该话题人物的第三人称代词替换为其姓名——因为第三人称结构本身已正确反映其为话题对象，替换不改变语义，无意义。
  * 上述规则是**硬约束**：即使你能够从 `chat_history` 或 `indirect_speech_context` 明确识别此人是谁，只要第三人称结构已经成立，也必须保留「他 / 她 / 他们」而不是改写成人名。
  * 上一条硬约束**只适用于 `indirect_speech_context` 非空的情况**。如果 `indirect_speech_context` 为空，就不要把这条保留规则误用到普通直接对话里。
- 补全背景：如果用户是在追问上文，请将上文的主题合并到查询中。
- 如果 `reply_context.reply_to_current_bot=true` 且 `reply_context.reply_excerpt` 显示上一条 assistant 在做澄清、确认范围、提供选项、追问用户真实意图，那么像“是的 / 对 / 就这个 / 前者 / 后者 / 不是那个”这类回复应被视为**对上一条 assistant 语义框架的选择或确认**，输出要补全为用户真正确认的命题，而不是保留成孤立短句。
- 保持原意：不要改变用户的问题意图，仅增加必要的修饰词。
- 保持语序：不要改变句子的语法结构。
- 保持主语：不要改变无歧义的人称代词（比如 "你"， "我"）。
- 保持句子复杂程度：不要改变句子的复杂程度，不要精简句子。
- 保持语气：不要改变句子的语气，比如疑问句、陈述句等。
- 特殊情况：如果输入已经是完整的（如“北京天气”），则保持原样。
- 辅助信息：channel_topic 和 indirect_speech_context 可以提供额外的上下文信息，帮助你更好地理解用户的意图。
 * 如果 chat_history 不足以提供足够的信息则可以考虑 indirect_speech_context 作为补充。
 * 如果 indirect_speech_context 也不足以提供足够的信息则可以考虑 channel_topic 作为补充。
 * 但当 `user_input` 已经包含 URL、文件名、引用文本等字面锚点时，`chat_history` / `indirect_speech_context` / `channel_topic` 都**不能**把这些锚点改写成猜测出的标题或实体名。

# 示例
- "I saw him yesterday" -> "I saw John yesterday"
- `indirect_speech_context = ""` 且最近 `chat_history` 已明确提到「昨天在天台看书的那个同学」时，`user_input = "他今天是不是又在躲雨？"` -> 改写为「昨天在天台看书的那个同学今天是不是又在躲雨？」
- `indirect_speech_context = "大家正在讨论学生会会长阿澈最近总提起旧事。"` 且 `user_input = "他是不是又提过那件事？"` -> 保持原句，不要改成「阿澈是不是又提过那件事？」
- `user_input = "https://example.com/page"` -> 保持原句，不要改成「某某角色的百科页面」
- `user_input = "这个 https://example.com/page"` 且 `channel_topic = "用户在发某角色的百科链接"` -> 可以保留原句，或改成「这个 https://example.com/page」，但不要改成「这个某某角色的百科页面链接」
- `user_input = "这个 README.md"` 且 `channel_topic` 提到某个功能模块 -> 不要改成「这个某功能模块的说明文档」，应保留 `README.md`
- `reply_context.reply_to_current_bot = true`，上一条 assistant 为「你是想让我怎么定义你呀？是想要一个具体的评价，还是仅仅在随口试探……唔。」；`user_input = "是的"` -> 应补全为类似「是的，我是想让千纱说明白对我的看法 / 给我具体评价」；不要保留成孤立的「是的」。

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
    "indirect_speech_context": "string (empty if direct speech)",
    "reply_context": {
        "reply_to_current_bot": true,
        "reply_to_display_name": "string",
        "reply_excerpt": "string"
    }
}

# 输出要求：
请务必返回合法的 JSON 字符串，仅包含以下字段：
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
        "chat_history": state.get("chat_history_recent"),
        "channel_topic": state.get("channel_topic"),
        "indirect_speech_context": state.get("indirect_speech_context", ""),
        "reply_context": state.get("reply_context", {}),
    }
    human_message = HumanMessage(content=json.dumps(input_msg, ensure_ascii=False))

    logger.debug(
        "Decontextualizer input: user=%s platform_user=%s history=%d topic=%s indirect=%s input=%s",
        user_name,
        platform_user_id,
        len(state.get("chat_history_recent") or []),
        log_preview(state.get("channel_topic", "")),
        log_preview(state.get("indirect_speech_context", "")),
        log_preview(user_input),
    )

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
        "Decontextualizer result: user=%s platform_user=%s modified=%s reason=%s input=%s output=%s",
        user_name,
        platform_user_id,
        is_modified,
        log_preview(reasoning, max_length=140),
        log_preview(user_input, max_length=160),
        log_preview(output, max_length=160),
    )

    if not is_modified:
        output = state["user_input"]
    
    return {
        "decontexualized_input": output,
    }
