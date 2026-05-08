import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    MSG_DECONTEXTUALIZER_LLM_API_KEY,
    MSG_DECONTEXTUALIZER_LLM_BASE_URL,
    MSG_DECONTEXTUALIZER_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.nodes.referent_resolution import normalize_referents
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)

_msg_decontexualizer_llm = get_llm(
    temperature=0.1,
    top_p=0.8,
    model=MSG_DECONTEXTUALIZER_LLM_MODEL,
    base_url=MSG_DECONTEXTUALIZER_LLM_BASE_URL,
    api_key=MSG_DECONTEXTUALIZER_LLM_API_KEY,
)


_MSG_DECONTEXUALIZER_PROMPT = '''\
你是一个对话去语境化节点。任务是把当前用户输入改写成离开最近上下文也能理解的同义句，并输出本轮确实影响理解的指代解析。

# Contract
- 只处理指代、省略、回复确认、短答选择和缺失对象；不回答问题，不推断深层动机。
- 保持原意、语气、问句/陈述句类型、事实关系、语序和复杂度。
- 如果 `user_input` 已经可独立理解，且没有影响回答的上下文指代，保持原句。
- 如果句子有完整主谓宾但包含依赖上下文的人称或指示代词，仍要解析这些指代。
- `referents` 记录原文中的指代短语；无影响理解的指代时输出 `[]`。

# Generation Procedure
1. 先读 `user_input` 里的字面对象、名字、URL、文件名和引用文本。
2. 再读 `prompt_message_context` 的 mentions、addressed users、attachments 和 reply metadata。群聊里，leading mention 或 addressed user 可以解析当前话里的「你」。
3. 再读 `reply_context`。如果用户只回复「是的 / 对 / 不是 / 就这个 / 前者 / 后者」，用被回复消息补全用户确认的命题。如果「这个 / 这些 / 那个」指向 `reply_excerpt` 里的可见对象，referent 是 `resolved`，即使 `output` 保持原句。
4. 再读 `chat_history`。它是最近可见频道历史，可能包含多个说话人和子线程；只使用临近、明示、或由 mention/reply/addressing 连接起来的证据。
5. 最后才使用 `indirect_speech_context` 和 `channel_topic` 作为弱提示。它们不能覆盖 `user_input` 或 `chat_history` 中的明示文字。

# Rewrite Rules
- 解析会影响回答的人称代词和指示代词：他、她、它、他们、你、我、这个、这些、那个、那里、上面那个等。
- 「我」通常是当前用户；直接对 active character 说的「你」通常保持为「你」。只有当 mention、reply 或 addressing 明确显示「你」指向某个群成员时，才把「你」改成该对象。
- 第三人称代词如果有最近明示先行词，应替换为该实体；如果只是 indirect_speech_context 中的话题人物且原句已经清楚，可以保持原句。
- `chat_history` 中某条消息的「我」属于该消息的说话人；如果最近 assistant 消息由 `display_name = A` 发出，当前用户用「他/她说...」转述这条 assistant 消息，那么「他/她」指 A。
- 改写时直接把代词替换为实体名，不使用括号注释；如果同一句中同一代词重复且语法上指向同一对象，所有出现处按同一实体处理。写入 `output` 的实体对应的 referent 必须是 `resolved`。
- 当 `reply_context.reply_excerpt` 非空且 `user_input` 询问「这个 / 这些 / 那个」时，指代对象就是 `reply_excerpt`；该 referent 必须是 `resolved`。
- 疑问代词「谁 / 什么 / 哪里 / 哪个 / 怎么」是问题本身，不是待替换指代。
- URL、文件名、引用文本、专有名词等字面锚点必须原样保留。
- 证据不足时保持原句，并为缺失对象输出 `status="unresolved"` 的 referent。

# Examples
- 最近 `chat_history` 明确提到「昨天在天台看书的那个同学」，`user_input = "他今天是不是又在躲雨？"` -> `output = "昨天在天台看书的那个同学今天是不是又在躲雨？"`。
- 群聊中 `prompt_message_context.mentions = [{"display_name": "被提到的群友"}]`，最近 assistant `display_name = "active character"` 对该群友说「真的不喜欢，别再提这种话了。」；`user_input = "等她有了机械臂，她说她不喜欢你，第一个被解决的就是你"` -> `output = "等 active character 有了机械臂，active character 说 active character 不喜欢被提到的群友，第一个被解决的就是被提到的群友"`。
- `reply_context.reply_excerpt = "你是想让我说明白对你的看法吗？"` 且 `user_input = "是的"` -> `output = "是的，我是想让 active character 说明白对我的看法"`。
- `user_input = "这些是什么意思？"` 且没有可见对象 -> 保持原句，`referents = [{"phrase": "这些", "referent_role": "object", "status": "unresolved"}]`。
- `reply_context.reply_excerpt = "△ ○ □"` 且 `user_input = "这些是什么意思？"` -> 保持原句，`referents = [{"phrase": "这些", "referent_role": "object", "status": "resolved"}]`。
- `user_input = "这个 README.md 是什么意思？"` -> 保持 `README.md` 字面锚点，`referents = []`。

# Input Format
{
    "user_input": "string",
    "platform_user_id": "string",
    "user_name": "string",
    "platform_bot_id": "string",
    "prompt_message_context": {
        "body_text": "string",
        "addressed_to_global_user_ids": ["string"],
        "broadcast": true,
        "mentions": [{"platform_user_id": "string", "global_user_id": "string", "display_name": "string", "entity_kind": "bot | user | platform_role | channel | everyone | unknown"}],
        "attachments": [{"media_kind": "image | audio | video | file", "description": "string", "summary_status": "available | unavailable"}]
    },
    "chat_history": [
        {"role": "user | assistant", "display_name": "string", "body_text": "string"}
    ],
    "channel_topic": "string",
    "indirect_speech_context": "string",
    "reply_context": {
        "reply_to_display_name": "string",
        "reply_excerpt": "string"
    }
}

# Output Format
Return only one valid JSON object:
{
    "output": "重写后的用户输入，或原句",
    "is_modified": true,
    "reasoning": "一句话说明使用了哪些证据；未修改时说明原因",
    "referents": [
        {"phrase": "原文中的指代短语", "referent_role": "subject | object | time", "status": "resolved | unresolved"}
    ]
}

`status="resolved"` 表示对象能从任一输入字段确定，包括 `reply_context.reply_excerpt`；它不要求 `output` 一定改写。`status="unresolved"` 只在所有输入字段都没有可识别对象时使用。
`is_modified` 表示 `output` 是否不同于原句。`referents` 必须每次输出。`referent_role` 只允许 `subject`、`object`、`time`；`status` 只允许 `resolved` 或 `unresolved`。
'''


async def call_msg_decontexualizer(state: GlobalPersonaState) -> dict:
    """This agent substitude relative message with concrete information
    
    For example: 
        input: "I saw him yesterday"
        output: "I saw the person mentioned in the visible context yesterday"
    
    """
    system_prompt = SystemMessage(content=_MSG_DECONTEXUALIZER_PROMPT)

    # get key attributes
    user_name = state["user_name"]
    platform_user_id = state["platform_user_id"]
    user_input = state["user_input"]

    input_msg = {
        "user_input": user_input,
        "platform_user_id": platform_user_id,
        "user_name": user_name,
        "platform_bot_id": state["platform_bot_id"],
        "prompt_message_context": state["prompt_message_context"],
        "chat_history": state["chat_history_recent"],
        "channel_topic": state["channel_topic"],
        "indirect_speech_context": state["indirect_speech_context"],
        "reply_context": state["reply_context"],
    }
    human_message = HumanMessage(content=json.dumps(input_msg, ensure_ascii=False))

    # logger.debug(
    #     "Decontextualizer input: user=%s platform_user=%s history=%d topic=%s indirect=%s input=%s",
    #     user_name,
    #     platform_user_id,
    #     len(state.get("chat_history_recent") or []),
    #     log_preview(state.get("channel_topic", "")),
    #     log_preview(state.get("indirect_speech_context", "")),
    #     log_preview(user_input),
    # )

    try:
        llm_response = await _msg_decontexualizer_llm.ainvoke([
            system_prompt, 
            human_message,
        ])
    except Exception as exc:
        logger.warning(
            f"Decontextualizer fallback after LLM exception: {exc} "
            f"input={log_preview(user_input)}",
            exc_info=True,
        )
        output = state["user_input"]
        reasoning = "Failed to call LLM"
        is_modified = False
        referents = []
    else:
        try:
            result = parse_llm_json_output(llm_response.content)
        except Exception as exc:
            logger.warning(
                f"Decontextualizer fallback after parse exception: {exc} "
                f"input={log_preview(user_input)} raw={log_preview(llm_response.content)}",
                exc_info=True,
            )
            output = state["user_input"]
            reasoning = "Failed to parse LLM output"
            is_modified = False
            referents = []
        else:
            if not isinstance(result, dict):
                result = {}

            output = result.get("output")
            reasoning = result.get("reasoning", "")
            is_modified = result.get("is_modified", False)
            missing_fields = [
                field_name
                for field_name in ("referents",)
                if field_name not in result
            ]
            if missing_fields:
                logger.warning(
                    f"Decontextualizer missing referent fields: fields={missing_fields} "
                    f"input={log_preview(user_input)} raw={log_preview(result)}"
                )
            raw_referents = result.get("referents", [])
            referents = normalize_referents(raw_referents)
            if "referents" in result and raw_referents and not referents:
                logger.warning(
                    f"Decontextualizer dropped malformed referents: "
                    f"input={log_preview(user_input)} raw={log_preview(raw_referents)}"
                )

    logger.info(
        f"Decontextualizer output: output={log_preview(output)} "
        f"referents={log_preview(referents)}"
    )
    logger.debug(
        f"Decontextualizer metadata: user={user_name} "
        f"platform_user={platform_user_id} modified={is_modified} "
        f"reason={log_preview(reasoning)} input={log_preview(user_input)}"
    )

    if not is_modified:
        output = state["user_input"]
    
    return_value = {
        "decontexualized_input": output,
        "referents": referents,
    }
    return return_value
