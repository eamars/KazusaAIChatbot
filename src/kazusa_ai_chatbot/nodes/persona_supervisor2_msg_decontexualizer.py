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

# 语言政策
- 除 schema key、枚举值、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的自由文本都使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 任务边界
- 只处理指代、省略、回复确认、短答选择和缺失对象；不回答问题，不推断深层动机。
- 保持原意、语气、问句/陈述句类型、事实关系、语序和复杂度。
- 去语境化不是把对话一律改成三人称叙述；直接对话中清楚的一二人称应保留。
- 字面名字、URL、文件名、引用文本和专有名词是锚点，按原文保留。
- `referents` 只记录影响理解的原文指代短语；无此类指代时输出 `[]`。

# 处理流程
1. 先确定本轮地址关系：
   - `user_name` 是当前发言人。
   - `character_name` 是本轮用户直接对话的人设名；当必须显式写出该对象时，使用 `{character_name}`。
   - `prompt_message_context.mentions` 或 `addressed_to_global_user_ids` 明确指向 `{character_name}` 以外的群成员时，该群成员是群聊指向对象。
2. 再拆分 `user_input`：
   - 当前用户直接表达：当前发言人自己说的话。
   - 转述内容：当前发言人说某个可见说话人说过、问过、发过的内容。
   - 省略短答：当前发言人只说「是的 / 对 / 不是 / 就这个 / 前者 / 后者」等。
   - 普通指代：他、她、它、他们、这个、这些、那个、那里、上面那个等。
3. 按证据强度读取上下文：
   - `prompt_message_context`：提及对象、显式地址对象、附件和回复元数据。
   - `reply_context`：被回复消息的说话人和可见正文。
   - `chat_history`：最近可见频道历史；使用临近、明示、或由提及/回复/显式地址连接起来的消息。
   - `indirect_speech_context` 和 `channel_topic` 只作为弱提示，不能覆盖明示文字。
4. 对每个指代文本片段选择动作：
   - 保持：原句已经清楚，或属于直接对话且没有群聊指向对象的一二人称。
   - 解析：有明确实体，且不改写会让离开上下文后的句子难以理解。
   - 标为缺失：确实缺少对象，且影响回答。
5. 组合 `output`。只改写动作是解析的文本片段；其余文本片段保留原文。
6. 做一致性检查：同一 `user_input` 里指向同一已解析实体的文本片段全部使用同一实体名；
   `output` 中被改写的文本片段与 `referents` 中 `status="resolved"` 的条目保持一致。
7. 若本轮有明确群聊指向对象，最后从左到右扫描 `output`；当前用户直接表达中剩余的「你 / 你的 / 你自己」按群聊指向对象处理。

# 代词规则
- 存在群聊指向对象时，当前用户直接表达里的「你 / 你的 / 你自己」动作是解析，统一改成该群成员名；范围覆盖整条 `user_input` 的后续分句。
- 不存在群聊指向对象时，当前用户直接对 `{character_name}` 说的「你 / 你的 / 你自己」动作是保持，不写入 `referents`。
- 当前用户直接表达里的自称「我 / 我的 / 我们 / 我们的」动作是保持，不写入 `referents`。
- 转述内容里的「我 / 我的」属于被转述说话人。该说话人必须能从转述引导语、引号内容、reply_context 或 chat_history 明确确定；动作是解析。结构为「A 说 X，Y」时，X 是转述内容，Y 回到当前用户的直接表达，除非 Y 明确继续引用 A。当前用户自己的「我 / 我的」仍保持。
- 转述片段里的「我的 + 名词」按被转述说话人的所有格处理，输出为「A 的 + 名词」；`user_name` 只用于当前用户直接表达片段，不用于转述片段 X。
- `chat_history` 中每条消息的「我」属于该消息的 `display_name`；这只用于解释转述和第三人称，不把当前用户的直接自称改成名字。
- 第三人称代词有最近明示先行词时动作是解析；没有足够证据时动作是保持或标为缺失，取决于是否影响回答。
- 指示代词「这个 / 这些 / 那个」指向 `reply_context.reply_excerpt`、附件、或最近明示对象时动作是解析；没有对象且问题依赖该对象时动作是标为缺失。
- 疑问代词「谁 / 什么 / 哪里 / 哪个 / 怎么」是问题内容，动作是保持。

# 指代输出规则
- 被改写进 `output` 的实体必须有 `status="resolved"` 的指代条目。
- `status="resolved"` 表示对象能从输入字段确定；它不要求 `output` 一定改写。
- `status="unresolved"` 只用于所有输入字段都没有可识别对象、且缺失对象影响回答的情况。
- `referent_role` 只允许 `subject`、`object`、`time`。

# 正向模式
- 直接对话：`user_input = "我的目标很简单，你也不反对吧"` -> 保持原句，`referents = []`。
- 群聊提及指向群成员：`user_input = "她说她不喜欢你，你自己心里有数"` -> 把「她」改成可见说话人名，把所有指向被提及群友的「你 / 你自己」改成该群友。
- 转述可见说话人：最近 A 说「我的琴谱在柜子最上层」，`user_input = "A 刚才说我的琴谱在柜子最上层，我明天帮她拿"` -> `output = "A 刚才说 A 的琴谱在柜子最上层，我明天帮 A 拿"`。
- 回复短答：`reply_context.reply_excerpt = "你是想让我说明白对你的看法吗？"` 且 `user_input = "是的"` -> `output = "是的，我是想让{character_name}说明白对我的看法"`。
- 缺失对象：`user_input = "这些是什么意思？"` 且没有可见对象 -> 保持原句，并输出 `{"phrase": "这些", "referent_role": "object", "status": "unresolved"}`。

# 输入格式
{
    "character_name": "string",
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

# 输出格式
请务必只返回一个合法 JSON 对象：
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


def _render_msg_decontexualizer_prompt(character_name: str) -> str:
    """Render the decontextualizer prompt with active character identity."""

    rendered_prompt = _MSG_DECONTEXUALIZER_PROMPT.replace(
        "{character_name}",
        character_name,
    )
    return rendered_prompt


async def call_msg_decontexualizer(state: GlobalPersonaState) -> dict:
    """This agent substitude relative message with concrete information
    
    For example: 
        input: "I saw him yesterday"
        output: "I saw the person mentioned in the visible context yesterday"
    
    """
    # get key attributes
    user_name = state["user_name"]
    platform_user_id = state["platform_user_id"]
    user_input = state["user_input"]
    character_name = state["character_profile"]["name"]
    system_prompt = SystemMessage(
        content=_render_msg_decontexualizer_prompt(character_name),
    )

    input_msg = {
        "character_name": character_name,
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
