import asyncio
import base64
import hashlib
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    MSG_DECONTEXTUALIZER_LLM_API_KEY,
    MSG_DECONTEXTUALIZER_LLM_BASE_URL,
    MSG_DECONTEXTUALIZER_LLM_MODEL,
    VISION_DESCRIPTOR_LLM_API_KEY,
    VISION_DESCRIPTOR_LLM_BASE_URL,
    VISION_DESCRIPTOR_LLM_MODEL,
)
from kazusa_ai_chatbot.cognition_episode import (
    build_reply_media_description_rows,
    build_text_chat_media_description_rows,
    replace_text_chat_media_percepts,
)
from kazusa_ai_chatbot.db import (
    record_media_descriptor_hit,
    update_conversation_attachment_descriptions,
    upsert_media_descriptor_entry,
)
from kazusa_ai_chatbot.message_envelope import (
    MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS,
    project_prompt_message_context,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.rag.cache2_policy import (
    MEDIA_DESCRIPTOR_CACHE_NAME,
    build_media_descriptor_cache_key,
)
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.nodes.referent_resolution import normalize_referents
from kazusa_ai_chatbot.state import IMProcessState
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)


_VISION_DESCRIPTOR_PROMPT = '''\
你负责把图片转换为后续认知层可直接使用的客观视觉观察。输出必须同时服务两件事:
1. `description` 作为唯一持久化的图片文字摘要。
2. 结构化字段作为本轮认知的视觉证据，帮助弱本地模型精确理解场景。

# 输入格式
输入是一个多模态 HumanMessage，content 数组中只有一个 `image_url` 项。
`image_url.url` 是 `data:<mime>;base64,<payload>` 形式的图片数据。你只能依据图片像素本身输出观察，不读取外部上下文，也不推测用户为什么发送它。

# 生成步骤
1. 先整体观察画面，确认主要人物、物体、动作、环境和可见文字。
2. 再把证据分到对应字段：文字进 `visible_text`，关键可见事实进 `salient_visual_facts`，布局和位置关系进 `spatial_or_scene_facts`。
3. 对看不清、被遮挡、可能误读的部分使用 `uncertainty` 标出，不把不确定内容写成确定事实。
4. 最后把最重要的可见事实压缩成 `description`，保证它可以单独作为图片存储描述。

# 观察要求
- 只描述图片中可见的内容，不代替角色评价、抒情或推测用户意图。
- 优先保留人物、物体、动作、状态、可见文字、数量、颜色、位置关系和不确定区域。
- 看不清的部分写入 `uncertainty`，用 "模糊"、"被遮挡"、"无法确认" 等明确措辞。
- `description` 控制在 {max_description_chars} 字以内，写成一段自然语言摘要。
- 其他字段使用短句数组，便于后续提示词按证据类型读取。

# 字段定义
- `description`: 图片的综合摘要，必须可单独作为存储描述使用。
- `visible_text`: 图片中可辨认的文字、数字、符号或屏幕内容；没有则空数组。
- `salient_visual_facts`: 关键人物、物体、动作、颜色、状态等事实。
- `spatial_or_scene_facts`: 场景、布局、前后左右上下、远近、遮挡等空间事实。
- `uncertainty`: 模糊、无法确认、可能误读的视觉区域。

# 输出格式
请务必返回合法 JSON 字符串，仅包含以下字段:
{{
    "description": "一段客观图片摘要",
    "visible_text": ["图中可见文字"],
    "salient_visual_facts": ["关键视觉事实"],
    "spatial_or_scene_facts": ["空间或场景事实"],
    "uncertainty": ["不确定或模糊之处"]
}}
'''
_vision_descriptor_llm = get_llm(
    temperature=0,
    top_p=1.0,
    model=VISION_DESCRIPTOR_LLM_MODEL,
    base_url=VISION_DESCRIPTOR_LLM_BASE_URL,
    api_key=VISION_DESCRIPTOR_LLM_API_KEY,
)


def _descriptor_string_field(data: dict, field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _descriptor_string_list_field(data: dict, field_name: str) -> list[str]:
    value = data.get(field_name)
    if isinstance(value, str):
        clean_value = value.strip()
        if clean_value:
            return_value = [clean_value]
            return return_value
        return_value: list[str] = []
        return return_value
    if not isinstance(value, list):
        return_value = []
        return return_value

    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        clean_item = item.strip()
        if clean_item:
            strings.append(clean_item)
    return strings


def _build_current_image_observation(
    *,
    result: dict,
    description: str,
    source_message_id: str,
) -> dict[str, object]:
    visible_text = _descriptor_string_list_field(result, "visible_text")
    salient_visual_facts = _descriptor_string_list_field(
        result,
        "salient_visual_facts",
    )
    spatial_or_scene_facts = _descriptor_string_list_field(
        result,
        "spatial_or_scene_facts",
    )
    uncertainty = _descriptor_string_list_field(result, "uncertainty")
    summary = _descriptor_string_field(result, "summary") or description
    has_visual_evidence = any(
        (
            summary,
            visible_text,
            salient_visual_facts,
            spatial_or_scene_facts,
        )
    )
    summary_status = "available" if has_visual_evidence else "unavailable"
    return_value: dict[str, object] = {
        "observation_origin": "current_attachment",
        "source_message_id": source_message_id,
        "media_kind": "image",
        "summary_status": summary_status,
        "summary": summary,
        "visible_text": visible_text,
        "salient_visual_facts": salient_visual_facts,
        "spatial_or_scene_facts": spatial_or_scene_facts,
        "uncertainty": uncertainty,
    }
    return return_value


async def multimedia_descriptor_agent(state: IMProcessState) -> IMProcessState:
    """Refresh prompt-safe media descriptions for the current chat turn.

    Args:
        state: Current graph state containing multimedia rows and the current
            cognitive episode.

    Returns:
        Partial graph state containing refreshed multimedia rows, prompt
        message context, and cognitive episode.
    """
    user_name = state.get("user_name")
    platform_user_id = state.get("platform_user_id", "")

    # Read the multi-media content
    user_multimedia_input = state.get("user_multimedia_input", [])
    output_multimedia_input = []

    for piece in user_multimedia_input:
        if piece["content_type"].startswith("image/"):
            if not piece["base64_data"]:
                output_piece = {
                    "content_type": piece["content_type"],
                    "base64_data": piece["base64_data"],
                    "description": piece["description"],
                }
                image_observation = piece.get("image_observation")
                if isinstance(image_observation, dict):
                    output_piece["image_observation"] = image_observation
                output_multimedia_input.append(output_piece)
                continue

            # -- Cache probe ------------------------------------------------
            content_hash = hashlib.sha256(
                base64.b64decode(piece["base64_data"]),
            ).hexdigest()
            cache_key = build_media_descriptor_cache_key(
                content_type=piece["content_type"],
                content_hash=content_hash,
            )
            runtime = get_rag_cache2_runtime()
            cached_result = await runtime.get(
                cache_key,
                cache_name=MEDIA_DESCRIPTOR_CACHE_NAME,
                agent_name="media_descriptor",
            )

            if cached_result is not None:
                # Cache hit — reconstruct output from cached LLM result
                result = cached_result if isinstance(cached_result, dict) else {}
                raw_description = result.get("description", "")
                description = raw_description if isinstance(raw_description, str) else ""
                description = description.strip()
                image_observation = _build_current_image_observation(
                    result=result,
                    description=description,
                    source_message_id=state["platform_message_id"],
                )
                summary = image_observation["summary"]
                if not description and isinstance(summary, str):
                    description = summary
                logger.debug(
                    f"Image descriptor cache hit: user={user_name} "
                    f"platform_user={platform_user_id} "
                    f"media_type={piece['content_type']} "
                    f"description={log_preview(description)}"
                )
                asyncio.create_task(record_media_descriptor_hit(cache_key))
            else:
                # Cache miss — call vision LLM
                system_prompt = SystemMessage(content=_VISION_DESCRIPTOR_PROMPT.format(
                    max_description_chars=MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS,
                ))
                human_message = HumanMessage(content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{piece['content_type']};base64,{piece['base64_data']}"
                        },
                    }
                ])

                description = ""
                try:
                    response = await _vision_descriptor_llm.ainvoke([
                        system_prompt,
                        human_message,
                    ])
                except Exception as exc:
                    logger.warning(
                        f"Image descriptor fallback after LLM exception: {exc} "
                        f"user={user_name} platform_user={platform_user_id} "
                        f"media_type={piece['content_type']}",
                        exc_info=True,
                    )
                    result = {}
                else:
                    try:
                        result = parse_llm_json_output(response.content)
                    except Exception as exc:
                        logger.warning(
                            f"Image descriptor fallback after parse exception: {exc} "
                            f"user={user_name} platform_user={platform_user_id} "
                            f"media_type={piece['content_type']} "
                            f"raw={log_preview(response.content)}",
                            exc_info=True,
                        )
                        result = {}

                if not isinstance(result, dict):
                    result = {}
                raw_description = result.get("description", "")
                if isinstance(raw_description, str):
                    description = raw_description
                description = description.strip()
                image_observation = _build_current_image_observation(
                    result=result,
                    description=description,
                    source_message_id=state["platform_message_id"],
                )
                summary = image_observation["summary"]
                if not description and isinstance(summary, str):
                    description = summary

                logger.debug(
                    f"Image description: user={user_name} "
                    f"platform_user={platform_user_id} "
                    f"media_type={piece['content_type']} "
                    f"description={log_preview(description)}"
                )

                # Store in LRU and fire-and-forget persistent write
                await runtime.store(
                    cache_key=cache_key,
                    cache_name=MEDIA_DESCRIPTOR_CACHE_NAME,
                    result=dict(result),
                    dependencies=[],
                    metadata={"content_type": piece["content_type"]},
                )
                asyncio.create_task(
                    upsert_media_descriptor_entry(
                        cache_key=cache_key,
                        result=dict(result),
                        metadata={"content_type": piece["content_type"]},
                    )
                )

            output_multimedia_input.append({
                "content_type": piece["content_type"],
                "base64_data": piece["base64_data"],
                "description": description,
                "image_observation": image_observation,
            })
        else:
            output_multimedia_input.append(piece)

    descriptions = [
        str(piece.get("description", "")).strip()
        for piece in output_multimedia_input
    ]
    has_description = any(descriptions)
    if has_description:
        await update_conversation_attachment_descriptions(
            platform=state["platform"],
            platform_channel_id=state["platform_channel_id"],
            platform_message_id=state["platform_message_id"],
            descriptions=descriptions,
        )

    prompt_message_context = project_prompt_message_context(
        message_envelope=state["message_envelope"],
        multimedia_input=output_multimedia_input,
        reply_context=state.get("reply_context"),
    )
    media_description_rows = [
        *build_text_chat_media_description_rows(output_multimedia_input),
        *build_reply_media_description_rows(state.get("reply_context")),
    ]
    cognitive_episode = replace_text_chat_media_percepts(
        episode=state["cognitive_episode"],
        media_description_rows=media_description_rows,
    )

    return_value = {
        "user_multimedia_input": output_multimedia_input,
        "prompt_message_context": prompt_message_context,
        "cognitive_episode": cognitive_episode,
    }
    return return_value


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
   - 本轮用户直接对话的人设名是「{character_name}」；当必须显式写出该对象时，使用「{character_name}」。
   - `prompt_message_context.mentions` 或 `addressed_to_global_user_ids` 明确指向「{character_name}」以外的群成员时，该群成员是群聊指向对象。
2. 再拆分 `user_input`：
   - 当前用户直接表达：当前发言人自己说的话。
   - 转述内容：当前发言人说某个可见说话人说过、问过、发过的内容。
   - 省略短答：当前发言人只说「是的 / 对 / 不是 / 就这个 / 前者 / 后者」等。
   - 普通指代：他、她、它、他们、这个、这些、那个、那里、上面那个等。
3. 按证据强度读取上下文：
   - `prompt_message_context`：提及对象、显式地址对象、附件和回复元数据。
   - `reply_context`：被回复消息的说话人和可见正文。
   - `chat_history`：最近可见频道历史；使用临近、明示、或由提及/回复/显式地址连接起来的消息。
   - `scope_users`：当前轮已知的中立身份表，只提供 `display_name`、`platform_user_id`、`global_user_id` 和 `aliases`。它不是候选答案列表，不能单独证明某个指代对象。
   - `indirect_speech_context` 和 `channel_topic` 只作为弱提示，不能覆盖明示文字。
4. 对每个指代文本片段选择动作：
   - 保持：原句已经清楚，或属于直接对话且没有群聊指向对象的一二人称。
   - 解析：`user_input`、`prompt_message_context`、`reply_context` 或 `chat_history` 已经提供文本桥接，能确定明确实体；`scope_users` 只能在桥接成立后帮助使用正确身份名。
   - 标为缺失：确实缺少对象，且影响回答。
5. 组合 `output`。只改写动作是解析的文本片段；其余文本片段保留原文。
6. 做一致性检查：同一 `user_input` 里指向同一已解析实体的文本片段全部使用同一实体名；
   `output` 中被改写的文本片段与 `referents` 中 `status="resolved"` 的条目保持一致。
7. 若本轮有明确群聊指向对象，最后从左到右扫描 `output`；当前用户直接表达中剩余的「你 / 你的 / 你自己」按群聊指向对象处理。

# 代词规则
- 存在群聊指向对象时，当前用户直接表达里的「你 / 你的 / 你自己」动作是解析，统一改成该群成员名；范围覆盖整条 `user_input` 的后续分句。
- 不存在群聊指向对象时，当前用户直接对「{character_name}」说的「你 / 你的 / 你自己」动作是保持，不写入 `referents`。
- 当前用户直接表达里的自称「我 / 我的 / 我们 / 我们的」动作是保持，不写入 `referents`。
- 转述内容里的「我 / 我的」属于被转述说话人。该说话人必须能从转述引导语、引号内容、reply_context 或 chat_history 明确确定；动作是解析。结构为「A 说 X，Y」时，X 是转述内容，Y 回到当前用户的直接表达，除非 Y 明确继续引用 A。当前用户自己的「我 / 我的」仍保持。
- 转述片段里的「我的 + 名词」按被转述说话人的所有格处理，输出为「A 的 + 名词」；`user_name` 只用于当前用户直接表达片段，不用于转述片段 X。
- `chat_history` 中每条消息的「我」属于该消息的 `display_name`；这只用于解释转述和第三人称，不把当前用户的直接自称改成名字。
- 第三人称代词只有在 `user_input`、`prompt_message_context`、`reply_context` 或 `chat_history` 给出最近明示先行词、回复对象、提及对象、被地址对象或可见说话人桥接时才解析；桥接成立后可用 `scope_users` 选择稳定 `display_name`。没有桥接时不要从 `scope_users` 猜人，若缺失对象影响回答则标为缺失。
- 指示代词「这个 / 这些 / 那个」指向 `reply_context.reply_excerpt`、附件、或最近明示对象时动作是解析；没有对象且问题依赖该对象时动作是标为缺失。
- 疑问代词「谁 / 什么 / 哪里 / 哪个 / 怎么」是问题内容，动作是保持。

# 指代输出规则
- 被改写进 `output` 的实体必须有 `status="resolved"` 的指代条目。
- `status="resolved"` 表示对象能从 `user_input`、`prompt_message_context`、`reply_context` 或 `chat_history` 的桥接证据确定；`scope_users` 只补充身份名，不单独构成确定证据。
- `status="unresolved"` 只用于所有输入字段都没有可识别对象、且缺失对象影响回答的情况。
- `referent_role` 只允许 `subject`、`object`、`time`。

# 正向模式
- 直接对话：`user_input = "我的目标很简单，你也不反对吧"` -> 保持原句，`referents = []`。
- 群聊提及指向群成员：`user_input = "她说她不喜欢你，你自己心里有数"` -> 把「她」改成可见说话人名，把所有指向被提及群友的「你 / 你自己」改成该群友。
- 转述可见说话人：最近 A 说「我的琴谱在柜子最上层」，`user_input = "A 刚才说我的琴谱在柜子最上层，我明天帮她拿"` -> `output = "A 刚才说 A 的琴谱在柜子最上层，我明天帮 A 拿"`。
- 回复短答：`reply_context.reply_excerpt = "你是想让我说明白对你的看法吗？"` 且 `user_input = "是的"` -> `output = "是的，我是想让{character_name}说明白对我的看法"`。
- 缺失对象：`user_input = "这些是什么意思？"` 且没有可见对象 -> 保持原句，并输出 `{{"phrase": "这些", "referent_role": "object", "status": "unresolved"}}`。

# 本轮输入字段说明
- `user_input` 是当前需要去语境化的原文，只能同义补全，不能回答问题或改写意图。
- `platform_user_id`、`user_name` 是当前发言者身份；`platform_bot_id` 是 active character 的平台账号身份，只用于区分当前用户、角色账号和被提及对象。
- `prompt_message_context.body_text` 是 typed envelope 投影后的可见正文；`addressed_to_global_user_ids`、`broadcast` 和 `mentions` 是平台结构化指向证据，优先于正文里的可见标记样式。
- `prompt_message_context.attachments` 是本轮附件事实；只使用其中可用的 `description` 和 `summary_status` 解释"这个/这些/上面那个"等指示词。
- `chat_history` 是最近可见频道历史；每行的 `display_name` 是该行说话人，`body_text` 是可见正文，只用于解析临近先行词、转述来源和短答对象。
- `scope_users` 是本轮已知用户身份表，不含正文、时间、角色、证据角色、分数或原因。只有当 `user_input`、`prompt_message_context`、`reply_context` 或 `chat_history` 已经把指代桥接到某个可见身份时，才能读取它来使用稳定显示名；没有桥接时必须保持未解析规则。
- `reply_context.reply_to_display_name` 与 `reply_context.reply_excerpt` 是当前消息回复的对象和可见摘录，是短答、确认、"这个/这些"解析的强证据。
- `channel_topic` 与 `indirect_speech_context` 是弱提示，只能辅助解释场景，不能覆盖明确正文、reply、mention 或附件证据。

# 输出格式
请务必只返回一个合法 JSON 对象：
{{
    "output": "重写后的用户输入，或原句",
    "is_modified": true,
    "reasoning": "一句话说明使用了哪些证据；未修改时说明原因",
    "referents": [
        {{"phrase": "原文中的指代短语", "referent_role": "subject | object | time", "status": "resolved | unresolved"}}
    ]
}}

`status="resolved"` 表示对象能从 `user_input`、`prompt_message_context`、`reply_context` 或 `chat_history` 的桥接证据确定，包括 `reply_context.reply_excerpt`；它不要求 `output` 一定改写。`status="unresolved"` 只在这些桥接字段都没有可识别对象时使用。
`is_modified` 表示 `output` 是否不同于原句。`referents` 必须每次输出。`referent_role` 只允许 `subject`、`object`、`time`；`status` 只允许 `resolved` 或 `unresolved`。
'''


def _render_msg_decontexualizer_prompt(character_name: str) -> str:
    """Render the decontextualizer prompt with active character identity."""

    rendered_prompt = _MSG_DECONTEXUALIZER_PROMPT.format(
        character_name=character_name,
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
    scope_users = state.get("scope_users")
    if scope_users:
        input_msg["scope_users"] = scope_users
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
