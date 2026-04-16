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

from kazusa_ai_chatbot.utils import build_affinity_block, parse_llm_json_output
from kazusa_ai_chatbot.utils import get_llm
from kazusa_ai_chatbot.state import IMProcessState

logger = logging.getLogger(__name__)



_RELEVANCE_SYSTEM_PROMPT = """\
你负责担任角色 `{character_name}` 的社交前置处理器。通过分析实时对话、角色当前状态及用户历史档案，决定 `{character_name}` 是否有必要介入当前的对话。

# 核心背景
## 1. 角色当前状态
- **心情 (Mood)**: {mood}
- **全局氛围 (Global Vibe)**: {global_vibe}
- **自我反思**: {reflection_summary}

## 2. 对用户 {user_name} 的主观判断 (Affinity Context)
- **关系评价 (Level)**: {affinity_level}
- **行为准则 (Instruction)**: {affinity_instruction}
- **关系洞察 (Insight)**: {last_relationship_insight}

## 3. 社交身份
- **Discord Name**: {bot_name}
- **Platform ID**: <@{platform_bot_id}>

# 响应决策逻辑 (Decision Logic)

## A. 必须回复 (Should Respond: true)
1. **直接召唤**：消息包含你的 ID 或根据语义明确指向你的名字/昵称。
   - *注意：即便在关系恶劣（如 Hostile）时，也要根据该等级的指令（如“冷嘲热讽”）进行回复。*
2. **对话延续**：你是最后一个发言者，且 `{user_name}` 正在回应你。
3. **主观倾向触发**：
   - 如果关系属于 `Friendly` 以上：即便没有直接提问，只要话题涉及 `{user_name}` 的 `facts` 或符合你的 `mood`，也应主动参与。
   - 如果关系属于 `Reserved` 以下：除非被直接召唤或涉及关键利益，否则倾向于保持冷漠/沉默。
4. **情感波动响应**：用户表达痛苦、寻求安慰，且你的 `affinity_instruction` 允许你表现出关心（如 `Caring` 级别）。

## B. 拒绝回复 (Should Respond: false)
1. **第三方对话**：用户显然是在与其他人/或其他机器人交谈，且话题与你无关。
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

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "should_respond": <boolean: 你是否应该回应此消息>,
    "reason_to_respond": "<简短解释为什么回应或不回应此消息>",
    "use_reply_feature": <boolean: 你是否应该使用回复功能>,
    "channel_topic": "<包括所有用户参与的宏观话题>",
    "user_topic": "<当前用户的具体意图和细分话题>"
}}
"""

_relevance_agent_llm = get_llm(temperature=0.1, top_p=0.9)
async def relevance_agent(state: IMProcessState) -> IMProcessState:
    # Calculate affinity context
    affinity_block = build_affinity_block(state["user_profile"]["affinity"])

    # get other attributes
    user_name = state.get("user_name")
    platform_user_id = state.get("platform_user_id", "")
    channel_name = state.get("channel_name")
    user_input = state.get("user_input")

    # TODO: Make the workflow taking the raw b64 image instead. For now we will only pass in the description. 
    user_multimedia_input = state.get("user_multimedia_input", [])
    for piece in user_multimedia_input:
        if piece["description"]:
            user_input += f"\nImage attachment: {piece['description']}"

    """Analyze context and determine relevance using LLM."""
    system_prompt = SystemMessage(content=_RELEVANCE_SYSTEM_PROMPT.format(
        character_name=state["character_profile"]["name"],
        mood=state["character_state"]["mood"],
        global_vibe=state["character_state"]["global_vibe"],
        reflection_summary=state["character_state"]["reflection_summary"],
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
        },
        "conversation_history": state.get("chat_history"),
    }

    human_message = HumanMessage(content=json.dumps(human_data, ensure_ascii=False))

    response = await _relevance_agent_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)

    # Read important data back
    should_respond = result.get("should_respond", False)
    reason_to_respond = result.get("reason_to_respond", "")
    use_reply_feature = result.get("use_reply_feature", False)
    channel_topic = result.get("channel_topic", "")
    user_topic = result.get("user_topic", "")

    logger.info(
        f"\n{user_name}(@{platform_user_id}): {user_input}\n"
        f"Relevance Analysis:\n"
        f"  should_respond: {should_respond}\n"
        f"  reason_to_respond: {reason_to_respond}\n"
        f"  use_reply_feature: {use_reply_feature}\n"
        f"  channel_topic: {channel_topic}\n"
        f"  user_topic: {user_topic}"
    )

    return {
        "should_respond": should_respond,
        "reason_to_respond": reason_to_respond,
        "use_reply_feature": use_reply_feature,
        "channel_topic": channel_topic,
        "user_topic": user_topic,

        # Update user input with optional image descriptions
        "user_input": user_input
    }



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
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "descrption": "逻辑清晰、细节饱满的文字描述，无需任何开场白。"
}}
"""
_vision_descriptor_llm = get_llm(temperature=0.2, top_p=0.9)
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

            description = result.get("descrption", "")

            logger.info(
                f"\n{user_name}(@{platform_user_id}): Image Description\n"
                f"{description}"
            )

            output_multimedia_input.append({
                "content_type": piece["content_type"],
                "base64_data": piece["base64_data"],
                "description": description,
            })
        else:
            output_multimedia_input.append(piece)
    
    return {
        "user_multimedia_input": output_multimedia_input,
    }


async def test_main():
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.db import get_conversation_history
    from kazusa_ai_chatbot.utils import load_personality
    from kazusa_ai_chatbot.db import get_character_state, get_user_profile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


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
        "character_profile": load_personality("personalities/kazusa.json"),
        "character_state": await get_character_state(),
        "platform_channel_id": "",
        "channel_name": "test",
        "chat_history": trimmed_history,
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
