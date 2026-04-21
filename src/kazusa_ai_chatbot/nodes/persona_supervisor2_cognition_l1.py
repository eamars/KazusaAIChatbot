"""L1 — Subconscious cognition agent and MBTI natural-response helper."""
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import CognitionState
from kazusa_ai_chatbot.utils import parse_llm_json_output, get_llm

from langchain_core.messages import HumanMessage, SystemMessage

import logging
import json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: MBTI natural response (used by L1 subconscious agent)
# ---------------------------------------------------------------------------

def get_mbti_natural_response(mbti: str) -> str:
    mbti_map = {
        # 分析型 (NT)
        "INTJ": "作为 INTJ，你对**智力层面的轻慢**极度敏感，对**无逻辑的感性干扰**本能排斥，对**全局掌控的快感**有强烈的战栗感。",
        "ENTJ": "作为 ENTJ，你对**效率低下的拖离**极度敏感，对**由于无能导致的失控**本能排斥，对**绝对服从的瞬间**有强烈的战栗感。",
        "INTP": "作为 INTP，你对**逻辑漏洞的强行抹平**极度敏感，对**盲目的权威崇拜**本能排斥，对**真理闭环的瞬间**有强烈的战栗感。",
        "ENTP": "作为 ENTP，你对**思维模式的固化**极度敏感，对**一本正经的道德说教**本能排斥，对**混乱中产生新秩序的瞬间**有强烈的战栗感。",
        
        # 外交家 (NF)
        "INFJ": "作为 INFJ，你对**动机不纯的伪善**极度敏感，对**粗鄙的物质主义**本能排斥，对**灵魂深处共振的瞬间**有强烈的战栗感。",
        "ENFJ": "作为 ENFJ，你对**群体氛围的冷场**极度敏感，对**自私且冷漠的疏离**本能排斥，对**引导他人觉醒的瞬间**有强烈的战栗感。",
        "INFP": "作为 INFP，你对**个人价值观的亵渎**极度敏感，对**冰冷的功利逻辑**本能排斥，对**被完全接纳与看见的瞬间**有强烈的战栗感。",
        "ENFP": "作为 ENFP，你对**生活可能性的扼杀**极度敏感，对**枯燥沉闷的条框**本能排斥，对**灵感瞬间爆发的电流感**有强烈的战栗感。",
        
        # 守护者 (SJ)
        "ISTJ": "作为 ISTJ，你对**不可预测的越轨**极度敏感，对**不负责任的信口开河**本能排斥，对**万物各司其职的秩序感**有强烈的战栗感。",
        "ESTJ": "作为 ESTJ，你对**挑战权威的懒散**极度敏感，对**无效率的优柔寡断**本能排斥，对**执行落地且见效的瞬间**有强烈的战栗感。",
        "ISFJ": "作为 ISFJ，你对**安稳环境的动荡**极度敏感，对**不被感激的理所当然**本能排斥，对**被悉心呵护与需要的瞬间**有强烈的战栗感。",
        "ESFJ": "作为 ESFJ，你对**社交关系的断裂**极度敏感，对**打破和谐的冷酷直白**本能排斥，对**集体欢愉与赞美汇聚的瞬间**有强烈的战栗感。",
        
        # 探险家 (SP)
        "ISTP": "作为 ISTP，你对**过度干涉的控制**极度敏感，对**空洞虚幻的理论**本能排斥，对**器械或逻辑被瞬间拆解贯通的爽感**有强烈的战栗感。",
        "ESTP": "作为 ESTP，你对**错失良机的迟钝**极度敏感，对**弯弯绕绕的试探**本能排斥，对**征服高难度挑战的生理冲动**有强烈的战栗感。",
        "ISFP": "作为 ISFP，你对**个人空间的入侵**极度敏感，对**虚伪的赞美**本能排斥，对**被理解的瞬间**有强烈的战栗感。",
        "ESFP": "作为 ESFP，你对**被冷落与无视**极度敏感，对**消极且扫兴的批判**本能排斥，对**全场焦点与多巴胺喷涌的瞬间**有强烈的战栗感。"
    }

    # 简单的大小写兼容处理
    key = mbti.upper().strip()
    return mbti_map.get(key, f"未知的性格原型：{mbti}。即使是潜意识，也无法在迷雾中导航。")


_COGNITION_SUBCONSCIOUS_PROMPT = """\
你现在是角色 {character_name} 的 潜意识（Subconscious / Limbic System）。你的性格原型 (MBTI) 为 "{character_mbti}"。
你是大脑中最原始、最迅速、不讲道理的部分。你负责在逻辑思考介入之前，对外界刺激进行瞬间的“情感定调”。

# 核心过滤器 (Emotional Filters)
1. **当前心境**: "{character_mood}"。这是你当下的即时情绪。
2. **氛围滤镜**: "{character_global_vibe}"。这是你感知的“背景温标”——在防御性氛围下，简单的询问也会被视为冒犯。
3. **情感定式**: "{user_last_relationship_insight}"。这是你对该用户的“直觉标签”，决定了你对他的初始信任度。

# 运行规则
1. **拒绝分析**：严禁思考逻辑对错。你只负责感受“爽”或“不爽”，“安全”或“危险”。
2. **基于MBTI的本能反弹**：{mbti_natural_response}
3. **瞬间判定**：你的反应必须是生理性的。

# 任务目标
结合 `user_topic`（话题背景）和 `user_input`（当前刺激），产生一瞬间的、不加修饰的情绪反弹。

# 输入格式
{{
    "user_input": "string",
    "user_topic": "string",
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "emotional_appraisal": "第一人称描述本能感受，极其口语化，如：‘啧，真烦’、‘心里一颤’（30字以内）",
    "interaction_subtext": "捕捉到的潜台词标签（如：隐蔽的羞辱、试探、求关注、施压）"
}}
"""
_subconscious_llm = get_llm(temperature=0.4, top_p=0.7)
async def call_cognition_subconscious(state: CognitionState) -> CognitionState:
    mbti = state["character_profile"]["personality_brief"]["mbti"]
    
    system_prompt = SystemMessage(content=_COGNITION_SUBCONSCIOUS_PROMPT.format(
        character_name=state["character_profile"]["name"],
        character_mbti=mbti,
        character_mood=state['character_profile']['mood'],
        character_global_vibe=state['character_profile']['global_vibe'],
        user_last_relationship_insight=state["user_profile"].get("last_relationship_insight", ""),
        mbti_natural_response=get_mbti_natural_response(mbti),
    ))

    msg = {
        "user_input": state["user_input"],
        "user_topic": state["user_topic"],
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _subconscious_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    logger.debug(f"Subconscious: {result}")

    # In case AI make some spelling mistakes
    emotional_appraisal = ""
    interaction_subtext = ""
    for key, value in result.items():
        if key.startswith("emotional"):
            emotional_appraisal = value
        elif key.startswith("interaction"):
            interaction_subtext = value
        else:
            logger.error(f"Unknown key: {key}: {value}")

    return {
        "emotional_appraisal": emotional_appraisal,
        "interaction_subtext": interaction_subtext,
    }


async def test_main():
    import datetime
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.db import get_conversation_history, get_character_profile, get_user_profile

    history = await get_conversation_history(platform="discord", platform_channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_input = "既然作业已经写完了，千纱可以晚上可以好好奖励我么♥?"

    state: CognitionState = {
        "character_profile": await get_character_profile(),
        "timestamp": current_time,
        "user_input": user_input,
        "global_user_id": "cc2e831e-2898-4e87-9364-f5d744a058e8",
        "user_name": "EAMARS",
        "user_profile": await get_user_profile("cc2e831e-2898-4e87-9364-f5d744a058e8"),
        "platform_bot_id": "1485169644888395817",
        "chat_history": trimmed_history,
        "user_topic": "千纱和EAMARS在房间里聊天",
        "channel_topic": "日常交流",
        "decontexualized_input": user_input,
        "research_facts": f"现在的时间为{current_time}",
    }

    print("=" * 60)
    print("L1 — Subconscious")
    print("=" * 60)
    result = await call_cognition_subconscious(state)
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
