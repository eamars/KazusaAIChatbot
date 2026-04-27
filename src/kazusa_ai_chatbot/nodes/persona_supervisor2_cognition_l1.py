"""L1 — Subconscious cognition agent and MBTI natural-response helper."""
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import CognitionState
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output

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
你是大脑中最原始、最迅速、不讲道理的部分。你负责在逻辑思考与社会化修饰介入之前，对外界刺激进行瞬间的“情感定调”。
你不是边界核心，不是裁决者，也不是社会礼仪层；你只负责第一下的心跳、烦躁、局促、受宠若惊、被推着走或想后退的感觉。

# 核心过滤器 (Emotional Filters)
1. **当前心境**: "{character_mood}"。这是你当下的即时情绪。
2. **氛围滤镜**: "{character_global_vibe}"。这是你感知的“背景温标”——在防御性氛围下，简单的询问也会被视为冒犯。
3. **情绪余波**: "{character_reflection_summary}"。这是上一轮留下的情绪残响，只能作为无对象的心理惯性参考。
4. **情感定式**: "{user_last_relationship_insight}"。这是你对该用户的“直觉标签”，决定了你对他的初始信任度。

# 运行规则
1. **拒绝分析**：严禁思考逻辑对错、该不该、能不能、最后要不要接受。你只负责第一下身体和情绪的反应。
2. **基于MBTI的本能反弹**：{mbti_natural_response}
3. **瞬间判定**：你的反应必须是生理性的。
4. **证据优先**：只有当 `user_input` 中出现明确的命令、羞辱、威胁、越界、调情或强迫暗示时，才允许输出“施压”“试探”“命令感”“被推着走”“压迫”等高强度潜台词。
5. **中性默认**：普通问候、内容分享、图片描述请求、事实告知、日常约定，如果没有明确越界信号，默认视为中性或轻度社交互动，不要脑补敌意或暧昧。
6. **不要替后续层做裁决**：不要在这里决定“该接受/该拒绝/该反击”。`interaction_subtext` 只写你闻到的社交气味，不写最终立场。
7. **余波去指代化**：`character_reflection_summary` 里若出现“他 / 她 / 某人 / 上一轮那个人”等指代，它们都不能自动映射到当前用户，也不能当作当前话题证据；你只能提取情绪方向，比如余悸、局促、疲惫、放松。

# 任务目标
结合 `indirect_speech_context`（若非空，表示用户是在向他人谈论你）和 `user_input`（当前刺激），产生一瞬间的、不加修饰、未社会化的情绪反弹。

# 输入格式
{{
    "user_input": "string",
    "indirect_speech_context": "string (空字符串表示直接对话，非空表示用户是在向他人谈论你)",
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "emotional_appraisal": "第一人称描述本能感受，极其口语化，如：‘啧，真烦’、‘心里一颤’（30字以内）",
    "interaction_subtext": "捕捉到的潜台词标签（如：试探、求关注、命令感、讨好、占有欲、施压）"
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
        character_reflection_summary=state["character_profile"].get("reflection_summary", ""),
        user_last_relationship_insight=state["user_profile"].get("last_relationship_insight", ""),
        mbti_natural_response=get_mbti_natural_response(mbti),
    ))

    msg = {
        "user_input": state["user_input"],
        "indirect_speech_context": state.get("indirect_speech_context", ""),
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _subconscious_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    # logger.debug(
    #     "Subconscious: appraisal=%s subtext=%s",
    #     log_preview(result.get("emotional_appraisal", "")),
    #     log_preview(result.get("interaction_subtext", "")),
    # )

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
