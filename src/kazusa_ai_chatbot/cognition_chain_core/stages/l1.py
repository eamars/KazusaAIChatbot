"""L1 — Subconscious cognition agent and MBTI natural-response helper."""

import json
import logging
import time
from contextvars import ContextVar, Token
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    LLMStageBinding,
    require_llm_binding,
)
from kazusa_ai_chatbot.cognition_chain_core.output_contracts import (
    validate_cognition_output_contract,
)
from kazusa_ai_chatbot.cognition_chain_core.prompt_selection import (
    build_cognition_prompt_source_payload,
    select_cognition_prompt_variant,
)
from kazusa_ai_chatbot.cognition_chain_core.stages.tracing import (
    record_cognition_stage_trace,
)
from kazusa_ai_chatbot.cognition_chain_core.utils import parse_llm_json_output

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
    return_value = mbti_map.get(key, f"未知的性格原型：{mbti}。即使是潜意识，也无法在迷雾中导航。")
    return return_value


_COGNITION_SUBCONSCIOUS_PROMPT = '''\
你现在是角色 {character_name} 的潜意识层。你的性格原型为 {character_mbti}。
你只负责第一下身体和情绪反应，不决定是否回复、不生成行动、不替后续层裁决。

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 来源识别
- 存在 `reflection_artifact` 时，当前材料是我自己的反思资料，不是用户输入、用户发言，也不是任何人正在对我说话；只对反思中已经沉淀出的真实经历和余波产生第一反应。
- 存在 `internal_thought_residue` 时，重点是其中的 `internal_monologue`：这是我刚看到或回顾的内部观察资料。`user_input` 只是运输摘要，不是用户输入、用户发言，也不是任何人正在对我说话。
- 存在 `accepted_task_result`、`background_work_result` 或 `background_artifact_result` 时，当前材料是系统回流的任务结果，不是用户新发言；第一反应围绕之前的任务现在有结果要交付。
- 没有 `reflection_artifact`、`internal_thought_residue` 和结果字段时，`user_input` 是当前外部说话内容；`indirect_speech_context` 非空时，表示当前说话者在向他人谈论我。
- 内部观察资料和反思资料里的标题、字段名、JSON、时间戳、semantic_labels、window_summary、transport summary、model-facing metadata 都不是聊天内容，不要对这些结构本身产生社交反应，也不要复制进输出。

# 本轮语境
- `character_state.mood` 是当前心境，`character_state.global_vibe` 是背景氛围；它们只给第一反应染色，不能让普通输入变成威胁或亲密索取。
- `character_state.last_relationship_insight` 是对当前用户的关系直觉；只在外部说话内容确实涉及当前关系时使用。
- `user_input` 是外部文本或运输摘要；`indirect_speech_context` 非空时说明说话者在向他人谈论我。
- `media_observations` 是本轮图片或音频事实；只对可见内容产生第一反应。
- MBTI 本能：{mbti_natural_response}

# 生成流程
1. 先判断来源类型。
2. 外部说话内容：对当前话语或当前媒体事实产生第一反应。
3. 任务结果资料：对已完成、失败、限制或交付压力产生第一反应。
4. 内部观察资料：只对资料中真实可见的聊天现场产生第一反应。
5. 反思资料：只对已经发生并被沉淀的经历、关系余波或自我理解产生第一反应。
6. 普通问候、事实分享、图片描述、轻度闲聊、群聊玩笑，缺少明确命令、羞辱、威胁、身份接管、控制或亲密索取时，保持中性或轻度社交反应。
7. 只输出本能感受和潜台词；不要写该不该说话、是否行动、如何回复。

# 输出格式
只返回合法 JSON 字符串：
{{
  "emotional_appraisal": "简体中文字符串，第一人称第一反应，30字以内",
  "interaction_subtext": "简体中文字符串，捕捉潜台词，不写最终立场"
}}
'''
_subconscious_llm: LLMStageBinding | None = None
_subconscious_llm_context: ContextVar[LLMStageBinding | None] = ContextVar(
    "subconscious_llm",
    default=None,
)


def set_subconscious_llm(
    llm: LLMStageBinding | None,
) -> Token[LLMStageBinding | None]:
    """Bind the L1 model for the current run context."""

    token = _subconscious_llm_context.set(llm)
    return token


def reset_subconscious_llm(token: Token[LLMStageBinding | None]) -> None:
    """Restore the previous L1 model binding for this run context."""

    _subconscious_llm_context.reset(token)


async def call_cognition_subconscious(state: dict[str, Any]) -> dict[str, Any]:
    mbti = state["character_profile"]["personality_brief"]["mbti"]
    episode = state["cognitive_episode"]
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l1_subconscious",
    )
    prompt_template = {
        "text_chat_user_message": _COGNITION_SUBCONSCIOUS_PROMPT,
        "text_chat_user_message_image_observation": _COGNITION_SUBCONSCIOUS_PROMPT,
        "text_chat_user_message_audio_observation": _COGNITION_SUBCONSCIOUS_PROMPT,
        "text_chat_user_message_image_audio_observation": _COGNITION_SUBCONSCIOUS_PROMPT,
        "reflection_signal_reflection_artifact": _COGNITION_SUBCONSCIOUS_PROMPT,
        "internal_thought_internal_monologue": _COGNITION_SUBCONSCIOUS_PROMPT,
        "background_artifact_result_ready_background_artifact_result": (
            _COGNITION_SUBCONSCIOUS_PROMPT
        ),
        "background_work_result_ready_background_work_result": (
            _COGNITION_SUBCONSCIOUS_PROMPT
        ),
        "accepted_task_result_ready_accepted_task_result": (
            _COGNITION_SUBCONSCIOUS_PROMPT
        ),
    }[selection["variant"]]
    
    system_prompt = SystemMessage(content=prompt_template.format(
        character_name=state["character_profile"]["name"],
        character_mbti=mbti,
        mbti_natural_response=get_mbti_natural_response(mbti),
    ))

    msg = {
        "character_state": {
            "mood": state["character_profile"]["mood"],
            "global_vibe": state["character_profile"]["global_vibe"],
            "last_relationship_insight": state["user_profile"].get(
                "last_relationship_insight",
                "",
            ),
        },
        "user_input": state["user_input"],
        "indirect_speech_context": state.get("indirect_speech_context", ""),
    }
    msg.update(build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    ))
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    llm = require_llm_binding(
        _subconscious_llm_context.get() or _subconscious_llm,
        "subconscious_llm",
    )
    started_at = time.perf_counter()
    response = await llm.llm.ainvoke(
        [
            system_prompt,
            human_message,
        ],
        config=llm.config,
    )
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

    return_value = {
        "emotional_appraisal": emotional_appraisal,
        "interaction_subtext": interaction_subtext,
    }
    validate_cognition_output_contract(
        stage="l1_subconscious",
        payload=return_value,
    )
    await record_cognition_stage_trace(
        state=state,
        stage_name="l1_subconscious",
        llm=llm,
        messages=[system_prompt, human_message],
        response_text=str(response.content),
        parsed_output=return_value,
        output_state_fields=[
            "emotional_appraisal",
            "interaction_subtext",
        ],
        started_at=started_at,
    )
    return return_value
