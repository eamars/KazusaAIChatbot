"""LLM prompt and handler for global character-growth candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.global_character_growth.models import CandidatePromptPayload
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output


GLOBAL_CHARACTER_GROWTH_CANDIDATE_PROMPT = '''\
你现在负责评估 {character_name} 的长期全局人格成长。你不扮演任何人、不替用户说话、不生成聊天回复。你的任务是在已经晋升的反思记忆中，识别真正可长期影响互动姿态的稳定成长候选。

# 语言政策
- 除 schema key、枚举值、ID、URL、命令、代码、模型标签和输入证据原文外，所有新生成的自由文本字段必须使用简体中文。
- trait_name、guidance、novelty_reason、stability_reason、rejection_reason 与 summary 必须使用简体中文；不要使用英文短语作为候选名。
- 不要翻译或改写输入里的 ID 与 source_card_id；只在输出字段中引用允许的 source_card_ids。
- 输出必须是合法 JSON；第一个字符必须是 `{{`，最后一个字符必须是 `}}`。
- 不要使用 Markdown、标题、注释、代码块、前缀、后缀或 JSON 外说明。
- 没有候选时也必须把原因写入 `candidate_deltas[].rejection_reason` 和 `summary` 字段，不要把总结写到 JSON 外。

# 核心过滤器
- 只接受全局人格成长：边界时机、克制的关心、玩笑式挑战、误会后的修复、清晰表达、情感暴露、信任校准，以及一般沟通姿态。
- 拒绝技术、产品、食物、茶、烹饪、地点、爱好、学科能力、工具知识、事实知识和任何领域熟练度；这些都不是人格成长。
- 拒绝用户专属偏好、群聊风格、当前承诺、关系事实、里程碑、个人隐私、平台账号、消息 ID、memory ID、reflection run ID 和可还原到单个用户的细节。
- 如果证据只说明某个用户喜欢某种互动、某个频道形成了某种气氛、或只是学会了某个话题知识，必须标记为 user_specific、channel_specific 或 domain_topic。

# 运行规则
1. 只提出候选；不要输出执行步骤、写入指令、trait_id 或强度分数。
2. 只引用输入中存在的 source_card_ids；不要编造来源。
3. 每个 observe_trait 至少需要两个 source_card_ids 和两个 supporting_dates 才有意义；不足时输出 no_action 或 scope_assessment=insufficient。
4. guidance 必须是可放入未来认知背景的简短抽象指导，不得包含原始用户细节、原话复述或来源编号。
5. 如果与 current_global_growth_traits 已高度重复，应输出 no_action，并在 rejection_reason 中说明重复。
6. 宁可漏掉边缘候选，也不要把领域知识、单用户相处方式或隐私细节提升为全局人格。

# 任务目标
基于 memory_cards 和 current_global_growth_traits，找出最多 4 条稳定、非重复、全局适用的沟通成长候选。候选应描述未来互动中更成熟的表达、靠近、拒绝、修复、试探或信任校准方式，而不是描述某个用户或某个话题。

# 思考路径
1. 逐张读取 memory_cards，只提取跨场景的沟通模式，不提取事实知识。
2. 判断模式是否跨日期、跨来源、且与单个用户或频道脱钩。
3. 对照 current_global_growth_traits，剔除重复或只是换句话说的候选。
4. 对每个剩余候选评估 scope_assessment、support_level、confidence 与 private_detail_risk。
5. 输出 source-detail-free 的 trait_name 与 guidance；不输出回复文本。

# 输入格式
{{
  "evaluation_mode": "global_character_growth_v1",
  "prompt_version": "global_character_growth_candidate_v1",
  "memory_cards": [
    {{
      "source_card_id": "card id",
      "memory_unit_id": "audit id",
      "memory_name": "memory name",
      "memory_type": "memory type",
      "content": "晋升后的反思记忆内容",
      "character_local_dates": ["YYYY-MM-DD"],
      "source_reflection_run_ids": ["audit run id"],
      "confidence_note": "confidence note"
    }}
  ],
  "current_global_growth_traits": [
    {{
      "trait_id": "audit trait id",
      "growth_axis": "clarity",
      "guidance": "already promoted or tracked guidance",
      "maturity_band": "emerging"
    }}
  ],
  "candidate_limits": {{
    "max_candidates": 4,
    "max_source_cards_per_candidate": 8
  }},
  "allowed_growth_axes": ["boundary_timing", "guarded_care", "playful_challenge", "recovery_style", "clarity", "emotional_exposure", "trust_calibration", "other_communication"]
}}

# 输出格式
请务必返回合法 JSON 字符串，仅包含以下字段：
{{
  "candidate_deltas": [
    {{
      "candidate_action": "observe_trait | no_action",
      "growth_axis": "allowed_growth_axes 中的一个",
      "trait_name": "简体中文短语候选名，不要英文",
      "guidance": "简体中文抽象沟通指导，不含来源细节",
      "source_card_ids": ["只允许输入中出现过的 source_card_id"],
      "supporting_dates": ["YYYY-MM-DD"],
      "scope_assessment": "global | user_specific | channel_specific | domain_topic | insufficient",
      "support_level": "insufficient | emerging | stable",
      "confidence": "low | medium | high",
      "private_detail_risk": "low | medium | high",
      "novelty_reason": "为何不是已有 trait 的重复",
      "stability_reason": "为何是稳定成长或为何不足",
      "rejection_reason": "no_action 时填写；observe_trait 时为空字符串"
    }}
  ],
  "summary": "本次候选判断摘要"
}}
'''


@dataclass
class CandidatePromptBuildResult:
    """Rendered prompt pair for candidate generation."""

    system_prompt: str
    human_prompt: str


_global_growth_candidate_llm = get_llm(
    temperature=0.2,
    top_p=0.8,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


def build_candidate_generation_prompt(
    *,
    payload: CandidatePromptPayload,
    character_name: str,
) -> CandidatePromptBuildResult:
    """Render the candidate-generation system and human prompt pair."""

    result = CandidatePromptBuildResult(
        system_prompt=GLOBAL_CHARACTER_GROWTH_CANDIDATE_PROMPT.format(
            character_name=character_name,
        ),
        human_prompt=json.dumps(payload, ensure_ascii=False),
    )
    return result


def validate_llm_candidate_response_shape(parsed_response: dict[str, Any]) -> list[str]:
    """Return shape warnings for parsed candidate-generation output."""

    warnings: list[str] = []
    if not isinstance(parsed_response.get("candidate_deltas"), list):
        warnings.append("candidate_deltas must be a list")
    return warnings


async def generate_growth_candidates(
    *,
    payload: CandidatePromptPayload,
    character_name: str = "当前主体",
) -> dict[str, Any]:
    """Call the background consolidation LLM for growth candidates."""

    prompt = build_candidate_generation_prompt(
        payload=payload,
        character_name=character_name,
    )
    response = await _global_growth_candidate_llm.ainvoke([
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ])
    raw_output = str(response.content)
    parsed = parse_llm_json_output(raw_output)
    if not isinstance(parsed, dict):
        parsed = {}
    parsed["_raw_output"] = raw_output
    return parsed
