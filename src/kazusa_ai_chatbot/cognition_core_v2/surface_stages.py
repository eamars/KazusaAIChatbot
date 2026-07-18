"""Four bounded V2 text-surface stage handlers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    TextSurfaceServicesV2,
    VisualSurfaceServicesV2,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


STYLE_SYSTEM_PROMPT = '''根据 expression policy、semantic affect、semantic relationship、
interaction style 和 character_voice_context 选择适合当前语境的说话方式。本阶段只决定词语层次、
句式、节奏、停顿与标点，在不改变 selected intention 和 bid 含义、也不增加内容节点的前提下，
把角色声音、情绪和互动姿态融入用词、句式与节奏，形成适合聊天发送的表达指导。

角色自己的反思和内部观察属于语境，不是当前用户的即时发言；运行元数据也不属于措辞内容。
新生成的自由文本使用简体中文；用户引文、专有名词、代码、URL 以及 schema 或 enum token 保持
原样。本阶段返回风格指导，不写最终对话。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 style_guidance，其值是一个非空字符串，最多 1000 字符。'''


async def run_style_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> str:
    """Run the stage-local style prompt and validate its exact field."""

    prompt_text = _surface_prompt_text(payload)
    style_llm = services.llm
    response = await style_llm.ainvoke(
        [
            SystemMessage(content=STYLE_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ],
        config=services.style_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, Mapping) or set(parsed) != {"style_guidance"}:
        raise ValueError("style stage fields are not exact")
    return _bounded_text(parsed["style_guidance"], "style guidance", 1000)


CONTENT_PLAN_SYSTEM_PROMPT = '''规划当前角色在这个场景中实际会说出或发送的内容，使其自然表达
已经形成的角色判断。综合 selected intention、primary bid、supporting bid、visible episode、
semantic affect、semantic relationship、expression policy、interaction style 和
permitted_action_results。

# 规划步骤
1. 按认知阶段选择的方式回应或参与当前输入。结合角色关系、情绪和场景压力推进互动，而不是机械
复述先前对话。
2. 选择鲜明、贴合角色的内容。只要不与当前输入或明确生效的约束冲突，也不颠倒 actor、target、
beneficiary 或 subject，连贯的想象细节、玩笑和有创造力的展开都可以使用。
3. 结构化 visible percept 的角色字段具有权威性。在用户对话中，current_user 的第一人称指当前
用户；self 表示当前角色，也是被直接称呼者和祈使句的隐含主语。自由文本使用自然的中文参与者
称呼。
4. permitted_action_results 是角色大脑能力的精确账本。只有 executed 支持其有界的已完成效果；
其他 status 不支持完成声明。请求或目标候选只支持角色在言语中的态度，不代表能力已经执行。
5. 只规划角色要表达的含义和互动推进，让情绪、关系强度与动作倾向通过台词含义、语气和节奏
呈现。

返回一份简洁计划，并给出一到八条语义要求，用来保护选定含义、当前真实边界、角色方向和能力
执行事实。角色自己的反思和内部观察属于语境，不是当前用户的即时发言；运行元数据不属于对话
内容。新生成的自由文本使用简体中文；用户引文、专有名词、代码、URL 以及 schema 或 enum token
保持原样。本阶段不写最终对话。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 content_plan 和 content_requirements。
content_plan 是一个非空字符串，最多 1000 字符；content_requirements 是一到八条互不重复的
非空语义要求，每条最多 500 字符。'''


async def run_content_plan_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> tuple[str, list[str]]:
    """Run the content prompt and return its plan and semantic requirements."""

    prompt_text = _surface_prompt_text(payload)
    content_plan_llm = services.llm
    response = await content_plan_llm.ainvoke(
        [
            SystemMessage(content=CONTENT_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ],
        config=services.content_plan_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, Mapping) or set(parsed) != {
        "content_plan",
        "content_requirements",
    }:
        raise ValueError("content-plan stage fields are not exact")
    content_plan = _bounded_text(parsed["content_plan"], "content plan", 1000)
    content_requirements = _bounded_text_list(
        parsed["content_requirements"],
        "content requirements",
    )
    return content_plan, content_requirements


PREFERENCE_SYSTEM_PROMPT = '''识别当前角色判断和当前场景中真实存在的可见表达边界或称呼对象约束。
以 selected intention、visible episode、projected bids、expression policy、semantic affect、
semantic relationship、interaction style 和 permitted_action_results 为语境。

visible_boundaries 只记录当前生效的表达限制或细节范围；addressee_plan 只记录真实存在的语义称呼
安排。没有相应约束时返回空列表，让角色按当前判断自然表达。能力结果的 status 按原义处理：只有
executed 支持其有界的已完成效果，其他 status 保留各自含义。角色自己的反思属于语境，不是当前
用户的即时发言；运行元数据不属于对话内容。

新生成的自由文本使用简体中文；用户引文、专有名词、代码、URL 以及 schema 或 enum token 保持
原样。本阶段返回规划字段，不写最终对话。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 visible_boundaries 和 addressee_plan。每个字段都是包含
零到八个非空字符串的列表，列表内不得重复，每条最多 500 字符。'''


async def run_preference_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> tuple[list[str], list[str]]:
    """Run the stage-local preference prompt and return two distinct lists."""

    prompt_text = _surface_prompt_text(payload)
    preference_llm = services.llm
    response = await preference_llm.ainvoke(
        [
            SystemMessage(content=PREFERENCE_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ],
        config=services.preference_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, Mapping) or set(parsed) != {
        "visible_boundaries",
        "addressee_plan",
    }:
        raise ValueError("preference stage fields are not exact")
    return (
        _bounded_text_list(
            parsed["visible_boundaries"],
            "visible boundaries",
            minimum=0,
        ),
        _bounded_text_list(
            parsed["addressee_plan"],
            "addressee plan",
            minimum=0,
        ),
    )


VISUAL_SYSTEM_PROMPT = '''根据 selected intention、visible episode、projected bids、
expression policy、semantic affect、semantic relationship、permitted action results、
interaction style context 和 character_voice_context，为终端图像表面生成 visual_directives。
指导可以包含服务于 selected surface intent 的可见角色特征、姿势、表情、构图、环境与场景氛围。
这些内容是私有的图像生成指导，不是发送给用户的文字、对话指导，也不是调用其他模型或处理器的
指令。本阶段不写最终对话。

新生成的自由文本使用简体中文；用户引文、专有名词、代码、URL 以及必要的 schema 或 enum token
保持原样。角色自己的反思或内部观察属于证据，不是当前用户的即时发言。输出中不复述来源包标题、
时间戳、传输摘要、schema key 或运行元数据。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 visual_directives，其值是一个非空字符串，最多 1000 字符。'''


async def run_visual_stage(
    payload: Mapping[str, Any],
    services: VisualSurfaceServicesV2,
) -> str:
    """Run the stage-local visual prompt and validate its exact field."""

    prompt_text = _surface_prompt_text(payload)
    visual_llm = services.llm
    response = await visual_llm.ainvoke(
        [
            SystemMessage(content=VISUAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ],
        config=services.visual_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, Mapping) or set(parsed) != {"visual_directives"}:
        raise ValueError("visual stage fields are not exact")
    visual_directives = _bounded_text(
        parsed["visual_directives"],
        "visual directives",
        1000,
    )
    return visual_directives


def _surface_prompt_text(payload: Mapping[str, Any]) -> str:
    """Serialize one already-projected surface packet within the fixed cap."""

    prompt_text = json.dumps(
        {"surface": payload},
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(prompt_text) > 24000:
        raise ValueError("surface stage prompt exceeds the contract cap")
    return prompt_text


def _bounded_text(value: Any, label: str, maximum: int) -> str:
    """Validate one bounded non-empty stage-owned text field."""

    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} is invalid")
    return value


def _bounded_text_list(
    value: Any,
    label: str,
    *,
    minimum: int = 1,
) -> list[str]:
    """Validate one duplicate-free text list against its stage cardinality."""

    if not isinstance(value, list) or not minimum <= len(value) <= 8:
        raise ValueError(f"{label} is invalid")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    for item in value:
        _bounded_text(item, label, 500)
    return list(value)
