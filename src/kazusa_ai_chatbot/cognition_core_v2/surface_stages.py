"""Four bounded V2 text-surface stage handlers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
    TextSurfaceServicesV2,
    VisualSurfaceServicesV2,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


SURFACE_STAGE_ATTEMPT_LIMIT = 2
SURFACE_STAGE_REPAIR_OUTPUT_CAP = 8000
SURFACE_STAGE_REPAIR_PROMPT_CAP = 24000

_SURFACE_REPAIR_PROMPT = '''上一份 surface 阶段输出没有通过当前节点的 contract 校验。
请在完全相同的 surface 语境和语义判断下，生成一份完整替代对象。保留原始的角色判断、
情绪方向、关系方向、selected intention、能力结果和事实；只修复字段集合、字段类型、长度、
列表基数和 JSON 语法。当前上下文是中文时，所有新生成的自由文本和角色称谓使用简体中文；
schema key、ID、URL、代码、命令、enum token 和用户原文按原样保留。只返回当前阶段规定的
JSON 对象，不添加解释、markdown 或额外字段。'''


STYLE_SYSTEM_PROMPT = '''根据 expression policy、semantic affect、semantic relationship、
interaction style 和 character_voice_context 选择适合当前语境的说话方式。本阶段只决定词语层次、
句式、节奏、停顿与标点，在不改变 selected intention 和 bid 含义、也不增加内容节点的前提下，
把角色声音、情绪和互动姿态融入用词、句式与节奏，形成适合聊天发送的表达指导。

角色自己的反思和内部观察属于语境，不是当前用户的即时发言；运行元数据也不属于措辞内容。
新生成的自由文本使用简体中文；用户引文、专有名词、代码、URL 以及 schema 或 enum token 保持
原样。内部角色句柄或英文角色称谓仅作为结构化值或原文内容保留；中文自由文本使用配置名称、当前
角色、当前用户或其他参与者。本阶段返回风格指导，不写最终对话。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 style_guidance，其值是一个非空字符串，最多 1000 字符。'''


async def run_style_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> str:
    """Run the stage-local style prompt and validate its exact field."""

    return await _run_surface_stage(
        payload=payload,
        system_prompt=STYLE_SYSTEM_PROMPT,
        llm=services.llm,
        config=services.style_config,
        stage_name="style",
        validator=_validate_style_result,
    )


CONTENT_PLAN_SYSTEM_PROMPT = '''规划当前角色在这个场景中实际会说出或发送的内容，使其自然表达
已经形成的角色判断。综合 selected intention、primary bid、supporting bid、visible episode、
semantic affect、semantic relationship、expression policy、interaction style 和
permitted_action_results。runtime_capability_limits 是运行时已经确认的能力边界；其中标记不可用的
能力不能被表达为已经安排、发送或完成，但可以自然表达当前限制、等待或下一步条件。

goal_resolution 是 cognition 对当前目标可回答性的已确认判断：answerable_now 表示可以在当前
证据范围内直接回答；requires_required_evidence、requires_user_input 和 blocked 表示目标仍未
完成。后面三种状态应把真实缺口、当前限制或下一步条件表达给用户，保持 selected intention 的
主题，不把尚未获得的证据写成已经读取、分析或完成的结果。
blocked 的可见表达停留在当前回合的真实边界，或请用户提供当前可访问的材料；它不产生新的未来
阅读、分析、回传或完成承诺。只有 permitted_action_results 中已经存在的实际结果，才能支持对应
的事实陈述或后续动作状态。runtime limits 若指出仓库代码读取 owner 不可用，直接表达“当前无法
读取”并邀请用户提供材料；不要把“先访问、先阅读、等我处理”当作当前可执行步骤。

# 规划步骤
1. 按认知阶段选择的方式回应或参与当前输入。结合角色关系、情绪和场景压力推进互动，而不是机械
复述先前对话。
2. 选择鲜明、贴合角色的内容。只要不与当前输入或明确生效的约束冲突，也不颠倒行动者、对象、
受益者或主语，连贯的想象细节、玩笑和有创造力的展开都可以使用。
3. 结构化 visible percept 的角色字段具有权威性。在用户对话中，“当前用户”的第一人称指当前
用户；“当前角色”表示当前角色，也是被直接称呼者和祈使句的隐含主语。自由文本使用自然的中文
参与者称呼。
4. permitted_action_results 是角色大脑能力的精确账本。只有 executed 支持其有界的已完成效果；
pending 或 scheduled 的正向语义是“已记录、已排队、待执行”，可在当前发言中确认请求和执行
条件，保持后续 worker 结果开放。其他 status 不支持完成声明；请求或目标候选只支持角色在言语中的
态度，不代表能力已经执行。
5. 只规划角色要表达的含义和互动推进，让情绪、关系强度与动作倾向通过台词含义、语气和节奏
呈现。

返回一份简洁计划，并给出一到八条语义要求，用来保护选定含义、当前真实边界、角色方向和能力
执行事实。角色自己的反思和内部观察属于语境，不是当前用户的即时发言；运行元数据不属于对话
内容。新生成的自由文本使用简体中文；用户引文、专有名词、代码、URL 以及 schema 或 enum token
保持原样。内部角色句柄或英文角色称谓仅作为结构化值或原文内容保留；中文自由文本使用配置名称、
当前角色、当前用户或其他参与者。本阶段不写最终对话。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 content_plan 和 content_requirements。
content_plan 是一个非空字符串，最多 1000 字符；content_requirements 是一到八条互不重复的
非空语义要求，每条最多 500 字符。'''


async def run_content_plan_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> tuple[str, list[str]]:
    """Run the content prompt and return its plan and semantic requirements."""

    return await _run_surface_stage(
        payload=payload,
        system_prompt=CONTENT_PLAN_SYSTEM_PROMPT,
        llm=services.llm,
        config=services.content_plan_config,
        stage_name="content_plan",
        validator=_validate_content_plan_result,
    )


PREFERENCE_SYSTEM_PROMPT = '''识别当前角色判断和当前场景中真实存在的可见表达边界或称呼对象约束。
以 selected intention、visible episode、projected bids、expression policy、semantic affect、
semantic relationship、interaction style 和 permitted_action_results 为语境。
runtime_capability_limits 是可信的运行时能力边界，只用于保持表达与现实能力一致。

visible_boundaries 只记录当前生效的表达限制或细节范围；addressee_plan 只记录真实存在的语义称呼
安排。没有相应约束时返回空列表，让角色按当前判断自然表达。能力结果的 status 按原义处理：只有
executed 支持其有界的已完成效果，其他 status 保留各自含义。角色自己的反思属于语境，不是当前
用户的即时发言；运行元数据不属于对话内容。

新生成的自由文本使用简体中文；用户引文、专有名词、代码、URL 以及 schema 或 enum token 保持
原样。内部角色句柄或英文角色称谓仅作为结构化值或原文内容保留；中文自由文本使用配置名称、当前
角色、当前用户或其他参与者。本阶段返回规划字段，不写最终对话。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 visible_boundaries 和 addressee_plan。每个字段都是包含
零到八个非空字符串的列表，列表内不得重复，每条最多 500 字符。'''


async def run_preference_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> tuple[list[str], list[str]]:
    """Run the stage-local preference prompt and return two distinct lists."""

    return await _run_surface_stage(
        payload=payload,
        system_prompt=PREFERENCE_SYSTEM_PROMPT,
        llm=services.llm,
        config=services.preference_config,
        stage_name="preference",
        validator=_validate_preference_result,
    )


VISUAL_SYSTEM_PROMPT = '''根据 selected intention、visible episode、projected bids、
expression policy、semantic affect、semantic relationship、permitted action results、
runtime_capability_limits、interaction style context 和 character_voice_context，为终端图像表面生成
visual_directives。
指导可以包含服务于 selected surface intent 的可见角色特征、姿势、表情、构图、环境与场景氛围。
这些内容是私有的图像生成指导，不是发送给用户的文字、对话指导，也不是调用其他模型或处理器的
指令。本阶段不写最终对话。

新生成的自由文本使用简体中文；用户引文、专有名词、代码、URL 以及必要的 schema 或 enum token
保持原样。内部角色句柄或英文角色称谓仅作为结构化值或原文内容保留；中文自由文本使用配置名称、
当前角色、当前用户或其他参与者。角色自己的反思或内部观察属于证据，不是当前用户的即时发言。输出中不复述来源包标题、
时间戳、传输摘要、schema key 或运行元数据。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 visual_directives，其值是一个非空字符串，最多 1000 字符。'''


async def run_visual_stage(
    payload: Mapping[str, Any],
    services: VisualSurfaceServicesV2,
) -> str:
    """Run the stage-local visual prompt and validate its exact field."""

    return await _run_surface_stage(
        payload=payload,
        system_prompt=VISUAL_SYSTEM_PROMPT,
        llm=services.llm,
        config=services.visual_config,
        stage_name="visual",
        validator=_validate_visual_result,
    )


async def _run_surface_stage(
    *,
    payload: Mapping[str, Any],
    system_prompt: str,
    llm: Any,
    config: Any,
    stage_name: str,
    validator: Callable[[object], Any],
) -> Any:
    """Run one surface owner with bounded parse, repair, and fail-closed handling."""

    prompt_text = _surface_prompt_text(payload)
    request_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_text),
    ]
    last_error: Exception | None = None
    for attempt_index in range(SURFACE_STAGE_ATTEMPT_LIMIT):
        try:
            response = await llm.ainvoke(
                request_messages,
                config=config,
            )
        except Exception as exc:
            last_error = exc
            if attempt_index + 1 >= SURFACE_STAGE_ATTEMPT_LIMIT:
                raise _surface_execution_error(
                    stage_name=stage_name,
                    error_code="provider_exhausted",
                    attempt_count=attempt_index + 1,
                    detail="surface 阶段模型调用在有界重试后仍未完成",
                ) from exc
            request_messages = _surface_repair_messages(
                payload=payload,
                invalid_candidate="",
                reason="上一轮模型调用未返回可用候选，请在相同语境下重新生成完整 JSON。",
            )
            continue

        parsed: object = {}
        response_text = getattr(response, "content", "")
        try:
            parsed = parse_llm_json_output(response_text)
            return validator(parsed)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt_index + 1 >= SURFACE_STAGE_ATTEMPT_LIMIT:
                raise _surface_execution_error(
                    stage_name=stage_name,
                    error_code="contract_exhausted",
                    attempt_count=attempt_index + 1,
                    detail="surface 阶段候选在有界重生成后仍未通过 contract 校验",
                ) from exc
            request_messages = _surface_repair_messages(
                payload=payload,
                invalid_candidate=str(response_text),
                reason="上一份候选未通过当前阶段的字段、类型、长度或 JSON contract 校验。",
            )

    raise _surface_execution_error(
        stage_name=stage_name,
        error_code="contract_exhausted",
        attempt_count=SURFACE_STAGE_ATTEMPT_LIMIT,
        detail=f"surface 阶段执行失败: {last_error}",
    )


def _surface_execution_error(
    *,
    stage_name: str,
    error_code: str,
    attempt_count: int,
    detail: str,
) -> CognitionExecutionError:
    """Build a typed failure that keeps the pre-state checkpoint explicit."""

    return CognitionExecutionError(
        detail,
        error_code=f"surface_{stage_name}_{error_code}",
        stage=f"surface.{stage_name}",
        attempt_count=attempt_count,
        safe_checkpoint="pre_state_commit",
        retryable=False,
    )


def _surface_repair_messages(
    *,
    payload: Mapping[str, Any],
    invalid_candidate: str,
    reason: str,
) -> list[SystemMessage | HumanMessage]:
    """Build a bounded same-context repair request with Chinese instructions."""

    repair_payload = {
        "surface": payload,
        "contract_repair": {
            "reason": reason,
            "invalid_candidate": _bounded_repair_text(invalid_candidate),
        },
    }
    prompt_text = json.dumps(
        repair_payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(prompt_text) > SURFACE_STAGE_REPAIR_PROMPT_CAP:
        prompt_text = json.dumps(
            {
                "surface": payload,
                "contract_repair": {
                    "reason": reason,
                    "invalid_candidate": "上一份候选已省略；请依据 surface 语境返回完整合法对象。",
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    return [
        SystemMessage(content=_SURFACE_REPAIR_PROMPT),
        HumanMessage(content=prompt_text),
    ]


def _bounded_repair_text(value: str) -> str:
    """Bound rejected model text before placing it in a repair prompt."""

    if len(value) <= SURFACE_STAGE_REPAIR_OUTPUT_CAP:
        return value
    half_cap = SURFACE_STAGE_REPAIR_OUTPUT_CAP // 2
    return (
        value[:half_cap]
        + "\n... 已截断的不合格候选 ...\n"
        + value[-half_cap:]
    )


def _validate_style_result(value: object) -> str:
    """Validate the exact style-stage object."""

    if not isinstance(value, Mapping) or set(value) != {"style_guidance"}:
        raise ValueError("style stage fields are not exact")
    return _bounded_text(value["style_guidance"], "style guidance", 1000)


def _validate_content_plan_result(value: object) -> tuple[str, list[str]]:
    """Validate the exact content-plan-stage object."""

    if not isinstance(value, Mapping) or set(value) != {
        "content_plan",
        "content_requirements",
    }:
        raise ValueError("content-plan stage fields are not exact")
    content_plan = _bounded_text(value["content_plan"], "content plan", 1000)
    content_requirements = _bounded_text_list(
        value["content_requirements"],
        "content requirements",
    )
    return content_plan, content_requirements


def _validate_preference_result(value: object) -> tuple[list[str], list[str]]:
    """Validate the exact preference-stage object."""

    if not isinstance(value, Mapping) or set(value) != {
        "visible_boundaries",
        "addressee_plan",
    }:
        raise ValueError("preference stage fields are not exact")
    return (
        _bounded_text_list(
            value["visible_boundaries"],
            "visible boundaries",
            minimum=0,
        ),
        _bounded_text_list(
            value["addressee_plan"],
            "addressee plan",
            minimum=0,
        ),
    )


def _validate_visual_result(value: object) -> str:
    """Validate the exact visual-stage object."""

    if not isinstance(value, Mapping) or set(value) != {"visual_directives"}:
        raise ValueError("visual stage fields are not exact")
    return _bounded_text(
        value["visual_directives"],
        "visual directives",
        1000,
    )


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
