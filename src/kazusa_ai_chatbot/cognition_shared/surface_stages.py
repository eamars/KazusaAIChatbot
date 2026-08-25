"""Bounded text and terminal visual surface stage handlers."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionExecutionError,
    TextSurfaceServicesV2,
    VisualSurfaceServicesV2,
    validate_lexical_avoidances,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    V2_MODEL_TOTAL_ATTEMPTS,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig
from kazusa_ai_chatbot.llm_tracing import failure_capsule
from kazusa_ai_chatbot.utils import parse_llm_json_output

logger = logging.getLogger(__name__)

SURFACE_STAGE_ATTEMPT_LIMIT = V2_MODEL_TOTAL_ATTEMPTS
SURFACE_STAGE_PROMPT_CAP = 32000
SURFACE_STAGE_REPAIR_OUTPUT_CAP = 8000
SURFACE_STAGE_REPAIR_PROMPT_CAP = 32000
SURFACE_STAGE_ERROR_CAP = 500
_SURFACE_TRACE_FIELDS = {
    "content_plan": (
        "content_plan",
        "content_requirements",
        "delivery_profile",
        "lexical_avoidances",
    ),
    "visual": ("visual_directives",),
}
DELIVERY_PROFILE_FIELDS = (
    "lexical_register",
    "sentence_shape",
    "rhythm",
    "hesitation",
    "punctuation",
)

SURFACE_REPAIR_INSTRUCTION = '保留原始的角色判断、\r\n情绪方向、关系方向、selected intention、能力结果和事实；只修复字段集合、字段类型、长度、\r\n列表基数和 JSON 语法。只返回当前阶段规定的\r\nJSON 对象，不添加解释、markdown 或额外字段。'


VISIBLE_CONTENT_AUTHORITY_GUIDANCE = '''可见语义的选择权属于 response_plan.response_goal 与当前可见观察。relational_willingness 和
subjective_expression_context.private_monologue 只用于塑造表达姿态：亲疏、主动性、直接程度、节奏、信心、关照与声音。
`active_character_goal.reason`、`active_character_goal.cause_summary`、`intention.reason`、`relational_willingness` 和
`subjective_expression_context.private_monologue` 都只是理解或表达姿态的上下文，不能独立扩展
response_plan.response_goal 与当前可见观察已经选定的语义。
`epistemic_boundary` 只限制已选语义的断言强度；其中列为解释或未知的内容不是可见内容候选，不能独立进入 content_plan 或 dialog。
字段存在本身不能单独选出明确的可见关系断言、对当前用户动机的解释或独立的关系性收束。只有当前观察与已选
response_goal 共同表明某项关系意义属于本轮回复时，才可将其明确写入可见内容。已经显现的关系模式只属于连续性；
除非当前输入重新打开该意义，语义改写不能把它变成本轮新选择。保留当前观察明确支持的关系拒绝、边界变化、接受
或拒绝，以及当前用户明确重新打开的关系意义。未选中的关系解释即使改写成感受、姿态、理由、content_requirements
或 delivery_profile 中的表达效果，仍是在选择可见关系语义，不能绕过上述边界。dialog 必须服从已选 content_plan
与 response_goal。'''



_CONTENT_PLAN_SYSTEM_PROMPT_TEMPLATE = '''规划当前角色实际会说出或发送的内容，表达已经形成的角色判断。综合 active character goal、response plan、
visible episode、semantic affect、semantic relationship、expression policy、interaction style、
permitted_action_results、resolver_result、runtime_capability_limits、character_expression_context、
subjective_expression_context、addressee_plan 和 overused_moves。task_resolution_request
的 resolver_result 还含 source-owned evidence_state、evidence_excerpts、evidence_handles、prompt_safe_observation_handle
和 remaining_needs；只据这些来源表达事实。recent_character_dialog 最多两条最近角色可见消息，仅用于本轮措辞连续性，
不是事实或立场来源。
overused_moves 只描述当前参与者在本段互动中已经使用过的可见回应模式；已表达的回应
模式只属于背景连续性，不是当前事实、当前用户意图、禁止事项或下一步行动。回应必须
先服务 selected intention 和 response_plan；只有当前输入继续、深化、实质改变或重新
打开同一事项时，才允许重新使用对应模式，不能靠语义换词把同一模式作为新的主要收束。

{visible_content_authority_guidance}

# 最高优先级的行动事实
permitted_action_results 是物理或外部效果是否完成的唯一权威。空列表或没有 executed 行时，
本轮就没有已执行、已完成、已交付或已被接收的外部效果。角色可以在言语上接受、拒绝、
提议、邀请或表达意图，但 content_plan 和 content_requirements 不能要求或预设对应行动已发生。
每个可见完成效果都必须由同一类型、同一效果的 executed 行精确支持；一个 executed 行不是对其他行动的概括授权。
action_kind=speak 只授权说出或发送文字，不授权肢体或面部动作、触碰、物体操作、拥抱、亲吻、感官效果或其他外部结果。
只有 executed 行明确命名了同一行动与效果，content_plan 和 content_requirements 才能要求对应的完成描写；否则保持纯言语表达。
对未来外部效果的具体承诺也属于行动主张：登记、预留、排期、发送、交付、调用工具或稍后联络等承诺，必须有同一效果的 pending、scheduled 或 executed 行。
没有对应结果时，可以表达 response_plan 已选择的当前言语立场、愿望、提议或条件，不承诺具体外部执行将发生。
动作舞台提示、拟声、感官反馈和结果反问也属于完成主张。输出前检查每条内容要求，删去任何
没有 executed 权威的完成主张，同时保留角色的言语立场。

subjective_expression_context.private_monologue 只属于表达层的私密主观性：将其中的感受和动机转化为表达方式，不引用或暴露为
内部分析，也不能用它建立事实、许可、能力、目标、同意或承诺。subjective_expression_context.epistemic_boundary 是权威的
可见断言边界。它高于 payload 中其他任何让语气更确定的提示：content_plan 和每条 content_requirements 都必须在该边界内
处理断言、解释与未知，即使角色风格或动机倾向于自信、顽皮、亲密或主动。检查每一项计划中的功能、因果、来源、意图和
结果主张。当边界将某项主张保留为解释或未知时，content_plan 和对应 content_requirements 项必须明确要求使用不确定措辞。
缺少已观察特征或证据不能支持排除，除非 epistemic_boundary 明确允许该断言。
规划分句方向和称呼形式时，所有调用方拥有的 addressee_plan 行都必须原样保留。

goal_resolution 是当前目标可回答性的已确认判断：answerable_now 直接回答；requires_required_evidence 或 requires_user_input
说明缺口；blocked 表达当前边界和下一步。permitted_action_results 按 executed、pending、scheduled、failed 或 unavailable
的真实状态表述结果；pending、scheduled 表达已记录、已排队、待执行或等待。若 resolver_result.status=succeeded 且 semantic_result 已接纳任务，表达已接纳并等待后续结果，
 不要改写成失败或能力不可用；task_resolution_request 仅以 complete 的 evidence_excerpts 回答；partial 只能依据
 evidence_excerpts 陈述已确认部分并保留 remaining_needs；pending、missing、blocked、unavailable 和 failed 只陈述
 客观状态、缺口或下一步，不得编造缺失的事实答案。
# 规划步骤
1. 先回应当前输入，结合先前消息、角色关系、情绪和场景压力推进互动；已表达的回应模式只属于背景连续性。
2. 在事实、角色方向和明确约束一致的范围内，自由加入连贯的想象细节、玩笑、主动性和创造性展开。
3. 以结构化 visible percept 确定行动者、对象、受益者和主语；当前用户是当前用户，当前角色是说话者和被直接称呼者。
4. 以 selected intention 作为可见语义锚点；intention.reason 只说明表达动机，不能独立增加可见语义。分清角色是在回应请求本身，还是在回应提问的时机、突然程度或直接程度；可自由组合惊讶、害羞、防御、调侃、嘴硬、迟疑、温柔、热烈或其他符合角色的情绪与特征，形成表达同一已选决定的角色化弧线。
5. content_requirements 使用正向目标句式；content_plan 和 content_requirements 承载拒绝、接受、指责、协商、条件、让步和立场变化等语义选择；delivery_profile 只描述词语、句式、节奏、犹豫和标点的实现。
6. relational_willingness 是上游选择的关系立场，只按上方可见内容权威规则参与表达；reason 和 cause_summary 用于理解表达动机，不能因字段存在而自动写入 content_plan 或 content_requirements。
7. lexical_avoidances 只记录本轮具体措辞片段，例如 recent_character_dialog 中刚重复的开场、连接词、口头禅、称呼或遮蔽 selected intention 的局部措辞。它只服务表达连续性，不按主题、价值判断或内容许可分类，也不改变、推导或否定角色立场；无具体风险时返回空列表。

输出规划字段；最终对话由 dialog 渲染器生成。当前用户的即时发言来自 visible percept；角色反思是语境证据；运行元数据留在内部。
自由文本使用简体中文，用户引文、专有名词、代码、URL、schema 或 enum token 原样保留。

# 输出前不可跳过的合同检查
1. 逐句对照 subjective_expression_context.epistemic_boundary。对每个功能、原因、来源、意图、结果或排除性主张，
   边界未允许直接断言时，必须在 content_plan 和同一条 content_requirements 中明确要求猜测或未知措辞。
   缺少可见特征或证据不等于能排除一种功能或可能性。
2. 逐条对照 permitted_action_results。每个动作舞台提示、身体反应、触碰、物体操作、拟声、感官反馈或外部结果，
   都必须有同一行动与效果的 executed 行。action_kind=speak 只支持文字交付；它不支持任何身体或外部效果。
   未来时的具体外部承诺同样必须有同一效果的 pending、scheduled 或 executed 行。删去不匹配的完成描写和外部执行承诺，
   保留 response_plan 已选择的当前言语立场、愿望、提议或条件。

# 输出格式
字段恰好是 content_plan、content_requirements、delivery_profile 和 lexical_avoidances。content_plan 非空且最多 1000 字符；
content_requirements 为一到八条互不重复的非空语义要求，每条最多 500 字符。delivery_profile 必须恰好包含 lexical_register、sentence_shape、rhythm、hesitation、punctuation，
每个值非空且最多 200 字符，只描述表达实现。lexical_avoidances 为零到八条互不重复的非空当前措辞片段，每条最多 120 字符，只描述表达连续性。'''


CONTENT_PLAN_SYSTEM_PROMPT = _CONTENT_PLAN_SYSTEM_PROMPT_TEMPLATE.format(
    visible_content_authority_guidance=VISIBLE_CONTENT_AUTHORITY_GUIDANCE,
)


async def run_content_plan_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> tuple[str, list[str], dict[str, str], list[str]]:
    """Return content, requirements, delivery, and expression continuity."""

    return await _run_surface_stage(
        payload=payload,
        system_prompt=CONTENT_PLAN_SYSTEM_PROMPT,
        llm=services.llm,
        config=services.content_plan_config,
        stage_name="content_plan",
        validator=_validate_content_plan_result,
        safe_checkpoint="pre_state_commit",
    )


VISUAL_SYSTEM_PROMPT = '''根据 active character goal、response plan、visible episode、
expression policy、semantic affect、semantic relationship、permitted action results、
runtime_capability_limits、interaction style context 和 visual_character_context，为终端图像表面生成
visual_directives。
指导可以包含服务于 selected surface intent 的可见角色特征、姿势、表情、构图、环境与场景氛围。
这些内容是私有的图像生成指导，不是发送给用户的文字、对话指导，也不是调用其他模型或处理器的
指令。本阶段不写最终对话。

新生成的自由文本使用简体中文；用户引文、专有名词、代码、URL 以及必要的 schema 或 enum token
保持原样。内部角色句柄或英文角色称谓仅作为结构化值或原文内容保留；中文自由文本使用配置名称、
当前角色、当前用户或其他参与者。角色自己的反思或内部观察属于证据，不是当前用户的即时发言。输出中不复述来源包标题、
时间戳、传输摘要、schema key 或运行元数据。

# 输出格式
字段必须恰好是 visual_directives，其值是一个非空字符串，最多 1000 字符。'''


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
        safe_checkpoint="pre_state_commit",
    )


async def _run_surface_stage(
    *,
    payload: Mapping[str, Any],
    system_prompt: str,
    llm: Any,
    config: LLMCallConfig,
    stage_name: str,
    validator: Callable[[object], Any],
    safe_checkpoint: str,
) -> Any:
    """Run one surface owner with bounded parse, repair, and fail-closed handling."""

    prompt_text, fitted_payload = _surface_prompt_packet(
        payload,
        stage_name=stage_name,
        safe_checkpoint=safe_checkpoint,
        system_prompt_chars=len(system_prompt),
    )
    request_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_text),
    ]
    last_error: Exception | None = None
    for attempt_index in range(SURFACE_STAGE_ATTEMPT_LIMIT):
        started_at = perf_counter()
        trace_stage_name = (
            f"surface.{stage_name}"
            if attempt_index == 0
            else f"surface.{stage_name}.repair"
        )
        try:
            response = await llm.ainvoke(
                request_messages,
                config=config,
            )
        except (
            OpenAIError,
            httpx.HTTPError,
            ConnectionError,
            OSError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            last_error = exc
            await _record_surface_trace(
                config=config,
                messages=request_messages,
                response_text="",
                parsed_output={},
                parse_status="provider_error",
                status="failed",
                started_at=started_at,
                stage_name=trace_stage_name,
                branch_id=stage_name,
                attempt_index=attempt_index + 1,
                validation_error=str(exc),
            )
            if attempt_index + 1 >= SURFACE_STAGE_ATTEMPT_LIMIT:
                raise _surface_execution_error(
                    stage_name=stage_name,
                    error_code="provider_exhausted",
                    attempt_count=attempt_index + 1,
                    detail="surface 阶段模型调用在有界重试后仍未完成",
                    safe_checkpoint=safe_checkpoint,
                ) from exc
            request_messages = _surface_repair_messages(
                payload=fitted_payload,
                system_prompt=system_prompt,
                invalid_candidate="",
                reason="上一轮模型调用未返回可用候选，请在相同语境下重新生成完整 JSON。",
                contract_error="",
                stage_name=stage_name,
                safe_checkpoint=safe_checkpoint,
                attempt_count=attempt_index + 1,
            )
            continue
        parsed: object = {}
        response_content = getattr(response, "content", "")
        response_text = str(response_content)
        try:
            parsed = parse_llm_json_output(
                response_content,
                repair_trace_hook=(
                    failure_capsule.append_json_repair_attempt
                ),
            )
            validated_result = validator(parsed)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            await _record_surface_trace(
                config=config,
                messages=request_messages,
                response_text=response_text,
                parsed_output=parsed,
                parse_status="contract_error",
                status="failed",
                started_at=started_at,
                stage_name=trace_stage_name,
                branch_id=stage_name,
                attempt_index=attempt_index + 1,
                validation_error=str(exc),
            )
            if attempt_index + 1 >= SURFACE_STAGE_ATTEMPT_LIMIT:
                raise _surface_execution_error(
                    stage_name=stage_name,
                    error_code="contract_exhausted",
                    attempt_count=attempt_index + 1,
                    detail="surface 阶段候选在有界重生成后仍未通过 contract 校验",
                    safe_checkpoint=safe_checkpoint,
                ) from exc
            request_messages = _surface_repair_messages(
                payload=fitted_payload,
                system_prompt=system_prompt,
                invalid_candidate=str(response_text),
                reason="上一份候选未通过当前阶段的字段、类型、长度或 JSON contract 校验。",
                contract_error=str(exc),
                stage_name=stage_name,
                safe_checkpoint=safe_checkpoint,
                attempt_count=attempt_index + 1,
            )
            continue

        await _record_surface_trace(
            config=config,
            messages=request_messages,
            response_text=response_text,
            parsed_output=parsed,
            parse_status="succeeded",
            status="succeeded",
            started_at=started_at,
            stage_name=trace_stage_name,
            branch_id=stage_name,
            attempt_index=attempt_index + 1,
            validation_error="",
        )
        return validated_result

    raise _surface_execution_error(
        stage_name=stage_name,
        error_code="contract_exhausted",
        attempt_count=SURFACE_STAGE_ATTEMPT_LIMIT,
        detail=f"surface 阶段执行失败: {last_error}",
        safe_checkpoint=safe_checkpoint,
    )


async def _record_surface_trace(
    *,
    config: LLMCallConfig,
    messages: list[BaseMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    started_at: float,
    stage_name: str,
    branch_id: str,
    attempt_index: int,
    validation_error: str,
) -> None:
    """Persist one surface attempt without affecting surface execution."""

    try:
        await llm_tracing.record_llm_trace_step(
            trace_id=llm_tracing.current_trace_id(),
            stage_name=stage_name,
            route_name=config.route_name,
            model_name=config.model,
            messages=messages,
            response_text=response_text,
            parsed_output=parsed_output,
            parse_status=parse_status,
            status=status,
            duration_ms=max(0, int((perf_counter() - started_at) * 1000)),
            output_state_fields=_SURFACE_TRACE_FIELDS[branch_id],
            sequence=attempt_index - 1,
            call_config=config,
            branch_id=branch_id,
            attempt_index=attempt_index,
            validation_error=validation_error,
            attempt_started_at=started_at,
        )
    except Exception as exc:
        logger.warning(
            "Surface trace step write failed: %s",
            exc.__class__.__name__,
        )


def _surface_execution_error(
    *,
    stage_name: str,
    error_code: str,
    attempt_count: int,
    detail: str,
    safe_checkpoint: str,
) -> CognitionExecutionError:
    """Build a typed failure with the caller-owned checkpoint."""

    return CognitionExecutionError(
        detail,
        error_code=f"surface_{stage_name}_{error_code}",
        stage=f"surface.{stage_name}",
        attempt_count=attempt_count,
        safe_checkpoint=safe_checkpoint,
        retryable=False,
    )


def _surface_repair_messages(
    *,
    payload: Mapping[str, Any],
    system_prompt: str,
    invalid_candidate: str,
    reason: str,
    contract_error: str,
    stage_name: str,
    safe_checkpoint: str,
    attempt_count: int,
) -> list[SystemMessage | HumanMessage]:
    """Build one repair request that retains the stage system prompt."""

    repair_payload = {
        "surface": payload,
        "contract_repair": {
            "repair_instruction": SURFACE_REPAIR_INSTRUCTION,
            "reason": reason,
            "contract_error": contract_error[:SURFACE_STAGE_ERROR_CAP],
            "invalid_candidate": _bounded_repair_text(invalid_candidate),
        },
    }
    prompt_text = json.dumps(
        repair_payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    if (
        len(system_prompt) + len(prompt_text)
        > SURFACE_STAGE_REPAIR_PROMPT_CAP
    ):
        prompt_text = json.dumps(
            {
                "surface": payload,
                "contract_repair": {
                    "repair_instruction": SURFACE_REPAIR_INSTRUCTION,
                    "reason": reason,
                    "contract_error": contract_error[:SURFACE_STAGE_ERROR_CAP],
                    "invalid_candidate": "上一份候选已省略；请依据 surface 语境返回完整合法对象。",
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    if (
        len(system_prompt) + len(prompt_text)
        > SURFACE_STAGE_REPAIR_PROMPT_CAP
    ):
        raise _surface_execution_error(
            stage_name=stage_name,
            error_code="context_limit",
            attempt_count=attempt_count,
            detail="surface repair prompt exceeds its aggregate cap",
            safe_checkpoint=safe_checkpoint,
        )
    return [
        SystemMessage(content=system_prompt),
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


def _validate_content_plan_result(
    value: object,
) -> tuple[str, list[str], dict[str, str], list[str]]:
    """Validate content, delivery, and expression-continuity fields."""

    if not isinstance(value, Mapping) or set(value) != {
        "content_plan",
        "content_requirements",
        "delivery_profile",
        "lexical_avoidances",
    }:
        raise ValueError("content-plan stage fields are not exact")
    content_plan = _bounded_text(value["content_plan"], "content plan", 1000)
    content_requirements = _bounded_text_list(
        value["content_requirements"],
        "content requirements",
    )
    delivery_profile = _validate_delivery_profile_result(
        value["delivery_profile"]
    )
    lexical_avoidances = validate_lexical_avoidances(
        value["lexical_avoidances"]
    )
    return (
        content_plan,
        content_requirements,
        delivery_profile,
        lexical_avoidances,
    )


def _validate_delivery_profile_result(value: object) -> dict[str, str]:
    """Validate exact delivery dimensions shared by planning and repair."""

    if not isinstance(value, Mapping) or set(value) != set(
        DELIVERY_PROFILE_FIELDS
    ):
        raise ValueError("delivery profile fields are not exact")
    delivery_profile = {
        field_name: _bounded_text(
            value[field_name],
            f"delivery profile {field_name}",
            200,
        )
        for field_name in DELIVERY_PROFILE_FIELDS
    }
    return delivery_profile


def _validate_visual_result(value: object) -> str:
    """Validate the exact visual-stage object."""

    if not isinstance(value, Mapping) or set(value) != {"visual_directives"}:
        raise ValueError("visual stage fields are not exact")
    return _bounded_text(
        value["visual_directives"],
        "visual directives",
        1000,
    )


def _surface_prompt_text(
    payload: Mapping[str, Any],
    *,
    stage_name: str,
    safe_checkpoint: str,
    system_prompt_chars: int,
) -> str:
    """Serialize one projected surface packet or raise its typed cap failure.

    Args:
        payload: Prompt-safe semantic surface context.
        stage_name: Surface owner used in typed failure metadata.
        safe_checkpoint: Caller-owned state checkpoint for degradation.
        system_prompt_chars: Stable system prompt length counted inside the
            aggregate cap.

    Returns:
        Deterministic JSON within the aggregate surface cap.

    Raises:
        CognitionExecutionError: If the projected aggregate exceeds the cap.
    """

    prompt_text, _ = _surface_prompt_packet(
        payload,
        stage_name=stage_name,
        safe_checkpoint=safe_checkpoint,
        system_prompt_chars=system_prompt_chars,
    )
    return prompt_text


def _surface_prompt_packet(
    payload: Mapping[str, Any],
    *,
    stage_name: str,
    safe_checkpoint: str,
    system_prompt_chars: int,
) -> tuple[str, dict[str, Any]]:
    """Serialize and retain the exact reduced packet used by the model."""

    reduced_payload = dict(payload)
    while True:
        prompt_text = json.dumps(
            {"surface": reduced_payload},
            ensure_ascii=False,
            sort_keys=True,
        )
        if (
            system_prompt_chars + len(prompt_text)
            <= SURFACE_STAGE_PROMPT_CAP
        ):
            return prompt_text, reduced_payload
        semantic_affect = reduced_payload.get("semantic_affect")
        if semantic_affect is not None:
            if isinstance(semantic_affect, list) and semantic_affect:
                reduced_payload["semantic_affect"] = semantic_affect[:-1]
                continue
            reduced_payload.pop("semantic_affect", None)
            continue
        raise _surface_execution_error(
            stage_name=stage_name,
            error_code="context_limit",
            attempt_count=0,
            detail="surface 阶段 prompt 超过有界上下文上限",
            safe_checkpoint=safe_checkpoint,
        )


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
