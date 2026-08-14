"""Bounded V2 text and terminal visual surface stage handlers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
    TextSurfaceServicesV2,
    VisualSurfaceServicesV2,
    validate_surface_addressee_plan,
    validate_lexical_avoidances,
)
from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    V2_MODEL_TOTAL_ATTEMPTS,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig
from kazusa_ai_chatbot.llm_tracing import failure_capsule
from kazusa_ai_chatbot.utils import parse_llm_json_output


SURFACE_STAGE_ATTEMPT_LIMIT = V2_MODEL_TOTAL_ATTEMPTS
SURFACE_STAGE_PROMPT_CAP = 32000
SURFACE_STAGE_REPAIR_OUTPUT_CAP = 8000
SURFACE_STAGE_REPAIR_PROMPT_CAP = 32000
SURFACE_STAGE_ERROR_CAP = 500
MIN_SURFACE_SUPPORTING_BIDS = 2
DELIVERY_PROFILE_FIELDS = (
    "lexical_register",
    "sentence_shape",
    "rhythm",
    "hesitation",
    "punctuation",
)

SURFACE_REPAIR_INSTRUCTION = '保留原始的角色判断、\r\n情绪方向、关系方向、selected intention、能力结果和事实；只修复字段集合、字段类型、长度、\r\n列表基数和 JSON 语法。只返回当前阶段规定的\r\nJSON 对象，不添加解释、markdown 或额外字段。'



CONTENT_PLAN_SYSTEM_PROMPT = '''规划当前角色实际会说出或发送的内容，表达已经形成的角色判断。综合 selected intention、primary bid、
supporting bid、visible episode、semantic affect、semantic relationship、expression policy、interaction style、
permitted_action_results、resolver_result、runtime_capability_limits 和 character_expression_context。task_resolution_request
的 resolver_result 还含 source-owned evidence_state、evidence_excerpts、evidence_handles、prompt_safe_observation_handle
和 remaining_needs；只据这些来源表达事实。recent_character_dialog 最多两条最近角色可见消息，仅用于本轮措辞连续性，
不是事实或立场来源。

goal_resolution 是当前目标可回答性的已确认判断：answerable_now 直接回答；requires_required_evidence 或 requires_user_input
说明缺口；blocked 表达当前边界和下一步。permitted_action_results 按 executed、pending、scheduled、failed 或 unavailable
的真实状态表述结果；pending、scheduled 表达已记录、已排队、待执行或等待。若 resolver_result.status=succeeded 且 semantic_result 已接纳任务，表达已接纳并等待后续结果，
 不要改写成失败或能力不可用；task_resolution_request 仅以 complete 的 evidence_excerpts 回答；partial 只能依据
 evidence_excerpts 陈述已确认部分并保留 remaining_needs；pending、missing、blocked、unavailable 和 failed 只陈述
 客观状态、缺口或下一步，不得编造缺失的事实答案。

# 规划步骤
1. 回应当前输入，结合先前消息、角色关系、情绪和场景压力推进互动。
2. 在事实、角色方向和明确约束一致的范围内，自由加入连贯的想象细节、玩笑、主动性和创造性展开。
3. 以结构化 visible percept 确定行动者、对象、受益者和主语；当前用户是当前用户，当前角色是说话者和被直接称呼者。
4. 以 selected intention 及 intention.reason 为语义锚点，分清角色是在回应请求本身，还是在回应提问的时机、突然程度或直接程度；可自由组合惊讶、害羞、防御、调侃、嘴硬、迟疑、温柔、热烈或其他符合角色的情绪与特征，形成表达同一已选决定的角色化弧线。
5. content_requirements 使用正向目标句式；content_plan 和 content_requirements 承载拒绝、接受、指责、协商、条件、让步和立场变化等语义选择；delivery_profile 只描述词语、句式、节奏、犹豫和标点的实现。
6. relational_willingness 是上游已选择的角色关系立场（含 current_user_relationship_state）；内容字段保持其原样 stance。若权威语境选择立场变化，把新事实、动机、条件或约束写入内容字段。
7. lexical_avoidances 只记录本轮具体措辞片段，例如 recent_character_dialog 中刚重复的开场、连接词、口头禅、称呼或遮蔽 selected intention 的局部措辞。它只服务表达连续性，不按主题、价值判断或内容许可分类，也不改变、推导或否定角色立场；无具体风险时返回空列表。

只返回规划字段；最终对话由 dialog 渲染器生成。当前用户的即时发言来自 visible percept；角色反思是语境证据；运行元数据留在内部。
自由文本使用简体中文，用户引文、专有名词、代码、URL、schema 或 enum token 原样保留。

# 输出格式
只返回一个 JSON 对象，字段恰好是 content_plan、content_requirements、delivery_profile 和 lexical_avoidances。content_plan 非空且最多 1000 字符；
content_requirements 为一到八条互不重复的非空语义要求，每条最多 500 字符。delivery_profile 必须恰好包含 lexical_register、sentence_shape、rhythm、hesitation、punctuation，
每个值非空且最多 200 字符，只描述表达实现。lexical_avoidances 为零到八条互不重复的非空当前措辞片段，每条最多 120 字符，只描述表达连续性。'''


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


PREFERENCE_SYSTEM_PROMPT = '''识别当前角色判断和场景中真实存在的表达边界与称呼安排。以 selected intention、visible episode、projected bids、
expression policy、semantic affect、semantic relationship、interaction style 和 permitted_action_results 为语境；
relational_willingness、resolver_result 按原义保留。task_resolution_request 的 resolver_result 中 evidence_state、
evidence_excerpts、evidence_handles 和 remaining_needs 是来源自有的答案边界；complete 只支持 supplied excerpts，
 partial 只支持 supplied excerpts 和 remaining_needs；其他状态保留事实缺口。runtime_capability_limits 只约束现实能力。

当前版本没有 typed source-bound visible-boundary contract，因此 visible_boundaries 始终返回空列表；不要从关系、情绪、主题、能力
或一般常识推导可见边界。每一条 addressee_plan 都对应真实存在的称呼安排；逐条保留输入的 handle、display_name、semantic_role 和 wording_policy，不新增、删除、
改名或把第三方改成 current_user。相应约束为空时返回空列表。普通场景事实、时间、情绪、关系状态和已选回应立场分别归入内容规划字段；拒绝、接受、指责、协商、条件和立场变化归入 content_plan 或 content_requirements；
dialog 生成最终对话。

当前用户的即时发言来自 visible percept；角色反思是语境证据；运行元数据留内部。自由文本使用简体中文；用户引文、专有名词、
代码、URL、schema 或 enum token 原样保留；中文自由文本使用当前角色、当前用户或其他参与者。

# 输出格式
只返回一个 JSON 对象，字段恰好是 visible_boundaries 和 addressee_plan。visible_boundaries 是零到八个非空且唯一的字符串，
每条最多 500 字符；addressee_plan 是零到八个对象，每个对象恰好包含 handle、display_name、semantic_role 和 wording_policy，
并逐字保留输入的结构化目标行。'''


async def run_preference_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Run the stage-local preference prompt and return two distinct lists."""

    return await _run_surface_stage(
        payload=payload,
        system_prompt=PREFERENCE_SYSTEM_PROMPT,
        llm=services.llm,
        config=services.preference_config,
        stage_name="preference",
        validator=_validate_preference_result,
        safe_checkpoint="pre_state_commit",
    )


DIALOG_COMPLIANCE_REPAIR_SYSTEM_PROMPT = '''你负责在最终对话未通过硬错误检查后，重新生成一份
完整的文本 surface 语义。surface 中的 episode、intention、goal_resolution、supporting bids、
expression policy、semantic affect、semantic relationship、interaction style、
character_expression_context、permitted_action_results 和 runtime_capability_limits 是本轮权威
 语境。resolver_result 是本轮 resolver capability 的来源自有执行结果；recent_character_dialog 是最多两条
最近角色可见消息，只用于本轮措辞连续性。

surface.dialog_compliance_repair.verified_hard_issues 是已经确认的硬错误；

# 修复步骤
1. 以 selected intention 及 intention.reason 为语义锚点，保持角色判断、情绪方向、关系方向、
当前事实和能力执行结果。阅读完整语境，分清角色是在回应请求本身，还是在回应提问的时机、突然
程度或直接程度。可自由组合惊讶、害羞、防御、调侃、嘴硬、迟疑、温柔、热烈或其他符合角色的情绪
与特征。这些表达可以先于明确决定出现，并与收尾共同组成表达同一已选决定的角色化弧线。权威语境
选择实际立场变化时，把支持变化的新事实、动机、条件、让步或约束及其因果连接写入 content_plan
或 content_requirements。
surface.relational_willingness 是上游已选择的角色关系立场（含当前用户关系状态），替代内容必须保持其立场。
2. 修正每一项 verified_hard_issues，重新生成内容计划、语义要求、delivery profile 和
lexical_avoidances；lexical_avoidances 只保留具体的本轮表达连续性片段，不表达主题许可、道德判断、
拒绝理由或新的角色立场。
visible_boundaries 保持为空，并保留权威语境中的称呼安排。
3. 保持结构化角色中的行动者、对象、受益者、回应所有者和选择所有者。当前角色可以拒绝、协商或
附加条件，并按照权威语境保持这些语义选择的行动者和对象。
4. 当前版本没有 typed source-bound visible-boundary contract，visible_boundaries 必须返回空列表；
不要自行添加未绑定来源的通用边界。每一条 addressee_plan 都对应真实
存在的称呼安排。输入中的结构化 addressee_plan 是上游
已经确认的参与者目标和 wording_policy；逐条保留其 handle、display_name、semantic_role 和
wording_policy，不得新增、删除、改名或把第三方改成 current_user。普通场景事实、时间、情绪、关系状态
和已选回应立场分别归入 content_plan、content_requirements 或 delivery_profile。拒绝、接受、
指责、协商、条件和立场变化归入 content_plan 或 content_requirements；主题、比喻和已选立场进入
content_plan 或 content_requirements；情绪、强度、直接程度和表达节奏归入 delivery_profile。
verified_hard_issues 中的内容冲突对应 content_plan 和 content_requirements 中的正向修复目标；
visible_boundaries 返回空列表；addressee_plan 逐字保留输入
提供的结构化参与者、语义角色和 wording_policy；
存在具体称呼安排时列出，其他情况返回空列表。亲密感、语气词、词汇、句式和节奏由
delivery_profile 表达。
5. 按 permitted_action_results 和 runtime_capability_limits 的原义重建状态：executed 对应有界
的完成效果，pending 或 scheduled 对应等待状态，其他 status 对应当前限制。
resolver_result 明确任务已接纳并将继续执行时，保留该等待后续结果的事实，不得改写为任务失败。
task_resolution_request 的 evidence_state=complete 只能依据 evidence_excerpts；partial 只能依据
evidence_excerpts 陈述已确认部分并保留 remaining_needs；pending、missing、blocked、unavailable 和
failed 必须保留缺口、等待或 typed blocker，只陈述客观状态与下一步，不得把 semantic_result 当作缺失
答案，也不得编造事实内容。
6. content_requirements 使用正向目标句式，描述回应应呈现的立场、情绪流动、角色特征、事实和
互动推进；delivery_profile 用词语层次、句式、节奏、犹豫和标点实现这些语义选择，让角色声音
保持鲜明。
7. selected surface intent、能力结果和运行时边界由调用方从权威输入重建。

新生成的自由文本使用简体中文；用户引文、专有名词、代码、URL 以及 schema 或 enum token 保持
原样。本阶段输出完整的替代规划字段。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 content_plan、content_requirements、delivery_profile、
lexical_avoidances、visible_boundaries 和 addressee_plan。content_plan 是一个非空字符串，
最多 1000 字符；content_requirements 是一到八条互不重复的非空语义要求；delivery_profile
必须恰好包含 lexical_register、sentence_shape、rhythm、hesitation 和 punctuation，每个值最多
200 字符；lexical_avoidances 是零到八条互不重复的非空当前措辞片段，每条最多 120 字符；
visible_boundaries 是零到八条互不重复的非空字符串；addressee_plan 是零到八个结构化
对象，每个对象恰好包含 handle、display_name、semantic_role 和 wording_policy。'''


async def run_dialog_compliance_repair_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> dict[str, Any]:
    """Replace rejected semantics with bounded structural regeneration.

    Args:
        payload: Canonical projected owner context and verified hard issues.
        services: Configured text-surface model and route settings.

    Returns:
        A validated complete replacement for all producer-owned fields.

    Raises:
        CognitionExecutionError: If all provider or contract attempts fail.
    """

    result = await _run_surface_stage(
        payload=payload,
        system_prompt=DIALOG_COMPLIANCE_REPAIR_SYSTEM_PROMPT,
        llm=services.llm,
        config=services.content_plan_config,
        stage_name="dialog_compliance_repair",
        validator=_validate_dialog_compliance_repair_result,
        safe_checkpoint="post_cognition_commit",
    )
    return result


VISUAL_SYSTEM_PROMPT = '''根据 selected intention、visible episode、projected bids、
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
            _record_surface_trace(
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
            _record_surface_trace(
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

        _record_surface_trace(
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


def _record_surface_trace(
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
    """Preserve one protected surface-owner model boundary."""

    failure_capsule.append_model_attempt(
        stage_name=stage_name,
        messages=messages,
        response_text=response_text,
        parsed_output=parsed_output,
        parse_status=parse_status,
        status=status,
        config=config,
        branch_id=branch_id,
        attempt_index=attempt_index,
        validation_error=validation_error,
        started_at=started_at,
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


def _validate_preference_result(
    value: object,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Validate the exact preference-stage object."""

    if not isinstance(value, Mapping) or set(value) != {
        "visible_boundaries",
        "addressee_plan",
    }:
        raise ValueError("preference stage fields are not exact")
    if value["visible_boundaries"] != []:
        raise ValueError(
            "visible boundaries must remain empty until a typed source contract exists"
        )
    visible_boundaries: list[str] = []
    addressee_plan = value["addressee_plan"]
    validate_surface_addressee_plan(addressee_plan)
    return visible_boundaries, [dict(row) for row in addressee_plan]


def _validate_dialog_compliance_repair_result(
    value: object,
) -> dict[str, Any]:
    """Validate the complete semantic replacement returned by the L3 owner."""

    if not isinstance(value, Mapping) or set(value) != {
        "content_plan",
        "content_requirements",
        "delivery_profile",
        "lexical_avoidances",
        "visible_boundaries",
        "addressee_plan",
    }:
        raise ValueError("dialog compliance surface repair fields are not exact")
    (
        content_plan,
        content_requirements,
        delivery_profile,
        lexical_avoidances,
    ) = (
        _validate_content_plan_result({
            "content_plan": value["content_plan"],
            "content_requirements": value["content_requirements"],
            "delivery_profile": value["delivery_profile"],
            "lexical_avoidances": value["lexical_avoidances"],
        })
    )
    visible_boundaries, addressee_plan = _validate_preference_result({
        "visible_boundaries": value["visible_boundaries"],
        "addressee_plan": value["addressee_plan"],
    })
    replacement = {
        "content_plan": content_plan,
        "content_requirements": content_requirements,
        "delivery_profile": delivery_profile,
        "lexical_avoidances": lexical_avoidances,
        "visible_boundaries": visible_boundaries,
        "addressee_plan": addressee_plan,
    }
    return replacement


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
        supporting_bids = reduced_payload.get("supporting_bids")
        if supporting_bids is not None:
            if (
                isinstance(supporting_bids, list)
                and len(supporting_bids) > MIN_SURFACE_SUPPORTING_BIDS
            ):
                reduced_payload["supporting_bids"] = supporting_bids[:-1]
                continue
            reduced_payload.pop("supporting_bids", None)
            continue
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
