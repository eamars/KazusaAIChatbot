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

CONTENT_PLAN_SYSTEM_PROMPT = '''# 任务
规划当前角色本轮实际会说出或发送的内容。保持上游已经形成的角色判断、事实边界、行动状态和称呼方向，同时给最终文字渲染器留下自然的角色表达空间。

# 输入
`surface` 是本轮完整语义包：`episode` 提供当前可见事件和角色方向；`intention`、`active_character_goal` 与 `response_plan` 提供已选择的回应目标和可答性；`expression_policy`、`semantic_affect`、`semantic_relationship`、`relational_willingness`、`interaction_style_context` 与 `character_expression_context` 提供表达姿态；`subjective_expression_context` 提供私密动机和权威断言边界；`permitted_action_results` 提供行动结果事实；`resolver_result` 提供任务或证据结果；`runtime_capability_limits` 提供当前能力限制；`addressee_plan` 提供称呼方向；`recent_character_dialog` 与 `overused_moves` 只提供措辞连续性。`contract_repair` 若出现，只说明上一候选的结构问题。

# 规划步骤
1. 以 `response_plan.response_goal` 和当前 `episode` 为可见语义锚点，确定本轮要回应、推进或收束的内容。角色动机、关系意愿和表达语境塑造姿态；它们在当前观察支持的范围内进入可见语义。
2. 按 `subjective_expression_context.epistemic_boundary` 规划断言强度。可断言内容直接表达，解释内容使用明确推测语气，未知内容保留未知；缺少证据保持开放。
3. 按 `goal_resolution` 与 `resolver_result` 规划答案：完整证据只使用 `evidence_excerpts`；部分证据表达已确认部分和 `remaining_needs`；等待、缺失、阻塞、不可用或失败状态表达相应缺口与下一步。已接纳并继续的任务表达接纳和等待后续结果。
4. 按 `permitted_action_results` 规划行动事实。`executed` 支持同一行动与效果的完成描写；`pending` 或 `scheduled` 支持对应等待或未来执行状态；`failed` 或 `unavailable` 支持限制与下一步。`speak` 只支持文字交付。没有匹配结果时，表达角色已选择的言语立场、愿望、提议或条件。
5. 保持 `episode` 与 `addressee_plan` 的行动者、对象、受益者和主语方向。使用 `delivery_profile` 的五个维度规划用词、句式、节奏、犹豫和标点。
6. 用正向的 `content_requirements` 记录必须保留的语义。`lexical_avoidances` 只列出本轮应避免逐字重复的具体短语；没有具体重复风险时返回空列表。
7. 若有 `contract_repair`，依据原 `surface` 重新生成完整结果，只修正合同问题。

# 输出
只返回一个 JSON 对象，并且字段恰好是 `content_plan`、`content_requirements`、`delivery_profile` 与 `lexical_avoidances`。`content_plan` 是 1..1000 字符的非空字符串；`content_requirements` 是 1..8 条互不重复的非空字符串，每条最多 500 字符；`delivery_profile` 恰好包含 `lexical_register`、`sentence_shape`、`rhythm`、`hesitation`、`punctuation` 五个非空字符串，每项最多 200 字符；`lexical_avoidances` 是 0..8 条互不重复的具体短语，每条最多 120 字符。不添加其他字段。

自由文本使用简体中文；用户引文、专有名词、代码、URL、schema 与 enum token 保持原样。'''


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


VISUAL_SYSTEM_PROMPT = '''# 任务
为本轮已选择的终端图像表面生成一段可直接交给图像生成器的 `visual_directives`。

# 输入
`surface` 是本轮完整语义包：`episode` 提供当前可见事件和参与者方向；`intention`、`active_character_goal` 与 `response_plan` 提供已选择的图像意图；`expression_policy`、`semantic_affect`、`semantic_relationship`、`relational_willingness` 与 `interaction_style_context` 提供情绪和关系姿态；`permitted_action_results` 与 `resolver_result` 提供事实状态；`runtime_capability_limits` 提供能力限制；`addressee_plan` 提供参与者方向；`recent_character_dialog` 与 `overused_moves` 只提供连续性；`visual_character_context` 提供角色可见外观。`contract_repair` 若出现，只说明上一候选的结构问题。

# 规划步骤
1. 以已选择的回应目标和当前可见事件确定图像主题。
2. 组合 `visual_character_context` 中的外观、合适的姿势与表情、构图、环境、光线和场景氛围。
3. 保持参与者方向和已确认事实；行动结果与能力状态按输入原义呈现。
4. 写成终端图像描述，不加入对话文本、内部运行说明或后续处理指令。
5. 若有 `contract_repair`，依据原 `surface` 重新生成完整结果，只修正合同问题。

# 输出
只返回一个 JSON 对象，并且唯一字段是 `visual_directives`。其值为 1..1000 字符的非空字符串。不添加其他字段。自由文本使用简体中文；用户引文、专有名词、代码、URL、schema 与 enum token 保持原样。'''


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
                reason="provider_error",
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
                reason="contract_error",
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
                    "reason": reason,
                    "contract_error": contract_error[:SURFACE_STAGE_ERROR_CAP],
                    "invalid_candidate": "",
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
    return value[:half_cap] + value[-half_cap:]


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
