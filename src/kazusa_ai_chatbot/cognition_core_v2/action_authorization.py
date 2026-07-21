"""Focused semantic authorization and deterministic route ownership."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionAffordanceV2,
    ActionBidV2,
    CognitionCoreServicesV2,
    CognitionEvidenceV2,
    CognitionExecutionError,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


ACTION_AUTHORIZATION_PROMPT_CAP = 16000
ACTION_AUTHORIZATION_ATTEMPT_LIMIT = 2
ACTION_AUTHORIZATION_OUTPUT_CAP = 3000
ACTION_AUTHORIZATION_TEXT_CAP = 400


logger = logging.getLogger(__name__)


ACTION_AUTHORIZATION_PROMPT = '''你负责核准角色大脑提出的可执行动作。规划阶段已经给出候选项；
对每个候选项，只判断它声明的真实效果是否得到所引用当前证据的支持与授权。

当前证据具有最高依据。已经接纳的目标描述和候选目标只提供语境，不能代替证据。若证据只是在
讨论、想象、角色扮演或请求某种效果，而所给能力无法真实完成该效果，则拒绝候选项。持久化或
跨轮工作需要当前证据明确请求或接受其持久效果；编码工作需要当前证据明确请求代码、代码库或
软件工程工作。所给能力不会隐含驱动角色身体或现实场景，延迟工作也不承担生成、保存或稍后展示
动作表演描述的职责。

当引用的当前证据确实支持能力声明的具体真实效果时，可以核准候选项。这包括被明确接受的延迟
工作、定时发言、记忆生命周期操作、来自合格私聊来源的后续认知，或绑定可信运行上下文的动作。
判断能力与证据是否匹配即可；文笔、角色意愿、最终措辞以及其他候选项是否更合适不属于本阶段。

# 输出格式
只返回一个 JSON 对象，且字段必须恰好是 decisions。decisions 是一个 JSON 对象，键必须恰好
覆盖提供的 candidate handle，值必须是布尔值；true 表示核准，false 表示拒绝。候选项不得遗漏
或增添，只输出 JSON。
'''


def derive_action_route(
    *,
    episode: Mapping[str, Any],
    primary_bid: ActionBidV2 | None,
    action_requests: Sequence[Mapping[str, Any]],
    resolver_requests: Sequence[Mapping[str, Any]],
) -> Literal["speech", "evidence", "action", "deferral", "silence"]:
    """Derive protocol route from output mode and validated request sets."""

    if action_requests and resolver_requests:
        raise CognitionExecutionError(
            "action and resolver requests are mutually exclusive"
        )
    output_mode = episode.get("output_mode")
    if output_mode == "silence":
        if action_requests or resolver_requests:
            raise CognitionExecutionError(
                "silence output mode cannot contain capability requests"
            )
        return "silence"
    if output_mode == "scheduled_action_request":
        if resolver_requests:
            raise CognitionExecutionError(
                "scheduled action output cannot request resolver work"
            )
        return "action" if action_requests else "silence"
    if output_mode == "visible_reply":
        if resolver_requests:
            return "evidence"
        return "speech" if primary_bid is not None else "silence"
    if output_mode in {"think_only", "preview"}:
        if resolver_requests:
            return "evidence"
        if action_requests:
            return "action"
        return "silence"
    raise CognitionExecutionError("episode output mode is unsupported")


async def authorize_action_requests(
    *,
    action_requests: Sequence[Mapping[str, str]],
    bid_handles: Mapping[str, ActionBidV2],
    evidence: Sequence[CognitionEvidenceV2],
    action_handles: Mapping[str, ActionAffordanceV2],
    services: CognitionCoreServicesV2,
) -> list[dict[str, str]]:
    """Retain planner proposals whose real effects are semantically grounded."""

    if not action_requests:
        return []

    evidence_by_handle = {
        row["evidence_handle"]: row
        for row in evidence
    }
    candidates: dict[str, dict[str, Any]] = {}
    candidate_requests: dict[str, dict[str, str]] = {}
    for index, request in enumerate(action_requests, start=1):
        bid_handle = request["bid_handle"]
        action_handle = request["action_handle"]
        if bid_handle not in bid_handles:
            raise CognitionExecutionError(
                "action authorization received an unknown bid handle"
            )
        if action_handle not in action_handles:
            raise CognitionExecutionError(
                "action authorization received an unknown action handle"
            )
        bid = bid_handles[bid_handle]
        affordance = action_handles[action_handle]
        cited_evidence = [
            evidence_by_handle[handle]["semantic_text"]
            for handle in bid["evidence_handles"]
            if handle in evidence_by_handle
        ]
        candidate_handle = f"c{index}"
        candidate_requests[candidate_handle] = dict(request)
        candidates[candidate_handle] = {
            "capability_kind": affordance["action_kind"],
            "semantic_capability": affordance["capability"],
            "proposed_decision": request["decision"],
            "proposed_semantic_goal": request["semantic_goal"],
            "proposed_reason": request["reason"],
            "admitted_bid": {
                "intention": bid["intention"],
                "desired_outcome": bid["desired_outcome"],
                "concrete_detail": bid["concrete_detail"],
                "reason": bid["reason"],
            },
            "current_evidence": cited_evidence,
        }
    prompt_payload = {"candidates": candidates}
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    if len(prompt_text) > ACTION_AUTHORIZATION_PROMPT_CAP:
        raise CognitionExecutionError(
            "action-authorization prompt exceeds contract cap"
        )
    messages: list[BaseMessage] = [
        SystemMessage(content=ACTION_AUTHORIZATION_PROMPT),
        HumanMessage(content=prompt_text),
    ]
    decisions = await invoke_semantic_authorizer(
        services=services,
        messages=messages,
        candidate_handles=list(candidate_requests),
        stage_name="action_authorization",
        output_state_fields=["authorized_action_requests"],
    )
    authorized_handles = {
        handle for handle, authorized in decisions.items() if authorized
    }
    return [
        candidate_requests[handle]
        for handle in candidate_requests
        if handle in authorized_handles
    ]


async def invoke_semantic_authorizer(
    *,
    services: CognitionCoreServicesV2,
    messages: list[BaseMessage],
    candidate_handles: list[str],
    stage_name: str,
    output_state_fields: list[str],
) -> dict[str, bool]:
    """Invoke one focused semantic authorizer with one shape repair."""

    current_messages = list(messages)
    for attempt_index in range(ACTION_AUTHORIZATION_ATTEMPT_LIMIT):
        started_at = perf_counter()
        response = await services.llm.ainvoke(
            current_messages,
            config=services.action_selection_config,
        )
        response_text = str(response.content)
        parsed: object = {}
        current_stage_name = (
            stage_name
            if attempt_index == 0
            else f"{stage_name}.repair"
        )
        try:
            parsed = parse_llm_json_output(response_text)
            decisions = _validate_authorization_decisions(
                parsed,
                candidate_handles=candidate_handles,
            )
        except ValueError as exc:
            await _record_authorization_trace(
                services=services,
                messages=current_messages,
                response_text=response_text,
                parsed_output=parsed,
                parse_status="contract_error",
                status="failed",
                started_at=started_at,
                stage_name=current_stage_name,
                output_state_fields=output_state_fields,
            )
            if attempt_index + 1 >= ACTION_AUTHORIZATION_ATTEMPT_LIMIT:
                logger.warning(
                    "%s denied all candidates after an unusable replacement: "
                    "%s",
                    stage_name,
                    exc,
                )
                return {
                    handle: False
                    for handle in candidate_handles
                }
            current_messages.append(_authorization_repair_message(
                response_text=response_text,
                contract_error=str(exc),
                candidate_handles=candidate_handles,
            ))
            continue
        await _record_authorization_trace(
            services=services,
            messages=current_messages,
            response_text=response_text,
            parsed_output={"decisions": decisions},
            parse_status="succeeded",
            status="succeeded",
            started_at=started_at,
            stage_name=current_stage_name,
            output_state_fields=output_state_fields,
        )
        return decisions
    raise AssertionError("action-authorization attempt loop did not terminate")


def _validate_authorization_decisions(
    parsed: object,
    *,
    candidate_handles: list[str],
) -> dict[str, bool]:
    """Validate exact coverage and fixed semantic authorization shape."""

    if not isinstance(parsed, Mapping) or set(parsed) != {"decisions"}:
        raise ValueError("action authorization fields are not exact")
    decisions = parsed["decisions"]
    if not isinstance(decisions, Mapping):
        raise ValueError("action authorization decisions must be an object")
    if set(decisions) != set(candidate_handles):
        raise ValueError(
            "action authorization must cover every supplied candidate"
        )
    normalized: dict[str, bool] = {}
    for handle in candidate_handles:
        authorized = decisions[handle]
        if not isinstance(authorized, bool):
            raise ValueError("action authorization decision must be boolean")
        normalized[handle] = authorized
    return normalized


def _authorization_repair_message(
    *,
    response_text: str,
    contract_error: str,
    candidate_handles: list[str],
) -> HumanMessage:
    """Build one bounded exact-shape authorization replacement request."""

    bounded_response = response_text
    if len(bounded_response) > ACTION_AUTHORIZATION_OUTPUT_CAP:
        half_cap = ACTION_AUTHORIZATION_OUTPUT_CAP // 2
        bounded_response = (
            bounded_response[:half_cap]
            + "\n... 已截断的不合格输出 ...\n"
            + bounded_response[-half_cap:]
        )
    payload = {
        "repair_instruction": (
            "返回一个完整的替代核准对象。保留原有语义判断，为每个提供的 "
            "candidate handle 填写一个布尔值，并且只输出 JSON。"
        ),
        "candidate_handles_in_order": candidate_handles,
        "contract_error": contract_error[:ACTION_AUTHORIZATION_TEXT_CAP],
        "invalid_response": bounded_response,
    }
    return HumanMessage(
        content=json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


async def _record_authorization_trace(
    *,
    services: CognitionCoreServicesV2,
    messages: Sequence[BaseMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    started_at: float,
    stage_name: str,
    output_state_fields: list[str],
) -> None:
    """Preserve one protected semantic authorization model boundary."""

    trace_id = llm_tracing.current_trace_id()
    if not trace_id:
        return
    config = services.action_selection_config
    await llm_tracing.record_llm_trace_step(
        trace_id=trace_id,
        stage_name=stage_name,
        route_name=config.route_name,
        model_name=config.model,
        messages=messages,
        response_text=response_text,
        parsed_output=parsed_output,
        parse_status=parse_status,
        status=status,
        duration_ms=max(0, int((perf_counter() - started_at) * 1000)),
        output_state_fields=output_state_fields,
    )
