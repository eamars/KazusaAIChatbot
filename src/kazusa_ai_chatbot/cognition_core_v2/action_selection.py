"""Compositional semantic action planning over admitted cognition bids."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    authorize_action_requests,
)
from kazusa_ai_chatbot.cognition_core_v2.resolver_authorization import (
    authorize_resolver_requests,
)
from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionAffordanceV2,
    ActionBidV2,
    CognitionCoreServicesV2,
    CognitionEvidenceV2,
    CognitionExecutionError,
    ResolverAffordanceV2,
    ResolverCapabilityRequestV2,
    SelectedIntentionV2,
    SemanticActionRequestV2,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ALLOWED_PENDING_DECISIONS,
    RESOLVER_GOAL_PROGRESS_VERSION,
    ResolverValidationError,
    validate_resolver_goal_progress,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

ACTION_REQUEST_CAP = 3
ACTION_PLANNING_PROMPT_CAP = 24000
ACTION_PLANNING_REPAIR_OUTPUT_CAP = 4000
ACTION_PLANNING_ATTEMPT_LIMIT = 2
MODEL_TEXT_CAP = 500


logger = logging.getLogger(__name__)


ACTION_PLANNING_PROMPT = '''你负责为当前角色提出语义能力请求。根据已经接纳的目标，提出能够推进
目标的具体可执行 action request 或 resolver request。primary bid 决定可见意图；supporting bid
可以提供相容的私有动作或证据需求。本阶段不选择或复述 route，不改写目标候选，不生成最终对话，
不执行或核准能力，也不虚构未提供的能力。协议代码会在语义授权完成后派生 route。

输出一个语义提案对象。action_requests 与 resolver_requests 互斥，各自最多包含三项。即时可见
发言不是能力请求，不放入本输出。需要补充证据、持久化澄清或批准步骤时，只选择 resolver；后续
可见答案或问题由 resolver 的再次认知负责。

只有当引用的已接纳目标确实需要某项能力的持久化或跨轮效果来实现 desired_outcome 时，才提出
私有动作。目标候选不能扩大能力适用范围；当前证据本身必须支持所选能力声明的真实效果。task、
action、request、analysis 或 work 等泛化词不能证明能力匹配。编码能力要求当前证据明确请求实际
代码、代码库或软件工程工作；偏移的目标候选不能把身体互动或普通聊天变成编码任务。

规划者本轮的推理、记忆回想、回复准备、措辞排练或本轮即可完成的思考不构成 action request。
真正缺少证据时使用 resolver；已接纳回应现在即可完成时，直接发言而不附加私有动作。所给 action
能力不会驱动角色身体或现实场景，也不负责执行身体请求或生成、保存、稍后展示身体动作表演描述。
面对身体互动请求，通常由发言表达当前角色立场；只有另一个明确提供的能力确实具有不同且清晰的
非身体效果时，才选择该能力。

每项请求必须引用一个提供的 bid handle 和一个提供的 capability handle。action request 按
affordance.decision_mode 填写 decision：
- optional：使用 default_decision 或空字符串；
- required_text：给出一个具体、有界的语义决定；
- closed：复制 allowed_decisions 中的一个值。
decision_pattern 非空时，decision 必须完整匹配，不添加前后缀、解释或最终消息文本。
semantic_goal 描述具体语义目标，不写执行参数或最终措辞；reason 解释该请求如何推进所引用目标。
不输出 context_ref；选择 action_handle 后，确定性的 context_ref 会在验证后绑定。

角色自己的反思或内部观察属于证据，不是当前用户的即时发言。生成的文字不复述来源包标题、
时间戳、schema key、传输摘要或运行元数据。新生成的自由文本使用简体中文；用户引文、专有名词、
代码、URL、capability name 以及 schema 或 enum token 保持原样。

只有 resolver 上下文存在活跃 pending item 且当前证据支持决定时，
resolver_pending_resolution 才不是 null；此时恰好返回 decision 和 reason，活跃项由确定性代码绑定。
不需要目标进度时 resolver_goal_progress 为 null；需要时只返回发生变化的局部语义更新。协议代码
绑定 schema_version 和 original_goal，确定性代码从 current_resolver_goal_progress 保留省略的
已知 checklist 字段。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是：
- action_requests：零到三个对象，每个对象必须恰好包含 bid_handle、action_handle、decision、
  semantic_goal 和 reason；
- resolver_requests：零到三个对象，每个对象必须恰好包含 bid_handle、resolver_handle、
  semantic_goal 和 reason；
- resolver_pending_resolution：null，或恰好包含 decision 和 reason 的对象；
- resolver_goal_progress：null，或一个局部语义更新对象。
resolver_requests 非空时 action_requests 必须为空；action_requests 非空时 resolver_requests 必须
为空。不输出其他字段。
'''


async def plan_actions(
    *,
    primary_bid: ActionBidV2 | None,
    supporting_bids: Sequence[ActionBidV2],
    episode: Mapping[str, Any],
    evidence: Sequence[CognitionEvidenceV2],
    available_actions: Sequence[ActionAffordanceV2],
    available_resolvers: Sequence[ResolverAffordanceV2],
    resolver_context: str,
    services: CognitionCoreServicesV2,
    current_goal_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one route and bounded semantic requests from admitted motives.

    Args:
        primary_bid: Workspace-selected motive that owns the visible intention.
        supporting_bids: Other admitted motives that may contribute requests.
        episode: Canonical source envelope used only through semantic fields.
        evidence: Prompt-safe evidence available to admitted motives.
        available_actions: Registry-derived executable action affordances.
        available_resolvers: Registry-derived resolver affordances.
        resolver_context: Bounded prompt-safe resolver recurrence projection.
        services: Injected LLM binding and action-planning configuration.

    Returns:
        Selected intention, semantic requests, and resolver lifecycle decisions.
    """

    if primary_bid is None:
        return_value = _silence_result()
        return return_value

    bids = [primary_bid, *supporting_bids]
    bid_handles = {
        f"b{index}": bid for index, bid in enumerate(bids, start=1)
    }
    planner_actions = [
        affordance
        for affordance in available_actions
        if affordance["permission"] == "allowed"
        and affordance["action_kind"] not in {
            SPEAK_CAPABILITY,
            APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        }
    ]
    action_handles = {
        f"a{index}": affordance
        for index, affordance in enumerate(planner_actions, start=1)
    }
    resolver_handles = {
        f"r{index}": affordance
        for index, affordance in enumerate(
            (
                affordance
                for affordance in available_resolvers
                if affordance["availability"] == "available"
            ),
            start=1,
        )
    }
    prompt_payload = {
        "bids": {
            handle: {
                "intention": bid["intention"],
                "desired_outcome": bid["desired_outcome"],
                "concrete_detail": bid["concrete_detail"],
                "reason": bid["reason"],
                "expected_consequences": list(bid["expected_consequences"]),
                "confidence": bid["confidence"],
                "evidence_handles": list(bid["evidence_handles"]),
            }
            for handle, bid in bid_handles.items()
        },
        "episode": {
            "trigger_source": episode.get("trigger_source", ""),
        },
        "evidence": [
            {
                "handle": row["evidence_handle"],
                "source_kind": row["evidence_ref"]["source_kind"],
                "semantic_text": row["semantic_text"],
            }
            for row in evidence
        ],
        "action_handles": {
            handle: {
                "action_kind": affordance["action_kind"],
                "semantic_capability": affordance["capability"],
                "decision_mode": affordance["decision_mode"],
                "allowed_decisions": list(affordance["allowed_decisions"]),
                "default_decision": affordance["default_decision"],
                "decision_pattern": affordance["decision_pattern"],
            }
            for handle, affordance in action_handles.items()
        },
        "resolver_handles": {
            handle: {
                "capability": affordance["capability"],
                "semantic_capability": affordance["semantic_capability"],
            }
            for handle, affordance in resolver_handles.items()
        },
        "resolver_context": resolver_context,
        "current_resolver_goal_progress": current_goal_progress,
    }
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    if len(prompt_text) > ACTION_PLANNING_PROMPT_CAP:
        raise CognitionExecutionError("action-planning prompt exceeds contract cap")

    messages: list[BaseMessage] = [
        SystemMessage(content=ACTION_PLANNING_PROMPT),
        HumanMessage(content=prompt_text),
    ]
    decision = await _invoke_action_planner(
        services=services,
        messages=messages,
        bid_handles=bid_handles,
        action_handles=action_handles,
        resolver_handles=resolver_handles,
        current_goal_progress=current_goal_progress,
    )
    authorized_action_rows = await authorize_action_requests(
        action_requests=decision["action_requests"],
        bid_handles=bid_handles,
        evidence=evidence,
        action_handles=action_handles,
        services=services,
    )
    action_requests = _materialize_action_requests(
        authorized_action_rows,
        bid_handles,
        action_handles,
    )
    authorized_resolver_rows = await authorize_resolver_requests(
        resolver_requests=decision["resolver_requests"],
        bid_handles=bid_handles,
        evidence=evidence,
        resolver_handles=resolver_handles,
        resolver_context=resolver_context,
        services=services,
    )
    resolver_requests = _materialize_resolver_requests(
        authorized_resolver_rows,
        bid_handles,
        resolver_handles,
    )
    route = _derive_canonical_action_route(
        primary_bid=primary_bid,
        action_requests=action_requests,
        resolver_requests=resolver_requests,
    )
    intention: SelectedIntentionV2 = {
        "selected_branch_id": primary_bid["branch_id"],
        "route": route,
        "intention": primary_bid["intention"],
        "target_roles": list(primary_bid["target_roles"]),
        "reason": primary_bid["reason"],
    }
    return_value = {
        "intention": intention,
        "action_requests": action_requests,
        "resolver_requests": resolver_requests,
        "resolver_pending_resolution": decision[
            "resolver_pending_resolution"
        ],
        "resolver_goal_progress": decision["resolver_goal_progress"],
    }
    return return_value


def _derive_canonical_action_route(
    *,
    primary_bid: ActionBidV2 | None,
    action_requests: Sequence[Mapping[str, Any]],
    resolver_requests: Sequence[Mapping[str, Any]],
) -> str:
    """Derive route from canonical request sets and the admitted bid."""

    if action_requests and resolver_requests:
        raise CognitionExecutionError(
            "action and resolver requests are mutually exclusive"
        )
    if resolver_requests:
        return "evidence"
    if action_requests:
        return "action"
    return "speech" if primary_bid is not None else "silence"


async def _invoke_action_planner(
    *,
    services: CognitionCoreServicesV2,
    messages: list[BaseMessage],
    bid_handles: Mapping[str, ActionBidV2],
    action_handles: Mapping[str, ActionAffordanceV2],
    resolver_handles: Mapping[str, ResolverAffordanceV2],
    current_goal_progress: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Invoke the semantic planner with one bounded contract replacement."""

    current_messages = list(messages)
    for attempt_index in range(ACTION_PLANNING_ATTEMPT_LIMIT):
        started_at = perf_counter()
        response = await services.llm.ainvoke(
            current_messages,
            config=services.action_selection_config,
        )
        response_text = str(response.content)
        parsed: object = {}
        stage_name = (
            "action_planning"
            if attempt_index == 0
            else "action_planning.repair"
        )
        try:
            parsed = parse_llm_json_output(response_text)
            decision = _validate_action_plan_decision(
                parsed,
                bid_handles=bid_handles,
                action_handles=action_handles,
                resolver_handles=resolver_handles,
                current_goal_progress=current_goal_progress,
            )
        except (ResolverValidationError, ValueError) as exc:
            await _record_action_planning_trace(
                services=services,
                messages=current_messages,
                response_text=response_text,
                parsed_output=parsed,
                parse_status="contract_error",
                status="failed",
                started_at=started_at,
                stage_name=stage_name,
            )
            if attempt_index + 1 >= ACTION_PLANNING_ATTEMPT_LIMIT:
                logger.warning(
                    "Action planning dropped an unusable replacement: %s",
                    exc,
                )
                return _empty_action_plan_decision()
            current_messages.append(
                _action_planning_repair_message(
                    response_text=response_text,
                    contract_error=str(exc),
                )
            )
            continue

        await _record_action_planning_trace(
            services=services,
            messages=current_messages,
            response_text=response_text,
            parsed_output=decision,
            parse_status="succeeded",
            status="succeeded",
            started_at=started_at,
            stage_name=stage_name,
        )
        return decision

    raise AssertionError("action-planning attempt loop did not terminate")


def _action_planning_repair_message(
    *,
    response_text: str,
    contract_error: str,
) -> HumanMessage:
    """Build one bounded same-owner replacement request."""

    bounded_response = _bounded_repair_output(response_text)
    repair_payload = {
        "repair_instruction": (
            "返回一个完整对象替代原 action plan。只保留有依据的语义选择，"
            "满足所有精确字段和请求规则，并且只输出 JSON。"
        ),
        "contract_error": contract_error[:MODEL_TEXT_CAP],
        "invalid_response": bounded_response,
    }
    return HumanMessage(
        content=json.dumps(repair_payload, ensure_ascii=False, sort_keys=True)
    )


def _bounded_repair_output(response_text: str) -> str:
    """Keep both ends of one rejected output within the retry prompt cap."""

    if len(response_text) <= ACTION_PLANNING_REPAIR_OUTPUT_CAP:
        return response_text
    half_cap = ACTION_PLANNING_REPAIR_OUTPUT_CAP // 2
    return_value = (
        response_text[:half_cap]
        + "\n... 已截断的不合格输出 ...\n"
        + response_text[-half_cap:]
    )
    return return_value


def _validate_action_plan_decision(
    parsed: object,
    *,
    bid_handles: Mapping[str, ActionBidV2],
    action_handles: Mapping[str, ActionAffordanceV2],
    resolver_handles: Mapping[str, ResolverAffordanceV2],
    current_goal_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize semantic choices into the canonical planner contract."""

    if not isinstance(parsed, Mapping):
        raise ValueError("action plan must be an object")
    action_requests = parsed.get("action_requests", [])
    resolver_requests = parsed.get("resolver_requests", [])
    if not isinstance(action_requests, list):
        raise ValueError("action requests must be an array")
    if not isinstance(resolver_requests, list):
        raise ValueError("resolver requests must be an array")
    if action_requests and resolver_requests:
        raise ValueError("action and resolver requests are mutually exclusive")

    normalized_actions = _normalize_action_request_rows(
        action_requests,
        bid_handles,
        action_handles,
    )
    normalized_resolvers = _normalize_resolver_request_rows(
        resolver_requests,
        bid_handles,
        resolver_handles,
    )
    pending_resolution = _validate_pending_resolution_choice(
        parsed.get("resolver_pending_resolution")
    )
    goal_progress = _validate_goal_progress_choice(
        parsed.get("resolver_goal_progress"),
        current_goal_progress=current_goal_progress,
    )
    return_value = {
        "action_requests": normalized_actions,
        "resolver_requests": normalized_resolvers,
        "resolver_pending_resolution": pending_resolution,
        "resolver_goal_progress": goal_progress,
    }
    return return_value


def _normalize_action_request_rows(
    values: Sequence[object],
    bids: Mapping[str, ActionBidV2],
    actions: Mapping[str, ActionAffordanceV2],
) -> list[dict[str, str]]:
    """Keep bounded canonical action proposals with valid trusted handles."""

    normalized: list[dict[str, str]] = []
    for value in values:
        try:
            row = _validate_action_request_row(value, bids, actions)
        except ValueError as exc:
            logger.warning("Action planning dropped an invalid action row: %s", exc)
            continue
        normalized.append(row)
        if len(normalized) >= ACTION_REQUEST_CAP:
            break
    return normalized


def _normalize_resolver_request_rows(
    values: Sequence[object],
    bids: Mapping[str, ActionBidV2],
    resolvers: Mapping[str, ResolverAffordanceV2],
) -> list[dict[str, str]]:
    """Keep bounded canonical resolver proposals with valid trusted handles."""

    normalized: list[dict[str, str]] = []
    for value in values:
        try:
            row = _validate_resolver_request_row(value, bids, resolvers)
        except ValueError as exc:
            logger.warning(
                "Action planning dropped an invalid resolver row: %s",
                exc,
            )
            continue
        normalized.append(row)
        if len(normalized) >= ACTION_REQUEST_CAP:
            break
    return normalized


def _empty_action_plan_decision() -> dict[str, Any]:
    """Return the canonical fail-contained semantic proposal."""

    return {
        "action_requests": [],
        "resolver_requests": [],
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
    }


def _validate_action_request_row(
    value: object,
    bids: Mapping[str, ActionBidV2],
    actions: Mapping[str, ActionAffordanceV2],
) -> dict[str, str]:
    """Validate one action row and its registry-derived decision semantics."""

    required = {
        "bid_handle",
        "action_handle",
        "decision",
        "semantic_goal",
        "reason",
    }
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError("action request fields are incomplete")
    bid_handle = value["bid_handle"]
    action_handle = value["action_handle"]
    if bid_handle not in bids:
        raise ValueError("action request bid handle is unavailable")
    if action_handle not in actions:
        raise ValueError("action request action handle is unavailable")
    decision = _bounded_model_text(
        value["decision"],
        "action request decision",
        maximum=200,
        allow_empty=True,
    )
    semantic_goal = _bounded_model_text(
        value["semantic_goal"],
        "action request semantic_goal",
    )
    reason = _bounded_model_text(value["reason"], "action request reason")
    affordance = actions[action_handle]
    mode = affordance["decision_mode"]
    if mode == "required_text" and not decision:
        raise ValueError(
            f"action request {action_handle} requires a concrete decision"
        )
    if mode == "closed" and decision not in affordance["allowed_decisions"]:
        allowed_decisions = affordance["allowed_decisions"]
        raise ValueError(
            f"action request {action_handle} decision must be one of "
            f"{allowed_decisions!r}"
        )
    if mode == "optional" and decision not in {
        "",
        affordance["default_decision"],
    }:
        default_decision = affordance["default_decision"]
        raise ValueError(
            f"action request {action_handle} decision must be empty or "
            f"{default_decision!r}"
        )
    decision_pattern = affordance["decision_pattern"]
    if decision_pattern and re.fullmatch(decision_pattern, decision) is None:
        raise ValueError(
            f"action request {action_handle} decision must full-match "
            f"{decision_pattern!r}"
        )
    return_value = {
        "bid_handle": bid_handle,
        "action_handle": action_handle,
        "decision": decision,
        "semantic_goal": semantic_goal,
        "reason": reason,
    }
    return return_value


def _validate_resolver_request_row(
    value: object,
    bids: Mapping[str, ActionBidV2],
    resolvers: Mapping[str, ResolverAffordanceV2],
) -> dict[str, str]:
    """Validate one resolver row and its admitted-bid provenance."""

    required = {
        "bid_handle",
        "resolver_handle",
        "semantic_goal",
        "reason",
    }
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError("resolver request fields are incomplete")
    bid_handle = value["bid_handle"]
    resolver_handle = value["resolver_handle"]
    if bid_handle not in bids:
        raise ValueError("resolver request bid handle is unavailable")
    if resolver_handle not in resolvers:
        raise ValueError("resolver request resolver handle is unavailable")
    semantic_goal = _bounded_model_text(
        value["semantic_goal"],
        "resolver request semantic_goal",
    )
    reason = _bounded_model_text(value["reason"], "resolver request reason")
    return_value = {
        "bid_handle": bid_handle,
        "resolver_handle": resolver_handle,
        "semantic_goal": semantic_goal,
        "reason": reason,
    }
    return return_value


def _validate_pending_resolution_choice(value: object) -> dict | None:
    """Validate the model-owned semantic choice before active-row binding."""

    if value is None:
        return_value = None
        return return_value
    if not isinstance(value, Mapping) or set(value) != {"decision", "reason"}:
        raise ValueError("pending resolution fields are not exact")
    decision = value["decision"]
    if decision not in ALLOWED_PENDING_DECISIONS:
        raise ValueError("pending resolution decision is invalid")
    reason = _bounded_model_text(value["reason"], "pending resolution reason")
    return_value = {"decision": decision, "reason": reason}
    return return_value


def _validate_goal_progress_choice(
    value: object,
    *,
    current_goal_progress: Mapping[str, Any] | None,
) -> dict | None:
    """Merge one semantic delta into protocol-owned resolver progress."""

    if value is None:
        return_value = None
        return return_value
    if not isinstance(value, Mapping):
        raise ValueError("resolver goal progress must be an object or null")
    if current_goal_progress is None:
        raw_progress = dict(value)
        raw_progress.setdefault(
            "schema_version",
            RESOLVER_GOAL_PROGRESS_VERSION,
        )
        validated = validate_resolver_goal_progress(raw_progress)
        return_value = dict(validated)
        return return_value

    current = dict(validate_resolver_goal_progress(current_goal_progress))
    allowed_fields = set(current)
    if not set(value).issubset(allowed_fields):
        raise ValueError("resolver goal progress update fields are invalid")
    supplied_version = value.get("schema_version")
    if supplied_version not in {None, RESOLVER_GOAL_PROGRESS_VERSION}:
        raise ValueError("resolver goal progress schema_version is invalid")
    supplied_goal = value.get("original_goal")
    if supplied_goal not in {None, current["original_goal"]}:
        raise ValueError("resolver goal progress cannot replace original_goal")
    raw_progress = dict(current)
    raw_progress.update({
        key: item
        for key, item in value.items()
        if key not in {"schema_version", "original_goal"}
    })
    validated = validate_resolver_goal_progress(raw_progress)
    return_value = dict(validated)
    return return_value


def _materialize_action_requests(
    requests: Sequence[Mapping[str, str]],
    bids: Mapping[str, ActionBidV2],
    actions: Mapping[str, ActionAffordanceV2],
) -> list[SemanticActionRequestV2]:
    """Copy admitted provenance into planner-selected action requests."""

    result: list[SemanticActionRequestV2] = []
    for request in requests:
        bid = bids[request["bid_handle"]]
        affordance = actions[request["action_handle"]]
        result.append({
            "action_kind": affordance["action_kind"],
            "decision": request["decision"],
            "context_ref": affordance["context_ref"],
            "semantic_goal": request["semantic_goal"],
            "reason": request["reason"],
            "target_roles": list(bid["target_roles"]),
            "evidence_handles": list(bid["evidence_handles"]),
        })
    return result


def _materialize_resolver_requests(
    requests: Sequence[Mapping[str, str]],
    bids: Mapping[str, ActionBidV2],
    resolvers: Mapping[str, ResolverAffordanceV2],
) -> list[ResolverCapabilityRequestV2]:
    """Copy admitted evidence provenance into resolver requests."""

    result: list[ResolverCapabilityRequestV2] = []
    for request in requests:
        bid = bids[request["bid_handle"]]
        affordance = resolvers[request["resolver_handle"]]
        result.append({
            "capability": affordance["capability"],
            "semantic_goal": request["semantic_goal"],
            "reason": request["reason"],
            "evidence_handles": list(bid["evidence_handles"]),
        })
    return result


async def _record_action_planning_trace(
    *,
    services: CognitionCoreServicesV2,
    messages: Sequence[BaseMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    started_at: float,
    stage_name: str,
) -> None:
    """Preserve the protected action-planning model boundary."""

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
        output_state_fields=[
            "intention",
            "action_requests",
            "resolver_requests",
            "resolver_pending_resolution",
            "resolver_goal_progress",
        ],
    )


def _bounded_model_text(
    value: object,
    label: str,
    *,
    maximum: int = MODEL_TEXT_CAP,
    allow_empty: bool = False,
) -> str:
    """Validate one bounded model-authored semantic string."""

    if not isinstance(value, str):
        raise ValueError(f"{label} is invalid")
    normalized = value.strip()
    if (not allow_empty and not normalized) or len(normalized) > maximum:
        raise ValueError(f"{label} is invalid")
    return normalized


def _silence_result() -> dict[str, Any]:
    """Return deterministic silence when workspace admits no motive."""

    return_value = {
        "intention": {
            "route": "silence",
            "intention": "remain silent",
            "target_roles": [],
            "reason": "no valid admitted bid",
        },
        "action_requests": [],
        "resolver_requests": [],
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
    }
    return return_value
