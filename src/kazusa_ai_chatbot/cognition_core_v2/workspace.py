"""Provenance-safe workspace collapse for complete V2 action bids."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionBidV2,
    CollapsedIntentionV2,
    CognitionCoreServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    branch_order_key,
)
from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    V2_MODEL_TOTAL_ATTEMPTS,
)
from kazusa_ai_chatbot.cognition_core_v2.prompt_budget import PromptBudgetError
from kazusa_ai_chatbot.llm_tracing import failure_capsule
from kazusa_ai_chatbot.utils import parse_llm_json_output


WORKSPACE_COLLAPSE_ATTEMPT_LIMIT = V2_MODEL_TOTAL_ATTEMPTS
WORKSPACE_COLLAPSE_PROMPT_CAP = 24000
WORKSPACE_COLLAPSE_REPAIR_OUTPUT_CAP = 4000
WORKSPACE_CONTEXT_TEXT_FLOOR = 256


COLLAPSE_PROMPT = '''把完整的目标候选划分为本轮主目标、支持目标和抑制目标。先依据 current_event
判断每个候选及其 persistent_goal 是否与本轮直接相关，再比较候选质量。ordinary_response 是当前
回应的基线候选。非 ordinary_response 的持久目标只有在当前事件直接推进、阻碍、威胁或要求处理
同一具体事项时才可成为主目标或支持目标。仅有目标仍在进行中、同一用户、一般关系互动、关系
appraisal 存在、角色重视某项驱动，或分支 action tendency，不能证明当前相关。若
persistent_goal 与 current_event 是不同事项，必须抑制该候选，即使候选文本把当前请求改写成该
旧目标的边界问题。本阶段不判断工具、resolver、worker 或运行时能力，也不改写候选。

只返回包含 primary_bid_handle、supporting_bid_handles 和 suppressed_bid_handles 的 JSON。
保持候选内容原样，不复制内容，也不增添细节。每个提供的 bid handle 必须在三个分区中恰好出现
一次。
'''


async def collapse_bids(
    bids: Sequence[ActionBidV2],
    services: CognitionCoreServicesV2,
    *,
    current_event: Sequence[Mapping[str, object]],
    goal_context_by_ref: Mapping[str, Mapping[str, object]],
) -> CollapsedIntentionV2:
    """Partition complete bids using current-event and goal provenance.

    Args:
        bids: Complete branch-owned candidates eligible for partition.
        services: Stage-local LLM and route configuration.
        current_event: Typed current episode evidence projected for relevance.
        goal_context_by_ref: Bounded persistent goals keyed by bid goal id.

    Returns:
        The model-authored partition with complete bids copied by handle.
    """

    ordered = sorted(bids, key=lambda bid: branch_order_key(bid["branch_id"]))
    if not ordered:
        raise ValueError("workspace collapse requires at least one bid")
    if len(ordered) == 1:
        return {
            "primary_branch_id": ordered[0]["branch_id"],
            "supporting_branch_ids": [],
            "suppressed_branch_ids": [],
            "primary_bid": ordered[0],
            "supporting_bids": [],
            "competing_bids": [],
        }
    handles = {f"b{index}": bid for index, bid in enumerate(ordered, start=1)}
    prompt_payload = {
        "current_event": [dict(row) for row in current_event],
        "bids": {
            handle: {
                "branch_id": bid["branch_id"],
                "persistent_goal": (
                    None
                    if bid["branch_id"] == "ordinary_response"
                    else dict(goal_context_by_ref[
                        bid["goal_ref"]["entity_id"]
                    ])
                ),
                "intention": bid["intention"],
                "desired_outcome": bid["desired_outcome"],
                "reason": bid["reason"],
                "confidence": bid["confidence"],
            }
            for handle, bid in handles.items()
        }
    }
    system_message = SystemMessage(content=COLLAPSE_PROMPT)
    payload_cap = WORKSPACE_COLLAPSE_PROMPT_CAP - len(COLLAPSE_PROMPT)
    available_attempts = WORKSPACE_COLLAPSE_ATTEMPT_LIMIT
    try:
        prompt_text = _fit_workspace_prompt_payload(
            prompt_payload,
            maximum_chars=payload_cap,
        )
    except PromptBudgetError:
        prompt_text = ""
        available_attempts = 0
    request_messages = [
        system_message,
        HumanMessage(content=prompt_text),
    ]
    partition: dict[str, Any] | None = None
    for attempt_index in range(available_attempts):
        started_at = perf_counter()
        stage_name = (
            "workspace_collapse"
            if attempt_index == 0
            else "workspace_collapse.repair"
        )
        try:
            response = await services.llm.ainvoke(
                request_messages,
                config=services.workspace_collapse_config,
            )
        except (
            OpenAIError,
            httpx.HTTPError,
            ConnectionError,
            OSError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            _record_workspace_trace(
                services=services,
                messages=request_messages,
                response_text="",
                parsed_output={},
                parse_status="provider_error",
                status="failed",
                started_at=started_at,
                stage_name=stage_name,
                attempt_index=attempt_index + 1,
                validation_error=str(exc),
            )
            if attempt_index + 1 >= WORKSPACE_COLLAPSE_ATTEMPT_LIMIT:
                break
            request_messages = [
                system_message,
                HumanMessage(content=prompt_text),
            ]
            continue
        response_text = str(getattr(response, "content", ""))
        parsed: object = {}
        try:
            parsed = parse_llm_json_output(
                response_text,
                repair_trace_hook=(
                    failure_capsule.append_json_repair_attempt
                ),
            )
            partition = _validate_partition(parsed, set(handles))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            _record_workspace_trace(
                services=services,
                messages=request_messages,
                response_text=response_text,
                parsed_output=parsed,
                parse_status="contract_error",
                status="failed",
                started_at=started_at,
                stage_name=stage_name,
                attempt_index=attempt_index + 1,
                validation_error=str(exc),
            )
            if attempt_index + 1 >= WORKSPACE_COLLAPSE_ATTEMPT_LIMIT:
                break
            invalid_candidate = response_text
            if len(invalid_candidate) > WORKSPACE_COLLAPSE_REPAIR_OUTPUT_CAP:
                half_cap = WORKSPACE_COLLAPSE_REPAIR_OUTPUT_CAP // 2
                invalid_candidate = (
                    invalid_candidate[:half_cap]
                    + "\n... 已截断的不合格候选 ...\n"
                    + invalid_candidate[-half_cap:]
                )
            repair_payload = {
                **prompt_payload,
                "contract_repair": {
                    "reason": str(exc)[:500],
                    "invalid_candidate": invalid_candidate,
                },
            }
            try:
                repair_text = _fit_workspace_prompt_payload(
                    repair_payload,
                    maximum_chars=payload_cap,
                )
            except PromptBudgetError:
                break
            request_messages = [
                system_message,
                HumanMessage(content=repair_text),
            ]
            continue
        _record_workspace_trace(
            services=services,
            messages=request_messages,
            response_text=response_text,
            parsed_output=parsed,
            parse_status="succeeded",
            status="succeeded",
            started_at=started_at,
            stage_name=stage_name,
            attempt_index=attempt_index + 1,
            validation_error="",
        )
        break

    if partition is None:
        primary = ordered[0]
        suppressed = ordered[1:]
        fallback: CollapsedIntentionV2 = {
            "primary_branch_id": primary["branch_id"],
            "supporting_branch_ids": [],
            "suppressed_branch_ids": [
                bid["branch_id"] for bid in suppressed
            ],
            "primary_bid": primary,
            "supporting_bids": [],
            "competing_bids": list(suppressed),
        }
        return fallback

    primary_handle = partition["primary_bid_handle"]
    primary = handles[primary_handle]
    declared_supporting = [
        handles[handle] for handle in partition["supporting_bid_handles"]
    ]
    suppressed = [
        handles[handle] for handle in partition["suppressed_bid_handles"]
    ]
    result: CollapsedIntentionV2 = {
        "primary_branch_id": primary["branch_id"],
        "supporting_branch_ids": [
            bid["branch_id"] for bid in declared_supporting
        ],
        "suppressed_branch_ids": [
            bid["branch_id"] for bid in suppressed
        ],
        "primary_bid": primary,
        "supporting_bids": declared_supporting,
        "competing_bids": suppressed,
    }
    return result


def _fit_workspace_prompt_payload(
    prompt_payload: dict[str, Any],
    *,
    maximum_chars: int,
) -> str:
    """Fit relevance text while retaining every bid and provenance handle."""

    if maximum_chars <= 0:
        raise PromptBudgetError(
            "workspace system prompt exhausts the aggregate character cap"
        )
    serialized = json.dumps(
        prompt_payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(serialized) <= maximum_chars:
        return serialized

    text_slots: list[tuple[dict[str, Any], str, str]] = []
    for event in prompt_payload["current_event"]:
        semantic_text = event["semantic_text"]
        if not isinstance(semantic_text, str):
            raise TypeError("workspace current-event text must be a string")
        text_slots.append((event, "semantic_text", semantic_text))
    for bid_handle in sorted(prompt_payload["bids"]):
        persistent_goal = prompt_payload["bids"][bid_handle][
            "persistent_goal"
        ]
        if persistent_goal is None:
            continue
        description = persistent_goal["description"]
        if not isinstance(description, str):
            raise TypeError("workspace persistent-goal description must be a string")
        text_slots.append((persistent_goal, "description", description))

    for owner, field_name, original in text_slots:
        owner[field_name] = _middle_truncate_text(
            original,
            WORKSPACE_CONTEXT_TEXT_FLOOR,
        )
    floor_serialized = json.dumps(
        prompt_payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(floor_serialized) > maximum_chars:
        raise PromptBudgetError(
            "workspace required structure exceeds the aggregate character cap"
        )

    for owner, field_name, original in text_slots:
        if len(original) <= WORKSPACE_CONTEXT_TEXT_FLOOR:
            continue
        lower_bound = WORKSPACE_CONTEXT_TEXT_FLOOR
        upper_bound = len(original)
        retained_chars = WORKSPACE_CONTEXT_TEXT_FLOOR
        while lower_bound <= upper_bound:
            candidate_chars = (lower_bound + upper_bound) // 2
            owner[field_name] = _middle_truncate_text(
                original,
                candidate_chars,
            )
            candidate = json.dumps(
                prompt_payload,
                ensure_ascii=False,
                sort_keys=True,
            )
            if len(candidate) <= maximum_chars:
                retained_chars = candidate_chars
                lower_bound = candidate_chars + 1
            else:
                upper_bound = candidate_chars - 1
        owner[field_name] = _middle_truncate_text(original, retained_chars)

    return json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)


def _middle_truncate_text(value: str, maximum_chars: int) -> str:
    """Retain both semantic ends of one bounded workspace text field."""

    if len(value) <= maximum_chars:
        return value
    marker = "..."
    retained_chars = maximum_chars - len(marker)
    head_chars = (retained_chars + 1) // 2
    tail_chars = retained_chars - head_chars
    return value[:head_chars] + marker + value[-tail_chars:]


def _record_workspace_trace(
    *,
    services: CognitionCoreServicesV2,
    messages: Sequence[BaseMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    started_at: float,
    stage_name: str,
    attempt_index: int,
    validation_error: str,
) -> None:
    """Preserve one protected workspace-collapse model boundary."""

    config = services.workspace_collapse_config
    failure_capsule.append_model_attempt(
        stage_name=stage_name,
        messages=messages,
        response_text=response_text,
        parsed_output=parsed_output,
        parse_status=parse_status,
        status=status,
        config=config,
        attempt_index=attempt_index,
        validation_error=validation_error,
        started_at=started_at,
    )


def _validate_partition(parsed: object, handles: set[str]) -> dict[str, Any]:
    """Validate exact handle partition output from workspace collapse."""

    if not isinstance(parsed, Mapping):
        raise ValueError("workspace partition must be an object")
    required = {
        "primary_bid_handle",
        "supporting_bid_handles",
        "suppressed_bid_handles",
    }
    if set(parsed) != required:
        raise ValueError("workspace partition fields are not exact")
    primary = parsed["primary_bid_handle"]
    if primary not in handles:
        raise ValueError("workspace primary handle is unavailable")
    partitions = []
    for field_name in ("supporting_bid_handles", "suppressed_bid_handles"):
        values = parsed[field_name]
        if not isinstance(values, list) or any(
            value not in handles for value in values
        ):
            raise ValueError("workspace partition handle is unavailable")
        if len(values) != len(set(values)):
            raise ValueError("workspace partition contains duplicate handles")
        partitions.extend(values)
    all_handles = [primary] + partitions
    if len(all_handles) != len(handles) or set(all_handles) != handles:
        raise ValueError("workspace partition is incomplete")
    if len(all_handles) != len(set(all_handles)):
        raise ValueError("workspace partition overlaps")
    return dict(parsed)
