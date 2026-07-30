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
from kazusa_ai_chatbot.llm_tracing import failure_capsule
from kazusa_ai_chatbot.utils import parse_llm_json_output


WORKSPACE_COLLAPSE_ATTEMPT_LIMIT = V2_MODEL_TOTAL_ATTEMPTS
WORKSPACE_COLLAPSE_PROMPT_CAP = 24000
WORKSPACE_COLLAPSE_REPAIR_OUTPUT_CAP = 4000


COLLAPSE_PROMPT = '''把完整的目标候选划分为本次 prompt 内的主目标、支持目标和抑制目标。
只返回包含 primary_bid_handle、supporting_bid_handles 和
suppressed_bid_handles 的 JSON。保持候选内容原样，不复制内容，也不增添细节。
每个提供的 bid handle 必须在三个分区中恰好出现一次。
'''


async def collapse_bids(
    bids: Sequence[ActionBidV2],
    services: CognitionCoreServicesV2,
) -> CollapsedIntentionV2:
    """Collapse complete bids while preserving whole-bid ownership in code."""

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
        "bids": {
            handle: {
                "intention": bid["intention"],
                "desired_outcome": bid["desired_outcome"],
                "reason": bid["reason"],
                "confidence": bid["confidence"],
            }
            for handle, bid in handles.items()
        }
    }
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    system_message = SystemMessage(content=COLLAPSE_PROMPT)
    request_messages = [
        system_message,
        HumanMessage(content=prompt_text),
    ]
    partition: dict[str, Any] | None = None
    available_attempts = WORKSPACE_COLLAPSE_ATTEMPT_LIMIT
    if len(prompt_text) > WORKSPACE_COLLAPSE_PROMPT_CAP:
        available_attempts = 0
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
            repair_text = json.dumps(
                repair_payload,
                ensure_ascii=False,
                sort_keys=True,
            )
            if len(repair_text) > WORKSPACE_COLLAPSE_PROMPT_CAP:
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
